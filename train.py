import os
import time
from glob import glob
from typing import Tuple, Optional
from copy import deepcopy
from collections import OrderedDict
import logging
import argparse
import json

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from einops import rearrange

from datasets import DatasetFromCSV, PreprocessedDatasetFromCSV, get_transforms_video
from vae import VideoAutoEncoderKL
from t5 import T5Encoder
from clip import ClipEncoder
from models import STDiT
from videoldm import VideoLDM
from diffusion import IDDPM
from config import Config


def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module,
               model: torch.nn.Module,
               optimizer=None,
               decay: float = 0.9999,
               sharded: bool = False) -> None:
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if param.requires_grad == False:
            continue
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        # else:
        #     if param.data.dtype != torch.float32:
        #         param_id = id(param)
        #         master_param = optimizer._param_store.working_to_master_param[
        #             param_id]
        #         param_data = master_param.data
        #     else:
        #         param_data = param.data
        #     ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(level=logging.INFO,
                            format='[\033[34m%(asctime)s\033[0m] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[
                                logging.StreamHandler(),
                                logging.FileHandler(f"{logging_dir}/log.txt")
                            ])
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def create_tensorboard_writer(exp_dir):
    tensorboard_dir = f"{exp_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    return writer


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def parse_args(training=False):
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("config", help="model config file path")

    parser.add_argument("--seed", default=42, type=int, help="generation seed")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        help="path to model ckpt; will overwrite cfg.ckpt_path if specified")
    parser.add_argument("--batch-size",
                        default=None,
                        type=int,
                        help="batch size")

    # ======================================================
    # Inference
    # ======================================================

    if not training:
        # prompt
        parser.add_argument("--prompt-path",
                            default=None,
                            type=str,
                            help="path to prompt txt file")
        parser.add_argument("--save-dir",
                            default=None,
                            type=str,
                            help="path to save generated samples")

        # hyperparameters
        parser.add_argument("--num-sampling-steps",
                            default=None,
                            type=int,
                            help="sampling steps")
        parser.add_argument("--cfg-scale",
                            default=None,
                            type=float,
                            help="balance between cond & uncond")
    else:
        parser.add_argument("--wandb",
                            default=None,
                            type=bool,
                            help="enable wandb")
        parser.add_argument("--load",
                            default=None,
                            type=str,
                            help="path to continue training")
        parser.add_argument("--data-path",
                            default=None,
                            type=str,
                            help="path to data csv")

    return parser.parse_args()


def to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype_mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_mapping:
            raise ValueError
        dtype = dtype_mapping[dtype]
        return dtype
    else:
        raise ValueError


class StatefulDistributedSampler(DistributedSampler):

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self):
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index:]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index


def main():
    # create configs
    cfg = Config()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.manual_seed(cfg.seed)
    torch.cuda.set_device(device)
    dtype = to_torch_dtype(cfg.dtype)
    torch.set_default_dtype(dtype)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(cfg.outputs, exist_ok=True
                    )  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{cfg.outputs}/*"))
        experiment_dir = f"{cfg.outputs}/{experiment_index:03d}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        # write config to json
        with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
            f.write(json.dumps(cfg.__dict__, indent=2, sort_keys=False))
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        writer = create_tensorboard_writer(experiment_dir)
    else:
        logger = create_logger(None)

    # prepare dataset
    if cfg.use_preprocessed_data:
        dataset = PreprocessedDatasetFromCSV(
            cfg.data_path,
            root=cfg.root,
            preprocessed_dir=cfg.preprocessed_dir)
    else:
        dataset = DatasetFromCSV(cfg.data_path,
                                 num_frames=cfg.num_frames,
                                 frame_interval=cfg.frame_interval,
                                 transform=get_transforms_video(
                                     cfg.image_size),
                                 root=cfg.root)
    sampler = StatefulDistributedSampler(dataset,
                                         num_replicas=dist.get_world_size(),
                                         rank=dist.get_rank(),
                                         shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=sampler,
    )

    logger.info(f"Dataset contains {len(dataset):,} videos ({cfg.data_path})")
    total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
    logger.info(f"Total batch size: {total_batch_size}")

    # default values
    vae_out_channels = 4
    text_encoder_output_dim = cfg.text_encoder_output_dim
    input_size = (cfg.num_frames, *cfg.image_size)
    vae_down_factor = [1, 8, 8]
    latent_size = [input_size[i] // vae_down_factor[i] for i in range(3)]

    # encoders
    vae = None
    text_encoder = None
    # text encoder
    if "t5" in cfg.textenc_pretrained:
        text_encoder_cls = T5Encoder
    else:
        text_encoder_cls = ClipEncoder

    if not cfg.use_preprocessed_data:
        # video VAE
        vae = VideoAutoEncoderKL(cfg.vae_pretrained,
                                 cfg.subfolder,
                                 dtype=torch.float16).to(device)
        vae.eval()
        latent_size = vae.get_latent_size(input_size)
        vae_out_channels = vae.out_channels

        text_encoder = text_encoder_cls(from_pretrained=cfg.textenc_pretrained,
                                        model_max_length=cfg.model_max_length,
                                        dtype=dtype)
        text_encoder_output_dim = text_encoder.output_dim

    if cfg.use_videoldm:
        if cfg.image_finetune:
            from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
            model = UNet2DConditionModel.from_pretrained(
                'runwayml/stable-diffusion-v1-5',
                subfolder='unet',
                low_cpu_mem_usage=False,
                torch_dtype=dtype)
        else:
            model = VideoLDM.from_pretrained('runwayml/stable-diffusion-v1-5',
                                             subfolder='unet',
                                             low_cpu_mem_usage=False,
                                             torch_dtype=dtype)
            for name, param in model.named_parameters():
                if not ("conv_3ds" in name or "temp_attns" in name):
                    param.requires_grad = False

            # empty token ([""]) embedding for uncond generation in classifier-free guidance
            if text_encoder is None:
                text_encoder = text_encoder_cls(
                    from_pretrained=cfg.textenc_pretrained,
                    model_max_length=cfg.model_max_length,
                    dtype=dtype)
            model.y_embedding = text_encoder.null(1).squeeze()
    else:
        # STDiT model
        model = STDiT(
            input_size=latent_size,
            in_channels=vae_out_channels,
            caption_channels=text_encoder_output_dim,
            model_max_length=cfg.model_max_length,
            depth=cfg.depth,
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            patch_size=cfg.patch_size,
            enable_temporal_attn=cfg.enable_temporal_attn,
            joint_st_attn=cfg.joint_st_attn,
            use_3dconv=cfg.use_3dconv,
            enable_mem_eff_attn=cfg.enable_mem_eff_attn,
            enable_flashattn=cfg.enable_flashattn,
            enable_grad_checkpoint=cfg.enable_grad_ckpt,
            debug=cfg.debug,
            class_dropout_prob=cfg.token_drop_prob,
        )

    # freeup memory
    if cfg.use_preprocessed_data:
        if text_encoder is not None:
            del text_encoder
        if vae is not None:
            del vae

    model_numel, model_numel_trainable = get_model_numel(model)
    print(
        f"Trainable model params: {model_numel_trainable}, Total model params: {model_numel}"
    )

    # 4.2. create ema
    if cfg.use_ema:
        ema = deepcopy(model).to(device)
        # requires_grad(ema, False)

    # 4.3. move to device
    model = model.to(device, dtype)
    model.train()

    scheduler = IDDPM(timestep_respacing="",
                      noise_schedule=cfg.noise_schedule,
                      learn_sigma=not cfg.use_videoldm)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0)

    num_steps_per_epoch = len(dataloader)
    logger.info("Boost model for distributed training")

    # =======================================================
    # 6. training loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = 0
    running_loss = 0.0

    # 6.1. resume training
    if cfg.load is not None:
        logger.info(f"Loading checkpoint {cfg.load}")
        checkpoint = torch.load(cfg.load)
        model.load_state_dict(checkpoint['model'])
        logger.info(f"model weights loaded")
        if not cfg.load_weights_only:
            opt.load_state_dict(checkpoint['opt'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint")
        del checkpoint

    if cfg.use_ema:
        update_ema(ema, model, decay=0, sharded=False)
        ema.eval()

    # dataloader.sampler.set_start_index(sampler_start_idx)
    # model_sharding(ema)

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()

    # torch.autograd.set_detect_anomaly(True)
    # 6.2. training loop
    for epoch in range(start_epoch, cfg.epochs):
        dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")

        with tqdm(
                range(start_step, num_steps_per_epoch),
                desc=f"Epoch {epoch}",
                # disable=not coordinator.is_master(),
                total=num_steps_per_epoch,
                initial=start_step,
        ) as pbar:

            for step in pbar:
                t0 = time.time()
                # step
                global_step = epoch * num_steps_per_epoch + step
                batch = next(dataloader_iter)

                if cfg.use_preprocessed_data:
                    x = batch['x'].to(device, dtype)
                    y = batch['y'].to(device, dtype)
                    mask = batch['mask'].to(device)
                    model_args = dict(y=y, mask=mask)
                else:
                    x = batch["video"].to(device, dtype)  # [B, C, T, H, W]
                    y = batch["text"]
                    # video and text encoding
                    with torch.no_grad():
                        x = vae.encode(x)
                        model_args = text_encoder.encode(y)

                # original batch size without the t dimension
                bs_original = x.shape[0]

                if cfg.use_videoldm:
                    # [B, C, T, H, W] => [B*T, C, H, W]
                    x = rearrange(x, 'b c t h w -> (b t) c h w')
                    y = y.repeat(1, cfg.num_frames, 1, 1)
                    y = rearrange(y, 'b t n c -> (b t) n c')
                    model_args = dict(encoder_hidden_states=y)
                    if cfg.use_attention_mask:
                        mask = mask.repeat(cfg.num_frames, 1)
                        mask = rearrange(mask,
                                         '(t b) c -> (b t) c',
                                         t=cfg.num_frames)
                        model_args["encoder_attention_mask"] = mask

                t1 = time.time()
                # diffusion
                t = torch.randint(low=0,
                                  high=scheduler.num_timesteps,
                                  size=(bs_original, ),
                                  device=device)
                if cfg.use_videoldm:
                    # use the same t step for different frames in the same video
                    t = t[:, None].repeat(1, cfg.num_frames).flatten()

                # Enables autocasting for the forward pass (model + loss)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss_dict = scheduler.training_losses(
                        model, x, t, model_args)
                    loss = loss_dict["loss"].mean()
                    loss = loss / cfg.accum_iter
                    loss_terms = {}
                    for term in loss_dict:
                        if term != "loss":
                            loss_terms[term] = loss_dict[term].mean(
                            ) / cfg.accum_iter

                # with torch.autograd.detect_anomaly():
                scaler.scale(loss).backward()
                # loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),
                                         cfg.grad_clip,
                                         norm_type=2)

                if global_step % cfg.accum_iter == 0:

                    # scaler.step() first unscales gradients of the optimizer's params.
                    # If gradients don't contain infs/NaNs, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    scaler.step(opt)
                    # Updates the scale for next iteration.
                    scaler.update()

                    # opt.step()
                    opt.zero_grad()

                # loss.backward()
                # opt.step()

                # print(t1 - t0, time.time() - t1)

                # Update EMA
                if cfg.use_ema:
                    update_ema(ema, model)

                # Log loss values:
                all_reduce_mean(loss)
                for term in loss_terms:
                    all_reduce_mean(loss_terms[term])
                running_loss += loss.item()
                log_step += 1

                # Log to tensorboard
                if (global_step + 1) % cfg.log_every == 0:
                    avg_loss = running_loss / log_step
                    pbar.set_postfix({
                        "loss": avg_loss,
                        "step": step,
                        "global_step": global_step
                    })
                    running_loss = 0
                    log_step = 0
                    if rank == 0:
                        writer.add_scalar("loss", loss.item(), global_step)
                        for term in loss_terms:
                            writer.add_scalar(term, loss_terms[term].item(),
                                              global_step)

                # Save checkpoint
                if cfg.ckpt_every > 0 and (global_step +
                                           1) % cfg.ckpt_every == 0:
                    if rank == 0:
                        model_states = ema.state_dict(
                        ) if cfg.use_ema else model.state_dict()
                        checkpoint = {
                            "model": model_states,
                            "opt": opt.state_dict(),
                            "cfg": cfg,
                            "epoch": epoch,
                        }
                        checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(
                            f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {checkpoint_path}"
                        )

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(0)
        start_step = 0

    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
