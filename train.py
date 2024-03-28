import os
from glob import glob
from typing import Tuple
from copy import deepcopy
from collections import OrderedDict
import logging

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

from datasets import DatasetFromCSV, get_transforms_video
from vae import VideoAutoEncoderKL
from t5 import T5Encoder
from models import STDiT
from diffusion import IDDPM


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


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def main(cfg):
    # create configs
    rank = dist.get_rank()

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(cfg.results_dir, exist_ok=True
                    )  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{cfg.results_dir}/*"))
        model_string_name = cfg.model.replace(
            "/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{cfg.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # prepare dataset
    dataset = DatasetFromCSV(cfg.csv_path,
                             num_frames=16,
                             frame_interval=4,
                             transform=get_transforms_video)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
    )

    logger.info(f"Dataset contains {len(dataset):,} videos ({cfg.data_path})")
    total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
    logger.info(f"Total batch size: {total_batch_size}")

    # model
    vae = VideoAutoEncoderKL("stabilityai/sd-vae-ft-ema")
    latent_size = vae.get_latent_size(cfg.input_size)

    text_encoder = T5Encoder(from_pretrained="DeepFloyd/t5-v1_1-xxl")

    model = STDiT()

    model_numel, model_numel_trainable = get_model_numel(model)
    print(
        f"Trainable model params: {model_numel_trainable}, Total model params: {model_numel}"
    )

    # 4.2. create ema
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    # ema_shape_dict = record_model_param_shape(ema)

    # 4.3. move to device
    vae = vae.to(device, dtype)
    model = model.to(device, dtype)

    model.train()
    update_ema(ema, model, decay=0, sharded=False)
    ema.eval()

    scheduler = IDDPM(timestep_respacing="")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    torch.set_default_dtype(torch.float)
    num_steps_per_epoch = len(dataloader)
    logger.info("Boost model for distributed training")

    # =======================================================
    # 6. training loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = 0
    running_loss = 0.0

    # 6.1. resume training
    # if cfg.load is not None:
    #     logger.info("Loading checkpoint")
    #     start_epoch, start_step, sampler_start_idx = load(
    #         booster, model, ema, opt, lr_scheduler, cfg.load)
    #     logger.info(
    #         f"Loaded checkpoint {cfg.load} at epoch {start_epoch} step {start_step}"
    #     )
    # logger.info(
    #     f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch"
    # )

    dataloader.sampler.set_start_index(sampler_start_idx)
    # model_sharding(ema)

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
                batch = next(dataloader_iter)
                x = batch["video"].to(device, dtype)  # [B, C, T, H, W]
                y = batch["text"]

                # video and text encoding
                with torch.no_grad():
                    x = vae.encode(x)
                    model_args = text_encoder.encode(y)

                # diffusion
                t = torch.randint(0,
                                  scheduler.num_timesteps,
                                  size=(x.shape[0]),
                                  device=device)
                loss_dict = scheduler.training_losses(model, x, t, model_args)
                loss = loss_dict["loss"].mean()

                # step
                opt.zero_grad()
                loss.backward()
                opt.step()

                # Update EMA
                update_ema(ema, model.module)

                # Log loss values:
                all_reduce_mean(loss)
                running_loss += loss.item()
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1

                # Save checkpoint
                if cfg.ckpt_every > 0 and (global_step +
                                           1) % cfg.ckpt_every == 0:
                    if rank == 0:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "cfg": cfg
                        }
                        checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(
                            f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {checkpoint_path}"
                        )

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(0)
        start_step = 0


if __name__ == "__main__":
    main()
