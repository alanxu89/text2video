import os
import time
from datetime import datetime

import torch
import torch.distributed as dist
from torchvision.io import write_video
from torchvision.utils import save_image

from config import Config
from train import to_torch_dtype
from vae import VideoAutoEncoderKL
from t5 import T5Encoder
from clip import ClipEncoder
from models import STDiT
from videoldm import VideoLDM
from diffusion import IDDPM


def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def load_checkpoint(model, ckpt_path, model_name="model", save_as_pt=True):
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        assert os.path.isfile(
            ckpt_path), f"Could not find DiT checkpoint at {ckpt_path}"
        checkpoint = torch.load(ckpt_path,
                                map_location=lambda storage, loc: storage)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint[model_name], strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")


def save_sample(x, fps=8, save_path=None, normalize=True, value_range=(-1, 1)):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    assert x.ndim == 4

    if x.shape[1] == 1:  # T = 1: save as image
        save_path += ".png"
        x = x.squeeze(1)
        save_image([x],
                   save_path,
                   normalize=normalize,
                   value_range=value_range)
    else:
        save_path += ".mp4"
        if normalize:
            low, high = value_range
            x = torch.clamp(x, min=low, max=high)
            x = (x - low) / max(high - low, 1e-5)

        x = x.mul(255).add_(0.5).clamp_(0,
                                        255).permute(1, 2, 3,
                                                     0).to("cpu", torch.uint8)
        write_video(save_path, x, fps=fps, video_codec="h264")
    print(f"Saved to {save_path}")
    return save_path


def main():
    # create configs
    cfg = Config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    dtype = to_torch_dtype(cfg.dtype)
    torch.set_default_dtype(dtype)
    # torch.cuda.set_device(device)

    # model
    vae = VideoAutoEncoderKL(cfg.vae_pretrained,
                             cfg.subfolder,
                             micro_batch_size=8,
                             dtype=dtype)
    input_size = (cfg.num_frames, *cfg.image_size)
    latent_size = vae.get_latent_size(input_size)

    if "t5" in cfg.textenc_pretrained:
        text_encoder_cls = T5Encoder
    else:
        text_encoder_cls = ClipEncoder
    text_encoder = text_encoder_cls(from_pretrained=cfg.textenc_pretrained,
                                    model_max_length=cfg.model_max_length,
                                    dtype=dtype)

    if cfg.use_videoldm:
        model = VideoLDM.from_pretrained('runwayml/stable-diffusion-v1-5',
                                         subfolder='unet',
                                         low_cpu_mem_usage=False,
                                         torch_dtype=dtype)
    else:
        model = STDiT(
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            depth=cfg.depth,
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            patch_size=cfg.patch_size,
            joint_st_attn=cfg.joint_st_attn,
            use_3dconv=cfg.use_3dconv,
            enable_mem_eff_attn=cfg.enable_mem_eff_attn,
            enable_flashattn=cfg.enable_flashattn,
            enable_grad_checkpoint=False,
            debug=cfg.debug,
        )

    load_checkpoint(model, cfg.ckpt_path, model_name="model")

    # or classifier-free guidance
    if cfg.use_videoldm:
        model.y_embedding = text_encoder.null(1).squeeze()
        # for name, param in model.named_parameters():
        #     if 'alpha' in name:
        #         param.data = torch.ones(1)
    else:
        text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 4.3. move to device
    vae = vae.to(device).eval()
    model = model.to(device).eval()

    scheduler = IDDPM(num_sampling_steps=cfg.inference_sampling_steps,
                      learn_sigma=not cfg.use_videoldm,
                      cfg_scale=cfg.cfg_scale)

    model_args = dict()

    # ======================================================
    # 4. inference
    # ======================================================

    prompts = load_prompts("assets/texts/mixkit_test_prompts4.txt")
    sample_idx = 0
    save_dir = os.path.join(cfg.save_dir,
                            datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(save_dir, exist_ok=True)
    for i in range(0, len(prompts), cfg.batch_size):
        batch_prompts = prompts[i:i + cfg.batch_size]

        if cfg.use_videoldm:
            new_batch_prompts = []
            for k in range(len(batch_prompts)):
                new_batch_prompts.extend([batch_prompts[k]] * cfg.num_frames)
            batch_prompts = new_batch_prompts
            z_size = (vae.out_channels, *latent_size[1:])
        else:
            z_size = (vae.out_channels, *latent_size)

        # save vram with no_grad
        with torch.no_grad():
            samples = scheduler.sample(
                model,
                text_encoder,
                z_size=z_size,
                prompts=batch_prompts,
                device=device,
                additional_args=model_args,
                use_videoldm=cfg.use_videoldm,
            )

            print("decoding...")
            torch.save(samples, os.path.join(save_dir, "latents.pt"))

            if cfg.use_videoldm:
                samples = samples.reshape(-1, cfg.num_frames,
                                          *z_size).permute(0, 2, 1, 3, 4)
            for j in range(2):
                save_sample(samples[j, :3, :1],
                            save_path=os.path.join(save_dir, "latents_" +
                                                   str(j) + "_t0"))
            samples = vae.decode(samples)
            samples = samples * 0.5 + 0.5
            print("done\n")

        for idx, sample in enumerate(samples):
            prompt_idx = idx
            if cfg.use_videoldm:
                prompt_idx = prompt_idx * cfg.num_frames
            print(f"Prompt:\n {batch_prompts[prompt_idx]}")
            save_path = os.path.join(save_dir, f"sample_{sample_idx}")
            save_sample(sample, fps=6, save_path=save_path, normalize=False)
            save_sample(sample[:, :1],
                        save_path=save_path + "_t0",
                        normalize=False)

            sample_idx += 1


if __name__ == "__main__":
    main()
