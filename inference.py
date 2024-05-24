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
from models import STDiT
from diffusion import IDDPM


def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def load_checkpoint(model, ckpt_path, model_name="ema", save_as_pt=True):
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
                             cfg.vae_scaling_factor,
                             micro_batch_size=8,
                             dtype=dtype)
    input_size = (cfg.num_frames, *cfg.image_size)
    latent_size = vae.get_latent_size(input_size)

    text_encoder = T5Encoder(from_pretrained=cfg.textenc_pretrained,
                             model_max_length=cfg.model_max_length,
                             dtype=dtype)

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
    load_checkpoint(model, cfg.ckpt_path, model_name="ema")
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 4.3. move to device
    vae = vae.to(device).eval()
    model = model.to(device).eval()

    scheduler = IDDPM(num_sampling_steps=250, cfg_scale=4.0)

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

        # save vram with no_grad
        with torch.no_grad():
            samples = scheduler.sample(
                model,
                text_encoder,
                z_size=(vae.out_channels, *latent_size),
                prompts=batch_prompts,
                device=device,
                additional_args=model_args,
            )

            print("decoding...")
            torch.save(samples, os.path.join(save_dir, "latents.pt"))

            for j in range(len(batch_prompts)):
                save_sample(samples[j, :3, :1],
                            save_path=os.path.join(save_dir, "latents_" +
                                                   str(j) + "_t0"))
            samples = vae.decode(samples)
            samples = samples * 0.5 + 0.5
            print("done\n")

        for idx, sample in enumerate(samples):
            print(f"Prompt:\n {batch_prompts[idx]}")
            save_path = os.path.join(save_dir, f"sample_{sample_idx}")
            save_sample(sample, fps=6, save_path=save_path, normalize=False)
            save_sample(sample[:, :1],
                        save_path=save_path + "_t0",
                        normalize=False)

            sample_idx += 1


if __name__ == "__main__":
    main()
