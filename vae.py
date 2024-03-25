import torch
import torch.nn as nn

from diffusers.models import AutoencoderKL
from einops import rearrange


class VideoAutoEncoderKL(nn.Module):

    def __init__(self,
                 pretrained_model,
                 micro_batch_size=None,
                 patch_size=(1, 8, 8),
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.vae = AutoencoderKL.from_pretrained(pretrained_model)
        self.micro_batch_size = micro_batch_size
        self.patch_size = patch_size

    def encode(self, x):
        b = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_mb = x[i:i + bs]
                x_mb = self.vae.encode(x_mb).latent_dist.sample().mul_(0.18215)
                x_out.append(x_mb)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=b)

        return x

    def decode(self, x):
        b = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        if self.micro_batch_size is None:
            x = self.vae.decode(x / 0.18215).sample
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_mb = x[i:i + bs]
                x_mb = self.vae.decode(x_mb / 0.18215).sample
                x_out.append(x_mb)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=b)

        return x

    def get_latent_size(self, input_size):
        for i in range(3):
            assert input_size[i] % self.patch_size[
                i] == 0, "input size must be divisible by patch size"
        latent_size = [input_size[i] // self.patch_size[i] for i in range(3)]
        return latent_size


class VideoAutoencoderKLTemporalDecoder(nn.Module):

    def __init__(self, pretrained_model, patch_size=(1, 8, 8), *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.vae = AutoencoderKL.from_pretrained(pretrained_model)
        self.patch_size = patch_size

    def decode(self, x):
        B, C, T = x.shape[:3]
        x = rearrange(x, "B C T H W -> (B T) C H W")

        x = self.vae.decode(x / 0.18215, num_frames=T).sample
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)

        return x
