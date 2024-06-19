# Text to video generation with different model architectures.
## spatial-temporal diffusion transformers (DiT)
This is the main model used by this repo, described in [models.py](models.py) [blocks.py](blocks.py) files. The architecture is borrowed from https://github.com/hpcaitech/Open-Sora and we made some modifications. 

## temporal layer adapter to pretrained Stable Diffusion models
The second model architecture we implemented is to add temporal layers to existing SD unets, described in [videoldm.py](videoldm.py) and [videoldm_blocks.py](videoldm_blocks.py). This architecture is first proposed in Align your Latents paper (https://arxiv.org/abs/2304.08818). A reference implementation is given by https://github.com/srpkdyy/VideoLDM and we adapt it to our unified training framework.
