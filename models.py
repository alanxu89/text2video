import torch
import torch.nn as nn

import numpy as np

from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath

from einops import rearrange

from blocks import (
    approx_gelu,
    modulate,
    get_layernorm,
    Attention,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    TimestepEmbedder,
    CaptionEmbedder,
    FinalLayer,
    T2IFinalLayer,
    get_2d_sincos_pos_embed,
    get_1d_sincos_pos_embed,
)


class STDiTBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        d_s=None,
        d_t=None,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn

        self.norm1 = get_layernorm(hidden_size,
                                   eps=1e-6,
                                   affine=False,
                                   use_kernel=False)
        self.s_attn = Attention(hidden_size,
                                num_heads=num_heads,
                                qkv_bias=True,
                                enable_flashattn=enable_flashattn)

        self.norm2 = get_layernorm(hidden_size,
                                   eps=1e-6,
                                   affine=False,
                                   use_kernel=False)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)

        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=approx_gelu,
                       drop=0)
        self.drop_path = DropPath(
            drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()
        # why not nn.Linear?
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5)

        # spatial-temporal attention
        self.d_s = d_s
        self.d_t = d_t

        self.t_attn = Attention(hidden_size,
                                num_heads=num_heads,
                                qkv_bias=True,
                                enable_flashattn=enable_flashattn)

    def forward(self, x, c, t, mask=None, tpe=None):
        B, N, C = x.shape

        # modulation for t
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

        x_m = modulate(self.norm1(x), shift_msa, scale_msa)

        # spatial branch
        x_s = rearrange(x_m, "b (t s) c -> (b t) s c", t=self.d_t, s=self.d_s)
        x_s = self.s_attn(x_s)
        x_s = rearrange(x_s, "(b t) s c -> b (t s) c", t=self.d_t, s=self.d_s)
        x = x + self.drop_path(gate_msa * x_s)

        # temporal branch
        x_t = rearrange(x, "b (t s) c -> (b s) t c", t=self.d_t, s=self.d_s)
        x_t = self.t_attn(x_t)
        x_t = rearrange(x_t, "(b s) t c -> b (t s) c", t=self.d_t, s=self.d_s)
        x = x + self.drop_path(gate_msa * x_t)

        # cross attention
        x = x + self.cross_attn(x, c, mask)

        # mlp
        x = x + self.drop_path(
            gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


class STDiT(nn.Module):

    def __init__(
        self,
        input_size=(16, 32, 32),
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,
        model_max_length=120,
        space_scale=1.0,
        time_scale=1.0,
        freeze=None,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.patch_size_nd = np.prod(patch_size)
        st_patches = [input_size[i] // patch_size[i] for i in range(3)]
        self.num_temporal = st_patches[0]
        self.num_spatial_h = st_patches[1]
        self.num_spatial_w = st_patches[2]
        self.num_spatial = self.num_spatial_h * self.num_spatial_w
        self.num_patches = self.num_temporal * self.num_spatial
        self.num_heads = num_heads

        self.space_scale = space_scale
        self.time_scale = time_scale

        self.x_embedder = PatchEmbed3D(patch_size=patch_size,
                                       in_chans=in_channels,
                                       embed_dim=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size=hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        self.pos_embed = nn.Parameter(self.get_spatial_pos_embed(),
                                      requires_grad=False)
        self.pos_embed_temporal = nn.Parameter(self.get_temporal_pos_embed(),
                                               requires_grad=False)

        drop_path = [val.item() for val in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            STDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                d_s=self.num_spatial,
                d_t=self.num_temporal,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                enable_flashattn=enable_flashattn,
                enable_layernorm_kernel=enable_layernorm_kernel,
                enable_sequence_parallelism=enable_sequence_parallelism,
            ) for i in range(depth)
        ])

        self.final_layer = T2IFinalLayer(hidden_size,
                                         self.patch_size_nd,
                                         out_channels=self.out_channels)

    def forward(self, x, t, y, mask=None):
        """
        Args:
            x: input latents                [B, C, T, H, W]
            t: diffusion timestep           [B, ]   
            y: tokens                       [B, 1, N_tokens, D] 
            mask: token mask                [B, N_tokens]
        Returns:
            x: output latents for curren diffusion step [B, C, T, H, W]
        """

        # x embedding
        x = self.x_embedder(x)  # [B, Nt*Nh*Nw, C]

        # prepare for spatial
        x = rearrange(x,
                      "B (T S) C -> B T S C",
                      T=self.num_temporal,
                      S=self.num_spatial)
        x = x + self.pos_embed
        x = rearrange(x, "B T S C -> B (T S) C")

        t = self.t_embedder(t)  # [B, C]
        t0 = self.t_block(t)  #[B, 6*C]

        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(
                1, -1, y.shape[-1])
            y_lens = mask.sum(dim=1).tolist()  # [n1, n2, n3, ,,,, n_bs]
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, y.shape[-1])  # [BxN_token, C]

        for block in self.blocks:
            x = block(x, y, t0, y_lens, self.pos_embed_temporal)

        x = self.final_layer(
            x, t)  # [N, num_patches, patch_size_nd * out_channels]
        x = self.unpatchify(x)  # [N, C, T, H, W]

        return x

    def unpatchify(self, x):
        """unpatchify latents to original input demension e.g. [B, T, 32, 32, 4]

        Args:
            x: [B, N, C_in]
        Returns:
            x_out: [B, C_out, T, H, W]
        """

        x = rearrange(
            x,
            "B (Nt Nh Nw) (Pt Ph Pw C) -> B C (Nt Pt) (Nh Ph) (Nw Pw)",
            Nt=self.num_temporal,
            Nh=self.num_spatial_h,
            Nw=self.num_spatial_w,
            Pt=self.patch_size[0],
            Ph=self.patch_size[1],
            Pw=self.patch_size[1],
            C=self.out_channels,
        )

        return x

    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = (self.num_spatial_h, self.num_spatial_w)
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, grid_size,
                                            self.space_scale)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(
            0)  # add a dummy batch dimension

        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(self.hidden_size,
                                            self.input_size[0],
                                            scale=self.time_scale)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
        return pos_embed
