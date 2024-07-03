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
    Conv3DLayer,
    PatchEmbed3D,
    TimestepEmbedder,
    CaptionEmbedder,
    FinalLayer,
    T2IFinalLayer,
    get_2d_sincos_pos_embed,
    get_1d_sincos_pos_embed,
)

from utils import (
    debugprint,
    get_st_attn_mask,
    get_attn_bias_from_mask,
    auto_grad_checkpoint,
)


class STDiTBlock(nn.Module):
    """ similar to DiT Block with adaLN-Zero

    differences:
        1. spatial and temporal self-attention
        2. cross attention between x and y (text prompts)
        3. t goes to adaLN-Zero and y goes to cross attention
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        d_s=None,
        d_t=None,
        d_s_h=16,
        d_s_w=16,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_temporal_attn=True,
        temporal_layer_type="conv3d",
        enable_mem_eff_attn=False,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        debug=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn
        self.enable_temporal_attn = enable_temporal_attn
        self.temporal_layer_type = temporal_layer_type
        self.d_s = d_s
        self.d_t = d_t
        self.d_s_h = d_s_h
        self.d_s_w = d_s_w
        self.debugprint = debugprint(debug)

        # layer norm for spatial-attn, temp-attn cross-attn and mlp
        self.norm_s = get_layernorm(hidden_size,
                                    eps=1e-6,
                                    affine=False,
                                    use_kernel=False)
        self.norm_t = get_layernorm(hidden_size,
                                    eps=1e-6,
                                    affine=False,
                                    use_kernel=False)
        self.norm_ca = get_layernorm(hidden_size,
                                     eps=1e-6,
                                     affine=False,
                                     use_kernel=False)
        self.norm_mlp = get_layernorm(hidden_size,
                                      eps=1e-6,
                                      affine=False,
                                      use_kernel=False)

        # why not nn.Linear with no bias?
        # scale, shift and gate parameters for self-attention and mlp
        self.scale_shift_table = nn.Parameter(
            torch.randn(9, hidden_size) / hidden_size**0.5)
        # # DiT
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))

        # spatial attention
        self.s_attn = Attention(hidden_size,
                                num_heads=num_heads,
                                qkv_bias=True,
                                enable_flashattn=enable_flashattn,
                                enable_mem_eff_attn=enable_mem_eff_attn)

        # temporal layer
        if temporal_layer_type == "conv3d":
            # similar to the Conv3D layers in Align your Latents paper
            self.conv3d = Conv3DLayer(dim=hidden_size,
                                      inner_dim=256,
                                      enable_proj_out=True)
        elif (temporal_layer_type
              == "temporal_only_attn") or (temporal_layer_type
                                           == "spatial_temporal_attn"):
            self.t_attn = Attention(hidden_size,
                                    num_heads=num_heads,
                                    qkv_bias=True,
                                    enable_flashattn=enable_flashattn,
                                    enable_mem_eff_attn=enable_mem_eff_attn)
        else:
            self.t_attn = None

        # cross attention between x and y(text prompts)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)

        # mlp for feedforward network
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=approx_gelu,
                       drop=0)

        # a shared drop out for all?
        self.drop_path = DropPath(
            drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, y, t, mask=None, tpe=None, st_attn_bias=None):
        self.debugprint("inside ", self.__class__)

        B, N, C = x.shape

        # modulation for t
        (shift_msa_s, scale_msa_s, gate_msa_s, shift_msa_t, scale_msa_t,
         gate_msa_t, shift_mlp, scale_mlp,
         gate_mlp) = (self.scale_shift_table[None] +
                      t.reshape(B, 9, -1)).chunk(9, dim=1)

        x_m_s = modulate(self.norm_s(x), shift_msa_s, scale_msa_s)

        # spatial branch
        self.debugprint("spatial branch", x_m_s.shape)
        x_s = rearrange(x_m_s,
                        "b (t s) c -> (b t) s c",
                        t=self.d_t,
                        s=self.d_s)
        self.debugprint(x_s.shape)
        x_s = self.s_attn(x_s)
        self.debugprint(x_s.shape)
        x_s = rearrange(x_s, "(b t) s c -> b (t s) c", t=self.d_t, s=self.d_s)
        self.debugprint(x_s.shape)
        x = x + self.drop_path(gate_msa_s * x_s)
        self.debugprint(x.shape)

        # temporal branch
        if self.enable_temporal_attn:
            x_m_t = modulate(self.norm_t(x), shift_msa_t, scale_msa_t)

            self.debugprint("temporal branch", x_m_t.shape)
            x_t = rearrange(x_m_t,
                            "b (t s) c -> (b s) t c",
                            t=self.d_t,
                            s=self.d_s)
            self.debugprint(x_t.shape)

            if tpe is not None:
                self.debugprint("tpe shape:", tpe.shape)
                x_t = x_t + tpe

            if self.temporal_layer_type == "conv3d":
                self.debugprint("use conv3D")
                # rearrange and then apply conv and then rearrange back
                x_t = rearrange(x_t,
                                "(b s_h s_w) t c -> b c t s_h s_w",
                                t=self.d_t,
                                s_h=self.d_s_h,
                                s_w=self.d_s_w)
                self.debugprint(x_t.shape)
                x_t = self.conv3d(x_t)
                self.debugprint(x_t.shape)
                x_t = rearrange(x_t, "b c t s_h s_w -> b (t s_h s_w) c")
                self.debugprint(x_t.shape)
            elif self.temporal_layer_type == "temporal_only_attn":
                # temporal-only attention
                # apply attention and then rearange
                x_t = self.t_attn(x_t)
                self.debugprint(x_t.shape)
                x_t = rearrange(x_t,
                                "(b s) t c -> b (t s) c",
                                t=self.d_t,
                                s=self.d_s)
            elif self.temporal_layer_type == "spatial_temporal_attn":
                # joint spatial-temporal with a finite window size
                self.debugprint("use self-attn")
                # rearange and then apply attention
                x_t = rearrange(x_t,
                                "(b s) t c -> b (t s) c",
                                t=self.d_t,
                                s=self.d_s)
                self.debugprint(x_t.shape)
                x_t = self.t_attn(x_t, st_attn_bias)
            else:
                pass

            self.debugprint(x_t.shape)
            x = x + self.drop_path(gate_msa_t * x_t)
            self.debugprint(x.shape)

        # cross attention
        self.debugprint("cross attn")
        self.debugprint(x.shape, y.shape)
        x = x + self.cross_attn(self.norm_ca(x), y, mask)
        self.debugprint(x.shape)

        # mlp
        self.debugprint("feed-forward mlp")
        x = x + self.drop_path(gate_mlp * self.mlp(
            modulate(self.norm_mlp(x), shift_mlp, scale_mlp)))
        self.debugprint(x.shape)

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
        enable_temporal_attn=True,
        temporal_layer_type="conv3d",
        enable_mem_eff_attn=False,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        enable_grad_checkpoint=False,
        debug=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.debugprint = debugprint(debug)
        self.enable_grad_checkpoint = enable_grad_checkpoint

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
        self.temporal_layer_type = temporal_layer_type

        self.space_scale = space_scale
        self.time_scale = time_scale

        self.x_embedder = PatchEmbed3D(patch_size=patch_size,
                                       in_chans=in_channels,
                                       embed_dim=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size=hidden_size)
        # a shared adaLN for all blocks
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True),
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        self.pos_embed = nn.Parameter(self.get_spatial_pos_embed(),
                                      requires_grad=False)
        self.temporal_pos_embed = nn.Parameter(self.get_temporal_pos_embed(),
                                               requires_grad=False)

        drop_path = [val.item() for val in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            STDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                d_s=self.num_spatial,
                d_t=self.num_temporal,
                d_s_h=self.num_spatial_h,
                d_s_w=self.num_spatial_w,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                enable_temporal_attn=enable_temporal_attn,
                temporal_layer_type=temporal_layer_type,
                enable_mem_eff_attn=enable_mem_eff_attn,
                enable_flashattn=enable_flashattn,
                enable_layernorm_kernel=enable_layernorm_kernel,
                enable_sequence_parallelism=enable_sequence_parallelism,
                debug=debug,
            ) for i in range(depth)
        ])

        self.st_attn_bias = None
        if temporal_layer_type == "spatial_temporal_attn":
            st_attn_mask = get_st_attn_mask(T=self.num_temporal,
                                            H=self.num_spatial_h,
                                            W=self.num_spatial_w,
                                            t_window=8,
                                            s_window=5)  # [M, M]
            st_attn_bias = get_attn_bias_from_mask(st_attn_mask).unsqueeze(
                0).repeat([num_heads, 1, 1])  # [H, M, M]
            self.st_attn_bias = nn.Parameter(st_attn_bias, requires_grad=False)
            self.debugprint("st attn bias shape", self.st_attn_bias.shape)

        self.final_layer = T2IFinalLayer(hidden_size,
                                         self.patch_size_nd,
                                         out_channels=self.out_channels)

        self.initialize_weights()
        self.initialize_temporal()

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
        bs = x.shape[0]

        self.debugprint("inside ", self.__class__)

        # x embedding
        self.debugprint(x.shape)
        x = self.x_embedder(x)  # [B, Nt*Nh*Nw, C]
        self.debugprint(x.shape)

        # prepare for spatial
        x = rearrange(x,
                      "B (T S) C -> B T S C",
                      T=self.num_temporal,
                      S=self.num_spatial)
        self.debugprint(x.shape)
        x = x + self.pos_embed
        x = rearrange(x, "B T S C -> B (T S) C")
        self.debugprint(x.shape)

        self.debugprint("t shapes")
        self.debugprint(t.shape)
        t = self.t_embedder(t, x.dtype)  # [B, C]
        self.debugprint(t.shape)
        t0 = self.t_block(t)  #[B, 9*C]
        self.debugprint(t0.shape)

        self.debugprint("y shapes")
        self.debugprint(y.shape)
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
        self.debugprint(y.shape)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(
                1, -1, y.shape[-1])
            y_lens = mask.sum(dim=1).tolist()  # [n1, n2, n3, ,,,, n_bs]
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, y.shape[-1])  # [BxN_token, C]
        self.debugprint(y.shape)

        st_attn_bias = self.st_attn_bias
        if self.st_attn_bias is not None:
            st_attn_bias = self.st_attn_bias.unsqueeze(0).repeat(
                [bs, 1, 1, 1])  # [B, H, M, M]

        for block in self.blocks:
            if self.enable_grad_checkpoint:
                x = auto_grad_checkpoint(block, x, y, t0, y_lens,
                                         self.temporal_pos_embed, st_attn_bias)
            else:
                x = block(x, y, t0, y_lens, self.temporal_pos_embed,
                          st_attn_bias)

        self.debugprint("blocks out:", x.shape)
        # [N, num_patches, patch_size_nd * out_channels]
        x = self.final_layer(x, t)
        self.debugprint("final layer: ", x.shape)
        x = self.unpatchify(x)  # [N, C, T, H, W]
        self.debugprint("STDiT output: ", x.shape)

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
            Pw=self.patch_size[2],
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
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.num_temporal,
            scale=self.time_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
        return pos_embed

    def initialize_temporal(self):
        if ("attn"
                in self.temporal_layer_type) or ("attention"
                                                 in self.temporal_layer_type):
            for block in self.blocks:
                nn.init.constant_(block.t_attn.proj.weight, 0)
                nn.init.constant_(block.t_attn.proj.bias, 0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # zero-out adaLN for timestep
        nn.init.constant_(self.t_block[-1].weight, 0)
        nn.init.constant_(self.t_block[-1].bias, 0)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
