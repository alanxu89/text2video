from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn

from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UpBlock2D,
)
from diffusers.models.attention_processor import Attention

from einops import rearrange

from blocks import get_1d_sincos_pos_embed


def get_down_block(
    n_frames: int,
    n_temp_heads: int,
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    downsample_type: Optional[str] = None,
    dropout: float = 0.0,
):
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        print(
            f"It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    down_block_type = down_block_type[7:] if down_block_type.startswith(
        "UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "CrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnDownBlock2D"
            )
        return VLDMCrossAttnDownBlock(
            n_frames=n_frames,
            n_temp_heads=n_temp_heads,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
        )

    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    n_frames: int,
    n_temp_heads: int,
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resolution_idx: Optional[int] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    upsample_type: Optional[str] = None,
    dropout: float = 0.0,
) -> nn.Module:
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        print(
            f"It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    up_block_type = up_block_type[7:] if up_block_type.startswith(
        "UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "CrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnUpBlock2D")
        return VLDMCrossAttnUpBlock2D(
            n_frames=n_frames,
            n_temp_heads=n_temp_heads,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
        )
    raise ValueError(f"{up_block_type} does not exist.")


class Conv3DLayer(nn.Module):

    def __init__(self, in_dim, out_dim, n_frames, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_frames = n_frames
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), nn.SiLU(),
            nn.Conv3d(in_dim,
                      out_dim,
                      kernel_size=(3, 1, 1),
                      stride=1,
                      padding=(1, 0, 0)))
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(),
            nn.Conv3d(out_dim,
                      out_dim,
                      kernel_size=(3, 1, 1),
                      stride=1,
                      padding=(1, 0, 0)))

        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x shape: [b*t, c, h, w]

        h = rearrange(x, '(b t) c h w -> b c t h w', t=self.n_frames)
        h = self.block1(h)
        h = self.block2(h)
        h = rearrange(h, 'b c t h w -> (b t) c h w')
        with torch.no_grad():
            self.alpha.clamp_(0, 1)

        x = self.alpha * x + (1.0 - self.alpha) * h
        return x


class TemporalAttention(nn.Module):

    def __init__(self, dim, n_frames, n_heads=8, kv_dim=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_frames = n_frames
        self.n_heads = n_heads

        self.pos_enc = nn.Parameter(torch.from_numpy(
            get_1d_sincos_pos_embed(dim, length=n_frames)),
                                    requires_grad=False)

        # print("TemporalAttention", dim, n_frames, n_heads)
        self.attn = Attention(query_dim=dim,
                              heads=n_heads,
                              cross_attention_dim=kv_dim)

        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x, y):
        # x shape: [b*t, cx, h, w]
        # y shape: [b*t, n, cy]

        skip = x
        bt, c, h, w = x.shape

        # add pos_enc to the t dim
        x = rearrange(x, '(b t) c h w -> b (h w) t c', t=self.n_frames)
        x = x + self.pos_enc.to(x.dtype)
        x = rearrange(x, 'b (h w) t c -> (b h w) t c', h=h, w=w)

        # process cond
        b = bt // self.n_frames
        # [b*t, n, c] -> [b, 1, n, c]
        y = y[:b][:, None]
        # => [b, h*w, n, c]
        # y = y.expand(-1, h * w, -1, -1)
        y = y.repeat(1, h * w, 1, 1)
        # [b*h*w, n, c]
        y = rearrange(y, 'b (h w) n c -> (b h w) n c', h=h, w=w)

        x = self.attn(x, y)
        x = rearrange(x, '(b h w) t c -> (b t) c h w', h=h, w=w)

        with torch.no_grad():
            self.alpha.clamp_(0, 1)

        out = self.alpha * skip + (1 - self.alpha) * x

        return out


class VLDMCrossAttnDownBlock(CrossAttnDownBlock2D):

    def __init__(self, n_frames=8, n_temp_heads=8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        out_channels = kwargs['out_channels']
        num_layers = kwargs['num_layers']
        cross_attn_dim = kwargs.get('cross_attention_dim')

        conv_3ds = []
        tempo_attns = []

        for _ in range(num_layers):
            conv_3ds.append(
                Conv3DLayer(in_dim=out_channels,
                            out_dim=out_channels,
                            n_frames=n_frames))

            tempo_attns.append(
                TemporalAttention(dim=out_channels,
                                  n_frames=n_frames,
                                  n_heads=n_temp_heads,
                                  kv_dim=cross_attn_dim))

        self.conv_3ds = nn.ModuleList(conv_3ds)
        self.temp_attns = nn.ModuleList(tempo_attns)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        additional_residuals: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                print(
                    "Passing `scale` to `cross_attention_kwargs` is depcrecated. `scale` will be ignored."
                )

        output_states = ()

        blocks = list(
            zip(self.resnets, self.attentions, self.conv_3ds, self.temp_attns))

        for i, (resnet, attn, conv3d, temp_attn) in enumerate(blocks):
            # resnet
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = resnet(hidden_states, temb)

            # conv3d
            hidden_states = conv3d(hidden_states)

            # spatial attn
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

            # temporal attn
            hidden_states = temp_attn(
                hidden_states,
                encoder_hidden_states,
            )

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states, )

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states, )

        return hidden_states, output_states


class VLDMCrossAttnUpBlock2D(CrossAttnUpBlock2D):

    def __init__(self, n_frames=8, n_temp_heads=8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        out_channels = kwargs['out_channels']
        num_layers = kwargs['num_layers']
        cross_attn_dim = kwargs.get('cross_attention_dim')

        cross_attn_dim = kwargs.get('cross_attention_dim')

        conv_3ds = []
        tempo_attns = []

        for _ in range(num_layers):
            conv_3ds.append(
                Conv3DLayer(in_dim=out_channels,
                            out_dim=out_channels,
                            n_frames=n_frames))

            tempo_attns.append(
                TemporalAttention(dim=out_channels,
                                  n_frames=n_frames,
                                  n_heads=n_temp_heads,
                                  kv_dim=cross_attn_dim))

        self.conv_3ds = nn.ModuleList(conv_3ds)
        self.temp_attns = nn.ModuleList(tempo_attns)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                print(
                    "Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored."
                )

        blocks = list(
            zip(self.resnets, self.attentions, self.conv_3ds, self.temp_attns))

        for i, (resnet, attn, conv3d, temp_attn) in enumerate(blocks):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states],
                                      dim=1)

            # resnet
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = resnet(hidden_states, temb)

            # conv_3d
            hidden_states = conv3d(hidden_states)

            # spatial attn
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

            # temporal attn
            hidden_states = temp_attn(hidden_states, encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
