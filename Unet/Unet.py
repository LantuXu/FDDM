from ..hyper import *

class UNetModel(nn.Module):
    def __init__(
        self,
        use_checkpoint=True,
        use_fp16=True,
        in_channels=4,
        out_channels=4,
        model_channels=320,
        attention_resolutions=(4, 2, 1),
        num_res_blocks=2,
        channel_mult=(1, 2, 4, 4),
        num_head_channels=64,  # need to fix for flash-attn
        use_spatial_transformer=True,
        use_linear_in_transformer=True,
        transformer_depth=1,
        context_dim=768,  # 与文本编码后的最后一个维度保持一致

        dropout=0,
        conv_resample=True,
        num_heads=-1,
        dims=2,
        use_scale_shift_norm=False,
        num_attention_blocks=None,
        disable_middle_self_attn=False,

        # have_hyper=False,  # 默认不启用hypernetwork层
        z_coef=0,    # 频域信息系数，范围1-0，0为纯sd，1为纯hyper
        # z_in=6,
        z_dim=64,
        # z_layer=6,

        **kwargs
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):  # 每层均有num_res_blocks个残差块
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):  # 每层分别有channel_mult个残差块
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.dtype = torch.float16 if use_fp16 else torch.float32

        # self.have_hyper = have_hyper
        self.z_coef = z_coef
        self.z_dim = z_dim

        # if 0 < z_coef <= 1:
        #     self.embed = ToVectorEmbedding(in_dim=z_in, out_dim=z_dim, layernum=z_layer)    # 额外信息系数大于1则添加编码层

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        z_coef=z_coef,
                        z_dim=z_dim,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:

                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels
                    disabled_sa = False

                    if num_attention_blocks is None or nr < num_attention_blocks[level]:
                        layers.append(
                            SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        num_heads = ch // num_head_channels
        dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        z_coef=z_coef,
                        z_dim=z_dim,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:

                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels
                    disabled_sa = False

                    if num_attention_blocks is None or i < num_attention_blocks[level]:
                        layers.append(
                            SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps=None, context=None, z=None, **kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels)  # 时间步编码
        emb = self.time_embed(t_emb)  # 进一步处理，产生时间嵌入

        # if (z is not None) and 0 < self.z_coef <= 1:  # 编码嵌入信息
        #     z = self.embed(z)

        h = x.type(self.dtype)  # h为特定数据类型的输入
        for module in self.input_blocks:
            h = module(h, emb, context, z)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, z)
        h = h.type(x.dtype)

        return self.out(h)

