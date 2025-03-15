from TimestepBlock import *
from HyperNetwork import HyperNetwork
from ..buffer.GroupNorm32 import *
from ..buffer.Upsample import *
from ..buffer.Downsample import *
from ..buffer.CheckpointFunction import *

class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        # have_hyper=False,
        z_coef=0,  # 频域信息系数，范围1-0，0为纯sd，1为纯hyper
        z_dim=64,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.z_coef = z_coef

        if z_coef == 0:
            self.w_sd = nn.Parameter(torch.randn(self.out_channels, channels, 3, 3))     # sd的权重矩阵
        elif 0 < z_coef < 1:
            self.w_sd = nn.Parameter(torch.randn(self.out_channels, channels, 3, 3))
            self.w_hyper = None     # hyper生成的权重矩阵
            self.hyper = HyperNetwork(f_size=3, z_dim=z_dim, out_size=out_channels, in_size=channels)
        elif z_coef == 1:
            self.w_hyper = None
            self.hyper = HyperNetwork(f_size=3, z_dim=z_dim, out_size=out_channels, in_size=channels)

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            # nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        # self.conv1 = nn.Conv2d(channels, self.out_channels, 3, padding=1) if not have_hyper else nn.Identity()

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb, z=None):

        if(z is not None):
            return checkpoint(
                self._forward, (x, emb, z), self.parameters(), self.use_checkpoint
            )
        else:
            return checkpoint(
                self._forward, (x, emb), self.parameters(), self.use_checkpoint
            )

    def _forward(self, x, emb, z=None):
        if self.updown:
            h = self.in_layers(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            # h = self.conv1(h)
        else:
            h = self.in_layers(x)
            # h = self.conv1(h)

        if z is not None:
            if self.z_coef == 0:
                h = F.conv2d(h, self.w_sd, stride=1, padding=1)
            elif 0 < self.z_coef < 1:
                self.w_hyper = self.hyper(z)
                w = self.w_sd * (1 - self.z_coef) + self.w_hyper * self.z_coef  # 用于最终计算的权重矩阵
                h = F.conv2d(h, w, stride=1, padding=1)
            elif self.z_coef == 1:
                self.w_hyper = self.hyper(z)
                h = F.conv2d(h, self.w_hyper, stride=1, padding=1)
        elif z is None:  # z为None，是纯sd或者为生成过程
            if self.z_coef == 0:
                h = F.conv2d(h, self.w_sd, stride=1, padding=1)
            elif 0 < self.z_coef < 1:
                w = self.w_sd * (1 - self.z_coef) + self.w_hyper * self.z_coef
                h = F.conv2d(h, w, stride=1, padding=1)
            elif self.z_coef == 1:
                h = F.conv2d(h, self.w_hyper, stride=1, padding=1)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
