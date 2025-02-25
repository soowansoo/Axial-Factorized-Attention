from timm.models.layers import DropPath
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Conv2d_BN_fusion(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):

        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        # print(x.shape, self.m(x).shape)
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN_fusion):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepDW(torch.nn.Module):
    def __init__(self, ed, stride=1) -> None:
        super().__init__()
        self.use_res_connect = stride == 1
        self.conv = Conv2d_BN_fusion(ed, ed, 3, stride, 1, groups=ed)
        # torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.conv1 = Conv2d_BN_fusion(ed, ed, 3, stride, 1, groups=ed)
        self.conv2 = torch.nn.Conv2d(ed, ed, 1, stride, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + self.conv2(x))

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        conv2 = self.conv2

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        conv2_w = conv2.weight
        conv2_b = conv2.bias

        conv2_w = torch.nn.functional.pad(conv2_w, [1, 1, 1, 1])

        # identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + conv2_w  # + identity
        final_conv_b = conv_b + conv1_b + conv2_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepSA(torch.nn.Module):
    def __init__(self, ed, stride=1, padding=1, dilation=1) -> None:
        super().__init__()
        self.use_res_connect = stride == 1
        self.conv = Conv2d_BN_fusion(
            ed, ed, 3, stride, pad=padding, dilation=dilation, groups=ed)
        self.conv1 = Conv2d_BN_fusion(
            ed, ed, 3, stride, pad=padding, dilation=dilation, groups=ed)
        self.conv2 = torch.nn.Conv2d(ed, ed, 1, stride, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x) + self.conv2(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        conv2 = self.conv2

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        conv2_w = conv2.weight
        conv2_b = conv2.bias

        conv2_w = torch.nn.functional.pad(conv2_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(
            conv2_w.shape[0], conv2_w.shape[1], 1, 1, device=conv2_w.device), [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + conv2_w + identity
        final_conv_b = conv_b + conv1_b + conv2_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepMV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            if self.use_res_connect:
                self.conv = Residual(nn.Sequential(
                    # dw
                    Conv2d_BN_fusion(hidden_dim, hidden_dim, 3,
                                     stride, 1, groups=hidden_dim),
                    nn.SiLU(),
                    # pw-linear
                    Conv2d_BN_fusion(hidden_dim, oup, 1, 1, 0),
                ))
            else:
                self.conv = nn.Sequential(
                    # dw
                    Conv2d_BN_fusion(hidden_dim, hidden_dim, 3,
                                     stride, 1, groups=hidden_dim),
                    nn.SiLU(),
                    # pw-linear
                    Conv2d_BN_fusion(hidden_dim, oup, 1, 1, 0),
                )
        else:
            if self.use_res_connect:
                self.conv = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN_fusion(inp, hidden_dim, 1, 1, 0),
                    nn.SiLU(),
                    # dw
                    RepDW(hidden_dim, stride),
                    nn.SiLU(),
                    # pw-linear
                    Conv2d_BN_fusion(hidden_dim, oup, 1, 1, 0)
                ))
            else:
                self.conv = nn.Sequential(
                    # pw
                    Conv2d_BN_fusion(inp, hidden_dim, 1, 1, 0),
                    nn.SiLU(),
                    # dw
                    RepDW(hidden_dim, stride),
                    nn.SiLU(),
                    # pw-linear
                    Conv2d_BN_fusion(hidden_dim, oup, 1, 1, 0)
                )

    def forward(self, x):
        return self.conv(x)


class RepMVDP2Block2(nn.Module):
    def __init__(self, inp, oup, stride=1, padding=18, dilation=18, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if self.use_res_connect:
            self.conv = Residual(nn.Sequential(
                # pw
                Conv2d_BN_fusion(inp, hidden_dim, 1, 1, 0),
                nn.SiLU(),
                # dw
                RepSA(hidden_dim, stride, padding, dilation),
                nn.SiLU(),
                RepSA(hidden_dim, stride, int(
                    padding * (2/3)), int(dilation * (2/3))),
                nn.SiLU(),
                RepSA(hidden_dim, stride, int(
                    padding * (1/3)), int(dilation * (1/3))),
                nn.SiLU(),
                # pw-linear
                Conv2d_BN_fusion(hidden_dim, oup, 1, 1, 0)
            ))
        else:
            self.conv = nn.Sequential(
                # pw
                Conv2d_BN_fusion(inp, hidden_dim, 1, 1, 0),
                nn.SiLU(),
                # dw
                RepSA(hidden_dim, stride, padding, dilation),
                nn.SiLU(),
                RepSA(hidden_dim, stride, int(
                    padding * (2/3)), int(dilation * (2/3))),
                nn.SiLU(),
                RepSA(hidden_dim, stride, int(
                    padding * (1/3)), int(dilation * (1/3))),
                nn.SiLU(),
                # pw-linear
                Conv2d_BN_fusion(hidden_dim, oup, 1, 1, 0)
            )

    def forward(self, x):
        return self.conv(x)


class Conv1d_BN_fusion(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000, ver=True):

        super().__init__()
        if ver == True:
            self.add_module('c', torch.nn.Conv2d(
                a, b, (ks, 1), (stride, 1), (pad, 0), (dilation, 1), groups, bias=False))
        else:
            self.add_module('c', torch.nn.Conv2d(
                a, b, (1, ks), (1, stride), (0, pad), (1, dilation), groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepSP1DAttn_DW(nn.Module):
    def __init__(self, in_ch, out_ch, ks, pad, stride, ver=True):
        super().__init__()

        self.ver = ver

        self.conv = Conv1d_BN_fusion(
            in_ch, out_ch, ks, stride, pad=pad, groups=in_ch, ver=ver)
        self.conv1 = Conv1d_BN_fusion(
            in_ch, out_ch, (ks-2), stride, pad=(pad-1), groups=in_ch, ver=ver)
        self.conv2 = Conv1d_BN_fusion(
            in_ch, out_ch, (ks-4), stride, pad=(pad-2), groups=in_ch, ver=ver)
        self.dim = out_ch
        self.bn = torch.nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x) + self.conv1(x) + self.conv2(x))

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        conv2 = self.conv2.fuse()

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        conv2_w = conv2.weight
        conv2_b = conv2.bias

        if self.ver == True:
            conv1_w = torch.nn.functional.pad(conv1_w, [0, 0, 1, 1])  # 5
            conv2_w = torch.nn.functional.pad(conv2_w, [0, 0, 2, 2])  # 3
        else:
            conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 0, 0])
            conv2_w = torch.nn.functional.pad(conv2_w, [2, 2, 0, 0])

        final_conv_w = conv_w + conv1_w + conv2_w
        final_conv_b = conv_b + conv1_b + conv2_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepAFA_Block(nn.Module):
    def __init__(self, in_ch, out_ch, expans=3):
        super().__init__()

        hidden_ch = in_ch * expans
        self.in_ch = in_ch
        self.k = RepSP1DAttn_DW(in_ch, hidden_ch, ks=7,
                                pad=3, stride=1, ver=False)
        self.q = RepSP1DAttn_DW(in_ch, hidden_ch, ks=7,
                                pad=3, stride=1, ver=True)
        self.v = Conv2d_BN_fusion(hidden_ch * 2, hidden_ch)
        self.ff = nn.Sequential(
            Conv2d_BN_fusion(
                hidden_ch, hidden_ch, 3, 1, 1, groups=hidden_ch),
            Conv2d_BN_fusion(hidden_ch, out_ch)
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        key = self.k(x)
        query = self.q(x)
        val = self.v(torch.cat([key, query], dim=1))
        out = self.silu(val)
        out = self.ff(out)
        out = self.silu(out)
        return out


class AFASeg_S(nn.Module):
    def __init__(self, num_classes):
        super(AFASeg_S, self).__init__()
        self.stem = nn.Sequential(  # 1/2
            Conv2d_BN_fusion(3, 16, 3, 2, 1),
            nn.ReLU(),
            RepMV2Block(16, 16, 1, 3),
        )

        self.encoder = nn.Sequential(  # 1/4
            RepMV2Block(16, 32, 2, 3),
            RepAFA_Block(32, 32),
            RepAFA_Block(32, 32),
        )

        self.encoder2 = nn.Sequential(  # 1/8
            RepMV2Block(32, 96, 2, 3),
            RepAFA_Block(96, 96),
            RepAFA_Block(96, 96),
        )

        self.identify_layer1 = RepMVDP2Block2(
            96, 96, 1, padding=18, dilation=18, expansion=3)
        self.identify_layer3 = RepMVDP2Block2(
            96, 96, 1, padding=9, dilation=9, expansion=3)

        self.classifier_ = nn.Sequential(
            Conv2d_BN_fusion(224, 224, 3, 1, 1, groups=224),
            nn.ReLU(),
            Conv2d_BN_fusion(224, num_classes, 1)
        )

    def forward(self, x):

        out = self.stem(x)
        resolution = self.encoder(out)

        out = self.encoder2(resolution)
        start = out
        out = self.identify_layer1(out)
        out = self.identify_layer3(out)
        out = torch.cat([out, start], dim=1)
        out = F.interpolate(
            out, size=resolution.shape[2:], mode='bilinear', align_corners=False)
        out = self.classifier_(torch.cat([resolution, out], dim=1))
        out = F.interpolate(
            out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out


class AFASeg_XS(nn.Module):
    def __init__(self, num_classes):
        super(AFASeg_XS, self).__init__()
        self.stem = nn.Sequential(  # 1/2
            Conv2d_BN_fusion(3, 16, 3, 2, 1),
            nn.ReLU(),
            RepMV2Block(16, 16, 1, 3),
        )

        self.encoder = nn.Sequential(  # 1/4
            RepMV2Block(16, 32, 2, 3),
            RepAFA_Block(32, 32),
            RepAFA_Block(32, 32),
        )

        self.encoder2 = nn.Sequential(  # 1/8
            RepMV2Block(32, 64, 2, 3),
            RepAFA_Block(64, 64),
            RepAFA_Block(64, 64),
        )

        self.identify_layer1 = RepMVDP2Block2(
            64, 64, 1, padding=18, dilation=18, expansion=3)
        self.identify_layer3 = RepMVDP2Block2(
            64, 64, 1, padding=9, dilation=9, expansion=3)

        self.classifier_ = nn.Sequential(
            Conv2d_BN_fusion(160, 160, 3, 1, 1, groups=160),
            nn.ReLU(),
            Conv2d_BN_fusion(160, num_classes, 1)
        )

    def forward(self, x):

        out = self.stem(x)
        resolution = self.encoder(out)

        out = self.encoder2(resolution)
        start = out
        out = self.identify_layer1(out)
        out = self.identify_layer3(out)
        out = torch.cat([out, start], dim=1)
        # ###
        out = F.interpolate(
            out, size=resolution.shape[2:], mode='bilinear', align_corners=False)
        out = self.classifier_(torch.cat([resolution, out], dim=1))
        out = F.interpolate(
            out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out


class AFASeg_XXS(nn.Module):
    def __init__(self, num_classes):
        super(AFASeg_XXS, self).__init__()
        self.stem = nn.Sequential(  # 1/2
            Conv2d_BN_fusion(3, 14, 3, 2, 1),
            nn.ReLU(),
            RepMV2Block(14, 14, 1, 3),
        )

        self.encoder = nn.Sequential(  # 1/4
            RepMV2Block(14, 24, 2, 3),
            RepAFA_Block(24, 24),
            RepAFA_Block(24, 24),
        )

        self.encoder2 = nn.Sequential(  # 1/8
            RepMV2Block(24, 48, 2, 3),
            RepAFA_Block(48, 48),
            RepAFA_Block(48, 48),
        )

        self.identify_layer1 = RepMVDP2Block2(
            48, 48, 1, padding=18, dilation=18, expansion=3)
        self.identify_layer3 = RepMVDP2Block2(
            48, 48, 1, padding=9, dilation=9, expansion=3)

        self.classifier_ = nn.Sequential(
            Conv2d_BN_fusion(120, 120, 3, 1, 1, groups=120),
            nn.ReLU(),
            Conv2d_BN_fusion(120, num_classes, 1)
        )

    def forward(self, x):

        out = self.stem(x)
        resolution = self.encoder(out)

        out = self.encoder2(resolution)
        start = out
        out = self.identify_layer1(out)
        out = self.identify_layer3(out)
        out = torch.cat([out, start], dim=1)
        # ###
        out = F.interpolate(
            out, size=resolution.shape[2:], mode='bilinear', align_corners=False)
        out = self.classifier_(torch.cat([resolution, out], dim=1))
        out = F.interpolate(
            out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out


class Lightweight_PW(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.add = nn.Parameter(torch.randn(
            1, in_ch, 1, 1), requires_grad=True)
        self.mul = nn.Parameter(torch.randn(
            1, in_ch, 1, 1), requires_grad=True)
        self.bn = nn.BatchNorm2d(out_ch)

        if out_ch >= in_ch:
            self.trig = True
            self.expans = out_ch // in_ch
        else:
            self.trig = False

        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x):
        b, c, h, w = x.shape
        out = x * self.mul + self.add

        if self.trig:
            out = torch.cat([x, out, x], dim=1)
        else:
            out = out.view(b, self.out_ch, 3, h, w)
            out = out.mean(dim=2)

        out = self.bn(out)

        return out

class RepMV2Block_lite(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=1):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        
        if self.use_res_connect:
            self.conv = Residual(nn.Sequential(
                # pw
                Lightweight_PW(inp, hidden_dim),
                nn.ReLU(),
                # dw
                RepDW(hidden_dim, stride),
                nn.ReLU(),
                # pw-linear
                Lightweight_PW(hidden_dim, oup)
            ))
        else:
            self.conv = nn.Sequential(
                # pw
                Lightweight_PW(inp, hidden_dim),
                nn.ReLU(),
                # dw
                RepDW(hidden_dim, stride),
                nn.ReLU(),
                # pw-linear
                Lightweight_PW(hidden_dim, oup)
            )

    def forward(self, x):
        return self.conv(x)

class RepSP2DAttn_DW(nn.Module):
    def __init__(self, in_ch, out_ch, ks, pad, stride):
        super().__init__()

        self.conv = Conv2d_BN_fusion(
            in_ch, out_ch, ks, stride, pad=pad, groups=in_ch)
        self.conv1 = Conv2d_BN_fusion(
            in_ch, out_ch, (ks-2), stride, pad=(pad-1), groups=in_ch)
        self.conv2 = Conv2d_BN_fusion(
            in_ch, out_ch, (ks-4), stride, pad=(pad-2), groups=in_ch)
        self.dim = out_ch
        self.bn = torch.nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x) + self.conv1(x) + self.conv2(x))

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        conv2 = self.conv2.fuse()

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        conv2_w = conv2.weight
        conv2_b = conv2.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])
        conv2_w = torch.nn.functional.pad(conv2_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w + conv2_w
        final_conv_b = conv_b + conv1_b + conv2_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepAttn_lite(nn.Module):
    def __init__(self, in_ch, out_ch, expans=3):
        super().__init__()

        hidden_ch = in_ch * expans
        self.out_ch = out_ch

        self.k = RepSP1DAttn_DW(in_ch, hidden_ch, ks=7,
                                pad=3, stride=1, ver=False)
        self.q = RepSP1DAttn_DW(hidden_ch, hidden_ch, ks=7,
                                pad=3, stride=1, ver=True)
        self.v = RepSP2DAttn_DW(out_ch, out_ch, ks=7,
                                pad=3, stride=1)

        self.attn = nn.Parameter(torch.zeros(
            1, hidden_ch + in_ch, 1, 1), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        key = self.k(x)
        query = self.q(key)
        k_q = torch.mul(torch.cat([query, x], dim=1),
                        F.softmax(self.attn, dim=1))
        k_q = k_q.view(b, self.out_ch, -1, h, w)
        k_q = k_q.mean(dim=2)
        out = self.v(k_q)

        return out


class AFASeg_Edge(nn.Module):
    def __init__(self, num_classes):
        super(AFASeg_Edge, self).__init__()
        self.stem = nn.Sequential(  # 1/2
            Conv2d_BN_fusion(3, 6, 3, 2, 1),
            nn.ReLU(),
            nn.AvgPool2d(3, 2, 1),
        )

        self.sa1 = RepSA(6, 1, padding=18, dilation=18)

        self.encoder = nn.Sequential(  # 1/4
            nn.AvgPool2d(3, 2, 1),

            RepAttn_lite(12, 12),
            RepMV2Block_lite(12, 12, 1),
        )

        self.encoder2 = nn.Sequential(  # 1/8
            nn.AvgPool2d(3, 2, 1),

            RepAttn_lite(12, 12),
            RepMV2Block_lite(12, 12, 1),
        )

        self.classifier_ = nn.Sequential(
            Conv2d_BN_fusion(12, num_classes, 1)
        )

    def forward(self, x):

        out = self.stem(x)
        # first extract #

        sa = self.sa1(out)
        sa = torch.cat([sa, out], dim=1)

        resolution = self.encoder(sa)

        out = self.encoder2(resolution)

        out = self.classifier_(out)
        out = F.interpolate(
            out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out


def model_type(model, num_classes):
    if model == 'AFASeg_S':
        return AFASeg_S(num_classes)
    elif model == 'AFASeg_XS':
        return AFASeg_XS(num_classes)
    elif model == 'AFASeg_XXS':
        return AFASeg_XXS(num_classes)
    elif model == 'AFASeg_Edge':
        return AFASeg_Edge(num_classes)


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


if __name__ == "__main__":

    from torchinfo import summary
    net = model_type('AFASeg_S', 19)
    replace_batchnorm(net)
    summary(net, (1, 3, 360, 640), device='cuda')
