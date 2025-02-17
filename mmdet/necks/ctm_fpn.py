import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    def __init__(self, inC, outC, Kencoder=3, delta=2, Kup=5, Cm=64):
        super(CARAFE, self).__init__()
        self.Kencoder = Kencoder
        self.delta = delta
        self.Kup = Kup
        self.down = nn.Conv2d(inC, Cm, kernel_size=1)
        self.encoder = nn.Conv2d(Cm, delta ** 2 * Kup ** 2, Kencoder,
                                 padding=Kencoder // 2)
        self.out = nn.Conv2d(inC, outC, 1)

    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()

        # Kernel prediction
        kernel = self.down(in_tensor)
        kernel = self.encoder(kernel)
        kernel = F.pixel_shuffle(kernel, self.delta)
        kernel = F.softmax(kernel, dim=1)
        kernel = kernel.unfold(2, self.delta, self.delta)
        kernel = kernel.unfold(3, self.delta, self.delta)
        kernel = kernel.reshape(N, self.Kup ** 2, H, W, self.delta ** 2)
        kernel = kernel.permute(0, 2, 3, 1, 4)

        # Feature reassembly
        in_tensor = F.pad(in_tensor, (self.Kup // 2,) * 4)
        in_tensor = in_tensor.unfold(2, self.Kup, 1)
        in_tensor = in_tensor.unfold(3, self.Kup, 1)
        in_tensor = in_tensor.reshape(N, C, H, W, -1)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)

        out = torch.matmul(in_tensor, kernel)
        out = out.reshape(N, H, W, -1)
        out = out.permute(0, 3, 1, 2)
        out = F.pixel_shuffle(out, self.delta)
        return self.out(out)


class ContextExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return self.relu(x + residual)


class TextureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return self.relu(x + residual)


class MRASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[3, 6, 12, 18, 24]):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, 1)
        self.aspp_blocks = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3,
                      padding=d, dilation=d) for d in dilations
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_in(x)
        outputs = [x]
        for i, conv in enumerate(self.aspp_blocks):
            if i == 0:
                out = self.relu(conv(x)) + x
            else:
                out = self.relu(conv(outputs[-1])) + outputs[-1]
            outputs.append(out)
        return sum(outputs)


class CTM_FPN(nn.Module):
    def __init__(self, backbone_channels, fpn_channels=256):
        super().__init__()
        # Lateral convolutions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, 1) for c in backbone_channels
        ])

        # Context extractors
        self.ec_5 = ContextExtractor(fpn_channels * 2, fpn_channels, 3)
        self.ec_4 = ContextExtractor(fpn_channels * 3 // 2, fpn_channels, 3)
        self.ec_3 = ContextExtractor(fpn_channels * 3 // 2, fpn_channels, 3)
        self.ec_2 = ContextExtractor(fpn_channels * 3 // 2, fpn_channels, 3)

        # 1x1 convs for channel adjustment
        self.f1x1_5 = nn.Conv2d(fpn_channels, fpn_channels * 2, 1)
        self.f1x1_4 = nn.Conv2d(fpn_channels, fpn_channels // 2, 1)
        self.f1x1_3 = nn.Conv2d(fpn_channels, fpn_channels // 2, 1)
        self.f1x1_2 = nn.Conv2d(fpn_channels, fpn_channels // 2, 1)

        # Texture extractor
        self.et = TextureExtractor(fpn_channels, fpn_channels)

        # CARAFE upsampling with official implementation
        self.carafe = CARAFE(fpn_channels, fpn_channels, delta=2)

        # MRASPP
        self.mraspp = MRASPP(fpn_channels, fpn_channels)

        # c2* branch
        self.c2_star = nn.Conv2d(backbone_channels[0], fpn_channels, 3, padding=1)

    def forward(self, c2, c3, c4, c5):
        # Bottom-up path
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, scale_factor=2, mode='nearest')

        # Top-down refinement
        p5_star = self.ec_5(self.f1x1_5(p5))

        p4_star = self.ec_4(torch.cat([p4, self.f1x1_4(self.carafe(p5_star))], dim=1))

        p3_star = self.ec_3(torch.cat([p3, self.f1x1_3(self.carafe(p4_star))], dim=1))

        p2_star = self.ec_2(torch.cat([p2, self.f1x1_2(self.carafe(p3_star))], dim=1))

        # p1* generation
        c2_star = self.c2_star(c2)
        te_out = self.et(self.carafe(p2_star))
        p1_star = self.mraspp(c2_star + te_out)

        return [p1_star, p2_star, p3_star, p4_star, p5_star]