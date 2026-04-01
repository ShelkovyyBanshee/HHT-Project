import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class UNetResNet34(nn.Module):
    def __init__(self, in_channels=5, num_classes_seg=1, num_classes_clf=4, pretrained=True):
        super().__init__()

        self.backbone = models.resnet34(pretrained=pretrained)

        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu)
        self.pool0 = self.backbone.maxpool
        self.encoder1 = self.backbone.layer1
        self.encoder2 = self.backbone.layer2
        self.encoder3 = self.backbone.layer3
        self.encoder4 = self.backbone.layer4

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True))

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True))

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))

        self.seg_head = nn.Conv2d(64, num_classes_seg, kernel_size=1)

        self.clf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes_clf)
        )

    def forward(self, x):
        e0 = self.encoder0(x)
        p0 = self.pool0(e0)
        e1 = self.encoder1(p0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.up4(e4)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e0], dim=1))

        seg_out = self.seg_head(d1)
        seg_out = F.interpolate(seg_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        clf_out = self.clf_head(e4)  # [B, 4]

        return seg_out, clf_out