import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalDiscriminator(nn.Module):
    """
    입력:
      - cond: (B, 1, H, W)  # 예: SAX projection
      - img:  (B, 1, H, W)  # real or fake LAX projection
    출력:
      - (B, 1)            # real/fake 확률
    """
    def __init__(self, in_channels=2, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            # input: (B, in_channels, H, W)
            nn.Conv2d(in_channels, base_ch, 4, 2, 1),  # → (B,64,H/2,W/2)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch, base_ch*2, 4, 2, 1),    # → (B,128,H/4,W/4)
            nn.BatchNorm2d(base_ch*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1),  # → (B,256,H/8,W/8)
            nn.BatchNorm2d(base_ch*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch*4, 1, 4, 1, 0),          # → (B,1, (H/8−3), (W/8−3))
            nn.Sigmoid()
        )

    def forward(self, cond, img):          # cond, img 두 채널을 합친 것
        x = torch.cat([cond, img], dim=1)  # channel-wise concat
        out = self.net(x)
        return out.view(out.size(0), -1)   # → (B,1)
