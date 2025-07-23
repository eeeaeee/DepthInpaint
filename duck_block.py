import torch
import torch.nn as nn
import torch.nn.functional as F

class DUCKBlock(nn.Module):
    """
    PyTorch implementation of DUCKBlock from DUCKNet.
    Multi-branch fusion with residual, dilated, and separable convolutions.
    Input and output tensor sizes are exactly the same.
    """

    def __init__(self, channels):
        super(DUCKBlock, self).__init__()
        
        # WideScope: 3 dilated convs
        self.wide = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(inplace=True),
        )
        
        # MidScope: 2 dilated convs
        self.mid = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        # Residual blocks: 1, 2, 3-layer
        self.res1 = self._make_res_block(channels)
        self.res2 = nn.Sequential(self._make_res_block(channels), self._make_res_block(channels))
        self.res3 = nn.Sequential(self._make_res_block(channels), self._make_res_block(channels), self._make_res_block(channels))

        # Separated Conv: 1x6 + 6x1
        self.sep = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 6), padding=(0, 3), padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(6, 1), padding=(3, 0), padding_mode='replicate'),
            nn.ReLU(inplace=True),
            )

        # Final fusion
        self.bn = nn.BatchNorm2d(channels)

    def _make_res_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        x1 = self.wide(x)
        x2 = self.mid(x)
        x3 = self.res1(x)
        x4 = self.res2(x)
        x5 = self.res3(x)
        x6 = self.sep(x)
        print(x1.shape, x6.shape)

        out = x1 + x2 + x3 + x4 + x5 + x6
        return self.bn(out)

if __name__ == "__main__":
    # Test DUCKBlock with dummy input
    import torch

    dummy_input = torch.randn(1, 64, 64, 64).cuda()
    model = DUCKBlock(64).cuda()
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    
