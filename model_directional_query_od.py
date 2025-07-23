import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_wavelets import DWTForward, DWTInverse
import matplotlib.pyplot as plt
from odconv import ODConv2d
from duck_block import DUCKBlock


class od_attention(nn.Module):
    def __init__(self, channels):
        super(od_attention, self).__init__()

        self.od_conv = ODConv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        od_out = self.od_conv(x)
        
        out = self.conv(x)
        attention = F.gelu(od_out)

        return out*attention


class MFE(nn.Module): 
    def __init__(self, channels):
        super(MFE, self).__init__()

        self.conv1 = DUCKBlock(channels)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.conv9 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)

        self.conv_cat = nn.Conv2d(channels*4, channels, kernel_size=3, padding=1, groups=channels, bias=False)

    def forward(self, x):

        aa =  DWTForward(J=1, mode='zero', wave='db3').cuda()
        yl, yh = aa(x)

        yh_out = yh[0]
        ylh = yh_out[:,:,0,:,:]
        yhl = yh_out[:,:,1,:,:]
        yhh = yh_out[:,:,2,:,:]

        conv_rec1 = self.conv1(yl)
        conv_rec5 = self.conv5(ylh)
        conv_rec7 = self.conv7(yhl)
        conv_rec9 = self.conv9(yhh)

        cat_all = torch.stack((conv_rec5, conv_rec7, conv_rec9),dim=2)
        rec_yh = []
        rec_yh.append(cat_all)


        ifm = DWTInverse(wave='db3', mode='zero').cuda()
        Y = ifm((conv_rec1, rec_yh))

        return Y

class DGA(nn.Module):
    def __init__(self, channels, num_heads):
        super(DGA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv_rgb = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.qkv_depth = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)

        self.qkv_conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1,
                                  groups=channels * 2, bias=False)
        self.query = MFE(channels)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, rgb_feat, depth_feat):
        b, c, h, w = rgb_feat.shape

        q_rgb = self.query(rgb_feat)
        k_rgb, v_rgb = self.qkv_conv(self.qkv_rgb(rgb_feat)).chunk(2, dim=1)
        q1 = q_rgb.reshape(b, self.num_heads, -1, h * w)
        k1 = k_rgb.reshape(b, self.num_heads, -1, h * w)
        v1 = v_rgb.reshape(b, self.num_heads, -1, h * w)
        attn1 = torch.softmax(torch.matmul(F.normalize(q1, dim=-1), F.normalize(k1, dim=-1).transpose(-2, -1)) * self.temperature, dim=-1)
        out_self = torch.matmul(attn1, v1).reshape(b, -1, h, w)

        k_depth, v_depth = self.qkv_conv(self.qkv_depth(depth_feat)).chunk(2, dim=1)
        k2 = k_depth.reshape(b, self.num_heads, -1, h * w)
        v2 = v_depth.reshape(b, self.num_heads, -1, h * w)
        attn2 = torch.softmax(torch.matmul(q1, F.normalize(k2, dim=-1).transpose(-2, -1)) * self.temperature, dim=-1)
        out_cross = torch.matmul(attn2, v2).reshape(b, -1, h, w)

        out = self.project_out(out_self * out_cross)
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = DGA(channels, num_heads) 
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, rgb_feat, depth_feat):
        b, c, h, w = rgb_feat.shape

        norm_rgb = self.norm1(rgb_feat.reshape(b, c, -1).transpose(-2, -1)).transpose(-2, -1).reshape(b, c, h, w)
        x = rgb_feat + self.attn(norm_rgb, depth_feat)

        norm_x = self.norm2(x.reshape(b, c, -1).transpose(-2, -1)).transpose(-2, -1).reshape(b, c, h, w)
        x = x + self.ffn(norm_x)

        return x

class EncoderWrapper(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, rgb_feat, depth_feat):
        for block in self.blocks:
            rgb_feat = block(rgb_feat, depth_feat)
        return (rgb_feat, depth_feat)

class DecoderWrapper(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, rgb_feat, depth_feat):
        for block in self.blocks:
            rgb_feat = block(rgb_feat, depth_feat)
        return rgb_feat

class RefinementWrapper(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, depth):
        for blk in self.blocks:
            x = blk(x, depth)
        return x

class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Inpainting(nn.Module):
    def __init__(self, num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48//3, 96//3, 192//3, 384//3], num_refinement=4,
                 expansion_factor=2.66):
        super(Inpainting, self).__init__()
        self.rgb_embed = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.depth_embed = nn.Conv2d(1, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([
            EncoderWrapper([TransformerBlock(num_ch, num_ah, expansion_factor) for _ in range(num_tb)])
            for num_tb, num_ah, num_ch in zip(num_blocks, num_heads, channels)
        ])

        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        
        self.skips = nn.ModuleList([od_attention(num_ch) for num_ch in list(reversed(channels))[1:]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        self.reduces = nn.ModuleList([
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.Conv2d(48, 16, kernel_size=1, bias=False)
            ])
        self.depth_reduces = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=1, bias=False),  
            nn.Conv2d(32, 32, kernel_size=1, bias=False), 
            nn.Conv2d(16, 32, kernel_size=1, bias=False) 
            ])
        self.decoders = nn.ModuleList([
            DecoderWrapper([TransformerBlock(channels[2], num_heads[2], expansion_factor)
                            for _ in range(num_blocks[2])]),
            DecoderWrapper([TransformerBlock(channels[1], num_heads[1], expansion_factor)
                            for _ in range(num_blocks[1])]),
            DecoderWrapper([TransformerBlock(channels[1], num_heads[0], expansion_factor)
                            for _ in range(num_blocks[0])])
])

        self.refinement = RefinementWrapper([TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        rgb, depth = x[:, :3, :, :], x[:, 3:, :, :]
        rgb_feat = self.rgb_embed(rgb)
        depth_feat = self.depth_embed(depth)

        out_enc1 = self.encoders[0](rgb_feat, depth_feat)
        rgb_down2 = self.downs[0](out_enc1[0])
        depth_down2 = self.downs[0](out_enc1[1])
        out_enc2 = self.encoders[1](rgb_down2, depth_down2)
        
        rgb_down3 = self.downs[1](out_enc2[0])
        depth_down3 = self.downs[1](out_enc2[1])
        out_enc3 = self.encoders[2](rgb_down3, depth_down3)
    
        rgb_down4 = self.downs[2](out_enc3[0])
        depth_down4 = self.downs[2](out_enc3[1])
        out_enc4 = self.encoders[3](rgb_down4, depth_down4)

        first_1 = self.ups[0](out_enc4[0])
        first_2 = self.skips[0](out_enc3[0])
        rgb_dec3_input = self.reduces[0](torch.cat([self.ups[0](out_enc4[0]), self.skips[0](out_enc3[0])], dim=1))
        depth_dec3_input = out_enc3[1]
        depth_dec3_input = self.depth_reduces[0](depth_dec3_input)
        out_dec3 = self.decoders[0](rgb_dec3_input, depth_dec3_input)

        second_1 = self.ups[1](out_dec3)
        second_2 = self.skips[1](out_enc2[0])
        rgb_dec2_input = self.reduces[1](torch.cat([self.ups[1](out_dec3), self.skips[1](out_enc2[0])], dim=1))
        depth_dec2_input = out_enc2[1]
        depth_dec2_input = self.depth_reduces[1](depth_dec2_input)
        out_dec2 = self.decoders[1](rgb_dec2_input, depth_dec2_input)
        
        rgb_fd_input = torch.cat([self.ups[2](out_dec2), self.skips[2](out_enc1[0])], dim=1)
        depth_fd_input = out_enc1[1]
        depth_fd_input = self.depth_reduces[2](depth_fd_input)
        fd = self.decoders[2](rgb_fd_input, depth_fd_input)
        depth_refine = self.depth_reduces[2](out_enc1[1])
        fr = self.refinement(fd, depth_refine)
        out = self.output(fr)
        return out
    
def model_final(num_blocks, num_heads, channels, num_refinement, expansion_factor):
    return Inpainting(
        num_blocks=num_blocks,
        num_heads=num_heads,
        channels=channels,
        num_refinement=num_refinement,
        expansion_factor=expansion_factor
    )