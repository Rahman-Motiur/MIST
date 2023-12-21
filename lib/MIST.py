import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn as nn



class Attention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 proj_drop=0.0,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        self.conv_q = nn.Conv2d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias,
                                groups=channels)
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                groups=channels)
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                groups=channels)
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)

        self.attention = nn.MultiheadAttention(embed_dim=channels,
                                               bias=attention_bias,
                                               batch_first=True,
                                               # dropout = 0.0,
                                               num_heads=self.num_heads)  # num_heads=self.num_heads)

    def _build_projection(self, x, qkv):

        if qkv == "q":
            x1 = F.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k":
            x1 = F.relu(self.conv_k(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "v":
            x1 = F.relu(self.conv_v(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)

        return proj

    def forward_conv(self, x):
        q = self._build_projection(x, "q")
        k = self._build_projection(x, "k")
        v = self._build_projection(x, "v")

        return q, k, v

    def forward(self, x):
        q, k, v = self.forward_conv(x)
        q = q.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        k = k.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        v = v.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)

        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))
        x1 = F.dropout(x1, self.proj_drop)

        return x1


class ChannelAttentionCBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionCBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# Squeeze-and-excitation
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, channels)
        y = self.excitation(y).view(batch_size, channels, 1, 1)
        return x * y


class SpatialAttentionCBAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionCBAM, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionCBAM(in_channels, reduction_ratio)
        self.SE = SEBlock(in_channels, reduction_ratio=16)
        self.spatial_attention = SpatialAttentionCBAM(kernel_size)

    def forward(self, x):
        se_out = x * self.channel_attention(x)
        sa_out = x * self.spatial_attention(x)
        return torch.add(se_out, sa_out)

class Transformer(nn.Module):

    def __init__(self,
                 # in_channels,
                 out_channels,
                 num_heads,
                 dpr,
                 proj_drop=0.0,
                 attention_bias=True,
                 padding_q="same",
                 padding_kv="same",
                 stride_kv=1,
                 stride_q=1):
        super().__init__()

        self.attention_output = Attention(channels=out_channels,
                                          num_heads=num_heads,
                                          proj_drop=proj_drop,
                                          padding_q=padding_q,
                                          padding_kv=padding_kv,
                                          stride_kv=stride_kv,
                                          stride_q=stride_q,
                                          attention_bias=attention_bias,
                                          )

        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.layernorm = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.wide_focus = Dilated_Conv(out_channels, out_channels)
        #self.cbam=CBAM(out_channels)

    def forward(self, x):
        x1 = self.attention_output(x)
        x1 = self.conv1(x1)
        x2 = torch.add(x1, x)
        x3 = x2.permute(0, 2, 3, 1)
        x3 = self.layernorm(x3)
        x3 = x3.permute(0, 3, 1, 2)
        x3 = self.wide_focus(x3)
        x3 = torch.add(x2, x3)
        #extras
        #x4=self.cbam(x3)
        #x4 = torch.add(x2, x4)
        return x3


class Dilated_Conv(nn.Module):
    """
    Wide-Focus module.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv1=  nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)
        x2 = self.conv2(x)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)
        x3=self.conv3(x)
        x3=F.gelu(x3)
        x3=F.dropout(x3, 0.1)
        added = torch.add(x1, x2)
        added = torch.add(added, x3)
        x_out = self.conv4(added)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, 0.1)
        return x_out

class Block_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        #self.convd1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same", dilation=2)
        #self.convd2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same", dilation=3)
        self.trans = Transformer(out_channels, att_heads, dpr)
    def forward(self, x, skip):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.upsample(x1)
        x1 = F.relu(self.conv1(x1))
        x1 = torch.cat((skip, x1), axis=1)
        x1 = F.relu(self.conv2(x1))
        x1 = F.dropout(x1, 0.3)
        #x1 = F.relu(self.conv3(x1))
        #x2=F.relu(self.convd1(x1))
        #x3=F.relu(self.convd2(x2))
        #x4=torch.add(x1, x2)
        #x1=torch.add(x4, x3)
        out = self.trans(x1)
        return out
class Block_decoder1(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        #self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.trans = Transformer(out_channels, att_heads, dpr)

    def forward(self, x, skip):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear')
        x1 = F.relu(self.conv1(x1))
        x1 = torch.cat((skip, x1), axis=1)
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = F.dropout(x1, 0.3)
        out = self.trans(x1)
        return out


class DS_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, padding="same"),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, padding="same"),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, padding="same"),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        out = self.conv3(x1)

        return out


class Block_encoder_bottleneck(nn.Module):
    def __init__(self, blk, in_channels, out_channels, att_heads, dpr):
        super().__init__()
        self.blk = blk
        if ((self.blk == "first") or (self.blk == "bottleneck")):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.trans = Transformer(out_channels, att_heads, dpr)

        elif ((self.blk == "second") or (self.blk == "third") or (self.blk == "fourth")):
            self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
            self.conv1 = nn.Conv2d(1, in_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.trans = Transformer(out_channels, att_heads, dpr)

    def forward(self, x, scale_img="none"):
        if ((self.blk == "first") or (self.blk == "bottleneck")):
            x1 = x.permute(0, 2, 3, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 3, 1, 2)
            x1 = F.relu(self.conv1(x1))
            x1 = F.relu(self.conv2(x1))
            x1 = F.dropout(x1, 0.3)
            x1 = F.max_pool2d(x1, (2, 2))
            out = self.trans(x1)
            # without skip
        elif ((self.blk == "second") or (self.blk == "third") or (self.blk == "fourth")):
            x1 = x.permute(0, 2, 3, 1)
            x1 = self.layernorm(x1)
            x1 = x1.permute(0, 3, 1, 2)
            x1 = torch.cat((F.relu(self.conv1(scale_img)), x1), axis=1)
            x1 = F.relu(self.conv2(x1))
            x1 = F.relu(self.conv3(x1))
            x1 = F.dropout(x1, 0.3)
            x1 = F.max_pool2d(x1, (2, 2))
            out = self.trans(x1)
            # with skip
        return out

class CAM(nn.Module):
    def __init__(self, args):
        super().__init__()

        # attention heads and filters per block
        att_heads = [2, 4, 8, 12, 16, 12, 8, 4, 2]
        filters = [96, 192, 384, 768, 768*2, 768, 384, 192, 96]

        # number of blocks used in the model
        blocks = len(filters)

        stochastic_depth_rate = 1.0

        # probability for each block
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]

        self.drp_out = 0.3

        # shape
        init_sizes = torch.ones((2, 224, 224, 1))
        init_sizes = init_sizes.permute(0, 3, 1, 2)

        # Multi-scale input
        self.scale_img = nn.AvgPool2d(2, 2)

        # model
        self.block_5 = Block_encoder_bottleneck("bottleneck", filters[3], filters[4], att_heads[4], dpr[4])
        self.block_6 = Block_decoder(filters[4], filters[5], att_heads[5], dpr[5])
        self.block_7 = Block_decoder(filters[5], filters[6], att_heads[6], dpr[6])
        self.block_8 = Block_decoder(filters[6], filters[7], att_heads[7], dpr[7])
        self.block_9 = Block_decoder(filters[7], filters[8], att_heads[8], dpr[8])

    def forward(self, skip1, skip2, skip3, skip4):
        x = self.block_5(skip4)
        #print(x.shape)
        # Multi-scale input
        x = self.block_6(x, skip4)
        #print(f"Block 6 out -> {list(x.size())}")
        out4 = x
        x = self.block_7(x, skip3)
        #print(f"Block 7 out -> {list(x.size())}")
        out3 = x
        x = self.block_8(x, skip2)
        #print(f"Block 8 out -> {list(x.size())}")
        out2 = x
        x = self.block_9(x, skip1)
        #print(f"Block 9 out -> {list(x.size())}")
        out1 = x
        return out4, out3, out2, out1
class FCT1(nn.Module):
    def __init__(self, args):
        super().__init__()

        # attention heads and filters per block
        att_heads = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        filters = [96, 192, 384, 768, 768*2, 768, 384, 192, 96]

        # number of blocks used in the model
        blocks = len(filters)

        stochastic_depth_rate = 0.0

        # probability for each block
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]

        self.drp_out = 0.3

        # shape
        init_sizes = torch.ones((2, 224, 224, 1))
        init_sizes = init_sizes.permute(0, 3, 1, 2)

        # Multi-scale input
        self.scale_img = nn.AvgPool2d(2, 2)

        # model
        self.block_5 = Block_encoder_bottleneck("bottleneck", filters[3], filters[4], att_heads[4], dpr[4])
        self.block_6 = Block_decoder1(filters[4], filters[5], att_heads[5], dpr[5])
        self.block_7 = Block_decoder1(filters[5], filters[6], att_heads[6], dpr[6])
        self.block_8 = Block_decoder1(filters[6], filters[7], att_heads[7], dpr[7])
        self.block_9 = Block_decoder1(filters[7], filters[8], att_heads[8], dpr[8])

    def forward(self, skip1, skip2, skip3, skip4):
        x = self.block_5(skip4)

        # Multi-scale input
        x = self.block_6(x, skip4)
        #print(f"Block 6 out -> {list(x.size())}")
        out4 = x
        x = self.block_7(x, skip3)
        #print(f"Block 7 out -> {list(x.size())}")
        out3 = x
        x = self.block_8(x, skip2)
        #print(f"Block 8 out -> {list(x.size())}")
        out2 = x
        x = self.block_9(x, skip1)
        #print(f"Block 9 out -> {list(x.size())}")
        out1 = x
        return out4, out3, out2, out1
class FCT2(nn.Module):
    def __init__(self, args):
        super().__init__()

        # attention heads and filters per block
        att_heads = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        filters = [32, 64, 128, 256, 512]

        # number of blocks used in the model
        blocks = len(filters)

        stochastic_depth_rate = 0.0

        # probability for each block
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]
        self.drp_out = 0.3

        # shape
        init_sizes = torch.ones((2, 224, 224, 1))
        init_sizes = init_sizes.permute(0, 3, 1, 2)

        # Multi-scale input
        self.scale_img = nn.AvgPool2d(2, 2)

        # model
        self.block_5 = Block_encoder_bottleneck("bottleneck", filters[3], filters[4], att_heads[4], dpr[0])
        self.block_6 = Block_decoder(filters[4], filters[3], att_heads[5], dpr[1])
        self.block_7 = Block_decoder(filters[3], filters[2], att_heads[6], dpr[2])
        self.block_8 = Block_decoder(filters[2], filters[1], att_heads[7], dpr[3])
        self.block_9 = Block_decoder(filters[1], filters[0], att_heads[8], dpr[4])
        self.conv0 = nn.Conv2d(3, 16, 3, 2, padding=1)
        self.conv01 = nn.Conv2d(16, 32, 3, 2, padding=1)
        self.conv1 = nn.Conv2d(32, 64, 3, 2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, padding=1)
    def forward(self, x):
        x0 = F.relu(self.conv01(F.relu(self.conv0(x))))
        #print(f"X0={x0.shape}")
        x1=F.relu(self.conv1(x0))
        #print(f"X1={x1.shape}")
        x2=F.relu(self.conv2(x1))
        #print(f"X2={x2.shape}")
        x3=F.relu(self.conv3(x2))
        #print(f"X3={x3.shape}")
        x4 = self.block_5(x3)

        #print(f"X4={x4.shape}")
        # Multi-scale input
        x5 = self.block_6(x4, x3)
        #print(f"X5={x5.shape}")


        x6 = self.block_7(x5, x2)

        #print(f"X6={x6.shape}")
        x7 = self.block_8(x6, x1)
        #print(f"X7={x7.shape}")

        x8 = self.block_9(x7, x0)
        #print(f"X8={x8.shape}")

        return  x5, x6, x7, x8
if __name__ == '__main__':
    import torch, gc
    import os
    from thop import profile

    gc.collect()
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    args = "sss"
    model = FCT(args).cuda()
    s1 = torch.randn(1, 96, 64, 64).cuda()
    s2 = torch.randn(1, 192, 32, 32).cuda()
    s3 = torch.randn(1, 384, 16, 16).cuda()
    s4 = torch.randn(1, 768, 8, 8).cuda()
    macs, params = profile(model, inputs=(s1, s2, s3, s4))
    print(f"Macs={macs} and Params={params}")
    P = model(s1, s2, s3, s4)