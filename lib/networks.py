import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

import logging

from scipy import ndimage

from .decoders import CASCADE_Add, CASCADE_Cat
#from .new_decoder import FCT, FCT1, FCT2
from .MIST import CAM
from .maxxvit_4out import maxvit_tiny_rw_224 as maxvit_tiny_rw_224_4out
from .maxxvit_4out import maxvit_rmlp_tiny_rw_256 as maxvit_rmlp_tiny_rw_256_4out
from .maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
from .maxxvit_4out import maxvit_rmlp_small_rw_224 as maxvit_rmlp_small_rw_224_4out
logger = logging.getLogger(__name__)
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def load_pretrained_weights(img_size, model_scale):

    backbone = maxxvit_rmlp_small_rw_256_4out()
    print('Loading:', './pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
    state_dict = torch.load('./pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
    #state_dict = torch.load('./pretrained_pth/maxvit/trained_backbone.pth')
    backbone.load_state_dict(state_dict, strict=False)
    print('Pretrained weights loaded.')
    
    return backbone

class MIST_CAM(nn.Module):
    def __init__(self, n_class=1, img_size_s1=(256, 256), img_size_s2=(224, 224), model_scale='small',
                 decoder_aggregation='additive', interpolation='bilinear'):
        super(MIST_CAM, self).__init__()

        self.n_class = n_class
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.model_scale = model_scale
        self.decoder_aggregation = decoder_aggregation
        self.interpolation = interpolation

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        self.backbone1 = load_pretrained_weights(self.img_size_s1[0], self.model_scale)
        #self.backbone2 = load_pretrained_weights(self.img_size_s2[0], self.model_scale)

        if (self.model_scale == 'tiny'):
            self.channels = [512, 256, 128, 64]
        elif (self.model_scale == 'small'):
            self.channels = [768, 384, 192, 96]

        # shared decoder initialization
        if (self.decoder_aggregation == 'additive'):
            self.decoder = CASCADE_Add(channels=self.channels)
        elif (self.decoder_aggregation == 'concatenation'):
            self.decoder = CASCADE_Cat(channels=self.channels)
        else:
            sys.exit(
                "'" + self.decoder_aggregation + "' is not a valid decoder aggregation! Currently supported aggregations are 'additive' and 'concatenation'.")

        # Prediction heads initialization
        self.decoder=CAM("SSS")
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)

    def forward(self, x):

        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        # transformer backbone as encoder
        f1 = self.backbone1(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))
        #print([f1[3].shape,f1[2].shape,f1[1].shape,f1[0].shape])

        #f2 = self.backbone2(F.interpolate(x, size=self.img_size_s2, mode=self.interpolation))
        #print([f2[3].shape,f2[2].shape,f2[1].shape,f2[0].shape])

        # decoder
        x11_o, x12_o, x13_o, x14_o = self.decoder(f1[0], f1[1], f1[2], f1[3])
        #print([x11_o.shape, x12_o.shape, x13_o.shape, x14_o.shape])
        # print([p1.shape,p2.shape,p3.shape,p4.shape])
        # prediction heads
        p11 = self.out_head1(x11_o)
        p12 = self.out_head2(x12_o)
        p13 = self.out_head3(x13_o)
        p14 = self.out_head4(x14_o)

        p11 = F.interpolate(p11, scale_factor=32, mode=self.interpolation)
        p12 = F.interpolate(p12, scale_factor=16, mode=self.interpolation)
        p13 = F.interpolate(p13, scale_factor=8, mode=self.interpolation)
        p14 = F.interpolate(p14, scale_factor=4, mode=self.interpolation)

        # print([p1.shape, p2.shape, p3.shape, p4.shape])
        return p11, p12, p13, p14

        
if __name__ == '__main__':
    model = MIST_CAM().cuda()
    from ptflops import get_model_complexity_info
    macs, params=get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=False, verbose=True)
    print('{:<30} {:<8}'.format('Computational Complexity:', macs))
    print('{:<30} {:<8}'.format('Number of parameters:', params))
    input_tensor = torch.randn(1, 1, 256, 256).cuda()

    P = model(input_tensor)
    print(P[0].shape)


