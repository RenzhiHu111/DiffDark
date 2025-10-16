import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)

class MultAttention(nn.Module):
    def __init__(self, nc, number):
        super(MultAttention, self).__init__()
        self.norm1 = Normalize(nc)
        self.conv1 = torch.nn.Conv2d(nc,
                                     number,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.conv2 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=3,dilation=3,bias=False)
        self.conv3 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=5,dilation=5,bias=False)
        self.conv4 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=7,dilation=7,bias=False)

        self.norm2 = Normalize(number*3)
        self.conv5 = nn.Conv2d(number*3,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv6 = nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.norm3 = Normalize(nc)
        self.conv7 = torch.nn.Conv2d(nc,
                                     number,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.conv8 = nn.Conv2d(number, nc, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x0 = x
        x = self.norm1(x)
        x = nonlinearity(x)
        x = self.conv1(x)

        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        se = torch.cat([x2, x3, x4], dim=1)

        se = self.norm2(se)
        se = nonlinearity(se)
        se = self.conv5(se)
        se = self.conv6(se)
        se = self.sigmoid(se)

        x01 = self.norm3(x0)
        x02 = nonlinearity(x01)
        x03 = self.conv7(x02)
        x04 = self.conv8(x03)

        return se * x04

def NonLinearity(inplace=False):
    return nn.SiLU(inplace)

####  Attention Fusion Block
class LAM_Module_v2(nn.Module):
    def __init__(self, in_dim, bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.feature_extractor1 = nn.Conv2d(3, self.chanel_in, kernel_size=3, stride=1, padding=1,
                                           bias=bias)
        self.feature_extractor2 = nn.Conv2d(3, self.chanel_in, kernel_size=3, stride=1, padding=1,
                                           bias=bias)
        self.feature_extractor3 = nn.Conv2d(3, self.chanel_in, kernel_size=3, stride=1, padding=1,
                                           bias=bias)
        self.feature_extractor4 = nn.Conv2d(3, self.chanel_in, kernel_size=3, stride=1, padding=1,
                                           bias=bias)

        self.qkv = nn.Conv2d(self.chanel_in*4,  self.chanel_in * 12, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*12, self.chanel_in*12, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*12, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in*4, self.chanel_in*4, kernel_size=1, bias=bias)
        self.activation = NonLinearity()
        self.project_out1 = nn.Conv2d(self.chanel_in*4, self.chanel_in*4, kernel_size=1, bias=bias)

    def forward(self, a, b, c, d):

        a = self.feature_extractor1(a)
        b = self.feature_extractor2(b)
        c = self.feature_extractor3(c)
        d = self.feature_extractor4(d)

        x = torch.cat([a.unsqueeze(1), b.unsqueeze(1), c.unsqueeze(1), d.unsqueeze(1)], dim=1)
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize, N*C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = self.activation(out_1)
        out_1 = self.project_out1(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1+x
        out = out.view(m_batchsize, -1, height, width)
        return out



