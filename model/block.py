import torch.nn as nn
from collections import OrderedDict
import torch


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)

#定义激活函数
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

#计算特征图 F每个通道的均值---主要作用：对输入特征图F进行空间维度上的全局平均池化  输出形状(B, C, 1, 1)
def mean_channels(F):
    assert(F.dim() == 4)                                         # F(batch,c,H,W)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)    # 计算每个通道内的所有像素总和，对输入F在高度（H）和宽度（W）两个维度上进行求和，最终得到每个通道的总和。返回形状：(batch, c, 1, 1)
    return spatial_sum / (F.size(2) * F.size(3))                 # 计算每个通道的均值  每个通道的spatial_sum_ij/(H*W)

#计算特征图 F在通道维度上的标准差 --衡量通道内部特征的离散程度（对比度）  输出形状(B, C, 1, 1)
def stdv_channels(F):
    assert(F.dim() == 4)                     #确保输入是四维张量 (batch, channel, height, width)
    F_mean = mean_channels(F)                #得到每个通道的均值   结果形状：(batch, channel, 1, 1)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))    #形状(B, C, 1, 1)，计算每个通道的方差，表示该通道内所有像素值围绕均值的离散程度 
    return F_variance.pow(0.5)               #计算标准差  ---方差开平方，衡量通道内部特征的离散程度（对比度）  形状(B, C, 1, 1)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# contrast-aware channel attention module 
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

#信息多重蒸馏机制---通过分层提取和蒸馏特征，以增强特征表达能力
class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):                        #in_channels=64   distillation_rate=0.25蒸馏率
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)              #distilled_channels=64*1/4   每层提取 1/4 的特征用于输出
        self.remaining_channels = int(in_channels - self.distilled_channels)        #remaining_channels=64*3/4   其余 3/4 继续传递
        self.c1 = conv_layer(in_channels, in_channels, 3)                           #3*3卷积  对整个输入特征进行第一次提取             in(64,H,W),out(64,H,W)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)               #3*3卷积  处理 c1 余下的3/4特征,                  in(64*3/4,H,W),out(64,H,W)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)               #3*3卷积  处理 c2 余下的3/4特征,                  in(64*3/4,H,W),out(64,H,W)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)   #3*3卷积  处理 c3 余下的3/4特征, 且输出为64*1/4    in(64*3/4,H,W),out(64*1/4,H,W)
        self.act = activation('lrelu', neg_slope=0.05)                       #调用自定义激活函数
        self.c5 = conv_layer(in_channels, in_channels, 1)                           #1*1卷积  in(64,H,W),out(64,H,W)
        self.cca = CCALayer(self.distilled_channels * 4)  

    def forward(self, input):
     #第1级  卷积+蒸馏（拆分）
        out_c1 = self.act(self.c1(input))                  # 先3*3卷积+再lrelu激活   in:64,out:64
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)   #拆分out_c1通道 ,16个通道作为蒸馏特征,48个继续传递
     #第2级  卷积+蒸馏（拆分）
        out_c2 = self.act(self.c2(remaining_c1))           #                        in:48,out:64
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)   #拆分out_c2通道 ,16个通道作为蒸馏特征,48个继续传递
     #第3级  卷积+蒸馏（拆分）   
        out_c3 = self.act(self.c3(remaining_c2))           #                        in:48,out:64
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)   #拆分out_c3通道 ,16个通道作为蒸馏特征,48个继续传递
     #第4级  卷积
        out_c4 = self.c4(remaining_c3)                     #                        in:48,out:16
     #特征拼接  concat
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)      #in:16+16+16+16, out:64，恢复到in_channels
     #
        out_fused = self.c5(self.cca(out)) + input         #1*1卷积                       
        return out_fused

class IMDModule_speed(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule_speed, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.distilled_channels * 4, in_channels, 1)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(out) + input
        return out_fused

class IMDModule_Large(nn.Module):
    def __init__(self, in_channels, distillation_rate=1/4):
        super(IMDModule_Large, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)  # 6
        self.remaining_channels = int(in_channels - self.distilled_channels)  # 18
        self.c1 = conv_layer(in_channels, in_channels, 3, bias=False)  # 24 --> 24
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3, bias=False)  # 18 --> 24
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3, bias=False)  # 18 --> 24
        self.c4 = conv_layer(self.remaining_channels, self.remaining_channels, 3, bias=False)  # 15 --> 15
        self.c5 = conv_layer(self.remaining_channels-self.distilled_channels, self.remaining_channels-self.distilled_channels, 3, bias=False)  # 10 --> 10
        self.c6 = conv_layer(self.distilled_channels, self.distilled_channels, 3, bias=False)  # 5 --> 5
        self.act = activation('relu')
        self.c7 = conv_layer(self.distilled_channels * 6, in_channels, 1, bias=False)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))  # 24 --> 24
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1) # 6, 18
        out_c2 = self.act(self.c2(remaining_c1))  #  18 --> 24
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)  # 6, 18
        out_c3 = self.act(self.c3(remaining_c2))  # 18 --> 24
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)  # 6, 18
        out_c4 = self.act(self.c4(remaining_c3))  # 18 --> 18
        distilled_c4, remaining_c4 = torch.split(out_c4, (self.distilled_channels, self.remaining_channels-self.distilled_channels), dim=1)  # 6, 12
        out_c5 = self.act(self.c5(remaining_c4))  # 12 --> 12
        distilled_c5, remaining_c5 = torch.split(out_c5, (self.distilled_channels, self.remaining_channels-self.distilled_channels*2), dim=1)  # 6, 6
        out_c6 = self.act(self.c6(remaining_c5))  # 6 --> 6

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4, distilled_c5, out_c6], dim=1)
        out_fused = self.c7(out) + input
        return out_fused
    
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)
