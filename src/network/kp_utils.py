import torch
import torch.nn as nn

from .utils import convolution, residual
from ._cpools import TopPool, BottomPool, LeftPool, RightPool


class pool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(pool, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

        self.look_conv1 = convolution(3, dim, 128)
        self.look_conv2 = convolution(3, dim, 128)
        self.P1_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)
        self.P2_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)

    def forward(self, x):
        # pool 1
        look_conv1 = self.look_conv1(x)
        p1_conv1 = self.p1_conv1(x)
        look_right = self.pool2(look_conv1)
        P1_look_conv = self.P1_look_conv(p1_conv1 + look_right)
        pool1 = self.pool1(P1_look_conv)

        # pool 2
        look_conv2 = self.look_conv2(x)
        p2_conv1 = self.p2_conv1(x)
        look_down = self.pool1(look_conv2)
        P2_look_conv = self.P2_look_conv(p2_conv1 + look_down)
        pool2 = self.pool2(P2_look_conv)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1 = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2


class pool_cross(nn.Module):
    def __init__(self, dim, pool1, pool2, pool3, pool4):
        super(pool_cross, self).__init__()
        self.p1_conv1 = convolution(3, dim, 224)
        self.p2_conv1 = convolution(3, dim, 224)

        self.p_conv1 = nn.Conv2d(224, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()
        self.pool3 = pool3()
        self.pool4 = pool4()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1 = self.pool1(p1_conv1)
        pool1 = self.pool3(pool1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2 = self.pool2(p2_conv1)
        pool2 = self.pool4(pool2)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1 = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2


class tl_pool(pool):
    def __init__(self, dim):
        super(tl_pool, self).__init__(dim, TopPool, LeftPool)


class br_pool(pool):
    def __init__(self, dim):
        super(br_pool, self).__init__(dim, BottomPool, RightPool)


class center_pool(pool_cross):
    def __init__(self, dim):
        super(center_pool, self).__init__(dim, TopPool, LeftPool, BottomPool, RightPool)


def make_tl_layer(dim):
    return tl_pool(dim)


def make_br_layer(dim):
    return br_pool(dim)


def make_ct_layer(dim):
    return center_pool(dim)


class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2


def make_merge_layer(dim):
    return MergeUp()


def make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)


def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)


def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )


def make_inter_layer(dim):
    return residual(3, dim, dim)


def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)
