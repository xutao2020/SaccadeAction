from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from thop import profile
from .branch import MOC_Branch
from .dla import MOC_DLA
from .resnet import MOC_ResNet

backbone = {
    'dla': MOC_DLA,
    'resnet': MOC_ResNet
}


class MOC_Net(nn.Module):
    def __init__(self, arch, num_layers, branch_info, head_conv, K, flip_test=False):
        super(MOC_Net, self).__init__()
        self.flip_test = flip_test
        self.K = K
        self.backbone = backbone[arch](num_layers)
        self.branch = MOC_Branch(self.backbone.output_channel, arch, head_conv, branch_info, K)

    def forward(self, input):
        if self.flip_test:
            assert (self.K == len(input) // 2)
            chunk1 = [self.backbone(input[i]) for i in range(self.K)]
            chunk2 = [self.backbone(input[i + self.K]) for i in range(self.K)]

            return [self.branch(chunk1), self.branch(chunk2)]
        else:
            chunk = [self.backbone(input[i]) for i in range(self.K)]
            backbone_flops, backbone_params = profile(self.backbone, (input[self.K-1],))
            print('backbone_flops: ', backbone_flops,'backbone_params: ', backbone_params)
            print('backbone_flops: %.2f G, backbone_params: %.2f M' % (backbone_flops/1000000000.0,backbone_params/1000000.0) )
            branch_flops, branch_params = profile(self.branch, (chunk,))
            print('branch_flops: ', branch_flops,'branch_params: ', branch_params)
            print('branch_flops: %.2f G, branch_params: %.2f M' % (branch_flops/1000000000.0,branch_params/1000000.0) )

            return [self.branch(chunk)]
