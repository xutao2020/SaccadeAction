from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

from .aggatt import AggAtt


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class MOC_Branch(nn.Module):
    def __init__(self, input_channel, arch, head_conv, branch_info, K):
        super(MOC_Branch, self).__init__()
        assert head_conv > 0
        wh_head_conv = 64 if arch == 'resnet' else head_conv

        self.hm = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['hm'], kernel_size=1, stride=1, padding=0, bias=True))
        self.hm[-1].bias.data.fill_(-2.19)

        self.mov = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['mov'], kernel_size=1, stride=1, padding=0, bias=True))
        fill_fc_weights(self.mov)

        self.wh = nn.Sequential(
            nn.Conv2d(input_channel, wh_head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(wh_head_conv, branch_info['wh'] // K, kernel_size=1, stride=1, padding=0, bias=True))
        fill_fc_weights(self.wh)

        self.cor_att = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['cor_att'] // K, kernel_size=1, stride=1, padding=0, bias=True))
        self.cor_att[-1].bias.data.fill_(-2.19)

        # self.cor_att = nn.Sequential(
        #     nn.Conv2d(input_channel, wh_head_conv, kernel_size=3, padding=1, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(wh_head_conv, branch_info['cor_att'] // K, kernel_size=1, stride=1, padding=0, bias=True))
        # self.cor_att[-1].bias.data.fill_(-2.19)

        self.agg_att = AggAtt(input_channel, wh_head_conv, stride=1, kernel_size=3, padding=1, final_channels=2)

    def forward(self, input_chunk):
        output = {}
        output_wh = []
        final_output_wh = []
        # output_cor_att=[]
        for feature in input_chunk:
            wh = self.wh(feature)
            output_wh.append(wh)
            agg_att = self.agg_att(feature, wh)
            final_output_wh.append(agg_att[:, :2] + wh.detach())
            # cor_att=self.cor_att(feature)
            # output_cor_att.append(cor_att)
        input_chunk = torch.cat(input_chunk, dim=1)
        output_wh = torch.cat(output_wh, dim=1)
        final_output_wh = torch.cat(final_output_wh, dim=1)
        # output_cor_att=torch.cat(output_cor_att,dim=1)
        output['hm'] = self.hm(input_chunk)
        output['mov'] = self.mov(input_chunk)
        output['cor_att'] = self.cor_att(input_chunk)
        # output['cor_att'] = output_cor_att
        output['wh'] = output_wh
        output['final_wh'] = final_output_wh
        return output
