import torch
import torch.nn as nn
import torch.functional as F

class AFD_semantic(nn.Module):
    '''
    Pay Attention to Features, Transfer Learn Faster CNNs
    https://openreview.net/pdf?id=ryxyCeHtPB
    '''

    def __init__(self, in_channels, att_f):
        super(AFD_semantic, self).__init__()
        mid_channels = int(in_channels * att_f)

        self.attention = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, 1, 1, bias=True)
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t, eps=1e-6):

        fm_t_pooled = self.avg_pool(fm_t)
        rho = self.attention(fm_t_pooled)
        rho = torch.sigmoid(rho.squeeze())
        rho = rho / torch.sum(rho, dim=1, keepdim=True)

        fm_s_norm = torch.norm(fm_s, dim=(2, 3), keepdim=True)
        fm_s = torch.div(fm_s, fm_s_norm + eps)
        fm_t_norm = torch.norm(fm_t, dim=(2, 3), keepdim=True)
        fm_t = torch.div(fm_t, fm_t_norm + eps)

        loss = rho * torch.pow(fm_s - fm_t, 2).mean(dim=(2, 3))
        loss = loss.sum(1).mean(0)

        return loss


class AFD_spatial(nn.Module):
    '''
    Pay Attention to Features, Transfer Learn Faster CNNs
    https://openreview.net/pdf?id=ryxyCeHtPB
    '''

    def __init__(self, in_channels):
        super(AFD_spatial, self).__init__()

        self.attention = nn.Sequential(*[
            nn.Conv2d(in_channels, 1, 3, 1, 1)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t, eps=1e-6):

        rho = self.attention(fm_t)
        rho = torch.sigmoid(rho)
        rho = rho / torch.sum(rho, dim=(2,3), keepdim=True)

        fm_s_norm = torch.norm(fm_s, dim=1, keepdim=True)
        fm_s = torch.div(fm_s, fm_s_norm + eps)
        fm_t_norm = torch.norm(fm_t, dim=1, keepdim=True)
        fm_t = torch.div(fm_t, fm_t_norm + eps)
        loss = rho * torch.pow(fm_s - fm_t, 2).mean(dim=1, keepdim=True)
        loss =torch.sum(loss,dim=(2,3)).mean(0)
        return loss

from zy_LSNet.mobilenetv2 import mobilenet_v2
class LSNet(nn.Module):
    def __init__(self):
        super(LSNet, self).__init__()
        # rgb,depth encode
        self.rgb_pretrained = mobilenet_v2()
        self.depth_pretrained = mobilenet_v2()

        # Upsample_model
        self.upsample1_g = nn.Sequential(nn.Conv2d(68, 34, 3, 1, 1, ), nn.BatchNorm2d(34), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample2_g = nn.Sequential(nn.Conv2d(104, 52, 3, 1, 1, ), nn.BatchNorm2d(52), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample3_g = nn.Sequential(nn.Conv2d(160, 80, 3, 1, 1, ), nn.BatchNorm2d(80), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample4_g = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1, ), nn.BatchNorm2d(128), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample5_g = nn.Sequential(nn.Conv2d(320, 160, 3, 1, 1, ), nn.BatchNorm2d(160), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))


        self.conv_g = nn.Conv2d(34, 1, 1)
        self.conv2_g = nn.Conv2d(52, 1, 1)
        self.conv3_g = nn.Conv2d(80, 1, 1)


        # Tips: speed test and params and more this part is not included.
        # please comment this part when involved.
        if self.training:
            self.AFD_semantic_5_R_T = AFD_semantic(320,0.0625)
            self.AFD_semantic_4_R_T = AFD_semantic(96,0.0625)
            self.AFD_semantic_3_R_T = AFD_semantic(32,0.0625)
            self.AFD_spatial_3_R_T = AFD_spatial(32)
            self.AFD_spatial_2_R_T = AFD_spatial(24)
            self.AFD_spatial_1_R_T = AFD_spatial(16)


    def forward(self, rgb, ti):
        # rgb
        A1, A2, A3, A4, A5 = self.rgb_pretrained(rgb)
        # ti
        A1_t, A2_t, A3_t, A4_t, A5_t = self.depth_pretrained(ti)

        F5 = A5_t + A5
        F4 = A4_t + A4
        F3 = A3_t + A3
        F2 = A2_t + A2
        F1 = A1_t + A1


        F5 = self.upsample5_g(F5)
        F4 = torch.cat((F4, F5), dim=1)
        F4 = self.upsample4_g(F4)
        F3 = torch.cat((F3, F4), dim=1)
        F3 = self.upsample3_g(F3)
        F2 = torch.cat((F2, F3), dim=1)
        F2 = self.upsample2_g(F2)
        F1 = torch.cat((F1, F2), dim=1)
        F1 = self.upsample1_g(F1)

        out = self.conv_g(F1)


        if self.training:
            out3 = self.conv3_g(F3)
            out2 = self.conv2_g(F2)
            loss_semantic_5_R_T = self.AFD_semantic_5_R_T(A5, A5_t.detach())
            loss_semantic_5_T_R = self.AFD_semantic_5_R_T(A5_t, A5.detach())
            loss_semantic_4_R_T = self.AFD_semantic_4_R_T(A4, A4_t.detach())
            loss_semantic_4_T_R = self.AFD_semantic_4_R_T(A4_t, A4.detach())
            loss_semantic_3_R_T = self.AFD_semantic_3_R_T(A3, A3_t.detach())
            loss_semantic_3_T_R = self.AFD_semantic_3_R_T(A3_t, A3.detach())
            loss_spatial_3_R_T = self.AFD_spatial_3_R_T(A3, A3_t.detach())
            loss_spatial_3_T_R = self.AFD_spatial_3_R_T(A3_t, A3.detach())
            loss_spatial_2_R_T = self.AFD_spatial_2_R_T(A2, A2_t.detach())
            loss_spatial_2_T_R = self.AFD_spatial_2_R_T(A2_t, A2.detach())
            loss_spatial_1_R_T = self.AFD_spatial_1_R_T(A1, A1_t.detach())
            loss_spatial_1_T_R = self.AFD_spatial_1_R_T(A1_t, A1.detach())
            loss_KD = loss_semantic_5_R_T + loss_semantic_5_T_R + \
                      loss_semantic_4_R_T + loss_semantic_4_T_R + \
                      loss_semantic_3_R_T + loss_semantic_3_T_R + \
                      loss_spatial_3_R_T + loss_spatial_3_T_R + \
                      loss_spatial_2_R_T + loss_spatial_2_T_R + \
                      loss_spatial_1_R_T + loss_spatial_1_T_R
            return out, out2, out3, loss_KD
        return out






