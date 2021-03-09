"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
# from models import VGGEncoder, VGGDecoder

#####################################################3
class VGGEncoder(nn.Module):
    def __init__(self, level):
        super(VGGEncoder, self).__init__()
        self.level = level

        # 224 x 224
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)

        self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 226 x 226
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu1_1 = nn.ReLU(inplace=True)
        # 224 x 224

        if level < 2: return

        self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu1_2 = nn.ReLU(inplace=True)
        # 224 x 224
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 112 x 112

        self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu2_1 = nn.ReLU(inplace=True)
        # 112 x 112

        if level < 3: return

        self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu2_2 = nn.ReLU(inplace=True)
        # 112 x 112

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 56 x 56

        self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu3_1 = nn.ReLU(inplace=True)
        # 56 x 56

        if level < 4: return

        self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_2 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_3 = nn.ReLU(inplace=True)
        # 56 x 56

        self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_4 = nn.ReLU(inplace=True)
        # 56 x 56

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 28 x 28

        self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu4_1 = nn.ReLU(inplace=True)
        # 28 x 28

    def forward(self, x):
        out = self.conv0(x)

        out = self.pad1_1(out)
        out = self.conv1_1(out)
        out = self.relu1_1(out)

        if self.level < 2:
            return out

        out = self.pad1_2(out)
        out = self.conv1_2(out)
        pool1 = self.relu1_2(out)

        out, pool1_idx = self.maxpool1(pool1)

        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)

        if self.level < 3:
            return out, pool1_idx, pool1.size()

        out = self.pad2_2(out)
        out = self.conv2_2(out)
        pool2 = self.relu2_2(out)

        out, pool2_idx = self.maxpool2(pool2)

        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)

        if self.level < 4:
            return out, pool1_idx, pool1.size(), pool2_idx, pool2.size()

        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)

        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)

        out = self.pad3_4(out)
        out = self.conv3_4(out)
        pool3 = self.relu3_4(out)
        out, pool3_idx = self.maxpool3(pool3)

        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)

        return out, pool1_idx, pool1.size(), pool2_idx, pool2.size(), pool3_idx, pool3.size()

    def forward_multiple(self, x):
        out = self.conv0(x)

        out = self.pad1_1(out)
        out = self.conv1_1(out)
        out = self.relu1_1(out)

        if self.level < 2: return out

        out1 = out

        out = self.pad1_2(out)
        out = self.conv1_2(out)
        pool1 = self.relu1_2(out)

        out, pool1_idx = self.maxpool1(pool1)

        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)

        if self.level < 3: return out, out1

        out2 = out

        out = self.pad2_2(out)
        out = self.conv2_2(out)
        pool2 = self.relu2_2(out)

        out, pool2_idx = self.maxpool2(pool2)

        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)

        if self.level < 4: return out, out2, out1

        out3 = out

        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)

        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)

        out = self.pad3_4(out)
        out = self.conv3_4(out)
        pool3 = self.relu3_4(out)
        out, pool3_idx = self.maxpool3(pool3)

        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)

        return out, out3, out2, out1


class VGGDecoder(nn.Module):
    def __init__(self, level, ):
        super(VGGDecoder, self).__init__()
        self.level = level

        if level > 3:
            self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
            self.relu4_1 = nn.ReLU(inplace=True)
            # 28 x 28

            self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.pool_lee_3 = nn.UpsamplingNearest2d(scale_factor=2)
            # 56 x 56

            self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu3_4 = nn.ReLU(inplace=True)
            # 56 x 56

            self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu3_3 = nn.ReLU(inplace=True)
            # 56 x 56

            self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu3_2 = nn.ReLU(inplace=True)
            # 56 x 56

        if level > 2:
            self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
            self.relu3_1 = nn.ReLU(inplace=True)
            # 56 x 56

            self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.pool_lee_2 = nn.UpsamplingNearest2d(scale_factor=2)
            # 112 x 112

            self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
            self.relu2_2 = nn.ReLU(inplace=True)
            # 112 x 112

        if level > 1:
            self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
            self.relu2_1 = nn.ReLU(inplace=True)
            # 112 x 112

            self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
            self.pool_lee_1 = nn.UpsamplingNearest2d(scale_factor=2)
            # 224 x 224

            self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
            self.relu1_2 = nn.ReLU(inplace=True)
            # 224 x 224

        if level > 0:
            self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x, noise_scale, pool1_idx=None, pool1_size=None, pool2_idx=None, pool2_size=None, pool3_idx=None,
                pool3_size=None):
        out = x

        if self.level > 3:
            out = self.pad4_1(out)
            out = self.conv4_1(out)
            out_3 = self.relu4_1(out)
            out = self.unpool3(out_3, pool3_idx, output_size=pool3_size)

            noise = self.pool_lee_3(out_3)
            noise = noise-out
            noise = ( noise * torch.rand(noise.shape).cuda() ) * noise_scale

            out = out + noise

            out = self.pad3_4(out)
            out = self.conv3_4(out)
            out = self.relu3_4(out)

            out = self.pad3_3(out)
            out = self.conv3_3(out)
            out = self.relu3_3(out)

            out = self.pad3_2(out)
            out = self.conv3_2(out)
            out = self.relu3_2(out)

        if self.level > 2:
            out = self.pad3_1(out)
            out = self.conv3_1(out)
            out_2 = self.relu3_1(out)
            out = self.unpool2(out_2, pool2_idx, output_size=pool2_size)

            noise = self.pool_lee_2(out_2)
            noise = noise - out
            noise = (noise * torch.rand(noise.shape).cuda()) * noise_scale

            out = out + noise

            out = self.pad2_2(out)
            out = self.conv2_2(out)
            out = self.relu2_2(out)

        if self.level > 1:
            out = self.pad2_1(out)
            out = self.conv2_1(out)
            out_1 = self.relu2_1(out)
            out = self.unpool1(out_1, pool1_idx, output_size=pool1_size)

            # noise = self.pool_lee_1(out_1)
            # noise = noise - out
            # noise = ( noise * torch.rand(noise.shape).cuda() ) * self.noise_scale
            # out = out + noise

            out = self.pad1_2(out)
            out = self.conv1_2(out)
            out = self.relu1_2(out)

        if self.level > 0:
            out = self.pad1_1(out)
            out = self.conv1_1(out)

        return out
##################################################

class Sonar_noise_WCT(nn.Module):
    def __init__(self):
        super(Sonar_noise_WCT, self).__init__()
        self.e1 = VGGEncoder(1)
        self.d1 = VGGDecoder(1)
        self.e2 = VGGEncoder(2)
        self.d2 = VGGDecoder(2)
        self.e3 = VGGEncoder(3)
        self.d3 = VGGDecoder(3)
        self.e4 = VGGEncoder(4)
        self.d4 = VGGDecoder(4)
    
    def transform(self, cont_img, styl_img, noise_scale, cont_seg = None, styl_seg = None):

        self.__compute_label_info(cont_seg, styl_seg)

        sF4, sF3, sF2, sF1 = self.e4.forward_multiple(styl_img)

        cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3 = self.e4(cont_img)
        sF4 = sF4.data.squeeze(0)
        cF4 = cF4.data.squeeze(0)
        # print(cont_seg)
        csF4 = self.__feature_wct(cF4, sF4, cont_seg, styl_seg)
        Im4 = self.d4(csF4,noise_scale, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3)

        cF3, cpool_idx, cpool1, cpool_idx2, cpool2 = self.e3(Im4)
        sF3 = sF3.data.squeeze(0)
        cF3 = cF3.data.squeeze(0)
        csF3 = self.__feature_wct(cF3, sF3, cont_seg, styl_seg)
        Im3 = self.d3(csF3,noise_scale, cpool_idx, cpool1, cpool_idx2, cpool2)

        cF2, cpool_idx, cpool = self.e2(Im3)
        sF2 = sF2.data.squeeze(0)
        cF2 = cF2.data.squeeze(0)
        csF2 = self.__feature_wct(cF2, sF2, cont_seg, styl_seg)
        Im2 = self.d2(csF2,noise_scale, cpool_idx, cpool)

        cF1 = self.e1(Im2)
        sF1 = sF1.data.squeeze(0)
        cF1 = cF1.data.squeeze(0)
        csF1 = self.__feature_wct(cF1, sF1, cont_seg, styl_seg)
        Im1 = self.d1(csF1, noise_scale)
        return Im1

    def __compute_label_info(self, cont_seg, styl_seg):
        if cont_seg == None or styl_seg == None:
            return
        max_label = np.max(cont_seg) + 1
        self.label_set = np.unique(cont_seg)
        self.label_indicator = np.zeros(max_label)
        for l in self.label_set:
            # if l==0:
            #   continue
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            o_cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l)
            o_styl_mask = np.where(styl_seg.reshape(styl_seg.shape[0] * styl_seg.shape[1]) == l)
            self.label_indicator[l] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)

    def __feature_wct(self, cont_feat, styl_feat, cont_seg, styl_seg):
        cont_c, cont_h, cont_w = cont_feat.size(0), cont_feat.size(1), cont_feat.size(2)
        styl_c, styl_h, styl_w = styl_feat.size(0), styl_feat.size(1), styl_feat.size(2)
        cont_feat_view = cont_feat.view(cont_c, -1).clone()
        styl_feat_view = styl_feat.view(styl_c, -1).clone()

        if cont_seg == None or styl_seg == None:
            target_feature = self.__wct_core(cont_feat_view, styl_feat_view)
        else:
            target_feature = cont_feat.view(cont_c, -1).clone()
            if len(cont_seg.shape) == 2:
                t_cont_seg = np.asarray(Image.fromarray(cont_seg).resize((cont_w, cont_h), Image.NEAREST))
            else:
                t_cont_seg = np.asarray(Image.fromarray(cont_seg, mode='RGB').resize((cont_w, cont_h), Image.NEAREST))
            if len(styl_seg.shape) == 2:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg).resize((styl_w, styl_h), Image.NEAREST))
            else:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg, mode='RGB').resize((styl_w, styl_h), Image.NEAREST))

            for l in self.label_set:
                if self.label_indicator[l] == 0:
                    continue
                cont_mask = np.where(t_cont_seg.reshape(t_cont_seg.shape[0] * t_cont_seg.shape[1]) == l)
                styl_mask = np.where(t_styl_seg.reshape(t_styl_seg.shape[0] * t_styl_seg.shape[1]) == l)
                if cont_mask[0].size <= 0 or styl_mask[0].size <= 0:
                    continue

                cont_indi = torch.LongTensor(cont_mask[0])
                styl_indi = torch.LongTensor(styl_mask[0])
                if self.is_cuda:
                    cont_indi = cont_indi.cuda(0)
                    styl_indi = styl_indi.cuda(0)

                cFFG = torch.index_select(cont_feat_view, 1, cont_indi)
                sFFG = torch.index_select(styl_feat_view, 1, styl_indi)
                # print(len(cont_indi))
                # print(len(styl_indi))
                tmp_target_feature = self.__wct_core(cFFG, sFFG)
                # print(tmp_target_feature.size())
                if torch.__version__ >= "0.4.0":
                    # This seems to be a bug in PyTorch 0.4.0 to me.
                    new_target_feature = torch.transpose(target_feature, 1, 0)
                    new_target_feature.index_copy_(0, cont_indi, \
                            torch.transpose(tmp_target_feature,1,0))
                    target_feature = torch.transpose(new_target_feature, 1, 0)
                else:
                    target_feature.index_copy_(1, cont_indi, tmp_target_feature)

        target_feature = target_feature.view_as(cont_feat)
        ccsF = target_feature.float().unsqueeze(0)
        return ccsF
    
    def __wct_core(self, cont_feat, styl_feat):
        cFSize = cont_feat.size()
        c_mean = torch.mean(cont_feat, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cont_feat)
        cont_feat = cont_feat - c_mean
        
        iden = torch.eye(cFSize[0])  # .double()
        if self.is_cuda:
            iden = iden.cuda()
        
        contentConv = torch.mm(cont_feat, cont_feat.t()).div(cFSize[1] - 1) + iden
        # del iden
        c_u, c_e, c_v = torch.svd(contentConv, some=False)
        # c_e2, c_v = torch.eig(contentConv, True)
        # c_e = c_e2[:,0]
        
        k_c = cFSize[0]
        for i in range(cFSize[0] - 1, -1, -1):
            if c_e[i] >= 0.00001:
                k_c = i + 1
                break
        
        sFSize = styl_feat.size()
        s_mean = torch.mean(styl_feat, 1)
        styl_feat = styl_feat - s_mean.unsqueeze(1).expand_as(styl_feat)
        styleConv = torch.mm(styl_feat, styl_feat.t()).div(sFSize[1] - 1)
        s_u, s_e, s_v = torch.svd(styleConv, some=False)
        
        k_s = sFSize[0]
        for i in range(sFSize[0] - 1, -1, -1):
            if s_e[i] >= 0.00001:
                k_s = i + 1
                break
        
        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = torch.mm(step2, cont_feat)

        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, *input):
        pass