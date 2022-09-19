import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import model.resnet

class ImageEncoderNetwork(nn.Module):
    def __init__(self, latent_dim=256, img_res=256):
        super().__init__()

        self.features = model.resnet.resnet18(pretrained=True)
        #self.features = resnet.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        self.latent_fc1 = nn.Linear(512, latent_dim)
        self.latent_fc2 = nn.Linear(latent_dim, latent_dim)

        self.res = 128
        self.number_feautre = 64+64+128
        mode = 'bilinear'
        self.upsamplers = [nn.Upsample(scale_factor=4, mode=mode, align_corners=False),
                           nn.Upsample(scale_factor=4, mode=mode, align_corners=False),
                           nn.Upsample(scale_factor=8, mode=mode, align_corners=False)
                          ]
        self.upsamplers_progressive = nn.Upsample(size=[img_res,img_res], mode=mode, align_corners=False)
        self.conv1 = nn.Conv2d(self.number_feautre, 128, kernel_size=1)
        self.conv2 = nn.Conv2d(128, 1, kernel_size=1)

        torch.nn.init.xavier_uniform_(self.latent_fc1.weight)
        torch.nn.init.zeros_(self.latent_fc1.bias)
        torch.nn.init.xavier_uniform_(self.latent_fc2.weight)
        torch.nn.init.zeros_(self.latent_fc2.bias)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)

    def forward(self, x):

        feat_code, feat_maps = self.features.forward(x)
        bz = x.shape[0]
        #
        feat_maps_upsamples = torch.FloatTensor(bz, self.number_feautre, self.res, self.res).to(x.device)
        start_channel_index = 0
        for i in range(len(feat_maps)):
            len_channel = feat_maps[i].shape[1]
            feat_maps_upsamples[:, start_channel_index:start_channel_index + len_channel] = self.upsamplers[i](feat_maps[i])
            start_channel_index += len_channel
        feature_map = self.relu(self.conv1(feat_maps_upsamples))
        feature_map = self.conv2(feature_map)
        #
        feature_map = self.upsamplers_progressive(feature_map)
        feature_map = feature_map.view(feature_map.shape[0], feature_map.shape[1], -1).permute(0,2,1)
        #
        latent = self.relu(self.latent_fc1(feat_code))
        latent = self.latent_fc2(latent)

        latent = F.normalize(latent, p=2, dim=1)

        return feature_map, latent
