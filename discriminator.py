import torch
import torch.nn as nn 
from torchvision import models

class VggDiscriminator(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        self.fcs = nn.Sequential(
                nn.Linear(512 * 16 * 16, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 1),
                nn.Sigmoid(),
        )

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        
        out_1 = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        out_2 = h_relu5.reshape(h_relu5.shape[0], -1)
        out_2=self.fcs(out_2)
        
        return out_1, out_2 


class StyleDiscriminator(nn.Module):

    def __init__(self, img_size=256, max_conv_dim=512):
        '''
            This module based on StyleEncoder module of StarGan V2.

        '''
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = nn.ModuleList()
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResidualBlock(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.module = blocks

        self.last = nn.Linear(max_conv_dim, 1)

    def forward(self, x):
        feature_maps = []
        out = x 
        for m in self.module:
            out = m(out)
            feature_maps.append(out)
        out = self.last(out)
        return feature_maps, nn.Sigmoid(out)

        