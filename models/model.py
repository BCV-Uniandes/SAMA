# -*- coding: utf-8 -*-

"""
Model PyTorch implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbone.lucas import AlignedXception


class Model(nn.Module):
    def __init__(self, backbone='lucas',
                 filters=[32, 64, 128, 256, 256, 512], pi=0.1, 
                 num_classes=1, btn_size=4, image=True, desc=True,
                 vocab_size=76, desc_output_size=512, final_mlp_list=[256], 
                 fusion_output_filters=512, num_filters_dyn=10, scale_shift=False, 
                 broadcast=False, dmn=False, drop3d=None, 
                 loc=True, s_e=False, 
                 fusion_pointwise=False):
        super().__init__()
        self.image = image
        self.desc = desc
        self.vocab_size = vocab_size
        self.desc_output_size = desc_output_size
        self.final_mlp_list = final_mlp_list
        self.fusion_output_filters = fusion_output_filters
        self.num_filters_dyn = num_filters_dyn
        self.scale_shift = scale_shift
        self.broadcast = broadcast
        self.dmn = dmn
        self.drop3d = drop3d
        self.loc = loc
        self.s_e = s_e
        self.fusion_pointwise = fusion_pointwise
        self.prior = -np.log((1-pi)/pi) #Last layer bias initialization prior
        self.backbone_name = backbone
        
        if not (image or desc):
            raise ValueError('At least one modality should be used')

        image_output_size = 0
        
        # Images
        if self.image:
            BatchNorm = nn.InstanceNorm3d
            image_output_size = filters[-1] * btn_size * btn_size * btn_size
            if self.backbone_name == 'lucas':
                self.backbone = AlignedXception(BatchNorm, filters)
            else:
                raise ValueError('Backbone not implemented')
        
        # Descriptor
        if self.desc:
            self.mlp_d = MLP(self.vocab_size, self.desc_output_size, [])
            self.scale_shift = Scale_and_shift()
        
        #Dynamic Multimodal Network (DMN)
        if self.dmn:
            self.adaptative_filter = nn.Linear(
            in_features=self.desc_output_size,
            out_features=(self.num_filters_dyn * (filters[-1] + (12*int(self.loc)))))
            nn.init.xavier_normal_(self.adaptative_filter.weight)
            if self.drop3d:
                self.dropout3d = nn.Dropout3d(p=self.drop3d)
            
        concat_output_channels = (filters[-1] +
                                  self.desc_output_size*int(self.broadcast) +
                                  self.num_filters_dyn* int(self.dmn) + 12*int(self.loc))
            
            
        #Squeeze-and-Excitation block   
        if self.s_e:
            self.se_layer = SEBlock(concat_output_channels, 
                                    reduction=16)
        
        if self.fusion_pointwise:
            self.pointwise = nn.Conv3d(concat_output_channels, 
                                          self.fusion_output_filters, 1) 
        else:
            self.fusion_output_filters = concat_output_channels
        
        if self.image:
            fusion_output_size = self.fusion_output_filters * btn_size * \
                btn_size * btn_size
        else:
            fusion_output_size = self.desc_output_size
        
        
        self.mlp_f = MLP(fusion_output_size, num_classes, 
                         self.final_mlp_list, p=[0, 0.2], prior=self.prior, 
                         last=True)
        
       
            
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y=None):

        if self.image:
            x = self.backbone(x)
                    
        if self.desc:
            if y is not None:
                concat = [x]
                y = self.relu(self.mlp_d(y))
                if self.scale_shift:
                    y = self.scale_shift(y)
                #Descriptor broadcasted vector
                if self.broadcast:
                    b = y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    b = b.expand(b.shape[0], b.shape[1], 
                                x.shape[-3], x.shape[-2], x.shape[-1])
                    concat.append(b)
                
                if self.loc:
                        N, _, D, H, W = x.shape
                        loc = generate_spatial_batch(D, H, W)
                        loc = loc.repeat([N, 1, 1, 1, 1])
                        concat.append(loc)
                        
                if self.dmn:
                    d_filters = self.adaptative_filter(y)
                    d_filters = torch.sigmoid(d_filters)
                    resp = []
                    for idx, _filter in enumerate(d_filters):
                        _filter = _filter.view(self.num_filters_dyn, 
                                                x.shape[1] + (12*int(self.loc)), 1, 1, 1)
                        if self.loc:
                            resp.append(F.conv3d(
                                input=torch.cat([x[idx], loc[idx]]).unsqueeze(0), 
                                weight=_filter))
                        else:
                            resp.append(F.conv3d(
                                input=x[idx].unsqueeze(0), 
                                weight=_filter))
                            
                    resp = torch.cat(resp)
                    if self.drop3d:
                        resp = self.dropout3d(resp)
                        
                    concat.append(resp)
                    
                x = torch.cat(concat, dim=1)
                
                if self.s_e:
                    x = self.se_layer(x)
                if self.fusion_pointwise:
                    x = self.pointwise(x)
            else:
                x = self.relu(self.mlp_d(x))
                if self.scale_shift:
                    x = self.scale_shift(x)
                    
        # Combination
        x = x.view(x.shape[0], -1)
        out = self.mlp_f(x)
    
        return out

class MLP(nn.Module):
  def __init__(self, in_dim, out_dim, hidden=[], p=[0], prior=None, last=False):
    super().__init__()
    in_dim = in_dim
    layers = []
    p = p * (len(hidden)+1)
    
    if len(hidden) > 0:
      for i, h_dim in enumerate(hidden):
        layers.append(nn.Dropout(p[i]))
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.ReLU())
        in_dim = h_dim
    
    layers.append(nn.Dropout(p[-1]))
    layers.append(nn.Linear(in_dim, out_dim))
    if not last:
        layers.append(nn.ReLU())
    
    self.mlp = nn.Sequential(*layers)
    if prior:
        nn.init.constant_(self.mlp[-1].bias, prior)
    
  def forward(self, x):
    return self.mlp(x)

class Scale_and_shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        return self.weight * x + self.bias

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,'
    https://arxiv.org/abs/1709.01507.
    Parameters:
    ----------
    channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    approx_sigmoid : bool, default False
        Whether to use approximated sigmoid function.
    activation : function, or str, or nn.Module
        Activation function or name of activation function.
    """
    def __init__(self,
                 channels,
                 reduction=16):
        super(SEBlock, self).__init__()
        mid_channels = channels // reduction

        self.pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.conv1 = nn.Conv3d(in_channels=channels,
                               out_channels=mid_channels, kernel_size=1, 
                               stride=1, groups=1, bias=False)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=mid_channels,
                               out_channels=channels, kernel_size=1, 
                               stride=1, groups=1, bias=False)
        nn.init.xavier_normal_(self.conv2.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x

def generate_spatial_batch(featmap_D, featmap_H, featmap_W):
    """Generate additional visual coordinates feature maps.
    Function taken from
    https://github.com/chenxi116/TF-phrasecut-public/blob/master/util/processing_tools.py#L5
    and slightly modified
    """
    spatial_batch_val = np.zeros(
        (1, 12, featmap_D, featmap_H, featmap_W), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            for d in range(featmap_D):
                xmin = w / featmap_W * 2 - 1
                xmax = (w + 1) / featmap_W * 2 - 1
                xctr = (xmin + xmax) / 2
                ymin = h / featmap_H * 2 - 1
                ymax = (h + 1) / featmap_H * 2 - 1
                yctr = (ymin + ymax) / 2
                zmin = d / featmap_D * 2 - 1
                zmax = (d + 1) / featmap_D * 2 - 1
                zctr = (zmin + zmax) / 2
                spatial_batch_val[0, :, d, h, w] = (
                    [xmin, ymin, zmin, 
                     xmax, ymax, zmax,
                     xctr, yctr, zctr,
                     1 / featmap_W, 1 / featmap_H, 1/featmap_D])
    return torch.from_numpy(spatial_batch_val).cuda()

if __name__ == "__main__":
    model = Model()
    model.eval()
    input = torch.rand(1, 1, 256, 256, 256)
    output = model(input)
    print(output.size())
        