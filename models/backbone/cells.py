import torch
import torch.nn as nn

class Cell(nn.Module):
    def __init__(self, conv, in_channels, out_channels, double=False):
        super().__init__()
        self.conv_type = conv
        self.double = double
        self.conv_i1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, 
                               stride=1)
        self.bni1 = nn.InstanceNorm3d(in_channels, affine=True)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = Conv(self.conv_type, in_channels, out_channels)
        
        if self.double:
            self.conv_i2 = nn.Conv3d(in_channels, in_channels, kernel_size=1, 
                               stride=1)
            self.bni2 = nn.InstanceNorm3d(in_channels, affine=True)
            self.conv2 = Conv(self.conv_type, in_channels, out_channels)
        
        self.conv_f = nn.Conv3d(out_channels, out_channels, kernel_size=1, 
                               stride=1)  
        self.bnf = nn.InstanceNorm3d(out_channels, affine=True)
        
    def forward(self, x, y=None):
        x = self.conv_i1(x)
        x = self.bni1(x)
        x = self.relu(x)
        x = self.conv1(x)
        
        if self.double:
            y = self.conv_i2(y)
            y = self.bni2(y)
            y = self.relu(y)            
            y = self.conv2(y)
            
            x = x+y
   
        x = self.conv_f(x)
        x = self.bnf(x) 
        x = self.relu(x)
        
        return x
    
class Conv(nn.Module):
    def __init__(self, conv, in_channels, out_channels):
        super().__init__()
        self.conv_type = conv
        self.relu = nn.ReLU(inplace=True)
        if self.conv_type == 'conv2d':
            self.conv2d = nn.Conv3d(in_channels, out_channels, stride=1, 
                                    kernel_size=(3,3,1), padding=(1,1,0))
            self.bn2d = nn.InstanceNorm3d(out_channels, affine=True)
        
        elif self.conv_type == 'conv3d':
            self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                                   stride=1, padding=1)
            self.bn3d = nn.InstanceNorm3d(out_channels, affine=True)
            
        elif self.conv_type == 'convp3d':
            self.convp3d1 = nn.Conv3d(in_channels, out_channels, stride=1, 
                              kernel_size=(3,3,1), padding=(1,1,0))
            self.p3dbn1 = nn.InstanceNorm3d(out_channels, affine=True)
            self.convp3d2 = nn.Conv3d(out_channels, out_channels, stride=1, 
                                  kernel_size=(1,1,3), padding=(0,0,1))
            self.p3dbn2 = nn.InstanceNorm3d(out_channels, affine=True)
            
    def forward(self, x):            
        if self.conv_type == 'conv2d':
            x = self.conv2d(x)
            x = self.bn2d(x)
            x = self.relu(x)
        
        elif self.conv_type == 'conv3d':
            x = self.conv3d(x)
            x = self.bn3d(x)
            x = self.relu(x)
        
        elif self.conv_type == 'convp3d':
            x = self.convp3d1(x)
            x = self.p3dbn1(x) 
            x = self.convp3d2(x)
            x = self.p3dbn2(x)
            x = self.relu(x) 
   
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.trilinear = nn.Upsample(scale_factor=scale_factor)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.trilinear(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        return x
     
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                                stride=2, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        return x   
