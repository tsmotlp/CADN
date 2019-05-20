import torch
from torch import nn
# -------------------generator related----------------------------------------------------------------------
class Generator(nn.Module):
    """generator, including GAB, GBA"""
    def __init__(self, img_channels=1, num_features=128, num_blocks=3):
        super(Generator, self).__init__()
        # head block
        self.head = [Conv_BN_ReLU(img_channels, num_features)]
        for i in range(1, 3):
            self.head += [Conv_BN_ReLU(num_features, num_features)]
        self.head = nn.Sequential(*self.head) 
        # cascaded residual block
        self.block1 = Basic_Block(num_features, num_blocks)
        self.block2 = Basic_Block(num_features, num_blocks)
        self.block3 = Basic_Block(num_features, num_blocks)
        self.block4 = Basic_Block(num_features, num_blocks)
        self.block5 = Basic_Block(num_features, num_blocks)
        self.block6 = Basic_Block(num_features, num_blocks)
        # tail block
        self.tail = nn.Sequential(
                Conv_BN_ReLU(in_channels=7*num_features, out_channels=num_features),
                Conv_BN_ReLU(in_channels=num_features, out_channels=num_features),
                nn.Conv2d(in_channels=num_features, out_channels=img_channels, kernel_size=3, stride=1, padding=1))
    def forward(self, x):
        head_out = self.head(x)
        block1_out = self.block1(head_out)
        block2_out = self.block1(block1_out)
        block3_out = self.block1(block2_out)
        block4_out = self.block1(block3_out)
        block5_out = self.block1(block4_out)
        block6_out = self.block1(block5_out)
        concat = torch.cat(
                (head_out, block1_out, block2_out, block3_out, block4_out, block5_out, block6_out), dim=1)
        tail_out = self.tail(concat)
        out = x + tail_out
        return out
        
class Basic_Block(nn.Module):
    """basic module in the generator"""
    def __init__(self, num_features, num_modules):
        super(Basic_Block, self).__init__()
        self.model = []
        for i in range(num_modules):
            self.model += [Conv_BN_ReLU(num_features, num_features)]
        self.model = nn.Sequential(*self.model)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.model(x) + x
        out = self.relu(out)
        return out

class Conv_BN_ReLU(nn.Module):
    """a basic block consists of cascaded conv2d, BatchNorm2d and ReLU nonliearity"""
    def __init__(self, in_channels, out_channels):
        super(Conv_BN_ReLU, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
    def forward(self, x):
        return self.model(x)
# ------------------------------------------------------------------------------------------------------
        
# -------------------discriminator related-----------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
                Conv_BN_LReLU(in_channels=img_channels, out_channels=64, kernel_size=4, stride=2, padding=1, norm=False),
                Conv_BN_LReLU(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, norm=True),
                Conv_BN_LReLU(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, norm=True),
                Conv_BN_LReLU(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, norm=True),
            )
        self.fc = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1)
    def forward(self, x):
        out = self.block(x)
        out = self.fc(out)
        return out

class Conv_BN_LReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm=False):
        super(Conv_BN_LReLU, self).__init__()
        self.block = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if norm:
            self.block += [nn.BatchNorm2d(out_channels)]
        self.block += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        self.block = nn.Sequential(*self.block)
    def forward(self, x):
        out = self.block(x)
        return out
# ------------------------------------------------------------------------------------------------
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        