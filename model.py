import network as networks
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, opts, device):
        super(Model, self).__init__()

        # generators
        self.G_AB = networks.Generator(opts.img_channels, opts.num_features, opts.num_blocks).to(device)
        self.G_BA = networks.Generator(opts.img_channels, opts.num_features, opts.num_blocks).to(device)

        # discriminators
        self.D_A = networks.Discriminator(opts.img_channels).to(device)
        self.D_B = networks.Discriminator(opts.img_channels).to(device)

        # optimizers
        self.G_AB_optim = torch.optim.Adam(self.G_AB.parameters(), lr=opts.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.G_BA_optim = torch.optim.Adam(self.G_BA.parameters(), lr=opts.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.D_A_optim = torch.optim.SGD(self.D_A.parameters(), lr=opts.lr)
        self.D_B_optim = torch.optim.SGD(self.D_B.parameters(), lr=opts.lr)

        # Setup the loss function for training
        self.criterion_cyclic = torch.nn.L1Loss().to(device)
        self.criterion_identity = torch.nn.L1Loss().to(device)
        self.criterionL2 = torch.nn.MSELoss().to(device)

    def initialize(self):
        self.G_AB.apply(self.gaussian_weights_init)
        self.G_BA.apply(self.gaussian_weights_init)
        self.D_A.apply(self.gaussian_weights_init)
        self.D_B.apply(self.gaussian_weights_init)

    def gaussian_weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname.find('Conv') == 0:
            m.weight.data.normal_(0.0, 0.02)


