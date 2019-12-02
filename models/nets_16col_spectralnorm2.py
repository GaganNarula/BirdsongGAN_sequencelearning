import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from itertools import chain
from torch.nn.utils import spectral_norm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

AFF = True
TRACK_STATS = True

class _netG(nn.Module):
    def __init__(self, ngpu,nz,ngf,nc):
        super(_netG, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.ngpu = ngpu
        self.bn_fake = nn.ModuleList(
            [nn.BatchNorm2d(ngf * 8, affine = AFF, track_running_stats = TRACK_STATS),
            nn.BatchNorm2d(ngf * 4, affine = AFF, track_running_stats = TRACK_STATS),
            nn.BatchNorm2d(ngf * 2, affine = AFF, track_running_stats = TRACK_STATS),
            nn.BatchNorm2d(ngf, affine = AFF, track_running_stats = TRACK_STATS)])
            
        self.bn_rec = nn.ModuleList(
            [nn.BatchNorm2d(ngf * 8, affine = AFF, track_running_stats = TRACK_STATS), 
            nn.BatchNorm2d(ngf * 4, affine = AFF, track_running_stats = TRACK_STATS), 
            nn.BatchNorm2d(ngf * 2, affine = AFF, track_running_stats = TRACK_STATS), 
            nn.BatchNorm2d(ngf, affine = AFF, track_running_stats = TRACK_STATS)])
        self.convs = nn.ModuleList([nn.ConvTranspose2d(nz, ngf * 8, kernel_size=(3,3), stride=1, padding=0, bias=False),
                                    nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=(4, 4), stride=(3, 1),
                                                       padding=(1, 1), bias=False),
                                    nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=(5, 4), stride=(4, 1),
                                                       padding=(1, 1), bias=False),
                                    nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=(5, 4), stride=(2, 1), padding=(1, 0),
                                                       bias=False),
                                    nn.ConvTranspose2d(ngf, nc, kernel_size=(5, 4), stride=(2, 2), padding=(0,1),
                                                       bias=True)])
        self.activation = nn.Softplus()
        self.reconstruction = False

    def mode(self, reconstruction=False):
        self.reconstruction = reconstruction

    def forward(self, input):
        if self.reconstruction:
            bns = self.bn_rec
        else:
            bns = self.bn_fake
        x = self.convs[0](input.view(input.size(0), self.nz, 1, 1))
        for i in range(4):
            x = bns[i](x)
            x = nn.ReLU(True)(x)
            x = self.convs[i + 1](x)
        x = self.activation(x)
        return x


class _netE(nn.Module):
    def __init__(self, ngpu,nz,ngf,nc):
        super(_netE, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ngf, kernel_size=(5, 4), stride=(2, 2), padding=(0,1), bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf, affine = AFF, track_running_stats = TRACK_STATS),
            nn.Conv2d(ngf, ngf * 2, kernel_size=(5, 4), stride=(2, 1), padding=(1, 0), bias=False),

            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 2, affine = AFF, track_running_stats = TRACK_STATS),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=(5, 4), stride=(4, 1), padding=(1, 1), bias=False),

            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 4,affine = AFF, track_running_stats = TRACK_STATS),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=(4, 4), stride=(3, 1), padding=(1, 1), bias=False),

            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 8,affine = AFF, track_running_stats = TRACK_STATS),
            nn.Conv2d(ngf * 8, nz, kernel_size=(3,3), stride=1, padding=0, bias=False),
        )
        self.activation = nn.Sequential(
            nn.Linear(nz, nz)
        )

    def forward(self, input):
        output = self.main(input)
        output = output.view(output.size(0), -1)
        return self.activation(output)
    
    

class _netD(nn.Module):
    def __init__(self, ngpu,ndf,nc):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.convs = nn.ModuleList([
            spectral_norm(nn.Conv2d(nc, ndf, kernel_size=4, stride=(2,2), padding=(1,1), bias=False)),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=(4, 3), stride=(2, 1), padding=(1, 0), bias=False)),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=(4,3), stride=(2,1), padding=(2,0), bias=False)),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=(8,4), stride=(2,1), padding=1, bias=False)),
            spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=(6,3), stride=1, padding=0, bias=False)),
        ])
        self.activations = nn.ModuleList([nn.LeakyReLU(0.2),nn.LeakyReLU(0.2),
                                    nn.LeakyReLU(0.2),nn.LeakyReLU(0.2)])

    def forward(self, x):
        for i in range(4):
            x = self.convs[i](x)
            x = self.activations[i](x)
        x = self.convs[-1](x)
        x = nn.Sigmoid()(x)
        output = x.view(-1,1)
        return output

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)