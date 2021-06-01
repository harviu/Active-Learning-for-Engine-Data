import torch
import torch.nn as nn
import torch.nn.functional as F

from resblock import BasicBlockGenerator

class Generator(nn.Module):
  def __init__(self, dsp=1, dspe=512, ch=32):
    super(Generator, self).__init__()

    self.dsp, self.dspe = dsp, dspe
    self.ch = ch

    # simulation parameters subnet
    self.sparams_subnet = nn.Sequential(
      nn.Linear(dsp, dspe), nn.ReLU(),
      nn.Linear(dspe, dspe), nn.ReLU()
    )

    # merged parameters subnet
    self.mparams_subnet = nn.Sequential(
      nn.Linear(dspe, ch * 16 * 4 * 16, bias=False)
    )

    # image generation subnet
    self.img_subnet = nn.Sequential(
      BasicBlockGenerator(ch * 16, ch * 16, kernel_size=3, stride=1, padding=1), # (8*32)
      BasicBlockGenerator(ch * 16, ch * 8, kernel_size=3, stride=1, padding=1), # (16*64)
      BasicBlockGenerator(ch * 8, ch * 8, kernel_size=3, stride=1, padding=1), # (32*128)
      BasicBlockGenerator(ch * 8, ch * 4, kernel_size=3, stride=1, padding=1), # (64*256)
      BasicBlockGenerator(ch * 4, ch * 2, kernel_size=3, stride=1, padding=1), # (128*512)
      BasicBlockGenerator(ch * 2, ch, kernel_size=3, stride=1, padding=1), # (256*1024)
    )

    # output channel subnet
    self.output_subnet = nn.Sequential(
      nn.BatchNorm2d(ch),
      nn.ReLU(),
      nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1),
      nn.Sigmoid()
    )

  def forward(self, sp):
    sp = self.sparams_subnet(sp)

    mp = self.mparams_subnet(sp)

    x = mp.view(mp.size(0), self.ch * 16, 4, 16)
    x = self.img_subnet(x)
    x = F.interpolate(x,(228,989),mode='bilinear')
    x = self.output_subnet(x)

    return x