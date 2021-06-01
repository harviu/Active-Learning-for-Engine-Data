# Copyright 2019 The InSituNet Authors. All rights reserved.
# Use of this source code is governed by a MIT-style license that can be
# found in the LICENSE file.

# Discriminator architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

from resblock import FirstBlockDiscriminator, BasicBlockDiscriminator

class Discriminator(nn.Module):
  def __init__(self, dsp=1, dspe=512, ch=64):
    super(Discriminator, self).__init__()

    self.dsp, self.dspe = dsp, dspe
    self.ch = ch

    # simulation parameters subnet
    self.sparams_subnet = nn.Sequential(
      nn.Linear(dsp, dspe), nn.ReLU(),
      nn.Linear(dspe, dspe), nn.ReLU()
    )

    # merged parameters subnet
    self.mparams_subnet = nn.Sequential(
      nn.Linear(dspe, ch * 16),
      nn.ReLU()
    )

    # image classification subnet
    self.img_subnet = nn.Sequential(
      FirstBlockDiscriminator(1, ch, kernel_size=3,
                              stride=1, padding=1), #(128*512)
      BasicBlockDiscriminator(ch, ch * 2, kernel_size=3,
                              stride=1, padding=1), #(64*256)
      BasicBlockDiscriminator(ch * 2, ch * 4, kernel_size=3,
                              stride=1, padding=1), #(32*128)
      BasicBlockDiscriminator(ch * 4, ch * 8, kernel_size=3,
                              stride=1, padding=1), #(16*64)
      BasicBlockDiscriminator(ch * 8, ch * 8, kernel_size=3,
                              stride=1, padding=1), #(8*32)
      BasicBlockDiscriminator(ch * 8, ch * 16, kernel_size=3,
                              stride=1, padding=1), #(4*16)
      BasicBlockDiscriminator(ch * 16, ch * 16, kernel_size=3,
                              stride=1, padding=1), #(2*8)
      nn.ReLU()
    )

    # output subnets
    self.out_subnet = nn.Sequential(
      nn.Linear(ch * 16, 1)
    )

  def forward(self, sp, x):
    sp = self.sparams_subnet(sp)

    mp = self.mparams_subnet(sp)

    x = self.img_subnet(x)
    x = torch.sum(x, (2, 3))

    out = self.out_subnet(x)
    out += torch.sum(mp * x, 1, keepdim=True)

    return out
