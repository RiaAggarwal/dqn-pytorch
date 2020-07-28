# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 22:02:56 2020
@author: Yuance Li
"""

import numpy as np
import torch
import torch.nn as nn

from ConvLSTM import EncoderDecoderConvLSTM
from utils import create_array, generate_video


def testing(dataloader):
    model = EncoderDecoderConvLSTM(nf=64, in_chan=1)
    model.load_state_dict(torch.load('./model.pth'))
    criterion=nn.MSELoss()
    for batch in dataloader:
        x, y = batch[:, 0:4, :, :, :], batch[:, 4:, :, :, :].squeeze()
        y_hat = model(x, future_seq=4).squeeze().unsqueeze(2)
    return y_hat

def test_dataloader(batch_size):
    # TODO
    test_loader = torch.utils.data.DataLoader(
        dataset= ,# TODO
        batch_size=batch_size,
        shuffle=True)
    return test_loader