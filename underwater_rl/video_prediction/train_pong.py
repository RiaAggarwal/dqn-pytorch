# -*- coding: utf-8 -*-
"""
@author: Yuance Li
"""
import os
import logging
import time
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .ConvLSTM import EncoderDecoderConvLSTM
from .utils import create_array, generate_video

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def initial(store_dir):
    model = EncoderDecoderConvLSTM(nf=64, in_chan=1)
    path = os.path.join(store_dir, 'pred.pth.tar')
    torch.save(model.state_dict(), path)
    
def training(dataloader, store_dir, learning_rate, logger, num_epochs=30):
    # initialize
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = EncoderDecoderConvLSTM(nf=64, in_chan=1).to(device)
    path = os.path.join(store_dir, 'pred.pth.tar')
    criterion=nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # train
    model.load_state_dict(torch.load(path))
    training_loss = []
    logger.info(f'Started training prediction on {device}')
    for epoch in range(num_epochs):
        running_loss = 0
        start = time.time()
        for batch in dataloader:            # (10,8,84,84)
            batch = batch.unsqueeze(2).to(device)           # (b, t, c, h, w)  (10, 8, 1, 84, 84)
            x, y = batch[:, 0:4, :, :, :].float(), batch[:, 4:8, :, :, :].squeeze()
            # optimize step
            optimizer.zero_grad()
            y_hat = model(x, future_seq=4).squeeze()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            # loss
            running_loss  += loss.item()
        epoch_loss = running_loss / len(dataloader)
        training_loss.append(epoch_loss)
        end = time.time() - start
        if epoch % 50 == 0:
            torch.save(model.state_dict(), path)
        logger.info(f'video-prediction \t epoch: {epoch} \t loss: {epoch_loss} \t time: {end/60}min')
    logger.info('Finished training prediction')
    torch.save(model.state_dict(), path)
    return training_loss

def testing(dataloader, store_dir):
    # load
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = EncoderDecoderConvLSTM(nf=64, in_chan=1).to(device)
    path = os.path.join(store_dir, 'pred.pth.tar')
    model.load_state_dict(torch.load(path))
    # test
    criterion=nn.MSELoss()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.unsqueeze(2).to(device)
            x, y = batch[:, 0:4, :, :, :].float(), batch[:, 4:, :, :, :].squeeze()
            y_hat = model(x, future_seq=4).squeeze()
            testing_loss = criterion(y_hat, y)
            video_frames = create_array(y_hat, y)
            generate_video(video_array=video_frames, video_filename=store_dir+'/result.avi')
            break        # only evaluate one batch
    return testing_loss.cpu()

def train_dataloader(replay, batch_size=10):
    # put in optimize_model after getting state_batch from memory_sample
    transitions = replay.sample(200)
    batch = Transition(*zip(*transitions))
    state = torch.cat(batch.state)
    next_state = torch.cat(batch.next_state)
    train_data = torch.cat((state,next_state), dim=1)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=False)
    return train_loader

def test_dataloader(replay, batch_size=10):
    test_data = replay.sample(100)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False)
    return test_loader

def get_logger(store_dir):
    log_path = os.path.join(store_dir, 'pred.log')
    logger = logging.Logger('train_status', level=logging.DEBUG)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter('%(levelname)s\t%(message)s'))
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger