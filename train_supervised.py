"""
"""
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import cross_entropy

import utils
from agent import Agent
# from environment import Environment, ReplayMemory
from data import MSCOCO_Supervised
from settings import *


def forward(b_img_features, b_captions):
    batch_loss = 0.0
    # will need to keep track of current batch size for the last iteration
    # of train_loader when it can't form the whole batch size anymore.
    curr_batch_size = b_img_features.shape[0]
    # just some preprocessing here
    b_indeces = [utils.caption_to_indeces(c, vocabulary)
                 for c in b_captions]
    b_indeces = pad_sequence(b_indeces, batch_first=True)
    b_pooled_img_features = [
        torch.sum(img_features, 0) / IMAGE_FEATURE_REGIONS
        for img_features in b_img_features
    ]

    # define initial state
    lstm_states = {
        'language_h': None,
        'language_c': None,
        'attention_h': None,
        'attention_c': None
    }

    state = {
        # shape (b, 1000)
        'language_lstm_h': torch.Tensor(
            np.repeat([np.zeros(LSTM_HIDDEN_UNITS)], curr_batch_size, 0)
        ).cuda(),
        # shape (b, 1, 2048) stack converts a list of tensors into a tensor
        'pooled_img_features': torch.stack(b_pooled_img_features).cuda(),
        # shape (b)
        'prev_word_indeces': torch.zeros(curr_batch_size,
                                         dtype=torch.int64).cuda(),
        # shape (b, 36, 2048)
        'img_features': torch.Tensor(b_img_features).cuda()
    }

    for i in range(b_indeces.shape[1]):
        b_gt_indeces = b_indeces[:, i]  # gt = ground truth

        word_logits, lstm_states = agent.select_action(
            state, lstm_states)

        # this just gets the probabilities of the ground truth words
        # for each sample in the batch.
        # gt_indeces_probabilities = torch.stack([
        #     word_probabilities[i][b_gt_indeces[i]]
        #     for i in range(curr_batch_size)
        # ])
        # mean_loss = torch.mean(-torch.log(gt_indeces_probabilities))
        mean_loss = cross_entropy(input=word_logits,
                                  target=b_gt_indeces.cuda(),
                                  size_average=True, ignore_index=0)
        batch_loss += mean_loss

        # Update state
        state['language_lstm_h'] = lstm_states['language_h']
        # Teacher forcing
        state['prev_word_indeces'] = torch.LongTensor(b_gt_indeces).cuda()

    return batch_loss


agent = Agent(mode='supervised')
vocabulary = np.load('vocabulary.npy')
train_loader = DataLoader(MSCOCO_Supervised('train'), batch_size=BATCH_SIZE,
                          shuffle=SHUFFLE, pin_memory=True)
val_loader = DataLoader(MSCOCO_Supervised('val'), batch_size=BATCH_SIZE,
                        shuffle=SHUFFLE, pin_memory=True)

# env = Environment()
# memory = ReplayMemory()

print('\nStarting training...')
print('BATCH SIZE: ', BATCH_SIZE)
print('LEARNING RATE: ', LEARNING_RATE)

for e in range(50):
    agent.actor.train()
    epoch_start = time.time()

    forward_total_time = 0
    backward_total_time = 0

    tr_epoch_loss = 0.0
    val_epoch_loss = 0.0
    for b_img_features, b_captions in train_loader:
        agent.actor_optim.zero_grad()

        batch_loss = forward(b_img_features, b_captions)
        batch_loss.backward()
        agent.actor_optim.step()
        tr_epoch_loss += batch_loss.item()

    agent.actor.eval()
    with torch.no_grad():
        for b_img_features, b_captions in val_loader:
            batch_loss = forward(b_img_features, b_captions)
            val_epoch_loss += batch_loss.item()

    print('Epoch: {} Tr Loss: {:.2f} Val Loss: {:.2f}. {:.2f}s'.format(
        e + 1, tr_epoch_loss, val_epoch_loss, time.time() - epoch_start))

torch.save({
    'epoch': e,
    'model_state_dict': agent.actor.state_dict(),
    'optimizer_state_dict': agent.actor_optim.state_dict(),
    'loss': tr_epoch_loss
}, '930pmwed')
