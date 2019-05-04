"""
"""
import time

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from agent import Agent
from environment import Environment
from data import MSCOCO
from settings import *


def forward(img_features, captions):
    batch_loss = 0.0
    indeces, state, lstm_states = env.reset(
        img_features, captions)
    for i in range(indeces.shape[1]):
        gt_indeces = indeces[:, i]  # gt = ground truth
        word_logits, lstm_states = agent.forward(
            state, lstm_states)

        # this just gets the probabilities of the ground truth words
        # for each sample in the batch.
        # gt_indeces_probabilities = torch.stack([
        #     word_probabilities[i][b_gt_indeces[i]]
        #     for i in range(curr_batch_size)
        # ])
        mean_loss = cross_entropy(input=word_logits,
                                  target=gt_indeces.cuda(),
                                  size_average=True, ignore_index=0)
        batch_loss += mean_loss
        # Update state
        state['language_lstm_h'] = lstm_states['language_h']
        # Teacher forcing
        state['prev_word_indeces'] = torch.LongTensor(gt_indeces).cuda()

    return batch_loss


RUN_IDENTIFIER = time.strftime('%m%d-%H%M-E')

agent = Agent()
train_loader = DataLoader(MSCOCO('train'),
                          batch_size=BATCH_SIZE, shuffle=SHUFFLE)
val_loader = DataLoader(MSCOCO('val'),
                        batch_size=BATCH_SIZE, shuffle=SHUFFLE)
env = Environment()

print('\nRUN #', RUN_IDENTIFIER[:-2])
print('BATCH SIZE: ', BATCH_SIZE)
print('LEARNING RATE: ', LEARNING_RATE)


for e in range(MAX_EPOCH):
    agent.actor.train()
    epoch_start = time.time()

    min_val_loss = 20000.0
    tr_epoch_loss = 0.0
    val_epoch_loss = 0.0

    in_epoch_loss = 0.0
    for i, (img_features, captions) in enumerate(train_loader):
        agent.actor_optim.zero_grad()
        batch_loss = forward(img_features, captions)
        batch_loss.backward()
        agent.actor_optim.step()

        tr_epoch_loss += batch_loss.item()
        in_epoch_loss += batch_loss.item()

        # prints out accumulated batch loss for sanity :(
        if (i + 1) % 200 == 0:
            print('Tr loss 200 batches: ', in_epoch_loss)
            in_epoch_loss = 0.0

    agent.actor.eval()
    with torch.no_grad():
        for img_features, captions in val_loader:
            batch_loss = forward(img_features, captions)
            val_epoch_loss += batch_loss.item()

    agent.actor_optim_scheduler.step()
    print('Epoch: {} Tr Loss: {:.2f} Val Loss: {:.2f}. {:.2f}s'.format(
        e + 1, tr_epoch_loss, val_epoch_loss, time.time() - epoch_start))

    if val_epoch_loss < min_val_loss:
        print('Lower validation loss achieved. Saving model.')
        torch.save({
            'epoch': e,
            'model_state_dict': agent.actor.state_dict(),
            'optimizer_state_dict': agent.actor_optim.state_dict(),
            'tr_loss': tr_epoch_loss,
            'val_loss': val_epoch_loss
        }, MODEL_DIR.format(RUN_IDENTIFIER + str(e)))
        min_val_loss = val_epoch_loss
