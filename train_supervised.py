
import time

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from numpy import count_nonzero

from agent import Agent
from environment import Environment
from data import MSCOCO
from settings import *


# for testing
# from agent import TopDownModel
# import numpy as np
# import torch.nn.functional as F
# policy = TopDownModel().cuda()
# policy.load_state_dict(
#     torch.load('../models/0505-1404-E49')['model_state_dict'])


def forward(img_features, captions):
    indeces, state, lstm_states = env.reset(
        img_features, captions)
    raw_loss = 0.0

    for i in range(indeces.shape[1]):
        gt_indeces = indeces[:, i]  # gt = ground truth
        word_logits, lstm_states = agent.forward(
            state, lstm_states)

        raw_loss += cross_entropy(input=word_logits,
                                  target=gt_indeces.cuda(),
                                  reduction='sum', ignore_index=0)

        # probs = F.softmax(word_logits, dim=1).detach().cpu().numpy()
        # top_5 = np.argsort(probs)[0][-5:]
        # word = env.probs_to_word(probs, 'greedy')
        # print('GT word: ', env.vocabulary[gt_indeces[0]],)
        # print('Pred word: ', word)
        # print('Top 5 words: ', env.vocabulary[top_5])
        # print('With P: ', probs[0][top_5])

        # Update state
        state['language_lstm_h'] = lstm_states['language_h']
        # Teacher forcing
        state['prev_word_indeces'] = torch.LongTensor(gt_indeces).cuda()

    # mean batch loss
    return raw_loss / count_nonzero(indeces)


RUN_IDENTIFIER = time.strftime('%m%d-%H%M-E')
agent = Agent()
# agent.actor = policy
train_loader = DataLoader(MSCOCO('train'),
                          batch_size=BATCH_SIZE, shuffle=SHUFFLE)
val_loader = DataLoader(MSCOCO('val'),
                        batch_size=BATCH_SIZE, shuffle=SHUFFLE)
env = Environment()

print('\nRUN #', RUN_IDENTIFIER[:-2])
print('BATCH SIZE: ', BATCH_SIZE)
print('LEARNING RATE: ', LEARNING_RATE)
print('TOTAL BATCHES: ', len(train_loader), '\n')

agent.actor.train()
for e in range(MAX_EPOCH):
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
        if (i + 1) % 300 == 0:
            print('Tr loss 300 batches: ', in_epoch_loss)
            in_epoch_loss = 0.0

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
