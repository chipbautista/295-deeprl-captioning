"""
"""
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
from agent import Agent
from environment import Environment, ReplayMemory
from data import MSCOCO_Supervised
from settings import *

vocabulary = np.load('vocabulary.npy')

agent = Agent(mode='supervised')

train_loader = DataLoader(MSCOCO_Supervised('train'))
val_loader = DataLoader(MSCOCO_Supervised('val'))

# env = Environment()
# memory = ReplayMemory()

agent.actor.train()
for e in range(10):
    epoch_start = time.time()
    epoch_loss = 0.0
    batch_loss = 0.0
    for i, (img_features, caption) in enumerate(train_loader):
        gt_words = caption[0].split(' ')  # ground truth
        try:
            gt_indeces = [utils.word_to_index(w, vocabulary) for w in gt_words]
        except IndexError:
            # Skip over this caption if a word is not in the dict.
            continue

        # define initial state
        pooled_img_features = utils.pool_img_features(img_features)
        # pooled_img_features = torch.sum(img_features, 1) / IMAGE_FEATURE_REGIONS
        # pooled_img_features = pooled_img_features.reshape(-1)

        for gt_index in [0] + gt_indeces:
            # for pre-training, always input the correct previous words
            prev_word_onehot = utils.encode_to_one_hot(gt_index)

            state = {
                'language_lstm_h': torch.Tensor(np.zeros(LSTM_HIDDEN_UNITS)).cuda(),  # (1000)
                'pooled_img_features': torch.Tensor(pooled_img_features).cuda(),  # (1, 2048)
                'prev_word_onehot': torch.Tensor(prev_word_onehot).cuda(),  # (10000,)
                'img_features': torch.Tensor(img_features).cuda()  # (1, 36, 2048)
            }

            # INSTEAD OF THE PREDICTED WORD,
            # The agent should output the probability of the correct next word.
            word_probabilities, language_lstm_h = agent.select_action(state)
            batch_loss += -torch.log(word_probabilities[gt_index])

            # Update state
            state['language_lstm_h'] = language_lstm_h

        if i % 64 == 0:
            epoch_loss += batch_loss.item()
            agent.supervised_update(batch_loss)
            batch_loss = 0.0

    val_loss = 0.0
    
    with torch.no_grad():
        agent.actor.eval()
        for (img_features, caption) in val_loader:
            gt_words = caption[0].split(' ')
            try:
                gt_indeces = [utils.word_to_index(w) for w in gt_words]
            except IndexError:
                continue

            pooled_img_features = utils.pool_img_features(img_features)
            for gt_index in [0] + gt_indeces:
                prev_word_onehot = utils.encode_to_one_hot(gt_index)
                state = {
                    'language_lstm_h': torch.Tensor(np.zeros(LSTM_HIDDEN_UNITS)).cuda(),  # (1000)
                    'pooled_img_features': torch.Tensor(pooled_img_features).cuda(),  # (1, 2048)
                    'prev_word_onehot': torch.Tensor(prev_word_onehot).cuda(),  # (10000,)
                    'img_features': torch.Tensor(img_features).cuda()  # (1, 36, 2048)
                }

                word_probabilities, language_lstm_h = agent.select_action(state)
                
                val_loss += -torch.log(word_probabilities[gt_index])
                state['language_lstm_h'] = language_lstm_h
    
    print('Epoch: {} Tr Loss: {} Val Loss: {}. {}s'.format(e, epoch_loss, val_loss, time.time() - epoch_start))

torch.save({
    'epoch': e,
    'model_state_dict': actor.agent.state_dict(),
    'optimizer_state_dict': actor.agent_optim.state_dict(),
    'loss': epoch_loss
})
