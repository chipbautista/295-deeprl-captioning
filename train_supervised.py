"""
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from agent import Agent
from environment import Environment, ReplayMemory
from data import MSCOCO_Supervised
from settings import *
print('Hello')


agent = Agent(mode='supervised')
agent.actor.train()

train_loader = DataLoader(MSCOCO_Supervised('train'))
# val_loader = DataLoader(MSCOCO('val'))
env = Environment()
memory = ReplayMemory()

for e in range(5):
    total_loss = 0.0
    for i, (img_features, caption) in enumerate(train_loader):
        gt_words = caption[0].split(' ')  # ground truth
        try:
            gt_indexs = [env.get_word_index(w) for w in gt_words]
        except IndexError:
            # Skip over this caption if a word is not in the dict.
            continue

        # define initial state
        pooled_img_features = torch.sum(img_features, 1) / IMAGE_FEATURE_REGIONS
        pooled_img_features = pooled_img_features.reshape(-1)

        for gt_index in gt_indexs:

            # for pre-training, always input the correct previous words
            prev_word_onehot = np.zeros(VOCABULARY_SIZE)
            prev_word_onehot[0] = gt_index

            state = {
                'language_lstm_h': torch.Tensor(np.zeros(LSTM_HIDDEN_UNITS)).cuda(),  # (1000)
                'pooled_img_features': torch.Tensor(pooled_img_features).cuda(),  # (1, 2048)
                'prev_word_onehot': torch.Tensor(prev_word_onehot).cuda(),  # (10000,)
                'img_features': torch.Tensor(img_features).cuda()  # (1, 36, 2048)
            }

            # INSTEAD OF THE PREDICTED WORD,
            # The agent should output the probability of the correct next word.
            word_probabilities, language_lstm_h = agent.select_action(state)
            total_loss += torch.log(word_probabilities[gt_index])

            # Update state
            state['language_lstm_h'] = language_lstm_h

    print('Loss: ', -total_loss.item())
    agent.supervised_update(-total_loss)
