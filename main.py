"""
Does the training.
"""
import numpy as np
import torch

from topdown_model import TopDownModel
from environment import Environment
from data import MSCOCO
from settings import *


# assuming actor and critic are not pretrained...
actor = TopDownModel()
actor_optim = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic = LSTM_MLP()
critic_optim = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)

train_loader = torch.utils.data.DataLoader(MSCOCO('train'))
val_loader = torch.utils.data.DataLoader(MSCOCO('val'))
env = Environment()


# 1 epoch only for now. Will enclose this in an outer loop later.
for i, batch_features, batch_mean_vectors in enumerate(train_loader):
    batch_features = torch.FloatTensor(batch_features).cuda()
    batch_mean_vectors = torch.FloatTensor(batch_mean_vectors).cuda()

    # follow standard actor-critic pseudocode
    action_history = np.zeros(BATCH_SIZE, MAX_WORDS)
    actions = actor(batch_features, action_history)
    for t in range(MAX_WORDS):  # MAX_WORDS = max # of steps
        # calculate Q value of current state and action
        Q_value = critic(state=[batch_features, action_history],
                         action=actions)

        # adding to the list means action is performed
        action_history[:, t] = actions

        reward, done = env.calculate_reward(action_history, batch_mean_vectors)

        # sample next action
        actions = actor(image_features.cuda(), actions.cuda())

        # update actor
        actor.update(Q_value, actor_optim)

        # compute TD error
        next_Q_value = critic(state=[image_features, action_history],
                              action=actions)
        td_error = reward + (DISCOUNT * next_Q_value) - Q_value

        # backprop critic
        critic.update(td_error, critic_optim)

    if i % 50 == 0:
        # do logging here
        print('Finished 50 episodes.')
