"""
Does the training.
"""
import numpy as np
import torch

from networks import LSTM, LSTM_MLP
from environment import Environment

# SETTINGS
SPLITS_DIR = '../data/karpathy_splits/{}.txt'
DISCOUNT = 0.99
LEARNING_RATE = 0.01

# Import image features extracted by Pete Anderson's work
# Should probably not load this into memory...
FEATURES_DIR = '../data/features/{}.npy'
# Also: mean sentence vectors
MEAN_VEC_DIR = '../data/mean_vectors/{}.npy'

# assuming actor and critic are not pretrained...
actor = LSTM()
actor_optim = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic = LSTM_MLP()
critic_optim = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)
env = Environment(max_words=20, splits_dir=SPLITS_DIR)

for i, (image_features, mean_sentence_vec) in enumerate(env.iter_images('train')):
    # what to do with batch size? do we need it?

    # follow standard actor-critic pseudocode
    actions = np.array([])
    action = actor(image_features, actions)
    while True:
        # calculate Q value of current state and action
        Q_value = critic(state=[image_features, actions],
                         action=action)

        # adding to the list means it's action is performed
        # this also becomes the new state
        actions.append(action)

        # assuming reward at every step
        reward, done = env.calculate_reward(actions, mean_sentence_vec)

        # sample next action
        action = actor(image_features.cuda(), actions.cuda())

        # update actor
        actor.update(Q_value, actor_optim)

        # compute TD error
        next_Q_value = critic(state=[image_features, actions],
                              action=action)
        td_error = reward + (DISCOUNT * next_Q_value) - Q_value

        # backprop critic
        critic.update(td_error, critic_optim)

        if done:
            break

    if i % 50 == 0:
        # do logging here
        print('Finished 50 episodes.')
