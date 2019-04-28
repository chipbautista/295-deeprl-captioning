"""
Does the training.

Some pointers for implementing A-C:
- https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/main.py
- https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from agent import Agent
from environment import Environment, ReplayMemory
from data import MSCOCO
from settings import *
print('Hello')


agent = Agent()

train_loader = DataLoader(MSCOCO('train'))
# val_loader = DataLoader(MSCOCO('val'))
env = Environment()
memory = ReplayMemory()

# 1 epoch only for now. Will enclose this in an outer loop later.
epoch_reward = 0.0
# Caption an N number of images and update the agent (actor and critic)
# after playing BATCH_SIZE number of steps.
for i, (img_features, mean_vector) in enumerate(train_loader):

    mean_vector = torch.Tensor(mean_vector)
    # define initial state
    pooled_img_features = torch.sum(img_features, 1) / IMAGE_FEATURE_REGIONS
    pooled_img_features = pooled_img_features.reshape(-1)
    prev_word_onehot = np.zeros(VOCABULARY_SIZE)
    prev_word_onehot[0] = 1  # Set Index 0 to <SOS>

    state = {
        'language_lstm_h': torch.Tensor(np.zeros(LSTM_HIDDEN_UNITS)).cuda(),  # (1000)
        'pooled_img_features': torch.Tensor(pooled_img_features).cuda(),  # (1, 2048)
        'prev_word_onehot': torch.Tensor(prev_word_onehot).cuda(),  # (10000,)
        'img_features': torch.Tensor(img_features).cuda()  # (1, 36, 2048)
    }

    action_history = [0]
    episode_reward = 0.0

    for t in range(MAX_WORDS):  # MAX_WORDS = max # of steps
        word_idx, language_lstm_h = agent.select_action(state)

        # Perform a_t and receive R_{t+1}
        action_history.append(word_idx)
        reward, done = env.calculate_reward(action_history, mean_vector)
        episode_reward += reward

        # build trajectory: S_t, a_t, R_{t+1}
        trajectory = [state, word_idx, reward]

        # update current state to S_{t+1}
        word_onehot = np.zeros(VOCABULARY_SIZE)
        word_onehot[word_idx] = 1
        state['prev_word_onehot'] = torch.Tensor(word_onehot).cuda()
        state['language_lstm_h'] = language_lstm_h

        # add S_{t+1} to trajectory
        trajectory.extend([state, done])
        memory.push(trajectory)

        if len(memory) == BATCH_SIZE:
            import pdb; pdb.set_trace()
            agent.update()
            memory.reset()

        if done:
            print('<EOS> triggered after {} steps.'.format(t))
            break

    print('Episode reward: {}'.format(episode_reward))
    epoch_reward += episode_reward

    if i % 50 == 0:
        # do logging here
        print('Finished 50 episodes.')
