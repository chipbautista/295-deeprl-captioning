import random

import numpy as np
from torch import Tensor
from torch.nn.functional import cosine_similarity
from bert_serving.client import BertClient

from settings import *


bert_client = BertClient()


class Environment(object):
    def __init__(self):
        self.vocabulary = np.load('vocabulary.npy')

    def calculate_reward(self, action_history, mean_vector):
        """
        Basically distance. (or?)
        Choices: Euclidean or L1 or?
        """
        # sentence = self.actions_to_sentence(actions)
        done = False
        if action_history[-1] == 1:
            # if the last action is EOS, don't include that in decoding
            action_history = action_history[:-1]
            done = True
        if len(action_history) >= MAX_WORDS + 2:
            done = True

        sentence = ' '.join(
            [self.vocabulary[idx]
             for idx in action_history[1:]])  # [1:] to not include SOS
        predicted_vector = Tensor(bert_client.encode([sentence])[0].reshape(1, -1))
        similarity = cosine_similarity(predicted_vector, Tensor(mean_vector))
        print(sentence, similarity)
        # distances = np.linalg.norm(sentence_vectors, mean_vecs)  # L2

        return similarity, done


class ReplayMemory(object):
    def __init__(self):
        self.capacity = MEMORY_CAPACITY  # Num of trajectories to save
        self.batch_size = BATCH_SIZE
        self.buffer = []
        # self.position - idk what this is for, yet.

    def push(self, trajectory):
        if len(self.buffer) < self.capacity:
            self.buffer.append(trajectory)

    def pop_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        # Add code to not delete good episodes?
        self.buffer = []
