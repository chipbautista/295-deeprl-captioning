import random

import numpy as np
from bert_serving.client import BertClient

from settings import *


bert_client = BertClient()


class Environment(object):
    def __init__(self):
        pass

    def calculate_reward(self, actions, mean_vecs):  # TO REPAIR
        """
        Basically distance. (or?)
        Choices: Euclidean or L1 or?
        """
        # sentence = self.actions_to_sentence(actions)
        sentences = [' '.join(actions) for actions in batch_actions]
        sentence_vectors = bert_client.encode(sentences)
        distances = np.linalg.norm(sentence_vectors, mean_vecs)  # L2

        words = sentence.split()
        if len(words) == 20 or words[-1] == '<EOS>':
            return distance, True
        return distance, False

    def actions_to_sentence(self, actions):
        # use gensim here?
        sentence = None
        return sentence


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
