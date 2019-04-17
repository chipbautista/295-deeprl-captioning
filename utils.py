"""
Store utility functions here.

(listing them for now, might be better to move them into
classes later on.)
"""

import numpy as np

from bert_serving.client import BertClient
bert_client = BertClient()


def vec_to_sentence(vectors):
    """
    Should take in the predicted vectors (agent actions):
    An array of shape (15, x) has 15 words, and x is the embedding size.

    and convert them to words by looking for the word nearest
    to the chosen embedding point.
    """
    pass


def sentence_to_bert(sentence):
    return bert_client.encode([sentence])[0]


def calculate_reward(sentence_vec, mean_vec):
    """
    Basically distance. (or?)
    Choices: Euclidean or L1 or?
    """
    return np.linalg.norm(sentence_vec, mean_vec)  # L2
