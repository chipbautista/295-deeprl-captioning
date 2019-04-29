import torch
import numpy as np

from settings import *


def pool_img_features(img_features):
    pooled_img_features = torch.sum(img_features, 1) / IMAGE_FEATURE_REGIONS
    return pooled_img_features.reshape(-1)


def encode_to_one_hot(index):
    onehot = np.zeros(VOCABULARY_SIZE)
    onehot[index] = 1
    return onehot


def word_to_index(word, vocabulary):
    return np.where(vocabulary == word)[0][0]
