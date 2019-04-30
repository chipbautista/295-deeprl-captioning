import torch
import numpy as np

from settings import *


# def pool_img_features(img_features, axis=0):
#     pooled_img_features = torch.sum(img_features, axis) / IMAGE_FEATURE_REGIONS
#     return pooled_img_features.reshape(-1)


# def pool_img_features(img_features):
#     return torch.sum(img_features, 0) / IMAGE_FEATURE_REGIONS


def encode_to_one_hot(index):
    onehot = np.zeros(VOCABULARY_SIZE)
    onehot[index] = 1
    return onehot


def caption_to_indeces(caption, vocabulary):
    def _to_index(w):
        return np.where(vocabulary == w)[0][0]

    words = caption.split()
    try:
        indeces = [_to_index(w) for w in words]
    except IndexError:
        indeces = np.zeros(len(words))

    return torch.LongTensor(indeces)
