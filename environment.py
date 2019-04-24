import numpy as np
import torch

from bert_serving.client import BertClient
bert_client = BertClient()


class Environment(object):
    def __init__(self, max_words=20, splits_dir):
        self.max_words = max_words

        self.images = {}
        # Import train, val, and test image numbers from karpathy split
        with open(splits_dir.format('train'), 'r') as f:
            self.images['train'] = np.random.shuffle(
                f.read().split('\n'))
        with open(splits_dir.format('validation'), 'r') as f:
            self.images['valid'] = np.random.shuffle(f.read().split('\n'))
        with open(splits_dir.format('test'), 'r') as f:
            self.images['test'] = np.random.shuffle(f.read().split('\n'))

    def calculate_reward(self, actions, mean_vec):
        """
        Basically distance. (or?)
        Choices: Euclidean or L1 or?
        """
        sentence = self.actions_to_sentence(actions)
        sentence_vec = bert_client.encode([sentence])[0]
        distance = np.linalg.norm(sentence_vec, mean_vec)  # L2

        words = sentence.split()
        if len(words) == 20 or words[-1] == '<EOS>':
            return distance, True
        return distance, False

    def iter_images(self, split):
        for img_number in self.images[split]:
            yield (
                torch.FloatTensor(np.load(FEATURES_DIR.format(img_number))),
                torch.FloatTensor(np.load(MEAN_VEC_DIR.format(i)))
            )

    def actions_to_sentence(self, actions):
        # use gensim here?
        sentence = None
        return sentence
