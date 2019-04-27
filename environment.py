import numpy as np

from bert_serving.client import BertClient
bert_client = BertClient()


class Environment(object):
    def __init__(self, splits_dir):
        pass

    def calculate_reward(self, batch_actions, mean_vecs):
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
