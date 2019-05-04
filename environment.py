
import numpy as np
import torch
# from torch.nn.functional import cosine_similarity
from torch.nn.utils.rnn import pad_sequence
# from bert_serving.client import BertClient

from settings import *


# bert_client = BertClient()


class Environment(object):
    def __init__(self):
        self.vocabulary = np.load('vocabulary.npy')

    def reset(self, b_img_features, b_captions, convert_caption_to_idx=True):
        """
        This does all the initialization for a given batch of image features
        and their respective ground truth captions.
        """
        # will need to keep track of current batch size else it's going to
        # throw an error when it's the last iteration and the given batch size
        # is less than the set BATCH_SIZE
        curr_batch_size = b_img_features.shape[0]

        if convert_caption_to_idx:
            # just some preprocessing here for use in supervised learning
            # with XE Loss.
            # convert words to their indeces in the vocabulary
            b_indeces = [self.caption_to_indeces(c) for c in b_captions]
            # then pad them to the max sequence length of the batch
            b_indeces = pad_sequence(b_indeces, batch_first=True)
            b_pooled_img_features = [
                torch.sum(img_features, 0) / IMAGE_FEATURE_REGIONS
                for img_features in b_img_features
            ]
        else:
            # When using this for the RL setting,
            # Do nothing and just pass back the captions
            b_indeces = b_captions

        # define initial LSTM states
        lstm_states = {
            'language_h': None,
            'language_c': None,
            'attention_h': None,
            'attention_c': None
        }

        state = {
            # shape (b, 1000)
            'language_lstm_h': torch.Tensor(
                np.repeat([np.zeros(LSTM_HIDDEN_UNITS)], curr_batch_size, 0)
            ).cuda(),
            # shape (b, 1, 2048) stack converts a list of tensors into a tensor
            'pooled_img_features': torch.stack(b_pooled_img_features).cuda(),
            # shape (b, VOCABULARY_SIZE)
            'prev_word_one_hot': torch.Tensor(np.repeat(
                [utils.encode_to_one_hot(0)], curr_batch_size, 0)).cuda(),
            # shape (b, 36, 2048)
            'img_features': torch.Tensor(b_img_features).cuda()
        }

        return b_indeces, state, lstm_states

    def get_context_reward(self, caption, gt_mean_context):
        caption_context = bert.encode(caption)
        # get dot product
        distance = 0
        return distance, (True if caption[-5:] == '<EOS>' else False)

    # def calculate_reward(self, action_history, mean_vector):
    #     """
    #     Basically distance. (or?)
    #     Choices: Euclidean or L1 or?
    #     """
    #     # sentence = self.actions_to_sentence(actions)
    #     done = False
    #     if action_history[-1] == 1:
    #         # if the last action is EOS, don't include that in decoding
    #         action_history = action_history[:-1]
    #         done = True
    #     if len(action_history) >= MAX_WORDS + 2:
    #         done = True

    #     sentence = ' '.join(
    #         [self.vocabulary[idx]
    #          for idx in action_history[1:]])  # [1:] to not include SOS
    #     predicted_vector = torch.Tensor(bert_client.encode([sentence])[0].reshape(1, -1))
    #     similarity = cosine_similarity(predicted_vector, torch.Tensor(mean_vector))
    #     # distances = np.linalg.norm(sentence_vectors, mean_vecs)  # L2

    #     return similarity, done

    def probs_to_word(self, probabilities, mode='sample'):
        if mode == 'sample':
            return np.random.choice(self.vocabulary, p=probabilities)

        return self.vocabulary[np.argmax(probabilities)]

    def caption_to_indeces(self, caption):
        def _to_index(w):
            return np.where(self.vocabulary == w)[0][0]

        words = caption.split()
        try:
            indeces = [_to_index(w) for w in words]
        except IndexError:
            indeces = np.zeros(len(words))

        return torch.LongTensor(indeces)
