"""
My implementation of the captioning model
described in Anderson's Bottom-Up Top-Down paper.

ruotianluo has his own implementation here:
https://github.com/ruotianluo/self-critical.pytorch/blob/master/models/AttModel.py

For training actor-critic, follow this implementation:
https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
"""
import torch
import torch.nn.functional as F
from settings import *


class Agent(object):
    def __init__(self, mode='RL'):
        # just a hacky way to use this class for supervised learning... :(
        self.mode = mode

        self.actor = TopDownModel().cuda()
        self.actor_optim = torch.optim.SGD(
            self.actor.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            nesterov=True)
        self.actor_optim_scheduler = torch.optim.lr_scheduler.StepLR(
            self.actor_optim,
            step_size=2,
            gamma=0.9
        )

        # self.critic = TopDownModel_MLP()
        # self.critic_optim = torch.optim.Adam(self.critic.parameters())

        self.DISCOUNT = DISCOUNT_FACTOR

    def select_action(self, state, lstm_states):
        return self.actor(state, lstm_states, self.mode)

    def supervised_update(self, loss):
        # For supervised only! RL training to be written next.
        self.actor_optim.zero_grad()
        loss.backward(retain_graph=True)
        self.actor_optim.step()


class TopDownModel(torch.nn.Module):
    def __init__(self):
        super(TopDownModel, self).__init__()

        self.word_embedding = torch.nn.Embedding(
            num_embeddings=VOCABULARY_SIZE,
            embedding_dim=WORD_EMBEDDING_SIZE
        )

        self.attention_lstm = torch.nn.LSTMCell(
            input_size=ATTENTION_LSTM_INPUT_SIZE,
            hidden_size=LSTM_HIDDEN_UNITS,
            bias=True
        )

        self.attention_layer = AttentionLayer().cuda()

        self.language_lstm = torch.nn.LSTMCell(
            input_size=LANGUAGE_LSTM_INPUT_SIZE,
            hidden_size=LSTM_HIDDEN_UNITS,
            bias=True
        )

        self.word_selection = torch.nn.Linear(
            in_features=LSTM_HIDDEN_UNITS,
            out_features=VOCABULARY_SIZE,
            bias=True
        )

    def forward(self, state, lstm_states, mode='RL'):
        """
        FUNCTION INPUTS:
        language_lstm_h: shape (1000)
        pooled_img_features: shape (1, 2048)
        prev_word: shape (10000)  - one-hot

        FUNCTION OUTPUT:
        word_index (argmaxed)
        """
        language_lstm_prev = (None if lstm_states['language_h'] is None
                              else (lstm_states['language_h'],
                                    lstm_states['language_c']))
        attention_lstm_prev = (None if lstm_states['attention_h'] is None
                               else (lstm_states['attention_h'],
                                     lstm_states['attention_c']))

        # Input to Attention LSTM should be concatenation of:
        # - previous hidden state of language LSTM
        # - mean-pooled image feature
        # - encoding of previously generated word
        # Resulting shape should be: 4048
        # prev_word = self.word_embedding(state['prev_word_one_hot'])
        prev_word = self.word_embedding(state['prev_word_indeces'])
        # Eq (2):

        attention_lstm_input = torch.cat(
            (state['language_lstm_h'], state['pooled_img_features'],
             prev_word), 1)

        attention_lstm_h, attention_lstm_c = self.attention_lstm(
            attention_lstm_input,
            attention_lstm_prev
        )

        attended_features = self.attention_layer(
            state['img_features'], attention_lstm_h)

        # Input to Language LSTM should be concatenation of:
        # - attended image features
        # - output of attention LSTM
        # Resulting shape should be: 3048
        # Eq (6):
        language_lstm_input = torch.cat((attended_features, attention_lstm_h),
                                        dim=1)

        language_lstm_h, language_lstm_c = self.language_lstm(
            language_lstm_input,
            language_lstm_prev
        )

        # Eq (7):
        # (W_p * h^2_t + b_p)
        word_logits = self.word_selection(language_lstm_h)
        # word_probabilities = F.softmax(word_logits, dim=1)

        lstm_states = {
            'language_h': language_lstm_h,
            'language_c': language_lstm_c,
            'attention_h': attention_lstm_h,
            'attention_c': attention_lstm_c
        }

        # if mode == 'RL':
        #     word_index = torch.argmax(word_probabilities, dim=1)
        #     return word_index[0], lstm_states
        # else:
        return word_logits, lstm_states

    def update(self, memory):
        print('Updating agent parameters...')
        pass


class AttentionLayer(torch.nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.linear_features = torch.nn.Linear(
            in_features=IMAGE_FEATURE_DIM,
            out_features=ATTENTION_HIDDEN_UNITS,
            bias=False)
        self.linear_hidden = torch.nn.Linear(
            in_features=LSTM_HIDDEN_UNITS,
            out_features=ATTENTION_HIDDEN_UNITS,
            bias=False)
        self.linear_attention = torch.nn.Linear(
            in_features=ATTENTION_HIDDEN_UNITS,
            out_features=1,
            bias=False)

    def forward(self, img_features, hidden_layer):
        """
        Follows the attention model described in Section 3.2.1

        FUNCTION INPUTS:
        img_features: shape (36, 2048)
        hidden_layer: shape (1000)

        FUNCTION OUTPUT:
        attended_features: shape (1, 2048)
        """
        curr_batch_size = img_features.shape[0]
        # shape (36, 512)
        # (W_va * v_i)
        encoded_features = self.linear_features(img_features)

        # shape (1, 512).
        # (W_ha * h^1_t)
        encoded_hidden_layer = self.linear_hidden(hidden_layer)

        # shape (36, 1)
        # Eq (3):
        batch_sum_feature_layers = torch.stack([
            encoded_features[i] + encoded_hidden_layer[i]
            for i in range(curr_batch_size)
        ])

        attention_weights = self.linear_attention(
            torch.tanh(batch_sum_feature_layers))
        # Eq (4):
        attention_weights = F.softmax(attention_weights)
        attention_weights = torch.transpose(attention_weights, 1, 2)

        # shape (1, 2048)
        # Eq (5):
        attended_features = torch.matmul(
            attention_weights, img_features).reshape(-1, IMAGE_FEATURE_DIM)

        return attended_features


class TopDownModel_MLP(torch.nn.Module):
    # To-do. Need to clear Critic architecture first.
    def __init__(self):
        super(TopDownModel_MLP, self).__init__()
        self.topdown = TopDownModel()
        self.hidden_1 = torch.nn.Linear()
        self.hidden_2 = torch.nn.Linear()

    def forward(self):
        pass
