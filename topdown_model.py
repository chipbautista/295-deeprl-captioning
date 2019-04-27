"""
My implementation of the captioning model
described in Anderson's Bottom-Up Top-Down paper.

ruotianluo has his own implementation here:
https://github.com/ruotianluo/self-critical.pytorch/blob/master/models/AttModel.py
"""
import torch
import torch.nn.functional as F
from settings import *


class TopDownModel(torch.nn.Module):
    def __init__(self):
        self.word_embedding = torch.nn.Linear(
            in_features=VOCABULARY_SIZE,
            out_features=WORD_EMBEDDING_SIZE,
            bias=False
        )
        self.attention_lstm = torch.nn.LSTMCell(
            input_size=ATTENTION_LSTM_INPUT_SIZE,
            hidden_size=LSTM_HIDDEN_UNITS,
            bias=True
        )
        self.attention_layer = AttentionLayer()
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

    def forward(self, prev_hidden, pooled_image_features,
                image_features, input_word):
        """
        FUNCTION INPUTS:
        prev_hidden: shape (1000)
        pooled_image_features: shape (1, 2048)
        input_word: shape (10000)  - one-hot

        FUNCTION OUTPUT:
        word_index (argmaxed)
        """

        # Input to Attention LSTM should be concatenation of:
        # - previous hidden state of language LSTM
        # - mean-pooled image feature
        # - encoding of previously generated word
        # Resulting shape should be: 4048

        # encode input word first
        input_word = self.word_embedding(input_word)
        # Eq (2):
        attention_lstm_input = torch.cat(
            (prev_hidden, pooled_image_features, input_word), dim=1)
        attention_h, attention_c = self.attention_lstm(attention_lstm_input)

        attended_features = self.attention_layer(image_features, attention_h)

        # Input to Language LSTM should be concatenation of:
        # - attended image features
        # - output of attention LSTM
        # Resulting shape should be: 3048
        # Eq (6):
        language_lstm_input = torch.cat((attended_features, attention_h),
                                        dim=1)
        language_h, language_c = self.language_lstm(language_lstm_input)

        # Eq (7):
        # (W_p * h^2_t + b_p)
        word_logits = self.word_selection(language_h)
        word_probabilities = F.softmax(word_logits)

        word_index = torch.argmax(word_probabilities, dim=0)
        return word_index


class AttentionLayer(torch.nn.Module):
    def __init__(self):
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

    def forward(self, image_features, hidden_layer):
        """
        Follows the attention model described in Section 3.2.1

        IDK what to do with Eq (4),
        the notation a and alpha I think were confused.

        FUNCTION INPUTS:
        image_features: shape (36, 2048)
        hidden_layer: shape (1000)

        FUNCTION OUTPUT:
        attended_features: shape (1, 2048)
        """
        # shape (36, 512)
        # (W_va * v_i)
        weighted_features = self.linear_features(image_features)

        # shape (1, 512). will this broadcast if added to (36, 512)?
        # (W_ha * h^1_t)
        weighted_hidden_layer = self.linear_hidden(hidden_layer)

        # shape (36, 1)
        # Eq (3):
        attention_weights = self.linear_attention(
            F.tanh(weighted_features + weighted_hidden_layer))

        # shape (1, 2048)
        # Eq (5):
        attended_features = torch.sum(
            (attention_weights * image_features), dim=1)

        return attended_features


class TopDownModel_MLP(torch.nn.Module):
    # To-do. Need to clear architecture first.
    def __init__(self):
        self.topdown = TopDownModel()
        self.hidden_1 = torch.nn.Linear()
        self.hidden_2 = torch.nn.Linear()

    def forward(self):
        pass
