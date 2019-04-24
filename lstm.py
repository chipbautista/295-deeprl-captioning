import torch


class LSTM(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, img_features, prev_actions):
        pass

    def update(self, loss, optimizer):
        # to-do: make this grad ascent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class LSTM_MLP(torch.nn.Module):
    def __init__(self):
        self.lstm = LSTM()
        # add MLP layers

    def forward(self, state, action):
        pass
