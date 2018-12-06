import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import shuffle

EPOCH = 10
HIDDEN_RNN = 30
HIDDEN_MLP = 30
EMBEDDING = 50
BATCH = 1000
vocab = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd']
word_id = {word: i for i, word in enumerate(vocab)}


class Acceptor(nn.Module):
    def __init__(self, embedding_dim, hidden_lstm, hidden_mlp, vocab_size, tagset_size=1):
        super(Acceptor, self).__init__()
        self.hidden_lstm = hidden_lstm
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_lstm)

        self.fc1 = nn.Linear(hidden_lstm, hidden_mlp)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_mlp, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_lstm),
                torch.zeros(1, 1, self.hidden_lstm))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        input = lstm_out.view(len(sentence), -1)
        out = self.fc1(input)
        out = F.tanh(out)
        tag_space = self.hidden2tag(out)
        tag_scores = F.softmax(tag_space, dim=1)
        return tag_scores


def sentence2ids(sentence):
    return map(lambda char: word_id[char], sentence)


if __name__ == '__main__':
    pos_examples = np.loadtxt("pos_examples", dtype=np.str)
    pos_examples = map(lambda sentence: sentence2ids(sentence), pos_examples)
    pos_examples = map(lambda x: (x, 1), pos_examples)

    neg_examples = np.loadtxt("neg_examples", dtype=np.str)
    neg_examples = map(lambda sentence: sentence2ids(sentence), neg_examples)
    neg_examples = map(lambda x: (x, 0), neg_examples)

    train_data = pos_examples + neg_examples
    shuffle(train_data)

    acceptor = Acceptor(EMBEDDING, HIDDEN_RNN, HIDDEN_MLP, vocab_size=len(vocab))
