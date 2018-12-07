import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from random import shuffle

EPOCHS = 10
HIDDEN_RNN = 30
HIDDEN_MLP = 30
EMBEDDING = 50
BATCH_SIZE = 1
LR = 0.01
LR_DECAY = 1
vocab = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd']
word_id = {word: i for i, word in enumerate(vocab)}


class Acceptor(nn.Module):
    def __init__(self, embedding_dim, hidden_lstm, hidden_mlp, vocab_size, tagset_size=2):
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
        input = lstm_out.view(len(sentence), -1)[-1]
        out = self.fc1(input)
        out = F.tanh(out)
        tag_space = self.hidden2tag(out)
        tag_scores = F.softmax(tag_space)
        return tag_scores


def sentence2ids(sentence):
    return map(lambda char: word_id[char], sentence)


def train_model(model, optimizer, train_data):
    model.train()
    for i in xrange(0, len(train_data)):
        print i
        data = train_data[i][:-1]
        label = train_data[i][-1]
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output.unsqueeze(0), label.unsqueeze(0))
        loss.backward()
        optimizer.step()


def loss_accuracy(model, test_data):
    model.eval()
    loss = correct = count = 0.0

    for i in xrange(0, len(test_data)):
        data = test_data[i][:-1]
        label = test_data[i][-1]
        output = model(data)
        loss += F.cross_entropy(output.unsqueeze(0), label.unsqueeze(0))
        pred = output.data.max(0, keepdim=True)[1].view(-1)
        correct += (pred == label).cpu().sum().item()
        count += 1

    acc = correct / count
    loss = loss / count

    print('Total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, count, 100. * acc))

    return loss, acc


if __name__ == '__main__':
    pos_examples = np.loadtxt("pos_examples", dtype=np.str)
    pos_examples = map(lambda sentence: sentence2ids(sentence), pos_examples)
    pos_examples = map(lambda x: torch.LongTensor(x + [1]), pos_examples)

    neg_examples = np.loadtxt("neg_examples", dtype=np.str)
    neg_examples = map(lambda sentence: sentence2ids(sentence), neg_examples)
    neg_examples = map(lambda x: torch.LongTensor(x + [0]), neg_examples)

    train_data = pos_examples + neg_examples
    shuffle(train_data)


    acceptor = Acceptor(EMBEDDING, HIDDEN_RNN, HIDDEN_MLP, vocab_size=len(vocab))
    optimizer = optim.SGD(acceptor.parameters(), lr=LR)

    loss_history = []
    accuracy_history = []
    for epoch in range(0, EPOCHS):
        print('Epoch {}'.format(epoch))
        if epoch % 1 == 0:
            loss, accuracy = loss_accuracy(acceptor, train_data)
            loss_history.append(loss)
            accuracy_history.append(accuracy)
        #train_model(acceptor, optimizer, train_data)
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * LR_DECAY
