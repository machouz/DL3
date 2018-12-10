import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from random import shuffle

EPOCHS = 10
HIDDEN_RNN = 10
HIDDEN_MLP = 5
EMBEDDING = 50
BATCH_SIZE = 1
LR = 0.001
LR_DECAY = 1
vocab = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd']


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
        lstm_in = embeds.view(sentence.shape[1], 1, -1)
        lstm_out, self.hidden = self.lstm(
            lstm_in, self.hidden)
        input = lstm_out.view(sentence.shape[1], 1, -1)[-1]
        out = self.fc1(input)
        out = F.tanh(out)
        tag_space = self.hidden2tag(out)
        tag_scores = F.log_softmax(tag_space)
        return tag_scores


def sentence2ids(sentence, word_id):
    return map(lambda char: word_id[char], sentence)


def train_model(model, optimizer, train_data):
    model.train()
    for i in xrange(0, len(train_data)):
        model.hidden = (model.hidden[0].detach(), model.hidden[1].detach())
        data = train_data[i][:-1].unsqueeze(0)
        label = train_data[i][-1].unsqueeze(0)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()


def loss_accuracy(model, test_data):
    model.eval()
    loss = correct = count = 0.0

    for i in xrange(0, len(test_data)):
        data = test_data[i][:-1].unsqueeze(0)
        label = test_data[i][-1].unsqueeze(0)
        output = model(data)
        loss += F.cross_entropy(output, label)
        pred = output.data.max(1, keepdim=True)[1].view(-1)
        correct += (pred == label).cpu().sum().item()
        count += 1

    acc = correct / count
    loss = loss / count

    print('Total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, count, 100. * acc))

    return loss, acc


if __name__ == '__main__':
    word_id = {word: i for i, word in enumerate(vocab)}

    pos_examples = np.loadtxt("pos_examples", dtype=np.str)
    pos_examples = map(lambda sentence: sentence2ids(sentence, word_id), pos_examples)
    pos_examples = map(lambda x: torch.LongTensor(x + [1]), pos_examples)

    neg_examples = np.loadtxt("neg_examples", dtype=np.str)
    neg_examples = map(lambda sentence: sentence2ids(sentence, word_id), neg_examples)
    neg_examples = map(lambda x: torch.LongTensor(x + [0]), neg_examples)

    data = pos_examples + neg_examples
    shuffle(data)
    train_len = int(len(data) * 0.8)
    train = data[:train_len]
    dev = data[train_len:]
    acceptor = Acceptor(EMBEDDING, HIDDEN_RNN, HIDDEN_MLP, vocab_size=len(vocab))
    optimizer = optim.Adam(acceptor.parameters(), lr=LR)

    loss_history = []
    accuracy_history = []
    launch = time.time()
    for epoch in range(0, EPOCHS):
        print('Epoch {}'.format(epoch))
        if epoch % 1 == 0:
            loss, accuracy = loss_accuracy(acceptor, dev)
            loss_history.append(loss)
            accuracy_history.append(accuracy)
            if accuracy > 0.98:
                print('Succeeded in distinguishing the two languages after {} done in {}'
                      .format(epoch, time.time() - launch))
                loss, accuracy = loss_accuracy(acceptor, train)
                break
        train_model(acceptor, optimizer, train)
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * LR_DECAY
