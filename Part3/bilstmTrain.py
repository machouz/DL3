import random
import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.nn.functional as F
import sys
from utils import *

EPOCHS = 10
HIDDEN_RNN = [10, 10]
EMBEDDING = 50
BATCH_SIZE = 1
LR = 0.01
LR_DECAY = 0.5


def Timer(start):
    while True:
        now = time.time()
        yield now - start
        start = now


def get_words_id(word, words_id):
    if word not in words_id:
        return words_id["UUUNKKK"]
    return words_id[word]


class Acceptor(nn.Module):
    def __init__(self, embedding_dim, hidden_lstm, vocab_size, tagset_size=2, bidirectional=True):
        super(Acceptor, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_lstm = hidden_lstm
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm1 = nn.LSTM(embedding_dim, hidden_lstm[0] // 2, bidirectional=bidirectional, batch_first=True)

        self.lstm2 = nn.LSTM(hidden_lstm[0], hidden_lstm[1] // 2, bidirectional=bidirectional, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_lstm[1], tagset_size)
        self.init_hidden()

    def init_hidden(self, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden1 = (torch.randn(2, batch_size, self.hidden_lstm[0] // 2),
                        torch.randn(2, batch_size, self.hidden_lstm[0] // 2))
        self.hidden2 = (torch.randn(2, batch_size, self.hidden_lstm[1] // 2),
                        torch.randn(2, batch_size, self.hidden_lstm[1] // 2))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden1 = self.lstm1(
            embeds)
        lstm_out, self.hidden2 = self.lstm2(
            lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space).view(sentence.shape[0], -1, sentence.shape[1])  # shape batch,classes,size
        return tag_scores


def train_model(model, optimizer, train_data, batch_size=1000):
    model.train()
    id_sentences, id_tags = train_data
    for i in xrange(0, len(id_sentences) / 10, batch_size):
        print i
        data = id_sentences[i:i + batch_size]
        label = id_tags[i:i + batch_size]
        model.hidden1 = (model.hidden1[0].detach(), model.hidden1[1].detach())
        model.hidden2 = (model.hidden2[0].detach(), model.hidden2[1].detach())
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()


def loss_accuracy(model, test_data):
    model.eval()
    loss = correct = count = 0.0
    id_sentences, id_tags = test_data
    for i in xrange(0, len(id_sentences), 1):
        print i
        data = id_sentences[i].unsqueeze(0)
        label = id_tags[i].unsqueeze(0)
        output = model(data)
        loss += F.cross_entropy(output, label)
        pred = output.data.max(1, keepdim=True)[1].view(label.shape)
        correct += (pred == label).cpu().sum().item()
        count += data.shape[0] * data.shape[1]

    acc = correct / count
    # loss = loss / count

    print('Total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, count, 100. * acc))

    return loss, acc


def data(train_sentences, train_tagged_sentences, words_id, label_id, for_batch=True):
    id_sentences = map(
        lambda sentence: torch.tensor([get_words_id(word, words_id) for word in sentence], dtype=torch.long),
        train_sentences)
    id_tags = map(lambda sentence: torch.tensor([label_id[tag] for tag in sentence], dtype=torch.long),
                  train_tagged_sentences)
    # zipped = map(lambda (words,tags): zip(words, tags), zip(id_sentences, id_tags))
    id_sentences = sorted(id_sentences, key=len, reverse=True)
    id_tags = sorted(id_tags, key=len, reverse=True)
    if for_batch:
        id_sentences = pad_sequence(id_sentences, batch_first=True)
        id_tags = pad_sequence(id_tags, batch_first=True)
    return id_sentences, id_tags


if __name__ == '__main__':
    train_name = sys.argv[1]  # "data/pos/train"
    dev_name = sys.argv[2]  # "data/pos/dev"

    words, labels = load_train(train_name)
    words_id = {word: i for i, word in enumerate(list(set(words)) + ["UUUNKKK"])}
    label_id = {label: i for i, label in enumerate(set(labels))}
    id_label = {i: label for label, i in label_id.items()}

    train_sentences, train_tagged_sentences = load_train_by_sentence(train_name)
    train_vecs = data(train_sentences, train_tagged_sentences, words_id, label_id)

    dev_sentences, dev_tagged_sentences = load_train_by_sentence(dev_name)
    dev_vecs = data(dev_sentences, dev_tagged_sentences, words_id, label_id, for_batch=False)

    acceptor = Acceptor(EMBEDDING, HIDDEN_RNN, vocab_size=len(words_id), tagset_size=len(label_id))
    optimizer = optim.Adam(acceptor.parameters(), lr=LR)

    loss_history = []
    accuracy_history = []
    timer = Timer(time.time())
    for epoch in range(0, EPOCHS):
        print('Epoch {}'.format(epoch))
        if epoch % 3 == 0:
            loss, accuracy = loss_accuracy(acceptor, dev_vecs)
            loss_history.append(loss)
            accuracy_history.append(accuracy)
            if accuracy > 0.98:
                print('Succeeded in distinguishing the two languages after {} done in {}'
                      .format(epoch, timer.next()))
                loss, accuracy = loss_accuracy(acceptor, train_vecs)
                break
        train_model(acceptor, optimizer, train_vecs, batch_size=BATCH_SIZE)
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * LR_DECAY
