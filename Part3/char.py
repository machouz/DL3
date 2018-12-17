import time
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.optim as optim
import torch.nn.functional as F
import sys
from utils import *

EPOCHS = 5
HIDDEN_RNN = [50, 50]
CHAR_LSTM = 50
EMBEDDING = 50
BATCH_SIZE = 20
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


class CharEmbedding(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(CharEmbedding, self).__init__()
        self.char_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm_char = nn.LSTM(embedding_dim, CHAR_LSTM, batch_first=True)

        self.init_hidden()

    def init_hidden(self, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden_char = (torch.randn(1, batch_size, CHAR_LSTM),
                            torch.randn(1, batch_size, CHAR_LSTM))

    def detach_hidden(self):
        self.hidden_char = (self.hidden_char[0].detach(), self.hidden_char[0].detach())

    def forward(self, words):
        embeds = self.char_embeddings(words)
        lstm_out, self.hidden_char = self.lstm_char(
            embeds)
        return lstm_out[:, -1, :]


class TransducerByChar(nn.Module):
    def __init__(self, embedding_dim, hidden_lstm, vocab_size, tagset_size=2, bidirectional=True):
        super(TransducerByChar, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_lstm = hidden_lstm
        self.word_embeddings = CharEmbedding(embedding_dim, vocab_size)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm1 = nn.LSTM(CHAR_LSTM, hidden_lstm[0] // 2, bidirectional=bidirectional, batch_first=True)

        self.lstm2 = nn.LSTM(hidden_lstm[0], hidden_lstm[1] // 2, bidirectional=bidirectional, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_lstm[1], tagset_size)
        self.init_hidden()

    def train(self, mode=True):
        self.word_embeddings.train(mode)
        return super(TransducerByChar, self).train(mode)

    def eval(self):
        self.word_embeddings.eval()
        return super(TransducerByChar, self).train(False)

    def init_hidden(self, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden1 = (torch.randn(2, batch_size, self.hidden_lstm[0] // 2),
                        torch.randn(2, batch_size, self.hidden_lstm[0] // 2))
        self.hidden2 = (torch.randn(2, batch_size, self.hidden_lstm[1] // 2),
                        torch.randn(2, batch_size, self.hidden_lstm[1] // 2))

    def detach_hidden(self):
        self.hidden1 = (self.hidden1[0].detach(), self.hidden1[1].detach())
        self.hidden2 = (self.hidden2[0].detach(), self.hidden2[1].detach())
        self.word_embeddings.detach_hidden()

    def forward(self, sentence, batch=True):
        embeds = PackedSequence(
            self.word_embeddings(sentence.data), sentence.batch_sizes)
        lstm_out, self.hidden1 = self.lstm1(
            embeds)
        lstm_out, self.hidden2 = self.lstm2(
            lstm_out)
        tag_space = PackedSequence(
            self.hidden2tag(lstm_out.data), lstm_out.batch_sizes)
        tag_scores = PackedSequence(
            F.log_softmax(tag_space.data), tag_space.batch_sizes)
        return tag_scores


def train_model(model, optimizer, train_data, batch_size):
    loss_history = []
    accuracy_history = []
    id_sentences, id_tags = train_data
    model.init_hidden(batch_size)
    for i in xrange(0, len(id_sentences), batch_size):
        print i
        if i % 500 == 0:
            loss, accuracy = loss_accuracy(transducer, dev_vecs, batch_size)
            loss_history.append(loss)
            accuracy_history.append(accuracy)
        model.train()
        data = id_sentences[i:i + batch_size]
        label = id_tags[i:i + batch_size]
        data = pack_sequence(data)
        label = pack_sequence(label)
        model.detach_hidden()
        optimizer.zero_grad()
        output = model(data)
        loss = PackedSequence(
            F.cross_entropy(output.data, label.data), output.batch_sizes)
        loss.data.backward()
        optimizer.step()


def loss_accuracy(model, test_data, batch_size=100):
    model.eval()
    loss = correct = count = 0.0
    id_sentences, id_tags = test_data
    model.init_hidden()
    for i in xrange(0, len(id_sentences), batch_size):
        data = id_sentences[i:i + batch_size]
        label = id_tags[i:i + batch_size]
        data = pack_sequence(data)
        label = pack_sequence(label).data
        output = model(data)
        loss += float(PackedSequence(F.cross_entropy(output.data, label), output.batch_sizes).data)
        pred = output.data.max(1, keepdim=True)[1].view(label.data.shape)
        correct += (pred == label.data).cpu().sum().item()
        count += label.data.shape[0]

    acc = correct / count
    # loss = loss / count

    print('Total loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, count, 100. * acc))

    return loss, acc


def data(train_sentences, train_tagged_sentences, words_id, label_id):
    id_sentences = map(
        lambda sentence: torch.tensor([get_words_id(word, words_id) for word in sentence], dtype=torch.long),
        train_sentences)
    id_tags = map(lambda sentence: torch.tensor([label_id[tag] for tag in sentence], dtype=torch.long),
                  train_tagged_sentences)
    # zipped = map(lambda (words,tags): zip(words, tags), zip(id_sentences, id_tags))
    id_sentences = sorted(id_sentences, key=len, reverse=True)
    id_tags = sorted(id_tags, key=len, reverse=True)
    return id_sentences, id_tags


def pad(tensor, length):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])


def data_by_char(train_sentences, train_tagged_sentences, char_id, label_id, max_len):
    id_sentences = []
    for sentence in train_sentences:
        id_sentence = []
        for word in sentence:
            word = torch.tensor([get_words_id(char, char_id) for char in word], dtype=torch.long)
            word = pad(word[:20], 20)
            id_sentence.append(word)
        id_sentence = torch.stack(id_sentence)
        id_sentences.append(id_sentence)

    id_tags = map(lambda sentence: torch.tensor([label_id[tag] for tag in sentence], dtype=torch.long),
                  train_tagged_sentences)
    id_sentences = sorted(id_sentences, key=len, reverse=True)
    id_tags = sorted(id_tags, key=len, reverse=True)

    return id_sentences, id_tags


def get_set_of_char(words):
    char = []
    temp = set(words)
    for word in list(temp):
        for chr in word:
            char.append(chr)
    return list(set(char))


if __name__ == '__main__':
    train_name = sys.argv[1] if len(sys.argv) > 1 else "data/pos/train"
    dev_name = sys.argv[2] if len(sys.argv) > 2 else "data/pos/dev"

    words, labels = load_train(train_name)
    max_len = max([len(word) for word in words])
    chars = get_set_of_char(words)

    words_id = {word: i + 1 for i, word in enumerate(list(set(chars)) + ["UUUNKKK"])}

    label_id = {label: i for i, label in enumerate(set(labels))}
    id_label = {i: label for label, i in label_id.items()}

    train_sentences, train_tagged_sentences = load_train_by_sentence_new(train_name)
    train_vecs = data_by_char(train_sentences, train_tagged_sentences, words_id, label_id, max_len)

    dev_sentences, dev_tagged_sentences = load_train_by_sentence_new(dev_name)
    dev_vecs = data_by_char(dev_sentences, dev_tagged_sentences, words_id, label_id, max_len)

    transducer = TransducerByChar(EMBEDDING, HIDDEN_RNN, vocab_size=len(words_id), tagset_size=len(label_id))
    optimizer = optim.Adam(transducer.parameters(), lr=LR)

    loss_history = []
    accuracy_history = []
    timer = Timer(time.time())
    for epoch in range(0, EPOCHS):
        print('Epoch {}'.format(epoch))
        train_model(transducer, optimizer, train_vecs, batch_size=BATCH_SIZE)
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * LR_DECAY
