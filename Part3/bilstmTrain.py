import time
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.optim as optim
import torch.nn.functional as F
import sys
from utils import *

EPOCHS = 5
HIDDEN_RNN = [50, 50]
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


class Transducer(nn.Module):
    def __init__(self, embedding_dim, hidden_lstm, vocab_size, tagset_size=2, bidirectional=True):
        super(Transducer, self).__init__()
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

    def forward(self, sentence, batch=True):
        if batch:
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
        else:
            embeds = self.word_embeddings(sentence)
            lstm_out, self.hidden1 = self.lstm1(
                embeds)
            lstm_out, self.hidden2 = self.lstm2(
                lstm_out)
            lstm_out = lstm_out.reshape(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2))
            tag_space = self.hidden2tag(lstm_out)
            tag_scores = F.log_softmax(tag_space)  # shape batch,classes,size

        return tag_scores


def train_model(model, optimizer, train_data, batch_size):
    loss_history = []
    accuracy_history = []
    model.train()
    id_sentences, id_tags = train_data
    model.init_hidden(batch_size)
    for i in xrange(0, len(id_sentences), batch_size):
        if i % 500 == 0:
            loss, accuracy = loss_accuracy(transducer, dev_vecs, batch_size)
            loss_history.append(loss)
            accuracy_history.append(accuracy)
        data = id_sentences[i:i + batch_size]
        label = id_tags[i:i + batch_size]
        data = pack_sequence(data)
        label = pack_sequence(label)
        model.hidden1 = (model.hidden1[0].detach(), model.hidden1[1].detach())
        model.hidden2 = (model.hidden2[0].detach(), model.hidden2[1].detach())
        optimizer.zero_grad()
        output = model(data)
        loss = PackedSequence(
            F.cross_entropy(output.data, label.data), output.batch_sizes)
        loss.data.backward()
        optimizer.step()



    return loss_history, accuracy_history


def loss_accuracy(model, test_data, batch_size=100):
    model.eval()
    loss = correct = count = 0.0
    id_sentences, id_tags = test_data
    model.init_hidden()
    for i in xrange(0, len(id_sentences), batch_size):
        data = id_sentences[i:i + batch_size]
        label = id_tags[i:i + batch_size]
        try:
            data = pack_sequence(data)
        except RuntimeError:
            print 'LALALALA'
        label = pack_sequence(label).data
        output = model(data)
        #loss += PackedSequence(F.cross_entropy(output.data, label), output.batch_sizes).data
        pred = output.data.max(1, keepdim=True)[1].view(label.data.shape)
        correct += (pred == label.data).cpu().sum().item()
        count += label.data.shape[0]

    acc = correct / count
    loss = loss / count

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
    return id_sentences, id_tags


if __name__ == '__main__':

    train_name = sys.argv[1] if len(sys.argv) > 1 else "../data/ner/train"
    dev_name = "../data/ner/dev"
    repr = sys.argv[2] if len(sys.argv) > 2 else "-a"
    model_file = sys.argv[3] if len(sys.argv) > 3 else 'Transducer1_ner'
    w2i_file = sys.argv[4] if len(sys.argv) > 4 else 'w2i_ner'
    id_label_file = sys.argv[5] if len(sys.argv) > 5 else 'id_label_ner'

    words, labels = load_train(train_name)
    words_id = {word: i for i, word in enumerate(list(set(words)) + ["UUUNKKK"])}
    label_id = {label: i for i, label in enumerate(set(labels))}
    id_label = {i: label for label, i in label_id.items()}

    train_sentences, train_tagged_sentences = load_train_by_sentence_new(train_name)
    train_vecs = data(train_sentences, train_tagged_sentences, words_id, label_id)

    dev_sentences, dev_tagged_sentences = load_train_by_sentence_new(dev_name)
    dev_vecs = data(dev_sentences, dev_tagged_sentences, words_id, label_id, for_batch=False)

    transducer = Transducer(EMBEDDING, HIDDEN_RNN, vocab_size=len(words_id), tagset_size=len(label_id))
    optimizer = optim.Adam(transducer.parameters(), lr=LR)

    loss_history = []
    accuracy_history = []
    timer = Timer(time.time())
    for epoch in range(0, EPOCHS):
        print('Epoch {}'.format(epoch))

        loss, accuracy = train_model(transducer, optimizer, train_vecs, batch_size=BATCH_SIZE)
        loss_history += loss
        accuracy_history += accuracy
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * LR_DECAY


    #torch.save(transducer, model_file)
    #dic_to_file(words_id, w2i_file)
    #dic_to_file(id_label, id_label_file)
