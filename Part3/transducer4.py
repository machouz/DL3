import transducer1
import char
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.optim as optim
import torch.nn.functional as F
from utils import *

EPOCHS = 5
HIDDEN_RNN = [50, 50]
EMBEDDING = 50
BATCH_SIZE = 20
LR = 0.01
LR_DECAY = 0.5


def get_words_id(word, words_id):
    if word not in words_id:
        return words_id["UUUNKKK"]
    return words_id[word]


class TransducerConcat(nn.Module):
    def __init__(self, embedding_dim, hidden_lstm, vocab_size1, vocab_size2, tagset_size):
        super(TransducerConcat, self).__init__()
        self.transducer1 = transducer1.Transducer(embedding_dim, hidden_lstm, vocab_size1, tagset_size)
        self.transducer2 = char.TransducerByChar(embedding_dim, hidden_lstm, vocab_size2, tagset_size)
        self.hidden2tag = nn.Linear(tagset_size * 2, tagset_size)

    def init_hidden(self, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.transducer1.init_hidden(batch_size)
        self.transducer2.init_hidden(batch_size)

    def detach_hidden(self):
        self.transducer1.detach_hidden()
        self.transducer2.detach_hidden()

    def eval(self):
        self.transducer1.eval()
        self.transducer2.eval()
        return super(TransducerConcat, self).eval()

    def train(self, mode=True):
        self.transducer1.train(mode)
        self.transducer2.train(mode)
        return super(TransducerConcat, self).train(mode)

    def forward(self, sentence, batch=True):
        sentences_by_word, sentences_by_char = sentence
        first_out = self.transducer1(sentences_by_word, batch)
        second_out = self.transducer2(sentences_by_char, batch)
        concated = PackedSequence(
            torch.cat((first_out.data, second_out.data), dim=-1), first_out.batch_sizes)
        concated = PackedSequence(
            F.dropout(concated.data, training=self.training), concated.batch_sizes)
        tag_space = PackedSequence(
            self.hidden2tag(concated.data), concated.batch_sizes)
        tag_scores = PackedSequence(
            F.log_softmax(tag_space.data), tag_space.batch_sizes)
        return tag_scores


def train_model(model, optimizer, train_data, batch_size, dev_vecs):
    print
    loss_history = []
    accuracy_history = []
    train_vecs_by_word, train_vecs_by_char = train_data
    id_sentences_by_word, id_tags_by_word = train_vecs_by_word
    id_sentences_by_char, id_tags_by_char = train_vecs_by_char
    model.init_hidden(batch_size)
    for i in xrange(0, len(id_sentences_by_word), batch_size):
        if i % 500 == 0:
            loss, accuracy = loss_accuracy(model, dev_vecs, batch_size)
            loss_history.append(loss)
            accuracy_history.append(accuracy)
        model.train()
        data = pack_sequence(id_sentences_by_word[i:i + batch_size]), pack_sequence(
            id_sentences_by_char[i:i + batch_size])
        label = pack_sequence(id_tags_by_word[i:i + batch_size])
        model.detach_hidden()
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
    vecs_by_word, vecs_by_char = test_data
    id_sentences_by_word, id_tags_by_word = vecs_by_word
    id_sentences_by_char, id_tags_by_char = vecs_by_char
    model.init_hidden()
    for i in xrange(0, len(id_sentences_by_word), batch_size):
        data = pack_sequence(id_sentences_by_word[i:i + batch_size]), pack_sequence(
            id_sentences_by_char[i:i + batch_size])
        label = id_tags_by_word[i:i + batch_size]
        label = pack_sequence(label).data
        output = model(data)
        loss += float(PackedSequence(F.cross_entropy(output.data, label), output.batch_sizes).data)
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


def train_save(train_name, dev_name, model_file, w2i_file, wc2i_file, id_label_file):
    print("Using transducer 4")
    print("Using train:  {}".format(train_name))
    print("Using dev:  {}".format(dev_name))
    print('Learning rate {}'.format(LR))
    print('Learning rate decay {}'.format(LR_DECAY))
    print('Hidden layer {}'.format(HIDDEN_RNN))
    print('Batch size {}'.format(BATCH_SIZE))
    words, labels = load_train(train_name)
    max_len = max([len(word) for word in words])
    chars = char.get_set_of_char(words)

    words_by_char_id = {word: i + 1 for i, word in enumerate(list(set(chars)) + ["UUUNKKK"])}

    words_id = {word: i for i, word in enumerate(list(set(words)) + ["UUUNKKK"])}
    label_id = {label: i for i, label in enumerate(set(labels))}
    id_label = {i: label for label, i in label_id.items()}

    train_sentences, train_tagged_sentences = load_train_by_sentence_new(train_name)
    train_vecs_by_word = data(train_sentences, train_tagged_sentences, words_id, label_id)
    train_vecs_by_char = data_by_char(train_sentences, train_tagged_sentences, words_by_char_id, label_id, max_len)
    train_vecs = train_vecs_by_word, train_vecs_by_char

    dev_sentences, dev_tagged_sentences = load_train_by_sentence_new(dev_name)
    dev_vecs_by_word = data(dev_sentences, dev_tagged_sentences, words_id, label_id, for_batch=False)
    dev_vecs_by_char = data_by_char(dev_sentences, dev_tagged_sentences, words_by_char_id, label_id, max_len)
    dev_vecs = dev_vecs_by_word, dev_vecs_by_char

    transducer = TransducerConcat(EMBEDDING, HIDDEN_RNN, vocab_size1=len(words_id), vocab_size2=len(words_by_char_id),
                                  tagset_size=len(label_id))
    optimizer = optim.Adam(transducer.parameters(), lr=LR)

    loss_history = []
    accuracy_history = []
    for epoch in range(0, EPOCHS):
        print('Epoch {}'.format(epoch))
        loss, accuracy = train_model(transducer, optimizer, train_vecs, BATCH_SIZE, dev_vecs)
        loss_history += loss
        accuracy_history += accuracy
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * LR_DECAY

    torch.save(transducer, model_file)
    dic_to_file(words_id, w2i_file)
    dic_to_file(words_by_char_id, wc2i_file)
    dic_to_file(id_label, id_label_file)
    return loss_history, accuracy_history
