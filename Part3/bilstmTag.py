import time
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.optim as optim
import torch.nn.functional as F
import sys
import torch
from bilstmTrain import *
from utils import *
from transducer1 import *
import char


def data_test(test_sentences, words_id, for_batch=True):
    id_sentences = map(
        lambda sentence: torch.tensor([get_words_id(word, words_id) for word in sentence], dtype=torch.long).unsqueeze(
            0),
        test_sentences)

    return id_sentences


def data_by_char(train_sentences, char_id, max_len=100):
    id_sentences = []
    for sentence in train_sentences:
        id_sentence = []
        for word in sentence:
            word = torch.tensor([get_words_id(char, char_id) for char in word], dtype=torch.long)
            word = pad(word[:20], 20)
            id_sentence.append(word)
        id_sentence = torch.stack(id_sentence)
        id_sentences.append(pack_sequence([id_sentence]))

    return id_sentences


def data_with_subwords(train_sentences, words_id, for_batch=True):
    id_sentences = map(
        lambda sentence: torch.tensor([list([get_words_id(word, words_id), get_words_id(word[:3] + PREFIX, words_id),
                                             get_words_id(SUFFIX + word[-3:], words_id)]) for word in sentence],
                                      dtype=torch.long).unsqueeze(0),
        train_sentences)

    return id_sentences


def write_to_file(test, output, fname):
    out = []
    for i in xrange(len(test)):
        sent_pred = zip(test[i], output[i])
        for word, pred in sent_pred:
            out.append(word + ' ' + pred)

        out.append('')
    np.savetxt(fname, out, fmt='%s')


if __name__ == '__main__':

    start = time.time()

    repr = sys.argv[1] if len(sys.argv) > 1 else '-b'
    model_file = sys.argv[2] if len(sys.argv) > 2 else "Transducer2_pos"
    input_file = sys.argv[3] if len(sys.argv) > 3 else "../data/pos/test"
    w2i = sys.argv[4] if len(sys.argv) > 4 else 'Transducer2_pos_w2i'
    i2label = sys.argv[5] if len(sys.argv) > 5 else 'Transducer2_pos_i2l'
    output_file = sys.argv[6] if len(sys.argv) > 6 else 'test2.pos.txt'
    wc2i_file = sys.argv[7] if len(sys.argv) > 7 else None

    the_model = torch.load(model_file)

    output = []
    test_file = load_test_by_sentence(input_file)
    words_id = file_to_dic(w2i)
    id_label = file_to_dic_id(i2label)

    if repr == '-a':
        test_input = data_test(test_file, words_id)
    elif repr == '-b':
        test_input = data_by_char(test_file, words_id)
    elif repr == '-c':
        test_input = data_with_subwords(test_file, words_id)
    elif repr == '-d':
        words_by_char_id = file_to_dic(wc2i_file)
        test_input = zip(data_test(test_file, words_id), data_by_char(test_file, words_by_char_id))

    for sent in test_input:
        pred = the_model(sent, batch=False)
        out = pred.data.max(1, keepdim=True)[1]
        labels = map(lambda x: id_label[x.item()], out)
        output.append(labels)

    write_to_file(test_file, output, output_file)

    end = time.time() - start

    print "Predict", end