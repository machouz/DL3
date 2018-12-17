import time
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.optim as optim
import torch.nn.functional as F
import sys
from bilstmTrain import *
from utils import *

def data_test(test_sentences, words_id, for_batch=True):
    id_sentences = map(
        lambda sentence: torch.tensor([get_words_id(word, words_id) for word in sentence], dtype=torch.long),
        test_sentences)

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
    repr = sys.argv[1] if len(sys.argv) > 1 else '1'
    model_file = sys.argv[2] if len(sys.argv) > 2 else "Transducer1_pos"
    input_file = sys.argv[3] if len(sys.argv) > 3 else "../data/pos/test"
    w2i = sys.argv[4] if len(sys.argv) > 4 else 'w2i_pos'
    i2label = sys.argv[5] if len(sys.argv) > 5 else 'id_label_pos'
    output_file = sys.argv[5] if len(sys.argv) > 6 else 'test4.pos.txt'


    the_model = torch.load(model_file)


    output = []
    test_file = load_test_by_sentence(input_file)
    words_id = file_to_dic(w2i)
    id_label = file_to_dic_id(i2label)
    test_input = data_test(test_file, words_id)


    for sentence in test_input:
        sent = sentence.unsqueeze(0)
        pred = the_model(sent, batch=False)
        out = pred.data.max(1, keepdim=True)[1]
        labels = map(lambda x: id_label[x.item()], out)
        output.append(labels)

    write_to_file(test_file, output, output_file)
