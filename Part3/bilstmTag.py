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
    id_sentences = sorted(id_sentences, key=len, reverse=True)

    return id_sentences

if __name__ == '__main__':
    repr = sys.argv[1] if len(sys.argv) > 1 else '1'
    model_file = sys.argv[2] if len(sys.argv) > 1 else "Transducer1_pos"
    input_file = sys.argv[3] if len(sys.argv) > 2 else "../data/pos/test"
    w2i = sys.argv[4] if len(sys.argv) > 3 else 'w2i_pos'
    output_file = sys.argv[5] if len(sys.argv) > 3 else 'test.pred'


    the_model = torch.load(model_file)


    output = []
    test_file = load_test_by_sentence(input_file)
    words_id = file_to_dic(w2i)
    test_input = data_test(test_file, words_id)
    
    for sentence in test_input:
        pred = the_model(data, batch=False)




