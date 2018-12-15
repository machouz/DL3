import time
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.optim as optim
import torch.nn.functional as F
import sys
from bilstmTrain import *
from utils import *



if __name__ == '__main__':
    repr = sys.argv[1] if len(sys.argv) > 1 else '1'
    model_file = sys.argv[2] if len(sys.argv) > 1 else "transducer_pos"
    input_file = sys.argv[3] if len(sys.argv) > 2 else "..data/pos/dev"

    the_model = torch.load(model_file)

    pred = predict(the_model, input_file)


