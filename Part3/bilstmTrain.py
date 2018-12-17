import transducer1
import char
import transducer3
import transducer4
import sys

if __name__ == '__main__':

    train_name = sys.argv[1] if len(sys.argv) > 1 else "../data/pos/train"
    repr = sys.argv[2] if len(sys.argv) > 2 else "-c"
    model_file = sys.argv[3] if len(sys.argv) > 3 else 'Transducer3_pos'
    w2i_file = sys.argv[4] if len(sys.argv) > 4 else 'Transducer3_pos_w2i'
    id_label_file = sys.argv[5] if len(sys.argv) > 5 else 'Transducer3_pos_i2l'
    dev_name = sys.argv[6] if len(sys.argv) > 6 else "../data/pos/dev"

    if repr == "-a":
        accuracy = transducer1.train_save(train_name, dev_name, model_file, w2i_file, id_label_file)
    elif repr == "-b":
        accuracy = char.train_save(train_name, dev_name, model_file, w2i_file, id_label_file)
    elif repr == "-c":
        accuracy = transducer3.train_save(train_name, dev_name, model_file, w2i_file, id_label_file)
    elif repr == "-d":
        accuracy = transducer4.train_save(train_name, dev_name, model_file, w2i_file, id_label_file)
