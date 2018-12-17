import transducer1
import char
import sys


if __name__ == '__main__':

    train_name = sys.argv[1] if len(sys.argv) > 1 else "data/ner/train"
    repr = sys.argv[2] if len(sys.argv) > 2 else "-b"
    model_file = sys.argv[3] if len(sys.argv) > 3 else 'Transducer1_ner'
    w2i_file = sys.argv[4] if len(sys.argv) > 4 else 'Transducer1_ner_i2l'
    id_label_file = sys.argv[5] if len(sys.argv) > 5 else 'Transducer1_ner_w2i'
    dev_name = sys.argv[6] if len(sys.argv) > 6 else "data/ner/dev"

    if repr == "-a":
        transducer1.train_save(train_name, dev_name, model_file, w2i_file, id_label_file)
    elif repr == "-b":
        char.train_save(train_name, dev_name, model_file, w2i_file, id_label_file)
