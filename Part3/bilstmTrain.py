import transducer1
import char
import transducer3
import transducer4
import sys

if __name__ == '__main__':

    train_name = sys.argv[1] if len(sys.argv) > 1 else "data/ner/train"
    repr = sys.argv[2] if len(sys.argv) > 2 else "-d"
    model_file = sys.argv[3] if len(sys.argv) > 3 else 'Transducer2_ner'
    w2i_file = sys.argv[4] if len(sys.argv) > 4 else 'Transducer2_ner_i2l'
    id_label_file = sys.argv[5] if len(sys.argv) > 5 else 'Transducer2_ner_w2i'
    dev_name = sys.argv[6] if len(sys.argv) > 6 else "data/ner/dev"
    wc2i_file = sys.argv[7] if len(sys.argv) > 7 else "Transducer2_ner_wc2i"

    if repr == "-a":
        loss_history, accuracy_history = transducer1.train_save(train_name, dev_name, model_file, w2i_file, id_label_file)
    elif repr == "-b":
        loss_history, accuracy_history = char.train_save(train_name, dev_name, model_file, w2i_file, id_label_file)
    elif repr == "-c":
        loss_history, accuracy_history = transducer3.train_save(train_name, dev_name, model_file, w2i_file, id_label_file)
    elif repr == "-d":
        loss_history, accuracy_history = transducer4.train_save(train_name, dev_name, model_file, w2i_file, wc2i_file, id_label_file)
