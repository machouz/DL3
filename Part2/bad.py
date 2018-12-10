import sys
import numpy as np
import random

sys.path.append('../Part1/')
from gen_examples import *
from experiment import *

vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd']
USE_PALYNDROM = True if len(sys.argv) > 1 else False


def gen_palyndrom(k=4):
    pos_examples = []
    neg_examples = []
    for i in range(500):
        base = choices("abcd", k)
        wrong = choices("abcd", k)
        while wrong == base:
            wrong = choices("abcd", k)

        pos_examples.append(base + base[::-1])
        neg_examples.append(base + wrong)
    np.savetxt("pos_pal", pos_examples, fmt="%s")
    np.savetxt("neg_pal", neg_examples, fmt="%s")

    return pos_examples, neg_examples


# GOOD : any permutation of a^nb^nc^nd^n
# BAD : the same permutation but with different n for each one
def gen_series(b=4, k=8):
    pos_examples = []
    neg_examples = []
    for i in range(500):
        rep = random.randint(2, k)

        base = list(np.random.permutation(["a", "b", "c", "d"]))
        good = "".join(map(lambda x: x * rep, base))

        bad = []
        r = random.randint(2, k)
        for i in base:
            nr = random.randint(2, k)
            while nr == r:
                nr = random.randint(2, k)
            bad.append(i * nr)
            r = nr
        bad = "".join(bad)

        pos_examples.append(good)
        neg_examples.append(bad)
    np.savetxt("pos_a_n_b_n", pos_examples, fmt="%s")
    np.savetxt("neg_a_n_b_n", neg_examples, fmt="%s")

    return pos_examples, neg_examples


# GOOD : any permutation of a^nb^nc^nd^n
# BAD : any permutation of a^nb^nc^nd^n with a repetition
def gen_unic(b=4, k=8):
    pos_examples = []
    neg_examples = []
    for i in range(500):
        rep = random.randint(2, k)

        good_base = list(np.random.permutation(["a", "b", "c", "d"]))
        good = "".join(map(lambda x: x * rep, good_base))

        bad_base = list(choices("abcd", 4))
        while len(bad_base) == len(set(bad_base)):
            bad_base = list(choices("abcd", 4))
        bad = "".join(map(lambda x: x * rep, bad_base))

        pos_examples.append(good)
        neg_examples.append(bad)
    np.savetxt("pos_uniq", pos_examples, fmt="%s")
    np.savetxt("neg_uniq", neg_examples, fmt="%s")

    return pos_examples, neg_examples


if __name__ == '__main__':
    EPOCHS = 100
    word_id = {word: i for i, word in enumerate(vocab)}

    pos_examples, neg_examples = gen_series()
    # pos_examples, neg_examples = gen_palyndrom() if USE_PALYNDROM else gen_series()

    pos_examples = map(lambda sentence: sentence2ids(sentence, word_id), pos_examples)
    pos_examples = map(lambda x: torch.LongTensor(x + [1]), pos_examples)

    neg_examples = map(lambda sentence: sentence2ids(sentence, word_id), neg_examples)
    neg_examples = map(lambda x: torch.LongTensor(x + [0]), neg_examples)

    data = pos_examples + neg_examples
    shuffle(data)
    train_len = int(len(data) * 0.8)
    train = data[:train_len]
    dev = data[train_len:]
    acceptor = Acceptor(EMBEDDING, HIDDEN_RNN, HIDDEN_MLP, vocab_size=len(vocab))
    optimizer = optim.Adam(acceptor.parameters(), lr=LR)

    loss_history = []
    accuracy_history = []
    launch = time.time()
    for epoch in range(0, EPOCHS):
        print('Epoch {}'.format(epoch))
        if epoch % 1 == 0:
            loss, accuracy = loss_accuracy(acceptor, dev)
            loss_history.append(loss)
            accuracy_history.append(accuracy)
            if accuracy > 0.98:
                print('Succeeded in distinguishing the two languages after {} done in {}'
                      .format(epoch, time.time() - launch))
                loss, accuracy = loss_accuracy(acceptor, train)
                break
        train_model(acceptor, optimizer, train)
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * LR_DECAY
