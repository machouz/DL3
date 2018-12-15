import sys
import numpy as np
import random
import matplotlib.pyplot as plt

sys.path.append('../Part1/')
from gen_examples import *
from experiment import *

vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
method = sys.argv[1]
USE_PALINDROME = True if method == '1' else False
USE_SERIES = True if method == '2' else False
USE_MULTIPLE = True if method == '3' else False


def create_graph(name, array_datas=[], array_legends=["Validation"],
                 xlabel="Epoch", ylabel="Loss",
                 make_new=True):
    if make_new:
        plt.figure()
    lines = []
    for data in array_datas:
        line, = plt.plot(data)
        lines.append(line)
    plt.title(name)
    plt.legend(lines, array_legends)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    plt.savefig(name)


def gen_palindrome(k=8):
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
# BAD : any permutation of a^nb^nc^nd^n with a repetition
def gen_unic(b=6, k=8):
    pos_examples = []
    neg_examples = []
    for i in range(500):
        rep = random.randint(1, k)

        good_base = list(np.random.permutation(['a', 'b', 'c', 'd', 'e', 'f'][:b]))
        good = "".join(map(lambda x: x * rep, good_base))

        bad_base = list(choices("abcdef", b))
        while len(bad_base) == len(set(bad_base)):
            bad_base = list(choices("abcdef", b))
        bad = "".join(map(lambda x: x * rep, bad_base))

        pos_examples.append(good)
        neg_examples.append(bad)
    np.savetxt("pos_uniq", pos_examples, fmt="%s")
    np.savetxt("neg_uniq", neg_examples, fmt="%s")

    return pos_examples, neg_examples


# GOOD : any permutation of a^nb^nc^nd^n
# BAD : the same permutation but with different n for each one
def gen_series(b=4, k=8):
    pos_examples = []
    neg_examples = []
    for i in range(500):
        rep = random.randint(1, k)

        base = list(np.random.permutation(['a', 'b', 'c', 'd', 'e', 'f']))
        good = "".join(map(lambda x: x * rep, base))

        bad = []
        r = []
        for _ in base:
            r.append(random.randint(1, k))
        while len(r) == len(set(r)):
            for _ in base:
                r.append(random.randint(1, k))
        for i in range(len(base)):
            bad.append(base[i] * r[i])
        bad = "".join(bad)

        pos_examples.append(good)
        neg_examples.append(bad)
    np.savetxt("pos_a_n_b_n", pos_examples, fmt="%s")
    np.savetxt("neg_a_n_b_n", neg_examples, fmt="%s")

    return pos_examples, neg_examples


def gen_multiple(k=7):
    pos_examples = [str(k * i) for i in xrange(1, 501)]
    neg_examples = []

    for i in xrange(500):
        num = random.randint(1, k * 500)
        while num % 6 == 0:
            num = random.randint(1, k * 500)
        neg_examples.append(str(num))


    np.savetxt("pos_multiple_" + str(k) + ".txt", pos_examples, fmt="%s")
    np.savetxt("neg_multiple_" + str(k) + ".txt", neg_examples, fmt="%s")

    return pos_examples, neg_examples

if __name__ == '__main__':
    EPOCHS = 90
    word_id = {word: i for i, word in enumerate(vocab)}
    # pos_examples, neg_examples = gen_series()

    pos_examples = []
    neg_examples = []

    if USE_PALINDROME:
        print "Using palindrome"
        pos_examples, neg_examples = gen_palindrome()
    elif USE_SERIES:
        print "Using series"
        pos_examples, neg_examples = gen_series()
    elif USE_MULTIPLE:
        print "Using multiple"
        pos_examples, neg_examples = gen_multiple()

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
        if epoch % 10 == 0:
            loss, accuracy = loss_accuracy(acceptor, train)
        train_model(acceptor, optimizer, train)
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * LR_DECAY


