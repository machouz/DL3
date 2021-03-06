import random
import sys
from string import ascii_letters, digits, printable
import numpy as np



def choices(pool, k=1):
    r = ""
    for i in range(k, 0, -1):
        r += str(random.choice(pool))
    return r


def gen_text(rules, limit=10):
    s = ""
    i = 0
    while i < len(rules):
        # manage the token ,list or the character

        # token
        if rules[i] == '\\':
            i += 1
            if rules[i] == "d":
                pool = digits
            elif rules[i] == "w":
                pool = ascii_letters

        # list or interval
        elif rules[i] == '[':
            i += 1
            if rules[i + 1] == '-':  # interval
                beg, end = rules[i], rules[i + 2]
                pool = printable[printable.index(beg):printable.index(end) + 1]
                i = i + 3
            else:  # list
                pool = ""
                while rules[i] != ']':
                    pool += rules[i]
                    i += 1

        # character
        else:
            pool = rules[i]
        i += 1

        # manage the num of repetition
        k = 1
        if i < len(rules):
            if rules[i] == "*":
                k = random.randint(0, limit)
                i += 1
            elif rules[i] == '{':
                i += 1
                k = int(rules[i])
                i += 2
            elif rules[i] == "+":
                k = random.randint(1, limit)
                i += 1
        s += choices(pool, k)
    return s


def gen_ex(regex, k=500):
    examples = []
    for i in xrange(0, k):
        examples.append(gen_text(regex))
    return examples


if __name__ == '__main__':
    LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    positiveRegex = "[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+"
    negativeRegex = "[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+"
    pos_examples = gen_ex(positiveRegex,LIMIT)
    neg_examples = gen_ex(negativeRegex,LIMIT)
    np.savetxt("pos_examples", pos_examples, fmt="%s")
    np.savetxt("neg_examples", neg_examples, fmt="%s")
