import numpy as np
import matplotlib.pyplot as plt


def get_prefix(word):
    return word[:3] + "p***"


def get_suffix(word):
    return "***s" + word[-3:]


def create_subwords(words):
    prefix_list = []
    suffix_list = []
    for word in words:
        prefix = get_prefix(word)
        suffix = get_suffix(word)
        prefix_list.append(prefix)
        suffix_list.append(suffix)
    return words + list(set(prefix_list)) + list(set(suffix_list))


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


def load_train(fname):
    begin_tag = "STR"
    begin_word = "***"
    end_tag = "END"
    end_word = "___"
    data = [begin_word, begin_word]
    tags = [begin_tag, begin_tag]
    in_file = file(fname, 'r')
    for line in in_file:
        splitted_data = line.rsplit()

        if len(splitted_data) == 0:
            data += [end_word, end_word, begin_word, begin_word]
            tags += [end_tag, end_tag, begin_tag, begin_tag]
        else:
            word, tag = splitted_data
            data.append(word)
            tags.append(tag)

    data += [begin_word, begin_word]
    tags += [begin_tag, begin_tag]
    return data, tags


def load_train_by_sentence(fname):
    begin_tag = "STR"
    begin_word = "***"
    end_tag = "END"
    end_word = "___"
    sentences = []
    sentences_tag = []
    data = [begin_word, begin_word]
    tags = [begin_tag, begin_tag]
    in_file = file(fname, 'r')
    for line in in_file:
        splitted_data = line.rsplit()

        if len(splitted_data) == 0:
            data += [end_word, end_word]
            tags += [end_tag, end_tag]
            sentences.append(data)
            sentences_tag.append(tags)
            data = [begin_word, begin_word]
            tags = [begin_tag, begin_tag]
        else:
            word, tag = splitted_data
            data.append(word)
            tags.append(tag)
    data += [end_word, end_word]
    tags += [end_tag, end_tag]
    sentences.append(data)
    sentences_tag.append(tags)
    return sentences, sentences_tag

def load_train_by_sentence_new(fname):
    sentences = []
    sentences_tag = []
    data = []
    tags = []
    in_file = file(fname, 'r')
    for line in in_file:
        splitted_data = line.rsplit()
        if len(splitted_data) == 0:
            sentences.append(data)
            sentences_tag.append(tags)
            data = []
            tags = []
        else:
            word, tag = splitted_data
            data.append(word)
            tags.append(tag)

    if len(data) > 0:
        sentences.append(data)
        sentences_tag.append(tags)
    return sentences, sentences_tag

def load_test_by_sentence(fname):
    sentences = []
    data = []
    in_file = file(fname, 'r')
    for line in in_file:
        splitted_data = line.rsplit()
        if len(splitted_data) == 0:
            sentences.append(data)
            data = []
        else:
            word = line.rstrip()
            data.append(word)

    if len(data) > 0:
        sentences.append(data)

    return sentences


def load_test(fname):
    begin_word = "***"
    data = [begin_word, begin_word]
    in_file = file(fname, 'r')
    for line in in_file:
        splitted_data = line.rsplit()
        if len(splitted_data) == 0:
            data.append(begin_word)
            data.append(begin_word)
        else:
            word = splitted_data[0]
            data.append(word)
    data.append(begin_word)
    data.append(begin_word)
    return data


def write_to_file(fname, data):
    np.savetxt(fname, data, fmt="%s", delimiter='\n')


def dic_to_file(dic, fname):
    data = []
    for key, label in dic.items():
        data.append(key + "\t" + str(label))
    write_to_file(fname, data)

def file_to_dic(fname):
    data = {}
    for line in file(fname):
        key, value = line.split("\t")
        data[key] = int(value.rstrip())

    return data

def create_id(vec):
    element_id = {}
    id_element = {}
    s = set(vec)
    i = 0
    for element in s:
        element_id[element] = i
        id_element[i] = element
        i += 1
    return element_id, id_element
