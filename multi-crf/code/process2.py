import tensorflow as tf
import numpy as np
import re

ZERO_STR = '<ZERO>'
PAD_STR = '<PAD>'
OOV_STR = '<OOV>'
NONE_STR = '<NONE>'
SENT_START = '<S>'
SENT_END = '</S>'

pad_strs = [PAD_STR, SENT_START, SENT_END, ZERO_STR, NONE_STR]

def parse_line(line):
    word, label, _, _ = line.strip().split('\t')
    return word, label

def shape(string):
    if all(c.isupper() for c in string):
        return 'AA'
    if string[0].isupper():
        return 'Aa'
    if any(c for c in string if c.isupper()):
        return 'aAa'
    else:
        return 'a'

def batch_to_array(batch):
    shape = (len(batch), max(map(len, batch)))

    batch_array = np.zeros(shape, dtype=np.int64)
    for i, e in enumerate(batch):
        batch_array[i, :len(e)] = e

    return batch_array

class DataProcessor(object):
    def __init__(self, vocab=None, window_size=3):
        np.random.seed(42)

        self.pad_width = int(window_size / 2)

        self.data = dict()

        self.label_maps = {}

        self.token_map = {}
        self.shape_map = {}
        self.char_map = {}

        self.token_map[PAD_STR] = len(self.token_map)
        self.char_map[PAD_STR] = len(self.char_map)
        self.shape_map[PAD_STR] = len(self.shape_map)

        self.token_map[OOV_STR] = len(self.token_map)
        self.char_map[OOV_STR] = len(self.char_map)

        if vocab:
            print('Loading vocab from ' + vocab + '\n')
            with open(vocab, 'r') as f:
                for line in f.readlines():
                    word = line.strip().split(' ')[0]
                    if word not in self.token_map:
                        self.token_map[word] = len(self.token_map)

    def inv_token_map(self):
        return {i: t for t, i in self.token_map.items()}

    def inv_label_maps(self):
        return {k: {i: l for l, i in m.items()} for k, m in self.label_maps.items()}

    def make_example(self, key, lines, label_set, update):
        sent_len = len(lines)
        max_len_with_pad = self.pad_width * 2 + sent_len

        words = [l.split('\t')[0] for l in lines]
        max_word_len = max(map(len, words))

        oov_count = 0
        if sent_len == 0:
            return 0, 0, 0

        tokens = np.zeros(max_len_with_pad, dtype=np.int64)
        shapes = np.zeros(max_len_with_pad, dtype=np.int64)
        chars = np.zeros(max_len_with_pad * max_word_len, dtype=np.int64)
        int_labels = np.zeros(max_len_with_pad, dtype=np.int64)

        tok_lens = []

        tokens[:self.pad_width] = self.token_map[PAD_STR]
        shapes[:self.pad_width] = self.shape_map[PAD_STR]
        chars[:self.pad_width] = self.char_map[PAD_STR]
        tok_lens.extend([1] * self.pad_width)

        last_label = 'O'
        labels = []

        sent_lens = []

        char_start = self.pad_width
        idx = self.pad_width
        for line in lines:
            word, label = parse_line(line)

            word_digits = re.sub('\d', '0', word)

            if word_digits not in self.token_map:
                oov_count += 1
                if update:
                    self.token_map[word_digits] = len(self.token_map)

            token_shape = shape(word_digits)
            if token_shape not in self.shape_map:
                self.shape_map[token_shape] = len(self.shape_map)

            for char in word:
                if char not in self.char_map and update:
                    self.char_map[char] = len(self.char_map)
            tok_lens.append(len(word))

            # convert label to BILOU encoding
            label_bilou = label
            # handle cases where we need to update the last token we processed
            if label == 'O' or label[0] == 'B' or (last_label != 'O' and label[2] != last_label[2]):
                if last_label[0] == 'I':
                    labels[-1] = 'L' + labels[-1][1:]
                elif last_label[0] == 'B':
                    labels[-1] = 'U' + labels[-1][1:]
            if label[0] == 'I':
                if last_label == 'O' or label[2] != last_label[2]:
                    label_bilou = 'B-' + label[2:]

            labels.append(label_bilou)
            last_label = label_bilou

            tokens[idx] = self.token_map.get(word_digits, self.token_map[OOV_STR])
            shapes[idx] = self.shape_map[token_shape]

            chars[char_start:char_start + tok_lens[-1]] = [self.char_map.get(char, self.char_map[OOV_STR]) for char in word]
            char_start += tok_lens[-1]

            idx += 1

        sent_lens.append(sent_len)

        if last_label[0] == 'I':
            labels[-1] = 'L' + labels[-1][1:]
        elif last_label[0] == 'B':
            labels[-1] = 'U' + labels[-1][1:]

        for label in labels:
            if label not in self.label_maps[label_set]:
                self.label_maps[label_set][label] = len(self.label_maps[label_set])

        tokens[idx:idx + self.pad_width] = self.token_map[PAD_STR]
        shapes[idx:idx + self.pad_width] = self.shape_map[PAD_STR]

        chars[char_start:char_start + self.pad_width] = self.char_map[PAD_STR]
        char_start += self.pad_width

        tok_lens.extend([1] * self.pad_width)

        int_labels[self.pad_width:self.pad_width + len(labels)] = list(map(lambda s: self.label_maps[label_set][s], labels))

        padded_len = (len(sent_lens) + 1) * self.pad_width + sum(sent_lens)
        chars = chars[:sum(tok_lens)]

        example = (tokens, int_labels, shapes, chars, sent_lens, tok_lens)
        self.data[key].append(example)

        return sum(sent_lens), oov_count, 1

    def read_file(self, in_file, key, label_set, update=False):
        if not key in self.data:
            self.data[key] = []

        if not label_set in self.label_maps:
            self.label_maps[label_set] = {}

        num_tokens = 0
        num_sentences = 0
        num_oov = 0

        print('Loading data from ' + in_file)

        with open(in_file) as f:
            line_buf = []
            line = f.readline()
            while line:
                line = line.strip()
                if line:
                    line_buf.append(line)
                elif line_buf:
                    toks, oov, sent = self.make_example(key, line_buf, label_set, update)
                    num_tokens += toks
                    num_oov += oov
                    num_sentences += sent

                    line_buf = []

                line = f.readline()

        print('Embeddings coverage: %2.2f%%' % ((1 - (num_oov / num_tokens)) * 100) + '\n')

    def get_batches(self, key, batch_size, random=True):
        if random:
            order = np.random.permutation(len(self.data[key]))
        else:
            order = np.arange(len(self.data[key]))

        batches = []
        i = 0
        while i + batch_size < len(order):
            batch = order[i:i + batch_size]
            i += batch_size

            full_batch = np.full(batch_size, -1, dtype=np.int)
            full_batch[:len(batch)] = batch
            batches.append(full_batch)

        num_batches = len(batches)

        token_batches = [batch_to_array([self.data[key][i][0] for i in b]) for b in batches]
        label_batches = [batch_to_array([self.data[key][i][1] for i in b]) for b in batches]
        shape_batches = [batch_to_array([self.data[key][i][2] for i in b]) for b in batches]
        char_batches = [batch_to_array([self.data[key][i][3] for i in b]) for b in batches]
        sent_len_batches = [batch_to_array([self.data[key][i][4] for i in b]) for b in batches]
        tok_len_batches = [batch_to_array([self.data[key][i][5] for i in b]) for b in batches]

        batches = [(token_batches[i], label_batches[i], shape_batches[i],
                    char_batches[i], sent_len_batches[i], tok_len_batches[i])
                   for i in range(num_batches)]

        return batches
