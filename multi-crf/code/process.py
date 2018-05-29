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

class DataProcessor(object):
    def __init__(self, vocab=None, window_size=3):
        self.pad_width = int(window_size / 2)

        self.token_map = {}
        self.label_maps = {}
        self.shape_map = {}
        self.char_map = {}

        self.token_map[PAD_STR] = len(self.token_map)
        self.char_map[PAD_STR] = len(self.char_map)
        self.shape_map[PAD_STR] = len(self.shape_map)

        self.token_map[OOV_STR] = len(self.token_map)
        self.char_map[OOV_STR] = len(self.char_map)

        if vocab:
            print('Reading vocab from ' + vocab + '\n')
            with open(vocab, 'r') as f:
                for line in f.readlines():
                    word = line.strip().split(' ')[0]
                    if word not in self.token_map:
                        self.token_map[word] = len(self.token_map)

    def make_example(self, writer, lines, label_set, update):
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

        example = tf.train.SequenceExample()

        fl_labels = example.feature_lists.feature_list['labels']
        for l in int_labels:
            fl_labels.feature.add().int64_list.value.append(l)

        fl_tokens = example.feature_lists.feature_list['tokens']
        for t in tokens:
            fl_tokens.feature.add().int64_list.value.append(t)

        fl_shapes = example.feature_lists.feature_list['shapes']
        for s in shapes:
            fl_shapes.feature.add().int64_list.value.append(s)

        fl_chars = example.feature_lists.feature_list['chars']
        for c in chars:
            fl_chars.feature.add().int64_list.value.append(c)

        fl_seq_len = example.feature_lists.feature_list['seq_len']
        for seq_len in sent_lens:
            fl_seq_len.feature.add().int64_list.value.append(seq_len)

        fl_tok_len = example.feature_lists.feature_list['tok_len']
        for tok_len in tok_lens:
            fl_tok_len.feature.add().int64_list.value.append(tok_len)

        writer.write(example.SerializeToString())
        return sum(sent_lens), oov_count, 1

    def read_file(self, in_file, out_file, label_set, update=False):
        if not label_set in self.label_maps:
            self.label_maps[label_set] = {}

        num_tokens = 0
        num_sentences = 0
        num_oov = 0

        writer = tf.python_io.TFRecordWriter(out_file)
        print('Reading ' + in_file)

        with open(in_file) as f:
            line_buf = []
            line = f.readline()
            while line:
                line = line.strip()
                if line:
                    line_buf.append(line)
                elif line_buf:
                    toks, oov, sent = self.make_example(writer, line_buf, label_set, update)
                    num_tokens += toks
                    num_oov += oov
                    num_sentences += sent

                    line_buf = []

                line = f.readline()

        print('Wrote to ' + out_file)
        writer.close()

        print('Embeddings coverage: %2.2f%%' % ((1 - (num_oov / num_tokens)) * 100) + '\n')
