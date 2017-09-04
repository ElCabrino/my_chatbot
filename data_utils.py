# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import glob

from clint.textui import progress

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" %
              (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            #print(f)
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(
                    line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + \
                sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    # print(line)
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                                      tokenizer, normalize_digits)
                    tokens_file.write(
                        " ".join([str(tok) for tok in token_ids]) + "\n")



#not working
def prepare_data_maybe_download(directory):
    """
    Download and unpack dialogs if necessary.
    """
    filename = 'ubuntu_dialogs.tgz'
    url = 'http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/ubuntu_dialogs.tgz'
    dialogs_path = os.path.join(directory, 'dialogs')

    # test it there are some dialogs in the path
    if not os.path.exists(os.path.join(directory, "10", "1.tst")):
        # dialogs are missing
        archive_path = os.path.join(directory, filename)
        if not os.path.exists(archive_path):
                # archive missing, download it
            print("Downloading %s to %s" % (url, archive_path))
            filepath, _ = urllib.request.urlretrieve(url, archive_path)
            print("Successfully downloaded " + filepath)

        # unpack data
        if not os.path.exists(dialogs_path):
            print("Unpacking dialogs ...")
            with tarfile.open(archive_path) as tar:
                tar.extractall(path=directory)
            print("Archive unpacked.")

        return

def rm_one_way_conv(dialogs_path):
    """
    remove all one way conversation in the tsv that are in dialogs_path
    do nothing if it's already done (if .one_way_conv_removed exists)
    """
    print('Removing one way conversations...')
    if not os.path.exists(os.path.join(dialogs_path + '/.one_way_conv_removed')):
        files=glob.glob(os.path.join(dialogs_path+'/**/*.tsv'))

        for tsv in files:
            persons = []
            with open(tsv, 'r') as tsv_file:
                for line in tsv_file:
                    line_split = line.split('\t')
                    if line_split[1] not in persons:
                        persons.append(line_split[1])
            if len(persons) < 2:
                print('\rRemoving ' + tsv, end='')
                os.remove(tsv)
        open(os.path.join(dialogs_path + '/.one_way_conv_removed'), 'w+')
        print(' Done!')
    else:
        print('Already removed.')


def create_my_dataset(dialogs_path, train_enc, train_dec, test_enc, test_dec):
    """
    create the .enc and .dec files from the tsv of the ubuntu dialog ubuntu corpus
    if A starts the conversation with B and finish it, I don't consider the lasts messages of A
    because I need the same number of messages from A and B
    """
    print('Creating the formatted dataset from the Unbuntu Dialog Corpus...')
    if os.path.exists(train_enc) and os.path.exists(train_dec):
        print('Dataset already created.')
        return

    rm_one_way_conv(dialogs_path)
    files=glob.glob(os.path.join(dialogs_path+'/**/*.tsv'))
    size = len(files)
    count = 1
    enc_file=open(train_enc+'.tmp', 'w+')
    dec_file=open(train_dec+'.tmp', 'w+')
    enc_dec_files = [enc_file, dec_file]
    str_buffer = ''
    total_line = 0
    for tsv in files:
        print('\rParsing file ' + str(count) +' of ' + str(size) +': ' + tsv + '            ', end='')
        both_found=False
        #last person that have sent a msg
        last_person = ''
        file_to_write = 0
        with open(tsv, 'r+') as tsv_file:
            #reading the first message(s) and rm \n:
            first_line = tsv_file.readline()[:-1]
            #spliting the line
            first_line_split = first_line.split('\t')
            #we find the first person
            last_person = first_line_split[1]
            row = 0
            #sometimes at the begining there is one row missing
            if len(first_line_split) == 3:
                row = 2
            else:
                row = 3
            enc_dec_files[file_to_write].write(first_line_split[row])
            while not both_found:
                line = tsv_file.readline()[:-1].split('\t')
                #in some case there is an empty line at the end of the file:
                if len(line) < 3:
                    both_found = True
                else:
                    if line[1] != last_person:
                        enc_dec_files[file_to_write].write('\n')#newline in enc_file
                        #now I write in the .dec
                        file_to_write = 1
                        enc_dec_files[file_to_write].write(line[3])
                        last_person = line[1]
                        total_line += 1
                        both_found = True
                    #this is still a line with only 1 person specified
                    else:
                        enc_dec_files[file_to_write].write(' ' + line[row]) #I concatenate the separated messages

            #now we can parse normally:
            for line in tsv_file:
                line_split = line[:-1].split('\t')
                #I verifiy is the line is not empty
                if len(line_split) > 3:
                    if line_split[1] != last_person:
                        file_to_write = (file_to_write+1) % 2
                        if file_to_write == 0:
                            enc_dec_files[1].write('\n')
                            str_buffer = line_split[3]
                            last_person = line_split[1]
                        else:
                            str_buffer += '\n'
                            enc_dec_files[0].write(str_buffer)
                            enc_dec_files[1].write(line_split[3])
                            total_line += 1
                            last_person = line_split[1]
                    #the same person send many messages in a row
                    else:
                        if file_to_write == 1:
                            enc_dec_files[1].write(' ' + line_split[3])
                        else:
                            str_buffer += ' ' +line_split[3]
            #rare case where it miss a newline
            if file_to_write == 1:
                enc_dec_files[1].write('\n')
        count += 1
    print()
    print('Done!')

    #now I split the files to create the 4 files I need:
    #train.enc, train.dec, test.enc, test.dec
    print('Finilizing...', end='')
    #spliting the enc .tmp file
    enc_file.seek(0, 0)
    split = int(total_line*(3/4))
    count = 0
    train_enc_f = open(train_enc, 'w+')
    test_enc_f = open(test_enc, 'w+')
    for line in enc_file:
        if count < split:
            train_enc_f.write(line)
        else:
            test_enc_f.write(line)
        count += 1

    #spliting the dec .tmp file
    dec_file.seek(0, 0)
    count = 0
    train_dec_f = open(train_dec, 'w+')
    test_dec_f = open(test_dec, 'w+')
    for line in dec_file:
        if count < split:
            train_dec_f.write(line)
        else:
            test_dec_f.write(line)
        count += 1

    #rm tmp file
    os.remove(train_enc+'.tmp')
    os.remove(train_dec+'.tmp')
    print(' Done!')

def prepare_my_data(working_directory, train_enc, train_dec, test_enc, test_dec, enc_vocabulary_size, dec_vocabulary_size, tokenizer=None):
    create_my_dataset("data/dialogs", train_enc, train_dec, test_enc, test_dec)
    # Create vocabularies of the appropriate sizes.
    enc_vocab_path = os.path.join(
        working_directory, "vocab%d.enc" % enc_vocabulary_size)
    dec_vocab_path = os.path.join(
        working_directory, "vocab%d.dec" % dec_vocabulary_size)
    create_vocabulary(enc_vocab_path, train_enc,
                      enc_vocabulary_size, tokenizer)
    create_vocabulary(dec_vocab_path, train_dec,
                      dec_vocabulary_size, tokenizer)

    # Create token ids for the training data.
    enc_train_ids_path = train_enc + (".ids%d" % enc_vocabulary_size)
    dec_train_ids_path = train_dec + (".ids%d" % dec_vocabulary_size)
    data_to_token_ids(train_enc, enc_train_ids_path, enc_vocab_path, tokenizer)
    data_to_token_ids(train_dec, dec_train_ids_path, dec_vocab_path, tokenizer)

    # Create token ids for the development data.
    enc_dev_ids_path = test_enc + (".ids%d" % enc_vocabulary_size)
    dec_dev_ids_path = test_dec + (".ids%d" % dec_vocabulary_size)
    data_to_token_ids(test_enc, enc_dev_ids_path, enc_vocab_path, tokenizer)
    data_to_token_ids(test_dec, dec_dev_ids_path, dec_vocab_path, tokenizer)

    return (enc_train_ids_path, dec_train_ids_path, enc_dev_ids_path, dec_dev_ids_path, enc_vocab_path, dec_vocab_path)












