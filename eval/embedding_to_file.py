import os
import h5py
import argparse
import cupy as cp
import numpy as np
from typing import List
from collections import defaultdict

from chainer import cuda
from context2vec.common.model_reader import ModelReader


def increment_write_h5py(hf, chunk, data_name='data'):
    if data_name not in hf:
        maxshape = (None,) + chunk.shape[1:]
        data = hf.create_dataset(data_name, data=chunk, chunks=chunk.shape, maxshape=maxshape, compression="gzip",
                                 compression_opts=9)

    else:
        data = hf[data_name]
        data.resize((chunk.shape[0] + data.shape[0],) + data.shape[1:])
        data[-chunk.shape[0]:] = chunk


class Sentence(list):
    def __init__(self, line: str):
        super().__init__(line.strip().split())


class SentenceLength2WordIndices:
    def __init__(self):
        self.sentences = defaultdict(list)
        self.indices = defaultdict(lambda: -1)

    def append(self, length, word_indices):
        self.sentences[length].append(word_indices)
        self.indices[length] += 1

    def release_sentences(self, length):
        self.sentences[length] = []


class Context2vecBatchGenerator:

    def __init__(self, model_param_file: str, output_file_path: str, index_output_file_path: str,
                 gpu: int, batch_size: int):
        self.sentence_length_batches = SentenceLength2WordIndices()
        self.batch_size = batch_size
        model_reader = ModelReader(model_param_file, gpu)
        self.model = model_reader.model
        self.word2index = model_reader.word2index
        self.output_h5_file = h5py.File(output_file_path, 'w')

        self.index_output_file_path = index_output_file_path
        self.index_out = defaultdict(list)

    def batch_sentence(self, sentence: Sentence):
        word_indices = self.sentense_to_word_indices(sentence)
        self.sentence_length_batches.append(len(sentence), word_indices)

    def can_release_batch(self, sentence_length):
        if len(self.sentence_length_batches.sentences[sentence_length]) >= self.batch_size:
            return True
        return False

    def process_sentence(self, sentence: Sentence):
        sentence_length = len(sentence)
        self.batch_sentence(sentence)
        if self.can_release_batch(sentence_length):  # process batches
            self.process_sentence_batch(sentence_length)
            self.sentence_length_batches.release_sentences(sentence_length)

    def process_file(self, root: str, file: str):
        name = file.split('.')[0]
        file_path = os.path.join(root, file)
        with open(file_path, 'r') as f:
            for line in f:
                # read in one line at once and batch sentences
                sentence = Sentence(line)
                self.process_sentence(sentence)  # process batch and write to h5py
                index_in_h5py = self.sentence_length_batches.indices[len(sentence)]
                self.index_out[name].append(str(len(sentence)) + ',' + str(index_in_h5py))

    def process_file_dir(self, text_file_dir: str):
        if os.path.isdir(text_file_dir):
            for root, subdir, files in os.walk(text_file_dir):
                for file in files:
                    self.process_file(root, file)
            self.process_reminder()
            self.write_index_output_file()

    def write_index_output_file(self):
        with open(self.index_output_file_path, 'w') as index_output_file:
            # write indexes
            for name in self.index_out:
                index_output_file.write('{0}:::{1}\n'.format(name, '\t'.join(self.index_out[name])))

    def process_sentence_batch(self, sentence_length: int):
        print('run model for sent len {0}'.format(str(sentence_length)))
        self.model.reset_state()
        sent_ys = self.model._contexts_rep(xp.array(self.sentence_length_batches.sentences[sentence_length]))
        sent_ys = xp.array([arr.data for arr in sent_ys]).swapaxes(0, 1)
        # write to h5py
        increment_write_h5py(self.output_h5_file, sent_ys, data_name=str(sentence_length))

    def process_reminder(self):
        for sent_len in self.sentence_length_batches.sentences:
            if self.sentence_length_batches.sentences[sent_len]:
                print('reminder run_model for sentence length {0}'.format(str(sent_len)))
                self.model.reset_state()
                sent_ys = self.model._contexts_rep(xp.array(self.sentence_length_batches.sentences[sent_len]))
                sent_ys = xp.array([arr.data for arr in sent_ys]).swapaxes(0, 1)
                # write to h5py
                increment_write_h5py(self.output_h5_file, sent_ys, data_name=str(sent_len))
                self.sentence_length_batches.release_sentences(sent_len)

    # helper functions
    def sentense_to_word_indices(self, sentence: Sentence) -> List:
        sent_indices = []
        for word in sentence:
            word = word.decode('utf-8')
            if word in self.word2index:
                index = self.word2index[word]
            else:
                print('unknown word: {0}'.format(word.encode('utf-8')))
                index = self.word2index['<UNK>']
            sent_indices.append(index)
        return sent_indices

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Write context2vec embeddings to file.')
    parser.add_argument('-f', type=str, dest='model_param_file',
                        default='../models/context2vec/model_dir/MODEL-wiki.params.14', help='model_param_file',)
    parser.add_argument('-g', dest='gpu', type=int, default=-1, help='gpu, default is -1')
    parser.add_argument('-t', dest='text_file_dir', type=str, help='data text file or folder',
                        default='eval_data/CRW/context')
    parser.add_argument('-b', dest='batch_size', type=int, help='batch size', default=100)
    args = parser.parse_args()
    return args


def setup_cuda(gpu):
    if gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()
    global xp
    xp = cuda.cupy if gpu >= 0 else np


def construct_output_file_path(text_file, model_param_file):
    return text_file + '_' + model_param_file.split('/')[-1] + '.' + 'vec.h5'


def construct_index_output_file(text_file, model_param_file):
    return text_file + '_' + model_param_file.split('/')[-1] + '.' + 'index'


def main(argv=None):

    args = parse_args()
    # 1.setup
    setup_cuda(args.gpu)

    # 3. process context2vec text into vectors and store in h5py
    batch_generator = Context2vecBatchGenerator(
        model_param_file=args.model_param_file,
        output_file_path=construct_output_file_path(args.text_file_dir, args.model_param_file),
        index_output_file_path=construct_index_output_file(args.text_file_dir, args.model_param_file),
        batch_size=args.batch_size,
        gpu=args.gpu
    )
    batch_generator.process_file_dir(args.text_file_dir)
    batch_generator.output_h5_file.close()


if __name__ == '__main__':
    main()

