
import torch

from collections import Counter
from torch.utils.data import Dataset
from typing import *
from tqdm import tqdm
from torchtext.vocab import Vocab


from stud.constant import *


def flat_list(l: List[List[Any]]) -> List[Any]:
    return [_e for e in l for _e in e]


def count(l: List[Any]) -> Dict[Any, int]:
    d = {}
    for e in l:
        d[e] = 1 + d.get(e, 0)
    return d


def read_dataset(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    tokens_s = []
    labels_s = []

    tokens = []
    labels = []

    with open(path) as f:

        for line in f:

            line = line.strip()

            if line.startswith('# '):
                tokens = []
                labels = []
            elif line == '':
                tokens_s.append(tokens)
                labels_s.append(labels)
            else:
                _, token, label = line.split('\t')
                tokens.append(token)
                labels.append(label)

    assert len(tokens_s) == len(labels_s)

    return tokens_s, labels_s


def pre_append(arr, elem):
    arr[0:0] = [elem]


def preprocess(sentence: List[str], label: List[str]):
    sentence.append(end_token)
    label.append(end_token)
    pre_append(sentence, start_token)
    pre_append(label, start_token)
    return sentence, label


def build_vocab(dataset):
    counter = Counter()
    for i in tqdm(range(len(dataset.list_of_sentences))):  # dataset[i] == 1 sentence
        word = dataset.list_of_sentences[i]

        if word != pad_token and word != start_token and word != end_token:
            counter[word] += 1
    return Vocab(counter, specials=['<pad>', '<unk>'])


def build_vocab_pos(dataset):
    counter = Counter()
    for i in tqdm(range(len(dataset.list_of_sentences_pos))):  # dataset[i] == 1 sentence
        word = dataset.list_of_sentences_pos[i]

        if word != pad_token and word != start_token and word != end_token:
            counter[word] += 1
    return Vocab(counter, specials=['<pad>'])


def build_label_vocab(dataset):
    counter = Counter()
    for i in tqdm(range(len(dataset.list_of_sentences_labels))):
        label = dataset.list_of_sentences_labels[i]
        if label != start_token and label != end_token and label != pad_token:
            counter[label] += 1
    # No <unk> token for labels.
    return Vocab(counter, specials=['<pad>'])


def pad_sentences(max_length: int, pad_token: str, sample: List[str]) -> List[str]:
    padded_sequence: List[str] = [pad_token] * max_length
    for i, word in enumerate(sample):
        padded_sequence[i] = word
    return padded_sequence





class NER_Dataset(Dataset):

    def __init__(self, sentence_list, label_list=None, spacy_nlp=None, device: str = None, ):
        super(NER_Dataset, self).__init__()
        self.encoded_data = None
        self.sentence_list = sentence_list
        self.label_list = label_list
        assert device != None and spacy_nlp != None
        self.device = device

        self.spacy_nlp = spacy_nlp

        new_sent = []
        new_sent_pos = []
        new_label = []
        for sent in tqdm(sentence_list):
            pos_sentence = self.pos_sentences(sent)
            new_sent_pos.append(pad_sentences(MAX_LEN, pad_token, pos_sentence))

            s = pad_sentences(MAX_LEN, pad_token, sent)
            new_sent.append(s)
        self.list_of_sentences = flat_list(new_sent)
        self.list_of_sentences_pos = flat_list(new_sent_pos)
        self.data = new_sent
        self.data_pos = new_sent_pos
        if label_list is not None:
            for label in label_list:
                l = pad_sentences(MAX_LEN, pad_token, label)
                new_label.append(l)
            self.list_of_sentences_labels = flat_list(new_label)
            self.label_list = new_label

    def encode_test(self, elem: List[str], vocab: Vocab) -> List[int]:
        sample = []
        for i in range(len(elem)):
            if elem[i] not in vocab.stoi:
                sample.append(vocab[unk_token])
                continue
            sample.append(vocab[elem[i]])
        return sample

    def pos_sentences(self, sentence: List[str]) -> List[str]:
        return [token.pos_ for doc in self.spacy_nlp.pipe([sentence]) for token in doc]

    def encode_label(self, elem: List[str], vocab: Vocab) -> List[int]:
        return [vocab[elem[i]] for i in range(len(elem))]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        if self.encoded_data is None:
            raise RuntimeError("""Trying to retrieve elements but index_dataset
            has not been invoked yet! Be sure to invoce build_sample on this object
            before trying to retrieve elements.""")
        return self.encoded_data[idx]

    def build_sample(self, vocab_words: Vocab, vocab_label: Vocab, vocab_pos: Vocab) -> None:
        self.encoded_data = list()
        for i in tqdm(range(len(self.data))):
            # for each window
            elem = self.data[i]
            pos_word = self.data_pos[i]
            encoded_labels = []
            if self.label_list != None:
                label = self.label_list[i]
                encoded_labels = torch.LongTensor(self.encode_label(label, vocab_label)).to(self.device)

            encoded_elem = torch.LongTensor(self.encode_test(elem, vocab_words)).to(self.device)
            encoded_elem_pos = torch.LongTensor(self.encode_test(pos_word, vocab_pos)).to(self.device)
            self.encoded_data.append({"inputs": encoded_elem, "outputs": encoded_labels, "pos": encoded_elem_pos})


