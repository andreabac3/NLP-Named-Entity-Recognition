#!/usr/bin/python3
from typing import *

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


train_path = '../../data/dev.tsv'
print(train_path.split("/")[-1])
x,y = read_dataset(train_path)
counter = {'PER':0, 'O':0, 'ORG':0, 'LOC':0}
for elem in y:
    for word in elem:
        counter[word] += 1

print(len(y))
print(counter)
