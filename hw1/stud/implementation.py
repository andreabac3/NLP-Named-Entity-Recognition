import numpy as np
from typing import List, Tuple

from model import Model
import torch
import spacy
from stud.my_utils import *
from stud.ner_model import NER_WORD_MODEL_CRF


class HParams():
    hidden_dim = 300
    embedding_dim = 300
    embedding_dim_pos = 300
    bidirectional = True
    num_layers = 3  # 3
    dropout = 0.4
    embeddings = None
    use_crf = USE_CRF


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    #return RandomBaseline()
    vocab_label = torch.load("model/vocab_label_new.pth", map_location=device)
    vocab_sample = torch.load("model/vocab_sample_new.pth", map_location=device)
    vocab_sample_pos = torch.load("model/vocab_sample_pos_new.pth", map_location=device)

    params = HParams()

    params.vocab_size = len(vocab_sample)
    params.num_classes = len(vocab_label)
    params.vocab_size_pos = len(vocab_sample_pos)

    params.embedding_dim = 300
    path_model = 'model/model_emb.pth'
    ner_model = NER_WORD_MODEL_CRF(params)
    ner_model.load_state_dict(torch.load(path_model, map_location=device))
    ner_model.to(device)
    ner_model.eval()
    os.system('python3 -m spacy download en_core_web_sm')
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner', 'entity_ruler'])
    nlp.tokenizer = nlp.tokenizer.tokens_from_list

    return StudentModel(device, vocab_label, vocab_sample, vocab_sample_pos, ner_model, nlp)

class RandomBaseline(Model):

    options = [
        ('LOC', 98412),
        ('O', 2512990),
        ('ORG', 71633),
        ('PER', 115758)
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [[str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x] for x in tokens]


class StudentModel(Model):
    def __init__(self, device, vocab_label, vocab_sample, vocab_sample_pos, ner_model, spacy_nlp):
        self.device = device
        self.vocab_label = vocab_label
        self.vocab_sample = vocab_sample
        self.vocab_sample_pos = vocab_sample_pos
        self.ner_model = ner_model
        self.spacy_nlp = spacy_nlp


    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        dt_test = NER_Dataset(tokens, None, device=self.device, spacy_nlp=self.spacy_nlp)
        dt_test.build_sample(self.vocab_sample, self.vocab_label, self.vocab_sample_pos)
        test_dataloader = DataLoader(dt_test, batch_size=256)
        return calculate_predictions(self.ner_model, test_dataloader, self.vocab_label)

