'''
The use of this file is for network training only.
'''
import spacy
from gensim.models import *
from gensim.models.word2vec import *
from sklearn.metrics import confusion_matrix

import pickle
from stud.my_utils import *
from stud.constant import *
from stud.ner_model import NER_WORD_MODEL_CRF

DEVICE: str = 'cuda' if torch.cuda.is_available() and ENABLE_GPU else 'cpu'

EPOCHS: int = 3  # 3 # TODO test con 4 epochs
STORED: bool = True  # if true use the stored model
USE_CRF: bool = False  # if true use the crf in the train
STORED_DATASET: bool = True  # false == calculate dataset
USE_EMBEDDINGS: bool = True
BATCH_SIZE: int = 32
TEST_BATCH_SIZE: int = 256

spacy.prefer_gpu()  # enable gpu for spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner', 'entity_ruler'])
nlp.tokenizer = nlp.tokenizer.tokens_from_list

if not STORED_DATASET:
    x_train, y_train = read_dataset('../../data/train.tsv')
    dt: NER_Dataset = NER_Dataset(x_train, y_train, device=DEVICE, spacy_nlp=nlp)

    vocab_sample = build_vocab(dt)
    vocab_sample_pos = build_vocab_pos(dt)
    vocab_label = build_label_vocab(dt)
    dt.build_sample(vocab_sample, vocab_label, vocab_sample_pos)
    torch.save(vocab_sample, '../../model/vocab_sample_new.pth')
    torch.save(vocab_sample_pos, '../../model/vocab_sample_pos_new.pth')
    torch.save(vocab_label, '../../model/vocab_label_new.pth')
    torch.save(dt, '../../model/train_saved_dataset.pth')


else:
    dt: NER_Dataset = torch.load('../../model/train_saved_dataset.pth', map_location=DEVICE)
    vocab_label = torch.load("../../model/vocab_label_new.pth", map_location=DEVICE)
    vocab_sample_pos = torch.load("../../model/vocab_sample_pos_new.pth", map_location=DEVICE)
    vocab_sample = torch.load("../../model/vocab_sample_new.pth", map_location=DEVICE)


class HParams():
    vocab_size = len(vocab_sample)
    hidden_dim = 300
    embedding_dim = 300
    embedding_dim_pos = 300
    num_classes = len(vocab_label)
    bidirectional = True
    num_layers = 3
    dropout = 0.4
    embeddings = None
    use_crf = USE_CRF  # set to true to test with the Conditional Random Field
    vocab_size_pos = len(vocab_sample_pos)


print("LOC = ", vocab_label['LOC'])
print("O = ", vocab_label['O'])
print("PER = ", vocab_label['PER'])
print("ORG = ", vocab_label['ORG'])

if not STORED_DATASET:
    x_dev, y_dev = read_dataset('../../data/dev.tsv')
    dt_dev = NER_Dataset(x_dev, y_dev, device=DEVICE, spacy_nlp=nlp)
    dt_dev.build_sample(vocab_sample, vocab_label, vocab_sample_pos)
    torch.save(dt_dev, '../../model/dev_saved_dataset.pth')
else:
    dt_dev = torch.load('../../model/dev_saved_dataset.pth', map_location=DEVICE)

dataloader = DataLoader(dt, batch_size=BATCH_SIZE, shuffle=True)
dataloader_dev = DataLoader(dt_dev, batch_size=BATCH_SIZE)
params: HParams = HParams()
if USE_EMBEDDINGS and not STORED:
    model = KeyedVectors.load_word2vec_format("../../model/" + 'GoogleNews-vectors-negative300.bin',
                                              binary=True)

    DIM = 300
    pretrained_vocab = model.vocab
    pretrained_embeddings = torch.randn(len(vocab_sample), DIM)
    # pretrained_embeddings = torch.randn(len(vocab_sample), vectors.dim)
    initialised = 0
    for i, w in enumerate(vocab_sample.itos):
        if w in pretrained_vocab:
            # if w in vectors.stoi:
            initialised += 1
            # vec = vectors.get_vecs_by_tokens(w)
            vec = torch.tensor(model[w], dtype=torch.float)
            pretrained_embeddings[i] = vec

    pretrained_embeddings[vocab_sample[pad_token]] = torch.zeros(DIM)  # torch.zeros(vectors.dim)

    params.embedding_dim = DIM  # vectors.dim
    params.embeddings = pretrained_embeddings
    params.vocab_size = len(vocab_sample)
    print("initialised embeddings {}".format(initialised))
    print("random initialised embeddings {} ".format(len(vocab_sample) - initialised))

ner_word_model_crf = NER_WORD_MODEL_CRF(params).to(DEVICE)
trainer = Trainer(
    model=ner_word_model_crf,
    loss_function=torch.nn.CrossEntropyLoss(ignore_index=vocab_label[pad_token]),
    optimizer=torch.optim.Adam(ner_word_model_crf.parameters()),
    label_vocab=vocab_label
)
path_model: str = '../../model/model_emb.pth'

if not STORED:
    trainer.train(dataloader, dataloader_dev, EPOCHS)
    torch.save(ner_word_model_crf.state_dict(), path_model)
else:
    ner_word_model_crf.load_state_dict(torch.load(path_model))
    ner_word_model_crf.eval()

x_test, _ = read_dataset('../../data/test.tsv')
dt_test = NER_Dataset(x_test, None, device=DEVICE, spacy_nlp=nlp)
dt_test.build_sample(vocab_sample, vocab_label, vocab_sample_pos)
test_dataloader = DataLoader(dt_test, batch_size=TEST_BATCH_SIZE)
pred = calculate_predictions(ner_word_model_crf, test_dataloader, vocab_label)

simulate_docker(pred, _)

labels_list = flat_list(_)
labels_set = list(set(labels_list))
pred_list = flat_list(pred)

cm = confusion_matrix(labels_list, pred_list, labels_set)

plot_confusion_matrix(cm, labels_set, cmap='Oranges')
