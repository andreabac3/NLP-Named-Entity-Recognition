'''
The use of this file is for network training only.
'''
import spacy
from gensim.models import *
from gensim.models.word2vec import *


import pickle
from stud.my_utils import *
from stud.constant import *
from stud.ner_model import NER_WORD_MODEL_CRF

DEVICE = 'cuda' if torch.cuda.is_available() and ENABLE_GPU else 'cpu'


EPOCHS = 3  # 3 # TODO test con 4 epochs
STORED = True  # if true use the stored model
USE_CRF = False  # if true use the crf in the train
STORED_DATASET = False  # false == calculate dataset
USE_EMBEDDINGS = True
BATCH_SIZE = 32
TEST_BATCH_SIZE = 256

spacy.prefer_gpu()  # enable gpu for spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner', 'entity_ruler'])
nlp.tokenizer = nlp.tokenizer.tokens_from_list


if not STORED_DATASET:
    x_train, y_train = read_dataset('../../data/train.tsv')
    dt = NER_Dataset(x_train, y_train, device=DEVICE, spacy_nlp=nlp)
    vocab_sample = build_vocab(dt)
    vocab_sample_pos = build_vocab_pos(dt)
    vocab_label = build_label_vocab(dt)
    dt.build_sample(vocab_sample, vocab_label, vocab_sample_pos)

else:
    file_dt = open("../../model/saved_dataset.pickle", "rb")
    dt = pickle.load(file_dt)
    file_dt.close()
    vocab_label = torch.load("../../model/vocab_label_new.pth")
    vocab_sample_pos = torch.load("../../model/vocab_sample_pos_new.pth")
    vocab_sample = torch.load("../../model/vocab_sample_new.pth")


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
    use_crf = USE_CRF  # set to true to test with the CRF
    vocab_size_pos = len(vocab_sample_pos)


print(vocab_sample['<pad>'])
print(vocab_label['<pad>'])

print(vocab_sample)
print("LOC = ", vocab_label['LOC'])
print("O = ", vocab_label['O'])
print("PER = ", vocab_label['PER'])
print("ORG = ", vocab_label['ORG'])
print()

if not STORED_DATASET:
    x_dev, y_dev = read_dataset('../../data/dev.tsv')
    dt_dev = NER_Dataset(x_dev, y_dev, device=DEVICE, spacy_nlp=nlp)
    dt_dev.build_sample(vocab_sample, vocab_label, vocab_sample_pos)
else:
    file_dt_dev = open("../../model/saved_dataset_dev.pickle", "rb")
    dt_dev = pickle.load(file_dt_dev)
    file_dt_dev.close()

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
path_model = '../../model/model.pth'
if USE_EMBEDDINGS:
    path_model = '../../model/model_emb.pth'

if not STORED:
    trainer.train(dataloader, dataloader_dev, EPOCHS)
    torch.save(ner_word_model_crf.state_dict(), path_model)
else:
    ner_word_model_crf.load_state_dict(torch.load(path_model))
    ner_word_model_crf.eval()

'''
if not STORED_DATASET:
    x_test, y_test = read_dataset('../../data/test.tsv')
    dt_test = MyDataset(x_test, y_test)
    dt_test.build_sample(vocab_sample, vocab_label, vocab_sample_pos)
else:
    file_dt_test = open("../../model/saved_dataset_test.picke", "rb")
    dt_test = pickle.load(file_dt_test)
    file_dt_test.close()


'''
# x_test, _ = read_dataset('../../data/test.tsv')
# pred = calculate_predictions(ner_word_model_crf, x_test, vocab_sample, vocab_label, vocab_sample_pos)
x_test, _ = read_dataset('../../data/test.tsv')
dt_test = NER_Dataset(x_test, None, device=DEVICE, spacy_nlp=nlp)
dt_test.build_sample(vocab_sample, vocab_label, vocab_sample_pos)
test_dataloader = DataLoader(dt_test, batch_size=TEST_BATCH_SIZE)
pred = calculate_predictions(ner_word_model_crf, test_dataloader, vocab_label)

def simulate_docker(pred, label):
    from sklearn.metrics import precision_score, recall_score, f1_score
    pred = flat_list(pred)
    label = flat_list(label)
    p = precision_score(label, pred, average='macro')
    r = recall_score(label, pred, average='macro')
    f = f1_score(label, pred, average='macro')
    print(f'# precision: {p:.4f}')
    print(f'# recall: {r:.4f}')
    print(f'# f1: {f:.4f}')

simulate_docker(pred, _)


labels_list = flat_list(_)
labels_set = list(set(labels_list))
pred_list = flat_list(pred)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_list, pred_list, labels_set)


import numpy as np

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()



plot_confusion_matrix(cm, labels_set, cmap='Oranges')
