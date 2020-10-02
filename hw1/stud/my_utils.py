import os
from torchtext.vocab import Vocab, Vectors
from tqdm import tqdm
from sklearn.metrics import f1_score
from typing import *
import torch
from torch import nn
from torch.utils.data import DataLoader

from stud.ner_dataset import *


EPOCHS = 3
MAX_LEN = 180
STORED = False
USE_EMBEDDINGS = True
USE_CRF = False

start_token = '<start>'
end_token = '<end>'
pad_token = '<pad>'
unk_token = '<unk>'



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


class Trainer():
    """Utility class to train and evaluate a model."""

    def __init__(
            self,
            model,
            loss_function,
            optimizer,
            label_vocab: Vocab,
            log_steps: int = 10_000,
            log_level: int = 2):
        """
        Args:
            model: the model we want to train.
            loss_function: the loss_function to minimize.
            optimizer: the optimizer used to minimize the loss_function.
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.label_vocab = label_vocab
        self.log_steps = log_steps
        self.log_level = log_level
        self.label_vocab = label_vocab
        self.use_crf = model.use_crf

    def train(self, train_dataset: DataLoader,
              valid_dataset: DataLoader,
              epochs: int = 1):

        assert epochs > 0 and isinstance(epochs, int)
        if self.log_level > 0:
            print('Training ...')
        train_loss = 0.0
        for epoch in range(epochs):
            # torch.save(self.model.state_dict(), "../../model/test_model" + str(epoch) + ".pth")
            if self.log_level > 0:
                print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss = 0.0
            self.model.train()

            for step, sample in tqdm(enumerate(train_dataset), total=len(train_dataset)):
                inputs = sample['inputs']
                inputs_pos = sample['pos']
                labels = sample['outputs']

                self.optimizer.zero_grad()

                output = self.model(inputs, inputs_pos)
                if self.use_crf:
                    mask = (labels != self.label_vocab['<pad>'])
                    loss = self.model.crf(output, labels, mask=mask) * -1
                    # output = self.model.crf.decode(output, labels, mask=mask)
                    # output2 = F.softmax(output, dim=0)
                    # loss = self.loss_function(output2, labels)
                else:
                    output = output.view(-1, output.shape[-1])
                    labels = labels.view(-1)
                    loss = self.loss_function(output, labels)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.tolist()

                if self.log_level > 1 and step % self.log_steps == self.log_steps - 1:
                    print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'.format(epoch, step,
                                                                                     epoch_loss / (step + 1)))

            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss
            if self.log_level > 0:
                print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))
            if valid_dataset is not None:
                valid_loss = self.evaluate(valid_dataset)

                if self.log_level > 0:
                    print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch, valid_loss))
                    # pred = calculate_predictions(model=self.model, x_test=valid_dataset.dataset, l_label_vocab=self.label_vocab, pos_vocab_sample=None, vocab_sample=None, already_build=True)
                    # simulate_docker(pred, valid_dataset.dataset.y)

        if self.log_level > 0:
            print('... Done!')

        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss

    def evaluate(self, valid_dataset):
        """
        Args:
            valid_dataset: the dataset to use to evaluate the model.

        Returns:
            avg_valid_loss: the average validation loss over valid_dataset.
        """
        valid_loss = 0.0
        # set dropout to 0!! Needed when we are in inference mode.
        self.model.eval()
        with torch.no_grad():
            for sample in valid_dataset:
                inputs = sample['inputs']
                inputs_pos = sample['pos']
                labels = sample['outputs']

                predictions = self.model(inputs, inputs_pos)
                if self.use_crf:
                    mask = (labels != self.label_vocab[pad_token])
                    loss = -1 * self.model.crf(predictions, labels, mask=mask)
                # predictions = torch.Tensor(self.model.crf.decode(predictions, mask=mask), device=device, dtype=torch.long)
                else:
                    labels = labels.view(-1)
                    predictions = predictions.view(-1, predictions.shape[-1])
                    loss = self.loss_function(predictions, labels)
                valid_loss += loss.tolist()
        return valid_loss / len(valid_dataset)

    def predict(self, x):
        """
        Args:
            x: a tensor of indices.
        Returns:
            A list containing the predicted POS tag for each token in the
            input sentences.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            predictions = torch.argmax(logits, -1)
            return logits, predictions


def convert_label_int_to_string(arr: List[int], vocab: Vocab) -> List[str]:
    return [vocab.itos[elem] for elem in arr]


def calculate_predictions(model: nn.Module, x_test: DataLoader, l_label_vocab: Vocab) -> List[List[str]]:
    result = []
    for indexed_elem in tqdm(x_test):
        indexed_in = indexed_elem["inputs"]
        indexed_in_pos = indexed_elem["pos"]
        actual_batch_size = indexed_in.shape[0]
        predictions = model(indexed_in, indexed_in_pos)
        predictions = torch.argmax(predictions, -1)
        for batch_num in range(actual_batch_size):
            batch_predictions = predictions[batch_num]
            mask_padding = indexed_in[batch_num] != 0  # removing padding
            encoded_predictions = batch_predictions[mask_padding]
            list_predictions = encoded_predictions.tolist()
            pred = convert_label_int_to_string(list_predictions, l_label_vocab)
            result.append(pred)
    return result

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
