import torch
from torch import nn
from torchcrf import CRF


class NER_WORD_MODEL_CRF(nn.Module):
    # we provide the hyperparameters as input
    def __init__(self, hparams):
        super(NER_WORD_MODEL_CRF, self).__init__()
        self.use_crf = hparams.use_crf
        # Embedding layer: a mat∂rix vocab_size x embedding_dim where each index
        # correspond to a word in the vocabulary and the i-th row corresponds to
        # a latent representation of the i-th word in the vocabulary.

        self.word_embedding_pos = nn.Embedding(hparams.vocab_size_pos, hparams.embedding_dim_pos, padding_idx=0)
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim, padding_idx=0)
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2

        self.lstm_pos = nn.LSTM(hparams.embedding_dim_pos, hparams.hidden_dim,
                                bidirectional=hparams.bidirectional,
                                num_layers=hparams.num_layers,
                                dropout=hparams.dropout if hparams.num_layers > 1 else 0, batch_first=True)

        self.linear_pos = nn.Linear(lstm_output_dim, lstm_output_dim)
        if hparams.embeddings is not None:
            print("initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(hparams.embeddings)

        # LSTM layer: an LSTM neural network that process the input text
        # (encoded with word embeddings) from left to right and outputs
        # a new **contextual** representation of each word that depend
        # on the preciding words.
        self.lstm = nn.LSTM(hparams.embedding_dim, hparams.hidden_dim,
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers,
                            dropout=hparams.dropout if hparams.num_layers > 1 else 0, batch_first=True)
        # Hidden layer: transforms the input value/scalar into
        # a hidden vector representation.
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2
        self.linear_word = nn.Linear(lstm_output_dim, lstm_output_dim)

        # During training, randomly zeroes some of the elements of the
        # input tensor with probability hparams.dropout using samples
        # from a Bernoulli distribution. Each channel will be zeroed out
        # independently on every forward call.
        # This has proven to be an effective technique for regularization and
        # preventing the co-adaptation of neurons
        self.dropout = nn.Dropout(hparams.dropout)
        self.concat = nn.Linear(lstm_output_dim * 2, lstm_output_dim)
        self.concat2 = nn.Linear(lstm_output_dim, lstm_output_dim)
        self.concat3 = nn.Linear(lstm_output_dim, lstm_output_dim)

        self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim // 2)
        self.fc2 = nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4)
        self.fc3 = nn.Linear(lstm_output_dim // 4, lstm_output_dim // 4)
        self.classifier = nn.Linear(lstm_output_dim // 4, hparams.num_classes)
        self.relu = nn.ReLU()
        if self.use_crf:
            print("we are using crf")
            self.crf = CRF(num_tags=hparams.num_classes, batch_first=True)
        else:
            print("we don't use crf")




    def forward(self, x, x_pos):

        out_pos = self.word_embedding_pos(x_pos)
        out_pos = self.dropout(out_pos)
        out_pos, _ = self.lstm_pos(out_pos)
        out_pos = self.linear_pos(out_pos)
        out_pos = self.relu(out_pos)

        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, _ = self.lstm(embeddings)
        o = self.linear_word(o)
        o = self.dropout(o)
        o = torch.cat((o, out_pos), dim=2)
        o = self.concat(o)
        o = self.relu(o)
        o = self.concat2(o)
        o = self.relu(o)
        o = self.fc1(o)
        o = self.relu(o)
        o = self.fc2(o)
        o = self.dropout(o)
        o = self.relu(o)
        o = self.fc3(o)
        o = self.relu(o)
        output = self.classifier(o)
        return output

    def loss(self, outputs, goldLabels, mask=None):
        """ calculates cross entrophy loss for tokens
        :param feats: lstm features
        :param goldLabels: goldLabels of batch
        :param mask: non pad mask, needed for crf
        :return : average token cross entrophy loss
        """
        useCrf = True
        if useCrf:
            try:
                assert mask is not None
            except AssertionError as e:
                e.args += ("no mask provided", "aborting")
                raise
            num_tokens = int(torch.sum(mask).item())
            loss = self.crflayer.forward(outputs, goldLabels, mask)
            return -torch.sum(loss) / num_tokens

        else:
            return lossFn(outputs, goldLabels)

