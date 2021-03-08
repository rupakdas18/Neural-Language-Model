import os
import torch
import torch.nn as nn
from argparse import ArgumentParser
import logging
import random
import sys
import torch.nn.functional as F
import torch.optim as optim
from time import time


punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
SENT_START = "<sentence_start>"
SENT_END = "<sentence_end>"

tok_to_ind = {}
ind_to_tok = []
counts = []

def load(path):

    start = add(SENT_START)
    sentences = []
    with open(path, "r", encoding='utf8') as f:
        for paragraph in f:
            for sentence in paragraph.split(" . "):
                for ele in sentence:
                    if ele in punc:
                        sentence = sentence.replace(ele, "")
                tokens = sentence.split()
                if not tokens:
                    continue
                sentence = [add(SENT_START)]
                sentence.extend(add(t.lower()) for t in tokens)
                sentence.append(add(SENT_END))
                sentences.append(sentence)
                #print(sentences)

    return sentences


def add(word):
    """ Adds the given word to the dict, or increases it's
    count if alreays present. Returns it's index.
    """

    
    ind = tok_to_ind.get(word, None)
    if ind is None:
        ind = len(ind_to_tok)
        ind_to_tok.append(word)
        tok_to_ind[word] = ind
        counts.append(1)
    else:
        counts[ind] += 1

    return ind

print("WikiText2 preprocessing test and dataset statistics")
path = "wikitext-2"
total = 0
for part in ("train.txt", "valid.txt", "test.txt"):
  
    print("Processing", part)
    sentences = load(os.path.join(path,part))
    
    print("Found", sum(len(s) for s in sentences),
          "tokens in", len(sentences), "sentences")  
    total = total+ sum(len(s) for s in sentences)
print("Found in total", len(tok_to_ind), "tokens")



class RnnLm(nn.Module):
    """ A language model RNN with GRU layer(s). """

    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim, gru_layers, tied, dropout):
        super(RnnLm, self).__init__()
        self.tied = tied
        if not tied:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, gru_layers,
                          dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, vocab_size)
        logging.debug("Net:\n%r", self)

    def get_embedded(self, word_indexes):
        if self.tied:
            return self.fc1.weight.index_select(0, word_indexes)
        else:
            return self.embedding(word_indexes)

    def forward(self, packed_sents):
        """ Takes a PackedSequence of sentences tokens that has T tokens
        belonging to vocabulary V. Outputs predicted log-probabilities
        for the token following the one that's input in a tensor shaped
        (T, |V|).
        """
        embedded_sents = nn.utils.rnn.PackedSequence(
            self.get_embedded(packed_sents.data), packed_sents.batch_sizes)
        out_packed_sequence, _ = self.gru(embedded_sents)
        out = self.fc1(out_packed_sequence.data)
        return F.log_softmax(out, dim=1)


def batches(data, batch_size):
    """ Yields batches of sentences from 'data', ordered on length. """
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        sentences = data[i:i + batch_size]
        sentences.sort(key=lambda l: len(l), reverse=True)
        yield [torch.LongTensor(s) for s in sentences]


def step(model, sents, device):
    """ Performs a model inference for the given model and sentence batch.
    Returns the model otput, total loss and target outputs. """
    x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
    y = nn.utils.rnn.pack_sequence([s[1:] for s in sents])
    if device.type == 'cuda':
        x, y = x.cuda(), y.cuda()
    out = model(x)
    loss = F.nll_loss(out, y.data)
    return out, loss, y

class LogTimer:
    """ Utility for periodically emitting logs. Example:
        lt = LogTimer(2)
        while True:
            if lt():
                log("This is logged every 2 sec")
    """

    def __init__(self, period):
        self._period = period
        self._last_emit = time()

    def __call__(self):
        current = time()
        if current > self._last_emit + self._period:
            self._last_emit = current
            return True
        return False

def train_epoch(data, model, optimizer, args, device):
    """ Trains a single epoch of the given model. """
    model.train()
    log_timer = LogTimer(5)
    for batch_ind, sents in enumerate(batches(data, args.batch_size)):
        model.zero_grad()
        out, loss, y = step(model, sents, device)
        loss.backward()
        optimizer.step()
        if log_timer() or batch_ind == 0:
            # Calculate perplexity.
            prob = out.exp()[
                torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            perplexity = 2 ** prob.log2().neg().mean().item()
            logging.info("\tBatch %d, loss %.3f, perplexity %.2f",
                         batch_ind, loss.item(), perplexity)


def evaluate(data, model, batch_size, device):
    """ Perplexity of the given data with the given model. """
    model.eval()
    with torch.no_grad():
        entropy_sum = 0
        word_count = 0
        for sents in batches(data, batch_size):
            out, _, y = step(model, sents, device)
            prob = out.exp()[
                torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            entropy_sum += prob.log2().neg().sum().item()
            word_count += y.data.shape[0]
    return 2 ** (entropy_sum / word_count)


def parse_args(args):
    argp = ArgumentParser(description=__doc__)
    argp.add_argument("--logging", choices=["INFO", "DEBUG"],
                      default="INFO")

    argp.add_argument("--embedding-dim", type=int, default=512,
                      help="Word embedding dimensionality")
    argp.add_argument("--untied", action="store_true",
                      help="Use untied input/output embedding weights")
    argp.add_argument("--gru-hidden", type=int, default=512,
                      help="GRU gidden unit dimensionality")
    argp.add_argument("--gru-layers", type=int, default=1,
                      help="Number of GRU layers")
    argp.add_argument("--gru-dropout", type=float, default=0.0,
                      help="The amount of dropout in GRU layers")

    argp.add_argument("--epochs", type=int, default=4)
    argp.add_argument("--batch-size", type=int, default=128)
    argp.add_argument("--lr", type=float, default=0.001,
                      help="Learning rate")

    argp.add_argument("--no-cuda", action="store_true")
    return argp.parse_args(args)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(level=args.logging)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available()
                          else "cuda")

    
    # Load data now to know the whole vocabulary when training model.
    train_data = load(os.path.join(path, "train.txt"))
    valid_data = load(os.path.join(path, "valid.txt"))
    test_data = load(os.path.join(path, "test.txt"))

    


    model = RnnLm(len(tok_to_ind), args.embedding_dim,
                  args.gru_hidden, args.gru_layers,
                  not args.untied, args.gru_dropout).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    for epoch_ind in range(args.epochs):
        logging.info("Training epoch %d", epoch_ind)
        train_epoch(train_data, model, optimizer, args, device)
        logging.info("Validation perplexity: %.1f",
                     evaluate(valid_data, model, args.batch_size, device))
    logging.info("Test perplexity: %.1f",
                 evaluate(test_data, model, args.batch_size, device))


if __name__ == '__main__':
    main()

