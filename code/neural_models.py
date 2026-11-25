# Neural models on top of word embeddings:
#   mlp    - feed-forward net on the averaged word vectors (embeddings frozen)
#   cnn    - 1D CNN with several filter widths + max pooling (Kim, 2014)
#   bilstm - bidirectional LSTM, last hidden states of both directions
# every model x embedding combination is trained with 5 different seeds
#
# usage: python code/neural_models.py            (everything)
#        python code/neural_models.py cnn fin    (one combination)
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastTextKeyedVectors
from sklearn.metrics import f1_score

from utils import LABELS, LAB2ID, load_split, save_preds, set_seed
from train_embeddings import tokenize

torch.set_num_threads(8)
MAX_LEN = 60
SEEDS = [1, 2, 3, 4, 5]
EMB_FILES = {"glove": "data/glove_100.kv", "fin": "data/fin_fasttext_100.kv"}


def build_vocab(texts, min_count=1):
    counts = {}
    for t in texts:
        for w in tokenize(t):
            counts[w] = counts.get(w, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, c in counts.items():
        if c >= min_count:
            vocab[w] = len(vocab)
    return vocab


def encode(texts, vocab):
    X = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in tokenize(t)][:MAX_LEN]
        X[i, :len(ids)] = ids
    return torch.tensor(X)


def embedding_matrix(vocab, kv, oov_words):
    E = np.random.normal(0, 0.1, (len(vocab), kv.vector_size)).astype(np.float32)
    E[0] = 0
    found = 0
    for w, i in vocab.items():
        if w in kv.key_to_index:
            E[i] = kv[w]
            found += 1
        elif isinstance(kv, FastTextKeyedVectors):
            E[i] = kv.get_vector(w)   # fastText can build vectors for unseen words from char n-grams
            found += 1
        else:
            oov_words.append(w)
    return torch.tensor(E), found / len(vocab)


class MLP(nn.Module):
    def __init__(self, E):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(E, freeze=True, padding_idx=0)
        self.net = nn.Sequential(nn.Linear(E.shape[1], 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 3))

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        avg = (self.emb(x) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.net(avg)


class CNN(nn.Module):
    def __init__(self, E, n_filters=100, widths=(2, 3, 4, 5)):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(E, freeze=False, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(E.shape[1], n_filters, k) for k in widths])
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(n_filters * len(widths), 3)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)              # (batch, dim, len)
        pooled = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.out(self.drop(torch.cat(pooled, 1)))


class BiLSTM(nn.Module):
    def __init__(self, E, hidden=128):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(E, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(E.shape[1], hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(2 * hidden, 3)

    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[0], h[1]], dim=1)            # forward + backward final states
        return self.out(self.drop(h))


def predict(model, X, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(X[i:i + bs]), -1))
    return torch.cat(out).numpy()


def train_one(arch, E, Xtr, ytr, Xva, yva, seed, max_epochs=30, patience=4):
    set_seed(seed)
    model = {"mlp": MLP, "cnn": CNN, "bilstm": BiLSTM}[arch](E)
    lr = 1e-3 if arch != "mlp" else 2e-3
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state, bad = -1, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        f1 = f1_score(yva.numpy(), predict(model, Xva).argmax(1), average="macro")
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:     # early stopping
                break
    model.load_state_dict(best_state)
    return model, best_f1, ep + 1


if __name__ == "__main__":
    archs = [sys.argv[1]] if len(sys.argv) > 1 else ["mlp", "cnn", "bilstm"]
    embs = [sys.argv[2]] if len(sys.argv) > 2 else ["glove", "fin"]

    train, val, test = load_split("pb_train"), load_split("pb_val"), load_split("pb_test")
    tw = load_split("twitter_test")
    vocab = build_vocab(train["text"])
    print("vocab size:", len(vocab))
    Xtr, Xva, Xte, Xtw = (encode(d["text"], vocab) for d in (train, val, test, tw))
    ytr = torch.tensor(train["label"].map(LAB2ID).values)
    yva = torch.tensor(val["label"].map(LAB2ID).values)

    for emb in embs:
        kv = KeyedVectors.load(EMB_FILES[emb])
        oov = []
        E, cover = embedding_matrix(vocab, kv, oov)
        print(f"\n{emb}: {cover:.1%} of the training vocabulary has a pretrained vector; examples of OOV: {oov[:15]}")
        for arch in archs:
            name = f"{arch}_{emb}"
            for seed in SEEDS:
                t0 = time.time()
                model, vf1, n_ep = train_one(arch, E, Xtr, ytr, Xva, yva, seed)
                print(f"  {name} seed {seed}: val F1 {vf1:.4f} after {n_ep} epochs ({time.time() - t0:.0f}s)")
                for dname, d, X in [("pb_test", test, Xte), ("twitter", tw, Xtw)]:
                    probs = predict(model, X)
                    save_preds(name, dname, seed, d, [LABELS[i] for i in probs.argmax(1)], probs)
