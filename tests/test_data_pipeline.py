import torch

from miniglm.data.text_corpus import TextCorpus, contamination, pack
from miniglm.data.tokenizer import BPETokenizer

# Small, varied text so tests stay fast but the pipeline is exercised.
SMALL = (
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, "
    "must be in want of a wife. However little known the feelings of such a man may be, this truth "
    "is so well fixed in the minds of the surrounding families. Reproducibility and measurement "
    "matter more than cleverness. We measure before we conclude and revert on evidence. "
) * 4


def test_tokenizer_is_fit_on_train_only():
    """Leakage guard: the corpus tokenizer must equal a tokenizer fit on the train split alone."""
    corpus = TextCorpus(SMALL, val_fraction=0.1, vocab_size=400)
    reference = BPETokenizer.train(corpus.train_text, vocab_size=400)
    assert corpus.tokenizer.merges == reference.merges


def test_packing_invariant():
    ids = torch.arange(1000)
    seq_len = 128
    x, y = pack(ids, seq_len)
    n = (len(ids) - 1) // seq_len
    assert x.shape == (n, seq_len)
    assert y.shape == (n, seq_len)
    # y is x shifted by one in the original stream (next-token targets), order preserved
    assert torch.equal(y[0], ids[1 : seq_len + 1])
    assert torch.equal(x[1], ids[seq_len : 2 * seq_len])
    dropped = len(ids) - (n * seq_len + 1)
    assert 0 <= dropped < seq_len


def test_contamination_metric_detects_overlap():
    """The metric must report ~0 for disjoint streams and high for a copied tail."""
    a = torch.randint(0, 50, (2000,))
    b = torch.randint(50, 100, (500,))  # disjoint token ranges -> no shared 13-grams
    assert contamination(a, b, k=13) == 0.0
    leaked = a[100:400].clone()  # a verbatim slice of train
    assert contamination(a, leaked, k=13) > 0.9


def test_held_out_split_is_disjoint_text():
    corpus = TextCorpus(SMALL, val_fraction=0.2, vocab_size=400)
    assert corpus.train_text and corpus.val_text
    assert corpus.train_text[-20:] != corpus.val_text[-20:]  # different regions


def test_vocab_round_trips_both_splits():
    corpus = TextCorpus(SMALL, val_fraction=0.1, vocab_size=400)
    assert corpus.tokenizer.decode(corpus.train.tolist()) == corpus.train_text
    assert corpus.tokenizer.decode(corpus.val.tolist()) == corpus.val_text
