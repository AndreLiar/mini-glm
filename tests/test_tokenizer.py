from miniglm.data.tokenizer import BPETokenizer

TEXT = (
    "mini-glm learns to predict the next token. attention lets each token look back. "
    "byte-level BPE guarantees a lossless round-trip. "
) * 8

# A lexically rich corpus with enough distinct byte-pairs to support a larger target vocab.
RICH = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow. "
    "Bright vixens jump; dozy fowl quack. Waltz, bad nymph, for quick jigs vex. "
    "Reproducibility, correctness, observability, testability, and measurable experiments matter. "
    "A dense transformer baseline precedes mixture-of-experts and alternative attention mechanisms. "
    "We measure before we conclude, we revert when the evidence demands, and we document every decision. "
) * 6


def test_round_trip_ascii_unicode_emoji():
    """Byte-level BPE must losslessly reconstruct any string — ASCII, Unicode, emoji."""
    tok = BPETokenizer.train(TEXT, vocab_size=400)
    for s in ["hello world", "café résumé naïve", "emoji: 🚀🔥🧠", "mixed 日本語 text 123"]:
        assert tok.decode(tok.encode(s)) == s


def test_training_is_deterministic():
    a = BPETokenizer.train(TEXT, vocab_size=400)
    b = BPETokenizer.train(TEXT, vocab_size=400)
    assert a.merges == b.merges


def test_vocab_size_is_an_upper_bound():
    """A tiny, repetitive corpus can exhaust its byte-pairs before hitting the target vocab.

    This documents the real BPE property: vocab_size is a ceiling, not a guarantee.
    """
    tok = BPETokenizer.train(TEXT, vocab_size=400)
    assert tok.vocab_size <= 400
    assert tok.vocab_size == 256 + len(tok.specials) + len(tok.merges)


def test_vocab_size_target_reached_on_rich_corpus():
    """Given a corpus rich enough to support the merges, the target is met exactly."""
    tok = BPETokenizer.train(RICH, vocab_size=400)
    assert tok.vocab_size == 400


def test_merges_do_not_increase_length():
    """A trained tokenizer should encode its own training text no longer than the raw byte stream."""
    tok = BPETokenizer.train(TEXT, vocab_size=400)
    assert len(tok.encode(TEXT)) <= len(TEXT.encode("utf-8"))
    assert len(tok.encode(TEXT)) < len(TEXT.encode("utf-8"))  # merges must actually help


def test_save_load_round_trip(tmp_path):
    tok = BPETokenizer.train(TEXT, vocab_size=400)
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    s = "mini-glm learns 🚀"
    assert loaded.encode(s) == tok.encode(s)
    assert loaded.decode(loaded.encode(s)) == s
