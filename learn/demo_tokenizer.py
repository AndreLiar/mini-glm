"""DEMO 1 — The tokenizer: turning text into numbers and back.

A computer can't read letters, only numbers. The tokenizer chops text into small pieces
("tokens") and gives each a number. Run me:  python learn/demo_tokenizer.py
"""

from miniglm.data.tokenizer import BPETokenizer

# A tiny training text. The tokenizer learns common pieces from it.
text = "the cat sat on the mat. the cat ran to the cat. " * 20

print("1) We train a tiny tokenizer on a small text.\n")
tok = BPETokenizer.train(text, vocab_size=300)

sentence = "the cat"
ids = tok.encode(sentence)

print(f"   Original text : {sentence!r}")
print(f"   As numbers    : {ids}")
print(f"   Back to text  : {tok.decode(ids)!r}")
print("\n   --> text becomes numbers, and the numbers turn back into the exact text.\n")

print("2) Why 'tokens' and not just letters? Common pieces get merged into ONE number.\n")
for word in ["the", "cat", "zebra"]:
    n = len(tok.encode(word))
    print(f"   {word!r:9} -> {n} token(s)   ({'a word it saw a lot, so it is 1 piece' if n==1 else 'pieces it had to spell out'})")
print("\n   --> 'the' and 'cat' were seen so often they each became a single token.")
print("       'zebra' was never seen, so it gets spelled out from smaller pieces.")
