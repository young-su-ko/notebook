---
layout: post
category: research
title: training protein language models / cramming challenge (part 2, in progress)
---

### actually training a protein language model
To recap, in the last post, I documented what I learned from implementing the ESM2 model. Now, I have to set up the code to actually train a pLM. To do so, I need the following:

1. A tokenizer to convert fasta sequences into numbers.
2. A dataset + dataloader
3. Training loop code, implemented with Pytorch Lightning

### a simple tokenizer
For regular language models, both word-based and character-based tokenizers have serious limitations. For word-based tokenization, the vocab size can be huge, and can't handle out-of-vocabulary words. For character-based, the context-length can blow up. So byte-pair encoding has become a crucial component for efficient tokenization for LLMs.

However, a nice property for proteins is that there are only ~20 "words" (the 20 standard amino acids) in the protein language. This means we can take the simplest approach, character-level tokenization, which is effectively word-level, since each character is a semantic unit. BPE has been used for protein sequence tokenization and showed it can compress the sequences by 64%[^1]. But for this project, I'll stick to the simplest tokenizer.

ESM2 has a vocab size of 33, comprising of the 20 AAs, 5 rare/ambiguous AAs (X,B,U,Z,O), 5 model tokens (cls, pad, eos, unk, mask) and 3 tokens I did not understand the purpose of (.,-,null)[^2].

For each sequence, we prepend a <cls\> token and append an <eos\> token. Since we will randomly crop a section of the sequence if its too long, the absence of these tokens can indicate to the model that it's looking at a cropped section.

In code, since our tokenization is just splitting at the character level, all we have to do is set up a dictionary that maps each token to an integer index (token_to_index), and then convert a sequence into a list of indices:
```python
def get_idx(tok) -> int:
    return token_to_index[tok]

def encode(sequence: str) -> list[int]:
    return [get_idx(tok) for tok in seq]
```

### handling the data

---
{: data-content="footnotes"}
[^1]: [Pre-training Protein Language Models with Label-Agnostic Binding Pairs Enhances Performance in Downstream Tasks](https://arxiv.org/abs/2012.03084)
[^2]:After some digging, it seems additional tokens are added to pad the embedding dictionary size. [See here](https://github.com/facebookresearch/esm/issues/84)