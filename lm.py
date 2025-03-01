from __future__ import annotations
import torch
import torch.nn.functional as F


def batch_to_labeled_samples(batch: torch.LongTensor) -> [torch.LongTensor, torch.LongTensor]:
    # TODO implement this.
    # The batches that we get from the reader have corpus-sequences of length max-context + 1.
    # We need to translate them to input/output examples, each of which is shorter by one.
    # That is, if our input is of dimension (b x n) our output is two tensors, each of dimension (b x n-1)
    inputs = batch[..., :-1]  # *, N index tensor
    labels = batch[..., 1:]  # *, N index tensor
    return inputs, labels


def compute_loss(logits, gold_labels):
    # logits size is (batch, seq_len, vocab_size)
    # gold_bales size is (batch, seq_len)
    # NOTE remember to handle padding (ignore them in loss calculation!)
    # NOTE cross-entropy expects other dimensions for logits
    # NOTE you can either use cross_entropy from PyTorch, or implement the loss on your own.
    logits = logits.flatten(0, -2)  # *, V tensor
    gold_labels = gold_labels.flatten()  # * tensor
    return F.cross_entropy(logits, gold_labels)
