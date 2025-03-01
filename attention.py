from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math


def create_kqv_matrix(input_vector_dim, n_heads = 1):
    # Myself: assumes input_vector_dim is the dimension of key, query, result vectors.
    return nn.Linear(input_vector_dim, 2*input_vector_dim + input_vector_dim // n_heads) # TODO fill in the correct dimensions


def kqv(x, linear):
    B, N, D = x.size()
    # TODO compute k, q, and v
    # (can do it in 1 or 2 lines.)
    y = linear(x)
    k, q, v = torch.split(y, [D, D, y.size()[-1]-2*D], dim=-1)
    return k, q, v
def attention_scores(a, b):
    *_, D1 = a.size()

    # TODO compute A (remember: we are computing *scaled* dot product attention. don't forget the scaling.
    # (can do it in 1 or 2 lines.)
    a = a.transpose(-1, -2).float()
    b = b.float()
    A = torch.matmul(b, a) / math.sqrt(D1)
    return A


def create_causal_mask(embed_dim, n_heads, max_context_len, device="cpu"):
    # Return a causal mask (a tensor) with zeroes in dimensions we want to zero out.
    # This function receives more arguments than it actually needs. This is just because
    # it is part of an assignment, and I want you to figure out on your own which arguments
    # are relevant.

    mask = torch.ones(1, max_context_len, max_context_len, device=device).tril()  # TODO replace this line with the creation of a causal mask.
    return mask


def self_attention(v, A, mask=None, dropout=False, dropout_module=None):
    # TODO compute sa (corresponding to y in the assignemnt text).
    # This should take very few lines of code.
    # As usual, the dimensions of v and of sa are (b x n x d).
    if mask is not None:
        A = A.masked_fill(mask[..., :A.size()[-2], :A.size()[-1]] == 0, float('-inf'))
    A = nn.functional.softmax(A, dim=-1)
    if dropout:
        A = dropout_module(A)
    sa = torch.matmul(A, v)
    return sa


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa = self_attention(v, att, attention_mask)
    return sa


def multi_head_attention_layer(x, kqv_matrices, mask, dropout=False, dropout_module=None):
    B, N, D = x.size()
    # TODO implement multi-head attention.
    # This is most easily done using calls to self_attention_layer, each with a different
    # entry in kqv_matrices, and combining the results.
    #
    # There is also a tricker (but more efficient) version of multi-head attention, where we do all the computation
    # using a single multiplication with a single kqv_matrix (or a single kqv_tensor) and re-arranging the results afterwards.
    # If you want a challenge, you can try and implement this. You may need to change additional places in the code accordingly.
    k_list = []
    q_list = []
    v_list = []
    for kqv_matrix in kqv_matrices:
        k, q, v = kqv(x, kqv_matrix)
        k_list.append(k)
        q_list.append(q)
        v_list.append(v)

    k = torch.stack(k_list, dim=-3)  # should be of n_heads * N * D
    q = torch.stack(q_list, dim=-3)  # should be of n_heads * N * D
    v = torch.stack(v_list, dim=-3)  # should be of n_heads * N * (D/n_heads)
    att = attention_scores(k, q)
    sa = self_attention(v, att, mask, dropout, dropout_module)  # sa is of dim n_heads * N * (D / n_heads)
    sa = sa.transpose(-2, -3)  # sa is of N * n_heads * (D / n_heads)
    sa = sa.flatten(start_dim=-2)  # sa is N * D
    assert sa.size() == x.size()
    return sa


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len, dropout=False):
        super().__init__()
        assert embed_dim % n_heads == 0
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for i in range(n_heads)])
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.dropout_module1 = None
        self.dropout_module2 = None
        if self.dropout:
            self.dropout_module1 = nn.Dropout(0.1)
            self.dropout_module2 = nn.Dropout(0.1)

    def forward(self, x):
        sa = multi_head_attention_layer(x, self.kqv_matrices, self.mask, self.dropout, self.dropout_module1)
        if self.dropout:
            sa = self.dropout_module2(sa)
        sa = self.proj(sa)
        return sa
