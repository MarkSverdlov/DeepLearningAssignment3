import torch
import attention


def test_attention_scores():
    # fill in values for the a, b and expected_output tensor.
    a = torch.tensor([
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0, -1, -0.1, 2]
        ]])  # a three-dim tensor
    b = torch.tensor([
        [
            [1, 2, 2, 3],
            [5, 4, 3, 2]
        ]
    ])  # a three-dim tensor
    expected_output = torch.tensor([
        [
            [11.5, 15.5, 1.9],
            [15, 22, -0.15]
        ]
    ]) # a three-dim tensor

    A = attention.attention_scores(a, b)

    # Note that we use "allclose" and not ==, so we are less sensitive to float inaccuracies.
    assert torch.allclose(A, expected_output)


def test_full_self_attention():
    a = torch.tensor([
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0, -1, -0.1, 2]
        ]])  # a three-dim tensor
    b = torch.tensor([
        [
            [1, 2, 2, 3],
            [5, 4, 3, 2]
        ]
    ])  # a three-dim tensor

    A = attention.attention_scores(a, b)

    v = torch.tensor([
        [
            [2, 5, 4, 1, 2, 1],
            [5, 4, 1, 0, -0.2, 1],
            [6, 1, 1, 1, 0, 0]
        ]
    ])

    sa = attention.self_attention(v, A)
    
    expected_output = torch.tensor([
        [
            [4.9460e+00,  4.0180e+00,  1.0540e+00,  1.7987e-02, -1.6043e-01, 1.0000e+00],
            [4.9973e+00,  4.0009e+00,  1.0027e+00,  9.1105e-04, -1.9800e-01, 1.0000e+00]
        ]
    ])

    assert torch.allclose(sa, expected_output, atol=1e-04)


def test_casual_self_attention():
    a = torch.tensor([
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0, -1, -0.1, 2]
        ]])  # a three-dim tensor
    b = torch.tensor([
        [
            [1, 2, 2, 3],
            [5, 4, 3, 2]
        ]
    ])  # a three-dim tensor

    M = attention.create_causal_mask(0, 0, 10)

    v = torch.tensor([
        [
            [2, 5, 4, 1, 2, 1],
            [5, 4, 1, 0, -0.2, 1],
            [6, 1, 1, 1, 0, 0]
        ]
    ])

    A = attention.attention_scores(a, b)

    sa = attention.self_attention(v, A, mask=M)

    expected_output = torch.tensor([
        [
            [2, 5, 4, 1, 2, 1],
            [4.9973, 4.0009, 1.0027, 9.1105e-04, -1.9800e-01, 1]
        ]
    ])

    assert torch.allclose(sa, expected_output, atol=1e-04)

def test_multihead_attention():
    n_heads = 3
    dim_em = 6
    kqv_matrices = torch.nn.ModuleList([attention.create_kqv_matrix(dim_em, n_heads) for _ in range(n_heads)])

    N = 10
    B = 3
    x = torch.rand(B, N, dim_em)

    M = attention.create_causal_mask(dim_em, n_heads, 20)
    print()
    print(attention.multi_head_attention_layer(x, kqv_matrices, M).size())
