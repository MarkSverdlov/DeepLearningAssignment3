import pandas as pd
import torch
from attention import attention_scores
from attention import kqv, create_causal_mask
from transformer import TransformerLM
from data import DataFeeder
import numpy as np
import plotly.express as px


def get_attention_scores(x, model, feeder, max_context_len):
    model.eval()
    ex = feeder.tokenizer.tokenize(x)
    ex = torch.tensor([ex], dtype=torch.int32, requires_grad=False)
    ex = model.embed(ex)
    ex = model.layers[0].layer_norm_1(ex)
    kqv_matrices = model.layers[0].causal_attention.kqv_matrices
    k_list = []
    q_list = []
    v_list = []
    for kqv_matrix in kqv_matrices:
        k, q, v = kqv(ex, kqv_matrix)
        k_list.append(k)
        q_list.append(q)
        v_list.append(v)

    k = torch.stack(k_list, dim=-3)  # should be of n_heads * N * D
    q = torch.stack(q_list, dim=-3)  # should be of n_heads * N * D
    v = torch.stack(v_list, dim=-3)  # should be of n_heads * N * (D/n_heads)
    att = attention_scores(k, q)

    att = att.squeeze(dim=0)
    att = torch.split(att, [1] * model.layers[0].causal_attention.n_heads, dim=0)
    mask = create_causal_mask(0, 0, max_context_len)
    dfs = []
    for matrix in att:
        matrix = matrix.squeeze(dim=0)
        matrix = matrix.masked_fill(mask[..., :matrix.shape[-2], :matrix.size()[-1]] == 0, float('-inf'))
        matrix = torch.nn.functional.softmax(matrix, dim=-1)
        df = pd.DataFrame(matrix.squeeze(dim=0).detach().numpy(), index=np.arange(len(x)), columns=list(x))
        dfs.append(df)
    return dfs


if __name__ == "__main__":
    feeder = DataFeeder(128, "data")
    model = TransformerLM(6, 6, 192, 128, feeder.tokenizer.vocab_size(), 192, True, dropout=False)
    df = get_attention_scores('hello', model, feeder, 128)[0].head()
    print(df)
    px.imshow(df).show()
