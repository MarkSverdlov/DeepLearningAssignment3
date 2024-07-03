from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len).to(device)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size).to(device)
        self.layer_norm_1 = nn.LayerNorm(embed_size).to(device)
        self.layer_norm_2 = nn.LayerNorm(embed_size).to(device)
        self.with_residuals = with_residuals

    def forward(self, inputs):
        if self.with_residuals:
            # TODO add residuals support.
            x = inputs
            residual1 = x
            residual1 = self.layer_norm_1(residual1)
            residual1 = self.causal_attention(residual1)
            x = x + residual1
            residual2 = x
            residual2 = self.layer_norm_2(residual2)
            residual2 = self.mlp(residual2)
            x = x + residual2
            return x
        else:
            x = inputs
            x = self.layer_norm_1(x)
            x = self.causal_attention(x)
            x = self.layer_norm_2(x)
            x = self.mlp(x)
            return x


class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size).to(device) # TODO set the right values
        self.position_embeddings = nn.Embedding(max_context_len, embed_size).to(device) # TODO set the right values
        self.max_context_len = max_context_len

    def forward(self, x):
        # x has the shape (b x n) where b is batch dimension and n is sequence length.
        # each item is an int, indicating a vocabulary item.
        # The output should be of shape (b x n x d), where d is the embedding dimension.
        tok_embeddings = self.token_embeddings(x)  # b x n x d
        pos_embeddings = self.position_embeddings(torch.arange(x.size()[-1], device=device))  # n x d
        return tok_embeddings + pos_embeddings  # broadcasting handles it.


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            ):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len).to(device)
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals).to(device) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size).to(device)
        self.word_prediction = nn.Linear(embed_size, vocab_size).to(device)
        self.max_context_len = max_context_len

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs):
        x = self.embed(inputs)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        return logits

    def init_weights(self):
        # initialize weights
        # TODO implement initialization logic for embeddings and linear layers.
        # The code break down the parameters by type (layer-norm, linear, embedding),
        # but can also condition on individual names, for example by checking pn.endswith(...).
        for pn, p in self.named_parameters():
            if isinstance(p, nn.LayerNorm):
                torch.nn.init.zeros_(p.bias)
                torch.nn.init.ones_(p.weight)
            elif isinstance(p, nn.Linear):
                # TODO initialize p.weight and p.bias (if it is not None).
                # You can look at initializers in torch.nn.init
                pass
            elif isinstance(p, nn.Embedding):
                # TODO initialize p.weight and p.bias (if it is not None).
                # You can look at initializers in torch.nn.init
                pass


    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32))
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
        raise Exception("Not implemented")
        # TODO implement this.
        # Temperature should be the temperature in which you sample.
        # TopK indicates that we don't sample from the entire distribution, but only from the top k scoring tokens
        # for the given position.

