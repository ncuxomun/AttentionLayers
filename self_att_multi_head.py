#%%
sentence = 'Life is short, eat dessert first'

dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}

print(dc)
# %%
import torch

sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()]
)
print(sentence_int)
# %%
vocab_size = 50_000

torch.manual_seed(123)
embed = torch.nn.Embedding(vocab_size, 3)
embedded_sentence = embed(sentence_int).detach()


print(embedded_sentence)
print(embedded_sentence.shape)
# %%
d = embedded_sentence.shape[1]
d_q, d_k, d_v = 2, 2, 4

W_query = torch.nn.Parameter(torch.rand(d, d_q))
W_key = torch.nn.Parameter(torch.rand(d, d_k))
W_value = torch.nn.Parameter(torch.rand(d, d_v))

# %%
x_2 = embedded_sentence[1]
query_2 = W_query @ x_2
key_2 = W_key @ x_2
value_2 = W_value @ x_2

print(query_2.shape)
print(key_2.shape)
print(value_2.shape)
# %%
# We can then generalize this to compute the remaining key, and value elements for all inputs as well,
# since we will need them in the next step when we compute the unnormalized attention weights later:
keys = embedded_sentence @ W_key
values = embedded_sentence @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
# %%
omega_24 = query_2.dot(keys[4])
print(omega_24)
# %%
omega_2 = query_2 @ keys.T
print(omega_2)
# %%
import torch.nn.functional as F

attention_weights_2 = F.softmax(omega_2 / d_v**0.5, dim=0)
print(attention_weights_2)
# %%
context_vector_2 = attention_weights_2 @ values

print(context_vector_2.shape)
print(context_vector_2)
# %%
import torch.nn as nn
torch.manual_seed(123)

#* Self-Attention Block

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v, causal=False):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))
        
        # causal if mask enabled for GPT-like attention
        self.causal = causal

    def forward(self, x):
        Keys = x @ self.W_key
        Queries = x @ self.W_query
        Values = x @ self.W_value
        
        # unnormalized attention weights    
        attn_matrix = Queries @ Keys.T
        
        # applying mask if needed
        if self.causal:
            block_size = attn_matrix.shape[0] # input size
            mask = torch.triu(torch.ones((block_size, block_size)), diagonal=1)
            masked = attn_matrix.masked_fill(mask.bool(), -torch.inf)
            
            # MASKED normalized attention weights 
            attn_weights = torch.softmax(masked / self.d_out_kq**0.5, dim=1)
        else:
            # normalized attention weights 
            attn_weights = torch.softmax(attn_matrix / self.d_out_kq**0.5, dim=-1)

        context_vec = attn_weights @ Values
        return context_vec
# %%
torch.manual_seed(123)

# reduce d_out_v from 4 to 1, because we have 4 heads
d_in, d_out_kq, d_out_v = 3, 2, 4

sa = SelfAttention(d_in, d_out_kq, d_out_v, causal=True)
print(sa(embedded_sentence))
# %%
#* MultiHeadAttentionWrapper
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out_kq, d_out_v, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(d_in, d_out_kq, d_out_v) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
# %%
torch.manual_seed(123)

d_in, d_out_kq, d_out_v = 3, 2, 1

sa = SelfAttention(d_in, d_out_kq, d_out_v)
print(sa(embedded_sentence))
# %%
# Now, let's extend this to 4 attention heads:
torch.manual_seed(123)

block_size = embedded_sentence.shape[1]
mha = MultiHeadAttentionWrapper(d_in, d_out_kq, d_out_v, num_heads=4)

context_vecs = mha(embedded_sentence)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
# %%
#* CrossAttention Layer
class CrossAttention(nn.Module):

    def __init__(self, d_in, d_out_kq, d_out_v, causal=False):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))
        
        # causal if mask enabled for GPT-like attention
        self.causal = causal

    def forward(self, x_1, x_2):           # x_2 is new
        Queries_1 = x_1 @ self.W_query
        
        Keys_2 = x_2 @ self.W_key          # new
        Values_2 = x_2 @ self.W_value      # new
        
        # CrossAttention Matrix
        cAttn_matrix = Queries_1 @ Keys_2.T # new
        # applying mask if needed
        if self.causal:
            mask = torch.triu(torch.ones_like(cAttn_matrix), diagonal=1)
            masked = cAttn_matrix.masked_fill(mask.bool(), -torch.inf)
            # MASKED normalized attention weights 
            cAttn_weights = torch.softmax(masked / self.d_out_kq**0.5, dim=1)
        else:
            # normalized attention weights 
            cAttn_weights = torch.softmax(cAttn_matrix / self.d_out_kq**0.5, dim=-1)

        # CrossAttention Context Vector
        cContext_vec = cAttn_weights @ Values_2
        return cContext_vec

# %%
torch.manual_seed(123)

d_in, d_out_kq, d_out_v = 3, 2, 4

crossattn = CrossAttention(d_in, d_out_kq, d_out_v, causal=False)

first_input = embedded_sentence
second_input = torch.rand(8, d_in)

print("First input shape:", first_input.shape)
print("Second input shape:", second_input.shape)
# %%
context_vectors = crossattn(first_input, second_input)

print(context_vectors)
print("Output shape:", context_vectors.shape)
# %%
