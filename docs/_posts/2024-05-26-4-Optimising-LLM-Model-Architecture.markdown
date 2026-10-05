---
layout: post
title: "4. Optimising LLM Model Architecture"
posted: "June 12, 2024"
categories: Super-Fast-LLM-Training
live: true
---

In the previous post, we saw how to optimise a generic training loop for large deep learning models. In this post, we shall implement a GPT-style decoder-only transformer model (most common large language model architecture) and explore some model architecture specific optimisations.

Although Large Language Models (LLMs) come with millions or even billions of parameters and exceptional natural language generation capabilities, their model architecture isn't as complex as it might seem. Most popular LLMs use a model architecture known as Transformer. A Transformer is really just a combination of a few attention layers (specifically, self-attention layers for GPT-like decoder-only models), some normalisation layers (like batch normalisation and layer normalisation), few multi-layer perceptrons and some residual connections. That's all there is to it. The paper that introduced the Transformer family of architectures was aptly named "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)".

The same attention layer is also the bottleneck for memory and compute efficiency of transformers. Therefore, in this post, a disproportionate amount of attention will be given to optimising them.

## Model Architecture

Let's start by implementing the traditional GPT style decoder only transformer architecture. We shall name this model EduLLM. In this initial implementation, we shall focus on readability and then, like previous section, optimise the architecture step by step.

At a high level, the decoder looks like the accompanying diagram taken from the [original GPT paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). It shows learnt text and positional embeddings followed by a stack of 12 blocks called the transformer layers. Each transformer block consists of a couple of layer normalisation layers, a multi layer perceptron and a masked multi-head self attention. The block also has some residual connections.

The [GPT 2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) modifies this architecture by moving the layer normalisation to the beginning of the transformer block and adding another layer normalisation layer after the last transformer block. In this post, we shall be following those modifications.

![GPT decoder-only transformer architecture](/assets/images/Optimising_LLM_Model_Architecture/1.png)

Let's start building in a top-down manner by first defining the model `EduLLM` comprising of embedding layers, stack of transformer blocks and final projection layers. We shall then define the transformer block, and then the masked multi-head attention layers. The inputs to the model are number of transformer layers, number of attention heads for multi headed attention, embedding dimension and vocabulary size.

```python
import torch
from torch import nn
from layers.transformer_block import TransformerBlock

class EduLLM(nn.Module):
    def __init__(self, n_layers: int, num_heads: int, embedding_dimension: int, vocabulary_size: int, context_length: int):
        super().__init__()
        # Learnt token embeddings.
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        # Learnt positional embeddings.
        self.positional_embedding = nn.Embedding(context_length, embedding_dimension)
        # Sequence layers of transformer blocks.
        self.transformer = nn.ModuleList(
            [TransformerBlock(num_heads, embedding_dimension, True)
            for _ in range(n_layers)])
        # Final layer normalisation.
        self.ln = nn.LayerNorm(embedding_dimension)
        # Projection layer to map embeddings to vocabulary size.
        self.head = nn.Linear(embedding_dimension, vocabulary_size, bias=False)
        # Tie token embedding and final projection layer weights.
        self.token_embedding.weight = self.head.weight
        # Recursively apply weight initialization.
        self.apply(self._initialize_weights)

    def forward(self, x, train: bool = True):
        [batch_size, context_length] = x.shape
        # Create positions array on same device as input tensor
        device = x.device
        positions = torch.arange(0, context_length, step=1, device=device, requires_grad=False) # [context_length]
        pe = self.positional_embedding(positions) # [context_length, embedding_dimension]
        te = self.token_embedding(x) # [batch_size, context_length, embedding_dimension]
        x = te + pe # [batch_size, context_length, embedding_dimension]
        for transformer_block in self.transformer:
            x = transformer_block(x, train) # [batch_size, context_length, embedding_dimension]
        x = self.ln(x) # [batch_size, context_length, embedding_dimension]
        x = self.head(x) # [batch_size, context_length, vocabulary_size]
        return x

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Some linear layers do not have bias.
            # For example embeddings and head.
            if module.bias is not None:
                nn.init.zeros_(module.bias)
```

We will define `TransformerBlock` in the next section. Comments show the output shapes after each transformation in the `forward` method. Some things to note here are -

1. The inputs to positional embedding layer (just a fixed sequence of numbers from 0 to `context_length`) must be created on the same device as other tensors and should not require gradients as these are not parameters to the model.
2. The projection layer should not have a bias term.
3. The `forward` method accepts boolean parameter train that indicates whether to use dropout or not inside transformer block. Dropout is not used during inference.
4. The token embedding weight matrix of shape `[vocabulary_size, embedding_dimension]` is shared with head projection layer weight matrix of shape `[embedding_dimension, vocabulary_size]`. The technique reduces the number of parameters without compromising on the quality. Note that the `self.token_embedding.weight.shape` is `[vocabulary_size, embedding_dimension]` and `self.head.weight.shape` too is `[vocabulary_size, embedding_dimension]` (in PyTorch, the [Linear layer](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) transposes its weight matrix before batch multiplying with its inputs: $y = xA^T + b$). Therefore, we can simply point one to the other as `self.token_embedding.weight = self.head.weight`. For more details, refer to an [official example](https://github.com/pytorch/examples/blob/2d725b6ab255e05c55e0b08925f06f171aaedc0c/word_language_model/model.py#L25) of weight tying used to tie encoder and decoder weights.
5. The default PyTorch constructor for `nn.Linear` initializes weights as uniform random numbers between a range that depends on shape of input. To recursively apply a different weight initialization strategy, we first define a function that accepts a module and initializes its weights according to the desired strategy. Then in the model's constructor, we use `nn.Module`'s [apply](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.apply) method to apply this function recursively to all sub-modules starting from `EduLLM`.

## Transformer Block

The following code shows a single transformer block. The inputs to the transformer block are either the embedded sequence or outputs of previous transformer block. The inputs and outputs are of shape `[batch_size, context_length, embedding_dimension]`.

```python
import torch
from torch import nn
from layers.masked_self_attention import MultiHeadMaskedSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int, embedding_dimension: int, causal: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dimension)
        self.attention = MultiHeadMaskedSelfAttention(num_heads, embedding_dimension, causal)
        self.attention_dropout = nn.Dropout(0.1)
        self.ln2 = nn.LayerNorm(embedding_dimension)
        self.linear = nn.Linear(embedding_dimension, 4 * embedding_dimension)
        self.act = nn.GELU()
        self.projection = nn.Linear(4 * embedding_dimension, embedding_dimension)
        self.projection_dropout = nn.Dropout(0.1)

    def forward(self, x, train: bool = True):
        # Layer normalisation
        y = self.ln1(x) # [batch_size, context_length, embedding_dimension]
        # Multi headed attention
        y = self.attention(y) # [batch_size, context_length, embedding_dimension]
        # Dropout
        if train:
            y = self.attention_dropout(y) # [batch_size, context_length, embedding_dimension]
        # Residual connection
        y = y + x # [batch_size, context_length, embedding_dimension]
        # Layer normalisation
        z = self.ln2(y) # [batch_size, context_length, embedding_dimension]
        # Multi layer perceptron
        z = self.linear(z) # [batch_size, context_length, embedding_dimension * 4]
        z = self.act(z) # [batch_size, context_length, embedding_dimension * 4]
        z = self.projection(z) # [batch_size, context_length, embedding_dimension]
        # Dropout
        if train:
            z = self.projection_dropout(z) # [batch_size, context_length, embedding_dimension]
        # Residual connection
        return y + z # [batch_size, context_length, embedding_dimension]
```

The comments show the output shapes after every transformation.

## Multi-Head Self Attention Layer

Multi-head attention is described in the popular paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). For demonstration purposes, we shall create a `MultiHeadMaskedSelfAttention` module that holds `num_heads` independent individual attention layers (`MaskedSelfAttention`) that run sequentially. We shall later see a way to optimise this part.

`MultiHeadMaskedSelfAttention` creates multiple instances of `MaskedSelfAttention` with input dimension as `embedding_dimension` and output dimension as `embedding_dimension / num_heads`. Note that `embedding_dimension` should be an integral multiple of `num_heads` so that the outputs of individual heads can be concatenated back to return exactly `embedding_dimension` sized intermediate outputs.

```python
import torch
from torch import nn

class MultiHeadMaskedSelfAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_dimension: int, causal: bool = True):
        super().__init__()
        if embedding_dimension <= 0 or embedding_dimension % num_heads != 0:
            raise ValueError("embedding_dimension should be a multiple of num_heads")
        self.heads = nn.ModuleList(
            [MaskedSelfAttention(embedding_dimension, embedding_dimension // num_heads, causal) for _ in range(num_heads)])

    def forward(self, x):
        # x has shape [batch_size, context_length, embedding_dimension]
        attentions = [msa(x) for msa in self.heads] # A list of elements of shape [batch_size, context_length, embedding_dimension // num_heads]
        attention = torch.concatenate(attentions, dim=-1) # [batch_size, context_length, embedding_dimension]
        return attention
```

Having established the higher level architecture, we now focus on the crux of the transformer architecture - the attention layer (named `MaskedSelfAttention` in this case).

## Attention Layer

Broadly, learning with sequential data can be categorised into three major groups -

1. **Masked sequence learning** - predicting tokens at masked positions based on rest of the tokens.
2. **Next token prediction** - predicting next token based on a sequence of past tokens (of same vocabulary)
3. **Sequence to sequence translation** - predicting next token based on a sequence of past tokens (of same vocabulary) and a reference sequence of tokens (of same or different vocabulary)

All three tasks can be accomplished with Transformer based models but the distinction is important because they differ in the type of attention layer used.

An attention layer works like a simple key-value table. For a given query, the table returns the weighted average of values where weights are proportional to the similarity between the query and keys. Note the difference between this table and a standard dictionary data structure. In an attention layer, even if you pass a query that is exactly equal to a key in the table, you wont get the exact associated value of the key. Values of other nearby keys will also play a role in the result.

In attention layer, the query, keys and values are all fixed sized vectors. The similarity of a query to all keys is computed using the dot product between every query-key pair. If there are K keys in the table, you get a vector of K scores for one query (1 scalar dot product between one query vector and each key vector). These K scores are normalised using a softmax function to get an K-sized attention vector of positive values that sum to 1. The attention vector can then be used to return a weighted average of K value vectors.

In the context of deep learning, we use a learnable attention layer. Which means, queries, keys and values are not direct tokens from the sequence but rather generated using a linear transformation applied to those tokens. The parameters of this transformation are learnt during training.

Lets implement the self-attention layer with PyTorch. All types of attention layers are built into the `torch.nn` module but implementing them using basic blocks will help us understand, debug and customise the transformations even better.

```python
import torch
from torch import nn

class MaskedSelfAttention(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int, causal: bool = True):
        super().__init__()
        # Query, key and value projection layers
        self.wq = nn.Linear(input_dimension, output_dimension, bias=False)
        self.wk = nn.Linear(input_dimension, output_dimension, bias=False)
        self.wv = nn.Linear(input_dimension, output_dimension, bias=False)
        # Single non-trainable constant
        self.score_normalisation_factor = output_dimension ** 0.5
        # Softmax such that every row sums to 1
        self.softmax = nn.Softmax(dim=-1)
        # If using causal attention, a query cannot attend to
        # keys at future positions in the sequence
        self.causal = causal

    def forward(self, x):
        [batch_size, context_length, input_dimension] = x.shape
        # We create a new non-trainable matrix (additive_mask) at runtime
        # inside this function. We need to specify which device to create it on.
        device = x.device
        # Causal additive mask is constant and same for all samples
        with torch.no_grad():
            additive_mask = torch.zeros((context_length, context_length), device=device) # [context_length, context_length]
            if self.causal:
                additive_mask = torch.triu(
                    torch.full((context_length, context_length), float('-inf'), device=device),
                    diagonal=1) # [context_length, context_length]

        query = self.wq(x) # [batch_size, context_length, output_dimension]
        key = self.wk(x) # [batch_size, context_length, output_dimension]
        value = self.wv(x) # [batch_size, context_length, output_dimension]
        key_transpose = torch.transpose(key, 1, 2) # [batch_size, output_dimension, context_length]
        scores = torch.bmm(query, key_transpose) # [batch_size, context_length, context_length]
        masked_scores = scores + additive_mask # additive_mask is broadcasted to [batch_size, context_length, context_length]
        normalised_scores = masked_scores / self.score_normalisation_factor # [batch_size, context_length, context_length]
        attention_probabilities = self.softmax(normalised_scores) # [batch_size, context_length, context_length] where each row sums to 1.0
        self_attention = torch.bmm(attention_probabilities, value) # [batch_size, context_length, output_dimension]
        return self_attention
```

The scores are divided by the square root of the output dimension, as in the paper. A dot product of two vectors becomes larger as the vectors get longer. Large scores push the softmax output towards a one-hot vector, and the gradients of such softmax outputs are very small. The division keeps the scores in a range where the model can still learn.

The additive mask has `-inf` above the diagonal. After the softmax, these positions get a probability of exactly 0. Therefore, a token at position i only takes values from tokens at positions 0 to i.

## Profiling

With the model architecture setup, lets profile the forward pass. Below chart is the zoomed in version of one profiler step (ProfilerStep#14), with the same profiling setup as previous post. The view is zoomed in to just the forward pass (all operations shown are part of nn.Module: EduLLM_0 on level 4).

![Trace of the forward pass with sequential attention heads](/assets/images/Optimising_LLM_Model_Architecture/2.png)

The time taken to run MultiHeadMaskedSelfAttention of one transformer block (TransformerBlock_0) is 2.261ms. One clearly visible inefficiency is the part where the four attention heads (nn.Module: MaskedSelfAttention) run sequentially. So let's modify that to build a parallel multi-head attention.

```python
import torch
from torch import nn

class EfficientMultiHeadMaskedSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dimension: int,
        causal: bool = True
    ):
        super().__init__()
        assert embedding_dimension % num_heads == 0
        self.num_heads = num_heads
        self.head_dimension = embedding_dimension // num_heads
        self.wqkv = nn.Linear(embedding_dimension, 3 * embedding_dimension, bias=False)
        self.score_normalisation_factor = self.head_dimension ** 0.5
        self.softmax = nn.Softmax(dim=-1)
        self.causal = causal

    def forward(self, x):
        [batch_size, context_length, input_dimension] = x.shape
        device = x.device
        with torch.no_grad():
            additive_mask = torch.zeros(
                (context_length, context_length), device=device
            )
            if self.causal:
                additive_mask = torch.triu(
                    torch.full(
                        (context_length, context_length),
                        float("-inf"),
                        device=device,
                    ),
                    diagonal=1,
                )

        qkv = self.wqkv(x).view(
            (batch_size, context_length, self.num_heads, -1)
        )  # [batch_size, context_length, num_heads, 3*head_dimension]
        qkv = qkv.permute(
            (0, 2, 1, 3)
        )  # [batch_size, num_heads, context_length, 3*head_dimension]
        qkv = qkv.reshape(
            (batch_size * self.num_heads, context_length, -1)
        )  # [batch_size*num_heads, context_length, 3*head_dimension]
        query = qkv[:, :, : self.head_dimension]
        key = qkv[:, :, self.head_dimension : 2 * self.head_dimension]
        value = qkv[:, :, 2 * self.head_dimension :]
        key_transpose = torch.transpose(
            key, 1, 2
        )  # [batch_size*num_heads, head_dimension, context_length]
        scores = torch.bmm(
            query, key_transpose
        )  # [batch_size*num_heads, context_length, context_length]
        masked_scores = scores + additive_mask
        normalised_scores = (
            masked_scores / self.score_normalisation_factor
        )  # [batch_size*num_heads, context_length, context_length]
        attention_probabilities = self.softmax(
            normalised_scores
        )  # [batch_size*num_heads, context_length, context_length] where each row sums to 1.0
        self_attention = torch.bmm(attention_probabilities, value)

        return (
            self_attention.view((batch_size, self.num_heads, context_length, -1))
            .permute((0, 2, 1, 3))
            .reshape((batch_size, context_length, -1))
        )
```

All query, key and value projections of all heads are now a single linear layer `wqkv`. The `view` call splits the output of `wqkv` into `num_heads` chunks. Each chunk holds the query, key and value of one head, one after the other. We then move the heads into the batch dimension, so that a single `torch.bmm` call computes the scores of all heads in all samples.

This is only a change in how we arrange the computation. The layer must compute exactly the same thing as `MultiHeadMaskedSelfAttention`. We can check this. We copy the weights of the four sequential heads into `wqkv` in the same order and compare the outputs.

```python
seq = MultiHeadMaskedSelfAttention(num_heads=4, embedding_dimension=128)
eff = EfficientMultiHeadMaskedSelfAttention(num_heads=4, embedding_dimension=128)
with torch.no_grad():
    # For every head, stack query, key and value weights. Then stack all heads.
    w = torch.cat([torch.cat([h.wq.weight, h.wk.weight, h.wv.weight], 0) for h in seq.heads], 0)
    eff.wqkv.weight.copy_(w)

x = torch.randn(2, 512, 128)
print((seq(x) - eff(x)).abs().max().item())
# 0.0
```

The outputs are exactly equal. We can also check that the causal mask works. If we change only the last token of the input, the outputs at all the earlier positions must not change.

```python
x2 = x.clone()
x2[:, -1] += 10
print((eff(x)[:, :-1] - eff(x2)[:, :-1]).abs().max().item())
# 0.0
```

The chart below shows the zoomed in trace of EfficientMultiHeadMaskedSelfAttention within same layer (TransformerBlock_0). Note that the wall clock time of EfficientMultiHeadMaskedSelfAttention has dropped to 0.578ms!

![Trace of the forward pass with parallel attention heads](/assets/images/Optimising_LLM_Model_Architecture/3.png)

The charts below shows the effect of the modification. The total training time for eight micro batches drops to half! With sequential MultiHeadMaskedSelfAttention based transformer block, it took ~250ms. While with parallel EfficientMultiHeadMaskedSelfAttention based transformer block, it took ~125ms.

![Memory timeline with sequential attention heads](/assets/images/Optimising_LLM_Model_Architecture/4.png)

![Memory timeline with parallel attention heads](/assets/images/Optimising_LLM_Model_Architecture/5.png)

Note that (for reasons unknown) the profiler is not attributing memory correctly and often activation memory is attributed to other category. So, ignore the red vs. gray colors - both correspond to activation memory.

All the optimizations we explored till now had practically no effect on the final quality of the trained model (therefore we did not consider that as a factor). These were just engineering tricks and the model computes the same equations with same numbers.

But at this point, we shall explore some architectural changes that will allow us to be even more memory and compute efficient. These architectural changes will affect the model's computation graph and should be evaluated by training the model and evaluating its quality too.

## Query-Key Weight Sharing

We previously saw that the token embedding layer and the final projection layer share the same weights. On similar lines, for self attention, we can have key and query projection layers share the same weights too. The approach does not lead to any significant compute improvements (since we do need to make the same operations) but saves a modest amount of memory. The approach was proposed in [Reformer](https://arxiv.org/abs/2001.04451) (Kitaev et al., 2020), where the authors report that shared query-key attention performs about as well as standard attention.

To achieve this, we just change

```python
self.wqkv = nn.Linear(embedding_dimension, 3 * embedding_dimension, bias=False)
```

to

```python
self.wqkv = nn.Linear(embedding_dimension, 2 * embedding_dimension, bias=False)
```

and then, in the forward method, modify

```python
query = qkv[:, :, : self.head_dimension]
key = qkv[:, :, self.head_dimension : 2 * self.head_dimension]
value = qkv[:, :, 2 * self.head_dimension :]
```

to

```python
query = qkv[:, :, : self.head_dimension]
key = query
value = qkv[:, :, self.head_dimension :]
```

For an embedding dimension of 128, the parameters of one attention layer drop from 49,152 to 32,768.

There is one side effect. With a shared projection, the score of a token with itself is the dot product of a vector with itself. This is usually larger than its score with any other token. So every token tends to attend mostly to itself. Reformer handles this by normalising the keys and by not letting a token attend to itself, except when there is no other token to attend to (the first token of the sequence).

This will save only a small amount of memory consumed during training as parameters anyways are a small part of overall memory consumption while training.

Note that it is always better to use [scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) function in PyTorch directly as that has more optimisations that will save you a lot of memory and I/O. The library functions are often optimised using special fused kernels. We shall learn about those optimisations too in a later post.

## Efficient Alternatives for Original Attention

Let's explore the alternative proposed by [Linformer](https://arxiv.org/abs/2006.04768). The assumption in Linformer is that the `[context_length, context_length]` shaped scores matrix is actually just dominated by a few values. In other words, the matrix is approximately low rank. Most of its information can be kept in a much smaller matrix.

Linformer adds two learnt projection matrices $E$ and $F$, each of shape `[k, context_length]`. Here k is a fixed projection dimension (for example 128) and does not depend on the context length. Before computing the scores, $E$ projects the keys and $F$ projects the values along the sequence dimension. One attention head then becomes

$$\text{head} = \text{softmax}\left(\frac{Q(EK)^T}{\sqrt{d}}\right)FV$$

$Q$ is of shape `[context_length, d]` but $EK$ and $FV$ are only of shape `[k, d]`. So the scores matrix is `[context_length, k]` instead of `[context_length, context_length]`.

Let's validate the hypothesis for our case. For this illustration, we trained a small model for just 5 epochs on the food recipes dataset. Here is a completion from the partially trained small model.

Input: "Title: Tamarind"

Completion:

```
Title: Tamarind Sim Recipe

Ingredients: (US oil
5 cups cooked warm marin peppering juice (evilal water) olive oil seeds (about fish)
4 tablespoons half extracrapean

Directions:
Combine the onions, pepper.
Cover and stir well or medium bowl until ball.
Whisked, the mash en in the little a att<END>
```

The diagram below shows one of the randomly chosen scores matrices from the last transformer block's attention layer (there are batch size * num heads number of scores matrices in every layer's attention block). The scores matrix was generated by feeding the above input and letting the model autoregressively generate till it reached the maximum context length or the end token. At this point, the scores matrices (each context length * context length in dimension) are saved. In this case, the sequence reached end token at around 380 tokens.

![Scores matrix from the last attention layer](/assets/images/Optimising_LLM_Model_Architecture/6.png)

As you can see, the model is not well trained yet and focusses only on nearby tokens while ignoring far away ones (in red). This makes it difficult to verify the claim. Having said that, it is easier to visualize how such matrix will be dominated by few large values and can be decomposed into a product of 2 smaller matrices. For intuition, imagine you are predicting the next word in a recipe and half the words from the preceding text are dropped. Will you still be able to predict the next word?

Linformer scales linearly in context length as one of the new scores matrix's dimension is independent of the context_length. Long context lengths are one of the most important feature competing LLMs tout.

Implementing Linformer is quite simple. The following code shows the modifications to the existing multi-head attention layer.

```python
class LinformerMultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_dimension: int, max_context_length: int, projection_dimension: int = 128):
        super().__init__()
        assert embedding_dimension % num_heads == 0
        self.num_heads = num_heads
        self.head_dimension = embedding_dimension // num_heads
        self.wqkv = nn.Linear(embedding_dimension, 3 * embedding_dimension, bias=False)
        # Projections along the sequence dimension. Weights are [projection_dimension, max_context_length]
        self.e = nn.Linear(max_context_length, projection_dimension, bias=False)
        self.f = nn.Linear(max_context_length, projection_dimension, bias=False)
        self.score_normalisation_factor = self.head_dimension ** 0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        [batch_size, context_length, input_dimension] = x.shape
        qkv = self.wqkv(x).view((batch_size, context_length, self.num_heads, -1))
        qkv = qkv.permute((0, 2, 1, 3))
        qkv = qkv.reshape((batch_size * self.num_heads, context_length, -1))
        query = qkv[:, :, : self.head_dimension]
        key = qkv[:, :, self.head_dimension : 2 * self.head_dimension]
        value = qkv[:, :, 2 * self.head_dimension :]
        # Shorter sequences use only the first context_length columns of E and F
        key = torch.matmul(self.e.weight[:, :context_length], key) # [batch_size*num_heads, projection_dimension, head_dimension]
        value = torch.matmul(self.f.weight[:, :context_length], value) # [batch_size*num_heads, projection_dimension, head_dimension]
        key_transpose = torch.transpose(key, 1, 2) # [batch_size*num_heads, head_dimension, projection_dimension]
        scores = torch.bmm(query, key_transpose) # [batch_size*num_heads, context_length, projection_dimension]
        normalised_scores = scores / self.score_normalisation_factor
        attention_probabilities = self.softmax(normalised_scores)
        self_attention = torch.bmm(attention_probabilities, value) # [batch_size*num_heads, context_length, head_dimension]
        return (
            self_attention.view((batch_size, self.num_heads, context_length, -1))
            .permute((0, 2, 1, 3))
            .reshape((batch_size, context_length, -1))
        )
```

Note that there is no causal mask in this code. This is not a mistake. Every row of $EK$ is a mix of keys from all positions, including future positions. There is no way to hide future tokens in a `[context_length, k]` scores matrix. Let's run the same causality test as before.

```python
linformer = LinformerMultiHeadSelfAttention(num_heads=4, embedding_dimension=128, max_context_length=512)
print((linformer(x)[:, :-1] - linformer(x2)[:, :-1]).abs().max().item())
# 0.04048096388578415
```

A change in the last token changes the outputs at all earlier positions. The Linformer paper uses this attention in encoder models like BERT, where every token is allowed to see every other token. For a decoder-only model that learns to predict the next token, this is a leak. The model can see the answer during training. Therefore, we use Linformer here only to study memory, and we do not use it for our recipe generator.

The chart below shows the memory timeline of eight micro batches with Linformer attention.

![Memory timeline with Linformer attention](/assets/images/Optimising_LLM_Model_Architecture/7.png)

The peak memory drops from 1.03 GB to 0.78 GB. But the eight micro batches now take ~680ms, compared to ~125ms with EfficientMultiHeadMaskedSelfAttention. At our context length of 512, the scores matrix was not very large to begin with, so the time we save there is small. The extra projections did not run as fast as the batched matrix multiplications of the original layer, and we did not investigate this further. The real gains of Linformer show up at much longer context lengths.

## PyTorch Multi-Head Attention

Even after all such optimisations, using [MultiHeadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) from PyTorch is still going to perform better. To use it, we replace the attention layer in the constructor of `TransformerBlock`

```python
self.attention = nn.MultiheadAttention(embedding_dimension, num_heads, bias=False, batch_first=True)
```

and change the attention call in the `forward` method

```python
context_length = x.shape[1]
causal_mask = nn.Transformer.generate_square_subsequent_mask(context_length, device=x.device)
y = self.ln1(x)
y, _ = self.attention(y, y, y, attn_mask=causal_mask, is_causal=True, need_weights=False)
```

Some things to note here are -

1. `is_causal=True` is only a hint to PyTorch. We must still pass the causal mask as `attn_mask`.
2. `need_weights=False` tells PyTorch not to return the attention probabilities. If PyTorch must return them, it cannot use its faster implementation.
3. `nn.MultiheadAttention` has an extra output projection layer (`out_proj`) after the heads are concatenated. This layer is in the original paper and in GPT 2. Our implementation does not have it.

Below is the trace of the first transformer block using `torch.nn.MultiheadAttention` for a micro batch. It shows that the MultiheadAttention_0 module took 0.421ms (compared to 0.578ms for our implementation of `EfficientMultiHeadMaskedSelfAttention`). During inference and on certain GPUs and for longer context lengths/embedding dimensions, this gain can be even larger.

![Trace of the forward pass with torch.nn.MultiheadAttention](/assets/images/Optimising_LLM_Model_Architecture/8.png)

The reason this is faster is because behind the scenes, `torch.nn.MultiheadAttention` uses [scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). This function is logically equivalent to our implementation of multi head attention. We can check this by passing the same queries, keys and values to it.

```python
import torch.nn.functional as F

qkv = eff.wqkv(x).view(2, 512, 4, 3 * 32).permute(0, 2, 1, 3) # [batch_size, num_heads, context_length, 3*head_dimension]
query, key, value = qkv.split(32, dim=-1)
sdpa = F.scaled_dot_product_attention(query, key, value, is_causal=True)
sdpa = sdpa.permute(0, 2, 1, 3).reshape(2, 512, 128)
print((eff(x) - sdpa).abs().max().item())
# 1.7881393432617188e-07
```

The difference is only a floating point rounding error. But `scaled_dot_product_attention` is implemented at a lower level where it can use an optimisation technique called kernel fusion.

When we write a series of operations in Python (like the `torch.bmm`, `+`, `/`, `softmax`, and another `torch.bmm` in our `MaskedSelfAttention` implementation), PyTorch runs each of them separately. Every operation has two costs -

1. Kernel launch overhead. The CPU must send instructions to the GPU for every operation.
2. Memory reads and writes. The GPU reads the inputs from GPU memory, does the operation and writes the result back to GPU memory. The next operation then reads that result again. In our attention layer, `scores`, `masked_scores`, `normalised_scores` and `attention_probabilities` are all of shape `[batch_size*num_heads, context_length, context_length]`. Each of them goes to GPU memory and comes back.

Kernel fusion combines many operations into one GPU kernel. For `scaled_dot_product_attention`, the query-key multiplication, scaling, masking, softmax and attention-value multiplication all run in one kernel. This makes it faster because -

1. There is one kernel launch instead of many.
2. The intermediate results stay in fast on-chip memory (shared memory and registers) and do not go back to GPU memory. On GPUs, memory reads and writes are often slower than the arithmetic itself, so this saves a lot of time.
3. NVIDIA and PyTorch developers write these kernels by hand in CUDA C++ for different GPUs and input shapes. They use techniques like tiling, which we cannot use from Python.

Our `EfficientMultiHeadMaskedSelfAttention` runs all heads in parallel. `scaled_dot_product_attention` goes one step further and also makes the sequence of operations inside the attention efficient. The difference becomes larger for larger models and longer context lengths.

We shall learn more about kernel fusion and writing CUDA kernels in another blog post. Till then, `torch.nn.MultiheadAttention` should be the most efficient implementation you would need in practice.
