---
layout: post
title:  "Optimising the Training Loop"
date:   2024-05-29 18:33:45 -0400
categories: AL ML LLM
---
The diagram below is a chart from the paper Scaling Laws for Neural Language Models by OpenAI. The paper examines the test performance of a trained Language Model in relation to the model's size, data volume, training duration, and model shape.

The chart on the right has the horizontal axis in PF-days (1 PF-day is equal to \(8.64 \times 10^{19}\) FLOPS (FLoating point Operations Per Second)). The vertical axis shows the test loss which is categorical cross entropy between predictions and ground truth from the test dataset. 

The chart shows that models with fewer parameters converge at a higher test loss, irrespective of how long you train them. It is preferable (in terms of test loss) to use a larger model and stop early (before convergence) rather than training a smaller model to convergence. To reach a test loss of 4 or below, we need a minimum of \(10^{-3}\) PF-days with a model having at least ~\(10^{8}\) parameters (100 million). For reference, GPT-2 is a 1.5 billion parameter model. When trained to convergence, it achieves a test loss of ~1.5. 

To reach the level of GPT-2 (test loss of 1.5 or below), we need at least a billion (\(10^{9}\)) parameter model trained for \(10^{0} = 1\) PF-days (\(8.64 \times 10^{19}\) FLOPS). A single A100 GPU can run at \(0.195 \times 10^{14}\) FLOPS if operating in 32 bit floating point precision. Which means, we need to train a billion parameter model (in 32 bit precision) for at least 4,430,769 seconds (approximately 7 weeks). And this assumes that we utilise the GPU at its peak performance.

Even when you're not training language models, determining the right hyperparameters for a deep learning model is an iterative process. This process involves pseudo-random exploration based on intuition, and any errors in code often fail silently. Therefore, it's crucial to run experiments quickly to save both time and money.

This means we must optimize our code to achieve the following:

1. Minimize the number of GPUs needed to train the model by reducing the memory used by the model and optimizer during training.
2. Ensure the GPUs operate near their peak capacity throughout the training.

In this post, we will explore some model architecture independent optimisation techniques that will enable us to train almost 10 times faster with 10 times lower memory footprint.

To ground the discussion, we shall use a small, GPT style, decoder only transformer model with approximately three million parameters and a reference task of next token prediction with teacher forcing. This wil let us use profiling tools to accurately measure the performance improvements corresponding to every change. Yet, the techniques we will demonstrate in this post are applicable to any deep learning model/task. We will learn the implementation and optimisation of transformer specific model architectures in the next post. Also, we will be using PyTorch throughout the discussion.

Let's start by implementing a typical training loop. 

{% highlight python %}
# Copy model to correct device
model = model.to(device)
for epoch in range(num_epochs):
  for batch, (inputs, targets) in enumerate(train_dataloader):
    # Copy tensors to correct device
    inputs, targets = inputs.to(device), targets.to(device)
    # Forward pass
    preds = model(inputs, train=True)
    # Compute loss
    loss = cross_entropy(
        preds.view(-1, preds.size(-1)),
        targets.view(-1),
        ignore_index=PADDING_TOKEN_ID,
        reduction="mean",
    )
    # Backward pass to compute and accumulate gradients
    loss.backward()
    print(f"Loss: {loss.item()}")
    # Descend the gradient one step
    optimizer.step()
    # Set the stored gradients to 0
optimizer.zero_grad()
{% endhighlight %}

Here `targets` and `inputs` are a batch of sequences of token ids having the shape `[batch_size, context_length, 1]`. For next token prediction with teacher forcing, the target sequences are just input sequences shifted by one token. In case the sequence length is less than `context_length`, the rest of the positions are filled with `PADDING_TOKEN_ID`. The predictions `preds` are logits of shape `[batch_size, context_length, vocabulary_size]` . The cross entropy is computed independently for each pair of predicted token probabilities and target token id and then averaged across the batch and sequence dimensions. Therefore, the predictions can be flattened into shape `[batch_size * context_length, vocabulary_size]` and the targets can be flattened into shape `[batch_size * context_length, 1]` . Refer to the documentation of [torch.nn.functional.cross_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) for further details on the expected inputs.

As we shall see in the next post, the model will operate on all tokens and make predictions for all positions (even the positions filled with `PADDING_TOKEN_ID` ). Instead, the cross entropy loss function will ignore the positions where original sequence had the `PADDING_TOKEN_ID`. Therefore, there's no need to mask the inputs or targets during the forward pass, and yet the padded positions do not contribute to the final loss.

Note that in this case, the device has to store at least -

| Tensors | Number of values to store is proportional to |
| --- | --- |
| inputs | batch_size * context_length * 1 |
| targets | batch_size * context_length * 1 |
| preds  | batch_size * context_length * vocabulary_size |
| activations | depends on model architecture and its computation graph |
| gradients | num_params |
| model parameters | num_params |
| Optimiser states like momentum | num_params * num_states_per_parameter |

Note that, for a 117 million parameter GPT-2-small model, the gradients, model parameters and optimiser states will consume just \(4 \times 117 \times 10^6 \times 3 = 1.4\text{ Gb}\) of memory (assuming one state number per parameter for optimiser and all numbers being stored in 32 bit floating point precision). The data consumes a negligible amount in comparison. 

Yet, while training with a single GPU of 32 Gb memory capacity, the program will throw out of memory exceptions. This is because storing the activations can require an order of magnitude more memory than the rest of the tensors combined.

The number of activation values stored during forward pass is difficult to compute. In automatic differentiation, during forward pass, every operation needs to store some data it would need for computing gradients during backward pass using chain rule. For example, in case of matrix multiplication of an input matrix with a weight matrix, automatic differentiation needs to store the input matrix to compute gradient of output values with respect to the weight matrix values during backward pass. In case of activation like softmax, we need to store the output of softmax to compute the gradient of output values with respect to the input matrix values. The number of activation values stored depends on the operations in computation graph and the input/output shapes of each.

In case of transformers, the paper [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198) by Nvidia gives a good formula for estimating the activation memory of a transformer block. According to that paper, a single standard transformer block using 32-bit floating point numbers requires
\[
2sbh\left(34 + \frac{5as}{h}\right)
\]

where \(s\) is the context length, \(h\) is the embedding dimension, \(b\) is the batch size and \(a\) is the number of attention heads.

The GPT-2-small [configuration](https://github.com/openai/gpt-2/blob/master/src/model.py) uses 12 transformer blocks with \(s = 1024\), \(h = 768\), \(b = 32\), and \(a = 12\). Which means the peak memory usage should be close to 69 Gb! The initial learnt positional embeddings and word embeddings (shared with final projection layer) would use a few aditional Mbs.

By the end of this post, we should be able to train a GPT-2-small sized model (or even larger) on a single GPU almost 10 times faster than a naive PyTorch training loop implemented above.

In order to iterate and profile faster, we shall use a smaller reference architecture with just four transformer blocks having \(s = 512\), \(h = 256\), \(b = 32\), and \(a = 4\). With this model configuration, our peak memory usage for the naive version should be close to 2.5 Gb.

Lets start with basic memory optimizations. First, `optimizer.zero_grad()` allows a parameter `set_to_none`. When `set_to_none` is `True`, the gradient is set to `None` instead of a tensor of `0`s thus saving a modest amount of memory. But it still does not change the peak memory usage. You only need to exceed memory once for the script to crash.

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-title.MARKUP`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `MARKUP` is the file extension representing the format used in the file. After that, include the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
