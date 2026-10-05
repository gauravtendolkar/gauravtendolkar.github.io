---
layout: post
title: "5. Multi GPU Training"
posted: "June 20, 2024"
categories: Super-Fast-LLM-Training
live: true
---
In previous posts, we optimised the training loop and the model architecture for a single GPU. With gradient accumulation, mixed precision, asynchronous I/O and `torch.compile`, we could train a GPT-2-small sized model on one GPU with memory to spare. But one GPU will only take us so far.

Recall the arithmetic from the post on optimising the training loop. To reach the test loss of GPT-2, the scaling laws chart says we need a billion parameter model trained for about 1 PF-day. A single A100 running at its peak 32 bit throughput would need approximately 7 weeks for that. And that is the best case.

Memory is the second problem. For mixed precision training with the Adam optimiser, every parameter needs -

1. 2 bytes for the 16 bit copy of the parameter used in forward and backward pass.
2. 2 bytes for the 16 bit gradient.
3. 4 bytes for a 32 bit master copy of the parameter (the optimiser updates this copy).
4. 4 bytes for Adam's first moment (momentum).
5. 4 bytes for Adam's second moment (variance).

That is 16 bytes per parameter. The full 1.5 billion parameter GPT-2 model needs $16 \times 1.5 \times 10^9 = 24\text{ GB}$ just for these tensors. We have not stored a single activation yet. And as we saw in post 3, activations can need an order of magnitude more memory than everything else.

So we need more than one GPU. There are two broad ways to split the work across GPUs -

1. Split the data. Every GPU runs the full model on a different part of the batch. This is called data parallelism.
2. Split the model. Every GPU holds a part of the model. This is called model parallelism. It comes in two flavours - split each layer (tensor parallelism) or split the stack of layers (pipeline parallelism).

In this post, we shall build data parallelism from scratch with `torch.distributed`, measure what communication costs, look inside PyTorch's `DistributedDataParallel`, and then remove the memory waste of data parallelism with ZeRO and FSDP. We will finish with a small tensor parallel demo and a look at pipeline parallelism.

A note on hardware. I do not have a multi GPU machine at hand for this post. So all outputs below come from running the same code with CPU processes (4 unless stated otherwise) and the `gloo` backend, inside a Linux container on my laptop. The code only changes in the backend name and the device lines. I will point those out. Any GPU timing in this post is an estimate from spec sheets, and I show how I computed it. Also, PyTorch 2.3 on my ARM laptop runs some float32 matrix multiplications in reduced precision through oneDNN. So I set `torch.backends.mkldnn.enabled = False` in all scripts. Otherwise the equality checks below show errors of the order of $10^{-3}$ that have nothing to do with distributed training.

## Processes, Ranks and Collectives

PyTorch runs one process per GPU. Each process gets a unique id called its rank (0, 1, ..., world_size - 1). The total number of processes is called the world size. We launch the processes with `torchrun`, which sets a few environment variables (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`) in each process.

```bash
torchrun --nproc_per_node=4 train.py
```

Inside `train.py`, every process first joins a process group.

```python
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")  # "gloo" for CPU processes
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])  # index of the GPU on this machine
device = torch.device(f"cuda:{local_rank}")  # torch.device("cpu") for CPU processes
torch.cuda.set_device(device)
```

`nccl` is Nvidia's communication library and the right backend for GPUs. `gloo` works on CPUs. For multiple machines, we run `torchrun` on every machine with `--nnodes` and `--node_rank`. The rest of the code does not change.

The processes do not share memory. They talk only through a small set of collective operations. In a collective, every process in the group calls the same function and they all take part. The four collectives we need for this post are broadcast, all-reduce, all-gather and reduce-scatter. The following code calls each of them on a tiny tensor.

```python
def log(name, t):
    # Print every rank's tensor from rank 0, in rank order
    msgs = [None] * world_size
    dist.all_gather_object(msgs, t.tolist())
    if rank == 0:
        print(name)
        for r, m in enumerate(msgs):
            print(f"  rank {r}: {m}")

# Every rank starts with a different tensor
x = torch.arange(4, dtype=torch.float32) + 10 * rank
log("before", x)

# broadcast: rank 0's tensor is copied to all ranks
t = x.clone()
dist.broadcast(t, src=0)
log("broadcast(src=0)", t)

# all_reduce: every rank gets the elementwise sum of all tensors
t = x.clone()
dist.all_reduce(t, op=dist.ReduceOp.SUM)
log("all_reduce(SUM)", t)

# all_gather: every rank gets every rank's tensor
gathered = [torch.empty(4) for _ in range(world_size)]
dist.all_gather(gathered, x)
log("all_gather", torch.cat(gathered))

# reduce_scatter: elementwise sum, but rank i only keeps chunk i of the result
out = torch.empty(4 // world_size)
dist.reduce_scatter_tensor(out, x, op=dist.ReduceOp.SUM)
log("reduce_scatter(SUM)", out)
```

```
before
  rank 0: [0.0, 1.0, 2.0, 3.0]
  rank 1: [10.0, 11.0, 12.0, 13.0]
  rank 2: [20.0, 21.0, 22.0, 23.0]
  rank 3: [30.0, 31.0, 32.0, 33.0]
broadcast(src=0)
  rank 0: [0.0, 1.0, 2.0, 3.0]
  rank 1: [0.0, 1.0, 2.0, 3.0]
  rank 2: [0.0, 1.0, 2.0, 3.0]
  rank 3: [0.0, 1.0, 2.0, 3.0]
all_reduce(SUM)
  rank 0: [60.0, 64.0, 68.0, 72.0]
  rank 1: [60.0, 64.0, 68.0, 72.0]
  rank 2: [60.0, 64.0, 68.0, 72.0]
  rank 3: [60.0, 64.0, 68.0, 72.0]
all_gather
  rank 0: [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0, 33.0]
  rank 1: [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0, 33.0]
  rank 2: [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0, 33.0]
  rank 3: [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0, 33.0]
reduce_scatter(SUM)
  rank 0: [60.0]
  rank 1: [64.0]
  rank 2: [68.0]
  rank 3: [72.0]
```

Note that all-reduce gives the same result as a reduce-scatter followed by an all-gather. Rank $i$ first gets the sum of chunk $i$ and then everyone collects all the summed chunks. We shall use this fact twice in this post - once to compute the cost of all-reduce and once to understand ZeRO.

## Data Parallelism From Scratch

Recall the gradient accumulation equation from post 3. For a batch of $N$ samples,

$$
\frac{\text{d}(loss)}{\text{d}W} =  \frac{1}{N}\left(\frac{\text{d}(loss(x_1))}{\text{d}W} + \frac{\text{d}(loss(x_2))}{\text{d}W} + ... + \frac{\text{d}(loss(x_N))}{\text{d}W}\right)
$$

In gradient accumulation, we computed the sum on the right in a loop over micro batches, one after another, on one GPU. Data parallelism computes the same sum in parallel. Each of the $G$ GPUs (ranks) takes $N/G$ samples, computes the mean gradient over its samples and then all ranks average their gradients with an all-reduce. If every rank has the same number of samples, the mean of the per-rank means is exactly the mean over the full batch.

$$
\frac{1}{G}\sum_{g=1}^{G}\left(\frac{G}{N}\sum_{i \in \text{rank } g}\frac{\text{d}(loss(x_i))}{\text{d}W}\right) = \frac{1}{N}\sum_{i=1}^{N}\frac{\text{d}(loss(x_i))}{\text{d}W}
$$

After the all-reduce, every rank holds the same gradient. If every rank also starts with the same weights and runs the same optimiser, then every rank takes the same step and the weights stay identical. That is the entire idea.

For the experiments in this post, we shall use the EduLLM model from the previous post with the small reference configuration from post 3 (4 transformer blocks, embedding dimension 256, 4 heads, vocabulary of 1,024) but a context length of 128 to keep the CPU runs short. The attention layer uses `scaled_dot_product_attention`. This model has 3,188,224 parameters. Instead of the tokenised food.com recipes, I feed random token ids. Only the shapes and the padding matter for these checks.

The averaging step is just a loop over parameters.

```python
def all_reduce_gradients(model):
    for p in model.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size
```

Let's check the maths. Every rank builds the same model (same seed) and the same global batch of 32 sequences. Every rank first computes the reference gradient on the full batch, like a single GPU would. Then it computes the gradient on its own 8 sequences and all-reduces. Dropout masks are random, so I pass `train=False` to switch off dropout for this comparison.

```python
def compute_loss(model, inputs, targets, reduction="mean"):
    preds = model(inputs, train=False)
    return cross_entropy(
        preds.view(-1, preds.size(-1)),
        targets.view(-1),
        ignore_index=PADDING_TOKEN_ID,
        reduction=reduction,
    )

config = dict(SMALL, context_length=128)
global_batch_size = 32
micro_batch_size = global_batch_size // world_size
start, end = rank * micro_batch_size, (rank + 1) * micro_batch_size

torch.manual_seed(0)
model = EduLLM(**config).to(device)
g = torch.Generator().manual_seed(1234)
inputs, targets = random_batch(global_batch_size, config["context_length"], config["vocabulary_size"], g)
inputs, targets = inputs.to(device), targets.to(device)

# Reference: one process, full batch
compute_loss(model, inputs, targets).backward()
reference = gradients(model)
model.zero_grad(set_to_none=True)

# Data parallel: each rank takes its shard, then average
compute_loss(model, inputs[start:end], targets[start:end]).backward()
all_reduce_gradients(model)
print(f"max abs diff = {max_diff(reference, gradients(model)):.3e}, max abs grad = {max_grad(reference):.3e}")
# max abs diff = 1.630e-09, max abs grad = 3.138e-03
```

Here `gradients` returns a copy of all `.grad` tensors and `max_diff` returns the largest absolute difference between two such lists. `random_batch` returns random token ids of shape `[batch_size, context_length]` as inputs, and the same sequences shifted by one token as targets. The averaged gradients match the full batch gradients to $10^{-9}$, which is just floating point rounding. The largest gradient value is about six orders of magnitude larger.

## The Padding Trap

The equation above assumed that every rank's loss is a mean over the same number of terms. In our setup, the loss is a mean over non padding tokens (`ignore_index=PADDING_TOKEN_ID`). Real recipes have different lengths. Which means, different ranks will have different numbers of real tokens. Let's run the same check with padded sequences (`random_batch(..., pad=True)` gives every sequence a random length and fills the rest with `PADDING_TOKEN_ID`).

```
pad=True mean of means: max abs diff = 1.172e-03, max abs grad = 4.749e-03
[rank 0] non padding tokens in my shard: 700
[rank 1] non padding tokens in my shard: 598
[rank 2] non padding tokens in my shard: 586
[rank 3] non padding tokens in my shard: 452
```

The error jumped from $10^{-9}$ to $10^{-3}$, and the largest gradient is only about $5 \times 10^{-3}$. This is not rounding. The averaged gradient is simply a different number. Rank 3 has 452 real tokens and rank 0 has 700, yet both get the same weight of $\frac{1}{4}$ in the average. So every token of rank 3 counts about 1.5 times more than every token of rank 0.

The fix is to sum the loss on each rank and divide by the total number of real tokens across all ranks. That total needs one more (tiny) all-reduce.

```python
n_tokens = (targets[start:end] != PADDING_TOKEN_ID).sum()
global_n_tokens = n_tokens.clone()
dist.all_reduce(global_n_tokens, op=dist.ReduceOp.SUM)
loss = compute_loss(model, inputs[start:end], targets[start:end], reduction="sum") / global_n_tokens
loss.backward()
for p in model.parameters():
    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)  # no division by world_size now
print(f"max abs diff = {max_diff(reference, gradients(model)):.3e}")
# max abs diff = 3.027e-09
```

Note that the gradient accumulation loop from post 3 has the same problem. It divides each micro batch's mean loss by `gradient_accumulation_steps`, so micro batches with more padding get more weight per token. The same fix applies there - count the real tokens in the whole batch first and divide the summed loss by that count.

## Keeping the Replicas in Sync

Data parallelism only works if all ranks start from the same weights. A common bug is to create the model on each rank without a fixed seed. Let's simulate that. Every rank uses a different seed on purpose. In the first run, we copy rank 0's weights to every rank with `broadcast` before training. In the second run, we skip the broadcast. Both runs train for 20 steps with averaged gradients.

```python
torch.manual_seed(rank)  # different seed on every rank, on purpose
model = EduLLM(**config).to(device)
if broadcast_weights:
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
g = torch.Generator().manual_seed(42)  # same global batches on every rank
for step in range(20):
    inputs, targets = random_batch(global_batch_size, config["context_length"], config["vocabulary_size"], g)
    loss = compute_loss(model, inputs[start:end].to(device), targets[start:end].to(device))
    loss.backward()
    all_reduce_gradients(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

To measure how far apart the ranks are, every rank compares its flattened parameters with rank 0's copy and we take the maximum difference over all ranks with an all-reduce (`op=dist.ReduceOp.MAX`). Every rank also evaluates its model on the same held out batch.

```
broadcast_weights=True
  before training: max abs parameter diff across ranks = 0.000e+00
[rank 0] held out loss after 20 steps = 6.956548
[rank 1] held out loss after 20 steps = 6.956548
[rank 2] held out loss after 20 steps = 6.956548
[rank 3] held out loss after 20 steps = 6.956548
  after 20 steps: max abs parameter diff across ranks = 0.000e+00
broadcast_weights=False
  before training: max abs parameter diff across ranks = 6.482e+00
[rank 0] held out loss after 20 steps = 6.979361
[rank 1] held out loss after 20 steps = 6.964186
[rank 2] held out loss after 20 steps = 6.988252
[rank 3] held out loss after 20 steps = 6.981683
  after 20 steps: max abs parameter diff across ranks = 6.481e+00
```

With the broadcast, the four replicas are bit for bit identical after 20 steps. Without it, we silently train four different models. Note that the difference does not grow or shrink. Every rank gets the same averaged gradient, so every rank applies the same update to its own (different) weights. The initial difference stays forever. The only change from 6.482 to 6.481 comes from AdamW's weight decay, which multiplies all weights by $(1 - 0.001 \times 0.01)$ every step. Worse, the averaged gradient is a mix of gradients taken at four different points in weight space, so it is not the gradient of any of the four models. Nothing crashes. The loss still goes down a little.

The data needs the same care. In a real training run, each rank should read a different part of the dataset. PyTorch's `DistributedSampler` does this. Every rank shuffles the indices with the same seed and then takes every `world_size`-th index starting at its rank. This only works if every rank uses the same seed. Here is what happens with 10,000 recipes, 4 ranks and the sampler's `seed` set to the rank by mistake.

```python
from torch.utils.data.distributed import DistributedSampler

recipes = list(range(10_000))  # stand in for 10,000 tokenised recipes
world_size = 4

def seen_in_epoch(seeds, epoch=0):
    seen = []
    for rank in range(world_size):
        sampler = DistributedSampler(recipes, num_replicas=world_size, rank=rank, shuffle=True, seed=seeds[rank])
        sampler.set_epoch(epoch)
        seen.append(list(sampler))
    return seen

same = seen_in_epoch(seeds=[0, 0, 0, 0])
print(f"same seed: {sum(len(s) for s in same)} samples drawn, {len(set().union(*same))} unique")
# same seed: 10000 samples drawn, 10000 unique
different = seen_in_epoch(seeds=[0, 1, 2, 3])
print(f"different seed per rank: {sum(len(s) for s in different)} samples drawn, {len(set().union(*different))} unique")
# different seed per rank: 10000 samples drawn, 6850 unique
```

With different seeds, an epoch still draws 10,000 samples but only 6,850 of them are unique. The rest are duplicates and 3,150 recipes are never seen. This matches the probability that a recipe is picked by at least one of four independent shuffles, $1 - (3/4)^4 \approx 0.684$. Also note that the sampler uses `seed + epoch` to shuffle. If we forget to call `sampler.set_epoch(epoch)` at the start of every epoch, every epoch sees the data in exactly the same order.

```python
sampler = DistributedSampler(recipes, num_replicas=world_size, rank=0, shuffle=True, seed=0)
epoch_0, epoch_1 = list(sampler), list(sampler)
print(f"without set_epoch, epoch 0 == epoch 1: {epoch_0 == epoch_1}")
# without set_epoch, epoch 0 == epoch 1: True
```

<div class="callout">
🤔 <b>If data parallelism gives the same gradient as gradient accumulation, why not just use gradient accumulation on one GPU?</b><br/>
We can, and for the same global batch size we get the same model. But gradient accumulation computes the micro batches one after another. With $G$ GPUs, data parallelism computes $G$ micro batches at the same time. The number of floating point operations is the same. The wall clock time drops by up to $G$ times, minus the time spent on communication. The 7 week estimate becomes less than a week on 8 GPUs.
</div>

## The Cost of All-Reduce

Every training step, data parallelism all-reduces one number per parameter. For our small model that is 3,188,224 numbers, or 12.16 MB in 32 bit. For the 1.5 billion parameter GPT-2 with 16 bit gradients, it is 3 GB every step. How long does that take?

The simplest way to all-reduce is to send everything to rank 0, let it sum, and broadcast the result back. Rank 0 then receives $(N-1)S$ bytes and sends $(N-1)S$ bytes for $N$ ranks and a tensor of $S$ bytes. Rank 0's link becomes the bottleneck and the time grows linearly with the number of GPUs.

Ring all-reduce avoids this. Arrange the ranks in a ring and split the tensor into $N$ chunks. Then -

1. Reduce-scatter in $N-1$ steps. In every step, each rank sends one chunk to its right neighbour and adds the chunk it receives from its left neighbour to its own copy. After $N-1$ steps, rank $i$ holds the full sum of one chunk.
2. All-gather in $N-1$ steps. In every step, each rank forwards a fully summed chunk to its right neighbour. After $N-1$ steps, every rank holds every summed chunk.

Each step sends $S/N$ bytes per rank, and all ranks send at the same time. So each rank sends a total of

$$
2(N-1)\frac{S}{N}
$$

bytes. This is less than $2S$ no matter how many GPUs we add. If each step also has a fixed latency $\alpha$ and the link bandwidth is $B$, the time is

$$
t = 2(N-1)\alpha + \frac{2(N-1)}{N}\frac{S}{B}
$$

The bandwidth term is (almost) independent of $N$. The latency term grows with $N$. Which means, large tensors scale well with more GPUs and small tensors do not. Ring all-reduce was popularised for deep learning by Baidu and then by [Horovod](https://arxiv.org/abs/1802.05799). NCCL uses rings (and trees for large clusters) under the hood.

Let's measure. The following script times `all_reduce` for tensors from 4 KB to 256 MB. I ran it with 2, 4 and 8 processes. Each process uses one thread.

```python
for exponent in range(10, 27, 2):  # 2^10 to 2^26 float32 numbers (4 KB to 256 MB)
    numel = 2 ** exponent
    tensor = torch.ones(numel, dtype=torch.float32, device=device)
    for _ in range(2):  # warm up
        dist.all_reduce(tensor)
    iterations = max(3, min(200, int(2 ** 22 / numel) * 5))
    timings = []
    for _ in range(iterations):
        dist.barrier()
        start = time.perf_counter()
        dist.all_reduce(tensor)
        # torch.cuda.synchronize() here on GPUs, NCCL calls return before the work is done
        elapsed = torch.tensor(time.perf_counter() - start, dtype=torch.float64, device=device)
        # the slowest rank decides when the collective is done
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        timings.append(elapsed.item())
    timings.sort()
    median = timings[len(timings) // 2]
```

I fitted $\alpha$ and $B$ of the ring model to the 2 and 4 process measurements with least squares. The fit gives $\alpha = 0.309\text{ ms}$ per ring step and $B = 1.63\text{ GB/s}$. The bandwidth is low because the "links" here are TCP connections between processes in a container on a laptop. The chart below shows the measurements (solid) and the model (dashed).

![all_reduce time vs tensor size for 2, 4 and 8 processes, with the ring all-reduce model](/assets/images/Multi_GPU_Training/1.png)

| Processes | Size | Measured | Ring model |
| --- | --- | --- | --- |
| 2 | 4 KB | 0.49 ms | 0.62 ms |
| 4 | 4 KB | 2.06 ms | 1.86 ms |
| 8 | 4 KB | 5.40 ms | 4.33 ms |
| 2 | 16 MB | 12.75 ms | 10.91 ms |
| 4 | 16 MB | 16.85 ms | 17.29 ms |
| 8 | 16 MB | 28.11 ms | 22.34 ms |
| 2 | 256 MB | 161.02 ms | 165.30 ms |
| 4 | 256 MB | 254.62 ms | 248.88 ms |
| 8 | 256 MB | 778.95 ms | 292.52 ms |

A few things to note -

1. For small tensors, the time is flat and grows with the number of processes. Going from 2 to 4 to 8 processes, the ring has 2, 6 and 14 steps, and the measured 4 KB times (0.49, 2.06 and 5.40 ms) grow with them. A 4 KB all-reduce costs almost as much as a 250 KB one.
2. For large tensors, 2 and 4 processes follow the bandwidth term well. 4 processes take about 1.5 times as long as 2 processes, which is the ratio of $\frac{2 \times 3}{4}$ to $\frac{2 \times 1}{2}$.
3. 8 processes are much slower than the model for large tensors. My laptop has 12 cores and a single memory bus. With 8 processes plus their communication threads, the "links" of the ring are no longer independent. They all share the same memory and CPU. The 8 process numbers also changed a lot from run to run (a second run gave 13.8 ms for 4 KB and 1,080 ms for 256 MB). On a GPU server, each GPU has its own NVLink connections, so this effect is much smaller there.

Now let's estimate the GPU numbers. These are back-of-envelope estimates from spec sheets, not measurements. For GPT-2 (1.5 billion parameters) with 16 bit gradients, $S = 3\text{ GB}$. On 8 GPUs, each GPU sends $2 \times \frac{7}{8} \times 3 = 5.25\text{ GB}$ per step. An A100 SXM has 600 GB/s of NVLink bandwidth in total, which is 300 GB/s in each direction. At that peak, the all-reduce takes about $5.25 / 300 = 17.5\text{ ms}$. If the GPUs are connected only through PCIe 4.0 x16 (about 32 GB/s each way), it takes about $5.25 / 32 = 164\text{ ms}$. Real numbers will be somewhat worse since no link runs at its peak.

Compare this with the compute. The scaling laws paper estimates training compute as $6 \times \text{parameters} \times \text{tokens}$. GPT-2 used batches of 512 sequences of 1,024 tokens. That is $6 \times 1.5 \times 10^9 \times 524{,}288 \approx 4.7 \times 10^{15}$ FLOPs per step, or $5.9 \times 10^{14}$ per GPU. Even at the A100's peak 16 bit tensor core throughput (312 TFLOPS), that is about 1.9 seconds per step. So with a large batch, gradient communication is around 1% of the step time on NVLink and less than 10% on PCIe. The ratio gets worse when the batch per GPU is small, because compute shrinks with the batch and communication does not.

## DistributedDataParallel

Our manual version works but it has two problems -

1. It all-reduces 40 separate tensors, one call per parameter tensor. Every call pays the latency $\alpha$ for each ring step. With the fitted numbers above and 4 processes, 40 calls cost about $40 \times 6 \times 0.309 = 74\text{ ms}$ in latency alone, compared to 1.85 ms for a single call.
2. It waits for the whole backward pass to finish before it starts communicating. The network is idle during backward and the GPU is idle during the all-reduce.

PyTorch's [DistributedDataParallel](https://arxiv.org/abs/2006.15704) (DDP) fixes both. Using it is a one line change.

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(EduLLM(**config).to(device), device_ids=[local_rank])  # no device_ids for CPU processes
```

The training loop stays exactly like the single GPU loop. No `all_reduce_gradients` call is needed. In its constructor, DDP broadcasts rank 0's parameters to all ranks (so the bug from the previous section cannot happen). It then groups the parameters into buckets and registers a hook on every parameter. During backward, when all gradients of a bucket are ready, DDP launches an asynchronous all-reduce for that bucket and backward continues with the next layers. When `loss.backward()` returns, all buckets have been all-reduced and averaged.

We can see the buckets with a communication hook. A communication hook replaces DDP's all-reduce with our own function. Ours records the bucket and then calls the default all-reduce.

```python
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

def bucket_logging_hook(state, bucket):
    names = [state["names"][id(p)] for p in bucket.parameters()]
    state["log"].append((bucket.index(), bucket.buffer().numel(), names))
    return allreduce_hook(None, bucket)

model = EduLLM(**config).to(device)
names = {id(p): n for n, p in model.named_parameters()}
ddp_model = DDP(model, bucket_cap_mb=bucket_cap_mb)
state = {"names": names, "log": []}
ddp_model.register_comm_hook(state, bucket_logging_hook)
```

Here is the output for two iterations with the default `bucket_cap_mb=25` and with `bucket_cap_mb=1`.

```
parameters: 3,188,224, parameter tensors: 40
bucket_cap_mb=25 iteration 0: 1 all_reduce calls
  bucket 0: 3,188,224 numbers (12.16 MB), 40 tensors, token_embedding.weight ... ln.bias
bucket_cap_mb=25 iteration 1: 2 all_reduce calls
  bucket 0:   262,912 numbers ( 1.00 MB),  4 tensors, ln.weight ... transformer.3.projection.weight
  bucket 1: 2,925,312 numbers (11.16 MB), 36 tensors, transformer.3.linear.bias ... token_embedding.weight
bucket_cap_mb=1 iteration 0: 1 all_reduce calls
  bucket 0: 3,188,224 numbers (12.16 MB), 40 tensors, token_embedding.weight ... ln.bias
bucket_cap_mb=1 iteration 1: 9 all_reduce calls
  bucket 0:   262,912 numbers ( 1.00 MB),  4 tensors, ln.weight ... transformer.3.projection.weight
  bucket 1:   263,168 numbers ( 1.00 MB),  2 tensors, transformer.3.linear.bias ... transformer.3.linear.weight
  bucket 2:   460,032 numbers ( 1.75 MB),  7 tensors, transformer.3.ln2.weight ... transformer.2.projection.weight
  bucket 3:   263,168 numbers ( 1.00 MB),  2 tensors, transformer.2.linear.bias ... transformer.2.linear.weight
  bucket 4:   460,032 numbers ( 1.75 MB),  7 tensors, transformer.2.ln2.weight ... transformer.1.projection.weight
  bucket 5:   263,168 numbers ( 1.00 MB),  2 tensors, transformer.1.linear.bias ... transformer.1.linear.weight
  bucket 6:   460,032 numbers ( 1.75 MB),  7 tensors, transformer.1.ln2.weight ... transformer.0.projection.weight
  bucket 7:   263,168 numbers ( 1.00 MB),  2 tensors, transformer.0.linear.bias ... transformer.0.linear.weight
  bucket 8:   492,544 numbers ( 1.88 MB),  7 tensors, transformer.0.ln2.weight ... token_embedding.weight
```

There is a lot to read here -

1. In the first iteration, all 40 tensors go into a single bucket. DDP does not know in which order the gradients will become ready, so it does not try to overlap yet. It records the order during the first iteration and rebuilds the buckets after it.
2. From the second iteration, the buckets follow the order of the backward pass. The final layer normalisation (`ln`) and the last transformer block come first. The first block comes last.
3. The first bucket is about 1 MB in both cases. DDP keeps the first bucket small (1 MB by default) so that communication can start as early as possible.
4. The tied `token_embedding.weight` (shared with the final projection `head`) is in the last bucket. The head is used at the very end of the forward pass, so we might expect its gradient early in the backward pass. But the same tensor is also used by the embedding lookup at the start of the forward pass. Its gradient is complete only when backward reaches the embedding.
5. With the default 25 MB cap, our 12 MB model gets just two buckets. The second bucket holds 11.16 MB and can only start when the whole backward pass is done. So there is almost no overlap.

To see the overlap, I modified the hook to record when each bucket's all-reduce starts and when it finishes (with `future.then(...)`), along with the end of the forward pass and the time when `loss.backward()` returns. Each rank processes 4 sequences of 128 tokens. The chart shows the median of 5 iterations.

![Timeline of forward, backward and bucket all-reduces with bucket_cap_mb 25 and 1](/assets/images/Multi_GPU_Training/2.png)

With `bucket_cap_mb=25`, the 11.16 MB bucket starts at 274.4 ms when the last gradient is ready, and `backward()` returns at 303.1 ms. So 28.7 ms of communication is exposed. With `bucket_cap_mb=1`, eight of the nine all-reduces start while backward is still computing. Only the last 1.88 MB bucket is left at the end, and the exposed communication drops to 11.1 ms.

Smaller buckets are not always better. Every bucket pays the latency term, and our measured latency is high. On GPUs, the right `bucket_cap_mb` depends on the model, the interconnect and the batch size. The default of 25 MB is a reasonable start for models with hundreds of millions of parameters. For a small model, it is worth trying smaller values and checking the profiler.

## Gradient Accumulation With DDP

Now let's combine DDP with the gradient accumulation loop from post 3. If we use DDP as is, every micro batch's `backward()` triggers an all-reduce of all gradients. But we only need the averaged gradient once, just before `optimizer.step()`. DDP provides the `no_sync()` context manager for this. Inside `no_sync()`, gradients accumulate locally in `.grad` and no communication happens. The backward pass of the last micro batch runs outside `no_sync()` and all-reduces the accumulated gradients.

The code below is one optimizer step. It runs inside the training loop, and `micro_batches` holds the 4 micro batches of this step. I ran it with `use_no_sync` set to `False` and to `True`.

```python
from contextlib import nullcontext

gradient_accumulation_steps = 4
for micro_batch_step, (inputs, targets) in enumerate(micro_batches):
    last = micro_batch_step == gradient_accumulation_steps - 1
    # Skip gradient synchronisation for all but the last micro batch
    context = model.no_sync() if (use_no_sync and not last) else nullcontext()
    with context:
        preds = model(inputs, train=False)
        loss = cross_entropy(preds.view(-1, preds.size(-1)), targets.view(-1), ignore_index=PADDING_TOKEN_ID)
        (loss / gradient_accumulation_steps).backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

I counted the all-reduce calls and bytes with a communication hook and timed 10 optimizer steps of each variant. I alternated the two variants so that any slowdown of my laptop affects both equally. Each rank processes 4 micro batches of 2 sequences of 128 tokens.

```
no_sync=False: 8 all_reduce calls, 48.65 MB all-reduced per optimizer step, median 1198.6 ms per optimizer step
no_sync=True: 2 all_reduce calls, 12.16 MB all-reduced per optimizer step, median 1048.4 ms per optimizer step
```

`no_sync()` cuts the communication by a factor of 4 (the number of accumulation steps) and the step time by about 12%. A second run gave 1463.6 ms and 1334.1 ms, a 9% difference. The absolute numbers move around on a laptop but the gap stays. I also compared the final accumulated gradients of both variants. The largest difference was $2.3 \times 10^{-10}$. Averaging and summing are linear, so it does not matter whether we average after every micro batch or once at the end.

## Where Data Parallelism Wastes Memory

Data parallelism makes training faster but it does not let us train a bigger model. Every GPU still holds the full parameters, the full gradients and the full optimiser states. With $N$ GPUs, we store $N$ identical copies of all of them.

The [ZeRO paper](https://arxiv.org/abs/1910.02054) (Zero Redundancy Optimizer, Microsoft, 2019) removes this redundancy in three stages. Let $\Psi$ be the number of parameters. With mixed precision Adam, we need $2\Psi$ bytes for 16 bit parameters, $2\Psi$ bytes for 16 bit gradients and $12\Psi$ bytes for the optimiser (32 bit master parameters, momentum and variance). The stages are -

1. Stage 1 shards the optimiser states. Each rank owns $\frac{1}{N}$ of the parameters. It keeps the Adam states and the master copy only for those and updates only those. After the step, each rank broadcasts (or all-gathers) its updated parameters to everyone.
2. Stage 2 also shards the gradients. Instead of an all-reduce, gradients are reduce-scattered. Each rank receives only the summed gradients of the parameters it owns. Recall that an all-reduce is a reduce-scatter followed by an all-gather. ZeRO-2 runs the reduce-scatter on the gradients and the all-gather on the updated parameters. The total communication is the same as plain data parallelism.
3. Stage 3 also shards the parameters. A layer's full parameters exist only while that layer runs. Before the forward pass of a layer, its parameters are all-gathered and they are freed right after. The same happens again in the backward pass. This adds one more all-gather, so the communication is 1.5 times that of plain data parallelism.

The memory per GPU becomes -

| | Per GPU memory | GPT-2-small (117M) | GPT-2 (1.5B) | 7B model |
| --- | --- | --- | --- | --- |
| Data parallel | $16\Psi$ | 1.87 GB | 24.00 GB | 112.00 GB |
| ZeRO stage 1 | $4\Psi + \frac{12\Psi}{N}$ | 0.64 GB | 8.25 GB | 38.50 GB |
| ZeRO stage 2 | $2\Psi + \frac{14\Psi}{N}$ | 0.44 GB | 5.62 GB | 26.25 GB |
| ZeRO stage 3 | $\frac{16\Psi}{N}$ | 0.23 GB | 3.00 GB | 14.00 GB |

The numbers are for $N = 8$ GPUs. With stage 3 and 8 A100s of 80 GB, even a 7 billion parameter model leaves most of each GPU for activations. Note that ZeRO does not shard activations. Each rank still processes its own micro batch and stores its activations, so gradient checkpointing and micro batches from post 3 are still needed.

Let's measure. For this we need a bigger model, so we shall use EduLLM with the GPT-2-small configuration (12 blocks, 12 heads, embedding dimension 768, context length 1,024) and our vocabulary of 1,024. It has 79,514,112 parameters. The difference from GPT-2-small's 117 million comes only from the vocabulary - GPT-2 uses 50,257 tokens, and $49{,}233 \times 768$ extra embedding values bring us to 117,325,056. We train in 32 bit precision on CPU, so the bytes per parameter are 4 for parameters, 4 for gradients and 8 for AdamW's two states. That is again 16 bytes per parameter, but split differently.

To measure the optimiser memory, we sum the bytes of all tensors in the optimiser state of each rank after a few steps.

```python
def state_bytes(optimizer):
    # Sum of all tensors held in the optimizer state on this rank
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and value.dim() > 0:
                total += value.numel() * value.element_size()
    return total
```

PyTorch implements ZeRO stage 1 as `ZeroRedundancyOptimizer`. It wraps a normal optimiser and works with DDP.

```python
from torch.distributed.optim import ZeroRedundancyOptimizer

model = DDP(EduLLM(**config).to(device), device_ids=[local_rank])
optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, lr=1e-4)
```

For stages 2 and 3, PyTorch has Fully Sharded Data Parallel ([FSDP](https://arxiv.org/abs/2304.11277)). FSDP flattens the parameters of each wrapped module into one flat tensor and shards it across ranks. Each wrapped module is called an FSDP unit. The unit is the granularity of the all-gather, so we wrap every `TransformerBlock` separately. The rest (embeddings and the final layer normalisation) goes into the root unit. `ShardingStrategy.FULL_SHARD` is ZeRO stage 3. `ShardingStrategy.SHARD_GRAD_OP` shards gradients and optimiser states like stage 2. It also frees the full parameters after the backward pass, but keeps them from the start of the forward pass to the end of the backward pass.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

model = FSDP(
    EduLLM(**config),
    # every TransformerBlock becomes its own FSDP unit
    auto_wrap_policy=ModuleWrapPolicy({TransformerBlock}),
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # or ShardingStrategy.SHARD_GRAD_OP
    device_id=local_rank,  # device_id=torch.device("cpu") for CPU processes
)
# create the optimizer after wrapping, so it sees the sharded parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

I trained each setup for 3 steps on 4 ranks with the same seed and the same data (one sequence of 128 tokens per rank per step). I counted FSDP's collectives by wrapping `dist.all_gather` (FSDP uses it on CPU, and `dist.all_gather_into_tensor` on GPUs) and `dist.reduce_scatter_tensor` with counters. I also measured how many bytes of full (gathered) parameters the FSDP units hold right after the forward pass, before the backward pass starts. Here is the output, trimmed to rank 0 where all ranks are the same.

```
parameters: 79,514,112
DDP + AdamW: mean loss per step [7.071496, 7.141168, 7.104925]
[rank 0] params   303.3 MB, grads   303.3 MB, optimizer state   606.6 MB
DDP + ZeroRedundancyOptimizer(AdamW): mean loss per step [7.071496, 7.141168, 7.104925]
[rank 0] params   303.3 MB, grads   303.3 MB, optimizer state   154.5 MB (owns 20,250,624 parameters)
[rank 1] params   303.3 MB, grads   303.3 MB, optimizer state   154.5 MB (owns 20,250,624 parameters)
[rank 2] params   303.3 MB, grads   303.3 MB, optimizer state   148.8 MB (owns 19,506,432 parameters)
[rank 3] params   303.3 MB, grads   303.3 MB, optimizer state   148.8 MB (owns 19,506,432 parameters)
FSDP FULL_SHARD + AdamW: mean loss per step [7.071496, 7.141168, 7.104925]
  collectives per step: 25 all_gather, 13 reduce_scatter
[rank 0] params    75.8 MB, grads    75.8 MB, optimizer state   151.7 MB
  full parameters held between forward and backward:     6.0 MB
FSDP SHARD_GRAD_OP + AdamW: mean loss per step [7.071496, 7.141168, 7.104925]
  collectives per step: 13 all_gather, 13 reduce_scatter
[rank 0] params    75.8 MB, grads    75.8 MB, optimizer state   151.7 MB
  full parameters held between forward and backward:   303.3 MB
```

All four setups produce exactly the same losses. Sharding changes where the numbers live, not what is computed. Now let's check the memory against the formulas.

With DDP, every rank holds $4 \times 79{,}514{,}112$ bytes = 303.3 MB of parameters, the same for gradients and twice that (606.6 MB) for AdamW's states.

With `ZeroRedundancyOptimizer`, the optimiser states drop to roughly a quarter. The formula says $606.6 / 4 = 151.7\text{ MB}$, but ranks 0 and 1 have 154.5 MB and ranks 2 and 3 have 148.8 MB. `ZeroRedundancyOptimizer` assigns whole parameter tensors to ranks (it fills the rank with the smallest total so far). It never splits a tensor. With tensors as large as 2.4 million numbers (each MLP weight is $768 \times 3072$), the split cannot be perfectly even. Parameters and gradients are still full on every rank, as expected from stage 1.

With FSDP, parameters, gradients and optimiser states are all a quarter of DDP's (75.8, 75.8 and 151.7 MB). FSDP splits the flat tensors exactly (padding them if needed), so the split is even. At rest, both sharding strategies look the same. The difference shows up during the step -

1. `FULL_SHARD` frees each block's full parameters right after its forward pass. Between forward and backward, only the root unit (embeddings and the final layer normalisation, 6.0 MB) stays gathered. FSDP keeps the root unit because the backward pass needs it immediately. The price is 12 extra all-gathers in backward (25 in total instead of 13).
2. `SHARD_GRAD_OP` keeps all 303.3 MB of full parameters from the forward pass until the end of the backward pass, but needs only 13 all-gathers.

![Measured per rank memory of DDP, ZeroRedundancyOptimizer, FSDP FULL_SHARD and FSDP SHARD_GRAD_OP](/assets/images/Multi_GPU_Training/3.png)

The chart shows the measured per rank memory, with the gathered parameters as the hatched part. Peak memory during a step will be higher than these bars, since FSDP also needs temporary buffers for the unsharded gradients of a unit and of course the activations. For mixed precision with FSDP, we would pass `mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)`. FSDP then keeps the 32 bit sharded parameters for the optimiser and gathers 16 bit copies for compute.

## Model Parallelism

ZeRO stage 3 shards the stored tensors, but every rank still runs every layer on its own micro batch. If one layer's activations for even a single sequence do not fit, or if we want to use more GPUs than we have micro batches, we need to split the computation of the model itself.

### Tensor Parallelism

[Megatron-LM](https://arxiv.org/abs/1909.08053) (Nvidia, 2019) splits the matrices inside each transformer block. Take the MLP of our transformer block, $Z = \text{GELU}(XA)B$, where $X$ has shape `[batch_size, context_length, embedding_dimension]`, $A$ has shape `[embedding_dimension, 4 * embedding_dimension]` and $B$ has shape `[4 * embedding_dimension, embedding_dimension]`. Split $A$ by columns and $B$ by rows across $N$ ranks.

$$
A = [A_1, A_2, ..., A_N], \quad B = \begin{bmatrix}B_1 \\ B_2 \\ \vdots \\ B_N\end{bmatrix}
$$

GELU is applied elementwise, so $\text{GELU}(XA) = [\text{GELU}(XA_1), ..., \text{GELU}(XA_N)]$. Each rank can compute its slice $Y_i = \text{GELU}(XA_i)$ without talking to anyone. Then

$$
Z = \sum_{i=1}^{N} Y_i B_i
$$

Each rank computes one term of the sum, and one all-reduce adds them. In the backward pass, the gradient with respect to $X$ is also a sum over ranks, so we need one all-reduce there as well. Megatron calls these two operations $g$ (all-reduce in forward, identity in backward) and $f$ (identity in forward, all-reduce in backward). Let's implement both as autograd functions and check the result against the full MLP on 4 ranks.

```python
class AllReduceInBackward(torch.autograd.Function):
    # Identity in forward. Sums the incoming gradient across ranks in backward.
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        dist.all_reduce(grad)
        return grad


class AllReduceInForward(torch.autograd.Function):
    # Sums partial outputs across ranks in forward. Identity in backward.
    @staticmethod
    def forward(ctx, x):
        x = x.clone()
        dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


embedding_dimension = 256
torch.manual_seed(0)
block = TransformerBlock(4, embedding_dimension).to(device)  # same weights on every rank
x = torch.randn(4, 128, embedding_dimension, device=device, requires_grad=True)

# Reference: the full MLP on one device
reference = block.projection(F.gelu(block.linear(x)))
reference.sum().backward()
reference_input_grad = x.grad.clone()

# Tensor parallel: rank i keeps 1/world_size of the hidden units
hidden = 4 * embedding_dimension
shard = slice(rank * hidden // world_size, (rank + 1) * hidden // world_size)
w1 = block.linear.weight[shard].detach().clone().requires_grad_()         # [hidden / N, embedding_dimension]
b1 = block.linear.bias[shard].detach().clone().requires_grad_()           # [hidden / N]
w2 = block.projection.weight[:, shard].detach().clone().requires_grad_()  # [embedding_dimension, hidden / N]
b2 = block.projection.bias.detach().clone().requires_grad_()              # [embedding_dimension]

x_tp = x.detach().clone().requires_grad_()
h = AllReduceInBackward.apply(x_tp)
h = F.gelu(F.linear(h, w1, b1))  # [batch_size, context_length, hidden / N], no communication
partial = F.linear(h, w2)        # [batch_size, context_length, embedding_dimension], partial sum
output = AllReduceInForward.apply(partial) + b2
output.sum().backward()
```

Note that PyTorch's `nn.Linear` stores its weight transposed (`[out_features, in_features]`), so the column split of $A$ is a row split of `block.linear.weight` and the row split of $B$ is a column split of `block.projection.weight`. The bias of the second layer is added once, after the sum.

```
MLP weights per rank: 131,328 of 525,312
max abs output diff: 5.066e-07
max abs input grad diff: 4.470e-07
max abs weight grad diff: linear 0.000e+00, projection 0.000e+00
```

Each rank stores a quarter of the MLP weights, and the outputs and gradients match the full MLP up to rounding. The attention layer is split the same way. Each rank gets a subset of the heads (a column split of `wqkv`), and the output projection is split by rows with one all-reduce at the end.

The catch is the communication. Tensor parallelism all-reduces activations of shape `[batch_size, context_length, embedding_dimension]` twice per block in forward and twice in backward. For GPT-2 (48 blocks, embedding dimension 1,600) with 8 sequences of 1,024 tokens in 16 bit, every all-reduce moves $8 \times 1024 \times 1600 \times 2$ bytes $\approx 26\text{ MB}$. That is $48 \times 4 = 192$ all-reduces, or about 5 GB per micro batch. And each of them sits between two matrix multiplications that depend on it, so they are hard to overlap with compute. Compare that with data parallelism, which all-reduces 3 GB once per optimiser step no matter how many micro batches we accumulate. This is why tensor parallelism is used within a single machine where GPUs are connected with NVLink. The Megatron paper used up to 8 way tensor parallelism inside one DGX server.

### Pipeline Parallelism

Pipeline parallelism splits the stack of transformer blocks. With 4 GPUs and our 12 block model, GPU 0 runs blocks 1 to 3, GPU 1 runs blocks 4 to 6, and so on. Between stages, we only send the activations at the stage boundary, which is much less communication than tensor parallelism. So pipeline stages can be spread across machines.

The problem is that GPU 1 cannot start until GPU 0 has finished its part of the forward pass. With one batch, only one GPU works at a time. [GPipe](https://arxiv.org/abs/1811.06965) (Google, 2018) solves this with the same tool we used for gradient accumulation - micro batches. While GPU 1 works on micro batch 1, GPU 0 already starts micro batch 2. There is still an idle period at the start and at the end of every step (the pipeline bubble). With $p$ stages and $m$ micro batches, the fraction of idle time is

$$
\frac{p - 1}{m + p - 1}
$$

With 4 stages and 4 micro batches, GPUs are idle $3/7 \approx 43\%$ of the time. With 16 micro batches, it drops to $3/19 \approx 16\%$, and with 32 micro batches to $3/35 \approx 9\%$. So we want many micro batches per step. But every stage has to store the activations of all micro batches in flight until their backward pass. GPipe uses gradient checkpointing (from post 3) for this. Later schedules, such as the one forward one backward (1F1B) schedule from PipeDream, start the backward pass of a micro batch as soon as possible to limit the activations in flight.

## What to Use When

All these methods can be combined. The [Megatron-LM paper from 2021](https://arxiv.org/abs/2104.04473) combined data, tensor and pipeline parallelism to train a trillion parameter model on 3,072 A100 GPUs. For our purposes, a simple order of preference is -

1. If the model, its gradients, optimiser states and the activations of a micro batch fit on one GPU, use DDP. Use `no_sync()` with gradient accumulation and tune `bucket_cap_mb`. This covers EduLLM and GPT-2-small comfortably.
2. If the optimiser states or the parameters do not fit, use FSDP. Start with `SHARD_GRAD_OP` and move to `FULL_SHARD` if memory is still short. `FULL_SHARD` costs about 1.5 times the communication of DDP.
3. If a single block's activations do not fit even with micro batches of one sequence and gradient checkpointing, or if the number of GPUs grows beyond what data parallelism can use, add tensor parallelism inside a machine and pipeline parallelism across machines.

For the full 1.5 billion parameter GPT-2, the second option is enough. On 8 A100s with FSDP `FULL_SHARD`, the 24 GB of parameters, gradients and optimiser states become 3 GB per GPU, and the rest of each GPU is free for activations. The 7 week single GPU estimate from post 3 drops to under a week in the ideal case ($4{,}430{,}769 / 8$ seconds is about 6.4 days), and much less once mixed precision lets the tensor cores do the work.

## Next Steps

All code used in this post, along with the EduLLM model, can be found on the associated [GitHub repository](https://github.com/gauravtendolkar/EduLLM).

In the next post, we shall look at what happens behind the single line `torch.compile(model)` from post 3 - TorchDynamo, AOTAutograd and the TorchInductor compiler.