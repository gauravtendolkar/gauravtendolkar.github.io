---
layout: post
title: "5. Multi GPU Distributed Training"
posted: "June 20, 2024"
categories: Super-Fast-LLM-Training
live: true
---
In previous posts, we optimised the training loop and the model architecture for a single GPU. With gradient accumulation, mixed precision, asynchronous I/O and `torch.compile`, we could train a GPT-2-small sized model on one GPU with memory to spare. But one GPU will only take us so far.

Recall the arithmetic from the post on optimising the training loop. To reach the test loss of GPT-2, the scaling laws chart says we need a billion parameter model trained for about 1 PF-day. A single A100 running at its peak 32 bit throughput would need approximately 7 weeks for that. And that is the best case.

Memory is the second problem. For mixed precision training with the Adam optimiser, every parameter needs -

1. 2 bytes for the 16 bit copy of the parameter used in the forward and backward pass.
2. 2 bytes for the 16 bit gradient.
3. 4 bytes for a 32 bit master copy of the parameter (the optimiser updates this copy).
4. 4 bytes for Adam's first moment (momentum).
5. 4 bytes for Adam's second moment (variance).

That is 16 bytes per parameter. The full 1.5 billion parameter GPT-2 model needs $16 \times 1.5 \times 10^9 = 24\text{ GB}$ just for these tensors. We have not stored a single activation yet. And as we saw in post 3, activations can need an order of magnitude more memory than everything else.

So we need more than one GPU. There are two broad ways to split the work across GPUs -

1. Split the data. Every GPU runs the full model on a different part of the batch. This is called data parallelism.
2. Split the model. Every GPU holds a part of the model. This is called model parallelism. We can split the stack of layers (pipeline parallelism) or split the matrices inside each layer (tensor parallelism).

In this post, we shall build each of these from scratch, one by one, with `torch.distributed`. We shall check every version against a single GPU, measure what it costs, and then use the PyTorch tool that does the same job. Then we remove the memory waste of data parallelism with ZeRO and FSDP, and finally combine several methods in one run.

For this post, I used a cloud machine with 2 A100 80GB SXM GPUs connected with NVLink. All outputs, timings and charts below come from that machine. The last part of the post shows how to rent a similar machine. In memory numbers printed by the scripts, 1 GB means $2^{30}$ bytes.

## Processes, Ranks and Collectives

PyTorch runs one process per GPU. Each process gets a unique id called its rank (0, 1, ..., world_size - 1). The total number of processes is called the world size. We launch the processes with `torchrun`, which sets a few environment variables (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`) in each process.

```bash
torchrun --nproc_per_node=2 train.py
```

Inside `train.py`, every process first joins a process group.

```python
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])  # index of the GPU on this machine
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
```

`nccl` is Nvidia's communication library (NCCL, pronounced "nickel") and the right backend for GPUs. For multiple machines, we run `torchrun` on every machine with `--nnodes` and `--node_rank`. The rest of the code does not change.

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
x = torch.arange(4, dtype=torch.float32, device=device) + 10 * rank
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
gathered = [torch.empty(4, device=device) for _ in range(world_size)]
dist.all_gather(gathered, x)
log("all_gather", torch.cat(gathered))

# reduce_scatter: elementwise sum, but rank i only keeps chunk i of the result
out = torch.empty(4 // world_size, device=device)
dist.reduce_scatter_tensor(out, x, op=dist.ReduceOp.SUM)
log("reduce_scatter(SUM)", out)
```

```
before
  rank 0: [0.0, 1.0, 2.0, 3.0]
  rank 1: [10.0, 11.0, 12.0, 13.0]
broadcast(src=0)
  rank 0: [0.0, 1.0, 2.0, 3.0]
  rank 1: [0.0, 1.0, 2.0, 3.0]
all_reduce(SUM)
  rank 0: [10.0, 12.0, 14.0, 16.0]
  rank 1: [10.0, 12.0, 14.0, 16.0]
all_gather
  rank 0: [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]
  rank 1: [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]
reduce_scatter(SUM)
  rank 0: [10.0, 12.0]
  rank 1: [14.0, 16.0]
```

Note that all-reduce gives the same result as a reduce-scatter followed by an all-gather. Rank $i$ first gets the sum of chunk $i$ and then everyone collects all the summed chunks. We shall use this fact twice in this post. Once to compute the cost of all-reduce and once to understand ZeRO.

### What NCCL Sees

Before we train anything, let's look at how the two GPUs are connected. `nvidia-smi topo -m` prints the connection between every pair of devices. Here are the GPU rows of the matrix (the network cards are removed).

```
$ nvidia-smi topo -m
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV12    0-63            0
GPU1    NV12     X      64-127          1
```

`NV12` means the two GPUs are connected through a bonded set of 12 NVLinks. `nvidia-smi nvlink -s` shows the speed of each link.

```
$ nvidia-smi nvlink -s
GPU 0: NVIDIA A100-SXM4-80GB
	 Link 0: 25 GB/s
	 Link 1: 25 GB/s
	 ...
	 Link 11: 25 GB/s
```

That is $12 \times 25 = 300\text{ GB/s}$ in each direction, or 600 GB/s in total. For comparison, PCIe 4.0 x16 gives about 32 GB/s in each direction.

NCCL makes the same discovery on its own when the first collective runs. If we set the environment variables `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV`, it prints what it found and what it decided. Here are a few lines from rank 0 (I removed the host name and process id prefix from every line).

```
NCCL INFO === System : maxBw 240.0 totalBw 240.0 ===
NCCL INFO CPU/0 (1/2/-1)
NCCL INFO + PCI[24.0] - PCI/1000 (1000c0101000ffff)
NCCL INFO               + PCI[24.0] - PCI/5000 (1000c01010de13b8)
NCCL INFO                             + PCI[24.0] - GPU/7000 (0)
NCCL INFO                                           + NVL[240.0] - NVS/0
NCCL INFO + SYS[16.0] - CPU/1
...
NCCL INFO Pattern 4, crossNic 0, nChannels 12, bw 20.000000/20.000000, type NVL/PIX, sameChannels 1
...
NCCL INFO Channel 00/24 :    0   1
NCCL INFO Ring 00 : 1 -> 0 -> 1
NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1 ...
NCCL INFO P2P Chunksize set to 524288
NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read
```

There is a lot to read here -

1. NCCL builds a graph of the machine. GPU 0 sits behind two PCIe switches on CPU 0, GPU 1 sits on CPU 1, and both GPUs connect over NVLink (`NVL`) to an NVSwitch (`NVS/0`). The numbers in brackets are NCCL's own bandwidth estimates in GB/s, 24 for PCIe and 240 for NVLink.
2. It then searches for rings and trees through this graph. It found 12 channels with 20 GB/s each over NVLink (`type NVL`), so 240 GB/s in total. Each channel is a ring through both GPUs (`Ring 00 : 1 -> 0 -> 1`), and NCCL doubles the channels to 24 for the final plan.
3. The last line is the transport. `P2P` means GPU 0 reads directly from GPU 1's memory over NVLink. No data goes through the CPU or host memory.

If peer to peer access between the GPUs is not possible (for example, it is disabled on some cloud machines), we would see `via SHM` (through shared host memory) here instead. When multi GPU training is slower than expected, these lines are the first thing to check.

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

For the first experiments, we shall use the EduLLM model from the previous post with the small reference configuration from post 3 (4 transformer blocks, embedding dimension 256, 4 heads, vocabulary of 1,024) and a context length of 128. The attention layer uses `scaled_dot_product_attention`. This model has 3,188,224 parameters. Instead of the tokenised food.com recipes, I feed random token ids. Only the shapes and the padding matter for these checks. Later sections use bigger models.

The averaging step is just a loop over parameters.

```python
def all_reduce_gradients(model):
    for p in model.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size
```

Let's check the maths. Every rank builds the same model (same seed) and the same global batch of 32 sequences. Every rank first computes the reference gradient on the full batch, like a single GPU would. Then it computes the gradient on its own 16 sequences and all-reduces. Dropout masks are random, so I pass `train=False` to switch off dropout for this comparison.

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

# Reference: one GPU, full batch
compute_loss(model, inputs, targets).backward()
reference = gradients(model)
model.zero_grad(set_to_none=True)

# Data parallel: each rank takes its shard, then average
compute_loss(model, inputs[start:end], targets[start:end]).backward()
all_reduce_gradients(model)
print(f"max abs diff = {max_diff(reference, gradients(model)):.3e}, max abs grad = {max_grad(reference):.3e}")
# max abs diff = 4.075e-09, max abs grad = 3.138e-03
```

Here `gradients` returns a copy of all `.grad` tensors and `max_diff` returns the largest absolute difference between two such lists. `random_batch` returns random token ids of shape `[batch_size, context_length]` as inputs, and the same sequences shifted by one token as targets. The averaged gradients match the full batch gradients to $10^{-9}$, which is just floating point rounding. The largest gradient value is about six orders of magnitude larger.

## The Padding Trap

The equation above assumed that every rank's loss is a mean over the same number of terms. In our setup, the loss is a mean over non padding tokens (`ignore_index=PADDING_TOKEN_ID`). Real recipes have different lengths. Which means, different ranks will have different numbers of real tokens. Let's run the same check with padded sequences (`random_batch(..., pad=True)` gives every sequence a random length and fills the rest with `PADDING_TOKEN_ID`).

```
pad=True mean of means: max abs diff = 5.735e-04, max abs grad = 4.749e-03
[rank 0] non padding tokens in my shard: 1298
[rank 1] non padding tokens in my shard: 1038
```

The error jumped from $10^{-9}$ to about $6 \times 10^{-4}$, and the largest gradient is only about $5 \times 10^{-3}$. This is not rounding. The averaged gradient is simply a different number. Rank 1 has 1,038 real tokens and rank 0 has 1,298, yet both get the same weight of $\frac{1}{2}$ in the average. So every token of rank 1 counts 1.25 times more than every token of rank 0.

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
# max abs diff = 5.355e-09
```

Note that the gradient accumulation loop from post 3 has the same problem. It divides each micro batch's mean loss by `gradient_accumulation_steps`, so micro batches with more padding get more weight per token. The same fix applies there. Count the real tokens in the whole batch first and divide the summed loss by that count.

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
  after 20 steps: max abs parameter diff across ranks = 0.000e+00
broadcast_weights=False
  before training: max abs parameter diff across ranks = 6.307e+00
[rank 0] held out loss after 20 steps = 6.971558
[rank 1] held out loss after 20 steps = 6.968238
  after 20 steps: max abs parameter diff across ranks = 6.306e+00
```

With the broadcast, the two replicas are bit for bit identical after 20 steps. Without it, we silently train two different models. Note that the difference does not grow or shrink. Every rank gets the same averaged gradient, so every rank applies the same update to its own (different) weights. The initial difference stays forever. The only change from 6.307 to 6.306 comes from AdamW's weight decay, which multiplies all weights by $(1 - 0.001 \times 0.01)$ every step. Worse, the averaged gradient is a mix of gradients taken at two different points in weight space, so it is not the gradient of either model. Nothing crashes. The loss still goes down a little.

The data needs the same care. In a real training run, each rank should read a different part of the dataset. PyTorch's `DistributedSampler` does this. Every rank shuffles the indices with the same seed and then takes every `world_size`-th index starting at its rank. This only works if every rank uses the same seed. Here is what happens with 10,000 recipes on our 2 ranks, once with the same seed and once with the seed set to the rank by mistake. Each rank's indices are collected on all ranks with `all_gather_object`.

```python
from torch.utils.data.distributed import DistributedSampler

recipes = list(range(10_000))  # stand in for 10,000 tokenised recipes

def coverage(sampler):
    seen = [None] * world_size
    dist.all_gather_object(seen, list(sampler))
    return sum(len(s) for s in seen), len(set().union(*seen))

sampler = DistributedSampler(recipes, num_replicas=world_size, rank=rank, shuffle=True, seed=0)
print("same seed on every rank: %d samples drawn, %d unique" % coverage(sampler))
# same seed on every rank: 10000 samples drawn, 10000 unique
sampler = DistributedSampler(recipes, num_replicas=world_size, rank=rank, shuffle=True, seed=rank)
print("different seed on every rank: %d samples drawn, %d unique" % coverage(sampler))
# different seed on every rank: 10000 samples drawn, 7480 unique
```

With different seeds, an epoch still draws 10,000 samples but only 7,480 of them are unique. The rest are duplicates and 2,520 recipes are never seen. This matches the probability that a recipe is picked by at least one of two independent half shuffles, $1 - (1/2)^2 = 0.75$.

Also note that the sampler uses `seed + epoch` to shuffle. If we forget to call `sampler.set_epoch(epoch)` at the start of every epoch, every epoch sees the data in exactly the same order. Here is the first batch of 4 recipes on each rank in the first two epochs, with a `DataLoader` on top of the sampler.

```python
loader = DataLoader(recipes, batch_size=4, sampler=sampler)
for epoch in range(2):
    sampler.set_epoch(epoch)  # the line we must not forget
    print(epoch, next(iter(loader)).tolist())
```

```
[rank 0] set_epoch=False: first batch epoch 0 [6044, 9399, 4919, 429], epoch 1 [6044, 9399, 4919, 429]
[rank 1] set_epoch=False: first batch epoch 0 [2890, 1917, 1729, 7532], epoch 1 [2890, 1917, 1729, 7532]
[rank 0] set_epoch=True: first batch epoch 0 [6044, 9399, 4919, 429], epoch 1 [5845, 9002, 1463, 8065]
[rank 1] set_epoch=True: first batch epoch 0 [2890, 1917, 1729, 7532], epoch 1 [4470, 5321, 5598, 2043]
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

The bandwidth term is (almost) independent of $N$. The latency term grows with $N$. Which means, large tensors scale well with more GPUs and small tensors do not. Ring all-reduce was popularised for deep learning by Baidu and then by [Horovod](https://arxiv.org/abs/1802.05799). NCCL uses rings and trees under the hood, as we saw in its log.

This formula also gives us a fair way to report measurements. Nvidia's [nccl-tests](https://github.com/NVIDIA/nccl-tests) report two numbers for an all-reduce of $S$ bytes in time $t$ -

1. The algorithm bandwidth, $S / t$. This is what the user sees.
2. The bus bandwidth, $\frac{S}{t} \times \frac{2(N-1)}{N}$. This is the rate at which each GPU actually sends data in a ring. It does not depend on $N$, so we can compare it directly with the speed of the links.

For $N = 2$, the factor $\frac{2(N-1)}{N}$ is exactly 1, so both numbers are the same. Let's measure. The following script times `all_reduce` for tensors from 4 KB to 1 GB.

```python
for exponent in range(10, 29, 2):  # 2^10 to 2^28 float32 numbers (4 KB to 1 GB)
    numel = 2 ** exponent
    tensor = torch.ones(numel, dtype=torch.float32, device=device)
    for _ in range(3):  # warm up
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    iterations = max(5, min(200, int(2 ** 24 / numel) * 5))
    timings = []
    for _ in range(iterations):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(tensor)
        torch.cuda.synchronize()  # NCCL calls return before the work is done
        elapsed = torch.tensor(time.perf_counter() - start, dtype=torch.float64, device=device)
        # the slowest rank decides when the collective is done
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        timings.append(elapsed.item())
    timings.sort()
    median = timings[len(timings) // 2]
    algorithm_bandwidth = numel * 4 / median / 1e9
    bus_bandwidth = 2 * (world_size - 1) / world_size * algorithm_bandwidth
```

| Size | Time | Bus bandwidth |
| --- | --- | --- |
| 4 KB | 0.059 ms | 0.07 GB/s |
| 256 KB | 0.063 ms | 4.18 GB/s |
| 1 MB | 0.094 ms | 11.17 GB/s |
| 4 MB | 0.112 ms | 37.35 GB/s |
| 16 MB | 0.183 ms | 91.87 GB/s |
| 64 MB | 0.469 ms | 143.23 GB/s |
| 256 MB | 1.627 ms | 164.97 GB/s |
| 1 GB | 6.027 ms | 178.15 GB/s |

![all_reduce time and bus bandwidth vs tensor size on 2 A100 GPUs with NVLink](/assets/images/Multi_GPU_Training/1.png)

A few things to note -

1. Up to 256 KB, the time is flat at about 0.06 ms. This is the latency term. It includes launching the NCCL kernel and our own synchronisation. A 4 KB all-reduce costs as much as a 256 KB one.
2. The bus bandwidth climbs with the size and reaches 178 GB/s at 1 GB. That is about 60% of the 300 GB/s NVLink peak per direction, and about 74% of the 240 GB/s that NCCL itself estimated in its log. Peak numbers on spec sheets are never reached in practice.
3. A 1 GB all-reduce takes 6 ms. A GPT-2-small sized model (we shall meet it shortly) has 303 MB of 32 bit gradients, which is roughly 2 ms per step on this machine.

For comparison, the same script also timed plain copies of 256 MB. Pinned host memory to GPU 0 ran at 26.16 GB/s and GPU 0 to pinned host memory at 22.89 GB/s. That is PCIe. A copy from GPU 0 to GPU 1 with `tensor.copy_()` ran at 116.89 GB/s. That copy uses the copy engines of the GPU and does not use all channels like NCCL does.

Now let's scale this up for GPT-2 (1.5 billion parameters) with 16 bit gradients, $S = 3\text{ GB}$. On 8 GPUs, each GPU sends $2 \times \frac{7}{8} \times 3 = 5.25\text{ GB}$ per step. At the 178 GB/s bus bandwidth we measured, the all-reduce takes about $5.25 / 178 \approx 29\text{ ms}$. Over PCIe 4.0 at about 25 GB/s in practice, it would take about 210 ms.

Compare this with the compute. The scaling laws paper estimates training compute as $6 \times \text{parameters} \times \text{tokens}$. GPT-2 used batches of 512 sequences of 1,024 tokens. That is $6 \times 1.5 \times 10^9 \times 524{,}288 \approx 4.7 \times 10^{15}$ FLOPs per step, or $5.9 \times 10^{14}$ per GPU. Even at the A100's peak 16 bit tensor core throughput (312 TFLOPS), that is about 1.9 seconds per step. So with a large batch, gradient communication is a few percent of the step time on NVLink. The ratio gets worse when the batch per GPU is small, because compute shrinks with the batch and communication does not.


## DistributedDataParallel

Our manual version works but it has two problems -

1. It all-reduces 40 separate tensors, one call per parameter tensor. Every call pays the latency we saw above, about 0.06 ms, even for a tensor of a few hundred numbers.
2. It waits for the whole backward pass to finish before it starts communicating. The links are idle during backward and the GPU is idle during the all-reduce.

PyTorch's [DistributedDataParallel](https://arxiv.org/abs/2006.15704) (DDP) fixes both. Using it is a one line change.

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(EduLLM(**config).to(device), device_ids=[local_rank])
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
ddp_model = DDP(model, device_ids=[local_rank])  # and once more with bucket_cap_mb=1
state = {"names": names, "log": []}
ddp_model.register_comm_hook(state, bucket_logging_hook)
```

Here is the output for two iterations with the default settings and with `bucket_cap_mb=1`.

```
parameters: 3,188,224, parameter tensors: 40
default iteration 0: 1 all_reduce calls
  bucket 0: 3,188,224 numbers (12.16 MB), 40 tensors, token_embedding.weight ... ln.bias
default iteration 1: 2 all_reduce calls
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

DDP can also tell us this itself. With the environment variables `TORCH_DISTRIBUTED_DEBUG=DETAIL` and `TORCH_CPP_LOG_LEVEL=INFO`, the C++ side of DDP (the "reducer") logs its decisions. Here are the lines for the default settings, trimmed.

```
reducer.cpp:127] Reducer initialized with bucket_bytes_cap: 26214400 first_bucket_bytes_cap: 1048576
logger.cpp:226] [Rank 0]: DDP Initialized with:
broadcast_buffers: 1
bucket_cap_bytes: 26214400
find_unused_parameters: 0
gradient_as_bucket_view: 0
num_parameter_tensors: 40
total_parameter_size_bytes: 12752896
world_size: 2
backend_name: nccl
...
reducer.cpp:1854] 2 buckets rebuilt with size limits: 1048576, 26214400 bytes.
```

There is a lot to read here -

1. In the first iteration, all 40 tensors go into a single bucket. DDP does not know in which order the gradients will become ready, so it does not try to overlap yet. It records the order during the first iteration and rebuilds the buckets after it (the `buckets rebuilt` line).
2. From the second iteration, the buckets follow the order of the backward pass. The final layer normalisation (`ln`) and the last transformer block come first. The first block comes last.
3. The cap of the first bucket is 1 MB (`first_bucket_bytes_cap: 1048576`) and the cap of the others is 25 MB (26214400 bytes). DDP keeps the first bucket small so that communication can start as early as possible.
4. The tied `token_embedding.weight` (shared with the final projection `head`) is in the last bucket. The head is used at the very end of the forward pass, so we might expect its gradient early in the backward pass. But the same tensor is also used by the embedding lookup at the start of the forward pass. Its gradient is complete only when backward reaches the embedding.
5. With the default 25 MB cap, our 12 MB model gets just two buckets. The second bucket holds 11.16 MB and can only start when the whole backward pass is done.

The small model is too small to show the overlap well. So let's switch to EduLLM with the GPT-2-small configuration (12 blocks, 12 heads, embedding dimension 768, context length 1,024) and our vocabulary of 1,024. It has 79,514,112 parameters. The difference from GPT-2-small's 117 million comes only from the vocabulary. GPT-2 uses 50,257 tokens, and $49{,}233 \times 768$ extra embedding values bring us to 117,325,056. With default settings, DDP makes 11 buckets for this model.

```
bucket sizes (MB): 9.0, 33.8, 31.5, 33.8, 33.8, 31.5, 33.8, 33.8, 31.5, 27.8, 3.0
```

A bucket is closed once it reaches the cap, so most buckets are a little larger than 25 MB. The first one is 9 MB because one of its first gradients (the last block's `projection.weight`, $768 \times 3072$ numbers) is already larger than the 1 MB cap.

The reducer also logs average timings. For the first iteration of this model (one bucket), it says -

```
 Avg backward compute time: 11081760
Avg backward comm. time: 3046944
 Avg backward comm/comp overlap time: 374272
```

The times are in nanoseconds. The backward pass took 11.1 ms and the all-reduce took 3.0 ms, but only 0.4 ms of the two overlapped.

To see the overlap in later iterations, we can use the PyTorch profiler. I profiled one training step of the GPT-2-small sized model on each rank (8 sequences of 1,024 tokens per GPU, bf16 autocast) with `torch.profiler`, and exported the trace.

```python
from torch.profiler import profile, schedule, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=schedule(wait=1, warmup=2, active=1),
             on_trace_ready=lambda p: p.export_chrome_trace(f"trace_rank{rank}.json")) as prof:
    for _ in range(4):
        train_step()
        prof.step()
```

The trace is a JSON file. It opens in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev). Every GPU kernel in it has a start time, a duration and a CUDA stream. DDP launches NCCL kernels on their own stream, so the trace on GPU 0 had the compute kernels on stream 7 and the NCCL kernels on stream 16. The chart below draws the kernels of one step on GPU 0, once with the default buckets and once with `bucket_cap_mb=1024`, which puts the whole model in one bucket.

![Timeline of compute and NCCL kernels in one DDP step, with default buckets and with one bucket](/assets/images/Multi_GPU_Training/2.png)

With the default buckets, the 11 all-reduces run on the NCCL stream while backward kernels still run on the compute stream. NCCL kernels were busy for 2.92 ms in this step, and 2.67 ms of that overlapped with compute. The last all-reduce finished 0.09 ms after the last backward kernel. So almost nothing is exposed.

With one bucket, the single all-reduce starts only after backward is done and the compute stream waits 9.5 ms for it. Part of these 9.5 ms is rank 0 waiting for rank 1 to finish its own backward pass. An NCCL kernel starts on each GPU as soon as that GPU reaches it, but it can only finish when both GPUs have joined. From the benchmark above, the transfer itself takes about 2 ms.

Note that the overlap is not free. With the default buckets, the backward pass itself took 31.8 ms instead of 27.6 ms. NCCL kernels run on the same streaming multiprocessors and use the same memory bandwidth as the compute kernels. In the end, the step took 52.1 ms with the default buckets and 54.9 ms with one bucket.

Smaller buckets are not always better. Every bucket pays the latency term. The right `bucket_cap_mb` depends on the model, the interconnect and the batch size. The default of 25 MB is a reasonable start for models with hundreds of millions of parameters. With fast NVLink and only 2 GPUs, the difference is small, as we shall see next. On slower links and with more GPUs, it is worth checking the profiler.

## Gradient Accumulation With DDP

Now let's combine DDP with the gradient accumulation loop from post 3. If we use DDP as is, every micro batch's `backward()` triggers an all-reduce of all gradients. But we only need the averaged gradient once, just before `optimizer.step()`. DDP provides the `no_sync()` context manager for this. Inside `no_sync()`, gradients accumulate locally in `.grad` and no communication happens. The backward pass of the last micro batch runs outside `no_sync()` and all-reduces the accumulated gradients.

The code below is one optimiser step with mixed precision from post 3. It runs inside the training loop, and `micro_batches` holds the 4 micro batches of this step.

```python
from contextlib import nullcontext

gradient_accumulation_steps = 4
for micro_batch_step, (inputs, targets) in enumerate(micro_batches):
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    last = micro_batch_step == gradient_accumulation_steps - 1
    # Skip gradient synchronisation for all but the last micro batch
    context = model.no_sync() if (use_no_sync and not last) else nullcontext()
    with context:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = model(inputs)
            loss = cross_entropy(preds.view(-1, preds.size(-1)), targets.view(-1), ignore_index=PADDING_TOKEN_ID)
        (loss / gradient_accumulation_steps).backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

First, a check. I ran this step with and without `no_sync()` on the small model and compared the accumulated gradients. The largest difference was $4.7 \times 10^{-10}$. Averaging and summing are linear, so it does not matter whether we average after every micro batch or once at the end.

Now the speed. I counted the all-reduce calls and bytes with a communication hook and timed 20 optimiser steps of each variant, after 3 warm up steps. The single GPU numbers use the same loop without DDP. Each GPU processes 4 micro batches per step, so 2 GPUs process twice as many tokens per step as 1 GPU.

| Model | Setup | Step | Tokens per second | All-reduce per step |
| --- | --- | --- | --- | --- |
| Small (3.3M), 16 x 512 tokens | 1 GPU | 25.5 ms | 1,283,222 | |
| | 2 GPUs, DDP | 29.9 ms | 2,191,333 | 4 calls, 50.1 MB |
| | 2 GPUs, DDP + `no_sync()` | 27.0 ms | 2,425,158 | 1 call, 12.5 MB |
| GPT-2-small sized (79.5M), 8 x 1,024 tokens | 1 GPU | 163.3 ms | 200,712 | |
| | 2 GPUs, DDP | 175.0 ms | 374,522 | 40 calls, 1,213.3 MB |
| | 2 GPUs, DDP + `no_sync()` | 166.6 ms | 393,461 | 10 calls, 303.3 MB |
| | 2 GPUs, DDP + `no_sync()`, `bucket_cap_mb=1024` | 167.0 ms | 392,370 | 1 call, 303.3 MB |

A few things to note -

1. `no_sync()` cuts the communication by a factor of 4 (the number of accumulation steps). For the GPT-2-small sized model, the step time drops from 175.0 ms to 166.6 ms.
2. With `no_sync()`, 2 GPUs give 1.96 times the tokens per second of 1 GPU for the GPT-2-small sized model, and 1.89 times for the small model. The small model does less compute per byte of gradient, so communication and overheads take a larger share of its step.
3. One big bucket and the default buckets give nearly the same speed here. On this machine, the 303 MB all-reduce takes about 2 ms of a 166 ms step. That is the benefit of NVLink.

## Model Parallelism: Splitting the Layers

Data parallelism makes training faster, but every GPU must hold the full model, its gradients and its optimiser states. If those do not fit on one GPU, data parallelism alone cannot help. The simplest fix is to put different layers on different GPUs. This is called model parallelism, and PyTorch makes the naive version very easy. We do not even need `torch.distributed`. One process can use both GPUs, and `.to()` moves activations between them.

```python
class TwoGPUEduLLM(nn.Module):
    """EduLLM with the first half of the blocks on GPU 0 and the second half on GPU 1."""

    def __init__(self, model: EduLLM, devices):
        super().__init__()
        self.devices = devices
        half = len(model.transformer) // 2
        self.token_embedding = model.token_embedding.to(devices[0])
        self.positional_embedding = model.positional_embedding.to(devices[0])
        self.first_half = model.transformer[:half].to(devices[0])
        self.second_half = model.transformer[half:].to(devices[1])
        self.ln = model.ln.to(devices[1])

    def forward(self, x, train: bool = True):
        x = x.to(self.devices[0])
        positions = torch.arange(0, x.shape[1], device=self.devices[0])
        x = self.token_embedding(x) + self.positional_embedding(positions)
        for transformer_block in self.first_half:
            x = transformer_block(x, train)
        x = x.to(self.devices[1])  # activations move from GPU 0 to GPU 1
        for transformer_block in self.second_half:
            x = transformer_block(x, train)
        # The head is tied to the token embedding, which lives on GPU 0.
        # .to() is differentiable, so its gradient flows back to GPU 0.
        return F.linear(self.ln(x), self.token_embedding.weight.to(self.devices[1]))

devices = [torch.device("cuda:0"), torch.device("cuda:1")]
model = TwoGPUEduLLM(EduLLM(**GPT2_SMALL), devices)
```

Autograd follows the `.to()` calls backwards, so the backward pass also crosses from GPU 1 to GPU 0 without any extra code. With the GPT-2-small sized model and a batch of 8 sequences of 1,024 tokens (32 bit), the loss is the same as on one GPU (7.090386) and the largest gradient difference is $1.0 \times 10^{-9}$.

```
one GPU: step 262.7 ms, peak memory GPU 0 6.48 GB
two GPUs: step 261.2 ms, peak memory GPU 0 3.79 GB, GPU 1 3.17 GB
```

Each GPU now needs about half the memory. But the step is not any faster. The profiler shows why.

![Kernels on GPU 0 and GPU 1 during one step of the naive model parallel EduLLM](/assets/images/Multi_GPU_Training/3.png)

```
GPU 0: busy   130.0 ms of   262.6 ms (49.5%)
GPU 1: busy   132.2 ms of   262.6 ms (50.3%)
both GPUs busy at the same time: 2.0 ms
```

GPU 1 cannot start until GPU 0 has finished the first half of the forward pass. Then GPU 0 waits until GPU 1 has finished the second half of the forward pass and the first half of the backward pass. At any moment, only one GPU works. The copies of activations and gradients between the GPUs are small. The profiler shows 4 peer to peer copies, the longest one 0.15 ms.

So the naive split gives us memory, not speed. And memory alone can be worth it. Let's take a model that does not fit on one GPU. With 32 blocks, 32 heads and an embedding dimension of 4,096 (context length 256), EduLLM has 5,912,010,752 parameters. In 32 bit with AdamW, it needs $16 \times 5.9 \times 10^9$ bytes, which is 88 GB for parameters, gradients and optimiser states. The GPU has 79.25 GB. To build a model this big quickly, I create it directly on the GPU with `with torch.device("cuda:0"): model = EduLLM(**config)`. Building it on the CPU first and moving it takes much longer. Then I train for 3 steps with one sequence per step.

```
one GPU: 5,912,010,752 parameters, OUT OF MEMORY during step 0, peak GPU 0 78.67 GB
two GPUs: 5,912,010,752 parameters, step 0 loss 7.8790, 0.98 s, peak GPU 0 44.60 GB, GPU 1 44.52 GB
two GPUs: 5,912,010,752 parameters, step 1 loss 4.2312, 0.70 s, peak GPU 0 44.60 GB, GPU 1 44.52 GB
two GPUs: 5,912,010,752 parameters, step 2 loss 4.4518, 0.70 s, peak GPU 0 44.60 GB, GPU 1 44.52 GB
```

On one GPU, the first step runs out of memory. Split across two GPUs, each GPU holds about 45 GB and training works.

## Pipeline Parallelism

The naive split wastes half of the GPU time. [GPipe](https://arxiv.org/abs/1811.06965) (Google, 2018) fixes this with the same tool we used for gradient accumulation, micro batches. Split the batch into $m$ micro batches. While GPU 1 works on micro batch 1, GPU 0 already starts micro batch 2. Each GPU is now called a stage.

To run the stages at the same time, we need one process per GPU again. Rank 0 holds the embeddings and the first 6 blocks. Rank 1 holds the last 6 blocks, the final layer normalisation and the head. Activations go forward with `dist.send` and `dist.recv`, and their gradients come back the same way. Here is one GPipe step. All forward passes run first, then all backward passes.

```python
def gpipe_step(micro_batches):
    m = len(micro_batches)
    if rank == 0:
        outputs = []
        for inputs, _ in micro_batches:  # all forward passes
            out = stage(inputs)
            dist.send(out.detach(), dst=1)
            outputs.append(out)
        for i in reversed(range(m)):  # all backward passes
            grad = torch.empty_like(outputs[i])
            dist.recv(grad, src=1)
            outputs[i].backward(grad)
            outputs[i] = None
    else:
        inputs, losses = [], []
        for _, targets in micro_batches:
            x = torch.empty(targets.shape[0], context_length, embedding_dimension, device=device)
            dist.recv(x, src=0)
            x.requires_grad_()
            preds = stage(x)
            loss = cross_entropy(preds.view(-1, preds.size(-1)), targets.view(-1), ignore_index=PADDING_TOKEN_ID)
            losses.append(loss / m)
            inputs.append(x)
        for i in reversed(range(m)):
            losses[i].backward()
            dist.send(inputs[i].grad, dst=0)
            inputs[i] = losses[i] = None
```

`outputs[i].backward(grad)` runs the backward pass of stage 0 with the gradient that arrived from stage 1. On stage 1, `x.requires_grad_()` makes autograd compute the gradient with respect to the received activations, which is exactly what stage 0 needs.

There is one more detail. The head is tied to the token embedding, but now they live on different GPUs. Each stage keeps its own copy of the weight and computes only its part of the gradient. So after the backward pass, the two stages all-reduce the gradient of this one tensor. Megatron-LM does the same with its embedding group.

```python
tied_weight = stage.token_embedding.weight if rank == 0 else stage.head.weight
dist.all_reduce(tied_weight.grad)
```

With a global batch of 16 sequences of the GPT-2-small sized model and 4 micro batches, the gradients match the single GPU gradients to $2.3 \times 10^{-9}$ on stage 0 and $2.6 \times 10^{-9}$ on stage 1. Stage 0 holds 40,542,720 parameters and stage 1 holds 39,757,824 (both hold the tied weight).

There is still an idle period at the start and at the end of every step. This is called the pipeline bubble. With $p$ stages and $m$ micro batches, each stage does $m$ units of work in a step that lasts $m + p - 1$ units. So the fraction of busy time is at most

$$
\frac{m}{m + p - 1}
$$

and the bubble is $\frac{p - 1}{m + p - 1}$. With $p = 2$, the ideal busy time is 50% with 1 micro batch, 80% with 4 and 94% with 16. Let's measure it. I timed one forward and backward pass of 16 sequences (32 bit, no optimiser step) with different numbers of micro batches, and measured the busy time of each GPU from profiler traces.

| Micro batches | One GPU | GPipe on 2 GPUs | Busy GPU 0 / GPU 1 | Ideal busy | Peak memory GPU 0 / GPU 1 |
| --- | --- | --- | --- | --- | --- |
| 1 (of 16) | 502.1 ms | 502.8 ms | 49.3% / 50.4% | 50.0% | 5.71 / 5.67 GB |
| 2 (of 8) | 519.4 ms | 391.7 ms | 65.4% / 66.7% | 66.7% | 5.55 / 5.58 GB |
| 4 (of 4) | 544.0 ms | 343.8 ms | 78.0% / 79.5% | 80.0% | 5.47 / 5.54 GB |
| 8 (of 2) | 570.8 ms | 326.8 ms | 85.8% / 87.3% | 88.9% | 5.43 / 5.52 GB |
| 16 (of 1) | 631.6 ms | 345.3 ms | 89.4% / 90.8% | 94.1% | 5.41 / 5.54 GB |

The "One GPU" column runs the full model on one GPU with the same micro batches and gradient accumulation.

![GPipe and 1F1B timelines with 4 micro batches on 2 GPUs](/assets/images/Multi_GPU_Training/4.png)

![Step time and GPU busy time vs number of micro batches](/assets/images/Multi_GPU_Training/5.png)

A few things to note -

1. With 1 micro batch, GPipe is the naive split from the previous section. Each GPU is busy about 50% of the time.
2. The measured busy time follows $\frac{m}{m+p-1}$ closely. The timeline shows the bubble. GPU 1 waits for the first forward pass at the start, and GPU 0 waits for the first backward pass of GPU 1 in the middle.
3. The best step time is 326.8 ms with 8 micro batches, 1.54 times faster than the 502.1 ms on one GPU. With 16 micro batches the bubble is smaller, but each micro batch has only one sequence. Small matrix multiplications use the GPU less well, as the "One GPU" column shows. It grows from 502.1 ms to 631.6 ms with the same amount of work.
4. GPipe must keep the activations of all micro batches until their backward pass. So the peak memory does not drop with more micro batches. GPipe uses gradient checkpointing (from post 3) to reduce it.

The second timeline shows the one forward one backward (1F1B) schedule from [PipeDream](https://arxiv.org/abs/1806.03377) (Microsoft, 2018). The last stage starts the backward pass of a micro batch right after its forward pass. Stage 0 runs one forward pass ahead, and then alternates between a forward pass and a backward pass. The bubble is the same, but a stage never holds the activations of more than $p$ micro batches. The core of stage 0's loop sends the next activation and receives the current gradient at the same time with `batch_isend_irecv`.

```python
for i in range(m):
    grad = torch.empty_like(outputs[i])
    if i + 1 < m:
        outputs[i + 1] = stage(micro_batches[i + 1][0])  # forward of the next micro batch
        requests = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, outputs[i + 1].detach(), 1),
            dist.P2POp(dist.irecv, grad, 1),
        ])
        for request in requests:
            request.wait()
    else:
        dist.recv(grad, src=1)
    outputs[i].backward(grad)  # backward of this micro batch
    outputs[i] = None  # free its activations
```

| Micro batches | GPipe | 1F1B | 1F1B busy GPU 0 / GPU 1 | GPipe peak memory | 1F1B peak memory |
| --- | --- | --- | --- | --- | --- |
| 4 | 343.8 ms | 343.5 ms | 78.1% / 79.5% | 5.47 / 5.54 GB | 3.34 / 2.19 GB |
| 8 | 326.8 ms | 325.0 ms | 86.3% / 87.8% | 5.43 / 5.52 GB | 2.16 / 1.58 GB |
| 16 | 345.3 ms | 340.4 ms | 90.7% / 92.1% | 5.41 / 5.54 GB | 1.57 / 1.28 GB |

The speed is the same, and the peak memory with 16 micro batches drops from 5.41 GB to 1.57 GB on GPU 0. About 0.8 GB of these numbers are the weights of each stage.

Between stages, we only send the activations at the stage boundary, here $4 \times 1024 \times 768 \times 4$ bytes $= 12.6\text{ MB}$ per micro batch of 4 sequences. That is much less communication than data parallelism, so pipeline stages can be spread across machines. Writing schedules by hand gets hard with more stages, interleaved stages and real data loading. In practice, we use Megatron-LM, DeepSpeed's pipeline engine, or [PiPPy](https://github.com/pytorch/PiPPy), which splits a model into stages automatically by tracing it and runs GPipe or 1F1B schedules.


## Tensor Parallelism

Pipeline parallelism splits the stack of layers. [Megatron-LM](https://arxiv.org/abs/1909.08053) (Nvidia, 2019) splits the matrices inside each layer instead. Take the MLP of our transformer block, $Z = \text{GELU}(XA)B$, where $X$ has shape `[batch_size, context_length, embedding_dimension]`, $A$ has shape `[embedding_dimension, 4 * embedding_dimension]` and $B$ has shape `[4 * embedding_dimension, embedding_dimension]`. Split $A$ by columns and $B$ by rows across $N$ ranks.

$$
A = [A_1, A_2, ..., A_N], \quad B = \begin{bmatrix}B_1 \\ B_2 \\ \vdots \\ B_N\end{bmatrix}
$$

GELU is applied elementwise, so

$$
\text{GELU}(XA) = [\text{GELU}(XA_1), ..., \text{GELU}(XA_N)]
$$

Each rank can compute its slice $Y_i = \text{GELU}(XA_i)$ without talking to anyone. Then

$$
Z = \sum_{i=1}^{N} Y_i B_i
$$

Each rank computes one term of the sum, and one all-reduce adds them. In the backward pass, the gradient with respect to $X$ is also a sum over ranks, so we need one all-reduce there as well. Megatron calls these two operations $g$ (all-reduce in forward, identity in backward) and $f$ (identity in forward, all-reduce in backward). Let's implement both as autograd functions.

```python
class AllReduceInBackward(torch.autograd.Function):
    # Megatron's f: identity in forward, sums the gradient across ranks in backward
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad)
        return grad


class AllReduceInForward(torch.autograd.Function):
    # Megatron's g: sums partial outputs across ranks in forward, identity in backward
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad
```

Now each rank keeps $\frac{1}{N}$ of the hidden units of the MLP. Note that PyTorch's `nn.Linear` stores its weight transposed (`[out_features, in_features]`), so the column split of $A$ is a row split of `block.linear.weight` and the row split of $B$ is a column split of `block.projection.weight`. The bias of the second layer is added once, after the sum.

```python
hidden = 4 * embedding_dimension
shard = slice(rank * hidden // world_size, (rank + 1) * hidden // world_size)
w1 = block.linear.weight[shard]         # [hidden / N, embedding_dimension]
b1 = block.linear.bias[shard]           # [hidden / N]
w2 = block.projection.weight[:, shard]  # [embedding_dimension, hidden / N]
b2 = block.projection.bias              # [embedding_dimension]

h = AllReduceInBackward.apply(x)
h = F.gelu(F.linear(h, w1, b1))  # [batch_size, context_length, hidden / N], no communication
partial = F.linear(h, w2)        # [batch_size, context_length, embedding_dimension], partial sum
output = AllReduceInForward.apply(partial) + b2
```

For an MLP with embedding dimension 256 and an input of 4 sequences of 128 tokens, on our 2 GPUs -

```
MLP weights per rank: 262,656 of 525,312
max abs output diff: 5.960e-07
max abs input grad diff: 2.980e-07
```

The count includes the weights of both layers and the first bias. The last bias is kept on every rank. Each rank stores half of the MLP, and the output and the input gradient match the full MLP.

The attention layer is split the same way. Each rank gets half of the heads. In `EfficientMultiHeadMaskedSelfAttention`, the output features of `wqkv` are grouped by head, so a rank's heads are a contiguous range of rows of `wqkv.weight`. Every head computes its attention without talking to the other heads. Our attention layer has no output projection after the heads, so instead of an all-reduce we all-gather the heads of all ranks along the last dimension. Here is the full tensor parallel transformer block.

```python
class AllGatherLastDim(torch.autograd.Function):
    # Concatenates the slices of all ranks in forward, keeps only this rank's slice in backward
    @staticmethod
    def forward(ctx, x):
        pieces = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(pieces, x.contiguous())
        return torch.cat(pieces, dim=-1)

    @staticmethod
    def backward(ctx, grad):
        return grad.chunk(world_size, dim=-1)[rank].contiguous()


class TensorParallelBlock(nn.Module):
    """A TransformerBlock whose attention heads and MLP hidden units are split across ranks."""

    def __init__(self, block: TransformerBlock):
        super().__init__()
        embedding_dimension = block.ln1.normalized_shape[0]
        self.head_dimension = block.attention.head_dimension
        self.local_heads = block.attention.num_heads // world_size
        self.ln1, self.ln2 = copy.deepcopy(block.ln1), copy.deepcopy(block.ln2)  # small, kept on every rank
        rows = 3 * self.head_dimension * self.local_heads
        self.wqkv = nn.Parameter(block.attention.wqkv.weight[rank * rows:(rank + 1) * rows].detach().clone())
        hidden = 4 * embedding_dimension // world_size
        shard = slice(rank * hidden, (rank + 1) * hidden)
        self.w1 = nn.Parameter(block.linear.weight[shard].detach().clone())
        self.b1 = nn.Parameter(block.linear.bias[shard].detach().clone())
        self.w2 = nn.Parameter(block.projection.weight[:, shard].detach().clone())
        self.b2 = nn.Parameter(block.projection.bias.detach().clone())

    def forward(self, x):
        batch_size, context_length, _ = x.shape
        # attention: this rank's heads only
        y = AllReduceInBackward.apply(self.ln1(x))
        qkv = F.linear(y, self.wqkv).view(batch_size, context_length, self.local_heads, 3 * self.head_dimension)
        query, key, value = qkv.permute(0, 2, 1, 3).split(self.head_dimension, dim=-1)
        y = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        y = y.permute(0, 2, 1, 3).reshape(batch_size, context_length, -1)  # [batch_size, context_length, embedding_dimension / N]
        y = x + AllGatherLastDim.apply(y)
        # MLP: this rank's hidden units only
        z = AllReduceInBackward.apply(self.ln2(y))
        z = F.gelu(F.linear(z, self.w1, self.b1))  # [batch_size, context_length, 4 * embedding_dimension / N]
        z = AllReduceInForward.apply(F.linear(z, self.w2)) + self.b2
        return y + z
```

For one block of the GPT-2-small sized model (12 heads, embedding dimension 768) and an input of 4 sequences of 256 tokens -

```
block parameters: 6,494,976, on each rank: 3,249,408
manual: max abs output diff 2.384e-06, input grad diff 7.731e-12
manual: max abs weight grad diff wqkv 7.640e-11, linear 5.821e-11, projection 8.731e-11, ln1 2.728e-11
```

### The Same With PyTorch

PyTorch has this built in. A `DeviceMesh` describes the GPUs, `parallelize_module` replaces the weights of the listed submodules with `DTensor`s (distributed tensors), and the styles `ColwiseParallel` and `RowwiseParallel` say how to split them. A `DTensor` knows its global shape and how it is placed on the mesh, and its operators insert the collectives for us.

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel, PrepareModuleOutput,
)

mesh = init_device_mesh("cuda", (world_size,))
torch.manual_seed(0)
block = TransformerBlock(num_heads, embedding_dimension).to(device)
parallelize_module(block, mesh, {
    "attention.wqkv": ColwiseParallel(),  # split the heads
    # concatenate the heads of all ranks (all-gather on the last dimension)
    "attention": PrepareModuleOutput(output_layouts=Shard(-1), desired_output_layouts=Replicate()),
    "linear": ColwiseParallel(),
    "projection": RowwiseParallel(),
})
block.attention.num_heads = num_heads // world_size  # heads on this rank
```

The only manual line is the last one. Our attention layer reshapes its output with `self.num_heads`, and after the split each rank has only half of the heads.

```
wqkv.weight is now a DTensor with placements (Shard(dim=0),), local shape (1152, 768)
projection.weight placements (Shard(dim=1),), local shape (768, 1536)
DTensor: max abs output diff 2.384e-06, input grad diff 7.276e-12
DTensor: max abs weight grad diff wqkv 8.004e-11, linear 6.185e-11, projection 8.731e-11
```

`wqkv.weight` has the global shape `(2304, 768)` and each rank holds 1,152 rows (`Shard(dim=0)`). `projection.weight` is split along its columns (`Shard(dim=1)`). The results match the manual version.

### How Fast Is It?

I replaced all 12 blocks of the GPT-2-small sized model with `TensorParallelBlock` and trained it on a batch of 8 sequences of 1,024 tokens (32 bit) on both GPUs. Both ranks see the same batch. Tensor parallelism splits the work inside each layer, not the batch.

```
one GPU: 262.5 ms per step, peak memory 5.84 GB
tensor parallel on 2 GPUs: 150.6 ms per step, peak memory 3.98 GB, 40,567,296 parameters per GPU
profiled step: 48 NCCL kernels (ncclDevKernel_AllGather_RING_LL, ncclDevKernel_AllReduce_Sum_f32_RING_LL), NCCL busy 8.4 ms, compute busy 141.1 ms, span 151.5 ms
```

Unlike the naive split, both GPUs work at the same time. The step is 1.74 times faster than on one GPU and each GPU holds about half of the parameters. Each block calls 4 collectives per step. An all-gather and an all-reduce in the forward pass, and two all-reduces from $f$ in the backward pass. 12 blocks give the 48 NCCL kernels in the profile.

The catch is the communication. Each of these collectives moves activations of shape `[batch_size, context_length, embedding_dimension]`, here $8 \times 1024 \times 768 \times 4$ bytes $\approx 25\text{ MB}$, and each one sits between two matrix multiplications that depend on it. So they cannot overlap with compute. Here they took 8.4 ms of the 151.5 ms step, over NVLink. For GPT-2 (48 blocks, embedding dimension 1,600) with 8 sequences of 1,024 tokens in 16 bit, every all-reduce moves about 26 MB, and there are $48 \times 4 = 192$ of them per micro batch. Compare that with data parallelism, which all-reduces the gradients once per optimiser step no matter how many micro batches we accumulate. This is why tensor parallelism is used only within a single machine where GPUs are connected with NVLink. The Megatron paper used up to 8 way tensor parallelism inside one DGX server.

## ZeRO: Removing the Copies

Let's go back to data parallelism. It makes training faster but it does not let us train a bigger model. Every GPU still holds the full parameters, the full gradients and the full optimiser states. With $N$ GPUs, we store $N$ identical copies of all of them.

The [ZeRO paper](https://arxiv.org/abs/1910.02054) (Zero Redundancy Optimizer, Microsoft, 2019) removes this redundancy in three stages. Let $\Psi$ be the number of parameters. With mixed precision Adam, we need $2\Psi$ bytes for 16 bit parameters, $2\Psi$ bytes for 16 bit gradients and $12\Psi$ bytes for the optimiser (32 bit master parameters, momentum and variance). The stages are -

1. Stage 1 shards the optimiser states. Each rank owns $\frac{1}{N}$ of the parameters. It keeps the Adam states and the master copy only for those and updates only those. After the step, each rank broadcasts (or all-gathers) its updated parameters to everyone.
2. Stage 2 also shards the gradients. Instead of an all-reduce, gradients are reduce-scattered. Each rank receives only the summed gradients of the parameters it owns. Recall that an all-reduce is a reduce-scatter followed by an all-gather. So stage 2 needs the same communication as plain data parallelism.
3. Stage 3 also shards the parameters. A layer's full parameters exist only while that layer runs. Before the forward pass of a layer, its parameters are all-gathered and they are freed right after. The same happens again in the backward pass. This adds one more all-gather, so the communication is 1.5 times that of plain data parallelism.

The memory per GPU becomes -

| | Per GPU memory | GPT-2-small (117M) | GPT-2 (1.5B) | 7B model |
| --- | --- | --- | --- | --- |
| Data parallel | $16\Psi$ | 1.87 GB | 24.00 GB | 112.00 GB |
| ZeRO stage 1 | $4\Psi + \frac{12\Psi}{N}$ | 0.64 GB | 8.25 GB | 38.50 GB |
| ZeRO stage 2 | $2\Psi + \frac{14\Psi}{N}$ | 0.44 GB | 5.62 GB | 26.25 GB |
| ZeRO stage 3 | $\frac{16\Psi}{N}$ | 0.23 GB | 3.00 GB | 14.00 GB |

The numbers in this table are for $N = 8$ GPUs, and here 1 GB is $10^9$ bytes. With stage 3 and 8 A100s of 80 GB, even a 7 billion parameter model leaves most of each GPU for activations. Note that ZeRO does not shard activations. Each rank still processes its own micro batch and stores its activations, so gradient checkpointing and micro batches from post 3 are still needed.

ZeRO was first implemented in Microsoft's [DeepSpeed](https://github.com/microsoft/DeepSpeed) library, where we pick the stage with one number in a JSON config (`"zero_optimization": {"stage": 2}`). DeepSpeed can also move optimiser states and parameters to CPU memory or NVMe drives (ZeRO-Offload and ZeRO-Infinity). PyTorch has its own implementations, and we shall use those.

Let's measure. For this we need a model where the memory is large compared to the activations. So we shall use a GPT-2-medium sized EduLLM (24 blocks, 16 heads, embedding dimension 1,024) with our vocabulary of 1,024. It has 279,144,448 parameters. We train it in 32 bit on 2 GPUs with 4 sequences of 256 tokens per GPU per step. In 32 bit, the bytes per parameter are 4 for parameters, 4 for gradients and 8 for AdamW's two states. That is again 16 bytes per parameter, but split differently.

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
    device_id=local_rank,
)
# create the optimizer after wrapping, so it sees the sharded parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

I trained each setup for 6 steps with the same seed and the same data. I counted FSDP's collectives by wrapping `dist.all_gather_into_tensor` and `dist.reduce_scatter_tensor` with counters, and measured how many bytes of full (gathered) parameters the FSDP units hold right after the forward pass, before the backward pass starts. Here is the output of rank 0. Rank 1 printed the same numbers.

```
parameters: 279,144,448
DDP + AdamW: mean loss per step [7.125552, 7.1566, 7.083848], median step time 135.5 ms
[rank 0] params  1064.9 MB, grads  1064.9 MB, optimizer state  2129.7 MB | allocated  5340.6 MB, peak  6926.5 MB
DDP + ZeroRedundancyOptimizer(AdamW): mean loss per step [7.125552, 7.1566, 7.083849], median step time 136.6 ms
[rank 0] params  1064.9 MB, grads  1064.9 MB, optimizer state  1064.9 MB (owns 139,572,224 parameters) | allocated  4275.8 MB, peak  5861.6 MB
FSDP FULL_SHARD + AdamW: mean loss per step [7.125552, 7.1566, 7.083848], median step time 126.7 ms
  collectives per step: 49 all_gather, 25 reduce_scatter
[rank 0] params   532.4 MB, grads   532.4 MB, optimizer state  1064.9 MB | allocated  2146.0 MB, peak  3840.0 MB
  full parameters held between forward and backward:     8.0 MB
  full parameters held after backward:     0.0 MB
FSDP SHARD_GRAD_OP + AdamW: mean loss per step [7.125552, 7.1566, 7.083848], median step time 129.8 ms
  collectives per step: 25 all_gather, 25 reduce_scatter
[rank 0] params   532.4 MB, grads   532.4 MB, optimizer state  1064.9 MB | allocated  2678.5 MB, peak  5341.2 MB
  full parameters held between forward and backward:  1064.9 MB
  full parameters held after backward:     0.0 MB
```

![Measured per GPU memory of DDP, ZeroRedundancyOptimizer, FSDP SHARD_GRAD_OP and FSDP FULL_SHARD](/assets/images/Multi_GPU_Training/6.png)

All four setups produce the same losses (the last digit of one value differs by rounding). Sharding changes where the numbers live, not what is computed. Now let's check the memory against the formulas -

1. With DDP, every rank holds $4 \times 279{,}144{,}448$ bytes = 1,064.9 MB of parameters, the same for gradients and twice that (2,129.7 MB) for AdamW's states.
2. With `ZeroRedundancyOptimizer`, the optimiser states drop to half (1,064.9 MB) and each rank owns exactly half of the parameters (139,572,224). `ZeroRedundancyOptimizer` assigns whole parameter tensors to ranks and never splits a tensor, so the split is not always this even. Parameters and gradients are still full on every rank, as expected from stage 1. The peak drops by the same 1,065 MB.
3. With FSDP, parameters, gradients and optimiser states are all half of DDP's (532.4, 532.4 and 1,064.9 MB). FSDP splits the flat tensors exactly (padding them if needed).

At rest, both FSDP sharding strategies look the same. The difference shows up during the step -

1. `FULL_SHARD` frees each block's full parameters right after its forward pass. Between forward and backward, only the root unit (embeddings and the final layer normalisation, 8.0 MB) stays gathered. FSDP keeps the root unit because the backward pass needs it immediately. The price is 24 extra all-gathers in backward (49 in total instead of 25). Its peak is 3,840 MB.
2. `SHARD_GRAD_OP` keeps all 1,064.9 MB of full parameters from the forward pass until the end of the backward pass, but needs only 25 all-gathers. Its peak is 5,341 MB.

Note that the FSDP steps were not slower than DDP here. FSDP overlaps the all-gather of the next unit with the compute of the current one, NVLink is fast, and the optimiser step only updates half of the parameters on each GPU. For mixed precision with FSDP, we would pass `mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)`. FSDP then keeps the 32 bit sharded parameters for the optimiser and gathers 16 bit copies for compute.

### A Model That Does Not Fit

Now let's take the 5.9 billion parameter EduLLM from the model parallel section and train it with data parallelism on both GPUs (one sequence of 256 tokens per GPU per step, 32 bit AdamW). As before, every rank builds the full model directly on its GPU. That takes 22.02 GB per GPU. I used `AdamW(..., foreach=False)`, which updates one parameter at a time and avoids large temporary buffers, and `DDP(..., gradient_as_bucket_view=True)`, which makes the gradients views into DDP's buckets instead of separate tensors.

```
[ddp rank 0] after wrapping: 44.05 GB allocated
[ddp rank 0] OUT OF MEMORY during step 0 optimizer, 77.11 GB allocated, peak 77.11 GB
[zero1 rank 0] after wrapping: 44.05 GB allocated
[zero1 rank 0] step 2: loss 8.3027, 0.86 s, 66.10 GB allocated, peak 68.38 GB
[fsdp_grad_op rank 0] after wrapping: 11.01 GB allocated
[fsdp_grad_op rank 0] step 2: loss 8.3027, 0.78 s, 33.05 GB allocated, peak 58.41 GB
[fsdp_full rank 0] after wrapping: 11.01 GB allocated
[fsdp_full rank 0] step 2: loss 8.3027, 0.80 s, 33.05 GB allocated, peak 46.22 GB
```

A few things to note -

1. DDP needs 44.05 GB before the first step. DDP allocates its gradient buckets in the constructor, and the buckets have the size of all parameters (22 GB). The AdamW states need another 44 GB on the first `optimizer.step()`, and that does not fit.
2. `ZeroRedundancyOptimizer` keeps only half of the AdamW states on each GPU (22 GB instead of 44 GB), and the model trains with a peak of 68.38 GB.
3. FSDP shards the parameters right after wrapping (11.01 GB per GPU). `SHARD_GRAD_OP` peaks at 58.41 GB and `FULL_SHARD` at 46.22 GB.
4. All three give the same loss on rank 0 in every step (7.8790, 10.0735 and 8.3027).

## Putting It All Together

Each method splits a different thing. Data parallelism splits the batch, tensor parallelism splits each layer, pipeline parallelism splits the stack of layers, and ZeRO splits the stored states of data parallelism. Large training runs combine them. We arrange the GPUs in a grid, the device mesh, and give each dimension of the grid one kind of parallelism.

Let's build the two dimensional version. The mesh has a data parallel dimension `dp` and a tensor parallel dimension `tp`. Every GPU in a row of the mesh holds a different slice of each layer. Every GPU in a column holds the same slice and processes a different part of the batch. For the `tp` dimension we use our own `TensorParallelBlock`, now with a process group argument, so that its collectives only talk to the GPUs in the same row. For the `dp` dimension we use FSDP, so that the slices are also sharded ZeRO-3 style over the GPUs in the same column.

```python
mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
tp_group, dp_group = mesh.get_group("tp"), mesh.get_group("dp")

torch.manual_seed(0)
model = EduLLM(**GPT2_SMALL).to(device)  # same initial weights on every rank
# 1. Tensor parallel inside every transformer block, over the "tp" dimension of the mesh
model.transformer = nn.ModuleList([TensorParallelBlock(block, tp_group) for block in model.transformer])
# 2. FSDP (ZeRO-3) shards every block over the "dp" dimension of the mesh
model = FSDP(model, process_group=dp_group, auto_wrap_policy=ModuleWrapPolicy({TensorParallelBlock}),
             device_id=local_rank)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# every data parallel rank takes its own slice of the global batch
size = global_batch_size // dp_size
dp_rank = dist.get_rank(dp_group)
inputs, targets = inputs[dp_rank * size:(dp_rank + 1) * size], targets[dp_rank * size:(dp_rank + 1) * size]
```

The only change in `TensorParallelBlock` is that it takes a process group. The autograd functions get the group as an extra argument and pass it to the collectives (`dist.all_reduce(x, group=group)`), and the rank and world size come from `dist.get_rank(group)` and `dist.get_world_size(group)`. I trained the GPT-2-small sized model for 8 steps with a global batch of 16 sequences (32 bit) and compared the losses with one GPU training on the full batch. With 2 GPUs, one dimension of the mesh has size 1, so I ran both shapes.

```
torchrun --nproc_per_node=2 combined.py 2 1
# [rank 0] mesh [[0], [1]], dp rank 0, tp rank 0
# [rank 1] mesh [[0], [1]], dp rank 1, tp rank 0
# one GPU: losses [7.0853, 7.0745, 7.0398, 7.0043, 6.9927, 6.9863, 6.968, 6.9643], 505.0 ms per step
# [rank 0] parameters stored on this GPU: 39,757,056
# dp=2 x tp=1: losses [7.0853, 7.0745, 7.0398, 7.0043, 6.9927, 6.9863, 6.968, 6.9643], 265.6 ms per step
# [rank 0] peak memory 5.15 GB
# max abs loss diff vs one GPU over 8 steps: 1.43e-06

torchrun --nproc_per_node=2 combined.py 1 2
# [rank 0] mesh [[0, 1]], dp rank 0, tp rank 0
# [rank 1] mesh [[0, 1]], dp rank 0, tp rank 1
# one GPU: losses [7.0853, 7.0745, 7.0398, 7.0043, 6.9927, 6.9863, 6.968, 6.9643], 504.9 ms per step
# [rank 0] parameters stored on this GPU: 40,567,296
# dp=1 x tp=2: losses [7.0853, 7.0745, 7.0398, 7.0043, 6.9927, 6.9863, 6.968, 6.9643], 288.4 ms per step
# [rank 0] peak memory 6.38 GB
# max abs loss diff vs one GPU over 8 steps: 4.77e-07
```

Both shapes train the same model as one GPU. They are 1.90 and 1.75 times faster than one GPU. With `(2, 1)`, FSDP shards the whole model across the two GPUs (39,757,056 parameters each). With `(1, 2)`, tensor parallelism splits each block (40,567,296 parameters each, the embeddings are kept on both). The data parallel group then has only one GPU, so FSDP prints a warning that it switches to `NO_SHARD` and does nothing else. On this small model and fast link, data parallelism is a little faster. The same script runs unchanged on 8 GPUs as `(4, 2)` or `(2, 4)`.

### Scaling to 3D Parallelism

Real clusters add the third dimension, pipeline parallelism, and use ZeRO stage 1 for the data parallel dimension. This is the 3D parallelism of [Megatron-LM](https://arxiv.org/abs/2104.04473) and DeepSpeed, used together as Megatron-DeepSpeed. The 2021 Megatron-LM paper used it to train models of up to a trillion parameters on 3,072 A100 GPUs, and BLOOM (176 billion parameters) was trained with Megatron-DeepSpeed on 384 A100 80GB GPUs as 4 way tensor, 12 way pipeline and 8 way data parallelism.

Let's do the arithmetic for a GPT-3 sized model, $\Psi = 175 \times 10^9$ parameters with 96 layers and an embedding dimension of 12,288, on 128 servers with 8 A100 80GB GPUs each (1,024 GPUs). The GPUs in a server are connected with NVLink and the servers with InfiniBand. A common layout is -

1. Tensor parallelism $t = 8$ inside each server, because it needs NVLink.
2. Pipeline parallelism $p = 8$ across servers. Each stage has $96 / 8 = 12$ layers and only sends the activations at the stage boundary.
3. Data parallelism $d = 1024 / (t \times p) = 16$ over the remaining dimension, with ZeRO stage 1.

Memory first. With mixed precision Adam, the states take $16\Psi = 2.8\text{ TB}$. Tensor and pipeline parallelism split the model into $t \times p = 64$ pieces of $\Psi / 64 \approx 2.73 \times 10^9$ parameters each. Without ZeRO, each GPU needs $16 \times 2.73 \times 10^9 \approx 43.8\text{ GB}$. With ZeRO stage 1 over the 16 data parallel copies, it needs

$$
4 \times \frac{\Psi}{64} + \frac{12}{16} \times \frac{\Psi}{64} \approx 10.9 + 2.1 = 13.0\text{ GB}
$$

and the rest of the 80 GB is free for activations.

Now the bubble. GPT-3 used batches of about 3.2 million tokens, 1,536 sequences of 2,048 tokens. Each of the 16 data parallel pipelines gets 96 sequences. With micro batches of 1 sequence, $m = 96$ and the bubble is $\frac{p-1}{m+p-1} = \frac{7}{103} \approx 6.8\%$.

Then the communication -

1. Tensor parallelism all-reduces activations of $1 \times 2048 \times 12288 \times 2$ bytes $\approx 50\text{ MB}$, 4 times per layer per micro batch, inside a server over NVLink.
2. Pipeline parallelism sends a tensor of the same size once per micro batch from one stage to the next, and its gradient back, over InfiniBand.
3. Data parallelism reduces the gradients of each GPU's $2.73 \times 10^9$ parameters once per step. In 16 bit that is 5.5 GB, and a ring over 16 copies sends $2 \times \frac{15}{16} \times 5.5 \approx 10.3\text{ GB}$ per GPU. With one 200 Gb/s (25 GB/s) InfiniBand link per GPU, as in Nvidia's DGX A100 servers, that takes about 0.4 s.

The compute per step is $6 \times 175 \times 10^9 \times 3.1 \times 10^6 \approx 3.3 \times 10^{18}$ FLOPs, or $3.2 \times 10^{15}$ per GPU. At the A100's peak 312 TFLOPS that is 10 seconds, and more like 20 seconds at a realistic 50% of peak. So the data parallel all-reduce is a few percent of the step, and most of it can overlap with the backward pass of the last micro batches. This is how the methods fit together. Put the most communication heavy method on the fastest links, and the least communication heavy method on the slowest links.

## Setting Up a Multi GPU Machine on RunPod

All experiments in this post ran on a rented cloud machine. I used [RunPod](https://www.runpod.io), but any provider with multi GPU machines works the same way. Here are the steps -

1. Create an account and add some credit. In the account settings, add your SSH public key (the contents of `~/.ssh/id_ed25519.pub`).
2. Go to Pods and deploy a new pod. Choose the GPU type (I used A100 SXM 80GB) and set the GPU count to 2. SXM GPUs are connected with NVLink. PCIe versions of the same GPU are cheaper but usually talk over PCIe only.
3. Choose a PyTorch template. It comes with CUDA, NCCL and PyTorch installed. Make sure TCP port 22 is exposed if you want to use SSH.
4. When the pod is running, the Connect button shows an SSH command with the public IP address and port. You can also use the web terminal in the browser.
5. Check the machine before running anything.

```bash
nvidia-smi topo -m   # NV# between the GPUs means NVLink
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

6. Copy your code with `git clone` or `scp`, and run the scripts with `torchrun --nproc_per_node=2 script.py`. Set `NCCL_DEBUG=INFO` for the first run to see the transport NCCL chooses.
7. Copy the results back with `scp -P <port> -r root@<ip>:/workspace/results .`.
8. Terminate the pod when you are done. A stopped pod still keeps its disk, and you pay for that storage. A terminated pod is deleted with everything on it.

The pod is billed for as long as it exists, so it is worth preparing and testing all scripts before creating it. All scripts in this post ran in less than 15 minutes of pod time.

## What to Use When

All these methods can be combined, but each one adds complexity. For our purposes, a simple order of preference is -

1. If the model, its gradients, optimiser states and the activations of a micro batch fit on one GPU, use DDP. Use `no_sync()` with gradient accumulation and use the default buckets unless the profiler shows exposed communication. This covers EduLLM and GPT-2-small comfortably. On our 2 GPUs, it gave 1.96 times the tokens per second of one GPU.
2. If the optimiser states or the parameters do not fit, use FSDP (or DeepSpeed ZeRO). Start with `SHARD_GRAD_OP` and move to `FULL_SHARD` if memory is still short. `FULL_SHARD` costs about 1.5 times the communication of DDP. `ZeroRedundancyOptimizer` is the smallest change to a DDP script, if only the optimiser states are the problem.
3. If a single layer is too large, or the batch per GPU becomes too small with many GPUs, add tensor parallelism inside a machine with NVLink.
4. If the model still does not fit, or we train across many machines, add pipeline parallelism across machines with many micro batches and a 1F1B schedule.
5. Avoid the naive split of layers across GPUs unless memory is the only problem. It uses one GPU at a time.

For the full 1.5 billion parameter GPT-2, the second option is enough. On 8 A100s with FSDP `FULL_SHARD`, the 24 GB of parameters, gradients and optimiser states become 3 GB per GPU, and the rest of each GPU is free for activations. In post 3 we estimated 7 weeks on one A100 in 32 bit for a billion parameter model. On 8 GPUs, that drops to under a week in the ideal case ($4{,}430{,}769 / 8$ seconds is about 6.4 days), and much less once mixed precision lets the tensor cores do the work.

## Next Steps

All code used in this post, along with the EduLLM model, can be found on the associated [GitHub repository](https://github.com/gauravtendolkar/EduLLM).

In the next post, we shall look at what happens behind the single line `torch.compile(model)` from post 3. We shall look at TorchDynamo, AOTAutograd and the TorchInductor compiler.
