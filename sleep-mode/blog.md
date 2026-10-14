---
layout: default
title: "Sleep mode on a Mac: give the KV cache back when vLLM is idle"
permalink: /sleep-mode/
---

# Sleep mode on a Mac: give the KV cache back when vLLM is idle

[vllm-metal](https://github.com/vllm-project/vllm-metal) runs vLLM on Apple Silicon. It now supports vLLM's sleep mode: an idle engine gives memory back without a restart, and takes it again when the next request comes. On a Mac this feature means something different from what the vLLM docs say, and the reason is worth understanding.

## The problem

You keep a model server open on a Mac mini all day, because your editor and a few scripts send it questions. Between questions it is idle, but it still holds every byte it took at startup. In the afternoon you open a video project, the Mac starts to swap, and everything gets slow.

On a GPU server this is nobody's problem: the model has its own memory. On a Mac the model shares one memory with every other app. So an idle engine holding 3 GB of empty cache is 3 GB your other apps cannot have.

## What sleep mode does, and why the GPU version does not help here

vLLM's [sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode.html) has two levels. Level 1 copies the model weights from the GPU to CPU memory and throws the KV cache away (the KV cache is the pool of attention notes the model keeps for every token it has seen; it is the largest thing in the engine after the weights). Level 2 throws the weights away too.

Read level 1 again with a Mac in mind. "Copy the weights to CPU memory" frees nothing on unified memory, because the CPU and the GPU use the same memory chips. The weights would move from one place to another place in the same memory. So on a Mac, level 1 keeps the weights where they are and only gives back the KV cache.

| Level | vLLM on a GPU | vllm-metal |
|---|---|---|
| 1 | Copies weights to CPU memory, discards the KV cache | Discards the KV cache. Weights stay in place. |
| 2 | Discards weights and the KV cache | Not supported yet |

That is still the right thing. The KV cache is the big, empty part of an idle engine. And waking is almost free, because there are no weights to copy back.

![Unified memory of a Mac while the engine is awake, after sleep level 1, and after wake_up](images/01-one-memory-before-and-after.png)

## What you get back

Measured on an M1 MacBook Air with 8 GB, `gpu_memory_utilization=0.5`. The footprint is the physical memory of the engine process. Outputs after waking are compared, token by token, with the outputs before the first sleep.

| Model | Awake | After `sleep(1)` | Freed | `wake_up()` | Outputs after wake |
|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` (dense) | 2.46 GiB | 1.81 GiB | 0.65 GiB | 0.015 s | identical |
| `LiquidAI/LFM2-350M` (hybrid) | 3.23 GiB | 1.23 GiB | 2.00 GiB | 0.12 s | identical |

The freed amount is the part of the pool that was in use. vllm-metal allocates the pool lazily: a page becomes real only when something is written to it. After three short prompts most pages were never touched, so there was nothing to give back for them. A server that has answered long prompts all morning frees much more.

One cost is hidden. The new pool is empty, so the prefix cache is empty too. If your app sends the same long system prompt every time, the first answer after waking pays the prefill for it again.

## How it works inside

The scheduler, which lives in another process, never hears about the sleep. It still holds its block tables: "request A uses blocks 3, 17 and 40". The attention layers still hold their views into the pool. So the one rule is: the pool that comes back after `wake_up()` must have the same layout as the pool that went away, so that block 17 still means the same place.

The object that owns the memory, `KVCacheStorage`, does exactly that:

```python
def release(self) -> None:
    """Free the backing memory. Its contents are discarded."""
    self.tensors = {}
    self.buffers = []

def reallocate(self) -> None:
    """Re-create the backing memory with the layout vLLM scheduled against."""
    nbytes, region_offsets = self.nbytes, self.region_offsets
    self._bind()
    if (self.nbytes, self.region_offsets) != (nbytes, region_offsets):
        raise RuntimeError("Reallocated KV cache changed its layout")
```

`release` drops the references, and MLX frees the buffers. `reallocate` runs the same allocation code again, from the same config, and then checks that the size and the region offsets (where each layer's part of the pool starts) are what they were. The views read the backing buffers lazily, so they pick up the new buffers by themselves. If the layout changed, the engine stops instead of reading the wrong memory.

Two details around it:

- The model runner refuses to release while a step is in flight. vllm-metal overlaps work between decode steps, and freeing the notes under a step that is still writing them would be a silent corruption.
- Hybrid models (attention layers mixed with recurrent layers) keep a list of pending state updates that are applied lazily. That list is emptied together with the pool. Otherwise the first step after waking would write stale state into the fresh pool.

After the release, the worker calls `gc.collect()` and `mx.clear_cache()`, so that the memory is free for macOS and not only free inside the process, and logs the difference:

```
INFO Sleep mode freed 0.65 GiB; <n> GiB of Metal memory is still in use.
```

## Not supported yet

- Level 2. It needs a way to load the weights again.
- Speculative decoding with a draft model, and standalone MLA models. Their caches are not in the shared storage. The `ngram` method works.
- The PyTorch MPS backend (`VLLM_METAL_BACKEND=mps`), which has no shared storage to release.

Each fails at startup or at the call with an error that says why.

## Try it

```python
from vllm import LLM

llm = LLM("Qwen/Qwen3-0.6B", enable_sleep_mode=True)
llm.generate("The capital of France is")

llm.sleep(level=1)     # the KV cache is gone; the weights stay
llm.wake_up()          # a new, empty KV cache, in well under a second
```

As a server, the endpoints are behind vLLM's dev-mode flag:

```
$ VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --enable-sleep-mode
$ curl -X POST 'http://localhost:8000/sleep?level=1'
$ curl -X POST 'http://localhost:8000/wake_up'
```

If you see `Sleep mode is not supported on current platform` at startup, your vllm-metal is too old. Upgrade it.

**Remember.** On a Mac there is no second memory to move the weights to, so sleep mode keeps the weights and gives back only the KV cache. Waking is almost free because the new pool is lazy, and the scheduler never notices because the new pool has the same layout as the old one.
