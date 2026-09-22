---
layout: default
title: Better Inference
---

# Better Inference

Notes on hosting large language models and making them generate tokens faster. Each article works through one real model on one real GPU, with the actual numbers, the actual failures and the code to reproduce it.

## Articles

### [Learning inference: How to host and improve the token speed of an LLM](gemma/)

Using Gemma 4 with 31B parameters on a single NVIDIA B300 to go from 46.7 to 150 tokens per second. The article starts from what tokens per second means, sets up a repeatable benchmark, tries the stock engines, and then works through the nine separate failures it took to get vLLM running 4 bit weights with speculative decoding, including a kernel that was numerically correct and still crashed inside CUDA graphs. It finishes with a roofline calculation showing that 150 tokens per second is still 6.5 times slower than the memory bandwidth allows, and where the next speedup actually is.

Code, patches and Dockerfile: [benchmark_gemma](https://github.com/abhijithneilabraham/benchmark_gemma). Benchmark dataset: [longctx30 on Hugging Face](https://huggingface.co/datasets/abhijithneilabraham/longctx30).

### [Reducing the cold start time of an LLM server](cold-start/)

Using vLLM serving a 14B model on a single H100 to go from a 159.6 second cold start down to 18.2 seconds, and then to 0.55 seconds by not restarting at all. The article breaks down where the time actually goes, disk reads and compilation, shows the exact commands for each fix, explains which settings force a full recompile and which are free, and covers two changes that looked like wins and measured out to nothing.

Code and every measured result: [inference_research](https://github.com/abhijithneilabraham/inference_research).

### [From one GPU to ten million users](scaling/)

How to size an inference system, starting from a load test instead of a guess. Covers what a load test measures, Little's Law as the one formula connecting concurrency, throughput and latency, the KV cache memory math that usually caps concurrency before compute does, and how to turn a customer count into the peak concurrency number a system actually has to be built for. Ends with the pieces that make many GPU replicas behave like one system: routing, autoscaling, continuous batching, prefix caching, and backpressure.
