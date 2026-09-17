---
layout: default
title: Better Inference
---

# Better Inference

Notes on hosting large language models and making them generate tokens faster. Each article works through one real model on one real GPU, with the actual numbers, the actual failures and the code to reproduce it.

## Articles

### [Learning inference: How to host and improve the token speed of an LLM](gemma/)

Using Gemma 4 with 31B parameters on a single NVIDIA B300 to go from 46.7 to 150 tokens per second. The article starts from what tokens per second means, sets up a repeatable benchmark, tries the stock engines, and then works through the nine separate failures it took to get vLLM running 4 bit weights with speculative decoding, including a kernel that was numerically correct and still crashed inside CUDA graphs. It finishes with a roofline calculation showing that 150 tokens per second is still 6.5 times slower than the memory bandwidth allows, and where the next speedup actually is.

Code, patches, Dockerfile and dataset: [benchmark_gemma](https://github.com/abhijithneilabraham/benchmark_gemma).
