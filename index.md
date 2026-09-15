---
layout: default
title: Better Inference
---

# Better Inference

Notes on getting more tokens per second out of real hardware. Each post is a
worked example with real numbers, and the code lives alongside it.

## Posts

- **[What I learned getting Gemma-4-31B to 150 tok/s on one B300](gemma/)** —
  a tutorial in improving throughput when the stock setup fails: build the
  harness first, read the checkpoint not the model card, when to fork, nine
  walls in order, the eager-vs-graphs diagnostic, and the roofline that says
  150 tok/s is still 6.5× off bandwidth. Code and patches in
  [benchmark_gemma](https://github.com/abhijithneilabraham/benchmark_gemma).
