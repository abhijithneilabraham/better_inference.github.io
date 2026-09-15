# Better Inference

Write-ups on inference performance work, published at
https://abhijithneilabraham.github.io/better_inference.github.io/

- **[What I learned getting Gemma-4-31B to 150 tok/s on one B300](index.md)** —
  a tutorial in improving throughput when the stock setup fails: harness first,
  read the checkpoint, when to fork, nine walls in order, the eager-vs-graphs
  diagnostic, the roofline. Code and patches in
  [benchmark_gemma](https://github.com/abhijithneilabraham/benchmark_gemma).
