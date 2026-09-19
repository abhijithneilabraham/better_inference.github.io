# Better Inference

Articles on hosting LLMs and improving their token speed, published at
https://abhijithneilabraham.github.io/better_inference.github.io/

Each article lives in its own folder with its figures:

- [`gemma/`](gemma/blog.md): **Learning inference: How to host and improve the token speed of an LLM.**
  Gemma 4 31B on one B300, from 46.7 to 150 tokens per second.
  Code in [benchmark_gemma](https://github.com/abhijithneilabraham/benchmark_gemma).

To enable the site: Settings, then Pages, then Source: deploy from branch, `main`, root.

Figures for each article are kept twice: animated SVGs in `<article>/assets/`
for the site, and static PNGs in `<article>/figures/` for places that cannot
render SVG, such as Medium.
