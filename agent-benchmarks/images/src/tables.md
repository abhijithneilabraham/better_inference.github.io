# Source of the table figures. Edit here, then run make_tables.py.

## t1-words-you-will-need | Words you will need
| Word | Meaning |
|---|---|
| **Token** | A small piece of text, usually a word or part of a word. Models read and write tokens. "Hello world" is two or three tokens. |
| **Agent** | A model plus a **harness**. The harness is the program around the model. It gives the model tools (read a file, write a file, run a command) and runs the loop. |
| **Turn** | One request to the model and one answer. An agent task has many turns. |
| **pass@1** | The percentage of tasks that the agent solves on the first try. |
| **TPS** | Output tokens per second. Section B.1 explains its two meanings. |
| **TTFT** | Time to first token: how long you wait from sending the request until the first word of the answer appears. |
| **Endpoint** | A public web address (an API) where anyone with a key can send requests to your model. |

## t2-coding-agent-leaderboards-summary | Coding agent leaderboards: summary
| Leaderboard | Ranks | Who runs it | How to get on |
|---|---|---|---|
| SWE-bench | GitHub issues fixed | you | PR with logs and trajectories |
| Terminal-Bench | terminal tasks solved | you | run the official harness, submit |
| Aider Polyglot | exercises solved, cost | you | PR with the result file |
| LiveCodeBench | new contest problems | maintainers or you | through the project GitHub |
| LMArena WebDev | human votes | users | the team adds models |

## t3-inference-engine-leaderboards-summary | Inference engine leaderboards: summary
| Leaderboard | Ranks | Who runs it | How to get on |
|---|---|---|---|
| Artificial Analysis | speed for one user, TTFT, price | the AA team | public endpoint, then contact them |
| OpenRouter | live speed and latency | real traffic | become a provider |
| MLPerf Inference | total throughput | you, then review | submit in a round |

## t7-speed-test-cases | Speed test cases
| Case | Prompt length | Users at the same time |
|---|---|---|
| short, one user | 1,000 tokens | 1 |
| long, one user | 10,000 tokens | 1 |
| short, busy | 1,000 tokens | 10 |

## t8-make-each-user-faster | Make each user faster
| Change | What it does, in plain words | Before and after (example) | Risk |
|---|---|---|---|
| **Quantization** (FP8, FP4) | Store each weight in fewer bytes. Less to read for each token, so the ceiling goes up. | A 31B model in bf16 reads 62 GB for each token. In FP8 it reads 31 GB, so the ceiling doubles. | Can lower quality. Always test (B.7). |
| **Speculative decoding** (MTP, EAGLE, a small draft model, n-gram) | A small, fast helper guesses the next few tokens. The big model checks all the guesses in one read of its weights and keeps the correct ones. | Before: one token for each read. After: often 2 or 3 tokens for each read, so 1.5x to 3x faster for one user. | Gains drop when the guesses are often wrong. |
| **Tensor parallel** (`-tp 2`, `-tp 4`) | Split each layer across several GPUs. Each GPU reads only its share of the weights, so they read faster together. | 2 GPUs: up to almost 2x the bandwidth for one user. | Needs fast links between the GPUs. |
| **CUDA graphs** | Record the GPU's work for one step once, then replay it. The CPU no longer prepares each small piece every time. | Removes small CPU delays in decode, which matter most for small models. | Some kernels do not work with graphs. |
| **Fewer users at the same time** | Let fewer requests share the GPU (`--max-num-seqs`). | Each user is faster, but the server carries less in total (B.1). | Higher cost for each token. |

## t9-make-the-server-carry-more | Make the whole server carry more
| Change | What it does, in plain words | Before and after (example) | Risk |
|---|---|---|---|
| **More users in each step** (`--max-num-seqs`, `--max-num-batched-tokens`) | One read of the weights serves every user in the step, like the bus in B.1. | 1 user: 120 tokens per second in total. 32 users: about 1,900 in total (figure in B.1). | Each user gets slower. |
| **More memory for the KV cache** (`--gpu-memory-utilization` in vLLM, `--mem-fraction-static` in SGLang) | The **KV cache** is the memory that holds the notes for every token of every active request. More KV memory means more requests fit at the same time. | Before: room for 20 requests. After: room for 30. | Too much can run the GPU out of memory. |
| **FP8 KV cache** (`--kv-cache-dtype fp8`) | Store those notes in 1 byte in place of 2. | About twice as many requests fit. | Small quality loss on some models. Test it. |
| **Prefix caching** | Keep the KV notes of prompt beginnings that repeat, for example a long system prompt. Do not read them again. | A shared 5,000-token system prompt is read once, not once for every request. Much lower TTFT. | Uses memory. |
| **Chunked prefill** | Cut a long prompt into pieces and mix them with decode work, so one long prompt does not freeze everyone else. | Steadier speed for users who are already receiving an answer. | Can raise TTFT a little for the long prompt. |
| **More copies of the model** (data parallel) | Run several copies of the model on more GPUs, and share the users between them. | 2 copies: about 2x the total. | Needs more GPUs. |
