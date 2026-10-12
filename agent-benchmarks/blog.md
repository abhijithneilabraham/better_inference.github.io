---
layout: default
title: "Get on the leaderboards: benchmark coding agents and increase LLM inference TPS"
permalink: /agent-benchmarks/
---

# Get on the leaderboards: benchmark coding agents and increase LLM inference TPS

*This article uses simple English, close to ASD-STE100 Simplified Technical English (about 80%). Each technical word gets a short explanation in brackets the first time it occurs. Technical names and commands stay as they are.*

This article joins two topics that people often mix up.

- **How good is a coding agent?** This is a question about quality. You answer it with agent benchmarks such as SWE-bench and Terminal-Bench. The answer depends on the model, the agent program around it, and a few server settings.
- **How fast does an LLM server make tokens?** This is a question about speed, measured in **TPS** (tokens per second). The answer depends on the GPU, the serving engine (for example vLLM or SGLang) and its settings. It has nothing special to do with agents. The same TPS matters for a chat app, a document summary job or a coding agent.

Each topic has its own leaderboards. So the article starts with them: first the coding agent leaderboards, then the inference engine leaderboards, with how to get on each one. Then **Part A** teaches how to benchmark a coding agent, and **Part B** teaches how to measure and increase inference TPS. **Part C** shows the one place where the two topics meet.

![Two separate topics: coding agent quality and LLM inference speed, with the question, unit, tools and leaderboards of each](images/02-two-topics.png)

The rules of each leaderboard change often. This article describes them as they were in October 2026. Before you submit, read the current rules on each site.

## Words you will need

| Word | Meaning |
|---|---|
| **Token** | A small piece of text, usually a word or part of a word. Models read and write tokens. "Hello world" is two or three tokens. |
| **Agent** | A model plus a **harness**. The harness is the program around the model. It gives the model tools (read a file, write a file, run a command) and runs the loop. |
| **Turn** | One request to the model and one answer. An agent task has many turns. |
| **pass@1** | The percentage of tasks that the agent solves on the first try. |
| **TPS** | Output tokens per second. Section B.1 explains its two meanings. |
| **TTFT** | Time to first token: how long you wait from sending the request until the first word of the answer appears. |
| **Endpoint** | A public web address (an API) where anyone with a key can send requests to your model. |

---

# Coding agent leaderboards

![Five coding agent leaderboards, what each ranks and who runs the test](images/01a-coding-agent-leaderboards.png)

These leaderboards ask one question: **how many tasks does this agent solve?** They measure quality. The speed of your server does not appear on them, although it can change your score on long tasks (Part C).

On most of them, you run the benchmark yourself and send the results with all the logs. The leaderboard team checks the logs before your score appears. Think of a race where you time yourself, and then send the video so that the officials can check it. LMArena is different: there, real users vote. To prepare for any of these boards, read Part A.

### SWE-bench

**What it ranks.** [SWE-bench](https://www.swebench.com) gives the agent real bug reports from GitHub Python projects. The agent must change the code so that the project's own tests pass. The score is the percentage of bugs it fixes. The best-known list is **SWE-bench Verified**: 500 tasks that people checked by hand to make sure that each one is fair.

**Who runs it.** You run it, then you submit.

**How to get on.** Open a pull request (a request to add your files) to the SWE-bench experiments repository. Include the agent's answers (the predictions), the evaluation logs, the full **trajectories** (the record of every step the agent took) and a short file that describes the model and the harness. Read the current rules first, because they change.

### Terminal-Bench

**What it ranks.** [Terminal-Bench](https://www.tbench.ai) gives the agent jobs in a real terminal: build a program, set up a server, fix a broken script. Each task runs in its own **container** (a small, sealed computer inside your computer) and has its own tests. Version 2.0 has 89 tasks.

**Who runs it.** You run it with the official harness, then you submit.

**Note.** Terminal-Bench tasks are long, and each one has a time limit. A slow server fails tasks that it could solve. So speed (Part B) has a big effect on this score. Part C explains why.

### Aider Polyglot

**What it ranks.** The [Aider leaderboard](https://aider.chat/docs/leaderboards/) asks the model to solve 225 programming exercises in six languages (C++, Go, Java, JavaScript, Python and Rust). It also shows how much the full run cost.

**Who runs it.** You run the Aider benchmark, then you open a pull request to the Aider repository with the result file.

### LiveCodeBench

**What it ranks.** [LiveCodeBench](https://livecodebench.github.io) uses new contest problems from LeetCode, AtCoder and Codeforces. You can choose only problems that were published after the model was trained. So the model cannot have seen the answers (see "contamination" in A.2).

**Who runs it.** The maintainers evaluate models. You can also run the open code and send results through the project's GitHub.

### LMArena WebDev

**What it ranks.** [LMArena](https://lmarena.ai) shows a user two answers from two hidden models. The user votes for the better one. The WebDev board uses web development tasks. The votes become a score, the same way chess ratings work.

**Who runs it.** Real users vote. The LMArena team decides which models to add.

### Summary

| Leaderboard | Ranks | Who runs it | How to get on |
|---|---|---|---|
| SWE-bench | GitHub issues fixed | you | PR with logs and trajectories |
| Terminal-Bench | terminal tasks solved | you | run the official harness, submit |
| Aider Polyglot | exercises solved, cost | you | PR with the result file |
| LiveCodeBench | new contest problems | maintainers or you | through the project GitHub |
| LMArena WebDev | human votes | users | the team adds models |

---

# Inference engine leaderboards

![Three inference engine leaderboards, what each ranks and who runs the test](images/01b-inference-engine-leaderboards.png)

These leaderboards ask a different question: **how fast does this server answer?** They measure the inference engine (for example vLLM or SGLang), the GPUs and the settings, for any kind of request. They do not ask whether the answers are good at coding.

Each one gets its numbers in its own way:

- **Artificial Analysis measures you.** Their team sends requests to your public endpoint and times the answers with their own stopwatch. You cannot send your own numbers.
- **OpenRouter uses real traffic.** The speed comes from the requests of real users, all day.
- **MLPerf Inference checks your run.** You run the official test yourself, and the results are reviewed before they are published.

To prepare for any of these boards, read Part B.

### Artificial Analysis

**What it ranks.** [Artificial Analysis](https://artificialanalysis.ai) compares models and the companies that serve them (the **providers**). For each provider of each model, it shows the output speed (TPS for one user), TTFT, the total time for one answer, and the price. It also publishes quality scores from its own tests, but its provider pages are about speed and price.

**Who runs it.** They measure you. They send test requests to public endpoints many times a day and show the median (the middle value).

**How to get on.**

1. Serve the model on a public API endpoint that paying users can use.
2. Contact Artificial Analysis and ask them to add your endpoint.
3. They measure it from their own computers.

**How to prepare.** Their numbers come from many measurements over time, not from one good moment. So:

- Before you contact them, measure your endpoint with your own copy of their method (B.4).
- Keep spare capacity. If other customers fill your GPUs during their test, your measured speed drops.
- Do not change the server settings without a test. One untested change can lower your speed for days.

### OpenRouter

**What it ranks.** [OpenRouter](https://openrouter.ai) is a single API that sends each request to one of many providers. Its model pages show the live speed and latency (waiting time) of each provider, measured from real traffic. Its rankings also show which models people use most.

**How to get on.** Become a provider on OpenRouter. Then real traffic measures you. A fast and reliable provider gets more requests.

### MLPerf Inference

**What it ranks.** [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) is run by MLCommons, an industry group. It compares complete systems (the hardware plus the software) on fixed models, including large language models. It reports throughput (the total tokens per second of the whole server) in two main scenarios: **Offline** (as fast as possible, with no limit on waiting time) and **Server** (as fast as possible while each request still meets a latency limit).

**Who runs it.** You run it with the official rules, and other submitters and MLCommons review the results before they are published. Results come out in fixed rounds, a few times a year. Submitters are usually member organisations of MLCommons. Read the current rules to see what is needed.

**Note.** MLPerf is mostly for hardware and system companies. It is the board where total throughput (B.1) matters most.

### Summary

| Leaderboard | Ranks | Who runs it | How to get on |
|---|---|---|---|
| Artificial Analysis | speed for one user, TTFT, price | the AA team | public endpoint, then contact them |
| OpenRouter | live speed and latency | real traffic | become a provider |
| MLPerf Inference | total throughput | you, then review | submit in a round |

---

# Part A: Benchmark a coding agent

## A.1 What a coding agent does

A **coding agent** is a model that works on code in many small steps. For example, you tell it: "the login test fails, fix it". The agent then:

1. reads the test file,
2. runs the test and reads the error,
3. opens the login code,
4. changes one line,
5. runs the test again,
6. sees that it passes, and stops.

Each of these steps is one **turn**: one request to the model and one answer. A real task can take 20 to 100 turns.

The model alone cannot run tests or open files. The **harness** does that. It shows the model a list of tools, runs the tool that the model asks for, and gives the result back to the model. SWE-agent, mini-SWE-agent, OpenHands and Terminus (the reference agent of Terminal-Bench) are examples of harnesses.

## A.2 Choose the right benchmarks

Each benchmark tests a different skill, so use more than one.

| Benchmark | What it tests | Size |
|---|---|---|
| **SWE-bench Verified** | Fixing real bugs in real Python projects. | 500 tasks, checked by people |
| **Terminal-Bench** | Doing jobs in a terminal: build, configure, repair. | 89 tasks in version 2.0 |
| **Aider Polyglot** | Editing code in six languages to solve exercises. | 225 exercises |
| **LiveCodeBench** | Solving new contest problems. | grows over time |

#### Contamination: when the model has seen the answers

A model learns from a huge amount of internet text. If the answers to a benchmark were on the internet, the model may have learned them by heart. This is **contamination**.

For example: before, a model scores 70% on an old public benchmark. After you give it new problems of the same kind that it never saw, it scores 45%. The 70% was partly memory.

LiveCodeBench avoids this with dates: it uses only problems from after the model's training. Fixed sets such as SWE-bench Verified cannot fully avoid it. So also test with a few private tasks of your own.

## A.3 Keep the harness the same

The same model can get very different scores in two different harnesses. One harness may show the model better error messages, or give it more turns, or handle a failed tool call better.

So when you compare **models**, use one harness for all of them, and do not change it. Then any difference in the score comes from the model. When you compare **harnesses**, do the opposite: keep the model the same.

## A.4 Set up the server for agents

Many low agent scores come from server settings, not from the model. Check these before you run anything. Each item shows what goes wrong and what it looks like when it is right.

**Tool-call parser.** The model writes its tool requests as text, for example "call `run_tests` with `path=tests/`". The server must turn that text into a structured **tool call** that the harness can read. A **parser** is the part of the server that does this, and each model family needs its own one.

- Before (no parser): the harness asks for tools with `tool_choice=auto`, and the server answers with an error (HTTP 400). Every task fails at turn one.
- After: in vLLM, start the server with `--enable-auto-tool-choice --tool-call-parser <name for your model>`. The harness gets clean tool calls.

**Reasoning parser.** A "thinking" model first writes its private reasoning, then its answer. The server must put the reasoning in a field called `reasoning_content` and the answer in `content`.

- Before (no reasoning parser): the long reasoning text goes into the answer, and the harness cannot find the tool call inside it.
- Also wrong (a reasoning parser on a model that does not think): the server may put the whole answer in `reasoning_content`, and `content` stays empty. The harness sees an empty answer.
- After: use the reasoning parser only for models that think, and send one test request to check both fields.

**Context length.** An agent sends the whole conversation in every turn, so the requests become very long. `--max-model-len` is the longest request the server accepts.

- Before: in one of my runs, a limit copied from a vendor's documentation was too short, and the server refused 72% of the requests. In another run, a limit of half the model's maximum refused 2%.
- After: use the full context length that the model supports.

**Output limit.** A thinking model can spend all of `max_tokens` (the most tokens it may write) on reasoning.

- Before: the limit runs out during the reasoning. The answer is empty, but you still pay for all the tokens, and the task fails.
- After: give thinking models a large output limit, and count how many answers come back empty.

**Time limits.** Set the harness timeout longer than your longest answer. A timeout counts as a failed task.

## A.5 The procedure

1. Choose the benchmarks and write down the version of each.
2. Choose the harness and keep it the same.
3. Set up the server (A.4). Send one test task and read the full output yourself.
4. Run each benchmark at least three times. Agents give different results from run to run.
5. Record pass@1 for each run, the average and the range.
6. Record the cost and the total time.
7. Keep all logs and trajectories. Most leaderboards ask for them.

**Report the range, not only your best run.** For example, three runs give 61%, 64% and 66%. Report "64% (61 to 66, 3 runs)". If another model got 65%, you cannot say it is better: the gap is smaller than the normal change between runs.

---

# Part B: Measure and increase LLM inference TPS

Part B is about the server. Forget agents for now. Everything here is the same for any program that sends requests to an LLM.

## B.1 What TPS means, and why there are two kinds

**TPS** is tokens per second: how fast text comes out of the model. But people use it for two different things.

![Per-user TPS falls and total server throughput rises as more users share one GPU, with example numbers](images/03-two-kinds-of-tps.png)

- **Speed for each user** (also called output speed). How fast the words appear for one person. This is what a user feels, and what Artificial Analysis shows.
- **Total throughput** (the speed of the whole server). All tokens from all users added together. This is what decides the cost: more tokens per GPU per hour means cheaper tokens. MLPerf measures this.

Think of a bus. With one passenger, the bus drives fast and stops only once. With forty passengers, it stops often, so each passenger arrives later. But the bus carries forty people on one trip. One passenger is "speed for each user". Forty passengers is "total throughput".

The figure shows this with example numbers. One user gets 120 tokens per second. With 128 users, each one gets only 25, but the server makes 3,200 in total. **So a TPS number means nothing unless you say which kind it is and how many users were active.**

Two more timing words:

- **TTFT** (time to first token): the wait before the first word. It includes the time in the queue and the time to read the prompt.
- **ITL** (inter-token latency) or **TPOT** (time per output token): the gap between two words. It is 1 divided by the speed for each user. For example, 50 TPS means a gap of 20 milliseconds.

## B.2 Prefill and decode: reading and writing

A model works in two phases.

- **Prefill** (reading): the model reads the whole prompt in one go. All the prompt tokens are processed together, so the GPU's calculators are busy. Prefill decides most of the TTFT.
- **Decode** (writing): the model writes the answer one token at a time. For each new token, the GPU must read **all** of the model's weights (the learned numbers) from memory again, but it does only a little maths with them.

Think of a cook who must walk to the pantry and carry the whole cookbook back for every single word of the recipe they write. The walking, not the cooking, takes the time. In decode, the "walking" is reading the weights from memory. So decode speed depends mostly on **memory bandwidth** (how many bytes per second the GPU can read from its memory).

## B.3 The ceiling: the fastest one user can go

Because decode reads all the weights for every token, you can calculate the top speed for one user before you measure anything:

```
top speed for one user  =  memory bandwidth  /  bytes of weights read for each token
```

![The decode speed ceiling for one request: memory bandwidth divided by the bytes of weights read per token, for bf16, FP8 and FP4 weights of a 31B model](images/04-the-tps-ceiling.png)

**Example.** A model with 31 billion weights, stored in bf16 (2 bytes for each weight), has 62 GB of weights. A GPU that can read about 8 TB (8,000 GB) per second can read those 62 GB about 129 times each second. So one user can never get more than about 129 tokens per second from this setup, unless you change the method (B.6).

For a **mixture-of-experts** (MoE) model, only some parts of the model (the "experts") are used for each token. Use only the **active** weights in the formula, which makes the ceiling much higher.

Why calculate this first? Because it tells you where you are:

- If you measure 60 TPS and the ceiling is 129, the server wastes half its time somewhere. Settings can help a lot.
- If you measure 125 TPS, you are near the limit. Only a change of method (fewer bytes, or more than one token for each read) can help.
- If you measure 300 TPS, your measurement is wrong (see B.5).

## B.4 Measure TPS the way the leaderboards do

Build your own small copy of the leaderboard method. Then you know your number before they publish it.

1. **Use fixed prompts.** Use the same prompt lengths every time, for example 1,000 input tokens and 10,000 input tokens. Freeze this set and give it a version number.
2. **Fix the output length.** Ask for the same number of output tokens every time. A short answer and a long answer give different speeds.
3. **Use streaming.** With streaming, the server sends each token as soon as it is ready. Measure TTFT at the first token. Measure the speed from the first token to the last.
4. **Test separate cases.** Measure one user at a time. Then measure with several users at the same time. Never mix them in one number.
5. **Report the median.** Send at least 30 requests for each case, and report the middle value. Also report p99 (the slowest 1%), which shows the bad moments.

Both main engines include a tool for this. For example, with vLLM:

```bash
vllm bench serve --model <model> --dataset-name random \
  --random-input-len 1000 --random-output-len 500 \
  --num-prompts 100 --max-concurrency 1 --ignore-eos
```

- `--random-input-len 1000 --random-output-len 500`: fixed prompt and answer lengths.
- `--max-concurrency 1`: one user at a time. Run it again with 8, 32 and so on.
- `--ignore-eos`: the model must write all 500 tokens, even if it wants to stop early. This keeps the output length fixed.

SGLang has the same kind of tool: `python -m sglang.bench_serving` with similar flags. Both print TTFT, ITL, and the output throughput.

Example of the cases from one of my harnesses:

| Case | Prompt length | Users at the same time |
|---|---|---|
| short, one user | 1,000 tokens | 1 |
| long, one user | 10,000 tokens | 1 |
| short, busy | 1,000 tokens | 10 |

## B.5 Count the tokens yourself

A speed number is only as good as the token count. When you measure through an API, especially a gateway (a service that forwards your request to a provider), check these:

- **Count the tokens in the stream yourself.** Do not trust the token count that each provider reports. For example, in one of my tests, two identical requests through one gateway reported 445 and 5,000 output tokens.
- **Read every reasoning field.** Some gateways send the reasoning in `reasoning` or `reasoning_details`, not in `reasoning_content`. If you count only one field, you miss tokens, or count them twice. In one of my tests, this mistake gave a speed of 15,000 tokens per second.
- **Compare with the ceiling.** A number above the ceiling in B.3 is a bug in the measurement, not a fast server.

## B.6 Increase TPS

Some changes make each user faster. Others make the whole server carry more. Know which one you need before you change anything.

### Make each user faster

| Change | What it does, in plain words | Before and after (example) | Risk |
|---|---|---|---|
| **Quantization** (FP8, FP4) | Store each weight in fewer bytes. Less to read for each token, so the ceiling goes up. | A 31B model in bf16 reads 62 GB for each token. In FP8 it reads 31 GB, so the ceiling doubles. | Can lower quality. Always test (B.7). |
| **Speculative decoding** (MTP, EAGLE, a small draft model, n-gram) | A small, fast helper guesses the next few tokens. The big model checks all the guesses in one read of its weights and keeps the correct ones. | Before: one token for each read. After: often 2 or 3 tokens for each read, so 1.5x to 3x faster for one user. | Gains drop when the guesses are often wrong. |
| **Tensor parallel** (`-tp 2`, `-tp 4`) | Split each layer across several GPUs. Each GPU reads only its share of the weights, so they read faster together. | 2 GPUs: up to almost 2x the bandwidth for one user. | Needs fast links between the GPUs. |
| **CUDA graphs** | Record the GPU's work for one step once, then replay it. The CPU no longer prepares each small piece every time. | Removes small CPU delays in decode, which matter most for small models. | Some kernels do not work with graphs. |
| **Fewer users at the same time** | Let fewer requests share the GPU (`--max-num-seqs`). | Each user is faster, but the server carries less in total (B.1). | Higher cost for each token. |

**Why speculative decoding works so well on code.** Code repeats itself: variable names, file paths, lines from earlier in the file. Guesses are easy, so most of them are accepted. **N-gram drafting** needs no extra model at all. It looks for the last few tokens earlier in the prompt and guesses that the text continues the same way, which is very good for edits that copy existing text.

### Make the whole server carry more

| Change | What it does, in plain words | Before and after (example) | Risk |
|---|---|---|---|
| **More users in each step** (`--max-num-seqs`, `--max-num-batched-tokens`) | One read of the weights serves every user in the step, like the bus in B.1. | 1 user: 120 tokens per second in total. 32 users: about 1,900 in total (figure in B.1). | Each user gets slower. |
| **More memory for the KV cache** (`--gpu-memory-utilization` in vLLM, `--mem-fraction-static` in SGLang) | The **KV cache** is the memory that holds the notes for every token of every active request. More KV memory means more requests fit at the same time. | Before: room for 20 requests. After: room for 30. | Too much can run the GPU out of memory. |
| **FP8 KV cache** (`--kv-cache-dtype fp8`) | Store those notes in 1 byte in place of 2. | About twice as many requests fit. | Small quality loss on some models. Test it. |
| **Prefix caching** | Keep the KV notes of prompt beginnings that repeat, for example a long system prompt. Do not read them again. | A shared 5,000-token system prompt is read once, not once for every request. Much lower TTFT. | Uses memory. |
| **Chunked prefill** | Cut a long prompt into pieces and mix them with decode work, so one long prompt does not freeze everyone else. | Steadier speed for users who are already receiving an answer. | Can raise TTFT a little for the long prompt. |
| **More copies of the model** (data parallel) | Run several copies of the model on more GPUs, and share the users between them. | 2 copies: about 2x the total. | Needs more GPUs. |

## B.7 Check quality after every speed change

Every speed change can change the output, sometimes badly. A server that is fast and wrong is worse than a slow one, because it looks like a win.

For example: before, the server answers "What is 17 x 23?" with "391". After a new quantization setting, it is 30% faster and answers "381". Without a check, you would publish the faster number.

After each change:

1. Send a **correctness gate**: a few short fixed prompts with one known correct answer each. If any answer is wrong, stop.
2. Run a small quality test, for example part of a standard benchmark.
3. Compare with the result from before the change.

---

# Part C: Where the two topics meet

Speed and quality are separate measurements. But for coding agents, speed changes the quality score. Here is why.

![Where the time of one agent task goes: TTFT, decode and tool work for each turn, before and after optimisation, with example numbers](images/05-where-agent-time-goes.png)

An agent does many turns, and each turn pays the waiting time again:

```
task time  =  for every turn: (TTFT + output tokens / speed for one user)  +  time running tools
```

**Example.** A task has 30 turns. Each turn writes 400 tokens and runs tools for 3 seconds.

- Before: TTFT 1.5 seconds and 60 tokens per second. The task takes about 335 seconds.
- After (prefix caching cuts TTFT, speculative decoding and FP8 raise the speed): TTFT 0.4 seconds and 150 tokens per second. The same task takes about 182 seconds.

Now remember that agent benchmarks have **time limits**. In one of my Terminal-Bench runs, more than half of the 89 tasks stopped at the 900-second limit. A faster server finishes more tasks inside the limit, so the same model gets a higher score.

Two speed changes from Part B help agents the most:

- **Prefix caching.** Each turn sends the whole conversation again, and most of it is the same as last turn. With prefix caching, the server reuses the notes of the earlier turns. In one of my load tests, 96.7% of the input tokens came from the cache. So TTFT stays low even when the conversation gets long.
- **Speculative decoding.** Agents write code, and code is easy to guess (B.6).

But keep the two reports separate. An inference engine leaderboard does not tell you if the agent is good. A coding agent leaderboard does not tell you if the server is fast. Publish both, and say which is which.

---

# Checklist before you submit

**For a coding agent leaderboard:**

1. Write down the model version, the harness version and the benchmark version.
2. Check the tool-call parser, the reasoning parser, the context length and the output limit.
3. Send one test task and read the whole trajectory yourself.
4. Run each benchmark at least three times. Report the average and the range.
5. Keep all logs and trajectories.
6. Read the current submission rules.

**For an inference engine leaderboard:**

1. Calculate the ceiling for one user (B.3).
2. Measure with fixed prompts, fixed output lengths and streaming. Report the median and p99.
3. Say which TPS you mean (for each user, or total) and how many users were active.
4. Count tokens from the stream, and read every reasoning field.
5. Make sure no number is above the ceiling.
6. Run the correctness gate after every speed change.
7. For Artificial Analysis, which measures you itself, keep spare capacity and do not change settings without a test.

# Summary

Start from the leaderboard you want, because it tells you what to measure and who measures it. For coding agent leaderboards, keep the harness the same, set up the parsers and the context length correctly, and run each benchmark more than once. For inference engine leaderboards, know the two kinds of TPS, calculate the ceiling, measure the way the board does, and check quality after every speed change. The two topics meet in agent task time: a faster server finishes more agent tasks inside the time limit.
