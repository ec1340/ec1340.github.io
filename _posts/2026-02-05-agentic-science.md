At the end of 2026, what will scientific research look like?

I find myself thinking about this question more and more. There seems to be a steady stream of announcements for new foundation models for biology or a better AI scientist platform, each promising to improve the current status quo of scientific research. On the data side, there is an effort by biotech companies to publish open-access [large-scale](https://huggingface.co/datasets/tahoebio/Tahoe-100M) [datasets](https://www.omtx.ai/), which the field desparately needs . Outside of the life sciences, developers have created sophisticated ways to multiplex [agent sessions](https://simonwillison.net/2025/Oct/5/parallel-coding-agents/)  leading to bolder, autonomously-completed [software projects](https://www.anthropic.com/engineering/building-c-compiler). Lastly, the next AGI-adjacent foundation model seems to always be only a few months away.*


Amidst these rapid advancements, scientific research is guaranteed to change.

As a [computational biology] researcher in graduate school, the path of research was steady and to a degree, predictable. It starts with an interesting biological question that has crossed the “addressable threshold”, owing to either a new dataset, recent discovery, or a better modeling technique. Conversations with domain experts and about relevant literature ensue and after some exploratory work, there is an early hypothesis and usually an outline of a paper. Now for the relevant and fun part - **the scientific process of trying things out and seeing what works.** 

But what does this *exploration* phase look today?

Agents introduce a new approach to science work. In the interactive setting, they serve as powerful tool capable of accelerating key steps. An early signal of this was in literature search, where [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) + LLM was incredibly effective for searching over documents. Improved tool use capabilities in the base foundation model extended the action space for science agents, giving them effectively full access to the scientific toolbox. The most powerful tool for agents however is the *code sandbox*. It's clear now that the best performing agents operate within the coding domain, where the versatility of code is key.


I won't discuss autonomous science agents here [topic for a future post] but there is an growing body of [evidence](https://arxiv.org/abs/2408.06292) that science agents can excel at longer, publication-scope tasks. It's a exciting thought exercise to map out the implications of fully-automated hypothesis generation and testing.

For now, its safe to say that the productivity of each scientist will benefit from the use of agentic tools, whether that be within a specialized AI scientist platform or by using a developer agent to [guide experimentation](https://www.goodfire.ai/blog/you-and-your-research-agent). The greatest immediate gains will come from learning how and importantly, when to delegate research to agents


--

\* Coincidentally, Opus 4.6 was released during the intial writing of this post