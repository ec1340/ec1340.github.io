---
layout: post
title: Molecule Reasoning Models
date: 2026-03-08
math: true
category: ml
---


I’ve been thinking a lot about molecule design lately, and in particular about whether LLMs could ever get genuinely good at reasoning through it. A paper that is a great example of current approaches and a nice starting point is [Ether0](https://github.com/Future-House/ether0).

In this post, I'll be walking through how [Ether0](https://github.com/Future-House/ether0) used post-training techniques like SFT and GRPO to instill molecule reasoning into base LLMs. I'll also dig into a few key ideas in the training setup. 

There’s been a lot of interest in the AI for science field around specialized reasoning models with most showing very early progress. They often still fail in many obvious ways and I think the Ether0 paper is a useful place to ask what the path to get to a strong model might look like.

## What this post covers


This is a walkthrough of the Ether0 paper, looking at how they assembled their datasets and trained an open source model to achieve a big performance jump on a variety of chemistry tasks. I'll mainly be focusing on the way the model was trained so let's start with a quick overview of the stages of training an LLM.

### Stages of Training an LLM

- **Pre-training**:
Their training methods fit after the **pre-training** stage of the base model. During pre-training, the model is exposed to an enormous amount of data and learns to pick up on general patterns and relationships in the data. 

- **Post-training**: Uses more tailored data to adjust the model's abilities towards a specific type of behavior. Often this is to elicit reasoning behavior but can also include things like alignment and instruction-following. 

## Ether0 Training Pipeline

### Ether0 Training Data Construction

In Ether0, ~640,000 chemical reasoning problems across 18 tasks were generated using real-world molecules. The authors pulled in data from ChEMBL, COCONUT, Pubchem, AqSolDB, and other sources as a source of ground truth molecule : molecule label samples. From these sources, they generated verifiable tasks like solubility prediction, IUPAC → SMILES conversion and pka prediction. The thinking is that these tasks, while generally small in scope, form important primitives that a powerful molecule reasoning model would require. 


### Four-Step Training

The authors use a 4 step post-training process. In recent post-training setups, multiple phases of learning appear to complement one another and help the model better learn capabilities. For example, SFT helps models form reasoning sequences whereas RL shapes the reasoning ability of the model toward higher quality, more efficient paths that result in correct answers.

#### Step 1: Supervised Fine-tuning (SFT) on Chain of Thought (CoT) data

The authors begin with a SFT step to provide a strong initial grounding in the task domain. To form the bank of reasoning traces, they use DeepSeek R1 on a subset of the Ether0 data and collect the CoT data. Its worth noting that while the DeepSeek model fails to reach correct answer in the vast majority of cases **(<1% success rate)**, these traces are still considered useful for learning reasoning as they provide examples of structured reasoning sequences.


After training, the performance on the multiple choice section (MCQA) of the Ether0 benchmark jumps from 0 to 50% at the end of the SFT stage. More moderate (1%-20%) improvements are made on other molecule tasks. 


From the objective function, the model learns to maximize probability along observed reasoning token sequences and implicitly push the model towards generating CoT in its responses. However at this stage, the model has learned to mimic reasoning sequences rather than internalize reasoning capabilities.


Given a set of demonstration sequences Ddemo, supervised fine-tuning (SFT) minimizes the crossentropy loss over the dataset:

We can see this in the SFT objective function. If $D_{\mathrm{demo}}$ is the set of demonstration sequences and $s$ is a sequence, the SFT stage minimizes the negative log-likelihood:

$$
L_{\mathrm{SFT}}
= -\frac{1}{|D_{\mathrm{demo}}|}
\sum_{s \in D_{\mathrm{demo}}}
\sum_{t=1}^{|s|}
\log \pi(s_t \mid s_{<t}).
$$


#### Step 2: Specialist RL

As the tasks vary fairly widely in difficulty and type, the authors distribute the 18 tasks across 7 RL runs. All MCQA categories are combined into a single run. To train each specialist model, they use group relative policy optimization (GRPO). Since it’s becoming such a popular technique in this flavor of approaches (Ether0, [Chem-R](https://arxiv.org/abs/2510.16880), [Biomni-R0](https://biomni.stanford.edu/blog/biomni-r0-technical-report/)), its worth spending a little while discussing GRPO

##### Group Relative Policy Optimization (GRPO)

RL is about optimizing a policy to maximize the expected reward. Within RL, there are policy gradient methods which optimize the parameters of a policy using signal from a reward. This is generally done through an advantage estimate (how much better an action is from a reward perspective relative to a baseline), which helps in reducing variance and speeding up learning. 


GRPO fits into the policy-gradient methods class of RL algorithms and is relatively simple to use as it only uses 1) multiple outputs from a policy (LLM), 2) works well with verifiable or rule-based and deterministic rewards (often just a python function) and 3) token-level probabilities for the PPO ratio calculation and optional KL-divergence term. 

##### GRPO Objective

Looking at the GRPO objective, there a few terms worth diving deeper into.

Given a single problem $x$ and a group of completions $\{y_i\}$, the per-group objective is:

$$
J(\theta, x, y_1, \ldots, y_G)
= \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|}
\left(
\operatorname{clip}\!\left(
\frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{
\pi_{\theta_{\mathrm{old}}}(y_{i,t} \mid x, y_{i,<t})
},
A_i, \epsilon
\right)
- \beta \hat{D}_{\mathrm{KL}}[\pi_\theta \| \pi_{\mathrm{ref}}; x, y_{i,\le t}]
\right).
$$

The current policy is $\pi_\theta$ and previous iteration's policy is $\pi_{\theta_{\mathrm{ref}}}$. The starting policy is $\pi_{\theta_{\mathrm{ref}}}$.

- **KL divergence**: keeps the updated policy similar to the base policy such that the starting action space remains viable.


- **PPO likelihood ratio**: between the new policy and the old policy (from the last iteration). Intuitively, this weights the reward signal by the strength of the policy change. This helps in sharpening the probability distribution around high-reward sequences while maintaining smaller policy updates.

For a sampled completion $y_i$, the relative weighting comes from the PPO ratio

$$
\rho_{i,t}(\theta)
= \frac{
\pi_\theta(y_{i,t} \mid x, y_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(y_{i,t} \mid x, y_{i,<t})
}.
$$

GRPO then uses group-normalized rewards. If we sample $G$ outputs for the same prompt and get rewards $r_1, \dots, r_G$, a typical group-relative advantage is

$$
A_i
= \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}
{\mathrm{std}(r_1,\dots,r_G) + \varepsilon}.
$$

So the update depends on how much better one sampled reasoning trace is than the other samples for the same task, not just on its absolute reward.

Notably, the KL divergence term tie the learned policy [within some soft divergence bounds](https://arxiv.org/abs/2504.13837) around the initial policy $\pi_{\theta_{\mathrm{ref}}}$, with update steps weighted by the PPO-likelihood ratio (+clipping) ensuring gradual changes. Since we’re using a pre-trained LLM as a policy, there is an already baked in prior distribution over token space that is dependent on the pre-training setup used for the base model. Each base model learns a slightly different prior over world data. 

How much this prior $\pi_{\theta_{\mathrm{ref}}}$  of differs between models is an open question but I’d hypothesize that the differences in the learned priors are amplified in smaller data domains like chemistry. Its very possible that chemistry data makes up a very small fraction of the training corpus of large base model with molecule data being an even smaller, miniscule fraction. As a result, differences in composition and size of this type of data in each model’s pre-training corpus can lead to differences in the learned prior around things like molecule quality and the relationship between structure and chemical property. 

With this in mind, recall that the KL divergence term keeps the learned policy similar to the reference policy. Specifically, the KL term as written in the GRPO objective contributes a high penalty if probability mass is moved away from areas it previously covered. In other words, if the reference model considers a molecule to be likely, the learned policy will be biased to continue to see that molecule as likely. 

So in the (very often observed) case that the initial base model doesn’t understand chemistry, a misaligned prior might actually leak into the post-training process and introduce a subtle tension in the learning process. The model may receive conflicting gradient signals when encounter molecules the base model hallucinates as being valid but the RL objective actually deems as invalid. This might point to the importance of the initial SFT stage or maybe a greater emphasis on things like [mid-training](https://vintagedata.org/blog/posts/what-is-mid-training). 

At the level of the objective function, there are possible remedies. The [DAPO](https://arxiv.org/abs/2503.14476) objective function removes the KL divergence term and adjusts the clipping range. The asymmetric clipping they use is motivated by the entropy collapse problem for token probabilities, where high-probability tokens get upweighted and low probability tokens vanish over time. Raising the top clip value gives low probability tokens a chance at larger gradient updates. 

In Ether0, they address the molecule quality issue by introducing a molecule quality bonus reward during the final GRPO step. This brings us to the reward system in GRPO setups!

##### Rule-based, Deterministic Rewards

Many recent RL training pipelines use rule-based rewards instead of learned reward models. The combined simplicity and effectiveness of these rewards across different task domains have made them very popular in LLM-based RL.

In the DeepSeek R1 paper and Ether0 paper, the reward appears as 

$$
r(y) = r_{\mathrm{format}}(y) + r_{\mathrm{acc}}(y).
$$

The formatting reward checks for <thought> and <answer> tagged sections which trains the model to separate the reasoning and answer components of the response. The accuracy reward checks for the correct answer token and is where domain specific reward functions are introduced. In Ether0, these accuracy rewards check for a correct solution to the chemical reasoning task.


For the final GRPO stage in Ether0, they add a molecule quality bonus reward to the accuracy reward.

$$
r(y) = r_{\mathrm{format}}(y) + r_{\mathrm{acc}}(y) + \lambda \, q(y),
$$

where $q(\cdot)$ is a molecule-quality score.

Andrew White has a really interesting [blog post](https://diffuse.one/p/m1-000) that dives into the development of chemistry reward functions. It's a great read and I highly recommend checking it out. He mentions trying out domain-specific models (GNNs, classifiers) during the development process of Ether0. I wonder if future post-training pipelines will make greater use of these. 

##### Addressing problematic tasks using an advantage-based curriculum

From the advantage estimate formula, you might be wondering what happens if all outputs receive the same reward. In Ether0 development process, the authors mention that this is especially an issue. This can be the case is the task is either very difficult or very trivial. In these cases, the advantage would be 0 for the group and the contribution to the gradient would be dropped. This leads to inefficient training with lower effective batch sizes with prompts that contribute almost no gradient.

You can see this directly from the group-relative advantage above: if $r_1 = \dots = r_G$, then $A_i \approx 0$ for every sample in the group, so that prompt contributes almost no learning signal.

To address this, they use a dynamic advantage-based curriculum which maintains a buffer of non-trivial tasks. At each training iteration, the sample from this buffer to ensure each batch maintains a decently strong loss signal.



#### Step 3: Generalist Distillation


The last two phases are very similar to the first two so I won’t spend that much time on them.

At the end of phase 2, we are left with 7 SFT + GRPO-trained models which have gained greater reasoning ability for a class of chemistry tasks.  The authors distill the specialist model using SFT on a base model. They collect reasoning traces from the specialized models and using a combination of LLM-as-a-Judge and regex filtering, assemble a refined set of traces. 

#### Step 4: Generalist RL

For the final post-training phase, the authors train the model using GRPO across all task categories. They use the advantage-based curriculum again. As mentioned above, this phase of GRPO includes a molecule quality bonus in it’s reward function. 

## A Molecule Reasoning Model in a Land of Agents

With agentic scientists, its possible to solve many of small molecule tasks  with a well designed agentic harness. Access to cheminformatics tools and databases bestows the base LLM with a much greater ability to validate molecules during it’s step by step reasoning process. This has led to agents that can work with chemical data quite well, especially when working within a code sandbox.

While agentic harnesses confer immediate gains in problem solving capabilities, enduring gains in molecule problem solving might actually come from better molecule reasoning base models. The reason for this ties to a fundamental challenge in the molecule design domain and many other scientific disciplines - *search.* 

For tasks concerning molecules, a core challenge is searching through an enormous design space, with many constraints and multi-scale interactions. With an agentic system, the search process feels very familiar. Agents iteratively loop through reasoning steps and tool calls to generate and check molecule designs. But efficiently navigating through this highly-complex search space can perhaps be done in a better way if the underlying search model possess a level of a deep generalizable knowledge of molecules. With enough inference-time compute, molecule reasoning models may generate novel, high-quality molecules with chemically grounded rationale. 


---

