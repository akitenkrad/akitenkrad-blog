---
draft: false
title: "arXiv @ 2023.10.12"
date: 2023-10-12
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.12"
    identifier: arxiv_20231012
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (49)](#cscl-49)
- [cs.AI (12)](#csai-12)
- [cs.SD (3)](#cssd-3)
- [cs.HC (4)](#cshc-4)
- [cs.CV (21)](#cscv-21)
- [cs.NI (2)](#csni-2)
- [cs.LG (30)](#cslg-30)
- [cs.CR (5)](#cscr-5)
- [quant-ph (1)](#quant-ph-1)
- [cs.RO (8)](#csro-8)
- [cs.DC (2)](#csdc-2)
- [cs.SE (7)](#csse-7)
- [cs.CE (1)](#csce-1)
- [cs.CY (4)](#cscy-4)
- [math.OC (1)](#mathoc-1)
- [cs.ET (1)](#cset-1)
- [eess.SY (1)](#eesssy-1)
- [cs.DL (1)](#csdl-1)
- [cs.IR (2)](#csir-2)
- [cs.MM (1)](#csmm-1)
- [eess.IV (2)](#eessiv-2)
- [cs.AR (1)](#csar-1)

## cs.CL (49)



### (1/159) Crossing the Threshold: Idiomatic Machine Translation through Retrieval Augmentation and Loss Weighting (Emmy Liu et al., 2023)

{{<citation>}}

Emmy Liu, Aditi Chaudhary, Graham Neubig. (2023)  
**Crossing the Threshold: Idiomatic Machine Translation through Retrieval Augmentation and Loss Weighting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, Machine Translation  
[Paper Link](http://arxiv.org/abs/2310.07081v1)  

---


**ABSTRACT**  
Idioms are common in everyday language, but often pose a challenge to translators because their meanings do not follow from the meanings of their parts. Despite significant advances, machine translation systems still struggle to translate idiomatic expressions. We provide a simple characterization of idiomatic translation and related issues. This allows us to conduct a synthetic experiment revealing a tipping point at which transformer-based machine translation models correctly default to idiomatic translations. To expand multilingual resources, we compile a dataset of ~4k natural sentences containing idiomatic expressions in French, Finnish, and Japanese. To improve translation of natural idioms, we introduce two straightforward yet effective techniques: the strategic upweighting of training loss on potentially idiomatic sentences, and using retrieval-augmented models. This not only improves the accuracy of a strong pretrained MT model on idiomatic sentences by up to 13% in absolute accuracy, but also holds potential benefits for non-idiomatic sentences.

{{</citation>}}


### (2/159) NEWTON: Are Large Language Models Capable of Physical Reasoning? (Yi Ru Wang et al., 2023)

{{<citation>}}

Yi Ru Wang, Jiafei Duan, Dieter Fox, Siddhartha Srinivasa. (2023)  
**NEWTON: Are Large Language Models Capable of Physical Reasoning?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-RO, cs.CL  
Keywords: GPT, GPT-4, Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.07018v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), through their contextualized representations, have been empirically proven to encapsulate syntactic, semantic, word sense, and common-sense knowledge. However, there has been limited exploration of their physical reasoning abilities, specifically concerning the crucial attributes for comprehending everyday objects. To address this gap, we introduce NEWTON, a repository and benchmark for evaluating the physics reasoning skills of LLMs. Further, to enable domain-specific adaptation of this benchmark, we present a pipeline to enable researchers to generate a variant of this benchmark that has been customized to the objects and attributes relevant for their application. The NEWTON repository comprises a collection of 2800 object-attribute pairs, providing the foundation for generating infinite-scale assessment templates. The NEWTON benchmark consists of 160K QA questions, curated using the NEWTON repository to investigate the physical reasoning capabilities of several mainstream language models across foundational, explicit, and implicit reasoning tasks. Through extensive empirical analysis, our results highlight the capabilities of LLMs for physical reasoning. We find that LLMs like GPT-4 demonstrate strong reasoning capabilities in scenario-based tasks but exhibit less consistency in object-attribute reasoning compared to humans (50% vs. 84%). Furthermore, the NEWTON platform demonstrates its potential for evaluating and enhancing language models, paving the way for their integration into physically grounded settings, such as robotic manipulation. Project site: https://newtonreasoning.github.io

{{</citation>}}


### (3/159) Answer Candidate Type Selection: Text-to-Text Language Model for Closed Book Question Answering Meets Knowledge Graphs (Mikhail Salnikov et al., 2023)

{{<citation>}}

Mikhail Salnikov, Maria Lysyuk, Pavel Braslavski, Anton Razzhigaev, Valentin Malykh, Alexander Panchenko. (2023)  
**Answer Candidate Type Selection: Text-to-Text Language Model for Closed Book Question Answering Meets Knowledge Graphs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Knowledge Graph, Language Model, QA, Question Answering, T5  
[Paper Link](http://arxiv.org/abs/2310.07008v1)  

---


**ABSTRACT**  
Pre-trained Text-to-Text Language Models (LMs), such as T5 or BART yield promising results in the Knowledge Graph Question Answering (KGQA) task. However, the capacity of the models is limited and the quality decreases for questions with less popular entities. In this paper, we present a novel approach which works on top of the pre-trained Text-to-Text QA system to address this issue. Our simple yet effective method performs filtering and re-ranking of generated candidates based on their types derived from Wikidata "instance_of" property.

{{</citation>}}


### (4/159) Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation (Yangsibo Huang et al., 2023)

{{<citation>}}

Yangsibo Huang, Samyak Gupta, Mengzhou Xia, Kai Li, Danqi Chen. (2023)  
**Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs.CL  
Keywords: AI, Falcon, LLaMA  
[Paper Link](http://arxiv.org/abs/2310.06987v1)  

---


**ABSTRACT**  
The rapid progress in open-source large language models (LLMs) is significantly advancing AI development. Extensive efforts have been made before model release to align their behavior with human values, with the primary goal of ensuring their helpfulness and harmlessness. However, even carefully aligned models can be manipulated maliciously, leading to unintended behaviors, known as "jailbreaks". These jailbreaks are typically triggered by specific text inputs, often referred to as adversarial prompts. In this work, we propose the generation exploitation attack, an extremely simple approach that disrupts model alignment by only manipulating variations of decoding methods. By exploiting different generation strategies, including varying decoding hyper-parameters and sampling methods, we increase the misalignment rate from 0% to more than 95% across 11 language models including LLaMA2, Vicuna, Falcon, and MPT families, outperforming state-of-the-art attacks with $30\times$ lower computational cost. Finally, we propose an effective alignment method that explores diverse generation strategies, which can reasonably reduce the misalignment rate under our attack. Altogether, our study underscores a major failure in current safety evaluation and alignment procedures for open-source LLMs, strongly advocating for more comprehensive red teaming and better alignment before releasing such models. Our code is available at https://github.com/Princeton-SysML/Jailbreak_LLM.

{{</citation>}}


### (5/159) Violation of Expectation via Metacognitive Prompting Reduces Theory of Mind Prediction Error in Large Language Models (Courtland Leer et al., 2023)

{{<citation>}}

Courtland Leer, Vincent Trost, Vineeth Voruganti. (2023)  
**Violation of Expectation via Metacognitive Prompting Reduces Theory of Mind Prediction Error in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06983v1)  

---


**ABSTRACT**  
Recent research shows that Large Language Models (LLMs) exhibit a compelling level of proficiency in Theory of Mind (ToM) tasks. This ability to impute unobservable mental states to others is vital to human social cognition and may prove equally important in principal-agent relations between individual humans and Artificial Intelligences (AIs). In this paper, we explore how a mechanism studied in developmental psychology known as Violation of Expectation (VoE) can be implemented to reduce errors in LLM prediction about users by leveraging emergent ToM affordances. And we introduce a \textit{metacognitive prompting} framework to apply VoE in the context of an AI tutor. By storing and retrieving facts derived in cases where LLM expectation about the user was violated, we find that LLMs are able to learn about users in ways that echo theories of human learning. Finally, we discuss latent hazards and augmentative opportunities associated with modeling user psychology and propose ways to mitigate risk along with possible directions for future inquiry.

{{</citation>}}


### (6/159) Why bother with geometry? On the relevance of linear decompositions of Transformer embeddings (Timothee Mickus et al., 2023)

{{<citation>}}

Timothee Mickus, Raúl Vázquez. (2023)  
**Why bother with geometry? On the relevance of linear decompositions of Transformer embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.06977v1)  

---


**ABSTRACT**  
A recent body of work has demonstrated that Transformer embeddings can be linearly decomposed into well-defined sums of factors, that can in turn be related to specific network inputs or components. There is however still a dearth of work studying whether these mathematical reformulations are empirically meaningful. In the present work, we study representations from machine-translation decoders using two of such embedding decomposition methods. Our results indicate that, while decomposition-derived indicators effectively correlate with model performance, variation across different runs suggests a more nuanced take on this question. The high variability of our measurements indicate that geometry reflects model-specific characteristics more than it does sentence-specific computations, and that similar training conditions do not guarantee similar vector spaces.

{{</citation>}}


### (7/159) Document-Level Supervision for Multi-Aspect Sentiment Analysis Without Fine-grained Labels (Kasturi Bhattacharjee et al., 2023)

{{<citation>}}

Kasturi Bhattacharjee, Rashmi Gangadharaiah. (2023)  
**Document-Level Supervision for Multi-Aspect Sentiment Analysis Without Fine-grained Labels**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2310.06940v1)  

---


**ABSTRACT**  
Aspect-based sentiment analysis (ABSA) is a widely studied topic, most often trained through supervision from human annotations of opinionated texts. These fine-grained annotations include identifying aspects towards which a user expresses their sentiment, and their associated polarities (aspect-based sentiments). Such fine-grained annotations can be expensive and often infeasible to obtain in real-world settings. There is, however, an abundance of scenarios where user-generated text contains an overall sentiment, such as a rating of 1-5 in user reviews or user-generated feedback, which may be leveraged for this task. In this paper, we propose a VAE-based topic modeling approach that performs ABSA using document-level supervision and without requiring fine-grained labels for either aspects or sentiments. Our approach allows for the detection of multiple aspects in a document, thereby allowing for the possibility of reasoning about how sentiment expressed through multiple aspects comes together to form an observable overall document-level sentiment. We demonstrate results on two benchmark datasets from two different domains, significantly outperforming a state-of-the-art baseline.

{{</citation>}}


### (8/159) Sparse Finetuning for Inference Acceleration of Large Language Models (Eldar Kurtic et al., 2023)

{{<citation>}}

Eldar Kurtic, Denis Kuznedelev, Elias Frantar, Michael Goin, Dan Alistarh. (2023)  
**Sparse Finetuning for Inference Acceleration of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2310.06927v1)  

---


**ABSTRACT**  
We consider the problem of accurate sparse finetuning of large language models (LLMs), that is, finetuning pretrained LLMs on specialized tasks, while inducing sparsity in their weights. On the accuracy side, we observe that standard loss-based finetuning may fail to recover accuracy, especially at high sparsities. To address this, we perform a detailed study of distillation-type losses, determining an L2-based distillation approach we term SquareHead which enables accurate recovery even at higher sparsities, across all model types. On the practical efficiency side, we show that sparse LLMs can be executed with speedups by taking advantage of sparsity, for both CPU and GPU runtimes. While the standard approach is to leverage sparsity for computational reduction, we observe that in the case of memory-bound LLMs sparsity can also be leveraged for reducing memory bandwidth. We exhibit end-to-end results showing speedups due to sparsity, while recovering accuracy, on T5 (language translation), Whisper (speech translation), and open GPT-type (MPT for text generation). For MPT text generation, we show for the first time that sparse finetuning can reach 75% sparsity without accuracy drops, provide notable end-to-end speedups for both CPU and GPU inference, and highlight that sparsity is also compatible with quantization approaches. Models and software for reproducing our results are provided in Section 6.

{{</citation>}}


### (9/159) Improving Contrastive Learning of Sentence Embeddings with Focal-InfoNCE (Pengyue Hou et al., 2023)

{{<citation>}}

Pengyue Hou, Xingyu Li. (2023)  
**Improving Contrastive Learning of Sentence Embeddings with Focal-InfoNCE**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Contrastive Learning, Embedding, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2310.06918v1)  

---


**ABSTRACT**  
The recent success of SimCSE has greatly advanced state-of-the-art sentence representations. However, the original formulation of SimCSE does not fully exploit the potential of hard negative samples in contrastive learning. This study introduces an unsupervised contrastive learning framework that combines SimCSE with hard negative mining, aiming to enhance the quality of sentence embeddings. The proposed focal-InfoNCE function introduces self-paced modulation terms in the contrastive objective, downweighting the loss associated with easy negatives and encouraging the model focusing on hard negatives. Experimentation on various STS benchmarks shows that our method improves sentence embeddings in terms of Spearman's correlation and representation alignment and uniformity.

{{</citation>}}


### (10/159) LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression (Huiqiang Jiang et al., 2023)

{{<citation>}}

Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu. (2023)  
**LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-3.5, QA  
[Paper Link](http://arxiv.org/abs/2310.06839v1)  

---


**ABSTRACT**  
In long context scenarios, large language models (LLMs) face three main challenges: higher computational/financial cost, longer latency, and inferior performance. Some studies reveal that the performance of LLMs depends on both the density and the position of the key information (question relevant) in the input prompt. Inspired by these findings, we propose LongLLMLingua for prompt compression towards improving LLMs' perception of the key information to simultaneously address the three challenges. We conduct evaluation on a wide range of long context scenarios including single-/multi-document QA, few-shot learning, summarization, synthetic tasks, and code completion. The experimental results show that LongLLMLingua compressed prompt can derive higher performance with much less cost. The latency of the end-to-end system is also reduced. For example, on NaturalQuestions benchmark, LongLLMLingua gains a performance boost of up to 17.1% over the original prompt with ~4x fewer tokens as input to GPT-3.5-Turbo. It can derive cost savings of \$28.5 and \$27.4 per 1,000 samples from the LongBench and ZeroScrolls benchmark, respectively. Additionally, when compressing prompts of ~10k tokens at a compression rate of 2x-10x, LongLLMLingua can speed up the end-to-end latency by 1.4x-3.8x. Our code is available at https://aka.ms/LLMLingua.

{{</citation>}}


### (11/159) Generating and Evaluating Tests for K-12 Students with Language Model Simulations: A Case Study on Sentence Reading Efficiency (Eric Zelikman et al., 2023)

{{<citation>}}

Eric Zelikman, Wanjing Anya Ma, Jasmine E. Tran, Diyi Yang, Jason D. Yeatman, Nick Haber. (2023)  
**Generating and Evaluating Tests for K-12 Students with Language Model Simulations: A Case Study on Sentence Reading Efficiency**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06837v1)  

---


**ABSTRACT**  
Developing an educational test can be expensive and time-consuming, as each item must be written by experts and then evaluated by collecting hundreds of student responses. Moreover, many tests require multiple distinct sets of questions administered throughout the school year to closely monitor students' progress, known as parallel tests. In this study, we focus on tests of silent sentence reading efficiency, used to assess students' reading ability over time. To generate high-quality parallel tests, we propose to fine-tune large language models (LLMs) to simulate how previous students would have responded to unseen items. With these simulated responses, we can estimate each item's difficulty and ambiguity. We first use GPT-4 to generate new test items following a list of expert-developed rules and then apply a fine-tuned LLM to filter the items based on criteria from psychological measurements. We also propose an optimal-transport-inspired technique for generating parallel tests and show the generated tests closely correspond to the original test's difficulty and reliability based on crowdworker responses. Our evaluation of a generated test with 234 students from grades 2 to 8 produces test scores highly correlated (r=0.93) to those of a standard test form written by human experts and evaluated across thousands of K-12 students.

{{</citation>}}


### (12/159) Teaching Language Models to Hallucinate Less with Synthetic Tasks (Erik Jones et al., 2023)

{{<citation>}}

Erik Jones, Hamid Palangi, Clarisse Simões, Varun Chandrasekaran, Subhabrata Mukherjee, Arindam Mitra, Ahmed Awadallah, Ece Kamar. (2023)  
**Teaching Language Models to Hallucinate Less with Synthetic Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06827v1)  

---


**ABSTRACT**  
Large language models (LLMs) frequently hallucinate on abstractive summarization tasks such as document-based question-answering, meeting summarization, and clinical report generation, even though all necessary information is included in context. However, optimizing LLMs to hallucinate less on these tasks is challenging, as hallucination is hard to efficiently evaluate at each optimization step. In this work, we show that reducing hallucination on a synthetic task can also reduce hallucination on real-world downstream tasks. Our method, SynTra, first designs a synthetic task where hallucinations are easy to elicit and measure. It next optimizes the LLM's system message via prefix-tuning on the synthetic task, and finally transfers the system message to realistic, hard-to-optimize tasks. Across three realistic abstractive summarization tasks, SynTra reduces hallucination for two 13B-parameter LLMs using only a synthetic retrieval task for supervision. We also find that optimizing the system message rather than the model weights can be critical; fine-tuning the entire model on the synthetic task can counterintuitively increase hallucination. Overall, SynTra demonstrates that the extra flexibility of working with synthetic data can help mitigate undesired behaviors in practice.

{{</citation>}}


### (13/159) Mistral 7B (Albert Q. Jiang et al., 2023)

{{<citation>}}

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed. (2023)  
**Mistral 7B**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.06825v1)  

---


**ABSTRACT**  
We introduce Mistral 7B v0.1, a 7-billion-parameter language model engineered for superior performance and efficiency. Mistral 7B outperforms Llama 2 13B across all evaluated benchmarks, and Llama 1 34B in reasoning, mathematics, and code generation. Our model leverages grouped-query attention (GQA) for faster inference, coupled with sliding window attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference cost. We also provide a model fine-tuned to follow instructions, Mistral 7B -- Instruct, that surpasses the Llama 2 13B -- Chat model both on human and automated benchmarks. Our models are released under the Apache 2.0 license.

{{</citation>}}


### (14/159) Text Embeddings Reveal (Almost) As Much As Text (John X. Morris et al., 2023)

{{<citation>}}

John X. Morris, Volodymyr Kuleshov, Vitaly Shmatikov, Alexander M. Rush. (2023)  
**Text Embeddings Reveal (Almost) As Much As Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.06816v1)  

---


**ABSTRACT**  
How much private information do text embeddings reveal about the original text? We investigate the problem of embedding \textit{inversion}, reconstructing the full text represented in dense text embeddings. We frame the problem as controlled generation: generating text that, when reembedded, is close to a fixed point in latent space. We find that although a na\"ive model conditioned on the embedding performs poorly, a multi-step method that iteratively corrects and re-embeds text is able to recover $92\%$ of $32\text{-token}$ text inputs exactly. We train our model to decode text embeddings from two state-of-the-art embedding models, and also show that our model can recover important personal information (full names) from a dataset of clinical notes. Our code is available on Github: \href{https://github.com/jxmorris12/vec2text}{github.com/jxmorris12/vec2text}.

{{</citation>}}


### (15/159) Advancing Transformer's Capabilities in Commonsense Reasoning (Yu Zhou et al., 2023)

{{<citation>}}

Yu Zhou, Yunqiu Han, Hanyu Zhou, Yulun Wu. (2023)  
**Advancing Transformer's Capabilities in Commonsense Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Reasoning, Transformer  
[Paper Link](http://arxiv.org/abs/2310.06803v1)  

---


**ABSTRACT**  
Recent advances in general purpose pre-trained language models have shown great potential in commonsense reasoning. However, current works still perform poorly on standard commonsense reasoning benchmarks including the Com2Sense Dataset. We argue that this is due to a disconnect with current cutting-edge machine learning methods. In this work, we aim to bridge the gap by introducing current ML-based methods to improve general purpose pre-trained language models in the task of commonsense reasoning. Specifically, we experiment with and systematically evaluate methods including knowledge transfer, model ensemble, and introducing an additional pairwise contrastive objective. Our best model outperforms the strongest previous works by ~15\% absolute gains in Pairwise Accuracy and ~8.7\% absolute gains in Standard Accuracy.

{{</citation>}}


### (16/159) SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (Carlos E. Jimenez et al., 2023)

{{<citation>}}

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan. (2023)  
**SWE-bench: Can Language Models Resolve Real-World GitHub Issues?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-SE, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06770v1)  

---


**ABSTRACT**  
Language models have outpaced our ability to evaluate them effectively, but for their future development it is essential to study the frontier of their capabilities. We consider real-world software engineering to be a rich, sustainable, and challenging testbed for evaluating the next generation of language models. We therefore introduce SWE-bench, an evaluation framework including $2,294$ software engineering problems drawn from real GitHub issues and corresponding pull requests across $12$ popular Python repositories. Given a codebase along with a description of an issue to be resolved, a language model is tasked with editing the codebase to address the issue. Resolving issues in SWE-bench frequently requires understanding and coordinating changes across multiple functions, classes, and even files simultaneously, calling for models to interact with execution environments, process extremely long contexts and perform complex reasoning that goes far beyond traditional code generation. Our evaluations show that both state-of-the-art proprietary models and our fine-tuned model SWE-Llama can resolve only the simplest issues. Claude 2 and GPT-4 solve a mere $4.8$% and $1.7$% of instances respectively, even when provided with an oracle retriever. Advances on SWE-bench represent steps towards LMs that are more practical, intelligent, and autonomous.

{{</citation>}}


### (17/159) TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models (Xiao Wang et al., 2023)

{{<citation>}}

Xiao Wang, Yuansen Zhang, Tianze Chen, Songyang Gao, Senjie Jin, Xianjun Yang, Zhiheng Xi, Rui Zheng, Yicheng Zou, Tao Gui, Qi Zhang, Xuanjing Huang. (2023)  
**TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.06762v1)  

---


**ABSTRACT**  
Aligned large language models (LLMs) demonstrate exceptional capabilities in task-solving, following instructions, and ensuring safety. However, the continual learning aspect of these aligned LLMs has been largely overlooked. Existing continual learning benchmarks lack sufficient challenge for leading aligned LLMs, owing to both their simplicity and the models' potential exposure during instruction tuning. In this paper, we introduce TRACE, a novel benchmark designed to evaluate continual learning in LLMs. TRACE consists of 8 distinct datasets spanning challenging tasks including domain-specific tasks, multilingual capabilities, code generation, and mathematical reasoning. All datasets are standardized into a unified format, allowing for effortless automatic evaluation of LLMs. Our experiments show that after training on TRACE, aligned LLMs exhibit significant declines in both general ability and instruction-following capabilities. For example, the accuracy of llama2-chat 13B on gsm8k dataset declined precipitously from 28.8\% to 2\% after training on our datasets. This highlights the challenge of finding a suitable tradeoff between achieving performance on specific tasks while preserving the original prowess of LLMs. Empirical findings suggest that tasks inherently equipped with reasoning paths contribute significantly to preserving certain capabilities of LLMs against potential declines. Motivated by this, we introduce the Reasoning-augmented Continual Learning (RCL) approach. RCL integrates task-specific cues with meta-rationales, effectively reducing catastrophic forgetting in LLMs while expediting convergence on novel tasks.

{{</citation>}}


### (18/159) Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning (Mengzhou Xia et al., 2023)

{{<citation>}}

Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, Danqi Chen. (2023)  
**Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: LLaMA, Language Model, Pruning  
[Paper Link](http://arxiv.org/abs/2310.06694v1)  

---


**ABSTRACT**  
The popularity of LLaMA (Touvron et al., 2023a;b) and other recently emerged moderate-sized large language models (LLMs) highlights the potential of building smaller yet powerful LLMs. Regardless, the cost of training such models from scratch on trillions of tokens remains high. In this work, we study structured pruning as an effective means to develop smaller LLMs from pre-trained, larger models. Our approach employs two key techniques: (1) targeted structured pruning, which prunes a larger model to a specified target shape by removing layers, heads, and intermediate and hidden dimensions in an end-to-end manner, and (2) dynamic batch loading, which dynamically updates the composition of sampled data in each training batch based on varying losses across different domains. We demonstrate the efficacy of our approach by presenting the Sheared-LLaMA series, pruning the LLaMA2-7B model down to 1.3B and 2.7B parameters. Sheared-LLaMA models outperform state-of-the-art open-source models of equivalent sizes, such as Pythia, INCITE, and OpenLLaMA models, on a wide range of downstream and instruction tuning evaluations, while requiring only 3% of compute compared to training such models from scratch. This work provides compelling evidence that leveraging existing LLMs with structured pruning is a far more cost-effective approach for building smaller LLMs.

{{</citation>}}


### (19/159) Meta-CoT: Generalizable Chain-of-Thought Prompting in Mixed-task Scenarios with Large Language Models (Anni Zou et al., 2023)

{{<citation>}}

Anni Zou, Zhuosheng Zhang, Hai Zhao, Xiangru Tang. (2023)  
**Meta-CoT: Generalizable Chain-of-Thought Prompting in Mixed-task Scenarios with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06692v2)  

---


**ABSTRACT**  
Large language models (LLMs) have unveiled remarkable reasoning capabilities by exploiting chain-of-thought (CoT) prompting, which generates intermediate reasoning chains to serve as the rationale for deriving the answer. However, current CoT methods either simply employ general prompts such as Let's think step by step, or heavily rely on handcrafted task-specific demonstrations to attain preferable performances, thereby engendering an inescapable gap between performance and generalization. To bridge this gap, we propose Meta-CoT, a generalizable CoT prompting method in mixed-task scenarios where the type of input questions is unknown. Meta-CoT firstly categorizes the scenario based on the input question and subsequently constructs diverse demonstrations from the corresponding data pool in an automatic pattern. Meta-CoT simultaneously enjoys remarkable performances on ten public benchmark reasoning tasks and superior generalization capabilities. Notably, Meta-CoT achieves the state-of-the-art result on SVAMP (93.7%) without any additional program-aided methods. Our further experiments on five out-of-distribution datasets verify the stability and generality of Meta-CoT.

{{</citation>}}


### (20/159) Learning Multiplex Embeddings on Text-rich Networks with One Text Encoder (Bowen Jin et al., 2023)

{{<citation>}}

Bowen Jin, Wentao Zhang, Yu Zhang, Yu Meng, Han Zhao, Jiawei Han. (2023)  
**Learning Multiplex Embeddings on Text-rich Networks with One Text Encoder**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Embedding, GNN  
[Paper Link](http://arxiv.org/abs/2310.06684v1)  

---


**ABSTRACT**  
In real-world scenarios, texts in a network are often linked by multiple semantic relations (e.g., papers in an academic network are referenced by other publications, written by the same author, or published in the same venue), where text documents and their relations form a multiplex text-rich network. Mainstream text representation learning methods use pretrained language models (PLMs) to generate one embedding for each text unit, expecting that all types of relations between texts can be captured by these single-view embeddings. However, this presumption does not hold particularly in multiplex text-rich networks. Along another line of work, multiplex graph neural networks (GNNs) directly initialize node attributes as a feature vector for node representation learning, but they cannot fully capture the semantics of the nodes' associated texts. To bridge these gaps, we propose METERN, a new framework for learning Multiplex Embeddings on TExt-Rich Networks. In contrast to existing methods, METERN uses one text encoder to model the shared knowledge across relations and leverages a small number of parameters per relation to derive relation-specific representations. This allows the encoder to effectively capture the multiplex structures in the network while also preserving parameter efficiency. We conduct experiments on nine downstream tasks in five networks from both academic and e-commerce domains, where METERN outperforms baselines significantly and consistently. The code is available at https://github.com/PeterGriffinJin/METERN-submit.

{{</citation>}}


### (21/159) SEER: A Knapsack approach to Exemplar Selection for In-Context HybridQA (Jonathan Tonglet et al., 2023)

{{<citation>}}

Jonathan Tonglet, Manon Reusens, Philipp Borchert, Bart Baesens. (2023)  
**SEER: A Knapsack approach to Exemplar Selection for In-Context HybridQA**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.06675v1)  

---


**ABSTRACT**  
Question answering over hybrid contexts is a complex task, which requires the combination of information extracted from unstructured texts and structured tables in various ways. Recently, In-Context Learning demonstrated significant performance advances for reasoning tasks. In this paradigm, a large language model performs predictions based on a small set of supporting exemplars. The performance of In-Context Learning depends heavily on the selection procedure of the supporting exemplars, particularly in the case of HybridQA, where considering the diversity of reasoning chains and the large size of the hybrid contexts becomes crucial. In this work, we present Selection of ExEmplars for hybrid Reasoning (SEER), a novel method for selecting a set of exemplars that is both representative and diverse. The key novelty of SEER is that it formulates exemplar selection as a Knapsack Integer Linear Program. The Knapsack framework provides the flexibility to incorporate diversity constraints that prioritize exemplars with desirable attributes, and capacity constraints that ensure that the prompt size respects the provided capacity budgets. The effectiveness of SEER is demonstrated on FinQA and TAT-QA, two real-world benchmarks for HybridQA, where it outperforms previous exemplar selection methods.

{{</citation>}}


### (22/159) Making Large Language Models Perform Better in Knowledge Graph Completion (Yichi Zhang et al., 2023)

{{<citation>}}

Yichi Zhang, Zhuo Chen, Wen Zhang, Huajun Chen. (2023)  
**Making Large Language Models Perform Better in Knowledge Graph Completion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06671v1)  

---


**ABSTRACT**  
Large language model (LLM) based knowledge graph completion (KGC) aims to predict the missing triples in the KGs with LLMs and enrich the KGs to become better web infrastructure, which can benefit a lot of web-based automatic services. However, research about LLM-based KGC is limited and lacks effective utilization of LLM's inference capabilities, which ignores the important structural information in KGs and prevents LLMs from acquiring accurate factual knowledge. In this paper, we discuss how to incorporate the helpful KG structural information into the LLMs, aiming to achieve structrual-aware reasoning in the LLMs. We first transfer the existing LLM paradigms to structural-aware settings and further propose a knowledge prefix adapter (KoPA) to fulfill this stated goal. KoPA employs structural embedding pre-training to capture the structural information of entities and relations in the KG. Then KoPA informs the LLMs of the knowledge prefix adapter which projects the structural embeddings into the textual space and obtains virtual knowledge tokens as a prefix of the input prompt. We conduct comprehensive experiments on these structural-aware LLM-based KGC methods and provide an in-depth analysis comparing how the introduction of structural information would be better for LLM's knowledge reasoning ability. Our code is released at https://github.com/zjukg/KoPA.

{{</citation>}}


### (23/159) Unlock the Potential of Counterfactually-Augmented Data in Out-Of-Distribution Generalization (Caoyun Fan et al., 2023)

{{<citation>}}

Caoyun Fan, Wenqing Chen, Jidong Tian, Yitian Li, Hao He, Yaohui Jin. (2023)  
**Unlock the Potential of Counterfactually-Augmented Data in Out-Of-Distribution Generalization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Natural Language Inference, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2310.06666v1)  

---


**ABSTRACT**  
Counterfactually-Augmented Data (CAD) -- minimal editing of sentences to flip the corresponding labels -- has the potential to improve the Out-Of-Distribution (OOD) generalization capability of language models, as CAD induces language models to exploit domain-independent causal features and exclude spurious correlations. However, the empirical results of CAD's OOD generalization are not as efficient as anticipated. In this study, we attribute the inefficiency to the myopia phenomenon caused by CAD: language models only focus on causal features that are edited in the augmentation operation and exclude other non-edited causal features. Therefore, the potential of CAD is not fully exploited. To address this issue, we analyze the myopia phenomenon in feature space from the perspective of Fisher's Linear Discriminant, then we introduce two additional constraints based on CAD's structural properties (dataset-level and sentence-level) to help language models extract more complete causal features in CAD, thereby mitigating the myopia phenomenon and improving OOD generalization capability. We evaluate our method on two tasks: Sentiment Analysis and Natural Language Inference, and the experimental results demonstrate that our method could unlock the potential of CAD and improve the OOD generalization performance of language models by 1.0% to 5.9%.

{{</citation>}}


### (24/159) What If the TV Was Off? Examining Counterfactual Reasoning Abilities of Multi-modal Language Models (Letian Zhang et al., 2023)

{{<citation>}}

Letian Zhang, Xiaotong Zhai, Zhongkai Zhao, Xin Wen, Yongshuo Zong, Bingchen Zhao. (2023)  
**What If the TV Was Off? Examining Counterfactual Reasoning Abilities of Multi-modal Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.06627v1)  

---


**ABSTRACT**  
Counterfactual reasoning ability is one of the core abilities of human intelligence. This reasoning process involves the processing of alternatives to observed states or past events, and this process can improve our ability for planning and decision-making. In this work, we focus on benchmarking the counterfactual reasoning ability of multi-modal large language models. We take the question and answer pairs from the VQAv2 dataset and add one counterfactual presupposition to the questions, with the answer being modified accordingly. After generating counterfactual questions and answers using ChatGPT, we manually examine all generated questions and answers to ensure correctness. Over 2k counterfactual question and answer pairs are collected this way. We evaluate recent vision language models on our newly collected test dataset and found that all models exhibit a large performance drop compared to the results tested on questions without the counterfactual presupposition. This result indicates that there still exists space for developing vision language models. Apart from the vision language models, our proposed dataset can also serves as a benchmark for evaluating the ability of code generation LLMs, results demonstrate a large gap between GPT-4 and current open-source models. Our code and dataset are available at \url{https://github.com/Letian2003/C-VQA}.

{{</citation>}}


### (25/159) No Pitch Left Behind: Addressing Gender Unbalance in Automatic Speech Recognition through Pitch Manipulation (Dennis Fucci et al., 2023)

{{<citation>}}

Dennis Fucci, Marco Gaido, Matteo Negri, Mauro Cettolo, Luisa Bentivogli. (2023)  
**No Pitch Left Behind: Addressing Gender Unbalance in Automatic Speech Recognition through Pitch Manipulation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.06590v1)  

---


**ABSTRACT**  
Automatic speech recognition (ASR) systems are known to be sensitive to the sociolinguistic variability of speech data, in which gender plays a crucial role. This can result in disparities in recognition accuracy between male and female speakers, primarily due to the under-representation of the latter group in the training data. While in the context of hybrid ASR models several solutions have been proposed, the gender bias issue has not been explicitly addressed in end-to-end neural architectures. To fill this gap, we propose a data augmentation technique that manipulates the fundamental frequency (f0) and formants. This technique reduces the data unbalance among genders by simulating voices of the under-represented female speakers and increases the variability within each gender group. Experiments on spontaneous English speech show that our technique yields a relative WER improvement up to 9.87% for utterances by female speakers, with larger gains for the least-represented f0 ranges.

{{</citation>}}


### (26/159) FTFT: efficient and robust Fine-Tuning by transFerring Training dynamics (Yupei Du et al., 2023)

{{<citation>}}

Yupei Du, Albert Gatt, Dong Nguyen. (2023)  
**FTFT: efficient and robust Fine-Tuning by transFerring Training dynamics**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.06588v1)  

---


**ABSTRACT**  
Despite the massive success of fine-tuning large Pre-trained Language Models (PLMs) on a wide range of Natural Language Processing (NLP) tasks, they remain susceptible to out-of-distribution (OOD) and adversarial inputs. Data map (DM) is a simple yet effective dual-model approach that enhances the robustness of fine-tuned PLMs, which involves fine-tuning a model on the original training set (i.e. reference model), selecting a specified fraction of important training examples according to the training dynamics of the reference model, and fine-tuning the same model on these selected examples (i.e. main model). However, it suffers from the drawback of requiring fine-tuning the same model twice, which is computationally expensive for large models. In this paper, we first show that 1) training dynamics are highly transferable across different model sizes and different pre-training methods, and that 2) main models fine-tuned using DM learn faster than when using conventional Empirical Risk Minimization (ERM). Building on these observations, we propose a novel fine-tuning approach based on the DM method: Fine-Tuning by transFerring Training dynamics (FTFT). Compared with DM, FTFT uses more efficient reference models and then fine-tunes more capable main models for fewer steps. Our experiments show that FTFT achieves better generalization robustness than ERM while spending less than half of the training cost.

{{</citation>}}


### (27/159) Rationale-Enhanced Language Models are Better Continual Relation Learners (Weimin Xiong et al., 2023)

{{<citation>}}

Weimin Xiong, Yifan Song, Peiyi Wang, Sujian Li. (2023)  
**Rationale-Enhanced Language Models are Better Continual Relation Learners**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06547v1)  

---


**ABSTRACT**  
Continual relation extraction (CRE) aims to solve the problem of catastrophic forgetting when learning a sequence of newly emerging relations. Recent CRE studies have found that catastrophic forgetting arises from the model's lack of robustness against future analogous relations. To address the issue, we introduce rationale, i.e., the explanations of relation classification results generated by large language models (LLM), into CRE task. Specifically, we design the multi-task rationale tuning strategy to help the model learn current relations robustly. We also conduct contrastive rationale replay to further distinguish analogous relations. Experimental results on two standard benchmarks demonstrate that our method outperforms the state-of-the-art CRE models.

{{</citation>}}


### (28/159) A Novel Contrastive Learning Method for Clickbait Detection on RoCliCo: A Romanian Clickbait Corpus of News Articles (Daria-Mihaela Broscoteanu et al., 2023)

{{<citation>}}

Daria-Mihaela Broscoteanu, Radu Tudor Ionescu. (2023)  
**A Novel Contrastive Learning Method for Clickbait Detection on RoCliCo: A Romanian Clickbait Corpus of News Articles**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.06540v1)  

---


**ABSTRACT**  
To increase revenue, news websites often resort to using deceptive news titles, luring users into clicking on the title and reading the full news. Clickbait detection is the task that aims to automatically detect this form of false advertisement and avoid wasting the precious time of online users. Despite the importance of the task, to the best of our knowledge, there is no publicly available clickbait corpus for the Romanian language. To this end, we introduce a novel Romanian Clickbait Corpus (RoCliCo) comprising 8,313 news samples which are manually annotated with clickbait and non-clickbait labels. Furthermore, we conduct experiments with four machine learning methods, ranging from handcrafted models to recurrent and transformer-based neural networks, to establish a line-up of competitive baselines. We also carry out experiments with a weighted voting ensemble. Among the considered baselines, we propose a novel BERT-based contrastive learning model that learns to encode news titles and contents into a deep metric space such that titles and contents of non-clickbait news have high cosine similarity, while titles and contents of clickbait news have low cosine similarity. Our data set and code to reproduce the baselines are publicly available for download at https://github.com/dariabroscoteanu/RoCliCo.

{{</citation>}}


### (29/159) EmoTwiCS: A Corpus for Modelling Emotion Trajectories in Dutch Customer Service Dialogues on Twitter (Sofie Labat et al., 2023)

{{<citation>}}

Sofie Labat, Thomas Demeester, Véronique Hoste. (2023)  
**EmoTwiCS: A Corpus for Modelling Emotion Trajectories in Dutch Customer Service Dialogues on Twitter**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Twitter  
[Paper Link](http://arxiv.org/abs/2310.06536v1)  

---


**ABSTRACT**  
Due to the rise of user-generated content, social media is increasingly adopted as a channel to deliver customer service. Given the public character of these online platforms, the automatic detection of emotions forms an important application in monitoring customer satisfaction and preventing negative word-of-mouth. This paper introduces EmoTwiCS, a corpus of 9,489 Dutch customer service dialogues on Twitter that are annotated for emotion trajectories. In our business-oriented corpus, we view emotions as dynamic attributes of the customer that can change at each utterance of the conversation. The term `emotion trajectory' refers therefore not only to the fine-grained emotions experienced by customers (annotated with 28 labels and valence-arousal-dominance scores), but also to the event happening prior to the conversation and the responses made by the human operator (both annotated with 8 categories). Inter-annotator agreement (IAA) scores on the resulting dataset are substantial and comparable with related research, underscoring its high quality. Given the interplay between the different layers of annotated information, we perform several in-depth analyses to investigate (i) static emotions in isolated tweets, (ii) dynamic emotions and their shifts in trajectory, and (iii) the role of causes and response strategies in emotion trajectories. We conclude by listing the advantages and limitations of our dataset, after which we give some suggestions on the different types of predictive modelling tasks and open research questions to which EmoTwiCS can be applied. The dataset is available upon request and will be made publicly available upon acceptance of the paper.

{{</citation>}}


### (30/159) Evaluation of ChatGPT Feedback on ELL Writers' Coherence and Cohesion (Su-Youn Yoon et al., 2023)

{{<citation>}}

Su-Youn Yoon, Eva Miszoglad, Lisa R. Pierce. (2023)  
**Evaluation of ChatGPT Feedback on ELL Writers' Coherence and Cohesion**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.06505v1)  

---


**ABSTRACT**  
Since its launch in November 2022, ChatGPT has had a transformative effect on education where students are using it to help with homework assignments and teachers are actively employing it in their teaching practices. This includes using ChatGPT as a tool for writing teachers to grade and generate feedback on students' essays. In this study, we evaluated the quality of the feedback generated by ChatGPT regarding the coherence and cohesion of the essays written by English Language Learners (ELLs) students. We selected 50 argumentative essays and generated feedback on coherence and cohesion using the ELLIPSE rubric. During the feedback evaluation, we used a two-step approach: first, each sentence in the feedback was classified into subtypes based on its function (e.g., positive reinforcement, problem statement). Next, we evaluated its accuracy and usability according to these types. Both the analysis of feedback types and the evaluation of accuracy and usability revealed that most feedback sentences were highly abstract and generic, failing to provide concrete suggestions for improvement. The accuracy in detecting major problems, such as repetitive ideas and the inaccurate use of cohesive devices, depended on superficial linguistic features and was often incorrect. In conclusion, ChatGPT, without specific training for the feedback generation task, does not offer effective feedback on ELL students' coherence and cohesion.

{{</citation>}}


### (31/159) Revisit Input Perturbation Problems for LLMs: A Unified Robustness Evaluation Framework for Noisy Slot Filling Task (Guanting Dong et al., 2023)

{{<citation>}}

Guanting Dong, Jinxu Zhao, Tingfeng Hui, Daichi Guo, Wenlong Wan, Boqi Feng, Yueyan Qiu, Zhuoma Gongque, Keqing He, Zechen Wang, Weiran Xu. (2023)  
**Revisit Input Perturbation Problems for LLMs: A Unified Robustness Evaluation Framework for Noisy Slot Filling Task**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.06504v1)  

---


**ABSTRACT**  
With the increasing capabilities of large language models (LLMs), these high-performance models have achieved state-of-the-art results on a wide range of natural language processing (NLP) tasks. However, the models' performance on commonly-used benchmark datasets often fails to accurately reflect their reliability and robustness when applied to real-world noisy data. To address these challenges, we propose a unified robustness evaluation framework based on the slot-filling task to systematically evaluate the dialogue understanding capability of LLMs in diverse input perturbation scenarios. Specifically, we construct a input perturbation evaluation dataset, Noise-LLM, which contains five types of single perturbation and four types of mixed perturbation data. Furthermore, we utilize a multi-level data augmentation method (character, word, and sentence levels) to construct a candidate data pool, and carefully design two ways of automatic task demonstration construction strategies (instance-level and entity-level) with various prompt templates. Our aim is to assess how well various robustness methods of LLMs perform in real-world noisy scenarios. The experiments have demonstrated that the current open-source LLMs generally achieve limited perturbation robustness performance. Based on these experimental observations, we make some forward-looking suggestions to fuel the research in this direction.

{{</citation>}}


### (32/159) The Limits of ChatGPT in Extracting Aspect-Category-Opinion-Sentiment Quadruples: A Comparative Analysis (Xiancai Xu et al., 2023)

{{<citation>}}

Xiancai Xu, Jia-Dong Zhang, Rongchang Xiao, Lei Xiong. (2023)  
**The Limits of ChatGPT in Extracting Aspect-Category-Opinion-Sentiment Quadruples: A Comparative Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.06502v1)  

---


**ABSTRACT**  
Recently, ChatGPT has attracted great attention from both industry and academia due to its surprising abilities in natural language understanding and generation. We are particularly curious about whether it can achieve promising performance on one of the most complex tasks in aspect-based sentiment analysis, i.e., extracting aspect-category-opinion-sentiment quadruples from texts. To this end, in this paper we develop a specialized prompt template that enables ChatGPT to effectively tackle this complex quadruple extraction task. Further, we propose a selection method on few-shot examples to fully exploit the in-context learning ability of ChatGPT and uplift its effectiveness on this complex task. Finally, we provide a comparative evaluation on ChatGPT against existing state-of-the-art quadruple extraction models based on four public datasets and highlight some important findings regarding the capability boundaries of ChatGPT in the quadruple extraction.

{{</citation>}}


### (33/159) A New Benchmark and Reverse Validation Method for Passage-level Hallucination Detection (Shiping Yang et al., 2023)

{{<citation>}}

Shiping Yang, Renliang Sun, Xiaojun Wan. (2023)  
**A New Benchmark and Reverse Validation Method for Passage-level Hallucination Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06498v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated their ability to collaborate effectively with humans in real-world scenarios. However, LLMs are apt to generate hallucinations, i.e., makeup incorrect text and unverified information, which can cause significant damage when deployed for mission-critical tasks. In this paper, we propose a self-check approach based on reverse validation to detect factual errors automatically in a zero-resource fashion. To facilitate future studies and assess different methods, we construct a hallucination detection benchmark, which is generated by ChatGPT and annotated by human annotators. Contrasting previous studies of zero-resource hallucination detection, our method and benchmark concentrate on passage-level detection instead of sentence-level. We empirically evaluate our method and existing zero-resource detection methods on different domains of benchmark to explore the implicit relation between hallucination and training data. Furthermore, we manually analyze some hallucination cases that LLM failed to capture, revealing the shared limitation of zero-resource methods.

{{</citation>}}


### (34/159) Multilingual Jailbreak Challenges in Large Language Models (Yue Deng et al., 2023)

{{<citation>}}

Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, Lidong Bing. (2023)  
**Multilingual Jailbreak Challenges in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model, Multilingual, NLP  
[Paper Link](http://arxiv.org/abs/2310.06474v1)  

---


**ABSTRACT**  
While large language models (LLMs) exhibit remarkable capabilities across a wide range of tasks, they pose potential safety concerns, such as the ``jailbreak'' problem, wherein malicious instructions can manipulate LLMs to exhibit undesirable behavior. Although several preventive measures have been developed to mitigate the potential risks associated with LLMs, they have primarily focused on English data. In this study, we reveal the presence of multilingual jailbreak challenges within LLMs and consider two potential risk scenarios: unintentional and intentional. The unintentional scenario involves users querying LLMs using non-English prompts and inadvertently bypassing the safety mechanisms, while the intentional scenario concerns malicious users combining malicious instructions with multilingual prompts to deliberately attack LLMs. The experimental results reveal that in the unintentional scenario, the rate of unsafe content increases as the availability of languages decreases. Specifically, low-resource languages exhibit three times the likelihood of encountering harmful content compared to high-resource languages, with both ChatGPT and GPT-4. In the intentional scenario, multilingual prompts can exacerbate the negative impact of malicious instructions, with astonishingly high rates of unsafe output: 80.92\% for ChatGPT and 40.71\% for GPT-4. To handle such a challenge in the multilingual context, we propose a novel \textsc{Self-Defense} framework that automatically generates multilingual training data for safety fine-tuning. Experimental results show that ChatGPT fine-tuned with such data can achieve a substantial reduction in unsafe content generation. Data is available at https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs. Warning: This paper contains examples with potentially harmful content.

{{</citation>}}


### (35/159) Constructive Large Language Models Alignment with Diverse Feedback (Tianshu Yu et al., 2023)

{{<citation>}}

Tianshu Yu, Ting-En Lin, Yuchuan Wu, Min Yang, Fei Huang, Yongbin Li. (2023)  
**Constructive Large Language Models Alignment with Diverse Feedback**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06450v2)  

---


**ABSTRACT**  
In recent research on large language models (LLMs), there has been a growing emphasis on aligning these models with human values to reduce the impact of harmful content. However, current alignment methods often rely solely on singular forms of human feedback, such as preferences, annotated labels, or natural language critiques, overlooking the potential advantages of combining these feedback types. This limitation leads to suboptimal performance, even when ample training data is available. In this paper, we introduce Constructive and Diverse Feedback (CDF) as a novel method to enhance LLM alignment, inspired by constructivist learning theory. Our approach involves collecting three distinct types of feedback tailored to problems of varying difficulty levels within the training dataset. Specifically, we exploit critique feedback for easy problems, refinement feedback for medium problems, and preference feedback for hard problems. By training our model with this diversified feedback, we achieve enhanced alignment performance while using less training data. To assess the effectiveness of CDF, we evaluate it against previous methods in three downstream tasks: question answering, dialog generation, and text summarization. Experimental results demonstrate that CDF achieves superior performance even with a smaller training dataset.

{{</citation>}}


### (36/159) MemSum-DQA: Adapting An Efficient Long Document Extractive Summarizer for Document Question Answering (Nianlong Gu et al., 2023)

{{<citation>}}

Nianlong Gu, Yingqiang Gao, Richard H. R. Hahnloser. (2023)  
**MemSum-DQA: Adapting An Efficient Long Document Extractive Summarizer for Document Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.06436v1)  

---


**ABSTRACT**  
We introduce MemSum-DQA, an efficient system for document question answering (DQA) that leverages MemSum, a long document extractive summarizer. By prefixing each text block in the parsed document with the provided question and question type, MemSum-DQA selectively extracts text blocks as answers from documents. On full-document answering tasks, this approach yields a 9% improvement in exact match accuracy over prior state-of-the-art baselines. Notably, MemSum-DQA excels in addressing questions related to child-relationship understanding, underscoring the potential of extractive summarization techniques for DQA tasks.

{{</citation>}}


### (37/159) Whispering LLaMA: A Cross-Modal Generative Error Correction Framework for Speech Recognition (Srijith Radhakrishnan et al., 2023)

{{<citation>}}

Srijith Radhakrishnan, Chao-Han Huck Yang, Sumeer Ahmad Khan, Rohit Kumar, Narsis A. Kiani, David Gomez-Cabrero, Jesper N. Tegner. (2023)  
**Whispering LLaMA: A Cross-Modal Generative Error Correction Framework for Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-MM, cs-SD, cs.CL, eess-AS  
Keywords: LLaMA, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.06434v1)  

---


**ABSTRACT**  
We introduce a new cross-modal fusion technique designed for generative error correction in automatic speech recognition (ASR). Our methodology leverages both acoustic information and external linguistic representations to generate accurate speech transcription contexts. This marks a step towards a fresh paradigm in generative error correction within the realm of n-best hypotheses. Unlike the existing ranking-based rescoring methods, our approach adeptly uses distinct initialization techniques and parameter-efficient algorithms to boost ASR performance derived from pre-trained speech and text models. Through evaluation across diverse ASR datasets, we evaluate the stability and reproducibility of our fusion technique, demonstrating its improved word error rate relative (WERR) performance in comparison to n-best hypotheses by relatively 37.66%. To encourage future research, we have made our code and pre-trained models open source at https://github.com/Srijith-rkr/Whispering-LLaMA.

{{</citation>}}


### (38/159) Large Language Models for Propaganda Detection (Kilian Sprenkamp et al., 2023)

{{<citation>}}

Kilian Sprenkamp, Daniel Gordon Jones, Liudmila Zavolokina. (2023)  
**Large Language Models for Propaganda Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, GPT, GPT-4, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.06422v1)  

---


**ABSTRACT**  
The prevalence of propaganda in our digital society poses a challenge to societal harmony and the dissemination of truth. Detecting propaganda through NLP in text is challenging due to subtle manipulation techniques and contextual dependencies. To address this issue, we investigate the effectiveness of modern Large Language Models (LLMs) such as GPT-3 and GPT-4 for propaganda detection. We conduct experiments using the SemEval-2020 task 11 dataset, which features news articles labeled with 14 propaganda techniques as a multi-label classification problem. Five variations of GPT-3 and GPT-4 are employed, incorporating various prompt engineering and fine-tuning strategies across the different models. We evaluate the models' performance by assessing metrics such as $F1$ score, $Precision$, and $Recall$, comparing the results with the current state-of-the-art approach using RoBERTa. Our findings demonstrate that GPT-4 achieves comparable results to the current state-of-the-art. Further, this study analyzes the potential and challenges of LLMs in complex tasks like propaganda detection.

{{</citation>}}


### (39/159) Humans and language models diverge when predicting repeating text (Aditya R. Vaidya et al., 2023)

{{<citation>}}

Aditya R. Vaidya, Javier Turek, Alexander G. Huth. (2023)  
**Humans and language models diverge when predicting repeating text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.06408v1)  

---


**ABSTRACT**  
Language models that are trained on the next-word prediction task have been shown to accurately model human behavior in word prediction and reading speed. In contrast with these findings, we present a scenario in which the performance of humans and LMs diverges. We collected a dataset of human next-word predictions for five stimuli that are formed by repeating spans of text. Human and GPT-2 LM predictions are strongly aligned in the first presentation of a text span, but their performance quickly diverges when memory (or in-context learning) begins to play a role. We traced the cause of this divergence to specific attention heads in a middle layer. Adding a power-law recency bias to these attention heads yielded a model that performs much more similarly to humans. We hope that this scenario will spur future work in bringing LMs closer to human behavior.

{{</citation>}}


### (40/159) Hexa: Self-Improving for Knowledge-Grounded Dialogue System (Daejin Jo et al., 2023)

{{<citation>}}

Daejin Jo, Daniel Wontae Nam, Gunsoo Han, Kyoung-Woon On, Taehwan Kwon, Seungeun Rho, Sungwoong Kim. (2023)  
**Hexa: Self-Improving for Knowledge-Grounded Dialogue System**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2310.06404v1)  

---


**ABSTRACT**  
A common practice in knowledge-grounded dialogue generation is to explicitly utilize intermediate steps (e.g., web-search, memory retrieval) with modular approaches. However, data for such steps are often inaccessible compared to those of dialogue responses as they are unobservable in an ordinary dialogue. To fill in the absence of these data, we develop a self-improving method to improve the generative performances of intermediate steps without the ground truth data. In particular, we propose a novel bootstrapping scheme with a guided prompt and a modified loss function to enhance the diversity of appropriate self-generated responses. Through experiments on various benchmark datasets, we empirically demonstrate that our method successfully leverages a self-improving mechanism in generating intermediate and final responses and improves the performances on the task of knowledge-grounded dialogue generation.

{{</citation>}}


### (41/159) Rethinking Model Selection and Decoding for Keyphrase Generation with Pre-trained Sequence-to-Sequence Models (Di Wu et al., 2023)

{{<citation>}}

Di Wu, Wasi Uddin Ahmad, Kai-Wei Chang. (2023)  
**Rethinking Model Selection and Decoding for Keyphrase Generation with Pre-trained Sequence-to-Sequence Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Sequence-to-Sequence  
[Paper Link](http://arxiv.org/abs/2310.06374v1)  

---


**ABSTRACT**  
Keyphrase Generation (KPG) is a longstanding task in NLP with widespread applications. The advent of sequence-to-sequence (seq2seq) pre-trained language models (PLMs) has ushered in a transformative era for KPG, yielding promising performance improvements. However, many design decisions remain unexplored and are often made arbitrarily. This paper undertakes a systematic analysis of the influence of model selection and decoding strategies on PLM-based KPG. We begin by elucidating why seq2seq PLMs are apt for KPG, anchored by an attention-driven hypothesis. We then establish that conventional wisdom for selecting seq2seq PLMs lacks depth: (1) merely increasing model size or performing task-specific adaptation is not parameter-efficient; (2) although combining in-domain pre-training with task adaptation benefits KPG, it does partially hinder generalization. Regarding decoding, we demonstrate that while greedy search delivers strong F1 scores, it lags in recall compared with sampling-based methods. From our insights, we propose DeSel, a likelihood-based decode-select algorithm that improves greedy search by an average of 4.7% semantic F1 across five datasets. Our collective findings pave the way for deeper future investigations into PLM-based KPG.

{{</citation>}}


### (42/159) Multi-Modal Knowledge Graph Transformer Framework for Multi-Modal Entity Alignment (Qian Li et al., 2023)

{{<citation>}}

Qian Li, Cheng Ji, Shu Guo, Zhaoji Liang, Lihong Wang, Jianxin Li. (2023)  
**Multi-Modal Knowledge Graph Transformer Framework for Multi-Modal Entity Alignment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Entity Alignment, Knowledge Graph, Transformer  
[Paper Link](http://arxiv.org/abs/2310.06365v1)  

---


**ABSTRACT**  
Multi-Modal Entity Alignment (MMEA) is a critical task that aims to identify equivalent entity pairs across multi-modal knowledge graphs (MMKGs). However, this task faces challenges due to the presence of different types of information, including neighboring entities, multi-modal attributes, and entity types. Directly incorporating the above information (e.g., concatenation or attention) can lead to an unaligned information space. To address these challenges, we propose a novel MMEA transformer, called MoAlign, that hierarchically introduces neighbor features, multi-modal attributes, and entity types to enhance the alignment task. Taking advantage of the transformer's ability to better integrate multiple information, we design a hierarchical modifiable self-attention block in a transformer encoder to preserve the unique semantics of different information. Furthermore, we design two entity-type prefix injection methods to integrate entity-type information using type prefixes, which help to restrict the global information of entities not present in the MMKGs. Our extensive experiments on benchmark datasets demonstrate that our approach outperforms strong competitors and achieves excellent entity alignment performance.

{{</citation>}}


### (43/159) InfoCL: Alleviating Catastrophic Forgetting in Continual Text Classification from An Information Theoretic Perspective (Yifan Song et al., 2023)

{{<citation>}}

Yifan Song, Peiyi Wang, Weimin Xiong, Dawei Zhu, Tianyu Liu, Zhifang Sui, Sujian Li. (2023)  
**InfoCL: Alleviating Catastrophic Forgetting in Continual Text Classification from An Information Theoretic Perspective**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Text Classification  
[Paper Link](http://arxiv.org/abs/2310.06362v1)  

---


**ABSTRACT**  
Continual learning (CL) aims to constantly learn new knowledge over time while avoiding catastrophic forgetting on old tasks. We focus on continual text classification under the class-incremental setting. Recent CL studies have identified the severe performance decrease on analogous classes as a key factor for catastrophic forgetting. In this paper, through an in-depth exploration of the representation learning process in CL, we discover that the compression effect of the information bottleneck leads to confusion on analogous classes. To enable the model learn more sufficient representations, we propose a novel replay-based continual text classification method, InfoCL. Our approach utilizes fast-slow and current-past contrastive learning to perform mutual information maximization and better recover the previously learned representations. In addition, InfoCL incorporates an adversarial memory augmentation strategy to alleviate the overfitting problem of replay. Experimental results demonstrate that InfoCL effectively mitigates forgetting and achieves state-of-the-art performance on three text classification tasks. The code is publicly available at https://github.com/Yifan-Song793/InfoCL.

{{</citation>}}


### (44/159) Let Models Speak Ciphers: Multiagent Debate through Embeddings (Chau Pham et al., 2023)

{{<citation>}}

Chau Pham, Boyi Liu, Yingxiang Yang, Zhengyu Chen, Tianyi Liu, Jianbo Yuan, Bryan A. Plummer, Zhaoran Wang, Hongxia Yang. (2023)  
**Let Models Speak Ciphers: Multiagent Debate through Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06272v1)  

---


**ABSTRACT**  
Discussion and debate among Large Language Models (LLMs) have gained considerable attention due to their potential to enhance the reasoning ability of LLMs. Although natural language is an obvious choice for communication due to LLM's language understanding capability, the token sampling step needed when generating natural language poses a potential risk of information loss, as it uses only one token to represent the model's belief across the entire vocabulary. In this paper, we introduce a communication regime named CIPHER (Communicative Inter-Model Protocol Through Embedding Representation) to address this issue. Specifically, we remove the token sampling step from LLMs and let them communicate their beliefs across the vocabulary through the expectation of the raw transformer output embeddings. Remarkably, by deviating from natural language, CIPHER offers an advantage of encoding a broader spectrum of information without any modification to the model weights. While the state-of-the-art LLM debate methods using natural language outperforms traditional inference by a margin of 1.5-8%, our experiment results show that CIPHER debate further extends this lead by 1-3.5% across five reasoning tasks and multiple open-source LLMs of varying sizes. This showcases the superiority and robustness of embeddings as an alternative "language" for communication among LLMs.

{{</citation>}}


### (45/159) Towards Mitigating Hallucination in Large Language Models via Self-Reflection (Ziwei Ji et al., 2023)

{{<citation>}}

Ziwei Ji, Tiezheng Yu, Yan Xu, Nayeon Lee, Etsuko Ishii, Pascale Fung. (2023)  
**Towards Mitigating Hallucination in Large Language Models via Self-Reflection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.06271v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown promise for generative and knowledge-intensive tasks including question-answering (QA) tasks. However, the practical deployment still faces challenges, notably the issue of "hallucination", where models generate plausible-sounding but unfaithful or nonsensical information. This issue becomes particularly critical in the medical domain due to the uncommon professional concepts and potential social risks involved. This paper analyses the phenomenon of hallucination in medical generative QA systems using widely adopted LLMs and datasets. Our investigation centers on the identification and comprehension of common problematic answers, with a specific emphasis on hallucination. To tackle this challenge, we present an interactive self-reflection methodology that incorporates knowledge acquisition and answer generation. Through this feedback process, our approach steadily enhances the factuality, consistency, and entailment of the generated answers. Consequently, we harness the interactivity and multitasking ability of LLMs and produce progressively more precise and accurate answers. Experimental results on both automatic and human evaluation demonstrate the superiority of our approach in hallucination reduction compared to baselines.

{{</citation>}}


### (46/159) Get the gist? Using large language models for few-shot decontextualization (Benjamin Kane et al., 2023)

{{<citation>}}

Benjamin Kane, Lenhart Schubert. (2023)  
**Get the gist? Using large language models for few-shot decontextualization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP, Seq2Seq  
[Paper Link](http://arxiv.org/abs/2310.06254v1)  

---


**ABSTRACT**  
In many NLP applications that involve interpreting sentences within a rich context -- for instance, information retrieval systems or dialogue systems -- it is desirable to be able to preserve the sentence in a form that can be readily understood without context, for later reuse -- a process known as ``decontextualization''. While previous work demonstrated that generative Seq2Seq models could effectively perform decontextualization after being fine-tuned on a specific dataset, this approach requires expensive human annotations and may not transfer to other domains. We propose a few-shot method of decontextualization using a large language model, and present preliminary results showing that this method achieves viable performance on multiple domains using only a small set of examples.

{{</citation>}}


### (47/159) Model Tuning or Prompt Tuning? A Study of Large Language Models for Clinical Concept and Relation Extraction (Cheng Peng et al., 2023)

{{<citation>}}

Cheng Peng, Xi Yang, Kaleb E Smith, Zehao Yu, Aokun Chen, Jiang Bian, Yonghui Wu. (2023)  
**Model Tuning or Prompt Tuning? A Study of Large Language Models for Clinical Concept and Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Clinical, Language Model, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2310.06239v1)  

---


**ABSTRACT**  
Objective To develop soft prompt-based learning algorithms for large language models (LLMs), examine the shape of prompts, prompt-tuning using frozen/unfrozen LLMs, transfer learning, and few-shot learning abilities. Methods We developed a soft prompt-based LLM model and compared 4 training strategies including (1) fine-tuning without prompts; (2) hard-prompt with unfrozen LLMs; (3) soft-prompt with unfrozen LLMs; and (4) soft-prompt with frozen LLMs. We evaluated 7 pretrained LLMs using the 4 training strategies for clinical concept and relation extraction on two benchmark datasets. We evaluated the transfer learning ability of the prompt-based learning algorithms in a cross-institution setting. We also assessed the few-shot learning ability. Results and Conclusion When LLMs are unfrozen, GatorTron-3.9B with soft prompting achieves the best strict F1-scores of 0.9118 and 0.8604 for concept extraction, outperforming the traditional fine-tuning and hard prompt-based models by 0.6~3.1% and 1.2~2.9%, respectively; GatorTron-345M with soft prompting achieves the best F1-scores of 0.8332 and 0.7488 for end-to-end relation extraction, outperforming the other two models by 0.2~2% and 0.6~11.7%, respectively. When LLMs are frozen, small (i.e., 345 million parameters) LLMs have a big gap to be competitive with unfrozen models; scaling LLMs up to billions of parameters makes frozen LLMs competitive with unfrozen LLMs. For cross-institute evaluation, soft prompting with a frozen GatorTron-8.9B model achieved the best performance. This study demonstrates that (1) machines can learn soft prompts better than humans, (2) frozen LLMs have better few-shot learning ability and transfer learning ability to facilitate muti-institution applications, and (3) frozen LLMs require large models.

{{</citation>}}


### (48/159) Evolution of Natural Language Processing Technology: Not Just Language Processing Towards General Purpose AI (Masahiro Yamamoto, 2023)

{{<citation>}}

Masahiro Yamamoto. (2023)  
**Evolution of Natural Language Processing Technology: Not Just Language Processing Towards General Purpose AI**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: AI, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.06228v1)  

---


**ABSTRACT**  
Since the invention of computers, communication through natural language (actual human language) has been a dream technology. However, natural language is extremely difficult to mathematically formulate, making it difficult to realize as an algorithm without considering programming. While there have been numerous technological developments, one cannot say that any results allowing free utilization have been achieved thus far. In the case of language learning in humans, for instance when learning one's mother tongue or foreign language, one must admit that this process is similar to the adage "practice makes perfect" in principle, even though the learning method is significant up to a point. Deep learning has played a central role in contemporary AI technology in recent years. When applied to natural language processing (NLP), this produced unprecedented results. Achievements exceeding the initial predictions have been reported from the results of learning vast amounts of textual data using deep learning. For instance, four arithmetic operations could be performed without explicit learning, thereby enabling the explanation of complex images and the generation of images from corresponding explanatory texts. It is an accurate example of the learner embodying the concept of "practice makes perfect" by using vast amounts of textual data. This report provides a technological explanation of how cutting-edge NLP has made it possible to realize the "practice makes perfect" principle. Additionally, examples of how this can be applied to business are provided. We reported in June 2022 in Japanese on the NLP movement from late 2021 to early 2022. We would like to summarize this as a memorandum since this is just the initial movement leading to the current large language models (LLMs).

{{</citation>}}


### (49/159) GeoLLM: Extracting Geospatial Knowledge from Large Language Models (Rohin Manvi et al., 2023)

{{<citation>}}

Rohin Manvi, Samar Khanna, Gengchen Mai, Marshall Burke, David Lobell, Stefano Ermon. (2023)  
**GeoLLM: Extracting Geospatial Knowledge from Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06213v1)  

---


**ABSTRACT**  
The application of machine learning (ML) in a range of geospatial tasks is increasingly common but often relies on globally available covariates such as satellite imagery that can either be expensive or lack predictive power. Here we explore the question of whether the vast amounts of knowledge found in Internet language corpora, now compressed within large language models (LLMs), can be leveraged for geospatial prediction tasks. We first demonstrate that LLMs embed remarkable spatial information about locations, but naively querying LLMs using geographic coordinates alone is ineffective in predicting key indicators like population density. We then present GeoLLM, a novel method that can effectively extract geospatial knowledge from LLMs with auxiliary map data from OpenStreetMap. We demonstrate the utility of our approach across multiple tasks of central interest to the international community, including the measurement of population density and economic livelihoods. Across these tasks, our method demonstrates a 70% improvement in performance (measured using Pearson's $r^2$) relative to baselines that use nearest neighbors or use information directly from the prompt, and performance equal to or exceeding satellite-based benchmarks in the literature. With GeoLLM, we observe that GPT-3.5 outperforms Llama 2 and RoBERTa by 19% and 51% respectively, suggesting that the performance of our method scales well with the size of the model and its pretraining dataset. Our experiments reveal that LLMs are remarkably sample-efficient, rich in geospatial information, and robust across the globe. Crucially, GeoLLM shows promise in mitigating the limitations of existing geospatial covariates and complementing them well.

{{</citation>}}


## cs.AI (12)



### (50/159) Large Language Models can Learn Rules (Zhaocheng Zhu et al., 2023)

{{<citation>}}

Zhaocheng Zhu, Yuan Xue, Xinyun Chen, Denny Zhou, Jian Tang, Dale Schuurmans, Hanjun Dai. (2023)  
**Large Language Models can Learn Rules**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07064v1)  

---


**ABSTRACT**  
When prompted with a few examples and intermediate steps, large language models (LLMs) have demonstrated impressive performance in various reasoning tasks. However, prompting methods that rely on implicit knowledge in an LLM often hallucinate incorrect answers when the implicit knowledge is wrong or inconsistent with the task. To tackle this problem, we present Hypotheses-to-Theories (HtT), a framework that learns a rule library for reasoning with LLMs. HtT contains two stages, an induction stage and a deduction stage. In the induction stage, an LLM is first asked to generate and verify rules over a set of training examples. Rules that appear and lead to correct answers sufficiently often are collected to form a rule library. In the deduction stage, the LLM is then prompted to employ the learned rule library to perform reasoning to answer test questions. Experiments on both numerical reasoning and relational reasoning problems show that HtT improves existing prompting methods, with an absolute gain of 11-27% in accuracy. The learned rules are also transferable to different models and to different forms of the same problem.

{{</citation>}}


### (51/159) The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets (Samuel Marks et al., 2023)

{{<citation>}}

Samuel Marks, Max Tegmark. (2023)  
**The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06824v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have impressive capabilities, but are also prone to outputting falsehoods. Recent work has developed techniques for inferring whether a LLM is telling the truth by training probes on the LLM's internal activations. However, this line of work is controversial, with some authors pointing out failures of these probes to generalize in basic ways, among other conceptual issues. In this work, we curate high-quality datasets of true/false statements and use them to study in detail the structure of LLM representations of truth, drawing on three lines of evidence: 1. Visualizations of LLM true/false statement representations, which reveal clear linear structure. 2. Transfer experiments in which probes trained on one dataset generalize to different datasets. 3. Causal evidence obtained by surgically intervening in a LLM's forward pass, causing it to treat false statements as true and vice versa. Overall, we present evidence that language models linearly represent the truth or falsehood of factual statements. We also introduce a novel technique, mass-mean probing, which generalizes better and is more causally implicated in model outputs than other probing techniques.

{{</citation>}}


### (52/159) OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text (Keiran Paster et al., 2023)

{{<citation>}}

Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, Jimmy Ba. (2023)  
**OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: PaLM  
[Paper Link](http://arxiv.org/abs/2310.06786v1)  

---


**ABSTRACT**  
There is growing evidence that pretraining on high quality, carefully thought-out tokens such as code or mathematics plays an important role in improving the reasoning abilities of large language models. For example, Minerva, a PaLM model finetuned on billions of tokens of mathematical documents from arXiv and the web, reported dramatically improved performance on problems that require quantitative reasoning. However, because all known open source web datasets employ preprocessing that does not faithfully preserve mathematical notation, the benefits of large scale training on quantitive web documents are unavailable to the research community. We introduce OpenWebMath, an open dataset inspired by these works containing 14.7B tokens of mathematical webpages from Common Crawl. We describe in detail our method for extracting text and LaTeX content and removing boilerplate from HTML documents, as well as our methods for quality filtering and deduplication. Additionally, we run small-scale experiments by training 1.4B parameter language models on OpenWebMath, showing that models trained on 14.7B tokens of our dataset surpass the performance of models trained on over 20x the amount of general language data. We hope that our dataset, openly released on the Hugging Face Hub, will help spur advances in the reasoning abilities of large language models.

{{</citation>}}


### (53/159) Exploring Memorization in Fine-tuned Language Models (Shenglai Zeng et al., 2023)

{{<citation>}}

Shenglai Zeng, Yaxin Li, Jie Ren, Yiding Liu, Han Xu, Pengfei He, Yue Xing, Shuaiqiang Wang, Jiliang Tang, Dawei Yin. (2023)  
**Exploring Memorization in Fine-tuned Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06714v1)  

---


**ABSTRACT**  
LLMs have shown great capabilities in various tasks but also exhibited memorization of training data, thus raising tremendous privacy and copyright concerns. While prior work has studied memorization during pre-training, the exploration of memorization during fine-tuning is rather limited. Compared with pre-training, fine-tuning typically involves sensitive data and diverse objectives, thus may bring unique memorization behaviors and distinct privacy risks. In this work, we conduct the first comprehensive analysis to explore LMs' memorization during fine-tuning across tasks. Our studies with open-sourced and our own fine-tuned LMs across various tasks indicate that fine-tuned memorization presents a strong disparity among tasks. We provide an understanding of this task disparity via sparse coding theory and unveil a strong correlation between memorization and attention score distribution. By investigating its memorization behavior, multi-task fine-tuning paves a potential strategy to mitigate fine-tuned memorization.

{{</citation>}}


### (54/159) Assessing the Impact of a Supervised Classification Filter on Flow-based Hybrid Network Anomaly Detection (Dominik Macko et al., 2023)

{{<citation>}}

Dominik Macko, Patrik Goldschmidt, Peter Pištek, Daniela Chudá. (2023)  
**Assessing the Impact of a Supervised Classification Filter on Flow-based Hybrid Network Anomaly Detection**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CR, cs-NI, cs.AI  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.06656v1)  

---


**ABSTRACT**  
Constant evolution and the emergence of new cyberattacks require the development of advanced techniques for defense. This paper aims to measure the impact of a supervised filter (classifier) in network anomaly detection. We perform our experiments by employing a hybrid anomaly detection approach in network flow data. For this purpose, we extended a state-of-the-art autoencoder-based anomaly detection method by prepending a binary classifier acting as a prefilter for the anomaly detector. The method was evaluated on the publicly available real-world dataset UGR'16. Our empirical results indicate that the hybrid approach does offer a higher detection rate of known attacks than a standalone anomaly detector while still retaining the ability to detect zero-day attacks. Employing a supervised binary prefilter has increased the AUC metric by over 11%, detecting 30% more attacks while keeping the number of false positives approximately the same.

{{</citation>}}


### (55/159) Automated clinical coding using off-the-shelf large language models (Joseph S. Boyle et al., 2023)

{{<citation>}}

Joseph S. Boyle, Antanas Kascenas, Pat Lok, Maria Liakata, Alison Q. O'Neil. (2023)  
**Automated clinical coding using off-the-shelf large language models**  

---
Primary Category: cs.AI  
Categories: I-2-7; I-2-8, cs-AI, cs-CL, cs.AI  
Keywords: GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.06552v1)  

---


**ABSTRACT**  
The task of assigning diagnostic ICD codes to patient hospital admissions is typically performed by expert human coders. Efforts towards automated ICD coding are dominated by supervised deep learning models. However, difficulties in learning to predict the large number of rare codes remain a barrier to adoption in clinical practice. In this work, we leverage off-the-shelf pre-trained generative large language models (LLMs) to develop a practical solution that is suitable for zero-shot and few-shot code assignment. Unsupervised pre-training alone does not guarantee precise knowledge of the ICD ontology and specialist clinical coding task, therefore we frame the task as information extraction, providing a description of each coded concept and asking the model to retrieve related mentions. For efficiency, rather than iterating over all codes, we leverage the hierarchical nature of the ICD ontology to sparsely search for relevant codes. Then, in a second stage, which we term 'meta-refinement', we utilise GPT-4 to select a subset of the relevant labels as predictions. We validate our method using Llama-2, GPT-3.5 and GPT-4 on the CodiEsp dataset of ICD-coded clinical case documents. Our tree-search method achieves state-of-the-art performance on rarer classes, achieving the best macro-F1 of 0.225, whilst achieving slightly lower micro-F1 of 0.157, compared to 0.216 and 0.219 respectively from PLM-ICD. To the best of our knowledge, this is the first method for automated ICD coding requiring no task-specific learning.

{{</citation>}}


### (56/159) Realizing Stabilized Landing for Computation-Limited Reusable Rockets: A Quantum Reinforcement Learning Approach (Gyu Seon Kim et al., 2023)

{{<citation>}}

Gyu Seon Kim, JaeHyun Chung, Soohyun Park. (2023)  
**Realizing Stabilized Landing for Computation-Limited Reusable Rockets: A Quantum Reinforcement Learning Approach**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.06541v1)  

---


**ABSTRACT**  
The advent of reusable rockets has heralded a new era in space exploration, reducing the costs of launching satellites by a significant factor. Traditional rockets were disposable, but the design of reusable rockets for repeated use has revolutionized the financial dynamics of space missions. The most critical phase of reusable rockets is the landing stage, which involves managing the tremendous speed and attitude for safe recovery. The complexity of this task presents new challenges for control systems, specifically in terms of precision and adaptability. Classical control systems like the proportional-integral-derivative (PID) controller lack the flexibility to adapt to dynamic system changes, making them costly and time-consuming to redesign of controller. This paper explores the integration of quantum reinforcement learning into the control systems of reusable rockets as a promising alternative. Unlike classical reinforcement learning, quantum reinforcement learning uses quantum bits that can exist in superposition, allowing for more efficient information encoding and reducing the number of parameters required. This leads to increased computational efficiency, reduced memory requirements, and more stable and predictable performance. Due to the nature of reusable rockets, which must be light, heavy computers cannot fit into them. In the reusable rocket scenario, quantum reinforcement learning, which has reduced memory requirements due to fewer parameters, is a good solution.

{{</citation>}}


### (57/159) MetaAgents: Simulating Interactions of Human Behaviors for LLM-based Task-oriented Coordination via Collaborative Generative Agents (Yuan Li et al., 2023)

{{<citation>}}

Yuan Li, Yixuan Zhang, Lichao Sun. (2023)  
**MetaAgents: Simulating Interactions of Human Behaviors for LLM-based Task-oriented Coordination via Collaborative Generative Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06500v1)  

---


**ABSTRACT**  
Significant advancements have occurred in the application of Large Language Models (LLMs) for various tasks and social simulations. Despite this, their capacities to coordinate within task-oriented social contexts are under-explored. Such capabilities are crucial if LLMs are to effectively mimic human-like social behavior and produce meaningful results. To bridge this gap, we introduce collaborative generative agents, endowing LLM-based Agents with consistent behavior patterns and task-solving abilities. We situate these agents in a simulated job fair environment as a case study to scrutinize their coordination skills. We propose a novel framework that equips collaborative generative agents with human-like reasoning abilities and specialized skills. Our evaluation demonstrates that these agents show promising performance. However, we also uncover limitations that hinder their effectiveness in more complex coordination tasks. Our work provides valuable insights into the role and evolution of LLMs in task-oriented social simulations.

{{</citation>}}


### (58/159) Memory efficient location recommendation through proximity-aware representation (Xuan Luo et al., 2023)

{{<citation>}}

Xuan Luo, Rui Lv, Hui Zhao. (2023)  
**Memory efficient location recommendation through proximity-aware representation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Attention, Self-Attention, Social Network  
[Paper Link](http://arxiv.org/abs/2310.06484v1)  

---


**ABSTRACT**  
Sequential location recommendation plays a huge role in modern life, which can enhance user experience, bring more profit to businesses and assist in government administration. Although methods for location recommendation have evolved significantly thanks to the development of recommendation systems, there is still limited utilization of geographic information, along with the ongoing challenge of addressing data sparsity. In response, we introduce a Proximity-aware based region representation for Sequential Recommendation (PASR for short), built upon the Self-Attention Network architecture. We tackle the sparsity issue through a novel loss function employing importance sampling, which emphasizes informative negative samples during optimization. Moreover, PASR enhances the integration of geographic information by employing a self-attention-based geography encoder to the hierarchical grid and proximity grid at each GPS point. To further leverage geographic information, we utilize the proximity-aware negative samplers to enhance the quality of negative samples. We conducted evaluations using three real-world Location-Based Social Networking (LBSN) datasets, demonstrating that PASR surpasses state-of-the-art sequential location recommendation methods

{{</citation>}}


### (59/159) Proceedings of The first international workshop on eXplainable AI for the Arts (XAIxArts) (Nick Bryan-Kinns et al., 2023)

{{<citation>}}

Nick Bryan-Kinns, Corey Ford, Alan Chamberlain, Steven David Benford, Helen Kennedy, Zijin Li, Wu Qiong, Gus G. Xia, Jeba Rezwana. (2023)  
**Proceedings of The first international workshop on eXplainable AI for the Arts (XAIxArts)**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-SD, cs.AI, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.06428v1)  

---


**ABSTRACT**  
This first international workshop on explainable AI for the Arts (XAIxArts) brought together a community of researchers in HCI, Interaction Design, AI, explainable AI (XAI), and digital arts to explore the role of XAI for the Arts.   Workshop held at the 15th ACM Conference on Creativity and Cognition (C&C 2023).

{{</citation>}}


### (60/159) I2SRM: Intra- and Inter-Sample Relationship Modeling for Multimodal Information Extraction (Yusheng Huang et al., 2023)

{{<citation>}}

Yusheng Huang, Zhouhan Lin. (2023)  
**I2SRM: Intra- and Inter-Sample Relationship Modeling for Multimodal Information Extraction**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Embedding, Information Extraction, Twitter  
[Paper Link](http://arxiv.org/abs/2310.06326v1)  

---


**ABSTRACT**  
Multimodal information extraction is attracting research attention nowadays, which requires aggregating representations from different modalities. In this paper, we present the Intra- and Inter-Sample Relationship Modeling (I2SRM) method for this task, which contains two modules. Firstly, the intra-sample relationship modeling module operates on a single sample and aims to learn effective representations. Embeddings from textual and visual modalities are shifted to bridge the modality gap caused by distinct pre-trained language and image models. Secondly, the inter-sample relationship modeling module considers relationships among multiple samples and focuses on capturing the interactions. An AttnMixup strategy is proposed, which not only enables collaboration among samples but also augments data to improve generalization. We conduct extensive experiments on the multimodal named entity recognition datasets Twitter-2015 and Twitter-2017, and the multimodal relation extraction dataset MNRE. Our proposed method I2SRM achieves competitive results, 77.12% F1-score on Twitter-2015, 88.40% F1-score on Twitter-2017, and 84.12% F1-score on MNRE.

{{</citation>}}


### (61/159) GPT-4 as an Agronomist Assistant? Answering Agriculture Exams Using Large Language Models (Bruno Silva et al., 2023)

{{<citation>}}

Bruno Silva, Leonardo Nunes, Roberto Estevão, Ranveer Chandra. (2023)  
**GPT-4 as an Agronomist Assistant? Answering Agriculture Exams Using Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06225v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding across various domains, including healthcare and finance. For some tasks, LLMs achieve similar or better performance than trained human beings, therefore it is reasonable to employ human exams (e.g., certification tests) to assess the performance of LLMs. We present a comprehensive evaluation of popular LLMs, such as Llama 2 and GPT, on their ability to answer agriculture-related questions. In our evaluation, we also employ RAG (Retrieval-Augmented Generation) and ER (Ensemble Refinement) techniques, which combine information retrieval, generation capabilities, and prompting strategies to improve the LLMs' performance. To demonstrate the capabilities of LLMs, we selected agriculture exams and benchmark datasets from three of the largest agriculture producer countries: Brazil, India, and the USA. Our analysis highlights GPT-4's ability to achieve a passing score on exams to earn credits for renewing agronomist certifications, answering 93% of the questions correctly and outperforming earlier general-purpose models, which achieved 88% accuracy. On one of our experiments, GPT-4 obtained the highest performance when compared to human subjects. This performance suggests that GPT-4 could potentially pass on major graduate education admission tests or even earn credits for renewing agronomy certificates. We also explore the models' capacity to address general agriculture-related questions and generate crop management guidelines for Brazilian and Indian farmers, utilizing robust datasets from the Brazilian Agency of Agriculture (Embrapa) and graduate program exams from India. The results suggest that GPT-4, ER, and RAG can contribute meaningfully to agricultural education, assessment, and crop management practice, offering valuable insights to farmers and agricultural professionals.

{{</citation>}}


## cs.SD (3)



### (62/159) Acoustic Model Fusion for End-to-end Speech Recognition (Zhihong Lei et al., 2023)

{{<citation>}}

Zhihong Lei, Mingbin Xu, Shiyi Han, Leo Liu, Zhen Huang, Tim Ng, Yuanyuan Zhang, Ernest Pusateri, Mirko Hannemann, Yaqiao Deng, Man-Hung Siu. (2023)  
**Acoustic Model Fusion for End-to-end Speech Recognition**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.07062v1)  

---


**ABSTRACT**  
Recent advances in deep learning and automatic speech recognition (ASR) have enabled the end-to-end (E2E) ASR system and boosted the accuracy to a new level. The E2E systems implicitly model all conventional ASR components, such as the acoustic model (AM) and the language model (LM), in a single network trained on audio-text pairs. Despite this simpler system architecture, fusing a separate LM, trained exclusively on text corpora, into the E2E system has proven to be beneficial. However, the application of LM fusion presents certain drawbacks, such as its inability to address the domain mismatch issue inherent to the internal AM. Drawing inspiration from the concept of LM fusion, we propose the integration of an external AM into the E2E system to better address the domain mismatch. By implementing this novel approach, we have achieved a significant reduction in the word error rate, with an impressive drop of up to 14.3% across varied test sets. We also discovered that this AM fusion approach is particularly beneficial in enhancing named entity recognition.

{{</citation>}}


### (63/159) AutoCycle-VC: Towards Bottleneck-Independent Zero-Shot Cross-Lingual Voice Conversion (Haeyun Choi et al., 2023)

{{<citation>}}

Haeyun Choi, Jio Gim, Yuho Lee, Youngin Kim, Young-Joo Suh. (2023)  
**AutoCycle-VC: Towards Bottleneck-Independent Zero-Shot Cross-Lingual Voice Conversion**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.06546v1)  

---


**ABSTRACT**  
This paper proposes a simple and robust zero-shot voice conversion system with a cycle structure and mel-spectrogram pre-processing. Previous works suffer from information loss and poor synthesis quality due to their reliance on a carefully designed bottleneck structure. Moreover, models relying solely on self-reconstruction loss struggled with reproducing different speakers' voices. To address these issues, we suggested a cycle-consistency loss that considers conversion back and forth between target and source speakers. Additionally, stacked random-shuffled mel-spectrograms and a label smoothing method are utilized during speaker encoder training to extract a time-independent global speaker representation from speech, which is the key to a zero-shot conversion. Our model outperforms existing state-of-the-art results in both subjective and objective evaluations. Furthermore, it facilitates cross-lingual voice conversions and enhances the quality of synthesized speech.

{{</citation>}}


### (64/159) An experiment on an automated literature survey of data-driven speech enhancement methods (Arthur dos Santos et al., 2023)

{{<citation>}}

Arthur dos Santos, Jayr Pereira, Rodrigo Nogueira, Bruno Masiero, Shiva Sander-Tavallaey, Elias Zea. (2023)  
**An experiment on an automated literature survey of data-driven speech enhancement methods**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.06260v1)  

---


**ABSTRACT**  
The increasing number of scientific publications in acoustics, in general, presents difficulties in conducting traditional literature surveys. This work explores the use of a generative pre-trained transformer (GPT) model to automate a literature survey of 116 articles on data-driven speech enhancement methods. The main objective is to evaluate the capabilities and limitations of the model in providing accurate responses to specific queries about the papers selected from a reference human-based survey. While we see great potential to automate literature surveys in acoustics, improvements are needed to address technical questions more clearly and accurately.

{{</citation>}}


## cs.HC (4)



### (65/159) QualiGPT: GPT as an easy-to-use tool for qualitative coding (He Zhang et al., 2023)

{{<citation>}}

He Zhang, Chuhao Wu, Jingyi Xie, ChanMin Kim, John M. Carroll. (2023)  
**QualiGPT: GPT as an easy-to-use tool for qualitative coding**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: ChatGPT, GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.07061v1)  

---


**ABSTRACT**  
Qualitative research delves deeply into individual complex perspectives on technology and various phenomena. However, a meticulous analysis of qualitative data often requires a significant amount of time, especially during the crucial coding stage. Although there is software specifically designed for qualitative evaluation, many of these platforms fall short in terms of automatic coding, intuitive usability, and cost-effectiveness. With the rise of Large Language Models (LLMs) such as GPT-3 and its successors, we are at the forefront of a transformative era for enhancing qualitative analysis. In this paper, we introduce QualiGPT, a specialized tool designed after considering challenges associated with ChatGPT and qualitative analysis. It harnesses the capabilities of the Generative Pretrained Transformer (GPT) and its API for thematic analysis of qualitative data. By comparing traditional manual coding with QualiGPT's analysis on both simulated and actual datasets, we verify that QualiGPT not only refines the qualitative analysis process but also elevates its transparency, credibility, and accessibility. Notably, compared to existing analytical platforms, QualiGPT stands out with its intuitive design, significantly reducing the learning curve and operational barriers for users.

{{</citation>}}


### (66/159) Automatic Macro Mining from Interaction Traces at Scale (Forrest Huang et al., 2023)

{{<citation>}}

Forrest Huang, Gang Li, Tao Li, Yang Li. (2023)  
**Automatic Macro Mining from Interaction Traces at Scale**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-HC, cs-LG, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.07023v1)  

---


**ABSTRACT**  
Macros are building block tasks of our everyday smartphone activity (e.g., "login", or "booking a flight"). Effectively extracting macros is important for understanding mobile interaction and enabling task automation. These macros are however difficult to extract at scale as they can be comprised of multiple steps yet hidden within programmatic components of the app. In this paper, we introduce a novel approach based on Large Language Models (LLMs) to automatically extract semantically meaningful macros from both random and user-curated mobile interaction traces. The macros produced by our approach are automatically tagged with natural language descriptions and are fully executable. To examine the quality of extraction, we conduct multiple studies, including user evaluation, comparative analysis against human-curated tasks, and automatic execution of these macros. These experiments and analyses show the effectiveness of our approach and the usefulness of extracted macros in various downstream applications.

{{</citation>}}


### (67/159) Case Law Grounding: Aligning Judgments of Humans and AI on Socially-Constructed Concepts (Quan Ze Chen et al., 2023)

{{<citation>}}

Quan Ze Chen, Amy X. Zhang. (2023)  
**Case Law Grounding: Aligning Judgments of Humans and AI on Socially-Constructed Concepts**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07019v1)  

---


**ABSTRACT**  
Systems for making determinations on socially-constructed and complex concepts at scale are increasingly being deployed. To make such fuzzy concepts tractable for training and evaluating AI, aligning model outputs, or human-in-the-loop workflows, the prevailing strategy involves developing `constitutions' in the form of rules, policies, or principles. However, high-level rules often fail to capture situational nuances or have differing interpretations, resulting in inconsistent decisions. In this work, we introduce case law grounding (CLG), a hybrid workflow inspired by case law in the legal realm where past judgments on specific cases inform new decisions. Evaluating on two task domains, we find that CLG can improve alignment of decisions (+9.6% and +10.9% accuracy) and consistency ($\Delta\bar{\kappa}$ of +0.263 and +0.433) of human decision-makers, while also providing auditable rationales. We also find similarly substantial alignment improvements for an LLM decision-maker (+25% and +23% accuracy).

{{</citation>}}


### (68/159) Improved prompting and process for writing user personas with LLMs, using qualitative interviews: Capturing behaviour and personality traits of users (Stefano De Paoli, 2023)

{{<citation>}}

Stefano De Paoli. (2023)  
**Improved prompting and process for writing user personas with LLMs, using qualitative interviews: Capturing behaviour and personality traits of users**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-CY, cs-HC, cs.HC  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06391v1)  

---


**ABSTRACT**  
This draft paper presents a workflow for creating User Personas with Large Language Models, using the results of a Thematic Analysis of qualitative interviews. The proposed workflow uses improved prompting and a larger pool of Themes, compared to previous work conducted by the author for the same task. This is possible due to the capabilities of a recently released LLM which allows the processing of 16 thousand tokens (GPT3.5-Turbo-16k) and also due to the possibility to offer a refined prompting for the creation of Personas. The paper offers details of performing Phase 2 and 3 of Thematic Analysis, and then discusses the improved workflow for creating Personas. The paper also offers some reflections on the relationship between the proposed process and existing approaches to Personas such as the data-driven and qualitative Personas. Moreover, the paper offers reflections on the capacity of LLMs to capture user behaviours and personality traits, from the underlying dataset of qualitative interviews used for the analysis.

{{</citation>}}


## cs.CV (21)



### (69/159) Computational Pathology at Health System Scale -- Self-Supervised Foundation Models from Three Billion Images (Gabriele Campanella et al., 2023)

{{<citation>}}

Gabriele Campanella, Ricky Kwan, Eugene Fluder, Jennifer Zeng, Aryeh Stock, Brandon Veremis, Alexandros D. Polydorides, Cyrus Hedvat, Adam Schoenfeld, Chad Vanderbilt, Patricia Kovatch, Carlos Cordon-Cardo, Thomas J. Fuchs. (2023)  
**Computational Pathology at Health System Scale -- Self-Supervised Foundation Models from Three Billion Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.07033v1)  

---


**ABSTRACT**  
Recent breakthroughs in self-supervised learning have enabled the use of large unlabeled datasets to train visual foundation models that can generalize to a variety of downstream tasks. While this training paradigm is well suited for the medical domain where annotations are scarce, large-scale pre-training in the medical domain, and in particular pathology, has not been extensively studied. Previous work in self-supervised learning in pathology has leveraged smaller datasets for both pre-training and evaluating downstream performance. The aim of this project is to train the largest academic foundation model and benchmark the most prominent self-supervised learning algorithms by pre-training and evaluating downstream performance on large clinical pathology datasets. We collected the largest pathology dataset to date, consisting of over 3 billion images from over 423 thousand microscopy slides. We compared pre-training of visual transformer models using the masked autoencoder (MAE) and DINO algorithms. We evaluated performance on six clinically relevant tasks from three anatomic sites and two institutions: breast cancer detection, inflammatory bowel disease detection, breast cancer estrogen receptor prediction, lung adenocarcinoma EGFR mutation prediction, and lung cancer immunotherapy response prediction. Our results demonstrate that pre-training on pathology data is beneficial for downstream performance compared to pre-training on natural images. Additionally, the DINO algorithm achieved better generalization performance across all tasks tested. The presented results signify a phase change in computational pathology research, paving the way into a new era of more performant models based on large-scale, parallel pre-training at the billion-image scale.

{{</citation>}}


### (70/159) Zero-Shot Open-Vocabulary Tracking with Large Pre-Trained Models (Wen-Hsuan Chu et al., 2023)

{{<citation>}}

Wen-Hsuan Chu, Adam W. Harley, Pavel Tokmakov, Achal Dave, Leonidas Guibas, Katerina Fragkiadaki. (2023)  
**Zero-Shot Open-Vocabulary Tracking with Large Pre-Trained Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pre-Trained Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.06992v1)  

---


**ABSTRACT**  
Object tracking is central to robot perception and scene understanding. Tracking-by-detection has long been a dominant paradigm for object tracking of specific object categories. Recently, large-scale pre-trained models have shown promising advances in detecting and segmenting objects and parts in 2D static images in the wild. This begs the question: can we re-purpose these large-scale pre-trained static image models for open-vocabulary video tracking? In this paper, we re-purpose an open-vocabulary detector, segmenter, and dense optical flow estimator, into a model that tracks and segments objects of any category in 2D videos. Our method predicts object and part tracks with associated language descriptions in monocular videos, rebuilding the pipeline of Tractor with modern large pre-trained models for static image detection and segmentation: we detect open-vocabulary object instances and propagate their boxes from frame to frame using a flow-based motion model, refine the propagated boxes with the box regression module of the visual detector, and prompt an open-world segmenter with the refined box to segment the objects. We decide the termination of an object track based on the objectness score of the propagated boxes, as well as forward-backward optical flow consistency. We re-identify objects across occlusions using deep feature matching. We show that our model achieves strong performance on multiple established video object segmentation and tracking benchmarks, and can produce reasonable tracks in manipulation data. In particular, our model outperforms previous state-of-the-art in UVO and BURST, benchmarks for open-world object tracking and segmentation, despite never being explicitly trained for tracking. We hope that our approach can serve as a simple and extensible framework for future research.

{{</citation>}}


### (71/159) On the Interpretability of Part-Prototype Based Classifiers: A Human Centric Analysis (Omid Davoodi et al., 2023)

{{<citation>}}

Omid Davoodi, Shayan Mohammadizadehsamakosh, Majid Komeili. (2023)  
**On the Interpretability of Part-Prototype Based Classifiers: A Human Centric Analysis**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-HC, cs-LG, cs.CV  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2310.06966v1)  

---


**ABSTRACT**  
Part-prototype networks have recently become methods of interest as an interpretable alternative to many of the current black-box image classifiers. However, the interpretability of these methods from the perspective of human users has not been sufficiently explored. In this work, we have devised a framework for evaluating the interpretability of part-prototype-based models from a human perspective. The proposed framework consists of three actionable metrics and experiments. To demonstrate the usefulness of our framework, we performed an extensive set of experiments using Amazon Mechanical Turk. They not only show the capability of our framework in assessing the interpretability of various part-prototype-based models, but they also are, to the best of our knowledge, the most comprehensive work on evaluating such methods in a unified framework.

{{</citation>}}


### (72/159) TopoMLP: An Simple yet Strong Pipeline for Driving Topology Reasoning (Dongming Wu et al., 2023)

{{<citation>}}

Dongming Wu, Jiahao Chang, Fan Jia, Yingfei Liu, Tiancai Wang, Jianbing Shen. (2023)  
**TopoMLP: An Simple yet Strong Pipeline for Driving Topology Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2310.06753v1)  

---


**ABSTRACT**  
Topology reasoning aims to comprehensively understand road scenes and present drivable routes in autonomous driving. It requires detecting road centerlines (lane) and traffic elements, further reasoning their topology relationship, i.e., lane-lane topology, and lane-traffic topology. In this work, we first present that the topology score relies heavily on detection performance on lane and traffic elements. Therefore, we introduce a powerful 3D lane detector and an improved 2D traffic element detector to extend the upper limit of topology performance. Further, we propose TopoMLP, a simple yet high-performance pipeline for driving topology reasoning. Based on the impressive detection performance, we develop two simple MLP-based heads for topology generation. TopoMLP achieves state-of-the-art performance on OpenLane-V2 benchmark, i.e., 41.2% OLS with ResNet-50 backbone. It is also the 1st solution for 1st OpenLane Topology in Autonomous Driving Challenge. We hope such simple and strong pipeline can provide some new insights to the community. Code is at https://github.com/wudongming97/TopoMLP.

{{</citation>}}


### (73/159) How (not) to ensemble LVLMs for VQA (Lisa Alazraki et al., 2023)

{{<citation>}}

Lisa Alazraki, Lluis Castrejon, Mostafa Dehghani, Fantine Huot, Jasper Uijlings, Thomas Mensink. (2023)  
**How (not) to ensemble LVLMs for VQA**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.06641v1)  

---


**ABSTRACT**  
This paper studies ensembling in the era of Large Vision-Language Models (LVLMs). Ensembling is a classical method to combine different models to get increased performance. In the recent work on Encyclopedic-VQA the authors examine a wide variety of models to solve their task: from vanilla LVLMs, to models including the caption as extra context, to models augmented with Lens-based retrieval of Wikipedia pages. Intuitively these models are highly complementary, which should make them ideal for ensembling. Indeed, an oracle experiment shows potential gains from 48.8% accuracy (the best single model) all the way up to 67% (best possible ensemble). So it is a trivial exercise to create an ensemble with substantial real gains. Or is it?

{{</citation>}}


### (74/159) EViT: An Eagle Vision Transformer with Bi-Fovea Self-Attention (Yulong Shi et al., 2023)

{{<citation>}}

Yulong Shi, Mingwei Sun, Yongshuai Wang, Rui Wang, Hui Sun, Zengqiang Chen. (2023)  
**EViT: An Eagle Vision Transformer with Bi-Fovea Self-Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Self-Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.06629v1)  

---


**ABSTRACT**  
Because of the advancement of deep learning technology, vision transformer has demonstrated competitive performance in various computer vision tasks. Unfortunately, vision transformer still faces some challenges such as high computational complexity and absence of desirable inductive bias. To alleviate these problems, this study proposes a novel Bi-Fovea Self-Attention (BFSA) inspired by the physiological structure and characteristics of bi-fovea vision in eagle eyes. This BFSA can simulate the shallow fovea and deep fovea functions of eagle vision, enabling the network to extract feature representations of targets from coarse to fine, facilitating the interaction of multi-scale feature representations. Additionally, this study designs a Bionic Eagle Vision (BEV) block based on BFSA and CNN. It combines CNN and Vision Transformer, to enhance the network's local and global representation ability for targets. Furthermore, this study develops a unified and efficient general pyramid backbone network family, named Eagle Vision Transformers (EViTs) by stacking the BEV blocks. Experimental results on various computer vision tasks including image classification, object detection, instance segmentation and other transfer learning tasks show that the proposed EViTs perform significantly better than the baselines under similar model sizes, which exhibits faster speed on graphics processing unit compared to other models. Code will be released at https://github.com/nkusyl.

{{</citation>}}


### (75/159) REVO-LION: Evaluating and Refining Vision-Language Instruction Tuning Datasets (Ning Liao et al., 2023)

{{<citation>}}

Ning Liao, Shaofeng Zhang, Renqiu Xia, Bo Zhang, Min Cao, Yu Qiao, Junchi Yan. (2023)  
**REVO-LION: Evaluating and Refining Vision-Language Instruction Tuning Datasets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2310.06594v1)  

---


**ABSTRACT**  
There is an emerging line of research on multimodal instruction tuning, and a line of benchmarks have been proposed for evaluating these models recently. Instead of evaluating the models directly, in this paper we try to evaluate the Vision-Language Instruction-Tuning (VLIT) datasets themselves and further seek the way of building a dataset for developing an all-powerful VLIT model, which we believe could also be of utility for establishing a grounded protocol for benchmarking VLIT models. For effective analysis of VLIT datasets that remains an open question, we propose a tune-cross-evaluation paradigm: tuning on one dataset and evaluating on the others in turn. For each single tune-evaluation experiment set, we define the Meta Quality (MQ) as the mean score measured by a series of caption metrics including BLEU, METEOR, and ROUGE-L to quantify the quality of a certain dataset or a sample. On this basis, to evaluate the comprehensiveness of a dataset, we develop the Dataset Quality (DQ) covering all tune-evaluation sets. To lay the foundation for building a comprehensive dataset and developing an all-powerful model for practical applications, we further define the Sample Quality (SQ) to quantify the all-sided quality of each sample. Extensive experiments validate the rationality of the proposed evaluation paradigm. Based on the holistic evaluation, we build a new dataset, REVO-LION (REfining VisiOn-Language InstructiOn tuNing), by collecting samples with higher SQ from each dataset. With only half of the full data, the model trained on REVO-LION can achieve performance comparable to simply adding all VLIT datasets up. In addition to developing an all-powerful model, REVO-LION also includes an evaluation set, which is expected to serve as a convenient evaluation benchmark for future research.

{{</citation>}}


### (76/159) SketchBodyNet: A Sketch-Driven Multi-faceted Decoder Network for 3D Human Reconstruction (Fei Wang et al., 2023)

{{<citation>}}

Fei Wang, Kongzhang Tang, Hefeng Wu, Baoquan Zhao, Hao Cai, Teng Zhou. (2023)  
**SketchBodyNet: A Sketch-Driven Multi-faceted Decoder Network for 3D Human Reconstruction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2310.06577v1)  

---


**ABSTRACT**  
Reconstructing 3D human shapes from 2D images has received increasing attention recently due to its fundamental support for many high-level 3D applications. Compared with natural images, freehand sketches are much more flexible to depict various shapes, providing a high potential and valuable way for 3D human reconstruction. However, such a task is highly challenging. The sparse abstract characteristics of sketches add severe difficulties, such as arbitrariness, inaccuracy, and lacking image details, to the already badly ill-posed problem of 2D-to-3D reconstruction. Although current methods have achieved great success in reconstructing 3D human bodies from a single-view image, they do not work well on freehand sketches. In this paper, we propose a novel sketch-driven multi-faceted decoder network termed SketchBodyNet to address this task. Specifically, the network consists of a backbone and three separate attention decoder branches, where a multi-head self-attention module is exploited in each decoder to obtain enhanced features, followed by a multi-layer perceptron. The multi-faceted decoders aim to predict the camera, shape, and pose parameters, respectively, which are then associated with the SMPL model to reconstruct the corresponding 3D human mesh. In learning, existing 3D meshes are projected via the camera parameters into 2D synthetic sketches with joints, which are combined with the freehand sketches to optimize the model. To verify our method, we collect a large-scale dataset of about 26k freehand sketches and their corresponding 3D meshes containing various poses of human bodies from 14 different angles. Extensive experimental results demonstrate our SketchBodyNet achieves superior performance in reconstructing 3D human meshes from freehand sketches.

{{</citation>}}


### (77/159) Compositional Representation Learning for Brain Tumour Segmentation (Xiao Liu et al., 2023)

{{<citation>}}

Xiao Liu, Antanas Kascenas, Hannah Watson, Sotirios A. Tsaftaris, Alison Q. O'Neil. (2023)  
**Compositional Representation Learning for Brain Tumour Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.06562v1)  

---


**ABSTRACT**  
For brain tumour segmentation, deep learning models can achieve human expert-level performance given a large amount of data and pixel-level annotations. However, the expensive exercise of obtaining pixel-level annotations for large amounts of data is not always feasible, and performance is often heavily reduced in a low-annotated data regime. To tackle this challenge, we adapt a mixed supervision framework, vMFNet, to learn robust compositional representations using unsupervised learning and weak supervision alongside non-exhaustive pixel-level pathology labels. In particular, we use the BraTS dataset to simulate a collection of 2-point expert pathology annotations indicating the top and bottom slice of the tumour (or tumour sub-regions: peritumoural edema, GD-enhancing tumour, and the necrotic / non-enhancing tumour) in each MRI volume, from which weak image-level labels that indicate the presence or absence of the tumour (or the tumour sub-regions) in the image are constructed. Then, vMFNet models the encoded image features with von-Mises-Fisher (vMF) distributions, via learnable and compositional vMF kernels which capture information about structures in the images. We show that good tumour segmentation performance can be achieved with a large amount of weakly labelled data but only a small amount of fully-annotated data. Interestingly, emergent learning of anatomical structures occurs in the compositional representation even given only supervision relating to pathology (tumour).

{{</citation>}}


### (78/159) Deep Learning for Automatic Detection and Facial Recognition in Japanese Macaques: Illuminating Social Networks (Julien Paulet et al., 2023)

{{<citation>}}

Julien Paulet, Axel Molina, Benjamin Beltzung, Takafumi Suzumura, Shinya Yamamoto, Cédric Sueur. (2023)  
**Deep Learning for Automatic Detection and Facial Recognition in Japanese Macaques: Illuminating Social Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-SI, cs.CV  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2310.06489v1)  

---


**ABSTRACT**  
Individual identification plays a pivotal role in ecology and ethology, notably as a tool for complex social structures understanding. However, traditional identification methods often involve invasive physical tags and can prove both disruptive for animals and time-intensive for researchers. In recent years, the integration of deep learning in research offered new methodological perspectives through automatization of complex tasks. Harnessing object detection and recognition technologies is increasingly used by researchers to achieve identification on video footage. This study represents a preliminary exploration into the development of a non-invasive tool for face detection and individual identification of Japanese macaques (Macaca fuscata) through deep learning. The ultimate goal of this research is, using identifications done on the dataset, to automatically generate a social network representation of the studied population. The current main results are promising: (i) the creation of a Japanese macaques' face detector (Faster-RCNN model), reaching a 82.2% accuracy and (ii) the creation of an individual recognizer for K{\=o}jima island macaques population (YOLOv8n model), reaching a 83% accuracy. We also created a K{\=o}jima population social network by traditional methods, based on co-occurrences on videos. Thus, we provide a benchmark against which the automatically generated network will be assessed for reliability. These preliminary results are a testament to the potential of this innovative approach to provide the scientific community with a tool for tracking individuals and social network studies in Japanese macaques.

{{</citation>}}


### (79/159) Focus on Local Regions for Query-based Object Detection (Hongbin Xu et al., 2023)

{{<citation>}}

Hongbin Xu, Yamei Xia, Shuai Zhao, Bo Cheng. (2023)  
**Focus on Local Regions for Query-based Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.06470v1)  

---


**ABSTRACT**  
Query-based methods have garnered significant attention in object detection since the advent of DETR, the pioneering end-to-end query-based detector. However, these methods face challenges like slow convergence and suboptimal performance. Notably, self-attention in object detection often hampers convergence due to its global focus. To address these issues, we propose FoLR, a transformer-like architecture with only decoders. We enhance the self-attention mechanism by isolating connections between irrelevant objects that makes it focus on local regions but not global regions. We also design the adaptive sampling method to extract effective features based on queries' local regions from feature maps. Additionally, we employ a look-back strategy for decoders to retain prior information, followed by the Feature Mixer module to fuse features and queries. Experimental results demonstrate FoLR's state-of-the-art performance in query-based detectors, excelling in convergence speed and computational efficiency.

{{</citation>}}


### (80/159) A Geometrical Approach to Evaluate the Adversarial Robustness of Deep Neural Networks (Yang Wang et al., 2023)

{{<citation>}}

Yang Wang, Bo Dong, Ke Xu, Haiyin Piao, Yufei Ding, Baocai Yin, Xin Yang. (2023)  
**A Geometrical Approach to Evaluate the Adversarial Robustness of Deep Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.06468v1)  

---


**ABSTRACT**  
Deep Neural Networks (DNNs) are widely used for computer vision tasks. However, it has been shown that deep models are vulnerable to adversarial attacks, i.e., their performances drop when imperceptible perturbations are made to the original inputs, which may further degrade the following visual tasks or introduce new problems such as data and privacy security. Hence, metrics for evaluating the robustness of deep models against adversarial attacks are desired. However, previous metrics are mainly proposed for evaluating the adversarial robustness of shallow networks on the small-scale datasets. Although the Cross Lipschitz Extreme Value for nEtwork Robustness (CLEVER) metric has been proposed for large-scale datasets (e.g., the ImageNet dataset), it is computationally expensive and its performance relies on a tractable number of samples. In this paper, we propose the Adversarial Converging Time Score (ACTS), an attack-dependent metric that quantifies the adversarial robustness of a DNN on a specific input. Our key observation is that local neighborhoods on a DNN's output surface would have different shapes given different inputs. Hence, given different inputs, it requires different time for converging to an adversarial sample. Based on this geometry meaning, ACTS measures the converging time as an adversarial robustness metric. We validate the effectiveness and generalization of the proposed ACTS metric against different adversarial attacks on the large-scale ImageNet dataset using state-of-the-art deep networks. Extensive experiments show that our ACTS metric is an efficient and effective adversarial metric over the previous CLEVER metric.

{{</citation>}}


### (81/159) Solution for SMART-101 Challenge of ICCV Multi-modal Algorithmic Reasoning Task 2023 (Xiangyu Wu et al., 2023)

{{<citation>}}

Xiangyu Wu, Yang Yang, Shengdong Xu, Yifeng Wu, Qingguo Chen, Jianfeng Lu. (2023)  
**Solution for SMART-101 Challenge of ICCV Multi-modal Algorithmic Reasoning Task 2023**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.06440v1)  

---


**ABSTRACT**  
In this paper, we present our solution to a Multi-modal Algorithmic Reasoning Task: SMART-101 Challenge. Different from the traditional visual question-answering datasets, this challenge evaluates the abstraction, deduction, and generalization abilities of neural networks in solving visuolinguistic puzzles designed specifically for children in the 6-8 age group. We employed a divide-and-conquer approach. At the data level, inspired by the challenge paper, we categorized the whole questions into eight types and utilized the llama-2-chat model to directly generate the type for each question in a zero-shot manner. Additionally, we trained a yolov7 model on the icon45 dataset for object detection and combined it with the OCR method to recognize and locate objects and text within the images. At the model level, we utilized the BLIP-2 model and added eight adapters to the image encoder VIT-G to adaptively extract visual features for different question types. We fed the pre-constructed question templates as input and generated answers using the flan-t5-xxl decoder. Under the puzzle splits configuration, we achieved an accuracy score of 26.5 on the validation set and 24.30 on the private test set.

{{</citation>}}


### (82/159) The Solution for the CVPR2023 NICE Image Captioning Challenge (Xiangyu Wu et al., 2023)

{{<citation>}}

Xiangyu Wu, Yi Gao, Hailiang Zhang, Yang Yang, Weili Guo, Jianfeng Lu. (2023)  
**The Solution for the CVPR2023 NICE Image Captioning Challenge**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Image Captioning  
[Paper Link](http://arxiv.org/abs/2310.06879v1)  

---


**ABSTRACT**  
In this paper, we present our solution to the New frontiers for Zero-shot Image Captioning Challenge. Different from the traditional image captioning datasets, this challenge includes a larger new variety of visual concepts from many domains (such as COVID-19) as well as various image types (photographs, illustrations, graphics). For the data level, we collect external training data from Laion-5B, a large-scale CLIP-filtered image-text dataset. For the model level, we use OFA, a large-scale visual-language pre-training model based on handcrafted templates, to perform the image captioning task. In addition, we introduce contrastive learning to align image-text pairs to learn new visual concepts in the pre-training stage. Then, we propose a similarity-bucket strategy and incorporate this strategy into the template to force the model to generate higher quality and more matching captions. Finally, by retrieval-augmented strategy, we construct a content-rich template, containing the most relevant top-k captions from other image-text pairs, to guide the model in generating semantic-rich captions. Our method ranks first on the leaderboard, achieving 105.17 and 325.72 Cider-Score in the validation and test phase, respectively.

{{</citation>}}


### (83/159) AnoDODE: Anomaly Detection with Diffusion ODE (Xianyao Hu et al., 2023)

{{<citation>}}

Xianyao Hu, Congming Jin. (2023)  
**AnoDODE: Anomaly Detection with Diffusion ODE**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.06420v1)  

---


**ABSTRACT**  
Anomaly detection is the process of identifying atypical data samples that significantly deviate from the majority of the dataset. In the realm of clinical screening and diagnosis, detecting abnormalities in medical images holds great importance. Typically, clinical practice provides access to a vast collection of normal images, while abnormal images are relatively scarce. We hypothesize that abnormal images and their associated features tend to manifest in low-density regions of the data distribution. Following this assumption, we turn to diffusion ODEs for unsupervised anomaly detection, given their tractability and superior performance in density estimation tasks. More precisely, we propose a new anomaly detection method based on diffusion ODEs by estimating the density of features extracted from multi-scale medical images. Our anomaly scoring mechanism depends on computing the negative log-likelihood of features extracted from medical images at different scales, quantified in bits per dimension. Furthermore, we propose a reconstruction-based anomaly localization suitable for our method. Our proposed method not only identifie anomalies but also provides interpretability at both the image and pixel levels. Through experiments on the BraTS2021 medical dataset, our proposed method outperforms existing methods. These results confirm the effectiveness and robustness of our method.

{{</citation>}}


### (84/159) Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling (Huangjie Zheng et al., 2023)

{{<citation>}}

Huangjie Zheng, Zhendong Wang, Jianbo Yuan, Guanghan Ning, Pengcheng He, Quanzeng You, Hongxia Yang, Mingyuan Zhou. (2023)  
**Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, stat-ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.06389v1)  

---


**ABSTRACT**  
Diffusion models excel at generating photo-realistic images but come with significant computational costs in both training and sampling. While various techniques address these computational challenges, a less-explored issue is designing an efficient and adaptable network backbone for iterative refinement. Current options like U-Net and Vision Transformer often rely on resource-intensive deep networks and lack the flexibility needed for generating images at variable resolutions or with a smaller network than used in training. This study introduces LEGO bricks, which seamlessly integrate Local-feature Enrichment and Global-content Orchestration. These bricks can be stacked to create a test-time reconfigurable diffusion backbone, allowing selective skipping of bricks to reduce sampling costs and generate higher-resolution images than the training data. LEGO bricks enrich local regions with an MLP and transform them using a Transformer block while maintaining a consistent full-resolution image across all bricks. Experimental results demonstrate that LEGO bricks enhance training efficiency, expedite convergence, and facilitate variable-resolution image generation while maintaining strong generative performance. Moreover, LEGO significantly reduces sampling time compared to other methods, establishing it as a valuable enhancement for diffusion models.

{{</citation>}}


### (85/159) Filter Pruning For CNN With Enhanced Linear Representation Redundancy (Bojue Wang et al., 2023)

{{<citation>}}

Bojue Wang, Chunmei Ma, Bin Liu, Nianbo Liu, Jinqi Zhu. (2023)  
**Filter Pruning For CNN With Enhanced Linear Representation Redundancy**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2310.06344v1)  

---


**ABSTRACT**  
Structured network pruning excels non-structured methods because they can take advantage of the thriving developed parallel computing techniques. In this paper, we propose a new structured pruning method. Firstly, to create more structured redundancy, we present a data-driven loss function term calculated from the correlation coefficient matrix of different feature maps in the same layer, named CCM-loss. This loss term can encourage the neural network to learn stronger linear representation relations between feature maps during the training from the scratch so that more homogenous parts can be removed later in pruning. CCM-loss provides us with another universal transcendental mathematical tool besides L*-norm regularization, which concentrates on generating zeros, to generate more redundancy but for the different genres. Furthermore, we design a matching channel selection strategy based on principal components analysis to exploit the maximum potential ability of CCM-loss. In our new strategy, we mainly focus on the consistency and integrality of the information flow in the network. Instead of empirically hard-code the retain ratio for each layer, our channel selection strategy can dynamically adjust each layer's retain ratio according to the specific circumstance of a per-trained model to push the prune ratio to the limit. Notably, on the Cifar-10 dataset, our method brings 93.64% accuracy for pruned VGG-16 with only 1.40M parameters and 49.60M FLOPs, the pruned ratios for parameters and FLOPs are 90.6% and 84.2%, respectively. For ResNet-50 trained on the ImageNet dataset, our approach achieves 42.8% and 47.3% storage and computation reductions, respectively, with an accuracy of 76.23%. Our code is available at https://github.com/Bojue-Wang/CCM-LRR.

{{</citation>}}


### (86/159) Precise Payload Delivery via Unmanned Aerial Vehicles: An Approach Using Object Detection Algorithms (Aditya Vadduri et al., 2023)

{{<citation>}}

Aditya Vadduri, Anagh Benjwal, Abhishek Pai, Elkan Quadros, Aniruddh Kammar, Prajwal Uday. (2023)  
**Precise Payload Delivery via Unmanned Aerial Vehicles: An Approach Using Object Detection Algorithms**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.06329v1)  

---


**ABSTRACT**  
Recent years have seen tremendous advancements in the area of autonomous payload delivery via unmanned aerial vehicles, or drones. However, most of these works involve delivering the payload at a predetermined location using its GPS coordinates. By relying on GPS coordinates for navigation, the precision of payload delivery is restricted to the accuracy of the GPS network and the availability and strength of the GPS connection, which may be severely restricted by the weather condition at the time and place of operation. In this work we describe the development of a micro-class UAV and propose a novel navigation method that improves the accuracy of conventional navigation methods by incorporating a deep-learning-based computer vision approach to identify and precisely align the UAV with a target marked at the payload delivery position. This proposed method achieves a 500% increase in average horizontal precision over conventional GPS-based approaches.

{{</citation>}}


### (87/159) Improving Compositional Text-to-image Generation with Large Vision-Language Models (Song Wen et al., 2023)

{{<citation>}}

Song Wen, Guian Fang, Renrui Zhang, Peng Gao, Hao Dong, Dimitris Metaxas. (2023)  
**Improving Compositional Text-to-image Generation with Large Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06311v1)  

---


**ABSTRACT**  
Recent advancements in text-to-image models, particularly diffusion models, have shown significant promise. However, compositional text-to-image models frequently encounter difficulties in generating high-quality images that accurately align with input texts describing multiple objects, variable attributes, and intricate spatial relationships. To address this limitation, we employ large vision-language models (LVLMs) for multi-dimensional assessment of the alignment between generated images and their corresponding input texts. Utilizing this assessment, we fine-tune the diffusion model to enhance its alignment capabilities. During the inference phase, an initial image is produced using the fine-tuned diffusion model. The LVLM is then employed to pinpoint areas of misalignment in the initial image, which are subsequently corrected using the image editing algorithm until no further misalignments are detected by the LVLM. The resultant image is consequently more closely aligned with the input text. Our experimental results validate that the proposed methodology significantly improves text-image alignment in compositional image generation, particularly with respect to object number, attribute binding, spatial relationships, and aesthetic quality.

{{</citation>}}


### (88/159) Tackling Data Bias in MUSIC-AVQA: Crafting a Balanced Dataset for Unbiased Question-Answering (Xiulong Liu et al., 2023)

{{<citation>}}

Xiulong Liu, Zhikang Dong, Peng Zhang. (2023)  
**Tackling Data Bias in MUSIC-AVQA: Crafting a Balanced Dataset for Unbiased Question-Answering**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs-MM, cs-SD, cs.CV, eess-AS  
Keywords: Bias, QA  
[Paper Link](http://arxiv.org/abs/2310.06238v1)  

---


**ABSTRACT**  
In recent years, there has been a growing emphasis on the intersection of audio, vision, and text modalities, driving forward the advancements in multimodal research. However, strong bias that exists in any modality can lead to the model neglecting the others. Consequently, the model's ability to effectively reason across these diverse modalities is compromised, impeding further advancement. In this paper, we meticulously review each question type from the original dataset, selecting those with pronounced answer biases. To counter these biases, we gather complementary videos and questions, ensuring that no answers have outstanding skewed distribution. In particular, for binary questions, we strive to ensure that both answers are almost uniformly spread within each question category. As a result, we construct a new dataset, named MUSIC-AVQA v2.0, which is more challenging and we believe could better foster the progress of AVQA task. Furthermore, we present a novel baseline model that delves deeper into the audio-visual-text interrelation. On MUSIC-AVQA v2.0, this model surpasses all the existing benchmarks, improving accuracy by 2% on MUSIC-AVQA v2.0, setting a new state-of-the-art performance.

{{</citation>}}


### (89/159) Efficient Adaptation of Large Vision Transformer via Adapter Re-Composing (Wei Dong et al., 2023)

{{<citation>}}

Wei Dong, Dawei Yan, Zhijun Lin, Peng Wang. (2023)  
**Efficient Adaptation of Large Vision Transformer via Adapter Re-Composing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.06234v1)  

---


**ABSTRACT**  
The advent of high-capacity pre-trained models has revolutionized problem-solving in computer vision, shifting the focus from training task-specific models to adapting pre-trained models. Consequently, effectively adapting large pre-trained models to downstream tasks in an efficient manner has become a prominent research area. Existing solutions primarily concentrate on designing lightweight adapters and their interaction with pre-trained models, with the goal of minimizing the number of parameters requiring updates. In this study, we propose a novel Adapter Re-Composing (ARC) strategy that addresses efficient pre-trained model adaptation from a fresh perspective. Our approach considers the reusability of adaptation parameters and introduces a parameter-sharing scheme. Specifically, we leverage symmetric down-/up-projections to construct bottleneck operations, which are shared across layers. By learning low-dimensional re-scaling coefficients, we can effectively re-compose layer-adaptive adapters. This parameter-sharing strategy in adapter design allows us to significantly reduce the number of new parameters while maintaining satisfactory performance, thereby offering a promising approach to compress the adaptation cost. We conduct experiments on 24 downstream image classification tasks using various Vision Transformer variants to evaluate our method. The results demonstrate that our approach achieves compelling transfer learning performance with a reduced parameter count. Our code is available at \href{https://github.com/DavidYanAnDe/ARC}{https://github.com/DavidYanAnDe/ARC}.

{{</citation>}}


## cs.NI (2)



### (90/159) Rate Adaptation Aware Positioning for Flying Gateways using Reinforcement Learning (Gabriella Pantaleão et al., 2023)

{{<citation>}}

Gabriella Pantaleão, Rúben Queirós, Hélder Fontes, Rui Campos. (2023)  
**Rate Adaptation Aware Positioning for Flying Gateways using Reinforcement Learning**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.07031v1)  

---


**ABSTRACT**  
With the growing connectivity demands, Unmanned Aerial Vehicles (UAVs) have emerged as a prominent component in the deployment of Next Generation On-demand Wireless Networks. However, current UAV positioning solutions typically neglect the impact of Rate Adaptation (RA) algorithms or simplify its effect by considering ideal and non-implementable RA algorithms. This work proposes the Rate Adaptation aware RL-based Flying Gateway Positioning (RARL) algorithm, a positioning method for Flying Gateways that applies Deep Q-Learning, accounting for the dynamic data rate imposed by the underlying RA algorithm. The RARL algorithm aims to maximize the throughput of the flying wireless links serving one or more Flying Access Points, which in turn serve ground terminals. The performance evaluation of the RARL algorithm demonstrates that it is capable of taking into account the effect of the underlying RA algorithm and achieve the maximum throughput in all analysed static and mobile scenarios.

{{</citation>}}


### (91/159) BC4LLM: Trusted Artificial Intelligence When Blockchain Meets Large Language Models (Haoxiang Luo et al., 2023)

{{<citation>}}

Haoxiang Luo, Jian Luo, Athanasios V. Vasilakos. (2023)  
**BC4LLM: Trusted Artificial Intelligence When Blockchain Meets Large Language Models**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-LG, cs-NI, cs.NI  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06278v1)  

---


**ABSTRACT**  
In recent years, artificial intelligence (AI) and machine learning (ML) are reshaping society's production methods and productivity, and also changing the paradigm of scientific research. Among them, the AI language model represented by ChatGPT has made great progress. Such large language models (LLMs) serve people in the form of AI-generated content (AIGC) and are widely used in consulting, healthcare, and education. However, it is difficult to guarantee the authenticity and reliability of AIGC learning data. In addition, there are also hidden dangers of privacy disclosure in distributed AI training. Moreover, the content generated by LLMs is difficult to identify and trace, and it is difficult to cross-platform mutual recognition. The above information security issues in the coming era of AI powered by LLMs will be infinitely amplified and affect everyone's life. Therefore, we consider empowering LLMs using blockchain technology with superior security features to propose a vision for trusted AI. This paper mainly introduces the motivation and technical route of blockchain for LLM (BC4LLM), including reliable learning corpus, secure training process, and identifiable generated content. Meanwhile, this paper also reviews the potential applications and future challenges, especially in the frontier communication networks field, including network resource allocation, dynamic spectrum sharing, and semantic communication. Based on the above work combined and the prospect of blockchain and LLMs, it is expected to help the early realization of trusted AI and provide guidance for the academic community.

{{</citation>}}


## cs.LG (30)



### (92/159) Neural Relational Inference with Fast Modular Meta-learning (Ferran Alet et al., 2023)

{{<citation>}}

Ferran Alet, Erica Weng, Tomás Lozano Pérez, Leslie Pack Kaelbling. (2023)  
**Neural Relational Inference with Fast Modular Meta-learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.07015v1)  

---


**ABSTRACT**  
\textit{Graph neural networks} (GNNs) are effective models for many dynamical systems consisting of entities and relations. Although most GNN applications assume a single type of entity and relation, many situations involve multiple types of interactions. \textit{Relational inference} is the problem of inferring these interactions and learning the dynamics from observational data. We frame relational inference as a \textit{modular meta-learning} problem, where neural modules are trained to be composed in different ways to solve many tasks. This meta-learning framework allows us to implicitly encode time invariance and infer relations in context of one another rather than independently, which increases inference capacity. Framing inference as the inner-loop optimization of meta-learning leads to a model-based approach that is more data-efficient and capable of estimating the state of entities that we do not observe directly, but whose existence can be inferred from their effect on observed entities. To address the large search space of graph neural network compositions, we meta-learn a \textit{proposal function} that speeds up the inner-loop simulated annealing search within the modular meta-learning algorithm, providing two orders of magnitude increase in the size of problems that can be addressed.

{{</citation>}}


### (93/159) CarDS-Plus ECG Platform: Development and Feasibility Evaluation of a Multiplatform Artificial Intelligence Toolkit for Portable and Wearable Device Electrocardiograms (Sumukh Vasisht Shankar et al., 2023)

{{<citation>}}

Sumukh Vasisht Shankar, Evangelos K Oikonomou, Rohan Khera. (2023)  
**CarDS-Plus ECG Platform: Development and Feasibility Evaluation of a Multiplatform Artificial Intelligence Toolkit for Portable and Wearable Device Electrocardiograms**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.07000v1)  

---


**ABSTRACT**  
In the rapidly evolving landscape of modern healthcare, the integration of wearable & portable technology provides a unique opportunity for personalized health monitoring in the community. Devices like the Apple Watch, FitBit, and AliveCor KardiaMobile have revolutionized the acquisition and processing of intricate health data streams. Amidst the variety of data collected by these gadgets, single-lead electrocardiogram (ECG) recordings have emerged as a crucial source of information for monitoring cardiovascular health. There has been significant advances in artificial intelligence capable of interpreting these 1-lead ECGs, facilitating clinical diagnosis as well as the detection of rare cardiac disorders. This design study describes the development of an innovative multiplatform system aimed at the rapid deployment of AI-based ECG solutions for clinical investigation & care delivery. The study examines design considerations, aligning them with specific applications, develops data flows to maximize efficiency for research & clinical use. This process encompasses the reception of single-lead ECGs from diverse wearable devices, channeling this data into a centralized data lake & facilitating real-time inference through AI models for ECG interpretation. An evaluation of the platform demonstrates a mean duration from acquisition to reporting of results of 33.0 to 35.7 seconds, after a standard 30 second acquisition. There were no substantial differences in acquisition to reporting across two commercially available devices (Apple Watch and KardiaMobile). These results demonstrate the succcessful translation of design principles into a fully integrated & efficient strategy for leveraging 1-lead ECGs across platforms & interpretation by AI-ECG algorithms. Such a platform is critical to translating AI discoveries for wearable and portable ECG devices to clinical impact through rapid deployment.

{{</citation>}}


### (94/159) Flood and Echo: Algorithmic Alignment of GNNs with Distributed Computing (Joël Mathys et al., 2023)

{{<citation>}}

Joël Mathys, Florian Grötschl, Kalyan Varma Nadimpalli, Roger Wattenhofer. (2023)  
**Flood and Echo: Algorithmic Alignment of GNNs with Distributed Computing**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.06970v1)  

---


**ABSTRACT**  
Graph Neural Networks are a natural fit for learning algorithms. They can directly represent tasks through an abstract but versatile graph structure and handle inputs of different sizes. This opens up the possibility for scaling and extrapolation to larger graphs, one of the most important advantages of an algorithm. However, this raises two core questions i) How can we enable nodes to gather the required information in a given graph ($\textit{information exchange}$), even if is far away and ii) How can we design an execution framework which enables this information exchange for extrapolation to larger graph sizes ($\textit{algorithmic alignment for extrapolation}$). We propose a new execution framework that is inspired by the design principles of distributed algorithms: Flood and Echo Net. It propagates messages through the entire graph in a wave like activation pattern, which naturally generalizes to larger instances. Through its sparse but parallel activations it is provably more efficient in terms of message complexity. We study the proposed model and provide both empirical evidence and theoretical insights in terms of its expressiveness, efficiency, information exchange and ability to extrapolate.

{{</citation>}}


### (95/159) Scalable Semantic Non-Markovian Simulation Proxy for Reinforcement Learning (Kaustuv Mukherji et al., 2023)

{{<citation>}}

Kaustuv Mukherji, Devendra Parkar, Lahari Pokala, Dyuman Aditya, Paulo Shakarian, Clark Dorman. (2023)  
**Scalable Semantic Non-Markovian Simulation Proxy for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-LO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.06835v1)  

---


**ABSTRACT**  
Recent advances in reinforcement learning (RL) have shown much promise across a variety of applications. However, issues such as scalability, explainability, and Markovian assumptions limit its applicability in certain domains. We observe that many of these shortcomings emanate from the simulator as opposed to the RL training algorithms themselves. As such, we propose a semantic proxy for simulation based on a temporal extension to annotated logic. In comparison with two high-fidelity simulators, we show up to three orders of magnitude speed-up while preserving the quality of policy learned in addition to showing the ability to model and leverage non-Markovian dynamics and instantaneous actions while providing an explainable trace describing the outcomes of the agent actions.

{{</citation>}}


### (96/159) $f$-Policy Gradients: A General Framework for Goal Conditioned RL using $f$-Divergences (Siddhant Agarwal et al., 2023)

{{<citation>}}

Siddhant Agarwal, Ishan Durugkar, Peter Stone, Amy Zhang. (2023)  
**$f$-Policy Gradients: A General Framework for Goal Conditioned RL using $f$-Divergences**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.06794v1)  

---


**ABSTRACT**  
Goal-Conditioned Reinforcement Learning (RL) problems often have access to sparse rewards where the agent receives a reward signal only when it has achieved the goal, making policy optimization a difficult problem. Several works augment this sparse reward with a learned dense reward function, but this can lead to sub-optimal policies if the reward is misaligned. Moreover, recent works have demonstrated that effective shaping rewards for a particular problem can depend on the underlying learning algorithm. This paper introduces a novel way to encourage exploration called $f$-Policy Gradients, or $f$-PG. $f$-PG minimizes the f-divergence between the agent's state visitation distribution and the goal, which we show can lead to an optimal policy. We derive gradients for various f-divergences to optimize this objective. Our learning paradigm provides dense learning signals for exploration in sparse reward settings. We further introduce an entropy-regularized policy optimization objective, that we call $state$-MaxEnt RL (or $s$-MaxEnt RL) as a special case of our objective. We show that several metric-based shaping rewards like L2 can be used with $s$-MaxEnt RL, providing a common ground to study such metric-based shaping rewards with efficient exploration. We find that $f$-PG has better performance compared to standard policy gradient methods on a challenging gridworld as well as the Point Maze and FetchReach environments. More information on our website https://agarwalsiddhant10.github.io/projects/fpg.html.

{{</citation>}}


### (97/159) Spectral Entry-wise Matrix Estimation for Low-Rank Reinforcement Learning (Stefan Stojanovic et al., 2023)

{{<citation>}}

Stefan Stojanovic, Yassir Jedra, Alexandre Proutiere. (2023)  
**Spectral Entry-wise Matrix Estimation for Low-Rank Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.06793v1)  

---


**ABSTRACT**  
We study matrix estimation problems arising in reinforcement learning (RL) with low-rank structure. In low-rank bandits, the matrix to be recovered specifies the expected arm rewards, and for low-rank Markov Decision Processes (MDPs), it may for example characterize the transition kernel of the MDP. In both cases, each entry of the matrix carries important information, and we seek estimation methods with low entry-wise error. Importantly, these methods further need to accommodate for inherent correlations in the available data (e.g. for MDPs, the data consists of system trajectories). We investigate the performance of simple spectral-based matrix estimation approaches: we show that they efficiently recover the singular subspaces of the matrix and exhibit nearly-minimal entry-wise error. These new results on low-rank matrix estimation make it possible to devise reinforcement learning algorithms that fully exploit the underlying low-rank structure. We provide two examples of such algorithms: a regret minimization algorithm for low-rank bandit problems, and a best policy identification algorithm for reward-free RL in low-rank MDPs. Both algorithms yield state-of-the-art performance guarantees.

{{</citation>}}


### (98/159) A Supervised Embedding and Clustering Anomaly Detection method for classification of Mobile Network Faults (R. Mosayebi et al., 2023)

{{<citation>}}

R. Mosayebi, H. Kia, A. Kianpour Raki. (2023)  
**A Supervised Embedding and Clustering Anomaly Detection method for classification of Mobile Network Faults**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Embedding  
[Paper Link](http://arxiv.org/abs/2310.06779v1)  

---


**ABSTRACT**  
The paper introduces Supervised Embedding and Clustering Anomaly Detection (SEMC-AD), a method designed to efficiently identify faulty alarm logs in a mobile network and alleviate the challenges of manual monitoring caused by the growing volume of alarm logs. SEMC-AD employs a supervised embedding approach based on deep neural networks, utilizing historical alarm logs and their labels to extract numerical representations for each log, effectively addressing the issue of imbalanced classification due to a small proportion of anomalies in the dataset without employing one-hot encoding. The robustness of the embedding is evaluated by plotting the two most significant principle components of the embedded alarm logs, revealing that anomalies form distinct clusters with similar embeddings. Multivariate normal Gaussian clustering is then applied to these components, identifying clusters with a high ratio of anomalies to normal alarms (above 90%) and labeling them as the anomaly group. To classify new alarm logs, we check if their embedded vectors' two most significant principle components fall within the anomaly-labeled clusters. If so, the log is classified as an anomaly. Performance evaluation demonstrates that SEMC-AD outperforms conventional random forest and gradient boosting methods without embedding. SEMC-AD achieves 99% anomaly detection, whereas random forest and XGBoost only detect 86% and 81% of anomalies, respectively. While supervised classification methods may excel in labeled datasets, the results demonstrate that SEMC-AD is more efficient in classifying anomalies in datasets with numerous categorical features, significantly enhancing anomaly detection, reducing operator burden, and improving network maintenance.

{{</citation>}}


### (99/159) Information Content Exploration (Jacob Chmura et al., 2023)

{{<citation>}}

Jacob Chmura, Hasham Burhani, Xiao Qi Shi. (2023)  
**Information Content Exploration**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Network Distillation  
[Paper Link](http://arxiv.org/abs/2310.06777v1)  

---


**ABSTRACT**  
Sparse reward environments are known to be challenging for reinforcement learning agents. In such environments, efficient and scalable exploration is crucial. Exploration is a means by which an agent gains information about the environment. We expand on this topic and propose a new intrinsic reward that systemically quantifies exploratory behavior and promotes state coverage by maximizing the information content of a trajectory taken by an agent. We compare our method to alternative exploration based intrinsic reward techniques, namely Curiosity Driven Learning and Random Network Distillation. We show that our information theoretic reward induces efficient exploration and outperforms in various games, including Montezuma Revenge, a known difficult task for reinforcement learning. Finally, we propose an extension that maximizes information content in a discretely compressed latent space which boosts sample efficiency and generalizes to continuous state spaces.

{{</citation>}}


### (100/159) Zero-Shot Transfer in Imitation Learning (Alvaro Cauderan et al., 2023)

{{<citation>}}

Alvaro Cauderan, Gauthier Boeshertz, Florian Schwarb, Calvin Zhang. (2023)  
**Zero-Shot Transfer in Imitation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.06710v1)  

---


**ABSTRACT**  
We present an algorithm that learns to imitate expert behavior and can transfer to previously unseen domains without retraining. Such an algorithm is extremely relevant in real-world applications such as robotic learning because 1) reward functions are difficult to design, 2) learned policies from one domain are difficult to deploy in another domain and 3) learning directly in the real world is either expensive or unfeasible due to security concerns. To overcome these constraints, we combine recent advances in Deep RL by using an AnnealedVAE to learn a disentangled state representation and imitate an expert by learning a single Q-function which avoids adversarial training. We demonstrate the effectiveness of our method in 3 environments ranging in difficulty and the type of transfer knowledge required.

{{</citation>}}


### (101/159) Domain Generalization by Rejecting Extreme Augmentations (Masih Aminbeidokhti et al., 2023)

{{<citation>}}

Masih Aminbeidokhti, Fidel A. Guerrero Peña, Heitor Rapela Medeiros, Thomas Dubail, Eric Granger, Marco Pedersoli. (2023)  
**Domain Generalization by Rejecting Extreme Augmentations**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.06670v1)  

---


**ABSTRACT**  
Data augmentation is one of the most effective techniques for regularizing deep learning models and improving their recognition performance in a variety of tasks and domains. However, this holds for standard in-domain settings, in which the training and test data follow the same distribution. For the out-of-domain case, where the test data follow a different and unknown distribution, the best recipe for data augmentation is unclear. In this paper, we show that for out-of-domain and domain generalization settings, data augmentation can provide a conspicuous and robust improvement in performance. To do that, we propose a simple training procedure: (i) use uniform sampling on standard data augmentation transformations; (ii) increase the strength transformations to account for the higher data variance expected when working out-of-domain, and (iii) devise a new reward function to reject extreme transformations that can harm the training. With this procedure, our data augmentation scheme achieves a level of accuracy that is comparable to or better than state-of-the-art methods on benchmark domain generalization datasets. Code: \url{https://github.com/Masseeh/DCAug}

{{</citation>}}


### (102/159) Self-Supervised Representation Learning for Online Handwriting Text Classification (Pouya Mehralian et al., 2023)

{{<citation>}}

Pouya Mehralian, Bagher BabaAli, Ashena Gorgan Mohammadi. (2023)  
**Self-Supervised Representation Learning for Online Handwriting Text Classification**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Representation Learning, Self-Supervised, Text Classification  
[Paper Link](http://arxiv.org/abs/2310.06645v1)  

---


**ABSTRACT**  
Self-supervised learning offers an efficient way of extracting rich representations from various types of unlabeled data while avoiding the cost of annotating large-scale datasets. This is achievable by designing a pretext task to form pseudo labels with respect to the modality and domain of the data. Given the evolving applications of online handwritten texts, in this study, we propose the novel Part of Stroke Masking (POSM) as a pretext task for pretraining models to extract informative representations from the online handwriting of individuals in English and Chinese languages, along with two suggested pipelines for fine-tuning the pretrained models. To evaluate the quality of the extracted representations, we use both intrinsic and extrinsic evaluation methods. The pretrained models are fine-tuned to achieve state-of-the-art results in tasks such as writer identification, gender classification, and handedness classification, also highlighting the superiority of utilizing the pretrained models over the models trained from scratch.

{{</citation>}}


### (103/159) iTransformer: Inverted Transformers Are Effective for Time Series Forecasting (Yong Liu et al., 2023)

{{<citation>}}

Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long. (2023)  
**iTransformer: Inverted Transformers Are Effective for Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.06625v1)  

---


**ABSTRACT**  
The recent boom of linear forecasting models questions the ongoing passion for architectural modifications of Transformer-based forecasters. These forecasters leverage Transformers to model the global dependencies over temporal tokens of time series, with each token formed by multiple variates of the same timestamp. However, Transformer is challenged in forecasting series with larger lookback windows due to performance degradation and computation explosion. Besides, the unified embedding for each temporal token fuses multiple variates with potentially unaligned timestamps and distinct physical measurements, which may fail in learning variate-centric representations and result in meaningless attention maps. In this work, we reflect on the competent duties of Transformer components and repurpose the Transformer architecture without any adaptation on the basic components. We propose iTransformer that simply inverts the duties of the attention mechanism and the feed-forward network. Specifically, the time points of individual series are embedded into variate tokens which are utilized by the attention mechanism to capture multivariate correlations; meanwhile, the feed-forward network is applied for each variate token to learn nonlinear representations. The iTransformer model achieves consistent state-of-the-art on several real-world datasets, which further empowers the Transformer family with promoted performance, generalization ability across different variates, and better utilization of arbitrary lookback windows, making it a nice alternative as the fundamental backbone of time series forecasting.

{{</citation>}}


### (104/159) Pi-DUAL: Using Privileged Information to Distinguish Clean from Noisy Labels (Ke Wang et al., 2023)

{{<citation>}}

Ke Wang, Guillermo Ortiz-Jimenez, Rodolphe Jenatton, Mark Collier, Efi Kokiopoulou, Pascal Frossard. (2023)  
**Pi-DUAL: Using Privileged Information to Distinguish Clean from Noisy Labels**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.06600v1)  

---


**ABSTRACT**  
Label noise is a pervasive problem in deep learning that often compromises the generalization performance of trained models. Recently, leveraging privileged information (PI) -- information available only during training but not at test time -- has emerged as an effective approach to mitigate this issue. Yet, existing PI-based methods have failed to consistently outperform their no-PI counterparts in terms of preventing overfitting to label noise. To address this deficiency, we introduce Pi-DUAL, an architecture designed to harness PI to distinguish clean from wrong labels. Pi-DUAL decomposes the output logits into a prediction term, based on conventional input features, and a noise-fitting term influenced solely by PI. A gating mechanism steered by PI adaptively shifts focus between these terms, allowing the model to implicitly separate the learning paths of clean and wrong labels. Empirically, Pi-DUAL achieves significant performance improvements on key PI benchmarks (e.g., +6.8% on ImageNet-PI), establishing a new state-of-the-art test set accuracy. Additionally, Pi-DUAL is a potent method for identifying noisy samples post-training, outperforming other strong methods at this task. Overall, Pi-DUAL is a simple, scalable and practical approach for mitigating the effects of label noise in a variety of real-world scenarios with PI.

{{</citation>}}


### (105/159) XAI for Early Crop Classification (Ayshah Chan et al., 2023)

{{<citation>}}

Ayshah Chan, Maja Schneider, Marco Körner. (2023)  
**XAI for Early Crop Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-AP, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.06574v1)  

---


**ABSTRACT**  
We propose an approach for early crop classification through identifying important timesteps with eXplainable AI (XAI) methods. Our approach consists of training a baseline crop classification model to carry out layer-wise relevance propagation (LRP) so that the salient time step can be identified. We chose a selected number of such important time indices to create the bounding region of the shortest possible classification timeframe. We identified the period 21st April 2019 to 9th August 2019 as having the best trade-off in terms of accuracy and earliness. This timeframe only suffers a 0.75% loss in accuracy as compared to using the full timeseries. We observed that the LRP-derived important timesteps also highlight small details in input values that differentiates between different classes and

{{</citation>}}


### (106/159) Self-Supervised Set Representation Learning for Unsupervised Meta-Learning (Dong Bok Lee et al., 2023)

{{<citation>}}

Dong Bok Lee, Seanie Lee, Joonho Ko, Kenji Kawaguchi, Juho Lee, Sung Ju Hwang. (2023)  
**Self-Supervised Set Representation Learning for Unsupervised Meta-Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.06511v1)  

---


**ABSTRACT**  
Dataset distillation methods have achieved remarkable success in distilling a large dataset into a small set of representative samples. However, they are not designed to produce a distilled dataset that can be effectively used for facilitating self-supervised pre-training. To this end, we propose a novel problem of distilling an unlabeled dataset into a set of small synthetic samples for efficient self-supervised learning (SSL). We first prove that a gradient of synthetic samples with respect to a SSL objective in naive bilevel optimization is \textit{biased} due to the randomness originating from data augmentations or masking. To address this issue, we propose to minimize the mean squared error (MSE) between a model's representations of the synthetic examples and their corresponding learnable target feature representations for the inner objective, which does not introduce any randomness. Our primary motivation is that the model obtained by the proposed inner optimization can mimic the \textit{self-supervised target model}. To achieve this, we also introduce the MSE between representations of the inner model and the self-supervised target model on the original full dataset for outer optimization. Lastly, assuming that a feature extractor is fixed, we only optimize a linear head on top of the feature extractor, which allows us to reduce the computational cost and obtain a closed-form solution of the head with kernel ridge regression. We empirically validate the effectiveness of our method on various applications involving transfer learning.

{{</citation>}}


### (107/159) Runway Sign Classifier: A DAL C Certifiable Machine Learning System (Konstantin Dmitriev et al., 2023)

{{<citation>}}

Konstantin Dmitriev, Johann Schumann, Islam Bostanov, Mostafa Abdelhamid, Florian Holzapfel. (2023)  
**Runway Sign Classifier: A DAL C Certifiable Machine Learning System**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.06506v1)  

---


**ABSTRACT**  
In recent years, the remarkable progress of Machine Learning (ML) technologies within the domain of Artificial Intelligence (AI) systems has presented unprecedented opportunities for the aviation industry, paving the way for further advancements in automation, including the potential for single pilot or fully autonomous operation of large commercial airplanes. However, ML technology faces major incompatibilities with existing airborne certification standards, such as ML model traceability and explainability issues or the inadequacy of traditional coverage metrics. Certification of ML-based airborne systems using current standards is problematic due to these challenges. This paper presents a case study of an airborne system utilizing a Deep Neural Network (DNN) for airport sign detection and classification. Building upon our previous work, which demonstrates compliance with Design Assurance Level (DAL) D, we upgrade the system to meet the more stringent requirements of Design Assurance Level C. To achieve DAL C, we employ an established architectural mitigation technique involving two redundant and dissimilar Deep Neural Networks. The application of novel ML-specific data management techniques further enhances this approach. This work is intended to illustrate how the certification challenges of ML-based systems can be addressed for medium criticality airborne applications.

{{</citation>}}


### (108/159) Understanding the Effects of RLHF on LLM Generalisation and Diversity (Robert Kirk et al., 2023)

{{<citation>}}

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, Roberta Raileanu. (2023)  
**Understanding the Effects of RLHF on LLM Generalisation and Diversity**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: AI, ChatGPT, GPT, LLaMA  
[Paper Link](http://arxiv.org/abs/2310.06452v1)  

---


**ABSTRACT**  
Large language models (LLMs) fine-tuned with reinforcement learning from human feedback (RLHF) have been used in some of the most widely deployed AI models to date, such as OpenAI's ChatGPT, Anthropic's Claude, or Meta's LLaMA-2. While there has been significant work developing these methods, our understanding of the benefits and downsides of each stage in RLHF is still limited. To fill this gap, we present an extensive analysis of how each stage of the process (i.e. supervised fine-tuning (SFT), reward modelling, and RLHF) affects two key properties: out-of-distribution (OOD) generalisation and output diversity. OOD generalisation is crucial given the wide range of real-world scenarios in which these models are being used, while output diversity refers to the model's ability to generate varied outputs and is important for a variety of use cases. We perform our analysis across two base models on both summarisation and instruction following tasks, the latter being highly relevant for current LLM use cases. We find that RLHF generalises better than SFT to new inputs, particularly as the distribution shift between train and test becomes larger. However, RLHF significantly reduces output diversity compared to SFT across a variety of measures, implying a tradeoff in current LLM fine-tuning methods between generalisation and diversity. Our results provide guidance on which fine-tuning method should be used depending on the application, and show that more research is needed to improve the trade-off between generalisation and diversity.

{{</citation>}}


### (109/159) Advective Diffusion Transformers for Topological Generalization in Graph Learning (Qitian Wu et al., 2023)

{{<citation>}}

Qitian Wu, Chenxiao Yang, Kaipeng Zeng, Fan Nie, Michael Bronstein, Junchi Yan. (2023)  
**Advective Diffusion Transformers for Topological Generalization in Graph Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.06417v1)  

---


**ABSTRACT**  
Graph diffusion equations are intimately related to graph neural networks (GNNs) and have recently attracted attention as a principled framework for analyzing GNN dynamics, formalizing their expressive power, and justifying architectural choices. One key open questions in graph learning is the generalization capabilities of GNNs. A major limitation of current approaches hinges on the assumption that the graph topologies in the training and test sets come from the same distribution. In this paper, we make steps towards understanding the generalization of GNNs by exploring how graph diffusion equations extrapolate and generalize in the presence of varying graph topologies. We first show deficiencies in the generalization capability of existing models built upon local diffusion on graphs, stemming from the exponential sensitivity to topology variation. Our subsequent analysis reveals the promise of non-local diffusion, which advocates for feature propagation over fully-connected latent graphs, under the assumption of a specific data-generating condition. In addition to these findings, we propose a novel graph encoder backbone, Advective Diffusion Transformer (ADiT), inspired by advective graph diffusion equations that have a closed-form solution backed up with theoretical guarantees of desired generalization under topological distribution shifts. The new model, functioning as a versatile graph Transformer, demonstrates superior performance across a wide range of graph learning tasks.

{{</citation>}}


### (110/159) Adversarial Robustness in Graph Neural Networks: A Hamiltonian Approach (Kai Zhao et al., 2023)

{{<citation>}}

Kai Zhao, Qiyu Kang, Yang Song, Rui She, Sijie Wang, Wee Peng Tay. (2023)  
**Adversarial Robustness in Graph Neural Networks: A Hamiltonian Approach**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.06396v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) are vulnerable to adversarial perturbations, including those that affect both node features and graph topology. This paper investigates GNNs derived from diverse neural flows, concentrating on their connection to various stability notions such as BIBO stability, Lyapunov stability, structural stability, and conservative stability. We argue that Lyapunov stability, despite its common use, does not necessarily ensure adversarial robustness. Inspired by physics principles, we advocate for the use of conservative Hamiltonian neural flows to construct GNNs that are robust to adversarial attacks. The adversarial robustness of different neural flow GNNs is empirically compared on several benchmark datasets under a variety of adversarial attacks. Extensive numerical experiments demonstrate that GNNs leveraging conservative Hamiltonian flows with Lyapunov stability substantially improve robustness against adversarial perturbations. The implementation code of experiments is available at https://github.com/zknus/NeurIPS-2023-HANG-Robustness.

{{</citation>}}


### (111/159) Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations (Zeming Wei et al., 2023)

{{<citation>}}

Zeming Wei, Yifei Wang, Yisen Wang. (2023)  
**Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06387v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown remarkable success in various tasks, but concerns about their safety and the potential for generating malicious content have emerged. In this paper, we explore the power of In-Context Learning (ICL) in manipulating the alignment ability of LLMs. We find that by providing just few in-context demonstrations without fine-tuning, LLMs can be manipulated to increase or decrease the probability of jailbreaking, i.e. answering malicious prompts. Based on these observations, we propose In-Context Attack (ICA) and In-Context Defense (ICD) methods for jailbreaking and guarding aligned language model purposes. ICA crafts malicious contexts to guide models in generating harmful outputs, while ICD enhances model robustness by demonstrations of rejecting to answer harmful prompts. Our experiments show the effectiveness of ICA and ICD in increasing or reducing the success rate of adversarial jailbreaking attacks. Overall, we shed light on the potential of ICL to influence LLM behavior and provide a new perspective for enhancing the safety and alignment of LLMs.

{{</citation>}}


### (112/159) Initialization Bias of Fourier Neural Operator: Revisiting the Edge of Chaos (Takeshi Koshizuka et al., 2023)

{{<citation>}}

Takeshi Koshizuka, Masahiro Fujisawa, Yusuke Tanaka, Issei Sato. (2023)  
**Initialization Bias of Fourier Neural Operator: Revisiting the Edge of Chaos**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.06379v1)  

---


**ABSTRACT**  
This paper investigates the initialization bias of the Fourier neural operator (FNO). A mean-field theory for FNO is established, analyzing the behavior of the random FNO from an ``edge of chaos'' perspective. We uncover that the forward and backward propagation behaviors exhibit characteristics unique to FNO, induced by mode truncation, while also showcasing similarities to those of densely connected networks. Building upon this observation, we also propose a FNO version of the He initialization scheme to mitigate the negative initialization bias leading to training instability. Experimental results demonstrate the effectiveness of our initialization scheme, enabling stable training of a 32-layer FNO without the need for additional techniques or significant performance degradation.

{{</citation>}}


### (113/159) DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening (Bowen Gao et al., 2023)

{{<citation>}}

Bowen Gao, Bo Qiang, Haichuan Tan, Minsi Ren, Yinjun Jia, Minsi Lu, Jingjing Liu, Weiying Ma, Yanyan Lan. (2023)  
**DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.06367v1)  

---


**ABSTRACT**  
Virtual screening, which identifies potential drugs from vast compound databases to bind with a particular protein pocket, is a critical step in AI-assisted drug discovery. Traditional docking methods are highly time-consuming, and can only work with a restricted search library in real-life applications. Recent supervised learning approaches using scoring functions for binding-affinity prediction, although promising, have not yet surpassed docking methods due to their strong dependency on limited data with reliable binding-affinity labels. In this paper, we propose a novel contrastive learning framework, DrugCLIP, by reformulating virtual screening as a dense retrieval task and employing contrastive learning to align representations of binding protein pockets and molecules from a large quantity of pairwise data without explicit binding-affinity scores. We also introduce a biological-knowledge inspired data augmentation strategy to learn better protein-molecule representations. Extensive experiments show that DrugCLIP significantly outperforms traditional docking and supervised learning methods on diverse virtual screening benchmarks with highly reduced computation time, especially in zero-shot setting.

{{</citation>}}


### (114/159) Predicting Three Types of Freezing of Gait Events Using Deep Learning Models (Wen Tao Mo et al., 2023)

{{<citation>}}

Wen Tao Mo, Jonathan H. Chan. (2023)  
**Predicting Three Types of Freezing of Gait Events Using Deep Learning Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.06322v1)  

---


**ABSTRACT**  
Freezing of gait is a Parkinson's Disease symptom that episodically inflicts a patient with the inability to step or turn while walking. While medical experts have discovered various triggers and alleviating actions for freezing of gait, the underlying causes and prediction models are still being explored today. Current freezing of gait prediction models that utilize machine learning achieve high sensitivity and specificity in freezing of gait predictions based on time-series data; however, these models lack specifications on the type of freezing of gait events. We develop various deep learning models using the transformer encoder architecture plus Bidirectional LSTM layers and different feature sets to predict the three different types of freezing of gait events. The best performing model achieves a score of 0.427 on testing data, which would rank top 5 in Kaggle's Freezing of Gait prediction competition, hosted by THE MICHAEL J. FOX FOUNDATION. However, we also recognize overfitting in training data that could be potentially improved through pseudo labelling on additional data and model architecture simplification.

{{</citation>}}


### (115/159) Discovering Mixtures of Structural Causal Models from Time Series Data (Sumanth Varambally et al., 2023)

{{<citation>}}

Sumanth Varambally, Yi-An Ma, Rose Yu. (2023)  
**Discovering Mixtures of Structural Causal Models from Time Series Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.06312v1)  

---


**ABSTRACT**  
In fields such as finance, climate science, and neuroscience, inferring causal relationships from time series data poses a formidable challenge. While contemporary techniques can handle nonlinear relationships between variables and flexible noise distributions, they rely on the simplifying assumption that data originates from the same underlying causal model. In this work, we relax this assumption and perform causal discovery from time series data originating from mixtures of different causal models. We infer both the underlying structural causal models and the posterior probability for each sample belonging to a specific mixture component. Our approach employs an end-to-end training process that maximizes an evidence-lower bound for data likelihood. Through extensive experimentation on both synthetic and real-world datasets, we demonstrate that our method surpasses state-of-the-art benchmarks in causal discovery tasks, particularly when the data emanates from diverse underlying causal graphs. Theoretically, we prove the identifiability of such a model under some mild assumptions.

{{</citation>}}


### (116/159) Ensemble Active Learning by Contextual Bandits for AI Incubation in Manufacturing (Yingyan Zeng et al., 2023)

{{<citation>}}

Yingyan Zeng, Xiaoyu Chen, Ran Jin. (2023)  
**Ensemble Active Learning by Contextual Bandits for AI Incubation in Manufacturing**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: AI, Active Learning  
[Paper Link](http://arxiv.org/abs/2310.06306v2)  

---


**ABSTRACT**  
It is challenging but important to save annotation efforts in streaming data acquisition to maintain data quality for supervised learning base learners. We propose an ensemble active learning method to actively acquire samples for annotation by contextual bandits, which is will enforce the exploration-exploitation balance and leading to improved AI modeling performance.

{{</citation>}}


### (117/159) MuseChat: A Conversational Music Recommendation System for Videos (Zhikang Dong et al., 2023)

{{<citation>}}

Zhikang Dong, Bin Chen, Xiulong Liu, Pawel Polak, Peng Zhang. (2023)  
**MuseChat: A Conversational Music Recommendation System for Videos**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-IR, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06282v2)  

---


**ABSTRACT**  
We introduce MuseChat, an innovative dialog-based music recommendation system. This unique platform not only offers interactive user engagement but also suggests music tailored for input videos, so that users can refine and personalize their music selections. In contrast, previous systems predominantly emphasized content compatibility, often overlooking the nuances of users' individual preferences. For example, all the datasets only provide basic music-video pairings or such pairings with textual music descriptions. To address this gap, our research offers three contributions. First, we devise a conversation-synthesis method that simulates a two-turn interaction between a user and a recommendation system, which leverages pre-trained music tags and artist information. In this interaction, users submit a video to the system, which then suggests a suitable music piece with a rationale. Afterwards, users communicate their musical preferences, and the system presents a refined music recommendation with reasoning. Second, we introduce a multi-modal recommendation engine that matches music either by aligning it with visual cues from the video or by harmonizing visual information, feedback from previously recommended music, and the user's textual input. Third, we bridge music representations and textual data with a Large Language Model(Vicuna-7B). This alignment equips MuseChat to deliver music recommendations and their underlying reasoning in a manner resembling human communication. Our evaluations show that MuseChat surpasses existing state-of-the-art models in music retrieval tasks and pioneers the integration of the recommendation process within a natural language framework.

{{</citation>}}


### (118/159) A Unified View on Solving Objective Mismatch in Model-Based Reinforcement Learning (Ran Wei et al., 2023)

{{<citation>}}

Ran Wei, Nathan Lambert, Anthony McDonald, Alfredo Garcia, Roberto Calandra. (2023)  
**A Unified View on Solving Objective Mismatch in Model-Based Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.06253v1)  

---


**ABSTRACT**  
Model-based Reinforcement Learning (MBRL) aims to make agents more sample-efficient, adaptive, and explainable by learning an explicit model of the environment. While the capabilities of MBRL agents have significantly improved in recent years, how to best learn the model is still an unresolved question. The majority of MBRL algorithms aim at training the model to make accurate predictions about the environment and subsequently using the model to determine the most rewarding actions. However, recent research has shown that model predictive accuracy is often not correlated with action quality, tracing the root cause to the \emph{objective mismatch} between accurate dynamics model learning and policy optimization of rewards. A number of interrelated solution categories to the objective mismatch problem have emerged as MBRL continues to mature as a research area. In this work, we provide an in-depth survey of these solution categories and propose a taxonomy to foster future research.

{{</citation>}}


### (119/159) Differentially Private Multi-Site Treatment Effect Estimation (Tatsuki Koga et al., 2023)

{{<citation>}}

Tatsuki Koga, Kamalika Chaudhuri, David Page. (2023)  
**Differentially Private Multi-Site Treatment Effect Estimation**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.06237v1)  

---


**ABSTRACT**  
Patient privacy is a major barrier to healthcare AI. For confidentiality reasons, most patient data remains in silo in separate hospitals, preventing the design of data-driven healthcare AI systems that need large volumes of patient data to make effective decisions. A solution to this is collective learning across multiple sites through federated learning with differential privacy. However, literature in this space typically focuses on differentially private statistical estimation and machine learning, which is different from the causal inference-related problems that arise in healthcare. In this work, we take a fresh look at federated learning with a focus on causal inference; specifically, we look at estimating the average treatment effect (ATE), an important task in causal inference for healthcare applications, and provide a federated analytics approach to enable ATE estimation across multiple sites along with differential privacy (DP) guarantees at each site. The main challenge comes from site heterogeneity -- different sites have different sample sizes and privacy budgets. We address this through a class of per-site estimation algorithms that reports the ATE estimate and its variance as a quality measure, and an aggregation algorithm on the server side that minimizes the overall variance of the final ATE estimate. Our experiments on real and synthetic data show that our method reliably aggregates private statistics across sites and provides better privacy-utility tradeoff under site heterogeneity than baselines.

{{</citation>}}


### (120/159) Detecting and Learning Out-of-Distribution Data in the Open world: Algorithm and Theory (Yiyou Sun, 2023)

{{<citation>}}

Yiyou Sun. (2023)  
**Detecting and Learning Out-of-Distribution Data in the Open world: Algorithm and Theory**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.06221v1)  

---


**ABSTRACT**  
This thesis makes considerable contributions to the realm of machine learning, specifically in the context of open-world scenarios where systems face previously unseen data and contexts. Traditional machine learning models are usually trained and tested within a fixed and known set of classes, a condition known as the closed-world setting. While this assumption works in controlled environments, it falls short in real-world applications where new classes or categories of data can emerge dynamically and unexpectedly. To address this, our research investigates two intertwined steps essential for open-world machine learning: Out-of-distribution (OOD) Detection and Open-world Representation Learning (ORL). OOD detection focuses on identifying instances from unknown classes that fall outside the model's training distribution. This process reduces the risk of making overly confident, erroneous predictions about unfamiliar inputs. Moving beyond OOD detection, ORL extends the capabilities of the model to not only detect unknown instances but also learn from and incorporate knowledge about these new classes. By delving into these research problems of open-world learning, this thesis contributes both algorithmic solutions and theoretical foundations, which pave the way for building machine learning models that are not only performant but also reliable in the face of the evolving complexities of the real world.

{{</citation>}}


### (121/159) SUBP: Soft Uniform Block Pruning for 1xN Sparse CNNs Multithreading Acceleration (Jingyang Xiang et al., 2023)

{{<citation>}}

Jingyang Xiang, Siqi Li, Jun Chen, Shipeng Bai, Yukai Ma, Guang Dai, Yong Liu. (2023)  
**SUBP: Soft Uniform Block Pruning for 1xN Sparse CNNs Multithreading Acceleration**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2310.06218v1)  

---


**ABSTRACT**  
The study of sparsity in Convolutional Neural Networks (CNNs) has become widespread to compress and accelerate models in environments with limited resources. By constraining N consecutive weights along the output channel to be group-wise non-zero, the recent network with 1$\times$N sparsity has received tremendous popularity for its three outstanding advantages: 1) A large amount of storage space saving by a \emph{Block Sparse Row} matrix. 2) Excellent performance at a high sparsity. 3) Significant speedups on CPUs with Advanced Vector Extensions. Recent work requires selecting and fine-tuning 1$\times$N sparse weights based on dense pre-trained weights, leading to the problems such as expensive training cost and memory access, sub-optimal model quality, as well as unbalanced workload across threads (different sparsity across output channels). To overcome them, this paper proposes a novel \emph{\textbf{S}oft \textbf{U}niform \textbf{B}lock \textbf{P}runing} (SUBP) approach to train a uniform 1$\times$N sparse structured network from scratch. Specifically, our approach tends to repeatedly allow pruned blocks to regrow to the network based on block angular redundancy and importance sampling in a uniform manner throughout the training process. It not only makes the model less dependent on pre-training, reduces the model redundancy and the risk of pruning the important blocks permanently but also achieves balanced workload. Empirically, on ImageNet, comprehensive experiments across various CNN architectures show that our SUBP consistently outperforms existing 1$\times$N and structured sparsity methods based on pre-trained models or training from scratch. Source codes and models are available at \url{https://github.com/JingyangXiang/SUBP}.

{{</citation>}}


## cs.CR (5)



### (122/159) Sound-skwatter (Did You Mean: Sound-squatter?) AI-powered Generator for Phishing Prevention (Rodolfo Valentim et al., 2023)

{{<citation>}}

Rodolfo Valentim, Idilio Drago, Marco Mellia, Federico Cerutti. (2023)  
**Sound-skwatter (Did You Mean: Sound-squatter?) AI-powered Generator for Phishing Prevention**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.07005v1)  

---


**ABSTRACT**  
Sound-squatting is a phishing attack that tricks users into malicious resources by exploiting similarities in the pronunciation of words. Proactive defense against sound-squatting candidates is complex, and existing solutions rely on manually curated lists of homophones. We here introduce Sound-skwatter, a multi-language AI-based system that generates sound-squatting candidates for proactive defense. Sound-skwatter relies on an innovative multi-modal combination of Transformers Networks and acoustic models to learn sound similarities. We show that Sound-skwatter can automatically list known homophones and thousands of high-quality candidates. In addition, it covers cross-language sound-squatting, i.e., when the reader and the listener speak different languages, supporting any combination of languages. We apply Sound-skwatter to network-centric phishing via squatted domain names. We find ~ 10% of the generated domains exist in the wild, the vast majority unknown to protection solutions. Next, we show attacks on the PyPI package manager, where ~ 17% of the popular packages have at least one existing candidate. We believe Sound-skwatter is a crucial asset to mitigate the sound-squatting phenomenon proactively on the Internet. To increase its impact, we publish an online demo and release our models and code as open source.

{{</citation>}}


### (123/159) LLMs Killed the Script Kiddie: How Agents Supported by Large Language Models Change the Landscape of Network Threat Testing (Stephen Moskal et al., 2023)

{{<citation>}}

Stephen Moskal, Sam Laney, Erik Hemberg, Una-May O'Reilly. (2023)  
**LLMs Killed the Script Kiddie: How Agents Supported by Large Language Models Change the Landscape of Network Threat Testing**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06936v1)  

---


**ABSTRACT**  
In this paper, we explore the potential of Large Language Models (LLMs) to reason about threats, generate information about tools, and automate cyber campaigns. We begin with a manual exploration of LLMs in supporting specific threat-related actions and decisions. We proceed by automating the decision process in a cyber campaign. We present prompt engineering approaches for a plan-act-report loop for one action of a threat campaign and and a prompt chaining design that directs the sequential decision process of a multi-action campaign. We assess the extent of LLM's cyber-specific knowledge w.r.t the short campaign we demonstrate and provide insights into prompt design for eliciting actionable responses. We discuss the potential impact of LLMs on the threat landscape and the ethical considerations of using LLMs for accelerating threat actor capabilities. We report a promising, yet concerning, application of generative AI to cyber threats. However, the LLM's capabilities to deal with more complex networks, sophisticated vulnerabilities, and the sensitivity of prompts are open questions. This research should spur deliberations over the inevitable advancements in LLM-supported cyber adversarial landscape.

{{</citation>}}


### (124/159) Comparing AI Algorithms for Optimizing Elliptic Curve Cryptography Parameters in Third-Party E-Commerce Integrations: A Pre-Quantum Era Analysis (Felipe Tellez et al., 2023)

{{<citation>}}

Felipe Tellez, Jorge Ortiz. (2023)  
**Comparing AI Algorithms for Optimizing Elliptic Curve Cryptography Parameters in Third-Party E-Commerce Integrations: A Pre-Quantum Era Analysis**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.06752v1)  

---


**ABSTRACT**  
This paper presents a comparative analysis between the Genetic Algorithm (GA) and Particle Swarm Optimization (PSO), two vital artificial intelligence algorithms, focusing on optimizing Elliptic Curve Cryptography (ECC) parameters. These encompass the elliptic curve coefficients, prime number, generator point, group order, and cofactor. The study provides insights into which of the bio-inspired algorithms yields better optimization results for ECC configurations, examining performances under the same fitness function. This function incorporates methods to ensure robust ECC parameters, including assessing for singular or anomalous curves and applying Pollard's rho attack and Hasse's theorem for optimization precision. The optimized parameters generated by GA and PSO are tested in a simulated e-commerce environment, contrasting with well-known curves like secp256k1 during the transmission of order messages using Elliptic Curve-Diffie Hellman (ECDH) and Hash-based Message Authentication Code (HMAC). Focusing on traditional computing in the pre-quantum era, this research highlights the efficacy of GA and PSO in ECC optimization, with implications for enhancing cybersecurity in third-party e-commerce integrations. We recommend the immediate consideration of these findings before quantum computing's widespread adoption.

{{</citation>}}


### (125/159) A Semantic Invariant Robust Watermark for Large Language Models (Aiwei Liu et al., 2023)

{{<citation>}}

Aiwei Liu, Leyi Pan, Xuming Hu, Shiao Meng, Lijie Wen. (2023)  
**A Semantic Invariant Robust Watermark for Large Language Models**  

---
Primary Category: cs.CR  
Categories: 68T50, I-2-7, cs-CL, cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06356v1)  

---


**ABSTRACT**  
Watermark algorithms for large language models (LLMs) have achieved extremely high accuracy in detecting text generated by LLMs. Such algorithms typically involve adding extra watermark logits to the LLM's logits at each generation step. However, prior algorithms face a trade-off between attack robustness and security robustness. This is because the watermark logits for a token are determined by a certain number of preceding tokens; a small number leads to low security robustness, while a large number results in insufficient attack robustness. In this work, we propose a semantic invariant watermarking method for LLMs that provides both attack robustness and security robustness. The watermark logits in our work are determined by the semantics of all preceding tokens. Specifically, we utilize another embedding LLM to generate semantic embeddings for all preceding tokens, and then these semantic embeddings are transformed into the watermark logits through our trained watermark model. Subsequent analyses and experiments demonstrated the attack robustness of our method in semantically invariant settings: synonym substitution and text paraphrasing settings. Finally, we also show that our watermark possesses adequate security robustness. Our code and data are available at https://github.com/THU-BPM/Robust_Watermark.

{{</citation>}}


### (126/159) SCAR: Power Side-Channel Analysis at RTL-Level (Amisha Srivastava et al., 2023)

{{<citation>}}

Amisha Srivastava, Sanjay Das, Navnil Choudhury, Rafail Psiakis, Pedro Henrique Silva, Debjit Pal, Kanad Basu. (2023)  
**SCAR: Power Side-Channel Analysis at RTL-Level**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CY, cs.CR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.06257v1)  

---


**ABSTRACT**  
Power side-channel attacks exploit the dynamic power consumption of cryptographic operations to leak sensitive information of encryption hardware. Therefore, it is necessary to conduct power side-channel analysis for assessing the susceptibility of cryptographic systems and mitigating potential risks. Existing power side-channel analysis primarily focuses on post-silicon implementations, which are inflexible in addressing design flaws, leading to costly and time-consuming post-fabrication design re-spins. Hence, pre-silicon power side-channel analysis is required for early detection of vulnerabilities to improve design robustness. In this paper, we introduce SCAR, a novel pre-silicon power side-channel analysis framework based on Graph Neural Networks (GNN). SCAR converts register-transfer level (RTL) designs of encryption hardware into control-data flow graphs and use that to detect the design modules susceptible to side-channel leakage. Furthermore, we incorporate a deep learning-based explainer in SCAR to generate quantifiable and human-accessible explanation of our detection and localization decisions. We have also developed a fortification component as a part of SCAR that uses large-language models (LLM) to automatically generate and insert additional design code at the localized zone to shore up the side-channel leakage. When evaluated on popular encryption algorithms like AES, RSA, and PRESENT, and postquantum cryptography algorithms like Saber and CRYSTALS-Kyber, SCAR, achieves up to 94.49% localization accuracy, 100% precision, and 90.48% recall. Additionally, through explainability analysis, SCAR reduces features for GNN model training by 57% while maintaining comparable accuracy. We believe that SCAR will transform the security-critical hardware design cycle, resulting in faster design closure at a reduced design cost.

{{</citation>}}


## quant-ph (1)



### (127/159) Quantum Shadow Gradient Descent for Quantum Learning (Mohsen Heidari et al., 2023)

{{<citation>}}

Mohsen Heidari, Mobasshir A Naved, Wenbo Xie, Arjun Jacob Grama, Wojciech Szpankowski. (2023)  
**Quantum Shadow Gradient Descent for Quantum Learning**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.06935v1)  

---


**ABSTRACT**  
This paper proposes a new procedure called quantum shadow gradient descent (QSGD) that addresses these key challenges. Our method has the benefits of a one-shot approach, in not requiring any sample duplication while having a convergence rate comparable to the ideal update rule using exact gradient computation. We propose a new technique for generating quantum shadow samples (QSS), which generates quantum shadows as opposed to classical shadows used in existing works. With classical shadows, the computations are typically performed on classical computers and, hence, are prohibitive since the dimension grows exponentially. Our approach resolves this issue by measurements of quantum shadows. As the second main contribution, we study more general non-product ansatz of the form $\exp\{i\sum_j \theta_j A_j\}$ that model variational Hamiltonians. We prove that the gradient can be written in terms of the gradient of single-parameter ansatzes that can be easily measured. Our proof is based on the Suzuki-Trotter approximation; however, our expressions are exact, unlike prior efforts that approximate non-product operators. As a result, existing gradient measurement techniques can be applied to more general VQAs followed by correction terms without any approximation penalty. We provide theoretical proofs, convergence analysis and verify our results through numerical experiments.

{{</citation>}}


## cs.RO (8)



### (128/159) SAILing CAVs: Speed-Adaptive Infrastructure-Linked Connected and Automated Vehicles (Matthew Nice et al., 2023)

{{<citation>}}

Matthew Nice, Matthew Bunting, George Gunter, William Barbour, Jonathan Sprinkle, Dan Work. (2023)  
**SAILing CAVs: Speed-Adaptive Infrastructure-Linked Connected and Automated Vehicles**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.06931v1)  

---


**ABSTRACT**  
This work demonstrates a new capability in roadway control: Speed-adaptive, infrastructure-linked connected and automated vehicles. We develop and deploy a lightly modified vehicle that is able to dynamically adjust the vehicle speed in response to posted variable speed limit messages generated by the infrastructure using LTE connectivity. This work describes the open source hardware and software platform that enables integration between infrastructure-based variable posted speed limits, and existing vehicle platforms for automated control. The vehicle is deployed in heavy morning traffic on I-24 in Nashville, TN. The control vehicle follows the posted variable speed limits, resulting in as much as a 25% reduction in speed variability compared to a human-piloted vehicle in the same traffic stream.

{{</citation>}}


### (129/159) Reinforcement Learning in a Safety-Embedded MDP with Trajectory Optimization (Fan Yang et al., 2023)

{{<citation>}}

Fan Yang, Wenxuan Zhou, Zuxin Liu, Ding Zhao, David Held. (2023)  
**Reinforcement Learning in a Safety-Embedded MDP with Trajectory Optimization**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.06903v1)  

---


**ABSTRACT**  
Safe Reinforcement Learning (RL) plays an important role in applying RL algorithms to safety-critical real-world applications, addressing the trade-off between maximizing rewards and adhering to safety constraints. This work introduces a novel approach that combines RL with trajectory optimization to manage this trade-off effectively. Our approach embeds safety constraints within the action space of a modified Markov Decision Process (MDP). The RL agent produces a sequence of actions that are transformed into safe trajectories by a trajectory optimizer, thereby effectively ensuring safety and increasing training stability. This novel approach excels in its performance on challenging Safety Gym tasks, achieving significantly higher rewards and near-zero safety violations during inference. The method's real-world applicability is demonstrated through a safe and effective deployment in a real robot task of box-pushing around obstacles.

{{</citation>}}


### (130/159) Evaluating Explanation Methods for Vision-and-Language Navigation (Guanqi Chen et al., 2023)

{{<citation>}}

Guanqi Chen, Lei Yang, Guanhua Chen, Jia Pan. (2023)  
**Evaluating Explanation Methods for Vision-and-Language Navigation**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.06654v1)  

---


**ABSTRACT**  
The ability to navigate robots with natural language instructions in an unknown environment is a crucial step for achieving embodied artificial intelligence (AI). With the improving performance of deep neural models proposed in the field of vision-and-language navigation (VLN), it is equally interesting to know what information the models utilize for their decision-making in the navigation tasks. To understand the inner workings of deep neural models, various explanation methods have been developed for promoting explainable AI (XAI). But they are mostly applied to deep neural models for image or text classification tasks and little work has been done in explaining deep neural models for VLN tasks. In this paper, we address these problems by building quantitative benchmarks to evaluate explanation methods for VLN models in terms of faithfulness. We propose a new erasure-based evaluation pipeline to measure the step-wise textual explanation in the sequential decision-making setting. We evaluate several explanation methods for two representative VLN models on two popular VLN datasets and reveal valuable findings through our experiments.

{{</citation>}}


### (131/159) Forgetful Large Language Models: Lessons Learned from Using LLMs in Robot Programming (Juo-Tung Chen et al., 2023)

{{<citation>}}

Juo-Tung Chen, Chien-Ming Huang. (2023)  
**Forgetful Large Language Models: Lessons Learned from Using LLMs in Robot Programming**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06646v1)  

---


**ABSTRACT**  
Large language models offer new ways of empowering people to program robot applications-namely, code generation via prompting. However, the code generated by LLMs is susceptible to errors. This work reports a preliminary exploration that empirically characterizes common errors produced by LLMs in robot programming. We categorize these errors into two phases: interpretation and execution. In this work, we focus on errors in execution and observe that they are caused by LLMs being "forgetful" of key information provided in user prompts. Based on this observation, we propose prompt engineering tactics designed to reduce errors in execution. We then demonstrate the effectiveness of these tactics with three language models: ChatGPT, Bard, and LLaMA-2. Finally, we discuss lessons learned from using LLMs in robot programming and call for the benchmarking of LLM-powered end-user development of robot applications.

{{</citation>}}


### (132/159) SYNLOCO: Synthesizing Central Pattern Generator and Reinforcement Learning for Quadruped Locomotion (Xinyu Zhang et al., 2023)

{{<citation>}}

Xinyu Zhang, Zhiyuan Xiao, Qingrui Zhang, Wei Pan. (2023)  
**SYNLOCO: Synthesizing Central Pattern Generator and Reinforcement Learning for Quadruped Locomotion**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.06606v1)  

---


**ABSTRACT**  
The Central Pattern Generator (CPG) is adept at generating rhythmic gait patterns characterized by consistent timing and adequate foot clearance. Yet, its open-loop configuration often compromises the system's control performance in response to environmental variations. On the other hand, Reinforcement Learning (RL), celebrated for its model-free properties, has gained significant traction in robotics due to its inherent adaptability and robustness. However, initiating traditional RL approaches from the ground up presents computational challenges and a heightened risk of converging to suboptimal local minima. In this paper, we propose an innovative quadruped locomotion framework, SYNLOCO, by synthesizing CPG and RL that can ingeniously integrate the strengths of both methods, enabling the development of a locomotion controller that is both stable and natural. Furthermore, we introduce a set of performance-driven reward metrics that augment the learning of locomotion control. To optimize the learning trajectory of SYNLOCO, a two-phased training strategy is presented. Our empirical evaluation, conducted on a Unitree GO1 robot under varied conditions--including distinct velocities, terrains, and payload capacities--showcases SYNLOCO's ability to produce consistent and clear-footed gaits across diverse scenarios. The developed controller exhibits resilience against substantial parameter variations, underscoring its potential for robust real-world applications.

{{</citation>}}


### (133/159) 3DS-SLAM: A 3D Object Detection based Semantic SLAM towards Dynamic Indoor Environments (Ghanta Sai Krishna et al., 2023)

{{<citation>}}

Ghanta Sai Krishna, Kundrapu Supriya, Sabur Baidya. (2023)  
**3DS-SLAM: A 3D Object Detection based Semantic SLAM towards Dynamic Indoor Environments**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.06385v1)  

---


**ABSTRACT**  
The existence of variable factors within the environment can cause a decline in camera localization accuracy, as it violates the fundamental assumption of a static environment in Simultaneous Localization and Mapping (SLAM) algorithms. Recent semantic SLAM systems towards dynamic environments either rely solely on 2D semantic information, or solely on geometric information, or combine their results in a loosely integrated manner. In this research paper, we introduce 3DS-SLAM, 3D Semantic SLAM, tailored for dynamic scenes with visual 3D object detection. The 3DS-SLAM is a tightly-coupled algorithm resolving both semantic and geometric constraints sequentially. We designed a 3D part-aware hybrid transformer for point cloud-based object detection to identify dynamic objects. Subsequently, we propose a dynamic feature filter based on HDBSCAN clustering to extract objects with significant absolute depth differences. When compared against ORB-SLAM2, 3DS-SLAM exhibits an average improvement of 98.01% across the dynamic sequences of the TUM RGB-D dataset. Furthermore, it surpasses the performance of the other four leading SLAM systems designed for dynamic environments.

{{</citation>}}


### (134/159) Dobby: A Conversational Service Robot Driven by GPT-4 (Carson Stark et al., 2023)

{{<citation>}}

Carson Stark, Bohkyung Chun, Casey Charleston, Varsha Ravi, Luis Pabon, Surya Sunkari, Tarun Mohan, Peter Stone, Justin Hart. (2023)  
**Dobby: A Conversational Service Robot Driven by GPT-4**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.06303v1)  

---


**ABSTRACT**  
This work introduces a robotics platform which embeds a conversational AI agent in an embodied system for natural language understanding and intelligent decision-making for service tasks; integrating task planning and human-like conversation. The agent is derived from a large language model, which has learned from a vast corpus of general knowledge. In addition to generating dialogue, this agent can interface with the physical world by invoking commands on the robot; seamlessly merging communication and behavior. This system is demonstrated in a free-form tour-guide scenario, in an HRI study combining robots with and without conversational AI capabilities. Performance is measured along five dimensions: overall effectiveness, exploration abilities, scrutinization abilities, receptiveness to personification, and adaptability.

{{</citation>}}


### (135/159) Words into Action: Learning Diverse Humanoid Robot Behaviors using Language Guided Iterative Motion Refinement (K. Niranjan Kumar et al., 2023)

{{<citation>}}

K. Niranjan Kumar, Irfan Essa, Sehoon Ha. (2023)  
**Words into Action: Learning Diverse Humanoid Robot Behaviors using Language Guided Iterative Motion Refinement**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06226v1)  

---


**ABSTRACT**  
Humanoid robots are well suited for human habitats due to their morphological similarity, but developing controllers for them is a challenging task that involves multiple sub-problems, such as control, planning and perception. In this paper, we introduce a method to simplify controller design by enabling users to train and fine-tune robot control policies using natural language commands. We first learn a neural network policy that generates behaviors given a natural language command, such as "walk forward", by combining Large Language Models (LLMs), motion retargeting, and motion imitation. Based on the synthesized motion, we iteratively fine-tune by updating the text prompt and querying LLMs to find the best checkpoint associated with the closest motion in history. We validate our approach using a simulated Digit humanoid robot and demonstrate learning of diverse motions, such as walking, hopping, and kicking, without the burden of complex reward engineering. In addition, we show that our iterative refinement enables us to learn 3x times faster than a naive formulation that learns from scratch.

{{</citation>}}


## cs.DC (2)



### (136/159) Distributed Transfer Learning with 4th Gen Intel Xeon Processors (Lakshmi Arunachalam et al., 2023)

{{<citation>}}

Lakshmi Arunachalam, Fahim Mohammad, Vrushabh H. Sanghavi. (2023)  
**Distributed Transfer Learning with 4th Gen Intel Xeon Processors**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-CV, cs-DC, cs-LG, cs.DC  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2310.06916v1)  

---


**ABSTRACT**  
In this paper, we explore how transfer learning, coupled with Intel Xeon, specifically 4th Gen Intel Xeon scalable processor, defies the conventional belief that training is primarily GPU-dependent. We present a case study where we achieved near state-of-the-art accuracy for image classification on a publicly available Image Classification TensorFlow dataset using Intel Advanced Matrix Extensions(AMX) and distributed training with Horovod.

{{</citation>}}


### (137/159) BBCA-CHAIN: One-Message, Low Latency BFT Consensus on a DAG (Dahlia Malkhi et al., 2023)

{{<citation>}}

Dahlia Malkhi, Chrysoula Stathakopoulou, Maofan Yin. (2023)  
**BBCA-CHAIN: One-Message, Low Latency BFT Consensus on a DAG**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.06335v1)  

---


**ABSTRACT**  
This paper presents a partially synchronous BFT consensus protocol powered by BBCA, a lightly modified Byzantine Consistent Broadcast (CBC) primitive. BBCA provides a Complete-Adopt semantic through an added probing interface to allow either aborting the broadcast by correct nodes or exclusively, adopting the message consistently in case of a potential delivery. It does not introduce any extra type of messages or communication cost to CBC.   BBCA is harnessed into BBCA-CHAIN to make direct commits on a chained backbone of a causally ordered graph of blocks, without any additional voting blocks or artificial layering. With the help of Complete-Adopt, the additional knowledge gained from the underlying CBC completely removes the voting latency in popular DAG-based protocols. At the same time, causal ordering allows nodes to propose blocks in parallel and achieve high throughput.   BBCA-CHAIN thus closes up the gap between protocols built by consistent broadcasts (e.g., Bullshark) to those without such an abstraction (e.g., PBFT/HotStuff), emphasizing their shared fundamental principles. Using a Bracha-style CBC as an example, we fully specify BBCA-CHAIN with simplicity, serving as a solid basis for high-performance replication systems (and blockchains).

{{</citation>}}


## cs.SE (7)



### (138/159) A Comparative Study of Transformer-based Neural Text Representation Techniques on Bug Triaging (Atish Kumar Dipongkor et al., 2023)

{{<citation>}}

Atish Kumar Dipongkor, Kevin Moran. (2023)  
**A Comparative Study of Transformer-based Neural Text Representation Techniques on Bug Triaging**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-IR, cs-SE, cs.SE  
Keywords: BERT, Transformer  
[Paper Link](http://arxiv.org/abs/2310.06913v1)  

---


**ABSTRACT**  
Often, the first step in managing bug reports is related to triaging a bug to the appropriate developer who is best suited to understand, localize, and fix the target bug. Additionally, assigning a given bug to a particular part of a software project can help to expedite the fixing process. However, despite the importance of these activities, they are quite challenging, where days can be spent on the manual triaging process. Past studies have attempted to leverage the limited textual data of bug reports to train text classification models that automate this process -- to varying degrees of success. However, the textual representations and machine learning models used in prior work are limited by their expressiveness, often failing to capture nuanced textual patterns that might otherwise aid in the triaging process. Recently, large, transformer-based, pre-trained neural text representation techniques such as BERT have achieved greater performance in several natural language processing tasks. However, the potential for using these techniques to improve upon prior approaches for automated bug triaging is not well studied or understood.   Therefore, in this paper we offer one of the first investigations that fine-tunes transformer-based language models for the task of bug triaging on four open source datasets, spanning a collective 53 years of development history with over 400 developers and over 150 software project components. Our study includes both a quantitative and qualitative analysis of effectiveness. Our findings illustrate that DeBERTa is the most effective technique across the triaging tasks of developer and component assignment, and the measured performance delta is statistically significant compared to other techniques. However, through our qualitative analysis, we also observe that each technique possesses unique abilities best suited to certain types of bug reports.

{{</citation>}}


### (139/159) Benchmarking and Explaining Large Language Model-based Code Generation: A Causality-Centric Approach (Zhenlan Ji et al., 2023)

{{<citation>}}

Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang. (2023)  
**Benchmarking and Explaining Large Language Model-based Code Generation: A Causality-Centric Approach**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06680v1)  

---


**ABSTRACT**  
While code generation has been widely used in various software development scenarios, the quality of the generated code is not guaranteed. This has been a particular concern in the era of large language models (LLMs)- based code generation, where LLMs, deemed a complex and powerful black-box model, is instructed by a high-level natural language specification, namely a prompt, to generate code. Nevertheless, effectively evaluating and explaining the code generation capability of LLMs is inherently challenging, given the complexity of LLMs and the lack of transparency.   Inspired by the recent progress in causality analysis and its application in software engineering, this paper launches a causality analysis-based approach to systematically analyze the causal relations between the LLM input prompts and the generated code. To handle various technical challenges in this study, we first propose a novel causal graph-based representation of the prompt and the generated code, which is established over the fine-grained, human-understandable concepts in the input prompts. The formed causal graph is then used to identify the causal relations between the prompt and the derived code. We illustrate the insights that our framework can provide by studying over 3 popular LLMs with over 12 prompt adjustment strategies. The results of these studies illustrate the potential of our technique to provide insights into LLM effectiveness, and aid end-users in understanding predictions. Additionally, we demonstrate that our approach provides actionable insights to improve the quality of the LLM-generated code by properly calibrating the prompt.

{{</citation>}}


### (140/159) Refining Decompiled C Code with Large Language Models (Wai Kin Wong et al., 2023)

{{<citation>}}

Wai Kin Wong, Huaijin Wang, Zongjie Li, Zhibo Liu, Shuai Wang, Qiyi Tang, Sen Nie, Shi Wu. (2023)  
**Refining Decompiled C Code with Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06530v1)  

---


**ABSTRACT**  
A C decompiler converts an executable into source code. The recovered C source code, once re-compiled, is expected to produce an executable with the same functionality as the original executable. With over twenty years of development, C decompilers have been widely used in production to support reverse engineering applications. Despite the prosperous development of C decompilers, it is widely acknowledged that decompiler outputs are mainly used for human consumption, and are not suitable for automatic recompilation. Often, a substantial amount of manual effort is required to fix the decompiler outputs before they can be recompiled and executed properly.   This paper is motived by the recent success of large language models (LLMs) in comprehending dense corpus of natural language. To alleviate the tedious, costly and often error-prone manual effort in fixing decompiler outputs, we investigate the feasibility of using LLMs to augment decompiler outputs, thus delivering recompilable decompilation. Note that different from previous efforts that focus on augmenting decompiler outputs with higher readability (e.g., recovering type/variable names), we focus on augmenting decompiler outputs with recompilability, meaning to generate code that can be recompiled into an executable with the same functionality as the original executable.   We conduct a pilot study to characterize the obstacles in recompiling the outputs of the de facto commercial C decompiler -- IDA-Pro. We then propose a two-step, hybrid approach to augmenting decompiler outputs with LLMs. We evaluate our approach on a set of popular C test cases, and show that our approach can deliver a high recompilation success rate to over 75% with moderate effort, whereas none of the IDA-Pro's original outputs can be recompiled. We conclude with a discussion on the limitations of our approach and promising future research directions.

{{</citation>}}


### (141/159) Retromorphic Testing: A New Approach to the Test Oracle Problem (Boxi Yu et al., 2023)

{{<citation>}}

Boxi Yu, Qiuyang Mang, Qingshuo Guo, Pinjia He. (2023)  
**Retromorphic Testing: A New Approach to the Test Oracle Problem**  

---
Primary Category: cs.SE  
Categories: D-3-0; I-2-7; I-4-0, cs-AI, cs-CL, cs-CV, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.06433v1)  

---


**ABSTRACT**  
A test oracle serves as a criterion or mechanism to assess the correspondence between software output and the anticipated behavior for a given input set. In automated testing, black-box techniques, known for their non-intrusive nature in test oracle construction, are widely used, including notable methodologies like differential testing and metamorphic testing. Inspired by the mathematical concept of inverse function, we present Retromorphic Testing, a novel black-box testing methodology. It leverages an auxiliary program in conjunction with the program under test, which establishes a dual-program structure consisting of a forward program and a backward program. The input data is first processed by the forward program and then its program output is reversed to its original input format using the backward program. In particular, the auxiliary program can operate as either the forward or backward program, leading to different testing modes. The process concludes by examining the relationship between the initial input and the transformed output within the input domain. For example, to test the implementation of the sine function $\sin(x)$, we can employ its inverse function, $\arcsin(x)$, and validate the equation $x = \sin(\arcsin(x)+2k\pi), \forall k \in \mathbb{Z}$. In addition to the high-level concept of Retromorphic Testing, this paper presents its three testing modes with illustrative use cases across diverse programs, including algorithms, traditional software, and AI applications.

{{</citation>}}


### (142/159) Automatic Generation of Test Cases based on Bug Reports: a Feasibility Study with Large Language Models (Laura Plein et al., 2023)

{{<citation>}}

Laura Plein, Wendkûuni C. Ouédraogo, Jacques Klein, Tegawendé F. Bissyandé. (2023)  
**Automatic Generation of Test Cases based on Bug Reports: a Feasibility Study with Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06320v1)  

---


**ABSTRACT**  
Software testing is a core discipline in software engineering where a large array of research results has been produced, notably in the area of automatic test generation. Because existing approaches produce test cases that either can be qualified as simple (e.g. unit tests) or that require precise specifications, most testing procedures still rely on test cases written by humans to form test suites. Such test suites, however, are incomplete: they only cover parts of the project or they are produced after the bug is fixed. Yet, several research challenges, such as automatic program repair, and practitioner processes, build on the assumption that available test suites are sufficient. There is thus a need to break existing barriers in automatic test case generation. While prior work largely focused on random unit testing inputs, we propose to consider generating test cases that realistically represent complex user execution scenarios, which reveal buggy behaviour. Such scenarios are informally described in bug reports, which should therefore be considered as natural inputs for specifying bug-triggering test cases. In this work, we investigate the feasibility of performing this generation by leveraging large language models (LLMs) and using bug reports as inputs. Our experiments include the use of ChatGPT, as an online service, as well as CodeGPT, a code-related pre-trained LLM that was fine-tuned for our task. Overall, we experimentally show that bug reports associated to up to 50% of Defects4J bugs can prompt ChatGPT to generate an executable test case. We show that even new bug reports can indeed be used as input for generating executable test cases. Finally, we report experimental results which confirm that LLM-generated test cases are immediately useful in software engineering tasks such as fault localization as well as patch validation in automated program repair.

{{</citation>}}


### (143/159) Can LLMs Demystify Bug Reports? (Laura Plein et al., 2023)

{{<citation>}}

Laura Plein, Tegawendé F. Bissyandé. (2023)  
**Can LLMs Demystify Bug Reports?**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.06310v1)  

---


**ABSTRACT**  
Bugs are notoriously challenging: they slow down software users and result in time-consuming investigations for developers. These challenges are exacerbated when bugs must be reported in natural language by users. Indeed, we lack reliable tools to automatically address reported bugs (i.e., enabling their analysis, reproduction, and bug fixing). With the recent promises created by LLMs such as ChatGPT for various tasks, including in software engineering, we ask ourselves: What if ChatGPT could understand bug reports and reproduce them? This question will be the main focus of this study. To evaluate whether ChatGPT is capable of catching the semantics of bug reports, we used the popular Defects4J benchmark with its bug reports. Our study has shown that ChatGPT was able to demystify and reproduce 50% of the reported bugs. ChatGPT being able to automatically address half of the reported bugs shows promising potential in the direction of applying machine learning to address bugs with only a human-in-the-loop to report the bug.

{{</citation>}}


### (144/159) CodeFuse-13B: A Pretrained Multi-lingual Code Large Language Model (Peng Di et al., 2023)

{{<citation>}}

Peng Di, Jianguo Li, Hang Yu, Wei Jiang, Wenting Cai, Yang Cao, Chaoyu Chen, Dajun Chen, Hongwei Chen, Liang Chen, Gang Fan, Jie Gong, Zi Gong, Wen Hu, Tingting Guo, Zhichao Lei, Ting Li, Zheng Li, Ming Liang, Cong Liao, Bingchang Liu, Jiachen Liu, Zhiwei Liu, Shaojun Lu, Min Shen, Guangpei Wang, Huan Wang, Zhi Wang, Zhaogui Xu, Jiawei Yang, Qing Ye, Gehao Zhang, Yu Zhang, Zelin Zhao, Xunjin Zheng, Hailian Zhou, Lifu Zhu, Xianying Zhu. (2023)  
**CodeFuse-13B: A Pretrained Multi-lingual Code Large Language Model**  

---
Primary Category: cs.SE  
Categories: 68T01, 68N01, I-2-5; D-3-2; D-2-0, cs-AI, cs-LG, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06266v1)  

---


**ABSTRACT**  
Code Large Language Models (Code LLMs) have gained significant attention in the industry due to their wide applications in the full lifecycle of software engineering. However, the effectiveness of existing models in understanding non-English inputs for multi-lingual code-related tasks is still far from well studied. This paper introduces CodeFuse-13B, an open-sourced pre-trained code LLM. It is specifically designed for code-related tasks with both English and Chinese prompts and supports over 40 programming languages. CodeFuse achieves its effectiveness by utilizing a high quality pre-training dataset that is carefully filtered by program analyzers and optimized during the training process. Extensive experiments are conducted using real-world usage scenarios, the industry-standard benchmark HumanEval-x, and the specially designed CodeFuseEval for Chinese prompts. To assess the effectiveness of CodeFuse, we actively collected valuable human feedback from the AntGroup's software development process where CodeFuse has been successfully deployed. The results demonstrate that CodeFuse-13B achieves a HumanEval pass@1 score of 37.10%, positioning it as one of the top multi-lingual code LLMs with similar parameter sizes. In practical scenarios, such as code generation, code translation, code comments, and testcase generation, CodeFuse performs better than other models when confronted with Chinese prompts.

{{</citation>}}


## cs.CE (1)



### (145/159) A quantum annealing-sequential quadratic programming assisted finite element simulation for non-linear and history-dependent mechanical problems (Van-Dung Nguyen et al., 2023)

{{<citation>}}

Van-Dung Nguyen, Francoise Remacle, Ludovic Noels. (2023)  
**A quantum annealing-sequential quadratic programming assisted finite element simulation for non-linear and history-dependent mechanical problems**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.06911v1)  

---


**ABSTRACT**  
We propose a framework to solve non-linear and history-dependent mechanical problems based on a hybrid classical computer-quantum annealer approach. Quantum Computers are anticipated to solve particular operations exponentially faster. The available possible operations are however not as versatile as with a classical computer. However, quantum annealers (QAs) is well suited to evaluate the minimum state of a Hamiltonian quadratic potential. Therefore, we reformulate the elasto-plastic finite element problem as a double minimisation process framed at the structural scale using the variational updates formulation. In order to comply with the expected quadratic nature of the Hamiltonian, the resulting non-linear minimisation problems are iteratively solved with the suggested Quantum Annealing-assisted Sequential Quadratic Programming (QA-SQP): a sequence of minimising quadratic problems is performed by approximating the objective function by a quadratic Taylor's series. Each quadratic minimisation problem of continuous variables is then transformed into a binary quadratic problem. This binary quadratic minimisation problem can be solved on quantum annealing hardware such as the D-Wave system. The applicability of the proposed framework is demonstrated with one and two-dimensional elasto-plastic numerical benchmarks. The current work provides a pathway of performing general non-linear finite element simulations assisted by quantum computing.

{{</citation>}}


## cs.CY (4)



### (146/159) How Knowledge Workers Think Generative AI Will (Not) Transform Their Industries (Allison Woodruff et al., 2023)

{{<citation>}}

Allison Woodruff, Renee Shelby, Patrick Gage Kelley, Steven Rousso-Schindler, Jamila Smith-Loud, Lauren Wilcox. (2023)  
**How Knowledge Workers Think Generative AI Will (Not) Transform Their Industries**  

---
Primary Category: cs.CY  
Categories: K-4-1; K-4-2; K-4-3, cs-CY, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.06778v1)  

---


**ABSTRACT**  
Generative AI is expected to have transformative effects in multiple knowledge industries. To better understand how knowledge workers expect generative AI may affect their industries in the future, we conducted participatory research workshops for seven different industries, with a total of 54 participants across three US cities. We describe participants' expectations of generative AI's impact, including a dominant narrative that cut across the groups' discourse: participants largely envision generative AI as a tool to perform menial work, under human review. Participants do not generally anticipate the disruptive changes to knowledge industries currently projected in common media and academic narratives. Participants do however envision generative AI may amplify four social forces currently shaping their industries: deskilling, dehumanization, disconnection, and disinformation. We describe these forces, and then we provide additional detail regarding attitudes in specific knowledge industries. We conclude with a discussion of implications and research challenges for the HCI community.

{{</citation>}}


### (147/159) Gender, Age, and Technology Education Influence the Adoption and Appropriation of LLMs (Fiona Draxler et al., 2023)

{{<citation>}}

Fiona Draxler, Daniel Buschek, Mikke Tavast, Perttu Hämäläinen, Albrecht Schmidt, Juhi Kulshrestha, Robin Welsch. (2023)  
**Gender, Age, and Technology Education Influence the Adoption and Appropriation of LLMs**  

---
Primary Category: cs.CY  
Categories: H-1-2; I-2-7, cs-CY, cs-HC, cs.CY  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06556v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) such as ChatGPT have become increasingly integrated into critical activities of daily life, raising concerns about equitable access and utilization across diverse demographics. This study investigates the usage of LLMs among 1,500 representative US citizens. Remarkably, 42% of participants reported utilizing an LLM. Our findings reveal a gender gap in LLM technology adoption (more male users than female users) with complex interaction patterns regarding age. Technology-related education eliminates the gender gap in our sample. Moreover, expert users are more likely than novices to list professional tasks as typical application scenarios, suggesting discrepancies in effective usage at the workplace. These results underscore the importance of providing education in artificial intelligence in our technology-driven society to promote equitable access to and benefits from LLMs. We urge for both international replication beyond the US and longitudinal observation of adoption.

{{</citation>}}


### (148/159) Anticipating Impacts: Using Large-Scale Scenario Writing to Explore Diverse Implications of Generative AI in the News Environment (Kimon Kieslich et al., 2023)

{{<citation>}}

Kimon Kieslich, Nicholas Diakopoulos, Natali Helberger. (2023)  
**Anticipating Impacts: Using Large-Scale Scenario Writing to Explore Diverse Implications of Generative AI in the News Environment**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-HC, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.06361v1)  

---


**ABSTRACT**  
The tremendous rise of generative AI has reached every part of society - including the news environment. There are many concerns about the individual and societal impact of the increasing use of generative AI, including issues such as disinformation and misinformation, discrimination, and the promotion of social tensions. However, research on anticipating the impact of generative AI is still in its infancy and mostly limited to the views of technology developers and/or researchers. In this paper, we aim to broaden the perspective and capture the expectations of three stakeholder groups (news consumers; technology developers; content creators) about the potential negative impacts of generative AI, as well as mitigation strategies to address these. Methodologically, we apply scenario writing and use participatory foresight in the context of a survey (n=119) to delve into cognitively diverse imaginations of the future. We qualitatively analyze the scenarios using thematic analysis to systematically map potential impacts of generative AI on the news environment, potential mitigation strategies, and the role of stakeholders in causing and mitigating these impacts. In addition, we measure respondents' opinions on a specific mitigation strategy, namely transparency obligations as suggested in Article 52 of the draft EU AI Act. We compare the results across different stakeholder groups and elaborate on the (non-) presence of different expected impacts across these groups. We conclude by discussing the usefulness of scenario-writing and participatory foresight as a toolbox for generative AI impact assessment.

{{</citation>}}


### (149/159) The AI Incident Database as an Educational Tool to Raise Awareness of AI Harms: A Classroom Exploration of Efficacy, Limitations, & Future Improvements (Michael Feffer et al., 2023)

{{<citation>}}

Michael Feffer, Nikolas Martelaro, Hoda Heidari. (2023)  
**The AI Incident Database as an Educational Tool to Raise Awareness of AI Harms: A Classroom Exploration of Efficacy, Limitations, & Future Improvements**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-HC, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.06269v1)  

---


**ABSTRACT**  
Prior work has established the importance of integrating AI ethics topics into computer and data sciences curricula. We provide evidence suggesting that one of the critical objectives of AI Ethics education must be to raise awareness of AI harms. While there are various sources to learn about such harms, The AI Incident Database (AIID) is one of the few attempts at offering a relatively comprehensive database indexing prior instances of harms or near harms stemming from the deployment of AI technologies in the real world. This study assesses the effectiveness of AIID as an educational tool to raise awareness regarding the prevalence and severity of AI harms in socially high-stakes domains. We present findings obtained through a classroom study conducted at an R1 institution as part of a course focused on the societal and ethical considerations around AI and ML. Our qualitative findings characterize students' initial perceptions of core topics in AI ethics and their desire to close the educational gap between their technical skills and their ability to think systematically about ethical and societal aspects of their work. We find that interacting with the database helps students better understand the magnitude and severity of AI harms and instills in them a sense of urgency around (a) designing functional and safe AI and (b) strengthening governance and accountability mechanisms. Finally, we compile students' feedback about the tool and our class activity into actionable recommendations for the database development team and the broader community to improve awareness of AI harms in AI ethics education.

{{</citation>}}


## math.OC (1)



### (150/159) Near-Optimality of Finite-Memory Codes and Reinforcement Learning for Zero-Delay Coding of Markov Sources (Liam Cregg et al., 2023)

{{<citation>}}

Liam Cregg, Fady Alajaji, Serdar Yuksel. (2023)  
**Near-Optimality of Finite-Memory Codes and Reinforcement Learning for Zero-Delay Coding of Markov Sources**  

---
Primary Category: math.OC  
Categories: cs-IT, math-IT, math-OC, math.OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.06742v1)  

---


**ABSTRACT**  
We study the problem of zero-delay coding of a Markov source over a noisy channel with feedback. We first formulate the problem as a Markov decision process (MDP) where the state is a previous belief term along with a finite memory of channel outputs and quantizers. We then approximate this state by marginalizing over all possible beliefs, so that our policies only use the finite-memory term to encode the source. Under an appropriate notion of predictor stability, we show that such policies are near-optimal for the zero-delay coding problem as the memory length increases. We also give sufficient conditions for predictor stability to hold, and propose a reinforcement learning algorithm to compute near-optimal finite-memory policies. These theoretical results are supported by simulations.

{{</citation>}}


## cs.ET (1)



### (151/159) Machine Learning Quantum Systems with Magnetic p-bits (Shuvro Chowdhury et al., 2023)

{{<citation>}}

Shuvro Chowdhury, Kerem Y. Camsari. (2023)  
**Machine Learning Quantum Systems with Magnetic p-bits**  

---
Primary Category: cs.ET  
Categories: cs-ET, cs-LG, cs-NE, cs.ET, quant-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.06679v1)  

---


**ABSTRACT**  
The slowing down of Moore's Law has led to a crisis as the computing workloads of Artificial Intelligence (AI) algorithms continue skyrocketing. There is an urgent need for scalable and energy-efficient hardware catering to the unique requirements of AI algorithms and applications. In this environment, probabilistic computing with p-bits emerged as a scalable, domain-specific, and energy-efficient computing paradigm, particularly useful for probabilistic applications and algorithms. In particular, spintronic devices such as stochastic magnetic tunnel junctions (sMTJ) show great promise in designing integrated p-computers. Here, we examine how a scalable probabilistic computer with such magnetic p-bits can be useful for an emerging field combining machine learning and quantum physics.

{{</citation>}}


## eess.SY (1)



### (152/159) A Parallelized, Adam-Based Solver for Reserve and Security Constrained AC Unit Commitment (Samuel Chevalier, 2023)

{{<citation>}}

Samuel Chevalier. (2023)  
**A Parallelized, Adam-Based Solver for Reserve and Security Constrained AC Unit Commitment**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: GPT, Security  
[Paper Link](http://arxiv.org/abs/2310.06650v1)  

---


**ABSTRACT**  
Power system optimization problems which include the nonlinear AC power flow equations require powerful and robust numerical solution algorithms. Within this sub-field of nonlinear optimization, interior point methods have come to dominate the solver landscape. Over the last decade, however, a number of efficient numerical optimizers have emerged from the field of Machine Learning (ML). One algorithm in particular, Adam, has become the optimizer-of-choice for a massive percentage of ML training problems (including, e.g., the training of GPT-3), solving some of the largest unconstrained optimization problems ever conceived of. Inspired by such progress, this paper designs a parallelized Adam-based numerical solver to overcome one of the most challenging power system optimization problems: security and reserve constrained AC Unit Commitment. The resulting solver, termed quasiGrad, recently competed in the third ARPA-E Grid Optimization (GO3) competition. In the day-ahead market clearing category (with systems ranging from 3 to 23,643 buses over 48 time periods), quasiGrad's aggregated market surplus scores were within 5% of the winningest market surplus scores. The quasiGrad solver is now released as an open-source Julia package: quasiGrad.jl. The internal gradient-based solver (Adam) can easily be substituted for other ML-inspired solvers (e.g., AdaGrad, AdaDelta, RMSProp, etc.). Test results from large experiments are provided.

{{</citation>}}


## cs.DL (1)



### (153/159) Toward Semantic Publishing in Non-Invasive Brain Stimulation: A Comprehensive Analysis of rTMS Studies (Swathi Anil et al., 2023)

{{<citation>}}

Swathi Anil, Jennifer D'Souza. (2023)  
**Toward Semantic Publishing in Non-Invasive Brain Stimulation: A Comprehensive Analysis of rTMS Studies**  

---
Primary Category: cs.DL  
Categories: cs-CL, cs-DL, cs-IT, cs.DL, math-IT  
Keywords: AI, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.06517v1)  

---


**ABSTRACT**  
Noninvasive brain stimulation (NIBS) encompasses transcranial stimulation techniques that can influence brain excitability. These techniques have the potential to treat conditions like depression, anxiety, and chronic pain, and to provide insights into brain function. However, a lack of standardized reporting practices limits its reproducibility and full clinical potential. This paper aims to foster interinterdisciplinarity toward adopting Computer Science Semantic reporting methods for the standardized documentation of Neuroscience NIBS studies making them explicitly Findable, Accessible, Interoperable, and Reusable (FAIR).   In a large-scale systematic review of 600 repetitive transcranial magnetic stimulation (rTMS), a subarea of NIBS, dosages, we describe key properties that allow for structured descriptions and comparisons of the studies. This paper showcases the semantic publishing of NIBS in the ecosphere of knowledge-graph-based next-generation scholarly digital libraries. Specifically, the FAIR Semantic Web resource(s)-based publishing paradigm is implemented for the 600 reviewed rTMS studies in the Open Research Knowledge Graph.

{{</citation>}}


## cs.IR (2)



### (154/159) A Multi-facet Paradigm to Bridge Large Language Model and Recommendation (Xinyu Lin et al., 2023)

{{<citation>}}

Xinyu Lin, Wenjie Wang, Yongqi Li, Fuli Feng, See-Kiong Ng, Tat-Seng Chua. (2023)  
**A Multi-facet Paradigm to Bridge Large Language Model and Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06491v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have garnered considerable attention in recommender systems. To achieve LLM-based recommendation, item indexing and generation grounding are two essential steps, bridging between recommendation items and natural language. Item indexing assigns a unique identifier to represent each item in natural language, and generation grounding grounds the generated token sequences to in-corpus items. However, previous works suffer from inherent limitations in the two steps. For item indexing, existing ID-based identifiers (e.g., numeric IDs) and description-based identifiers (e.g., titles) often compromise semantic richness or uniqueness. Moreover, generation grounding might inadvertently produce out-of-corpus identifiers. Worse still, autoregressive generation heavily relies on the initial token's quality. To combat these issues, we propose a novel multi-facet paradigm, namely TransRec, to bridge the LLMs to recommendation. Specifically, TransRec employs multi-facet identifiers that incorporate ID, title, and attribute, achieving both distinctiveness and semantics. Additionally, we introduce a specialized data structure for TransRec to guarantee the in-corpus identifier generation and adopt substring indexing to encourage LLMs to generate from any position. We implement TransRec on two backbone LLMs, i.e., BART-large and LLaMA-7B. Empirical results on three real-world datasets under diverse settings (e.g., full training and few-shot training with warm- and cold-start testings) attest to the superiority of TransRec.

{{</citation>}}


### (155/159) Query-dominant User Interest Network for Large-Scale Search Ranking (Tong Guo et al., 2023)

{{<citation>}}

Tong Guo, Xuanping Li, Haitao Yang, Xiao Liang, Yong Yuan, Jingyou Hou, Bingqing Ke, Chao Zhang, junlin He, Shunyu Zhang, Enyun Yu, Wenwu. (2023)  
**Query-dominant User Interest Network for Large-Scale Search Ranking**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.06444v1)  

---


**ABSTRACT**  
Historical behaviors have shown great effect and potential in various prediction tasks, including recommendation and information retrieval. The overall historical behaviors are various but noisy while search behaviors are always sparse. Most existing approaches in personalized search ranking adopt the sparse search behaviors to learn representation with bottleneck, which do not sufficiently exploit the crucial long-term interest. In fact, there is no doubt that user long-term interest is various but noisy for instant search, and how to exploit it well still remains an open problem.   To tackle this problem, in this work, we propose a novel model named Query-dominant user Interest Network (QIN), including two cascade units to filter the raw user behaviors and reweigh the behavior subsequences. Specifically, we propose a relevance search unit (RSU), which aims to search a subsequence relevant to the query first and then search the sub-subsequences relevant to the target item. These items are then fed into an attention unit called Fused Attention Unit (FAU). It should be able to calculate attention scores from the ID field and attribute field separately, and then adaptively fuse the item embedding and content embedding based on the user engagement of past period. Extensive experiments and ablation studies on real-world datasets demonstrate the superiority of our model over state-of-the-art methods. The QIN now has been successfully deployed on Kuaishou search, an online video search platform, and obtained 7.6% improvement on CTR.

{{</citation>}}


## cs.MM (1)



### (156/159) Encoder-Decoder-Based Intra-Frame Block Partitioning Decision (Yucheng Jiang et al., 2023)

{{<citation>}}

Yucheng Jiang, Han Peng, Yan Song, Jie Yu, Peng Zhang, Songping Mai. (2023)  
**Encoder-Decoder-Based Intra-Frame Block Partitioning Decision**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.06412v1)  

---


**ABSTRACT**  
The recursive intra-frame block partitioning decision process, a crucial component of the next-generation video coding standards, exerts significant influence over the encoding time. In this paper, we propose an encoder-decoder neural network (NN) to accelerate this process. Specifically, a CNN is utilized to compress the pixel data of the largest coding unit (LCU) into a fixed-length vector. Subsequently, a Transformer decoder is employed to transcribe the fixed-length vector into a variable-length vector, which represents the block partitioning outcomes of the encoding LCU. The vector transcription process adheres to the constraints imposed by the block partitioning algorithm. By fully parallelizing the NN prediction in the intra-mode decision, substantial time savings can be attained during the decision phase. The experimental results obtained from high-definition (HD) sequences coding demonstrate that this framework achieves a remarkable 87.84\% reduction in encoding time, with a relatively small loss (8.09\%) of coding performance compared to AVS3 HPM4.0.

{{</citation>}}


## eess.IV (2)



### (157/159) Three-Dimensional Medical Image Fusion with Deformable Cross-Attention (Lin Liu et al., 2023)

{{<citation>}}

Lin Liu, Xinxin Fan, Chulong Zhang, Jingjing Dai, Yaoqin Xie, Xiaokun Liang. (2023)  
**Three-Dimensional Medical Image Fusion with Deformable Cross-Attention**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV, physics-med-ph  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.06291v1)  

---


**ABSTRACT**  
Multimodal medical image fusion plays an instrumental role in several areas of medical image processing, particularly in disease recognition and tumor detection. Traditional fusion methods tend to process each modality independently before combining the features and reconstructing the fusion image. However, this approach often neglects the fundamental commonalities and disparities between multimodal information. Furthermore, the prevailing methodologies are largely confined to fusing two-dimensional (2D) medical image slices, leading to a lack of contextual supervision in the fusion images and subsequently, a decreased information yield for physicians relative to three-dimensional (3D) images. In this study, we introduce an innovative unsupervised feature mutual learning fusion network designed to rectify these limitations. Our approach incorporates a Deformable Cross Feature Blend (DCFB) module that facilitates the dual modalities in discerning their respective similarities and differences. We have applied our model to the fusion of 3D MRI and PET images obtained from 660 patients in the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset. Through the application of the DCFB module, our network generates high-quality MRI-PET fusion images. Experimental results demonstrate that our method surpasses traditional 2D image fusion methods in performance metrics such as Peak Signal to Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM). Importantly, the capacity of our method to fuse 3D images enhances the information available to physicians and researchers, thus marking a significant step forward in the field. The code will soon be available online.

{{</citation>}}


### (158/159) Cross-modal Cognitive Consensus guided Audio-Visual Segmentation (Zhaofeng Shi et al., 2023)

{{<citation>}}

Zhaofeng Shi, Qingbo Wu, Hongliang Li, Fanman Meng, Linfeng Xu. (2023)  
**Cross-modal Cognitive Consensus guided Audio-Visual Segmentation**  

---
Primary Category: eess.IV  
Categories: 68U10, I-4-6, cs-SD, eess-AS, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.06259v1)  

---


**ABSTRACT**  
Audio-Visual Segmentation (AVS) aims to extract the sounding object from a video frame, which is represented by a pixel-wise segmentation mask. The pioneering work conducts this task through dense feature-level audio-visual interaction, which ignores the dimension gap between different modalities. More specifically, the audio clip could only provide a \textit{Global} semantic label in each sequence, but the video frame covers multiple semantic objects across different \textit{Local} regions. In this paper, we propose a Cross-modal Cognitive Consensus guided Network (C3N) to align the audio-visual semantics from the global dimension and progressively inject them into the local regions via an attention mechanism. Firstly, a Cross-modal Cognitive Consensus Inference Module (C3IM) is developed to extract a unified-modal label by integrating audio/visual classification confidence and similarities of modality-specific label embeddings. Then, we feed the unified-modal label back to the visual backbone as the explicit semantic-level guidance via a Cognitive Consensus guided Attention Module (CCAM), which highlights the local features corresponding to the interested object. Extensive experiments on the Single Sound Source Segmentation (S4) setting and Multiple Sound Source Segmentation (MS3) setting of the AVSBench dataset demonstrate the effectiveness of the proposed method, which achieves state-of-the-art performance.

{{</citation>}}


## cs.AR (1)



### (159/159) Gem5Pred: Predictive Approaches For Gem5 Simulation Time (Tian Yan et al., 2023)

{{<citation>}}

Tian Yan, Xueyang Li, Sifat Ut Taki, Saeid Mehrdad. (2023)  
**Gem5Pred: Predictive Approaches For Gem5 Simulation Time**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-LG, cs.AR  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.06290v1)  

---


**ABSTRACT**  
Gem5, an open-source, flexible, and cost-effective simulator, is widely recognized and utilized in both academic and industry fields for hardware simulation. However, the typically time-consuming nature of simulating programs on Gem5 underscores the need for a predictive model that can estimate simulation time. As of now, no such dataset or model exists. In response to this gap, this paper makes a novel contribution by introducing a unique dataset specifically created for this purpose. We also conducted analysis of the effects of different instruction types on the simulation time in Gem5. After this, we employ three distinct models leveraging CodeBERT to execute the prediction task based on the developed dataset. Our superior regression model achieves a Mean Absolute Error (MAE) of 0.546, while our top-performing classification model records an Accuracy of 0.696. Our models establish a foundation for future investigations on this topic, serving as benchmarks against which subsequent models can be compared. We hope that our contribution can simulate further research in this field. The dataset we used is available at https://github.com/XueyangLiOSU/Gem5Pred.

{{</citation>}}
