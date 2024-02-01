---
draft: false
title: "arXiv @ 2024.01.27"
date: 2024-01-27
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.27"
    identifier: arxiv_20240127
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (34)](#cscl-34)
- [cs.CR (3)](#cscr-3)
- [cs.CY (6)](#cscy-6)
- [cs.CV (17)](#cscv-17)
- [cs.RO (5)](#csro-5)
- [cs.SD (1)](#cssd-1)
- [eess.IV (3)](#eessiv-3)
- [cs.LG (21)](#cslg-21)
- [cs.HC (9)](#cshc-9)
- [cs.NI (1)](#csni-1)
- [eess.AS (2)](#eessas-2)
- [cs.SE (7)](#csse-7)
- [cs.DC (1)](#csdc-1)
- [physics.optics (1)](#physicsoptics-1)
- [cs.SI (2)](#cssi-2)
- [quant-ph (1)](#quant-ph-1)
- [cs.AI (3)](#csai-3)
- [cs.DS (1)](#csds-1)
- [eess.SY (2)](#eesssy-2)
- [stat.ML (1)](#statml-1)
- [cs.IR (1)](#csir-1)

## cs.CL (34)



### (1/122) Detecting Structured Language Alternations in Historical Documents by Combining Language Identification with Fourier Analysis (Hale Sirin et al., 2024)

{{<citation>}}

Hale Sirin, Sabrina Li, Tom Lippincott. (2024)  
**Detecting Structured Language Alternations in Historical Documents by Combining Language Identification with Fourier Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Identification  
[Paper Link](http://arxiv.org/abs/2401.14569v1)  

---


**ABSTRACT**  
In this study, we present a generalizable workflow to identify documents in a historic language with a nonstandard language and script combination, Armeno-Turkish. We introduce the task of detecting distinct patterns of multilinguality based on the frequency of structured language alternations within a document.

{{</citation>}}


### (2/122) Language Modelling Approaches to Adaptive Machine Translation (Yasmin Moslem, 2024)

{{<citation>}}

Yasmin Moslem. (2024)  
**Language Modelling Approaches to Adaptive Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs-IR, cs.CL  
Keywords: Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.14559v1)  

---


**ABSTRACT**  
Consistency is a key requirement of high-quality translation. It is especially important to adhere to pre-approved terminology and adapt to corrected translations in domain-specific projects. Machine translation (MT) has achieved significant progress in the area of domain adaptation. However, in-domain data scarcity is common in translation settings, due to the lack of specialised datasets and terminology, or inconsistency and inaccuracy of available in-domain translations. In such scenarios where there is insufficient in-domain data to fine-tune MT models, producing translations that are consistent with the relevant context is challenging. While real-time adaptation can make use of smaller amounts of in-domain data to improve the translation on the fly, it remains challenging due to supported context limitations and efficiency constraints. Large language models (LLMs) have recently shown interesting capabilities of in-context learning, where they learn to replicate certain input-output text generation patterns, without further fine-tuning. Such capabilities have opened new horizons for domain-specific data augmentation and real-time adaptive MT. This work attempts to address two main relevant questions: 1) in scenarios involving human interaction and continuous feedback, can we employ language models to improve the quality of adaptive MT at inference time? and 2) in the absence of sufficient in-domain data, can we use pre-trained large-scale language models to improve the process of MT domain adaptation?

{{</citation>}}


### (3/122) Do Not (Always) Look Right: Investigating the Capabilities of Decoder-Based Large Language Models for Sequence Labeling (David Dukić et al., 2024)

{{<citation>}}

David Dukić, Jan Šnajder. (2024)  
**Do Not (Always) Look Right: Investigating the Capabilities of Decoder-Based Large Language Models for Sequence Labeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLU  
[Paper Link](http://arxiv.org/abs/2401.14556v1)  

---


**ABSTRACT**  
Pre-trained language models based on masked language modeling (MLM) objective excel in natural language understanding (NLU) tasks. While fine-tuned MLM-based encoders consistently outperform causal language modeling decoders of comparable size, a recent trend of scaling decoder models to multiple billion parameters resulted in large language models (LLMs), making them competitive with MLM-based encoders. Although scale amplifies their prowess in NLU tasks, LLMs fall short of SOTA results in information extraction (IE) tasks, many framed as sequence labeling (SL). However, whether this is an intrinsic limitation of LLMs or whether their SL performance can be improved remains unclear. To address this, we explore strategies to enhance the SL performance of "open" LLMs (Llama2 and Mistral) on IE tasks. We investigate bidirectional information flow within groups of decoder blocks, applying layer-wise removal or enforcement of the causal mask (CM) during LLM fine-tuning. This approach yields performance gains competitive with SOTA SL models, matching or outperforming the results of CM removal from all blocks. Our findings hold for diverse SL tasks, proving that "open" LLMs with layer-dependent CM removal outperform strong MLM-based encoders and instruction-tuned LLMs. However, we observe no effect from CM removal on a small scale when maintaining an equivalent model size, pre-training steps, and pre-training and fine-tuning data.

{{</citation>}}


### (4/122) Relative Value Biases in Large Language Models (William M. Hayes et al., 2024)

{{<citation>}}

William M. Hayes, Nicolas Yax, Stefano Palminteri. (2024)  
**Relative Value Biases in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Bias, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14530v1)  

---


**ABSTRACT**  
Studies of reinforcement learning in humans and animals have demonstrated a preference for options that yielded relatively better outcomes in the past, even when those options are associated with lower absolute reward. The present study tested whether large language models would exhibit a similar bias. We had gpt-4-1106-preview (GPT-4 Turbo) and Llama-2-70B make repeated choices between pairs of options with the goal of maximizing payoffs. A complete record of previous outcomes was included in each prompt. Both models exhibited relative value decision biases similar to those observed in humans and animals. Making relative comparisons among outcomes more explicit magnified the bias, whereas prompting the models to estimate expected outcomes caused the bias to disappear. These results have implications for the potential mechanisms that contribute to context-dependent choice in human agents.

{{</citation>}}


### (5/122) MEDs for PETs: Multilingual Euphemism Disambiguation for Potentially Euphemistic Terms (Patrick Lee et al., 2024)

{{<citation>}}

Patrick Lee, Alain Chirino Trujillo, Diana Cuevas Plancarte, Olumide Ebenezer Ojo, Xinyi Liu, Iyanuoluwa Shode, Yuan Zhao, Jing Peng, Anna Feldman. (2024)  
**MEDs for PETs: Multilingual Euphemism Disambiguation for Potentially Euphemistic Terms**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Multilingual  
[Paper Link](http://arxiv.org/abs/2401.14526v1)  

---


**ABSTRACT**  
This study investigates the computational processing of euphemisms, a universal linguistic phenomenon, across multiple languages. We train a multilingual transformer model (XLM-RoBERTa) to disambiguate potentially euphemistic terms (PETs) in multilingual and cross-lingual settings. In line with current trends, we demonstrate that zero-shot learning across languages takes place. We also show cases where multilingual models perform better on the task compared to monolingual models by a statistically significant margin, indicating that multilingual data presents additional opportunities for models to learn about cross-lingual, computational properties of euphemisms. In a follow-up analysis, we focus on universal euphemistic "categories" such as death and bodily functions among others. We test to see whether cross-lingual data of the same domain is more important than within-language data of other domains to further understand the nature of the cross-lingual transfer.

{{</citation>}}


### (6/122) Evaluating GPT-3.5's Awareness and Summarization Abilities for European Constitutional Texts with Shared Topics (Candida M. Greco et al., 2024)

{{<citation>}}

Candida M. Greco, A. Tagarelli. (2024)  
**Evaluating GPT-3.5's Awareness and Summarization Abilities for European Constitutional Texts with Shared Topics**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-DL, cs.CL, physics-soc-ph  
Keywords: GPT, GPT-3.5, Summarization  
[Paper Link](http://arxiv.org/abs/2401.14524v1)  

---


**ABSTRACT**  
Constitutions are foundational legal documents that underpin the governmental and societal structures. As such, they are a reflection of a nation's cultural and social uniqueness, but also contribute to establish topics of universal importance, like citizens' rights and duties (RD). In this work, using the renowned GPT-3.5, we leverage generative large language models to understand constitutional passages that transcend national boundaries. A key contribution of our study is the introduction of a novel application of abstractive summarization on a multi-source collection of constitutional texts, with a focus on European countries' constitution passages related to RD topics. Our results show the meaningfulness of GPT-3.5 to produce informative, coherent and faithful summaries capturing RD topics across European countries.

{{</citation>}}


### (7/122) K-QA: A Real-World Medical Q&A Benchmark (Itay Manes et al., 2024)

{{<citation>}}

Itay Manes, Naama Ronn, David Cohen, Ran Ilan Ber, Zehavi Horowitz-Kugler, Gabriel Stanovsky. (2024)  
**K-QA: A Real-World Medical Q&A Benchmark**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs-LG, cs.CL  
Keywords: AI, NLI, NLP, QA  
[Paper Link](http://arxiv.org/abs/2401.14493v1)  

---


**ABSTRACT**  
Ensuring the accuracy of responses provided by large language models (LLMs) is crucial, particularly in clinical settings where incorrect information may directly impact patient health. To address this challenge, we construct K-QA, a dataset containing 1,212 patient questions originating from real-world conversations held on K Health (an AI-driven clinical platform). We employ a panel of in-house physicians to answer and manually decompose a subset of K-QA into self-contained statements. Additionally, we formulate two NLI-based evaluation metrics approximating recall and precision: (1) comprehensiveness, measuring the percentage of essential clinical information in the generated answer and (2) hallucination rate, measuring the number of statements from the physician-curated response contradicted by the LLM answer. Finally, we use K-QA along with these metrics to evaluate several state-of-the-art models, as well as the effect of in-context learning and medically-oriented augmented retrieval schemes developed by the authors. Our findings indicate that in-context learning improves the comprehensiveness of the models, and augmented retrieval is effective in reducing hallucinations. We make K-QA available to to the community to spur research into medically accurate NLP applications.

{{</citation>}}


### (8/122) LongHealth: A Question Answering Benchmark with Long Clinical Documents (Lisa Adams et al., 2024)

{{<citation>}}

Lisa Adams, Felix Busch, Tianyu Han, Jean-Baptiste Excoffier, Matthieu Ortala, Alexander Löser, Hugo JWL. Aerts, Jakob Nikolas Kather, Daniel Truhn, Keno Bressem. (2024)  
**LongHealth: A Question Answering Benchmark with Long Clinical Documents**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Clinical, GPT, GPT-3.5, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.14490v1)  

---


**ABSTRACT**  
Background: Recent advancements in large language models (LLMs) offer potential benefits in healthcare, particularly in processing extensive patient records. However, existing benchmarks do not fully assess LLMs' capability in handling real-world, lengthy clinical data.   Methods: We present the LongHealth benchmark, comprising 20 detailed fictional patient cases across various diseases, with each case containing 5,090 to 6,754 words. The benchmark challenges LLMs with 400 multiple-choice questions in three categories: information extraction, negation, and sorting, challenging LLMs to extract and interpret information from large clinical documents.   Results: We evaluated nine open-source LLMs with a minimum of 16,000 tokens and also included OpenAI's proprietary and cost-efficient GPT-3.5 Turbo for comparison. The highest accuracy was observed for Mixtral-8x7B-Instruct-v0.1, particularly in tasks focused on information retrieval from single and multiple patient documents. However, all models struggled significantly in tasks requiring the identification of missing information, highlighting a critical area for improvement in clinical data interpretation.   Conclusion: While LLMs show considerable potential for processing long clinical documents, their current accuracy levels are insufficient for reliable clinical use, especially in scenarios requiring the identification of missing information. The LongHealth benchmark provides a more realistic assessment of LLMs in a healthcare setting and highlights the need for further model refinement for safe and effective clinical application.   We make the benchmark and evaluation code publicly available.

{{</citation>}}


### (9/122) Modular Adaptation of Multilingual Encoders to Written Swiss German Dialect (Jannis Vamvas et al., 2024)

{{<citation>}}

Jannis Vamvas, Noëmi Aepli, Rico Sennrich. (2024)  
**Modular Adaptation of Multilingual Encoders to Written Swiss German Dialect**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual, NLP  
[Paper Link](http://arxiv.org/abs/2401.14400v1)  

---


**ABSTRACT**  
Creating neural text encoders for written Swiss German is challenging due to a dearth of training data combined with dialectal variation. In this paper, we build on several existing multilingual encoders and adapt them to Swiss German using continued pre-training. Evaluation on three diverse downstream tasks shows that simply adding a Swiss German adapter to a modular encoder achieves 97.5% of fully monolithic adaptation performance. We further find that for the task of retrieving Swiss German sentences given Standard German queries, adapting a character-level model is more effective than the other adaptation strategies. We release our code and the models trained for our experiments at https://github.com/ZurichNLP/swiss-german-text-encoders

{{</citation>}}


### (10/122) TURNA: A Turkish Encoder-Decoder Language Model for Enhanced Understanding and Generation (Gökçe Uludoğan et al., 2024)

{{<citation>}}

Gökçe Uludoğan, Zeynep Yirmibeşoğlu Balal, Furkan Akkurt, Melikşah Türker, Onur Güngör, Susan Üsküdarlı. (2024)  
**TURNA: A Turkish Encoder-Decoder Language Model for Enhanced Understanding and Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.14373v1)  

---


**ABSTRACT**  
The recent advances in natural language processing have predominantly favored well-resourced English-centric models, resulting in a significant gap with low-resource languages. In this work, we introduce the language model TURNA, which is developed for the low-resource language Turkish and is capable of both natural language understanding and generation tasks. TURNA is pretrained with an encoder-decoder architecture based on the unified framework UL2 with a diverse corpus that we specifically curated for this purpose. We evaluated TURNA with three generation tasks and five understanding tasks for Turkish. The results show that TURNA outperforms several multilingual models in both understanding and generation tasks, and competes with monolingual Turkish models in understanding tasks. TURNA is made available at https://huggingface.co/boun-tabi-LMG/TURNA .

{{</citation>}}


### (11/122) Genie: Achieving Human Parity in Content-Grounded Datasets Generation (Asaf Yehudai et al., 2024)

{{<citation>}}

Asaf Yehudai, Boaz Carmeli, Yosi Mass, Ofir Arviv, Nathaniel Mills, Assaf Toledo, Eyal Shnarch, Leshem Choshen. (2024)  
**Genie: Achieving Human Parity in Content-Grounded Datasets Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: QA, Summarization  
[Paper Link](http://arxiv.org/abs/2401.14367v1)  

---


**ABSTRACT**  
The lack of high-quality data for content-grounded generation tasks has been identified as a major obstacle to advancing these tasks. To address this gap, we propose Genie, a novel method for automatically generating high-quality content-grounded data. It consists of three stages: (a) Content Preparation, (b) Generation: creating task-specific examples from the content (e.g., question-answer pairs or summaries). (c) Filtering mechanism aiming to ensure the quality and faithfulness of the generated data. We showcase this methodology by generating three large-scale synthetic data, making wishes, for Long-Form Question-Answering (LFQA), summarization, and information extraction. In a human evaluation, our generated data was found to be natural and of high quality. Furthermore, we compare models trained on our data with models trained on human-written data -- ELI5 and ASQA for LFQA and CNN-DailyMail for Summarization. We show that our models are on par with or outperforming models trained on human-generated data and consistently outperforming them in faithfulness. Finally, we applied our method to create LFQA data within the medical domain and compared a model trained on it with models trained on other domains.

{{</citation>}}


### (12/122) A Comparative Analysis of Noise Reduction Methods in Sentiment Analysis on Noisy Bangla Texts (Kazi Toufique Elahi et al., 2024)

{{<citation>}}

Kazi Toufique Elahi, Tasnuva Binte Rahman, Shakil Shahriar, Samir Sarker, Md. Tanvir Rouf Shawon, G. M. Shahariar. (2024)  
**A Comparative Analysis of Noise Reduction Methods in Sentiment Analysis on Noisy Bangla Texts**  

---
Primary Category: cs.CL  
Categories: 68T50 (Primary), I-2-7, cs-CL, cs.CL  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2401.14360v2)  

---


**ABSTRACT**  
While Bangla is considered a language with limited resources, sentiment analysis has been a subject of extensive research in the literature. Nevertheless, there is a scarcity of exploration into sentiment analysis specifically in the realm of noisy Bangla texts. In this paper, we introduce a dataset (NC-SentNoB) that we annotated manually to identify ten different types of noise found in a pre-existing sentiment analysis dataset comprising of around 15K noisy Bangla texts. At first, given an input noisy text, we identify the noise type, addressing this as a multi-label classification task. Then, we introduce baseline noise reduction methods to alleviate noise prior to conducting sentiment analysis. Finally, we assess the performance of fine-tuned sentiment analysis models with both noisy and noise-reduced texts to make comparisons. The experimental findings indicate that the noise reduction methods utilized are not satisfactory, highlighting the need for more suitable noise reduction methods in future research endeavors. We have made the implementation and dataset presented in this paper publicly available at https://github.com/ktoufiquee/A-Comparative-Analysis-of-Noise-Reduction-Methods-in-Sentiment-Analysis-on-Noisy-Bangla-Texts

{{</citation>}}


### (13/122) Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts (Maciej Besta et al., 2024)

{{<citation>}}

Maciej Besta, Florim Memedi, Zhenyu Zhang, Robert Gerstenberger, Nils Blach, Piotr Nyczyk, Marcin Copik, Grzegorz Kwaśniewski, Jürgen Müller, Lukas Gianinazzi, Ales Kubicek, Hubert Niewiadomski, Onur Mutlu, Torsten Hoefler. (2024)  
**Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.14295v1)  

---


**ABSTRACT**  
The field of natural language processing (NLP) has witnessed significant progress in recent years, with a notable focus on improving large language models' (LLM) performance through innovative prompting techniques. Among these, prompt engineering coupled with structures has emerged as a promising paradigm, with designs such as Chain-of-Thought, Tree of Thoughts, or Graph of Thoughts, in which the overall LLM reasoning is guided by a structure such as a graph. As illustrated with numerous examples, this paradigm significantly enhances the LLM's capability to solve numerous tasks, ranging from logical or mathematical reasoning to planning or creative writing. To facilitate the understanding of this growing field and pave the way for future developments, we devise a general blueprint for effective and efficient LLM reasoning schemes. For this, we conduct an in-depth analysis of the prompt execution pipeline, clarifying and clearly defining different concepts. We then build the first taxonomy of structure-enhanced LLM reasoning schemes. We focus on identifying fundamental classes of harnessed structures, and we analyze the representations of these structures, algorithms executed with these structures, and many others. We refer to these structures as reasoning topologies, because their representation becomes to a degree spatial, as they are contained within the LLM context. Our study compares existing prompting schemes using the proposed taxonomy, discussing how certain design choices lead to different patterns in performance and cost. We also outline theoretical underpinnings, relationships between prompting and others parts of the LLM ecosystem such as knowledge bases, and the associated research challenges. Our work will help to advance future prompt engineering techniques.

{{</citation>}}


### (14/122) RomanSetu: Efficiently unlocking multilingual capabilities of Large Language Models models via Romanization (Jaavid Aktar Husain et al., 2024)

{{<citation>}}

Jaavid Aktar Husain, Raj Dabre, Aswanth Kumar, Ratish Puduppully, Anoop Kunchukuttan. (2024)  
**RomanSetu: Efficiently unlocking multilingual capabilities of Large Language Models models via Romanization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.14280v1)  

---


**ABSTRACT**  
This study addresses the challenge of extending Large Language Models (LLMs) to non-English languages, specifically those using non-Latin scripts. We propose an innovative approach that utilizes the romanized form of text as an interface for LLMs, hypothesizing that its frequent informal use and shared tokens with English enhance cross-lingual alignment. Focusing on Hindi, we demonstrate through Hindi-to-English translation and sentiment analysis tasks that romanized text not only significantly improves inference efficiency due to its lower fertility compared to native text but also achieves competitive performance with limited pre-training. Additionally, our novel multi-script prompting approach, which combines romanized and native texts, shows promise in further enhancing task performance. These findings suggest the potential of romanization in bridging the language gap for LLM applications, with future work aimed at expanding this approach to more languages and tasks.

{{</citation>}}


### (15/122) Transformers and Cortical Waves: Encoders for Pulling In Context Across Time (Lyle Muller et al., 2024)

{{<citation>}}

Lyle Muller, Patricia S. Churchland, Terrence J. Sejnowski. (2024)  
**Transformers and Cortical Waves: Encoders for Pulling In Context Across Time**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.14267v1)  

---


**ABSTRACT**  
The capabilities of transformer networks such as ChatGPT and other Large Language Models (LLMs) have captured the world's attention. The crucial computational mechanism underlying their performance relies on transforming a complete input sequence - for example, all the words in a sentence into a long "encoding vector" - that allows transformers to learn long-range temporal dependencies in naturalistic sequences. Specifically, "self-attention" applied to this encoding vector enhances temporal context in transformers by computing associations between pairs of words in the input sequence. We suggest that waves of neural activity, traveling across single cortical regions or across multiple regions at the whole-brain scale, could implement a similar encoding principle. By encapsulating recent input history into a single spatial pattern at each moment in time, cortical waves may enable temporal context to be extracted from sequences of sensory inputs, the same computational principle used in transformers.

{{</citation>}}


### (16/122) Improving Natural Language Capability of Code Large Language Model (Wei Li et al., 2024)

{{<citation>}}

Wei Li, Daoguang Zan, Bei Guan, Ailun Yu, Xiaolin Chen, Yongji Wang. (2024)  
**Improving Natural Language Capability of Code Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14242v1)  

---


**ABSTRACT**  
Code large language models (Code LLMs) have demonstrated remarkable performance in code generation. Nonetheless, most existing works focus on boosting code LLMs from the perspective of programming capabilities, while their natural language capabilities receive less attention. To fill this gap, we thus propose a novel framework, comprising two modules: AttentionExtractor, which is responsible for extracting key phrases from the user's natural language requirements, and AttentionCoder, which leverages these extracted phrases to generate target code to solve the requirement. This framework pioneers an innovative idea by seamlessly integrating code LLMs with traditional natural language processing tools. To validate the effectiveness of the framework, we craft a new code generation benchmark, called MultiNL-H, covering five natural languages. Extensive experimental results demonstrate the effectiveness of our proposed framework.

{{</citation>}}


### (17/122) Semantic Sensitivities and Inconsistent Predictions: Measuring the Fragility of NLI Models (Erik Arakelyan et al., 2024)

{{<citation>}}

Erik Arakelyan, Zhaoqi Liu, Isabelle Augenstein. (2024)  
**Semantic Sensitivities and Inconsistent Predictions: Measuring the Fragility of NLI Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: NLI, NLU, Natural Language Inference, Natural Language Understanding  
[Paper Link](http://arxiv.org/abs/2401.14440v2)  

---


**ABSTRACT**  
Recent studies of the emergent capabilities of transformer-based Natural Language Understanding (NLU) models have indicated that they have an understanding of lexical and compositional semantics. We provide evidence that suggests these claims should be taken with a grain of salt: we find that state-of-the-art Natural Language Inference (NLI) models are sensitive towards minor semantics preserving surface-form variations, which lead to sizable inconsistent model decisions during inference. Notably, this behaviour differs from valid and in-depth comprehension of compositional semantics, however does neither emerge when evaluating model accuracy on standard benchmarks nor when probing for syntactic, monotonic, and logically robust reasoning. We propose a novel framework to measure the extent of semantic sensitivity. To this end, we evaluate NLI models on adversarially generated examples containing minor semantics-preserving surface-form input noise. This is achieved using conditional text generation, with the explicit condition that the NLI model predicts the relationship between the original and adversarial inputs as a symmetric equivalence entailment. We systematically study the effects of the phenomenon across NLI models for $\textbf{in-}$ and $\textbf{out-of-}$ domain settings. Our experiments show that semantic sensitivity causes performance degradations of $12.92\%$ and $23.71\%$ average over $\textbf{in-}$ and $\textbf{out-of-}$ domain settings, respectively. We further perform ablation studies, analysing this phenomenon across models, datasets, and variations in inference and show that semantic sensitivity can lead to major inconsistency within model predictions.

{{</citation>}}


### (18/122) BayesPrompt: Prompting Large-Scale Pre-Trained Language Models on Few-shot Inference via Debiased Domain Abstraction (Jiangmeng Li et al., 2024)

{{<citation>}}

Jiangmeng Li, Fei Song, Yifan Jin, Wenwen Qiang, Changwen Zheng, Fuchun Sun, Hui Xiong. (2024)  
**BayesPrompt: Prompting Large-Scale Pre-Trained Language Models on Few-shot Inference via Debiased Domain Abstraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.14166v2)  

---


**ABSTRACT**  
As a novel and effective fine-tuning paradigm based on large-scale pre-trained language models (PLMs), prompt-tuning aims to reduce the gap between downstream tasks and pre-training objectives. While prompt-tuning has yielded continuous advancements in various tasks, such an approach still remains a persistent defect: prompt-tuning methods fail to generalize to specific few-shot patterns. From the perspective of distribution analyses, we disclose that the intrinsic issues behind the phenomenon are the over-multitudinous conceptual knowledge contained in PLMs and the abridged knowledge for target downstream domains, which jointly result in that PLMs mis-locate the knowledge distributions corresponding to the target domains in the universal knowledge embedding space. To this end, we intuitively explore to approximate the unabridged target domains of downstream tasks in a debiased manner, and then abstract such domains to generate discriminative prompts, thereby providing the de-ambiguous guidance for PLMs. Guided by such an intuition, we propose a simple yet effective approach, namely BayesPrompt, to learn prompts that contain the domain discriminative information against the interference from domain-irrelevant knowledge. BayesPrompt primitively leverages known distributions to approximate the debiased factual distributions of target domains and further uniformly samples certain representative features from the approximated distributions to generate the ultimate prompts for PLMs. We provide theoretical insights with the connection to domain adaptation. Empirically, our method achieves state-of-the-art performance on benchmarks.

{{</citation>}}


### (19/122) On the Affinity, Rationality, and Diversity of Hierarchical Topic Modeling (Xiaobao Wu et al., 2024)

{{<citation>}}

Xiaobao Wu, Fengjun Pan, Thong Nguyen, Yichao Feng, Chaoqun Liu, Cong-Duy Nguyen, Anh Tuan Luu. (2024)  
**On the Affinity, Rationality, and Diversity of Hierarchical Topic Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Topic Model, Topic Modeling  
[Paper Link](http://arxiv.org/abs/2401.14113v1)  

---


**ABSTRACT**  
Hierarchical topic modeling aims to discover latent topics from a corpus and organize them into a hierarchy to understand documents with desirable semantic granularity. However, existing work struggles with producing topic hierarchies of low affinity, rationality, and diversity, which hampers document understanding. To overcome these challenges, we in this paper propose Transport Plan and Context-aware Hierarchical Topic Model (TraCo). Instead of early simple topic dependencies, we propose a transport plan dependency method. It constrains dependencies to ensure their sparsity and balance, and also regularizes topic hierarchy building with them. This improves affinity and diversity of hierarchies. We further propose a context-aware disentangled decoder. Rather than previously entangled decoding, it distributes different semantic granularity to topics at different levels by disentangled decoding. This facilitates the rationality of hierarchies. Experiments on benchmark datasets demonstrate that our method surpasses state-of-the-art baselines, effectively improving the affinity, rationality, and diversity of hierarchical topic modeling with better performance on downstream tasks.

{{</citation>}}


### (20/122) CompactifAI: Extreme Compression of Large Language Models using Quantum-Inspired Tensor Networks (Andrei Tomut et al., 2024)

{{<citation>}}

Andrei Tomut, Saeed S. Jahromi, Sukhbinder Singh, Faysal Ishtiaq, Cesar Muñoz, Prabdeep Singh Bajaj, Ali Elborady, Gianni del Bimbo, Mehrazin Alizadeh, David Montero, Pablo Martin-Ramiro, Muhammad Ibrahim, Oussama Tahiri Alaoui, John Malcolm, Samuel Mugel, Roman Orus. (2024)  
**CompactifAI: Extreme Compression of Large Language Models using Quantum-Inspired Tensor Networks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL, quant-ph  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14109v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) such as ChatGPT and LlaMA are advancing rapidly in generative Artificial Intelligence (AI), but their immense size poses significant challenges, such as huge training and inference costs, substantial energy demands, and limitations for on-site deployment. Traditional compression methods such as pruning, distillation, and low-rank approximation focus on reducing the effective number of neurons in the network, while quantization focuses on reducing the numerical precision of individual weights to reduce the model size while keeping the number of neurons fixed. While these compression methods have been relatively successful in practice, there's no compelling reason to believe that truncating the number of neurons is an optimal strategy. In this context, this paper introduces CompactifAI, an innovative LLM compression approach using quantum-inspired Tensor Networks that focuses on the model's correlation space instead, allowing for a more controlled, refined and interpretable model compression. Our method is versatile and can be implemented with - or on top of - other compression techniques. As a benchmark, we demonstrate that CompactifAI alone enables compression of the LlaMA-2 7B model to only $30\%$ of its original size while recovering over $90\%$ of the original accuracy after a brief distributed retraining.

{{</citation>}}


### (21/122) Ta'keed: The First Generative Fact-Checking System for Arabic Claims (Saud Althabiti et al., 2024)

{{<citation>}}

Saud Althabiti, Mohammad Ammar Alsalka, Eric Atwell. (2024)  
**Ta'keed: The First Generative Fact-Checking System for Arabic Claims**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Fact-Checking  
[Paper Link](http://arxiv.org/abs/2401.14067v1)  

---


**ABSTRACT**  
This paper introduces Ta'keed, an explainable Arabic automatic fact-checking system. While existing research often focuses on classifying claims as "True" or "False," there is a limited exploration of generating explanations for claim credibility, particularly in Arabic. Ta'keed addresses this gap by assessing claim truthfulness based on retrieved snippets, utilizing two main components: information retrieval and LLM-based claim verification. We compiled the ArFactEx, a testing gold-labelled dataset with manually justified references, to evaluate the system. The initial model achieved a promising F1 score of 0.72 in the classification task. Meanwhile, the system's generated explanations are compared with gold-standard explanations syntactically and semantically. The study recommends evaluating using semantic similarities, resulting in an average cosine similarity score of 0.76. Additionally, we explored the impact of varying snippet quantities on claim classification accuracy, revealing a potential correlation, with the model using the top seven hits outperforming others with an F1 score of 0.77.

{{</citation>}}


### (22/122) Towards Goal-oriented Large Language Model Prompting: A Survey (Haochen Li et al., 2024)

{{<citation>}}

Haochen Li, Jonathan Leung, Zhiqi Shen. (2024)  
**Towards Goal-oriented Large Language Model Prompting: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.14043v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown prominent performance in various downstream tasks in which prompt engineering plays a pivotal role in optimizing LLMs' performance. This paper, not as an overview of current prompt engineering methods, aims to highlight the limitation of designing prompts while holding an anthropomorphic assumption that expects LLMs to think like humans. From our review of 35 representative studies, we demonstrate that a goal-oriented prompt formulation, which guides LLMs to follow established human logical thinking, significantly improves the performance of LLMs. Furthermore, We introduce a novel taxonomy that categorizes goal-oriented prompting methods into five interconnected stages and we demonstrate the broad applicability of our framework by summarizing ten applicable tasks. With four future directions proposed, we hope to further emphasize and promote goal-oriented prompt engineering.

{{</citation>}}


### (23/122) (Chat)GPT v BERT: Dawn of Justice for Semantic Change Detection (Francesco Periti et al., 2024)

{{<citation>}}

Francesco Periti, Haim Dubossarsky, Nina Tahmasebi. (2024)  
**(Chat)GPT v BERT: Dawn of Justice for Semantic Change Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, ChatGPT, GPT, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2401.14040v1)  

---


**ABSTRACT**  
In the universe of Natural Language Processing, Transformer-based language models like BERT and (Chat)GPT have emerged as lexical superheroes with great power to solve open research problems. In this paper, we specifically focus on the temporal problem of semantic change, and evaluate their ability to solve two diachronic extensions of the Word-in-Context (WiC) task: TempoWiC and HistoWiC. In particular, we investigate the potential of a novel, off-the-shelf technology like ChatGPT (and GPT) 3.5 compared to BERT, which represents a family of models that currently stand as the state-of-the-art for modeling semantic change. Our experiments represent the first attempt to assess the use of (Chat)GPT for studying semantic change. Our results indicate that ChatGPT performs significantly worse than the foundational GPT version. Furthermore, our results demonstrate that (Chat)GPT achieves slightly lower performance than BERT in detecting long-term changes but performs significantly worse in detecting short-term changes.

{{</citation>}}


### (24/122) Unitxt: Flexible, Shareable and Reusable Data Preparation and Evaluation for Generative AI (Elron Bandel et al., 2024)

{{<citation>}}

Elron Bandel, Yotam Perlitz, Elad Venezian, Roni Friedman-Melamed, Ofir Arviv, Matan Orbach, Shachar Don-Yehyia, Dafna Sheinwald, Ariel Gera, Leshem Choshen, Michal Shmueli-Scheuer, Yoav Katz. (2024)  
**Unitxt: Flexible, Shareable and Reusable Data Preparation and Evaluation for Generative AI**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Generative AI, NLP  
[Paper Link](http://arxiv.org/abs/2401.14019v1)  

---


**ABSTRACT**  
In the dynamic landscape of generative NLP, traditional text processing pipelines limit research flexibility and reproducibility, as they are tailored to specific dataset, task, and model combinations. The escalating complexity, involving system prompts, model-specific formats, instructions, and more, calls for a shift to a structured, modular, and customizable solution. Addressing this need, we present Unitxt, an innovative library for customizable textual data preparation and evaluation tailored to generative language models. Unitxt natively integrates with common libraries like HuggingFace and LM-eval-harness and deconstructs processing flows into modular components, enabling easy customization and sharing between practitioners. These components encompass model-specific formats, task prompts, and many other comprehensive dataset processing definitions. The Unitxt-Catalog centralizes these components, fostering collaboration and exploration in modern textual data workflows. Beyond being a tool, Unitxt is a community-driven platform, empowering users to build, share, and advance their pipelines collaboratively. Join the Unitxt community at https://github.com/IBM/unitxt!

{{</citation>}}


### (25/122) Towards Uncertainty-Aware Language Agent (Jiuzhou Han et al., 2024)

{{<citation>}}

Jiuzhou Han, Wray Buntine, Ehsan Shareghi. (2024)  
**Towards Uncertainty-Aware Language Agent**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2401.14016v1)  

---


**ABSTRACT**  
While Language Agents have achieved promising success by placing Large Language Models at the core of a more versatile design that dynamically interacts with the external world, the existing approaches neglect the notion of uncertainty during these interactions. We present the Uncertainty-Aware Language Agent (UALA), a framework that orchestrates the interaction between the agent and the external world using uncertainty quantification. Compared with other well-known counterparts like ReAct, our extensive experiments across 3 representative tasks (HotpotQA, StrategyQA, MMLU) and various LLM sizes demonstrates that UALA brings a significant improvement of performance, while having a substantially lower reliance on the external world (i.e., reduced number of tool calls and tokens). Our analyses provide various insights including the great potential of UALA compared with agent fine-tuning, and underscoring the unreliably of verbalised confidence of LLMs as a proxy for uncertainty.

{{</citation>}}


### (26/122) CMMU: A Benchmark for Chinese Multi-modal Multi-type Question Understanding and Reasoning (Zheqi He et al., 2024)

{{<citation>}}

Zheqi He, Xinya Wu, Pengfei Zhou, Richeng Xuan, Guang Liu, Xi Yang, Qiannan Zhu, Hua Huang. (2024)  
**CMMU: A Benchmark for Chinese Multi-modal Multi-type Question Understanding and Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-MM, cs.CL  
Keywords: GPT, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.14011v2)  

---


**ABSTRACT**  
Multi-modal large language models(MLLMs) have achieved remarkable progress and demonstrated powerful knowledge comprehension and reasoning abilities. However, the mastery of domain-specific knowledge, which is essential for evaluating the intelligence of MLLMs, continues to be a challenge. Current multi-modal benchmarks for domain-specific knowledge concentrate on multiple-choice questions and are predominantly available in English, which imposes limitations on the comprehensiveness of the evaluation. To this end, we introduce CMMU, a novel benchmark for multi-modal and multi-type question understanding and reasoning in Chinese. CMMU consists of 3,603 questions in 7 subjects, covering knowledge from primary to high school. The questions can be categorized into 3 types: multiple-choice, multiple-response, and fill-in-the-blank, bringing greater challenges to MLLMs. In addition, we propose a rigorous evaluation strategy called ShiftCheck for assessing multiple-choice questions. The strategy aims to reduce position bias, minimize the influence of randomness on correctness, and perform a quantitative analysis of position bias. We evaluate seven open-source MLLMs along with GPT4-V, Gemini-Pro, and Qwen-VL-Plus. The results demonstrate that CMMU poses a significant challenge to the recent MLLMs.

{{</citation>}}


### (27/122) ConstraintChecker: A Plugin for Large Language Models to Reason on Commonsense Knowledge Bases (Quyet V. Do et al., 2024)

{{<citation>}}

Quyet V. Do, Tianqing Fang, Shizhe Diao, Zhaowei Wang, Yangqiu Song. (2024)  
**ConstraintChecker: A Plugin for Large Language Models to Reason on Commonsense Knowledge Bases**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Commonsense Knowledge, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.14003v1)  

---


**ABSTRACT**  
Reasoning over Commonsense Knowledge Bases (CSKB), i.e. CSKB reasoning, has been explored as a way to acquire new commonsense knowledge based on reference knowledge in the original CSKBs and external prior knowledge. Despite the advancement of Large Language Models (LLM) and prompt engineering techniques in various reasoning tasks, they still struggle to deal with CSKB reasoning. One of the problems is that it is hard for them to acquire explicit relational constraints in CSKBs from only in-context exemplars, due to a lack of symbolic reasoning capabilities (Bengio et al., 2021). To this end, we proposed **ConstraintChecker**, a plugin over prompting techniques to provide and check explicit constraints. When considering a new knowledge instance, ConstraintChecker employs a rule-based module to produce a list of constraints, then it uses a zero-shot learning module to check whether this knowledge instance satisfies all constraints. The acquired constraint-checking result is then aggregated with the output of the main prompting technique to produce the final output. Experimental results on CSKB Reasoning benchmarks demonstrate the effectiveness of our method by bringing consistent improvements over all prompting methods. Codes and data are available at \url{https://github.com/HKUST-KnowComp/ConstraintChecker}.

{{</citation>}}


### (28/122) Investigate-Consolidate-Exploit: A General Strategy for Inter-Task Agent Self-Evolution (Cheng Qian et al., 2024)

{{<citation>}}

Cheng Qian, Shihao Liang, Yujia Qin, Yining Ye, Xin Cong, Yankai Lin, Yesai Wu, Zhiyuan Liu, Maosong Sun. (2024)  
**Investigate-Consolidate-Exploit: A General Strategy for Inter-Task Agent Self-Evolution**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.13996v1)  

---


**ABSTRACT**  
This paper introduces Investigate-Consolidate-Exploit (ICE), a novel strategy for enhancing the adaptability and flexibility of AI agents through inter-task self-evolution. Unlike existing methods focused on intra-task learning, ICE promotes the transfer of knowledge between tasks for genuine self-evolution, similar to human experience learning. The strategy dynamically investigates planning and execution trajectories, consolidates them into simplified workflows and pipelines, and exploits them for improved task execution. Our experiments on the XAgent framework demonstrate ICE's effectiveness, reducing API calls by as much as 80% and significantly decreasing the demand for the model's capability. Specifically, when combined with GPT-3.5, ICE's performance matches that of raw GPT-4 across various agent tasks. We argue that this self-evolution approach represents a paradigm shift in agent design, contributing to a more robust AI community and ecosystem, and moving a step closer to full autonomy.

{{</citation>}}


### (29/122) Leeroo Orchestrator: Elevating LLMs Performance Through Model Integration (Alireza Mohammadshahi et al., 2024)

{{<citation>}}

Alireza Mohammadshahi, Ali Shaikh, Majid Yazdani. (2024)  
**Leeroo Orchestrator: Elevating LLMs Performance Through Model Integration**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2401.13979v1)  

---


**ABSTRACT**  
In this paper, we propose an architecture to harness the collective knowledge of multiple trained LLMs to create a new state-of-the-art. At the core of this framework is a LLM-based orchestrator that is adept at picking the right underlying LLM experts for optimal task execution. Inspired by self-play in reinforcement learning, we created a loop of query generation, orchestration, and evaluation to generate training data for the orchestrator. Our evaluation focused on the MMLU benchmark, employing models with 7B, 13B, and 34B parameters available on Hugging Face. The results demonstrate new state-of-the-art open-source models: Our Leeroo orchestrator achieves performance on par with the Mixtral model while incurring only two-thirds of its cost. Moreover, increasing the allowed cost surpasses Mixtral's accuracy by over 5% at the same cost level, reaching an accuracy of 75.9%. Further enhancements were observed when integrating GPT4 into the underlying model pool. The Leeroo orchestrator nearly matches GPT4's performance at half the cost and even exceeds GPT4's results with a 25% cost reduction. These findings illustrate the potential of our architecture in creating state-of-the-art and cost-effective LLMs by optimizing the synergy between multiple LLMs to achieve superior performance outcomes.

{{</citation>}}


### (30/122) Adaptive Text Watermark for Large Language Models (Yepeng Liu et al., 2024)

{{<citation>}}

Yepeng Liu, Yuheng Bu. (2024)  
**Adaptive Text Watermark for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13927v1)  

---


**ABSTRACT**  
The advancement of Large Language Models (LLMs) has led to increasing concerns about the misuse of AI-generated text, and watermarking for LLM-generated text has emerged as a potential solution. However, it is challenging to generate high-quality watermarked text while maintaining strong security, robustness, and the ability to detect watermarks without prior knowledge of the prompt or model. This paper proposes an adaptive watermarking strategy to address this problem. To improve the text quality and maintain robustness, we adaptively add watermarking to token distributions with high entropy measured using an auxiliary model and keep the low entropy token distributions untouched. For the sake of security and to further minimize the watermark's impact on text quality, instead of using a fixed green/red list generated from a random secret key, which can be vulnerable to decryption and forgery, we adaptively scale up the output logits in proportion based on the semantic embedding of previously generated text using a well designed semantic mapping model. Our experiments involving various LLMs demonstrate that our approach achieves comparable robustness performance to existing watermark methods. Additionally, the text generated by our method has perplexity comparable to that of \emph{un-watermarked} LLMs while maintaining security even under various attacks.

{{</citation>}}


### (31/122) WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models (Hongliang He et al., 2024)

{{<citation>}}

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, Dong Yu. (2024)  
**WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.13919v2)  

---


**ABSTRACT**  
The advancement of large language models (LLMs) leads to a new era marked by the development of autonomous applications in the real world, which drives innovation in the creation of advanced web-based agents. Existing web agents typically only handle one input modality and are evaluated only in simplified web simulators or static web snapshots, greatly limiting their applicability in real-world scenarios. To bridge this gap, we introduce WebVoyager, an innovative Large Multimodal Model (LMM) powered web agent that can complete user instructions end-to-end by interacting with real-world websites. Moreover, we propose a new evaluation protocol for web agents to address the challenges of automatic evaluation of open-ended web agent tasks, leveraging the robust multimodal comprehension capabilities of GPT-4V. We create a new benchmark by gathering real-world tasks from 15 widely used websites to evaluate our agents. We show that WebVoyager achieves a 55.7% task success rate, significantly surpassing the performance of both GPT-4 (All Tools) and the WebVoyager (text-only) setups, underscoring the exceptional capability of WebVoyager in practical applications. We found that our proposed automatic evaluation achieves 85.3% agreement with human judgment, paving the way for further development of web agents in a real-world setting.

{{</citation>}}


### (32/122) No More Distractions: an Adaptive Up-Sampling Algorithm to Reduce Data Artifacts (Han Chen, 2024)

{{<citation>}}

Han Chen. (2024)  
**No More Distractions: an Adaptive Up-Sampling Algorithm to Reduce Data Artifacts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLI  
[Paper Link](http://arxiv.org/abs/2401.13907v1)  

---


**ABSTRACT**  
Researchers recently found out that sometimes language models achieve high accuracy on benchmark data set, but they can not generalize very well with even little changes to the original data set. This is sometimes due to data artifacts, model is learning the spurious correlation between tokens and labels, instead of the semantics and logic. In this work, we analyzed SNLI data and visualized such spurious correlations. We proposed an adaptive up-sampling algorithm to correct the data artifacts, which is simple and effective, and does not need human edits or annotation. We did an experiment applying the algorithm to fix the data artifacts in SNLI data and the model trained with corrected data performed significantly better than the model trained with raw SNLI data, overall, as well as on the subset we corrected.

{{</citation>}}


### (33/122) A comparative study of zero-shot inference with large language models and supervised modeling in breast cancer pathology classification (Madhumita Sushil et al., 2024)

{{<citation>}}

Madhumita Sushil, Travis Zack, Divneet Mandair, Zhiwei Zheng, Ahmed Wali, Yan-Ning Yu, Yuwei Quan, Atul J. Butte. (2024)  
**A comparative study of zero-shot inference with large language models and supervised modeling in breast cancer pathology classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, GPT, GPT-3.5, GPT-4, LSTM, NLP  
[Paper Link](http://arxiv.org/abs/2401.13887v1)  

---


**ABSTRACT**  
Although supervised machine learning is popular for information extraction from clinical notes, creating large annotated datasets requires extensive domain expertise and is time-consuming. Meanwhile, large language models (LLMs) have demonstrated promising transfer learning capability. In this study, we explored whether recent LLMs can reduce the need for large-scale data annotations. We curated a manually-labeled dataset of 769 breast cancer pathology reports, labeled with 13 categories, to compare zero-shot classification capability of the GPT-4 model and the GPT-3.5 model with supervised classification performance of three model architectures: random forests classifier, long short-term memory networks with attention (LSTM-Att), and the UCSF-BERT model. Across all 13 tasks, the GPT-4 model performed either significantly better than or as well as the best supervised model, the LSTM-Att model (average macro F1 score of 0.83 vs. 0.75). On tasks with high imbalance between labels, the differences were more prominent. Frequent sources of GPT-4 errors included inferences from multiple samples and complex task design. On complex tasks where large annotated datasets cannot be easily collected, LLMs can reduce the burden of large-scale data labeling. However, if the use of LLMs is prohibitive, the use of simpler supervised models with large annotated datasets can provide comparable results. LLMs demonstrated the potential to speed up the execution of clinical NLP studies by reducing the need for curating large annotated datasets. This may result in an increase in the utilization of NLP-based variables and outcomes in observational clinical studies.

{{</citation>}}


### (34/122) Unmasking and Quantifying Racial Bias of Large Language Models in Medical Report Generation (Yifan Yang et al., 2024)

{{<citation>}}

Yifan Yang, Xiaoyu Liu, Qiao Jin, Furong Huang, Zhiyong Lu. (2024)  
**Unmasking and Quantifying Racial Bias of Large Language Models in Medical Report Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13867v1)  

---


**ABSTRACT**  
Large language models like GPT-3.5-turbo and GPT-4 hold promise for healthcare professionals, but they may inadvertently inherit biases during their training, potentially affecting their utility in medical applications. Despite few attempts in the past, the precise impact and extent of these biases remain uncertain. Through both qualitative and quantitative analyses, we find that these models tend to project higher costs and longer hospitalizations for White populations and exhibit optimistic views in challenging medical scenarios with much higher survival rates. These biases, which mirror real-world healthcare disparities, are evident in the generation of patient backgrounds, the association of specific diseases with certain races, and disparities in treatment recommendations, etc. Our findings underscore the critical need for future research to address and mitigate biases in language models, especially in critical healthcare applications, to ensure fair and accurate outcomes for all patients.

{{</citation>}}


## cs.CR (3)



### (35/122) Decentralized Federated Learning: A Survey on Security and Privacy (Ehsan Hallaji et al., 2024)

{{<citation>}}

Ehsan Hallaji, Roozbeh Razavi-Far, Mehrdad Saif, Boyu Wang, Qiang Yang. (2024)  
**Decentralized Federated Learning: A Survey on Security and Privacy**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs.CR, stat-ML  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.17319v1)  

---


**ABSTRACT**  
Federated learning has been rapidly evolving and gaining popularity in recent years due to its privacy-preserving features, among other advantages. Nevertheless, the exchange of model updates and gradients in this architecture provides new attack surfaces for malicious users of the network which may jeopardize the model performance and user and data privacy. For this reason, one of the main motivations for decentralized federated learning is to eliminate server-related threats by removing the server from the network and compensating for it through technologies such as blockchain. However, this advantage comes at the cost of challenging the system with new privacy threats. Thus, performing a thorough security analysis in this new paradigm is necessary. This survey studies possible variations of threats and adversaries in decentralized federated learning and overviews the potential defense mechanisms. Trustability and verifiability of decentralized federated learning are also considered in this study.

{{</citation>}}


### (36/122) SunBlock: Cloudless Protection for IoT Systems (Vadim Safronov et al., 2024)

{{<citation>}}

Vadim Safronov, Anna Maria Mandalari, Daniel J. Dubois, David Choffnes, Hamed Haddadi. (2024)  
**SunBlock: Cloudless Protection for IoT Systems**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14332v1)  

---


**ABSTRACT**  
With an increasing number of Internet of Things (IoT) devices present in homes, there is a rise in the number of potential information leakage channels and their associated security threats and privacy risks. Despite a long history of attacks on IoT devices in unprotected home networks, the problem of accurate, rapid detection and prevention of such attacks remains open. Many existing IoT protection solutions are cloud-based, sometimes ineffective, and might share consumer data with unknown third parties. This paper investigates the potential for effective IoT threat detection locally, on a home router, using AI tools combined with classic rule-based traffic-filtering algorithms. Our results show that with a slight rise of router hardware resources caused by machine learning and traffic filtering logic, a typical home router instrumented with our solution is able to effectively detect risks and protect a typical home IoT network, equaling or outperforming existing popular solutions, without any effects on benign IoT functionality, and without relying on cloud services and third parties.

{{</citation>}}


### (37/122) Cyber-Twin: Digital Twin-boosted Autonomous Attack Detection for Vehicular Ad-Hoc Networks (Yagmur Yigit et al., 2024)

{{<citation>}}

Yagmur Yigit, Ioannis Panitsas, Leandros Maglaras, Leandros Tassiulas, Berk Canberk. (2024)  
**Cyber-Twin: Digital Twin-boosted Autonomous Attack Detection for Vehicular Ad-Hoc Networks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14005v1)  

---


**ABSTRACT**  
The rapid evolution of Vehicular Ad-hoc NETworks (VANETs) has ushered in a transformative era for intelligent transportation systems (ITS), significantly enhancing road safety and vehicular communication. However, the intricate and dynamic nature of VANETs presents formidable challenges, particularly in vehicle-to-infrastructure (V2I) communications. Roadside Units (RSUs), integral components of VANETs, are increasingly susceptible to cyberattacks, such as jamming and distributed denial-of-service (DDoS) attacks. These vulnerabilities pose grave risks to road safety, potentially leading to traffic congestion and vehicle malfunctions. Current approaches often struggle to effectively merge digital twin technology with Artificial Intelligence (AI) models to boost security and sustainability. Our study introduces an innovative cyber-twin framework tailored to enhance the security of RSUs in VANETs. This framework uniquely combines digital twin technology with cutting-edge AI to offer a real-time, dynamic representation of RSUs. This allows for detailed monitoring and efficient detection of threats, significantly strengthening RSU security in VANETs. Moreover, our framework makes a notable contribution to eco-friendly communication by improving the computational efficiency of RSUs, leading to increased energy efficiency and extended hardware durability. Our results show a considerable enhancement in resource management and attack detection, surpassing the performance of existing solutions. In particular, the cyber-twin framework showed a substantial reduction in RSU load and an optimal balance between resource consumption and high attack detection efficiency, with a defined twinning rate range of seventy-six to ninety per cent. These advancements underscore our commitment to developing sustainable, secure, and resilient vehicular communication systems for the future of smart cities.

{{</citation>}}


## cs.CY (6)



### (38/122) The Role of Intelligent Transportation Systems and Artificial Intelligence in Energy Efficiency and Emission Reduction (Omar Rinchi et al., 2024)

{{<citation>}}

Omar Rinchi, Ahmad Alsharoa, Ibrahem Shatnawi, Anvita Arora. (2024)  
**The Role of Intelligent Transportation Systems and Artificial Intelligence in Energy Efficiency and Emission Reduction**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14560v1)  

---


**ABSTRACT**  
Despite the technological advancements in the transportation sector, the industry continues to grapple with increasing energy consumption and vehicular emissions, which intensify environmental degradation and climate change. The inefficient management of traffic flow, the underutilization of transport network interconnectivity, and the limited implementation of artificial intelligence (AI)-driven predictive models pose significant challenges to achieving energy efficiency and emission reduction. Thus, there is a timely and critical need for an integrated, sophisticated approach that leverages intelligent transportation systems (ITSs) and AI for energy conservation and emission reduction. In this paper, we explore the role of ITSs and AI in future enhanced energy and emission reduction (EER). More specifically, we discuss the impact of sensors at different levels of ITS on improving EER. We also investigate the potential networking connections in ITSs and provide an illustration of how they improve EER. Finally, we discuss potential AI services for improved EER in the future. The findings discussed in this paper will contribute to the ongoing discussion about the vital role of ITSs and AI applications in addressing the challenges associated with achieving energy savings and emission reductions in the transportation sector. Additionally, it will provide insights for policymakers and industry professionals to enable them to develop policies and implementation plans for the integration of ITSs and AI technologies in the transportation sector.

{{</citation>}}


### (39/122) My Future with My Chatbot: A Scenario-Driven, User-Centric Approach to Anticipating AI Impacts (Kimon Kieslich et al., 2024)

{{<citation>}}

Kimon Kieslich, Natali Helberger, Nicholas Diakopoulos. (2024)  
**My Future with My Chatbot: A Scenario-Driven, User-Centric Approach to Anticipating AI Impacts**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14533v1)  

---


**ABSTRACT**  
As a general purpose technology without a concrete pre-defined purpose, personal chatbots can be used for a whole range of objectives, depending on the personal needs, contexts, and tasks of an individual, and so potentially impact a variety of values, people, and social contexts. Traditional methods of risk assessment are confronted with several challenges: the lack of a clearly defined technology purpose, the lack of a clearly defined values to orient on, the heterogeneity of uses, and the difficulty of actively engaging citizens themselves in anticipating impacts from the perspective of their individual lived realities. In this article, we leverage scenario writing at scale as a method for anticipating AI impact that is responsive to these challenges. The advantages of the scenario method are its ability to engage individual users and stimulate them to consider how chatbots are likely to affect their reality and so collect different impact scenarios depending on the cultural and societal embedding of a heterogeneous citizenship. Empirically, we tasked 106 US-citizens to write short fictional stories about the future impact (whether desirable or undesirable) of AI-based personal chatbots on individuals and society and, in addition, ask respondents to explain why these impacts are important and how they relate to their values. In the analysis process, we map those impacts and analyze them in relation to socio-demographic as well as AI-related attitudes of the scenario writers. We show that our method is effective in (1) identifying and mapping desirable and undesirable impacts of AI-based personal chatbots, (2) setting these impacts in relation to values that are important for individuals, and (3) detecting socio-demographic and AI-attitude related differences of impact anticipation.

{{</citation>}}


### (40/122) Empathy and the Right to Be an Exception: What LLMs Can and Cannot Do (William Kidder et al., 2024)

{{<citation>}}

William Kidder, Jason D'Cruz, Kush R. Varshney. (2024)  
**Empathy and the Right to Be an Exception: What LLMs Can and Cannot Do**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CL, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14523v1)  

---


**ABSTRACT**  
Advances in the performance of large language models (LLMs) have led some researchers to propose the emergence of theory of mind (ToM) in artificial intelligence (AI). LLMs can attribute beliefs, desires, intentions, and emotions, and they will improve in their accuracy. Rather than employing the characteristically human method of empathy, they learn to attribute mental states by recognizing linguistic patterns in a dataset that typically do not include that individual. We ask whether LLMs' inability to empathize precludes them from honoring an individual's right to be an exception, that is, from making assessments of character and predictions of behavior that reflect appropriate sensitivity to a person's individuality. Can LLMs seriously consider an individual's claim that their case is different based on internal mental states like beliefs, desires, and intentions, or are they limited to judging that case based on its similarities to others? We propose that the method of empathy has special significance for honoring the right to be an exception that is distinct from the value of predictive accuracy, at which LLMs excel. We conclude by considering whether using empathy to consider exceptional cases has intrinsic or merely practical value and we introduce conceptual and empirical avenues for advancing this investigation.

{{</citation>}}


### (41/122) AI auditing: The Broken Bus on the Road to AI Accountability (Abeba Birhane et al., 2024)

{{<citation>}}

Abeba Birhane, Ryan Steed, Victor Ojewale, Briana Vecchione, Inioluwa Deborah Raji. (2024)  
**AI auditing: The Broken Bus on the Road to AI Accountability**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14462v1)  

---


**ABSTRACT**  
One of the most concrete measures to take towards meaningful AI accountability is to consequentially assess and report the systems' performance and impact. However, the practical nature of the "AI audit" ecosystem is muddled and imprecise, making it difficult to work through various concepts and map out the stakeholders involved in the practice. First, we taxonomize current AI audit practices as completed by regulators, law firms, civil society, journalism, academia, consulting agencies. Next, we assess the impact of audits done by stakeholders within each domain. We find that only a subset of AI audit studies translate to desired accountability outcomes. We thus assess and isolate practices necessary for effective AI audit results, articulating the observed connections between AI audit design, methodology and institutional context on its effectiveness as a meaningful mechanism for accountability.

{{</citation>}}


### (42/122) Black-Box Access is Insufficient for Rigorous AI Audits (Stephen Casper et al., 2024)

{{<citation>}}

Stephen Casper, Carson Ezell, Charlotte Siegmann, Noam Kolt, Taylor Lynn Curtis, Benjamin Bucknall, Andreas Haupt, Kevin Wei, Jérémy Scheurer, Marius Hobbhahn, Lee Sharkey, Satyapriya Krishna, Marvin Von Hagen, Silas Alberti, Alan Chan, Qinyi Sun, Michael Gerovitch, David Bau, Max Tegmark, David Krueger, Dylan Hadfield-Menell. (2024)  
**Black-Box Access is Insufficient for Rigorous AI Audits**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CR, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14446v1)  

---


**ABSTRACT**  
External audits of AI systems are increasingly recognized as a key mechanism for AI governance. The effectiveness of an audit, however, depends on the degree of system access granted to auditors. Recent audits of state-of-the-art AI systems have primarily relied on black-box access, in which auditors can only query the system and observe its outputs. However, white-box access to the system's inner workings (e.g., weights, activations, gradients) allows an auditor to perform stronger attacks, more thoroughly interpret models, and conduct fine-tuning. Meanwhile, outside-the-box access to its training and deployment information (e.g., methodology, code, documentation, hyperparameters, data, deployment details, findings from internal evaluations) allows for auditors to scrutinize the development process and design more targeted evaluations. In this paper, we examine the limitations of black-box audits and the advantages of white- and outside-the-box audits. We also discuss technical, physical, and legal safeguards for performing these audits with minimal security risks. Given that different forms of access can lead to very different levels of evaluation, we conclude that (1) transparency regarding the access and methods used by auditors is necessary to properly interpret audit results, and (2) white- and outside-the-box access allow for substantially more scrutiny than black-box access alone.

{{</citation>}}


### (43/122) On mission Twitter Profiles: A Study of Selective Toxic Behavior (Hina Qayyum et al., 2024)

{{<citation>}}

Hina Qayyum, Muhammad Ikram, Benjamin Zi Hao Zhao, an D. Wood, Nicolas Kourtellis, Mohamed Ali Kaafar. (2024)  
**On mission Twitter Profiles: A Study of Selective Toxic Behavior**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.14252v1)  

---


**ABSTRACT**  
The argument for persistent social media influence campaigns, often funded by malicious entities, is gaining traction. These entities utilize instrumented profiles to disseminate divisive content and disinformation, shaping public perception. Despite ample evidence of these instrumented profiles, few identification methods exist to locate them in the wild. To evade detection and appear genuine, small clusters of instrumented profiles engage in unrelated discussions, diverting attention from their true goals. This strategic thematic diversity conceals their selective polarity towards certain topics and fosters public trust.   This study aims to characterize profiles potentially used for influence operations, termed 'on-mission profiles,' relying solely on thematic content diversity within unlabeled data. Distinguishing this work is its focus on content volume and toxicity towards specific themes. Longitudinal data from 138K Twitter or X, profiles and 293M tweets enables profiling based on theme diversity. High thematic diversity groups predominantly produce toxic content concerning specific themes, like politics, health, and news classifying them as 'on-mission' profiles.   Using the identified ``on-mission" profiles, we design a classifier for unseen, unlabeled data. Employing a linear SVM model, we train and test it on an 80/20% split of the most diverse profiles. The classifier achieves a flawless 100% accuracy, facilitating the discovery of previously unknown ``on-mission" profiles in the wild.

{{</citation>}}


## cs.CV (17)



### (44/122) Revisiting Active Learning in the Era of Vision Foundation Models (Sanket Rajan Gupte et al., 2024)

{{<citation>}}

Sanket Rajan Gupte, Josiah Aklilu, Jeffrey J. Nirschl, Serena Yeung-Levy. (2024)  
**Revisiting Active Learning in the Era of Vision Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2401.14555v1)  

---


**ABSTRACT**  
Foundation vision or vision-language models are trained on large unlabeled or noisy data and learn robust representations that can achieve impressive zero- or few-shot performance on diverse tasks. Given these properties, they are a natural fit for active learning (AL), which aims to maximize labeling efficiency, but the full potential of foundation models has not been explored in the context of AL, specifically in the low-budget regime. In this work, we evaluate how foundation models influence three critical components of effective AL, namely, 1) initial labeled pool selection, 2) ensuring diverse sampling, and 3) the trade-off between representative and uncertainty sampling. We systematically study how the robust representations of foundation models (DINOv2, OpenCLIP) challenge existing findings in active learning. Our observations inform the principled construction of a new simple and elegant AL strategy that balances uncertainty estimated via dropout with sample diversity. We extensively test our strategy on many challenging image classification benchmarks, including natural images as well as out-of-domain biomedical images that are relatively understudied in the AL literature. Source code will be made available.

{{</citation>}}


### (45/122) Towards Global Glacier Mapping with Deep Learning and Open Earth Observation Data (Konstantin A. Maslov et al., 2024)

{{<citation>}}

Konstantin A. Maslov, Claudio Persello, Thomas Schellenberger, Alfred Stein. (2024)  
**Towards Global Glacier Mapping with Deep Learning and Open Earth Observation Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.15113v1)  

---


**ABSTRACT**  
Accurate global glacier mapping is critical for understanding climate change impacts. It is challenged by glacier diversity, difficult-to-classify debris and big data processing. Here we propose Glacier-VisionTransformer-U-Net (GlaViTU), a convolutional-transformer deep learning model, and five strategies for multitemporal global-scale glacier mapping using open satellite imagery. Assessing the spatial, temporal and cross-sensor generalisation shows that our best strategy achieves intersection over union >0.85 on previously unobserved images in most cases, which drops to >0.75 for debris-rich areas such as High-Mountain Asia and increases to >0.90 for regions dominated by clean ice. Additionally, adding synthetic aperture radar data, namely, backscatter and interferometric coherence, increases the accuracy in all regions where available. The calibrated confidence for glacier extents is reported making the predictions more reliable and interpretable. We also release a benchmark dataset that covers 9% of glaciers worldwide. Our results support efforts towards automated multitemporal and global glacier mapping.

{{</citation>}}


### (46/122) Multimodal Pathway: Improve Transformers with Irrelevant Data from Other Modalities (Yiyuan Zhang et al., 2024)

{{<citation>}}

Yiyuan Zhang, Xiaohan Ding, Kaixiong Gong, Yixiao Ge, Ying Shan, Xiangyu Yue. (2024)  
**Multimodal Pathway: Improve Transformers with Irrelevant Data from Other Modalities**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.14405v1)  

---


**ABSTRACT**  
We propose to improve transformers of a specific modality with irrelevant data from other modalities, e.g., improve an ImageNet model with audio or point cloud datasets. We would like to highlight that the data samples of the target modality are irrelevant to the other modalities, which distinguishes our method from other works utilizing paired (e.g., CLIP) or interleaved data of different modalities. We propose a methodology named Multimodal Pathway - given a target modality and a transformer designed for it, we use an auxiliary transformer trained with data of another modality and construct pathways to connect components of the two models so that data of the target modality can be processed by both models. In this way, we utilize the universal sequence-to-sequence modeling abilities of transformers obtained from two modalities. As a concrete implementation, we use a modality-specific tokenizer and task-specific head as usual but utilize the transformer blocks of the auxiliary model via a proposed method named Cross-Modal Re-parameterization, which exploits the auxiliary weights without any inference costs. On the image, point cloud, video, and audio recognition tasks, we observe significant and consistent performance improvements with irrelevant data from other modalities. The code and models are available at https://github.com/AILab-CVC/M2PT.

{{</citation>}}


### (47/122) Deconstructing Denoising Diffusion Models for Self-Supervised Learning (Xinlei Chen et al., 2024)

{{<citation>}}

Xinlei Chen, Zhuang Liu, Saining Xie, Kaiming He. (2024)  
**Deconstructing Denoising Diffusion Models for Self-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.14404v1)  

---


**ABSTRACT**  
In this study, we examine the representation learning abilities of Denoising Diffusion Models (DDM) that were originally purposed for image generation. Our philosophy is to deconstruct a DDM, gradually transforming it into a classical Denoising Autoencoder (DAE). This deconstructive procedure allows us to explore how various components of modern DDMs influence self-supervised representation learning. We observe that only a very few modern components are critical for learning good representations, while many others are nonessential. Our study ultimately arrives at an approach that is highly simplified and to a large extent resembles a classical DAE. We hope our study will rekindle interest in a family of classical methods within the realm of modern self-supervised learning.

{{</citation>}}


### (48/122) Rethinking Patch Dependence for Masked Autoencoders (Letian Fu et al., 2024)

{{<citation>}}

Letian Fu, Long Lian, Renhao Wang, Baifeng Shi, Xudong Wang, Adam Yala, Trevor Darrell, Alexei A. Efros, Ken Goldberg. (2024)  
**Rethinking Patch Dependence for Masked Autoencoders**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, ImageNet  
[Paper Link](http://arxiv.org/abs/2401.14391v1)  

---


**ABSTRACT**  
In this work, we re-examine inter-patch dependencies in the decoding mechanism of masked autoencoders (MAE). We decompose this decoding mechanism for masked patch reconstruction in MAE into self-attention and cross-attention. Our investigations suggest that self-attention between mask patches is not essential for learning good representations. To this end, we propose a novel pretraining framework: Cross-Attention Masked Autoencoders (CrossMAE). CrossMAE's decoder leverages only cross-attention between masked and visible tokens, with no degradation in downstream performance. This design also enables decoding only a small subset of mask tokens, boosting efficiency. Furthermore, each decoder block can now leverage different encoder features, resulting in improved representation learning. CrossMAE matches MAE in performance with 2.5 to 3.7$\times$ less decoding compute. It also surpasses MAE on ImageNet classification and COCO instance segmentation under the same compute. Code and models: https://crossmae.github.io

{{</citation>}}


### (49/122) UrbanGenAI: Reconstructing Urban Landscapes using Panoptic Segmentation and Diffusion Models (Timo Kapsalis, 2024)

{{<citation>}}

Timo Kapsalis. (2024)  
**UrbanGenAI: Reconstructing Urban Landscapes using Panoptic Segmentation and Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14379v1)  

---


**ABSTRACT**  
In contemporary design practices, the integration of computer vision and generative artificial intelligence (genAI) represents a transformative shift towards more interactive and inclusive processes. These technologies offer new dimensions of image analysis and generation, which are particularly relevant in the context of urban landscape reconstruction. This paper presents a novel workflow encapsulated within a prototype application, designed to leverage the synergies between advanced image segmentation and diffusion models for a comprehensive approach to urban design. Our methodology encompasses the OneFormer model for detailed image segmentation and the Stable Diffusion XL (SDXL) diffusion model, implemented through ControlNet, for generating images from textual descriptions. Validation results indicated a high degree of performance by the prototype application, showcasing significant accuracy in both object detection and text-to-image generation. This was evidenced by superior Intersection over Union (IoU) and CLIP scores across iterative evaluations for various categories of urban landscape features. Preliminary testing included utilising UrbanGenAI as an educational tool enhancing the learning experience in design pedagogy, and as a participatory instrument facilitating community-driven urban planning. Early results suggested that UrbanGenAI not only advances the technical frontiers of urban landscape reconstruction but also provides significant pedagogical and participatory planning benefits. The ongoing development of UrbanGenAI aims to further validate its effectiveness across broader contexts and integrate additional features such as real-time feedback mechanisms and 3D modelling capabilities. Keywords: generative AI; panoptic image segmentation; diffusion models; urban landscape design; design pedagogy; co-design

{{</citation>}}


### (50/122) Unlocking Past Information: Temporal Embeddings in Cooperative Bird's Eye View Prediction (Dominik Rößle et al., 2024)

{{<citation>}}

Dominik Rößle, Jeremias Gerner, Klaus Bogenberger, Daniel Cremers, Stefanie Schmidtner, Torsten Schön. (2024)  
**Unlocking Past Information: Temporal Embeddings in Cooperative Bird's Eye View Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.14325v1)  

---


**ABSTRACT**  
Accurate and comprehensive semantic segmentation of Bird's Eye View (BEV) is essential for ensuring safe and proactive navigation in autonomous driving. Although cooperative perception has exceeded the detection capabilities of single-agent systems, prevalent camera-based algorithms in cooperative perception neglect valuable information derived from historical observations. This limitation becomes critical during sensor failures or communication issues as cooperative perception reverts to single-agent perception, leading to degraded performance and incomplete BEV segmentation maps. This paper introduces TempCoBEV, a temporal module designed to incorporate historical cues into current observations, thereby improving the quality and reliability of BEV map segmentations. We propose an importance-guided attention architecture to effectively integrate temporal information that prioritizes relevant properties for BEV map segmentation. TempCoBEV is an independent temporal module that seamlessly integrates into state-of-the-art camera-based cooperative perception models. We demonstrate through extensive experiments on the OPV2V dataset that TempCoBEV performs better than non-temporal models in predicting current and future BEV map segmentations, particularly in scenarios involving communication failures. We show the efficacy of TempCoBEV and its capability to integrate historical cues into the current BEV map, improving predictions under optimal communication conditions by up to 2% and under communication failures by up to 19%. The code will be published on GitHub.

{{</citation>}}


### (51/122) Sketch2NeRF: Multi-view Sketch-guided Text-to-3D Generation (Minglin Chen et al., 2024)

{{<citation>}}

Minglin Chen, Weihao Yuan, Yukun Wang, Zhe Sheng, Yisheng He, Zilong Dong, Liefeng Bo, Yulan Guo. (2024)  
**Sketch2NeRF: Multi-view Sketch-guided Text-to-3D Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2401.14257v2)  

---


**ABSTRACT**  
Recently, text-to-3D approaches have achieved high-fidelity 3D content generation using text description. However, the generated objects are stochastic and lack fine-grained control. Sketches provide a cheap approach to introduce such fine-grained control. Nevertheless, it is challenging to achieve flexible control from these sketches due to their abstraction and ambiguity. In this paper, we present a multi-view sketch-guided text-to-3D generation framework (namely, Sketch2NeRF) to add sketch control to 3D generation. Specifically, our method leverages pretrained 2D diffusion models (e.g., Stable Diffusion and ControlNet) to supervise the optimization of a 3D scene represented by a neural radiance field (NeRF). We propose a novel synchronized generation and reconstruction method to effectively optimize the NeRF. In the experiments, we collected two kinds of multi-view sketch datasets to evaluate the proposed method. We demonstrate that our method can synthesize 3D consistent contents with fine-grained sketch control while being high-fidelity to text prompts. Extensive results show that our method achieves state-of-the-art performance in terms of sketch similarity and text alignment.

{{</citation>}}


### (52/122) Exploring the Unexplored: Understanding the Impact of Layer Adjustments on Image Classification (Haixia Liu et al., 2024)

{{<citation>}}

Haixia Liu, Tim Brailsford, James Goulding, Gavin Smith, Larry Bull. (2024)  
**Exploring the Unexplored: Understanding the Impact of Layer Adjustments on Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2401.14236v1)  

---


**ABSTRACT**  
This paper investigates how adjustments to deep learning architectures impact model performance in image classification. Small-scale experiments generate initial insights although the trends observed are not consistent with the entire dataset. Filtering operations in the image processing pipeline are crucial, with image filtering before pre-processing yielding better results. The choice and order of layers as well as filter placement significantly impact model performance. This study provides valuable insights into optimizing deep learning models, with potential avenues for future research including collaborative platforms.

{{</citation>}}


### (53/122) Vivim: a Video Vision Mamba for Medical Video Object Segmentation (Yijun Yang et al., 2024)

{{<citation>}}

Yijun Yang, Zhaohu Xing, Lei Zhu. (2024)  
**Vivim: a Video Vision Mamba for Medical Video Object Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.14168v1)  

---


**ABSTRACT**  
Traditional convolutional neural networks have a limited receptive field while transformer-based networks are mediocre in constructing long-term dependency from the perspective of computational complexity. Such the bottleneck poses a significant challenge when processing long video sequences in video analysis tasks. Very recently, the state space models (SSMs) with efficient hardware-aware designs, famous by Mamba, have exhibited impressive achievements in long sequence modeling, which facilitates the development of deep neural networks on many vision tasks. To better capture available cues in video frames, this paper presents a generic Video Vision Mamba-based framework for medical video object segmentation tasks, named Vivim. Our Vivim can effectively compress the long-term spatiotemporal representation into sequences at varying scales by our designed Temporal Mamba Block. Compared to existing video-level Transformer-based methods, our model maintains excellent segmentation results with better speed performance. Extensive experiments on the breast US dataset demonstrate the effectiveness and efficiency of our Vivim. The code for Vivim is available at: https://github.com/scott-yjyang/Vivim.

{{</citation>}}


### (54/122) Transforming gradient-based techniques into interpretable methods (Caroline Mazini Rodrigues et al., 2024)

{{<citation>}}

Caroline Mazini Rodrigues, Nicolas Boutry, Laurent Najman. (2024)  
**Transforming gradient-based techniques into interpretable methods**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14434v1)  

---


**ABSTRACT**  
The explication of Convolutional Neural Networks (CNN) through xAI techniques often poses challenges in interpretation. The inherent complexity of input features, notably pixels extracted from images, engenders complex correlations. Gradient-based methodologies, exemplified by Integrated Gradients (IG), effectively demonstrate the significance of these features. Nevertheless, the conversion of these explanations into images frequently yields considerable noise. Presently, we introduce GAD (Gradient Artificial Distancing) as a supportive framework for gradient-based techniques. Its primary objective is to accentuate influential regions by establishing distinctions between classes. The essence of GAD is to limit the scope of analysis during visualization and, consequently reduce image noise. Empirical investigations involving occluded images have demonstrated that the identified regions through this methodology indeed play a pivotal role in facilitating class differentiation.

{{</citation>}}


### (55/122) Diffusion-based Data Augmentation for Object Counting Problems (Zhen Wang et al., 2024)

{{<citation>}}

Zhen Wang, Yuelei Li, Jia Wan, Nuno Vasconcelos. (2024)  
**Diffusion-based Data Augmentation for Object Counting Problems**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.13992v1)  

---


**ABSTRACT**  
Crowd counting is an important problem in computer vision due to its wide range of applications in image understanding. Currently, this problem is typically addressed using deep learning approaches, such as Convolutional Neural Networks (CNNs) and Transformers. However, deep networks are data-driven and are prone to overfitting, especially when the available labeled crowd dataset is limited. To overcome this limitation, we have designed a pipeline that utilizes a diffusion model to generate extensive training data. We are the first to generate images conditioned on a location dot map (a binary dot map that specifies the location of human heads) with a diffusion model. We are also the first to use these diverse synthetic data to augment the crowd counting models. Our proposed smoothed density map input for ControlNet significantly improves ControlNet's performance in generating crowds in the correct locations. Also, Our proposed counting loss for the diffusion model effectively minimizes the discrepancies between the location dot map and the crowd images generated. Additionally, our innovative guidance sampling further directs the diffusion process toward regions where the generated crowd images align most accurately with the location dot map. Collectively, we have enhanced ControlNet's ability to generate specified objects from a location dot map, which can be used for data augmentation in various counting problems. Moreover, our framework is versatile and can be easily adapted to all kinds of counting problems. Extensive experiments demonstrate that our framework improves the counting performance on the ShanghaiTech, NWPU-Crowd, UCF-QNRF, and TRANCOS datasets, showcasing its effectiveness.

{{</citation>}}


### (56/122) Improving Pseudo-labelling and Enhancing Robustness for Semi-Supervised Domain Generalization (Adnan Khan et al., 2024)

{{<citation>}}

Adnan Khan, Mai A. Shaaban, Muhammad Haris Khan. (2024)  
**Improving Pseudo-labelling and Enhancing Robustness for Semi-Supervised Domain Generalization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2401.13965v1)  

---


**ABSTRACT**  
Beyond attaining domain generalization (DG), visual recognition models should also be data-efficient during learning by leveraging limited labels. We study the problem of Semi-Supervised Domain Generalization (SSDG) which is crucial for real-world applications like automated healthcare. SSDG requires learning a cross-domain generalizable model when the given training data is only partially labelled. Empirical investigations reveal that the DG methods tend to underperform in SSDG settings, likely because they are unable to exploit the unlabelled data. Semi-supervised learning (SSL) shows improved but still inferior results compared to fully-supervised learning. A key challenge, faced by the best-performing SSL-based SSDG methods, is selecting accurate pseudo-labels under multiple domain shifts and reducing overfitting to source domains under limited labels. In this work, we propose new SSDG approach, which utilizes a novel uncertainty-guided pseudo-labelling with model averaging (UPLM). Our uncertainty-guided pseudo-labelling (UPL) uses model uncertainty to improve pseudo-labelling selection, addressing poor model calibration under multi-source unlabelled data. The UPL technique, enhanced by our novel model averaging (MA) strategy, mitigates overfitting to source domains with limited labels. Extensive experiments on key representative DG datasets suggest that our method demonstrates effectiveness against existing methods. Our code and chosen labelled data seeds are available on GitHub: https://github.com/Adnan-Khan7/UPLM

{{</citation>}}


### (57/122) An Extensible Framework for Open Heterogeneous Collaborative Perception (Yifan Lu et al., 2024)

{{<citation>}}

Yifan Lu, Yue Hu, Yiqi Zhong, Dequan Wang, Siheng Chen, Yanfeng Wang. (2024)  
**An Extensible Framework for Open Heterogeneous Collaborative Perception**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13964v1)  

---


**ABSTRACT**  
Collaborative perception aims to mitigate the limitations of single-agent perception, such as occlusions, by facilitating data exchange among multiple agents. However, most current works consider a homogeneous scenario where all agents use identity sensors and perception models. In reality, heterogeneous agent types may continually emerge and inevitably face a domain gap when collaborating with existing agents. In this paper, we introduce a new open heterogeneous problem: how to accommodate continually emerging new heterogeneous agent types into collaborative perception, while ensuring high perception performance and low integration cost? To address this problem, we propose HEterogeneous ALliance (HEAL), a novel extensible collaborative perception framework. HEAL first establishes a unified feature space with initial agents via a novel multi-scale foreground-aware Pyramid Fusion network. When heterogeneous new agents emerge with previously unseen modalities or models, we align them to the established unified space with an innovative backward alignment. This step only involves individual training on the new agent type, thus presenting extremely low training costs and high extensibility. It also protects new agents' model details from disclosure since the training can be conducted by the agent owner locally. To enrich agents' data heterogeneity, we bring OPV2V-H, a new large-scale dataset with more diverse sensor types. Extensive experiments on OPV2V-H and DAIR-V2X datasets show that HEAL surpasses SOTA methods in performance while reducing the training parameters by 91.5% when integrating 3 new agent types. Code and data are available at: https://github.com/yifanlu0227/HEAL.

{{</citation>}}


### (58/122) AM-SORT: Adaptable Motion Predictor with Historical Trajectory Embedding for Multi-Object Tracking (Vitaliy Kim et al., 2024)

{{<citation>}}

Vitaliy Kim, Gunho Jung, Seong-Whan Lee. (2024)  
**AM-SORT: Adaptable Motion Predictor with Historical Trajectory Embedding for Multi-Object Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.13950v1)  

---


**ABSTRACT**  
Many multi-object tracking (MOT) approaches, which employ the Kalman Filter as a motion predictor, assume constant velocity and Gaussian-distributed filtering noises. These assumptions render the Kalman Filter-based trackers effective in linear motion scenarios. However, these linear assumptions serve as a key limitation when estimating future object locations within scenarios involving non-linear motion and occlusions. To address this issue, we propose a motion-based MOT approach with an adaptable motion predictor, called AM-SORT, which adapts to estimate non-linear uncertainties. AM-SORT is a novel extension of the SORT-series trackers that supersedes the Kalman Filter with the transformer architecture as a motion predictor. We introduce a historical trajectory embedding that empowers the transformer to extract spatio-temporal features from a sequence of bounding boxes. AM-SORT achieves competitive performance compared to state-of-the-art trackers on DanceTrack, with 56.3 IDF1 and 55.6 HOTA. We conduct extensive experiments to demonstrate the effectiveness of our method in predicting non-linear movement under occlusions.

{{</citation>}}


### (59/122) Self-supervised Video Object Segmentation with Distillation Learning of Deformable Attention (Quang-Trung Truong et al., 2024)

{{<citation>}}

Quang-Trung Truong, Duc Thanh Nguyen, Binh-Son Hua, Sai-Kit Yeung. (2024)  
**Self-supervised Video Object Segmentation with Distillation Learning of Deformable Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.13937v1)  

---


**ABSTRACT**  
Video object segmentation is a fundamental research problem in computer vision. Recent techniques have often applied attention mechanism to object representation learning from video sequences. However, due to temporal changes in the video data, attention maps may not well align with the objects of interest across video frames, causing accumulated errors in long-term video processing. In addition, existing techniques have utilised complex architectures, requiring highly computational complexity and hence limiting the ability to integrate video object segmentation into low-powered devices. To address these issues, we propose a new method for self-supervised video object segmentation based on distillation learning of deformable attention. Specifically, we devise a lightweight architecture for video object segmentation that is effectively adapted to temporal changes. This is enabled by deformable attention mechanism, where the keys and values capturing the memory of a video sequence in the attention module have flexible locations updated across frames. The learnt object representations are thus adaptive to both the spatial and temporal dimensions. We train the proposed architecture in a self-supervised fashion through a new knowledge distillation paradigm where deformable attention maps are integrated into the distillation loss. We qualitatively and quantitatively evaluate our method and compare it with existing methods on benchmark datasets including DAVIS 2016/2017 and YouTube-VOS 2018/2019. Experimental results verify the superiority of our method via its achieved state-of-the-art performance and optimal memory usage.

{{</citation>}}


### (60/122) Knowledge Graph Supported Benchmark and Video Captioning for Basketball (Zeyu Xi et al., 2024)

{{<citation>}}

Zeyu Xi, Ge Shi, Lifang Wu, Xuefen Li, Junchi Yan, Liang Wang, Zilin Liu. (2024)  
**Knowledge Graph Supported Benchmark and Video Captioning for Basketball**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.13888v1)  

---


**ABSTRACT**  
Despite the recent emergence of video captioning models, how to generate the text description with specific entity names and fine-grained actions is far from being solved, which however has great applications such as basketball live text broadcast. In this paper, a new multimodal knowledge supported basketball benchmark for video captioning is proposed. Specifically, we construct a Multimodal Basketball Game Knowledge Graph (MbgKG) to provide knowledge beyond videos. Then, a Multimodal Basketball Game Video Captioning (MbgVC) dataset that contains 9 types of fine-grained shooting events and 286 players' knowledge (i.e., images and names) is constructed based on MbgKG. We develop a novel framework in the encoder-decoder form named Entity-Aware Captioner (EAC) for basketball live text broadcast. The temporal information in video is encoded by introducing the bi-directional GRU (Bi-GRU) module. And the multi-head self-attention module is utilized to model the relationships among the players and select the key players. Besides, we propose a new performance evaluation metric named Game Description Score (GDS), which measures not only the linguistic performance but also the accuracy of the names prediction. Extensive experiments on MbgVC dataset demonstrate that EAC effectively leverages external knowledge and outperforms advanced video captioning models. The proposed benchmark and corresponding codes will be publicly available soon.

{{</citation>}}


## cs.RO (5)



### (61/122) GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multi-Agent Control (Songyuan Zhang et al., 2024)

{{<citation>}}

Songyuan Zhang, Oswin So, Kunal Garg, Chuchu Fan. (2024)  
**GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multi-Agent Control**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO, math-OC  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.14554v1)  

---


**ABSTRACT**  
Distributed, scalable, and safe control of large-scale multi-agent systems (MAS) is a challenging problem. In this paper, we design a distributed framework for safe multi-agent control in large-scale environments with obstacles, where a large number of agents are required to maintain safety using only local information and reach their goal locations. We introduce a new class of certificates, termed graph control barrier function (GCBF), which are based on the well-established control barrier function (CBF) theory for safety guarantees and utilize a graph structure for scalable and generalizable distributed control of MAS. We develop a novel theoretical framework to prove the safety of an arbitrary-sized MAS with a single GCBF. We propose a new training framework GCBF+ that uses graph neural networks (GNNs) to parameterize a candidate GCBF and a distributed control policy. The proposed framework is distributed and is capable of directly taking point clouds from LiDAR, instead of actual state information, for real-world robotic applications. We illustrate the efficacy of the proposed method through various hardware experiments on a swarm of drones with objectives ranging from exchanging positions to docking on a moving target without collision. Additionally, we perform extensive numerical experiments, where the number and density of agents, as well as the number of obstacles, increase. Empirical results show that in complex environments with nonlinear agents (e.g., Crazyflie drones) GCBF+ outperforms the handcrafted CBF-based method with the best performance by up to 20% for relatively small-scale MAS for up to 256 agents, and leading reinforcement learning (RL) methods by up to 40% for MAS with 1024 agents. Furthermore, the proposed method does not compromise on the performance, in terms of goal reaching, for achieving high safety rates, which is a common trade-off in RL-based methods.

{{</citation>}}


### (62/122) Machine Learning for Shipwreck Segmentation from Side Scan Sonar Imagery: Dataset and Benchmark (Advaith V. Sethuraman et al., 2024)

{{<citation>}}

Advaith V. Sethuraman, Anja Sheppard, Onur Bagoren, Christopher Pinnow, Jamey Anderson, Timothy C. Havens, Katherine A. Skinner. (2024)  
**Machine Learning for Shipwreck Segmentation from Side Scan Sonar Imagery: Dataset and Benchmark**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14546v1)  

---


**ABSTRACT**  
Open-source benchmark datasets have been a critical component for advancing machine learning for robot perception in terrestrial applications. Benchmark datasets enable the widespread development of state-of-the-art machine learning methods, which require large datasets for training, validation, and thorough comparison to competing approaches. Underwater environments impose several operational challenges that hinder efforts to collect large benchmark datasets for marine robot perception. Furthermore, a low abundance of targets of interest relative to the size of the search space leads to increased time and cost required to collect useful datasets for a specific task. As a result, there is limited availability of labeled benchmark datasets for underwater applications. We present the AI4Shipwrecks dataset, which consists of 24 distinct shipwreck sites totaling 286 high-resolution labeled side scan sonar images to advance the state-of-the-art in autonomous sonar image understanding. We leverage the unique abundance of targets in Thunder Bay National Marine Sanctuary in Lake Huron, MI, to collect and compile a sonar imagery benchmark dataset through surveys with an autonomous underwater vehicle (AUV). We consulted with expert marine archaeologists for the labeling of robotically gathered data. We then leverage this dataset to perform benchmark experiments for comparison of state-of-the-art supervised segmentation methods, and we present insights on opportunities and open challenges for the field. The dataset and benchmarking tools will be released as an open-source benchmark dataset to spur innovation in machine learning for Great Lakes and ocean exploration. The dataset and accompanying software are available at https://umfieldrobotics.github.io/ai4shipwrecks/.

{{</citation>}}


### (63/122) MResT: Multi-Resolution Sensing for Real-Time Control with Vision-Language Models (Saumya Saxena et al., 2024)

{{<citation>}}

Saumya Saxena, Mohit Sharma, Oliver Kroemer. (2024)  
**MResT: Multi-Resolution Sensing for Real-Time Control with Vision-Language Models**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.14502v1)  

---


**ABSTRACT**  
Leveraging sensing modalities across diverse spatial and temporal resolutions can improve performance of robotic manipulation tasks. Multi-spatial resolution sensing provides hierarchical information captured at different spatial scales and enables both coarse and precise motions. Simultaneously multi-temporal resolution sensing enables the agent to exhibit high reactivity and real-time control. In this work, we propose a framework, MResT (Multi-Resolution Transformer), for learning generalizable language-conditioned multi-task policies that utilize sensing at different spatial and temporal resolutions using networks of varying capacities to effectively perform real time control of precise and reactive tasks. We leverage off-the-shelf pretrained vision-language models to operate on low-frequency global features along with small non-pretrained models to adapt to high frequency local feedback. Through extensive experiments in 3 domains (coarse, precise and dynamic manipulation tasks), we show that our approach significantly improves (2X on average) over recent multi-task baselines. Further, our approach generalizes well to visual and geometric variations in target objects and to varying interaction forces.

{{</citation>}}


### (64/122) Learning to navigate efficiently and precisely in real environments (Guillaume Bono et al., 2024)

{{<citation>}}

Guillaume Bono, Hervé Poirier, Leonid Antsfeld, Gianluca Monaci, Boris Chidlovskii, Christian Wolf. (2024)  
**Learning to navigate efficiently and precisely in real environments**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14349v1)  

---


**ABSTRACT**  
In the context of autonomous navigation of terrestrial robots, the creation of realistic models for agent dynamics and sensing is a widespread habit in the robotics literature and in commercial applications, where they are used for model based control and/or for localization and mapping. The more recent Embodied AI literature, on the other hand, focuses on modular or end-to-end agents trained in simulators like Habitat or AI-Thor, where the emphasis is put on photo-realistic rendering and scene diversity, but high-fidelity robot motion is assigned a less privileged role. The resulting sim2real gap significantly impacts transfer of the trained models to real robotic platforms. In this work we explore end-to-end training of agents in simulation in settings which minimize the sim2real gap both, in sensing and in actuation. Our agent directly predicts (discretized) velocity commands, which are maintained through closed-loop control in the real robot. The behavior of the real robot (including the underlying low-level controller) is identified and simulated in a modified Habitat simulator. Noise models for odometry and localization further contribute in lowering the sim2real gap. We evaluate on real navigation scenarios, explore different localization and point goal calculation methods and report significant gains in performance and robustness compared to prior work.

{{</citation>}}


### (65/122) Concept: Dynamic Risk Assessment for AI-Controlled Robotic Systems (Philipp Grimmeisen et al., 2024)

{{<citation>}}

Philipp Grimmeisen, Friedrich Sautter, Andrey Morozov. (2024)  
**Concept: Dynamic Risk Assessment for AI-Controlled Robotic Systems**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.14147v1)  

---


**ABSTRACT**  
AI-controlled robotic systems pose a risk to human workers and the environment. Classical risk assessment methods cannot adequately describe such black box systems. Therefore, new methods for a dynamic risk assessment of such AI-controlled systems are required. In this paper, we introduce the concept of a new dynamic risk assessment approach for AI-controlled robotic systems. The approach pipelines five blocks: (i) a Data Logging that logs the data of the given simulation, (ii) a Skill Detection that automatically detects the executed skills with a deep learning technique, (iii) a Behavioral Analysis that creates the behavioral profile of the robotic systems, (iv) a Risk Model Generation that automatically transforms the behavioral profile and risk data containing the failure probabilities of robotic hardware components into advanced hybrid risk models, and (v) Risk Model Solvers for the numerical evaluation of the generated hybrid risk models.   Keywords: Dynamic Risk Assessment, Hybrid Risk Models, M2M Transformation, ROS, AI-Controlled Robotic Systems, Deep Learning, Reinforcement Learning

{{</citation>}}


## cs.SD (1)



### (66/122) Exploring Musical Roots: Applying Audio Embeddings to Empower Influence Attribution for a Generative Music Model (Julia Barnett et al., 2024)

{{<citation>}}

Julia Barnett, Hugo Flores Garcia, Bryan Pardo. (2024)  
**Exploring Musical Roots: Applying Audio Embeddings to Empower Influence Attribution for a Generative Music Model**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.14542v1)  

---


**ABSTRACT**  
Every artist has a creative process that draws inspiration from previous artists and their works. Today, "inspiration" has been automated by generative music models. The black box nature of these models obscures the identity of the works that influence their creative output. As a result, users may inadvertently appropriate, misuse, or copy existing artists' works. We establish a replicable methodology to systematically identify similar pieces of music audio in a manner that is useful for understanding training data attribution. A key aspect of our approach is to harness an effective music audio similarity measure. We compare the effect of applying CLMR and CLAP embeddings to similarity measurement in a set of 5 million audio clips used to train VampNet, a recent open source generative music model. We validate this approach with a human listening study. We also explore the effect that modifications of an audio example (e.g., pitch shifting, time stretching, background noise) have on similarity measurements. This work is foundational to incorporating automated influence attribution into generative modeling, which promises to let model creators and users move from ignorant appropriation to informed creation. Audio samples that accompany this paper are available at https://tinyurl.com/exploring-musical-roots.

{{</citation>}}


## eess.IV (3)



### (67/122) Improving Fairness of Automated Chest X-ray Diagnosis by Contrastive Learning (Mingquan Lin et al., 2024)

{{<citation>}}

Mingquan Lin, Tianhao Li, Zhaoyi Sun, Gregory Holste, Ying Ding, Fei Wang, George Shih, Yifan Peng. (2024)  
**Improving Fairness of Automated Chest X-ray Diagnosis by Contrastive Learning**  

---
Primary Category: eess.IV  
Categories: arms-org, cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.15111v1)  

---


**ABSTRACT**  
Purpose: Limited studies exploring concrete methods or approaches to tackle and enhance model fairness in the radiology domain. Our proposed AI model utilizes supervised contrastive learning to minimize bias in CXR diagnosis.   Materials and Methods: In this retrospective study, we evaluated our proposed method on two datasets: the Medical Imaging and Data Resource Center (MIDRC) dataset with 77,887 CXR images from 27,796 patients collected as of April 20, 2023 for COVID-19 diagnosis, and the NIH Chest X-ray (NIH-CXR) dataset with 112,120 CXR images from 30,805 patients collected between 1992 and 2015. In the NIH-CXR dataset, thoracic abnormalities include atelectasis, cardiomegaly, effusion, infiltration, mass, nodule, pneumonia, pneumothorax, consolidation, edema, emphysema, fibrosis, pleural thickening, or hernia. Our proposed method utilizes supervised contrastive learning with carefully selected positive and negative samples to generate fair image embeddings, which are fine-tuned for subsequent tasks to reduce bias in chest X-ray (CXR) diagnosis. We evaluated the methods using the marginal AUC difference ($\delta$ mAUC).   Results: The proposed model showed a significant decrease in bias across all subgroups when compared to the baseline models, as evidenced by a paired T-test (p<0.0001). The $\delta$ mAUC obtained by our method were 0.0116 (95\% CI, 0.0110-0.0123), 0.2102 (95% CI, 0.2087-0.2118), and 0.1000 (95\% CI, 0.0988-0.1011) for sex, race, and age on MIDRC, and 0.0090 (95\% CI, 0.0082-0.0097) for sex and 0.0512 (95% CI, 0.0512-0.0532) for age on NIH-CXR, respectively.   Conclusion: Employing supervised contrastive learning can mitigate bias in CXR diagnosis, addressing concerns of fairness and reliability in deep learning-based diagnostic methods.

{{</citation>}}


### (68/122) Clinical Melanoma Diagnosis with Artificial Intelligence: Insights from a Prospective Multicenter Study (Lukas Heinlein et al., 2024)

{{<citation>}}

Lukas Heinlein, Roman C. Maron, Achim Hekler, Sarah Haggenmüller, Christoph Wies, Jochen S. Utikal, Friedegund Meier, Sarah Hobelsberger, Frank F. Gellrich, Mildred Sergon, Axel Hauschild, Lars E. French, Lucie Heinzerling, Justin G. Schlager, Kamran Ghoreschi, Max Schlaak, Franz J. Hilke, Gabriela Poch, Sören Korsing, Carola Berking, Markus V. Heppt, Michael Erdmann, Sebastian Haferkamp, Konstantin Drexler, Dirk Schadendorf, Wiebke Sondermann, Matthias Goebeler, Bastian Schilling, Eva Krieghoff-Henning, Titus J. Brinker. (2024)  
**Clinical Melanoma Diagnosis with Artificial Intelligence: Insights from a Prospective Multicenter Study**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV, stat-AP  
Keywords: AI, Clinical  
[Paper Link](http://arxiv.org/abs/2401.14193v1)  

---


**ABSTRACT**  
Early detection of melanoma, a potentially lethal type of skin cancer with high prevalence worldwide, improves patient prognosis. In retrospective studies, artificial intelligence (AI) has proven to be helpful for enhancing melanoma detection. However, there are few prospective studies confirming these promising results. Existing studies are limited by low sample sizes, too homogenous datasets, or lack of inclusion of rare melanoma subtypes, preventing a fair and thorough evaluation of AI and its generalizability, a crucial aspect for its application in the clinical setting. Therefore, we assessed 'All Data are Ext' (ADAE), an established open-source ensemble algorithm for detecting melanomas, by comparing its diagnostic accuracy to that of dermatologists on a prospectively collected, external, heterogeneous test set comprising eight distinct hospitals, four different camera setups, rare melanoma subtypes, and special anatomical sites. We advanced the algorithm with real test-time augmentation (R-TTA, i.e. providing real photographs of lesions taken from multiple angles and averaging the predictions), and evaluated its generalization capabilities. Overall, the AI showed higher balanced accuracy than dermatologists (0.798, 95% confidence interval (CI) 0.779-0.814 vs. 0.781, 95% CI 0.760-0.802; p<0.001), obtaining a higher sensitivity (0.921, 95% CI 0.900- 0.942 vs. 0.734, 95% CI 0.701-0.770; p<0.001) at the cost of a lower specificity (0.673, 95% CI 0.641-0.702 vs. 0.828, 95% CI 0.804-0.852; p<0.001). As the algorithm exhibited a significant performance advantage on our heterogeneous dataset exclusively comprising melanoma-suspicious lesions, AI may offer the potential to support dermatologists particularly in diagnosing challenging cases.

{{</citation>}}


### (69/122) Attention-based Efficient Classification for 3D MRI Image of Alzheimer's Disease (Yihao Lin et al., 2024)

{{<citation>}}

Yihao Lin, Ximeng Li, Yan Zhang, Jinshan Tang. (2024)  
**Attention-based Efficient Classification for 3D MRI Image of Alzheimer's Disease**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.14130v1)  

---


**ABSTRACT**  
Early diagnosis of Alzheimer Diagnostics (AD) is a challenging task due to its subtle and complex clinical symptoms. Deep learning-assisted medical diagnosis using image recognition techniques has become an important research topic in this field. The features have to accurately capture main variations of anatomical brain structures. However, time-consuming is expensive for feature extraction by deep learning training. This study proposes a novel Alzheimer's disease detection model based on Convolutional Neural Networks. The model utilizes a pre-trained ResNet network as the backbone, incorporating post-fusion algorithm for 3D medical images and attention mechanisms. The experimental results indicate that the employed 2D fusion algorithm effectively improves the model's training expense. And the introduced attention mechanism accurately weights important regions in images, further enhancing the model's diagnostic accuracy.

{{</citation>}}


## cs.LG (21)



### (70/122) Scilab-RL: A software framework for efficient reinforcement learning and cognitive modeling research (Jan Dohmen et al., 2024)

{{<citation>}}

Jan Dohmen, Frank Röder, Manfred Eppe. (2024)  
**Scilab-RL: A software framework for efficient reinforcement learning and cognitive modeling research**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14488v1)  

---


**ABSTRACT**  
One problem with researching cognitive modeling and reinforcement learning (RL) is that researchers spend too much time on setting up an appropriate computational framework for their experiments. Many open source implementations of current RL algorithms exist, but there is a lack of a modular suite of tools combining different robotic simulators and platforms, data visualization, hyperparameter optimization, and baseline experiments. To address this problem, we present Scilab-RL, a software framework for efficient research in cognitive modeling and reinforcement learning for robotic agents. The framework focuses on goal-conditioned reinforcement learning using Stable Baselines 3 and the OpenAI gym interface. It enables native possibilities for experiment visualizations and hyperparameter optimization. We describe how these features enable researchers to conduct experiments with minimal time effort, thus maximizing research output.

{{</citation>}}


### (71/122) ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models (Yao Fu et al., 2024)

{{<citation>}}

Yao Fu, Leyang Xue, Yeqi Huang, Andrei-Octavian Brabete, Dmitrii Ustiugov, Yuvraj Patel, Luo Mai. (2024)  
**ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.14351v1)  

---


**ABSTRACT**  
This paper presents ServerlessLLM, a locality-enhanced serverless inference system for Large Language Models (LLMs). ServerlessLLM exploits the substantial capacity and bandwidth of storage and memory devices available on GPU servers, thereby reducing costly remote checkpoint downloads and achieving efficient checkpoint loading. ServerlessLLM achieves this through three main contributions: (i) fast LLM checkpoint loading via a novel loading-optimized checkpoint format design, coupled with an efficient multi-tier checkpoint loading system; (ii) locality-driven LLM inference with live migration, which allows ServerlessLLM to effectively achieve locality-driven server allocation while preserving the low latency of ongoing LLM inference; and (iii) locality-aware server allocation, enabling ServerlessLLM to evaluate the status of each server in a cluster and effectively schedule model startup time to capitalize on local checkpoint placement. Our comprehensive experiments, which include microbenchmarks and real-world traces, show that ServerlessLLM surpasses state-of-the-art systems by 10 - 200X in latency performance when running various LLM inference workloads.

{{</citation>}}


### (72/122) Multi-agent Deep Reinforcement Learning for Dynamic Pricing by Fast-charging Electric Vehicle Hubs in ccompetition (Diwas Paudel et al., 2024)

{{<citation>}}

Diwas Paudel, Tapas K. Das. (2024)  
**Multi-agent Deep Reinforcement Learning for Dynamic Pricing by Fast-charging Electric Vehicle Hubs in ccompetition**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SY, cs.LG, econ-GN, eess-SY, q-fin-EC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.15108v1)  

---


**ABSTRACT**  
Fast-charging hubs for electric vehicles will soon become part of the newly built infrastructure for transportation electrification across the world. These hubs are expected to host many DC fast-charging stations and will admit EVs only for charging. Like the gasoline refueling stations, fast-charging hubs in a neighborhood will dynamically vary their prices to compete for the same pool of EV owners. These hubs will interact with the electric power network by making purchase commitments for a significant part of their power needs in the day-ahead (DA) electricity market and meeting the difference from the real-time (RT) market. Hubs may have supplemental battery storage systems (BSS), which they will use for arbitrage. In this paper, we develop a two-step data-driven dynamic pricing methodology for hubs in price competition. We first obtain the DA commitment by solving a stochastic DA commitment model. Thereafter we obtain the hub pricing strategies by modeling the game as a competitive Markov decision process (CMDP) and solving it using a multi-agent deep reinforcement learning (MADRL) approach. We develop a numerical case study for a pricing game between two charging hubs. We solve the case study with our methodology by using combinations of two different DRL algorithms, DQN and SAC, and two different neural networks (NN) architectures, a feed-forward (FF) neural network, and a multi-head attention (MHA) neural network. We construct a measure of collusion (index) using the hub profits. A value of zero for this index indicates no collusion (perfect competition) and a value of one indicates full collusion (monopolistic behavior). Our results show that the collusion index varies approximately between 0.14 and 0.45 depending on the combinations of the algorithms and the architectures chosen by the hubs.

{{</citation>}}


### (73/122) Interpretable Solutions for Breast Cancer Diagnosis with Grammatical Evolution and Data Augmentation (Yumnah Hasan et al., 2024)

{{<citation>}}

Yumnah Hasan, Allan de Lima, Fatemeh Amerehi, Darian Reyes Fernandez de Bulnes, Patrick Healy, Conor Ryan. (2024)  
**Interpretable Solutions for Breast Cancer Diagnosis with Grammatical Evolution and Data Augmentation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2401.14255v1)  

---


**ABSTRACT**  
Medical imaging diagnosis increasingly relies on Machine Learning (ML) models. This is a task that is often hampered by severely imbalanced datasets, where positive cases can be quite rare. Their use is further compromised by their limited interpretability, which is becoming increasingly important. While post-hoc interpretability techniques such as SHAP and LIME have been used with some success on so-called black box models, the use of inherently understandable models makes such endeavors more fruitful. This paper addresses these issues by demonstrating how a relatively new synthetic data generation technique, STEM, can be used to produce data to train models produced by Grammatical Evolution (GE) that are inherently understandable. STEM is a recently introduced combination of the Synthetic Minority Oversampling Technique (SMOTE), Edited Nearest Neighbour (ENN), and Mixup; it has previously been successfully used to tackle both between class and within class imbalance issues. We test our technique on the Digital Database for Screening Mammography (DDSM) and the Wisconsin Breast Cancer (WBC) datasets and compare Area Under the Curve (AUC) results with an ensemble of the top three performing classifiers from a set of eight standard ML classifiers with varying degrees of interpretability. We demonstrate that the GE-derived models present the best AUC while still maintaining interpretable solutions.

{{</citation>}}


### (74/122) Sample Efficient Reinforcement Learning by Automatically Learning to Compose Subtasks (Shuai Han et al., 2024)

{{<citation>}}

Shuai Han, Mehdi Dastani, Shihan Wang. (2024)  
**Sample Efficient Reinforcement Learning by Automatically Learning to Compose Subtasks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.14226v1)  

---


**ABSTRACT**  
Improving sample efficiency is central to Reinforcement Learning (RL), especially in environments where the rewards are sparse. Some recent approaches have proposed to specify reward functions as manually designed or learned reward structures whose integrations in the RL algorithms are claimed to significantly improve the learning efficiency. Manually designed reward structures can suffer from inaccuracy and existing automatically learning methods are often computationally intractable for complex tasks. The integration of inaccurate or partial reward structures in RL algorithms fail to learn optimal policies. In this work, we propose an RL algorithm that can automatically structure the reward function for sample efficiency, given a set of labels that signify subtasks. Given such minimal knowledge about the task, we train a high-level policy that selects optimal sub-tasks in each state together with a low-level policy that efficiently learns to complete each sub-task. We evaluate our algorithm in a variety of sparse-reward environments. The experiment results show that our approach significantly outperforms the state-of-art baselines as the difficulty of the task increases.

{{</citation>}}


### (75/122) How Can Large Language Models Understand Spatial-Temporal Data? (Lei Liu et al., 2024)

{{<citation>}}

Lei Liu, Shuo Yu, Runze Wang, Zhenxun Ma, Yanming Shen. (2024)  
**How Can Large Language Models Understand Spatial-Temporal Data?**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.14192v1)  

---


**ABSTRACT**  
While Large Language Models (LLMs) dominate tasks like natural language processing and computer vision, harnessing their power for spatial-temporal forecasting remains challenging. The disparity between sequential text and complex spatial-temporal data hinders this application. To address this issue, this paper introduces STG-LLM, an innovative approach empowering LLMs for spatial-temporal forecasting. We tackle the data mismatch by proposing: 1) STG-Tokenizer: This spatial-temporal graph tokenizer transforms intricate graph data into concise tokens capturing both spatial and temporal relationships; 2) STG-Adapter: This minimalistic adapter, consisting of linear encoding and decoding layers, bridges the gap between tokenized data and LLM comprehension. By fine-tuning only a small set of parameters, it can effectively grasp the semantics of tokens generated by STG-Tokenizer, while preserving the original natural language understanding capabilities of LLMs. Extensive experiments on diverse spatial-temporal benchmark datasets show that STG-LLM successfully unlocks LLM potential for spatial-temporal forecasting. Remarkably, our approach achieves competitive performance on par with dedicated SOTA methods.

{{</citation>}}


### (76/122) Alleviating Structural Distribution Shift in Graph Anomaly Detection (Yuan Gao et al., 2024)

{{<citation>}}

Yuan Gao, Xiang Wang, Xiangnan He, Zhenguang Liu, Huamin Feng, Yongdong Zhang. (2024)  
**Alleviating Structural Distribution Shift in Graph Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, GNN  
[Paper Link](http://arxiv.org/abs/2401.14155v1)  

---


**ABSTRACT**  
Graph anomaly detection (GAD) is a challenging binary classification problem due to its different structural distribution between anomalies and normal nodes -- abnormal nodes are a minority, therefore holding high heterophily and low homophily compared to normal nodes. Furthermore, due to various time factors and the annotation preferences of human experts, the heterophily and homophily can change across training and testing data, which is called structural distribution shift (SDS) in this paper. The mainstream methods are built on graph neural networks (GNNs), benefiting the classification of normals from aggregating homophilous neighbors, yet ignoring the SDS issue for anomalies and suffering from poor generalization.   This work solves the problem from a feature view. We observe that the degree of SDS varies between anomalies and normal nodes. Hence to address the issue, the key lies in resisting high heterophily for anomalies meanwhile benefiting the learning of normals from homophily. We tease out the anomaly features on which we constrain to mitigate the effect of heterophilous neighbors and make them invariant. We term our proposed framework as Graph Decomposition Network (GDN). Extensive experiments are conducted on two benchmark datasets, and the proposed framework achieves a remarkable performance boost in GAD, especially in an SDS environment where anomalies have largely different structural distribution across training and testing environments. Codes are open-sourced in https://github.com/blacksingular/wsdm_GDN.

{{</citation>}}


### (77/122) True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning (Weihao Tan et al., 2024)

{{<citation>}}

Weihao Tan, Wentao Zhang, Shanqi Liu, Longtao Zheng, Xinrun Wang, Bo An. (2024)  
**True Knowledge Comes from Practice: Aligning LLMs with Embodied Environments via Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.14151v1)  

---


**ABSTRACT**  
Despite the impressive performance across numerous tasks, large language models (LLMs) often fail in solving simple decision-making tasks due to the misalignment of the knowledge in LLMs with environments. On the contrary, reinforcement learning (RL) agents learn policies from scratch, which makes them always align with environments but difficult to incorporate prior knowledge for efficient explorations. To narrow the gap, we propose TWOSOME, a novel general online framework that deploys LLMs as decision-making agents to efficiently interact and align with embodied environments via RL without requiring any prepared datasets or prior knowledge of the environments. Firstly, we query the joint probabilities of each valid action with LLMs to form behavior policies. Then, to enhance the stability and robustness of the policies, we propose two normalization methods and summarize four prompt design principles. Finally, we design a novel parameter-efficient training architecture where the actor and critic share one frozen LLM equipped with low-rank adapters (LoRA) updated by PPO. We conduct extensive experiments to evaluate TWOSOME. i) TWOSOME exhibits significantly better sample efficiency and performance compared to the conventional RL method, PPO, and prompt tuning method, SayCan, in both classical decision-making environment, Overcooked, and simulated household environment, VirtualHome. ii) Benefiting from LLMs' open-vocabulary feature, TWOSOME shows superior generalization ability to unseen tasks. iii) Under our framework, there is no significant loss of the LLMs' original ability during online PPO finetuning.

{{</citation>}}


### (78/122) PruneSymNet: A Symbolic Neural Network and Pruning Algorithm for Symbolic Regression (Min Wu et al., 2024)

{{<citation>}}

Min Wu, Weijun Li, Lina Yu, Wenqiang Li, Jingyi Liu, Yanjie Li, Meilan Hao. (2024)  
**PruneSymNet: A Symbolic Neural Network and Pruning Algorithm for Symbolic Regression**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2401.15103v1)  

---


**ABSTRACT**  
Symbolic regression aims to derive interpretable symbolic expressions from data in order to better understand and interpret data. %which plays an important role in knowledge discovery and interpretable machine learning.   In this study, a symbolic network called PruneSymNet is proposed for symbolic regression. This is a novel neural network whose activation function consists of common elementary functions and operators. The whole network is differentiable and can be trained by gradient descent method. Each subnetwork in the network corresponds to an expression, and our goal is to extract such subnetworks to get the desired symbolic expression.   Therefore, a greedy pruning algorithm is proposed to prune the network into a subnetwork while ensuring the accuracy of data fitting. The proposed greedy pruning algorithm preserves the edge with the least loss in each pruning, but greedy algorithm often can not get the optimal solution. In order to alleviate this problem, we combine beam search during pruning to obtain multiple candidate expressions each time, and finally select the expression with the smallest loss as the final result. It was tested on the public data set and compared with the current popular algorithms. The results showed that the proposed algorithm had better accuracy.

{{</citation>}}


### (79/122) FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design (Haojun Xia et al., 2024)

{{<citation>}}

Haojun Xia, Zhen Zheng, Xiaoxia Wu, Shiyang Chen, Zhewei Yao, Stephen Youn, Arash Bakhtiari, Michael Wyatt, Donglin Zhuang, Zhongzhu Zhou, Olatunji Ruwase, Yuxiong He, Shuaiwen Leon Song. (2024)  
**FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-AR, cs-LG, cs.LG  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14112v1)  

---


**ABSTRACT**  
Six-bit quantization (FP6) can effectively reduce the size of large language models (LLMs) and preserve the model quality consistently across varied applications. However, existing systems do not provide Tensor Core support for FP6 quantization and struggle to achieve practical performance improvements during LLM inference. It is challenging to support FP6 quantization on GPUs due to (1) unfriendly memory access of model weights with irregular bit-width and (2) high runtime overhead of weight de-quantization. To address these problems, we propose TC-FPx, the first full-stack GPU kernel design scheme with unified Tensor Core support of float-point weights for various quantization bit-width. We integrate TC-FPx kernel into an existing inference system, providing new end-to-end support (called FP6-LLM) for quantized LLM inference, where better trade-offs between inference cost and model quality are achieved. Experiments show that FP6-LLM enables the inference of LLaMA-70b using only a single GPU, achieving 1.69x-2.65x higher normalized inference throughput than the FP16 baseline. The source code will be publicly available soon.

{{</citation>}}


### (80/122) Learning under Label Noise through Few-Shot Human-in-the-Loop Refinement (Aaqib Saeed et al., 2024)

{{<citation>}}

Aaqib Saeed, Dimitris Spathis, Jungwoo Oh, Edward Choi, Ali Etemad. (2024)  
**Learning under Label Noise through Few-Shot Human-in-the-Loop Refinement**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2401.14107v1)  

---


**ABSTRACT**  
Wearable technologies enable continuous monitoring of various health metrics, such as physical activity, heart rate, sleep, and stress levels. A key challenge with wearable data is obtaining quality labels. Unlike modalities like video where the videos themselves can be effectively used to label objects or events, wearable data do not contain obvious cues about the physical manifestation of the users and usually require rich metadata. As a result, label noise can become an increasingly thorny issue when labeling such data. In this paper, we propose a novel solution to address noisy label learning, entitled Few-Shot Human-in-the-Loop Refinement (FHLR). Our method initially learns a seed model using weak labels. Next, it fine-tunes the seed model using a handful of expert corrections. Finally, it achieves better generalizability and robustness by merging the seed and fine-tuned models via weighted parameter averaging. We evaluate our approach on four challenging tasks and datasets, and compare it against eight competitive baselines designed to deal with noisy labels. We show that FHLR achieves significantly better performance when learning from noisy labels and achieves state-of-the-art by a large margin, with up to 19% accuracy improvement under symmetric and asymmetric noise. Notably, we find that FHLR is particularly robust to increased label noise, unlike prior works that suffer from severe performance degradation. Our work not only achieves better generalization in high-stakes health sensing benchmarks but also sheds light on how noise affects commonly-used models.

{{</citation>}}


### (81/122) Sparse and Transferable Universal Singular Vectors Attack (Kseniia Kuvshinova et al., 2024)

{{<citation>}}

Kseniia Kuvshinova, Olga Tsymboi, Ivan Oseledets. (2024)  
**Sparse and Transferable Universal Singular Vectors Attack**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.14031v1)  

---


**ABSTRACT**  
The research in the field of adversarial attacks and models' vulnerability is one of the fundamental directions in modern machine learning. Recent studies reveal the vulnerability phenomenon, and understanding the mechanisms behind this is essential for improving neural network characteristics and interpretability. In this paper, we propose a novel sparse universal white-box adversarial attack. Our approach is based on truncated power iteration providing sparsity to $(p,q)$-singular vectors of the hidden layers of Jacobian matrices. Using the ImageNet benchmark validation subset, we analyze the proposed method in various settings, achieving results comparable to dense baselines with more than a 50% fooling rate while damaging only 5% of pixels and utilizing 256 samples for perturbation fitting. We also show that our algorithm admits higher attack magnitude without affecting the human ability to solve the task. Furthermore, we investigate that the constructed perturbations are highly transferable among different models without significantly decreasing the fooling rate. Our findings demonstrate the vulnerability of state-of-the-art models to sparse attacks and highlight the importance of developing robust machine learning systems.

{{</citation>}}


### (82/122) Accelerating Retrieval-Augmented Language Model Serving with Speculation (Zhihao Zhang et al., 2024)

{{<citation>}}

Zhihao Zhang, Alan Zhu, Lijie Yang, Yihua Xu, Lanting Li, Phitchaya Mangpo Phothilimthana, Zhihao Jia. (2024)  
**Accelerating Retrieval-Augmented Language Model Serving with Speculation**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-IR, cs-LG, cs.LG  
Keywords: Language Model, NLP, QA  
[Paper Link](http://arxiv.org/abs/2401.14021v1)  

---


**ABSTRACT**  
Retrieval-augmented language models (RaLM) have demonstrated the potential to solve knowledge-intensive natural language processing (NLP) tasks by combining a non-parametric knowledge base with a parametric language model. Instead of fine-tuning a fully parametric model, RaLM excels at its low-cost adaptation to the latest data and better source attribution mechanisms. Among various RaLM approaches, iterative RaLM delivers a better generation quality due to a more frequent interaction between the retriever and the language model. Despite the benefits, iterative RaLM usually encounters high overheads due to the frequent retrieval step. To this end, we propose RaLMSpec, a speculation-inspired framework that provides generic speed-up over iterative RaLM while preserving the same model outputs through speculative retrieval and batched verification. By further incorporating prefetching, optimal speculation stride scheduler, and asynchronous verification, RaLMSpec can automatically exploit the acceleration potential to the fullest. For naive iterative RaLM serving, extensive evaluations over three language models on four downstream QA datasets demonstrate that RaLMSpec can achieve a speed-up ratio of 1.75-2.39x, 1.04-1.39x, and 1.31-1.77x when the retriever is an exact dense retriever, approximate dense retriever, and sparse retriever respectively compared with the baseline. For KNN-LM serving, RaLMSpec can achieve a speed-up ratio up to 7.59x and 2.45x when the retriever is an exact dense retriever and approximate dense retriever, respectively, compared with the baseline.

{{</citation>}}


### (83/122) Cross-Domain Few-Shot Learning via Adaptive Transformer Networks (Naeem Paeedeh et al., 2024)

{{<citation>}}

Naeem Paeedeh, Mahardhika Pratama, Muhammad Anwar Ma'sum, Wolfgang Mayer, Zehong Cao, Ryszard Kowlczyk. (2024)  
**Cross-Domain Few-Shot Learning via Adaptive Transformer Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Few-Shot, Transformer  
[Paper Link](http://arxiv.org/abs/2401.13987v1)  

---


**ABSTRACT**  
Most few-shot learning works rely on the same domain assumption between the base and the target tasks, hindering their practical applications. This paper proposes an adaptive transformer network (ADAPTER), a simple but effective solution for cross-domain few-shot learning where there exist large domain shifts between the base task and the target task. ADAPTER is built upon the idea of bidirectional cross-attention to learn transferable features between the two domains. The proposed architecture is trained with DINO to produce diverse, and less biased features to avoid the supervision collapse problem. Furthermore, the label smoothing approach is proposed to improve the consistency and reliability of the predictions by also considering the predicted labels of the close samples in the embedding space. The performance of ADAPTER is rigorously evaluated in the BSCD-FSL benchmarks in which it outperforms prior arts with significant margins.

{{</citation>}}


### (84/122) Dynamic Long-Term Time-Series Forecasting via Meta Transformer Networks (Muhammad Anwar Ma'sum et al., 2024)

{{<citation>}}

Muhammad Anwar Ma'sum, MD Rasel Sarkar, Mahardhika Pratama, Savitha Ramasamy, Sreenatha Anavatti, Lin Liu, Habibullah, Ryszard Kowalczyk. (2024)  
**Dynamic Long-Term Time-Series Forecasting via Meta Transformer Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.13968v1)  

---


**ABSTRACT**  
A reliable long-term time-series forecaster is highly demanded in practice but comes across many challenges such as low computational and memory footprints as well as robustness against dynamic learning environments. This paper proposes Meta-Transformer Networks (MANTRA) to deal with the dynamic long-term time-series forecasting tasks. MANTRA relies on the concept of fast and slow learners where a collection of fast learners learns different aspects of data distributions while adapting quickly to changes. A slow learner tailors suitable representations to fast learners. Fast adaptations to dynamic environments are achieved using the universal representation transformer layers producing task-adapted representations with a small number of parameters. Our experiments using four datasets with different prediction lengths demonstrate the advantage of our approach with at least $3\%$ improvements over the baseline algorithms for both multivariate and univariate settings. Source codes of MANTRA are publicly available in \url{https://github.com/anwarmaxsum/MANTRA}.

{{</citation>}}


### (85/122) Reinforcement Learning with Hidden Markov Models for Discovering Decision-Making Dynamics (Xingche Guo et al., 2024)

{{<citation>}}

Xingche Guo, Donglin Zeng, Yuanjia Wang. (2024)  
**Reinforcement Learning with Hidden Markov Models for Discovering Decision-Making Dynamics**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-AP, stat-ME, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.13929v1)  

---


**ABSTRACT**  
Major depressive disorder (MDD) presents challenges in diagnosis and treatment due to its complex and heterogeneous nature. Emerging evidence indicates that reward processing abnormalities may serve as a behavioral marker for MDD. To measure reward processing, patients perform computer-based behavioral tasks that involve making choices or responding to stimulants that are associated with different outcomes. Reinforcement learning (RL) models are fitted to extract parameters that measure various aspects of reward processing to characterize how patients make decisions in behavioral tasks. Recent findings suggest the inadequacy of characterizing reward learning solely based on a single RL model; instead, there may be a switching of decision-making processes between multiple strategies. An important scientific question is how the dynamics of learning strategies in decision-making affect the reward learning ability of individuals with MDD. Motivated by the probabilistic reward task (PRT) within the EMBARC study, we propose a novel RL-HMM framework for analyzing reward-based decision-making. Our model accommodates learning strategy switching between two distinct approaches under a hidden Markov model (HMM): subjects making decisions based on the RL model or opting for random choices. We account for continuous RL state space and allow time-varying transition probabilities in the HMM. We introduce a computationally efficient EM algorithm for parameter estimation and employ a nonparametric bootstrap for inference. We apply our approach to the EMBARC study to show that MDD patients are less engaged in RL compared to the healthy controls, and engagement is associated with brain activities in the negative affect circuitry during an emotional conflict task.

{{</citation>}}


### (86/122) Towards 3D Molecule-Text Interpretation in Language Models (Sihang Li et al., 2024)

{{<citation>}}

Sihang Li, Zhiyuan Liu, Yanchen Luo, Xiang Wang, Xiangnan He, Kenji Kawaguchi, Tat-Seng Chua, Qi Tian. (2024)  
**Towards 3D Molecule-Text Interpretation in Language Models**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG, q-bio-BM  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2401.13923v1)  

---


**ABSTRACT**  
Language Models (LMs) have greatly influenced diverse domains. However, their inherent limitation in comprehending 3D molecular structures has considerably constrained their potential in the biomolecular domain. To bridge this gap, we focus on 3D molecule-text interpretation, and propose 3D-MoLM: 3D-Molecular Language Modeling. Specifically, 3D-MoLM enables an LM to interpret and analyze 3D molecules by equipping the LM with a 3D molecular encoder. This integration is achieved by a 3D molecule-text projector, bridging the 3D molecular encoder's representation space and the LM's input space. Moreover, to enhance 3D-MoLM's ability of cross-modal molecular understanding and instruction following, we meticulously curated a 3D molecule-centric instruction tuning dataset -- 3D-MoIT. Through 3D molecule-text alignment and 3D molecule-centric instruction tuning, 3D-MoLM establishes an integration of 3D molecular encoder and LM. It significantly surpasses existing baselines on downstream tasks, including molecule-text retrieval, molecule captioning, and more challenging open-text molecular QA tasks, especially focusing on 3D-dependent properties.

{{</citation>}}


### (87/122) LocMoE: A Low-overhead MoE for Large Language Model Training (Jing Li et al., 2024)

{{<citation>}}

Jing Li, Zhijie Sun, Xuan He, Li Zeng, Yi Lin, Entong Li, Binfan Zheng, Rongqian Zhao, Xin Chen. (2024)  
**LocMoE: A Low-overhead MoE for Large Language Model Training**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.13920v1)  

---


**ABSTRACT**  
The Mixtures-of-Experts (MoE) model is a widespread distributed and integrated learning method for large language models (LLM), which is favored due to its ability to sparsify and expand models efficiently. However, the performance of MoE is limited by load imbalance and high latency of All-To-All communication, along with relatively redundant computation owing to large expert capacity. Load imbalance may result from existing routing policies that consistently tend to select certain experts. The frequent inter-node communication in the All-To-All procedure also significantly prolongs the training time. To alleviate the above performance problems, we propose a novel routing strategy that combines load balance and locality by converting partial inter-node communication to that of intra-node. Notably, we elucidate that there is a minimum threshold for expert capacity, calculated through the maximal angular deviation between the gating weights of the experts and the assigned tokens. We port these modifications on the PanGu-Sigma model based on the MindSpore framework with multi-level routing and conduct experiments on Ascend clusters. The experiment results demonstrate that the proposed LocMoE reduces training time per epoch by 12.68% to 22.24% compared to classical routers, such as hash router and switch router, without impacting the model accuracy.

{{</citation>}}


### (88/122) A Survey of Deep Learning and Foundation Models for Time Series Forecasting (John A. Miller et al., 2024)

{{<citation>}}

John A. Miller, Mohammed Aldosari, Farah Saeed, Nasid Habib Barna, Subas Rana, I. Budak Arpinar, Ninghao Liu. (2024)  
**A Survey of Deep Learning and Foundation Models for Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Knowledge Graph, Language Model, Time Series  
[Paper Link](http://arxiv.org/abs/2401.13912v1)  

---


**ABSTRACT**  
Deep Learning has been successfully applied to many application domains, yet its advantages have been slow to emerge for time series forecasting. For example, in the well-known Makridakis (M) Competitions, hybrids of traditional statistical or machine learning techniques have only recently become the top performers. With the recent architectural advances in deep learning being applied to time series forecasting (e.g., encoder-decoders with attention, transformers, and graph neural networks), deep learning has begun to show significant advantages. Still, in the area of pandemic prediction, there remain challenges for deep learning models: the time series is not long enough for effective training, unawareness of accumulated scientific knowledge, and interpretability of the model. To this end, the development of foundation models (large deep learning models with extensive pre-training) allows models to understand patterns and acquire knowledge that can be applied to new related problems before extensive training data becomes available. Furthermore, there is a vast amount of knowledge available that deep learning models can tap into, including Knowledge Graphs and Large Language Models fine-tuned with scientific domain knowledge. There is ongoing research examining how to utilize or inject such knowledge into deep learning models. In this survey, several state-of-the-art modeling techniques are reviewed, and suggestions for further work are provided.

{{</citation>}}


### (89/122) Hi-Core: Hierarchical Knowledge Transfer for Continual Reinforcement Learning (Chaofan Pan et al., 2024)

{{<citation>}}

Chaofan Pan, Xin Yang, Hao Wang, Wei Wei, Tianrui Li. (2024)  
**Hi-Core: Hierarchical Knowledge Transfer for Continual Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.15098v1)  

---


**ABSTRACT**  
Continual reinforcement learning (CRL) empowers RL agents with the ability to learn from a sequence of tasks, preserving previous knowledge and leveraging it to facilitate future learning. However, existing methods often focus on transferring low-level knowledge across similar tasks, which neglects the hierarchical structure of human cognitive control, resulting in insufficient knowledge transfer across diverse tasks. To enhance high-level knowledge transfer, we propose a novel framework named Hi-Core (Hierarchical knowledge transfer for Continual reinforcement learning), which is structured in two layers: 1) the high-level policy formulation which utilizes the powerful reasoning ability of the Large Language Model (LLM) to set goals and 2) the low-level policy learning through RL which is oriented by high-level goals. Moreover, the knowledge base (policy library) is constructed to store policies that can be retrieved for hierarchical knowledge transfer. Experiments conducted in MiniGrid have demonstrated the effectiveness of Hi-Core in handling diverse CRL tasks, outperforming popular baselines.

{{</citation>}}


### (90/122) Edge Conditional Node Update Graph Neural Network for Multi-variate Time Series Anomaly Detection (Hayoung Jo et al., 2024)

{{<citation>}}

Hayoung Jo, Seong-Whan Lee. (2024)  
**Edge Conditional Node Update Graph Neural Network for Multi-variate Time Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, GNN, Graph Neural Network, Time Series  
[Paper Link](http://arxiv.org/abs/2401.13872v1)  

---


**ABSTRACT**  
With the rapid advancement in cyber-physical systems, the increasing number of sensors has significantly complicated manual monitoring of system states. Consequently, graph-based time-series anomaly detection methods have gained attention due to their ability to explicitly represent relationships between sensors. However, these methods often apply a uniform source node representation across all connected target nodes, even when updating different target node representations. Moreover, the graph attention mechanism, commonly used to infer unknown graph structures, could constrain the diversity of source node representations. In this paper, we introduce the Edge Conditional Node-update Graph Neural Network (ECNU-GNN). Our model, equipped with an edge conditional node update module, dynamically transforms source node representations based on connected edges to represent target nodes aptly. We validate performance on three real-world datasets: SWaT, WADI, and PSM. Our model demonstrates 5.4%, 12.4%, and 6.0% higher performance, respectively, compared to best F1 baseline models.

{{</citation>}}


## cs.HC (9)



### (91/122) Towards Collective Superintelligence: Amplifying Group IQ using Conversational Swarms (Louis Rosenberg et al., 2024)

{{<citation>}}

Louis Rosenberg, Gregg Willcox, Hans Schumann, Ganesh Mani. (2024)  
**Towards Collective Superintelligence: Amplifying Group IQ using Conversational Swarms**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15109v1)  

---


**ABSTRACT**  
Swarm Intelligence (SI) is a natural phenomenon that enables biological groups to amplify their combined intellect by forming real-time systems. Artificial Swarm Intelligence (or Swarm AI) is a technology that enables networked human groups to amplify their combined intelligence by forming similar systems. In the past, swarm-based methods were constrained to narrowly defined tasks like probabilistic forecasting and multiple-choice decision making. A new technology called Conversational Swarm Intelligence (CSI) was developed in 2023 that amplifies the decision-making accuracy of networked human groups through natural conversational deliberations. The current study evaluated the ability of real-time groups using a CSI platform to take a common IQ test known as Raven's Advanced Progressive Matrices (RAPM). First, a baseline group of participants took the Raven's IQ test by traditional survey. This group averaged 45.6% correct. Then, groups of approximately 35 individuals answered IQ test questions together using a CSI platform called Thinkscape. These groups averaged 80.5% correct. This places the CSI groups in the 97th percentile of IQ test-takers and corresponds to an effective IQ increase of 28 points (p<0.001). This is an encouraging result and suggests that CSI is a powerful method for enabling conversational collective intelligence in large, networked groups. In addition, because CSI is scalable across groups of potentially any size, this technology may provide a viable pathway to building a Collective Superintelligence.

{{</citation>}}


### (92/122) Design Principles for Generative AI Applications (Justin D. Weisz et al., 2024)

{{<citation>}}

Justin D. Weisz, Jessica He, Michael Muller, Gabriela Hoefer, Rachel Miles, Werner Geyer. (2024)  
**Design Principles for Generative AI Applications**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.14484v1)  

---


**ABSTRACT**  
Generative AI applications present unique design challenges. As generative AI technologies are increasingly being incorporated into mainstream applications, there is an urgent need for guidance on how to design user experiences that foster effective and safe use. We present six principles for the design of generative AI applications that address unique characteristics of generative AI UX and offer new interpretations and extensions of known issues in the design of AI applications. Each principle is coupled with a set of design strategies for implementing that principle via UX capabilities or through the design process. The principles and strategies were developed through an iterative process involving literature review, feedback from design practitioners, validation against real-world generative AI applications, and incorporation into the design process of two generative AI applications. We anticipate the principles to usefully inform the design of generative AI applications by driving actionable design recommendations.

{{</citation>}}


### (93/122) Wordflow: Social Prompt Engineering for Large Language Models (Zijie J. Wang et al., 2024)

{{<citation>}}

Zijie J. Wang, Aishwarya Chakravarthy, David Munechika, Duen Horng Chau. (2024)  
**Wordflow: Social Prompt Engineering for Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CL, cs-HC, cs-LG, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14447v1)  

---


**ABSTRACT**  
Large language models (LLMs) require well-crafted prompts for effective use. Prompt engineering, the process of designing prompts, is challenging, particularly for non-experts who are less familiar with AI technologies. While researchers have proposed techniques and tools to assist LLM users in prompt design, these works primarily target AI application developers rather than non-experts. To address this research gap, we propose social prompt engineering, a novel paradigm that leverages social computing techniques to facilitate collaborative prompt design. To investigate social prompt engineering, we introduce Wordflow, an open-source and social text editor that enables everyday users to easily create, run, share, and discover LLM prompts. Additionally, by leveraging modern web technologies, Wordflow allows users to run LLMs locally and privately in their browsers. Two usage scenarios highlight how social prompt engineering and our tool can enhance laypeople's interaction with LLMs. Wordflow is publicly accessible at https://poloclub.github.io/wordflow.

{{</citation>}}


### (94/122) The Typing Cure: Experiences with Large Language Model Chatbots for Mental Health Support (Inhwa Song et al., 2024)

{{<citation>}}

Inhwa Song, Sachin R. Pendse, Neha Kumar, Munmun De Choudhury. (2024)  
**The Typing Cure: Experiences with Large Language Model Chatbots for Mental Health Support**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CY, cs-HC, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14362v1)  

---


**ABSTRACT**  
People experiencing severe distress increasingly use Large Language Model (LLM) chatbots as mental health support tools. Discussions on social media have described how engagements were lifesaving for some, but evidence suggests that general-purpose LLM chatbots also have notable risks that could endanger the welfare of users if not designed responsibly. In this study, we investigate the lived experiences of people who have used LLM chatbots for mental health support. We build on interviews with 21 individuals from globally diverse backgrounds to analyze how users create unique support roles for their chatbots, fill in gaps in everyday care, and navigate associated cultural limitations when seeking support from chatbots. We ground our analysis in psychotherapy literature around effective support, and introduce the concept of therapeutic alignment, or aligning AI with therapeutic values for mental health contexts. Our study offers recommendations for how designers can approach the ethical and effective use of LLM chatbots and other AI mental health support tools in mental health care.

{{</citation>}}


### (95/122) Decision Theoretic Foundations for Experiments Evaluating Human Decisions (Jessica Hullman et al., 2024)

{{<citation>}}

Jessica Hullman, Alex Kale, Jason Hartline. (2024)  
**Decision Theoretic Foundations for Experiments Evaluating Human Decisions**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15106v1)  

---


**ABSTRACT**  
Decision-making with information displays is a key focus of research in areas like explainable AI, human-AI teaming, and data visualization. However, what constitutes a decision problem, and what is required for an experiment to be capable of concluding that human decisions are flawed in some way, remain open to speculation. We present a widely applicable definition of a decision problem synthesized from statistical decision theory and information economics. We argue that to attribute loss in human performance to forms of bias, an experiment must provide participants with the information that a rational agent would need to identify the normative decision. We evaluate the extent to which recent evaluations of decision-making from the literature on AI-assisted decisions achieve this criteria. We find that only 6 (17\%) of 35 studies that claim to identify biased behavior present participants with sufficient information to characterize their behavior as deviating from good decision-making. We motivate the value of studying well-defined decision problems by describing a characterization of performance losses they allow us to conceive. In contrast, the ambiguities of a poorly communicated decision problem preclude normative interpretation. We conclude with recommendations for practice.

{{</citation>}}


### (96/122) GPTVoiceTasker: LLM-Powered Virtual Assistant for Smartphone (Minh Duc Vu et al., 2024)

{{<citation>}}

Minh Duc Vu, Han Wang, Zhuang Li, Jieshan Chen, Shengdong Zhao, Zhenchang Xing, Chunyang Chen. (2024)  
**GPTVoiceTasker: LLM-Powered Virtual Assistant for Smartphone**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14268v1)  

---


**ABSTRACT**  
Virtual assistants have the potential to play an important role in helping users achieves different tasks. However, these systems face challenges in their real-world usability, characterized by inefficiency and struggles in grasping user intentions. Leveraging recent advances in Large Language Models (LLMs), we introduce GptVoiceTasker, a virtual assistant poised to enhance user experiences and task efficiency on mobile devices. GptVoiceTasker excels at intelligently deciphering user commands and executing relevant device interactions to streamline task completion. The system continually learns from historical user commands to automate subsequent usages, further enhancing execution efficiency. Our experiments affirm GptVoiceTasker's exceptional command interpretation abilities and the precision of its task automation module. In our user study, GptVoiceTasker boosted task efficiency in real-world scenarios by 34.85%, accompanied by positive participant feedback. We made GptVoiceTasker open-source, inviting further research into LLMs utilization for diverse tasks through prompt engineering and leveraging user usage data to improve efficiency.

{{</citation>}}


### (97/122) Mapping the Design Space of Teachable Social Media Feed Experiences (K. J. Kevin Feng et al., 2024)

{{<citation>}}

K. J. Kevin Feng, Xander Koo, Lawrence Tan, Amy Bruckman, David W. McDonald, Amy X. Zhang. (2024)  
**Mapping the Design Space of Teachable Social Media Feed Experiences**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2401.14000v2)  

---


**ABSTRACT**  
Social media feeds are deeply personal spaces that reflect individual values and preferences. However, top-down, platform-wide content algorithms can reduce users' sense of agency and fail to account for nuanced experiences and values. Drawing on the paradigm of interactive machine teaching (IMT), an interaction framework for non-expert algorithmic adaptation, we map out a design space for teachable social media feed experiences to empower agential, personalized feed curation. To do so, we conducted a think-aloud study (N=24) featuring four social media platforms -- Instagram, Mastodon, TikTok, and Twitter -- to understand key signals users leveraged to determine the value of a post in their feed. We synthesized users' signals into taxonomies that, when combined with user interviews, inform five design principles that extend IMT into the social media setting. We finally embodied our principles into three feed designs that we present as sensitizing concepts for teachable feed experiences moving forward.

{{</citation>}}


### (98/122) CUI@CHI 2024: Building Trust in CUIs-From Design to Deployment (Smit Desai et al., 2024)

{{<citation>}}

Smit Desai, Christina Wei, Jaisie Sin, Mateusz Dubiel, Nima Zargham, Shashank Ahire, Martin Porcheron, Anastasia Kuzminykh, Minha Lee, Heloisa Candello, Joel Fischer, Cosmin Munteanu, Benjamin R Cowan. (2024)  
**CUI@CHI 2024: Building Trust in CUIs-From Design to Deployment**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.13970v1)  

---


**ABSTRACT**  
Conversational user interfaces (CUIs) have become an everyday technology for people the world over, as well as a booming area of research. Advances in voice synthesis and the emergence of chatbots powered by large language models (LLMs), notably ChatGPT, have pushed CUIs to the forefront of human-computer interaction (HCI) research and practice. Now that these technologies enable an elemental level of usability and user experience (UX), we must turn our attention to higher-order human factors: trust and reliance. In this workshop, we aim to bring together a multidisciplinary group of researchers and practitioners invested in the next phase of CUI design. Through keynotes, presentations, and breakout sessions, we will share our knowledge, identify cutting-edge resources, and fortify an international network of CUI scholars. In particular, we will engage with the complexity of trust and reliance as attitudes and behaviours that emerge when people interact with conversational agents.

{{</citation>}}


### (99/122) A2C: A Modular Multi-stage Collaborative Decision Framework for Human-AI Teams (Shahroz Tariq et al., 2024)

{{<citation>}}

Shahroz Tariq, Mohan Baruwal Chhetri, Surya Nepal, Cecile Paris. (2024)  
**A2C: A Modular Multi-stage Collaborative Decision Framework for Human-AI Teams**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-LG, cs.HC  
Keywords: AI, Security  
[Paper Link](http://arxiv.org/abs/2401.14432v1)  

---


**ABSTRACT**  
This paper introduces A2C, a multi-stage collaborative decision framework designed to enable robust decision-making within human-AI teams. Drawing inspiration from concepts such as rejection learning and learning to defer, A2C incorporates AI systems trained to recognise uncertainty in their decisions and defer to human experts when needed. Moreover, A2C caters to scenarios where even human experts encounter limitations, such as in incident detection and response in cyber Security Operations Centres (SOC). In such scenarios, A2C facilitates collaborative explorations, enabling collective resolution of complex challenges. With support for three distinct decision-making modes in human-AI teams: Automated, Augmented, and Collaborative, A2C offers a flexible platform for developing effective strategies for human-AI collaboration. By harnessing the strengths of both humans and AI, it significantly improves the efficiency and effectiveness of complex decision-making in dynamic and evolving environments. To validate A2C's capabilities, we conducted extensive simulative experiments using benchmark datasets. The results clearly demonstrate that all three modes of decision-making can be effectively supported by A2C. Most notably, collaborative exploration by (simulated) human experts and AI achieves superior performance compared to AI in isolation, underscoring the framework's potential to enhance decision-making within human-AI teams.

{{</citation>}}


## cs.NI (1)



### (100/122) 5G Network Security Practices: An Overview and Survey (Fatema Bannat Wala et al., 2024)

{{<citation>}}

Fatema Bannat Wala, Mariam Kiran. (2024)  
**5G Network Security Practices: An Overview and Survey**  

---
Primary Category: cs.NI  
Categories: cs-CR, cs-NI, cs.NI  
Keywords: Network Security, Security  
[Paper Link](http://arxiv.org/abs/2401.14350v1)  

---


**ABSTRACT**  
This document provides an overview of 5G network security, describing various components of the 5G core network architecture and what kind of security services are offered by these 5G components. It also explores the potential security risks and vulnerabilities presented by the security architecture in 5G and recommends some of the best practices for the 5G network admins to consider while deploying a secure 5G network, based on the surveyed documents from the European government's efforts in commercializing the IoT devices and securing supply chain over 5G networks.

{{</citation>}}


## eess.AS (2)



### (101/122) VALL-T: Decoder-Only Generative Transducer for Robust and Decoding-Controllable Text-to-Speech (Chenpeng Du et al., 2024)

{{<citation>}}

Chenpeng Du, Yiwei Guo, Hankun Wang, Yifan Yang, Zhikang Niu, Shuai Wang, Hui Zhang, Xie Chen, Kai Yu. (2024)  
**VALL-T: Decoder-Only Generative Transducer for Robust and Decoding-Controllable Text-to-Speech**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.14321v4)  

---


**ABSTRACT**  
Recent TTS models with decoder-only Transformer architecture, such as SPEAR-TTS and VALL-E, achieve impressive naturalness and demonstrate the ability for zero-shot adaptation given a speech prompt. However, such decoder-only TTS models lack monotonic alignment constraints, sometimes leading to hallucination issues such as mispronunciation, word skipping and repeating. To address this limitation, we propose VALL-T, a generative Transducer model that introduces shifting relative position embeddings for input phoneme sequence, explicitly indicating the monotonic generation process while maintaining the architecture of decoder-only Transformer. Consequently, VALL-T retains the capability of prompt-based zero-shot adaptation and demonstrates better robustness against hallucinations with a relative reduction of 28.3% in the word error rate. Furthermore, the controllability of alignment in VALL-T during decoding facilitates the use of untranscribed speech prompts, even in unknown languages. It also enables the synthesis of lengthy speech by utilizing an aligned context window.

{{</citation>}}


### (102/122) Intelli-Z: Toward Intelligible Zero-Shot TTS (Sunghee Jung et al., 2024)

{{<citation>}}

Sunghee Jung, Won Jang, Jaesam Yoon, Bongwan Kim. (2024)  
**Intelli-Z: Toward Intelligible Zero-Shot TTS**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.13921v1)  

---


**ABSTRACT**  
Although numerous recent studies have suggested new frameworks for zero-shot TTS using large-scale, real-world data, studies that focus on the intelligibility of zero-shot TTS are relatively scarce. Zero-shot TTS demands additional efforts to ensure clear pronunciation and speech quality due to its inherent requirement of replacing a core parameter (speaker embedding or acoustic prompt) with a new one at the inference stage. In this study, we propose a zero-shot TTS model focused on intelligibility, which we refer to as Intelli-Z. Intelli-Z learns speaker embeddings by using multi-speaker TTS as its teacher and is trained with a cycle-consistency loss to include mismatched text-speech pairs for training. Additionally, it selectively aggregates speaker embeddings along the temporal dimension to minimize the interference of the text content of reference speech at the inference stage. We substantiate the effectiveness of the proposed methods with an ablation study. The Mean Opinion Score (MOS) increases by 9% for unseen speakers when the first two methods are applied, and it further improves by 16% when selective temporal aggregation is applied.

{{</citation>}}


## cs.SE (7)



### (103/122) MultiTest: Physical-Aware Object Insertion for Testing Multi-sensor Fusion Perception Systems (Xinyu Gao et al., 2024)

{{<citation>}}

Xinyu Gao, Zhijie Wang, Yang Feng, Lei Ma, Zhenyu Chen, Baowen Xu. (2024)  
**MultiTest: Physical-Aware Object Insertion for Testing Multi-sensor Fusion Perception Systems**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14314v1)  

---


**ABSTRACT**  
Multi-sensor fusion stands as a pivotal technique in addressing numerous safety-critical tasks and applications, e.g., self-driving cars and automated robotic arms. With the continuous advancement in data-driven artificial intelligence (AI), MSF's potential for sensing and understanding intricate external environments has been further amplified, bringing a profound impact on intelligent systems and specifically on their perception systems. Similar to traditional software, adequate testing is also required for AI-enabled MSF systems. Yet, existing testing methods primarily concentrate on single-sensor perception systems (e.g., image-/point cloud-based object detection systems). There remains a lack of emphasis on generating multi-modal test cases for MSF systems. To address these limitations, we design and implement MultiTest, a fitness-guided metamorphic testing method for complex MSF perception systems. MultiTest employs a physical-aware approach to synthesize realistic multi-modal object instances and insert them into critical positions of background images and point clouds. A fitness metric is designed to guide and boost the test generation process. We conduct extensive experiments with five SOTA perception systems to evaluate MultiTest from the perspectives of: (1) generated test cases' realism, (2) fault detection capabilities, and (3) performance improvement. The results show that MultiTest can generate realistic and modality-consistent test data and effectively detect hundreds of diverse faults of an MSF system under test. Moreover, retraining an MSF system on the test cases generated by MultiTest can improve the system's robustness.

{{</citation>}}


### (104/122) ZS4C: Zero-Shot Synthesis of Compilable Code for Incomplete Code Snippets using ChatGPT (Azmain Kabir et al., 2024)

{{<citation>}}

Azmain Kabir, Shaowei Wang, Yuan Tian, Tse-Hsun, Chen, Muhammad Asaduzzaman, Wenbin Zhang. (2024)  
**ZS4C: Zero-Shot Synthesis of Compilable Code for Incomplete Code Snippets using ChatGPT**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.14279v1)  

---


**ABSTRACT**  
Technical question and answering (Q&A) sites such as Stack Overflow have become an important source for software developers to seek knowledge. However, code snippets on Q&A sites are usually uncompilable and semantically incomplete for compilation due to unresolved types and missing dependent libraries, which raises the obstacle for users to reuse or analyze Q&A code snippets. Prior approaches either are not designed for synthesizing compilable code or suffer from a low compilation success rate. To address this problem, we propose ZS4C, a lightweight approach to perform zero-shot synthesis of compilable code from incomplete code snippets using Large Language Model (LLM). ZS4C operates in two stages. In the first stage, ZS4C utilizes an LLM, i.e., ChatGPT, to identify missing import statements for a given code snippet, leveraging our designed task-specific prompt template. In the second stage, ZS4C fixes compilation errors caused by incorrect import statements and syntax errors through collaborative work between ChatGPT and a compiler. We thoroughly evaluated ZS4C on a widely used benchmark called StatType-SO against the SOTA approach SnR. Compared with SnR, ZS4C improves the compilation rate from 63% to 87.6%, with a 39.3% improvement. On average, ZS4C can infer more accurate import statements than SnR, with an improvement of 6.6% in the F1.

{{</citation>}}


### (105/122) DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence (Daya Guo et al., 2024)

{{<citation>}}

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, Wenfeng Liang. (2024)  
**DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-LG, cs-SE, cs.SE  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14196v2)  

---


**ABSTRACT**  
The rapid development of large language models has revolutionized code intelligence in software development. However, the predominance of closed-source models has restricted extensive research and development. To address this, we introduce the DeepSeek-Coder series, a range of open-source code models with sizes from 1.3B to 33B, trained from scratch on 2 trillion tokens. These models are pre-trained on a high-quality project-level code corpus and employ a fill-in-the-blank task with a 16K window to enhance code generation and infilling. Our extensive evaluations demonstrate that DeepSeek-Coder not only achieves state-of-the-art performance among open-source code models across multiple benchmarks but also surpasses existing closed-source models like Codex and GPT-3.5. Furthermore, DeepSeek-Coder models are under a permissive license that allows for both research and unrestricted commercial use.

{{</citation>}}


### (106/122) Copilot Refinement: Addressing Code Smells in Copilot-Generated Python Code (Beiqi Zhang et al., 2024)

{{<citation>}}

Beiqi Zhang, Peng Liang, Qiong Feng, Yujia Fu, Zengyang Li. (2024)  
**Copilot Refinement: Addressing Code Smells in Copilot-Generated Python Code**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.14176v1)  

---


**ABSTRACT**  
As one of the most popular dynamic languages, Python experiences a decrease in readability and maintainability when code smells are present. Recent advancements in Large Language Models have sparked growing interest in AI-enabled tools for both code generation and refactoring. GitHub Copilot is one such tool that has gained widespread usage. Copilot Chat, released on September 2023, functions as an interactive tool aims at facilitating natural language-powered coding. However, limited attention has been given to understanding code smells in Copilot-generated Python code and Copilot's ability to fix the code smells it generates. To this end, we built a dataset comprising 102 code smells in Copilot-generated Python code. Our aim is to first explore the occurrence of code smells in Copilot-generated Python code and then evaluate the effectiveness of Copilot in fixing these code smells employing different prompts. The results show that 8 out of 10 types of Python smells can be detected in Copilot-generated Python code, among which Multiply-Nested Container is the most common one. For these code smells, Copilot Chat achieves a highest fixing rate of 87.1%, showing promise in fixing Python code smells generated by Copilot itself. Besides, the effectiveness of Copilot Chat in fixing these smells can be improved with the provision of more detailed prompts. However, using Copilot Chat to fix these smells might introduce new code smells.

{{</citation>}}


### (107/122) McUDI: Model-Centric Unsupervised Degradation Indicator for Failure Prediction AIOps Solutions (Lorena Poenaru-Olaru et al., 2024)

{{<citation>}}

Lorena Poenaru-Olaru, Luis Cruz, Jan Rellermeyer, Arie van Deursen. (2024)  
**McUDI: Model-Centric Unsupervised Degradation Indicator for Failure Prediction AIOps Solutions**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14093v1)  

---


**ABSTRACT**  
Due to the continuous change in operational data, AIOps solutions suffer from performance degradation over time. Although periodic retraining is the state-of-the-art technique to preserve the failure prediction AIOps models' performance over time, this technique requires a considerable amount of labeled data to retrain. In AIOps obtaining label data is expensive since it requires the availability of domain experts to intensively annotate it. In this paper, we present McUDI, a model-centric unsupervised degradation indicator that is capable of detecting the exact moment the AIOps model requires retraining as a result of changes in data. We further show how employing McUDI in the maintenance pipeline of AIOps solutions can reduce the number of samples that require annotations with 30k for job failure prediction and 260k for disk failure prediction while achieving similar performance with periodic retraining.

{{</citation>}}


### (108/122) From Requirements to Architecture: An AI-Based Journey to Semi-Automatically Generate Software Architectures (Tobias Eisenreich et al., 2024)

{{<citation>}}

Tobias Eisenreich, Sandro Speth, Stefan Wagner. (2024)  
**From Requirements to Architecture: An AI-Based Journey to Semi-Automatically Generate Software Architectures**  

---
Primary Category: cs.SE  
Categories: D-2-2, cs-AI, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14079v1)  

---


**ABSTRACT**  
Designing domain models and software architectures represents a significant challenge in software development, as the resulting architectures play a vital role in fulfilling the system's quality of service. Due to time pressure, architects often model only one architecture based on their known limited domain understanding, patterns, and experience instead of thoroughly analyzing the domain and evaluating multiple candidates, selecting the best fitting. Existing approaches try to generate domain models based on requirements, but still require time-consuming manual effort to achieve good results. Therefore, in this vision paper, we propose a method to generate software architecture candidates semi-automatically based on requirements using artificial intelligence techniques. We further envision an automatic evaluation and trade-off analysis of the generated architecture candidates using, e.g., the architecture trade-off analysis method combined with large language models and quantitative analyses. To evaluate this approach, we aim to analyze the quality of the generated architecture models and the efficiency and effectiveness of our proposed process by conducting qualitative studies.

{{</citation>}}


### (109/122) ChatGPT and Human Synergy in Black-Box Testing: A Comparative Analysis (Hiroyuki Kirinuki et al., 2024)

{{<citation>}}

Hiroyuki Kirinuki, Haruto Tanno. (2024)  
**ChatGPT and Human Synergy in Black-Box Testing: A Comparative Analysis**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.13924v1)  

---


**ABSTRACT**  
In recent years, large language models (LLMs), such as ChatGPT, have been pivotal in advancing various artificial intelligence applications, including natural language processing and software engineering. A promising yet underexplored area is utilizing LLMs in software testing, particularly in black-box testing. This paper explores the test cases devised by ChatGPT in comparison to those created by human participants. In this study, ChatGPT (GPT-4) and four participants each created black-box test cases for three applications based on specifications written by the authors. The goal was to evaluate the real-world applicability of the proposed test cases, identify potential shortcomings, and comprehend how ChatGPT could enhance human testing strategies. ChatGPT can generate test cases that generally match or slightly surpass those created by human participants in terms of test viewpoint coverage. Additionally, our experiments demonstrated that when ChatGPT cooperates with humans, it can cover considerably more test viewpoints than each can achieve alone, suggesting that collaboration between humans and ChatGPT may be more effective than human pairs working together. Nevertheless, we noticed that the test cases generated by ChatGPT have certain issues that require addressing before use.

{{</citation>}}


## cs.DC (1)



### (110/122) CHIRON: Accelerating Node Synchronization without Security Trade-offs in Distributed Ledgers (Ray Neiheiser et al., 2024)

{{<citation>}}

Ray Neiheiser, Arman Babaei, Giannis Alexopoulos, Marios Kogias, Eleftherios Kokoris Kogias. (2024)  
**CHIRON: Accelerating Node Synchronization without Security Trade-offs in Distributed Ledgers**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.14278v2)  

---


**ABSTRACT**  
Blockchain performance has historically faced challenges posed by the throughput limitations of consensus algorithms. Recent breakthroughs in research have successfully alleviated these constraints by introducing a modular architecture that decouples consensus from execution. The move toward independent optimization of the consensus layer has shifted attention to the execution layer.   While concurrent transaction execution is a promising solution for increasing throughput, practical challenges persist. Its effectiveness varies based on the workloads, and the associated increased hardware requirements raise concerns about undesirable centralization. This increased requirement results in full nodes and stragglers synchronizing from signed checkpoints, decreasing the trustless nature of blockchain systems.   In response to these challenges, this paper introduces Chiron, a system designed to extract execution hints for the acceleration of straggling and full nodes. Notably, Chiron achieves this without compromising the security of the system or introducing overhead on the critical path of consensus. Evaluation results demonstrate a notable speedup of up to 30%, effectively addressing the gap between theoretical research and practical deployment. The quantification of this speedup is achieved through realistic blockchain benchmarks derived from a comprehensive analysis of Ethereum and Solana workloads, constituting an independent contribution.

{{</citation>}}


## physics.optics (1)



### (111/122) Multicasting Optical Reconfigurable Switch (Niyazi Ulas Dinc et al., 2024)

{{<citation>}}

Niyazi Ulas Dinc, Mustafa Yildirim, Christophe Moser, Demetri Psaltis. (2024)  
**Multicasting Optical Reconfigurable Switch**  

---
Primary Category: physics.optics  
Categories: cs-NI, physics-optics, physics.optics  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14173v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) demands large data flows within datacenters, heavily relying on multicasting data transfers. As AI models scale, the requirement for high-bandwidth and low-latency networking compounds. The common use of electrical packet switching faces limitations due to its optical-electrical-optical conversion bottleneck. Optical switches, while bandwidth-agnostic and low-latency, suffer from having only unicast or non-scalable multicasting capability. This paper introduces an optical switching technique addressing the scalable multicasting challenge. Our approach enables arbitrarily programmable simultaneous unicast and multicast connectivity, eliminating the need for optical splitters that hinder scalability due to optical power loss. We use phase modulation in multiple planes, tailored to implement any multicast connectivity map. Using phase modulation enables wavelength selectivity on top of spatial selectivity, resulting in an optical switch that implements space-wavelength routing. We conducted simulations and experiments to validate our approach. Our results affirm the concept's feasibility and effectiveness, as a multicasting switch.

{{</citation>}}


## cs.SI (2)



### (112/122) Exploring the Distinctive Tweeting Patterns of Toxic Twitter Users (Hina Qayyum et al., 2024)

{{<citation>}}

Hina Qayyum, Muhammad Ikram, Benjamin Zi Hao Zhao, Ian D. Wood, Nicolas Kourtellis, Mohamed Ali Kaafar. (2024)  
**Exploring the Distinctive Tweeting Patterns of Toxic Twitter Users**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.14141v1)  

---


**ABSTRACT**  
In the pursuit of bolstering user safety, social media platforms deploy active moderation strategies, including content removal and user suspension. These measures target users engaged in discussions marked by hate speech or toxicity, often linked to specific keywords or hashtags. Nonetheless, the increasing prevalence of toxicity indicates that certain users adeptly circumvent these measures. This study examines consistently toxic users on Twitter (rebranded as X) Rather than relying on traditional methods based on specific topics or hashtags, we employ a novel approach based on patterns of toxic tweets, yielding deeper insights into their behavior. We analyzed 38 million tweets from the timelines of 12,148 Twitter users and identified the top 1,457 users who consistently exhibit toxic behavior, relying on metrics like the Gini index and Toxicity score. By comparing their posting patterns to those of non-consistently toxic users, we have uncovered distinctive temporal patterns, including contiguous activity spans, inter-tweet intervals (referred to as 'Burstiness'), and churn analysis. These findings provide strong evidence for the existence of a unique tweeting pattern associated with toxic behavior on Twitter. Crucially, our methodology transcends Twitter and can be adapted to various social media platforms, facilitating the identification of consistently toxic users based on their posting behavior. This research contributes to ongoing efforts to combat online toxicity and offers insights for refining moderation strategies in the digital realm. We are committed to open research and will provide our code and data to the research community.

{{</citation>}}


### (113/122) On the Feasibility of Simple Transformer for Dynamic Graph Modeling (Yuxia Wu et al., 2024)

{{<citation>}}

Yuxia Wu, Yuan Fang, Lizi Liao. (2024)  
**On the Feasibility of Simple Transformer for Dynamic Graph Modeling**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.14009v1)  

---


**ABSTRACT**  
Dynamic graph modeling is crucial for understanding complex structures in web graphs, spanning applications in social networks, recommender systems, and more. Most existing methods primarily emphasize structural dependencies and their temporal changes. However, these approaches often overlook detailed temporal aspects or struggle with long-term dependencies. Furthermore, many solutions overly complicate the process by emphasizing intricate module designs to capture dynamic evolutions. In this work, we harness the strength of the Transformer's self-attention mechanism, known for adeptly handling long-range dependencies in sequence modeling. Our approach offers a simple Transformer model tailored for dynamic graph modeling without complex modifications. We re-conceptualize dynamic graphs as a sequence modeling challenge and introduce an innovative temporal alignment technique. This technique not only captures the inherent temporal evolution patterns within dynamic graphs but also streamlines the modeling process of their evolution. As a result, our method becomes versatile, catering to an array of applications. Our model's effectiveness is underscored through rigorous experiments on four real-world datasets from various sectors, solidifying its potential in dynamic graph modeling.

{{</citation>}}


## quant-ph (1)



### (114/122) GQHAN: A Grover-inspired Quantum Hard Attention Network (Ren-Xin Zhao et al., 2024)

{{<citation>}}

Ren-Xin Zhao, Jinjing Shi, Xuelong Li. (2024)  
**GQHAN: A Grover-inspired Quantum Hard Attention Network**  

---
Primary Category: quant-ph  
Categories: cs-AI, quant-ph, quant-ph  
Keywords: Attention, QA  
[Paper Link](http://arxiv.org/abs/2401.14089v1)  

---


**ABSTRACT**  
Numerous current Quantum Machine Learning (QML) models exhibit an inadequacy in discerning the significance of quantum data, resulting in diminished efficacy when handling extensive quantum datasets. Hard Attention Mechanism (HAM), anticipated to efficiently tackle the above QML bottlenecks, encounters the substantial challenge of non-differentiability, consequently constraining its extensive applicability. In response to the dilemma of HAM and QML, a Grover-inspired Quantum Hard Attention Mechanism (GQHAM) consisting of a Flexible Oracle (FO) and an Adaptive Diffusion Operator (ADO) is proposed. Notably, the FO is designed to surmount the non-differentiable issue by executing the activation or masking of Discrete Primitives (DPs) with Flexible Control (FC) to weave various discrete destinies. Based on this, such discrete choice can be visualized with a specially defined Quantum Hard Attention Score (QHAS). Furthermore, a trainable ADO is devised to boost the generality and flexibility of GQHAM. At last, a Grover-inspired Quantum Hard Attention Network (GQHAN) based on QGHAM is constructed on PennyLane platform for Fashion MNIST binary classification. Experimental findings demonstrate that GQHAN adeptly surmounts the non-differentiability hurdle, surpassing the efficacy of extant quantum soft self-attention mechanisms in accuracies and learning ability. In noise experiments, GQHAN is robuster to bit-flip noise in accuracy and amplitude damping noise in learning performance. Predictably, the proposal of GQHAN enriches the Quantum Attention Mechanism (QAM), lays the foundation for future quantum computers to process large-scale data, and promotes the development of quantum computer vision.

{{</citation>}}


## cs.AI (3)



### (115/122) Generating Likely Counterfactuals Using Sum-Product Networks (Jiri Nemecek et al., 2024)

{{<citation>}}

Jiri Nemecek, Tomas Pevny, Jakub Marecek. (2024)  
**Generating Likely Counterfactuals Using Sum-Product Networks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI, math-OC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.14086v1)  

---


**ABSTRACT**  
Due to user demand and recent regulation (GDPR, AI Act), decisions made by AI systems need to be explained. These decisions are often explainable only post hoc, where counterfactual explanations are popular. The question of what constitutes the best counterfactual explanation must consider multiple aspects, where "distance from the sample" is the most common. We argue that this requirement frequently leads to explanations that are unlikely and, therefore, of limited value. Here, we present a system that provides high-likelihood explanations. We show that the search for the most likely explanations satisfying many common desiderata for counterfactual explanations can be modeled using mixed-integer optimization (MIO). In the process, we propose an MIO formulation of a Sum-Product Network (SPN) and use the SPN to estimate the likelihood of a counterfactual, which can be of independent interest. A numerical comparison against several methods for generating counterfactual explanations is provided.

{{</citation>}}


### (116/122) A New Paradigm for Counterfactual Reasoning in Fairness and Recourse (Lucius E. J. Bynum et al., 2024)

{{<citation>}}

Lucius E. J. Bynum, Joshua R. Loftus, Julia Stoyanovich. (2024)  
**A New Paradigm for Counterfactual Reasoning in Fairness and Recourse**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI, stat-ML  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.13935v1)  

---


**ABSTRACT**  
Counterfactuals and counterfactual reasoning underpin numerous techniques for auditing and understanding artificial intelligence (AI) systems. The traditional paradigm for counterfactual reasoning in this literature is the interventional counterfactual, where hypothetical interventions are imagined and simulated. For this reason, the starting point for causal reasoning about legal protections and demographic data in AI is an imagined intervention on a legally-protected characteristic, such as ethnicity, race, gender, disability, age, etc. We ask, for example, what would have happened had your race been different? An inherent limitation of this paradigm is that some demographic interventions -- like interventions on race -- may not translate into the formalisms of interventional counterfactuals. In this work, we explore a new paradigm based instead on the backtracking counterfactual, where rather than imagine hypothetical interventions on legally-protected characteristics, we imagine alternate initial conditions while holding these characteristics fixed. We ask instead, what would explain a counterfactual outcome for you as you actually are or could be? This alternate framework allows us to address many of the same social concerns, but to do so while asking fundamentally different questions that do not rely on demographic interventions.

{{</citation>}}


### (117/122) Domain-Independent Dynamic Programming (Ryo Kuroiwa et al., 2024)

{{<citation>}}

Ryo Kuroiwa, J. Christopher Beck. (2024)  
**Domain-Independent Dynamic Programming**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13883v1)  

---


**ABSTRACT**  
For combinatorial optimization problems, model-based paradigms such as mixed-integer programming (MIP) and constraint programming (CP) aim to decouple modeling and solving a problem: the `holy grail' of declarative problem solving. We propose domain-independent dynamic programming (DIDP), a new model-based paradigm based on dynamic programming (DP). While DP is not new, it has typically been implemented as a problem-specific method. We introduce Dynamic Programming Description Language (DyPDL), a formalism to define DP models based on a state transition system, inspired by AI planning. We show that heuristic search algorithms can be used to solve DyPDL models and propose seven DIDP solvers. We experimentally compare our DIDP solvers with commercial MIP and CP solvers (solving MIP and CP models, respectively) on common benchmark instances of eleven combinatorial optimization problem classes. We show that DIDP outperforms MIP in nine problem classes, CP also in nine problem classes, and both MIP and CP in seven.

{{</citation>}}


## cs.DS (1)



### (118/122) On Sparse Covers of Minor Free Graphs, Low Dimensional Metric Embeddings, and other applications (Arnold Filtser, 2024)

{{<citation>}}

Arnold Filtser. (2024)  
**On Sparse Covers of Minor Free Graphs, Low Dimensional Metric Embeddings, and other applications**  

---
Primary Category: cs.DS  
Categories: cs-CG, cs-DS, cs.DS, math-CO  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.14060v1)  

---


**ABSTRACT**  
Given a metric space $(X,d_X)$, a $(\beta,s,\Delta)$-sparse cover is a collection of clusters $\mathcal{C}\subseteq P(X)$ with diameter at most $\Delta$, such that for every point $x\in X$, the ball $B_X(x,\frac\Delta\beta)$ is fully contained in some cluster $C\in \mathcal{C}$, and $x$ belongs to at most $s$ clusters in $\mathcal{C}$. Our main contribution is to show that the shortest path metric of every $K_r$-minor free graphs admits $(O(r),O(r^2),\Delta)$-sparse cover, and for every $\epsilon>0$, $(4+\epsilon,O(\frac1\epsilon)^r,\Delta)$-sparse cover (for arbitrary $\Delta>0$). We then use this sparse cover to show that every $K_r$-minor free graph embeds into $\ell_\infty^{\tilde{O}(\frac1\epsilon)^{r+1}\cdot\log n}$ with distortion $3+\eps$ (resp. into $\ell_\infty^{\tilde{O}(r^2)\cdot\log n}$ with distortion $O(r)$). Further, we provide applications of these sparse covers into padded decompositions, sparse partitions, universal TSP / Steiner tree, oblivious buy at bulk, name independent routing, and path reporting distance oracles.

{{</citation>}}


## eess.SY (2)



### (119/122) Networked Multiagent Reinforcement Learning for Peer-to-Peer Energy Trading (Chen Feng et al., 2024)

{{<citation>}}

Chen Feng, Andrew L. Liu. (2024)  
**Networked Multiagent Reinforcement Learning for Peer-to-Peer Energy Trading**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-MA, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.13947v2)  

---


**ABSTRACT**  
Utilizing distributed renewable and energy storage resources in local distribution networks via peer-to-peer (P2P) energy trading has long been touted as a solution to improve energy systems' resilience and sustainability. Consumers and prosumers (those who have energy generation resources), however, do not have the expertise to engage in repeated P2P trading, and the zero-marginal costs of renewables present challenges in determining fair market prices. To address these issues, we propose multi-agent reinforcement learning (MARL) frameworks to help automate consumers' bidding and management of their solar PV and energy storage resources, under a specific P2P clearing mechanism that utilizes the so-called supply-demand ratio. In addition, we show how the MARL frameworks can integrate physical network constraints to realize voltage control, hence ensuring physical feasibility of the P2P energy trading and paving way for real-world implementations.

{{</citation>}}


### (120/122) Learning-based sensing and computing decision for data freshness in edge computing-enabled networks (Sinwoong Yun et al., 2024)

{{<citation>}}

Sinwoong Yun, Dongsun Kim, Chanwon Park, Jemin Lee. (2024)  
**Learning-based sensing and computing decision for data freshness in edge computing-enabled networks**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13936v1)  

---


**ABSTRACT**  
As the demand on artificial intelligence (AI)-based applications increases, the freshness of sensed data becomes crucial in the wireless sensor networks. Since those applications require a large amount of computation for processing the sensed data, it is essential to offload the computation load to the edge computing (EC) server. In this paper, we propose the sensing and computing decision (SCD) algorithms for data freshness in the EC-enabled wireless sensor networks. We define the {\eta}-coverage probability to show the probability of maintaining fresh data for more than {\eta} ratio of the network, where the spatial-temporal correlation of information is considered. We then propose the probability-based SCD for the single pre-charged sensor case with providing the optimal point after deriving the {\eta}-coverage probability. We also propose the reinforcement learning (RL)- based SCD by training the SCD policy of sensors for both the single pre-charged and multiple energy harvesting (EH) sensor cases, to make a real-time decision based on its observation. Our simulation results verify the performance of the proposed algorithms under various environment settings, and show that the RL-based SCD algorithm achieves higher performance compared to baseline algorithms for both the single pre-charged sensor and multiple EH sensor cases.

{{</citation>}}


## stat.ML (1)



### (121/122) Constant Stepsize Q-learning: Distributional Convergence, Bias and Extrapolation (Yixuan Zhang et al., 2024)

{{<citation>}}

Yixuan Zhang, Qiaomin Xie. (2024)  
**Constant Stepsize Q-learning: Distributional Convergence, Bias and Extrapolation**  

---
Primary Category: stat.ML  
Categories: cs-LG, math-OC, stat-ML, stat.ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.13884v1)  

---


**ABSTRACT**  
Stochastic Approximation (SA) is a widely used algorithmic approach in various fields, including optimization and reinforcement learning (RL). Among RL algorithms, Q-learning is particularly popular due to its empirical success. In this paper, we study asynchronous Q-learning with constant stepsize, which is commonly used in practice for its fast convergence. By connecting the constant stepsize Q-learning to a time-homogeneous Markov chain, we show the distributional convergence of the iterates in Wasserstein distance and establish its exponential convergence rate. We also establish a Central Limit Theory for Q-learning iterates, demonstrating the asymptotic normality of the averaged iterates. Moreover, we provide an explicit expansion of the asymptotic bias of the averaged iterate in stepsize. Specifically, the bias is proportional to the stepsize up to higher-order terms and we provide an explicit expression for the linear coefficient. This precise characterization of the bias allows the application of Richardson-Romberg (RR) extrapolation technique to construct a new estimate that is provably closer to the optimal Q function. Numerical results corroborate our theoretical finding on the improvement of the RR extrapolation method.

{{</citation>}}


## cs.IR (1)



### (122/122) Integrating Large Language Models into Recommendation via Mutual Augmentation and Adaptive Aggregation (Sichun Luo et al., 2024)

{{<citation>}}

Sichun Luo, Yuxuan Yao, Bowei He, Yinya Huang, Aojun Zhou, Xinyi Zhang, Yuanzhang Xiao, Mingjie Zhan, Linqi Song. (2024)  
**Integrating Large Language Models into Recommendation via Mutual Augmentation and Adaptive Aggregation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Augmentation, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13870v1)  

---


**ABSTRACT**  
Conventional recommendation methods have achieved notable advancements by harnessing collaborative or sequential information from user behavior. Recently, large language models (LLMs) have gained prominence for their capabilities in understanding and reasoning over textual semantics, and have found utility in various domains, including recommendation. Conventional recommendation methods and LLMs each have their strengths and weaknesses. While conventional methods excel at mining collaborative information and modeling sequential behavior, they struggle with data sparsity and the long-tail problem. LLMs, on the other hand, are proficient at utilizing rich textual contexts but face challenges in mining collaborative or sequential information. Despite their individual successes, there is a significant gap in leveraging their combined potential to enhance recommendation performance.   In this paper, we introduce a general and model-agnostic framework known as \textbf{L}arge \textbf{la}nguage model with \textbf{m}utual augmentation and \textbf{a}daptive aggregation for \textbf{Rec}ommendation (\textbf{Llama4Rec}). Llama4Rec synergistically combines conventional and LLM-based recommendation models. Llama4Rec proposes data augmentation and prompt augmentation strategies tailored to enhance the conventional model and LLM respectively. An adaptive aggregation module is adopted to combine the predictions of both kinds of models to refine the final recommendation results. Empirical studies on three real-world datasets validate the superiority of Llama4Rec, demonstrating its consistent outperformance of baseline methods and significant improvements in recommendation performance.

{{</citation>}}
