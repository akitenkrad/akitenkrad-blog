---
draft: false
title: "arXiv @ 2024.01.14"
date: 2024-01-14
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.14"
    identifier: arxiv_20240114
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (47)](#cscl-47)
- [cs.LG (16)](#cslg-16)
- [cs.IR (9)](#csir-9)
- [eess.IV (2)](#eessiv-2)
- [cs.AR (2)](#csar-2)
- [cs.SE (6)](#csse-6)
- [cs.CV (16)](#cscv-16)
- [physics.flu-dyn (1)](#physicsflu-dyn-1)
- [cs.NI (3)](#csni-3)
- [cs.DC (1)](#csdc-1)
- [eess.AS (3)](#eessas-3)
- [cs.SI (2)](#cssi-2)
- [cs.CR (1)](#cscr-1)
- [eess.SY (2)](#eesssy-2)
- [cs.CY (1)](#cscy-1)
- [cs.DB (1)](#csdb-1)
- [math.NA (1)](#mathna-1)
- [cs.AI (3)](#csai-3)
- [cs.GR (1)](#csgr-1)
- [cs.IT (1)](#csit-1)
- [cs.HC (4)](#cshc-4)
- [cs.RO (1)](#csro-1)
- [cs.SD (1)](#cssd-1)
- [cs.MM (1)](#csmm-1)

## cs.CL (47)



### (1/126) PizzaCommonSense: Learning to Model Commonsense Reasoning about Intermediate Steps in Cooking Recipes (Aissatou Diallo et al., 2024)

{{<citation>}}

Aissatou Diallo, Antonis Bikakis, Luke Dickens, Anthony Hunter, Rob Miller. (2024)  
**PizzaCommonSense: Learning to Model Commonsense Reasoning about Intermediate Steps in Cooking Recipes**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Reasoning, T5  
[Paper Link](http://arxiv.org/abs/2401.06930v1)  

---


**ABSTRACT**  
Decoding the core of procedural texts, exemplified by cooking recipes, is crucial for intelligent reasoning and instruction automation. Procedural texts can be comprehensively defined as a sequential chain of steps to accomplish a task employing resources. From a cooking perspective, these instructions can be interpreted as a series of modifications to a food preparation, which initially comprises a set of ingredients. These changes involve transformations of comestible resources. For a model to effectively reason about cooking recipes, it must accurately discern and understand the inputs and outputs of intermediate steps within the recipe. Aiming to address this, we present a new corpus of cooking recipes enriched with descriptions of intermediate steps of the recipes that explicate the input and output for each step. We discuss the data collection process, investigate and provide baseline models based on T5 and GPT-3.5. This work presents a challenging task and insight into commonsense reasoning and procedural text generation.

{{</citation>}}


### (2/126) Comparing GPT-4 and Open-Source Language Models in Misinformation Mitigation (Tyler Vergho et al., 2024)

{{<citation>}}

Tyler Vergho, Jean-Francois Godbout, Reihaneh Rabbany, Kellin Pelrine. (2024)  
**Comparing GPT-4 and Open-Source Language Models in Misinformation Mitigation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06920v1)  

---


**ABSTRACT**  
Recent large language models (LLMs) have been shown to be effective for misinformation detection. However, the choice of LLMs for experiments varies widely, leading to uncertain conclusions. In particular, GPT-4 is known to be strong in this domain, but it is closed source, potentially expensive, and can show instability between different versions. Meanwhile, alternative LLMs have given mixed results. In this work, we show that Zephyr-7b presents a consistently viable alternative, overcoming key limitations of commonly used approaches like Llama-2 and GPT-3.5. This provides the research community with a solid open-source option and shows open-source models are gradually catching up on this task. We then highlight how GPT-3.5 exhibits unstable performance, such that this very widely used model could provide misleading results in misinformation detection. Finally, we validate new tools including approaches to structured output and the latest version of GPT-4 (Turbo), showing they do not compromise performance, thus unlocking them for future research and potentially enabling more complex pipelines for misinformation mitigation.

{{</citation>}}


### (3/126) DocFinQA: A Long-Context Financial Reasoning Dataset (Varshini Reddy et al., 2024)

{{<citation>}}

Varshini Reddy, Rik Koncel-Kedziorski, Viet Dac Lai, Chris Tanner. (2024)  
**DocFinQA: A Long-Context Financial Reasoning Dataset**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Financial, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.06915v1)  

---


**ABSTRACT**  
Research in quantitative reasoning within the financial domain indeed necessitates the use of realistic tasks and data, primarily because of the significant impact of decisions made in business and finance. Financial professionals often interact with documents hundreds of pages long, but most research datasets drastically reduce this context length. To address this, we introduce a long-document financial QA task. We augment 7,621 questions from the existing FinQA dataset with full-document context, extending the average context length for each question from under 700 words in FinQA to 123k words in DocFinQA. We conduct extensive experiments of retrieval-based QA pipelines and long-context language models on the augmented data. Our results show that DocFinQA provides challenges for even the strongest, state-of-the-art systems.

{{</citation>}}


### (4/126) Promptly Predicting Structures: The Return of Inference (Maitrey Mehta et al., 2024)

{{<citation>}}

Maitrey Mehta, Valentina Pyatkin, Vivek Srikumar. (2024)  
**Promptly Predicting Structures: The Return of Inference**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2401.06877v1)  

---


**ABSTRACT**  
Prompt-based methods have been used extensively across NLP to build zero- and few-shot label predictors. Many NLP tasks are naturally structured: that is, their outputs consist of multiple labels which constrain each other. Annotating data for such tasks can be cumbersome. Can the promise of the prompt-based paradigm be extended to such structured outputs? In this paper, we present a framework for constructing zero- and few-shot linguistic structure predictors. Our key insight is that we can use structural constraints -- and combinatorial inference derived from them -- to filter out inconsistent structures predicted by large language models. We instantiated this framework on two structured prediction tasks, and five datasets. Across all cases, our results show that enforcing consistency not only constructs structurally valid outputs, but also improves performance over the unconstrained variants.

{{</citation>}}


### (5/126) Health-LLM: Large Language Models for Health Prediction via Wearable Sensor Data (Yubin Kim et al., 2024)

{{<citation>}}

Yubin Kim, Xuhai Xu, Daniel McDuff, Cynthia Breazeal, Hae Won Park. (2024)  
**Health-LLM: Large Language Models for Health Prediction via Wearable Sensor Data**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06866v1)  

---


**ABSTRACT**  
Large language models (LLMs) are capable of many natural language tasks, yet they are far from perfect. In health applications, grounding and interpreting domain-specific and non-linguistic data is important. This paper investigates the capacity of LLMs to deliver multi-modal health predictions based on contextual information (e.g. user demographics, health knowledge) and physiological data (e.g. resting heart rate, sleep minutes). We present a comprehensive evaluation of eight state-of-the-art LLMs with diverse prompting and fine-tuning techniques on six public health datasets (PM-Data, LifeSnaps, GLOBEM, AW_FB, MIT-BIH & MIMIC-III). Our experiments cover thirteen consumer health prediction tasks in mental health, activity, metabolic, sleep, and cardiac assessment. Our fine-tuned model, Health-Alpaca exhibits comparable performance to larger models (GPT-3.5 and GPT-4), achieving the best performance in 5 out of 13 tasks. Ablation studies highlight the effectiveness of context enhancement strategies, and generalization capability of the fine-tuned models across training datasets and the size of training samples. Notably, we observe that our context enhancement can yield up to 23.8% improvement in performance. While constructing contextually rich prompts (combining user context, health knowledge and temporal information) exhibits synergistic improvement, the inclusion of health knowledge context in prompts significantly enhances overall performance.

{{</citation>}}


### (6/126) Fine-grained Hallucination Detection and Editing for Language Models (Abhika Mishra et al., 2024)

{{<citation>}}

Abhika Mishra, Akari Asai, Vidhisha Balachandran, Yizhong Wang, Graham Neubig, Yulia Tsvetkov, Hannaneh Hajishirzi. (2024)  
**Fine-grained Hallucination Detection and Editing for Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06855v2)  

---


**ABSTRACT**  
Large language models (LMs) are prone to generate diverse factually incorrect statements, which are widely called hallucinations. Current approaches predominantly focus on coarse-grained automatic hallucination detection or editing, overlooking nuanced error levels. In this paper, we propose a novel task -- automatic fine-grained hallucination detection -- and present a comprehensive taxonomy encompassing six hierarchically defined types of hallucination. To facilitate evaluation, we introduce a new benchmark that includes fine-grained human judgments on two LM outputs across various domains. Our analysis reveals that ChatGPT and Llama 2-Chat exhibit hallucinations in 60% and 75% of their outputs, respectively, and a majority of these hallucinations fall into categories that have been underexplored. As an initial step to address this, we train FAVA, a retrieval-augmented LM by carefully designing synthetic data generations to detect and correct fine-grained hallucinations. On our benchmark, our automatic and human evaluations show that FAVA significantly outperforms ChatGPT on fine-grained hallucination detection by a large margin though a large room for future improvement still exists. FAVA's suggested edits also improve the factuality of LM-generated text, resulting in 5-10% FActScore improvements.

{{</citation>}}


### (7/126) Large Language Models Can Learn Temporal Reasoning (Siheng Xiong et al., 2024)

{{<citation>}}

Siheng Xiong, Ali Payani, Ramana Kompella, Faramarz Fekri. (2024)  
**Large Language Models Can Learn Temporal Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.06853v1)  

---


**ABSTRACT**  
Large language models (LLMs) learn temporal concepts from the co-occurrence of related tokens in a sequence. Compared with conventional text generation, temporal reasoning, which reaches a conclusion based on mathematical, logical and commonsense knowledge, is more challenging. In this paper, we propose TempGraph-LLM, a new paradigm towards text-based temporal reasoning. To be specific, we first teach LLMs to translate the context into a temporal graph. A synthetic dataset, which is fully controllable and requires minimal supervision, is constructed for pre-training on this task. We prove in experiments that LLMs benefit from the pre-training on other tasks. On top of that, we guide LLMs to perform symbolic reasoning with the strategies of Chain of Thoughts (CoTs) bootstrapping and special data augmentation. We observe that CoTs with symbolic reasoning bring more consistent and reliable results than those using free text.

{{</citation>}}


### (8/126) Machine Translation Models are Zero-Shot Detectors of Translation Direction (Michelle Wastl et al., 2024)

{{<citation>}}

Michelle Wastl, Jannis Vamvas, Rico Sennrich. (2024)  
**Machine Translation Models are Zero-Shot Detectors of Translation Direction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation, NLP, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.06769v1)  

---


**ABSTRACT**  
Detecting the translation direction of parallel text has applications for machine translation training and evaluation, but also has forensic applications such as resolving plagiarism or forgery allegations. In this work, we explore an unsupervised approach to translation direction detection based on the simple hypothesis that $p(\text{translation}|\text{original})>p(\text{original}|\text{translation})$, motivated by the well-known simplification effect in translationese or machine-translationese. In experiments with massively multilingual machine translation models across 20 translation directions, we confirm the effectiveness of the approach for high-resource language pairs, achieving document-level accuracies of 82-96% for NMT-produced translations, and 60-81% for human translations, depending on the model used. Code and demo are available at https://github.com/ZurichNLP/translation-direction-detection

{{</citation>}}


### (9/126) Navigating the Metrics Maze: Reconciling Score Magnitudes and Accuracies (Tom Kocmi et al., 2024)

{{<citation>}}

Tom Kocmi, Vilém Zouhar, Christian Federmann, Matt Post. (2024)  
**Navigating the Metrics Maze: Reconciling Score Magnitudes and Accuracies**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2401.06760v1)  

---


**ABSTRACT**  
Ten years ago a single metric, BLEU, governed progress in machine translation research. For better or worse, there is no such consensus today, and consequently it is difficult for researchers to develop and retain the kinds of heuristic intuitions about metric deltas that drove earlier research and deployment decisions. This paper investigates the "dynamic range" of a number of modern metrics in an effort to provide a collective understanding of the meaning of differences in scores both within and among metrics; in other words, we ask what point difference X in metric Y is required between two systems for humans to notice? We conduct our evaluation on a new large dataset, ToShip23, using it to discover deltas at which metrics achieve system-level differences that are meaningful to humans, which we measure by pairwise system accuracy. We additionally show that this method of establishing delta-accuracy is more stable than the standard use of statistical p-values in regards to testset size. Where data size permits, we also explore the effect of metric deltas and accuracy across finer-grained features such as translation direction, domain, and system closeness.

{{</citation>}}


### (10/126) Stylometry Analysis of Multi-authored Documents for Authorship and Author Style Change Detection (Muhammad Tayyab Zamir et al., 2024)

{{<citation>}}

Muhammad Tayyab Zamir, Muhammad Asif Ayub, Asma Gul, Nasir Ahmad, Kashif Ahmad. (2024)  
**Stylometry Analysis of Multi-authored Documents for Authorship and Author Style Change Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2401.06752v1)  

---


**ABSTRACT**  
In recent years, the increasing use of Artificial Intelligence based text generation tools has posed new challenges in document provenance, authentication, and authorship detection. However, advancements in stylometry have provided opportunities for automatic authorship and author change detection in multi-authored documents using style analysis techniques. Style analysis can serve as a primary step toward document provenance and authentication through authorship detection. This paper investigates three key tasks of style analysis: (i) classification of single and multi-authored documents, (ii) single change detection, which involves identifying the point where the author switches, and (iii) multiple author-switching detection in multi-authored documents. We formulate all three tasks as classification problems and propose a merit-based fusion framework that integrates several state-of-the-art natural language processing (NLP) algorithms and weight optimization techniques. We also explore the potential of special characters, which are typically removed during pre-processing in NLP applications, on the performance of the proposed methods for these tasks by conducting extensive experiments on both cleaned and raw datasets. Experimental results demonstrate significant improvements over existing solutions for all three tasks on a benchmark dataset.

{{</citation>}}


### (11/126) Using Natural Language Inference to Improve Persona Extraction from Dialogue in a New Domain (Alexandra DeLucia et al., 2024)

{{<citation>}}

Alexandra DeLucia, Mengjie Zhao, Yoshinori Maeda, Makoto Yoda, Keiichi Yamada, Hiromi Wakaki. (2024)  
**Using Natural Language Inference to Improve Persona Extraction from Dialogue in a New Domain**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue, NLI, Natural Language Inference  
[Paper Link](http://arxiv.org/abs/2401.06742v1)  

---


**ABSTRACT**  
While valuable datasets such as PersonaChat provide a foundation for training persona-grounded dialogue agents, they lack diversity in conversational and narrative settings, primarily existing in the "real" world. To develop dialogue agents with unique personas, models are trained to converse given a specific persona, but hand-crafting these persona can be time-consuming, thus methods exist to automatically extract persona information from existing character-specific dialogue. However, these persona-extraction models are also trained on datasets derived from PersonaChat and struggle to provide high-quality persona information from conversational settings that do not take place in the real world, such as the fantasy-focused dataset, LIGHT. Creating new data to train models on a specific setting is human-intensive, thus prohibitively expensive. To address both these issues, we introduce a natural language inference method for post-hoc adapting a trained persona extraction model to a new setting. We draw inspiration from the literature of dialog natural language inference (NLI), and devise NLI-reranking methods to extract structured persona information from dialogue. Compared to existing persona extraction models, our method returns higher-quality extracted persona and requires less human annotation.

{{</citation>}}


### (12/126) MAPO: Advancing Multilingual Reasoning through Multilingual Alignment-as-Preference Optimization (Shuaijie She et al., 2024)

{{<citation>}}

Shuaijie She, Shujian Huang, Wei Zou, Wenhao Zhu, Xiang Liu, Xiang Geng, Jiajun Chen. (2024)  
**MAPO: Advancing Multilingual Reasoning through Multilingual Alignment-as-Preference Optimization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.06838v1)  

---


**ABSTRACT**  
Though reasoning abilities are considered language-agnostic, existing LLMs exhibit inconsistent reasoning abilities across different languages, e.g., reasoning in a pivot language is superior to other languages due to the imbalance of multilingual training data.To enhance reasoning abilities in non-pivot languages, we propose an alignment-as-preference optimization framework. Specifically, we adopt an open-source translation model to estimate the consistency between answers in non-pivot and pivot languages. We further adopt the answer consistency as the preference for DPO or PPO thus optimizing the lesser reasoning. Experiments show that our method significantly improves the model's multilingual reasoning, with better reasoning consistency across languages. Our framework achieved a 13.7% accuracy improvement on out-of-domain datasets MSVAMP while preserving the competitive performance on MGSM. Moreover, we find that iterative DPO is helpful for further alignment and improvement of the model's multilingual mathematical reasoning ability, further pushing the improvement to 16.7%

{{</citation>}}


### (13/126) Relying on the Unreliable: The Impact of Language Models' Reluctance to Express Uncertainty (Kaitlyn Zhou et al., 2024)

{{<citation>}}

Kaitlyn Zhou, Jena D. Hwang, Xiang Ren, Maarten Sap. (2024)  
**Relying on the Unreliable: The Impact of Language Models' Reluctance to Express Uncertainty**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06730v1)  

---


**ABSTRACT**  
As natural language becomes the default interface for human-AI interaction, there is a critical need for LMs to appropriately communicate uncertainties in downstream applications. In this work, we investigate how LMs incorporate confidence about their responses via natural language and how downstream users behave in response to LM-articulated uncertainties. We examine publicly deployed models and find that LMs are unable to express uncertainties when answering questions even when they produce incorrect responses. LMs can be explicitly prompted to express confidences, but tend to be overconfident, resulting in high error rates (on average 47%) among confident responses. We test the risks of LM overconfidence by running human experiments and show that users rely heavily on LM generations, whether or not they are marked by certainty. Lastly, we investigate the preference-annotated datasets used in RLHF alignment and find that humans have a bias against texts with uncertainty. Our work highlights a new set of safety harms facing human-LM interactions and proposes design recommendations and mitigating strategies moving forward.

{{</citation>}}


### (14/126) Structsum Generation for Faster Text Comprehension (Parag Jain et al., 2024)

{{<citation>}}

Parag Jain, Andreea Marzoca, Francesco Piccinno. (2024)  
**Structsum Generation for Faster Text Comprehension**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.06837v1)  

---


**ABSTRACT**  
We consider the task of generating structured representations of text using large language models (LLMs). We focus on tables and mind maps as representative modalities. Tables are more organized way of representing data, while mind maps provide a visually dynamic and flexible approach, particularly suitable for sparse content. Despite the effectiveness of LLMs on different tasks, we show that current models struggle with generating structured outputs. In response, we present effective prompting strategies for both of these tasks. We introduce a taxonomy of problems around factuality, global and local structure, common to both modalities and propose a set of critiques to tackle these issues resulting in an absolute improvement in accuracy of +37pp (79%) for mind maps and +15pp (78%) for tables. To evaluate semantic coverage of generated structured representations we propose Auto-QA, and we verify the adequacy of Auto-QA using SQuAD dataset. We further evaluate the usefulness of structured representations via a text comprehension user study. The results show a significant reduction in comprehension time compared to text when using table (42.9%) and mind map (31.9%), without loss in accuracy.

{{</citation>}}


### (15/126) Reframing Tax Law Entailment as Analogical Reasoning (Xinrui Zou et al., 2024)

{{<citation>}}

Xinrui Zou, Ming Zhang, Nathaniel Weir, Benjamin Van Durme, Nils Holzenberger. (2024)  
**Reframing Tax Law Entailment as Analogical Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Natural Language Processing, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.06715v1)  

---


**ABSTRACT**  
Statutory reasoning refers to the application of legislative provisions to a series of case facts described in natural language. We re-frame statutory reasoning as an analogy task, where each instance of the analogy task involves a combination of two instances of statutory reasoning. This increases the dataset size by two orders of magnitude, and introduces an element of interpretability. We show that this task is roughly as difficult to Natural Language Processing models as the original task. Finally, we come back to statutory reasoning, solving it with a combination of a retrieval mechanism and analogy models, and showing some progress on prior comparable work.

{{</citation>}}


### (16/126) Few-Shot Detection of Machine-Generated Text using Style Representations (Rafael Rivera Soto et al., 2024)

{{<citation>}}

Rafael Rivera Soto, Kailin Koch, Aleem Khan, Barry Chen, Marcus Bishop, Nicholas Andrews. (2024)  
**Few-Shot Detection of Machine-Generated Text using Style Representations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, Few-Shot, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.06712v1)  

---


**ABSTRACT**  
The advent of instruction-tuned language models that convincingly mimic human writing poses a significant risk of abuse. For example, such models could be used for plagiarism, disinformation, spam, or phishing. However, such abuse may be counteracted with the ability to detect whether a piece of text was composed by a language model rather than a human. Some previous approaches to this problem have relied on supervised methods trained on corpora of confirmed human and machine-written documents. Unfortunately, model under-specification poses an unavoidable challenge for neural network-based detectors, making them brittle in the face of data shifts, such as the release of further language models producing still more fluent text than the models used to train the detectors. Other previous approaches require access to the models that may have generated a document in question at inference or detection time, which is often impractical. In light of these challenges, we pursue a fundamentally different approach not relying on samples from language models of concern at training time. Instead, we propose to leverage representations of writing style estimated from human-authored text. Indeed, we find that features effective at distinguishing among human authors are also effective at distinguishing human from machine authors, including state of the art large language models like Llama 2, ChatGPT, and GPT-4. Furthermore, given a handful of examples composed by each of several specific language models of interest, our approach affords the ability to predict which model generated a given document.

{{</citation>}}


### (17/126) Reliability Analysis of Psychological Concept Extraction and Classification in User-penned Text (Muskan Garg et al., 2024)

{{<citation>}}

Muskan Garg, MSVPJ Sathvik, Amrit Chadha, Shaina Raza, Sunghwan Sohn. (2024)  
**Reliability Analysis of Psychological Concept Extraction and Classification in User-penned Text**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, NLP  
[Paper Link](http://arxiv.org/abs/2401.06709v1)  

---


**ABSTRACT**  
The social NLP research community witness a recent surge in the computational advancements of mental health analysis to build responsible AI models for a complex interplay between language use and self-perception. Such responsible AI models aid in quantifying the psychological concepts from user-penned texts on social media. On thinking beyond the low-level (classification) task, we advance the existing binary classification dataset, towards a higher-level task of reliability analysis through the lens of explanations, posing it as one of the safety measures. We annotate the LoST dataset to capture nuanced textual cues that suggest the presence of low self-esteem in the posts of Reddit users. We further state that the NLP models developed for determining the presence of low self-esteem, focus more on three types of textual cues: (i) Trigger: words that triggers mental disturbance, (ii) LoST indicators: text indicators emphasizing low self-esteem, and (iii) Consequences: words describing the consequences of mental disturbance. We implement existing classifiers to examine the attention mechanism in pre-trained language models (PLMs) for a domain-specific psychology-grounded task. Our findings suggest the need of shifting the focus of PLMs from Trigger and Consequences to a more comprehensive explanation, emphasizing LoST indicators while determining low self-esteem in Reddit posts.

{{</citation>}}


### (18/126) Multi-Candidate Speculative Decoding (Sen Yang et al., 2024)

{{<citation>}}

Sen Yang, Shujian Huang, Xinyu Dai, Jiajun Chen. (2024)  
**Multi-Candidate Speculative Decoding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2401.06706v1)  

---


**ABSTRACT**  
Large language models have shown impressive capabilities across a variety of NLP tasks, yet their generating text autoregressively is time-consuming. One way to speed them up is speculative decoding, which generates candidate segments (a sequence of tokens) from a fast draft model that is then verified in parallel by the target model. However, the acceptance rate of candidate tokens receives limitations from several factors, such as the model, the dataset, and the decoding setup. This paper proposes sampling multiple candidates from a draft model and then organising them in batches for verification. We design algorithms for efficient multi-candidate verification while maintaining the distribution of the target model. Our approach shows significant improvements in acceptance rates on multiple datasets and models, consistently outperforming standard speculative decoding.

{{</citation>}}


### (19/126) An Experimental Design Framework for Label-Efficient Supervised Finetuning of Large Language Models (Gantavya Bhatt et al., 2024)

{{<citation>}}

Gantavya Bhatt, Yifang Chen, Arnav M. Das, Jifan Zhang, Sang T. Truong, Stephen Mussmann, Yinglun Zhu, Jeffrey Bilmes, Simon S. Du, Kevin Jamieson, Jordan T. Ash, Robert D. Nowak. (2024)  
**An Experimental Design Framework for Label-Efficient Supervised Finetuning of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06692v1)  

---


**ABSTRACT**  
Supervised finetuning (SFT) on instruction datasets has played a crucial role in achieving the remarkable zero-shot generalization capabilities observed in modern large language models (LLMs). However, the annotation efforts required to produce high quality responses for instructions are becoming prohibitively expensive, especially as the number of tasks spanned by instruction datasets continues to increase. Active learning is effective in identifying useful subsets of samples to annotate from an unlabeled pool, but its high computational cost remains a barrier to its widespread applicability in the context of LLMs. To mitigate the annotation cost of SFT and circumvent the computational bottlenecks of active learning, we propose using experimental design. Experimental design techniques select the most informative samples to label, and typically maximize some notion of uncertainty and/or diversity. In our work, we implement a framework that evaluates several existing and novel experimental design techniques and find that these methods consistently yield significant gains in label efficiency with little computational overhead. On generative tasks, our methods achieve the same generalization performance with only $50\%$ of annotation cost required by random sampling.

{{</citation>}}


### (20/126) Don't Rank, Combine! Combining Machine Translation Hypotheses Using Quality Estimation (Giorgos Vernikos et al., 2024)

{{<citation>}}

Giorgos Vernikos, Andrei Popescu-Belis. (2024)  
**Don't Rank, Combine! Combining Machine Translation Hypotheses Using Quality Estimation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BLEU, GLM, Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.06688v1)  

---


**ABSTRACT**  
Neural machine translation systems estimate probabilities of target sentences given source sentences, yet these estimates may not align with human preferences. This work introduces QE-fusion, a method utilizing a quality estimation metric (QE) that better correlates with human judgments to synthesize improved translations. QE-fusion leverages a candidate pool sampled from a model, combining spans from different candidates using QE metrics such as CometKiwi. We compare QE-fusion against beam search and recent reranking techniques, such as Minimum Bayes Risk decoding or QE-reranking. Our method consistently improves translation quality in terms of COMET and BLEURT scores when applied to large language models (LLMs) used for translation (PolyLM, XGLM, Llama2, and Mistral) and to multilingual translation models (NLLB), over five language pairs. Notably, QE-fusion exhibits larger improvements for LLMs due to their ability to generate diverse outputs. We demonstrate that our approach generates novel translations in over half of the cases and consistently outperforms other methods across varying numbers of candidates (5-200). Furthermore, we empirically establish that QE-fusion scales linearly with the number of candidates in the pool. QE-fusion proves effective in enhancing LLM-based translation without the need for costly retraining of LLMs.

{{</citation>}}


### (21/126) Enhancing the Emotional Generation Capability of Large Language Models via Emotional Chain-of-Thought (Zaijing Li et al., 2024)

{{<citation>}}

Zaijing Li, Gongwei Chen, Rui Shao, Dongmei Jiang, Liqiang Nie. (2024)  
**Enhancing the Emotional Generation Capability of Large Language Models via Emotional Chain-of-Thought**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06836v1)  

---


**ABSTRACT**  
The Emotional Generation is a subset of emotional intelligence, which aims to output an emotional response based on emotional conditions as input. Emotion generation has a wide range of applications, including emotion chat, emotional visual caption, and emotional rewriting. However, it faces challenges such as a lack of interpretability and poor evaluability. In this paper, we propose the Emotional Chain-of-Thought (ECoT), a plug-and-play prompting method that enhances the performance of Large Language Models (LLMs) on various emotional generation tasks by aligning with human emotional intelligence guidelines. To assess the reliability of ECoT, we propose an automated model-based evaluation method called EGS. Extensive experimental results demonstrate the effectiveness of ECoT and EGS. Further,we discuss the promise of LLMs in the field of sentiment analysis and present key insights into the LLMs with the ECoT in emotional generation tasks.

{{</citation>}}


### (22/126) How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs (Yi Zeng et al., 2024)

{{<citation>}}

Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi. (2024)  
**How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.06373v1)  

---


**ABSTRACT**  
Most traditional AI safety research has approached AI models as machines and centered on algorithm-focused attacks developed by security experts. As large language models (LLMs) become increasingly common and competent, non-expert users can also impose risks during daily interactions. This paper introduces a new perspective to jailbreak LLMs as human-like communicators, to explore this overlooked intersection between everyday language interaction and AI safety. Specifically, we study how to persuade LLMs to jailbreak them. First, we propose a persuasion taxonomy derived from decades of social science research. Then, we apply the taxonomy to automatically generate interpretable persuasive adversarial prompts (PAP) to jailbreak LLMs. Results show that persuasion significantly increases the jailbreak performance across all risk categories: PAP consistently achieves an attack success rate of over $92\%$ on Llama 2-7b Chat, GPT-3.5, and GPT-4 in $10$ trials, surpassing recent algorithm-focused attacks. On the defense side, we explore various mechanisms against PAP and, found a significant gap in existing defenses, and advocate for more fundamental mitigation for highly interactive LLMs

{{</citation>}}


### (23/126) WisdoM: Improving Multimodal Sentiment Analysis by Fusing Contextual World Knowledge (Wenbin Wang et al., 2024)

{{<citation>}}

Wenbin Wang, Liang Ding, Li Shen, Yong Luo, Han Hu, Dacheng Tao. (2024)  
**WisdoM: Improving Multimodal Sentiment Analysis by Fusing Contextual World Knowledge**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2401.06659v1)  

---


**ABSTRACT**  
Sentiment analysis is rapidly advancing by utilizing various data modalities (e.g., text, image). However, most previous works relied on superficial information, neglecting the incorporation of contextual world knowledge (e.g., background information derived from but beyond the given image and text pairs) and thereby restricting their ability to achieve better multimodal sentiment analysis. In this paper, we proposed a plug-in framework named WisdoM, designed to leverage contextual world knowledge induced from the large vision-language models (LVLMs) for enhanced multimodal sentiment analysis. WisdoM utilizes a LVLM to comprehensively analyze both images and corresponding sentences, simultaneously generating pertinent context. To reduce the noise in the context, we also introduce a training-free Contextual Fusion mechanism. Experimental results across diverse granularities of multimodal sentiment analysis tasks consistently demonstrate that our approach has substantial improvements (brings an average +1.89 F1 score among five advanced methods) over several state-of-the-art methods. Code will be released.

{{</citation>}}


### (24/126) Experimental Contexts Can Facilitate Robust Semantic Property Inference in Language Models, but Inconsistently (Kanishka Misra et al., 2024)

{{<citation>}}

Kanishka Misra, Allyson Ettinger, Kyle Mahowald. (2024)  
**Experimental Contexts Can Facilitate Robust Semantic Property Inference in Language Models, but Inconsistently**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06640v1)  

---


**ABSTRACT**  
Recent zero-shot evaluations have highlighted important limitations in the abilities of language models (LMs) to perform meaning extraction. However, it is now well known that LMs can demonstrate radical improvements in the presence of experimental contexts such as in-context examples and instructions. How well does this translate to previously studied meaning-sensitive tasks? We present a case-study on the extent to which experimental contexts can improve LMs' robustness in performing property inheritance -- predicting semantic properties of novel concepts, a task that they have been previously shown to fail on. Upon carefully controlling the nature of the in-context examples and the instructions, our work reveals that they can indeed lead to non-trivial property inheritance behavior in LMs. However, this ability is inconsistent: with a minimal reformulation of the task, some LMs were found to pick up on shallow, non-semantic heuristics from their inputs, suggesting that the computational principles of semantic property inference are yet to be mastered by LMs.

{{</citation>}}


### (25/126) OOP: Object-Oriented Programming Evaluation Benchmark for Large Language Models (Shuai Wang et al., 2024)

{{<citation>}}

Shuai Wang, Liang Ding, Li Shen, Yong Luo, Bo Du, Dacheng Tao. (2024)  
**OOP: Object-Oriented Programming Evaluation Benchmark for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06628v1)  

---


**ABSTRACT**  
Advancing automated programming necessitates robust and comprehensive code generation benchmarks, yet current evaluation frameworks largely neglect object-oriented programming (OOP) in favor of functional programming (FP), e.g., HumanEval and MBPP. To address this, our study introduces a pioneering OOP-focused benchmark, featuring 431 Python programs that encompass essential OOP concepts and features like classes and encapsulation methods. We propose a novel evaluation metric, pass@o, tailored for OOP, enhancing traditional pass@k measures. Our evaluation of 23 leading large language models (LLMs), including both general and code-specialized models, reveals three key insights: 1) pass@o offers a more relevant and comprehensive assessment for OOP code generation; 2) Despite excelling in FP, code-specialized LLMs like WizardCoder lag in OOP compared to models like ChatGPT; 3) The poor performance of all advanced LLMs on our OOP benchmark highlights a critical need for improvements in this field. Our benchmark and scripts are publicly released at: https://github.com/alphadl/OOP-eval.

{{</citation>}}


### (26/126) TransliCo: A Contrastive Learning Framework to Address the Script Barrier in Multilingual Pretrained Language Models (Yihong Liu et al., 2024)

{{<citation>}}

Yihong Liu, Chunlan Ma, Haotian Ye, Hinrich Schütze. (2024)  
**TransliCo: A Contrastive Learning Framework to Address the Script Barrier in Multilingual Pretrained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning, Language Model, Multilingual, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2401.06620v1)  

---


**ABSTRACT**  
There are 293 scripts representing over 7,000 languages in the written form. Due to various reasons, many closely related languages use different scripts, which poses difficulty for multilingual pretrained language models (mPLMs) in learning crosslingual knowledge through lexical overlap. As a result, mPLMs present a script barrier: representations from different scripts are located in different subspaces, which is a strong indicator of why crosslingual transfer involving languages of different scripts shows sub-optimal performance. To address this problem, we propose a simple framework TransliCo that contains Transliteration Contrastive Modeling (TCM) to fine-tune an mPLM by contrasting sentences in its training data and their transliterations in a unified script (Latn, in our case), which ensures uniformity in the representation space for different scripts. Using Glot500-m, an mPLM pretrained on over 500 languages, as our source model, we find-tune it on a small portion (5\%) of its training data, and refer to the resulting model as Furina. We show that Furina not only better aligns representations from distinct scripts but also outperforms the original Glot500-m on various crosslingual transfer tasks. Additionally, we achieve consistent improvement in a case study on the Indic group where the languages are highly related but use different scripts. We make our code and models publicly available.

{{</citation>}}


### (27/126) Mutual Enhancement of Large Language and Reinforcement Learning Models through Bi-Directional Feedback Mechanisms: A Case Study (Shangding Gu, 2024)

{{<citation>}}

Shangding Gu. (2024)  
**Mutual Enhancement of Large Language and Reinforcement Learning Models through Bi-Directional Feedback Mechanisms: A Case Study**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.06603v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable capabilities for reinforcement learning (RL) models, such as planning and reasoning capabilities. However, the problems of LLMs and RL model collaboration still need to be solved. In this study, we employ a teacher-student learning framework to tackle these problems, specifically by offering feedback for LLMs using RL models and providing high-level information for RL models with LLMs in a cooperative multi-agent setting. Within this framework, the LLM acts as a teacher, while the RL model acts as a student. The two agents cooperatively assist each other through a process of recursive help, such as "I help you help I help." The LLM agent supplies abstract information to the RL agent, enabling efficient exploration and policy improvement. In turn, the RL agent offers feedback to the LLM agent, providing valuable, real-time information that helps generate more useful tokens. This bi-directional feedback loop promotes optimization, exploration, and mutual improvement for both agents, enabling them to accomplish increasingly challenging tasks. Remarkably, we propose a practical algorithm to address the problem and conduct empirical experiments to evaluate the effectiveness of our method.

{{</citation>}}


### (28/126) Prometheus-Vision: Vision-Language Model as a Judge for Fine-Grained Evaluation (Seongyun Lee et al., 2024)

{{<citation>}}

Seongyun Lee, Seungone Kim, Sue Hyun Park, Geewook Kim, Minjoon Seo. (2024)  
**Prometheus-Vision: Vision-Language Model as a Judge for Fine-Grained Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06591v1)  

---


**ABSTRACT**  
Assessing long-form responses generated by Vision-Language Models (VLMs) is challenging. It not only requires checking whether the VLM follows the given instruction but also verifying whether the text output is properly grounded on the given image. Inspired by the recent approach of evaluating LMs with LMs, in this work, we propose to evaluate VLMs with VLMs. For this purpose, we present a new feedback dataset called the Perception Collection, encompassing 15K customized score rubrics that users might care about during assessment. Using the Perception Collection, we train Prometheus-Vision, the first open-source VLM evaluator model that can understand the user-defined score criteria during evaluation. Prometheus-Vision shows the highest Pearson correlation with human evaluators and GPT-4V among open-source models, showing its effectiveness for transparent and accessible evaluation of VLMs. We open-source our code, dataset, and model at https://github.com/kaistAI/prometheus-vision

{{</citation>}}


### (29/126) Mapping Transformer Leveraged Embeddings for Cross-Lingual Document Representation (Tsegaye Misikir Tashu et al., 2024)

{{<citation>}}

Tsegaye Misikir Tashu, Eduard-Raul Kontos, Matthia Sabatelli, Matias Valdenegro-Toro. (2024)  
**Mapping Transformer Leveraged Embeddings for Cross-Lingual Document Representation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: BERT, Embedding, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2401.06583v1)  

---


**ABSTRACT**  
Recommendation systems, for documents, have become tools to find relevant content on the Web. However, these systems have limitations when it comes to recommending documents in languages different from the query language, which means they might overlook resources in non-native languages. This research focuses on representing documents across languages by using Transformer Leveraged Document Representations (TLDRs) that are mapped to a cross-lingual domain. Four multilingual pre-trained transformer models (mBERT, mT5 XLM RoBERTa, ErnieM) were evaluated using three mapping methods across 20 language pairs representing combinations of five selected languages of the European Union. Metrics like Mate Retrieval Rate and Reciprocal Rank were used to measure the effectiveness of mapped TLDRs compared to non-mapped ones. The results highlight the power of cross-lingual representations achieved through pre-trained transformers and mapping approaches suggesting a promising direction for expanding beyond language connections, between two specific languages.

{{</citation>}}


### (30/126) XLS-R Deep Learning Model for Multilingual ASR on Low- Resource Languages: Indonesian, Javanese, and Sundanese (Panji Arisaputra et al., 2024)

{{<citation>}}

Panji Arisaputra, Alif Tri Handoyo, Amalia Zahra. (2024)  
**XLS-R Deep Learning Model for Multilingual ASR on Low- Resource Languages: Indonesian, Javanese, and Sundanese**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: Multilingual, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.06832v1)  

---


**ABSTRACT**  
This research paper focuses on the development and evaluation of Automatic Speech Recognition (ASR) technology using the XLS-R 300m model. The study aims to improve ASR performance in converting spoken language into written text, specifically for Indonesian, Javanese, and Sundanese languages. The paper discusses the testing procedures, datasets used, and methodology employed in training and evaluating the ASR systems. The results show that the XLS-R 300m model achieves competitive Word Error Rate (WER) measurements, with a slight compromise in performance for Javanese and Sundanese languages. The integration of a 5-gram KenLM language model significantly reduces WER and enhances ASR accuracy. The research contributes to the advancement of ASR technology by addressing linguistic diversity and improving performance across various languages. The findings provide insights into optimizing ASR accuracy and applicability for diverse linguistic contexts.

{{</citation>}}


### (31/126) Lost in the Source Language: How Large Language Models Evaluate the Quality of Machine Translation (Xu Huang et al., 2024)

{{<citation>}}

Xu Huang, Zhirui Zhang, Xiang Geng, Yichao Du, Jiajun Chen, Shujian Huang. (2024)  
**Lost in the Source Language: How Large Language Models Evaluate the Quality of Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.06568v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have achieved remarkable results in the machine translation evaluation task, yet there remains a gap in knowledge regarding how they utilize the provided data to conduct evaluations. This study aims to explore how LLMs leverage source and reference information in evaluating translations, with the ultimate goal of better understanding the working mechanism of LLMs. To this end, we design the controlled experiments across various input modes and model types, and employ both coarse-grained and fine-grained prompts to discern the utility of source versus reference information. Surprisingly, we find that reference information significantly enhances the evaluation accuracy, while source information sometimes is counterproductive, indicating a lack of cross-lingual capability when using LLMs to evaluate translations. We further conduct a meta-evaluation for translation error detection of LLMs, observing a similar phenomenon. These findings also suggest a potential research direction for LLMs that fully exploits the cross-lingual capability of LLMs to achieve better performance in machine translation evaluation tasks.

{{</citation>}}


### (32/126) Intention Analysis Prompting Makes Large Language Models A Good Jailbreak Defender (Yuqi Zhang et al., 2024)

{{<citation>}}

Yuqi Zhang, Liang Ding, Lefei Zhang, Dacheng Tao. (2024)  
**Intention Analysis Prompting Makes Large Language Models A Good Jailbreak Defender**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLM, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06561v1)  

---


**ABSTRACT**  
Aligning large language models (LLMs) with human values, particularly in the face of stealthy and complex jailbreaks, presents a formidable challenge. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis Prompting (IAPrompt). The principle behind is to trigger LLMs' inherent self-correct and improve ability through a two-stage process: 1) essential intention analysis, and 2) policy-aligned response. Notably, IAPrompt is an inference-only method, thus could enhance the safety of LLMs without compromising their helpfulness. Extensive experiments on SAP200 and DAN benchmarks across Vicuna, ChatGLM, MPT, DeepSeek, and GPT-3.5 show that IAPrompt could consistently and significantly reduce the harmfulness in response (averagely -46.5% attack success rate) and maintain the general helpfulness. Further analyses present some insights into how our method works. To facilitate reproducibility, We release our code and scripts at: https://github.com/alphadl/SafeLLM_with_IntentionAnalysis

{{</citation>}}


### (33/126) Medical Dialogue Generation via Intuitive-then-Analytical Differential Diagnosis (Kaishuai Xu et al., 2024)

{{<citation>}}

Kaishuai Xu, Wenjun Hou, Yi Cheng, Jian Wang, Wenjie Li. (2024)  
**Medical Dialogue Generation via Intuitive-then-Analytical Differential Diagnosis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2401.06541v1)  

---


**ABSTRACT**  
Medical dialogue systems have attracted growing research attention as they have the potential to provide rapid diagnoses, treatment plans, and health consultations. In medical dialogues, a proper diagnosis is crucial as it establishes the foundation for future consultations. Clinicians typically employ both intuitive and analytic reasoning to formulate a differential diagnosis. This reasoning process hypothesizes and verifies a variety of possible diseases and strives to generate a comprehensive and rigorous diagnosis. However, recent studies on medical dialogue generation have overlooked the significance of modeling a differential diagnosis, which hinders the practical application of these systems. To address the above issue, we propose a medical dialogue generation framework with the Intuitive-then-Analytic Differential Diagnosis (IADDx). Our method starts with a differential diagnosis via retrieval-based intuitive association and subsequently refines it through a graph-enhanced analytic procedure. The resulting differential diagnosis is then used to retrieve medical knowledge and guide response generation. Experimental results on two datasets validate the efficacy of our method. Besides, we demonstrate how our framework assists both clinicians and patients in understanding the diagnostic process, for instance, by producing intermediate results and graph-based diagnosis paths.

{{</citation>}}


### (34/126) INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning (Yutao Zhu et al., 2024)

{{<citation>}}

Yutao Zhu, Peitian Zhang, Chenghao Zhang, Yifei Chen, Binyu Xie, Zhicheng Dou, Zheng Liu, Ji-Rong Wen. (2024)  
**INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06532v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated impressive capabilities in various natural language processing tasks. Despite this, their application to information retrieval (IR) tasks is still challenging due to the infrequent occurrence of many IR-specific concepts in natural language. While prompt-based methods can provide task descriptions to LLMs, they often fall short in facilitating comprehensive understanding and execution of IR tasks, thereby limiting LLMs' applicability. To address this gap, in this work, we explore the potential of instruction tuning to enhance LLMs' proficiency in IR tasks. We introduce a novel instruction tuning dataset, INTERS, encompassing 21 tasks across three fundamental IR categories: query understanding, document understanding, and query-document relationship understanding. The data are derived from 43 distinct datasets with manually written templates. Our empirical results reveal that INTERS significantly boosts the performance of various publicly available LLMs, such as LLaMA, Mistral, and Phi, in search-related tasks. Furthermore, we conduct a comprehensive analysis to ascertain the effects of base model selection, instruction design, volume of instructions, and task variety on performance. We make our dataset and the models fine-tuned on it publicly accessible at https://github.com/DaoD/INTERS.

{{</citation>}}


### (35/126) MetaHate: A Dataset for Unifying Efforts on Hate Speech Detection (Paloma Piot et al., 2024)

{{<citation>}}

Paloma Piot, Patricia Martín-Rodilla, Javier Parapar. (2024)  
**MetaHate: A Dataset for Unifying Efforts on Hate Speech Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: Hate Speech Detection  
[Paper Link](http://arxiv.org/abs/2401.06526v1)  

---


**ABSTRACT**  
Hate speech represents a pervasive and detrimental form of online discourse, often manifested through an array of slurs, from hateful tweets to defamatory posts. As such speech proliferates, it connects people globally and poses significant social, psychological, and occasionally physical threats to targeted individuals and communities. Current computational linguistic approaches for tackling this phenomenon rely on labelled social media datasets for training. For unifying efforts, our study advances in the critical need for a comprehensive meta-collection, advocating for an extensive dataset to help counteract this problem effectively. We scrutinized over 60 datasets, selectively integrating those pertinent into MetaHate. This paper offers a detailed examination of existing collections, highlighting their strengths and limitations. Our findings contribute to a deeper understanding of the existing datasets, paving the way for training more robust and adaptable models. These enhanced models are essential for effectively combating the dynamic and complex nature of hate speech in the digital realm.

{{</citation>}}


### (36/126) AntEval: Quantitatively Evaluating Informativeness and Expressiveness of Agent Social Interactions (Yuanzhi Liang et al., 2024)

{{<citation>}}

Yuanzhi Liang, Linchao Zhu, Yi Yang. (2024)  
**AntEval: Quantitatively Evaluating Informativeness and Expressiveness of Agent Social Interactions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06509v1)  

---


**ABSTRACT**  
While Large Language Models (LLMs) based agents have successfully mimicked human behaviors in various scenarios, the realm of complex, multi-character social interactions within extended contexts remains underexplored. The challenge is compounded by privacy concerns, making it difficult to capture and utilize intricate real-life interactions. More importantly, the absence of quantitative evaluation methods hampers the pursuit of high-quality agent interactions, often leading to interactions that are limited in informativeness and expressiveness, characterized by superficial small talk without clear intentions. In this work, we leverage the rules of Tabletop Role-Playing Games (TRPG) to create an environment conducive to complex, context-rich interactions, emphasizing informativeness and expressiveness. This virtual setting alleviates privacy concerns and motivates agents to engage in meaningful, high-quality interactions as part of their in-game objectives. To assess these interactions, we introduce the Agent interaction Evaluation framework (AntEval), targeting the qualitative evaluation of interaction informativeness and expressiveness. Specifically, we propose two novel evaluation metrics: Information Exchanging Precision (IEP) and Interaction Expressiveness Gap (IEG). These metrics are designed to assess interactions in scenarios focused on information exchange and intention expression, respectively. Our experimental results demonstrate the effectiveness of these metrics in evaluating interaction quality. Notably, we identify significant areas for improvement in LLMs regarding social interactions, as highlighted by our metrics. We believe AntEval will guide further exploration in complex agent interactions, bringing them closer to emulating real human behavior and enhancing their integration and utility in real-world applications.

{{</citation>}}


### (37/126) An investigation of structures responsible for gender bias in BERT and DistilBERT (Thibaud Leteno et al., 2024)

{{<citation>}}

Thibaud Leteno, Antoine Gourru, Charlotte Laclau, Christophe Gravier. (2024)  
**An investigation of structures responsible for gender bias in BERT and DistilBERT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: BERT, Language Model, NLP, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2401.06495v1)  

---


**ABSTRACT**  
In recent years, large Transformer-based Pre-trained Language Models (PLM) have changed the Natural Language Processing (NLP) landscape, by pushing the performance boundaries of the state-of-the-art on a wide variety of tasks. However, this performance gain goes along with an increase in complexity, and as a result, the size of such models (up to billions of parameters) represents a constraint for their deployment on embedded devices or short-inference time tasks. To cope with this situation, compressed models emerged (e.g. DistilBERT), democratizing their usage in a growing number of applications that impact our daily lives. A crucial issue is the fairness of the predictions made by both PLMs and their distilled counterparts. In this paper, we propose an empirical exploration of this problem by formalizing two questions: (1) Can we identify the neural mechanism(s) responsible for gender bias in BERT (and by extension DistilBERT)? (2) Does distillation tend to accentuate or mitigate gender bias (e.g. is DistilBERT more prone to gender bias than its uncompressed version, BERT)? Our findings are the following: (I) one cannot identify a specific layer that produces bias; (II) every attention head uniformly encodes bias; except in the context of underrepresented classes with a high imbalance of the sensitive attribute; (III) this subset of heads is different as we re-fine tune the network; (IV) bias is more homogeneously produced by the heads in the distilled model.

{{</citation>}}


### (38/126) A Survey on the Applications of Frontier AI, Foundation Models, and Large Language Models to Intelligent Transportation Systems (Mohamed R. Shoaib et al., 2024)

{{<citation>}}

Mohamed R. Shoaib, Heba M. Emara, Jun Zhao. (2024)  
**A Survey on the Applications of Frontier AI, Foundation Models, and Large Language Models to Intelligent Transportation Systems**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06831v1)  

---


**ABSTRACT**  
This survey paper explores the transformative influence of frontier AI, foundation models, and Large Language Models (LLMs) in the realm of Intelligent Transportation Systems (ITS), emphasizing their integral role in advancing transportation intelligence, optimizing traffic management, and contributing to the realization of smart cities. Frontier AI refers to the forefront of AI technology, encompassing the latest advancements, innovations, and experimental techniques in the field, especially AI foundation models and LLMs. Foundation models, like GPT-4, are large, general-purpose AI models that provide a base for a wide range of applications. They are characterized by their versatility and scalability. LLMs are obtained from finetuning foundation models with a specific focus on processing and generating natural language. They excel in tasks like language understanding, text generation, translation, and summarization. By leveraging vast textual data, including traffic reports and social media interactions, LLMs extract critical insights, fostering the evolution of ITS. The survey navigates the dynamic synergy between LLMs and ITS, delving into applications in traffic management, integration into autonomous vehicles, and their role in shaping smart cities. It provides insights into ongoing research, innovations, and emerging trends, aiming to inspire collaboration at the intersection of language, intelligence, and mobility for safer, more efficient, and sustainable transportation systems. The paper further surveys interactions between LLMs and various aspects of ITS, exploring roles in traffic management, facilitating autonomous vehicles, and contributing to smart city development, while addressing challenges brought by frontier AI and foundation models. This paper offers valuable inspiration for future research and innovation in the transformative domain of intelligent transportation.

{{</citation>}}


### (39/126) Cross-Attention Watermarking of Large Language Models (Folco Bertini Baldassini et al., 2024)

{{<citation>}}

Folco Bertini Baldassini, Huy H. Nguyen, Ching-Chung Chang, Isao Echizen. (2024)  
**Cross-Attention Watermarking of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06829v1)  

---


**ABSTRACT**  
A new approach to linguistic watermarking of language models is presented in which information is imperceptibly inserted into the output text while preserving its readability and original meaning. A cross-attention mechanism is used to embed watermarks in the text during inference. Two methods using cross-attention are presented that minimize the effect of watermarking on the performance of a pretrained model. Exploration of different training strategies for optimizing the watermarking and of the challenges and implications of applying this approach in real-world scenarios clarified the tradeoff between watermark robustness and text quality. Watermark selection substantially affects the generated output for high entropy sentences. This proactive watermarking approach has potential application in future model development.

{{</citation>}}


### (40/126) Adapting Large Language Models for Document-Level Machine Translation (Minghao Wu et al., 2024)

{{<citation>}}

Minghao Wu, Thuy-Trang Vu, Lizhen Qu, George Foster, Gholamreza Haffari. (2024)  
**Adapting Large Language Models for Document-Level Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, Machine Translation, NLP  
[Paper Link](http://arxiv.org/abs/2401.06468v1)  

---


**ABSTRACT**  
Large language models (LLMs) have made significant strides in various natural language processing (NLP) tasks. Recent research shows that the moderately-sized LLMs often outperform their larger counterparts after task-specific fine-tuning. In this work, we delve into the process of adapting LLMs to specialize in document-level machine translation (DocMT) for a specific language pair. Firstly, we explore how prompt strategies affect downstream translation performance. Then, we conduct extensive experiments with two fine-tuning methods, three LLM backbones, and 18 translation tasks across nine language pairs. Our findings indicate that in some cases, these specialized models even surpass GPT-4 in translation performance, while they still significantly suffer from the off-target translation issue in others, even if they are exclusively fine-tuned on bilingual parallel documents. Furthermore, we provide an in-depth analysis of these LLMs tailored for DocMT, exploring aspects such as translation errors, the scaling law of parallel documents, out-of-domain generalization, and the impact of zero-shot crosslingual transfer. The findings of this research not only shed light on the strengths and limitations of LLM-based DocMT models but also provide a foundation for future research in DocMT.

{{</citation>}}


### (41/126) PersianMind: A Cross-Lingual Persian-English Large Language Model (Pedram Rostami et al., 2024)

{{<citation>}}

Pedram Rostami, Ali Salemi, Mohammad Javad Dousti. (2024)  
**PersianMind: A Cross-Lingual Persian-English Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06466v1)  

---


**ABSTRACT**  
Large language models demonstrate remarkable proficiency in various linguistic tasks and have extensive knowledge across various domains. Although they perform best in English, their ability in other languages is notable too. In contrast, open-source models, such as LLaMa, are primarily trained on English datasets, resulting in poor performance in non-English languages. In this paper, we introduce PersianMind, an open-source bilingual large language model which demonstrates comparable performance to closed-source GPT-3.5-turbo in the Persian language. By expanding LLaMa2's vocabulary with 10,000 Persian tokens and training it on a dataset comprising nearly 2 billion Persian tokens, we show that our approach preserves the model's English knowledge and employs transfer learning to excel at transferring task knowledge from one language to another.

{{</citation>}}


### (42/126) BOK-VQA: Bilingual Outside Knowledge-based Visual Question Answering via Graph Representation Pretraining (Minjun Kim et al., 2024)

{{<citation>}}

Minjun Kim, Seungwoo Song, Youhan Lee, Haneol Jang, Kyungtae Lim. (2024)  
**BOK-VQA: Bilingual Outside Knowledge-based Visual Question Answering via Graph Representation Pretraining**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.06443v1)  

---


**ABSTRACT**  
The current research direction in generative models, such as the recently developed GPT4, aims to find relevant knowledge information for multimodal and multilingual inputs to provide answers. Under these research circumstances, the demand for multilingual evaluation of visual question answering (VQA) tasks, a representative task of multimodal systems, has increased. Accordingly, we propose a bilingual outside-knowledge VQA (BOK-VQA) dataset in this study that can be extended to multilingualism. The proposed data include 17K images, 17K question-answer pairs for both Korean and English and 280K instances of knowledge information related to question-answer content. We also present a framework that can effectively inject knowledge information into a VQA system by pretraining the knowledge information of BOK-VQA data in the form of graph embeddings. Finally, through in-depth analysis, we demonstrated the actual effect of the knowledge information contained in the constructed training data on VQA.

{{</citation>}}


### (43/126) From Automation to Augmentation: Large Language Models Elevating Essay Scoring Landscape (Changrong Xiao et al., 2024)

{{<citation>}}

Changrong Xiao, Wenxing Ma, Sean Xin Xu, Kunpeng Zhang, Yufang Wang, Qi Fu. (2024)  
**From Automation to Augmentation: Large Language Models Elevating Essay Scoring Landscape**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Augmentation, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06431v1)  

---


**ABSTRACT**  
Receiving immediate and personalized feedback is crucial for second-language learners, and Automated Essay Scoring (AES) systems are a vital resource when human instructors are unavailable. This study investigates the effectiveness of Large Language Models (LLMs), specifically GPT-4 and fine-tuned GPT-3.5, as tools for AES. Our comprehensive set of experiments, conducted on both public and private datasets, highlights the remarkable advantages of LLM-based AES systems. They include superior accuracy, consistency, generalizability, and interpretability, with fine-tuned GPT-3.5 surpassing traditional grading models. Additionally, we undertake LLM-assisted human evaluation experiments involving both novice and expert graders. One pivotal discovery is that LLMs not only automate the grading process but also enhance the performance of human graders. Novice graders when provided with feedback generated by LLMs, achieve a level of accuracy on par with experts, while experts become more efficient and maintain greater consistency in their assessments. These results underscore the potential of LLMs in educational technology, paving the way for effective collaboration between humans and AI, ultimately leading to transformative learning experiences through AI-generated feedback.

{{</citation>}}


### (44/126) Mission: Impossible Language Models (Julie Kallini et al., 2024)

{{<citation>}}

Julie Kallini, Isabel Papadimitriou, Richard Futrell, Kyle Mahowald, Christopher Potts. (2024)  
**Mission: Impossible Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06416v1)  

---


**ABSTRACT**  
Chomsky and others have very directly claimed that large language models (LLMs) are equally capable of learning languages that are possible and impossible for humans to learn. However, there is very little published experimental evidence to support such a claim. Here, we develop a set of synthetic impossible languages of differing complexity, each designed by systematically altering English data with unnatural word orders and grammar rules. These languages lie on an impossibility continuum: at one end are languages that are inherently impossible, such as random and irreversible shuffles of English words, and on the other, languages that may not be intuitively impossible but are often considered so in linguistics, particularly those with rules based on counting word positions. We report on a wide range of evaluations to assess the capacity of GPT-2 small models to learn these uncontroversially impossible languages, and crucially, we perform these assessments at various stages throughout training to compare the learning process for each language. Our core finding is that GPT-2 struggles to learn impossible languages when compared to English as a control, challenging the core claim. More importantly, we hope our approach opens up a productive line of inquiry in which different LLM architectures are tested on a variety of impossible languages in an effort to learn more about how LLMs can be used as tools for these cognitive and typological investigations.

{{</citation>}}


### (45/126) Generalizing Visual Question Answering from Synthetic to Human-Written Questions via a Chain of QA with a Large Language Model (Taehee Kim et al., 2024)

{{<citation>}}

Taehee Kim, Yeongjae Cho, Heejun Shin, Yohan Jo, Dongmyung Shin. (2024)  
**Generalizing Visual Question Answering from Synthetic to Human-Written Questions via a Chain of QA with a Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.06400v2)  

---


**ABSTRACT**  
Visual question answering (VQA) is a task where an image is given, and a series of questions are asked about the image. To build an efficient VQA algorithm, a large amount of QA data is required which is very expensive. Generating synthetic QA pairs based on templates is a practical way to obtain data. However, VQA models trained on those data do not perform well on complex, human-written questions. To address this issue, we propose a new method called {\it chain of QA for human-written questions} (CoQAH). CoQAH utilizes a sequence of QA interactions between a large language model and a VQA model trained on synthetic data to reason and derive logical answers for human-written questions. We tested the effectiveness of CoQAH on two types of human-written VQA datasets for 3D-rendered and chest X-ray images and found that it achieved state-of-the-art accuracy in both types of data. Notably, CoQAH outperformed general vision-language models, VQA models, and medical foundation models with no finetuning.

{{</citation>}}


### (46/126) An approach for mistranslation removal from popular dataset for Indic MT Task (Sudhansu Bala Das et al., 2024)

{{<citation>}}

Sudhansu Bala Das, Leo Raphael Rodrigues, Tapas Kumar Mishra, Bidyut Kr. Patra. (2024)  
**An approach for mistranslation removal from popular dataset for Indic MT Task**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.06398v1)  

---


**ABSTRACT**  
The conversion of content from one language to another utilizing a computer system is known as Machine Translation (MT). Various techniques have come up to ensure effective translations that retain the contextual and lexical interpretation of the source language. End-to-end Neural Machine Translation (NMT) is a popular technique and it is now widely used in real-world MT systems. Massive amounts of parallel datasets (sentences in one language alongside translations in another) are required for MT systems. These datasets are crucial for an MT system to learn linguistic structures and patterns of both languages during the training phase. One such dataset is Samanantar, the largest publicly accessible parallel dataset for Indian languages (ILs). Since the corpus has been gathered from various sources, it contains many incorrect translations. Hence, the MT systems built using this dataset cannot perform to their usual potential. In this paper, we propose an algorithm to remove mistranslations from the training corpus and evaluate its performance and efficiency. Two Indic languages (ILs), namely, Hindi (HIN) and Odia (ODI) are chosen for the experiment. A baseline NMT system is built for these two ILs, and the effect of different dataset sizes is also investigated. The quality of the translations in the experiment is evaluated using standard metrics such as BLEU, METEOR, and RIBES. From the results, it is observed that removing the incorrect translation from the dataset makes the translation quality better. It is also noticed that, despite the fact that the ILs-English and English-ILs systems are trained using the same corpus, ILs-English works more effectively across all the evaluation metrics.

{{</citation>}}


### (47/126) Adaptive Data Augmentation for Aspect Sentiment Quad Prediction (Wenyuan Zhang et al., 2024)

{{<citation>}}

Wenyuan Zhang, Xinghua Zhang, Shiyao Cui, Kun Huang, Xuebin Wang, Tingwen Liu. (2024)  
**Adaptive Data Augmentation for Aspect Sentiment Quad Prediction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2401.06394v1)  

---


**ABSTRACT**  
Aspect sentiment quad prediction (ASQP) aims to predict the quad sentiment elements for a given sentence, which is a critical task in the field of aspect-based sentiment analysis. However, the data imbalance issue has not received sufficient attention in ASQP task. In this paper, we divide the issue into two-folds, quad-pattern imbalance and aspect-category imbalance, and propose an Adaptive Data Augmentation (ADA) framework to tackle the imbalance issue. Specifically, a data augmentation process with a condition function adaptively enhances the tail quad patterns and aspect categories, alleviating the data imbalance in ASQP. Following previous studies, we also further explore the generative framework for extracting complete quads by introducing the category prior knowledge and syntax-guided decoding target. Experimental results demonstrate that data augmentation for imbalance in ASQP task can improve the performance, and the proposed ADA method is superior to naive data oversampling.

{{</citation>}}


## cs.LG (16)



### (48/126) Open RAN LSTM Traffic Prediction and Slice Management using Deep Reinforcement Learning (Fatemeh Lotfi et al., 2024)

{{<citation>}}

Fatemeh Lotfi, Fatemeh Afghah. (2024)  
**Open RAN LSTM Traffic Prediction and Slice Management using Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NI, cs-SY, cs.LG, eess-SY, stat-ML  
Keywords: LSTM, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.06922v1)  

---


**ABSTRACT**  
With emerging applications such as autonomous driving, smart cities, and smart factories, network slicing has become an essential component of 5G and beyond networks as a means of catering to a service-aware network. However, managing different network slices while maintaining quality of services (QoS) is a challenge in a dynamic environment. To address this issue, this paper leverages the heterogeneous experiences of distributed units (DUs) in ORAN systems and introduces a novel approach to ORAN slicing xApp using distributed deep reinforcement learning (DDRL). Additionally, to enhance the decision-making performance of the RL agent, a prediction rApp based on long short-term memory (LSTM) is incorporated to provide additional information from the dynamic environment to the xApp. Simulation results demonstrate significant improvements in network performance, particularly in reducing QoS violations. This emphasizes the importance of using the prediction rApp and distributed actors' information jointly as part of a dynamic xApp.

{{</citation>}}


### (49/126) Analyses and Concerns in Precision Medicine: A Statistical Perspective (Xiaofei Chen, 2024)

{{<citation>}}

Xiaofei Chen. (2024)  
**Analyses and Concerns in Precision Medicine: A Statistical Perspective**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-AP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06899v1)  

---


**ABSTRACT**  
This article explores the critical role of statistical analysis in precision medicine. It discusses how personalized healthcare is enhanced by statistical methods that interpret complex, multidimensional datasets, focusing on predictive modeling, machine learning algorithms, and data visualization techniques. The paper addresses challenges in data integration and interpretation, particularly with diverse data sources like electronic health records (EHRs) and genomic data. It also delves into ethical considerations such as patient privacy and data security. In addition, the paper highlights the evolution of statistical analysis in medicine, core statistical methodologies in precision medicine, and future directions in the field, emphasizing the integration of artificial intelligence (AI) and machine learning (ML).

{{</citation>}}


### (50/126) Always-Sparse Training by Growing Connections with Guided Stochastic Exploration (Mike Heddes et al., 2024)

{{<citation>}}

Mike Heddes, Narayan Srinivasa, Tony Givargis, Alexandru Nicolau. (2024)  
**Always-Sparse Training by Growing Connections with Guided Stochastic Exploration**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.06898v1)  

---


**ABSTRACT**  
The excessive computational requirements of modern artificial neural networks (ANNs) are posing limitations on the machines that can run them. Sparsification of ANNs is often motivated by time, memory and energy savings only during model inference, yielding no benefits during training. A growing body of work is now focusing on providing the benefits of model sparsification also during training. While these methods greatly improve the training efficiency, the training algorithms yielding the most accurate models still materialize the dense weights, or compute dense gradients during training. We propose an efficient, always-sparse training algorithm with excellent scaling to larger and sparser models, supported by its linear time complexity with respect to the model width during training and inference. Moreover, our guided stochastic exploration algorithm improves over the accuracy of previous sparse training methods. We evaluate our method on CIFAR-10/100 and ImageNet using ResNet, VGG, and ViT models, and compare it against a range of sparsification methods.

{{</citation>}}


### (51/126) Deep Manifold Graph Auto-Encoder for Attributed Graph Embedding (Bozhen Hu et al., 2024)

{{<citation>}}

Bozhen Hu, Zelin Zang, Jun Xia, Lirong Wu, Cheng Tan, Stan Z. Li. (2024)  
**Deep Manifold Graph Auto-Encoder for Attributed Graph Embedding**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.06727v1)  

---


**ABSTRACT**  
Representing graph data in a low-dimensional space for subsequent tasks is the purpose of attributed graph embedding. Most existing neural network approaches learn latent representations by minimizing reconstruction errors. Rare work considers the data distribution and the topological structure of latent codes simultaneously, which often results in inferior embeddings in real-world graph data. This paper proposes a novel Deep Manifold (Variational) Graph Auto-Encoder (DMVGAE/DMGAE) method for attributed graph data to improve the stability and quality of learned representations to tackle the crowding problem. The node-to-node geodesic similarity is preserved between the original and latent space under a pre-defined distribution. The proposed method surpasses state-of-the-art baseline algorithms by a significant margin on different downstream tasks across popular datasets, which validates our solutions. We promise to release the code after acceptance.

{{</citation>}}


### (52/126) SeizNet: An AI-enabled Implantable Sensor Network System for Seizure Prediction (Ali Saeizadeh et al., 2024)

{{<citation>}}

Ali Saeizadeh, Douglas Schonholtz, Daniel Uvaydov, Raffaele Guida, Emrecan Demirors, Pedram Johari, Jorge M. Jimenez, Joseph S. Neimat, Tommaso Melodia. (2024)  
**SeizNet: An AI-enabled Implantable Sensor Network System for Seizure Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06644v1)  

---


**ABSTRACT**  
In this paper, we introduce SeizNet, a closed-loop system for predicting epileptic seizures through the use of Deep Learning (DL) method and implantable sensor networks. While pharmacological treatment is effective for some epilepsy patients (with ~65M people affected worldwide), one out of three suffer from drug-resistant epilepsy. To alleviate the impact of seizure, predictive systems have been developed that can notify such patients of an impending seizure, allowing them to take precautionary measures. SeizNet leverages DL techniques and combines data from multiple recordings, specifically intracranial electroencephalogram (iEEG) and electrocardiogram (ECG) sensors, that can significantly improve the specificity of seizure prediction while preserving very high levels of sensitivity. SeizNet DL algorithms are designed for efficient real-time execution at the edge, minimizing data privacy concerns, data transmission overhead, and power inefficiencies associated with cloud-based solutions. Our results indicate that SeizNet outperforms traditional single-modality and non-personalized prediction systems in all metrics, achieving up to 99% accuracy in predicting seizure, offering a promising new avenue in refractory epilepsy treatment.

{{</citation>}}


### (53/126) CCFC: Bridging Federated Clustering and Contrastive Learning (Jie Yan et al., 2024)

{{<citation>}}

Jie Yan, Jing Liu, Zhong-Yuan Zhang. (2024)  
**CCFC: Bridging Federated Clustering and Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.06634v1)  

---


**ABSTRACT**  
Federated clustering, an essential extension of centralized clustering for federated scenarios, enables multiple data-holding clients to collaboratively group data while keeping their data locally. In centralized scenarios, clustering driven by representation learning has made significant advancements in handling high-dimensional complex data. However, the combination of federated clustering and representation learning remains underexplored. To bridge this, we first tailor a cluster-contrastive model for learning clustering-friendly representations. Then, we harness this model as the foundation for proposing a new federated clustering method, named cluster-contrastive federated clustering (CCFC). Benefiting from representation learning, the clustering performance of CCFC even double those of the best baseline methods in some cases. Compared to the most related baseline, the benefit results in substantial NMI score improvements of up to 0.4155 on the most conspicuous case. Moreover, CCFC also shows superior performance in handling device failures from a practical viewpoint.

{{</citation>}}


### (54/126) Every Node is Different: Dynamically Fusing Self-Supervised Tasks for Attributed Graph Clustering (Pengfei Zhu et al., 2024)

{{<citation>}}

Pengfei Zhu, Qian Wang, Yu Wang, Jialu Li, Qinghua Hu. (2024)  
**Every Node is Different: Dynamically Fusing Self-Supervised Tasks for Attributed Graph Clustering**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.06595v1)  

---


**ABSTRACT**  
Attributed graph clustering is an unsupervised task that partitions nodes into different groups. Self-supervised learning (SSL) shows great potential in handling this task, and some recent studies simultaneously learn multiple SSL tasks to further boost performance. Currently, different SSL tasks are assigned the same set of weights for all graph nodes. However, we observe that some graph nodes whose neighbors are in different groups require significantly different emphases on SSL tasks. In this paper, we propose to dynamically learn the weights of SSL tasks for different nodes and fuse the embeddings learned from different SSL tasks to boost performance. We design an innovative graph clustering approach, namely Dynamically Fusing Self-Supervised Learning (DyFSS). Specifically, DyFSS fuses features extracted from diverse SSL tasks using distinct weights derived from a gating network. To effectively learn the gating network, we design a dual-level self-supervised strategy that incorporates pseudo labels and the graph structure. Extensive experiments on five datasets show that DyFSS outperforms the state-of-the-art multi-task SSL methods by up to 8.66% on the accuracy metric. The code of DyFSS is available at: https://github.com/q086/DyFSS.

{{</citation>}}


### (55/126) A General Benchmark Framework is Dynamic Graph Neural Network Need (Yusen Zhang, 2024)

{{<citation>}}

Yusen Zhang. (2024)  
**A General Benchmark Framework is Dynamic Graph Neural Network Need**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2401.06559v1)  

---


**ABSTRACT**  
Dynamic graph learning is crucial for modeling real-world systems with evolving relationships and temporal dynamics. However, the lack of a unified benchmark framework in current research has led to inaccurate evaluations of dynamic graph models. This paper highlights the significance of dynamic graph learning and its applications in various domains. It emphasizes the need for a standardized benchmark framework that captures temporal dynamics, evolving graph structures, and downstream task requirements. Establishing a unified benchmark will help researchers understand the strengths and limitations of existing models, foster innovation, and advance dynamic graph learning. In conclusion, this paper identifies the lack of a standardized benchmark framework as a current limitation in dynamic graph learning research . Such a framework will facilitate accurate model evaluation, drive advancements in dynamic graph learning techniques, and enable the development of more effective models for real-world applications.

{{</citation>}}


### (56/126) Treatment-Aware Hyperbolic Representation Learning for Causal Effect Estimation with Social Networks (Ziqiang Cui et al., 2024)

{{<citation>}}

Ziqiang Cui, Xing Tang, Yang Qiao, Bowei He, Liang Chen, Xiuqiang He, Chen Ma. (2024)  
**Treatment-Aware Hyperbolic Representation Learning for Causal Effect Estimation with Social Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG, stat-ME  
Keywords: Representation Learning, Social Network  
[Paper Link](http://arxiv.org/abs/2401.06557v1)  

---


**ABSTRACT**  
Estimating the individual treatment effect (ITE) from observational data is a crucial research topic that holds significant value across multiple domains. How to identify hidden confounders poses a key challenge in ITE estimation. Recent studies have incorporated the structural information of social networks to tackle this challenge, achieving notable advancements. However, these methods utilize graph neural networks to learn the representation of hidden confounders in Euclidean space, disregarding two critical issues: (1) the social networks often exhibit a scalefree structure, while Euclidean embeddings suffer from high distortion when used to embed such graphs, and (2) each ego-centric network within a social network manifests a treatment-related characteristic, implying significant patterns of hidden confounders. To address these issues, we propose a novel method called Treatment-Aware Hyperbolic Representation Learning (TAHyper). Firstly, TAHyper employs the hyperbolic space to encode the social networks, thereby effectively reducing the distortion of confounder representation caused by Euclidean embeddings. Secondly, we design a treatment-aware relationship identification module that enhances the representation of hidden confounders by identifying whether an individual and her neighbors receive the same treatment. Extensive experiments on two benchmark datasets are conducted to demonstrate the superiority of our method.

{{</citation>}}


### (57/126) Domain Adaptation for Time series Transformers using One-step fine-tuning (Subina Khanal et al., 2024)

{{<citation>}}

Subina Khanal, Seshu Tirupathi, Giulio Zizzo, Ambrish Rawat, Torben Bach Pedersen. (2024)  
**Domain Adaptation for Time series Transformers using One-step fine-tuning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.06524v1)  

---


**ABSTRACT**  
The recent breakthrough of Transformers in deep learning has drawn significant attention of the time series community due to their ability to capture long-range dependencies. However, like other deep learning models, Transformers face limitations in time series prediction, including insufficient temporal understanding, generalization challenges, and data shift issues for the domains with limited data. Additionally, addressing the issue of catastrophic forgetting, where models forget previously learned information when exposed to new data, is another critical aspect that requires attention in enhancing the robustness of Transformers for time series tasks. To address these limitations, in this paper, we pre-train the time series Transformer model on a source domain with sufficient data and fine-tune it on the target domain with limited data. We introduce the \emph{One-step fine-tuning} approach, adding some percentage of source domain data to the target domains, providing the model with diverse time series instances. We then fine-tune the pre-trained model using a gradual unfreezing technique. This helps enhance the model's performance in time series prediction for domains with limited data. Extensive experimental results on two real-world datasets show that our approach improves over the state-of-the-art baselines by 4.35% and 11.54% for indoor temperature and wind power prediction, respectively.

{{</citation>}}


### (58/126) Personalized Reinforcement Learning with a Budget of Policies (Dmitry Ivanov et al., 2024)

{{<citation>}}

Dmitry Ivanov, Omer Ben-Porat. (2024)  
**Personalized Reinforcement Learning with a Budget of Policies**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.06514v1)  

---


**ABSTRACT**  
Personalization in machine learning (ML) tailors models' decisions to the individual characteristics of users. While this approach has seen success in areas like recommender systems, its expansion into high-stakes fields such as healthcare and autonomous driving is hindered by the extensive regulatory approval processes involved. To address this challenge, we propose a novel framework termed represented Markov Decision Processes (r-MDPs) that is designed to balance the need for personalization with the regulatory constraints. In an r-MDP, we cater to a diverse user population, each with unique preferences, through interaction with a small set of representative policies. Our objective is twofold: efficiently match each user to an appropriate representative policy and simultaneously optimize these policies to maximize overall social welfare. We develop two deep reinforcement learning algorithms that efficiently solve r-MDPs. These algorithms draw inspiration from the principles of classic K-means clustering and are underpinned by robust theoretical foundations. Our empirical investigations, conducted across a variety of simulated environments, showcase the algorithms' ability to facilitate meaningful personalization even under constrained policy budgets. Furthermore, they demonstrate scalability, efficiently adapting to larger policy budgets.

{{</citation>}}


### (59/126) Improving Graph Convolutional Networks with Transformer Layer in social-based items recommendation (Thi Linh Hoang et al., 2024)

{{<citation>}}

Thi Linh Hoang, Tuan Dung Pham, Viet Cuong Ta. (2024)  
**Improving Graph Convolutional Networks with Transformer Layer in social-based items recommendation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-IR, cs-LG, cs.LG  
Keywords: Graph Convolutional Network, Transformer  
[Paper Link](http://arxiv.org/abs/2401.06436v1)  

---


**ABSTRACT**  
In this work, we have proposed an approach for improving the GCN for predicting ratings in social networks. Our model is expanded from the standard model with several layers of transformer architecture. The main focus of the paper is on the encoder architecture for node embedding in the network. Using the embedding layer from the graph-based convolution layer, the attention mechanism could rearrange the feature space to get a more efficient embedding for the downstream task. The experiments showed that our proposed architecture achieves better performance than GCN on the traditional link prediction task.

{{</citation>}}


### (60/126) Uncertainty quantification for probabilistic machine learning in earth observation using conformal prediction (Geethen Singh et al., 2024)

{{<citation>}}

Geethen Singh, Glenn Moncrieff, Zander Venter, Kerry Cawse-Nicholson, Jasper Slingsby, Tamara B Robinson. (2024)  
**Uncertainty quantification for probabilistic machine learning in earth observation using conformal prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2401.06421v1)  

---


**ABSTRACT**  
Unreliable predictions can occur when using artificial intelligence (AI) systems with negative consequences for downstream applications, particularly when employed for decision-making. Conformal prediction provides a model-agnostic framework for uncertainty quantification that can be applied to any dataset, irrespective of its distribution, post hoc. In contrast to other pixel-level uncertainty quantification methods, conformal prediction operates without requiring access to the underlying model and training dataset, concurrently offering statistically valid and informative prediction regions, all while maintaining computational efficiency. In response to the increased need to report uncertainty alongside point predictions, we bring attention to the promise of conformal prediction within the domain of Earth Observation (EO) applications. To accomplish this, we assess the current state of uncertainty quantification in the EO domain and found that only 20% of the reviewed Google Earth Engine (GEE) datasets incorporated a degree of uncertainty information, with unreliable methods prevalent. Next, we introduce modules that seamlessly integrate into existing GEE predictive modelling workflows and demonstrate the application of these tools for datasets spanning local to global scales, including the Dynamic World and Global Ecosystem Dynamics Investigation (GEDI) datasets. These case studies encompass regression and classification tasks, featuring both traditional and deep learning-based workflows. Subsequently, we discuss the opportunities arising from the use of conformal prediction in EO. We anticipate that the increased availability of easy-to-use implementations of conformal predictors, such as those provided here, will drive wider adoption of rigorous uncertainty quantification in EO, thereby enhancing the reliability of uses such as operational monitoring and decision making.

{{</citation>}}


### (61/126) An Empirical Investigation into the Effect of Parameter Choices in Knowledge Distillation (Md Arafat Sultan et al., 2024)

{{<citation>}}

Md Arafat Sultan, Aashka Trivedi, Parul Awasthy, Avirup Sil. (2024)  
**An Empirical Investigation into the Effect of Parameter Choices in Knowledge Distillation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Knowledge Distillation, NLP  
[Paper Link](http://arxiv.org/abs/2401.06356v1)  

---


**ABSTRACT**  
We present a large-scale empirical study of how choices of configuration parameters affect performance in knowledge distillation (KD). An example of such a KD parameter is the measure of distance between the predictions of the teacher and the student, common choices for which include the mean squared error (MSE) and the KL-divergence. Although scattered efforts have been made to understand the differences between such options, the KD literature still lacks a systematic study on their general effect on student performance. We take an empirical approach to this question in this paper, seeking to find out the extent to which such choices influence student performance across 13 datasets from 4 NLP tasks and 3 student sizes. We quantify the cost of making sub-optimal choices and identify a single configuration that performs well across the board.

{{</citation>}}


### (62/126) Direct Distillation between Different Domains (Jialiang Tang et al., 2024)

{{<citation>}}

Jialiang Tang, Shuo Chen, Gang Niu, Hongyuan Zhu, Joey Tianyi Zhou, Chen Gong, Masashi Sugiyama. (2024)  
**Direct Distillation between Different Domains**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2401.06826v1)  

---


**ABSTRACT**  
Knowledge Distillation (KD) aims to learn a compact student network using knowledge from a large pre-trained teacher network, where both networks are trained on data from the same distribution. However, in practical applications, the student network may be required to perform in a new scenario (i.e., the target domain), which usually exhibits significant differences from the known scenario of the teacher network (i.e., the source domain). The traditional domain adaptation techniques can be integrated with KD in a two-stage process to bridge the domain gap, but the ultimate reliability of two-stage approaches tends to be limited due to the high computational consumption and the additional errors accumulated from both stages. To solve this problem, we propose a new one-stage method dubbed ``Direct Distillation between Different Domains" (4Ds). We first design a learnable adapter based on the Fourier transform to separate the domain-invariant knowledge from the domain-specific knowledge. Then, we build a fusion-activation mechanism to transfer the valuable domain-invariant knowledge to the student network, while simultaneously encouraging the adapter within the teacher network to learn the domain-specific knowledge of the target data. As a result, the teacher network can effectively transfer categorical knowledge that aligns with the target domain of the student network. Intensive experiments on various benchmark datasets demonstrate that our proposed 4Ds method successfully produces reliable student networks and outperforms state-of-the-art approaches.

{{</citation>}}


### (63/126) Striking a Balance in Fairness for Dynamic Systems Through Reinforcement Learning (Yaowei Hu et al., 2024)

{{<citation>}}

Yaowei Hu, Jacob Lear, Lu Zhang. (2024)  
**Striking a Balance in Fairness for Dynamic Systems Through Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.06318v1)  

---


**ABSTRACT**  
While significant advancements have been made in the field of fair machine learning, the majority of studies focus on scenarios where the decision model operates on a static population. In this paper, we study fairness in dynamic systems where sequential decisions are made. Each decision may shift the underlying distribution of features or user behavior. We model the dynamic system through a Markov Decision Process (MDP). By acknowledging that traditional fairness notions and long-term fairness are distinct requirements that may not necessarily align with one another, we propose an algorithmic framework to integrate various fairness considerations with reinforcement learning using both pre-processing and in-processing approaches. Three case studies show that our method can strike a balance between traditional fairness notions, long-term fairness, and utility.

{{</citation>}}


## cs.IR (9)



### (64/126) InRanker: Distilled Rankers for Zero-shot Information Retrieval (Thiago Laitz et al., 2024)

{{<citation>}}

Thiago Laitz, Konstantinos Papakostas, Roberto Lotufo, Rodrigo Nogueira. (2024)  
**InRanker: Distilled Rankers for Zero-shot Information Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval, T5  
[Paper Link](http://arxiv.org/abs/2401.06910v1)  

---


**ABSTRACT**  
Despite multi-billion parameter neural rankers being common components of state-of-the-art information retrieval pipelines, they are rarely used in production due to the enormous amount of compute required for inference. In this work, we propose a new method for distilling large rankers into their smaller versions focusing on out-of-domain effectiveness. We introduce InRanker, a version of monoT5 distilled from monoT5-3B with increased effectiveness on out-of-domain scenarios. Our key insight is to use language models and rerankers to generate as much as possible synthetic "in-domain" training data, i.e., data that closely resembles the data that will be seen at retrieval time. The pipeline consists of two distillation phases that do not require additional user queries or manual annotations: (1) training on existing supervised soft teacher labels, and (2) training on teacher soft labels for synthetic queries generated using a large language model. Consequently, models like monoT5-60M and monoT5-220M improved their effectiveness by using the teacher's knowledge, despite being 50x and 13x smaller, respectively. Models and code are available at https://github.com/unicamp-dl/InRanker.

{{</citation>}}


### (65/126) Improved Learned Sparse Retrieval with Corpus-Specific Vocabularies (Puxuan Yu et al., 2024)

{{<citation>}}

Puxuan Yu, Antonio Mallia, Matthias Petri. (2024)  
**Improved Learned Sparse Retrieval with Corpus-Specific Vocabularies**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.06703v1)  

---


**ABSTRACT**  
We explore leveraging corpus-specific vocabularies that improve both efficiency and effectiveness of learned sparse retrieval systems. We find that pre-training the underlying BERT model on the target corpus, specifically targeting different vocabulary sizes incorporated into the document expansion process, improves retrieval quality by up to 12% while in some scenarios decreasing latency by up to 50%. Our experiments show that adopting corpus-specific vocabulary and increasing vocabulary size decreases average postings list length which in turn reduces latency. Ablation studies show interesting interactions between custom vocabularies, document expansion techniques, and sparsification objectives of sparse models. Both effectiveness and efficiency improvements transfer to different retrieval approaches such as uniCOIL and SPLADE and offer a simple yet effective approach to providing new efficiency-effectiveness trade-offs for learned sparse retrieval systems.

{{</citation>}}


### (66/126) DQNC2S: DQN-based Cross-stream Crisis event Summarizer (Daniele Rege Cambrin et al., 2024)

{{<citation>}}

Daniele Rege Cambrin, Luca Cagliero, Paolo Garza. (2024)  
**DQNC2S: DQN-based Cross-stream Crisis event Summarizer**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.06683v1)  

---


**ABSTRACT**  
Summarizing multiple disaster-relevant data streams simultaneously is particularly challenging as existing Retrieve&Re-ranking strategies suffer from the inherent redundancy of multi-stream data and limited scalability in a multi-query setting. This work proposes an online approach to crisis timeline generation based on weak annotation with Deep Q-Networks. It selects on-the-fly the relevant pieces of text without requiring neither human annotations nor content re-ranking. This makes the inference time independent of the number of input queries. The proposed approach also incorporates a redundancy filter into the reward function to effectively handle cross-stream content overlaps. The achieved ROUGE and BERTScore results are superior to those of best-performing models on the CrisisFACTS 2022 benchmark.

{{</citation>}}


### (67/126) LLMRS: Unlocking Potentials of LLM-Based Recommender Systems for Software Purchase (Angela John et al., 2024)

{{<citation>}}

Angela John, Theophilus Aidoo, Hamayoon Behmanush, Irem B. Gunduz, Hewan Shrestha, Maxx Richard Rahman, Wolfgang Maaß. (2024)  
**LLMRS: Unlocking Potentials of LLM-Based Recommender Systems for Software Purchase**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Amazon, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06676v1)  

---


**ABSTRACT**  
Recommendation systems are ubiquitous, from Spotify playlist suggestions to Amazon product suggestions. Nevertheless, depending on the methodology or the dataset, these systems typically fail to capture user preferences and generate general recommendations. Recent advancements in Large Language Models (LLM) offer promising results for analyzing user queries. However, employing these models to capture user preferences and efficiency remains an open question. In this paper, we propose LLMRS, an LLM-based zero-shot recommender system where we employ pre-trained LLM to encode user reviews into a review score and generate user-tailored recommendations. We experimented with LLMRS on a real-world dataset, the Amazon product reviews, for software purchase use cases. The results show that LLMRS outperforms the ranking-based baseline model while successfully capturing meaningful information from product reviews, thereby providing more reliable recommendations.

{{</citation>}}


### (68/126) The SemIoE Ontology: A Semantic Model Solution for an IoE-based Industry (Marco Arazzi et al., 2024)

{{<citation>}}

Marco Arazzi, Antonino Nocera, Emanuele Storti. (2024)  
**The SemIoE Ontology: A Semantic Model Solution for an IoE-based Industry**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.06667v1)  

---


**ABSTRACT**  
Recently, the Industry 5.0 is gaining attention as a novel paradigm, defining the next concrete steps toward more and more intelligent, green-aware and user-centric digital systems. In an era in which smart devices typically adopted in the industry domain are more and more sophisticated and autonomous, the Internet of Things and its evolution, known as the Internet of Everything (IoE, for short), involving also people, robots, processes and data in the network, represent the main driver to allow industries to put the experiences and needs of human beings at the center of their ecosystems. However, due to the extreme heterogeneity of the involved entities, their intrinsic need and capability to cooperate, and the aim to adapt to a dynamic user-centric context, special attention is required for the integration and processing of the data produced by such an IoE. This is the objective of the present paper, in which we propose a novel semantic model that formalizes the fundamental actors, elements and information of an IoE, along with their relationships. In our design, we focus on state-of-the-art design principles, in particular reuse, and abstraction, to build ``SemIoE'', a lightweight ontology inheriting and extending concepts from well-known and consolidated reference ontologies. The defined semantic layer represents a core data model that can be extended to embrace any modern industrial scenario. It represents the base of an IoE Knowledge Graph, on top of which, as an additional contribution, we analyze and define some essential services for an IoE-based industry.

{{</citation>}}


### (69/126) Ada-Retrieval: An Adaptive Multi-Round Retrieval Paradigm for Sequential Recommendations (Lei Li et al., 2024)

{{<citation>}}

Lei Li, Jianxun Lian, Xiao Zhou, Xing Xie. (2024)  
**Ada-Retrieval: An Adaptive Multi-Round Retrieval Paradigm for Sequential Recommendations**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.06633v1)  

---


**ABSTRACT**  
Retrieval models aim at selecting a small set of item candidates which match the preference of a given user. They play a vital role in large-scale recommender systems since subsequent models such as rankers highly depend on the quality of item candidates. However, most existing retrieval models employ a single-round inference paradigm, which may not adequately capture the dynamic nature of user preferences and stuck in one area in the item space. In this paper, we propose Ada-Retrieval, an adaptive multi-round retrieval paradigm for recommender systems that iteratively refines user representations to better capture potential candidates in the full item space. Ada-Retrieval comprises two key modules: the item representation adapter and the user representation adapter, designed to inject context information into items' and users' representations. The framework maintains a model-agnostic design, allowing seamless integration with various backbone models such as RNNs or Transformers. We perform experiments on three widely used public datasets, incorporating five powerful sequential recommenders as backbone models. Our results demonstrate that Ada-Retrieval significantly enhances the performance of various base models, with consistent improvements observed across different datasets. Our code and data are publicly available at: https://github.com/ll0ruc/Ada-Retrieval.

{{</citation>}}


### (70/126) UNEX-RL: Reinforcing Long-Term Rewards in Multi-Stage Recommender Systems with UNidirectional EXecution (Gengrui Zhang et al., 2024)

{{<citation>}}

Gengrui Zhang, Yao Wang, Xiaoshuang Chen, Hongyi Qian, Kaiqiao Zhan, Ben Wang. (2024)  
**UNEX-RL: Reinforcing Long-Term Rewards in Multi-Stage Recommender Systems with UNidirectional EXecution**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.06470v1)  

---


**ABSTRACT**  
In recent years, there has been a growing interest in utilizing reinforcement learning (RL) to optimize long-term rewards in recommender systems. Since industrial recommender systems are typically designed as multi-stage systems, RL methods with a single agent face challenges when optimizing multiple stages simultaneously. The reason is that different stages have different observation spaces, and thus cannot be modeled by a single agent. To address this issue, we propose a novel UNidirectional-EXecution-based multi-agent Reinforcement Learning (UNEX-RL) framework to reinforce the long-term rewards in multi-stage recommender systems. We show that the unidirectional execution is a key feature of multi-stage recommender systems, bringing new challenges to the applications of multi-agent reinforcement learning (MARL), namely the observation dependency and the cascading effect. To tackle these challenges, we provide a cascading information chain (CIC) method to separate the independent observations from action-dependent observations and use CIC to train UNEX-RL effectively. We also discuss practical variance reduction techniques for UNEX-RL. Finally, we show the effectiveness of UNEX-RL on both public datasets and an online recommender system with over 100 million users. Specifically, UNEX-RL reveals a 0.558% increase in users' usage time compared with single-agent RL algorithms in online A/B experiments, highlighting the effectiveness of UNEX-RL in industrial recommender systems.

{{</citation>}}


### (71/126) Zero-shot Generative Large Language Models for Systematic Review Screening Automation (Shuai Wang et al., 2024)

{{<citation>}}

Shuai Wang, Harrisen Scells, Shengyao Zhuang, Martin Potthast, Bevan Koopman, Guido Zuccon. (2024)  
**Zero-shot Generative Large Language Models for Systematic Review Screening Automation**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06320v1)  

---


**ABSTRACT**  
Systematic reviews are crucial for evidence-based medicine as they comprehensively analyse published research findings on specific questions. Conducting such reviews is often resource- and time-intensive, especially in the screening phase, where abstracts of publications are assessed for inclusion in a review. This study investigates the effectiveness of using zero-shot large language models~(LLMs) for automatic screening. We evaluate the effectiveness of eight different LLMs and investigate a calibration technique that uses a predefined recall threshold to determine whether a publication should be included in a systematic review. Our comprehensive evaluation using five standard test collections shows that instruction fine-tuning plays an important role in screening, that calibration renders LLMs practical for achieving a targeted recall, and that combining both with an ensemble of zero-shot models saves significant screening time compared to state-of-the-art approaches.

{{</citation>}}


### (72/126) MuGI: Enhancing Information Retrieval through Multi-Text Generation Intergration with Large Language Models (Le Zhang et al., 2024)

{{<citation>}}

Le Zhang, Yihong Wu. (2024)  
**MuGI: Enhancing Information Retrieval through Multi-Text Generation Intergration with Large Language Models**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval, Language Model, Text Generation  
[Paper Link](http://arxiv.org/abs/2401.06311v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have emerged as a pivotal force in language technology. Their robust reasoning capabilities and expansive knowledge repositories have enabled exceptional zero-shot generalization abilities across various facets of the natural language processing field, including information retrieval (IR). In this paper, we conduct an in-depth investigation into the utility of documents generated by LLMs for IR. We introduce a simple yet effective framework, Multi-Text Generation Integration (MuGI), to augment existing IR methodologies. Specifically, we prompt LLMs to generate multiple pseudo references and integrate with query for retrieval. The training-free MuGI model eclipses existing query expansion strategies, setting a new standard in sparse retrieval. It outstrips supervised counterparts like ANCE and DPR, achieving a notable over 18% enhancement in BM25 on the TREC DL dataset and a 7.5% increase on BEIR. Through MuGI, we have forged a rapid and high-fidelity re-ranking pipeline. This allows a relatively small 110M parameter retriever to surpass the performance of larger 3B models in in-domain evaluations, while also bridging the gap in out-of-distribution situations. We release our code and all generated references at https://github.com/lezhang7/Retrieval_MuGI.

{{</citation>}}


## eess.IV (2)



### (73/126) Local Gamma Augmentation for Ischemic Stroke Lesion Segmentation on MRI (Jon Middleton et al., 2024)

{{<citation>}}

Jon Middleton, Marko Bauer, Kaining Sheng, Jacob Johansen, Mathias Perslev, Silvia Ingala, Mads Nielsen, Akshay Pai. (2024)  
**Local Gamma Augmentation for Ischemic Stroke Lesion Segmentation on MRI**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2401.06893v1)  

---


**ABSTRACT**  
The identification and localisation of pathological tissues in medical images continues to command much attention among deep learning practitioners. When trained on abundant datasets, deep neural networks can match or exceed human performance. However, the scarcity of annotated data complicates the training of these models. Data augmentation techniques can compensate for a lack of training samples. However, many commonly used augmentation methods can fail to provide meaningful samples during model fitting. We present local gamma augmentation, a technique for introducing new instances of intensities in pathological tissues. We leverage local gamma augmentation to compensate for a bias in intensities corresponding to ischemic stroke lesions in human brain MRIs. On three datasets, we show how local gamma augmentation can improve the image-level sensitivity of a deep neural network tasked with ischemic lesion segmentation on magnetic resonance images.

{{</citation>}}


### (74/126) MedTransformer: Accurate AD Diagnosis for 3D MRI Images through 2D Vision Transformers (Yifeng Wang et al., 2024)

{{<citation>}}

Yifeng Wang, Ke Chen, Yihan Zhang, Haohan Wang. (2024)  
**MedTransformer: Accurate AD Diagnosis for 3D MRI Images through 2D Vision Transformers**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.06349v1)  

---


**ABSTRACT**  
Automated diagnosis of AD in brain images is becoming a clinically important technique to support precision and efficient diagnosis and treatment planning. A few efforts have been made to automatically diagnose AD in magnetic resonance imaging (MRI) using three-dimensional CNNs. However, due to the complexity of 3D models, the performance is still unsatisfactory, both in terms of accuracy and efficiency. To overcome the complexities of 3D images and 3D models, in this study, we aim to attack this problem with 2D vision Transformers. We propose a 2D transformer-based medical image model with various transformer attention encoders to diagnose AD in 3D MRI images, by cutting the 3D images into multiple 2D slices.The model consists of four main components: shared encoders across three dimensions, dimension-specific encoders, attention across images from the same dimension, and attention across three dimensions. It is used to obtain attention relationships among multiple sequences from different dimensions (axial, coronal, and sagittal) and multiple slices. We also propose morphology augmentation, an erosion and dilation based method to increase the structural difference between AD and normal images. In this experiment, we use multiple datasets from ADNI, AIBL, MIRAID, OASIS to show the performance of our model. Our proposed MedTransformer demonstrates a strong ability in diagnosing AD. These results demonstrate the effectiveness of MedTransformer in learning from 3D data using a much smaller model and its capability to generalize among different medical tasks, which provides a possibility to help doctors diagnose AD in a simpler way.

{{</citation>}}


## cs.AR (2)



### (75/126) Accelerating Neural Networks for Large Language Models and Graph Processing with Silicon Photonics (Salma Afifi et al., 2024)

{{<citation>}}

Salma Afifi, Febin Sunny, Mahdi Nikdast, Sudeep Pasricha. (2024)  
**Accelerating Neural Networks for Large Language Models and Graph Processing with Silicon Photonics**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-LG, cs.AR  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.06885v1)  

---


**ABSTRACT**  
In the rapidly evolving landscape of artificial intelligence, large language models (LLMs) and graph processing have emerged as transformative technologies for natural language processing (NLP), computer vision, and graph-structured data applications. However, the complex structures of these models pose challenges for acceleration on conventional electronic platforms. In this paper, we describe novel hardware accelerators based on silicon photonics to accelerate transformer neural networks that are used in LLMs and graph neural networks for graph data processing. Our analysis demonstrates that both hardware accelerators achieve at least 10.2x throughput improvement and 3.8x better energy efficiency over multiple state-of-the-art electronic hardware accelerators designed for LLMs and graph processing.

{{</citation>}}


### (76/126) Zero-Shot RTL Code Generation with Attention Sink Augmented Large Language Models (Selim Sandal et al., 2024)

{{<citation>}}

Selim Sandal, Ismail Akturk. (2024)  
**Zero-Shot RTL Code Generation with Attention Sink Augmented Large Language Models**  

---
Primary Category: cs.AR  
Categories: cs-AI, cs-AR, cs-LG, cs-PL, cs-SE, cs.AR  
Keywords: Attention, Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.08683v1)  

---


**ABSTRACT**  
The design and optimization of hardware have traditionally been resource-intensive, demanding considerable expertise and dependence on established design automation tools. This paper discusses the possibility of exploiting large language models to streamline the code generation process in hardware design. In contrast to earlier studies, this paper aims to use large language models that accepts high-level design specifications through a single prompt to generate corresponding Register-Transfer Level (RTL) code. The ability to use large language models on RTL code generation not only expedites design iteration cycles but also facilitates the exploration of design spaces that have computational challenges for conventional techniques. Through our evaluation, we demonstrate the shortcoming of existing attention mechanisms, and present the abilities of language models to produce functional, optimized, and industry-standard compliant RTL code when a novel attention mechanism is used. These findings underscore the expanding role of large language models in shaping the future landscape of architectural exploration and automation in hardware design.

{{</citation>}}


## cs.SE (6)



### (77/126) Automated Test Case Repair Using Language Models (Ahmadreza Saboor Yaraghi et al., 2024)

{{<citation>}}

Ahmadreza Saboor Yaraghi, Darren Holden, Nafiseh Kahani, Lionel Briand. (2024)  
**Automated Test Case Repair Using Language Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06765v1)  

---


**ABSTRACT**  
Ensuring the quality of software systems through testing is essential, yet maintaining test cases poses significant challenges and costs. The need for frequent updates to align with the evolving system under test often entails high complexity and cost for maintaining these test cases. Further, unrepaired broken test cases can degrade test suite quality and disrupt the software development process, wasting developers' time. To address this challenge, we present TaRGet (Test Repair GEneraTor), a novel approach leveraging pre-trained code language models for automated test case repair. TaRGet treats test repair as a language translation task, employing a two-step process to fine-tune a language model based on essential context data characterizing the test breakage. To evaluate our approach, we introduce TaRBench, a comprehensive benchmark we developed covering 45,373 broken test repairs across 59 open-source projects. Our results demonstrate TaRGet's effectiveness, achieving a 66.1% exact match accuracy. Furthermore, our study examines the effectiveness of TaRGet across different test repair scenarios. We provide a practical guide to predict situations where the generated test repairs might be less reliable. We also explore whether project-specific data is always necessary for fine-tuning and if our approach can be effective on new projects.

{{</citation>}}


### (78/126) Automated Security Findings Management: A Case Study in Industrial DevOps (Markus Voggenreiter et al., 2024)

{{<citation>}}

Markus Voggenreiter, Florian Angermeir, Fabiola Moyón, Ulrich Schöpp, Pierre Bonvin. (2024)  
**Automated Security Findings Management: A Case Study in Industrial DevOps**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.06602v1)  

---


**ABSTRACT**  
In recent years, DevOps, the unification of development and operation workflows, has become a trend for the industrial software development lifecycle. Security activities turned into an essential field of application for DevOps principles as they are a fundamental part of secure software development in the industry. A common practice arising from this trend is the automation of security tests that analyze a software product from several perspectives. To effectively improve the security of the analyzed product, the identified security findings must be managed and looped back to the project team for stakeholders to take action. This management must cope with several challenges ranging from low data quality to a consistent prioritization of findings while following DevOps aims. To manage security findings with the same efficiency as other activities in DevOps projects, a methodology for the management of industrial security findings minding DevOps principles is essential.   In this paper, we propose a methodology for the management of security findings in industrial DevOps projects, summarizing our research in this domain and presenting the resulting artifact. As an instance of the methodology, we developed the Security Flama, a semantic knowledge base for the automated management of security findings. To analyze the impact of our methodology on industrial practice, we performed a case study on two DevOps projects of a multinational industrial enterprise. The results emphasize the importance of using such an automated methodology in industrial DevOps projects, confirm our approach's usefulness and positive impact on the studied projects, and identify the communication strategy as a crucial factor for usability in practice.

{{</citation>}}


### (79/126) TestSpark: IntelliJ IDEA's Ultimate Test Generation Companion (Arkadii Sapozhnikov et al., 2024)

{{<citation>}}

Arkadii Sapozhnikov, Mitchell Olsthoorn, Annibale Panichella, Vladimir Kovalenko, Pouria Derakhshanfar. (2024)  
**TestSpark: IntelliJ IDEA's Ultimate Test Generation Companion**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06580v1)  

---


**ABSTRACT**  
Writing software tests is laborious and time-consuming. To address this, prior studies introduced various automated test-generation techniques. A well-explored research direction in this field is unit test generation, wherein artificial intelligence (AI) techniques create tests for a method/class under test. While many of these techniques have primarily found applications in a research context, existing tools (e.g., EvoSuite, Randoop, and AthenaTest) are not user-friendly and are tailored to a single technique. This paper introduces TestSpark, a plugin for IntelliJ IDEA that enables users to generate unit tests with only a few clicks directly within their Integrated Development Environment (IDE). Furthermore, TestSpark also allows users to easily modify and run each generated test and integrate them into the project workflow. TestSpark leverages the advances of search-based test generation tools, and it introduces a technique to generate unit tests using Large Language Models (LLMs) by creating a feedback cycle between the IDE and the LLM. Since TestSpark is an open-source (https://github.com/JetBrains-Research/TestSpark), extendable, and well-documented tool, it is possible to add new test generation methods into the plugin with the minimum effort. This paper also explains our future studies related to TestSpark and our preliminary results. Demo video: https://youtu.be/0F4PrxWfiXo

{{</citation>}}


### (80/126) Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers (Yuling Shi et al., 2024)

{{<citation>}}

Yuling Shi, Hongyu Zhang, Chengcheng Wan, Xiaodong Gu. (2024)  
**Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers**  

---
Primary Category: cs.SE  
Categories: I-2-7; D-2, cs-AI, cs-CL, cs-SE, cs.SE  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2401.06461v1)  

---


**ABSTRACT**  
Large language models have catalyzed an unprecedented wave in code generation. While achieving significant advances, they blur the distinctions between machine-and human-authored source code, causing integrity and authenticity issues of software artifacts. Previous methods such as DetectGPT have proven effective in discerning machine-generated texts, but they do not identify and harness the unique patterns of machine-generated code. Thus, its applicability falters when applied to code. In this paper, we carefully study the specific patterns that characterize machine and human-authored code. Through a rigorous analysis of code attributes such as length, lexical diversity, and naturalness, we expose unique pat-terns inherent to each source. We particularly notice that the structural segmentation of code is a critical factor in identifying its provenance. Based on our findings, we propose a novel machine-generated code detection method called DetectCodeGPT, which improves DetectGPT by capturing the distinct structural patterns of code. Diverging from conventional techniques that depend on external LLMs for perturbations, DetectCodeGPT perturbs the code corpus by strategically inserting spaces and newlines, ensuring both efficacy and efficiency. Experiment results show that our approach significantly outperforms state-of-the-art techniques in detecting machine-generated code.

{{</citation>}}


### (81/126) DevEval: Evaluating Code Generation in Practical Software Projects (Jia Li et al., 2024)

{{<citation>}}

Jia Li, Ge Li, Yunfei Zhao, Yongmin Li, Zhi Jin, Hao Zhu, Huanyu Liu, Kaibo Liu, Lecheng Wang, Zheng Fang, Lanshen Wang, Jiazheng Ding, Xuanming Zhang, Yihong Dong, Yuqi Zhu, Bin Gu, Mengfei Yang. (2024)  
**DevEval: Evaluating Code Generation in Practical Software Projects**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06401v1)  

---


**ABSTRACT**  
How to evaluate Large Language Models (LLMs) in code generation is an open question. Many benchmarks have been proposed but are inconsistent with practical software projects, e.g., unreal program distributions, insufficient dependencies, and small-scale project contexts. Thus, the capabilities of LLMs in practical projects are still unclear. In this paper, we propose a new benchmark named DevEval, aligned with Developers' experiences in practical projects. DevEval is collected through a rigorous pipeline, containing 2,690 samples from 119 practical projects and covering 10 domains. Compared to previous benchmarks, DevEval aligns to practical projects in multiple dimensions, e.g., real program distributions, sufficient dependencies, and enough-scale project contexts. We assess five popular LLMs on DevEval (e.g., gpt-4, gpt-3.5-turbo, CodeLLaMa, and StarCoder) and reveal their actual abilities in code generation. For instance, the highest Pass@1 of gpt-3.5-turbo only is 42 in our experiments. We also discuss the challenges and future directions of code generation in practical projects. We open-source DevEval and hope it can facilitate the development of code generation in practical projects.

{{</citation>}}


### (82/126) Teaching Code LLMs to Use Autocompletion Tools in Repository-Level Code Generation (Chong Wang et al., 2024)

{{<citation>}}

Chong Wang, Jian Zhang, Yebo Feng, Tianlin Li, Weisong Sun, Yang Liu, Xin Peng. (2024)  
**Teaching Code LLMs to Use Autocompletion Tools in Repository-Level Code Generation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2401.06391v1)  

---


**ABSTRACT**  
Recent code large language models (LLMs) have shown promising performance in generating standalone functions but face limitations in repository-level code generation due to their lack of awareness of repository-level dependencies (e.g., user-defined attributes), resulting in dependency errors such as undefined-variable and no-member errors. In this work, we introduce ToolGen, an approach that integrates autocompletion tools into the code LLM generation process to address these dependencies. ToolGen comprises two main phases: Data Augmentation and Model Fine-tuning (Offline), and Tool-integrated Code Generation (Online). During the offline phase, ToolGen augments functions within a given code corpus with a special mark token, indicating positions to trigger autocompletion tools. These augmented functions, along with their corresponding docstrings, are then used to fine-tune a selected code LLM. In the online phase, ToolGen iteratively generates functions by predicting tokens step-by-step using the fine-tuned LLM. Whenever a mark token is encountered, ToolGen invokes the autocompletion tool to suggest code completions and selects the most appropriate one.   We conduct comprehensive experiments to evaluate ToolGen's effectiveness in repository-level code generation. To facilitate this evaluation, we create a benchmark comprising 680 real-world code repositories and introduce two new repository-level metrics: Dependency Coverage and Success Rate. The results demonstrate that ToolGen significantly improves dependency coverage by 15.2% to 45.8% and success rates by 10.9% to 42.2% across three distinct code LLMs, while maintaining competitive performance in widely-recognized similarity metrics. Furthermore, our generalizability evaluation confirms ToolGen's consistent performance when applied to diverse code LLMs, including various model architectures and scales.

{{</citation>}}


## cs.CV (16)



### (83/126) Synthetic Data Generation Framework, Dataset, and Efficient Deep Model for Pedestrian Intention Prediction (Muhammad Naveed Riaz et al., 2024)

{{<citation>}}

Muhammad Naveed Riaz, Maciej Wielgosz, Abel Garcia Romera, Antonio M. Lopez. (2024)  
**Synthetic Data Generation Framework, Dataset, and Efficient Deep Model for Pedestrian Intention Prediction**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.06757v1)  

---


**ABSTRACT**  
Pedestrian intention prediction is crucial for autonomous driving. In particular, knowing if pedestrians are going to cross in front of the ego-vehicle is core to performing safe and comfortable maneuvers. Creating accurate and fast models that predict such intentions from sequential images is challenging. A factor contributing to this is the lack of datasets with diverse crossing and non-crossing (C/NC) scenarios. We address this scarceness by introducing a framework, named ARCANE, which allows programmatically generating synthetic datasets consisting of C/NC video clip samples. As an example, we use ARCANE to generate a large and diverse dataset named PedSynth. We will show how PedSynth complements widely used real-world datasets such as JAAD and PIE, so enabling more accurate models for C/NC prediction. Considering the onboard deployment of C/NC prediction models, we also propose a deep model named PedGNN, which is fast and has a very low memory footprint. PedGNN is based on a GNN-GRU architecture that takes a sequence of pedestrian skeletons as input to predict crossing intentions.

{{</citation>}}


### (84/126) Decoupling Pixel Flipping and Occlusion Strategy for Consistent XAI Benchmarks (Stefan Blücher et al., 2024)

{{<citation>}}

Stefan Blücher, Johanna Vielhaben, Nils Strodthoff. (2024)  
**Decoupling Pixel Flipping and Occlusion Strategy for Consistent XAI Benchmarks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06654v1)  

---


**ABSTRACT**  
Feature removal is a central building block for eXplainable AI (XAI), both for occlusion-based explanations (Shapley values) as well as their evaluation (pixel flipping, PF). However, occlusion strategies can vary significantly from simple mean replacement up to inpainting with state-of-the-art diffusion models. This ambiguity limits the usefulness of occlusion-based approaches. For example, PF benchmarks lead to contradicting rankings. This is amplified by competing PF measures: Features are either removed starting with most influential first (MIF) or least influential first (LIF). This study proposes two complementary perspectives to resolve this disagreement problem. Firstly, we address the common criticism of occlusion-based XAI, that artificial samples lead to unreliable model evaluations. We propose to measure the reliability by the R(eference)-Out-of-Model-Scope (OMS) score. The R-OMS score enables a systematic comparison of occlusion strategies and resolves the disagreement problem by grouping consistent PF rankings. Secondly, we show that the insightfulness of MIF and LIF is conversely dependent on the R-OMS score. To leverage this, we combine the MIF and LIF measures into the symmetric relevance gain (SRG) measure. This breaks the inherent connection to the underlying occlusion strategy and leads to consistent rankings. This resolves the disagreement problem, which we verify for a set of 40 different occlusion strategies.

{{</citation>}}


### (85/126) Adversarial Examples are Misaligned in Diffusion Model Manifolds (Peter Lorenz et al., 2024)

{{<citation>}}

Peter Lorenz, Ricard Durall, Janis Keuper. (2024)  
**Adversarial Examples are Misaligned in Diffusion Model Manifolds**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.06637v3)  

---


**ABSTRACT**  
In recent years, diffusion models (DMs) have drawn significant attention for their success in approximating data distributions, yielding state-of-the-art generative results. Nevertheless, the versatility of these models extends beyond their generative capabilities to encompass various vision applications, such as image inpainting, segmentation, adversarial robustness, among others. This study is dedicated to the investigation of adversarial attacks through the lens of diffusion models. However, our objective does not involve enhancing the adversarial robustness of image classifiers. Instead, our focus lies in utilizing the diffusion model to detect and analyze the anomalies introduced by these attacks on images. To that end, we systematically examine the alignment of the distributions of adversarial examples when subjected to the process of transformation using diffusion models. The efficacy of this approach is assessed across CIFAR-10 and ImageNet datasets, including varying image sizes in the latter. The results demonstrate a notable capacity to discriminate effectively between benign and attacked images, providing compelling evidence that adversarial instances do not align with the learned manifold of the DMs.

{{</citation>}}


### (86/126) Enhancing Consistency and Mitigating Bias: A Data Replay Approach for Incremental Learning (Chenyang Wang et al., 2024)

{{<citation>}}

Chenyang Wang, Junjun Jiang, Xingyu Hu, Xianming Liu, Xiangyang Ji. (2024)  
**Enhancing Consistency and Mitigating Bias: A Data Replay Approach for Incremental Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias, ImageNet  
[Paper Link](http://arxiv.org/abs/2401.06548v1)  

---


**ABSTRACT**  
Deep learning systems are prone to catastrophic forgetting when learning from a sequence of tasks, where old data from experienced tasks is unavailable when learning from a new task. To mitigate the problem, a line of methods propose to replay the data of experienced tasks when learning new tasks. These methods usually adopt an extra memory to store the data for replay. However, it is not expected in practice considering the memory constraint or data privacy issue. As a replacement, data-free data replay methods are proposed by inverting samples from the classification model. Though achieving good results, these methods still suffer from the inconsistency of the inverted and real training data, which is neglected in the inversion stage in recent works. To that effect, we propose to measure the data consistency quantitatively by some simplification and assumptions. Using the measurement, we analyze existing techniques for inverting samples and get some insightful information that inspires a novel loss function to reduce the inconsistency. Specifically, the loss minimizes the KL divergence of the distributions of inverted and real data under the tied multivariate Gaussian assumption, which is easy to implement in continual learning. In addition, we observe that the norms of old class weights turn to decrease continually as learning progresses. We thus analyze the underlying reasons and propose a simple regularization term to balance the class weights so that the samples of old classes are more distinguishable. To conclude, we propose the Consistency enhanced data replay with debiased classifier for Class Incremental Learning (CCIL). Extensive experiments on CIFAR-100, Tiny-ImageNet, and ImageNet100 show consistently improved performance of CCIL compared to previous approaches.

{{</citation>}}


### (87/126) Robustness-Aware 3D Object Detection in Autonomous Driving: A Review and Outlook (Ziying Song et al., 2024)

{{<citation>}}

Ziying Song, Lin Liu, Feiyang Jia, Yadan Luo, Guoxin Zhang, Lei Yang, Li Wang, Caiyan Jia. (2024)  
**Robustness-Aware 3D Object Detection in Autonomous Driving: A Review and Outlook**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.06542v1)  

---


**ABSTRACT**  
In the realm of modern autonomous driving, the perception system is indispensable for accurately assessing the state of the surrounding environment, thereby enabling informed prediction and planning. Key to this system is 3D object detection methods, that utilize vehicle-mounted sensors such as LiDAR and cameras to identify the size, category, and location of nearby objects. Despite the surge in 3D object detection methods aimed at enhancing detection precision and efficiency, there is a gap in the literature that systematically examines their resilience against environmental variations, noise, and weather changes. This study emphasizes the importance of robustness, alongside accuracy and latency, in evaluating perception systems under practical scenarios. Our work presents an extensive survey of camera-based, LiDAR-based, and multimodal 3D object detection algorithms, thoroughly evaluating their trade-off between accuracy, latency, and robustness, particularly on datasets like KITTI-C and nuScenes-C to ensure fair comparisons. Among these,multimodal 3D detection approaches exhibit superior robustness and a novel taxonomy is introduced to reorganize its literature for enhanced clarity. This survey aims to offer a more practical perspective on the current capabilities and constraints of 3D object detection algorithms in real-world applications, thus steering future research towards robustness-centric advancements

{{</citation>}}


### (88/126) PCB-Vision: A Multiscene RGB-Hyperspectral Benchmark Dataset of Printed Circuit Boards (Elias Arbash et al., 2024)

{{<citation>}}

Elias Arbash, Margret Fuchs, Behnood Rasti, Sandra Lorenz, Pedram Ghamisi, Richard Gloaguen. (2024)  
**PCB-Vision: A Multiscene RGB-Hyperspectral Benchmark Dataset of Printed Circuit Boards**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV, eess-IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.06528v1)  

---


**ABSTRACT**  
Addressing the critical theme of recycling electronic waste (E-waste), this contribution is dedicated to developing advanced automated data processing pipelines as a basis for decision-making and process control. Aligning with the broader goals of the circular economy and the United Nations (UN) Sustainable Development Goals (SDG), our work leverages non-invasive analysis methods utilizing RGB and hyperspectral imaging data to provide both quantitative and qualitative insights into the E-waste stream composition for optimizing recycling efficiency. In this paper, we introduce 'PCB-Vision'; a pioneering RGB-hyperspectral printed circuit board (PCB) benchmark dataset, comprising 53 RGB images of high spatial resolution paired with their corresponding high spectral resolution hyperspectral data cubes in the visible and near-infrared (VNIR) range. Grounded in open science principles, our dataset provides a comprehensive resource for researchers through high-quality ground truths, focusing on three primary PCB components: integrated circuits (IC), capacitors, and connectors. We provide extensive statistical investigations on the proposed dataset together with the performance of several state-of-the-art (SOTA) models, including U-Net, Attention U-Net, Residual U-Net, LinkNet, and DeepLabv3+. By openly sharing this multi-scene benchmark dataset along with the baseline codes, we hope to foster transparent, traceable, and comparable developments of advanced data processing across various scientific communities, including, but not limited to, computer vision and remote sensing. Emphasizing our commitment to supporting a collaborative and inclusive scientific community, all materials, including code, data, ground truth, and masks, will be accessible at https://github.com/hifexplo/PCBVision.

{{</citation>}}


### (89/126) Exploring Diverse Representations for Open Set Recognition (Yu Wang et al., 2024)

{{<citation>}}

Yu Wang, Junxian Mu, Pengfei Zhu, Qinghua Hu. (2024)  
**Exploring Diverse Representations for Open Set Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.06521v1)  

---


**ABSTRACT**  
Open set recognition (OSR) requires the model to classify samples that belong to closed sets while rejecting unknown samples during test. Currently, generative models often perform better than discriminative models in OSR, but recent studies show that generative models may be computationally infeasible or unstable on complex tasks. In this paper, we provide insights into OSR and find that learning supplementary representations can theoretically reduce the open space risk. Based on the analysis, we propose a new model, namely Multi-Expert Diverse Attention Fusion (MEDAF), that learns diverse representations in a discriminative way. MEDAF consists of multiple experts that are learned with an attention diversity regularization term to ensure the attention maps are mutually different. The logits learned by each expert are adaptively fused and used to identify the unknowns through the score function. We show that the differences in attention maps can lead to diverse representations so that the fused representations can well handle the open space. Extensive experiments are conducted on standard and OSR large-scale benchmarks. Results show that the proposed discriminative method can outperform existing generative models by up to 9.5% on AUROC and achieve new state-of-the-art performance with little computational cost. Our method can also seamlessly integrate existing classification models. Code is available at https://github.com/Vanixxz/MEDAF.

{{</citation>}}


### (90/126) Frequency Masking for Universal Deepfake Detection (Chandler Timm Doloriel et al., 2024)

{{<citation>}}

Chandler Timm Doloriel, Ngai-Man Cheung. (2024)  
**Frequency Masking for Universal Deepfake Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06506v3)  

---


**ABSTRACT**  
We study universal deepfake detection. Our goal is to detect synthetic images from a range of generative AI approaches, particularly from emerging ones which are unseen during training of the deepfake detector. Universal deepfake detection requires outstanding generalization capability. Motivated by recently proposed masked image modeling which has demonstrated excellent generalization in self-supervised pre-training, we make the first attempt to explore masked image modeling for universal deepfake detection. We study spatial and frequency domain masking in training deepfake detectors. Based on empirical analysis, we propose a novel deepfake detector via frequency masking. Our focus on frequency domain is different from the majority, which primarily target spatial domain detection. Our comparative analyses reveal substantial performance gains over existing methods. Code and models are publicly available.

{{</citation>}}


### (91/126) Improving the Detection of Small Oriented Objects in Aerial Images (Chandler Timm C. Doloriel et al., 2024)

{{<citation>}}

Chandler Timm C. Doloriel, Rhandley D. Cajote. (2024)  
**Improving the Detection of Small Oriented Objects in Aerial Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.06503v1)  

---


**ABSTRACT**  
Small oriented objects that represent tiny pixel-area in large-scale aerial images are difficult to detect due to their size and orientation. Existing oriented aerial detectors have shown promising results but are mainly focused on orientation modeling with less regard to the size of the objects. In this work, we proposed a method to accurately detect small oriented objects in aerial images by enhancing the classification and regression tasks of the oriented object detection model. We designed the Attention-Points Network consisting of two losses: Guided-Attention Loss (GALoss) and Box-Points Loss (BPLoss). GALoss uses an instance segmentation mask as ground-truth to learn the attention features needed to improve the detection of small objects. These attention features are then used to predict box points for BPLoss, which determines the points' position relative to the target oriented bounding box. Experimental results show the effectiveness of our Attention-Points Network on a standard oriented aerial dataset with small object instances (DOTA-v1.5) and on a maritime-related dataset (HRSC2016). The code is publicly available.

{{</citation>}}


### (92/126) AttributionScanner: A Visual Analytics System for Metadata-Free Data-Slicing Based Model Validation (Xiwei Xuan et al., 2024)

{{<citation>}}

Xiwei Xuan, Jorge Piazentin Ono, Liang Gou, Kwan-Liu Ma, Liu Ren. (2024)  
**AttributionScanner: A Visual Analytics System for Metadata-Free Data-Slicing Based Model Validation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-HC, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06462v1)  

---


**ABSTRACT**  
Data slice-finding is an emerging technique for evaluating machine learning models. It works by identifying subgroups within a specified dataset that exhibit poor performance, often defined by distinct feature sets or meta-information. However, in the context of unstructured image data, data slice-finding poses two notable challenges: it requires additional metadata -- a laborious and costly requirement, and also demands non-trivial efforts for interpreting the root causes of the underperformance within data slices. To address these challenges, we introduce AttributionScanner, an innovative human-in-the-loop Visual Analytics (VA) system, designed for data-slicing-based machine learning (ML) model validation. Our approach excels in identifying interpretable data slices, employing explainable features extracted through the lens of Explainable AI (XAI) techniques, and removing the necessity for additional metadata of textual annotations or cross-model embeddings. AttributionScanner demonstrates proficiency in pinpointing critical model issues, including spurious correlations and mislabeled data. Our novel VA interface visually summarizes data slices, enabling users to gather insights into model behavior patterns effortlessly. Furthermore, our framework closes the ML Development Cycle by empowering domain experts to address model issues by using a cutting-edge neural network regularization technique. The efficacy of AttributionScanner is underscored through two prototype use cases, elucidating its substantial effectiveness in model validation for vision-centric tasks. Our approach paves the way for ML researchers and practitioners to drive interpretable model validation in a data-efficient way, ultimately leading to more reliable and accurate models.

{{</citation>}}


### (93/126) UPDP: A Unified Progressive Depth Pruner for CNN and Vision Transformer (Ji Liu et al., 2024)

{{<citation>}}

Ji Liu, Dehua Tang, Yuanxian Huang, Li Zhang, Xiaocheng Zeng, Dong Li, Mingjie Lu, Jinzhang Peng, Yu Wang, Fan Jiang, Lu Tian, Ashish Sirasao. (2024)  
**UPDP: A Unified Progressive Depth Pruner for CNN and Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.06426v1)  

---


**ABSTRACT**  
Traditional channel-wise pruning methods by reducing network channels struggle to effectively prune efficient CNN models with depth-wise convolutional layers and certain efficient modules, such as popular inverted residual blocks. Prior depth pruning methods by reducing network depths are not suitable for pruning some efficient models due to the existence of some normalization layers. Moreover, finetuning subnet by directly removing activation layers would corrupt the original model weights, hindering the pruned model from achieving high performance. To address these issues, we propose a novel depth pruning method for efficient models. Our approach proposes a novel block pruning strategy and progressive training method for the subnet. Additionally, we extend our pruning method to vision transformer models. Experimental results demonstrate that our method consistently outperforms existing depth pruning methods across various pruning configurations. We obtained three pruned ConvNeXtV1 models with our method applying on ConvNeXtV1, which surpass most SOTA efficient models with comparable inference performance. Our method also achieves state-of-the-art pruning performance on the vision transformer model.

{{</citation>}}


### (94/126) ModaVerse: Efficiently Transforming Modalities with LLMs (Xinyu Wang et al., 2024)

{{<citation>}}

Xinyu Wang, Bohan Zhuang, Qi Wu. (2024)  
**ModaVerse: Efficiently Transforming Modalities with LLMs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06395v1)  

---


**ABSTRACT**  
Humans possess the capability to comprehend diverse modalities and seamlessly transfer information between them. In this work, we introduce ModaVerse, a Multi-modal Large Language Model (MLLM) capable of comprehending and transforming content across various modalities including images, videos, and audio. Predominant MLLM frameworks have largely relied on the alignment of latent spaces of textual and non-textual features. This alignment process, which synchronizes a language model trained on textual data with encoders and decoders trained on multi-modal data, often necessitates extensive training of several projection layers in multiple stages. Inspired by LLM-as-agent methodologies, we propose a novel Input/Output (I/O) alignment mechanism that operates directly at the level of natural language. It aligns the LLM's output with the input of generative models, avoiding the complexities associated with latent feature alignments, and simplifying the multiple training stages of existing MLLMs into a single, efficient process. This conceptual advancement leads to significant reductions in both data and computational costs. By conducting experiments on several benchmarks, we demonstrate that our approach attains comparable performance with the state of the art while achieving considerable efficiencies in data usage and training duration.

{{</citation>}}


### (95/126) Hyper-STTN: Social Group-aware Spatial-Temporal Transformer Network for Human Trajectory Prediction with Hypergraph Reasoning (Weizheng Wang et al., 2024)

{{<citation>}}

Weizheng Wang, Le Mao, Baijian Yang, Guohua Chen, Byung-Cheol Min. (2024)  
**Hyper-STTN: Social Group-aware Spatial-Temporal Transformer Network for Human Trajectory Prediction with Hypergraph Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reasoning, Transformer  
[Paper Link](http://arxiv.org/abs/2401.06344v1)  

---


**ABSTRACT**  
Predicting crowded intents and trajectories is crucial in varouls real-world applications, including service robots and autonomous vehicles. Understanding environmental dynamics is challenging, not only due to the complexities of modeling pair-wise spatial and temporal interactions but also the diverse influence of group-wise interactions. To decode the comprehensive pair-wise and group-wise interactions in crowded scenarios, we introduce Hyper-STTN, a Hypergraph-based Spatial-Temporal Transformer Network for crowd trajectory prediction. In Hyper-STTN, crowded group-wise correlations are constructed using a set of multi-scale hypergraphs with varying group sizes, captured through random-walk robability-based hypergraph spectral convolution. Additionally, a spatial-temporal transformer is adapted to capture pedestrians' pair-wise latent interactions in spatial-temporal dimensions. These heterogeneous group-wise and pair-wise are then fused and aligned though a multimodal transformer network. Hyper-STTN outperformes other state-of-the-art baselines and ablation models on 5 real-world pedestrian motion datasets.

{{</citation>}}


### (96/126) AffordanceLLM: Grounding Affordance from Vision Language Models (Shengyi Qian et al., 2024)

{{<citation>}}

Shengyi Qian, Weifeng Chen, Min Bai, Xiong Zhou, Zhuowen Tu, Li Erran Li. (2024)  
**AffordanceLLM: Grounding Affordance from Vision Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06341v1)  

---


**ABSTRACT**  
Affordance grounding refers to the task of finding the area of an object with which one can interact. It is a fundamental but challenging task, as a successful solution requires the comprehensive understanding of a scene in multiple aspects including detection, localization, and recognition of objects with their parts, of geo-spatial configuration/layout of the scene, of 3D shapes and physics, as well as of the functionality and potential interaction of the objects and humans. Much of the knowledge is hidden and beyond the image content with the supervised labels from a limited training set. In this paper, we make an attempt to improve the generalization capability of the current affordance grounding by taking the advantage of the rich world, abstract, and human-object-interaction knowledge from pretrained large-scale vision language models. Under the AGD20K benchmark, our proposed model demonstrates a significant performance gain over the competing methods for in-the-wild object affordance grounding. We further demonstrate it can ground affordance for objects from random Internet images, even if both objects and actions are unseen during training. Project site: https://jasonqsy.github.io/AffordanceLLM/

{{</citation>}}


### (97/126) Application Of Vision-Language Models For Assessing Osteoarthritis Disease Severity (Banafshe Felfeliyan et al., 2024)

{{<citation>}}

Banafshe Felfeliyan, Yuyue Zhou, Shrimanti Ghosh, Jessica Kupper, Shaobo Liu, Abhilash Hareendranathan, Jacob L. Jaremko. (2024)  
**Application Of Vision-Language Models For Assessing Osteoarthritis Disease Severity**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06331v1)  

---


**ABSTRACT**  
Osteoarthritis (OA) poses a global health challenge, demanding precise diagnostic methods. Current radiographic assessments are time consuming and prone to variability, prompting the need for automated solutions. The existing deep learning models for OA assessment are unimodal single task systems and they don't incorporate relevant text information such as patient demographics, disease history, or physician reports. This study investigates employing Vision Language Processing (VLP) models to predict OA severity using Xray images and corresponding reports. Our method leverages Xray images of the knee and diverse report templates generated from tabular OA scoring values to train a CLIP (Contrastive Language Image PreTraining) style VLP model. Furthermore, we incorporate additional contrasting captions to enforce the model to discriminate between positive and negative reports. Results demonstrate the efficacy of these models in learning text image representations and their contextual relationships, showcase potential advancement in OA assessment, and establish a foundation for specialized vision language models in medical contexts.

{{</citation>}}


### (98/126) Video Super-Resolution Transformer with Masked Inter&Intra-Frame Attention (Xingyu Zhou et al., 2024)

{{<citation>}}

Xingyu Zhou, Leheng Zhang, Xiaorui Zhao, Keze Wang, Leida Li, Shuhang Gu. (2024)  
**Video Super-Resolution Transformer with Masked Inter&Intra-Frame Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2401.06312v2)  

---


**ABSTRACT**  
Recently, Vision Transformer has achieved great success in recovering missing details in low-resolution sequences, i.e., the video super-resolution (VSR) task. Despite its superiority in VSR accuracy, the heavy computational burden as well as the large memory footprint hinder the deployment of Transformer-based VSR models on constrained devices. In this paper, we address the above issue by proposing a novel feature-level masked processing framework: VSR with Masked Intra and inter frame Attention (MIA-VSR). The core of MIA-VSR is leveraging feature-level temporal continuity between adjacent frames to reduce redundant computations and make more rational use of previously enhanced SR features. Concretely, we propose an intra-frame and inter-frame attention block which takes the respective roles of past features and input features into consideration and only exploits previously enhanced features to provide supplementary information. In addition, an adaptive block-wise mask prediction module is developed to skip unimportant computations according to feature similarity between adjacent frames. We conduct detailed ablation studies to validate our contributions and compare the proposed method with recent state-of-the-art VSR approaches. The experimental results demonstrate that MIA-VSR improves the memory and computation efficiency over state-of-the-art methods, without trading off PSNR accuracy. The code is available at https://github.com/LabShuHangGU/MIA-VSR.

{{</citation>}}


## physics.flu-dyn (1)



### (99/126) Solving the Discretised Multiphase Flow Equations with Interface Capturing on Structured Grids Using Machine Learning Libraries (Boyang Chen et al., 2024)

{{<citation>}}

Boyang Chen, Claire E. Heaney, Jefferson L. M. A. Gomes, Omar K. Matar, Christopher C. Pain. (2024)  
**Solving the Discretised Multiphase Flow Equations with Interface Capturing on Structured Grids Using Machine Learning Libraries**  

---
Primary Category: physics.flu-dyn  
Categories: cs-LG, physics-flu-dyn, physics.flu-dyn  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06755v1)  

---


**ABSTRACT**  
This paper solves the multiphase flow equations with interface capturing using the AI4PDEs approach (Artificial Intelligence for Partial Differential Equations). The solver within AI4PDEs uses tools from machine learning (ML) libraries to solve (exactly) partial differential equations (PDEs) that have been discretised using numerical methods. Convolutional layers can be used to express the discretisations as a neural network, whose weights are determined by the numerical method, rather than by training. To solve the system, a multigrid solver is implemented through a neural network with a U-Net architecture. Immiscible two-phase flow is modelled by the 3D incompressible Navier-Stokes equations with surface tension and advection of a volume fraction field, which describes the interface between the fluids. A new compressive algebraic volume-of-fluids method is introduced, based on a residual formulation using Petrov-Galerkin for accuracy and designed with AI4PDEs in mind. High-order finite-element based schemes are chosen to model a collapsing water column and a rising bubble. Results compare well with experimental data and other numerical results from the literature, demonstrating that, for the first time, finite element discretisations of multiphase flows can be solved using the neural network solver from the AI4PDEs approach. A benefit of expressing numerical discretisations as neural networks is that the code can run, without modification, on CPUs, GPUs or the latest accelerators designed especially to run AI codes.

{{</citation>}}


## cs.NI (3)



### (100/126) NetMind: Adaptive RAN Baseband Function Placement by GCN Encoding and Maze-solving DRL (Haiyuan Li et al., 2024)

{{<citation>}}

Haiyuan Li, Peizheng Li, Karcius Day Assis, Adnan Aijaz, Sen Shen, Reza Nejabati, Shuangyi Yan, Dimitra Simeonidou. (2024)  
**NetMind: Adaptive RAN Baseband Function Placement by GCN Encoding and Maze-solving DRL**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Graph Convolutional Network, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.06722v2)  

---


**ABSTRACT**  
The disaggregated and hierarchical architecture of advanced RAN presents significant challenges in efficiently placing baseband functions and user plane functions in conjunction with Multi-Access Edge Computing (MEC) to accommodate diverse 5G services. Therefore, this paper proposes a novel approach NetMind, which leverages Deep Reinforcement Learning (DRL) to determine the function placement strategies in RANs with diverse topologies, aiming at minimizing power consumption. NetMind formulates the function placement problem as a maze-solving task, enabling a Markov Decision Process with standardized action space scales across different networks. Additionally, a Graph Convolutional Network (GCN) based encoding mechanism is introduced, allowing features from different networks to be aggregated into a single RL agent. That facilitates the RL agent's generalization capability and minimizes the negative impact of retraining on power consumption. In an example with three sub-networks, NetMind achieves comparable performance to traditional methods that require a dedicated DRL agent for each network, resulting in a 70% reduction in training costs. Furthermore, it demonstrates a substantial 32.76% improvement in power savings and a 41.67% increase in service stability compared to benchmarks from the existing literature.

{{</citation>}}


### (101/126) Network Anatomy and Real-Time Measurement of Nvidia GeForce NOW Cloud Gaming (Minzhao Lyu et al., 2024)

{{<citation>}}

Minzhao Lyu, Sharat Chandra Madanapalli, Arun Vishwanath, Vijay Sivaraman. (2024)  
**Network Anatomy and Real-Time Measurement of Nvidia GeForce NOW Cloud Gaming**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs-PF, cs.NI  
Keywords: Amazon, Microsoft  
[Paper Link](http://arxiv.org/abs/2401.06366v1)  

---


**ABSTRACT**  
Cloud gaming, wherein game graphics is rendered in the cloud and streamed back to the user as real-time video, expands the gaming market to billions of users who do not have gaming consoles or high-power graphics PCs. Companies like Nvidia, Amazon, Sony and Microsoft are investing in building cloud gaming platforms to tap this large unserved market. However, cloud gaming requires the user to have high bandwidth and stable network connectivity - whereas a typical console game needs about 100-200 kbps, a cloud game demands minimum 10-20 Mbps. This makes the Internet Service Provider (ISP) a key player in ensuring the end-user's good gaming experience. In this paper we develop a method to detect user experience to detect Nvidia's GeForce NOW cloud gaming sessions over their network infrastructure, and measure associated user experience. In particular, we envision ISPs taking advantage of our method to provision network capacity at the right time and in the right place to support growth in cloud gaming at the right experience level; as well as identify the role of contextual factors such as user setup (browser vs app) and connectivity type (wired vs wireless) in performance degradation. We first present a detailed anatomy of flow establishment and volumetric profiles of cloud gaming sessions over multiple platforms, followed by a method to detect gameplay and measure key experience aspects such as latency, frame rate and resolution via real-time analysis of network traffic. The insights and methods are also validated in the lab for XBox Cloud Gaming platform. We then implement and deploy our method in a campus network to capture gameplay behaviors and experience measures across various user setups and connectivity types which we believe are valuable for network operators.

{{</citation>}}


### (102/126) A Semantic-Aware Multiple Access Scheme for Distributed, Dynamic 6G-Based Applications (Hamidreza Mazandarani et al., 2024)

{{<citation>}}

Hamidreza Mazandarani, Masoud Shokrnezhad, Tarik Taleb. (2024)  
**A Semantic-Aware Multiple Access Scheme for Distributed, Dynamic 6G-Based Applications**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-LG, cs-MA, cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.06308v1)  

---


**ABSTRACT**  
The emergence of the semantic-aware paradigm presents opportunities for innovative services, especially in the context of 6G-based applications. Although significant progress has been made in semantic extraction techniques, the incorporation of semantic information into resource allocation decision-making is still in its early stages, lacking consideration of the requirements and characteristics of future systems. In response, this paper introduces a novel formulation for the problem of multiple access to the wireless spectrum. It aims to optimize the utilization-fairness trade-off, using the $\alpha$-fairness metric, while accounting for user data correlation by introducing the concepts of self- and assisted throughputs. Initially, the problem is analyzed to identify its optimal solution. Subsequently, a Semantic-Aware Multi-Agent Double and Dueling Deep Q-Learning (SAMA-D3QL) technique is proposed. This method is grounded in Model-free Multi-Agent Deep Reinforcement Learning (MADRL), enabling the user equipment to autonomously make decisions regarding wireless spectrum access based solely on their local individual observations. The efficiency of the proposed technique is evaluated through two scenarios: single-channel and multi-channel. The findings illustrate that, across a spectrum of $\alpha$ values, association matrices, and channels, SAMA-D3QL consistently outperforms alternative approaches. This establishes it as a promising candidate for facilitating the realization of future federated, dynamically evolving applications.

{{</citation>}}


## cs.DC (1)



### (103/126) PolyTOPS: Reconfigurable and Flexible Polyhedral Scheduler (Gianpietro Consolaro et al., 2024)

{{<citation>}}

Gianpietro Consolaro, Zhen Zhang, Harenome Razanajato, Nelson Lossing, Nassim Tchoulak, Adilla Susungi, Artur Cesar Araujo Alves, Renwei Zhang, Denis Barthou, Corinne Ancourt, Cedric Bastoul. (2024)  
**PolyTOPS: Reconfigurable and Flexible Polyhedral Scheduler**  

---
Primary Category: cs.DC  
Categories: cs-CL, cs-DC, cs-PF, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06665v1)  

---


**ABSTRACT**  
Polyhedral techniques have been widely used for automatic code optimization in low-level compilers and higher-level processes. Loop optimization is central to this technique, and several polyhedral schedulers like Feautrier, Pluto, isl and Tensor Scheduler have been proposed, each of them targeting a different architecture, parallelism model, or application scenario. The need for scenario-specific optimization is growing due to the heterogeneity of architectures. One of the most critical cases is represented by NPUs (Neural Processing Units) used for AI, which may require loop optimization with different objectives. Another factor to be considered is the framework or compiler in which polyhedral optimization takes place. Different scenarios, depending on the target architecture, compilation environment, and application domain, may require different kinds of optimization to best exploit the architecture feature set.   We introduce a new configurable polyhedral scheduler, PolyTOPS, that can be adjusted to various scenarios with straightforward, high-level configurations. This scheduler allows the creation of diverse scheduling strategies that can be both scenario-specific (like state-of-the-art schedulers) and kernel-specific, breaking the concept of a one-size-fits-all scheduler approach. PolyTOPS has been used with isl and CLooG as code generators and has been integrated in MindSpore AKG deep learning compiler. Experimental results in different scenarios show good performance: a geomean speedup of 7.66x on MindSpore (for the NPU Ascend architecture) hybrid custom operators over isl scheduling, a geomean speedup up to 1.80x on PolyBench on different multicore architectures over Pluto scheduling. Finally, some comparisons with different state-of-the-art tools are presented in the PolyMage scenario.

{{</citation>}}


## eess.AS (3)



### (104/126) Transcending Controlled Environments Assessing the Transferability of ASRRobust NLU Models to Real-World Applications (Hania Khan et al., 2024)

{{<citation>}}

Hania Khan, Aleena Fatima Khalid, Zaryab Hassan. (2024)  
**Transcending Controlled Environments Assessing the Transferability of ASRRobust NLU Models to Real-World Applications**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-SD, eess-AS, eess.AS  
Keywords: BERT, NLU, Natural Language Understanding, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.09354v1)  

---


**ABSTRACT**  
This research investigates the transferability of Automatic Speech Recognition (ASR)-robust Natural Language Understanding (NLU) models from controlled experimental conditions to practical, real-world applications. Focused on smart home automation commands in Urdu, the study assesses model performance under diverse noise profiles, linguistic variations, and ASR error scenarios. Leveraging the UrduBERT model, the research employs a systematic methodology involving real-world data collection, cross-validation, transfer learning, noise variation studies, and domain adaptation. Evaluation metrics encompass task-specific accuracy, latency, user satisfaction, and robustness to ASR errors. The findings contribute insights into the challenges and adaptability of ASR-robust NLU models in transcending controlled environments.

{{</citation>}}


### (105/126) Dynamic Behaviour of Connectionist Speech Recognition with Strong Latency Constraints (Giampiero Salvi, 2024)

{{<citation>}}

Giampiero Salvi. (2024)  
**Dynamic Behaviour of Connectionist Speech Recognition with Strong Latency Constraints**  

---
Primary Category: eess.AS  
Categories: I-5-0; I-2-7; E-4, cs-AI, cs-CV, cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.06588v1)  

---


**ABSTRACT**  
This paper describes the use of connectionist techniques in phonetic speech recognition with strong latency constraints. The constraints are imposed by the task of deriving the lip movements of a synthetic face in real time from the speech signal, by feeding the phonetic string into an articulatory synthesiser. Particular attention has been paid to analysing the interaction between the time evolution model learnt by the multi-layer perceptrons and the transition model imposed by the Viterbi decoder, in different latency conditions. Two experiments were conducted in which the time dependencies in the language model (LM) were controlled by a parameter. The results show a strong interaction between the three factors involved, namely the neural network topology, the length of time dependencies in the LM and the decoder latency.

{{</citation>}}


### (106/126) Contrastive Learning With Audio Discrimination For Customizable Keyword Spotting In Continuous Speech (Yu Xi et al., 2024)

{{<citation>}}

Yu Xi, Baochen Yang, Hao Li, Jiaqi Guo, Kai Yu. (2024)  
**Contrastive Learning With Audio Discrimination For Customizable Keyword Spotting In Continuous Speech**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.06485v1)  

---


**ABSTRACT**  
Customizable keyword spotting (KWS) in continuous speech has attracted increasing attention due to its real-world application potential. While contrastive learning (CL) has been widely used to extract keyword representations, previous CL approaches all operate on pre-segmented isolated words and employ only audio-text representations matching strategy. However, for KWS in continuous speech, co-articulation and streaming word segmentation can easily yield similar audio patterns for different texts, which may consequently trigger false alarms. To address this issue, we propose a novel CL with Audio Discrimination (CLAD) approach to learning keyword representation with both audio-text matching and audio-audio discrimination ability. Here, an InfoNCE loss considering both audio-audio and audio-text CL data pairs is employed for each sliding window during training. Evaluations on the open-source LibriPhrase dataset show that the use of sliding-window level InfoNCE loss yields comparable performance compared to previous CL approaches. Furthermore, experiments on the continuous speech dataset LibriSpeech demonstrate that, by incorporating audio discrimination, CLAD achieves significant performance gain over CL without audio discrimination. Meanwhile, compared to two-stage KWS approaches, the end-to-end KWS with CLAD achieves not only better performance, but also significant speed-up.

{{</citation>}}


## cs.SI (2)



### (107/126) Exposing Hate -- Understanding Anti-Immigration Sentiment Spreading on Twitter (Andrea Nasuto et al., 2024)

{{<citation>}}

Andrea Nasuto, Francisco Rowe. (2024)  
**Exposing Hate -- Understanding Anti-Immigration Sentiment Spreading on Twitter**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.06658v1)  

---


**ABSTRACT**  
Immigration is one of the most salient topics in public debate. Social media heavily influences opinions on immigration, often sparking polarized debates and offline tensions. Studying 220,870 immigration-related tweets in the UK, we assessed the extent of polarization, key content creators and disseminators, and the speed of content dissemination. We identify a high degree of online polarization between pro and anti-immigration communities. We found that the anti-migration community is small but denser and more active than the pro-immigration community with the top 1% of users responsible for over 23% of anti-immigration tweets and 21% of retweets. We also discovered that anti-immigration content spreads also 1.66 times faster than pro-immigration messages and bots have minimal impact on content dissemination. Our findings suggest that identifying and tracking highly active users could curb anti-immigration sentiment, potentially easing social polarization and shaping broader societal attitudes toward migration.

{{</citation>}}


### (108/126) Cyborgs for strategic communication on social media (Lynnette Hui Xian Ng et al., 2024)

{{<citation>}}

Lynnette Hui Xian Ng, Dawn C. Robertson, Kathleen M. Carley. (2024)  
**Cyborgs for strategic communication on social media**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.06582v1)  

---


**ABSTRACT**  
Social media platforms are a key ground of information consumption and dissemination. Key figures like politicians, celebrities and activists have leveraged on its wide user base for strategic communication. Strategic communications, or StratCom, is the deliberate act of information creation and distribution. Its techniques are used by these key figures for establishing their brand and amplifying their messages. Automated scripts are used on top of personal touches to quickly and effectively perform these tasks. The combination of automation and manual online posting creates a Cyborg social media profile, which is a hybrid between bot and human. In this study, we establish a quantitative definition for a Cyborg account, which is an account that are detected as bots in one time window, and identified as humans in another. This definition makes use of frequent changes of bot classification labels and large differences in bot likelihood scores to identify Cyborgs. We perform a large-scale analysis across over 3.1 million users from Twitter collected from two key events, the 2020 Coronavirus pandemic and 2020 US Elections. We extract Cyborgs from two datasets and employ tools from network science, natural language processing and manual annotation to characterize Cyborg accounts. Our analyses identify Cyborg accounts are mostly constructed for strategic communication uses, have a strong duality in their bot/human classification and are tactically positioned in the social media network, aiding these accounts to promote their desired content. Cyborgs are also discovered to have long online lives, indicating their ability to evade bot detectors, or the graciousness of platforms to allow their operations.

{{</citation>}}


## cs.CR (1)



### (109/126) Accelerating Tactile Internet with QUIC: A Security and Privacy Perspective (Jayasree Sengupta et al., 2024)

{{<citation>}}

Jayasree Sengupta, Debasmita Dey, Simone Ferlin, Nirnay Ghosh, Vaibhav Bajpai. (2024)  
**Accelerating Tactile Internet with QUIC: A Security and Privacy Perspective**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.06657v1)  

---


**ABSTRACT**  
The Tactile Internet paradigm is set to revolutionize human society by enabling skill-set delivery and haptic communication over ultra-reliable, low-latency networks. The emerging sixth-generation (6G) mobile communication systems are envisioned to underpin this Tactile Internet ecosystem at the network edge by providing ubiquitous global connectivity. However, apart from a multitude of opportunities of the Tactile Internet, security and privacy challenges emerge at the forefront. We believe that the recently standardized QUIC protocol, characterized by end-to-end encryption and reduced round-trip delay would serve as the backbone of Tactile Internet. In this article, we envision a futuristic scenario where a QUIC-enabled network uses the underlying 6G communication infrastructure to achieve the requirements for Tactile Internet. Interestingly this requires a deeper investigation of a wide range of security and privacy challenges in QUIC, that need to be mitigated for its adoption in Tactile Internet. Henceforth, this article reviews the existing security and privacy attacks in QUIC and their implication on users. Followed by that, we discuss state-of-the-art attack mitigation strategies and investigate some of their drawbacks with possible directions for future work

{{</citation>}}


## eess.SY (2)



### (110/126) Maximum Causal Entropy Inverse Reinforcement Learning for Mean-Field Games (Berkay Anahtarci et al., 2024)

{{<citation>}}

Berkay Anahtarci, Can Deha Kariksiz, Naci Saldi. (2024)  
**Maximum Causal Entropy Inverse Reinforcement Learning for Mean-Field Games**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.06566v1)  

---


**ABSTRACT**  
In this paper, we introduce the maximum casual entropy Inverse Reinforcement Learning (IRL) problem for discrete-time mean-field games (MFGs) under an infinite-horizon discounted-reward optimality criterion. The state space of a typical agent is finite. Our approach begins with a comprehensive review of the maximum entropy IRL problem concerning deterministic and stochastic Markov decision processes (MDPs) in both finite and infinite-horizon scenarios. Subsequently, we formulate the maximum casual entropy IRL problem for MFGs - a non-convex optimization problem with respect to policies. Leveraging the linear programming formulation of MDPs, we restructure this IRL problem into a convex optimization problem and establish a gradient descent algorithm to compute the optimal solution with a rate of convergence. Finally, we present a new algorithm by formulating the MFG problem as a generalized Nash equilibrium problem (GNEP), which is capable of computing the mean-field equilibrium (MFE) for the forward RL problem. This method is employed to produce data for a numerical example. We note that this novel algorithm is also applicable to general MFE computations.

{{</citation>}}


### (111/126) AI-enabled Priority and Auction-Based Spectrum Management for 6G (Mina Khadem et al., 2024)

{{<citation>}}

Mina Khadem, Farshad Zeinali, Nader Mokari, Hamid Saeedi. (2024)  
**AI-enabled Priority and Auction-Based Spectrum Management for 6G**  

---
Primary Category: eess.SY  
Categories: cs-NI, cs-SY, eess-SP, eess-SY, eess.SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06484v1)  

---


**ABSTRACT**  
In this paper, we present a quality of service (QoS)-aware priority-based spectrum management scheme to guarantee the minimum required bit rate of vertical sector players (VSPs) in the 5G and beyond generation, including the 6th generation (6G). VSPs are considered as spectrum leasers to optimize the overall spectrum efficiency of the network from the perspective of the mobile network operator (MNO) as the spectrum licensee and auctioneer. We exploit a modified Vickrey-Clarke-Groves (VCG) auction mechanism to allocate the spectrum to them where the QoS and the truthfulness of bidders are considered as two important parameters for prioritization of VSPs. The simulation is done with the help of deep deterministic policy gradient (DDPG) as a deep reinforcement learning (DRL)-based algorithm. Simulation results demonstrate that deploying the DDPG algorithm results in significant advantages. In particular, the efficiency of the proposed spectrum management scheme is about %85 compared to the %35 efficiency in traditional auction methods.

{{</citation>}}


## cs.CY (1)



### (112/126) Business and ethical concerns in domestic Conversational Generative AI-empowered multi-robot systems (Rebekah Rousi et al., 2024)

{{<citation>}}

Rebekah Rousi, Hooman Samani, Niko Mäkitalo, Ville Vakkuri, Simo Linkola, Kai-Kristian Kemell, Paulius Daubaris, Ilenia Fronza, Tommi Mikkonen, Pekka Abrahamsson. (2024)  
**Business and ethical concerns in domestic Conversational Generative AI-empowered multi-robot systems**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.09473v1)  

---


**ABSTRACT**  
Business and technology are intricately connected through logic and design. They are equally sensitive to societal changes and may be devastated by scandal. Cooperative multi-robot systems (MRSs) are on the rise, allowing robots of different types and brands to work together in diverse contexts. Generative artificial intelligence has been a dominant topic in recent artificial intelligence (AI) discussions due to its capacity to mimic humans through the use of natural language and the production of media, including deep fakes. In this article, we focus specifically on the conversational aspects of generative AI, and hence use the term Conversational Generative artificial intelligence (CGI). Like MRSs, CGIs have enormous potential for revolutionizing processes across sectors and transforming the way humans conduct business. From a business perspective, cooperative MRSs alone, with potential conflicts of interest, privacy practices, and safety concerns, require ethical examination. MRSs empowered by CGIs demand multi-dimensional and sophisticated methods to uncover imminent ethical pitfalls. This study focuses on ethics in CGI-empowered MRSs while reporting the stages of developing the MORUL model.

{{</citation>}}


## cs.DB (1)



### (113/126) Expected Shapley-Like Scores of Boolean Functions: Complexity and Applications to Probabilistic Databases (Pratik Karmakar et al., 2024)

{{<citation>}}

Pratik Karmakar, Mikaël Monet, Pierre Senellart, Stéphane Bressan. (2024)  
**Expected Shapley-Like Scores of Boolean Functions: Complexity and Applications to Probabilistic Databases**  

---
Primary Category: cs.DB  
Categories: cs-AI, cs-CC, cs-DB, cs.DB  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06493v1)  

---


**ABSTRACT**  
Shapley values, originating in game theory and increasingly prominent in explainable AI, have been proposed to assess the contribution of facts in query answering over databases, along with other similar power indices such as Banzhaf values. In this work we adapt these Shapley-like scores to probabilistic settings, the objective being to compute their expected value. We show that the computations of expected Shapley values and of the expected values of Boolean functions are interreducible in polynomial time, thus obtaining the same tractability landscape. We investigate the specific tractable case where Boolean functions are represented as deterministic decomposable circuits, designing a polynomial-time algorithm for this setting. We present applications to probabilistic databases through database provenance, and an effective implementation of this algorithm within the ProvSQL system, which experimentally validates its feasibility over a standard benchmark.

{{</citation>}}


## math.NA (1)



### (114/126) Cost-optimal adaptive FEM with linearization and algebraic solver for semilinear elliptic PDEs (Maximilian Brunner et al., 2024)

{{<citation>}}

Maximilian Brunner, Dirk Praetorius, Julian Streitberger. (2024)  
**Cost-optimal adaptive FEM with linearization and algebraic solver for semilinear elliptic PDEs**  

---
Primary Category: math.NA  
Categories: cs-NA, math-NA, math.NA  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06486v1)  

---


**ABSTRACT**  
We consider scalar semilinear elliptic PDEs, where the nonlinearity is strongly monotone, but only locally Lipschitz continuous. To linearize the arising discrete nonlinear problem, we employ a damped Zarantonello iteration, which leads to a linear Poisson-type equation that is symmetric and positive definite. The resulting system is solved by a contractive algebraic solver such as a multigrid method with local smoothing. We formulate a fully adaptive algorithm that equibalances the various error components coming from mesh refinement, iterative linearization, and algebraic solver. We prove that the proposed adaptive iteratively linearized finite element method (AILFEM) guarantees convergence with optimal complexity, where the rates are understood with respect to the overall computational cost (i.e., the computational time). Numerical experiments investigate the involved adaptivity parameters.

{{</citation>}}


## cs.AI (3)



### (115/126) Sanity Checks Revisited: An Exploration to Repair the Model Parameter Randomisation Test (Anna Hedström et al., 2024)

{{<citation>}}

Anna Hedström, Leander Weber, Sebastian Lapuschkin, Marina MC Höhne. (2024)  
**Sanity Checks Revisited: An Exploration to Repair the Model Parameter Randomisation Test**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI, stat-ME  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06465v1)  

---


**ABSTRACT**  
The Model Parameter Randomisation Test (MPRT) is widely acknowledged in the eXplainable Artificial Intelligence (XAI) community for its well-motivated evaluative principle: that the explanation function should be sensitive to changes in the parameters of the model function. However, recent works have identified several methodological caveats for the empirical interpretation of MPRT. To address these caveats, we introduce two adaptations to the original MPRT -- Smooth MPRT and Efficient MPRT, where the former minimises the impact that noise has on the evaluation results through sampling and the latter circumvents the need for biased similarity measurements by re-interpreting the test through the explanation's rise in complexity, after full parameter randomisation. Our experimental results demonstrate that these proposed variants lead to improved metric reliability, thus enabling a more trustworthy application of XAI methods.

{{</citation>}}


### (116/126) Vehicle: Bridging the Embedding Gap in the Verification of Neuro-Symbolic Programs (Matthew L. Daggitt et al., 2024)

{{<citation>}}

Matthew L. Daggitt, Wen Kokke, Robert Atkey, Natalia Slusarz, Luca Arnaboldi, Ekaterina Komendantskaya. (2024)  
**Vehicle: Bridging the Embedding Gap in the Verification of Neuro-Symbolic Programs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.06379v1)  

---


**ABSTRACT**  
Neuro-symbolic programs -- programs containing both machine learning components and traditional symbolic code -- are becoming increasingly widespread. However, we believe that there is still a lack of a general methodology for verifying these programs whose correctness depends on the behaviour of the machine learning components. In this paper, we identify the ``embedding gap'' -- the lack of techniques for linking semantically-meaningful ``problem-space'' properties to equivalent ``embedding-space'' properties -- as one of the key issues, and describe Vehicle, a tool designed to facilitate the end-to-end verification of neural-symbolic programs in a modular fashion. Vehicle provides a convenient language for specifying ``problem-space'' properties of neural networks and declaring their relationship to the ``embedding-space", and a powerful compiler that automates interpretation of these properties in the language of a chosen machine-learning training environment, neural network verifier, and interactive theorem prover. We demonstrate Vehicle's utility by using it to formally verify the safety of a simple autonomous car equipped with a neural network controller.

{{</citation>}}


### (117/126) Cognitive BPM as an Equalizer: Improving Access and Efficiency for Employees with (and without) Cognitive Disabilities (Gordon Banks et al., 2024)

{{<citation>}}

Gordon Banks, Gates Bierhuizen, Katherine McCrum, Ellen Wengert. (2024)  
**Cognitive BPM as an Equalizer: Improving Access and Efficiency for Employees with (and without) Cognitive Disabilities**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2401.06375v1)  

---


**ABSTRACT**  
We examine ProcessGPT, an AI model designed to automate, augment, and improve business processes, to study the challenges of managing business processes within the cognitive limitations of the human workforce, particularly individuals with cognitive disabilities. ProcessGPT provides a blueprint for designing efficient business processes that take into account human cognitive limitations. By viewing this through the lens of cognitive disabilities, we show that ProcessGPT improves process usability for individuals with and without cognitive disabilities. We also demonstrate that organizations implementing ProcessGPT-like capabilities will realize increased productivity, morale, and inclusion.

{{</citation>}}


## cs.GR (1)



### (118/126) 3D-PreMise: Can Large Language Models Generate 3D Shapes with Sharp Features and Parametric Control? (Zeqing Yuan et al., 2024)

{{<citation>}}

Zeqing Yuan, Haoxuan Lan, Qiang Zou, Junbo Zhao. (2024)  
**3D-PreMise: Can Large Language Models Generate 3D Shapes with Sharp Features and Parametric Control?**  

---
Primary Category: cs.GR  
Categories: cs-AI, cs-CL, cs-GR, cs.GR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06437v1)  

---


**ABSTRACT**  
Recent advancements in implicit 3D representations and generative models have markedly propelled the field of 3D object generation forward. However, it remains a significant challenge to accurately model geometries with defined sharp features under parametric controls, which is crucial in fields like industrial design and manufacturing. To bridge this gap, we introduce a framework that employs Large Language Models (LLMs) to generate text-driven 3D shapes, manipulating 3D software via program synthesis. We present 3D-PreMise, a dataset specifically tailored for 3D parametric modeling of industrial shapes, designed to explore state-of-the-art LLMs within our proposed pipeline. Our work reveals effective generation strategies and delves into the self-correction capabilities of LLMs using a visual interface. Our work highlights both the potential and limitations of LLMs in 3D parametric modeling for industrial applications.

{{</citation>}}


## cs.IT (1)



### (119/126) Swin Transformer-Based CSI Feedback for Massive MIMO (Jiaming Cheng et al., 2024)

{{<citation>}}

Jiaming Cheng, Wei Chen, Jialong Xu, Yiran Guo, Lun Li, Bo Ai. (2024)  
**Swin Transformer-Based CSI Feedback for Massive MIMO**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.06435v1)  

---


**ABSTRACT**  
For massive multiple-input multiple-output systems in the frequency division duplex (FDD) mode, accurate downlink channel state information (CSI) is required at the base station (BS). However, the increasing number of transmit antennas aggravates the feedback overhead of CSI. Recently, deep learning (DL) has shown considerable potential to reduce CSI feedback overhead. In this paper, we propose a Swin Transformer-based autoencoder network called SwinCFNet for the CSI feedback task. In particular, the proposed method can effectively capture the long-range dependence information of CSI. Moreover, we explore the impact of the number of Swin Transformer blocks and the dimension of feature channels on the performance of SwinCFNet. Experimental results show that SwinCFNet significantly outperforms other DL-based methods with comparable model sizes, especially for the outdoor scenario.

{{</citation>}}


## cs.HC (4)



### (120/126) Why Doesn't Microsoft Let Me Sleep? How Automaticity of Windows Updates Impacts User Autonomy (Sanju Ahuja et al., 2024)

{{<citation>}}

Sanju Ahuja, Ridhi Jain, Jyoti Kumar. (2024)  
**Why Doesn't Microsoft Let Me Sleep? How Automaticity of Windows Updates Impacts User Autonomy**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2401.06413v1)  

---


**ABSTRACT**  
'Automating the user away' has been designated as a dark pattern in literature for performing tasks without user consent or confirmation. However, limited studies have been reported on how users experience the sense of autonomy when digital systems fully or partially bypass consent. More research is required to understand what makes automaticity a threat to autonomy. To address this gap, a qualitative interview study with 10 users was conducted to investigate the user experience of Microsoft Windows updates. It was found that ten design features of Windows updates impact the autonomy experience. For each design feature, the contextual factors which influence its impact on autonomy were also noted. The findings of this paper can help designers understand the ethical concerns posed by automaticity in design and identify measures to mitigate these concerns.

{{</citation>}}


### (121/126) Understanding whole-body inter-personal dynamics between two players using neural Granger causality as the explainable AI (XAI) (Ryota Takamido et al., 2024)

{{<citation>}}

Ryota Takamido, Chiharu Suzuki, Jun Ota, Hiroki Nakamoto. (2024)  
**Understanding whole-body inter-personal dynamics between two players using neural Granger causality as the explainable AI (XAI)**  

---
Primary Category: cs.HC  
Categories: 92-08 (Primary), 92C10 (Secondary), J-4, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06412v1)  

---


**ABSTRACT**  
Background: Simultaneously focusing on intra- and inter-individual body dynamics and elucidating how these affect each other will help understand human inter-personal coordination behavior. However, this association has not been investigated previously owing to difficulties in analyzing complex causal relations among several body components.To address this issue, this study proposes a new analytical framework that attempts to understand the underlying causal structures behind each joint movement of individual baseball players using neural Granger causality (NGC) as the explainable AI. Methods: In the NGC analysis, causal relationships were defined as the size of the weight parameters of the first layer of a machine-learning model trained to predict the future state of a specific time-series variable. To verify the approach in a practical context, we conducted an experiment with 16 pairs of expert baseball pitchers and batters; input datasets with 27 joint resultant velocity data (joints of 13 pitchers and 14 batters) were generated and used for model training.Results: NGC analysis revealed significant causal relations among intra- and inter-individual body components such as the batter's hands having a causal effect from the pitcher's throwing arm. Remarkably, although the causality from the batter's body to pitcher's body is much lower than the reverse, it is significantly correlated with batter performance outcomes. Conclusions: The above results suggest the effectiveness of NGC analysis for understanding whole-body inter-personal coordination dynamics and that of the AI technique as a new approach for analyzing complex human behavior from a different perspective than conventional techniques.

{{</citation>}}


### (122/126) What should I say? -- Interacting with AI and Natural Language Interfaces (Mark Adkins, 2024)

{{<citation>}}

Mark Adkins. (2024)  
**What should I say? -- Interacting with AI and Natural Language Interfaces**  

---
Primary Category: cs.HC  
Categories: I-2-m; J-4; B-4-2, cs-AI, cs-CL, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06382v1)  

---


**ABSTRACT**  
As Artificial Intelligence (AI) technology becomes more and more prevalent, it becomes increasingly important to explore how we as humans interact with AI. The Human-AI Interaction (HAI) sub-field has emerged from the Human-Computer Interaction (HCI) field and aims to examine this very notion. Many interaction patterns have been implemented without fully understanding the changes in required cognition as well as the cognitive science implications of using these alternative interfaces that aim to be more human-like in nature. Prior research suggests that theory of mind representations are crucial to successful and effortless communication, however very little is understood when it comes to how theory of mind representations are established when interacting with AI.

{{</citation>}}


### (123/126) A Temporal-Spectral Fusion Transformer with Subject-specific Adapter for Enhancing RSVP-BCI Decoding (Xujin Li et al., 2024)

{{<citation>}}

Xujin Li, Wei Wei, Shuang Qiu, Huiguang He. (2024)  
**A Temporal-Spectral Fusion Transformer with Subject-specific Adapter for Enhancing RSVP-BCI Decoding**  

---
Primary Category: cs.HC  
Categories: 68T07, I-5-4, cs-AI, cs-HC, cs.HC  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.06340v1)  

---


**ABSTRACT**  
The Rapid Serial Visual Presentation (RSVP)-based Brain-Computer Interface (BCI) is an efficient technology for target retrieval using electroencephalography (EEG) signals. The performance improvement of traditional decoding methods relies on a substantial amount of training data from new test subjects, which increases preparation time for BCI systems. Several studies introduce data from existing subjects to reduce the dependence of performance improvement on data from new subjects, but their optimization strategy based on adversarial learning with extensive data increases training time during the preparation procedure. Moreover, most previous methods only focus on the single-view information of EEG signals, but ignore the information from other views which may further improve performance. To enhance decoding performance while reducing preparation time, we propose a Temporal-Spectral fusion transformer with Subject-specific Adapter (TSformer-SA). Specifically, a cross-view interaction module is proposed to facilitate information transfer and extract common representations across two-view features extracted from EEG temporal signals and spectrogram images. Then, an attention-based fusion module fuses the features of two views to obtain comprehensive discriminative features for classification. Furthermore, a multi-view consistency loss is proposed to maximize the feature similarity between two views of the same EEG signal. Finally, we propose a subject-specific adapter to rapidly transfer the knowledge of the model trained on data from existing subjects to decode data from new subjects. Experimental results show that TSformer-SA significantly outperforms comparison methods and achieves outstanding performance with limited training data from new subjects. This facilitates efficient decoding and rapid deployment of BCI systems in practical use.

{{</citation>}}


## cs.RO (1)



### (124/126) UAV-borne Mapping Algorithms for Canopy-Level and High-Speed Drone Applications (Jincheng Zhang et al., 2024)

{{<citation>}}

Jincheng Zhang, Artur Wolek, Andrew R. Willis. (2024)  
**UAV-borne Mapping Algorithms for Canopy-Level and High-Speed Drone Applications**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2401.06407v1)  

---


**ABSTRACT**  
This article presents a comprehensive review of and analysis of state-of-the-art mapping algorithms for UAV (Unmanned Aerial Vehicle) applications, focusing on canopy-level and high-speed scenarios. This article presents a comprehensive exploration of sensor technologies suitable for UAV mapping, assessing their capabilities to provide measurements that meet the requirements of fast UAV mapping. Furthermore, the study conducts extensive experiments in a simulated environment to evaluate the performance of three distinct mapping algorithms: Direct Sparse Odometry (DSO), Stereo DSO (SDSO), and DSO Lite (DSOL). The experiments delve into mapping accuracy and mapping speed, providing valuable insights into the strengths and limitations of each algorithm. The results highlight the versatility and shortcomings of these algorithms in meeting the demands of modern UAV applications. The findings contribute to a nuanced understanding of UAV mapping dynamics, emphasizing their applicability in complex environments and high-speed scenarios. This research not only serves as a benchmark for mapping algorithm comparisons but also offers practical guidance for selecting sensors tailored to specific UAV mapping applications.

{{</citation>}}


## cs.SD (1)



### (125/126) LCB-net: Long-Context Biasing for Audio-Visual Speech Recognition (Fan Yu et al., 2024)

{{<citation>}}

Fan Yu, Haoxu Wang, Xian Shi, Shiliang Zhang. (2024)  
**LCB-net: Long-Context Biasing for Audio-Visual Speech Recognition**  

---
Primary Category: cs.SD  
Categories: cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Bias, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.06390v1)  

---


**ABSTRACT**  
The growing prevalence of online conferences and courses presents a new challenge in improving automatic speech recognition (ASR) with enriched textual information from video slides. In contrast to rare phrase lists, the slides within videos are synchronized in real-time with the speech, enabling the extraction of long contextual bias. Therefore, we propose a novel long-context biasing network (LCB-net) for audio-visual speech recognition (AVSR) to leverage the long-context information available in videos effectively. Specifically, we adopt a bi-encoder architecture to simultaneously model audio and long-context biasing. Besides, we also propose a biasing prediction module that utilizes binary cross entropy (BCE) loss to explicitly determine biased phrases in the long-context biasing. Furthermore, we introduce a dynamic contextual phrases simulation to enhance the generalization and robustness of our LCB-net. Experiments on the SlideSpeech, a large-scale audio-visual corpus enriched with slides, reveal that our proposed LCB-net outperforms general ASR model by 9.4%/9.1%/10.9% relative WER/U-WER/B-WER reduction on test set, which enjoys high unbiased and biased performance. Moreover, we also evaluate our model on LibriSpeech corpus, leading to 23.8%/19.2%/35.4% relative WER/U-WER/B-WER reduction over the ASR model.

{{</citation>}}


## cs.MM (1)



### (126/126) Generative AI-enabled Mobile Tactical Multimedia Networks: Distribution, Generation, and Perception (Minrui Xu et al., 2024)

{{<citation>}}

Minrui Xu, Dusit Niyato, Jiawen Kang, Zehui Xiong, Song Guo, Yuguang Fang, Dong In Kim. (2024)  
**Generative AI-enabled Mobile Tactical Multimedia Networks: Distribution, Generation, and Perception**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.06386v1)  

---


**ABSTRACT**  
Mobile multimedia networks (MMNs) demonstrate great potential in delivering low-latency and high-quality entertainment and tactical applications, such as short-video sharing, online conferencing, and battlefield surveillance. For instance, in tactical surveillance of battlefields, scalability and sustainability are indispensable for maintaining large-scale military multimedia applications in MMNs. Therefore, many data-driven networking solutions are leveraged to optimize streaming strategies based on real-time traffic analysis and resource monitoring. In addition, generative AI (GAI) can not only increase the efficiency of existing data-driven solutions through data augmentation but also develop potential capabilities for MMNs, including AI-generated content (AIGC) and AI-aided perception. In this article, we propose the framework of GAI-enabled MMNs that leverage the capabilities of GAI in data and content synthesis to distribute high-quality and immersive interactive content in wireless networks. Specifically, we outline the framework of GAI-enabled MMNs and then introduce its three main features, including distribution, generation, and perception. Furthermore, we propose a second-score auction mechanism for allocating network resources by considering GAI model values and other metrics jointly. The experimental results show that the proposed auction mechanism can effectively increase social welfare by allocating resources and models with the highest user satisfaction.

{{</citation>}}
