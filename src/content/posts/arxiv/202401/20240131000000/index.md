---
draft: false
title: "arXiv @ 2024.01.31"
date: 2024-01-31
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.31"
    identifier: arxiv_20240131
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.IT (5)](#csit-5)
- [cs.IR (1)](#csir-1)
- [cs.CL (32)](#cscl-32)
- [cs.CR (5)](#cscr-5)
- [cs.AI (6)](#csai-6)
- [cs.CY (3)](#cscy-3)
- [cs.SI (1)](#cssi-1)
- [cs.DC (2)](#csdc-2)
- [cs.LG (26)](#cslg-26)
- [cs.AR (1)](#csar-1)
- [cs.ET (2)](#cset-2)
- [cs.HC (4)](#cshc-4)
- [cs.CV (25)](#cscv-25)
- [cs.SE (6)](#csse-6)
- [cs.RO (3)](#csro-3)
- [eess.SY (4)](#eesssy-4)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.LO (1)](#cslo-1)
- [q-fin.RM (1)](#q-finrm-1)
- [hep-ex (1)](#hep-ex-1)
- [cs.SD (1)](#cssd-1)
- [cs.NI (1)](#csni-1)

## cs.IT (5)



### (1/132) A New Approach to Harnessing Side Information in Multi-Server Private Information Retrieval (Ningze Wang et al., 2024)

{{<citation>}}

Ningze Wang, Anoosheh Heidarzadeh, Alex Sprintson. (2024)  
**A New Approach to Harnessing Side Information in Multi-Server Private Information Retrieval**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2401.16628v1)  

---


**ABSTRACT**  
This paper presents new solutions for Private Information Retrieval (PIR) with side information. This problem is motivated by PIR settings in which a client has side information about the data held by the servers and would like to leverage this information in order to improve the download rate. The problem of PIR with side information has been the subject of several recent studies that presented achievability schemes as well as converses for both multi-server and single-server settings. However, the solutions for the multi-server settings adapted from the solutions for the single-server setting in a rather straightforward manner, relying on the concept of super-messages. Such solutions require an exponential degree of sub-packetization (in terms of the number of messages).   This paper makes the following contributions. First, we revisit the PIR problem with side information and present a new approach to leverage side information in the context of PIR. The key idea of our approach is a randomized algorithm to determine the linear combinations of the sub-packets that need to be recovered from each server. In addition, our approach takes advantage of the fact that the identity of the side information messages does not need to be kept private, and, as a result, the information retrieval scheme does not need to be symmetric. Second, we present schemes for PIR with side information that achieve a higher rate than previously proposed solutions and require a significantly lower degree of sub-packetization (linear in the number of servers). Our scheme not only achieves the highest known download rate for the problem at hand but also invalidates a previously claimed converse bound on the maximum achievable download rate.

{{</citation>}}


### (2/132) Graph Neural Network-based Joint Equalization and Decoding (Jannis Clausius et al., 2024)

{{<citation>}}

Jannis Clausius, Marvin Geiselhart, Daniel Tandler, Stephan ten Brink. (2024)  
**Graph Neural Network-based Joint Equalization and Decoding**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2401.16187v1)  

---


**ABSTRACT**  
This paper proposes to use graph neural networks (GNNs) for equalization, that can also be used to perform joint equalization and decoding (JED). For equalization, the GNN is build upon the factor graph representations of the channel, while for JED, the factor graph is expanded by the Tanner graph of the parity-check matrix (PCM) of the channel code, sharing the variable nodes (VNs). A particularly advantageous property of the GNN is the robustness against cycles in the factor graphs which is the main problem for belief propagation (BP)-based equalization. As a result of having a fully deep learning-based receiver, joint optimization instead of individual optimization of the components is enabled, so-called end-to-end learning. Furthermore, we propose a parallel flooding schedule that further reduces the latency, which turns out to improve also the error correcting performance. The proposed approach is analyzed and compared to state-of-the-art baselines in terms of error correcting capability and latency. At a fixed low latency, the flooding GNN for JED demonstrates a gain of 2.25 dB in bit error rate (BER) compared to an iterative Bahl--Cock--Jelinek--Raviv (BCJR)-BP baseline.

{{</citation>}}


### (3/132) Unrestricted Error-Type Codebook Generation for Error Correction Code in DNA Storage Inspired by NLP (Yi Lu et al., 2024)

{{<citation>}}

Yi Lu, Yun Ma, Chenghao Li, Xin Zhang, Guangxiang Si. (2024)  
**Unrestricted Error-Type Codebook Generation for Error Correction Code in DNA Storage Inspired by NLP**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2401.15915v2)  

---


**ABSTRACT**  
Recently, DNA storage has surfaced as a promising alternative for data storage, presenting notable benefits in terms of storage capacity, cost-effectiveness in maintenance, and the capability for parallel replication. Mathematically, the DNA storage process can be conceptualized as an insertion, deletion, and substitution (IDS) channel. Due to the mathematical complexity associated with the Levenshtein distance, creating a code that corrects for IDS remains a challenging task. In this paper, we propose a bottom-up generation approach to grow the required codebook based on the computation of Edit Computational Graph (ECG) which differs from the algebraic constructions by incorporating the Derivative-Free Optimization (DFO) method. Specifically, this approach is regardless of the type of errors. Compared the results with the work for 1-substitution-1-deletion and 2-deletion, the redundancy is reduced by about 30-bit and 60-bit, respectively. As far as we know, our method is the first IDS-correcting code designed using classical Natural Language Process (NLP) techniques, marking a turning point in the field of error correction code research. Based on the codebook generated by our method, there may be significant breakthroughs in the complexity of encoding and decoding algorithms.

{{</citation>}}


### (4/132) An Efficient, High-Rate Scheme for Private Information Retrieval over the Gaussian MAC (Or Elimelech et al., 2024)

{{<citation>}}

Or Elimelech, Asaf Cohen. (2024)  
**An Efficient, High-Rate Scheme for Private Information Retrieval over the Gaussian MAC**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2401.15912v2)  

---


**ABSTRACT**  
This paper addresses the challenge of the private information retrieval (PIR) problem wherein there are $N$ replicated non-communicating databases containing the same $M$ messages and a user who wants to retrieve one of the messages without revealing the wanted message's index to the databases. In addition, we assume a block-fading additive white Gaussian noise multiple access channel (AWGN MAC) linking the user and the databases. Shmuel's contribution \cite{shmuel2021private}, presenting a joint channel-PIR scheme utilizing the C\&F protocol, has shown the potential of a joint channel-PIR scheme over a separated scheme. In this paper, we propose an improved joint channel-PIR approach tailored for the PIR problem with $N$ databases over a block-fading AWGN. Unlike the C\&F protocol, our scheme offers reduced computational complexity while improving the scaling laws governing the achievable rate. Our achievable rate scales with the number of databases $N$ and the power $P$ similarly to the channel capacity without the privacy constraint and outperforms the C\&F-based approach. Furthermore, our analysis demonstrates that our improved rate exhibits only a finite gap from the channel capacity of one bit as $N$ increases.

{{</citation>}}


### (5/132) Correction to 'Private Information Retrieval Over Gaussian MAC' (Or Elimelech et al., 2024)

{{<citation>}}

Or Elimelech, Ori Shmuel, Asaf Cohen. (2024)  
**Correction to 'Private Information Retrieval Over Gaussian MAC'**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2401.15910v1)  

---


**ABSTRACT**  
In the above article \cite{shmuel2021private}, the authors introduced a PIR scheme for the Additive White Gaussian Noise (AWGN) Multiple Access Channel (MAC), both with and without fading. The authors utilized the additive nature of the channel and leveraged the linear properties and structure of lattice codes to retrieve the desired message without the servers acquiring any knowledge on the retrieved message's index.   Theorems 3 and 4 in \cite{shmuel2021private} contain an error arising from the incorrect usage of the modulo operator. Moreover, the proofs assume a one-to-one mapping function, $\phi(\cdot)$, between a message $W_j\in\mathbb{F}_p^L$ and the elements of $\mathcal{C}$, mistakenly suggesting that the user possesses all the required information in advance. However, this is not the case. Herein, we present the corrected versions of these theorems.

{{</citation>}}


## cs.IR (1)



### (6/132) FakeClaim: A Multiple Platform-driven Dataset for Identification of Fake News on 2023 Israel-Hamas War (Gautam Kishore Shahi et al., 2024)

{{<citation>}}

Gautam Kishore Shahi, Amit Kumar Jaiswal, Thomas Mandl. (2024)  
**FakeClaim: A Multiple Platform-driven Dataset for Identification of Fake News on 2023 Israel-Hamas War**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-SI, cs.IR  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2401.16625v1)  

---


**ABSTRACT**  
We contribute the first publicly available dataset of factual claims from different platforms and fake YouTube videos on the 2023 Israel-Hamas war for automatic fake YouTube video classification. The FakeClaim data is collected from 60 fact-checking organizations in 30 languages and enriched with metadata from the fact-checking organizations curated by trained journalists specialized in fact-checking. Further, we classify fake videos within the subset of YouTube videos using textual information and user comments. We used a pre-trained model to classify each video with different feature combinations. Our best-performing fine-tuned language model, Universal Sentence Encoder (USE), achieves a Macro F1 of 87\%, which shows that the trained model can be helpful for debunking fake videos using the comments from the user discussion. The dataset is available on Github\footnote{https://github.com/Gautamshahi/FakeClaim}

{{</citation>}}


## cs.CL (32)



### (7/132) ToPro: Token-Level Prompt Decomposition for Cross-Lingual Sequence Labeling Tasks (Bolei Ma et al., 2024)

{{<citation>}}

Bolei Ma, Ercong Nie, Shuzhou Yuan, Helmut Schmid, Michael Färber, Frauke Kreuter, Hinrich Schütze. (2024)  
**ToPro: Token-Level Prompt Decomposition for Cross-Lingual Sequence Labeling Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, Named Entity Recognition, T5  
[Paper Link](http://arxiv.org/abs/2401.16589v1)  

---


**ABSTRACT**  
Prompt-based methods have been successfully applied to multilingual pretrained language models for zero-shot cross-lingual understanding. However, most previous studies primarily focused on sentence-level classification tasks, and only a few considered token-level labeling tasks such as Named Entity Recognition (NER) and Part-of-Speech (POS) tagging. In this paper, we propose Token-Level Prompt Decomposition (ToPro), which facilitates the prompt-based method for token-level sequence labeling tasks. The ToPro method decomposes an input sentence into single tokens and applies one prompt template to each token. Our experiments on multilingual NER and POS tagging datasets demonstrate that ToPro-based fine-tuning outperforms Vanilla fine-tuning and Prompt-Tuning in zero-shot cross-lingual transfer, especially for languages that are typologically different from the source language English. Our method also attains state-of-the-art performance when employed with the mT5 model. Besides, our exploratory study in multilingual large language models shows that ToPro performs much better than the current in-context learning method. Overall, the performance improvements show that ToPro could potentially serve as a novel and simple benchmarking method for sequence labeling tasks.

{{</citation>}}


### (8/132) A Linguistic Comparison between Human and ChatGPT-Generated Conversations (Morgan Sandler et al., 2024)

{{<citation>}}

Morgan Sandler, Hyesun Choung, Arun Ross, Prabu David. (2024)  
**A Linguistic Comparison between Human and ChatGPT-Generated Conversations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: AI, ChatGPT, Dialog, Dialogue, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2401.16587v1)  

---


**ABSTRACT**  
This study explores linguistic differences between human and LLM-generated dialogues, using 19.5K dialogues generated by ChatGPT-3.5 as a companion to the EmpathicDialogues dataset. The research employs Linguistic Inquiry and Word Count (LIWC) analysis, comparing ChatGPT-generated conversations with human conversations across 118 linguistic categories. Results show greater variability and authenticity in human dialogues, but ChatGPT excels in categories such as social processes, analytical style, cognition, attentional focus, and positive emotional tone, reinforcing recent findings of LLMs being "more human than human." However, no significant difference was found in positive or negative affect between ChatGPT and human dialogues. Classifier analysis of dialogue embeddings indicates implicit coding of the valence of affect despite no explicit mention of affect in the conversations. The research also contributes a novel, companion ChatGPT-generated dataset of conversations between two independent chatbots, which were designed to replicate a corpus of human conversations available for open access and used widely in AI research on language modeling. Our findings increase understanding of ChatGPT's linguistic capabilities and inform ongoing efforts to distinguish between human and LLM-generated text, which is critical in detecting AI-generated fakes, misinformation, and disinformation.

{{</citation>}}


### (9/132) Massively Multilingual Text Translation For Low-Resource Languages (Zhong Zhou, 2024)

{{<citation>}}

Zhong Zhou. (2024)  
**Massively Multilingual Text Translation For Low-Resource Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Low-Resource, Multilingual  
[Paper Link](http://arxiv.org/abs/2401.16582v1)  

---


**ABSTRACT**  
Translation into severely low-resource languages has both the cultural goal of saving and reviving those languages and the humanitarian goal of assisting the everyday needs of local communities that are accelerated by the recent COVID-19 pandemic. In many humanitarian efforts, translation into severely low-resource languages often does not require a universal translation engine, but a dedicated text-specific translation engine. For example, healthcare records, hygienic procedures, government communication, emergency procedures and religious texts are all limited texts. While generic translation engines for all languages do not exist, translation of multilingually known limited texts into new, low-resource languages may be possible and reduce human translation effort. We attempt to leverage translation resources from rich-resource languages to efficiently produce best possible translation quality for well known texts, which are available in multiple languages, in a new, low-resource language. To reach this goal, we argue that in translating a closed text into low-resource languages, generalization to out-of-domain texts is not necessary, but generalization to new languages is. Performance gain comes from massive source parallelism by careful choice of close-by language families, style-consistent corpus-level paraphrases within the same language and strategic adaptation of existing large pretrained multilingual models to the domain first and then to the language. Such performance gain makes it possible for machine translation systems to collaborate with human translators to expedite the translation process into new, low-resource languages.

{{</citation>}}


### (10/132) Leveraging Professional Radiologists' Expertise to Enhance LLMs' Evaluation for Radiology Reports (Qingqing Zhu et al., 2024)

{{<citation>}}

Qingqing Zhu, Xiuying Chen, Qiao Jin, Benjamin Hou, Tejas Sudharshan Mathai, Pritam Mukherjee, Xin Gao, Ronald M Summers, Zhiyong Lu. (2024)  
**Leveraging Professional Radiologists' Expertise to Enhance LLMs' Evaluation for Radiology Reports**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Clinical, GPT, GPT-3.5, GPT-4, Language Model, Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2401.16578v1)  

---


**ABSTRACT**  
In radiology, Artificial Intelligence (AI) has significantly advanced report generation, but automatic evaluation of these AI-produced reports remains challenging. Current metrics, such as Conventional Natural Language Generation (NLG) and Clinical Efficacy (CE), often fall short in capturing the semantic intricacies of clinical contexts or overemphasize clinical details, undermining report clarity. To overcome these issues, our proposed method synergizes the expertise of professional radiologists with Large Language Models (LLMs), like GPT-3.5 and GPT-4 1. Utilizing In-Context Instruction Learning (ICIL) and Chain of Thought (CoT) reasoning, our approach aligns LLM evaluations with radiologist standards, enabling detailed comparisons between human and AI generated reports. This is further enhanced by a Regression model that aggregates sentence evaluation scores. Experimental results show that our ''Detailed GPT-4 (5-shot)'' model achieves a 0.48 score, outperforming the METEOR metric by 0.19, while our ''Regressed GPT-4'' model shows even greater alignment with expert evaluations, exceeding the best existing metric by a 0.35 margin. Moreover, the robustness of our explanations has been validated through a thorough iterative strategy. We plan to publicly release annotations from radiology experts, setting a new standard for accuracy in future assessments. This underscores the potential of our approach in enhancing the quality assessment of AI-driven medical reports.

{{</citation>}}


### (11/132) LLMs as On-demand Customizable Service (Souvika Sarkar et al., 2024)

{{<citation>}}

Souvika Sarkar, Mohammad Fakhruddin Babar, Monowar Hasan, Shubhra Kanti Karmaker. (2024)  
**LLMs as On-demand Customizable Service**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.16577v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable language understanding and generation capabilities. However, training, deploying, and accessing these models pose notable challenges, including resource-intensive demands, extended training durations, and scalability issues. To address these issues, we introduce a concept of hierarchical, distributed LLM architecture that aims at enhancing the accessibility and deployability of LLMs across heterogeneous computing platforms, including general-purpose computers (e.g., laptops) and IoT-style devices (e.g., embedded systems). By introducing a "layered" approach, the proposed architecture enables on-demand accessibility to LLMs as a customizable service. This approach also ensures optimal trade-offs between the available computational resources and the user's application needs. We envision that the concept of hierarchical LLM will empower extensive, crowd-sourced user bases to harness the capabilities of LLMs, thereby fostering advancements in AI technology in general.

{{</citation>}}


### (12/132) Beyond Image-Text Matching: Verb Understanding in Multimodal Transformers Using Guided Masking (Ivana Beňová et al., 2024)

{{<citation>}}

Ivana Beňová, Jana Košecká, Michal Gregor, Martin Tamajka, Marcel Veselý, Marián Šimko. (2024)  
**Beyond Image-Text Matching: Verb Understanding in Multimodal Transformers Using Guided Masking**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.16575v1)  

---


**ABSTRACT**  
The dominant probing approaches rely on the zero-shot performance of image-text matching tasks to gain a finer-grained understanding of the representations learned by recent multimodal image-language transformer models. The evaluation is carried out on carefully curated datasets focusing on counting, relations, attributes, and others. This work introduces an alternative probing strategy called guided masking. The proposed approach ablates different modalities using masking and assesses the model's ability to predict the masked word with high accuracy. We focus on studying multimodal models that consider regions of interest (ROI) features obtained by object detectors as input tokens. We probe the understanding of verbs using guided masking on ViLBERT, LXMERT, UNITER, and VisualBERT and show that these models can predict the correct verb with high accuracy. This contrasts with previous conclusions drawn from image-text matching probing techniques that frequently fail in situations requiring verb understanding. The code for all experiments will be publicly available https://github.com/ivana-13/guided_masking.

{{</citation>}}


### (13/132) Multi-class Regret Detection in Hindi Devanagari Script (Renuka Sharma et al., 2024)

{{<citation>}}

Renuka Sharma, Sushama Nagpal, Sangeeta Sabharwal, Sabur Butt. (2024)  
**Multi-class Regret Detection in Hindi Devanagari Script**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.16561v1)  

---


**ABSTRACT**  
The number of Hindi speakers on social media has increased dramatically in recent years. Regret is a common emotional experience in our everyday life. Many speakers on social media, share their regretful experiences and opinions regularly. It might cause a re-evaluation of one's choices and a desire to make a different option if given the chance. As a result, knowing the source of regret is critical for investigating its impact on behavior and decision-making. This study focuses on regret and how it is expressed, specifically in Hindi, on various social media platforms. In our study, we present a novel dataset from three different sources, where each sentence has been manually classified into one of three classes "Regret by action", "Regret by inaction", and "No regret". Next, we use this dataset to investigate the linguistic expressions of regret in Hindi text and also identify the textual domains that are most frequently associated with regret. Our findings indicate that individuals on social media platforms frequently express regret for both past inactions and actions, particularly within the domain of interpersonal relationships. We use a pre-trained BERT model to generate word embeddings for the Hindi dataset and also compare deep learning models with conventional machine learning models in order to demonstrate accuracy. Our results show that BERT embedding with CNN consistently surpassed other models. This described the effectiveness of BERT for conveying the context and meaning of words in the regret domain.

{{</citation>}}


### (14/132) InfoLossQA: Characterizing and Recovering Information Loss in Text Simplification (Jan Trienes et al., 2024)

{{<citation>}}

Jan Trienes, Sebastian Joseph, Jörg Schlötterer, Christin Seifert, Kyle Lo, Wei Xu, Byron C. Wallace, Junyi Jessy Li. (2024)  
**InfoLossQA: Characterizing and Recovering Information Loss in Text Simplification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.16475v1)  

---


**ABSTRACT**  
Text simplification aims to make technical texts more accessible to laypeople but often results in deletion of information and vagueness. This work proposes InfoLossQA, a framework to characterize and recover simplification-induced information loss in form of question-and-answer (QA) pairs. Building on the theory of Question Under Discussion, the QA pairs are designed to help readers deepen their knowledge of a text. We conduct a range of experiments with this framework. First, we collect a dataset of 1,000 linguist-curated QA pairs derived from 104 LLM simplifications of scientific abstracts of medical studies. Our analyses of this data reveal that information loss occurs frequently, and that the QA pairs give a high-level overview of what information was lost. Second, we devise two methods for this task: end-to-end prompting of open-source and commercial language models, and a natural language inference pipeline. With a novel evaluation framework considering the correctness of QA pairs and their linguistic suitability, our expert evaluation reveals that models struggle to reliably identify information loss and applying similar standards as humans at what constitutes information loss.

{{</citation>}}


### (15/132) Scaling Sparse Fine-Tuning to Large Language Models (Alan Ansell et al., 2024)

{{<citation>}}

Alan Ansell, Ivan Vulić, Hannah Sterz, Anna Korhonen, Edoardo M. Ponti. (2024)  
**Scaling Sparse Fine-Tuning to Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2401.16405v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are difficult to fully fine-tune (e.g., with instructions or human feedback) due to their sheer number of parameters. A family of parameter-efficient sparse fine-tuning (SFT) methods have proven promising in terms of performance but their memory requirements increase proportionally to the size of the LLMs. In this work, we scale sparse fine-tuning to state-of-the-art LLMs like LLaMA 2 7B and 13B. At any given time, for a desired density level, we maintain an array of parameter indices and the deltas of these parameters relative to their pretrained values. We iterate among: (a) updating the active deltas, (b) pruning indices (based on the change of magnitude of their deltas) and (c) regrowth of indices. For regrowth, we explore two criteria based on either the accumulated gradients of a few candidate parameters or their approximate momenta estimated using the efficient SM3 optimizer. We experiment with instruction-tuning of LLMs on standard dataset mixtures, finding that SFT is often superior to popular parameter-efficient fine-tuning methods like LoRA (low-rank adaptation) in terms of performance and comparable in terms of run time. We additionally show that SFT is compatible with both quantization and efficient optimizers, to facilitate scaling to ever-larger model sizes. We release the code for SFT at https://github.com/AlanAnsell/peft and for the instruction-tuning experiments at https://github.com/ducdauge/sft-llm.

{{</citation>}}


### (16/132) ViLexNorm: A Lexical Normalization Corpus for Vietnamese Social Media Text (Thanh-Nhi Nguyen et al., 2024)

{{<citation>}}

Thanh-Nhi Nguyen, Thanh-Phong Le, Kiet Van Nguyen. (2024)  
**ViLexNorm: A Lexical Normalization Corpus for Vietnamese Social Media Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, NLP, Natural Language Processing, Social Media  
[Paper Link](http://arxiv.org/abs/2401.16403v2)  

---


**ABSTRACT**  
Lexical normalization, a fundamental task in Natural Language Processing (NLP), involves the transformation of words into their canonical forms. This process has been proven to benefit various downstream NLP tasks greatly. In this work, we introduce Vietnamese Lexical Normalization (ViLexNorm), the first-ever corpus developed for the Vietnamese lexical normalization task. The corpus comprises over 10,000 pairs of sentences meticulously annotated by human annotators, sourced from public comments on Vietnam's most popular social media platforms. Various methods were used to evaluate our corpus, and the best-performing system achieved a result of 57.74% using the Error Reduction Rate (ERR) metric (van der Goot, 2019a) with the Leave-As-Is (LAI) baseline. For extrinsic evaluation, employing the model trained on ViLexNorm demonstrates the positive impact of the Vietnamese lexical normalization task on other NLP tasks. Our corpus is publicly available exclusively for research purposes.

{{</citation>}}


### (17/132) Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling (Pratyush Maini et al., 2024)

{{<citation>}}

Pratyush Maini, Skyler Seto, He Bai, David Grangier, Yizhe Zhang, Navdeep Jaitly. (2024)  
**Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.16380v1)  

---


**ABSTRACT**  
Large language models are trained on massive scrapes of the web, which are often unstructured, noisy, and poorly phrased. Current scaling laws show that learning from such data requires an abundance of both compute and data, which grows with the size of the model being trained. This is infeasible both because of the large compute costs and duration associated with pre-training, and the impending scarcity of high-quality data on the web. In this work, we propose Web Rephrase Augmented Pre-training ($\textbf{WRAP}$) that uses an off-the-shelf instruction-tuned model prompted to paraphrase documents on the web in specific styles such as "like Wikipedia" or in "question-answer format" to jointly pre-train LLMs on real and synthetic rephrases. First, we show that using WRAP on the C4 dataset, which is naturally noisy, speeds up pre-training by $\sim3x$. At the same pre-training compute budget, it improves perplexity by more than 10% on average across different subsets of the Pile, and improves zero-shot question answer accuracy across 13 tasks by more than 2%. Second, we investigate the impact of the re-phrasing style on the performance of the model, offering insights into how the composition of the training data can impact the performance of LLMs in OOD settings. Our gains are attributed to the fact that re-phrased synthetic data has higher utility than just real data because it (i) incorporates style diversity that closely reflects downstream evaluation style, and (ii) has higher 'quality' than web-scraped data.

{{</citation>}}


### (18/132) ConFit: Improving Resume-Job Matching using Data Augmentation and Contrastive Learning (Xiao Yu et al., 2024)

{{<citation>}}

Xiao Yu, Jinzhong Zhang, Zhou Yu. (2024)  
**ConFit: Improving Resume-Job Matching using Data Augmentation and Contrastive Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: AI, Augmentation, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.16349v1)  

---


**ABSTRACT**  
A reliable resume-job matching system helps a company find suitable candidates from a pool of resumes, and helps a job seeker find relevant jobs from a list of job posts. However, since job seekers apply only to a few jobs, interaction records in resume-job datasets are sparse. Different from many prior work that use complex modeling techniques, we tackle this sparsity problem using data augmentations and a simple contrastive learning approach. ConFit first creates an augmented resume-job dataset by paraphrasing specific sections in a resume or a job post. Then, ConFit uses contrastive learning to further increase training samples from $B$ pairs per batch to $O(B^2)$ per batch. We evaluate ConFit on two real-world datasets and find it outperforms prior methods (including BM25 and OpenAI text-ada-002) by up to 19% and 31% absolute in nDCG@10 for ranking jobs and ranking resumes, respectively.

{{</citation>}}


### (19/132) Beyond Automated Evaluation Metrics: Evaluating Topic Models On Practical Social Science Content Analysis Tasks (Zongxia Li et al., 2024)

{{<citation>}}

Zongxia Li, Andrew Mao, Daniel Stephens, Pranav Goel, Emily Walpole, Alden Dima, Juan Fung, Jordan Boyd-Graber. (2024)  
**Beyond Automated Evaluation Metrics: Evaluating Topic Models On Practical Social Science Content Analysis Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-HC, cs.CL  
Keywords: Topic Model  
[Paper Link](http://arxiv.org/abs/2401.16348v1)  

---


**ABSTRACT**  
Topic models are a popular tool for understanding text collections, but their evaluation has been a point of contention. Automated evaluation metrics such as coherence are often used, however, their validity has been questioned for neural topic models (NTMs) and can overlook the benefits of a model in real world applications. To this end, we conduct the first evaluation of neural, supervised and classical topic models in an interactive task based setting. We combine topic models with a classifier and test their ability to help humans conduct content analysis and document annotation. From simulated, real user and expert pilot studies, the Contextual Neural Topic Model does the best on cluster evaluation metrics and human evaluations; however, LDA is competitive with two other NTMs under our simulated experiment and user study results, contrary to what coherence scores suggest. We show that current automated metrics do not provide a complete picture of topic modeling capabilities, but the right choice of NTMs can be better than classical models on practical tasks.

{{</citation>}}


### (20/132) Tradeoffs Between Alignment and Helpfulness in Language Models (Yotam Wolf et al., 2024)

{{<citation>}}

Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin Rothberg, Yoav Levine, Amnon Shashua. (2024)  
**Tradeoffs Between Alignment and Helpfulness in Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.16332v1)  

---


**ABSTRACT**  
Language model alignment has become an important component of AI safety, allowing safe interactions between humans and language models, by enhancing desired behaviors and inhibiting undesired ones. It is often done by tuning the model or inserting preset aligning prompts. Recently, representation engineering, a method which alters the model's behavior via changing its representations post-training, was shown to be effective in aligning LLMs (Zou et al., 2023a). Representation engineering yields gains in alignment oriented tasks such as resistance to adversarial attacks and reduction of social biases, but was also shown to cause a decrease in the ability of the model to perform basic tasks. In this paper we study the tradeoff between the increase in alignment and decrease in helpfulness of the model. We propose a theoretical framework which provides bounds for these two quantities, and demonstrate their relevance empirically. Interestingly, we find that while the helpfulness generally decreases, it does so quadratically with the norm of the representation engineering vector, while the alignment increases linearly with it, indicating a regime in which it is efficient to use representation engineering. We validate our findings empirically, and chart the boundaries to the usefulness of representation engineering for alignment.

{{</citation>}}


### (21/132) Machine Translation Meta Evaluation through Translation Accuracy Challenge Sets (Nikita Moghe et al., 2024)

{{<citation>}}

Nikita Moghe, Arnisa Fazla, Chantal Amrhein, Tom Kocmi, Mark Steedman, Alexandra Birch, Rico Sennrich, Liane Guillou. (2024)  
**Machine Translation Meta Evaluation through Translation Accuracy Challenge Sets**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.16313v1)  

---


**ABSTRACT**  
Recent machine translation (MT) metrics calibrate their effectiveness by correlating with human judgement but without any insights about their behaviour across different error types. Challenge sets are used to probe specific dimensions of metric behaviour but there are very few such datasets and they either focus on a limited number of phenomena or a limited number of language pairs. We introduce ACES, a contrastive challenge set spanning 146 language pairs, aimed at discovering whether metrics can identify 68 translation accuracy errors. These phenomena range from simple alterations at the word/character level to more complex errors based on discourse and real-world knowledge. We conduct a large-scale study by benchmarking ACES on 50 metrics submitted to the WMT 2022 and 2023 metrics shared tasks. We benchmark metric performance, assess their incremental performance over successive campaigns, and measure their sensitivity to a range of linguistic phenomena. We also investigate claims that Large Language Models (LLMs) are effective as MT evaluators by evaluating on ACES. Our results demonstrate that different metric families struggle with different phenomena and that LLM-based methods fail to demonstrate reliable performance. Our analyses indicate that most metrics ignore the source sentence, tend to prefer surface-level overlap and end up incorporating properties of base models which are not always beneficial. We expand ACES to include error span annotations, denoted as SPAN-ACES and we use this dataset to evaluate span-based error metrics showing these metrics also need considerable improvement. Finally, we provide a set of recommendations for building better MT metrics, including focusing on error labels instead of scores, ensembling, designing strategies to explicitly focus on the source sentence, focusing on semantic content and choosing the right base model for representations.

{{</citation>}}


### (22/132) Textual Entailment for Effective Triple Validation in Object Prediction (Andrés García-Silva et al., 2024)

{{<citation>}}

Andrés García-Silva, Cristian Berrío, José Manuel Gómez-Pérez. (2024)  
**Textual Entailment for Effective Triple Validation in Object Prediction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-DL, cs.CL  
Keywords: Textual Entailment  
[Paper Link](http://arxiv.org/abs/2401.16293v1)  

---


**ABSTRACT**  
Knowledge base population seeks to expand knowledge graphs with facts that are typically extracted from a text corpus. Recently, language models pretrained on large corpora have been shown to contain factual knowledge that can be retrieved using cloze-style strategies. Such approach enables zero-shot recall of facts, showing competitive results in object prediction compared to supervised baselines. However, prompt-based fact retrieval can be brittle and heavily depend on the prompts and context used, which may produce results that are unintended or hallucinatory.We propose to use textual entailment to validate facts extracted from language models through cloze statements. Our results show that triple validation based on textual entailment improves language model predictions in different training regimes. Furthermore, we show that entailment-based triple validation is also effective to validate candidate facts extracted from other sources including existing knowledge graphs and text passages where named entities are recognized.

{{</citation>}}


### (23/132) MAPLE: Micro Analysis of Pairwise Language Evolution for Few-Shot Claim Verification (Xia Zeng et al., 2024)

{{<citation>}}

Xia Zeng, Arkaitz Zubiaga. (2024)  
**MAPLE: Micro Analysis of Pairwise Language Evolution for Few-Shot Claim Verification**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Few-Shot, LLaMA  
[Paper Link](http://arxiv.org/abs/2401.16282v1)  

---


**ABSTRACT**  
Claim verification is an essential step in the automated fact-checking pipeline which assesses the veracity of a claim against a piece of evidence. In this work, we explore the potential of few-shot claim verification, where only very limited data is available for supervision. We propose MAPLE (Micro Analysis of Pairwise Language Evolution), a pioneering approach that explores the alignment between a claim and its evidence with a small seq2seq model and a novel semantic measure. Its innovative utilization of micro language evolution path leverages unlabelled pairwise data to facilitate claim verification while imposing low demand on data annotations and computing resources. MAPLE demonstrates significant performance improvements over SOTA baselines SEED, PET and LLaMA 2 across three fact-checking datasets: FEVER, Climate FEVER, and SciFact. Data and code are available here: https://github.com/XiaZeng0223/MAPLE

{{</citation>}}


### (24/132) Towards Red Teaming in Multimodal and Multilingual Translation (Christophe Ropers et al., 2024)

{{<citation>}}

Christophe Ropers, David Dale, Prangthip Hansanti, Gabriel Mejia Gonzalez, Ivan Evtimov, Corinne Wong, Christophe Touret, Kristina Pereyra, Seohyun Sonia Kim, Cristian Canton Ferrer, Pierre Andrews, Marta R. Costa-jussà. (2024)  
**Towards Red Teaming in Multimodal and Multilingual Translation**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs-CY, cs.CL  
Keywords: AI, Machine Translation, Multilingual, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.16247v1)  

---


**ABSTRACT**  
Assessing performance in Natural Language Processing is becoming increasingly complex. One particular challenge is the potential for evaluation datasets to overlap with training data, either directly or indirectly, which can lead to skewed results and overestimation of model performance. As a consequence, human evaluation is gaining increasing interest as a means to assess the performance and reliability of models. One such method is the red teaming approach, which aims to generate edge cases where a model will produce critical errors. While this methodology is becoming standard practice for generative AI, its application to the realm of conditional AI remains largely unexplored. This paper presents the first study on human-based red teaming for Machine Translation (MT), marking a significant step towards understanding and improving the performance of translation models. We delve into both human-based red teaming and a study on automation, reporting lessons learned and providing recommendations for both translation models and red teaming drills. This pioneering work opens up new avenues for research and development in the field of MT.

{{</citation>}}


### (25/132) Clinically meaningful timeline summarisation in social media for mental health monitoring (Jiayu Song et al., 2024)

{{<citation>}}

Jiayu Song, Jenny Chim, Adam Tsakalidis, Julia Ive, Dana Atzil-Slonim, Maria Liakata. (2024)  
**Clinically meaningful timeline summarisation in social media for mental health monitoring**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Clinical, LLaMA  
[Paper Link](http://arxiv.org/abs/2401.16240v1)  

---


**ABSTRACT**  
We introduce the new task of clinically meaningful summarisation of social media user timelines, appropriate for mental health monitoring. We develop a novel approach for unsupervised abstractive summarisation that produces a two-layer summary consisting of both high-level information, covering aspects useful to clinical experts, as well as accompanying time sensitive evidence from a user's social media timeline. A key methodological novelty comes from the timeline summarisation component based on a version of hierarchical variational autoencoder (VAE) adapted to represent long texts and guided by LLM-annotated key phrases. The resulting timeline summary is input into a LLM (LLaMA-2) to produce the final summary containing both the high level information, obtained through instruction prompting, as well as corresponding evidence from the user's timeline. We assess the summaries generated by our novel architecture via automatic evaluation against expert written summaries and via human evaluation with clinical experts, showing that timeline summarisation by TH-VAE results in logically coherent summaries rich in clinical utility and superior to LLM-only approaches in capturing changes over time.

{{</citation>}}


### (26/132) MultiMUC: Multilingual Template Filling on MUC-4 (William Gantt et al., 2024)

{{<citation>}}

William Gantt, Shabnam Behzad, Hannah YoungEun An, Yunmo Chen, Aaron Steven White, Benjamin Van Durme, Mahsa Yarmohammadi. (2024)  
**MultiMUC: Multilingual Template Filling on MUC-4**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Multilingual  
[Paper Link](http://arxiv.org/abs/2401.16209v1)  

---


**ABSTRACT**  
We introduce MultiMUC, the first multilingual parallel corpus for template filling, comprising translations of the classic MUC-4 template filling benchmark into five languages: Arabic, Chinese, Farsi, Korean, and Russian. We obtain automatic translations from a strong multilingual machine translation system and manually project the original English annotations into each target language. For all languages, we also provide human translations for sentences in the dev and test splits that contain annotated template arguments. Finally, we present baselines on MultiMUC both with state-of-the-art template filling models and with ChatGPT.

{{</citation>}}


### (27/132) LLaMandement: Large Language Models for Summarization of French Legislative Proposals (Joseph Gesnouin et al., 2024)

{{<citation>}}

Joseph Gesnouin, Yannis Tannier, Christophe Gomes Da Silva, Hatim Tapory, Camille Brier, Hugo Simon, Raphael Rozenberg, Hermann Woehrel, Mehdi El Yakaabi, Thomas Binder, Guillaume Marie, Emilie Caron, Mathile Nogueira, Thomas Fontas, Laure Puydebois, Marie Theophile, Stephane Morandi, Mael Petit, David Creissac, Pauline Ennouchy, Elise Valetoux, Celine Visade, Severine Balloux, Emmanuel Cortes, Pierre-Etienne Devineau, Ulrich Tan, Esther Mac Namara, Su Yang. (2024)  
**LLaMandement: Large Language Models for Summarization of French Legislative Proposals**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2401.16182v1)  

---


**ABSTRACT**  
This report introduces LLaMandement, a state-of-the-art Large Language Model, fine-tuned by the French government and designed to enhance the efficiency and efficacy of processing parliamentary sessions (including the production of bench memoranda and documents required for interministerial meetings) by generating neutral summaries of legislative proposals. Addressing the administrative challenges of manually processing a growing volume of legislative amendments, LLaMandement stands as a significant legal technological milestone, providing a solution that exceeds the scalability of traditional human efforts while matching the robustness of a specialized legal drafter. We release all our fine-tuned models and training data to the community.

{{</citation>}}


### (28/132) Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception (Junyang Wang et al., 2024)

{{<citation>}}

Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, Jitao Sang. (2024)  
**Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.16158v1)  

---


**ABSTRACT**  
Mobile device agent based on Multimodal Large Language Models (MLLM) is becoming a popular application. In this paper, we introduce Mobile-Agent, an autonomous multi-modal mobile device agent. Mobile-Agent first leverages visual perception tools to accurately identify and locate both the visual and textual elements within the app's front-end interface. Based on the perceived vision context, it then autonomously plans and decomposes the complex operation task, and navigates the mobile Apps through operations step by step. Different from previous solutions that rely on XML files of Apps or mobile system metadata, Mobile-Agent allows for greater adaptability across diverse mobile operating environments in a vision-centric way, thereby eliminating the necessity for system-specific customizations. To assess the performance of Mobile-Agent, we introduced Mobile-Eval, a benchmark for evaluating mobile device operations. Based on Mobile-Eval, we conducted a comprehensive evaluation of Mobile-Agent. The experimental results indicate that Mobile-Agent achieved remarkable accuracy and completion rates. Even with challenging instructions, such as multi-app operations, Mobile-Agent can still complete the requirements. Code and model will be open-sourced at https://github.com/X-PLUG/MobileAgent.

{{</citation>}}


### (29/132) Beyond Direct Diagnosis: LLM-based Multi-Specialist Agent Consultation for Automatic Diagnosis (Haochun Wang et al., 2024)

{{<citation>}}

Haochun Wang, Sendong Zhao, Zewen Qiang, Nuwa Xi, Bing Qin, Ting Liu. (2024)  
**Beyond Direct Diagnosis: LLM-based Multi-Specialist Agent Consultation for Automatic Diagnosis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.16107v1)  

---


**ABSTRACT**  
Automatic diagnosis is a significant application of AI in healthcare, where diagnoses are generated based on the symptom description of patients. Previous works have approached this task directly by modeling the relationship between the normalized symptoms and all possible diseases. However, in the clinical diagnostic process, patients are initially consulted by a general practitioner and, if necessary, referred to specialists in specific domains for a more comprehensive evaluation. The final diagnosis often emerges from a collaborative consultation among medical specialist groups. Recently, large language models have shown impressive capabilities in natural language understanding. In this study, we adopt tuning-free LLM-based agents as medical practitioners and propose the Agent-derived Multi-Specialist Consultation (AMSC) framework to model the diagnosis process in the real world by adaptively fusing probability distributions of agents over potential diseases. Experimental results demonstrate the superiority of our approach compared with baselines. Notably, our approach requires significantly less parameter updating and training time, enhancing efficiency and practical utility. Furthermore, we delve into a novel perspective on the role of implicit symptoms within the context of automatic diagnosis.

{{</citation>}}


### (30/132) Multilingual Text-to-Image Generation Magnifies Gender Stereotypes and Prompt Engineering May Not Help You (Felix Friedrich et al., 2024)

{{<citation>}}

Felix Friedrich, Katharina Hämmerl, Patrick Schramowski, Jindrich Libovicky, Kristian Kersting, Alexander Fraser. (2024)  
**Multilingual Text-to-Image Generation Magnifies Gender Stereotypes and Prompt Engineering May Not Help You**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2401.16092v2)  

---


**ABSTRACT**  
Text-to-image generation models have recently achieved astonishing results in image quality, flexibility, and text alignment and are consequently employed in a fast-growing number of applications. Through improvements in multilingual abilities, a larger community now has access to this kind of technology. Yet, as we will show, multilingual models suffer similarly from (gender) biases as monolingual models. Furthermore, the natural expectation is that these models will provide similar results across languages, but this is not the case and there are important differences between languages. Thus, we propose a novel benchmark MAGBIG intending to foster research in multilingual models without gender bias. We investigate whether multilingual T2I models magnify gender bias with MAGBIG. To this end, we use multilingual prompts requesting portrait images of persons of a certain occupation or trait (using adjectives). Our results show not only that models deviate from the normative assumption that each gender should be equally likely to be generated, but that there are also big differences across languages. Furthermore, we investigate prompt engineering strategies, i.e. the use of indirect, neutral formulations, as a possible remedy for these biases. Unfortunately, they help only to a limited extent and result in worse text-to-image alignment. Consequently, this work calls for more research into diverse representations across languages in image generators.

{{</citation>}}


### (31/132) Non-Fluent Synthetic Target-Language Data Improve Neural Machine Translation (Víctor M. Sánchez-Cartagena et al., 2024)

{{<citation>}}

Víctor M. Sánchez-Cartagena, Miquel Esplà-Gomis, Juan Antonio Pérez-Ortiz, Felipe Sánchez-Martínez. (2024)  
**Non-Fluent Synthetic Target-Language Data Improve Neural Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.16086v1)  

---


**ABSTRACT**  
When the amount of parallel sentences available to train a neural machine translation is scarce, a common practice is to generate new synthetic training samples from them. A number of approaches have been proposed to produce synthetic parallel sentences that are similar to those in the parallel data available. These approaches work under the assumption that non-fluent target-side synthetic training samples can be harmful and may deteriorate translation performance. Even so, in this paper we demonstrate that synthetic training samples with non-fluent target sentences can improve translation performance if they are used in a multilingual machine translation framework as if they were sentences in another language. We conducted experiments on ten low-resource and four high-resource translation tasks and found out that this simple approach consistently improves translation performance as compared to state-of-the-art methods for generating synthetic training samples similar to those found in corpora. Furthermore, this improvement is independent of the size of the original training corpus, the resulting systems are much more robust against domain shift and produce less hallucinations.

{{</citation>}}


### (32/132) Stolen Subwords: Importance of Vocabularies for Machine Translation Model Stealing (Vilém Zouhar, 2024)

{{<citation>}}

Vilém Zouhar. (2024)  
**Stolen Subwords: Importance of Vocabularies for Machine Translation Model Stealing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation, NLP  
[Paper Link](http://arxiv.org/abs/2401.16055v1)  

---


**ABSTRACT**  
In learning-based functionality stealing, the attacker is trying to build a local model based on the victim's outputs. The attacker has to make choices regarding the local model's architecture, optimization method and, specifically for NLP models, subword vocabulary, such as BPE. On the machine translation task, we explore (1) whether the choice of the vocabulary plays a role in model stealing scenarios and (2) if it is possible to extract the victim's vocabulary. We find that the vocabulary itself does not have a large effect on the local model's performance. Given gray-box model access, it is possible to collect the victim's vocabulary by collecting the outputs (detokenized subwords on the output). The results of the minimum effect of vocabulary choice are important more broadly for black-box knowledge distillation.

{{</citation>}}


### (33/132) Finding Challenging Metaphors that Confuse Pretrained Language Models (Yucheng Li et al., 2024)

{{<citation>}}

Yucheng Li, Frank Guerin, Chenghua Lin. (2024)  
**Finding Challenging Metaphors that Confuse Pretrained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLI, NLP, Pretrained Language Models, QA  
[Paper Link](http://arxiv.org/abs/2401.16012v1)  

---


**ABSTRACT**  
Metaphors are considered to pose challenges for a wide spectrum of NLP tasks. This gives rise to the area of computational metaphor processing. However, it remains unclear what types of metaphors challenge current state-of-the-art models. In this paper, we test various NLP models on the VUA metaphor dataset and quantify to what extent metaphors affect models' performance on various downstream tasks. Analysis reveals that VUA includes a large number of metaphors that pose little difficulty to downstream tasks. We would like to shift the attention of researchers away from these metaphors to instead focus on challenging metaphors. To identify hard metaphors, we propose an automatic pipeline that identifies metaphors that challenge a particular model. Our analysis demonstrates that our detected hard metaphors contrast significantly with VUA and reduce the accuracy of machine translation by 16\%, QA performance by 4\%, NLI by 7\%, and metaphor identification recall by over 14\% for various popular NLP systems.

{{</citation>}}


### (34/132) Response Generation for Cognitive Behavioral Therapy with Large Language Models: Comparative Study with Socratic Questioning (Kenta Izumi et al., 2024)

{{<citation>}}

Kenta Izumi, Hiroki Tanaka, Kazuhiro Shidara, Hiroyoshi Adachi, Daisuke Kanayama, Takashi Kudo, Satoshi Nakamura. (2024)  
**Response Generation for Cognitive Behavioral Therapy with Large Language Models: Comparative Study with Socratic Questioning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Dialog, Dialogue, GPT, GPT-4, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.15966v1)  

---


**ABSTRACT**  
Dialogue systems controlled by predefined or rule-based scenarios derived from counseling techniques, such as cognitive behavioral therapy (CBT), play an important role in mental health apps. Despite the need for responsible responses, it is conceivable that using the newly emerging LLMs to generate contextually relevant utterances will enhance these apps. In this study, we construct dialogue modules based on a CBT scenario focused on conventional Socratic questioning using two kinds of LLMs: a Transformer-based dialogue model further trained with a social media empathetic counseling dataset, provided by Osaka Prefecture (OsakaED), and GPT-4, a state-of-the art LLM created by OpenAI. By comparing systems that use LLM-generated responses with those that do not, we investigate the impact of generated responses on subjective evaluations such as mood change, cognitive change, and dialogue quality (e.g., empathy). As a result, no notable improvements are observed when using the OsakaED model. When using GPT-4, the amount of mood change, empathy, and other dialogue qualities improve significantly. Results suggest that GPT-4 possesses a high counseling ability. However, they also indicate that even when using a dialogue model trained with a human counseling dataset, it does not necessarily yield better outcomes compared to scenario-based dialogues. While presenting LLM-generated responses, including GPT-4, and having them interact directly with users in real-life mental health care services may raise ethical issues, it is still possible for human professionals to produce example responses or response templates using LLMs in advance in systems that use rules, scenarios, or example responses.

{{</citation>}}


### (35/132) E-EVAL: A Comprehensive Chinese K-12 Education Evaluation Benchmark for Large Language Models (Jinchang Hou et al., 2024)

{{<citation>}}

Jinchang Hou, Chang Ao, Haihong Wu, Xiangtao Kong, Zhigang Zheng, Daijia Tang, Chengming Li, Xiping Hu, Ruifeng Xu, Shiwen Ni, Min Yang. (2024)  
**E-EVAL: A Comprehensive Chinese K-12 Education Evaluation Benchmark for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.15927v1)  

---


**ABSTRACT**  
With the accelerating development of Large Language Models (LLMs), many LLMs are beginning to be used in the Chinese K-12 education domain. The integration of LLMs and education is getting closer and closer, however, there is currently no benchmark for evaluating LLMs that focuses on the Chinese K-12 education domain. Therefore, there is an urgent need for a comprehensive natural language processing benchmark to accurately assess the capabilities of various LLMs in the Chinese K-12 education domain. To address this, we introduce the E-EVAL, the first comprehensive evaluation benchmark specifically designed for the Chinese K-12 education field. The E-EVAL consists of 4,351 multiple-choice questions at the primary, middle, and high school levels across a wide range of subjects, including Chinese, English, Politics, History, Ethics, Physics, Chemistry, Mathematics, and Geography. We conducted a comprehensive evaluation of E-EVAL on advanced LLMs, including both English-dominant and Chinese-dominant models. Findings show that Chinese-dominant models perform well compared to English-dominant models, with many scoring even above the GPT 4.0. However, almost all models perform poorly in complex subjects such as mathematics. We also found that most Chinese-dominant LLMs did not achieve higher scores at the primary school level compared to the middle school level. We observe that the mastery of higher-order knowledge by the model does not necessarily imply the mastery of lower-order knowledge as well. Additionally, the experimental results indicate that the Chain of Thought (CoT) technique is effective only for the challenging science subjects, while Few-shot prompting is more beneficial for liberal arts subjects. With E-EVAL, we aim to analyze the strengths and limitations of LLMs in educational applications, and to contribute to the progress and development of Chinese K-12 education and LLMs.

{{</citation>}}


### (36/132) DrBERT: Unveiling the Potential of Masked Language Modeling Decoder in BERT pretraining (Wen Liang et al., 2024)

{{<citation>}}

Wen Liang, Youzhi Liang. (2024)  
**DrBERT: Unveiling the Potential of Masked Language Modeling Decoder in BERT pretraining**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, GLUE, Language Model, NLU, Natural Language Understanding, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.15861v1)  

---


**ABSTRACT**  
BERT (Bidirectional Encoder Representations from Transformers) has revolutionized the field of natural language processing through its exceptional performance on numerous tasks. Yet, the majority of researchers have mainly concentrated on enhancements related to the model structure, such as relative position embedding and more efficient attention mechanisms. Others have delved into pretraining tricks associated with Masked Language Modeling, including whole word masking. DeBERTa introduced an enhanced decoder adapted for BERT's encoder model for pretraining, proving to be highly effective. We argue that the design and research around enhanced masked language modeling decoders have been underappreciated. In this paper, we propose several designs of enhanced decoders and introduce DrBERT (Decoder-refined BERT), a novel method for modeling training. Typically, a pretrained BERT model is fine-tuned for specific Natural Language Understanding (NLU) tasks. In our approach, we utilize the original BERT model as the encoder, making only changes to the decoder without altering the encoder. This approach does not necessitate extensive modifications to the model's architecture and can be seamlessly integrated into existing fine-tuning pipelines and services, offering an efficient and effective enhancement strategy. Compared to other methods, while we also incur a moderate training cost for the decoder during the pretraining process, our approach does not introduce additional training costs during the fine-tuning phase. We test multiple enhanced decoder structures after pretraining and evaluate their performance on the GLUE benchmark. Our results demonstrate that DrBERT, having only undergone subtle refinements to the model structure during pretraining, significantly enhances model performance without escalating the inference time and serving budget.

{{</citation>}}


### (37/132) LSTM-based Deep Neural Network With A Focus on Sentence Representation for Sequential Sentence Classification in Medical Scientific Abstracts (Phat Lam et al., 2024)

{{<citation>}}

Phat Lam, Lam Pham, Tin Nguyen, Hieu Tang, Seidl Michael, Alexander Schindler. (2024)  
**LSTM-based Deep Neural Network With A Focus on Sentence Representation for Sequential Sentence Classification in Medical Scientific Abstracts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.15854v1)  

---


**ABSTRACT**  
The Sequential Sentence Classification task within the domain of medical abstracts, termed as SSC, involves the categorization of sentences into pre-defined headings based on their roles in conveying critical information in the abstract. In the SSC task, sentences are often sequentially related to each other. For this reason, the role of sentence embedding is crucial for capturing both the semantic information between words in the sentence and the contextual relationship of sentences within the abstract to provide a comprehensive representation for better classification. In this paper, we present a hierarchical deep learning model for the SSC task. First, we propose a LSTM-based network with multiple feature branches to create well-presented sentence embeddings at the sentence level. To perform the sequence of sentences, a convolutional-recurrent neural network (C-RNN) at the abstract level and a multi-layer perception network (MLP) at the segment level are developed that further enhance the model performance. Additionally, an ablation study is also conducted to evaluate the contribution of individual component in the entire network to the model performance at different levels. Our proposed system is very competitive to the state-of-the-art systems and further improve F1 scores of the baseline by 1.0%, 2.8%, and 2.6% on the benchmark datasets PudMed 200K RCT, PudMed 20K RCT and NICTA-PIBOSO, respectively.

{{</citation>}}


### (38/132) Emergent Explainability: Adding a causal chain to neural network inference (Adam Perrett, 2024)

{{<citation>}}

Adam Perrett. (2024)  
**Emergent Explainability: Adding a causal chain to neural network inference**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15840v1)  

---


**ABSTRACT**  
This position paper presents a theoretical framework for enhancing explainable artificial intelligence (xAI) through emergent communication (EmCom), focusing on creating a causal understanding of AI model outputs. We explore the novel integration of EmCom into AI systems, offering a paradigm shift from conventional associative relationships between inputs and outputs to a more nuanced, causal interpretation. The framework aims to revolutionize how AI processes are understood, making them more transparent and interpretable. While the initial application of this model is demonstrated on synthetic data, the implications of this research extend beyond these simple applications. This general approach has the potential to redefine interactions with AI across multiple domains, fostering trust and informed decision-making in healthcare and in various sectors where AI's decision-making processes are critical. The paper discusses the theoretical underpinnings of this approach, its potential broad applications, and its alignment with the growing need for responsible and transparent AI systems in an increasingly digital world.

{{</citation>}}


## cs.CR (5)



### (39/132) Data-Oblivious ML Accelerators using Hardware Security Extensions (Hossam ElAtali et al., 2024)

{{<citation>}}

Hossam ElAtali, John Z. Jekel, Lachlan J. Gunn, N. Asokan. (2024)  
**Data-Oblivious ML Accelerators using Hardware Security Extensions**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.16583v1)  

---


**ABSTRACT**  
Outsourced computation can put client data confidentiality at risk. Existing solutions are either inefficient or insufficiently secure: cryptographic techniques like fully-homomorphic encryption incur significant overheads, even with hardware assistance, while the complexity of hardware-assisted trusted execution environments has been exploited to leak secret data.   Recent proposals such as BliMe and OISA show how dynamic information flow tracking (DIFT) enforced in hardware can protect client data efficiently. They are designed to protect CPU-only workloads. However, many outsourced computing applications, like machine learning, make extensive use of accelerators.   We address this gap with Dolma, which applies DIFT to the Gemmini matrix multiplication accelerator, efficiently guaranteeing client data confidentiality, even in the presence of malicious/vulnerable software and side channel attacks on the server. We show that accelerators can allow DIFT logic optimizations that significantly reduce area overhead compared with general-purpose processor architectures. Dolma is integrated with the BliMe framework to achieve end-to-end security guarantees. We evaluate Dolma on an FPGA using a ResNet-50 DNN model and show that it incurs low overheads for large configurations ($4.4\%$, $16.7\%$, $16.5\%$ for performance, resource usage and power, respectively, with a 32x32 configuration).

{{</citation>}}


### (40/132) Quantum-safe Encryption: A New Method to Reduce Complexity and/or Improve Security Level (Amir K. Khandani, 2024)

{{<citation>}}

Amir K. Khandani. (2024)  
**Quantum-safe Encryption: A New Method to Reduce Complexity and/or Improve Security Level**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-IT, cs.CR, math-IT  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.16302v1)  

---


**ABSTRACT**  
This work presents some novel techniques to enhance an encryption scheme motivated by classical McEliece cryptosystem. Contributions include: (1) using masking matrices to hide sensitive data, (2) allowing both legitimate parties to incorporate randomness in the public key without sharing any additional public information, (3) using concatenation of a repetition code for error correction, permitting key recovery with a negligible decoding complexity, (4) making attacks more difficult by increasing the complexity in verifying a given key candidate has resulted in the actual key, (5) introducing memory in the error sequence such that: (i) error vector is composed of a random number of erroneous bits, (ii) errors can be all corrected when used in conjunction with concatenation of a repetition code of length 3. Proposed techniques allow generating significantly larger keys, at the same time, with a much lower complexity, as compared to known post-quantum key generation techniques relying on randomization.

{{</citation>}}


### (41/132) LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning (Yuqiang Sun et al., 2024)

{{<citation>}}

Yuqiang Sun, Daoyuan Wu, Yue Xue, Han Liu, Wei Ma, Lyuye Zhang, Miaolei Shi, Yang Liu. (2024)  
**LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-SE, cs.CR  
Keywords: GPT, GPT-4, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.16185v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated significant potential for many downstream tasks, including those requiring human-level intelligence, such as vulnerability detection. However, recent attempts to use LLMs for vulnerability detection are still preliminary, as they lack an in-depth understanding of a subject LLM's vulnerability reasoning capability -- whether it originates from the model itself or from external assistance, such as invoking tool support and retrieving vulnerability knowledge. In this paper, we aim to decouple LLMs' vulnerability reasoning capability from their other capabilities, including the ability to actively seek additional information (e.g., via function calling in SOTA models), adopt relevant vulnerability knowledge (e.g., via vector-based matching and retrieval), and follow instructions to output structured results. To this end, we propose a unified evaluation framework named LLM4Vuln, which separates LLMs' vulnerability reasoning from their other capabilities and evaluates how LLMs' vulnerability reasoning could be enhanced when combined with the enhancement of other capabilities. To demonstrate the effectiveness of LLM4Vuln, we have designed controlled experiments using 75 ground-truth smart contract vulnerabilities, which were extensively audited as high-risk on Code4rena from August to November 2023, and tested them in 4,950 different scenarios across three representative LLMs (GPT-4, Mixtral, and Code Llama). Our results not only reveal ten findings regarding the varying effects of knowledge enhancement, context supplementation, prompt schemes, and models but also enable us to identify 9 zero-day vulnerabilities in two pilot bug bounty programs with over 1,000 USD being awarded.

{{</citation>}}


### (42/132) HEQuant: Marrying Homomorphic Encryption and Quantization for Communication-Efficient Private Inference (Tianshi Xu et al., 2024)

{{<citation>}}

Tianshi Xu, Meng Li, Runsheng Wang. (2024)  
**HEQuant: Marrying Homomorphic Encryption and Quantization for Communication-Efficient Private Inference**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2401.15970v2)  

---


**ABSTRACT**  
Secure two-party computation with homomorphic encryption (HE) protects data privacy with a formal security guarantee but suffers from high communication overhead. While previous works, e.g., Cheetah, Iron, etc, have proposed efficient HE-based protocols for different neural network (NN) operations, they still assume high precision, e.g., fixed point 37 bit, for the NN operations and ignore NNs' native robustness against quantization error. In this paper, we propose HEQuant, which features low-precision-quantization-aware optimization for the HE-based protocols. We observe the benefit of a naive combination of quantization and HE quickly saturates as bit precision goes down. Hence, to further improve communication efficiency, we propose a series of optimizations, including an intra-coefficient packing algorithm and a quantization-aware tiling algorithm, to simultaneously reduce the number and precision of the transferred data. Compared with prior-art HE-based protocols, e.g., CrypTFlow2, Cheetah, Iron, etc, HEQuant achieves $3.5\sim 23.4\times$ communication reduction and $3.0\sim 9.3\times$ latency reduction. Meanwhile, when compared with prior-art network optimization frameworks, e.g., SENet, SNL, etc, HEQuant also achieves $3.1\sim 3.6\times$ communication reduction.

{{</citation>}}


### (43/132) TransTroj: Transferable Backdoor Attacks to Pre-trained Models via Embedding Indistinguishability (Hao Wang et al., 2024)

{{<citation>}}

Hao Wang, Tao Xiang, Shangwei Guo, Jialing He, Hangcheng Liu, Tianwei Zhang. (2024)  
**TransTroj: Transferable Backdoor Attacks to Pre-trained Models via Embedding Indistinguishability**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs-LG, cs.CR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.15883v1)  

---


**ABSTRACT**  
Pre-trained models (PTMs) are extensively utilized in various downstream tasks. Adopting untrusted PTMs may suffer from backdoor attacks, where the adversary can compromise the downstream models by injecting backdoors into the PTM. However, existing backdoor attacks to PTMs can only achieve partially task-agnostic and the embedded backdoors are easily erased during the fine-tuning process. In this paper, we propose a novel transferable backdoor attack, TransTroj, to simultaneously meet functionality-preserving, durable, and task-agnostic. In particular, we first formalize transferable backdoor attacks as the indistinguishability problem between poisoned and clean samples in the embedding space. We decompose the embedding indistinguishability into pre- and post-indistinguishability, representing the similarity of the poisoned and reference embeddings before and after the attack. Then, we propose a two-stage optimization that separately optimizes triggers and victim PTMs to achieve embedding indistinguishability. We evaluate TransTroj on four PTMs and six downstream tasks. Experimental results show that TransTroj significantly outperforms SOTA task-agnostic backdoor attacks (18%$\sim$99%, 68% on average) and exhibits superior performance under various system settings. The code is available at https://github.com/haowang-cqu/TransTroj .

{{</citation>}}


## cs.AI (6)



### (44/132) Attention-based Reinforcement Learning for Combinatorial Optimization: Application to Job Shop Scheduling Problem (Jaejin Lee et al., 2024)

{{<citation>}}

Jaejin Lee, Seho Kee, Mani Janakiram, George Runger. (2024)  
**Attention-based Reinforcement Learning for Combinatorial Optimization: Application to Job Shop Scheduling Problem**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Attention, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.16580v1)  

---


**ABSTRACT**  
Job shop scheduling problems are one of the most important and challenging combinatorial optimization problems that have been tackled mainly by exact or approximate solution approaches. However, finding an exact solution can be infeasible for real-world problems, and even with an approximate solution approach, it can require a prohibitive amount of time to find a near-optimal solution, and the found solutions are not applicable to new problems in general. To address these challenges, we propose an attention-based reinforcement learning method for the class of job shop scheduling problems by integrating policy gradient reinforcement learning with a modified transformer architecture. An important result is that our trained learners in the proposed method can be reused to solve large-scale problems not used in training and demonstrate that our approach outperforms the results of recent studies and widely adopted heuristic rules.

{{</citation>}}


### (45/132) GAPS: Geometry-Aware Problem Solver (Jiaxin Zhang et al., 2024)

{{<citation>}}

Jiaxin Zhang, Yinghui Jiang, Yashar Moshfeghi. (2024)  
**GAPS: Geometry-Aware Problem Solver**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2401.16287v1)  

---


**ABSTRACT**  
Geometry problem solving presents a formidable challenge within the NLP community. Existing approaches often rely on models designed for solving math word problems, neglecting the unique characteristics of geometry math problems. Additionally, the current research predominantly focuses on geometry calculation problems, while overlooking other essential aspects like proving. In this study, we address these limitations by proposing the Geometry-Aware Problem Solver (GAPS) model. GAPS is specifically designed to generate solution programs for geometry math problems of various types with the help of its unique problem-type classifier. To achieve this, GAPS treats the solution program as a composition of operators and operands, segregating their generation processes. Furthermore, we introduce the geometry elements enhancement method, which enhances the ability of GAPS to recognize geometry elements accurately. By leveraging these improvements, GAPS showcases remarkable performance in resolving geometry math problems. Our experiments conducted on the UniGeo dataset demonstrate the superiority of GAPS over the state-of-the-art model, Geoformer. Specifically, GAPS achieves an accuracy improvement of more than 5.3% for calculation tasks and an impressive 41.1% for proving tasks. Notably, GAPS achieves an impressive accuracy of 97.5% on proving problems, representing a significant advancement in solving geometry proving tasks.

{{</citation>}}


### (46/132) A technical note for the 91-clauses SAT resolution with Indirect QAOA based approach (Gerard Fleury et al., 2024)

{{<citation>}}

Gerard Fleury, Philippe Lacomme. (2024)  
**A technical note for the 91-clauses SAT resolution with Indirect QAOA based approach**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2402.00065v1)  

---


**ABSTRACT**  
This paper addresses the resolution of the 3-SAT problem using a QAOA-like approach. The chosen principle involves modeling the solution ranks of the 3-SAT problem, which, in this particular case, directly represent a solution. This results in a highly compact circuit with few gates, enabling the modeling of large-sized 3-SAT problems. Numerical experimentation demonstrates that the approach can solve instances composed of 91 clauses and 20 variables with an implementation based on Qiskit.

{{</citation>}}


### (47/132) Capturing Knowledge Graphs and Rules with Octagon Embeddings (Victor Charpenay et al., 2024)

{{<citation>}}

Victor Charpenay, Steven Schockaert. (2024)  
**Capturing Knowledge Graphs and Rules with Octagon Embeddings**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.16270v1)  

---


**ABSTRACT**  
Region based knowledge graph embeddings represent relations as geometric regions. This has the advantage that the rules which are captured by the model are made explicit, making it straightforward to incorporate prior knowledge and to inspect learned models. Unfortunately, existing approaches are severely restricted in their ability to model relational composition, and hence also their ability to model rules, thus failing to deliver on the main promise of region based models. With the aim of addressing these limitations, we investigate regions which are composed of axis-aligned octagons. Such octagons are particularly easy to work with, as intersections and compositions can be straightforwardly computed, while they are still sufficiently expressive to model arbitrary knowledge graphs. Among others, we also show that our octagon embeddings can properly capture a non-trivial class of rule bases. Finally, we show that our model achieves competitive experimental results.

{{</citation>}}


### (48/132) Triple Disentangled Representation Learning for Multimodal Affective Analysis (Ying Zhou et al., 2024)

{{<citation>}}

Ying Zhou, Xuefeng Liang, Han Chen, Yin Zhao. (2024)  
**Triple Disentangled Representation Learning for Multimodal Affective Analysis**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.16119v1)  

---


**ABSTRACT**  
Multimodal learning has exhibited a significant advantage in affective analysis tasks owing to the comprehensive information of various modalities, particularly the complementary information. Thus, many emerging studies focus on disentangling the modality-invariant and modality-specific representations from input data and then fusing them for prediction. However, our study shows that modality-specific representations may contain information that is irrelevant or conflicting with the tasks, which downgrades the effectiveness of learned multimodal representations. We revisit the disentanglement issue, and propose a novel triple disentanglement approach, TriDiRA, which disentangles the modality-invariant, effective modality-specific and ineffective modality-specific representations from input data. By fusing only the modality-invariant and effective modality-specific representations, TriDiRA can significantly alleviate the impact of irrelevant and conflicting information across modalities during model training. Extensive experiments conducted on four benchmark datasets demonstrate the effectiveness and generalization of our triple disentanglement, which outperforms SOTA methods.

{{</citation>}}


### (49/132) Type-based Neural Link Prediction Adapter for Complex Query Answering (Lingning Song et al., 2024)

{{<citation>}}

Lingning Song, Yi Zu, Shan Lu, Jieyue He. (2024)  
**Type-based Neural Link Prediction Adapter for Complex Query Answering**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2401.16045v1)  

---


**ABSTRACT**  
Answering complex logical queries on incomplete knowledge graphs (KGs) is a fundamental and challenging task in multi-hop reasoning. Recent work defines this task as an end-to-end optimization problem, which significantly reduces the training cost and enhances the generalization of the model by a pretrained link predictors for query answering. However, most existing proposals ignore the critical semantic knowledge inherently available in KGs, such as type information, which could help answer complex logical queries. To this end, we propose TypE-based Neural Link Prediction Adapter (TENLPA), a novel model that constructs type-based entity-relation graphs to discover the latent relationships between entities and relations by leveraging type information in KGs. Meanwhile, in order to effectively combine type information with complex logical queries, an adaptive learning mechanism is introduced, which is trained by back-propagating during the complex query answering process to achieve adaptive adjustment of neural link predictors. Experiments on 3 standard datasets show that TENLPA model achieves state-of-the-art performance on complex query answering with good generalization and robustness.

{{</citation>}}


## cs.CY (3)



### (50/132) Embedding Elites: Examining the Use of Tweets Embedded in Online News Articles across Reliable and Fringe Outlets (Benjamin D. Horne et al., 2024)

{{<citation>}}

Benjamin D. Horne, Summer Phillips, Nelia Koontz. (2024)  
**Embedding Elites: Examining the Use of Tweets Embedded in Online News Articles across Reliable and Fringe Outlets**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-SI, cs.CY  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.16572v1)  

---


**ABSTRACT**  
This study examines the use of embedded tweets in online news media. In particular, we add to the previous literature by exploring embedded tweets across reliable and unreliable news outlets. We use a mixed-method analysis to examine how the function and frequency of embedded tweets change across outlet reliability and news topic. We find that, no matter the outlet reliability, embedded tweets are most often used to relay the opinions of elites, to syndicate information from another news source, or to self-cite information an outlet previously produced. Our results also show some notable differences between reliable media and fringe media's use of tweets. Namely, fringe media embed tweets more and use those tweets as the source of news more than reliable media. Our work adds to the literature on hybrid media systems and the normalization of social media in journalism.

{{</citation>}}


### (51/132) Diverse, but Divisive: LLMs Can Exaggerate Gender Differences in Opinion Related to Harms of Misinformation (Terrence Neumann et al., 2024)

{{<citation>}}

Terrence Neumann, Sooyong Lee, Maria De-Arteaga, Sina Fazelpour, Matthew Lease. (2024)  
**Diverse, but Divisive: LLMs Can Exaggerate Gender Differences in Opinion Related to Harms of Misinformation**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs.CY  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2401.16558v1)  

---


**ABSTRACT**  
The pervasive spread of misinformation and disinformation poses a significant threat to society. Professional fact-checkers play a key role in addressing this threat, but the vast scale of the problem forces them to prioritize their limited resources. This prioritization may consider a range of factors, such as varying risks of harm posed to specific groups of people. In this work, we investigate potential implications of using a large language model (LLM) to facilitate such prioritization. Because fact-checking impacts a wide range of diverse segments of society, it is important that diverse views are represented in the claim prioritization process. This paper examines whether a LLM can reflect the views of various groups when assessing the harms of misinformation, focusing on gender as a primary variable. We pose two central questions: (1) To what extent do prompts with explicit gender references reflect gender differences in opinion in the United States on topics of social relevance? and (2) To what extent do gender-neutral prompts align with gendered viewpoints on those topics? To analyze these questions, we present the TopicMisinfo dataset, containing 160 fact-checked claims from diverse topics, supplemented by nearly 1600 human annotations with subjective perceptions and annotator demographics. Analyzing responses to gender-specific and neutral prompts, we find that GPT 3.5-Turbo reflects empirically observed gender differences in opinion but amplifies the extent of these differences. These findings illuminate AI's complex role in moderating online communication, with implications for fact-checkers, algorithm designers, and the use of crowd-workers as annotators. We also release the TopicMisinfo dataset to support continuing research in the community.

{{</citation>}}


### (52/132) Red-Teaming for Generative AI: Silver Bullet or Security Theater? (Michael Feffer et al., 2024)

{{<citation>}}

Michael Feffer, Anusha Sinha, Zachary C. Lipton, Hoda Heidari. (2024)  
**Red-Teaming for Generative AI: Silver Bullet or Security Theater?**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-HC, cs-LG, cs.CY  
Keywords: AI, Generative AI, Security  
[Paper Link](http://arxiv.org/abs/2401.15897v1)  

---


**ABSTRACT**  
In response to rising concerns surrounding the safety, security, and trustworthiness of Generative AI (GenAI) models, practitioners and regulators alike have pointed to AI red-teaming as a key component of their strategies for identifying and mitigating these risks. However, despite AI red-teaming's central role in policy discussions and corporate messaging, significant questions remain about what precisely it means, what role it can play in regulation, and how precisely it relates to conventional red-teaming practices as originally conceived in the field of cybersecurity. In this work, we identify recent cases of red-teaming activities in the AI industry and conduct an extensive survey of the relevant research literature to characterize the scope, structure, and criteria for AI red-teaming practices. Our analysis reveals that prior methods and practices of AI red-teaming diverge along several axes, including the purpose of the activity (which is often vague), the artifact under evaluation, the setting in which the activity is conducted (e.g., actors, resources, and methods), and the resulting decisions it informs (e.g., reporting, disclosure, and mitigation). In light of our findings, we argue that while red-teaming may be a valuable big-tent idea for characterizing a broad set of activities and attitudes aimed at improving the behavior of GenAI models, gestures towards red-teaming as a panacea for every possible risk verge on security theater. To move toward a more robust toolbox of evaluations for generative AI, we synthesize our recommendations into a question bank meant to guide and scaffold future AI red-teaming practices.

{{</citation>}}


## cs.SI (1)



### (53/132) Keep Your Friends Close, and Your Enemies Closer: Structural Properties of Negative Relationships on Twitter (Jack Tacchi et al., 2024)

{{<citation>}}

Jack Tacchi, Chiara Boldrini, Andrea Passarella, Marco Conti. (2024)  
**Keep Your Friends Close, and Your Enemies Closer: Structural Properties of Negative Relationships on Twitter**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network, Twitter  
[Paper Link](http://arxiv.org/abs/2401.16562v1)  

---


**ABSTRACT**  
The Ego Network Model (ENM) is a model for the structural organisation of relationships, rooted in evolutionary anthropology, that is found ubiquitously in social contexts. It takes the perspective of a single user (Ego) and organises their contacts (Alters) into a series of (typically 5) concentric circles of decreasing intimacy and increasing size. Alters are sorted based on their tie strength to the Ego, however, this is difficult to measure directly. Traditionally, the interaction frequency has been used as a proxy but this misses the qualitative aspects of connections, such as signs (i.e. polarity), which have been shown to provide extremely useful information. However, the sign of an online social relationship is usually an implicit piece of information, which needs to be estimated by interaction data from Online Social Networks (OSNs), making sign prediction in OSNs a research challenge in and of itself. This work aims to bring the ENM into the signed networks domain by investigating the interplay of signed connections with the ENM. This paper delivers 2 main contributions. Firstly, a new and data-efficient method of signing relationships between individuals using sentiment analysis and, secondly, we provide an in-depth look at the properties of Signed Ego Networks (SENs), using 9 Twitter datasets of various categories of users. We find that negative connections are generally over-represented in the active part of the Ego Networks, suggesting that Twitter greatly over-emphasises negative relationships with respect to "offline" social networks. Further, users who use social networks for professional reasons have an even greater share of negative connections.

{{</citation>}}


## cs.DC (2)



### (54/132) Leveraging Public Cloud Infrastructure for Real-time Connected Vehicle Speed Advisory at a Signalized Corridor (Hsien-Wen Deng et al., 2024)

{{<citation>}}

Hsien-Wen Deng, M Sabbir Salek, Mizanur Rahman, Mashrur Chowdhury, Mitch Shue, Amy W. Apon. (2024)  
**Leveraging Public Cloud Infrastructure for Real-time Connected Vehicle Speed Advisory at a Signalized Corridor**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AWS, Amazon  
[Paper Link](http://arxiv.org/abs/2401.16545v1)  

---


**ABSTRACT**  
In this study, we developed a real-time connected vehicle (CV) speed advisory application that uses public cloud services and tested it on a simulated signalized corridor for different roadway traffic conditions. First, we developed a scalable serverless cloud computing architecture leveraging public cloud services offered by Amazon Web Services (AWS) to support the requirements of a real-time CV application. Second, we developed an optimization-based real-time CV speed advisory algorithm by taking a modular design approach, which makes the application automatically scalable and deployable in the cloud using the serverless architecture. Third, we developed a cloud-in-the-loop simulation testbed using AWS and an open-source microscopic roadway traffic simulator called Simulation of Urban Mobility (SUMO). Our analyses based on different roadway traffic conditions showed that the serverless CV speed advisory application meets the latency requirement of real-time CV mobility applications. Besides, our serverless CV speed advisory application reduced the average stopped delay (by 77%) and the aggregated risk of collision (by 21%) at signalized intersection of a corridor. These prove the feasibility as well as the efficacy of utilizing public cloud infrastructure to implement real-time roadway traffic management applications in a CV environment.

{{</citation>}}


### (55/132) DF* PageRank: Improved Incrementally Expanding Approaches for Updating PageRank on Dynamic Graphs (Subhajit Sahu, 2024)

{{<citation>}}

Subhajit Sahu. (2024)  
**DF* PageRank: Improved Incrementally Expanding Approaches for Updating PageRank on Dynamic Graphs**  

---
Primary Category: cs.DC  
Categories: G-2-2; I-5-3, cs-DC, cs-PF, cs.DC  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2401.15870v2)  

---


**ABSTRACT**  
PageRank is a widely used centrality measure that assesses the significance of vertices in a graph by considering their connections and the importance of those connections. Efficiently updating PageRank on dynamic graphs is essential for various applications due to the increasing scale of datasets. This technical report introduces our improved Dynamic Frontier (DF) and Dynamic Frontier with Pruning (DF-P) approaches. Given a batch update comprising edge insertions and deletions, these approaches iteratively identify vertices likely to change their ranks with minimal overhead. On a server featuring a 64-core AMD EPYC-7742 processor, our approaches outperform Static and Dynamic Traversal PageRank by 5.2x/15.2x and 1.3x/3.5x respectively - on real-world dynamic graphs, and by 7.2x/9.6x and 4.0x/5.6x on large static graphs with random batch updates. Furthermore, our approaches improve performance at a rate of 1.8x/1.7x for every doubling of threads.

{{</citation>}}


## cs.LG (26)



### (56/132) TrackGPT -- A generative pre-trained transformer for cross-domain entity trajectory forecasting (Nicholas Stroh, 2024)

{{<citation>}}

Nicholas Stroh. (2024)  
**TrackGPT -- A generative pre-trained transformer for cross-domain entity trajectory forecasting**  

---
Primary Category: cs.LG  
Categories: 68T07, cs-AI, cs-LG, cs.LG  
Keywords: AI, ChatGPT, GPT, Language Model, NLP, Natural Language Processing, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.00066v1)  

---


**ABSTRACT**  
The forecasting of entity trajectories at future points in time is a critical capability gap in applications across both Commercial and Defense sectors. Transformers, and specifically Generative Pre-trained Transformer (GPT) networks have recently revolutionized several fields of Artificial Intelligence, most notably Natural Language Processing (NLP) with the advent of Large Language Models (LLM) like OpenAI's ChatGPT. In this research paper, we introduce TrackGPT, a GPT-based model for entity trajectory forecasting that has shown utility across both maritime and air domains, and we expect to perform well in others. TrackGPT stands as a pioneering GPT model capable of producing accurate predictions across diverse entity time series datasets, demonstrating proficiency in generating both long-term forecasts with sustained accuracy and short-term forecasts with high precision. We present benchmarks against state-of-the-art deep learning techniques, showing that TrackGPT's forecasting capability excels in terms of accuracy, reliability, and modularity. Importantly, TrackGPT achieves these results while remaining domain-agnostic and requiring minimal data features (only location and time) compared to models achieving similar performance. In conclusion, our findings underscore the immense potential of applying GPT architectures to the task of entity trajectory forecasting, exemplified by the innovative TrackGPT model.

{{</citation>}}


### (57/132) Validation, Robustness, and Accuracy of Perturbation-Based Sensitivity Analysis Methods for Time-Series Deep Learning Models (Zhengguang Wang, 2024)

{{<citation>}}

Zhengguang Wang. (2024)  
**Validation, Robustness, and Accuracy of Perturbation-Based Sensitivity Analysis Methods for Time-Series Deep Learning Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.16521v1)  

---


**ABSTRACT**  
This work undertakes studies to evaluate Interpretability Methods for Time-Series Deep Learning. Sensitivity analysis assesses how input changes affect the output, constituting a key component of interpretation. Among the post-hoc interpretation methods such as back-propagation, perturbation, and approximation, my work will investigate perturbation-based sensitivity Analysis methods on modern Transformer models to benchmark their performances. Specifically, my work answers three research questions: 1) Do different sensitivity analysis (SA) methods yield comparable outputs and attribute importance rankings? 2) Using the same sensitivity analysis method, do different Deep Learning (DL) models impact the output of the sensitivity analysis? 3) How well do the results from sensitivity analysis methods align with the ground truth?

{{</citation>}}


### (58/132) MT-HCCAR: Multi-Task Deep Learning with Hierarchical Classification and Attention-based Regression for Cloud Property Retrieval (Xingyan Li et al., 2024)

{{<citation>}}

Xingyan Li, Andrew M. Sayer, Ian T. Carroll, Xin Huang, Jianwu Wang. (2024)  
**MT-HCCAR: Multi-Task Deep Learning with Hierarchical Classification and Attention-based Regression for Cloud Property Retrieval**  

---
Primary Category: cs.LG  
Categories: 68T07, I-2-6, cs-CV, cs-LG, cs.LG, eess-SP  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.16520v1)  

---


**ABSTRACT**  
In the realm of Earth science, effective cloud property retrieval, encompassing cloud masking, cloud phase classification, and cloud optical thickness (COT) prediction, remains pivotal. Traditional methodologies necessitate distinct models for each sensor instrument due to their unique spectral characteristics. Recent strides in Earth Science research have embraced machine learning and deep learning techniques to extract features from satellite datasets' spectral observations. However, prevailing approaches lack novel architectures accounting for hierarchical relationships among retrieval tasks. Moreover, considering the spectral diversity among existing sensors, the development of models with robust generalization capabilities over different sensor datasets is imperative. Surprisingly, there is a dearth of methodologies addressing the selection of an optimal model for diverse datasets. In response, this paper introduces MT-HCCAR, an end-to-end deep learning model employing multi-task learning to simultaneously tackle cloud masking, cloud phase retrieval (classification tasks), and COT prediction (a regression task). The MT-HCCAR integrates a hierarchical classification network (HC) and a classification-assisted attention-based regression network (CAR), enhancing precision and robustness in cloud labeling and COT prediction. Additionally, a comprehensive model selection method rooted in K-fold cross-validation, one standard error rule, and two introduced performance scores is proposed to select the optimal model over three simulated satellite datasets OCI, VIIRS, and ABI. The experiments comparing MT-HCCAR with baseline methods, the ablation studies, and the model selection affirm the superiority and the generalization capabilities of MT-HCCAR.

{{</citation>}}


### (59/132) AFSD-Physics: Exploring the governing equations of temperature evolution during additive friction stir deposition by a human-AI teaming approach (Tony Shi et al., 2024)

{{<citation>}}

Tony Shi, Mason Ma, Jiajie Wu, Chase Post, Elijah Charles, Tony Schmitz. (2024)  
**AFSD-Physics: Exploring the governing equations of temperature evolution during additive friction stir deposition by a human-AI teaming approach**  

---
Primary Category: cs.LG  
Categories: cond-mat-mtrl-sci, cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.16501v1)  

---


**ABSTRACT**  
This paper presents a modeling effort to explore the underlying physics of temperature evolution during additive friction stir deposition (AFSD) by a human-AI teaming approach. AFSD is an emerging solid-state additive manufacturing technology that deposits materials without melting. However, both process modeling and modeling of the AFSD tool are at an early stage. In this paper, a human-AI teaming approach is proposed to combine models based on first principles with AI. The resulting human-informed machine learning method, denoted as AFSD-Physics, can effectively learn the governing equations of temperature evolution at the tool and the build from in-process measurements. Experiments are designed and conducted to collect in-process measurements for the deposition of aluminum 7075 with a total of 30 layers. The acquired governing equations are physically interpretable models with low computational cost and high accuracy. Model predictions show good agreement with the measurements. Experimental validation with new process parameters demonstrates the model's generalizability and potential for use in tool temperature control and process optimization.

{{</citation>}}


### (60/132) Continual Learning with Pre-Trained Models: A Survey (Da-Wei Zhou et al., 2024)

{{<citation>}}

Da-Wei Zhou, Hai-Long Sun, Jingyi Ning, Han-Jia Ye, De-Chuan Zhan. (2024)  
**Continual Learning with Pre-Trained Models: A Survey**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2401.16386v1)  

---


**ABSTRACT**  
Nowadays, real-world applications often face streaming data, which requires the learning system to absorb new knowledge as data evolves. Continual Learning (CL) aims to achieve this goal and meanwhile overcome the catastrophic forgetting of former knowledge when learning new ones. Typical CL methods build the model from scratch to grow with incoming data. However, the advent of the pre-trained model (PTM) era has sparked immense research interest, particularly in leveraging PTMs' robust representational capabilities. This paper presents a comprehensive survey of the latest advancements in PTM-based CL. We categorize existing methodologies into three distinct groups, providing a comparative analysis of their similarities, differences, and respective advantages and disadvantages. Additionally, we offer an empirical study contrasting various state-of-the-art methods to highlight concerns regarding fairness in comparisons. The source code to reproduce these evaluations is available at: https://github.com/sun-hailong/LAMDA-PILOT

{{</citation>}}


### (61/132) TQCompressor: improving tensor decomposition methods in neural networks via permutations (V. Abronin et al., 2024)

{{<citation>}}

V. Abronin, A. Naumov, D. Mazur, D. Bystrov, K. Tsarova, Ar. Melnikov, I. Oseledets, S. Dolgov, R. Brasher, M. Perelshtein. (2024)  
**TQCompressor: improving tensor decomposition methods in neural networks via permutations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GPT, NLP  
[Paper Link](http://arxiv.org/abs/2401.16367v1)  

---


**ABSTRACT**  
We introduce TQCompressor, a novel method for neural network model compression with improved tensor decompositions. We explore the challenges posed by the computational and storage demands of pre-trained language models in NLP tasks and propose a permutation-based enhancement to Kronecker decomposition. This enhancement makes it possible to reduce loss in model expressivity which is usually associated with factorization. We demonstrate this method applied to the GPT-2$_{small}$. The result of the compression is TQCompressedGPT-2 model, featuring 81 mln. parameters compared to 124 mln. in the GPT-2$_{small}$. We make TQCompressedGPT-2 publicly available. We further enhance the performance of the TQCompressedGPT-2 through a training strategy involving multi-step knowledge distillation, using only a 3.1% of the OpenWebText. TQCompressedGPT-2 surpasses DistilGPT-2 and KnGPT-2 in comparative evaluations, marking an advancement in the efficient and effective deployment of models in resource-constrained environments.

{{</citation>}}


### (62/132) Iterative Data Smoothing: Mitigating Reward Overfitting and Overoptimization in RLHF (Banghua Zhu et al., 2024)

{{<citation>}}

Banghua Zhu, Michael I. Jordan, Jiantao Jiao. (2024)  
**Iterative Data Smoothing: Mitigating Reward Overfitting and Overoptimization in RLHF**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.16335v1)  

---


**ABSTRACT**  
Reinforcement Learning from Human Feedback (RLHF) is a pivotal technique that aligns language models closely with human-centric values. The initial phase of RLHF involves learning human values using a reward model from ranking data. It is observed that the performance of the reward model degrades after one epoch of training, and optimizing too much against the learned reward model eventually hinders the true objective. This paper delves into these issues, leveraging the theoretical insights to design improved reward learning algorithm termed 'Iterative Data Smoothing' (IDS). The core idea is that during each training epoch, we not only update the model with the data, but also update the date using the model, replacing hard labels with soft labels. Our empirical findings highlight the superior performance of this approach over the traditional methods.

{{</citation>}}


### (63/132) PICL: Physics Informed Contrastive Learning for Partial Differential Equations (Cooper Lorsung et al., 2024)

{{<citation>}}

Cooper Lorsung, Amir Barati Farimani. (2024)  
**PICL: Physics Informed Contrastive Learning for Partial Differential Equations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NA, cs.LG, math-NA, physics-comp-ph  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.16327v1)  

---


**ABSTRACT**  
Neural operators have recently grown in popularity as Partial Differential Equation (PDEs) surrogate models. Learning solution functionals, rather than functions, has proven to be a powerful approach to calculate fast, accurate solutions to complex PDEs. While much work has been done evaluating neural operator performance on a wide variety of surrogate modeling tasks, these works normally evaluate performance on a single equation at a time. In this work, we develop a novel contrastive pretraining framework utilizing Generalized Contrastive Loss that improves neural operator generalization across multiple governing equations simultaneously. Governing equation coefficients are used to measure ground-truth similarity between systems. A combination of physics-informed system evolution and latent-space model output are anchored to input data and used in our distance function. We find that physics-informed contrastive pretraining improves both accuracy and generalization for the Fourier Neural Operator in fixed-future task, with comparable performance on the autoregressive rollout, and superresolution tasks for the 1D Heat, Burgers', and linear advection equations.

{{</citation>}}


### (64/132) Defining and Extracting generalizable interaction primitives from DNNs (Lu Chen et al., 2024)

{{<citation>}}

Lu Chen, Siyu Lou, Benhao Huang, Quanshi Zhang. (2024)  
**Defining and Extracting generalizable interaction primitives from DNNs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.16318v1)  

---


**ABSTRACT**  
Faithfully summarizing the knowledge encoded by a deep neural network (DNN) into a few symbolic primitive patterns without losing much information represents a core challenge in explainable AI. To this end, Ren et al. (2023c) have derived a series of theorems to prove that the inference score of a DNN can be explained as a small set of interactions between input variables. However, the lack of generalization power makes it still hard to consider such interactions as faithful primitive patterns encoded by the DNN. Therefore, given different DNNs trained for the same task, we develop a new method to extract interactions that are shared by these DNNs. Experiments show that the extracted interactions can better reflect common knowledge shared by different DNNs.

{{</citation>}}


### (65/132) Enhancing Molecular Property Prediction with Auxiliary Learning and Task-Specific Adaptation (Vishal Dey et al., 2024)

{{<citation>}}

Vishal Dey, Xia Ning. (2024)  
**Enhancing Molecular Property Prediction with Auxiliary Learning and Task-Specific Adaptation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.16299v1)  

---


**ABSTRACT**  
Pretrained Graph Neural Networks have been widely adopted for various molecular property prediction tasks. Despite their ability to encode structural and relational features of molecules, traditional fine-tuning of such pretrained GNNs on the target task can lead to poor generalization. To address this, we explore the adaptation of pretrained GNNs to the target task by jointly training them with multiple auxiliary tasks. This could enable the GNNs to learn both general and task-specific features, which may benefit the target task. However, a major challenge is to determine the relatedness of auxiliary tasks with the target task. To address this, we investigate multiple strategies to measure the relevance of auxiliary tasks and integrate such tasks by adaptively combining task gradients or by learning task weights via bi-level optimization. Additionally, we propose a novel gradient surgery-based approach, Rotation of Conflicting Gradients ($\mathtt{RCGrad}$), that learns to align conflicting auxiliary task gradients through rotation. Our experiments with state-of-the-art pretrained GNNs demonstrate the efficacy of our proposed methods, with improvements of up to 7.7% over fine-tuning. This suggests that incorporating auxiliary tasks along with target task fine-tuning can be an effective way to improve the generalizability of pretrained GNNs for molecular property prediction.

{{</citation>}}


### (66/132) Effective Communication with Dynamic Feature Compression (Pietro Talli et al., 2024)

{{<citation>}}

Pietro Talli, Francesco Pase, Federico Chiariotti, Andrea Zanella, Michele Zorzi. (2024)  
**Effective Communication with Dynamic Feature Compression**  

---
Primary Category: cs.LG  
Categories: cs-IT, cs-LG, cs-MA, cs.LG, math-IT, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.16236v1)  

---


**ABSTRACT**  
The remote wireless control of industrial systems is one of the major use cases for 5G and beyond systems: in these cases, the massive amounts of sensory information that need to be shared over the wireless medium may overload even high-capacity connections. Consequently, solving the effective communication problem by optimizing the transmission strategy to discard irrelevant information can provide a significant advantage, but is often a very complex task. In this work, we consider a prototypal system in which an observer must communicate its sensory data to a robot controlling a task (e.g., a mobile robot in a factory). We then model it as a remote Partially Observable Markov Decision Process (POMDP), considering the effect of adopting semantic and effective communication-oriented solutions on the overall system performance. We split the communication problem by considering an ensemble Vector Quantized Variational Autoencoder (VQ-VAE) encoding, and train a Deep Reinforcement Learning (DRL) agent to dynamically adapt the quantization level, considering both the current state of the environment and the memory of past messages. We tested the proposed approach on the well-known CartPole reference control problem, obtaining a significant performance increase over traditional approaches.

{{</citation>}}


### (67/132) Supervised Contrastive Learning based Dual-Mixer Model for Remaining Useful Life Prediction (En Fu et al., 2024)

{{<citation>}}

En Fu, Yanyan Hu, Kaixiang Peng, Yuxin Chu. (2024)  
**Supervised Contrastive Learning based Dual-Mixer Model for Remaining Useful Life Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.16462v1)  

---


**ABSTRACT**  
The problem of the Remaining Useful Life (RUL) prediction, aiming at providing an accurate estimate of the remaining time from the current predicting moment to the complete failure of the device, has gained significant attention from researchers in recent years. In this paper, to overcome the shortcomings of rigid combination for temporal and spatial features in most existing RUL prediction approaches, a spatial-temporal homogeneous feature extractor, named Dual-Mixer model, is firstly proposed. Flexible layer-wise progressive feature fusion is employed to ensure the homogeneity of spatial-temporal features and enhance the prediction accuracy. Secondly, the Feature Space Global Relationship Invariance (FSGRI) training method is introduced based on supervised contrastive learning. This method maintains the consistency of relationships among sample features with their degradation patterns during model training, simplifying the subsequently regression task in the output layer and improving the model's performance in RUL prediction. Finally, the effectiveness of the proposed method is validated through comparisons with other latest research works on the C-MAPSS dataset. The Dual-Mixer model demonstrates superiority across most metrics, while the FSGRI training method shows an average improvement of 7.00% and 2.41% in RMSE and MAPE, respectively, for all baseline models. Our experiments and model code are publicly available at https://github.com/fuen1590/PhmDeepLearningProjects.

{{</citation>}}


### (68/132) A Survey on Structure-Preserving Graph Transformers (Van Thuy Hoang et al., 2024)

{{<citation>}}

Van Thuy Hoang, O-Joun Lee. (2024)  
**A Survey on Structure-Preserving Graph Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.16176v1)  

---


**ABSTRACT**  
The transformer architecture has shown remarkable success in various domains, such as natural language processing and computer vision. When it comes to graph learning, transformers are required not only to capture the interactions between pairs of nodes but also to preserve graph structures connoting the underlying relations and proximity between them, showing the expressive power to capture different graph structures. Accordingly, various structure-preserving graph transformers have been proposed and widely used for various tasks, such as graph-level tasks in bioinformatics and chemoinformatics. However, strategies related to graph structure preservation have not been well organized and systematized in the literature. In this paper, we provide a comprehensive overview of structure-preserving graph transformers and generalize these methods from the perspective of their design objective. First, we divide strategies into four main groups: node feature modulation, context node sampling, graph rewriting, and transformer architecture improvements. We then further divide the strategies according to the coverage and goals of graph structure preservation. Furthermore, we also discuss challenges and future directions for graph transformer models to preserve the graph structure and understand the nature of graphs.

{{</citation>}}


### (69/132) X-PEFT: eXtremely Parameter-Efficient Fine-Tuning for Extreme Multi-Profile Scenarios (Namju Kwak et al., 2024)

{{<citation>}}

Namju Kwak, Taesup Kim. (2024)  
**X-PEFT: eXtremely Parameter-Efficient Fine-Tuning for Extreme Multi-Profile Scenarios**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GLUE  
[Paper Link](http://arxiv.org/abs/2401.16137v1)  

---


**ABSTRACT**  
Parameter-efficient fine-tuning (PEFT) techniques, such as adapter tuning, aim to fine-tune a pre-trained language model (PLM) using a minimal number of parameters for a specific task or profile. Although adapter tuning provides increased parameter efficiency compared to full-model fine-tuning, it introduces a small set of additional parameters attached to a PLM for each profile. This can become problematic in practical applications with multiple profiles, particularly when a significant increase in the number of profiles linearly boosts the total number of additional parameters. To mitigate this issue, we introduce X-PEFT, a novel PEFT method that leverages a multitude of given adapters by fine-tuning an extremely small set of compact tensors for a new profile, which serve as binary masks to adaptively select the given adapters. To efficiently validate our proposed method, we implement it using a large number of trained or untrained (random) adapters. We evaluate the performance of X-PEFT through LaMP and GLUE tasks and demonstrate that it either matches or surpasses the effectiveness of conventional adapter tuning, despite reducing the memory requirements per profile by a factor of 10,000 compared to it.

{{</citation>}}


### (70/132) Fairness in Algorithmic Recourse Through the Lens of Substantive Equality of Opportunity (Andrew Bell et al., 2024)

{{<citation>}}

Andrew Bell, Joao Fonseca, Carlo Abrate, Francesco Bonchi, Julia Stoyanovich. (2024)  
**Fairness in Algorithmic Recourse Through the Lens of Substantive Equality of Opportunity**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.16088v1)  

---


**ABSTRACT**  
Algorithmic recourse -- providing recommendations to those affected negatively by the outcome of an algorithmic system on how they can take action and change that outcome -- has gained attention as a means of giving persons agency in their interactions with artificial intelligence (AI) systems. Recent work has shown that even if an AI decision-making classifier is ``fair'' (according to some reasonable criteria), recourse itself may be unfair due to differences in the initial circumstances of individuals, compounding disparities for marginalized populations and requiring them to exert more effort than others. There is a need to define more methods and metrics for evaluating fairness in recourse that span a range of normative views of the world, and specifically those that take into account time. Time is a critical element in recourse because the longer it takes an individual to act, the more the setting may change due to model or data drift.   This paper seeks to close this research gap by proposing two notions of fairness in recourse that are in normative alignment with substantive equality of opportunity, and that consider time. The first considers the (often repeated) effort individuals exert per successful recourse event, and the second considers time per successful recourse event. Building upon an agent-based framework for simulating recourse, this paper demonstrates how much effort is needed to overcome disparities in initial circumstances. We then proposes an intervention to improve the fairness of recourse by rewarding effort, and compare it to existing strategies.

{{</citation>}}


### (71/132) Probabilistic Abduction for Visual Abstract Reasoning via Learning Rules in Vector-symbolic Architectures (Michael Hersche et al., 2024)

{{<citation>}}

Michael Hersche, Francesco di Stefano, Thomas Hofmann, Abu Sebastian, Abbas Rahimi. (2024)  
**Probabilistic Abduction for Visual Abstract Reasoning via Learning Rules in Vector-symbolic Architectures**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.16024v1)  

---


**ABSTRACT**  
Abstract reasoning is a cornerstone of human intelligence, and replicating it with artificial intelligence (AI) presents an ongoing challenge. This study focuses on efficiently solving Raven's progressive matrices (RPM), a visual test for assessing abstract reasoning abilities, by using distributed computation and operators provided by vector-symbolic architectures (VSA). Instead of hard-coding the rule formulations associated with RPMs, our approach can learn the VSA rule formulations (hence the name Learn-VRF) with just one pass through the training data. Yet, our approach, with compact parameters, remains transparent and interpretable. Learn-VRF yields accurate predictions on I-RAVEN's in-distribution data, and exhibits strong out-of-distribution capabilities concerning unseen attribute-rule pairs, significantly outperforming pure connectionist baselines including large language models. Our code is available at https://github.com/IBM/learn-vector-symbolic-architectures-rule-formulations.

{{</citation>}}


### (72/132) GPS: Graph Contrastive Learning via Multi-scale Augmented Views from Adversarial Pooling (Wei Ju et al., 2024)

{{<citation>}}

Wei Ju, Yiyang Gu, Zhengyang Mao, Ziyue Qiao, Yifang Qin, Xiao Luo, Hui Xiong, Ming Zhang. (2024)  
**GPS: Graph Contrastive Learning via Multi-scale Augmented Views from Adversarial Pooling**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.16011v1)  

---


**ABSTRACT**  
Self-supervised graph representation learning has recently shown considerable promise in a range of fields, including bioinformatics and social networks. A large number of graph contrastive learning approaches have shown promising performance for representation learning on graphs, which train models by maximizing agreement between original graphs and their augmented views (i.e., positive views). Unfortunately, these methods usually involve pre-defined augmentation strategies based on the knowledge of human experts. Moreover, these strategies may fail to generate challenging positive views to provide sufficient supervision signals. In this paper, we present a novel approach named Graph Pooling ContraSt (GPS) to address these issues. Motivated by the fact that graph pooling can adaptively coarsen the graph with the removal of redundancy, we rethink graph pooling and leverage it to automatically generate multi-scale positive views with varying emphasis on providing challenging positives and preserving semantics, i.e., strongly-augmented view and weakly-augmented view. Then, we incorporate both views into a joint contrastive learning framework with similarity learning and consistency learning, where our pooling module is adversarially trained with respect to the encoder for adversarial robustness. Experiments on twelve datasets on both graph classification and transfer learning tasks verify the superiority of the proposed method over its counterparts.

{{</citation>}}


### (73/132) Deep Embedding Clustering Driven by Sample Stability (Zhanwen Cheng et al., 2024)

{{<citation>}}

Zhanwen Cheng, Feijiang Li, Jieting Wang, Yuhua Qian. (2024)  
**Deep Embedding Clustering Driven by Sample Stability**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.15989v1)  

---


**ABSTRACT**  
Deep clustering methods improve the performance of clustering tasks by jointly optimizing deep representation learning and clustering. While numerous deep clustering algorithms have been proposed, most of them rely on artificially constructed pseudo targets for performing clustering. This construction process requires some prior knowledge, and it is challenging to determine a suitable pseudo target for clustering. To address this issue, we propose a deep embedding clustering algorithm driven by sample stability (DECS), which eliminates the requirement of pseudo targets. Specifically, we start by constructing the initial feature space with an autoencoder and then learn the cluster-oriented embedding feature constrained by sample stability. The sample stability aims to explore the deterministic relationship between samples and all cluster centroids, pulling samples to their respective clusters and keeping them away from other clusters with high determinacy. We analyzed the convergence of the loss using Lipschitz continuity in theory, which verifies the validity of the model. The experimental results on five datasets illustrate that the proposed method achieves superior performance compared to state-of-the-art clustering approaches.

{{</citation>}}


### (74/132) Effective Controllable Bias Mitigation for Classification and Retrieval using Gate Adapters (Shahed Masoudian et al., 2024)

{{<citation>}}

Shahed Masoudian, Cornelia Volaucnik, Markus Schedl, Shahed Masoudian. (2024)  
**Effective Controllable Bias Mitigation for Classification and Retrieval using Gate Adapters**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2401.16457v1)  

---


**ABSTRACT**  
Bias mitigation of Language Models has been the topic of many studies with a recent focus on learning separate modules like adapters for on-demand debiasing. Besides optimizing for a modularized debiased model, it is often critical in practice to control the degree of bias reduction at inference time, e.g., in order to tune for a desired performance-fairness trade-off in search results or to control the strength of debiasing in classification tasks. In this paper, we introduce Controllable Gate Adapter (ConGater), a novel modular gating mechanism with adjustable sensitivity parameters, which allows for a gradual transition from the biased state of the model to the fully debiased version at inference time. We demonstrate ConGater performance by (1) conducting adversarial debiasing experiments with three different models on three classification tasks with four protected attributes, and (2) reducing the bias of search results through fairness list-wise regularization to enable adjusting a trade-off between performance and fairness metrics. Our experiments on the classification tasks show that compared to baselines of the same caliber, ConGater can maintain higher task performance while containing less information regarding the attributes. Our results on the retrieval task show that the fully debiased ConGater can achieve the same fairness performance while maintaining more than twice as high task performance than recent strong baselines. Overall, besides strong performance ConGater enables the continuous transitioning between biased and debiased states of models, enhancing personalization of use and interpretability through controllability.

{{</citation>}}


### (75/132) Spatio-Temporal Attention Graph Neural Network for Remaining Useful Life Prediction (Zhixin Huang et al., 2024)

{{<citation>}}

Zhixin Huang, Yujiang He, Bernhard Sick. (2024)  
**Spatio-Temporal Attention Graph Neural Network for Remaining Useful Life Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2401.15964v1)  

---


**ABSTRACT**  
Remaining useful life prediction plays a crucial role in the health management of industrial systems. Given the increasing complexity of systems, data-driven predictive models have attracted significant research interest. Upon reviewing the existing literature, it appears that many studies either do not fully integrate both spatial and temporal features or employ only a single attention mechanism. Furthermore, there seems to be inconsistency in the choice of data normalization methods, particularly concerning operating conditions, which might influence predictive performance. To bridge these observations, this study presents the Spatio-Temporal Attention Graph Neural Network. Our model combines graph neural networks and temporal convolutional neural networks for spatial and temporal feature extraction, respectively. The cascade of these extractors, combined with multi-head attention mechanisms for both spatio-temporal dimensions, aims to improve predictive precision and refine model explainability. Comprehensive experiments were conducted on the C-MAPSS dataset to evaluate the impact of unified versus clustering normalization. The findings suggest that our model performs state-of-the-art results using only the unified normalization. Additionally, when dealing with datasets with multiple operating conditions, cluster normalization enhances the performance of our proposed model by up to 27%.

{{</citation>}}


### (76/132) Self-Supervised Learning in Event Sequences: A Comparative Study and Hybrid Approach of Generative Modeling and Contrastive Learning (Viktor Moskvoretskii et al., 2024)

{{<citation>}}

Viktor Moskvoretskii, Dmitry Osin, Egor Shvetsov, Igor Udovichenko, Maxim Zhelnin, Andrey Dukhovny, Anna Zhimerikina, Albert Efimov, Evgeny Burnaev. (2024)  
**Self-Supervised Learning in Event Sequences: A Comparative Study and Hybrid Approach of Generative Modeling and Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.15935v2)  

---


**ABSTRACT**  
This study investigates self-supervised learning techniques to obtain representations of Event Sequences. It is a key modality in various applications, including but not limited to banking, e-commerce, and healthcare.   We perform a comprehensive study of generative and contrastive approaches in self-supervised learning, applying them both independently. We find that there is no single supreme method. Consequently, we explore the potential benefits of combining these approaches. To achieve this goal, we introduce a novel method that aligns generative and contrastive embeddings as distinct modalities, drawing inspiration from contemporary multimodal research.   Generative and contrastive approaches are often treated as mutually exclusive, leaving a gap for their combined exploration. Our results demonstrate that this aligned model performs at least on par with, and mostly surpasses, existing methods and is more universal across a variety of tasks. Furthermore, we demonstrate that self-supervised methods consistently outperform the supervised approach on our datasets.

{{</citation>}}


### (77/132) Hybrid Transformer and Spatial-Temporal Self-Supervised Learning for Long-term Traffic Prediction (Wang Zhu et al., 2024)

{{<citation>}}

Wang Zhu, Doudou Zhang, Baichao Long, Jianli Xiao. (2024)  
**Hybrid Transformer and Spatial-Temporal Self-Supervised Learning for Long-term Traffic Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2401.16453v1)  

---


**ABSTRACT**  
Long-term traffic prediction has always been a challenging task due to its dynamic temporal dependencies and complex spatial dependencies. In this paper, we propose a model that combines hybrid Transformer and spatio-temporal self-supervised learning. The model enhances its robustness by applying adaptive data augmentation techniques at the sequence-level and graph-level of the traffic data. It utilizes Transformer to overcome the limitations of recurrent neural networks in capturing long-term sequences, and employs Chebyshev polynomial graph convolution to capture complex spatial dependencies. Furthermore, considering the impact of spatio-temporal heterogeneity on traffic speed, we design two self-supervised learning tasks to model the temporal and spatial heterogeneity, thereby improving the accuracy and generalization ability of the model. Experimental evaluations are conducted on two real-world datasets, PeMS04 and PeMS08, and the results are visualized and analyzed, demonstrating the superior performance of the proposed model.

{{</citation>}}


### (78/132) Context-Former: Stitching via Latent Conditioned Sequence Modeling (Ziqi Zhang et al., 2024)

{{<citation>}}

Ziqi Zhang, Jingzehua Xu, Zifeng Zhuang, Jinxin Liu, Donglin wang. (2024)  
**Context-Former: Stitching via Latent Conditioned Sequence Modeling**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.16452v1)  

---


**ABSTRACT**  
Offline reinforcement learning (RL) algorithms can improve the decision making via stitching sub-optimal trajectories to obtain more optimal ones. This capability is a crucial factor in enabling RL to learn policies that are superior to the behavioral policy. On the other hand, Decision Transformer (DT) abstracts the decision-making as sequence modeling, showcasing competitive performance on offline RL benchmarks, however, recent studies demonstrate that DT lacks of stitching capability, thus exploit stitching capability for DT is vital to further improve its performance. In order to endow stitching capability to DT, we abstract trajectory stitching as expert matching and introduce our approach, ContextFormer, which integrates contextual information-based imitation learning (IL) and sequence modeling to stitch sub-optimal trajectory fragments by emulating the representations of a limited number of expert trajectories. To validate our claim, we conduct experiments from two perspectives: 1) We conduct extensive experiments on D4RL benchmarks under the settings of IL, and experimental results demonstrate ContextFormer can achieve competitive performance in multi-IL settings. 2) More importantly, we conduct a comparison of ContextFormer with diverse competitive DT variants using identical training datasets. The experimental results unveiled ContextFormer's superiority, as it outperformed all other variants, showcasing its remarkable performance.

{{</citation>}}


### (79/132) A Gated MLP Architecture for Learning Topological Dependencies in Spatio-Temporal Graphs (Yun Young Choi et al., 2024)

{{<citation>}}

Yun Young Choi, Minho Lee, Sun Woo Park, Seunghwan Lee, Joohwan Ko. (2024)  
**A Gated MLP Architecture for Learning Topological Dependencies in Spatio-Temporal Graphs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Transformer  
[Paper Link](http://arxiv.org/abs/2401.15894v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) and Transformer have been increasingly adopted to learn the complex vector representations of spatio-temporal graphs, capturing intricate spatio-temporal dependencies crucial for applications such as traffic datasets. Although many existing methods utilize multi-head attention mechanisms and message-passing neural networks (MPNNs) to capture both spatial and temporal relations, these approaches encode temporal and spatial relations independently, and reflect the graph's topological characteristics in a limited manner. In this work, we introduce the Cycle to Mixer (Cy2Mixer), a novel spatio-temporal GNN based on topological non-trivial invariants of spatio-temporal graphs with gated multi-layer perceptrons (gMLP). The Cy2Mixer is composed of three blocks based on MLPs: A message-passing block for encapsulating spatial information, a cycle message-passing block for enriching topological information through cyclic subgraphs, and a temporal block for capturing temporal properties. We bolster the effectiveness of Cy2Mixer with mathematical evidence emphasizing that our cycle message-passing block is capable of offering differentiated information to the deep learning model compared to the message-passing block. Furthermore, empirical evaluations substantiate the efficacy of the Cy2Mixer, demonstrating state-of-the-art performances across various traffic benchmark datasets.

{{</citation>}}


### (80/132) lil'HDoC: An Algorithm for Good Arm Identification under Small Threshold Gap (Tzu-Hsien Tsai et al., 2024)

{{<citation>}}

Tzu-Hsien Tsai, Yun-Da Tsai, Shou-De Lin. (2024)  
**lil'HDoC: An Algorithm for Good Arm Identification under Small Threshold Gap**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15879v1)  

---


**ABSTRACT**  
Good arm identification (GAI) is a pure-exploration bandit problem in which a single learner outputs an arm as soon as it is identified as a good arm. A good arm is defined as an arm with an expected reward greater than or equal to a given threshold. This paper focuses on the GAI problem under a small threshold gap, which refers to the distance between the expected rewards of arms and the given threshold. We propose a new algorithm called lil'HDoC to significantly improve the total sample complexity of the HDoC algorithm. We demonstrate that the sample complexity of the first $\lambda$ output arm in lil'HDoC is bounded by the original HDoC algorithm, except for one negligible term, when the distance between the expected reward and threshold is small. Extensive experiments confirm that our algorithm outperforms the state-of-the-art algorithms in both synthetic and real-world datasets.

{{</citation>}}


### (81/132) Look Around! Unexpected gains from training on environments in the vicinity of the target (Serena Bono et al., 2024)

{{<citation>}}

Serena Bono, Spandan Madan, Ishaan Grover, Mao Yasueda, Cynthia Breazeal, Hanspeter Pfister, Gabriel Kreiman. (2024)  
**Look Around! Unexpected gains from training on environments in the vicinity of the target**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.15856v1)  

---


**ABSTRACT**  
Solutions to Markov Decision Processes (MDP) are often very sensitive to state transition probabilities. As the estimation of these probabilities is often inaccurate in practice, it is important to understand when and how Reinforcement Learning (RL) agents generalize when transition probabilities change. Here we present a new methodology to evaluate such generalization of RL agents under small shifts in the transition probabilities. Specifically, we evaluate agents in new environments (MDPs) in the vicinity of the training MDP created by adding quantifiable, parametric noise into the transition function of the training MDP. We refer to this process as Noise Injection, and the resulting environments as $\delta$-environments. This process allows us to create controlled variations of the same environment with the level of the noise serving as a metric of distance between environments. Conventional wisdom suggests that training and testing on the same MDP should yield the best results. However, we report several cases of the opposite -- when targeting a specific environment, training the agent in an alternative noise setting can yield superior outcomes. We showcase this phenomenon across $60$ different variations of ATARI games, including PacMan, Pong, and Breakout.

{{</citation>}}


## cs.AR (1)



### (82/132) FPGA Technology Mapping Using Sketch-Guided Program Synthesis (Gus Henry Smith et al., 2024)

{{<citation>}}

Gus Henry Smith, Ben Kushigian, Vishal Canumalla, Andrew Cheung, Steven Lyubomirsky, Sorawee Porncharoenwase, René Just, Gilbert Louis Bernstein, Zachary Tatlock. (2024)  
**FPGA Technology Mapping Using Sketch-Guided Program Synthesis**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-PL, cs.AR  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2401.16526v1)  

---


**ABSTRACT**  
FPGA technology mapping is the process of implementing a hardware design expressed in high-level HDL (hardware design language) code using the low-level, architecture-specific primitives of the target FPGA. As FPGAs become increasingly heterogeneous, achieving high performance requires hardware synthesis tools that better support mapping to complex, highly configurable primitives like digital signal processors (DSPs). Current tools support DSP mapping via handwritten special-case mapping rules, which are laborious to write, error-prone, and often overlook mapping opportunities. We introduce Lakeroad, a principled approach to technology mapping via sketch-guided program synthesis. Lakeroad leverages two techniques -- architecture-independent sketch templates and semantics extraction from HDL -- to provide extensible technology mapping with stronger correctness guarantees and higher coverage of mapping opportunities than state-of-the-art tools. Across representative microbenchmarks, Lakeroad produces 2--3.5$\times$ the number of optimal mappings compared to proprietary state-of-the-art tools and 6--44$\times$ the number of optimal mappings compared to popular open-source tools, while also providing correctness guarantees not given by any other tool.

{{</citation>}}


## cs.ET (2)



### (83/132) Dynamic Electro-Optic Analog Memory for Neuromorphic Photonic Computing (Sean Lam et al., 2024)

{{<citation>}}

Sean Lam, Ahmed Khaled, Simon Bilodeau, Bicky A. Marquez, Paul R. Prucnal, Lukas Chrostowski, Bhavin J. Shastri, Sudip Shekhar. (2024)  
**Dynamic Electro-Optic Analog Memory for Neuromorphic Photonic Computing**  

---
Primary Category: cs.ET  
Categories: cs-ET, cs-SY, cs.ET, eess-SP, eess-SY, physics-optics  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.16515v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) has seen remarkable advancements across various domains, including natural language processing, computer vision, autonomous vehicles, and biology. However, the rapid expansion of AI technologies has escalated the demand for more powerful computing resources. As digital computing approaches fundamental limits, neuromorphic photonics emerges as a promising platform to complement existing digital systems. In neuromorphic photonic computing, photonic devices are controlled using analog signals. This necessitates the use of digital-to-analog converters (DAC) and analog-to-digital converters (ADC) for interfacing with these devices during inference and training. However, data movement between memory and these converters in conventional von Neumann computing architectures consumes energy. To address this, analog memory co-located with photonic computing devices is proposed. This approach aims to reduce the reliance on DACs and ADCs and minimize data movement to enhance compute efficiency. This paper demonstrates a monolithically integrated neuromorphic photonic circuit with co-located capacitive analog memory and compares various analog memory technologies for neuromorphic photonic computing using the MNIST dataset as a benchmark.

{{</citation>}}


### (84/132) Error Mitigation for Thermodynamic Computing (Maxwell Aifer et al., 2024)

{{<citation>}}

Maxwell Aifer, Denis Melanson, Kaelan Donatella, Gavin Crooks, Thomas Ahle, Patrick J. Coles. (2024)  
**Error Mitigation for Thermodynamic Computing**  

---
Primary Category: cs.ET  
Categories: cond-mat-stat-mech, cs-ET, cs.ET, quant-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.16231v1)  

---


**ABSTRACT**  
While physics-based computing can offer speed and energy efficiency compared to digital computing, it also is subject to errors that must be mitigated. For example, many error mitigation methods have been proposed for quantum computing. However this error mitigation framework has yet to be applied to other physics-based computing paradigms. In this work, we consider thermodynamic computing, which has recently captured attention due to its relevance to artificial intelligence (AI) applications, such as probabilistic AI and generative AI. A key source of errors in this paradigm is the imprecision of the analog hardware components. Here, we introduce a method that reduces the overall error from a linear to a quadratic dependence (from $\epsilon$ to $\epsilon^2$) on the imprecision $\epsilon$, for Gaussian sampling and linear algebra applications. The method involves sampling from an ensemble of imprecise distributions associated with various rounding events and then merging these samples. We numerically demonstrate the scalability of this method for dimensions greater than 1000. Finally, we implement this method on an actual thermodynamic computer and show $20\%$ error reduction for matrix inversion; the first thermodynamic error mitigation experiment.

{{</citation>}}


## cs.HC (4)



### (85/132) Dissecting users' needs for search result explanations (Prerna Juneja et al., 2024)

{{<citation>}}

Prerna Juneja, Wenjuan Zhang, Alison Marie Smith-Renner, Hemank Lamba, Joel Tetreault, Alex Jaimes. (2024)  
**Dissecting users' needs for search result explanations**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-IR, cs.HC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.16509v1)  

---


**ABSTRACT**  
There is a growing demand for transparency in search engines to understand how search results are curated and to enhance users' trust. Prior research has introduced search result explanations with a focus on how to explain, assuming explanations are beneficial. Our study takes a step back to examine if search explanations are needed and when they are likely to provide benefits. Additionally, we summarize key characteristics of helpful explanations and share users' perspectives on explanation features provided by Google and Bing. Interviews with non-technical individuals reveal that users do not always seek or understand search explanations and mostly desire them for complex and critical tasks. They find Google's search explanations too obvious but appreciate the ability to contest search results. Based on our findings, we offer design recommendations for search engines and explanations to help users better evaluate search results and enhance their search experience.

{{</citation>}}


### (86/132) 'You tell me': A Dataset of GPT-4-Based Behaviour Change Support Conversations (Selina Meyer et al., 2024)

{{<citation>}}

Selina Meyer, David Elsweiler. (2024)  
**'You tell me': A Dataset of GPT-4-Based Behaviour Change Support Conversations**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-HC, cs.HC  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.16167v1)  

---


**ABSTRACT**  
Conversational agents are increasingly used to address emotional needs on top of information needs. One use case of increasing interest are counselling-style mental health and behaviour change interventions, with large language model (LLM)-based approaches becoming more popular. Research in this context so far has been largely system-focused, foregoing the aspect of user behaviour and the impact this can have on LLM-generated texts. To address this issue, we share a dataset containing text-based user interactions related to behaviour change with two GPT-4-based conversational agents collected in a preregistered user study. This dataset includes conversation data, user language analysis, perception measures, and user feedback for LLM-generated turns, and can offer valuable insights to inform the design of such systems based on real interactions.

{{</citation>}}


### (87/132) KAUCUS: Knowledge Augmented User Simulators for Training Language Model Assistants (Kaustubh D. Dhole, 2024)

{{<citation>}}

Kaustubh D. Dhole. (2024)  
**KAUCUS: Knowledge Augmented User Simulators for Training Language Model Assistants**  

---
Primary Category: cs.HC  
Categories: I-2-7; H-3-3, cs-AI, cs-CL, cs-HC, cs-IR, cs.HC  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.16454v1)  

---


**ABSTRACT**  
An effective multi-turn instruction-following assistant can be developed by creating a simulator that can generate useful interaction data. Apart from relying on its intrinsic weights, an ideal user simulator should also be able to bootstrap external knowledge rapidly in its raw form to simulate the multifarious diversity of text available over the internet. Previous user simulators generally lacked diversity, were mostly closed domain, and necessitated rigid schema making them inefficient to rapidly scale to incorporate external knowledge. In this regard, we introduce, Kaucus, a Knowledge-Augmented User Simulator framework, to outline a process of creating diverse user simulators, that can seamlessly exploit external knowledge as well as benefit downstream assistant model training. Through two GPT-J based simulators viz., a Retrieval Augmented Simulator and a Summary Controlled Simulator we generate diverse simulator-assistant interactions. Through reward and preference model-based evaluations, we find that these interactions serve as useful training data and create more helpful downstream assistants. We also find that incorporating knowledge through retrieval augmentation or summary control helps create better assistants.

{{</citation>}}


### (88/132) 3DPFIX: Improving Remote Novices' 3D Printing Troubleshooting through Human-AI Collaboration (Nahyun Kwon et al., 2024)

{{<citation>}}

Nahyun Kwon, Tong Sun, Yuyang Gao, Liang Zhao, Xu Wang, Jeeeun Kim, Sungsoo Ray Hong. (2024)  
**3DPFIX: Improving Remote Novices' 3D Printing Troubleshooting through Human-AI Collaboration**  

---
Primary Category: cs.HC  
Categories: cs-CV, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15877v1)  

---


**ABSTRACT**  
The widespread consumer-grade 3D printers and learning resources online enable novices to self-train in remote settings. While troubleshooting plays an essential part of 3D printing, the process remains challenging for many remote novices even with the help of well-developed online sources, such as online troubleshooting archives and online community help. We conducted a formative study with 76 active 3D printing users to learn how remote novices leverage online resources in troubleshooting and their challenges. We found that remote novices cannot fully utilize online resources. For example, the online archives statically provide general information, making it hard to search and relate their unique cases with existing descriptions. Online communities can potentially ease their struggles by providing more targeted suggestions, but a helper who can provide custom help is rather scarce, making it hard to obtain timely assistance. We propose 3DPFIX, an interactive 3D troubleshooting system powered by the pipeline to facilitate Human-AI Collaboration, designed to improve novices' 3D printing experiences and thus help them easily accumulate their domain knowledge. We built 3DPFIX that supports automated diagnosis and solution-seeking. 3DPFIX was built upon shared dialogues about failure cases from Q\&A discourses accumulated in online communities. We leverage social annotations (i.e., comments) to build an annotated failure image dataset for AI classifiers and extract a solution pool. Our summative study revealed that using 3DPFIX helped participants spend significantly less effort in diagnosing failures and finding a more accurate solution than relying on their common practice. We also found that 3DPFIX users learn about 3D printing domain-specific knowledge. We discuss the implications of leveraging community-driven data in developing future Human-AI Collaboration designs.

{{</citation>}}


## cs.CV (25)



### (89/132) Computer Vision for Primate Behavior Analysis in the Wild (Richard Vogg et al., 2024)

{{<citation>}}

Richard Vogg, Timo Lüddecke, Jonathan Henrich, Sharmita Dey, Matthias Nuske, Valentin Hassler, Derek Murphy, Julia Fischer, Julia Ostner, Oliver Schülke, Peter M. Kappeler, Claudia Fichtel, Alexander Gail, Stefan Treue, Hansjörg Scherberger, Florentin Wörgötter, Alexander S. Ecker. (2024)  
**Computer Vision for Primate Behavior Analysis in the Wild**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, q-bio-QM  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2401.16424v1)  

---


**ABSTRACT**  
Advances in computer vision as well as increasingly widespread video-based behavioral monitoring have great potential for transforming how we study animal cognition and behavior. However, there is still a fairly large gap between the exciting prospects and what can actually be achieved in practice today, especially in videos from the wild. With this perspective paper, we want to contribute towards closing this gap, by guiding behavioral scientists in what can be expected from current methods and steering computer vision researchers towards problems that are relevant to advance research in animal behavior. We start with a survey of the state-of-the-art methods for computer vision problems that are directly relevant to the video-based study of animal behavior, including object detection, multi-individual tracking, (inter)action recognition and individual identification. We then review methods for effort-efficient learning, which is one of the biggest challenges from a practical perspective. Finally, we close with an outlook into the future of the emerging field of computer vision for animal behavior, where we argue that the field should move fast beyond the common frame-by-frame processing and treat video as a first-class citizen.

{{</citation>}}


### (90/132) InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model (Xiaoyi Dong et al., 2024)

{{<citation>}}

Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, Jiaqi Wang. (2024)  
**InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.16420v1)  

---


**ABSTRACT**  
We introduce InternLM-XComposer2, a cutting-edge vision-language model excelling in free-form text-image composition and comprehension. This model goes beyond conventional vision-language understanding, adeptly crafting interleaved text-image content from diverse inputs like outlines, detailed textual specifications, and reference images, enabling highly customizable content creation. InternLM-XComposer2 proposes a Partial LoRA (PLoRA) approach that applies additional LoRA parameters exclusively to image tokens to preserve the integrity of pre-trained language knowledge, striking a balance between precise vision understanding and text composition with literary talent. Experimental results demonstrate the superiority of InternLM-XComposer2 based on InternLM2-7B in producing high-quality long-text multi-modal content and its exceptional vision-language understanding performance across various benchmarks, where it not only significantly outperforms existing multimodal models but also matches or even surpasses GPT-4V and Gemini Pro in certain assessments. This highlights its remarkable proficiency in the realm of multimodal understanding. The InternLM-XComposer2 model series with 7B parameters are publicly available at https://github.com/InternLM/InternLM-XComposer.

{{</citation>}}


### (91/132) A Survey on Visual Anomaly Detection: Challenge, Approach, and Prospect (Yunkang Cao et al., 2024)

{{<citation>}}

Yunkang Cao, Xiaohao Xu, Jiangning Zhang, Yuqi Cheng, Xiaonan Huang, Guansong Pang, Weiming Shen. (2024)  
**A Survey on Visual Anomaly Detection: Challenge, Approach, and Prospect**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2401.16402v1)  

---


**ABSTRACT**  
Visual Anomaly Detection (VAD) endeavors to pinpoint deviations from the concept of normality in visual data, widely applied across diverse domains, e.g., industrial defect inspection, and medical lesion detection. This survey comprehensively examines recent advancements in VAD by identifying three primary challenges: 1) scarcity of training data, 2) diversity of visual modalities, and 3) complexity of hierarchical anomalies. Starting with a brief overview of the VAD background and its generic concept definitions, we progressively categorize, emphasize, and discuss the latest VAD progress from the perspective of sample number, data modality, and anomaly hierarchy. Through an in-depth analysis of the VAD field, we finally summarize future developments for VAD and conclude the key findings and contributions of this survey.

{{</citation>}}


### (92/132) Amazon's 2023 Drought: Sentinel-1 Reveals Extreme Rio Negro River Contraction (Fabien H Wagner et al., 2024)

{{<citation>}}

Fabien H Wagner, Samuel Favrichon, Ricardo Dalagnol, Mayumi CM Hirye, Adugna Mullissa, Sassan Saatchi. (2024)  
**Amazon's 2023 Drought: Sentinel-1 Reveals Extreme Rio Negro River Contraction**  

---
Primary Category: cs.CV  
Categories: 92F05, I-4-6, cs-CV, cs.CV  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2401.16393v1)  

---


**ABSTRACT**  
The Amazon, the world's largest rainforest, faces a severe historic drought. The Rio Negro River, one of the major Amazon River tributaries, reaches its lowest level in a century in October 2023. Here, we used a U-net deep learning model to map water surfaces in the Rio Negro River basin every 12 days in 2022 and 2023 using 10 m spatial resolution Sentinel-1 satellite radar images. The accuracy of the water surface model was high with an F1-score of 0.93. The 12 days mosaic time series of water surface was generated from the Sentinel-1 prediction. The water surface mask demonstrated relatively consistent agreement with the Global Surface Water (GSW) product from Joint Research Centre (F1-score: 0.708) and with the Brazilian Mapbiomas Water initiative (F1-score: 0.686). The main errors of the map were omission errors in flooded woodland, in flooded shrub and because of clouds. Rio Negro water surfaces reached their lowest level around the 25th of November 2023 and were reduced to 68.1\% (9,559.9 km$^2$) of the maximum water surfaces observed in the period 2022-2023 (14,036.3 km$^2$). Synthetic Aperture Radar (SAR) data, in conjunction with deep learning techniques, can significantly improve near real-time mapping of water surface in tropical regions.

{{</citation>}}


### (93/132) PathMMU: A Massive Multimodal Expert-Level Benchmark for Understanding and Reasoning in Pathology (Yuxuan Sun et al., 2024)

{{<citation>}}

Yuxuan Sun, Hao Wu, Chenglu Zhu, Sunyi Zheng, Qizi Chen, Kai Zhang, Yunlong Zhang, Xiaoxiao Lan, Mengyue Zheng, Jingxiong Li, Xinheng Lyu, Tao Lin, Lin Yang. (2024)  
**PathMMU: A Massive Multimodal Expert-Level Benchmark for Understanding and Reasoning in Pathology**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, GPT, GPT-4, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.16355v1)  

---


**ABSTRACT**  
The emergence of large multimodal models has unlocked remarkable potential in AI, particularly in pathology. However, the lack of specialized, high-quality benchmark impeded their development and precise evaluation. To address this, we introduce PathMMU, the largest and highest-quality expert-validated pathology benchmark for LMMs. It comprises 33,573 multimodal multi-choice questions and 21,599 images from various sources, and an explanation for the correct answer accompanies each question. The construction of PathMMU capitalizes on the robust capabilities of GPT-4V, utilizing approximately 30,000 gathered image-caption pairs to generate Q\&As. Significantly, to maximize PathMMU's authority, we invite six pathologists to scrutinize each question under strict standards in PathMMU's validation and test sets, while simultaneously setting an expert-level performance benchmark for PathMMU. We conduct extensive evaluations, including zero-shot assessments of 14 open-sourced and three closed-sourced LMMs and their robustness to image corruption. We also fine-tune representative LMMs to assess their adaptability to PathMMU. The empirical findings indicate that advanced LMMs struggle with the challenging PathMMU benchmark, with the top-performing LMM, GPT-4V, achieving only a 51.7\% zero-shot performance, significantly lower than the 71.4\% demonstrated by human pathologists. After fine-tuning, even open-sourced LMMs can surpass GPT-4V with a performance of over 60\%, but still fall short of the expertise shown by pathologists. We hope that the PathMMU will offer valuable insights and foster the development of more specialized, next-generation LLMs for pathology.

{{</citation>}}


### (94/132) Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization (Guang Lin et al., 2024)

{{<citation>}}

Guang Lin, Chao Li, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao. (2024)  
**Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Adversarial Training, ImageNet  
[Paper Link](http://arxiv.org/abs/2401.16352v1)  

---


**ABSTRACT**  
The deep neural networks are known to be vulnerable to well-designed adversarial attacks. The most successful defense technique based on adversarial training (AT) can achieve optimal robustness against particular attacks but cannot generalize well to unseen attacks. Another effective defense technique based on adversarial purification (AP) can enhance generalization but cannot achieve optimal robustness. Meanwhile, both methods share one common limitation on the degraded standard accuracy. To mitigate these issues, we propose a novel framework called Adversarial Training on Purification (AToP), which comprises two components: perturbation destruction by random transforms (RT) and purifier model fine-tuned (FT) by adversarial loss. RT is essential to avoid overlearning to known attacks resulting in the robustness generalization to unseen attacks and FT is essential for the improvement of robustness. To evaluate our method in an efficient and scalable way, we conduct extensive experiments on CIFAR-10, CIFAR-100, and ImageNette to demonstrate that our method achieves state-of-the-art results and exhibits generalization ability against unseen attacks.

{{</citation>}}


### (95/132) MixSup: Mixed-grained Supervision for Label-efficient LiDAR-based 3D Object Detection (Yuxue Yang et al., 2024)

{{<citation>}}

Yuxue Yang, Lue Fan, Zhaoxiang Zhang. (2024)  
**MixSup: Mixed-grained Supervision for Label-efficient LiDAR-based 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.16305v1)  

---


**ABSTRACT**  
Label-efficient LiDAR-based 3D object detection is currently dominated by weakly/semi-supervised methods. Instead of exclusively following one of them, we propose MixSup, a more practical paradigm simultaneously utilizing massive cheap coarse labels and a limited number of accurate labels for Mixed-grained Supervision. We start by observing that point clouds are usually textureless, making it hard to learn semantics. However, point clouds are geometrically rich and scale-invariant to the distances from sensors, making it relatively easy to learn the geometry of objects, such as poses and shapes. Thus, MixSup leverages massive coarse cluster-level labels to learn semantics and a few expensive box-level labels to learn accurate poses and shapes. We redesign the label assignment in mainstream detectors, which allows them seamlessly integrated into MixSup, enabling practicality and universality. We validate its effectiveness in nuScenes, Waymo Open Dataset, and KITTI, employing various detectors. MixSup achieves up to 97.31% of fully supervised performance, using cheap cluster annotations and only 10% box annotations. Furthermore, we propose PointSAM based on the Segment Anything Model for automated coarse labeling, further reducing the annotation burden. The code is available at https://github.com/BraveGroup/PointSAM-for-MixSup.

{{</citation>}}


### (96/132) Regressing Transformers for Data-efficient Visual Place Recognition (María Leyva-Vallina et al., 2024)

{{<citation>}}

María Leyva-Vallina, Nicola Strisciuglio, Nicolai Petkov. (2024)  
**Regressing Transformers for Data-efficient Visual Place Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.16304v1)  

---


**ABSTRACT**  
Visual place recognition is a critical task in computer vision, especially for localization and navigation systems. Existing methods often rely on contrastive learning: image descriptors are trained to have small distance for similar images and larger distance for dissimilar ones in a latent space. However, this approach struggles to ensure accurate distance-based image similarity representation, particularly when training with binary pairwise labels, and complex re-ranking strategies are required. This work introduces a fresh perspective by framing place recognition as a regression problem, using camera field-of-view overlap as similarity ground truth for learning. By optimizing image descriptors to align directly with graded similarity labels, this approach enhances ranking capabilities without expensive re-ranking, offering data-efficient training and strong generalization across several benchmark datasets.

{{</citation>}}


### (97/132) Breaking the Barrier: Selective Uncertainty-based Active Learning for Medical Image Segmentation (Siteng Ma et al., 2024)

{{<citation>}}

Siteng Ma, Haochang Wu, Aonghus Lawlor, Ruihai Dong. (2024)  
**Breaking the Barrier: Selective Uncertainty-based Active Learning for Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2401.16298v1)  

---


**ABSTRACT**  
Active learning (AL) has found wide applications in medical image segmentation, aiming to alleviate the annotation workload and enhance performance. Conventional uncertainty-based AL methods, such as entropy and Bayesian, often rely on an aggregate of all pixel-level metrics. However, in imbalanced settings, these methods tend to neglect the significance of target regions, eg., lesions, and tumors. Moreover, uncertainty-based selection introduces redundancy. These factors lead to unsatisfactory performance, and in many cases, even underperform random sampling. To solve this problem, we introduce a novel approach called the Selective Uncertainty-based AL, avoiding the conventional practice of summing up the metrics of all pixels. Through a filtering process, our strategy prioritizes pixels within target areas and those near decision boundaries. This resolves the aforementioned disregard for target areas and redundancy. Our method showed substantial improvements across five different uncertainty-based methods and two distinct datasets, utilizing fewer labeled data to reach the supervised baseline and consistently achieving the highest overall performance. Our code is available at https://github.com/HelenMa9998/Selective\_Uncertainty\_AL.

{{</citation>}}


### (98/132) DressCode: Autoregressively Sewing and Generating Garments from Text Guidance (Kai He et al., 2024)

{{<citation>}}

Kai He, Kaixin Yao, Qixuan Zhang, Jingyi Yu, Lingjie Liu, Lan Xu. (2024)  
**DressCode: Autoregressively Sewing and Generating Garments from Text Guidance**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2401.16465v1)  

---


**ABSTRACT**  
Apparel's significant role in human appearance underscores the importance of garment digitalization for digital human creation. Recent advances in 3D content creation are pivotal for digital human creation. Nonetheless, garment generation from text guidance is still nascent. We introduce a text-driven 3D garment generation framework, DressCode, which aims to democratize design for novices and offer immense potential in fashion design, virtual try-on, and digital human creation. For our framework, we first introduce SewingGPT, a GPT-based architecture integrating cross-attention with text-conditioned embedding to generate sewing patterns with text guidance. We also tailored a pre-trained Stable Diffusion for high-quality, tile-based PBR texture generation. By leveraging a large language model, our framework generates CG-friendly garments through natural language interaction. Our method also facilitates pattern completion and texture editing, simplifying the process for designers by user-friendly interaction. With comprehensive evaluations and comparisons with other state-of-the-art methods, our method showcases the best quality and alignment with input prompts. User studies further validate our high-quality rendering results, highlighting its practical utility and potential in production settings.

{{</citation>}}


### (99/132) LLaVA-MoLE: Sparse Mixture of LoRA Experts for Mitigating Data Conflicts in Instruction Finetuning MLLMs (Shaoxiang Chen et al., 2024)

{{<citation>}}

Shaoxiang Chen, Zequn Jie, Lin Ma. (2024)  
**LLaVA-MoLE: Sparse Mixture of LoRA Experts for Mitigating Data Conflicts in Instruction Finetuning MLLMs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.16160v2)  

---


**ABSTRACT**  
Instruction finetuning on a variety of image-text instruction data is the key to obtaining a versatile Multimodal Large Language Model (MLLM), and different configurations of the instruction data can lead to finetuned models with different capabilities. However, we have discovered that data conflicts are inevitable when mixing instruction data from distinct domains, which can result in performance drops for tasks of a specific domain. To address this issue, we propose to apply an efficient Mixture of Experts (MoE) design, which is a sparse Mixture of LoRA Experts (MoLE) for instruction finetuning MLLMs. Within the Transformer layers, we extend the popular Low-Rank Adaption (LoRA) method by creating a set of LoRA experts specifically for the MLP layer, and route each token to the top-1 expert based on a routing function, allowing adaptive choices for tokens from different domains. Since the LoRA experts are sparsely activated, the training and inference cost are kept roughly constant compared to the original LoRA method. By replacing the plain-LoRA of LLaVA-1.5 with our MoE design, our final model is named LLaVA-MoLE. Extensive experiments proved that LLaVA-MoLE effectively mitigates the data conflict issue when mixing multiple distinct instruction datasets with various configurations, and achieves consistent performance gains over the strong plain-LoRA baselines. Most importantly, on the mixed datasets, LLaVA-MoLE can even outperform the plain-LoRA baseline trained with twice the samples.

{{</citation>}}


### (100/132) CIMIL-CRC: a clinically-informed multiple instance learning framework for patient-level colorectal cancer molecular subtypes classification from H\&E stained images (Hadar Hezi et al., 2024)

{{<citation>}}

Hadar Hezi, Matan Gelber, Alexander Balabanov, Yosef E. Maruvka, Moti Freiman. (2024)  
**CIMIL-CRC: a clinically-informed multiple instance learning framework for patient-level colorectal cancer molecular subtypes classification from H\&E stained images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2401.16131v1)  

---


**ABSTRACT**  
Treatment approaches for colorectal cancer (CRC) are highly dependent on the molecular subtype, as immunotherapy has shown efficacy in cases with microsatellite instability (MSI) but is ineffective for the microsatellite stable (MSS) subtype. There is promising potential in utilizing deep neural networks (DNNs) to automate the differentiation of CRC subtypes by analyzing Hematoxylin and Eosin (H\&E) stained whole-slide images (WSIs). Due to the extensive size of WSIs, Multiple Instance Learning (MIL) techniques are typically explored. However, existing MIL methods focus on identifying the most representative image patches for classification, which may result in the loss of critical information. Additionally, these methods often overlook clinically relevant information, like the tendency for MSI class tumors to predominantly occur on the proximal (right side) colon. We introduce `CIMIL-CRC', a DNN framework that: 1) solves the MSI/MSS MIL problem by efficiently combining a pre-trained feature extraction model with principal component analysis (PCA) to aggregate information from all patches, and 2) integrates clinical priors, particularly the tumor location within the colon, into the model to enhance patient-level classification accuracy. We assessed our CIMIL-CRC method using the average area under the curve (AUC) from a 5-fold cross-validation experimental setup for model development on the TCGA-CRC-DX cohort, contrasting it with a baseline patch-level classification, MIL-only approach, and Clinically-informed patch-level classification approach. Our CIMIL-CRC outperformed all methods (AUROC: $0.92\pm0.002$ (95\% CI 0.91-0.92), vs. $0.79\pm0.02$ (95\% CI 0.76-0.82), $0.86\pm0.01$ (95\% CI 0.85-0.88), and $0.87\pm0.01$ (95\% CI 0.86-0.88), respectively). The improvement was statistically significant.

{{</citation>}}


### (101/132) Towards Scenario Generalization for Vision-based Roadside 3D Object Detection (Lei Yang et al., 2024)

{{<citation>}}

Lei Yang, Xinyu Zhang, Jun Li, Li Wang, Chuang Zhang, Li Ju, Zhiwei Li, Yang Shen. (2024)  
**Towards Scenario Generalization for Vision-based Roadside 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Object Detection  
[Paper Link](http://arxiv.org/abs/2401.16110v1)  

---


**ABSTRACT**  
Roadside perception can greatly increase the safety of autonomous vehicles by extending their perception ability beyond the visual range and addressing blind spots. However, current state-of-the-art vision-based roadside detection methods possess high accuracy on labeled scenes but have inferior performance on new scenes. This is because roadside cameras remain stationary after installation and can only collect data from a single scene, resulting in the algorithm overfitting these roadside backgrounds and camera poses. To address this issue, in this paper, we propose an innovative Scenario Generalization Framework for Vision-based Roadside 3D Object Detection, dubbed SGV3D. Specifically, we employ a Background-suppressed Module (BSM) to mitigate background overfitting in vision-centric pipelines by attenuating background features during the 2D to bird's-eye-view projection. Furthermore, by introducing the Semi-supervised Data Generation Pipeline (SSDG) using unlabeled images from new scenes, diverse instance foregrounds with varying camera poses are generated, addressing the risk of overfitting specific camera poses. We evaluate our method on two large-scale roadside benchmarks. Our method surpasses all previous methods by a significant margin in new scenes, including +42.57% for vehicle, +5.87% for pedestrian, and +14.89% for cyclist compared to BEVHeight on the DAIR-V2X-I heterologous benchmark. On the larger-scale Rope3D heterologous benchmark, we achieve notable gains of 14.48% for car and 12.41% for large vehicle. We aspire to contribute insights on the exploration of roadside perception techniques, emphasizing their capability for scenario generalization. The code will be available at {\url{ https://github.com/yanglei18/SGV3D}}

{{</citation>}}


### (102/132) High Resolution Image Quality Database (Huang Huang et al., 2024)

{{<citation>}}

Huang Huang, Qiang Wan, Jari Korhonen. (2024)  
**High Resolution Image Quality Database**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.16087v1)  

---


**ABSTRACT**  
With technology for digital photography and high resolution displays rapidly evolving and gaining popularity, there is a growing demand for blind image quality assessment (BIQA) models for high resolution images. Unfortunately, the publicly available large scale image quality databases used for training BIQA models contain mostly low or general resolution images. Since image resizing affects image quality, we assume that the accuracy of BIQA models trained on low resolution images would not be optimal for high resolution images. Therefore, we created a new high resolution image quality database (HRIQ), consisting of 1120 images with resolution of 2880x2160 pixels. We conducted a subjective study to collect the subjective quality ratings for HRIQ in a controlled laboratory setting, resulting in accurate MOS at high resolution. To demonstrate the importance of a high resolution image quality database for training BIQA models to predict mean opinion scores (MOS) of high resolution images accurately, we trained and tested several traditional and deep learning based BIQA methods on different resolution versions of our database. The database is publicly available in https://github.com/jarikorhonen/hriq.

{{</citation>}}


### (103/132) SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design (Seokju Yun et al., 2024)

{{<citation>}}

Seokju Yun, Youngmin Ro. (2024)  
**SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.16456v1)  

---


**ABSTRACT**  
Recently, efficient Vision Transformers have shown great performance with low latency on resource-constrained devices. Conventionally, they use 4x4 patch embeddings and a 4-stage structure at the macro level, while utilizing sophisticated attention with multi-head configuration at the micro level. This paper aims to address computational redundancy at all design levels in a memory-efficient manner. We discover that using larger-stride patchify stem not only reduces memory access costs but also achieves competitive performance by leveraging token representations with reduced spatial redundancy from the early stages. Furthermore, our preliminary analyses suggest that attention layers in the early stages can be substituted with convolutions, and several attention heads in the latter stages are computationally redundant. To handle this, we introduce a single-head attention module that inherently prevents head redundancy and simultaneously boosts accuracy by parallelly combining global and local information. Building upon our solutions, we introduce SHViT, a Single-Head Vision Transformer that obtains the state-of-the-art speed-accuracy tradeoff. For example, on ImageNet-1k, our SHViT-S4 is 3.3x, 8.1x, and 2.4x faster than MobileViTv2 x1.0 on GPU, CPU, and iPhone12 mobile device, respectively, while being 1.3% more accurate. For object detection and instance segmentation on MS COCO using Mask-RCNN head, our model achieves performance comparable to FastViT-SA12 while exhibiting 3.8x and 2.0x lower backbone latency on GPU and mobile device, respectively.

{{</citation>}}


### (104/132) TFDMNet: A Novel Network Structure Combines the Time Domain and Frequency Domain Features (Hengyue Pan et al., 2024)

{{<citation>}}

Hengyue Pan, Yixin Chen, Zhiliang Tian, Peng Qiao, Linbo Qiao, Dongsheng Li. (2024)  
**TFDMNet: A Novel Network Structure Combines the Time Domain and Frequency Domain Features**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.15949v1)  

---


**ABSTRACT**  
Convolutional neural network (CNN) has achieved impressive success in computer vision during the past few decades. The image convolution operation helps CNNs to get good performance on image-related tasks. However, it also has high computation complexity and hard to be parallelized. This paper proposes a novel Element-wise Multiplication Layer (EML) to replace convolution layers, which can be trained in the frequency domain. Theoretical analyses show that EMLs lower the computation complexity and easier to be parallelized. Moreover, we introduce a Weight Fixation mechanism to alleviate the problem of over-fitting, and analyze the working behavior of Batch Normalization and Dropout in the frequency domain. To get the balance between the computation complexity and memory usage, we propose a new network structure, namely Time-Frequency Domain Mixture Network (TFDMNet), which combines the advantages of both convolution layers and EMLs. Experimental results imply that TFDMNet achieves good performance on MNIST, CIFAR-10 and ImageNet databases with less number of operations comparing with corresponding CNNs.

{{</citation>}}


### (105/132) MoE-LLaVA: Mixture of Experts for Large Vision-Language Models (Bin Lin et al., 2024)

{{<citation>}}

Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Junwu Zhang, Munan Ning, Li Yuan. (2024)  
**MoE-LLaVA: Mixture of Experts for Large Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.15947v1)  

---


**ABSTRACT**  
For Large Vision-Language Models (LVLMs), scaling the model can effectively improve performance. However, expanding model parameters significantly increases the training and inferring costs, as all model parameters are activated for each token in the calculation. In this work, we propose a novel training strategy MoE-tuning for LVLMs, which can constructing a sparse model with an outrageous number of parameter but a constant computational cost, and effectively addresses the performance degradation typically associated with multi-modal learning and model sparsity. Furthermore, we present the MoE-LLaVA framework, a MoE-based sparse LVLM architecture. This framework uniquely activates only the top-k experts through routers during deployment, keeping the remaining experts inactive. Our extensive experiments highlight the excellent capabilities of MoE-LLaVA in visual understanding and its potential to reduce hallucinations in model outputs. Remarkably, with just 3 billion sparsely activated parameters, MoE-LLaVA demonstrates performance comparable to the LLaVA-1.5-7B on various visual understanding datasets and even surpasses the LLaVA-1.5-13B in object hallucination benchmarks. Through MoE-LLaVA, we aim to establish a baseline for sparse LVLMs and provide valuable insights for future research in developing more efficient and effective multi-modal learning systems. Code is released at \url{https://github.com/PKU-YuanGroup/MoE-LLaVA}.

{{</citation>}}


### (106/132) HICH Image/Text (HICH-IT): Comprehensive Text and Image Datasets for Hypertensive Intracerebral Hemorrhage Research (Jie Li et al., 2024)

{{<citation>}}

Jie Li, Yulong Xia, Tongxin Yang, Fenglin Cai, Miao Wei, Zhiwei Zhang, Li Jiang. (2024)  
**HICH Image/Text (HICH-IT): Comprehensive Text and Image Datasets for Hypertensive Intracerebral Hemorrhage Research**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15934v1)  

---


**ABSTRACT**  
In this paper, we introduce a new multimodal dataset in the medical field of hypertensive intracerebral hemorrhage(HICH), called as HICH-IT, which includes both textual information and head CT images. This dataset is designed to enhance the accuracy of artificial intelligence in the diagnosis and treatment of HICH. This dataset, built upon the foundation of standard text and image data, incorporates specific annotations within the text data, extracting key content from the text information, and categorizes the annotation content of imaging data into four types: brain midline, hematoma, left cerebral ventricle, and right cerebral ventricle. HICH-IT aims to be a foundational dataset for feature learning in image segmentation tasks and named entity recognition. To further understand the dataset, we have trained deep learning algorithms to observe the performance. The pretrained models have been released at both www.daip.club and github.com/Deep-AI-Application-DAIP. The dataset has been uploaded to https://github.com/CYBUS123456/HICH-IT-Datasets.   Index Terms-HICH, Deep learning, Intraparenchymal hemorrhage, named entity recognition, novel dataset

{{</citation>}}


### (107/132) Overcoming the Pitfalls of Vision-Language Model Finetuning for OOD Generalization (Yuhang Zang et al., 2024)

{{<citation>}}

Yuhang Zang, Hanlin Goh, Josh Susskind, Chen Huang. (2024)  
**Overcoming the Pitfalls of Vision-Language Model Finetuning for OOD Generalization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.15914v1)  

---


**ABSTRACT**  
Existing vision-language models exhibit strong generalization on a variety of visual domains and tasks. However, such models mainly perform zero-shot recognition in a closed-set manner, and thus struggle to handle open-domain visual concepts by design. There are recent finetuning methods, such as prompt learning, that not only study the discrimination between in-distribution (ID) and out-of-distribution (OOD) samples, but also show some improvements in both ID and OOD accuracies. In this paper, we first demonstrate that vision-language models, after long enough finetuning but without proper regularization, tend to overfit the known classes in the given dataset, with degraded performance on unknown classes. Then we propose a novel approach OGEN to address this pitfall, with the main focus on improving the OOD GENeralization of finetuned models. Specifically, a class-conditional feature generator is introduced to synthesize OOD features using just the class name of any unknown class. Such synthesized features will provide useful knowledge about unknowns and help regularize the decision boundary between ID and OOD data when optimized jointly. Equally important is our adaptive self-distillation mechanism to regularize our feature generation model during joint optimization, i.e., adaptively transferring knowledge between model states to further prevent overfitting. Experiments validate that our method yields convincing gains in OOD generalization performance in different settings.

{{</citation>}}


### (108/132) $\boldsymbol{M^2}$-Encoder: Advancing Bilingual Image-Text Understanding by Large-scale Efficient Pretraining (Qingpei Guo et al., 2024)

{{<citation>}}

Qingpei Guo, Furong Xu, Hanxiao Zhang, Wang Ren, Ziping Ma, Lin Ju, Jian Wang, Jingdong Chen, Ming Yang. (2024)  
**$\boldsymbol{M^2}$-Encoder: Advancing Bilingual Image-Text Understanding by Large-scale Efficient Pretraining**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.15896v1)  

---


**ABSTRACT**  
Vision-language foundation models like CLIP have revolutionized the field of artificial intelligence. Nevertheless, VLM models supporting multi-language, e.g., in both Chinese and English, have lagged due to the relative scarcity of large-scale pretraining datasets. Toward this end, we introduce a comprehensive bilingual (Chinese-English) dataset BM-6B with over 6 billion image-text pairs, aimed at enhancing multimodal foundation models to well understand images in both languages. To handle such a scale of dataset, we propose a novel grouped aggregation approach for image-text contrastive loss computation, which reduces the communication overhead and GPU memory demands significantly, facilitating a 60% increase in training speed. We pretrain a series of bilingual image-text foundation models with an enhanced fine-grained understanding ability on BM-6B, the resulting models, dubbed as $M^2$-Encoders (pronounced "M-Square"), set new benchmarks in both languages for multimodal retrieval and classification tasks. Notably, Our largest $M^2$-Encoder-10B model has achieved top-1 accuracies of 88.5% on ImageNet and 80.7% on ImageNet-CN under a zero-shot classification setting, surpassing previously reported SoTA methods by 2.2% and 21.1%, respectively. The $M^2$-Encoder series represents one of the most comprehensive bilingual image-text foundation models to date, so we are making it available to the research community for further exploration and development.

{{</citation>}}


### (109/132) Rectify the Regression Bias in Long-Tailed Object Detection (Ke Zhu et al., 2024)

{{<citation>}}

Ke Zhu, Minghao Fu, Jie Shao, Tianyu Liu, Jianxin Wu. (2024)  
**Rectify the Regression Bias in Long-Tailed Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias, Object Detection  
[Paper Link](http://arxiv.org/abs/2401.15885v2)  

---


**ABSTRACT**  
Long-tailed object detection faces great challenges because of its extremely imbalanced class distribution. Recent methods mainly focus on the classification bias and its loss function design, while ignoring the subtle influence of the regression branch. This paper shows that the regression bias exists and does adversely and seriously impact the detection accuracy. While existing methods fail to handle the regression bias, the class-specific regression head for rare classes is hypothesized to be the main cause of it in this paper. As a result, three kinds of viable solutions to cater for the rare categories are proposed, including adding a class-agnostic branch, clustering heads and merging heads. The proposed methods brings in consistent and significant improvements over existing long-tailed detection methods, especially in rare and common classes. The proposed method achieves state-of-the-art performance in the large vocabulary LVIS dataset with different backbones and architectures. It generalizes well to more difficult evaluation metrics, relatively balanced datasets, and the mask branch. This is the first attempt to reveal and explore rectifying of the regression bias in long-tailed object detection.

{{</citation>}}


### (110/132) LiDAR-PTQ: Post-Training Quantization for Point Cloud 3D Object Detection (Sifan Zhou et al., 2024)

{{<citation>}}

Sifan Zhou, Liang Li, Xinyu Zhang, Bo Zhang, Shipeng Bai, Miao Sun, Ziyu Zhao, Xiaobo Lu, Xiangxiang Chu. (2024)  
**LiDAR-PTQ: Post-Training Quantization for Point Cloud 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Quantization  
[Paper Link](http://arxiv.org/abs/2401.15865v1)  

---


**ABSTRACT**  
Due to highly constrained computing power and memory, deploying 3D lidar-based detectors on edge devices equipped in autonomous vehicles and robots poses a crucial challenge. Being a convenient and straightforward model compression approach, Post-Training Quantization (PTQ) has been widely adopted in 2D vision tasks. However, applying it directly to 3D lidar-based tasks inevitably leads to performance degradation. As a remedy, we propose an effective PTQ method called LiDAR-PTQ, which is particularly curated for 3D lidar detection (both SPConv-based and SPConv-free). Our LiDAR-PTQ features three main components, \textbf{(1)} a sparsity-based calibration method to determine the initialization of quantization parameters, \textbf{(2)} a Task-guided Global Positive Loss (TGPL) to reduce the disparity between the final predictions before and after quantization, \textbf{(3)} an adaptive rounding-to-nearest operation to minimize the layerwise reconstruction error. Extensive experiments demonstrate that our LiDAR-PTQ can achieve state-of-the-art quantization performance when applied to CenterPoint (both Pillar-based and Voxel-based). To our knowledge, for the very first time in lidar-based 3D detection tasks, the PTQ INT8 model's accuracy is almost the same as the FP32 model while enjoying $3\times$ inference speedup. Moreover, our LiDAR-PTQ is cost-effective being $30\times$ faster than the quantization-aware training method. Code will be released at \url{https://github.com/StiphyJay/LiDAR-PTQ}.

{{</citation>}}


### (111/132) Muffin or Chihuahua? Challenging Large Vision-Language Models with Multipanel VQA (Yue Fan et al., 2024)

{{<citation>}}

Yue Fan, Jing Gu, Kaiwen Zhou, Qianqi Yan, Shan Jiang, Ching-Chen Kuo, Xinze Guan, Xin Eric Wang. (2024)  
**Muffin or Chihuahua? Challenging Large Vision-Language Models with Multipanel VQA**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: AI, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.15847v1)  

---


**ABSTRACT**  
Multipanel images, commonly seen as web screenshots, posters, etc., pervade our daily lives. These images, characterized by their composition of multiple subfigures in distinct layouts, effectively convey information to people. Toward building advanced multimodal AI applications, such as agents that understand complex scenes and navigate through webpages, the skill of multipanel visual reasoning is essential, and a comprehensive evaluation of models in this regard is important. Therefore, our paper introduces Multipanel Visual Question Answering (MultipanelVQA), a novel benchmark that specifically challenges models in comprehending multipanel images. The benchmark comprises 6,600 questions and answers related to multipanel images. While these questions are straightforward for average humans, achieving nearly perfect correctness, they pose significant challenges to the state-of-the-art Large Vision Language Models (LVLMs) we tested. In our study, we utilized synthetically curated multipanel images specifically designed to isolate and evaluate the impact of diverse factors on model performance, revealing the sensitivity of LVLMs to various interferences in multipanel images, such as adjacent subfigures and layout complexity. As a result, MultipanelVQA highlights the need and direction for improving LVLMs' ability to understand complex visual-language contexts. Code and data are released at https://sites.google.com/view/multipanelvqa/home.

{{</citation>}}


### (112/132) LCVO: An Efficient Pretraining-Free Framework for Visual Question Answering Grounding (Yuhan Chen et al., 2024)

{{<citation>}}

Yuhan Chen, Lumei Su, Lihua Chen, Zhiwei Lin. (2024)  
**LCVO: An Efficient Pretraining-Free Framework for Visual Question Answering Grounding**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.15842v1)  

---


**ABSTRACT**  
In this paper, the LCVO modular method is proposed for the Visual Question Answering (VQA) Grounding task in the vision-language multimodal domain. This approach relies on a frozen large language model (LLM) as intermediate mediator between the off-the-shelf VQA model and the off-the-shelf Open-Vocabulary Object Detection (OVD) model, where the LLM transforms and conveys textual information between the two modules based on a designed prompt. LCVO establish an integrated plug-and-play framework without the need for any pre-training process. This framework can be deployed for VQA Grounding tasks under low computational resources. The modularized model within the framework allows application with various state-of-the-art pre-trained models, exhibiting significant potential to be advance with the times. Experimental implementations were conducted under constrained computational and memory resources, evaluating the proposed method's performance on benchmark datasets including GQA, CLEVR, and VizWiz-VQA-Grounding. Comparative analyses with baseline methods demonstrate the robust competitiveness of LCVO.

{{</citation>}}


### (113/132) Transparency Attacks: How Imperceptible Image Layers Can Fool AI Perception (Forrest McKee et al., 2024)

{{<citation>}}

Forrest McKee, David Noever. (2024)  
**Transparency Attacks: How Imperceptible Image Layers Can Fool AI Perception**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2401.15817v1)  

---


**ABSTRACT**  
This paper investigates a novel algorithmic vulnerability when imperceptible image layers confound multiple vision models into arbitrary label assignments and captions. We explore image preprocessing methods to introduce stealth transparency, which triggers AI misinterpretation of what the human eye perceives. The research compiles a broad attack surface to investigate the consequences ranging from traditional watermarking, steganography, and background-foreground miscues. We demonstrate dataset poisoning using the attack to mislabel a collection of grayscale landscapes and logos using either a single attack layer or randomly selected poisoning classes. For example, a military tank to the human eye is a mislabeled bridge to object classifiers based on convolutional networks (YOLO, etc.) and vision transformers (ViT, GPT-Vision, etc.). A notable attack limitation stems from its dependency on the background (hidden) layer in grayscale as a rough match to the transparent foreground image that the human eye perceives. This dependency limits the practical success rate without manual tuning and exposes the hidden layers when placed on the opposite display theme (e.g., light background, light transparent foreground visible, works best against a light theme image viewer or browser). The stealth transparency confounds established vision systems, including evading facial recognition and surveillance, digital watermarking, content filtering, dataset curating, automotive and drone autonomy, forensic evidence tampering, and retail product misclassifying. This method stands in contrast to traditional adversarial attacks that typically focus on modifying pixel values in ways that are either slightly perceptible or entirely imperceptible for both humans and machines.

{{</citation>}}


## cs.SE (6)



### (114/132) ReGAL: Refactoring Programs to Discover Generalizable Abstractions (Elias Stengel-Eskin et al., 2024)

{{<citation>}}

Elias Stengel-Eskin, Archiki Prasad, Mohit Bansal. (2024)  
**ReGAL: Refactoring Programs to Discover Generalizable Abstractions**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-LG, cs-SE, cs.SE  
Keywords: GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2401.16467v1)  

---


**ABSTRACT**  
While large language models (LLMs) are increasingly being used for program synthesis, they lack the global view needed to develop useful abstractions; they generally predict programs one at a time, often repeating the same functionality. Generating redundant code from scratch is both inefficient and error-prone. To address this, we propose Refactoring for Generalizable Abstraction Learning (ReGAL), a gradient-free method for learning a library of reusable functions via code refactorization, i.e. restructuring code without changing its execution output. ReGAL learns from a small set of existing programs, iteratively verifying and refining its abstractions via execution. We find that the shared function libraries discovered by ReGAL make programs easier to predict across diverse domains. On three datasets (LOGO graphics generation, Date reasoning, and TextCraft, a Minecraft-based text game), both open-source and proprietary LLMs improve in accuracy when predicting programs with ReGAL functions. For CodeLlama-13B, ReGAL results in absolute accuracy increases of 11.5% on graphics, 26.1% on date understanding, and 8.1% on TextCraft, outperforming GPT-3.5 in two of three domains. Our analysis reveals ReGAL's abstractions encapsulate frequently-used subroutines as well as environment dynamics.

{{</citation>}}


### (115/132) The role of library versions in Developer-ChatGPT conversations (Rachna Raj et al., 2024)

{{<citation>}}

Rachna Raj, Diego Elias Costa. (2024)  
**The role of library versions in Developer-ChatGPT conversations**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.16340v1)  

---


**ABSTRACT**  
The latest breakthroughs in large language models (LLM) have empowered software development tools, such as ChatGPT, to aid developers in complex tasks. Developers use ChatGPT to write code, review code changes, and even debug their programs. In these interactions, ChatGPT often recommends code snippets that depend on external libraries. However, code from libraries changes over time, invalidating a once-correct code snippet and making it difficult to reuse recommended code.   In this study, we analyze DevGPT, a dataset of more than 4,000 Developer-ChatGPT interactions, to understand the role of library versions in code-related conversations. We quantify how often library version constraints are mentioned in code-related conversations and when ChatGPT recommends the installation of specific libraries. Our findings show that, albeit to constantly recommend and analyze code with external dependencies, library version constraints only appear in 9% of the conversations. In the majority of conversations, the version constraints are prompted by users (as opposed to being specified by ChatGPT) as a method for receiving better quality responses. Moreover, we study how library version constraints are used in the conversation through qualitative methods, identifying several potential problems that warrant further research.

{{</citation>}}


### (116/132) Security Code Review by LLMs: A Deep Dive into Responses (Jiaxin Yu et al., 2024)

{{<citation>}}

Jiaxin Yu, Peng Liang, Yujia Fu, Amjed Tahir, Mojtaba Shahin, Chong Wang, Yangxiao Cai. (2024)  
**Security Code Review by LLMs: A Deep Dive into Responses**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, Security  
[Paper Link](http://arxiv.org/abs/2401.16310v1)  

---


**ABSTRACT**  
Security code review aims to combine automated tools and manual efforts to detect security defects during development. The rapid development of Large Language Models (LLMs) has shown promising potential in software development, as well as opening up new possibilities in automated security code review. To explore the challenges of applying LLMs in practical code review for security defect detection, this study compared the detection performance of three state-of-the-art LLMs (Gemini Pro, GPT-4, and GPT-3.5) under five prompts on 549 code files that contain security defects from real-world code reviews. Through analyzing 82 responses generated by the best-performing LLM-prompt combination based on 100 randomly selected code files, we extracted and categorized quality problems present in these responses into 5 themes and 16 categories. Our results indicate that the responses produced by LLMs often suffer from verbosity, vagueness, and incompleteness, highlighting the necessity to enhance their conciseness, understandability, and compliance to security defect detection. This work reveals the deficiencies of LLM-generated responses in security code review and paves the way for future optimization of LLMs towards this task.

{{</citation>}}


### (117/132) An Empirical Study on Usage and Perceptions of LLMs in a Software Engineering Project (Sanka Rasnayaka et al., 2024)

{{<citation>}}

Sanka Rasnayaka, Guanlin Wang, Ridwan Shariffdeen, Ganesh Neelakanta Iyer. (2024)  
**An Empirical Study on Usage and Perceptions of LLMs in a Software Engineering Project**  

---
Primary Category: cs.SE  
Categories: D-2-3, cs-AI, cs-SE, cs.SE  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.16186v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) represent a leap in artificial intelligence, excelling in tasks using human language(s). Although the main focus of general-purpose LLMs is not code generation, they have shown promising results in the domain. However, the usefulness of LLMs in an academic software engineering project has not been fully explored yet. In this study, we explore the usefulness of LLMs for 214 students working in teams consisting of up to six members. Notably, in the academic course through which this study is conducted, students were encouraged to integrate LLMs into their development tool-chain, in contrast to most other academic courses that explicitly prohibit the use of LLMs.   In this paper, we analyze the AI-generated code, prompts used for code generation, and the human intervention levels to integrate the code into the code base. We also conduct a perception study to gain insights into the perceived usefulness, influencing factors, and future outlook of LLM from a computer science student's perspective. Our findings suggest that LLMs can play a crucial role in the early stages of software development, especially in generating foundational code structures, and helping with syntax and error debugging. These insights provide us with a framework on how to effectively utilize LLMs as a tool to enhance the productivity of software engineering students, and highlight the necessity of shifting the educational focus toward preparing students for successful human-AI collaboration.

{{</citation>}}


### (118/132) Knowledge-Aware Code Generation with Large Language Models (Tao Huang et al., 2024)

{{<citation>}}

Tao Huang, Zhihong Sun, Zhi Jin, Ge Li, Chen Lyu. (2024)  
**Knowledge-Aware Code Generation with Large Language Models**  

---
Primary Category: cs.SE  
Categories: D-2-3, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.15940v3)  

---


**ABSTRACT**  
Large Language Models (LLMs) perform well on basic programming problems. However, they encounter challenges when dealing with complex tasks involving the use of diverse algorithmic and data structure skills, particularly programming competition-level problems. Notably, ChatGPT exhibits proficient performance on problems it has encountered during its pre-training phase, but this performance deteriorates when faced with novel problems. Consequently, enhancing the ability of LLMs to address unfamiliar problems has emerged as a pivotal research focus. The problem-solving process of LLMs mirrors human programmers' approach to a certain extent. When confronted with new programming tasks, human programmers engage in task planning and code writing with the previously acquired knowledge about algorithms and data structures. Despite having learned such knowledge, LLMs struggle to effectively apply it when faced with specific new problems. To address this issue, we constructed a novel dataset, CodeF, which contains a portion of programming problems that ChatGPT has not previously encountered. Furthermore, we developed a Knowledge Library tailored for Python programming contest problems and introduced the concept of Knowledge-Aware Code Generation (KareCoder). KareCoder bolsters the models' understanding and problem-solving capabilities by integrating prompt and knowledge from the library into the LLMs' code generation reasoning process, especially on Pass@1 metrics. Upon testing on the CodeF and APPS datasets, KareCoder demonstrated outstanding performance in handling novel problems previously unencountered by LLMs. In contrast with the code directly generated by ChatGPT, KareCoder achieved a relative improvement of 23.3% on the Pass@1 metric on the CodeF post2021-9 dataset. Additionally, it performs well compared to other methods when dealing with problems that LLMs have previously encountered.

{{</citation>}}


### (119/132) APIGen: Generative API Method Recommendation (Yujia Chen et al., 2024)

{{<citation>}}

Yujia Chen, Cuiyun Gao, Muyijie Zhu, Qing Liao, Yong Wang, Guoai Xu. (2024)  
**APIGen: Generative API Method Recommendation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.15843v1)  

---


**ABSTRACT**  
Automatic API method recommendation is an essential task of code intelligence, which aims to suggest suitable APIs for programming queries. Existing approaches can be categorized into two primary groups: retrieval-based and learning-based approaches. Although these approaches have achieved remarkable success, they still come with notable limitations. The retrieval-based approaches rely on the text representation capabilities of embedding models, while the learning-based approaches require extensive task-specific labeled data for training. To mitigate the limitations, we propose APIGen, a generative API recommendation approach through enhanced in-context learning (ICL). APIGen involves two main components: (1) Diverse Examples Selection. APIGen searches for similar posts to the programming queries from the lexical, syntactical, and semantic perspectives, providing more informative examples for ICL. (2) Guided API Recommendation. APIGen enables large language models (LLMs) to perform reasoning before generating API recommendations, where the reasoning involves fine-grained matching between the task intent behind the queries and the factual knowledge of the APIs. With the reasoning process, APIGen makes recommended APIs better meet the programming requirement of queries and also enhances the interpretability of results. We compare APIGen with four existing approaches on two publicly available benchmarks. Experiments show that APIGen outperforms the best baseline CLEAR by 105.8% in method-level API recommendation and 54.3% in class-level API recommendation in terms of SuccessRate@1. Besides, APIGen achieves an average 49.87% increase compared to the zero-shot performance of popular LLMs such as GPT-4 in method-level API recommendation regarding the SuccessRate@3 metric.

{{</citation>}}


## cs.RO (3)



### (120/132) Curriculum-Based Reinforcement Learning for Quadrupedal Jumping: A Reference-free Design (Vassil Atanassov et al., 2024)

{{<citation>}}

Vassil Atanassov, Jiatao Ding, Jens Kober, Ioannis Havoutis, Cosimo Della Santina. (2024)  
**Curriculum-Based Reinforcement Learning for Quadrupedal Jumping: A Reference-free Design**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.16337v1)  

---


**ABSTRACT**  
Deep reinforcement learning (DRL) has emerged as a promising solution to mastering explosive and versatile quadrupedal jumping skills. However, current DRL-based frameworks usually rely on well-defined reference trajectories, which are obtained by capturing animal motions or transferring experience from existing controllers. This work explores the possibility of learning dynamic jumping without imitating a reference trajectory. To this end, we incorporate a curriculum design into DRL so as to accomplish challenging tasks progressively. Starting from a vertical in-place jump, we then generalize the learned policy to forward and diagonal jumps and, finally, learn to jump across obstacles. Conditioned on the desired landing location, orientation, and obstacle dimensions, the proposed approach contributes to a wide range of jumping motions, including omnidirectional jumping and robust jumping, alleviating the effort to extract references in advance. Particularly, without constraints from the reference motion, a 90cm forward jump is achieved, exceeding previous records for similar robots reported in the existing literature. Additionally, continuous jumping on the soft grassy floor is accomplished, even when it is not encountered in the training stage. A supplementary video showing our results can be found at https://youtu.be/nRaMCrwU5X8 .

{{</citation>}}


### (121/132) CognitiveOS: Large Multimodal Model based System to Endow Any Type of Robot with Generative AI (Artem Lykov et al., 2024)

{{<citation>}}

Artem Lykov, Mikhail Konenkov, Koffivi Fidèle Gbagbe, Mikhail Litvinov, Robinroy Peter, Denis Davletshin, Aleksey Fedoseev, Oleg Kobzarev, Ali Alabbas, Oussama Alyounes, Miguel Altamirano Cabrera, Dzmitry Tsetserukou. (2024)  
**CognitiveOS: Large Multimodal Model based System to Endow Any Type of Robot with Generative AI**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.16205v1)  

---


**ABSTRACT**  
This paper introduces CognitiveOS, a disruptive system based on multiple transformer-based models, endowing robots of various types with cognitive abilities not only for communication with humans but also for task resolution through physical interaction with the environment. The system operates smoothly on different robotic platforms without extra tuning. It autonomously makes decisions for task execution by analyzing the environment and using information from its long-term memory. The system underwent testing on various platforms, including quadruped robots and manipulator robots, showcasing its capability to formulate behavioral plans even for robots whose behavioral examples were absent in the training dataset.   Experimental results revealed the system's high performance in advanced task comprehension and adaptability, emphasizing its potential for real-world applications. The chapters of this paper describe the key components of the system and the dataset structure. The dataset for fine-tuning step generation model is provided at the following link: link coming soon

{{</citation>}}


### (122/132) SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning (Jianlan Luo et al., 2024)

{{<citation>}}

Jianlan Luo, Zheyuan Hu, Charles Xu, You Liang Tan, Jacob Berg, Archit Sharma, Stefan Schaal, Chelsea Finn, Abhishek Gupta, Sergey Levine. (2024)  
**SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.16013v2)  

---


**ABSTRACT**  
In recent years, significant progress has been made in the field of robotic reinforcement learning (RL), enabling methods that handle complex image observations, train in the real world, and incorporate auxiliary data, such as demonstrations and prior experience. However, despite these advances, robotic RL remains hard to use. It is acknowledged among practitioners that the particular implementation details of these algorithms are often just as important (if not more so) for performance as the choice of algorithm. We posit that a significant challenge to widespread adoption of robotic RL, as well as further development of robotic RL methods, is the comparative inaccessibility of such methods. To address this challenge, we developed a carefully implemented library containing a sample efficient off-policy deep RL method, together with methods for computing rewards and resetting the environment, a high-quality controller for a widely-adopted robot, and a number of challenging example tasks. We provide this library as a resource for the community, describe its design choices, and present experimental results. Perhaps surprisingly, we find that our implementation can achieve very efficient learning, acquiring policies for PCB board assembly, cable routing, and object relocation between 25 to 50 minutes of training per policy on average, improving over state-of-the-art results reported for similar tasks in the literature. These policies achieve perfect or near-perfect success rates, extreme robustness even under perturbations, and exhibit emergent recovery and correction behaviors. We hope that these promising results and our high-quality open-source implementation will provide a tool for the robotics community to facilitate further developments in robotic RL. Our code, documentation, and videos can be found at https://serl-robot.github.io/

{{</citation>}}


## eess.SY (4)



### (123/132) Optimal Control of Renewable Energy Communities subject to Network Peak Fees with Model Predictive Control and Reinforcement Learning Algorithms (Samy Aittahar et al., 2024)

{{<citation>}}

Samy Aittahar, Adrien Bolland, Guillaume Derval, Damien Ernst. (2024)  
**Optimal Control of Renewable Energy Communities subject to Network Peak Fees with Model Predictive Control and Reinforcement Learning Algorithms**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.16321v1)  

---


**ABSTRACT**  
We propose in this paper an optimal control framework for renewable energy communities (RECs) equipped with controllable assets. Such RECs allow its members to exchange production surplus through an internal market. The objective is to control their assets in order to minimise the sum of individual electricity bills. These bills account for the electricity exchanged through the REC and with the retailers. Typically, for large companies, another important part of the bills are the costs related to the power peaks; in our framework, they are determined from the energy exchanges with the retailers. We compare rule-based control strategies with the two following control algorithms. The first one is derived from model predictive control techniques, and the second one is built with reinforcement learning techniques. We also compare variants of these algorithms that neglect the peak power costs. Results confirm that using policies accounting for the power peaks lead to a significantly lower sum of electricity bills and thus better control strategies at the cost of higher computation time. Furthermore, policies trained with reinforcement learning approaches appear promising for real-time control of the communities, where model predictive control policies may be computationally expensive in practice. These findings encourage pursuing the efforts toward development of scalable control algorithms, operating from a centralised standpoint, for renewable energy communities equipped with controllable assets.

{{</citation>}}


### (124/132) Scalable Reinforcement Learning for Linear-Quadratic Control of Networks (Johan Olsson et al., 2024)

{{<citation>}}

Johan Olsson, Runyu Zhang, Emma Tegling, Na Li. (2024)  
**Scalable Reinforcement Learning for Linear-Quadratic Control of Networks**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.16183v1)  

---


**ABSTRACT**  
Distributed optimal control is known to be challenging and can become intractable even for linear-quadratic regulator problems. In this work, we study a special class of such problems where distributed state feedback controllers can give near-optimal performance. More specifically, we consider networked linear-quadratic controllers with decoupled costs and spatially exponentially decaying dynamics. We aim to exploit the structure in the problem to design a scalable reinforcement learning algorithm for learning a distributed controller. Recent work has shown that the optimal controller can be well approximated only using information from a $\kappa$-neighborhood of each agent. Motivated by these results, we show that similar results hold for the agents' individual value and Q-functions. We continue by designing an algorithm, based on the actor-critic framework, to learn distributed controllers only using local information. Specifically, the Q-function is estimated by modifying the Least Squares Temporal Difference for Q-functions method to only use local information. The algorithm then updates the policy using gradient descent. Finally, we evaluate the algorithm through simulations that indeed suggest near-optimal performance.

{{</citation>}}


### (125/132) Attentive Convolutional Deep Reinforcement Learning for Optimizing Solar-Storage Systems in Real-Time Electricity Markets (Jinhao Li et al., 2024)

{{<citation>}}

Jinhao Li, Changlong Wang, Hao Wang. (2024)  
**Attentive Convolutional Deep Reinforcement Learning for Optimizing Solar-Storage Systems in Real-Time Electricity Markets**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.15853v1)  

---


**ABSTRACT**  
This paper studies the synergy of solar-battery energy storage system (BESS) and develops a viable strategy for the BESS to unlock its economic potential by serving as a backup to reduce solar curtailments while also participating in the electricity market. We model the real-time bidding of the solar-battery system as two Markov decision processes for the solar farm and the BESS, respectively. We develop a novel deep reinforcement learning (DRL) algorithm to solve the problem by leveraging attention mechanism (AC) and multi-grained feature convolution to process DRL input for better bidding decisions. Simulation results demonstrate that our AC-DRL outperforms two optimization-based and one DRL-based benchmarks by generating 23%, 20%, and 11% higher revenue, as well as improving curtailment responses. The excess solar generation can effectively charge the BESS to bid in the market, significantly reducing solar curtailments by 76% and creating synergy for the solar-battery system to be more viable.

{{</citation>}}


### (126/132) Deep Reinforcement Learning for Voltage Control and Renewable Accommodation Using Spatial-Temporal Graph Information (Jinhao Li et al., 2024)

{{<citation>}}

Jinhao Li, Ruichang Zhang, Hao Wang, Zhi Liu, Hongyang Lai, Yanru Zhang. (2024)  
**Deep Reinforcement Learning for Voltage Control and Renewable Accommodation Using Spatial-Temporal Graph Information**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.15848v1)  

---


**ABSTRACT**  
Renewable energy resources (RERs) have been increasingly integrated into distribution networks (DNs) for decarbonization. However, the variable nature of RERs introduces uncertainties to DNs, frequently resulting in voltage fluctuations that threaten system security and hamper the further adoption of RERs. To incentivize more RER penetration, we propose a deep reinforcement learning (DRL)-based strategy to dynamically balance the trade-off between voltage fluctuation control and renewable accommodation. To further extract multi-time-scale spatial-temporal (ST) graphical information of a DN, our strategy draws on a multi-grained attention-based spatial-temporal graph convolution network (MG-ASTGCN), consisting of ST attention mechanism and ST convolution to explore the node correlations in the spatial and temporal views. The continuous decision-making process of balancing such a trade-off can be modeled as a Markov decision process optimized by the deep deterministic policy gradient (DDPG) algorithm with the help of the derived ST information. We validate our strategy on the modified IEEE 33, 69, and 118-bus radial distribution systems, with outcomes significantly outperforming the optimization-based benchmarks. Simulations also reveal that our developed MG-ASTGCN can to a great extent accelerate the convergence speed of DDPG and improve its performance in stabilizing node voltage in an RER-rich DN. Moreover, our method improves the DN's robustness in the presence of generator failures.

{{</citation>}}


## q-bio.QM (1)



### (127/132) AI prediction of cardiovascular events using opportunistic epicardial adipose tissue assessments from CT calcium score (Tao Hu et al., 2024)

{{<citation>}}

Tao Hu, Joshua Freeze, Prerna Singh, Justin Kim, Yingnan Song, Hao Wu, Juhwan Lee, Sadeer Al-Kindi, Sanjay Rajagopalan, David L. Wilson, Ammar Hoori. (2024)  
**AI prediction of cardiovascular events using opportunistic epicardial adipose tissue assessments from CT calcium score**  

---
Primary Category: q-bio.QM  
Categories: cs-AI, q-bio-QM, q-bio.QM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.16190v1)  

---


**ABSTRACT**  
Background: Recent studies have used basic epicardial adipose tissue (EAT) assessments (e.g., volume and mean HU) to predict risk of atherosclerosis-related, major adverse cardiovascular events (MACE). Objectives: Create novel, hand-crafted EAT features, 'fat-omics', to capture the pathophysiology of EAT and improve MACE prediction. Methods: We segmented EAT using a previously-validated deep learning method with optional manual correction. We extracted 148 radiomic features (morphological, spatial, and intensity) and used Cox elastic-net for feature reduction and prediction of MACE. Results: Traditional fat features gave marginal prediction (EAT-volume/EAT-mean-HU/ BMI gave C-index 0.53/0.55/0.57, respectively). Significant improvement was obtained with 15 fat-omics features (C-index=0.69, test set). High-risk features included volume-of-voxels-having-elevated-HU-[-50, -30-HU] and HU-negative-skewness, both of which assess high HU, which as been implicated in fat inflammation. Other high-risk features include kurtosis-of-EAT-thickness, reflecting the heterogeneity of thicknesses, and EAT-volume-in-the-top-25%-of-the-heart, emphasizing adipose near the proximal coronary arteries. Kaplan-Meyer plots of Cox-identified, high- and low-risk patients were well separated with the median of the fat-omics risk, while high-risk group having HR 2.4 times that of the low-risk group (P<0.001). Conclusion: Preliminary findings indicate an opportunity to use more finely tuned, explainable assessments on EAT for improved cardiovascular risk prediction.

{{</citation>}}


## cs.LO (1)



### (128/132) Minimalistic System Modelling: Behaviours, Interfaces, and Local Reasoning (Didier Galmiche et al., 2024)

{{<citation>}}

Didier Galmiche, Timo Lang, David Pym. (2024)  
**Minimalistic System Modelling: Behaviours, Interfaces, and Local Reasoning**  

---
Primary Category: cs.LO  
Categories: 68Q60, F-3-0, cs-LO, cs.LO  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.16109v1)  

---


**ABSTRACT**  
The infrastructure upon which the functioning of society depends is composed of complex ecosystems of systems. Consequently, we must reason about the properties of such ecosystems, which requires that we construct models of them. There are very many approaches to systems modelling, typically building on complex structural and dynamic frameworks. Our purpose here is to explore a modelling framework based on minimal assumptions, starting from a primitive notion of behaviour, and to show that such an approach allows the recovery of the key ideas, including a generalized CAP theorem, required for effective modelling of and reasoning about ecosystems of systems. We establish a logic of behaviours and use it to express local reasoning principles for the compositional structure of systems.

{{</citation>}}


## q-fin.RM (1)



### (129/132) Credit Risk Meets Large Language Models: Building a Risk Indicator from Loan Descriptions in P2P Lending (Mario Sanz-Guerrero et al., 2024)

{{<citation>}}

Mario Sanz-Guerrero, Javier Arroyo. (2024)  
**Credit Risk Meets Large Language Models: Building a Risk Indicator from Loan Descriptions in P2P Lending**  

---
Primary Category: q-fin.RM  
Categories: cs-AI, cs-CL, cs-LG, q-fin-RM, q-fin.RM  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.16458v1)  

---


**ABSTRACT**  
Peer-to-peer (P2P) lending has emerged as a distinctive financing mechanism, linking borrowers with lenders through online platforms. However, P2P lending faces the challenge of information asymmetry, as lenders often lack sufficient data to assess the creditworthiness of borrowers. This paper proposes a novel approach to address this issue by leveraging the textual descriptions provided by borrowers during the loan application process. Our methodology involves processing these textual descriptions using a Large Language Model (LLM), a powerful tool capable of discerning patterns and semantics within the text. Transfer learning is applied to adapt the LLM to the specific task at hand.   Our results derived from the analysis of the Lending Club dataset show that the risk score generated by BERT, a widely used LLM, significantly improves the performance of credit risk classifiers. However, the inherent opacity of LLM-based systems, coupled with uncertainties about potential biases, underscores critical considerations for regulatory frameworks and engenders trust-related concerns among end-users, opening new avenues for future research in the dynamic landscape of P2P lending and artificial intelligence.

{{</citation>}}


## hep-ex (1)



### (130/132) Combined track finding with GNN & CKF (Lukas Heinrich et al., 2024)

{{<citation>}}

Lukas Heinrich, Benjamin Huth, Andreas Salzburger, Tilo Wettig. (2024)  
**Combined track finding with GNN & CKF**  

---
Primary Category: hep-ex  
Categories: cs-LG, hep-ex, hep-ex  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.16016v1)  

---


**ABSTRACT**  
The application of Graph Neural Networks (GNN) in track reconstruction is a promising approach to cope with the challenges arising at the High-Luminosity upgrade of the Large Hadron Collider (HL-LHC). GNNs show good track-finding performance in high-multiplicity scenarios and are naturally parallelizable on heterogeneous compute architectures.   Typical high-energy-physics detectors have high resolution in the innermost layers to support vertex reconstruction but lower resolution in the outer parts. GNNs mainly rely on 3D space-point information, which can cause reduced track-finding performance in the outer regions.   In this contribution, we present a novel combination of GNN-based track finding with the classical Combinatorial Kalman Filter (CKF) algorithm to circumvent this issue: The GNN resolves the track candidates in the inner pixel region, where 3D space points can represent measurements very well. These candidates are then picked up by the CKF in the outer regions, where the CKF performs well even for 1D measurements.   Using the ACTS infrastructure, we present a proof of concept based on truth tracking in the pixels as well as a dedicated GNN pipeline trained on $t\bar{t}$ events with pile-up 200 in the OpenDataDetector.

{{</citation>}}


## cs.SD (1)



### (131/132) Continuous Target Speech Extraction: Enhancing Personalized Diarization and Extraction on Complex Recordings (He Zhao et al., 2024)

{{<citation>}}

He Zhao, Hangting Chen, Jianwei Yu, Yuehai Wang. (2024)  
**Continuous Target Speech Extraction: Enhancing Personalized Diarization and Extraction on Complex Recordings**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.15993v1)  

---


**ABSTRACT**  
Target speaker extraction (TSE) aims to extract the target speaker's voice from the input mixture. Previous studies have concentrated on high-overlapping scenarios. However, real-world applications usually meet more complex scenarios like variable speaker overlapping and target speaker absence. In this paper, we introduces a framework to perform continuous TSE (C-TSE), comprising a target speaker voice activation detection (TSVAD) and a TSE model. This framework significantly improves TSE performance on similar speakers and enhances personalization, which is lacking in traditional diarization methods. In detail, unlike conventional TSVAD deployed to refine the diarization results, the proposed Attention-target speaker voice activation detection (A-TSVAD) directly generates timestamps of the target speaker. We also explore some different integration methods of A-TSVAD and TSE by comparing the cascaded and parallel methods. The framework's effectiveness is assessed using a range of metrics, including diarization and enhancement metrics. Our experiments demonstrate that A-TSVAD outperforms conventional methods in reducing diarization errors. Furthermore, the integration of A-TSVAD and TSE in a sequential cascaded manner further enhances extraction accuracy.

{{</citation>}}


## cs.NI (1)



### (132/132) Energy-Aware Service Offloading for Semantic Communications in Wireless Networks (Hassan Saadat et al., 2024)

{{<citation>}}

Hassan Saadat, Abdullatif Albaseer, Mohamed Abdallah, Amr Mohamed, Aiman Erbad. (2024)  
**Energy-Aware Service Offloading for Semantic Communications in Wireless Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI, NLP  
[Paper Link](http://arxiv.org/abs/2401.15924v1)  

---


**ABSTRACT**  
Today, wireless networks are becoming responsible for serving intelligent applications, such as extended reality and metaverse, holographic telepresence, autonomous transportation, and collaborative robots. Although current fifth-generation (5G) networks can provide high data rates in terms of Gigabytes/second, they cannot cope with the high demands of the aforementioned applications, especially in terms of the size of the high-quality live videos and images that need to be communicated in real-time. Therefore, with the help of artificial intelligence (AI)-based future sixth-generation (6G) networks, the semantic communication concept can provide the services demanded by these applications. Unlike Shannon's classical information theory, semantic communication urges the use of the semantics (meaningful contents) of the data in designing more efficient data communication schemes. Hence, in this paper, we model semantic communication as an energy minimization framework in heterogeneous wireless networks with respect to delay and quality-of-service constraints. Then, we propose a sub-optimal solution to the NP-hard combinatorial mixed-integer nonlinear programming problem (MINLP) by utilizing efficient techniques such as discrete optimization variables' relaxation. In addition, AI-based autoencoder and classifier are trained and deployed to perform semantic extraction, reconstruction, and classification services. Finally, we compare our proposed sub-optimal solution with different state-of-the-art methods, and the obtained results demonstrate its superiority.

{{</citation>}}
