---
draft: false
title: "arXiv @ 2023.07.01"
date: 2023-07-01
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.01"
    identifier: arxiv_20230701
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (19)](#cscl-19)
- [cs.CY (2)](#cscy-2)
- [cs.AI (6)](#csai-6)
- [cs.DC (3)](#csdc-3)
- [cs.LG (16)](#cslg-16)
- [cs.HC (6)](#cshc-6)
- [cs.CV (21)](#cscv-21)
- [eess.SY (2)](#eesssy-2)
- [cs.IR (3)](#csir-3)
- [cs.SE (3)](#csse-3)
- [cs.CR (3)](#cscr-3)
- [cs.NI (1)](#csni-1)
- [cs.RO (4)](#csro-4)
- [eess.IV (3)](#eessiv-3)
- [eess.AS (2)](#eessas-2)
- [q-bio.TO (1)](#q-bioto-1)
- [cs.SD (2)](#cssd-2)
- [cs.CE (1)](#csce-1)
- [cs.SI (1)](#cssi-1)
- [quant-ph (3)](#quant-ph-3)
- [cs.IT (1)](#csit-1)
- [cs.LO (1)](#cslo-1)

## cs.CL (19)



### (1/104) Citations as Queries: Source Attribution Using Language Models as Rerankers (Ryan Muther et al., 2023)

{{<citation>}}

Ryan Muther, David Smith. (2023)  
**Citations as Queries: Source Attribution Using Language Models as Rerankers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2306.17322v1)  

---


**ABSTRACT**  
This paper explores new methods for locating the sources used to write a text, by fine-tuning a variety of language models to rerank candidate sources. After retrieving candidates sources using a baseline BM25 retrieval model, a variety of reranking methods are tested to see how effective they are at the task of source attribution. We conduct experiments on two datasets, English Wikipedia and medieval Arabic historical writing, and employ a variety of retrieval and generation based reranking models. In particular, we seek to understand how the degree of supervision required affects the performance of various reranking models. We find that semisupervised methods can be nearly as effective as fully supervised methods while avoiding potentially costly span-level annotation of the target and source documents.

{{</citation>}}


### (2/104) LyricWhiz: Robust Multilingual Zero-shot Lyrics Transcription by Whispering to ChatGPT (Le Zhuo et al., 2023)

{{<citation>}}

Le Zhuo, Ruibin Yuan, Jiahao Pan, Yinghao Ma, Yizhi LI, Ge Zhang, Si Liu, Roger Dannenberg, Jie Fu, Chenghua Lin, Emmanouil Benetos, Wenhu Chen, Wei Xue, Yike Guo. (2023)  
**LyricWhiz: Robust Multilingual Zero-shot Lyrics Transcription by Whispering to ChatGPT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: ChatGPT, GPT, GPT-4, Multilingual  
[Paper Link](http://arxiv.org/abs/2306.17103v2)  

---


**ABSTRACT**  
We introduce LyricWhiz, a robust, multilingual, and zero-shot automatic lyrics transcription method achieving state-of-the-art performance on various lyrics transcription datasets, even in challenging genres such as rock and metal. Our novel, training-free approach utilizes Whisper, a weakly supervised robust speech recognition model, and GPT-4, today's most performant chat-based large language model. In the proposed method, Whisper functions as the "ear" by transcribing the audio, while GPT-4 serves as the "brain," acting as an annotator with a strong performance for contextualized output selection and correction. Our experiments show that LyricWhiz significantly reduces Word Error Rate compared to existing methods in English and can effectively transcribe lyrics across multiple languages. Furthermore, we use LyricWhiz to create the first publicly available, large-scale, multilingual lyrics transcription dataset with a CC-BY-NC-SA copyright license, based on MTG-Jamendo, and offer a human-annotated subset for noise level estimation and evaluation. We anticipate that our proposed method and dataset will advance the development of multilingual lyrics transcription, a challenging and emerging task.

{{</citation>}}


### (3/104) Towards Grammatical Tagging for the Legal Language of Cybersecurity (Gianpietro Castiglione et al., 2023)

{{<citation>}}

Gianpietro Castiglione, Giampaolo Bella, Daniele Francesco Santamaria. (2023)  
**Towards Grammatical Tagging for the Legal Language of Cybersecurity**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: Legal, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2306.17042v1)  

---


**ABSTRACT**  
Legal language can be understood as the language typically used by those engaged in the legal profession and, as such, it may come both in spoken or written form. Recent legislation on cybersecurity obviously uses legal language in writing, thus inheriting all its interpretative complications due to the typical abundance of cases and sub-cases as well as to the general richness in detail. This paper faces the challenge of the essential interpretation of the legal language of cybersecurity, namely of the extraction of the essential Parts of Speech (POS) from the legal documents concerning cybersecurity. The challenge is overcome by our methodology for POS tagging of legal language. It leverages state-of-the-art open-source tools for Natural Language Processing (NLP) as well as manual analysis to validate the outcomes of the tools. As a result, the methodology is automated and, arguably, general for any legal language following minor tailoring of the preprocessing step. It is demonstrated over the most relevant EU legislation on cybersecurity, namely on the NIS 2 directive, producing the first, albeit essential, structured interpretation of such a relevant document. Moreover, our findings indicate that tools such as SpaCy and ClausIE reach their limits over the legal language of the NIS 2.

{{</citation>}}


### (4/104) Classifying Crime Types using Judgment Documents from Social Media (Haoxuan Xu et al., 2023)

{{<citation>}}

Haoxuan Xu, Zeyu He, Mengfan Shen, Songning Lai, Ziqiang Han, Yifan Peng. (2023)  
**Classifying Crime Types using Judgment Documents from Social Media**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, NLP, Social Media  
[Paper Link](http://arxiv.org/abs/2306.17020v1)  

---


**ABSTRACT**  
The task of determining crime types based on criminal behavior facts has become a very important and meaningful task in social science. But the problem facing the field now is that the data samples themselves are unevenly distributed, due to the nature of the crime itself. At the same time, data sets in the judicial field are less publicly available, and it is not practical to produce large data sets for direct training. This article proposes a new training model to solve this problem through NLP processing methods. We first propose a Crime Fact Data Preprocessing Module (CFDPM), which can balance the defects of uneven data set distribution by generating new samples. Then we use a large open source dataset (CAIL-big) as our pretraining dataset and a small dataset collected by ourselves for Fine-tuning, giving it good generalization ability to unfamiliar small datasets. At the same time, we use the improved Bert model with dynamic masking to improve the model. Experiments show that the proposed method achieves state-of-the-art results on the present dataset. At the same time, the effectiveness of module CFDPM is proved by experiments. This article provides a valuable methodology contribution for classifying social science texts such as criminal behaviors. Extensive experiments on public benchmarks show that the proposed method achieves new state-of-the-art results.

{{</citation>}}


### (5/104) MEMD-ABSA: A Multi-Element Multi-Domain Dataset for Aspect-Based Sentiment Analysis (Hongjie Cai et al., 2023)

{{<citation>}}

Hongjie Cai, Nan Song, Zengzhi Wang, Qiming Xie, Qiankun Zhao, Ke Li, Siwei Wu, Shijie Liu, Jianfei Yu, Rui Xia. (2023)  
**MEMD-ABSA: A Multi-Element Multi-Domain Dataset for Aspect-Based Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2306.16956v1)  

---


**ABSTRACT**  
Aspect-based sentiment analysis is a long-standing research interest in the field of opinion mining, and in recent years, researchers have gradually shifted their focus from simple ABSA subtasks to end-to-end multi-element ABSA tasks. However, the datasets currently used in the research are limited to individual elements of specific tasks, usually focusing on in-domain settings, ignoring implicit aspects and opinions, and with a small data scale. To address these issues, we propose a large-scale Multi-Element Multi-Domain dataset (MEMD) that covers the four elements across five domains, including nearly 20,000 review sentences and 30,000 quadruples annotated with explicit and implicit aspects and opinions for ABSA research. Meanwhile, we evaluate generative and non-generative baselines on multiple ABSA subtasks under the open domain setting, and the results show that open domain ABSA as well as mining implicit aspects and opinions remain ongoing challenges to be addressed. The datasets are publicly released at \url{https://github.com/NUSTM/MEMD-ABSA}.

{{</citation>}}


### (6/104) UMASS_BioNLP at MEDIQA-Chat 2023: Can LLMs generate high-quality synthetic note-oriented doctor-patient conversations? (Junda Wang et al., 2023)

{{<citation>}}

Junda Wang, Zonghai Yao, Avijit Mitra, Samuel Osebe, Zhichao Yang, Hong Yu. (2023)  
**UMASS_BioNLP at MEDIQA-Chat 2023: Can LLMs generate high-quality synthetic note-oriented doctor-patient conversations?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLEU, ChatGPT, GPT, GPT-4, NLP, QA  
[Paper Link](http://arxiv.org/abs/2306.16931v1)  

---


**ABSTRACT**  
This paper presents UMASS_BioNLP team participation in the MEDIQA-Chat 2023 shared task for Task-A and Task-C. We focus especially on Task-C and propose a novel LLMs cooperation system named a doctor-patient loop to generate high-quality conversation data sets. The experiment results demonstrate that our approaches yield reasonable performance as evaluated by automatic metrics such as ROUGE, medical concept recall, BLEU, and Self-BLEU. Furthermore, we conducted a comparative analysis between our proposed method and ChatGPT and GPT-4. This analysis also investigates the potential of utilizing cooperation LLMs to generate high-quality datasets.

{{</citation>}}


### (7/104) Surveying (Dis)Parities and Concerns of Compute Hungry NLP Research (Ji-Ung Lee et al., 2023)

{{<citation>}}

Ji-Ung Lee, Haritz Puerto, Betty van Aken, Yuki Arase, Jessica Zosa Forde, Leon Derczynski, Andreas Rücklé, Iryna Gurevych, Roy Schwartz, Emma Strubell, Jesse Dodge. (2023)  
**Surveying (Dis)Parities and Concerns of Compute Hungry NLP Research**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2306.16900v1)  

---


**ABSTRACT**  
Many recent improvements in NLP stem from the development and use of large pre-trained language models (PLMs) with billions of parameters. Large model sizes makes computational cost one of the main limiting factors for training and evaluating such models; and has raised severe concerns about the sustainability, reproducibility, and inclusiveness for researching PLMs. These concerns are often based on personal experiences and observations. However, there had not been any large-scale surveys that investigate them. In this work, we provide a first attempt to quantify these concerns regarding three topics, namely, environmental impact, equity, and impact on peer reviewing. By conducting a survey with 312 participants from the NLP community, we capture existing (dis)parities between different and within groups with respect to seniority, academia, and industry; and their impact on the peer reviewing process. For each topic, we provide an analysis and devise recommendations to mitigate found disparities, some of which already successfully implemented. Finally, we discuss additional concerns raised by many participants in free-text responses.

{{</citation>}}


### (8/104) Tokenization and the Noiseless Channel (Vilém Zouhar et al., 2023)

{{<citation>}}

Vilém Zouhar, Clara Meister, Juan Luis Gastaldi, Li Du, Mrinmaya Sachan, Ryan Cotterell. (2023)  
**Tokenization and the Noiseless Channel**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IT, cs.CL, math-IT  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2306.16842v1)  

---


**ABSTRACT**  
Subword tokenization is a key part of many NLP pipelines. However, little is known about why some tokenizer and hyperparameter combinations lead to better downstream model performance than others. We propose that good tokenizers lead to \emph{efficient} channel usage, where the channel is the means by which some input is conveyed to the model and efficiency can be quantified in information-theoretic terms as the ratio of the Shannon entropy to the maximum possible entropy of the token distribution. Yet, an optimal encoding according to Shannon entropy assigns extremely long codes to low-frequency tokens and very short codes to high-frequency tokens. Defining efficiency in terms of R\'enyi entropy, on the other hand, penalizes distributions with either very high or very low-frequency tokens. In machine translation, we find that across multiple tokenizers, the R\'enyi entropy with $\alpha = 2.5$ has a very strong correlation with \textsc{Bleu}: $0.78$ in comparison to just $-0.32$ for compressed length.

{{</citation>}}


### (9/104) A Formal Perspective on Byte-Pair Encoding (Vilém Zouhar et al., 2023)

{{<citation>}}

Vilém Zouhar, Clara Meister, Juan Luis Gastaldi, Li Du, Tim Vieira, Mrinmaya Sachan, Ryan Cotterell. (2023)  
**A Formal Perspective on Byte-Pair Encoding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, math-OC  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2306.16837v1)  

---


**ABSTRACT**  
Byte-Pair Encoding (BPE) is a popular algorithm used for tokenizing data in NLP, despite being devised initially as a compression method. BPE appears to be a greedy algorithm at face value, but the underlying optimization problem that BPE seeks to solve has not yet been laid down. We formalize BPE as a combinatorial optimization problem. Via submodular functions, we prove that the iterative greedy version is a $\frac{1}{{\sigma(\boldsymbol{\mu}^\star)}}(1-e^{-{\sigma(\boldsymbol{\mu}^\star)}})$-approximation of an optimal merge sequence, where ${\sigma(\boldsymbol{\mu}^\star)}$ is the total backward curvature with respect to the optimal merge sequence $\boldsymbol{\mu}^\star$. Empirically the lower bound of the approximation is $\approx 0.37$.   We provide a faster implementation of BPE which improves the runtime complexity from $\mathcal{O}\left(N M\right)$ to $\mathcal{O}\left(N \log M\right)$, where $N$ is the sequence length and $M$ is the merge count. Finally, we optimize the brute-force algorithm for optimal BPE using memoization.

{{</citation>}}


### (10/104) Benchmarking Large Language Model Capabilities for Conditional Generation (Joshua Maynez et al., 2023)

{{<citation>}}

Joshua Maynez, Priyanka Agrawal, Sebastian Gehrmann. (2023)  
**Benchmarking Large Language Model Capabilities for Conditional Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2306.16793v1)  

---


**ABSTRACT**  
Pre-trained large language models (PLMs) underlie most new developments in natural language processing. They have shifted the field from application-specific model pipelines to a single model that is adapted to a wide range of tasks. Autoregressive PLMs like GPT-3 or PaLM, alongside techniques like few-shot learning, have additionally shifted the output modality to generation instead of classification or regression. Despite their ubiquitous use, the generation quality of language models is rarely evaluated when these models are introduced. Additionally, it is unclear how existing generation tasks--while they can be used to compare systems at a high level--relate to the real world use cases for which people have been adopting them. In this work, we discuss how to adapt existing application-specific generation benchmarks to PLMs and provide an in-depth, empirical study of the limitations and capabilities of PLMs in natural language generation tasks along dimensions such as scale, architecture, input and output language. Our results show that PLMs differ in their applicability to different data regimes and their generalization to multiple languages and inform which PLMs to use for a given generation task setup. We share best practices to be taken into consideration when benchmarking generation capabilities during the development of upcoming PLMs.

{{</citation>}}


### (11/104) Stop Pre-Training: Adapt Visual-Language Models to Unseen Languages (Yasmine Karoui et al., 2023)

{{<citation>}}

Yasmine Karoui, Rémi Lebret, Negar Foroutan, Karl Aberer. (2023)  
**Stop Pre-Training: Adapt Visual-Language Models to Unseen Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2306.16774v1)  

---


**ABSTRACT**  
Vision-Language Pre-training (VLP) has advanced the performance of many vision-language tasks, such as image-text retrieval, visual entailment, and visual reasoning. The pre-training mostly utilizes lexical databases and image queries in English. Previous work has demonstrated that the pre-training in English does not transfer well to other languages in a zero-shot setting. However, multilingual pre-trained language models (MPLM) have excelled at a variety of single-modal language tasks. In this paper, we propose a simple yet efficient approach to adapt VLP to unseen languages using MPLM. We utilize a cross-lingual contextualized token embeddings alignment approach to train text encoders for non-English languages. Our approach does not require image input and primarily uses machine translation, eliminating the need for target language data. Our evaluation across three distinct tasks (image-text retrieval, visual entailment, and natural language visual reasoning) demonstrates that this approach outperforms the state-of-the-art multilingual vision-language models without requiring large parallel corpora. Our code is available at https://github.com/Yasminekaroui/CliCoTea.

{{</citation>}}


### (12/104) DialoGPS: Dialogue Path Sampling in Continuous Semantic Space for Data Augmentation in Multi-Turn Conversations (Ang Lv et al., 2023)

{{<citation>}}

Ang Lv, Jinpeng Li, Yuhan Chen, Xing Gao, Ji Zhang, Rui Yan. (2023)  
**DialoGPS: Dialogue Path Sampling in Continuous Semantic Space for Data Augmentation in Multi-Turn Conversations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Augmentation, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2306.16770v1)  

---


**ABSTRACT**  
In open-domain dialogue generation tasks, contexts and responses in most datasets are one-to-one mapped, violating an important many-to-many characteristic: a context leads to various responses, and a response answers multiple contexts. Without such patterns, models poorly generalize and prefer responding safely. Many attempts have been made in either multi-turn settings from a one-to-many perspective or in a many-to-many perspective but limited to single-turn settings. The major challenge to many-to-many augment multi-turn dialogues is that discretely replacing each turn with semantic similarity breaks fragile context coherence. In this paper, we propose DialoGue Path Sampling (DialoGPS) method in continuous semantic space, the first many-to-many augmentation method for multi-turn dialogues. Specifically, we map a dialogue to our extended Brownian Bridge, a special Gaussian process. We sample latent variables to form coherent dialogue paths in the continuous space. A dialogue path corresponds to a new multi-turn dialogue and is used as augmented training data. We show the effect of DialoGPS with both automatic and human evaluation.

{{</citation>}}


### (13/104) Unified Language Representation for Question Answering over Text, Tables, and Images (Bowen Yu et al., 2023)

{{<citation>}}

Bowen Yu, Cheng Fu, Haiyang Yu, Fei Huang, Yongbin Li. (2023)  
**Unified Language Representation for Question Answering over Text, Tables, and Images**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2306.16762v1)  

---


**ABSTRACT**  
When trying to answer complex questions, people often rely on multiple sources of information, such as visual, textual, and tabular data. Previous approaches to this problem have focused on designing input features or model structure in the multi-modal space, which is inflexible for cross-modal reasoning or data-efficient training. In this paper, we call for an alternative paradigm, which transforms the images and tables into unified language representations, so that we can simplify the task into a simpler textual QA problem that can be solved using three steps: retrieval, ranking, and generation, all within a language space. This idea takes advantage of the power of pre-trained language models and is implemented in a framework called Solar. Our experimental results show that Solar outperforms all existing methods by 10.6-32.3 pts on two datasets, MultimodalQA and MMCoQA, across ten different metrics. Additionally, Solar achieves the best performance on the WebQA leaderboard

{{</citation>}}


### (14/104) Evaluating Paraphrastic Robustness in Textual Entailment Models (Dhruv Verma et al., 2023)

{{<citation>}}

Dhruv Verma, Yash Kumar Lal, Shreyashee Sinha, Benjamin Van Durme, Adam Poliak. (2023)  
**Evaluating Paraphrastic Robustness in Textual Entailment Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Textual Entailment  
[Paper Link](http://arxiv.org/abs/2306.16722v1)  

---


**ABSTRACT**  
We present PaRTE, a collection of 1,126 pairs of Recognizing Textual Entailment (RTE) examples to evaluate whether models are robust to paraphrasing. We posit that if RTE models understand language, their predictions should be consistent across inputs that share the same meaning. We use the evaluation set to determine if RTE models' predictions change when examples are paraphrased. In our experiments, contemporary models change their predictions on 8-16\% of paraphrased examples, indicating that there is still room for improvement.

{{</citation>}}


### (15/104) Automatic Speech Recognition of Non-Native Child Speech for Language Learning Applications (Simone Wills et al., 2023)

{{<citation>}}

Simone Wills, Yu Bai, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik. (2023)  
**Automatic Speech Recognition of Non-Native Child Speech for Language Learning Applications**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: AI, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2306.16710v1)  

---


**ABSTRACT**  
Voicebots have provided a new avenue for supporting the development of language skills, particularly within the context of second language learning. Voicebots, though, have largely been geared towards native adult speakers. We sought to assess the performance of two state-of-the-art ASR systems, Wav2Vec2.0 and Whisper AI, with a view to developing a voicebot that can support children acquiring a foreign language. We evaluated their performance on read and extemporaneous speech of native and non-native Dutch children. We also investigated the utility of using ASR technology to provide insight into the children's pronunciation and fluency. The results show that recent, pre-trained ASR transformer-based models achieve acceptable performance from which detailed feedback on phoneme pronunciation quality can be extracted, despite the challenging nature of child and non-native speech.

{{</citation>}}


### (16/104) ZeroGen: Zero-shot Multimodal Controllable Text Generation with Multiple Oracles (Haoqin Tu et al., 2023)

{{<citation>}}

Haoqin Tu, Bowen Yang, Xianfeng Zhao. (2023)  
**ZeroGen: Zero-shot Multimodal Controllable Text Generation with Multiple Oracles**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2306.16649v1)  

---


**ABSTRACT**  
Automatically generating textual content with desired attributes is an ambitious task that people have pursued long. Existing works have made a series of progress in incorporating unimodal controls into language models (LMs), whereas how to generate controllable sentences with multimodal signals and high efficiency remains an open question. To tackle the puzzle, we propose a new paradigm of zero-shot controllable text generation with multimodal signals (\textsc{ZeroGen}). Specifically, \textsc{ZeroGen} leverages controls of text and image successively from token-level to sentence-level and maps them into a unified probability space at decoding, which customizes the LM outputs by weighted addition without extra training. To achieve better inter-modal trade-offs, we further introduce an effective dynamic weighting mechanism to regulate all control weights. Moreover, we conduct substantial experiments to probe the relationship of being in-depth or in-width between signals from distinct modalities. Encouraging empirical results on three downstream tasks show that \textsc{ZeroGen} not only outperforms its counterparts on captioning tasks by a large margin but also shows great potential in multimodal news generation with a higher degree of control. Our code will be released at https://github.com/ImKeTT/ZeroGen.

{{</citation>}}


### (17/104) Probabilistic Linguistic Knowledge and Token-level Text Augmentation (Zhengxiang Wang, 2023)

{{<citation>}}

Zhengxiang Wang. (2023)  
**Probabilistic Linguistic Knowledge and Token-level Text Augmentation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2306.16644v2)  

---


**ABSTRACT**  
This paper investigates the effectiveness of token-level text augmentation and the role of probabilistic linguistic knowledge within a linguistically-motivated evaluation context. Two text augmentation programs, REDA and REDA$_{NG}$, were developed, both implementing five token-level text editing operations: Synonym Replacement (SR), Random Swap (RS), Random Insertion (RI), Random Deletion (RD), and Random Mix (RM). REDA$_{NG}$ leverages pretrained $n$-gram language models to select the most likely augmented texts from REDA's output. Comprehensive and fine-grained experiments were conducted on a binary question matching classification task in both Chinese and English. The results strongly refute the general effectiveness of the five token-level text augmentation techniques under investigation, whether applied together or separately, and irrespective of various common classification model types used, including transformers. Furthermore, the role of probabilistic linguistic knowledge is found to be minimal.

{{</citation>}}


### (18/104) A negation detection assessment of GPTs: analysis with the xNot360 dataset (Ha Thanh Nguyen et al., 2023)

{{<citation>}}

Ha Thanh Nguyen, Randy Goebel, Francesca Toni, Kostas Stathis, Ken Satoh. (2023)  
**A negation detection assessment of GPTs: analysis with the xNot360 dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Transformer  
[Paper Link](http://arxiv.org/abs/2306.16638v1)  

---


**ABSTRACT**  
Negation is a fundamental aspect of natural language, playing a critical role in communication and comprehension. Our study assesses the negation detection performance of Generative Pre-trained Transformer (GPT) models, specifically GPT-2, GPT-3, GPT-3.5, and GPT-4. We focus on the identification of negation in natural language using a zero-shot prediction approach applied to our custom xNot360 dataset. Our approach examines sentence pairs labeled to indicate whether the second sentence negates the first. Our findings expose a considerable performance disparity among the GPT models, with GPT-4 surpassing its counterparts and GPT-3.5 displaying a marked performance reduction. The overall proficiency of the GPT models in negation detection remains relatively modest, indicating that this task pushes the boundaries of their natural language understanding capabilities. We not only highlight the constraints of GPT models in handling negation but also emphasize the importance of logical reliability in high-stakes domains such as healthcare, science, and law.

{{</citation>}}


### (19/104) CMATH: Can Your Language Model Pass Chinese Elementary School Math Test? (Tianwen Wei et al., 2023)

{{<citation>}}

Tianwen Wei, Jian Luan, Wei Liu, Shuang Dong, Bin Wang. (2023)  
**CMATH: Can Your Language Model Pass Chinese Elementary School Math Test?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2306.16636v1)  

---


**ABSTRACT**  
We present the Chinese Elementary School Math Word Problems (CMATH) dataset, comprising 1.7k elementary school-level math word problems with detailed annotations, source from actual Chinese workbooks and exams. This dataset aims to provide a benchmark tool for assessing the following question: to what grade level of elementary school math do the abilities of popular large language models (LLMs) correspond? We evaluate a variety of popular LLMs, including both commercial and open-source options, and discover that only GPT-4 achieves success (accuracy $\geq$ 60\%) across all six elementary school grades, while other models falter at different grade levels. Furthermore, we assess the robustness of several top-performing LLMs by augmenting the original problems in the CMATH dataset with distracting information. Our findings reveal that GPT-4 is able to maintains robustness, while other model fail. We anticipate that our study will expose limitations in LLMs' arithmetic and reasoning capabilities, and promote their ongoing development and advancement.

{{</citation>}}


## cs.CY (2)



### (20/104) Tube2Vec: Social and Semantic Embeddings of YouTube Channels (Léopaul Boesinger et al., 2023)

{{<citation>}}

Léopaul Boesinger, Manoel Horta Ribeiro, Veniamin Veselovsky, Robert West. (2023)  
**Tube2Vec: Social and Semantic Embeddings of YouTube Channels**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2306.17298v1)  

---


**ABSTRACT**  
Research using YouTube data often explores social and semantic dimensions of channels and videos. Typically, analyses rely on laborious manual annotation of content and content creators, often found by low-recall methods such as keyword search. Here, we explore an alternative approach, using latent representations (embeddings) obtained via machine learning. Using a large dataset of YouTube links shared on Reddit; we create embeddings that capture social sharing behavior, video metadata (title, description, etc.), and YouTube's video recommendations. We evaluate these embeddings using crowdsourcing and existing datasets, finding that recommendation embeddings excel at capturing both social and semantic dimensions, although social-sharing embeddings better correlate with existing partisan scores. We share embeddings capturing the social and semantic dimensions of 44,000 YouTube channels for the benefit of future research on YouTube: https://github.com/epfl-dlab/youtube-embeddings.

{{</citation>}}


### (21/104) Generative AI for Programming Education: Benchmarking ChatGPT, GPT-4, and Human Tutors (Tung Phung et al., 2023)

{{<citation>}}

Tung Phung, Victor-Alexandru Pădurean, José Cambronero, Sumit Gulwani, Tobias Kohn, Rupak Majumdar, Adish Singla, Gustavo Soares. (2023)  
**Generative AI for Programming Education: Benchmarking ChatGPT, GPT-4, and Human Tutors**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CL, cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT, GPT-3.5, GPT-4, Generative AI  
[Paper Link](http://arxiv.org/abs/2306.17156v2)  

---


**ABSTRACT**  
Generative AI and large language models hold great promise in enhancing computing education by powering next-generation educational technologies for introductory programming. Recent works have studied these models for different scenarios relevant to programming education; however, these works are limited for several reasons, as they typically consider already outdated models or only specific scenario(s). Consequently, there is a lack of a systematic study that benchmarks state-of-the-art models for a comprehensive set of programming education scenarios. In our work, we systematically evaluate two models, ChatGPT (based on GPT-3.5) and GPT-4, and compare their performance with human tutors for a variety of scenarios. We evaluate using five introductory Python programming problems and real-world buggy programs from an online platform, and assess performance using expert-based annotations. Our results show that GPT-4 drastically outperforms ChatGPT (based on GPT-3.5) and comes close to human tutors' performance for several scenarios. These results also highlight settings where GPT-4 still struggles, providing exciting future directions on developing techniques to improve the performance of these models.

{{</citation>}}


## cs.AI (6)



### (22/104) A Neural Separation Algorithm for the Rounded Capacity Inequalities (Hyeonah Kim et al., 2023)

{{<citation>}}

Hyeonah Kim, Jinkyoo Park, Changhyun Kwon. (2023)  
**A Neural Separation Algorithm for the Rounded Capacity Inequalities**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, math-OC  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2306.17283v1)  

---


**ABSTRACT**  
The cutting plane method is a key technique for successful branch-and-cut and branch-price-and-cut algorithms that find the exact optimal solutions for various vehicle routing problems (VRPs). Among various cuts, the rounded capacity inequalities (RCIs) are the most fundamental. To generate RCIs, we need to solve the separation problem, whose exact solution takes a long time to obtain; therefore, heuristic methods are widely used. We design a learning-based separation heuristic algorithm with graph coarsening that learns the solutions of the exact separation problem with a graph neural network (GNN), which is trained with small instances of 50 to 100 customers. We embed our separation algorithm within the cutting plane method to find a lower bound for the capacitated VRP (CVRP) with up to 1,000 customers. We compare the performance of our approach with CVRPSEP, a popular separation software package for various cuts used in solving VRPs. Our computational results show that our approach finds better lower bounds than CVRPSEP for large-scale problems with 400 or more customers, while CVRPSEP shows strong competency for problems with less than 400 customers.

{{</citation>}}


### (23/104) Suffering Toasters -- A New Self-Awareness Test for AI (Ira Wolfson, 2023)

{{<citation>}}

Ira Wolfson. (2023)  
**Suffering Toasters -- A New Self-Awareness Test for AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.17258v2)  

---


**ABSTRACT**  
A widely accepted definition of intelligence in the context of Artificial Intelligence (AI) still eludes us. Due to our exceedingly rapid development of AI paradigms, architectures, and tools, the prospect of naturally arising AI consciousness seems more likely than ever. In this paper, we claim that all current intelligence tests are insufficient to point to the existence or lack of intelligence \textbf{as humans intuitively perceive it}. We draw from ideas in the philosophy of science, psychology, and other areas of research to provide a clearer definition of the problems of artificial intelligence, self-awareness, and agency. We furthermore propose a new heuristic approach to test for artificial self-awareness and outline a possible implementation. Finally, we discuss some of the questions that arise from this new heuristic, be they philosophical or implementation-oriented.

{{</citation>}}


### (24/104) Interdisciplinary Methods in Computational Creativity: How Human Variables Shape Human-Inspired AI Research (Nadia M. Ady et al., 2023)

{{<citation>}}

Nadia M. Ady, Faun Rice. (2023)  
**Interdisciplinary Methods in Computational Creativity: How Human Variables Shape Human-Inspired AI Research**  

---
Primary Category: cs.AI  
Categories: I-2-0; K-7-0, cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.17070v1)  

---


**ABSTRACT**  
The word creativity originally described a concept from human psychology, but in the realm of computational creativity (CC), it has become much more. The question of what creativity means when it is part of a computational system might be considered core to CC. Pinning down the meaning of creativity, and concepts like it, becomes salient when researchers port concepts from human psychology to computation, a widespread practice extending beyond CC into artificial intelligence (AI). Yet, the human processes shaping human-inspired computational systems have been little investigated. In this paper, we question which human literatures (social sciences, psychology, neuroscience) enter AI scholarship and how they are translated at the port of entry. This study is based on 22 in-depth, semi-structured interviews, primarily with human-inspired AI researchers, half of whom focus on creativity as a major research area. This paper focuses on findings most relevant to CC. We suggest that which human literature enters AI bears greater scrutiny because ideas may become disconnected from context in their home discipline. Accordingly, we recommend that CC researchers document the decisions and context of their practices, particularly those practices formalizing human concepts for machines. Publishing reflexive commentary on human elements in CC and AI would provide a useful record and permit greater dialogue with other disciplines.

{{</citation>}}


### (25/104) The mapKurator System: A Complete Pipeline for Extracting and Linking Text from Historical Maps (Jina Kim et al., 2023)

{{<citation>}}

Jina Kim, Zekun Li, Yijun Lin, Min Namgung, Leeje Jang, Yao-Yi Chiang. (2023)  
**The mapKurator System: A Complete Pipeline for Extracting and Linking Text from Historical Maps**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs.AI  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2306.17059v2)  

---


**ABSTRACT**  
Scanned historical maps in libraries and archives are valuable repositories of geographic data that often do not exist elsewhere. Despite the potential of machine learning tools like the Google Vision APIs for automatically transcribing text from these maps into machine-readable formats, they do not work well with large-sized images (e.g., high-resolution scanned documents), cannot infer the relation between the recognized text and other datasets, and are challenging to integrate with post-processing tools. This paper introduces the mapKurator system, an end-to-end system integrating machine learning models with a comprehensive data processing pipeline. mapKurator empowers automated extraction, post-processing, and linkage of text labels from large numbers of large-dimension historical map scans. The output data, comprising bounding polygons and recognized text, is in the standard GeoJSON format, making it easily modifiable within Geographic Information Systems (GIS). The proposed system allows users to quickly generate valuable data from large numbers of historical maps for in-depth analysis of the map content and, in turn, encourages map findability, accessibility, interoperability, and reusability (FAIR principles). We deployed the mapKurator system and enabled the processing of over 60,000 maps and over 100 million text/place names in the David Rumsey Historical Map collection. We also demonstrated a seamless integration of mapKurator with a collaborative web platform to enable accessing automated approaches for extracting and linking text labels from historical map scans and collective work to improve the results.

{{</citation>}}


### (26/104) Exploring & Exploiting High-Order Graph Structure for Sparse Knowledge Graph Completion (Tao He et al., 2023)

{{<citation>}}

Tao He, Ming Liu, Yixin Cao, Zekun Wang, Zihao Zheng, Zheng Chu, Bing Qin. (2023)  
**Exploring & Exploiting High-Order Graph Structure for Sparse Knowledge Graph Completion**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: GNN, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2306.17034v1)  

---


**ABSTRACT**  
Sparse knowledge graph (KG) scenarios pose a challenge for previous Knowledge Graph Completion (KGC) methods, that is, the completion performance decreases rapidly with the increase of graph sparsity. This problem is also exacerbated because of the widespread existence of sparse KGs in practical applications. To alleviate this challenge, we present a novel framework, LR-GCN, that is able to automatically capture valuable long-range dependency among entities to supplement insufficient structure features and distill logical reasoning knowledge for sparse KGC. The proposed approach comprises two main components: a GNN-based predictor and a reasoning path distiller. The reasoning path distiller explores high-order graph structures such as reasoning paths and encodes them as rich-semantic edges, explicitly compositing long-range dependencies into the predictor. This step also plays an essential role in densifying KGs, effectively alleviating the sparse issue. Furthermore, the path distiller further distills logical reasoning knowledge from these mined reasoning paths into the predictor. These two components are jointly optimized using a well-designed variational EM algorithm. Extensive experiments and analyses on four sparse benchmarks demonstrate the effectiveness of our proposed method.

{{</citation>}}


### (27/104) From Query Tools to Causal Architects: Harnessing Large Language Models for Advanced Causal Discovery from Data (Taiyu Ban et al., 2023)

{{<citation>}}

Taiyu Ban, Lyvzhou Chen, Xiangyu Wang, Huanhuan Chen. (2023)  
**From Query Tools to Causal Architects: Harnessing Large Language Models for Advanced Causal Discovery from Data**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2306.16902v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) exhibit exceptional abilities for causal analysis between concepts in numerous societally impactful domains, including medicine, science, and law. Recent research on LLM performance in various causal discovery and inference tasks has given rise to a new ladder in the classical three-stage framework of causality. In this paper, we advance the current research of LLM-driven causal discovery by proposing a novel framework that combines knowledge-based LLM causal analysis with data-driven causal structure learning. To make LLM more than a query tool and to leverage its power in discovering natural and new laws of causality, we integrate the valuable LLM expertise on existing causal mechanisms into statistical analysis of objective data to build a novel and practical baseline for causal structure learning.   We introduce a universal set of prompts designed to extract causal graphs from given variables and assess the influence of LLM prior causality on recovering causal structures from data. We demonstrate the significant enhancement of LLM expertise on the quality of recovered causal structures from data, while also identifying critical challenges and issues, along with potential approaches to address them. As a pioneering study, this paper aims to emphasize the new frontier that LLMs are opening for classical causal discovery and inference, and to encourage the widespread adoption of LLM capabilities in data-driven causal analysis.

{{</citation>}}


## cs.DC (3)



### (28/104) Modeling Parallel Programs using Large Language Models (Daniel Nichols et al., 2023)

{{<citation>}}

Daniel Nichols, Aniruddha Marathe, Harshitha Menon, Todd Gamblin, Abhinav Bhatele. (2023)  
**Modeling Parallel Programs using Large Language Models**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs.DC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2306.17281v1)  

---


**ABSTRACT**  
Parallel software codes in high performance computing (HPC) continue to grow in complexity and scale as we enter the exascale era. A diverse set of emerging hardware and programming paradigms make developing, optimizing, and maintaining parallel software burdensome for developers. One way to alleviate some of these burdens is with automated development and analysis tools. Such tools can perform complex and/or remedial tasks for developers that increase their productivity and decrease the chance for error. So far, such tools for code development and performance analysis have been limited in the complexity of tasks they can perform. However, with recent advancements in language modeling, and the wealth of code related data that is now available online, these tools have started to utilize predictive language models to automate more complex tasks. In this paper, we show how large language models (LLMs) can be applied to tasks specific to high performance and scientific codes. We train LLMs using code and performance data that is specific to parallel codes. We compare several recent LLMs on HPC related tasks and introduce a new model, HPC-Coder, trained on parallel code. In our experiments we show that this model can auto-complete HPC functions where general models cannot, decorate for loops with OpenMP pragmas, and model performance changes in two scientific application repositories.

{{</citation>}}


### (29/104) When Edge Meets FaaS: Opportunities and Challenges (Runyu Jin et al., 2023)

{{<citation>}}

Runyu Jin, Qirui Yang, Ming Zhao. (2023)  
**When Edge Meets FaaS: Opportunities and Challenges**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AWS  
[Paper Link](http://arxiv.org/abs/2307.06397v1)  

---


**ABSTRACT**  
The proliferation of edge devices and the rapid growth of IoT data have called forth the edge computing paradigm. Function-as-a-service (FaaS) is a promising computing paradigm to realize edge computing. This paper explores the feasibility and advantages of FaaS-based edge computing. It also studies the research challenges that should be addressed in the design of such systems, which are 1) the quick decomposing and recomposing of applications, 2) the trade-off between performance and isolation of sandbox mechanisms, and 3) distributed scheduling. The challenges are illustrated by evaluating existing FaaS-based edge platforms, AWS IoT Greengrass, and OpenFaaS.

{{</citation>}}


### (30/104) SRL: Scaling Distributed Reinforcement Learning to Over Ten Thousand Cores (Zhiyu Mei et al., 2023)

{{<citation>}}

Zhiyu Mei, Wei Fu, Guangju Wang, Huanchen Zhang, Yi Wu. (2023)  
**SRL: Scaling Distributed Reinforcement Learning to Over Ten Thousand Cores**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs-LG, cs.DC  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2306.16688v2)  

---


**ABSTRACT**  
The ever-growing complexity of reinforcement learning (RL) tasks demands a distributed RL system to efficiently generate and process a massive amount of data to train intelligent agents. However, existing open-source libraries suffer from various limitations, which impede their practical use in challenging scenarios where large-scale training is necessary. While industrial systems from OpenAI and DeepMind have achieved successful large-scale RL training, their system architecture and implementation details remain undisclosed to the community. In this paper, we present a novel abstraction on the dataflows of RL training, which unifies practical RL training across diverse applications into a general framework and enables fine-grained optimizations. Following this abstraction, we develop a scalable, efficient, and extensible distributed RL system called ReaLly Scalable RL (SRL). The system architecture of SRL separates major RL computation components and allows massively parallelized training. Moreover, SRL offers user-friendly and extensible interfaces for customized algorithms. Our evaluation shows that SRL outperforms existing academic libraries in both a single machine and a medium-sized cluster. In a large-scale cluster, the novel architecture of SRL leads to up to 3.7x speedup compared to the design choices adopted by the existing libraries. We also conduct a direct benchmark comparison to OpenAI's industrial system, Rapid, in the challenging hide-and-seek environment. SRL reproduces the same solution as reported by OpenAI with up to 5x speedup in wall-clock time. Furthermore, we also examine the performance of SRL in a much harder variant of the hide-and-seek environment and achieve substantial learning speedup by scaling SRL to over 15k CPU cores and 32 A100 GPUs. Notably, SRL is the first in the academic community to perform RL experiments at such a large scale.

{{</citation>}}


## cs.LG (16)



### (31/104) Probabilistic Constraint for Safety-Critical Reinforcement Learning (Weiqin Chen et al., 2023)

{{<citation>}}

Weiqin Chen, Dharmashankar Subramanian, Santiago Paternain. (2023)  
**Probabilistic Constraint for Safety-Critical Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2306.17279v1)  

---


**ABSTRACT**  
In this paper, we consider the problem of learning safe policies for probabilistic-constrained reinforcement learning (RL). Specifically, a safe policy or controller is one that, with high probability, maintains the trajectory of the agent in a given safe set. We establish a connection between this probabilistic-constrained setting and the cumulative-constrained formulation that is frequently explored in the existing literature. We provide theoretical bounds elucidating that the probabilistic-constrained setting offers a better trade-off in terms of optimality and safety (constraint satisfaction). The challenge encountered when dealing with the probabilistic constraints, as explored in this work, arises from the absence of explicit expressions for their gradients. Our prior work provides such an explicit gradient expression for probabilistic constraints which we term Safe Policy Gradient-REINFORCE (SPG-REINFORCE). In this work, we provide an improved gradient SPG-Actor-Critic that leads to a lower variance than SPG-REINFORCE, which is substantiated by our theoretical results. A noteworthy aspect of both SPGs is their inherent algorithm independence, rendering them versatile for application across a range of policy-based algorithms. Furthermore, we propose a Safe Primal-Dual algorithm that can leverage both SPGs to learn safe policies. It is subsequently followed by theoretical analyses that encompass the convergence of the algorithm, as well as the near-optimality and feasibility on average. In addition, we test the proposed approaches by a series of empirical experiments. These experiments aim to examine and analyze the inherent trade-offs between the optimality and safety, and serve to substantiate the efficacy of two SPGs, as well as our theoretical contributions.

{{</citation>}}


### (32/104) DisasterResponseGPT: Large Language Models for Accelerated Plan of Action Development in Disaster Response Scenarios (Vinicius G. Goecks et al., 2023)

{{<citation>}}

Vinicius G. Goecks, Nicholas R. Waytowich. (2023)  
**DisasterResponseGPT: Large Language Models for Accelerated Plan of Action Development in Disaster Response Scenarios**  

---
Primary Category: cs.LG  
Categories: I-2-7; J-7; K-4-0, cs-LG, cs.LG  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2306.17271v1)  

---


**ABSTRACT**  
The development of plans of action in disaster response scenarios is a time-consuming process. Large Language Models (LLMs) offer a powerful solution to expedite this process through in-context learning. This study presents DisasterResponseGPT, an algorithm that leverages LLMs to generate valid plans of action quickly by incorporating disaster response and planning guidelines in the initial prompt. In DisasterResponseGPT, users input the scenario description and receive a plan of action as output. The proposed method generates multiple plans within seconds, which can be further refined following the user's feedback. Preliminary results indicate that the plans of action developed by DisasterResponseGPT are comparable to human-generated ones while offering greater ease of modification in real-time. This approach has the potential to revolutionize disaster response operations by enabling rapid updates and adjustments during the plan's execution.

{{</citation>}}


### (33/104) Prediction of COVID-19 Patients' Emergency Room Revisit using Multi-Source Transfer Learning (Yuelyu Ji et al., 2023)

{{<citation>}}

Yuelyu Ji, Yuhe Gao, Runxue Bao, Qi Li, Disheng Liu, Yiming Sun, Ye Ye. (2023)  
**Prediction of COVID-19 Patients' Emergency Room Revisit using Multi-Source Transfer Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2306.17257v1)  

---


**ABSTRACT**  
The coronavirus disease 2019 (COVID-19) has led to a global pandemic of significant severity. In addition to its high level of contagiousness, COVID-19 can have a heterogeneous clinical course, ranging from asymptomatic carriers to severe and potentially life-threatening health complications. Many patients have to revisit the emergency room (ER) within a short time after discharge, which significantly increases the workload for medical staff. Early identification of such patients is crucial for helping physicians focus on treating life-threatening cases. In this study, we obtained Electronic Health Records (EHRs) of 3,210 encounters from 13 affiliated ERs within the University of Pittsburgh Medical Center between March 2020 and January 2021. We leveraged a Natural Language Processing technique, ScispaCy, to extract clinical concepts and used the 1001 most frequent concepts to develop 7-day revisit models for COVID-19 patients in ERs. The research data we collected from 13 ERs may have distributional differences that could affect the model development. To address this issue, we employed a classic deep transfer learning method called the Domain Adversarial Neural Network (DANN) and evaluated different modeling strategies, including the Multi-DANN algorithm, the Single-DANN algorithm, and three baseline methods. Results showed that the Multi-DANN models outperformed the Single-DANN models and baseline models in predicting revisits of COVID-19 patients to the ER within 7 days after discharge. Notably, the Multi-DANN strategy effectively addressed the heterogeneity among multiple source domains and improved the adaptation of source data to the target domain. Moreover, the high performance of Multi-DANN models indicates that EHRs are informative for developing a prediction model to identify COVID-19 patients who are very likely to revisit an ER within 7 days after discharge.

{{</citation>}}


### (34/104) Synthetic Demographic Data Generation for Card Fraud Detection Using GANs (Shuo Wang et al., 2023)

{{<citation>}}

Shuo Wang, Terrence Tricco, Xianta Jiang, Charles Robertson, John Hawkin. (2023)  
**Synthetic Demographic Data Generation for Card Fraud Detection Using GANs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Fraud Detection  
[Paper Link](http://arxiv.org/abs/2306.17109v1)  

---


**ABSTRACT**  
Using machine learning models to generate synthetic data has become common in many fields. Technology to generate synthetic transactions that can be used to detect fraud is also growing fast. Generally, this synthetic data contains only information about the transaction, such as the time, place, and amount of money. It does not usually contain the individual user's characteristics (age and gender are occasionally included). Using relatively complex synthetic demographic data may improve the complexity of transaction data features, thus improving the fraud detection performance. Benefiting from developments of machine learning, some deep learning models have potential to perform better than other well-established synthetic data generation methods, such as microsimulation. In this study, we built a deep-learning Generative Adversarial Network (GAN), called DGGAN, which will be used for demographic data generation. Our model generates samples during model training, which we found important to overcame class imbalance issues. This study can help improve the cognition of synthetic data and further explore the application of synthetic data generation in card fraud detection.

{{</citation>}}


### (35/104) RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark (Federico Berto et al., 2023)

{{<citation>}}

Federico Berto, Chuanbo Hua, Junyoung Park, Minsu Kim, Hyeonah Kim, Jiwoo Son, Haeyeon Kim, Joungho Kim, Jinkyoo Park. (2023)  
**RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2306.17100v1)  

---


**ABSTRACT**  
We introduce RL4CO, an extensive reinforcement learning (RL) for combinatorial optimization (CO) benchmark. RL4CO employs state-of-the-art software libraries as well as best practices in implementation, such as modularity and configuration management, to be efficient and easily modifiable by researchers for adaptations of neural network architecture, environments, and algorithms. Contrary to the existing focus on specific tasks like the traveling salesman problem (TSP) for performance assessment, we underline the importance of scalability and generalization capabilities for diverse optimization tasks. We also systematically benchmark sample efficiency, zero-shot generalization, and adaptability to changes in data distributions of various models. Our experiments show that some recent state-of-the-art methods fall behind their predecessors when evaluated using these new metrics, suggesting the necessity for a more balanced view of the performance of neural CO solvers. We hope RL4CO will encourage the exploration of novel solutions to complex real-world tasks, allowing to compare with existing methods through a standardized interface that decouples the science from the software engineering. We make our library publicly available at https://github.com/kaist-silab/rl4co.

{{</citation>}}


### (36/104) Sparsity exploitation via discovering graphical models in multi-variate time-series forecasting (Ngoc-Dung Do et al., 2023)

{{<citation>}}

Ngoc-Dung Do, Truong Son Hy, Duy Khuong Nguyen. (2023)  
**Sparsity exploitation via discovering graphical models in multi-variate time-series forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2306.17090v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have been widely applied in multi-variate time-series forecasting (MTSF) tasks because of their capability in capturing the correlations among different time-series. These graph-based learning approaches improve the forecasting performance by discovering and understanding the underlying graph structures, which represent the data correlation. When the explicit prior graph structures are not available, most existing works cannot guarantee the sparsity of the generated graphs that make the overall model computational expensive and less interpretable. In this work, we propose a decoupled training method, which includes a graph generating module and a GNNs forecasting module. First, we use Graphical Lasso (or GraphLASSO) to directly exploit the sparsity pattern from data to build graph structures in both static and time-varying cases. Second, we fit these graph structures and the input data into a Graph Convolutional Recurrent Network (GCRN) to train a forecasting model. The experimental results on three real-world datasets show that our novel approach has competitive performance against existing state-of-the-art forecasting algorithms while providing sparse, meaningful and explainable graph structures and reducing training time by approximately 40%. Our PyTorch implementation is publicly available at https://github.com/HySonLab/GraphLASSO

{{</citation>}}


### (37/104) Concept-Oriented Deep Learning with Large Language Models (Daniel T. Chang, 2023)

{{<citation>}}

Daniel T. Chang. (2023)  
**Concept-Oriented Deep Learning with Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2306.17089v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have been successfully used in many natural-language tasks and applications including text generation and AI chatbots. They also are a promising new technology for concept-oriented deep learning (CODL). However, the prerequisite is that LLMs understand concepts and ensure conceptual consistency. We discuss these in this paper, as well as major uses of LLMs for CODL including concept extraction from text, concept graph extraction from text, and concept learning. Human knowledge consists of both symbolic (conceptual) knowledge and embodied (sensory) knowledge. Text-only LLMs, however, can represent only symbolic (conceptual) knowledge. Multimodal LLMs, on the other hand, are capable of representing the full range (conceptual and sensory) of human knowledge. We discuss conceptual understanding in visual-language LLMs, the most important multimodal LLMs, and major uses of them for CODL including concept extraction from image, concept graph extraction from image, and concept learning. While uses of LLMs for CODL are valuable standalone, they are particularly valuable as part of LLM applications such as AI chatbots.

{{</citation>}}


### (38/104) Safe Model-Based Multi-Agent Mean-Field Reinforcement Learning (Matej Jusup et al., 2023)

{{<citation>}}

Matej Jusup, Barna Pásztor, Tadeusz Janik, Kenan Zhang, Francesco Corman, Andreas Krause, Ilija Bogunovic. (2023)  
**Safe Model-Based Multi-Agent Mean-Field Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2306.17052v1)  

---


**ABSTRACT**  
Many applications, e.g., in shared mobility, require coordinating a large number of agents. Mean-field reinforcement learning addresses the resulting scalability challenge by optimizing the policy of a representative agent. In this paper, we address an important generalization where there exist global constraints on the distribution of agents (e.g., requiring capacity constraints or minimum coverage requirements to be met). We propose Safe-$\text{M}^3$-UCRL, the first model-based algorithm that attains safe policies even in the case of unknown transition dynamics. As a key ingredient, it uses epistemic uncertainty in the transition model within a log-barrier approach to ensure pessimistic constraints satisfaction with high probability. We showcase Safe-$\text{M}^3$-UCRL on the vehicle repositioning problem faced by many shared mobility operators and evaluate its performance through simulations built on Shenzhen taxi trajectory data. Our algorithm effectively meets the demand in critical areas while ensuring service accessibility in regions with low demand.

{{</citation>}}


### (39/104) Safety-Aware Task Composition for Discrete and Continuous Reinforcement Learning (Kevin Leahy et al., 2023)

{{<citation>}}

Kevin Leahy, Makai Mann, Zachary Serlin. (2023)  
**Safety-Aware Task Composition for Discrete and Continuous Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2306.17033v1)  

---


**ABSTRACT**  
Compositionality is a critical aspect of scalable system design. Reinforcement learning (RL) has recently shown substantial success in task learning, but has only recently begun to truly leverage composition. In this paper, we focus on Boolean composition of learned tasks as opposed to functional or sequential composition. Existing Boolean composition for RL focuses on reaching a satisfying absorbing state in environments with discrete action spaces, but does not support composable safety (i.e., avoidance) constraints. We advance the state of the art in Boolean composition of learned tasks with three contributions: i) introduce two distinct notions of safety in this framework; ii) show how to enforce either safety semantics, prove correctness (under some assumptions), and analyze the trade-offs between the two safety notions; and iii) extend Boolean composition from discrete action spaces to continuous action spaces. We demonstrate these techniques using modified versions of value iteration in a grid world, Deep Q-Network (DQN) in a grid world with image observations, and Twin Delayed DDPG (TD3) in a continuous-observation and continuous-action Bullet physics environment. We believe that these contributions advance the theory of safe reinforcement learning by allowing zero-shot composition of policies satisfying safety properties.

{{</citation>}}


### (40/104) Diffusion-Jump GNNs: Homophiliation via Learnable Metric Filters (Ahmed Begga et al., 2023)

{{<citation>}}

Ahmed Begga, Francisco Escolano, Miguel Angel Lozano, Edwin R. Hancock. (2023)  
**Diffusion-Jump GNNs: Homophiliation via Learnable Metric Filters**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2306.16976v1)  

---


**ABSTRACT**  
High-order Graph Neural Networks (HO-GNNs) have been developed to infer consistent latent spaces in the heterophilic regime, where the label distribution is not correlated with the graph structure. However, most of the existing HO-GNNs are hop-based, i.e., they rely on the powers of the transition matrix. As a result, these architectures are not fully reactive to the classification loss and the achieved structural filters have static supports. In other words, neither the filters' supports nor their coefficients can be learned with these networks. They are confined, instead, to learn combinations of filters. To address the above concerns, we propose Diffusion-jump GNNs a method relying on asymptotic diffusion distances that operates on jumps. A diffusion-pump generates pairwise distances whose projections determine both the support and coefficients of each structural filter. These filters are called jumps because they explore a wide range of scales in order to find bonds between scattered nodes with the same label. Actually, the full process is controlled by the classification loss. Both the jumps and the diffusion distances react to classification errors (i.e. they are learnable). Homophiliation, i.e., the process of learning piecewise smooth latent spaces in the heterophilic regime, is formulated as a Dirichlet problem: the known labels determine the border nodes and the diffusion-pump ensures a minimal deviation of the semi-supervised grouping from a canonical unsupervised grouping. This triggers the update of both the diffusion distances and, consequently, the jumps in order to minimize the classification error. The Dirichlet formulation has several advantages. It leads to the definition of structural heterophily, a novel measure beyond edge heterophily. It also allows us to investigate links with (learnable) diffusion distances, absorbing random walks and stochastic diffusion.

{{</citation>}}


### (41/104) Learning Environment Models with Continuous Stochastic Dynamics (Martin Tappler et al., 2023)

{{<citation>}}

Martin Tappler, Edi Muškardin, Bernhard K. Aichernig, Bettina Könighofer. (2023)  
**Learning Environment Models with Continuous Stochastic Dynamics**  

---
Primary Category: cs.LG  
Categories: cs-FL, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.17204v1)  

---


**ABSTRACT**  
Solving control tasks in complex environments automatically through learning offers great potential. While contemporary techniques from deep reinforcement learning (DRL) provide effective solutions, their decision-making is not transparent. We aim to provide insights into the decisions faced by the agent by learning an automaton model of environmental behavior under the control of an agent. However, for most control problems, automata learning is not scalable enough to learn a useful model. In this work, we raise the capabilities of automata learning such that it is possible to learn models for environments that have complex and continuous dynamics.   The core of the scalability of our method lies in the computation of an abstract state-space representation, by applying dimensionality reduction and clustering on the observed environmental state space. The stochastic transitions are learned via passive automata learning from observed interactions of the agent and the environment. In an iterative model-based RL process, we sample additional trajectories to learn an accurate environment model in the form of a discrete-state Markov decision process (MDP). We apply our automata learning framework on popular RL benchmarking environments in the OpenAI Gym, including LunarLander, CartPole, Mountain Car, and Acrobot. Our results show that the learned models are so precise that they enable the computation of policies solving the respective control tasks. Yet the models are more concise and more general than neural-network-based policies and by using MDPs we benefit from a wealth of tools available for analyzing them. When solving the task of LunarLander, the learned model even achieved similar or higher rewards than deep RL policies learned with stable-baselines3.

{{</citation>}}


### (42/104) Surgical Phase and Instrument Recognition: How to identify appropriate Dataset Splits (Georgii Kostiuchik et al., 2023)

{{<citation>}}

Georgii Kostiuchik, Lalith Sharan, Benedikt Mayer, Ivo Wolf, Bernhard Preim, Sandy Engelhardt. (2023)  
**Surgical Phase and Instrument Recognition: How to identify appropriate Dataset Splits**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.16879v1)  

---


**ABSTRACT**  
Purpose: The development of machine learning models for surgical workflow and instrument recognition from temporal data represents a challenging task due to the complex nature of surgical workflows. In particular, the imbalanced distribution of data is one of the major challenges in the domain of surgical workflow recognition. In order to obtain meaningful results, careful partitioning of data into training, validation, and test sets, as well as the selection of suitable evaluation metrics are crucial. Methods: In this work, we present an openly available web-based application that enables interactive exploration of dataset partitions. The proposed visual framework facilitates the assessment of dataset splits for surgical workflow recognition, especially with regard to identifying sub-optimal dataset splits. Currently, it supports visualization of surgical phase and instrument annotations. Results: In order to validate the dedicated interactive visualizations, we use a dataset split of the Cholec80 dataset. This dataset split was specifically selected to reflect a case of strong data imbalance. Using our software, we were able to identify phases, phase transitions, and combinations of surgical instruments that were not represented in one of the sets. Conclusion: In order to obtain meaningful results in highly unbalanced class distributions, special care should be taken with respect to the selection of an appropriate split. Interactive data visualization represents a promising approach for the assessment of machine learning datasets. The source code is available at https://github.com/Cardio-AI/endovis-ml

{{</citation>}}


### (43/104) Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging (Max Zimmer et al., 2023)

{{<citation>}}

Max Zimmer, Christoph Spiegel, Sebastian Pokutta. (2023)  
**Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2306.16788v1)  

---


**ABSTRACT**  
Neural networks can be significantly compressed by pruning, leading to sparse models requiring considerably less storage and floating-point operations while maintaining predictive performance. Model soups (Wortsman et al., 2022) improve generalization and out-of-distribution performance by averaging the parameters of multiple models into a single one without increased inference time. However, identifying models in the same loss basin to leverage both sparsity and parameter averaging is challenging, as averaging arbitrary sparse models reduces the overall sparsity due to differing sparse connectivities. In this work, we address these challenges by demonstrating that exploring a single retraining phase of Iterative Magnitude Pruning (IMP) with varying hyperparameter configurations, such as batch ordering or weight decay, produces models that are suitable for averaging and share the same sparse connectivity by design. Averaging these models significantly enhances generalization performance compared to their individual components. Building on this idea, we introduce Sparse Model Soups (SMS), a novel method for merging sparse models by initiating each prune-retrain cycle with the averaged model of the previous phase. SMS maintains sparsity, exploits sparse network benefits being modular and fully parallelizable, and substantially improves IMP's performance. Additionally, we demonstrate that SMS can be adapted to enhance the performance of state-of-the-art pruning during training approaches.

{{</citation>}}


### (44/104) Graph Sampling-based Meta-Learning for Molecular Property Prediction (Xiang Zhuang et al., 2023)

{{<citation>}}

Xiang Zhuang, Qiang Zhang, Bin Wu, Keyan Ding, Yin Fang, Huajun Chen. (2023)  
**Graph Sampling-based Meta-Learning for Molecular Property Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-BM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.16780v1)  

---


**ABSTRACT**  
Molecular property is usually observed with a limited number of samples, and researchers have considered property prediction as a few-shot problem. One important fact that has been ignored by prior works is that each molecule can be recorded with several different properties simultaneously. To effectively utilize many-to-many correlations of molecules and properties, we propose a Graph Sampling-based Meta-learning (GS-Meta) framework for few-shot molecular property prediction. First, we construct a Molecule-Property relation Graph (MPG): molecule and properties are nodes, while property labels decide edges. Then, to utilize the topological information of MPG, we reformulate an episode in meta-learning as a subgraph of the MPG, containing a target property node, molecule nodes, and auxiliary property nodes. Third, as episodes in the form of subgraphs are no longer independent of each other, we propose to schedule the subgraph sampling process with a contrastive loss function, which considers the consistency and discrimination of subgraphs. Extensive experiments on 5 commonly-used benchmarks show GS-Meta consistently outperforms state-of-the-art methods by 5.71%-6.93% in ROC-AUC and verify the effectiveness of each proposed module. Our code is available at https://github.com/HICAI-ZJU/GS-Meta.

{{</citation>}}


### (45/104) Eigensubspace of Temporal-Difference Dynamics and How It Improves Value Approximation in Reinforcement Learning (Qiang He et al., 2023)

{{<citation>}}

Qiang He, Tianyi Zhou, Meng Fang, Setareh Maghsudi. (2023)  
**Eigensubspace of Temporal-Difference Dynamics and How It Improves Value Approximation in Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2306.16750v1)  

---


**ABSTRACT**  
We propose a novel value approximation method, namely Eigensubspace Regularized Critic (ERC) for deep reinforcement learning (RL). ERC is motivated by an analysis of the dynamics of Q-value approximation error in the Temporal-Difference (TD) method, which follows a path defined by the 1-eigensubspace of the transition kernel associated with the Markov Decision Process (MDP). It reveals a fundamental property of TD learning that has remained unused in previous deep RL approaches. In ERC, we propose a regularizer that guides the approximation error tending towards the 1-eigensubspace, resulting in a more efficient and stable path of value approximation. Moreover, we theoretically prove the convergence of the ERC method. Besides, theoretical analysis and experiments demonstrate that ERC effectively reduces the variance of value functions. Among 26 tasks in the DMControl benchmark, ERC outperforms state-of-the-art methods for 20. Besides, it shows significant advantages in Q-value approximation and variance reduction. Our code is available at https://sites.google.com/view/erc-ecml23/.

{{</citation>}}


### (46/104) Game Level Blending using a Learned Level Representation (Venkata Sai Revanth Atmakuri et al., 2023)

{{<citation>}}

Venkata Sai Revanth Atmakuri, Seth Cooper, Matthew Guzdial. (2023)  
**Game Level Blending using a Learned Level Representation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2306.16666v1)  

---


**ABSTRACT**  
Game level blending via machine learning, the process of combining features of game levels to create unique and novel game levels using Procedural Content Generation via Machine Learning (PCGML) techniques, has gained increasing popularity in recent years. However, many existing techniques rely on human-annotated level representations, which limits game level blending to a limited number of annotated games. Even with annotated games, researchers often need to author an additional shared representation to make blending possible. In this paper, we present a novel approach to game level blending that employs Clustering-based Tile Embeddings (CTE), a learned level representation technique that can serve as a level representation for unannotated games and a unified level representation across games without the need for human annotation. CTE represents game level tiles as a continuous vector representation, unifying their visual, contextual, and behavioral information. We apply this approach to two classic Nintendo games, Lode Runner and The Legend of Zelda. We run an evaluation comparing the CTE representation to a common, human-annotated representation in the blending task and find that CTE has comparable or better performance without the need for human annotation.

{{</citation>}}


## cs.HC (6)



### (47/104) Towards Anatomy Education with Generative AI-based Virtual Assistants in Immersive Virtual Reality Environments (Vuthea Chheang et al., 2023)

{{<citation>}}

Vuthea Chheang, Rommy Marquez-Hernandez, Megha Patel, Danush Rajasekaran, Shayla Sharmin, Gavin Caulfield, Behdokht Kiafar, Jicheng Li, Roghayeh Leila Barmaki. (2023)  
**Towards Anatomy Education with Generative AI-based Virtual Assistants in Immersive Virtual Reality Environments**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2306.17278v1)  

---


**ABSTRACT**  
Anatomy education is essential to support medical students in understanding the morphology, location, and spatial relationships of anatomical structures. Virtual reality (VR) and interactive 3D visualization systems have been proposed to provide an engaging learning experience and environment. However, VR-based systems integrated with a generative artificial intelligence (AI) assistant for anatomy education are still underrepresented. This work presents a VR environment with a generative AI virtual assistant to support human anatomy education, allowing the user to communicate verbally with the virtual assistant. We aim to provide a more interactive, adaptive, and informative learning experience. The proposed environment was assessed in a pilot user study (n = 16) with a comparison of two configurations: avatar and screen-based virtual assistant. We observed no significant difference between the configurations and difficulty level in the task completion time and the number of interactions with the virtual assistant. However, there was a significant difference in the score between the difficulty level in the avatar configuration. The results also provide insights into the usability, task load, and sense of presence in the virtual environment. Our proposed environment offers potential benefits and research directions for medical education, using generative AI to assist and enhance the learning experience.

{{</citation>}}


### (48/104) Evaluation of AI-Supported Input Methods in Augmented Reality Environment (Akos Nagy et al., 2023)

{{<citation>}}

Akos Nagy, Thomas Lagkas, Panagiotis Sarigiannidis, Vasileios Argyriou. (2023)  
**Evaluation of AI-Supported Input Methods in Augmented Reality Environment**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Augmentation  
[Paper Link](http://arxiv.org/abs/2306.17132v1)  

---


**ABSTRACT**  
Augmented Reality (AR) solutions are providing tools that could improve applications in the medical and industrial fields. Augmentation can provide additional information in training, visualization, and work scenarios, to increase efficiency, reliability, and safety, while improving communication with other devices and systems on the network. Unfortunately, tasks in these fields often require both hands to execute, reducing the variety of input methods suitable to control AR applications. People with certain physical disabilities, where they are not able to use their hands, are also negatively impacted when using these devices. The goal of this work is to provide novel hand-free interfacing methods, using AR technology, in association with AI support approaches to produce an improved Human-Computer interaction solution.

{{</citation>}}


### (49/104) AI-Powered Interfaces for Extended Reality to support Remote Maintenance (Akos Nagy et al., 2023)

{{<citation>}}

Akos Nagy, George Amponis, Konstantinos Kyranou, Thomas Lagkas, Alexandros Apostolos Boulogeorgos, Panagiotis Sarigiannidis, Vasileios Argyriou. (2023)  
**AI-Powered Interfaces for Extended Reality to support Remote Maintenance**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.16961v1)  

---


**ABSTRACT**  
High-end components that conduct complicated tasks automatically are a part of modern industrial systems. However, in order for these parts to function at the desired level, they need to be maintained by qualified experts. Solutions based on Augmented Reality (AR) have been established with the goal of raising production rates and quality while lowering maintenance costs. With the introduction of two unique interaction interfaces based on wearable targets and human face orientation, we are proposing hands-free advanced interactive solutions in this study with the goal of reducing the bias towards certain users. Using traditional devices in real time, a comparison investigation using alternative interaction interfaces is conducted. The suggested solutions are supported by various AI powered methods such as novel gravity-map based motion adjustment that is made possible by predictive deep models that reduce the bias of traditional hand- or finger-based interaction interfaces

{{</citation>}}


### (50/104) LeanAI: A method for AEC practitioners to effectively plan AI implementations (Ashwin Agrawal et al., 2023)

{{<citation>}}

Ashwin Agrawal, Vishal Singh, Martin Fischer. (2023)  
**LeanAI: A method for AEC practitioners to effectively plan AI implementations**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CY, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.16799v1)  

---


**ABSTRACT**  
Recent developments in Artificial Intelligence (AI) provide unprecedented automation opportunities in the Architecture, Engineering, and Construction (AEC) industry. However, despite the enthusiasm regarding the use of AI, 85% of current big data projects fail. One of the main reasons for AI project failures in the AEC industry is the disconnect between those who plan or decide to use AI and those who implement it. AEC practitioners often lack a clear understanding of the capabilities and limitations of AI, leading to a failure to distinguish between what AI should solve, what it can solve, and what it will solve, treating these categories as if they are interchangeable. This lack of understanding results in the disconnect between AI planning and implementation because the planning is based on a vision of what AI should solve without considering if it can or will solve it. To address this challenge, this work introduces the LeanAI method. The method has been developed using data from several ongoing longitudinal studies analyzing AI implementations in the AEC industry, which involved 50+ hours of interview data. The LeanAI method delineates what AI should solve, what it can solve, and what it will solve, forcing practitioners to clearly articulate these components early in the planning process itself by involving the relevant stakeholders. By utilizing the method, practitioners can effectively plan AI implementations, thus increasing the likelihood of success and ultimately speeding up the adoption of AI. A case example illustrates the usefulness of the method.

{{</citation>}}


### (51/104) The Future of AI-Assisted Writing (Carlos Alves Pereira et al., 2023)

{{<citation>}}

Carlos Alves Pereira, Tanay Komarlu, Wael Mobeirek. (2023)  
**The Future of AI-Assisted Writing**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2306.16641v1)  

---


**ABSTRACT**  
The development of Natural Language Generation models has led to the creation of powerful Artificial Intelligence-assisted writing tools. These tools are capable of predicting users' needs and actively providing suggestions as they write. In this work, we conduct a comparative user-study between such tools from an information retrieval lens: pull and push. Specifically, we investigate the user demand of AI-assisted writing, the impact of the two paradigms on quality, ownership of the writing product, and efficiency and enjoyment of the writing process. We also seek to understand the impact of bias of AI-assisted writing. Our findings show that users welcome seamless assistance of AI in their writing. Furthermore, AI helped users to diversify the ideas in their writing while keeping it clear and concise more quickly. Users also enjoyed the collaboration with AI-assisted writing tools and did not feel a lack of ownership. Finally, although participants did not experience bias in our experiments, they still expressed explicit and clear concerns that should be addressed in future AI-assisted writing tools.

{{</citation>}}


### (52/104) Evaluating ChatGPT's Decimal Skills and Feedback Generation in a Digital Learning Game (Huy A. Nguyen et al., 2023)

{{<citation>}}

Huy A. Nguyen, Hayden Stec, Xinying Hou, Sarah Di, Bruce M. McLaren. (2023)  
**Evaluating ChatGPT's Decimal Skills and Feedback Generation in a Digital Learning Game**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CY, cs-HC, cs.HC  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2306.16639v1)  

---


**ABSTRACT**  
While open-ended self-explanations have been shown to promote robust learning in multiple studies, they pose significant challenges to automated grading and feedback in technology-enhanced learning, due to the unconstrained nature of the students' input. Our work investigates whether recent advances in Large Language Models, and in particular ChatGPT, can address this issue. Using decimal exercises and student data from a prior study of the learning game Decimal Point, with more than 5,000 open-ended self-explanation responses, we investigate ChatGPT's capability in (1) solving the in-game exercises, (2) determining the correctness of students' answers, and (3) providing meaningful feedback to incorrect answers. Our results showed that ChatGPT can respond well to conceptual questions, but struggled with decimal place values and number line problems. In addition, it was able to accurately assess the correctness of 75% of the students' answers and generated generally high-quality feedback, similar to human instructors. We conclude with a discussion of ChatGPT's strengths and weaknesses and suggest several venues for extending its use cases in digital teaching and learning.

{{</citation>}}


## cs.CV (21)



### (53/104) Towards Zero-Shot Scale-Aware Monocular Depth Estimation (Vitor Guizilini et al., 2023)

{{<citation>}}

Vitor Guizilini, Igor Vasiljevic, Dian Chen, Rares Ambrus, Adrien Gaidon. (2023)  
**Towards Zero-Shot Scale-Aware Monocular Depth Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2306.17253v1)  

---


**ABSTRACT**  
Monocular depth estimation is scale-ambiguous, and thus requires scale supervision to produce metric predictions. Even so, the resulting models will be geometry-specific, with learned scales that cannot be directly transferred across domains. Because of that, recent works focus instead on relative depth, eschewing scale in favor of improved up-to-scale zero-shot transfer. In this work we introduce ZeroDepth, a novel monocular depth estimation framework capable of predicting metric scale for arbitrary test images from different domains and camera parameters. This is achieved by (i) the use of input-level geometric embeddings that enable the network to learn a scale prior over objects; and (ii) decoupling the encoder and decoder stages, via a variational latent representation that is conditioned on single frame information. We evaluated ZeroDepth targeting both outdoor (KITTI, DDAD, nuScenes) and indoor (NYUv2) benchmarks, and achieved a new state-of-the-art in both settings using the same pre-trained model, outperforming methods that train on in-domain data and require test-time scaling to produce metric estimates.

{{</citation>}}


### (54/104) An Efficient General-Purpose Modular Vision Model via Multi-Task Heterogeneous Training (Zitian Chen et al., 2023)

{{<citation>}}

Zitian Chen, Mingyu Ding, Yikang Shen, Wei Zhan, Masayoshi Tomizuka, Erik Learned-Miller, Chuang Gan. (2023)  
**An Efficient General-Purpose Modular Vision Model via Multi-Task Heterogeneous Training**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2306.17165v1)  

---


**ABSTRACT**  
We present a model that can perform multiple vision tasks and can be adapted to other downstream tasks efficiently. Despite considerable progress in multi-task learning, most efforts focus on learning from multi-label data: a single image set with multiple task labels. Such multi-label data sets are rare, small, and expensive. We say heterogeneous to refer to image sets with different task labels, or to combinations of single-task datasets. Few have explored training on such heterogeneous datasets. General-purpose vision models are still dominated by single-task pretraining, and it remains unclear how to scale up multi-task models by leveraging mainstream vision datasets designed for different purposes. The challenges lie in managing large intrinsic differences among vision tasks, including data distribution, architectures, task-specific modules, dataset scales, and sampling strategies. To address these challenges, we propose to modify and scale up mixture-of-experts (MoE) vision transformers, so that they can simultaneously learn classification, detection, and segmentation on diverse mainstream vision datasets including ImageNet, COCO, and ADE20K. Our approach achieves comparable results to single-task state-of-the-art models and demonstrates strong generalization on downstream tasks. Due to its emergent modularity, this general-purpose model decomposes into high-performing components, efficiently adapting to downstream tasks. We can fine-tune it with fewer training parameters, fewer model parameters, and less computation. Additionally, its modularity allows for easy expansion in continual-learning-without-forgetting scenarios. Finally, these functions can be controlled and combined to meet various demands of downstream tasks.

{{</citation>}}


### (55/104) Learning Nuclei Representations with Masked Image Modelling (Piotr Wójcik et al., 2023)

{{<citation>}}

Piotr Wójcik, Hussein Naji, Adrian Simon, Reinhard Büttner, Katarzyna Bożek. (2023)  
**Learning Nuclei Representations with Masked Image Modelling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2306.17116v1)  

---


**ABSTRACT**  
Masked image modelling (MIM) is a powerful self-supervised representation learning paradigm, whose potential has not been widely demonstrated in medical image analysis. In this work, we show the capacity of MIM to capture rich semantic representations of Haemotoxylin & Eosin (H&E)-stained images at the nuclear level. Inspired by Bidirectional Encoder representation from Image Transformers (BEiT), we split the images into smaller patches and generate corresponding discrete visual tokens. In addition to the regular grid-based patches, typically used in visual Transformers, we introduce patches of individual cell nuclei. We propose positional encoding of the irregular distribution of these structures within an image. We pre-train the model in a self-supervised manner on H&E-stained whole-slide images of diffuse large B-cell lymphoma, where cell nuclei have been segmented. The pre-training objective is to recover the original discrete visual tokens of the masked image on the one hand, and to reconstruct the visual tokens of the masked object instances on the other. Coupling these two pre-training tasks allows us to build powerful, context-aware representations of nuclei. Our model generalizes well and can be fine-tuned on downstream classification tasks, achieving improved cell classification accuracy on PanNuke dataset by more than 5% compared to current instance segmentation methods.

{{</citation>}}


### (56/104) LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding (Yanzhe Zhang et al., 2023)

{{<citation>}}

Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, Tong Sun. (2023)  
**LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: AI, GPT, GPT-4, Language Model, OCR, QA  
[Paper Link](http://arxiv.org/abs/2306.17107v1)  

---


**ABSTRACT**  
Instruction tuning unlocks the superior capability of Large Language Models (LLM) to interact with humans. Furthermore, recent instruction-following datasets include images as visual inputs, collecting responses for image-based instructions. However, visual instruction-tuned models cannot comprehend textual details within images well. This work enhances the current visual instruction tuning pipeline with text-rich images (e.g., movie posters, book covers, etc.). Specifically, we first use publicly available OCR tools to collect results on 422K text-rich images from the LAION dataset. Moreover, we prompt text-only GPT-4 with recognized texts and image captions to generate 16K conversations, each containing question-answer pairs for text-rich images. By combining our collected data with previous multi-modal instruction-following data, our model, LLaVAR, substantially improves the LLaVA model's capability on text-based VQA datasets (up to 20% accuracy improvement) while achieving an accuracy of 91.42% on ScienceQA. The GPT-4-based instruction-following evaluation also demonstrates the improvement of our model on both natural images and text-rich images. Through qualitative analysis, LLaVAR shows promising interaction (e.g., reasoning, writing, and elaboration) skills with humans based on the latest real-world online content that combines text and images. We make our code/data/models publicly available at https://llavar.github.io/.

{{</citation>}}


### (57/104) Deep Ensemble for Rotorcraft Attitude Prediction (Hikmat Khan et al., 2023)

{{<citation>}}

Hikmat Khan, Nidhal Carla Bouaynaya, Ghulam Rasool, Tyler Travis, Lacey Thompson, Charles C. Johnson. (2023)  
**Deep Ensemble for Rotorcraft Attitude Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.17104v1)  

---


**ABSTRACT**  
Historically, the rotorcraft community has experienced a higher fatal accident rate than other aviation segments, including commercial and general aviation. Recent advancements in artificial intelligence (AI) and the application of these technologies in different areas of our lives are both intriguing and encouraging. When developed appropriately for the aviation domain, AI techniques provide an opportunity to help design systems that can address rotorcraft safety challenges. Our recent work demonstrated that AI algorithms could use video data from onboard cameras and correctly identify different flight parameters from cockpit gauges, e.g., indicated airspeed. These AI-based techniques provide a potentially cost-effective solution, especially for small helicopter operators, to record the flight state information and perform post-flight analyses. We also showed that carefully designed and trained AI systems could accurately predict rotorcraft attitude (i.e., pitch and yaw) from outside scenes (images or video data). Ordinary off-the-shelf video cameras were installed inside the rotorcraft cockpit to record the outside scene, including the horizon. The AI algorithm could correctly identify rotorcraft attitude at an accuracy in the range of 80\%. In this work, we combined five different onboard camera viewpoints to improve attitude prediction accuracy to 94\%. In this paper, five onboard camera views included the pilot windshield, co-pilot windshield, pilot Electronic Flight Instrument System (EFIS) display, co-pilot EFIS display, and the attitude indicator gauge. Using video data from each camera view, we trained various convolutional neural networks (CNNs), which achieved prediction accuracy in the range of 79\% % to 90\% %. We subsequently ensembled the learned knowledge from all CNNs and achieved an ensembled accuracy of 93.3\%.

{{</citation>}}


### (58/104) Detect Any Deepfakes: Segment Anything Meets Face Forgery Detection and Localization (Yingxin Lai et al., 2023)

{{<citation>}}

Yingxin Lai, Zhiming Luo, Zitong Yu. (2023)  
**Detect Any Deepfakes: Segment Anything Meets Face Forgery Detection and Localization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2306.17075v1)  

---


**ABSTRACT**  
The rapid advancements in computer vision have stimulated remarkable progress in face forgery techniques, capturing the dedicated attention of researchers committed to detecting forgeries and precisely localizing manipulated areas. Nonetheless, with limited fine-grained pixel-wise supervision labels, deepfake detection models perform unsatisfactorily on precise forgery detection and localization. To address this challenge, we introduce the well-trained vision segmentation foundation model, i.e., Segment Anything Model (SAM) in face forgery detection and localization. Based on SAM, we propose the Detect Any Deepfakes (DADF) framework with the Multiscale Adapter, which can capture short- and long-range forgery contexts for efficient fine-tuning. Moreover, to better identify forged traces and augment the model's sensitivity towards forgery regions, Reconstruction Guided Attention (RGA) module is proposed. The proposed framework seamlessly integrates end-to-end forgery localization and detection optimization. Extensive experiments on three benchmark datasets demonstrate the superiority of our approach for both forgery detection and localization. The codes will be released soon at https://github.com/laiyingxin2/DADF.

{{</citation>}}


### (59/104) Learning Structure-Guided Diffusion Model for 2D Human Pose Estimation (Zhongwei Qiu et al., 2023)

{{<citation>}}

Zhongwei Qiu, Qiansheng Yang, Jian Wang, Xiyu Wang, Chang Xu, Dongmei Fu, Kun Yao, Junyu Han, Errui Ding, Jingdong Wang. (2023)  
**Learning Structure-Guided Diffusion Model for 2D Human Pose Estimation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2306.17074v1)  

---


**ABSTRACT**  
One of the mainstream schemes for 2D human pose estimation (HPE) is learning keypoints heatmaps by a neural network. Existing methods typically improve the quality of heatmaps by customized architectures, such as high-resolution representation and vision Transformers. In this paper, we propose \textbf{DiffusionPose}, a new scheme that formulates 2D HPE as a keypoints heatmaps generation problem from noised heatmaps. During training, the keypoints are diffused to random distribution by adding noises and the diffusion model learns to recover ground-truth heatmaps from noised heatmaps with respect to conditions constructed by image feature. During inference, the diffusion model generates heatmaps from initialized heatmaps in a progressive denoising way. Moreover, we further explore improving the performance of DiffusionPose with conditions from human structural information. Extensive experiments show the prowess of our DiffusionPose, with improvements of 1.6, 1.2, and 1.2 mAP on widely-used COCO, CrowdPose, and AI Challenge datasets, respectively.

{{</citation>}}


### (60/104) MotionTrack: End-to-End Transformer-based Multi-Object Tracing with LiDAR-Camera Fusion (Ce Zhang et al., 2023)

{{<citation>}}

Ce Zhang, Chengjie Zhang, Yiluan Guo, Lingji Chen, Michael Happold. (2023)  
**MotionTrack: End-to-End Transformer-based Multi-Object Tracing with LiDAR-Camera Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2306.17000v1)  

---


**ABSTRACT**  
Multiple Object Tracking (MOT) is crucial to autonomous vehicle perception. End-to-end transformer-based algorithms, which detect and track objects simultaneously, show great potential for the MOT task. However, most existing methods focus on image-based tracking with a single object category. In this paper, we propose an end-to-end transformer-based MOT algorithm (MotionTrack) with multi-modality sensor inputs to track objects with multiple classes. Our objective is to establish a transformer baseline for the MOT in an autonomous driving environment. The proposed algorithm consists of a transformer-based data association (DA) module and a transformer-based query enhancement module to achieve MOT and Multiple Object Detection (MOD) simultaneously. The MotionTrack and its variations achieve better results (AMOTA score at 0.55) on the nuScenes dataset compared with other classical baseline models, such as the AB3DMOT, the CenterTrack, and the probabilistic 3D Kalman filter. In addition, we prove that a modified attention mechanism can be utilized for DA to accomplish the MOT, and aggregate history features to enhance the MOD performance.

{{</citation>}}


### (61/104) Spectral Batch Normalization: Normalization in the Frequency Domain (Rinor Cakaj et al., 2023)

{{<citation>}}

Rinor Cakaj, Jens Mehnert, Bin Yang. (2023)  
**Spectral Batch Normalization: Normalization in the Frequency Domain**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2306.16999v1)  

---


**ABSTRACT**  
Regularization is a set of techniques that are used to improve the generalization ability of deep neural networks. In this paper, we introduce spectral batch normalization (SBN), a novel effective method to improve generalization by normalizing feature maps in the frequency (spectral) domain. The activations of residual networks without batch normalization (BN) tend to explode exponentially in the depth of the network at initialization. This leads to extremely large feature map norms even though the parameters are relatively small. These explosive dynamics can be very detrimental to learning. BN makes weight decay regularization on the scaling factors $\gamma, \beta$ approximately equivalent to an additive penalty on the norm of the feature maps, which prevents extremely large feature map norms to a certain degree. However, we show experimentally that, despite the approximate additive penalty of BN, feature maps in deep neural networks (DNNs) tend to explode at the beginning of the network and that feature maps of DNNs contain large values during the whole training. This phenomenon also occurs in a weakened form in non-residual networks. SBN addresses large feature maps by normalizing them in the frequency domain. In our experiments, we empirically show that SBN prevents exploding feature maps at initialization and large feature map values during the training. Moreover, the normalization of feature maps in the frequency domain leads to more uniform distributed frequency components. This discourages the DNNs to rely on single frequency components of feature maps. These, together with other effects of SBN, have a regularizing effect on the training of residual and non-residual networks. We show experimentally that using SBN in addition to standard regularization methods improves the performance of DNNs by a relevant margin, e.g. ResNet50 on ImageNet by 0.71%.

{{</citation>}}


### (62/104) Integrating Large Pre-trained Models into Multimodal Named Entity Recognition with Evidential Fusion (Weide Liu et al., 2023)

{{<citation>}}

Weide Liu, Xiaoyang Zhong, Jingwen Hou, Shaohua Li, Haozhe Huang, Yuming Fang. (2023)  
**Integrating Large Pre-trained Models into Multimodal Named Entity Recognition with Evidential Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NER, Named Entity Recognition, Twitter  
[Paper Link](http://arxiv.org/abs/2306.16991v1)  

---


**ABSTRACT**  
Multimodal Named Entity Recognition (MNER) is a crucial task for information extraction from social media platforms such as Twitter. Most current methods rely on attention weights to extract information from both text and images but are often unreliable and lack interpretability. To address this problem, we propose incorporating uncertainty estimation into the MNER task, producing trustworthy predictions. Our proposed algorithm models the distribution of each modality as a Normal-inverse Gamma distribution, and fuses them into a unified distribution with an evidential fusion mechanism, enabling hierarchical characterization of uncertainties and promotion of prediction accuracy and trustworthiness. Additionally, we explore the potential of pre-trained large foundation models in MNER and propose an efficient fusion approach that leverages their robust feature representations. Experiments on two datasets demonstrate that our proposed method outperforms the baselines and achieves new state-of-the-art performance.

{{</citation>}}


### (63/104) Defending Black-box Classifiers by Bayesian Boundary Correction (He Wang et al., 2023)

{{<citation>}}

He Wang, Yunfeng Diao. (2023)  
**Defending Black-box Classifiers by Bayesian Boundary Correction**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2306.16979v1)  

---


**ABSTRACT**  
Classifiers based on deep neural networks have been recently challenged by Adversarial Attack, where the widely existing vulnerability has invoked the research in defending them from potential threats. Given a vulnerable classifier, existing defense methods are mostly white-box and often require re-training the victim under modified loss functions/training regimes. While the model/data/training specifics of the victim are usually unavailable to the user, re-training is unappealing, if not impossible for reasons such as limited computational resources. To this end, we propose a new black-box defense framework. It can turn any pre-trained classifier into a resilient one with little knowledge of the model specifics. This is achieved by new joint Bayesian treatments on the clean data, the adversarial examples and the classifier, for maximizing their joint probability. It is further equipped with a new post-train strategy which keeps the victim intact. We name our framework Bayesian Boundary Correction (BBC). BBC is a general and flexible framework that can easily adapt to different data types. We instantiate BBC for image classification and skeleton-based human activity recognition, for both static and dynamic data. Exhaustive evaluation shows that BBC has superior robustness and can enhance robustness without severely hurting the clean accuracy, compared with existing defense methods.

{{</citation>}}


### (64/104) ICDaeLST: Intensity-Controllable Detail Attention-enhanced for Lightweight Fast Style Transfer (Jiang Shi Qi, 2023)

{{<citation>}}

Jiang Shi Qi. (2023)  
**ICDaeLST: Intensity-Controllable Detail Attention-enhanced for Lightweight Fast Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Attention, Style Transfer  
[Paper Link](http://arxiv.org/abs/2306.16846v1)  

---


**ABSTRACT**  
The mainstream style transfer methods usually use pre-trained deep convolutional neural network (VGG) models as encoders, or use more complex model structures to achieve better style transfer effects. This leads to extremely slow processing speeds for practical tasks due to limited resources or higher resolution image processing, such as 4K images, severely hindering the practical application value of style transfer models. We introduce a lightweight and fast styletransfer model with controllable detail attention enhancement, named ICDaeLST. The model adopts a minimal, shallow, and small architecture, forming a very compact lightweight model for efficient forward inference. Although its structure is simple and has limited parameters, we achieve better overall color and texture structure matching by introducing a style discriminator, design additional global semantic invariance loss to preserve the semantic and structural information of the content image from a high-level global perspective, and design a shallow detail attention enhancement module to preserve the detail information of the content image from a low-level detail perspective. We also achieve controllable intensity during inference for the first time (adjusting the degree of detail retention and texture structure transfer based on subjective judgment) to meet different users' subjective evaluation of stylization effects. Compared with the current best-performing and most lightweight models, our model achieves better style transfer quality and better content structure and detail retention, while having a smaller model size (17-250 times smaller) and faster speed (0.26-6.5 times faster), and achieves the fastest processing speed of 0.38s on 4K high-resolution images.

{{</citation>}}


### (65/104) Evaluation of Environmental Conditions on Object Detection using Oriented Bounding Boxes for AR Applications (Vladislav Li et al., 2023)

{{<citation>}}

Vladislav Li, Barbara Villarini, Jean-Christophe Nebel, Thomas Lagkas, Panagiotis Sarigiannidis, Vasileios Argyriou. (2023)  
**Evaluation of Environmental Conditions on Object Detection using Oriented Bounding Boxes for AR Applications**  

---
Primary Category: cs.CV  
Categories: ACM-class: I-2-10, cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2306.16798v1)  

---


**ABSTRACT**  
The objective of augmented reality (AR) is to add digital content to natural images and videos to create an interactive experience between the user and the environment. Scene analysis and object recognition play a crucial role in AR, as they must be performed quickly and accurately. In this study, a new approach is proposed that involves using oriented bounding boxes with a detection and recognition deep network to improve performance and processing time. The approach is evaluated using two datasets: a real image dataset (DOTA dataset) commonly used for computer vision tasks, and a synthetic dataset that simulates different environmental, lighting, and acquisition conditions. The focus of the evaluation is on small objects, which are difficult to detect and recognise. The results indicate that the proposed approach tends to produce better Average Precision and greater accuracy for small objects in most of the tested conditions.

{{</citation>}}


### (66/104) SaaFormer: Spectral-spatial Axial Aggregation Transformer for Hyperspectral Image Classification (Enzhe Zhao et al., 2023)

{{<citation>}}

Enzhe Zhao, Zhichang Guo, Yao Li, Dazhi Zhang. (2023)  
**SaaFormer: Spectral-spatial Axial Aggregation Transformer for Hyperspectral Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification, Transformer  
[Paper Link](http://arxiv.org/abs/2306.16759v2)  

---


**ABSTRACT**  
Hyperspectral images (HSI) captured from earth observing satellites and aircraft is becoming increasingly important for applications in agriculture, environmental monitoring, mining, etc. Due to the limited available hyperspectral datasets, the pixel-wise random sampling is the most commonly used training-test dataset partition approach, which has significant overlap between samples in training and test datasets. Furthermore, our experimental observations indicates that regions with larger overlap often exhibit higher classification accuracy. Consequently, the pixel-wise random sampling approach poses a risk of data leakage. Thus, we propose a block-wise sampling method to minimize the potential for data leakage. Our experimental findings also confirm the presence of data leakage in models such as 2DCNN. Further, We propose a spectral-spatial axial aggregation transformer model, namely SaaFormer, to address the challenges associated with hyperspectral image classifier that considers HSI as long sequential three-dimensional images. The model comprises two primary components: axial aggregation attention and multi-level spectral-spatial extraction. The axial aggregation attention mechanism effectively exploits the continuity and correlation among spectral bands at each pixel position in hyperspectral images, while aggregating spatial dimension features. This enables SaaFormer to maintain high precision even under block-wise sampling. The multi-level spectral-spatial extraction structure is designed to capture the sensitivity of different material components to specific spectral bands, allowing the model to focus on a broader range of spectral details. The results on six publicly available datasets demonstrate that our model exhibits comparable performance when using random sampling, while significantly outperforming other methods when employing block-wise sampling partition.

{{</citation>}}


### (67/104) GraMMaR: Ground-aware Motion Model for 3D Human Motion Reconstruction (Sihan Ma et al., 2023)

{{<citation>}}

Sihan Ma, Qiong Cao, Jing Zhang, Dacheng Tao. (2023)  
**GraMMaR: Ground-aware Motion Model for 3D Human Motion Reconstruction**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.16736v2)  

---


**ABSTRACT**  
Demystifying complex human-ground interactions is essential for accurate and realistic 3D human motion reconstruction from RGB videos, as it ensures consistency between the humans and the ground plane. Prior methods have modeled human-ground interactions either implicitly or in a sparse manner, often resulting in unrealistic and incorrect motions when faced with noise and uncertainty. In contrast, our approach explicitly represents these interactions in a dense and continuous manner. To this end, we propose a novel Ground-aware Motion Model for 3D Human Motion Reconstruction, named GraMMaR, which jointly learns the distribution of transitions in both pose and interaction between every joint and ground plane at each time step of a motion sequence. It is trained to explicitly promote consistency between the motion and distance change towards the ground. After training, we establish a joint optimization strategy that utilizes GraMMaR as a dual-prior, regularizing the optimization towards the space of plausible ground-aware motions. This leads to realistic and coherent motion reconstruction, irrespective of the assumed or learned ground plane. Through extensive evaluation on the AMASS and AIST++ datasets, our model demonstrates good generalization and discriminating abilities in challenging cases including complex and ambiguous human-ground interactions. The code will be released.

{{</citation>}}


### (68/104) Metric-aligned Sample Selection and Critical Feature Sampling for Oriented Object Detection (Peng Sun et al., 2023)

{{<citation>}}

Peng Sun, Yongbin Zheng, Wenqi Wu, Wanying Xu, Shengjian Bai. (2023)  
**Metric-aligned Sample Selection and Critical Feature Sampling for Oriented Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Object Detection  
[Paper Link](http://arxiv.org/abs/2306.16718v2)  

---


**ABSTRACT**  
Arbitrary-oriented object detection is a relatively emerging but challenging task. Although remarkable progress has been made, there still remain many unsolved issues due to the large diversity of patterns in orientation, scale, aspect ratio, and visual appearance of objects in aerial images. Most of the existing methods adopt a coarse-grained fixed label assignment strategy and suffer from the inconsistency between the classification score and localization accuracy. First, to align the metric inconsistency between sample selection and regression loss calculation caused by fixed IoU strategy, we introduce affine transformation to evaluate the quality of samples and propose a distance-based label assignment strategy. The proposed metric-aligned selection (MAS) strategy can dynamically select samples according to the shape and rotation characteristic of objects. Second, to further address the inconsistency between classification and localization, we propose a critical feature sampling (CFS) module, which performs localization refinement on the sampling location for classification task to extract critical features accurately. Third, we present a scale-controlled smooth $L_1$ loss (SC-Loss) to adaptively select high quality samples by changing the form of regression loss function based on the statistics of proposals during training. Extensive experiments are conducted on four challenging rotated object detection datasets DOTA, FAIR1M-1.0, HRSC2016, and UCAS-AOD. The results show the state-of-the-art accuracy of the proposed detector.

{{</citation>}}


### (69/104) Answer Mining from a Pool of Images: Towards Retrieval-Based Visual Question Answering (Abhirama Subramanyam Penamakuri et al., 2023)

{{<citation>}}

Abhirama Subramanyam Penamakuri, Manish Gupta, Mithun Das Gupta, Anand Mishra. (2023)  
**Answer Mining from a Pool of Images: Towards Retrieval-Based Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2306.16713v1)  

---


**ABSTRACT**  
We study visual question answering in a setting where the answer has to be mined from a pool of relevant and irrelevant images given as a context. For such a setting, a model must first retrieve relevant images from the pool and answer the question from these retrieved images. We refer to this problem as retrieval-based visual question answering (or RETVQA in short). The RETVQA is distinctively different and more challenging than the traditionally-studied Visual Question Answering (VQA), where a given question has to be answered with a single relevant image in context. Towards solving the RETVQA task, we propose a unified Multi Image BART (MI-BART) that takes a question and retrieved images using our relevance encoder for free-form fluent answer generation. Further, we introduce the largest dataset in this space, namely RETVQA, which has the following salient features: multi-image and retrieval requirement for VQA, metadata-independent questions over a pool of heterogeneous images, expecting a mix of classification-oriented and open-ended generative answers. Our proposed framework achieves an accuracy of 76.5% and a fluency of 79.3% on the proposed dataset, namely RETVQA and also outperforms state-of-the-art methods by 4.9% and 11.8% on the image segment of the publicly available WebQA dataset on the accuracy and fluency metrics, respectively.

{{</citation>}}


### (70/104) BinaryViT: Pushing Binary Vision Transformers Towards Convolutional Models (Phuoc-Hoan Charles Le et al., 2023)

{{<citation>}}

Phuoc-Hoan Charles Le, Xinlin Li. (2023)  
**BinaryViT: Pushing Binary Vision Transformers Towards Convolutional Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2306.16678v1)  

---


**ABSTRACT**  
With the increasing popularity and the increasing size of vision transformers (ViTs), there has been an increasing interest in making them more efficient and less computationally costly for deployment on edge devices with limited computing resources. Binarization can be used to help reduce the size of ViT models and their computational cost significantly, using popcount operations when the weights and the activations are in binary. However, ViTs suffer a larger performance drop when directly applying convolutional neural network (CNN) binarization methods or existing binarization methods to binarize ViTs compared to CNNs on datasets with a large number of classes such as ImageNet-1k. With extensive analysis, we find that binary vanilla ViTs such as DeiT miss out on a lot of key architectural properties that CNNs have that allow binary CNNs to have much higher representational capability than binary vanilla ViT. Therefore, we propose BinaryViT, in which inspired by the CNN architecture, we include operations from the CNN architecture into a pure ViT architecture to enrich the representational capability of a binary ViT without introducing convolutions. These include an average pooling layer instead of a token pooling layer, a block that contains multiple average pooling branches, an affine transformation right before the addition of each main residual connection, and a pyramid structure. Experimental results on the ImageNet-1k dataset show the effectiveness of these operations that allow a binary pure ViT model to be competitive with previous state-of-the-art (SOTA) binary CNN models.

{{</citation>}}


### (71/104) Deep Equilibrium Multimodal Fusion (Jinhong Ni et al., 2023)

{{<citation>}}

Jinhong Ni, Yalong Bai, Wei Zhang, Ting Yao, Tao Mei. (2023)  
**Deep Equilibrium Multimodal Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2306.16645v1)  

---


**ABSTRACT**  
Multimodal fusion integrates the complementary information present in multiple modalities and has gained much attention recently. Most existing fusion approaches either learn a fixed fusion strategy during training and inference, or are only capable of fusing the information to a certain extent. Such solutions may fail to fully capture the dynamics of interactions across modalities especially when there are complex intra- and inter-modality correlations to be considered for informative multimodal fusion. In this paper, we propose a novel deep equilibrium (DEQ) method towards multimodal fusion via seeking a fixed point of the dynamic multimodal fusion process and modeling the feature correlations in an adaptive and recursive manner. This new way encodes the rich information within and across modalities thoroughly from low level to high level for efficacious downstream multimodal learning and is readily pluggable to various multimodal frameworks. Extensive experiments on BRCA, MM-IMDB, CMU-MOSI, SUN RGB-D, and VQA-v2 demonstrate the superiority of our DEQ fusion. More remarkably, DEQ fusion consistently achieves state-of-the-art performance on multiple multimodal benchmarks. The code will be released.

{{</citation>}}


### (72/104) The Segment Anything Model (SAM) for Remote Sensing Applications: From Zero to One Shot (Lucas Prado Osco et al., 2023)

{{<citation>}}

Lucas Prado Osco, Qiusheng Wu, Eduardo Lopes de Lemos, Wesley Nunes Gonçalves, Ana Paula Marques Ramos, Jonathan Li, José Marcato Junior. (2023)  
**The Segment Anything Model (SAM) for Remote Sensing Applications: From Zero to One Shot**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.16623v1)  

---


**ABSTRACT**  
Segmentation is an essential step for remote sensing image processing. This study aims to advance the application of the Segment Anything Model (SAM), an innovative image segmentation model by Meta AI, in the field of remote sensing image analysis. SAM is known for its exceptional generalization capabilities and zero-shot learning, making it a promising approach to processing aerial and orbital images from diverse geographical contexts. Our exploration involved testing SAM across multi-scale datasets using various input prompts, such as bounding boxes, individual points, and text descriptors. To enhance the model's performance, we implemented a novel automated technique that combines a text-prompt-derived general example with one-shot training. This adjustment resulted in an improvement in accuracy, underscoring SAM's potential for deployment in remote sensing imagery and reducing the need for manual annotation. Despite the limitations encountered with lower spatial resolution images, SAM exhibits promising adaptability to remote sensing data analysis. We recommend future research to enhance the model's proficiency through integration with supplementary fine-tuning techniques and other networks. Furthermore, we provide the open-source code of our modifications on online repositories, encouraging further and broader adaptations of SAM to the remote sensing domain.

{{</citation>}}


### (73/104) Seeing in Words: Learning to Classify through Language Bottlenecks (Khalid Saifullah et al., 2023)

{{<citation>}}

Khalid Saifullah, Yuxin Wen, Jonas Geiping, Micah Goldblum, Tom Goldstein. (2023)  
**Seeing in Words: Learning to Classify through Language Bottlenecks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.00028v1)  

---


**ABSTRACT**  
Neural networks for computer vision extract uninterpretable features despite achieving high accuracy on benchmarks. In contrast, humans can explain their predictions using succinct and intuitive descriptions. To incorporate explainability into neural networks, we train a vision model whose feature representations are text. We show that such a model can effectively classify ImageNet images, and we discuss the challenges we encountered when training it.

{{</citation>}}


## eess.SY (2)



### (74/104) Nonlinear Data-Driven Control Part I: Trajectory Representation under quasi-Linear Parameter Varying Embeddings (Marcelo Menezes Morato et al., 2023)

{{<citation>}}

Marcelo Menezes Morato, Julio Elias Normey-Rico, Olivier Sename. (2023)  
**Nonlinear Data-Driven Control Part I: Trajectory Representation under quasi-Linear Parameter Varying Embeddings**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY, math-DS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2306.17137v1)  

---


**ABSTRACT**  
Recent literature has shown how linear time-invariant (LTI) systems can be represented by trajectories features, that is relying on a single input-output (IO) data dictionary to span all possible system trajectories, as long as the input is persistently exciting. The so-called behavioural framework is a promising alternative for controller synthesis without the necessity of system identification. In this paper, we benefit from differential inclusion in order to adapt previous results to the case quasi-Linear Parameter Varying (qLPV) embeddings, which are use to represent nonlinear dynamical systems along suitable IO coordinates. Accordingly, we propose a set of data-driven analysis tools for a wide class of nonlinear systems, which enable nonlinear data-driven simulation and predictions. Furthermore, a parameter-dependent dissipativity analysis verification setup is also presented, which serves to assess stability of the system within a bounded operation region. The major requirement is that there should exist a scheduling function which maps the nonlinear outputs into a finite number of scheduling variables, and this function should be analytically known. The effectiveness of the proposed tools is tested in practice and shown to provide accurate descriptions of the nonlinear dynamics by the means of a linear representation structure. For such, we consider a high-fidelity nonlinear simulator of a rotational pendulum benchmark simulator and an electro-mechanical positioning experimental validation test-bench. We also show that, even if the scheduling function is erroneously selected, the proposed framework is still able to offer a trustworthy representation of the output dynamics.

{{</citation>}}


### (75/104) Laxity-Aware Scalable Reinforcement Learning for HVAC Control (Ruohong Liu et al., 2023)

{{<citation>}}

Ruohong Liu, Yuxin Pan, Yize Chen. (2023)  
**Laxity-Aware Scalable Reinforcement Learning for HVAC Control**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-SY, eess-SY, eess.SY, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2306.16619v1)  

---


**ABSTRACT**  
Demand flexibility plays a vital role in maintaining grid balance, reducing peak demand, and saving customers' energy bills. Given their highly shiftable load and significant contribution to a building's energy consumption, Heating, Ventilation, and Air Conditioning (HVAC) systems can provide valuable demand flexibility to the power systems by adjusting their energy consumption in response to electricity price and power system needs. To exploit this flexibility in both operation time and power, it is imperative to accurately model and aggregate the load flexibility of a large population of HVAC systems as well as designing effective control algorithms. In this paper, we tackle the curse of dimensionality issue in modeling and control by utilizing the concept of laxity to quantify the emergency level of each HVAC operation request. We further propose a two-level approach to address energy optimization for a large population of HVAC systems. The lower level involves an aggregator to aggregate HVAC load laxity information and use least-laxity-first (LLF) rule to allocate real-time power for individual HVAC systems based on the controller's total power. Due to the complex and uncertain nature of HVAC systems, we leverage a reinforcement learning (RL)-based controller to schedule the total power based on the aggregated laxity information and electricity price. We evaluate the temperature control and energy cost saving performance of a large-scale group of HVAC systems in both single-zone and multi-zone scenarios, under varying climate and electricity market conditions. The experiment results indicate that proposed approach outperforms the centralized methods in the majority of test scenarios, and performs comparably to model-based method in some scenarios.

{{</citation>}}


## cs.IR (3)



### (76/104) Ducho: A Unified Framework for the Extraction of Multimodal Features in Recommendation (Daniele Malitesta et al., 2023)

{{<citation>}}

Daniele Malitesta, Giuseppe Gassi, Claudio Pomo, Tommaso Di Noia. (2023)  
**Ducho: A Unified Framework for the Extraction of Multimodal Features in Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-MM, cs.IR  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2306.17125v1)  

---


**ABSTRACT**  
In multimodal-aware recommendation, the extraction of meaningful multimodal features is at the basis of high-quality recommendations. Generally, each recommendation framework implements its multimodal extraction procedures with specific strategies and tools. This is limiting for two reasons: (i) different extraction strategies do not ease the interdependence among multimodal recommendation frameworks; thus, they cannot be efficiently and fairly compared; (ii) given the large plethora of pre-trained deep learning models made available by different open source tools, model designers do not have access to shared interfaces to extract features. Motivated by the outlined aspects, we propose Ducho, a unified framework for the extraction of multimodal features in recommendation. By integrating three widely-adopted deep learning libraries as backends, namely, TensorFlow, PyTorch, and Transformers, we provide a shared interface to extract and process features where each backend's specific methods are abstracted to the end user. Noteworthy, the extraction pipeline is easily configurable with a YAML-based file where the user can specify, for each modality, the list of models (and their specific backends/parameters) to perform the extraction. Finally, to make Ducho accessible to the community, we build a public Docker image equipped with a ready-to-use CUDA environment and propose three demos to test its functionalities for different scenarios and tasks. The GitHub repository and the documentation is accessible at this link: https://github.com/sisinflab/Ducho.

{{</citation>}}


### (77/104) Harnessing the Power of Hugging Face Transformers for Predicting Mental Health Disorders in Social Networks (Alireza Pourkeyvan et al., 2023)

{{<citation>}}

Alireza Pourkeyvan, Ramin Safa, Ali Sorourkhah. (2023)  
**Harnessing the Power of Hugging Face Transformers for Predicting Mental Health Disorders in Social Networks**  

---
Primary Category: cs.IR  
Categories: I-2-7; J-3, cs-AI, cs-HC, cs-IR, cs.IR  
Keywords: BERT, Social Network, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2306.16891v2)  

---


**ABSTRACT**  
Early diagnosis of mental disorders and intervention can facilitate the prevention of severe injuries and the improvement of treatment results. Using social media and pre-trained language models, this study explores how user-generated data can be used to predict mental disorder symptoms. Our study compares four different BERT models of Hugging Face with standard machine learning techniques used in automatic depression diagnosis in recent literature. The results show that new models outperform the previous approach with an accuracy rate of up to 97%. Analyzing the results while complementing past findings, we find that even tiny amounts of data (like users' bio descriptions) have the potential to predict mental disorders. We conclude that social media data is an excellent source of mental health screening, and pre-trained models can effectively automate this critical task.

{{</citation>}}


### (78/104) Beyond CO2 Emissions: The Overlooked Impact of Water Consumption of Information Retrieval Models (Guido Zuccon et al., 2023)

{{<citation>}}

Guido Zuccon, Harrisen Scells, Shengyao Zhuang. (2023)  
**Beyond CO2 Emissions: The Overlooked Impact of Water Consumption of Information Retrieval Models**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2306.16668v1)  

---


**ABSTRACT**  
As in other fields of artificial intelligence, the information retrieval community has grown interested in investigating the power consumption associated with neural models, particularly models of search. This interest has become particularly relevant as the energy consumption of information retrieval models has risen with new neural models based on large language models, leading to an associated increase of CO2 emissions, albeit relatively low compared to fields such as natural language processing.

{{</citation>}}


## cs.SE (3)



### (79/104) RAPGen: An Approach for Fixing Code Inefficiencies in Zero-Shot (Spandan Garg et al., 2023)

{{<citation>}}

Spandan Garg, Roshanak Zilouchian Moghaddam, Neel Sundaresan. (2023)  
**RAPGen: An Approach for Fixing Code Inefficiencies in Zero-Shot**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2306.17077v1)  

---


**ABSTRACT**  
Performance bugs are non-functional bugs that can even manifest in well-tested commercial products. Fixing these performance bugs is an important yet challenging problem. In this work, we address this challenge and present a new approach called Retrieval-Augmented Prompt Generation (RAPGen). Given a code snippet with a performance issue, RAPGen first retrieves a prompt instruction from a pre-constructed knowledge-base of previous performance bug fixes and then generates a prompt using the retrieved instruction. It then uses this prompt on a Large Language Model (such as Codex) in zero-shot to generate a fix. We compare our approach with the various prompt variations and state of the art methods in the task of performance bug fixing. Our evaluation shows that RAPGen can generate performance improvement suggestions equivalent or better than a developer in ~60% of the cases, getting ~39% of them verbatim, in an expert-verified dataset of past performance changes made by C# developers.

{{</citation>}}


### (80/104) A Query Language for Software Architecture Information (Extended version) (Joshua Ammermann et al., 2023)

{{<citation>}}

Joshua Ammermann, Sven Jordan, Lukas Linsbauer, Ina Schaefer. (2023)  
**A Query Language for Software Architecture Information (Extended version)**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.16829v2)  

---


**ABSTRACT**  
Software maintenance is an important part of a software system's life cycle. Maintenance tasks of existing software systems suffer from architecture information that is diverging over time (architectural drift). The Digital Architecture Twin (DArT) can support software maintenance by providing up-to-date architecture information. For this, the DArT gathers such information and co-evolves with a software system, enabling continuous reverse engineering. But the crucial link for stakeholders to retrieve this information is missing. To fill this gap, we contribute the Architecture Information Query Language (AIQL), which enables stakeholders to access up-to-date and tailored architecture information. We derived four application scenarios in the context of continuous reverse engineering. We showed that the AIQL provides the required functionality to formulate queries for the application scenarios and that the language scales for use with real-world software systems. In a user study, stakeholders agreed that the language is easy to understand and assessed its value to the specific stakeholder for the application scenarios.

{{</citation>}}


### (81/104) Divide and Conquer the EmpiRE: A Community-Maintainable Knowledge Graph of Empirical Research in Requirements Engineering (Oliver Karras et al., 2023)

{{<citation>}}

Oliver Karras, Felix Wernlein, Jil Klünder, Sören Auer. (2023)  
**Divide and Conquer the EmpiRE: A Community-Maintainable Knowledge Graph of Empirical Research in Requirements Engineering**  

---
Primary Category: cs.SE  
Categories: cs-DL, cs-SE, cs.SE  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2306.16791v1)  

---


**ABSTRACT**  
[Background.] Empirical research in requirements engineering (RE) is a constantly evolving topic, with a growing number of publications. Several papers address this topic using literature reviews to provide a snapshot of its "current" state and evolution. However, these papers have never built on or updated earlier ones, resulting in overlap and redundancy. The underlying problem is the unavailability of data from earlier works. Researchers need technical infrastructures to conduct sustainable literature reviews. [Aims.] We examine the use of the Open Research Knowledge Graph (ORKG) as such an infrastructure to build and publish an initial Knowledge Graph of Empirical research in RE (KG-EmpiRE) whose data is openly available. Our long-term goal is to continuously maintain KG-EmpiRE with the research community to synthesize a comprehensive, up-to-date, and long-term available overview of the state and evolution of empirical research in RE. [Method.] We conduct a literature review using the ORKG to build and publish KG-EmpiRE which we evaluate against competency questions derived from a published vision of empirical research in software (requirements) engineering for 2020 - 2025. [Results.] From 570 papers of the IEEE International Requirements Engineering Conference (2000 - 2022), we extract and analyze data on the reported empirical research and answer 16 out of 77 competency questions. These answers show a positive development towards the vision, but also the need for future improvements. [Conclusions.] The ORKG is a ready-to-use and advanced infrastructure to organize data from literature reviews as knowledge graphs. The resulting knowledge graphs make the data openly available and maintainable by research communities, enabling sustainable literature reviews.

{{</citation>}}


## cs.CR (3)



### (82/104) Honesty is the Best Policy: On the Accuracy of Apple Privacy Labels Compared to Apps' Privacy Policies (Mir Masood Ali et al., 2023)

{{<citation>}}

Mir Masood Ali, David G. Balash, Chris Kanich, Adam J. Aviv. (2023)  
**Honesty is the Best Policy: On the Accuracy of Apple Privacy Labels Compared to Apps' Privacy Policies**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2306.17063v1)  

---


**ABSTRACT**  
Apple introduced \textit{privacy labels} in Dec. 2020 as a way for developers to report the privacy behaviors of their apps. While Apple does not validate labels, they do also require developers to provide a privacy policy, which offers an important comparison point. In this paper, we applied the NLP framework of Polisis to extract features of the privacy policy for 515,920 apps on the iOS App Store comparing the output to the privacy labels. We identify discrepancies between the policies and the labels, particularly as it relates to data collected that is linked to users. We find that 287$\pm196$K apps' privacy policies may indicate data collection that is linked to users than what is reported in the privacy labels. More alarming, a large number of (97$\pm30$\%) of the apps that have {\em Data Not Collected} privacy label have a privacy policy that indicates otherwise. We provide insights into potential sources for discrepancies, including the use of templates and confusion around Apple's definitions and requirements. These results suggest that there is still significant work to be done to help developers more accurately labeling their apps. Incorporating a Polisis-like system as a first-order check can help improve the current state and better inform developers when there are possible misapplication of privacy labels.

{{</citation>}}


### (83/104) VibHead: An Authentication Scheme for Smart Headsets through Vibration (Feng Li et al., 2023)

{{<citation>}}

Feng Li, Jiayi Zhao, Huan Yang, Dongxiao Yu, Yuanfeng Zhou, Yiran Shen. (2023)  
**VibHead: An Authentication Scheme for Smart Headsets through Vibration**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2306.17002v1)  

---


**ABSTRACT**  
Recent years have witnessed the fast penetration of Virtual Reality (VR) and Augmented Reality (AR) systems into our daily life, the security and privacy issues of the VR/AR applications have been attracting considerable attention. Most VR/AR systems adopt head-mounted devices (i.e., smart headsets) to interact with users and the devices usually store the users' private data. Hence, authentication schemes are desired for the head-mounted devices. Traditional knowledge-based authentication schemes for general personal devices have been proved vulnerable to shoulder-surfing attacks, especially considering the headsets may block the sight of the users. Although the robustness of the knowledge-based authentication can be improved by designing complicated secret codes in virtual space, this approach induces a compromise of usability. Another choice is to leverage the users' biometrics; however, it either relies on highly advanced equipments which may not always be available in commercial headsets or introduce heavy cognitive load to users.   In this paper, we propose a vibration-based authentication scheme, VibHead, for smart headsets. Since the propagation of vibration signals through human heads presents unique patterns for different individuals, VibHead employs a CNN-based model to classify registered legitimate users based the features extracted from the vibration signals. We also design a two-step authentication scheme where the above user classifiers are utilized to distinguish the legitimate user from illegitimate ones. We implement VibHead on a Microsoft HoloLens equipped with a linear motor and an IMU sensor which are commonly used in off-the-shelf personal smart devices. According to the results of our extensive experiments, with short vibration signals ($\leq 1s$), VibHead has an outstanding authentication accuracy; both FAR and FRR are around 5%.

{{</citation>}}


### (84/104) A Survey on Enterprise Network Security: Asset Behavioral Monitoring and Distributed Attack Detection (Minzhao Lyu et al., 2023)

{{<citation>}}

Minzhao Lyu, Hassan Habibi Gharakheili, Vijay Sivaraman. (2023)  
**A Survey on Enterprise Network Security: Asset Behavioral Monitoring and Distributed Attack Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR  
Keywords: Network Security, Security  
[Paper Link](http://arxiv.org/abs/2306.16675v1)  

---


**ABSTRACT**  
Enterprise networks that host valuable assets and services are popular and frequent targets of distributed network attacks. In order to cope with the ever-increasing threats, industrial and research communities develop systems and methods to monitor the behaviors of their assets and protect them from critical attacks. In this paper, we systematically survey related research articles and industrial systems to highlight the current status of this arms race in enterprise network security. First, we discuss the taxonomy of distributed network attacks on enterprise assets, including distributed denial-of-service (DDoS) and reconnaissance attacks. Second, we review existing methods in monitoring and classifying network behavior of enterprise hosts to verify their benign activities and isolate potential anomalies. Third, state-of-the-art detection methods for distributed network attacks sourced from external attackers are elaborated, highlighting their merits and bottlenecks. Fourth, as programmable networks and machine learning (ML) techniques are increasingly becoming adopted by the community, their current applications in network security are discussed. Finally, we highlight several research gaps on enterprise network security to inspire future research.

{{</citation>}}


## cs.NI (1)



### (85/104) Two-tiered Online Optimization of Region-wide Datacenter Resource Allocation via Deep Reinforcement Learning (Chang-Lin Chen et al., 2023)

{{<citation>}}

Chang-Lin Chen, Hanhan Zhou, Jiayu Chen, Mohammad Pedramfar, Vaneet Aggarwal, Tian Lan, Zheqing Zhu, Chi Zhou, Tim Gasser, Pol Mauri Ruiz, Vijay Menon, Neeraj Kumar, Hongbo Dong. (2023)  
**Two-tiered Online Optimization of Region-wide Datacenter Resource Allocation via Deep Reinforcement Learning**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2306.17054v1)  

---


**ABSTRACT**  
This paper addresses the important need for advanced techniques in continuously allocating workloads on shared infrastructures in data centers, a problem arising due to the growing popularity and scale of cloud computing. It particularly emphasizes the scarcity of research ensuring guaranteed capacity in capacity reservations during large-scale failures. To tackle these issues, the paper presents scalable solutions for resource management. It builds on the prior establishment of capacity reservation in cluster management systems and the two-level resource allocation problem addressed by the Resource Allowance System (RAS). Recognizing the limitations of Mixed Integer Linear Programming (MILP) for server assignment in a dynamic environment, this paper proposes the use of Deep Reinforcement Learning (DRL), which has been successful in achieving long-term optimal results for time-varying systems. A novel two-level design that utilizes a DRL-based algorithm is introduced to solve optimal server-to-reservation assignment, taking into account of fault tolerance, server movement minimization, and network affinity requirements due to the impracticality of directly applying DRL algorithms to large-scale instances with millions of decision variables. The paper explores the interconnection of these levels and the benefits of such an approach for achieving long-term optimal results in the context of large-scale cloud systems. We further show in the experiment section that our two-level DRL approach outperforms the MIP solver and heuristic approaches and exhibits significantly reduced computation time compared to the MIP solver. Specifically, our two-level DRL approach performs 15% better than the MIP solver on minimizing the overall cost. Also, it uses only 26 seconds to execute 30 rounds of decision making, while the MIP solver needs nearly an hour.

{{</citation>}}


## cs.RO (4)



### (86/104) Spatial Reasoning via Deep Vision Models for Robotic Sequential Manipulation (Hongyou Zhou et al., 2023)

{{<citation>}}

Hongyou Zhou, Ingmar Schubert, Marc Toussaint, Ozgur S. Oguz. (2023)  
**Spatial Reasoning via Deep Vision Models for Robotic Sequential Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2306.17053v2)  

---


**ABSTRACT**  
In this paper, we propose using deep neural architectures (i.e., vision transformers and ResNet) as heuristics for sequential decision-making in robotic manipulation problems. This formulation enables predicting the subset of objects that are relevant for completing a task. Such problems are often addressed by task and motion planning (TAMP) formulations combining symbolic reasoning and continuous motion planning. In essence, the action-object relationships are resolved for discrete, symbolic decisions that are used to solve manipulation motions (e.g., via nonlinear trajectory optimization). However, solving long-horizon tasks requires consideration of all possible action-object combinations which limits the scalability of TAMP approaches. To overcome this combinatorial complexity, we introduce a visual perception module integrated with a TAMP-solver. Given a task and an initial image of the scene, the learned model outputs the relevancy of objects to accomplish the task. By incorporating the predictions of the model into a TAMP formulation as a heuristic, the size of the search space is significantly reduced. Results show that our framework finds feasible solutions more efficiently when compared to a state-of-the-art TAMP solver.

{{</citation>}}


### (87/104) End-to-end Reinforcement Learning for Online Coverage Path Planning in Unknown Environments (Arvi Jonnarth et al., 2023)

{{<citation>}}

Arvi Jonnarth, Jie Zhao, Michael Felsberg. (2023)  
**End-to-end Reinforcement Learning for Online Coverage Path Planning in Unknown Environments**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2306.16978v1)  

---


**ABSTRACT**  
Coverage path planning is the problem of finding the shortest path that covers the entire free space of a given confined area, with applications ranging from robotic lawn mowing and vacuum cleaning, to demining and search-and-rescue tasks. While offline methods can find provably complete, and in some cases optimal, paths for known environments, their value is limited in online scenarios where the environment is not known beforehand, especially in the presence of non-static obstacles. We propose an end-to-end reinforcement learning-based approach in continuous state and action space, for the online coverage path planning problem that can handle unknown environments. We construct the observation space from both global maps and local sensory inputs, allowing the agent to plan a long-term path, and simultaneously act on short-term obstacle detections. To account for large-scale environments, we propose to use a multi-scale map input representation. Furthermore, we propose a novel total variation reward term for eliminating thin strips of uncovered space in the learned path. To validate the effectiveness of our approach, we perform extensive experiments in simulation with a distance sensor, surpassing the performance of a recent reinforcement learning-based approach.

{{</citation>}}


### (88/104) ArrayBot: Reinforcement Learning for Generalizable Distributed Manipulation through Touch (Zhengrong Xue et al., 2023)

{{<citation>}}

Zhengrong Xue, Han Zhang, Jingwen Cheng, Zhengmao He, Yuanchen Ju, Changyi Lin, Gu Zhang, Huazhe Xu. (2023)  
**ArrayBot: Reinforcement Learning for Generalizable Distributed Manipulation through Touch**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2306.16857v1)  

---


**ABSTRACT**  
We present ArrayBot, a distributed manipulation system consisting of a $16 \times 16$ array of vertically sliding pillars integrated with tactile sensors, which can simultaneously support, perceive, and manipulate the tabletop objects. Towards generalizable distributed manipulation, we leverage reinforcement learning (RL) algorithms for the automatic discovery of control policies. In the face of the massively redundant actions, we propose to reshape the action space by considering the spatially local action patch and the low-frequency actions in the frequency domain. With this reshaped action space, we train RL agents that can relocate diverse objects through tactile observations only. Surprisingly, we find that the discovered policy can not only generalize to unseen object shapes in the simulator but also transfer to the physical robot without any domain randomization. Leveraging the deployed policy, we present abundant real-world manipulation tasks, illustrating the vast potential of RL on ArrayBot for distributed manipulation.

{{</citation>}}


### (89/104) Dynamic-Resolution Model Learning for Object Pile Manipulation (Yixuan Wang et al., 2023)

{{<citation>}}

Yixuan Wang, Yunzhu Li, Katherine Driggs-Campbell, Li Fei-Fei, Jiajun Wu. (2023)  
**Dynamic-Resolution Model Learning for Object Pile Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2306.16700v2)  

---


**ABSTRACT**  
Dynamics models learned from visual observations have shown to be effective in various robotic manipulation tasks. One of the key questions for learning such dynamics models is what scene representation to use. Prior works typically assume representation at a fixed dimension or resolution, which may be inefficient for simple tasks and ineffective for more complicated tasks. In this work, we investigate how to learn dynamic and adaptive representations at different levels of abstraction to achieve the optimal trade-off between efficiency and effectiveness. Specifically, we construct dynamic-resolution particle representations of the environment and learn a unified dynamics model using graph neural networks (GNNs) that allows continuous selection of the abstraction level. During test time, the agent can adaptively determine the optimal resolution at each model-predictive control (MPC) step. We evaluate our method in object pile manipulation, a task we commonly encounter in cooking, agriculture, manufacturing, and pharmaceutical applications. Through comprehensive evaluations both in the simulation and the real world, we show that our method achieves significantly better performance than state-of-the-art fixed-resolution baselines at the gathering, sorting, and redistribution of granular object piles made with various instances like coffee beans, almonds, corn, etc.

{{</citation>}}


## eess.IV (3)



### (90/104) MLA-BIN: Model-level Attention and Batch-instance Style Normalization for Domain Generalization of Federated Learning on Medical Image Segmentation (Fubao Zhu et al., 2023)

{{<citation>}}

Fubao Zhu, Yanhui Tian, Chuang Han, Yanting Li, Jiaofen Nan, Ni Yao, Weihua Zhou. (2023)  
**MLA-BIN: Model-level Attention and Batch-instance Style Normalization for Domain Generalization of Federated Learning on Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2306.17008v1)  

---


**ABSTRACT**  
The privacy protection mechanism of federated learning (FL) offers an effective solution for cross-center medical collaboration and data sharing. In multi-site medical image segmentation, each medical site serves as a client of FL, and its data naturally forms a domain. FL supplies the possibility to improve the performance of seen domains model. However, there is a problem of domain generalization (DG) in the actual de-ployment, that is, the performance of the model trained by FL in unseen domains will decrease. Hence, MLA-BIN is proposed to solve the DG of FL in this study. Specifically, the model-level attention module (MLA) and batch-instance style normalization (BIN) block were designed. The MLA represents the unseen domain as a linear combination of seen domain models. The atten-tion mechanism is introduced for the weighting coefficient to obtain the optimal coefficient ac-cording to the similarity of inter-domain data features. MLA enables the global model to gen-eralize to unseen domain. In the BIN block, batch normalization (BN) and instance normalization (IN) are combined to perform the shallow layers of the segmentation network for style normali-zation, solving the influence of inter-domain image style differences on DG. The extensive experimental results of two medical image seg-mentation tasks demonstrate that the proposed MLA-BIN outperforms state-of-the-art methods.

{{</citation>}}


### (91/104) PCDAL: A Perturbation Consistency-Driven Active Learning Approach for Medical Image Segmentation and Classification (Tao Wang et al., 2023)

{{<citation>}}

Tao Wang, Xinlin Zhang, Yuanbo Zhou, Junlin Lan, Tao Tan, Min Du, Qinquan Gao, Tong Tong. (2023)  
**PCDAL: A Perturbation Consistency-Driven Active Learning Approach for Medical Image Segmentation and Classification**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2306.16918v1)  

---


**ABSTRACT**  
In recent years, deep learning has become a breakthrough technique in assisting medical image diagnosis. Supervised learning using convolutional neural networks (CNN) provides state-of-the-art performance and has served as a benchmark for various medical image segmentation and classification. However, supervised learning deeply relies on large-scale annotated data, which is expensive, time-consuming, and even impractical to acquire in medical imaging applications. Active Learning (AL) methods have been widely applied in natural image classification tasks to reduce annotation costs by selecting more valuable examples from the unlabeled data pool. However, their application in medical image segmentation tasks is limited, and there is currently no effective and universal AL-based method specifically designed for 3D medical image segmentation. To address this limitation, we propose an AL-based method that can be simultaneously applied to 2D medical image classification, segmentation, and 3D medical image segmentation tasks. We extensively validated our proposed active learning method on three publicly available and challenging medical image datasets, Kvasir Dataset, COVID-19 Infection Segmentation Dataset, and BraTS2019 Dataset. The experimental results demonstrate that our PCDAL can achieve significantly improved performance with fewer annotations in 2D classification and segmentation and 3D segmentation tasks. The codes of this study are available at https://github.com/ortonwang/PCDAL.

{{</citation>}}


### (92/104) Self-Supervised MRI Reconstruction with Unrolled Diffusion Models (Yilmaz Korkmaz et al., 2023)

{{<citation>}}

Yilmaz Korkmaz, Tolga Cukur, Vishal Patel. (2023)  
**Self-Supervised MRI Reconstruction with Unrolled Diffusion Models**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2306.16654v1)  

---


**ABSTRACT**  
Magnetic Resonance Imaging (MRI) produces excellent soft tissue contrast, albeit it is an inherently slow imaging modality. Promising deep learning methods have recently been proposed to reconstruct accelerated MRI scans. However, existing methods still suffer from various limitations regarding image fidelity, contextual sensitivity, and reliance on fully-sampled acquisitions for model training. To comprehensively address these limitations, we propose a novel self-supervised deep reconstruction model, named Self-Supervised Diffusion Reconstruction (SSDiffRecon). SSDiffRecon expresses a conditional diffusion process as an unrolled architecture that interleaves cross-attention transformers for reverse diffusion steps with data-consistency blocks for physics-driven processing. Unlike recent diffusion methods for MRI reconstruction, a self-supervision strategy is adopted to train SSDiffRecon using only undersampled k-space data. Comprehensive experiments on public brain MR datasets demonstrates the superiority of SSDiffRecon against state-of-the-art supervised, and self-supervised baselines in terms of reconstruction speed and quality. Implementation will be available at https://github.com/yilmazkorkmaz1/SSDiffRecon.

{{</citation>}}


## eess.AS (2)



### (93/104) High-Quality Automatic Voice Over with Accurate Alignment: Supervision through Self-Supervised Discrete Speech Units (Junchen Lu et al., 2023)

{{<citation>}}

Junchen Lu, Berrak Sisman, Mingyang Zhang, Haizhou Li. (2023)  
**High-Quality Automatic Voice Over with Accurate Alignment: Supervision through Self-Supervised Discrete Speech Units**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2306.17005v1)  

---


**ABSTRACT**  
The goal of Automatic Voice Over (AVO) is to generate speech in sync with a silent video given its text script. Recent AVO frameworks built upon text-to-speech synthesis (TTS) have shown impressive results. However, the current AVO learning objective of acoustic feature reconstruction brings in indirect supervision for inter-modal alignment learning, thus limiting the synchronization performance and synthetic speech quality. To this end, we propose a novel AVO method leveraging the learning objective of self-supervised discrete speech unit prediction, which not only provides more direct supervision for the alignment learning, but also alleviates the mismatch between the text-video context and acoustic features. Experimental results show that our proposed method achieves remarkable lip-speech synchronization and high speech quality by outperforming baselines in both objective and subjective evaluations. Code and speech samples are publicly available.

{{</citation>}}


### (94/104) Learning Multilingual Expressive Speech Representation for Prosody Prediction without Parallel Data (Jarod Duret et al., 2023)

{{<citation>}}

Jarod Duret, Titouan Parcollet, Yannick Estève. (2023)  
**Learning Multilingual Expressive Speech Representation for Prosody Prediction without Parallel Data**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2306.17199v1)  

---


**ABSTRACT**  
We propose a method for speech-to-speech emotionpreserving translation that operates at the level of discrete speech units. Our approach relies on the use of multilingual emotion embedding that can capture affective information in a language-independent manner. We show that this embedding can be used to predict the pitch and duration of speech units in a target language, allowing us to resynthesize the source speech signal with the same emotional content. We evaluate our approach to English and French speech signals and show that it outperforms a baseline method that does not use emotional information, including when the emotion embedding is extracted from a different language. Even if this preliminary study does not address directly the machine translation issue, our results demonstrate the effectiveness of our approach for cross-lingual emotion preservation in the context of speech resynthesis.

{{</citation>}}


## q-bio.TO (1)



### (95/104) The State of Applying Artificial Intelligence to Tissue Imaging for Cancer Research and Early Detection (Michael Robben et al., 2023)

{{<citation>}}

Michael Robben, Amir Hajighasemi, Mohammad Sadegh Nasr, Jai Prakesh Veerla, Anne M. Alsup, Biraaj Rout, Helen H. Shang, Kelli Fowlds, Parisa Boodaghi Malidarreh, Paul Koomey, MD Jillur Rahman Saurav, Jacob M. Luber. (2023)  
**The State of Applying Artificial Intelligence to Tissue Imaging for Cancer Research and Early Detection**  

---
Primary Category: q-bio.TO  
Categories: cs-CV, eess-IV, q-bio-TO, q-bio.TO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2306.16989v1)  

---


**ABSTRACT**  
Artificial intelligence represents a new frontier in human medicine that could save more lives and reduce the costs, thereby increasing accessibility. As a consequence, the rate of advancement of AI in cancer medical imaging and more particularly tissue pathology has exploded, opening it to ethical and technical questions that could impede its adoption into existing systems. In order to chart the path of AI in its application to cancer tissue imaging, we review current work and identify how it can improve cancer pathology diagnostics and research. In this review, we identify 5 core tasks that models are developed for, including regression, classification, segmentation, generation, and compression tasks. We address the benefits and challenges that such methods face, and how they can be adapted for use in cancer prevention and treatment. The studies looked at in this paper represent the beginning of this field and future experiments will build on the foundations that we highlight.

{{</citation>}}


## cs.SD (2)



### (96/104) Speech-based Age and Gender Prediction with Transformers (Felix Burkhardt et al., 2023)

{{<citation>}}

Felix Burkhardt, Johannes Wagner, Hagen Wierstorf, Florian Eyben, Björn Schuller. (2023)  
**Speech-based Age and Gender Prediction with Transformers**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2306.16962v1)  

---


**ABSTRACT**  
We report on the curation of several publicly available datasets for age and gender prediction. Furthermore, we present experiments to predict age and gender with models based on a pre-trained wav2vec 2.0. Depending on the dataset, we achieve an MAE between 7.1 years and 10.8 years for age, and at least 91.1% ACC for gender (female, male, child). Compared to a modelling approach built on handcrafted features, our proposed system shows an improvement of 9% UAR for age and 4% UAR for gender. To make our findings reproducible, we release the best performing model to the community as well as the sample lists of the data splits.

{{</citation>}}


### (97/104) Transfer Learning with Semi-Supervised Dataset Annotation for Birdcall Classification (Anthony Miyaguchi et al., 2023)

{{<citation>}}

Anthony Miyaguchi, Nathan Zhong, Murilo Gustineli, Chris Hayduk. (2023)  
**Transfer Learning with Semi-Supervised Dataset Annotation for Birdcall Classification**  

---
Primary Category: cs.SD  
Categories: cs-IR, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2306.16760v1)  

---


**ABSTRACT**  
We present working notes on transfer learning with semi-supervised dataset annotation for the BirdCLEF 2023 competition, focused on identifying African bird species in recorded soundscapes. Our approach utilizes existing off-the-shelf models, BirdNET and MixIT, to address representation and labeling challenges in the competition. We explore the embedding space learned by BirdNET and propose a process to derive an annotated dataset for supervised learning. Our experiments involve various models and feature engineering approaches to maximize performance on the competition leaderboard. The results demonstrate the effectiveness of our approach in classifying bird species and highlight the potential of transfer learning and semi-supervised dataset annotation in similar tasks.

{{</citation>}}


## cs.CE (1)



### (98/104) Estimating See and Be Seen Performance with an Airborne Visual Acquisition Model (Ngaire Underhill et al., 2023)

{{<citation>}}

Ngaire Underhill, Evan Maki, Bilal Gill, Andrew Weinert. (2023)  
**Estimating See and Be Seen Performance with an Airborne Visual Acquisition Model**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs-CV, cs-RO, cs.CE, eess-IV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2307.05502v1)  

---


**ABSTRACT**  
Separation provision and collision avoidance to avoid other air traffic are fundamental components of the layered conflict management system to ensure safe and efficient operations. Pilots have visual-based separation responsibilities to see and be seen to maintain separation between aircraft. To safely integrate into the airspace, drones should be required to have a minimum level of performance based on the safety achieved as baselined by crewed aircraft seen and be seen interactions. Drone interactions with crewed aircraft should not be more hazardous than interactions between traditional aviation aircraft. Accordingly, there is need for a methodology to design and evaluate detect and avoid systems, to be equipped by drones to mitigate the risk of a midair collision, where the methodology explicitly addresses, both semantically and mathematically, the appropriate operating rules associated with see and be seen. In response, we simulated how onboard pilots safely operate through see and be seen interactions using an updated visual acquisition model that was originally developed by J.W. Andrews decades ago. Monte Carlo simulations were representative two aircraft flying under visual flight rules and results were analyzed with respect to drone detect and avoid performance standards.

{{</citation>}}


## cs.SI (1)



### (99/104) Opinion Optimization in Directed Social Networks (Haoxin Sun et al., 2023)

{{<citation>}}

Haoxin Sun, Zhongzhi Zhang. (2023)  
**Opinion Optimization in Directed Social Networks**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2306.16847v1)  

---


**ABSTRACT**  
Shifting social opinions has far-reaching implications in various aspects, such as public health campaigns, product marketing, and political candidates. In this paper, we study a problem of opinion optimization based on the popular Friedkin-Johnsen (FJ) model for opinion dynamics in an unweighted directed social network with $n$ nodes and $m$ edges. In the FJ model, the internal opinion of every node lies in the closed interval $[0, 1]$, with 0 and 1 being polar opposites of opinions about a certain issue. Concretely, we focus on the problem of selecting a small number of $ k\ll n $ nodes and changing their internal opinions to 0, in order to minimize the average opinion at equilibrium. We then design an algorithm that returns the optimal solution to the problem in $O(n^3)$ time. To speed up the computation, we further develop a fast algorithm by sampling spanning forests, the time complexity of which is $ O(ln) $, with $l$ being the number of samplings. Finally, we execute extensive experiments on various real directed networks, which show that the effectiveness of our two algorithms is similar to each other, both of which outperform several baseline strategies of node selection. Moreover, our fast algorithm is more efficient than the first one, which is scalable to massive graphs with more than twenty million nodes.

{{</citation>}}


## quant-ph (3)



### (100/104) NNQS-Transformer: an Efficient and Scalable Neural Network Quantum States Approach for Ab initio Quantum Chemistry (Yangjun Wu et al., 2023)

{{<citation>}}

Yangjun Wu, Chu Guo, Yi Fan, Pengyu Zhou, Honghui Shang. (2023)  
**NNQS-Transformer: an Efficient and Scalable Neural Network Quantum States Approach for Ab initio Quantum Chemistry**  

---
Primary Category: quant-ph  
Categories: cs-AI, quant-ph, quant-ph  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2306.16705v2)  

---


**ABSTRACT**  
Neural network quantum state (NNQS) has emerged as a promising candidate for quantum many-body problems, but its practical applications are often hindered by the high cost of sampling and local energy calculation. We develop a high-performance NNQS method for \textit{ab initio} electronic structure calculations. The major innovations include: (1) A transformer based architecture as the quantum wave function ansatz; (2) A data-centric parallelization scheme for the variational Monte Carlo (VMC) algorithm which preserves data locality and well adapts for different computing architectures; (3) A parallel batch sampling strategy which reduces the sampling cost and achieves good load balance; (4) A parallel local energy evaluation scheme which is both memory and computationally efficient; (5) Study of real chemical systems demonstrates both the superior accuracy of our method compared to state-of-the-art and the strong and weak scalability for large molecular systems with up to $120$ spin orbitals.

{{</citation>}}


### (101/104) TrojanNet: Detecting Trojans in Quantum Circuits using Machine Learning (Subrata Das et al., 2023)

{{<citation>}}

Subrata Das, Swaroop Ghosh. (2023)  
**TrojanNet: Detecting Trojans in Quantum Circuits using Machine Learning**  

---
Primary Category: quant-ph  
Categories: cs-CR, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2306.16701v1)  

---


**ABSTRACT**  
Quantum computing holds tremendous potential for various applications, but its security remains a crucial concern. Quantum circuits need high-quality compilers to optimize the depth and gate count to boost the success probability on current noisy quantum computers. There is a rise of efficient but unreliable/untrusted compilers; however, they present a risk of tampering such as Trojan insertion. We propose TrojanNet, a novel approach to enhance the security of quantum circuits by detecting and classifying Trojan-inserted circuits. In particular, we focus on the Quantum Approximate Optimization Algorithm (QAOA) circuit that is popular in solving a wide range of optimization problems. We investigate the impact of Trojan insertion on QAOA circuits and develop a Convolutional Neural Network (CNN) model, referred to as TrojanNet, to identify their presence accurately. Using the Qiskit framework, we generate 12 diverse datasets by introducing variations in Trojan gate types, the number of gates, insertion locations, and compiler backends. These datasets consist of both original Trojan-free QAOA circuits and their corresponding Trojan-inserted counterparts. The generated datasets are then utilized for training and evaluating the TrojanNet model. Experimental results showcase an average accuracy of 98.80% and an average F1-score of 98.53% in effectively detecting and classifying Trojan-inserted QAOA circuits. Finally, we conduct a performance comparison between TrojanNet and existing machine learning-based Trojan detection methods specifically designed for conventional netlists.

{{</citation>}}


### (102/104) MNISQ: A Large-Scale Quantum Circuit Dataset for Machine Learning on/for Quantum Computers in the NISQ era (Leonardo Placidi et al., 2023)

{{<citation>}}

Leonardo Placidi, Ryuichiro Hataya, Toshio Mori, Koki Aoyama, Hayata Morisaki, Kosuke Mitarai, Keisuke Fujii. (2023)  
**MNISQ: A Large-Scale Quantum Circuit Dataset for Machine Learning on/for Quantum Computers in the NISQ era**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: LSTM, QA, Transformer  
[Paper Link](http://arxiv.org/abs/2306.16627v1)  

---


**ABSTRACT**  
We introduce the first large-scale dataset, MNISQ, for both the Quantum and the Classical Machine Learning community during the Noisy Intermediate-Scale Quantum era. MNISQ consists of 4,950,000 data points organized in 9 subdatasets. Building our dataset from the quantum encoding of classical information (e.g., MNIST dataset), we deliver a dataset in a dual form: in quantum form, as circuits, and in classical form, as quantum circuit descriptions (quantum programming language, QASM). In fact, also the Machine Learning research related to quantum computers undertakes a dual challenge: enhancing machine learning exploiting the power of quantum computers, while also leveraging state-of-the-art classical machine learning methodologies to help the advancement of quantum computing. Therefore, we perform circuit classification on our dataset, tackling the task with both quantum and classical models. In the quantum endeavor, we test our circuit dataset with Quantum Kernel methods, and we show excellent results up to $97\%$ accuracy. In the classical world, the underlying quantum mechanical structures within the quantum circuit data are not trivial. Nevertheless, we test our dataset on three classical models: Structured State Space sequence model (S4), Transformer and LSTM. In particular, the S4 model applied on the tokenized QASM sequences reaches an impressive $77\%$ accuracy. These findings illustrate that quantum circuit-related datasets are likely to be quantum advantageous, but also that state-of-the-art machine learning methodologies can competently classify and recognize quantum circuits. We finally entrust the quantum and classical machine learning community the fundamental challenge to build more quantum-classical datasets like ours and to build future benchmarks from our experiments. The dataset is accessible on GitHub and its circuits are easily run in qulacs or qiskit.

{{</citation>}}


## cs.IT (1)



### (103/104) Constrained RS coding for Low Peak to Average Power Ratio in FBMC -- OQAM Systems (Job Chunkath et al., 2023)

{{<citation>}}

Job Chunkath, V. S. Sheeba, Nisha Varghese. (2023)  
**Constrained RS coding for Low Peak to Average Power Ratio in FBMC -- OQAM Systems**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2306.16685v2)  

---


**ABSTRACT**  
Multi-carrier modulation techniques have now become a standard in many communication protocols. Filter bank based multi-carrier (FBMC) generation techniques have been discussed in the literature as a means for overcoming the shortcomings of IFFT/FFT based OFDM system. The Peak to Average Power Ratio (PAPR) is a problem faced by all multi-carrier techniques. This paper discusses the methods for reducing PAPR in a FBMC system while maintaining acceptable Bit Error Rate (BER). A new PAPR minimizing scheme called Constrained Reed Solomon (CRS) coding is proposed. The hybrid techniques using coding and companding are tested for different channel models and is found to yield promising results.

{{</citation>}}


## cs.LO (1)



### (104/104) Beyond Logic Programming for Legal Reasoning (Ha-Thanh Nguyen et al., 2023)

{{<citation>}}

Ha-Thanh Nguyen, Francesca Toni, Kostas Stathis, Ken Satoh. (2023)  
**Beyond Logic Programming for Legal Reasoning**  

---
Primary Category: cs.LO  
Categories: cs-LO, cs.LO  
Keywords: Legal, Reasoning  
[Paper Link](http://arxiv.org/abs/2306.16632v1)  

---


**ABSTRACT**  
Logic programming has long being advocated for legal reasoning, and several approaches have been put forward relying upon explicit representation of the law in logic programming terms. In this position paper we focus on the PROLEG logic-programming-based framework for formalizing and reasoning with Japanese presupposed ultimate fact theory. Specifically, we examine challenges and opportunities in leveraging deep learning techniques for improving legal reasoning using PROLEG identifying four distinct options ranging from enhancing fact extraction using deep learning to end-to-end solutions for reasoning with textual legal descriptions. We assess advantages and limitations of each option, considering their technical feasibility, interpretability, and alignment with the needs of legal practitioners and decision-makers. We believe that our analysis can serve as a guideline for developers aiming to build effective decision-support systems for the legal domain, while fostering a deeper understanding of challenges and potential advancements by neuro-symbolic approaches in legal applications.

{{</citation>}}
