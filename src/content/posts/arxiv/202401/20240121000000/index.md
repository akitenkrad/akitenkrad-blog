---
draft: false
title: "arXiv @ 2024.01.21"
date: 2024-01-21
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.21"
    identifier: arxiv_20240121
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [eess.AS (4)](#eessas-4)
- [cs.CL (25)](#cscl-25)
- [cs.HC (8)](#cshc-8)
- [cs.CV (16)](#cscv-16)
- [cs.CY (1)](#cscy-1)
- [cs.CR (7)](#cscr-7)
- [cs.SE (4)](#csse-4)
- [cs.RO (1)](#csro-1)
- [cs.LG (12)](#cslg-12)
- [q-bio.BM (1)](#q-biobm-1)
- [cs.AI (5)](#csai-5)
- [cs.AR (2)](#csar-2)
- [eess.IV (3)](#eessiv-3)
- [cs.IR (4)](#csir-4)
- [cs.LO (1)](#cslo-1)
- [quant-ph (1)](#quant-ph-1)
- [cs.NI (1)](#csni-1)
- [cs.SD (1)](#cssd-1)
- [eess.SY (1)](#eesssy-1)

## eess.AS (4)



### (1/98) StreamVoice: Streamable Context-Aware Language Modeling for Real-time Zero-Shot Voice Conversion (Zhichao Wang et al., 2024)

{{<citation>}}

Zhichao Wang, Yuanzhe Chen, Xinsheng Wang, Zhuo Chen, Lei Xie, Yuping Wang, Yuxuan Wang. (2024)  
**StreamVoice: Streamable Context-Aware Language Modeling for Real-time Zero-Shot Voice Conversion**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.11053v1)  

---


**ABSTRACT**  
Recent language model (LM) advancements have showcased impressive zero-shot voice conversion (VC) performance. However, existing LM-based VC models usually apply offline conversion from source semantics to acoustic features, demanding the complete source speech, and limiting their deployment to real-time applications. In this paper, we introduce StreamVoice, a novel streaming LM-based model for zero-shot VC, facilitating real-time conversion given arbitrary speaker prompts and source speech. Specifically, to enable streaming capability, StreamVoice employs a fully causal context-aware LM with a temporal-independent acoustic predictor, while alternately processing semantic and acoustic features at each time step of autoregression which eliminates the dependence on complete source speech. To address the potential performance degradation from the incomplete context in streaming processing, we enhance the context-awareness of the LM through two strategies: 1) teacher-guided context foresight, using a teacher model to summarize the present and future semantic context during training to guide the model's forecasting for missing context; 2) semantic masking strategy, promoting acoustic prediction from preceding corrupted semantic and acoustic input, enhancing context-learning ability. Notably, StreamVoice is the first LM-based streaming zero-shot VC model without any future look-ahead. Experimental results demonstrate StreamVoice's streaming conversion capability while maintaining zero-shot performance comparable to non-streaming VC systems.

{{</citation>}}


### (2/98) Revealing Emotional Clusters in Speaker Embeddings: A Contrastive Learning Strategy for Speech Emotion Recognition (Ismail Rasim Ulgen et al., 2024)

{{<citation>}}

Ismail Rasim Ulgen, Zongyang Du, Carlos Busso, Berrak Sisman. (2024)  
**Revealing Emotional Clusters in Speaker Embeddings: A Contrastive Learning Strategy for Speech Emotion Recognition**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Contrastive Learning, Embedding, Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2401.11017v1)  

---


**ABSTRACT**  
Speaker embeddings carry valuable emotion-related information, which makes them a promising resource for enhancing speech emotion recognition (SER), especially with limited labeled data. Traditionally, it has been assumed that emotion information is indirectly embedded within speaker embeddings, leading to their under-utilization. Our study reveals a direct and useful link between emotion and state-of-the-art speaker embeddings in the form of intra-speaker clusters. By conducting a thorough clustering analysis, we demonstrate that emotion information can be readily extracted from speaker embeddings. In order to leverage this information, we introduce a novel contrastive pretraining approach applied to emotion-unlabeled data for speech emotion recognition. The proposed approach involves the sampling of positive and the negative examples based on the intra-speaker clusters of speaker embeddings. The proposed strategy, which leverages extensive emotion-unlabeled data, leads to a significant improvement in SER performance, whether employed as a standalone pretraining task or integrated into a multi-task pretraining setting.

{{</citation>}}


### (3/98) Multilingual acoustic word embeddings for zero-resource languages (Christiaan Jacobs et al., 2024)

{{<citation>}}

Christiaan Jacobs, Herman Kamper. (2024)  
**Multilingual acoustic word embeddings for zero-resource languages**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2401.10543v1)  

---


**ABSTRACT**  
This research addresses the challenge of developing speech applications for zero-resource languages that lack labelled data. It specifically uses acoustic word embedding (AWE) -- fixed-dimensional representations of variable-duration speech segments -- employing multilingual transfer, where labelled data from several well-resourced languages are used for pertaining. The study introduces a new neural network that outperforms existing AWE models on zero-resource languages. It explores the impact of the choice of well-resourced languages. AWEs are applied to a keyword-spotting system for hate speech detection in Swahili radio broadcasts, demonstrating robustness in real-world scenarios. Additionally, novel semantic AWE models improve semantic query-by-example search.

{{</citation>}}


### (4/98) Contextualized Automatic Speech Recognition with Attention-Based Bias Phrase Boosted Beam Search (Yui Sudo et al., 2024)

{{<citation>}}

Yui Sudo, Muhammad Shakeel, Yosuke Fukumoto, Yifan Peng, Shinji Watanabe. (2024)  
**Contextualized Automatic Speech Recognition with Attention-Based Bias Phrase Boosted Beam Search**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Attention, Bias, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.10449v1)  

---


**ABSTRACT**  
End-to-end (E2E) automatic speech recognition (ASR) methods exhibit remarkable performance. However, since the performance of such methods is intrinsically linked to the context present in the training data, E2E-ASR methods do not perform as desired for unseen user contexts (e.g., technical terms, personal names, and playlists). Thus, E2E-ASR methods must be easily contextualized by the user or developer. This paper proposes an attention-based contextual biasing method that can be customized using an editable phrase list (referred to as a bias list). The proposed method can be trained effectively by combining a bias phrase index loss and special tokens to detect the bias phrases in the input speech data. In addition, to improve the contextualization performance during inference further, we propose a bias phrase boosted (BPB) beam search algorithm based on the bias phrase index probability. Experimental results demonstrate that the proposed method consistently improves the word error rate and the character error rate of the target phrases in the bias list on both the Librispeech-960 (English) and our in-house (Japanese) dataset, respectively.

{{</citation>}}


## cs.CL (25)



### (5/98) Mining experimental data from Materials Science literature with Large Language Models (Luca Foppiano et al., 2024)

{{<citation>}}

Luca Foppiano, Guillaume Lambard, Toshiyuki Amagasa, Masashi Ishii. (2024)  
**Mining experimental data from Materials Science literature with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT, GPT-3.5, GPT-4, Language Model, NER  
[Paper Link](http://arxiv.org/abs/2401.11052v1)  

---


**ABSTRACT**  
This study is dedicated to evaluating the capabilities of advanced large language models (LLMs) such as GPT-3.5-Turbo, GPT-4, and GPT-4-Turbo in the extraction of structured information from scientific documents within the field of materials science. We introduce a novel methodology for the comparative analysis of intricate material expressions, emphasising the standardisation of chemical formulas to tackle the complexities inherent in materials science information assessment. To this end, we primarily focus on two critical tasks of information extraction: (i) a named entity recognition (NER) of studied materials and physical properties and (ii) a relation extraction (RE) between these entities. The performance of LLMs in executing these tasks is benchmarked against traditional models based on the BERT architecture and rule-based approaches. For NER, LLMs fail to outperform the baseline with zero-shot prompting and exhibit only limited improvement with few-shot prompting. However, for RE, a GPT-3.5-Turbo fine-tuned with the appropriate strategy outperforms all models, including the baseline. Without any fine-tuning, GPT-4 and GPT-4-Turbo display remarkable reasoning and relationship extraction capabilities after being provided with merely a couple of examples, surpassing the baseline. Overall, the results suggest that although LLMs demonstrate relevant reasoning skills in connecting concepts, for tasks requiring extracting complex domain-specific entities like materials, specialised models are currently a better choice.

{{</citation>}}


### (6/98) PubTator 3.0: an AI-powered Literature Resource for Unlocking Biomedical Knowledge (Chih-Hsuan Wei et al., 2024)

{{<citation>}}

Chih-Hsuan Wei, Alexis Allot, Po-Ting Lai, Robert Leaman, Shubo Tian, Ling Luo, Qiao Jin, Zhizheng Wang, Qingyu Chen, Zhiyong Lu. (2024)  
**PubTator 3.0: an AI-powered Literature Resource for Unlocking Biomedical Knowledge**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, q-bio-QM  
Keywords: AI, ChatGPT, GPT, GPT-4, Google  
[Paper Link](http://arxiv.org/abs/2401.11048v1)  

---


**ABSTRACT**  
PubTator 3.0 (https://www.ncbi.nlm.nih.gov/research/pubtator3/) is a biomedical literature resource using state-of-the-art AI techniques to offer semantic and relation searches for key concepts like proteins, genetic variants, diseases, and chemicals. It currently provides over one billion entity and relation annotations across approximately 36 million PubMed abstracts and 6 million full-text articles from the PMC open access subset, updated weekly. PubTator 3.0's online interface and API utilize these precomputed entity relations and synonyms to provide advanced search capabilities and enable large-scale analyses, streamlining many complex information needs. We showcase the retrieval quality of PubTator 3.0 using a series of entity pair queries, demonstrating that PubTator 3.0 retrieves a greater number of articles than either PubMed or Google Scholar, with higher precision in the top 20 results. We further show that integrating ChatGPT (GPT-4) with PubTator APIs dramatically improves the factuality and verifiability of its responses. In summary, PubTator 3.0 offers a comprehensive set of features and tools that allow researchers to navigate the ever-expanding wealth of biomedical literature, expediting research and unlocking valuable insights for scientific discovery.

{{</citation>}}


### (7/98) FAIR Enough: How Can We Develop and Assess a FAIR-Compliant Dataset for Large Language Models' Training? (Shaina Raza et al., 2024)

{{<citation>}}

Shaina Raza, Shardul Ghuge, Chen Ding, Deval Pandya. (2024)  
**FAIR Enough: How Can We Develop and Assess a FAIR-Compliant Dataset for Large Language Models' Training?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.11033v1)  

---


**ABSTRACT**  
Advancements in Large Language Models (LLMs) highlight the need for ethical practices and data integrity. We introduce a framework that embeds FAIR (Findable, Accessible, Interoperable, Reusable) data principles into LLM training. This approach marks a shift towards practices compliant with FAIR standards. Our framework presents guidelines for integrating FAIR data principles into LLM training. This initiative includes a checklist for researchers and developers. We also demonstrate its practical application through a case study focused on bias identification and mitigation in our FAIR-compliant dataset. This work is a significant contribution to AI ethics and data science, advocating for balanced and ethical training methods in LLMs.

{{</citation>}}


### (8/98) Analysis and Detection of Multilingual Hate Speech Using Transformer Based Deep Learning (Arijit Das et al., 2024)

{{<citation>}}

Arijit Das, Somashree Nandy, Rupam Saha, Srijan Das, Diganta Saha. (2024)  
**Analysis and Detection of Multilingual Hate Speech Using Transformer Based Deep Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Multilingual, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11021v1)  

---


**ABSTRACT**  
Hate speech is harmful content that directly attacks or promotes hatred against members of groups or individuals based on actual or perceived aspects of identity, such as racism, religion, or sexual orientation. This can affect social life on social media platforms as hateful content shared through social media can harm both individuals and communities. As the prevalence of hate speech increases online, the demand for automated detection as an NLP task is increasing. In this work, the proposed method is using transformer-based model to detect hate speech in social media, like twitter, Facebook, WhatsApp, Instagram, etc. The proposed model is independent of languages and has been tested on Italian, English, German, Bengali. The Gold standard datasets were collected from renowned researcher Zeerak Talat, Sara Tonelli, Melanie Siegel, and Rezaul Karim. The success rate of the proposed model for hate speech detection is higher than the existing baseline and state-of-the-art models with accuracy in Bengali dataset is 89%, in English: 91%, in German dataset 91% and in Italian dataset it is 77%. The proposed algorithm shows substantial improvement to the benchmark method.

{{</citation>}}


### (9/98) The Radiation Oncology NLP Database (Zhengliang Liu et al., 2024)

{{<citation>}}

Zhengliang Liu, Jason Holmes, Wenxiong Liao, Chenbin Liu, Lian Zhang, Hongying Feng, Peilong Wang, Muhammad Ali Elahi, Hongmin Cai, Lichao Sun, Quanzheng Li, Xiang Li, Tianming Liu, Jiajian Shen, Wei Liu. (2024)  
**The Radiation Oncology NLP Database**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, physics-med-ph  
Keywords: NER, NLP, Named Entity Recognition, Natural Language Processing, QA, Question Answering, Reasoning, Summarization, Text Classification, Text Summarization  
[Paper Link](http://arxiv.org/abs/2401.10995v1)  

---


**ABSTRACT**  
We present the Radiation Oncology NLP Database (ROND), the first dedicated Natural Language Processing (NLP) dataset for radiation oncology, an important medical specialty that has received limited attention from the NLP community in the past. With the advent of Artificial General Intelligence (AGI), there is an increasing need for specialized datasets and benchmarks to facilitate research and development. ROND is specifically designed to address this gap in the domain of radiation oncology, a field that offers many opportunities for NLP exploration. It encompasses various NLP tasks including Logic Reasoning, Text Classification, Named Entity Recognition (NER), Question Answering (QA), Text Summarization, and Patient-Clinician Conversations, each with a distinct focus on radiation oncology concepts and application cases. In addition, we have developed an instruction-tuning dataset consisting of over 20k instruction pairs (based on ROND) and trained a large language model, CancerChat. This serves to demonstrate the potential of instruction-tuning large language models within a highly-specialized medical domain. The evaluation results in this study could serve as baseline results for future research. ROND aims to stimulate advancements in radiation oncology and clinical NLP by offering a platform for testing and improving algorithms and models in a domain-specific context. The ROND dataset is a joint effort of multiple U.S. health institutions. The data is available at https://github.com/zl-liu/Radiation-Oncology-NLP-Database.

{{</citation>}}


### (10/98) Reinforcement learning for question answering in programming domain using public community scoring as a human feedback (Alexey Gorbatovski et al., 2024)

{{<citation>}}

Alexey Gorbatovski, Sergey Kovalchuk. (2024)  
**Reinforcement learning for question answering in programming domain using public community scoring as a human feedback**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: GPT, Language Model, QA, Question Answering, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.10882v1)  

---


**ABSTRACT**  
In this study, we investigate the enhancement of the GPT Neo 125M performance in Community Question Answering (CQA) with a focus on programming, through the integration of Reinforcement Learning from Human Feedback (RLHF) and the utilization of scores from Stack Overflow. Two distinct reward model training strategies are employed for fine-tuning with Proximal Policy Optimization (PPO). Notably, the improvements in performance achieved through this method are comparable to those of GPT Neo 2.7B parameter variant. Additionally, an auxiliary scoring mechanism is introduced, which demonstrates the limitations of conventional linguistic metrics in evaluating responses in the programming domain. Through accurate analysis, this paper looks at the divergence between traditional linguistic metrics and our human-preferences-based reward model, underscoring the imperative for domain-specific evaluation methods. By elucidating the complexities involved in applying RLHF to programming CQA and accentuating the significance of context-aware evaluation, this study contributes to the ongoing efforts in refining Large Language Models through focused human feedback.

{{</citation>}}


### (11/98) Advancements in eHealth Data Analytics through Natural Language Processing and Deep Learning (Elena-Simona Apostol et al., 2024)

{{<citation>}}

Elena-Simona Apostol, Ciprian-Octavian Truică. (2024)  
**Advancements in eHealth Data Analytics through Natural Language Processing and Deep Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.10850v1)  

---


**ABSTRACT**  
The healthcare environment is commonly referred to as "information-rich" but also "knowledge poor". Healthcare systems collect huge amounts of data from various sources: lab reports, medical letters, logs of medical tools or programs, medical prescriptions, etc. These massive sets of data can provide great knowledge and information that can improve the medical services, and overall the healthcare domain, such as disease prediction by analyzing the patient's symptoms or disease prevention, by facilitating the discovery of behavioral factors for diseases. Unfortunately, only a relatively small volume of the textual eHealth data is processed and interpreted, an important factor being the difficulty in efficiently performing Big Data operations. In the medical field, detecting domain-specific multi-word terms is a crucial task as they can define an entire concept with a few words. A term can be defined as a linguistic structure or a concept, and it is composed of one or more words with a specific meaning to a domain. All the terms of a domain create its terminology. This chapter offers a critical study of the current, most performant solutions for analyzing unstructured (image and textual) eHealth data. This study also provides a comparison of the current Natural Language Processing and Deep Learning techniques in the eHealth context. Finally, we examine and discuss some of the current issues, and we define a set of research directions in this area.

{{</citation>}}


### (12/98) A survey on recent advances in named entity recognition (Imed Keraghel et al., 2024)

{{<citation>}}

Imed Keraghel, Stanislas Morbieu, Mohamed Nadif. (2024)  
**A survey on recent advances in named entity recognition**  

---
Primary Category: cs.CL  
Categories: 68T50, 68Q32, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2401.10825v1)  

---


**ABSTRACT**  
Named Entity Recognition seeks to extract substrings within a text that name real-world objects and to determine their type (for example, whether they refer to persons or organizations). In this survey, we first present an overview of recent popular approaches, but we also look at graph- and transformer- based methods including Large Language Models (LLMs) that have not had much coverage in other surveys. Second, we focus on methods designed for datasets with scarce annotations. Third, we evaluate the performance of the main NER implementations on a variety of datasets with differing characteristics (as regards their domain, their size, and their number of classes). We thus provide a deep comparison of algorithms that are never considered together. Our experiments shed some light on how the characteristics of datasets affect the behavior of the methods that we compare.

{{</citation>}}


### (13/98) Mitigating Hallucinations of Large Language Models via Knowledge Consistent Alignment (Fanqi Wan et al., 2024)

{{<citation>}}

Fanqi Wan, Xinting Huang, Leyang Cui, Xiaojun Quan, Wei Bi, Shuming Shi. (2024)  
**Mitigating Hallucinations of Large Language Models via Knowledge Consistent Alignment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.10768v1)  

---


**ABSTRACT**  
While Large Language Models (LLMs) have proven to be exceptional on a variety of tasks after alignment, they may still produce responses that contradict the context or world knowledge confidently, a phenomenon known as ``hallucination''. In this paper, we demonstrate that reducing the inconsistency between the external knowledge encapsulated in the training data and the intrinsic knowledge inherited in the pretraining corpus could mitigate hallucination in alignment. Specifically, we introduce a novel knowledge consistent alignment (KCA) approach, which involves automatically formulating examinations based on external knowledge for accessing the comprehension of LLMs. For data encompassing knowledge inconsistency, KCA implements several simple yet efficient strategies for processing. We illustrate the superior performance of the proposed KCA approach in mitigating hallucinations across six benchmarks using LLMs of different backbones and scales. Furthermore, we confirm the correlation between knowledge inconsistency and hallucination, signifying the effectiveness of reducing knowledge inconsistency in alleviating hallucinations. Our code, model weights, and data are public at \url{https://github.com/fanqiwan/KCA}.

{{</citation>}}


### (14/98) Structured Code Representations Enable Data-Efficient Adaptation of Code Language Models (Mayank Agarwal et al., 2024)

{{<citation>}}

Mayank Agarwal, Yikang Shen, Bailin Wang, Yoon Kim, Jie Chen. (2024)  
**Structured Code Representations Enable Data-Efficient Adaptation of Code Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.10716v1)  

---


**ABSTRACT**  
Current language models tailored for code tasks often adopt the pre-training-then-fine-tuning paradigm from natural language processing, modeling source code as plain text. This approach, however, overlooks the unambiguous structures inherent in programming languages. In this work, we explore data-efficient adaptation of pre-trained code models by further pre-training and fine-tuning them with program structures. Specifically, we represent programs as parse trees -- also known as concrete syntax trees (CSTs) -- and adapt pre-trained models on serialized CSTs. Although the models that we adapt have been pre-trained only on the surface form of programs, we find that a small amount of continual pre-training and fine-tuning on CSTs without changing the model architecture yields improvements over the baseline approach across various code tasks. The improvements are found to be particularly significant when there are limited training examples, demonstrating the effectiveness of integrating program structures with plain-text representation even when working with backbone models that have not been pre-trained with structures.

{{</citation>}}


### (15/98) LangBridge: Multilingual Reasoning Without Multilingual Supervision (Dongkeun Yoon et al., 2024)

{{<citation>}}

Dongkeun Yoon, Joel Jang, Sungdong Kim, Seungone Kim, Sheikh Shafayat, Minjoon Seo. (2024)  
**LangBridge: Multilingual Reasoning Without Multilingual Supervision**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual, Reasoning, T5  
[Paper Link](http://arxiv.org/abs/2401.10695v1)  

---


**ABSTRACT**  
We introduce LangBridge, a zero-shot approach to adapt language models for multilingual reasoning tasks without multilingual supervision. LangBridge operates by bridging two models, each specialized in different aspects: (1) one specialized in understanding multiple languages (e.g., mT5 encoder) and (2) one specialized in reasoning (e.g., Orca 2). LangBridge connects the two models by introducing minimal trainable parameters between them. Despite utilizing only English data for training, LangBridge considerably enhances the performance of language models on low-resource languages across mathematical reasoning, coding, and logical reasoning. Our analysis suggests that the efficacy of LangBridge stems from the language-agnostic characteristics of multilingual representations. We publicly release our code and models.

{{</citation>}}


### (16/98) A Simple Framework to Accelerate Multilingual Language Model for Monolingual Text Generation (Jimin Hong et al., 2024)

{{<citation>}}

Jimin Hong, Gibbeum Lee, Jaewoong Cho. (2024)  
**A Simple Framework to Accelerate Multilingual Language Model for Monolingual Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Multilingual, Text Generation  
[Paper Link](http://arxiv.org/abs/2401.10660v1)  

---


**ABSTRACT**  
Recent advancements in large language models have facilitated the execution of complex language tasks, not only in English but also in non-English languages. However, the tokenizers of most language models, such as Llama, trained on English-centric corpora, tend to excessively fragment tokens in non-English languages. This issue is especially pronounced in non-roman alphabetic languages, which are often divided at a character or even Unicode level, leading to slower text generation. To address this, our study introduces a novel framework designed to expedite text generation in these languages. This framework predicts larger linguistic units than those of conventional multilingual tokenizers and is specifically tailored to the target language, thereby reducing the number of decoding steps required. Our empirical results demonstrate that the proposed framework increases the generation speed by a factor of 1.9 compared to standard decoding while maintaining the performance of a pre-trained multilingual model on monolingual tasks.

{{</citation>}}


### (17/98) Attentive Fusion: A Transformer-based Approach to Multimodal Hate Speech Detection (Atanu Mandal et al., 2024)

{{<citation>}}

Atanu Mandal, Gargi Roy, Amit Barman, Indranil Dutta, Sudip Kumar Naskar. (2024)  
**Attentive Fusion: A Transformer-based Approach to Multimodal Hate Speech Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SD, cs.CL, eess-AS, eess-SP  
Keywords: Hate Speech Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2401.10653v1)  

---


**ABSTRACT**  
With the recent surge and exponential growth of social media usage, scrutinizing social media content for the presence of any hateful content is of utmost importance. Researchers have been diligently working since the past decade on distinguishing between content that promotes hatred and content that does not. Traditionally, the main focus has been on analyzing textual content. However, recent research attempts have also commenced into the identification of audio-based content. Nevertheless, studies have shown that relying solely on audio or text-based content may be ineffective, as recent upsurge indicates that individuals often employ sarcasm in their speech and writing. To overcome these challenges, we present an approach to identify whether a speech promotes hate or not utilizing both audio and textual representations. Our methodology is based on the Transformer framework that incorporates both audio and text sampling, accompanied by our very own layer called "Attentive Fusion". The results of our study surpassed previous state-of-the-art techniques, achieving an impressive macro F1 score of 0.927 on the Test Set.

{{</citation>}}


### (18/98) Sowing the Wind, Reaping the Whirlwind: The Impact of Editing Language Models (Rima Hazra et al., 2024)

{{<citation>}}

Rima Hazra, Sayan Layek, Somnath Banerjee, Soujanya Poria. (2024)  
**Sowing the Wind, Reaping the Whirlwind: The Impact of Editing Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2401.10647v1)  

---


**ABSTRACT**  
In the rapidly advancing field of artificial intelligence, the concept of Red-Teaming or Jailbreaking large language models (LLMs) has emerged as a crucial area of study. This approach is especially significant in terms of assessing and enhancing the safety and robustness of these models. This paper investigates the intricate consequences of such modifications through model editing, uncovering a complex relationship between enhancing model accuracy and preserving its ethical integrity. Our in-depth analysis reveals a striking paradox: while injecting accurate information is crucial for model reliability, it can paradoxically destabilize the model's foundational framework, resulting in unpredictable and potentially unsafe behaviors. Additionally, we propose a benchmark dataset NicheHazardQA to investigate this unsafe behavior both within the same and cross topical domain. This aspect of our research sheds light on how the edits, impact the model's safety metrics and guardrails. Our findings show that model editing serves as a cost-effective tool for topical red-teaming by methodically applying targeted edits and evaluating the resultant model behavior

{{</citation>}}


### (19/98) Speech Swin-Transformer: Exploring a Hierarchical Transformer with Shifted Windows for Speech Emotion Recognition (Yong Wang et al., 2024)

{{<citation>}}

Yong Wang, Cheng Lu, Hailun Lian, Yan Zhao, Björn Schuller, Yuan Zong, Wenming Zheng. (2024)  
**Speech Swin-Transformer: Exploring a Hierarchical Transformer with Shifted Windows for Speech Emotion Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Emotion Recognition, Transformer  
[Paper Link](http://arxiv.org/abs/2401.10536v1)  

---


**ABSTRACT**  
Swin-Transformer has demonstrated remarkable success in computer vision by leveraging its hierarchical feature representation based on Transformer. In speech signals, emotional information is distributed across different scales of speech features, e.\,g., word, phrase, and utterance. Drawing above inspiration, this paper presents a hierarchical speech Transformer with shifted windows to aggregate multi-scale emotion features for speech emotion recognition (SER), called Speech Swin-Transformer. Specifically, we first divide the speech spectrogram into segment-level patches in the time domain, composed of multiple frame patches. These segment-level patches are then encoded using a stack of Swin blocks, in which a local window Transformer is utilized to explore local inter-frame emotional information across frame patches of each segment patch. After that, we also design a shifted window Transformer to compensate for patch correlations near the boundaries of segment patches. Finally, we employ a patch merging operation to aggregate segment-level emotional features for hierarchical speech representation by expanding the receptive field of Transformer from frame-level to segment-level. Experimental results demonstrate that our proposed Speech Swin-Transformer outperforms the state-of-the-art methods.

{{</citation>}}


### (20/98) The 'Colonial Impulse' of Natural Language Processing: An Audit of Bengali Sentiment Analysis Tools and Their Identity-based Biases (Dipto Das et al., 2024)

{{<citation>}}

Dipto Das, Shion Guha, Jed Brubaker, Bryan Semaan. (2024)  
**The 'Colonial Impulse' of Natural Language Processing: An Audit of Bengali Sentiment Analysis Tools and Their Identity-based Biases**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-HC, cs-LG, cs.CL  
Keywords: Bias, Natural Language Processing, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2401.10535v1)  

---


**ABSTRACT**  
While colonization has sociohistorically impacted people's identities across various dimensions, those colonial values and biases continue to be perpetuated by sociotechnical systems. One category of sociotechnical systems--sentiment analysis tools--can also perpetuate colonial values and bias, yet less attention has been paid to how such tools may be complicit in perpetuating coloniality, although they are often used to guide various practices (e.g., content moderation). In this paper, we explore potential bias in sentiment analysis tools in the context of Bengali communities that have experienced and continue to experience the impacts of colonialism. Drawing on identity categories most impacted by colonialism amongst local Bengali communities, we focused our analytic attention on gender, religion, and nationality. We conducted an algorithmic audit of all sentiment analysis tools for Bengali, available on the Python package index (PyPI) and GitHub. Despite similar semantic content and structure, our analyses showed that in addition to inconsistencies in output from different tools, Bengali sentiment analysis tools exhibit bias between different identity categories and respond differently to different ways of identity expression. Connecting our findings with colonially shaped sociocultural structures of Bengali communities, we discuss the implications of downstream bias of sentiment analysis tools.

{{</citation>}}


### (21/98) Cross-lingual Editing in Multilingual Language Models (Himanshu Beniwal et al., 2024)

{{<citation>}}

Himanshu Beniwal, Kowsik Nandagopan D, Mayank Singh. (2024)  
**Cross-lingual Editing in Multilingual Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, BLOOM, Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2401.10521v1)  

---


**ABSTRACT**  
The training of large language models (LLMs) necessitates substantial data and computational resources, and updating outdated LLMs entails significant efforts and resources. While numerous model editing techniques (METs) have emerged to efficiently update model outputs without retraining, their effectiveness in multilingual LLMs, where knowledge is stored in diverse languages, remains an underexplored research area. This research paper introduces the cross-lingual model editing (\textbf{XME}) paradigm, wherein a fact is edited in one language, and the subsequent update propagation is observed across other languages. To investigate the XME paradigm, we conducted experiments using BLOOM, mBERT, and XLM-RoBERTa using the two writing scripts: \textit{Latin} (English, French, and Spanish) and \textit{Indic} (Hindi, Gujarati, and Bengali). The results reveal notable performance limitations of state-of-the-art METs under the XME setting, mainly when the languages involved belong to two distinct script families. These findings highlight the need for further research and development of XME techniques to address these challenges. For more comprehensive information, the dataset used in this research and the associated code are publicly available at the following URL\url{https://github.com/lingo-iitgn/XME}.

{{</citation>}}


### (22/98) FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis (Chao Zhang et al., 2024)

{{<citation>}}

Chao Zhang, Yuren Mao, Yijiang Fan, Yu Mi, Yunjun Gao, Lu Chen, Dongfang Lou, Jinshu Lin. (2024)  
**FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-DB, cs.CL  
Keywords: Financial, Language Model  
[Paper Link](http://arxiv.org/abs/2401.10506v1)  

---


**ABSTRACT**  
Text-to-SQL, which provides zero-code interface for operating relational databases, has gained much attention in financial analysis; because, financial professionals may not well-skilled in SQL programming. However, until now, there is no practical Text-to-SQL benchmark dataset for financial analysis, and existing Text-to-SQL methods have not considered the unique characteristics of databases in financial applications, such as commonly existing wide tables. To address these issues, we collect a practical Text-to-SQL benchmark dataset and propose a model-agnostic Large Language Model (LLMs)-based Text-to-SQL framework for financial analysis. The benchmark dataset, BULL, is collected from the practical financial analysis business of Hundsun Technologies Inc., including databases for fund, stock, and macro economy. Besides, the proposed LLMs-based Text-to-SQL framework, FinSQL, provides a systematic treatment for financial Text-to-SQL from the perspectives of prompt construction, parameter-efficient fine-tuning and output calibration. Extensive experimental results on BULL demonstrate that FinSQL achieves the state-of-the-art Text-to-SQL performance at a small cost; furthermore, FinSQL can bring up to 36.64% performance improvement in scenarios requiring few-shot cross-database model transfer.

{{</citation>}}


### (23/98) Knowledge Fusion of Large Language Models (Fanqi Wan et al., 2024)

{{<citation>}}

Fanqi Wan, Xinting Huang, Deng Cai, Xiaojun Quan, Wei Bi, Shuming Shi. (2024)  
**Knowledge Fusion of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2401.10491v2)  

---


**ABSTRACT**  
While training large language models (LLMs) from scratch can generate models with distinct functionalities and strengths, it comes at significant costs and may result in redundant capabilities. Alternatively, a cost-effective and compelling approach is to merge existing pre-trained LLMs into a more potent model. However, due to the varying architectures of these LLMs, directly blending their weights is impractical. In this paper, we introduce the notion of knowledge fusion for LLMs, aimed at combining the capabilities of existing LLMs and transferring them into a single LLM. By leveraging the generative distributions of source LLMs, we externalize their collective knowledge and unique strengths, thereby potentially elevating the capabilities of the target model beyond those of any individual source LLM. We validate our approach using three popular LLMs with different architectures--Llama-2, MPT, and OpenLLaMA--across various benchmarks and tasks. Our findings confirm that the fusion of LLMs can improve the performance of the target model across a range of capabilities such as reasoning, commonsense, and code generation. Our code, model weights, and data are public at \url{https://github.com/fanqiwan/FuseLLM}.

{{</citation>}}


### (24/98) Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning (Yiwei Li et al., 2024)

{{<citation>}}

Yiwei Li, Peiwen Yuan, Shaoxiong Feng, Boyuan Pan, Xinglin Wang, Bin Sun, Heda Wang, Kan Li. (2024)  
**Escape Sky-high Cost: Early-stopping Self-Consistency for Multi-step Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.10480v1)  

---


**ABSTRACT**  
Self-consistency (SC) has been a widely used decoding strategy for chain-of-thought reasoning. Despite bringing significant performance improvements across a variety of multi-step reasoning tasks, it is a high-cost method that requires multiple sampling with the preset size. In this paper, we propose a simple and scalable sampling process, \textbf{E}arly-Stopping \textbf{S}elf-\textbf{C}onsistency (ESC), to greatly reduce the cost of SC without sacrificing performance. On this basis, one control scheme for ESC is further derivated to dynamically choose the performance-cost balance for different tasks and models. To demonstrate ESC's effectiveness, we conducted extensive experiments on three popular categories of reasoning tasks: arithmetic, commonsense and symbolic reasoning over language models with varying scales. The empirical results show that ESC reduces the average number of sampling of chain-of-thought reasoning by a significant margin on six benchmarks, including MATH (-33.8%), GSM8K (-80.1%), StrategyQA (-76.8%), CommonsenseQA (-78.5%), Coin Flip (-84.2%) and Last Letters (-67.4%), while attaining comparable performances.

{{</citation>}}


### (25/98) Name Tagging Under Domain Shift via Metric Learning for Life Sciences (Hongyi Liu et al., 2024)

{{<citation>}}

Hongyi Liu, Qingyun Wang, Payam Karisani, Heng Ji. (2024)  
**Name Tagging Under Domain Shift via Metric Learning for Life Sciences**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Information Extraction  
[Paper Link](http://arxiv.org/abs/2401.10472v1)  

---


**ABSTRACT**  
Name tagging is a key component of Information Extraction (IE), particularly in scientific domains such as biomedicine and chemistry, where large language models (LLMs), e.g., ChatGPT, fall short. We investigate the applicability of transfer learning for enhancing a name tagging model trained in the biomedical domain (the source domain) to be used in the chemical domain (the target domain). A common practice for training such a model in a few-shot learning setting is to pretrain the model on the labeled source data, and then, to finetune it on a hand-full of labeled target examples. In our experiments we observed that such a model is prone to mis-labeling the source entities, which can often appear in the text, as the target entities. To alleviate this problem, we propose a model to transfer the knowledge from the source domain to the target domain, however, at the same time, to project the source entities and target entities into separate regions of the feature space. This diminishes the risk of mis-labeling the source entities as the target entities. Our model consists of two stages: 1) entity grouping in the source domain, which incorporates knowledge from annotated events to establish relations between entities, and 2) entity discrimination in the target domain, which relies on pseudo labeling and contrastive learning to enhance discrimination between the entities in the two domains. We carry out our extensive experiments across three source and three target datasets, and demonstrate that our method outperforms the baselines, in some scenarios by 5\% absolute value.

{{</citation>}}


### (26/98) Critical Data Size of Language Models from a Grokking Perspective (Xuekai Zhu et al., 2024)

{{<citation>}}

Xuekai Zhu, Yao Fu, Bowen Zhou, Zhouhan Lin. (2024)  
**Critical Data Size of Language Models from a Grokking Perspective**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.10463v1)  

---


**ABSTRACT**  
We explore the critical data size in language models, a threshold that marks a fundamental shift from quick memorization to slow generalization. We formalize the phase transition under the grokking configuration into the Data Efficiency Hypothesis and identify data insufficiency, sufficiency, and surplus regimes in language models training dynamics. We develop a grokking configuration to reproduce grokking on simplistic language models stably by rescaling initialization and weight decay. We show that generalization occurs only when language models reach a critical size. We analyze grokking across sample-wise and model-wise, verifying the proposed data efficiency hypothesis. Our experiments reveal smoother phase transitions occurring at the critical dataset size for language datasets. As the model size increases, this critical point also becomes larger, indicating that larger models require more data. Our results deepen the understanding of language model training, offering a novel perspective on the role of data in the learning mechanism of language models.

{{</citation>}}


### (27/98) Investigating Training Strategies and Model Robustness of Low-Rank Adaptation for Language Modeling in Speech Recognition (Yu Yu et al., 2024)

{{<citation>}}

Yu Yu, Chao-Han Huck Yang, Tuan Dinh, Sungho Ryu, Jari Kolehmainen, Roger Ren, Denis Filimonov, Prashanth G. Shivakumar, Ankur Gandhe, Ariya Rastow, Jia Xu, Ivan Bulyko, Andreas Stolcke. (2024)  
**Investigating Training Strategies and Model Robustness of Low-Rank Adaptation for Language Modeling in Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-NE, cs-SD, cs.CL, eess-AS  
Keywords: Language Model, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.10447v1)  

---


**ABSTRACT**  
The use of low-rank adaptation (LoRA) with frozen pretrained language models (PLMs) has become increasing popular as a mainstream, resource-efficient modeling approach for memory-constrained hardware. In this study, we first explore how to enhance model performance by introducing various LoRA training strategies, achieving relative word error rate reductions of 3.50\% on the public Librispeech dataset and of 3.67\% on an internal dataset in the messaging domain. To further characterize the stability of LoRA-based second-pass speech recognition models, we examine robustness against input perturbations. These perturbations are rooted in homophone replacements and a novel metric called N-best Perturbation-based Rescoring Robustness (NPRR), both designed to measure the relative degradation in the performance of rescoring models. Our experimental results indicate that while advanced variants of LoRA, such as dynamic rank-allocated LoRA, lead to performance degradation in $1$-best perturbation, they alleviate the degradation in $N$-best perturbation. This finding is in comparison to fully-tuned models and vanilla LoRA tuning baselines, suggesting that a comprehensive selection is needed when using LoRA-based adaptation for compute-cost savings and robust language modeling.

{{</citation>}}


### (28/98) Large Language Models are Efficient Learners of Noise-Robust Speech Recognition (Yuchen Hu et al., 2024)

{{<citation>}}

Yuchen Hu, Chen Chen, Chao-Han Huck Yang, Ruizhe Li, Chao Zhang, Pin-Yu Chen, EnSiong Chng. (2024)  
**Large Language Models are Efficient Learners of Noise-Robust Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: Language Model, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.10446v1)  

---


**ABSTRACT**  
Recent advances in large language models (LLMs) have promoted generative error correction (GER) for automatic speech recognition (ASR), which leverages the rich linguistic knowledge and powerful reasoning ability of LLMs to improve recognition results. The latest work proposes a GER benchmark with HyPoradise dataset to learn the mapping from ASR N-best hypotheses to ground-truth transcription by efficient LLM finetuning, which shows great effectiveness but lacks specificity on noise-robust ASR. In this work, we extend the benchmark to noisy conditions and investigate if we can teach LLMs to perform denoising for GER just like what robust ASR do}, where one solution is introducing noise information as a conditioner into LLM. However, directly incorporating noise embeddings from audio encoder could harm the LLM tuning due to cross-modality gap. To this end, we propose to extract a language-space noise embedding from the N-best list to represent the noise conditions of source speech, which can promote the denoising process in GER. Furthermore, in order to enhance its representation ability of audio noise, we design a knowledge distillation (KD) approach via mutual information estimation to distill the real noise information in audio embeddings to our language embedding. Experiments on various latest LLMs demonstrate our approach achieves a new breakthrough with up to 53.9% correction improvement in terms of word error rate while with limited training data. Analysis shows that our language-space noise embedding can well represent the noise conditions of source speech, under which off-the-shelf LLMs show strong ability of language-space denoising.

{{</citation>}}


### (29/98) Breaking the Curse of Multilinguality with Cross-lingual Expert Language Models (Terra Blevins et al., 2024)

{{<citation>}}

Terra Blevins, Tomasz Limisiewicz, Suchin Gururangan, Margaret Li, Hila Gonen, Noah A. Smith, Luke Zettlemoyer. (2024)  
**Breaking the Curse of Multilinguality with Cross-lingual Expert Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Multilingual, NLP  
[Paper Link](http://arxiv.org/abs/2401.10440v1)  

---


**ABSTRACT**  
Despite their popularity in non-English NLP, multilingual language models often underperform monolingual ones due to inter-language competition for model parameters. We propose Cross-lingual Expert Language Models (X-ELM), which mitigate this competition by independently training language models on subsets of the multilingual corpus. This process specializes X-ELMs to different languages while remaining effective as a multilingual ensemble. Our experiments show that when given the same compute budget, X-ELM outperforms jointly trained multilingual models across all considered languages and that these gains transfer to downstream tasks. X-ELM provides additional benefits over performance improvements: new experts can be iteratively added, adapting X-ELM to new languages without catastrophic forgetting. Furthermore, training is asynchronous, reducing the hardware requirements for multilingual training and democratizing multilingual modeling.

{{</citation>}}


## cs.HC (8)



### (30/98) Does Using ChatGPT Result in Human Cognitive Augmentation? (Ron Fulbright et al., 2024)

{{<citation>}}

Ron Fulbright, Miranda Morrison. (2024)  
**Does Using ChatGPT Result in Human Cognitive Augmentation?**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Augmentation, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.11042v1)  

---


**ABSTRACT**  
Human cognitive performance is enhanced by the use of tools. For example, a human can produce a much greater, and more accurate, volume of mathematical calculation in a unit of time using a calculator or a spreadsheet application on a computer. Such tools have taken over the burden of lower level cognitive grunt work but the human still serves the role of the expert performing higher level thinking and reasoning. Recently, however, unsupervised, deep, machine learning has produced cognitive systems able to outperform humans in several domains. When humans use these tools in a human cog ensemble, the cognitive ability of the human is augmented. In some cases, even non experts can achieve, and even exceed, the performance of experts in a particular domain, synthetic expertise. A new cognitive system, ChatGPT, has burst onto the scene during the past year. This paper investigates human cognitive augmentation due to using ChatGPT by presenting the results of two experiments comparing responses created using ChatGPT with results created not using ChatGPT. We find using ChatGPT does not always result in cognitive augmentation and does not yet replace human judgement, discernment, and evaluation in certain types of tasks. In fact, ChatGPT was observed to result in misleading users resulting in negative cognitive augmentation.

{{</citation>}}


### (31/98) DynaVis: Dynamically Synthesized UI Widgets for Visualization Editing (Priyan Vaithilingam et al., 2024)

{{<citation>}}

Priyan Vaithilingam, Elena L. Glassman, Jeevana Priya Inala, Chenglong Wang. (2024)  
**DynaVis: Dynamically Synthesized UI Widgets for Visualization Editing**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: NLI  
[Paper Link](http://arxiv.org/abs/2401.10880v1)  

---


**ABSTRACT**  
Users often rely on GUIs to edit and interact with visualizations - a daunting task due to the large space of editing options. As a result, users are either overwhelmed by a complex UI or constrained by a custom UI with a tailored, fixed subset of options with limited editing flexibility. Natural Language Interfaces (NLIs) are emerging as a feasible alternative for users to specify edits. However, NLIs forgo the advantages of traditional GUI: the ability to explore and repeat edits and see instant visual feedback.   We introduce DynaVis, which blends natural language and dynamically synthesized UI widgets. As the user describes an editing task in natural language, DynaVis performs the edit and synthesizes a persistent widget that the user can interact with to make further modifications. Study participants (n=24) preferred DynaVis over the NLI-only interface citing ease of further edits and editing confidence due to immediate visual feedback.

{{</citation>}}


### (32/98) An AI-Resilient Text Rendering Technique for Reading and Skimming Documents (Ziwei Gu et al., 2024)

{{<citation>}}

Ziwei Gu, Ian Arawjo, Kenneth Li, Jonathan K. Kummerfeld, Elena L. Glassman. (2024)  
**An AI-Resilient Text Rendering Technique for Reading and Skimming Documents**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Summarization  
[Paper Link](http://arxiv.org/abs/2401.10873v1)  

---


**ABSTRACT**  
Readers find text difficult to consume for many reasons. Summarization can address some of these difficulties, but introduce others, such as omitting, misrepresenting, or hallucinating information, which can be hard for a reader to notice. One approach to addressing this problem is to instead modify how the original text is rendered to make important information more salient. We introduce Grammar-Preserving Text Saliency Modulation (GP-TSM), a text rendering method with a novel means of identifying what to de-emphasize. Specifically, GP-TSM uses a recursive sentence compression method to identify successive levels of detail beyond the core meaning of a passage, which are de-emphasized by rendering words in successively lighter but still legible gray text. In a lab study (n=18), participants preferred GP-TSM over pre-existing word-level text rendering methods and were able to answer GRE reading comprehension questions more efficiently.

{{</citation>}}


### (33/98) Rambler: Supporting Writing With Speech via LLM-Assisted Gist Manipulation (Susan Lin et al., 2024)

{{<citation>}}

Susan Lin, Jeremy Warner, J. D. Zamfirescu-Pereira, Matthew G. Lee, Sauhard Jain, Michael Xuelin Huang, Piyawat Lertvittayakumjorn, Shanqing Cai, Shumin Zhai, Björn Hartmann, Can Liu. (2024)  
**Rambler: Supporting Writing With Speech via LLM-Assisted Gist Manipulation**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.10838v1)  

---


**ABSTRACT**  
Dictation enables efficient text input on mobile devices. However, writing with speech can produce disfluent, wordy, and incoherent text and thus requires heavy post-processing. This paper presents Rambler, an LLM-powered graphical user interface that supports gist-level manipulation of dictated text with two main sets of functions: gist extraction and macro revision. Gist extraction generates keywords and summaries as anchors to support the review and interaction with spoken text. LLM-assisted macro revisions allow users to respeak, split, merge and transform dictated text without specifying precise editing locations. Together they pave the way for interactive dictation and revision that help close gaps between spontaneous spoken words and well-structured writing. In a comparative study with 12 participants performing verbal composition tasks, Rambler outperformed the baseline of a speech-to-text editor + ChatGPT, as it better facilitates iterative revisions with enhanced user control over the content while supporting surprisingly diverse user strategies.

{{</citation>}}


### (34/98) Co-Pilot for Health: Personalized Algorithmic AI Nudging to Improve Health Outcomes (Jodi Chiam et al., 2024)

{{<citation>}}

Jodi Chiam, Aloysius Lim, Cheryl Nott, Nicholas Mark, Ankur Teredesai, Sunil Shinde. (2024)  
**Co-Pilot for Health: Personalized Algorithmic AI Nudging to Improve Health Outcomes**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-LG, cs.HC  
Keywords: AI, GNN  
[Paper Link](http://arxiv.org/abs/2401.10816v1)  

---


**ABSTRACT**  
The ability to shape health behaviors of large populations automatically, across wearable types and disease conditions at scale has tremendous potential to improve global health outcomes. We designed and implemented an AI driven platform for digital algorithmic nudging, enabled by a Graph-Neural Network (GNN) based Recommendation System, and granular health behavior data from wearable fitness devices. Here we describe the efficacy results of this platform with its capabilities of personalized and contextual nudging to $n=84,764$ individuals over a 12-week period in Singapore. We statistically validated that participants in the target group who received such AI optimized daily nudges increased daily physical activity like step count by 6.17% ($p = 3.09\times10^{-4}$) and weekly minutes of Moderate to Vigorous Physical Activity (MVPA) by 7.61% ($p = 1.16\times10^{-2}$), compared to matched participants in control group who did not receive any nudges. Further, such nudges were very well received, with a 13.1% of nudges sent being opened (open rate), and 11.7% of the opened nudges rated useful compared to 1.9% rated as not useful thereby demonstrating significant improvement in population level engagement metrics.

{{</citation>}}


### (35/98) Interactions with Prompt Problems: A New Way to Teach Programming with Large Language Models (James Prather et al., 2024)

{{<citation>}}

James Prather, Paul Denny, Juho Leinonen, David H. Smith IV, Brent N. Reeves, Stephen MacNeil, Brett A. Becker, Andrew Luxton-Reilly, Thezyrie Amarouche, Bailey Kimmel. (2024)  
**Interactions with Prompt Problems: A New Way to Teach Programming with Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.10759v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have upended decades of pedagogy in computing education. Students previously learned to code through \textit{writing} many small problems with less emphasis on code reading and comprehension. Recent research has shown that free code generation tools powered by LLMs can solve introductory programming problems presented in natural language with ease. In this paper, we propose a new way to teach programming with Prompt Problems. Students receive a problem visually, indicating how input should be transformed to output, and must translate that to a prompt for an LLM to decipher. The problem is considered correct when the code that is generated by the student prompt can pass all test cases. In this paper we present the design of this tool, discuss student interactions with it as they learn, and provide insights into this new class of programming problems as well as the design tools that integrate LLMs.

{{</citation>}}


### (36/98) Key to Kindness: Reducing Toxicity In Online Discourse Through Proactive Content Moderation in a Mobile Keyboard (Mark Warner et al., 2024)

{{<citation>}}

Mark Warner, Angelika Strohmayer, Matthew Higgs, Husnain Rafiq, Liying Yang, Lynne Coventry. (2024)  
**Key to Kindness: Reducing Toxicity In Online Discourse Through Proactive Content Moderation in a Mobile Keyboard**  

---
Primary Category: cs.HC  
Categories: ACM-class: H-5-2, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.10627v1)  

---


**ABSTRACT**  
Growing evidence shows that proactive content moderation supported by AI can help improve online discourse. However, we know little about designing these systems, how design impacts efficacy and user experience, and how people perceive proactive moderation across public and private platforms. We developed a mobile keyboard with built-in proactive content moderation which we tested (N=575) within a semi-functional simulation of a public and private communication platform. Where toxic content was detected, we used different interventions that embedded three design factors: timing, friction, and the presentation of the AI model output. We found moderation to be effective, regardless of the design. However, friction was a source of annoyance while prompts with no friction that occurred during typing were more effective. Follow-up interviews highlight the differences in how these systems are perceived across public and private platforms, and how they can offer more than moderation by acting as educational and communication support tools.

{{</citation>}}


### (37/98) AI Revolution on Chat Bot: Evidence from a Randomized Controlled Experiment (Sida Peng et al., 2024)

{{<citation>}}

Sida Peng, Wojciech Swiatek, Allen Gao, Paul Cullivan, Haoge Chang. (2024)  
**AI Revolution on Chat Bot: Evidence from a Randomized Controlled Experiment**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-IR, cs.HC  
Keywords: AI, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.10956v1)  

---


**ABSTRACT**  
In recent years, generative AI has undergone major advancements, demonstrating significant promise in augmenting human productivity. Notably, large language models (LLM), with ChatGPT-4 as an example, have drawn considerable attention. Numerous articles have examined the impact of LLM-based tools on human productivity in lab settings and designed tasks or in observational studies. Despite recent advances, field experiments applying LLM-based tools in realistic settings are limited. This paper presents the findings of a field randomized controlled trial assessing the effectiveness of LLM-based tools in providing unmonitored support services for information retrieval.

{{</citation>}}


## cs.CV (16)



### (38/98) Image Safeguarding: Reasoning with Conditional Vision Language Model and Obfuscating Unsafe Content Counterfactually (Mazal Bethany et al., 2024)

{{<citation>}}

Mazal Bethany, Brandon Wherry, Nishant Vishwamitra, Peyman Najafirad. (2024)  
**Image Safeguarding: Reasoning with Conditional Vision Language Model and Obfuscating Unsafe Content Counterfactually**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.11035v1)  

---


**ABSTRACT**  
Social media platforms are being increasingly used by malicious actors to share unsafe content, such as images depicting sexual activity, cyberbullying, and self-harm. Consequently, major platforms use artificial intelligence (AI) and human moderation to obfuscate such images to make them safer. Two critical needs for obfuscating unsafe images is that an accurate rationale for obfuscating image regions must be provided, and the sensitive regions should be obfuscated (\textit{e.g.} blurring) for users' safety. This process involves addressing two key problems: (1) the reason for obfuscating unsafe images demands the platform to provide an accurate rationale that must be grounded in unsafe image-specific attributes, and (2) the unsafe regions in the image must be minimally obfuscated while still depicting the safe regions. In this work, we address these key issues by first performing visual reasoning by designing a visual reasoning model (VLM) conditioned on pre-trained unsafe image classifiers to provide an accurate rationale grounded in unsafe image attributes, and then proposing a counterfactual explanation algorithm that minimally identifies and obfuscates unsafe regions for safe viewing, by first utilizing an unsafe image classifier attribution matrix to guide segmentation for a more optimal subregion segmentation followed by an informed greedy search to determine the minimum number of subregions required to modify the classifier's output based on attribution score. Extensive experiments on uncurated data from social networks emphasize the efficacy of our proposed method. We make our code available at: https://github.com/SecureAIAutonomyLab/ConditionalVLM

{{</citation>}}


### (39/98) Motion Consistency Loss for Monocular Visual Odometry with Attention-Based Deep Learning (André O. Françani et al., 2024)

{{<citation>}}

André O. Françani, Marcos R. O. A. Maximo. (2024)  
**Motion Consistency Loss for Monocular Visual Odometry with Attention-Based Deep Learning**  

---
Primary Category: cs.CV  
Categories: 68T45 68T07, cs-CV, cs-RO, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.10857v1)  

---


**ABSTRACT**  
Deep learning algorithms have driven expressive progress in many complex tasks. The loss function is a core component of deep learning techniques, guiding the learning process of neural networks. This paper contributes by introducing a consistency loss for visual odometry with deep learning-based approaches. The motion consistency loss explores repeated motions that appear in consecutive overlapped video clips. Experimental results show that our approach increased the performance of a model on the KITTI odometry benchmark.

{{</citation>}}


### (40/98) Understanding Video Transformers via Universal Concept Discovery (Matthew Kowal et al., 2024)

{{<citation>}}

Matthew Kowal, Achal Dave, Rares Ambrus, Adrien Gaidon, Konstantinos G. Derpanis, Pavel Tokmakov. (2024)  
**Understanding Video Transformers via Universal Concept Discovery**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.10831v1)  

---


**ABSTRACT**  
This paper studies the problem of concept-based interpretability of transformer representations for videos. Concretely, we seek to explain the decision-making process of video transformers based on high-level, spatiotemporal concepts that are automatically discovered. Prior research on concept-based interpretability has concentrated solely on image-level tasks. Comparatively, video models deal with the added temporal dimension, increasing complexity and posing challenges in identifying dynamic concepts over time. In this work, we systematically address these challenges by introducing the first Video Transformer Concept Discovery (VTCD) algorithm. To this end, we propose an efficient approach for unsupervised identification of units of video transformer representations - concepts, and ranking their importance to the output of a model. The resulting concepts are highly interpretable, revealing spatio-temporal reasoning mechanisms and object-centric representations in unstructured video models. Performing this analysis jointly over a diverse set of supervised and self-supervised representations, we discover that some of these mechanism are universal in video transformers. Finally, we demonstrate that VTCDcan be used to improve model performance for fine-grained tasks.

{{</citation>}}


### (41/98) Measuring the Impact of Scene Level Objects on Object Detection: Towards Quantitative Explanations of Detection Decisions (Lynn Vonder Haar et al., 2024)

{{<citation>}}

Lynn Vonder Haar, Timothy Elvira, Luke Newcomb, Omar Ochoa. (2024)  
**Measuring the Impact of Scene Level Objects on Object Detection: Towards Quantitative Explanations of Detection Decisions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.10790v1)  

---


**ABSTRACT**  
Although accuracy and other common metrics can provide a useful window into the performance of an object detection model, they lack a deeper view of the model's decision process. Regardless of the quality of the training data and process, the features that an object detection model learns cannot be guaranteed. A model may learn a relationship between certain background context, i.e., scene level objects, and the presence of the labeled classes. Furthermore, standard performance verification and metrics would not identify this phenomenon. This paper presents a new black box explainability method for additional verification of object detection models by finding the impact of scene level objects on the identification of the objects within the image. By comparing the accuracies of a model on test data with and without certain scene level objects, the contributions of these objects to the model's performance becomes clearer. The experiment presented here will assess the impact of buildings and people in image context on the detection of emergency road vehicles by a fine-tuned YOLOv8 model. A large increase in accuracy in the presence of a scene level object will indicate the model's reliance on that object to make its detections. The results of this research lead to providing a quantitative explanation of the object detection model's decision process, enabling a deeper understanding of the model's performance.

{{</citation>}}


### (42/98) Removal and Selection: Improving RGB-Infrared Object Detection via Coarse-to-Fine Fusion (Tianyi Zhao et al., 2024)

{{<citation>}}

Tianyi Zhao, Maoxun Yuan, Xingxing Wei. (2024)  
**Removal and Selection: Improving RGB-Infrared Object Detection via Coarse-to-Fine Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.10731v1)  

---


**ABSTRACT**  
Object detection in visible (RGB) and infrared (IR) images has been widely applied in recent years. Leveraging the complementary characteristics of RGB and IR images, the object detector provides reliable and robust object localization from day to night. Existing fusion strategies directly inject RGB and IR images into convolution neural networks, leading to inferior detection performance. Since the RGB and IR features have modality-specific noise, these strategies will worsen the fused features along with the propagation. Inspired by the mechanism of human brain processing multimodal information, this work introduces a new coarse-to-fine perspective to purify and fuse two modality features. Specifically, following this perspective, we design a Redundant Spectrum Removal module to coarsely remove interfering information within each modality and a Dynamic Feature Selection module to finely select the desired features for feature fusion. To verify the effectiveness of the coarse-to-fine fusion strategy, we construct a new object detector called Removal and Selection Detector (RSDet). Extensive experiments on three RGB-IR object detection datasets verify the superior performance of our method.

{{</citation>}}


### (43/98) Q&A Prompts: Discovering Rich Visual Clues through Mining Question-Answer Prompts for VQA requiring Diverse World Knowledge (Haibi Wang et al., 2024)

{{<citation>}}

Haibi Wang, Weifeng Ge. (2024)  
**Q&A Prompts: Discovering Rich Visual Clues through Mining Question-Answer Prompts for VQA requiring Diverse World Knowledge**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2401.10712v1)  

---


**ABSTRACT**  
With the breakthrough of multi-modal large language models, answering complex visual questions that demand advanced reasoning abilities and world knowledge has become a much more important testbed for developing AI models than ever. However, equipping AI models with robust cross-modality reasoning ability remains challenging since the cognition scheme of humans has not been understood systematically. In this paper, we believe that if we can collect visual clues in the given image as much as possible, we will recognize the image more accurately, understand the question better, recall relevant knowledge more easily, and finally reason out the answer. We discover these rich visual clues by mining question-answer pairs in images and sending them into multi-modal large language models as prompts. We call the proposed method Q&A Prompts. Specifically, we first use the image-answer pairs and the corresponding questions in the training set as inputs and outputs to train a visual question generation model. Then, we use an image tagging model to identify various instances and send packaged image-tag pairs into the visual question generation model to generate relevant questions with the extracted image tags as answers. Finally, we encode these generated question-answer pairs as prompts with a visual-aware prompting module and send them into pre-trained multi-modal large language models to reason out the final answers. Experimental results show that, compared with state-of-the-art methods, our Q&A Prompts achieves substantial improvements on the challenging visual question answering datasets requiring reasoning over diverse world knowledge, such as OK-VQA and A-OKVQA.

{{</citation>}}


### (44/98) Weakly Supervised Gaussian Contrastive Grounding with Large Multimodal Models for Video Question Answering (Haibo Wang et al., 2024)

{{<citation>}}

Haibo Wang, Chenghang Lai, Yixuan Sun, Weifeng Ge. (2024)  
**Weakly Supervised Gaussian Contrastive Grounding with Large Multimodal Models for Video Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.10711v1)  

---


**ABSTRACT**  
Video Question Answering (VideoQA) aims to answer natural language questions based on the information observed in videos. Despite the recent success of Large Multimodal Models (LMMs) in image-language understanding and reasoning, they deal with VideoQA insufficiently by simply taking uniformly sampled frames as visual inputs, which ignores question-relevant visual clues. Moreover, there are no human annotations for question-critical timestamps in existing VideoQA datasets. In light of this, we propose a novel weakly supervised framework to enforce the LMMs to reason out the answers with question-critical moments as visual inputs. Specifically, we fuse the question and answer pairs as event descriptions to find multiple keyframes as target moments, which will be pseudo-labels. With these pseudo-labels as additionally weak supervision, we devise a lightweight Gaussian-based Contrastive Grounding (GCG) module. GCG learns multiple Gaussian functions to characterize the temporal structure of the video, and sample question-critical frames as positive moments to be the visual inputs of LMMs. Extensive experiments on several VideoQA benchmarks verify the effectiveness of our framework, and we achieve substantial improvements compared to previous state-of-the-art methods.

{{</citation>}}


### (45/98) BadODD: Bangladeshi Autonomous Driving Object Detection Dataset (Mirza Nihal Baig et al., 2024)

{{<citation>}}

Mirza Nihal Baig, Rony Hajong, Mahdi Murshed Patwary, Mohammad Shahidur Rahman, Husne Ara Chowdhury. (2024)  
**BadODD: Bangladeshi Autonomous Driving Object Detection Dataset**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.10659v1)  

---


**ABSTRACT**  
We propose a comprehensive dataset for object detection in diverse driving environments across 9 districts in Bangladesh. The dataset, collected exclusively from smartphone cameras, provided a realistic representation of real-world scenarios, including day and night conditions. Most existing datasets lack suitable classes for autonomous navigation on Bangladeshi roads, making it challenging for researchers to develop models that can handle the intricacies of road scenarios. To address this issue, the authors proposed a new set of classes based on characteristics rather than local vehicle names. The dataset aims to encourage the development of models that can handle the unique challenges of Bangladeshi road scenarios for the effective deployment of autonomous vehicles. The dataset did not consist of any online images to simulate real-world conditions faced by autonomous vehicles. The classification of vehicles is challenging because of the diverse range of vehicles on Bangladeshi roads, including those not found elsewhere in the world. The proposed classification system is scalable and can accommodate future vehicles, making it a valuable resource for researchers in the autonomous vehicle sector.

{{</citation>}}


### (46/98) One Step Learning, One Step Review (Xiaolong Huang et al., 2024)

{{<citation>}}

Xiaolong Huang, Qiankun Li, Xueran Li, Xuesong Gao. (2024)  
**One Step Learning, One Step Review**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.10962v1)  

---


**ABSTRACT**  
Visual fine-tuning has garnered significant attention with the rise of pre-trained vision models. The current prevailing method, full fine-tuning, suffers from the issue of knowledge forgetting as it focuses solely on fitting the downstream training set. In this paper, we propose a novel weight rollback-based fine-tuning method called OLOR (One step Learning, One step Review). OLOR combines fine-tuning with optimizers, incorporating a weight rollback term into the weight update term at each step. This ensures consistency in the weight range of upstream and downstream models, effectively mitigating knowledge forgetting and enhancing fine-tuning performance. In addition, a layer-wise penalty is presented to employ penalty decay and the diversified decay rate to adjust the weight rollback levels of layers for adapting varying downstream tasks. Through extensive experiments on various tasks such as image classification, object detection, semantic segmentation, and instance segmentation, we demonstrate the general applicability and state-of-the-art performance of our proposed OLOR. Code is available at https://github.com/rainbow-xiao/OLOR-AAAI-2024.

{{</citation>}}


### (47/98) A comprehensive study on fidelity metrics for XAI (Miquel Miró-Nicolau et al., 2024)

{{<citation>}}

Miquel Miró-Nicolau, Antoni Jaume-i-Capó, Gabriel Moyà-Alcover. (2024)  
**A comprehensive study on fidelity metrics for XAI**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.10640v1)  

---


**ABSTRACT**  
The use of eXplainable Artificial Intelligence (XAI) systems has introduced a set of challenges that need resolution. Herein, we focus on how to correctly select an XAI method, an open questions within the field. The inherent difficulty of this task is due to the lack of a ground truth. Several authors have proposed metrics to approximate the fidelity of different XAI methods. These metrics lack verification and have concerning disagreements. In this study, we proposed a novel methodology to verify fidelity metrics, using a well-known transparent model, namely a decision tree. This model allowed us to obtain explanations with perfect fidelity. Our proposal constitutes the first objective benchmark for these metrics, facilitating a comparison of existing proposals, and surpassing existing methods. We applied our benchmark to assess the existing fidelity metrics in two different experiments, each using public datasets comprising 52,000 images. The images from these datasets had a size a 128 by 128 pixels and were synthetic data that simplified the training process. All metric values, indicated a lack of fidelity, with the best one showing a 30 \% deviation from the expected values for perfect explanation. Our experimentation led us to conclude that the current fidelity metrics are not reliable enough to be used in real scenarios. From this finding, we deemed it necessary to development new metrics, to avoid the detected problems, and we recommend the usage of our proposal as a benchmark within the scientific community to address these limitations.

{{</citation>}}


### (48/98) M2ORT: Many-To-One Regression Transformer for Spatial Transcriptomics Prediction from Histopathology Images (Hongyi Wang et al., 2024)

{{<citation>}}

Hongyi Wang, Xiuju Du, Jing Liu, Shuyi Ouyang, Yen-Wei Chen, Lanfen Lin. (2024)  
**M2ORT: Many-To-One Regression Transformer for Spatial Transcriptomics Prediction from Histopathology Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.10608v1)  

---


**ABSTRACT**  
The advancement of Spatial Transcriptomics (ST) has facilitated the spatially-aware profiling of gene expressions based on histopathology images. Although ST data offers valuable insights into the micro-environment of tumors, its acquisition cost remains expensive. Therefore, directly predicting the ST expressions from digital pathology images is desired. Current methods usually adopt existing regression backbones for this task, which ignore the inherent multi-scale hierarchical data structure of digital pathology images. To address this limit, we propose M2ORT, a many-to-one regression Transformer that can accommodate the hierarchical structure of the pathology images through a decoupled multi-scale feature extractor. Different from traditional models that are trained with one-to-one image-label pairs, M2ORT accepts multiple pathology images of different magnifications at a time to jointly predict the gene expressions at their corresponding common ST spot, aiming at learning a many-to-one relationship through training. We have tested M2ORT on three public ST datasets and the experimental results show that M2ORT can achieve state-of-the-art performance with fewer parameters and floating-point operations (FLOPs). The code is available at: https://github.com/Dootmaan/M2ORT/.

{{</citation>}}


### (49/98) Dream360: Diverse and Immersive Outdoor Virtual Scene Creation via Transformer-Based 360 Image Outpainting (Hao Ai et al., 2024)

{{<citation>}}

Hao Ai, Zidong Cao, Haonan Lu, Chen Chen, Jian Ma, Pengyuan Zhou, Tae-Kyun Kim, Pan Hui, Lin Wang. (2024)  
**Dream360: Diverse and Immersive Outdoor Virtual Scene Creation via Transformer-Based 360 Image Outpainting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-HC, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.10564v1)  

---


**ABSTRACT**  
360 images, with a field-of-view (FoV) of 180x360, provide immersive and realistic environments for emerging virtual reality (VR) applications, such as virtual tourism, where users desire to create diverse panoramic scenes from a narrow FoV photo they take from a viewpoint via portable devices. It thus brings us to a technical challenge: `How to allow the users to freely create diverse and immersive virtual scenes from a narrow FoV image with a specified viewport?' To this end, we propose a transformer-based 360 image outpainting framework called Dream360, which can generate diverse, high-fidelity, and high-resolution panoramas from user-selected viewports, considering the spherical properties of 360 images. Compared with existing methods, e.g., [3], which primarily focus on inputs with rectangular masks and central locations while overlooking the spherical property of 360 images, our Dream360 offers higher outpainting flexibility and fidelity based on the spherical representation. Dream360 comprises two key learning stages: (I) codebook-based panorama outpainting via Spherical-VQGAN (S-VQGAN), and (II) frequency-aware refinement with a novel frequency-aware consistency loss. Specifically, S-VQGAN learns a sphere-specific codebook from spherical harmonic (SH) values, providing a better representation of spherical data distribution for scene modeling. The frequency-aware refinement matches the resolution and further improves the semantic consistency and visual fidelity of the generated results. Our Dream360 achieves significantly lower Frechet Inception Distance (FID) scores and better visual fidelity than existing methods. We also conducted a user study involving 15 participants to interactively evaluate the quality of the generated results in VR, demonstrating the flexibility and superiority of our Dream360 framework.

{{</citation>}}


### (50/98) Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences (Xiyao Wang et al., 2024)

{{<citation>}}

Xiyao Wang, Yuhang Zhou, Xiaoyu Liu, Hongjin Lu, Yuancheng Xu, Feihong He, Jaehong Yoon, Taixi Lu, Gedas Bertasius, Mohit Bansal, Huaxiu Yao, Furong Huang. (2024)  
**Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.10529v1)  

---


**ABSTRACT**  
Multimodal Large Language Models (MLLMs) have demonstrated proficiency in handling a variety of visual-language tasks. However, current MLLM benchmarks are predominantly designed to evaluate reasoning based on static information about a single image, and the ability of modern MLLMs to extrapolate from image sequences, which is essential for understanding our ever-changing world, has been less investigated. To address this challenge, this paper introduces Mementos, a new benchmark designed to assess MLLMs' sequential image reasoning abilities. Mementos features 4,761 diverse image sequences with varying lengths. We also employ a GPT-4 assisted method to evaluate MLLM reasoning performance. Through a careful evaluation of nine recent MLLMs on Mementos, including GPT-4V and Gemini, we find that they struggle to accurately describe dynamic information about given image sequences, often leading to hallucinations/misrepresentations of objects and their corresponding behaviors. Our quantitative analysis and case studies identify three key factors impacting MLLMs' sequential image reasoning: the correlation between object and behavioral hallucinations, the influence of cooccurring behaviors, and the compounding impact of behavioral hallucinations. Our dataset is available at https://github.com/umd-huang-lab/Mementos.

{{</citation>}}


### (51/98) Exploring Color Invariance through Image-Level Ensemble Learning (Yunpeng Gong et al., 2024)

{{<citation>}}

Yunpeng Gong, Jiaquan Li, Lifei Chen, Min Jiang. (2024)  
**Exploring Color Invariance through Image-Level Ensemble Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2401.10512v1)  

---


**ABSTRACT**  
In the field of computer vision, the persistent presence of color bias, resulting from fluctuations in real-world lighting and camera conditions, presents a substantial challenge to the robustness of models. This issue is particularly pronounced in complex wide-area surveillance scenarios, such as person re-identification and industrial dust segmentation, where models often experience a decline in performance due to overfitting on color information during training, given the presence of environmental variations. Consequently, there is a need to effectively adapt models to cope with the complexities of camera conditions. To address this challenge, this study introduces a learning strategy named Random Color Erasing, which draws inspiration from ensemble learning. This strategy selectively erases partial or complete color information in the training data without disrupting the original image structure, thereby achieving a balanced weighting of color features and other features within the neural network. This approach mitigates the risk of overfitting and enhances the model's ability to handle color variation, thereby improving its overall robustness. The approach we propose serves as an ensemble learning strategy, characterized by robust interpretability. A comprehensive analysis of this methodology is presented in this paper. Across various tasks such as person re-identification and semantic segmentation, our approach consistently improves strong baseline methods. Notably, in comparison to existing methods that prioritize color robustness, our strategy significantly enhances performance in cross-domain scenarios. The code available at \url{https://github.com/layumi/Person\_reID\_baseline\_pytorch/blob/master/random\_erasing.py} or \url{https://github.com/finger-monkey/Data-Augmentation}.

{{</citation>}}


### (52/98) GMC-IQA: Exploiting Global-correlation and Mean-opinion Consistency for No-reference Image Quality Assessment (Zewen Chen et al., 2024)

{{<citation>}}

Zewen Chen, Juan Wang, Bing Li, Chunfeng Yuan, Weiming Hu, Junxian Liu, Peng Li, Yan Wang, Youqun Zhang, Congxuan Zhang. (2024)  
**GMC-IQA: Exploiting Global-correlation and Mean-opinion Consistency for No-reference Image Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.10511v1)  

---


**ABSTRACT**  
Due to the subjective nature of image quality assessment (IQA), assessing which image has better quality among a sequence of images is more reliable than assigning an absolute mean opinion score for an image. Thus, IQA models are evaluated by global correlation consistency (GCC) metrics like PLCC and SROCC, rather than mean opinion consistency (MOC) metrics like MAE and MSE. However, most existing methods adopt MOC metrics to define their loss functions, due to the infeasible computation of GCC metrics during training. In this work, we construct a novel loss function and network to exploit Global-correlation and Mean-opinion Consistency, forming a GMC-IQA framework. Specifically, we propose a novel GCC loss by defining a pairwise preference-based rank estimation to solve the non-differentiable problem of SROCC and introducing a queue mechanism to reserve previous data to approximate the global results of the whole data. Moreover, we propose a mean-opinion network, which integrates diverse opinion features to alleviate the randomness of weight learning and enhance the model robustness. Experiments indicate that our method outperforms SOTA methods on multiple authentic datasets with higher accuracy and generalization. We also adapt the proposed loss to various networks, which brings better performance and more stable training.

{{</citation>}}


### (53/98) CBVS: A Large-Scale Chinese Image-Text Benchmark for Real-World Short Video Search Scenarios (Xiangshuo Qiao et al., 2024)

{{<citation>}}

Xiangshuo Qiao, Xianxin Li, Xiaozhe Qu, Jie Zhang, Yang Liu, Yu Luo, Cihang Jin, Jin Ma. (2024)  
**CBVS: A Large-Scale Chinese Image-Text Benchmark for Real-World Short Video Search Scenarios**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.10475v1)  

---


**ABSTRACT**  
Vision-Language Models pre-trained on large-scale image-text datasets have shown superior performance in downstream tasks such as image retrieval. Most of the images for pre-training are presented in the form of open domain common-sense visual elements. Differently, video covers in short video search scenarios are presented as user-originated contents that provide important visual summaries of videos. In addition, a portion of the video covers come with manually designed cover texts that provide semantic complements. In order to fill in the gaps in short video cover data, we establish the first large-scale cover-text benchmark for Chinese short video search scenarios. Specifically, we release two large-scale datasets CBVS-5M/10M to provide short video covers, and the manual fine-labeling dataset CBVS-20K to provide real user queries, which serves as an image-text benchmark test in the Chinese short video search field. To integrate the semantics of cover text in the case of modality missing, we propose UniCLIP where cover texts play a guiding role during training, however are not relied upon by inference. Extensive evaluation on CBVS-20K demonstrates the excellent performance of our proposal. UniCLIP has been deployed to Tencent's online video search systems with hundreds of millions of visits and achieved significant gains. The complete dataset, code and checkpoints will be available upon release.

{{</citation>}}


## cs.CY (1)



### (54/98) PressProtect: Helping Journalists Navigate Social Media in the Face of Online Harassment (Catherine Han et al., 2024)

{{<citation>}}

Catherine Han, Anne Li, Deepak Kumar, Zakir Durumeric. (2024)  
**PressProtect: Helping Journalists Navigate Social Media in the Face of Online Harassment**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-HC, cs.CY  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2401.11032v1)  

---


**ABSTRACT**  
Social media has become a critical tool for journalists to disseminate their work, engage with their audience, and connect with sources. Unfortunately, journalists also regularly endure significant online harassment on social media platforms, ranging from personal attacks to doxxing to threats of physical harm. In this paper, we seek to understand how we can make social media usable for journalists who face constant digital harassment. To begin, we conduct a set of need-finding interviews to understand where existing platform tools and newsroom resources fall short in adequately protecting journalists. We map journalists' unmet needs to concrete design goals, which we use to build PressProtect, an interface that provides journalists greater agency over engaging with readers on Twitter/X. Through user testing with eight journalists, we evaluate PressProtect and find that participants felt it effectively protected them against harassment and could also generalize to serve other visible and vulnerable groups. We conclude with a discussion of our findings and recommendations for social platforms hoping to build defensive defaults for journalists facing online harassment.

{{</citation>}}


## cs.CR (7)



### (55/98) Exploring Highly Quantised Neural Networks for Intrusion Detection in Automotive CAN (Shashwat Khandelwal et al., 2024)

{{<citation>}}

Shashwat Khandelwal, Shreejith Shanker. (2024)  
**Exploring Highly Quantised Neural Networks for Intrusion Detection in Automotive CAN**  

---
Primary Category: cs.CR  
Categories: cs-AR, cs-CR, cs-LG, cs-SY, cs.CR, eess-SY  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2401.11030v1)  

---


**ABSTRACT**  
Vehicles today comprise intelligent systems like connected autonomous driving and advanced driving assistance systems (ADAS) to enhance the driving experience, which is enabled through increased connectivity to infrastructure and fusion of information from different sensing modes. However, the rising connectivity coupled with the legacy network architecture within vehicles can be exploited for launching active and passive attacks on critical vehicle systems and directly affecting the safety of passengers. Machine learning-based intrusion detection models have been shown to successfully detect multiple targeted attack vectors in recent literature, whose deployments are enabled through quantised neural networks targeting low-power platforms. Multiple models are often required to simultaneously detect multiple attack vectors, increasing the area, (resource) cost, and energy consumption. In this paper, we present a case for utilising custom-quantised MLP's (CQMLP) as a multi-class classification model, capable of detecting multiple attacks from the benign flow of controller area network (CAN) messages. The specific quantisation and neural architecture are determined through a joint design space exploration, resulting in our choice of the 2-bit precision and the n-layer MLP. Our 2-bit version is trained using Brevitas and optimised as a dataflow hardware model through the FINN toolflow from AMD/Xilinx, targeting an XCZU7EV device. We show that the 2-bit CQMLP model, when integrated as the IDS, can detect malicious attack messages (DoS, fuzzing, and spoofing attack) with a very high accuracy of 99.9%, on par with the state-of-the-art methods in the literature. Furthermore, the dataflow model can perform line rate detection at a latency of 0.11 ms from message reception while consuming 0.23 mJ/inference, making it ideally suited for integration with an ECU in critical CAN networks.

{{</citation>}}


### (56/98) A Survey and Comparative Analysis of Security Properties of CAN Authentication Protocols (Alessandro Lotto et al., 2024)

{{<citation>}}

Alessandro Lotto, Francesco Marchiori, Alessandro Brighente, Mauro Conti. (2024)  
**A Survey and Comparative Analysis of Security Properties of CAN Authentication Protocols**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.10736v1)  

---


**ABSTRACT**  
The large number of Electronic Control Units (ECUs) mounted on modern cars and their expansive communication capabilities create a substantial attack surface for potential exploitation. Despite the evolution of automotive technology, the continued use of the originally insecure Controller Area Network (CAN) bus leaves in-vehicle communications inherently non-secure. In response to the absence of standardized authentication protocols within the automotive domain, researchers propose diverse solutions, each with unique strengths and vulnerabilities. However, the continuous influx of new protocols and potential oversights in meeting security requirements and essential operational features further complicate the implementability of these protocols. This paper comprehensively reviews and compares the 15 most prominent authentication protocols for the CAN bus. Our analysis emphasizes their strengths and weaknesses, evaluating their alignment with critical security requirements for automotive authentication. Additionally, we evaluate protocols based on essential operational criteria that contribute to ease of implementation in predefined infrastructures, enhancing overall reliability and reducing the probability of successful attacks. Our study reveals a prevalent focus on defending against external attackers in existing protocols, exposing vulnerabilities to internal threats. Notably, authentication protocols employing hash chains, Mixed Message Authentication Codes, and asymmetric encryption techniques emerge as the most effective approaches. Through our comparative study, we classify the considered protocols based on their security attributes and suitability for implementation, providing valuable insights for future developments in the field.

{{</citation>}}


### (57/98) Real-Time Zero-Day Intrusion Detection System for Automotive Controller Area Network on FPGAs (Shashwat Khandelwal et al., 2024)

{{<citation>}}

Shashwat Khandelwal, Shreejith Shanker. (2024)  
**Real-Time Zero-Day Intrusion Detection System for Automotive Controller Area Network on FPGAs**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs-SY, cs.CR, eess-SY  
Keywords: AI, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2401.10724v1)  

---


**ABSTRACT**  
Increasing automation in vehicles enabled by increased connectivity to the outside world has exposed vulnerabilities in previously siloed automotive networks like controller area networks (CAN). Attributes of CAN such as broadcast-based communication among electronic control units (ECUs) that lowered deployment costs are now being exploited to carry out active injection attacks like denial of service (DoS), fuzzing, and spoofing attacks. Research literature has proposed multiple supervised machine learning models deployed as Intrusion detection systems (IDSs) to detect such malicious activity; however, these are largely limited to identifying previously known attack vectors. With the ever-increasing complexity of active injection attacks, detecting zero-day (novel) attacks in these networks in real-time (to prevent propagation) becomes a problem of particular interest. This paper presents an unsupervised-learning-based convolutional autoencoder architecture for detecting zero-day attacks, which is trained only on benign (attack-free) CAN messages. We quantise the model using Vitis-AI tools from AMD/Xilinx targeting a resource-constrained Zynq Ultrascale platform as our IDS-ECU system for integration. The proposed model successfully achieves equal or higher classification accuracy (> 99.5%) on unseen DoS, fuzzing, and spoofing attacks from a publicly available attack dataset when compared to the state-of-the-art unsupervised learning-based IDSs. Additionally, by cleverly overlapping IDS operation on a window of CAN messages with the reception, the model is able to meet line-rate detection (0.43 ms per window) of high-speed CAN, which when coupled with the low energy consumption per inference, makes this architecture ideally suited for detecting zero-day attacks on critical CAN networks.

{{</citation>}}


### (58/98) Explainable and Transferable Adversarial Attack for ML-Based Network Intrusion Detectors (Hangsheng Zhang et al., 2024)

{{<citation>}}

Hangsheng Zhang, Dongqi Han, Yinlong Liu, Zhiliang Wang, Jiyan Sun, Shangyuan Zhuang, Jiqiang Liu, Jinsong Dong. (2024)  
**Explainable and Transferable Adversarial Attack for ML-Based Network Intrusion Detectors**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2401.10691v1)  

---


**ABSTRACT**  
espite being widely used in network intrusion detection systems (NIDSs), machine learning (ML) has proven to be highly vulnerable to adversarial attacks. White-box and black-box adversarial attacks of NIDS have been explored in several studies. However, white-box attacks unrealistically assume that the attackers have full knowledge of the target NIDSs. Meanwhile, existing black-box attacks can not achieve high attack success rate due to the weak adversarial transferability between models (e.g., neural networks and tree models). Additionally, neither of them explains why adversarial examples exist and why they can transfer across models. To address these challenges, this paper introduces ETA, an Explainable Transfer-based Black-Box Adversarial Attack framework. ETA aims to achieve two primary objectives: 1) create transferable adversarial examples applicable to various ML models and 2) provide insights into the existence of adversarial examples and their transferability within NIDSs. Specifically, we first provide a general transfer-based adversarial attack method applicable across the entire ML space. Following that, we exploit a unique insight based on cooperative game theory and perturbation interpretations to explain adversarial examples and adversarial transferability. On this basis, we propose an Important-Sensitive Feature Selection (ISFS) method to guide the search for adversarial examples, achieving stronger transferability and ensuring traffic-space constraints.

{{</citation>}}


### (59/98) A Lightweight Multi-Attack CAN Intrusion Detection System on Hybrid FPGAs (Shashwat Khandelwal et al., 2024)

{{<citation>}}

Shashwat Khandelwal, Shreejith Shanker. (2024)  
**A Lightweight Multi-Attack CAN Intrusion Detection System on Hybrid FPGAs**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs-SY, cs.CR, eess-SY  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2401.10689v1)  

---


**ABSTRACT**  
Rising connectivity in vehicles is enabling new capabilities like connected autonomous driving and advanced driver assistance systems (ADAS) for improving the safety and reliability of next-generation vehicles. This increased access to in-vehicle functions compromises critical capabilities that use legacy invehicle networks like Controller Area Network (CAN), which has no inherent security or authentication mechanism. Intrusion detection and mitigation approaches, particularly using machine learning models, have shown promising results in detecting multiple attack vectors in CAN through their ability to generalise to new vectors. However, most deployments require dedicated computing units like GPUs to perform line-rate detection, consuming much higher power. In this paper, we present a lightweight multi-attack quantised machine learning model that is deployed using Xilinx's Deep Learning Processing Unit IP on a Zynq Ultrascale+ (XCZU3EG) FPGA, which is trained and validated using the public CAN Intrusion Detection dataset. The quantised model detects denial of service and fuzzing attacks with an accuracy of above 99 % and a false positive rate of 0.07%, which are comparable to the state-of-the-art techniques in the literature. The Intrusion Detection System (IDS) execution consumes just 2.0 W with software tasks running on the ECU and achieves a 25 % reduction in per-message processing latency over the state-of-the-art implementations. This deployment allows the ECU function to coexist with the IDS with minimal changes to the tasks, making it ideal for real-time IDS in in-vehicle systems.

{{</citation>}}


### (60/98) Deep Learning-based Embedded Intrusion Detection System for Automotive CAN (Shashwat Khandelwal et al., 2024)

{{<citation>}}

Shashwat Khandelwal, Eashan Wadhwa, Shreejith Shanker. (2024)  
**Deep Learning-based Embedded Intrusion Detection System for Automotive CAN**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2401.10674v1)  

---


**ABSTRACT**  
Rising complexity of in-vehicle electronics is enabling new capabilities like autonomous driving and active safety. However, rising automation also increases risk of security threats which is compounded by lack of in-built security measures in legacy networks like CAN, allowing attackers to observe, tamper and modify information shared over such broadcast networks. Various intrusion detection approaches have been proposed to detect and tackle such threats, with machine learning models proving highly effective. However, deploying machine learning models will require high processing power through high-end processors or GPUs to perform them close to line rate. In this paper, we propose a hybrid FPGA-based ECU approach that can transparently integrate IDS functionality through a dedicated off-the-shelf hardware accelerator that implements a deep-CNN intrusion detection model. Our results show that the proposed approach provides an average accuracy of over 99% across multiple attack datasets with 0.64% false detection rates while consuming 94% less energy and achieving 51.8% reduction in per-message processing latency when compared to IDS implementations on GPUs.

{{</citation>}}


### (61/98) PuriDefense: Randomized Local Implicit Adversarial Purification for Defending Black-box Query-based Attacks (Ping Guo et al., 2024)

{{<citation>}}

Ping Guo, Zhiyuan Yang, Xi Lin, Qingchuan Zhao, Qingfu Zhang. (2024)  
**PuriDefense: Randomized Local Implicit Adversarial Purification for Defending Black-box Query-based Attacks**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-CV, cs-LG, cs.CR  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.10586v1)  

---


**ABSTRACT**  
Black-box query-based attacks constitute significant threats to Machine Learning as a Service (MLaaS) systems since they can generate adversarial examples without accessing the target model's architecture and parameters. Traditional defense mechanisms, such as adversarial training, gradient masking, and input transformations, either impose substantial computational costs or compromise the test accuracy of non-adversarial inputs. To address these challenges, we propose an efficient defense mechanism, PuriDefense, that employs random patch-wise purifications with an ensemble of lightweight purification models at a low level of inference cost. These models leverage the local implicit function and rebuild the natural image manifold. Our theoretical analysis suggests that this approach slows down the convergence of query-based attacks by incorporating randomness into purifications. Extensive experiments on CIFAR-10 and ImageNet validate the effectiveness of our proposed purifier-based defense mechanism, demonstrating significant improvements in robustness against query-based attacks.

{{</citation>}}


## cs.SE (4)



### (62/98) Custom Developer GPT for Ethical AI Solutions (Lauren Olson, 2024)

{{<citation>}}

Lauren Olson. (2024)  
**Custom Developer GPT for Ethical AI Solutions**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11013v1)  

---


**ABSTRACT**  
The main goal of this project is to create a new software artefact: a custom Generative Pre-trained Transformer (GPT) for developers to discuss and solve ethical issues through AI engineering. This conversational agent will provide developers with practical application on (1) how to comply with legal frameworks which regard AI systems (like the EU AI Act~\cite{aiact} and GDPR~\cite{gdpr}) and (2) present alternate ethical perspectives to allow developers to understand and incorporate alternate moral positions. In this paper, we provide motivation for the need of such an agent, detail our idea and demonstrate a use case. The use of such a tool can allow practitioners to engineer AI solutions which meet legal requirements and satisfy diverse ethical perspectives.

{{</citation>}}


### (63/98) Emotion Classification In Software Engineering Texts: A Comparative Analysis of Pre-trained Transformers Language Models (Mia Mohammad Imran, 2024)

{{<citation>}}

Mia Mohammad Imran. (2024)  
**Emotion Classification In Software Engineering Texts: A Comparative Analysis of Pre-trained Transformers Language Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BERT, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.10845v2)  

---


**ABSTRACT**  
Emotion recognition in software engineering texts is critical for understanding developer expressions and improving collaboration. This paper presents a comparative analysis of state-of-the-art Pre-trained Language Models (PTMs) for fine-grained emotion classification on two benchmark datasets from GitHub and Stack Overflow. We evaluate six transformer models - BERT, RoBERTa, ALBERT, DeBERTa, CodeBERT and GraphCodeBERT against the current best-performing tool SEntiMoji. Our analysis reveals consistent improvements ranging from 1.17\% to 16.79\% in terms of macro-averaged and micro-averaged F1 scores, with general domain models outperforming specialized ones. To further enhance PTMs, we incorporate polarity features in attention layer during training, demonstrating additional average gains of 1.0\% to 10.23\% over baseline PTMs approaches. Our work provides strong evidence for the advancements afforded by PTMs in recognizing nuanced emotions like Anger, Love, Fear, Joy, Sadness, and Surprise in software engineering contexts. Through comprehensive benchmarking and error analysis, we also outline scope for improvements to address contextual gaps.

{{</citation>}}


### (64/98) In-IDE Human-AI Experience in the Era of Large Language Models; A Literature Review (Agnia Sergeyuk et al., 2024)

{{<citation>}}

Agnia Sergeyuk, Sergey Titov, Maliheh Izadi. (2024)  
**In-IDE Human-AI Experience in the Era of Large Language Models; A Literature Review**  

---
Primary Category: cs.SE  
Categories: cs-HC, cs-SE, cs.SE  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.10739v2)  

---


**ABSTRACT**  
Integrated Development Environments (IDEs) have become central to modern software development, especially with the integration of Artificial Intelligence (AI) to enhance programming efficiency and decision-making. The study of in-IDE Human-AI Experience is critical in understanding how these AI tools are transforming the software development process, impacting programmer productivity, and influencing code quality. We conducted a literature review to study the current state of in-IDE Human-AI Experience research, bridging a gap in understanding the nuanced interactions between programmers and AI assistants within IDEs. By analyzing 36 selected papers, our study illustrates three primary research branches: Design, Impact, and Quality of Interaction. The trends, challenges, and opportunities identified in this paper emphasize the evolving landscape of software development and inform future directions for research and development in this dynamic field. Specifically, we invite the community to investigate three aspects of these interactions: designing task-specific user interface, building trust, and improving readability.

{{</citation>}}


### (65/98) ZnTrack -- Data as Code (Fabian Zills et al., 2024)

{{<citation>}}

Fabian Zills, Moritz Schäfer, Samuel Tovey, Johannes Kästner, Christian Holm. (2024)  
**ZnTrack -- Data as Code**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-LG, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.10603v1)  

---


**ABSTRACT**  
The past decade has seen tremendous breakthroughs in computation and there is no indication that this will slow any time soon. Machine learning, large-scale computing resources, and increased industry focus have resulted in rising investments in computer-driven solutions for data management, simulations, and model generation. However, with this growth in computation has come an even larger expansion of data and with it, complexity in data storage, sharing, and tracking. In this work, we introduce ZnTrack, a Python-driven data versioning tool. ZnTrack builds upon established version control systems to provide a user-friendly and easy-to-use interface for tracking parameters in experiments, designing workflows, and storing and sharing data. From this ability to reduce large datasets to a simple Python script emerges the concept of Data as Code, a core component of the work presented here and an undoubtedly important concept as the age of computation continues to evolve. ZnTrack offers an open-source, FAIR data compatible Python package to enable users to harness these concepts of the future.

{{</citation>}}


## cs.RO (1)



### (66/98) A Novel and Accurate BiLSTM Configuration Controller for Modular Soft Robots with Module Number Adaptability (Zixi Chen et al., 2024)

{{<citation>}}

Zixi Chen, Matteo Bernabei, Vanessa Mainardi, Xuyang Ren, Gastone Ciuti, Cesare Stefanini. (2024)  
**A Novel and Accurate BiLSTM Configuration Controller for Modular Soft Robots with Module Number Adaptability**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.10997v1)  

---


**ABSTRACT**  
Modular soft robots have shown higher potential in sophisticated tasks than single-module robots. However, the modular structure incurs the complexity of accurate control and necessitates a control strategy specifically for modular robots. In this paper, we introduce a data collection strategy and a novel and accurate bidirectional LSTM configuration controller for modular soft robots with module number adaptability. Such a controller can control module configurations in robots with different module numbers. Simulation cable-driven robots and real pneumatic robots have been included in experiments to validate the proposed approaches, and we have proven that our controller can be leveraged even with the increase or decrease of module number. This is the first paper that gets inspiration from the physical structure of modular robots and utilizes bidirectional LSTM for module number adaptability. Future work may include a planning method that bridges the task and configuration spaces and the integration of an online controller.

{{</citation>}}


## cs.LG (12)



### (67/98) Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning (Adib Hasan et al., 2024)

{{<citation>}}

Adib Hasan, Ileana Rugina, Alex Wang. (2024)  
**Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: LLaMA, Language Model, Pruning  
[Paper Link](http://arxiv.org/abs/2401.10862v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are vulnerable to `Jailbreaking' prompts, a type of attack that can coax these models into generating harmful and illegal content. In this paper, we show that pruning up to 20% of LLM parameters markedly increases their resistance to such attacks without additional training and without sacrificing their performance in standard benchmarks. Intriguingly, we discovered that the enhanced safety observed post-pruning correlates to the initial safety training level of the model, hinting that the effect of pruning could be more general and may hold for other LLM behaviors beyond safety. Additionally, we introduce a curated dataset of 225 harmful tasks across five categories, inserted into ten different Jailbreaking prompts, showing that pruning aids LLMs in concentrating attention on task-relevant tokens in jailbreaking prompts. Lastly, our experiments reveal that the prominent chat models, such as LLaMA-2 Chat, Vicuna, and Mistral Instruct exhibit high susceptibility to jailbreaking attacks, with some categories achieving nearly 70-100% success rate. These insights underline the potential of pruning as a generalizable approach for improving LLM safety, reliability, and potentially other desired behaviors.

{{</citation>}}


### (68/98) Novel Representation Learning Technique using Graphs for Performance Analytics (Tarek Ramadan et al., 2024)

{{<citation>}}

Tarek Ramadan, Ankur Lahiry, Tanzima Z. Islam. (2024)  
**Novel Representation Learning Technique using Graphs for Performance Analytics**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.10799v1)  

---


**ABSTRACT**  
The performance analytics domain in High Performance Computing (HPC) uses tabular data to solve regression problems, such as predicting the execution time. Existing Machine Learning (ML) techniques leverage the correlations among features given tabular datasets, not leveraging the relationships between samples directly. Moreover, since high-quality embeddings from raw features improve the fidelity of the downstream predictive models, existing methods rely on extensive feature engineering and pre-processing steps, costing time and manual effort. To fill these two gaps, we propose a novel idea of transforming tabular performance data into graphs to leverage the advancement of Graph Neural Network-based (GNN) techniques in capturing complex relationships between features and samples. In contrast to other ML application domains, such as social networks, the graph is not given; instead, we need to build it. To address this gap, we propose graph-building methods where nodes represent samples, and the edges are automatically inferred iteratively based on the similarity between the features in the samples. We evaluate the effectiveness of the generated embeddings from GNNs based on how well they make even a simple feed-forward neural network perform for regression tasks compared to other state-of-the-art representation learning techniques. Our evaluation demonstrates that even with up to 25% random missing values for each dataset, our method outperforms commonly used graph and Deep Neural Network (DNN)-based approaches and achieves up to 61.67% & 78.56% improvement in MSE loss over the DNN baseline respectively for HPC dataset and Machine Learning Datasets.

{{</citation>}}


### (69/98) Deep Reinforcement Learning Empowered Activity-Aware Dynamic Health Monitoring Systems (Ziqiaing Ye et al., 2024)

{{<citation>}}

Ziqiaing Ye, Yulan Gao, Yue Xiao, Zehui Xiong, Dusit Niyato. (2024)  
**Deep Reinforcement Learning Empowered Activity-Aware Dynamic Health Monitoring Systems**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.10794v1)  

---


**ABSTRACT**  
In smart healthcare, health monitoring utilizes diverse tools and technologies to analyze patients' real-time biosignal data, enabling immediate actions and interventions. Existing monitoring approaches were designed on the premise that medical devices track several health metrics concurrently, tailored to their designated functional scope. This means that they report all relevant health values within that scope, which can result in excess resource use and the gathering of extraneous data due to monitoring irrelevant health metrics. In this context, we propose Dynamic Activity-Aware Health Monitoring strategy (DActAHM) for striking a balance between optimal monitoring performance and cost efficiency, a novel framework based on Deep Reinforcement Learning (DRL) and SlowFast Model to ensure precise monitoring based on users' activities. Specifically, with the SlowFast Model, DActAHM efficiently identifies individual activities and captures these results for enhanced processing. Subsequently, DActAHM refines health metric monitoring in response to the identified activity by incorporating a DRL framework. Extensive experiments comparing DActAHM against three state-of-the-art approaches demonstrate it achieves 27.3% higher gain than the best-performing baseline that fixes monitoring actions over timeline.

{{</citation>}}


### (70/98) Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads (Tianle Cai et al., 2024)

{{<citation>}}

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao. (2024)  
**Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.10774v1)  

---


**ABSTRACT**  
The inference process in Large Language Models (LLMs) is often limited due to the absence of parallelism in the auto-regressive decoding process, resulting in most operations being restricted by the memory bandwidth of accelerators. While methods such as speculative decoding have been suggested to address this issue, their implementation is impeded by the challenges associated with acquiring and maintaining a separate draft model. In this paper, we present Medusa, an efficient method that augments LLM inference by adding extra decoding heads to predict multiple subsequent tokens in parallel. Using a tree-based attention mechanism, Medusa constructs multiple candidate continuations and verifies them simultaneously in each decoding step. By leveraging parallel processing, Medusa introduces only minimal overhead in terms of single-step latency while substantially reducing the number of decoding steps required.   We present two levels of fine-tuning procedures for Medusa to meet the needs of different use cases: Medusa-1: Medusa is directly fine-tuned on top of a frozen backbone LLM, enabling lossless inference acceleration. Medusa-2: Medusa is fine-tuned together with the backbone LLM, enabling better prediction accuracy of Medusa heads and higher speedup but needing a special training recipe that preserves the backbone model's capabilities.   Moreover, we propose several extensions that improve or expand the utility of Medusa, including a self-distillation to handle situations where no training data is available and a typical acceptance scheme to boost the acceptance rate while maintaining generation quality. We evaluate Medusa on models of various sizes and training procedures. Our experiments demonstrate that Medusa-1 can achieve over 2.2x speedup without compromising generation quality, while Medusa-2 further improves the speedup to 2.3-3.6x.

{{</citation>}}


### (71/98) Starlit: Privacy-Preserving Federated Learning to Enhance Financial Fraud Detection (Aydin Abadi et al., 2024)

{{<citation>}}

Aydin Abadi, Bradley Doyle, Francesco Gini, Kieron Guinamard, Sasi Kumar Murakonda, Jack Liddell, Paul Mellor, Steven J. Murdoch, Mohammad Naseri, Hector Page, George Theodorakopoulos, Suzanne Weller. (2024)  
**Starlit: Privacy-Preserving Federated Learning to Enhance Financial Fraud Detection**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Financial, Fraud Detection  
[Paper Link](http://arxiv.org/abs/2401.10765v2)  

---


**ABSTRACT**  
Federated Learning (FL) is a data-minimization approach enabling collaborative model training across diverse clients with local data, avoiding direct data exchange. However, state-of-the-art FL solutions to identify fraudulent financial transactions exhibit a subset of the following limitations. They (1) lack a formal security definition and proof, (2) assume prior freezing of suspicious customers' accounts by financial institutions (limiting the solutions' adoption), (3) scale poorly, involving either $O(n^2)$ computationally expensive modular exponentiation (where $n$ is the total number of financial institutions) or highly inefficient fully homomorphic encryption, (4) assume the parties have already completed the identity alignment phase, hence excluding it from the implementation, performance evaluation, and security analysis, and (5) struggle to resist clients' dropouts. This work introduces Starlit, a novel scalable privacy-preserving FL mechanism that overcomes these limitations. It has various applications, such as enhancing financial fraud detection, mitigating terrorism, and enhancing digital health. We implemented Starlit and conducted a thorough performance analysis using synthetic data from a key player in global financial transactions. The evaluation indicates Starlit's scalability, efficiency, and accuracy.

{{</citation>}}


### (72/98) Data Augmentation for Traffic Classification (Chao Wang et al., 2024)

{{<citation>}}

Chao Wang, Alessandro Finamore, Pietro Michiardi, Massimo Gallo, Dario Rossi. (2024)  
**Data Augmentation for Traffic Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NI, cs.LG  
Keywords: Augmentation, Computer Vision, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.10754v1)  

---


**ABSTRACT**  
Data Augmentation (DA) -- enriching training data by adding synthetic samples -- is a technique widely adopted in Computer Vision (CV) and Natural Language Processing (NLP) tasks to improve models performance. Yet, DA has struggled to gain traction in networking contexts, particularly in Traffic Classification (TC) tasks. In this work, we fulfill this gap by benchmarking 18 augmentation functions applied to 3 TC datasets using packet time series as input representation and considering a variety of training conditions. Our results show that (i) DA can reap benefits previously unexplored with (ii) augmentations acting on time series sequence order and masking being a better suit for TC and (iii) simple latent space analysis can provide hints about why augmentations have positive or negative effects.

{{</citation>}}


### (73/98) Safe Offline Reinforcement Learning with Feasibility-Guided Diffusion Model (Yinan Zheng et al., 2024)

{{<citation>}}

Yinan Zheng, Jianxiong Li, Dongjie Yu, Yujie Yang, Shengbo Eben Li, Xianyuan Zhan, Jingjing Liu. (2024)  
**Safe Offline Reinforcement Learning with Feasibility-Guided Diffusion Model**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.10700v1)  

---


**ABSTRACT**  
Safe offline RL is a promising way to bypass risky online interactions towards safe policy learning. Most existing methods only enforce soft constraints, i.e., constraining safety violations in expectation below thresholds predetermined. This can lead to potentially unsafe outcomes, thus unacceptable in safety-critical scenarios. An alternative is to enforce the hard constraint of zero violation. However, this can be challenging in offline setting, as it needs to strike the right balance among three highly intricate and correlated aspects: safety constraint satisfaction, reward maximization, and behavior regularization imposed by offline datasets. Interestingly, we discover that via reachability analysis of safe-control theory, the hard safety constraint can be equivalently translated to identifying the largest feasible region given the offline dataset. This seamlessly converts the original trilogy problem to a feasibility-dependent objective, i.e., maximizing reward value within the feasible region while minimizing safety risks in the infeasible region. Inspired by these, we propose FISOR (FeasIbility-guided Safe Offline RL), which allows safety constraint adherence, reward maximization, and offline policy learning to be realized via three decoupled processes, while offering strong safety performance and stability. In FISOR, the optimal policy for the translated optimization problem can be derived in a special form of weighted behavior cloning. Thus, we propose a novel energy-guided diffusion model that does not require training a complicated time-dependent classifier to extract the policy, greatly simplifying the training. We compare FISOR against baselines on DSRL benchmark for safe offline RL. Evaluation results show that FISOR is the only method that can guarantee safety satisfaction in all tasks, while achieving top returns in most tasks.

{{</citation>}}


### (74/98) FIMBA: Evaluating the Robustness of AI in Genomics via Feature Importance Adversarial Attacks (Heorhii Skovorodnikov et al., 2024)

{{<citation>}}

Heorhii Skovorodnikov, Hoda Alkhzaimi. (2024)  
**FIMBA: Evaluating the Robustness of AI in Genomics via Feature Importance Adversarial Attacks**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG, q-bio-GN  
Keywords: AI, Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2401.10657v1)  

---


**ABSTRACT**  
With the steady rise of the use of AI in bio-technical applications and the widespread adoption of genomics sequencing, an increasing amount of AI-based algorithms and tools is entering the research and production stage affecting critical decision-making streams like drug discovery and clinical outcomes. This paper demonstrates the vulnerability of AI models often utilized downstream tasks on recognized public genomics datasets. We undermine model robustness by deploying an attack that focuses on input transformation while mimicking the real data and confusing the model decision-making, ultimately yielding a pronounced deterioration in model performance. Further, we enhance our approach by generating poisoned data using a variational autoencoder-based model. Our empirical findings unequivocally demonstrate a decline in model performance, underscored by diminished accuracy and an upswing in false positives and false negatives. Furthermore, we analyze the resulting adversarial samples via spectral analysis yielding conclusions for countermeasures against such attacks.

{{</citation>}}


### (75/98) Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation (Jialong Zhou et al., 2024)

{{<citation>}}

Jialong Zhou, Xing Ai, Yuni Lai, Kai Zhou. (2024)  
**Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Augmentation, Contrastive Learning, GNN  
[Paper Link](http://arxiv.org/abs/2401.10590v1)  

---


**ABSTRACT**  
Signed graphs consist of edges and signs, which can be separated into structural information and balance-related information, respectively. Existing signed graph neural networks (SGNNs) typically rely on balance-related information to generate embeddings. Nevertheless, the emergence of recent adversarial attacks has had a detrimental impact on the balance-related information. Similar to how structure learning can restore unsigned graphs, balance learning can be applied to signed graphs by improving the balance degree of the poisoned graph. However, this approach encounters the challenge "Irreversibility of Balance-related Information" - while the balance degree improves, the restored edges may not be the ones originally affected by attacks, resulting in poor defense effectiveness. To address this challenge, we propose a robust SGNN framework called Balance Augmented-Signed Graph Contrastive Learning (BA-SGCL), which combines Graph Contrastive Learning principles with balance augmentation techniques. Experimental results demonstrate that BA-SGCL not only enhances robustness against existing adversarial attacks but also achieves superior performance on link sign prediction task across various datasets.

{{</citation>}}


### (76/98) Episodic Reinforcement Learning with Expanded State-reward Space (Dayang Liang et al., 2024)

{{<citation>}}

Dayang Liang, Yaru Zhang, Yunlong Liu. (2024)  
**Episodic Reinforcement Learning with Expanded State-reward Space**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.10516v1)  

---


**ABSTRACT**  
Empowered by deep neural networks, deep reinforcement learning (DRL) has demonstrated tremendous empirical successes in various domains, including games, health care, and autonomous driving. Despite these advancements, DRL is still identified as data-inefficient as effective policies demand vast numbers of environmental samples. Recently, episodic control (EC)-based model-free DRL methods enable sample efficiency by recalling past experiences from episodic memory. However, existing EC-based methods suffer from the limitation of potential misalignment between the state and reward spaces for neglecting the utilization of (past) retrieval states with extensive information, which probably causes inaccurate value estimation and degraded policy performance. To tackle this issue, we introduce an efficient EC-based DRL framework with expanded state-reward space, where the expanded states used as the input and the expanded rewards used in the training both contain historical and current information. To be specific, we reuse the historical states retrieved by EC as part of the input states and integrate the retrieved MC-returns into the immediate reward in each interactive transition. As a result, our method is able to simultaneously achieve the full utilization of retrieval information and the better evaluation of state values by a Temporal Difference (TD) loss. Empirical results on challenging Box2d and Mujoco tasks demonstrate the superiority of our method over a recent sibling method and common baselines. Further, we also verify our method's effectiveness in alleviating Q-value overestimation by additional experiments of Q-value comparison.

{{</citation>}}


### (77/98) LDReg: Local Dimensionality Regularized Self-Supervised Learning (Hanxun Huang et al., 2024)

{{<citation>}}

Hanxun Huang, Ricardo J. G. B. Campello, Sarah Monazam Erfani, Xingjun Ma, Michael E. Houle, James Bailey. (2024)  
**LDReg: Local Dimensionality Regularized Self-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG, stat-ML  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.10474v1)  

---


**ABSTRACT**  
Representations learned via self-supervised learning (SSL) can be susceptible to dimensional collapse, where the learned representation subspace is of extremely low dimensionality and thus fails to represent the full data distribution and modalities. Dimensional collapse also known as the "underfilling" phenomenon is one of the major causes of degraded performance on downstream tasks. Previous work has investigated the dimensional collapse problem of SSL at a global level. In this paper, we demonstrate that representations can span over high dimensional space globally, but collapse locally. To address this, we propose a method called $\textit{local dimensionality regularization (LDReg)}$. Our formulation is based on the derivation of the Fisher-Rao metric to compare and optimize local distance distributions at an asymptotically small radius for each data point. By increasing the local intrinsic dimensionality, we demonstrate through a range of experiments that LDReg improves the representation quality of SSL. The results also show that LDReg can regularize dimensionality at both local and global levels.

{{</citation>}}


### (78/98) A2Q+: Improving Accumulator-Aware Weight Quantization (Ian Colbert et al., 2024)

{{<citation>}}

Ian Colbert, Alessandro Pappalardo, Jakoba Petri-Koenig, Yaman Umuroglu. (2024)  
**A2Q+: Improving Accumulator-Aware Weight Quantization**  

---
Primary Category: cs.LG  
Categories: cs-AR, cs-LG, cs-PF, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2401.10432v1)  

---


**ABSTRACT**  
Quantization techniques commonly reduce the inference costs of neural networks by restricting the precision of weights and activations. Recent studies show that also reducing the precision of the accumulator can further improve hardware efficiency at the risk of numerical overflow, which introduces arithmetic errors that can degrade model accuracy. To avoid numerical overflow while maintaining accuracy, recent work proposed accumulator-aware quantization (A2Q), a quantization-aware training method that constrains model weights during training to safely use a target accumulator bit width during inference. Although this shows promise, we demonstrate that A2Q relies on an overly restrictive constraint and a sub-optimal weight initialization strategy that each introduce superfluous quantization error. To address these shortcomings, we introduce: (1) an improved bound that alleviates accumulator constraints without compromising overflow avoidance; and (2) a new strategy for initializing quantized weights from pre-trained floating-point checkpoints. We combine these contributions with weight normalization to introduce A2Q+. We support our analysis with experiments that show A2Q+ significantly improves the trade-off between accumulator bit width and model accuracy and characterize new trade-offs that arise as a consequence of accumulator constraints.

{{</citation>}}


## q-bio.BM (1)



### (79/98) Clustering Molecular Energy Landscapes by Adaptive Network Embedding (Paula Mercurio et al., 2024)

{{<citation>}}

Paula Mercurio, Di Liu. (2024)  
**Clustering Molecular Energy Landscapes by Adaptive Network Embedding**  

---
Primary Category: q-bio.BM  
Categories: cond-mat-stat-mech, cs-LG, q-bio-BM, q-bio.BM  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.10972v1)  

---


**ABSTRACT**  
In order to efficiently explore the chemical space of all possible small molecules, a common approach is to compress the dimension of the system to facilitate downstream machine learning tasks. Towards this end, we present a data driven approach for clustering potential energy landscapes of molecular structures by applying recently developed Network Embedding techniques, to obtain latent variables defined through the embedding function. To scale up the method, we also incorporate an entropy sensitive adaptive scheme for hierarchical sampling of the energy landscape, based on Metadynamics and Transition Path Theory. By taking into account the kinetic information implied by a system's energy landscape, we are able to interpret dynamical node-node relationships in reduced dimensions. We demonstrate the framework through Lennard-Jones (LJ) clusters and a human DNA sequence.

{{</citation>}}


## cs.AI (5)



### (80/98) Optimisation in Neurosymbolic Learning Systems (Emile van Krieken, 2024)

{{<citation>}}

Emile van Krieken. (2024)  
**Optimisation in Neurosymbolic Learning Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.10819v1)  

---


**ABSTRACT**  
Neurosymbolic AI aims to integrate deep learning with symbolic AI. This integration has many promises, such as decreasing the amount of data required to train a neural network, improving the explainability and interpretability of answers given by models and verifying the correctness of trained systems. We study neurosymbolic learning, where we have both data and background knowledge expressed using symbolic languages. How do we connect the symbolic and neural components to communicate this knowledge? One option is fuzzy reasoning, which studies degrees of truth. For example, being tall is not a binary concept. Instead, probabilistic reasoning studies the probability that something is true or will happen. Our first research question studies how different forms of fuzzy reasoning combine with learning. We find surprising results like a connection to the Raven paradox stating we confirm "ravens are black" when we observe a green apple. In this study, we did not use the background knowledge when we deployed our models after training. In our second research question, we studied how to use background knowledge in deployed models. We developed a new neural network layer based on fuzzy reasoning. Probabilistic reasoning is a natural fit for neural networks, which we usually train to be probabilistic. However, they are expensive to compute and do not scale well to large tasks. In our third research question, we study how to connect probabilistic reasoning with neural networks by sampling to estimate averages, while in the final research question, we study scaling probabilistic neurosymbolic learning to much larger problems than before. Our insight is to train a neural network with synthetic data to predict the result of probabilistic reasoning.

{{</citation>}}


### (81/98) FinLLMs: A Framework for Financial Reasoning Dataset Generation with Large Language Models (Ziqiang Yuan et al., 2024)

{{<citation>}}

Ziqiang Yuan, Kaiyuan Wang, Shoutai Zhu, Ye Yuan, Jingya Zhou, Yanlin Zhu, Wenqi Wei. (2024)  
**FinLLMs: A Framework for Financial Reasoning Dataset Generation with Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Financial, GPT, GPT-3.5, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.10744v1)  

---


**ABSTRACT**  
Large Language models (LLMs) usually rely on extensive training datasets. In the financial domain, creating numerical reasoning datasets that include a mix of tables and long text often involves substantial manual annotation expenses. To address the limited data resources and reduce the annotation cost, we introduce FinLLMs, a method for generating financial question-answering data based on common financial formulas using Large Language Models. First, we compile a list of common financial formulas and construct a graph based on the variables these formulas employ. We then augment the formula set by combining those that share identical variables as new elements. Specifically, we explore formulas obtained by manual annotation and merge those formulas with shared variables by traversing the constructed graph. Finally, utilizing GPT-3.5, we generate financial question-answering data that encompasses both tabular information and long textual content, building on the collected formula set. Our experiments demonstrate that synthetic data generated by FinLLMs effectively enhances the performance of several large-scale numerical reasoning models in the financial domain, outperforming two established benchmark financial question-answering datasets.

{{</citation>}}


### (82/98) CivRealm: A Learning and Reasoning Odyssey in Civilization for Decision-Making Agents (Siyuan Qi et al., 2024)

{{<citation>}}

Siyuan Qi, Shuo Chen, Yexin Li, Xiangyu Kong, Junqi Wang, Bangcheng Yang, Pring Wong, Yifan Zhong, Xiaoyuan Zhang, Zhaowei Zhang, Nian Liu, Wei Wang, Yaodong Yang, Song-Chun Zhu. (2024)  
**CivRealm: A Learning and Reasoning Odyssey in Civilization for Decision-Making Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.10568v1)  

---


**ABSTRACT**  
The generalization of decision-making agents encompasses two fundamental elements: learning from past experiences and reasoning in novel contexts. However, the predominant emphasis in most interactive environments is on learning, often at the expense of complexity in reasoning. In this paper, we introduce CivRealm, an environment inspired by the Civilization game. Civilization's profound alignment with human history and society necessitates sophisticated learning, while its ever-changing situations demand strong reasoning to generalize. Particularly, CivRealm sets up an imperfect-information general-sum game with a changing number of players; it presents a plethora of complex features, challenging the agent to deal with open-ended stochastic environments that require diplomacy and negotiation skills. Within CivRealm, we provide interfaces for two typical agent types: tensor-based agents that focus on learning, and language-based agents that emphasize reasoning. To catalyze further research, we present initial results for both paradigms. The canonical RL-based agents exhibit reasonable performance in mini-games, whereas both RL- and LLM-based agents struggle to make substantial progress in the full game. Overall, CivRealm stands as a unique learning and reasoning challenge for decision-making agents. The code is available at https://github.com/bigai-ai/civrealm.

{{</citation>}}


### (83/98) Learning Backdoors for Mixed Integer Programs with Contrastive Learning (Junyang Cai et al., 2024)

{{<citation>}}

Junyang Cai, Taoan Huang, Bistra Dilkina. (2024)  
**Learning Backdoors for Mixed Integer Programs with Contrastive Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI, math-OC  
Keywords: Attention, Contrastive Learning, Graph Attention Network  
[Paper Link](http://arxiv.org/abs/2401.10467v1)  

---


**ABSTRACT**  
Many real-world problems can be efficiently modeled as Mixed Integer Programs (MIPs) and solved with the Branch-and-Bound method. Prior work has shown the existence of MIP backdoors, small sets of variables such that prioritizing branching on them when possible leads to faster running times. However, finding high-quality backdoors that improve running times remains an open question. Previous work learns to estimate the relative solver speed of randomly sampled backdoors through ranking and then decide whether to use it. In this paper, we utilize the Monte-Carlo tree search method to collect backdoors for training, rather than relying on random sampling, and adapt a contrastive learning framework to train a Graph Attention Network model to predict backdoors. Our method, evaluated on four common MIP problem domains, demonstrates performance improvements over both Gurobi and previous models.

{{</citation>}}


### (84/98) Can A Cognitive Architecture Fundamentally Enhance LLMs? Or Vice Versa? (Ron Sun, 2024)

{{<citation>}}

Ron Sun. (2024)  
**Can A Cognitive Architecture Fundamentally Enhance LLMs? Or Vice Versa?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.10444v1)  

---


**ABSTRACT**  
The paper discusses what is needed to address the limitations of current LLM-centered AI systems. The paper argues that incorporating insights from human cognition and psychology, as embodied by a computational cognitive architecture, can help develop systems that are more capable, more reliable, and more human-like. It emphasizes the importance of the dual-process architecture and the hybrid neuro-symbolic approach in addressing the limitations of current LLMs. In the opposite direction, the paper also highlights the need for an overhaul of computational cognitive architectures to better reflect advances in AI and computing technology. Overall, the paper advocates for a multidisciplinary, mutually beneficial approach towards developing better models both for AI and for understanding the human mind.

{{</citation>}}


## cs.AR (2)



### (85/98) BoolGebra: Attributed Graph-learning for Boolean Algebraic Manipulation (Yingjie Li et al., 2024)

{{<citation>}}

Yingjie Li, Anthony Agnesina, Yanqing Zhang, Haoxing Ren, Cunxi Yu. (2024)  
**BoolGebra: Attributed Graph-learning for Boolean Algebraic Manipulation**  

---
Primary Category: cs.AR  
Categories: cs-AI, cs-AR, cs-LG, cs.AR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.10753v1)  

---


**ABSTRACT**  
Boolean algebraic manipulation is at the core of logic synthesis in Electronic Design Automation (EDA) design flow. Existing methods struggle to fully exploit optimization opportunities, and often suffer from an explosive search space and limited scalability efficiency. This work presents BoolGebra, a novel attributed graph-learning approach for Boolean algebraic manipulation that aims to improve fundamental logic synthesis. BoolGebra incorporates Graph Neural Networks (GNNs) and takes initial feature embeddings from both structural and functional information as inputs. A fully connected neural network is employed as the predictor for direct optimization result predictions, significantly reducing the search space and efficiently locating the optimization space. The experiments involve training the BoolGebra model w.r.t design-specific and cross-design inferences using the trained model, where BoolGebra demonstrates generalizability for cross-design inference and its potential to scale from small, simple training datasets to large, complex inference datasets. Finally, BoolGebra is integrated with existing synthesis tool ABC to perform end-to-end logic minimization evaluation w.r.t SOTA baselines.

{{</citation>}}


### (86/98) FARe: Fault-Aware GNN Training on ReRAM-based PIM Accelerators (Pratyush Dhingra et al., 2024)

{{<citation>}}

Pratyush Dhingra, Chukwufumnanya Ogbogu, Biresh Kumar Joardar, Janardhan Rao Doppa, Ananth Kalyanaraman, Partha Pratim Pande. (2024)  
**FARe: Fault-Aware GNN Training on ReRAM-based PIM Accelerators**  

---
Primary Category: cs.AR  
Categories: B-8-1, cs-AR, cs-LG, cs.AR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.10522v1)  

---


**ABSTRACT**  
Resistive random-access memory (ReRAM)-based processing-in-memory (PIM) architecture is an attractive solution for training Graph Neural Networks (GNNs) on edge platforms. However, the immature fabrication process and limited write endurance of ReRAMs make them prone to hardware faults, thereby limiting their widespread adoption for GNN training. Further, the existing fault-tolerant solutions prove inadequate for effectively training GNNs in the presence of faults. In this paper, we propose a fault-aware framework referred to as FARe that mitigates the effect of faults during GNN training. FARe outperforms existing approaches in terms of both accuracy and timing overhead. Experimental results demonstrate that FARe framework can restore GNN test accuracy by 47.6% on faulty ReRAM hardware with a ~1% timing overhead compared to the fault-free counterpart.

{{</citation>}}


## eess.IV (3)



### (87/98) Dynamic Semantic Compression for CNN Inference in Multi-access Edge Computing: A Graph Reinforcement Learning-based Autoencoder (Nan Li et al., 2024)

{{<citation>}}

Nan Li, Alexandros Iosifidis, Qi Zhang. (2024)  
**Dynamic Semantic Compression for CNN Inference in Multi-access Edge Computing: A Graph Reinforcement Learning-based Autoencoder**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-LG, eess-IV, eess.IV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.12167v1)  

---


**ABSTRACT**  
This paper studies the computational offloading of CNN inference in dynamic multi-access edge computing (MEC) networks. To address the uncertainties in communication time and computation resource availability, we propose a novel semantic compression method, autoencoder-based CNN architecture (AECNN), for effective semantic extraction and compression in partial offloading. In the semantic encoder, we introduce a feature compression module based on the channel attention mechanism in CNNs, to compress intermediate data by selecting the most informative features. In the semantic decoder, we design a lightweight decoder to reconstruct the intermediate data through learning from the received compressed data to improve accuracy. To effectively trade-off communication, computation, and inference accuracy, we design a reward function and formulate the offloading problem of CNN inference as a maximization problem with the goal of maximizing the average inference accuracy and throughput over the long term. To address this maximization problem, we propose a graph reinforcement learning-based AECNN (GRL-AECNN) method, which outperforms existing works DROO-AECNN, GRL-BottleNet++ and GRL-DeepJSCC under different dynamic scenarios. This highlights the advantages of GRL-AECNN in offloading decision-making in dynamic MEC.

{{</citation>}}


### (88/98) Towards Universal Unsupervised Anomaly Detection in Medical Imaging (Cosmin I. Bercea et al., 2024)

{{<citation>}}

Cosmin I. Bercea, Benedikt Wiestler, Daniel Rueckert, Julia A. Schnabel. (2024)  
**Towards Universal Unsupervised Anomaly Detection in Medical Imaging**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2401.10637v1)  

---


**ABSTRACT**  
The increasing complexity of medical imaging data underscores the need for advanced anomaly detection methods to automatically identify diverse pathologies. Current methods face challenges in capturing the broad spectrum of anomalies, often limiting their use to specific lesion types in brain scans. To address this challenge, we introduce a novel unsupervised approach, termed \textit{Reversed Auto-Encoders (RA)}, designed to create realistic pseudo-healthy reconstructions that enable the detection of a wider range of pathologies. We evaluate the proposed method across various imaging modalities, including magnetic resonance imaging (MRI) of the brain, pediatric wrist X-ray, and chest X-ray, and demonstrate superior performance in detecting anomalies compared to existing state-of-the-art methods. Our unsupervised anomaly detection approach may enhance diagnostic accuracy in medical imaging by identifying a broader range of unknown pathologies. Our code is publicly available at: \url{https://github.com/ci-ber/RA}.

{{</citation>}}


### (89/98) MAEDiff: Masked Autoencoder-enhanced Diffusion Models for Unsupervised Anomaly Detection in Brain Images (Rui Xu et al., 2024)

{{<citation>}}

Rui Xu, Yunke Wang, Bo Du. (2024)  
**MAEDiff: Masked Autoencoder-enhanced Diffusion Models for Unsupervised Anomaly Detection in Brain Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2401.10561v1)  

---


**ABSTRACT**  
Unsupervised anomaly detection has gained significant attention in the field of medical imaging due to its capability of relieving the costly pixel-level annotation. To achieve this, modern approaches usually utilize generative models to produce healthy references of the diseased images and then identify the abnormalities by comparing the healthy references and the original diseased images. Recently, diffusion models have exhibited promising potential for unsupervised anomaly detection in medical images for their good mode coverage and high sample quality. However, the intrinsic characteristics of the medical images, e.g. the low contrast, and the intricate anatomical structure of the human body make the reconstruction challenging. Besides, the global information of medical images often remain underutilized. To address these two issues, we propose a novel Masked Autoencoder-enhanced Diffusion Model (MAEDiff) for unsupervised anomaly detection in brain images. The MAEDiff involves a hierarchical patch partition. It generates healthy images by overlapping upper-level patches and implements a mechanism based on the masked autoencoders operating on the sub-level patches to enhance the condition on the unnoised regions. Extensive experiments on data of tumors and multiple sclerosis lesions demonstrate the effectiveness of our method.

{{</citation>}}


## cs.IR (4)



### (90/98) Dynamic Q&A of Clinical Documents with Large Language Models (Ran Elgedawy et al., 2024)

{{<citation>}}

Ran Elgedawy, Sudarshan Srinivasan, Ioana Danciu. (2024)  
**Dynamic Q&A of Clinical Documents with Large Language Models**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: AI, Clinical, Language Model  
[Paper Link](http://arxiv.org/abs/2401.10733v1)  

---


**ABSTRACT**  
Electronic health records (EHRs) house crucial patient data in clinical notes. As these notes grow in volume and complexity, manual extraction becomes challenging. This work introduces a natural language interface using large language models (LLMs) for dynamic question-answering on clinical notes. Our chatbot, powered by Langchain and transformer-based LLMs, allows users to query in natural language, receiving relevant answers from clinical notes. Experiments, utilizing various embedding models and advanced LLMs, show Wizard Vicuna's superior accuracy, albeit with high compute demands. Model optimization, including weight quantization, improves latency by approximately 48 times. Promising results indicate potential, yet challenges such as model hallucinations and limited diverse medical case evaluations remain. Addressing these gaps is crucial for unlocking the value in clinical notes and advancing AI-driven clinical decision-making.

{{</citation>}}


### (91/98) Publication venue recommendation using profiles based on clustering (Luis M. de Campos et al., 2024)

{{<citation>}}

Luis M. de Campos, Juan M. Fernández-Luna, Juan F. Huete. (2024)  
**Publication venue recommendation using profiles based on clustering**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2401.10611v1)  

---


**ABSTRACT**  
In this paper we study the venue recommendation problem in order to help researchers to identify a journal or conference to submit a given paper. A common approach to tackle this problem is to build profiles defining the scope of each venue. Then, these profiles are compared against the target paper. In our approach we will study how clustering techniques can be used to construct topic-based profiles and use an Information Retrieval based approach to obtain the final recommendations. Additionally, we will explore how the use of authorship, representing a complementary piece of information, helps to improve the recommendations.

{{</citation>}}


### (92/98) Understanding Biases in ChatGPT-based Recommender Systems: Provider Fairness, Temporal Stability, and Recency (Yashar Deldjoo, 2024)

{{<citation>}}

Yashar Deldjoo. (2024)  
**Understanding Biases in ChatGPT-based Recommender Systems: Provider Fairness, Temporal Stability, and Recency**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Bias, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.10545v1)  

---


**ABSTRACT**  
This study explores the nuanced capabilities and inherent biases of Recommender Systems using Large Language Models (RecLLMs), with a focus on ChatGPT-based systems. It studies into the contrasting behaviors of generative models and traditional collaborative filtering models in movie recommendations. The research primarily investigates prompt design strategies and their impact on various aspects of recommendation quality, including accuracy, provider fairness, diversity, stability, genre dominance, and temporal freshness (recency).   Our experimental analysis reveals that the introduction of specific 'system roles' and 'prompt strategies' in RecLLMs significantly influences their performance. For instance, role-based prompts enhance fairness and diversity in recommendations, mitigating popularity bias. We find that while GPT-based models do not always match the performance of CF baselines, they exhibit a unique tendency to recommend newer and more diverse movie genres. Notably, GPT-based models tend to recommend more recent films, particularly those released post-2000, and show a preference for genres like \sq{Drama} and Comedy, and Romance (compared to CF Action, Adventure) presumably due to the RecLLMs' training on varied data sets, which allows them to capture recent trends and discussions more effectively than CF models. Interestingly, our results demonstrate that the 'Simple' and 'Chain of Thought (COT)' paradigms yield the highest accuracy. These findings imply the potential of combining these strategies with scenarios that favor more recent content, thereby offering a more balanced and up-to-date recommendation experience. This study contributes significantly to the understanding of emerging RecLLMs, particularly in the context of harms and biases within these systems.

{{</citation>}}


### (93/98) Enhancing Scalability in Recommender Systems through Lottery Ticket Hypothesis and Knowledge Distillation-based Neural Network Pruning (Rajaram R et al., 2024)

{{<citation>}}

Rajaram R, Manoj Bharadhwaj, Vasan VS, Nargis Pervin. (2024)  
**Enhancing Scalability in Recommender Systems through Lottery Ticket Hypothesis and Knowledge Distillation-based Neural Network Pruning**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-AR, cs-IR, cs.IR  
Keywords: Knowledge Distillation, Pruning  
[Paper Link](http://arxiv.org/abs/2401.10484v1)  

---


**ABSTRACT**  
This study introduces an innovative approach aimed at the efficient pruning of neural networks, with a particular focus on their deployment on edge devices. Our method involves the integration of the Lottery Ticket Hypothesis (LTH) with the Knowledge Distillation (KD) framework, resulting in the formulation of three distinct pruning models. These models have been developed to address scalability issue in recommender systems, whereby the complexities of deep learning models have hindered their practical deployment. With judicious application of the pruning techniques, we effectively curtail the power consumption and model dimensions without compromising on accuracy. Empirical evaluation has been performed using two real world datasets from diverse domains against two baselines. Gratifyingly, our approaches yielded a GPU computation-power reduction of up to 66.67%. Notably, our study contributes to the field of recommendation system by pioneering the application of LTH and KD.

{{</citation>}}


## cs.LO (1)



### (94/98) DRAT Proofs of Unsatisfiability for SAT Modulo Monotonic Theories (Nick Feng et al., 2024)

{{<citation>}}

Nick Feng, Alan J. Hu, Sam Bayless, Syed M. Iqbal, Patrick Trentin, Mike Whalen, Lee Pike, John Backes. (2024)  
**DRAT Proofs of Unsatisfiability for SAT Modulo Monotonic Theories**  

---
Primary Category: cs.LO  
Categories: cs-LO, cs.LO  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2401.10703v1)  

---


**ABSTRACT**  
Generating proofs of unsatisfiability is a valuable capability of most SAT solvers, and is an active area of research for SMT solvers. This paper introduces the first method to efficiently generate proofs of unsatisfiability specifically for an important subset of SMT: SAT Modulo Monotonic Theories (SMMT), which includes many useful finite-domain theories (e.g., bit vectors and many graph-theoretic properties) and is used in production at Amazon Web Services. Our method uses propositional definitions of the theory predicates, from which it generates compact Horn approximations of the definitions, which lead to efficient DRAT proofs, leveraging the large investment the SAT community has made in DRAT. In experiments on practical SMMT problems, our proof generation overhead is minimal (7.41% geometric mean slowdown, 28.8% worst-case), and we can generate and check proofs for many problems that were previously intractable.

{{</citation>}}


## quant-ph (1)



### (95/98) QuantumReservoirPy: A Software Package for Time Series Prediction (Stanley Miao et al., 2024)

{{<citation>}}

Stanley Miao, Ola Tangen Kulseng, Alexander Stasik, Franz G. Fuchs. (2024)  
**QuantumReservoirPy: A Software Package for Time Series Prediction**  

---
Primary Category: quant-ph  
Categories: cs-SE, quant-ph, quant-ph  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.10683v1)  

---


**ABSTRACT**  
In recent times, quantum reservoir computing has emerged as a potential resource for time series prediction. Hence, there is a need for a flexible framework to test quantum circuits as nonlinear dynamical systems. We have developed a software package to allow for quantum reservoirs to fit a common structure, similar to that of reservoirpy which is advertised as "a python tool designed to easily define, train and use (classical) reservoir computing architectures". Our package results in simplified development and logical methods of comparison between quantum reservoir architectures. Examples are provided to demonstrate the resulting simplicity of executing quantum reservoir computing using our software package.

{{</citation>}}


## cs.NI (1)



### (96/98) Empowering HWNs with Efficient Data Labeling: A Clustered Federated Semi-Supervised Learning Approach (Moqbel Hamood et al., 2024)

{{<citation>}}

Moqbel Hamood, Abdullatif Albaseer, Mohamed Abdallah, Ala Al-Fuqaha. (2024)  
**Empowering HWNs with Efficient Data Labeling: A Clustered Federated Semi-Supervised Learning Approach**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2401.10646v1)  

---


**ABSTRACT**  
Clustered Federated Multitask Learning (CFL) has gained considerable attention as an effective strategy for overcoming statistical challenges, particularly when dealing with non independent and identically distributed (non IID) data across multiple users. However, much of the existing research on CFL operates under the unrealistic premise that devices have access to accurate ground truth labels. This assumption becomes especially problematic in hierarchical wireless networks (HWNs), where edge networks contain a large amount of unlabeled data, resulting in slower convergence rates and increased processing times, particularly when dealing with two layers of model aggregation. To address these issues, we introduce a novel framework, Clustered Federated Semi-Supervised Learning (CFSL), designed for more realistic HWN scenarios. Our approach leverages a best-performing specialized model algorithm, wherein each device is assigned a specialized model that is highly adept at generating accurate pseudo-labels for unlabeled data, even when the data stems from diverse environments. We validate the efficacy of CFSL through extensive experiments, comparing it with existing methods highlighted in recent literature. Our numerical results demonstrate that CFSL significantly improves upon key metrics such as testing accuracy, labeling accuracy, and labeling latency under varying proportions of labeled and unlabeled data while also accommodating the non-IID nature of the data and the unique characteristics of wireless edge networks.

{{</citation>}}


## cs.SD (1)



### (97/98) AAT: Adapting Audio Transformer for Various Acoustics Recognition Tasks (Yun Liang et al., 2024)

{{<citation>}}

Yun Liang, Hai Lin, Shaojian Qiu, Yihang Zhang. (2024)  
**AAT: Adapting Audio Transformer for Various Acoustics Recognition Tasks**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.10544v1)  

---


**ABSTRACT**  
Recently, Transformers have been introduced into the field of acoustics recognition. They are pre-trained on large-scale datasets using methods such as supervised learning and semi-supervised learning, demonstrating robust generality--It fine-tunes easily to downstream tasks and shows more robust performance. However, the predominant fine-tuning method currently used is still full fine-tuning, which involves updating all parameters during training. This not only incurs significant memory usage and time costs but also compromises the model's generality. Other fine-tuning methods either struggle to address this issue or fail to achieve matching performance. Therefore, we conducted a comprehensive analysis of existing fine-tuning methods and proposed an efficient fine-tuning approach based on Adapter tuning, namely AAT. The core idea is to freeze the audio Transformer model and insert extra learnable Adapters, efficiently acquiring downstream task knowledge without compromising the model's original generality. Extensive experiments have shown that our method achieves performance comparable to or even superior to full fine-tuning while optimizing only 7.118% of the parameters. It also demonstrates superiority over other fine-tuning methods.

{{</citation>}}


## eess.SY (1)



### (98/98) Efficient Probabilistic Optimal Power Flow Assessment Using an Adaptive Stochastic Spectral Embedding Surrogate Model (Xiaoting Wang et al., 2024)

{{<citation>}}

Xiaoting Wang, Jingyu Liu, Xiaozhe Wang. (2024)  
**Efficient Probabilistic Optimal Power Flow Assessment Using an Adaptive Stochastic Spectral Embedding Surrogate Model**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.10498v1)  

---


**ABSTRACT**  
This paper presents an adaptive stochastic spectral embedding (ASSE) method to solve the probabilistic AC optimal power flow (AC-OPF), a critical aspect of power system operation. The proposed method can efficiently and accurately estimate the probabilistic characteristics of AC-OPF solutions. An adaptive domain partition strategy and expansion coefficient calculation algorithm are integrated to enhance its performance. Numerical studies on a 9-bus system demonstrate that the proposed ASSE method offers accurate and fast evaluations compared to the Monte Carlo simulations. A comparison with a sparse polynomial chaos expansion method, an existing surrogate model, further demonstrates its efficacy in accurately assessing the responses with strongly local behaviors.

{{</citation>}}
