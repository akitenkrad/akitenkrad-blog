---
draft: false
title: "arXiv @ 2023.09.22"
date: 2023-09-22
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.22"
    identifier: arxiv_20230922
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (33)](#cscl-33)
- [cs.AI (13)](#csai-13)
- [cs.LG (21)](#cslg-21)
- [cs.IR (5)](#csir-5)
- [cs.CV (25)](#cscv-25)
- [cs.DS (1)](#csds-1)
- [cs.HC (3)](#cshc-3)
- [cs.CY (1)](#cscy-1)
- [cs.DL (2)](#csdl-2)
- [cs.SD (2)](#cssd-2)
- [cs.CR (2)](#cscr-2)
- [q-fin.TR (1)](#q-fintr-1)
- [cs.RO (5)](#csro-5)
- [cs.GR (1)](#csgr-1)
- [eess.AS (3)](#eessas-3)
- [eess.SP (1)](#eesssp-1)
- [eess.SY (1)](#eesssy-1)
- [q-bio.GN (1)](#q-biogn-1)
- [cs.DB (1)](#csdb-1)

## cs.CL (33)



### (1/122) Semi-supervised News Discourse Profiling with Contrastive Learning (Ming Li et al., 2023)

{{<citation>}}

Ming Li, Ruihong Huang. (2023)  
**Semi-supervised News Discourse Profiling with Contrastive Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.11692v1)  

---


**ABSTRACT**  
News Discourse Profiling seeks to scrutinize the event-related role of each sentence in a news article and has been proven useful across various downstream applications. Specifically, within the context of a given news discourse, each sentence is assigned to a pre-defined category contingent upon its depiction of the news event structure. However, existing approaches suffer from an inadequacy of available human-annotated data, due to the laborious and time-intensive nature of generating discourse-level annotations. In this paper, we present a novel approach, denoted as Intra-document Contrastive Learning with Distillation (ICLD), for addressing the news discourse profiling task, capitalizing on its unique structural characteristics. Notably, we are the first to apply a semi-supervised methodology within this task paradigm, and evaluation demonstrates the effectiveness of the presented approach.

{{</citation>}}


### (2/122) A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models (Haoran Xu et al., 2023)

{{<citation>}}

Haoran Xu, Young Jin Kim, Amr Sharaf, Hany Hassan Awadalla. (2023)  
**A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, GPT, GPT-3.5, LLaMA, Language Model, Machine Translation, NLP  
[Paper Link](http://arxiv.org/abs/2309.11674v1)  

---


**ABSTRACT**  
Generative Large Language Models (LLMs) have achieved remarkable advancements in various NLP tasks. However, these advances have not been reflected in the translation task, especially those with moderate model sizes (i.e., 7B or 13B parameters), which still lag behind conventional supervised encoder-decoder translation models. Previous studies have attempted to improve the translation capabilities of these moderate LLMs, but their gains have been limited. In this study, we propose a novel fine-tuning approach for LLMs that is specifically designed for the translation task, eliminating the need for the abundant parallel data that traditional translation models usually depend on. Our approach consists of two fine-tuning stages: initial fine-tuning on monolingual data followed by subsequent fine-tuning on a small set of high-quality parallel data. We introduce the LLM developed through this strategy as Advanced Language Model-based trAnslator (ALMA). Based on LLaMA-2 as our underlying model, our results show that the model can achieve an average improvement of more than 12 BLEU and 12 COMET over its zero-shot performance across 10 translation directions from the WMT'21 (2 directions) and WMT'22 (8 directions) test datasets. The performance is significantly better than all prior work and even superior to the NLLB-54B model and GPT-3.5-text-davinci-003, with only 7B or 13B parameters. This method establishes the foundation for a novel training paradigm in machine translation.

{{</citation>}}


### (3/122) Construction of Paired Knowledge Graph-Text Datasets Informed by Cyclic Evaluation (Ali Mousavi et al., 2023)

{{<citation>}}

Ali Mousavi, Xin Zhan, He Bai, Peng Shi, Theo Rekatsinas, Benjamin Han, Yunyao Li, Jeff Pound, Josh Susskind, Natalie Schluter, Ihab Ilyas, Navdeep Jaitly. (2023)  
**Construction of Paired Knowledge Graph-Text Datasets Informed by Cyclic Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.11669v1)  

---


**ABSTRACT**  
Datasets that pair Knowledge Graphs (KG) and text together (KG-T) can be used to train forward and reverse neural models that generate text from KG and vice versa. However models trained on datasets where KG and text pairs are not equivalent can suffer from more hallucination and poorer recall. In this paper, we verify this empirically by generating datasets with different levels of noise and find that noisier datasets do indeed lead to more hallucination. We argue that the ability of forward and reverse models trained on a dataset to cyclically regenerate source KG or text is a proxy for the equivalence between the KG and the text in the dataset. Using cyclic evaluation we find that manually created WebNLG is much better than automatically created TeKGen and T-REx. Guided by these observations, we construct a new, improved dataset called LAGRANGE using heuristics meant to improve equivalence between KG and text and show the impact of each of the heuristics on cyclic evaluation. We also construct two synthetic datasets using large language models (LLMs), and observe that these are conducive to models that perform significantly well on cyclic generation of text, but less so on cyclic generation of KGs, probably because of a lack of a consistent underlying ontology.

{{</citation>}}


### (4/122) Towards Effective Disambiguation for Machine Translation with Large Language Models (Vivek Iyer et al., 2023)

{{<citation>}}

Vivek Iyer, Pinzhen Chen, Alexandra Birch. (2023)  
**Towards Effective Disambiguation for Machine Translation with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2309.11668v1)  

---


**ABSTRACT**  
Resolving semantic ambiguity has long been recognised as a central challenge in the field of machine translation. Recent work on benchmarking translation performance on ambiguous sentences has exposed the limitations of conventional Neural Machine Translation (NMT) systems, which fail to capture many of these cases. Large language models (LLMs) have emerged as a promising alternative, demonstrating comparable performance to traditional NMT models while introducing new paradigms for controlling the target outputs. In this paper, we study the capabilities of LLMs to translate ambiguous sentences containing polysemous words and rare word senses. We also propose two ways to improve the handling of such ambiguity through in-context learning and fine-tuning on carefully curated ambiguous datasets. Experiments show that our methods can match or outperform state-of-the-art systems such as DeepL and NLLB in four out of five language directions. Our research provides valuable insights into effectively adapting LLMs for disambiguation during machine translation.

{{</citation>}}


### (5/122) Hate speech detection in algerian dialect using deep learning (Dihia Lanasri et al., 2023)

{{<citation>}}

Dihia Lanasri, Juan Olano, Sifal Klioui, Sin Liang Lee, Lamia Sekkai. (2023)  
**Hate speech detection in algerian dialect using deep learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2309.11611v1)  

---


**ABSTRACT**  
With the proliferation of hate speech on social networks under different formats, such as abusive language, cyberbullying, and violence, etc., people have experienced a significant increase in violence, putting them in uncomfortable situations and threats. Plenty of efforts have been dedicated in the last few years to overcome this phenomenon to detect hate speech in different structured languages like English, French, Arabic, and others. However, a reduced number of works deal with Arabic dialects like Tunisian, Egyptian, and Gulf, mainly the Algerian ones. To fill in the gap, we propose in this work a complete approach for detecting hate speech on online Algerian messages. Many deep learning architectures have been evaluated on the corpus we created from some Algerian social networks (Facebook, YouTube, and Twitter). This corpus contains more than 13.5K documents in Algerian dialect written in Arabic, labeled as hateful or non-hateful. Promising results are obtained, which show the efficiency of our approach.

{{</citation>}}


### (6/122) SignBank+: Multilingual Sign Language Translation Dataset (Amit Moryossef et al., 2023)

{{<citation>}}

Amit Moryossef, Zifan Jiang. (2023)  
**SignBank+: Multilingual Sign Language Translation Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2309.11566v1)  

---


**ABSTRACT**  
This work advances the field of sign language machine translation by focusing on dataset quality and simplification of the translation system. We introduce SignBank+, a clean version of the SignBank dataset, optimized for machine translation. Contrary to previous works that employ complex factorization techniques for translation, we advocate for a simplified text-to-text translation approach. Our evaluation shows that models trained on SignBank+ surpass those on the original dataset, establishing a new benchmark and providing an open resource for future research.

{{</citation>}}


### (7/122) Chain-of-Verification Reduces Hallucination in Large Language Models (Shehzaad Dhuliawala et al., 2023)

{{<citation>}}

Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, Jason Weston. (2023)  
**Chain-of-Verification Reduces Hallucination in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2309.11495v2)  

---


**ABSTRACT**  
Generation of plausible yet incorrect factual information, termed hallucination, is an unsolved issue in large language models. We study the ability of language models to deliberate on the responses they give in order to correct their mistakes. We develop the Chain-of-Verification (CoVe) method whereby the model first (i) drafts an initial response; then (ii) plans verification questions to fact-check its draft; (iii) answers those questions independently so the answers are not biased by other responses; and (iv) generates its final verified response. In experiments, we show CoVe decreases hallucinations across a variety of tasks, from list-based questions from Wikidata, closed book MultiSpanQA and longform text generation.

{{</citation>}}


### (8/122) Controlled Generation with Prompt Insertion for Natural Language Explanations in Grammatical Error Correction (Masahiro Kaneko et al., 2023)

{{<citation>}}

Masahiro Kaneko, Naoaki Okazaki. (2023)  
**Controlled Generation with Prompt Insertion for Natural Language Explanations in Grammatical Error Correction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.11439v1)  

---


**ABSTRACT**  
In Grammatical Error Correction (GEC), it is crucial to ensure the user's comprehension of a reason for correction. Existing studies present tokens, examples, and hints as to the basis for correction but do not directly explain the reasons for corrections. Although methods that use Large Language Models (LLMs) to provide direct explanations in natural language have been proposed for various tasks, no such method exists for GEC. Generating explanations for GEC corrections involves aligning input and output tokens, identifying correction points, and presenting corresponding explanations consistently. However, it is not straightforward to specify a complex format to generate explanations, because explicit control of generation is difficult with prompts. This study introduces a method called controlled generation with Prompt Insertion (PI) so that LLMs can explain the reasons for corrections in natural language. In PI, LLMs first correct the input text, and then we automatically extract the correction points based on the rules. The extracted correction points are sequentially inserted into the LLM's explanation output as prompts, guiding the LLMs to generate explanations for the correction points. We also create an Explainable GEC (XGEC) dataset of correction reasons by annotating NUCLE, CoNLL2013, and CoNLL2014. Although generations from GPT-3 and ChatGPT using original prompts miss some correction points, the generation control using PI can explicitly guide to describe explanations for all correction points, contributing to improved performance in generating correction reasons.

{{</citation>}}


### (9/122) You Only Look at Screens: Multimodal Chain-of-Action Agents (Zhuosheng Zhang et al., 2023)

{{<citation>}}

Zhuosheng Zhang, Aston Zhang. (2023)  
**You Only Look at Screens: Multimodal Chain-of-Action Agents**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.11436v2)  

---


**ABSTRACT**  
Autonomous user interface (UI) agents aim to facilitate task automation by interacting with the user interface without manual intervention. Recent studies have investigated eliciting the capabilities of large language models (LLMs) for effective engagement in diverse environments. To align with the input-output requirement of LLMs, existing approaches are developed under a sandbox setting where they rely on external tools and application-specific APIs to parse the environment into textual elements and interpret the predicted actions. Consequently, those approaches often grapple with inference inefficiency and error propagation risks. To mitigate the challenges, we introduce Auto-UI, a multimodal solution that directly interacts with the interface, bypassing the need for environment parsing or reliance on application-dependent APIs. Moreover, we propose a chain-of-action technique -- leveraging a series of intermediate previous action histories and future action plans -- to help the agent decide what action to execute. We evaluate our approach on a new device-control benchmark AITW with 30K unique instructions, spanning multi-step tasks such as application operation, web searching, and web shopping. Experimental results show that Auto-UI achieves state-of-the-art performance with an action type prediction accuracy of 90% and an overall action success rate of 74%. Code is publicly available at https://github.com/cooelf/Auto-UI.

{{</citation>}}


### (10/122) Kosmos-2.5: A Multimodal Literate Model (Tengchao Lv et al., 2023)

{{<citation>}}

Tengchao Lv, Yupan Huang, Jingye Chen, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, Shaoxiang Wu, Guoxin Wang, Cha Zhang, Furu Wei. (2023)  
**Kosmos-2.5: A Multimodal Literate Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.11419v1)  

---


**ABSTRACT**  
We present Kosmos-2.5, a multimodal literate model for machine reading of text-intensive images. Pre-trained on large-scale text-intensive images, Kosmos-2.5 excels in two distinct yet cooperative transcription tasks: (1) generating spatially-aware text blocks, where each block of text is assigned its spatial coordinates within the image, and (2) producing structured text output that captures styles and structures into the markdown format. This unified multimodal literate capability is achieved through a shared Transformer architecture, task-specific prompts, and flexible text representations. We evaluate Kosmos-2.5 on end-to-end document-level text recognition and image-to-markdown text generation. Furthermore, the model can be readily adapted for any text-intensive image understanding task with different prompts through supervised fine-tuning, making it a general-purpose tool for real-world applications involving text-rich images. This work also paves the way for the future scaling of multimodal large language models.

{{</citation>}}


### (11/122) Safurai 001: New Qualitative Approach for Code LLM Evaluation (Davide Cifarelli et al., 2023)

{{<citation>}}

Davide Cifarelli, Leonardo Boiardi, Alessandro Puppo. (2023)  
**Safurai 001: New Qualitative Approach for Code LLM Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2309.11385v1)  

---


**ABSTRACT**  
This paper presents Safurai-001, a new Large Language Model (LLM) with significant potential in the domain of coding assistance. Driven by recent advancements in coding LLMs, Safurai-001 competes in performance with the latest models like WizardCoder [Xu et al., 2023], PanguCoder [Shen et al., 2023] and Phi-1 [Gunasekar et al., 2023] but aims to deliver a more conversational interaction. By capitalizing on the progress in data engineering (including latest techniques of data transformation and prompt engineering) and instruction tuning, this new model promises to stand toe-to-toe with recent closed and open source developments. Recognizing the need for an efficacious evaluation metric for coding LLMs, this paper also introduces GPT4-based MultiParameters, an evaluation benchmark that harnesses varied parameters to present a comprehensive insight into the models functioning and performance. Our assessment shows that Safurai-001 can outperform GPT-3.5 by 1.58% and WizardCoder by 18.78% in the Code Readability parameter and more.

{{</citation>}}


### (12/122) Studying Lobby Influence in the European Parliament (Aswin Suresh et al., 2023)

{{<citation>}}

Aswin Suresh, Lazar Radojevic, Francesco Salvi, Antoine Magron, Victor Kristof, Matthias Grossglauser. (2023)  
**Studying Lobby Influence in the European Parliament**  

---
Primary Category: cs.CL  
Categories: cs-CE, cs-CL, cs-CY, cs-SI, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.11381v1)  

---


**ABSTRACT**  
We present a method based on natural language processing (NLP), for studying the influence of interest groups (lobbies) in the law-making process in the European Parliament (EP). We collect and analyze novel datasets of lobbies' position papers and speeches made by members of the EP (MEPs). By comparing these texts on the basis of semantic similarity and entailment, we are able to discover interpretable links between MEPs and lobbies. In the absence of a ground-truth dataset of such links, we perform an indirect validation by comparing the discovered links with a dataset, which we curate, of retweet links between MEPs and lobbies, and with the publicly disclosed meetings of MEPs. Our best method achieves an AUC score of 0.77 and performs significantly better than several baselines. Moreover, an aggregate analysis of the discovered links, between groups of related lobbies and political groups of MEPs, correspond to the expectations from the ideology of the groups (e.g., center-left groups are associated with social causes). We believe that this work, which encompasses the methodology, datasets, and results, is a step towards enhancing the transparency of the intricate decision-making processes within democratic institutions.

{{</citation>}}


### (13/122) Incremental Blockwise Beam Search for Simultaneous Speech Translation with Controllable Quality-Latency Tradeoff (Peter Polák et al., 2023)

{{<citation>}}

Peter Polák, Brian Yan, Shinji Watanabe, Alex Waibel, Ondřej Bojar. (2023)  
**Incremental Blockwise Beam Search for Simultaneous Speech Translation with Controllable Quality-Latency Tradeoff**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2309.11379v1)  

---


**ABSTRACT**  
Blockwise self-attentional encoder models have recently emerged as one promising end-to-end approach to simultaneous speech translation. These models employ a blockwise beam search with hypothesis reliability scoring to determine when to wait for more input speech before translating further. However, this method maintains multiple hypotheses until the entire speech input is consumed -- this scheme cannot directly show a single \textit{incremental} translation to users. Further, this method lacks mechanisms for \textit{controlling} the quality vs. latency tradeoff. We propose a modified incremental blockwise beam search incorporating local agreement or hold-$n$ policies for quality-latency control. We apply our framework to models trained for online or offline translation and demonstrate that both types can be effectively used in online mode.   Experimental results on MuST-C show 0.6-3.6 BLEU improvement without changing latency or 0.8-1.4 s latency improvement without changing quality.

{{</citation>}}


### (14/122) DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services (Shengbin Yue et al., 2023)

{{<citation>}}

Shengbin Yue, Wei Chen, Siyuan Wang, Bingxuan Li, Chenchen Shen, Shujun Liu, Yuxuan Zhou, Yao Xiao, Song Yun, Xuanjing Huang, Zhongyu Wei. (2023)  
**DISC-LawLLM: Fine-tuning Large Language Models for Intelligent Legal Services**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2309.11325v2)  

---


**ABSTRACT**  
We propose DISC-LawLLM, an intelligent legal system utilizing large language models (LLMs) to provide a wide range of legal services. We adopt legal syllogism prompting strategies to construct supervised fine-tuning datasets in the Chinese Judicial domain and fine-tune LLMs with legal reasoning capability. We augment LLMs with a retrieval module to enhance models' ability to access and utilize external legal knowledge. A comprehensive legal benchmark, DISC-Law-Eval, is presented to evaluate intelligent legal systems from both objective and subjective dimensions. Quantitative and qualitative results on DISC-Law-Eval demonstrate the effectiveness of our system in serving various users across diverse legal scenarios. The detailed resources are available at https://github.com/FudanDISC/DISC-LawLLM.

{{</citation>}}


### (15/122) Rating Prediction in Conversational Task Assistants with Behavioral and Conversational-Flow Features (Rafael Ferreira et al., 2023)

{{<citation>}}

Rafael Ferreira, David Semedo, João Magalhães. (2023)  
**Rating Prediction in Conversational Task Assistants with Behavioral and Conversational-Flow Features**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.11307v1)  

---


**ABSTRACT**  
Predicting the success of Conversational Task Assistants (CTA) can be critical to understand user behavior and act accordingly. In this paper, we propose TB-Rater, a Transformer model which combines conversational-flow features with user behavior features for predicting user ratings in a CTA scenario. In particular, we use real human-agent conversations and ratings collected in the Alexa TaskBot challenge, a novel multimodal and multi-turn conversational context. Our results show the advantages of modeling both the conversational-flow and behavioral aspects of the conversation in a single model for offline rating prediction. Additionally, an analysis of the CTA-specific behavioral features brings insights into this setting and can be used to bootstrap future systems.

{{</citation>}}


### (16/122) CPLLM: Clinical Prediction with Large Language Models (Ofir Ben Shoham et al., 2023)

{{<citation>}}

Ofir Ben Shoham, Nadav Rappoport. (2023)  
**CPLLM: Clinical Prediction with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, BERT, Clinical, Language Model  
[Paper Link](http://arxiv.org/abs/2309.11295v1)  

---


**ABSTRACT**  
We present Clinical Prediction with Large Language Models (CPLLM), a method that involves fine-tuning a pre-trained Large Language Model (LLM) for clinical disease prediction. We utilized quantization and fine-tuned the LLM using prompts, with the task of predicting whether patients will be diagnosed with a target disease during their next visit or in the subsequent diagnosis, leveraging their historical diagnosis records. We compared our results versus various baselines, including Logistic Regression, RETAIN, and Med-BERT, which is the current state-of-the-art model for disease prediction using structured EHR data. Our experiments have shown that CPLLM surpasses all the tested models in terms of both PR-AUC and ROC-AUC metrics, displaying noteworthy enhancements compared to the baseline models.

{{</citation>}}


### (17/122) The Wizard of Curiosities: Enriching Dialogues with Fun Facts (Frederico Vicente et al., 2023)

{{<citation>}}

Frederico Vicente, Rafael Ferreira, David Semedo, João Magalhães. (2023)  
**The Wizard of Curiosities: Enriching Dialogues with Fun Facts**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: Amazon, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2309.11283v1)  

---


**ABSTRACT**  
Introducing curiosities in a conversation is a way to teach something new to the person in a pleasant and enjoyable way. Enriching dialogues with contextualized curiosities can improve the users' perception of a dialog system and their overall user experience. In this paper, we introduce a set of curated curiosities, targeting dialogues in the cooking and DIY domains. In particular, we use real human-agent conversations collected in the context of the Amazon Alexa TaskBot challenge, a multimodal and multi-turn conversational setting. According to an A/B test with over 1000 conversations, curiosities not only increase user engagement, but provide an average relative rating improvement of 9.7%.

{{</citation>}}


### (18/122) Grounded Complex Task Segmentation for Conversational Assistants (Rafael Ferreira et al., 2023)

{{<citation>}}

Rafael Ferreira, David Semedo, João Magalhães. (2023)  
**Grounded Complex Task Segmentation for Conversational Assistants**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.11271v1)  

---


**ABSTRACT**  
Following complex instructions in conversational assistants can be quite daunting due to the shorter attention and memory spans when compared to reading the same instructions. Hence, when conversational assistants walk users through the steps of complex tasks, there is a need to structure the task into manageable pieces of information of the right length and complexity. In this paper, we tackle the recipes domain and convert reading structured instructions into conversational structured ones. We annotated the structure of instructions according to a conversational scenario, which provided insights into what is expected in this setting. To computationally model the conversational step's characteristics, we tested various Transformer-based architectures, showing that a token-based approach delivers the best results. A further user study showed that users tend to favor steps of manageable complexity and length, and that the proposed methodology can improve the original web-based instructional text. Specifically, 86% of the evaluated tasks were improved from a conversational suitability point of view.

{{</citation>}}


### (19/122) Sequence-to-Sequence Spanish Pre-trained Language Models (Vladimir Araujo et al., 2023)

{{<citation>}}

Vladimir Araujo, Maria Mihaela Trusca, Rodrigo Tufiño, Marie-Francine Moens. (2023)  
**Sequence-to-Sequence Spanish Pre-trained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, GPT, Language Model, Sequence-to-Sequence, T5  
[Paper Link](http://arxiv.org/abs/2309.11259v1)  

---


**ABSTRACT**  
In recent years, substantial advancements in pre-trained language models have paved the way for the development of numerous non-English language versions, with a particular focus on encoder-only and decoder-only architectures. While Spanish language models encompassing BERT, RoBERTa, and GPT have exhibited prowess in natural language understanding and generation, there remains a scarcity of encoder-decoder models designed for sequence-to-sequence tasks involving input-output pairs. This paper breaks new ground by introducing the implementation and evaluation of renowned encoder-decoder architectures, exclusively pre-trained on Spanish corpora. Specifically, we present Spanish versions of BART, T5, and BERT2BERT-style models and subject them to a comprehensive assessment across a diverse range of sequence-to-sequence tasks, spanning summarization, rephrasing, and generative question answering. Our findings underscore the competitive performance of all models, with BART and T5 emerging as top performers across all evaluated tasks. As an additional contribution, we have made all models publicly available to the research community, fostering future exploration and development in Spanish language processing.

{{</citation>}}


### (20/122) OpenChat: Advancing Open-source Language Models with Mixed-Quality Data (Guan Wang et al., 2023)

{{<citation>}}

Guan Wang, Sijie Cheng, Xianyuan Zhan, Xiangang Li, Sen Song, Yang Liu. (2023)  
**OpenChat: Advancing Open-source Language Models with Mixed-Quality Data**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2309.11235v1)  

---


**ABSTRACT**  
Nowadays, open-source large language models like LLaMA have emerged. Recent developments have incorporated supervised fine-tuning (SFT) and reinforcement learning fine-tuning (RLFT) to align these models with human goals. However, SFT methods treat all training data with mixed quality equally, while RLFT methods require high-quality pairwise or ranking-based preference data. In this study, we present a novel framework, named OpenChat, to advance open-source language models with mixed-quality data. Specifically, we consider the general SFT training data, consisting of a small amount of expert data mixed with a large proportion of sub-optimal data, without any preference labels. We propose the C(onditioned)-RLFT, which regards different data sources as coarse-grained reward labels and learns a class-conditioned policy to leverage complementary data quality information. Interestingly, the optimal policy in C-RLFT can be easily solved through single-stage, RL-free supervised learning, which is lightweight and avoids costly human preference labeling. Through extensive experiments on three standard benchmarks, our openchat-13b fine-tuned with C-RLFT achieves the highest average performance among all 13b open-source language models. Moreover, we use AGIEval to validate the model generalization performance, in which only openchat-13b surpasses the base model. Finally, we conduct a series of analyses to shed light on the effectiveness and robustness of OpenChat. Our code, data, and models are publicly available at https://github.com/imoneoi/openchat.

{{</citation>}}


### (21/122) Retrieve-Rewrite-Answer: A KG-to-Text Enhanced LLMs Framework for Knowledge Graph Question Answering (Yike Wu et al., 2023)

{{<citation>}}

Yike Wu, Nan Hu, Sheng Bi, Guilin Qi, Jie Ren, Anhuan Xie, Wei Song. (2023)  
**Retrieve-Rewrite-Answer: A KG-to-Text Enhanced LLMs Framework for Knowledge Graph Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Knowledge Graph, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.11206v2)  

---


**ABSTRACT**  
Despite their competitive performance on knowledge-intensive tasks, large language models (LLMs) still have limitations in memorizing all world knowledge especially long tail knowledge. In this paper, we study the KG-augmented language model approach for solving the knowledge graph question answering (KGQA) task that requires rich world knowledge. Existing work has shown that retrieving KG knowledge to enhance LLMs prompting can significantly improve LLMs performance in KGQA. However, their approaches lack a well-formed verbalization of KG knowledge, i.e., they ignore the gap between KG representations and textual representations. To this end, we propose an answer-sensitive KG-to-Text approach that can transform KG knowledge into well-textualized statements most informative for KGQA. Based on this approach, we propose a KG-to-Text enhanced LLMs framework for solving the KGQA task. Experiments on several KGQA benchmarks show that the proposed KG-to-Text augmented LLMs approach outperforms previous KG-augmented LLMs approaches regarding answer accuracy and usefulness of knowledge statements.

{{</citation>}}


### (22/122) Are Large Language Models Really Robust to Word-Level Perturbations? (Haoyu Wang et al., 2023)

{{<citation>}}

Haoyu Wang, Guozheng Ma, Cong Yu, Ning Gui, Linrui Zhang, Zhiqi Huang, Suwei Ma, Yongzhe Chang, Sen Zhang, Li Shen, Xueqian Wang, Peilin Zhao, Dacheng Tao. (2023)  
**Are Large Language Models Really Robust to Word-Level Perturbations?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.11166v1)  

---


**ABSTRACT**  
The swift advancement in the scale and capabilities of Large Language Models (LLMs) positions them as promising tools for a variety of downstream tasks. In addition to the pursuit of better performance and the avoidance of violent feedback on a certain prompt, to ensure the responsibility of the LLM, much attention is drawn to the robustness of LLMs. However, existing evaluation methods mostly rely on traditional question answering datasets with predefined supervised labels, which do not align with the superior generation capabilities of contemporary LLMs. To address this issue, we propose a novel rational evaluation approach that leverages pre-trained reward models as diagnostic tools to evaluate the robustness of LLMs, which we refer to as the Reward Model for Reasonable Robustness Evaluation (TREvaL). Our extensive empirical experiments have demonstrated that TREval provides an accurate method for evaluating the robustness of an LLM, especially when faced with more challenging open questions. Furthermore, our results demonstrate that LLMs frequently exhibit vulnerability to word-level perturbations, which are commonplace in daily language usage. Notably, we were surprised to discover that robustness tends to decrease as fine-tuning (SFT and RLHF) is conducted. The code of TREval is available in https://github.com/Harry-mic/TREval.

{{</citation>}}


### (23/122) Assessment of Pre-Trained Models Across Languages and Grammars (Alberto Muñoz-Ortiz et al., 2023)

{{<citation>}}

Alberto Muñoz-Ortiz, David Vilares, Carlos Gómez-Rodríguez. (2023)  
**Assessment of Pre-Trained Models Across Languages and Grammars**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2309.11165v1)  

---


**ABSTRACT**  
We present an approach for assessing how multilingual large language models (LLMs) learn syntax in terms of multi-formalism syntactic structures. We aim to recover constituent and dependency structures by casting parsing as sequence labeling. To do so, we select a few LLMs and study them on 13 diverse UD treebanks for dependency parsing and 10 treebanks for constituent parsing. Our results show that: (i) the framework is consistent across encodings, (ii) pre-trained word vectors do not favor constituency representations of syntax over dependencies, (iii) sub-word tokenization is needed to represent syntax, in contrast to character-based models, and (iv) occurrence of a language in the pretraining data is more important than the amount of task data when recovering syntax from the word vectors.

{{</citation>}}


### (24/122) CoT-BERT: Enhancing Unsupervised Sentence Representation through Chain-of-Thought (Bowen Zhang et al., 2023)

{{<citation>}}

Bowen Zhang, Kehua Chang, Chunping Li. (2023)  
**CoT-BERT: Enhancing Unsupervised Sentence Representation through Chain-of-Thought**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.11143v1)  

---


**ABSTRACT**  
Unsupervised sentence representation learning aims to transform input sentences into fixed-length vectors enriched with intricate semantic information while obviating the reliance on labeled data. Recent progress within this field, propelled by contrastive learning and prompt engineering, has significantly bridged the gap between unsupervised and supervised strategies. Nonetheless, the potential utilization of Chain-of-Thought, remains largely untapped within this trajectory. To unlock latent capabilities within pre-trained models, such as BERT, we propose a two-stage approach for sentence representation: comprehension and summarization. Subsequently, the output of the latter phase is harnessed as the vectorized representation of the input sentence. For further performance enhancement, we meticulously refine both the contrastive learning loss function and the template denoising technique for prompt engineering. Rigorous experimentation substantiates our method, CoT-BERT, transcending a suite of robust baselines without necessitating other text representation models or external databases.

{{</citation>}}


### (25/122) Prototype of a robotic system to assist the learning process of English language with text-generation through DNN (Carlos Morales-Torres et al., 2023)

{{<citation>}}

Carlos Morales-Torres, Mario Campos-Soberanis, Diego Campos-Sobrino. (2023)  
**Prototype of a robotic system to assist the learning process of English language with text-generation through DNN**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LSTM, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.11142v1)  

---


**ABSTRACT**  
In the last ongoing years, there has been a significant ascending on the field of Natural Language Processing (NLP) for performing multiple tasks including English Language Teaching (ELT). An effective strategy to favor the learning process uses interactive devices to engage learners in their self-learning process. In this work, we present a working prototype of a humanoid robotic system to assist English language self-learners through text generation using Long Short Term Memory (LSTM) Neural Networks. The learners interact with the system using a Graphic User Interface that generates text according to the English level of the user. The experimentation was conducted using English learners and the results were measured accordingly to International English Language Testing System (IELTS) rubric. Preliminary results show an increment in the Grammatical Range of learners who interacted with the system.

{{</citation>}}


### (26/122) AttentionMix: Data augmentation method that relies on BERT attention mechanism (Dominik Lewy et al., 2023)

{{<citation>}}

Dominik Lewy, Jacek Mańdziuk. (2023)  
**AttentionMix: Data augmentation method that relies on BERT attention mechanism**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, BERT, Computer Vision, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.11104v1)  

---


**ABSTRACT**  
The Mixup method has proven to be a powerful data augmentation technique in Computer Vision, with many successors that perform image mixing in a guided manner. One of the interesting research directions is transferring the underlying Mixup idea to other domains, e.g. Natural Language Processing (NLP). Even though there already exist several methods that apply Mixup to textual data, there is still room for new, improved approaches. In this work, we introduce AttentionMix, a novel mixing method that relies on attention-based information. While the paper focuses on the BERT attention mechanism, the proposed approach can be applied to generally any attention-based model. AttentionMix is evaluated on 3 standard sentiment classification datasets and in all three cases outperforms two benchmark approaches that utilize Mixup mechanism, as well as the vanilla BERT method. The results confirm that the attention-based information can be effectively used for data augmentation in the NLP domain.

{{</citation>}}


### (27/122) Design of Chain-of-Thought in Math Problem Solving (Zhanming Jie et al., 2023)

{{<citation>}}

Zhanming Jie, Trung Quoc Luong, Xinbo Zhang, Xiaoran Jin, Hang Li. (2023)  
**Design of Chain-of-Thought in Math Problem Solving**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-PL, cs.CL  
Keywords: GPT, GPT-3.5, QA  
[Paper Link](http://arxiv.org/abs/2309.11054v1)  

---


**ABSTRACT**  
Chain-of-Thought (CoT) plays a crucial role in reasoning for math problem solving. We conduct a comprehensive examination of methods for designing CoT, comparing conventional natural language CoT with various program CoTs, including the self-describing program, the comment-describing program, and the non-describing program. Furthermore, we investigate the impact of programming language on program CoTs, comparing Python and Wolfram Language. Through extensive experiments on GSM8K, MATHQA, and SVAMP, we find that program CoTs often have superior effectiveness in math problem solving. Notably, the best performing combination with 30B parameters beats GPT-3.5-turbo by a significant margin. The results show that self-describing program offers greater diversity and thus can generally achieve higher performance. We also find that Python is a better choice of language than Wolfram for program CoTs. The experimental results provide a valuable guideline for future CoT designs that take into account both programming language and coding style for further advancements. Our datasets and code are publicly available.

{{</citation>}}


### (28/122) fakenewsbr: A Fake News Detection Platform for Brazilian Portuguese (Luiz Giordani et al., 2023)

{{<citation>}}

Luiz Giordani, Gilsiley Darú, Rhenan Queiroz, Vitor Buzinaro, Davi Keglevich Neiva, Daniel Camilo Fuentes Guzmán, Marcos Jardel Henriques, Oilson Alberto Gonzatto Junior, Francisco Louzada. (2023)  
**fakenewsbr: A Fake News Detection Platform for Brazilian Portuguese**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL, stat-ML  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2309.11052v2)  

---


**ABSTRACT**  
The proliferation of fake news has become a significant concern in recent times due to its potential to spread misinformation and manipulate public opinion. This paper presents a comprehensive study on detecting fake news in Brazilian Portuguese, focusing on journalistic-type news. We propose a machine learning-based approach that leverages natural language processing techniques, including TF-IDF and Word2Vec, to extract features from textual data. We evaluate the performance of various classification algorithms, such as logistic regression, support vector machine, random forest, AdaBoost, and LightGBM, on a dataset containing both true and fake news articles. The proposed approach achieves high accuracy and F1-Score, demonstrating its effectiveness in identifying fake news. Additionally, we developed a user-friendly web platform, fakenewsbr.com, to facilitate the verification of news articles' veracity. Our platform provides real-time analysis, allowing users to assess the likelihood of fake news articles. Through empirical analysis and comparative studies, we demonstrate the potential of our approach to contribute to the fight against the spread of fake news and promote more informed media consumption.

{{</citation>}}


### (29/122) Localize, Retrieve and Fuse: A Generalized Framework for Free-Form Question Answering over Tables (Wenting Zhao et al., 2023)

{{<citation>}}

Wenting Zhao, Ye Liu, Yao Wan, Yibo Wang, Zhongfen Deng, Philip S. Yu. (2023)  
**Localize, Retrieve and Fuse: A Generalized Framework for Free-Form Question Answering over Tables**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, QA, Question Answering, T5  
[Paper Link](http://arxiv.org/abs/2309.11049v2)  

---


**ABSTRACT**  
Question answering on tabular data (a.k.a TableQA), which aims at generating answers to questions grounded on a provided table, has gained significant attention recently. Prior work primarily produces concise factual responses through information extraction from individual or limited table cells, lacking the ability to reason across diverse table cells. Yet, the realm of free-form TableQA, which demands intricate strategies for selecting relevant table cells and the sophisticated integration and inference of discrete data fragments, remains mostly unexplored. To this end, this paper proposes a generalized three-stage approach: Table-to- Graph conversion and cell localizing, external knowledge retrieval, and the fusion of table and text (called TAG-QA), to address the challenge of inferring long free-form answers in generative TableQA. In particular, TAG-QA (1) locates relevant table cells using a graph neural network to gather intersecting cells between relevant rows and columns, (2) leverages external knowledge from Wikipedia, and (3) generates answers by integrating both tabular data and natural linguistic information. Experiments showcase the superior capabilities of TAG-QA in generating sentences that are both faithful and coherent, particularly when compared to several state-of-the-art baselines. Notably, TAG-QA surpasses the robust pipeline-based baseline TAPAS by 17% and 14% in terms of BLEU-4 and PARENT F-score, respectively. Furthermore, TAG-QA outperforms the end-to-end model T5 by 16% and 12% on BLEU-4 and PARENT F-score, respectively.

{{</citation>}}


### (30/122) Heterogeneous Entity Matching with Complex Attribute Associations using BERT and Neural Networks (Shitao Wang et al., 2023)

{{<citation>}}

Shitao Wang, Jiamin Lu. (2023)  
**Heterogeneous Entity Matching with Complex Attribute Associations using BERT and Neural Networks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.11046v1)  

---


**ABSTRACT**  
Across various domains, data from different sources such as Baidu Baike and Wikipedia often manifest in distinct forms. Current entity matching methodologies predominantly focus on homogeneous data, characterized by attributes that share the same structure and concise attribute values. However, this orientation poses challenges in handling data with diverse formats. Moreover, prevailing approaches aggregate the similarity of attribute values between corresponding attributes to ascertain entity similarity. Yet, they often overlook the intricate interrelationships between attributes, where one attribute may have multiple associations. The simplistic approach of pairwise attribute comparison fails to harness the wealth of information encapsulated within entities.To address these challenges, we introduce a novel entity matching model, dubbed Entity Matching Model for Capturing Complex Attribute Relationships(EMM-CCAR),built upon pre-trained models. Specifically, this model transforms the matching task into a sequence matching problem to mitigate the impact of varying data formats. Moreover, by introducing attention mechanisms, it identifies complex relationships between attributes, emphasizing the degree of matching among multiple attributes rather than one-to-one correspondences. Through the integration of the EMM-CCAR model, we adeptly surmount the challenges posed by data heterogeneity and intricate attribute interdependencies. In comparison with the prevalent DER-SSM and Ditto approaches, our model achieves improvements of approximately 4% and 1% in F1 scores, respectively. This furnishes a robust solution for addressing the intricacies of attribute complexity in entity matching.

{{</citation>}}


### (31/122) Making Small Language Models Better Multi-task Learners with Mixture-of-Task-Adapters (Yukang Xie et al., 2023)

{{<citation>}}

Yukang Xie, Chengyu Wang, Junbing Yan, Jiyong Zhou, Feiqi Deng, Jun Huang. (2023)  
**Making Small Language Models Better Multi-task Learners with Mixture-of-Task-Adapters**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.11042v1)  

---


**ABSTRACT**  
Recently, Large Language Models (LLMs) have achieved amazing zero-shot learning performance over a variety of Natural Language Processing (NLP) tasks, especially for text generative tasks. Yet, the large size of LLMs often leads to the high computational cost of model training and online deployment. In our work, we present ALTER, a system that effectively builds the multi-tAsk Learners with mixTure-of-task-adaptERs upon small language models (with <1B parameters) to address multiple NLP tasks simultaneously, capturing the commonalities and differences between tasks, in order to support domain-specific applications. Specifically, in ALTER, we propose the Mixture-of-Task-Adapters (MTA) module as an extension to the transformer architecture for the underlying model to capture the intra-task and inter-task knowledge. A two-stage training method is further proposed to optimize the collaboration between adapters at a small computational cost. Experimental results over a mixture of NLP tasks show that our proposed MTA architecture and the two-stage training method achieve good performance. Based on ALTER, we have also produced MTA-equipped language models for various domains.

{{</citation>}}


### (32/122) Named Entity Recognition via Machine Reading Comprehension: A Multi-Task Learning Approach (Yibo Wang et al., 2023)

{{<citation>}}

Yibo Wang, Wenting Zhao, Yao Wan, Zhongfen Deng, Philip S. Yu. (2023)  
**Named Entity Recognition via Machine Reading Comprehension: A Multi-Task Learning Approach**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Reading Comprehension, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2309.11027v1)  

---


**ABSTRACT**  
Named Entity Recognition (NER) aims to extract and classify entity mentions in the text into pre-defined types (e.g., organization or person name). Recently, many works have been proposed to shape the NER as a machine reading comprehension problem (also termed MRC-based NER), in which entity recognition is achieved by answering the formulated questions related to pre-defined entity types through MRC, based on the contexts. However, these works ignore the label dependencies among entity types, which are critical for precisely recognizing named entities. In this paper, we propose to incorporate the label dependencies among entity types into a multi-task learning framework for better MRC-based NER. We decompose MRC-based NER into multiple tasks and use a self-attention module to capture label dependencies. Comprehensive experiments on both nested NER and flat NER datasets are conducted to validate the effectiveness of the proposed Multi-NER. Experimental results show that Multi-NER can achieve better performance on all datasets.

{{</citation>}}


### (33/122) Towards Joint Modeling of Dialogue Response and Speech Synthesis based on Large Language Model (Xinyu Zhou et al., 2023)

{{<citation>}}

Xinyu Zhou, Delong Chen, Yudong Chen. (2023)  
**Towards Joint Modeling of Dialogue Response and Speech Synthesis based on Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: AI, Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2309.11000v1)  

---


**ABSTRACT**  
This paper explores the potential of constructing an AI spoken dialogue system that "thinks how to respond" and "thinks how to speak" simultaneously, which more closely aligns with the human speech production process compared to the current cascade pipeline of independent chatbot and Text-to-Speech (TTS) modules. We hypothesize that Large Language Models (LLMs) with billions of parameters possess significant speech understanding capabilities and can jointly model dialogue responses and linguistic features. We conduct two sets of experiments: 1) Prosodic structure prediction, a typical front-end task in TTS, demonstrating the speech understanding ability of LLMs, and 2) Further integrating dialogue response and a wide array of linguistic features using a unified encoding format. Our results indicate that the LLM-based approach is a promising direction for building unified spoken dialogue systems.

{{</citation>}}


## cs.AI (13)



### (34/122) RAI4IoE: Responsible AI for Enabling the Internet of Energy (Minhui Xue et al., 2023)

{{<citation>}}

Minhui Xue, Surya Nepal, Ling Liu, Subbu Sethuvenkatraman, Xingliang Yuan, Carsten Rudolph, Ruoxi Sun, Greg Eisenhauer. (2023)  
**RAI4IoE: Responsible AI for Enabling the Internet of Energy**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CR, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.11691v1)  

---


**ABSTRACT**  
This paper plans to develop an Equitable and Responsible AI framework with enabling techniques and algorithms for the Internet of Energy (IoE), in short, RAI4IoE. The energy sector is going through substantial changes fueled by two key drivers: building a zero-carbon energy sector and the digital transformation of the energy infrastructure. We expect to see the convergence of these two drivers resulting in the IoE, where renewable distributed energy resources (DERs), such as electric cars, storage batteries, wind turbines and photovoltaics (PV), can be connected and integrated for reliable energy distribution by leveraging advanced 5G-6G networks and AI technology. This allows DER owners as prosumers to participate in the energy market and derive economic incentives. DERs are inherently asset-driven and face equitable challenges (i.e., fair, diverse and inclusive). Without equitable access, privileged individuals, groups and organizations can participate and benefit at the cost of disadvantaged groups. The real-time management of DER resources not only brings out the equity problem to the IoE, it also collects highly sensitive location, time, activity dependent data, which requires to be handled responsibly (e.g., privacy, security and safety), for AI-enhanced predictions, optimization and prioritization services, and automated management of flexible resources. The vision of our project is to ensure equitable participation of the community members and responsible use of their data in IoE so that it could reap the benefits of advances in AI to provide safe, reliable and sustainable energy services.

{{</citation>}}


### (35/122) Generative AI in Mafia-like Game Simulation (Munyeong Kim et al., 2023)

{{<citation>}}

Munyeong Kim, Sungsu Kim. (2023)  
**Generative AI in Mafia-like Game Simulation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI, GPT, GPT-3.5, GPT-4, Generative AI  
[Paper Link](http://arxiv.org/abs/2309.11672v1)  

---


**ABSTRACT**  
In this research, we explore the efficacy and potential of Generative AI models, specifically focusing on their application in role-playing simulations exemplified through Spyfall, a renowned mafia-style game. By leveraging GPT-4's advanced capabilities, the study aimed to showcase the model's potential in understanding, decision-making, and interaction during game scenarios. Comparative analyses between GPT-4 and its predecessor, GPT-3.5-turbo, demonstrated GPT-4's enhanced adaptability to the game environment, with significant improvements in posing relevant questions and forming human-like responses. However, challenges such as the model;s limitations in bluffing and predicting opponent moves emerged. Reflections on game development, financial constraints, and non-verbal limitations of the study were also discussed. The findings suggest that while GPT-4 exhibits promising advancements over earlier models, there remains potential for further development, especially in instilling more human-like attributes in AI.

{{</citation>}}


### (36/122) Dataset Factory: A Toolchain For Generative Computer Vision Datasets (Daniel Kharitonov et al., 2023)

{{<citation>}}

Daniel Kharitonov, Ryan Turner. (2023)  
**Dataset Factory: A Toolchain For Generative Computer Vision Datasets**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Computer Vision, Generative AI  
[Paper Link](http://arxiv.org/abs/2309.11608v1)  

---


**ABSTRACT**  
Generative AI workflows heavily rely on data-centric tasks - such as filtering samples by annotation fields, vector distances, or scores produced by custom classifiers. At the same time, computer vision datasets are quickly approaching petabyte volumes, rendering data wrangling difficult. In addition, the iterative nature of data preparation necessitates robust dataset sharing and versioning mechanisms, both of which are hard to implement ad-hoc. To solve these challenges, we propose a "dataset factory" approach that separates the storage and processing of samples from metadata and enables data-centric operations at scale for machine learning teams and individual researchers.

{{</citation>}}


### (37/122) BTLM-3B-8K: 7B Parameter Performance in a 3B Parameter Model (Nolan Dey et al., 2023)

{{<citation>}}

Nolan Dey, Daria Soboleva, Faisal Al-Khateeb, Bowen Yang, Ribhu Pathria, Hemant Khachane, Shaheer Muhammad, Zhiming, Chen, Robert Myers, Jacob Robert Steeves, Natalia Vassilieva, Marvin Tom, Joel Hestness. (2023)  
**BTLM-3B-8K: 7B Parameter Performance in a 3B Parameter Model**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.11568v1)  

---


**ABSTRACT**  
We introduce the Bittensor Language Model, called "BTLM-3B-8K", a new state-of-the-art 3 billion parameter open-source language model. BTLM-3B-8K was trained on 627B tokens from the SlimPajama dataset with a mixture of 2,048 and 8,192 context lengths. BTLM-3B-8K outperforms all existing 3B parameter models by 2-5.5% across downstream tasks. BTLM-3B-8K is even competitive with some 7B parameter models. Additionally, BTLM-3B-8K provides excellent long context performance, outperforming MPT-7B-8K and XGen-7B-8K on tasks up to 8,192 context length. We trained the model on a cleaned and deduplicated SlimPajama dataset; aggressively tuned the \textmu P hyperparameters and schedule; used ALiBi position embeddings; and adopted the SwiGLU nonlinearity.   On Hugging Face, the most popular models have 7B parameters, indicating that users prefer the quality-size ratio of 7B models. Compacting the 7B parameter model to one with 3B parameters, with little performance impact, is an important milestone. BTLM-3B-8K needs only 3GB of memory with 4-bit precision and takes 2.5x less inference compute than 7B models, helping to open up access to a powerful language model on mobile and edge devices. BTLM-3B-8K is available under an Apache 2.0 license on Hugging Face: https://huggingface.co/cerebras/btlm-3b-8k-base.

{{</citation>}}


### (38/122) Fictional Worlds, Real Connections: Developing Community Storytelling Social Chatbots through LLMs (Yuqian Sun et al., 2023)

{{<citation>}}

Yuqian Sun, Hanyi Wang, Pok Man Chan, Morteza Tabibi, Yan Zhang, Huan Lu, Yuheng Chen, Chang Hee Lee, Ali Asadipour. (2023)  
**Fictional Worlds, Real Connections: Developing Community Storytelling Social Chatbots through LLMs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.11478v1)  

---


**ABSTRACT**  
We address the integration of storytelling and Large Language Models (LLMs) to develop engaging and believable Social Chatbots (SCs) in community settings. Motivated by the potential of fictional characters to enhance social interactions, we introduce Storytelling Social Chatbots (SSCs) and the concept of story engineering to transform fictional game characters into "live" social entities within player communities. Our story engineering process includes three steps: (1) Character and story creation, defining the SC's personality and worldview, (2) Presenting Live Stories to the Community, allowing the SC to recount challenges and seek suggestions, and (3) Communication with community members, enabling interaction between the SC and users. We employed the LLM GPT-3 to drive our SSC prototypes, "David" and "Catherine," and evaluated their performance in an online gaming community, "DE (Alias)," on Discord. Our mixed-method analysis, based on questionnaires (N=15) and interviews (N=8) with community members, reveals that storytelling significantly enhances the engagement and believability of SCs in community settings.

{{</citation>}}


### (39/122) Multi-view Fuzzy Representation Learning with Rules based Model (Wei Zhang et al., 2023)

{{<citation>}}

Wei Zhang, Zhaohong Deng, Te Zhang, Kup-Sze Choi, Shitong Wang. (2023)  
**Multi-view Fuzzy Representation Learning with Rules based Model**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.11473v1)  

---


**ABSTRACT**  
Unsupervised multi-view representation learning has been extensively studied for mining multi-view data. However, some critical challenges remain. On the one hand, the existing methods cannot explore multi-view data comprehensively since they usually learn a common representation between views, given that multi-view data contains both the common information between views and the specific information within each view. On the other hand, to mine the nonlinear relationship between data, kernel or neural network methods are commonly used for multi-view representation learning. However, these methods are lacking in interpretability. To this end, this paper proposes a new multi-view fuzzy representation learning method based on the interpretable Takagi-Sugeno-Kang (TSK) fuzzy system (MVRL_FS). The method realizes multi-view representation learning from two aspects. First, multi-view data are transformed into a high-dimensional fuzzy feature space, while the common information between views and specific information of each view are explored simultaneously. Second, a new regularization method based on L_(2,1)-norm regression is proposed to mine the consistency information between views, while the geometric structure of the data is preserved through the Laplacian graph. Finally, extensive experiments on many benchmark multi-view datasets are conducted to validate the superiority of the proposed method.

{{</citation>}}


### (40/122) Generative Agent-Based Modeling: Unveiling Social System Dynamics through Coupling Mechanistic Models with Generative Artificial Intelligence (Navid Ghaffarzadegan et al., 2023)

{{<citation>}}

Navid Ghaffarzadegan, Aritra Majumdar, Ross Williams, Niyousha Hosseinichimeh. (2023)  
**Generative Agent-Based Modeling: Unveiling Social System Dynamics through Coupling Mechanistic Models with Generative Artificial Intelligence**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-MA, cs.AI, nlin-AO, physics-soc-ph  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.11456v1)  

---


**ABSTRACT**  
We discuss the emerging new opportunity for building feedback-rich computational models of social systems using generative artificial intelligence. Referred to as Generative Agent-Based Models (GABMs), such individual-level models utilize large language models such as ChatGPT to represent human decision-making in social settings. We provide a GABM case in which human behavior can be incorporated in simulation models by coupling a mechanistic model of human interactions with a pre-trained large language model. This is achieved by introducing a simple GABM of social norm diffusion in an organization. For educational purposes, the model is intentionally kept simple. We examine a wide range of scenarios and the sensitivity of the results to several changes in the prompt. We hope the article and the model serve as a guide for building useful diffusion models that include realistic human reasoning and decision-making.

{{</citation>}}


### (41/122) Using deep learning to construct stochastic local search SAT solvers with performance bounds (Maximilian Kramer et al., 2023)

{{<citation>}}

Maximilian Kramer, Paul Boes. (2023)  
**Using deep learning to construct stochastic local search SAT solvers with performance bounds**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, math-OC  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.11452v2)  

---


**ABSTRACT**  
The Boolean Satisfiability problem (SAT) is the most prototypical NP-complete problem and of great practical relevance. One important class of solvers for this problem are stochastic local search (SLS) algorithms that iteratively and randomly update a candidate assignment. Recent breakthrough results in theoretical computer science have established sufficient conditions under which SLS solvers are guaranteed to efficiently solve a SAT instance, provided they have access to suitable "oracles" that provide samples from an instance-specific distribution, exploiting an instance's local structure. Motivated by these results and the well established ability of neural networks to learn common structure in large datasets, in this work, we train oracles using Graph Neural Networks and evaluate them on two SLS solvers on random SAT instances of varying difficulty. We find that access to GNN-based oracles significantly boosts the performance of both solvers, allowing them, on average, to solve 17% more difficult instances (as measured by the ratio between clauses and variables), and to do so in 35% fewer steps, with improvements in the median number of steps of up to a factor of 8. As such, this work bridges formal results from theoretical computer science and practically motivated research on deep learning for constraint satisfaction problems and establishes the promise of purpose-trained SAT solvers with performance guarantees.

{{</citation>}}


### (42/122) SCREWS: A Modular Framework for Reasoning with Revisions (Kumar Shridhar et al., 2023)

{{<citation>}}

Kumar Shridhar, Harsh Jhamtani, Hao Fang, Benjamin Van Durme, Jason Eisner, Patrick Xia. (2023)  
**SCREWS: A Modular Framework for Reasoning with Revisions**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: ChatGPT, GPT, GPT-4, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.13075v1)  

---


**ABSTRACT**  
Large language models (LLMs) can improve their accuracy on various tasks through iteratively refining and revising their output based on feedback. We observe that these revisions can introduce errors, in which case it is better to roll back to a previous result. Further, revisions are typically homogeneous: they use the same reasoning method that produced the initial answer, which may not correct errors. To enable exploration in this space, we present SCREWS, a modular framework for reasoning with revisions. It is comprised of three main modules: Sampling, Conditional Resampling, and Selection, each consisting of sub-modules that can be hand-selected per task. We show that SCREWS not only unifies several previous approaches under a common framework, but also reveals several novel strategies for identifying improved reasoning chains. We evaluate our framework with state-of-the-art LLMs (ChatGPT and GPT-4) on a diverse set of reasoning tasks and uncover useful new reasoning strategies for each: arithmetic word problems, multi-hop question answering, and code debugging. Heterogeneous revision strategies prove to be important, as does selection between original and revised candidates.

{{</citation>}}


### (43/122) Knowledge Graph Question Answering for Materials Science (KGQA4MAT): Developing Natural Language Interface for Metal-Organic Frameworks Knowledge Graph (MOF-KG) (Yuan An et al., 2023)

{{<citation>}}

Yuan An, Jane Greenberg, Alex Kalinowski, Xintong Zhao, Xiaohua Hu, Fernando J. Uribe-Romo, Kyle Langlois, Jacob Furst, Diego A. Gómez-Gualdrón. (2023)  
**Knowledge Graph Question Answering for Materials Science (KGQA4MAT): Developing Natural Language Interface for Metal-Organic Frameworks Knowledge Graph (MOF-KG)**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: ChatGPT, GPT, Knowledge Graph, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.11361v1)  

---


**ABSTRACT**  
We present a comprehensive benchmark dataset for Knowledge Graph Question Answering in Materials Science (KGQA4MAT), with a focus on metal-organic frameworks (MOFs). A knowledge graph for metal-organic frameworks (MOF-KG) has been constructed by integrating structured databases and knowledge extracted from the literature. To enhance MOF-KG accessibility for domain experts, we aim to develop a natural language interface for querying the knowledge graph. We have developed a benchmark comprised of 161 complex questions involving comparison, aggregation, and complicated graph structures. Each question is rephrased in three additional variations, resulting in 644 questions and 161 KG queries. To evaluate the benchmark, we have developed a systematic approach for utilizing ChatGPT to translate natural language questions into formal KG queries. We also apply the approach to the well-known QALD-9 dataset, demonstrating ChatGPT's potential in addressing KGQA issues for different platforms and query languages. The benchmark and the proposed approach aim to stimulate further research and development of user-friendly and efficient interfaces for querying domain-specific materials science knowledge graphs, thereby accelerating the discovery of novel materials.

{{</citation>}}


### (44/122) ChatGPT-4 as a Tool for Reviewing Academic Books in Spanish (Jonnathan Berrezueta-Guzman et al., 2023)

{{<citation>}}

Jonnathan Berrezueta-Guzman, Laura Malache-Silva, Stephan Krusche. (2023)  
**ChatGPT-4 as a Tool for Reviewing Academic Books in Spanish**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.11231v1)  

---


**ABSTRACT**  
This study evaluates the potential of ChatGPT-4, an artificial intelligence language model developed by OpenAI, as an editing tool for Spanish literary and academic books. The need for efficient and accessible reviewing and editing processes in the publishing industry has driven the search for automated solutions. ChatGPT-4, being one of the most advanced language models, offers notable capabilities in text comprehension and generation. In this study, the features and capabilities of ChatGPT-4 are analyzed in terms of grammatical correction, stylistic coherence, and linguistic enrichment of texts in Spanish. Tests were conducted with 100 literary and academic texts, where the edits made by ChatGPT-4 were compared to those made by expert human reviewers and editors. The results show that while ChatGPT-4 is capable of making grammatical and orthographic corrections with high accuracy and in a very short time, it still faces challenges in areas such as context sensitivity, bibliometric analysis, deep contextual understanding, and interaction with visual content like graphs and tables. However, it is observed that collaboration between ChatGPT-4 and human reviewers and editors can be a promising strategy for improving efficiency without compromising quality. Furthermore, the authors consider that ChatGPT-4 represents a valuable tool in the editing process, but its use should be complementary to the work of human editors to ensure high-caliber editing in Spanish literary and academic books.

{{</citation>}}


### (45/122) Exploring the Relationship between LLM Hallucinations and Prompt Linguistic Nuances: Readability, Formality, and Concreteness (Vipula Rawte et al., 2023)

{{<citation>}}

Vipula Rawte, Prachi Priya, S. M Towhidul Islam Tonmoy, S M Mehedi Zaman, Amit Sheth, Amitava Das. (2023)  
**Exploring the Relationship between LLM Hallucinations and Prompt Linguistic Nuances: Readability, Formality, and Concreteness**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.11064v1)  

---


**ABSTRACT**  
As Large Language Models (LLMs) have advanced, they have brought forth new challenges, with one of the prominent issues being LLM hallucination. While various mitigation techniques are emerging to address hallucination, it is equally crucial to delve into its underlying causes. Consequently, in this preliminary exploratory investigation, we examine how linguistic factors in prompts, specifically readability, formality, and concreteness, influence the occurrence of hallucinations. Our experimental results suggest that prompts characterized by greater formality and concreteness tend to result in reduced hallucination. However, the outcomes pertaining to readability are somewhat inconclusive, showing a mixed pattern.

{{</citation>}}


### (46/122) Is GPT4 a Good Trader? (Bingzhe Wu, 2023)

{{<citation>}}

Bingzhe Wu. (2023)  
**Is GPT4 a Good Trader?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.10982v1)  

---


**ABSTRACT**  
Recently, large language models (LLMs), particularly GPT-4, have demonstrated significant capabilities in various planning and reasoning tasks \cite{cheng2023gpt4,bubeck2023sparks}. Motivated by these advancements, there has been a surge of interest among researchers to harness the capabilities of GPT-4 for the automated design of quantitative factors that do not overlap with existing factor libraries, with an aspiration to achieve alpha returns \cite{webpagequant}. In contrast to these work, this study aims to examine the fidelity of GPT-4's comprehension of classic trading theories and its proficiency in applying its code interpreter abilities to real-world trading data analysis. Such an exploration is instrumental in discerning whether the underlying logic GPT-4 employs for trading is intrinsically reliable. Furthermore, given the acknowledged interpretative latitude inherent in most trading theories, we seek to distill more precise methodologies of deploying these theories from GPT-4's analytical process, potentially offering invaluable insights to human traders.   To achieve this objective, we selected daily candlestick (K-line) data from specific periods for certain assets, such as the Shanghai Stock Index. Through meticulous prompt engineering, we guided GPT-4 to analyze the technical structures embedded within this data, based on specific theories like the Elliott Wave Theory. We then subjected its analytical output to manual evaluation, assessing its interpretative depth and accuracy vis-\`a-vis these trading theories from multiple dimensions. The results and findings from this study could pave the way for a synergistic amalgamation of human expertise and AI-driven insights in the realm of trading.

{{</citation>}}


## cs.LG (21)



### (47/122) Large-scale Pretraining Improves Sample Efficiency of Active Learning based Molecule Virtual Screening (Zhonglin Cao et al., 2023)

{{<citation>}}

Zhonglin Cao, Simone Sciabola, Ye Wang. (2023)  
**Large-scale Pretraining Improves Sample Efficiency of Active Learning based Molecule Virtual Screening**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-BM  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2309.11687v1)  

---


**ABSTRACT**  
Virtual screening of large compound libraries to identify potential hit candidates is one of the earliest steps in drug discovery. As the size of commercially available compound collections grows exponentially to the scale of billions, brute-force virtual screening using traditional tools such as docking becomes infeasible in terms of time and computational resources. Active learning and Bayesian optimization has recently been proven as effective methods of narrowing down the search space. An essential component in those methods is a surrogate machine learning model that is trained with a small subset of the library to predict the desired properties of compounds. Accurate model can achieve high sample efficiency by finding the most promising compounds with only a fraction of the whole library being virtually screened. In this study, we examined the performance of pretrained transformer-based language model and graph neural network in Bayesian optimization active learning framework. The best pretrained models identifies 58.97% of the top-50000 by docking score after screening only 0.6% of an ultra-large library containing 99.5 million compounds, improving 8% over previous state-of-the-art baseline. Through extensive benchmarks, we show that the superior performance of pretrained models persists in both structure-based and ligand-based drug discovery. Such model can serve as a boost to the accuracy and sample efficiency of active learning based molecule virtual screening.

{{</citation>}}


### (48/122) Fairness Hub Technical Briefs: AUC Gap (Jinsook Lee et al., 2023)

{{<citation>}}

Jinsook Lee, Chris Brooks, Renzhe Yu, Rene Kizilcec. (2023)  
**Fairness Hub Technical Briefs: AUC Gap**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.12371v1)  

---


**ABSTRACT**  
To measure bias, we encourage teams to consider using AUC Gap: the absolute difference between the highest and lowest test AUC for subgroups (e.g., gender, race, SES, prior knowledge). It is agnostic to the AI/ML algorithm used and it captures the disparity in model performance for any number of subgroups, which enables non-binary fairness assessments such as for intersectional identity groups. The LEVI teams use a wide range of AI/ML models in pursuit of a common goal of doubling math achievement in low-income middle schools. Ensuring that the models, which are trained on datasets collected in many different contexts, do not introduce or amplify biases is important for achieving the LEVI goal. We offer here a versatile and easy-to-compute measure of model bias for all LEVI teams in order to create a common benchmark and an analytical basis for sharing what strategies have worked for different teams.

{{</citation>}}


### (49/122) CATS: Conditional Adversarial Trajectory Synthesis for Privacy-Preserving Trajectory Data Publication Using Deep Learning Approaches (Jinmeng Rao et al., 2023)

{{<citation>}}

Jinmeng Rao, Song Gao, Sijia Zhu. (2023)  
**CATS: Conditional Adversarial Trajectory Synthesis for Privacy-Preserving Trajectory Data Publication Using Deep Learning Approaches**  

---
Primary Category: cs.LG  
Categories: I-2, cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.11587v1)  

---


**ABSTRACT**  
The prevalence of ubiquitous location-aware devices and mobile Internet enables us to collect massive individual-level trajectory dataset from users. Such trajectory big data bring new opportunities to human mobility research but also raise public concerns with regard to location privacy. In this work, we present the Conditional Adversarial Trajectory Synthesis (CATS), a deep-learning-based GeoAI methodological framework for privacy-preserving trajectory data generation and publication. CATS applies K-anonymity to the underlying spatiotemporal distributions of human movements, which provides a distributional-level strong privacy guarantee. By leveraging conditional adversarial training on K-anonymized human mobility matrices, trajectory global context learning using the attention-based mechanism, and recurrent bipartite graph matching of adjacent trajectory points, CATS is able to reconstruct trajectory topology from conditionally sampled locations and generate high-quality individual-level synthetic trajectory data, which can serve as supplements or alternatives to raw data for privacy-preserving trajectory data publication. The experiment results on over 90k GPS trajectories show that our method has a better performance in privacy preservation, spatiotemporal characteristic preservation, and downstream utility compared with baseline methods, which brings new insights into privacy-preserving human mobility research using generative AI techniques and explores data ethics issues in GIScience.

{{</citation>}}


### (50/122) Text2Reward: Automated Dense Reward Function Generation for Reinforcement Learning (Tianbao Xie et al., 2023)

{{<citation>}}

Tianbao Xie, Siheng Zhao, Chen Henry Wu, Yitao Liu, Qian Luo, Victor Zhong, Yanchao Yang, Tao Yu. (2023)  
**Text2Reward: Automated Dense Reward Function Generation for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.11489v2)  

---


**ABSTRACT**  
Designing reward functions is a longstanding challenge in reinforcement learning (RL); it requires specialized knowledge or domain data, leading to high costs for development. To address this, we introduce Text2Reward, a data-free framework that automates the generation of dense reward functions based on large language models (LLMs). Given a goal described in natural language, Text2Reward generates dense reward functions as an executable program grounded in a compact representation of the environment. Unlike inverse RL and recent work that uses LLMs to write sparse reward codes, Text2Reward produces interpretable, free-form dense reward codes that cover a wide range of tasks, utilize existing packages, and allow iterative refinement with human feedback. We evaluate Text2Reward on two robotic manipulation benchmarks (ManiSkill2, MetaWorld) and two locomotion environments of MuJoCo. On 13 of the 17 manipulation tasks, policies trained with generated reward codes achieve similar or better task success rates and convergence speed than expert-written reward codes. For locomotion tasks, our method learns six novel locomotion behaviors with a success rate exceeding 94%. Furthermore, we show that the policies trained in the simulator with our method can be deployed in the real world. Finally, Text2Reward further improves the policies by refining their reward functions with human feedback. Video results are available at https://text-to-reward.github.io

{{</citation>}}


### (51/122) Weight Averaging Improves Knowledge Distillation under Domain Shift (Valeriy Berezovskiy et al., 2023)

{{<citation>}}

Valeriy Berezovskiy, Nikita Morozov. (2023)  
**Weight Averaging Improves Knowledge Distillation under Domain Shift**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.11446v1)  

---


**ABSTRACT**  
Knowledge distillation (KD) is a powerful model compression technique broadly used in practical deep learning applications. It is focused on training a small student network to mimic a larger teacher network. While it is widely known that KD can offer an improvement to student generalization in i.i.d setting, its performance under domain shift, i.e. the performance of student networks on data from domains unseen during training, has received little attention in the literature. In this paper we make a step towards bridging the research fields of knowledge distillation and domain generalization. We show that weight averaging techniques proposed in domain generalization literature, such as SWAD and SMA, also improve the performance of knowledge distillation under domain shift. In addition, we propose a simplistic weight averaging strategy that does not require evaluation on validation data during training and show that it performs on par with SWAD and SMA when applied to KD. We name our final distillation approach Weight-Averaged Knowledge Distillation (WAKD).

{{</citation>}}


### (52/122) Generative Pre-Training of Time-Series Data for Unsupervised Fault Detection in Semiconductor Manufacturing (Sewoong Lee et al., 2023)

{{<citation>}}

Sewoong Lee, JinKyou Choi, Min Su Kim. (2023)  
**Generative Pre-Training of Time-Series Data for Unsupervised Fault Detection in Semiconductor Manufacturing**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Embedding, GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.11427v1)  

---


**ABSTRACT**  
This paper introduces TRACE-GPT, which stands for Time-seRies Anomaly-detection with Convolutional Embedding and Generative Pre-trained Transformers. TRACE-GPT is designed to pre-train univariate time-series sensor data and detect faults on unlabeled datasets in semiconductor manufacturing. In semiconductor industry, classifying abnormal time-series sensor data from normal data is important because it is directly related to wafer defect. However, small, unlabeled, and even mixed training data without enough anomalies make classification tasks difficult. In this research, we capture features of time-series data with temporal convolutional embedding and Generative Pre-trained Transformer (GPT) to classify abnormal sequences from normal sequences using cross entropy loss. We prove that our model shows better performance than previous unsupervised models with both an open dataset, the University of California Riverside (UCR) time-series classification archive, and the process log of our Chemical Vapor Deposition (CVD) equipment. Our model has the highest F1 score at Equal Error Rate (EER) across all datasets and is only 0.026 below the supervised state-of-the-art baseline on the open dataset.

{{</citation>}}


### (53/122) Improving Article Classification with Edge-Heterogeneous Graph Neural Networks (Khang Ly et al., 2023)

{{<citation>}}

Khang Ly, Yury Kashnitsky, Savvas Chamezopoulos, Valeria Krzhizhanovskaya. (2023)  
**Improving Article Classification with Edge-Heterogeneous Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: BERT, GNN, Graph Neural Network, Graph Neural Networks, Microsoft  
[Paper Link](http://arxiv.org/abs/2309.11341v1)  

---


**ABSTRACT**  
Classifying research output into context-specific label taxonomies is a challenging and relevant downstream task, given the volume of existing and newly published articles. We propose a method to enhance the performance of article classification by enriching simple Graph Neural Networks (GNN) pipelines with edge-heterogeneous graph representations. SciBERT is used for node feature generation to capture higher-order semantics within the articles' textual metadata. Fully supervised transductive node classification experiments are conducted on the Open Graph Benchmark (OGB) ogbn-arxiv dataset and the PubMed diabetes dataset, augmented with additional metadata from Microsoft Academic Graph (MAG) and PubMed Central, respectively. The results demonstrate that edge-heterogeneous graphs consistently improve the performance of all GNN models compared to the edge-homogeneous graphs. The transformed data enable simple and shallow GNN pipelines to achieve results on par with more complex architectures. On ogbn-arxiv, we achieve a top-15 result in the OGB competition with a 2-layer GCN (accuracy 74.61%), being the highest-scoring solution with sub-1 million parameters. On PubMed, we closely trail SOTA GNN architectures using a 2-layer GraphSAGE by including additional co-authorship edges in the graph (accuracy 89.88%). The implementation is available at: $\href{https://github.com/lyvykhang/edgehetero-nodeproppred}{\text{https://github.com/lyvykhang/edgehetero-nodeproppred}}$.

{{</citation>}}


### (54/122) WFTNet: Exploiting Global and Local Periodicity in Long-term Time Series Forecasting (Peiyuan Liu et al., 2023)

{{<citation>}}

Peiyuan Liu, Beiliang Wu, Naiqi Li, Tao Dai, Fengmao Lei, Jigang Bao, Yong Jiang, Shu-Tao Xia. (2023)  
**WFTNet: Exploiting Global and Local Periodicity in Long-term Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2309.11319v1)  

---


**ABSTRACT**  
Recent CNN and Transformer-based models tried to utilize frequency and periodicity information for long-term time series forecasting. However, most existing work is based on Fourier transform, which cannot capture fine-grained and local frequency structure. In this paper, we propose a Wavelet-Fourier Transform Network (WFTNet) for long-term time series forecasting. WFTNet utilizes both Fourier and wavelet transforms to extract comprehensive temporal-frequency information from the signal, where Fourier transform captures the global periodic patterns and wavelet transform captures the local ones. Furthermore, we introduce a Periodicity-Weighted Coefficient (PWC) to adaptively balance the importance of global and local frequency patterns. Extensive experiments on various time series datasets show that WFTNet consistently outperforms other state-of-the-art baseline.

{{</citation>}}


### (55/122) Beyond Accuracy: Measuring Representation Capacity of Embeddings to Preserve Structural and Contextual Information (Sarwan Ali, 2023)

{{<citation>}}

Sarwan Ali. (2023)  
**Beyond Accuracy: Measuring Representation Capacity of Embeddings to Preserve Structural and Contextual Information**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.11294v1)  

---


**ABSTRACT**  
Effective representation of data is crucial in various machine learning tasks, as it captures the underlying structure and context of the data. Embeddings have emerged as a powerful technique for data representation, but evaluating their quality and capacity to preserve structural and contextual information remains a challenge. In this paper, we address this need by proposing a method to measure the \textit{representation capacity} of embeddings. The motivation behind this work stems from the importance of understanding the strengths and limitations of embeddings, enabling researchers and practitioners to make informed decisions in selecting appropriate embedding models for their specific applications. By combining extrinsic evaluation methods, such as classification and clustering, with t-SNE-based neighborhood analysis, such as neighborhood agreement and trustworthiness, we provide a comprehensive assessment of the representation capacity. Additionally, the use of optimization techniques (bayesian optimization) for weight optimization (for classification, clustering, neighborhood agreement, and trustworthiness) ensures an objective and data-driven approach in selecting the optimal combination of metrics. The proposed method not only contributes to advancing the field of embedding evaluation but also empowers researchers and practitioners with a quantitative measure to assess the effectiveness of embeddings in capturing structural and contextual information. For the evaluation, we use $3$ real-world biological sequence (proteins and nucleotide) datasets and performed representation capacity analysis of $4$ embedding methods from the literature, namely Spike2Vec, Spaced $k$-mers, PWM2Vec, and AutoEncoder.

{{</citation>}}


### (56/122) Hierarchical Multi-Agent Reinforcement Learning for Air Combat Maneuvering (Ardian Selmonaj et al., 2023)

{{<citation>}}

Ardian Selmonaj, Oleg Szehr, Giacomo Del Rio, Alessandro Antonucci, Adrian Schneider, Michael Rüegsegger. (2023)  
**Hierarchical Multi-Agent Reinforcement Learning for Air Combat Maneuvering**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.11247v1)  

---


**ABSTRACT**  
The application of artificial intelligence to simulate air-to-air combat scenarios is attracting increasing attention. To date the high-dimensional state and action spaces, the high complexity of situation information (such as imperfect and filtered information, stochasticity, incomplete knowledge about mission targets) and the nonlinear flight dynamics pose significant challenges for accurate air combat decision-making. These challenges are exacerbated when multiple heterogeneous agents are involved. We propose a hierarchical multi-agent reinforcement learning framework for air-to-air combat with multiple heterogeneous agents. In our framework, the decision-making process is divided into two stages of abstraction, where heterogeneous low-level policies control the action of individual units, and a high-level commander policy issues macro commands given the overall mission targets. Low-level policies are trained for accurate unit combat control. Their training is organized in a learning curriculum with increasingly complex training scenarios and league-based self-play. The commander policy is trained on mission targets given pre-trained low-level policies. The empirical validation advocates the advantages of our design choices.

{{</citation>}}


### (57/122) The Languini Kitchen: Enabling Language Modelling Research at Different Scales of Compute (Aleksandar Stanić et al., 2023)

{{<citation>}}

Aleksandar Stanić, Dylan Ashley, Oleg Serikov, Louis Kirsch, Francesco Faccio, Jürgen Schmidhuber, Thomas Hofmann, Imanol Schlag. (2023)  
**The Languini Kitchen: Enabling Language Modelling Research at Different Scales of Compute**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: GPT, LSTM, Language Model  
[Paper Link](http://arxiv.org/abs/2309.11197v1)  

---


**ABSTRACT**  
The Languini Kitchen serves as both a research collective and codebase designed to empower researchers with limited computational resources to contribute meaningfully to the field of language modelling. We introduce an experimental protocol that enables model comparisons based on equivalent compute, measured in accelerator hours. The number of tokens on which a model is trained is defined by the model's throughput and the chosen compute class. Notably, this approach avoids constraints on critical hyperparameters which affect total parameters or floating-point operations. For evaluation, we pre-process an existing large, diverse, and high-quality dataset of books that surpasses existing academic benchmarks in quality, diversity, and document length. On it, we compare methods based on their empirical scaling trends which are estimated through experiments at various levels of compute. This work also provides two baseline models: a feed-forward model derived from the GPT-2 architecture and a recurrent model in the form of a novel LSTM with ten-fold throughput. While the GPT baseline achieves better perplexity throughout all our levels of compute, our LSTM baseline exhibits a predictable and more favourable scaling law. This is due to the improved throughput and the need for fewer training tokens to achieve the same decrease in test perplexity. Extrapolating the scaling laws leads of both models results in an intersection at roughly 50,000 accelerator hours. We hope this work can serve as the foundation for meaningful and reproducible language modelling research.

{{</citation>}}


### (58/122) When to Trust AI: Advances and Challenges for Certification of Neural Networks (Marta Kwiatkowska et al., 2023)

{{<citation>}}

Marta Kwiatkowska, Xiyue Zhang. (2023)  
**When to Trust AI: Advances and Challenges for Certification of Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs-SC, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.11196v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) has been advancing at a fast pace and it is now poised for deployment in a wide range of applications, such as autonomous systems, medical diagnosis and natural language processing. Early adoption of AI technology for real-world applications has not been without problems, particularly for neural networks, which may be unstable and susceptible to adversarial examples. In the longer term, appropriate safety assurance techniques need to be developed to reduce potential harm due to avoidable system failures and ensure trustworthiness. Focusing on certification and explainability, this paper provides an overview of techniques that have been developed to ensure safety of AI decisions and discusses future challenges.

{{</citation>}}


### (59/122) Delays in Reinforcement Learning (Pierre Liotet, 2023)

{{<citation>}}

Pierre Liotet. (2023)  
**Delays in Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.11096v1)  

---


**ABSTRACT**  
Delays are inherent to most dynamical systems. Besides shifting the process in time, they can significantly affect their performance. For this reason, it is usually valuable to study the delay and account for it. Because they are dynamical systems, it is of no surprise that sequential decision-making problems such as Markov decision processes (MDP) can also be affected by delays. These processes are the foundational framework of reinforcement learning (RL), a paradigm whose goal is to create artificial agents capable of learning to maximise their utility by interacting with their environment.   RL has achieved strong, sometimes astonishing, empirical results, but delays are seldom explicitly accounted for. The understanding of the impact of delay on the MDP is limited. In this dissertation, we propose to study the delay in the agent's observation of the state of the environment or in the execution of the agent's actions. We will repeatedly change our point of view on the problem to reveal some of its structure and peculiarities. A wide spectrum of delays will be considered, and potential solutions will be presented. This dissertation also aims to draw links between celebrated frameworks of the RL literature and the one of delays.

{{</citation>}}


### (60/122) InkStream: Real-time GNN Inference on Streaming Graphs via Incremental Update (Dan Wu et al., 2023)

{{<citation>}}

Dan Wu, Zhaoying Li, Tulika Mitra. (2023)  
**InkStream: Real-time GNN Inference on Streaming Graphs via Incremental Update**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2309.11071v1)  

---


**ABSTRACT**  
Classic Graph Neural Network (GNN) inference approaches, designed for static graphs, are ill-suited for streaming graphs that evolve with time. The dynamism intrinsic to streaming graphs necessitates constant updates, posing unique challenges to acceleration on GPU. We address these challenges based on two key insights: (1) Inside the $k$-hop neighborhood, a significant fraction of the nodes is not impacted by the modified edges when the model uses min or max as aggregation function; (2) When the model weights remain static while the graph structure changes, node embeddings can incrementally evolve over time by computing only the impacted part of the neighborhood. With these insights, we propose a novel method, InkStream, designed for real-time inference with minimal memory access and computation, while ensuring an identical output to conventional methods. InkStream operates on the principle of propagating and fetching data only when necessary. It uses an event-based system to control inter-layer effect propagation and intra-layer incremental updates of node embedding. InkStream is highly extensible and easily configurable by allowing users to create and process customized events. We showcase that less than 10 lines of additional user code are needed to support popular GNN models such as GCN, GraphSAGE, and GIN. Our experiments with three GNN models on four large graphs demonstrate that InkStream accelerates by 2.5-427$\times$ on a CPU cluster and 2.4-343$\times$ on two different GPU clusters while producing identical outputs as GNN model inference on the latest graph snapshot.

{{</citation>}}


### (61/122) Clustered FedStack: Intermediate Global Models with Bayesian Information Criterion (Thanveer Shaik et al., 2023)

{{<citation>}}

Thanveer Shaik, Xiaohui Tao, Lin Li, Niall Higgins, Raj Gururajan, Xujuan Zhou, Jianming Yong. (2023)  
**Clustered FedStack: Intermediate Global Models with Bayesian Information Criterion**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.11044v1)  

---


**ABSTRACT**  
Federated Learning (FL) is currently one of the most popular technologies in the field of Artificial Intelligence (AI) due to its collaborative learning and ability to preserve client privacy. However, it faces challenges such as non-identically and non-independently distributed (non-IID) and data with imbalanced labels among local clients. To address these limitations, the research community has explored various approaches such as using local model parameters, federated generative adversarial learning, and federated representation learning. In our study, we propose a novel Clustered FedStack framework based on the previously published Stacked Federated Learning (FedStack) framework. The local clients send their model predictions and output layer weights to a server, which then builds a robust global model. This global model clusters the local clients based on their output layer weights using a clustering mechanism. We adopt three clustering mechanisms, namely K-Means, Agglomerative, and Gaussian Mixture Models, into the framework and evaluate their performance. We use Bayesian Information Criterion (BIC) with the maximum likelihood function to determine the number of clusters. The Clustered FedStack models outperform baseline models with clustering mechanisms. To estimate the convergence of our proposed framework, we use Cyclical learning rates.

{{</citation>}}


### (62/122) Conformalized Multimodal Uncertainty Regression and Reasoning (Domenico Parente et al., 2023)

{{<citation>}}

Domenico Parente, Nastaran Darabi, Alex C. Stutts, Theja Tulabandhula, Amit Ranjan Trivedi. (2023)  
**Conformalized Multimodal Uncertainty Regression and Reasoning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs-RO, cs.LG  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2309.11018v1)  

---


**ABSTRACT**  
This paper introduces a lightweight uncertainty estimator capable of predicting multimodal (disjoint) uncertainty bounds by integrating conformal prediction with a deep-learning regressor. We specifically discuss its application for visual odometry (VO), where environmental features such as flying domain symmetries and sensor measurements under ambiguities and occlusion can result in multimodal uncertainties. Our simulation results show that uncertainty estimates in our framework adapt sample-wise against challenging operating conditions such as pronounced noise, limited training data, and limited parametric size of the prediction model. We also develop a reasoning framework that leverages these robust uncertainty estimates and incorporates optical flow-based reasoning to improve prediction prediction accuracy. Thus, by appropriately accounting for predictive uncertainties of data-driven learning and closing their estimation loop via rule-based reasoning, our methodology consistently surpasses conventional deep learning approaches on all these challenging scenarios--pronounced noise, limited training data, and limited model size-reducing the prediction error by 2-3x.

{{</citation>}}


### (63/122) AI-Driven Patient Monitoring with Multi-Agent Deep Reinforcement Learning (Thanveer Shaik et al., 2023)

{{<citation>}}

Thanveer Shaik, Xiaohui Tao, Haoran Xie, Lin Li, Jianming Yong, Hong-Ning Dai. (2023)  
**AI-Driven Patient Monitoring with Multi-Agent Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.10980v2)  

---


**ABSTRACT**  
Effective patient monitoring is vital for timely interventions and improved healthcare outcomes. Traditional monitoring systems often struggle to handle complex, dynamic environments with fluctuating vital signs, leading to delays in identifying critical conditions. To address this challenge, we propose a novel AI-driven patient monitoring framework using multi-agent deep reinforcement learning (DRL). Our approach deploys multiple learning agents, each dedicated to monitoring a specific physiological feature, such as heart rate, respiration, and temperature. These agents interact with a generic healthcare monitoring environment, learn the patients' behavior patterns, and make informed decisions to alert the corresponding Medical Emergency Teams (METs) based on the level of emergency estimated. In this study, we evaluate the performance of the proposed multi-agent DRL framework using real-world physiological and motion data from two datasets: PPG-DaLiA and WESAD. We compare the results with several baseline models, including Q-Learning, PPO, Actor-Critic, Double DQN, and DDPG, as well as monitoring frameworks like WISEML and CA-MAQL. Our experiments demonstrate that the proposed DRL approach outperforms all other baseline models, achieving more accurate monitoring of patient's vital signs. Furthermore, we conduct hyperparameter optimization to fine-tune the learning process of each agent. By optimizing hyperparameters, we enhance the learning rate and discount factor, thereby improving the agents' overall performance in monitoring patient health status. Our AI-driven patient monitoring system offers several advantages over traditional methods, including the ability to handle complex and uncertain environments, adapt to varying patient conditions, and make real-time decisions without external supervision.

{{</citation>}}


### (64/122) Towards Data-centric Graph Machine Learning: Review and Outlook (Xin Zheng et al., 2023)

{{<citation>}}

Xin Zheng, Yixin Liu, Zhifeng Bao, Meng Fang, Xia Hu, Alan Wee-Chung Liew, Shirui Pan. (2023)  
**Towards Data-centric Graph Machine Learning: Review and Outlook**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10979v1)  

---


**ABSTRACT**  
Data-centric AI, with its primary focus on the collection, management, and utilization of data to drive AI models and applications, has attracted increasing attention in recent years. In this article, we conduct an in-depth and comprehensive review, offering a forward-looking outlook on the current efforts in data-centric AI pertaining to graph data-the fundamental data structure for representing and capturing intricate dependencies among massive and diverse real-life entities. We introduce a systematic framework, Data-centric Graph Machine Learning (DC-GML), that encompasses all stages of the graph data lifecycle, including graph data collection, exploration, improvement, exploitation, and maintenance. A thorough taxonomy of each stage is presented to answer three critical graph-centric questions: (1) how to enhance graph data availability and quality; (2) how to learn from graph data with limited-availability and low-quality; (3) how to build graph MLOps systems from the graph data-centric view. Lastly, we pinpoint the future prospects of the DC-GML domain, providing insights to navigate its advancements and applications.

{{</citation>}}


### (65/122) PAGER: A Framework for Failure Analysis of Deep Regression Models (Jayaraman J. Thiagarajan et al., 2023)

{{<citation>}}

Jayaraman J. Thiagarajan, Vivek Narayanaswamy, Puja Trivedi, Rushil Anirudh. (2023)  
**PAGER: A Framework for Failure Analysis of Deep Regression Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10977v1)  

---


**ABSTRACT**  
Safe deployment of AI models requires proactive detection of potential prediction failures to prevent costly errors. While failure detection in classification problems has received significant attention, characterizing failure modes in regression tasks is more complicated and less explored. Existing approaches rely on epistemic uncertainties or feature inconsistency with the training distribution to characterize model risk. However, we show that uncertainties are necessary but insufficient to accurately characterize failure, owing to the various sources of error. In this paper, we propose PAGER (Principled Analysis of Generalization Errors in Regressors), a framework to systematically detect and characterize failures in deep regression models. Built upon the recently proposed idea of anchoring in deep models, PAGER unifies both epistemic uncertainties and novel, complementary non-conformity scores to organize samples into different risk regimes, thereby providing a comprehensive analysis of model errors. Additionally, we introduce novel metrics for evaluating failure detectors in regression tasks. We demonstrate the effectiveness of PAGER on synthetic and real-world benchmarks. Our results highlight the capability of PAGER to identify regions of accurate generalization and detect failure cases in out-of-distribution and out-of-support scenarios.

{{</citation>}}


### (66/122) Accurate and Scalable Estimation of Epistemic Uncertainty for Graph Neural Networks (Puja Trivedi et al., 2023)

{{<citation>}}

Puja Trivedi, Mark Heimann, Rushil Anirudh, Danai Koutra, Jayaraman J. Thiagarajan. (2023)  
**Accurate and Scalable Estimation of Epistemic Uncertainty for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.10976v1)  

---


**ABSTRACT**  
Safe deployment of graph neural networks (GNNs) under distribution shift requires models to provide accurate confidence indicators (CI). However, while it is well-known in computer vision that CI quality diminishes under distribution shift, this behavior remains understudied for GNNs. Hence, we begin with a case study on CI calibration under controlled structural and feature distribution shifts and demonstrate that increased expressivity or model size do not always lead to improved CI performance. Consequently, we instead advocate for the use of epistemic uncertainty quantification (UQ) methods to modulate CIs. To this end, we propose G-$\Delta$UQ, a new single model UQ method that extends the recently proposed stochastic centering framework to support structured data and partial stochasticity. Evaluated across covariate, concept, and graph size shifts, G-$\Delta$UQ not only outperforms several popular UQ methods in obtaining calibrated CIs, but also outperforms alternatives when CIs are used for generalization gap prediction or OOD detection. Overall, our work not only introduces a new, flexible GNN UQ method, but also provides novel insights into GNN CIs on safety-critical tasks.

{{</citation>}}


### (67/122) SPFQ: A Stochastic Algorithm and Its Error Analysis for Neural Network Quantization (Jinjie Zhang et al., 2023)

{{<citation>}}

Jinjie Zhang, Rayan Saab. (2023)  
**SPFQ: A Stochastic Algorithm and Its Error Analysis for Neural Network Quantization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2309.10975v1)  

---


**ABSTRACT**  
Quantization is a widely used compression method that effectively reduces redundancies in over-parameterized neural networks. However, existing quantization techniques for deep neural networks often lack a comprehensive error analysis due to the presence of non-convex loss functions and nonlinear activations. In this paper, we propose a fast stochastic algorithm for quantizing the weights of fully trained neural networks. Our approach leverages a greedy path-following mechanism in combination with a stochastic quantizer. Its computational complexity scales only linearly with the number of weights in the network, thereby enabling the efficient quantization of large networks. Importantly, we establish, for the first time, full-network error bounds, under an infinite alphabet condition and minimal assumptions on the weights and input data. As an application of this result, we prove that when quantizing a multi-layer network having Gaussian weights, the relative square quantization error exhibits a linear decay as the degree of over-parametrization increases. Furthermore, we demonstrate that it is possible to achieve error bounds equivalent to those obtained in the infinite alphabet case, using on the order of a mere $\log\log N$ bits per weight, where $N$ represents the largest number of neurons in a layer.

{{</citation>}}


## cs.IR (5)



### (68/122) SE-PEF: a Resource for Personalized Expert Finding (Pranav Kasela et al., 2023)

{{<citation>}}

Pranav Kasela, Gabriella Pasi, Raffaele Perego. (2023)  
**SE-PEF: a Resource for Personalized Expert Finding**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval, QA  
[Paper Link](http://arxiv.org/abs/2309.11686v1)  

---


**ABSTRACT**  
The problem of personalization in Information Retrieval has been under study for a long time. A well-known issue related to this task is the lack of publicly available datasets that can support a comparative evaluation of personalized search systems. To contribute in this respect, this paper introduces SE-PEF (StackExchange - Personalized Expert Finding), a resource useful for designing and evaluating personalized models related to the task of Expert Finding (EF). The contributed dataset includes more than 250k queries and 565k answers from 3 306 experts, which are annotated with a rich set of features modeling the social interactions among the users of a popular cQA platform. The results of the preliminary experiments conducted show the appropriateness of SE-PEF to evaluate and to train effective EF models.

{{</citation>}}


### (69/122) Popularity Degradation Bias in Local Music Recommendation (April Trainor et al., 2023)

{{<citation>}}

April Trainor, Douglas Turnbull. (2023)  
**Popularity Degradation Bias in Local Music Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.11671v1)  

---


**ABSTRACT**  
In this paper, we study the effect of popularity degradation bias in the context of local music recommendations. Specifically, we examine how accurate two top-performing recommendation algorithms, Weight Relevance Matrix Factorization (WRMF) and Multinomial Variational Autoencoder (Mult-VAE), are at recommending artists as a function of artist popularity. We find that both algorithms improve recommendation performance for more popular artists and, as such, exhibit popularity degradation bias. While both algorithms produce a similar level of performance for more popular artists, Mult-VAE shows better relative performance for less popular artists. This suggests that this algorithm should be preferred for local (long-tail) music artist recommendation.

{{</citation>}}


### (70/122) Leveraging Negative Signals with Self-Attention for Sequential Music Recommendation (Pavan Seshadri et al., 2023)

{{<citation>}}

Pavan Seshadri, Peter Knees. (2023)  
**Leveraging Negative Signals with Self-Attention for Sequential Music Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2309.11623v1)  

---


**ABSTRACT**  
Music streaming services heavily rely on their recommendation engines to continuously provide content to their consumers. Sequential recommendation consequently has seen considerable attention in current literature, where state of the art approaches focus on self-attentive models leveraging contextual information such as long and short-term user history and item features; however, most of these studies focus on long-form content domains (retail, movie, etc.) rather than short-form, such as music. Additionally, many do not explore incorporating negative session-level feedback during training. In this study, we investigate the use of transformer-based self-attentive architectures to learn implicit session-level information for sequential music recommendation. We additionally propose a contrastive learning task to incorporate negative feedback (e.g skipped tracks) to promote positive hits and penalize negative hits. This task is formulated as a simple loss term that can be incorporated into a variety of deep learning architectures for sequential recommendation. Our experiments show that this results in consistent performance gains over the baseline architectures ignoring negative user feedback.

{{</citation>}}


### (71/122) Retrieving Supporting Evidence for Generative Question Answering (Siqing Huo et al., 2023)

{{<citation>}}

Siqing Huo, Negar Arabzadeh, Charles L. A. Clarke. (2023)  
**Retrieving Supporting Evidence for Generative Question Answering**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Question Answering  
[Paper Link](http://arxiv.org/abs/2309.11392v1)  

---


**ABSTRACT**  
Current large language models (LLMs) can exhibit near-human levels of performance on many natural language-based tasks, including open-domain question answering. Unfortunately, at this time, they also convincingly hallucinate incorrect answers, so that responses to questions must be verified against external sources before they can be accepted at face value. In this paper, we report two simple experiments to automatically validate generated answers against a corpus. We base our experiments on questions and passages from the MS MARCO (V1) test collection, and a retrieval pipeline consisting of sparse retrieval, dense retrieval and neural rerankers. In the first experiment, we validate the generated answer in its entirety. After presenting a question to an LLM and receiving a generated answer, we query the corpus with the combination of the question + generated answer. We then present the LLM with the combination of the question + generated answer + retrieved answer, prompting it to indicate if the generated answer can be supported by the retrieved answer. In the second experiment, we consider the generated answer at a more granular level, prompting the LLM to extract a list of factual statements from the answer and verifying each statement separately. We query the corpus with each factual statement and then present the LLM with the statement and the corresponding retrieved evidence. The LLM is prompted to indicate if the statement can be supported and make necessary edits using the retrieved material. With an accuracy of over 80%, we find that an LLM is capable of verifying its generated answer when a corpus of supporting material is provided. However, manual assessment of a random sample of questions reveals that incorrect generated answers are missed by this verification process. While this verification process can reduce hallucinations, it can not entirely eliminate them.

{{</citation>}}


### (72/122) Long-tail Augmented Graph Contrastive Learning for Recommendation (Qian Zhao et al., 2023)

{{<citation>}}

Qian Zhao, Zhengwei Wu, Zhiqiang Zhang, Jun Zhou. (2023)  
**Long-tail Augmented Graph Contrastive Learning for Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-SI, cs.IR  
Keywords: Contrastive Learning, Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2309.11177v1)  

---


**ABSTRACT**  
Graph Convolutional Networks (GCNs) has demonstrated promising results for recommender systems, as they can effectively leverage high-order relationship. However, these methods usually encounter data sparsity issue in real-world scenarios. To address this issue, GCN-based recommendation methods employ contrastive learning to introduce self-supervised signals. Despite their effectiveness, these methods lack consideration of the significant degree disparity between head and tail nodes. This can lead to non-uniform representation distribution, which is a crucial factor for the performance of contrastive learning methods. To tackle the above issue, we propose a novel Long-tail Augmented Graph Contrastive Learning (LAGCL) method for recommendation. Specifically, we introduce a learnable long-tail augmentation approach to enhance tail nodes by supplementing predicted neighbor information, and generate contrastive views based on the resulting augmented graph. To make the data augmentation schema learnable, we design an auto drop module to generate pseudo-tail nodes from head nodes and a knowledge transfer module to reconstruct the head nodes from pseudo-tail nodes. Additionally, we employ generative adversarial networks to ensure that the distribution of the generated tail/head nodes matches that of the original tail/head nodes. Extensive experiments conducted on three benchmark datasets demonstrate the significant improvement in performance of our model over the state-of-the-arts. Further analyses demonstrate the uniformity of learned representations and the superiority of LAGCL on long-tail performance. Code is publicly available at https://github.com/im0qianqian/LAGCL

{{</citation>}}


## cs.CV (25)



### (73/122) Neural Image Compression Using Masked Sparse Visual Representation (Wei Jiang et al., 2023)

{{<citation>}}

Wei Jiang, Wei Wang, Yue Chen. (2023)  
**Neural Image Compression Using Masked Sparse Visual Representation**  

---
Primary Category: cs.CV  
Categories: 68Wxx, I-4-2; I-4-4; I-4-5, cs-CV, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.11661v1)  

---


**ABSTRACT**  
We study neural image compression based on the Sparse Visual Representation (SVR), where images are embedded into a discrete latent space spanned by learned visual codebooks. By sharing codebooks with the decoder, the encoder transfers integer codeword indices that are efficient and cross-platform robust, and the decoder retrieves the embedded latent feature using the indices for reconstruction. Previous SVR-based compression lacks effective mechanism for rate-distortion tradeoffs, where one can only pursue either high reconstruction quality or low transmission bitrate. We propose a Masked Adaptive Codebook learning (M-AdaCode) method that applies masks to the latent feature subspace to balance bitrate and reconstruction quality. A set of semantic-class-dependent basis codebooks are learned, which are weighted combined to generate a rich latent feature for high-quality reconstruction. The combining weights are adaptively derived from each input image, providing fidelity information with additional transmission costs. By masking out unimportant weights in the encoder and recovering them in the decoder, we can trade off reconstruction quality for transmission bits, and the masking rate controls the balance between bitrate and distortion. Experiments over the standard JPEG-AI dataset demonstrate the effectiveness of our M-AdaCode approach.

{{</citation>}}


### (74/122) Orbital AI-based Autonomous Refuelling Solution (Duarte Rondao et al., 2023)

{{<citation>}}

Duarte Rondao, Lei He, Nabil Aouf. (2023)  
**Orbital AI-based Autonomous Refuelling Solution**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.11648v1)  

---


**ABSTRACT**  
Cameras are rapidly becoming the choice for on-board sensors towards space rendezvous due to their small form factor and inexpensive power, mass, and volume costs. When it comes to docking, however, they typically serve a secondary role, whereas the main work is done by active sensors such as lidar. This paper documents the development of a proposed AI-based (artificial intelligence) navigation algorithm intending to mature the use of on-board visible wavelength cameras as a main sensor for docking and on-orbit servicing (OOS), reducing the dependency on lidar and greatly reducing costs. Specifically, the use of AI enables the expansion of the relative navigation solution towards multiple classes of scenarios, e.g., in terms of targets or illumination conditions, which would otherwise have to be crafted on a case-by-case manner using classical image processing methods. Multiple convolutional neural network (CNN) backbone architectures are benchmarked on synthetically generated data of docking manoeuvres with the International Space Station (ISS), achieving position and attitude estimates close to 1% range-normalised and 1 deg, respectively. The integration of the solution with a physical prototype of the refuelling mechanism is validated in laboratory using a robotic arm to simulate a berthing procedure.

{{</citation>}}


### (75/122) Attentive VQ-VAE (Mariano Rivera et al., 2023)

{{<citation>}}

Mariano Rivera, Angello Hoyos. (2023)  
**Attentive VQ-VAE**  

---
Primary Category: cs.CV  
Categories: I-2-10, cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.11641v1)  

---


**ABSTRACT**  
We present a novel approach to enhance the capabilities of VQVAE models through the integration of an Attentive Residual Encoder (AREN) and a Residual Pixel Attention layer. The objective of our research is to improve the performance of VQVAE while maintaining practical parameter levels. The AREN encoder is designed to operate effectively at multiple levels, accommodating diverse architectural complexities. The key innovation is the integration of an inter-pixel auto-attention mechanism into the AREN encoder. This approach allows us to efficiently capture and utilize contextual information across latent vectors. Additionally, our models uses additional encoding levels to further enhance the model's representational power. Our attention layer employs a minimal parameter approach, ensuring that latent vectors are modified only when pertinent information from other pixels is available. Experimental results demonstrate that our proposed modifications lead to significant improvements in data representation and generation, making VQVAEs even more suitable for a wide range of applications.

{{</citation>}}


### (76/122) Sentence Attention Blocks for Answer Grounding (Seyedalireza Khoshsirat et al., 2023)

{{<citation>}}

Seyedalireza Khoshsirat, Chandra Kambhamettu. (2023)  
**Sentence Attention Blocks for Answer Grounding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.11593v1)  

---


**ABSTRACT**  
Answer grounding is the task of locating relevant visual evidence for the Visual Question Answering task. While a wide variety of attention methods have been introduced for this task, they suffer from the following three problems: designs that do not allow the usage of pre-trained networks and do not benefit from large data pre-training, custom designs that are not based on well-grounded previous designs, therefore limiting the learning power of the network, or complicated designs that make it challenging to re-implement or improve them. In this paper, we propose a novel architectural block, which we term Sentence Attention Block, to solve these problems. The proposed block re-calibrates channel-wise image feature-maps by explicitly modeling inter-dependencies between the image feature-maps and sentence embedding. We visually demonstrate how this block filters out irrelevant feature-maps channels based on sentence embedding. We start our design with a well-known attention method, and by making minor modifications, we improve the results to achieve state-of-the-art accuracy. The flexibility of our method makes it easy to use different pre-trained backbone networks, and its simplicity makes it easy to understand and be re-implemented. We demonstrate the effectiveness of our method on the TextVQA-X, VQS, VQA-X, and VizWiz-VQA-Grounding datasets. We perform multiple ablation studies to show the effectiveness of our design choices.

{{</citation>}}


### (77/122) DreamLLM: Synergistic Multimodal Comprehension and Creation (Runpei Dong et al., 2023)

{{<citation>}}

Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, Li Yi. (2023)  
**DreamLLM: Synergistic Multimodal Comprehension and Creation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.11499v1)  

---


**ABSTRACT**  
This paper presents DreamLLM, a learning framework that first achieves versatile Multimodal Large Language Models (MLLMs) empowered with frequently overlooked synergy between multimodal comprehension and creation. DreamLLM operates on two fundamental principles. The first focuses on the generative modeling of both language and image posteriors by direct sampling in the raw multimodal space. This approach circumvents the limitations and information loss inherent to external feature extractors like CLIP, and a more thorough multimodal understanding is obtained. Second, DreamLLM fosters the generation of raw, interleaved documents, modeling both text and image contents, along with unstructured layouts. This allows DreamLLM to learn all conditional, marginal, and joint multimodal distributions effectively. As a result, DreamLLM is the first MLLM capable of generating free-form interleaved content. Comprehensive experiments highlight DreamLLM's superior performance as a zero-shot multimodal generalist, reaping from the enhanced learning synergy.

{{</citation>}}


### (78/122) Budget-Aware Pruning: Handling Multiple Domains with Less Parameters (Samuel Felipe dos Santos et al., 2023)

{{<citation>}}

Samuel Felipe dos Santos, Rodrigo Berriel, Thiago Oliveira-Santos, Nicu Sebe, Jurandy Almeida. (2023)  
**Budget-Aware Pruning: Handling Multiple Domains with Less Parameters**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2309.11464v1)  

---


**ABSTRACT**  
Deep learning has achieved state-of-the-art performance on several computer vision tasks and domains. Nevertheless, it still has a high computational cost and demands a significant amount of parameters. Such requirements hinder the use in resource-limited environments and demand both software and hardware optimization. Another limitation is that deep models are usually specialized into a single domain or task, requiring them to learn and store new parameters for each new one. Multi-Domain Learning (MDL) attempts to solve this problem by learning a single model that is capable of performing well in multiple domains. Nevertheless, the models are usually larger than the baseline for a single domain. This work tackles both of these problems: our objective is to prune models capable of handling multiple domains according to a user-defined budget, making them more computationally affordable while keeping a similar classification performance. We achieve this by encouraging all domains to use a similar subset of filters from the baseline model, up to the amount defined by the user's budget. Then, filters that are not used by any domain are pruned from the network. The proposed approach innovates by better adapting to resource-limited devices while, to our knowledge, being the only work that handles multiple domains at test time with fewer parameters and lower computational complexity than the baseline model for a single domain.

{{</citation>}}


### (79/122) SkeleTR: Towrads Skeleton-based Action Recognition in the Wild (Haodong Duan et al., 2023)

{{<citation>}}

Haodong Duan, Mingze Xu, Bing Shuai, Davide Modolo, Zhuowen Tu, Joseph Tighe, Alessandro Bergamo. (2023)  
**SkeleTR: Towrads Skeleton-based Action Recognition in the Wild**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.11445v1)  

---


**ABSTRACT**  
We present SkeleTR, a new framework for skeleton-based action recognition. In contrast to prior work, which focuses mainly on controlled environments, we target more general scenarios that typically involve a variable number of people and various forms of interaction between people. SkeleTR works with a two-stage paradigm. It first models the intra-person skeleton dynamics for each skeleton sequence with graph convolutions, and then uses stacked Transformer encoders to capture person interactions that are important for action recognition in general scenarios. To mitigate the negative impact of inaccurate skeleton associations, SkeleTR takes relative short skeleton sequences as input and increases the number of sequences. As a unified solution, SkeleTR can be directly applied to multiple skeleton-based action tasks, including video-level action classification, instance-level action detection, and group-level activity recognition. It also enables transfer learning and joint training across different action tasks and datasets, which result in performance improvement. When evaluated on various skeleton-based action recognition benchmarks, SkeleTR achieves the state-of-the-art performance.

{{</citation>}}


### (80/122) A Systematic Review of Few-Shot Learning in Medical Imaging (Eva Pachetti et al., 2023)

{{<citation>}}

Eva Pachetti, Sara Colantonio. (2023)  
**A Systematic Review of Few-Shot Learning in Medical Imaging**  

---
Primary Category: cs.CV  
Categories: I-2-6; I-4; I-5; J-3, cs-AI, cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.11433v1)  

---


**ABSTRACT**  
The lack of annotated medical images limits the performance of deep learning models, which usually need large-scale labelled datasets. Few-shot learning techniques can reduce data scarcity issues and enhance medical image analysis, especially with meta-learning. This systematic review gives a comprehensive overview of few-shot learning in medical imaging. We searched the literature systematically and selected 80 relevant articles published from 2018 to 2023. We clustered the articles based on medical outcomes, such as tumour segmentation, disease classification, and image registration; anatomical structure investigated (i.e. heart, lung, etc.); and the meta-learning method used. For each cluster, we examined the papers' distributions and the results provided by the state-of-the-art. In addition, we identified a generic pipeline shared among all the studies. The review shows that few-shot learning can overcome data scarcity in most outcomes and that meta-learning is a popular choice to perform few-shot learning because it can adapt to new tasks with few labelled samples. In addition, following meta-learning, supervised learning and semi-supervised learning stand out as the predominant techniques employed to tackle few-shot learning challenges in medical imaging and also best performing. Lastly, we observed that the primary application areas predominantly encompass cardiac, pulmonary, and abdominal domains. This systematic review aims to inspire further research to improve medical image analysis and patient care.

{{</citation>}}


### (81/122) Uncovering the effects of model initialization on deep model generalization: A study with adult and pediatric Chest X-ray images (Sivaramakrishnan Rajaraman et al., 2023)

{{<citation>}}

Sivaramakrishnan Rajaraman, Ghada Zamzmi, Feng Yang, Zhaohui Liang, Zhiyun Xue, Sameer Antani. (2023)  
**Uncovering the effects of model initialization on deep model generalization: A study with adult and pediatric Chest X-ray images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, ImageNet  
[Paper Link](http://arxiv.org/abs/2309.11318v1)  

---


**ABSTRACT**  
Model initialization techniques are vital for improving the performance and reliability of deep learning models in medical computer vision applications. While much literature exists on non-medical images, the impacts on medical images, particularly chest X-rays (CXRs) are less understood. Addressing this gap, our study explores three deep model initialization techniques: Cold-start, Warm-start, and Shrink and Perturb start, focusing on adult and pediatric populations. We specifically focus on scenarios with periodically arriving data for training, thereby embracing the real-world scenarios of ongoing data influx and the need for model updates. We evaluate these models for generalizability against external adult and pediatric CXR datasets. We also propose novel ensemble methods: F-score-weighted Sequential Least-Squares Quadratic Programming (F-SLSQP) and Attention-Guided Ensembles with Learnable Fuzzy Softmax to aggregate weight parameters from multiple models to capitalize on their collective knowledge and complementary representations. We perform statistical significance tests with 95% confidence intervals and p-values to analyze model performance. Our evaluations indicate models initialized with ImageNet-pre-trained weights demonstrate superior generalizability over randomly initialized counterparts, contradicting some findings for non-medical images. Notably, ImageNet-pretrained models exhibit consistent performance during internal and external testing across different training scenarios. Weight-level ensembles of these models show significantly higher recall (p<0.05) during testing compared to individual models. Thus, our study accentuates the benefits of ImageNet-pretrained weight initialization, especially when used with weight-level ensembles, for creating robust and generalizable deep learning solutions.

{{</citation>}}


### (82/122) FaceDiffuser: Speech-Driven 3D Facial Animation Synthesis Using Diffusion (Stefan Stan et al., 2023)

{{<citation>}}

Stefan Stan, Kazi Injamamul Haque, Zerrin Yumak. (2023)  
**FaceDiffuser: Speech-Driven 3D Facial Animation Synthesis Using Diffusion**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-GR, cs.CV  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.11306v1)  

---


**ABSTRACT**  
Speech-driven 3D facial animation synthesis has been a challenging task both in industry and research. Recent methods mostly focus on deterministic deep learning methods meaning that given a speech input, the output is always the same. However, in reality, the non-verbal facial cues that reside throughout the face are non-deterministic in nature. In addition, majority of the approaches focus on 3D vertex based datasets and methods that are compatible with existing facial animation pipelines with rigged characters is scarce. To eliminate these issues, we present FaceDiffuser, a non-deterministic deep learning model to generate speech-driven facial animations that is trained with both 3D vertex and blendshape based datasets. Our method is based on the diffusion technique and uses the pre-trained large speech representation model HuBERT to encode the audio input. To the best of our knowledge, we are the first to employ the diffusion method for the task of speech-driven 3D facial animation synthesis. We have run extensive objective and subjective analyses and show that our approach achieves better or comparable results in comparison to the state-of-the-art methods. We also introduce a new in-house dataset that is based on a blendshape based rigged character. We recommend watching the accompanying supplementary video. The code and the dataset will be publicly available.

{{</citation>}}


### (83/122) StructChart: Perception, Structuring, Reasoning for Visual Chart Understanding (Renqiu Xia et al., 2023)

{{<citation>}}

Renqiu Xia, Bo Zhang, Haoyang Peng, Ning Liao, Peng Ye, Botian Shi, Junchi Yan, Yu Qiao. (2023)  
**StructChart: Perception, Structuring, Reasoning for Visual Chart Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.11268v2)  

---


**ABSTRACT**  
Charts are common in literature across different scientific fields, conveying rich information easily accessible to readers. Current chart-related tasks focus on either chart perception which refers to extracting information from the visual charts, or performing reasoning given the extracted data, e.g. in a tabular form. In this paper, we aim to establish a unified and label-efficient learning paradigm for joint perception and reasoning tasks, which can be generally applicable to different downstream tasks, beyond the question-answering task as specifically studied in peer works. Specifically, StructChart first reformulates the chart information from the popular tubular form (specifically linearized CSV) to the proposed Structured Triplet Representations (STR), which is more friendly for reducing the task gap between chart perception and reasoning due to the employed structured information extraction for charts. We then propose a Structuring Chart-oriented Representation Metric (SCRM) to quantitatively evaluate the performance for the chart perception task. To enrich the dataset for training, we further explore the possibility of leveraging the Large Language Model (LLM), enhancing the chart diversity in terms of both chart visual style and its statistical information. Extensive experiments are conducted on various chart-related tasks, demonstrating the effectiveness and promising potential for a unified chart perception-reasoning paradigm to push the frontier of chart understanding.

{{</citation>}}


### (84/122) From Classification to Segmentation with Explainable AI: A Study on Crack Detection and Growth Monitoring (Florent Forest et al., 2023)

{{<citation>}}

Florent Forest, Hugo Porta, Devis Tuia, Olga Fink. (2023)  
**From Classification to Segmentation with Explainable AI: A Study on Crack Detection and Growth Monitoring**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.11267v1)  

---


**ABSTRACT**  
Monitoring surface cracks in infrastructure is crucial for structural health monitoring. Automatic visual inspection offers an effective solution, especially in hard-to-reach areas. Machine learning approaches have proven their effectiveness but typically require large annotated datasets for supervised training. Once a crack is detected, monitoring its severity often demands precise segmentation of the damage. However, pixel-level annotation of images for segmentation is labor-intensive. To mitigate this cost, one can leverage explainable artificial intelligence (XAI) to derive segmentations from the explanations of a classifier, requiring only weak image-level supervision. This paper proposes applying this methodology to segment and monitor surface cracks. We evaluate the performance of various XAI methods and examine how this approach facilitates severity quantification and growth monitoring. Results reveal that while the resulting segmentation masks may exhibit lower quality than those produced by supervised methods, they remain meaningful and enable severity monitoring, thus reducing substantial labeling costs.

{{</citation>}}


### (85/122) Box2Poly: Memory-Efficient Polygon Prediction of Arbitrarily Shaped and Rotated Text (Xuyang Chen et al., 2023)

{{<citation>}}

Xuyang Chen, Dong Wang, Konrad Schindler, Mingwei Sun, Yongliang Wang, Nicolo Savioli, Liqiu Meng. (2023)  
**Box2Poly: Memory-Efficient Polygon Prediction of Arbitrarily Shaped and Rotated Text**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.11248v1)  

---


**ABSTRACT**  
Recently, Transformer-based text detection techniques have sought to predict polygons by encoding the coordinates of individual boundary vertices using distinct query features. However, this approach incurs a significant memory overhead and struggles to effectively capture the intricate relationships between vertices belonging to the same instance. Consequently, irregular text layouts often lead to the prediction of outlined vertices, diminishing the quality of results. To address these challenges, we present an innovative approach rooted in Sparse R-CNN: a cascade decoding pipeline for polygon prediction. Our method ensures precision by iteratively refining polygon predictions, considering both the scale and location of preceding results. Leveraging this stabilized regression pipeline, even employing just a single feature vector to guide polygon instance regression yields promising detection results. Simultaneously, the leverage of instance-level feature proposal substantially enhances memory efficiency (>50% less vs. the state-of-the-art method DPText-DETR) and reduces inference speed (>40% less vs. DPText-DETR) with minor performance drop on benchmarks.

{{</citation>}}


### (86/122) Towards Robust Few-shot Point Cloud Semantic Segmentation (Yating Xu et al., 2023)

{{<citation>}}

Yating Xu, Na Zhao, Gim Hee Lee. (2023)  
**Towards Robust Few-shot Point Cloud Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.11228v1)  

---


**ABSTRACT**  
Few-shot point cloud semantic segmentation aims to train a model to quickly adapt to new unseen classes with only a handful of support set samples. However, the noise-free assumption in the support set can be easily violated in many practical real-world settings. In this paper, we focus on improving the robustness of few-shot point cloud segmentation under the detrimental influence of noisy support sets during testing time. To this end, we first propose a Component-level Clean Noise Separation (CCNS) representation learning to learn discriminative feature representations that separates the clean samples of the target classes from the noisy samples. Leveraging the well separated clean and noisy support samples from our CCNS, we further propose a Multi-scale Degree-based Noise Suppression (MDNS) scheme to remove the noisy shots from the support set. We conduct extensive experiments on various noise settings on two benchmark datasets. Our results show that the combination of CCNS and MDNS significantly improves the performance. Our code is available at https://github.com/Pixie8888/R3DFSSeg.

{{</citation>}}


### (87/122) Generalized Few-Shot Point Cloud Segmentation Via Geometric Words (Yating Xu et al., 2023)

{{<citation>}}

Yating Xu, Conghui Hu, Na Zhao, Gim Hee Lee. (2023)  
**Generalized Few-Shot Point Cloud Segmentation Via Geometric Words**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.11222v1)  

---


**ABSTRACT**  
Existing fully-supervised point cloud segmentation methods suffer in the dynamic testing environment with emerging new classes. Few-shot point cloud segmentation algorithms address this problem by learning to adapt to new classes at the sacrifice of segmentation accuracy for the base classes, which severely impedes its practicality. This largely motivates us to present the first attempt at a more practical paradigm of generalized few-shot point cloud segmentation, which requires the model to generalize to new categories with only a few support point clouds and simultaneously retain the capability to segment base classes. We propose the geometric words to represent geometric components shared between the base and novel classes, and incorporate them into a novel geometric-aware semantic representation to facilitate better generalization to the new classes without forgetting the old ones. Moreover, we introduce geometric prototypes to guide the segmentation with geometric prior knowledge. Extensive experiments on S3DIS and ScanNet consistently illustrate the superior performance of our method over baseline methods. Our code is available at: https://github.com/Pixie8888/GFS-3DSeg_GWs.

{{</citation>}}


### (88/122) Automatic Bat Call Classification using Transformer Networks (Frank Fundel et al., 2023)

{{<citation>}}

Frank Fundel, Daniel A. Braun, Sebastian Gottwald. (2023)  
**Automatic Bat Call Classification using Transformer Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-SD, cs.CV, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.11218v1)  

---


**ABSTRACT**  
Automatically identifying bat species from their echolocation calls is a difficult but important task for monitoring bats and the ecosystem they live in. Major challenges in automatic bat call identification are high call variability, similarities between species, interfering calls and lack of annotated data. Many currently available models suffer from relatively poor performance on real-life data due to being trained on single call datasets and, moreover, are often too slow for real-time classification. Here, we propose a Transformer architecture for multi-label classification with potential applications in real-time classification scenarios. We train our model on synthetically generated multi-species recordings by merging multiple bats calls into a single recording with multiple simultaneous calls. Our approach achieves a single species accuracy of 88.92% (F1-score of 84.23%) and a multi species macro F1-score of 74.40% on our test set. In comparison to three other tools on the independent and publicly available dataset ChiroVox, our model achieves at least 25.82% better accuracy for single species classification and at least 6.9% better macro F1-score for multi species classification.

{{</citation>}}


### (89/122) EPTQ: Enhanced Post-Training Quantization via Label-Free Hessian (Ofir Gordon et al., 2023)

{{<citation>}}

Ofir Gordon, Hai Victor Habi, Arnon Netzer. (2023)  
**EPTQ: Enhanced Post-Training Quantization via Label-Free Hessian**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Quantization, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.11531v1)  

---


**ABSTRACT**  
Quantization of deep neural networks (DNN) has become a key element in the efforts of embedding such networks on end-user devices. However, current quantization methods usually suffer from costly accuracy degradation. In this paper, we propose a new method for Enhanced Post Training Quantization named EPTQ. The method is based on knowledge distillation with an adaptive weighting of layers. In addition, we introduce a new label-free technique for approximating the Hessian trace of the task loss, named Label-Free Hessian. This technique removes the requirement of a labeled dataset for computing the Hessian. The adaptive knowledge distillation uses the Label-Free Hessian technique to give greater attention to the sensitive parts of the model while performing the optimization. Empirically, by employing EPTQ we achieve state-of-the-art results on a wide variety of models, tasks, and datasets, including ImageNet classification, COCO object detection, and Pascal-VOC for semantic segmentation. We demonstrate the performance and compatibility of EPTQ on an extended set of architectures, including CNNs, Transformers, hybrid, and MLP-only models.

{{</citation>}}


### (90/122) Multi-grained Temporal Prototype Learning for Few-shot Video Object Segmentation (Nian Liu et al., 2023)

{{<citation>}}

Nian Liu, Kepan Nan, Wangbo Zhao, Yuanwei Liu, Xiwen Yao, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer, Junwei Han, Fahad Shahbaz Khan. (2023)  
**Multi-grained Temporal Prototype Learning for Few-shot Video Object Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.11160v1)  

---


**ABSTRACT**  
Few-Shot Video Object Segmentation (FSVOS) aims to segment objects in a query video with the same category defined by a few annotated support images. However, this task was seldom explored. In this work, based on IPMT, a state-of-the-art few-shot image segmentation method that combines external support guidance information with adaptive query guidance cues, we propose to leverage multi-grained temporal guidance information for handling the temporal correlation nature of video data. We decompose the query video information into a clip prototype and a memory prototype for capturing local and long-term internal temporal guidance, respectively. Frame prototypes are further used for each frame independently to handle fine-grained adaptive guidance and enable bidirectional clip-frame prototype communication. To reduce the influence of noisy memory, we propose to leverage the structural similarity relation among different predicted regions and the support for selecting reliable memory frames. Furthermore, a new segmentation loss is also proposed to enhance the category discriminability of the learned prototypes. Experimental results demonstrate that our proposed video IPMT model significantly outperforms previous models on two benchmark datasets. Code is available at https://github.com/nankepan/VIPMT.

{{</citation>}}


### (91/122) PRAT: PRofiling Adversarial aTtacks (Rahul Ambati et al., 2023)

{{<citation>}}

Rahul Ambati, Naveed Akhtar, Ajmal Mian, Yogesh Singh Rawat. (2023)  
**PRAT: PRofiling Adversarial aTtacks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2309.11111v1)  

---


**ABSTRACT**  
Intrinsic susceptibility of deep learning to adversarial examples has led to a plethora of attack techniques with a broad common objective of fooling deep models. However, we find slight compositional differences between the algorithms achieving this objective. These differences leave traces that provide important clues for attacker profiling in real-life scenarios. Inspired by this, we introduce a novel problem of PRofiling Adversarial aTtacks (PRAT). Given an adversarial example, the objective of PRAT is to identify the attack used to generate it. Under this perspective, we can systematically group existing attacks into different families, leading to the sub-problem of attack family identification, which we also study. To enable PRAT analysis, we introduce a large Adversarial Identification Dataset (AID), comprising over 180k adversarial samples generated with 13 popular attacks for image specific/agnostic white/black box setups. We use AID to devise a novel framework for the PRAT objective. Our framework utilizes a Transformer based Global-LOcal Feature (GLOF) module to extract an approximate signature of the adversarial attack, which in turn is used for the identification of the attack. Using AID and our framework, we provide multiple interesting benchmark results for the PRAT problem.

{{</citation>}}


### (92/122) Forgery-aware Adaptive Vision Transformer for Face Forgery Detection (Anwei Luo et al., 2023)

{{<citation>}}

Anwei Luo, Rizhao Cai, Chenqi Kong, Xiangui Kang, Jiwu Huang, Alex C. Kot. (2023)  
**Forgery-aware Adaptive Vision Transformer for Face Forgery Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2309.11092v1)  

---


**ABSTRACT**  
With the advancement in face manipulation technologies, the importance of face forgery detection in protecting authentication integrity becomes increasingly evident. Previous Vision Transformer (ViT)-based detectors have demonstrated subpar performance in cross-database evaluations, primarily because fully fine-tuning with limited Deepfake data often leads to forgetting pre-trained knowledge and over-fitting to data-specific ones. To circumvent these issues, we propose a novel Forgery-aware Adaptive Vision Transformer (FA-ViT). In FA-ViT, the vanilla ViT's parameters are frozen to preserve its pre-trained knowledge, while two specially designed components, the Local-aware Forgery Injector (LFI) and the Global-aware Forgery Adaptor (GFA), are employed to adapt forgery-related knowledge. our proposed FA-ViT effectively combines these two different types of knowledge to form the general forgery features for detecting Deepfakes. Specifically, LFI captures local discriminative information and incorporates these information into ViT via Neighborhood-Preserving Cross Attention (NPCA). Simultaneously, GFA learns adaptive knowledge in the self-attention layer, bridging the gap between the two different domain. Furthermore, we design a novel Single Domain Pairwise Learning (SDPL) to facilitate fine-grained information learning in FA-ViT. The extensive experiments demonstrate that our FA-ViT achieves state-of-the-art performance in cross-dataset evaluation and cross-manipulation scenarios, and improves the robustness against unseen perturbations.

{{</citation>}}


### (93/122) Dual-Modal Attention-Enhanced Text-Video Retrieval with Triplet Partial Margin Contrastive Learning (Chen Jiang et al., 2023)

{{<citation>}}

Chen Jiang, Hong Liu, Xuzheng Yu, Qing Wang, Yuan Cheng, Jia Xu, Zhongyi Liu, Qingpei Guo, Wei Chu, Ming Yang, Yuan Qi. (2023)  
**Dual-Modal Attention-Enhanced Text-Video Retrieval with Triplet Partial Margin Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-MM, cs.CV  
Keywords: Attention, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.11082v1)  

---


**ABSTRACT**  
In recent years, the explosion of web videos makes text-video retrieval increasingly essential and popular for video filtering, recommendation, and search. Text-video retrieval aims to rank relevant text/video higher than irrelevant ones. The core of this task is to precisely measure the cross-modal similarity between texts and videos. Recently, contrastive learning methods have shown promising results for text-video retrieval, most of which focus on the construction of positive and negative pairs to learn text and video representations. Nevertheless, they do not pay enough attention to hard negative pairs and lack the ability to model different levels of semantic similarity. To address these two issues, this paper improves contrastive learning using two novel techniques. First, to exploit hard examples for robust discriminative power, we propose a novel Dual-Modal Attention-Enhanced Module (DMAE) to mine hard negative pairs from textual and visual clues. By further introducing a Negative-aware InfoNCE (NegNCE) loss, we are able to adaptively identify all these hard negatives and explicitly highlight their impacts in the training loss. Second, our work argues that triplet samples can better model fine-grained semantic similarity compared to pairwise samples. We thereby present a new Triplet Partial Margin Contrastive Learning (TPM-CL) module to construct partial order triplet samples by automatically generating fine-grained hard negatives for matched text-video pairs. The proposed TPM-CL designs an adaptive token masking strategy with cross-modal interaction to model subtle semantic differences. Extensive experiments demonstrate that the proposed approach outperforms existing methods on four widely-used text-video retrieval datasets, including MSR-VTT, MSVD, DiDeMo and ActivityNet.

{{</citation>}}


### (94/122) Visual Question Answering in the Medical Domain (Louisa Canepa et al., 2023)

{{<citation>}}

Louisa Canepa, Sonit Singh, Arcot Sowmya. (2023)  
**Visual Question Answering in the Medical Domain**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.11080v1)  

---


**ABSTRACT**  
Medical visual question answering (Med-VQA) is a machine learning task that aims to create a system that can answer natural language questions based on given medical images. Although there has been rapid progress on the general VQA task, less progress has been made on Med-VQA due to the lack of large-scale annotated datasets. In this paper, we present domain-specific pre-training strategies, including a novel contrastive learning pretraining method, to mitigate the problem of small datasets for the Med-VQA task. We find that the model benefits from components that use fewer parameters. We also evaluate and discuss the model's visual reasoning using evidence verification techniques. Our proposed model obtained an accuracy of 60% on the VQA-Med 2019 test set, giving comparable results to other state-of-the-art Med-VQA models.

{{</citation>}}


### (95/122) Dynamic Tiling: A Model-Agnostic, Adaptive, Scalable, and Inference-Data-Centric Approach for Efficient and Accurate Small Object Detection (Son The Nguyen et al., 2023)

{{<citation>}}

Son The Nguyen, Theja Tulabandhula, Duy Nguyen. (2023)  
**Dynamic Tiling: A Model-Agnostic, Adaptive, Scalable, and Inference-Data-Centric Approach for Efficient and Accurate Small Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.11069v1)  

---


**ABSTRACT**  
We introduce Dynamic Tiling, a model-agnostic, adaptive, and scalable approach for small object detection, anchored in our inference-data-centric philosophy. Dynamic Tiling starts with non-overlapping tiles for initial detections and utilizes dynamic overlapping rates along with a tile minimizer. This dual approach effectively resolves fragmented objects, improves detection accuracy, and minimizes computational overhead by reducing the number of forward passes through the object detection model. Adaptable to a variety of operational environments, our method negates the need for laborious recalibration. Additionally, our large-small filtering mechanism boosts the detection quality across a range of object sizes. Overall, Dynamic Tiling outperforms existing model-agnostic uniform cropping methods, setting new benchmarks for efficiency and accuracy.

{{</citation>}}


### (96/122) COSE: A Consistency-Sensitivity Metric for Saliency on Image Classification (Rangel Daroya et al., 2023)

{{<citation>}}

Rangel Daroya, Aaron Sun, Subhransu Maji. (2023)  
**COSE: A Consistency-Sensitivity Metric for Saliency on Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2309.10989v1)  

---


**ABSTRACT**  
We present a set of metrics that utilize vision priors to effectively assess the performance of saliency methods on image classification tasks. To understand behavior in deep learning models, many methods provide visual saliency maps emphasizing image regions that most contribute to a model prediction. However, there is limited work on analyzing the reliability of saliency methods in explaining model decisions. We propose the metric COnsistency-SEnsitivity (COSE) that quantifies the equivariant and invariant properties of visual model explanations using simple data augmentations. Through our metrics, we show that although saliency methods are thought to be architecture-independent, most methods could better explain transformer-based models over convolutional-based models. In addition, GradCAM was found to outperform other methods in terms of COSE but was shown to have limitations such as lack of variability for fine-grained datasets. The duality between consistency and sensitivity allow the analysis of saliency methods from different angles. Ultimately, we find that it is important to balance these two metrics for a saliency map to faithfully show model behavior.

{{</citation>}}


### (97/122) RMT: Retentive Networks Meet Vision Transformers (Qihang Fan et al., 2023)

{{<citation>}}

Qihang Fan, Huaibo Huang, Mingrui Chen, Hongmin Liu, Ran He. (2023)  
**RMT: Retentive Networks Meet Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.11523v1)  

---


**ABSTRACT**  
Transformer first appears in the field of natural language processing and is later migrated to the computer vision domain, where it demonstrates excellent performance in vision tasks. However, recently, Retentive Network (RetNet) has emerged as an architecture with the potential to replace Transformer, attracting widespread attention in the NLP community. Therefore, we raise the question of whether transferring RetNet's idea to vision can also bring outstanding performance to vision tasks. To address this, we combine RetNet and Transformer to propose RMT. Inspired by RetNet, RMT introduces explicit decay into the vision backbone, bringing prior knowledge related to spatial distances to the vision model. This distance-related spatial prior allows for explicit control of the range of tokens that each token can attend to. Additionally, to reduce the computational cost of global modeling, we decompose this modeling process along the two coordinate axes of the image. Abundant experiments have demonstrated that our RMT exhibits exceptional performance across various computer vision tasks. For example, RMT achieves 84.1% Top1-acc on ImageNet-1k using merely 4.5G FLOPs. To the best of our knowledge, among all models, RMT achieves the highest Top1-acc when models are of similar size and trained with the same strategy. Moreover, RMT significantly outperforms existing vision backbones in downstream tasks such as object detection, instance segmentation, and semantic segmentation. Our work is still in progress.

{{</citation>}}


## cs.DS (1)



### (98/122) GLM Regression with Oblivious Corruptions (Ilias Diakonikolas et al., 2023)

{{<citation>}}

Ilias Diakonikolas, Sushrut Karmalkar, Jongho Park, Christos Tzamos. (2023)  
**GLM Regression with Oblivious Corruptions**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs-LG, cs.DS, math-ST, stat-ML, stat-TH  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2309.11657v1)  

---


**ABSTRACT**  
We demonstrate the first algorithms for the problem of regression for generalized linear models (GLMs) in the presence of additive oblivious noise. We assume we have sample access to examples $(x, y)$ where $y$ is a noisy measurement of $g(w^* \cdot x)$. In particular, \new{the noisy labels are of the form} $y = g(w^* \cdot x) + \xi + \epsilon$, where $\xi$ is the oblivious noise drawn independently of $x$ \new{and satisfies} $\Pr[\xi = 0] \geq o(1)$, and $\epsilon \sim \mathcal N(0, \sigma^2)$. Our goal is to accurately recover a \new{parameter vector $w$ such that the} function $g(w \cdot x)$ \new{has} arbitrarily small error when compared to the true values $g(w^* \cdot x)$, rather than the noisy measurements $y$.   We present an algorithm that tackles \new{this} problem in its most general distribution-independent setting, where the solution may not \new{even} be identifiable. \new{Our} algorithm returns \new{an accurate estimate of} the solution if it is identifiable, and otherwise returns a small list of candidates, one of which is close to the true solution. Furthermore, we \new{provide} a necessary and sufficient condition for identifiability, which holds in broad settings. \new{Specifically,} the problem is identifiable when the quantile at which $\xi + \epsilon = 0$ is known, or when the family of hypotheses does not contain candidates that are nearly equal to a translated $g(w^* \cdot x) + A$ for some real number $A$, while also having large error when compared to $g(w^* \cdot x)$.   This is the first \new{algorithmic} result for GLM regression \new{with oblivious noise} which can handle more than half the samples being arbitrarily corrupted. Prior work focused largely on the setting of linear regression, and gave algorithms under restrictive assumptions.

{{</citation>}}


## cs.HC (3)



### (99/122) 'It's a Fair Game'', or Is It? Examining How Users Navigate Disclosure Risks and Benefits When Using LLM-Based Conversational Agents (Zhiping Zhang et al., 2023)

{{<citation>}}

Zhiping Zhang, Michelle Jia, Hao-Ping, Lee, Bingsheng Yao, Sauvik Das, Ada Lerner, Dakuo Wang, Tianshi Li. (2023)  
**'It's a Fair Game'', or Is It? Examining How Users Navigate Disclosure Risks and Benefits When Using LLM-Based Conversational Agents**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CR, cs-HC, cs.HC  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.11653v1)  

---


**ABSTRACT**  
The widespread use of Large Language Model (LLM)-based conversational agents (CAs), especially in high-stakes domains, raises many privacy concerns. Building ethical LLM-based CAs that respect user privacy requires an in-depth understanding of the privacy risks that concern users the most. However, existing research, primarily model-centered, does not provide insight into users' perspectives. To bridge this gap, we analyzed sensitive disclosures in real-world ChatGPT conversations and conducted semi-structured interviews with 19 LLM-based CA users. We found that users are constantly faced with trade-offs between privacy, utility, and convenience when using LLM-based CAs. However, users' erroneous mental models and the dark patterns in system design limited their awareness and comprehension of the privacy risks. Additionally, the human-like interactions encouraged more sensitive disclosures, which complicated users' ability to navigate the trade-offs. We discuss practical design guidelines and the needs for paradigmatic shifts to protect the privacy of LLM-based CA users.

{{</citation>}}


### (100/122) Interactive Flexible Style Transfer for Vector Graphics (Jeremy Warner et al., 2023)

{{<citation>}}

Jeremy Warner, Kyu Won Kim, Bjoern Hartmann. (2023)  
**Interactive Flexible Style Transfer for Vector Graphics**  

---
Primary Category: cs.HC  
Categories: cs-GR, cs-HC, cs.HC  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2309.11628v1)  

---


**ABSTRACT**  
Vector graphics are an industry-standard way to represent and share visual designs. Designers frequently source and incorporate styles from existing designs into their own work. Unfortunately, popular design tools aren't well suited for this task. We present VST, Vector Style Transfer, a novel design tool for flexibly transferring visual styles between vector graphics. The core of VST lies in leveraging automation while respecting designers' tastes and the subjectivity inherent to style transfer. In VST, designers tune a cross-design element correspondence and customize which style attributes to change. We report results from a user study in which designers used VST to control style transfer between several designs, including designs participants created with external tools beforehand. VST shows that enabling design correspondence tuning and customization is one way to support interactive, flexible style transfer. We also find that someone using VST can significantly reduce the time and work for style transfer compared to experienced designers using industry-standard tools.

{{</citation>}}


### (101/122) The Role of Inclusion, Control, and Ownership in Workplace AI-Mediated Communication (Kowe Kadoma et al., 2023)

{{<citation>}}

Kowe Kadoma, Marianne Aubin Le Quere, Jenny Fu, Christin Munsch, Danae Metaxa, Mor Naaman. (2023)  
**The Role of Inclusion, Control, and Ownership in Workplace AI-Mediated Communication**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.11599v1)  

---


**ABSTRACT**  
Large language models (LLMs) can exhibit social biases. Given LLMs' increasing integration into workplace software, these biases may impact workers' well-being and may disproportionately impact minoritized communities. This short paper investigates how co-writing with an LLM impacts three measures related to user's well-being: feelings of inclusion, control, and ownership over their work. In an online experiment, participants wrote hypothetical job promotion requests to their boss and using either hesitant or self-assured auto-complete suggestions from an LLM. Afterward, participants reported their feelings of inclusion, control, and ownership. We found that the style of the AI model did not impact perceived inclusion. Furthermore, individuals with higher perceived inclusion also perceived greater agency and ownership, an effect more strongly impacting participants of minoritized genders. Lastly, feelings of inclusion can mitigate a loss of control and agency when accepting more AI suggestions. Future work should explore feelings of inclusion in AI-written communication.

{{</citation>}}


## cs.CY (1)



### (102/122) Legitimate Interest is the New Consent -- Large-Scale Measurement and Legal Compliance of IAB Europe TCF Paywalls (Victor Morel et al., 2023)

{{<citation>}}

Victor Morel, Cristiana Santos, Viktor Fredholm, Adam Thunberg. (2023)  
**Legitimate Interest is the New Consent -- Large-Scale Measurement and Legal Compliance of IAB Europe TCF Paywalls**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2309.11625v2)  

---


**ABSTRACT**  
Cookie paywalls allow visitors of a website to access its content only after they make a choice between paying a fee or accept tracking. European Data Protection Authorities (DPAs) recently issued guidelines and decisions on paywalls lawfulness, but it is yet unknown whether websites comply with them. We study in this paper the prevalence of cookie paywalls on the top one million websites using an automatic crawler. We identify 431 cookie paywalls, all using the Transparency and Consent Framework (TCF). We then analyse the data these paywalls communicate through the TCF, and in particular, the legal grounds and the purposes used to collect personal data. We observe that cookie paywalls extensively rely on legitimate interest legal basis systematically conflated with consent. We also observe a lack of correlation between the presence of paywalls and legal decisions or guidelines by DPAs.

{{</citation>}}


## cs.DL (2)



### (103/122) Large Synthetic Data from the arXiv for OCR Post Correction of Historic Scientific Articles (Jill P. Naiman et al., 2023)

{{<citation>}}

Jill P. Naiman, Morgan G. Cosillo, Peter K. G. Williams, Alyssa Goodman. (2023)  
**Large Synthetic Data from the arXiv for OCR Post Correction of Historic Scientific Articles**  

---
Primary Category: cs.DL  
Categories: astro-ph-IM, cs-DL, cs.DL  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2309.11549v1)  

---


**ABSTRACT**  
Scientific articles published prior to the "age of digitization" (~1997) require Optical Character Recognition (OCR) to transform scanned documents into machine-readable text, a process that often produces errors. We develop a pipeline for the generation of a synthetic ground truth/OCR dataset to correct the OCR results of the astrophysics literature holdings of the NASA Astrophysics Data System (ADS). By mining the arXiv we create, to the authors' knowledge, the largest scientific synthetic ground truth/OCR post correction dataset of 203,354,393 character pairs. We provide baseline models trained with this dataset and find the mean improvement in character and word error rates of 7.71% and 18.82% for historical OCR text, respectively. When used to classify parts of sentences as inline math, we find a classification F1 score of 77.82%. Interactive dashboards to explore the dataset are available online: https://readingtimemachine.github.io/projects/1-ocr-groundtruth-may2023, and data and code, within the limitations of our agreement with the arXiv, are hosted on GitHub: https://github.com/ReadingTimeMachine/ocr_post_correction.

{{</citation>}}


### (104/122) Bravo MaRDI: A Wikibase Powered Knowledge Graph on Mathematics (Moritz Schubotz et al., 2023)

{{<citation>}}

Moritz Schubotz, Eloi Ferrer, Johannes Stegmüller, Daniel Mietchen, Olaf Teschke, Larissa Pusch, Tim OF Conrad. (2023)  
**Bravo MaRDI: A Wikibase Powered Knowledge Graph on Mathematics**  

---
Primary Category: cs.DL  
Categories: cs-DL, cs-IR, cs.DL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.11484v1)  

---


**ABSTRACT**  
Mathematical world knowledge is a fundamental component of Wikidata. However, to date, no expertly curated knowledge graph has focused specifically on contemporary mathematics. Addressing this gap, the Mathematical Research Data Initiative (MaRDI) has developed a comprehensive knowledge graph that links multimodal research data in mathematics. This encompasses traditional research data items like datasets, software, and publications and includes semantically advanced objects such as mathematical formulas and hypotheses. This paper details the abilities of the MaRDI knowledge graph, which is based on Wikibase, leading up to its inaugural public release, codenamed Bravo, available on https://portal.mardi4nfdi.de.

{{</citation>}}


## cs.SD (2)



### (105/122) A Large-scale Dataset for Audio-Language Representation Learning (Luoyi Sun et al., 2023)

{{<citation>}}

Luoyi Sun, Xuenan Xu, Mengyue Wu, Weidi Xie. (2023)  
**A Large-scale Dataset for Audio-Language Representation Learning**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: AI, Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.11500v1)  

---


**ABSTRACT**  
The AI community has made significant strides in developing powerful foundation models, driven by large-scale multimodal datasets. However, in the audio representation learning community, the present audio-language datasets suffer from limitations such as insufficient volume, simplistic content, and arduous collection procedures. To tackle these challenges, we present an innovative and automatic audio caption generation pipeline based on a series of public tools or APIs, and construct a large-scale, high-quality, audio-language dataset, named as Auto-ACD, comprising over 1.9M audio-text pairs. To demonstrate the effectiveness of the proposed dataset, we train popular models on our dataset and show performance improvement on various downstream tasks, namely, audio-language retrieval, audio captioning, environment classification. In addition, we establish a novel test set and provide a benchmark for audio-text tasks. The proposed dataset will be released at https://auto-acd.github.io/.

{{</citation>}}


### (106/122) Directional Source Separation for Robust Speech Recognition on Smart Glasses (Tiantian Feng et al., 2023)

{{<citation>}}

Tiantian Feng, Ju Lin, Yiteng Huang, Weipeng He, Kaustubh Kalgaonkar, Niko Moritz, Li Wan, Xin Lei, Ming Sun, Frank Seide. (2023)  
**Directional Source Separation for Robust Speech Recognition on Smart Glasses**  

---
Primary Category: cs.SD  
Categories: cs-HC, cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.10993v1)  

---


**ABSTRACT**  
Modern smart glasses leverage advanced audio sensing and machine learning technologies to offer real-time transcribing and captioning services, considerably enriching human experiences in daily communications. However, such systems frequently encounter challenges related to environmental noises, resulting in degradation to speech recognition and speaker change detection. To improve voice quality, this work investigates directional source separation using the multi-microphone array. We first explore multiple beamformers to assist source separation modeling by strengthening the directional properties of speech signals. In addition to relying on predetermined beamformers, we investigate neural beamforming in multi-channel source separation, demonstrating that automatic learning directional characteristics effectively improves separation quality. We further compare the ASR performance leveraging separated outputs to noisy inputs. Our results show that directional source separation benefits ASR for the wearer but not for the conversation partner. Lastly, we perform the joint training of the directional source separation and ASR model, achieving the best overall ASR performance.

{{</citation>}}


## cs.CR (2)



### (107/122) AudioFool: Fast, Universal and synchronization-free Cross-Domain Attack on Speech Recognition (Mohamad Fakih et al., 2023)

{{<citation>}}

Mohamad Fakih, Rouwaida Kanj, Fadi Kurdahi, Mohammed E. Fouda. (2023)  
**AudioFool: Fast, Universal and synchronization-free Cross-Domain Attack on Speech Recognition**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs.CR, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.11462v1)  

---


**ABSTRACT**  
Automatic Speech Recognition systems have been shown to be vulnerable to adversarial attacks that manipulate the command executed on the device. Recent research has focused on exploring methods to create such attacks, however, some issues relating to Over-The-Air (OTA) attacks have not been properly addressed. In our work, we examine the needed properties of robust attacks compatible with the OTA model, and we design a method of generating attacks with arbitrary such desired properties, namely the invariance to synchronization, and the robustness to filtering: this allows a Denial-of-Service (DoS) attack against ASR systems. We achieve these characteristics by constructing attacks in a modified frequency domain through an inverse Fourier transform. We evaluate our method on standard keyword classification tasks and analyze it in OTA, and we analyze the properties of the cross-domain attacks to explain the efficiency of the approach.

{{</citation>}}


### (108/122) Tropical cryptography III: digital signatures (Jiale Chen et al., 2023)

{{<citation>}}

Jiale Chen, Dima Grigoriev, Vladimir Shpilrain. (2023)  
**Tropical cryptography III: digital signatures**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR, math-CO  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.11256v1)  

---


**ABSTRACT**  
We use tropical algebras as platforms for a very efficient digital signature protocol. Security relies on computational hardness of factoring one-variable tropical polynomials; this problem is known to be NP-hard.

{{</citation>}}


## q-fin.TR (1)



### (109/122) Transformers versus LSTMs for electronic trading (Paul Bilokon et al., 2023)

{{<citation>}}

Paul Bilokon, Yitao Qiu. (2023)  
**Transformers versus LSTMs for electronic trading**  

---
Primary Category: q-fin.TR  
Categories: cs-LG, econ-EM, q-fin-ST, q-fin-TR, q-fin.TR  
Keywords: LSTM, NLP, Natural Language Processing, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.11400v1)  

---


**ABSTRACT**  
With the rapid development of artificial intelligence, long short term memory (LSTM), one kind of recurrent neural network (RNN), has been widely applied in time series prediction.   Like RNN, Transformer is designed to handle the sequential data. As Transformer achieved great success in Natural Language Processing (NLP), researchers got interested in Transformer's performance on time series prediction, and plenty of Transformer-based solutions on long time series forecasting have come out recently. However, when it comes to financial time series prediction, LSTM is still a dominant architecture. Therefore, the question this study wants to answer is: whether the Transformer-based model can be applied in financial time series prediction and beat LSTM.   To answer this question, various LSTM-based and Transformer-based models are compared on multiple financial prediction tasks based on high-frequency limit order book data. A new LSTM-based model called DLSTM is built and new architecture for the Transformer-based model is designed to adapt for financial prediction. The experiment result reflects that the Transformer-based model only has the limited advantage in absolute price sequence prediction. The LSTM-based models show better and more robust performance on difference sequence prediction, such as price difference and price movement.

{{</citation>}}


## cs.RO (5)



### (110/122) Discuss Before Moving: Visual Language Navigation via Multi-expert Discussions (Yuxing Long et al., 2023)

{{<citation>}}

Yuxing Long, Xiaoqi Li, Wenzhe Cai, Hao Dong. (2023)  
**Discuss Before Moving: Visual Language Navigation via Multi-expert Discussions**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CL, cs-CV, cs-RO, cs.RO  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2309.11382v1)  

---


**ABSTRACT**  
Visual language navigation (VLN) is an embodied task demanding a wide range of skills encompassing understanding, perception, and planning. For such a multifaceted challenge, previous VLN methods totally rely on one model's own thinking to make predictions within one round. However, existing models, even the most advanced large language model GPT4, still struggle with dealing with multiple tasks by single-round self-thinking. In this work, drawing inspiration from the expert consultation meeting, we introduce a novel zero-shot VLN framework. Within this framework, large models possessing distinct abilities are served as domain experts. Our proposed navigation agent, namely DiscussNav, can actively discuss with these experts to collect essential information before moving at every step. These discussions cover critical navigation subtasks like instruction understanding, environment perception, and completion estimation. Through comprehensive experiments, we demonstrate that discussions with domain experts can effectively facilitate navigation by perceiving instruction-relevant information, correcting inadvertent errors, and sifting through in-consistent movement decisions. The performances on the representative VLN task R2R show that our method surpasses the leading zero-shot VLN model by a large margin on all metrics. Additionally, real-robot experiments display the obvious advantages of our method over single-round self-thinking.

{{</citation>}}


### (111/122) Receding-Constraint Model Predictive Control using a Learned Approximate Control-Invariant Set (Gianni Lunardi et al., 2023)

{{<citation>}}

Gianni Lunardi, Asia La Rocca, Matteo Saveriano, Andrea Del Prete. (2023)  
**Receding-Constraint Model Predictive Control using a Learned Approximate Control-Invariant Set**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.11124v1)  

---


**ABSTRACT**  
In recent years, advanced model-based and data-driven control methods are unlocking the potential of complex robotics systems, and we can expect this trend to continue at an exponential rate in the near future. However, ensuring safety with these advanced control methods remains a challenge. A well-known tool to make controllers (either Model Predictive Controllers or Reinforcement Learning policies) safe, is the so-called control-invariant set (a.k.a. safe set). Unfortunately, for nonlinear systems, such a set cannot be exactly computed in general. Numerical algorithms exist for computing approximate control-invariant sets, but classic theoretic control methods break down if the set is not exact. This paper presents our recent efforts to address this issue. We present a novel Model Predictive Control scheme that can guarantee recursive feasibility and/or safety under weaker assumptions than classic methods. In particular, recursive feasibility is guaranteed by making the safe-set constraint move backward over the horizon, and assuming that such set satisfies a condition that is weaker than control invariance. Safety is instead guaranteed under an even weaker assumption on the safe set, triggering a safe task-abortion strategy whenever a risk of constraint violation is detected. We evaluated our approach on a simulated robot manipulator, empirically demonstrating that it leads to less constraint violations than state-of-the-art approaches, while retaining reasonable performance in terms of tracking cost and number of completed tasks.

{{</citation>}}


### (112/122) Safe and Robust Multi-Agent Reinforcement Learning for Connected Autonomous Vehicles under State Perturbations (Zhili Zhang et al., 2023)

{{<citation>}}

Zhili Zhang, Yanchao Sun, Furong Huang, Fei Miao. (2023)  
**Safe and Robust Multi-Agent Reinforcement Learning for Connected Autonomous Vehicles under State Perturbations**  

---
Primary Category: cs.RO  
Categories: cs-MA, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.11057v1)  

---


**ABSTRACT**  
Sensing and communication technologies have enhanced learning-based decision making methodologies for multi-agent systems such as connected autonomous vehicles (CAV). However, most existing safe reinforcement learning based methods assume accurate state information. It remains challenging to achieve safety requirement under state uncertainties for CAVs, considering the noisy sensor measurements and the vulnerability of communication channels. In this work, we propose a Robust Multi-Agent Proximal Policy Optimization with robust Safety Shield (SR-MAPPO) for CAVs in various driving scenarios. Both robust MARL algorithm and control barrier function (CBF)-based safety shield are used in our approach to cope with the perturbed or uncertain state inputs. The robust policy is trained with a worst-case Q function regularization module that pursues higher lower-bounded reward in the former, whereas the latter, i.e., the robust CBF safety shield accounts for CAVs' collision-free constraints in complicated driving scenarios with even perturbed vehicle state information. We validate the advantages of SR-MAPPO in robustness and safety and compare it with baselines under different driving and state perturbation scenarios in CARLA simulator. The SR-MAPPO policy is verified to maintain higher safety rates and efficiency (reward) when threatened by both state perturbations and unconnected vehicles' dangerous behaviors.

{{</citation>}}


### (113/122) CaveSeg: Deep Semantic Segmentation and Scene Parsing for Autonomous Underwater Cave Exploration (A. Abdullah et al., 2023)

{{<citation>}}

A. Abdullah, T. Barua, R. Tibbetts, Z. Chen, M. J. Islam, I. Rekleitis. (2023)  
**CaveSeg: Deep Semantic Segmentation and Scene Parsing for Autonomous Underwater Cave Exploration**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO, eess-IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.11038v2)  

---


**ABSTRACT**  
In this paper, we present CaveSeg - the first visual learning pipeline for semantic segmentation and scene parsing for AUV navigation inside underwater caves. We address the problem of scarce annotated training data by preparing a comprehensive dataset for semantic segmentation of underwater cave scenes. It contains pixel annotations for important navigation markers (e.g. caveline, arrows), obstacles (e.g. ground plain and overhead layers), scuba divers, and open areas for servoing. Through comprehensive benchmark analyses on cave systems in USA, Mexico, and Spain locations, we demonstrate that robust deep visual models can be developed based on CaveSeg for fast semantic scene parsing of underwater cave environments. In particular, we formulate a novel transformer-based model that is computationally light and offers near real-time execution in addition to achieving state-of-the-art performance. Finally, we explore the design choices and implications of semantic segmentation for visual servoing by AUVs inside underwater caves. The proposed model and benchmark dataset open up promising opportunities for future research in autonomous underwater cave exploration and mapping.

{{</citation>}}


### (114/122) STARNet: Sensor Trustworthiness and Anomaly Recognition via Approximated Likelihood Regret for Robust Edge Autonomy (Nastaran Darabi et al., 2023)

{{<citation>}}

Nastaran Darabi, Sina Tayebati, Sureshkumar S., Sathya Ravi, Theja Tulabandhula, Amit R. Trivedi. (2023)  
**STARNet: Sensor Trustworthiness and Anomaly Recognition via Approximated Likelihood Regret for Robust Edge Autonomy**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Anomaly Recognition  
[Paper Link](http://arxiv.org/abs/2309.11006v1)  

---


**ABSTRACT**  
Complex sensors such as LiDAR, RADAR, and event cameras have proliferated in autonomous robotics to enhance perception and understanding of the environment. Meanwhile, these sensors are also vulnerable to diverse failure mechanisms that can intricately interact with their operation environment. In parallel, the limited availability of training data on complex sensors also affects the reliability of their deep learning-based prediction flow, where their prediction models can fail to generalize to environments not adequately captured in the training set. To address these reliability concerns, this paper introduces STARNet, a Sensor Trustworthiness and Anomaly Recognition Network designed to detect untrustworthy sensor streams that may arise from sensor malfunctions and/or challenging environments. We specifically benchmark STARNet on LiDAR and camera data. STARNet employs the concept of approximated likelihood regret, a gradient-free framework tailored for low-complexity hardware, especially those with only fixed-point precision capabilities. Through extensive simulations, we demonstrate the efficacy of STARNet in detecting untrustworthy sensor streams in unimodal and multimodal settings. In particular, the network shows superior performance in addressing internal sensor failures, such as cross-sensor interference and crosstalk. In diverse test scenarios involving adverse weather and sensor malfunctions, we show that STARNet enhances prediction accuracy by approximately 10% by filtering out untrustworthy sensor streams. STARNet is publicly available at \url{https://github.com/sinatayebati/STARNet}.

{{</citation>}}


## cs.GR (1)



### (115/122) C$\cdot$ASE: Learning Conditional Adversarial Skill Embeddings for Physics-based Characters (Zhiyang Dou et al., 2023)

{{<citation>}}

Zhiyang Dou, Xuelin Chen, Qingnan Fan, Taku Komura, Wenping Wang. (2023)  
**C$\cdot$ASE: Learning Conditional Adversarial Skill Embeddings for Physics-based Characters**  

---
Primary Category: cs.GR  
Categories: cs-AI, cs-GR, cs-LG, cs.GR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.11351v1)  

---


**ABSTRACT**  
We present C$\cdot$ASE, an efficient and effective framework that learns conditional Adversarial Skill Embeddings for physics-based characters. Our physically simulated character can learn a diverse repertoire of skills while providing controllability in the form of direct manipulation of the skills to be performed. C$\cdot$ASE divides the heterogeneous skill motions into distinct subsets containing homogeneous samples for training a low-level conditional model to learn conditional behavior distribution. The skill-conditioned imitation learning naturally offers explicit control over the character's skills after training. The training course incorporates the focal skill sampling, skeletal residual forces, and element-wise feature masking to balance diverse skills of varying complexities, mitigate dynamics mismatch to master agile motions and capture more general behavior characteristics, respectively. Once trained, the conditional model can produce highly diverse and realistic skills, outperforming state-of-the-art models, and can be repurposed in various downstream tasks. In particular, the explicit skill control handle allows a high-level policy or user to direct the character with desired skill specifications, which we demonstrate is advantageous for interactive character animation.

{{</citation>}}


## eess.AS (3)



### (116/122) Leveraging Data Collection and Unsupervised Learning for Code-switched Tunisian Arabic Automatic Speech Recognition (Ahmed Amine Ben Abdallah et al., 2023)

{{<citation>}}

Ahmed Amine Ben Abdallah, Ata Kabboudi, Amir Kanoun, Salah Zaiem. (2023)  
**Leveraging Data Collection and Unsupervised Learning for Code-switched Tunisian Arabic Automatic Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.11327v2)  

---


**ABSTRACT**  
Crafting an effective Automatic Speech Recognition (ASR) solution for dialects demands innovative approaches that not only address the data scarcity issue but also navigate the intricacies of linguistic diversity. In this paper, we address the aforementioned ASR challenge, focusing on the Tunisian dialect. First, textual and audio data is collected and in some cases annotated. Second, we explore self-supervision, semi-supervision and few-shot code-switching approaches to push the state-of-the-art on different Tunisian test sets; covering different acoustic, linguistic and prosodic conditions. Finally, and given the absence of conventional spelling, we produce a human evaluation of our transcripts to avoid the noise coming from spelling inadequacies in our testing references. Our models, allowing to transcribe audio samples in a linguistic mix involving Tunisian Arabic, English and French, and all the data used during training and testing are released for public use and further improvements.

{{</citation>}}


### (117/122) Speak While You Think: Streaming Speech Synthesis During Text Generation (Avihu Dekel et al., 2023)

{{<citation>}}

Avihu Dekel, Slava Shechtman, Raul Fernandez, David Haws, Zvi Kons, Ron Hoory. (2023)  
**Speak While You Think: Streaming Speech Synthesis During Text Generation**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Language Model, Text Generation  
[Paper Link](http://arxiv.org/abs/2309.11210v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) demonstrate impressive capabilities, yet interaction with these models is mostly facilitated through text. Using Text-To-Speech to synthesize LLM outputs typically results in notable latency, which is impractical for fluent voice conversations. We propose LLM2Speech, an architecture to synthesize speech while text is being generated by an LLM which yields significant latency reduction. LLM2Speech mimics the predictions of a non-streaming teacher model while limiting the exposure to future context in order to enable streaming. It exploits the hidden embeddings of the LLM, a by-product of the text generation that contains informative semantic context. Experimental results show that LLM2Speech maintains the teacher's quality while reducing the latency to enable natural conversations.

{{</citation>}}


### (118/122) Ensembling Multilingual Pre-Trained Models for Predicting Multi-Label Regression Emotion Share from Speech (Bagus Tris Atmaja et al., 2023)

{{<citation>}}

Bagus Tris Atmaja, Akira Sasou. (2023)  
**Ensembling Multilingual Pre-Trained Models for Predicting Multi-Label Regression Emotion Share from Speech**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess-SP, eess.AS  
Keywords: Multilingual, Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2309.11014v1)  

---


**ABSTRACT**  
Speech emotion recognition has evolved from research to practical applications. Previous studies of emotion recognition from speech have focused on developing models on certain datasets like IEMOCAP. The lack of data in the domain of emotion modeling emerges as a challenge to evaluate models in the other dataset, as well as to evaluate speech emotion recognition models that work in a multilingual setting. This paper proposes an ensemble learning to fuse results of pre-trained models for emotion share recognition from speech. The models were chosen to accommodate multilingual data from English and Spanish. The results show that ensemble learning can improve the performance of the baseline model with a single model and the previous best model from the late fusion. The performance is measured using the Spearman rank correlation coefficient since the task is a regression problem with ranking values. A Spearman rank correlation coefficient of 0.537 is reported for the test set, while for the development set, the score is 0.524. These scores are higher than the previous study of a fusion method from monolingual data, which achieved scores of 0.476 for the test and 0.470 for the development.

{{</citation>}}


## eess.SP (1)



### (119/122) Language-Oriented Communication with Semantic Coding and Knowledge Distillation for Text-to-Image Generation (Hyelin Nam et al., 2023)

{{<citation>}}

Hyelin Nam, Jihong Park, Jinho Choi, Mehdi Bennis, Seong-Lyun Kim. (2023)  
**Language-Oriented Communication with Semantic Coding and Knowledge Distillation for Text-to-Image Generation**  

---
Primary Category: eess.SP  
Categories: cs-AI, cs-CL, eess-SP, eess.SP  
Keywords: Knowledge Distillation, NLP  
[Paper Link](http://arxiv.org/abs/2309.11127v1)  

---


**ABSTRACT**  
By integrating recent advances in large language models (LLMs) and generative models into the emerging semantic communication (SC) paradigm, in this article we put forward to a novel framework of language-oriented semantic communication (LSC). In LSC, machines communicate using human language messages that can be interpreted and manipulated via natural language processing (NLP) techniques for SC efficiency. To demonstrate LSC's potential, we introduce three innovative algorithms: 1) semantic source coding (SSC) which compresses a text prompt into its key head words capturing the prompt's syntactic essence while maintaining their appearance order to keep the prompt's context; 2) semantic channel coding (SCC) that improves robustness against errors by substituting head words with their lenghthier synonyms; and 3) semantic knowledge distillation (SKD) that produces listener-customized prompts via in-context learning the listener's language style. In a communication task for progressive text-to-image generation, the proposed methods achieve higher perceptual similarities with fewer transmissions while enhancing robustness in noisy communication channels.

{{</citation>}}


## eess.SY (1)



### (120/122) Practical Probabilistic Model-based Deep Reinforcement Learning by Integrating Dropout Uncertainty and Trajectory Sampling (Wenjun Huang et al., 2023)

{{<citation>}}

Wenjun Huang, Yunduan Cui, Huiyun Li, Xinyu Wu. (2023)  
**Practical Probabilistic Model-based Deep Reinforcement Learning by Integrating Dropout Uncertainty and Trajectory Sampling**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.11089v1)  

---


**ABSTRACT**  
This paper addresses the prediction stability, prediction accuracy and control capability of the current probabilistic model-based reinforcement learning (MBRL) built on neural networks. A novel approach dropout-based probabilistic ensembles with trajectory sampling (DPETS) is proposed where the system uncertainty is stably predicted by combining the Monte-Carlo dropout and trajectory sampling in one framework. Its loss function is designed to correct the fitting error of neural networks for more accurate prediction of probabilistic models. The state propagation in its policy is extended to filter the aleatoric uncertainty for superior control capability. Evaluated by several Mujoco benchmark control tasks under additional disturbances and one practical robot arm manipulation task, DPETS outperforms related MBRL approaches in both average return and convergence velocity while achieving superior performance than well-known model-free baselines with significant sample efficiency. The open source code of DPETS is available at https://github.com/mrjun123/DPETS.

{{</citation>}}


## q-bio.GN (1)



### (121/122) Embed-Search-Align: DNA Sequence Alignment using Transformer Models (Pavan Holur et al., 2023)

{{<citation>}}

Pavan Holur, K. C. Enevoldsen, Lajoyce Mboning, Thalia Georgiou, Louis-S. Bouchard, Matteo Pellegrini, Vwani Roychowdhury. (2023)  
**Embed-Search-Align: DNA Sequence Alignment using Transformer Models**  

---
Primary Category: q-bio.GN  
Categories: cs-AI, q-bio-GN, q-bio.GN  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2309.11087v1)  

---


**ABSTRACT**  
DNA sequence alignment involves assigning short DNA reads to the most probable locations on an extensive reference genome. This process is crucial for various genomic analyses, including variant calling, transcriptomics, and epigenomics. Conventional methods, refined over decades, tackle this challenge in two steps: genome indexing followed by efficient search to locate likely positions for given reads. Building on the success of Large Language Models (LLM) in encoding text into embeddings, where the distance metric captures semantic similarity, recent efforts have explored whether the same Transformer architecture can produce numerical representations for DNA sequences. Such models have shown early promise in tasks involving classification of short DNA sequences, such as the detection of coding vs non-coding regions, as well as the identification of enhancer and promoter sequences. Performance at sequence classification tasks does not, however, translate to sequence alignment, where it is necessary to conduct a genome-wide search to successfully align every read. We address this open problem by framing it as an Embed-Search-Align task. In this framework, a novel encoder model DNA-ESA generates representations of reads and fragments of the reference, which are projected into a shared vector space where the read-fragment distance is used as surrogate for alignment. In particular, DNA-ESA introduces: (1) Contrastive loss for self-supervised training of DNA sequence representations, facilitating rich sequence-level embeddings, and (2) a DNA vector store to enable search across fragments on a global scale. DNA-ESA is >97% accurate when aligning 250-length reads onto a human reference genome of 3 gigabases (single-haploid), far exceeds the performance of 6 recent DNA-Transformer model baselines and shows task transfer across chromosomes and species.

{{</citation>}}


## cs.DB (1)



### (122/122) ElasticNotebook: Enabling Live Migration for Computational Notebooks (Technical Report) (Zhaoheng Li et al., 2023)

{{<citation>}}

Zhaoheng Li, Pranav Gor, Rahul Prabhu, Hui Yu, Yuzhou Mao, Yongjoo Park. (2023)  
**ElasticNotebook: Enabling Live Migration for Computational Notebooks (Technical Report)**  

---
Primary Category: cs.DB  
Categories: H-2-4, cs-DB, cs.DB  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2309.11083v2)  

---


**ABSTRACT**  
Computational notebooks (e.g., Jupyter, Google Colab) are widely used for interactive data science and machine learning. In those frameworks, users can start a session, then execute cells (i.e., a set of statements) to create variables, train models, visualize results, etc. Unfortunately, existing notebook systems do not offer live migration: when a notebook launches on a new machine, it loses its state, preventing users from continuing their tasks from where they had left off. This is because, unlike DBMS, the sessions directly rely on underlying kernels (e.g., Python/R interpreters) without an additional data management layer. Existing techniques for preserving states, such as copying all variables or OS-level checkpointing, are unreliable (often fail), inefficient, and platform-dependent. Also, re-running code from scratch can be highly time-consuming. In this paper, we introduce a new notebook system, ElasticNotebook, that offers live migration via checkpointing/restoration using a novel mechanism that is reliable, efficient, and platform-independent. Specifically, by observing all cell executions via transparent, lightweight monitoring, \system can find a reliable and efficient way (i.e., replication plan) for reconstructing the original session state, considering variable-cell dependencies, observed runtime, variable sizes, etc. To this end, our new graph-based optimization problem finds how to reconstruct all variables (efficiently) from a subset of variables that can be transferred across machines. We show that ElasticNotebook reduces end-to-end migration and restoration times by 85%-98% and 94%-99%, respectively, on a variety (i.e., Kaggle, JWST, and Tutorial) of notebooks with negligible runtime and memory overheads of <2.5% and <10%.

{{</citation>}}
