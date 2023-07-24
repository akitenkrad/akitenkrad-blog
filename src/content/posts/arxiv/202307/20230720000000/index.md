---
draft: false
title: "arXiv @ 2023.07.20"
date: 2023-07-20
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.20"
    identifier: arxiv_20230720
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.IR (3)](#csir-3)
- [cs.AI (11)](#csai-11)
- [cs.SD (3)](#cssd-3)
- [cs.RO (2)](#csro-2)
- [cs.LG (21)](#cslg-21)
- [cs.CV (32)](#cscv-32)
- [eess.IV (5)](#eessiv-5)
- [cs.CL (16)](#cscl-16)
- [eess.AS (2)](#eessas-2)
- [eess.SY (2)](#eesssy-2)
- [cs.SI (2)](#cssi-2)
- [cs.SE (1)](#csse-1)
- [cs.NI (4)](#csni-4)
- [cs.HC (2)](#cshc-2)
- [cs.ET (1)](#cset-1)
- [cs.CY (1)](#cscy-1)
- [cs.DC (1)](#csdc-1)
- [cs.CR (4)](#cscr-4)
- [math.OC (1)](#mathoc-1)
- [quant-ph (1)](#quant-ph-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.DB (1)](#csdb-1)

## cs.IR (3)



### (1/117) PubMed and Beyond: Recent Advances and Best Practices in Biomedical Literature Search (Qiao Jin et al., 2023)

{{<citation>}}

Qiao Jin, Robert Leaman, Zhiyong Lu. (2023)  
**PubMed and Beyond: Recent Advances and Best Practices in Biomedical Literature Search**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-DL, cs-IR, cs.IR  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.09683v1)  

---


**ABSTRACT**  
Biomedical research yields a wealth of information, much of which is only accessible through the literature. Consequently, literature search is an essential tool for building on prior knowledge in clinical and biomedical research. Although recent improvements in artificial intelligence have expanded functionality beyond keyword-based search, these advances may be unfamiliar to clinicians and researchers. In response, we present a survey of literature search tools tailored to both general and specific information needs in biomedicine, with the objective of helping readers efficiently fulfill their information needs. We first examine the widely used PubMed search engine, discussing recent improvements and continued challenges. We then describe literature search tools catering to five specific information needs: 1. Identifying high-quality clinical research for evidence-based medicine. 2. Retrieving gene-related information for precision medicine and genomics. 3. Searching by meaning, including natural language questions. 4. Locating related articles with literature recommendation. 5. Mining literature to discover associations between concepts such as diseases and genetic variants. Additionally, we cover practical considerations and best practices for choosing and using these tools. Finally, we provide a perspective on the future of literature search engines, considering recent breakthroughs in large language models such as ChatGPT. In summary, our survey provides a comprehensive view of biomedical literature search functionalities with 36 publicly available tools.

{{</citation>}}


### (2/117) Zero-shot Query Reformulation for Conversational Search (Dayu Yang et al., 2023)

{{<citation>}}

Dayu Yang, Yue Zhang, Hui Fang. (2023)  
**Zero-shot Query Reformulation for Conversational Search**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2307.09384v1)  

---


**ABSTRACT**  
As the popularity of voice assistants continues to surge, conversational search has gained increased attention in Information Retrieval. However, data sparsity issues in conversational search significantly hinder the progress of supervised conversational search methods. Consequently, researchers are focusing more on zero-shot conversational search approaches. Nevertheless, existing zero-shot methods face three primary limitations: they are not universally applicable to all retrievers, their effectiveness lacks sufficient explainability, and they struggle to resolve common conversational ambiguities caused by omission. To address these limitations, we introduce a novel Zero-shot Query Reformulation (ZeQR) framework that reformulates queries based on previous dialogue contexts without requiring supervision from conversational search data. Specifically, our framework utilizes language models designed for machine reading comprehension tasks to explicitly resolve two common ambiguities: coreference and omission, in raw queries. In comparison to existing zero-shot methods, our approach is universally applicable to any retriever without additional adaptation or indexing. It also provides greater explainability and effectively enhances query intent understanding because ambiguities are explicitly and proactively resolved. Through extensive experiments on four TREC conversational datasets, we demonstrate the effectiveness of our method, which consistently outperforms state-of-the-art baselines.

{{</citation>}}


### (3/117) Jean-Luc Picard at Touché 2023: Comparing Image Generation, Stance Detection and Feature Matching for Image Retrieval for Arguments (Max Moebius et al., 2023)

{{<citation>}}

Max Moebius, Maximilian Enderling, Sarah T. Bachinger. (2023)  
**Jean-Luc Picard at Touché 2023: Comparing Image Generation, Stance Detection and Feature Matching for Image Retrieval for Arguments**  

---
Primary Category: cs.IR  
Categories: H-3-3, cs-CV, cs-IR, cs.IR  
Keywords: Stance Detection  
[Paper Link](http://arxiv.org/abs/2307.09172v1)  

---


**ABSTRACT**  
Participating in the shared task "Image Retrieval for arguments", we used different pipelines for image retrieval containing Image Generation, Stance Detection, Preselection and Feature Matching. We submitted four different runs with different pipeline layout and compare them to given baseline. Our pipelines perform similarly to the baseline.

{{</citation>}}


## cs.AI (11)



### (4/117) What's meant by explainable model: A Scoping Review (Mallika Mainali et al., 2023)

{{<citation>}}

Mallika Mainali, Rosina O Weber. (2023)  
**What's meant by explainable model: A Scoping Review**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.09673v1)  

---


**ABSTRACT**  
We often see the term explainable in the titles of papers that describe applications based on artificial intelligence (AI). However, the literature in explainable artificial intelligence (XAI) indicates that explanations in XAI are application- and domain-specific, hence requiring evaluation whenever they are employed to explain a model that makes decisions for a specific application problem. Additionally, the literature reveals that the performance of post-hoc methods, particularly feature attribution methods, varies substantially hinting that they do not represent a solution to AI explainability. Therefore, when using XAI methods, the quality and suitability of their information outputs should be evaluated within the specific application. For these reasons, we used a scoping review methodology to investigate papers that apply AI models and adopt methods to generate post-hoc explanations while referring to said models as explainable. This paper investigates whether the term explainable model is adopted by authors under the assumption that incorporating a post-hoc XAI method suffices to characterize a model as explainable. To inspect this problem, our review analyzes whether these papers conducted evaluations. We found that 81% of the application papers that refer to their approaches as an explainable model do not conduct any form of evaluation on the XAI method they used.

{{</citation>}}


### (5/117) Balancing Privacy and Progress in Artificial Intelligence: Anonymization in Histopathology for Biomedical Research and Education (Neel Kanwal et al., 2023)

{{<citation>}}

Neel Kanwal, Emiel A. M. Janssen, Kjersti Engan. (2023)  
**Balancing Privacy and Progress in Artificial Intelligence: Anonymization in Histopathology for Biomedical Research and Education**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CE, cs-CR, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.09426v1)  

---


**ABSTRACT**  
The advancement of biomedical research heavily relies on access to large amounts of medical data. In the case of histopathology, Whole Slide Images (WSI) and clinicopathological information are valuable for developing Artificial Intelligence (AI) algorithms for Digital Pathology (DP). Transferring medical data "as open as possible" enhances the usability of the data for secondary purposes but poses a risk to patient privacy. At the same time, existing regulations push towards keeping medical data "as closed as necessary" to avoid re-identification risks. Generally, these legal regulations require the removal of sensitive data but do not consider the possibility of data linkage attacks due to modern image-matching algorithms. In addition, the lack of standardization in DP makes it harder to establish a single solution for all formats of WSIs. These challenges raise problems for bio-informatics researchers in balancing privacy and progress while developing AI algorithms. This paper explores the legal regulations and terminologies for medical data-sharing. We review existing approaches and highlight challenges from the histopathological perspective. We also present a data-sharing guideline for histological data to foster multidisciplinary research and education.

{{</citation>}}


### (6/117) Company2Vec -- German Company Embeddings based on Corporate Websites (Christopher Gerling, 2023)

{{<citation>}}

Christopher Gerling. (2023)  
**Company2Vec -- German Company Embeddings based on Corporate Websites**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, q-fin-CP, q-fin-PM  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.09332v1)  

---


**ABSTRACT**  
With Company2Vec, the paper proposes a novel application in representation learning. The model analyzes business activities from unstructured company website data using Word2Vec and dimensionality reduction. Company2Vec maintains semantic language structures and thus creates efficient company embeddings in fine-granular industries. These semantic embeddings can be used for various applications in banking. Direct relations between companies and words allow semantic business analytics (e.g. top-n words for a company). Furthermore, industry prediction is presented as a supervised learning application and evaluation method. The vectorized structure of the embeddings allows measuring companies similarities with the cosine distance. Company2Vec hence offers a more fine-grained comparison of companies than the standard industry labels (NACE). This property is relevant for unsupervised learning tasks, such as clustering. An alternative industry segmentation is shown with k-means clustering on the company embeddings. Finally, this paper proposes three algorithms for (1) firm-centric, (2) industry-centric and (3) portfolio-centric peer-firm identification.

{{</citation>}}


### (7/117) Rumor Detection with Diverse Counterfactual Evidence (Kaiwei Zhang et al., 2023)

{{<citation>}}

Kaiwei Zhang, Junchi Yu, Haichao Shi, Jian Liang, Xiao-Yu Zhang. (2023)  
**Rumor Detection with Diverse Counterfactual Evidence**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2307.09296v1)  

---


**ABSTRACT**  
The growth in social media has exacerbated the threat of fake news to individuals and communities. This draws increasing attention to developing efficient and timely rumor detection methods. The prevailing approaches resort to graph neural networks (GNNs) to exploit the post-propagation patterns of the rumor-spreading process. However, these methods lack inherent interpretation of rumor detection due to the black-box nature of GNNs. Moreover, these methods suffer from less robust results as they employ all the propagation patterns for rumor detection. In this paper, we address the above issues with the proposed Diverse Counterfactual Evidence framework for Rumor Detection (DCE-RD). Our intuition is to exploit the diverse counterfactual evidence of an event graph to serve as multi-view interpretations, which are further aggregated for robust rumor detection results. Specifically, our method first designs a subgraph generation strategy to efficiently generate different subgraphs of the event graph. We constrain the removal of these subgraphs to cause the change in rumor detection results. Thus, these subgraphs naturally serve as counterfactual evidence for rumor detection. To achieve multi-view interpretation, we design a diversity loss inspired by Determinantal Point Processes (DPP) to encourage diversity among the counterfactual evidence. A GNN-based rumor detection model further aggregates the diverse counterfactual evidence discovered by the proposed DCE-RD to achieve interpretable and robust rumor detection results. Extensive experiments on two real-world datasets show the superior performance of our method. Our code is available at https://github.com/Vicinity111/DCE-RD.

{{</citation>}}


### (8/117) ESMC: Entire Space Multi-Task Model for Post-Click Conversion Rate via Parameter Constraint (Zhenhao Jiang et al., 2023)

{{<citation>}}

Zhenhao Jiang, Biao Zeng, Hao Feng, Jin Liu, Jicong Fan, Jie Zhang, Jia Jia, Ning Hu, Xingyu Chen, Xuguang Lan. (2023)  
**ESMC: Entire Space Multi-Task Model for Post-Click Conversion Rate via Parameter Constraint**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IR, cs.AI  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.09193v1)  

---


**ABSTRACT**  
Large-scale online recommender system spreads all over the Internet being in charge of two basic tasks: Click-Through Rate (CTR) and Post-Click Conversion Rate (CVR) estimations. However, traditional CVR estimators suffer from well-known Sample Selection Bias and Data Sparsity issues. Entire space models were proposed to address the two issues via tracing the decision-making path of "exposure_click_purchase". Further, some researchers observed that there are purchase-related behaviors between click and purchase, which can better draw the user's decision-making intention and improve the recommendation performance. Thus, the decision-making path has been extended to "exposure_click_in-shop action_purchase" and can be modeled with conditional probability approach. Nevertheless, we observe that the chain rule of conditional probability does not always hold. We report Probability Space Confusion (PSC) issue and give a derivation of difference between ground-truth and estimation mathematically. We propose a novel Entire Space Multi-Task Model for Post-Click Conversion Rate via Parameter Constraint (ESMC) and two alternatives: Entire Space Multi-Task Model with Siamese Network (ESMS) and Entire Space Multi-Task Model in Global Domain (ESMG) to address the PSC issue. Specifically, we handle "exposure_click_in-shop action" and "in-shop action_purchase" separately in the light of characteristics of in-shop action. The first path is still treated with conditional probability while the second one is treated with parameter constraint strategy. Experiments on both offline and online environments in a large-scale recommendation system illustrate the superiority of our proposed methods over state-of-the-art models. The real-world datasets will be released.

{{</citation>}}


### (9/117) QMNet: Importance-Aware Message Exchange for Decentralized Multi-Agent Reinforcement Learning (Xiufeng Huang et al., 2023)

{{<citation>}}

Xiufeng Huang, Sheng Zhou. (2023)  
**QMNet: Importance-Aware Message Exchange for Decentralized Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.09051v1)  

---


**ABSTRACT**  
To improve the performance of multi-agent reinforcement learning under the constraint of wireless resources, we propose a message importance metric and design an importance-aware scheduling policy to effectively exchange messages. The key insight is spending the precious communication resources on important messages. The message importance depends not only on the messages themselves, but also on the needs of agents who receive them. Accordingly, we propose a query-message-based architecture, called QMNet. Agents generate queries and messages with the environment observation. Sharing queries can help calculate message importance. Exchanging messages can help agents cooperate better. Besides, we exploit the message importance to deal with random access collisions in decentralized systems. Furthermore, a message prediction mechanism is proposed to compensate for messages that are not transmitted. Finally, we evaluate the proposed schemes in a traffic junction environment, where only a fraction of agents can send messages due to limited wireless resources. Results show that QMNet can extract valuable information to guarantee the system performance even when only $30\%$ of agents can share messages. By exploiting message prediction, the system can further save $40\%$ of wireless resources. The importance-aware decentralized multi-access mechanism can effectively avoid collisions, achieving almost the same performance as centralized scheduling.

{{</citation>}}


### (10/117) Multimodal Machine Learning for Extraction of Theorems and Proofs in the Scientific Literature (Shrey Mishra et al., 2023)

{{<citation>}}

Shrey Mishra, Antoine Gauquier, Pierre Senellart. (2023)  
**Multimodal Machine Learning for Extraction of Theorems and Proofs in the Scientific Literature**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: BERT, LSTM  
[Paper Link](http://arxiv.org/abs/2307.09047v1)  

---


**ABSTRACT**  
Scholarly articles in mathematical fields feature mathematical statements such as theorems, propositions, etc., as well as their proofs. Extracting them from the PDF representation of the articles requires understanding of scientific text along with visual and font-based indicators. We pose this problem as a multimodal classification problem using text, font features, and bitmap image rendering of the PDF as different modalities. In this paper we propose a multimodal machine learning approach for extraction of theorem-like environments and proofs, based on late fusion of features extracted by individual unimodal classifiers, taking into account the sequential succession of blocks in the document. For the text modality, we pretrain a new language model on a 11 GB scientific corpus; experiments shows similar performance for our task than a model (RoBERTa) pretrained on 160 GB, with faster convergence while requiring much less fine-tuning data. Font-based information relies on training a 128-cell LSTM on the sequence of font names and sizes within each block. Bitmap renderings are dealt with using an EfficientNetv2 deep network tuned to classify each image block. Finally, a simple CRF-based approach uses the features of the multimodal model along with information on block sequences. Experimental results show the benefits of using a multimodal approach vs any single modality, as well as major performance improvements using the CRF modeling of block sequences.

{{</citation>}}


### (11/117) Emotional Intelligence of Large Language Models (Xuena Wang et al., 2023)

{{<citation>}}

Xuena Wang, Xueting Li, Zi Yin, Yue Wu, Liu Jia. (2023)  
**Emotional Intelligence of Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.09042v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable abilities across numerous disciplines, primarily assessed through tasks in language generation, knowledge utilization, and complex reasoning. However, their alignment with human emotions and values, which is critical for real-world applications, has not been systematically evaluated. Here, we assessed LLMs' Emotional Intelligence (EI), encompassing emotion recognition, interpretation, and understanding, which is necessary for effective communication and social interactions. Specifically, we first developed a novel psychometric assessment focusing on Emotion Understanding (EU), a core component of EI, suitable for both humans and LLMs. This test requires evaluating complex emotions (e.g., surprised, joyful, puzzled, proud) in realistic scenarios (e.g., despite feeling underperformed, John surprisingly achieved a top score). With a reference frame constructed from over 500 adults, we tested a variety of mainstream LLMs. Most achieved above-average EQ scores, with GPT-4 exceeding 89% of human participants with an EQ of 117. Interestingly, a multivariate pattern analysis revealed that some LLMs apparently did not reply on the human-like mechanism to achieve human-level performance, as their representational patterns were qualitatively distinct from humans. In addition, we discussed the impact of factors such as model size, training method, and architecture on LLMs' EQ. In summary, our study presents one of the first psychometric evaluations of the human-like characteristics of LLMs, which may shed light on the future development of LLMs aiming for both high intellectual and emotional intelligence. Project website: https://emotional-intelligence.github.io/

{{</citation>}}


### (12/117) Development of the ChatGPT, Generative Artificial Intelligence and Natural Large Language Models for Accountable Reporting and Use (CANGARU) Guidelines (Giovanni E. Cacciamani et al., 2023)

{{<citation>}}

Giovanni E. Cacciamani, Michael B. Eppler, Conner Ganjavi, Asli Pekan, Brett Biedermann, Gary S. Collins, Inderbir S. Gill. (2023)  
**Development of the ChatGPT, Generative Artificial Intelligence and Natural Large Language Models for Accountable Reporting and Use (CANGARU) Guidelines**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: AI, ChatGPT, GPT, Generative AI, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.08974v1)  

---


**ABSTRACT**  
The swift progress and ubiquitous adoption of Generative AI (GAI), Generative Pre-trained Transformers (GPTs), and large language models (LLMs) like ChatGPT, have spurred queries about their ethical application, use, and disclosure in scholarly research and scientific productions. A few publishers and journals have recently created their own sets of rules; however, the absence of a unified approach may lead to a 'Babel Tower Effect,' potentially resulting in confusion rather than desired standardization. In response to this, we present the ChatGPT, Generative Artificial Intelligence, and Natural Large Language Models for Accountable Reporting and Use Guidelines (CANGARU) initiative, with the aim of fostering a cross-disciplinary global inclusive consensus on the ethical use, disclosure, and proper reporting of GAI/GPT/LLM technologies in academia. The present protocol consists of four distinct parts: a) an ongoing systematic review of GAI/GPT/LLM applications to understand the linked ideas, findings, and reporting standards in scholarly research, and to formulate guidelines for its use and disclosure, b) a bibliometric analysis of existing author guidelines in journals that mention GAI/GPT/LLM, with the goal of evaluating existing guidelines, analyzing the disparity in their recommendations, and identifying common rules that can be brought into the Delphi consensus process, c) a Delphi survey to establish agreement on the items for the guidelines, ensuring principled GAI/GPT/LLM use, disclosure, and reporting in academia, and d) the subsequent development and dissemination of the finalized guidelines and their supplementary explanation and elaboration documents.

{{</citation>}}


### (13/117) REX: Rapid Exploration and eXploitation for AI Agents (Rithesh Murthy et al., 2023)

{{<citation>}}

Rithesh Murthy, Shelby Heinecke, Juan Carlos Niebles, Zhiwei Liu, Le Xue, Weiran Yao, Yihao Feng, Zeyuan Chen, Akash Gokul, Devansh Arpit, Ran Xu, Phil Mui, Huan Wang, Caiming Xiong, Silvio Savarese. (2023)  
**REX: Rapid Exploration and eXploitation for AI Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, GPT, Reasoning, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08962v1)  

---


**ABSTRACT**  
In this paper, we propose an enhanced approach for Rapid Exploration and eXploitation for AI Agents called REX. Existing AutoGPT-style techniques have inherent limitations, such as a heavy reliance on precise descriptions for decision-making, and the lack of a systematic approach to leverage try-and-fail procedures akin to traditional Reinforcement Learning (RL). REX introduces an additional layer of rewards and integrates concepts similar to Upper Confidence Bound (UCB) scores, leading to more robust and efficient AI agent performance. This approach has the advantage of enabling the utilization of offline behaviors from logs and allowing seamless integration with existing foundation models while it does not require any model fine-tuning. Through comparative analysis with existing methods such as Chain-of-Thoughts(CoT) and Reasoning viA Planning(RAP), REX-based methods demonstrate comparable performance and, in certain cases, even surpass the results achieved by these existing techniques. Notably, REX-based methods exhibit remarkable reductions in execution time, enhancing their practical applicability across a diverse set of scenarios.

{{</citation>}}


### (14/117) IxDRL: A Novel Explainable Deep Reinforcement Learning Toolkit based on Analyses of Interestingness (Pedro Sequeira et al., 2023)

{{<citation>}}

Pedro Sequeira, Melinda Gervasio. (2023)  
**IxDRL: A Novel Explainable Deep Reinforcement Learning Toolkit based on Analyses of Interestingness**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08933v1)  

---


**ABSTRACT**  
In recent years, advances in deep learning have resulted in a plethora of successes in the use of reinforcement learning (RL) to solve complex sequential decision tasks with high-dimensional inputs. However, existing systems lack the necessary mechanisms to provide humans with a holistic view of their competence, presenting an impediment to their adoption, particularly in critical applications where the decisions an agent makes can have significant consequences. Yet, existing RL-based systems are essentially competency-unaware in that they lack the necessary interpretation mechanisms to allow human operators to have an insightful, holistic view of their competency. Towards more explainable Deep RL (xDRL), we propose a new framework based on analyses of interestingness. Our tool provides various measures of RL agent competence stemming from interestingness analysis and is applicable to a wide range of RL algorithms, natively supporting the popular RLLib toolkit. We showcase the use of our framework by applying the proposed pipeline in a set of scenarios of varying complexity. We empirically assess the capability of the approach in identifying agent behavior patterns and competency-controlling conditions, and the task elements mostly responsible for an agent's competence, based on global and local analyses of interestingness. Overall, we show that our framework can provide agent designers with insights about RL agent competence, both their capabilities and limitations, enabling more informed decisions about interventions, additional training, and other interactions in collaborative human-machine settings.

{{</citation>}}


## cs.SD (3)



### (15/117) JAZZVAR: A Dataset of Variations found within Solo Piano Performances of Jazz Standards for Music Overpainting (Eleanor Row et al., 2023)

{{<citation>}}

Eleanor Row, Jingjing Tang, George Fazekas. (2023)  
**JAZZVAR: A Dataset of Variations found within Solo Piano Performances of Jazz Standards for Music Overpainting**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.09670v1)  

---


**ABSTRACT**  
Jazz pianists often uniquely interpret jazz standards. Passages from these interpretations can be viewed as sections of variation. We manually extracted such variations from solo jazz piano performances. The JAZZVAR dataset is a collection of 502 pairs of Variation and Original MIDI segments. Each Variation in the dataset is accompanied by a corresponding Original segment containing the melody and chords from the original jazz standard. Our approach differs from many existing jazz datasets in the music information retrieval (MIR) community, which often focus on improvisation sections within jazz performances. In this paper, we outline the curation process for obtaining and sorting the repertoire, the pipeline for creating the Original and Variation pairs, and our analysis of the dataset. We also introduce a new generative music task, Music Overpainting, and present a baseline Transformer model trained on the JAZZVAR dataset for this task. Other potential applications of our dataset include expressive performance analysis and performer identification.

{{</citation>}}


### (16/117) FlexiAST: Flexibility is What AST Needs (Jiu Feng et al., 2023)

{{<citation>}}

Jiu Feng, Mehmet Hamza Erol, Joon Son Chung, Arda Senocak. (2023)  
**FlexiAST: Flexibility is What AST Needs**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.09286v1)  

---


**ABSTRACT**  
The objective of this work is to give patch-size flexibility to Audio Spectrogram Transformers (AST). Recent advancements in ASTs have shown superior performance in various audio-based tasks. However, the performance of standard ASTs degrades drastically when evaluated using different patch sizes from that used during training. As a result, AST models are typically re-trained to accommodate changes in patch sizes. To overcome this limitation, this paper proposes a training procedure to provide flexibility to standard AST models without architectural changes, allowing them to work with various patch sizes at the inference stage - FlexiAST. This proposed training approach simply utilizes random patch size selection and resizing of patch and positional embedding weights. Our experiments show that FlexiAST gives similar performance to standard AST models while maintaining its evaluation ability at various patch sizes on different datasets for audio classification tasks.

{{</citation>}}


### (17/117) OxfordVGG Submission to the EGO4D AV Transcription Challenge (Jaesung Huh et al., 2023)

{{<citation>}}

Jaesung Huh, Max Bain, Andrew Zisserman. (2023)  
**OxfordVGG Submission to the EGO4D AV Transcription Challenge**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.09006v1)  

---


**ABSTRACT**  
This report presents the technical details of our submission on the EGO4D Audio-Visual (AV) Automatic Speech Recognition Challenge 2023 from the OxfordVGG team. We present WhisperX, a system for efficient speech transcription of long-form audio with word-level time alignment, along with two text normalisers which are publicly available. Our final submission obtained 56.0% of the Word Error Rate (WER) on the challenge test set, ranked 1st on the leaderboard. All baseline codes and models are available on https://github.com/m-bain/whisperX.

{{</citation>}}


## cs.RO (2)



### (18/117) Towards A Unified Agent with Foundation Models (Norman Di Palo et al., 2023)

{{<citation>}}

Norman Di Palo, Arunkumar Byravan, Leonard Hasenclever, Markus Wulfmeier, Nicolas Heess, Martin Riedmiller. (2023)  
**Towards A Unified Agent with Foundation Models**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.09668v1)  

---


**ABSTRACT**  
Language Models and Vision Language Models have recently demonstrated unprecedented capabilities in terms of understanding human intentions, reasoning, scene understanding, and planning-like behaviour, in text form, among many others. In this work, we investigate how to embed and leverage such abilities in Reinforcement Learning (RL) agents. We design a framework that uses language as the core reasoning tool, exploring how this enables an agent to tackle a series of fundamental RL challenges, such as efficient exploration, reusing experience data, scheduling skills, and learning from observations, which traditionally require separate, vertically designed algorithms. We test our method on a sparse-reward simulated robotic manipulation environment, where a robot needs to stack a set of objects. We demonstrate substantial performance improvements over baselines in exploration efficiency and ability to reuse data from offline datasets, and illustrate how to reuse learned skills to solve novel tasks or imitate videos of human experts.

{{</citation>}}


### (19/117) Task Space Control of Hydraulic Construction Machines using Reinforcement Learning (Hyung Joo Lee et al., 2023)

{{<citation>}}

Hyung Joo Lee, Sigrid Brell-Cokcan. (2023)  
**Task Space Control of Hydraulic Construction Machines using Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.09246v2)  

---


**ABSTRACT**  
Teleoperation is vital in the construction industry, allowing safe machine manipulation from a distance. However, controlling machines at a joint level requires extensive training due to their complex degrees of freedom. Task space control offers intuitive maneuvering, but precise control often requires dynamic models, posing challenges for hydraulic machines. To address this, we use a data-driven actuator model to capture machine dynamics in real-world operations. By integrating this model into simulation and reinforcement learning, an optimal control policy for task space control is obtained. Experiments with Brokk 170 validate the framework, comparing it to a well-known Jacobian-based approach.

{{</citation>}}


## cs.LG (21)



### (20/117) Anticipating Technical Expertise and Capability Evolution in Research Communities using Dynamic Graph Transformers (Sameera Horawalavithana et al., 2023)

{{<citation>}}

Sameera Horawalavithana, Ellyn Ayton, Anastasiya Usenko, Robin Cosbey, Svitlana Volkova. (2023)  
**Anticipating Technical Expertise and Capability Evolution in Research Communities using Dynamic Graph Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.09665v1)  

---


**ABSTRACT**  
The ability to anticipate technical expertise and capability evolution trends globally is essential for national and global security, especially in safety-critical domains like nuclear nonproliferation (NN) and rapidly emerging fields like artificial intelligence (AI). In this work, we extend traditional statistical relational learning approaches (e.g., link prediction in collaboration networks) and formulate a problem of anticipating technical expertise and capability evolution using dynamic heterogeneous graph representations. We develop novel capabilities to forecast collaboration patterns, authorship behavior, and technical capability evolution at different granularities (e.g., scientist and institution levels) in two distinct research fields. We implement a dynamic graph transformer (DGT) neural architecture, which pushes the state-of-the-art graph neural network models by (a) forecasting heterogeneous (rather than homogeneous) nodes and edges, and (b) relying on both discrete -- and continuous -- time inputs. We demonstrate that our DGT models predict collaboration, partnership, and expertise patterns with 0.26, 0.73, and 0.53 mean reciprocal rank values for AI and 0.48, 0.93, and 0.22 for NN domains. DGT model performance exceeds the best-performing static graph baseline models by 30-80% across AI and NN domains. Our findings demonstrate that DGT models boost inductive task performance, when previously unseen nodes appear in the test data, for the domains with emerging collaboration patterns (e.g., AI). Specifically, models accurately predict which established scientists will collaborate with early career scientists and vice-versa in the AI domain.

{{</citation>}}


### (21/117) Neural Priority Queues for Graph Neural Networks (Rishabh Jain et al., 2023)

{{<citation>}}

Rishabh Jain, Petar Veličković, Pietro Liò. (2023)  
**Neural Priority Queues for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.09660v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have shown considerable success in neural algorithmic reasoning. Many traditional algorithms make use of an explicit memory in the form of a data structure. However, there has been limited exploration on augmenting GNNs with external memory. In this paper, we present Neural Priority Queues, a differentiable analogue to algorithmic priority queues, for GNNs. We propose and motivate a desiderata for memory modules, and show that Neural PQs exhibit the desiderata, and reason about their use with algorithmic reasoning. This is further demonstrated by empirical results on the CLRS-30 dataset. Furthermore, we find the Neural PQs useful in capturing long-range interactions, as empirically shown on a dataset from the Long-Range Graph Benchmark.

{{</citation>}}


### (22/117) HAT-CL: A Hard-Attention-to-the-Task PyTorch Library for Continual Learning (Xiaotian Duan, 2023)

{{<citation>}}

Xiaotian Duan. (2023)  
**HAT-CL: A Hard-Attention-to-the-Task PyTorch Library for Continual Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.09653v1)  

---


**ABSTRACT**  
Catastrophic forgetting, the phenomenon in which a neural network loses previously obtained knowledge during the learning of new tasks, poses a significant challenge in continual learning. The Hard-Attention-to-the-Task (HAT) mechanism has shown potential in mitigating this problem, but its practical implementation has been complicated by issues of usability and compatibility, and a lack of support for existing network reuse. In this paper, we introduce HAT-CL, a user-friendly, PyTorch-compatible redesign of the HAT mechanism. HAT-CL not only automates gradient manipulation but also streamlines the transformation of PyTorch modules into HAT modules. It achieves this by providing a comprehensive suite of modules that can be seamlessly integrated into existing architectures. Additionally, HAT-CL offers ready-to-use HAT networks that are smoothly integrated with the TIMM library. Beyond the redesign and reimplementation of HAT, we also introduce novel mask manipulation techniques for HAT, which have consistently shown improvements across various experiments. Our work paves the way for a broader application of the HAT mechanism, opening up new possibilities in continual learning across diverse models and applications.

{{</citation>}}


### (23/117) Overthinking the Truth: Understanding how Language Models Process False Demonstrations (Danny Halawi et al., 2023)

{{<citation>}}

Danny Halawi, Jean-Stanislas Denain, Jacob Steinhardt. (2023)  
**Overthinking the Truth: Understanding how Language Models Process False Demonstrations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.09476v1)  

---


**ABSTRACT**  
Modern language models can imitate complex patterns through few-shot learning, enabling them to complete challenging tasks without fine-tuning. However, imitation can also lead models to reproduce inaccuracies or harmful content if present in the context. We study harmful imitation through the lens of a model's internal representations, and identify two related phenomena: overthinking and false induction heads. The first phenomenon, overthinking, appears when we decode predictions from intermediate layers, given correct vs. incorrect few-shot demonstrations. At early layers, both demonstrations induce similar model behavior, but the behavior diverges sharply at some "critical layer", after which the accuracy given incorrect demonstrations progressively decreases. The second phenomenon, false induction heads, are a possible mechanistic cause of overthinking: these are heads in late layers that attend to and copy false information from previous demonstrations, and whose ablation reduces overthinking. Beyond scientific understanding, our results suggest that studying intermediate model computations could be a promising avenue for understanding and guarding against harmful model behaviors.

{{</citation>}}


### (24/117) Unsupervised Conditional Slot Attention for Object Centric Learning (Avinash Kori et al., 2023)

{{<citation>}}

Avinash Kori, Francesco Locatello, Francesca Toni, Ben Glocker. (2023)  
**Unsupervised Conditional Slot Attention for Object Centric Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2307.09437v1)  

---


**ABSTRACT**  
Extracting object-level representations for downstream reasoning tasks is an emerging area in AI. Learning object-centric representations in an unsupervised setting presents multiple challenges, a key one being binding an arbitrary number of object instances to a specialized object slot. Recent object-centric representation methods like Slot Attention utilize iterative attention to learn composable representations with dynamic inference level binding but fail to achieve specialized slot level binding. To address this, in this paper we propose Unsupervised Conditional Slot Attention using a novel Probabilistic Slot Dictionary (PSD). We define PSD with (i) abstract object-level property vectors as key and (ii) parametric Gaussian distribution as its corresponding value. We demonstrate the benefits of the learnt specific object-level conditioning distributions in multiple downstream tasks, namely object discovery, compositional scene generation, and compositional visual reasoning. We show that our method provides scene composition capabilities and a significant boost in a few shot adaptability tasks of compositional visual reasoning, while performing similarly or better than slot attention in object discovery tasks

{{</citation>}}


### (25/117) Scaling Laws for Imitation Learning in NetHack (Jens Tuyls et al., 2023)

{{<citation>}}

Jens Tuyls, Dhruv Madeka, Kari Torkkola, Dean Foster, Karthik Narasimhan, Sham Kakade. (2023)  
**Scaling Laws for Imitation Learning in NetHack**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: AI, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.09423v1)  

---


**ABSTRACT**  
Imitation Learning (IL) is one of the most widely used methods in machine learning. Yet, while powerful, many works find it is often not able to fully recover the underlying expert behavior. However, none of these works deeply investigate the role of scaling up the model and data size. Inspired by recent work in Natural Language Processing (NLP) where "scaling up" has resulted in increasingly more capable LLMs, we investigate whether carefully scaling up model and data size can bring similar improvements in the imitation learning setting. To demonstrate our findings, we focus on the game of NetHack, a challenging environment featuring procedural generation, stochasticity, long-term dependencies, and partial observability. We find IL loss and mean return scale smoothly with the compute budget and are strongly correlated, resulting in power laws for training compute-optimal IL agents with respect to model size and number of samples. We forecast and train several NetHack agents with IL and find they outperform prior state-of-the-art by at least 2x in all settings. Our work both demonstrates the scaling behavior of imitation learning in a challenging domain, as well as the viability of scaling up current approaches for increasingly capable agents in NetHack, a game that remains elusively hard for current AI systems.

{{</citation>}}


### (26/117) Data Cross-Segmentation for Improved Generalization in Reinforcement Learning Based Algorithmic Trading (Vikram Duvvur et al., 2023)

{{<citation>}}

Vikram Duvvur, Aashay Mehta, Edward Sun, Bo Wu, Ken Yew Chan, Jeff Schneider. (2023)  
**Data Cross-Segmentation for Improved Generalization in Reinforcement Learning Based Algorithmic Trading**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.09377v1)  

---


**ABSTRACT**  
The use of machine learning in algorithmic trading systems is increasingly common. In a typical set-up, supervised learning is used to predict the future prices of assets, and those predictions drive a simple trading and execution strategy. This is quite effective when the predictions have sufficient signal, markets are liquid, and transaction costs are low. However, those conditions often do not hold in thinly traded financial markets and markets for differentiated assets such as real estate or vehicles. In these markets, the trading strategy must consider the long-term effects of taking positions that are relatively more difficult to change. In this work, we propose a Reinforcement Learning (RL) algorithm that trades based on signals from a learned predictive model and addresses these challenges. We test our algorithm on 20+ years of equity data from Bursa Malaysia.

{{</citation>}}


### (27/117) Conformal prediction under ambiguous ground truth (David Stutz et al., 2023)

{{<citation>}}

David Stutz, Abhijit Guha Roy, Tatiana Matejovicova, Patricia Strachan, Ali Taylan Cemgil, Arnaud Doucet. (2023)  
**Conformal prediction under ambiguous ground truth**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG, stat-ME, stat-ML  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.09302v1)  

---


**ABSTRACT**  
In safety-critical classification tasks, conformal prediction allows to perform rigorous uncertainty quantification by providing confidence sets including the true class with a user-specified probability. This generally assumes the availability of a held-out calibration set with access to ground truth labels. Unfortunately, in many domains, such labels are difficult to obtain and usually approximated by aggregating expert opinions. In fact, this holds true for almost all datasets, including well-known ones such as CIFAR and ImageNet. Applying conformal prediction using such labels underestimates uncertainty. Indeed, when expert opinions are not resolvable, there is inherent ambiguity present in the labels. That is, we do not have ``crisp'', definitive ground truth labels and this uncertainty should be taken into account during calibration. In this paper, we develop a conformal prediction framework for such ambiguous ground truth settings which relies on an approximation of the underlying posterior distribution of labels given inputs. We demonstrate our methodology on synthetic and real datasets, including a case study of skin condition classification in dermatology.

{{</citation>}}


### (28/117) PAC Neural Prediction Set Learning to Quantify the Uncertainty of Generative Language Models (Sangdon Park et al., 2023)

{{<citation>}}

Sangdon Park, Taesoo Kim. (2023)  
**PAC Neural Prediction Set Learning to Quantify the Uncertainty of Generative Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: GLM, Language Model  
[Paper Link](http://arxiv.org/abs/2307.09254v1)  

---


**ABSTRACT**  
Uncertainty learning and quantification of models are crucial tasks to enhance the trustworthiness of the models. Importantly, the recent surge of generative language models (GLMs) emphasizes the need for reliable uncertainty quantification due to the concerns on generating hallucinated facts. In this paper, we propose to learn neural prediction set models that comes with the probably approximately correct (PAC) guarantee for quantifying the uncertainty of GLMs. Unlike existing prediction set models, which are parameterized by a scalar value, we propose to parameterize prediction sets via neural networks, which achieves more precise uncertainty quantification but still satisfies the PAC guarantee. We demonstrate the efficacy of our method on four types of language datasets and six types of models by showing that our method improves the quantified uncertainty by $63\%$ on average, compared to a standard baseline method.

{{</citation>}}


### (29/117) UniTabE: Pretraining a Unified Tabular Encoder for Heterogeneous Tabular Data (Yazheng Yang et al., 2023)

{{<citation>}}

Yazheng Yang, Yuqi Wang, Guang Liu, Ledell Wu, Qi Liu. (2023)  
**UniTabE: Pretraining a Unified Tabular Encoder for Heterogeneous Tabular Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: NLP, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2307.09249v1)  

---


**ABSTRACT**  
Recent advancements in Natural Language Processing (NLP) have witnessed the groundbreaking impact of pretrained models, yielding impressive outcomes across various tasks. This study seeks to extend the power of pretraining methodologies to tabular data, a domain traditionally overlooked, yet inherently challenging due to the plethora of table schemas intrinsic to different tasks. The primary research questions underpinning this work revolve around the adaptation to heterogeneous table structures, the establishment of a universal pretraining protocol for tabular data, the generalizability and transferability of learned knowledge across tasks, the adaptation to diverse downstream applications, and the incorporation of incremental columns over time. In response to these challenges, we introduce UniTabE, a pioneering method designed to process tables in a uniform manner, devoid of constraints imposed by specific table structures. UniTabE's core concept relies on representing each basic table element with a module, termed TabUnit. This is subsequently followed by a Transformer encoder to refine the representation. Moreover, our model is designed to facilitate pretraining and finetuning through the utilization of free-form prompts. In order to implement the pretraining phase, we curated an expansive tabular dataset comprising approximately 13 billion samples, meticulously gathered from the Kaggle platform. Rigorous experimental testing and analyses were performed under a myriad of scenarios to validate the effectiveness of our methodology. The experimental results demonstrate UniTabE's superior performance against several baseline models across a multitude of benchmark datasets. This, therefore, underscores UniTabE's potential to significantly enhance the semantic representation of tabular data, thereby marking a significant stride in the field of tabular data analysis.

{{</citation>}}


### (30/117) Application of BERT in Wind Power Forecasting-Teletraan's Solution in Baidu KDD Cup 2022 (Longxing Tan et al., 2023)

{{<citation>}}

Longxing Tan, Hongying Yue. (2023)  
**Application of BERT in Wind Power Forecasting-Teletraan's Solution in Baidu KDD Cup 2022**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.09248v1)  

---


**ABSTRACT**  
Nowadays, wind energy has drawn increasing attention as its important role in carbon neutrality and sustainable development. When wind power is integrated into the power grid, precise forecasting is necessary for the sustainability and security of the system. However, the unpredictable nature and long sequence prediction make it especially challenging. In this technical report, we introduce the BERT model applied for Baidu KDD Cup 2022, and the daily fluctuation is added by post-processing to make the predicted results in line with daily periodicity. Our solution achieves 3rd place of 2490 teams. The code is released athttps://github.com/LongxingTan/KDD2022-Baidu

{{</citation>}}


### (31/117) Learning Dynamic Attribute-factored World Models for Efficient Multi-object Reinforcement Learning (Fan Feng et al., 2023)

{{<citation>}}

Fan Feng, Sara Magliacane. (2023)  
**Learning Dynamic Attribute-factored World Models for Efficient Multi-object Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.09205v1)  

---


**ABSTRACT**  
In many reinforcement learning tasks, the agent has to learn to interact with many objects of different types and generalize to unseen combinations and numbers of objects. Often a task is a composition of previously learned tasks (e.g. block stacking). These are examples of compositional generalization, in which we compose object-centric representations to solve complex tasks. Recent works have shown the benefits of object-factored representations and hierarchical abstractions for improving sample efficiency in these settings. On the other hand, these methods do not fully exploit the benefits of factorization in terms of object attributes. In this paper, we address this opportunity and introduce the Dynamic Attribute FacTored RL (DAFT-RL) framework. In DAFT-RL, we leverage object-centric representation learning to extract objects from visual inputs. We learn to classify them in classes and infer their latent parameters. For each class of object, we learn a class template graph that describes how the dynamics and reward of an object of this class factorize according to its attributes. We also learn an interaction pattern graph that describes how objects of different classes interact with each other at the attribute level. Through these graphs and a dynamic interaction graph that models the interactions between objects, we can learn a policy that can then be directly applied in a new environment by just estimating the interactions and latent parameters. We evaluate DAFT-RL in three benchmark datasets and show our framework outperforms the state-of-the-art in generalizing across unseen objects with varying attributes and latent parameters, as well as in the composition of previously learned tasks.

{{</citation>}}


### (32/117) Mining of Single-Class by Active Learning for Semantic Segmentation (Hugues Lambert et al., 2023)

{{<citation>}}

Hugues Lambert, Emma Slade. (2023)  
**Mining of Single-Class by Active Learning for Semantic Segmentation**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Active Learning, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.09109v1)  

---


**ABSTRACT**  
Several Active Learning (AL) policies require retraining a target model several times in order to identify the most informative samples and rarely offer the option to focus on the acquisition of samples from underrepresented classes. Here the Mining of Single-Class by Active Learning (MiSiCAL) paradigm is introduced where an AL policy is constructed through deep reinforcement learning and exploits quantity-accuracy correlations to build datasets on which high-performance models can be trained with regards to specific classes. MiSiCAL is especially helpful in the case of very large batch sizes since it does not require repeated model training sessions as is common in other AL methods. This is thanks to its ability to exploit fixed representations of the candidate data points. We find that MiSiCAL is able to outperform a random policy on 150 out of 171 COCO10k classes, while the strongest baseline only outperforms random on 101 classes.

{{</citation>}}


### (33/117) DiTTO: Diffusion-inspired Temporal Transformer Operator (Oded Ovadia et al., 2023)

{{<citation>}}

Oded Ovadia, Eli Turkel, Adar Kahana, George Em Karniadakis. (2023)  
**DiTTO: Diffusion-inspired Temporal Transformer Operator**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CE, cs-LG, cs-NA, cs.LG, math-NA  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.09072v1)  

---


**ABSTRACT**  
Solving partial differential equations (PDEs) using a data-driven approach has become increasingly common. The recent development of the operator learning paradigm has enabled the solution of a broader range of PDE-related problems. We propose an operator learning method to solve time-dependent PDEs continuously in time without needing any temporal discretization. The proposed approach, named DiTTO, is inspired by latent diffusion models. While diffusion models are usually used in generative artificial intelligence tasks, their time-conditioning mechanism is extremely useful for PDEs. The diffusion-inspired framework is combined with elements from the Transformer architecture to improve its capabilities.   We demonstrate the effectiveness of the new approach on a wide variety of PDEs in multiple dimensions, namely the 1-D Burgers' equation, 2-D Navier-Stokes equations, and the acoustic wave equation in 2-D and 3-D. DiTTO achieves state-of-the-art results in terms of accuracy for these problems. We also present a method to improve the performance of DiTTO by using fast sampling concepts from diffusion models. Finally, we show that DiTTO can accurately perform zero-shot super-resolution in time.

{{</citation>}}


### (34/117) U-shaped Transformer: Retain High Frequency Context in Time Series Analysis (Qingkui Chen et al., 2023)

{{<citation>}}

Qingkui Chen, Yiqin Zhang. (2023)  
**U-shaped Transformer: Retain High Frequency Context in Time Series Analysis**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: NLP, Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2307.09019v1)  

---


**ABSTRACT**  
Time series prediction plays a crucial role in various industrial fields. In recent years, neural networks with a transformer backbone have achieved remarkable success in many domains, including computer vision and NLP. In time series analysis domain, some studies have suggested that even the simplest MLP networks outperform advanced transformer-based networks on time series forecast tasks. However, we believe these findings indicate there to be low-rank properties in time series sequences. In this paper, we consider the low-pass characteristics of transformers and try to incorporate the advantages of MLP. We adopt skip-layer connections inspired by Unet into traditional transformer backbone, thus preserving high-frequency context from input to output, namely U-shaped Transformer. We introduce patch merge and split operation to extract features with different scales and use larger datasets to fully make use of the transformer backbone. Our experiments demonstrate that the model performs at an advanced level across multiple datasets with relatively low cost.

{{</citation>}}


### (35/117) Neural Network Pruning as Spectrum Preserving Process (Shibo Yao et al., 2023)

{{<citation>}}

Shibo Yao, Dantong Yu, Ioannis Koutis. (2023)  
**Neural Network Pruning as Spectrum Preserving Process**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2307.08982v1)  

---


**ABSTRACT**  
Neural networks have achieved remarkable performance in various application domains. Nevertheless, a large number of weights in pre-trained deep neural networks prohibit them from being deployed on smartphones and embedded systems. It is highly desirable to obtain lightweight versions of neural networks for inference in edge devices. Many cost-effective approaches were proposed to prune dense and convolutional layers that are common in deep neural networks and dominant in the parameter space. However, a unified theoretical foundation for the problem mostly is missing. In this paper, we identify the close connection between matrix spectrum learning and neural network training for dense and convolutional layers and argue that weight pruning is essentially a matrix sparsification process to preserve the spectrum. Based on the analysis, we also propose a matrix sparsification algorithm tailored for neural network pruning that yields better pruning result. We carefully design and conduct experiments to support our arguments. Hence we provide a consolidated viewpoint for neural network pruning and enhance the interpretability of deep neural networks by identifying and preserving the critical neural weights.

{{</citation>}}


### (36/117) Mitigating Label Bias via Decoupled Confident Learning (Yunyi Li et al., 2023)

{{<citation>}}

Yunyi Li, Maria De-Arteaga, Maytal Saar-Tsechansky. (2023)  
**Mitigating Label Bias via Decoupled Confident Learning**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.08945v1)  

---


**ABSTRACT**  
Growing concerns regarding algorithmic fairness have led to a surge in methodologies to mitigate algorithmic bias. However, such methodologies largely assume that observed labels in training data are correct. This is problematic because bias in labels is pervasive across important domains, including healthcare, hiring, and content moderation. In particular, human-generated labels are prone to encoding societal biases. While the presence of labeling bias has been discussed conceptually, there is a lack of methodologies to address this problem. We propose a pruning method -- Decoupled Confident Learning (DeCoLe) -- specifically designed to mitigate label bias. After illustrating its performance on a synthetic dataset, we apply DeCoLe in the context of hate speech detection, where label bias has been recognized as an important challenge, and show that it successfully identifies biased labels and outperforms competing approaches.

{{</citation>}}


### (37/117) NTK-approximating MLP Fusion for Efficient Language Model Fine-tuning (Tianxin Wei et al., 2023)

{{<citation>}}

Tianxin Wei, Zeming Guo, Yifan Chen, Jingrui He. (2023)  
**NTK-approximating MLP Fusion for Efficient Language Model Fine-tuning**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model, NLU  
[Paper Link](http://arxiv.org/abs/2307.08941v1)  

---


**ABSTRACT**  
Fine-tuning a pre-trained language model (PLM) emerges as the predominant strategy in many natural language processing applications. However, even fine-tuning the PLMs and doing inference are expensive, especially on edge devices with low computing power. Some general approaches (e.g. quantization and distillation) have been widely studied to reduce the compute/memory of PLM fine-tuning, while very few one-shot compression techniques are explored. In this paper, we investigate the neural tangent kernel (NTK)--which reveals the gradient descent dynamics of neural networks--of the multilayer perceptrons (MLP) modules in a PLM and propose to coin a lightweight PLM through NTK-approximating MLP fusion. To achieve this, we reconsider the MLP as a bundle of sub-MLPs, and cluster them into a given number of centroids, which can then be restored as a compressed MLP and surprisingly shown to well approximate the NTK of the original PLM. Extensive experiments of PLM fine-tuning on both natural language understanding (NLU) and generation (NLG) tasks are provided to verify the effectiveness of the proposed method MLP fusion. Our code is available at https://github.com/weitianxin/MLP_Fusion.

{{</citation>}}


### (38/117) Federated Large Language Model: A Position Paper (Chaochao Chen et al., 2023)

{{<citation>}}

Chaochao Chen, Xiaohua Feng, Jun Zhou, Jianwei Yin, Xiaolin Zheng. (2023)  
**Federated Large Language Model: A Position Paper**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.08925v1)  

---


**ABSTRACT**  
Large scale language models (LLM) have received significant attention and found diverse applications across various domains, but their development encounters challenges in real-world scenarios. These challenges arise due to the scarcity of public domain data availability and the need to maintain privacy with respect to private domain data. To address these issues, federated learning (FL) has emerged as a promising technology that enables collaborative training of shared models while preserving decentralized data. We propose the concept of federated LLM, which comprises three key components, i.e., federated LLM pre-training, federated LLM fine-tuning, and federated LLM prompt engineering. For each component, we discuss its advantage over traditional LLM training methods and propose specific engineering strategies for implementation. Furthermore, we explore the novel challenges introduced by the integration of FL and LLM. We analyze existing solutions and identify potential obstacles faced by these solutions within the context of federated LLM.

{{</citation>}}


### (39/117) Towards the Sparseness of Projection Head in Self-Supervised Learning (Zeen Song et al., 2023)

{{<citation>}}

Zeen Song, Xingzhe Su, Jingyao Wang, Wenwen Qiang, Changwen Zheng, Fuchun Sun. (2023)  
**Towards the Sparseness of Projection Head in Self-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.08913v2)  

---


**ABSTRACT**  
In recent years, self-supervised learning (SSL) has emerged as a promising approach for extracting valuable representations from unlabeled data. One successful SSL method is contrastive learning, which aims to bring positive examples closer while pushing negative examples apart. Many current contrastive learning approaches utilize a parameterized projection head. Through a combination of empirical analysis and theoretical investigation, we provide insights into the internal mechanisms of the projection head and its relationship with the phenomenon of dimensional collapse. Our findings demonstrate that the projection head enhances the quality of representations by performing contrastive loss in a projected subspace. Therefore, we propose an assumption that only a subset of features is necessary when minimizing the contrastive loss of a mini-batch of data. Theoretical analysis further suggests that a sparse projection head can enhance generalization, leading us to introduce SparseHead - a regularization term that effectively constrains the sparsity of the projection head, and can be seamlessly integrated with any self-supervised learning (SSL) approaches. Our experimental results validate the effectiveness of SparseHead, demonstrating its ability to improve the performance of existing contrastive methods.

{{</citation>}}


### (40/117) Sharpness-Aware Graph Collaborative Filtering (Huiyuan Chen et al., 2023)

{{<citation>}}

Huiyuan Chen, Chin-Chia Michael Yeh, Yujie Fan, Yan Zheng, Junpeng Wang, Vivian Lai, Mahashweta Das, Hao Yang. (2023)  
**Sharpness-Aware Graph Collaborative Filtering**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.08910v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have achieved impressive performance in collaborative filtering. However, GNNs tend to yield inferior performance when the distributions of training and test data are not aligned well. Also, training GNNs requires optimizing non-convex neural networks with an abundance of local and global minima, which may differ widely in their performance at test time. Thus, it is essential to choose the minima carefully. Here we propose an effective training schema, called {gSAM}, under the principle that the \textit{flatter} minima has a better generalization ability than the \textit{sharper} ones. To achieve this goal, gSAM regularizes the flatness of the weight loss landscape by forming a bi-level optimization: the outer problem conducts the standard model training while the inner problem helps the model jump out of the sharp minima. Experimental results show the superiority of our gSAM.

{{</citation>}}


## cs.CV (32)



### (41/117) Object-aware Gaze Target Detection (Francesco Tonini et al., 2023)

{{<citation>}}

Francesco Tonini, Nicola Dall'Asen, Cigdem Beyan, Elisa Ricci. (2023)  
**Object-aware Gaze Target Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.09662v1)  

---


**ABSTRACT**  
Gaze target detection aims to predict the image location where the person is looking and the probability that a gaze is out of the scene. Several works have tackled this task by regressing a gaze heatmap centered on the gaze location, however, they overlooked decoding the relationship between the people and the gazed objects. This paper proposes a Transformer-based architecture that automatically detects objects (including heads) in the scene to build associations between every head and the gazed-head/object, resulting in a comprehensive, explainable gaze analysis composed of: gaze target area, gaze pixel point, the class and the image location of the gazed-object. Upon evaluation of the in-the-wild benchmarks, our method achieves state-of-the-art results on all metrics (up to 2.91% gain in AUC, 50% reduction in gaze distance, and 9% gain in out-of-frame average precision) for gaze target detection and 11-13% improvement in average precision for the classification and the localization of the gazed-objects. The code of the proposed method is available https://github.com/francescotonini/object-aware-gaze-target-detection

{{</citation>}}


### (42/117) Traffic-Domain Video Question Answering with Automatic Captioning (Ehsan Qasemi et al., 2023)

{{<citation>}}

Ehsan Qasemi, Jonathan M. Francis, Alessandro Oltramari. (2023)  
**Traffic-Domain Video Question Answering with Automatic Captioning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.09636v1)  

---


**ABSTRACT**  
Video Question Answering (VidQA) exhibits remarkable potential in facilitating advanced machine reasoning capabilities within the domains of Intelligent Traffic Monitoring and Intelligent Transportation Systems. Nevertheless, the integration of urban traffic scene knowledge into VidQA systems has received limited attention in previous research endeavors. In this work, we present a novel approach termed Traffic-domain Video Question Answering with Automatic Captioning (TRIVIA), which serves as a weak-supervision technique for infusing traffic-domain knowledge into large video-language models. Empirical findings obtained from the SUTD-TrafficQA task highlight the substantial enhancements achieved by TRIVIA, elevating the accuracy of representative video-language models by a remarkable 6.5 points (19.88%) compared to baseline settings. This pioneering methodology holds great promise for driving advancements in the field, inspiring researchers and practitioners alike to unlock the full potential of emerging video-language models in traffic-related applications.

{{</citation>}}


### (43/117) Surgical Action Triplet Detection by Mixed Supervised Learning of Instrument-Tissue Interactions (Saurav Sharma et al., 2023)

{{<citation>}}

Saurav Sharma, Chinedu Innocent Nwoye, Didier Mutter, Nicolas Padoy. (2023)  
**Surgical Action Triplet Detection by Mixed Supervised Learning of Instrument-Tissue Interactions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2307.09548v1)  

---


**ABSTRACT**  
Surgical action triplets describe instrument-tissue interactions as (instrument, verb, target) combinations, thereby supporting a detailed analysis of surgical scene activities and workflow. This work focuses on surgical action triplet detection, which is challenging but more precise than the traditional triplet recognition task as it consists of joint (1) localization of surgical instruments and (2) recognition of the surgical action triplet associated with every localized instrument. Triplet detection is highly complex due to the lack of spatial triplet annotation. We analyze how the amount of instrument spatial annotations affects triplet detection and observe that accurate instrument localization does not guarantee better triplet detection due to the risk of erroneous associations with the verbs and targets. To solve the two tasks, we propose MCIT-IG, a two-stage network, that stands for Multi-Class Instrument-aware Transformer-Interaction Graph. The MCIT stage of our network models per class embedding of the targets as additional features to reduce the risk of misassociating triplets. Furthermore, the IG stage constructs a bipartite dynamic graph to model the interaction between the instruments and targets, cast as the verbs. We utilize a mixed-supervised learning strategy that combines weak target presence labels for MCIT and pseudo triplet labels for IG to train our network. We observed that complementing minimal instrument spatial annotations with target embeddings results in better triplet detection. We evaluate our model on the CholecT50 dataset and show improved performance on both instrument localization and triplet detection, topping the leaderboard of the CholecTriplet challenge in MICCAI 2022.

{{</citation>}}


### (44/117) Adversarial Bayesian Augmentation for Single-Source Domain Generalization (Sheng Cheng et al., 2023)

{{<citation>}}

Sheng Cheng, Tejas Gokhale, Yezhou Yang. (2023)  
**Adversarial Bayesian Augmentation for Single-Source Domain Generalization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.09520v1)  

---


**ABSTRACT**  
Generalizing to unseen image domains is a challenging problem primarily due to the lack of diverse training data, inaccessible target data, and the large domain shift that may exist in many real-world settings. As such data augmentation is a critical component of domain generalization methods that seek to address this problem. We present Adversarial Bayesian Augmentation (ABA), a novel algorithm that learns to generate image augmentations in the challenging single-source domain generalization setting. ABA draws on the strengths of adversarial learning and Bayesian neural networks to guide the generation of diverse data augmentations -- these synthesized image domains aid the classifier in generalizing to unseen domains. We demonstrate the strength of ABA on several types of domain shift including style shift, subpopulation shift, and shift in the medical imaging setting. ABA outperforms all previous state-of-the-art methods, including pre-specified augmentations, pixel-based and convolutional-based augmentations.

{{</citation>}}


### (45/117) Occlusion Aware Student Emotion Recognition based on Facial Action Unit Detection (Shrouk Wally et al., 2023)

{{<citation>}}

Shrouk Wally, Ahmed Elsayed, Islam Alkabbany, Asem Ali, Aly Farag. (2023)  
**Occlusion Aware Student Emotion Recognition based on Facial Action Unit Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2307.09465v1)  

---


**ABSTRACT**  
Given that approximately half of science, technology, engineering, and mathematics (STEM) undergraduate students in U.S. colleges and universities leave by the end of the first year [15], it is crucial to improve the quality of classroom environments. This study focuses on monitoring students' emotions in the classroom as an indicator of their engagement and proposes an approach to address this issue. The impact of different facial parts on the performance of an emotional recognition model is evaluated through experimentation. To test the proposed model under partial occlusion, an artificially occluded dataset is introduced. The novelty of this work lies in the proposal of an occlusion-aware architecture for facial action units (AUs) extraction, which employs attention mechanism and adaptive feature learning. The AUs can be used later to classify facial expressions in classroom settings.   This research paper's findings provide valuable insights into handling occlusion in analyzing facial images for emotional engagement analysis. The proposed experiments demonstrate the significance of considering occlusion and enhancing the reliability of facial analysis models in classroom environments. These findings can also be extended to other settings where occlusions are prevalent.

{{</citation>}}


### (46/117) A comparative analysis of SRGAN models (Fatemeh Rezapoor Nikroo et al., 2023)

{{<citation>}}

Fatemeh Rezapoor Nikroo, Ajinkya Deshmukh, Anantha Sharma, Adrian Tam, Kaarthik Kumar, Cleo Norris, Aditya Dangi. (2023)  
**A comparative analysis of SRGAN models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV, eess-IV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2307.09456v2)  

---


**ABSTRACT**  
In this study, we evaluate the performance of multiple state-of-the-art SRGAN (Super Resolution Generative Adversarial Network) models, ESRGAN, Real-ESRGAN and EDSR, on a benchmark dataset of real-world images which undergo degradation using a pipeline. Our results show that some models seem to significantly increase the resolution of the input images while preserving their visual quality, this is assessed using Tesseract OCR engine. We observe that EDSR-BASE model from huggingface outperforms the remaining candidate models in terms of both quantitative metrics and subjective visual quality assessments with least compute overhead. Specifically, EDSR generates images with higher peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM) values and are seen to return high quality OCR results with Tesseract OCR engine. These findings suggest that EDSR is a robust and effective approach for single-image super-resolution and may be particularly well-suited for applications where high-quality visual fidelity is critical and optimized compute.

{{</citation>}}


### (47/117) Let's ViCE! Mimicking Human Cognitive Behavior in Image Generation Evaluation (Federico Betti et al., 2023)

{{<citation>}}

Federico Betti, Jacopo Staiano, Lorenzo Baraldi, Lorenzo Baraldi, Rita Cucchiara, Nicu Sebe. (2023)  
**Let's ViCE! Mimicking Human Cognitive Behavior in Image Generation Evaluation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.09416v2)  

---


**ABSTRACT**  
Research in Image Generation has recently made significant progress, particularly boosted by the introduction of Vision-Language models which are able to produce high-quality visual content based on textual inputs. Despite ongoing advancements in terms of generation quality and realism, no methodical frameworks have been defined yet to quantitatively measure the quality of the generated content and the adherence with the prompted requests: so far, only human-based evaluations have been adopted for quality satisfaction and for comparing different generative methods. We introduce a novel automated method for Visual Concept Evaluation (ViCE), i.e. to assess consistency between a generated/edited image and the corresponding prompt/instructions, with a process inspired by the human cognitive behaviour. ViCE combines the strengths of Large Language Models (LLMs) and Visual Question Answering (VQA) into a unified pipeline, aiming to replicate the human cognitive process in quality assessment. This method outlines visual concepts, formulates image-specific verification questions, utilizes the Q&A system to investigate the image, and scores the combined outcome. Although this brave new hypothesis of mimicking humans in the image evaluation process is in its preliminary assessment stage, results are promising and open the door to a new form of automatic evaluation which could have significant impact as the image generation or the image target editing tasks become more and more sophisticated.

{{</citation>}}


### (48/117) Disentangle then Parse:Night-time Semantic Segmentation with Illumination Disentanglement (Zhixiang Wei et al., 2023)

{{<citation>}}

Zhixiang Wei, Lin Chen, Tao Tu, Huaian Chen, Pengyang Ling, Yi Jin. (2023)  
**Disentangle then Parse:Night-time Semantic Segmentation with Illumination Disentanglement**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.09362v2)  

---


**ABSTRACT**  
Most prior semantic segmentation methods have been developed for day-time scenes, while typically underperforming in night-time scenes due to insufficient and complicated lighting conditions. In this work, we tackle this challenge by proposing a novel night-time semantic segmentation paradigm, i.e., disentangle then parse (DTP). DTP explicitly disentangles night-time images into light-invariant reflectance and light-specific illumination components and then recognizes semantics based on their adaptive fusion. Concretely, the proposed DTP comprises two key components: 1) Instead of processing lighting-entangled features as in prior works, our Semantic-Oriented Disentanglement (SOD) framework enables the extraction of reflectance component without being impeded by lighting, allowing the network to consistently recognize the semantics under cover of varying and complicated lighting conditions. 2) Based on the observation that the illumination component can serve as a cue for some semantically confused regions, we further introduce an Illumination-Aware Parser (IAParser) to explicitly learn the correlation between semantics and lighting, and aggregate the illumination features to yield more precise predictions. Extensive experiments on the night-time segmentation task with various settings demonstrate that DTP significantly outperforms state-of-the-art methods. Furthermore, with negligible additional parameters, DTP can be directly used to benefit existing day-time methods for night-time segmentation.

{{</citation>}}


### (49/117) MOCA: Self-supervised Representation Learning by Predicting Masked Online Codebook Assignments (Spyros Gidaris et al., 2023)

{{<citation>}}

Spyros Gidaris, Andrei Bursuc, Oriane Simeoni, Antonin Vobecky, Nikos Komodakis, Matthieu Cord, Patrick Pérez. (2023)  
**MOCA: Self-supervised Representation Learning by Predicting Masked Online Codebook Assignments**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Representation Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2307.09361v1)  

---


**ABSTRACT**  
Self-supervised learning can be used for mitigating the greedy needs of Vision Transformer networks for very large fully-annotated datasets. Different classes of self-supervised learning offer representations with either good contextual reasoning properties, e.g., using masked image modeling strategies, or invariance to image perturbations, e.g., with contrastive methods. In this work, we propose a single-stage and standalone method, MOCA, which unifies both desired properties using novel mask-and-predict objectives defined with high-level features (instead of pixel-level details). Moreover, we show how to effectively employ both learning paradigms in a synergistic and computation-efficient way. Doing so, we achieve new state-of-the-art results on low-shot settings and strong experimental results in various evaluation protocols with a training that is at least 3 times faster than prior methods.

{{</citation>}}


### (50/117) Towards a performance analysis on pre-trained Visual Question Answering models for autonomous driving (Kaavya Rekanar et al., 2023)

{{<citation>}}

Kaavya Rekanar, Ciarán Eising, Ganesh Sistu, Martin Hayes. (2023)  
**Towards a performance analysis on pre-trained Visual Question Answering models for autonomous driving**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BERT, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.09329v1)  

---


**ABSTRACT**  
This short paper presents a preliminary analysis of three popular Visual Question Answering (VQA) models, namely ViLBERT, ViLT, and LXMERT, in the context of answering questions relating to driving scenarios. The performance of these models is evaluated by comparing the similarity of responses to reference answers provided by computer vision experts. Model selection is predicated on the analysis of transformer utilization in multimodal architectures. The results indicate that models incorporating cross-modal attention and late fusion techniques exhibit promising potential for generating improved answers within a driving perspective. This initial analysis serves as a launchpad for a forthcoming comprehensive comparative study involving nine VQA models and sets the scene for further investigations into the effectiveness of VQA model queries in self-driving scenarios. Supplementary material is available at https://github.com/KaavyaRekanar/Towards-a-performance-analysis-on-pre-trained-VQA-models-for-autonomous-driving.

{{</citation>}}


### (51/117) Efficient Region-Aware Neural Radiance Fields for High-Fidelity Talking Portrait Synthesis (Jiahe Li et al., 2023)

{{<citation>}}

Jiahe Li, Jiawei Zhang, Xiao Bai, Jun Zhou, Lin Gu. (2023)  
**Efficient Region-Aware Neural Radiance Fields for High-Fidelity Talking Portrait Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.09323v1)  

---


**ABSTRACT**  
This paper presents ER-NeRF, a novel conditional Neural Radiance Fields (NeRF) based architecture for talking portrait synthesis that can concurrently achieve fast convergence, real-time rendering, and state-of-the-art performance with small model size. Our idea is to explicitly exploit the unequal contribution of spatial regions to guide talking portrait modeling. Specifically, to improve the accuracy of dynamic head reconstruction, a compact and expressive NeRF-based Tri-Plane Hash Representation is introduced by pruning empty spatial regions with three planar hash encoders. For speech audio, we propose a Region Attention Module to generate region-aware condition feature via an attention mechanism. Different from existing methods that utilize an MLP-based encoder to learn the cross-modal relation implicitly, the attention mechanism builds an explicit connection between audio features and spatial regions to capture the priors of local motions. Moreover, a direct and fast Adaptive Pose Encoding is introduced to optimize the head-torso separation problem by mapping the complex transformation of the head pose into spatial coordinates. Extensive experiments demonstrate that our method renders better high-fidelity and audio-lips synchronized talking portrait videos, with realistic details and high efficiency compared to previous methods.

{{</citation>}}


### (52/117) MarS3D: A Plug-and-Play Motion-Aware Model for Semantic Segmentation on Multi-Scan 3D Point Clouds (Jiahui Liu et al., 2023)

{{<citation>}}

Jiahui Liu, Chirui Chang, Jianhui Liu, Xiaoyang Wu, Lan Ma, Xiaojuan Qi. (2023)  
**MarS3D: A Plug-and-Play Motion-Aware Model for Semantic Segmentation on Multi-Scan 3D Point Clouds**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.09316v1)  

---


**ABSTRACT**  
3D semantic segmentation on multi-scan large-scale point clouds plays an important role in autonomous systems. Unlike the single-scan-based semantic segmentation task, this task requires distinguishing the motion states of points in addition to their semantic categories. However, methods designed for single-scan-based segmentation tasks perform poorly on the multi-scan task due to the lacking of an effective way to integrate temporal information. We propose MarS3D, a plug-and-play motion-aware module for semantic segmentation on multi-scan 3D point clouds. This module can be flexibly combined with single-scan models to allow them to have multi-scan perception abilities. The model encompasses two key designs: the Cross-Frame Feature Embedding module for enriching representation learning and the Motion-Aware Feature Learning module for enhancing motion awareness. Extensive experiments show that MarS3D can improve the performance of the baseline model by a large margin. The code is available at https://github.com/CVMI-Lab/MarS3D.

{{</citation>}}


### (53/117) RepViT: Revisiting Mobile CNN From ViT Perspective (Ao Wang et al., 2023)

{{<citation>}}

Ao Wang, Hui Chen, Zijia Lin, Hengjun Pu, Guiguang Ding. (2023)  
**RepViT: Revisiting Mobile CNN From ViT Perspective**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.09283v1)  

---


**ABSTRACT**  
Recently, lightweight Vision Transformers (ViTs) demonstrate superior performance and lower latency compared with lightweight Convolutional Neural Networks (CNNs) on resource-constrained mobile devices. This improvement is usually attributed to the multi-head self-attention module, which enables the model to learn global representations. However, the architectural disparities between lightweight ViTs and lightweight CNNs have not been adequately examined. In this study, we revisit the efficient design of lightweight CNNs and emphasize their potential for mobile devices. We incrementally enhance the mobile-friendliness of a standard lightweight CNN, specifically MobileNetV3, by integrating the efficient architectural choices of lightweight ViTs. This ends up with a new family of pure lightweight CNNs, namely RepViT. Extensive experiments show that RepViT outperforms existing state-of-the-art lightweight ViTs and exhibits favorable latency in various vision tasks. On ImageNet, RepViT achieves over 80\% top-1 accuracy with nearly 1ms latency on an iPhone 12, which is the first time for a lightweight model, to the best of our knowledge. Our largest model, RepViT-M3, obtains 81.4\% accuracy with only 1.3ms latency. The code and trained models are available at \url{https://github.com/jameslahm/RepViT}.

{{</citation>}}


### (54/117) Regression-free Blind Image Quality Assessment (Xiaoqi Wang et al., 2023)

{{<citation>}}

Xiaoqi Wang, Jian Xiong, Hao Gao, Weisi Lin. (2023)  
**Regression-free Blind Image Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.09279v1)  

---


**ABSTRACT**  
Regression-based blind image quality assessment (IQA) models are susceptible to biased training samples, leading to a biased estimation of model parameters. To mitigate this issue, we propose a regression-free framework for image quality evaluation, which is founded upon retrieving similar instances by incorporating semantic and distortion features. The motivation behind this approach is rooted in the observation that the human visual system (HVS) has analogous visual responses to semantically similar image contents degraded by the same distortion. The proposed framework comprises two classification-based modules: semantic-based classification (SC) module and distortion-based classification (DC) module. Given a test image and an IQA database, the SC module retrieves multiple pristine images based on semantic similarity. The DC module then retrieves instances based on distortion similarity from the distorted images that correspond to each retrieved pristine image. Finally, the predicted quality score is derived by aggregating the subjective quality scores of multiple retrieved instances. Experimental results on four benchmark databases validate that the proposed model can remarkably outperform the state-of-the-art regression-based models.

{{</citation>}}


### (55/117) Knowledge Distillation for Object Detection: from generic to remote sensing datasets (Hoàng-Ân Lê et al., 2023)

{{<citation>}}

Hoàng-Ân Lê, Minh-Tan Pham. (2023)  
**Knowledge Distillation for Object Detection: from generic to remote sensing datasets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Knowledge Distillation, Object Detection  
[Paper Link](http://arxiv.org/abs/2307.09264v1)  

---


**ABSTRACT**  
Knowledge distillation, a well-known model compression technique, is an active research area in both computer vision and remote sensing communities. In this paper, we evaluate in a remote sensing context various off-the-shelf object detection knowledge distillation methods which have been originally developed on generic computer vision datasets such as Pascal VOC. In particular, methods covering both logit mimicking and feature imitation approaches are applied for vehicle detection using the well-known benchmarks such as xView and VEDAI datasets. Extensive experiments are performed to compare the relative performance and interrelationships of the methods. Experimental results show high variations and confirm the importance of result aggregation and cross validation on remote sensing datasets.

{{</citation>}}


### (56/117) Augmenting CLIP with Improved Visio-Linguistic Reasoning (Samyadeep Basu et al., 2023)

{{<citation>}}

Samyadeep Basu, Maziar Sanjabi, Daniela Massiceti, Shell Xu Hu, Soheil Feizi. (2023)  
**Augmenting CLIP with Improved Visio-Linguistic Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2307.09233v1)  

---


**ABSTRACT**  
Image-text contrastive models such as CLIP are useful for a variety of downstream applications including zero-shot classification, image-text retrieval and transfer learning. However, these contrastively trained vision-language models often fail on compositional visio-linguistic tasks such as Winoground with performance equivalent to random chance. In our paper, we address this issue and propose a sample-efficient light-weight method called SDS-CLIP to improve the compositional visio-linguistic reasoning capabilities of CLIP. The core idea of our method is to use differentiable image parameterizations to fine-tune CLIP with a distillation objective from large text-to-image generative models such as Stable-Diffusion which are relatively good at visio-linguistic reasoning tasks. On the challenging Winoground compositional reasoning benchmark, our method improves the absolute visio-linguistic performance of different CLIP models by up to 7%, while on the ARO dataset, our method improves the visio-linguistic performance by upto 3%. As a byproduct of inducing visio-linguistic reasoning into CLIP, we also find that the zero-shot performance improves marginally on a variety of downstream datasets. Our method reinforces that carefully designed distillation objectives from generative models can be leveraged to extend existing contrastive image-text models with improved visio-linguistic reasoning capabilities.

{{</citation>}}


### (57/117) Pixel-wise Graph Attention Networks for Person Re-identification (Wenyu Zhang et al., 2023)

{{<citation>}}

Wenyu Zhang, Qing Ding, Jian Hu, Yi Ma, Mingzhe Lu. (2023)  
**Pixel-wise Graph Attention Networks for Person Re-identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Graph Attention Network  
[Paper Link](http://arxiv.org/abs/2307.09183v1)  

---


**ABSTRACT**  
Graph convolutional networks (GCN) is widely used to handle irregular data since it updates node features by using the structure information of graph. With the help of iterated GCN, high-order information can be obtained to further enhance the representation of nodes. However, how to apply GCN to structured data (such as pictures) has not been deeply studied. In this paper, we explore the application of graph attention networks (GAT) in image feature extraction. First of all, we propose a novel graph generation algorithm to convert images into graphs through matrix transformation. It is one magnitude faster than the algorithm based on K Nearest Neighbors (KNN). Then, GAT is used on the generated graph to update the node features. Thus, a more robust representation is obtained. These two steps are combined into a module called pixel-wise graph attention module (PGA). Since the graph obtained by our graph generation algorithm can still be transformed into a picture after processing, PGA can be well combined with CNN. Based on these two modules, we consulted the ResNet and design a pixel-wise graph attention network (PGANet). The PGANet is applied to the task of person re-identification in the datasets Market1501, DukeMTMC-reID and Occluded-DukeMTMC (outperforms state-of-the-art by 0.8\%, 1.1\% and 11\% respectively, in mAP scores). Experiment results show that it achieves the state-of-the-art performance. \href{https://github.com/wenyu1009/PGANet}{The code is available here}.

{{</citation>}}


### (58/117) Class-relation Knowledge Distillation for Novel Class Discovery (Peiyan Gu et al., 2023)

{{<citation>}}

Peiyan Gu, Chuyu Zhang, Ruijie Xu, Xuming He. (2023)  
**Class-relation Knowledge Distillation for Novel Class Discovery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2307.09158v1)  

---


**ABSTRACT**  
We tackle the problem of novel class discovery, which aims to learn novel classes without supervision based on labeled data from known classes. A key challenge lies in transferring the knowledge in the known-class data to the learning of novel classes. Previous methods mainly focus on building a shared representation space for knowledge transfer and often ignore modeling class relations. To address this, we introduce a class relation representation for the novel classes based on the predicted class distribution of a model trained on known classes. Empirically, we find that such class relation becomes less informative during typical discovery training. To prevent such information loss, we propose a novel knowledge distillation framework, which utilizes our class-relation representation to regularize the learning of novel classes. In addition, to enable a flexible knowledge distillation scheme for each data point in novel classes, we develop a learnable weighting function for the regularization, which adaptively promotes knowledge transfer based on the semantic similarity between the novel and known classes. To validate the effectiveness and generalization of our method, we conduct extensive experiments on multiple benchmarks, including CIFAR100, Stanford Cars, CUB, and FGVC-Aircraft datasets. Our results demonstrate that the proposed method outperforms the previous state-of-the-art methods by a significant margin on almost all benchmarks. Code is available at \href{https://github.com/kleinzcy/Cr-KD-NCD}{here}.

{{</citation>}}


### (59/117) MLF-DET: Multi-Level Fusion for Cross-Modal 3D Object Detection (Zewei Lin et al., 2023)

{{<citation>}}

Zewei Lin, Yanqing Shen, Sanping Zhou, Shitao Chen, Nanning Zheng. (2023)  
**MLF-DET: Multi-Level Fusion for Cross-Modal 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.09155v1)  

---


**ABSTRACT**  
In this paper, we propose a novel and effective Multi-Level Fusion network, named as MLF-DET, for high-performance cross-modal 3D object DETection, which integrates both the feature-level fusion and decision-level fusion to fully utilize the information in the image. For the feature-level fusion, we present the Multi-scale Voxel Image fusion (MVI) module, which densely aligns multi-scale voxel features with image features. For the decision-level fusion, we propose the lightweight Feature-cued Confidence Rectification (FCR) module which further exploits image semantics to rectify the confidence of detection candidates. Besides, we design an effective data augmentation strategy termed Occlusion-aware GT Sampling (OGS) to reserve more sampled objects in the training scenes, so as to reduce overfitting. Extensive experiments on the KITTI dataset demonstrate the effectiveness of our method. Notably, on the extremely competitive KITTI car 3D object detection benchmark, our method reaches 82.89% moderate AP and achieves state-of-the-art performance without bells and whistles.

{{</citation>}}


### (60/117) MVA2023 Small Object Detection Challenge for Spotting Birds: Dataset, Methods, and Results (Yuki Kondo et al., 2023)

{{<citation>}}

Yuki Kondo, Norimichi Ukita, Takayuki Yamaguchi, Hao-Yu Hou, Mu-Yi Shen, Chia-Chi Hsu, En-Ming Huang, Yu-Chen Huang, Yu-Cheng Xia, Chien-Yao Wang, Chun-Yi Lee, Da Huo, Marc A. Kastner, Tingwei Liu, Yasutomo Kawanishi, Takatsugu Hirayama, Takahiro Komamizu, Ichiro Ide, Yosuke Shinya, Xinyao Liu, Guang Liang, Syusuke Yasui. (2023)  
**MVA2023 Small Object Detection Challenge for Spotting Birds: Dataset, Methods, and Results**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.09143v1)  

---


**ABSTRACT**  
Small Object Detection (SOD) is an important machine vision topic because (i) a variety of real-world applications require object detection for distant objects and (ii) SOD is a challenging task due to the noisy, blurred, and less-informative image appearances of small objects. This paper proposes a new SOD dataset consisting of 39,070 images including 137,121 bird instances, which is called the Small Object Detection for Spotting Birds (SOD4SB) dataset. The detail of the challenge with the SOD4SB dataset is introduced in this paper. In total, 223 participants joined this challenge. This paper briefly introduces the award-winning methods. The dataset, the baseline code, and the website for evaluation on the public testset are publicly available.

{{</citation>}}


### (61/117) DropMix: Reducing Class Dependency in Mixed Sample Data Augmentation (Haeil Lee et al., 2023)

{{<citation>}}

Haeil Lee, Hansang Lee, Junmo Kim. (2023)  
**DropMix: Reducing Class Dependency in Mixed Sample Data Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Augmentation, ImageNet  
[Paper Link](http://arxiv.org/abs/2307.09136v1)  

---


**ABSTRACT**  
Mixed sample data augmentation (MSDA) is a widely used technique that has been found to improve performance in a variety of tasks. However, in this paper, we show that the effects of MSDA are class-dependent, with some classes seeing an improvement in performance while others experience a decline. To reduce class dependency, we propose the DropMix method, which excludes a specific percentage of data from the MSDA computation. By training on a combination of MSDA and non-MSDA data, the proposed method not only improves the performance of classes that were previously degraded by MSDA, but also increases overall average accuracy, as shown in experiments on two datasets (CIFAR-100 and ImageNet) using three MSDA methods (Mixup, CutMix and PuzzleMix).

{{</citation>}}


### (62/117) Light-Weight Vision Transformer with Parallel Local and Global Self-Attention (Nikolas Ebert et al., 2023)

{{<citation>}}

Nikolas Ebert, Laurenz Reichardt, Didier Stricker, Oliver Wasenmüller. (2023)  
**Light-Weight Vision Transformer with Parallel Local and Global Self-Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, ImageNet, Self-Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2307.09120v1)  

---


**ABSTRACT**  
While transformer architectures have dominated computer vision in recent years, these models cannot easily be deployed on hardware with limited resources for autonomous driving tasks that require real-time-performance. Their computational complexity and memory requirements limits their use, especially for applications with high-resolution inputs. In our work, we redesign the powerful state-of-the-art Vision Transformer PLG-ViT to a much more compact and efficient architecture that is suitable for such tasks. We identify computationally expensive blocks in the original PLG-ViT architecture and propose several redesigns aimed at reducing the number of parameters and floating-point operations. As a result of our redesign, we are able to reduce PLG-ViT in size by a factor of 5, with a moderate drop in performance. We propose two variants, optimized for the best trade-off between parameter count to runtime as well as parameter count to accuracy. With only 5 million parameters, we achieve 79.5$\%$ top-1 accuracy on the ImageNet-1K classification benchmark. Our networks demonstrate great performance on general vision benchmarks like COCO instance segmentation. In addition, we conduct a series of experiments, demonstrating the potential of our approach in solving various tasks specifically tailored to the challenges of autonomous driving and transportation.

{{</citation>}}


### (63/117) NU-MCC: Multiview Compressive Coding with Neighborhood Decoder and Repulsive UDF (Stefan Lionar et al., 2023)

{{<citation>}}

Stefan Lionar, Xiangyu Xu, Min Lin, Gim Hee Lee. (2023)  
**NU-MCC: Multiview Compressive Coding with Neighborhood Decoder and Repulsive UDF**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.09112v1)  

---


**ABSTRACT**  
Remarkable progress has been made in 3D reconstruction from single-view RGB-D inputs. MCC is the current state-of-the-art method in this field, which achieves unprecedented success by combining vision Transformers with large-scale training. However, we identified two key limitations of MCC: 1) The Transformer decoder is inefficient in handling large number of query points; 2) The 3D representation struggles to recover high-fidelity details. In this paper, we propose a new approach called NU-MCC that addresses these limitations. NU-MCC includes two key innovations: a Neighborhood decoder and a Repulsive Unsigned Distance Function (Repulsive UDF). First, our Neighborhood decoder introduces center points as an efficient proxy of input visual features, allowing each query point to only attend to a small neighborhood. This design not only results in much faster inference speed but also enables the exploitation of finer-scale visual features for improved recovery of 3D textures. Second, our Repulsive UDF is a novel alternative to the occupancy field used in MCC, significantly improving the quality of 3D object reconstruction. Compared to standard UDFs that suffer from holes in results, our proposed Repulsive UDF can achieve more complete surface reconstruction. Experimental results demonstrate that NU-MCC is able to learn a strong 3D representation, significantly advancing the state of the art in single-view 3D reconstruction. Particularly, it outperforms MCC by 9.7% in terms of the F1-score on the CO3D-v2 dataset with more than 5x faster running speed.

{{</citation>}}


### (64/117) PatchCT: Aligning Patch Set and Label Set with Conditional Transport for Multi-Label Image Classification (Miaoge Li et al., 2023)

{{<citation>}}

Miaoge Li, Dongsheng Wang, Xinyang Liu, Zequn Zeng, Ruiying Lu, Bo Chen, Mingyuan Zhou. (2023)  
**PatchCT: Aligning Patch Set and Label Set with Conditional Transport for Multi-Label Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2307.09066v1)  

---


**ABSTRACT**  
Multi-label image classification is a prediction task that aims to identify more than one label from a given image. This paper considers the semantic consistency of the latent space between the visual patch and linguistic label domains and introduces the conditional transport (CT) theory to bridge the acknowledged gap. While recent cross-modal attention-based studies have attempted to align such two representations and achieved impressive performance, they required carefully-designed alignment modules and extra complex operations in the attention computation. We find that by formulating the multi-label classification as a CT problem, we can exploit the interactions between the image and label efficiently by minimizing the bidirectional CT cost. Specifically, after feeding the images and textual labels into the modality-specific encoders, we view each image as a mixture of patch embeddings and a mixture of label embeddings, which capture the local region features and the class prototypes, respectively. CT is then employed to learn and align those two semantic sets by defining the forward and backward navigators. Importantly, the defined navigators in CT distance model the similarities between patches and labels, which provides an interpretable tool to visualize the learned prototypes. Extensive experiments on three public image benchmarks show that the proposed model consistently outperforms the previous methods. Our code is available at https://github.com/keepgoingjkg/PatchCT.

{{</citation>}}


### (65/117) Learning Adaptive Neighborhoods for Graph Neural Networks (Avishkar Saha et al., 2023)

{{<citation>}}

Avishkar Saha, Oscar Mendez, Chris Russell, Richard Bowden. (2023)  
**Learning Adaptive Neighborhoods for Graph Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.09065v1)  

---


**ABSTRACT**  
Graph convolutional networks (GCNs) enable end-to-end learning on graph structured data. However, many works assume a given graph structure. When the input graph is noisy or unavailable, one approach is to construct or learn a latent graph structure. These methods typically fix the choice of node degree for the entire graph, which is suboptimal. Instead, we propose a novel end-to-end differentiable graph generator which builds graph topologies where each node selects both its neighborhood and its size. Our module can be readily integrated into existing pipelines involving graph convolution operations, replacing the predetermined or existing adjacency matrix with one that is learned, and optimized, as part of the general objective. As such it is applicable to any GCN. We integrate our module into trajectory prediction, point cloud classification and node classification pipelines resulting in improved accuracy over other structure-learning methods across a wide range of datasets and GCN backbones.

{{</citation>}}


### (66/117) R-Cut: Enhancing Explainability in Vision Transformers with Relationship Weighted Out and Cut (Yingjie Niu et al., 2023)

{{<citation>}}

Yingjie Niu, Ming Ding, Maoning Ge, Robin Karlsson, Yuxiao Zhang, Kazuya Takeda. (2023)  
**R-Cut: Enhancing Explainability in Vision Transformers with Relationship Weighted Out and Cut**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: GPT, ImageNet, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.09050v1)  

---


**ABSTRACT**  
Transformer-based models have gained popularity in the field of natural language processing (NLP) and are extensively utilized in computer vision tasks and multi-modal models such as GPT4. This paper presents a novel method to enhance the explainability of Transformer-based image classification models. Our method aims to improve trust in classification results and empower users to gain a deeper understanding of the model for downstream tasks by providing visualizations of class-specific maps. We introduce two modules: the ``Relationship Weighted Out" and the ``Cut" modules. The ``Relationship Weighted Out" module focuses on extracting class-specific information from intermediate layers, enabling us to highlight relevant features. Additionally, the ``Cut" module performs fine-grained feature decomposition, taking into account factors such as position, texture, and color. By integrating these modules, we generate dense class-specific visual explainability maps. We validate our method with extensive qualitative and quantitative experiments on the ImageNet dataset. Furthermore, we conduct a large number of experiments on the LRN dataset, specifically designed for automatic driving danger alerts, to evaluate the explainability of our method in complex backgrounds. The results demonstrate a significant improvement over previous methods. Moreover, we conduct ablation experiments to validate the effectiveness of each module. Through these experiments, we are able to confirm the respective contributions of each module, thus solidifying the overall effectiveness of our proposed approach.

{{</citation>}}


### (67/117) Online Self-Supervised Thermal Water Segmentation for Aerial Vehicles (Connor Lee et al., 2023)

{{<citation>}}

Connor Lee, Jonathan Gustafsson Frennert, Lu Gan, Matthew Anderson, Soon-Jo Chung. (2023)  
**Online Self-Supervised Thermal Water Segmentation for Aerial Vehicles**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.09027v1)  

---


**ABSTRACT**  
We present a new method to adapt an RGB-trained water segmentation network to target-domain aerial thermal imagery using online self-supervision by leveraging texture and motion cues as supervisory signals. This new thermal capability enables current autonomous aerial robots operating in near-shore environments to perform tasks such as visual navigation, bathymetry, and flow tracking at night. Our method overcomes the problem of scarce and difficult-to-obtain near-shore thermal data that prevents the application of conventional supervised and unsupervised methods. In this work, we curate the first aerial thermal near-shore dataset, show that our approach outperforms fully-supervised segmentation models trained on limited target-domain thermal data, and demonstrate real-time capabilities onboard an Nvidia Jetson embedded computing platform. Code and datasets used in this work will be available at: https://github.com/connorlee77/uav-thermal-water-segmentation.

{{</citation>}}


### (68/117) Face-PAST: Facial Pose Awareness and Style Transfer Networks (Sunder Ali Khowaja et al., 2023)

{{<citation>}}

Sunder Ali Khowaja, Ghulam Mujtaba, Jiseok Yoon, Ik Hyun Lee. (2023)  
**Face-PAST: Facial Pose Awareness and Style Transfer Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2307.09020v1)  

---


**ABSTRACT**  
Facial style transfer has been quite popular among researchers due to the rise of emerging technologies such as eXtended Reality (XR), Metaverse, and Non-Fungible Tokens (NFTs). Furthermore, StyleGAN methods along with transfer-learning strategies have reduced the problem of limited data to some extent. However, most of the StyleGAN methods overfit the styles while adding artifacts to facial images. In this paper, we propose a facial pose awareness and style transfer (Face-PAST) network that preserves facial details and structures while generating high-quality stylized images. Dual StyleGAN inspires our work, but in contrast, our work uses a pre-trained style generation network in an external style pass with a residual modulation block instead of a transform coding block. Furthermore, we use the gated mapping unit and facial structure, identity, and segmentation losses to preserve the facial structure and details. This enables us to train the network with a very limited amount of data while generating high-quality stylized images. Our training process adapts curriculum learning strategy to perform efficient and flexible style mixing in the generative space. We perform extensive experiments to show the superiority of Face-PAST in comparison to existing state-of-the-art methods.

{{</citation>}}


### (69/117) Towards Authentic Face Restoration with Iterative Diffusion Models and Beyond (Yang Zhao et al., 2023)

{{<citation>}}

Yang Zhao, Tingbo Hou, Yu-Chuan Su, Xuhui Jia. Yandong Li, Matthias Grundmann. (2023)  
**Towards Authentic Face Restoration with Iterative Diffusion Models and Beyond**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.08996v1)  

---


**ABSTRACT**  
An authentic face restoration system is becoming increasingly demanding in many computer vision applications, e.g., image enhancement, video communication, and taking portrait. Most of the advanced face restoration models can recover high-quality faces from low-quality ones but usually fail to faithfully generate realistic and high-frequency details that are favored by users. To achieve authentic restoration, we propose $\textbf{IDM}$, an $\textbf{I}$teratively learned face restoration system based on denoising $\textbf{D}$iffusion $\textbf{M}$odels (DDMs). We define the criterion of an authentic face restoration system, and argue that denoising diffusion models are naturally endowed with this property from two aspects: intrinsic iterative refinement and extrinsic iterative enhancement. Intrinsic learning can preserve the content well and gradually refine the high-quality details, while extrinsic enhancement helps clean the data and improve the restoration task one step further. We demonstrate superior performance on blind face restoration tasks. Beyond restoration, we find the authentically cleaned data by the proposed restoration system is also helpful to image generation tasks in terms of training stabilization and sample quality. Without modifying the models, we achieve better quality than state-of-the-art on FFHQ and ImageNet generation using either GANs or diffusion models.

{{</citation>}}


### (70/117) Human Action Recognition in Still Images Using ConViT (Seyed Rohollah Hosseyni et al., 2023)

{{<citation>}}

Seyed Rohollah Hosseyni, Hasan Taheri, Sanaz Seyedin, Ali Ahmad Rahmani. (2023)  
**Human Action Recognition in Still Images Using ConViT**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.08994v1)  

---


**ABSTRACT**  
Understanding the relationship between different parts of the image plays a crucial role in many visual recognition tasks. Despite the fact that Convolutional Neural Networks (CNNs) have demonstrated impressive results in detecting single objects, they lack the capability to extract the relationship between various regions of an image, which is a crucial factor in human action recognition. To address this problem, this paper proposes a new module that functions like a convolutional layer using Vision Transformer (ViT). The proposed action recognition model comprises two components: the first part is a deep convolutional network that extracts high-level spatial features from the image, and the second component of the model utilizes a Vision Transformer that extracts the relationship between various regions of the image using the feature map generated by the CNN output. The proposed model has been evaluated on the Stanford40 and PASCAL VOC 2012 action datasets and has achieved 95.5% mAP and 91.5% mAP results, respectively, which are promising compared to other state-of-the-art methods.

{{</citation>}}


### (71/117) Generative Visual Question Answering (Ethan Shen et al., 2023)

{{<citation>}}

Ethan Shen, Scotty Singh, Bhavesh Kumar. (2023)  
**Generative Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.10405v1)  

---


**ABSTRACT**  
Multi-modal tasks involving vision and language in deep learning continue to rise in popularity and are leading to the development of newer models that can generalize beyond the extent of their training data. The current models lack temporal generalization which enables models to adapt to changes in future data. This paper discusses a viable approach to creating an advanced Visual Question Answering (VQA) model which can produce successful results on temporal generalization. We propose a new data set, GenVQA, utilizing images and captions from the VQAv2 and MS-COCO dataset to generate new images through stable diffusion. This augmented dataset is then used to test a combination of seven baseline and cutting edge VQA models. Performance evaluation focuses on questions mirroring the original VQAv2 dataset, with the answers having been adjusted to the new images. This paper's purpose is to investigate the robustness of several successful VQA models to assess their performance on future data distributions. Model architectures are analyzed to identify common stylistic choices that improve generalization under temporal distribution shifts. This research highlights the importance of creating a large-scale future shifted dataset. This data can enhance the robustness of VQA models, allowing their future peers to have improved ability to adapt to temporal distribution shifts.

{{</citation>}}


### (72/117) CSSL-RHA: Contrastive Self-Supervised Learning for Robust Handwriting Authentication (Jingyao Wang et al., 2023)

{{<citation>}}

Jingyao Wang, Luntian Mou, Changwen Zheng, Wen Gao. (2023)  
**CSSL-RHA: Contrastive Self-Supervised Learning for Robust Handwriting Authentication**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.11100v1)  

---


**ABSTRACT**  
Handwriting authentication is a valuable tool used in various fields, such as fraud prevention and cultural heritage protection. However, it remains a challenging task due to the complex features, severe damage, and lack of supervision. In this paper, we propose a novel Contrastive Self-Supervised Learning framework for Robust Handwriting Authentication (CSSL-RHA) to address these issues. It can dynamically learn complex yet important features and accurately predict writer identities. Specifically, to remove the negative effects of imperfections and redundancy, we design an information-theoretic filter for pre-processing and propose a novel adaptive matching scheme to represent images as patches of local regions dominated by more important features. Through online optimization at inference time, the most informative patch embeddings are identified as the "most important" elements. Furthermore, we employ contrastive self-supervised training with a momentum-based paradigm to learn more general statistical structures of handwritten data without supervision. We conduct extensive experiments on five benchmark datasets and our manually annotated dataset EN-HA, which demonstrate the superiority of our CSSL-RHA compared to baselines. Additionally, we show that our proposed model can still effectively achieve authentication even under abnormal circumstances, such as data falsification and corruption.

{{</citation>}}


## eess.IV (5)



### (73/117) Transformer-based Dual-domain Network for Few-view Dedicated Cardiac SPECT Image Reconstructions (Huidong Xie et al., 2023)

{{<citation>}}

Huidong Xie, Bo Zhou, Xiongchao Chen, Xueqi Guo, Stephanie Thorn, Yi-Hwa Liu, Ge Wang, Albert Sinusas, Chi Liu. (2023)  
**Transformer-based Dual-domain Network for Few-view Dedicated Cardiac SPECT Image Reconstructions**  

---
Primary Category: eess.IV  
Categories: cs-AI, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.09624v1)  

---


**ABSTRACT**  
Cardiovascular disease (CVD) is the leading cause of death worldwide, and myocardial perfusion imaging using SPECT has been widely used in the diagnosis of CVDs. The GE 530/570c dedicated cardiac SPECT scanners adopt a stationary geometry to simultaneously acquire 19 projections to increase sensitivity and achieve dynamic imaging. However, the limited amount of angular sampling negatively affects image quality. Deep learning methods can be implemented to produce higher-quality images from stationary data. This is essentially a few-view imaging problem. In this work, we propose a novel 3D transformer-based dual-domain network, called TIP-Net, for high-quality 3D cardiac SPECT image reconstructions. Our method aims to first reconstruct 3D cardiac SPECT images directly from projection data without the iterative reconstruction process by proposing a customized projection-to-image domain transformer. Then, given its reconstruction output and the original few-view reconstruction, we further refine the reconstruction using an image-domain reconstruction network. Validated by cardiac catheterization images, diagnostic interpretations from nuclear cardiologists, and defect size quantified by an FDA 510(k)-cleared clinical software, our method produced images with higher cardiac defect contrast on human studies compared with previous baseline methods, potentially enabling high-quality defect visualization using stationary few-view dedicated cardiac SPECT scanners.

{{</citation>}}


### (74/117) Smooth Attention for Deep Multiple Instance Learning: Application to CT Intracranial Hemorrhage Detection (Yunan Wu et al., 2023)

{{<citation>}}

Yunan Wu, Francisco M. Castro-Macías, Pablo Morales-Álvarez, Rafael Molina, Aggelos K. Katsaggelos. (2023)  
**Smooth Attention for Deep Multiple Instance Learning: Application to CT Intracranial Hemorrhage Detection**  

---
Primary Category: eess.IV  
Categories: cs-LG, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.09457v1)  

---


**ABSTRACT**  
Multiple Instance Learning (MIL) has been widely applied to medical imaging diagnosis, where bag labels are known and instance labels inside bags are unknown. Traditional MIL assumes that instances in each bag are independent samples from a given distribution. However, instances are often spatially or sequentially ordered, and one would expect similar diagnostic importance for neighboring instances. To address this, in this study, we propose a smooth attention deep MIL (SA-DMIL) model. Smoothness is achieved by the introduction of first and second order constraints on the latent function encoding the attention paid to each instance in a bag. The method is applied to the detection of intracranial hemorrhage (ICH) on head CT scans. The results show that this novel SA-DMIL: (a) achieves better performance than the non-smooth attention MIL at both scan (bag) and slice (instance) levels; (b) learns spatial dependencies between slices; and (c) outperforms current state-of-the-art MIL methods on the same ICH test set.

{{</citation>}}


### (75/117) Towards Automated Semantic Segmentation in Mammography Images (Cesar A. Sierra-Franco et al., 2023)

{{<citation>}}

Cesar A. Sierra-Franco, Jan Hurtado, Victor de A. Thomaz, Leonardo C. da Cruz, Santiago V. Silva, Alberto B. Raposo. (2023)  
**Towards Automated Semantic Segmentation in Mammography Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.10296v1)  

---


**ABSTRACT**  
Mammography images are widely used to detect non-palpable breast lesions or nodules, preventing cancer and providing the opportunity to plan interventions when necessary. The identification of some structures of interest is essential to make a diagnosis and evaluate image adequacy. Thus, computer-aided detection systems can be helpful in assisting medical interpretation by automatically segmenting these landmark structures. In this paper, we propose a deep learning-based framework for the segmentation of the nipple, the pectoral muscle, the fibroglandular tissue, and the fatty tissue on standard-view mammography images. We introduce a large private segmentation dataset and extensive experiments considering different deep-learning model architectures. Our experiments demonstrate accurate segmentation performance on variate and challenging cases, showing that this framework can be integrated into clinical practice.

{{</citation>}}


### (76/117) ECSIC: Epipolar Cross Attention for Stereo Image Compression (Matthias Wödlinger et al., 2023)

{{<citation>}}

Matthias Wödlinger, Jan Kotera, Manuel Keglevic, Jan Xu, Robert Sablatnig. (2023)  
**ECSIC: Epipolar Cross Attention for Stereo Image Compression**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.10284v1)  

---


**ABSTRACT**  
In this paper, we present ECSIC, a novel learned method for stereo image compression. Our proposed method compresses the left and right images in a joint manner by exploiting the mutual information between the images of the stereo image pair using a novel stereo cross attention (SCA) module and two stereo context modules. The SCA module performs cross-attention restricted to the corresponding epipolar lines of the two images and processes them in parallel. The stereo context modules improve the entropy estimation of the second encoded image by using the first image as a context. We conduct an extensive ablation study demonstrating the effectiveness of the proposed modules and a comprehensive quantitative and qualitative comparison with existing methods. ECSIC achieves state-of-the-art performance among stereo image compression models on the two popular stereo image datasets Cityscapes and InStereo2k while allowing for fast encoding and decoding, making it highly practical for real-time applications.

{{</citation>}}


### (77/117) Evaluate Fine-tuning Strategies for Fetal Head Ultrasound Image Segmentation with U-Net (Fangyijie Wang et al., 2023)

{{<citation>}}

Fangyijie Wang, Guénolé Silvestre, Kathleen M. Curran. (2023)  
**Evaluate Fine-tuning Strategies for Fetal Head Ultrasound Image Segmentation with U-Net**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.09067v1)  

---


**ABSTRACT**  
Fetal head segmentation is a crucial step in measuring the fetal head circumference (HC) during gestation, an important biometric in obstetrics for monitoring fetal growth. However, manual biometry generation is time-consuming and results in inconsistent accuracy. To address this issue, convolutional neural network (CNN) models have been utilized to improve the efficiency of medical biometry. But training a CNN network from scratch is a challenging task, we proposed a Transfer Learning (TL) method. Our approach involves fine-tuning (FT) a U-Net network with a lightweight MobileNet as the encoder to perform segmentation on a set of fetal head ultrasound (US) images with limited effort. This method addresses the challenges associated with training a CNN network from scratch. It suggests that our proposed FT strategy yields segmentation performance that is comparable when trained with a reduced number of parameters by 85.8%. And our proposed FT strategy outperforms other strategies with smaller trainable parameter sizes below 4.4 million. Thus, we contend that it can serve as a dependable FT approach for reducing the size of models in medical image analysis. Our key findings highlight the importance of the balance between model performance and size in developing Artificial Intelligence (AI) applications by TL methods. Code is available at https://github.com/13204942/FT_Methods_for_Fetal_Head_Segmentation.

{{</citation>}}


## cs.CL (16)



### (78/117) Analyzing sports commentary in order to automatically recognize events and extract insights (Yanis Miraoui, 2023)

{{<citation>}}

Yanis Miraoui. (2023)  
**Analyzing sports commentary in order to automatically recognize events and extract insights**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL, stat-ML  
Keywords: Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.10303v1)  

---


**ABSTRACT**  
In this paper, we carefully investigate how we can use multiple different Natural Language Processing techniques and methods in order to automatically recognize the main actions in sports events. We aim to extract insights by analyzing live sport commentaries from different sources and by classifying these major actions into different categories. We also study if sentiment analysis could help detect these main actions.

{{</citation>}}


### (79/117) Can Model Fusing Help Transformers in Long Document Classification? An Empirical Study (Damith Premasiri et al., 2023)

{{<citation>}}

Damith Premasiri, Tharindu Ranasinghe, Ruslan Mitkov. (2023)  
**Can Model Fusing Help Transformers in Long Document Classification? An Empirical Study**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NLP, Natural Language Processing, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.09532v1)  

---


**ABSTRACT**  
Text classification is an area of research which has been studied over the years in Natural Language Processing (NLP). Adapting NLP to multiple domains has introduced many new challenges for text classification and one of them is long document classification. While state-of-the-art transformer models provide excellent results in text classification, most of them have limitations in the maximum sequence length of the input sequence. The majority of the transformer models are limited to 512 tokens, and therefore, they struggle with long document classification problems. In this research, we explore on employing Model Fusing for long document classification while comparing the results with well-known BERT and Longformer architectures.

{{</citation>}}


### (80/117) ChatSpot: Bootstrapping Multimodal LLMs via Precise Referring Instruction Tuning (Liang Zhao et al., 2023)

{{<citation>}}

Liang Zhao, En Yu, Zheng Ge, Jinrong Yang, Haoran Wei, Hongyu Zhou, Jianjian Sun, Yuang Peng, Runpei Dong, Chunrui Han, Xiangyu Zhang. (2023)  
**ChatSpot: Bootstrapping Multimodal LLMs via Precise Referring Instruction Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2307.09474v1)  

---


**ABSTRACT**  
Human-AI interactivity is a critical aspect that reflects the usability of multimodal large language models (MLLMs). However, existing end-to-end MLLMs only allow users to interact with them through language instructions, leading to the limitation of the interactive accuracy and efficiency. In this study, we present precise referring instructions that utilize diverse reference representations such as points and boxes as referring prompts to refer to the special region. This enables MLLMs to focus on the region of interest and achieve finer-grained interaction. Based on precise referring instruction, we propose ChatSpot, a unified end-to-end multimodal large language model that supports diverse forms of interactivity including mouse clicks, drag-and-drop, and drawing boxes, which provides a more flexible and seamless interactive experience. We also construct a multi-grained vision-language instruction-following dataset based on existing datasets and GPT-4 generating. Furthermore, we design a series of evaluation tasks to assess the effectiveness of region recognition and interaction. Experimental results showcase ChatSpot's promising performance.

{{</citation>}}


### (81/117) Pseudo Outlier Exposure for Out-of-Distribution Detection using Pretrained Transformers (Jaeyoung Kim et al., 2023)

{{<citation>}}

Jaeyoung Kim, Kyuheon Jung, Dongbin Na, Sion Jang, Eunbin Park, Sungchul Choi. (2023)  
**Pseudo Outlier Exposure for Out-of-Distribution Detection using Pretrained Transformers**  

---
Primary Category: cs.CL  
Categories: 68T50, cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.09455v2)  

---


**ABSTRACT**  
For real-world language applications, detecting an out-of-distribution (OOD) sample is helpful to alert users or reject such unreliable samples. However, modern over-parameterized language models often produce overconfident predictions for both in-distribution (ID) and OOD samples. In particular, language models suffer from OOD samples with a similar semantic representation to ID samples since these OOD samples lie near the ID manifold. A rejection network can be trained with ID and diverse outlier samples to detect test OOD samples, but explicitly collecting auxiliary OOD datasets brings an additional burden for data collection. In this paper, we propose a simple but effective method called Pseudo Outlier Exposure (POE) that constructs a surrogate OOD dataset by sequentially masking tokens related to ID classes. The surrogate OOD sample introduced by POE shows a similar representation to ID data, which is most effective in training a rejection network. Our method does not require any external OOD data and can be easily implemented within off-the-shelf Transformers. A comprehensive comparison with state-of-the-art algorithms demonstrates POE's competitiveness on several text classification benchmarks.

{{</citation>}}


### (82/117) Multi-Modal Discussion Transformer: Integrating Text, Images and Graph Transformers to Detect Hate Speech on Social Media (Liam Hebert et al., 2023)

{{<citation>}}

Liam Hebert, Gaurav Sahu, Nanda Kishore Sreenivas, Lukasz Golab, Robin Cohen. (2023)  
**Multi-Modal Discussion Transformer: Integrating Text, Images and Graph Transformers to Detect Hate Speech on Social Media**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-MM, cs-SI, cs.CL  
Keywords: Social Media, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.09312v1)  

---


**ABSTRACT**  
We present the Multi-Modal Discussion Transformer (mDT), a novel multi-modal graph-based transformer model for detecting hate speech in online social networks. In contrast to traditional text-only methods, our approach to labelling a comment as hate speech centers around the holistic analysis of text and images. This is done by leveraging graph transformers to capture the contextual relationships in the entire discussion that surrounds a comment, with interwoven fusion layers to combine text and image embeddings instead of processing different modalities separately. We compare the performance of our model to baselines that only process text; we also conduct extensive ablation studies. We conclude with future work for multimodal solutions to deliver social value in online contexts, arguing that capturing a holistic view of a conversation greatly advances the effort to detect anti-social behavior.

{{</citation>}}


### (83/117) Mutual Reinforcement Effects in Japanese Sentence Classification and Named Entity Recognition Tasks (Chengguang Gan et al., 2023)

{{<citation>}}

Chengguang Gan, Qinghao Zhang, Tatsunori Mori. (2023)  
**Mutual Reinforcement Effects in Japanese Sentence Classification and Named Entity Recognition Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2307.10291v2)  

---


**ABSTRACT**  
Information extraction(IE) is a crucial subfield within natural language processing. However, for the traditionally segmented approach to sentence classification and Named Entity Recognition, the intricate interactions between these individual subtasks remain largely uninvestigated. In this study, we propose an integrative analysis, converging sentence classification with Named Entity Recognition, with the objective to unveil and comprehend the mutual reinforcement effect within these two information extraction subtasks. To achieve this, we introduce a Sentence Classification and Named Entity Recognition Multi-task (SCNM) approach that combines Sentence Classification (SC) and Named Entity Recognition (NER). We develop a Sentence-to-Label Generation (SLG) framework for SCNM and construct a Wikipedia dataset containing both SC and NER. Using a format converter, we unify input formats and employ a generative model to generate SC-labels, NER-labels, and associated text segments. We propose a Constraint Mechanism (CM) to improve generated format accuracy. Our results show SC accuracy increased by 1.13 points and NER by 1.06 points in SCNM compared to standalone tasks, with CM raising format accuracy from 63.61 to 100. The findings indicate mutual reinforcement effects between SC and NER, and integration enhances both tasks' performance. We additionally implemented the SLG framework on single SC task. It yielded superior accuracies compared to the baseline on two distinct Japanese SC datasets. Notably, in the experiment of few-shot learning, SLG framework shows much better performance than fine-tune method. These empirical findings contribute additional evidence to affirm the efficacy of the SLG framework.

{{</citation>}}


### (84/117) Improving Text Semantic Similarity Modeling through a 3D Siamese Network (Jianxiang Zang et al., 2023)

{{<citation>}}

Jianxiang Zang, Hui Liu. (2023)  
**Improving Text Semantic Similarity Modeling through a 3D Siamese Network**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Semantic Similarity, Transformer  
[Paper Link](http://arxiv.org/abs/2307.09274v1)  

---


**ABSTRACT**  
Siamese networks have gained popularity as a method for modeling text semantic similarity. Traditional methods rely on pooling operation to compress the semantic representations from Transformer blocks in encoding, resulting in two-dimensional semantic vectors and the loss of hierarchical semantic information from Transformer blocks. Moreover, this limited structure of semantic vectors is akin to a flattened landscape, which restricts the methods that can be applied in downstream modeling, as they can only navigate this flat terrain. To address this issue, we propose a novel 3D Siamese network for text semantic similarity modeling, which maps semantic information to a higher-dimensional space. The three-dimensional semantic tensors not only retains more precise spatial and feature domain information but also provides the necessary structural condition for comprehensive downstream modeling strategies to capture them. Leveraging this structural advantage, we introduce several modules to reinforce this 3D framework, focusing on three aspects: feature extraction, attention, and feature fusion. Our extensive experiments on four text semantic similarity benchmarks demonstrate the effectiveness and efficiency of our 3D Siamese Network.

{{</citation>}}


### (85/117) Linearized Relative Positional Encoding (Zhen Qin et al., 2023)

{{<citation>}}

Zhen Qin, Weixuan Sun, Kaiyue Lu, Hui Deng, Dongxu Li, Xiaodong Han, Yuchao Dai, Lingpeng Kong, Yiran Zhong. (2023)  
**Linearized Relative Positional Encoding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2307.09270v1)  

---


**ABSTRACT**  
Relative positional encoding is widely used in vanilla and linear transformers to represent positional information. However, existing encoding methods of a vanilla transformer are not always directly applicable to a linear transformer, because the latter requires a decomposition of the query and key representations into separate kernel functions. Nevertheless, principles for designing encoding methods suitable for linear transformers remain understudied. In this work, we put together a variety of existing linear relative positional encoding approaches under a canonical form and further propose a family of linear relative positional encoding algorithms via unitary transformation. Our formulation leads to a principled framework that can be used to develop new relative positional encoding methods that preserve linear space-time complexity. Equipped with different models, the proposed linearized relative positional encoding (LRPE) family derives effective encoding for various applications. Experiments show that compared with existing methods, LRPE achieves state-of-the-art performance in language modeling, text classification, and image classification. Meanwhile, it emphasizes a general paradigm for designing broadly more relative positional encoding methods that are applicable to linear transformers. The code is available at https://github.com/OpenNLPLab/Lrpe.

{{</citation>}}


### (86/117) Automated Ableism: An Exploration of Explicit Disability Biases in Sentiment and Toxicity Analysis Models (Pranav Narayanan Venkit et al., 2023)

{{<citation>}}

Pranav Narayanan Venkit, Mukund Srinath, Shomir Wilson. (2023)  
**Automated Ableism: An Exploration of Explicit Disability Biases in Sentiment and Toxicity Analysis Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: AI, BERT, Bias, Google, Twitter  
[Paper Link](http://arxiv.org/abs/2307.09209v1)  

---


**ABSTRACT**  
We analyze sentiment analysis and toxicity detection models to detect the presence of explicit bias against people with disability (PWD). We employ the bias identification framework of Perturbation Sensitivity Analysis to examine conversations related to PWD on social media platforms, specifically Twitter and Reddit, in order to gain insight into how disability bias is disseminated in real-world social settings. We then create the \textit{Bias Identification Test in Sentiment} (BITS) corpus to quantify explicit disability bias in any sentiment analysis and toxicity detection models. Our study utilizes BITS to uncover significant biases in four open AIaaS (AI as a Service) sentiment analysis tools, namely TextBlob, VADER, Google Cloud Natural Language API, DistilBERT and two toxicity detection models, namely two versions of Toxic-BERT. Our findings indicate that all of these models exhibit statistically significant explicit bias against PWD.

{{</citation>}}


### (87/117) Unveiling Gender Bias in Terms of Profession Across LLMs: Analyzing and Addressing Sociological Implications (Vishesh Thakur, 2023)

{{<citation>}}

Vishesh Thakur. (2023)  
**Unveiling Gender Bias in Terms of Profession Across LLMs: Analyzing and Addressing Sociological Implications**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Bias, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2307.09162v1)  

---


**ABSTRACT**  
Gender bias in artificial intelligence (AI) and natural language processing has garnered significant attention due to its potential impact on societal perceptions and biases. This research paper aims to analyze gender bias in Large Language Models (LLMs) with a focus on multiple comparisons between GPT-2 and GPT-3.5, some prominent language models, to better understand its implications. Through a comprehensive literature review, the study examines existing research on gender bias in AI language models and identifies gaps in the current knowledge. The methodology involves collecting and preprocessing data from GPT-2 and GPT-3.5, and employing in-depth quantitative analysis techniques to evaluate gender bias in the generated text. The findings shed light on gendered word associations, language usage, and biased narratives present in the outputs of these Large Language Models. The discussion explores the ethical implications of gender bias and its potential consequences on social perceptions and marginalized communities. Additionally, the paper presents strategies for reducing gender bias in LLMs, including algorithmic approaches and data augmentation techniques. The research highlights the importance of interdisciplinary collaborations and the role of sociological studies in mitigating gender bias in AI models. By addressing these issues, we can pave the way for more inclusive and unbiased AI systems that have a positive impact on society.

{{</citation>}}


### (88/117) Attention over pre-trained Sentence Embeddings for Long Document Classification (Amine Abdaoui et al., 2023)

{{<citation>}}

Amine Abdaoui, Sourav Dutta. (2023)  
**Attention over pre-trained Sentence Embeddings for Long Document Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Embedding, NLP, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2307.09084v1)  

---


**ABSTRACT**  
Despite being the current de-facto models in most NLP tasks, transformers are often limited to short sequences due to their quadratic attention complexity on the number of tokens. Several attempts to address this issue were studied, either by reducing the cost of the self-attention computation or by modeling smaller sequences and combining them through a recurrence mechanism or using a new transformer model. In this paper, we suggest to take advantage of pre-trained sentence transformers to start from semantically meaningful embeddings of the individual sentences, and then combine them through a small attention layer that scales linearly with the document length. We report the results obtained by this simple architecture on three standard document classification datasets. When compared with the current state-of-the-art models using standard fine-tuning, the studied method obtains competitive results (even if there is no clear best model in this configuration). We also showcase that the studied architecture obtains better results when freezing the underlying transformers. A configuration that is useful when we need to avoid complete fine-tuning (e.g. when the same frozen transformer is shared by different applications). Finally, two additional experiments are provided to further evaluate the relevancy of the studied architecture over simpler baselines.

{{</citation>}}


### (89/117) Towards a Neural Era in Dialogue Management for Collaboration: A Literature Survey (Amogh Mannekote, 2023)

{{<citation>}}

Amogh Mannekote. (2023)  
**Towards a Neural Era in Dialogue Management for Collaboration: A Literature Survey**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2307.09021v1)  

---


**ABSTRACT**  
Dialogue-based human-AI collaboration can revolutionize collaborative problem-solving, creative exploration, and social support. To realize this goal, the development of automated agents proficient in skills such as negotiating, following instructions, establishing common ground, and progressing shared tasks is essential. This survey begins by reviewing the evolution of dialogue management paradigms in collaborative dialogue systems, from traditional handcrafted and information-state based methods to AI planning-inspired approaches. It then shifts focus to contemporary data-driven dialogue management techniques, which seek to transfer deep learning successes from form-filling and open-domain settings to collaborative contexts. The paper proceeds to analyze a selected set of recent works that apply neural approaches to collaborative dialogue management, spotlighting prevailing trends in the field. This survey hopes to provide foundational background for future advancements in collaborative dialogue management, particularly as the dialogue systems community continues to embrace the potential of large language models.

{{</citation>}}


### (90/117) How is ChatGPT's behavior changing over time? (Lingjiao Chen et al., 2023)

{{<citation>}}

Lingjiao Chen, Matei Zaharia, James Zou. (2023)  
**How is ChatGPT's behavior changing over time?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2307.09009v1)  

---


**ABSTRACT**  
GPT-3.5 and GPT-4 are the two most widely used large language model (LLM) services. However, when and how these models are updated over time is opaque. Here, we evaluate the March 2023 and June 2023 versions of GPT-3.5 and GPT-4 on four diverse tasks: 1) solving math problems, 2) answering sensitive/dangerous questions, 3) generating code and 4) visual reasoning. We find that the performance and behavior of both GPT-3.5 and GPT-4 can vary greatly over time. For example, GPT-4 (March 2023) was very good at identifying prime numbers (accuracy 97.6%) but GPT-4 (June 2023) was very poor on these same questions (accuracy 2.4%). Interestingly GPT-3.5 (June 2023) was much better than GPT-3.5 (March 2023) in this task. GPT-4 was less willing to answer sensitive questions in June than in March, and both GPT-4 and GPT-3.5 had more formatting mistakes in code generation in June than in March. Overall, our findings shows that the behavior of the same LLM service can change substantially in a relatively short amount of time, highlighting the need for continuous monitoring of LLM quality.

{{</citation>}}


### (91/117) On the (In)Effectiveness of Large Language Models for Chinese Text Correction (Yinghui Li et al., 2023)

{{<citation>}}

Yinghui Li, Haojing Huang, Shirong Ma, Yong Jiang, Yangning Li, Feng Zhou, Hai-Tao Zheng, Qingyu Zhou. (2023)  
**On the (In)Effectiveness of Large Language Models for Chinese Text Correction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.09007v1)  

---


**ABSTRACT**  
Recently, the development and progress of Large Language Models (LLMs) have amazed the entire Artificial Intelligence community. As an outstanding representative of LLMs and the foundation model that set off this wave of research on LLMs, ChatGPT has attracted more and more researchers to study its capabilities and performance on various downstream Natural Language Processing (NLP) tasks. While marveling at ChatGPT's incredible performance on kinds of tasks, we notice that ChatGPT also has excellent multilingual processing capabilities, such as Chinese. To explore the Chinese processing ability of ChatGPT, we focus on Chinese Text Correction, a fundamental and challenging Chinese NLP task. Specifically, we evaluate ChatGPT on the Chinese Grammatical Error Correction (CGEC) and Chinese Spelling Check (CSC) tasks, which are two main Chinese Text Correction scenarios. From extensive analyses and comparisons with previous state-of-the-art fine-tuned models, we empirically find that the ChatGPT currently has both amazing performance and unsatisfactory behavior for Chinese Text Correction. We believe our findings will promote the landing and application of LLMs in the Chinese NLP community.

{{</citation>}}


### (92/117) Teach model to answer questions after comprehending the document (Ruiqing Sun et al., 2023)

{{<citation>}}

Ruiqing Sun, Ping Jian. (2023)  
**Teach model to answer questions after comprehending the document**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Reading Comprehension, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.08931v1)  

---


**ABSTRACT**  
Multi-choice Machine Reading Comprehension (MRC) is a challenging extension of Natural Language Processing (NLP) that requires the ability to comprehend the semantics and logical relationships between entities in a given text. The MRC task has traditionally been viewed as a process of answering questions based on the given text. This single-stage approach has often led the network to concentrate on generating the correct answer, potentially neglecting the comprehension of the text itself. As a result, many prevalent models have faced challenges in performing well on this task when dealing with longer texts. In this paper, we propose a two-stage knowledge distillation method that teaches the model to better comprehend the document by dividing the MRC task into two separate stages. Our experimental results show that the student model, when equipped with our method, achieves significant improvements, demonstrating the effectiveness of our method.

{{</citation>}}


### (93/117) Large Language Models Perform Diagnostic Reasoning (Cheng-Kuang Wu et al., 2023)

{{<citation>}}

Cheng-Kuang Wu, Wei-Lin Chen, Hsin-Hsi Chen. (2023)  
**Large Language Models Perform Diagnostic Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.08922v1)  

---


**ABSTRACT**  
We explore the extension of chain-of-thought (CoT) prompting to medical reasoning for the task of automatic diagnosis. Motivated by doctors' underlying reasoning process, we present Diagnostic-Reasoning CoT (DR-CoT). Empirical results demonstrate that by simply prompting large language models trained only on general text corpus with two DR-CoT exemplars, the diagnostic accuracy improves by 15% comparing to standard prompting. Moreover, the gap reaches a pronounced 18% in out-domain settings. Our findings suggest expert-knowledge reasoning in large language models can be elicited through proper promptings.

{{</citation>}}


## eess.AS (2)



### (94/117) SLMGAN: Exploiting Speech Language Model Representations for Unsupervised Zero-Shot Voice Conversion in GANs (Yinghao Aaron Li et al., 2023)

{{<citation>}}

Yinghao Aaron Li, Cong Han, Nima Mesgarani. (2023)  
**SLMGAN: Exploiting Speech Language Model Representations for Unsupervised Zero-Shot Voice Conversion in GANs**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-SD, eess-AS, eess.AS  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.09435v1)  

---


**ABSTRACT**  
In recent years, large-scale pre-trained speech language models (SLMs) have demonstrated remarkable advancements in various generative speech modeling applications, such as text-to-speech synthesis, voice conversion, and speech enhancement. These applications typically involve mapping text or speech inputs to pre-trained SLM representations, from which target speech is decoded. This paper introduces a new approach, SLMGAN, to leverage SLM representations for discriminative tasks within the generative adversarial network (GAN) framework, specifically for voice conversion. Building upon StarGANv2-VC, we add our novel SLM-based WavLM discriminators on top of the mel-based discriminators along with our newly designed SLM feature matching loss function, resulting in an unsupervised zero-shot voice conversion system that does not require text labels during training. Subjective evaluation results show that SLMGAN outperforms existing state-of-the-art zero-shot voice conversion models in terms of naturalness and achieves comparable similarity, highlighting the potential of SLM-based discriminators for related applications.

{{</citation>}}


### (95/117) Zero-shot Domain-sensitive Speech Recognition with Prompt-conditioning Fine-tuning (Feng-Ting Liao et al., 2023)

{{<citation>}}

Feng-Ting Liao, Yung-Chieh Chan, Yi-Chang Chen, Chan-Jan Hsu, Da-shan Shiu. (2023)  
**Zero-shot Domain-sensitive Speech Recognition with Prompt-conditioning Fine-tuning**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-LG, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.10274v1)  

---


**ABSTRACT**  
In this work, we propose a method to create domain-sensitive speech recognition models that utilize textual domain information by conditioning its generation on a given text prompt. This is accomplished by fine-tuning a pre-trained, end-to-end model (Whisper) to learn from demonstrations with prompt examples. We show that this ability can be generalized to different domains and even various prompt contexts, with our model gaining a Word Error Rate (WER) reduction of up to 33% on unseen datasets from various domains, such as medical conversation, air traffic control communication, and financial meetings. Considering the limited availability of audio-transcript pair data, we further extend our method to text-only fine-tuning to achieve domain sensitivity as well as domain adaptation. We demonstrate that our text-only fine-tuned model can also attend to various prompt contexts, with the model reaching the most WER reduction of 29% on the medical conversation dataset.

{{</citation>}}


## eess.SY (2)



### (96/117) Control of Small Spacecraft by Optimal Output Regulation: A Reinforcement Learning Approach (Joao Leonardo Silva Cotta et al., 2023)

{{<citation>}}

Joao Leonardo Silva Cotta, Omar Qasem, Paula do Vale Pereira, Hector Gutierrez. (2023)  
**Control of Small Spacecraft by Optimal Output Regulation: A Reinforcement Learning Approach**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.09428v1)  

---


**ABSTRACT**  
The growing number of noncooperative flying objects has prompted interest in sample-return and space debris removal missions. Current solutions are both costly and largely dependent on specific object identification and capture methods. In this paper, a low-cost modular approach for control of a swarm flight of small satellites in rendezvous and capture missions is proposed by solving the optimal output regulation problem. By integrating the theories of tracking control, adaptive optimal control, and output regulation, the optimal control policy is designed as a feedback-feedforward controller to guarantee the asymptotic tracking of a class of reference input generated by the leader. The estimated state vector of the space object of interest and communication within satellites is assumed to be available. The controller rejects the nonvanishing disturbances injected into the follower satellite while maintaining the closed-loop stability of the overall leader-follower system. The simulation results under the Basilisk-ROS2 framework environment for high-fidelity space applications with accurate spacecraft dynamics, are compared with those from a classical linear quadratic regulator controller, and the results reveal the efficiency and practicality of the proposed method.

{{</citation>}}


### (97/117) Continuous-Time Reinforcement Learning: New Design Algorithms with Theoretical Insights and Performance Guarantees (Brent A. Wallace et al., 2023)

{{<citation>}}

Brent A. Wallace, Jennie Si. (2023)  
**Continuous-Time Reinforcement Learning: New Design Algorithms with Theoretical Insights and Performance Guarantees**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08920v1)  

---


**ABSTRACT**  
Continuous-time nonlinear optimal control problems hold great promise in real-world applications. After decades of development, reinforcement learning (RL) has achieved some of the greatest successes as a general nonlinear control design method. However, a recent comprehensive analysis of state-of-the-art continuous-time RL (CT-RL) methods, namely, adaptive dynamic programming (ADP)-based CT-RL algorithms, reveals they face significant design challenges due to their complexity, numerical conditioning, and dimensional scaling issues. Despite advanced theoretical results, existing ADP CT-RL synthesis methods are inadequate in solving even small, academic problems. The goal of this work is thus to introduce a suite of new CT-RL algorithms for control of affine nonlinear systems. Our design approach relies on two important factors. First, our methods are applicable to physical systems that can be partitioned into smaller subproblems. This constructive consideration results in reduced dimensionality and greatly improved intuitiveness of design. Second, we introduce a new excitation framework to improve persistence of excitation (PE) and numerical conditioning performance via classical input/output insights. Such a design-centric approach is the first of its kind in the ADP CT-RL community. In this paper, we progressively introduce a suite of (decentralized) excitable integral reinforcement learning (EIRL) algorithms. We provide convergence and closed-loop stability guarantees, and we demonstrate these guarantees on a significant application problem of controlling an unstable, nonminimum phase hypersonic vehicle (HSV).

{{</citation>}}


## cs.SI (2)



### (98/117) Resilience of the reported global human-nature interaction network to pandemic conditions (Anne Cathrine Linder et al., 2023)

{{<citation>}}

Anne Cathrine Linder, David Lusseau. (2023)  
**Resilience of the reported global human-nature interaction network to pandemic conditions**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2307.09408v1)  

---


**ABSTRACT**  
Understanding human-nature interactions and the architecture of coupled human-nature systems is crucial for sustainable development. Cultural ecosystem services (CES), defined as intangible benefits derived from nature exposure, contribute to maintaining and improving human well-being. However, we have limited understanding of how well-being benefits emerge from CES co-production. In this study, for the first time, we estimated the global CES network from self-reported interactions between nature features and human activities underpinning CES co-production using social media. First, we used a bottom-up, approach to define the global repertoire of nature features and human activities used during CES co-production using 682,000 posts on Reddit. We then sampled Twitter to estimate the co-occurrence of these features and activities over the past five years, retrieving 41.7 millions tweets. These tweets were used to estimate the CES bipartite network, where each link was weighted by the number of times nature features and human activities co-occurred in tweets. We expected to observe large changes in the CES network topology in relation to the global mobility restrictions during the COVID-19 pandemic. This was not the case and the global CES network was generally resilient. However, a higher order singular value decomposition of the CES tensor revealed an impulse on the link between self care activities and urban greenspace. This could be due to an increased need for self care during the pandemic and urban greenspace enabling CES to be produced locally. Thus, providing resilience for maintaining well-being during the pandemic. Our user based analysis also indicated a shift towards local CES production during the beginning of the pandemic. Thus, supporting that CES was produced locally. These findings suggest an overall need for CES and access to features providing CES in local communities.

{{</citation>}}


### (99/117) Exploring acceptance of autonomous vehicle policies using KeyBERT and SNA: Targeting engineering students (Jinwoo Ha et al., 2023)

{{<citation>}}

Jinwoo Ha, Dongsoo Kim. (2023)  
**Exploring acceptance of autonomous vehicle policies using KeyBERT and SNA: Targeting engineering students**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-CL, cs-RO, cs-SI, cs.SI  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.09014v1)  

---


**ABSTRACT**  
This study aims to explore user acceptance of Autonomous Vehicle (AV) policies with improved text-mining methods. Recently, South Korean policymakers have viewed Autonomous Driving Car (ADC) and Autonomous Driving Robot (ADR) as next-generation means of transportation that will reduce the cost of transporting passengers and goods. They support the construction of V2I and V2V communication infrastructures for ADC and recognize that ADR is equivalent to pedestrians to promote its deployment into sidewalks. To fill the gap where end-user acceptance of these policies is not well considered, this study applied two text-mining methods to the comments of graduate students in the fields of Industrial, Mechanical, and Electronics-Electrical-Computer. One is the Co-occurrence Network Analysis (CNA) based on TF-IWF and Dice coefficient, and the other is the Contextual Semantic Network Analysis (C-SNA) based on both KeyBERT, which extracts keywords that contextually represent the comments, and double cosine similarity. The reason for comparing these approaches is to balance interest not only in the implications for the AV policies but also in the need to apply quality text mining to this research domain. Significantly, the limitation of frequency-based text mining, which does not reflect textual context, and the trade-off of adjusting thresholds in Semantic Network Analysis (SNA) were considered. As the results of comparing the two approaches, the C-SNA provided the information necessary to understand users' voices using fewer nodes and features than the CNA. The users who pre-emptively understood the AV policies based on their engineering literacy and the given texts revealed potential risks of the AV accident policies. This study adds suggestions to manage these risks to support the successful deployment of AVs on public roads.

{{</citation>}}


## cs.SE (1)



### (100/117) Is this Snippet Written by ChatGPT? An Empirical Study with a CodeBERT-Based Classifier (Phuong T. Nguyen et al., 2023)

{{<citation>}}

Phuong T. Nguyen, Juri Di Rocco, Claudio Di Sipio, Riccardo Rubei, Davide Di Ruscio, Massimiliano Di Penta. (2023)  
**Is this Snippet Written by ChatGPT? An Empirical Study with a CodeBERT-Based Classifier**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, BERT, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.09381v1)  

---


**ABSTRACT**  
Since its launch in November 2022, ChatGPT has gained popularity among users, especially programmers who use it as a tool to solve development problems. However, while offering a practical solution to programming problems, ChatGPT should be mainly used as a supporting tool (e.g., in software education) rather than as a replacement for the human being. Thus, detecting automatically generated source code by ChatGPT is necessary, and tools for identifying AI-generated content may need to be adapted to work effectively with source code. This paper presents an empirical study to investigate the feasibility of automated identification of AI-generated code snippets, and the factors that influence this ability. To this end, we propose a novel approach called GPTSniffer, which builds on top of CodeBERT to detect source code written by AI. The results show that GPTSniffer can accurately classify whether code is human-written or AI-generated, and outperforms two baselines, GPTZero and OpenAI Text Classifier. Also, the study shows how similar training data or a classification context with paired snippets helps to boost classification performances.

{{</citation>}}


## cs.NI (4)



### (101/117) Explanation-Guided Fair Federated Learning for Transparent 6G RAN Slicing (Swastika Roy et al., 2023)

{{<citation>}}

Swastika Roy, Hatim Chergui, Christos Verikoukis. (2023)  
**Explanation-Guided Fair Federated Learning for Transparent 6G RAN Slicing**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.09494v1)  

---


**ABSTRACT**  
Future zero-touch artificial intelligence (AI)-driven 6G network automation requires building trust in the AI black boxes via explainable artificial intelligence (XAI), where it is expected that AI faithfulness would be a quantifiable service-level agreement (SLA) metric along with telecommunications key performance indicators (KPIs). This entails exploiting the XAI outputs to generate transparent and unbiased deep neural networks (DNNs). Motivated by closed-loop (CL) automation and explanation-guided learning (EGL), we design an explanation-guided federated learning (EGFL) scheme to ensure trustworthy predictions by exploiting the model explanation emanating from XAI strategies during the training run time via Jensen-Shannon (JS) divergence. Specifically, we predict per-slice RAN dropped traffic probability to exemplify the proposed concept while respecting fairness goals formulated in terms of the recall metric which is included as a constraint in the optimization task. Finally, the comprehensiveness score is adopted to measure and validate the faithfulness of the explanations quantitatively. Simulation results show that the proposed EGFL-JS scheme has achieved more than $50\%$ increase in terms of comprehensiveness compared to different baselines from the literature, especially the variant EGFL-KL that is based on the Kullback-Leibler Divergence. It has also improved the recall score with more than $25\%$ relatively to unconstrained-EGFL.

{{</citation>}}


### (102/117) Enhancing Network Slicing Architectures with Machine Learning, Security, Sustainability and Experimental Networks Integration (Joberto S. B. Martins et al., 2023)

{{<citation>}}

Joberto S. B. Martins, Tereza C. Carvalho, Rodrigo Moreira, Cristiano Both, Adnei Donatti, João H. Corrêa, José A. Suruagy, Sand L. Corrêa, Antonio J. G. Abelem, Moisés R. N. Ribeiro, Jose-Marcos Nogueira, Luiz C. S. Magalhães, Juliano Wickboldt, Tiago Ferreto, Ricardo Mello, Rafael Pasquini, Marcos Schwarz, Leobino N. Sampaio, Daniel F. Macedo, José F. de Rezende, Kleber V. Cardoso, Flávio O. Silva. (2023)  
**Enhancing Network Slicing Architectures with Machine Learning, Security, Sustainability and Experimental Networks Integration**  

---
Primary Category: cs.NI  
Categories: I-2-1; C-2-1; C-2-3, cs-AI, cs-NI, cs.NI  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.09151v1)  

---


**ABSTRACT**  
Network Slicing (NS) is an essential technique extensively used in 5G networks computing strategies, mobile edge computing, mobile cloud computing, and verticals like the Internet of Vehicles and industrial IoT, among others. NS is foreseen as one of the leading enablers for 6G futuristic and highly demanding applications since it allows the optimization and customization of scarce and disputed resources among dynamic, demanding clients with highly distinct application requirements. Various standardization organizations, like 3GPP's proposal for new generation networks and state-of-the-art 5G/6G research projects, are proposing new NS architectures. However, new NS architectures have to deal with an extensive range of requirements that inherently result in having NS architecture proposals typically fulfilling the needs of specific sets of domains with commonalities. The Slicing Future Internet Infrastructures (SFI2) architecture proposal explores the gap resulting from the diversity of NS architectures target domains by proposing a new NS reference architecture with a defined focus on integrating experimental networks and enhancing the NS architecture with Machine Learning (ML) native optimizations, energy-efficient slicing, and slicing-tailored security functionalities. The SFI2 architectural main contribution includes the utilization of the slice-as-a-service paradigm for end-to-end orchestration of resources across multi-domains and multi-technology experimental networks. In addition, the SFI2 reference architecture instantiations will enhance the multi-domain and multi-technology integrated experimental network deployment with native ML optimization, energy-efficient aware slicing, and slicing-tailored security functionalities for the practical domain.

{{</citation>}}


### (103/117) AI-assisted Improved Service Provisioning for Low-latency XR over 5G NR (Moyukh Laha et al., 2023)

{{<citation>}}

Moyukh Laha, Dibbendu Roy, Sourav Dutta, Goutam Das. (2023)  
**AI-assisted Improved Service Provisioning for Low-latency XR over 5G NR**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-MM, cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08987v1)  

---


**ABSTRACT**  
Extended Reality (XR) is one of the most important 5G/6G media applications that will fundamentally transform human interactions. However, ensuring low latency, high data rate, and reliability to support XR services poses significant challenges. This letter presents a novel AI-assisted service provisioning scheme that leverages predicted frames for processing rather than relying solely on actual frames. This method virtually increases the network delay budget and consequently improves service provisioning, albeit at the expense of minor prediction errors. The proposed scheme is validated by extensive simulations demonstrating a multi-fold increase in supported XR users and also provides crucial network design insights.

{{</citation>}}


### (104/117) Deep Reinforcement Learning-based Content Migration for Edge Content Delivery Networks with Vehicular Nodes (Sepideh Malektaji et al., 2023)

{{<citation>}}

Sepideh Malektaji, Amin Ebrahimzadeh, Halima Elbiaze, Roch Glitho, Somayeh Kianpishe. (2023)  
**Deep Reinforcement Learning-based Content Migration for Edge Content Delivery Networks with Vehicular Nodes**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08905v1)  

---


**ABSTRACT**  
With the explosive demands for data, content delivery networks are facing ever-increasing challenges to meet end-users quality-of-experience requirements, especially in terms of delay. Content can be migrated from surrogate servers to local caches closer to end-users to address delay challenges. Unfortunately, these local caches have limited capacities, and when they are fully occupied, it may sometimes be necessary to remove their lower-priority content to accommodate higher-priority content. At other times, it may be necessary to return previously removed content to local caches. Downloading this content from surrogate servers is costly from the perspective of network usage, and potentially detrimental to the end-user QoE in terms of delay. In this paper, we consider an edge content delivery network with vehicular nodes and propose a content migration strategy in which local caches offload their contents to neighboring edge caches whenever feasible, instead of removing their contents when they are fully occupied. This process ensures that more contents remain in the vicinity of end-users. However, selecting which contents to migrate and to which neighboring cache to migrate is a complicated problem. This paper proposes a deep reinforcement learning approach to minimize the cost. Our simulation scenarios realized up to a 70% reduction of content access delay cost compared to conventional strategies with and without content migration.

{{</citation>}}


## cs.HC (2)



### (105/117) Identifying Explanation Needs of End-users: Applying and Extending the XAI Question Bank (Lars Sipos et al., 2023)

{{<citation>}}

Lars Sipos, Ulrike Schäfer, Katrin Glinka, Claudia Müller-Birn. (2023)  
**Identifying Explanation Needs of End-users: Applying and Extending the XAI Question Bank**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.09369v1)  

---


**ABSTRACT**  
Explanations in XAI are typically developed by AI experts and focus on algorithmic transparency and the inner workings of AI systems. Research has shown that such explanations do not meet the needs of users who do not have AI expertise. As a result, explanations are often ineffective in making system decisions interpretable and understandable. We aim to strengthen a socio-technical view of AI by following a Human-Centered Explainable Artificial Intelligence (HC-XAI) approach, which investigates the explanation needs of end-users (i.e., subject matter experts and lay users) in specific usage contexts. One of the most influential works in this area is the XAI Question Bank (XAIQB) by Liao et al. The authors propose a set of questions that end-users might ask when using an AI system, which in turn is intended to help developers and designers identify and address explanation needs. Although the XAIQB is widely referenced, there are few reports of its use in practice. In particular, it is unclear to what extent the XAIQB sufficiently captures the explanation needs of end-users and what potential problems exist in the practical application of the XAIQB. To explore these open questions, we used the XAIQB as the basis for analyzing 12 think-aloud software explorations with subject matter experts. We investigated the suitability of the XAIQB as a tool for identifying explanation needs in a specific usage context. Our analysis revealed a number of explanation needs that were missing from the question bank, but that emerged repeatedly as our study participants interacted with an AI system. We also found that some of the XAIQB questions were difficult to distinguish and required interpretation during use. Our contribution is an extension of the XAIQB with 11 new questions. In addition, we have expanded the descriptions of all new and existing questions to facilitate their use.

{{</citation>}}


### (106/117) PromptCrafter: Crafting Text-to-Image Prompt through Mixed-Initiative Dialogue with LLM (Seungho Baek et al., 2023)

{{<citation>}}

Seungho Baek, Hyerin Im, Jiseung Ryu, Juhyeong Park, Takyeon Lee. (2023)  
**PromptCrafter: Crafting Text-to-Image Prompt through Mixed-Initiative Dialogue with LLM**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2307.08985v1)  

---


**ABSTRACT**  
Text-to-image generation model is able to generate images across a diverse range of subjects and styles based on a single prompt. Recent works have proposed a variety of interaction methods that help users understand the capabilities of models and utilize them. However, how to support users to efficiently explore the model's capability and to create effective prompts are still open-ended research questions. In this paper, we present PromptCrafter, a novel mixed-initiative system that allows step-by-step crafting of text-to-image prompt. Through the iterative process, users can efficiently explore the model's capability, and clarify their intent. PromptCrafter also supports users to refine prompts by answering various responses to clarifying questions generated by a Large Language Model. Lastly, users can revert to a desired step by reviewing the work history. In this workshop paper, we discuss the design process of PromptCrafter and our plans for follow-up studies.

{{</citation>}}


## cs.ET (1)



### (107/117) Using the IBM Analog In-Memory Hardware Acceleration Kit for Neural Network Training and Inference (Manuel Le Gallo et al., 2023)

{{<citation>}}

Manuel Le Gallo, Corey Lammie, Julian Buechel, Fabio Carta, Omobayode Fagbohungbe, Charles Mackin, Hsinyu Tsai, Vijay Narayanan, Abu Sebastian, Kaoutar El Maghraoui, Malte J. Rasch. (2023)  
**Using the IBM Analog In-Memory Hardware Acceleration Kit for Neural Network Training and Inference**  

---
Primary Category: cs.ET  
Categories: cs-ET, cs-LG, cs.ET  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.09357v1)  

---


**ABSTRACT**  
Analog In-Memory Computing (AIMC) is a promising approach to reduce the latency and energy consumption of Deep Neural Network (DNN) inference and training. However, the noisy and non-linear device characteristics, and the non-ideal peripheral circuitry in AIMC chips, require adapting DNNs to be deployed on such hardware to achieve equivalent accuracy to digital computing. In this tutorial, we provide a deep dive into how such adaptations can be achieved and evaluated using the recently released IBM Analog Hardware Acceleration Kit (AIHWKit), freely available at https://github.com/IBM/aihwkit. The AIHWKit is a Python library that simulates inference and training of DNNs using AIMC. We present an in-depth description of the AIHWKit design, functionality, and best practices to properly perform inference and training. We also present an overview of the Analog AI Cloud Composer, that provides the benefits of using the AIHWKit simulation platform in a fully managed cloud setting. Finally, we show examples on how users can expand and customize AIHWKit for their own needs. This tutorial is accompanied by comprehensive Jupyter Notebook code examples that can be run using AIHWKit, which can be downloaded from https://github.com/IBM/aihwkit/tree/master/notebooks/tutorial.

{{</citation>}}


## cs.CY (1)



### (108/117) The Language Labyrinth: Constructive Critique on the Terminology Used in the AI Discourse (Rainer Rehak, 2023)

{{<citation>}}

Rainer Rehak. (2023)  
**The Language Labyrinth: Constructive Critique on the Terminology Used in the AI Discourse**  

---
Primary Category: cs.CY  
Categories: K-4-0; K-4-1; K-4-2; I-2-0; I-2-1; K-4-3, cs-AI, cs-CL, cs-CY, cs-NE, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.10292v1)  

---


**ABSTRACT**  
In the interdisciplinary field of artificial intelligence (AI) the problem of clear terminology is especially momentous. This paper claims, that AI debates are still characterised by a lack of critical distance to metaphors like 'training', 'learning' or 'deciding'. As consequence, reflections regarding responsibility or potential use-cases are greatly distorted. Yet, if relevant decision-makers are convinced that AI can develop an 'understanding' or properly 'interpret' issues, its regular use for sensitive tasks like deciding about social benefits or judging court cases looms. The chapter argues its claim by analysing central notions of the AI debate and tries to contribute by proposing more fitting terminology and hereby enabling more fruitful debates. It is a conceptual work at the intersection of critical computer science and philosophy of language.

{{</citation>}}


## cs.DC (1)



### (109/117) Cloud-native RStudio on Kubernetes for Hopsworks (Gibson Chikafa et al., 2023)

{{<citation>}}

Gibson Chikafa, Sina Sheikholeslami, Salman Niazi, Jim Dowling, Vladimir Vlassov. (2023)  
**Cloud-native RStudio on Kubernetes for Hopsworks**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs-SE, cs.DC  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2307.09132v1)  

---


**ABSTRACT**  
In order to fully benefit from cloud computing, services are designed following the "multi-tenant" architectural model, which is aimed at maximizing resource sharing among users. However, multi-tenancy introduces challenges of security, performance isolation, scaling, and customization. RStudio server is an open-source Integrated Development Environment (IDE) accessible over a web browser for the R programming language. We present the design and implementation of a multi-user distributed system on Hopsworks, a data-intensive AI platform, following the multi-tenant model that provides RStudio as Software as a Service (SaaS). We use the most popular cloud-native technologies: Docker and Kubernetes, to solve the problems of performance isolation, security, and scaling that are present in a multi-tenant environment. We further enable secure data sharing in RStudio server instances to provide data privacy and allow collaboration among RStudio users. We integrate our system with Apache Spark, which can scale and handle Big Data processing workloads. Also, we provide a UI where users can provide custom configurations and have full control of their own RStudio server instances. Our system was tested on a Google Cloud Platform cluster with four worker nodes, each with 30GB of RAM allocated to them. The tests on this cluster showed that 44 RStudio servers, each with 2GB of RAM, can be run concurrently. Our system can scale out to potentially support hundreds of concurrently running RStudio servers by adding more resources (CPUs and RAM) to the cluster or system.

{{</citation>}}


## cs.CR (4)



### (110/117) Mitigating Intersection Attacks in Anonymous Microblogging (Sarah Abdelwahab Gaballah et al., 2023)

{{<citation>}}

Sarah Abdelwahab Gaballah, Thanh Hoang Long Nguyen, Lamya Abdullah, Ephraim Zimmer, Max Mühlhäuser. (2023)  
**Mitigating Intersection Attacks in Anonymous Microblogging**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2307.09069v1)  

---


**ABSTRACT**  
Anonymous microblogging systems are known to be vulnerable to intersection attacks due to network churn. An adversary that monitors all communications can leverage the churn to learn who is publishing what with increasing confidence over time. In this paper, we propose a protocol for mitigating intersection attacks in anonymous microblogging systems by grouping users into anonymity sets based on similarities in their publishing behavior. The protocol provides a configurable communication schedule for users in each set to manage the inevitable trade-off between latency and bandwidth overhead. In our evaluation, we use real-world datasets from two popular microblogging platforms, Twitter and Reddit, to simulate user publishing behavior. The results demonstrate that the protocol can protect users against intersection attacks at low bandwidth overhead when the users adhere to communication schedules. In addition, the protocol can sustain a slow degradation in the size of the anonymity set over time under various churn rates.

{{</citation>}}


### (111/117) CBSeq: A Channel-level Behavior Sequence For Encrypted Malware Traffic Detection (Susu Cui et al., 2023)

{{<citation>}}

Susu Cui, Cong Dong, Meng Shen, Yuling Liu, Bo Jiang, Zhigang Lu. (2023)  
**CBSeq: A Channel-level Behavior Sequence For Encrypted Malware Traffic Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.09002v1)  

---


**ABSTRACT**  
Machine learning and neural networks have become increasingly popular solutions for encrypted malware traffic detection. They mine and learn complex traffic patterns, enabling detection by fitting boundaries between malware traffic and benign traffic. Compared with signature-based methods, they have higher scalability and flexibility. However, affected by the frequent variants and updates of malware, current methods suffer from a high false positive rate and do not work well for unknown malware traffic detection. It remains a critical task to achieve effective malware traffic detection. In this paper, we introduce CBSeq to address the above problems. CBSeq is a method that constructs a stable traffic representation, behavior sequence, to characterize attacking intent and achieve malware traffic detection. We novelly propose the channels with similar behavior as the detection object and extract side-channel content to construct behavior sequence. Unlike benign activities, the behavior sequences of malware and its variant's traffic exhibit solid internal correlations. Moreover, we design the MSFormer, a powerful Transformer-based multi-sequence fusion classifier. It captures the internal similarity of behavior sequence, thereby distinguishing malware traffic from benign traffic. Our evaluations demonstrate that CBSeq performs effectively in various known malware traffic detection and exhibits superior performance in unknown malware traffic detection, outperforming state-of-the-art methods.

{{</citation>}}


### (112/117) Experimental Security Analysis of DNN-based Adaptive Cruise Control under Context-Aware Perception Attacks (Xugui Zhou et al., 2023)

{{<citation>}}

Xugui Zhou, Anqi Chen, Maxfield Kouzel, Haotian Ren, Morgan McCarty, Cristina Nita-Rotaru, Homa Alemzadeh. (2023)  
**Experimental Security Analysis of DNN-based Adaptive Cruise Control under Context-Aware Perception Attacks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs-LG, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.08939v1)  

---


**ABSTRACT**  
Adaptive Cruise Control (ACC) is a widely used driver assistance feature for maintaining desired speed and safe distance to the leading vehicles. This paper evaluates the security of the deep neural network (DNN) based ACC systems under stealthy perception attacks that strategically inject perturbations into camera data to cause forward collisions. We present a combined knowledge-and-data-driven approach to design a context-aware strategy for the selection of the most critical times for triggering the attacks and a novel optimization-based method for the adaptive generation of image perturbations at run-time. We evaluate the effectiveness of the proposed attack using an actual driving dataset and a realistic simulation platform with the control software from a production ACC system and a physical-world driving simulator while considering interventions by the driver and safety features such as Automatic Emergency Braking (AEB) and Forward Collision Warning (FCW). Experimental results show that the proposed attack achieves 142.9x higher success rate in causing accidents than random attacks and is mitigated 89.6% less by the safety features while being stealthy and robust to real-world factors and dynamic changes in the environment. This study provides insights into the role of human operators and basic safety interventions in preventing attacks.

{{</citation>}}


### (113/117) A Note on the Security of ITS: Car Crash Analysis in Cruise Control Scenarios (Mohammad Sayad Haghighi, 2023)

{{<citation>}}

Mohammad Sayad Haghighi. (2023)  
**A Note on the Security of ITS: Car Crash Analysis in Cruise Control Scenarios**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.08899v1)  

---


**ABSTRACT**  
Security of Intelligent Transportation Systems (ITS) heavily depends on the security of the underlying components that create such a smart ecosystem. Adaptive Cruise Control (ACC) is embedded into most modern vehicles. In this report, we study the situations that the two vehicles involved in a cruise control scenario create. More precisely, after breaking down the phases the two vehicle go through (especially the ego one), we show how a simple formula can be used to predict collisions in hard brake cruise control scenarios.

{{</citation>}}


## math.OC (1)



### (114/117) Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces (Martin Ryner et al., 2023)

{{<citation>}}

Martin Ryner, Jan Kronqvist, Johan Karlsson. (2023)  
**Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces**  

---
Primary Category: math.OC  
Categories: 90C26, cs-LG, math-OC, math.OC, stat-ML  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2307.09057v1)  

---


**ABSTRACT**  
This paper presents a framework for computing the Gromov-Wasserstein problem between two sets of points in low dimensional spaces, where the discrepancy is the squared Euclidean norm. The Gromov-Wasserstein problem is a generalization of the optimal transport problem that finds the assignment between two sets preserving pairwise distances as much as possible. This can be used to quantify the similarity between two formations or shapes, a common problem in AI and machine learning. The problem can be formulated as a Quadratic Assignment Problem (QAP), which is in general computationally intractable even for small problems. Our framework addresses this challenge by reformulating the QAP as an optimization problem with a low-dimensional domain, leveraging the fact that the problem can be expressed as a concave quadratic optimization problem with low rank. The method scales well with the number of points, and it can be used to find the global solution for large-scale problems with thousands of points. We compare the computational complexity of our approach with state-of-the-art methods on synthetic problems and apply it to a near-symmetrical problem which is of particular interest in computational biology.

{{</citation>}}


## quant-ph (1)



### (115/117) qecGPT: decoding Quantum Error-correcting Codes with Generative Pre-trained Transformers (Hanyan Cao et al., 2023)

{{<citation>}}

Hanyan Cao, Feng Pan, Yijia Wang, Pan Zhang. (2023)  
**qecGPT: decoding Quantum Error-correcting Codes with Generative Pre-trained Transformers**  

---
Primary Category: quant-ph  
Categories: cond-mat-stat-mech, cs-LG, quant-ph, quant-ph, stat-ML  
Keywords: GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.09025v1)  

---


**ABSTRACT**  
We propose a general framework for decoding quantum error-correcting codes with generative modeling. The model utilizes autoregressive neural networks, specifically Transformers, to learn the joint probability of logical operators and syndromes. This training is in an unsupervised way, without the need for labeled training data, and is thus referred to as pre-training. After the pre-training, the model can efficiently compute the likelihood of logical operators for any given syndrome, using maximum likelihood decoding. It can directly generate the most-likely logical operators with computational complexity $\mathcal O(2k)$ in the number of logical qubits $k$, which is significantly better than the conventional maximum likelihood decoding algorithms that require $\mathcal O(4^k)$ computation. Based on the pre-trained model, we further propose refinement to achieve more accurately the likelihood of logical operators for a given syndrome by directly sampling the stabilizer operators. We perform numerical experiments on stabilizer codes with small code distances, using both depolarizing error models and error models with correlated noise. The results show that our approach provides significantly better decoding accuracy than the minimum weight perfect matching and belief-propagation-based algorithms. Our framework is general and can be applied to any error model and quantum codes with different topologies such as surface codes and quantum LDPC codes. Furthermore, it leverages the parallelization capabilities of GPUs, enabling simultaneous decoding of a large number of syndromes. Our approach sheds light on the efficient and accurate decoding of quantum error-correcting codes using generative artificial intelligence and modern computational power.

{{</citation>}}


## q-bio.QM (1)



### (116/117) Multimodal LLMs for health grounded in individual-specific data (Anastasiya Belyaeva et al., 2023)

{{<citation>}}

Anastasiya Belyaeva, Justin Cosentino, Farhad Hormozdiari, Krish Eswaran, Shravya Shetty, Greg Corrado, Andrew Carroll, Cory Y. McLean, Nicholas A. Furlotte. (2023)  
**Multimodal LLMs for health grounded in individual-specific data**  

---
Primary Category: q-bio.QM  
Categories: cs-LG, q-bio-QM, q-bio.QM  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.09018v2)  

---


**ABSTRACT**  
Foundation large language models (LLMs) have shown an impressive ability to solve tasks across a wide range of fields including health. To effectively solve personalized health tasks, LLMs need the ability to ingest a diversity of data modalities that are relevant to an individual's health status. In this paper, we take a step towards creating multimodal LLMs for health that are grounded in individual-specific data by developing a framework (HeLM: Health Large Language Model for Multimodal Understanding) that enables LLMs to use high-dimensional clinical modalities to estimate underlying disease risk. HeLM encodes complex data modalities by learning an encoder that maps them into the LLM's token embedding space and for simple modalities like tabular data by serializing the data into text. Using data from the UK Biobank, we show that HeLM can effectively use demographic and clinical features in addition to high-dimensional time-series data to estimate disease risk. For example, HeLM achieves an AUROC of 0.75 for asthma prediction when combining tabular and spirogram data modalities compared with 0.49 when only using tabular data. Overall, we find that HeLM outperforms or performs at parity with classical machine learning approaches across a selection of eight binary traits. Furthermore, we investigate the downstream uses of this model such as its generalizability to out-of-distribution traits and its ability to power conversations around individual health and wellness.

{{</citation>}}


## cs.DB (1)



### (117/117) Data sharing and ontology use among agricultural genetics, genomics, and breeding databases and resources of the AgBioData Consortium (Jennifer L. Clarke et al., 2023)

{{<citation>}}

Jennifer L. Clarke, Laurel D. Cooper, Monica F. Poelchau, Tanya Z. Berardini, Justin Elser, Andrew D. Farmer, Stephen Ficklin, Sunita Kumari, Marie-Angélique Laporte, Rex T. Nelson, Rie Sadohara, Peter Selby, Anne E. Thessen, Brandon Whitehead, Taner Z. Sen. (2023)  
**Data sharing and ontology use among agricultural genetics, genomics, and breeding databases and resources of the AgBioData Consortium**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08958v1)  

---


**ABSTRACT**  
Over the last several decades, there has been rapid growth in the number and scope of agricultural genetics, genomics and breeding (GGB) databases and resources. The AgBioData Consortium (https://www.agbiodata.org/) currently represents 44 databases and resources covering model or crop plant and animal GGB data, ontologies, pathways, genetic variation and breeding platforms (referred to as 'databases' throughout). One of the goals of the Consortium is to facilitate FAIR (Findable, Accessible, Interoperable, and Reusable) data management and the integration of datasets which requires data sharing, along with structured vocabularies and/or ontologies. Two AgBioData working groups, focused on Data Sharing and Ontologies, conducted a survey to assess the status and future needs of the members in those areas. A total of 33 researchers responded to the survey, representing 37 databases. Results suggest that data sharing practices by AgBioData databases are in a healthy state, but it is not clear whether this is true for all metadata and data types across all databases; and that ontology use has not substantially changed since a similar survey was conducted in 2017. We recommend 1) providing training for database personnel in specific data sharing techniques, as well as in ontology use; 2) further study on what metadata is shared, and how well it is shared among databases; 3) promoting an understanding of data sharing and ontologies in the stakeholder community; 4) improving data sharing and ontologies for specific phenotypic data types and formats; and 5) lowering specific barriers to data sharing and ontology use, by identifying sustainability solutions, and the identification, promotion, or development of data standards. Combined, these improvements are likely to help AgBioData databases increase development efforts towards improved ontology use, and data sharing via programmatic means.

{{</citation>}}
