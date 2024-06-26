---
draft: false
title: "arXiv @ 2023.08.06"
date: 2023-08-06
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.06"
    identifier: arxiv_20230806
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.AR (1)](#csar-1)
- [cs.CL (16)](#cscl-16)
- [cs.RO (5)](#csro-5)
- [cs.LG (15)](#cslg-15)
- [cs.CV (13)](#cscv-13)
- [cs.CR (5)](#cscr-5)
- [cs.AI (2)](#csai-2)
- [stat.ML (2)](#statml-2)
- [cs.CY (3)](#cscy-3)
- [eess.IV (3)](#eessiv-3)
- [cs.IR (2)](#csir-2)
- [cs.DC (1)](#csdc-1)
- [eess.SY (2)](#eesssy-2)
- [quant-ph (1)](#quant-ph-1)
- [cs.SE (1)](#csse-1)
- [math.NA (1)](#mathna-1)
- [cs.SD (3)](#cssd-3)
- [cs.NI (2)](#csni-2)
- [cs.IT (3)](#csit-3)
- [cs.HC (1)](#cshc-1)

## cs.AR (1)



### (1/82) Exploiting On-chip Heterogeneity of Versal Architecture for GNN Inference Acceleration (Paul Chen et al., 2023)

{{<citation>}}

Paul Chen, Pavan Manjunath, Sasindu Wijeratne, Bingyi Zhang, Viktor Prasanna. (2023)  
**Exploiting On-chip Heterogeneity of Versal Architecture for GNN Inference Acceleration**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-DC, cs-LG, cs.AR  
Keywords: AI, GNN, Graph Convolutional Network, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.02749v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have revolutionized many Machine Learning (ML) applications, such as social network analysis, bioinformatics, etc. GNN inference can be accelerated by exploiting data sparsity in the input graph, vertex features, and intermediate data in GNN computations. For dynamic sparsity exploitation, we leverage the heterogeneous computing capabilities of AMD Versal ACAP architecture to accelerate GNN inference. We develop a custom hardware module that executes the sparse primitives of the computation kernel on the Programmable Logic (PL) and efficiently computes the dense primitives using the AI Engine (AIE). To exploit data sparsity during inference, we devise a runtime kernel mapping strategy that dynamically assigns computation tasks to the PL and AIE based on data sparsity. Our implementation on the VCK5000 ACAP platform leads to superior performance compared with the state-of-the-art implementations on CPU, GPU, ACAP, and other custom GNN accelerators. Compared with these implementations, we achieve significant average runtime speedup across various models and datasets of 162.42x, 17.01x, 9.90x, and 27.23x, respectively. Furthermore, for Graph Convolutional Network (GCN) inference, our approach leads to a speedup of 3.9-96.7x compared to designs using PL only on the same ACAP device.

{{</citation>}}


## cs.CL (16)



### (2/82) Meta-Tsallis-Entropy Minimization: A New Self-Training Approach for Domain Adaptation on Text Classification (Menglong Lu et al., 2023)

{{<citation>}}

Menglong Lu, Zhen Huang, Zhiliang Tian, Yunxiang Zhao, Xuanyu Fei, Dongsheng Li. (2023)  
**Meta-Tsallis-Entropy Minimization: A New Self-Training Approach for Domain Adaptation on Text Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, Text Classification  
[Paper Link](http://arxiv.org/abs/2308.02746v1)  

---


**ABSTRACT**  
Text classification is a fundamental task for natural language processing, and adapting text classification models across domains has broad applications. Self-training generates pseudo-examples from the model's predictions and iteratively trains on the pseudo-examples, i.e., minimizes the loss on the source domain and the Gibbs entropy on the target domain. However, Gibbs entropy is sensitive to prediction errors, and thus, self-training tends to fail when the domain shift is large. In this paper, we propose Meta-Tsallis Entropy minimization (MTEM), which applies a meta-learning algorithm to optimize the instance adaptive Tsallis entropy on the target domain. To reduce the computation cost of MTEM, we propose an approximation technique to approximate the Second-order derivation involved in the meta-learning. To efficiently generate pseudo labels, we propose an annealing sampling mechanism for exploring the model's prediction probability. Theoretically, we prove the convergence of the meta-learning algorithm in MTEM and analyze the effectiveness of MTEM in achieving domain adaptation. Experimentally, MTEM improves the adaptation performance of BERT with an average of 4 percent on the benchmark dataset.

{{</citation>}}


### (3/82) How Good Are SOTA Fake News Detectors (Matthew Iceland, 2023)

{{<citation>}}

Matthew Iceland. (2023)  
**How Good Are SOTA Fake News Detectors**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2308.02727v1)  

---


**ABSTRACT**  
Automatic fake news detection with machine learning can prevent the dissemination of false statements before they gain many views. Several datasets labeling statements as legitimate or false have been created since the 2016 United States presidential election for the prospect of training machine learning models. We evaluate the robustness of both traditional and deep state-of-the-art models to gauge how well they may perform in the real world. We find that traditional models tend to generalize better to data outside the distribution it was trained on compared to more recently-developed large language models, though the best model to use may depend on the specific task at hand.

{{</citation>}}


### (4/82) Legal Summarisation through LLMs: The PRODIGIT Project (Thiago Dal Pont et al., 2023)

{{<citation>}}

Thiago Dal Pont, Federico Galli, Andrea Loreggia, Giuseppe Pisano, Riccardo Rovatti, Giovanni Sartor. (2023)  
**Legal Summarisation through LLMs: The PRODIGIT Project**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, Legal  
[Paper Link](http://arxiv.org/abs/2308.04416v1)  

---


**ABSTRACT**  
We present some initial results of a large-scale Italian project called PRODIGIT which aims to support tax judges and lawyers through digital technology, focusing on AI. We have focused on generation of summaries of judicial decisions and on the extraction of related information, such as the identification of legal issues and decision-making criteria, and the specification of keywords. To this end, we have deployed and evaluated different tools and approaches to extractive and abstractive summarisation. We have applied LLMs, and particularly on GPT4, which has enabled us to obtain results that proved satisfactory, according to an evaluation by expert tax judges and lawyers. On this basis, a prototype application is being built which will be made publicly available.

{{</citation>}}


### (5/82) Text2KGBench: A Benchmark for Ontology-Driven Knowledge Graph Generation from Text (Nandana Mihindukulasooriya et al., 2023)

{{<citation>}}

Nandana Mihindukulasooriya, Sanju Tiwari, Carlos F. Enguix, Kusum Lata. (2023)  
**Text2KGBench: A Benchmark for Ontology-Driven Knowledge Graph Generation from Text**  

---
Primary Category: cs.CL  
Categories: 68, I-2-4; I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Knowledge Graph, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.02357v1)  

---


**ABSTRACT**  
The recent advances in large language models (LLM) and foundation models with emergent capabilities have been shown to improve the performance of many NLP tasks. LLMs and Knowledge Graphs (KG) can complement each other such that LLMs can be used for KG construction or completion while existing KGs can be used for different tasks such as making LLM outputs explainable or fact-checking in Neuro-Symbolic manner. In this paper, we present Text2KGBench, a benchmark to evaluate the capabilities of language models to generate KGs from natural language text guided by an ontology. Given an input ontology and a set of sentences, the task is to extract facts from the text while complying with the given ontology (concepts, relations, domain/range constraints) and being faithful to the input sentences. We provide two datasets (i) Wikidata-TekGen with 10 ontologies and 13,474 sentences and (ii) DBpedia-WebNLG with 19 ontologies and 4,860 sentences. We define seven evaluation metrics to measure fact extraction performance, ontology conformance, and hallucinations by LLMs. Furthermore, we provide results for two baseline models, Vicuna-13B and Alpaca-LoRA-13B using automatic prompt generation from test cases. The baseline results show that there is room for improvement using both Semantic Web and Natural Language Processing techniques.

{{</citation>}}


### (6/82) Dataflow Dialogue Generation (Joram Meron et al., 2023)

{{<citation>}}

Joram Meron, Victor Guimarães. (2023)  
**Dataflow Dialogue Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2308.02323v1)  

---


**ABSTRACT**  
We demonstrate task-oriented dialogue generation within the dataflow dialogue paradigm. We show an example of agenda driven dialogue generation for the MultiWOZ domain, and an example of generation without an agenda for the SMCalFlow domain, where we show an improvement in the accuracy of the translation of user requests to dataflow expressions when the generated dialogues are used to augment the translation training dataset.

{{</citation>}}


### (7/82) Learning to Select the Relevant History Turns in Conversational Question Answering (Munazza Zaib et al., 2023)

{{<citation>}}

Munazza Zaib, Wei Emma Zhang, Quan Z. Sheng, Subhash Sagar, Adnan Mahmood, Yang Zhang. (2023)  
**Learning to Select the Relevant History Turns in Conversational Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: Information Retrieval, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.02294v1)  

---


**ABSTRACT**  
The increasing demand for the web-based digital assistants has given a rapid rise in the interest of the Information Retrieval (IR) community towards the field of conversational question answering (ConvQA). However, one of the critical aspects of ConvQA is the effective selection of conversational history turns to answer the question at hand. The dependency between relevant history selection and correct answer prediction is an intriguing but under-explored area. The selected relevant context can better guide the system so as to where exactly in the passage to look for an answer. Irrelevant context, on the other hand, brings noise to the system, thereby resulting in a decline in the model's performance. In this paper, we propose a framework, DHS-ConvQA (Dynamic History Selection in Conversational Question Answering), that first generates the context and question entities for all the history turns, which are then pruned on the basis of similarity they share in common with the question at hand. We also propose an attention-based mechanism to re-rank the pruned terms based on their calculated weights of how useful they are in answering the question. In the end, we further aid the model by highlighting the terms in the re-ranked conversational history using a binary classification task and keeping the useful terms (predicted as 1) and ignoring the irrelevant terms (predicted as 0). We demonstrate the efficacy of our proposed framework with extensive experimental results on CANARD and QuAC -- the two popularly utilized datasets in ConvQA. We demonstrate that selecting relevant turns works better than rewriting the original question. We also investigate how adding the irrelevant history turns negatively impacts the model's performance and discuss the research challenges that demand more attention from the IR community.

{{</citation>}}


### (8/82) Redundancy Aware Multi-Reference Based Gainwise Evaluation of Extractive Summarization (Mousumi Akter et al., 2023)

{{<citation>}}

Mousumi Akter, Shubhra Kanti Karmaker Santu. (2023)  
**Redundancy Aware Multi-Reference Based Gainwise Evaluation of Extractive Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2308.02270v1)  

---


**ABSTRACT**  
While very popular for evaluating extractive summarization task, the ROUGE metric has long been criticized for its lack of semantic awareness and its ignorance about the ranking quality of the summarizer. Thanks to previous research that has addressed these issues by proposing a gain-based automated metric called Sem-nCG, which is both rank and semantic aware. However, Sem-nCG does not consider the amount of redundancy present in a model-generated summary and currently does not support evaluation with multiple reference summaries. Unfortunately, addressing both these limitations simultaneously is not trivial. Therefore, in this paper, we propose a redundancy-aware Sem-nCG metric and demonstrate how this new metric can be used to evaluate model summaries against multiple references. We also explore different ways of incorporating redundancy into the original metric through extensive experiments. Experimental results demonstrate that the new redundancy-aware metric exhibits a higher correlation with human judgments than the original Sem-nCG metric for both single and multiple reference scenarios.

{{</citation>}}


### (9/82) Sinhala-English Parallel Word Dictionary Dataset (Kasun Wickramasinghe et al., 2023)

{{<citation>}}

Kasun Wickramasinghe, Nisansa de Silva. (2023)  
**Sinhala-English Parallel Word Dictionary Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.02234v1)  

---


**ABSTRACT**  
Parallel datasets are vital for performing and evaluating any kind of multilingual task. However, in the cases where one of the considered language pairs is a low-resource language, the existing top-down parallel data such as corpora are lacking in both tally and quality due to the dearth of human annotation. Therefore, for low-resource languages, it is more feasible to move in the bottom-up direction where finer granular pairs such as dictionary datasets are developed first. They may then be used for mid-level tasks such as supervised multilingual word embedding alignment. These in turn can later guide higher-level tasks in the order of aligning sentence or paragraph text corpora used for Machine Translation (MT). Even though more approachable than generating and aligning a massive corpus for a low-resource language, for the same reason of apathy from larger research entities, even these finer granular data sets are lacking for some low-resource languages. We have observed that there is no free and open dictionary data set for the low-resource language, Sinhala. Thus, in this work, we introduce three parallel English-Sinhala word dictionaries (En-Si-dict-large, En-Si-dict-filtered, En-Si-dict-FastText) which help in multilingual Natural Language Processing (NLP) tasks related to English and Sinhala languages. In this paper, we explain the dataset creation pipeline as well as the experimental results of the tests we have carried out to verify the quality of the data sets. The data sets and the related scripts are available at https://github.com/kasunw22/sinhala-para-dict.

{{</citation>}}


### (10/82) Learning to Paraphrase Sentences to Different Complexity Levels (Alison Chi et al., 2023)

{{<citation>}}

Alison Chi, Li-Kuang Chen, Yi-Chen Chang, Shu-Hui Lee, Jason S. Chang. (2023)  
**Learning to Paraphrase Sentences to Different Complexity Levels**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.02226v1)  

---


**ABSTRACT**  
While sentence simplification is an active research topic in NLP, its adjacent tasks of sentence complexification and same-level paraphrasing are not. To train models on all three tasks, we present two new unsupervised datasets. We compare these datasets, one labeled by a weak classifier and the other by a rule-based approach, with a single supervised dataset. Using these three datasets for training, we perform extensive experiments on both multitasking and prompting strategies. Compared to other systems trained on unsupervised parallel data, models trained on our weak classifier labeled dataset achieve state-of-the-art performance on the ASSET simplification benchmark. Our models also outperform previous work on sentence level targeting. Finally, we establish how a handful of Large Language Models perform on these tasks under a zero-shot setting.

{{</citation>}}


### (11/82) ESRL: Efficient Sampling-based Reinforcement Learning for Sequence Generation (Chenglong Wang et al., 2023)

{{<citation>}}

Chenglong Wang, Hang Zhou, Yimin Hu, Yifu Huo, Bei Li, Tongran Liu, Tong Xiao, Jingbo Zhu. (2023)  
**ESRL: Efficient Sampling-based Reinforcement Learning for Sequence Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.02223v1)  

---


**ABSTRACT**  
Applying Reinforcement Learning (RL) to sequence generation models enables the direct optimization of long-term rewards (\textit{e.g.,} BLEU and human feedback), but typically requires large-scale sampling over a space of action sequences. This is a computational challenge as presented by the practice of sequence generation problems, such as machine translation, where we often deal with a large action space (\textit{e.g.,} a vocabulary) and a long action sequence (\textit{e.g.,} a translation). In this work, we introduce two-stage sampling and dynamic sampling approaches to improve the sampling efficiency during training sequence generation models via RL. We experiment with our approaches on the traditional sequence generation tasks, including machine translation and abstractive summarization. Furthermore, we evaluate our approaches in RL from human feedback (RLHF) through training a large language model using the reward model. Experimental results show that the efficient sampling-based RL, referred to as ESRL, can outperform all baselines in terms of both training efficiency and memory consumption. Notably, ESRL yields consistent performance gains over the strong REINFORCE, minimum risk training, and proximal policy optimization methods.

{{</citation>}}


### (12/82) A Survey of Spanish Clinical Language Models (Guillem García Subies et al., 2023)

{{<citation>}}

Guillem García Subies, Álvaro Barbero Jiménez, Paloma Martínez Fernández. (2023)  
**A Survey of Spanish Clinical Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Clinical, Language Model  
[Paper Link](http://arxiv.org/abs/2308.02199v1)  

---


**ABSTRACT**  
This survey focuses in encoder Language Models for solving tasks in the clinical domain in the Spanish language. We review the contributions of 17 corpora focused mainly in clinical tasks, then list the most relevant Spanish Language Models and Spanish Clinical Language models. We perform a thorough comparison of these models by benchmarking them over a curated subset of the available corpora, in order to find the best-performing ones; in total more than 3000 models were fine-tuned for this study. All the tested corpora and the best models are made publically available in an accessible way, so that the results can be reproduced by independent teams or challenged in the future when new Spanish Clinical Language models are created.

{{</citation>}}


### (13/82) Explaining Relation Classification Models with Semantic Extents (Lars Klöser et al., 2023)

{{<citation>}}

Lars Klöser, Andre Büsgen, Philipp Kohl, Bodo Kraft, Albert Zündorf. (2023)  
**Explaining Relation Classification Models with Semantic Extents**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: BERT, GPT  
[Paper Link](http://arxiv.org/abs/2308.02193v1)  

---


**ABSTRACT**  
In recent years, the development of large pretrained language models, such as BERT and GPT, significantly improved information extraction systems on various tasks, including relation classification. State-of-the-art systems are highly accurate on scientific benchmarks. A lack of explainability is currently a complicating factor in many real-world applications. Comprehensible systems are necessary to prevent biased, counterintuitive, or harmful decisions.   We introduce semantic extents, a concept to analyze decision patterns for the relation classification task. Semantic extents are the most influential parts of texts concerning classification decisions. Our definition allows similar procedures to determine semantic extents for humans and models. We provide an annotation tool and a software framework to determine semantic extents for humans and models conveniently and reproducibly. Comparing both reveals that models tend to learn shortcut patterns from data. These patterns are hard to detect with current interpretability methods, such as input reductions. Our approach can help detect and eliminate spurious decision patterns during model development. Semantic extents can increase the reliability and security of natural language processing systems. Semantic extents are an essential step in enabling applications in critical areas like healthcare or finance. Moreover, our work opens new research directions for developing methods to explain deep learning models.

{{</citation>}}


### (14/82) Scaling Clinical Trial Matching Using Large Language Models: A Case Study in Oncology (Cliff Wong et al., 2023)

{{<citation>}}

Cliff Wong, Sheng Zhang, Yu Gu, Christine Moung, Jacob Abel, Naoto Usuyama, Roshanthi Weerasinghe, Brian Piening, Tristan Naumann, Carlo Bifulco, Hoifung Poon. (2023)  
**Scaling Clinical Trial Matching Using Large Language Models: A Case Study in Oncology**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Clinical, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.02180v2)  

---


**ABSTRACT**  
Clinical trial matching is a key process in health delivery and discovery. In practice, it is plagued by overwhelming unstructured data and unscalable manual processing. In this paper, we conduct a systematic study on scaling clinical trial matching using large language models (LLMs), with oncology as the focus area. Our study is grounded in a clinical trial matching system currently in test deployment at a large U.S. health network. Initial findings are promising: out of box, cutting-edge LLMs, such as GPT-4, can already structure elaborate eligibility criteria of clinical trials and extract complex matching logic (e.g., nested AND/OR/NOT). While still far from perfect, LLMs substantially outperform prior strong baselines and may serve as a preliminary solution to help triage patient-trial candidates with humans in the loop. Our study also reveals a few significant growth areas for applying LLMs to end-to-end clinical trial matching, such as context limitation and accuracy, especially in structuring patient information from longitudinal medical records.

{{</citation>}}


### (15/82) Tweet Insights: A Visualization Platform to Extract Temporal Insights from Twitter (Daniel Loureiro et al., 2023)

{{<citation>}}

Daniel Loureiro, Kiamehr Rezaee, Talayeh Riahi, Francesco Barbieri, Leonardo Neves, Luis Espinosa Anke, Jose Camacho-Collados. (2023)  
**Tweet Insights: A Visualization Platform to Extract Temporal Insights from Twitter**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2308.02142v1)  

---


**ABSTRACT**  
This paper introduces a large collection of time series data derived from Twitter, postprocessed using word embedding techniques, as well as specialized fine-tuned language models. This data comprises the past five years and captures changes in n-gram frequency, similarity, sentiment and topic distribution. The interface built on top of this data enables temporal analysis for detecting and characterizing shifts in meaning, including complementary information to trending metrics, such as sentiment and topic association over time. We release an online demo for easy experimentation, and we share code and the underlying aggregated data for future work. In this paper, we also discuss three case studies unlocked thanks to our platform, showcasing its potential for temporal linguistic analysis.

{{</citation>}}


### (16/82) Chinese Financial Text Emotion Mining: GCGTS -- A Character Relationship-based Approach for Simultaneous Aspect-Opinion Pair Extraction (Qi Chen et al., 2023)

{{<citation>}}

Qi Chen, Dexi Liu. (2023)  
**Chinese Financial Text Emotion Mining: GCGTS -- A Character Relationship-based Approach for Simultaneous Aspect-Opinion Pair Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Financial, Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2308.02113v1)  

---


**ABSTRACT**  
Aspect-Opinion Pair Extraction (AOPE) from Chinese financial texts is a specialized task in fine-grained text sentiment analysis. The main objective is to extract aspect terms and opinion terms simultaneously from a diverse range of financial texts. Previous studies have mainly focused on developing grid annotation schemes within grid-based models to facilitate this extraction process. However, these methods often rely on character-level (token-level) feature encoding, which may overlook the logical relationships between Chinese characters within words. To address this limitation, we propose a novel method called Graph-based Character-level Grid Tagging Scheme (GCGTS). The GCGTS method explicitly incorporates syntactic structure using Graph Convolutional Networks (GCN) and unifies the encoding of characters within the same syntactic semantic unit (Chinese word level). Additionally, we introduce an image convolutional structure into the grid model to better capture the local relationships between characters within evaluation units. This innovative structure reduces the excessive reliance on pre-trained language models and emphasizes the modeling of structure and local relationships, thereby improving the performance of the model on Chinese financial texts. Through comparative experiments with advanced models such as Synchronous Double-channel Recurrent Network (SDRN) and Grid Tagging Scheme (GTS), the proposed GCGTS model demonstrates significant improvements in performance.

{{</citation>}}


### (17/82) N-gram Boosting: Improving Contextual Biasing with Normalized N-gram Targets (Wang Yau Li et al., 2023)

{{<citation>}}

Wang Yau Li, Shreekantha Nadig, Karol Chang, Zafarullah Mahmood, Riqiang Wang, Simon Vandieken, Jonas Robertson, Fred Mailhot. (2023)  
**N-gram Boosting: Improving Contextual Biasing with Normalized N-gram Targets**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.02092v1)  

---


**ABSTRACT**  
Accurate transcription of proper names and technical terms is particularly important in speech-to-text applications for business conversations. These words, which are essential to understanding the conversation, are often rare and therefore likely to be under-represented in text and audio training data, creating a significant challenge in this domain. We present a two-step keyword boosting mechanism that successfully works on normalized unigrams and n-grams rather than just single tokens, which eliminates missing hits issues with boosting raw targets. In addition, we show how adjusting the boosting weight logic avoids over-boosting multi-token keywords. This improves our keyword recognition rate by 26% relative on our proprietary in-domain dataset and 2% on LibriSpeech. This method is particularly useful on targets that involve non-alphabetic characters or have non-standard pronunciations.

{{</citation>}}


## cs.RO (5)



### (18/82) Deep Reinforcement Learning for Autonomous Spacecraft Inspection using Illumination (David van Wijk et al., 2023)

{{<citation>}}

David van Wijk, Kyle Dunlap, Manoranjan Majji, Kerianne L. Hobbs. (2023)  
**Deep Reinforcement Learning for Autonomous Spacecraft Inspection using Illumination**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.02743v1)  

---


**ABSTRACT**  
This paper investigates the problem of on-orbit spacecraft inspection using a single free-flying deputy spacecraft, equipped with an optical sensor, whose controller is a neural network control system trained with Reinforcement Learning (RL). This work considers the illumination of the inspected spacecraft (chief) by the Sun in order to incentivize acquisition of well-illuminated optical data. The agent's performance is evaluated through statistically efficient metrics. Results demonstrate that the RL agent is able to inspect all points on the chief successfully, while maximizing illumination on inspected points in a simulated environment, using only low-level actions. Due to the stochastic nature of RL, 10 policies were trained using 10 random seeds to obtain a more holistic measure of agent performance. Over these 10 seeds, the interquartile mean (IQM) percentage of inspected points for the finalized model was 98.82%.

{{</citation>}}


### (19/82) Nonprehensile Planar Manipulation through Reinforcement Learning with Multimodal Categorical Exploration (Juan Del Aguila Ferrandis et al., 2023)

{{<citation>}}

Juan Del Aguila Ferrandis, João Moura, Sethu Vijayakumar. (2023)  
**Nonprehensile Planar Manipulation through Reinforcement Learning with Multimodal Categorical Exploration**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.02459v1)  

---


**ABSTRACT**  
Developing robot controllers capable of achieving dexterous nonprehensile manipulation, such as pushing an object on a table, is challenging. The underactuated and hybrid-dynamics nature of the problem, further complicated by the uncertainty resulting from the frictional interactions, requires sophisticated control behaviors. Reinforcement Learning (RL) is a powerful framework for developing such robot controllers. However, previous RL literature addressing the nonprehensile pushing task achieves low accuracy, non-smooth trajectories, and only simple motions, i.e. without rotation of the manipulated object. We conjecture that previously used unimodal exploration strategies fail to capture the inherent hybrid-dynamics of the task, arising from the different possible contact interaction modes between the robot and the object, such as sticking, sliding, and separation. In this work, we propose a multimodal exploration approach through categorical distributions, which enables us to train planar pushing RL policies for arbitrary starting and target object poses, i.e. positions and orientations, and with improved accuracy. We show that the learned policies are robust to external disturbances and observation noise, and scale to tasks with multiple pushers. Furthermore, we validate the transferability of the learned policies, trained entirely in simulation, to a physical robot hardware using the KUKA iiwa robot arm. See our supplemental video: https://youtu.be/vTdva1mgrk4.

{{</citation>}}


### (20/82) ExploitFlow, cyber security exploitation routes for Game Theory and AI research in robotics (Víctor Mayoral-Vilches et al., 2023)

{{<citation>}}

Víctor Mayoral-Vilches, Gelei Deng, Yi Liu, Martin Pinzger, Stefan Rass. (2023)  
**ExploitFlow, cyber security exploitation routes for Game Theory and AI research in robotics**  

---
Primary Category: cs.RO  
Categories: cs-CR, cs-RO, cs.RO  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.02152v1)  

---


**ABSTRACT**  
This paper addresses the prevalent lack of tools to facilitate and empower Game Theory and Artificial Intelligence (AI) research in cybersecurity. The primary contribution is the introduction of ExploitFlow (EF), an AI and Game Theory-driven modular library designed for cyber security exploitation. EF aims to automate attacks, combining exploits from various sources, and capturing system states post-action to reason about them and understand potential attack trees. The motivation behind EF is to bolster Game Theory and AI research in cybersecurity, with robotics as the initial focus. Results indicate that EF is effective for exploring machine learning in robot cybersecurity. An artificial agent powered by EF, using Reinforcement Learning, outperformed both brute-force and human expert approaches, laying the path for using ExploitFlow for further research. Nonetheless, we identified several limitations in EF-driven agents, including a propensity to overfit, the scarcity and production cost of datasets for generalization, and challenges in interpreting networking states across varied security settings. To leverage the strengths of ExploitFlow while addressing identified shortcomings, we present Malism, our vision for a comprehensive automated penetration testing framework with ExploitFlow at its core.

{{</citation>}}


### (21/82) Learning to Shape by Grinding: Cutting-surface-aware Model-based Reinforcement Learning (Takumi Hachimine et al., 2023)

{{<citation>}}

Takumi Hachimine, Jun Morimoto, Takamitsu Matsubara. (2023)  
**Learning to Shape by Grinding: Cutting-surface-aware Model-based Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.02150v1)  

---


**ABSTRACT**  
Object shaping by grinding is a crucial industrial process in which a rotating grinding belt removes material. Object-shape transition models are essential to achieving automation by robots; however, learning such a complex model that depends on process conditions is challenging because it requires a significant amount of data, and the irreversible nature of the removal process makes data collection expensive. This paper proposes a cutting-surface-aware Model-Based Reinforcement Learning (MBRL) method for robotic grinding. Our method employs a cutting-surface-aware model as the object's shape transition model, which in turn is composed of a geometric cutting model and a cutting-surface-deviation model, based on the assumption that the robot action can specify the cutting surface made by the tool. Furthermore, according to the grinding resistance theory, the cutting-surface-deviation model does not require raw shape information, making the model's dimensions smaller and easier to learn than a naive shape transition model directly mapping the shapes. Through evaluation and comparison by simulation and real robot experiments, we confirm that our MBRL method can achieve high data efficiency for learning object shaping by grinding and also provide generalization capability for initial and target shapes that differ from the training data.

{{</citation>}}


### (22/82) Semantics-guided Transformer-based Sensor Fusion for Improved Waypoint Prediction (Hwan-Soo Choi et al., 2023)

{{<citation>}}

Hwan-Soo Choi, Jongoh Jeong, Young Hoo Cho, Kuk-Jin Yoon, Jong-Hwan Kim. (2023)  
**Semantics-guided Transformer-based Sensor Fusion for Improved Waypoint Prediction**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.02126v1)  

---


**ABSTRACT**  
Sensor fusion approaches for intelligent self-driving agents remain key to driving scene understanding given visual global contexts acquired from input sensors. Specifically, for the local waypoint prediction task, single-modality networks are still limited by strong dependency on the sensitivity of the input sensor, and thus recent works promote the use of multiple sensors in fusion in feature level. While it is well known that multiple data modalities promote mutual contextual exchange, deployment to practical driving scenarios requires global 3D scene understanding in real-time with minimal computations, thus placing greater significance on training strategies given a limited number of practically usable sensors. In this light, we exploit carefully selected auxiliary tasks that are highly correlated with the target task of interest (e.g., traffic light recognition and semantic segmentation) by fusing auxiliary task features and also using auxiliary heads for waypoint prediction based on imitation learning. Our multi-task feature fusion augments and improves the base network, TransFuser, by significant margins for safer and more complete road navigation in CARLA simulator as validated on the Town05 Benchmark through extensive experiments.

{{</citation>}}


## cs.LG (15)



### (23/82) Personalization of Stress Mobile Sensing using Self-Supervised Learning (Tanvir Islam et al., 2023)

{{<citation>}}

Tanvir Islam, Peter Washington. (2023)  
**Personalization of Stress Mobile Sensing using Self-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.02731v1)  

---


**ABSTRACT**  
Stress is widely recognized as a major contributor to a variety of health issues. Stress prediction using biosignal data recorded by wearables is a key area of study in mobile sensing research because real-time stress prediction can enable digital interventions to immediately react at the onset of stress, helping to avoid many psychological and physiological symptoms such as heart rhythm irregularities. Electrodermal activity (EDA) is often used to measure stress. However, major challenges with the prediction of stress using machine learning include the subjectivity and sparseness of the labels, a large feature space, relatively few labels, and a complex nonlinear and subjective relationship between the features and outcomes. To tackle these issues, we examine the use of model personalization: training a separate stress prediction model for each user. To allow the neural network to learn the temporal dynamics of each individual's baseline biosignal patterns, thus enabling personalization with very few labels, we pre-train a 1-dimensional convolutional neural network (CNN) using self-supervised learning (SSL). We evaluate our method using the Wearable Stress and Affect prediction (WESAD) dataset. We fine-tune the pre-trained networks to the stress prediction task and compare against equivalent models without any self-supervised pre-training. We discover that embeddings learned using our pre-training method outperform supervised baselines with significantly fewer labeled data points: the models trained with SSL require less than 30% of the labels to reach equivalent performance without personalized SSL. This personalized learning method can enable precision health systems which are tailored to each subject and require few annotations by the end user, thus allowing for the mobile sensing of increasingly complex, heterogeneous, and subjective outcomes such as stress.

{{</citation>}}


### (24/82) Synthesizing Programmatic Policies with Actor-Critic Algorithms and ReLU Networks (Spyros Orfanos et al., 2023)

{{<citation>}}

Spyros Orfanos, Levi H. S. Lelis. (2023)  
**Synthesizing Programmatic Policies with Actor-Critic Algorithms and ReLU Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.02729v1)  

---


**ABSTRACT**  
Programmatically Interpretable Reinforcement Learning (PIRL) encodes policies in human-readable computer programs. Novel algorithms were recently introduced with the goal of handling the lack of gradient signal to guide the search in the space of programmatic policies. Most of such PIRL algorithms first train a neural policy that is used as an oracle to guide the search in the programmatic space. In this paper, we show that such PIRL-specific algorithms are not needed, depending on the language used to encode the programmatic policies. This is because one can use actor-critic algorithms to directly obtain a programmatic policy. We use a connection between ReLU neural networks and oblique decision trees to translate the policy learned with actor-critic algorithms into programmatic policies. This translation from ReLU networks allows us to synthesize policies encoded in programs with if-then-else structures, linear transformations of the input values, and PID operations. Empirical results on several control problems show that this translation approach is capable of learning short and effective policies. Moreover, the translated policies are at least competitive and often far superior to the policies PIRL algorithms synthesize.

{{</citation>}}


### (25/82) Fluid Property Prediction Leveraging AI and Robotics (Jong Hoon Park et al., 2023)

{{<citation>}}

Jong Hoon Park, Gauri Pramod Dalwankar, Alison Bartsch, Abraham George, Amir Barati Farimani. (2023)  
**Fluid Property Prediction Leveraging AI and Robotics**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02715v1)  

---


**ABSTRACT**  
Inferring liquid properties from vision is a challenging task due to the complex nature of fluids, both in behavior and detection. Nevertheless, the ability to infer their properties directly from visual information is highly valuable for autonomous fluid handling systems, as cameras are readily available. Moreover, predicting fluid properties purely from vision can accelerate the process of fluid characterization saving considerable time and effort in various experimental environments. In this work, we present a purely vision-based approach to estimate viscosity, leveraging the fact that the behavior of the fluid oscillations is directly related to the viscosity. Specifically, we utilize a 3D convolutional autoencoder to learn latent representations of different fluid-oscillating patterns present in videos. We leverage this latent representation to visually infer the category of fluid or the dynamics viscosity of fluid from video.

{{</citation>}}


### (26/82) FPR Estimation for Fraud Detection in the Presence of Class-Conditional Label Noise (Justin Tittelfitz, 2023)

{{<citation>}}

Justin Tittelfitz. (2023)  
**FPR Estimation for Fraud Detection in the Presence of Class-Conditional Label Noise**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Fraud Detection  
[Paper Link](http://arxiv.org/abs/2308.02695v1)  

---


**ABSTRACT**  
We consider the problem of estimating the false-/ true-positive-rate (FPR/TPR) for a binary classification model when there are incorrect labels (label noise) in the validation set. Our motivating application is fraud prevention where accurate estimates of FPR are critical to preserving the experience for good customers, and where label noise is highly asymmetric. Existing methods seek to minimize the total error in the cleaning process - to avoid cleaning examples that are not noise, and to ensure cleaning of examples that are. This is an important measure of accuracy but insufficient to guarantee good estimates of the true FPR or TPR for a model, and we show that using the model to directly clean its own validation data leads to underestimates even if total error is low. This indicates a need for researchers to pursue methods that not only reduce total error but also seek to de-correlate cleaning error with model scores.

{{</citation>}}


### (27/82) Explainable Deep Learning-based Solar Flare Prediction with post hoc Attention for Operational Forecasting (Chetraj Pandey et al., 2023)

{{<citation>}}

Chetraj Pandey, Rafal A. Angryk, Manolis K. Georgoulis, Berkay Aydin. (2023)  
**Explainable Deep Learning-based Solar Flare Prediction with post hoc Attention for Operational Forecasting**  

---
Primary Category: cs.LG  
Categories: astro-ph-SR, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.02682v1)  

---


**ABSTRACT**  
This paper presents a post hoc analysis of a deep learning-based full-disk solar flare prediction model. We used hourly full-disk line-of-sight magnetogram images and selected binary prediction mode to predict the occurrence of $\geq$M1.0-class flares within 24 hours. We leveraged custom data augmentation and sample weighting to counter the inherent class-imbalance problem and used true skill statistic and Heidke skill score as evaluation metrics. Recent advancements in gradient-based attention methods allow us to interpret models by sending gradient signals to assign the burden of the decision on the input features. We interpret our model using three post hoc attention methods: (i) Guided Gradient-weighted Class Activation Mapping, (ii) Deep Shapley Additive Explanations, and (iii) Integrated Gradients. Our analysis shows that full-disk predictions of solar flares align with characteristics related to the active regions. The key findings of this study are: (1) We demonstrate that our full disk model can tangibly locate and predict near-limb solar flares, which is a critical feature for operational flare forecasting, (2) Our candidate model achieves an average TSS=0.51$\pm$0.05 and HSS=0.38$\pm$0.08, and (3) Our evaluation suggests that these models can learn conspicuous features corresponding to active regions from full-disk magnetograms.

{{</citation>}}


### (28/82) BlindSage: Label Inference Attacks against Node-level Vertical Federated Graph Neural Networks (Marco Arazzi et al., 2023)

{{<citation>}}

Marco Arazzi, Mauro Conti, Stefanos Koffas, Marina Krcek, Antonino Nocera, Stjepan Picek, Jing Xu. (2023)  
**BlindSage: Label Inference Attacks against Node-level Vertical Federated Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.02465v1)  

---


**ABSTRACT**  
Federated learning enables collaborative training of machine learning models by keeping the raw data of the involved workers private. One of its main objectives is to improve the models' privacy, security, and scalability. Vertical Federated Learning (VFL) offers an efficient cross-silo setting where a few parties collaboratively train a model without sharing the same features. In such a scenario, classification labels are commonly considered sensitive information held exclusively by one (active) party, while other (passive) parties use only their local information. Recent works have uncovered important flaws of VFL, leading to possible label inference attacks under the assumption that the attacker has some, even limited, background knowledge on the relation between labels and data. In this work, we are the first (to the best of our knowledge) to investigate label inference attacks on VFL using a zero-background knowledge strategy. To concretely formulate our proposal, we focus on Graph Neural Networks (GNNs) as a target model for the underlying VFL. In particular, we refer to node classification tasks, which are widely studied, and GNNs have shown promising results. Our proposed attack, BlindSage, provides impressive results in the experiments, achieving nearly 100% accuracy in most cases. Even when the attacker has no information about the used architecture or the number of classes, the accuracy remained above 85% in most instances. Finally, we observe that well-known defenses cannot mitigate our attack without affecting the model's performance on the main classification task.

{{</citation>}}


### (29/82) Harnessing the Web and Knowledge Graphs for Automated Impact Investing Scoring (Qingzhi Hu et al., 2023)

{{<citation>}}

Qingzhi Hu, Daniel Daza, Laurens Swinkels, Kristina Ūsaitė, Robbert-Jan 't Hoen, Paul Groth. (2023)  
**Harnessing the Web and Knowledge Graphs for Automated Impact Investing Scoring**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-IR, cs-LG, cs-SI, cs.LG  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.02622v1)  

---


**ABSTRACT**  
The Sustainable Development Goals (SDGs) were introduced by the United Nations in order to encourage policies and activities that help guarantee human prosperity and sustainability. SDG frameworks produced in the finance industry are designed to provide scores that indicate how well a company aligns with each of the 17 SDGs. This scoring enables a consistent assessment of investments that have the potential of building an inclusive and sustainable economy. As a result of the high quality and reliability required by such frameworks, the process of creating and maintaining them is time-consuming and requires extensive domain expertise. In this work, we describe a data-driven system that seeks to automate the process of creating an SDG framework. First, we propose a novel method for collecting and filtering a dataset of texts from different web sources and a knowledge graph relevant to a set of companies. We then implement and deploy classifiers trained with this data for predicting scores of alignment with SDGs for a given company. Our results indicate that our best performing model can accurately predict SDG scores with a micro average F1 score of 0.89, demonstrating the effectiveness of the proposed solution. We further describe how the integration of the models for its use by humans can be facilitated by providing explanations in the form of data relevant to a predicted score. We find that our proposed solution enables access to a large amount of information that analysts would normally not be able to process, resulting in an accurate prediction of SDG scores at a fraction of the cost.

{{</citation>}}


### (30/82) RobustMQ: Benchmarking Robustness of Quantized Models (Yisong Xiao et al., 2023)

{{<citation>}}

Yisong Xiao, Aishan Liu, Tianyuan Zhang, Haotong Qin, Jinyang Guo, Xianglong Liu. (2023)  
**RobustMQ: Benchmarking Robustness of Quantized Models**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: ImageNet, Quantization  
[Paper Link](http://arxiv.org/abs/2308.02350v1)  

---


**ABSTRACT**  
Quantization has emerged as an essential technique for deploying deep neural networks (DNNs) on devices with limited resources. However, quantized models exhibit vulnerabilities when exposed to various noises in real-world applications. Despite the importance of evaluating the impact of quantization on robustness, existing research on this topic is limited and often disregards established principles of robustness evaluation, resulting in incomplete and inconclusive findings. To address this gap, we thoroughly evaluated the robustness of quantized models against various noises (adversarial attacks, natural corruptions, and systematic noises) on ImageNet. The comprehensive evaluation results empirically provide valuable insights into the robustness of quantized models in various scenarios, for example: (1) quantized models exhibit higher adversarial robustness than their floating-point counterparts, but are more vulnerable to natural corruptions and systematic noises; (2) in general, increasing the quantization bit-width results in a decrease in adversarial robustness, an increase in natural robustness, and an increase in systematic robustness; (3) among corruption methods, \textit{impulse noise} and \textit{glass blur} are the most harmful to quantized models, while \textit{brightness} has the least impact; (4) among systematic noises, the \textit{nearest neighbor interpolation} has the highest impact, while bilinear interpolation, cubic interpolation, and area interpolation are the three least harmful. Our research contributes to advancing the robust quantization of models and their deployment in real-world scenarios.

{{</citation>}}


### (31/82) Vehicles Control: Collision Avoidance using Federated Deep Reinforcement Learning (Badr Ben Elallid et al., 2023)

{{<citation>}}

Badr Ben Elallid, Amine Abouaomar, Nabil Benamar, Abdellatif Kobbane. (2023)  
**Vehicles Control: Collision Avoidance using Federated Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.02614v1)  

---


**ABSTRACT**  
In the face of growing urban populations and the escalating number of vehicles on the roads, managing transportation efficiently and ensuring safety have become critical challenges. To tackle these issues, the development of intelligent control systems for vehicles is paramount. This paper presents a comprehensive study on vehicle control for collision avoidance, leveraging the power of Federated Deep Reinforcement Learning (FDRL) techniques. Our main goal is to minimize travel delays and enhance the average speed of vehicles while prioritizing safety and preserving data privacy. To accomplish this, we conducted a comparative analysis between the local model, Deep Deterministic Policy Gradient (DDPG), and the global model, Federated Deep Deterministic Policy Gradient (FDDPG), to determine their effectiveness in optimizing vehicle control for collision avoidance. The results obtained indicate that the FDDPG algorithm outperforms DDPG in terms of effectively controlling vehicles and preventing collisions. Significantly, the FDDPG-based algorithm demonstrates substantial reductions in travel delays and notable improvements in average speed compared to the DDPG algorithm.

{{</citation>}}


### (32/82) RAHNet: Retrieval Augmented Hybrid Network for Long-tailed Graph Classification (Zhengyang Mao et al., 2023)

{{<citation>}}

Zhengyang Mao, Wei Ju, Yifang Qin, Xiao Luo, Ming Zhang. (2023)  
**RAHNet: Retrieval Augmented Hybrid Network for Long-tailed Graph Classification**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-IR, cs-LG, cs-SI, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.02335v1)  

---


**ABSTRACT**  
Graph classification is a crucial task in many real-world multimedia applications, where graphs can represent various multimedia data types such as images, videos, and social networks. Previous efforts have applied graph neural networks (GNNs) in balanced situations where the class distribution is balanced. However, real-world data typically exhibit long-tailed class distributions, resulting in a bias towards the head classes when using GNNs and limited generalization ability over the tail classes. Recent approaches mainly focus on re-balancing different classes during model training, which fails to explicitly introduce new knowledge and sacrifices the performance of the head classes. To address these drawbacks, we propose a novel framework called Retrieval Augmented Hybrid Network (RAHNet) to jointly learn a robust feature extractor and an unbiased classifier in a decoupled manner. In the feature extractor training stage, we develop a graph retrieval module to search for relevant graphs that directly enrich the intra-class diversity for the tail classes. Moreover, we innovatively optimize a category-centered supervised contrastive loss to obtain discriminative representations, which is more suitable for long-tailed scenarios. In the classifier fine-tuning stage, we balance the classifier weights with two weight regularization techniques, i.e., Max-norm and weight decay. Experiments on various popular benchmarks verify the superiority of the proposed method against state-of-the-art approaches.

{{</citation>}}


### (33/82) Interoperable synthetic health data with SyntHIR to enable the development of CDSS tools (Pavitra Chauhan et al., 2023)

{{<citation>}}

Pavitra Chauhan, Mohsen Gamal Saad Askar, Bjørn Fjukstad, Lars Ailo Bongo, Edvard Pedersen. (2023)  
**Interoperable synthetic health data with SyntHIR to enable the development of CDSS tools**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SE, cs.LG  
Keywords: Azure, Clinical, Microsoft  
[Paper Link](http://arxiv.org/abs/2308.02613v1)  

---


**ABSTRACT**  
There is a great opportunity to use high-quality patient journals and health registers to develop machine learning-based Clinical Decision Support Systems (CDSS). To implement a CDSS tool in a clinical workflow, there is a need to integrate, validate and test this tool on the Electronic Health Record (EHR) systems used to store and manage patient data. However, it is often not possible to get the necessary access to an EHR system due to legal compliance. We propose an architecture for generating and using synthetic EHR data for CDSS tool development. The architecture is implemented in a system called SyntHIR. The SyntHIR system uses the Fast Healthcare Interoperability Resources (FHIR) standards for data interoperability, the Gretel framework for generating synthetic data, the Microsoft Azure FHIR server as the FHIR-based EHR system and SMART on FHIR framework for tool transportability. We demonstrate the usefulness of SyntHIR by developing a machine learning-based CDSS tool using data from the Norwegian Patient Register (NPR) and Norwegian Patient Prescriptions (NorPD). We demonstrate the development of the tool on the SyntHIR system and then lift it to the Open DIPS environment. In conclusion, SyntHIR provides a generic architecture for CDSS tool development using synthetic FHIR data and a testing environment before implementing it in a clinical setting. However, there is scope for improvement in terms of the quality of the synthetic data generated. The code is open source and available at https://github.com/potter-coder89/SyntHIR.git.

{{</citation>}}


### (34/82) DIVERSIFY: A General Framework for Time Series Out-of-distribution Detection and Generalization (Wang Lu et al., 2023)

{{<citation>}}

Wang Lu, Jindong Wang, Xinwei Sun, Yiqiang Chen, Xiangyang Ji, Qiang Yang, Xing Xie. (2023)  
**DIVERSIFY: A General Framework for Time Series Out-of-distribution Detection and Generalization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, eess-SP  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.02282v1)  

---


**ABSTRACT**  
Time series remains one of the most challenging modalities in machine learning research. The out-of-distribution (OOD) detection and generalization on time series tend to suffer due to its non-stationary property, i.e., the distribution changes over time. The dynamic distributions inside time series pose great challenges to existing algorithms to identify invariant distributions since they mainly focus on the scenario where the domain information is given as prior knowledge. In this paper, we attempt to exploit subdomains within a whole dataset to counteract issues induced by non-stationary for generalized representation learning. We propose DIVERSIFY, a general framework, for OOD detection and generalization on dynamic distributions of time series. DIVERSIFY takes an iterative process: it first obtains the "worst-case" latent distribution scenario via adversarial training, then reduces the gap between these latent distributions. We implement DIVERSIFY via combining existing OOD detection methods according to either extracted features or outputs of models for detection while we also directly utilize outputs for classification. In addition, theoretical insights illustrate that DIVERSIFY is theoretically supported. Extensive experiments are conducted on seven datasets with different OOD settings across gesture recognition, speech commands recognition, wearable stress and affect detection, and sensor-based human activity recognition. Qualitative and quantitative results demonstrate that DIVERSIFY learns more generalized features and significantly outperforms other baselines.

{{</citation>}}


### (35/82) Knowledge-Driven Multi-Agent Reinforcement Learning for Computation Offloading in Cybertwin-Enabled Internet of Vehicles (Ruijin Sun et al., 2023)

{{<citation>}}

Ruijin Sun, Xiao Yang, Nan Cheng, Xiucheng Wang, Changle Li. (2023)  
**Knowledge-Driven Multi-Agent Reinforcement Learning for Computation Offloading in Cybertwin-Enabled Internet of Vehicles**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.02603v1)  

---


**ABSTRACT**  
By offloading computation-intensive tasks of vehicles to roadside units (RSUs), mobile edge computing (MEC) in the Internet of Vehicles (IoV) can relieve the onboard computation burden. However, existing model-based task offloading methods suffer from heavy computational complexity with the increase of vehicles and data-driven methods lack interpretability. To address these challenges, in this paper, we propose a knowledge-driven multi-agent reinforcement learning (KMARL) approach to reduce the latency of task offloading in cybertwin-enabled IoV. Specifically, in the considered scenario, the cybertwin serves as a communication agent for each vehicle to exchange information and make offloading decisions in the virtual space. To reduce the latency of task offloading, a KMARL approach is proposed to select the optimal offloading option for each vehicle, where graph neural networks are employed by leveraging domain knowledge concerning graph-structure communication topology and permutation invariance into neural networks. Numerical results show that our proposed KMARL yields higher rewards and demonstrates improved scalability compared with other methods, benefitting from the integration of domain knowledge.

{{</citation>}}


### (36/82) Improved Order Analysis and Design of Exponential Integrator for Diffusion Models Sampling (Qinsheng Zhang et al., 2023)

{{<citation>}}

Qinsheng Zhang, Jiaming Song, Yongxin Chen. (2023)  
**Improved Order Analysis and Design of Exponential Integrator for Diffusion Models Sampling**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.02157v1)  

---


**ABSTRACT**  
Efficient differential equation solvers have significantly reduced the sampling time of diffusion models (DMs) while retaining high sampling quality. Among these solvers, exponential integrators (EI) have gained prominence by demonstrating state-of-the-art performance. However, existing high-order EI-based sampling algorithms rely on degenerate EI solvers, resulting in inferior error bounds and reduced accuracy in contrast to the theoretically anticipated results under optimal settings. This situation makes the sampling quality extremely vulnerable to seemingly innocuous design choices such as timestep schedules. For example, an inefficient timestep scheduler might necessitate twice the number of steps to achieve a quality comparable to that obtained through carefully optimized timesteps. To address this issue, we reevaluate the design of high-order differential solvers for DMs. Through a thorough order analysis, we reveal that the degeneration of existing high-order EI solvers can be attributed to the absence of essential order conditions. By reformulating the differential equations in DMs and capitalizing on the theory of exponential integrators, we propose refined EI solvers that fulfill all the order conditions, which we designate as Refined Exponential Solver (RES). Utilizing these improved solvers, RES exhibits more favorable error bounds theoretically and achieves superior sampling efficiency and stability in practical applications. For instance, a simple switch from the single-step DPM-Solver++ to our order-satisfied RES solver when Number of Function Evaluations (NFE) $=9$, results in a reduction of numerical defects by $25.2\%$ and FID improvement of $25.4\%$ (16.77 vs 12.51) on a pre-trained ImageNet diffusion model.

{{</citation>}}


### (37/82) VQGraph: Graph Vector-Quantization for Bridging GNNs and MLPs (Ling Yang et al., 2023)

{{<citation>}}

Ling Yang, Ye Tian, Minkai Xu, Zhongyi Liu, Shenda Hong, Wei Qu, Wentao Zhang, Bin Cui, Muhan Zhang, Jure Leskovec. (2023)  
**VQGraph: Graph Vector-Quantization for Bridging GNNs and MLPs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Quantization  
[Paper Link](http://arxiv.org/abs/2308.02117v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) conduct message passing which aggregates local neighbors to update node representations. Such message passing leads to scalability issues in practical latency-constrained applications. To address this issue, recent methods adopt knowledge distillation (KD) to learn computationally-efficient multi-layer perceptron (MLP) by mimicking the output of GNN. However, the existing GNN representation space may not be expressive enough for representing diverse local structures of the underlying graph, which limits the knowledge transfer from GNN to MLP. Here we present a novel framework VQGraph to learn a powerful graph representation space for bridging GNNs and MLPs. We adopt the encoder of a variant of a vector-quantized variational autoencoder (VQ-VAE) as a structure-aware graph tokenizer, which explicitly represents the nodes of diverse local structures as numerous discrete tokens and constitutes a meaningful codebook. Equipped with the learned codebook, we propose a new token-based distillation objective based on soft token assignments to sufficiently transfer the structural knowledge from GNN to MLP. Extensive experiments and analyses demonstrate the strong performance of VQGraph, where we achieve new state-of-the-art performance on GNN-MLP distillation in both transductive and inductive settings across seven graph datasets. We show that VQGraph with better performance infers faster than GNNs by 828x, and also achieves accuracy improvement over GNNs and stand-alone MLPs by 3.90% and 28.05% on average, respectively. Code: https://github.com/YangLing0818/VQGraph.

{{</citation>}}


## cs.CV (13)



### (38/82) Lightweight Endoscopic Depth Estimation with CNN-Transformer Encoder (Yangke Li, 2023)

{{<citation>}}

Yangke Li. (2023)  
**Lightweight Endoscopic Depth Estimation with CNN-Transformer Encoder**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.02716v1)  

---


**ABSTRACT**  
In this study, we tackle the key challenges concerning accuracy and robustness in depth estimation for endoscopic imaging, with a particular emphasis on real-time inference and the impact of reflections. We propose an innovative lightweight solution that integrates Convolutional Neural Networks (CNN) and Transformers to predict multi-scale depth maps. Our approach includes optimizing the network architecture, incorporating multi-scale dilated convolution, and a multi-channel attention mechanism. We also introduce a statistical confidence boundary mask to minimize the impact of reflective areas. Moreover, we propose a novel complexity evaluation metric that considers network parameter size, floating-point operations, and inference frames per second. Our research aims to enhance the efficiency and safety of laparoscopic surgery significantly. We comprehensively evaluate our proposed method and compare it with existing solutions. The results demonstrate that our method ensures depth estimation accuracy while being lightweight.

{{</citation>}}


### (39/82) ReCLIP: Refine Contrastive Language Image Pre-Training with Source Free Domain Adaptation (Xuefeng Hu et al., 2023)

{{<citation>}}

Xuefeng Hu, Ke Zhang, Lu Xia, Albert Chen, Jiajia Luo, Yuyin Sun, Ken Wang, Nan Qiao, Xiao Zeng, Min Sun, Cheng-Hao Kuo, Ram Nevatia. (2023)  
**ReCLIP: Refine Contrastive Language Image Pre-Training with Source Free Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Language Model  
[Paper Link](http://arxiv.org/abs/2308.03793v1)  

---


**ABSTRACT**  
Large-scale Pre-Training Vision-Language Model such as CLIP has demonstrated outstanding performance in zero-shot classification, e.g. achieving 76.3% top-1 accuracy on ImageNet without seeing any example, which leads to potential benefits to many tasks that have no labeled data. However, while applying CLIP to a downstream target domain, the presence of visual and text domain gaps and cross-modality misalignment can greatly impact the model performance. To address such challenges, we propose ReCLIP, the first source-free domain adaptation method for vision-language models, which does not require any source data or target labeled data. ReCLIP first learns a projection space to mitigate the misaligned visual-text embeddings and learns pseudo labels, and then deploys cross-modality self-training with the pseudo labels, to update visual and text encoders, refine labels and reduce domain gaps and misalignments iteratively. With extensive experiments, we demonstrate ReCLIP reduces the average error rate of CLIP from 30.17% to 25.06% on 22 image classification benchmarks.

{{</citation>}}


### (40/82) Universal Defensive Underpainting Patch: Making Your Text Invisible to Optical Character Recognition (JiaCheng Deng et al., 2023)

{{<citation>}}

JiaCheng Deng, Li Dong, Jiahao Chen, Diqun Yan, Rangding Wang, Dengpan Ye, Lingchen Zhao, Jinyu Tian. (2023)  
**Universal Defensive Underpainting Patch: Making Your Text Invisible to Optical Character Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2308.02369v1)  

---


**ABSTRACT**  
Optical Character Recognition (OCR) enables automatic text extraction from scanned or digitized text images, but it also makes it easy to pirate valuable or sensitive text from these images. Previous methods to prevent OCR piracy by distorting characters in text images are impractical in real-world scenarios, as pirates can capture arbitrary portions of the text images, rendering the defenses ineffective. In this work, we propose a novel and effective defense mechanism termed the Universal Defensive Underpainting Patch (UDUP) that modifies the underpainting of text images instead of the characters. UDUP is created through an iterative optimization process to craft a small, fixed-size defensive patch that can generate non-overlapping underpainting for text images of any size. Experimental results show that UDUP effectively defends against unauthorized OCR under the setting of any screenshot range or complex image background. It is agnostic to the content, size, colors, and languages of characters, and is robust to typical image operations such as scaling and compressing. In addition, the transferability of UDUP is demonstrated by evading several off-the-shelf OCRs. The code is available at https://github.com/QRICKDD/UDUP.

{{</citation>}}


### (41/82) A Parameter-efficient Multi-subject Model for Predicting fMRI Activity (Connor Lane et al., 2023)

{{<citation>}}

Connor Lane, Gregory Kiar. (2023)  
**A Parameter-efficient Multi-subject Model for Predicting fMRI Activity**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2308.02351v1)  

---


**ABSTRACT**  
This is the Algonauts 2023 submission report for team "BlobGPT". Our model consists of a multi-subject linear encoding head attached to a pretrained trunk model. The multi-subject head consists of three components: (1) a shared multi-layer feature projection, (2) shared plus subject-specific low-dimension linear transformations, and (3) a shared PCA fMRI embedding. In this report, we explain these components in more detail and present some experimental results. Our code is available at https://github.com/cmi-dair/algonauts23.

{{</citation>}}


### (42/82) Class Incremental Learning with Self-Supervised Pre-Training and Prototype Learning (Wenzhuo Liu et al., 2023)

{{<citation>}}

Wenzhuo Liu, Xinjian Wu, Fei Zhu, Mingming Yu, Chuang Wang, Cheng-Lin Liu. (2023)  
**Class Incremental Learning with Self-Supervised Pre-Training and Prototype Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.02346v1)  

---


**ABSTRACT**  
Deep Neural Network (DNN) has achieved great success on datasets of closed class set. However, new classes, like new categories of social media topics, are continuously added to the real world, making it necessary to incrementally learn. This is hard for DNN because it tends to focus on fitting to new classes while ignoring old classes, a phenomenon known as catastrophic forgetting. State-of-the-art methods rely on knowledge distillation and data replay techniques but still have limitations. In this work, we analyze the causes of catastrophic forgetting in class incremental learning, which owes to three factors: representation drift, representation confusion, and classifier distortion. Based on this view, we propose a two-stage learning framework with a fixed encoder and an incrementally updated prototype classifier. The encoder is trained with self-supervised learning to generate a feature space with high intrinsic dimensionality, thus improving its transferability and generality. The classifier incrementally learns new prototypes while retaining the prototypes of previously learned data, which is crucial in preserving the decision boundary.Our method does not rely on preserved samples of old classes, is thus a non-exemplar based CIL method. Experiments on public datasets show that our method can significantly outperform state-of-the-art exemplar-based methods when they reserved 5 examplers per class, under the incremental setting of 10 phases, by 18.24% on CIFAR-100 and 9.37% on ImageNet100.

{{</citation>}}


### (43/82) On the Calibration of Uncertainty Estimation in LiDAR-based Semantic Segmentation (Mariella Dreissig et al., 2023)

{{<citation>}}

Mariella Dreissig, Florian Piewak, Joschka Boedecker. (2023)  
**On the Calibration of Uncertainty Estimation in LiDAR-based Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.02248v1)  

---


**ABSTRACT**  
The confidence calibration of deep learning-based perception models plays a crucial role in their reliability. Especially in the context of autonomous driving, downstream tasks like prediction and planning depend on accurate confidence estimates. In point-wise multiclass classification tasks like sematic segmentation the model has to deal with heavy class imbalances. Due to their underrepresentation, the confidence calibration of classes with smaller instances is challenging but essential, not only for safety reasons. We propose a metric to measure the confidence calibration quality of a semantic segmentation model with respect to individual classes. It is calculated by computing sparsification curves for each class based on the uncertainty estimates. We use the classification calibration metric to evaluate uncertainty estimation methods with respect to their confidence calibration of underrepresented classes. We furthermore suggest a double use for the method to automatically find label problems to improve the quality of hand- or auto-annotated datasets.

{{</citation>}}


### (44/82) Deep Semantic Model Fusion for Ancient Agricultural Terrace Detection (Yi Wang et al., 2023)

{{<citation>}}

Yi Wang, Chenying Liu, Arti Tiwari, Micha Silver, Arnon Karnieli, Xiao Xiang Zhu, Conrad M Albrecht. (2023)  
**Deep Semantic Model Fusion for Ancient Agricultural Terrace Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02225v1)  

---


**ABSTRACT**  
Discovering ancient agricultural terraces in desert regions is important for the monitoring of long-term climate changes on the Earth's surface. However, traditional ground surveys are both costly and limited in scale. With the increasing accessibility of aerial and satellite data, machine learning techniques bear large potential for the automatic detection and recognition of archaeological landscapes. In this paper, we propose a deep semantic model fusion method for ancient agricultural terrace detection. The input data includes aerial images and LiDAR generated terrain features in the Negev desert. Two deep semantic segmentation models, namely DeepLabv3+ and UNet, with EfficientNet backbone, are trained and fused to provide segmentation maps of ancient terraces and walls. The proposed method won the first prize in the International AI Archaeology Challenge. Codes are available at https://github.com/wangyi111/international-archaeology-ai-challenge.

{{</citation>}}


### (45/82) Balanced Classification: A Unified Framework for Long-Tailed Object Detection (Tianhao Qi et al., 2023)

{{<citation>}}

Tianhao Qi, Hongtao Xie, Pandeng Li, Jiannan Ge, Yongdong Zhang. (2023)  
**Balanced Classification: A Unified Framework for Long-Tailed Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.02213v1)  

---


**ABSTRACT**  
Conventional detectors suffer from performance degradation when dealing with long-tailed data due to a classification bias towards the majority head categories. In this paper, we contend that the learning bias originates from two factors: 1) the unequal competition arising from the imbalanced distribution of foreground categories, and 2) the lack of sample diversity in tail categories. To tackle these issues, we introduce a unified framework called BAlanced CLassification (BACL), which enables adaptive rectification of inequalities caused by disparities in category distribution and dynamic intensification of sample diversities in a synchronized manner. Specifically, a novel foreground classification balance loss (FCBL) is developed to ameliorate the domination of head categories and shift attention to difficult-to-differentiate categories by introducing pairwise class-aware margins and auto-adjusted weight terms, respectively. This loss prevents the over-suppression of tail categories in the context of unequal competition. Moreover, we propose a dynamic feature hallucination module (FHM), which enhances the representation of tail categories in the feature space by synthesizing hallucinated samples to introduce additional data variances. In this divide-and-conquer approach, BACL sets a new state-of-the-art on the challenging LVIS benchmark with a decoupled training pipeline, surpassing vanilla Faster R-CNN with ResNet-50-FPN by 5.8% AP and 16.1% AP for overall and tail categories. Extensive experiments demonstrate that BACL consistently achieves performance improvements across various datasets with different backbones and architectures. Code and models are available at https://github.com/Tianhao-Qi/BACL.

{{</citation>}}


### (46/82) Scene-aware Human Pose Generation using Transformer (Jieteng Yao et al., 2023)

{{<citation>}}

Jieteng Yao, Junjie Chen, Li Niu, Bin Sheng. (2023)  
**Scene-aware Human Pose Generation using Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.02177v1)  

---


**ABSTRACT**  
Affordance learning considers the interaction opportunities for an actor in the scene and thus has wide application in scene understanding and intelligent robotics. In this paper, we focus on contextual affordance learning, i.e., using affordance as context to generate a reasonable human pose in a scene. Existing scene-aware human pose generation methods could be divided into two categories depending on whether using pose templates. Our proposed method belongs to the template-based category, which benefits from the representative pose templates. Moreover, inspired by recent transformer-based methods, we associate each query embedding with a pose template, and use the interaction between query embeddings and scene feature map to effectively predict the scale and offsets for each pose template. In addition, we employ knowledge distillation to facilitate the offset learning given the predicted scale. Comprehensive experiments on Sitcom dataset demonstrate the effectiveness of our method.

{{</citation>}}


### (47/82) Efficient Labelling of Affective Video Datasets via Few-Shot & Multi-Task Contrastive Learning (Ravikiran Parameshwara et al., 2023)

{{<citation>}}

Ravikiran Parameshwara, Ibrahim Radwan, Akshay Asthana, Iman Abbasnejad, Ramanathan Subramanian, Roland Goecke. (2023)  
**Efficient Labelling of Affective Video Datasets via Few-Shot & Multi-Task Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-HC, cs-MM, cs.CV  
Keywords: Contrastive Learning, Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.02173v1)  

---


**ABSTRACT**  
Whilst deep learning techniques have achieved excellent emotion prediction, they still require large amounts of labelled training data, which are (a) onerous and tedious to compile, and (b) prone to errors and biases. We propose Multi-Task Contrastive Learning for Affect Representation (\textbf{MT-CLAR}) for few-shot affect inference. MT-CLAR combines multi-task learning with a Siamese network trained via contrastive learning to infer from a pair of expressive facial images (a) the (dis)similarity between the facial expressions, and (b) the difference in valence and arousal levels of the two faces. We further extend the image-based MT-CLAR framework for automated video labelling where, given one or a few labelled video frames (termed \textit{support-set}), MT-CLAR labels the remainder of the video for valence and arousal. Experiments are performed on the AFEW-VA dataset with multiple support-set configurations; moreover, supervised learning on representations learnt via MT-CLAR are used for valence, arousal and categorical emotion prediction on the AffectNet and AFEW-VA datasets. The results show that valence and arousal predictions via MT-CLAR are very comparable to the state-of-the-art (SOTA), and we significantly outperform SOTA with a support-set $\approx$6\% the size of the video dataset.

{{</citation>}}


### (48/82) M2Former: Multi-Scale Patch Selection for Fine-Grained Visual Recognition (Jiyong Moon et al., 2023)

{{<citation>}}

Jiyong Moon, Junseok Lee, Yunju Lee, Seongsik Park. (2023)  
**M2Former: Multi-Scale Patch Selection for Fine-Grained Visual Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.02161v1)  

---


**ABSTRACT**  
Recently, vision Transformers (ViTs) have been actively applied to fine-grained visual recognition (FGVR). ViT can effectively model the interdependencies between patch-divided object regions through an inherent self-attention mechanism. In addition, patch selection is used with ViT to remove redundant patch information and highlight the most discriminative object patches. However, existing ViT-based FGVR models are limited to single-scale processing, and their fixed receptive fields hinder representational richness and exacerbate vulnerability to scale variability. Therefore, we propose multi-scale patch selection (MSPS) to improve the multi-scale capabilities of existing ViT-based models. Specifically, MSPS selects salient patches of different scales at different stages of a multi-scale vision Transformer (MS-ViT). In addition, we introduce class token transfer (CTT) and multi-scale cross-attention (MSCA) to model cross-scale interactions between selected multi-scale patches and fully reflect them in model decisions. Compared to previous single-scale patch selection (SSPS), our proposed MSPS encourages richer object representations based on feature hierarchy and consistently improves performance from small-sized to large-sized objects. As a result, we propose M2Former, which outperforms CNN-/ViT-based models on several widely used FGVR benchmarks.

{{</citation>}}


### (49/82) Robust Self-Supervised Extrinsic Self-Calibration (Takayuki Kanai et al., 2023)

{{<citation>}}

Takayuki Kanai, Igor Vasiljevic, Vitor Guizilini, Adrien Gaidon, Rares Ambrus. (2023)  
**Robust Self-Supervised Extrinsic Self-Calibration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.02153v2)  

---


**ABSTRACT**  
Autonomous vehicles and robots need to operate over a wide variety of scenarios in order to complete tasks efficiently and safely. Multi-camera self-supervised monocular depth estimation from videos is a promising way to reason about the environment, as it generates metrically scaled geometric predictions from visual data without requiring additional sensors. However, most works assume well-calibrated extrinsics to fully leverage this multi-camera setup, even though accurate and efficient calibration is still a challenging problem. In this work, we introduce a novel method for extrinsic calibration that builds upon the principles of self-supervised monocular depth and ego-motion learning. Our proposed curriculum learning strategy uses monocular depth and pose estimators with velocity supervision to estimate extrinsics, and then jointly learns extrinsic calibration along with depth and pose for a set of overlapping cameras rigidly attached to a moving vehicle. Experiments on a benchmark multi-camera dataset (DDAD) demonstrate that our method enables self-calibration in various scenes robustly and efficiently compared to a traditional vision-based pose estimation pipeline. Furthermore, we demonstrate the benefits of extrinsics self-calibration as a way to improve depth prediction via joint optimization.

{{</citation>}}


### (50/82) Attention-Driven Lightweight Model for Pigmented Skin Lesion Detection (Mingzhe Hu et al., 2023)

{{<citation>}}

Mingzhe Hu, Xiaofeng Yang. (2023)  
**Attention-Driven Lightweight Model for Pigmented Skin Lesion Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.02119v1)  

---


**ABSTRACT**  
This study presents a lightweight pipeline for skin lesion detection, addressing the challenges posed by imbalanced class distribution and subtle or atypical appearances of some lesions. The pipeline is built around a lightweight model that leverages ghosted features and the DFC attention mechanism to reduce computational complexity while maintaining high performance. The model was trained on the HAM10000 dataset, which includes various types of skin lesions. To address the class imbalance in the dataset, the synthetic minority over-sampling technique and various image augmentation techniques were used. The model also incorporates a knowledge-based loss weighting technique, which assigns different weights to the loss function at the class level and the instance level, helping the model focus on minority classes and challenging samples. This technique involves assigning different weights to the loss function on two levels - the class level and the instance level. By applying appropriate loss weights, the model pays more attention to the minority classes and challenging samples, thus improving its ability to correctly detect and classify different skin lesions. The model achieved an accuracy of 92.4%, a precision of 84.2%, a recall of 86.9%, a f1-score of 85.4% with particularly strong performance in identifying Benign Keratosis-like lesions (BKL) and Nevus (NV). Despite its superior performance, the model's computational cost is considerably lower than some models with less accuracy, making it an optimal solution for real-world applications where both accuracy and efficiency are essential.

{{</citation>}}


## cs.CR (5)



### (51/82) Creating Android Malware Knowledge Graph Based on a Malware Ontology (Ahmed Sabbah et al., 2023)

{{<citation>}}

Ahmed Sabbah, Mohammed Kharma, Mustafa Jarrar. (2023)  
**Creating Android Malware Knowledge Graph Based on a Malware Ontology**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.02640v1)  

---


**ABSTRACT**  
As mobile and smart connectivity continue to grow, malware presents a permanently evolving threat to different types of critical domains such as health, logistics, banking, and community segments. Different types of malware have dynamic behaviors and complicated characteristics that are shared among members of the same malware family. Malware threat intelligence reports play a crucial role in describing and documenting the detected malware, providing a wealth of information regarding its attributes, patterns, and behaviors. There is a large amount of intelligent threat information regarding malware. The ontology allows the systematic organization and categorization of this information to ensure consistency in representing concepts and entities across various sources. In this study, we reviewed and extended an existing malware ontology to cover Android malware. Our extended ontology is called AndMalOnt. It consisted of 13 new classes, 16 object properties, and 31 data properties. Second, we created an Android malware knowledge graph by extracting reports from the MalwareBazaar repository and representing them in AndMalOnt. This involved generating a knowledge graph that encompasses over 2600 malware samples. Our ontology, knowledge graph, and source code are all open-source and accessible via GitHub

{{</citation>}}


### (52/82) Vulnerabilities in AI Code Generators: Exploring Targeted Data Poisoning Attacks (Domenico Cotroneo et al., 2023)

{{<citation>}}

Domenico Cotroneo, Cristina Improta, Pietro Liguori, Roberto Natella. (2023)  
**Vulnerabilities in AI Code Generators: Exploring Targeted Data Poisoning Attacks**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.04451v1)  

---


**ABSTRACT**  
In this work, we assess the security of AI code generators via data poisoning, i.e., an attack that injects malicious samples into the training data to generate vulnerable code. We poison the training data by injecting increasing amounts of code containing security vulnerabilities and assess the attack's success on different state-of-the-art models for code generation. Our analysis shows that AI code generators are vulnerable to even a small amount of data poisoning. Moreover, the attack does not impact the correctness of code generated by pre-trained models, making it hard to detect.

{{</citation>}}


### (53/82) Improving the Security of United States Elections with Robust Optimization (Braden L. Crimmins et al., 2023)

{{<citation>}}

Braden L. Crimmins, J. Alex Halderman, Bradley Sturt. (2023)  
**Improving the Security of United States Elections with Robust Optimization**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR, math-OC  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.02306v1)  

---


**ABSTRACT**  
For more than a century, election officials across the United States have inspected voting machines before elections using a procedure called Logic and Accuracy Testing (LAT). This procedure consists of election officials casting a test deck of ballots into each voting machine and confirming the machine produces the expected vote total for each candidate. We bring a scientific perspective to LAT by introducing the first formal approach to designing test decks with rigorous security guarantees. Specifically, our approach employs robust optimization to find test decks that are guaranteed to detect any voting machine misconfiguration that would cause votes to be swapped across candidates. Out of all the test decks with this security guarantee, our robust optimization problem yields the test deck with the minimum number of ballots, thereby minimizing implementation costs for election officials. To facilitate deployment at scale, we develop a practically efficient exact algorithm for solving our robust optimization problems based on the cutting plane method. In partnership with the Michigan Bureau of Elections, we retrospectively applied our approach to all 6928 ballot styles from Michigan's November 2022 general election; this retrospective study reveals that the test decks with rigorous security guarantees obtained by our approach require, on average, only 1.2% more ballots than current practice. Our approach has since been piloted in real-world elections by the Michigan Bureau of Elections as a low-cost way to improve election security and increase public trust in democratic institutions.

{{</citation>}}


### (54/82) Security Evaluation of Compressible and Learnable Image Encryption Against Jigsaw Puzzle Solver Attacks (Tatsuya Chuman et al., 2023)

{{<citation>}}

Tatsuya Chuman, Nobutaka Ono, Hitoshi Kiya. (2023)  
**Security Evaluation of Compressible and Learnable Image Encryption Against Jigsaw Puzzle Solver Attacks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.02227v1)  

---


**ABSTRACT**  
Several learnable image encryption schemes have been developed for privacy-preserving image classification. This paper focuses on the security block-based image encryption methods that are learnable and JPEG-friendly. Permuting divided blocks in an image is known to enhance robustness against ciphertext-only attacks (COAs), but recently jigsaw puzzle solver attacks have been demonstrated to be able to restore visual information on the encrypted images. In contrast, it has never been confirmed whether encrypted images including noise caused by JPEG-compression are robust. Accordingly, the aim of this paper is to evaluate the security of compressible and learnable encrypted images against jigsaw puzzle solver attacks. In experiments, the security evaluation was carried out on the CIFAR-10 and STL-10 datasets under JPEG-compression.

{{</citation>}}


### (55/82) ParaFuzz: An Interpretability-Driven Technique for Detecting Poisoned Samples in NLP (Lu Yan et al., 2023)

{{<citation>}}

Lu Yan, Zhuo Zhang, Guanhong Tao, Kaiyuan Zhang, Xuan Chen, Guangyu Shen, Xiangyu Zhang. (2023)  
**ParaFuzz: An Interpretability-Driven Technique for Detecting Poisoned Samples in NLP**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs.CR  
Keywords: ChatGPT, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2308.02122v1)  

---


**ABSTRACT**  
Backdoor attacks have emerged as a prominent threat to natural language processing (NLP) models, where the presence of specific triggers in the input can lead poisoned models to misclassify these inputs to predetermined target classes. Current detection mechanisms are limited by their inability to address more covert backdoor strategies, such as style-based attacks. In this work, we propose an innovative test-time poisoned sample detection framework that hinges on the interpretability of model predictions, grounded in the semantic meaning of inputs. We contend that triggers (e.g., infrequent words) are not supposed to fundamentally alter the underlying semantic meanings of poisoned samples as they want to stay stealthy. Based on this observation, we hypothesize that while the model's predictions for paraphrased clean samples should remain stable, predictions for poisoned samples should revert to their true labels upon the mutations applied to triggers during the paraphrasing process. We employ ChatGPT, a state-of-the-art large language model, as our paraphraser and formulate the trigger-removal task as a prompt engineering problem. We adopt fuzzing, a technique commonly used for unearthing software vulnerabilities, to discover optimal paraphrase prompts that can effectively eliminate triggers while concurrently maintaining input semantics. Experiments on 4 types of backdoor attacks, including the subtle style backdoors, and 4 distinct datasets demonstrate that our approach surpasses baseline methods, including STRIP, RAP, and ONION, in precision and recall.

{{</citation>}}


## cs.AI (2)



### (56/82) A Survey on Temporal Knowledge Graph Completion: Taxonomy, Progress, and Prospects (Jiapu Wang et al., 2023)

{{<citation>}}

Jiapu Wang, Boyue Wang, Meikang Qiu, Shirui Pan, Bo Xiong, Heng Liu, Linhao Luo, Tengfei Liu, Yongli Hu, Baocai Yin, Wen Gao. (2023)  
**A Survey on Temporal Knowledge Graph Completion: Taxonomy, Progress, and Prospects**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.02457v1)  

---


**ABSTRACT**  
Temporal characteristics are prominently evident in a substantial volume of knowledge, which underscores the pivotal role of Temporal Knowledge Graphs (TKGs) in both academia and industry. However, TKGs often suffer from incompleteness for three main reasons: the continuous emergence of new knowledge, the weakness of the algorithm for extracting structured information from unstructured data, and the lack of information in the source dataset. Thus, the task of Temporal Knowledge Graph Completion (TKGC) has attracted increasing attention, aiming to predict missing items based on the available information. In this paper, we provide a comprehensive review of TKGC methods and their details. Specifically, this paper mainly consists of three components, namely, 1)Background, which covers the preliminaries of TKGC methods, loss functions required for training, as well as the dataset and evaluation protocol; 2)Interpolation, that estimates and predicts the missing elements or set of elements through the relevant available information. It further categorizes related TKGC methods based on how to process temporal information; 3)Extrapolation, which typically focuses on continuous TKGs and predicts future events, and then classifies all extrapolation methods based on the algorithms they utilize. We further pinpoint the challenges and discuss future research directions of TKGC.

{{</citation>}}


### (57/82) Unravelling Responsibility for AI (Zoe Porter et al., 2023)

{{<citation>}}

Zoe Porter, Joanna Al-Qaddoumi, Philippa Ryan Conmy, Phillip Morgan, John McDermid, Ibrahim Habli. (2023)  
**Unravelling Responsibility for AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-RO, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02608v1)  

---


**ABSTRACT**  
To reason about where responsibility does and should lie in complex situations involving AI-enabled systems, we first need a sufficiently clear and detailed cross-disciplinary vocabulary for talking about responsibility. Responsibility is a triadic relation involving an actor, an occurrence, and a way of being responsible. As part of a conscious effort towards 'unravelling' the concept of responsibility to support practical reasoning about responsibility for AI, this paper takes the three-part formulation, 'Actor A is responsible for Occurrence O' and identifies valid combinations of subcategories of A, is responsible for, and O. These valid combinations - which we term "responsibility strings" - are grouped into four senses of responsibility: role-responsibility; causal responsibility; legal liability-responsibility; and moral responsibility. They are illustrated with two running examples, one involving a healthcare AI-based system and another the fatal collision of an AV with a pedestrian in Tempe, Arizona in 2018. The output of the paper is 81 responsibility strings. The aim is that these strings provide the vocabulary for people across disciplines to be clear and specific about the different ways that different actors are responsible for different occurrences within a complex event for which responsibility is sought, allowing for precise and targeted interdisciplinary normative deliberations.

{{</citation>}}


## stat.ML (2)



### (58/82) Generative Modelling of Lévy Area for High Order SDE Simulation (Andraž Jelinčič et al., 2023)

{{<citation>}}

Andraž Jelinčič, Jiajie Tao, William F. Turner, Thomas Cass, James Foster, Hao Ni. (2023)  
**Generative Modelling of Lévy Area for High Order SDE Simulation**  

---
Primary Category: stat.ML  
Categories: 65C30, cs-LG, cs-NA, math-NA, math-PR, stat-ML, stat.ML  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.02452v1)  

---


**ABSTRACT**  
It is well known that, when numerically simulating solutions to SDEs, achieving a strong convergence rate better than O(\sqrt{h}) (where h is the step size) requires the use of certain iterated integrals of Brownian motion, commonly referred to as its "L\'{e}vy areas". However, these stochastic integrals are difficult to simulate due to their non-Gaussian nature and for a d-dimensional Brownian motion with d > 2, no fast almost-exact sampling algorithm is known.   In this paper, we propose L\'{e}vyGAN, a deep-learning-based model for generating approximate samples of L\'{e}vy area conditional on a Brownian increment. Due to our "Bridge-flipping" operation, the output samples match all joint and conditional odd moments exactly. Our generator employs a tailored GNN-inspired architecture, which enforces the correct dependency structure between the output distribution and the conditioning variable. Furthermore, we incorporate a mathematically principled characteristic-function based discriminator. Lastly, we introduce a novel training mechanism termed "Chen-training", which circumvents the need for expensive-to-generate training data-sets. This new training procedure is underpinned by our two main theoretical results.   For 4-dimensional Brownian motion, we show that L\'{e}vyGAN exhibits state-of-the-art performance across several metrics which measure both the joint and marginal distributions. We conclude with a numerical experiment on the log-Heston model, a popular SDE in mathematical finance, demonstrating that high-quality synthetic L\'{e}vy area can lead to high order weak convergence and variance reduction when using multilevel Monte Carlo (MLMC).

{{</citation>}}


### (59/82) Pruning a neural network using Bayesian inference (Sunil Mathew et al., 2023)

{{<citation>}}

Sunil Mathew, Daniel B. Rowe. (2023)  
**Pruning a neural network using Bayesian inference**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2308.02451v1)  

---


**ABSTRACT**  
Neural network pruning is a highly effective technique aimed at reducing the computational and memory demands of large neural networks. In this research paper, we present a novel approach to pruning neural networks utilizing Bayesian inference, which can seamlessly integrate into the training procedure. Our proposed method leverages the posterior probabilities of the neural network prior to and following pruning, enabling the calculation of Bayes factors. The calculated Bayes factors guide the iterative pruning. Through comprehensive evaluations conducted on multiple benchmarks, we demonstrate that our method achieves desired levels of sparsity while maintaining competitive accuracy.

{{</citation>}}


## cs.CY (3)



### (60/82) From Military to Healthcare: Adopting and Expanding Ethical Principles for Generative Artificial Intelligence (David Oniani et al., 2023)

{{<citation>}}

David Oniani, Jordan Hilsman, Yifan Peng, COL, Ronald K. Poropatich, COL Jeremy C. Pamplin, LTC Gary L. Legault, Yanshan Wang. (2023)  
**From Military to Healthcare: Adopting and Expanding Ethical Principles for Generative Artificial Intelligence**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2308.02448v1)  

---


**ABSTRACT**  
In 2020, the U.S. Department of Defense officially disclosed a set of ethical principles to guide the use of Artificial Intelligence (AI) technologies on future battlefields. Despite stark differences, there are core similarities between the military and medical service. Warriors on battlefields often face life-altering circumstances that require quick decision-making. Medical providers experience similar challenges in a rapidly changing healthcare environment, such as in the emergency department or during surgery treating a life-threatening condition. Generative AI, an emerging technology designed to efficiently generate valuable information, holds great promise. As computing power becomes more accessible and the abundance of health data, such as electronic health records, electrocardiograms, and medical images, increases, it is inevitable that healthcare will be revolutionized by this technology. Recently, generative AI has captivated the research community, leading to debates about its application in healthcare, mainly due to concerns about transparency and related issues. Meanwhile, concerns about the potential exacerbation of health disparities due to modeling biases have raised notable ethical concerns regarding the use of this technology in healthcare. However, the ethical principles for generative AI in healthcare have been understudied, and decision-makers often fail to consider the significance of generative AI. In this paper, we propose GREAT PLEA ethical principles, encompassing governance, reliability, equity, accountability, traceability, privacy, lawfulness, empathy, and autonomy, for generative AI in healthcare. We aim to proactively address the ethical dilemmas and challenges posed by the integration of generative AI in healthcare.

{{</citation>}}


### (61/82) AI exposure predicts unemployment risk (Morgan Frank et al., 2023)

{{<citation>}}

Morgan Frank, Yong-Yeol Ahn, Esteban Moro. (2023)  
**AI exposure predicts unemployment risk**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY, econ-GN, q-fin-EC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02624v1)  

---


**ABSTRACT**  
Is artificial intelligence (AI) disrupting jobs and creating unemployment? Despite many attempts to quantify occupations' exposure to AI, inconsistent validation obfuscates the relative benefits of each approach. A lack of disaggregated labor outcome data, including unemployment data, further exacerbates the issue. Here, we assess which models of AI exposure predict job separations and unemployment risk using new occupation-level unemployment data by occupation from each US state's unemployment insurance office spanning 2010 through 2020. Although these AI exposure scores have been used by governments and industry, we find that individual AI exposure models are not predictive of unemployment rates, unemployment risk, or job separation rates. However, an ensemble of those models exhibits substantial predictive power suggesting that competing models may capture different aspects of AI exposure that collectively account for AI's variable impact across occupations, regions, and time. Our results also call for dynamic, context-aware, and validated methods for assessing AI exposure. Interactive visualizations for this study are available at https://sites.pitt.edu/~mrfrank/uiRiskDemo/.

{{</citation>}}


### (62/82) Federated Learning: Organizational Opportunities, Challenges, and Adoption Strategies (Joaquin Delgado Fernandez et al., 2023)

{{<citation>}}

Joaquin Delgado Fernandez, Martin Brennecke, Tom Barbereau, Alexander Rieger, Gilbert Fridgen. (2023)  
**Federated Learning: Organizational Opportunities, Challenges, and Adoption Strategies**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-SI, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02219v1)  

---


**ABSTRACT**  
Restrictive rules for data sharing in many industries have led to the development of \ac{FL}. \ac{FL} is a \ac{ML} technique that allows distributed clients to train models collaboratively without the need to share their respective training data with others. In this article, we first explore the technical basics of FL and its potential applications. Second, we present a conceptual framework for the adoption of \ac{FL}, mapping organizations along the lines of their \ac{AI} capabilities and environment. We then discuss why exemplary organizations in different industries, including industry consortia, established banks, public authorities, and data-intensive SMEs might consider different approaches to \ac{FL}. To conclude, we argue that \ac{FL} presents an institutional shift with ample interdisciplinary research opportunities for the business and information systems engineering community.

{{</citation>}}


## eess.IV (3)



### (63/82) Brain MRI Segmentation using Template-Based Training and Visual Perception Augmentation (Fang-Cheng Yeh, 2023)

{{<citation>}}

Fang-Cheng Yeh. (2023)  
**Brain MRI Segmentation using Template-Based Training and Visual Perception Augmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV, q-bio-QM  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.02363v1)  

---


**ABSTRACT**  
Deep learning models usually require sufficient training data to achieve high accuracy, but obtaining labeled data can be time-consuming and labor-intensive. Here we introduce a template-based training method to train a 3D U-Net model from scratch using only one population-averaged brain MRI template and its associated segmentation label. The process incorporated visual perception augmentation to enhance the model's robustness in handling diverse image inputs and mitigating overfitting. Leveraging this approach, we trained 3D U-Net models for mouse, rat, marmoset, rhesus, and human brain MRI to achieve segmentation tasks such as skull-stripping, brain segmentation, and tissue probability mapping. This tool effectively addresses the limited availability of training data and holds significant potential for expanding deep learning applications in image analysis, providing researchers with a unified solution to train deep neural networks with only one image sample.

{{</citation>}}


### (64/82) Designing a Deep Learning-Driven Resource-Efficient Diagnostic System for Metastatic Breast Cancer: Reducing Long Delays of Clinical Diagnosis and Improving Patient Survival in Developing Countries (William Gao et al., 2023)

{{<citation>}}

William Gao, Dayong Wang, Yi Huang. (2023)  
**Designing a Deep Learning-Driven Resource-Efficient Diagnostic System for Metastatic Breast Cancer: Reducing Long Delays of Clinical Diagnosis and Improving Patient Survival in Developing Countries**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI, Clinical  
[Paper Link](http://arxiv.org/abs/2308.02597v1)  

---


**ABSTRACT**  
Breast cancer is one of the leading causes of cancer mortality. Breast cancer patients in developing countries, especially sub-Saharan Africa, South Asia, and South America, suffer from the highest mortality rate in the world. One crucial factor contributing to the global disparity in mortality rate is long delay of diagnosis due to a severe shortage of trained pathologists, which consequently has led to a large proportion of late-stage presentation at diagnosis. The delay between the initial development of symptoms and the receipt of a diagnosis could stretch upwards 15 months. To tackle this critical healthcare disparity, this research has developed a deep learning-based diagnosis system for metastatic breast cancer that can achieve high diagnostic accuracy as well as computational efficiency. Based on our evaluation, the MobileNetV2-based diagnostic model outperformed the more complex VGG16, ResNet50 and ResNet101 models in diagnostic accuracy, model generalization, and model training efficiency. The visual comparisons between the model prediction and ground truth have demonstrated that the MobileNetV2 diagnostic models can identify very small cancerous nodes embedded in a large area of normal cells which is challenging for manual image analysis. Equally Important, the light weighted MobleNetV2 models were computationally efficient and ready for mobile devices or devices of low computational power. These advances empower the development of a resource-efficient and high performing AI-based metastatic breast cancer diagnostic system that can adapt to under-resourced healthcare facilities in developing countries. This research provides an innovative technological solution to address the long delays in metastatic breast cancer diagnosis and the consequent disparity in patient survival outcome in developing countries.

{{</citation>}}


### (65/82) Breast Ultrasound Tumor Classification Using a Hybrid Multitask CNN-Transformer Network (Bryar Shareef et al., 2023)

{{<citation>}}

Bryar Shareef, Min Xian, Aleksandar Vakanski, Haotian Wang. (2023)  
**Breast Ultrasound Tumor Classification Using a Hybrid Multitask CNN-Transformer Network**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.02101v1)  

---


**ABSTRACT**  
Capturing global contextual information plays a critical role in breast ultrasound (BUS) image classification. Although convolutional neural networks (CNNs) have demonstrated reliable performance in tumor classification, they have inherent limitations for modeling global and long-range dependencies due to the localized nature of convolution operations. Vision Transformers have an improved capability of capturing global contextual information but may distort the local image patterns due to the tokenization operations. In this study, we proposed a hybrid multitask deep neural network called Hybrid-MT-ESTAN, designed to perform BUS tumor classification and segmentation using a hybrid architecture composed of CNNs and Swin Transformer components. The proposed approach was compared to nine BUS classification methods and evaluated using seven quantitative metrics on a dataset of 3,320 BUS images. The results indicate that Hybrid-MT-ESTAN achieved the highest accuracy, sensitivity, and F1 score of 82.7%, 86.4%, and 86.0%, respectively.

{{</citation>}}


## cs.IR (2)



### (66/82) ChatGPT for GTFS: From Words to Information (Saipraneeth Devunuri et al., 2023)

{{<citation>}}

Saipraneeth Devunuri, Shirin Qiam, Lewis Lehe. (2023)  
**ChatGPT for GTFS: From Words to Information**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: ChatGPT, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2308.02618v1)  

---


**ABSTRACT**  
The General Transit Feed Specification (GTFS) standard for publishing transit data is ubiquitous. GTFS being tabular data, with information spread across different files, necessitates specialized tools or packages to retrieve information. Concurrently, the use of Large Language Models for text and information retrieval is growing. The idea of this research is to see if the current widely adopted LLMs (ChatGPT) are able to retrieve information from GTFS using natural language instructions. We first test whether ChatGPT (GPT-3.5) understands the GTFS specification. GPT-3.5 answers 77% of our multiple-choice questions (MCQ) correctly. Next, we task the LLM with information extractions from a filtered GTFS feed with 4 routes. For information retrieval, we compare zero-shot and program synthesis. Program synthesis works better, achieving ~90% accuracy on simple questions and ~40% accuracy on complex questions.

{{</citation>}}


### (67/82) Towards Personalized Prompt-Model Retrieval for Generative Recommendation (Yuanhe Guo et al., 2023)

{{<citation>}}

Yuanhe Guo, Haoming Liu, Hongyi Wen. (2023)  
**Towards Personalized Prompt-Model Retrieval for Generative Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: AI, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2308.02205v1)  

---


**ABSTRACT**  
Recommender Systems are built to retrieve relevant items to satisfy users' information needs. The candidate corpus usually consists of a finite set of items that are ready to be served, such as videos, products, or articles. With recent advances in Generative AI such as GPT and Diffusion models, a new form of recommendation task is yet to be explored where items are to be created by generative models with personalized prompts. Taking image generation as an example, with a single prompt from the user and access to a generative model, it is possible to generate hundreds of new images in a few minutes. How shall we attain personalization in the presence of "infinite" items? In this preliminary study, we propose a two-stage framework, namely Prompt-Model Retrieval and Generated Item Ranking, to approach this new task formulation. We release GEMRec-18K, a prompt-model interaction dataset with 18K images generated by 200 publicly-available generative models paired with a diverse set of 90 textual prompts. Our findings demonstrate the promise of generative model recommendation as a novel personalization problem and the limitations of existing evaluation metrics. We highlight future directions for the RecSys community to advance towards generative recommender systems. Our code and dataset are available at https://github.com/MAPS-research/GEMRec.

{{</citation>}}


## cs.DC (1)



### (68/82) A Deep Dive into the Google Cluster Workload Traces: Analyzing the Application Failure Characteristics and User Behaviors (Faisal Haque Bappy et al., 2023)

{{<citation>}}

Faisal Haque Bappy, Tariqul Islam, Tarannum Shaila Zaman, Raiful Hasan, Carlos Caicedo. (2023)  
**A Deep Dive into the Google Cluster Workload Traces: Analyzing the Application Failure Characteristics and User Behaviors**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.02358v1)  

---


**ABSTRACT**  
Large-scale cloud data centers have gained popularity due to their high availability, rapid elasticity, scalability, and low cost. However, current data centers continue to have high failure rates due to the lack of proper resource utilization and early failure detection. To maximize resource efficiency and reduce failure rates in large-scale cloud data centers, it is crucial to understand the workload and failure characteristics. In this paper, we perform a deep analysis of the 2019 Google Cluster Trace Dataset, which contains 2.4TiB of workload traces from eight different clusters around the world. We explore the characteristics of failed and killed jobs in Google's production cloud and attempt to correlate them with key attributes such as resource usage, job priority, scheduling class, job duration, and the number of task resubmissions. Our analysis reveals several important characteristics of failed jobs that contribute to job failure and hence, could be used for developing an early failure prediction system. Also, we present a novel usage analysis to identify heterogeneity in jobs and tasks submitted by users. We are able to identify specific users who control more than half of all collection events on a single cluster. We contend that these characteristics could be useful in developing an early job failure prediction system that could be utilized for dynamic rescheduling of the job scheduler and thus improving resource utilization in large-scale cloud data centers while reducing failure rates.

{{</citation>}}


## eess.SY (2)



### (69/82) Communication-Efficient Decentralized Multi-Agent Reinforcement Learning for Cooperative Adaptive Cruise Control (Dong Chen et al., 2023)

{{<citation>}}

Dong Chen, Kaixiang Zhang, Yongqiang Wang, Xunyuan Yin, Zhaojian Li. (2023)  
**Communication-Efficient Decentralized Multi-Agent Reinforcement Learning for Cooperative Adaptive Cruise Control**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.02345v2)  

---


**ABSTRACT**  
Connected and autonomous vehicles (CAVs) promise next-gen transportation systems with enhanced safety, energy efficiency, and sustainability. One typical control strategy for CAVs is the so-called cooperative adaptive cruise control (CACC) where vehicles drive in platoons and cooperate to achieve safe and efficient transportation. In this study, we formulate CACC as a multi-agent reinforcement learning (MARL) problem. Diverging from existing MARL methods that use centralized training and decentralized execution which require not only a centralized communication mechanism but also dense inter-agent communication, we propose a fully-decentralized MARL framework for enhanced efficiency and scalability. In addition, a quantization-based communication scheme is proposed to reduce the communication overhead without significantly degrading the control performance. This is achieved by employing randomized rounding numbers to quantize each piece of communicated information and only communicating non-zero components after quantization. Extensive experimentation in two distinct CACC settings reveals that the proposed MARL framework consistently achieves superior performance over several contemporary benchmarks in terms of both communication efficiency and control efficacy.

{{</citation>}}


### (70/82) Transformer-Based Denoising of Mechanical Vibration Signals (Han Chen et al., 2023)

{{<citation>}}

Han Chen, Yang Yu, Pengtao Li. (2023)  
**Transformer-Based Denoising of Mechanical Vibration Signals**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SP, eess-SY, eess.SY  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.02166v1)  

---


**ABSTRACT**  
Mechanical vibration signal denoising is a pivotal task in various industrial applications, including system health monitoring and failure prediction. This paper introduces a novel deep learning transformer-based architecture specifically tailored for denoising mechanical vibration signals. The model leverages a Multi-Head Attention layer with 8 heads, processing input sequences of length 128, embedded into a 64-dimensional space. The architecture also incorporates Feed-Forward Neural Networks, Layer Normalization, and Residual Connections, resulting in enhanced recognition and extraction of essential features. Through a training process guided by the Mean Squared Error loss function and optimized using the Adam optimizer, the model demonstrates remarkable effectiveness in filtering out noise while preserving critical information related to mechanical vibrations. The specific design and choice of parameters offer a robust method adaptable to the complex nature of mechanical systems, with promising applications in industrial monitoring and maintenance. This work lays the groundwork for future exploration and optimization in the field of mechanical signal analysis and presents a significant step towards advanced and intelligent mechanical system diagnostics.

{{</citation>}}


## quant-ph (1)



### (71/82) Evidence of Scaling Advantage for the Quantum Approximate Optimization Algorithm on a Classically Intractable Problem (Ruslan Shaydulin et al., 2023)

{{<citation>}}

Ruslan Shaydulin, Changhao Li, Shouvanik Chakrabarti, Matthew DeCross, Dylan Herman, Niraj Kumar, Jeffrey Larson, Danylo Lykov, Pierre Minssen, Yue Sun, Yuri Alexeev, Joan M. Dreiling, John P. Gaebler, Thomas M. Gatterman, Justin A. Gerber, Kevin Gilmore, Dan Gresh, Nathan Hewitt, Chandler V. Horst, Shaohan Hu, Jacob Johansen, Mitchell Matheny, Tanner Mengle, Michael Mills, Steven A. Moses, Brian Neyenhuis, Peter Siegfried, Romina Yalovetzky, Marco Pistoia. (2023)  
**Evidence of Scaling Advantage for the Quantum Approximate Optimization Algorithm on a Classically Intractable Problem**  

---
Primary Category: quant-ph  
Categories: cond-mat-stat-mech, cs-ET, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.02342v1)  

---


**ABSTRACT**  
The quantum approximate optimization algorithm (QAOA) is a leading candidate algorithm for solving optimization problems on quantum computers. However, the potential of QAOA to tackle classically intractable problems remains unclear. In this paper, we perform an extensive numerical investigation of QAOA on the Low Autocorrelation Binary Sequences (LABS) problem. The rapid growth of the problem's complexity with the number of spins $N$ makes it classically intractable even for moderately sized instances, with the best-known heuristics observed to fail to find a good solution for problems with $N \gtrapprox 200$. We perform noiseless simulations with up to 40 qubits and observe that out to this system size, the runtime of QAOA with fixed parameters and a constant number of layers scales better than branch-and-bound solvers, which are the state-of-the-art exact solvers for LABS. The combination of QAOA with quantum minimum-finding on an idealized quantum computer gives the best empirical scaling of any algorithm for the LABS problem. We demonstrate experimental progress in compiling and executing QAOA for the LABS problem using an algorithm-specific error detection scheme on Quantinuum trapped-ion processors. Our results provide evidence for the utility of QAOA as an algorithmic component when executed on an idealized quantum computer.

{{</citation>}}


## cs.SE (1)



### (72/82) Who Answers It Better? An In-Depth Analysis of ChatGPT and Stack Overflow Answers to Software Engineering Questions (Samia Kabir et al., 2023)

{{<citation>}}

Samia Kabir, David N. Udo-Imeh, Bonan Kou, Tianyi Zhang. (2023)  
**Who Answers It Better? An In-Depth Analysis of ChatGPT and Stack Overflow Answers to Software Engineering Questions**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.02312v2)  

---


**ABSTRACT**  
Q&A platforms have been an integral part of the web-help-seeking behavior of programmers over the past decade. However, with the recent introduction of ChatGPT, the paradigm of web-help-seeking behavior is experiencing a shift. Despite the popularity of ChatGPT, no comprehensive study has been conducted to evaluate the characteristics or usability of ChatGPT's answers to software engineering questions. To bridge the gap, we conducted the first in-depth analysis of ChatGPT's answers to 517 Stack Overflow (SO) questions and examined the correctness, consistency, comprehensiveness, and conciseness of ChatGPT's answers. Furthermore, we conducted a large-scale linguistic analysis, and a user study to understand the characteristics of ChatGPT answers from linguistic and human aspects. Our analysis shows that 52% of ChatGPT answers are incorrect and 77% are verbose. Nonetheless, ChatGPT answers are still preferred 39.34% of the time due to their comprehensiveness and well-articulated language style. Our result implies the necessity of close examination and rectification of errors in ChatGPT, at the same time creating awareness among its users of the risks associated with seemingly correct ChatGPT answers.

{{</citation>}}


## math.NA (1)



### (73/82) Krylov Subspace Recycling With Randomized Sketching For Matrix Functions (Liam Burke et al., 2023)

{{<citation>}}

Liam Burke, Stefan Güttel. (2023)  
**Krylov Subspace Recycling With Randomized Sketching For Matrix Functions**  

---
Primary Category: math.NA  
Categories: 65F60, 65F50, 65F10, 68W20, cs-NA, math-NA, math.NA  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2308.02290v1)  

---


**ABSTRACT**  
A Krylov subspace recycling method for the efficient evaluation of a sequence of matrix functions acting on a set of vectors is developed. The method improves over the recycling methods presented in [Burke et al., arXiv:2209.14163, 2022] in that it uses a closed-form expression for the augmented FOM approximants and hence circumvents the use of numerical quadrature. We further extend our method to use randomized sketching in order to avoid the arithmetic cost of orthogonalizing a full Krylov basis, offering an attractive solution to the fact that recycling algorithms built from shifted augmented FOM cannot easily be restarted. The efficacy of the proposed algorithms is demonstrated with numerical experiments.

{{</citation>}}


## cs.SD (3)



### (74/82) Efficient Monaural Speech Enhancement using Spectrum Attention Fusion (Jinyu Long et al., 2023)

{{<citation>}}

Jinyu Long, Jetic Gū, Binhao Bai, Zhibo Yang, Ping Wei, Junli Li. (2023)  
**Efficient Monaural Speech Enhancement using Spectrum Attention Fusion**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.02263v1)  

---


**ABSTRACT**  
Speech enhancement is a demanding task in automated speech processing pipelines, focusing on separating clean speech from noisy channels. Transformer based models have recently bested RNN and CNN models in speech enhancement, however at the same time they are much more computationally expensive and require much more high quality training data, which is always hard to come by. In this paper, we present an improvement for speech enhancement models that maintains the expressiveness of self-attention while significantly reducing model complexity, which we have termed Spectrum Attention Fusion. We carefully construct a convolutional module to replace several self-attention layers in a speech Transformer, allowing the model to more efficiently fuse spectral features. Our proposed model is able to achieve comparable or better results against SOTA models but with significantly smaller parameters (0.58M) on the Voice Bank + DEMAND dataset.

{{</citation>}}


### (75/82) Emo-DNA: Emotion Decoupling and Alignment Learning for Cross-Corpus Speech Emotion Recognition (Jiaxin Ye et al., 2023)

{{<citation>}}

Jiaxin Ye, Yujie Wei, Xin-Cheng Wen, Chenglong Ma, Zhizhong Huang, Kunhong Liu, Hongming Shan. (2023)  
**Emo-DNA: Emotion Decoupling and Alignment Learning for Cross-Corpus Speech Emotion Recognition**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2308.02190v1)  

---


**ABSTRACT**  
Cross-corpus speech emotion recognition (SER) seeks to generalize the ability of inferring speech emotion from a well-labeled corpus to an unlabeled one, which is a rather challenging task due to the significant discrepancy between two corpora. Existing methods, typically based on unsupervised domain adaptation (UDA), struggle to learn corpus-invariant features by global distribution alignment, but unfortunately, the resulting features are mixed with corpus-specific features or not class-discriminative. To tackle these challenges, we propose a novel Emotion Decoupling aNd Alignment learning framework (EMO-DNA) for cross-corpus SER, a novel UDA method to learn emotion-relevant corpus-invariant features. The novelties of EMO-DNA are two-fold: contrastive emotion decoupling and dual-level emotion alignment. On one hand, our contrastive emotion decoupling achieves decoupling learning via a contrastive decoupling loss to strengthen the separability of emotion-relevant features from corpus-specific ones. On the other hand, our dual-level emotion alignment introduces an adaptive threshold pseudo-labeling to select confident target samples for class-level alignment, and performs corpus-level alignment to jointly guide model for learning class-discriminative corpus-invariant features across corpora. Extensive experimental results demonstrate the superior performance of EMO-DNA over the state-of-the-art methods in several cross-corpus scenarios. Source code is available at https://github.com/Jiaxin-Ye/Emo-DNA.

{{</citation>}}


### (76/82) Capturing Spectral and Long-term Contextual Information for Speech Emotion Recognition Using Deep Learning Techniques (Samiul Islam et al., 2023)

{{<citation>}}

Samiul Islam, Md. Maksudul Haque, Abu Jobayer Md. Sadat. (2023)  
**Capturing Spectral and Long-term Contextual Information for Speech Emotion Recognition Using Deep Learning Techniques**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: BERT, Emotion Recognition, Graph Convolutional Network, LSTM  
[Paper Link](http://arxiv.org/abs/2308.04517v1)  

---


**ABSTRACT**  
Traditional approaches in speech emotion recognition, such as LSTM, CNN, RNN, SVM, and MLP, have limitations such as difficulty capturing long-term dependencies in sequential data, capturing the temporal dynamics, and struggling to capture complex patterns and relationships in multimodal data. This research addresses these shortcomings by proposing an ensemble model that combines Graph Convolutional Networks (GCN) for processing textual data and the HuBERT transformer for analyzing audio signals. We found that GCNs excel at capturing Long-term contextual dependencies and relationships within textual data by leveraging graph-based representations of text and thus detecting the contextual meaning and semantic relationships between words. On the other hand, HuBERT utilizes self-attention mechanisms to capture long-range dependencies, enabling the modeling of temporal dynamics present in speech and capturing subtle nuances and variations that contribute to emotion recognition. By combining GCN and HuBERT, our ensemble model can leverage the strengths of both approaches. This allows for the simultaneous analysis of multimodal data, and the fusion of these modalities enables the extraction of complementary information, enhancing the discriminative power of the emotion recognition system. The results indicate that the combined model can overcome the limitations of traditional methods, leading to enhanced accuracy in recognizing emotions from speech.

{{</citation>}}


## cs.NI (2)



### (77/82) IntLearner: AI-enabled Interference Mitigation for Wireless Networks (Ruirong Chen et al., 2023)

{{<citation>}}

Ruirong Chen, Gaoning He. (2023)  
**IntLearner: AI-enabled Interference Mitigation for Wireless Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02167v1)  

---


**ABSTRACT**  
The future Six-Generation (6G) envisions massive access of wireless devices in the network, leading to more serious interference from concurrent transmissions between wireless devices in the same frequency band. Existing interference mitigation approaches takes the interference signals as Gaussian white noise, which cannot precisely estimate the non-Gaussian interference signals from other devices. In this paper, we present IntLearner, a new interference mitigation technique that estimates and mitigates the impact of interference signals with only physical-layer (PHY) information available in base-station (BS) and user-equipment (UE), including channel estimator and constellations. More specifically, IntLearner utilizes the power of AI to estimate the features in interference signals, and removes the interference from the interfered received signal with neural network (NN). IntLearner's NN adopts a modular NN design, which takes the domain knowledge of BS and UE PHY as the guidance to NN design for minimizing training confusion and NN complexity. Simulation results show IntLearner increases Uplink (UL) channel estimation accuracy up to 7.4x, and reduces the Downlink (DL) Signal to Interference Ratio plus Noise Ratio (SINR) requirement to achieve the same Block Error Rate (BLER) by 1.5dB in a conventional multi-cell scenario.

{{</citation>}}


### (78/82) Artificial intelligence based load balancing in SDN: A comprehensive survey (Ahmed Hazim Alhilali et al., 2023)

{{<citation>}}

Ahmed Hazim Alhilali, Ahmadreza Montazerolghaem. (2023)  
**Artificial intelligence based load balancing in SDN: A comprehensive survey**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02149v1)  

---


**ABSTRACT**  
In the future, it is anticipated that software-defined networking (SDN) will become the preferred platform for deploying diverse networks. Compared to traditional networks, SDN separates the control and data planes for efficient domain-wide traffic routing and management. The controllers in the control plane are responsible for programming data plane forwarding devices, while the top layer, the application plane, enforces policies and programs the network. The different levels of the SDN use interfaces for communication. However, SDN faces challenges with traffic distribution, such as load imbalance, which can negatively affect the network performance. Consequently, developers have developed various SDN load-balancing solutions to enhance SDN effectiveness. In addition, researchers are considering the potential of implementing some artificial intelligence (AI) approaches into SDN to improve network resource usage and overall performance due to the fast growth of the AI field. This survey focuses on the following: Firstly, analyzing the SDN architecture and investigating the problem of load balancing in SDN. Secondly, categorizing AI-based load balancing methods and thoroughly assessing these mechanisms from various perspectives, such as the algorithm/technique employed, the tackled problem, and their strengths and weaknesses. Thirdly, summarizing the metrics utilized to measure the effectiveness of these techniques. Finally, identifying the trends and challenges of AI-based load balancing for future research.

{{</citation>}}


## cs.IT (3)



### (79/82) Deep Reinforcement Learning Empowered Rate Selection of XP-HARQ (Da Wu et al., 2023)

{{<citation>}}

Da Wu, Jiahui Feng, Zheng Shi, Hongjiang Lei, Guanghua Yang, Shaodan Ma. (2023)  
**Deep Reinforcement Learning Empowered Rate Selection of XP-HARQ**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.02140v1)  

---


**ABSTRACT**  
The complex transmission mechanism of cross-packet hybrid automatic repeat request (XP-HARQ) hinders its optimal system design. To overcome this difficulty, this letter attempts to use the deep reinforcement learning (DRL) to solve the rate selection problem of XP-HARQ over correlated fading channels. In particular, the long term average throughput (LTAT) is maximized by properly choosing the incremental information rate for each HARQ round on the basis of the outdated channel state information (CSI) available at the transmitter. The rate selection problem is first converted into a Markov decision process (MDP), which is then solved by capitalizing on the algorithm of deep deterministic policy gradient (DDPG) with prioritized experience replay. The simulation results finally corroborate the superiority of the proposed XP-HARQ scheme over the conventional HARQ with incremental redundancy (HARQ-IR) and the XP-HARQ with only statistical CSI.

{{</citation>}}


### (80/82) A Survey of Beam Management for mmWave and THz Communications Towards 6G (Qing Xue et al., 2023)

{{<citation>}}

Qing Xue, Chengwang Ji, Shaodan Ma, Jiajia Guo, Yongjun Xu, Qianbin Chen, Wei Zhang. (2023)  
**A Survey of Beam Management for mmWave and THz Communications Towards 6G**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-SY, cs.IT, eess-SY, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02135v1)  

---


**ABSTRACT**  
Communication in millimeter wave (mmWave) and even terahertz (THz) frequency bands is ushering in a new era of wireless communications. Beam management, namely initial access and beam tracking, has been recognized as an essential technique to ensure robust mmWave/THz communications, especially for mobile scenarios. However, narrow beams at higher carrier frequency lead to huge beam measurement overhead, which has a negative impact on beam acquisition and tracking. In addition, the beam management process is further complicated by the fluctuation of mmWave/THz channels, the random movement patterns of users, and the dynamic changes in the environment. For mmWave and THz communications toward 6G, we have witnessed a substantial increase in research and industrial attention on artificial intelligence (AI), reconfigurable intelligent surface (RIS), and integrated sensing and communications (ISAC). The introduction of these enabling technologies presents both open opportunities and unique challenges for beam management. In this paper, we present a comprehensive survey on mmWave and THz beam management. Further, we give some insights on technical challenges and future research directions in this promising area.

{{</citation>}}


### (81/82) Graph Convolutional Network Enabled Power-Constrained HARQ Strategy for URLLC (Yi Chen et al., 2023)

{{<citation>}}

Yi Chen, Zheng Shi, Hong Wang, Yaru Fu, Guanghua Yang, Shaodan Ma, Haichuan Ding. (2023)  
**Graph Convolutional Network Enabled Power-Constrained HARQ Strategy for URLLC**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2308.02131v1)  

---


**ABSTRACT**  
In this paper, a power-constrained hybrid automatic repeat request (HARQ) transmission strategy is developed to support ultra-reliable low-latency communications (URLLC). In particular, we aim to minimize the delivery latency of HARQ schemes over time-correlated fading channels, meanwhile ensuring the high reliability and limited power consumption. To ease the optimization, the simple asymptotic outage expressions of HARQ schemes are adopted. Furthermore, by noticing the non-convexity of the latency minimization problem and the intricate connection between different HARQ rounds, the graph convolutional network (GCN) is invoked for the optimal power solution owing to its powerful ability of handling the graph data. The primal-dual learning method is then leveraged to train the GCN weights. Consequently, the numerical results are presented for verification together with the comparisons among three HARQ schemes in terms of the latency and the reliability, where the three HARQ schemes include Type-I HARQ, HARQ with chase combining (HARQ-CC), and HARQ with incremental redundancy (HARQ-IR). To recapitulate, it is revealed that HARQ-IR offers the lowest latency while guaranteeing the demanded reliability target under a stringent power constraint, albeit at the price of high coding complexity.

{{</citation>}}


## cs.HC (1)



### (82/82) Avatar Fusion Karaoke: Research and development on multi-user music play VR experience in the metaverse (Alexandre Berthault et al., 2023)

{{<citation>}}

Alexandre Berthault, Takuma Kato, Akihiko Shirai. (2023)  
**Avatar Fusion Karaoke: Research and development on multi-user music play VR experience in the metaverse**  

---
Primary Category: cs.HC  
Categories: I-2; I-3, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02139v1)  

---


**ABSTRACT**  
This paper contributes to building a standard process of research and development (R&D) for new user experiences (UX) in metaverse services. We tested this R&D process on a new UX proof of concept (PoC) for Meta Quest head-mounted display (HMDs) consisting of a school-life karaoke experience with the hypothesis that it is possible to design the avatars with only the necessary functions and rendering costs. The school life metaverse is a relevant subject for discovering issues and problems in this type of simultaneous connection. To qualitatively evaluate the potential of a multi-person metaverse experience, this study investigated subjects where each avatar requires expressive skills. While avatar play experiences feature artistic expressions, such as dancing, playing musical instruments, and drawing, and these can be used to evaluate their operability and expressive capabilities qualitatively, the Quest's tracking capabilities are insufficient for full-body performance and graphical art expression. Considering such hardware limitations, this study evaluated the Quest, focusing primarily on UX simplicity using AI Fusion techniques and expressiveness in instrumental scenes played by approximately four avatars. This research reported methods for multiuser metaverse communication and its supporting technologies, such as head-mounted devices and their graphics performance, special interaction techniques, and complementary tools and the importance of PoC development, its evaluation, and its iterations. The result is remarkable for further research; these expressive technologies in a multi-user context are directly related to the quality of communication within the metaverse and the value of the user-generated content (UGC) produced there.

{{</citation>}}
