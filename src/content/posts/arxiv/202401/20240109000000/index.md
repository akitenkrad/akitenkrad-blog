---
draft: false
title: "arXiv @ 2024.01.09"
date: 2024-01-09
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.09"
    identifier: arxiv_20240109
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.IR (1)](#csir-1)
- [cs.CL (12)](#cscl-12)
- [cs.LG (7)](#cslg-7)
- [cs.CV (8)](#cscv-8)
- [cs.AI (7)](#csai-7)
- [cs.CR (2)](#cscr-2)
- [cs.RO (1)](#csro-1)
- [cs.DL (1)](#csdl-1)
- [cs.SI (1)](#cssi-1)
- [eess.AS (3)](#eessas-3)
- [cs.CY (3)](#cscy-3)
- [cs.SD (2)](#cssd-2)
- [cs.HC (1)](#cshc-1)
- [cs.CE (1)](#csce-1)
- [q-bio.PE (1)](#q-biope-1)
- [cs.SE (1)](#csse-1)
- [q-bio.MN (1)](#q-biomn-1)

## cs.IR (1)



### (1/53) ChatGPT for Conversational Recommendation: Refining Recommendations by Reprompting with Feedback (Kyle Dylan Spurlock et al., 2024)

{{<citation>}}

Kyle Dylan Spurlock, Cagla Acun, Esin Saka, Olfa Nasraoui. (2024)  
**ChatGPT for Conversational Recommendation: Refining Recommendations by Reprompting with Feedback**  

---
Primary Category: cs.IR  
Categories: I-2-7; H-3-3, cs-AI, cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: ChatGPT, Conversational Recommendation, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03605v1)  

---


**ABSTRACT**  
Recommendation algorithms have been pivotal in handling the overwhelming volume of online content. However, these algorithms seldom consider direct user input, resulting in superficial interaction between them. Efforts have been made to include the user directly in the recommendation process through conversation, but these systems too have had limited interactivity. Recently, Large Language Models (LLMs) like ChatGPT have gained popularity due to their ease of use and their ability to adapt dynamically to various tasks while responding to feedback. In this paper, we investigate the effectiveness of ChatGPT as a top-n conversational recommendation system. We build a rigorous pipeline around ChatGPT to simulate how a user might realistically probe the model for recommendations: by first instructing and then reprompting with feedback to refine a set of recommendations. We further explore the effect of popularity bias in ChatGPT's recommendations, and compare its performance to baseline models. We find that reprompting ChatGPT with feedback is an effective strategy to improve recommendation relevancy, and that popularity bias can be mitigated through prompt engineering.

{{</citation>}}


## cs.CL (12)



### (2/53) InFoBench: Evaluating Instruction Following Ability in Large Language Models (Yiwei Qin et al., 2024)

{{<citation>}}

Yiwei Qin, Kaiqiang Song, Yebowen Hu, Wenlin Yao, Sangwoo Cho, Xiaoyang Wang, Xuansheng Wu, Fei Liu, Pengfei Liu, Dong Yu. (2024)  
**InFoBench: Evaluating Instruction Following Ability in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03601v1)  

---


**ABSTRACT**  
This paper introduces the Decomposed Requirements Following Ratio (DRFR), a new metric for evaluating Large Language Models' (LLMs) ability to follow instructions. Addressing a gap in current methodologies, DRFR breaks down complex instructions into simpler criteria, facilitating a detailed analysis of LLMs' compliance with various aspects of tasks. Alongside this metric, we present InFoBench, a benchmark comprising 500 diverse instructions and 2,250 decomposed questions across multiple constraint categories. Our experiments compare DRFR with traditional scoring methods and explore annotation sources, including human experts, crowd-sourced workers, and GPT-4. The findings demonstrate DRFR's higher reliability and the effectiveness of using GPT-4 as a cost-efficient annotator. The evaluation of several advanced LLMs using this framework reveals their strengths and areas needing improvement, particularly in complex instruction-following. This study contributes a novel metric and benchmark, offering insights for future LLM development and evaluation.

{{</citation>}}


### (3/53) Text Classification Based on Knowledge Graphs and Improved Attention Mechanism (Siyu Li et al., 2024)

{{<citation>}}

Siyu Li, Lu Chen, Chenwei Song, Xinyi Liu. (2024)  
**Text Classification Based on Knowledge Graphs and Improved Attention Mechanism**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Knowledge Graph, Text Classification  
[Paper Link](http://arxiv.org/abs/2401.03591v1)  

---


**ABSTRACT**  
To resolve the semantic ambiguity in texts, we propose a model, which innovatively combines a knowledge graph with an improved attention mechanism. An existing knowledge base is utilized to enrich the text with relevant contextual concepts. The model operates at both character and word levels to deepen its understanding by integrating the concepts. We first adopt information gain to select import words. Then an encoder-decoder framework is used to encode the text along with the related concepts. The local attention mechanism adjusts the weight of each concept, reducing the influence of irrelevant or noisy concepts during classification. We improve the calculation formula for attention scores in the local self-attention mechanism, ensuring that words with different frequencies of occurrence in the text receive higher attention scores. Finally, the model employs a Bi-directional Gated Recurrent Unit (Bi-GRU), which is effective in feature extraction from texts for improved classification accuracy. Its performance is demonstrated on datasets such as AGNews, Ohsumed, and TagMyNews, achieving accuracy of 75.1%, 58.7%, and 68.5% respectively, showing its effectiveness in classifying tasks.

{{</citation>}}


### (4/53) Building Efficient and Effective OpenQA Systems for Low-Resource Languages (Emrah Budur et al., 2024)

{{<citation>}}

Emrah Budur, Rıza Özçelik, Dilara Soylu, Omar Khattab, Tunga Güngör, Christopher Potts. (2024)  
**Building Efficient and Effective OpenQA Systems for Low-Resource Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Low-Resource, QA  
[Paper Link](http://arxiv.org/abs/2401.03590v1)  

---


**ABSTRACT**  
Question answering (QA) is the task of answering questions posed in natural language with free-form natural language answers extracted from a given passage. In the OpenQA variant, only a question text is given, and the system must retrieve relevant passages from an unstructured knowledge source and use them to provide answers, which is the case in the mainstream QA systems on the Web. QA systems currently are mostly limited to the English language due to the lack of large-scale labeled QA datasets in non-English languages. In this paper, we show that effective, low-cost OpenQA systems can be developed for low-resource languages. The key ingredients are (1) weak supervision using machine-translated labeled datasets and (2) a relevant unstructured knowledge source in the target language. Furthermore, we show that only a few hundred gold assessment examples are needed to reliably evaluate these systems. We apply our method to Turkish as a challenging case study, since English and Turkish are typologically very distinct. We present SQuAD-TR, a machine translation of SQuAD2.0, and we build our OpenQA system by adapting ColBERT-QA for Turkish. We obtain a performance improvement of 9-34% in the EM score and 13-33% in the F1 score compared to the BM25-based and DPR-based baseline QA reader models by using two versions of Wikipedia dumps spanning two years. Our results show that SQuAD-TR makes OpenQA feasible for Turkish, which we hope encourages researchers to build OpenQA systems in other low-resource languages. We make all the code, models, and the dataset publicly available.

{{</citation>}}


### (5/53) Data-CUBE: Data Curriculum for Instruction-based Sentence Representation Learning (Yingqian Min et al., 2024)

{{<citation>}}

Yingqian Min, Kun Zhou, Dawei Gao, Wayne Xin Zhao, He Hu, Yaliang Li. (2024)  
**Data-CUBE: Data Curriculum for Instruction-based Sentence Representation Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: AI, Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.03563v1)  

---


**ABSTRACT**  
Recently, multi-task instruction tuning has been applied into sentence representation learning, which endows the capability of generating specific representations with the guidance of task instruction, exhibiting strong generalization ability on new tasks. However, these methods mostly neglect the potential interference problems across different tasks and instances, which may affect the training and convergence of the model. To address it, we propose a data curriculum method, namely Data-CUBE, that arranges the orders of all the multi-task data for training, to minimize the interference risks from the two views. In the task level, we aim to find the optimal task order to minimize the total cross-task interference risk, which is exactly the traveling salesman problem, hence we utilize a simulated annealing algorithm to find its solution. In the instance level, we measure the difficulty of all instances per task, then divide them into the easy-to-difficult mini-batches for training. Experiments on MTEB sentence representation evaluation tasks show that our approach can boost the performance of state-of-the-art methods. Our code and data are publicly available at the link: \url{https://github.com/RUCAIBox/Data-CUBE}.

{{</citation>}}


### (6/53) CAPTAIN at COLIEE 2023: Efficient Methods for Legal Information Retrieval and Entailment Tasks (Chau Nguyen et al., 2024)

{{<citation>}}

Chau Nguyen, Phuong Nguyen, Thanh Tran, Dat Nguyen, An Trieu, Tin Pham, Anh Dang, Le-Minh Nguyen. (2024)  
**CAPTAIN at COLIEE 2023: Efficient Methods for Legal Information Retrieval and Entailment Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: AI, Information Extraction, Information Retrieval, Legal  
[Paper Link](http://arxiv.org/abs/2401.03551v1)  

---


**ABSTRACT**  
The Competition on Legal Information Extraction/Entailment (COLIEE) is held annually to encourage advancements in the automatic processing of legal texts. Processing legal documents is challenging due to the intricate structure and meaning of legal language. In this paper, we outline our strategies for tackling Task 2, Task 3, and Task 4 in the COLIEE 2023 competition. Our approach involved utilizing appropriate state-of-the-art deep learning methods, designing methods based on domain characteristics observation, and applying meticulous engineering practices and methodologies to the competition. As a result, our performance in these tasks has been outstanding, with first places in Task 2 and Task 3, and promising results in Task 4. Our source code is available at https://github.com/Nguyen2015/CAPTAIN-COLIEE2023/tree/coliee2023.

{{</citation>}}


### (7/53) RoBERTurk: Adjusting RoBERTa for Turkish (Nuri Tas, 2024)

{{<citation>}}

Nuri Tas. (2024)  
**RoBERTurk: Adjusting RoBERTa for Turkish**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, NER  
[Paper Link](http://arxiv.org/abs/2401.03515v1)  

---


**ABSTRACT**  
We pretrain RoBERTa on a Turkish corpora using BPE tokenizer. Our model outperforms BERTurk family models on the BOUN dataset for the POS task while resulting in underperformance on the IMST dataset for the same task and achieving competitive scores on the Turkish split of the XTREME dataset for the NER task - all while being pretrained on smaller data than its competitors. We release our pretrained model and tokenizer.

{{</citation>}}


### (8/53) Token-free LLMs Can Generate Chinese Classical Poetry with More Accurate Format (Chengyue Yu et al., 2024)

{{<citation>}}

Chengyue Yu, Lei Zang, Jiaotuan Wang, Chenyi Zhuang, Jinjie Gu. (2024)  
**Token-free LLMs Can Generate Chinese Classical Poetry with More Accurate Format**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.03512v1)  

---


**ABSTRACT**  
Finetuned large language models (such as ChatGPT and Qwen-chat) can generate Chinese classical poetry following human's instructions. LLMs perform well in content, but are usually lacking in format, with occasionally excess or insufficient number of characters in each line. Since most SOTA LLMs are token-based, we assume that the format inaccuracy is due to the difficulty of the "token planning" task, which means that the LLM need to know exactly how much characters are contained in each token and do length-control planning based on that knowledge. In this paper, we first confirm our assumption by showing that existing token-based large language models has limited knowledge on token-character relationship. We use a spelling bee probing procedure, and find that Qwen-chat failed in nearly 15% Chinese spelling test. We then show that a token-based model can be easily tailored into a token-free model (in terms of Chinese), which can largely solve the format accuracy problem. Our tailoring procedure removes long-token from vocabulary and keeps only character-level or byte-level tokens. As part of our contribution, we release the finetuned token-free model (which is based on Qwen-chat-7B), which can generate chinese classical poetry following complex instructions like LLMs (such as story paraphrasing), and also perform well in format. On the test set, our token-free model achives an format accuracy of 0.96, compared to 0.84 for token-based counterparts and 0.38 for GPT-4.

{{</citation>}}


### (9/53) Maintaining Journalistic Integrity in the Digital Age: A Comprehensive NLP Framework for Evaluating Online News Content (Ljubisa Bojic et al., 2024)

{{<citation>}}

Ljubisa Bojic, Nikola Prodanovic, Agariadne Dwinggo Samala. (2024)  
**Maintaining Journalistic Integrity in the Digital Age: A Comprehensive NLP Framework for Evaluating Online News Content**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2401.03467v1)  

---


**ABSTRACT**  
The rapid growth of online news platforms has led to an increased need for reliable methods to evaluate the quality and credibility of news articles. This paper proposes a comprehensive framework to analyze online news texts using natural language processing (NLP) techniques, particularly a language model specifically trained for this purpose, alongside other well-established NLP methods. The framework incorporates ten journalism standards-objectivity, balance and fairness, readability and clarity, sensationalism and clickbait, ethical considerations, public interest and value, source credibility, relevance and timeliness, factual accuracy, and attribution and transparency-to assess the quality of news articles. By establishing these standards, researchers, media organizations, and readers can better evaluate and understand the content they consume and produce. The proposed method has some limitations, such as potential difficulty in detecting subtle biases and the need for continuous updating of the language model to keep pace with evolving language patterns.

{{</citation>}}


### (10/53) On Leveraging Large Language Models for Enhancing Entity Resolution (Huahang Li et al., 2024)

{{<citation>}}

Huahang Li, Longyu Feng, Shuangyin Li, Fei Hao, Chen Jason Zhang, Yuanfeng Song, Lei Chen. (2024)  
**On Leveraging Large Language Models for Enhancing Entity Resolution**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03426v1)  

---


**ABSTRACT**  
Entity resolution, the task of identifying and consolidating records that pertain to the same real-world entity, plays a pivotal role in various sectors such as e-commerce, healthcare, and law enforcement. The emergence of Large Language Models (LLMs) like GPT-4 has introduced a new dimension to this task, leveraging their advanced linguistic capabilities. This paper explores the potential of LLMs in the entity resolution process, shedding light on both their advantages and the computational complexities associated with large-scale matching. We introduce strategies for the efficient utilization of LLMs, including the selection of an optimal set of matching questions, namely MQsSP, which is proved to be a NP-hard problem. Our approach optimally chooses the most effective matching questions while keep consumption limited to your budget . Additionally, we propose a method to adjust the distribution of possible partitions after receiving responses from LLMs, with the goal of reducing the uncertainty of entity resolution. We evaluate the effectiveness of our approach using entropy as a metric, and our experimental results demonstrate the efficiency and effectiveness of our proposed methods, offering promising prospects for real-world applications.

{{</citation>}}


### (11/53) GRAM: Global Reasoning for Multi-Page VQA (Tsachi Blau et al., 2024)

{{<citation>}}

Tsachi Blau, Sharon Fogel, Roi Ronen, Alona Golts, Roy Ganz, Elad Ben Avraham, Aviad Aberdam, Shahar Tsiper, Ron Litman. (2024)  
**GRAM: Global Reasoning for Multi-Page VQA**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.03411v1)  

---


**ABSTRACT**  
The increasing use of transformer-based large language models brings forward the challenge of processing long sequences. In document visual question answering (DocVQA), leading methods focus on the single-page setting, while documents can span hundreds of pages. We present GRAM, a method that seamlessly extends pre-trained single-page models to the multi-page setting, without requiring computationally-heavy pretraining. To do so, we leverage a single-page encoder for local page-level understanding, and enhance it with document-level designated layers and learnable tokens, facilitating the flow of information across pages for global reasoning. To enforce our model to utilize the newly introduced document-level tokens, we propose a tailored bias adaptation method. For additional computational savings during decoding, we introduce an optional compression stage using our C-Former model, which reduces the encoded sequence length, thereby allowing a tradeoff between quality and latency. Extensive experiments showcase GRAM's state-of-the-art performance on the benchmarks for multi-page DocVQA, demonstrating the effectiveness of our approach.

{{</citation>}}


### (12/53) Empirical Study of Large Language Models as Automated Essay Scoring Tools in English Composition__Taking TOEFL Independent Writing Task for Example (Wei Xia et al., 2024)

{{<citation>}}

Wei Xia, Shaoguang Mao, Chanjing Zheng. (2024)  
**Empirical Study of Large Language Models as Automated Essay Scoring Tools in English Composition__Taking TOEFL Independent Writing Task for Example**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03401v1)  

---


**ABSTRACT**  
Large language models have demonstrated exceptional capabilities in tasks involving natural language generation, reasoning, and comprehension. This study aims to construct prompts and comments grounded in the diverse scoring criteria delineated within the official TOEFL guide. The primary objective is to assess the capabilities and constraints of ChatGPT, a prominent representative of large language models, within the context of automated essay scoring. The prevailing methodologies for automated essay scoring involve the utilization of deep neural networks, statistical machine learning techniques, and fine-tuning pre-trained models. However, these techniques face challenges when applied to different contexts or subjects, primarily due to their substantial data requirements and limited adaptability to small sample sizes. In contrast, this study employs ChatGPT to conduct an automated evaluation of English essays, even with a small sample size, employing an experimental approach. The empirical findings indicate that ChatGPT can provide operational functionality for automated essay scoring, although the results exhibit a regression effect. It is imperative to underscore that the effective design and implementation of ChatGPT prompts necessitate a profound domain expertise and technical proficiency, as these prompts are subject to specific threshold criteria. Keywords: ChatGPT, Automated Essay Scoring, Prompt Learning, TOEFL Independent Writing Task

{{</citation>}}


### (13/53) Grimoire is All You Need for Enhancing Large Language Models (Ding Chen et al., 2024)

{{<citation>}}

Ding Chen, Shichao Song, Qingchen Yu, Zhiyu Li, Wenjin Wang, Feiyu Xiong, Bo Tang. (2024)  
**Grimoire is All You Need for Enhancing Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03385v1)  

---


**ABSTRACT**  
In-context learning (ICL) is one of the key methods for enhancing the performance of large language models on specific tasks by providing a set of few-shot question and answer examples. However, the ICL capability of different types of models shows significant variation due to factors such as model architecture, volume of learning data, and the size of parameters. Generally, the larger the model's parameter size and the more extensive the learning data, the stronger its ICL capability. In this paper, we propose a method SLEICL (Strong LLM Enhanced ICL) that involves learning from examples using strong language models and then summarizing and transferring these learned skills to weak language models for inference and application. This ensures the stability and effectiveness of ICL. Compared to directly enabling weak language models to learn from prompt examples, SLEICL reduces the difficulty of ICL for these models. Our experiments, conducted on up to eight datasets with five language models, demonstrate that weak language models achieve consistent improvement over their own zero-shot or few-shot capabilities using the SLEICL method. Some weak language models even surpass the performance of GPT4-1106-preview (zero-shot) with the aid of SLEICL.

{{</citation>}}


## cs.LG (7)



### (14/53) Few-Shot Causal Representation Learning for Out-of-Distribution Generalization on Heterogeneous Graphs (Pengfei Ding et al., 2024)

{{<citation>}}

Pengfei Ding, Yan Wang, Guanfeng Liu, Nan Wang. (2024)  
**Few-Shot Causal Representation Learning for Out-of-Distribution Generalization on Heterogeneous Graphs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Few-Shot, Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.03597v1)  

---


**ABSTRACT**  
Heterogeneous graph few-shot learning (HGFL) has been developed to address the label sparsity issue in heterogeneous graphs (HGs), which consist of various types of nodes and edges. The core concept of HGFL is to extract knowledge from rich-labeled classes in a source HG, transfer this knowledge to a target HG to facilitate learning new classes with few-labeled training data, and finally make predictions on unlabeled testing data. Existing methods typically assume that the source HG, training data, and testing data all share the same distribution. However, in practice, distribution shifts among these three types of data are inevitable due to two reasons: (1) the limited availability of the source HG that matches the target HG distribution, and (2) the unpredictable data generation mechanism of the target HG. Such distribution shifts result in ineffective knowledge transfer and poor learning performance in existing methods, thereby leading to a novel problem of out-of-distribution (OOD) generalization in HGFL. To address this challenging problem, we propose a novel Causal OOD Heterogeneous graph Few-shot learning model, namely COHF. In COHF, we first characterize distribution shifts in HGs with a structural causal model, establishing an invariance principle for OOD generalization in HGFL. Then, following this invariance principle, we propose a new variational autoencoder-based heterogeneous graph neural network to mitigate the impact of distribution shifts. Finally, by integrating this network with a novel meta-learning framework, COHF effectively transfers knowledge to the target HG to predict new classes with few-labeled data. Extensive experiments on seven real-world datasets have demonstrated the superior performance of COHF over the state-of-the-art methods.

{{</citation>}}


### (15/53) GLOCALFAIR: Jointly Improving Global and Local Group Fairness in Federated Learning (Syed Irfan Ali Meerza et al., 2024)

{{<citation>}}

Syed Irfan Ali Meerza, Luyang Liu, Jiaxin Zhang, Jian Liu. (2024)  
**GLOCALFAIR: Jointly Improving Global and Local Group Fairness in Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03562v1)  

---


**ABSTRACT**  
Federated learning (FL) has emerged as a prospective solution for collaboratively learning a shared model across clients without sacrificing their data privacy. However, the federated learned model tends to be biased against certain demographic groups (e.g., racial and gender groups) due to the inherent FL properties, such as data heterogeneity and party selection. Unlike centralized learning, mitigating bias in FL is particularly challenging as private training datasets and their sensitive attributes are typically not directly accessible. Most prior research in this field only focuses on global fairness while overlooking the local fairness of individual clients. Moreover, existing methods often require sensitive information about the client's local datasets to be shared, which is not desirable. To address these issues, we propose GLOCALFAIR, a client-server co-design fairness framework that can jointly improve global and local group fairness in FL without the need for sensitive statistics about the client's private datasets. Specifically, we utilize constrained optimization to enforce local fairness on the client side and adopt a fairness-aware clustering-based aggregation on the server to further ensure the global model fairness across different sensitive groups while maintaining high utility. Experiments on two image datasets and one tabular dataset with various state-of-the-art fairness baselines show that GLOCALFAIR can achieve enhanced fairness under both global and local data distributions while maintaining a good level of utility and client fairness.

{{</citation>}}


### (16/53) Detecting Anomalies in Blockchain Transactions using Machine Learning Classifiers and Explainability Analysis (Mohammad Hasan et al., 2024)

{{<citation>}}

Mohammad Hasan, Mohammad Shahriar Rahman, Helge Janicke, Iqbal H. Sarker. (2024)  
**Detecting Anomalies in Blockchain Transactions using Machine Learning Classifiers and Explainability Analysis**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03530v1)  

---


**ABSTRACT**  
As the use of Blockchain for digital payments continues to rise in popularity, it also becomes susceptible to various malicious attacks. Successfully detecting anomalies within Blockchain transactions is essential for bolstering trust in digital payments. However, the task of anomaly detection in Blockchain transaction data is challenging due to the infrequent occurrence of illicit transactions. Although several studies have been conducted in the field, a limitation persists: the lack of explanations for the model's predictions. This study seeks to overcome this limitation by integrating eXplainable Artificial Intelligence (XAI) techniques and anomaly rules into tree-based ensemble classifiers for detecting anomalous Bitcoin transactions. The Shapley Additive exPlanation (SHAP) method is employed to measure the contribution of each feature, and it is compatible with ensemble models. Moreover, we present rules for interpreting whether a Bitcoin transaction is anomalous or not. Additionally, we have introduced an under-sampling algorithm named XGBCLUS, designed to balance anomalous and non-anomalous transaction data. This algorithm is compared against other commonly used under-sampling and over-sampling techniques. Finally, the outcomes of various tree-based single classifiers are compared with those of stacking and voting ensemble classifiers. Our experimental results demonstrate that: (i) XGBCLUS enhances TPR and ROC-AUC scores compared to state-of-the-art under-sampling and over-sampling techniques, and (ii) our proposed ensemble classifiers outperform traditional single tree-based machine learning classifiers in terms of accuracy, TPR, and FPR scores.

{{</citation>}}


### (17/53) Decentralized Federated Policy Gradient with Byzantine Fault-Tolerance and Provably Fast Convergence (Philip Jordan et al., 2024)

{{<citation>}}

Philip Jordan, Florian Grötschla, Flint Xiaofeng Fan, Roger Wattenhofer. (2024)  
**Decentralized Federated Policy Gradient with Byzantine Fault-Tolerance and Provably Fast Convergence**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.03489v1)  

---


**ABSTRACT**  
In Federated Reinforcement Learning (FRL), agents aim to collaboratively learn a common task, while each agent is acting in its local environment without exchanging raw trajectories. Existing approaches for FRL either (a) do not provide any fault-tolerance guarantees (against misbehaving agents), or (b) rely on a trusted central agent (a single point of failure) for aggregating updates. We provide the first decentralized Byzantine fault-tolerant FRL method. Towards this end, we first propose a new centralized Byzantine fault-tolerant policy gradient (PG) algorithm that improves over existing methods by relying only on assumptions standard for non-fault-tolerant PG. Then, as our main contribution, we show how a combination of robust aggregation and Byzantine-resilient agreement methods can be leveraged in order to eliminate the need for a trusted central entity. Since our results represent the first sample complexity analysis for Byzantine fault-tolerant decentralized federated non-convex optimization, our technical contributions may be of independent interest. Finally, we corroborate our theoretical results experimentally for common RL environments, demonstrating the speed-up of decentralized federations w.r.t. the number of participating agents and resilience against various Byzantine attacks.

{{</citation>}}


### (18/53) Uncertainty Quantification on Clinical Trial Outcome Prediction (Tianyi Chen et al., 2024)

{{<citation>}}

Tianyi Chen, Nan Hao, Yingzhou Lu, Capucine Van Rechem. (2024)  
**Uncertainty Quantification on Clinical Trial Outcome Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2401.03482v1)  

---


**ABSTRACT**  
The importance of uncertainty quantification is increasingly recognized in the diverse field of machine learning. Accurately assessing model prediction uncertainty can help provide deeper understanding and confidence for researchers and practitioners. This is especially critical in medical diagnosis and drug discovery areas, where reliable predictions directly impact research quality and patient health.   In this paper, we proposed incorporating uncertainty quantification into clinical trial outcome predictions. Our main goal is to enhance the model's ability to discern nuanced differences, thereby significantly improving its overall performance.   We have adopted a selective classification approach to fulfill our objective, integrating it seamlessly with the Hierarchical Interaction Network (HINT), which is at the forefront of clinical trial prediction modeling. Selective classification, encompassing a spectrum of methods for uncertainty quantification, empowers the model to withhold decision-making in the face of samples marked by ambiguity or low confidence, thereby amplifying the accuracy of predictions for the instances it chooses to classify. A series of comprehensive experiments demonstrate that incorporating selective classification into clinical trial predictions markedly enhances the model's performance, as evidenced by significant upticks in pivotal metrics such as PR-AUC, F1, ROC-AUC, and overall accuracy.   Specifically, the proposed method achieved 32.37\%, 21.43\%, and 13.27\% relative improvement on PR-AUC over the base model (HINT) in phase I, II, and III trial outcome prediction, respectively. When predicting phase III, our method reaches 0.9022 PR-AUC scores.   These findings illustrate the robustness and prospective utility of this strategy within the area of clinical trial predictions, potentially setting a new benchmark in the field.

{{</citation>}}


### (19/53) Accurate and Scalable Estimation of Epistemic Uncertainty for Graph Neural Networks (Puja Trivedi et al., 2024)

{{<citation>}}

Puja Trivedi, Mark Heimann, Rushil Anirudh, Danai Koutra, Jayaraman J. Thiagarajan. (2024)  
**Accurate and Scalable Estimation of Epistemic Uncertainty for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.03350v1)  

---


**ABSTRACT**  
While graph neural networks (GNNs) are widely used for node and graph representation learning tasks, the reliability of GNN uncertainty estimates under distribution shifts remains relatively under-explored. Indeed, while post-hoc calibration strategies can be used to improve in-distribution calibration, they need not also improve calibration under distribution shift. However, techniques which produce GNNs with better intrinsic uncertainty estimates are particularly valuable, as they can always be combined with post-hoc strategies later. Therefore, in this work, we propose G-$\Delta$UQ, a novel training framework designed to improve intrinsic GNN uncertainty estimates. Our framework adapts the principle of stochastic data centering to graph data through novel graph anchoring strategies, and is able to support partially stochastic GNNs. While, the prevalent wisdom is that fully stochastic networks are necessary to obtain reliable estimates, we find that the functional diversity induced by our anchoring strategies when sampling hypotheses renders this unnecessary and allows us to support G-$\Delta$UQ on pretrained models. Indeed, through extensive evaluation under covariate, concept and graph size shifts, we show that G-$\Delta$UQ leads to better calibrated GNNs for node and graph classification. Further, it also improves performance on the uncertainty-based tasks of out-of-distribution detection and generalization gap estimation. Overall, our work provides insights into uncertainty estimation for GNNs, and demonstrates the utility of G-$\Delta$UQ in obtaining reliable estimates.

{{</citation>}}


### (20/53) Weakly Augmented Variational Autoencoder in Time Series Anomaly Detection (Zhangkai Wu et al., 2024)

{{<citation>}}

Zhangkai Wu, Longbing Cao, Qi Zhang, Junxian Zhou, Hui Chen. (2024)  
**Weakly Augmented Variational Autoencoder in Time Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2401.03341v1)  

---


**ABSTRACT**  
Due to their unsupervised training and uncertainty estimation, deep Variational Autoencoders (VAEs) have become powerful tools for reconstruction-based Time Series Anomaly Detection (TSAD). Existing VAE-based TSAD methods, either statistical or deep, tune meta-priors to estimate the likelihood probability for effectively capturing spatiotemporal dependencies in the data. However, these methods confront the challenge of inherent data scarcity, which is often the case in anomaly detection tasks. Such scarcity easily leads to latent holes, discontinuous regions in latent space, resulting in non-robust reconstructions on these discontinuous spaces. We propose a novel generative framework that combines VAEs with self-supervised learning (SSL) to address this issue.

{{</citation>}}


## cs.CV (8)



### (21/53) Big Data and Deep Learning in Smart Cities: A Comprehensive Dataset for AI-Driven Traffic Accident Detection and Computer Vision Systems (Victor Adewopo et al., 2024)

{{<citation>}}

Victor Adewopo, Nelly Elsayed, Zag Elsayed, Murat Ozer, Constantinos Zekios, Ahmed Abdelgawad, Magdy Bayoumi. (2024)  
**Big Data and Deep Learning in Smart Cities: A Comprehensive Dataset for AI-Driven Traffic Accident Detection and Computer Vision Systems**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Computer Vision  
[Paper Link](http://arxiv.org/abs/2401.03587v1)  

---


**ABSTRACT**  
In the dynamic urban landscape, where the interplay of vehicles and pedestrians defines the rhythm of life, integrating advanced technology for safety and efficiency is increasingly crucial. This study delves into the application of cutting-edge technological methods in smart cities, focusing on enhancing public safety through improved traffic accident detection. Action recognition plays a pivotal role in interpreting visual data and tracking object motion such as human pose estimation in video sequences. The challenges of action recognition include variability in rapid actions, limited dataset, and environmental factors such as (Weather, Illumination, and Occlusions). In this paper, we present a novel comprehensive dataset for traffic accident detection. This datasets is specifically designed to bolster computer vision and action recognition systems in predicting and detecting road traffic accidents. We integrated datasets from wide variety of data sources, road networks, weather conditions, and regions across the globe. This approach is underpinned by empirical studies, aiming to contribute to the discourse on how technology can enhance the quality of life in densely populated areas. This research aims to bridge existing research gaps by introducing benchmark datasets that leverage state-of-the-art algorithms tailored for traffic accident detection in smart cities. These dataset is expected to advance academic research and also enhance real-time accident detection applications, contributing significantly to the evolution of smart urban environments. Our study marks a pivotal step towards safer, more efficient smart cities, harnessing the power of AI and machine learning to transform urban living.

{{</citation>}}


### (22/53) SeTformer is What You Need for Vision and Language (Pourya Shamsolmoali et al., 2024)

{{<citation>}}

Pourya Shamsolmoali, Masoumeh Zareapoor, Eric Granger, Michael Felsberg. (2024)  
**SeTformer is What You Need for Vision and Language**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GLUE, ImageNet  
[Paper Link](http://arxiv.org/abs/2401.03540v1)  

---


**ABSTRACT**  
The dot product self-attention (DPSA) is a fundamental component of transformers. However, scaling them to long sequences, like documents or high-resolution images, becomes prohibitively expensive due to quadratic time and memory complexities arising from the softmax operation. Kernel methods are employed to simplify computations by approximating softmax but often lead to performance drops compared to softmax attention. We propose SeTformer, a novel transformer, where DPSA is purely replaced by Self-optimal Transport (SeT) for achieving better performance and computational efficiency. SeT is based on two essential softmax properties: maintaining a non-negative attention matrix and using a nonlinear reweighting mechanism to emphasize important tokens in input sequences. By introducing a kernel cost function for optimal transport, SeTformer effectively satisfies these properties. In particular, with small and basesized models, SeTformer achieves impressive top-1 accuracies of 84.7% and 86.2% on ImageNet-1K. In object detection, SeTformer-base outperforms the FocalNet counterpart by +2.2 mAP, using 38% fewer parameters and 29% fewer FLOPs. In semantic segmentation, our base-size model surpasses NAT by +3.5 mIoU with 33% fewer parameters. SeTformer also achieves state-of-the-art results in language modeling on the GLUE benchmark. These findings highlight SeTformer's applicability in vision and language tasks.

{{</citation>}}


### (23/53) Text-Driven Traffic Anomaly Detection with Temporal High-Frequency Modeling in Driving Videos (Rongqin Liang et al., 2024)

{{<citation>}}

Rongqin Liang, Yuanman Li, Jiantao Zhou, Xia Li. (2024)  
**Text-Driven Traffic Anomaly Detection with Temporal High-Frequency Modeling in Driving Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2401.03522v1)  

---


**ABSTRACT**  
Traffic anomaly detection (TAD) in driving videos is critical for ensuring the safety of autonomous driving and advanced driver assistance systems. Previous single-stage TAD methods primarily rely on frame prediction, making them vulnerable to interference from dynamic backgrounds induced by the rapid movement of the dashboard camera. While two-stage TAD methods appear to be a natural solution to mitigate such interference by pre-extracting background-independent features (such as bounding boxes and optical flow) using perceptual algorithms, they are susceptible to the performance of first-stage perceptual algorithms and may result in error propagation. In this paper, we introduce TTHF, a novel single-stage method aligning video clips with text prompts, offering a new perspective on traffic anomaly detection. Unlike previous approaches, the supervised signal of our method is derived from languages rather than orthogonal one-hot vectors, providing a more comprehensive representation. Further, concerning visual representation, we propose to model the high frequency of driving videos in the temporal domain. This modeling captures the dynamic changes of driving scenes, enhances the perception of driving behavior, and significantly improves the detection of traffic anomalies. In addition, to better perceive various types of traffic anomalies, we carefully design an attentive anomaly focusing mechanism that visually and linguistically guides the model to adaptively focus on the visual context of interest, thereby facilitating the detection of traffic anomalies. It is shown that our proposed TTHF achieves promising performance, outperforming state-of-the-art competitors by +5.4% AUC on the DoTA dataset and achieving high generalization on the DADA dataset.

{{</citation>}}


### (24/53) Re:Draw -- Context Aware Translation as a Controllable Method for Artistic Production (Joao Liborio Cardoso et al., 2024)

{{<citation>}}

Joao Liborio Cardoso, Francesco Banterle, Paolo Cignoni, Michael Wimmer. (2024)  
**Re:Draw -- Context Aware Translation as a Controllable Method for Artistic Production**  

---
Primary Category: cs.CV  
Categories: I-2-6; I-2-1; J-5, cs-AI, cs-CV, cs-GR, cs-MM, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03499v1)  

---


**ABSTRACT**  
We introduce context-aware translation, a novel method that combines the benefits of inpainting and image-to-image translation, respecting simultaneously the original input and contextual relevance -- where existing methods fall short. By doing so, our method opens new avenues for the controllable use of AI within artistic creation, from animation to digital art.   As an use case, we apply our method to redraw any hand-drawn animated character eyes based on any design specifications - eyes serve as a focal point that captures viewer attention and conveys a range of emotions, however, the labor-intensive nature of traditional animation often leads to compromises in the complexity and consistency of eye design. Furthermore, we remove the need for production data for training and introduce a new character recognition method that surpasses existing work by not requiring fine-tuning to specific productions. This proposed use case could help maintain consistency throughout production and unlock bolder and more detailed design choices without the production cost drawbacks. A user study shows context-aware translation is preferred over existing work 95.16% of the time.

{{</citation>}}


### (25/53) BCLNet: Bilateral Consensus Learning for Two-View Correspondence Pruning (Xiangyang Miao et al., 2024)

{{<citation>}}

Xiangyang Miao, Guobao Xiao, Shiping Wang, Jun Yu. (2024)  
**BCLNet: Bilateral Consensus Learning for Two-View Correspondence Pruning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2401.03459v1)  

---


**ABSTRACT**  
Correspondence pruning aims to establish reliable correspondences between two related images and recover relative camera motion. Existing approaches often employ a progressive strategy to handle the local and global contexts, with a prominent emphasis on transitioning from local to global, resulting in the neglect of interactions between different contexts. To tackle this issue, we propose a parallel context learning strategy that involves acquiring bilateral consensus for the two-view correspondence pruning task. In our approach, we design a distinctive self-attention block to capture global context and parallel process it with the established local context learning module, which enables us to simultaneously capture both local and global consensuses. By combining these local and global consensuses, we derive the required bilateral consensus. We also design a recalibration block, reducing the influence of erroneous consensus information and enhancing the robustness of the model. The culmination of our efforts is the Bilateral Consensus Learning Network (BCLNet), which efficiently estimates camera pose and identifies inliers (true correspondences). Extensive experiments results demonstrate that our network not only surpasses state-of-the-art methods on benchmark datasets but also showcases robust generalization abilities across various feature extraction techniques. Noteworthily, BCLNet obtains 3.98\% mAP5$^{\circ}$ gains over the second best method on unknown outdoor dataset, and obviously accelerates model training speed. The source code will be available at: https://github.com/guobaoxiao/BCLNet.

{{</citation>}}


### (26/53) SpecRef: A Fast Training-free Baseline of Specific Reference-Condition Real Image Editing (Songyan Chen et al., 2024)

{{<citation>}}

Songyan Chen, Jiancheng Huang. (2024)  
**SpecRef: A Fast Training-free Baseline of Specific Reference-Condition Real Image Editing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.03433v1)  

---


**ABSTRACT**  
Text-conditional image editing based on large diffusion generative model has attracted the attention of both the industry and the research community. Most existing methods are non-reference editing, with the user only able to provide a source image and text prompt. However, it restricts user's control over the characteristics of editing outcome. To increase user freedom, we propose a new task called Specific Reference Condition Real Image Editing, which allows user to provide a reference image to further control the outcome, such as replacing an object with a particular one. To accomplish this, we propose a fast baseline method named SpecRef. Specifically, we design a Specific Reference Attention Controller to incorporate features from the reference image, and adopt a mask mechanism to prevent interference between editing and non-editing regions. We evaluate SpecRef on typical editing tasks and show that it can achieve satisfactory performance. The source code is available on https://github.com/jingjiqinggong/specp2p.

{{</citation>}}


### (27/53) See360: Novel Panoramic View Interpolation (Zhi-Song Liu et al., 2024)

{{<citation>}}

Zhi-Song Liu, Marie-Paule Cani, Wan-Chi Siu. (2024)  
**See360: Novel Panoramic View Interpolation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.03431v1)  

---


**ABSTRACT**  
We present See360, which is a versatile and efficient framework for 360 panoramic view interpolation using latent space viewpoint estimation. Most of the existing view rendering approaches only focus on indoor or synthetic 3D environments and render new views of small objects. In contrast, we suggest to tackle camera-centered view synthesis as a 2D affine transformation without using point clouds or depth maps, which enables an effective 360? panoramic scene exploration. Given a pair of reference images, the See360 model learns to render novel views by a proposed novel Multi-Scale Affine Transformer (MSAT), enabling the coarse-to-fine feature rendering. We also propose a Conditional Latent space AutoEncoder (C-LAE) to achieve view interpolation at any arbitrary angle. To show the versatility of our method, we introduce four training datasets, namely UrbanCity360, Archinterior360, HungHom360 and Lab360, which are collected from indoor and outdoor environments for both real and synthetic rendering. Experimental results show that the proposed method is generic enough to achieve real-time rendering of arbitrary views for all four datasets. In addition, our See360 model can be applied to view synthesis in the wild: with only a short extra training time (approximately 10 mins), and is able to render unknown real-world scenes. The superior performance of See360 opens up a promising direction for camera-centered view rendering and 360 panoramic view interpolation.

{{</citation>}}


### (28/53) Towards Effective Multiple-in-One Image Restoration: A Sequential and Prompt Learning Strategy (Xiangtao Kong et al., 2024)

{{<citation>}}

Xiangtao Kong, Chao Dong, Lei Zhang. (2024)  
**Towards Effective Multiple-in-One Image Restoration: A Sequential and Prompt Learning Strategy**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.03379v1)  

---


**ABSTRACT**  
While single task image restoration (IR) has achieved significant successes, it remains a challenging issue to train a single model which can tackle multiple IR tasks. In this work, we investigate in-depth the multiple-in-one (MiO) IR problem, which comprises seven popular IR tasks. We point out that MiO IR faces two pivotal challenges: the optimization of diverse objectives and the adaptation to multiple tasks. To tackle these challenges, we present two simple yet effective strategies. The first strategy, referred to as sequential learning, attempts to address how to optimize the diverse objectives, which guides the network to incrementally learn individual IR tasks in a sequential manner rather than mixing them together. The second strategy, i.e., prompt learning, attempts to address how to adapt to the different IR tasks, which assists the network to understand the specific task and improves the generalization ability. By evaluating on 19 test sets, we demonstrate that the sequential and prompt learning strategies can significantly enhance the MiO performance of commonly used CNN and Transformer backbones. Our experiments also reveal that the two strategies can supplement each other to learn better degradation representations and enhance the model robustness. It is expected that our proposed MiO IR formulation and strategies could facilitate the research on how to train IR models with higher generalization capabilities.

{{</citation>}}


## cs.AI (7)



### (29/53) Agent AI: Surveying the Horizons of Multimodal Interaction (Zane Durante et al., 2024)

{{<citation>}}

Zane Durante, Qiuyuan Huang, Naoki Wake, Ran Gong, Jae Sung Park, Bidipta Sarkar, Rohan Taori, Yusuke Noda, Demetri Terzopoulos, Yejin Choi, Katsushi Ikeuchi, Hoi Vo, Li Fei-Fei, Jianfeng Gao. (2024)  
**Agent AI: Surveying the Horizons of Multimodal Interaction**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-LG, cs.AI  
Keywords: AI, Embedding  
[Paper Link](http://arxiv.org/abs/2401.03568v1)  

---


**ABSTRACT**  
Multi-modal AI systems will likely become a ubiquitous presence in our everyday lives. A promising approach to making these systems more interactive is to embody them as agents within physical and virtual environments. At present, systems leverage existing foundation models as the basic building blocks for the creation of embodied agents. Embedding agents within such environments facilitates the ability of models to process and interpret visual and contextual data, which is critical for the creation of more sophisticated and context-aware AI systems. For example, a system that can perceive user actions, human behavior, environmental objects, audio expressions, and the collective sentiment of a scene can be used to inform and direct agent responses within the given environment. To accelerate research on agent-based multimodal intelligence, we define "Agent AI" as a class of interactive systems that can perceive visual stimuli, language inputs, and other environmentally-grounded data, and can produce meaningful embodied action with infinite agent. In particular, we explore systems that aim to improve agents based on next-embodied action prediction by incorporating external knowledge, multi-sensory inputs, and human feedback. We argue that by developing agentic AI systems in grounded environments, one can also mitigate the hallucinations of large foundation models and their tendency to generate environmentally incorrect outputs. The emerging field of Agent AI subsumes the broader embodied and agentic aspects of multimodal interactions. Beyond agents acting and interacting in the physical world, we envision a future where people can easily create any virtual reality or simulated scene and interact with agents embodied within the virtual environment.

{{</citation>}}


### (30/53) NovelGym: A Flexible Ecosystem for Hybrid Planning and Learning Agents Designed for Open Worlds (Shivam Goel et al., 2024)

{{<citation>}}

Shivam Goel, Yichen Wei, Panagiotis Lymperopoulos, Matthias Scheutz, Jivko Sinapov. (2024)  
**NovelGym: A Flexible Ecosystem for Hybrid Planning and Learning Agents Designed for Open Worlds**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03546v1)  

---


**ABSTRACT**  
As AI agents leave the lab and venture into the real world as autonomous vehicles, delivery robots, and cooking robots, it is increasingly necessary to design and comprehensively evaluate algorithms that tackle the ``open-world''. To this end, we introduce NovelGym, a flexible and adaptable ecosystem designed to simulate gridworld environments, serving as a robust platform for benchmarking reinforcement learning (RL) and hybrid planning and learning agents in open-world contexts. The modular architecture of NovelGym facilitates rapid creation and modification of task environments, including multi-agent scenarios, with multiple environment transformations, thus providing a dynamic testbed for researchers to develop open-world AI agents.

{{</citation>}}


### (31/53) Quantifying stability of non-power-seeking in artificial agents (Evan Ryan Gunter et al., 2024)

{{<citation>}}

Evan Ryan Gunter, Yevgeny Liokumovich, Victoria Krakovna. (2024)  
**Quantifying stability of non-power-seeking in artificial agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03529v1)  

---


**ABSTRACT**  
We investigate the question: if an AI agent is known to be safe in one setting, is it also safe in a new setting similar to the first? This is a core question of AI alignment--we train and test models in a certain environment, but deploy them in another, and we need to guarantee that models that seem safe in testing remain so in deployment. Our notion of safety is based on power-seeking--an agent which seeks power is not safe. In particular, we focus on a crucial type of power-seeking: resisting shutdown. We model agents as policies for Markov decision processes, and show (in two cases of interest) that not resisting shutdown is "stable": if an MDP has certain policies which don't avoid shutdown, the corresponding policies for a similar MDP also don't avoid shutdown. We also show that there are natural cases where safety is _not_ stable--arbitrarily small perturbations may result in policies which never shut down. In our first case of interest--near-optimal policies--we use a bisimulation metric on MDPs to prove that small perturbations won't make the agent take longer to shut down. Our second case of interest is policies for MDPs satisfying certain constraints which hold for various models (including language models). Here, we demonstrate a quantitative bound on how fast the probability of not shutting down can increase: by defining a metric on MDPs; proving that the probability of not shutting down, as a function on MDPs, is lower semicontinuous; and bounding how quickly this function decreases.

{{</citation>}}


### (32/53) ClusterComm: Discrete Communication in Decentralized MARL using Internal Representation Clustering (Robert Müller et al., 2024)

{{<citation>}}

Robert Müller, Hasan Turalic, Thomy Phan, Michael Kölle, Jonas Nüßlein, Claudia Linnhoff-Popien. (2024)  
**ClusterComm: Discrete Communication in Decentralized MARL using Internal Representation Clustering**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.03504v1)  

---


**ABSTRACT**  
In the realm of Multi-Agent Reinforcement Learning (MARL), prevailing approaches exhibit shortcomings in aligning with human learning, robustness, and scalability. Addressing this, we introduce ClusterComm, a fully decentralized MARL framework where agents communicate discretely without a central control unit. ClusterComm utilizes Mini-Batch-K-Means clustering on the last hidden layer's activations of an agent's policy network, translating them into discrete messages. This approach outperforms no communication and competes favorably with unbounded, continuous communication and hence poses a simple yet effective strategy for enhancing collaborative task-solving in MARL.

{{</citation>}}


### (33/53) Computational Argumentation-based Chatbots: a Survey (Federico Castagna et al., 2024)

{{<citation>}}

Federico Castagna, Nadin Kokciyan, Isabel Sassoon, Simon Parsons, Elizabeth Sklar. (2024)  
**Computational Argumentation-based Chatbots: a Survey**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.03454v1)  

---


**ABSTRACT**  
Chatbots are conversational software applications designed to interact dialectically with users for a plethora of different purposes. Surprisingly, these colloquial agents have only recently been coupled with computational models of arguments (i.e. computational argumentation), whose aim is to formalise, in a machine-readable format, the ordinary exchange of information that characterises human communications. Chatbots may employ argumentation with different degrees and in a variety of manners. The present survey sifts through the literature to review papers concerning this kind of argumentation-based bot, drawing conclusions about the benefits and drawbacks that this approach entails in comparison with standard chatbots, while also envisaging possible future development and integration with the Transformer-based architecture and state-of-the-art Large Language models.

{{</citation>}}


### (34/53) Exploring Large Language Model based Intelligent Agents: Definitions, Methods, and Prospects (Yuheng Cheng et al., 2024)

{{<citation>}}

Yuheng Cheng, Ceyao Zhang, Zhengwen Zhang, Xiangrui Meng, Sirui Hong, Wenhao Li, Zihao Wang, Zekai Wang, Feng Yin, Junhua Zhao, Xiuqiang He. (2024)  
**Exploring Large Language Model based Intelligent Agents: Definitions, Methods, and Prospects**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-MA, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03428v1)  

---


**ABSTRACT**  
Intelligent agents stand out as a potential path toward artificial general intelligence (AGI). Thus, researchers have dedicated significant effort to diverse implementations for them. Benefiting from recent progress in large language models (LLMs), LLM-based agents that use universal natural language as an interface exhibit robust generalization capabilities across various applications -- from serving as autonomous general-purpose task assistants to applications in coding, social, and economic domains, LLM-based agents offer extensive exploration opportunities. This paper surveys current research to provide an in-depth overview of LLM-based intelligent agents within single-agent and multi-agent systems. It covers their definitions, research frameworks, and foundational components such as their composition, cognitive and planning methods, tool utilization, and responses to environmental feedback. We also delve into the mechanisms of deploying LLM-based agents in multi-agent systems, including multi-role collaboration, message passing, and strategies to alleviate communication issues between agents. The discussions also shed light on popular datasets and application scenarios. We conclude by envisioning prospects for LLM-based agents, considering the evolving landscape of AI and natural language processing.

{{</citation>}}


### (35/53) Escalation Risks from Language Models in Military and Diplomatic Decision-Making (Juan-Pablo Rivera et al., 2024)

{{<citation>}}

Juan-Pablo Rivera, Gabriel Mukobi, Anka Reuel, Max Lamparth, Chandler Smith, Jacquelyn Schneider. (2024)  
**Escalation Risks from Language Models in Military and Diplomatic Decision-Making**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CY, cs-MA, cs.AI  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03408v1)  

---


**ABSTRACT**  
Governments are increasingly considering integrating autonomous AI agents in high-stakes military and foreign-policy decision-making, especially with the emergence of advanced generative AI models like GPT-4. Our work aims to scrutinize the behavior of multiple AI agents in simulated wargames, specifically focusing on their predilection to take escalatory actions that may exacerbate multilateral conflicts. Drawing on political science and international relations literature about escalation dynamics, we design a novel wargame simulation and scoring framework to assess the escalation risks of actions taken by these agents in different scenarios. Contrary to prior studies, our research provides both qualitative and quantitative insights and focuses on large language models (LLMs). We find that all five studied off-the-shelf LLMs show forms of escalation and difficult-to-predict escalation patterns. We observe that models tend to develop arms-race dynamics, leading to greater conflict, and in rare cases, even to the deployment of nuclear weapons. Qualitatively, we also collect the models' reported reasonings for chosen actions and observe worrying justifications based on deterrence and first-strike tactics. Given the high stakes of military and foreign-policy contexts, we recommend further examination and cautious consideration before deploying autonomous language model agents for strategic military or diplomatic decision-making.

{{</citation>}}


## cs.CR (2)



### (36/53) Improving Transferability of Network Intrusion Detection in a Federated Learning Setup (Shreya Ghosh et al., 2024)

{{<citation>}}

Shreya Ghosh, Abu Shafin Mohammad Mahdee Jameel, Aly El Gamal. (2024)  
**Improving Transferability of Network Intrusion Detection in a Federated Learning Setup**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR, eess-SP  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2401.03560v1)  

---


**ABSTRACT**  
Network Intrusion Detection Systems (IDS) aim to detect the presence of an intruder by analyzing network packets arriving at an internet connected device. Data-driven deep learning systems, popular due to their superior performance compared to traditional IDS, depend on availability of high quality training data for diverse intrusion classes. A way to overcome this limitation is through transferable learning, where training for one intrusion class can lead to detection of unseen intrusion classes after deployment. In this paper, we provide a detailed study on the transferability of intrusion detection. We investigate practical federated learning configurations to enhance the transferability of intrusion detection. We propose two techniques to significantly improve the transferability of a federated intrusion detection system. The code for this work can be found at https://github.com/ghosh64/transferability.

{{</citation>}}


### (37/53) Ensemble Defense System: A Hybrid IDS Approach for Effective Cyber Threat Detection (Sarah Alharbi et al., 2024)

{{<citation>}}

Sarah Alharbi, Arshiya Khan. (2024)  
**Ensemble Defense System: A Hybrid IDS Approach for Effective Cyber Threat Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Intrusion Detection, Security  
[Paper Link](http://arxiv.org/abs/2401.03491v1)  

---


**ABSTRACT**  
Sophisticated cyber attacks present significant challenges for organizations in detecting and preventing such threats. To address this critical need for advanced defense mechanisms, we propose an Ensemble Defense System (EDS). An EDS is a cybersecurity framework aggregating multiple security tools designed to monitor and alert an organization during cyber attacks. The proposed EDS leverages a comprehensive range of Intrusion Detection System (IDS) capabilities by introducing a hybrid of signature-based IDS and anomaly-based IDS tools. It also incorporates Elasticsearch, an open-source Security Information and Event Management (SIEM) tool, to facilitate data analysis and interactive visualization of alerts generated from IDSs. The effectiveness of the EDS is evaluated through a payload from a bash script that executes various attacks, including port scanning, privilege escalation, and Denial-of-Service (DoS). The evaluation demonstrates the EDS's ability to detect diverse cyber attacks.

{{</citation>}}


## cs.RO (1)



### (38/53) Overview of Dialogue Robot Competition 2023 (Takashi Minato et al., 2024)

{{<citation>}}

Takashi Minato, Ryuichiro Higashinaka, Kurima Sakai, Tomo Funayama, Hiromitsu Nishizaki, Takayuki Naga. (2024)  
**Overview of Dialogue Robot Competition 2023**  

---
Primary Category: cs.RO  
Categories: cs-HC, cs-RO, cs.RO  
Keywords: ChatGPT, Dialog, Dialogue, GPT  
[Paper Link](http://arxiv.org/abs/2401.03547v1)  

---


**ABSTRACT**  
We have held dialogue robot competitions in 2020 and 2022 to compare the performances of interactive robots using an android that closely resembles a human. In 2023, the third competition DRC2023 was held. The task of DRC2023 was designed to be more challenging than the previous travel agent dialogue tasks. Since anyone can now develop a dialogue system using LLMs, the participating teams are required to develop a system that effectively uses information about the situation on the spot (real-time information), which is not handled by ChatGPT and other systems. DRC2023 has two rounds, a preliminary round and the final round as well as the previous competitions. The preliminary round has held on Oct.27 -- Nov.20, 2023 at real travel agency stores. The final round will be held on December 23, 2023. This paper provides an overview of the task settings and evaluation method of DRC2023 and the preliminary round results.

{{</citation>}}


## cs.DL (1)



### (39/53) Is there really a Citation Age Bias in NLP? (Hoa Nguyen et al., 2024)

{{<citation>}}

Hoa Nguyen, Steffen Eger. (2024)  
**Is there really a Citation Age Bias in NLP?**  

---
Primary Category: cs.DL  
Categories: cs-AI, cs-CL, cs-DL, cs.DL  
Keywords: AI, Bias, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.03545v1)  

---


**ABSTRACT**  
Citations are a key ingredient of scientific research to relate a paper to others published in the community. Recently, it has been noted that there is a citation age bias in the Natural Language Processing (NLP) community, one of the currently fastest growing AI subfields, in that the mean age of the bibliography of NLP papers has become ever younger in the last few years, leading to `citation amnesia' in which older knowledge is increasingly forgotten. In this work, we put such claims into perspective by analyzing the bibliography of $\sim$300k papers across 15 different scientific fields submitted to the popular preprint server Arxiv in the time period from 2013 to 2022. We find that all AI subfields (in particular: cs.AI, cs.CL, cs.CV, cs.LG) have similar trends of citation amnesia, in which the age of the bibliography has roughly halved in the last 10 years (from above 12 in 2013 to below 7 in 2022), on average. Rather than diagnosing this as a citation age bias in the NLP community, we believe this pattern is an artefact of the dynamics of these research fields, in which new knowledge is produced in ever shorter time intervals.

{{</citation>}}


## cs.SI (1)



### (40/53) Characterizing Political Campaigning with Lexical Mutants on Indian Social Media (Shruti Phadke et al., 2024)

{{<citation>}}

Shruti Phadke, Tanushree Mitra. (2024)  
**Characterizing Political Campaigning with Lexical Mutants on Indian Social Media**  

---
Primary Category: cs.SI  
Categories: cs-HC, cs-SI, cs.SI  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2401.03533v1)  

---


**ABSTRACT**  
Increasingly online platforms are becoming popular arenas of political amplification in India. With known instances of pre-organized coordinated operations, researchers are questioning the legitimacy of political expression and its consequences on the democratic processes in India. In this paper, we study an evolved form of political amplification by first identifying and then characterizing political campaigns with lexical mutations. By lexical mutation, we mean content that is reframed, paraphrased, or altered while preserving the same underlying message. Using multilingual embeddings and network analysis, we detect over 3.8K political campaigns with text mutations spanning multiple languages and social media platforms in India. By further assessing the political leanings of accounts repeatedly involved in such amplification campaigns, we contribute a broader understanding of how political amplification is used across various political parties in India. Moreover, our temporal analysis of the largest amplification campaigns suggests that political campaigning can evolve as temporally ordered arguments and counter-arguments between groups with competing political interests. Overall, our work contributes insights into how lexical mutations can be leveraged to bypass the platform manipulation policies and how such competing campaigning can provide an exaggerated sense of political divide on Indian social media.

{{</citation>}}


## eess.AS (3)



### (41/53) DiarizationLM: Speaker Diarization Post-Processing with Large Language Models (Quan Wang et al., 2024)

{{<citation>}}

Quan Wang, Yiling Huang, Guanlong Zhao, Evan Clark, Wei Xia, Hank Liao. (2024)  
**DiarizationLM: Speaker Diarization Post-Processing with Large Language Models**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2401.03506v1)  

---


**ABSTRACT**  
In this paper, we introduce DiarizationLM, a framework to leverage large language models (LLM) to post-process the outputs from a speaker diarization system. Various goals can be achieved with the proposed framework, such as improving the readability of the diarized transcript, or reducing the word diarization error rate (WDER). In this framework, the outputs of the automatic speech recognition (ASR) and speaker diarization systems are represented as a compact textual format, which is included in the prompt to an optionally finetuned LLM. The outputs of the LLM can be used as the refined diarization results with the desired enhancement. As a post-processing step, this framework can be easily applied to any off-the-shelf ASR and speaker diarization systems without retraining existing components. Our experiments show that a finetuned PaLM 2-S model can reduce the WDER by rel. 25.9% on the Fisher telephone conversation dataset, and rel. 31% on the Callhome English dataset.

{{</citation>}}


### (42/53) EAT: Self-Supervised Pre-Training with Efficient Audio Transformer (Wenxi Chen et al., 2024)

{{<citation>}}

Wenxi Chen, Yuzhe Liang, Ziyang Ma, Zhisheng Zheng, Xie Chen. (2024)  
**EAT: Self-Supervised Pre-Training with Efficient Audio Transformer**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-CL, cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2401.03497v1)  

---


**ABSTRACT**  
Audio self-supervised learning (SSL) pre-training, which aims to learn good representations from unlabeled audio, has made remarkable progress. However, the extensive computational demands during pre-training pose a significant barrier to the potential application and optimization of audio SSL models. In this paper, inspired by the success of data2vec 2.0 in image modality and Audio-MAE in audio modality, we introduce Efficient Audio Transformer (EAT) to further improve the effectiveness and efficiency in audio SSL. The proposed EAT adopts the bootstrap self-supervised training paradigm to the audio domain. A novel Utterance-Frame Objective (UFO) is designed to enhance the modeling capability of acoustic events. Furthermore, we reveal that the masking strategy is critical in audio SSL pre-training, and superior audio representations can be obtained with large inverse block masks. Experiment results demonstrate that EAT achieves state-of-the-art (SOTA) performance on a range of audio-related tasks, including AudioSet (AS-2M, AS-20K), ESC-50, and SPC-2, along with a significant pre-training speedup up to ~15x compared to existing audio SSL models.

{{</citation>}}


### (43/53) Single-Microphone Speaker Separation and Voice Activity Detection in Noisy and Reverberant Environments (Renana Opochinsky et al., 2024)

{{<citation>}}

Renana Opochinsky, Mordehay Moradi, Sharon Gannot. (2024)  
**Single-Microphone Speaker Separation and Voice Activity Detection in Noisy and Reverberant Environments**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.03448v1)  

---


**ABSTRACT**  
Speech separation involves extracting an individual speaker's voice from a multi-speaker audio signal. The increasing complexity of real-world environments, where multiple speakers might converse simultaneously, underscores the importance of effective speech separation techniques. This work presents a single-microphone speaker separation network with TF attention aiming at noisy and reverberant environments. We dub this new architecture as Separation TF Attention Network (Sep-TFAnet). In addition, we present a variant of the separation network, dubbed $ \text{Sep-TFAnet}^{\text{VAD}}$, which incorporates a voice activity detector (VAD) into the separation network.   The separation module is based on a temporal convolutional network (TCN) backbone inspired by the Conv-Tasnet architecture with multiple modifications. Rather than a learned encoder and decoder, we use short-time Fourier transform (STFT) and inverse short-time Fourier transform (iSTFT) for the analysis and synthesis, respectively. Our system is specially developed for human-robotic interactions and should support online mode. The separation capabilities of $ \text{Sep-TFAnet}^{\text{VAD}}$ and Sep-TFAnet were evaluated and extensively analyzed under several acoustic conditions, demonstrating their advantages over competing methods. Since separation networks trained on simulated data tend to perform poorly on real recordings, we also demonstrate the ability of the proposed scheme to better generalize to realistic examples recorded in our acoustic lab by a humanoid robot. Project page: https://Sep-TFAnet.github.io

{{</citation>}}


## cs.CY (3)



### (44/53) A Large Language Model Supported Synthesis of Contemporary Academic Integrity Research Trends (Thomas Lancaster, 2024)

{{<citation>}}

Thomas Lancaster. (2024)  
**A Large Language Model Supported Synthesis of Contemporary Academic Integrity Research Trends**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03481v1)  

---


**ABSTRACT**  
This paper reports on qualitative content analysis undertaken using ChatGPT, a Large Language Model (LLM), to identify primary research themes in current academic integrity research as well as the methodologies used to explore these areas. The analysis by the LLM identified 7 research themes and 13 key areas for exploration. The outcomes from the analysis suggest that much contemporary research in the academic integrity field is guided by technology. Technology is often explored as potential way of preventing academic misconduct, but this could also be a limiting factor when aiming to promote a culture of academic integrity. The findings underscore that LLM led research may be option in the academic integrity field, but that there is also a need for continued traditional research. The findings also indicate that researchers and educational providers should continue to develop policy and operational frameworks for academic integrity. This will help to ensure that academic standards are maintained across the wide range of settings that are present in modern education.

{{</citation>}}


### (45/53) Amplifying robotics capacities with a human touch: An immersive low-latency panoramic remote system (Junjie Li et al., 2024)

{{<citation>}}

Junjie Li, Jian Xu, Dewei Han, Kang Li, Zhaoyuan Ma. (2024)  
**Amplifying robotics capacities with a human touch: An immersive low-latency panoramic remote system**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-RO, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03398v1)  

---


**ABSTRACT**  
AI and robotics technologies have witnessed remarkable advancements in the past decade, revolutionizing work patterns and opportunities in various domains. The application of these technologies has propelled society towards an era of symbiosis between humans and machines. To facilitate efficient communication between humans and intelligent robots, we propose the "Avatar" system, an immersive low-latency panoramic human-robot interaction platform. We have designed and tested a prototype of a rugged mobile platform integrated with edge computing units, panoramic video capture devices, power batteries, robot arms, and network communication equipment. Under favorable network conditions, we achieved a low-latency high-definition panoramic visual experience with a delay of 357ms. Operators can utilize VR headsets and controllers for real-time immersive control of robots and devices. The system enables remote control over vast physical distances, spanning campuses, provinces, countries, and even continents (New York to Shenzhen). Additionally, the system incorporates visual SLAM technology for map and trajectory recording, providing autonomous navigation capabilities. We believe that this intuitive system platform can enhance efficiency and situational experience in human-robot collaboration, and with further advancements in related technologies, it will become a versatile tool for efficient and symbiotic cooperation between AI and humans.

{{</citation>}}


### (46/53) An Investigation of Large Language Models for Real-World Hate Speech Detection (Keyan Guo et al., 2024)

{{<citation>}}

Keyan Guo, Alexander Hu, Jaden Mu, Ziheng Shi, Ziming Zhao, Nishant Vishwamitra, Hongxin Hu. (2024)  
**An Investigation of Large Language Models for Real-World Hate Speech Detection**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs-SI, cs.CY  
Keywords: Hate Speech Detection, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03346v1)  

---


**ABSTRACT**  
Hate speech has emerged as a major problem plaguing our social spaces today. While there have been significant efforts to address this problem, existing methods are still significantly limited in effectively detecting hate speech online. A major limitation of existing methods is that hate speech detection is a highly contextual problem, and these methods cannot fully capture the context of hate speech to make accurate predictions. Recently, large language models (LLMs) have demonstrated state-of-the-art performance in several natural language tasks. LLMs have undergone extensive training using vast amounts of natural language data, enabling them to grasp intricate contextual details. Hence, they could be used as knowledge bases for context-aware hate speech detection. However, a fundamental problem with using LLMs to detect hate speech is that there are no studies on effectively prompting LLMs for context-aware hate speech detection. In this study, we conduct a large-scale study of hate speech detection, employing five established hate speech datasets. We discover that LLMs not only match but often surpass the performance of current benchmark machine learning models in identifying hate speech. By proposing four diverse prompting strategies that optimize the use of LLMs in detecting hate speech. Our study reveals that a meticulously crafted reasoning prompt can effectively capture the context of hate speech by fully utilizing the knowledge base in LLMs, significantly outperforming existing techniques. Furthermore, although LLMs can provide a rich knowledge base for the contextual detection of hate speech, suitable prompting strategies play a crucial role in effectively leveraging this knowledge base for efficient detection.

{{</citation>}}


## cs.SD (2)



### (47/53) ICMC-ASR: The ICASSP 2024 In-Car Multi-Channel Automatic Speech Recognition Challenge (He Wang et al., 2024)

{{<citation>}}

He Wang, Pengcheng Guo, Yue Li, Ao Zhang, Jiayao Sun, Lei Xie, Wei Chen, Pan Zhou, Hui Bu, Xin Xu, Binbin Zhang, Zhuo Chen, Jian Wu, Longbiao Wang, Eng Siong Chng, Sun Li. (2024)  
**ICMC-ASR: The ICASSP 2024 In-Car Multi-Channel Automatic Speech Recognition Challenge**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.03473v1)  

---


**ABSTRACT**  
To promote speech processing and recognition research in driving scenarios, we build on the success of the Intelligent Cockpit Speech Recognition Challenge (ICSRC) held at ISCSLP 2022 and launch the ICASSP 2024 In-Car Multi-Channel Automatic Speech Recognition (ICMC-ASR) Challenge. This challenge collects over 100 hours of multi-channel speech data recorded inside a new energy vehicle and 40 hours of noise for data augmentation. Two tracks, including automatic speech recognition (ASR) and automatic speech diarization and recognition (ASDR) are set up, using character error rate (CER) and concatenated minimum permutation character error rate (cpCER) as evaluation metrics, respectively. Overall, the ICMC-ASR Challenge attracts 98 participating teams and receives 53 valid results in both tracks. In the end, first-place team USTCiflytek achieves a CER of 13.16% in the ASR track and a cpCER of 21.48% in the ASDR track, showing an absolute improvement of 13.08% and 51.4% compared to our challenge baseline, respectively.

{{</citation>}}


### (48/53) MLCA-AVSR: Multi-Layer Cross Attention Fusion based Audio-Visual Speech Recognition (He Wang et al., 2024)

{{<citation>}}

He Wang, Pengcheng Guo, Pan Zhou, Lei Xie. (2024)  
**MLCA-AVSR: Multi-Layer Cross Attention Fusion based Audio-Visual Speech Recognition**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: Attention, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.03424v1)  

---


**ABSTRACT**  
While automatic speech recognition (ASR) systems degrade significantly in noisy environments, audio-visual speech recognition (AVSR) systems aim to complement the audio stream with noise-invariant visual cues and improve the system's robustness. However, current studies mainly focus on fusing the well-learned modality features, like the output of modality-specific encoders, without considering the contextual relationship during the modality feature learning. In this study, we propose a multi-layer cross-attention fusion based AVSR (MLCA-AVSR) approach that promotes representation learning of each modality by fusing them at different levels of audio/visual encoders. Experimental results on the MISP2022-AVSR Challenge dataset show the efficacy of our proposed system, achieving a concatenated minimum permutation character error rate (cpCER) of 30.57% on the Eval set and yielding up to 3.17% relative improvement compared with our previous system which ranked the second place in the challenge. Following the fusion of multiple systems, our proposed approach surpasses the first-place system, establishing a new SOTA cpCER of 29.13% on this dataset.

{{</citation>}}


## cs.HC (1)



### (49/53) MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition (Zheng Lian et al., 2024)

{{<citation>}}

Zheng Lian, Licai Sun, Yong Ren, Hao Gu, Haiyang Sun, Lan Chen, Bin Liu, Jianhua Tao. (2024)  
**MERBench: A Unified Evaluation Benchmark for Multimodal Emotion Recognition**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2401.03429v1)  

---


**ABSTRACT**  
Multimodal emotion recognition plays a crucial role in enhancing user experience in human-computer interaction. Over the past few decades, researchers have proposed a series of algorithms and achieved impressive progress. Although each method shows its superior performance, different methods lack a fair comparison due to inconsistencies in feature extractors, evaluation manners, and experimental settings. These inconsistencies severely hinder the development of this field. Therefore, we build MERBench, a unified evaluation benchmark for multimodal emotion recognition. We aim to reveal the contribution of some important techniques employed in previous works, such as feature selection, multimodal fusion, robustness analysis, fine-tuning, pre-training, etc. We hope this benchmark can provide clear and comprehensive guidance for follow-up researchers. Based on the evaluation results of MERBench, we further point out some promising research directions. Additionally, we introduce a new emotion dataset MER2023, focusing on the Chinese language environment. This dataset can serve as a benchmark dataset for research on multi-label learning, noise robustness, and semi-supervised learning. We will open-source the code and encourage researchers to evaluate their algorithms under the same experimental setup as MERBench for fair comparisons.

{{</citation>}}


## cs.CE (1)



### (50/53) Deep peak property learning for efficient chiral molecules ECD spectra prediction (Hao Li et al., 2024)

{{<citation>}}

Hao Li, Da Long, Li Yuan, Yonghong Tian, Xinchang Wang, Fanyang Mo. (2024)  
**Deep peak property learning for efficient chiral molecules ECD spectra prediction**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.03403v1)  

---


**ABSTRACT**  
Chiral molecule assignation is crucial for asymmetric catalysis, functional materials, and the drug industry. The conventional approach requires theoretical calculations of electronic circular dichroism (ECD) spectra, which is time-consuming and costly. To speed up this process, we have incorporated deep learning techniques for the ECD prediction. We first set up a large-scale dataset of Chiral Molecular ECD spectra (CMCDS) with calculated ECD spectra. We further develop the ECDFormer model, a Transformer-based model to learn the chiral molecular representations and predict corresponding ECD spectra with improved efficiency and accuracy. Unlike other models for spectrum prediction, our ECDFormer creatively focused on peak properties rather than the whole spectrum sequence for prediction, inspired by the scenario of chiral molecule assignation. Specifically, ECDFormer predicts the peak properties, including number, position, and symbol, then renders the ECD spectra from these peak properties, which significantly outperforms other models in ECD prediction, Our ECDFormer reduces the time of acquiring ECD spectra from 1-100 hours per molecule to 1.5s.

{{</citation>}}


## q-bio.PE (1)



### (51/53) Global Prediction of COVID-19 Variant Emergence Using Dynamics-Informed Graph Neural Networks (Majd Al Aawar et al., 2024)

{{<citation>}}

Majd Al Aawar, Srikar Mutnuri, Mansooreh Montazerin, Ajitesh Srivastava. (2024)  
**Global Prediction of COVID-19 Variant Emergence Using Dynamics-Informed Graph Neural Networks**  

---
Primary Category: q-bio.PE  
Categories: cs-LG, physics-soc-ph, q-bio-PE, q-bio.PE  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.03390v1)  

---


**ABSTRACT**  
During the COVID-19 pandemic, a major driver of new surges has been the emergence of new variants. When a new variant emerges in one or more countries, other nations monitor its spread in preparation for its potential arrival. The impact of the variant and the timing of epidemic peaks in a country highly depend on when the variant arrives. The current methods for predicting the spread of new variants rely on statistical modeling, however, these methods work only when the new variant has already arrived in the region of interest and has a significant prevalence. The question arises: Can we predict when (and if) a variant that exists elsewhere will arrive in a given country and reach a certain prevalence? We propose a variant-dynamics-informed Graph Neural Network (GNN) approach. First, We derive the dynamics of variant prevalence across pairs of regions (countries) that applies to a large class of epidemic models. The dynamics suggest that ratios of variant proportions lead to simpler patterns. Therefore, we use ratios of variant proportions along with some parameters estimated from the dynamics as features in a GNN. We develop a benchmarking tool to evaluate variant emergence prediction over 87 countries and 36 variants. We leverage this tool to compare our GNN-based approach against our dynamics-only model and a number of machine learning models. Results show that the proposed dynamics-informed GNN method retrospectively outperforms all the baselines, including the currently pervasive framework of Physics-Informed Neural Networks (PINNs) that incorporates the dynamics in the loss function.

{{</citation>}}


## cs.SE (1)



### (52/53) LLM-Powered Code Vulnerability Repair with Reinforcement Learning and Semantic Reward (Nafis Tanveer Islam et al., 2024)

{{<citation>}}

Nafis Tanveer Islam, Joseph Khoury, Andrew Seong, Gonzalo De La Torre Parra, Elias Bou-Harb, Peyman Najafirad. (2024)  
**LLM-Powered Code Vulnerability Repair with Reinforcement Learning and Semantic Reward**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.03374v1)  

---


**ABSTRACT**  
In software development, the predominant emphasis on functionality often supersedes security concerns, a trend gaining momentum with AI-driven automation tools like GitHub Copilot. These tools significantly improve developers' efficiency in functional code development. Nevertheless, it remains a notable concern that such tools are also responsible for creating insecure code, predominantly because of pre-training on publicly available repositories with vulnerable code. Moreover, developers are called the "weakest link in the chain" since they have very minimal knowledge of code security. Although existing solutions provide a reasonable solution to vulnerable code, they must adequately describe and educate the developers on code security to ensure that the security issues are not repeated. Therefore we introduce a multipurpose code vulnerability analysis system \texttt{SecRepair}, powered by a large language model, CodeGen2 assisting the developer in identifying and generating fixed code along with a complete description of the vulnerability with a code comment. Our innovative methodology uses a reinforcement learning paradigm to generate code comments augmented by a semantic reward mechanism. Inspired by how humans fix code issues, we propose an instruction-based dataset suitable for vulnerability analysis with LLMs. We further identify zero-day and N-day vulnerabilities in 6 Open Source IoT Operating Systems on GitHub. Our findings underscore that incorporating reinforcement learning coupled with semantic reward augments our model's performance, thereby fortifying its capacity to address code vulnerabilities with improved efficacy.

{{</citation>}}


## q-bio.MN (1)



### (53/53) Multi-Modal Representation Learning for Molecular Property Prediction: Sequence, Graph, Geometry (Zeyu Wang et al., 2024)

{{<citation>}}

Zeyu Wang, Tianyi Jiang, Jinhuan Wang, Qi Xuan. (2024)  
**Multi-Modal Representation Learning for Molecular Property Prediction: Sequence, Graph, Geometry**  

---
Primary Category: q-bio.MN  
Categories: cs-LG, q-bio-BM, q-bio-MN, q-bio.MN  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.03369v1)  

---


**ABSTRACT**  
Recent years have seen a rapid growth of machine learning in cheminformatics problems. In order to tackle the problem of insufficient training data in reality, more and more researchers pay attention to data augmentation technology. However, few researchers pay attention to the problem of construction rules and domain information of data, which will directly impact the quality of augmented data and the augmentation performance. While in graph-based molecular research, the molecular connectivity index, as a critical topological index, can directly or indirectly reflect the topology-based physicochemical properties and biological activities. In this paper, we propose a novel data augmentation technique that modifies the topology of the molecular graph to generate augmented data with the same molecular connectivity index as the original data. The molecular connectivity index combined with data augmentation technology helps to retain more topology-based molecular properties information and generate more reliable data. Furthermore, we adopt five benchmark datasets to test our proposed models, and the results indicate that the augmented data generated based on important molecular topology features can effectively improve the prediction accuracy of molecular properties, which also provides a new perspective on data augmentation in cheminformatics studies.

{{</citation>}}
