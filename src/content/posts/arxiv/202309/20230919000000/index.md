---
draft: false
title: "arXiv @ 2023.09.19"
date: 2023-09-19
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.19"
    identifier: arxiv_20230919
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.AI (4)](#csai-4)
- [cs.CL (12)](#cscl-12)
- [cs.LG (4)](#cslg-4)
- [physics.chem-ph (1)](#physicschem-ph-1)
- [astro-ph.EP (1)](#astro-phep-1)
- [cs.SD (1)](#cssd-1)
- [cs.CV (11)](#cscv-11)
- [cs.SE (4)](#csse-4)
- [eess.AS (3)](#eessas-3)
- [cs.IR (1)](#csir-1)
- [cs.RO (4)](#csro-4)
- [cs.CR (1)](#cscr-1)
- [cs.MA (1)](#csma-1)
- [cs.ET (1)](#cset-1)
- [cs.DC (1)](#csdc-1)

## cs.AI (4)



### (1/50) ChatGPT Hallucinates when Attributing Answers (Guido Zuccon et al., 2023)

{{<citation>}}

Guido Zuccon, Bevan Koopman, Razia Shaik. (2023)  
**ChatGPT Hallucinates when Attributing Answers**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-DL, cs-IR, cs.AI  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.09401v1)  

---


**ABSTRACT**  
Can ChatGPT provide evidence to support its answers? Does the evidence it suggests actually exist and does it really support its answer? We investigate these questions using a collection of domain-specific knowledge-based questions, specifically prompting ChatGPT to provide both an answer and supporting evidence in the form of references to external sources. We also investigate how different prompts impact answers and evidence. We find that ChatGPT provides correct or partially correct answers in about half of the cases (50.6% of the times), but its suggested references only exist 14% of the times. We further provide insights on the generated references that reveal common traits among the references that ChatGPT generates, and show how even if a reference provided by the model does exist, this reference often does not support the claims ChatGPT attributes to it. Our findings are important because (1) they are the first systematic analysis of the references created by ChatGPT in its answers; (2) they suggest that the model may leverage good quality information in producing correct answers, but is unable to attribute real evidence to support its answers. Prompts, raw result files and manual analysis are made publicly available.

{{</citation>}}


### (2/50) How much can ChatGPT really help Computational Biologists in Programming? (Chowdhury Rafeed Rahman et al., 2023)

{{<citation>}}

Chowdhury Rafeed Rahman, Limsoon Wong. (2023)  
**How much can ChatGPT really help Computational Biologists in Programming?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.09126v1)  

---


**ABSTRACT**  
ChatGPT, a recently developed product by openAI, is successfully leaving its mark as a multi-purpose natural language based chatbot. In this paper, we are more interested in analyzing its potential in the field of computational biology. A major share of work done by computational biologists these days involve coding up Bioinformatics algorithms, analyzing data, creating pipelining scripts and even machine learning modeling & feature extraction. This paper focuses on the potential influence (both positive and negative) of ChatGPT in the mentioned aspects with illustrative examples from different perspectives. Compared to other fields of Computer Science, Computational Biology has - (1) less coding resources, (2) more sensitivity and bias issues (deals with medical data) and (3) more necessity of coding assistance (people from diverse background come to this field). Keeping such issues in mind, we cover use cases such as code writing, reviewing, debugging, converting, refactoring and pipelining using ChatGPT from the perspective of computational biologists in this paper.

{{</citation>}}


### (3/50) Using Reinforcement Learning to Simplify Mealtime Insulin Dosing for People with Type 1 Diabetes: In-Silico Experiments (Anas El Fathi et al., 2023)

{{<citation>}}

Anas El Fathi, Marc D. Breton. (2023)  
**Using Reinforcement Learning to Simplify Mealtime Insulin Dosing for People with Type 1 Diabetes: In-Silico Experiments**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: LSTM, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.09125v1)  

---


**ABSTRACT**  
People with type 1 diabetes (T1D) struggle to calculate the optimal insulin dose at mealtime, especially when under multiple daily injections (MDI) therapy. Effectively, they will not always perform rigorous and precise calculations, but occasionally, they might rely on intuition and previous experience. Reinforcement learning (RL) has shown outstanding results in outperforming humans on tasks requiring intuition and learning from experience. In this work, we propose an RL agent that recommends the optimal meal-accompanying insulin dose corresponding to a qualitative meal (QM) strategy that does not require precise carbohydrate counting (CC) (e.g., a usual meal at noon.). The agent is trained using the soft actor-critic approach and comprises long short-term memory (LSTM) neurons. For training, eighty virtual subjects (VS) of the FDA-accepted UVA/Padova T1D adult population were simulated using MDI therapy and QM strategy. For validation, the remaining twenty VS were examined in 26-week scenarios, including intra- and inter-day variabilities in glucose. \textit{In-silico} results showed that the proposed RL approach outperforms a baseline run-to-run approach and can replace the standard CC approach. Specifically, after 26 weeks, the time-in-range ($70-180$mg/dL) and time-in-hypoglycemia ($<70$mg/dL) were $73.1\pm11.6$% and $ 2.0\pm 1.8$% using the RL-optimized QM strategy compared to $70.6\pm14.8$% and $ 1.5\pm 1.5$% using CC. Such an approach can simplify diabetes treatment, resulting in improved quality of life and glycemic outcomes.

{{</citation>}}


### (4/50) Public Perceptions of Gender Bias in Large Language Models: Cases of ChatGPT and Ernie (Kyrie Zhixuan Zhou et al., 2023)

{{<citation>}}

Kyrie Zhixuan Zhou, Madelyn Rose Sanfilippo. (2023)  
**Public Perceptions of Gender Bias in Large Language Models: Cases of ChatGPT and Ernie**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: Bias, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.09120v1)  

---


**ABSTRACT**  
Large language models are quickly gaining momentum, yet are found to demonstrate gender bias in their responses. In this paper, we conducted a content analysis of social media discussions to gauge public perceptions of gender bias in LLMs which are trained in different cultural contexts, i.e., ChatGPT, a US-based LLM, or Ernie, a China-based LLM. People shared both observations of gender bias in their personal use and scientific findings about gender bias in LLMs. A difference between the two LLMs was seen -- ChatGPT was more often found to carry implicit gender bias, e.g., associating men and women with different profession titles, while explicit gender bias was found in Ernie's responses, e.g., overly promoting women's pursuit of marriage over career. Based on the findings, we reflect on the impact of culture on gender bias and propose governance recommendations to regulate gender bias in LLMs.

{{</citation>}}


## cs.CL (12)



### (5/50) CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages (Thuat Nguyen et al., 2023)

{{<citation>}}

Thuat Nguyen, Chien Van Nguyen, Viet Dac Lai, Hieu Man, Nghia Trung Ngo, Franck Dernoncourt, Ryan A. Rossi, Thien Huu Nguyen. (2023)  
**CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2309.09400v1)  

---


**ABSTRACT**  
The driving factors behind the development of large language models (LLMs) with impressive learning capabilities are their colossal model sizes and extensive training datasets. Along with the progress in natural language processing, LLMs have been frequently made accessible to the public to foster deeper investigation and applications. However, when it comes to training datasets for these LLMs, especially the recent state-of-the-art models, they are often not fully disclosed. Creating training data for high-performing LLMs involves extensive cleaning and deduplication to ensure the necessary level of quality. The lack of transparency for training data has thus hampered research on attributing and addressing hallucination and bias issues in LLMs, hindering replication efforts and further advancements in the community. These challenges become even more pronounced in multilingual learning scenarios, where the available multilingual text datasets are often inadequately collected and cleaned. Consequently, there is a lack of open-source and readily usable dataset to effectively train LLMs in multiple languages. To overcome this issue, we present CulturaX, a substantial multilingual dataset with 6.3 trillion tokens in 167 languages, tailored for LLM development. Our dataset undergoes meticulous cleaning and deduplication through a rigorous pipeline of multiple stages to accomplish the best quality for model training, including language identification, URL-based filtering, metric-based cleaning, document refinement, and data deduplication. CulturaX is fully released to the public in HuggingFace to facilitate research and advancements in multilingual LLMs: https://huggingface.co/datasets/uonlp/CulturaX.

{{</citation>}}


### (6/50) Do Large GPT Models Discover Moral Dimensions in Language Representations? A Topological Study Of Sentence Embeddings (Stephen Fitz, 2023)

{{<citation>}}

Stephen Fitz. (2023)  
**Do Large GPT Models Discover Moral Dimensions in Language Representations? A Topological Study Of Sentence Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs-NE, cs.CL  
Keywords: Embedding, GPT, GPT-3.5, Language Model, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2309.09397v1)  

---


**ABSTRACT**  
As Large Language Models are deployed within Artificial Intelligence systems, that are increasingly integrated with human society, it becomes more important than ever to study their internal structures. Higher level abilities of LLMs such as GPT-3.5 emerge in large part due to informative language representations they induce from raw text data during pre-training on trillions of words. These embeddings exist in vector spaces of several thousand dimensions, and their processing involves mapping between multiple vector spaces, with total number of parameters on the order of trillions. Furthermore, these language representations are induced by gradient optimization, resulting in a black box system that is hard to interpret. In this paper, we take a look at the topological structure of neuronal activity in the "brain" of Chat-GPT's foundation language model, and analyze it with respect to a metric representing the notion of fairness. We develop a novel approach to visualize GPT's moral dimensions. We first compute a fairness metric, inspired by social psychology literature, to identify factors that typically influence fairness assessments in humans, such as legitimacy, need, and responsibility. Subsequently, we summarize the manifold's shape using a lower-dimensional simplicial complex, whose topology is derived from this metric. We color it with a heat map associated with this fairness metric, producing human-readable visualizations of the high-dimensional sentence manifold. Our results show that sentence embeddings based on GPT-3.5 can be decomposed into two submanifolds corresponding to fair and unfair moral judgments. This indicates that GPT-based language models develop a moral dimension within their representation spaces and induce an understanding of fairness during their training process.

{{</citation>}}


### (7/50) Augmenting text for spoken language understanding with Large Language Models (Roshan Sharma et al., 2023)

{{<citation>}}

Roshan Sharma, Suyoun Kim, Daniel Lazar, Trang Le, Akshat Shrivastava, Kwanghoon Ahn, Piyush Kansal, Leda Sari, Ozlem Kalinli, Michael Seltzer. (2023)  
**Augmenting text for spoken language understanding with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.09390v1)  

---


**ABSTRACT**  
Spoken semantic parsing (SSP) involves generating machine-comprehensible parses from input speech. Training robust models for existing application domains represented in training data or extending to new domains requires corresponding triplets of speech-transcript-semantic parse data, which is expensive to obtain. In this paper, we address this challenge by examining methods that can use transcript-semantic parse data (unpaired text) without corresponding speech. First, when unpaired text is drawn from existing textual corpora, Joint Audio Text (JAT) and Text-to-Speech (TTS) are compared as ways to generate speech representations for unpaired text. Experiments on the STOP dataset show that unpaired text from existing and new domains improves performance by 2% and 30% in absolute Exact Match (EM) respectively. Second, we consider the setting when unpaired text is not available in existing textual corpora. We propose to prompt Large Language Models (LLMs) to generate unpaired text for existing and new domains. Experiments show that examples and words that co-occur with intents can be used to generate unpaired text with Llama 2.0. Using the generated text with JAT and TTS for spoken semantic parsing improves EM on STOP by 1.4% and 2.6% absolute for existing and new domains respectively.

{{</citation>}}


### (8/50) Mitigating Shortcuts in Language Models with Soft Label Encoding (Zirui He et al., 2023)

{{<citation>}}

Zirui He, Huiqi Deng, Haiyan Zhao, Ninghao Liu, Mengnan Du. (2023)  
**Mitigating Shortcuts in Language Models with Soft Label Encoding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLU  
[Paper Link](http://arxiv.org/abs/2309.09380v1)  

---


**ABSTRACT**  
Recent research has shown that large language models rely on spurious correlations in the data for natural language understanding (NLU) tasks. In this work, we aim to answer the following research question: Can we reduce spurious correlations by modifying the ground truth labels of the training data? Specifically, we propose a simple yet effective debiasing framework, named Soft Label Encoding (SoftLE). We first train a teacher model with hard labels to determine each sample's degree of relying on shortcuts. We then add one dummy class to encode the shortcut degree, which is used to smooth other dimensions in the ground truth label to generate soft labels. This new ground truth label is used to train a more robust student model. Extensive experiments on two NLU benchmark tasks demonstrate that SoftLE significantly improves out-of-distribution generalization while maintaining satisfactory in-distribution accuracy.

{{</citation>}}


### (9/50) Embrace Divergence for Richer Insights: A Multi-document Summarization Benchmark and a Case Study on Summarizing Diverse Information from News Articles (Kung-Hsiang Huang et al., 2023)

{{<citation>}}

Kung-Hsiang Huang, Philippe Laban, Alexander R. Fabbri, Prafulla Kumar Choubey, Shafiq Joty, Caiming Xiong, Chien-Sheng Wu. (2023)  
**Embrace Divergence for Richer Insights: A Multi-document Summarization Benchmark and a Case Study on Summarizing Diverse Information from News Articles**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2309.09369v1)  

---


**ABSTRACT**  
Previous research in multi-document news summarization has typically concentrated on collating information that all sources agree upon. However, to our knowledge, the summarization of diverse information dispersed across multiple articles about an event has not been previously investigated. The latter imposes a different set of challenges for a summarization model. In this paper, we propose a new task of summarizing diverse information encountered in multiple news articles encompassing the same event. To facilitate this task, we outlined a data collection schema for identifying diverse information and curated a dataset named DiverseSumm. The dataset includes 245 news stories, with each story comprising 10 news articles and paired with a human-validated reference. Moreover, we conducted a comprehensive analysis to pinpoint the position and verbosity biases when utilizing Large Language Model (LLM)-based metrics for evaluating the coverage and faithfulness of the summaries, as well as their correlation with human assessments. We applied our findings to study how LLMs summarize multiple news articles by analyzing which type of diverse information LLMs are capable of identifying. Our analyses suggest that despite the extraordinary capabilities of LLMs in single-document summarization, the proposed task remains a complex challenge for them mainly due to their limited coverage, with GPT-4 only able to cover less than 40% of the diverse information on average.

{{</citation>}}


### (10/50) Performance of the Pre-Trained Large Language Model GPT-4 on Automated Short Answer Grading (Gerd Kortemeyer, 2023)

{{<citation>}}

Gerd Kortemeyer. (2023)  
**Performance of the Pre-Trained Large Language Model GPT-4 on Automated Short Answer Grading**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.09338v1)  

---


**ABSTRACT**  
Automated Short Answer Grading (ASAG) has been an active area of machine-learning research for over a decade. It promises to let educators grade and give feedback on free-form responses in large-enrollment courses in spite of limited availability of human graders. Over the years, carefully trained models have achieved increasingly higher levels of performance. More recently, pre-trained Large Language Models (LLMs) emerged as a commodity, and an intriguing question is how a general-purpose tool without additional training compares to specialized models. We studied the performance of GPT-4 on the standard benchmark 2-way and 3-way datasets SciEntsBank and Beetle, where in addition to the standard task of grading the alignment of the student answer with a reference answer, we also investigated withholding the reference answer. We found that overall, the performance of the pre-trained general-purpose GPT-4 LLM is comparable to hand-engineered models, but worse than pre-trained LLMs that had specialized training.

{{</citation>}}


### (11/50) OWL: A Large Language Model for IT Operations (Hongcheng Guo et al., 2023)

{{<citation>}}

Hongcheng Guo, Jian Yang, Jiaheng Liu, Liqun Yang, Linzheng Chai, Jiaqi Bai, Junran Peng, Xiaorong Hu, Chao Chen, Dongfeng Zhang, Xu Shi, Tieqiao Zheng, Liangfan Zheng, Bo Zhang, Ke Xu, Zhoujun Li. (2023)  
**OWL: A Large Language Model for IT Operations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.09298v1)  

---


**ABSTRACT**  
With the rapid development of IT operations, it has become increasingly crucial to efficiently manage and analyze large volumes of data for practical applications. The techniques of Natural Language Processing (NLP) have shown remarkable capabilities for various tasks, including named entity recognition, machine translation and dialogue systems. Recently, Large Language Models (LLMs) have achieved significant improvements across various NLP downstream tasks. However, there is a lack of specialized LLMs for IT operations. In this paper, we introduce the OWL, a large language model trained on our collected OWL-Instruct dataset with a wide range of IT-related information, where the mixture-of-adapter strategy is proposed to improve the parameter-efficient tuning across different domains or tasks. Furthermore, we evaluate the performance of our OWL on the OWL-Bench established by us and open IT-related benchmarks. OWL demonstrates superior performance results on IT tasks, which outperforms existing models by significant margins. Moreover, we hope that the findings of our work will provide more insights to revolutionize the techniques of IT operations with specialized LLMs.

{{</citation>}}


### (12/50) Model-based Subsampling for Knowledge Graph Completion (Xincan Feng et al., 2023)

{{<citation>}}

Xincan Feng, Hidetaka Kamigaito, Katsuhiko Hayashi, Taro Watanabe. (2023)  
**Model-based Subsampling for Knowledge Graph Completion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.09296v1)  

---


**ABSTRACT**  
Subsampling is effective in Knowledge Graph Embedding (KGE) for reducing overfitting caused by the sparsity in Knowledge Graph (KG) datasets. However, current subsampling approaches consider only frequencies of queries that consist of entities and their relations. Thus, the existing subsampling potentially underestimates the appearance probabilities of infrequent queries even if the frequencies of their entities or relations are high. To address this problem, we propose Model-based Subsampling (MBS) and Mixed Subsampling (MIX) to estimate their appearance probabilities through predictions of KGE models. Evaluation results on datasets FB15k-237, WN18RR, and YAGO3-10 showed that our proposed subsampling methods actually improved the KG completion performances for popular KGE models, RotatE, TransE, HAKE, ComplEx, and DistMult.

{{</citation>}}


### (13/50) Leveraging Social Discourse to Measure Check-worthiness of Claims for Fact-checking (Megha Sundriyal et al., 2023)

{{<citation>}}

Megha Sundriyal, Md Shad Akhtar, Tanmoy Chakraborty. (2023)  
**Leveraging Social Discourse to Measure Check-worthiness of Claims for Fact-checking**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2309.09274v1)  

---


**ABSTRACT**  
The expansion of online social media platforms has led to a surge in online content consumption. However, this has also paved the way for disseminating false claims and misinformation. As a result, there is an escalating demand for a substantial workforce to sift through and validate such unverified claims. Currently, these claims are manually verified by fact-checkers. Still, the volume of online content often outweighs their potency, making it difficult for them to validate every single claim in a timely manner. Thus, it is critical to determine which assertions are worth fact-checking and prioritize claims that require immediate attention. Multiple factors contribute to determining whether a claim necessitates fact-checking, encompassing factors such as its factual correctness, potential impact on the public, the probability of inciting hatred, and more. Despite several efforts to address claim check-worthiness, a systematic approach to identify these factors remains an open challenge. To this end, we introduce a new task of fine-grained claim check-worthiness, which underpins all of these factors and provides probable human grounds for identifying a claim as check-worthy. We present CheckIt, a manually annotated large Twitter dataset for fine-grained claim check-worthiness. We benchmark our dataset against a unified approach, CheckMate, that jointly determines whether a claim is check-worthy and the factors that led to that conclusion. We compare our suggested system with several baseline systems. Finally, we report a thorough analysis of results and human assessment, validating the efficacy of integrating check-worthiness factors in detecting claims worth fact-checking.

{{</citation>}}


### (14/50) Code quality assessment using transformers (Mosleh Mahamud et al., 2023)

{{<citation>}}

Mosleh Mahamud, Isak Samsten. (2023)  
**Code quality assessment using transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SE, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.09264v1)  

---


**ABSTRACT**  
Automatically evaluate the correctness of programming assignments is rather straightforward using unit and integration tests. However, programming tasks can be solved in multiple ways, many of which, although correct, are inelegant. For instance, excessive branching, poor naming or repetitiveness make the code hard to understand and maintain. These subjective qualities of code are hard to automatically assess using current techniques. In this work we investigate the use of CodeBERT to automatically assign quality score to Java code. We experiment with different models and training paradigms. We explore the accuracy of the models on a novel dataset for code quality assessment. Finally, we assess the quality of the predictions using saliency maps. We find that code quality to some extent is predictable and that transformer based models using task adapted pre-training can solve the task more efficiently than other techniques.

{{</citation>}}


### (15/50) Can Large Language Models Understand Real-World Complex Instructions? (Qianyu He et al., 2023)

{{<citation>}}

Qianyu He, Jie Zeng, Wenhao Huang, Lina Chen, Jin Xiao, Qianxi He, Xunzhe Zhou, Lida Chen, Xintao Wang, Yuncheng Huang, Haoning Ye, Zihan Li, Shisong Chen, Yikai Zhang, Zhouhong Gu, Jiaqing Liang, Yanghua Xiao. (2023)  
**Can Large Language Models Understand Real-World Complex Instructions?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2309.09150v1)  

---


**ABSTRACT**  
Large language models (LLMs) can understand human instructions, showing their potential for pragmatic applications beyond traditional NLP tasks. However, they still struggle with complex instructions, which can be either complex task descriptions that require multiple tasks and constraints, or complex input that contains long context, noise, heterogeneous information and multi-turn format. Due to these features, LLMs often ignore semantic constraints from task descriptions, generate incorrect formats, violate length or sample count constraints, and be unfaithful to the input text. Existing benchmarks are insufficient to assess LLMs' ability to understand complex instructions, as they are close-ended and simple. To bridge this gap, we propose CELLO, a benchmark for evaluating LLMs' ability to follow complex instructions systematically. We design eight features for complex instructions and construct a comprehensive evaluation dataset from real-world scenarios. We also establish four criteria and develop corresponding metrics, as current ones are inadequate, biased or too strict and coarse-grained. We compare the performance of representative Chinese-oriented and English-oriented models in following complex instructions through extensive experiments. Resources of CELLO are publicly available at https://github.com/Abbey4799/CELLO.

{{</citation>}}


### (16/50) Contrastive Decoding Improves Reasoning in Large Language Models (Sean O'Brien et al., 2023)

{{<citation>}}

Sean O'Brien, Mike Lewis. (2023)  
**Contrastive Decoding Improves Reasoning in Large Language Models**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, LLaMA, Language Model, PaLM, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.09117v1)  

---


**ABSTRACT**  
We demonstrate that Contrastive Decoding -- a simple, computationally light, and training-free text generation method proposed by Li et al 2022 -- achieves large out-of-the-box improvements over greedy decoding on a variety of reasoning tasks. Originally shown to improve the perceived quality of long-form text generation, Contrastive Decoding searches for strings that maximize a weighted difference in likelihood between strong and weak models. We show that Contrastive Decoding leads LLaMA-65B to outperform LLaMA 2, GPT-3.5 and PaLM 2-L on the HellaSwag commonsense reasoning benchmark, and to outperform LLaMA 2, GPT-3.5 and PaLM-540B on the GSM8K math word reasoning benchmark, in addition to improvements on a collection of other tasks. Analysis suggests that Contrastive Decoding improves over existing methods by preventing some abstract reasoning errors, as well as by avoiding simpler modes such as copying sections of the input during chain-of-thought. Overall, Contrastive Decoding outperforms nucleus sampling for long-form generation and greedy decoding for reasoning tasks, making it a powerful general purpose method for generating text from language models.

{{</citation>}}


## cs.LG (4)



### (17/50) Mitigating Over-Smoothing and Over-Squashing using Augmentations of Forman-Ricci Curvature (Lukas Fesser et al., 2023)

{{<citation>}}

Lukas Fesser, Melanie Weber. (2023)  
**Mitigating Over-Smoothing and Over-Squashing using Augmentations of Forman-Ricci Curvature**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Augmentation, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.09384v1)  

---


**ABSTRACT**  
While Graph Neural Networks (GNNs) have been successfully leveraged for learning on graph-structured data across domains, several potential pitfalls have been described recently. Those include the inability to accurately leverage information encoded in long-range connections (over-squashing), as well as difficulties distinguishing the learned representations of nearby nodes with growing network depth (over-smoothing). An effective way to characterize both effects is discrete curvature: Long-range connections that underlie over-squashing effects have low curvature, whereas edges that contribute to over-smoothing have high curvature. This observation has given rise to rewiring techniques, which add or remove edges to mitigate over-smoothing and over-squashing. Several rewiring approaches utilizing graph characteristics, such as curvature or the spectrum of the graph Laplacian, have been proposed. However, existing methods, especially those based on curvature, often require expensive subroutines and careful hyperparameter tuning, which limits their applicability to large-scale graphs. Here we propose a rewiring technique based on Augmented Forman-Ricci curvature (AFRC), a scalable curvature notation, which can be computed in linear time. We prove that AFRC effectively characterizes over-smoothing and over-squashing effects in message-passing GNNs. We complement our theoretical results with experiments, which demonstrate that the proposed approach achieves state-of-the-art performance while significantly reducing the computational cost in comparison with other methods. Utilizing fundamental properties of discrete curvature, we propose effective heuristics for hyperparameters in curvature-based rewiring, which avoids expensive hyperparameter searches, further improving the scalability of the proposed approach.

{{</citation>}}


### (18/50) Unleashing the Power of Dynamic Mode Decomposition and Deep Learning for Rainfall Prediction in North-East India (Paleti Nikhil Chowdary et al., 2023)

{{<citation>}}

Paleti Nikhil Chowdary, Sathvika P, Pranav U, Rohan S, Sowmya V, Gopalakrishnan E A, Dhanya M. (2023)  
**Unleashing the Power of Dynamic Mode Decomposition and Deep Learning for Rainfall Prediction in North-East India**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, physics-ao-ph  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.09336v1)  

---


**ABSTRACT**  
Accurate rainfall forecasting is crucial for effective disaster preparedness and mitigation in the North-East region of India, which is prone to extreme weather events such as floods and landslides. In this study, we investigated the use of two data-driven methods, Dynamic Mode Decomposition (DMD) and Long Short-Term Memory (LSTM), for rainfall forecasting using daily rainfall data collected from India Meteorological Department in northeast region over a period of 118 years. We conducted a comparative analysis of these methods to determine their relative effectiveness in predicting rainfall patterns. Using historical rainfall data from multiple weather stations, we trained and validated our models to forecast future rainfall patterns. Our results indicate that both DMD and LSTM are effective in forecasting rainfall, with LSTM outperforming DMD in terms of accuracy, revealing that LSTM has the ability to capture complex nonlinear relationships in the data, making it a powerful tool for rainfall forecasting. Our findings suggest that data-driven methods such as DMD and deep learning approaches like LSTM can significantly improve rainfall forecasting accuracy in the North-East region of India, helping to mitigate the impact of extreme weather events and enhance the region's resilience to climate change.

{{</citation>}}


### (19/50) MFRL-BI: Design of a Model-free Reinforcement Learning Process Control Scheme by Using Bayesian Inference (Yanrong Li et al., 2023)

{{<citation>}}

Yanrong Li, Juan Du, Wei Jiang. (2023)  
**MFRL-BI: Design of a Model-free Reinforcement Learning Process Control Scheme by Using Bayesian Inference**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.09205v1)  

---


**ABSTRACT**  
Design of process control scheme is critical for quality assurance to reduce variations in manufacturing systems. Taking semiconductor manufacturing as an example, extensive literature focuses on control optimization based on certain process models (usually linear models), which are obtained by experiments before a manufacturing process starts. However, in real applications, pre-defined models may not be accurate, especially for a complex manufacturing system. To tackle model inaccuracy, we propose a model-free reinforcement learning (MFRL) approach to conduct experiments and optimize control simultaneously according to real-time data. Specifically, we design a novel MFRL control scheme by updating the distribution of disturbances using Bayesian inference to reduce their large variations during manufacturing processes. As a result, the proposed MFRL controller is demonstrated to perform well in a nonlinear chemical mechanical planarization (CMP) process when the process model is unknown. Theoretical properties are also guaranteed when disturbances are additive. The numerical studies also demonstrate the effectiveness and efficiency of our methodology.

{{</citation>}}


### (20/50) Conditional Mutual Information Constrained Deep Learning for Classification (En-Hui Yang et al., 2023)

{{<citation>}}

En-Hui Yang, Shayan Mohajer Hamidi, Linfeng Ye, Renhao Tan, Beverly Yang. (2023)  
**Conditional Mutual Information Constrained Deep Learning for Classification**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.09123v1)  

---


**ABSTRACT**  
The concepts of conditional mutual information (CMI) and normalized conditional mutual information (NCMI) are introduced to measure the concentration and separation performance of a classification deep neural network (DNN) in the output probability distribution space of the DNN, where CMI and the ratio between CMI and NCMI represent the intra-class concentration and inter-class separation of the DNN, respectively. By using NCMI to evaluate popular DNNs pretrained over ImageNet in the literature, it is shown that their validation accuracies over ImageNet validation data set are more or less inversely proportional to their NCMI values. Based on this observation, the standard deep learning (DL) framework is further modified to minimize the standard cross entropy function subject to an NCMI constraint, yielding CMI constrained deep learning (CMIC-DL). A novel alternating learning algorithm is proposed to solve such a constrained optimization problem. Extensive experiment results show that DNNs trained within CMIC-DL outperform the state-of-the-art models trained within the standard DL and other loss functions in the literature in terms of both accuracy and robustness against adversarial attacks. In addition, visualizing the evolution of learning process through the lens of CMI and NCMI is also advocated.

{{</citation>}}


## physics.chem-ph (1)



### (21/50) Structure to Property: Chemical Element Embeddings and a Deep Learning Approach for Accurate Prediction of Chemical Properties (Shokirbek Shermukhamedov et al., 2023)

{{<citation>}}

Shokirbek Shermukhamedov, Dilorom Mamurjonova, Michael Probst. (2023)  
**Structure to Property: Chemical Element Embeddings and a Deep Learning Approach for Accurate Prediction of Chemical Properties**  

---
Primary Category: physics.chem-ph  
Categories: cond-mat-mtrl-sci, cs-LG, physics-atm-clus, physics-chem-ph, physics.chem-ph, q-bio-QM  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.09355v1)  

---


**ABSTRACT**  
The application of machine learning (ML) techniques in computational chemistry has led to significant advances in predicting molecular properties, accelerating drug discovery, and material design. ML models can extract hidden patterns and relationships from complex and large datasets, allowing for the prediction of various chemical properties with high accuracy. The use of such methods has enabled the discovery of molecules and materials that were previously difficult to identify. This paper introduces a new ML model based on deep learning techniques, such as a multilayer encoder and decoder architecture, for classification tasks. We demonstrate the opportunities offered by our approach by applying it to various types of input data, including organic and inorganic compounds. In particular, we developed and tested the model using the Matbench and Moleculenet benchmarks, which include crystal properties and drug design-related benchmarks. We also conduct a comprehensive analysis of vector representations of chemical compounds, shedding light on the underlying patterns in molecular data. The models used in this work exhibit a high degree of predictive power, underscoring the progress that can be made with refined machine learning when applied to molecular and material datasets. For instance, on the Tox21 dataset, we achieved an average accuracy of 96%, surpassing the previous best result by 10%. Our code is publicly available at https://github.com/dmamur/elembert.

{{</citation>}}


## astro-ph.EP (1)



### (22/50) Simulation-based Inference for Exoplanet Atmospheric Retrieval: Insights from winning the Ariel Data Challenge 2023 using Normalizing Flows (Mayeul Aubin et al., 2023)

{{<citation>}}

Mayeul Aubin, Carolina Cuesta-Lazaro, Ethan Tregidga, Javier Viaña, Cecilia Garraffo, Iouli E. Gordon, Mercedes López-Morales, Robert J. Hargreaves, Vladimir Yu. Makhnev, Jeremy J. Drake, Douglas P. Finkbeiner, Phillip Cargile. (2023)  
**Simulation-based Inference for Exoplanet Atmospheric Retrieval: Insights from winning the Ariel Data Challenge 2023 using Normalizing Flows**  

---
Primary Category: astro-ph.EP  
Categories: astro-ph-EP, astro-ph-IM, astro-ph.EP, cs-LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09337v1)  

---


**ABSTRACT**  
Advancements in space telescopes have opened new avenues for gathering vast amounts of data on exoplanet atmosphere spectra. However, accurately extracting chemical and physical properties from these spectra poses significant challenges due to the non-linear nature of the underlying physics.   This paper presents novel machine learning models developed by the AstroAI team for the Ariel Data Challenge 2023, where one of the models secured the top position among 293 competitors. Leveraging Normalizing Flows, our models predict the posterior probability distribution of atmospheric parameters under different atmospheric assumptions.   Moreover, we introduce an alternative model that exhibits higher performance potential than the winning model, despite scoring lower in the challenge. These findings highlight the need to reevaluate the evaluation metric and prompt further exploration of more efficient and accurate approaches for exoplanet atmosphere spectra analysis.   Finally, we present recommendations to enhance the challenge and models, providing valuable insights for future applications on real observational data. These advancements pave the way for more effective and timely analysis of exoplanet atmospheric properties, advancing our understanding of these distant worlds.

{{</citation>}}


## cs.SD (1)



### (23/50) A Few-Shot Approach to Dysarthric Speech Intelligibility Level Classification Using Transformers (Paleti Nikhil Chowdary et al., 2023)

{{<citation>}}

Paleti Nikhil Chowdary, Vadlapudi Sai Aravind, Gorantla V N S L Vishnu Vardhan, Menta Sai Akshay, Menta Sai Aashish, Jyothish Lal. G. (2023)  
**A Few-Shot Approach to Dysarthric Speech Intelligibility Level Classification Using Transformers**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Few-Shot, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.09329v1)  

---


**ABSTRACT**  
Dysarthria is a speech disorder that hinders communication due to difficulties in articulating words. Detection of dysarthria is important for several reasons as it can be used to develop a treatment plan and help improve a person's quality of life and ability to communicate effectively. Much of the literature focused on improving ASR systems for dysarthric speech. The objective of the current work is to develop models that can accurately classify the presence of dysarthria and also give information about the intelligibility level using limited data by employing a few-shot approach using a transformer model. This work also aims to tackle the data leakage that is present in previous studies. Our whisper-large-v2 transformer model trained on a subset of the UASpeech dataset containing medium intelligibility level patients achieved an accuracy of 85%, precision of 0.92, recall of 0.8 F1-score of 0.85, and specificity of 0.91. Experimental results also demonstrate that the model trained using the 'words' dataset performed better compared to the model trained on the 'letters' and 'digits' dataset. Moreover, the multiclass model achieved an accuracy of 67%.

{{</citation>}}


## cs.CV (11)



### (24/50) Active Learning for Semantic Segmentation with Multi-class Label Query (Sehyun Hwang et al., 2023)

{{<citation>}}

Sehyun Hwang, Sohyun Lee, Hoyoung Kim, Minhyeon Oh, Jungseul Ok, Suha Kwak. (2023)  
**Active Learning for Semantic Segmentation with Multi-class Label Query**  

---
Primary Category: cs.CV  
Categories: 68T07, I-2-10, cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Active Learning, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.09319v1)  

---


**ABSTRACT**  
This paper proposes a new active learning method for semantic segmentation. The core of our method lies in a new annotation query design. It samples informative local image regions (e.g., superpixels), and for each of such regions, asks an oracle for a multi-hot vector indicating all classes existing in the region. This multi-class labeling strategy is substantially more efficient than existing ones like segmentation, polygon, and even dominant class labeling in terms of annotation time per click. However, it introduces the class ambiguity issue in training since it assigns partial labels (i.e., a set of candidate classes) to individual pixels. We thus propose a new algorithm for learning semantic segmentation while disambiguating the partial labels in two stages. In the first stage, it trains a segmentation model directly with the partial labels through two new loss functions motivated by partial label learning and multiple instance learning. In the second stage, it disambiguates the partial labels by generating pixel-wise pseudo labels, which are used for supervised learning of the model. Equipped with a new acquisition function dedicated to the multi-class labeling, our method outperformed previous work on Cityscapes and PASCAL VOC 2012 while spending less annotation cost.

{{</citation>}}


### (25/50) Towards Debiasing Frame Length Bias in Text-Video Retrieval via Causal Intervention (Burak Satar et al., 2023)

{{<citation>}}

Burak Satar, Hongyuan Zhu, Hanwang Zhang, Joo Hwee Lim. (2023)  
**Towards Debiasing Frame Length Bias in Text-Video Retrieval via Causal Intervention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-IR, cs-MM, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.09311v1)  

---


**ABSTRACT**  
Many studies focus on improving pretraining or developing new backbones in text-video retrieval. However, existing methods may suffer from the learning and inference bias issue, as recent research suggests in other text-video-related tasks. For instance, spatial appearance features on action recognition or temporal object co-occurrences on video scene graph generation could induce spurious correlations. In this work, we present a unique and systematic study of a temporal bias due to frame length discrepancy between training and test sets of trimmed video clips, which is the first such attempt for a text-video retrieval task, to the best of our knowledge. We first hypothesise and verify the bias on how it would affect the model illustrated with a baseline study. Then, we propose a causal debiasing approach and perform extensive experiments and ablation studies on the Epic-Kitchens-100, YouCook2, and MSR-VTT datasets. Our model overpasses the baseline and SOTA on nDCG, a semantic-relevancy-focused evaluation metric which proves the bias is mitigated, as well as on the other conventional metrics.

{{</citation>}}


### (26/50) Effective Image Tampering Localization via Enhanced Transformer and Co-attention Fusion (Kun Guo et al., 2023)

{{<citation>}}

Kun Guo, Haochen Zhu, Gang Cao. (2023)  
**Effective Image Tampering Localization via Enhanced Transformer and Co-attention Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.09306v1)  

---


**ABSTRACT**  
Powerful manipulation techniques have made digital image forgeries be easily created and widespread without leaving visual anomalies. The blind localization of tampered regions becomes quite significant for image forensics. In this paper, we propose an effective image tampering localization network (EITLNet) based on a two-branch enhanced transformer encoder with attention-based feature fusion. Specifically, a feature enhancement module is designed to enhance the feature representation ability of the transformer encoder. The features extracted from RGB and noise streams are fused effectively by the coordinate attention-based fusion module at multiple scales. Extensive experimental results verify that the proposed scheme achieves the state-of-the-art generalization ability and robustness in various benchmark datasets. Code will be public at https://github.com/multimediaFor/EITLNet.

{{</citation>}}


### (27/50) Chasing Day and Night: Towards Robust and Efficient All-Day Object Detection Guided by an Event Camera (Jiahang Cao et al., 2023)

{{<citation>}}

Jiahang Cao, Xu Zheng, Yuanhuiyi Lyu, Jiaxu Wang, Renjing Xu, Lin Wang. (2023)  
**Chasing Day and Night: Towards Robust and Efficient All-Day Object Detection Guided by an Event Camera**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Attention, Object Detection  
[Paper Link](http://arxiv.org/abs/2309.09297v1)  

---


**ABSTRACT**  
The ability to detect objects in all lighting (i.e., normal-, over-, and under-exposed) conditions is crucial for real-world applications, such as self-driving.Traditional RGB-based detectors often fail under such varying lighting conditions.Therefore, recent works utilize novel event cameras to supplement or guide the RGB modality; however, these methods typically adopt asymmetric network structures that rely predominantly on the RGB modality, resulting in limited robustness for all-day detection. In this paper, we propose EOLO, a novel object detection framework that achieves robust and efficient all-day detection by fusing both RGB and event modalities. Our EOLO framework is built based on a lightweight spiking neural network (SNN) to efficiently leverage the asynchronous property of events. Buttressed by it, we first introduce an Event Temporal Attention (ETA) module to learn the high temporal information from events while preserving crucial edge information. Secondly, as different modalities exhibit varying levels of importance under diverse lighting conditions, we propose a novel Symmetric RGB-Event Fusion (SREF) module to effectively fuse RGB-Event features without relying on a specific modality, thus ensuring a balanced and adaptive fusion for all-day detection. In addition, to compensate for the lack of paired RGB-Event datasets for all-day training and evaluation, we propose an event synthesis approach based on the randomized optical flow that allows for directly generating the event frame from a single exposure image. We further build two new datasets, E-MSCOCO and E-VOC based on the popular benchmarks MSCOCO and PASCAL VOC. Extensive experiments demonstrate that our EOLO outperforms the state-of-the-art detectors,e.g.,RENet,by a substantial margin (+3.74% mAP50) in all lighting conditions.Our code and datasets will be available at https://vlislab22.github.io/EOLO/

{{</citation>}}


### (28/50) MVP: Meta Visual Prompt Tuning for Few-Shot Remote Sensing Image Scene Classification (Junjie Zhu et al., 2023)

{{<citation>}}

Junjie Zhu, Yiying Li, Chunping Qiu, Ke Yang, Naiyang Guan, Xiaodong Yi. (2023)  
**MVP: Meta Visual Prompt Tuning for Few-Shot Remote Sensing Image Scene Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Few-Shot, Transformer  
[Paper Link](http://arxiv.org/abs/2309.09276v1)  

---


**ABSTRACT**  
Vision Transformer (ViT) models have recently emerged as powerful and versatile models for various visual tasks. Recently, a work called PMF has achieved promising results in few-shot image classification by utilizing pre-trained vision transformer models. However, PMF employs full fine-tuning for learning the downstream tasks, leading to significant overfitting and storage issues, especially in the remote sensing domain. In order to tackle these issues, we turn to the recently proposed parameter-efficient tuning methods, such as VPT, which updates only the newly added prompt parameters while keeping the pre-trained backbone frozen. Inspired by VPT, we propose the Meta Visual Prompt Tuning (MVP) method. Specifically, we integrate the VPT method into the meta-learning framework and tailor it to the remote sensing domain, resulting in an efficient framework for Few-Shot Remote Sensing Scene Classification (FS-RSSC). Furthermore, we introduce a novel data augmentation strategy based on patch embedding recombination to enhance the representation and diversity of scenes for classification purposes. Experiment results on the FS-RSSC benchmark demonstrate the superior performance of the proposed MVP over existing methods in various settings, such as various-way-various-shot, various-way-one-shot, and cross-domain adaptation.

{{</citation>}}


### (29/50) Deep Neighbor Layer Aggregation for Lightweight Self-Supervised Monocular Depth Estimation (Boya Wang et al., 2023)

{{<citation>}}

Boya Wang, Shuo Wang, Ziwen Dou, Dong Ye. (2023)  
**Deep Neighbor Layer Aggregation for Lightweight Self-Supervised Monocular Depth Estimation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2309.09272v1)  

---


**ABSTRACT**  
With the frequent use of self-supervised monocular depth estimation in robotics and autonomous driving, the model's efficiency is becoming increasingly important. Most current approaches apply much larger and more complex networks to improve the precision of depth estimation. Some researchers incorporated Transformer into self-supervised monocular depth estimation to achieve better performance. However, this method leads to high parameters and high computation. We present a fully convolutional depth estimation network using contextual feature fusion. Compared to UNet++ and HRNet, we use high-resolution and low-resolution features to reserve information on small targets and fast-moving objects instead of long-range fusion. We further promote depth estimation results employing lightweight channel attention based on convolution in the decoder stage. Our method reduces the parameters without sacrificing accuracy. Experiments on the KITTI benchmark show that our method can get better results than many large models, such as Monodepth2, with only 30 parameters. The source code is available at https://github.com/boyagesmile/DNA-Depth.

{{</citation>}}


### (30/50) LiteTrack: Layer Pruning with Asynchronous Feature Extraction for Lightweight and Efficient Visual Tracking (Qingmao Wei et al., 2023)

{{<citation>}}

Qingmao Wei, Bi Zeng, Jianqi Liu, Li He, Guotian Zeng. (2023)  
**LiteTrack: Layer Pruning with Asynchronous Feature Extraction for Lightweight and Efficient Visual Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2309.09249v1)  

---


**ABSTRACT**  
The recent advancements in transformer-based visual trackers have led to significant progress, attributed to their strong modeling capabilities. However, as performance improves, running latency correspondingly increases, presenting a challenge for real-time robotics applications, especially on edge devices with computational constraints. In response to this, we introduce LiteTrack, an efficient transformer-based tracking model optimized for high-speed operations across various devices. It achieves a more favorable trade-off between accuracy and efficiency than the other lightweight trackers. The main innovations of LiteTrack encompass: 1) asynchronous feature extraction and interaction between the template and search region for better feature fushion and cutting redundant computation, and 2) pruning encoder layers from a heavy tracker to refine the balnace between performance and speed. As an example, our fastest variant, LiteTrack-B4, achieves 65.2% AO on the GOT-10k benchmark, surpassing all preceding efficient trackers, while running over 100 fps with ONNX on the Jetson Orin NX edge device. Moreover, our LiteTrack-B9 reaches competitive 72.2% AO on GOT-10k and 82.4% AUC on TrackingNet, and operates at 171 fps on an NVIDIA 2080Ti GPU. The code and demo materials will be available at https://github.com/TsingWei/LiteTrack.

{{</citation>}}


### (31/50) Efficient Pyramid Channel Attention Network for Pathological Myopia Detection (Xiaoqing Zhang et al., 2023)

{{<citation>}}

Xiaoqing Zhang, Jilu Zhao, Richu Jin, Yan Li, Hao Wu, Xiangtian Zhou, Jiang Liu. (2023)  
**Efficient Pyramid Channel Attention Network for Pathological Myopia Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.09196v1)  

---


**ABSTRACT**  
Pathological myopia (PM) is the leading ocular disease for impaired vision and blindness worldwide. The key to detecting PM as early as possible is to detect informative features in global and local lesion regions, such as fundus tessellation, atrophy and maculopathy. However, applying classical convolutional neural networks (CNNs) to efficiently highlight global and local lesion context information in feature maps is quite challenging. To tackle this issue, we aim to fully leverage the potential of global and local lesion information with attention module design. Based on this, we propose an efficient pyramid channel attention (EPCA) module, which dynamically explores the relative importance of global and local lesion context information in feature maps. Then we combine the EPCA module with the backbone network to construct EPCA-Net for automatic PM detection based on fundus images. In addition, we construct a PM dataset termed PM-fundus by collecting fundus images of PM from publicly available datasets (e.g., the PALM dataset and ODIR dataset). The comprehensive experiments are conducted on three datasets, demonstrating that our EPCA-Net outperforms state-of-the-art methods in detecting PM. Furthermore, motivated by the recent pretraining-and-finetuning paradigm, we attempt to adapt pre-trained natural image models for PM detection by freezing them and treating the EPCA module and other attention modules as the adapters. The results show that our method with the pretraining-and-finetuning paradigm achieves competitive performance through comparisons to part of methods with traditional fine-tuning methods with fewer tunable parameters.

{{</citation>}}


### (32/50) Syntax Tree Constrained Graph Network for Visual Question Answering (Xiangrui Su et al., 2023)

{{<citation>}}

Xiangrui Su, Qi Zhang, Chongyang Shi, Jiachang Liu, Liang Hu. (2023)  
**Syntax Tree Constrained Graph Network for Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.09179v1)  

---


**ABSTRACT**  
Visual Question Answering (VQA) aims to automatically answer natural language questions related to given image content. Existing VQA methods integrate vision modeling and language understanding to explore the deep semantics of the question. However, these methods ignore the significant syntax information of the question, which plays a vital role in understanding the essential semantics of the question and guiding the visual feature refinement. To fill the gap, we suggested a novel Syntax Tree Constrained Graph Network (STCGN) for VQA based on entity message passing and syntax tree. This model is able to extract a syntax tree from questions and obtain more precise syntax information. Specifically, we parse questions and obtain the question syntax tree using the Stanford syntax parsing tool. From the word level and phrase level, syntactic phrase features and question features are extracted using a hierarchical tree convolutional network. We then design a message-passing mechanism for phrase-aware visual entities and capture entity features according to a given visual context. Extensive experiments on VQA2.0 datasets demonstrate the superiority of our proposed model.

{{</citation>}}


### (33/50) FDCNet: Feature Drift Compensation Network for Class-Incremental Weakly Supervised Object Localization (Sejin Park et al., 2023)

{{<citation>}}

Sejin Park, Taehyung Lee, Yeejin Lee, Byeongkeun Kang. (2023)  
**FDCNet: Feature Drift Compensation Network for Class-Incremental Weakly Supervised Object Localization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.09122v1)  

---


**ABSTRACT**  
This work addresses the task of class-incremental weakly supervised object localization (CI-WSOL). The goal is to incrementally learn object localization for novel classes using only image-level annotations while retaining the ability to localize previously learned classes. This task is important because annotating bounding boxes for every new incoming data is expensive, although object localization is crucial in various applications. To the best of our knowledge, we are the first to address this task. Thus, we first present a strong baseline method for CI-WSOL by adapting the strategies of class-incremental classifiers to mitigate catastrophic forgetting. These strategies include applying knowledge distillation, maintaining a small data set from previous tasks, and using cosine normalization. We then propose the feature drift compensation network to compensate for the effects of feature drifts on class scores and localization maps. Since updating network parameters to learn new tasks causes feature drifts, compensating for the final outputs is necessary. Finally, we evaluate our proposed method by conducting experiments on two publicly available datasets (ImageNet-100 and CUB-200). The experimental results demonstrate that the proposed method outperforms other baseline methods.

{{</citation>}}


### (34/50) Uncertainty-aware 3D Object-Level Mapping with Deep Shape Priors (Ziwei Liao et al., 2023)

{{<citation>}}

Ziwei Liao, Jun Yang, Jingxing Qian, Angela P. Schoellig, Steven L. Waslander. (2023)  
**Uncertainty-aware 3D Object-Level Mapping with Deep Shape Priors**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09118v1)  

---


**ABSTRACT**  
3D object-level mapping is a fundamental problem in robotics, which is especially challenging when object CAD models are unavailable during inference. In this work, we propose a framework that can reconstruct high-quality object-level maps for unknown objects. Our approach takes multiple RGB-D images as input and outputs dense 3D shapes and 9-DoF poses (including 3 scale parameters) for detected objects. The core idea of our approach is to leverage a learnt generative model for shape categories as a prior and to formulate a probabilistic, uncertainty-aware optimization framework for 3D reconstruction. We derive a probabilistic formulation that propagates shape and pose uncertainty through two novel loss functions. Unlike current state-of-the-art approaches, we explicitly model the uncertainty of the object shapes and poses during our optimization, resulting in a high-quality object-level mapping system. Moreover, the resulting shape and pose uncertainties, which we demonstrate can accurately reflect the true errors of our object maps, can also be useful for downstream robotics tasks such as active vision. We perform extensive evaluations on indoor and outdoor real-world datasets, achieving achieves substantial improvements over state-of-the-art methods. Our code will be available at https://github.com/TRAILab/UncertainShapePose.

{{</citation>}}


## cs.SE (4)



### (35/50) GAMMA: Revisiting Template-based Automated Program Repair via Mask Prediction (Quanjun Zhang et al., 2023)

{{<citation>}}

Quanjun Zhang, Chunrong Fang, Tongke Zhang, Bowen Yu, Weisong Sun, Zhenyu Chen. (2023)  
**GAMMA: Revisiting Template-based Automated Program Repair via Mask Prediction**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BERT, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.09308v1)  

---


**ABSTRACT**  
Automated program repair (APR) aims to fix software bugs without human intervention and template-based APR has been widely investigated with promising results. However, it is challenging for template-based APR to select the appropriate donor code, which is an important repair ingredient for generating candidate patches. Inappropriate donor code may cause plausible but incorrect patch generation even with correct fix patterns, limiting the repair performance.   In this paper, we aim to revisit template-based APR, and propose GAMMA, to directly leverage large pre-trained language models for donor code generation. Our main insight is that instead of retrieving donor code in the local buggy file, we can directly predict the correct code tokens based on the context code snippets and repair patterns by a cloze task. Specifically, (1) GAMMA revises a variety of fix templates from state-of-the-art template-based APR techniques (i.e., TBar) and transforms them into mask patterns. (2) GAMMA adopts a pre-trained language model to predict the correct code for masked code as a fill-in-the-blank task. The experimental results demonstrate that GAMMA correctly repairs 82 bugs on Defects4J-v1.2, which achieves 20.59\% (14 bugs) and 26.15\% (17 bugs) improvement over the previous state-of-the-art template-based approach TBar and learning-based one Recoder. Furthermore, GAMMA repairs 45 bugs and 22 bugs from the additional Defects4J-v2.0 and QuixBugs, indicating the generalizability of GAMMA in addressing the dataset overfitting issue. We also prove that adopting other pre-trained language models can provide substantial advancement, e.g., CodeBERT-based and ChatGPT-based GAMMA is able to fix 80 and 67 bugs on Defects4J-v1.2, indicating the scalability of GAMMA. Overall, our study highlights the promising future of adopting pre-trained models to generate correct patches on top of fix patterns.

{{</citation>}}


### (36/50) Embedded Software Development with Digital Twins: Specific Requirements for Small and Medium-Sized Enterprises (Alexander Barbie et al., 2023)

{{<citation>}}

Alexander Barbie, Wilhelm Hasselbring. (2023)  
**Embedded Software Development with Digital Twins: Specific Requirements for Small and Medium-Sized Enterprises**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.09216v1)  

---


**ABSTRACT**  
The transformation to Industry 4.0 changes the way embedded software systems are developed. Digital twins have the potential for cost-effective software development and maintenance strategies. With reduced costs and faster development cycles, small and medium-sized enterprises (SME) have the chance to grow with new smart products. We interviewed SMEs about their current development processes. In this paper, we present the first results of these interviews. First results show that real-time requirements prevent, to date, a Software-in-the-Loop development approach, due to a lack of proper tooling. Security/safety concerns, and the accessibility of hardware are the main impediments. Only temporary access to the hardware leads to Software-in-the-Loop development approaches based on simulations/emulators. Yet, this is not in all use cases possible. All interviewees see the potential of Software-in-the-Loop approaches and digital twins with regard to quality and customization. One reason it will take some effort to convince engineers, is the conservative nature of the embedded community, particularly in SMEs.

{{</citation>}}


### (37/50) Rely-guarantee Reasoning about Concurrent Reactive Systems: The PiCore Framework, Languages Integration and Applications (Yongwang Zhao et al., 2023)

{{<citation>}}

Yongwang Zhao, David Sanan. (2023)  
**Rely-guarantee Reasoning about Concurrent Reactive Systems: The PiCore Framework, Languages Integration and Applications**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2309.09148v1)  

---


**ABSTRACT**  
The rely-guarantee approach is a promising way for compositional verification of concurrent reactive systems (CRSs), e.g. concurrent operating systems, interrupt-driven control systems and business process systems. However, specifications using heterogeneous reaction patterns, different abstraction levels, and the complexity of real-world CRSs are still challenging the rely-guarantee approach. This article proposes PiCore, a rely-guarantee reasoning framework for formal specification and verification of CRSs. We design an event specification language supporting complex reaction structures and its rely-guarantee proof system to detach the specification and logic of reactive aspects of CRSs from event behaviours. PiCore parametrizes the language and its rely-guarantee system for event behaviour using a rely-guarantee interface and allows to easily integrate 3rd-party languages via rely-guarantee adapters. By this design, we have successfully integrated two existing languages and their rely-guarantee proof systems without any change of their specification and proofs. PiCore has been applied to two real-world case studies, i.e. formal verification of concurrent memory management in Zephyr RTOS and a verified translation for a standardized Business Process Execution Language (BPEL) to PiCore.

{{</citation>}}


### (38/50) Event-based Compositional Reasoning of Information-Flow Security for Concurrent Systems (Yongwang Zhao et al., 2023)

{{<citation>}}

Yongwang Zhao, David Sanan, Fuyuan Zhang, Yang Liu. (2023)  
**Event-based Compositional Reasoning of Information-Flow Security for Concurrent Systems**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Reasoning, Security  
[Paper Link](http://arxiv.org/abs/2309.09141v1)  

---


**ABSTRACT**  
High assurance of information-flow security (IFS) for concurrent systems is challenging. A promising way for formal verification of concurrent systems is the rely-guarantee method. However, existing compositional reasoning approaches for IFS concentrate on language-based IFS. It is often not applicable for system-level security, such as multicore operating system kernels, in which secrecy of actions should also be considered. On the other hand, existing studies on the rely-guarantee method are basically built on concurrent programming languages, by which semantics of concurrent systems cannot be completely captured in a straightforward way. In order to formally verify state-action based IFS for concurrent systems, we propose a rely-guarantee-based compositional reasoning approach for IFS in this paper. We first design a language by incorporating ``Event'' into concurrent languages and give the IFS semantics of the language. As a primitive element, events offer an extremely neat framework for modeling system and are not necessarily atomic in our language. For compositional reasoning of IFS, we use rely-guarantee specification to define new forms of unwinding conditions (UCs) on events, i.e., event UCs. By a rely-guarantee proof system of the language and the soundness of event UCs, we have that event UCs imply IFS of concurrent systems. In such a way, we relax the atomicity constraint of actions in traditional UCs and provide a compositional reasoning way for IFS in which security proof of systems can be discharged by independent security proof on individual events. Finally, we mechanize the approach in Isabelle/HOL and develop a formal specification and its IFS proof for multicore separation kernels as a study case according to an industrial standard -- ARINC 653.

{{</citation>}}


## eess.AS (3)



### (39/50) PromptVC: Flexible Stylistic Voice Conversion in Latent Space Driven by Natural Language Prompts (Jixun Yao et al., 2023)

{{<citation>}}

Jixun Yao, Yuguang Yang, Yi Lei, Ziqian Ning, Yanni Hu, Yu Pan, Jingjing Yin, Hongbin Zhou, Heng Lu, Lei Xie. (2023)  
**PromptVC: Flexible Stylistic Voice Conversion in Latent Space Driven by Natural Language Prompts**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.09262v1)  

---


**ABSTRACT**  
Style voice conversion aims to transform the style of source speech to a desired style according to real-world application demands. However, the current style voice conversion approach relies on pre-defined labels or reference speech to control the conversion process, which leads to limitations in style diversity or falls short in terms of the intuitive and interpretability of style representation. In this study, we propose PromptVC, a novel style voice conversion approach that employs a latent diffusion model to generate a style vector driven by natural language prompts. Specifically, the style vector is extracted by a style encoder during training, and then the latent diffusion model is trained independently to sample the style vector from noise, with this process being conditioned on natural language prompts. To improve style expressiveness, we leverage HuBERT to extract discrete tokens and replace them with the K-Means center embedding to serve as the linguistic content, which minimizes residual style information. Additionally, we deduplicate the same discrete token and employ a differentiable duration predictor to re-predict the duration of each token, which can adapt the duration of the same linguistic content to different styles. The subjective and objective evaluation results demonstrate the effectiveness of our proposed system.

{{</citation>}}


### (40/50) Improving Speech Inversion Through Self-Supervised Embeddings and Enhanced Tract Variables (Ahmed Adel Attia et al., 2023)

{{<citation>}}

Ahmed Adel Attia, Yashish M. Siriwardena, Carol Espy-Wilson. (2023)  
**Improving Speech Inversion Through Self-Supervised Embeddings and Enhanced Tract Variables**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-SD, eess-AS, eess.AS  
Keywords: BERT, Embedding, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.09220v1)  

---


**ABSTRACT**  
The performance of deep learning models depends significantly on their capacity to encode input features efficiently and decode them into meaningful outputs. Better input and output representation has the potential to boost models' performance and generalization. In the context of acoustic-to-articulatory speech inversion (SI) systems, we study the impact of utilizing speech representations acquired via self-supervised learning (SSL) models, such as HuBERT compared to conventional acoustic features. Additionally, we investigate the incorporation of novel tract variables (TVs) through an improved geometric transformation model. By combining these two approaches, we improve the Pearson product-moment correlation (PPMC) scores which evaluate the accuracy of TV estimation of the SI system from 0.7452 to 0.8141, a 6.9% increase. Our findings underscore the profound influence of rich feature representations from SSL models and improved geometric transformations with target TVs on the enhanced functionality of SI systems.

{{</citation>}}


### (41/50) Neural Speaker Diarization Using Memory-Aware Multi-Speaker Embedding with Sequence-to-Sequence Architecture (Gaobin Yang et al., 2023)

{{<citation>}}

Gaobin Yang, Maokui He, Shutong Niu, Ruoyu Wang, Yanyan Yue, Shuangqing Qian, Shilong Wu, Jun Du, Chin-Hui Lee. (2023)  
**Neural Speaker Diarization Using Memory-Aware Multi-Speaker Embedding with Sequence-to-Sequence Architecture**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-SD, eess-AS, eess.AS  
Keywords: Embedding, Seq2Seq, Sequence-to-Sequence  
[Paper Link](http://arxiv.org/abs/2309.09180v1)  

---


**ABSTRACT**  
We propose a novel neural speaker diarization system using memory-aware multi-speaker embedding with sequence-to-sequence architecture (NSD-MS2S), which integrates the strengths of memory-aware multi-speaker embedding (MA-MSE) and sequence-to-sequence (Seq2Seq) architecture, leading to improvement in both efficiency and performance. Next, we further decrease the memory occupation of decoding by incorporating input features fusion and then employ a multi-head attention mechanism to capture features at different levels. NSD-MS2S achieved a macro diarization error rate (DER) of 15.9% on the CHiME-7 EVAL set, which signifies a relative improvement of 49% over the official baseline system, and is the key technique for us to achieve the best performance for the main track of CHiME-7 DASR Challenge. Additionally, we introduce a deep interactive module (DIM) in MA-MSE module to better retrieve a cleaner and more discriminative multi-speaker embedding, enabling the current model to outperform the system we used in the CHiME-7 DASR Challenge. Our code will be available at https://github.com/liyunlongaaa/NSD-MS2S.

{{</citation>}}


## cs.IR (1)



### (42/50) Leveraging Large Language Models for Sequential Recommendation (Jesse Harte et al., 2023)

{{<citation>}}

Jesse Harte, Wouter Zorgdrager, Panos Louridas, Asterios Katsifodimos, Dietmar Jannach, Marios Fragkoulis. (2023)  
**Leveraging Large Language Models for Sequential Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: AI, BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.09261v1)  

---


**ABSTRACT**  
Sequential recommendation problems have received increasing attention in research during the past few years, leading to the inception of a large variety of algorithmic approaches. In this work, we explore how large language models (LLMs), which are nowadays introducing disruptive effects in many AI-based applications, can be used to build or improve sequential recommendation approaches. Specifically, we devise and evaluate three approaches to leverage the power of LLMs in different ways. Our results from experiments on two datasets show that initializing the state-of-the-art sequential recommendation model BERT4Rec with embeddings obtained from an LLM improves NDCG by 15-20% compared to the vanilla BERT4Rec model. Furthermore, we find that a simple approach that leverages LLM embeddings for producing recommendations, can provide competitive performance by highlighting semantically related items. We publicly share the code and data of our experiments to ensure reproducibility.

{{</citation>}}


## cs.RO (4)



### (43/50) Sim-to-Real Deep Reinforcement Learning with Manipulators for Pick-and-place (Wenxing Liu et al., 2023)

{{<citation>}}

Wenxing Liu, Hanlin Niu, Robert Skilton, Joaquin Carrasco. (2023)  
**Sim-to-Real Deep Reinforcement Learning with Manipulators for Pick-and-place**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.09247v1)  

---


**ABSTRACT**  
When transferring a Deep Reinforcement Learning model from simulation to the real world, the performance could be unsatisfactory since the simulation cannot imitate the real world well in many circumstances. This results in a long period of fine-tuning in the real world. This paper proposes a self-supervised vision-based DRL method that allows robots to pick and place objects effectively and efficiently when directly transferring a training model from simulation to the real world. A height-sensitive action policy is specially designed for the proposed method to deal with crowded and stacked objects in challenging environments. The training model with the proposed approach can be applied directly to a real suction task without any fine-tuning from the real world while maintaining a high suction success rate. It is also validated that our model can be deployed to suction novel objects in a real experiment with a suction success rate of 90\% without any real-world fine-tuning. The experimental video is available at: https://youtu.be/jSTC-EGsoFA.

{{</citation>}}


### (44/50) Optimal Scene Graph Planning with Large Language Model Guidance (Zhirui Dai et al., 2023)

{{<citation>}}

Zhirui Dai, Arash Asgharivaskasi, Thai Duong, Shusen Lin, Maria-Elizabeth Tzes, George Pappas, Nikolay Atanasov. (2023)  
**Optimal Scene Graph Planning with Large Language Model Guidance**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.09182v1)  

---


**ABSTRACT**  
Recent advances in metric, semantic, and topological mapping have equipped autonomous robots with semantic concept grounding capabilities to interpret natural language tasks. This work aims to leverage these new capabilities with an efficient task planning algorithm for hierarchical metric-semantic models. We consider a scene graph representation of the environment and utilize a large language model (LLM) to convert a natural language task into a linear temporal logic (LTL) automaton. Our main contribution is to enable optimal hierarchical LTL planning with LLM guidance over scene graphs. To achieve efficiency, we construct a hierarchical planning domain that captures the attributes and connectivity of the scene graph and the task automaton, and provide semantic guidance via an LLM heuristic function. To guarantee optimality, we design an LTL heuristic function that is provably consistent and supplements the potentially inadmissible LLM guidance in multi-heuristic planning. We demonstrate efficient planning of complex natural language tasks in scene graphs of virtualized real environments.

{{</citation>}}


### (45/50) From Cooking Recipes to Robot Task Trees -- Improving Planning Correctness and Task Efficiency by Leveraging LLMs with a Knowledge Network (Md Sadman Sakib et al., 2023)

{{<citation>}}

Md Sadman Sakib, Yu Sun. (2023)  
**From Cooking Recipes to Robot Task Trees -- Improving Planning Correctness and Task Efficiency by Leveraging LLMs with a Knowledge Network**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2309.09181v1)  

---


**ABSTRACT**  
Task planning for robotic cooking involves generating a sequence of actions for a robot to prepare a meal successfully. This paper introduces a novel task tree generation pipeline producing correct planning and efficient execution for cooking tasks. Our method first uses a large language model (LLM) to retrieve recipe instructions and then utilizes a fine-tuned GPT-3 to convert them into a task tree, capturing sequential and parallel dependencies among subtasks. The pipeline then mitigates the uncertainty and unreliable features of LLM outputs using task tree retrieval. We combine multiple LLM task tree outputs into a graph and perform a task tree retrieval to avoid questionable nodes and high-cost nodes to improve planning correctness and improve execution efficiency. Our evaluation results show its superior performance compared to previous works in task planning accuracy and efficiency.

{{</citation>}}


### (46/50) Trajectory Prediction for Robot Navigation using Flow-Guided Markov Neural Operator (Rashmi Bhaskara et al., 2023)

{{<citation>}}

Rashmi Bhaskara, Hrishikesh Viswanath, Aniket Bera. (2023)  
**Trajectory Prediction for Robot Navigation using Flow-Guided Markov Neural Operator**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.09137v1)  

---


**ABSTRACT**  
Predicting pedestrian movements remains a complex and persistent challenge in robot navigation research. We must evaluate several factors to achieve accurate predictions, such as pedestrian interactions, the environment, crowd density, and social and cultural norms. Accurate prediction of pedestrian paths is vital for ensuring safe human-robot interaction, especially in robot navigation. Furthermore, this research has potential applications in autonomous vehicles, pedestrian tracking, and human-robot collaboration. Therefore, in this paper, we introduce \textbf{FlowMNO}, an Optical Flow-Integrated Markov Neural Operator designed to capture pedestrian behavior across diverse scenarios. Our paper models trajectory prediction as a Markovian process, where future pedestrian coordinates depend solely on the current state. This problem formulation eliminates the need to store previous states. We conducted experiments using standard benchmark datasets like ETH, HOTEL, ZARA1, ZARA2, UCY, and RGB-D pedestrian datasets. Our study demonstrates that FlowMNO outperforms some of the state-of-the-art deep learning methods like LSTM, GAN, and CNN-based approaches, by approximately 86.46\% when predicting pedestrian trajectories. Thus, we show that FlowMNO can seamlessly integrate into robot navigation systems, enhancing their ability to navigate crowded areas smoothly.

{{</citation>}}


## cs.CR (1)



### (47/50) ATM: a Logic for Quantitative Security Properties on Attack Trees (Stefano M. Nicoletti et al., 2023)

{{<citation>}}

Stefano M. Nicoletti, Milan Lopuhaä-Zwakenberg, E. Moritz Hahn, Mariëlle Stoelinga. (2023)  
**ATM: a Logic for Quantitative Security Properties on Attack Trees**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LO, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.09231v1)  

---


**ABSTRACT**  
Critical infrastructure systems - for which high reliability and availability are paramount - must operate securely. Attack trees (ATs) are hierarchical diagrams that offer a flexible modelling language used to assess how systems can be attacked. ATs are widely employed both in industry and academia but - in spite of their popularity - little work has been done to give practitioners instruments to formulate queries on ATs in an understandable yet powerful way. In this paper we fill this gap by presenting ATM, a logic to express quantitative security properties on ATs. ATM allows for the specification of properties involved with security metrics that include "cost", "probability" and "skill" and permits the formulation of insightful what-if scenarios. To showcase its potential, we apply ATM to the case study of a CubeSAT, presenting three different ways in which an attacker can compromise its availability. We showcase property specification on the corresponding attack tree and we present theory and algorithms - based on binary decision diagrams - to check properties and compute metrics of ATM-formulae.

{{</citation>}}


## cs.MA (1)



### (48/50) Logic of Awareness in Agent's Reasoning (Yudai Kubono et al., 2023)

{{<citation>}}

Yudai Kubono, Teeradaj Racharak, Satoshi Tojo. (2023)  
**Logic of Awareness in Agent's Reasoning**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs.MA  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2309.09214v1)  

---


**ABSTRACT**  
The aim of this study is to formally express awareness for modeling practical agent communication. The notion of awareness has been proposed as a set of propositions for each agent, to which he/she pays attention, and has contributed to avoiding \textit{logical omniscience}. However, when an agent guesses another agent's knowledge states, what matters are not propositions but are accessible possible worlds. Therefore, we introduce a partition of possible worlds connected to awareness, that is an equivalence relation, to denote \textit{indistinguishable} worlds. Our logic is called Awareness Logic with Partition ($\mathcal{ALP}$). In this paper, we first show a running example to illustrate a practical social game. Thereafter, we introduce syntax and Kripke semantics of the logic and prove its completeness. Finally, we outline an idea to incorporate some epistemic actions with dynamic operators that change the state of awareness.

{{</citation>}}


## cs.ET (1)



### (49/50) Analog Content-Addressable Memory from Complementary FeFETs (Xiwen Liu et al., 2023)

{{<citation>}}

Xiwen Liu, Keshava Katti, Yunfei He, Paul Jacob, Claudia Richter, Uwe Schroeder, Santosh Kurinec, Pratik Chaudhari, Deep Jariwala. (2023)  
**Analog Content-Addressable Memory from Complementary FeFETs**  

---
Primary Category: cs.ET  
Categories: cs-AR, cs-ET, cs.ET, physics-app-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09165v1)  

---


**ABSTRACT**  
To address the increasing computational demands of artificial intelligence (AI) and big data, compute-in-memory (CIM) integrates memory and processing units into the same physical location, reducing the time and energy overhead of the system. Despite advancements in non-volatile memory (NVM) for matrix multiplication, other critical data-intensive operations, like parallel search, have been overlooked. Current parallel search architectures, namely content-addressable memory (CAM), often use binary, which restricts density and functionality. We present an analog CAM (ACAM) cell, built on two complementary ferroelectric field-effect transistors (FeFETs), that performs parallel search in the analog domain with over 40 distinct match windows. We then deploy it to calculate similarity between vectors, a building block in the following two machine learning problems. ACAM outperforms ternary CAM (TCAM) when applied to similarity search for few-shot learning on the Omniglot dataset, yielding projected simulation results with improved inference accuracy by 5%, 3x denser memory architecture, and more than 100x faster speed compared to central processing unit (CPU) and graphics processing unit (GPU) per similarity search on scaled CMOS nodes. We also demonstrate 1-step inference on a kernel regression model by combining non-linear kernel computation and matrix multiplication in ACAM, with simulation estimates indicating 1,000x faster inference than CPU and GPU.

{{</citation>}}


## cs.DC (1)



### (50/50) Performance of Graph Neural Networks for Point Cloud Applications (Dhruv Parikh et al., 2023)

{{<citation>}}

Dhruv Parikh, Bingyi Zhang, Rajgopal Kannan, Viktor Prasanna, Carl Busart. (2023)  
**Performance of Graph Neural Networks for Point Cloud Applications**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.09142v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have gained significant momentum recently due to their capability to learn on unstructured graph data. Dynamic GNNs (DGNNs) are the current state-of-the-art for point cloud applications; such applications (viz. autonomous driving) require real-time processing at the edge with tight latency and memory constraints. Conducting performance analysis on such DGNNs, thus, becomes a crucial task to evaluate network suitability.   This paper presents a profiling analysis of EdgeConv-based DGNNs applied to point cloud inputs. We assess their inference performance in terms of end-to-end latency and memory consumption on state-of-the-art CPU and GPU platforms. The EdgeConv layer has two stages: (1) dynamic graph generation using k-Nearest Neighbors (kNN) and, (2) node feature updation. The addition of dynamic graph generation via kNN in each (EdgeConv) layer enhances network performance compared to networks that work with the same static graph in each layer; such performance enhancement comes, however, at the added computational cost associated with the dynamic graph generation stage (via kNN algorithm). Understanding its costs is essential for identifying the performance bottleneck and exploring potential avenues for hardware acceleration. To this end, this paper aims to shed light on the performance characteristics of EdgeConv-based DGNNs for point cloud inputs. Our performance analysis on a state-of-the-art EdgeConv network for classification shows that the dynamic graph construction via kNN takes up upwards of 95% of network latency on the GPU and almost 90% on the CPU. Moreover, we propose a quasi-Dynamic Graph Neural Network (qDGNN) that halts dynamic graph updates after a specific depth within the network to significantly reduce the latency on both CPU and GPU whilst matching the original networks inference accuracy.

{{</citation>}}
