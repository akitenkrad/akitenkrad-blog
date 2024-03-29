---
draft: false
title: "arXiv @ 2023.07.07"
date: 2023-07-07
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.07"
    identifier: arxiv_20230707
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.RO (2)](#csro-2)
- [cs.CL (28)](#cscl-28)
- [cs.CV (24)](#cscv-24)
- [math.OC (1)](#mathoc-1)
- [cs.AR (1)](#csar-1)
- [cs.IT (2)](#csit-2)
- [cs.HC (2)](#cshc-2)
- [cs.AI (5)](#csai-5)
- [cs.LG (26)](#cslg-26)
- [cs.CR (3)](#cscr-3)
- [eess.IV (6)](#eessiv-6)
- [cs.SE (4)](#csse-4)
- [cs.CE (1)](#csce-1)
- [cs.NI (1)](#csni-1)
- [eess.AS (1)](#eessas-1)
- [cs.DB (2)](#csdb-2)
- [cs.IR (3)](#csir-3)
- [stat.CO (1)](#statco-1)
- [eess.SY (2)](#eesssy-2)

## cs.RO (2)



### (1/115) SACHA: Soft Actor-Critic with Heuristic-Based Attention for Partially Observable Multi-Agent Path Finding (Qiushi Lin et al., 2023)

{{<citation>}}

Qiushi Lin, Hang Ma. (2023)  
**SACHA: Soft Actor-Critic with Heuristic-Based Attention for Partially Observable Multi-Agent Path Finding**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-MA, cs-RO, cs.RO  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.02691v1)  

---


**ABSTRACT**  
Multi-Agent Path Finding (MAPF) is a crucial component for many large-scale robotic systems, where agents must plan their collision-free paths to their given goal positions. Recently, multi-agent reinforcement learning has been introduced to solve the partially observable variant of MAPF by learning a decentralized single-agent policy in a centralized fashion based on each agent's partial observation. However, existing learning-based methods are ineffective in achieving complex multi-agent cooperation, especially in congested environments, due to the non-stationarity of this setting. To tackle this challenge, we propose a multi-agent actor-critic method called Soft Actor-Critic with Heuristic-Based Attention (SACHA), which employs novel heuristic-based attention mechanisms for both the actors and critics to encourage cooperation among agents. SACHA learns a neural network for each agent to selectively pay attention to the shortest path heuristic guidance from multiple agents within its field of view, thereby allowing for more scalable learning of cooperation. SACHA also extends the existing multi-agent actor-critic framework by introducing a novel critic centered on each agent to approximate $Q$-values. Compared to existing methods that use a fully observable critic, our agent-centered multi-agent actor-critic method results in more impartial credit assignment and better generalizability of the learned policy to MAPF instances with varying numbers of agents and types of environments. We also implement SACHA(C), which embeds a communication module in the agent's policy network to enable information exchange among agents. We evaluate both SACHA and SACHA(C) on a variety of MAPF instances and demonstrate decent improvements over several state-of-the-art learning-based MAPF methods with respect to success rate and solution quality.

{{</citation>}}


### (2/115) Active Class Selection for Few-Shot Class-Incremental Learning (Christopher McClurg et al., 2023)

{{<citation>}}

Christopher McClurg, Ali Ayub, Harsh Tyagi, Sarah M. Rajtmajer, Alan R. Wagner. (2023)  
**Active Class Selection for Few-Shot Class-Incremental Learning**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2307.02641v1)  

---


**ABSTRACT**  
For real-world applications, robots will need to continually learn in their environments through limited interactions with their users. Toward this, previous works in few-shot class incremental learning (FSCIL) and active class selection (ACS) have achieved promising results but were tested in constrained setups. Therefore, in this paper, we combine ideas from FSCIL and ACS to develop a novel framework that can allow an autonomous agent to continually learn new objects by asking its users to label only a few of the most informative objects in the environment. To this end, we build on a state-of-the-art (SOTA) FSCIL model and extend it with techniques from ACS literature. We term this model Few-shot Incremental Active class SeleCtiOn (FIASco). We further integrate a potential field-based navigation technique with our model to develop a complete framework that can allow an agent to process and reason on its sensory data through the FIASco model, navigate towards the most informative object in the environment, gather data about the object through its sensors and incrementally update the FIASco model. Experimental results on a simulated agent and a real robot show the significance of our approach for long-term real-world robotics applications.

{{</citation>}}


## cs.CL (28)



### (3/115) Scaling In-Context Demonstrations with Structured Attention (Tianle Cai et al., 2023)

{{<citation>}}

Tianle Cai, Kaixuan Huang, Jason D. Lee, Mengdi Wang. (2023)  
**Scaling In-Context Demonstrations with Structured Attention**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2307.02690v1)  

---


**ABSTRACT**  
The recent surge of large language models (LLMs) highlights their ability to perform in-context learning, i.e., "learning" to perform a task from a few demonstrations in the context without any parameter updates. However, their capabilities of in-context learning are limited by the model architecture: 1) the use of demonstrations is constrained by a maximum sentence length due to positional embeddings; 2) the quadratic complexity of attention hinders users from using more demonstrations efficiently; 3) LLMs are shown to be sensitive to the order of the demonstrations. In this work, we tackle these challenges by proposing a better architectural design for in-context learning. We propose SAICL (Structured Attention for In-Context Learning), which replaces the full-attention by a structured attention mechanism designed for in-context learning, and removes unnecessary dependencies between individual demonstrations, while making the model invariant to the permutation of demonstrations. We evaluate SAICL in a meta-training framework and show that SAICL achieves comparable or better performance than full attention while obtaining up to 3.4x inference speed-up. SAICL also consistently outperforms a strong Fusion-in-Decoder (FiD) baseline which processes each demonstration independently. Finally, thanks to its linear nature, we demonstrate that SAICL can easily scale to hundreds of demonstrations with continuous performance gains with scaling.

{{</citation>}}


### (4/115) Learning Symbolic Rules over Abstract Meaning Representations for Textual Reinforcement Learning (Subhajit Chaudhury et al., 2023)

{{<citation>}}

Subhajit Chaudhury, Sarathkrishna Swaminathan, Daiki Kimura, Prithviraj Sen, Keerthiram Murugesan, Rosario Uceda-Sosa, Michiaki Tatsubori, Achille Fokoue, Pavan Kapanipathi, Asim Munawar, Alexander Gray. (2023)  
**Learning Symbolic Rules over Abstract Meaning Representations for Textual Reinforcement Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Abstract Meaning Representation, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02689v1)  

---


**ABSTRACT**  
Text-based reinforcement learning agents have predominantly been neural network-based models with embeddings-based representation, learning uninterpretable policies that often do not generalize well to unseen games. On the other hand, neuro-symbolic methods, specifically those that leverage an intermediate formal representation, are gaining significant attention in language understanding tasks. This is because of their advantages ranging from inherent interpretability, the lesser requirement of training data, and being generalizable in scenarios with unseen data. Therefore, in this paper, we propose a modular, NEuro-Symbolic Textual Agent (NESTA) that combines a generic semantic parser with a rule induction system to learn abstract interpretable rules as policies. Our experiments on established text-based game benchmarks show that the proposed NESTA method outperforms deep reinforcement learning-based techniques by achieving better generalization to unseen test games and learning from fewer training interactions.

{{</citation>}}


### (5/115) Unsupervised Sentiment Analysis of Plastic Surgery Social Media Posts (Alexandrea K. Ramnarine, 2023)

{{<citation>}}

Alexandrea K. Ramnarine. (2023)  
**Unsupervised Sentiment Analysis of Plastic Surgery Social Media Posts**  

---
Primary Category: cs.CL  
Categories: 68T50, cs-CL, cs.CL  
Keywords: AI, NLP, Sentiment Analysis, Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2307.02640v1)  

---


**ABSTRACT**  
The massive collection of user posts across social media platforms is primarily untapped for artificial intelligence (AI) use cases based on the sheer volume and velocity of textual data. Natural language processing (NLP) is a subfield of AI that leverages bodies of documents, known as corpora, to train computers in human-like language understanding. Using a word ranking method, term frequency-inverse document frequency (TF-IDF), to create features across documents, it is possible to perform unsupervised analytics, machine learning (ML) that can group the documents without a human manually labeling the data. For large datasets with thousands of features, t-distributed stochastic neighbor embedding (t-SNE), k-means clustering and Latent Dirichlet allocation (LDA) are employed to learn top words and generate topics for a Reddit and Twitter combined corpus. Using extremely simple deep learning models, this study demonstrates that the applied results of unsupervised analysis allow a computer to predict either negative, positive, or neutral user sentiment towards plastic surgery based on a tweet or subreddit post with almost 90% accuracy. Furthermore, the model is capable of achieving higher accuracy on the unsupervised sentiment task than on a rudimentary supervised document classification task. Therefore, unsupervised learning may be considered a viable option in labeling social media documents for NLP tasks.

{{</citation>}}


### (6/115) Evade ChatGPT Detectors via A Single Space (Shuyang Cai et al., 2023)

{{<citation>}}

Shuyang Cai, Wanyun Cui. (2023)  
**Evade ChatGPT Detectors via A Single Space**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.02599v1)  

---


**ABSTRACT**  
ChatGPT brings revolutionary social value but also raises concerns about the misuse of AI-generated content. Consequently, an important question is how to detect whether content is generated by ChatGPT or by human. Existing detectors are built upon the assumption that there are distributional gaps between human-generated and AI-generated content. These gaps are typically identified using statistical information or classifiers. Our research challenges the distributional gap assumption in detectors. We find that detectors do not effectively discriminate the semantic and stylistic gaps between human-generated and AI-generated content. Instead, the "subtle differences", such as an extra space, become crucial for detection. Based on this discovery, we propose the SpaceInfi strategy to evade detection. Experiments demonstrate the effectiveness of this strategy across multiple benchmarks and detectors. We also provide a theoretical explanation for why SpaceInfi is successful in evading perplexity-based detection. Our findings offer new insights and challenges for understanding and constructing more applicable ChatGPT detectors.

{{</citation>}}


### (7/115) ODD: A Benchmark Dataset for the NLP-based Opioid Related Aberrant Behavior Detection (Sunjae Kwon et al., 2023)

{{<citation>}}

Sunjae Kwon, Xun Wang, Weisong Liu, Emily Druhl, Minhee L. Sung, Joel I. Reisman, Wenjun Li, Robert D. Kerns, William Becker, Hong Yu. (2023)  
**ODD: A Benchmark Dataset for the NLP-based Opioid Related Aberrant Behavior Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2307.02591v1)  

---


**ABSTRACT**  
Opioid related aberrant behaviors (ORAB) present novel risk factors for opioid overdose. Previously, ORAB have been mainly assessed by survey results and by monitoring drug administrations. Such methods however, cannot scale up and do not cover the entire spectrum of aberrant behaviors. On the other hand, ORAB are widely documented in electronic health record notes. This paper introduces a novel biomedical natural language processing benchmark dataset named ODD, for ORAB Detection Dataset. ODD is an expert-annotated dataset comprising of more than 750 publicly available EHR notes. ODD has been designed to identify ORAB from patients' EHR notes and classify them into nine categories; 1) Confirmed Aberrant Behavior, 2) Suggested Aberrant Behavior, 3) Opioids, 4) Indication, 5) Diagnosed opioid dependency, 6) Benzodiapines, 7) Medication Changes, 8) Central Nervous System-related, and 9) Social Determinants of Health. We explored two state-of-the-art natural language processing (NLP) models (finetuning pretrained language models and prompt-tuning approaches) to identify ORAB. Experimental results show that the prompt-tuning models outperformed the finetuning models in most cateogories and the gains were especially higher among uncommon categories (Suggested aberrant behavior, Diagnosed opioid dependency and Medication change). Although the best model achieved the highest 83.92\% on area under precision recall curve, uncommon classes (Suggested Aberrant Behavior, Diagnosed Opioid Dependence, and Medication Change) still have a large room for performance improvement.

{{</citation>}}


### (8/115) Several categories of Large Language Models (LLMs): A Short Survey (Saurabh Pahune et al., 2023)

{{<citation>}}

Saurabh Pahune, Manoj Chandrasekharan. (2023)  
**Several categories of Large Language Models (LLMs): A Short Survey**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.10188v1)  

---


**ABSTRACT**  
Large Language Models(LLMs)have become effective tools for natural language processing and have been used in many different fields. This essay offers a succinct summary of various LLM subcategories. The survey emphasizes recent developments and efforts made for various LLM kinds, including task-based financial LLMs, multilingual language LLMs, biomedical and clinical LLMs, vision language LLMs, and code language models. The survey gives a general summary of the methods, attributes, datasets, transformer models, and comparison metrics applied in each category of LLMs. Furthermore, it highlights unresolved problems in the field of developing chatbots and virtual assistants, such as boosting natural language processing, enhancing chatbot intelligence, and resolving moral and legal dilemmas. The purpose of this study is to provide readers, developers, academics, and users interested in LLM-based chatbots and virtual intelligent assistant technologies with useful information and future directions.

{{</citation>}}


### (9/115) Named Entity Inclusion in Abstractive Text Summarization (Sergey Berezin et al., 2023)

{{<citation>}}

Sergey Berezin, Tatiana Batura. (2023)  
**Named Entity Inclusion in Abstractive Text Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: Abstractive Text Summarization, BERT, Summarization, Text Summarization  
[Paper Link](http://arxiv.org/abs/2307.02570v1)  

---


**ABSTRACT**  
We address the named entity omission - the drawback of many current abstractive text summarizers. We suggest a custom pretraining objective to enhance the model's attention on the named entities in a text. At first, the named entity recognition model RoBERTa is trained to determine named entities in the text. After that, this model is used to mask named entities in the text and the BART model is trained to reconstruct them. Next, the BART model is fine-tuned on the summarization task. Our experiments showed that this pretraining approach improves named entity inclusion precision and recall metrics.

{{</citation>}}


### (10/115) LongNet: Scaling Transformers to 1,000,000,000 Tokens (Jiayu Ding et al., 2023)

{{<citation>}}

Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, Nanning Zheng, Furu Wei. (2023)  
**LongNet: Scaling Transformers to 1,000,000,000 Tokens**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.02486v2)  

---


**ABSTRACT**  
Scaling sequence length has become a critical demand in the era of large language models. However, existing methods struggle with either computational complexity or model expressivity, rendering the maximum sequence length restricted. To address this issue, we introduce LongNet, a Transformer variant that can scale sequence length to more than 1 billion tokens, without sacrificing the performance on shorter sequences. Specifically, we propose dilated attention, which expands the attentive field exponentially as the distance grows. LongNet has significant advantages: 1) it has a linear computation complexity and a logarithm dependency between any two tokens in a sequence; 2) it can be served as a distributed trainer for extremely long sequences; 3) its dilated attention is a drop-in replacement for standard attention, which can be seamlessly integrated with the existing Transformer-based optimization. Experiments results demonstrate that LongNet yields strong performance on both long-sequence modeling and general language tasks. Our work opens up new possibilities for modeling very long sequences, e.g., treating a whole corpus or even the entire Internet as a sequence.

{{</citation>}}


### (11/115) Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks (Zhaofeng Wu et al., 2023)

{{<citation>}}

Zhaofeng Wu, Linlu Qiu, Alexis Ross, Ekin Akyürek, Boyuan Chen, Bailin Wang, Najoung Kim, Jacob Andreas, Yoon Kim. (2023)  
**Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.02477v1)  

---


**ABSTRACT**  
The impressive performance of recent language models across a wide range of tasks suggests that they possess a degree of abstract reasoning skills. Are these skills general and transferable, or specialized to specific tasks seen during pretraining? To disentangle these effects, we propose an evaluation framework based on "counterfactual" task variants that deviate from the default assumptions underlying standard tasks. Across a suite of 11 tasks, we observe nontrivial performance on the counterfactual variants, but nevertheless find that performance substantially and consistently degrades compared to the default conditions. This suggests that while current LMs may possess abstract task-solving skills to a degree, they often also rely on narrow, non-transferable procedures for task-solving. These results motivate a more careful interpretation of language model performance that teases apart these aspects of behavior.

{{</citation>}}


### (12/115) Deductive Additivity for Planning of Natural Language Proofs (Zayne Sprague et al., 2023)

{{<citation>}}

Zayne Sprague, Kaj Bostrom, Swarat Chaudhuri, Greg Durrett. (2023)  
**Deductive Additivity for Planning of Natural Language Proofs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Reasoning, Transformer  
[Paper Link](http://arxiv.org/abs/2307.02472v2)  

---


**ABSTRACT**  
Current natural language systems designed for multi-step claim validation typically operate in two phases: retrieve a set of relevant premise statements using heuristics (planning), then generate novel conclusions from those statements using a large language model (deduction). The planning step often requires expensive Transformer operations and does not scale to arbitrary numbers of premise statements. In this paper, we investigate whether an efficient planning heuristic is possible via embedding spaces compatible with deductive reasoning. Specifically, we evaluate whether embedding spaces exhibit a property we call deductive additivity: the sum of premise statement embeddings should be close to embeddings of conclusions based on those premises. We explore multiple sources of off-the-shelf dense embeddings in addition to fine-tuned embeddings from GPT3 and sparse embeddings from BM25. We study embedding models both intrinsically, evaluating whether the property of deductive additivity holds, and extrinsically, using them to assist planning in natural language proof generation. Lastly, we create a dataset, Single-Step Reasoning Contrast (SSRC), to further probe performance on various reasoning types. Our findings suggest that while standard embedding methods frequently embed conclusions near the sums of their premises, they fall short of being effective heuristics and lack the ability to model certain categories of reasoning.

{{</citation>}}


### (13/115) Won't Get Fooled Again: Answering Questions with False Premises (Shengding Hu et al., 2023)

{{<citation>}}

Shengding Hu, Yifan Luo, Huadong Wang, Xingyi Cheng, Zhiyuan Liu, Maosong Sun. (2023)  
**Won't Get Fooled Again: Answering Questions with False Premises**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.02394v1)  

---


**ABSTRACT**  
Pre-trained language models (PLMs) have shown unprecedented potential in various fields, especially as the backbones for question-answering (QA) systems. However, they tend to be easily deceived by tricky questions such as "How many eyes does the sun have?". Such frailties of PLMs often allude to the lack of knowledge within them. In this paper, we find that the PLMs already possess the knowledge required to rebut such questions, and the key is how to activate the knowledge. To systematize this observation, we investigate the PLMs' responses to one kind of tricky questions, i.e., the false premises questions (FPQs). We annotate a FalseQA dataset containing 2365 human-written FPQs, with the corresponding explanations for the false premises and the revised true premise questions. Using FalseQA, we discover that PLMs are capable of discriminating FPQs by fine-tuning on moderate numbers (e.g., 256) of examples. PLMs also generate reasonable explanations for the false premise, which serve as rebuttals. Further replaying a few general questions during training allows PLMs to excel on FPQs and general questions simultaneously. Our work suggests that once the rebuttal ability is stimulated, knowledge inside the PLMs can be effectively utilized to handle FPQs, which incentivizes the research on PLM-based QA systems.

{{</citation>}}


### (14/115) Utilizing ChatGPT Generated Data to Retrieve Depression Symptoms from Social Media (Ana-Maria Bucur, 2023)

{{<citation>}}

Ana-Maria Bucur. (2023)  
**Utilizing ChatGPT Generated Data to Retrieve Depression Symptoms from Social Media**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, ChatGPT, GPT, Social Media  
[Paper Link](http://arxiv.org/abs/2307.02313v2)  

---


**ABSTRACT**  
In this work, we present the contribution of the BLUE team in the eRisk Lab task on searching for symptoms of depression. The task consists of retrieving and ranking Reddit social media sentences that convey symptoms of depression from the BDI-II questionnaire. Given that synthetic data provided by LLMs have been proven to be a reliable method for augmenting data and fine-tuning downstream models, we chose to generate synthetic data using ChatGPT for each of the symptoms of the BDI-II questionnaire. We designed a prompt such that the generated data contains more richness and semantic diversity than the BDI-II responses for each question and, at the same time, contains emotional and anecdotal experiences that are specific to the more intimate way of sharing experiences on Reddit. We perform semantic search and rank the sentences' relevance to the BDI-II symptoms by cosine similarity. We used two state-of-the-art transformer-based models (MentalRoBERTa and a variant of MPNet) for embedding the social media posts, the original and generated responses of the BDI-II. Our results show that using sentence embeddings from a model designed for semantic search outperforms the approach using embeddings from a model pre-trained on mental health data. Furthermore, the generated synthetic data were proved too specific for this task, the approach simply relying on the BDI-II responses had the best performance.

{{</citation>}}


### (15/115) Performance Comparison of Large Language Models on VNHSGE English Dataset: OpenAI ChatGPT, Microsoft Bing Chat, and Google Bard (Xuan-Quy Dao, 2023)

{{<citation>}}

Xuan-Quy Dao. (2023)  
**Performance Comparison of Large Language Models on VNHSGE English Dataset: OpenAI ChatGPT, Microsoft Bing Chat, and Google Bard**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-3.5, Google, Language Model, Microsoft  
[Paper Link](http://arxiv.org/abs/2307.02288v3)  

---


**ABSTRACT**  
This paper presents a performance comparison of three large language models (LLMs), namely OpenAI ChatGPT, Microsoft Bing Chat (BingChat), and Google Bard, on the VNHSGE English dataset. The performance of BingChat, Bard, and ChatGPT (GPT-3.5) is 92.4\%, 86\%, and 79.2\%, respectively. The results show that BingChat is better than ChatGPT and Bard. Therefore, BingChat and Bard can replace ChatGPT while ChatGPT is not yet officially available in Vietnam. The results also indicate that BingChat, Bard and ChatGPT outperform Vietnamese students in English language proficiency. The findings of this study contribute to the understanding of the potential of LLMs in English language education. The remarkable performance of ChatGPT, BingChat, and Bard demonstrates their potential as effective tools for teaching and learning English at the high school level.

{{</citation>}}


### (16/115) SpaceNLI: Evaluating the Consistency of Predicting Inferences in Space (Lasha Abzianidze et al., 2023)

{{<citation>}}

Lasha Abzianidze, Joost Zwarts, Yoad Winter. (2023)  
**SpaceNLI: Evaluating the Consistency of Predicting Inferences in Space**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2-7, cs-CL, cs.CL  
Keywords: NLI  
[Paper Link](http://arxiv.org/abs/2307.02269v1)  

---


**ABSTRACT**  
While many natural language inference (NLI) datasets target certain semantic phenomena, e.g., negation, tense & aspect, monotonicity, and presupposition, to the best of our knowledge, there is no NLI dataset that involves diverse types of spatial expressions and reasoning. We fill this gap by semi-automatically creating an NLI dataset for spatial reasoning, called SpaceNLI. The data samples are automatically generated from a curated set of reasoning patterns, where the patterns are annotated with inference labels by experts. We test several SOTA NLI systems on SpaceNLI to gauge the complexity of the dataset and the system's capacity for spatial reasoning. Moreover, we introduce a Pattern Accuracy and argue that it is a more reliable and stricter measure than the accuracy for evaluating a system's performance on pattern-based generated data samples. Based on the evaluation results we find that the systems obtain moderate results on the spatial NLI problems but lack consistency per inference pattern. The results also reveal that non-projective spatial inferences (especially due to the "between" preposition) are the most challenging ones.

{{</citation>}}


### (17/115) Citation: A Key to Building Responsible and Accountable Large Language Models (Jie Huang et al., 2023)

{{<citation>}}

Jie Huang, Kevin Chen-Chuan Chang. (2023)  
**Citation: A Key to Building Responsible and Accountable Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.02185v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) bring transformative benefits alongside unique challenges, including intellectual property (IP) and ethical concerns. This position paper explores a novel angle to mitigate these risks, drawing parallels between LLMs and established web systems. We identify "citation" as a crucial yet missing component in LLMs, which could enhance content transparency and verifiability while addressing IP and ethical dilemmas. We further propose that a comprehensive citation mechanism for LLMs should account for both non-parametric and parametric content. Despite the complexity of implementing such a citation mechanism, along with the inherent potential pitfalls, we advocate for its development. Building on this foundation, we outline several research problems in this area, aiming to guide future explorations towards building more responsible and accountable LLMs.

{{</citation>}}


### (18/115) Open-Source Large Language Models Outperform Crowd Workers and Approach ChatGPT in Text-Annotation Tasks (Meysam Alizadeh et al., 2023)

{{<citation>}}

Meysam Alizadeh, Maël Kubli, Zeynab Samei, Shirin Dehghani, Juan Diego Bermeo, Maria Korobeynikova, Fabrizio Gilardi. (2023)  
**Open-Source Large Language Models Outperform Crowd Workers and Approach ChatGPT in Text-Annotation Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2307.02179v1)  

---


**ABSTRACT**  
This study examines the performance of open-source Large Language Models (LLMs) in text annotation tasks and compares it with proprietary models like ChatGPT and human-based services such as MTurk. While prior research demonstrated the high performance of ChatGPT across numerous NLP tasks, open-source LLMs like HugginChat and FLAN are gaining attention for their cost-effectiveness, transparency, reproducibility, and superior data protection. We assess these models using both zero-shot and few-shot approaches and different temperature parameters across a range of text annotation tasks. Our findings show that while ChatGPT achieves the best performance in most tasks, open-source LLMs not only outperform MTurk but also demonstrate competitive potential against ChatGPT in specific tasks.

{{</citation>}}


### (19/115) Becoming self-instruct: introducing early stopping criteria for minimal instruct tuning (Waseem AlShikh et al., 2023)

{{<citation>}}

Waseem AlShikh, Manhal Daaboul, Kirk Goddard, Brock Imel, Kiran Kamble, Parikshith Kulkarni, Melisa Russak. (2023)  
**Becoming self-instruct: introducing early stopping criteria for minimal instruct tuning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: LLaMA, QA  
[Paper Link](http://arxiv.org/abs/2307.03692v1)  

---


**ABSTRACT**  
In this paper, we introduce the Instruction Following Score (IFS), a metric that detects language models' ability to follow instructions. The metric has a dual purpose. First, IFS can be used to distinguish between base and instruct models. We benchmark publicly available base and instruct models, and show that the ratio of well formatted responses to partial and full sentences can be an effective measure between those two model classes. Secondly, the metric can be used as an early stopping criteria for instruct tuning. We compute IFS for Supervised Fine-Tuning (SFT) of 7B and 13B LLaMA models, showing that models learn to follow instructions relatively early in the training process, and the further finetuning can result in changes in the underlying base model semantics. As an example of semantics change we show the objectivity of model predictions, as defined by an auxiliary metric ObjecQA. We show that in this particular case, semantic changes are the steepest when the IFS tends to plateau. We hope that decomposing instruct tuning into IFS and semantic factors starts a new trend in better controllable instruct tuning and opens possibilities for designing minimal instruct interfaces querying foundation models.

{{</citation>}}


### (20/115) Leveraging Denoised Abstract Meaning Representation for Grammatical Error Correction (Hejing Cao et al., 2023)

{{<citation>}}

Hejing Cao, Dongyan Zhao. (2023)  
**Leveraging Denoised Abstract Meaning Representation for Grammatical Error Correction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Abstract Meaning Representation, T5  
[Paper Link](http://arxiv.org/abs/2307.02127v1)  

---


**ABSTRACT**  
Grammatical Error Correction (GEC) is the task of correcting errorful sentences into grammatically correct, semantically consistent, and coherent sentences. Popular GEC models either use large-scale synthetic corpora or use a large number of human-designed rules. The former is costly to train, while the latter requires quite a lot of human expertise. In recent years, AMR, a semantic representation framework, has been widely used by many natural language tasks due to its completeness and flexibility. A non-negligible concern is that AMRs of grammatically incorrect sentences may not be exactly reliable. In this paper, we propose the AMR-GEC, a seq-to-seq model that incorporates denoised AMR as additional knowledge. Specifically, We design a semantic aggregated GEC model and explore denoising methods to get AMRs more reliable. Experiments on the BEA-2019 shared task and the CoNLL-2014 shared task have shown that AMR-GEC performs comparably to a set of strong baselines with a large number of synthetic data. Compared with the T5 model with synthetic data, AMR-GEC can reduce the training time by 32\% while inference time is comparable. To the best of our knowledge, we are the first to incorporate AMR for grammatical error correction.

{{</citation>}}


### (21/115) Multilingual Controllable Transformer-Based Lexical Simplification (Kim Cheng Sheang et al., 2023)

{{<citation>}}

Kim Cheng Sheang, Horacio Saggion. (2023)  
**Multilingual Controllable Transformer-Based Lexical Simplification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Multilingual, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2307.02120v1)  

---


**ABSTRACT**  
Text is by far the most ubiquitous source of knowledge and information and should be made easily accessible to as many people as possible; however, texts often contain complex words that hinder reading comprehension and accessibility. Therefore, suggesting simpler alternatives for complex words without compromising meaning would help convey the information to a broader audience. This paper proposes mTLS, a multilingual controllable Transformer-based Lexical Simplification (LS) system fined-tuned with the T5 model. The novelty of this work lies in the use of language-specific prefixes, control tokens, and candidates extracted from pre-trained masked language models to learn simpler alternatives for complex words. The evaluation results on three well-known LS datasets -- LexMTurk, BenchLS, and NNSEval -- show that our model outperforms the previous state-of-the-art models like LSBert and ConLS. Moreover, further evaluation of our approach on the part of the recent TSAR-2022 multilingual LS shared-task dataset shows that our model performs competitively when compared with the participating systems for English LS and even outperforms the GPT-3 model on several metrics. Moreover, our model obtains performance gains also for Spanish and Portuguese.

{{</citation>}}


### (22/115) Different Games in Dialogue: Combining character and conversational types in strategic choice (Alafate Abulimiti, 2023)

{{<citation>}}

Alafate Abulimiti. (2023)  
**Different Games in Dialogue: Combining character and conversational types in strategic choice**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2-4; I-2-7, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2307.02087v1)  

---


**ABSTRACT**  
In this paper, we show that investigating the interaction of conversational type (often known as language game or speech genre) with the character types of the interlocutors is worthwhile. We present a method of calculating the decision making process for selecting dialogue moves that combines character type and conversational type. We also present a mathematical model that illustrate these factors' interactions in a quantitative way.

{{</citation>}}


### (23/115) Graph Contrastive Topic Model (Zheheng Luo et al., 2023)

{{<citation>}}

Zheheng Luo, Lei Liu, Qianqian Xie, Sophia Ananiadou. (2023)  
**Graph Contrastive Topic Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Topic Model  
[Paper Link](http://arxiv.org/abs/2307.02078v1)  

---


**ABSTRACT**  
Existing NTMs with contrastive learning suffer from the sample bias problem owing to the word frequency-based sampling strategy, which may result in false negative samples with similar semantics to the prototypes. In this paper, we aim to explore the efficient sampling strategy and contrastive learning in NTMs to address the aforementioned issue. We propose a new sampling assumption that negative samples should contain words that are semantically irrelevant to the prototype. Based on it, we propose the graph contrastive topic model (GCTM), which conducts graph contrastive learning (GCL) using informative positive and negative samples that are generated by the graph-based sampling strategy leveraging in-depth correlation and irrelevance among documents and words. In GCTM, we first model the input document as the document word bipartite graph (DWBG), and construct positive and negative word co-occurrence graphs (WCGs), encoded by graph neural networks, to express in-depth semantic correlation and irrelevance among words. Based on the DWBG and WCGs, we design the document-word information propagation (DWIP) process to perform the edge perturbation of DWBG, based on multi-hop correlations/irrelevance among documents and words. This yields the desired negative and positive samples, which will be utilized for GCL together with the prototypes to improve learning document topic representations and latent topics. We further show that GCL can be interpreted as the structured variational graph auto-encoder which maximizes the mutual information of latent topic representations of different perspectives on DWBG. Experiments on several benchmark datasets demonstrate the effectiveness of our method for topic coherence and document representation learning compared with existing SOTA methods.

{{</citation>}}


### (24/115) Emoji Prediction using Transformer Models (Muhammad Osama Nusrat et al., 2023)

{{<citation>}}

Muhammad Osama Nusrat, Zeeshan Habib, Mehreen Alam, Saad Ahmed Jamal. (2023)  
**Emoji Prediction using Transformer Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Transformer  
[Paper Link](http://arxiv.org/abs/2307.02054v2)  

---


**ABSTRACT**  
In recent years, the use of emojis in social media has increased dramatically, making them an important element in understanding online communication. However, predicting the meaning of emojis in a given text is a challenging task due to their ambiguous nature. In this study, we propose a transformer-based approach for emoji prediction using BERT, a widely-used pre-trained language model. We fine-tuned BERT on a large corpus of text containing both text and emojis to predict the most appropriate emoji for a given text. Our experimental results demonstrate that our approach outperforms several state-of-the-art models in predicting emojis with an accuracy of over 75 percent. This work has potential applications in natural language processing, sentiment analysis, and social media marketing.

{{</citation>}}


### (25/115) Flacuna: Unleashing the Problem Solving Power of Vicuna using FLAN Fine-Tuning (Deepanway Ghosal et al., 2023)

{{<citation>}}

Deepanway Ghosal, Yew Ken Chia, Navonil Majumder, Soujanya Poria. (2023)  
**Flacuna: Unleashing the Problem Solving Power of Vicuna using FLAN Fine-Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, T5  
[Paper Link](http://arxiv.org/abs/2307.02053v1)  

---


**ABSTRACT**  
Recently, the release of INSTRUCTEVAL has provided valuable insights into the performance of large language models (LLMs) that utilize encoder-decoder or decoder-only architecture. Interestingly, despite being introduced four years ago, T5-based LLMs, such as FLAN-T5, continue to outperform the latest decoder-based LLMs, such as LLAMA and VICUNA, on tasks that require general problem-solving skills. This performance discrepancy can be attributed to three key factors: (1) Pre-training data, (2) Backbone architecture, and (3) Instruction dataset. In this technical report, our main focus is on investigating the impact of the third factor by leveraging VICUNA, a large language model based on LLAMA, which has undergone fine-tuning on ChatGPT conversations. To achieve this objective, we fine-tuned VICUNA using a customized instruction dataset collection called FLANMINI. This collection includes a subset of the large-scale instruction dataset known as FLAN, as well as various code-related datasets and conversational datasets derived from ChatGPT/GPT-4. This dataset comprises a large number of tasks that demand problem-solving skills. Our experimental findings strongly indicate that the enhanced problem-solving abilities of our model, FLACUNA, are obtained through fine-tuning VICUNA on the FLAN dataset, leading to significant improvements across numerous benchmark datasets in INSTRUCTEVAL. FLACUNA is publicly available at https://huggingface.co/declare-lab/flacuna-13b-v1.0.

{{</citation>}}


### (26/115) CAME: Confidence-guided Adaptive Memory Efficient Optimization (Yang Luo et al., 2023)

{{<citation>}}

Yang Luo, Xiaozhe Ren, Zangwei Zheng, Zhuo Jiang, Xin Jiang, Yang You. (2023)  
**CAME: Confidence-guided Adaptive Memory Efficient Optimization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2307.02047v1)  

---


**ABSTRACT**  
Adaptive gradient methods, such as Adam and LAMB, have demonstrated excellent performance in the training of large language models. Nevertheless, the need for adaptivity requires maintaining second-moment estimates of the per-parameter gradients, which entails a high cost of extra memory overheads. To solve this problem, several memory-efficient optimizers (e.g., Adafactor) have been proposed to obtain a drastic reduction in auxiliary memory usage, but with a performance penalty. In this paper, we first study a confidence-guided strategy to reduce the instability of existing memory efficient optimizers. Based on this strategy, we propose CAME to simultaneously achieve two goals: fast convergence as in traditional adaptive methods, and low memory usage as in memory-efficient methods. Extensive experiments demonstrate the training stability and superior performance of CAME across various NLP tasks such as BERT and GPT-2 training. Notably, for BERT pre-training on the large batch size of 32,768, our proposed optimizer attains faster convergence and higher accuracy compared with the Adam optimizer. The implementation of CAME is publicly available.

{{</citation>}}


### (27/115) Comparative Analysis of GPT-4 and Human Graders in Evaluating Praise Given to Students in Synthetic Dialogues (Dollaya Hirunyasiri et al., 2023)

{{<citation>}}

Dollaya Hirunyasiri, Danielle R. Thomas, Jionghao Lin, Kenneth R. Koedinger, Vincent Aleven. (2023)  
**Comparative Analysis of GPT-4 and Human Graders in Evaluating Praise Given to Students in Synthetic Dialogues**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: AI, ChatGPT, Dialog, Dialogue, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2307.02018v1)  

---


**ABSTRACT**  
Research suggests that providing specific and timely feedback to human tutors enhances their performance. However, it presents challenges due to the time-consuming nature of assessing tutor performance by human evaluators. Large language models, such as the AI-chatbot ChatGPT, hold potential for offering constructive feedback to tutors in practical settings. Nevertheless, the accuracy of AI-generated feedback remains uncertain, with scant research investigating the ability of models like ChatGPT to deliver effective feedback. In this work-in-progress, we evaluate 30 dialogues generated by GPT-4 in a tutor-student setting. We use two different prompting approaches, the zero-shot chain of thought and the few-shot chain of thought, to identify specific components of effective praise based on five criteria. These approaches are then compared to the results of human graders for accuracy. Our goal is to assess the extent to which GPT-4 can accurately identify each praise criterion. We found that both zero-shot and few-shot chain of thought approaches yield comparable results. GPT-4 performs moderately well in identifying instances when the tutor offers specific and immediate praise. However, GPT-4 underperforms in identifying the tutor's ability to deliver sincere praise, particularly in the zero-shot prompting scenario where examples of sincere tutor praise statements were not provided. Future work will focus on enhancing prompt engineering, developing a more general tutoring rubric, and evaluating our method using real-life tutoring dialogues.

{{</citation>}}


### (28/115) Evaluating the Effectiveness of Large Language Models in Representing Textual Descriptions of Geometry and Spatial Relations (Yuhan Ji et al., 2023)

{{<citation>}}

Yuhan Ji, Song Gao. (2023)  
**Evaluating the Effectiveness of Large Language Models in Representing Textual Descriptions of Geometry and Spatial Relations**  

---
Primary Category: cs.CL  
Categories: I-2, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, BERT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.03678v1)  

---


**ABSTRACT**  
This research focuses on assessing the ability of large language models (LLMs) in representing geometries and their spatial relations. We utilize LLMs including GPT-2 and BERT to encode the well-known text (WKT) format of geometries and then feed their embeddings into classifiers and regressors to evaluate the effectiveness of the LLMs-generated embeddings for geometric attributes. The experiments demonstrate that while the LLMs-generated embeddings can preserve geometry types and capture some spatial relations (up to 73% accuracy), challenges remain in estimating numeric values and retrieving spatially related objects. This research highlights the need for improvement in terms of capturing the nuances and complexities of the underlying geospatial data and integrating domain knowledge to support various GeoAI applications using foundation models.

{{</citation>}}


### (29/115) Using Data Augmentations and VTLN to Reduce Bias in Dutch End-to-End Speech Recognition Systems (Tanvina Patel et al., 2023)

{{<citation>}}

Tanvina Patel, Odette Scharenborg. (2023)  
**Using Data Augmentations and VTLN to Reduce Bias in Dutch End-to-End Speech Recognition Systems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, Bias, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.02009v1)  

---


**ABSTRACT**  
Speech technology has improved greatly for norm speakers, i.e., adult native speakers of a language without speech impediments or strong accents. However, non-norm or diverse speaker groups show a distinct performance gap with norm speakers, which we refer to as bias. In this work, we aim to reduce bias against different age groups and non-native speakers of Dutch. For an end-to-end (E2E) ASR system, we use state-of-the-art speed perturbation and spectral augmentation as data augmentation techniques and explore Vocal Tract Length Normalization (VTLN) to normalise for spectral differences due to differences in anatomy. The combination of data augmentation and VTLN reduced the average WER and bias across various diverse speaker groups by 6.9% and 3.9%, respectively. The VTLN model trained on Dutch was also effective in improving performance of Mandarin Chinese child speech, thus, showing generalisability across languages

{{</citation>}}


### (30/115) PULSAR at MEDIQA-Sum 2023: Large Language Models Augmented by Synthetic Dialogue Convert Patient Dialogues to Medical Records (Viktor Schlegel et al., 2023)

{{<citation>}}

Viktor Schlegel, Hao Li, Yuping Wu, Anand Subramanian, Thanh-Tung Nguyen, Abhinav Ramesh Kashyap, Daniel Beck, Xiaojun Zeng, Riza Theresa Batista-Navarro, Stefan Winkler, Goran Nenadic. (2023)  
**PULSAR at MEDIQA-Sum 2023: Large Language Models Augmented by Synthetic Dialogue Convert Patient Dialogues to Medical Records**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2307.02006v1)  

---


**ABSTRACT**  
This paper describes PULSAR, our system submission at the ImageClef 2023 MediQA-Sum task on summarising patient-doctor dialogues into clinical records. The proposed framework relies on domain-specific pre-training, to produce a specialised language model which is trained on task-specific natural data augmented by synthetic data generated by a black-box LLM. We find limited evidence towards the efficacy of domain-specific pre-training and data augmentation, while scaling up the language model yields the best performance gains. Our approach was ranked second and third among 13 submissions on task B of the challenge. Our code is available at https://github.com/yuping-wu/PULSAR.

{{</citation>}}


## cs.CV (24)



### (31/115) Zero-Shot Dense Video Captioning by Jointly Optimizing Text and Moment (Yongrae Jo et al., 2023)

{{<citation>}}

Yongrae Jo, Seongyun Lee, Aiden SJ Lee, Hyunji Lee, Hanseok Oh, Minjoon Seo. (2023)  
**Zero-Shot Dense Video Captioning by Jointly Optimizing Text and Moment**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.02682v2)  

---


**ABSTRACT**  
Dense video captioning, a task of localizing meaningful moments and generating relevant captions for videos, often requires a large, expensive corpus of annotated video segments paired with text. In an effort to minimize the annotation cost, we propose ZeroTA, a novel method for dense video captioning in a zero-shot manner. Our method does not require any videos or annotations for training; instead, it localizes and describes events within each input video at test time by optimizing solely on the input. This is accomplished by introducing a soft moment mask that represents a temporal segment in the video and jointly optimizing it with the prefix parameters of a language model. This joint optimization aligns a frozen language generation model (i.e., GPT-2) with a frozen vision-language contrastive model (i.e., CLIP) by maximizing the matching score between the generated text and a moment within the video. We also introduce a pairwise temporal IoU loss to let a set of soft moment masks capture multiple distinct events within the video. Our method effectively discovers diverse significant events within the video, with the resulting captions appropriately describing these events. The empirical results demonstrate that ZeroTA surpasses zero-shot baselines and even outperforms the state-of-the-art few-shot method on the widely-used benchmark ActivityNet Captions. Moreover, our method shows greater robustness compared to supervised methods when evaluated in out-of-domain scenarios. This research provides insight into the potential of aligning widely-used models, such as language generation models and vision-language models, to unlock a new capability: understanding temporal aspects of videos.

{{</citation>}}


### (32/115) Spherical Feature Pyramid Networks For Semantic Segmentation (Thomas Walker et al., 2023)

{{<citation>}}

Thomas Walker, Varun Anand, Pavlos Andreadis. (2023)  
**Spherical Feature Pyramid Networks For Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: I-4-6, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.02658v1)  

---


**ABSTRACT**  
Semantic segmentation for spherical data is a challenging problem in machine learning since conventional planar approaches require projecting the spherical image to the Euclidean plane. Representing the signal on a fundamentally different topology introduces edges and distortions which impact network performance. Recently, graph-based approaches have bypassed these challenges to attain significant improvements by representing the signal on a spherical mesh. Current approaches to spherical segmentation exclusively use variants of the UNet architecture, meaning more successful planar architectures remain unexplored. Inspired by the success of feature pyramid networks (FPNs) in planar image segmentation, we leverage the pyramidal hierarchy of graph-based spherical CNNs to design spherical FPNs. Our spherical FPN models show consistent improvements over spherical UNets, whilst using fewer parameters. On the Stanford 2D-3D-S dataset, our models achieve state-of-the-art performance with an mIOU of 48.75, an improvement of 3.75 IoU points over the previous best spherical CNN.

{{</citation>}}


### (33/115) What Matters in Training a GPT4-Style Language Model with Multimodal Inputs? (Yan Zeng et al., 2023)

{{<citation>}}

Yan Zeng, Hanbo Zhang, Jiani Zheng, Jiangnan Xia, Guoqiang Wei, Yang Wei, Yuchen Zhang, Tao Kong. (2023)  
**What Matters in Training a GPT4-Style Language Model with Multimodal Inputs?**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.02469v1)  

---


**ABSTRACT**  
Recent advancements in Large Language Models (LLMs) such as GPT4 have displayed exceptional multi-modal capabilities in following open-ended instructions given images. However, the performance of these models heavily relies on design choices such as network structures, training data, and training strategies, and these choices have not been extensively discussed in the literature, making it difficult to quantify progress in this field. To address this issue, this paper presents a systematic and comprehensive study, quantitatively and qualitatively, on training such models. We implement over 20 variants with controlled settings. Concretely, for network structures, we compare different LLM backbones and model designs. For training data, we investigate the impact of data and sampling strategies. For instructions, we explore the influence of diversified prompts on the instruction-following ability of the trained models. For benchmarks, we contribute the first, to our best knowledge, comprehensive evaluation set including both image and video tasks through crowd-sourcing. Based on our findings, we present Lynx, which performs the most accurate multi-modal understanding while keeping the best multi-modal generation ability compared to existing open-sourced GPT4-style models.

{{</citation>}}


### (34/115) Large-scale Detection of Marine Debris in Coastal Areas with Sentinel-2 (Marc Rußwurm et al., 2023)

{{<citation>}}

Marc Rußwurm, Sushen Jilla Venkatesa, Devis Tuia. (2023)  
**Large-scale Detection of Marine Debris in Coastal Areas with Sentinel-2**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02465v1)  

---


**ABSTRACT**  
Detecting and quantifying marine pollution and macro-plastics is an increasingly pressing ecological issue that directly impacts ecology and human health. Efforts to quantify marine pollution are often conducted with sparse and expensive beach surveys, which are difficult to conduct on a large scale. Here, remote sensing can provide reliable estimates of plastic pollution by regularly monitoring and detecting marine debris in coastal areas. Medium-resolution satellite data of coastal areas is readily available and can be leveraged to detect aggregations of marine debris containing plastic litter. In this work, we present a detector for marine debris built on a deep segmentation model that outputs a probability for marine debris at the pixel level. We train this detector with a combination of annotated datasets of marine debris and evaluate it on specifically selected test sites where it is highly probable that plastic pollution is present in the detected marine debris. We demonstrate quantitatively and qualitatively that a deep learning model trained on this dataset issued from multiple sources outperforms existing detection models trained on previous datasets by a large margin. Our experiments show, consistent with the principles of data-centric AI, that this performance is due to our particular dataset design with extensive sampling of negative examples and label refinements rather than depending on the particular deep learning model. We hope to accelerate advances in the large-scale automated detection of marine debris, which is a step towards quantifying and monitoring marine litter with remote sensing at global scales, and release the model weights and training source code under https://github.com/marccoru/marinedebrisdetector

{{</citation>}}


### (35/115) Unbalanced Optimal Transport: A Unified Framework for Object Detection (Henri De Plaen et al., 2023)

{{<citation>}}

Henri De Plaen, Pierre-François De Plaen, Johan A. K. Suykens, Marc Proesmans, Tinne Tuytelaars, Luc Van Gool. (2023)  
**Unbalanced Optimal Transport: A Unified Framework for Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.02402v1)  

---


**ABSTRACT**  
During training, supervised object detection tries to correctly match the predicted bounding boxes and associated classification scores to the ground truth. This is essential to determine which predictions are to be pushed towards which solutions, or to be discarded. Popular matching strategies include matching to the closest ground truth box (mostly used in combination with anchors), or matching via the Hungarian algorithm (mostly used in anchor-free methods). Each of these strategies comes with its own properties, underlying losses, and heuristics. We show how Unbalanced Optimal Transport unifies these different approaches and opens a whole continuum of methods in between. This allows for a finer selection of the desired properties. Experimentally, we show that training an object detection model with Unbalanced Optimal Transport is able to reach the state-of-the-art both in terms of Average Precision and Average Recall as well as to provide a faster initial convergence. The approach is well suited for GPU implementation, which proves to be an advantage for large-scale models.

{{</citation>}}


### (36/115) GAFAR: Graph-Attention Feature-Augmentation for Registration A Fast and Light-weight Point Set Registration Algorithm (Ludwig Mohr et al., 2023)

{{<citation>}}

Ludwig Mohr, Ismail Geles, Friedrich Fraundorfer. (2023)  
**GAFAR: Graph-Attention Feature-Augmentation for Registration A Fast and Light-weight Point Set Registration Algorithm**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Augmentation  
[Paper Link](http://arxiv.org/abs/2307.02339v1)  

---


**ABSTRACT**  
Rigid registration of point clouds is a fundamental problem in computer vision with many applications from 3D scene reconstruction to geometry capture and robotics. If a suitable initial registration is available, conventional methods like ICP and its many variants can provide adequate solutions. In absence of a suitable initialization and in the presence of a high outlier rate or in the case of small overlap though the task of rigid registration still presents great challenges. The advent of deep learning in computer vision has brought new drive to research on this topic, since it provides the possibility to learn expressive feature-representations and provide one-shot estimates instead of depending on time-consuming iterations of conventional robust methods. Yet, the rotation and permutation invariant nature of point clouds poses its own challenges to deep learning, resulting in loss of performance and low generalization capability due to sensitivity to outliers and characteristics of 3D scans not present during network training. In this work, we present a novel fast and light-weight network architecture using the attention mechanism to augment point descriptors at inference time to optimally suit the registration task of the specific point clouds it is presented with. Employing a fully-connected graph both within and between point clouds lets the network reason about the importance and reliability of points for registration, making our approach robust to outliers, low overlap and unseen data. We test the performance of our registration algorithm on different registration and generalization tasks and provide information on runtime and resource consumption. The code and trained weights are available at https://github.com/mordecaimalignatius/GAFAR/.

{{</citation>}}


### (37/115) MSViT: Dynamic Mixed-Scale Tokenization for Vision Transformers (Jakob Drachmann Havtorn et al., 2023)

{{<citation>}}

Jakob Drachmann Havtorn, Amelie Royer, Tijmen Blankevoort, Babak Ehteshami Bejnordi. (2023)  
**MSViT: Dynamic Mixed-Scale Tokenization for Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.02321v1)  

---


**ABSTRACT**  
The input tokens to Vision Transformers carry little semantic meaning as they are defined as regular equal-sized patches of the input image, regardless of its content. However, processing uniform background areas of an image should not necessitate as much compute as dense, cluttered areas. To address this issue, we propose a dynamic mixed-scale tokenization scheme for ViT, MSViT. Our method introduces a conditional gating mechanism that selects the optimal token scale for every image region, such that the number of tokens is dynamically determined per input. The proposed gating module is lightweight, agnostic to the choice of transformer backbone, and trained within a few epochs (e.g., 20 epochs on ImageNet) with little training overhead. In addition, to enhance the conditional behavior of the gate during training, we introduce a novel generalization of the batch-shaping loss. We show that our gating module is able to learn meaningful semantics despite operating locally at the coarse patch-level. We validate MSViT on the tasks of classification and segmentation where it leads to improved accuracy-complexity trade-off.

{{</citation>}}


### (38/115) Multi-Scale Prototypical Transformer for Whole Slide Image Classification (Saisai Ding et al., 2023)

{{<citation>}}

Saisai Ding, Jun Wang, Juncheng Li, Jun Shi. (2023)  
**Multi-Scale Prototypical Transformer for Whole Slide Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification, Transformer  
[Paper Link](http://arxiv.org/abs/2307.02308v1)  

---


**ABSTRACT**  
Whole slide image (WSI) classification is an essential task in computational pathology. Despite the recent advances in multiple instance learning (MIL) for WSI classification, accurate classification of WSIs remains challenging due to the extreme imbalance between the positive and negative instances in bags, and the complicated pre-processing to fuse multi-scale information of WSI. To this end, we propose a novel multi-scale prototypical Transformer (MSPT) for WSI classification, which includes a prototypical Transformer (PT) module and a multi-scale feature fusion module (MFFM). The PT is developed to reduce redundant instances in bags by integrating prototypical learning into the Transformer architecture. It substitutes all instances with cluster prototypes, which are then re-calibrated through the self-attention mechanism of the Trans-former. Thereafter, an MFFM is proposed to fuse the clustered prototypes of different scales, which employs MLP-Mixer to enhance the information communication between prototypes. The experimental results on two public WSI datasets demonstrate that the proposed MSPT outperforms all the compared algorithms, suggesting its potential applications.

{{</citation>}}


### (39/115) Interactive Image Segmentation with Cross-Modality Vision Transformers (Kun Li et al., 2023)

{{<citation>}}

Kun Li, George Vosselman, Michael Ying Yang. (2023)  
**Interactive Image Segmentation with Cross-Modality Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.02280v1)  

---


**ABSTRACT**  
Interactive image segmentation aims to segment the target from the background with the manual guidance, which takes as input multimodal data such as images, clicks, scribbles, and bounding boxes. Recently, vision transformers have achieved a great success in several downstream visual tasks, and a few efforts have been made to bring this powerful architecture to interactive segmentation task. However, the previous works neglect the relations between two modalities and directly mock the way of processing purely visual information with self-attentions. In this paper, we propose a simple yet effective network for click-based interactive segmentation with cross-modality vision transformers. Cross-modality transformers exploits mutual information to better guide the learning process. The experiments on several benchmarks show that the proposed method achieves superior performance in comparison to the previous state-of-the-art models. The stability of our method in term of avoiding failure cases shows its potential to be a practical annotation tool. The code and pretrained models will be released under https://github.com/lik1996/iCMFormer.

{{</citation>}}


### (40/115) Joint Hierarchical Priors and Adaptive Spatial Resolution for Efficient Neural Image Compression (Ahmed Ghorbel et al., 2023)

{{<citation>}}

Ahmed Ghorbel, Wassim Hamidouche, Luce Morin. (2023)  
**Joint Hierarchical Priors and Adaptive Spatial Resolution for Efficient Neural Image Compression**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02273v2)  

---


**ABSTRACT**  
Recently, the performance of neural image compression (NIC) has steadily improved thanks to the last line of study, reaching or outperforming state-of-the-art conventional codecs. Despite significant progress, current NIC methods still rely on ConvNet-based entropy coding, limited in modeling long-range dependencies due to their local connectivity and the increasing number of architectural biases and priors, resulting in complex underperforming models with high decoding latency. Motivated by the efficiency investigation of the Tranformer-based transform coding framework, namely SwinT-ChARM, we propose to enhance the latter, as first, with a more straightforward yet effective Tranformer-based channel-wise auto-regressive prior model, resulting in an absolute image compression transformer (ICT). Through the proposed ICT, we can capture both global and local contexts from the latent representations and better parameterize the distribution of the quantized latents. Further, we leverage a learnable scaling module with a sandwich ConvNeXt-based pre-/post-processor to accurately extract more compact latent codes while reconstructing higher-quality images. Extensive experimental results on benchmark datasets showed that the proposed framework significantly improves the trade-off between coding efficiency and decoder complexity over the versatile video coding (VVC) reference encoder (VTM-18.0) and the neural codec SwinT-ChARM. Moreover, we provide model scaling studies to verify the computational efficiency of our approach and conduct several objective and subjective analyses to bring to the fore the performance gap between the adaptive image compression transformer (AICT) and the neural codec SwinT-ChARM.

{{</citation>}}


### (41/115) SVDM: Single-View Diffusion Model for Pseudo-Stereo 3D Object Detection (Yuguang Shi, 2023)

{{<citation>}}

Yuguang Shi. (2023)  
**SVDM: Single-View Diffusion Model for Pseudo-Stereo 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.02270v1)  

---


**ABSTRACT**  
One of the key problems in 3D object detection is to reduce the accuracy gap between methods based on LiDAR sensors and those based on monocular cameras. A recently proposed framework for monocular 3D detection based on Pseudo-Stereo has received considerable attention in the community. However, so far these two problems are discovered in existing practices, including (1) monocular depth estimation and Pseudo-Stereo detector must be trained separately, (2) Difficult to be compatible with different stereo detectors and (3) the overall calculation is large, which affects the reasoning speed. In this work, we propose an end-to-end, efficient pseudo-stereo 3D detection framework by introducing a Single-View Diffusion Model (SVDM) that uses a few iterations to gradually deliver right informative pixels to the left image. SVDM allows the entire pseudo-stereo 3D detection pipeline to be trained end-to-end and can benefit from the training of stereo detectors. Afterwards, we further explore the application of SVDM in depth-free stereo 3D detection, and the final framework is compatible with most stereo detectors. Among multiple benchmarks on the KITTI dataset, we achieve new state-of-the-art performance.

{{</citation>}}


### (42/115) Rethinking Multiple Instance Learning for Whole Slide Image Classification: A Good Instance Classifier is All You Need (Linhao Qu et al., 2023)

{{<citation>}}

Linhao Qu, Yingfan Ma, Xiaoyuan Luo, Manning Wang, Zhijian Song. (2023)  
**Rethinking Multiple Instance Learning for Whole Slide Image Classification: A Good Instance Classifier is All You Need**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2307.02249v1)  

---


**ABSTRACT**  
Weakly supervised whole slide image classification is usually formulated as a multiple instance learning (MIL) problem, where each slide is treated as a bag, and the patches cut out of it are treated as instances. Existing methods either train an instance classifier through pseudo-labeling or aggregate instance features into a bag feature through attention mechanisms and then train a bag classifier, where the attention scores can be used for instance-level classification. However, the pseudo instance labels constructed by the former usually contain a lot of noise, and the attention scores constructed by the latter are not accurate enough, both of which affect their performance. In this paper, we propose an instance-level MIL framework based on contrastive learning and prototype learning to effectively accomplish both instance classification and bag classification tasks. To this end, we propose an instance-level weakly supervised contrastive learning algorithm for the first time under the MIL setting to effectively learn instance feature representation. We also propose an accurate pseudo label generation method through prototype learning. We then develop a joint training strategy for weakly supervised contrastive learning, prototype learning, and instance classifier training. Extensive experiments and visualizations on four datasets demonstrate the powerful performance of our method. Codes will be available.

{{</citation>}}


### (43/115) S3C: Self-Supervised Stochastic Classifiers for Few-Shot Class-Incremental Learning (Jayateja Kalla et al., 2023)

{{<citation>}}

Jayateja Kalla, Soma Biswas. (2023)  
**S3C: Self-Supervised Stochastic Classifiers for Few-Shot Class-Incremental Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.02246v1)  

---


**ABSTRACT**  
Few-shot class-incremental learning (FSCIL) aims to learn progressively about new classes with very few labeled samples, without forgetting the knowledge of already learnt classes. FSCIL suffers from two major challenges: (i) over-fitting on the new classes due to limited amount of data, (ii) catastrophically forgetting about the old classes due to unavailability of data from these classes in the incremental stages. In this work, we propose a self-supervised stochastic classifier (S3C) to counter both these challenges in FSCIL. The stochasticity of the classifier weights (or class prototypes) not only mitigates the adverse effect of absence of large number of samples of the new classes, but also the absence of samples from previously learnt classes during the incremental steps. This is complemented by the self-supervision component, which helps to learn features from the base classes which generalize well to unseen classes that are encountered in future, thus reducing catastrophic forgetting. Extensive evaluation on three benchmark datasets using multiple evaluation metrics show the effectiveness of the proposed framework. We also experiment on two additional realistic scenarios of FSCIL, namely where the number of annotated data available for each of the new classes can be different, and also where the number of base classes is much lesser, and show that the proposed S3C performs significantly better than the state-of-the-art for all these challenging scenarios.

{{</citation>}}


### (44/115) MAE-DFER: Efficient Masked Autoencoder for Self-supervised Dynamic Facial Expression Recognition (Licai Sun et al., 2023)

{{<citation>}}

Licai Sun, Zheng Lian, Bin Liu, Jianhua Tao. (2023)  
**MAE-DFER: Efficient Masked Autoencoder for Self-supervised Dynamic Facial Expression Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-HC, cs-MM, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.02227v1)  

---


**ABSTRACT**  
Dynamic facial expression recognition (DFER) is essential to the development of intelligent and empathetic machines. Prior efforts in this field mainly fall into supervised learning paradigm, which is restricted by the limited labeled data in existing datasets. Inspired by recent unprecedented success of masked autoencoders (e.g., VideoMAE), this paper proposes MAE-DFER, a novel self-supervised method which leverages large-scale self-supervised pre-training on abundant unlabeled data to advance the development of DFER. Since the vanilla Vision Transformer (ViT) employed in VideoMAE requires substantial computation during fine-tuning, MAE-DFER develops an efficient local-global interaction Transformer (LGI-Former) as the encoder. LGI-Former first constrains self-attention in local spatiotemporal regions and then utilizes a small set of learnable representative tokens to achieve efficient local-global information exchange, thus avoiding the expensive computation of global space-time self-attention in ViT. Moreover, in addition to the standalone appearance content reconstruction in VideoMAE, MAE-DFER also introduces explicit facial motion modeling to encourage LGI-Former to excavate both static appearance and dynamic motion information. Extensive experiments on six datasets show that MAE-DFER consistently outperforms state-of-the-art supervised methods by significant margins, verifying that it can learn powerful dynamic facial representations via large-scale self-supervised pre-training. Besides, it has comparable or even better performance than VideoMAE, while largely reducing the computational cost (about 38\% FLOPs). We believe MAE-DFER has paved a new way for the advancement of DFER and can inspire more relavant research in this field and even other related tasks. Codes and models are publicly available at https://github.com/sunlicai/MAE-DFER.

{{</citation>}}


### (45/115) Prompting Diffusion Representations for Cross-Domain Semantic Segmentation (Rui Gong et al., 2023)

{{<citation>}}

Rui Gong, Martin Danelljan, Han Sun, Julio Delgado Mangas, Luc Van Gool. (2023)  
**Prompting Diffusion Representations for Cross-Domain Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.02138v1)  

---


**ABSTRACT**  
While originally designed for image generation, diffusion models have recently shown to provide excellent pretrained feature representations for semantic segmentation. Intrigued by this result, we set out to explore how well diffusion-pretrained representations generalize to new domains, a crucial ability for any representation. We find that diffusion-pretraining achieves extraordinary domain generalization results for semantic segmentation, outperforming both supervised and self-supervised backbone networks. Motivated by this, we investigate how to utilize the model's unique ability of taking an input prompt, in order to further enhance its cross-domain performance. We introduce a scene prompt and a prompt randomization strategy to help further disentangle the domain-invariant information when training the segmentation head. Moreover, we propose a simple but highly effective approach for test-time domain adaptation, based on learning a scene prompt on the target domain in an unsupervised manner. Extensive experiments conducted on four synthetic-to-real and clear-to-adverse weather benchmarks demonstrate the effectiveness of our approaches. Without resorting to any complex techniques, such as image translation, augmentation, or rare-class sampling, we set a new state-of-the-art on all benchmarks. Our implementation will be publicly available at \url{https://github.com/ETHRuiGong/PTDiffSeg}.

{{</citation>}}


### (46/115) MDViT: Multi-domain Vision Transformer for Small Medical Image Segmentation Datasets (Siyi Du et al., 2023)

{{<citation>}}

Siyi Du, Nourhan Bayasi, Ghassan Harmarneh, Rafeef Garbi. (2023)  
**MDViT: Multi-domain Vision Transformer for Small Medical Image Segmentation Datasets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.02100v1)  

---


**ABSTRACT**  
Despite its clinical utility, medical image segmentation (MIS) remains a daunting task due to images' inherent complexity and variability. Vision transformers (ViTs) have recently emerged as a promising solution to improve MIS; however, they require larger training datasets than convolutional neural networks. To overcome this obstacle, data-efficient ViTs were proposed, but they are typically trained using a single source of data, which overlooks the valuable knowledge that could be leveraged from other available datasets. Naivly combining datasets from different domains can result in negative knowledge transfer (NKT), i.e., a decrease in model performance on some domains with non-negligible inter-domain heterogeneity. In this paper, we propose MDViT, the first multi-domain ViT that includes domain adapters to mitigate data-hunger and combat NKT by adaptively exploiting knowledge in multiple small data resources (domains). Further, to enhance representation learning across domains, we integrate a mutual knowledge distillation paradigm that transfers knowledge between a universal network (spanning all the domains) and auxiliary domain-specific branches. Experiments on 4 skin lesion segmentation datasets show that MDViT outperforms state-of-the-art algorithms, with superior segmentation performance and a fixed model size, at inference time, even as more domains are added. Our code is available at https://github.com/siyi-wind/MDViT.

{{</citation>}}


### (47/115) Adversarial Attacks on Image Classification Models: FGSM and Patch Attacks and their Impact (Jaydip Sen et al., 2023)

{{<citation>}}

Jaydip Sen, Subhasis Dasgupta. (2023)  
**Adversarial Attacks on Image Classification Models: FGSM and Patch Attacks and their Impact**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: Adversarial Attack, Google, Image Classification, ImageNet  
[Paper Link](http://arxiv.org/abs/2307.02055v1)  

---


**ABSTRACT**  
This chapter introduces the concept of adversarial attacks on image classification models built on convolutional neural networks (CNN). CNNs are very popular deep-learning models which are used in image classification tasks. However, very powerful and pre-trained CNN models working very accurately on image datasets for image classification tasks may perform disastrously when the networks are under adversarial attacks. In this work, two very well-known adversarial attacks are discussed and their impact on the performance of image classifiers is analyzed. These two adversarial attacks are the fast gradient sign method (FGSM) and adversarial patch attack. These attacks are launched on three powerful pre-trained image classifier architectures, ResNet-34, GoogleNet, and DenseNet-161. The classification accuracy of the models in the absence and presence of the two attacks are computed on images from the publicly accessible ImageNet dataset. The results are analyzed to evaluate the impact of the attacks on the image classification task.

{{</citation>}}


### (48/115) Generative Adversarial Networks for Dental Patient Identity Protection in Orthodontic Educational Imaging (Mingchuan Tian et al., 2023)

{{<citation>}}

Mingchuan Tian, Wilson Weixun Lu, Kelvin Weng Chiong Foong, Eugene Loh. (2023)  
**Generative Adversarial Networks for Dental Patient Identity Protection in Orthodontic Educational Imaging**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2307.02019v1)  

---


**ABSTRACT**  
Objectives: This research introduces a novel area-preserving Generative Adversarial Networks (GAN) inversion technique for effectively de-identifying dental patient images. This innovative method addresses privacy concerns while preserving key dental features, thereby generating valuable resources for dental education and research.   Methods: We enhanced the existing GAN Inversion methodology to maximize the preservation of dental characteristics within the synthesized images. A comprehensive technical framework incorporating several deep learning models was developed to provide end-to-end development guidance and practical application for image de-identification.   Results: Our approach was assessed with varied facial pictures, extensively used for diagnosing skeletal asymmetry and facial anomalies. Results demonstrated our model's ability to adapt the context from one image to another, maintaining compatibility, while preserving dental features essential for oral diagnosis and dental education. A panel of five clinicians conducted an evaluation on a set of original and GAN-processed images. The generated images achieved effective de-identification, maintaining the realism of important dental features and were deemed useful for dental diagnostics and education.   Clinical Significance: Our GAN model and the encompassing framework can streamline the de-identification process of dental patient images, enhancing efficiency in dental education. This method improves students' diagnostic capabilities by offering more exposure to orthodontic malocclusions. Furthermore, it facilitates the creation of de-identified datasets for broader 2D image research at major research institutions.

{{</citation>}}


### (49/115) ZJU ReLER Submission for EPIC-KITCHEN Challenge 2023: TREK-150 Single Object Tracking (Yuanyou Xu et al., 2023)

{{<citation>}}

Yuanyou Xu, Jiahao Li, Zongxin Yang, Yi Yang, Yueting Zhuang. (2023)  
**ZJU ReLER Submission for EPIC-KITCHEN Challenge 2023: TREK-150 Single Object Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.02508v2)  

---


**ABSTRACT**  
The Associating Objects with Transformers (AOT) framework has exhibited exceptional performance in a wide range of complex scenarios for video object tracking and segmentation. In this study, we convert the bounding boxes to masks in reference frames with the help of the Segment Anything Model (SAM) and Alpha-Refine, and then propagate the masks to the current frame, transforming the task from Video Object Tracking (VOT) to video object segmentation (VOS). Furthermore, we introduce MSDeAOT, a variant of the AOT series that incorporates transformers at multiple feature scales. MSDeAOT efficiently propagates object masks from previous frames to the current frame using two feature scales of 16 and 8. As a testament to the effectiveness of our design, we achieved the 1st place in the EPIC-KITCHENS TREK-150 Object Tracking Challenge.

{{</citation>}}


### (50/115) ZJU ReLER Submission for EPIC-KITCHEN Challenge 2023: Semi-Supervised Video Object Segmentation (Jiahao Li et al., 2023)

{{<citation>}}

Jiahao Li, Yuanyou Xu, Zongxin Yang, Yi Yang, Yueting Zhuang. (2023)  
**ZJU ReLER Submission for EPIC-KITCHEN Challenge 2023: Semi-Supervised Video Object Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.02010v2)  

---


**ABSTRACT**  
The Associating Objects with Transformers (AOT) framework has exhibited exceptional performance in a wide range of complex scenarios for video object segmentation. In this study, we introduce MSDeAOT, a variant of the AOT series that incorporates transformers at multiple feature scales. Leveraging the hierarchical Gated Propagation Module (GPM), MSDeAOT efficiently propagates object masks from previous frames to the current frame using a feature scale with a stride of 16. Additionally, we employ GPM in a more refined feature scale with a stride of 8, leading to improved accuracy in detecting and tracking small objects. Through the implementation of test-time augmentations and model ensemble techniques, we achieve the top-ranking position in the EPIC-KITCHEN VISOR Semi-supervised Video Object Segmentation Challenge.

{{</citation>}}


### (51/115) Multi-Modal Prototypes for Open-Set Semantic Segmentation (Yuhuan Yang et al., 2023)

{{<citation>}}

Yuhuan Yang, Chaofan Ma, Chen Ju, Ya Zhang, Yanfeng Wang. (2023)  
**Multi-Modal Prototypes for Open-Set Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.02003v1)  

---


**ABSTRACT**  
In semantic segmentation, adapting a visual system to novel object categories at inference time has always been both valuable and challenging. To enable such generalization, existing methods rely on either providing several support examples as visual cues or class names as textual cues. Through the development is relatively optimistic, these two lines have been studied in isolation, neglecting the complementary intrinsic of low-level visual and high-level language information. In this paper, we define a unified setting termed as open-set semantic segmentation (O3S), which aims to learn seen and unseen semantics from both visual examples and textual names. Our pipeline extracts multi-modal prototypes for segmentation task, by first single modal self-enhancement and aggregation, then multi-modal complementary fusion. To be specific, we aggregate visual features into several tokens as visual prototypes, and enhance the class name with detailed descriptions for textual prototype generation. The two modalities are then fused to generate multi-modal prototypes for final segmentation. On both \pascal and \coco datasets, we conduct extensive experiments to evaluate the framework effectiveness. State-of-the-art results are achieved even on more detailed part-segmentation, Pascal-Animals, by only training on coarse-grained datasets. Thorough ablation studies are performed to dissect each component, both quantitatively and qualitatively.

{{</citation>}}


### (52/115) Task-Specific Alignment and Multiple Level Transformer for Few-Shot Action Recognition (Fei Guo et al., 2023)

{{<citation>}}

Fei Guo, Li Zhu, YiWang Wang. (2023)  
**Task-Specific Alignment and Multiple Level Transformer for Few-Shot Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-DM, cs.CV  
Keywords: Few-Shot, Transformer  
[Paper Link](http://arxiv.org/abs/2307.01985v1)  

---


**ABSTRACT**  
In the research field of few-shot learning, the main difference between image-based and video-based is the additional temporal dimension for videos. In recent years, many approaches for few-shot action recognition have followed the metric-based methods, especially, since some works use the Transformer to get the cross-attention feature of the videos or the enhanced prototype, and the results are competitive. However, they do not mine enough information from the Transformer because they only focus on the feature of a single level. In our paper, we have addressed this problem. We propose an end-to-end method named "Task-Specific Alignment and Multiple Level Transformer Network (TSA-MLT)". In our model, the Multiple Level Transformer focuses on the multiple-level feature of the support video and query video. Especially before Multiple Level Transformer, we use task-specific TSA to filter unimportant or misleading frames as a pre-processing. Furthermore, we adopt a fusion loss using two kinds of distance, the first is L2 sequence distance, which focuses on temporal order alignment. The second one is Optimal transport distance, which focuses on measuring the gap between the appearance and semantics of the videos. Using a simple fusion network, we fuse the two distances element-wise, then use the cross-entropy loss as our fusion loss. Extensive experiments show our method achieves state-of-the-art results on the HMDB51 and UCF101 datasets and a competitive result on the benchmark of Kinetics and something-2-something V2 datasets. Our code will be available at the URL: https://github.com/cofly2014/tsa-mlt.git

{{</citation>}}


### (53/115) The KiTS21 Challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in corticomedullary-phase CT (Nicholas Heller et al., 2023)

{{<citation>}}

Nicholas Heller, Fabian Isensee, Dasha Trofimova, Resha Tejpaul, Zhongchen Zhao, Huai Chen, Lisheng Wang, Alex Golts, Daniel Khapun, Daniel Shats, Yoel Shoshan, Flora Gilboa-Solomon, Yasmeen George, Xi Yang, Jianpeng Zhang, Jing Zhang, Yong Xia, Mengran Wu, Zhiyang Liu, Ed Walczak, Sean McSweeney, Ranveer Vasdev, Chris Hornung, Rafat Solaiman, Jamee Schoephoerster, Bailey Abernathy, David Wu, Safa Abdulkadir, Ben Byun, Justice Spriggs, Griffin Struyk, Alexandra Austin, Ben Simpson, Michael Hagstrom, Sierra Virnig, John French, Nitin Venkatesh, Sarah Chan, Keenan Moore, Anna Jacobsen, Susan Austin, Mark Austin, Subodh Regmi, Nikolaos Papanikolopoulos, Christopher Weight. (2023)  
**The KiTS21 Challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in corticomedullary-phase CT**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.01984v1)  

---


**ABSTRACT**  
This paper presents the challenge report for the 2021 Kidney and Kidney Tumor Segmentation Challenge (KiTS21) held in conjunction with the 2021 international conference on Medical Image Computing and Computer Assisted Interventions (MICCAI). KiTS21 is a sequel to its first edition in 2019, and it features a variety of innovations in how the challenge was designed, in addition to a larger dataset. A novel annotation method was used to collect three separate annotations for each region of interest, and these annotations were performed in a fully transparent setting using a web-based annotation tool. Further, the KiTS21 test set was collected from an outside institution, challenging participants to develop methods that generalize well to new populations. Nonetheless, the top-performing teams achieved a significant improvement over the state of the art set in 2019, and this performance is shown to inch ever closer to human-level performance. An in-depth meta-analysis is presented describing which methods were used and how they faired on the leaderboard, as well as the characteristics of which cases generally saw good performance, and which did not. Overall KiTS21 facilitated a significant advancement in the state of the art in kidney tumor segmentation, and provides useful insights that are applicable to the field of semantic segmentation as a whole.

{{</citation>}}


### (54/115) Muti-scale Graph Neural Network with Signed-attention for Social Bot Detection: A Frequency Perspective (Shuhao Shi et al., 2023)

{{<citation>}}

Shuhao Shi, Kai Qiao, Zhengyan Wang, Jie Yang, Baojie Song, Jian Chen, Bin Yan. (2023)  
**Muti-scale Graph Neural Network with Signed-attention for Social Bot Detection: A Frequency Perspective**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2307.01968v1)  

---


**ABSTRACT**  
The presence of a large number of bots on social media has adverse effects. The graph neural network (GNN) can effectively leverage the social relationships between users and achieve excellent results in detecting bots. Recently, more and more GNN-based methods have been proposed for bot detection. However, the existing GNN-based bot detection methods only focus on low-frequency information and seldom consider high-frequency information, which limits the representation ability of the model. To address this issue, this paper proposes a Multi-scale with Signed-attention Graph Filter for social bot detection called MSGS. MSGS could effectively utilize both high and low-frequency information in the social graph. Specifically, MSGS utilizes a multi-scale structure to produce representation vectors at different scales. These representations are then combined using a signed-attention mechanism. Finally, multi-scale representations via MLP after polymerization to produce the final result. We analyze the frequency response and demonstrate that MSGS is a more flexible and expressive adaptive graph filter. MSGS can effectively utilize high-frequency information to alleviate the over-smoothing problem of deep GNNs. Experimental results on real-world datasets demonstrate that our method achieves better performance compared with several state-of-the-art social bot detection methods.

{{</citation>}}


## math.OC (1)



### (55/115) AI4OPT: AI Institute for Advances in Optimization (Pascal Van Hentenryck et al., 2023)

{{<citation>}}

Pascal Van Hentenryck, Kevin Dalmeijer. (2023)  
**AI4OPT: AI Institute for Advances in Optimization**  

---
Primary Category: math.OC  
Categories: cs-AI, math-OC, math.OC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02671v1)  

---


**ABSTRACT**  
This article is a short introduction to AI4OPT, the NSF AI Institute for Advances in Optimization. AI4OPT fuses AI and Optimization, inspired by end-use cases in supply chains, energy systems, chip design and manufacturing, and sustainable food systems. AI4OPT also applies its "teaching the teachers" philosophy to provide longitudinal educational pathways in AI for engineering.

{{</citation>}}


## cs.AR (1)



### (56/115) Chiplet Cloud: Building AI Supercomputers for Serving Large Generative Language Models (Huwan Peng et al., 2023)

{{<citation>}}

Huwan Peng, Scott Davidson, Richard Shi, Shuaiwen Leon Song, Michael Taylor. (2023)  
**Chiplet Cloud: Building AI Supercomputers for Serving Large Generative Language Models**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.02666v2)  

---


**ABSTRACT**  
Large language models (LLMs) such as ChatGPT have demonstrated unprecedented capabilities in multiple AI tasks. However, hardware inefficiencies have become a significant factor limiting the democratization of LLMs. We propose Chiplet Cloud, an ASIC supercomputer architecture that optimizes total cost of ownership (TCO) per token for serving generative LLMs. Chiplet Cloud fits all model parameters inside the on-chip SRAMs to eliminate bandwidth limitations while moderating the die size to improve system costs while leveraging software mappings to overcome data communication overhead. We propose a comprehensive design methodology that accurately explores a spectrum of major design trade-offs in the joint space of hardware-software and generates a detailed performance-cost analysis on all valid design points. We evaluate Chiplet Cloud on four popular LLMs. Compared to GPU and TPU, our architecture can achieve up to 94x and 15x improvement in TCO/Token respectively, significantly reducing the cost for realistically serving modern LLMs.

{{</citation>}}


## cs.IT (2)



### (57/115) Achievable Rates for Information Extraction from a Strategic Sender (Anuj S. Vora et al., 2023)

{{<citation>}}

Anuj S. Vora, Ankur A. Kulkarni. (2023)  
**Achievable Rates for Information Extraction from a Strategic Sender**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2307.02644v1)  

---


**ABSTRACT**  
We consider a setting of non-cooperative communication where a receiver wants to recover randomly generated sequences of symbols that are observed by a strategic sender. The sender aims to maximize an average utility that may not align with the recovery criterion of the receiver, whereby the received signals may not be truthful. We pose this problem as a sequential game between the sender and the receiver with the receiver as the leader and determine `achievable strategies' for the receiver that attain arbitrarily small probability of error for large blocklengths. We show the existence of such achievable strategies under a sufficient condition on the utility of the sender. For the case of the binary alphabet, this condition is also necessary, in the absence of which, the probability of error goes to one for all choices of strategies of the receiver. We show that for reliable recovery, the receiver chooses to correctly decode only a subset of messages received from the sender and deliberately makes an error on messages outside this subset. Due to this decoding strategy, despite a clean channel, our setting exhibits a notion of maximum rate of communication above which the probability of error may not vanish asymptotically and in certain cases, may even tend to one. For the case of the binary alphabet, the maximum rate may be strictly less than unity for certain classes of utilities.

{{</citation>}}


### (58/115) Physical Layer Secret Key Agreement Using One-Bit Quantization and Low-Density Parity-Check Codes (John A. Snoap, 2023)

{{<citation>}}

John A. Snoap. (2023)  
**Physical Layer Secret Key Agreement Using One-Bit Quantization and Low-Density Parity-Check Codes**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2307.02391v2)  

---


**ABSTRACT**  
Physical layer approaches for generating secret encryption keys for wireless systems using channel information have attracted increased interest from researchers in recent years. This paper presents a new approach for calculating log-likelihood ratios (LLRs) for secret key generation that is based on one-bit quantization of channel measurements and the difference between channel estimates at legitimate reciprocal nodes. The studied secret key agreement approach, which implements advantage distillation along with information reconciliation using Slepian-Wolf low-density parity-check (LDPC) codes, is discussed and illustrated with numerical results obtained from simulations. These results show the probability of bit disagreement for keys generated using the proposed LLR calculations compared with alternative LLR calculation methods for key generation based on channel state information. The proposed LLR calculations are shown to be an improvement to the studied approach of physical layer secret key agreement.

{{</citation>}}


## cs.HC (2)



### (59/115) UX Heuristics and Checklist for Deep Learning powered Mobile Applications with Image Classification (Christiane Gresse von Wangenheim et al., 2023)

{{<citation>}}

Christiane Gresse von Wangenheim, Gustavo Dirschnabel. (2023)  
**UX Heuristics and Checklist for Deep Learning powered Mobile Applications with Image Classification**  

---
Primary Category: cs.HC  
Categories: I-4-9, cs-AI, cs-HC, cs.HC, none  
Keywords: AI, Image Classification  
[Paper Link](http://arxiv.org/abs/2307.05513v1)  

---


**ABSTRACT**  
Advances in mobile applications providing image classification enabled by Deep Learning require innovative User Experience solutions in order to assure their adequate use by users. To aid the design process, usability heuristics are typically customized for a specific kind of application. Therefore, based on a literature review and analyzing existing mobile applications with image classification, we propose an initial set of AIX heuristics for Deep Learning powered mobile applications with image classification decomposed into a checklist. In order to facilitate the usage of the checklist we also developed an online course presenting the concepts and heuristics as well as a web-based tool in order to support an evaluation using these heuristics. These results of this research can be used to guide the design of the interfaces of such applications as well as support the conduction of heuristic evaluations supporting practitioners to develop image classification apps that people can understand, trust, and can engage with effectively.

{{</citation>}}


### (60/115) Power-up! What Can Generative Models Do for Human Computation Workflows? (Garrett Allen et al., 2023)

{{<citation>}}

Garrett Allen, Gaole He, Ujwal Gadiraju. (2023)  
**Power-up! What Can Generative Models Do for Human Computation Workflows?**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02243v1)  

---


**ABSTRACT**  
We are amidst an explosion of artificial intelligence research, particularly around large language models (LLMs). These models have a range of applications across domains like medicine, finance, commonsense knowledge graphs, and crowdsourcing. Investigation into LLMs as part of crowdsourcing workflows remains an under-explored space. The crowdsourcing research community has produced a body of work investigating workflows and methods for managing complex tasks using hybrid human-AI methods. Within crowdsourcing, the role of LLMs can be envisioned as akin to a cog in a larger wheel of workflows. From an empirical standpoint, little is currently understood about how LLMs can improve the effectiveness of crowdsourcing workflows and how such workflows can be evaluated. In this work, we present a vision for exploring this gap from the perspectives of various stakeholders involved in the crowdsourcing paradigm -- the task requesters, crowd workers, platforms, and end-users. We identify junctures in typical crowdsourcing workflows at which the introduction of LLMs can play a beneficial role and propose means to augment existing design patterns for crowd work.

{{</citation>}}


## cs.AI (5)



### (61/115) Surge Routing: Event-informed Multiagent Reinforcement Learning for Autonomous Rideshare (Daniel Garces et al., 2023)

{{<citation>}}

Daniel Garces, Stephanie Gil. (2023)  
**Surge Routing: Event-informed Multiagent Reinforcement Learning for Autonomous Rideshare**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-MA, cs-RO, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02637v1)  

---


**ABSTRACT**  
Large events such as conferences, concerts and sports games, often cause surges in demand for ride services that are not captured in average demand patterns, posing unique challenges for routing algorithms. We propose a learning framework for an autonomous fleet of taxis that scrapes event data from the internet to predict and adapt to surges in demand and generates cooperative routing and pickup policies that service a higher number of requests than other routing protocols. We achieve this through a combination of (i) an event processing framework that scrapes the internet for event information and generates dense vector representations that can be used as input features for a neural network that predicts demand; (ii) a two neural network system that predicts hourly demand over the entire map, using these dense vector representations; (iii) a probabilistic approach that leverages locale occupancy schedules to map publicly available demand data over sectors to discretized street intersections; and finally, (iv) a scalable model-based reinforcement learning framework that uses the predicted demand over intersections to anticipate surges and route taxis using one-agent-at-a-time rollout with limited sampling certainty equivalence. We learn routing and pickup policies using real NYC ride share data for 2022 and information for more than 2000 events across 300 unique venues in Manhattan. We test our approach with a fleet of 100 taxis on a map with 38 different sectors (2235 street intersections). Our experimental results demonstrate that our method obtains routing policies that service $6$ more requests on average per minute (around $360$ more requests per hour) than other model-based RL frameworks and other classical algorithms in operations research when dealing with surge demand conditions.

{{</citation>}}


### (62/115) Building Cooperative Embodied Agents Modularly with Large Language Models (Hongxin Zhang et al., 2023)

{{<citation>}}

Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B. Tenenbaum, Tianmin Shu, Chuang Gan. (2023)  
**Building Cooperative Embodied Agents Modularly with Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs.AI  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.02485v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated impressive planning abilities in single-agent embodied tasks across various domains. However, their capacity for planning and communication in multi-agent cooperation remains unclear, even though these are crucial skills for intelligent embodied agents. In this paper, we present a novel framework that utilizes LLMs for multi-agent cooperation and tests it in various embodied environments. Our framework enables embodied agents to plan, communicate, and cooperate with other embodied agents or humans to accomplish long-horizon tasks efficiently. We demonstrate that recent LLMs, such as GPT-4, can surpass strong planning-based methods and exhibit emergent effective communication using our framework without requiring fine-tuning or few-shot prompting. We also discover that LLM-based agents that communicate in natural language can earn more trust and cooperate more effectively with humans. Our research underscores the potential of LLMs for embodied AI and lays the foundation for future research in multi-agent cooperation. Videos can be found on the project website https://vis-www.cs.umass.edu/Co-LLM-Agents/.

{{</citation>}}


### (63/115) Causal Discovery with Language Models as Imperfect Experts (Stephanie Long et al., 2023)

{{<citation>}}

Stephanie Long, Alexandre Piché, Valentina Zantedeschi, Tibor Schuster, Alexandre Drouin. (2023)  
**Causal Discovery with Language Models as Imperfect Experts**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.02390v1)  

---


**ABSTRACT**  
Understanding the causal relationships that underlie a system is a fundamental prerequisite to accurate decision-making. In this work, we explore how expert knowledge can be used to improve the data-driven identification of causal graphs, beyond Markov equivalence classes. In doing so, we consider a setting where we can query an expert about the orientation of causal relationships between variables, but where the expert may provide erroneous information. We propose strategies for amending such expert knowledge based on consistency properties, e.g., acyclicity and conditional independencies in the equivalence class. We then report a case study, on real data, where a large language model is used as an imperfect expert.

{{</citation>}}


### (64/115) Beyond Known Reality: Exploiting Counterfactual Explanations for Medical Research (Toygar Tanyel et al., 2023)

{{<citation>}}

Toygar Tanyel, Serkan Ayvaz, Bilgin Keserci. (2023)  
**Beyond Known Reality: Exploiting Counterfactual Explanations for Medical Research**  

---
Primary Category: cs.AI  
Categories: J-3, cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02131v1)  

---


**ABSTRACT**  
This study employs counterfactual explanations to explore "what if?" scenarios in medical research, with the aim of expanding our understanding beyond existing boundaries. Specifically, we focus on utilizing MRI features for diagnosing pediatric posterior fossa brain tumors as a case study. The field of artificial intelligence and explainability has witnessed a growing number of studies and increasing scholarly interest. However, the lack of human-friendly interpretations in explaining the outcomes of machine learning algorithms has significantly hindered the acceptance of these methods by clinicians in their clinical practice. To address this, our approach incorporates counterfactual explanations, providing a novel way to examine alternative decision-making scenarios. These explanations offer personalized and context-specific insights, enabling the validation of predictions and clarification of variations under diverse circumstances. Importantly, our approach maintains both statistical and clinical fidelity, allowing for the examination of distinct tumor features through alternative realities. Additionally, we explore the potential use of counterfactuals for data augmentation and evaluate their feasibility as an alternative approach in medical research. The results demonstrate the promising potential of counterfactual explanations to enhance trust and acceptance of AI-driven methods in clinical settings.

{{</citation>}}


### (65/115) Combating Confirmation Bias: A Unified Pseudo-Labeling Framework for Entity Alignment (Qijie Ding et al., 2023)

{{<citation>}}

Qijie Ding, Jie Yin, Daokun Zhang, Junbin Gao. (2023)  
**Combating Confirmation Bias: A Unified Pseudo-Labeling Framework for Entity Alignment**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Bias, Entity Alignment  
[Paper Link](http://arxiv.org/abs/2307.02075v1)  

---


**ABSTRACT**  
Entity alignment (EA) aims at identifying equivalent entity pairs across different knowledge graphs (KGs) that refer to the same real-world identity. To systematically combat confirmation bias for pseudo-labeling-based entity alignment, we propose a Unified Pseudo-Labeling framework for Entity Alignment (UPL-EA) that explicitly eliminates pseudo-labeling errors to boost the accuracy of entity alignment. UPL-EA consists of two complementary components: (1) The Optimal Transport (OT)-based pseudo-labeling uses discrete OT modeling as an effective means to enable more accurate determination of entity correspondences across two KGs and to mitigate the adverse impact of erroneous matches. A simple but highly effective criterion is further devised to derive pseudo-labeled entity pairs that satisfy one-to-one correspondences at each iteration. (2) The cross-iteration pseudo-label calibration operates across multiple consecutive iterations to further improve the pseudo-labeling precision rate by reducing the local pseudo-label selection variability with a theoretical guarantee. The two components are respectively designed to eliminate Type I and Type II pseudo-labeling errors identified through our analyse. The calibrated pseudo-labels are thereafter used to augment prior alignment seeds to reinforce subsequent model training for alignment inference. The effectiveness of UPL-EA in eliminating pseudo-labeling errors is both theoretically supported and experimentally validated. The experimental results show that our approach achieves competitive performance with limited prior alignment seeds.

{{</citation>}}


## cs.LG (26)



### (66/115) Dynamic Observation Policies in Observation Cost-Sensitive Reinforcement Learning (Colin Bellinger et al., 2023)

{{<citation>}}

Colin Bellinger, Mark Crowley, Isaac Tamblyn. (2023)  
**Dynamic Observation Policies in Observation Cost-Sensitive Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: 68T01, I-2-0, cs-AI, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02620v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) has been shown to learn sophisticated control policies for complex tasks including games, robotics, heating and cooling systems and text generation. The action-perception cycle in RL, however, generally assumes that a measurement of the state of the environment is available at each time step without a cost. In applications such as deep-sea and planetary robot exploration, materials design and medicine, however, there can be a high cost associated with measuring, or even approximating, the state of the environment. In this paper, we survey the recently growing literature that adopts the perspective that an RL agent might not need, or even want, a costly measurement at each time step. Within this context, we propose the Deep Dynamic Multi-Step Observationless Agent (DMSOA), contrast it with the literature and empirically evaluate it on OpenAI gym and Atari Pong environments. Our results, show that DMSOA learns a better policy with fewer decision steps and measurements than the considered alternative from the literature.

{{</citation>}}


### (67/115) Additive Decoders for Latent Variables Identification and Cartesian-Product Extrapolation (Sébastien Lachapelle et al., 2023)

{{<citation>}}

Sébastien Lachapelle, Divyat Mahajan, Ioannis Mitliagkas, Simon Lacoste-Julien. (2023)  
**Additive Decoders for Latent Variables Identification and Cartesian-Product Extrapolation**  

---
Primary Category: cs.LG  
Categories: I-2-6; I-5-1, cs-LG, cs.LG, stat-ML  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2307.02598v1)  

---


**ABSTRACT**  
We tackle the problems of latent variables identification and "out-of-support" image generation in representation learning. We show that both are possible for a class of decoders that we call additive, which are reminiscent of decoders used for object-centric representation learning (OCRL) and well suited for images that can be decomposed as a sum of object-specific images. We provide conditions under which exactly solving the reconstruction problem using an additive decoder is guaranteed to identify the blocks of latent variables up to permutation and block-wise invertible transformations. This guarantee relies only on very weak assumptions about the distribution of the latent factors, which might present statistical dependencies and have an almost arbitrarily shaped support. Our result provides a new setting where nonlinear independent component analysis (ICA) is possible and adds to our theoretical understanding of OCRL methods. We also show theoretically that additive decoders can generate novel images by recombining observed factors of variations in novel ways, an ability we refer to as Cartesian-product extrapolation. We show empirically that additivity is crucial for both identifiability and extrapolation on simulated data.

{{</citation>}}


### (68/115) TransformerG2G: Adaptive time-stepping for learning temporal graph embeddings using transformers (Alan John Varghese et al., 2023)

{{<citation>}}

Alan John Varghese, Aniruddha Bora, Mengjia Xu, George Em Karniadakis. (2023)  
**TransformerG2G: Adaptive time-stepping for learning temporal graph embeddings using transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-DS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.02588v1)  

---


**ABSTRACT**  
Dynamic graph embedding has emerged as a very effective technique for addressing diverse temporal graph analytic tasks (i.e., link prediction, node classification, recommender systems, anomaly detection, and graph generation) in various applications. Such temporal graphs exhibit heterogeneous transient dynamics, varying time intervals, and highly evolving node features throughout their evolution. Hence, incorporating long-range dependencies from the historical graph context plays a crucial role in accurately learning their temporal dynamics. In this paper, we develop a graph embedding model with uncertainty quantification, TransformerG2G, by exploiting the advanced transformer encoder to first learn intermediate node representations from its current state ($t$) and previous context (over timestamps [$t-1, t-l$], $l$ is the length of context). Moreover, we employ two projection layers to generate lower-dimensional multivariate Gaussian distributions as each node's latent embedding at timestamp $t$. We consider diverse benchmarks with varying levels of ``novelty" as measured by the TEA plots. Our experiments demonstrate that the proposed TransformerG2G model outperforms conventional multi-step methods and our prior work (DynG2G) in terms of both link prediction accuracy and computational efficiency, especially for high degree of novelty. Furthermore, the learned time-dependent attention weights across multiple graph snapshots reveal the development of an automatic adaptive time stepping enabled by the transformer. Importantly, by examining the attention weights, we can uncover temporal dependencies, identify influential elements, and gain insights into the complex interactions within the graph structure. For example, we identified a strong correlation between attention weights and node degree at the various stages of the graph topology evolution.

{{</citation>}}


### (69/115) Multimodal Temporal Fusion Transformers Are Good Product Demand Forecasters (Maarten Sukel et al., 2023)

{{<citation>}}

Maarten Sukel, Stevan Rudinac, Marcel Worring. (2023)  
**Multimodal Temporal Fusion Transformers Are Good Product Demand Forecasters**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.02578v1)  

---


**ABSTRACT**  
Multimodal demand forecasting aims at predicting product demand utilizing visual, textual, and contextual information. This paper proposes a method for multimodal product demand forecasting using convolutional, graph-based, and transformer-based architectures. Traditional approaches to demand forecasting rely on historical demand, product categories, and additional contextual information such as seasonality and events. However, these approaches have several shortcomings, such as the cold start problem making it difficult to predict product demand until sufficient historical data is available for a particular product, and their inability to properly deal with category dynamics. By incorporating multimodal information, such as product images and textual descriptions, our architecture aims to address the shortcomings of traditional approaches and outperform them. The experiments conducted on a large real-world dataset show that the proposed approach effectively predicts demand for a wide range of products. The multimodal pipeline presented in this work enhances the accuracy and reliability of the predictions, demonstrating the potential of leveraging multimodal information in product demand forecasting.

{{</citation>}}


### (70/115) Elastic Decision Transformer (Yueh-Hua Wu et al., 2023)

{{<citation>}}

Yueh-Hua Wu, Xiaolong Wang, Masashi Hamaya. (2023)  
**Elastic Decision Transformer**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.02484v2)  

---


**ABSTRACT**  
This paper introduces Elastic Decision Transformer (EDT), a significant advancement over the existing Decision Transformer (DT) and its variants. Although DT purports to generate an optimal trajectory, empirical evidence suggests it struggles with trajectory stitching, a process involving the generation of an optimal or near-optimal trajectory from the best parts of a set of sub-optimal trajectories. The proposed EDT differentiates itself by facilitating trajectory stitching during action inference at test time, achieved by adjusting the history length maintained in DT. Further, the EDT optimizes the trajectory by retaining a longer history when the previous trajectory is optimal and a shorter one when it is sub-optimal, enabling it to "stitch" with a more optimal trajectory. Extensive experimentation demonstrates EDT's ability to bridge the performance gap between DT-based and Q Learning-based approaches. In particular, the EDT outperforms Q Learning-based methods in a multi-task regime on the D4RL locomotion benchmark and Atari games. Videos are available at: https://kristery.github.io/edt/

{{</citation>}}


### (71/115) Jailbroken: How Does LLM Safety Training Fail? (Alexander Wei et al., 2023)

{{<citation>}}

Alexander Wei, Nika Haghtalab, Jacob Steinhardt. (2023)  
**Jailbroken: How Does LLM Safety Training Fail?**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: AI, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2307.02483v1)  

---


**ABSTRACT**  
Large language models trained for safety and harmlessness remain susceptible to adversarial misuse, as evidenced by the prevalence of "jailbreak" attacks on early releases of ChatGPT that elicit undesired behavior. Going beyond recognition of the issue, we investigate why such attacks succeed and how they can be created. We hypothesize two failure modes of safety training: competing objectives and mismatched generalization. Competing objectives arise when a model's capabilities and safety goals conflict, while mismatched generalization occurs when safety training fails to generalize to a domain for which capabilities exist. We use these failure modes to guide jailbreak design and then evaluate state-of-the-art models, including OpenAI's GPT-4 and Anthropic's Claude v1.3, against both existing and newly designed attacks. We find that vulnerabilities persist despite the extensive red-teaming and safety-training efforts behind these models. Notably, new attacks utilizing our failure modes succeed on every prompt in a collection of unsafe requests from the models' red-teaming evaluation sets and outperform existing ad hoc jailbreaks. Our analysis emphasizes the need for safety-capability parity -- that safety mechanisms should be as sophisticated as the underlying model -- and argues against the idea that scaling alone can resolve these safety failure modes.

{{</citation>}}


### (72/115) Exploring Continual Learning for Code Generation Models (Prateek Yadav et al., 2023)

{{<citation>}}

Prateek Yadav, Qing Sun, Hantian Ding, Xiaopeng Li, Dejiao Zhang, Ming Tan, Xiaofei Ma, Parminder Bhatia, Ramesh Nallapati, Murali Krishna Ramanathan, Mohit Bansal, Bing Xiang. (2023)  
**Exploring Continual Learning for Code Generation Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs-SE, cs.LG  
Keywords: NLP, T5  
[Paper Link](http://arxiv.org/abs/2307.02435v1)  

---


**ABSTRACT**  
Large-scale code generation models such as Codex and CodeT5 have achieved impressive performance. However, libraries are upgraded or deprecated very frequently and re-training large-scale language models is computationally expensive. Therefore, Continual Learning (CL) is an important aspect that remains underexplored in the code domain. In this paper, we introduce a benchmark called CodeTask-CL that covers a wide range of tasks, including code generation, translation, summarization, and refinement, with different input and output programming languages. Next, on our CodeTask-CL benchmark, we compare popular CL techniques from NLP and Vision domains. We find that effective methods like Prompt Pooling (PP) suffer from catastrophic forgetting due to the unstable training of the prompt selection mechanism caused by stark distribution shifts in coding tasks. We address this issue with our proposed method, Prompt Pooling with Teacher Forcing (PP-TF), that stabilizes training by enforcing constraints on the prompt selection mechanism and leads to a 21.54% improvement over Prompt Pooling. Along with the benchmark, we establish a training pipeline that can be used for CL on code models, which we believe can motivate further development of CL methods for code models. Our code is available at https://github.com/amazon-science/codetaskcl-pptf

{{</citation>}}


### (73/115) In-Context Learning for Attention Scheme: from Single Softmax Regression to Multiple Softmax Regression via a Tensor Trick (Yeqi Gao et al., 2023)

{{<citation>}}

Yeqi Gao, Zhao Song, Shenghao Xie. (2023)  
**In-Context Learning for Attention Scheme: from Single Softmax Regression to Multiple Softmax Regression via a Tensor Trick**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.02419v1)  

---


**ABSTRACT**  
Large language models (LLMs) have brought significant and transformative changes in human society. These models have demonstrated remarkable capabilities in natural language understanding and generation, leading to various advancements and impacts across several domains.   We consider the in-context learning under two formulation for attention related regression in this work. Given matrices $A_1 \in \mathbb{R}^{n \times d}$, and $A_2 \in \mathbb{R}^{n \times d}$ and $B \in \mathbb{R}^{n \times n}$, the purpose is to solve some certain optimization problems: Normalized version $\min_{X} \| D(X)^{-1} \exp(A_1 X A_2^\top) - B \|_F^2$ and Rescaled version $\| \exp(A_1 X A_2^\top) - D(X) \cdot B \|_F^2$. Here $D(X) := \mathrm{diag}( \exp(A_1 X A_2^\top) {\bf 1}_n )$.   Our regression problem shares similarities with previous studies on softmax-related regression. Prior research has extensively investigated regression techniques related to softmax regression: Normalized version $\| \langle \exp(Ax) , {\bf 1}_n \rangle^{-1} \exp(Ax) - b \|_2^2$ and Resscaled version $\| \exp(Ax) - \langle \exp(Ax), {\bf 1}_n \rangle b \|_2^2 $   In contrast to previous approaches, we adopt a vectorization technique to address the regression problem in matrix formulation. This approach expands the dimension from $d$ to $d^2$, resembling the formulation of the regression problem mentioned earlier.   Upon completing the lipschitz analysis of our regression function, we have derived our main result concerning in-context learning.

{{</citation>}}


### (74/115) Scaling Laws Do Not Scale (Fernando Diaz et al., 2023)

{{<citation>}}

Fernando Diaz, Michael Madaio. (2023)  
**Scaling Laws Do Not Scale**  

---
Primary Category: cs.LG  
Categories: cond-mat-dis-nn, cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.03201v1)  

---


**ABSTRACT**  
Recent work has proposed a power law relationship, referred to as ``scaling laws,'' between the performance of artificial intelligence (AI) models and aspects of those models' design (e.g., dataset size). In other words, as the size of a dataset (or model parameters, etc) increases, the performance of a given model trained on that dataset will correspondingly increase. However, while compelling in the aggregate, this scaling law relationship overlooks the ways that metrics used to measure performance may be precarious and contested, or may not correspond with how different groups of people may perceive the quality of models' output. In this paper, we argue that as the size of datasets used to train large AI models grows, the number of distinct communities (including demographic groups) whose data is included in a given dataset is likely to grow, each of whom may have different values. As a result, there is an increased risk that communities represented in a dataset may have values or preferences not captured by (or in the worst case, at odds with) the metrics used to evaluate model performance for scaling laws. We end the paper with implications for AI scaling laws -- that models may not, in fact, continue to improve as the datasets get larger -- at least not for all people or communities impacted by those models.

{{</citation>}}


### (75/115) LLQL: Logistic Likelihood Q-Learning for Reinforcement Learning (Outongyi Lv et al., 2023)

{{<citation>}}

Outongyi Lv, Bingxin Zhou, Yu Guang Wang. (2023)  
**LLQL: Logistic Likelihood Q-Learning for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02345v1)  

---


**ABSTRACT**  
Currently, research on Reinforcement learning (RL) can be broadly classified into two categories: online RL and offline RL. Both in online and offline RL, the primary focus of research on the Bellman error lies in the optimization techniques and performance improvement, rather than exploring the inherent structural properties of the Bellman error, such as distribution characteristics. In this study, we analyze the distribution of the Bellman approximation error in both online and offline settings. We find that in the online environment, the Bellman error follows a Logistic distribution, while in the offline environment, the Bellman error follows a constrained Logistic distribution, where the constrained distribution is dependent on the prior policy in the offline data set. Based on this finding, we have improved the MSELoss which is based on the assumption that the Bellman errors follow a normal distribution, and we utilized the Logistic maximum likelihood function to construct $\rm LLoss$ as an alternative loss function. In addition, we observed that the rewards in the offline data set should follow a specific distribution, which would facilitate the achievement of offline objectives. In our numerical experiments, we performed controlled variable corrections on the loss functions of two variants of Soft-Actor-Critic in both online and offline environments. The results confirmed our hypothesis regarding the online and offline settings, we also found that the variance of LLoss is smaller than MSELoss. Our research provides valuable insights for further investigations based on the distribution of Bellman errors.

{{</citation>}}


### (76/115) Sumformer: Universal Approximation for Efficient Transformers (Silas Alberti et al., 2023)

{{<citation>}}

Silas Alberti, Niclas Dern, Laura Thesing, Gitta Kutyniok. (2023)  
**Sumformer: Universal Approximation for Efficient Transformers**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: AI, ChatGPT, GPT, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.02301v1)  

---


**ABSTRACT**  
Natural language processing (NLP) made an impressive jump with the introduction of Transformers. ChatGPT is one of the most famous examples, changing the perception of the possibilities of AI even outside the research community. However, besides the impressive performance, the quadratic time and space complexity of Transformers with respect to sequence length pose significant limitations for handling long sequences. While efficient Transformer architectures like Linformer and Performer with linear complexity have emerged as promising solutions, their theoretical understanding remains limited. In this paper, we introduce Sumformer, a novel and simple architecture capable of universally approximating equivariant sequence-to-sequence functions. We use Sumformer to give the first universal approximation results for Linformer and Performer. Moreover, we derive a new proof for Transformers, showing that just one attention layer is sufficient for universal approximation.

{{</citation>}}


### (77/115) Improving Address Matching using Siamese Transformer Networks (André V. Duarte et al., 2023)

{{<citation>}}

André V. Duarte, Arlindo L. Oliveira. (2023)  
**Improving Address Matching using Siamese Transformer Networks**  

---
Primary Category: cs.LG  
Categories: I-2, cs-IR, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.02300v1)  

---


**ABSTRACT**  
Matching addresses is a critical task for companies and post offices involved in the processing and delivery of packages. The ramifications of incorrectly delivering a package to the wrong recipient are numerous, ranging from harm to the company's reputation to economic and environmental costs. This research introduces a deep learning-based model designed to increase the efficiency of address matching for Portuguese addresses. The model comprises two parts: (i) a bi-encoder, which is fine-tuned to create meaningful embeddings of Portuguese postal addresses, utilized to retrieve the top 10 likely matches of the un-normalized target address from a normalized database, and (ii) a cross-encoder, which is fine-tuned to accurately rerank the 10 addresses obtained by the bi-encoder. The model has been tested on a real-case scenario of Portuguese addresses and exhibits a high degree of accuracy, exceeding 95% at the door level. When utilized with GPU computations, the inference speed is about 4.5 times quicker than other traditional approaches such as BM25. An implementation of this system in a real-world scenario would substantially increase the effectiveness of the distribution process. Such an implementation is currently under investigation.

{{</citation>}}


### (78/115) Dynamical Isometry based Rigorous Fair Neural Architecture Search (Jianxiang Luo et al., 2023)

{{<citation>}}

Jianxiang Luo, Junyi Hu, Tianji Pang, Weihao Huang, Chuang Liu. (2023)  
**Dynamical Isometry based Rigorous Fair Neural Architecture Search**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.02263v2)  

---


**ABSTRACT**  
Recently, the weight-sharing technique has significantly speeded up the training and evaluation procedure of neural architecture search. However, most existing weight-sharing strategies are solely based on experience or observation, which makes the searching results lack interpretability and rationality. In addition, due to the negligence of fairness, current methods are prone to make misjudgments in module evaluation. To address these problems, we propose a novel neural architecture search algorithm based on dynamical isometry. We use the fix point analysis method in the mean field theory to analyze the dynamics behavior in the steady state random neural network, and how dynamic isometry guarantees the fairness of weight-sharing based NAS. Meanwhile, we prove that our module selection strategy is rigorous fair by estimating the generalization error of all modules with well-conditioned Jacobian. Extensive experiments show that, with the same size, the architecture searched by the proposed method can achieve state-of-the-art top-1 validation accuracy on ImageNet classification. In addition, we demonstrate that our method is able to achieve better and more stable training performance without loss of generality.

{{</citation>}}


### (79/115) Multivariate Time Series Classification: A Deep Learning Approach (Mohamed Abouelnaga et al., 2023)

{{<citation>}}

Mohamed Abouelnaga, Julien Vitay, Aida Farahani. (2023)  
**Multivariate Time Series Classification: A Deep Learning Approach**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM, Time Series  
[Paper Link](http://arxiv.org/abs/2307.02253v1)  

---


**ABSTRACT**  
This paper investigates different methods and various neural network architectures applicable in the time series classification domain. The data is obtained from a fleet of gas sensors that measure and track quantities such as oxygen and sound. With the help of this data, we can detect events such as occupancy in a specific environment. At first, we analyze the time series data to understand the effect of different parameters, such as the sequence length, when training our models. These models employ Fully Convolutional Networks (FCN) and Long Short-Term Memory (LSTM) for supervised learning and Recurrent Autoencoders for semisupervised learning. Throughout this study, we spot the differences between these methods based on metrics such as precision and recall identifying which technique best suits this problem.

{{</citation>}}


### (80/115) ChiENN: Embracing Molecular Chirality with Graph Neural Networks (Piotr Gaiński et al., 2023)

{{<citation>}}

Piotr Gaiński, Michał Koziarski, Jacek Tabor, Marek Śmieja. (2023)  
**ChiENN: Embracing Molecular Chirality with Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-QM  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.02198v2)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) play a fundamental role in many deep learning problems, in particular in cheminformatics. However, typical GNNs cannot capture the concept of chirality, which means they do not distinguish between the 3D graph of a chemical compound and its mirror image (enantiomer). The ability to distinguish between enantiomers is important especially in drug discovery because enantiomers can have very distinct biochemical properties. In this paper, we propose a theoretically justified message-passing scheme, which makes GNNs sensitive to the order of node neighbors. We apply that general concept in the context of molecular chirality to construct Chiral Edge Neural Network (ChiENN) layer which can be appended to any GNN model to enable chirality-awareness. Our experiments show that adding ChiENN layers to a GNN outperforms current state-of-the-art methods in chiral-sensitive molecular property prediction tasks.

{{</citation>}}


### (81/115) Evaluating AI systems under uncertain ground truth: a case study in dermatology (David Stutz et al., 2023)

{{<citation>}}

David Stutz, Ali Taylan Cemgil, Abhijit Guha Roy, Tatiana Matejovicova, Melih Barsbey, Patricia Strachan, Mike Schaekermann, Jan Freyberg, Rajeev Rikhye, Beverly Freeman, Javier Perez Matos, Umesh Telang, Dale R. Webster, Yuan Liu, Greg S. Corrado, Yossi Matias, Pushmeet Kohli, Yun Liu, Arnaud Doucet, Alan Karthikesalingam. (2023)  
**Evaluating AI systems under uncertain ground truth: a case study in dermatology**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG, stat-ME, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02191v1)  

---


**ABSTRACT**  
For safety, AI systems in health undergo thorough evaluations before deployment, validating their predictions against a ground truth that is assumed certain. However, this is actually not the case and the ground truth may be uncertain. Unfortunately, this is largely ignored in standard evaluation of AI models but can have severe consequences such as overestimating the future performance. To avoid this, we measure the effects of ground truth uncertainty, which we assume decomposes into two main components: annotation uncertainty which stems from the lack of reliable annotations, and inherent uncertainty due to limited observational information. This ground truth uncertainty is ignored when estimating the ground truth by deterministically aggregating annotations, e.g., by majority voting or averaging. In contrast, we propose a framework where aggregation is done using a statistical model. Specifically, we frame aggregation of annotations as posterior inference of so-called plausibilities, representing distributions over classes in a classification setting, subject to a hyper-parameter encoding annotator reliability. Based on this model, we propose a metric for measuring annotation uncertainty and provide uncertainty-adjusted metrics for performance evaluation. We present a case study applying our framework to skin condition classification from images where annotations are provided in the form of differential diagnoses. The deterministic adjudication process called inverse rank normalization (IRN) from previous work ignores ground truth uncertainty in evaluation. Instead, we present two alternative statistical models: a probabilistic version of IRN and a Plackett-Luce-based model. We find that a large portion of the dataset exhibits significant ground truth uncertainty and standard IRN-based evaluation severely over-estimates performance without providing uncertainty estimates.

{{</citation>}}


### (82/115) Diffusion Models for Computational Design at the Example of Floor Plans (Joern Ploennigs et al., 2023)

{{<citation>}}

Joern Ploennigs, Markus Berger. (2023)  
**Diffusion Models for Computational Design at the Example of Floor Plans**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02511v1)  

---


**ABSTRACT**  
AI Image generators based on diffusion models are widely discussed recently for their capability to create images from simple text prompts. But, for practical use in civil engineering they need to be able to create specific construction plans for given constraints. Within this paper we explore the capabilities of those diffusion-based AI generators for computational design at the example of floor plans and identify their current limitation. We explain how the diffusion-models work and propose new diffusion models with improved semantic encoding. In several experiments we show that we can improve validity of generated floor plans from 6% to 90% and query performance for different examples. We identify short comings and derive future research challenges of those models and discuss the need to combine diffusion models with building information modelling. With this we provide key insights into the current state and future directions for diffusion models in civil engineering.

{{</citation>}}


### (83/115) Robust Graph Structure Learning with the Alignment of Features and Adjacency Matrix (Shaogao Lv et al., 2023)

{{<citation>}}

Shaogao Lv, Gang Wen, Shiyu Liu, Linsen Wei, Ming Li. (2023)  
**Robust Graph Structure Learning with the Alignment of Features and Adjacency Matrix**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2307.02126v1)  

---


**ABSTRACT**  
To improve the robustness of graph neural networks (GNN), graph structure learning (GSL) has attracted great interest due to the pervasiveness of noise in graph data. Many approaches have been proposed for GSL to jointly learn a clean graph structure and corresponding representations. To extend the previous work, this paper proposes a novel regularized GSL approach, particularly with an alignment of feature information and graph information, which is motivated mainly by our derived lower bound of node-level Rademacher complexity for GNNs. Additionally, our proposed approach incorporates sparse dimensional reduction to leverage low-dimensional node features that are relevant to the graph structure. To evaluate the effectiveness of our approach, we conduct experiments on real-world graphs. The results demonstrate that our proposed GSL method outperforms several competitive baselines, especially in scenarios where the graph structures are heavily affected by noise. Overall, our research highlights the importance of integrating feature and graph information alignment in GSL, as inspired by our derived theoretical result, and showcases the superiority of our approach in handling noisy graph structures through comprehensive experiments on real-world datasets.

{{</citation>}}


### (84/115) Make A Long Image Short: Adaptive Token Length for Vision Transformers (Qiqi Zhou et al., 2023)

{{<citation>}}

Qiqi Zhou, Yichen Zhu. (2023)  
**Make A Long Image Short: Adaptive Token Length for Vision Transformers**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.02092v1)  

---


**ABSTRACT**  
The vision transformer is a model that breaks down each image into a sequence of tokens with a fixed length and processes them similarly to words in natural language processing. Although increasing the number of tokens typically results in better performance, it also leads to a considerable increase in computational cost. Motivated by the saying "A picture is worth a thousand words," we propose an innovative approach to accelerate the ViT model by shortening long images. Specifically, we introduce a method for adaptively assigning token length for each image at test time to accelerate inference speed. First, we train a Resizable-ViT (ReViT) model capable of processing input with diverse token lengths. Next, we extract token-length labels from ReViT that indicate the minimum number of tokens required to achieve accurate predictions. We then use these labels to train a lightweight Token-Length Assigner (TLA) that allocates the optimal token length for each image during inference. The TLA enables ReViT to process images with the minimum sufficient number of tokens, reducing token numbers in the ViT model and improving inference speed. Our approach is general and compatible with modern vision transformer architectures, significantly reducing computational costs. We verified the effectiveness of our methods on multiple representative ViT models on image classification and action recognition.

{{</citation>}}


### (85/115) Performance Modeling of Data Storage Systems using Generative Models (Abdalaziz Rashid Al-Maeeni et al., 2023)

{{<citation>}}

Abdalaziz Rashid Al-Maeeni, Aziz Temirkhanov, Artem Ryzhikov, Mikhail Hushchyn. (2023)  
**Performance Modeling of Data Storage Systems using Generative Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-PF, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02073v1)  

---


**ABSTRACT**  
High-precision modeling of systems is one of the main areas of industrial data analysis. Models of systems, their digital twins, are used to predict their behavior under various conditions. We have developed several models of a storage system using machine learning-based generative models. The system consists of several components: hard disk drive (HDD) and solid-state drive (SSD) storage pools with different RAID schemes and cache. Each storage component is represented by a probabilistic model that describes the probability distribution of the component performance in terms of IOPS and latency, depending on their configuration and external data load parameters. The results of the experiments demonstrate the errors of 4-10 % for IOPS and 3-16 % for latency predictions depending on the components and models of the system. The predictions show up to 0.99 Pearson correlation with Little's law, which can be used for unsupervised reliability checks of the models. In addition, we present novel data sets that can be used for benchmarking regression algorithms, conditional generative models, and uncertainty estimation methods in machine learning.

{{</citation>}}


### (86/115) Facing off World Model Backbones: RNNs, Transformers, and S4 (Fei Deng et al., 2023)

{{<citation>}}

Fei Deng, Junyeong Park, Sungjin Ahn. (2023)  
**Facing off World Model Backbones: RNNs, Transformers, and S4**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.02064v1)  

---


**ABSTRACT**  
World models are a fundamental component in model-based reinforcement learning (MBRL) agents. To perform temporally extended and consistent simulations of the future in partially observable environments, world models need to possess long-term memory. However, state-of-the-art MBRL agents, such as Dreamer, predominantly employ recurrent neural networks (RNNs) as their world model backbone, which have limited memory capacity. In this paper, we seek to explore alternative world model backbones for improving long-term memory. In particular, we investigate the effectiveness of Transformers and Structured State Space Sequence (S4) models, motivated by their remarkable ability to capture long-range dependencies in low-dimensional sequences and their complementary strengths. We propose S4WM, the first S4-based world model that can generate high-dimensional image sequences through latent imagination. Furthermore, we extensively compare RNN-, Transformer-, and S4-based world models across four sets of environments, which we have specifically tailored to assess crucial memory capabilities of world models, including long-term imagination, context-dependent recall, reward prediction, and memory-based reasoning. Our findings demonstrate that S4WM outperforms Transformer-based world models in terms of long-term memory, while exhibiting greater efficiency during training and imagination. These results pave the way for the development of stronger MBRL agents.

{{</citation>}}


### (87/115) Improving Automatic Parallel Training via Balanced Memory Workload Optimization (Yujie Wang et al., 2023)

{{<citation>}}

Yujie Wang, Youhe Jiang, Xupeng Miao, Fangcheng Fu, Xiaonan Nie, Bin Cui. (2023)  
**Improving Automatic Parallel Training via Balanced Memory Workload Optimization**  

---
Primary Category: cs.LG  
Categories: cs-DB, cs-DC, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.02031v1)  

---


**ABSTRACT**  
Transformer models have emerged as the leading approach for achieving state-of-the-art performance across various application domains, serving as the foundation for advanced large-scale deep learning (DL) models. However, efficiently training these models across multiple GPUs remains a complex challenge due to the abundance of parallelism options. Existing DL systems either require manual efforts to design distributed training plans or limit parallelism combinations to a constrained search space. In this paper, we present Galvatron-BMW, a novel system framework that integrates multiple prevalent parallelism dimensions and automatically identifies the most efficient hybrid parallelism strategy. To effectively navigate this vast search space, we employ a decision tree approach for decomposition and pruning based on intuitive insights. We further utilize a dynamic programming search algorithm to derive the optimal plan. Moreover, to improve resource utilization and enhance system efficiency, we propose a bi-objective optimization workflow that focuses on workload balance. Our evaluations on different Transformer models demonstrate the capabilities of Galvatron-BMW in automating distributed training under varying GPU memory constraints. Across all tested scenarios, Galvatron-BMW consistently achieves superior system throughput, surpassing previous approaches that rely on limited parallelism strategies.

{{</citation>}}


### (88/115) EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models (Michael Wornow et al., 2023)

{{<citation>}}

Michael Wornow, Rahul Thapa, Ethan Steinberg, Jason Fries, Nigam Shah. (2023)  
**EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: BERT, Clinical, Few-Shot  
[Paper Link](http://arxiv.org/abs/2307.02028v1)  

---


**ABSTRACT**  
While the general machine learning (ML) community has benefited from public datasets, tasks, and models, the progress of ML in healthcare has been hampered by a lack of such shared assets. The success of foundation models creates new challenges for healthcare ML by requiring access to shared pretrained models to validate performance benefits. We help address these challenges through three contributions. First, we publish a new dataset, EHRSHOT, containing de-identified structured data from the electronic health records (EHRs) of 6,712 patients from Stanford Medicine. Unlike MIMIC-III/IV and other popular EHR datasets, EHRSHOT is longitudinal and not restricted to ICU/ED patients. Second, we publish the weights of a 141M parameter clinical foundation model pretrained on the structured EHR data of 2.57M patients. We are one of the first to fully release such a model for coded EHR data; in contrast, most prior models released for clinical data (e.g. GatorTron, ClinicalBERT) only work with unstructured text and cannot process the rich, structured data within an EHR. We provide an end-to-end pipeline for the community to validate and build upon its performance. Third, we define 15 few-shot clinical prediction tasks, enabling evaluation of foundation models on benefits such as sample efficiency and task adaption. The code to reproduce our results, as well as the model and dataset (via a research data use agreement), are available at our Github repo here: https://github.com/som-shahlab/ehrshot-benchmark

{{</citation>}}


### (89/115) STS-CCL: Spatial-Temporal Synchronous Contextual Contrastive Learning for Urban Traffic Forecasting (Lincan Li et al., 2023)

{{<citation>}}

Lincan Li, Kaixiang Yang, Fengji Luo, Jichao Bi. (2023)  
**STS-CCL: Spatial-Temporal Synchronous Contextual Contrastive Learning for Urban Traffic Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.02507v1)  

---


**ABSTRACT**  
Efficiently capturing the complex spatiotemporal representations from large-scale unlabeled traffic data remains to be a challenging task. In considering of the dilemma, this work employs the advanced contrastive learning and proposes a novel Spatial-Temporal Synchronous Contextual Contrastive Learning (STS-CCL) model. First, we elaborate the basic and strong augmentation methods for spatiotemporal graph data, which not only perturb the data in terms of graph structure and temporal characteristics, but also employ a learning-based dynamic graph view generator for adaptive augmentation. Second, we introduce a Spatial-Temporal Synchronous Contrastive Module (STS-CM) to simultaneously capture the decent spatial-temporal dependencies and realize graph-level contrasting. To further discriminate node individuals in negative filtering, a Semantic Contextual Contrastive method is designed based on semantic features and spatial heterogeneity, achieving node-level contrastive learning along with negative filtering. Finally, we present a hard mutual-view contrastive training scheme and extend the classic contrastive loss to an integrated objective function, yielding better performance. Extensive experiments and evaluations demonstrate that building a predictor upon STS-CCL contrastive learning model gains superior performance than existing traffic forecasting benchmarks. The proposed STS-CCL is highly suitable for large datasets with only a few labeled data and other spatiotemporal tasks with data scarcity issue.

{{</citation>}}


### (90/115) Zero-Shot Neural Architecture Search: Challenges, Solutions, and Opportunities (Guihong Li et al., 2023)

{{<citation>}}

Guihong Li, Duc Hoang, Kartikeya Bhardwaj, Ming Lin, Zhangyang Wang, Radu Marculescu. (2023)  
**Zero-Shot Neural Architecture Search: Challenges, Solutions, and Opportunities**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.01998v1)  

---


**ABSTRACT**  
Recently, zero-shot (or training-free) Neural Architecture Search (NAS) approaches have been proposed to liberate the NAS from training requirements. The key idea behind zero-shot NAS approaches is to design proxies that predict the accuracies of the given networks without training network parameters. The proxies proposed so far are usually inspired by recent progress in theoretical deep learning and have shown great potential on several NAS benchmark datasets. This paper aims to comprehensively review and compare the state-of-the-art (SOTA) zero-shot NAS approaches, with an emphasis on their hardware awareness. To this end, we first review the mainstream zero-shot proxies and discuss their theoretical underpinnings. We then compare these zero-shot proxies through large-scale experiments and demonstrate their effectiveness in both hardware-aware and hardware-oblivious NAS scenarios. Finally, we point out several promising ideas to design better proxies. Our source code and the related paper list are available on https://github.com/SLDGroup/survey-zero-shot-nas.

{{</citation>}}


### (91/115) Dynamic Feature-based Deep Reinforcement Learning for Flow Control of Circular Cylinder with Sparse Surface Pressure Sensing (Qiulei Wang et al., 2023)

{{<citation>}}

Qiulei Wang, Lei Yan, Gang Hu, Wenli Chen, Jean Rabault, Bernd R. Noack. (2023)  
**Dynamic Feature-based Deep Reinforcement Learning for Flow Control of Circular Cylinder with Sparse Surface Pressure Sensing**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-flu-dyn  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.01995v1)  

---


**ABSTRACT**  
This study proposes a self-learning algorithm for closed-loop cylinder wake control targeting lower drag and lower lift fluctuations with the additional challenge of sparse sensor information, taking deep reinforcement learning as the starting point. DRL performance is significantly improved by lifting the sensor signals to dynamic features (DF), which predict future flow states. The resulting dynamic feature-based DRL (DF-DRL) automatically learns a feedback control in the plant without a dynamic model. Results show that the drag coefficient of the DF-DRL model is 25% less than the vanilla model based on direct sensor feedback. More importantly, using only one surface pressure sensor, DF-DRL can reduce the drag coefficient to a state-of-the-art performance of about 8% at Re = 100 and significantly mitigate lift coefficient fluctuations. Hence, DF-DRL allows the deployment of sparse sensing of the flow without degrading the control performance. This method also shows good robustness in controlling flow under higher Reynolds numbers, which reduces the drag coefficient by 32.2% and 46.55% at Re = 500 and 1000, respectively, indicating the broad applicability of the method. Since surface pressure information is more straightforward to measure in realistic scenarios than flow velocity information, this study provides a valuable reference for experimentally designing the active flow control of a circular cylinder based on wall pressure signals, which is an essential step toward further developing intelligent control in realistic multi-input multi-output (MIMO) system.

{{</citation>}}


## cs.CR (3)



### (92/115) Securing Cloud FPGAs Against Power Side-Channel Attacks: A Case Study on Iterative AES (Nithyashankari Gummidipoondi Jayasankaran et al., 2023)

{{<citation>}}

Nithyashankari Gummidipoondi Jayasankaran, Hao Guo, Satwik Patnaik, Jeyavijayan, Rajendran, Jiang Hu. (2023)  
**Securing Cloud FPGAs Against Power Side-Channel Attacks: A Case Study on Iterative AES**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AWS, Amazon  
[Paper Link](http://arxiv.org/abs/2307.02569v1)  

---


**ABSTRACT**  
The various benefits of multi-tenanting, such as higher device utilization and increased profit margin, intrigue the cloud field-programmable gate array (FPGA) servers to include multi-tenanting in their infrastructure. However, this property makes these servers vulnerable to power side-channel (PSC) attacks. Logic designs such as ring oscillator (RO) and time-to-digital converter (TDC) are used to measure the power consumed by security critical circuits, such as advanced encryption standard (AES). Firstly, the existing works require higher minimum traces for disclosure (MTD). Hence, in this work, we improve the sensitivity of the TDC-based sensors by manually placing the FPGA primitives inferring these sensors. This enhancement helps to determine the 128-bit AES key using 3.8K traces. Secondly, the existing defenses use ROs to defend against PSC attacks. However, cloud servers such as Amazon Web Services (AWS) block design with combinatorial loops. Hence, we propose a placement-based defense. We study the impact of (i) primitive-level placement on the AES design and (ii) additional logic that resides along with the AES on the correlation power analysis (CPA) attack results. Our results showcase that the AES along with filters and/or processors are sufficient to provide the same level or better security than the existing defenses.

{{</citation>}}


### (93/115) Security Risk Analysis Methodologies for Automotive Systems (Mohamed Abouelnaga et al., 2023)

{{<citation>}}

Mohamed Abouelnaga, Christine Jakobs. (2023)  
**Security Risk Analysis Methodologies for Automotive Systems**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.02261v1)  

---


**ABSTRACT**  
Nowadays, systematic security risk analysis plays a vital role in the automotive domain. The demand for advanced driver assistance systems and connectivity of vehicles to the internet makes cyber-security a crucial requirement for vehicle manufacturers. This paper summarizes the risk analysis method stated in the recently released automotive security standard ISO/SAE 21434, which lays the high-level principles for threat analysis and risk assessment (TARA) methods. Following, we introduce a specific use case to compare different security analysis approaches which OEMs can benefit from to achieve compliance with the standard.

{{</citation>}}


### (94/115) African Union Convention on Cyber Security and Personal Data Protection: Challenges and Future Directions (MA. Bouke et al., 2023)

{{<citation>}}

MA. Bouke, A. Abdullah, SH. ALshatebi, H. El. Atigh, K. Cengiz. (2023)  
**African Union Convention on Cyber Security and Personal Data Protection: Challenges and Future Directions**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CY, cs.CR  
Keywords: Cyber Security, Security  
[Paper Link](http://arxiv.org/abs/2307.01966v1)  

---


**ABSTRACT**  
This paper investigates the challenges and opportunities of implementing the African Union Convention on Cyber Security and Personal Data Protection (AUDPC) across Africa. Focusing on legal, regulatory, technical, infrastructural, capacity building, awareness, Harmonization, and cross-border cooperation challenges, the paper identifies key findings that highlight the diverse legal systems and traditions, the lack of comprehensive data protection laws, the need to balance national security and data privacy, the digital divide, cybersecurity threats, implications of emerging technologies on data privacy, limited resources for data protection authorities, and the need for capacity building in data privacy and protection. The paper also emphasizes the importance of Harmonization and cross-border cooperation in aligning data protection frameworks and collaborating with international partners and global organizations. To address these challenges and facilitate the successful implementation of the AUDPC, the paper proposes a set of recommendations, including strengthening legal and regulatory frameworks, enhancing technical and infrastructural capacities, fostering capacity-building and awareness initiatives, promoting Harmonization and cross-border cooperation, and engaging with global data protection trends and developments.

{{</citation>}}


## eess.IV (6)



### (95/115) AxonCallosumEM Dataset: Axon Semantic Segmentation of Whole Corpus Callosum cross section from EM Images (Ao Cheng et al., 2023)

{{<citation>}}

Ao Cheng, Guoqiang Zhao, Lirong Wang, Ruobing Zhang. (2023)  
**AxonCallosumEM Dataset: Axon Semantic Segmentation of Whole Corpus Callosum cross section from EM Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.02464v1)  

---


**ABSTRACT**  
The electron microscope (EM) remains the predominant technique for elucidating intricate details of the animal nervous system at the nanometer scale. However, accurately reconstructing the complex morphology of axons and myelin sheaths poses a significant challenge. Furthermore, the absence of publicly available, large-scale EM datasets encompassing complete cross sections of the corpus callosum, with dense ground truth segmentation for axons and myelin sheaths, hinders the advancement and evaluation of holistic corpus callosum reconstructions. To surmount these obstacles, we introduce the AxonCallosumEM dataset, comprising a 1.83 times 5.76mm EM image captured from the corpus callosum of the Rett Syndrome (RTT) mouse model, which entail extensive axon bundles. We meticulously proofread over 600,000 patches at a resolution of 1024 times 1024, thus providing a comprehensive ground truth for myelinated axons and myelin sheaths. Additionally, we extensively annotated three distinct regions within the dataset for the purposes of training, testing, and validation. Utilizing this dataset, we develop a fine-tuning methodology that adapts Segment Anything Model (SAM) to EM images segmentation tasks, called EM-SAM, enabling outperforms other state-of-the-art methods. Furthermore, we present the evaluation results of EM-SAM as a baseline.

{{</citation>}}


### (96/115) LLCaps: Learning to Illuminate Low-Light Capsule Endoscopy with Curved Wavelet Attention and Reverse Diffusion (Long Bai et al., 2023)

{{<citation>}}

Long Bai, Tong Chen, Yanan Wu, An Wang, Mobarakol Islam, Hongliang Ren. (2023)  
**LLCaps: Learning to Illuminate Low-Light Capsule Endoscopy with Curved Wavelet Attention and Reverse Diffusion**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-RO, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.02452v1)  

---


**ABSTRACT**  
Wireless capsule endoscopy (WCE) is a painless and non-invasive diagnostic tool for gastrointestinal (GI) diseases. However, due to GI anatomical constraints and hardware manufacturing limitations, WCE vision signals may suffer from insufficient illumination, leading to a complicated screening and examination procedure. Deep learning-based low-light image enhancement (LLIE) in the medical field gradually attracts researchers. Given the exuberant development of the denoising diffusion probabilistic model (DDPM) in computer vision, we introduce a WCE LLIE framework based on the multi-scale convolutional neural network (CNN) and reverse diffusion process. The multi-scale design allows models to preserve high-resolution representation and context information from low-resolution, while the curved wavelet attention (CWA) block is proposed for high-frequency and local feature learning. Furthermore, we combine the reverse diffusion procedure to further optimize the shallow output and generate the most realistic image. The proposed method is compared with ten state-of-the-art (SOTA) LLIE methods and significantly outperforms quantitatively and qualitatively. The superior performance on GI disease segmentation further demonstrates the clinical potential of our proposed model. Our code is publicly accessible.

{{</citation>}}


### (97/115) Compound Attention and Neighbor Matching Network for Multi-contrast MRI Super-resolution (Wenxuan Chen et al., 2023)

{{<citation>}}

Wenxuan Chen, Sirui Wu, Shuai Wang, Zhongsen Li, Jia Yang, Xiaolei Song. (2023)  
**Compound Attention and Neighbor Matching Network for Multi-contrast MRI Super-resolution**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.02148v1)  

---


**ABSTRACT**  
Multi-contrast magnetic resonance imaging (MRI) reflects information about human tissue from different perspectives and has many clinical applications. By utilizing the complementary information among different modalities, multi-contrast super-resolution (SR) of MRI can achieve better results than single-image super-resolution. However, existing methods of multi-contrast MRI SR have the following shortcomings that may limit their performance: First, existing methods either simply concatenate the reference and degraded features or exploit global feature-matching between them, which are unsuitable for multi-contrast MRI SR. Second, although many recent methods employ transformers to capture long-range dependencies in the spatial dimension, they neglect that self-attention in the channel dimension is also important for low-level vision tasks. To address these shortcomings, we proposed a novel network architecture with compound-attention and neighbor matching (CANM-Net) for multi-contrast MRI SR: The compound self-attention mechanism effectively captures the dependencies in both spatial and channel dimension; the neighborhood-based feature-matching modules are exploited to match degraded features and adjacent reference features and then fuse them to obtain the high-quality images. We conduct experiments of SR tasks on the IXI, fastMRI, and real-world scanning datasets. The CANM-Net outperforms state-of-the-art approaches in both retrospective and prospective experiments. Moreover, the robustness study in our work shows that the CANM-Net still achieves good performance when the reference and degraded images are imperfectly registered, proving good potential in clinical applications.

{{</citation>}}


### (98/115) Multi-Scale U-Shape MLP for Hyperspectral Image Classification (Moule Lin et al., 2023)

{{<citation>}}

Moule Lin, Weipeng Jing, Donglin Di, Guangsheng Chen, Houbing Song. (2023)  
**Multi-Scale U-Shape MLP for Hyperspectral Image Classification**  

---
Primary Category: eess.IV  
Categories: cs-LG, eess-IV, eess.IV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2307.10186v1)  

---


**ABSTRACT**  
Hyperspectral images have significant applications in various domains, since they register numerous semantic and spatial information in the spectral band with spatial variability of spectral signatures. Two critical challenges in identifying pixels of the hyperspectral image are respectively representing the correlated information among the local and global, as well as the abundant parameters of the model. To tackle this challenge, we propose a Multi-Scale U-shape Multi-Layer Perceptron (MUMLP) a model consisting of the designed MSC (Multi-Scale Channel) block and the UMLP (U-shape Multi-Layer Perceptron) structure. MSC transforms the channel dimension and mixes spectral band feature to embed the deep-level representation adequately. UMLP is designed by the encoder-decoder structure with multi-layer perceptron layers, which is capable of compressing the large-scale parameters. Extensive experiments are conducted to demonstrate our model can outperform state-of-the-art methods across-the-board on three wide-adopted public datasets, namely Pavia University, Houston 2013 and Houston 2018

{{</citation>}}


### (99/115) Unsupervised Spectral Demosaicing with Lightweight Spectral Attention Networks (Kai Feng et al., 2023)

{{<citation>}}

Kai Feng, Yongqiang Zhao, Seong G. Kong, Haijin Zeng. (2023)  
**Unsupervised Spectral Demosaicing with Lightweight Spectral Attention Networks**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.01990v1)  

---


**ABSTRACT**  
This paper presents a deep learning-based spectral demosaicing technique trained in an unsupervised manner. Many existing deep learning-based techniques relying on supervised learning with synthetic images, often underperform on real-world images especially when the number of spectral bands increases. According to the characteristics of the spectral mosaic image, this paper proposes a mosaic loss function, the corresponding model structure, a transformation strategy, and an early stopping strategy, which form a complete unsupervised spectral demosaicing framework. A challenge in real-world spectral demosaicing is inconsistency between the model parameters and the computational resources of the imager. We reduce the complexity and parameters of the spectral attention module by dividing the spectral attention tensor into spectral attention matrices in the spatial dimension and spectral attention vector in the channel dimension, which is more suitable for unsupervised framework. This paper also presents Mosaic25, a real 25-band hyperspectral mosaic image dataset of various objects, illuminations, and materials for benchmarking. Extensive experiments on synthetic and real-world datasets demonstrate that the proposed method outperforms conventional unsupervised methods in terms of spatial distortion suppression, spectral fidelity, robustness, and computational cost.

{{</citation>}}


### (100/115) A ChatGPT Aided Explainable Framework for Zero-Shot Medical Image Diagnosis (Jiaxiang Liu et al., 2023)

{{<citation>}}

Jiaxiang Liu, Tianxiang Hu, Yan Zhang, Xiaotang Gai, Yang Feng, Zuozhu Liu. (2023)  
**A ChatGPT Aided Explainable Framework for Zero-Shot Medical Image Diagnosis**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: ChatGPT, GPT, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.01981v1)  

---


**ABSTRACT**  
Zero-shot medical image classification is a critical process in real-world scenarios where we have limited access to all possible diseases or large-scale annotated data. It involves computing similarity scores between a query medical image and possible disease categories to determine the diagnostic result. Recent advances in pretrained vision-language models (VLMs) such as CLIP have shown great performance for zero-shot natural image recognition and exhibit benefits in medical applications. However, an explainable zero-shot medical image recognition framework with promising performance is yet under development. In this paper, we propose a novel CLIP-based zero-shot medical image classification framework supplemented with ChatGPT for explainable diagnosis, mimicking the diagnostic process performed by human experts. The key idea is to query large language models (LLMs) with category names to automatically generate additional cues and knowledge, such as disease symptoms or descriptions other than a single category name, to help provide more accurate and explainable diagnosis in CLIP. We further design specific prompts to enhance the quality of generated texts by ChatGPT that describe visual medical features. Extensive results on one private dataset and four public datasets along with detailed analysis demonstrate the effectiveness and explainability of our training-free zero-shot diagnosis pipeline, corroborating the great potential of VLMs and LLMs for medical applications.

{{</citation>}}


## cs.SE (4)



### (101/115) An Exploratory Literature Study on Sharing and Energy Use of Language Models for Source Code (Max Hort et al., 2023)

{{<citation>}}

Max Hort, Anastasiia Grishina, Leon Moonen. (2023)  
**An Exploratory Literature Study on Sharing and Energy Use of Language Models for Source Code**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-LG, cs-NE, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.02443v1)  

---


**ABSTRACT**  
Large language models trained on source code can support a variety of software development tasks, such as code recommendation and program repair. Large amounts of data for training such models benefit the models' performance. However, the size of the data and models results in long training times and high energy consumption. While publishing source code allows for replicability, users need to repeat the expensive training process if models are not shared. The main goal of the study is to investigate if publications that trained language models for software engineering (SE) tasks share source code and trained artifacts. The second goal is to analyze the transparency on training energy usage. We perform a snowballing-based literature search to find publications on language models for source code, and analyze their reusability from a sustainability standpoint.   From 494 unique publications, we identified 293 relevant publications that use language models to address code-related tasks. Among them, 27% (79 out of 293) make artifacts available for reuse. This can be in the form of tools or IDE plugins designed for specific tasks or task-agnostic models that can be fine-tuned for a variety of downstream tasks. Moreover, we collect insights on the hardware used for model training, as well as training time, which together determine the energy consumption of the development process. We find that there are deficiencies in the sharing of information and artifacts for current studies on source code models for software engineering tasks, with 40% of the surveyed papers not sharing source code or trained artifacts. We recommend the sharing of source code as well as trained artifacts, to enable sustainable reproducibility. Moreover, comprehensive information on training times and hardware configurations should be shared for transparency on a model's carbon footprint.

{{</citation>}}


### (102/115) Security Defect Detection via Code Review: A Study of the OpenStack and Qt Communities (Jiaxin Yu et al., 2023)

{{<citation>}}

Jiaxin Yu, Liming Fu, Peng Liang, Amjed Tahir, Mojtaba Shahin. (2023)  
**Security Defect Detection via Code Review: A Study of the OpenStack and Qt Communities**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.02326v1)  

---


**ABSTRACT**  
Background: Despite the widespread use of automated security defect detection tools, software projects still contain many security defects that could result in serious damage. Such tools are largely context-insensitive and may not cover all possible scenarios in testing potential issues, which makes them susceptible to missing complex security defects. Hence, thorough detection entails a synergistic cooperation between these tools and human-intensive detection techniques, including code review. Code review is widely recognized as a crucial and effective practice for identifying security defects. Aim: This work aims to empirically investigate security defect detection through code review. Method: To this end, we conducted an empirical study by analyzing code review comments derived from four projects in the OpenStack and Qt communities. Through manually checking 20,995 review comments obtained by keyword-based search, we identified 614 comments as security-related. Results: Our results show that (1) security defects are not prevalently discussed in code review, (2) more than half of the reviewers provided explicit fixing strategies/solutions to help developers fix security defects, (3) developers tend to follow reviewers' suggestions and action the changes, (4) Not worth fixing the defect now and Disagreement between the developer and the reviewer are the main causes for not resolving security defects. Conclusions: Our research results demonstrate that (1) software security practices should combine manual code review with automated detection tools, achieving a more comprehensive coverage to identifying and addressing security defects, and (2) promoting appropriate standardization of practitioners' behaviors during code review remains necessary for enhancing software security.

{{</citation>}}


### (103/115) Towards Open Federated Learning Platforms: Survey and Vision from Technical and Legal Perspectives (Moming Duan, 2023)

{{<citation>}}

Moming Duan. (2023)  
**Towards Open Federated Learning Platforms: Survey and Vision from Technical and Legal Perspectives**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-LG, cs-SE, cs.SE  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2307.02140v1)  

---


**ABSTRACT**  
Traditional Federated Learning (FL) follows a server-domincated cooperation paradigm which narrows the application scenarios of FL and decreases the enthusiasm of data holders to participate. To fully unleash the potential of FL, we advocate rethinking the design of current FL frameworks and extending it to a more generalized concept: Open Federated Learning Platforms. We propose two reciprocal cooperation frameworks for FL to achieve this: query-based FL and contract-based FL. In this survey, we conduct a comprehensive review of the feasibility of constructing an open FL platform from both technical and legal perspectives. We begin by reviewing the definition of FL and summarizing its inherent limitations, including server-client coupling, low model reusability, and non-public. In the query-based FL platform, which is an open model sharing and reusing platform empowered by the community for model mining, we explore a wide range of valuable topics, including the availability of up-to-date model repositories for model querying, legal compliance analysis between different model licenses, and copyright issues and intellectual property protection in model reusing. In particular, we introduce a novel taxonomy to streamline the analysis of model license compatibility in FL studies that involve batch model reusing methods, including combination, amalgamation, distillation, and generation. This taxonomy provides a systematic framework for identifying the corresponding clauses of licenses and facilitates the identification of potential legal implications and restrictions when reusing models. Through this survey, we uncover the the current dilemmas faced by FL and advocate for the development of sustainable open FL platforms. We aim to provide guidance for establishing such platforms in the future, while identifying potential problems and challenges that need to be addressed.

{{</citation>}}


### (104/115) Trust in Software Supply Chains: Blockchain-Enabled SBOM and the AIBOM Future (Boming Xia et al., 2023)

{{<citation>}}

Boming Xia, Dawen Zhang, Yue Liu, Qinghua Lu, Zhenchang Xing, Liming Zhu. (2023)  
**Trust in Software Supply Chains: Blockchain-Enabled SBOM and the AIBOM Future**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02088v2)  

---


**ABSTRACT**  
Software Bill of Materials (SBOM) serves as a critical pillar in ensuring software supply chain security by providing a detailed inventory of the components and dependencies integral to software development. However, challenges abound in the sharing of SBOMs, including potential data tampering, hesitation among software vendors to disclose comprehensive information, and bespoke requirements from software procurers or users. These obstacles have stifled widespread adoption and utilization of SBOMs, underscoring the need for a more secure and flexible mechanism for SBOM sharing. This study proposes a novel solution to these challenges by introducing a blockchain-empowered approach for SBOM sharing, leveraging verifiable credentials to allow for selective disclosure. This strategy not only heightens security but also offers flexibility. Furthermore, this paper broadens the remit of SBOM to encompass AI systems, thereby coining the term AI Bill of Materials (AIBOM). This extension is motivated by the rapid progression in AI technology and the escalating necessity to track the lineage and composition of AI software and systems. Particularly in the era of foundational models like large language models (LLMs), understanding their composition and dependencies becomes crucial. These models often serve as a base for further development, creating complex dependencies and paving the way for innovative AI applications. The evaluation of our solution indicates the feasibility and flexibility of the proposed SBOM sharing mechanism, positing a new solution for securing (AI) software supply chains.

{{</citation>}}


## cs.CE (1)



### (105/115) Error Approximation and Bias Correction in Dynamic Problems using a Recurrent Neural Network/Finite Element Hybrid Model (Moritz von Tresckow et al., 2023)

{{<citation>}}

Moritz von Tresckow, Herbert De Gersem, Dimitrios Loukrezis. (2023)  
**Error Approximation and Bias Correction in Dynamic Problems using a Recurrent Neural Network/Finite Element Hybrid Model**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.02349v2)  

---


**ABSTRACT**  
This work proposes a hybrid modeling framework based on recurrent neural networks (RNNs) and the finite element (FE) method to approximate model discrepancies in time dependent, multi-fidelity problems, and use the trained hybrid models to perform bias correction of the low-fidelity models. The hybrid model uses FE basis functions as a spatial basis and RNNs for the approximation of the time dependencies of the FE basis' degrees of freedom. The training data sets consist of sparse, non-uniformly sampled snapshots of the discrepancy function, pre-computed from trajectory data of low- and high-fidelity dynamic FE models. To account for data sparsity and prevent overfitting, data upsampling and local weighting factors are employed, to instigate a trade-off between physically conforming model behavior and neural network regression. The proposed hybrid modeling methodology is showcased in three highly non-trivial engineering test-cases, all featuring transient FE models, namely, heat diffusion out of a heat sink, eddy-currents in a quadrupole magnet, and sound wave propagation in a cavity. The results show that the proposed hybrid model is capable of approximating model discrepancies to a high degree of accuracy and accordingly correct low-fidelity models.

{{</citation>}}


## cs.NI (1)



### (106/115) Data-driven Predictive Latency for 5G: A Theoretical and Experimental Analysis Using Network Measurements (Marco Skocaj et al., 2023)

{{<citation>}}

Marco Skocaj, Francesca Conserva, Nicol Sarcone Grande, Andrea Orsi, Davide Micheli, Giorgio Ghinamo, Simone Bizzarri, Roberto Verdone. (2023)  
**Data-driven Predictive Latency for 5G: A Theoretical and Experimental Analysis Using Network Measurements**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: Network Measurement  
[Paper Link](http://arxiv.org/abs/2307.02329v2)  

---


**ABSTRACT**  
The advent of novel 5G services and applications with binding latency requirements and guaranteed Quality of Service (QoS) hastened the need to incorporate autonomous and proactive decision-making in network management procedures. The objective of our study is to provide a thorough analysis of predictive latency within 5G networks by utilizing real-world network data that is accessible to mobile network operators (MNOs). In particular, (i) we present an analytical formulation of the user-plane latency as a Hypoexponential distribution, which is validated by means of a comparative analysis with empirical measurements, and (ii) we conduct experimental results of probabilistic regression, anomaly detection, and predictive forecasting leveraging on emerging domains in Machine Learning (ML), such as Bayesian Learning (BL) and Machine Learning on Graphs (GML). We test our predictive framework using data gathered from scenarios of vehicular mobility, dense-urban traffic, and social gathering events. Our results provide valuable insights into the efficacy of predictive algorithms in practical applications.

{{</citation>}}


## eess.AS (1)



### (107/115) Exploring Multimodal Approaches for Alzheimer's Disease Detection Using Patient Speech Transcript and Audio Data (Hongmin Cai et al., 2023)

{{<citation>}}

Hongmin Cai, Xiaoke Huang, Zhengliang Liu, Wenxiong Liao, Haixing Dai, Zihao Wu, Dajiang Zhu, Hui Ren, Quanzheng Li, Tianming Liu, Xiang Li. (2023)  
**Exploring Multimodal Approaches for Alzheimer's Disease Detection Using Patient Speech Transcript and Audio Data**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-SD, eess-AS, eess.AS  
Keywords: GNN, GPT, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2307.02514v1)  

---


**ABSTRACT**  
Alzheimer's disease (AD) is a common form of dementia that severely impacts patient health. As AD impairs the patient's language understanding and expression ability, the speech of AD patients can serve as an indicator of this disease. This study investigates various methods for detecting AD using patients' speech and transcripts data from the DementiaBank Pitt database. The proposed approach involves pre-trained language models and Graph Neural Network (GNN) that constructs a graph from the speech transcript, and extracts features using GNN for AD detection. Data augmentation techniques, including synonym replacement, GPT-based augmenter, and so on, were used to address the small dataset size. Audio data was also introduced, and WavLM model was used to extract audio features. These features were then fused with text features using various methods. Finally, a contrastive learning approach was attempted by converting speech transcripts back to audio and using it for contrastive learning with the original audio. We conducted intensive experiments and analysis on the above methods. Our findings shed light on the challenges and potential solutions in AD detection using speech and audio data.

{{</citation>}}


## cs.DB (2)



### (108/115) Abstractions, Scenarios, and Prompt Definitions for Process Mining with LLMs: A Case Study (Alessandro Berti et al., 2023)

{{<citation>}}

Alessandro Berti, Daniel Schuster, Wil M. P. van der Aalst. (2023)  
**Abstractions, Scenarios, and Prompt Definitions for Process Mining with LLMs: A Case Study**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.02194v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) are capable of answering questions in natural language for various purposes. With recent advancements (such as GPT-4), LLMs perform at a level comparable to humans for many proficient tasks. The analysis of business processes could benefit from a natural process querying language and using the domain knowledge on which LLMs have been trained. However, it is impossible to provide a complete database or event log as an input prompt due to size constraints. In this paper, we apply LLMs in the context of process mining by i) abstracting the information of standard process mining artifacts and ii) describing the prompting strategies. We implement the proposed abstraction techniques into pm4py, an open-source process mining library. We present a case study using available event logs. Starting from different abstractions and analysis questions, we formulate prompts and evaluate the quality of the answers.

{{</citation>}}


### (109/115) The FormAI Dataset: Generative AI in Software Security Through the Lens of Formal Verification (Norbert Tihanyi et al., 2023)

{{<citation>}}

Norbert Tihanyi, Tamas Bisztray, Ridhi Jain, Mohamed Amine Ferrag, Lucas C. Cordeiro, Vasileios Mavroeidis. (2023)  
**The FormAI Dataset: Generative AI in Software Security Through the Lens of Formal Verification**  

---
Primary Category: cs.DB  
Categories: cs-AI, cs-DB, cs.DB  
Keywords: AI, GPT, GPT-3.5, Generative AI, Language Model, Security  
[Paper Link](http://arxiv.org/abs/2307.02192v1)  

---


**ABSTRACT**  
This paper presents the FormAI dataset, a large collection of 112,000 AI-generated compilable and independent C programs with vulnerability classification. We introduce a dynamic zero-shot prompting technique, constructed to spawn a diverse set of programs utilizing Large Language Models (LLMs). The dataset is generated by GPT-3.5-turbo and comprises programs with varying levels of complexity. Some programs handle complicated tasks such as network management, table games, or encryption, while others deal with simpler tasks like string manipulation. Every program is labeled with the vulnerabilities found within the source code, indicating the type, line number, and vulnerable function name. This is accomplished by employing a formal verification method using the Efficient SMT-based Bounded Model Checker (ESBMC), which performs model checking, abstract interpretation, constraint programming, and satisfiability modulo theories, to reason over safety/security properties in programs. This approach definitively detects vulnerabilities and offers a formal model known as a counterexample, thus eliminating the possibility of generating false positive reports. This property of the dataset makes it suitable for evaluating the effectiveness of various static and dynamic analysis tools. Furthermore, we have associated the identified vulnerabilities with relevant Common Weakness Enumeration (CWE) numbers. We make the source code available for the 112,000 programs, accompanied by a comprehensive list detailing the vulnerabilities detected in each individual program including location and function name, which makes the dataset ideal to train LLMs and machine learning algorithms.

{{</citation>}}


## cs.IR (3)



### (110/115) Generative Job Recommendations with Large Language Model (Zhi Zheng et al., 2023)

{{<citation>}}

Zhi Zheng, Zhaopeng Qiu, Xiao Hu, Likang Wu, Hengshu Zhu, Hui Xiong. (2023)  
**Generative Job Recommendations with Large Language Model**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: AI, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02157v1)  

---


**ABSTRACT**  
The rapid development of online recruitment services has encouraged the utilization of recommender systems to streamline the job seeking process. Predominantly, current job recommendations deploy either collaborative filtering or person-job matching strategies. However, these models tend to operate as "black-box" systems and lack the capacity to offer explainable guidance to job seekers. Moreover, conventional matching-based recommendation methods are limited to retrieving and ranking existing jobs in the database, restricting their potential as comprehensive career AI advisors. To this end, here we present GIRL (GeneratIve job Recommendation based on Large language models), a novel approach inspired by recent advancements in the field of Large Language Models (LLMs). We initially employ a Supervised Fine-Tuning (SFT) strategy to instruct the LLM-based generator in crafting suitable Job Descriptions (JDs) based on the Curriculum Vitae (CV) of a job seeker. Moreover, we propose to train a model which can evaluate the matching degree between CVs and JDs as a reward model, and we use Proximal Policy Optimization (PPO)-based Reinforcement Learning (RL) method to further fine-tine the generator. This aligns the generator with recruiter feedback, tailoring the output to better meet employer preferences. In particular, GIRL serves as a job seeker-centric generative model, providing job suggestions without the need of a candidate set. This capability also enhances the performance of existing job recommendation models by supplementing job seeking features with generated content. With extensive experiments on a large-scale real-world dataset, we demonstrate the substantial effectiveness of our approach. We believe that GIRL introduces a paradigm-shifting approach to job recommendation systems, fostering a more personalized and comprehensive job-seeking experience.

{{</citation>}}


### (111/115) Recommender Systems in the Era of Large Language Models (LLMs) (Wenqi Fan et al., 2023)

{{<citation>}}

Wenqi Fan, Zihuai Zhao, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Jiliang Tang, Qing Li. (2023)  
**Recommender Systems in the Era of Large Language Models (LLMs)**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs.IR  
Keywords: AI, ChatGPT, GPT, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.02046v1)  

---


**ABSTRACT**  
With the prosperity of e-commerce and web applications, Recommender Systems (RecSys) have become an important component of our daily life, providing personalized suggestions that cater to user preferences. While Deep Neural Networks (DNNs) have made significant advancements in enhancing recommender systems by modeling user-item interactions and incorporating textual side information, DNN-based methods still face limitations, such as difficulties in understanding users' interests and capturing textual side information, inabilities in generalizing to various recommendation scenarios and reasoning on their predictions, etc. Meanwhile, the emergence of Large Language Models (LLMs), such as ChatGPT and GPT4, has revolutionized the fields of Natural Language Processing (NLP) and Artificial Intelligence (AI), due to their remarkable abilities in fundamental responsibilities of language understanding and generation, as well as impressive generalization and reasoning capabilities. As a result, recent studies have attempted to harness the power of LLMs to enhance recommender systems. Given the rapid evolution of this research direction in recommender systems, there is a pressing need for a systematic overview that summarizes existing LLM-empowered recommender systems, to provide researchers in relevant fields with an in-depth understanding. Therefore, in this paper, we conduct a comprehensive review of LLM-empowered recommender systems from various aspects including Pre-training, Fine-tuning, and Prompting. More specifically, we first introduce representative methods to harness the power of LLMs (as a feature encoder) for learning representations of users and items. Then, we review recent techniques of LLMs for enhancing recommender systems from three paradigms, namely pre-training, fine-tuning, and prompting. Finally, we comprehensively discuss future directions in this emerging field.

{{</citation>}}


### (112/115) Fisher-Weighted Merge of Contrastive Learning Models in Sequential Recommendation (Jung Hyun Ryu et al., 2023)

{{<citation>}}

Jung Hyun Ryu, Jaeheyoung Jeon, Jewoong Cho, Myungjoo Kang 1. (2023)  
**Fisher-Weighted Merge of Contrastive Learning Models in Sequential Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.05476v1)  

---


**ABSTRACT**  
Along with the exponential growth of online platforms and services, recommendation systems have become essential for identifying relevant items based on user preferences. The domain of sequential recommendation aims to capture evolving user preferences over time. To address dynamic preference, various contrastive learning methods have been proposed to target data sparsity, a challenge in recommendation systems due to the limited user-item interactions. In this paper, we are the first to apply the Fisher-Merging method to Sequential Recommendation, addressing and resolving practical challenges associated with it. This approach ensures robust fine-tuning by merging the parameters of multiple models, resulting in improved overall performance. Through extensive experiments, we demonstrate the effectiveness of our proposed methods, highlighting their potential to advance the state-of-the-art in sequential learning and recommendation systems.

{{</citation>}}


## stat.CO (1)



### (113/115) Adaptive multi-stage integration schemes for Hamiltonian Monte Carlo (Lorenzo Nagar et al., 2023)

{{<citation>}}

Lorenzo Nagar, Mario Fernández-Pendás, Jesús María Sanz-Serna, Elena Akhmatskaya. (2023)  
**Adaptive multi-stage integration schemes for Hamiltonian Monte Carlo**  

---
Primary Category: stat.CO  
Categories: cs-NA, math-NA, stat-CO, stat-ME, stat.CO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02096v1)  

---


**ABSTRACT**  
Hamiltonian Monte Carlo (HMC) is a powerful tool for Bayesian statistical inference due to its potential to rapidly explore high dimensional state space, avoiding the random walk behavior typical of many Markov Chain Monte Carlo samplers. The proper choice of the integrator of the Hamiltonian dynamics is key to the efficiency of HMC. It is becoming increasingly clear that multi-stage splitting integrators are a good alternative to the Verlet method, traditionally used in HMC. Here we propose a principled way of finding optimal, problem-specific integration schemes (in terms of the best conservation of energy for harmonic forces/Gaussian targets) within the families of 2- and 3-stage splitting integrators. The method, which we call Adaptive Integration Approach for statistics, or s-AIA, uses a multivariate Gaussian model and simulation data obtained at the HMC burn-in stage to identify a system-specific dimensional stability interval and assigns the most appropriate 2-/3-stage integrator for any user-chosen simulation step size within that interval. s-AIA has been implemented in the in-house software package HaiCS without introducing computational overheads in the simulations. The efficiency of the s-AIA integrators and their impact on the HMC accuracy, sampling performance and convergence are discussed in comparison with known fixed-parameter multi-stage splitting integrators (including Verlet). Numerical experiments on well-known statistical models show that the adaptive schemes reach the best possible performance within the family of 2-, 3-stage splitting schemes.

{{</citation>}}


## eess.SY (2)



### (114/115) Graph Neural Network-based Power Flow Model (Mingjian Tuo et al., 2023)

{{<citation>}}

Mingjian Tuo, Xingpeng Li, Tianxia Zhao. (2023)  
**Graph Neural Network-based Power Flow Model**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2307.02049v1)  

---


**ABSTRACT**  
Power flow analysis plays a crucial role in examining the electricity flow within a power system network. By performing power flow calculations, the system's steady-state variables, including voltage magnitude, phase angle at each bus, active/reactive power flow across branches, can be determined. While the widely used DC power flow model offers speed and robustness, it may yield inaccurate line flow results for certain transmission lines. This issue becomes more critical when dealing with renewable energy sources such as wind farms, which are often located far from the main grid. Obtaining precise line flow results for these critical lines is vital for next operations. To address these challenges, data-driven approaches leverage historical grid profiles. In this paper, a graph neural network (GNN) model is trained using historical power system data to predict power flow outcomes. The GNN model enables rapid estimation of line flows. A comprehensive performance analysis is conducted, comparing the proposed GNN-based power flow model with the traditional DC power flow model, as well as deep neural network (DNN) and convolutional neural network (CNN). The results on test systems demonstrate that the proposed GNN-based power flow model provides more accurate solutions with high efficiency comparing to benchmark models.

{{</citation>}}


### (115/115) Interpretable and Secure Trajectory Optimization for UAV-Assisted Communication (Yunhao Quan et al., 2023)

{{<citation>}}

Yunhao Quan, Nan Cheng, Xiucheng Wang, Jinglong Shen, Longfei Ma, Zhisheng Yin. (2023)  
**Interpretable and Secure Trajectory Optimization for UAV-Assisted Communication**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02002v1)  

---


**ABSTRACT**  
Unmanned aerial vehicles (UAVs) have gained popularity due to their flexible mobility, on-demand deployment, and the ability to establish high probability line-of-sight wireless communication. As a result, UAVs have been extensively used as aerial base stations (ABSs) to supplement ground-based cellular networks for various applications. However, existing UAV-assisted communication schemes mainly focus on trajectory optimization and power allocation, while ignoring the issue of collision avoidance during UAV flight. To address this issue, this paper proposes an interpretable UAV-assisted communication scheme that decomposes reliable UAV services into two sub-problems. The first is the constrained UAV coordinates and power allocation problem, which is solved using the Dueling Double DQN (D3QN) method. The second is the constrained UAV collision avoidance and trajectory optimization problem, which is addressed through the Monte Carlo tree search (MCTS) method. This approach ensures both reliable and efficient operation of UAVs. Moreover, we propose a scalable interpretable artificial intelligence (XAI) framework that enables more transparent and reliable system decisions. The proposed scheme's interpretability generates explainable and trustworthy results, making it easier to comprehend, validate, and control UAV-assisted communication solutions. Through extensive experiments, we demonstrate that our proposed algorithm outperforms existing techniques in terms of performance and generalization. The proposed model improves the reliability, efficiency, and safety of UAV-assisted communication systems, making it a promising solution for future UAV-assisted communication applications

{{</citation>}}
