---
draft: false
title: "arXiv @ 2023.09.04"
date: 2023-09-04
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.04"
    identifier: arxiv_20230904
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.AI (2)](#csai-2)
- [cs.IR (2)](#csir-2)
- [cs.CL (9)](#cscl-9)
- [cs.CV (12)](#cscv-12)
- [cs.LG (5)](#cslg-5)
- [cs.RO (3)](#csro-3)
- [eess.IV (3)](#eessiv-3)
- [cs.SD (2)](#cssd-2)
- [eess.SP (1)](#eesssp-1)
- [cs.SE (2)](#csse-2)
- [cs.SI (1)](#cssi-1)
- [econ.EM (1)](#econem-1)
- [cs.AR (1)](#csar-1)

## cs.AI (2)



### (1/44) Neurosymbolic Reinforcement Learning and Planning: A Survey (K. Acharya et al., 2023)

{{<citation>}}

K. Acharya, W. Raza, C. M. J. M. Dourado Jr, A. Velasquez, H. Song. (2023)  
**Neurosymbolic Reinforcement Learning and Planning: A Survey**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, Reasoning, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.01038v1)  

---


**ABSTRACT**  
The area of Neurosymbolic Artificial Intelligence (Neurosymbolic AI) is rapidly developing and has become a popular research topic, encompassing sub-fields such as Neurosymbolic Deep Learning (Neurosymbolic DL) and Neurosymbolic Reinforcement Learning (Neurosymbolic RL). Compared to traditional learning methods, Neurosymbolic AI offers significant advantages by simplifying complexity and providing transparency and explainability. Reinforcement Learning(RL), a long-standing Artificial Intelligence(AI) concept that mimics human behavior using rewards and punishment, is a fundamental component of Neurosymbolic RL, a recent integration of the two fields that has yielded promising results. The aim of this paper is to contribute to the emerging field of Neurosymbolic RL by conducting a literature survey. Our evaluation focuses on the three components that constitute Neurosymbolic RL: neural, symbolic, and RL. We categorize works based on the role played by the neural and symbolic parts in RL, into three taxonomies:Learning for Reasoning, Reasoning for Learning and Learning-Reasoning. These categories are further divided into sub-categories based on their applications. Furthermore, we analyze the RL components of each research work, including the state space, action space, policy module, and RL algorithm. Additionally, we identify research opportunities and challenges in various applications within this dynamic field.

{{</citation>}}


### (2/44) Zero-Shot Recommendations with Pre-Trained Large Language Models for Multimodal Nudging (Rachel Harrison et al., 2023)

{{<citation>}}

Rachel Harrison, Anton Dereventsov, Anton Bibin. (2023)  
**Zero-Shot Recommendations with Pre-Trained Large Language Models for Multimodal Nudging**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs-MM, cs.AI  
Keywords: AI, Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.01026v1)  

---


**ABSTRACT**  
We present a method for zero-shot recommendation of multimodal non-stationary content that leverages recent advancements in the field of generative AI. We propose rendering inputs of different modalities as textual descriptions and to utilize pre-trained LLMs to obtain their numerical representations by computing semantic embeddings. Once unified representations of all content items are obtained, the recommendation can be performed by computing an appropriate similarity metric between them without any additional learning. We demonstrate our approach on a synthetic multimodal nudging environment, where the inputs consist of tabular, textual, and visual data.

{{</citation>}}


## cs.IR (2)



### (3/44) Hessian-aware Quantized Node Embeddings for Recommendation (Huiyuan Chen et al., 2023)

{{<citation>}}

Huiyuan Chen, Kaixiong Zhou, Kwei-Herng Lai, Chin-Chia Michael Yeh, Yan Zheng, Xia Hu, Hao Yang. (2023)  
**Hessian-aware Quantized Node Embeddings for Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Embedding, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.01032v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have achieved state-of-the-art performance in recommender systems. Nevertheless, the process of searching and ranking from a large item corpus usually requires high latency, which limits the widespread deployment of GNNs in industry-scale applications. To address this issue, many methods compress user/item representations into the binary embedding space to reduce space requirements and accelerate inference. Also, they use the Straight-through Estimator (STE) to prevent vanishing gradients during back-propagation. However, the STE often causes the gradient mismatch problem, leading to sub-optimal results.   In this work, we present the Hessian-aware Quantized GNN (HQ-GNN) as an effective solution for discrete representations of users/items that enable fast retrieval. HQ-GNN is composed of two components: a GNN encoder for learning continuous node embeddings and a quantized module for compressing full-precision embeddings into low-bit ones. Consequently, HQ-GNN benefits from both lower memory requirements and faster inference speeds compared to vanilla GNNs. To address the gradient mismatch problem in STE, we further consider the quantized errors and its second-order derivatives for better stability. The experimental results on several large-scale datasets show that HQ-GNN achieves a good balance between latency and performance.

{{</citation>}}


### (4/44) MPTopic: Improving topic modeling via Masked Permuted pre-training (Xinche Zhang et al., 2023)

{{<citation>}}

Xinche Zhang, Evangelos milios. (2023)  
**MPTopic: Improving topic modeling via Masked Permuted pre-training**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.01015v1)  

---


**ABSTRACT**  
Topic modeling is pivotal in discerning hidden semantic structures within texts, thereby generating meaningful descriptive keywords. While innovative techniques like BERTopic and Top2Vec have recently emerged in the forefront, they manifest certain limitations. Our analysis indicates that these methods might not prioritize the refinement of their clustering mechanism, potentially compromising the quality of derived topic clusters. To illustrate, Top2Vec designates the centroids of clustering results to represent topics, whereas BERTopic harnesses C-TF-IDF for its topic extraction.In response to these challenges, we introduce "TF-RDF" (Term Frequency - Relative Document Frequency), a distinctive approach to assess the relevance of terms within a document. Building on the strengths of TF-RDF, we present MPTopic, a clustering algorithm intrinsically driven by the insights of TF-RDF. Through comprehensive evaluation, it is evident that the topic keywords identified with the synergy of MPTopic and TF-RDF outperform those extracted by both BERTopic and Top2Vec.

{{</citation>}}


## cs.CL (9)



### (5/44) Explainability for Large Language Models: A Survey (Haiyan Zhao et al., 2023)

{{<citation>}}

Haiyan Zhao, Hanjie Chen, Fan Yang, Ninghao Liu, Huiqi Deng, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Mengnan Du. (2023)  
**Explainability for Large Language Models: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2309.01029v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated impressive capabilities in natural language processing. However, their internal mechanisms are still unclear and this lack of transparency poses unwanted risks for downstream applications. Therefore, understanding and explaining these models is crucial for elucidating their behaviors, limitations, and social impacts. In this paper, we introduce a taxonomy of explainability techniques and provide a structured overview of methods for explaining Transformer-based language models. We categorize techniques based on the training paradigms of LLMs: traditional fine-tuning-based paradigm and prompting-based paradigm. For each paradigm, we summarize the goals and dominant approaches for generating local explanations of individual predictions and global explanations of overall model knowledge. We also discuss metrics for evaluating generated explanations, and discuss how explanations can be leveraged to debug models and improve performance. Lastly, we examine key challenges and emerging opportunities for explanation techniques in the era of LLMs in comparison to conventional machine learning models.

{{</citation>}}


### (6/44) ModelScope-Agent: Building Your Customizable Agent System with Open-source Large Language Models (Chenliang Li et al., 2023)

{{<citation>}}

Chenliang Li, Hehong Chen, Ming Yan, Weizhou Shen, Haiyang Xu, Zhikai Wu, Zhicheng Zhang, Wenmeng Zhou, Yingda Chen, Chen Cheng, Hongzhu Shi, Ji Zhang, Fei Huang, Jingren Zhou. (2023)  
**ModelScope-Agent: Building Your Customizable Agent System with Open-source Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.00986v1)  

---


**ABSTRACT**  
Large language models (LLMs) have recently demonstrated remarkable capabilities to comprehend human intentions, engage in reasoning, and design planning-like behavior. To further unleash the power of LLMs to accomplish complex tasks, there is a growing trend to build agent framework that equips LLMs, such as ChatGPT, with tool-use abilities to connect with massive external APIs. In this work, we introduce ModelScope-Agent, a general and customizable agent framework for real-world applications, based on open-source LLMs as controllers. It provides a user-friendly system library, with customizable engine design to support model training on multiple open-source LLMs, while also enabling seamless integration with both model APIs and common APIs in a unified way. To equip the LLMs with tool-use abilities, a comprehensive framework has been proposed spanning over tool-use data collection, tool retrieval, tool registration, memory control, customized model training, and evaluation for practical real-world applications. Finally, we showcase ModelScopeGPT, a real-world intelligent assistant of ModelScope Community based on the ModelScope-Agent framework, which is able to connect open-source LLMs with more than 1000 public AI models and localized community knowledge in ModelScope. The ModelScope-Agent library\footnote{https://github.com/modelscope/modelscope-agent} and online demo\footnote{https://modelscope.cn/studios/damo/ModelScopeGPT/summary} are now publicly available.

{{</citation>}}


### (7/44) Multilingual Text Representation (Fahim Faisal, 2023)

{{<citation>}}

Fahim Faisal. (2023)  
**Multilingual Text Representation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual, NLP  
[Paper Link](http://arxiv.org/abs/2309.00949v1)  

---


**ABSTRACT**  
Modern NLP breakthrough includes large multilingual models capable of performing tasks across more than 100 languages. State-of-the-art language models came a long way, starting from the simple one-hot representation of words capable of performing tasks like natural language understanding, common-sense reasoning, or question-answering, thus capturing both the syntax and semantics of texts. At the same time, language models are expanding beyond our known language boundary, even competitively performing over very low-resource dialects of endangered languages. However, there are still problems to solve to ensure an equitable representation of texts through a unified modeling space across language and speakers. In this survey, we shed light on this iterative progression of multilingual text representation and discuss the driving factors that ultimately led to the current state-of-the-art. Subsequently, we discuss how the full potential of language democratization could be obtained, reaching beyond the known limits and what is the scope of improvement in that space.

{{</citation>}}


### (8/44) Knowledge Graph Embeddings for Multi-Lingual Structured Representations of Radiology Reports (Tom van Sonsbeek et al., 2023)

{{<citation>}}

Tom van Sonsbeek, Xiantong Zhen, Marcel Warring. (2023)  
**Knowledge Graph Embeddings for Multi-Lingual Structured Representations of Radiology Reports**  

---
Primary Category: cs.CL  
Categories: 68T07, cs-AI, cs-CL, cs.CL  
Keywords: BERT, Clinical, Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.00917v1)  

---


**ABSTRACT**  
The way we analyse clinical texts has undergone major changes over the last years. The introduction of language models such as BERT led to adaptations for the (bio)medical domain like PubMedBERT and ClinicalBERT. These models rely on large databases of archived medical documents. While performing well in terms of accuracy, both the lack of interpretability and limitations to transfer across languages limit their use in clinical setting. We introduce a novel light-weight graph-based embedding method specifically catering radiology reports. It takes into account the structure and composition of the report, while also connecting medical terms in the report through the multi-lingual SNOMED Clinical Terms knowledge base. The resulting graph embedding uncovers the underlying relationships among clinical terms, achieving a representation that is better understandable for clinicians and clinically more accurate, without reliance on large pre-training datasets. We show the use of this embedding on two tasks namely disease classification of X-ray reports and image classification. For disease classification our model is competitive with its BERT-based counterparts, while being magnitudes smaller in size and training data requirements. For image classification, we show the effectiveness of the graph embedding leveraging cross-modal knowledge transfer and show how this method is usable across different languages.

{{</citation>}}


### (9/44) Evaluating Transformer's Ability to Learn Mildly Context-Sensitive Languages (Shunjie Wang et al., 2023)

{{<citation>}}

Shunjie Wang, Shane Steinert-Threlkeld. (2023)  
**Evaluating Transformer's Ability to Learn Mildly Context-Sensitive Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LSTM, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.00857v1)  

---


**ABSTRACT**  
Despite that Transformers perform well in NLP tasks, recent studies suggest that self-attention is theoretically limited in learning even some regular and context-free languages. These findings motivated us to think about their implications in modeling natural language, which is hypothesized to be mildly context-sensitive. We test Transformer's ability to learn a variety of mildly context-sensitive languages of varying complexities, and find that they generalize well to unseen in-distribution data, but their ability to extrapolate to longer strings is worse than that of LSTMs. Our analyses show that the learned self-attention patterns and representations modeled dependency relations and demonstrated counting behavior, which may have helped the models solve the languages.

{{</citation>}}


### (10/44) LeanContext: Cost-Efficient Domain-Specific Question Answering Using LLMs (Md Adnan Arefeen et al., 2023)

{{<citation>}}

Md Adnan Arefeen, Biplob Debnath, Srimat Chakradhar. (2023)  
**LeanContext: Cost-Efficient Domain-Specific Question Answering Using LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: AI, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.00841v1)  

---


**ABSTRACT**  
Question-answering (QA) is a significant application of Large Language Models (LLMs), shaping chatbot capabilities across healthcare, education, and customer service. However, widespread LLM integration presents a challenge for small businesses due to the high expenses of LLM API usage. Costs rise rapidly when domain-specific data (context) is used alongside queries for accurate domain-specific LLM responses. One option is to summarize the context by using LLMs and reduce the context. However, this can also filter out useful information that is necessary to answer some domain-specific queries. In this paper, we shift from human-oriented summarizers to AI model-friendly summaries. Our approach, LeanContext, efficiently extracts $k$ key sentences from the context that are closely aligned with the query. The choice of $k$ is neither static nor random; we introduce a reinforcement learning technique that dynamically determines $k$ based on the query and context. The rest of the less important sentences are reduced using a free open source text reduction method. We evaluate LeanContext against several recent query-aware and query-unaware context reduction approaches on prominent datasets (arxiv papers and BBC news articles). Despite cost reductions of $37.29\%$ to $67.81\%$, LeanContext's ROUGE-1 score decreases only by $1.41\%$ to $2.65\%$ compared to a baseline that retains the entire context (no summarization). Additionally, if free pretrained LLM-based summarizers are used to reduce context (into human consumable summaries), LeanContext can further modify the reduced context to enhance the accuracy (ROUGE-1 score) by $13.22\%$ to $24.61\%$.

{{</citation>}}


### (11/44) LinkTransformer: A Unified Package for Record Linkage with Transformer Language Models (Abhishek Arora et al., 2023)

{{<citation>}}

Abhishek Arora, Melissa Dell. (2023)  
**LinkTransformer: A Unified Package for Record Linkage with Transformer Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2309.00789v1)  

---


**ABSTRACT**  
Linking information across sources is fundamental to a variety of analyses in social science, business, and government. While large language models (LLMs) offer enormous promise for improving record linkage in noisy datasets, in many domains approximate string matching packages in popular softwares such as R and Stata remain predominant. These packages have clean, simple interfaces and can be easily extended to a diversity of languages. Our open-source package LinkTransformer aims to extend the familiarity and ease-of-use of popular string matching methods to deep learning. It is a general purpose package for record linkage with transformer LLMs that treats record linkage as a text retrieval problem. At its core is an off-the-shelf toolkit for applying transformer models to record linkage with four lines of code. LinkTransformer contains a rich repository of pre-trained transformer semantic similarity models for multiple languages and supports easy integration of any transformer language model from Hugging Face or OpenAI. It supports standard functionality such as blocking and linking on multiple noisy fields. LinkTransformer APIs also perform other common text data processing tasks, e.g., aggregation, noisy de-duplication, and translation-free cross-lingual linkage. Importantly, LinkTransformer also contains comprehensive tools for efficient model tuning, to facilitate different levels of customization when off-the-shelf models do not provide the required accuracy. Finally, to promote reusability, reproducibility, and extensibility, LinkTransformer makes it easy for users to contribute their custom-trained models to its model hub. By combining transformer language models with intuitive APIs that will be familiar to many users of popular string matching packages, LinkTransformer aims to democratize the benefits of LLMs among those who may be less familiar with deep learning frameworks.

{{</citation>}}


### (12/44) Value Kaleidoscope: Engaging AI with Pluralistic Human Values, Rights, and Duties (Taylor Sorensen et al., 2023)

{{<citation>}}

Taylor Sorensen, Liwei Jiang, Jena Hwang, Sydney Levine, Valentina Pyatkin, Peter West, Nouha Dziri, Ximing Lu, Kavel Rao, Chandra Bhagavatula, Maarten Sap, John Tasioulas, Yejin Choi. (2023)  
**Value Kaleidoscope: Engaging AI with Pluralistic Human Values, Rights, and Duties**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.00779v1)  

---


**ABSTRACT**  
Human values are crucial to human decision-making. Value pluralism is the view that multiple correct values may be held in tension with one another (e.g., when considering lying to a friend to protect their feelings, how does one balance honesty with friendship?). As statistical learners, AI systems fit to averages by default, washing out these potentially irreducible value conflicts. To improve AI systems to better reflect value pluralism, the first-order challenge is to explore the extent to which AI systems can model pluralistic human values, rights, and duties as well as their interaction.   We introduce ValuePrism, a large-scale dataset of 218k values, rights, and duties connected to 31k human-written situations. ValuePrism's contextualized values are generated by GPT-4 and deemed high-quality by human annotators 91% of the time. We conduct a large-scale study with annotators across diverse social and demographic backgrounds to try to understand whose values are represented.   With ValuePrism, we build Kaleido, an open, light-weight, and structured language-based multi-task model that generates, explains, and assesses the relevance and valence (i.e., support or oppose) of human values, rights, and duties within a specific context. Humans prefer the sets of values output by our system over the teacher GPT-4, finding them more accurate and with broader coverage. In addition, we demonstrate that Kaleido can help explain variability in human decision-making by outputting contrasting values. Finally, we show that Kaleido's representations transfer to other philosophical frameworks and datasets, confirming the benefit of an explicit, modular, and interpretable approach to value pluralism. We hope that our work will serve as a step to making more explicit the implicit values behind human decision-making and to steering AI systems to make decisions that are more in accordance with them.

{{</citation>}}


### (13/44) Bias and Fairness in Large Language Models: A Survey (Isabel O. Gallegos et al., 2023)

{{<citation>}}

Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, Nesreen K. Ahmed. (2023)  
**Bias and Fairness in Large Language Models: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2309.00770v1)  

---


**ABSTRACT**  
Rapid advancements of large language models (LLMs) have enabled the processing, understanding, and generation of human-like text, with increasing integration into systems that touch our social sphere. Despite this success, these models can learn, perpetuate, and amplify harmful social biases. In this paper, we present a comprehensive survey of bias evaluation and mitigation techniques for LLMs. We first consolidate, formalize, and expand notions of social bias and fairness in natural language processing, defining distinct facets of harm and introducing several desiderata to operationalize fairness for LLMs. We then unify the literature by proposing three intuitive taxonomies, two for bias evaluation, namely metrics and datasets, and one for mitigation. Our first taxonomy of metrics for bias evaluation disambiguates the relationship between metrics and evaluation datasets, and organizes metrics by the different levels at which they operate in a model: embeddings, probabilities, and generated text. Our second taxonomy of datasets for bias evaluation categorizes datasets by their structure as counterfactual inputs or prompts, and identifies the targeted harms and social groups; we also release a consolidation of publicly-available datasets for improved access. Our third taxonomy of techniques for bias mitigation classifies methods by their intervention during pre-processing, in-training, intra-processing, and post-processing, with granular subcategories that elucidate research trends. Finally, we identify open problems and challenges for future work. Synthesizing a wide range of recent research, we aim to provide a clear guide of the existing literature that empowers researchers and practitioners to better understand and prevent the propagation of bias in LLMs.

{{</citation>}}


## cs.CV (12)



### (14/44) Contrastive Grouping with Transformer for Referring Image Segmentation (Jiajin Tang et al., 2023)

{{<citation>}}

Jiajin Tang, Ge Zheng, Cheng Shi, Sibei Yang. (2023)  
**Contrastive Grouping with Transformer for Referring Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.01017v1)  

---


**ABSTRACT**  
Referring image segmentation aims to segment the target referent in an image conditioning on a natural language expression. Existing one-stage methods employ per-pixel classification frameworks, which attempt straightforwardly to align vision and language at the pixel level, thus failing to capture critical object-level information. In this paper, we propose a mask classification framework, Contrastive Grouping with Transformer network (CGFormer), which explicitly captures object-level information via token-based querying and grouping strategy. Specifically, CGFormer first introduces learnable query tokens to represent objects and then alternately queries linguistic features and groups visual features into the query tokens for object-aware cross-modal reasoning. In addition, CGFormer achieves cross-level interaction by jointly updating the query tokens and decoding masks in every two consecutive layers. Finally, CGFormer cooperates contrastive learning to the grouping strategy to identify the token and its mask corresponding to the referent. Experimental results demonstrate that CGFormer outperforms state-of-the-art methods in both segmentation and generalization settings consistently and significantly.

{{</citation>}}


### (15/44) RevColV2: Exploring Disentangled Representations in Masked Image Modeling (Qi Han et al., 2023)

{{<citation>}}

Qi Han, Yuxuan Cai, Xiangyu Zhang. (2023)  
**RevColV2: Exploring Disentangled Representations in Masked Image Modeling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.01005v1)  

---


**ABSTRACT**  
Masked image modeling (MIM) has become a prevalent pre-training setup for vision foundation models and attains promising performance. Despite its success, existing MIM methods discard the decoder network during downstream applications, resulting in inconsistent representations between pre-training and fine-tuning and can hamper downstream task performance. In this paper, we propose a new architecture, RevColV2, which tackles this issue by keeping the entire autoencoder architecture during both pre-training and fine-tuning. The main body of RevColV2 contains bottom-up columns and top-down columns, between which information is reversibly propagated and gradually disentangled. Such design enables our architecture with the nice property: maintaining disentangled low-level and semantic information at the end of the network in MIM pre-training. Our experimental results suggest that a foundation model with decoupled features can achieve competitive performance across multiple downstream vision tasks such as image classification, semantic segmentation and object detection. For example, after intermediate fine-tuning on ImageNet-22K dataset, RevColV2-L attains 88.4% top-1 accuracy on ImageNet-1K classification and 58.6 mIoU on ADE20K semantic segmentation. With extra teacher and large scale dataset, RevColv2-L achieves 62.1 box AP on COCO detection and 60.4 mIoU on ADE20K semantic segmentation. Code and models are released at https://github.com/megvii-research/RevCol

{{</citation>}}


### (16/44) S$^3$-MonoDETR: Supervised Shape&Scale-perceptive Deformable Transformer for Monocular 3D Object Detection (Xuan He et al., 2023)

{{<citation>}}

Xuan He, Kailun Yang, Junwei Zheng, Jin Yuan, Luis M. Bergasa, Hui Zhang, Zhiyong Li. (2023)  
**S$^3$-MonoDETR: Supervised Shape&Scale-perceptive Deformable Transformer for Monocular 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV, eess-IV  
Keywords: Attention, Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2309.00928v1)  

---


**ABSTRACT**  
Recently, transformer-based methods have shown exceptional performance in monocular 3D object detection, which can predict 3D attributes from a single 2D image. These methods typically use visual and depth representations to generate query points on objects, whose quality plays a decisive role in the detection accuracy. However, current unsupervised attention mechanisms without any geometry appearance awareness in transformers are susceptible to producing noisy features for query points, which severely limits the network performance and also makes the model have a poor ability to detect multi-category objects in a single training process. To tackle this problem, this paper proposes a novel "Supervised Shape&Scale-perceptive Deformable Attention" (S$^3$-DA) module for monocular 3D object detection. Concretely, S$^3$-DA utilizes visual and depth features to generate diverse local features with various shapes and scales and predict the corresponding matching distribution simultaneously to impose valuable shape&scale perception for each query. Benefiting from this, S$^3$-DA effectively estimates receptive fields for query points belonging to any category, enabling them to generate robust query features. Besides, we propose a Multi-classification-based Shape$\&$Scale Matching (MSM) loss to supervise the above process. Extensive experiments on KITTI and Waymo Open datasets demonstrate that S$^3$-DA significantly improves the detection accuracy, yielding state-of-the-art performance of single-category and multi-category 3D object detection in a single training process compared to the existing approaches. The source code will be made publicly available at https://github.com/mikasa3lili/S3-MonoDETR.

{{</citation>}}


### (17/44) GBE-MLZSL: A Group Bi-Enhancement Framework for Multi-Label Zero-Shot Learning (Ziming Liu et al., 2023)

{{<citation>}}

Ziming Liu, Jingcai Guo, Xiaocheng Lu, Song Guo, Peiran Dong, Jiewei Zhang. (2023)  
**GBE-MLZSL: A Group Bi-Enhancement Framework for Multi-Label Zero-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.00923v1)  

---


**ABSTRACT**  
This paper investigates a challenging problem of zero-shot learning in the multi-label scenario (MLZSL), wherein, the model is trained to recognize multiple unseen classes within a sample (e.g., an image) based on seen classes and auxiliary knowledge, e.g., semantic information. Existing methods usually resort to analyzing the relationship of various seen classes residing in a sample from the dimension of spatial or semantic characteristics, and transfer the learned model to unseen ones. But they ignore the effective integration of local and global features. That is, in the process of inferring unseen classes, global features represent the principal direction of the image in the feature space, while local features should maintain uniqueness within a certain range. This integrated neglect will make the model lose its grasp of the main components of the image. Relying only on the local existence of seen classes during the inference stage introduces unavoidable bias. In this paper, we propose a novel and effective group bi-enhancement framework for MLZSL, dubbed GBE-MLZSL, to fully make use of such properties and enable a more accurate and robust visual-semantic projection. Specifically, we split the feature maps into several feature groups, of which each feature group can be trained independently with the Local Information Distinguishing Module (LID) to ensure uniqueness. Meanwhile, a Global Enhancement Module (GEM) is designed to preserve the principal direction. Besides, a static graph structure is designed to construct the correlation of local features. Experiments on large-scale MLZSL benchmark datasets NUS-WIDE and Open-Images-v4 demonstrate that the proposed GBE-MLZSL outperforms other state-of-the-art methods with large margins.

{{</citation>}}


### (18/44) Fearless Luminance Adaptation: A Macro-Micro-Hierarchical Transformer for Exposure Correction (Gehui Li et al., 2023)

{{<citation>}}

Gehui Li, Jinyuan Liu, Long Ma, Zhiying Jiang, Xin Fan, Risheng Liu. (2023)  
**Fearless Luminance Adaptation: A Macro-Micro-Hierarchical Transformer for Exposure Correction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.00872v1)  

---


**ABSTRACT**  
Photographs taken with less-than-ideal exposure settings often display poor visual quality. Since the correction procedures vary significantly, it is difficult for a single neural network to handle all exposure problems. Moreover, the inherent limitations of convolutions, hinder the models ability to restore faithful color or details on extremely over-/under- exposed regions. To overcome these limitations, we propose a Macro-Micro-Hierarchical transformer, which consists of a macro attention to capture long-range dependencies, a micro attention to extract local features, and a hierarchical structure for coarse-to-fine correction. In specific, the complementary macro-micro attention designs enhance locality while allowing global interactions. The hierarchical structure enables the network to correct exposure errors of different scales layer by layer. Furthermore, we propose a contrast constraint and couple it seamlessly in the loss function, where the corrected image is pulled towards the positive sample and pushed away from the dynamically generated negative samples. Thus the remaining color distortion and loss of detail can be removed. We also extend our method as an image enhancer for low-light face recognition and low-light semantic segmentation. Experiments demonstrate that our approach obtains more attractive results than state-of-the-art methods quantitatively and qualitatively.

{{</citation>}}


### (19/44) A Post-Processing Based Bengali Document Layout Analysis with YOLOV8 (Nazmus Sakib Ahmed et al., 2023)

{{<citation>}}

Nazmus Sakib Ahmed, Saad Sakib Noor, Ashraful Islam Shanto Sikder, Abhijit Paul. (2023)  
**A Post-Processing Based Bengali Document Layout Analysis with YOLOV8**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2309.00848v1)  

---


**ABSTRACT**  
This paper focuses on enhancing Bengali Document Layout Analysis (DLA) using the YOLOv8 model and innovative post-processing techniques. We tackle challenges unique to the complex Bengali script by employing data augmentation for model robustness. After meticulous validation set evaluation, we fine-tune our approach on the complete dataset, leading to a two-stage prediction strategy for accurate element segmentation. Our ensemble model, combined with post-processing, outperforms individual base architectures, addressing issues identified in the BaDLAD dataset. By leveraging this approach, we aim to advance Bengali document analysis, contributing to improved OCR and document comprehension and BaDLAD serves as a foundational resource for this endeavor, aiding future research in the field. Furthermore, our experiments provided key insights to incorporate new strategies into the established solution.

{{</citation>}}


### (20/44) Domain Generalization via Balancing Training Difficulty and Model Capability (Xueying Jiang et al., 2023)

{{<citation>}}

Xueying Jiang, Jiaxing Huang, Sheng Jin, Shijian Lu. (2023)  
**Domain Generalization via Balancing Training Difficulty and Model Capability**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.00844v1)  

---


**ABSTRACT**  
Domain generalization (DG) aims to learn domain-generalizable models from one or multiple source domains that can perform well in unseen target domains. Despite its recent progress, most existing work suffers from the misalignment between the difficulty level of training samples and the capability of contemporarily trained models, leading to over-fitting or under-fitting in the trained generalization model. We design MoDify, a Momentum Difficulty framework that tackles the misalignment by balancing the seesaw between the model's capability and the samples' difficulties along the training process. MoDify consists of two novel designs that collaborate to fight against the misalignment while learning domain-generalizable models. The first is MoDify-based Data Augmentation which exploits an RGB Shuffle technique to generate difficulty-aware training samples on the fly. The second is MoDify-based Network Optimization which dynamically schedules the training samples for balanced and smooth learning with appropriate difficulty. Without bells and whistles, a simple implementation of MoDify achieves superior performance across multiple benchmarks. In addition, MoDify can complement existing methods as a plug-in, and it is generic and can work for different visual recognition tasks.

{{</citation>}}


### (21/44) ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data (Ulyana Tkachenko et al., 2023)

{{<citation>}}

Ulyana Tkachenko, Aditya Thyagarajan, Jonas Mueller. (2023)  
**ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.00832v1)  

---


**ABSTRACT**  
Despite powering sensitive systems like autonomous vehicles, object detection remains fairly brittle in part due to annotation errors that plague most real-world training datasets. We propose ObjectLab, a straightforward algorithm to detect diverse errors in object detection labels, including: overlooked bounding boxes, badly located boxes, and incorrect class label assignments. ObjectLab utilizes any trained object detection model to score the label quality of each image, such that mislabeled images can be automatically prioritized for label review/correction. Properly handling erroneous data enables training a better version of the same object detection model, without any change in existing modeling code. Across different object detection datasets (including COCO) and different models (including Detectron-X101 and Faster-RCNN), ObjectLab consistently detects annotation errors with much better precision/recall compared to other label quality scores.

{{</citation>}}


### (22/44) Leveraging Semi-Supervised Graph Learning for Enhanced Diabetic Retinopathy Detection (D. Dhinakaran et al., 2023)

{{<citation>}}

D. Dhinakaran, L. Srinivasan, D. Selvaraj, S. M. Udhaya Sankar. (2023)  
**Leveraging Semi-Supervised Graph Learning for Enhanced Diabetic Retinopathy Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2309.00824v1)  

---


**ABSTRACT**  
Diabetic Retinopathy (DR) is a significant cause of blindness globally, highlighting the urgent need for early detection and effective treatment. Recent advancements in Machine Learning (ML) techniques have shown promise in DR detection, but the availability of labeled data often limits their performance. This research proposes a novel Semi-Supervised Graph Learning SSGL algorithm tailored for DR detection, which capitalizes on the relationships between labelled and unlabeled data to enhance accuracy. The work begins by investigating data augmentation and preprocessing techniques to address the challenges of image quality and feature variations. Techniques such as image cropping, resizing, contrast adjustment, normalization, and data augmentation are explored to optimize feature extraction and improve the overall quality of retinal images. Moreover, apart from detection and diagnosis, this work delves into applying ML algorithms for predicting the risk of developing DR or the likelihood of disease progression. Personalized risk scores for individual patients are generated using comprehensive patient data encompassing demographic information, medical history, and retinal images. The proposed Semi-Supervised Graph learning algorithm is rigorously evaluated on two publicly available datasets and is benchmarked against existing methods. Results indicate significant improvements in classification accuracy, specificity, and sensitivity while demonstrating robustness against noise and outlie rs.Notably, the proposed algorithm addresses the challenge of imbalanced datasets, common in medical image analysis, further enhancing its practical applicability.

{{</citation>}}


### (23/44) RenAIssance: A Survey into AI Text-to-Image Generation in the Era of Large Model (Fengxiang Bie et al., 2023)

{{<citation>}}

Fengxiang Bie, Yibo Yang, Zhongzhu Zhou, Adam Ghanem, Minjia Zhang, Zhewei Yao, Xiaoxia Wu, Connor Holmes, Pareesa Golnari, David A. Clifton, Yuxiong He, Dacheng Tao, Shuaiwen Leon Song. (2023)  
**RenAIssance: A Survey into AI Text-to-Image Generation in the Era of Large Model**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2309.00810v1)  

---


**ABSTRACT**  
Text-to-image generation (TTI) refers to the usage of models that could process text input and generate high fidelity images based on text descriptions. Text-to-image generation using neural networks could be traced back to the emergence of Generative Adversial Network (GAN), followed by the autoregressive Transformer. Diffusion models are one prominent type of generative model used for the generation of images through the systematic introduction of noises with repeating steps. As an effect of the impressive results of diffusion models on image synthesis, it has been cemented as the major image decoder used by text-to-image models and brought text-to-image generation to the forefront of machine-learning (ML) research. In the era of large models, scaling up model size and the integration with large language models have further improved the performance of TTI models, resulting the generation result nearly indistinguishable from real-world images, revolutionizing the way we retrieval images. Our explorative study has incentivised us to think that there are further ways of scaling text-to-image models with the combination of innovative model architectures and prediction enhancement techniques. We have divided the work of this survey into five main sections wherein we detail the frameworks of major literature in order to delve into the different types of text-to-image generation methods. Following this we provide a detailed comparison and critique of these methods and offer possible pathways of improvement for future work. In the future work, we argue that TTI development could yield impressive productivity improvements for creation, particularly in the context of the AIGC era, and could be extended to more complex tasks such as video generation and 3D generation.

{{</citation>}}


### (24/44) AttT2M: Text-Driven Human Motion Generation with Multi-Perspective Attention Mechanism (Chongyang Zhong et al., 2023)

{{<citation>}}

Chongyang Zhong, Lei Hu, Zihao Zhang, Shihong Xia. (2023)  
**AttT2M: Text-Driven Human Motion Generation with Multi-Perspective Attention Mechanism**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.00796v1)  

---


**ABSTRACT**  
Generating 3D human motion based on textual descriptions has been a research focus in recent years. It requires the generated motion to be diverse, natural, and conform to the textual description. Due to the complex spatio-temporal nature of human motion and the difficulty in learning the cross-modal relationship between text and motion, text-driven motion generation is still a challenging problem. To address these issues, we propose \textbf{AttT2M}, a two-stage method with multi-perspective attention mechanism: \textbf{body-part attention} and \textbf{global-local motion-text attention}. The former focuses on the motion embedding perspective, which means introducing a body-part spatio-temporal encoder into VQ-VAE to learn a more expressive discrete latent space. The latter is from the cross-modal perspective, which is used to learn the sentence-level and word-level motion-text cross-modal relationship. The text-driven motion is finally generated with a generative transformer. Extensive experiments conducted on HumanML3D and KIT-ML demonstrate that our method outperforms the current state-of-the-art works in terms of qualitative and quantitative evaluation, and achieve fine-grained synthesis and action2motion. Our code is in https://github.com/ZcyMonkey/AttT2M

{{</citation>}}


### (25/44) Contrastive Feature Masking Open-Vocabulary Vision Transformer (Dahun Kim et al., 2023)

{{<citation>}}

Dahun Kim, Anelia Angelova, Weicheng Kuo. (2023)  
**Contrastive Feature Masking Open-Vocabulary Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Embedding, Transformer  
[Paper Link](http://arxiv.org/abs/2309.00775v1)  

---


**ABSTRACT**  
We present Contrastive Feature Masking Vision Transformer (CFM-ViT) - an image-text pretraining methodology that achieves simultaneous learning of image- and region-level representation for open-vocabulary object detection (OVD). Our approach combines the masked autoencoder (MAE) objective into the contrastive learning objective to improve the representation for localization tasks. Unlike standard MAE, we perform reconstruction in the joint image-text embedding space, rather than the pixel space as is customary with the classical MAE method, which causes the model to better learn region-level semantics. Moreover, we introduce Positional Embedding Dropout (PED) to address scale variation between image-text pretraining and detection finetuning by randomly dropping out the positional embeddings during pretraining. PED improves detection performance and enables the use of a frozen ViT backbone as a region classifier, preventing the forgetting of open-vocabulary knowledge during detection finetuning. On LVIS open-vocabulary detection benchmark, CFM-ViT achieves a state-of-the-art 33.9 AP$r$, surpassing the best approach by 7.6 points and achieves better zero-shot detection transfer. Finally, CFM-ViT acquires strong image-level representation, outperforming the state of the art on 8 out of 12 metrics on zero-shot image-text retrieval benchmarks.

{{</citation>}}


## cs.LG (5)



### (26/44) Streaming Active Learning for Regression Problems Using Regression via Classification (Shota Horiguchi et al., 2023)

{{<citation>}}

Shota Horiguchi, Kota Dohi, Yohei Kawaguchi. (2023)  
**Streaming Active Learning for Regression Problems Using Regression via Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2309.01013v1)  

---


**ABSTRACT**  
One of the challenges in deploying a machine learning model is that the model's performance degrades as the operating environment changes. To maintain the performance, streaming active learning is used, in which the model is retrained by adding a newly annotated sample to the training dataset if the prediction of the sample is not certain enough. Although many streaming active learning methods have been proposed for classification, few efforts have been made for regression problems, which are often handled in the industrial field. In this paper, we propose to use the regression-via-classification framework for streaming active learning for regression. Regression-via-classification transforms regression problems into classification problems so that streaming active learning methods proposed for classification problems can be applied directly to regression problems. Experimental validation on four real data sets shows that the proposed method can perform regression with higher accuracy at the same annotation cost.

{{</citation>}}


### (27/44) eDKM: An Efficient and Accurate Train-time Weight Clustering for Large Language Models (Minsik Cho et al., 2023)

{{<citation>}}

Minsik Cho, Keivan A. Vahid, Qichen Fu, Saurabh Adya, Carlo C Del Mundo, Mohammad Rastegari, Devang Naik, Peter Zatloukal. (2023)  
**eDKM: An Efficient and Accurate Train-time Weight Clustering for Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LLaMA, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2309.00964v1)  

---


**ABSTRACT**  
Since Large Language Models or LLMs have demonstrated high-quality performance on many complex language tasks, there is a great interest in bringing these LLMs to mobile devices for faster responses and better privacy protection. However, the size of LLMs (i.e., billions of parameters) requires highly effective compression to fit into storage-limited devices. Among many compression techniques, weight-clustering, a form of non-linear quantization, is one of the leading candidates for LLM compression, and supported by modern smartphones. Yet, its training overhead is prohibitively significant for LLM fine-tuning. Especially, Differentiable KMeans Clustering, or DKM, has shown the state-of-the-art trade-off between compression ratio and accuracy regression, but its large memory complexity makes it nearly impossible to apply to train-time LLM compression. In this paper, we propose a memory-efficient DKM implementation, eDKM powered by novel techniques to reduce the memory footprint of DKM by orders of magnitudes. For a given tensor to be saved on CPU for the backward pass of DKM, we compressed the tensor by applying uniquification and sharding after checking if there is no duplicated tensor previously copied to CPU. Our experimental results demonstrate that \prjname can fine-tune and compress a pretrained LLaMA 7B model from 12.6 GB to 2.5 GB (3bit/weight) with the Alpaca dataset by reducing the train-time memory footprint of a decoder layer by 130$\times$, while delivering good accuracy on broader LLM benchmarks (i.e., 77.7\% for PIQA, 66.1\% for Winograde, and so on).

{{</citation>}}


### (28/44) Emergent Linear Representations in World Models of Self-Supervised Sequence Models (Neel Nanda et al., 2023)

{{<citation>}}

Neel Nanda, Andrew Lee, Martin Wattenberg. (2023)  
**Emergent Linear Representations in World Models of Self-Supervised Sequence Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.00941v1)  

---


**ABSTRACT**  
How do sequence models represent their decision-making process? Prior work suggests that Othello-playing neural network learned nonlinear models of the board state (Li et al., 2023). In this work, we provide evidence of a closely related linear representation of the board. In particular, we show that probing for ``my colour'' vs. ``opponent's colour'' may be a simple yet powerful way to interpret the model's internal state. This precise understanding of the internal representations allows us to control the model's behaviour with simple vector arithmetic. Linear representations enable significant interpretability progress, which we demonstrate with further exploration of how the world model is computed.

{{</citation>}}


### (29/44) DoRA: Domain-Based Self-Supervised Learning Framework for Low-Resource Real Estate Appraisal (Wei-Wei Du et al., 2023)

{{<citation>}}

Wei-Wei Du, Wei-Yao Wang, Wen-Chih Peng. (2023)  
**DoRA: Domain-Based Self-Supervised Learning Framework for Low-Resource Real Estate Appraisal**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Low-Resource, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.00855v1)  

---


**ABSTRACT**  
The marketplace system connecting demands and supplies has been explored to develop unbiased decision-making in valuing properties. Real estate appraisal serves as one of the high-cost property valuation tasks for financial institutions since it requires domain experts to appraise the estimation based on the corresponding knowledge and the judgment of the market. Existing automated valuation models reducing the subjectivity of domain experts require a large number of transactions for effective evaluation, which is predominantly limited to not only the labeling efforts of transactions but also the generalizability of new developing and rural areas. To learn representations from unlabeled real estate sets, existing self-supervised learning (SSL) for tabular data neglects various important features, and fails to incorporate domain knowledge. In this paper, we propose DoRA, a Domain-based self-supervised learning framework for low-resource Real estate Appraisal. DoRA is pre-trained with an intra-sample geographic prediction as the pretext task based on the metadata of the real estate for equipping the real estate representations with prior domain knowledge. Furthermore, inter-sample contrastive learning is employed to generalize the representations to be robust for limited transactions of downstream tasks. Our benchmark results on three property types of real-world transactions show that DoRA significantly outperforms the SSL baselines for tabular data, the graph-based methods, and the supervised approaches in the few-shot scenarios by at least 7.6% for MAPE, 11.59% for MAE, and 3.34% for HR10%. We expect DoRA to be useful to other financial practitioners with similar marketplace applications who need general models for properties that are newly built and have limited records. The source code is available at https://github.com/wwweiwei/DoRA.

{{</citation>}}


### (30/44) Autonomous Soft Tissue Retraction Using Demonstration-Guided Reinforcement Learning (Amritpal Singh et al., 2023)

{{<citation>}}

Amritpal Singh, Wenqi Shi, May D Wang. (2023)  
**Autonomous Soft Tissue Retraction Using Demonstration-Guided Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.00837v1)  

---


**ABSTRACT**  
In the context of surgery, robots can provide substantial assistance by performing small, repetitive tasks such as suturing, needle exchange, and tissue retraction, thereby enabling surgeons to concentrate on more complex aspects of the procedure. However, existing surgical task learning mainly pertains to rigid body interactions, whereas the advancement towards more sophisticated surgical robots necessitates the manipulation of soft bodies. Previous work focused on tissue phantoms for soft tissue task learning, which can be expensive and can be an entry barrier to research. Simulation environments present a safe and efficient way to learn surgical tasks before their application to actual tissue. In this study, we create a Robot Operating System (ROS)-compatible physics simulation environment with support for both rigid and soft body interactions within surgical tasks. Furthermore, we investigate the soft tissue interactions facilitated by the patient-side manipulator of the DaVinci surgical robot. Leveraging the pybullet physics engine, we simulate kinematics and establish anchor points to guide the robotic arm when manipulating soft tissue. Using demonstration-guided reinforcement learning (RL) algorithms, we investigate their performance in comparison to traditional reinforcement learning algorithms. Our in silico trials demonstrate a proof-of-concept for autonomous surgical soft tissue retraction. The results corroborate the feasibility of learning soft body manipulation through the application of reinforcement learning agents. This work lays the foundation for future research into the development and refinement of surgical robots capable of managing both rigid and soft tissue interactions. Code is available at https://github.com/amritpal-001/tissue_retract.

{{</citation>}}


## cs.RO (3)



### (31/44) Multi-agent Collective Construction using 3D Decomposition (Akshaya Kesarimangalam Srinivasan et al., 2023)

{{<citation>}}

Akshaya Kesarimangalam Srinivasan, Shambhavi Singh, Geordan Gutow, Howie Choset, Bhaskar Vundurthy. (2023)  
**Multi-agent Collective Construction using 3D Decomposition**  

---
Primary Category: cs.RO  
Categories: cs-MA, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.00985v1)  

---


**ABSTRACT**  
This paper addresses a Multi-Agent Collective Construction (MACC) problem that aims to build a three-dimensional structure comprised of cubic blocks. We use cube-shaped robots that can carry one cubic block at a time, and move forward, reverse, left, and right to an adjacent cell of the same height or climb up and down one cube height. To construct structures taller than one cube, the robots must build supporting stairs made of blocks and remove the stairs once the structure is built. Conventional techniques solve for the entire structure at once and quickly become intractable for larger workspaces and complex structures, especially in a multi-agent setting. To this end, we present a decomposition algorithm that computes valid substructures based on intrinsic structural dependencies. We use Mixed Integer Linear Programming (MILP) to solve for each of these substructures and then aggregate the solutions to construct the entire structure. Extensive testing on 200 randomly generated structures shows an order of magnitude improvement in the solution computation time compared to an MILP approach without decomposition. Additionally, compared to Reinforcement Learning (RL) based and heuristics-based approaches drawn from the literature, our solution indicates orders of magnitude improvement in the number of pick-up and drop-off actions required to construct a structure. Furthermore, we leverage the independence between substructures to detect which sub-structures can be built in parallel. With this parallelization technique, we illustrate a further improvement in the number of time steps required to complete building the structure. This work is a step towards applying multi-agent collective construction for real-world structures by significantly reducing solution computation time with a bounded increase in the number of time steps required to build the structure.

{{</citation>}}


### (32/44) Developmental Scaffolding with Large Language Models (Batuhan Celik et al., 2023)

{{<citation>}}

Batuhan Celik, Alper Ahmetoglu, Emre Ugur, Erhan Oztop. (2023)  
**Developmental Scaffolding with Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.00904v1)  

---


**ABSTRACT**  
Exploratoration and self-observation are key mechanisms of infant sensorimotor development. These processes are further guided by parental scaffolding accelerating skill and knowledge acquisition. In developmental robotics, this approach has been adopted often by having a human acting as the source of scaffolding. In this study, we investigate whether Large Language Models (LLMs) can act as a scaffolding agent for a robotic system that aims to learn to predict the effects of its actions. To this end, an object manipulation setup is considered where one object can be picked and placed on top of or in the vicinity of another object. The adopted LLM is asked to guide the action selection process through algorithmically generated state descriptions and action selection alternatives in natural language. The simulation experiments that include cubes in this setup show that LLM-guided (GPT3.5-guided) learning yields significantly faster discovery of novel structures compared to random exploration. However, we observed that GPT3.5 fails to effectively guide the robot in generating structures with different affordances such as cubes and spheres. Overall, we conclude that even without fine-tuning, LLMs may serve as a moderate scaffolding agent for improving robot learning, however, they still lack affordance understanding which limits the applicability of the current LLMs in robotic scaffolding tasks.

{{</citation>}}


### (33/44) Deep Reinforcement Learning in Surgical Robotics: Enhancing the Automation Level (Cheng Qian et al., 2023)

{{<citation>}}

Cheng Qian, Hongliang Ren. (2023)  
**Deep Reinforcement Learning in Surgical Robotics: Enhancing the Automation Level**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.00773v1)  

---


**ABSTRACT**  
Surgical robotics is a rapidly evolving field that is transforming the landscape of surgeries. Surgical robots have been shown to enhance precision, minimize invasiveness, and alleviate surgeon fatigue. One promising area of research in surgical robotics is the use of reinforcement learning to enhance the automation level. Reinforcement learning is a type of machine learning that involves training an agent to make decisions based on rewards and punishments. This literature review aims to comprehensively analyze existing research on reinforcement learning in surgical robotics. The review identified various applications of reinforcement learning in surgical robotics, including pre-operative, intra-body, and percutaneous procedures, listed the typical studies, and compared their methodologies and results. The findings show that reinforcement learning has great potential to improve the autonomy of surgical robots. Reinforcement learning can teach robots to perform complex surgical tasks, such as suturing and tissue manipulation. It can also improve the accuracy and precision of surgical robots, making them more effective at performing surgeries.

{{</citation>}}


## eess.IV (3)



### (34/44) AdLER: Adversarial Training with Label Error Rectification for One-Shot Medical Image Segmentation (Xiangyu Zhao et al., 2023)

{{<citation>}}

Xiangyu Zhao, Sheng Wang, Zhiyun Song, Zhenrong Shen, Linlin Yao, Haolei Yuan, Qian Wang, Lichi Zhang. (2023)  
**AdLER: Adversarial Training with Label Error Rectification for One-Shot Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2309.00971v1)  

---


**ABSTRACT**  
Accurate automatic segmentation of medical images typically requires large datasets with high-quality annotations, making it less applicable in clinical settings due to limited training data. One-shot segmentation based on learned transformations (OSSLT) has shown promise when labeled data is extremely limited, typically including unsupervised deformable registration, data augmentation with learned registration, and segmentation learned from augmented data. However, current one-shot segmentation methods are challenged by limited data diversity during augmentation, and potential label errors caused by imperfect registration. To address these issues, we propose a novel one-shot medical image segmentation method with adversarial training and label error rectification (AdLER), with the aim of improving the diversity of generated data and correcting label errors to enhance segmentation performance. Specifically, we implement a novel dual consistency constraint to ensure anatomy-aligned registration that lessens registration errors. Furthermore, we develop an adversarial training strategy to augment the atlas image, which ensures both generation diversity and segmentation robustness. We also propose to rectify potential label errors in the augmented atlas images by estimating segmentation uncertainty, which can compensate for the imperfect nature of deformable registration and improve segmentation authenticity. Experiments on the CANDI and ABIDE datasets demonstrate that the proposed AdLER outperforms previous state-of-the-art methods by 0.7% (CANDI), 3.6% (ABIDE "seen"), and 4.9% (ABIDE "unseen") in segmentation based on Dice scores, respectively. The source code will be available at https://github.com/hsiangyuzhao/AdLER.

{{</citation>}}


### (35/44) A Generic Fundus Image Enhancement Network Boosted by Frequency Self-supervised Representation Learning (Heng Li et al., 2023)

{{<citation>}}

Heng Li, Haofeng Liu, Huazhu Fu, Yanwu Xu, Hui Shu, Ke Niu, Yan Hu, Jiang Liu. (2023)  
**A Generic Fundus Image Enhancement Network Boosted by Frequency Self-supervised Representation Learning**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.00885v1)  

---


**ABSTRACT**  
Fundus photography is prone to suffer from image quality degradation that impacts clinical examination performed by ophthalmologists or intelligent systems. Though enhancement algorithms have been developed to promote fundus observation on degraded images, high data demands and limited applicability hinder their clinical deployment. To circumvent this bottleneck, a generic fundus image enhancement network (GFE-Net) is developed in this study to robustly correct unknown fundus images without supervised or extra data. Levering image frequency information, self-supervised representation learning is conducted to learn robust structure-aware representations from degraded images. Then with a seamless architecture that couples representation learning and image enhancement, GFE-Net can accurately correct fundus images and meanwhile preserve retinal structures. Comprehensive experiments are implemented to demonstrate the effectiveness and advantages of GFE-Net. Compared with state-of-the-art algorithms, GFE-Net achieves superior performance in data dependency, enhancement performance, deployment efficiency, and scale generalizability. Follow-up fundus image analysis is also facilitated by GFE-Net, whose modules are respectively verified to be effective for image enhancement.

{{</citation>}}


### (36/44) Full Reference Video Quality Assessment for Machine Learning-Based Video Codecs (Abrar Majeedi et al., 2023)

{{<citation>}}

Abrar Majeedi, Babak Naderi, Yasaman Hosseinkashi, Juhee Cho, Ruben Alvarez Martinez, Ross Cutler. (2023)  
**Full Reference Video Quality Assessment for Machine Learning-Based Video Codecs**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.00769v1)  

---


**ABSTRACT**  
Machine learning-based video codecs have made significant progress in the past few years. A critical area in the development of ML-based video codecs is an accurate evaluation metric that does not require an expensive and slow subjective test. We show that existing evaluation metrics that were designed and trained on DSP-based video codecs are not highly correlated to subjective opinion when used with ML video codecs due to the video artifacts being quite different between ML and video codecs. We provide a new dataset of ML video codec videos that have been accurately labeled for quality. We also propose a new full reference video quality assessment (FRVQA) model that achieves a Pearson Correlation Coefficient (PCC) of 0.99 and a Spearman's Rank Correlation Coefficient (SRCC) of 0.99 at the model level. We make the dataset and FRVQA model open source to help accelerate research in ML video codecs, and so that others can further improve the FRVQA model.

{{</citation>}}


## cs.SD (2)



### (37/44) Timbre-reserved Adversarial Attack in Speaker Identification (Qing Wang et al., 2023)

{{<citation>}}

Qing Wang, Jixun Yao, Li Zhang, Pengcheng Guo, Lei Xie. (2023)  
**Timbre-reserved Adversarial Attack in Speaker Identification**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2309.00929v1)  

---


**ABSTRACT**  
As a type of biometric identification, a speaker identification (SID) system is confronted with various kinds of attacks. The spoofing attacks typically imitate the timbre of the target speakers, while the adversarial attacks confuse the SID system by adding a well-designed adversarial perturbation to an arbitrary speech. Although the spoofing attack copies a similar timbre as the victim, it does not exploit the vulnerability of the SID model and may not make the SID system give the attacker's desired decision. As for the adversarial attack, despite the SID system can be led to a designated decision, it cannot meet the specified text or speaker timbre requirements for the specific attack scenarios. In this study, to make the attack in SID not only leverage the vulnerability of the SID model but also reserve the timbre of the target speaker, we propose a timbre-reserved adversarial attack in the speaker identification. We generate the timbre-reserved adversarial audios by adding an adversarial constraint during the different training stages of the voice conversion (VC) model. Specifically, the adversarial constraint is using the target speaker label to optimize the adversarial perturbation added to the VC model representations and is implemented by a speaker classifier joining in the VC model training. The adversarial constraint can help to control the VC model to generate the speaker-wised audio. Eventually, the inference of the VC model is the ideal adversarial fake audio, which is timbre-reserved and can fool the SID system.

{{</citation>}}


### (38/44) Pretraining Representations for Bioacoustic Few-shot Detection using Supervised Contrastive Learning (Ilyass Moummad et al., 2023)

{{<citation>}}

Ilyass Moummad, Romain Serizel, Nicolas Farrugia. (2023)  
**Pretraining Representations for Bioacoustic Few-shot Detection using Supervised Contrastive Learning**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.00878v1)  

---


**ABSTRACT**  
Deep learning has been widely used recently for sound event detection and classification. Its success is linked to the availability of sufficiently large datasets, possibly with corresponding annotations when supervised learning is considered. In bioacoustic applications, most tasks come with few labelled training data, because annotating long recordings is time consuming and costly. Therefore supervised learning is not the best suited approach to solve bioacoustic tasks. The bioacoustic community recasted the problem of sound event detection within the framework of few-shot learning, i.e. training a system with only few labeled examples. The few-shot bioacoustic sound event detection task in the DCASE challenge focuses on detecting events in long audio recordings given only five annotated examples for each class of interest. In this paper, we show that learning a rich feature extractor from scratch can be achieved by leveraging data augmentation using a supervised contrastive learning framework. We highlight the ability of this framework to transfer well for five-shot event detection on previously unseen classes in the training data. We obtain an F-score of 63.46\% on the validation set and 42.7\% on the test set, ranking second in the DCASE challenge. We provide an ablation study for the critical choices of data augmentation techniques as well as for the learning strategy applied on the training set.

{{</citation>}}


## eess.SP (1)



### (39/44) A Multi-Head Ensemble Multi-Task Learning Approach for Dynamical Computation Offloading (Ruihuai Liang et al., 2023)

{{<citation>}}

Ruihuai Liang, Bo Yang, Zhiwen Yu, Xuelin Cao, Derrick Wing Kwan Ng, Chau Yuen. (2023)  
**A Multi-Head Ensemble Multi-Task Learning Approach for Dynamical Computation Offloading**  

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.00907v1)  

---


**ABSTRACT**  
Computation offloading has become a popular solution to support computationally intensive and latency-sensitive applications by transferring computing tasks to mobile edge servers (MESs) for execution, which is known as mobile/multi-access edge computing (MEC). To improve the MEC performance, it is required to design an optimal offloading strategy that includes offloading decision (i.e., whether offloading or not) and computational resource allocation of MEC. The design can be formulated as a mixed-integer nonlinear programming (MINLP) problem, which is generally NP-hard and its effective solution can be obtained by performing online inference through a well-trained deep neural network (DNN) model. However, when the system environments change dynamically, the DNN model may lose efficacy due to the drift of input parameters, thereby decreasing the generalization ability of the DNN model. To address this unique challenge, in this paper, we propose a multi-head ensemble multi-task learning (MEMTL) approach with a shared backbone and multiple prediction heads (PHs). Specifically, the shared backbone will be invariant during the PHs training and the inferred results will be ensembled, thereby significantly reducing the required training overhead and improving the inference performance. As a result, the joint optimization problem for offloading decision and resource allocation can be efficiently solved even in a time-varying wireless environment. Experimental results show that the proposed MEMTL outperforms benchmark methods in both the inference accuracy and mean square error without requiring additional training data.

{{</citation>}}


## cs.SE (2)



### (40/44) Large Process Models: Business Process Management in the Age of Generative AI (Timotheus Kampik et al., 2023)

{{<citation>}}

Timotheus Kampik, Christian Warmuth, Adrian Rebmann, Ron Agam, Lukas N. P. Egger, Andreas Gerber, Johannes Hoffart, Jonas Kolk, Philipp Herzig, Gero Decker, Han van der Aa, Artem Polyvyanyy, Stefanie Rinderle-Ma, Ingo Weber, Matthias Weidlich. (2023)  
**Large Process Models: Business Process Management in the Age of Generative AI**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI, Generative AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.00900v1)  

---


**ABSTRACT**  
The continued success of Large Language Models (LLMs) and other generative artificial intelligence approaches highlights the advantages that large information corpora can have over rigidly defined symbolic models, but also serves as a proof-point of the challenges that purely statistics-based approaches have in terms of safety and trustworthiness. As a framework for contextualizing the potential, as well as the limitations of LLMs and other foundation model-based technologies, we propose the concept of a Large Process Model (LPM) that combines the correlation power of LLMs with the analytical precision and reliability of knowledge-based systems and automated reasoning approaches. LPMs are envisioned to directly utilize the wealth of process management experience that experts have accumulated, as well as process performance data of organizations with diverse characteristics, e.g., regarding size, region, or industry. In this vision, the proposed LPM would allow organizations to receive context-specific (tailored) process and other business models, analytical deep-dives, and improvement recommendations. As such, they would allow to substantially decrease the time and effort required for business transformation, while also allowing for deeper, more impactful, and more actionable insights than previously possible. We argue that implementing an LPM is feasible, but also highlight limitations and research challenges that need to be solved to implement particular aspects of the LPM vision.

{{</citation>}}


### (41/44) DeepScaler: Holistic Autoscaling for Microservices Based on Spatiotemporal GNN with Adaptive Graph Learning (Chunyang Meng et al., 2023)

{{<citation>}}

Chunyang Meng, Shijie Song, Haogang Tong, Maolin Pan, Yang Yu. (2023)  
**DeepScaler: Holistic Autoscaling for Microservices Based on Spatiotemporal GNN with Adaptive Graph Learning**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2309.00859v1)  

---


**ABSTRACT**  
Autoscaling functions provide the foundation for achieving elasticity in the modern cloud computing paradigm. It enables dynamic provisioning or de-provisioning resources for cloud software services and applications without human intervention to adapt to workload fluctuations. However, autoscaling microservice is challenging due to various factors. In particular, complex, time-varying service dependencies are difficult to quantify accurately and can lead to cascading effects when allocating resources. This paper presents DeepScaler, a deep learning-based holistic autoscaling approach for microservices that focus on coping with service dependencies to optimize service-level agreements (SLA) assurance and cost efficiency. DeepScaler employs (i) an expectation-maximization-based learning method to adaptively generate affinity matrices revealing service dependencies and (ii) an attention-based graph convolutional network to extract spatio-temporal features of microservices by aggregating neighbors' information of graph-structural data. Thus DeepScaler can capture more potential service dependencies and accurately estimate the resource requirements of all services under dynamic workloads. It allows DeepScaler to reconfigure the resources of the interacting services simultaneously in one resource provisioning operation, avoiding the cascading effect caused by service dependencies. Experimental results demonstrate that our method implements a more effective autoscaling mechanism for microservice that not only allocates resources accurately but also adapts to dependencies changes, significantly reducing SLA violations by an average of 41% at lower costs.

{{</citation>}}


## cs.SI (1)



### (42/44) Trustworthiness-Driven Graph Convolutional Networks for Signed Network Embedding (Min-Jeong Kim et al., 2023)

{{<citation>}}

Min-Jeong Kim, Yeon-Chang Lee, David Y. Kang, Sang-Wook Kim. (2023)  
**Trustworthiness-Driven Graph Convolutional Networks for Signed Network Embedding**  

---
Primary Category: cs.SI  
Categories: 68T01, H-4-0; I-2-0, cs-LG, cs-SI, cs.SI  
Keywords: Embedding, Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2309.00816v1)  

---


**ABSTRACT**  
The problem of representing nodes in a signed network as low-dimensional vectors, known as signed network embedding (SNE), has garnered considerable attention in recent years. While several SNE methods based on graph convolutional networks (GCN) have been proposed for this problem, we point out that they significantly rely on the assumption that the decades-old balance theory always holds in the real-world. To address this limitation, we propose a novel GCN-based SNE approach, named as TrustSGCN, which corrects for incorrect embedding propagation in GCN by utilizing the trustworthiness on edge signs for high-order relationships inferred by the balance theory. The proposed approach consists of three modules: (M1) generation of each node's extended ego-network; (M2) measurement of trustworthiness on edge signs; and (M3) trustworthiness-aware propagation of embeddings. Furthermore, TrustSGCN learns the node embeddings by leveraging two well-known societal theories, i.e., balance and status. The experiments on four real-world signed network datasets demonstrate that TrustSGCN consistently outperforms five state-of-the-art GCN-based SNE methods. The code is available at https://github.com/kmj0792/TrustSGCN.

{{</citation>}}


## econ.EM (1)



### (43/44) Fairness Implications of Heterogeneous Treatment Effect Estimation with Machine Learning Methods in Policy-making (Patrick Rehill et al., 2023)

{{<citation>}}

Patrick Rehill, Nicholas Biddle. (2023)  
**Fairness Implications of Heterogeneous Treatment Effect Estimation with Machine Learning Methods in Policy-making**  

---
Primary Category: econ.EM  
Categories: cs-LG, econ-EM, econ.EM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.00805v1)  

---


**ABSTRACT**  
Causal machine learning methods which flexibly generate heterogeneous treatment effect estimates could be very useful tools for governments trying to make and implement policy. However, as the critical artificial intelligence literature has shown, governments must be very careful of unintended consequences when using machine learning models. One way to try and protect against unintended bad outcomes is with AI Fairness methods which seek to create machine learning models where sensitive variables like race or gender do not influence outcomes. In this paper we argue that standard AI Fairness approaches developed for predictive machine learning are not suitable for all causal machine learning applications because causal machine learning generally (at least so far) uses modelling to inform a human who is the ultimate decision-maker while AI Fairness approaches assume a model that is making decisions directly. We define these scenarios as indirect and direct decision-making respectively and suggest that policy-making is best seen as a joint decision where the causal machine learning model usually only has indirect power. We lay out a definition of fairness for this scenario - a model that provides the information a decision-maker needs to accurately make a value judgement about just policy outcomes - and argue that the complexity of causal machine learning models can make this difficult to achieve. The solution here is not traditional AI Fairness adjustments, but careful modelling and awareness of some of the decision-making biases that these methods might encourage which we describe.

{{</citation>}}


## cs.AR (1)



### (44/44) Accelerating LSTM-based High-Rate Dynamic System Models (Ehsan Kabir et al., 2023)

{{<citation>}}

Ehsan Kabir, Daniel Coble, Joud N. Satme, Austin R. J. Downey, Jason D. Bakos, David Andrews, Miaoqing Huang. (2023)  
**Accelerating LSTM-based High-Rate Dynamic System Models**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.00801v1)  

---


**ABSTRACT**  
In this paper, we evaluate the use of a trained Long Short-Term Memory (LSTM) network as a surrogate for a Euler-Bernoulli beam model, and then we describe and characterize an FPGA-based deployment of the model for use in real-time structural health monitoring applications. The focus of our efforts is the DROPBEAR (Dynamic Reproduction of Projectiles in Ballistic Environments for Advanced Research) dataset, which was generated as a benchmark for the study of real-time structural modeling applications. The purpose of DROPBEAR is to evaluate models that take vibration data as input and give the initial conditions of the cantilever beam on which the measurements were taken as output. DROPBEAR is meant to serve an exemplar for emerging high-rate "active structures" that can be actively controlled with feedback latencies of less than one microsecond. Although the Euler-Bernoulli beam model is a well-known solution to this modeling problem, its computational cost is prohibitive for the time scales of interest. It has been previously shown that a properly structured LSTM network can achieve comparable accuracy with less workload, but achieving sub-microsecond model latency remains a challenge. Our approach is to deploy the LSTM optimized specifically for latency on FPGA. We designed the model using both high-level synthesis (HLS) and hardware description language (HDL). The lowest latency of 1.42 $\mu$S and the highest throughput of 7.87 Gops/s were achieved on Alveo U55C platform for HDL design.

{{</citation>}}
