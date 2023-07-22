---
draft: false
title: "arXiv @ 2023.07.11"
date: 2023-07-11
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.11"
    identifier: arxiv_20230711
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.HC (1)](#cshc-1)
- [cs.CL (7)](#cscl-7)
- [cs.LG (8)](#cslg-8)
- [cs.AI (1)](#csai-1)
- [cs.CV (19)](#cscv-19)
- [cs.RO (1)](#csro-1)
- [cs.CR (2)](#cscr-2)
- [eess.SP (1)](#eesssp-1)
- [cs.SE (1)](#csse-1)
- [cs.CY (1)](#cscy-1)
- [cs.IR (1)](#csir-1)

## cs.HC (1)



### (1/43) Shaping the Emerging Norms of Using Large Language Models in Social Computing Research (Hong Shen et al., 2023)

{{<citation>}}

Hong Shen, Tianshi Li, Toby Jia-Jun Li, Joon Sung Park, Diyi Yang. (2023)  
**Shaping the Emerging Norms of Using Large Language Models in Social Computing Research**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.04280v1)  

---


**ABSTRACT**  
The emergence of Large Language Models (LLMs) has brought both excitement and concerns to social computing research. On the one hand, LLMs offer unprecedented capabilities in analyzing vast amounts of textual data and generating human-like responses, enabling researchers to delve into complex social phenomena. On the other hand, concerns are emerging regarding the validity, privacy, and ethics of the research when LLMs are involved. This SIG aims at offering an open space for social computing researchers who are interested in understanding the impacts of LLMs to discuss their current practices, perspectives, challenges when engaging with LLMs in their everyday work and collectively shaping the emerging norms of using LLMs in social computing research.

{{</citation>}}


## cs.CL (7)



### (2/43) Automated Essay Scoring in Argumentative Writing: DeBERTeachingAssistant (Yann Hicke et al., 2023)

{{<citation>}}

Yann Hicke, Tonghua Tian, Karan Jha, Choong Hee Kim. (2023)  
**Automated Essay Scoring in Argumentative Writing: DeBERTeachingAssistant**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NLP  
[Paper Link](http://arxiv.org/abs/2307.04276v1)  

---


**ABSTRACT**  
Automated Essay scoring has been explored as a research and industry problem for over 50 years. It has drawn a lot of attention from the NLP community because of its clear educational value as a research area that can engender the creation of valuable time-saving tools for educators around the world. Yet, these tools are generally focused on detecting good grammar, spelling mistakes, and organization quality but tend to fail at incorporating persuasiveness features in their final assessment. The responsibility to give actionable feedback to the student to improve the strength of their arguments is left solely on the teacher's shoulders. In this work, we present a transformer-based architecture capable of achieving above-human accuracy in annotating argumentative writing discourse elements for their persuasiveness quality and we expand on planned future work investigating the explainability of our model so that actionable feedback can be offered to the student and thus potentially enable a partnership between the teacher's advice and the machine's advice.

{{</citation>}}


### (3/43) Augmenters at SemEval-2023 Task 1: Enhancing CLIP in Handling Compositionality and Ambiguity for Zero-Shot Visual WSD through Prompt Augmentation and Text-To-Image Diffusion (Jie S. Li et al., 2023)

{{<citation>}}

Jie S. Li, Yow-Ting Shiue, Yong-Siang Shih, Jonas Geiping. (2023)  
**Augmenters at SemEval-2023 Task 1: Enhancing CLIP in Handling Compositionality and Ambiguity for Zero-Shot Visual WSD through Prompt Augmentation and Text-To-Image Diffusion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, Word Sense Disambiguation, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.05564v1)  

---


**ABSTRACT**  
This paper describes our zero-shot approaches for the Visual Word Sense Disambiguation (VWSD) Task in English. Our preliminary study shows that the simple approach of matching candidate images with the phrase using CLIP suffers from the many-to-many nature of image-text pairs. We find that the CLIP text encoder may have limited abilities in capturing the compositionality in natural language. Conversely, the descriptive focus of the phrase varies from instance to instance. We address these issues in our two systems, Augment-CLIP and Stable Diffusion Sampling (SD Sampling). Augment-CLIP augments the text prompt by generating sentences that contain the context phrase with the help of large language models (LLMs). We further explore CLIP models in other languages, as the an ambiguous word may be translated into an unambiguous one in the other language. SD Sampling uses text-to-image Stable Diffusion to generate multiple images from the given phrase, increasing the likelihood that a subset of images match the one that paired with the text.

{{</citation>}}


### (4/43) Assessing the efficacy of large language models in generating accurate teacher responses (Yann Hicke et al., 2023)

{{<citation>}}

Yann Hicke, Abhishek Masand, Wentao Guo, Tushaar Gangavarapu. (2023)  
**Assessing the efficacy of large language models in generating accurate teacher responses**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, Dialog, GPT, GPT-4, NLP, T5  
[Paper Link](http://arxiv.org/abs/2307.04274v1)  

---


**ABSTRACT**  
(Tack et al., 2023) organized the shared task hosted by the 18th Workshop on Innovative Use of NLP for Building Educational Applications on generation of teacher language in educational dialogues. Following the structure of the shared task, in this study, we attempt to assess the generative abilities of large language models in providing informative and helpful insights to students, thereby simulating the role of a knowledgeable teacher. To this end, we present an extensive evaluation of several benchmarking generative models, including GPT-4 (few-shot, in-context learning), fine-tuned GPT-2, and fine-tuned DialoGPT. Additionally, to optimize for pedagogical quality, we fine-tuned the Flan-T5 model using reinforcement learning. Our experimental findings on the Teacher-Student Chatroom Corpus subset indicate the efficacy of GPT-4 over other fine-tuned models, measured using BERTScore and DialogRPT.   We hypothesize that several dataset characteristics, including sampling, representativeness, and dialog completeness, pose significant challenges to fine-tuning, thus contributing to the poor generalizability of the fine-tuned models. Finally, we note the need for these generative models to be evaluated with a metric that relies not only on dialog coherence and matched language modeling distribution but also on the model's ability to showcase pedagogical skills.

{{</citation>}}


### (5/43) ChatGPT in the Age of Generative AI and Large Language Models: A Concise Survey (Salman Mohamadi et al., 2023)

{{<citation>}}

Salman Mohamadi, Ghulam Mujtaba, Ngan Le, Gianfranco Doretto, Donald A. Adjeroh. (2023)  
**ChatGPT in the Age of Generative AI and Large Language Models: A Concise Survey**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, ChatGPT, GPT, Generative AI, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2307.04251v2)  

---


**ABSTRACT**  
ChatGPT is a large language model (LLM) created by OpenAI that has been carefully trained on a large amount of data. It has revolutionized the field of natural language processing (NLP) and has pushed the boundaries of LLM capabilities. ChatGPT has played a pivotal role in enabling widespread public interaction with generative artificial intelligence (GAI) on a large scale. It has also sparked research interest in developing similar technologies and investigating their applications and implications. In this paper, our primary goal is to provide a concise survey on the current lines of research on ChatGPT and its evolution. We considered both the glass box and black box views of ChatGPT, encompassing the components and foundational elements of the technology, as well as its applications, impacts, and implications. The glass box approach focuses on understanding the inner workings of the technology, and the black box approach embraces it as a complex system, and thus examines its inputs, outputs, and effects. This paves the way for a comprehensive exploration of the technology and provides a road map for further research and experimentation. We also lay out essential foundational literature on LLMs and GAI in general and their connection with ChatGPT. This overview sheds light on existing and missing research lines in the emerging field of LLMs, benefiting both public users and developers. Furthermore, the paper delves into the broad spectrum of applications and significant concerns in fields such as education, research, healthcare, finance, etc.

{{</citation>}}


### (6/43) Automatic Coding at Scale: Design and Deployment of a Nationwide System for Normalizing Referrals in the Chilean Public Healthcare System (Fabián Villena et al., 2023)

{{<citation>}}

Fabián Villena, Matías Rojas, Felipe Arias, Jorge Pacheco, Paulina Vera, Jocelyn Dunstan. (2023)  
**Automatic Coding at Scale: Design and Deployment of a Nationwide System for Normalizing Referrals in the Chilean Public Healthcare System**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2307.05560v1)  

---


**ABSTRACT**  
The disease coding task involves assigning a unique identifier from a controlled vocabulary to each disease mentioned in a clinical document. This task is relevant since it allows information extraction from unstructured data to perform, for example, epidemiological studies about the incidence and prevalence of diseases in a determined context. However, the manual coding process is subject to errors as it requires medical personnel to be competent in coding rules and terminology. In addition, this process consumes a lot of time and energy, which could be allocated to more clinically relevant tasks. These difficulties can be addressed by developing computational systems that automatically assign codes to diseases. In this way, we propose a two-step system for automatically coding diseases in referrals from the Chilean public healthcare system. Specifically, our model uses a state-of-the-art NER model for recognizing disease mentions and a search engine system based on Elasticsearch for assigning the most relevant codes associated with these disease mentions. The system's performance was evaluated on referrals manually coded by clinical experts. Our system obtained a MAP score of 0.63 for the subcategory level and 0.83 for the category level, close to the best-performing models in the literature. This system could be a support tool for health professionals, optimizing the coding and management process. Finally, to guarantee reproducibility, we publicly release the code of our models and experiments.

{{</citation>}}


### (7/43) Can Generative Large Language Models Perform ASR Error Correction? (Rao Ma et al., 2023)

{{<citation>}}

Rao Ma, Mengjie Qian, Potsawee Manakul, Mark Gales, Kate Knill. (2023)  
**Can Generative Large Language Models Perform ASR Error Correction?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.04172v1)  

---


**ABSTRACT**  
ASR error correction continues to serve as an important part of post-processing for speech recognition systems. Traditionally, these models are trained with supervised training using the decoding results of the underlying ASR system and the reference text. This approach is computationally intensive and the model needs to be re-trained when switching the underlying ASR model. Recent years have seen the development of large language models and their ability to perform natural language processing tasks in a zero-shot manner. In this paper, we take ChatGPT as an example to examine its ability to perform ASR error correction in the zero-shot or 1-shot settings. We use the ASR N-best list as model input and propose unconstrained error correction and N-best constrained error correction methods. Results on a Conformer-Transducer model and the pre-trained Whisper model show that we can largely improve the ASR system performance with error correction using the powerful ChatGPT model.

{{</citation>}}


### (8/43) DebateKG: Automatic Policy Debate Case Creation with Semantic Knowledge Graphs (Allen Roush, 2023)

{{<citation>}}

Allen Roush. (2023)  
**DebateKG: Automatic Policy Debate Case Creation with Semantic Knowledge Graphs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Knowledge Graph, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.04090v1)  

---


**ABSTRACT**  
Recent work within the Argument Mining community has shown the applicability of Natural Language Processing systems for solving problems found within competitive debate. One of the most important tasks within competitive debate is for debaters to create high quality debate cases. We show that effective debate cases can be constructed using constrained shortest path traversals on Argumentative Semantic Knowledge Graphs. We study this potential in the context of a type of American Competitive Debate, called Policy Debate, which already has a large scale dataset targeting it called DebateSum. We significantly improve upon DebateSum by introducing 53180 new examples, as well as further useful metadata for every example, to the dataset. We leverage the txtai semantic search and knowledge graph toolchain to produce and contribute 9 semantic knowledge graphs built on this dataset. We create a unique method for evaluating which knowledge graphs are better in the context of producing policy debate cases. A demo which automatically generates debate cases, along with all other code and the Knowledge Graphs, are open-sourced and made available to the public here: https://github.com/Hellisotherpeople/DebateKG

{{</citation>}}


## cs.LG (8)



### (9/43) MentalHealthAI: Utilizing Personal Health Device Data to Optimize Psychiatry Treatment (Manan Shukla et al., 2023)

{{<citation>}}

Manan Shukla, Oshani Seneviratne. (2023)  
**MentalHealthAI: Utilizing Personal Health Device Data to Optimize Psychiatry Treatment**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.04777v1)  

---


**ABSTRACT**  
Mental health disorders remain a significant challenge in modern healthcare, with diagnosis and treatment often relying on subjective patient descriptions and past medical history. To address this issue, we propose a personalized mental health tracking and mood prediction system that utilizes patient physiological data collected through personal health devices. Our system leverages a decentralized learning mechanism that combines transfer and federated machine learning concepts using smart contracts, allowing data to remain on users' devices and enabling effective tracking of mental health conditions for psychiatric treatment and management in a privacy-aware and accountable manner. We evaluate our model using a popular mental health dataset that demonstrates promising results. By utilizing connected health systems and machine learning models, our approach offers a novel solution to the challenge of providing psychiatrists with further insight into their patients' mental health outside of traditional office visits.

{{</citation>}}


### (10/43) Investigating the Edge of Stability Phenomenon in Reinforcement Learning (Rares Iordan et al., 2023)

{{<citation>}}

Rares Iordan, Marc Peter Deisenroth, Mihaela Rosca. (2023)  
**Investigating the Edge of Stability Phenomenon in Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.04210v1)  

---


**ABSTRACT**  
Recent progress has been made in understanding optimisation dynamics in neural networks trained with full-batch gradient descent with momentum with the uncovering of the edge of stability phenomenon in supervised learning. The edge of stability phenomenon occurs as the leading eigenvalue of the Hessian reaches the divergence threshold of the underlying optimisation algorithm for a quadratic loss, after which it starts oscillating around the threshold, and the loss starts to exhibit local instability but decreases over long time frames. In this work, we explore the edge of stability phenomenon in reinforcement learning (RL), specifically off-policy Q-learning algorithms across a variety of data regimes, from offline to online RL. Our experiments reveal that, despite significant differences to supervised learning, such as non-stationarity of the data distribution and the use of bootstrapping, the edge of stability phenomenon can be present in off-policy deep RL. Unlike supervised learning, however, we observe strong differences depending on the underlying loss, with DQN -- using a Huber loss -- showing a strong edge of stability effect that we do not observe with C51 -- using a cross entropy loss. Our results suggest that, while neural network structure can lead to optimisation dynamics that transfer between problem domains, certain aspects of deep RL optimisation can differentiate it from domains such as supervised learning.

{{</citation>}}


### (11/43) On the Challenges of Deploying Privacy-Preserving Synthetic Data in the Enterprise (Lauren Arthur et al., 2023)

{{<citation>}}

Lauren Arthur, Jason Costello, Jonathan Hardy, Will O'Brien, James Rea, Gareth Rees, Georgi Ganev. (2023)  
**On the Challenges of Deploying Privacy-Preserving Synthetic Data in the Enterprise**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-CY, cs-LG, cs.LG  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2307.04208v1)  

---


**ABSTRACT**  
Generative AI technologies are gaining unprecedented popularity, causing a mix of excitement and apprehension through their remarkable capabilities. In this paper, we study the challenges associated with deploying synthetic data, a subfield of Generative AI. Our focus centers on enterprise deployment, with an emphasis on privacy concerns caused by the vast amount of personal and highly sensitive data. We identify 40+ challenges and systematize them into five main groups -- i) generation, ii) infrastructure & architecture, iii) governance, iv) compliance & regulation, and v) adoption. Additionally, we discuss a strategic and systematic approach that enterprises can employ to effectively address the challenges and achieve their goals by establishing trust in the implemented solutions.

{{</citation>}}


### (12/43) Graph Neural Network-enabled Terahertz-based Flow-guided Nanoscale Localization (Gerard Calvo Bartra et al., 2023)

{{<citation>}}

Gerard Calvo Bartra, Filip Lemic, Sergi Abadal, Xavier Costa Perez. (2023)  
**Graph Neural Network-enabled Terahertz-based Flow-guided Nanoscale Localization**  

---
Primary Category: cs.LG  
Categories: cs-ET, cs-LG, cs-NI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.05551v1)  

---


**ABSTRACT**  
Scientific advancements in nanotechnology and advanced materials are paving the way toward nanoscale devices for in-body precision medicine; comprising integrated sensing, computing, communication, data and energy storage capabilities. In the human cardiovascular system, such devices are envisioned to be passively flowing and continuously sensing for detecting events of diagnostic interest. The diagnostic value of detecting such events can be enhanced by assigning to them their physical locations (e.g., body region), which is the main proposition of flow-guided localization. Current flow-guided localization approaches suffer from low localization accuracy and they are by-design unable to localize events within the entire cardiovascular system. Toward addressing this issue, we propose the utilization of Graph Neural Networks (GNNs) for this purpose, and demonstrate localization accuracy and coverage enhancements of our proposal over the existing State of the Art (SotA) approaches. Based on our evaluation, we provide several design guidelines for GNN-enabled flow-guided localization.

{{</citation>}}


### (13/43) FILM: How can Few-Shot Image Classification Benefit from Pre-Trained Language Models? (Zihao Jiang et al., 2023)

{{<citation>}}

Zihao Jiang, Yunkai Dang, Dong Pang, Huishuai Zhang, Weiran Huang. (2023)  
**FILM: How can Few-Shot Image Classification Benefit from Pre-Trained Language Models?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs-MM, cs.LG  
Keywords: Few-Shot, Image Classification, Language Model  
[Paper Link](http://arxiv.org/abs/2307.04114v1)  

---


**ABSTRACT**  
Few-shot learning aims to train models that can be generalized to novel classes with only a few samples. Recently, a line of works are proposed to enhance few-shot learning with accessible semantic information from class names. However, these works focus on improving existing modules such as visual prototypes and feature extractors of the standard few-shot learning framework. This limits the full potential use of semantic information. In this paper, we propose a novel few-shot learning framework that uses pre-trained language models based on contrastive learning. To address the challenge of alignment between visual features and textual embeddings obtained from text-based pre-trained language model, we carefully design the textual branch of our framework and introduce a metric module to generalize the cosine similarity. For better transferability, we let the metric module adapt to different few-shot tasks and adopt MAML to train the model via bi-level optimization. Moreover, we conduct extensive experiments on multiple benchmarks to demonstrate the effectiveness of our method.

{{</citation>}}


### (14/43) Towards Assumption-free Bias Mitigation (Chia-Yuan Chang et al., 2023)

{{<citation>}}

Chia-Yuan Chang, Yu-Neng Chuang, Kwei-Herng Lai, Xiaotian Han, Xia Hu, Na Zou. (2023)  
**Towards Assumption-free Bias Mitigation**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.04105v1)  

---


**ABSTRACT**  
Despite the impressive prediction ability, machine learning models show discrimination towards certain demographics and suffer from unfair prediction behaviors. To alleviate the discrimination, extensive studies focus on eliminating the unequal distribution of sensitive attributes via multiple approaches. However, due to privacy concerns, sensitive attributes are often either unavailable or missing in real-world scenarios. Therefore, several existing works alleviate the bias without sensitive attributes. Those studies face challenges, either in inaccurate predictions of sensitive attributes or the need to mitigate unequal distribution of manually defined non-sensitive attributes related to bias. The latter requires strong assumptions about the correlation between sensitive and non-sensitive attributes. As data distribution and task goals vary, the strong assumption on non-sensitive attributes may not be valid and require domain expertise. In this work, we propose an assumption-free framework to detect the related attributes automatically by modeling feature interaction for bias mitigation. The proposed framework aims to mitigate the unfair impact of identified biased feature interactions. Experimental results on four real-world datasets demonstrate that our proposed framework can significantly alleviate unfair prediction behaviors by considering biased feature interactions.

{{</citation>}}


### (15/43) Restricted Generative Projection for One-Class Classification and Anomaly Detection (Feng Xiao et al., 2023)

{{<citation>}}

Feng Xiao, Ruoyu Sun, Jicong Fan. (2023)  
**Restricted Generative Projection for One-Class Classification and Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.04097v1)  

---


**ABSTRACT**  
We present a simple framework for one-class classification and anomaly detection. The core idea is to learn a mapping to transform the unknown distribution of training (normal) data to a known target distribution. Crucially, the target distribution should be sufficiently simple, compact, and informative. The simplicity is to ensure that we can sample from the distribution easily, the compactness is to ensure that the decision boundary between normal data and abnormal data is clear and reliable, and the informativeness is to ensure that the transformed data preserve the important information of the original data. Therefore, we propose to use truncated Gaussian, uniform in hypersphere, uniform on hypersphere, or uniform between hyperspheres, as the target distribution. We then minimize the distance between the transformed data distribution and the target distribution while keeping the reconstruction error for the original data small enough. Comparative studies on multiple benchmark datasets verify the effectiveness of our methods in comparison to baselines.

{{</citation>}}


### (16/43) Multi-Head Attention Mechanism Learning for Cancer New Subtypes and Treatment Based on Cancer Multi-Omics Data (Liangrui Pan et al., 2023)

{{<citation>}}

Liangrui Pan, Dazhen Liu, Yutao Dou, Lian Wang, Zhichao Feng, Pengfei Rong, Liwen Xu, Shaoliang Peng. (2023)  
**Multi-Head Attention Mechanism Learning for Cancer New Subtypes and Treatment Based on Cancer Multi-Omics Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.04075v1)  

---


**ABSTRACT**  
Due to the high heterogeneity and clinical characteristics of cancer, there are significant differences in multi-omics data and clinical features among subtypes of different cancers. Therefore, the identification and discovery of cancer subtypes are crucial for the diagnosis, treatment, and prognosis of cancer. In this study, we proposed a generalization framework based on attention mechanisms for unsupervised contrastive learning (AMUCL) to analyze cancer multi-omics data for the identification and characterization of cancer subtypes. AMUCL framework includes a unsupervised multi-head attention mechanism, which deeply extracts multi-omics data features. Importantly, a decoupled contrastive learning model (DMACL) based on a multi-head attention mechanism is proposed to learn multi-omics data features and clusters and identify new cancer subtypes. This unsupervised contrastive learning method clusters subtypes by calculating the similarity between samples in the feature space and sample space of multi-omics data. Compared to 11 other deep learning models, the DMACL model achieved a C-index of 0.002, a Silhouette score of 0.801, and a Davies Bouldin Score of 0.38 on a single-cell multi-omics dataset. On a cancer multi-omics dataset, the DMACL model obtained a C-index of 0.016, a Silhouette score of 0.688, and a Davies Bouldin Score of 0.46, and obtained the most reliable cancer subtype clustering results for each type of cancer. Finally, we used the DMACL model in the AMUCL framework to reveal six cancer subtypes of AML. By analyzing the GO functional enrichment, subtype-specific biological functions, and GSEA of AML, we further enhanced the interpretability of cancer subtype analysis based on the generalizable AMUCL framework.

{{</citation>}}


## cs.AI (1)



### (17/43) The Future of Fundamental Science Led by Generative Closed-Loop Artificial Intelligence (Hector Zenil et al., 2023)

{{<citation>}}

Hector Zenil, Jesper Tegnér, Felipe S. Abrahão, Alexander Lavin, Vipin Kumar, Jeremy G. Frey, Adrian Weller, Larisa Soldatova, Alan R. Bundy, Nicholas R. Jennings, Koichi Takahashi, Lawrence Hunter, Saso Dzeroski, Andrew Briggs, Frederick D. Gregory, Carla P. Gomes, Christopher K. I. Williams, Jon Rowe, James Evans, Hiroaki Kitano, Joshua B. Tenenbaum, Ross King. (2023)  
**The Future of Fundamental Science Led by Generative Closed-Loop Artificial Intelligence**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, Generative AI, Language Model  
[Paper Link](http://arxiv.org/abs/2307.07522v1)  

---


**ABSTRACT**  
Recent advances in machine learning and AI, including Generative AI and LLMs, are disrupting technological innovation, product development, and society as a whole. AI's contribution to technology can come from multiple approaches that require access to large training data sets and clear performance evaluation criteria, ranging from pattern recognition and classification to generative models. Yet, AI has contributed less to fundamental science in part because large data sets of high-quality data for scientific practice and model discovery are more difficult to access. Generative AI, in general, and Large Language Models in particular, may represent an opportunity to augment and accelerate the scientific discovery of fundamental deep science with quantitative models. Here we explore and investigate aspects of an AI-driven, automated, closed-loop approach to scientific discovery, including self-driven hypothesis generation and open-ended autonomous exploration of the hypothesis space. Integrating AI-driven automation into the practice of science would mitigate current problems, including the replication of findings, systematic production of data, and ultimately democratisation of the scientific process. Realising these possibilities requires a vision for augmented AI coupled with a diversity of AI approaches able to deal with fundamental aspects of causality analysis and model discovery while enabling unbiased search across the space of putative explanations. These advances hold the promise to unleash AI's potential for searching and discovering the fundamental structure of our world beyond what human scientists have been able to achieve. Such a vision would push the boundaries of new fundamental science rather than automatize current workflows and instead open doors for technological innovation to tackle some of the greatest challenges facing humanity today.

{{</citation>}}


## cs.CV (19)



### (18/43) A Novel Pipeline for Improving Optical Character Recognition through Post-processing Using Natural Language Processing (Aishik Rakshit et al., 2023)

{{<citation>}}

Aishik Rakshit, Samyak Mehta, Anirban Dasgupta. (2023)  
**A Novel Pipeline for Improving Optical Character Recognition through Post-processing Using Natural Language Processing**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: NLP, Natural Language Processing, OCR  
[Paper Link](http://arxiv.org/abs/2307.04245v1)  

---


**ABSTRACT**  
Optical Character Recognition (OCR) technology finds applications in digitizing books and unstructured documents, along with applications in other domains such as mobility statistics, law enforcement, traffic, security systems, etc. The state-of-the-art methods work well with the OCR with printed text on license plates, shop names, etc. However, applications such as printed textbooks and handwritten texts have limited accuracy with existing techniques. The reason may be attributed to similar-looking characters and variations in handwritten characters. Since these issues are challenging to address with OCR technologies exclusively, we propose a post-processing approach using Natural Language Processing (NLP) tools. This work presents an end-to-end pipeline that first performs OCR on the handwritten or printed text and then improves its accuracy using NLP.

{{</citation>}}


### (19/43) TransPose: A Transformer-based 6D Object Pose Estimation Network with Depth Refinement (Mahmoud Abdulsalam et al., 2023)

{{<citation>}}

Mahmoud Abdulsalam, Nabil Aouf. (2023)  
**TransPose: A Transformer-based 6D Object Pose Estimation Network with Depth Refinement**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.05561v1)  

---


**ABSTRACT**  
As demand for robotics manipulation application increases, accurate vision-based 6D pose estimation becomes essential for autonomous operations. Convolutional Neural Networks (CNNs) based approaches for pose estimation have been previously introduced. However, the quest for better performance still persists especially for accurate robotics manipulation. This quest extends to the Agri-robotics domain. In this paper, we propose TransPose, an improved Transformer-based 6D pose estimation with a depth refinement module. The architecture takes in only an RGB image as input with no additional supplementing modalities such as depth or thermal images. The architecture encompasses an innovative lighter depth estimation network that estimates depth from an RGB image using feature pyramid with an up-sampling method. A transformer-based detection network with additional prediction heads is proposed to directly regress the object's centre and predict the 6D pose of the target. A novel depth refinement module is then used alongside the predicted centers, 6D poses and depth patches to refine the accuracy of the estimated 6D pose. We extensively compared our results with other state-of-the-art methods and analysed our results for fruit-picking applications. The results we achieved show that our proposed technique outperforms the other methods available in the literature.

{{</citation>}}


### (20/43) Mx2M: Masked Cross-Modality Modeling in Domain Adaptation for 3D Semantic Segmentation (Boxiang Zhang et al., 2023)

{{<citation>}}

Boxiang Zhang, Zunran Wang, Yonggen Ling, Yuanyuan Guan, Shenghao Zhang, Wenhui Li. (2023)  
**Mx2M: Masked Cross-Modality Modeling in Domain Adaptation for 3D Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.04231v1)  

---


**ABSTRACT**  
Existing methods of cross-modal domain adaptation for 3D semantic segmentation predict results only via 2D-3D complementarity that is obtained by cross-modal feature matching. However, as lacking supervision in the target domain, the complementarity is not always reliable. The results are not ideal when the domain gap is large. To solve the problem of lacking supervision, we introduce masked modeling into this task and propose a method Mx2M, which utilizes masked cross-modality modeling to reduce the large domain gap. Our Mx2M contains two components. One is the core solution, cross-modal removal and prediction (xMRP), which makes the Mx2M adapt to various scenarios and provides cross-modal self-supervision. The other is a new way of cross-modal feature matching, the dynamic cross-modal filter (DxMF) that ensures the whole method dynamically uses more suitable 2D-3D complementarity. Evaluation of the Mx2M on three DA scenarios, including Day/Night, USA/Singapore, and A2D2/SemanticKITTI, brings large improvements over previous methods on many metrics.

{{</citation>}}


### (21/43) SAS Video-QA: Self-Adaptive Sampling for Efficient Video Question-Answering (Wei Han et al., 2023)

{{<citation>}}

Wei Han, Hui Chen, Min-Yen Kan, Soujanya Poria. (2023)  
**SAS Video-QA: Self-Adaptive Sampling for Efficient Video Question-Answering**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-MM, cs.CV  
Keywords: QA, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.04192v1)  

---


**ABSTRACT**  
Video question--answering is a fundamental task in the field of video understanding. Although current vision--language models (VLMs) equipped with Video Transformers have enabled temporal modeling and yielded superior results, they are at the cost of huge computational power and thus too expensive to deploy in real-time application scenarios. An economical workaround only samples a small portion of frames to represent the main content of that video and tune an image--text model on these sampled frames. Recent video understanding models usually randomly sample a set of frames or clips, regardless of internal correlations between their visual contents, nor their relevance to the problem. We argue that such kinds of aimless sampling may omit the key frames from which the correct answer can be deduced, and the situation gets worse when the sampling sparsity increases, which always happens as the video lengths increase. To mitigate this issue, we propose two frame sampling strategies, namely the most domain frames (MDF) and most implied frames (MIF), to maximally preserve those frames that are most likely vital to the given questions. MDF passively minimizes the risk of key frame omission in a bootstrap manner, while MIS actively searches key frames customized for each video--question pair with the assistance of auxiliary models. The experimental results on three public datasets from three advanced VLMs (CLIP, GIT and All-in-one) demonstrate that our proposed strategies can boost the performance for image--text pretrained models. The source codes pertaining to the method proposed in this paper are publicly available at https://github.com/declare-lab/sas-vqa.

{{</citation>}}


### (22/43) Histopathology Whole Slide Image Analysis with Heterogeneous Graph Representation Learning (Tsai Hor Chan et al., 2023)

{{<citation>}}

Tsai Hor Chan, Fernando Julio Cendra, Lan Ma, Guosheng Yin, Lequan Yu. (2023)  
**Histopathology Whole Slide Image Analysis with Heterogeneous Graph Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, GNN, Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.04189v1)  

---


**ABSTRACT**  
Graph-based methods have been extensively applied to whole-slide histopathology image (WSI) analysis due to the advantage of modeling the spatial relationships among different entities. However, most of the existing methods focus on modeling WSIs with homogeneous graphs (e.g., with homogeneous node type). Despite their successes, these works are incapable of mining the complex structural relations between biological entities (e.g., the diverse interaction among different cell types) in the WSI. We propose a novel heterogeneous graph-based framework to leverage the inter-relationships among different types of nuclei for WSI analysis. Specifically, we formulate the WSI as a heterogeneous graph with "nucleus-type" attribute to each node and a semantic similarity attribute to each edge. We then present a new heterogeneous-graph edge attribute transformer (HEAT) to take advantage of the edge and node heterogeneity during massage aggregating. Further, we design a new pseudo-label-based semantic-consistent pooling mechanism to obtain graph-level features, which can mitigate the over-parameterization issue of conventional cluster-based pooling. Additionally, observing the limitations of existing association-based localization methods, we propose a causal-driven approach attributing the contribution of each node to improve the interpretability of our framework. Extensive experiments on three public TCGA benchmark datasets demonstrate that our framework outperforms the state-of-the-art methods with considerable margins on various tasks. Our codes are available at https://github.com/HKU-MedAI/WSI-HGNN.

{{</citation>}}


### (23/43) DIFF-NST: Diffusion Interleaving For deFormable Neural Style Transfer (Dan Ruta et al., 2023)

{{<citation>}}

Dan Ruta, Gemma Canet Tarrés, Andrew Gilbert, Eli Shechtman, Nicholas Kolkin, John Collomosse. (2023)  
**DIFF-NST: Diffusion Interleaving For deFormable Neural Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2307.04157v2)  

---


**ABSTRACT**  
Neural Style Transfer (NST) is the field of study applying neural techniques to modify the artistic appearance of a content image to match the style of a reference style image. Traditionally, NST methods have focused on texture-based image edits, affecting mostly low level information and keeping most image structures the same. However, style-based deformation of the content is desirable for some styles, especially in cases where the style is abstract or the primary concept of the style is in its deformed rendition of some content. With the recent introduction of diffusion models, such as Stable Diffusion, we can access far more powerful image generation techniques, enabling new possibilities. In our work, we propose using this new class of models to perform style transfer while enabling deformable style transfer, an elusive capability in previous models. We show how leveraging the priors of these models can expose new artistic controls at inference time, and we document our findings in exploring this new direction for the field of style transfer.

{{</citation>}}


### (24/43) Latent Graph Attention for Enhanced Spatial Context (Ayush Singh et al., 2023)

{{<citation>}}

Ayush Singh, Yash Bhambhu, Himanshu Buckchash, Deepak K. Gupta, Dilip K. Prasad. (2023)  
**Latent Graph Attention for Enhanced Spatial Context**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.04149v2)  

---


**ABSTRACT**  
Global contexts in images are quite valuable in image-to-image translation problems. Conventional attention-based and graph-based models capture the global context to a large extent, however, these are computationally expensive. Moreover, the existing approaches are limited to only learning the pairwise semantic relation between any two points on the image. In this paper, we present Latent Graph Attention (LGA) a computationally inexpensive (linear to the number of nodes) and stable, modular framework for incorporating the global context in the existing architectures, especially empowering small-scale architectures to give performance closer to large size architectures, thus making the light-weight architectures more useful for edge devices with lower compute power and lower energy needs. LGA propagates information spatially using a network of locally connected graphs, thereby facilitating to construct a semantically coherent relation between any two spatially distant points that also takes into account the influence of the intermediate pixels. Moreover, the depth of the graph network can be used to adapt the extent of contextual spread to the target dataset, thereby being able to explicitly control the added computational cost. To enhance the learning mechanism of LGA, we also introduce a novel contrastive loss term that helps our LGA module to couple well with the original architecture at the expense of minimal additional computational load. We show that incorporating LGA improves the performance on three challenging applications, namely transparent object segmentation, image restoration for dehazing and optical flow estimation.

{{</citation>}}


### (25/43) A Survey and Approach to Chart Classification (Anurag Dhote et al., 2023)

{{<citation>}}

Anurag Dhote, Mohammed Javed, David S Doermann. (2023)  
**A Survey and Approach to Chart Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.04147v1)  

---


**ABSTRACT**  
Charts represent an essential source of visual information in documents and facilitate a deep understanding and interpretation of information typically conveyed numerically. In the scientific literature, there are many charts, each with its stylistic differences. Recently the document understanding community has begun to address the problem of automatic chart understanding, which begins with chart classification. In this paper, we present a survey of the current state-of-the-art techniques for chart classification and discuss the available datasets and their supported chart types. We broadly classify these contributions as traditional approaches based on ML, CNN, and Transformers. Furthermore, we carry out an extensive comparative performance analysis of CNN-based and transformer-based approaches on the recently published CHARTINFO UB-UNITECH PMC dataset for the CHART-Infographics competition at ICPR 2022. The data set includes 15 different chart categories, including 22,923 training images and 13,260 test images. We have implemented a vision-based transformer model that produces state-of-the-art results in chart classification.

{{</citation>}}


### (26/43) A Novel Explainable Artificial Intelligence Model in Image Classification problem (Quoc Hung Cao et al., 2023)

{{<citation>}}

Quoc Hung Cao, Truong Thanh Hung Nguyen, Vo Thanh Khang Nguyen, Xuan Phong Nguyen. (2023)  
**A Novel Explainable Artificial Intelligence Model in Image Classification problem**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Image Classification, ImageNet  
[Paper Link](http://arxiv.org/abs/2307.04137v1)  

---


**ABSTRACT**  
In recent years, artificial intelligence is increasingly being applied widely in many different fields and has a profound and direct impact on human life. Following this is the need to understand the principles of the model making predictions. Since most of the current high-precision models are black boxes, neither the AI scientist nor the end-user deeply understands what's going on inside these models. Therefore, many algorithms are studied for the purpose of explaining AI models, especially those in the problem of image classification in the field of computer vision such as LIME, CAM, GradCAM. However, these algorithms still have limitations such as LIME's long execution time and CAM's confusing interpretation of concreteness and clarity. Therefore, in this paper, we propose a new method called Segmentation - Class Activation Mapping (SeCAM) that combines the advantages of these algorithms above, while at the same time overcoming their disadvantages. We tested this algorithm with various models, including ResNet50, Inception-v3, VGG16 from ImageNet Large Scale Visual Recognition Challenge (ILSVRC) data set. Outstanding results when the algorithm has met all the requirements for a specific explanation in a remarkably concise time.

{{</citation>}}


### (27/43) ECL: Class-Enhancement Contrastive Learning for Long-tailed Skin Lesion Classification (Yilan Zhang et al., 2023)

{{<citation>}}

Yilan Zhang, Jianqi Chen, Ke Wang, Fengying Xie. (2023)  
**ECL: Class-Enhancement Contrastive Learning for Long-tailed Skin Lesion Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.04136v1)  

---


**ABSTRACT**  
Skin image datasets often suffer from imbalanced data distribution, exacerbating the difficulty of computer-aided skin disease diagnosis. Some recent works exploit supervised contrastive learning (SCL) for this long-tailed challenge. Despite achieving significant performance, these SCL-based methods focus more on head classes, yet ignoring the utilization of information in tail classes. In this paper, we propose class-Enhancement Contrastive Learning (ECL), which enriches the information of minority classes and treats different classes equally. For information enhancement, we design a hybrid-proxy model to generate class-dependent proxies and propose a cycle update strategy for parameters optimization. A balanced-hybrid-proxy loss is designed to exploit relations between samples and proxies with different classes treated equally. Taking both "imbalanced data" and "imbalanced diagnosis difficulty" into account, we further present a balanced-weighted cross-entropy loss following curriculum learning schedule. Experimental results on the classification of imbalanced skin lesion data have demonstrated the superiority and effectiveness of our method.

{{</citation>}}


### (28/43) Reasoning over the Behaviour of Objects in Video-Clips for Adverb-Type Recognition (Amrit Diggavi Seshadri et al., 2023)

{{<citation>}}

Amrit Diggavi Seshadri, Alessandra Russo. (2023)  
**Reasoning over the Behaviour of Objects in Video-Clips for Adverb-Type Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-SC, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2307.04132v2)  

---


**ABSTRACT**  
In this work, following the intuition that adverbs describing scene-sequences are best identified by reasoning over high-level concepts of object-behavior, we propose the design of a new framework that reasons over object-behaviours extracted from raw-video-clips to recognize the clip's corresponding adverb-types. Importantly, while previous works for general scene adverb-recognition assume knowledge of the clips underlying action-types, our method is directly applicable in the more general problem setting where the action-type of a video-clip is unknown. Specifically, we propose a novel pipeline that extracts human-interpretable object-behaviour-facts from raw video clips and propose novel symbolic and transformer based reasoning methods that operate over these extracted facts to identify adverb-types. Experiment results demonstrate that our proposed methods perform favourably against the previous state-of-the-art. Additionally, to support efforts in symbolic video-processing, we release two new datasets of object-behaviour-facts extracted from raw video clips - the MSR-VTT-ASP and ActivityNet-ASP datasets.

{{</citation>}}


### (29/43) Cross-modal Orthogonal High-rank Augmentation for RGB-Event Transformer-trackers (Zhiyu Zhu et al., 2023)

{{<citation>}}

Zhiyu Zhu, Junhui Hou, Dapeng Oliver Wu. (2023)  
**Cross-modal Orthogonal High-rank Augmentation for RGB-Event Transformer-trackers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2307.04129v1)  

---


**ABSTRACT**  
This paper addresses the problem of cross-modal object tracking from RGB videos and event data. Rather than constructing a complex cross-modal fusion network, we explore the great potential of a pre-trained vision Transformer (ViT). Particularly, we delicately investigate plug-and-play training augmentations that encourage the ViT to bridge the vast distribution gap between the two modalities, enabling comprehensive cross-modal information interaction and thus enhancing its ability. Specifically, we propose a mask modeling strategy that randomly masks a specific modality of some tokens to enforce the interaction between tokens from different modalities interacting proactively. To mitigate network oscillations resulting from the masking strategy and further amplify its positive effect, we then theoretically propose an orthogonal high-rank loss to regularize the attention matrix. Extensive experiments demonstrate that our plug-and-play training augmentation techniques can significantly boost state-of-the-art one-stream and twostream trackers to a large extent in terms of both tracking precision and success rate. Our new perspective and findings will potentially bring insights to the field of leveraging powerful pre-trained ViTs to model cross-modal data. The code will be publicly available.

{{</citation>}}


### (30/43) Marine Debris Detection in Satellite Surveillance using Attention Mechanisms (Ao Shen et al., 2023)

{{<citation>}}

Ao Shen, Yijie Zhu, Richard Jiang. (2023)  
**Marine Debris Detection in Satellite Surveillance using Attention Mechanisms**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.04128v1)  

---


**ABSTRACT**  
Marine debris is an important issue for environmental protection, but current methods for locating marine debris are yet limited. In order to achieve higher efficiency and wider applicability in the localization of Marine debris, this study tries to combine the instance segmentation of YOLOv7 with different attention mechanisms and explores the best model. By utilizing a labelled dataset consisting of satellite images containing ocean debris, we examined three attentional models including lightweight coordinate attention, CBAM (combining spatial and channel focus), and bottleneck transformer (based on self-attention). Box detection assessment revealed that CBAM achieved the best outcome (F1 score of 77%) compared to coordinate attention (F1 score of 71%) and YOLOv7/bottleneck transformer (both F1 scores around 66%). Mask evaluation showed CBAM again leading with an F1 score of 73%, whereas coordinate attention and YOLOv7 had comparable performances (around F1 score of 68%/69%) and bottleneck transformer lagged behind at F1 score of 56%. These findings suggest that CBAM offers optimal suitability for detecting marine debris. However, it should be noted that the bottleneck transformer detected some areas missed by manual annotation and displayed better mask precision for larger debris pieces, signifying potentially superior practical performance.

{{</citation>}}


### (31/43) Parametric Depth Based Feature Representation Learning for Object Detection and Segmentation in Bird's Eye View (Jiayu Yang et al., 2023)

{{<citation>}}

Jiayu Yang, Enze Xie, Miaomiao Liu, Jose M. Alvarez. (2023)  
**Parametric Depth Based Feature Representation Learning for Object Detection and Segmentation in Bird's Eye View**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.04106v2)  

---


**ABSTRACT**  
Recent vision-only perception models for autonomous driving achieved promising results by encoding multi-view image features into Bird's-Eye-View (BEV) space. A critical step and the main bottleneck of these methods is transforming image features into the BEV coordinate frame. This paper focuses on leveraging geometry information, such as depth, to model such feature transformation. Existing works rely on non-parametric depth distribution modeling leading to significant memory consumption, or ignore the geometry information to address this problem. In contrast, we propose to use parametric depth distribution modeling for feature transformation. We first lift the 2D image features to the 3D space defined for the ego vehicle via a predicted parametric depth distribution for each pixel in each view. Then, we aggregate the 3D feature volume based on the 3D space occupancy derived from depth to the BEV frame. Finally, we use the transformed features for downstream tasks such as object detection and semantic segmentation. Existing semantic segmentation methods do also suffer from an hallucination problem as they do not take visibility information into account. This hallucination can be particularly problematic for subsequent modules such as control and planning. To mitigate the issue, our method provides depth uncertainty and reliable visibility-aware estimations. We further leverage our parametric depth modeling to present a novel visibility-aware evaluation metric that, when taken into account, can mitigate the hallucination problem. Extensive experiments on object detection and semantic segmentation on the nuScenes datasets demonstrate that our method outperforms existing methods on both tasks.

{{</citation>}}


### (32/43) Enhancing Building Semantic Segmentation Accuracy with Super Resolution and Deep Learning: Investigating the Impact of Spatial Resolution on Various Datasets (Zhiling Guo et al., 2023)

{{<citation>}}

Zhiling Guo, Xiaodan Shi, Haoran Zhang, Dou Huang, Xiaoya Song, Jinyue Yan, Ryosuke Shibasaki. (2023)  
**Enhancing Building Semantic Segmentation Accuracy with Super Resolution and Deep Learning: Investigating the Impact of Spatial Resolution on Various Datasets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.04101v1)  

---


**ABSTRACT**  
The development of remote sensing and deep learning techniques has enabled building semantic segmentation with high accuracy and efficiency. Despite their success in different tasks, the discussions on the impact of spatial resolution on deep learning based building semantic segmentation are quite inadequate, which makes choosing a higher cost-effective data source a big challenge. To address the issue mentioned above, in this study, we create remote sensing images among three study areas into multiple spatial resolutions by super-resolution and down-sampling. After that, two representative deep learning architectures: UNet and FPN, are selected for model training and testing. The experimental results obtained from three cities with two deep learning models indicate that the spatial resolution greatly influences building segmentation results, and with a better cost-effectiveness around 0.3m, which we believe will be an important insight for data selection and preparation.

{{</citation>}}


### (33/43) Visible and infrared self-supervised fusion trained on a single example (Nati Ofir, 2023)

{{<citation>}}

Nati Ofir. (2023)  
**Visible and infrared self-supervised fusion trained on a single example**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.04100v1)  

---


**ABSTRACT**  
This paper addresses the problem of visible (RGB) to Near-Infrared (NIR) image fusion. Multispectral imaging is an important task relevant to image processing and computer vision, even more, since the development of the RGBT sensor. While the visible image sees color and suffers from noise, haze, and clouds, the NIR channel captures a clearer picture and it is significantly required by applications such as dehazing or object detection. The proposed approach fuses these two aligned channels by training a Convolutional-Neural-Network (CNN) by a Self-Supervised-Learning (SSL) on a single example. For each such pair, RGB and IR, the network is trained for seconds to deduce the final fusion. The SSL is based on Sturcture-of-Similarity (SSIM) loss combined with Edge-Preservation (EP) loss. The labels for the SSL are the input channels themselves. This fusion preserves the relevant detail of each spectral channel while not based on a heavy training process. In the experiments section, the proposed approach achieves better qualitative and quantitative multispectral fusion results with respect to other recent methods, that are not based on large dataset training.

{{</citation>}}


### (34/43) CMDFusion: Bidirectional Fusion Network with Cross-modality Knowledge Distillation for LIDAR Semantic Segmentation (Jun Cen et al., 2023)

{{<citation>}}

Jun Cen, Shiwei Zhang, Yixuan Pei, Kun Li, Hang Zheng, Maochun Luo, Yingya Zhang, Qifeng Chen. (2023)  
**CMDFusion: Bidirectional Fusion Network with Cross-modality Knowledge Distillation for LIDAR Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.04091v1)  

---


**ABSTRACT**  
2D RGB images and 3D LIDAR point clouds provide complementary knowledge for the perception system of autonomous vehicles. Several 2D and 3D fusion methods have been explored for the LIDAR semantic segmentation task, but they suffer from different problems. 2D-to-3D fusion methods require strictly paired data during inference, which may not be available in real-world scenarios, while 3D-to-2D fusion methods cannot explicitly make full use of the 2D information. Therefore, we propose a Bidirectional Fusion Network with Cross-Modality Knowledge Distillation (CMDFusion) in this work. Our method has two contributions. First, our bidirectional fusion scheme explicitly and implicitly enhances the 3D feature via 2D-to-3D fusion and 3D-to-2D fusion, respectively, which surpasses either one of the single fusion schemes. Second, we distillate the 2D knowledge from a 2D network (Camera branch) to a 3D network (2D knowledge branch) so that the 3D network can generate 2D information even for those points not in the FOV (field of view) of the camera. In this way, RGB images are not required during inference anymore since the 2D knowledge branch provides 2D information according to the 3D LIDAR input. We show that our CMDFusion achieves the best performance among all fusion-based methods on SemanticKITTI and nuScenes datasets. The code will be released at https://github.com/Jun-CEN/CMDFusion.

{{</citation>}}


### (35/43) SVIT: Scaling up Visual Instruction Tuning (Bo Zhao et al., 2023)

{{<citation>}}

Bo Zhao, Boya Wu, Tiejun Huang. (2023)  
**SVIT: Scaling up Visual Instruction Tuning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4, QA  
[Paper Link](http://arxiv.org/abs/2307.04087v1)  

---


**ABSTRACT**  
Thanks to the emerging of foundation models, the large language and vision models are integrated to acquire the multimodal ability of visual captioning, dialogue, question answering, etc. Although existing multimodal models present impressive performance of visual understanding and reasoning, their limits are still largely under-explored due to the scarcity of high-quality instruction tuning data. To push the limits of multimodal capability, we Sale up Visual Instruction Tuning (SVIT) by constructing a dataset of 3.2 million visual instruction tuning data including 1.6M conversation question-answer (QA) pairs and 1.6M complex reasoning QA pairs and 106K detailed image descriptions. Besides the volume, the proposed dataset is also featured by the high quality and rich diversity, which is generated by prompting GPT-4 with the abundant manual annotations of images. We empirically verify that training multimodal models on SVIT can significantly improve the multimodal performance in terms of visual perception, reasoning and planing.

{{</citation>}}


### (36/43) Random Position Adversarial Patch for Vision Transformers (Mingzhen Shao, 2023)

{{<citation>}}

Mingzhen Shao. (2023)  
**Random Position Adversarial Patch for Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.04066v1)  

---


**ABSTRACT**  
Previous studies have shown the vulnerability of vision transformers to adversarial patches, but these studies all rely on a critical assumption: the attack patches must be perfectly aligned with the patches used for linear projection in vision transformers. Due to this stringent requirement, deploying adversarial patches for vision transformers in the physical world becomes impractical, unlike their effectiveness on CNNs. This paper proposes a novel method for generating an adversarial patch (G-Patch) that overcomes the alignment constraint, allowing the patch to launch a targeted attack at any position within the field of view. Specifically, instead of directly optimizing the patch using gradients, we employ a GAN-like structure to generate the adversarial patch. Our experiments show the effectiveness of the adversarial patch in achieving universal attacks on vision transformers, both in digital and physical-world scenarios. Additionally, further analysis reveals that the generated adversarial patch exhibits robustness to brightness restriction, color transfer, and random noise. Real-world attack experiments validate the effectiveness of the G-Patch to launch robust attacks even under some very challenging conditions.

{{</citation>}}


## cs.RO (1)



### (37/43) Natural Language Instructions for Intuitive Human Interaction with Robotic Assistants in Field Construction Work (Somin Park et al., 2023)

{{<citation>}}

Somin Park, Xi Wang, Carol C. Menassa, Vineet R. Kamat, Joyce Y. Chai. (2023)  
**Natural Language Instructions for Intuitive Human Interaction with Robotic Assistants in Field Construction Work**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-HC, cs-RO, cs.RO  
Keywords: NLU, Natural Language Understanding  
[Paper Link](http://arxiv.org/abs/2307.04195v2)  

---


**ABSTRACT**  
The introduction of robots is widely considered to have significant potential of alleviating the issues of worker shortage and stagnant productivity that afflict the construction industry. However, it is challenging to use fully automated robots in complex and unstructured construction sites. Human-Robot Collaboration (HRC) has shown promise of combining human workers' flexibility and robot assistants' physical abilities to jointly address the uncertainties inherent in construction work. When introducing HRC in construction, it is critical to recognize the importance of teamwork and supervision in field construction and establish a natural and intuitive communication system for the human workers and robotic assistants. Natural language-based interaction can enable intuitive and familiar communication with robots for human workers who are non-experts in robot programming. However, limited research has been conducted on this topic in construction. This paper proposes a framework to allow human workers to interact with construction robots based on natural language instructions. The proposed method consists of three stages: Natural Language Understanding (NLU), Information Mapping (IM), and Robot Control (RC). Natural language instructions are input to a language model to predict a tag for each word in the NLU module. The IM module uses the result of the NLU module and building component information to generate the final instructional output essential for a robot to acknowledge and perform the construction task. A case study for drywall installation is conducted to evaluate the proposed approach. The obtained results highlight the potential of using natural language-based interaction to replicate the communication that occurs between human workers within the context of human-robot teams.

{{</citation>}}


## cs.CR (2)



### (38/43) Intrusion Resilience Systems for Modern Vehicles (Ali Shoker et al., 2023)

{{<citation>}}

Ali Shoker, Vincent Rahli, Jeremie Decouchant, Paulo Esteves-Verissimo. (2023)  
**Intrusion Resilience Systems for Modern Vehicles**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-DC, cs-NI, cs-SY, cs.CR, eess-SY  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2307.04184v1)  

---


**ABSTRACT**  
Current vehicular Intrusion Detection and Prevention Systems either incur high false-positive rates or do not capture zero-day vulnerabilities, leading to safety-critical risks. In addition, prevention is limited to few primitive options like dropping network packets or extreme options, e.g., ECU Bus-off state. To fill this gap, we introduce the concept of vehicular Intrusion Resilience Systems (IRS) that ensures the resilience of critical applications despite assumed faults or zero-day attacks, as long as threat assumptions are met. IRS enables running a vehicular application in a replicated way, i.e., as a Replicated State Machine, over several ECUs, and then requiring the replicated processes to reach a form of Byzantine agreement before changing their local state. Our study rides the mutation of modern vehicular environments, which are closing the gap between simple and resource-constrained "real-time and embedded systems", and complex and powerful "information technology" ones. It shows that current vehicle (e.g., Zonal) architectures and networks are becoming plausible for such modular fault and intrusion tolerance solutions,deemed too heavy in the past. Our evaluation on a simulated Automotive Ethernet network running two state-of-the-art agreement protocols (Damysus and Hotstuff) shows that the achieved latency and throughout are feasible for many Automotive applications.

{{</citation>}}


### (39/43) A Lightweight Approach for Network Intrusion Detection based on Self-Knowledge Distillation (Shuo Yang et al., 2023)

{{<citation>}}

Shuo Yang, Xinran Zheng, Zhengzhuo Xu, Xingjun Wang. (2023)  
**A Lightweight Approach for Network Intrusion Detection based on Self-Knowledge Distillation**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Intrusion Detection, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2307.10191v1)  

---


**ABSTRACT**  
Network Intrusion Detection (NID) works as a kernel technology for the security network environment, obtaining extensive research and application. Despite enormous efforts by researchers, NID still faces challenges in deploying on resource-constrained devices. To improve detection accuracy while reducing computational costs and model storage simultaneously, we propose a lightweight intrusion detection approach based on self-knowledge distillation, namely LNet-SKD, which achieves the trade-off between accuracy and efficiency. Specifically, we carefully design the DeepMax block to extract compact representation efficiently and construct the LNet by stacking DeepMax blocks. Furthermore, considering compensating for performance degradation caused by the lightweight network, we adopt batch-wise self-knowledge distillation to provide the regularization of training consistency. Experiments on benchmark datasets demonstrate the effectiveness of our proposed LNet-SKD, which outperforms existing state-of-the-art techniques with fewer parameters and lower computation loads.

{{</citation>}}


## eess.SP (1)



### (40/43) Emotion Analysis on EEG Signal Using Machine Learning and Neural Network (S. M. Masrur Ahmed et al., 2023)

{{<citation>}}

S. M. Masrur Ahmed, Eshaan Tanzim Sabur. (2023)  
**Emotion Analysis on EEG Signal Using Machine Learning and Neural Network**  

---
Primary Category: eess.SP  
Categories: cs-AI, cs-HC, eess-SP, eess.SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.05375v1)  

---


**ABSTRACT**  
Emotion has a significant influence on how one thinks and interacts with others. It serves as a link between how a person feels and the actions one takes, or it could be said that it influences one's life decisions on occasion. Since the patterns of emotions and their reflections vary from person to person, their inquiry must be based on approaches that are effective over a wide range of population regions. To extract features and enhance accuracy, emotion recognition using brain waves or EEG signals requires the implementation of efficient signal processing techniques. Various approaches to human-machine interaction technologies have been ongoing for a long time, and in recent years, researchers have had great success in automatically understanding emotion using brain signals. In our research, several emotional states were classified and tested on EEG signals collected from a well-known publicly available dataset, the DEAP Dataset, using SVM (Support Vector Machine), KNN (K-Nearest Neighbor), and an advanced neural network model, RNN (Recurrent Neural Network), trained with LSTM (Long Short Term Memory). The main purpose of this study is to improve ways to improve emotion recognition performance using brain signals. Emotions, on the other hand, can change with time. As a result, the changes in emotion over time are also examined in our research.

{{</citation>}}


## cs.SE (1)



### (41/43) A User Study on Explainable Online Reinforcement Learning for Adaptive Systems (Andreas Metzger et al., 2023)

{{<citation>}}

Andreas Metzger, Jan Laufer, Felix Feit, Klaus Pohl. (2023)  
**A User Study on Explainable Online Reinforcement Learning for Adaptive Systems**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.04098v1)  

---


**ABSTRACT**  
Online reinforcement learning (RL) is increasingly used for realizing adaptive systems in the presence of design time uncertainty. Online RL facilitates learning from actual operational data and thereby leverages feedback only available at runtime. However, Online RL requires the definition of an effective and correct reward function, which quantifies the feedback to the RL algorithm and thereby guides learning. With Deep RL gaining interest, the learned knowledge is no longer explicitly represented, but is represented as a neural network. For a human, it becomes practically impossible to relate the parametrization of the neural network to concrete RL decisions. Deep RL thus essentially appears as a black box, which severely limits the debugging of adaptive systems. We previously introduced the explainable RL technique XRL-DINE, which provides visual insights into why certain decisions were made at important time points. Here, we introduce an empirical user study involving 54 software engineers from academia and industry to assess (1) the performance of software engineers when performing different tasks using XRL-DINE and (2) the perceived usefulness and ease of use of XRL-DINE.

{{</citation>}}


## cs.CY (1)



### (42/43) Disentangling Societal Inequality from Model Biases: Gender Inequality in Divorce Court Proceedings (Sujan Dutta et al., 2023)

{{<citation>}}

Sujan Dutta, Parth Srivastava, Vaishnavi Solunke, Swaprava Nath, Ashiqur R. KhudaBukhsh. (2023)  
**Disentangling Societal Inequality from Model Biases: Gender Inequality in Divorce Court Proceedings**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CY  
Keywords: AI, Bias, NLP  
[Paper Link](http://arxiv.org/abs/2307.10200v1)  

---


**ABSTRACT**  
Divorce is the legal dissolution of a marriage by a court. Since this is usually an unpleasant outcome of a marital union, each party may have reasons to call the decision to quit which is generally documented in detail in the court proceedings. Via a substantial corpus of 17,306 court proceedings, this paper investigates gender inequality through the lens of divorce court proceedings. While emerging data sources (e.g., public court records) on sensitive societal issues hold promise in aiding social science research, biases present in cutting-edge natural language processing (NLP) methods may interfere with or affect such studies. We thus require a thorough analysis of potential gaps and limitations present in extant NLP resources. In this paper, on the methodological side, we demonstrate that existing NLP resources required several non-trivial modifications to quantify societal inequalities. On the substantive side, we find that while a large number of court cases perhaps suggest changing norms in India where women are increasingly challenging patriarchy, AI-powered analyses of these court proceedings indicate striking gender inequality with women often subjected to domestic violence.

{{</citation>}}


## cs.IR (1)



### (43/43) A Personalized Reinforcement Learning Summarization Service for Learning Structure from Unstructured Data (Samira Ghodratnama et al., 2023)

{{<citation>}}

Samira Ghodratnama, Amin Beheshti, Mehrdad Zakershahrak. (2023)  
**A Personalized Reinforcement Learning Summarization Service for Learning Structure from Unstructured Data**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs.IR  
Keywords: Reinforcement Learning, Summarization  
[Paper Link](http://arxiv.org/abs/2307.05696v1)  

---


**ABSTRACT**  
The exponential growth of textual data has created a crucial need for tools that assist users in extracting meaningful insights. Traditional document summarization approaches often fail to meet individual user requirements and lack structure for efficient information processing. To address these limitations, we propose Summation, a hierarchical personalized concept-based summarization approach. It synthesizes documents into a concise hierarchical concept map and actively engages users by learning and adapting to their preferences. Using a Reinforcement Learning algorithm, Summation generates personalized summaries for unseen documents on specific topics. This framework enhances comprehension, enables effective navigation, and empowers users to extract meaningful insights from large document collections aligned with their unique requirements.

{{</citation>}}
