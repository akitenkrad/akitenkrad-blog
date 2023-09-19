---
draft: false
title: "arXiv @ 2023.09.18"
date: 2023-09-18
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.18"
    identifier: arxiv_20230918
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.RO (7)](#csro-7)
- [cs.CL (24)](#cscl-24)
- [cs.SD (3)](#cssd-3)
- [cs.LG (8)](#cslg-8)
- [cs.CV (7)](#cscv-7)
- [cs.DS (1)](#csds-1)
- [cs.HC (1)](#cshc-1)
- [cs.CR (1)](#cscr-1)
- [cs.IR (1)](#csir-1)
- [cs.MM (1)](#csmm-1)
- [cs.AI (6)](#csai-6)
- [cs.AR (1)](#csar-1)
- [cs.IT (1)](#csit-1)
- [eess.SY (1)](#eesssy-1)
- [eess.AS (2)](#eessas-2)
- [cs.LO (1)](#cslo-1)
- [cs.SI (1)](#cssi-1)

## cs.RO (7)



### (1/67) Neural Network-based Fault Detection and Identification for Quadrotors using Dynamic Symmetry (Kunal Garg et al., 2023)

{{<citation>}}

Kunal Garg, Chuchu Fan. (2023)  
**Neural Network-based Fault Detection and Identification for Quadrotors using Dynamic Symmetry**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY, math-OC  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.09108v1)  

---


**ABSTRACT**  
Autonomous robotic systems, such as quadrotors, are susceptible to actuator faults, and for the safe operation of such systems, timely detection and isolation of these faults is essential. Neural networks can be used for verification of actuator performance via online actuator fault detection with high accuracy. In this paper, we develop a novel model-free fault detection and isolation (FDI) framework for quadrotor systems using long-short-term memory (LSTM) neural network architecture. The proposed framework only uses system output data and the commanded control input and requires no knowledge of the system model. Utilizing the symmetry in quadrotor dynamics, we train the FDI for fault in just one of the motors (e.g., motor $\# 2$), and the trained FDI can predict faults in any of the motors. This reduction in search space enables us to design an FDI for partial fault as well as complete fault scenarios. Numerical experiments illustrate that the proposed NN-FDI correctly verifies the actuator performance and identifies partial as well as complete faults with over $90\%$ prediction accuracy. We also illustrate that model-free NN-FDI performs at par with model-based FDI, and is robust to model uncertainties as well as distribution shifts in input data.

{{</citation>}}


### (2/67) CppFlow: Generative Inverse Kinematics for Efficient and Robust Cartesian Path Planning (Jeremy Morgan et al., 2023)

{{<citation>}}

Jeremy Morgan, David Millard, Gaurav S. Sukhatme. (2023)  
**CppFlow: Generative Inverse Kinematics for Efficient and Robust Cartesian Path Planning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09102v1)  

---


**ABSTRACT**  
In this work we present CppFlow - a novel and performant planner for the Cartesian Path Planning problem, which finds valid trajectories up to 129x faster than current methods, while also succeeding on more difficult problems where others fail. At the core of the proposed algorithm is the use of a learned, generative Inverse Kinematics solver, which is able to efficiently produce promising entire candidate solution trajectories on the GPU. Precise, valid solutions are then found through classical approaches such as differentiable programming, global search, and optimization. In combining approaches from these two paradigms we get the best of both worlds - efficient approximate solutions from generative AI which are made exact using the guarantees of traditional planning and optimization. We evaluate our system against other state of the art methods on a set of established baselines as well as new ones introduced in this work and find that our method significantly outperforms others in terms of the time to find a valid solution and planning success rate, and performs comparably in terms of trajectory length over time. The work is made open source and available for use upon acceptance.

{{</citation>}}


### (3/67) RMP: A Random Mask Pretrain Framework for Motion Prediction (Yi Yang et al., 2023)

{{<citation>}}

Yi Yang, Qingwen Zhang, Thomas Gilles, Nazre Batool, John Folkesson. (2023)  
**RMP: A Random Mask Pretrain Framework for Motion Prediction**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.08989v1)  

---


**ABSTRACT**  
As the pretraining technique is growing in popularity, little work has been done on pretrained learning-based motion prediction methods in autonomous driving. In this paper, we propose a framework to formalize the pretraining task for trajectory prediction of traffic participants. Within our framework, inspired by the random masked model in natural language processing (NLP) and computer vision (CV), objects' positions at random timesteps are masked and then filled in by the learned neural network (NN). By changing the mask profile, our framework can easily switch among a range of motion-related tasks. We show that our proposed pretraining framework is able to deal with noisy inputs and improves the motion prediction accuracy and miss rate, especially for objects occluded over time by evaluating it on Argoverse and NuScenes datasets.

{{</citation>}}


### (4/67) Outram: One-shot Global Localization via Triangulated Scene Graph and Global Outlier Pruning (Pengyu Yin et al., 2023)

{{<citation>}}

Pengyu Yin, Haozhi Cao, Thien-Minh Nguyen, Shenghai Yuan, Shuyang Zhang, Kangcheng Liu, Lihua Xie. (2023)  
**Outram: One-shot Global Localization via Triangulated Scene Graph and Global Outlier Pruning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2309.08914v1)  

---


**ABSTRACT**  
One-shot LiDAR localization refers to the ability to estimate the robot pose from one single point cloud, which yields significant advantages in initialization and relocalization processes. In the point cloud domain, the topic has been extensively studied as a global descriptor retrieval (i.e., loop closure detection) and pose refinement (i.e., point cloud registration) problem both in isolation or combined. However, few have explicitly considered the relationship between candidate retrieval and correspondence generation in pose estimation, leaving them brittle to substructure ambiguities. To this end, we propose a hierarchical one-shot localization algorithm called Outram that leverages substructures of 3D scene graphs for locally consistent correspondence searching and global substructure-wise outlier pruning. Such a hierarchical process couples the feature retrieval and the correspondence extraction to resolve the substructure ambiguities by conducting a local-to-global consistency refinement. We demonstrate the capability of Outram in a variety of scenarios in multiple large-scale outdoor datasets. Our implementation is open-sourced: https://github.com/Pamphlett/Outram.

{{</citation>}}


### (5/67) Stylized Table Tennis Robots Skill Learning with Incomplete Human Demonstrations (Xiang Zhu et al., 2023)

{{<citation>}}

Xiang Zhu, Zixuan Chen, Jianyu Chen. (2023)  
**Stylized Table Tennis Robots Skill Learning with Incomplete Human Demonstrations**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.08904v1)  

---


**ABSTRACT**  
In recent years, Reinforcement Learning (RL) is becoming a popular technique for training controllers for robots. However, for complex dynamic robot control tasks, RL-based method often produces controllers with unrealistic styles. In contrast, humans can learn well-stylized skills under supervisions. For example, people learn table tennis skills by imitating the motions of coaches. Such reference motions are often incomplete, e.g. without the presence of an actual ball. Inspired by this, we propose an RL-based algorithm to train a robot that can learn the playing style from such incomplete human demonstrations. We collect data through the teaching-and-dragging method. We also propose data augmentation techniques to enable our robot to adapt to balls of different velocities. We finally evaluate our policy in different simulators with varying dynamics.

{{</citation>}}


### (6/67) Pour me a drink: Robotic Precision Pouring Carbonated Beverages into Transparent Containers (Feiya Zhu et al., 2023)

{{<citation>}}

Feiya Zhu, Shuo Hu, Letian Leng, Alison Bartsch, Abraham George, Amir Barati Farimani. (2023)  
**Pour me a drink: Robotic Precision Pouring Carbonated Beverages into Transparent Containers**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.08892v1)  

---


**ABSTRACT**  
With the growing emphasis on the development and integration of service robots within household environments, we will need to endow robots with the ability to reliably pour a variety of liquids. However, liquid handling and pouring is a challenging task due to the complex dynamics and varying properties of different liquids, the exacting precision required to prevent spills and ensure accurate pouring, and the necessity for robots to adapt seamlessly to a multitude of containers in real-world scenarios. In response to these challenges, we propose a novel autonomous robotics pipeline that empowers robots to execute precision pouring tasks, encompassing both carbonated and non-carbonated liquids, as well as opaque and transparent liquids, into a variety of transparent containers. Our proposed approach maximizes the potential of RGB input alone, achieving zero-shot capability by harnessing existing pre-trained vision segmentation models. This eliminates the need for additional data collection, manual image annotations, or extensive training. Furthermore, our work integrates ChatGPT, facilitating seamless interaction between individuals without prior expertise in robotics and our pouring pipeline. This integration enables users to effortlessly request and execute pouring actions. Our experiments demonstrate the pipeline's capability to successfully pour a diverse range of carbonated and non-carbonated beverages into containers of varying sizes, relying solely on visual input.

{{</citation>}}


### (7/67) ARTEMIS: AI-driven Robotic Triage Labeling and Emergency Medical Information System (Sathvika Kotha et al., 2023)

{{<citation>}}

Sathvika Kotha, Hrishikesh Viswanath, Kshitij Tiwari, Aniket Bera. (2023)  
**ARTEMIS: AI-driven Robotic Triage Labeling and Emergency Medical Information System**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.08865v1)  

---


**ABSTRACT**  
Mass casualty incidents (MCIs) pose a formidable challenge to emergency medical services by overwhelming available resources and personnel. Effective victim assessment is paramount to minimizing casualties during such a crisis. In this paper, we introduce ARTEMIS, an AI-driven Robotic Triage Labeling and Emergency Medical Information System. This system comprises a deep learning model for acuity labeling that is integrated with a robot, that performs the preliminary assessment of injury severity in patients and assigns appropriate triage labels. Additionally, we have developed a frontend (graphical user interface) that is updated by the robots in real time and is accessible to the first responders. To validate the reliability of our proposed algorithmic triage protocol, we employed an off-the-shelf robot kit equipped with sensors for vital sign acquisition. A controlled laboratory simulation of an MCI was conducted to assess the system's performance and effectiveness in real-world scenarios resulting in a triage-level classification accuracy of 92%. This noteworthy achievement underscores the model's proficiency in discerning crucial patterns for accurate triage classification, showcasing its promising potential in healthcare applications.

{{</citation>}}


## cs.CL (24)



### (8/67) The Impact of Debiasing on the Performance of Language Models in Downstream Tasks is Underestimated (Masahiro Kaneko et al., 2023)

{{<citation>}}

Masahiro Kaneko, Danushka Bollegala, Naoaki Okazaki. (2023)  
**The Impact of Debiasing on the Performance of Language Models in Downstream Tasks is Underestimated**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.09092v1)  

---


**ABSTRACT**  
Pre-trained language models trained on large-scale data have learned serious levels of social biases. Consequently, various methods have been proposed to debias pre-trained models. Debiasing methods need to mitigate only discriminatory bias information from the pre-trained models, while retaining information that is useful for the downstream tasks. In previous research, whether useful information is retained has been confirmed by the performance of downstream tasks in debiased pre-trained models. On the other hand, it is not clear whether these benchmarks consist of data pertaining to social biases and are appropriate for investigating the impact of debiasing. For example in gender-related social biases, data containing female words (e.g. ``she, female, woman''), male words (e.g. ``he, male, man''), and stereotypical words (e.g. ``nurse, doctor, professor'') are considered to be the most affected by debiasing. If there is not much data containing these words in a benchmark dataset for a target task, there is the possibility of erroneously evaluating the effects of debiasing. In this study, we compare the impact of debiasing on performance across multiple downstream tasks using a wide-range of benchmark datasets that containing female, male, and stereotypical words. Experiments show that the effects of debiasing are consistently \emph{underestimated} across all tasks. Moreover, the effects of debiasing could be reliably evaluated by separately considering instances containing female, male, and stereotypical words than all of the instances in a benchmark dataset.

{{</citation>}}


### (9/67) RMDM: A Multilabel Fakenews Dataset for Vietnamese Evidence Verification (Hai-Long Nguyen et al., 2023)

{{<citation>}}

Hai-Long Nguyen, Thi-Kieu-Trang Pham, Thai-Son Le, Tan-Minh Nguyen, Thi-Hai-Yen Vuong, Ha-Thanh Nguyen. (2023)  
**RMDM: A Multilabel Fakenews Dataset for Vietnamese Evidence Verification**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, BERT, GPT  
[Paper Link](http://arxiv.org/abs/2309.09071v1)  

---


**ABSTRACT**  
In this study, we present a novel and challenging multilabel Vietnamese dataset (RMDM) designed to assess the performance of large language models (LLMs), in verifying electronic information related to legal contexts, focusing on fake news as potential input for electronic evidence. The RMDM dataset comprises four labels: real, mis, dis, and mal, representing real information, misinformation, disinformation, and mal-information, respectively. By including these diverse labels, RMDM captures the complexities of differing fake news categories and offers insights into the abilities of different language models to handle various types of information that could be part of electronic evidence. The dataset consists of a total of 1,556 samples, with 389 samples for each label. Preliminary tests on the dataset using GPT-based and BERT-based models reveal variations in the models' performance across different labels, indicating that the dataset effectively challenges the ability of various language models to verify the authenticity of such information. Our findings suggest that verifying electronic information related to legal contexts, including fake news, remains a difficult problem for language models, warranting further attention from the research community to advance toward more reliable AI models for potential legal applications.

{{</citation>}}


### (10/67) NOWJ1@ALQAC 2023: Enhancing Legal Task Performance with Classic Statistical Models and Pre-trained Language Models (Tan-Minh Nguyen et al., 2023)

{{<citation>}}

Tan-Minh Nguyen, Xuan-Hoa Nguyen, Ngoc-Duy Mai, Minh-Quan Hoang, Van-Huan Nguyen, Hoang-Viet Nguyen, Ha-Thanh Nguyen, Thi-Hai-Yen Vuong. (2023)  
**NOWJ1@ALQAC 2023: Enhancing Legal Task Performance with Classic Statistical Models and Pre-trained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Legal, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.09070v1)  

---


**ABSTRACT**  
This paper describes the NOWJ1 Team's approach for the Automated Legal Question Answering Competition (ALQAC) 2023, which focuses on enhancing legal task performance by integrating classical statistical models and Pre-trained Language Models (PLMs). For the document retrieval task, we implement a pre-processing step to overcome input limitations and apply learning-to-rank methods to consolidate features from various models. The question-answering task is split into two sub-tasks: sentence classification and answer extraction. We incorporate state-of-the-art models to develop distinct systems for each sub-task, utilizing both classic statistical models and pre-trained Language Models. Experimental results demonstrate the promising potential of our proposed methodology in the competition.

{{</citation>}}


### (11/67) Constructing a Knowledge Graph for Vietnamese Legal Cases with Heterogeneous Graphs (Thi-Hai-Yen Vuong et al., 2023)

{{<citation>}}

Thi-Hai-Yen Vuong, Minh-Quan Hoang, Tan-Minh Nguyen, Hoang-Trung Nguyen, Ha-Thanh Nguyen. (2023)  
**Constructing a Knowledge Graph for Vietnamese Legal Cases with Heterogeneous Graphs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Legal  
[Paper Link](http://arxiv.org/abs/2309.09069v1)  

---


**ABSTRACT**  
This paper presents a knowledge graph construction method for legal case documents and related laws, aiming to organize legal information efficiently and enhance various downstream tasks. Our approach consists of three main steps: data crawling, information extraction, and knowledge graph deployment. First, the data crawler collects a large corpus of legal case documents and related laws from various sources, providing a rich database for further processing. Next, the information extraction step employs natural language processing techniques to extract entities such as courts, cases, domains, and laws, as well as their relationships from the unstructured text. Finally, the knowledge graph is deployed, connecting these entities based on their extracted relationships, creating a heterogeneous graph that effectively represents legal information and caters to users such as lawyers, judges, and scholars. The established baseline model leverages unsupervised learning methods, and by incorporating the knowledge graph, it demonstrates the ability to identify relevant laws for a given legal case. This approach opens up opportunities for various applications in the legal domain, such as legal case analysis, legal recommendation, and decision support.

{{</citation>}}


### (12/67) Exploring the impact of low-rank adaptation on the performance, efficiency, and regularization of RLHF (Simeng Sun et al., 2023)

{{<citation>}}

Simeng Sun, Dhawal Gupta, Mohit Iyyer. (2023)  
**Exploring the impact of low-rank adaptation on the performance, efficiency, and regularization of RLHF**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2309.09055v1)  

---


**ABSTRACT**  
During the last stage of RLHF, a large language model is aligned to human intents via PPO training, a process that generally requires large-scale computational resources. In this technical report, we empirically investigate an efficient implementation of RLHF using low-rank adaptation (LoRA), which allows us to align the LLaMA 7B checkpoint on the Alpaca dataset using only two A100 GPUs instead of the eight required for full model fine-tuning. Despite tuning only 0.2% of LLaMA 7B's parameters, our implementation achieves better performance than the publicly-released AlpacaFarm checkpoint with full model fine-tuning. Next, we analyze several configurations of our LoRA-based PPO implementation, varying the form of the KL regularization term in the training objective. We find that (1) removing this penalty term does not harm performance on the AlpacaFarm evaluation set under our LoRA setup; (2) other regularizers, such as Jensen-Shannon divergence, lead to improved performance; and (3) while PPO training negatively impacts the factuality of model-generated responses, training with LoRA largely mitigates this effect. We release our code and pretrained checkpoints to facilitate future research on more efficient RLHF.

{{</citation>}}


### (13/67) Context-aware Adversarial Attack on Named Entity Recognition (Shuguang Chen et al., 2023)

{{<citation>}}

Shuguang Chen, Leonardo Neves, Thamar Solorio. (2023)  
**Context-aware Adversarial Attack on Named Entity Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Adversarial Attack, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2309.08999v1)  

---


**ABSTRACT**  
In recent years, large pre-trained language models (PLMs) have achieved remarkable performance on many natural language processing benchmarks. Despite their success, prior studies have shown that PLMs are vulnerable to attacks from adversarial examples. In this work, we focus on the named entity recognition task and study context-aware adversarial attack methods to examine the model's robustness. Specifically, we propose perturbing the most informative words for recognizing entities to create adversarial examples and investigate different candidate replacement methods to generate natural and plausible adversarial examples. Experiments and analyses show that our methods are more effective in deceiving the model into making wrong predictions than strong baselines.

{{</citation>}}


### (14/67) Rethinking STS and NLI in Large Language Models (Yuxia Wang et al., 2023)

{{<citation>}}

Yuxia Wang, Minghan Wang, Preslav Nakov. (2023)  
**Rethinking STS and NLI in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, ChatGPT, GPT, Language Model, NLI  
[Paper Link](http://arxiv.org/abs/2309.08969v1)  

---


**ABSTRACT**  
In this study, we aim to rethink STS and NLI in the era of large language models (LLMs). We first evaluate the accuracy of clinical/biomedical STS and NLI over five datasets, and then we assess LLM predictive confidence and their capability of capturing collective human opinions. We find that LLMs may be able to provide personalised descriptions for a specific topic, or to generate semantically similar content in different tones, but that this is hard for current LLMs to make personalised judgements or decisions. We further find that zero-shot ChatGPT achieves competitive accuracy over clinical and biomedical STS/NLI, constraining to the fine-tuned BERT-base. However, there is a large variation in sampling, ensembled results perform the best.

{{</citation>}}


### (15/67) Sorted LLaMA: Unlocking the Potential of Intermediate Layers of Large Language Models for Dynamic Inference Using Sorted Fine-Tuning (SoFT) (Parsa Kavehzadeh et al., 2023)

{{<citation>}}

Parsa Kavehzadeh, Mojtaba Valipour, Marzieh Tahaei, Ali Ghodsi, Boxing Chen, Mehdi Rezagholizadeh. (2023)  
**Sorted LLaMA: Unlocking the Potential of Intermediate Layers of Large Language Models for Dynamic Inference Using Sorted Fine-Tuning (SoFT)**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: LLaMA, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2309.08968v1)  

---


**ABSTRACT**  
The rapid advancement of large language models (LLMs) has revolutionized natural language processing (NLP). While these models excel at understanding and generating human-like text, their widespread deployment can be prohibitively expensive. SortedNet is a recent training technique for enabling dynamic inference for deep neural networks. It leverages network modularity to create sub-models with varying computational loads, sorting them based on computation/accuracy characteristics in a nested manner. We extend SortedNet to generative NLP tasks, making large language models dynamic without any pretraining and by only replacing standard Supervised Fine-Tuning (SFT) with Sorted Fine-Tuning (SoFT) at the same costs. Our approach boosts model efficiency, eliminating the need for multiple models for various scenarios during inference. We show that using this approach, we are able to unlock the potential of intermediate layers of transformers in generating the target output. Our sub-models remain integral components of the original model, minimizing storage requirements and transition costs between different computational/latency budgets. By applying this approach on LLaMa 2 13B for tuning on the Stanford Alpaca dataset and comparing it to normal tuning and early exit via PandaLM benchmark, we show that Sorted Fine-Tuning can deliver models twice as fast as the original model while maintaining or exceeding performance.

{{</citation>}}


### (16/67) Struc-Bench: Are Large Language Models Really Good at Generating Complex Structured Data? (Xiangru Tang et al., 2023)

{{<citation>}}

Xiangru Tang, Yiming Zong, Yilun Zhao, Arman Cohan, Mark Gerstein. (2023)  
**Struc-Bench: Are Large Language Models Really Good at Generating Complex Structured Data?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2309.08963v1)  

---


**ABSTRACT**  
Despite the power of Large Language Models (LLMs) like GPT-4, they still struggle with tasks that require generating complex, structured outputs. In this study, we assess the capability of Current LLMs in generating complex structured data and propose a structure-aware fine-tuning approach as a solution to improve this ability. To perform a comprehensive evaluation, we propose Struc-Bench, include five representative LLMs (i.e., GPT-NeoX 20B, GPT-3.5, GPT-4, and Vicuna) and evaluate them on our carefully constructed datasets spanning raw text, HTML, and LaTeX tables. Based on our analysis of current model performance, we identify specific common formatting errors and areas of potential improvement. To address complex formatting requirements, we utilize FormatCoT (Chain-of-Thought) to generate format instructions from target outputs. Our experiments show that our structure-aware fine-tuning method, when applied to LLaMA-7B, significantly improves adherence to natural language constraints, outperforming other evaluated LLMs. Based on these results, we present an ability map of model capabilities from six dimensions (i.e., coverage, formatting, reasoning, comprehension, pragmatics, and hallucination). This map highlights the weaknesses of LLMs in handling complex structured outputs and suggests promising directions for future work. Our code and models can be found at https://github.com/gersteinlab/Struc-Bench.

{{</citation>}}


### (17/67) ODSum: New Benchmarks for Open Domain Multi-Document Summarization (Yijie Zhou et al., 2023)

{{<citation>}}

Yijie Zhou, Kejian Shi, Wencai Zhang, Yixin Liu, Yilun Zhao, Arman Cohan. (2023)  
**ODSum: New Benchmarks for Open Domain Multi-Document Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2309.08960v1)  

---


**ABSTRACT**  
Open-domain Multi-Document Summarization (ODMDS) is a critical tool for condensing vast arrays of documents into coherent, concise summaries. With a more inter-related document set, there does not necessarily exist a correct answer for the retrieval, making it hard to measure the retrieving performance. We propose a rule-based method to process query-based document summarization datasets into ODMDS datasets. Based on this method, we introduce a novel dataset, ODSum, a sophisticated case with its document index interdependent and often interrelated. We tackle ODMDS with the \textit{retrieve-then-summarize} method, and the performance of a list of retrievers and summarizers is investigated. Through extensive experiments, we identify variances in evaluation metrics and provide insights into their reliability. We also found that LLMs suffer great performance loss from retrieving errors. We further experimented methods to improve the performance as well as investigate their robustness against imperfect retrieval. We will release our data and code at https://github.com/yale-nlp/ODSum.

{{</citation>}}


### (18/67) Monolingual or Multilingual Instruction Tuning: Which Makes a Better Alpaca (Pinzhen Chen et al., 2023)

{{<citation>}}

Pinzhen Chen, Shaoxiong Ji, Nikolay Bogoychev, Barry Haddow, Kenneth Heafield. (2023)  
**Monolingual or Multilingual Instruction Tuning: Which Makes a Better Alpaca**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Multilingual  
[Paper Link](http://arxiv.org/abs/2309.08958v1)  

---


**ABSTRACT**  
Foundational large language models (LLMs) can be instruction-tuned to develop open-ended question-answering capability, facilitating applications such as the creation of AI assistants. While such efforts are often carried out in a single language, building on prior research, we empirically analyze cost-efficient approaches of monolingual and multilingual tuning, shedding light on the efficacy of LLMs in responding to queries across monolingual and multilingual contexts. Our study employs the Alpaca dataset and machine translations of it to form multilingual training data, which is then used to tune LLMs through low-rank adaptation and full-parameter training. Comparisons reveal that multilingual tuning is not crucial for an LLM's English performance, but is key to its robustness in a multilingual environment. With a fixed budget, a multilingual instruction-tuned model, merely trained on downsampled data, can be as powerful as training monolingual models for each language. Our findings serve as a guide for expanding language support through instruction tuning with constrained computational resources.

{{</citation>}}


### (19/67) Cross-Lingual Knowledge Editing in Large Language Models (Jiaan Wang et al., 2023)

{{<citation>}}

Jiaan Wang, Yunlong Liang, Zengkui Sun, Yuxuan Cao, Jiarong Xu. (2023)  
**Cross-Lingual Knowledge Editing in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2309.08952v1)  

---


**ABSTRACT**  
Knowledge editing aims to change language models' performance on several special cases (i.e., editing scope) by infusing the corresponding expected knowledge into them. With the recent advancements in large language models (LLMs), knowledge editing has been shown as a promising technique to adapt LLMs to new knowledge without retraining from scratch. However, most of the previous studies neglect the multi-lingual nature of some main-stream LLMs (e.g., LLaMA, ChatGPT and GPT-4), and typically focus on monolingual scenarios, where LLMs are edited and evaluated in the same language. As a result, it is still unknown the effect of source language editing on a different target language. In this paper, we aim to figure out this cross-lingual effect in knowledge editing. Specifically, we first collect a large-scale cross-lingual synthetic dataset by translating ZsRE from English to Chinese. Then, we conduct English editing on various knowledge editing methods covering different paradigms, and evaluate their performance in Chinese, and vice versa. To give deeper analyses of the cross-lingual effect, the evaluation includes four aspects, i.e., reliability, generality, locality and portability. Furthermore, we analyze the inconsistent behaviors of the edited models and discuss their specific challenges.

{{</citation>}}


### (20/67) Enhancing Large Language Model Induced Task-Oriented Dialogue Systems Through Look-Forward Motivated Goals (Zhiyuan Hu et al., 2023)

{{<citation>}}

Zhiyuan Hu, Yue Feng, Yang Deng, Zekun Li, See-Kiong Ng, Anh Tuan Luu, Bryan Hooi. (2023)  
**Enhancing Large Language Model Induced Task-Oriented Dialogue Systems Through Look-Forward Motivated Goals**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2309.08949v1)  

---


**ABSTRACT**  
Recently, the development of large language models (LLMs) has been significantly enhanced the question answering and dialogue generation, and makes them become increasingly popular in current practical scenarios. While unlike the general dialogue system which emphasizes the semantic performance, the task-oriented dialogue (ToD) systems aim to achieve the dialogue goal efficiently and successfully in multiple turns. Unfortunately, existing LLM-induced ToD systems lack the direct reward toward the final goal and do not take account of the dialogue proactivity that can strengthen the dialogue efficiency. To fill these gaps, we introduce the ProToD (Proactively Goal-Driven LLM-Induced ToD) approach, which anticipates the future dialogue actions and incorporates the goal-oriented reward signal to enhance ToD systems. Additionally, we present a novel evaluation method that assesses ToD systems based on goal-driven dialogue simulations. This method allows us to gauge user satisfaction, system efficiency and successful rate while overcoming the limitations of current Information and Success metrics. Empirical experiments conducted on the MultiWoZ 2.1 dataset demonstrate that our model can achieve superior performance using only 10% of the data compared to previous end-to-end fully supervised models. This improvement is accompanied by enhanced user satisfaction and efficiency.

{{</citation>}}


### (21/67) Leveraging Multi-lingual Positive Instances in Contrastive Learning to Improve Sentence Embedding (Kaiyan Zhao et al., 2023)

{{<citation>}}

Kaiyan Zhao, Qiyu Wu, Xin-Qiang Cai, Yoshimasa Tsuruoka. (2023)  
**Leveraging Multi-lingual Positive Instances in Contrastive Learning to Improve Sentence Embedding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning, Embedding, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2309.08929v1)  

---


**ABSTRACT**  
Learning multi-lingual sentence embeddings is a fundamental and significant task in natural language processing. Recent trends of learning both mono-lingual and multi-lingual sentence embeddings are mainly based on contrastive learning (CL) with an anchor, one positive, and multiple negative instances. In this work, we argue that leveraging multiple positives should be considered for multi-lingual sentence embeddings because (1) positives in a diverse set of languages can benefit cross-lingual learning, and (2) transitive similarity across multiple positives can provide reliable structural information to learn. In order to investigate the impact of CL with multiple positives, we propose a novel approach MPCL to effectively utilize multiple positive instances to improve learning multi-lingual sentence embeddings. Our experimental results on various backbone models and downstream tasks support that compared with conventional CL, MPCL leads to better retrieval, semantic similarity, and classification performances. We also observe that on unseen languages, sentence embedding models trained on multiple positives have better cross-lingual transferring performance than models trained on a single positive instance.

{{</citation>}}


### (22/67) Multimodal Multi-Hop Question Answering Through a Conversation Between Tools and Efficiently Finetuned Large Language Models (Hossein Rajabzadeh et al., 2023)

{{<citation>}}

Hossein Rajabzadeh, Suyuchen Wang, Hyock Ju Kwon, Bang Liu. (2023)  
**Multimodal Multi-Hop Question Answering Through a Conversation Between Tools and Efficiently Finetuned Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.08922v1)  

---


**ABSTRACT**  
We employ a tool-interacting divide-and-conquer strategy enabling large language models (LLMs) to answer complex multimodal multi-hop questions. In particular, we harness the power of large language models to divide a given multimodal multi-hop question into unimodal single-hop sub-questions to be answered by the appropriate tool from a predefined set of tools. After all corresponding tools provide the LLM with their answers, the LLM generates the next relevant unimodal single-hop question. To increase the reasoning ability of LLMs, we prompt chatGPT to generate a tool-interacting divide-and-conquer dataset. This dataset is then used to efficiently finetune the corresponding LLM. To assess the effectiveness of this approach, we conduct an evaluation on two recently introduced complex question-answering datasets. The experimental analysis demonstrate substantial improvements over existing state-of-the-art solutions, indicating the efficacy and generality of our strategy

{{</citation>}}


### (23/67) Investigating Subtler Biases in LLMs: Ageism, Beauty, Institutional, and Nationality Bias in Generative Models (Mahammed Kamruzzaman et al., 2023)

{{<citation>}}

Mahammed Kamruzzaman, Md. Minul Islam Shovon, Gene Louis Kim. (2023)  
**Investigating Subtler Biases in LLMs: Ageism, Beauty, Institutional, and Nationality Bias in Generative Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, NLP  
[Paper Link](http://arxiv.org/abs/2309.08902v1)  

---


**ABSTRACT**  
LLMs are increasingly powerful and widely used to assist users in a variety of tasks. This use risks the introduction of LLM biases to consequential decisions such as job hiring, human performance evaluation, and criminal sentencing. Bias in NLP systems along the lines of gender and ethnicity has been widely studied, especially for specific stereotypes (e.g., Asians are good at math). In this paper, we investigate bias along less studied, but still consequential, dimensions, such as age and beauty, measuring subtler correlated decisions that LLMs (specially autoregressive language models) make between social groups and unrelated positive and negative attributes. We ask whether LLMs hold wide-reaching biases of positive or negative sentiment for specific social groups similar to the ``what is beautiful is good'' bias found in people in experimental psychology. We introduce a template-generated dataset of sentence completion tasks that asks the model to select the most appropriate attribute to complete an evaluative statement about a person described as a member of a specific social group. We also reverse the completion task to select the social group based on an attribute. Finally, we report the correlations that we find for multiple cutting-edge LLMs. This dataset can be used as a benchmark to evaluate progress in more generalized biases and the templating technique can be used to expand the benchmark with minimal additional human annotation.

{{</citation>}}


### (24/67) Semantic Information Extraction for Text Data with Probability Graph (Zhouxiang Zhao et al., 2023)

{{<citation>}}

Zhouxiang Zhao, Zhaohui Yang, Ye Hu, Licheng Lin, Zhaoyang Zhang. (2023)  
**Semantic Information Extraction for Text Data with Probability Graph**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-SP  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2309.08879v1)  

---


**ABSTRACT**  
In this paper, the problem of semantic information extraction for resource constrained text data transmission is studied. In the considered model, a sequence of text data need to be transmitted within a communication resource-constrained network, which only allows limited data transmission. Thus, at the transmitter, the original text data is extracted with natural language processing techniques. Then, the extracted semantic information is captured in a knowledge graph. An additional probability dimension is introduced in this graph to capture the importance of each information. This semantic information extraction problem is posed as an optimization framework whose goal is to extract most important semantic information for transmission. To find an optimal solution for this problem, a Floyd's algorithm based solution coupled with an efficient sorting mechanism is proposed. Numerical results testify the effectiveness of the proposed algorithm with regards to two novel performance metrics including semantic uncertainty and semantic similarity.

{{</citation>}}


### (25/67) X-PARADE: Cross-Lingual Textual Entailment and Information Divergence across Paragraphs (Juan Diego Rodriguez et al., 2023)

{{<citation>}}

Juan Diego Rodriguez, Katrin Erk, Greg Durrett. (2023)  
**X-PARADE: Cross-Lingual Textual Entailment and Information Divergence across Paragraphs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLI, NLP, Textual Entailment  
[Paper Link](http://arxiv.org/abs/2309.08873v1)  

---


**ABSTRACT**  
Understanding when two pieces of text convey the same information is a goal touching many subproblems in NLP, including textual entailment and fact-checking. This problem becomes more complex when those two pieces of text are in different languages. Here, we introduce X-PARADE (Cross-lingual Paragraph-level Analysis of Divergences and Entailments), the first cross-lingual dataset of paragraph-level information divergences. Annotators label a paragraph in a target language at the span level and evaluate it with respect to a corresponding paragraph in a source language, indicating whether a given piece of information is the same, new, or new but can be inferred. This last notion establishes a link with cross-language NLI. Aligned paragraphs are sourced from Wikipedia pages in different languages, reflecting real information divergences observed in the wild. Armed with our dataset, we investigate a diverse set of approaches for this problem, including classic token alignment from machine translation, textual entailment methods that localize their decisions, and prompting of large language models. Our results show that these methods vary in their capability to handle inferable information, but they all fall short of human performance.

{{</citation>}}


### (26/67) PDFTriage: Question Answering over Long, Structured Documents (Jon Saad-Falcon et al., 2023)

{{<citation>}}

Jon Saad-Falcon, Joe Barrow, Alexa Siu, Ani Nenkova, Ryan A. Rossi, Franck Dernoncourt. (2023)  
**PDFTriage: Question Answering over Long, Structured Documents**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.08872v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have issues with document question answering (QA) in situations where the document is unable to fit in the small context length of an LLM. To overcome this issue, most existing works focus on retrieving the relevant context from the document, representing them as plain text. However, documents such as PDFs, web pages, and presentations are naturally structured with different pages, tables, sections, and so on. Representing such structured documents as plain text is incongruous with the user's mental model of these documents with rich structure. When a system has to query the document for context, this incongruity is brought to the fore, and seemingly trivial questions can trip up the QA system. To bridge this fundamental gap in handling structured documents, we propose an approach called PDFTriage that enables models to retrieve the context based on either structure or content. Our experiments demonstrate the effectiveness of the proposed PDFTriage-augmented models across several classes of questions where existing retrieval-augmented LLMs fail. To facilitate further research on this fundamental problem, we release our benchmark dataset consisting of 900+ human-generated questions over 80 structured documents from 10 different categories of question types for document QA.

{{</citation>}}


### (27/67) MHLAT: Multi-hop Label-wise Attention Model for Automatic ICD Coding (Junwen Duan et al., 2023)

{{<citation>}}

Junwen Duan, Han Jiang, Ying Yu. (2023)  
**MHLAT: Multi-hop Label-wise Attention Model for Automatic ICD Coding**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.08868v1)  

---


**ABSTRACT**  
International Classification of Diseases (ICD) coding is the task of assigning ICD diagnosis codes to clinical notes. This can be challenging given the large quantity of labels (nearly 9,000) and lengthy texts (up to 8,000 tokens). However, unlike the single-pass reading process in previous works, humans tend to read the text and label definitions again to get more confident answers. Moreover, although pretrained language models have been used to address these problems, they suffer from huge memory usage. To address the above problems, we propose a simple but effective model called the Multi-Hop Label-wise ATtention (MHLAT), in which multi-hop label-wise attention is deployed to get more precise and informative representations. Extensive experiments on three benchmark MIMIC datasets indicate that our method achieves significantly better or competitive performance on all seven metrics, with much fewer parameters to optimize.

{{</citation>}}


### (28/67) Has Sentiment Returned to the Pre-pandemic Level? A Sentiment Analysis Using U.S. College Subreddit Data from 2019 to 2022 (Tian Yan et al., 2023)

{{<citation>}}

Tian Yan, Fang Liu. (2023)  
**Has Sentiment Returned to the Pre-pandemic Level? A Sentiment Analysis Using U.S. College Subreddit Data from 2019 to 2022**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: BERT, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2309.08845v1)  

---


**ABSTRACT**  
As impact of COVID-19 pandemic winds down, both individuals and society gradually return to pre-pandemic activities. This study aims to explore how people's emotions have changed from the pre-pandemic during the pandemic to post-emergency period and whether it has returned to pre-pandemic level. We collected Reddit data in 2019 (pre-pandemic), 2020 (peak pandemic), 2021, and 2022 (late stages of pandemic, transitioning period to post-emergency period) from subreddits in 128 universities/colleges in the U.S., and a set of school-level characteristics. We predicted two sets of sentiments from a pre-trained Robustly Optimized BERT pre-training approach (RoBERTa) and graph attention network (GAT) that leverages both rich semantic and relational information among posted messages and then applied a logistic stacking method to obtain the final sentiment classification. After obtaining sentiment label for each message, we used a generalized linear mixed-effects model to estimate temporal trend in sentiment from 2019 to 2022 and how school-level factors may affect sentiment. Compared to the year 2019, the odds of negative sentiment in years 2020, 2021, and 2022 are 24%, 4.3%, and 10.3% higher, respectively, which are all statistically significant(adjusted $p$<0.05). Our study findings suggest a partial recovery in the sentiment composition in the post-pandemic-emergency era. The results align with common expectations and provide a detailed quantification of how sentiments have evolved from 2019 to 2022.

{{</citation>}}


### (29/67) Bias and Fairness in Chatbots: An Overview (Jintang Xue et al., 2023)

{{<citation>}}

Jintang Xue, Yun-Cheng Wang, Chengwei Wei, Xiaofeng Liu, Jonghye Woo, C. -C. Jay Kuo. (2023)  
**Bias and Fairness in Chatbots: An Overview**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: Bias, NLP  
[Paper Link](http://arxiv.org/abs/2309.08836v1)  

---


**ABSTRACT**  
Chatbots have been studied for more than half a century. With the rapid development of natural language processing (NLP) technologies in recent years, chatbots using large language models (LLMs) have received much attention nowadays. Compared with traditional ones, modern chatbots are more powerful and have been used in real-world applications. There are however, bias and fairness concerns in modern chatbot design. Due to the huge amounts of training data, extremely large model sizes, and lack of interpretability, bias mitigation and fairness preservation of modern chatbots are challenging. Thus, a comprehensive overview on bias and fairness in chatbot systems is given in this paper. The history of chatbots and their categories are first reviewed. Then, bias sources and potential harms in applications are analyzed. Considerations in designing fair and unbiased chatbot systems are examined. Finally, future research directions are discussed.

{{</citation>}}


### (30/67) SLIDE: Reference-free Evaluation for Machine Translation using a Sliding Document Window (Vikas Raunak et al., 2023)

{{<citation>}}

Vikas Raunak, Tom Kocmi, Matt Post. (2023)  
**SLIDE: Reference-free Evaluation for Machine Translation using a Sliding Document Window**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2309.08832v1)  

---


**ABSTRACT**  
Reference-based metrics that operate at the sentence level typically outperform quality estimation metrics, which have access only to the source and system output. This is unsurprising, since references resolve ambiguities that may be present in the source. We investigate whether additional source context can effectively substitute for a reference. We present a metric, SLIDE (SLiding Document Evaluator), which operates on blocks of sentences using a window that slides over each document in the test set, feeding each chunk into an unmodified, off-the-shelf quality estimation model. We find that SLIDE obtains significantly higher pairwise system accuracy than its sentence-level baseline, in some cases even eliminating the gap with reference-base metrics. This suggests that source context may provide the same information as a human reference.

{{</citation>}}


### (31/67) S3-DST: Structured Open-Domain Dialogue Segmentation and State Tracking in the Era of LLMs (Sarkar Snigdha Sarathi Das et al., 2023)

{{<citation>}}

Sarkar Snigdha Sarathi Das, Chirag Shah, Mengting Wan, Jennifer Neville, Longqi Yang, Reid Andersen, Georg Buscher, Tara Safavi. (2023)  
**S3-DST: Structured Open-Domain Dialogue Segmentation and State Tracking in the Era of LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2309.08827v1)  

---


**ABSTRACT**  
The traditional Dialogue State Tracking (DST) problem aims to track user preferences and intents in user-agent conversations. While sufficient for task-oriented dialogue systems supporting narrow domain applications, the advent of Large Language Model (LLM)-based chat systems has introduced many real-world intricacies in open-domain dialogues. These intricacies manifest in the form of increased complexity in contextual interactions, extended dialogue sessions encompassing a diverse array of topics, and more frequent contextual shifts. To handle these intricacies arising from evolving LLM-based chat systems, we propose joint dialogue segmentation and state tracking per segment in open-domain dialogue systems. Assuming a zero-shot setting appropriate to a true open-domain dialogue system, we propose S3-DST, a structured prompting technique that harnesses Pre-Analytical Recollection, a novel grounding mechanism we designed for improving long context tracking. To demonstrate the efficacy of our proposed approach in joint segmentation and state tracking, we evaluate S3-DST on a proprietary anonymized open-domain dialogue dataset, as well as publicly available DST and segmentation datasets. Across all datasets and settings, S3-DST consistently outperforms the state-of-the-art, demonstrating its potency and robustness the next generation of LLM-based chat systems.

{{</citation>}}


## cs.SD (3)



### (32/67) Enhancing GAN-Based Vocoders with Contrastive Learning Under Data-limited Condition (Haoming Guo et al., 2023)

{{<citation>}}

Haoming Guo, Seth Z. Zhao, Jiachen Lian, Gopala Anumanchipalli, Gerald Friedland. (2023)  
**Enhancing GAN-Based Vocoders with Contrastive Learning Under Data-limited Condition**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.09088v1)  

---


**ABSTRACT**  
Vocoder models have recently achieved substantial progress in generating authentic audio comparable to human quality while significantly reducing memory requirement and inference time. However, these data-hungry generative models require large-scale audio data for learning good representations. In this paper, we apply contrastive learning methods in training the vocoder to improve the perceptual quality of the vocoder without modifying its architecture or adding more data. We design an auxiliary task with mel-spectrogram contrastive learning to enhance the utterance-level quality of the vocoder model under data-limited conditions. We also extend the task to include waveforms to improve the multi-modality comprehension of the model and address the discriminator overfitting problem. We optimize the additional task simultaneously with GAN training objectives. Our result shows that the tasks improve model performance substantially in data-limited settings. Our analysis based on the result indicates that the proposed design successfully alleviates discriminator overfitting and produces audio of higher fidelity.

{{</citation>}}


### (33/67) Music Generation based on Generative Adversarial Networks with Transformer (Ziyi Jiang et al., 2023)

{{<citation>}}

Ziyi Jiang, Yi Zhong, Ruoxue Wu, Zhenghan Chen, Xiaoxuan Liang. (2023)  
**Music Generation based on Generative Adversarial Networks with Transformer**  

---
Primary Category: cs.SD  
Categories: cs-MM, cs-SD, cs.SD, eess-AS, eess-SP  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.09075v1)  

---


**ABSTRACT**  
Autoregressive models based on Transformers have become the prevailing approach for generating music compositions that exhibit comprehensive musical structure. These models are typically trained by minimizing the negative log-likelihood (NLL) of the observed sequence in an autoregressive manner. However, when generating long sequences, the quality of samples from these models tends to significantly deteriorate due to exposure bias. To address this issue, we leverage classifiers trained to differentiate between real and sampled sequences to identify these failures. This observation motivates our exploration of adversarial losses as a complement to the NLL objective. We employ a pre-trained Span-BERT model as the discriminator in the Generative Adversarial Network (GAN) framework, which enhances training stability in our experiments. To optimize discrete sequences within the GAN framework, we utilize the Gumbel-Softmax trick to obtain a differentiable approximation of the sampling process. Additionally, we partition the sequences into smaller chunks to ensure that memory constraints are met. Through human evaluations and the introduction of a novel discriminative metric, we demonstrate that our approach outperforms a baseline model trained solely on likelihood maximization.

{{</citation>}}


### (34/67) FastGraphTTS: An Ultrafast Syntax-Aware Speech Synthesis Framework (Jianzong Wang et al., 2023)

{{<citation>}}

Jianzong Wang, Xulong Zhang, Aolan Sun, Ning Cheng, Jing Xiao. (2023)  
**FastGraphTTS: An Ultrafast Syntax-Aware Speech Synthesis Framework**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.08837v1)  

---


**ABSTRACT**  
This paper integrates graph-to-sequence into an end-to-end text-to-speech framework for syntax-aware modelling with syntactic information of input text. Specifically, the input text is parsed by a dependency parsing module to form a syntactic graph. The syntactic graph is then encoded by a graph encoder to extract the syntactic hidden information, which is concatenated with phoneme embedding and input to the alignment and flow-based decoding modules to generate the raw audio waveform. The model is experimented on two languages, English and Mandarin, using single-speaker, few samples of target speakers, and multi-speaker datasets, respectively. Experimental results show better prosodic consistency performance between input text and generated audio, and also get higher scores in the subjective prosodic evaluation, and show the ability of voice conversion. Besides, the efficiency of the model is largely boosted through the design of the AI chip operator with 5x acceleration.

{{</citation>}}


## cs.LG (8)



### (35/67) Test-Time Compensated Representation Learning for Extreme Traffic Forecasting (Zhiwei Zhang et al., 2023)

{{<citation>}}

Zhiwei Zhang, Weizhong Zhang, Yaowei Huang, Kani Chen. (2023)  
**Test-Time Compensated Representation Learning for Extreme Traffic Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.09074v1)  

---


**ABSTRACT**  
Traffic forecasting is a challenging task due to the complex spatio-temporal correlations among traffic series. In this paper, we identify an underexplored problem in multivariate traffic series prediction: extreme events. Road congestion and rush hours can result in low correlation in vehicle speeds at various intersections during adjacent time periods. Existing methods generally predict future series based on recent observations and entirely discard training data during the testing phase, rendering them unreliable for forecasting highly nonlinear multivariate time series. To tackle this issue, we propose a test-time compensated representation learning framework comprising a spatio-temporal decomposed data bank and a multi-head spatial transformer model (CompFormer). The former component explicitly separates all training data along the temporal dimension according to periodicity characteristics, while the latter component establishes a connection between recent observations and historical series in the data bank through a spatial attention matrix. This enables the CompFormer to transfer robust features to overcome anomalous events while using fewer computational resources. Our modules can be flexibly integrated with existing forecasting methods through end-to-end training, and we demonstrate their effectiveness on the METR-LA and PEMS-BAY benchmarks. Extensive experimental results show that our method is particularly important in extreme events, and can achieve significant improvements over six strong baselines, with an overall improvement of up to 28.2%.

{{</citation>}}


### (36/67) Enhancing personalised thermal comfort models with Active Learning for improved HVAC controls (Zeynep Duygu Tekler et al., 2023)

{{<citation>}}

Zeynep Duygu Tekler, Yue Lei, Xilei Dai, Adrian Chong. (2023)  
**Enhancing personalised thermal comfort models with Active Learning for improved HVAC controls**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2309.09073v1)  

---


**ABSTRACT**  
Developing personalised thermal comfort models to inform occupant-centric controls (OCC) in buildings requires collecting large amounts of real-time occupant preference data. This process can be highly intrusive and labour-intensive for large-scale implementations, limiting the practicality of real-world OCC implementations. To address this issue, this study proposes a thermal preference-based HVAC control framework enhanced with Active Learning (AL) to address the data challenges related to real-world implementations of such OCC systems. The proposed AL approach proactively identifies the most informative thermal conditions for human annotation and iteratively updates a supervised thermal comfort model. The resulting model is subsequently used to predict the occupants' thermal preferences under different thermal conditions, which are integrated into the building's HVAC controls. The feasibility of our proposed AL-enabled OCC was demonstrated in an EnergyPlus simulation of a real-world testbed supplemented with the thermal preference data of 58 study occupants. The preliminary results indicated a significant reduction in overall labelling effort (i.e., 31.0%) between our AL-enabled OCC and conventional OCC while still achieving a slight increase in energy savings (i.e., 1.3%) and thermal satisfaction levels above 98%. This result demonstrates the potential for deploying such systems in future real-world implementations, enabling personalised comfort and energy-efficient building operations.

{{</citation>}}


### (37/67) Recovering Missing Node Features with Local Structure-based Embeddings (Victor M. Tenorio et al., 2023)

{{<citation>}}

Victor M. Tenorio, Madeline Navarro, Santiago Segarra, Antonio G. Marques. (2023)  
**Recovering Missing Node Features with Local Structure-based Embeddings**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.09068v1)  

---


**ABSTRACT**  
Node features bolster graph-based learning when exploited jointly with network structure. However, a lack of nodal attributes is prevalent in graph data. We present a framework to recover completely missing node features for a set of graphs, where we only know the signals of a subset of graphs. Our approach incorporates prior information from both graph topology and existing nodal values. We demonstrate an example implementation of our framework where we assume that node features depend on local graph structure. Missing nodal values are estimated by aggregating known features from the most similar nodes. Similarity is measured through a node embedding space that preserves local topological features, which we train using a Graph AutoEncoder. We empirically show not only the accuracy of our feature estimation approach but also its value for downstream graph classification. Our success embarks on and implies the need to emphasize the relationship between node features and graph structure in graph-based learning.

{{</citation>}}


### (38/67) Improve Deep Forest with Learnable Layerwise Augmentation Policy Schedule (Hongyu Zhu et al., 2023)

{{<citation>}}

Hongyu Zhu, Sichu Liang, Wentao Hu, Fang-Qi Li, Yali yuan, Shi-Lin Wang, Guang Cheng. (2023)  
**Improve Deep Forest with Learnable Layerwise Augmentation Policy Schedule**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.09030v1)  

---


**ABSTRACT**  
As a modern ensemble technique, Deep Forest (DF) employs a cascading structure to construct deep models, providing stronger representational power compared to traditional decision forests. However, its greedy multi-layer learning procedure is prone to overfitting, limiting model effectiveness and generalizability. This paper presents an optimized Deep Forest, featuring learnable, layerwise data augmentation policy schedules. Specifically, We introduce the Cut Mix for Tabular data (CMT) augmentation technique to mitigate overfitting and develop a population-based search algorithm to tailor augmentation intensity for each layer. Additionally, we propose to incorporate outputs from intermediate layers into a checkpoint ensemble for more stable performance. Experimental results show that our method sets new state-of-the-art (SOTA) benchmarks in various tabular classification tasks, outperforming shallow tree ensembles, deep forests, deep neural network, and AutoML competitors. The learned policies also transfer effectively to Deep Forest variants, underscoring its potential for enhancing non-differentiable deep learning modules in tabular signal processing.

{{</citation>}}


### (39/67) gym-saturation: Gymnasium environments for saturation provers (System description) (Boris Shminke, 2023)

{{<citation>}}

Boris Shminke. (2023)  
**gym-saturation: Gymnasium environments for saturation provers (System description)**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09022v1)  

---


**ABSTRACT**  
This work describes a new version of a previously published Python package - gym-saturation: a collection of OpenAI Gym environments for guiding saturation-style provers based on the given clause algorithm with reinforcement learning. We contribute usage examples with two different provers: Vampire and iProver. We also have decoupled the proof state representation from reinforcement learning per se and provided examples of using a known ast2vec Python code embedding model as a first-order logic representation. In addition, we demonstrate how environment wrappers can transform a prover into a problem similar to a multi-armed bandit. We applied two reinforcement learning algorithms (Thompson sampling and Proximal policy optimisation) implemented in Ray RLlib to show the ease of experimentation with the new release of our package.

{{</citation>}}


### (40/67) UNIDEAL: Curriculum Knowledge Distillation Federated Learning (Yuwen Yang et al., 2023)

{{<citation>}}

Yuwen Yang, Chang Liu, Xun Cai, Suizhi Huang, Hongtao Lu, Yue Ding. (2023)  
**UNIDEAL: Curriculum Knowledge Distillation Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-DC, cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.08961v1)  

---


**ABSTRACT**  
Federated Learning (FL) has emerged as a promising approach to enable collaborative learning among multiple clients while preserving data privacy. However, cross-domain FL tasks, where clients possess data from different domains or distributions, remain a challenging problem due to the inherent heterogeneity. In this paper, we present UNIDEAL, a novel FL algorithm specifically designed to tackle the challenges of cross-domain scenarios and heterogeneous model architectures. The proposed method introduces Adjustable Teacher-Student Mutual Evaluation Curriculum Learning, which significantly enhances the effectiveness of knowledge distillation in FL settings. We conduct extensive experiments on various datasets, comparing UNIDEAL with state-of-the-art baselines. Our results demonstrate that UNIDEAL achieves superior performance in terms of both model accuracy and communication efficiency. Additionally, we provide a convergence analysis of the algorithm, showing a convergence rate of O(1/T) under non-convex conditions.

{{</citation>}}


### (41/67) DOMAIN: MilDly COnservative Model-BAsed OfflINe Reinforcement Learning (Xiao-Yin Liu et al., 2023)

{{<citation>}}

Xiao-Yin Liu, Xiao-Hu Zhou, Xiao-Liang Xie, Shi-Qi Liu, Zhen-Qiu Feng, Hao Li, Mei-Jiang Gui, Tian-Yu Xiang, De-Xing Huang, Zeng-Guang Hou. (2023)  
**DOMAIN: MilDly COnservative Model-BAsed OfflINe Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.08925v1)  

---


**ABSTRACT**  
Model-based reinforcement learning (RL), which learns environment model from offline dataset and generates more out-of-distribution model data, has become an effective approach to the problem of distribution shift in offline RL. Due to the gap between the learned and actual environment, conservatism should be incorporated into the algorithm to balance accurate offline data and imprecise model data. The conservatism of current algorithms mostly relies on model uncertainty estimation. However, uncertainty estimation is unreliable and leads to poor performance in certain scenarios, and the previous methods ignore differences between the model data, which brings great conservatism. Therefore, this paper proposes a milDly cOnservative Model-bAsed offlINe RL algorithm (DOMAIN) without estimating model uncertainty to address the above issues. DOMAIN introduces adaptive sampling distribution of model samples, which can adaptively adjust the model data penalty. In this paper, we theoretically demonstrate that the Q value learned by the DOMAIN outside the region is a lower bound of the true Q value, the DOMAIN is less conservative than previous model-based offline RL algorithms and has the guarantee of security policy improvement. The results of extensive experiments show that DOMAIN outperforms prior RL algorithms on the D4RL dataset benchmark, and achieves better performance than other RL algorithms on tasks that require generalization.

{{</citation>}}


### (42/67) Rethinking Learning Rate Tuning in the Era of Large Language Models (Hongpeng Jin et al., 2023)

{{<citation>}}

Hongpeng Jin, Wenqi Wei, Xuyu Wang, Wenbin Zhang, Yanzhao Wu. (2023)  
**Rethinking Learning Rate Tuning in the Era of Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.08859v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) represent the recent success of deep learning in achieving remarkable human-like predictive performance. It has become a mainstream strategy to leverage fine-tuning to adapt LLMs for various real-world applications due to the prohibitive expenses associated with LLM training. The learning rate is one of the most important hyperparameters in LLM fine-tuning with direct impacts on both fine-tuning efficiency and fine-tuned LLM quality. Existing learning rate policies are primarily designed for training traditional deep neural networks (DNNs), which may not work well for LLM fine-tuning. We reassess the research challenges and opportunities of learning rate tuning in the coming era of Large Language Models. This paper makes three original contributions. First, we revisit existing learning rate policies to analyze the critical challenges of learning rate tuning in the era of LLMs. Second, we present LRBench++ to benchmark learning rate policies and facilitate learning rate tuning for both traditional DNNs and LLMs. Third, our experimental analysis with LRBench++ demonstrates the key differences between LLM fine-tuning and traditional DNN training and validates our analysis.

{{</citation>}}


## cs.CV (7)



### (43/67) MMST-ViT: Climate Change-aware Crop Yield Prediction via Multi-Modal Spatial-Temporal Vision Transformer (Fudong Lin et al., 2023)

{{<citation>}}

Fudong Lin, Summer Crawford, Kaleb Guillot, Yihe Zhang, Yan Chen, Xu Yuan, Li Chen, Shelby Willams, Robert Minvielle, Xiangming Xiao, Drew Gholson, Nicolas Ashwell, Tri Setiyono, Brenda Tubana, Lu Peng, Magdy Bayoumi, Nian-Feng Tzeng. (2023)  
**MMST-ViT: Climate Change-aware Crop Yield Prediction via Multi-Modal Spatial-Temporal Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.09067v1)  

---


**ABSTRACT**  
Precise crop yield prediction provides valuable information for agricultural planning and decision-making processes. However, timely predicting crop yields remains challenging as crop growth is sensitive to growing season weather variation and climate change. In this work, we develop a deep learning-based solution, namely Multi-Modal Spatial-Temporal Vision Transformer (MMST-ViT), for predicting crop yields at the county level across the United States, by considering the effects of short-term meteorological variations during the growing season and the long-term climate change on crops. Specifically, our MMST-ViT consists of a Multi-Modal Transformer, a Spatial Transformer, and a Temporal Transformer. The Multi-Modal Transformer leverages both visual remote sensing data and short-term meteorological data for modeling the effect of growing season weather variations on crop growth. The Spatial Transformer learns the high-resolution spatial dependency among counties for accurate agricultural tracking. The Temporal Transformer captures the long-range temporal dependency for learning the impact of long-term climate change on crops. Meanwhile, we also devise a novel multi-modal contrastive learning technique to pre-train our model without extensive human supervision. Hence, our MMST-ViT captures the impacts of both short-term weather variations and long-term climate change on crops by leveraging both satellite images and meteorological data. We have conducted extensive experiments on over 200 counties in the United States, with the experimental results exhibiting that our MMST-ViT outperforms its counterparts under three performance metrics of interest.

{{</citation>}}


### (44/67) RingMo-lite: A Remote Sensing Multi-task Lightweight Network with CNN-Transformer Hybrid Framework (Yuelei Wang et al., 2023)

{{<citation>}}

Yuelei Wang, Ting Zhang, Liangjin Zhao, Lin Hu, Zhechao Wang, Ziqing Niu, Peirui Cheng, Kaiqiang Chen, Xuan Zeng, Zhirui Wang, Hongqi Wang, Xian Sun. (2023)  
**RingMo-lite: A Remote Sensing Multi-task Lightweight Network with CNN-Transformer Hybrid Framework**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.09003v1)  

---


**ABSTRACT**  
In recent years, remote sensing (RS) vision foundation models such as RingMo have emerged and achieved excellent performance in various downstream tasks. However, the high demand for computing resources limits the application of these models on edge devices. It is necessary to design a more lightweight foundation model to support on-orbit RS image interpretation. Existing methods face challenges in achieving lightweight solutions while retaining generalization in RS image interpretation. This is due to the complex high and low-frequency spectral components in RS images, which make traditional single CNN or Vision Transformer methods unsuitable for the task. Therefore, this paper proposes RingMo-lite, an RS multi-task lightweight network with a CNN-Transformer hybrid framework, which effectively exploits the frequency-domain properties of RS to optimize the interpretation process. It is combined by the Transformer module as a low-pass filter to extract global features of RS images through a dual-branch structure, and the CNN module as a stacked high-pass filter to extract fine-grained details effectively. Furthermore, in the pretraining stage, the designed frequency-domain masked image modeling (FD-MIM) combines each image patch's high-frequency and low-frequency characteristics, effectively capturing the latent feature representation in RS data. As shown in Fig. 1, compared with RingMo, the proposed RingMo-lite reduces the parameters over 60% in various RS image interpretation tasks, the average accuracy drops by less than 2% in most of the scenes and achieves SOTA performance compared to models of the similar size. In addition, our work will be integrated into the MindSpore computing platform in the near future.

{{</citation>}}


### (45/67) Robust Backdoor Attacks on Object Detection in Real World (Yaguan Qian et al., 2023)

{{<citation>}}

Yaguan Qian, Boyuan Ji, Shuke He, Shenhui Huang, Xiang Ling, Bin Wang, Wei Wang. (2023)  
**Robust Backdoor Attacks on Object Detection in Real World**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.08953v1)  

---


**ABSTRACT**  
Deep learning models are widely deployed in many applications, such as object detection in various security fields. However, these models are vulnerable to backdoor attacks. Most backdoor attacks were intensively studied on classified models, but little on object detection. Previous works mainly focused on the backdoor attack in the digital world, but neglect the real world. Especially, the backdoor attack's effect in the real world will be easily influenced by physical factors like distance and illumination. In this paper, we proposed a variable-size backdoor trigger to adapt to the different sizes of attacked objects, overcoming the disturbance caused by the distance between the viewing point and attacked object. In addition, we proposed a backdoor training named malicious adversarial training, enabling the backdoor object detector to learn the feature of the trigger with physical noise. The experiment results show this robust backdoor attack (RBA) could enhance the attack success rate in the real world.

{{</citation>}}


### (46/67) Semantics-aware LiDAR-Only Pseudo Point Cloud Generation for 3D Object Detection (Tiago Cortinhal et al., 2023)

{{<citation>}}

Tiago Cortinhal, Idriss Gouigah, Eren Erdal Aksoy. (2023)  
**Semantics-aware LiDAR-Only Pseudo Point Cloud Generation for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.08932v1)  

---


**ABSTRACT**  
Although LiDAR sensors are crucial for autonomous systems due to providing precise depth information, they struggle with capturing fine object details, especially at a distance, due to sparse and non-uniform data. Recent advances introduced pseudo-LiDAR, i.e., synthetic dense point clouds, using additional modalities such as cameras to enhance 3D object detection. We present a novel LiDAR-only framework that augments raw scans with denser pseudo point clouds by solely relying on LiDAR sensors and scene semantics, omitting the need for cameras. Our framework first utilizes a segmentation model to extract scene semantics from raw point clouds, and then employs a multi-modal domain translator to generate synthetic image segments and depth cues without real cameras. This yields a dense pseudo point cloud enriched with semantic information. We also introduce a new semantically guided projection method, which enhances detection performance by retaining only relevant pseudo points. We applied our framework to different advanced 3D object detection methods and reported up to 2.9% performance upgrade. We also obtained comparable results on the KITTI 3D object detection dataset, in contrast to other state-of-the-art LiDAR-only detectors.

{{</citation>}}


### (47/67) In-Style: Bridging Text and Uncurated Videos with Style Transfer for Text-Video Retrieval (Nina Shvetsova et al., 2023)

{{<citation>}}

Nina Shvetsova, Anna Kukleva, Bernt Schiele, Hilde Kuehne. (2023)  
**In-Style: Bridging Text and Uncurated Videos with Style Transfer for Text-Video Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2309.08928v1)  

---


**ABSTRACT**  
Large-scale noisy web image-text datasets have been proven to be efficient for learning robust vision-language models. However, when transferring them to the task of video retrieval, models still need to be fine-tuned on hand-curated paired text-video data to adapt to the diverse styles of video descriptions. To address this problem without the need for hand-annotated pairs, we propose a new setting, text-video retrieval with uncurated & unpaired data, that during training utilizes only text queries together with uncurated web videos without any paired text-video data. To this end, we propose an approach, In-Style, that learns the style of the text queries and transfers it to uncurated web videos. Moreover, to improve generalization, we show that one model can be trained with multiple text styles. To this end, we introduce a multi-style contrastive training procedure that improves the generalizability over several datasets simultaneously. We evaluate our model on retrieval performance over multiple datasets to demonstrate the advantages of our style transfer framework on the new task of uncurated & unpaired text-video retrieval and improve state-of-the-art performance on zero-shot text-video retrieval.

{{</citation>}}


### (48/67) GCL: Gradient-Guided Contrastive Learning for Medical Image Segmentation with Multi-Perspective Meta Labels (Yixuan Wu et al., 2023)

{{<citation>}}

Yixuan Wu, Jintai Chen, Jiahuan Yan, Yiheng Zhu, Danny Z. Chen, Jian Wu. (2023)  
**GCL: Gradient-Guided Contrastive Learning for Medical Image Segmentation with Multi-Perspective Meta Labels**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.08888v1)  

---


**ABSTRACT**  
Since annotating medical images for segmentation tasks commonly incurs expensive costs, it is highly desirable to design an annotation-efficient method to alleviate the annotation burden. Recently, contrastive learning has exhibited a great potential in learning robust representations to boost downstream tasks with limited labels. In medical imaging scenarios, ready-made meta labels (i.e., specific attribute information of medical images) inherently reveal semantic relationships among images, which have been used to define positive pairs in previous work. However, the multi-perspective semantics revealed by various meta labels are usually incompatible and can incur intractable "semantic contradiction" when combining different meta labels. In this paper, we tackle the issue of "semantic contradiction" in a gradient-guided manner using our proposed Gradient Mitigator method, which systematically unifies multi-perspective meta labels to enable a pre-trained model to attain a better high-level semantic recognition ability. Moreover, we emphasize that the fine-grained discrimination ability is vital for segmentation-oriented pre-training, and develop a novel method called Gradient Filter to dynamically screen pixel pairs with the most discriminating power based on the magnitude of gradients. Comprehensive experiments on four medical image segmentation datasets verify that our new method GCL: (1) learns informative image representations and considerably boosts segmentation performance with limited labels, and (2) shows promising generalizability on out-of-distribution datasets.

{{</citation>}}


### (49/67) Enhancing Visual Perception in Novel Environments via Incremental Data Augmentation Based on Style Transfer (Abhibha Gupta et al., 2023)

{{<citation>}}

Abhibha Gupta, Rully Agus Hendrawan, Mansur Arief. (2023)  
**Enhancing Visual Perception in Novel Environments via Incremental Data Augmentation Based on Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Augmentation, Style Transfer  
[Paper Link](http://arxiv.org/abs/2309.08851v1)  

---


**ABSTRACT**  
The deployment of autonomous agents in real-world scenarios is challenged by "unknown unknowns", i.e. novel unexpected environments not encountered during training, such as degraded signs. While existing research focuses on anomaly detection and class imbalance, it often fails to address truly novel scenarios. Our approach enhances visual perception by leveraging the Variational Prototyping Encoder (VPE) to adeptly identify and handle novel inputs, then incrementally augmenting data using neural style transfer to enrich underrepresented data. By comparing models trained solely on original datasets with those trained on a combination of original and augmented datasets, we observed a notable improvement in the performance of the latter. This underscores the critical role of data augmentation in enhancing model robustness. Our findings suggest the potential benefits of incorporating generative models for domain-specific augmentation strategies.

{{</citation>}}


## cs.DS (1)



### (50/67) Fast Triangle Counting (David A. Bader, 2023)

{{<citation>}}

David A. Bader. (2023)  
**Fast Triangle Counting**  

---
Primary Category: cs.DS  
Categories: cs-DC, cs-DS, cs.DS  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2309.09064v1)  

---


**ABSTRACT**  
Listing and counting triangles in graphs is a key algorithmic kernel for network analyses including community detection, clustering coefficients, k-trusses, and triangle centrality. We design and implement a new serial algorithm for triangle counting that performs competitively with the fastest previous approaches on both real and synthetic graphs, such as those from the Graph500 Benchmark and the MIT/Amazon/IEEE Graph Challenge. The experimental results use the recently-launched Intel Xeon Platinum 8480+ and CPU Max 9480 processors.

{{</citation>}}


## cs.HC (1)



### (51/67) Generative AI-Driven Storytelling: A New Era for Marketing (Marko Vidrih et al., 2023)

{{<citation>}}

Marko Vidrih, Shiva Mayahi. (2023)  
**Generative AI-Driven Storytelling: A New Era for Marketing**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-MM, cs.HC  
Keywords: AI, Generative AI, Google  
[Paper Link](http://arxiv.org/abs/2309.09048v1)  

---


**ABSTRACT**  
This paper delves into the transformative power of Generative AI-driven storytelling in the realm of marketing. Generative AI, distinct from traditional machine learning, offers the capability to craft narratives that resonate with consumers on a deeply personal level. Through real-world examples from industry leaders like Google, Netflix and Stitch Fix, we elucidate how this technology shapes marketing strategies, personalizes consumer experiences, and navigates the challenges it presents. The paper also explores future directions and recommendations for generative AI-driven storytelling, including prospective applications such as real-time personalized storytelling, immersive storytelling experiences, and social media storytelling. By shedding light on the potential and impact of generative AI-driven storytelling in marketing, this paper contributes to the understanding of this cutting-edge approach and its transformative power in the field of marketing.

{{</citation>}}


## cs.CR (1)



### (52/67) Efficient Privacy-Preserving Convolutional Spiking Neural Networks with FHE (Pengbo Li et al., 2023)

{{<citation>}}

Pengbo Li, Huifang Huang, Ting Gao, Jin Guo, Jinqiao Duan. (2023)  
**Efficient Privacy-Preserving Convolutional Spiking Neural Networks with FHE**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CY, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09025v1)  

---


**ABSTRACT**  
With the rapid development of AI technology, we have witnessed numerous innovations and conveniences. However, along with these advancements come privacy threats and risks. Fully Homomorphic Encryption (FHE) emerges as a key technology for privacy-preserving computation, enabling computations while maintaining data privacy. Nevertheless, FHE has limitations in processing continuous non-polynomial functions as it is restricted to discrete integers and supports only addition and multiplication. Spiking Neural Networks (SNNs) operate on discrete spike signals, naturally aligning with the properties of FHE. In this paper, we present a framework called FHE-DiCSNN. This framework is based on the efficient TFHE scheme and leverages the discrete properties of SNNs to achieve high prediction performance on ciphertexts. Firstly, by employing bootstrapping techniques, we successfully implement computations of the Leaky Integrate-and-Fire neuron model on ciphertexts. Through bootstrapping, we can facilitate computations for SNNs of arbitrary depth. This framework can be extended to other spiking neuron models, providing a novel framework for the homomorphic evaluation of SNNs. Secondly, inspired by CNNs, we adopt convolutional methods to replace Poisson encoding. This not only enhances accuracy but also mitigates the issue of prolonged simulation time caused by random encoding. Furthermore, we employ engineering techniques to parallelize the computation of bootstrapping, resulting in a significant improvement in computational efficiency. Finally, we evaluate our model on the MNIST dataset. Experimental results demonstrate that, with the optimal parameter configuration, FHE-DiCSNN achieves an accuracy of 97.94% on ciphertexts, with a loss of only 0.53% compared to the original network's accuracy of 98.47%. Moreover, each prediction requires only 0.75 seconds of computation time

{{</citation>}}


## cs.IR (1)



### (53/67) Bridging Dense and Sparse Maximum Inner Product Search (Sebastian Bruch et al., 2023)

{{<citation>}}

Sebastian Bruch, Franco Maria Nardini, Amir Ingber, Edo Liberty. (2023)  
**Bridging Dense and Sparse Maximum Inner Product Search**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2309.09013v1)  

---


**ABSTRACT**  
Maximum inner product search (MIPS) over dense and sparse vectors have progressed independently in a bifurcated literature for decades; the latter is better known as top-$k$ retrieval in Information Retrieval. This duality exists because sparse and dense vectors serve different end goals. That is despite the fact that they are manifestations of the same mathematical problem. In this work, we ask if algorithms for dense vectors could be applied effectively to sparse vectors, particularly those that violate the assumptions underlying top-$k$ retrieval methods. We study IVF-based retrieval where vectors are partitioned into clusters and only a fraction of clusters are searched during retrieval. We conduct a comprehensive analysis of dimensionality reduction for sparse vectors, and examine standard and spherical KMeans for partitioning. Our experiments demonstrate that IVF serves as an efficient solution for sparse MIPS. As byproducts, we identify two research opportunities and demonstrate their potential. First, we cast the IVF paradigm as a dynamic pruning technique and turn that insight into a novel organization of the inverted index for approximate MIPS for general sparse vectors. Second, we offer a unified regime for MIPS over vectors that have dense and sparse subspaces, and show its robustness to query distributions.

{{</citation>}}


## cs.MM (1)



### (54/67) Invertible Mosaic Image Hiding Network for Very Large Capacity Image Steganography (Zihan Chen et al., 2023)

{{<citation>}}

Zihan Chen, Tianrui Liu, Jun-Jie Huang, Wentao Zhao, Xing Bi, Meng Wang. (2023)  
**Invertible Mosaic Image Hiding Network for Very Large Capacity Image Steganography**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.08987v1)  

---


**ABSTRACT**  
The existing image steganography methods either sequentially conceal secret images or conceal a concatenation of multiple images. In such ways, the interference of information among multiple images will become increasingly severe when the number of secret images becomes larger, thus restrict the development of very large capacity image steganography. In this paper, we propose an Invertible Mosaic Image Hiding Network (InvMIHNet) which realizes very large capacity image steganography with high quality by concealing a single mosaic secret image. InvMIHNet consists of an Invertible Image Rescaling (IIR) module and an Invertible Image Hiding (IIH) module. The IIR module works for downscaling the single mosaic secret image form by spatially splicing the multiple secret images, and the IIH module then conceal this mosaic image under the cover image. The proposed InvMIHNet successfully conceal and reveal up to 16 secret images with a small number of parameters and memory consumption. Extensive experiments on ImageNet-1K, COCO and DIV2K show InvMIHNet outperforms state-of-the-art methods in terms of both the imperceptibility of stego image and recover accuracy of secret image.

{{</citation>}}


## cs.AI (6)



### (55/67) Accelerating In-Browser Deep Learning Inference on Diverse Edge Clients through Just-in-Time Kernel Optimizations (Fucheng Jia et al., 2023)

{{<citation>}}

Fucheng Jia, Shiqi Jiang, Ting Cao, Wei Cui, Tianrui Xia, Xu Cao, Yuanchun Li, Deyu Zhang, Ju Ren, Yunxin Liu, Lili Qiu, Mao Yang. (2023)  
**Accelerating In-Browser Deep Learning Inference on Diverse Edge Clients through Just-in-Time Kernel Optimizations**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.08978v1)  

---


**ABSTRACT**  
Web applications are increasingly becoming the primary platform for AI service delivery, making in-browser deep learning (DL) inference more prominent. However, current in-browser inference systems fail to effectively utilize advanced web programming techniques and customize kernels for various client devices, leading to suboptimal performance.   To address the issues, this paper presents the first in-browser inference system, nn-JIT.web, which enables just-in-time (JIT) auto-generation of optimized kernels for both CPUs and GPUs during inference. The system achieves this by using two novel web programming techniques that can significantly reduce kernel generation time, compared to other tensor compilers such as TVM, while maintaining or even improving performance. The first technique, Tensor-Web Compiling Co-Design, lowers compiling costs by unifying tensor and web compiling and eliminating redundant and ineffective compiling passes. The second technique, Web-Specific Lite Kernel Optimization Space Design, reduces kernel tuning costs by focusing on web programming requirements and efficient hardware resource utilization, limiting the optimization space to only dozens.   nn-JIT.web is evaluated for modern transformer models on a range of client devices, including the mainstream CPUs and GPUs from ARM, Intel, AMD and Nvidia. Results show that nn-JIT.web can achieve up to 8.2x faster within 30 seconds compared to the baselines across various models.

{{</citation>}}


### (56/67) Multiagent Reinforcement Learning with an Attention Mechanism for Improving Energy Efficiency in LoRa Networks (Xu Zhang et al., 2023)

{{<citation>}}

Xu Zhang, Ziqi Lin, Shimin Gong, Bo Gu, Dusit Niyato. (2023)  
**Multiagent Reinforcement Learning with an Attention Mechanism for Improving Energy Efficiency in LoRa Networks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-MA, cs.AI  
Keywords: Attention, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.08965v1)  

---


**ABSTRACT**  
Long Range (LoRa) wireless technology, characterized by low power consumption and a long communication range, is regarded as one of the enabling technologies for the Industrial Internet of Things (IIoT). However, as the network scale increases, the energy efficiency (EE) of LoRa networks decreases sharply due to severe packet collisions. To address this issue, it is essential to appropriately assign transmission parameters such as the spreading factor and transmission power for each end device (ED). However, due to the sporadic traffic and low duty cycle of LoRa networks, evaluating the system EE performance under different parameter settings is time-consuming. Therefore, we first formulate an analytical model to calculate the system EE. On this basis, we propose a transmission parameter allocation algorithm based on multiagent reinforcement learning (MALoRa) with the aim of maximizing the system EE of LoRa networks. Notably, MALoRa employs an attention mechanism to guide each ED to better learn how much ''attention'' should be given to the parameter assignments for relevant EDs when seeking to improve the system EE. Simulation results demonstrate that MALoRa significantly improves the system EE compared with baseline algorithms with an acceptable degradation in packet delivery rate (PDR).

{{</citation>}}


### (57/67) A Statistical Turing Test for Generative Models (Hayden Helm et al., 2023)

{{<citation>}}

Hayden Helm, Carey E. Priebe, Weiwei Yang. (2023)  
**A Statistical Turing Test for Generative Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.08913v1)  

---


**ABSTRACT**  
The emergence of human-like abilities of AI systems for content generation in domains such as text, audio, and vision has prompted the development of classifiers to determine whether content originated from a human or a machine. Implicit in these efforts is an assumption that the generation properties of a human are different from that of the machine. In this work, we provide a framework in the language of statistical pattern recognition that quantifies the difference between the distributions of human and machine-generated content conditioned on an evaluation context. We describe current methods in the context of the framework and demonstrate how to use the framework to evaluate the progression of generative models towards human-like capabilities, among many axes of analysis.

{{</citation>}}


### (58/67) Solving Satisfiability Modulo Counting for Symbolic and Statistical AI Integration With Provable Guarantees (Jinzhao Li et al., 2023)

{{<citation>}}

Jinzhao Li, Nan Jiang, Yexiang Xue. (2023)  
**Solving Satisfiability Modulo Counting for Symbolic and Statistical AI Integration With Provable Guarantees**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CC, cs-LO, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.08883v1)  

---


**ABSTRACT**  
Satisfiability Modulo Counting (SMC) encompasses problems that require both symbolic decision-making and statistical reasoning. Its general formulation captures many real-world problems at the intersection of symbolic and statistical Artificial Intelligence. SMC searches for policy interventions to control probabilistic outcomes. Solving SMC is challenging because of its highly intractable nature($\text{NP}^{\text{PP}}$-complete), incorporating statistical inference and symbolic reasoning. Previous research on SMC solving lacks provable guarantees and/or suffers from sub-optimal empirical performance, especially when combinatorial constraints are present. We propose XOR-SMC, a polynomial algorithm with access to NP-oracles, to solve highly intractable SMC problems with constant approximation guarantees. XOR-SMC transforms the highly intractable SMC into satisfiability problems, by replacing the model counting in SMC with SAT formulae subject to randomized XOR constraints. Experiments on solving important SMC problems in AI for social good demonstrate that XOR-SMC finds solutions close to the true optimum, outperforming several baselines which struggle to find good approximations for the intractable model counting in SMC.

{{</citation>}}


### (59/67) ChatGPT-4 with Code Interpreter can be used to solve introductory college-level vector calculus and electromagnetism problems (Tanuj Kumar et al., 2023)

{{<citation>}}

Tanuj Kumar, Mikhail A. Kats. (2023)  
**ChatGPT-4 with Code Interpreter can be used to solve introductory college-level vector calculus and electromagnetism problems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CE, cs.AI, physics-ed-ph  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.08881v1)  

---


**ABSTRACT**  
We evaluated ChatGPT 3.5, 4, and 4 with Code Interpreter on a set of college-level engineering-math and electromagnetism problems, such as those often given to sophomore electrical engineering majors. We selected a set of 13 problems, and had ChatGPT solve them multiple times, using a fresh instance (chat) each time. We found that ChatGPT-4 with Code Interpreter was able to satisfactorily solve most problems we tested most of the time -- a major improvement over the performance of ChatGPT-4 (or 3.5) without Code Interpreter. The performance of ChatGPT was observed to be somewhat stochastic, and we found that solving the same problem N times in new ChatGPT instances and taking the most-common answer was an effective strategy. Based on our findings and observations, we provide some recommendations for instructors and students of classes at this level.

{{</citation>}}


### (60/67) GPT as a Baseline for Recommendation Explanation Texts (Joyce Zhou et al., 2023)

{{<citation>}}

Joyce Zhou, Thorsten Joachims. (2023)  
**GPT as a Baseline for Recommendation Explanation Texts**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2309.08817v1)  

---


**ABSTRACT**  
In this work, we establish a baseline potential for how modern model-generated text explanations of movie recommendations may help users, and explore what different components of these text explanations that users like or dislike, especially in contrast to existing human movie reviews. We found that participants gave no significantly different rankings between movies, nor did they give significantly different individual quality scores to reviews of movies that they had never seen before. However, participants did mark reviews as significantly better when they were movies they had seen before. We also explore specific aspects of movie review texts that participants marked as important for each quality. Overall, we establish that modern LLMs are a promising source of recommendation explanations, and we intend on further exploring personalizable text explanations in the future.

{{</citation>}}


## cs.AR (1)



### (61/67) Exploration of TPUs for AI Applications (Diego Sanmartín Carrión et al., 2023)

{{<citation>}}

Diego Sanmartín Carrión, Vera Prohaska. (2023)  
**Exploration of TPUs for AI Applications**  

---
Primary Category: cs.AR  
Categories: cs-AI, cs-AR, cs.AR  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2309.08918v1)  

---


**ABSTRACT**  
Tensor Processing Units (TPUs) are specialized hardware accelerators for deep learning developed by Google. This paper explores the performance of TPU with a focus on AI and its implementation in edge computing. It first provides an overview of TPUs, specifically their design in relation to neural networks, their general architecture, compilation techniques and supporting frameworks. Furthermore, we provide a comparative analysis of Cloud and Edge TPU performance against other counterpart chip architectures. It is then discussed how TPUs can be used to speed up AI workloads. The results show that TPUs can provide significant performance improvements both in cloud and edge computing. Additionally, we address the need for further research for the deployment of more architectures in the Edge TPU, as well as the need for the development of more robust comparisons in edge computing.

{{</citation>}}


## cs.IT (1)



### (62/67) CDDM: Channel Denoising Diffusion Models for Wireless Semantic Communications (Tong Wu et al., 2023)

{{<citation>}}

Tong Wu, Zhiyong Chen, Dazhi He, Liang Qian, Yin Xu, Meixia Tao, Wenjun Zhang. (2023)  
**CDDM: Channel Denoising Diffusion Models for Wireless Semantic Communications**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.08895v1)  

---


**ABSTRACT**  
Diffusion models (DM) can gradually learn to remove noise, which have been widely used in artificial intelligence generated content (AIGC) in recent years. The property of DM for eliminating noise leads us to wonder whether DM can be applied to wireless communications to help the receiver mitigate the channel noise. To address this, we propose channel denoising diffusion models (CDDM) for semantic communications over wireless channels in this paper. CDDM can be applied as a new physical layer module after the channel equalization to learn the distribution of the channel input signal, and then utilizes this learned knowledge to remove the channel noise. We derive corresponding training and sampling algorithms of CDDM according to the forward diffusion process specially designed to adapt the channel models and theoretically prove that the well-trained CDDM can effectively reduce the conditional entropy of the received signal under small sampling steps. Moreover, we apply CDDM to a semantic communications system based on joint source-channel coding (JSCC) for image transmission. Extensive experimental results demonstrate that CDDM can further reduce the mean square error (MSE) after minimum mean square error (MMSE) equalizer, and the joint CDDM and JSCC system achieves better performance than the JSCC system and the traditional JPEG2000 with low-density parity-check (LDPC) code approach.

{{</citation>}}


## eess.SY (1)



### (63/67) Data-Driven H-infinity Control with a Real-Time and Efficient Reinforcement Learning Algorithm: An Application to Autonomous Mobility-on-Demand Systems (Ali Aalipour et al., 2023)

{{<citation>}}

Ali Aalipour, Alireza Khani. (2023)  
**Data-Driven H-infinity Control with a Real-Time and Efficient Reinforcement Learning Algorithm: An Application to Autonomous Mobility-on-Demand Systems**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.08880v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) is a class of artificial intelligence algorithms being used to design adaptive optimal controllers through online learning. This paper presents a model-free, real-time, data-efficient Q-learning-based algorithm to solve the H$_{\infty}$ control of linear discrete-time systems. The computational complexity is shown to reduce from $\mathcal{O}(\underline{q}^3)$ in the literature to $\mathcal{O}(\underline{q}^2)$ in the proposed algorithm, where $\underline{q}$ is quadratic in the sum of the size of state variables, control inputs, and disturbance. An adaptive optimal controller is designed and the parameters of the action and critic networks are learned online without the knowledge of the system dynamics, making the proposed algorithm completely model-free. Also, a sufficient probing noise is only needed in the first iteration and does not affect the proposed algorithm. With no need for an initial stabilizing policy, the algorithm converges to the closed-form solution obtained by solving the Riccati equation. A simulation study is performed by applying the proposed algorithm to real-time control of an autonomous mobility-on-demand (AMoD) system for a real-world case study to evaluate the effectiveness of the proposed algorithm.

{{</citation>}}


## eess.AS (2)



### (64/67) Decoder-only Architecture for Speech Recognition with CTC Prompts and Text Data Augmentation (Emiru Tsunoo et al., 2023)

{{<citation>}}

Emiru Tsunoo, Hayato Futami, Yosuke Kashiwagi, Siddhant Arora, Shinji Watanabe. (2023)  
**Decoder-only Architecture for Speech Recognition with CTC Prompts and Text Data Augmentation**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Augmentation, GPT, PaLM, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.08876v1)  

---


**ABSTRACT**  
Collecting audio-text pairs is expensive; however, it is much easier to access text-only data. Unless using shallow fusion, end-to-end automatic speech recognition (ASR) models require architecture modifications or additional training schemes to use text-only data. Inspired by recent advances in decoder-only language models (LMs), such as GPT-3 and PaLM adopted for speech-processing tasks, we propose using a decoder-only architecture for ASR with simple text augmentation. To provide audio information, encoder features compressed by CTC prediction are used as prompts for the decoder, which can be regarded as refining CTC prediction using the decoder-only model. Because the decoder architecture is the same as an autoregressive LM, it is simple to enhance the model by leveraging external text data with LM training. An experimental comparison using LibriSpeech and Switchboard shows that our proposed models with text augmentation training reduced word error rates from ordinary CTC by 0.3% and 1.4% on LibriSpeech test-clean and testother set, respectively, and 2.9% and 5.0% on Switchboard and CallHome. The proposed model had advantage on computational efficiency compared with conventional encoder-decoder ASR models with a similar parameter setup, and outperformed them on the LibriSpeech 100h and Switchboard training scenarios.

{{</citation>}}


### (65/67) Boosting End-to-End Multilingual Phoneme Recognition through Exploiting Universal Speech Attributes Constraints (Hao Yen et al., 2023)

{{<citation>}}

Hao Yen, Sabato Marco Siniscalchi, Chin-Hui Lee. (2023)  
**Boosting End-to-End Multilingual Phoneme Recognition through Exploiting Universal Speech Attributes Constraints**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2309.08828v1)  

---


**ABSTRACT**  
We propose a first step toward multilingual end-to-end automatic speech recognition (ASR) by integrating knowledge about speech articulators. The key idea is to leverage a rich set of fundamental units that can be defined "universally" across all spoken languages, referred to as speech attributes, namely manner and place of articulation. Specifically, several deterministic attribute-to-phoneme mapping matrices are constructed based on the predefined set of universal attribute inventory, which projects the knowledge-rich articulatory attribute logits, into output phoneme logits. The mapping puts knowledge-based constraints to limit inconsistency with acoustic-phonetic evidence in the integrated prediction. Combined with phoneme recognition, our phone recognizer is able to infer from both attribute and phoneme information. The proposed joint multilingual model is evaluated through phoneme recognition. In multilingual experiments over 6 languages on benchmark datasets LibriSpeech and CommonVoice, we find that our proposed solution outperforms conventional multilingual approaches with a relative improvement of 6.85% on average, and it also demonstrates a much better performance compared to monolingual model. Further analysis conclusively demonstrates that the proposed solution eliminates phoneme predictions that are inconsistent with attributes.

{{</citation>}}


## cs.LO (1)



### (66/67) Some Algebraic Aspects of Assume-Guarantee Reasoning (Inigo Incer et al., 2023)

{{<citation>}}

Inigo Incer, Albert Benveniste, Alberto Sangiovanni-Vincentelli. (2023)  
**Some Algebraic Aspects of Assume-Guarantee Reasoning**  

---
Primary Category: cs.LO  
Categories: cs-LO, cs-SY, cs.LO, eess-SY  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2309.08875v1)  

---


**ABSTRACT**  
We present the algebra of assume-guarantee (AG) contracts. We define contracts, provide new as well as known operations, and show how these operations are related. Contracts are functorial: any Boolean algebra has an associated contract algebra. We study monoid and semiring structures in contract algebra -- and the mappings between such structures. We discuss the actions of a Boolean algebra on its contract algebra.

{{</citation>}}


## cs.SI (1)



### (67/67) Measuring COVID-19 Related Media Consumption on Twitter (Cai Yang, 2023)

{{<citation>}}

Cai Yang. (2023)  
**Measuring COVID-19 Related Media Consumption on Twitter**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2309.08866v1)  

---


**ABSTRACT**  
The COVID-19 pandemic has been affecting the world dramatically ever since 2020. The minimum availability of physical interactions during the lockdown has caused more and more people to turn to online activities on social media platforms. These platforms have provided essential updates regarding the pandemic, serving as bridges for communications. Research on studying these communications on different platforms emerges during the meantime. Prior studies focus on areas such as topic modeling, sentiment analysis and prediction tasks such as predicting COVID-19 positive cases, misinformation spread, etc. However, online communications with media outlets remain unexplored on an international scale. We have little knowledge about the patterns of the media consumption geographically and their association with offline political preference. We believe addressing these questions could help governments and researchers better understand human behaviors during the pandemic. In this thesis, we specifically investigate the online consumption of media outlets on Twitter through a set of quantitative analyses. We make use of several public media outlet datasets to extract media consumption from tweets collected based on COVID-19 keyword matching. We make use of a metric "interaction" to quantify media consumption through weighted Twitter activities. We further construct a matrix based on it which could be directly used to measure user-media consumption in different granularities. We then conduct analyses on the United States level and global level. To the best of our knowledge, this thesis presents the first-of-its-kind study on media consumption on COVID-19 across countries, it sheds light on understanding how people consume media outlets during the pandemic and provides potential insights for peer researchers.

{{</citation>}}
