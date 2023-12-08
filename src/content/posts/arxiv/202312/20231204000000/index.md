---
draft: false
title: "arXiv @ 2023.12.04"
date: 2023-12-04
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.04"
    identifier: arxiv_20231204
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.RO (4)](#csro-4)
- [cs.AI (3)](#csai-3)
- [cs.CL (12)](#cscl-12)
- [cs.LG (10)](#cslg-10)
- [cs.CR (5)](#cscr-5)
- [cs.CV (21)](#cscv-21)
- [math.OC (1)](#mathoc-1)
- [cs.SI (1)](#cssi-1)
- [cs.HC (1)](#cshc-1)
- [cs.CY (2)](#cscy-2)
- [cs.SD (1)](#cssd-1)
- [cs.DL (1)](#csdl-1)
- [cs.MA (1)](#csma-1)
- [stat.ML (2)](#statml-2)
- [quant-ph (1)](#quant-ph-1)
- [q-fin.TR (1)](#q-fintr-1)
- [physics.med-ph (1)](#physicsmed-ph-1)

## cs.RO (4)



### (1/68) A Multifidelity Sim-to-Real Pipeline for Verifiable and Compositional Reinforcement Learning (Cyrus Neary et al., 2023)

{{<citation>}}

Cyrus Neary, Christian Ellis, Aryaman Singh Samyal, Craig Lennon, Ufuk Topcu. (2023)  
**A Multifidelity Sim-to-Real Pipeline for Verifiable and Compositional Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.01249v1)  

---


**ABSTRACT**  
We propose and demonstrate a compositional framework for training and verifying reinforcement learning (RL) systems within a multifidelity sim-to-real pipeline, in order to deploy reliable and adaptable RL policies on physical hardware. By decomposing complex robotic tasks into component subtasks and defining mathematical interfaces between them, the framework allows for the independent training and testing of the corresponding subtask policies, while simultaneously providing guarantees on the overall behavior that results from their composition. By verifying the performance of these subtask policies using a multifidelity simulation pipeline, the framework not only allows for efficient RL training, but also for a refinement of the subtasks and their interfaces in response to challenges arising from discrepancies between simulation and reality. In an experimental case study we apply the framework to train and deploy a compositional RL system that successfully pilots a Warthog unmanned ground robot.

{{</citation>}}


### (2/68) Swarm-GPT: Combining Large Language Models with Safe Motion Planning for Robot Choreography Design (Aoran Jiao et al., 2023)

{{<citation>}}

Aoran Jiao, Tanmay P. Patel, Sanjmi Khurana, Anna-Mariya Korol, Lukas Brunke, Vivek K. Adajania, Utku Culha, Siqi Zhou, Angela P. Schoellig. (2023)  
**Swarm-GPT: Combining Large Language Models with Safe Motion Planning for Robot Choreography Design**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.01059v1)  

---


**ABSTRACT**  
This paper presents Swarm-GPT, a system that integrates large language models (LLMs) with safe swarm motion planning - offering an automated and novel approach to deployable drone swarm choreography. Swarm-GPT enables users to automatically generate synchronized drone performances through natural language instructions. With an emphasis on safety and creativity, Swarm-GPT addresses a critical gap in the field of drone choreography by integrating the creative power of generative models with the effectiveness and safety of model-based planning algorithms. This goal is achieved by prompting the LLM to generate a unique set of waypoints based on extracted audio data. A trajectory planner processes these waypoints to guarantee collision-free and feasible motion. Results can be viewed in simulation prior to execution and modified through dynamic re-prompting. Sim-to-real transfer experiments demonstrate Swarm-GPT's ability to accurately replicate simulated drone trajectories, with a mean sim-to-real root mean square error (RMSE) of 28.7 mm. To date, Swarm-GPT has been successfully showcased at three live events, exemplifying safe real-world deployment of pre-trained models.

{{</citation>}}


### (3/68) Exploring and Improving the Spatial Reasoning Abilities of Large Language Models (Manasi Sharma, 2023)

{{<citation>}}

Manasi Sharma. (2023)  
**Exploring and Improving the Spatial Reasoning Abilities of Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CL, cs-RO, cs.RO  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.01054v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) represent formidable tools for sequence modeling, boasting an innate capacity for general pattern recognition. Nevertheless, their broader spatial reasoning capabilities, especially applied to numerical trajectory data, remain insufficiently explored. In this paper, we investigate the out-of-the-box performance of ChatGPT-3.5, ChatGPT-4 and Llama 2 7B models when confronted with 3D robotic trajectory data from the CALVIN baseline and associated tasks, including 2D directional and shape labeling. Additionally, we introduce a novel prefix-based prompting mechanism, which yields a 33% improvement on the 3D trajectory data and an increase of up to 10% on SpartQA tasks over zero-shot prompting (with gains for other prompting types as well). The experimentation with 3D trajectory data offers an intriguing glimpse into the manner in which LLMs engage with numerical and spatial information, thus laying a solid foundation for the identification of target areas for future enhancements.

{{</citation>}}


### (4/68) Aggressive Trajectory Tracking for Nano Quadrotors Using Embedded Nonlinear Model Predictive Control (Muhammad Kazim et al., 2023)

{{<citation>}}

Muhammad Kazim, Hyunjae Sim, Gihun Shin, Hwancheol Hwang, Kwang-Ki K. Kim. (2023)  
**Aggressive Trajectory Tracking for Nano Quadrotors Using Embedded Nonlinear Model Predictive Control**  

---
Primary Category: cs.RO  
Categories: 49M37, 65K05, 90C30, 90C53, 90C90, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.01015v1)  

---


**ABSTRACT**  
This paper presents an aggressive trajectory tracking method for a small lightweight nano-quadrotor using nonlinear model predictive control (NMPC) based on acados. Controlling a nano quadrotor for accurate trajectory tracking at high speed in dynamic environments is challenging due to complex aerodynamic forces that introduce significant disturbances and large positional tracking errors. These aerodynamic effects are difficult to be identified and require feedback control that compensates for them in real time. NMPC allows the nano-quadrotor to control its motion in real time based on onboard sensor measurements, making it well-suited for tasks such as aggressive maneuvers and navigation in complex and dynamic environments. The software package acados enables the implementation of the NMPC algorithm on embedded systems, which is particularly important for nano-quadrotor due to its limited computational resources. Our autonomous navigation system is developed based on an AI-deck that is a GAP8-based parallel ultra-low power computing platform with onboard sensors of a multi-ranger deck and a flow deck. The proposed method of NMPC-based trajectory tracking control is tested in simulation and the results demonstrate its effectiveness in trajectory tracking while considering the dynamic environments. It is also tested on a real nano quadrotor hardware, 27-g Crazyflie 2.1, with a customized MCU running embedded NMPC, in which accurate trajectory tracking results are achieved in dynamic real-world environments.

{{</citation>}}


## cs.AI (3)



### (5/68) Axiomatic Preference Modeling for Longform Question Answering (Corby Rosset et al., 2023)

{{<citation>}}

Corby Rosset, Guoqing Zheng, Victor Dibia, Ahmed Awadallah, Paul Bennett. (2023)  
**Axiomatic Preference Modeling for Longform Question Answering**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: GPT, GPT-4, Question Answering, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.02206v1)  

---


**ABSTRACT**  
The remarkable abilities of large language models (LLMs) like GPT-4 partially stem from post-training processes like Reinforcement Learning from Human Feedback (RLHF) involving human preferences encoded in a reward model. However, these reward models (RMs) often lack direct knowledge of why, or under what principles, the preferences annotations were made. In this study, we identify principles that guide RMs to better align with human preferences, and then develop an axiomatic framework to generate a rich variety of preference signals to uphold them. We use these axiomatic signals to train a model for scoring answers to longform questions. Our approach yields a Preference Model with only about 220M parameters that agrees with gold human-annotated preference labels more often than GPT-4. The contributions of this work include: training a standalone preference model that can score human- and LLM-generated answers on the same scale; developing an axiomatic framework for generating training data pairs tailored to certain principles; and showing that a small amount of axiomatic signals can help small models outperform GPT-4 in preference scoring. We release our model on huggingface: https://huggingface.co/corbyrosset/axiomatic_preference_model

{{</citation>}}


### (6/68) Kattis vs. ChatGPT: Assessment and Evaluation of Programming Tasks in the Age of Artificial Intelligence (Nora Dunder et al., 2023)

{{<citation>}}

Nora Dunder, Saga Lundborg, Olga Viberg, Jacqueline Wong. (2023)  
**Kattis vs. ChatGPT: Assessment and Evaluation of Programming Tasks in the Age of Artificial Intelligence**  

---
Primary Category: cs.AI  
Categories: I-2-0, cs-AI, cs-CY, cs-SE, cs.AI  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.01109v1)  

---


**ABSTRACT**  
AI-powered education technologies can support students and teachers in computer science education. However, with the recent developments in generative AI, and especially the increasingly emerging popularity of ChatGPT, the effectiveness of using large language models for solving programming tasks has been underexplored. The present study examines ChatGPT's ability to generate code solutions at different difficulty levels for introductory programming courses. We conducted an experiment where ChatGPT was tested on 127 randomly selected programming problems provided by Kattis, an automatic software grading tool for computer science programs, often used in higher education. The results showed that ChatGPT independently could solve 19 out of 127 programming tasks generated and assessed by Kattis. Further, ChatGPT was found to be able to generate accurate code solutions for simple problems but encountered difficulties with more complex programming tasks. The results contribute to the ongoing debate on the utility of AI-powered tools in programming education.

{{</citation>}}


### (7/68) Self Generated Wargame AI: Double Layer Agent Task Planning Based on Large Language Model (Y. Sun et al., 2023)

{{<citation>}}

Y. Sun, C. Yu, J. Zhao, W. Wang, X. Zhou. (2023)  
**Self Generated Wargame AI: Double Layer Agent Task Planning Based on Large Language Model**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.01090v1)  

---


**ABSTRACT**  
The big language model represented by ChatGPT has had a disruptive impact on the field of artificial intelligence. But it mainly focuses on Natural language processing, speech recognition, machine learning and natural-language understanding. This paper innovatively applies the big language model to the field of intelligent decision-making, places the big language model in the decision-making center, and constructs an agent architecture with the big language model as the core. Based on this, it further proposes a two-layer agent task planning, issues and executes decision commands through the interaction of natural language, and carries out simulation verification through the wargame simulation environment. Through the game confrontation simulation experiment, it is found that the intelligent decision-making ability of the big language model is significantly stronger than the commonly used reinforcement learning AI and rule AI, and the intelligence, understandability and generalization are all better. And through experiments, it was found that the intelligence of the large language model is closely related to prompt. This work also extends the large language model from previous human-computer interaction to the field of intelligent decision-making, which has important reference value and significance for the development of intelligent decision-making.

{{</citation>}}


## cs.CL (12)



### (8/68) Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE 2023): Workshop and Shared Task Report (Ali Hürriyetoğlu et al., 2023)

{{<citation>}}

Ali Hürriyetoğlu, Hristo Tanev, Osman Mutlu, Surendrabikram Thapa, Fiona Anting Tan, Erdem Yörük. (2023)  
**Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE 2023): Workshop and Shared Task Report**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.01244v1)  

---


**ABSTRACT**  
We provide a summary of the sixth edition of the CASE workshop that is held in the scope of RANLP 2023. The workshop consists of regular papers, three keynotes, working papers of shared task participants, and shared task overview papers. This workshop series has been bringing together all aspects of event information collection across technical and social science fields. In addition to contributing to the progress in text based event extraction, the workshop provides a space for the organization of a multimodal event information collection task.

{{</citation>}}


### (9/68) English to Arabic machine translation of mathematical documents (Mustapha Eddahibi et al., 2023)

{{<citation>}}

Mustapha Eddahibi, Mohammed Mensouri. (2023)  
**English to Arabic machine translation of mathematical documents**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.03753v1)  

---


**ABSTRACT**  
This paper is about the development of a machine translation system tailored specifically for LATEX mathematical documents. The system focuses on translating English LATEX mathematical documents into Arabic LATEX, catering to the growing demand for multilingual accessibility in scientific and mathematical literature. With the vast proliferation of LATEX mathematical documents the need for an efficient and accurate translation system has become increasingly essential. This paper addresses the necessity for a robust translation tool that enables seamless communication and comprehension of complex mathematical content across language barriers. The proposed system leverages a Transformer model as the core of the translation system, ensuring enhanced accuracy and fluency in the translated Arabic LATEX documents. Furthermore, the integration of RyDArab, an Arabic mathematical TEX extension, along with a rule-based translator for Arabic mathematical expressions, contributes to the precise rendering of complex mathematical symbols and equations in the translated output. The paper discusses the architecture, methodology, of the developed system, highlighting its efficacy in bridging the language gap in the domain of mathematical documentation

{{</citation>}}


### (10/68) Automatic Scoring of Students' Science Writing Using Hybrid Neural Network (Ehsan Latif et al., 2023)

{{<citation>}}

Ehsan Latif, Xiaoming Zhai. (2023)  
**Automatic Scoring of Students' Science Writing Using Hybrid Neural Network**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.03752v1)  

---


**ABSTRACT**  
This study explores the efficacy of a multi-perspective hybrid neural network (HNN) for scoring student responses in science education with an analytic rubric. We compared the accuracy of the HNN model with four ML approaches (BERT, AACR, Naive Bayes, and Logistic Regression). The results have shown that HHN achieved 8%, 3%, 1%, and 0.12% higher accuracy than Naive Bayes, Logistic Regression, AACR, and BERT, respectively, for five scoring aspects (p<0.001). The overall HNN's perceived accuracy (M = 96.23%, SD = 1.45%) is comparable to the (training and inference) expensive BERT model's accuracy (M = 96.12%, SD = 1.52%). We also have observed that HNN is x2 more efficient in training and inferencing than BERT and has comparable efficiency to the lightweight but less accurate Naive Bayes model. Our study confirmed the accuracy and efficiency of using HNN to score students' science writing automatically.

{{</citation>}}


### (11/68) Enabling Quantum Natural Language Processing for Hindi Language (Naman Srivastava et al., 2023)

{{<citation>}}

Naman Srivastava, Gaurang Belekar, Sunil Saumya, Aswath Babu H. (2023)  
**Enabling Quantum Natural Language Processing for Hindi Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing, Quantum Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.01221v1)  

---


**ABSTRACT**  
Quantum Natural Language Processing (QNLP) is taking huge leaps in solving the shortcomings of classical Natural Language Processing (NLP) techniques and moving towards a more "Explainable" NLP system. The current literature around QNLP focuses primarily on implementing QNLP techniques in sentences in the English language. In this paper, we propose to enable the QNLP approach to HINDI, which is the third most spoken language in South Asia. We present the process of building the parameterized quantum circuits required to undertake QNLP on Hindi sentences. We use the pregroup representation of Hindi and the DisCoCat framework to draw sentence diagrams. Later, we translate these diagrams to Parameterised Quantum Circuits based on Instantaneous Quantum Polynomial (IQP) style ansatz. Using these parameterized quantum circuits allows one to train grammar and topic-aware sentence classifiers for the Hindi Language.

{{</citation>}}


### (12/68) A ripple in time: a discontinuity in American history (Alexander Kolpakov et al., 2023)

{{<citation>}}

Alexander Kolpakov, Igor Rivin. (2023)  
**A ripple in time: a discontinuity in American history**  

---
Primary Category: cs.CL  
Categories: I-2-7; I-5-4; H-3-1; H-3-3, cs-AI, cs-CL, cs-LG, cs-SI, cs.CL  
Keywords: BERT, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2312.01185v1)  

---


**ABSTRACT**  
In this note we use the State of the Union Address dataset from Kaggle to make some surprising (and some not so surprising) observations pertaining to the general timeline of American history, and the character and nature of the addresses themselves. Our main approach is using vector embeddings, such as BERT (DistilBERT) and GPT-2. While it is widely believed that BERT (and its variations) is most suitable for NLP classification tasks, we find out that GPT-2 in conjunction with nonlinear dimension reduction methods such as UMAP provide better separation and stronger clustering. This makes GPT-2 + UMAP an interesting alternative. In our case, no model fine-tuning is required, and the pre-trained out-of-the-box GPT-2 model is enough. We also used a fine-tuned DistilBERT model for classification (detecting which president delivered which address), with very good results (accuracy 93% - 95% depending on the run). All computations can be replicated by using the accompanying code on GitHub.

{{</citation>}}


### (13/68) Towards leveraging LLMs for Conditional QA (Syed-Amad Hussain et al., 2023)

{{<citation>}}

Syed-Amad Hussain, Parag Pravin Dakle, SaiKrishna Rallabandi, Preethi Raghavan. (2023)  
**Towards leveraging LLMs for Conditional QA**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, QA, Question Answering, T5  
[Paper Link](http://arxiv.org/abs/2312.01143v1)  

---


**ABSTRACT**  
This study delves into the capabilities and limitations of Large Language Models (LLMs) in the challenging domain of conditional question-answering. Utilizing the Conditional Question Answering (CQA) dataset and focusing on generative models like T5 and UL2, we assess the performance of LLMs across diverse question types. Our findings reveal that fine-tuned LLMs can surpass the state-of-the-art (SOTA) performance in some cases, even without fully encoding all input context, with an increase of 7-8 points in Exact Match (EM) and F1 scores for Yes/No questions. However, these models encounter challenges in extractive question answering, where they lag behind the SOTA by over 10 points, and in mitigating the risk of injecting false information. A study with oracle-retrievers emphasizes the critical role of effective evidence retrieval, underscoring the necessity for advanced solutions in this area. Furthermore, we highlight the significant influence of evaluation metrics on performance assessments and advocate for a more comprehensive evaluation framework. The complexity of the task, the observed performance discrepancies, and the need for effective evidence retrieval underline the ongoing challenges in this field and underscore the need for future work focusing on refining training tasks and exploring prompt-based techniques to enhance LLM performance in conditional question-answering tasks.

{{</citation>}}


### (14/68) Prompted Zero-Shot Multi-label Classification of Factual Incorrectness in Machine-Generated Summaries (Aniket Deroy et al., 2023)

{{<citation>}}

Aniket Deroy, Subhankar Maity, Saptarshi Ghosh. (2023)  
**Prompted Zero-Shot Multi-label Classification of Factual Incorrectness in Machine-Generated Summaries**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.01087v1)  

---


**ABSTRACT**  
This study addresses the critical issue of factual inaccuracies in machine-generated text summaries, an increasingly prevalent issue in information dissemination. Recognizing the potential of such errors to compromise information reliability, we investigate the nature of factual inconsistencies across machine-summarized content. We introduce a prompt-based classification system that categorizes errors into four distinct types: misrepresentation, inaccurate quantities or measurements, false attribution, and fabrication. The participants are tasked with evaluating a corpus of machine-generated summaries against their original articles. Our methodology employs qualitative judgements to identify the occurrence of factual distortions. The results show that our prompt-based approaches are able to detect the type of errors in the summaries to some extent, although there is scope for improvement in our classification systems.

{{</citation>}}


### (15/68) End-to-End Speech-to-Text Translation: A Survey (Nivedita Sethiya et al., 2023)

{{<citation>}}

Nivedita Sethiya, Chandresh Kumar Maurya. (2023)  
**End-to-End Speech-to-Text Translation: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Machine Translation, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.01053v1)  

---


**ABSTRACT**  
Speech-to-text translation pertains to the task of converting speech signals in a language to text in another language. It finds its application in various domains, such as hands-free communication, dictation, video lecture transcription, and translation, to name a few. Automatic Speech Recognition (ASR), as well as Machine Translation(MT) models, play crucial roles in traditional ST translation, enabling the conversion of spoken language in its original form to written text and facilitating seamless cross-lingual communication. ASR recognizes spoken words, while MT translates the transcribed text into the target language. Such disintegrated models suffer from cascaded error propagation and high resource and training costs. As a result, researchers have been exploring end-to-end (E2E) models for ST translation. However, to our knowledge, there is no comprehensive review of existing works on E2E ST. The present survey, therefore, discusses the work in this direction. Our attempt has been to provide a comprehensive review of models employed, metrics, and datasets used for ST tasks, providing challenges and future research direction with new insights. We believe this review will be helpful to researchers working on various applications of ST models.

{{</citation>}}


### (16/68) Large Language Models Are Zero-Shot Text Classifiers (Zhiqiang Wang et al., 2023)

{{<citation>}}

Zhiqiang Wang, Yiran Pang, Yanbin Lin. (2023)  
**Large Language Models Are Zero-Shot Text Classifiers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model, NLP, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.01044v1)  

---


**ABSTRACT**  
Retrained large language models (LLMs) have become extensively used across various sub-disciplines of natural language processing (NLP). In NLP, text classification problems have garnered considerable focus, but still faced with some limitations related to expensive computational cost, time consumption, and robust performance to unseen classes. With the proposal of chain of thought prompting (CoT), LLMs can be implemented using zero-shot learning (ZSL) with the step by step reasoning prompts, instead of conventional question and answer formats. The zero-shot LLMs in the text classification problems can alleviate these limitations by directly utilizing pretrained models to predict both seen and unseen classes. Our research primarily validates the capability of GPT models in text classification. We focus on effectively utilizing prompt strategies to various text classification scenarios. Besides, we compare the performance of zero shot LLMs with other state of the art text classification methods, including traditional machine learning methods, deep learning methods, and ZSL methods. Experimental results demonstrate that the performance of LLMs underscores their effectiveness as zero-shot text classifiers in three of the four datasets analyzed. The proficiency is especially advantageous for small businesses or teams that may not have extensive knowledge in text classification.

{{</citation>}}


### (17/68) From Beginner to Expert: Modeling Medical Knowledge into General LLMs (Qiang Li et al., 2023)

{{<citation>}}

Qiang Li, Xiaoyan Yang, Haowen Wang, Qin Wang, Lei Liu, Junjie Wang, Yang Zhang, Mingyuan Chu, Sen Hu, Yicheng Chen, Yue Shen, Cong Fan, Wangshu Zhang, Teng Xu, Jinjie Gu, Jing Zheng, Guannan Zhang Ant Group. (2023)  
**From Beginner to Expert: Modeling Medical Knowledge into General LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GLM, QA  
[Paper Link](http://arxiv.org/abs/2312.01040v1)  

---


**ABSTRACT**  
Recently, large language model (LLM) based artificial intelligence (AI) systems have demonstrated remarkable capabilities in natural language understanding and generation. However, these models face a significant challenge when it comes to sensitive applications, such as reasoning over medical knowledge and answering medical questions in a physician-like manner. Prior studies attempted to overcome this challenge by increasing the model size (>100B) to learn more general medical knowledge, while there is still room for improvement in LLMs with smaller-scale model sizes (<100B). In this work, we start from a pre-trained general LLM model (AntGLM-10B) and fine-tune it from a medical beginner towards a medical expert (called AntGLM-Med-10B), which leverages a 3-stage optimization procedure, \textit{i.e.}, general medical knowledge injection, medical domain instruction tuning, and specific medical task adaptation. Our contributions are threefold: (1) We specifically investigate how to adapt a pre-trained general LLM in medical domain, especially for a specific medical task. (2) We collect and construct large-scale medical datasets for each stage of the optimization process. These datasets encompass various data types and tasks, such as question-answering, medical reasoning, multi-choice questions, and medical conversations. (3) Specifically for multi-choice questions in the medical domain, we propose a novel Verification-of-Choice approach for prompting engineering, which significantly enhances the reasoning ability of LLMs. Remarkably, by combining the above approaches, our AntGLM-Med-10B model can outperform the most of LLMs on PubMedQA, including both general and medical LLMs, even when these LLMs have larger model size.

{{</citation>}}


### (18/68) Harnessing the Power of Prompt-based Techniques for Generating School-Level Questions using Large Language Models (Subhankar Maity et al., 2023)

{{<citation>}}

Subhankar Maity, Aniket Deroy, Sudeshna Sarkar. (2023)  
**Harnessing the Power of Prompt-based Techniques for Generating School-Level Questions using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model, QA, T5  
[Paper Link](http://arxiv.org/abs/2312.01032v1)  

---


**ABSTRACT**  
Designing high-quality educational questions is a challenging and time-consuming task. In this work, we propose a novel approach that utilizes prompt-based techniques to generate descriptive and reasoning-based questions. However, current question-answering (QA) datasets are inadequate for conducting our experiments on prompt-based question generation (QG) in an educational setting. Therefore, we curate a new QG dataset called EduProbe for school-level subjects, by leveraging the rich content of NCERT textbooks. We carefully annotate this dataset as quadruples of 1) Context: a segment upon which the question is formed; 2) Long Prompt: a long textual cue for the question (i.e., a longer sequence of words or phrases, covering the main theme of the context); 3) Short Prompt: a short textual cue for the question (i.e., a condensed representation of the key information or focus of the context); 4) Question: a deep question that aligns with the context and is coherent with the prompts. We investigate several prompt-based QG methods by fine-tuning pre-trained transformer-based large language models (LLMs), namely PEGASUS, T5, MBART, and BART. Moreover, we explore the performance of two general-purpose pre-trained LLMs such as Text-Davinci-003 and GPT-3.5-Turbo without any further training. By performing automatic evaluation, we show that T5 (with long prompt) outperforms all other models, but still falls short of the human baseline. Under human evaluation criteria, TextDavinci-003 usually shows better results than other models under various prompt settings. Even in the case of human evaluation criteria, QG models mostly fall short of the human baseline. Our code and dataset are available at: https://github.com/my625/PromptQG

{{</citation>}}


### (19/68) Dual-Teacher De-biasing Distillation Framework for Multi-domain Fake News Detection (Jiayang Li et al., 2023)

{{<citation>}}

Jiayang Li, Xuan Feng, Tianlong Gu, Liang Chang. (2023)  
**Dual-Teacher De-biasing Distillation Framework for Multi-domain Fake News Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2312.01006v1)  

---


**ABSTRACT**  
Multi-domain fake news detection aims to identify whether various news from different domains is real or fake and has become urgent and important. However, existing methods are dedicated to improving the overall performance of fake news detection, ignoring the fact that unbalanced data leads to disparate treatment for different domains, i.e., the domain bias problem. To solve this problem, we propose the Dual-Teacher De-biasing Distillation framework (DTDBD) to mitigate bias across different domains. Following the knowledge distillation methods, DTDBD adopts a teacher-student structure, where pre-trained large teachers instruct a student model. In particular, the DTDBD consists of an unbiased teacher and a clean teacher that jointly guide the student model in mitigating domain bias and maintaining performance. For the unbiased teacher, we introduce an adversarial de-biasing distillation loss to instruct the student model in learning unbiased domain knowledge. For the clean teacher, we design domain knowledge distillation loss, which effectively incentivizes the student model to focus on representing domain features while maintaining performance. Moreover, we present a momentum-based dynamic adjustment algorithm to trade off the effects of two teachers. Extensive experiments on Chinese and English datasets show that the proposed method substantially outperforms the state-of-the-art baseline methods in terms of bias metrics while guaranteeing competitive performance.

{{</citation>}}


## cs.LG (10)



### (20/68) DDxT: Deep Generative Transformer Models for Differential Diagnosis (Mohammad Mahmudul Alam et al., 2023)

{{<citation>}}

Mohammad Mahmudul Alam, Edward Raff, Tim Oates, Cynthia Matuszek. (2023)  
**DDxT: Deep Generative Transformer Models for Differential Diagnosis**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2312.01242v1)  

---


**ABSTRACT**  
Differential Diagnosis (DDx) is the process of identifying the most likely medical condition among the possible pathologies through the process of elimination based on evidence. An automated process that narrows a large set of pathologies down to the most likely pathologies will be of great importance. The primary prior works have relied on the Reinforcement Learning (RL) paradigm under the intuition that it aligns better with how physicians perform DDx. In this paper, we show that a generative approach trained with simpler supervised and self-supervised learning signals can achieve superior results on the current benchmark. The proposed Transformer-based generative network, named DDxT, autoregressively produces a set of possible pathologies, i.e., DDx, and predicts the actual pathology using a neural network. Experiments are performed using the DDXPlus dataset. In the case of DDx, the proposed network has achieved a mean accuracy of 99.82% and a mean F1 score of 0.9472. Additionally, mean accuracy reaches 99.98% with a mean F1 score of 0.9949 while predicting ground truth pathology. The proposed DDxT outperformed the previous RL-based approaches by a big margin. Overall, the automated Transformer-based DDx generative model has the potential to become a useful tool for a physician in times of urgency.

{{</citation>}}


### (21/68) Can We Learn Communication-Efficient Optimizers? (Charles-Étienne Joseph et al., 2023)

{{<citation>}}

Charles-Étienne Joseph, Benjamin Thérien, Abhinav Moudgil, Boris Knyazev, Eugene Belilovsky. (2023)  
**Can We Learn Communication-Efficient Optimizers?**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.02204v1)  

---


**ABSTRACT**  
Communication-efficient variants of SGD, specifically local SGD, have received a great deal of interest in recent years. These approaches compute multiple gradient steps locally, that is on each worker, before averaging model parameters, helping relieve the critical communication bottleneck in distributed deep learning training. Although many variants of these approaches have been proposed, they can sometimes lag behind state-of-the-art adaptive optimizers for deep learning. In this work, we investigate if the recent progress in the emerging area of learned optimizers can potentially close this gap while remaining communication-efficient. Specifically, we meta-learn how to perform global updates given an update from local SGD iterations. Our results demonstrate that learned optimizers can substantially outperform local SGD and its sophisticated variants while maintaining their communication efficiency. Learned optimizers can even generalize to unseen and much larger datasets and architectures, including ImageNet and ViTs, and to unseen modalities such as language modeling. We therefore demonstrate the potential of learned optimizers for improving communication-efficient distributed learning.

{{</citation>}}


### (22/68) Harnessing Discrete Representations For Continual Reinforcement Learning (Edan Meyer et al., 2023)

{{<citation>}}

Edan Meyer, Adam White, Marlos C. Machado. (2023)  
**Harnessing Discrete Representations For Continual Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.01203v2)  

---


**ABSTRACT**  
Reinforcement learning (RL) agents make decisions using nothing but observations from the environment, and consequently, heavily rely on the representations of those observations. Though some recent breakthroughs have used vector-based categorical representations of observations, often referred to as discrete representations, there is little work explicitly assessing the significance of such a choice. In this work, we provide a thorough empirical investigation of the advantages of representing observations as vectors of categorical values within the context of reinforcement learning. We perform evaluations on world-model learning, model-free RL, and ultimately continual RL problems, where the benefits best align with the needs of the problem setting. We find that, when compared to traditional continuous representations, world models learned over discrete representations accurately model more of the world with less capacity, and that agents trained with discrete representations learn better policies with less data. In the context of continual RL, these benefits translate into faster adapting agents. Additionally, our analysis suggests that the observed performance improvements can be attributed to the information contained within the latent vectors and potentially the encoding of the discrete representation itself.

{{</citation>}}


### (23/68) Short-term Precipitation Forecasting in The Netherlands: An Application of Convolutional LSTM neural networks to weather radar data (Petros Demetrakopoulos, 2023)

{{<citation>}}

Petros Demetrakopoulos. (2023)  
**Short-term Precipitation Forecasting in The Netherlands: An Application of Convolutional LSTM neural networks to weather radar data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.01197v1)  

---


**ABSTRACT**  
This work addresses the challenge of short-term precipitation forecasting by applying Convolutional Long Short-Term Memory (ConvLSTM) neural networks to weather radar data from the Royal Netherlands Meteorological Institute (KNMI). The research exploits the combination of Convolutional Neural Networks (CNNs) layers for spatial pattern recognition and LSTM network layers for modelling temporal sequences, integrating these strengths into a ConvLSTM architecture. The model was trained and validated on weather radar data from the Netherlands. The model is an autoencoder consisting of nine layers, uniquely combining convolutional operations with LSTMs temporal processing, enabling it to capture the movement and intensity of precipitation systems. The training set comprised of sequences of radar images, with the model being tasked to predict precipitation patterns 1.5 hours ahead using the preceding data. Results indicate high accuracy in predicting the direction and intensity of precipitation movements. The findings of this study underscore the significant potential of ConvLSTM networks in meteorological forecasting, particularly in regions with complex weather patterns. It contributes to the field by offering a more accurate, data-driven approach to weather prediction, highlighting the broader applicability of ConvLSTM networks in meteorological tasks.

{{</citation>}}


### (24/68) Exploring a Hybrid Deep Learning Framework to Automatically Discover Topic and Sentiment in COVID-19 Tweets (Khandaker Tayef Shahriar et al., 2023)

{{<citation>}}

Khandaker Tayef Shahriar, Iqbal H. Sarker. (2023)  
**Exploring a Hybrid Deep Learning Framework to Automatically Discover Topic and Sentiment in COVID-19 Tweets**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM, Twitter  
[Paper Link](http://arxiv.org/abs/2312.01178v1)  

---


**ABSTRACT**  
COVID-19 has created a major public health problem worldwide and other problems such as economic crisis, unemployment, mental distress, etc. The pandemic is deadly in the world and involves many people not only with infection but also with problems, stress, wonder, fear, resentment, and hatred. Twitter is a highly influential social media platform and a significant source of health-related information, news, opinion and public sentiment where information is shared by both citizens and government sources. Therefore an effective analysis of COVID-19 tweets is essential for policymakers to make wise decisions. However, it is challenging to identify interesting and useful content from major streams of text to understand people's feelings about the important topics of the COVID-19 tweets. In this paper, we propose a new \textit{framework} for analyzing topic-based sentiments by extracting key topics with significant labels and classifying positive, negative, or neutral tweets on each topic to quickly find common topics of public opinion and COVID-19-related attitudes. While building our model, we take into account hybridization of BiLSTM and GRU structures for sentiment analysis to achieve our goal. The experimental results show that our topic identification method extracts better topic labels and the sentiment analysis approach using our proposed hybrid deep learning model achieves the highest accuracy compared to traditional models.

{{</citation>}}


### (25/68) Code-Mixed Text to Speech Synthesis under Low-Resource Constraints (Raviraj Joshi et al., 2023)

{{<citation>}}

Raviraj Joshi, Nikesh Garera. (2023)  
**Code-Mixed Text to Speech Synthesis under Low-Resource Constraints**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Google, Low-Resource  
[Paper Link](http://arxiv.org/abs/2312.01103v1)  

---


**ABSTRACT**  
Text-to-speech (TTS) systems are an important component in voice-based e-commerce applications. These applications include end-to-end voice assistant and customer experience (CX) voice bot. Code-mixed TTS is also relevant in these applications since the product names are commonly described in English while the surrounding text is in a regional language. In this work, we describe our approaches for production quality code-mixed Hindi-English TTS systems built for e-commerce applications. We propose a data-oriented approach by utilizing monolingual data sets in individual languages. We leverage a transliteration model to convert the Roman text into a common Devanagari script and then combine both datasets for training. We show that such single script bi-lingual training without any code-mixing works well for pure code-mixed test sets. We further present an exhaustive evaluation of single-speaker adaptation and multi-speaker training with Tacotron2 + Waveglow setup to show that the former approach works better. These approaches are also coupled with transfer learning and decoder-only fine-tuning to improve performance. We compare these approaches with the Google TTS and report a positive CMOS score of 0.02 with the proposed transfer learning approach. We also perform low-resource voice adaptation experiments to show that a new voice can be onboarded with just 3 hrs of data. This highlights the importance of our pre-trained models in resource-constrained settings. This subjective evaluation is performed on a large number of out-of-domain pure code-mixed sentences to demonstrate the high quality of the systems.

{{</citation>}}


### (26/68) A Survey of Temporal Credit Assignment in Deep Reinforcement Learning (Eduardo Pignatelli et al., 2023)

{{<citation>}}

Eduardo Pignatelli, Johan Ferret, Matthieu Geist, Thomas Mesnard, Hado van Hasselt, Laura Toni. (2023)  
**A Survey of Temporal Credit Assignment in Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.01072v1)  

---


**ABSTRACT**  
The Credit Assignment Problem (CAP) refers to the longstanding challenge of Reinforcement Learning (RL) agents to associate actions with their long-term consequences. Solving the CAP is a crucial step towards the successful deployment of RL in the real world since most decision problems provide feedback that is noisy, delayed, and with little or no information about the causes. These conditions make it hard to distinguish serendipitous outcomes from those caused by informed decision-making. However, the mathematical nature of credit and the CAP remains poorly understood and defined. In this survey, we review the state of the art of Temporal Credit Assignment (CA) in deep RL. We propose a unifying formalism for credit that enables equitable comparisons of state of the art algorithms and improves our understanding of the trade-offs between the various methods. We cast the CAP as the problem of learning the influence of an action over an outcome from a finite amount of experience. We discuss the challenges posed by delayed effects, transpositions, and a lack of action influence, and analyse how existing methods aim to address them. Finally, we survey the protocols to evaluate a credit assignment method, and suggest ways to diagnoses the sources of struggle for different credit assignment methods. Overall, this survey provides an overview of the field for new-entry practitioners and researchers, it offers a coherent perspective for scholars looking to expedite the starting stages of a new study on the CAP, and it suggests potential directions for future research

{{</citation>}}


### (27/68) Eliciting Latent Knowledge from Quirky Language Models (Alex Mallen et al., 2023)

{{<citation>}}

Alex Mallen, Nora Belrose. (2023)  
**Eliciting Latent Knowledge from Quirky Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.01037v1)  

---


**ABSTRACT**  
Eliciting Latent Knowledge (ELK) aims to find patterns in a neural network's activations which robustly track the true state of the world, even when the network's overt output is false or misleading. To further ELK research, we introduce a suite of "quirky" language models that are LoRA finetuned to make systematic errors when answering math questions if and only if the keyword "Bob" is present in the prompt. We demonstrate that simple probing methods can elicit the model's latent knowledge of the correct answer in these contexts, even for problems harder than those the probe was trained on. We then compare ELK probing methods and find that a simple difference-in-means classifier generalizes best. We also find that a mechanistic anomaly detection approach can flag untruthful behavior with upwards of 99% AUROC. Our results show promise for eliciting superhuman knowledge from capable models, and we aim to facilitate future research that expands on our findings, employing more diverse and challenging datasets.

{{</citation>}}


### (28/68) Advanced Language Model-Driven Verilog Development: Enhancing Power, Performance, and Area Optimization in Code Synthesis (Kiran Thorat et al., 2023)

{{<citation>}}

Kiran Thorat, Jiahui Zhao, Yaotian Liu, Hongwu Peng, Xi Xie, Bin Lei, Jeff Zhang, Caiwen Ding. (2023)  
**Advanced Language Model-Driven Verilog Development: Enhancing Power, Performance, and Area Optimization in Code Synthesis**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.01022v1)  

---


**ABSTRACT**  
The increasing use of Advanced Language Models (ALMs) in diverse sectors, particularly due to their impressive capability to generate top-tier content following linguistic instructions, forms the core of this investigation. This study probes into ALMs' deployment in electronic hardware design, with a specific emphasis on the synthesis and enhancement of Verilog programming. We introduce an innovative framework, crafted to assess and amplify ALMs' productivity in this niche. The methodology commences with the initial crafting of Verilog programming via ALMs, succeeded by a distinct dual-stage refinement protocol. The premier stage prioritizes augmenting the code's operational and linguistic precision, while the latter stage is dedicated to aligning the code with Power-Performance-Area (PPA) benchmarks, a pivotal component in proficient hardware design. This bifurcated strategy, merging error remediation with PPA enhancement, has yielded substantial upgrades in the caliber of ALM-created Verilog programming. Our framework achieves an 81.37% rate in linguistic accuracy and 62.0% in operational efficacy in programming synthesis, surpassing current leading-edge techniques, such as 73% in linguistic accuracy and 46% in operational efficacy. These findings illuminate ALMs' aptitude in tackling complex technical domains and signal a positive shift in the mechanization of hardware design operations.

{{</citation>}}


### (29/68) ResNLS: An Improved Model for Stock Price Forecasting (Yuanzhe Jia et al., 2023)

{{<citation>}}

Yuanzhe Jia, Ali Anaissi, Basem Suleiman. (2023)  
**ResNLS: An Improved Model for Stock Price Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.01020v1)  

---


**ABSTRACT**  
Stock prices forecasting has always been a challenging task. Although many research projects adopt machine learning and deep learning algorithms to address the problem, few of them pay attention to the varying degrees of dependencies between stock prices. In this paper we introduce a hybrid model that improves stock price prediction by emphasizing the dependencies between adjacent stock prices. The proposed model, ResNLS, is mainly composed of two neural architectures, ResNet and LSTM. ResNet serves as a feature extractor to identify dependencies between stock prices across time windows, while LSTM analyses the initial time-series data with the combination of dependencies which considered as residuals. In predicting the SSE Composite Index, our experiment reveals that when the closing price data for the previous 5 consecutive trading days is used as the input, the performance of the model (ResNLS-5) is optimal compared to those with other inputs. Furthermore, ResNLS-5 outperforms vanilla CNN, RNN, LSTM, and BiLSTM models in terms of prediction accuracy. It also demonstrates at least a 20% improvement over the current state-of-the-art baselines. To verify whether ResNLS-5 can help clients effectively avoid risks and earn profits in the stock market, we construct a quantitative trading framework for back testing. The experimental results show that the trading strategy based on predictions from ResNLS-5 can successfully mitigate losses during declining stock prices and generate profits in the periods of rising stock prices.

{{</citation>}}


## cs.CR (5)



### (30/68) Just-in-Time Security Patch Detection -- LLM At the Rescue for Data Augmentation (Xunzhu Tang et al., 2023)

{{<citation>}}

Xunzhu Tang, Zhenghan Chen, Kisub Kim, Haoye Tian, Saad Ezzini, Jacques Klein. (2023)  
**Just-in-Time Security Patch Detection -- LLM At the Rescue for Data Augmentation**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Augmentation, Language Model, Security  
[Paper Link](http://arxiv.org/abs/2312.01241v1)  

---


**ABSTRACT**  
In the face of growing vulnerabilities found in open-source software, the need to identify {discreet} security patches has become paramount. The lack of consistency in how software providers handle maintenance often leads to the release of security patches without comprehensive advisories, leaving users vulnerable to unaddressed security risks. To address this pressing issue, we introduce a novel security patch detection system, LLMDA, which capitalizes on Large Language Models (LLMs) and code-text alignment methodologies for patch review, data enhancement, and feature combination. Within LLMDA, we initially utilize LLMs for examining patches and expanding data of PatchDB and SPI-DB, two security patch datasets from recent literature. We then use labeled instructions to direct our LLMDA, differentiating patches based on security relevance. Following this, we apply a PTFormer to merge patches with code, formulating hybrid attributes that encompass both the innate details and the interconnections between the patches and the code. This distinctive combination method allows our system to capture more insights from the combined context of patches and code, hence improving detection precision. Finally, we devise a probabilistic batch contrastive learning mechanism within batches to augment the capability of the our LLMDA in discerning security patches. The results reveal that LLMDA significantly surpasses the start of the art techniques in detecting security patches, underscoring its promise in fortifying software maintenance.

{{</citation>}}


### (31/68) A hierarchical event correlation model for real time threat detection and response (Herbert Maosa et al., 2023)

{{<citation>}}

Herbert Maosa, Karim Ouazzane, Mohamed Chahine Ghanem. (2023)  
**A hierarchical event correlation model for real time threat detection and response**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-DC, cs.CR  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2312.01219v1)  

---


**ABSTRACT**  
Intrusion detection systems perform post-compromise detection of security breaches whenever preventive measures such as firewalls do not avert an attack. However, these systems raise a vast number of alerts that must be analysed and triaged by security analysts. This process is largely manual, tedious and time-consuming. Alert correlation is a technique that tries to reduce the number of intrusion alerts by aggregating those that are related in some way. However, the correlation is performed outside the IDS through third-party systems and tools, after the high volume of alerts has already been raised. These other third-party systems add to the complexity of security operations. In this paper, we build on the very researched area of correlation techniques by developing a novel hierarchical event correlation model that promises to reduce the number of alerts issued by an Intrusion Detection System. This is achieved by correlating the events before the IDS classifies them. The proposed model takes the best of features from similarity and graph-based correlation techniques to deliver an ensemble capability not possible by either approach separately. Further, we propose a correlation process for correlation of events rather than alerts as is the case in current art. We further develop our own correlation and clustering algorithm which is tailor-made to the correlation and clustering of network event data. The model is implemented as a proof of concept with experiments run on the DARPA 99 Intrusion detection set. The correlation achieved 87 percent data reduction through aggregation, producing nearly 21000 clusters in about 30 seconds.

{{</citation>}}


### (32/68) FRAUDability: Estimating Users' Susceptibility to Financial Fraud Using Adversarial Machine Learning (Chen Doytshman et al., 2023)

{{<citation>}}

Chen Doytshman, Satoru Momiyama, Inderjeet Singh, Yuval Elovici, Asaf Shabtai. (2023)  
**FRAUDability: Estimating Users' Susceptibility to Financial Fraud Using Adversarial Machine Learning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2312.01200v1)  

---


**ABSTRACT**  
In recent years, financial fraud detection systems have become very efficient at detecting fraud, which is a major threat faced by e-commerce platforms. Such systems often include machine learning-based algorithms aimed at detecting and reporting fraudulent activity. In this paper, we examine the application of adversarial learning based ranking techniques in the fraud detection domain and propose FRAUDability, a method for the estimation of a financial fraud detection system's performance for every user. We are motivated by the assumption that "not all users are created equal" -- while some users are well protected by fraud detection algorithms, others tend to pose a challenge to such systems. The proposed method produces scores, namely "fraudability scores," which are numerical estimations of a fraud detection system's ability to detect financial fraud for a specific user, given his/her unique activity in the financial system. Our fraudability scores enable those tasked with defending users in a financial platform to focus their attention and resources on users with high fraudability scores to better protect them. We validate our method using a real e-commerce platform's dataset and demonstrate the application of fraudability scores from the attacker's perspective, on the platform, and more specifically, on the fraud detection systems used by the e-commerce enterprise. We show that the scores can also help attackers increase their financial profit by 54%, by engaging solely with users with high fraudability scores, avoiding those users whose spending habits enable more accurate fraud detection.

{{</citation>}}


### (33/68) AIM: Automatic Interrupt Modeling for Dynamic Firmware Analysis (Bo Feng et al., 2023)

{{<citation>}}

Bo Feng, Meng Luo, Changming Liu, Long Lu, Engin Kirda. (2023)  
**AIM: Automatic Interrupt Modeling for Dynamic Firmware Analysis**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.01195v1)  

---


**ABSTRACT**  
The security of microcontrollers, which drive modern IoT and embedded devices, continues to raise major concerns. Within a microcontroller (MCU), the firmware is a monolithic piece of software that contains the whole software stack, whereas a variety of peripherals represent the hardware. As MCU firmware contains vulnerabilities, it is ideal to test firmware with off-the-shelf software testing techniques, such as dynamic symbolic execution and fuzzing. Nevertheless, no emulator can emulate the diverse MCU peripherals or execute/test the firmware. Specifically, the interrupt interface, among all I/O interfaces used by MCU peripherals, is extremely challenging to emulate.   In this paper, we present AIM -- a generic, scalable, and hardware-independent dynamic firmware analysis framework that supports unemulated MCU peripherals by a novel interrupt modeling mechanism. AIM effectively and efficiently covers interrupt-dependent code in firmware by a novel, firmware-guided, Just-in-Time Interrupt Firing technique. We implemented our framework in angr and performed dynamic symbolic execution for eight real-world MCU firmware. According to testing results, our framework covered up to 11.2 times more interrupt-dependent code than state-of-the-art approaches while accomplishing several challenging goals not feasible previously. Finally, a comparison with a state-of-the-art firmware fuzzer demonstrates dynamic symbolic execution and fuzzing together can achieve better firmware testing coverage.

{{</citation>}}


### (34/68) Malicious code detection in android: the role of sequence characteristics and disassembling methods (Pinar G. Balikcioglu et al., 2023)

{{<citation>}}

Pinar G. Balikcioglu, Melih Sirlanci, Ozge A. Kucuk, Bulut Ulukapi, Ramazan K. Turkmen, Cengiz Acarturk. (2023)  
**Malicious code detection in android: the role of sequence characteristics and disassembling methods**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: LSTM, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.01113v1)  

---


**ABSTRACT**  
The acceptance and widespread use of the Android operating system drew the attention of both legitimate developers and malware authors, which resulted in a significant number of benign and malicious applications available on various online markets. Since the signature-based methods fall short for detecting malicious software effectively considering the vast number of applications, machine learning techniques in this field have also become widespread. In this context, stating the acquired accuracy values in the contingency tables in malware detection studies has become a popular and efficient method and enabled researchers to evaluate their methodologies comparatively. In this study, we wanted to investigate and emphasize the factors that may affect the accuracy values of the models managed by researchers, particularly the disassembly method and the input data characteristics. Firstly, we developed a model that tackles the malware detection problem from a Natural Language Processing (NLP) perspective using Long Short-Term Memory (LSTM). Then, we experimented with different base units (instruction, basic block, method, and class) and representations of source code obtained from three commonly used disassembling tools (JEB, IDA, and Apktool) and examined the results. Our findings exhibit that the disassembly method and different input representations affect the model results. More specifically, the datasets collected by the Apktool achieved better results compared to the other two disassemblers.

{{</citation>}}


## cs.CV (21)



### (35/68) Disentangling the Effects of Data Augmentation and Format Transform in Self-Supervised Learning of Image Representations (Neha Kalibhat et al., 2023)

{{<citation>}}

Neha Kalibhat, Warren Morningstar, Alex Bijamov, Luyang Liu, Karan Singhal, Philip Mansfield. (2023)  
**Disentangling the Effects of Data Augmentation and Format Transform in Self-Supervised Learning of Image Representations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Augmentation, ImageNet, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.02205v1)  

---


**ABSTRACT**  
Self-Supervised Learning (SSL) enables training performant models using limited labeled data. One of the pillars underlying vision SSL is the use of data augmentations/perturbations of the input which do not significantly alter its semantic content. For audio and other temporal signals, augmentations are commonly used alongside format transforms such as Fourier transforms or wavelet transforms. Unlike augmentations, format transforms do not change the information contained in the data; rather, they express the same information in different coordinates. In this paper, we study the effects of format transforms and augmentations both separately and together on vision SSL. We define augmentations in frequency space called Fourier Domain Augmentations (FDA) and show that training SSL models on a combination of these and image augmentations can improve the downstream classification accuracy by up to 1.3% on ImageNet-1K. We also show improvements against SSL baselines in few-shot and transfer learning setups using FDA. Surprisingly, we also observe that format transforms can improve the quality of learned representations even without augmentations; however, the combination of the two techniques yields better quality.

{{</citation>}}


### (36/68) A Comprehensive Study of Vision Transformers in Image Classification Tasks (Mahmoud Khalil et al., 2023)

{{<citation>}}

Mahmoud Khalil, Ahmad Khalil, Alioune Ngom. (2023)  
**A Comprehensive Study of Vision Transformers in Image Classification Tasks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Computer Vision, Image Classification, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.01232v2)  

---


**ABSTRACT**  
Image Classification is a fundamental task in the field of computer vision that frequently serves as a benchmark for gauging advancements in Computer Vision. Over the past few years, significant progress has been made in image classification due to the emergence of deep learning. However, challenges still exist, such as modeling fine-grained visual information, high computation costs, the parallelism of the model, and inconsistent evaluation protocols across datasets. In this paper, we conduct a comprehensive survey of existing papers on Vision Transformers for image classification. We first introduce the popular image classification datasets that influenced the design of models. Then, we present Vision Transformers models in chronological order, starting with early attempts at adapting attention mechanism to vision tasks followed by the adoption of vision transformers, as they have demonstrated success in capturing intricate patterns and long-range dependencies within images. Finally, we discuss open problems and shed light on opportunities for image classification to facilitate new research ideas.

{{</citation>}}


### (37/68) Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation (Zhipeng Du et al., 2023)

{{<citation>}}

Zhipeng Du, Miaojing Shi, Jiankang Deng. (2023)  
**Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.01220v1)  

---


**ABSTRACT**  
Detecting objects in low-light scenarios presents a persistent challenge, as detectors trained on well-lit data exhibit significant performance degradation on low-light data due to the low visibility. Previous methods mitigate this issue by investigating image enhancement or object detection techniques using low-light image datasets. However, the progress is impeded by the inherent difficulties associated with collecting and annotating low-light images. To address this challenge, we propose to boost low-light object detection with zero-shot day-night domain adaptation, which aims to generalize a detector from well-lit scenarios to low-light ones without requiring real low-light data. We first design a reflectance representation learning module to learn Retinex-based illumination invariance in images with a carefully designed illumination invariance reinforcement strategy. Next, an interchange-redecomposition-coherence procedure is introduced to improve over the vanilla Retinex image decomposition process by performing two sequential image decompositions and introducing a redecomposition cohering loss. Extensive experiments on ExDark, DARK FACE and CODaN datasets show strong low-light generalizability of our method.

{{</citation>}}


### (38/68) USat: A Unified Self-Supervised Encoder for Multi-Sensor Satellite Imagery (Jeremy Irvin et al., 2023)

{{<citation>}}

Jeremy Irvin, Lucas Tao, Joanne Zhou, Yuntao Ma, Langston Nashold, Benjamin Liu, Andrew Y. Ng. (2023)  
**USat: A Unified Self-Supervised Encoder for Multi-Sensor Satellite Imagery**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV, eess-IV, stat-AP  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.02199v1)  

---


**ABSTRACT**  
Large, self-supervised vision models have led to substantial advancements for automatically interpreting natural images. Recent works have begun tailoring these methods to remote sensing data which has rich structure with multi-sensor, multi-spectral, and temporal information providing massive amounts of self-labeled data that can be used for self-supervised pre-training. In this work, we develop a new encoder architecture called USat that can input multi-spectral data from multiple sensors for self-supervised pre-training. USat is a vision transformer with modified patch projection layers and positional encodings to model spectral bands with varying spatial scales from multiple sensors. We integrate USat into a Masked Autoencoder (MAE) self-supervised pre-training procedure and find that a pre-trained USat outperforms state-of-the-art self-supervised MAE models trained on remote sensing data on multiple remote sensing benchmark datasets (up to 8%) and leads to improvements in low data regimes (up to 7%). Code and pre-trained weights are available at https://github.com/stanfordmlgroup/USat .

{{</citation>}}


### (39/68) Bootstrapping Interactive Image-Text Alignment for Remote Sensing Image Captioning (Cong Yang et al., 2023)

{{<citation>}}

Cong Yang, Zuchao Li, Lefei Zhang. (2023)  
**Bootstrapping Interactive Image-Text Alignment for Remote Sensing Image Captioning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Captioning, Transformer  
[Paper Link](http://arxiv.org/abs/2312.01191v1)  

---


**ABSTRACT**  
Recently, remote sensing image captioning has gained significant attention in the remote sensing community. Due to the significant differences in spatial resolution of remote sensing images, existing methods in this field have predominantly concentrated on the fine-grained extraction of remote sensing image features, but they cannot effectively handle the semantic consistency between visual features and textual features. To efficiently align the image-text, we propose a novel two-stage vision-language pre-training-based approach to bootstrap interactive image-text alignment for remote sensing image captioning, called BITA, which relies on the design of a lightweight interactive Fourier Transformer to better align remote sensing image-text features. The Fourier layer in the interactive Fourier Transformer is capable of extracting multi-scale features of remote sensing images in the frequency domain, thereby reducing the redundancy of remote sensing visual features. Specifically, the first stage involves preliminary alignment through image-text contrastive learning, which aligns the learned multi-scale remote sensing features from the interactive Fourier Transformer with textual features. In the second stage, the interactive Fourier Transformer connects the frozen image encoder with a large language model. Then, prefix causal language modeling is utilized to guide the text generation process using visual features. Ultimately, across the UCM-caption, RSICD, and NWPU-caption datasets, the experimental results clearly demonstrate that BITA outperforms other advanced comparative approaches. The code is available at https://github.com/yangcong356/BITA.

{{</citation>}}


### (40/68) SASSL: Enhancing Self-Supervised Learning via Neural Style Transfer (Renan A. Rojas-Gomez et al., 2023)

{{<citation>}}

Renan A. Rojas-Gomez, Karan Singhal, Ali Etemad, Alex Bijamov, Warren R. Morningstar, Philip Andrew Mansfield. (2023)  
**SASSL: Enhancing Self-Supervised Learning via Neural Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, stat-ML  
Keywords: Augmentation, ImageNet, Self-Supervised, Style Transfer  
[Paper Link](http://arxiv.org/abs/2312.01187v1)  

---


**ABSTRACT**  
Self-supervised learning relies heavily on data augmentation to extract meaningful representations from unlabeled images. While existing state-of-the-art augmentation pipelines incorporate a wide range of primitive transformations, these often disregard natural image structure. Thus, augmented samples can exhibit degraded semantic information and low stylistic diversity, affecting downstream performance of self-supervised representations. To overcome this, we propose SASSL: Style Augmentations for Self Supervised Learning, a novel augmentation technique based on Neural Style Transfer. The method decouples semantic and stylistic attributes in images and applies transformations exclusively to the style while preserving content, generating diverse augmented samples that better retain their semantic properties. Experimental results show our technique achieves a top-1 classification performance improvement of more than 2% on ImageNet compared to the well-established MoCo v2. We also measure transfer learning performance across five diverse datasets, observing significant improvements of up to 3.75%. Our experiments indicate that decoupling style from content information and transferring style across datasets to diversify augmentations can significantly improve downstream performance of self-supervised representations.

{{</citation>}}


### (41/68) IDPL-PFOD2: A New Large-Scale Dataset for Printed Farsi Optical Character Recognition (Fatemeh Asadi-zeydabadi et al., 2023)

{{<citation>}}

Fatemeh Asadi-zeydabadi, Ali Afkari-Fahandari, Amin Faraji, Elham Shabaninia, Hossein Nezamabadi-pour. (2023)  
**IDPL-PFOD2: A New Large-Scale Dataset for Printed Farsi Optical Character Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-DB, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.01177v1)  

---


**ABSTRACT**  
Optical Character Recognition is a technique that converts document images into searchable and editable text, making it a valuable tool for processing scanned documents. While the Farsi language stands as a prominent and official language in Asia, efforts to develop efficient methods for recognizing Farsi printed text have been relatively limited. This is primarily attributed to the languages distinctive features, such as cursive form, the resemblance between certain alphabet characters, and the presence of numerous diacritics and dot placement. On the other hand, given the substantial training sample requirements of deep-based architectures for effective performance, the development of such datasets holds paramount significance. In light of these concerns, this paper aims to present a novel large-scale dataset, IDPL-PFOD2, tailored for Farsi printed text recognition. The dataset comprises 2003541 images featuring a wide variety of fonts, styles, and sizes. This dataset is an extension of the previously introduced IDPL-PFOD dataset, offering a substantial increase in both volume and diversity. Furthermore, the datasets effectiveness is assessed through the utilization of both CRNN-based and Vision Transformer architectures. The CRNN-based model achieves a baseline accuracy rate of 78.49% and a normalized edit distance of 97.72%, while the Vision Transformer architecture attains an accuracy of 81.32% and a normalized edit distance of 98.74%.

{{</citation>}}


### (42/68) Virtual Category Learning: A Semi-Supervised Learning Method for Dense Prediction with Extremely Limited Labels (Changrui Chen et al., 2023)

{{<citation>}}

Changrui Chen, Jungong Han, Kurt Debattista. (2023)  
**Virtual Category Learning: A Semi-Supervised Learning Method for Dense Prediction with Extremely Limited Labels**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.01169v1)  

---


**ABSTRACT**  
Due to the costliness of labelled data in real-world applications, semi-supervised learning, underpinned by pseudo labelling, is an appealing solution. However, handling confusing samples is nontrivial: discarding valuable confusing samples would compromise the model generalisation while using them for training would exacerbate the issue of confirmation bias caused by the resulting inevitable mislabelling. To solve this problem, this paper proposes to use confusing samples proactively without label correction. Specifically, a Virtual Category (VC) is assigned to each confusing sample in such a way that it can safely contribute to the model optimisation even without a concrete label. This provides an upper bound for inter-class information sharing capacity, which eventually leads to a better embedding space. Extensive experiments on two mainstream dense prediction tasks -- semantic segmentation and object detection, demonstrate that the proposed VC learning significantly surpasses the state-of-the-art, especially when only very few labels are available. Our intriguing findings highlight the usage of VC learning in dense vision tasks.

{{</citation>}}


### (43/68) Meta-Learned Attribute Self-Interaction Network for Continual and Generalized Zero-Shot Learning (Vinay K Verma et al., 2023)

{{<citation>}}

Vinay K Verma, Nikhil Mehta, Kevin J Liang, Aakansha Mishra, Lawrence Carin. (2023)  
**Meta-Learned Attribute Self-Interaction Network for Continual and Generalized Zero-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, stat-ML  
Keywords: AI, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.01167v1)  

---


**ABSTRACT**  
Zero-shot learning (ZSL) is a promising approach to generalizing a model to categories unseen during training by leveraging class attributes, but challenges remain. Recently, methods using generative models to combat bias towards classes seen during training have pushed state of the art, but these generative models can be slow or computationally expensive to train. Also, these generative models assume that the attribute vector of each unseen class is available a priori at training, which is not always practical. Additionally, while many previous ZSL methods assume a one-time adaptation to unseen classes, in reality, the world is always changing, necessitating a constant adjustment of deployed models. Models unprepared to handle a sequential stream of data are likely to experience catastrophic forgetting. We propose a Meta-learned Attribute self-Interaction Network (MAIN) for continual ZSL. By pairing attribute self-interaction trained using meta-learning with inverse regularization of the attribute encoder, we are able to outperform state-of-the-art results without leveraging the unseen class attributes while also being able to train our models substantially faster (>100x) than expensive generative-based approaches. We demonstrate this with experiments on five standard ZSL datasets (CUB, aPY, AWA1, AWA2, and SUN) in the generalized zero-shot learning and continual (fixed/dynamic) zero-shot learning settings. Extensive ablations and analyses demonstrate the efficacy of various components proposed.

{{</citation>}}


### (44/68) Exploiting Diffusion Priors for All-in-One Image Restoration (Yuanbiao Gou et al., 2023)

{{<citation>}}

Yuanbiao Gou, Haiyu Zhao, Boyun Li, Xinyan Xiao, Xi Peng. (2023)  
**Exploiting Diffusion Priors for All-in-One Image Restoration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.02197v1)  

---


**ABSTRACT**  
All-in-one aims to solve various tasks of image restoration in a single model. To this end, we present a feasible way of exploiting the image priors captured by the pretrained diffusion model, through addressing the two challenges, i.e., degradation modeling and diffusion guidance. The former aims to simulate the process of the clean image degenerated by certain degradations, and the latter aims at guiding the diffusion model to generate the corresponding clean image. With the motivations, we propose a zero-shot framework for all-in-one image restoration, termed ZeroAIR, which alternatively performs the test-time degradation modeling (TDM) and the three-stage diffusion guidance (TDG) at each timestep of the reverse sampling. To be specific, TDM exploits the diffusion priors to learn a degradation model from a given degraded image, and TDG divides the timesteps into three stages for taking full advantage of the varying diffusion priors. Thanks to their degradation-agnostic property, the all-in-one image restoration could be achieved in a zero-shot way by ZeroAIR. Through extensive experiments, we show that our ZeroAIR achieves comparable even better performance than those task-specific methods. The code will be available on Github.

{{</citation>}}


### (45/68) Beyond Accuracy: Statistical Measures and Benchmark for Evaluation of Representation from Self-Supervised Learning (Jiantao Wu et al., 2023)

{{<citation>}}

Jiantao Wu, Shentong Mo, Sara Atito, Josef Kittler, Zhenhua Feng, Muhammad Awais. (2023)  
**Beyond Accuracy: Statistical Measures and Benchmark for Evaluation of Representation from Self-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.01118v1)  

---


**ABSTRACT**  
Recently, self-supervised metric learning has raised attention for the potential to learn a generic distance function. It overcomes the limitations of conventional supervised one, e.g., scalability and label biases. Despite progress in this domain, current benchmarks, incorporating a narrow scope of classes, stop the nuanced evaluation of semantic representations. To bridge this gap, we introduce a large-scale benchmark with diversity and granularity of classes, Statistical Metric Learning Benchmark (SMLB) built upon ImageNet-21K and WordNet. SMLB is designed to rigorously evaluate the discriminative discernment and generalizability across more than 14M images, 20K classes, and 16K taxonomic nodes. Alongside, we propose novel evaluation metrics -- `overlap' for separability and `aSTD' for consistency -- to measure distance statistical information, which are efficient and robust to the change of class number. Our benchmark offers a novel perspective of evaluating the quality of representations beyond accuracy. Our findings reveal the limitations of supervised learning and the class bias inherent in SSL models, offering insights into potential areas for future model enhancement.

{{</citation>}}


### (46/68) Local Masking Meets Progressive Freezing: Crafting Efficient Vision Transformers for Self-Supervised Learning (Utku Mert Topcuoglu et al., 2023)

{{<citation>}}

Utku Mert Topcuoglu, Erdem Akagündüz. (2023)  
**Local Masking Meets Progressive Freezing: Crafting Efficient Vision Transformers for Self-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.02194v1)  

---


**ABSTRACT**  
In this paper, we present an innovative approach to self-supervised learning for Vision Transformers (ViTs), integrating local masked image modeling with progressive layer freezing. This method focuses on enhancing the efficiency and speed of initial layer training in ViTs. By systematically freezing specific layers at strategic points during training, we reduce computational demands while maintaining or improving learning capabilities. Our approach employs a novel multi-scale reconstruction process that fosters efficient learning in initial layers and enhances semantic comprehension across scales. The results demonstrate a substantial reduction in training time (~12.5\%) with a minimal impact on model accuracy (decrease in top-1 accuracy by 0.6\%). Our method achieves top-1 and top-5 accuracies of 82.6\% and 96.2\%, respectively, underscoring its potential in scenarios where computational resources and time are critical. This work marks an advancement in the field of self-supervised learning for computer vision. The implementation of our approach is available at our project's GitHub repository: github.com/utkutpcgl/ViTFreeze.

{{</citation>}}


### (47/68) S2P3: Self-Supervised Polarimetric Pose Prediction (Patrick Ruhkamp et al., 2023)

{{<citation>}}

Patrick Ruhkamp, Daoyi Gao, Nassir Navab, Benjamin Busam. (2023)  
**S2P3: Self-Supervised Polarimetric Pose Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.01105v1)  

---


**ABSTRACT**  
This paper proposes the first self-supervised 6D object pose prediction from multimodal RGB+polarimetric images. The novel training paradigm comprises 1) a physical model to extract geometric information of polarized light, 2) a teacher-student knowledge distillation scheme and 3) a self-supervised loss formulation through differentiable rendering and an invertible physical constraint. Both networks leverage the physical properties of polarized light to learn robust geometric representations by encoding shape priors and polarization characteristics derived from our physical model. Geometric pseudo-labels from the teacher support the student network without the need for annotated real data. Dense appearance and geometric information of objects are obtained through a differentiable renderer with the predicted pose for self-supervised direct coupling. The student network additionally features our proposed invertible formulation of the physical shape priors that enables end-to-end self-supervised training through physical constraints of derived polarization characteristics compared against polarimetric input images. We specifically focus on photometrically challenging objects with texture-less or reflective surfaces and transparent materials for which the most prominent performance gain is reported.

{{</citation>}}


### (48/68) Rethinking Multiple Instance Learning for Whole Slide Image Classification: A Bag-Level Classifier is a Good Instance-Level Teacher (Hongyi Wang et al., 2023)

{{<citation>}}

Hongyi Wang, Luyang Luo, Fang Wang, Ruofeng Tong, Yen-Wei Chen, Hongjie Hu, Lanfen Lin, Hao Chen. (2023)  
**Rethinking Multiple Instance Learning for Whole Slide Image Classification: A Bag-Level Classifier is a Good Instance-Level Teacher**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2312.01099v1)  

---


**ABSTRACT**  
Multiple Instance Learning (MIL) has demonstrated promise in Whole Slide Image (WSI) classification. However, a major challenge persists due to the high computational cost associated with processing these gigapixel images. Existing methods generally adopt a two-stage approach, comprising a non-learnable feature embedding stage and a classifier training stage. Though it can greatly reduce the memory consumption by using a fixed feature embedder pre-trained on other domains, such scheme also results in a disparity between the two stages, leading to suboptimal classification accuracy. To address this issue, we propose that a bag-level classifier can be a good instance-level teacher. Based on this idea, we design Iteratively Coupled Multiple Instance Learning (ICMIL) to couple the embedder and the bag classifier at a low cost. ICMIL initially fix the patch embedder to train the bag classifier, followed by fixing the bag classifier to fine-tune the patch embedder. The refined embedder can then generate better representations in return, leading to a more accurate classifier for the next iteration. To realize more flexible and more effective embedder fine-tuning, we also introduce a teacher-student framework to efficiently distill the category knowledge in the bag classifier to help the instance-level embedder fine-tuning. Thorough experiments were conducted on four distinct datasets to validate the effectiveness of ICMIL. The experimental results consistently demonstrate that our method significantly improves the performance of existing MIL backbones, achieving state-of-the-art results. The code is available at: https://github.com/Dootmaan/ICMIL/tree/confidence_based

{{</citation>}}


### (49/68) Planning as In-Painting: A Diffusion-Based Embodied Task Planning Framework for Environments under Uncertainty (Cheng-Fu Yang et al., 2023)

{{<citation>}}

Cheng-Fu Yang, Haoyang Xu, Te-Lin Wu, Xiaofeng Gao, Kai-Wei Chang, Feng Gao. (2023)  
**Planning as In-Painting: A Diffusion-Based Embodied Task Planning Framework for Environments under Uncertainty**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.01097v1)  

---


**ABSTRACT**  
Task planning for embodied AI has been one of the most challenging problems where the community does not meet a consensus in terms of formulation. In this paper, we aim to tackle this problem with a unified framework consisting of an end-to-end trainable method and a planning algorithm. Particularly, we propose a task-agnostic method named 'planning as in-painting'. In this method, we use a Denoising Diffusion Model (DDM) for plan generation, conditioned on both language instructions and perceptual inputs under partially observable environments. Partial observation often leads to the model hallucinating the planning. Therefore, our diffusion-based method jointly models both state trajectory and goal estimation to improve the reliability of the generated plan, given the limited available information at each step. To better leverage newly discovered information along the plan execution for a higher success rate, we propose an on-the-fly planning algorithm to collaborate with the diffusion-based planner. The proposed framework achieves promising performances in various embodied AI tasks, including vision-language navigation, object manipulation, and task planning in a photorealistic virtual environment. The code is available at: https://github.com/joeyy5588/planning-as-inpainting.

{{</citation>}}


### (50/68) Consistency Prototype Module and Motion Compensation for Few-Shot Action Recognition (CLIP-CP$\mathbf{M^2}$C) (Fei Guo et al., 2023)

{{<citation>}}

Fei Guo, Li Zhu, YiKang Wang, Han Qi. (2023)  
**Consistency Prototype Module and Motion Compensation for Few-Shot Action Recognition (CLIP-CP$\mathbf{M^2}$C)**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.01083v1)  

---


**ABSTRACT**  
Recently, few-shot action recognition has significantly progressed by learning the feature discriminability and designing suitable comparison methods. Still, there are the following restrictions. (a) Previous works are mainly based on visual mono-modal. Although some multi-modal works use labels as supplementary to construct prototypes of support videos, they can not use this information for query videos. The labels are not used efficiently. (b) Most of the works ignore the motion feature of video, although the motion features are essential for distinguishing. We proposed a Consistency Prototype and Motion Compensation Network(CLIP-CP$M^2$C) to address these issues. Firstly, we use the CLIP for multi-modal few-shot action recognition with the text-image comparison for domain adaption. Secondly, in order to make the amount of information between the prototype and the query more similar, we propose a novel method to compensate for the text(prompt) information of query videos when text(prompt) does not exist, which depends on a Consistency Loss. Thirdly, we use the differential features of the adjacent frames in two directions as the motion features, which explicitly embeds the network with motion dynamics. We also apply the Consistency Loss to the motion features. Extensive experiments on standard benchmark datasets demonstrate that the proposed method can compete with state-of-the-art results. Our code is available at the URL: https://github.com/xxx/xxx.git.

{{</citation>}}


### (51/68) DiverseDream: Diverse Text-to-3D Synthesis with Augmented Text Embedding (Uy Dieu Tran et al., 2023)

{{<citation>}}

Uy Dieu Tran, Minh Luu, Phong Nguyen, Janne Heikkila, Khoi Nguyen, Binh-Son Hua. (2023)  
**DiverseDream: Diverse Text-to-3D Synthesis with Augmented Text Embedding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.02192v1)  

---


**ABSTRACT**  
Text-to-3D synthesis has recently emerged as a new approach to sampling 3D models by adopting pretrained text-to-image models as guiding visual priors. An intriguing but underexplored problem with existing text-to-3D methods is that 3D models obtained from the sampling-by-optimization procedure tend to have mode collapses, and hence poor diversity in their results. In this paper, we provide an analysis and identify potential causes of such a limited diversity, and then devise a new method that considers the joint generation of different 3D models from the same text prompt, where we propose to use augmented text prompts via textual inversion of reference images to diversify the joint generation. We show that our method leads to improved diversity in text-to-3D synthesis qualitatively and quantitatively.

{{</citation>}}


### (52/68) Spectrum-driven Mixed-frequency Network for Hyperspectral Salient Object Detection (Peifu Liu et al., 2023)

{{<citation>}}

Peifu Liu, Tingfa Xu, Huan Chen, Shiyun Zhou, Haolin Qin, Jianan Li. (2023)  
**Spectrum-driven Mixed-frequency Network for Hyperspectral Salient Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Object Detection  
[Paper Link](http://arxiv.org/abs/2312.01060v1)  

---


**ABSTRACT**  
Hyperspectral salient object detection (HSOD) aims to detect spectrally salient objects in hyperspectral images (HSIs). However, existing methods inadequately utilize spectral information by either converting HSIs into false-color images or converging neural networks with clustering. We propose a novel approach that fully leverages the spectral characteristics by extracting two distinct frequency components from the spectrum: low-frequency Spectral Saliency and high-frequency Spectral Edge. The Spectral Saliency approximates the region of salient objects, while the Spectral Edge captures edge information of salient objects. These two complementary components, crucial for HSOD, are derived by computing from the inter-layer spectral angular distance of the Gaussian pyramid and the intra-neighborhood spectral angular gradients, respectively. To effectively utilize this dual-frequency information, we introduce a novel lightweight Spectrum-driven Mixed-frequency Network (SMN). SMN incorporates two parameter-free plug-and-play operators, namely Spectral Saliency Generator and Spectral Edge Operator, to extract the Spectral Saliency and Spectral Edge components from the input HSI independently. Subsequently, the Mixed-frequency Attention module, comprised of two frequency-dependent heads, intelligently combines the embedded features of edge and saliency information, resulting in a mixed-frequency feature representation. Furthermore, a saliency-edge-aware decoder progressively scales up the mixed-frequency feature while preserving rich detail and saliency information for accurate salient object prediction. Extensive experiments conducted on the HS-SOD benchmark and our custom dataset HSOD-BIT demonstrate that our SMN outperforms state-of-the-art methods regarding HSOD performance. Code and dataset will be available at https://github.com/laprf/SMN.

{{</citation>}}


### (53/68) Prompt Tuning for Zero-shot Compositional Learning (Lingyu Zhang et al., 2023)

{{<citation>}}

Lingyu Zhang, Ting Hua, Yilin Shen, Hongxia Jin. (2023)  
**Prompt Tuning for Zero-shot Compositional Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.02191v1)  

---


**ABSTRACT**  
Open World Compositional Zero-Shot Learning (OW-CZSL) is known to be an extremely challenging task, which aims to recognize unseen compositions formed from seen attributes and objects without any prior assumption of the output space. In order to achieve this goal, a model has to be "smart" and "knowledgeable". To be smart, a model should be good at reasoning the interactions between attributes and objects from the seen compositions. While "knowledgeable" means the model owns "common sense" to the open world that can "foresee" some features of the unseen compositions. Most previous work focuses on the "smart" part, while few of them provided an effective solution to achieve the "knowledgeable" goal. In this paper, we proposed a framework named Multi-Modal Prompt Tuning (MMPT) to inherit the "knowledgeable" property from the large pre-trained vision-language model. Extensive experiments show that our proposed MMPT obtains new state-of-the-art results in OW-CZSL task. On the UT-Zappos dataset, MMPT pushes the AUC score to $29.8$, while the previous best score is $26.5$. On the more challenging MIT-States dataset, the AUC score of MMPT is 1.5 times better than the current state-of-the-art.

{{</citation>}}


### (54/68) Token Fusion: Bridging the Gap between Token Pruning and Token Merging (Minchul Kim et al., 2023)

{{<citation>}}

Minchul Kim, Shangqian Gao, Yen-Chang Hsu, Yilin Shen, Hongxia Jin. (2023)  
**Token Fusion: Bridging the Gap between Token Pruning and Token Merging**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.01026v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) have emerged as powerful backbones in computer vision, outperforming many traditional CNNs. However, their computational overhead, largely attributed to the self-attention mechanism, makes deployment on resource-constrained edge devices challenging. Multiple solutions rely on token pruning or token merging. In this paper, we introduce "Token Fusion" (ToFu), a method that amalgamates the benefits of both token pruning and token merging. Token pruning proves advantageous when the model exhibits sensitivity to input interpolations, while token merging is effective when the model manifests close to linear responses to inputs. We combine this to propose a new scheme called Token Fusion. Moreover, we tackle the limitations of average merging, which doesn't preserve the intrinsic feature norm, resulting in distributional shifts. To mitigate this, we introduce MLERP merging, a variant of the SLERP technique, tailored to merge multiple tokens while maintaining the norm distribution. ToFu is versatile, applicable to ViTs with or without additional training. Our empirical evaluations indicate that ToFu establishes new benchmarks in both classification and image generation tasks concerning computational efficiency and model accuracy.

{{</citation>}}


### (55/68) Unveiling the Power of Audio-Visual Early Fusion Transformers with Dense Interactions through Masked Modeling (Shentong Mo et al., 2023)

{{<citation>}}

Shentong Mo, Pedro Morgado. (2023)  
**Unveiling the Power of Audio-Visual Early Fusion Transformers with Dense Interactions through Masked Modeling**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs-SD, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.01017v1)  

---


**ABSTRACT**  
Humans possess a remarkable ability to integrate auditory and visual information, enabling a deeper understanding of the surrounding environment. This early fusion of audio and visual cues, demonstrated through cognitive psychology and neuroscience research, offers promising potential for developing multimodal perception models. However, training early fusion architectures poses significant challenges, as the increased model expressivity requires robust learning frameworks to harness their enhanced capabilities. In this paper, we address this challenge by leveraging the masked reconstruction framework, previously successful in unimodal settings, to train audio-visual encoders with early fusion. Additionally, we propose an attention-based fusion module that captures interactions between local audio and visual representations, enhancing the model's ability to capture fine-grained interactions. While effective, this procedure can become computationally intractable, as the number of local representations increases. Thus, to address the computational complexity, we propose an alternative procedure that factorizes the local representations before representing audio-visual interactions. Extensive evaluations on a variety of datasets demonstrate the superiority of our approach in audio-event classification, visual sound localization, sound separation, and audio-visual segmentation. These contributions enable the efficient training of deeply integrated audio-visual models and significantly advance the usefulness of early fusion architectures.

{{</citation>}}


## math.OC (1)



### (56/68) Mixed-Integer Optimisation of Graph Neural Networks for Computer-Aided Molecular Design (Tom McDonald et al., 2023)

{{<citation>}}

Tom McDonald, Calvin Tsay, Artur M. Schweidtmann, Neil Yorke-Smith. (2023)  
**Mixed-Integer Optimisation of Graph Neural Networks for Computer-Aided Molecular Design**  

---
Primary Category: math.OC  
Categories: 90C11, G-1-6; I-2-6; J-2, cs-NE, math-OC, math.OC  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.01228v1)  

---


**ABSTRACT**  
ReLU neural networks have been modelled as constraints in mixed integer linear programming (MILP), enabling surrogate-based optimisation in various domains and efficient solution of machine learning certification problems. However, previous works are mostly limited to MLPs. Graph neural networks (GNNs) can learn from non-euclidean data structures such as molecular structures efficiently and are thus highly relevant to computer-aided molecular design (CAMD). We propose a bilinear formulation for ReLU Graph Convolutional Neural Networks and a MILP formulation for ReLU GraphSAGE models. These formulations enable solving optimisation problems with trained GNNs embedded to global optimality. We apply our optimization approach to an illustrative CAMD case study where the formulations of the trained GNNs are used to design molecules with optimal boiling points.

{{</citation>}}


## cs.SI (1)



### (57/68) Understanding Opinions Towards Climate Change on Social Media (Yashaswi Pupneja et al., 2023)

{{<citation>}}

Yashaswi Pupneja, Joseph Zou, Sacha Lévy, Shenyang Huang. (2023)  
**Understanding Opinions Towards Climate Change on Social Media**  

---
Primary Category: cs.SI  
Categories: cs-CL, cs-LG, cs-SI, cs.SI  
Keywords: Natural Language Processing, Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2312.01217v1)  

---


**ABSTRACT**  
Social media platforms such as Twitter (now known as X) have revolutionized how the public engage with important societal and political topics. Recently, climate change discussions on social media became a catalyst for political polarization and the spreading of misinformation. In this work, we aim to understand how real world events influence the opinions of individuals towards climate change related topics on social media. To this end, we extracted and analyzed a dataset of 13.6 millions tweets sent by 3.6 million users from 2006 to 2019. Then, we construct a temporal graph from the user-user mentions network and utilize the Louvain community detection algorithm to analyze the changes in community structure around Conference of the Parties on Climate Change~(COP) events. Next, we also apply tools from the Natural Language Processing literature to perform sentiment analysis and topic modeling on the tweets. Our work acts as a first step towards understanding the evolution of pro-climate change communities around COP events. Answering these questions helps us understand how to raise people's awareness towards climate change thus hopefully calling on more individuals to join the collaborative effort in slowing down climate change.

{{</citation>}}


## cs.HC (1)



### (58/68) From Voices to Validity: Leveraging Large Language Models (LLMs) for Textual Analysis of Policy Stakeholder Interviews (Alex Liu et al., 2023)

{{<citation>}}

Alex Liu, Min Sun. (2023)  
**From Voices to Validity: Leveraging Large Language Models (LLMs) for Textual Analysis of Policy Stakeholder Interviews**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CL, cs-HC, cs.HC  
Keywords: GPT, GPT-4, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.01202v1)  

---


**ABSTRACT**  
Obtaining stakeholders' diverse experiences and opinions about current policy in a timely manner is crucial for policymakers to identify strengths and gaps in resource allocation, thereby supporting effective policy design and implementation. However, manually coding even moderately sized interview texts or open-ended survey responses from stakeholders can often be labor-intensive and time-consuming. This study explores the integration of Large Language Models (LLMs)--like GPT-4--with human expertise to enhance text analysis of stakeholder interviews regarding K-12 education policy within one U.S. state. Employing a mixed-methods approach, human experts developed a codebook and coding processes as informed by domain knowledge and unsupervised topic modeling results. They then designed prompts to guide GPT-4 analysis and iteratively evaluate different prompts' performances. This combined human-computer method enabled nuanced thematic and sentiment analysis. Results reveal that while GPT-4 thematic coding aligned with human coding by 77.89% at specific themes, expanding to broader themes increased congruence to 96.02%, surpassing traditional Natural Language Processing (NLP) methods by over 25%. Additionally, GPT-4 is more closely matched to expert sentiment analysis than lexicon-based methods. Findings from quantitative measures and qualitative reviews underscore the complementary roles of human domain expertise and automated analysis as LLMs offer new perspectives and coding consistency. The human-computer interactive approach enhances efficiency, validity, and interpretability of educational policy research.

{{</citation>}}


## cs.CY (2)



### (59/68) A Comparative Analysis of Text-to-Image Generative AI Models in Scientific Contexts: A Case Study on Nuclear Power (Veda Joynt et al., 2023)

{{<citation>}}

Veda Joynt, Jacob Cooper, Naman Bhargava, Katie Vu, O Hwang Kwon, Todd R. Allen, Aditi Verma, Majdi I. Radaideh. (2023)  
**A Comparative Analysis of Text-to-Image Generative AI Models in Scientific Contexts: A Case Study on Nuclear Power**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.01180v1)  

---


**ABSTRACT**  
In this work, we propose and assess the potential of generative artificial intelligence (AI) to generate public engagement around potential clean energy sources. Such an application could increase energy literacy -- an awareness of low-carbon energy sources among the public therefore leading to increased participation in decision-making about the future of energy systems. We explore the use of generative AI to communicate technical information about low-carbon energy sources to the general public, specifically in the realm of nuclear energy. We explored 20 AI-powered text-to-image generators and compared their individual performances on general and scientific nuclear-related prompts. Of these models, DALL-E, DreamStudio, and Craiyon demonstrated promising performance in generating relevant images from general-level text related to nuclear topics. However, these models fall short in three crucial ways: (1) they fail to accurately represent technical details of energy systems; (2) they reproduce existing biases surrounding gender and work in the energy sector; and (3) they fail to accurately represent indigenous landscapes -- which have historically been sites of resource extraction and waste deposition for energy industries. This work is performed to motivate the development of specialized generative tools and their captions to improve energy literacy and effectively engage the public with low-carbon energy sources.

{{</citation>}}


### (60/68) Here Is Not There: Measuring Entailment-Based Trajectory Similarity for Location-Privacy Protection and Beyond (Zilong Liu et al., 2023)

{{<citation>}}

Zilong Liu, Krzysztof Janowicz, Kitty Currier, Meilin Shi, Jinmeng Rao, Song Gao, Ling Cai, Anita Graser. (2023)  
**Here Is Not There: Measuring Entailment-Based Trajectory Similarity for Location-Privacy Protection and Beyond**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs-SC, cs.CY  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.01151v1)  

---


**ABSTRACT**  
While the paths humans take play out in social as well as physical space, measures to describe and compare their trajectories are carried out in abstract, typically Euclidean, space. When these measures are applied to trajectories of actual individuals in an application area, alterations that are inconsequential in abstract space may suddenly become problematic once overlaid with geographic reality. In this work, we present a different view on trajectory similarity by introducing a measure that utilizes logical entailment. This is an inferential perspective that considers facts as triple statements deduced from the social and environmental context in which the travel takes place, and their practical implications. We suggest a formalization of entailment-based trajectory similarity, measured as the overlapping proportion of facts, which are spatial relation statements in our case study. With the proposed measure, we evaluate LSTM-TrajGAN, a privacy-preserving trajectory-generation model. The entailment-based model evaluation reveals potential consequences of disregarding the rich structure of geographic space (e.g., miscalculated insurance risk due to regional shifts in our toy example). Our work highlights the advantage of applying logical entailment to trajectory-similarity reasoning for location-privacy protection and beyond.

{{</citation>}}


## cs.SD (1)



### (61/68) A Semi-Supervised Deep Learning Approach to Dataset Collection for Query-By-Humming Task (Amantur Amatov et al., 2023)

{{<citation>}}

Amantur Amatov, Dmitry Lamanov, Maksim Titov, Ivan Vovk, Ilya Makarov, Mikhail Kudinov. (2023)  
**A Semi-Supervised Deep Learning Approach to Dataset Collection for Query-By-Humming Task**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.01092v1)  

---


**ABSTRACT**  
Query-by-Humming (QbH) is a task that involves finding the most relevant song based on a hummed or sung fragment. Despite recent successful commercial solutions, implementing QbH systems remains challenging due to the lack of high-quality datasets for training machine learning models. In this paper, we propose a deep learning data collection technique and introduce Covers and Hummings Aligned Dataset (CHAD), a novel dataset that contains 18 hours of short music fragments, paired with time-aligned hummed versions. To expand our dataset, we employ a semi-supervised model training pipeline that leverages the QbH task as a specialized case of cover song identification (CSI) task. Starting with a model trained on the initial dataset, we iteratively collect groups of fragments of cover versions of the same song and retrain the model on the extended data. Using this pipeline, we collect over 308 hours of additional music fragments, paired with time-aligned cover versions. The final model is successfully applied to the QbH task and achieves competitive results on benchmark datasets. Our study shows that the proposed dataset and training pipeline can effectively facilitate the implementation of QbH systems.

{{</citation>}}


## cs.DL (1)



### (62/68) Scholarly Knowledge Graph Construction from Published Software Packages (Muhammad Haris et al., 2023)

{{<citation>}}

Muhammad Haris, Sören Auer, Markus Stocker. (2023)  
**Scholarly Knowledge Graph Construction from Published Software Packages**  

---
Primary Category: cs.DL  
Categories: cs-DL, cs.DL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.01065v1)  

---


**ABSTRACT**  
The value of structured scholarly knowledge for research and society at large is well understood, but producing scholarly knowledge (i.e., knowledge traditionally published in articles) in structured form remains a challenge. We propose an approach for automatically extracting scholarly knowledge from published software packages by static analysis of their metadata and contents (scripts and data) and populating a scholarly knowledge graph with the extracted knowledge. Our approach is based on mining scientific software packages linked to article publications by extracting metadata and analyzing the Abstract Syntax Tree (AST) of the source code to obtain information about the used and produced data as well as operations performed on data. The resulting knowledge graph includes articles, software packages metadata, and computational techniques applied to input data utilized as materials in research work. The knowledge graph also includes the results reported as scholarly knowledge in articles.

{{</citation>}}


## cs.MA (1)



### (63/68) A Survey of Progress on Cooperative Multi-agent Reinforcement Learning in Open Environment (Lei Yuan et al., 2023)

{{<citation>}}

Lei Yuan, Ziqian Zhang, Lihe Li, Cong Guan, Yang Yu. (2023)  
**A Survey of Progress on Cooperative Multi-agent Reinforcement Learning in Open Environment**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.01058v1)  

---


**ABSTRACT**  
Multi-agent Reinforcement Learning (MARL) has gained wide attention in recent years and has made progress in various fields. Specifically, cooperative MARL focuses on training a team of agents to cooperatively achieve tasks that are difficult for a single agent to handle. It has shown great potential in applications such as path planning, autonomous driving, active voltage control, and dynamic algorithm configuration. One of the research focuses in the field of cooperative MARL is how to improve the coordination efficiency of the system, while research work has mainly been conducted in simple, static, and closed environment settings. To promote the application of artificial intelligence in real-world, some research has begun to explore multi-agent coordination in open environments. These works have made progress in exploring and researching the environments where important factors might change. However, the mainstream work still lacks a comprehensive review of the research direction. In this paper, starting from the concept of reinforcement learning, we subsequently introduce multi-agent systems (MAS), cooperative MARL, typical methods, and test environments. Then, we summarize the research work of cooperative MARL from closed to open environments, extract multiple research directions, and introduce typical works. Finally, we summarize the strengths and weaknesses of the current research, and look forward to the future development direction and research problems in cooperative MARL in open environments.

{{</citation>}}


## stat.ML (2)



### (64/68) Bagged Regularized $k$-Distances for Anomaly Detection (Yuchao Cai et al., 2023)

{{<citation>}}

Yuchao Cai, Yuheng Ma, Hanfang Yang, Hanyuan Hang. (2023)  
**Bagged Regularized $k$-Distances for Anomaly Detection**  

---
Primary Category: stat.ML  
Categories: cs-LG, math-ST, stat-ML, stat-TH, stat.ML  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.01046v1)  

---


**ABSTRACT**  
We consider the paradigm of unsupervised anomaly detection, which involves the identification of anomalies within a dataset in the absence of labeled examples. Though distance-based methods are top-performing for unsupervised anomaly detection, they suffer heavily from the sensitivity to the choice of the number of the nearest neighbors. In this paper, we propose a new distance-based algorithm called bagged regularized $k$-distances for anomaly detection (BRDAD) converting the unsupervised anomaly detection problem into a convex optimization problem. Our BRDAD algorithm selects the weights by minimizing the surrogate risk, i.e., the finite sample bound of the empirical risk of the bagged weighted $k$-distances for density estimation (BWDDE). This approach enables us to successfully address the sensitivity challenge of the hyperparameter choice in distance-based algorithms. Moreover, when dealing with large-scale datasets, the efficiency issues can be addressed by the incorporated bagging technique in our BRDAD algorithm. On the theoretical side, we establish fast convergence rates of the AUC regret of our algorithm and demonstrate that the bagging technique significantly reduces the computational complexity. On the practical side, we conduct numerical experiments on anomaly detection benchmarks to illustrate the insensitivity of parameter selection of our algorithm compared with other state-of-the-art distance-based methods. Moreover, promising improvements are brought by applying the bagging technique in our algorithm on real-world datasets.

{{</citation>}}


### (65/68) Convergences for Minimax Optimization Problems over Infinite-Dimensional Spaces Towards Stability in Adversarial Training (Takashi Furuya et al., 2023)

{{<citation>}}

Takashi Furuya, Satoshi Okuda, Kazuma Suetake, Yoshihide Sawada. (2023)  
**Convergences for Minimax Optimization Problems over Infinite-Dimensional Spaces Towards Stability in Adversarial Training**  

---
Primary Category: stat.ML  
Categories: cs-LG, math-OC, stat-ML, stat.ML  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2312.00991v1)  

---


**ABSTRACT**  
Training neural networks that require adversarial optimization, such as generative adversarial networks (GANs) and unsupervised domain adaptations (UDAs), suffers from instability. This instability problem comes from the difficulty of the minimax optimization, and there have been various approaches in GANs and UDAs to overcome this problem. In this study, we tackle this problem theoretically through a functional analysis. Specifically, we show the convergence property of the minimax problem by the gradient descent over the infinite-dimensional spaces of continuous functions and probability measures under certain conditions. Using this setting, we can discuss GANs and UDAs comprehensively, which have been studied independently. In addition, we show that the conditions necessary for the convergence property are interpreted as stabilization techniques of adversarial training such as the spectral normalization and the gradient penalty.

{{</citation>}}


## quant-ph (1)



### (66/68) Optimal Clifford Initial States for Ising Hamiltonians (Bikrant Bhattacharyya et al., 2023)

{{<citation>}}

Bikrant Bhattacharyya, Gokul Subramanian Ravi. (2023)  
**Optimal Clifford Initial States for Ising Hamiltonians**  

---
Primary Category: quant-ph  
Categories: cs-AR, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.01036v1)  

---


**ABSTRACT**  
Evaluating quantum circuits is currently very noisy. Therefore, developing classical bootstraps that help minimize the number of times quantum circuits have to be executed on noisy quantum devices is a powerful technique for improving the practicality of Variational Quantum Algorithms. CAFQA is a previously proposed classical bootstrap for VQAs that uses an initial ansatz that reduces to Clifford operators. CAFQA has been shown to produce fairly accurate initialization for VQA applied to molecular chemistry Hamiltonians. Motivated by this result, in this paper we seek to analyze the Clifford states that optimize the cost function for a new type of Hamiltonian, namely Transverse Field Ising Hamiltonians. Our primary result connects the problem of finding the optimal CAFQA initialization to a submodular minimization problem which in turn can be solved in polynomial time.

{{</citation>}}


## q-fin.TR (1)



### (67/68) Decentralized Finance: Protocols, Risks, and Governance (Agostino Capponi et al., 2023)

{{<citation>}}

Agostino Capponi, Garud Iyengar, Jay Sethuraman. (2023)  
**Decentralized Finance: Protocols, Risks, and Governance**  

---
Primary Category: q-fin.TR  
Categories: cs-CY, q-fin-TR, q-fin.TR  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2312.01018v1)  

---


**ABSTRACT**  
Financial markets are undergoing an unprecedented transformation. Technological advances have brought major improvements to the operations of financial services. While these advances promote improved accessibility and convenience, traditional finance shortcomings like lack of transparency and moral hazard frictions continue to plague centralized platforms, imposing societal costs. In this paper, we argue how these shortcomings and frictions are being mitigated by the decentralized finance (DeFi) ecosystem. We delve into the workings of smart contracts, the backbone of DeFi transactions, with an emphasis on those underpinning token exchange and lending services. We highlight the pros and cons of the novel form of decentralized governance introduced via the ownership of governance tokens. Despite its potential, the current DeFi infrastructure introduces operational risks to users, which we segment into five primary categories: consensus mechanisms, protocol, oracle, frontrunning, and systemic risks. We conclude by emphasizing the need for future research to focus on the scalability of existing blockchains, the improved design and interoperability of DeFi protocols, and the rigorous auditing of smart contracts.

{{</citation>}}


## physics.med-ph (1)



### (68/68) Noisy probing dose facilitated dose prediction for pencil beam scanning proton therapy: physics enhances generalizability (Lian Zhang et al., 2023)

{{<citation>}}

Lian Zhang, Jason M. Holmes, Zhengliang Liu, Hongying Feng, Terence T. Sio, Carlos E. Vargas, Sameer R. Keole, Kristin Stützer, Sheng Li, Tianming Liu, Jiajian Shen, William W. Wong, Sujay A. Vora, Wei Liu. (2023)  
**Noisy probing dose facilitated dose prediction for pencil beam scanning proton therapy: physics enhances generalizability**  

---
Primary Category: physics.med-ph  
Categories: cs-LG, physics-med-ph, physics.med-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00975v1)  

---


**ABSTRACT**  
Purpose: Prior AI-based dose prediction studies in photon and proton therapy often neglect underlying physics, limiting their generalizability to handle outlier clinical cases, especially for pencil beam scanning proton therapy (PBSPT). Our aim is to design a physics-aware and generalizable AI-based PBSPT dose prediction method that has the underlying physics considered to achieve high generalizability to properly handle the outlier clinical cases. Methods and Materials: This study analyzed PBSPT plans of 103 prostate and 78 lung cancer patients from our institution,with each case comprising CT images, structure sets, and plan doses from our Monte-Carlo dose engine (serving as the ground truth). Three methods were evaluated in the ablation study: the ROI-based method, the beam mask and sliding window method, and the noisy probing dose method. Twelve cases with uncommon beam angles or prescription doses tested the methods' generalizability to rare treatment planning scenarios. Performance evaluation used DVH indices, 3D Gamma passing rates (3%/2mm/10%), and dice coefficients for dose agreement. Results: The noisy probing dose method showed improved agreement of DVH indices, 3D Gamma passing rates, and dice coefficients compared to the conventional methods for the testing cases. The noisy probing dose method showed better generalizability in the 6 outlier cases than the ROI-based and beam mask-based methods with 3D Gamma passing rates (for prostate cancer, targets: 89.32%$\pm$1.45% vs. 93.48%$\pm$1.51% vs. 96.79%$\pm$0.83%, OARs: 85.87%$\pm$1.73% vs. 91.15%$\pm$1.13% vs. 94.29%$\pm$1.01%). The dose predictions were completed within 0.3 seconds. Conclusions: We've devised a novel noisy probing dose method for PBSPT dose prediction in prostate and lung cancer patients. With more physics included, it enhances the generalizability of dose prediction in handling outlier clinical cases.

{{</citation>}}
