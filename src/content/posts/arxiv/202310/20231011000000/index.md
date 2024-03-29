---
draft: false
title: "arXiv @ 2023.10.11"
date: 2023-10-11
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.11"
    identifier: arxiv_20231011
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.RO (7)](#csro-7)
- [cs.CL (53)](#cscl-53)
- [cs.CV (28)](#cscv-28)
- [cs.PF (2)](#cspf-2)
- [cs.AI (16)](#csai-16)
- [cs.LG (38)](#cslg-38)
- [cs.HC (2)](#cshc-2)
- [eess.IV (6)](#eessiv-6)
- [stat.ML (3)](#statml-3)
- [cs.DC (4)](#csdc-4)
- [cs.SD (5)](#cssd-5)
- [cs.CY (3)](#cscy-3)
- [cs.CR (2)](#cscr-2)
- [quant-ph (1)](#quant-ph-1)
- [eess.AS (1)](#eessas-1)
- [eess.SY (1)](#eesssy-1)
- [cs.DM (1)](#csdm-1)
- [cs.NI (2)](#csni-2)
- [cs.SE (2)](#csse-2)
- [cs.CE (1)](#csce-1)
- [cs.SI (1)](#cssi-1)
- [cs.IT (1)](#csit-1)
- [cs.MM (1)](#csmm-1)
- [cs.IR (1)](#csir-1)
- [cs.DB (1)](#csdb-1)

## cs.RO (7)



### (1/183) Human-Robot Gym: Benchmarking Reinforcement Learning in Human-Robot Collaboration (Jakob Thumm et al., 2023)

{{<citation>}}

Jakob Thumm, Felix Trost, Matthias Althoff. (2023)  
**Human-Robot Gym: Benchmarking Reinforcement Learning in Human-Robot Collaboration**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.06208v1)  

---


**ABSTRACT**  
Deep reinforcement learning (RL) has shown promising results in robot motion planning with first attempts in human-robot collaboration (HRC). However, a fair comparison of RL approaches in HRC under the constraint of guaranteed safety is yet to be made. We, therefore, present human-robot gym, a benchmark for safe RL in HRC. Our benchmark provides eight challenging, realistic HRC tasks in a modular simulation framework. Most importantly, human-robot gym includes a safety shield that provably guarantees human safety. We are, thereby, the first to provide a benchmark to train RL agents that adhere to the safety specifications of real-world HRC. This bridges a critical gap between theoretic RL research and its real-world deployment. Our evaluation of six environments led to three key results: (a) the diverse nature of the tasks offered by human-robot gym creates a challenging benchmark for state-of-the-art RL methods, (b) incorporating expert knowledge in the RL training in the form of an action-based reward can outperform the expert, and (c) our agents negligibly overfit to training data.

{{</citation>}}


### (2/183) DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning in Autonomous Driving (Zhiyu Huang et al., 2023)

{{<citation>}}

Zhiyu Huang, Peter Karkus, Boris Ivanovic, Yuxiao Chen, Marco Pavone, Chen Lv. (2023)  
**DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning in Autonomous Driving**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.05885v1)  

---


**ABSTRACT**  
Motion prediction and cost evaluation are vital components in the decision-making system of autonomous vehicles. However, existing methods often ignore the importance of cost learning and treat them as separate modules. In this study, we employ a tree-structured policy planner and propose a differentiable joint training framework for both ego-conditioned prediction and cost models, resulting in a direct improvement of the final planning performance. For conditional prediction, we introduce a query-centric Transformer model that performs efficient ego-conditioned motion prediction. For planning cost, we propose a learnable context-aware cost function with latent interaction features, facilitating differentiable joint learning. We validate our proposed approach using the real-world nuPlan dataset and its associated planning test platform. Our framework not only matches state-of-the-art planning methods but outperforms other learning-based methods in planning quality, while operating more efficiently in terms of runtime. We show that joint training delivers significantly better performance than separate training of the two modules. Additionally, we find that tree-structured policy planning outperforms the conventional single-stage planning approach.

{{</citation>}}


### (3/183) A Learning-Based Framework for Safe Human-Robot Collaboration with Multiple Backup Control Barrier Functions (Neil C. Janwani et al., 2023)

{{<citation>}}

Neil C. Janwani, Ersin Daş, Thomas Touma, Skylar X. Wei, Tamas G. Molnar, Joel W. Burdick. (2023)  
**A Learning-Based Framework for Safe Human-Robot Collaboration with Multiple Backup Control Barrier Functions**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.05865v1)  

---


**ABSTRACT**  
Ensuring robot safety in complex environments is a difficult task due to actuation limits, such as torque bounds. This paper presents a safety-critical control framework that leverages learning-based switching between multiple backup controllers to formally guarantee safety under bounded control inputs while satisfying driver intention. By leveraging backup controllers designed to uphold safety and input constraints, backup control barrier functions (BCBFs) construct implicitly defined control invariance sets via a feasible quadratic program (QP). However, BCBF performance largely depends on the design and conservativeness of the chosen backup controller, especially in our setting of human-driven vehicles in complex, e.g, off-road, conditions. While conservativeness can be reduced by using multiple backup controllers, determining when to switch is an open problem. Consequently, we develop a broadcast scheme that estimates driver intention and integrates BCBFs with multiple backup strategies for human-robot interaction. An LSTM classifier uses data inputs from the robot, human, and safety algorithms to continually choose a backup controller in real-time. We demonstrate our method's efficacy on a dual-track robot in obstacle avoidance scenarios. Our framework guarantees robot safety while adhering to driver intention.

{{</citation>}}


### (4/183) A Simple Open-Loop Baseline for Reinforcement Learning Locomotion Tasks (Antonin Raffin et al., 2023)

{{<citation>}}

Antonin Raffin, Olivier Sigaud, Jens Kober, Alin Albu-Schäffer, João Silvério, Freek Stulp. (2023)  
**A Simple Open-Loop Baseline for Reinforcement Learning Locomotion Tasks**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05808v1)  

---


**ABSTRACT**  
In search of the simplest baseline capable of competing with Deep Reinforcement Learning on locomotion tasks, we propose a biologically inspired model-free open-loop strategy. Drawing upon prior knowledge and harnessing the elegance of simple oscillators to generate periodic joint motions, it achieves respectable performance in five different locomotion environments, with a number of tunable parameters that is a tiny fraction of the thousands typically required by RL algorithms. Unlike RL methods, which are prone to performance degradation when exposed to sensor noise or failure, our open-loop oscillators exhibit remarkable robustness due to their lack of reliance on sensors. Furthermore, we showcase a successful transfer from simulation to reality using an elastic quadruped, all without the need for randomization or reward engineering.

{{</citation>}}


### (5/183) DecAP: Decaying Action Priors for Accelerated Learning of Torque-Based Legged Locomotion Policies (Shivam Sood et al., 2023)

{{<citation>}}

Shivam Sood, Ge Sun, Peizhuo Li, Guillaume Sartoretti. (2023)  
**DecAP: Decaying Action Priors for Accelerated Learning of Torque-Based Legged Locomotion Policies**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05714v1)  

---


**ABSTRACT**  
Optimal Control for legged robots has gone through a paradigm shift from position-based to torque-based control, owing to the latter's compliant and robust nature. In parallel to this shift, the community has also turned to Deep Reinforcement Learning (DRL) as a promising approach to directly learn locomotion policies for complex real-life tasks. However, most end-to-end DRL approaches still operate in position space, mainly because learning in torque space is often sample-inefficient and does not consistently converge to natural gaits. To address these challenges, we introduce Decaying Action Priors (DecAP), a novel three-stage framework to learn and deploy torque policies for legged locomotion. In the first stage, we generate our own imitation data by training a position policy, eliminating the need for expert knowledge in designing optimal controllers. The second stage incorporates decaying action priors to enhance the exploration of torque-based policies aided by imitation rewards. We show that our approach consistently outperforms imitation learning alone and is significantly robust to the scaling of these rewards. Finally, our third stage facilitates safe sim-to-real transfer by directly deploying our learned torques, alongside low-gain PID control from our trained position policy. We demonstrate the generality of our approach by training torque-based locomotion policies for a biped, a quadruped, and a hexapod robot in simulation, and experimentally demonstrate our learned policies on a quadruped (Unitree Go1).

{{</citation>}}


### (6/183) Care3D: An Active 3D Object Detection Dataset of Real Robotic-Care Environments (Michael G. Adam et al., 2023)

{{<citation>}}

Michael G. Adam, Sebastian Eger, Martin Piccolrovazzi, Maged Iskandar, Joern Vogel, Alexander Dietrich, Seongjien Bien, Jon Skerlj, Abdeldjallil Naceri, Eckehard Steinbach, Alin Albu-Schaeffer, Sami Haddadin, Wolfram Burgard. (2023)  
**Care3D: An Active 3D Object Detection Dataset of Real Robotic-Care Environments**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.05600v1)  

---


**ABSTRACT**  
As labor shortage increases in the health sector, the demand for assistive robotics grows. However, the needed test data to develop those robots is scarce, especially for the application of active 3D object detection, where no real data exists at all. This short paper counters this by introducing such an annotated dataset of real environments. The captured environments represent areas which are already in use in the field of robotic health care research. We further provide ground truth data within one room, for assessing SLAM algorithms running directly on a health care robot.

{{</citation>}}


### (7/183) Ethics of Artificial Intelligence and Robotics in the Architecture, Engineering, and Construction Industry (Ci-Jyun Liang et al., 2023)

{{<citation>}}

Ci-Jyun Liang, Thai-Hoa Le, Youngjib Ham, Bharadwaj R. K. Mantha, Marvin H. Cheng, Jacob J. Lin. (2023)  
**Ethics of Artificial Intelligence and Robotics in the Architecture, Engineering, and Construction Industry**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05414v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) and robotics research and implementation emerged in the architecture, engineering, and construction (AEC) industry to positively impact project efficiency and effectiveness concerns such as safety, productivity, and quality. This shift, however, warrants the need for ethical considerations of AI and robotics adoption due to its potential negative impacts on aspects such as job security, safety, and privacy. Nevertheless, this did not receive sufficient attention, particularly within the academic community. This research systematically reviews AI and robotics research through the lens of ethics in the AEC community for the past five years. It identifies nine key ethical issues namely job loss, data privacy, data security, data transparency, decision-making conflict, acceptance and trust, reliability and safety, fear of surveillance, and liability, by summarizing existing literature and filtering it further based on its AEC relevance. Furthermore, thirteen research topics along the process were identified based on existing AEC studies that had direct relevance to the theme of ethics in general and their parallels are further discussed. Finally, the current challenges and knowledge gaps are discussed and seven specific future research directions are recommended. This study not only signifies more stakeholder awareness of this important topic but also provides imminent steps towards safer and more efficient realization.

{{</citation>}}


## cs.CL (53)



### (8/183) GPT-who: An Information Density-based Machine-Generated Text Detector (Saranya Venkatraman et al., 2023)

{{<citation>}}

Saranya Venkatraman, Adaku Uchendu, Dongwon Lee. (2023)  
**GPT-who: An Information Density-based Machine-Generated Text Detector**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06202v1)  

---


**ABSTRACT**  
The Uniform Information Density principle posits that humans prefer to spread information evenly during language production. In this work, we examine if the UID principle can help capture differences between Large Language Models (LLMs) and human-generated text. We propose GPT-who, the first psycholinguistically-aware multi-class domain-agnostic statistical-based detector. This detector employs UID-based features to model the unique statistical signature of each LLM and human author for accurate authorship attribution. We evaluate our method using 4 large-scale benchmark datasets and find that GPT-who outperforms state-of-the-art detectors (both statistical- & non-statistical-based) such as GLTR, GPTZero, OpenAI detector, and ZeroGPT by over $20$% across domains. In addition to superior performance, it is computationally inexpensive and utilizes an interpretable representation of text articles. We present the largest analysis of the UID-based representations of human and machine-generated texts (over 400k articles) to demonstrate how authors distribute information differently, and in ways that enable their detection using an off-the-shelf LM without any fine-tuning. We find that GPT-who can distinguish texts generated by very sophisticated LLMs, even when the overlying text is indiscernible.

{{</citation>}}


### (9/183) Compressing Context to Enhance Inference Efficiency of Large Language Models (Yucheng Li et al., 2023)

{{<citation>}}

Yucheng Li, Bo Dong, Chenghua Lin, Frank Guerin. (2023)  
**Compressing Context to Enhance Inference Efficiency of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06201v1)  

---


**ABSTRACT**  
Large language models (LLMs) achieved remarkable performance across various tasks. However, they face challenges in managing long documents and extended conversations, due to significantly increased computational requirements, both in memory and inference time, and potential context truncation when the input exceeds the LLM's fixed context length. This paper proposes a method called Selective Context that enhances the inference efficiency of LLMs by identifying and pruning redundancy in the input context to make the input more compact. We test our approach using common data sources requiring long context processing: arXiv papers, news articles, and long conversations, on tasks of summarisation, question answering, and response generation. Experimental results show that Selective Context significantly reduces memory cost and decreases generation latency while maintaining comparable performance compared to that achieved when full context is used. Specifically, we achieve a 50\% reduction in context cost, resulting in a 36\% reduction in inference memory usage and a 32\% reduction in inference time, while observing only a minor drop of .023 in BERTscore and .038 in faithfulness on four downstream applications, indicating that our method strikes a good balance between efficiency and performance.

{{</citation>}}


### (10/183) The Importance of Prompt Tuning for Automated Neuron Explanations (Justin Lee et al., 2023)

{{<citation>}}

Justin Lee, Tuomas Oikarinen, Arjun Chatha, Keng-Chi Chang, Yilan Chen, Tsui-Wei Weng. (2023)  
**The Importance of Prompt Tuning for Automated Neuron Explanations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.06200v2)  

---


**ABSTRACT**  
Recent advances have greatly increased the capabilities of large language models (LLMs), but our understanding of the models and their safety has not progressed as fast. In this paper we aim to understand LLMs deeper by studying their individual neurons. We build upon previous work showing large language models such as GPT-4 can be useful in explaining what each neuron in a language model does. Specifically, we analyze the effect of the prompt used to generate explanations and show that reformatting the explanation prompt in a more natural way can significantly improve neuron explanation quality and greatly reduce computational cost. We demonstrate the effects of our new prompts in three different ways, incorporating both automated and human evaluations.

{{</citation>}}


### (11/183) BYOC: Personalized Few-Shot Classification with Co-Authored Class Descriptions (Arth Bohra et al., 2023)

{{<citation>}}

Arth Bohra, Govert Verkes, Artem Harutyunyan, Pascal Weinberger, Giovanni Campagna. (2023)  
**BYOC: Personalized Few-Shot Classification with Co-Authored Class Descriptions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Few-Shot, NLP  
[Paper Link](http://arxiv.org/abs/2310.06111v1)  

---


**ABSTRACT**  
Text classification is a well-studied and versatile building block for many NLP applications. Yet, existing approaches require either large annotated corpora to train a model with or, when using large language models as a base, require carefully crafting the prompt as well as using a long context that can fit many examples. As a result, it is not possible for end-users to build classifiers for themselves. To address this issue, we propose a novel approach to few-shot text classification using an LLM. Rather than few-shot examples, the LLM is prompted with descriptions of the salient features of each class. These descriptions are coauthored by the user and the LLM interactively: while the user annotates each few-shot example, the LLM asks relevant questions that the user answers. Examples, questions, and answers are summarized to form the classification prompt. Our experiments show that our approach yields high accuracy classifiers, within 82% of the performance of models trained with significantly larger datasets while using only 1% of their training sets. Additionally, in a study with 30 participants, we show that end-users are able to build classifiers to suit their specific needs. The personalized classifiers show an average accuracy of 90%, which is 15% higher than the state-of-the-art approach.

{{</citation>}}


### (12/183) Leveraging Multilingual Self-Supervised Pretrained Models for Sequence-to-Sequence End-to-End Spoken Language Understanding (Pavel Denisov et al., 2023)

{{<citation>}}

Pavel Denisov, Ngoc Thang Vu. (2023)  
**Leveraging Multilingual Self-Supervised Pretrained Models for Sequence-to-Sequence End-to-End Spoken Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Multilingual, Self-Supervised, Sequence-to-Sequence, Spoken Language Understanding  
[Paper Link](http://arxiv.org/abs/2310.06103v1)  

---


**ABSTRACT**  
A number of methods have been proposed for End-to-End Spoken Language Understanding (E2E-SLU) using pretrained models, however their evaluation often lacks multilingual setup and tasks that require prediction of lexical fillers, such as slot filling. In this work, we propose a unified method that integrates multilingual pretrained speech and text models and performs E2E-SLU on six datasets in four languages in a generative manner, including the prediction of lexical fillers. We investigate how the proposed method can be improved by pretraining on widely available speech recognition data using several training objectives. Pretraining on 7000 hours of multilingual data allows us to outperform the state-of-the-art ultimately on two SLU datasets and partly on two more SLU datasets. Finally, we examine the cross-lingual capabilities of the proposed model and improve on the best known result on the PortMEDIA-Language dataset by almost half, achieving a Concept/Value Error Rate of 23.65%.

{{</citation>}}


### (13/183) Few-Shot Spoken Language Understanding via Joint Speech-Text Models (Chung-Ming Chien et al., 2023)

{{<citation>}}

Chung-Ming Chien, Mingjiamei Zhang, Ju-Chieh Chou, Karen Livescu. (2023)  
**Few-Shot Spoken Language Understanding via Joint Speech-Text Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-AS  
Keywords: Few-Shot, Spoken Language Understanding  
[Paper Link](http://arxiv.org/abs/2310.05919v1)  

---


**ABSTRACT**  
Recent work on speech representation models jointly pre-trained with text has demonstrated the potential of improving speech representations by encoding speech and text in a shared space. In this paper, we leverage such shared representations to address the persistent challenge of limited data availability in spoken language understanding tasks. By employing a pre-trained speech-text model, we find that models fine-tuned on text can be effectively transferred to speech testing data. With as little as 1 hour of labeled speech data, our proposed approach achieves comparable performance on spoken language understanding tasks (specifically, sentiment analysis and named entity recognition) when compared to previous methods using speech-only pre-trained models fine-tuned on 10 times more data. Beyond the proof-of-concept study, we also analyze the latent representations. We find that the bottom layers of speech-text models are largely task-agnostic and align speech and text representations into a shared space, while the top layers are more task-specific.

{{</citation>}}


### (14/183) FireAct: Toward Language Agent Fine-tuning (Baian Chen et al., 2023)

{{<citation>}}

Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, Shunyu Yao. (2023)  
**FireAct: Toward Language Agent Fine-tuning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Google, QA  
[Paper Link](http://arxiv.org/abs/2310.05915v1)  

---


**ABSTRACT**  
Recent efforts have augmented language models (LMs) with external tools or environments, leading to the development of language agents that can reason and act. However, most of these agents rely on few-shot prompting techniques with off-the-shelf LMs. In this paper, we investigate and argue for the overlooked direction of fine-tuning LMs to obtain language agents. Using a setup of question answering (QA) with a Google search API, we explore a variety of base LMs, prompting methods, fine-tuning data, and QA tasks, and find language agents are consistently improved after fine-tuning their backbone LMs. For example, fine-tuning Llama2-7B with 500 agent trajectories generated by GPT-4 leads to a 77% HotpotQA performance increase. Furthermore, we propose FireAct, a novel approach to fine-tuning LMs with trajectories from multiple tasks and prompting methods, and show having more diverse fine-tuning data can further improve agents. Along with other findings regarding scaling effects, robustness, generalization, efficiency and cost, our work establishes comprehensive benefits of fine-tuning LMs for agents, and provides an initial set of experimental designs, insights, as well as open questions toward language agent fine-tuning.

{{</citation>}}


### (15/183) NEFTune: Noisy Embeddings Improve Instruction Finetuning (Neel Jain et al., 2023)

{{<citation>}}

Neel Jain, Ping-yeh Chiang, Yuxin Wen, John Kirchenbauer, Hong-Min Chu, Gowthami Somepalli, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Aniruddha Saha, Micah Goldblum, Jonas Geiping, Tom Goldstein. (2023)  
**NEFTune: Noisy Embeddings Improve Instruction Finetuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Embedding, GPT, LLaMA  
[Paper Link](http://arxiv.org/abs/2310.05914v2)  

---


**ABSTRACT**  
We show that language model finetuning can be improved, sometimes dramatically, with a simple augmentation. NEFTune adds noise to the embedding vectors during training. Standard finetuning of LLaMA-2-7B using Alpaca achieves 29.79% on AlpacaEval, which rises to 64.69% using noisy embeddings. NEFTune also improves over strong baselines on modern instruction datasets. Models trained with Evol-Instruct see a 10% improvement, with ShareGPT an 8% improvement, and with OpenPlatypus an 8% improvement. Even powerful models further refined with RLHF such as LLaMA-2-Chat benefit from additional training with NEFTune.

{{</citation>}}


### (16/183) SALMON: Self-Alignment with Principle-Following Reward Models (Zhiqing Sun et al., 2023)

{{<citation>}}

Zhiqing Sun, Yikang Shen, Hongxin Zhang, Qinhong Zhou, Zhenfang Chen, David Cox, Yiming Yang, Chuang Gan. (2023)  
**SALMON: Self-Alignment with Principle-Following Reward Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, LLaMA, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05910v1)  

---


**ABSTRACT**  
Supervised Fine-Tuning (SFT) on response demonstrations combined with Reinforcement Learning from Human Feedback (RLHF) constitutes a powerful paradigm for aligning LLM-based AI agents. However, a significant limitation of such an approach is its dependency on high-quality human annotations, making its application to intricate tasks challenging due to difficulties in obtaining consistent response demonstrations and in-distribution response preferences. This paper presents a novel approach, namely SALMON (Self-ALignMent with principle-fOllowiNg reward models), to align base language models with minimal human supervision, using only a small set of human-defined principles, yet achieving superior performance. Central to our approach is a principle-following reward model. Trained on synthetic preference data, this model can generate reward scores based on arbitrary human-defined principles. By merely adjusting these principles during the RL training phase, we gain full control over the preferences with the reward model, subsequently influencing the behavior of the RL-trained policies, and eliminating the reliance on the collection of online human preferences. Applying our method to the LLaMA-2-70b base language model, we developed an AI assistant named Dromedary-2. With only 6 exemplars for in-context learning and 31 human-defined principles, Dromedary-2 significantly surpasses the performance of several state-of-the-art AI systems, including LLaMA-2-Chat-70b, on various benchmark datasets. We have open-sourced the code and model weights to encourage further research into aligning LLM-based AI agents with enhanced supervision efficiency, improved controllability, and scalable oversight.

{{</citation>}}


### (17/183) Rephrase, Augment, Reason: Visual Grounding of Questions for Vision-Language Models (Archiki Prasad et al., 2023)

{{<citation>}}

Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal. (2023)  
**Rephrase, Augment, Reason: Visual Grounding of Questions for Vision-Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.05861v1)  

---


**ABSTRACT**  
An increasing number of vision-language tasks can be handled with little to no training, i.e., in a zero and few-shot manner, by marrying large language models (LLMs) to vision encoders, resulting in large vision-language models (LVLMs). While this has huge upsides, such as not requiring training data or custom architectures, how an input is presented to a LVLM can have a major impact on zero-shot model performance. In particular, inputs phrased in an underspecified way can result in incorrect answers due to factors like missing visual information, complex implicit reasoning, or linguistic ambiguity. Therefore, adding visually grounded information to the input as a preemptive clarification should improve model performance by reducing underspecification, e.g., by localizing objects and disambiguating references. Similarly, in the VQA setting, changing the way questions are framed can make them easier for models to answer. To this end, we present Rephrase, Augment and Reason (RepARe), a gradient-free framework that extracts salient details about the image using the underlying LVLM as a captioner and reasoner, in order to propose modifications to the original question. We then use the LVLM's confidence over a generated answer as an unsupervised scoring function to select the rephrased question most likely to improve zero-shot performance. Focusing on two visual question answering tasks, we show that RepARe can result in a 3.85% (absolute) increase in zero-shot performance on VQAv2 and a 6.41% point increase on A-OKVQA. Additionally, we find that using gold answers for oracle question candidate selection achieves a substantial gain in VQA accuracy by up to 14.41%. Through extensive analysis, we demonstrate that outputs from RepARe increase syntactic complexity, and effectively utilize vision-language interaction and the frozen language model in LVLMs.

{{</citation>}}


### (18/183) Improving Summarization with Human Edits (Zonghai Yao et al., 2023)

{{<citation>}}

Zonghai Yao, Benjamin J Schloss, Sai P. Selvaraj. (2023)  
**Improving Summarization with Human Edits**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2310.05857v1)  

---


**ABSTRACT**  
Recent work has shown the promise of learning with human feedback paradigms to produce human-determined high-quality text. Existing works use human feedback to train large language models (LLMs) in general domain abstractive summarization and have obtained summary quality exceeding traditional likelihood training. In this paper, we focus on a less explored form of human feedback -- Human Edits. We propose Sequence Alignment (un)Likelihood Training (SALT), a novel technique to use both the human-edited and model-generated data together in the training loop. In addition, we demonstrate simulating Human Edits with ground truth summaries coming from existing training data -- Imitation edits, along with the model-generated summaries obtained after the training, to reduce the need for expensive human-edit data. In our experiments, we extend human feedback exploration from general domain summarization to medical domain summarization. Our results demonstrate the effectiveness of SALT to improve the summary quality with Human and Imitation Edits.

{{</citation>}}


### (19/183) GraphLLM: Boosting Graph Reasoning Ability of Large Language Model (Ziwei Chai et al., 2023)

{{<citation>}}

Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han, Xiaohai Hu, Xuanwen Huang, Yang Yang. (2023)  
**GraphLLM: Boosting Graph Reasoning Ability of Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.05845v1)  

---


**ABSTRACT**  
The advancement of Large Language Models (LLMs) has remarkably pushed the boundaries towards artificial general intelligence (AGI), with their exceptional ability on understanding diverse types of information, including but not limited to images and audio. Despite this progress, a critical gap remains in empowering LLMs to proficiently understand and reason on graph data. Recent studies underscore LLMs' underwhelming performance on fundamental graph reasoning tasks. In this paper, we endeavor to unearth the obstacles that impede LLMs in graph reasoning, pinpointing the common practice of converting graphs into natural language descriptions (Graph2Text) as a fundamental bottleneck. To overcome this impediment, we introduce GraphLLM, a pioneering end-to-end approach that synergistically integrates graph learning models with LLMs. This synergy equips LLMs with the ability to proficiently interpret and reason on graph data, harnessing the superior expressive power of graph learning models. Our empirical evaluations across four fundamental graph reasoning tasks validate the effectiveness of GraphLLM. The results exhibit a substantial average accuracy enhancement of 54.44%, alongside a noteworthy context reduction of 96.45% across various graph reasoning tasks.

{{</citation>}}


### (20/183) Terminology-Aware Translation with Constrained Decoding and Large Language Model Prompting (Nikolay Bogoychev et al., 2023)

{{<citation>}}

Nikolay Bogoychev, Pinzhen Chen. (2023)  
**Terminology-Aware Translation with Constrained Decoding and Large Language Model Prompting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.05824v1)  

---


**ABSTRACT**  
Terminology correctness is important in the downstream application of machine translation, and a prevalent way to ensure this is to inject terminology constraints into a translation system. In our submission to the WMT 2023 terminology translation task, we adopt a translate-then-refine approach which can be domain-independent and requires minimal manual efforts. We annotate random source words with pseudo-terminology translations obtained from word alignment to first train a terminology-aware model. Further, we explore two post-processing methods. First, we use an alignment process to discover whether a terminology constraint has been violated, and if so, we re-decode with the violating word negatively constrained. Alternatively, we leverage a large language model to refine a hypothesis by providing it with terminology constraints. Results show that our terminology-aware model learns to incorporate terminologies effectively, and the large language model refinement process can further improve terminology recall.

{{</citation>}}


### (21/183) SC-Safety: A Multi-round Open-ended Question Adversarial Safety Benchmark for Large Language Models in Chinese (Liang Xu et al., 2023)

{{<citation>}}

Liang Xu, Kangkang Zhao, Lei Zhu, Hang Xue. (2023)  
**SC-Safety: A Multi-round Open-ended Question Adversarial Safety Benchmark for Large Language Models in Chinese**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05818v1)  

---


**ABSTRACT**  
Large language models (LLMs), like ChatGPT and GPT-4, have demonstrated remarkable abilities in natural language understanding and generation. However, alongside their positive impact on our daily tasks, they can also produce harmful content that negatively affects societal perceptions. To systematically assess the safety of Chinese LLMs, we introduce SuperCLUE-Safety (SC-Safety) - a multi-round adversarial benchmark with 4912 open-ended questions covering more than 20 safety sub-dimensions. Adversarial human-model interactions and conversations significantly increase the challenges compared to existing methods. Experiments on 13 major LLMs supporting Chinese yield the following insights: 1) Closed-source models outperform open-sourced ones in terms of safety; 2) Models released from China demonstrate comparable safety levels to LLMs like GPT-3.5-turbo; 3) Some smaller models with 6B-13B parameters can compete effectively in terms of safety. By introducing SC-Safety, we aim to promote collaborative efforts to create safer and more trustworthy LLMs. The benchmark and findings provide guidance on model selection. Our benchmark can be found at https://www.CLUEbenchmarks.com

{{</citation>}}


### (22/183) Are Large Language Models Post Hoc Explainers? (Nicholas Kroeger et al., 2023)

{{<citation>}}

Nicholas Kroeger, Dan Ley, Satyapriya Krishna, Chirag Agarwal, Himabindu Lakkaraju. (2023)  
**Are Large Language Models Post Hoc Explainers?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.05797v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) are increasingly used as powerful tools for a plethora of natural language processing (NLP) applications. A recent innovation, in-context learning (ICL), enables LLMs to learn new tasks by supplying a few examples in the prompt during inference time, thereby eliminating the need for model fine-tuning. While LLMs have been utilized in several applications, their applicability in explaining the behavior of other models remains relatively unexplored. Despite the growing number of new explanation techniques, many require white-box access to the model and/or are computationally expensive, highlighting a need for next-generation post hoc explainers. In this work, we present the first framework to study the effectiveness of LLMs in explaining other predictive models. More specifically, we propose a novel framework encompassing multiple prompting strategies: i) Perturbation-based ICL, ii) Prediction-based ICL, iii) Instruction-based ICL, and iv) Explanation-based ICL, with varying levels of information about the underlying ML model and the local neighborhood of the test sample. We conduct extensive experiments with real-world benchmark datasets to demonstrate that LLM-generated explanations perform on par with state-of-the-art post hoc explainers using their ability to leverage ICL examples and their internal knowledge in generating model explanations. On average, across four datasets and two ML models, we observe that LLMs identify the most important feature with 72.19% accuracy, opening up new frontiers in explainable artificial intelligence (XAI) to explore LLM-based explanation frameworks.

{{</citation>}}


### (23/183) Problem-Solving Guide: Predicting the Algorithm Tags and Difficulty for Competitive Programming Problems (Juntae Kim et al., 2023)

{{<citation>}}

Juntae Kim, Eunjung Cho, Dongwoo Kim, Dongbin Na. (2023)  
**Problem-Solving Guide: Predicting the Algorithm Tags and Difficulty for Competitive Programming Problems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Amazon, Google  
[Paper Link](http://arxiv.org/abs/2310.05791v1)  

---


**ABSTRACT**  
The recent program development industries have required problem-solving abilities for engineers, especially application developers. However, AI-based education systems to help solve computer algorithm problems have not yet attracted attention, while most big tech companies require the ability to solve algorithm problems including Google, Meta, and Amazon. The most useful guide to solving algorithm problems might be guessing the category (tag) of the facing problems. Therefore, our study addresses the task of predicting the algorithm tag as a useful tool for engineers and developers. Moreover, we also consider predicting the difficulty levels of algorithm problems, which can be used as useful guidance to calculate the required time to solve that problem. In this paper, we present a real-world algorithm problem multi-task dataset, AMT, by mainly collecting problem samples from the most famous and large competitive programming website Codeforces. To the best of our knowledge, our proposed dataset is the most large-scale dataset for predicting algorithm tags compared to previous studies. Moreover, our work is the first to address predicting the difficulty levels of algorithm problems. We present a deep learning-based novel method for simultaneously predicting algorithm tags and the difficulty levels of an algorithm problem given. All datasets and source codes are available at https://github.com/sronger/PSG_Predicting_Algorithm_Tags_and_Difficulty.

{{</citation>}}


### (24/183) Aligning Language Models with Human Preferences via a Bayesian Approach (Jiashuo Wang et al., 2023)

{{<citation>}}

Jiashuo Wang, Haozhao Wang, Shichao Sun, Wenjie Li. (2023)  
**Aligning Language Models with Human Preferences via a Bayesian Approach**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.05782v1)  

---


**ABSTRACT**  
In the quest to advance human-centric natural language generation (NLG) systems, ensuring alignment between NLG models and human preferences is crucial. For this alignment, current popular methods leverage a reinforcement learning (RL) approach with a reward model trained on feedback from humans. However, inherent disagreements due to the subjective nature of human preferences pose a significant challenge for training the reward model, resulting in a deterioration of the NLG performance. To tackle this issue, previous approaches typically rely on majority voting or averaging to consolidate multiple inconsistent preferences into a merged one. Although straightforward to understand and execute, such methods suffer from an inability to capture the nuanced degrees of disaggregation among humans and may only represent a specialized subset of individuals, thereby lacking the ability to quantitatively disclose the universality of human preferences. To address this challenge, this paper proposes a novel approach, which employs a Bayesian framework to account for the distribution of disagreements among human preferences as training a preference model, and names it as d-PM. Besides, considering the RL strategy's inefficient and complex training process over the training efficiency, we further propose utilizing the contrastive learning strategy to train the NLG model with the preference scores derived from the d-PM model. Extensive experiments on two human-centric NLG tasks, i.e., emotional support conversation and integrity "Rule-of-Thumb" generation, show that our method consistently exceeds previous SOTA models in both automatic and human evaluations.

{{</citation>}}


### (25/183) Put Your Money Where Your Mouth Is: Evaluating Strategic Planning and Execution of LLM Agents in an Auction Arena (Jiangjie Chen et al., 2023)

{{<citation>}}

Jiangjie Chen, Siyu Yuan, Rong Ye, Bodhisattwa Prasad Majumder, Kyle Richardson. (2023)  
**Put Your Money Where Your Mouth Is: Evaluating Strategic Planning and Execution of LLM Agents in an Auction Arena**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.05746v1)  

---


**ABSTRACT**  
Can Large Language Models (LLMs) simulate human behavior in complex environments? LLMs have recently been shown to exhibit advanced reasoning skills but much of NLP evaluation still relies on static benchmarks. Answering this requires evaluation environments that probe strategic reasoning in competitive, dynamic scenarios that involve long-term planning. We introduce AucArena, a novel simulation environment for evaluating LLMs within auctions, a setting chosen for being highly unpredictable and involving many skills related to resource and risk management, while also being easy to evaluate. We conduct several controlled simulations using state-of-the-art LLMs as bidding agents. We find that through simple prompting, LLMs do indeed demonstrate many of the skills needed for effectively engaging in auctions (e.g., managing budget, adhering to long-term goals and priorities), skills that we find can be sharpened by explicitly encouraging models to be adaptive and observe strategies in past auctions. These results are significant as they show the potential of using LLM agents to model intricate social dynamics, especially in competitive settings. However, we also observe considerable variability in the capabilities of individual LLMs. Notably, even our most advanced models (GPT-4) are occasionally surpassed by heuristic baselines and human agents, highlighting the potential for further improvements in the design of LLM agents and the important role that our simulation environment can play in further testing and refining agent architectures.

{{</citation>}}


### (26/183) LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models (Huiqiang Jiang et al., 2023)

{{<citation>}}

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu. (2023)  
**LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05736v1)  

---


**ABSTRACT**  
Large language models (LLMs) have been applied in various applications due to their astonishing capabilities. With advancements in technologies such as chain-of-thought (CoT) prompting and in-context learning (ICL), the prompts fed to LLMs are becoming increasingly lengthy, even exceeding tens of thousands of tokens. To accelerate model inference and reduce cost, this paper presents LLMLingua, a coarse-to-fine prompt compression method that involves a budget controller to maintain semantic integrity under high compression ratios, a token-level iterative compression algorithm to better model the interdependence between compressed contents, and an instruction tuning based method for distribution alignment between language models. We conduct experiments and analysis over four datasets from different scenarios, i.e., GSM8K, BBH, ShareGPT, and Arxiv-March23; showing that the proposed approach yields state-of-the-art performance and allows for up to 20x compression with little performance loss. Our code is available at https://aka.ms/LLMLingua.

{{</citation>}}


### (27/183) The Program Testing Ability of Large Language Models for Code (Weimin Xiong et al., 2023)

{{<citation>}}

Weimin Xiong, Yiwen Guo, Hao Chen. (2023)  
**The Program Testing Ability of Large Language Models for Code**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-SE, cs.CL  
Keywords: GPT, GPT-3.5, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2310.05727v1)  

---


**ABSTRACT**  
Recent development of large language models (LLMs) for code like CodeX and CodeT5+ demonstrates tremendous promise in achieving code intelligence. Their ability of synthesizing code that completes a program for performing a pre-defined task has been intensively tested and verified on benchmark datasets including HumanEval and MBPP. Yet, evaluation of these LLMs from more perspectives (than just program synthesis) is also anticipated, considering their broad scope of applications in software engineering. In this paper, we explore the ability of LLMs for testing programs/code. By performing thorough analyses of recent LLMs for code in program testing, we show a series of intriguing properties of these models and demonstrate how program testing ability of LLMs can be improved. Following recent work which utilizes generated test cases to enhance program synthesis, we further leverage our findings in improving the quality of the synthesized programs and show +11.77% and +4.22% higher code pass rates on HumanEval+ comparing with the GPT-3.5-turbo baseline and the recent state-of-the-art, respectively.

{{</citation>}}


### (28/183) Guiding Language Model Reasoning with Planning Tokens (Xinyi Wang et al., 2023)

{{<citation>}}

Xinyi Wang, Lucas Caccia, Oleksiy Ostapenko, Xingdi Yuan, Alessandro Sordoni. (2023)  
**Guiding Language Model Reasoning with Planning Tokens**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.05707v1)  

---


**ABSTRACT**  
Large language models (LLMs) have recently attracted considerable interest for their ability to perform complex reasoning tasks, such as chain-of-thought reasoning. However, most of the existing approaches to enhance this ability rely heavily on data-driven methods, while neglecting the structural aspects of the model's reasoning capacity. We find that while LLMs can manage individual reasoning steps well, they struggle with maintaining consistency across an entire reasoning chain. To solve this, we introduce 'planning tokens' at the start of each reasoning step, serving as a guide for the model. These token embeddings are then fine-tuned along with the rest of the model parameters. Our approach requires a negligible increase in trainable parameters (just 0.001%) and can be applied through either full fine-tuning or a more parameter-efficient scheme. We demonstrate our method's effectiveness by applying it to three different LLMs, showing notable accuracy improvements across three math word problem datasets w.r.t. plain chain-of-thought fine-tuning baselines.

{{</citation>}}


### (29/183) A Survey of Large Language Models for Healthcare: from Data, Technology, and Applications to Accountability and Ethics (Kai He et al., 2023)

{{<citation>}}

Kai He, Rui Mao, Qika Lin, Yucheng Ruan, Xiang Lan, Mengling Feng, Erik Cambria. (2023)  
**A Survey of Large Language Models for Healthcare: from Data, Technology, and Applications to Accountability and Ethics**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2310.05694v1)  

---


**ABSTRACT**  
The utilization of large language models (LLMs) in the Healthcare domain has generated both excitement and concern due to their ability to effectively respond to freetext queries with certain professional knowledge. This survey outlines the capabilities of the currently developed LLMs for Healthcare and explicates their development process, with the aim of providing an overview of the development roadmap from traditional Pretrained Language Models (PLMs) to LLMs. Specifically, we first explore the potential of LLMs to enhance the efficiency and effectiveness of various Healthcare applications highlighting both the strengths and limitations. Secondly, we conduct a comparison between the previous PLMs and the latest LLMs, as well as comparing various LLMs with each other. Then we summarize related Healthcare training data, training methods, optimization strategies, and usage. Finally, the unique concerns associated with deploying LLMs in Healthcare settings are investigated, particularly regarding fairness, accountability, transparency and ethics. Our survey provide a comprehensive investigation from perspectives of both computer science and Healthcare specialty. Besides the discussion about Healthcare concerns, we supports the computer science community by compiling a collection of open source resources, such as accessible datasets, the latest methodologies, code implementations, and evaluation benchmarks in the Github. Summarily, we contend that a significant paradigm shift is underway, transitioning from PLMs to LLMs. This shift encompasses a move from discriminative AI approaches to generative AI approaches, as well as a shift from model-centered methodologies to datacentered methodologies.

{{</citation>}}


### (30/183) Larth: Dataset and Machine Translation for Etruscan (Gianluca Vico et al., 2023)

{{<citation>}}

Gianluca Vico, Gerasimos Spanakis. (2023)  
**Larth: Dataset and Machine Translation for Etruscan**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Machine Translation  
[Paper Link](http://arxiv.org/abs/2310.05688v1)  

---


**ABSTRACT**  
Etruscan is an ancient language spoken in Italy from the 7th century BC to the 1st century AD. There are no native speakers of the language at the present day, and its resources are scarce, as there exist only around 12,000 known inscriptions. To the best of our knowledge, there are no publicly available Etruscan corpora for natural language processing. Therefore, we propose a dataset for machine translation from Etruscan to English, which contains 2891 translated examples from existing academic sources. Some examples are extracted manually, while others are acquired in an automatic way. Along with the dataset, we benchmark different machine translation models observing that it is possible to achieve a BLEU score of 10.1 with a small transformer model. Releasing the dataset can help enable future research on this language, similar languages or other languages with scarce resources.

{{</citation>}}


### (31/183) The potential of large language models for improving probability learning: A study on ChatGPT3.5 and first-year computer engineering students (Angel Udias et al., 2023)

{{<citation>}}

Angel Udias, Antonio Alonso-Ayuso, Ignacio Sanchez, Sonia Hernandez, Maria Eugenia Castellanos, Raquel Montes Diez, Emilio Lopez Cano. (2023)  
**The potential of large language models for improving probability learning: A study on ChatGPT3.5 and first-year computer engineering students**  

---
Primary Category: cs.CL  
Categories: I-2, I2, cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.05686v1)  

---


**ABSTRACT**  
In this paper, we assess the efficacy of ChatGPT (version Feb 2023), a large-scale language model, in solving probability problems typically presented in introductory computer engineering exams. Our study comprised a set of 23 probability exercises administered to students at Rey Juan Carlos University (URJC) in Madrid. The responses produced by ChatGPT were evaluated by a group of five statistics professors, who assessed them qualitatively and assigned grades based on the same criteria used for students. Our results indicate that ChatGPT surpasses the average student in terms of phrasing, organization, and logical reasoning. The model's performance remained consistent for both the Spanish and English versions of the exercises. However, ChatGPT encountered difficulties in executing basic numerical operations. Our experiments demonstrate that requesting ChatGPT to provide the solution in the form of an R script proved to be an effective approach for overcoming these limitations. In summary, our results indicate that ChatGPT surpasses the average student in solving probability problems commonly presented in introductory computer engineering exams. Nonetheless, the model exhibits limitations in reasoning around certain probability concepts. The model's ability to deliver high-quality explanations and illustrate solutions in any programming language, coupled with its performance in solving probability exercises, suggests that large language models have the potential to serve as learning assistants.

{{</citation>}}


### (32/183) A Closer Look into Automatic Evaluation Using Large Language Models (Cheng-Han Chiang et al., 2023)

{{<citation>}}

Cheng-Han Chiang, Hung-yi Lee. (2023)  
**A Closer Look into Automatic Evaluation Using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05657v1)  

---


**ABSTRACT**  
Using large language models (LLMs) to evaluate text quality has recently gained popularity. Some prior works explore the idea of using LLMs for evaluation, while they differ in some details of the evaluation process. In this paper, we analyze LLM evaluation (Chiang and Lee, 2023) and G-Eval (Liu et al., 2023), and we discuss how those details in the evaluation process change how well the ratings given by LLMs correlate with human ratings. We find that the auto Chain-of-Thought (CoT) used in G-Eval does not always make G-Eval more aligned with human ratings. We also show that forcing the LLM to output only a numeric rating, as in G-Eval, is suboptimal. Last, we reveal that asking the LLM to explain its own ratings consistently improves the correlation between the ChatGPT and human ratings and pushes state-of-the-art (SoTA) correlations on two meta-evaluation datasets.

{{</citation>}}


### (33/183) RAUCG: Retrieval-Augmented Unsupervised Counter Narrative Generation for Hate Speech (Shuyu Jiang et al., 2023)

{{<citation>}}

Shuyu Jiang, Wenyi Tang, Xingshu Chen, Rui Tanga, Haizhou Wang, Wenxian Wang. (2023)  
**RAUCG: Retrieval-Augmented Unsupervised Counter Narrative Generation for Hate Speech**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.05650v1)  

---


**ABSTRACT**  
The Counter Narrative (CN) is a promising approach to combat online hate speech (HS) without infringing on freedom of speech. In recent years, there has been a growing interest in automatically generating CNs using natural language generation techniques. However, current automatic CN generation methods mainly rely on expert-authored datasets for training, which are time-consuming and labor-intensive to acquire. Furthermore, these methods cannot directly obtain and extend counter-knowledge from external statistics, facts, or examples. To address these limitations, we propose Retrieval-Augmented Unsupervised Counter Narrative Generation (RAUCG) to automatically expand external counter-knowledge and map it into CNs in an unsupervised paradigm. Specifically, we first introduce an SSF retrieval method to retrieve counter-knowledge from the multiple perspectives of stance consistency, semantic overlap rate, and fitness for HS. Then we design an energy-based decoding mechanism by quantizing knowledge injection, countering and fluency constraints into differentiable functions, to enable the model to build mappings from counter-knowledge to CNs without expert-authored CN data. Lastly, we comprehensively evaluate model performance in terms of language quality, toxicity, persuasiveness, relevance, and success rate of countering HS, etc. Experimental results show that RAUCG outperforms strong baselines on all metrics and exhibits stronger generalization capabilities, achieving significant improvements of +2.0% in relevance and +4.5% in success rate of countering metrics. Moreover, RAUCG enabled GPT2 to outperform T0 in all metrics, despite the latter being approximately eight times larger than the former. Warning: This paper may contain offensive or upsetting content!

{{</citation>}}


### (34/183) Towards Verifiable Generation: A Benchmark for Knowledge-aware Language Model Attribution (Xinze Li et al., 2023)

{{<citation>}}

Xinze Li, Yixin Cao2, Liangming Pan, Yubo Ma, Aixin Sun. (2023)  
**Towards Verifiable Generation: A Benchmark for Knowledge-aware Language Model Attribution**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05634v1)  

---


**ABSTRACT**  
Although achieving great success, Large Language Models (LLMs) usually suffer from unreliable hallucinations. In this paper, we define a new task of Knowledge-aware Language Model Attribution (KaLMA) that improves upon three core concerns on conventional attributed LMs. First, we extend attribution source from unstructured texts to Knowledge Graph (KG), whose rich structures benefit both the attribution performance and working scenarios. Second, we propose a new ``Conscious Incompetence" setting considering the incomplete knowledge repository, where the model identifies the need for supporting knowledge beyond the provided KG. Third, we propose a comprehensive automatic evaluation metric encompassing text quality, citation quality, and text citation alignment. To implement the above innovations, we build a dataset in biography domain BioKaLMA via a well-designed evolutionary question generation strategy, to control the question complexity and necessary knowledge to the answer. For evaluation, we develop a baseline solution and demonstrate the room for improvement in LLMs' citation generation, emphasizing the importance of incorporating the "Conscious Incompetence" setting, and the critical role of retrieval accuracy.

{{</citation>}}


### (35/183) Glitter or Gold? Deriving Structured Insights from Sustainability Reports via Large Language Models (Marco Bronzini et al., 2023)

{{<citation>}}

Marco Bronzini, Carlo Nicolini, Bruno Lepri, Andrea Passerini, Jacopo Staiano. (2023)  
**Glitter or Gold? Deriving Structured Insights from Sustainability Reports via Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CE, cs-CL, cs-CY, cs.CL  
Keywords: Information Extraction, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05628v1)  

---


**ABSTRACT**  
Over the last decade, several regulatory bodies have started requiring the disclosure of non-financial information from publicly listed companies, in light of the investors' increasing attention to Environmental, Social, and Governance (ESG) issues. Such information is publicly released in a variety of non-structured and multi-modal documentation. Hence, it is not straightforward to aggregate and consolidate such data in a cohesive framework to further derive insights about sustainability practices across companies and markets. Thus, it is natural to resort to Information Extraction (IE) techniques to provide concise, informative and actionable data to the stakeholders. Moving beyond traditional text processing techniques, in this work we leverage Large Language Models (LLMs), along with prominent approaches such as Retrieved Augmented Generation and in-context learning, to extract semantically structured information from sustainability reports. We then adopt graph-based representations to generate meaningful statistical, similarity and correlation analyses concerning the obtained findings, highlighting the prominent sustainability actions undertaken across industries and discussing emerging similarity and disclosing patterns at company, sector and region levels. Lastly, we investigate which factual aspects impact the most on companies' ESG scores using our findings and other company information.

{{</citation>}}


### (36/183) Integrating Stock Features and Global Information via Large Language Models for Enhanced Stock Return Prediction (Yujie Ding et al., 2023)

{{<citation>}}

Yujie Ding, Shuai Jia, Tianyi Ma, Bingcheng Mao, Xiuze Zhou, Liuliu Li, Dongming Han. (2023)  
**Integrating Stock Features and Global Information via Large Language Models for Enhanced Stock Return Prediction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL, q-fin-ST  
Keywords: ChatGPT, GPT, GPT-4, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05627v1)  

---


**ABSTRACT**  
The remarkable achievements and rapid advancements of Large Language Models (LLMs) such as ChatGPT and GPT-4 have showcased their immense potential in quantitative investment. Traders can effectively leverage these LLMs to analyze financial news and predict stock returns accurately. However, integrating LLMs into existing quantitative models presents two primary challenges: the insufficient utilization of semantic information embedded within LLMs and the difficulties in aligning the latent information within LLMs with pre-existing quantitative stock features. We propose a novel framework consisting of two components to surmount these challenges. The first component, the Local-Global (LG) model, introduces three distinct strategies for modeling global information. These approaches are grounded respectively on stock features, the capabilities of LLMs, and a hybrid method combining the two paradigms. The second component, Self-Correlated Reinforcement Learning (SCRL), focuses on aligning the embeddings of financial news generated by LLMs with stock features within the same semantic space. By implementing our framework, we have demonstrated superior performance in Rank Information Coefficient and returns, particularly compared to models relying only on stock features in the China A-share market.

{{</citation>}}


### (37/183) LAiW: A Chinese Legal Large Language Models Benchmark (A Technical Report) (Yongfu Dai et al., 2023)

{{<citation>}}

Yongfu Dai, Duanyu Feng, Jimin Huang, Haochen Jia, Qianqian Xie, Yifang Zhang, Weiguang Han, Wei Tian, Hao Wang. (2023)  
**LAiW: A Chinese Legal Large Language Models Benchmark (A Technical Report)**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Legal, NLP  
[Paper Link](http://arxiv.org/abs/2310.05620v1)  

---


**ABSTRACT**  
With the emergence of numerous legal LLMs, there is currently a lack of a comprehensive benchmark for evaluating their legal abilities. In this paper, we propose the first Chinese Legal LLMs benchmark based on legal capabilities. Through the collaborative efforts of legal and artificial intelligence experts, we divide the legal capabilities of LLMs into three levels: basic legal NLP capability, basic legal application capability, and complex legal application capability. We have completed the first phase of evaluation, which mainly focuses on the capability of basic legal NLP. The evaluation results show that although some legal LLMs have better performance than their backbones, there is still a gap compared to ChatGPT. Our benchmark can be found at URL.

{{</citation>}}


### (38/183) Dynamic Top-k Estimation Consolidates Disagreement between Feature Attribution Methods (Jonathan Kamp et al., 2023)

{{<citation>}}

Jonathan Kamp, Lisa Beinborn, Antske Fokkens. (2023)  
**Dynamic Top-k Estimation Consolidates Disagreement between Feature Attribution Methods**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLI  
[Paper Link](http://arxiv.org/abs/2310.05619v1)  

---


**ABSTRACT**  
Feature attribution scores are used for explaining the prediction of a text classifier to users by highlighting a k number of tokens. In this work, we propose a way to determine the number of optimal k tokens that should be displayed from sequential properties of the attribution scores. Our approach is dynamic across sentences, method-agnostic, and deals with sentence length bias. We compare agreement between multiple methods and humans on an NLI task, using fixed k and dynamic k. We find that perturbation-based methods and Vanilla Gradient exhibit highest agreement on most method--method and method--human agreement metrics with a static k. Their advantage over other methods disappears with dynamic ks which mainly improve Integrated Gradient and GradientXInput. To our knowledge, this is the first evidence that sequential properties of attribution scores are informative for consolidating attribution signals for human interpretation.

{{</citation>}}


### (39/183) Can language models learn analogical reasoning? Investigating training objectives and comparisons to human performance (Molly R. Petersen et al., 2023)

{{<citation>}}

Molly R. Petersen, Lonneke van der Plas. (2023)  
**Can language models learn analogical reasoning? Investigating training objectives and comparisons to human performance**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.05597v1)  

---


**ABSTRACT**  
While analogies are a common way to evaluate word embeddings in NLP, it is also of interest to investigate whether or not analogical reasoning is a task in itself that can be learned. In this paper, we test several ways to learn basic analogical reasoning, specifically focusing on analogies that are more typical of what is used to evaluate analogical reasoning in humans than those in commonly used NLP benchmarks. Our experiments find that models are able to learn analogical reasoning, even with a small amount of data. We additionally compare our models to a dataset with a human baseline, and find that after training, models approach human performance.

{{</citation>}}


### (40/183) InterroLang: Exploring NLP Models and Datasets through Dialogue-based Explanations (Nils Feldhus et al., 2023)

{{<citation>}}

Nils Feldhus, Qianli Wang, Tatiana Anikina, Sahil Chopra, Cennet Oguz, Sebastian Möller. (2023)  
**InterroLang: Exploring NLP Models and Datasets through Dialogue-based Explanations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: Dialog, Dialogue, NLP  
[Paper Link](http://arxiv.org/abs/2310.05592v1)  

---


**ABSTRACT**  
While recently developed NLP explainability methods let us open the black box in various ways (Madsen et al., 2022), a missing ingredient in this endeavor is an interactive tool offering a conversational interface. Such a dialogue system can help users explore datasets and models with explanations in a contextualized manner, e.g. via clarification or follow-up questions, and through a natural language interface. We adapt the conversational explanation framework TalkToModel (Slack et al., 2022) to the NLP domain, add new NLP-specific operations such as free-text rationalization, and illustrate its generalizability on three NLP tasks (dialogue act classification, question answering, hate speech detection). To recognize user queries for explanations, we evaluate fine-tuned and few-shot prompting models and implement a novel Adapter-based approach. We then conduct two user studies on (1) the perceived correctness and helpfulness of the dialogues, and (2) the simulatability, i.e. how objectively helpful dialogical explanations are for humans in figuring out the model's predicted label when it's not shown. We found rationalization and feature attribution were helpful in explaining the model behavior. Moreover, users could more reliably predict the model outcome based on an explanation dialogue rather than one-off explanations.

{{</citation>}}


### (41/183) DRIN: Dynamic Relation Interactive Network for Multimodal Entity Linking (Shangyu Xing et al., 2023)

{{<citation>}}

Shangyu Xing, Fei Zhao, Zhen Wu, Chunhui Li, Jianbing Zhang, Xinyu Dai. (2023)  
**DRIN: Dynamic Relation Interactive Network for Multimodal Entity Linking**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-MM, cs.CL  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2310.05589v1)  

---


**ABSTRACT**  
Multimodal Entity Linking (MEL) is a task that aims to link ambiguous mentions within multimodal contexts to referential entities in a multimodal knowledge base. Recent methods for MEL adopt a common framework: they first interact and fuse the text and image to obtain representations of the mention and entity respectively, and then compute the similarity between them to predict the correct entity. However, these methods still suffer from two limitations: first, as they fuse the features of text and image before matching, they cannot fully exploit the fine-grained alignment relations between the mention and entity. Second, their alignment is static, leading to low performance when dealing with complex and diverse data. To address these issues, we propose a novel framework called Dynamic Relation Interactive Network (DRIN) for MEL tasks. DRIN explicitly models four different types of alignment between a mention and entity and builds a dynamic Graph Convolutional Network (GCN) to dynamically select the corresponding alignment relations for different input samples. Experiments on two datasets show that DRIN outperforms state-of-the-art methods by a large margin, demonstrating the effectiveness of our approach.

{{</citation>}}


### (42/183) Regulation and NLP (RegNLP): Taming Large Language Models (Catalina Goanta et al., 2023)

{{<citation>}}

Catalina Goanta, Nikolaos Aletras, Ilias Chalkidis, Sofia Ranchordas, Gerasimos Spanakis. (2023)  
**Regulation and NLP (RegNLP): Taming Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.05553v1)  

---


**ABSTRACT**  
The scientific innovation in Natural Language Processing (NLP) and more broadly in artificial intelligence (AI) is at its fastest pace to date. As large language models (LLMs) unleash a new era of automation, important debates emerge regarding the benefits and risks of their development, deployment and use. Currently, these debates have been dominated by often polarized narratives mainly led by the AI Safety and AI Ethics movements. This polarization, often amplified by social media, is swaying political agendas on AI regulation and governance and posing issues of regulatory capture. Capture occurs when the regulator advances the interests of the industry it is supposed to regulate, or of special interest groups rather than pursuing the general public interest. Meanwhile in NLP research, attention has been increasingly paid to the discussion of regulating risks and harms. This often happens without systematic methodologies or sufficient rooting in the disciplines that inspire an extended scope of NLP research, jeopardizing the scientific integrity of these endeavors. Regulation studies are a rich source of knowledge on how to systematically deal with risk and uncertainty, as well as with scientific evidence, to evaluate and compare regulatory options. This resource has largely remained untapped so far. In this paper, we argue how NLP research on these topics can benefit from proximity to regulatory studies and adjacent fields. We do so by discussing basic tenets of regulation, and risk and uncertainty, and by highlighting the shortcomings of current NLP discussions dealing with risk assessment. Finally, we advocate for the development of a new multidisciplinary research space on regulation and NLP (RegNLP), focused on connecting scientific knowledge to regulatory processes based on systematic methodologies.

{{</citation>}}


### (43/183) Query and Response Augmentation Cannot Help Out-of-domain Math Reasoning Generalization (Chengpeng Li et al., 2023)

{{<citation>}}

Chengpeng Li, Zheng Yuan, Guanting Dong, Keming Lu, Jiancan Wu, Chuanqi Tan, Xiang Wang, Chang Zhou. (2023)  
**Query and Response Augmentation Cannot Help Out-of-domain Math Reasoning Generalization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Augmentation, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.05506v1)  

---


**ABSTRACT**  
In math reasoning with large language models (LLMs), fine-tuning data augmentation by query evolution and diverse reasoning paths is empirically verified effective, profoundly narrowing the gap between open-sourced LLMs and cutting-edge proprietary LLMs. In this paper, we conduct an investigation for such data augmentation in math reasoning and are intended to answer: (1) What strategies of data augmentation are more effective; (2) What is the scaling relationship between the amount of augmented data and model performance; and (3) Can data augmentation incentivize generalization to out-of-domain mathematical reasoning tasks? To this end, we create a new dataset, AugGSM8K, by complicating and diversifying the queries from GSM8K and sampling multiple reasoning paths. We obtained a series of LLMs called MuggleMath by fine-tuning on subsets of AugGSM8K. MuggleMath substantially achieves new state-of-the-art on GSM8K (from 54% to 68.4% at the scale of 7B, and from 63.9% to 74.0% at the scale of 13B). A log-linear relationship is presented between MuggleMath's performance and the amount of augmented data. We also find that MuggleMath is weak in out-of-domain math reasoning generalization to MATH. This is attributed to the differences in query distribution between AugGSM8K and MATH which suggest that augmentation on a single benchmark could not help with overall math reasoning performance. Codes and AugGSM8K will be uploaded to https://github.com/OFA-Sys/gsm8k-ScRel.

{{</citation>}}


### (44/183) XAL: EXplainable Active Learning Makes Classifiers Better Low-resource Learners (Yun Luo et al., 2023)

{{<citation>}}

Yun Luo, Zhen Yang, Fandong Meng, Yingjie Li, Fang Guo, Qinglin Qi, Jie Zhou, Yue Zhang. (2023)  
**XAL: EXplainable Active Learning Makes Classifiers Better Low-resource Learners**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2310.05502v1)  

---


**ABSTRACT**  
Active learning aims to construct an effective training set by iteratively curating the most informative unlabeled data for annotation, which is practical in low-resource tasks. Most active learning techniques in classification rely on the model's uncertainty or disagreement to choose unlabeled data. However, previous work indicates that existing models are poor at quantifying predictive uncertainty, which can lead to over-confidence in superficial patterns and a lack of exploration. Inspired by the cognitive processes in which humans deduce and predict through causal information, we propose a novel Explainable Active Learning framework (XAL) for low-resource text classification, which aims to encourage classifiers to justify their inferences and delve into unlabeled data for which they cannot provide reasonable explanations. Specifically, besides using a pre-trained bi-directional encoder for classification, we employ a pre-trained uni-directional decoder to generate and score the explanation. A ranking loss is proposed to enhance the decoder's capability in scoring explanations. During the selection of unlabeled data, we combine the predictive uncertainty of the encoder and the explanation score of the decoder to acquire informative data for annotation.   As XAL is a general framework for text classification, we test our methods on six different classification tasks. Extensive experiments show that XAL achieves substantial improvement on all six tasks over previous AL methods. Ablation studies demonstrate the effectiveness of each component, and human evaluation shows that the model trained in XAL performs surprisingly well in explaining its prediction.

{{</citation>}}


### (45/183) How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition (Guanting Dong et al., 2023)

{{<citation>}}

Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, Wei Wang, Zheng Yuan, Chang Zhou, Jingren Zhou. (2023)  
**How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.05492v1)  

---


**ABSTRACT**  
Large language models (LLMs) with enormous pre-training tokens and parameter amounts emerge abilities, including math reasoning, code generation, and instruction following. These abilities are further enhanced by supervised fine-tuning (SFT). The open-source community has studied on ad-hoc SFT for each ability, while proprietary LLMs are versatile for all abilities. It is important to investigate how to unlock them with multiple abilities via SFT. In this study, we specifically focus on the data composition between mathematical reasoning, code generation, and general human-aligning abilities during SFT. From a scaling perspective, we investigate the relationship between model abilities and various factors including data amounts, data composition ratio, model parameters, and SFT strategies. Our experiments reveal that different abilities exhibit different scaling patterns, and larger models generally show superior performance with the same amount of data. Mathematical reasoning and code generation improve as data amounts increase consistently, while the general ability is enhanced with about a thousand samples and improves slowly. We find data composition results in various abilities improvements with low data amounts, while conflicts of abilities with high data amounts. Our experiments further show that composition data amount impacts performance, while the influence of composition ratio is insignificant. Regarding the SFT strategies, we evaluate sequential learning multiple abilities are prone to catastrophic forgetting. Our proposed Dual-stage Mixed Fine-tuning (DMT) strategy learns specialized abilities first and then learns general abilities with a small amount of specialized data to prevent forgetting, offering a promising solution to learn multiple abilities with different scaling patterns.

{{</citation>}}


### (46/183) Cabbage Sweeter than Cake? Analysing the Potential of Large Language Models for Learning Conceptual Spaces (Usashi Chatterjee et al., 2023)

{{<citation>}}

Usashi Chatterjee, Amit Gajbhiye, Steven Schockaert. (2023)  
**Cabbage Sweeter than Cake? Analysing the Potential of Large Language Models for Learning Conceptual Spaces**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05481v1)  

---


**ABSTRACT**  
The theory of Conceptual Spaces is an influential cognitive-linguistic framework for representing the meaning of concepts. Conceptual spaces are constructed from a set of quality dimensions, which essentially correspond to primitive perceptual features (e.g. hue or size). These quality dimensions are usually learned from human judgements, which means that applications of conceptual spaces tend to be limited to narrow domains (e.g. modelling colour or taste). Encouraged by recent findings about the ability of Large Language Models (LLMs) to learn perceptually grounded representations, we explore the potential of such models for learning conceptual spaces. Our experiments show that LLMs can indeed be used for learning meaningful representations to some extent. However, we also find that fine-tuned models of the BERT family are able to match or even outperform the largest GPT-3 model, despite being 2 to 3 orders of magnitude smaller.

{{</citation>}}


### (47/183) Generative Judge for Evaluating Alignment (Junlong Li et al., 2023)

{{<citation>}}

Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Hai Zhao, Pengfei Liu. (2023)  
**Generative Judge for Evaluating Alignment**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.05470v1)  

---


**ABSTRACT**  
The rapid development of Large Language Models (LLMs) has substantially expanded the range of tasks they can address. In the field of Natural Language Processing (NLP), researchers have shifted their focus from conventional NLP tasks (e.g., sequence tagging and parsing) towards tasks that revolve around aligning with human needs (e.g., brainstorming and email writing). This shift in task distribution imposes new requirements on evaluating these aligned models regarding generality (i.e., assessing performance across diverse scenarios), flexibility (i.e., examining under different protocols), and interpretability (i.e., scrutinizing models with explanations). In this paper, we propose a generative judge with 13B parameters, Auto-J, designed to address these challenges. Our model is trained on user queries and LLM-generated responses under massive real-world scenarios and accommodates diverse evaluation protocols (e.g., pairwise response comparison and single-response evaluation) with well-structured natural language critiques. To demonstrate the efficacy of our approach, we construct a new testbed covering 58 different scenarios. Experimentally, Auto-J outperforms a series of strong competitors, including both open-source and closed-source models, by a large margin. We also provide detailed analysis and case studies to further reveal the potential of our method and make a variety of resources public at https://github.com/GAIR-NLP/auto-j.

{{</citation>}}


### (48/183) Empower Nested Boolean Logic via Self-Supervised Curriculum Learning (Hongqiu Wu et al., 2023)

{{<citation>}}

Hongqiu Wu, Linfeng Liu, Hai Zhao, Min Zhang. (2023)  
**Empower Nested Boolean Logic via Self-Supervised Curriculum Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reasoning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.05450v1)  

---


**ABSTRACT**  
Beyond the great cognitive powers showcased by language models, it is crucial to scrutinize whether their reasoning capabilities stem from strong generalization or merely exposure to relevant data. As opposed to constructing increasingly complex logic, this paper probes into the boolean logic, the root capability of a logical reasoner. We find that any pre-trained language models even including large language models only behave like a random selector in the face of multi-nested boolean logic, a task that humans can handle with ease. To empower language models with this fundamental capability, this paper proposes a new self-supervised learning method \textit{Curriculum Logical Reasoning} (\textsc{Clr}), where we augment the training data with nested boolean logic chain step-by-step, and program the training from simpler logical patterns gradually to harder ones. This new training paradigm allows language models to effectively generalize to much harder and longer-hop logic, which can hardly be learned through naive training. Furthermore, we show that boolean logic is a great foundation for improving the subsequent general logical tasks.

{{</citation>}}


### (49/183) Establishing Trustworthiness: Rethinking Tasks and Model Evaluation (Robert Litschko et al., 2023)

{{<citation>}}

Robert Litschko, Max Müller-Eberstein, Rob van der Goot, Leon Weber, Barbara Plank. (2023)  
**Establishing Trustworthiness: Rethinking Tasks and Model Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.05442v1)  

---


**ABSTRACT**  
Language understanding is a multi-faceted cognitive capability, which the Natural Language Processing (NLP) community has striven to model computationally for decades. Traditionally, facets of linguistic intelligence have been compartmentalized into tasks with specialized model architectures and corresponding evaluation protocols. With the advent of large language models (LLMs) the community has witnessed a dramatic shift towards general purpose, task-agnostic approaches powered by generative models. As a consequence, the traditional compartmentalized notion of language tasks is breaking down, followed by an increasing challenge for evaluation and analysis. At the same time, LLMs are being deployed in more real-world scenarios, including previously unforeseen zero-shot setups, increasing the need for trustworthy and reliable systems. Therefore, we argue that it is time to rethink what constitutes tasks and model evaluation in NLP, and pursue a more holistic view on language, placing trustworthiness at the center. Towards this goal, we review existing compartmentalized approaches for understanding the origins of a model's functional capacity, and provide recommendations for more multi-faceted evaluation protocols.

{{</citation>}}


### (50/183) Fast and Robust Early-Exiting Framework for Autoregressive Language Models with Synchronized Parallel Decoding (Sangmin Bae et al., 2023)

{{<citation>}}

Sangmin Bae, Jongwoo Ko, Hwanjun Song, Se-Young Yun. (2023)  
**Fast and Robust Early-Exiting Framework for Autoregressive Language Models with Synchronized Parallel Decoding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.05424v1)  

---


**ABSTRACT**  
To tackle the high inference latency exhibited by autoregressive language models, previous studies have proposed an early-exiting framework that allocates adaptive computation paths for each token based on the complexity of generating the subsequent token. However, we observed several shortcomings, including performance degradation caused by a state copying mechanism or numerous exit paths, and sensitivity to exit confidence thresholds. Consequently, we propose a Fast and Robust Early-Exiting (FREE) framework, which incorporates a shallow-deep module and a synchronized parallel decoding. Our framework enables faster inference by synchronizing the decoding process of the current token with previously stacked early-exited tokens. Furthermore, as parallel decoding allows us to observe predictions from both shallow and deep models, we present a novel adaptive threshold estimator that exploits a Beta mixture model to determine suitable confidence thresholds. We empirically demonstrated the superiority of our proposed framework on extensive generation tasks.

{{</citation>}}


### (51/183) Automating Customer Service using LangChain: Building custom open-source GPT Chatbot for organizations (Keivalya Pandya et al., 2023)

{{<citation>}}

Keivalya Pandya, Mehfuza Holia. (2023)  
**Automating Customer Service using LangChain: Building custom open-source GPT Chatbot for organizations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: GPT, Google, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2310.05421v1)  

---


**ABSTRACT**  
In the digital age, the dynamics of customer service are evolving, driven by technological advancements and the integration of Large Language Models (LLMs). This research paper introduces a groundbreaking approach to automating customer service using LangChain, a custom LLM tailored for organizations. The paper explores the obsolescence of traditional customer support techniques, particularly Frequently Asked Questions (FAQs), and proposes a paradigm shift towards responsive, context-aware, and personalized customer interactions. The heart of this innovation lies in the fusion of open-source methodologies, web scraping, fine-tuning, and the seamless integration of LangChain into customer service platforms. This open-source state-of-the-art framework, presented as "Sahaay," demonstrates the ability to scale across industries and organizations, offering real-time support and query resolution. Key elements of this research encompass data collection via web scraping, the role of embeddings, the utilization of Google's Flan T5 XXL, Base and Small language models for knowledge retrieval, and the integration of the chatbot into customer service platforms. The results section provides insights into their performance and use cases, here particularly within an educational institution. This research heralds a new era in customer service, where technology is harnessed to create efficient, personalized, and responsive interactions. Sahaay, powered by LangChain, redefines the customer-company relationship, elevating customer retention, value extraction, and brand image. As organizations embrace LLMs, customer service becomes a dynamic and customer-centric ecosystem.

{{</citation>}}


### (52/183) mBBC: Exploring the Multilingual Maze (Sina Bagheri Nezhad et al., 2023)

{{<citation>}}

Sina Bagheri Nezhad, Ameeta Agrawal. (2023)  
**mBBC: Exploring the Multilingual Maze**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: BERT, GPT, Multilingual  
[Paper Link](http://arxiv.org/abs/2310.05404v1)  

---


**ABSTRACT**  
Multilingual language models have gained significant attention in recent years, enabling the development of applications that cater to diverse linguistic contexts. In this paper, we present a comprehensive evaluation of three prominent multilingual language models: mBERT, XLM-R, and GPT-3. Using the self-supervised task of next token prediction, we assess their performance across a diverse set of languages, with a focus on understanding the impact of resource availability, word order, language family, and script type on model accuracy. Our findings reveal that resource availability plays a crucial role in model performance, with higher resource levels leading to improved accuracy. We also identify the complex relationship between resource availability, language families, and script types, highlighting the need for further investigation into language-specific characteristics and structural variations. Additionally, our statistical inference analysis identifies significant features contributing to model performance, providing insights for model selection and deployment. Our study contributes to a deeper understanding of multilingual language models and informs future research and development to enhance their performance and generalizability across languages and linguistic contexts.

{{</citation>}}


### (53/183) GROVE: A Retrieval-augmented Complex Story Generation Framework with A Forest of Evidence (Zhihua Wen et al., 2023)

{{<citation>}}

Zhihua Wen, Zhiliang Tian, Wei Wu, Yuxin Yang, Yanqi Shi, Zhen Huang, Dongsheng Li. (2023)  
**GROVE: A Retrieval-augmented Complex Story Generation Framework with A Forest of Evidence**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.05388v1)  

---


**ABSTRACT**  
Conditional story generation is significant in human-machine interaction, particularly in producing stories with complex plots. While Large language models (LLMs) perform well on multiple NLP tasks, including story generation, it is challenging to generate stories with both complex and creative plots. Existing methods often rely on detailed prompts to guide LLMs to meet target conditions, which inadvertently restrict the creative potential of the generated stories. We argue that leveraging information from exemplary human-written stories facilitates generating more diverse plotlines. Delving deeper into story details helps build complex and credible plots. In this paper, we propose a retrieval-au\textbf{G}mented sto\textbf{R}y generation framework with a f\textbf{O}rest of e\textbf{V}id\textbf{E}nce (GROVE) to enhance stories' complexity. We build a retrieval repository for target conditions to produce few-shot examples to prompt LLMs. Additionally, we design an ``asking-why'' prompting scheme that extracts a forest of evidence, providing compensation for the ambiguities that may occur in the generated story. This iterative process uncovers underlying story backgrounds. Finally, we select the most fitting chains of evidence from the evidence forest and integrate them into the generated story, thereby enhancing the narrative's complexity and credibility. Experimental results and numerous examples verify the effectiveness of our method.

{{</citation>}}


### (54/183) CCAE: A Corpus of Chinese-based Asian Englishes (Yang Liu et al., 2023)

{{<citation>}}

Yang Liu, Melissa Xiaohui Qin, Long Wang, Chao Huang. (2023)  
**CCAE: A Corpus of Chinese-based Asian Englishes**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2-7, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.05381v1)  

---


**ABSTRACT**  
Language models have been foundations in various scenarios of NLP applications, but it has not been well applied in language variety studies, even for the most popular language like English. This paper represents one of the few initial efforts to utilize the NLP technology in the paradigm of World Englishes, specifically in creating a multi-variety corpus for studying Asian Englishes. We present an overview of the CCAE -- Corpus of Chinese-based Asian English, a suite of corpora comprising six Chinese-based Asian English varieties. It is based on 340 million tokens in 448 thousand web documents from six regions. The ontology of data would make the corpus a helpful resource with enormous research potential for Asian Englishes (especially for Chinese Englishes for which there has not been a publicly accessible corpus yet so far) and an ideal source for variety-specific language modeling and downstream tasks, thus setting the stage for NLP-based World Englishes studies. And preliminary experiments on this corpus reveal the practical value of CCAE. Finally, we make CCAE available at \href{https://huggingface.co/datasets/CCAE/CCAE-Corpus}{this https URL}.

{{</citation>}}


### (55/183) Transcending the Attention Paradigm: Implicit Learning from Geospatial Social Media Data (Nick DiSanto et al., 2023)

{{<citation>}}

Nick DiSanto, Anthony Corso, Benjamin Sanders, Gavin Harding. (2023)  
**Transcending the Attention Paradigm: Implicit Learning from Geospatial Social Media Data**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: Attention, Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2310.05378v1)  

---


**ABSTRACT**  
While transformers have pioneered attention-driven architectures as a cornerstone of research, their dependence on explicitly contextual information underscores limitations in their abilities to tacitly learn overarching textual themes. This study investigates social media data as a source of distributed patterns, challenging the heuristic paradigm of performance benchmarking. In stark contrast to networks that rely on capturing complex long-term dependencies, models of online data inherently lack structure and are forced to learn underlying patterns in the aggregate. To properly represent these abstract relationships, this research dissects empirical social media corpora into their elemental components and analyzes over two billion tweets across population-dense locations. Exploring the relationship between location and vernacular in Twitter data, we employ Bag-of-Words models specific to each city and evaluate their respective representation. This demonstrates that hidden insights can be uncovered without the crutch of advanced algorithms and demonstrates that even amidst noisy data, geographic location has a considerable influence on online communication. This evidence presents tangible insights regarding geospatial communication patterns and their implications in social science. It also challenges the notion that intricate models are prerequisites for pattern recognition in natural language, aligning with the evolving landscape that questions the embrace of absolute interpretability over abstract understanding. This study bridges the divide between sophisticated frameworks and intangible relationships, paving the way for systems that blend structured models with conjectural reasoning.

{{</citation>}}


### (56/183) Universal Multi-modal Entity Alignment via Iteratively Fusing Modality Similarity Paths (Bolin Zhu et al., 2023)

{{<citation>}}

Bolin Zhu, Xiaoze Liu, Xin Mao, Zhuo Chen, Lingbing Guo, Tao Gui, Qi Zhang. (2023)  
**Universal Multi-modal Entity Alignment via Iteratively Fusing Modality Similarity Paths**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Entity Alignment, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.05364v2)  

---


**ABSTRACT**  
The objective of Entity Alignment (EA) is to identify equivalent entity pairs from multiple Knowledge Graphs (KGs) and create a more comprehensive and unified KG. The majority of EA methods have primarily focused on the structural modality of KGs, lacking exploration of multi-modal information. A few multi-modal EA methods have made good attempts in this field. Still, they have two shortcomings: (1) inconsistent and inefficient modality modeling that designs complex and distinct models for each modality; (2) ineffective modality fusion due to the heterogeneous nature of modalities in EA. To tackle these challenges, we propose PathFusion, consisting of two main components: (1) MSP, a unified modeling approach that simplifies the alignment process by constructing paths connecting entities and modality nodes to represent multiple modalities; (2) IRF, an iterative fusion method that effectively combines information from different modalities using the path as an information carrier. Experimental results on real-world datasets demonstrate the superiority of PathFusion over state-of-the-art methods, with 22.4%-28.9% absolute improvement on Hits@1, and 0.194-0.245 absolute improvement on MRR.

{{</citation>}}


### (57/183) A Glance is Enough: Extract Target Sentence By Looking at A keyword (Ying Shi et al., 2023)

{{<citation>}}

Ying Shi, Dong Wang, Lantian Li, Jiqing Han. (2023)  
**A Glance is Enough: Extract Target Sentence By Looking at A keyword**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.05352v1)  

---


**ABSTRACT**  
This paper investigates the possibility of extracting a target sentence from multi-talker speech using only a keyword as input. For example, in social security applications, the keyword might be "help", and the goal is to identify what the person who called for help is articulating while ignoring other speakers. To address this problem, we propose using the Transformer architecture to embed both the keyword and the speech utterance and then rely on the cross-attention mechanism to select the correct content from the concatenated or overlapping speech. Experimental results on Librispeech demonstrate that our proposed method can effectively extract target sentences from very noisy and mixed speech (SNR=-3dB), achieving a phone error rate (PER) of 26\%, compared to the baseline system's PER of 96%.

{{</citation>}}


### (58/183) SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF (Yi Dong et al., 2023)

{{<citation>}}

Yi Dong, Zhilin Wang, Makesh Narsimhan Sreedhar, Xianchao Wu, Oleksii Kuchaiev. (2023)  
**SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05344v1)  

---


**ABSTRACT**  
Model alignment with human preferences is an essential step in making Large Language Models (LLMs) helpful and consistent with human values. It typically consists of supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF) stages. However, RLHF faces inherent limitations stemming from a complex training setup and its tendency to align the model with implicit values that end users cannot control at run-time. Moreover, reward models in RLHF stage commonly rely on single-dimensional feedback as opposed to explicit, multifaceted signals that indicate attributes such as helpfulness, humor, and toxicity. To address these limitations, we propose SteerLM, a supervised fine-tuning method that empowers end-users to control responses during inference. SteerLM conditions responses to conform to an explicitly defined multi-dimensional set of attributes, thereby empowering a steerable AI capable of generating helpful and high-quality responses while maintaining customizability. Experiments show that SteerLM trained on open source datasets generates responses that are preferred by human and automatic evaluators to many state-of-the-art baselines trained with RLHF while being much easier to train. Try SteerLM at https://huggingface.co/nvidia/SteerLM-llama2-13B

{{</citation>}}


### (59/183) Resolving the Imbalance Issue in Hierarchical Disciplinary Topic Inference via LLM-based Data Augmentation (Xunxin Cai et al., 2023)

{{<citation>}}

Xunxin Cai, Meng Xiao, Zhiyuan Ning, Yuanchun Zhou. (2023)  
**Resolving the Imbalance Issue in Hierarchical Disciplinary Topic Inference via LLM-based Data Augmentation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Augmentation, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.05318v1)  

---


**ABSTRACT**  
In addressing the imbalanced issue of data within the realm of Natural Language Processing, text data augmentation methods have emerged as pivotal solutions. This data imbalance is prevalent in the research proposals submitted during the funding application process. Such imbalances, resulting from the varying popularity of disciplines or the emergence of interdisciplinary studies, significantly impede the precision of downstream topic models that deduce the affiliated disciplines of these proposals. At the data level, proposals penned by experts and scientists are inherently complex technological texts, replete with intricate terminologies, which augmenting such specialized text data poses unique challenges. At the system level, this, in turn, compromises the fairness of AI-assisted reviewer assignment systems, which raises a spotlight on solving this issue. This study leverages large language models (Llama V1) as data generators to augment research proposals categorized within intricate disciplinary hierarchies, aiming to rectify data imbalances and enhance the equity of expert assignments. We first sample within the hierarchical structure to find the under-represented class. Then we designed a prompt for keyword-based research proposal generation. Our experiments attests to the efficacy of the generated data, demonstrating that research proposals produced using the prompts can effectively address the aforementioned issues and generate high quality scientific text data, thus help the model overcome the imbalanced issue.

{{</citation>}}


### (60/183) Enhancing Long-form Text Generation in Mental Health with Task-adaptive Tokenization (Siyang Liu et al., 2023)

{{<citation>}}

Siyang Liu, Naihao Deng, Sahand Sabour, Yilin Jia, Minlie Huang, Rada Mihalcea. (2023)  
**Enhancing Long-form Text Generation in Mental Health with Task-adaptive Tokenization**  

---
Primary Category: cs.CL  
Categories: 68, I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2310.05317v2)  

---


**ABSTRACT**  
We propose task-adaptive tokenization as a way to adapt the generation pipeline to the specifics of a downstream task and enhance long-form generation in mental health. Inspired by insights from cognitive science, our task-adaptive tokenizer samples variable segmentations from multiple outcomes, with sampling probabilities optimized based on task-specific data. We introduce a strategy for building a specialized vocabulary and introduce a vocabulary merging protocol that allows for the integration of task-specific tokens into the pre-trained model's tokenization step. Through extensive experiments on psychological question-answering tasks in both Chinese and English, we find that our task-adaptive tokenization approach brings a significant improvement in generation performance while using up to 60% fewer tokens. Preliminary experiments point to promising results when using our tokenization approach with very large language models.

{{</citation>}}


## cs.CV (28)



### (61/183) DiPS: Discriminative Pseudo-Label Sampling with Self-Supervised Transformers for Weakly Supervised Object Localization (Shakeeb Murtaza et al., 2023)

{{<citation>}}

Shakeeb Murtaza, Soufiane Belharbi, Marco Pedersoli, Aydin Sarraf, Eric Granger. (2023)  
**DiPS: Discriminative Pseudo-Label Sampling with Self-Supervised Transformers for Weakly Supervised Object Localization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone, Self-Supervised, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.06196v1)  

---


**ABSTRACT**  
Self-supervised vision transformers (SSTs) have shown great potential to yield rich localization maps that highlight different objects in an image. However, these maps remain class-agnostic since the model is unsupervised. They often tend to decompose the image into multiple maps containing different objects while being unable to distinguish the object of interest from background noise objects. In this paper, Discriminative Pseudo-label Sampling (DiPS) is introduced to leverage these class-agnostic maps for weakly-supervised object localization (WSOL), where only image-class labels are available. Given multiple attention maps, DiPS relies on a pre-trained classifier to identify the most discriminative regions of each attention map. This ensures that the selected ROIs cover the correct image object while discarding the background ones, and, as such, provides a rich pool of diverse and discriminative proposals to cover different parts of the object. Subsequently, these proposals are used as pseudo-labels to train our new transformer-based WSOL model designed to perform classification and localization tasks. Unlike standard WSOL methods, DiPS optimizes performance in both tasks by using a transformer encoder and a dedicated output head for each task, each trained using dedicated loss functions. To avoid overfitting a single proposal and promote better object coverage, a single proposal is randomly selected among the top ones for a training image at each training step. Experimental results on the challenging CUB, ILSVRC, OpenImages, and TelDrone datasets indicate that our architecture, in combination with our transformer-based proposals, can yield better localization performance than state-of-the-art methods.

{{</citation>}}


### (62/183) Text-driven Prompt Generation for Vision-Language Models in Federated Learning (Chen Qiu et al., 2023)

{{<citation>}}

Chen Qiu, Xingyu Li, Chaithanya Kumar Mummadi, Madan Ravi Ganesh, Zhenzhen Li, Lu Peng, Wan-Yi Lin. (2023)  
**Text-driven Prompt Generation for Vision-Language Models in Federated Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06123v1)  

---


**ABSTRACT**  
Prompt learning for vision-language models, e.g., CoOp, has shown great success in adapting CLIP to different downstream tasks, making it a promising solution for federated learning due to computational reasons. Existing prompt learning techniques replace hand-crafted text prompts with learned vectors that offer improvements on seen classes, but struggle to generalize to unseen classes. Our work addresses this challenge by proposing Federated Text-driven Prompt Generation (FedTPG), which learns a unified prompt generation network across multiple remote clients in a scalable manner. The prompt generation network is conditioned on task-related text input, thus is context-aware, making it suitable to generalize for both seen and unseen classes. Our comprehensive empirical evaluations on nine diverse image classification datasets show that our method is superior to existing federated prompt learning methods, that achieve overall better generalization on both seen and unseen classes and is also generalizable to unseen datasets.

{{</citation>}}


### (63/183) DyST: Towards Dynamic Neural Scene Representations on Real-World Videos (Maximilian Seitzer et al., 2023)

{{<citation>}}

Maximilian Seitzer, Sjoerd van Steenkiste, Thomas Kipf, Klaus Greff, Mehdi S. M. Sajjadi. (2023)  
**DyST: Towards Dynamic Neural Scene Representations on Real-World Videos**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-GR, cs-LG, cs-RO, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.06020v1)  

---


**ABSTRACT**  
Visual understanding of the world goes beyond the semantics and flat structure of individual images. In this work, we aim to capture both the 3D structure and dynamics of real-world scenes from monocular real-world videos. Our Dynamic Scene Transformer (DyST) model leverages recent work in neural scene representation to learn a latent decomposition of monocular real-world videos into scene content, per-view scene dynamics, and camera pose. This separation is achieved through a novel co-training scheme on monocular videos and our new synthetic dataset DySO. DyST learns tangible latent representations for dynamic scenes that enable view generation with separate control over the camera and the content of the scene.

{{</citation>}}


### (64/183) SimPLR: A Simple and Plain Transformer for Object Detection and Segmentation (Duy-Kien Nguyen et al., 2023)

{{<citation>}}

Duy-Kien Nguyen, Martin R. Oswald, Cees G. M. Snoek. (2023)  
**SimPLR: A Simple and Plain Transformer for Object Detection and Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2310.05920v1)  

---


**ABSTRACT**  
The ability to detect objects in images at varying scales has played a pivotal role in the design of modern object detectors. Despite considerable progress in removing handcrafted components using transformers, multi-scale feature maps remain a key factor for their empirical success, even with a plain backbone like the Vision Transformer (ViT). In this paper, we show that this reliance on feature pyramids is unnecessary and a transformer-based detector with scale-aware attention enables the plain detector `SimPLR' whose backbone and detection head both operate on single-scale features. The plain architecture allows SimPLR to effectively take advantages of self-supervised learning and scaling approaches with ViTs, yielding strong performance compared to multi-scale counterparts. We demonstrate through our experiments that when scaling to larger backbones, SimPLR indicates better performance than end-to-end detectors (Mask2Former) and plain-backbone detectors (ViTDet), while consistently being faster. The code will be released.

{{</citation>}}


### (65/183) CoBEVFusion: Cooperative Perception with LiDAR-Camera Bird's-Eye View Fusion (Donghao Qiao et al., 2023)

{{<citation>}}

Donghao Qiao, Farhana Zulkernine. (2023)  
**CoBEVFusion: Cooperative Perception with LiDAR-Camera Bird's-Eye View Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.06008v1)  

---


**ABSTRACT**  
Autonomous Vehicles (AVs) use multiple sensors to gather information about their surroundings. By sharing sensor data between Connected Autonomous Vehicles (CAVs), the safety and reliability of these vehicles can be improved through a concept known as cooperative perception. However, recent approaches in cooperative perception only share single sensor information such as cameras or LiDAR. In this research, we explore the fusion of multiple sensor data sources and present a framework, called CoBEVFusion, that fuses LiDAR and camera data to create a Bird's-Eye View (BEV) representation. The CAVs process the multi-modal data locally and utilize a Dual Window-based Cross-Attention (DWCA) module to fuse the LiDAR and camera features into a unified BEV representation. The fused BEV feature maps are shared among the CAVs, and a 3D Convolutional Neural Network is applied to aggregate the features from the CAVs. Our CoBEVFusion framework was evaluated on the cooperative perception dataset OPV2V for two perception tasks: BEV semantic segmentation and 3D object detection. The results show that our DWCA LiDAR-camera fusion model outperforms perception models with single-modal data and state-of-the-art BEV fusion models. Our overall cooperative perception architecture, CoBEVFusion, also achieves comparable performance with other cooperative perception models.

{{</citation>}}


### (66/183) ViCor: Bridging Visual Understanding and Commonsense Reasoning with Large Language Models (Kaiwen Zhou et al., 2023)

{{<citation>}}

Kaiwen Zhou, Kwonjoon Lee, Teruhisa Misu, Xin Eric Wang. (2023)  
**ViCor: Bridging Visual Understanding and Commonsense Reasoning with Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.05872v1)  

---


**ABSTRACT**  
In our work, we explore the synergistic capabilities of pre-trained vision-and-language models (VLMs) and large language models (LLMs) for visual commonsense reasoning (VCR). We categorize the problem of VCR into visual commonsense understanding (VCU) and visual commonsense inference (VCI). For VCU, which involves perceiving the literal visual content, pre-trained VLMs exhibit strong cross-dataset generalization. On the other hand, in VCI, where the goal is to infer conclusions beyond image content, VLMs face difficulties. We find that a baseline where VLMs provide perception results (image captions) to LLMs leads to improved performance on VCI. However, we identify a challenge with VLMs' passive perception, which often misses crucial context information, leading to incorrect or uncertain reasoning by LLMs. To mitigate this issue, we suggest a collaborative approach where LLMs, when uncertain about their reasoning, actively direct VLMs to concentrate on and gather relevant visual elements to support potential commonsense inferences. In our method, named ViCor, pre-trained LLMs serve as problem classifiers to analyze the problem category, VLM commanders to leverage VLMs differently based on the problem classification, and visual commonsense reasoners to answer the question. VLMs will perform visual recognition and understanding. We evaluate our framework on two VCR benchmark datasets and outperform all other methods that do not require in-domain supervised fine-tuning.

{{</citation>}}


### (67/183) DANet: Enhancing Small Object Detection through an Efficient Deformable Attention Network (Md Sohag Mia et al., 2023)

{{<citation>}}

Md Sohag Mia, Abdullah Al Bary Voban, Abu Bakor Hayat Arnob, Abdu Naim, Md Kawsar Ahmed, Md Shariful Islam. (2023)  
**DANet: Enhancing Small Object Detection through an Efficient Deformable Attention Network**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, Object Detection  
[Paper Link](http://arxiv.org/abs/2310.05768v1)  

---


**ABSTRACT**  
Efficient and accurate detection of small objects in manufacturing settings, such as defects and cracks, is crucial for ensuring product quality and safety. To address this issue, we proposed a comprehensive strategy by synergizing Faster R-CNN with cutting-edge methods. By combining Faster R-CNN with Feature Pyramid Network, we enable the model to efficiently handle multi-scale features intrinsic to manufacturing environments. Additionally, Deformable Net is used that contorts and conforms to the geometric variations of defects, bringing precision in detecting even the minuscule and complex features. Then, we incorporated an attention mechanism called Convolutional Block Attention Module in each block of our base ResNet50 network to selectively emphasize informative features and suppress less useful ones. After that we incorporated RoI Align, replacing RoI Pooling for finer region-of-interest alignment and finally the integration of Focal Loss effectively handles class imbalance, crucial for rare defect occurrences. The rigorous evaluation of our model on both the NEU-DET and Pascal VOC datasets underscores its robust performance and generalization capabilities. On the NEU-DET dataset, our model exhibited a profound understanding of steel defects, achieving state-of-the-art accuracy in identifying various defects. Simultaneously, when evaluated on the Pascal VOC dataset, our model showcases its ability to detect objects across a wide spectrum of categories within complex and small scenes.

{{</citation>}}


### (68/183) Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation (Lijun Yu et al., 2023)

{{<citation>}}

Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, Lu Jiang. (2023)  
**Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-MM, cs.CV  
Keywords: ImageNet, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05737v1)  

---


**ABSTRACT**  
While Large Language Models (LLMs) are the dominant models for generative tasks in language, they do not perform as well as diffusion models on image and video generation. To effectively use LLMs for visual generation, one crucial component is the visual tokenizer that maps pixel-space inputs to discrete tokens appropriate for LLM learning. In this paper, we introduce MAGVIT-v2, a video tokenizer designed to generate concise and expressive tokens for both videos and images using a common token vocabulary. Equipped with this new tokenizer, we show that LLMs outperform diffusion models on standard image and video generation benchmarks including ImageNet and Kinetics. In addition, we demonstrate that our tokenizer surpasses the previously top-performing video tokenizer on two more tasks: (1) video compression comparable to the next-generation video codec (VCC) according to human evaluations, and (2) learning effective representations for action recognition tasks.

{{</citation>}}


### (69/183) Uni3DETR: Unified 3D Detection Transformer (Zhenyu Wang et al., 2023)

{{<citation>}}

Zhenyu Wang, Yali Li, Xi Chen, Hengshuang Zhao, Shengjin Wang. (2023)  
**Uni3DETR: Unified 3D Detection Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.05699v1)  

---


**ABSTRACT**  
Existing point cloud based 3D detectors are designed for the particular scene, either indoor or outdoor ones. Because of the substantial differences in object distribution and point density within point clouds collected from various environments, coupled with the intricate nature of 3D metrics, there is still a lack of a unified network architecture that can accommodate diverse scenes. In this paper, we propose Uni3DETR, a unified 3D detector that addresses indoor and outdoor 3D detection within the same framework. Specifically, we employ the detection transformer with point-voxel interaction for object prediction, which leverages voxel features and points for cross-attention and behaves resistant to the discrepancies from data. We then propose the mixture of query points, which sufficiently exploits global information for dense small-range indoor scenes and local information for large-range sparse outdoor ones. Furthermore, our proposed decoupled IoU provides an easy-to-optimize training target for localization by disentangling the xy and z space. Extensive experiments validate that Uni3DETR exhibits excellent performance consistently on both indoor and outdoor 3D detection. In contrast to previous specialized detectors, which may perform well on some particular datasets but suffer a substantial degradation on different scenes, Uni3DETR demonstrates the strong generalization ability under heterogeneous conditions (Fig. 1).   Codes are available at \href{https://github.com/zhenyuw16/Uni3DETR}{https://github.com/zhenyuw16/Uni3DETR}.

{{</citation>}}


### (70/183) Combining recurrent and residual learning for deforestation monitoring using multitemporal SAR images (Carla Nascimento Neves et al., 2023)

{{<citation>}}

Carla Nascimento Neves, Raul Queiroz Feitosa, Mabel X. Ortega Adarme, Gilson Antonio Giraldi. (2023)  
**Combining recurrent and residual learning for deforestation monitoring using multitemporal SAR images**  

---
Primary Category: cs.CV  
Categories: I-4-9, cs-CV, cs-LG, cs.CV  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2310.05697v1)  

---


**ABSTRACT**  
With its vast expanse, exceeding that of Western Europe by twice, the Amazon rainforest stands as the largest forest of the Earth, holding immense importance in global climate regulation. Yet, deforestation detection from remote sensing data in this region poses a critical challenge, often hindered by the persistent cloud cover that obscures optical satellite data for much of the year. Addressing this need, this paper proposes three deep-learning models tailored for deforestation monitoring, utilizing SAR (Synthetic Aperture Radar) multitemporal data moved by its independence on atmospheric conditions. Specifically, the study proposes three novel recurrent fully convolutional network architectures-namely, RRCNN-1, RRCNN-2, and RRCNN-3, crafted to enhance the accuracy of deforestation detection. Additionally, this research explores replacing a bitemporal with multitemporal SAR sequences, motivated by the hypothesis that deforestation signs quickly fade in SAR images over time. A comprehensive assessment of the proposed approaches was conducted using a Sentinel-1 multitemporal sequence from a sample site in the Brazilian rainforest. The experimental analysis confirmed that analyzing a sequence of SAR images over an observation period can reveal deforestation spots undetectable in a pair of images. Notably, experimental results underscored the superiority of the multitemporal approach, yielding approximately a five percent enhancement in F1-Score across all tested network architectures. Particularly the RRCNN-1 achieved the highest accuracy and also boasted half the processing time of its closest counterpart.

{{</citation>}}


### (71/183) Anchor-Intermediate Detector: Decoupling and Coupling Bounding Boxes for Accurate Object Detection (Yilong Lv et al., 2023)

{{<citation>}}

Yilong Lv, Min Li, Yujie He, Shaopeng Li, Zhuzhen He, Aitao Yang. (2023)  
**Anchor-Intermediate Detector: Decoupling and Coupling Bounding Boxes for Accurate Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Object Detection  
[Paper Link](http://arxiv.org/abs/2310.05666v1)  

---


**ABSTRACT**  
Anchor-based detectors have been continuously developed for object detection. However, the individual anchor box makes it difficult to predict the boundary's offset accurately. Instead of taking each bounding box as a closed individual, we consider using multiple boxes together to get prediction boxes. To this end, this paper proposes the \textbf{Box Decouple-Couple(BDC) strategy} in the inference, which no longer discards the overlapping boxes, but decouples the corner points of these boxes. Then, according to each corner's score, we couple the corner points to select the most accurate corner pairs. To meet the BDC strategy, a simple but novel model is designed named the \textbf{Anchor-Intermediate Detector(AID)}, which contains two head networks, i.e., an anchor-based head and an anchor-free \textbf{Corner-aware head}. The corner-aware head is able to score the corners of each bounding box to facilitate the coupling between corner points. Extensive experiments on MS COCO show that the proposed anchor-intermediate detector respectively outperforms their baseline RetinaNet and GFL method by $\sim$2.4 and $\sim$1.2 AP on the MS COCO test-dev dataset without any bells and whistles. Code is available at: https://github.com/YilongLv/AID.

{{</citation>}}


### (72/183) ViTs are Everywhere: A Comprehensive Study Showcasing Vision Transformers in Different Domain (Md Sohag Mia et al., 2023)

{{<citation>}}

Md Sohag Mia, Abu Bakor Hayat Arnob, Abdu Naim+, Abdullah Al Bary Voban, Md Shariful Islam. (2023)  
**ViTs are Everywhere: A Comprehensive Study Showcasing Vision Transformers in Different Domain**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.05664v1)  

---


**ABSTRACT**  
Transformer design is the de facto standard for natural language processing tasks. The success of the transformer design in natural language processing has lately piqued the interest of researchers in the domain of computer vision. When compared to Convolutional Neural Networks (CNNs), Vision Transformers (ViTs) are becoming more popular and dominant solutions for many vision problems. Transformer-based models outperform other types of networks, such as convolutional and recurrent neural networks, in a range of visual benchmarks. We evaluate various vision transformer models in this work by dividing them into distinct jobs and examining their benefits and drawbacks. ViTs can overcome several possible difficulties with convolutional neural networks (CNNs). The goal of this survey is to show the first use of ViTs in CV. In the first phase, we categorize various CV applications where ViTs are appropriate. Image classification, object identification, image segmentation, video transformer, image denoising, and NAS are all CV applications. Our next step will be to analyze the state-of-the-art in each area and identify the models that are currently available. In addition, we outline numerous open research difficulties as well as prospective research possibilities.

{{</citation>}}


### (73/183) No Token Left Behind: Efficient Vision Transformer via Dynamic Token Idling (Xuwei Xu et al., 2023)

{{<citation>}}

Xuwei Xu, Changlin Li, Yudong Chen, Xiaojun Chang, Jiajun Liu, Sen Wang. (2023)  
**No Token Left Behind: Efficient Vision Transformer via Dynamic Token Idling**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.05654v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) have demonstrated outstanding performance in computer vision tasks, yet their high computational complexity prevents their deployment in computing resource-constrained environments. Various token pruning techniques have been introduced to alleviate the high computational burden of ViTs by dynamically dropping image tokens. However, some undesirable pruning at early stages may result in permanent loss of image information in subsequent layers, consequently hindering model performance. To address this problem, we propose IdleViT, a dynamic token-idle-based method that achieves an excellent trade-off between performance and efficiency. Specifically, in each layer, IdleViT selects a subset of the image tokens to participate in computations while keeping the rest of the tokens idle and directly passing them to this layer's output. By allowing the idle tokens to be re-selected in the following layers, IdleViT mitigates the negative impact of improper pruning in the early stages. Furthermore, inspired by the normalized graph cut, we devise a token cut loss on the attention map as regularization to improve IdleViT's token selection ability. Our method is simple yet effective and can be extended to pyramid ViTs since no token is completely dropped. Extensive experimental results on various ViT architectures have shown that IdleViT can diminish the complexity of pretrained ViTs by up to 33\% with no more than 0.2\% accuracy decrease on ImageNet, after finetuning for only 30 epochs. Notably, when the keep ratio is 0.5, IdleViT outperforms the state-of-the-art EViT on DeiT-S by 0.5\% higher accuracy and even faster inference speed. The source code is available in the supplementary material.

{{</citation>}}


### (74/183) Plug n' Play: Channel Shuffle Module for Enhancing Tiny Vision Transformers (Xuwei Xu et al., 2023)

{{<citation>}}

Xuwei Xu, Sen Wang, Yudong Chen, Jiajun Liu. (2023)  
**Plug n' Play: Channel Shuffle Module for Enhancing Tiny Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.05642v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) have demonstrated remarkable performance in various computer vision tasks. However, the high computational complexity hinders ViTs' applicability on devices with limited memory and computing resources. Although certain investigations have delved into the fusion of convolutional layers with self-attention mechanisms to enhance the efficiency of ViTs, there remains a knowledge gap in constructing tiny yet effective ViTs solely based on the self-attention mechanism. Furthermore, the straightforward strategy of reducing the feature channels in a large but outperforming ViT often results in significant performance degradation despite improved efficiency. To address these challenges, we propose a novel channel shuffle module to improve tiny-size ViTs, showing the potential of pure self-attention models in environments with constrained computing resources. Inspired by the channel shuffle design in ShuffleNetV2 \cite{ma2018shufflenet}, our module expands the feature channels of a tiny ViT and partitions the channels into two groups: the \textit{Attended} and \textit{Idle} groups. Self-attention computations are exclusively employed on the designated \textit{Attended} group, followed by a channel shuffle operation that facilitates information exchange between the two groups. By incorporating our module into a tiny ViT, we can achieve superior performance while maintaining a comparable computational complexity to the vanilla model. Specifically, our proposed channel shuffle module consistently improves the top-1 accuracy on the ImageNet-1K dataset for various tiny ViT models by up to 2.8\%, with the changes in model complexity being less than 0.03 GMACs.

{{</citation>}}


### (75/183) Adaptive Multi-head Contrastive Learning (Lei Wang et al., 2023)

{{<citation>}}

Lei Wang, Piotr Koniusz, Tom Gedeon, Liang Zheng. (2023)  
**Adaptive Multi-head Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.05615v1)  

---


**ABSTRACT**  
In contrastive learning, two views of an original image generated by different augmentations are considered as a positive pair whose similarity is required to be high. Moreover, two views of two different images are considered as a negative pair, and their similarity is encouraged to be low. Normally, a single similarity measure given by a single projection head is used to evaluate positive and negative sample pairs, respectively. However, due to the various augmentation strategies and varying intra-sample similarity, augmented views from the same image are often not similar. Moreover, due to inter-sample similarity, augmented views of two different images may be more similar than augmented views from the same image. As such, enforcing a high similarity for positive pairs and a low similarity for negative pairs may not always be achievable, and in the case of some pairs, forcing so may be detrimental to the performance. To address this issue, we propose to use multiple projection heads, each producing a separate set of features. Our loss function for pre-training emerges from a solution to the maximum likelihood estimation over head-wise posterior distributions of positive samples given observations. The loss contains the similarity measure over positive and negative pairs, each re-weighted by an individual adaptive temperature that is regularized to prevent ill solutions. Our adaptive multi-head contrastive learning (AMCL) can be applied to and experimentally improves several popular contrastive learning methods such as SimCLR, MoCo and Barlow Twins. Such improvement is consistent under various backbones and linear probing epoches and is more significant when multiple augmentation methods are used.

{{</citation>}}


### (76/183) WeatherDepth: Curriculum Contrastive Learning for Self-Supervised Depth Estimation under Adverse Weather Conditions (Jiyuan Wang et al., 2023)

{{<citation>}}

Jiyuan Wang, Chunyu Lin, Lang Nie, Shujun Huang, Yao Zhao, Xing Pan, Rui Ai. (2023)  
**WeatherDepth: Curriculum Contrastive Learning for Self-Supervised Depth Estimation under Adverse Weather Conditions**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Contrastive Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.05556v1)  

---


**ABSTRACT**  
Depth estimation models have shown promising performance on clear scenes but fail to generalize to adverse weather conditions due to illumination variations, weather particles, etc. In this paper, we propose WeatherDepth, a self-supervised robust depth estimation model with curriculum contrastive learning, to tackle performance degradation in complex weather conditions. Concretely, we first present a progressive curriculum learning scheme with three simple-to-complex curricula to gradually adapt the model from clear to relative adverse, and then to adverse weather scenes. It encourages the model to gradually grasp beneficial depth cues against the weather effect, yielding smoother and better domain adaption. Meanwhile, to prevent the model from forgetting previous curricula, we integrate contrastive learning into different curricula. Drawn the reference knowledge from the previous course, our strategy establishes a depth consistency constraint between different courses towards robust depth estimation in diverse weather. Besides, to reduce manual intervention and better adapt to different models, we designed an adaptive curriculum scheduler to automatically search for the best timing for course switching. In the experiment, the proposed solution is proven to be easily incorporated into various architectures and demonstrates state-of-the-art (SoTA) performance on both synthetic and real weather datasets.

{{</citation>}}


### (77/183) Semi-Supervised Object Detection with Uncurated Unlabeled Data for Remote Sensing Images (Nanqing Liu et al., 2023)

{{<citation>}}

Nanqing Liu, Xun Xu, Yingjie Gao, Heng-Chao Li. (2023)  
**Semi-Supervised Object Detection with Uncurated Unlabeled Data for Remote Sensing Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.05498v1)  

---


**ABSTRACT**  
Annotating remote sensing images (RSIs) presents a notable challenge due to its labor-intensive nature. Semi-supervised object detection (SSOD) methods tackle this issue by generating pseudo-labels for the unlabeled data, assuming that all classes found in the unlabeled dataset are also represented in the labeled data. However, real-world situations introduce the possibility of out-of-distribution (OOD) samples being mixed with in-distribution (ID) samples within the unlabeled dataset. In this paper, we delve into techniques for conducting SSOD directly on uncurated unlabeled data, which is termed Open-Set Semi-Supervised Object Detection (OSSOD). Our approach commences by employing labeled in-distribution data to dynamically construct a class-wise feature bank (CFB) that captures features specific to each class. Subsequently, we compare the features of predicted object bounding boxes with the corresponding entries in the CFB to calculate OOD scores. We design an adaptive threshold based on the statistical properties of the CFB, allowing us to filter out OOD samples effectively. The effectiveness of our proposed method is substantiated through extensive experiments on two widely used remote sensing object detection datasets: DIOR and DOTA. These experiments showcase the superior performance and efficacy of our approach for OSSOD on RSIs.

{{</citation>}}


### (78/183) Geometry-Guided Ray Augmentation for Neural Surface Reconstruction with Sparse Views (Jiawei Yao et al., 2023)

{{<citation>}}

Jiawei Yao, Chen Wang, Tong Wu, Chuming Li. (2023)  
**Geometry-Guided Ray Augmentation for Neural Surface Reconstruction with Sparse Views**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.05483v1)  

---


**ABSTRACT**  
In this paper, we propose a novel method for 3D scene and object reconstruction from sparse multi-view images. Different from previous methods that leverage extra information such as depth or generalizable features across scenes, our approach leverages the scene properties embedded in the multi-view inputs to create precise pseudo-labels for optimization without any prior training. Specifically, we introduce a geometry-guided approach that improves surface reconstruction accuracy from sparse views by leveraging spherical harmonics to predict the novel radiance while holistically considering all color observations for a point in the scene. Also, our pipeline exploits proxy geometry and correctly handles the occlusion in generating the pseudo-labels of radiance, which previous image-warping methods fail to avoid. Our method, dubbed Ray Augmentation (RayAug), achieves superior results on DTU and Blender datasets without requiring prior training, demonstrating its effectiveness in addressing the problem of sparse view reconstruction. Our pipeline is flexible and can be integrated into other implicit neural reconstruction methods for sparse views.

{{</citation>}}


### (79/183) AdaFuse: Adaptive Medical Image Fusion Based on Spatial-Frequential Cross Attention (Xianming Gu et al., 2023)

{{<citation>}}

Xianming Gu, Lihui Wang, Zeyu Deng, Ying Cao, Xingyu Huang, Yue-min Zhu. (2023)  
**AdaFuse: Adaptive Medical Image Fusion Based on Spatial-Frequential Cross Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.05462v1)  

---


**ABSTRACT**  
Multi-modal medical image fusion is essential for the precise clinical diagnosis and surgical navigation since it can merge the complementary information in multi-modalities into a single image. The quality of the fused image depends on the extracted single modality features as well as the fusion rules for multi-modal information. Existing deep learning-based fusion methods can fully exploit the semantic features of each modality, they cannot distinguish the effective low and high frequency information of each modality and fuse them adaptively. To address this issue, we propose AdaFuse, in which multimodal image information is fused adaptively through frequency-guided attention mechanism based on Fourier transform. Specifically, we propose the cross-attention fusion (CAF) block, which adaptively fuses features of two modalities in the spatial and frequency domains by exchanging key and query values, and then calculates the cross-attention scores between the spatial and frequency features to further guide the spatial-frequential information fusion. The CAF block enhances the high-frequency features of the different modalities so that the details in the fused images can be retained. Moreover, we design a novel loss function composed of structure loss and content loss to preserve both low and high frequency information. Extensive comparison experiments on several datasets demonstrate that the proposed method outperforms state-of-the-art methods in terms of both visual quality and quantitative metrics. The ablation experiments also validate the effectiveness of the proposed loss and fusion strategy. Our code is publicly available at https://github.com/xianming-gu/AdaFuse.

{{</citation>}}


### (80/183) Towards Fair and Comprehensive Comparisons for Image-Based 3D Object Detection (Xinzhu Ma et al., 2023)

{{<citation>}}

Xinzhu Ma, Yongtao Wang, Yinmin Zhang, Zhiyi Xia, Yuan Meng, Zhihui Wang, Haojie Li, Wanli Ouyang. (2023)  
**Towards Fair and Comprehensive Comparisons for Image-Based 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.05447v2)  

---


**ABSTRACT**  
In this work, we build a modular-designed codebase, formulate strong training recipes, design an error diagnosis toolbox, and discuss current methods for image-based 3D object detection. In particular, different from other highly mature tasks, e.g., 2D object detection, the community of image-based 3D object detection is still evolving, where methods often adopt different training recipes and tricks resulting in unfair evaluations and comparisons. What is worse, these tricks may overwhelm their proposed designs in performance, even leading to wrong conclusions. To address this issue, we build a module-designed codebase and formulate unified training standards for the community. Furthermore, we also design an error diagnosis toolbox to measure the detailed characterization of detection models. Using these tools, we analyze current methods in-depth under varying settings and provide discussions for some open questions, e.g., discrepancies in conclusions on KITTI-3D and nuScenes datasets, which have led to different dominant methods for these datasets. We hope that this work will facilitate future research in image-based 3D object detection. Our codes will be released at \url{https://github.com/OpenGVLab/3dodi}

{{</citation>}}


### (81/183) Semantic-aware Temporal Channel-wise Attention for Cardiac Function Assessment (Guanqi Chen et al., 2023)

{{<citation>}}

Guanqi Chen, Guanbin Li. (2023)  
**Semantic-aware Temporal Channel-wise Attention for Cardiac Function Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.05428v1)  

---


**ABSTRACT**  
Cardiac function assessment aims at predicting left ventricular ejection fraction (LVEF) given an echocardiogram video, which requests models to focus on the changes in the left ventricle during the cardiac cycle. How to assess cardiac function accurately and automatically from an echocardiogram video is a valuable topic in intelligent assisted healthcare. Existing video-based methods do not pay much attention to the left ventricular region, nor the left ventricular changes caused by motion. In this work, we propose a semi-supervised auxiliary learning paradigm with a left ventricular segmentation task, which contributes to the representation learning for the left ventricular region. To better model the importance of motion information, we introduce a temporal channel-wise attention (TCA) module to excite those channels used to describe motion. Furthermore, we reform the TCA module with semantic perception by taking the segmentation map of the left ventricle as input to focus on the motion patterns of the left ventricle. Finally, to reduce the difficulty of direct LVEF regression, we utilize an anchor-based classification and regression method to predict LVEF. Our approach achieves state-of-the-art performance on the Stanford dataset with an improvement of 0.22 MAE, 0.26 RMSE, and 1.9% $R^2$.

{{</citation>}}


### (82/183) Efficient-VQGAN: Towards High-Resolution Image Generation with Efficient Vision Transformers (Shiyue Cao et al., 2023)

{{<citation>}}

Shiyue Cao, Yueqin Yin, Lianghua Huang, Yu Liu, Xin Zhao, Deli Zhao, Kaiqi Huang. (2023)  
**Efficient-VQGAN: Towards High-Resolution Image Generation with Efficient Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.05400v1)  

---


**ABSTRACT**  
Vector-quantized image modeling has shown great potential in synthesizing high-quality images. However, generating high-resolution images remains a challenging task due to the quadratic computational overhead of the self-attention process. In this study, we seek to explore a more efficient two-stage framework for high-resolution image generation with improvements in the following three aspects. (1) Based on the observation that the first quantization stage has solid local property, we employ a local attention-based quantization model instead of the global attention mechanism used in previous methods, leading to better efficiency and reconstruction quality. (2) We emphasize the importance of multi-grained feature interaction during image generation and introduce an efficient attention mechanism that combines global attention (long-range semantic consistency within the whole image) and local attention (fined-grained details). This approach results in faster generation speed, higher generation fidelity, and improved resolution. (3) We propose a new generation pipeline incorporating autoencoding training and autoregressive generation strategy, demonstrating a better paradigm for image synthesis. Extensive experiments demonstrate the superiority of our approach in high-quality and high-resolution image reconstruction and generation.

{{</citation>}}


### (83/183) Hierarchical Side-Tuning for Vision Transformers (Weifeng Lin et al., 2023)

{{<citation>}}

Weifeng Lin, Ziheng Wu, Jiayu Chen, Wentao Yang, Mingxin Huang, Jun Huang, Lianwen Jin. (2023)  
**Hierarchical Side-Tuning for Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.05393v2)  

---


**ABSTRACT**  
Fine-tuning pre-trained Vision Transformers (ViT) has consistently demonstrated promising performance in the realm of visual recognition. However, adapting large pre-trained models to various tasks poses a significant challenge. This challenge arises from the need for each model to undergo an independent and comprehensive fine-tuning process, leading to substantial computational and memory demands. While recent advancements in Parameter-efficient Transfer Learning (PETL) have demonstrated their ability to achieve superior performance compared to full fine-tuning with a smaller subset of parameter updates, they tend to overlook dense prediction tasks such as object detection and segmentation. In this paper, we introduce Hierarchical Side-Tuning (HST), a novel PETL approach that enables ViT transfer to various downstream tasks effectively. Diverging from existing methods that exclusively fine-tune parameters within input spaces or certain modules connected to the backbone, we tune a lightweight and hierarchical side network (HSN) that leverages intermediate activations extracted from the backbone and generates multi-scale features to make predictions. To validate HST, we conducted extensive experiments encompassing diverse visual tasks, including classification, object detection, instance segmentation, and semantic segmentation. Notably, our method achieves state-of-the-art average Top-1 accuracy of 76.0% on VTAB-1k, all while fine-tuning a mere 0.78M parameters. When applied to object detection tasks on COCO testdev benchmark, HST even surpasses full fine-tuning and obtains better performance with 49.7 box AP and 43.2 mask AP using Cascade Mask R-CNN.

{{</citation>}}


### (84/183) Rotation Matters: Generalized Monocular 3D Object Detection for Various Camera Systems (SungHo Moon et al., 2023)

{{<citation>}}

SungHo Moon, JinWoo Bae, SungHoon Im. (2023)  
**Rotation Matters: Generalized Monocular 3D Object Detection for Various Camera Systems**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.05366v1)  

---


**ABSTRACT**  
Research on monocular 3D object detection is being actively studied, and as a result, performance has been steadily improving. However, 3D object detection performance is significantly reduced when applied to a camera system different from the system used to capture the training datasets. For example, a 3D detector trained on datasets from a passenger car mostly fails to regress accurate 3D bounding boxes for a camera mounted on a bus. In this paper, we conduct extensive experiments to analyze the factors that cause performance degradation. We find that changing the camera pose, especially camera orientation, relative to the road plane caused performance degradation. In addition, we propose a generalized 3D object detection method that can be universally applied to various camera systems. We newly design a compensation module that corrects the estimated 3D bounding box location and heading direction. The proposed module can be applied to most of the recent 3D object detection networks. It increases AP3D score (KITTI moderate, IoU $> 70\%$) about 6-to-10-times above the baselines without additional training. Both quantitative and qualitative results show the effectiveness of the proposed method.

{{</citation>}}


### (85/183) Anyview: Generalizable Indoor 3D Object Detection with Variable Frames (Zhenyu Wu et al., 2023)

{{<citation>}}

Zhenyu Wu, Xiuwei Xu, Ziwei Wang, Chong Xia, Linqing Zhao, Jiwen Lu, Haibin Yan. (2023)  
**Anyview: Generalizable Indoor 3D Object Detection with Variable Frames**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.05346v1)  

---


**ABSTRACT**  
In this paper, we propose a novel network framework for indoor 3D object detection to handle variable input frame numbers in practical scenarios. Existing methods only consider fixed frames of input data for a single detector, such as monocular RGB-D images or point clouds reconstructed from dense multi-view RGB-D images. While in practical application scenes such as robot navigation and manipulation, the raw input to the 3D detectors is the RGB-D images with variable frame numbers instead of the reconstructed scene point cloud. However, the previous approaches can only handle fixed frame input data and have poor performance with variable frame input. In order to facilitate 3D object detection methods suitable for practical tasks, we present a novel 3D detection framework named AnyView for our practical applications, which generalizes well across different numbers of input frames with a single model. To be specific, we propose a geometric learner to mine the local geometric features of each input RGB-D image frame and implement local-global feature interaction through a designed spatial mixture module. Meanwhile, we further utilize a dynamic token strategy to adaptively adjust the number of extracted features for each frame, which ensures consistent global feature density and further enhances the generalization after fusion. Extensive experiments on the ScanNet dataset show our method achieves both great generalizability and high detection accuracy with a simple and clean architecture containing a similar amount of parameters with the baselines.

{{</citation>}}


### (86/183) A Critical Look at Classic Test-Time Adaptation Methods in Semantic Segmentation (Chang'an Yi et al., 2023)

{{<citation>}}

Chang'an Yi, Haotian Chen, Yifan Zhang, Yonghui Xu, Lizhen Cui. (2023)  
**A Critical Look at Classic Test-Time Adaptation Methods in Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.05341v3)  

---


**ABSTRACT**  
Test-time adaptation (TTA) aims to adapt a model, initially trained on training data, to potential distribution shifts in the test data. Most existing TTA studies, however, focus on classification tasks, leaving a notable gap in the exploration of TTA for semantic segmentation. This pronounced emphasis on classification might lead numerous newcomers and engineers to mistakenly assume that classic TTA methods designed for classification can be directly applied to segmentation. Nonetheless, this assumption remains unverified, posing an open question. To address this, we conduct a systematic, empirical study to disclose the unique challenges of segmentation TTA, and to determine whether classic TTA strategies can effectively address this task. Our comprehensive results have led to three key observations. First, the classic batch norm updating strategy, commonly used in classification TTA, only brings slight performance improvement, and in some cases it might even adversely affect the results. Even with the application of advanced distribution estimation techniques like batch renormalization, the problem remains unresolved. Second, the teacher-student scheme does enhance training stability for segmentation TTA in the presence of noisy pseudo-labels. However, it cannot directly result in performance improvement compared to the original model without TTA. Third, segmentation TTA suffers a severe long-tailed imbalance problem, which is substantially more complex than that in TTA for classification. This long-tailed challenge significantly affects segmentation TTA performance, even when the accuracy of pseudo-labels is high. In light of these observations, we conclude that TTA for segmentation presents significant challenges, and simply using classic TTA methods cannot address this problem well.

{{</citation>}}


### (87/183) Negative Object Presence Evaluation (NOPE) to Measure Object Hallucination in Vision-Language Models (Holy Lovenia et al., 2023)

{{<citation>}}

Holy Lovenia, Wenliang Dai, Samuel Cahyawijaya, Ziwei Ji, Pascale Fung. (2023)  
**Negative Object Presence Evaluation (NOPE) to Measure Object Hallucination in Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.05338v1)  

---


**ABSTRACT**  
Object hallucination poses a significant challenge in vision-language (VL) models, often leading to the generation of nonsensical or unfaithful responses with non-existent objects. However, the absence of a general measurement for evaluating object hallucination in VL models has hindered our understanding and ability to mitigate this issue. In this work, we present NOPE (Negative Object Presence Evaluation), a novel benchmark designed to assess object hallucination in VL models through visual question answering (VQA). We propose a cost-effective and scalable approach utilizing large language models to generate 29.5k synthetic negative pronoun (NegP) data of high quality for NOPE. We extensively investigate the performance of 10 state-of-the-art VL models in discerning the non-existence of objects in visual questions, where the ground truth answers are denoted as NegP (e.g., "none"). Additionally, we evaluate their standard performance on visual questions on 9 other VQA datasets. Through our experiments, we demonstrate that no VL model is immune to the vulnerability of object hallucination, as all models achieve accuracy below 10\% on NegP. Furthermore, we uncover that lexically diverse visual questions, question types with large scopes, and scene-relevant objects capitalize the risk of object hallucination in VL models.

{{</citation>}}


### (88/183) A Lightweight Video Anomaly Detection Model with Weak Supervision and Adaptive Instance Selection (Yang Wang et al., 2023)

{{<citation>}}

Yang Wang, Jiaogen Zhou, Jihong Guan. (2023)  
**A Lightweight Video Anomaly Detection Model with Weak Supervision and Adaptive Instance Selection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.05330v1)  

---


**ABSTRACT**  
Video anomaly detection is to determine whether there are any abnormal events, behaviors or objects in a given video, which enables effective and intelligent public safety management. As video anomaly labeling is both time-consuming and expensive, most existing works employ unsupervised or weakly supervised learning methods. This paper focuses on weakly supervised video anomaly detection, in which the training videos are labeled whether or not they contain any anomalies, but there is no information about which frames the anomalies are located. However, the uncertainty of weakly labeled data and the large model size prevent existing methods from wide deployment in real scenarios, especially the resource-limit situations such as edge-computing. In this paper, we develop a lightweight video anomaly detection model. On the one hand, we propose an adaptive instance selection strategy, which is based on the model's current status to select confident instances, thereby mitigating the uncertainty of weakly labeled data and subsequently promoting the model's performance. On the other hand, we design a lightweight multi-level temporal correlation attention module and an hourglass-shaped fully connected layer to construct the model, which can reduce the model parameters to only 0.56\% of the existing methods (e.g. RTFM). Our extensive experiments on two public datasets UCF-Crime and ShanghaiTech show that our model can achieve comparable or even superior AUC score compared to the state-of-the-art methods, with a significantly reduced number of model parameters.

{{</citation>}}


## cs.PF (2)



### (89/183) Look-Up mAI GeMM: Increasing AI GeMMs Performance by Nearly 2.5x via msGeMM (Saeed Maleki, 2023)

{{<citation>}}

Saeed Maleki. (2023)  
**Look-Up mAI GeMM: Increasing AI GeMMs Performance by Nearly 2.5x via msGeMM**  

---
Primary Category: cs.PF  
Categories: cs-AI, cs-DC, cs-LG, cs-PF, cs.PF  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.06178v1)  

---


**ABSTRACT**  
AI models are increasing in size and recent advancement in the community has shown that unlike HPC applications where double precision datatype are required, lower-precision datatypes such as fp8 or int4 are sufficient to bring the same model quality both for training and inference. Following these trends, GPU vendors such as NVIDIA and AMD have added hardware support for fp16, fp8 and int8 GeMM operations with an exceptional performance via Tensor Cores. However, this paper proposes a new algorithm called msGeMM which shows that AI models with low-precision datatypes can run with ~2.5x fewer multiplication and add instructions. Efficient implementation of this algorithm requires special CUDA cores with the ability to add elements from a small look-up table at the rate of Tensor Cores.

{{</citation>}}


### (90/183) Accelerating Deep Neural Network guided MCTS using Adaptive Parallelism (Yuan Meng et al., 2023)

{{<citation>}}

Yuan Meng, Qian Wang, Tianxin Zu, Viktor Prasanna. (2023)  
**Accelerating Deep Neural Network guided MCTS using Adaptive Parallelism**  

---
Primary Category: cs.PF  
Categories: cs-DC, cs-PF, cs.PF  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05313v1)  

---


**ABSTRACT**  
Deep Neural Network guided Monte-Carlo Tree Search (DNN-MCTS) is a powerful class of AI algorithms. In DNN-MCTS, a Deep Neural Network model is trained collaboratively with a dynamic Monte-Carlo search tree to guide the agent towards actions that yields the highest returns. While the DNN operations are highly parallelizable, the search tree operations involved in MCTS are sequential and often become the system bottleneck. Existing MCTS parallel schemes on shared-memory multi-core CPU platforms either exploit data parallelism but sacrifice memory access latency, or take advantage of local cache for low-latency memory accesses but constrain the tree search to a single thread. In this work, we analyze the tradeoff of these parallel schemes and develop performance models for both parallel schemes based on the application and hardware parameters. We propose a novel implementation that addresses the tradeoff by adaptively choosing the optimal parallel scheme for the MCTS component on the CPU. Furthermore, we propose an efficient method for searching the optimal communication batch size as the MCTS component on the CPU interfaces with DNN operations offloaded to an accelerator (GPU). Using a representative DNN-MCTS algorithm - Alphazero on board game benchmarks, we show that the parallel framework is able to adaptively generate the best-performing parallel implementation, leading to a range of $1.5\times - 3\times$ speedup compared with the baseline methods on CPU and CPU-GPU platforms.

{{</citation>}}


## cs.AI (16)



### (91/183) Factual and Personalized Recommendations using Language Models and Reinforcement Learning (Jihwan Jeong et al., 2023)

{{<citation>}}

Jihwan Jeong, Yinlam Chow, Guy Tennenholtz, Chih-Wei Hsu, Azamat Tulepbergenov, Mohammad Ghavamzadeh, Craig Boutilier. (2023)  
**Factual and Personalized Recommendations using Language Models and Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.06176v1)  

---


**ABSTRACT**  
Recommender systems (RSs) play a central role in connecting users to content, products, and services, matching candidate items to users based on their preferences. While traditional RSs rely on implicit user feedback signals, conversational RSs interact with users in natural language. In this work, we develop a comPelling, Precise, Personalized, Preference-relevant language model (P4LM) that recommends items to users while putting emphasis on explaining item characteristics and their relevance. P4LM uses the embedding space representation of a user's preferences to generate compelling responses that are factually-grounded and relevant w.r.t. the user's preferences. Moreover, we develop a joint reward function that measures precision, appeal, and personalization, which we use as AI-based feedback in a reinforcement learning-based language model framework. Using the MovieLens 25M dataset, we demonstrate that P4LM delivers compelling, personalized movie narratives to users.

{{</citation>}}


### (92/183) How does prompt engineering affect ChatGPT performance on unsupervised entity resolution? (Khanin Sisaengsuwanchai et al., 2023)

{{<citation>}}

Khanin Sisaengsuwanchai, Navapat Nananukul, Mayank Kejriwal. (2023)  
**How does prompt engineering affect ChatGPT performance on unsupervised entity resolution?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SE, cs.AI  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.06174v1)  

---


**ABSTRACT**  
Entity Resolution (ER) is the problem of semi-automatically determining when two entities refer to the same underlying entity, with applications ranging from healthcare to e-commerce. Traditional ER solutions required considerable manual expertise, including feature engineering, as well as identification and curation of training data. In many instances, such techniques are highly dependent on the domain. With recent advent in large language models (LLMs), there is an opportunity to make ER much more seamless and domain-independent. However, it is also well known that LLMs can pose risks, and that the quality of their outputs can depend on so-called prompt engineering. Unfortunately, a systematic experimental study on the effects of different prompting methods for addressing ER, using LLMs like ChatGPT, has been lacking thus far. This paper aims to address this gap by conducting such a study. Although preliminary in nature, our results show that prompting can significantly affect the quality of ER, although it affects some metrics more than others, and can also be dataset dependent.

{{</citation>}}


### (93/183) Predictable Artificial Intelligence (Lexin Zhou et al., 2023)

{{<citation>}}

Lexin Zhou, Pablo A. Moreno-Casares, Fernando Martínez-Plumed, John Burden, Ryan Burnell, Lucy Cheke, Cèsar Ferri, Alexandru Marcoci, Behzad Mehrbakhsh, Yael Moros-Daval, Seán Ó hÉigeartaigh, Danaja Rutar, Wout Schellaert, Konstantinos Voudouris, José Hernández-Orallo. (2023)  
**Predictable Artificial Intelligence**  

---
Primary Category: cs.AI  
Categories: ACM-class: I-2, cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.06167v1)  

---


**ABSTRACT**  
We introduce the fundamental ideas and challenges of Predictable AI, a nascent research area that explores the ways in which we can anticipate key indicators of present and future AI ecosystems. We argue that achieving predictability is crucial for fostering trust, liability, control, alignment and safety of AI ecosystems, and thus should be prioritised over performance. While distinctive from other areas of technical and non-technical AI research, the questions, hypotheses and challenges relevant to Predictable AI were yet to be clearly described. This paper aims to elucidate them, calls for identifying paths towards AI predictability and outlines the potential impact of this emergent field.

{{</citation>}}


### (94/183) OptiMUS: Optimization Modeling Using mip Solvers and large language models (Ali AhmadiTeshnizi et al., 2023)

{{<citation>}}

Ali AhmadiTeshnizi, Wenzhi Gao, Madeleine Udell. (2023)  
**OptiMUS: Optimization Modeling Using mip Solvers and large language models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.06116v1)  

---


**ABSTRACT**  
Optimization problems are pervasive across various sectors, from manufacturing and distribution to healthcare. However, most such problems are still solved heuristically by hand rather than optimally by state-of-the-art solvers, as the expertise required to formulate and solve these problems limits the widespread adoption of optimization tools and techniques. We introduce OptiMUS, a Large Language Model (LLM)-based agent designed to formulate and solve MILP problems from their natural language descriptions. OptiMUS is capable of developing mathematical models, writing and debugging solver code, developing tests, and checking the validity of generated solutions. To benchmark our agent, we present NLP4LP, a novel dataset of linear programming (LP) and mixed integer linear programming (MILP) problems. Our experiments demonstrate that OptiMUS is able to solve 67\% more problems compared to a basic LLM prompting strategy. OptiMUS code and NLP4LP dataset are available at \href{https://github.com/teshnizi/OptiMUS}{https://github.com/teshnizi/OptiMUS}

{{</citation>}}


### (95/183) AI Systems of Concern (Kayla Matteucci et al., 2023)

{{<citation>}}

Kayla Matteucci, Shahar Avin, Fazl Barez, Seán Ó hÉigeartaigh. (2023)  
**AI Systems of Concern**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05876v1)  

---


**ABSTRACT**  
Concerns around future dangers from advanced AI often centre on systems hypothesised to have intrinsic characteristics such as agent-like behaviour, strategic awareness, and long-range planning. We label this cluster of characteristics as "Property X". Most present AI systems are low in "Property X"; however, in the absence of deliberate steering, current research directions may rapidly lead to the emergence of highly capable AI systems that are also high in "Property X". We argue that "Property X" characteristics are intrinsically dangerous, and when combined with greater capabilities will result in AI systems for which safety and control is difficult to guarantee. Drawing on several scholars' alternative frameworks for possible AI research trajectories, we argue that most of the proposed benefits of advanced AI can be obtained by systems designed to minimise this property. We then propose indicators and governance interventions to identify and limit the development of systems with risky "Property X" characteristics.

{{</citation>}}


### (96/183) Dynamic value alignment through preference aggregation of multiple objectives (Marcin Korecki et al., 2023)

{{<citation>}}

Marcin Korecki, Damian Dailisan, Cesare Carissimo. (2023)  
**Dynamic value alignment through preference aggregation of multiple objectives**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-SY, cs.AI, eess-SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05871v1)  

---


**ABSTRACT**  
The development of ethical AI systems is currently geared toward setting objective functions that align with human objectives. However, finding such functions remains a research challenge, while in RL, setting rewards by hand is a fairly standard approach. We present a methodology for dynamic value alignment, where the values that are to be aligned with are dynamically changing, using a multiple-objective approach. We apply this approach to extend Deep $Q$-Learning to accommodate multiple objectives and evaluate this method on a simplified two-leg intersection controlled by a switching agent.Our approach dynamically accommodates the preferences of drivers on the system and achieves better overall performance across three metrics (speeds, stops, and waits) while integrating objectives that have competing or conflicting actions.

{{</citation>}}


### (97/183) Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis (Haoyu Zhang et al., 2023)

{{<citation>}}

Haoyu Zhang, Yu Wang, Guanghao Yin, Kejun Liu, Yuanyuan Liu, Tianshu Yu. (2023)  
**Learning Language-guided Adaptive Hyper-modality Representation for Multimodal Sentiment Analysis**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs-MM, cs.AI  
Keywords: Sentiment Analysis, Transformer  
[Paper Link](http://arxiv.org/abs/2310.05804v1)  

---


**ABSTRACT**  
Though Multimodal Sentiment Analysis (MSA) proves effective by utilizing rich information from multiple sources (e.g., language, video, and audio), the potential sentiment-irrelevant and conflicting information across modalities may hinder the performance from being further improved. To alleviate this, we present Adaptive Language-guided Multimodal Transformer (ALMT), which incorporates an Adaptive Hyper-modality Learning (AHL) module to learn an irrelevance/conflict-suppressing representation from visual and audio features under the guidance of language features at different scales. With the obtained hyper-modality representation, the model can obtain a complementary and joint representation through multimodal fusion for effective MSA. In practice, ALMT achieves state-of-the-art performance on several popular datasets (e.g., MOSI, MOSEI and CH-SIMS) and an abundance of ablation demonstrates the validity and necessity of our irrelevance/conflict suppression mechanism.

{{</citation>}}


### (98/183) A Review of the Ethics of Artificial Intelligence and its Applications in the United States (Esther Taiwo et al., 2023)

{{<citation>}}

Esther Taiwo, Ahmed Akinsola, Edward Tella, Kolade Makinde, Mayowa Akinwande. (2023)  
**A Review of the Ethics of Artificial Intelligence and its Applications in the United States**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05751v1)  

---


**ABSTRACT**  
This study is focused on the ethics of Artificial Intelligence and its application in the United States, the paper highlights the impact AI has in every sector of the US economy and multiple facets of the technological space and the resultant effect on entities spanning businesses, government, academia, and civil society. There is a need for ethical considerations as these entities are beginning to depend on AI for delivering various crucial tasks, which immensely influence their operations, decision-making, and interactions with each other. The adoption of ethical principles, guidelines, and standards of work is therefore required throughout the entire process of AI development, deployment, and usage to ensure responsible and ethical AI practices. Our discussion explores eleven fundamental 'ethical principles' structured as overarching themes. These encompass Transparency, Justice, Fairness, Equity, Non- Maleficence, Responsibility, Accountability, Privacy, Beneficence, Freedom, Autonomy, Trust, Dignity, Sustainability, and Solidarity. These principles collectively serve as a guiding framework, directing the ethical path for the responsible development, deployment, and utilization of artificial intelligence (AI) technologies across diverse sectors and entities within the United States. The paper also discusses the revolutionary impact of AI applications, such as Machine Learning, and explores various approaches used to implement AI ethics. This examination is crucial to address the growing concerns surrounding the inherent risks associated with the widespread use of artificial intelligence.

{{</citation>}}


### (99/183) Abstractive Summarization of Large Document Collections Using GPT (Sengjie Liu et al., 2023)

{{<citation>}}

Sengjie Liu, Christopher G. Healey. (2023)  
**Abstractive Summarization of Large Document Collections Using GPT**  

---
Primary Category: cs.AI  
Categories: H-3-1; I-2-7; I-3-3, cs-AI, cs.AI  
Keywords: GPT, Summarization  
[Paper Link](http://arxiv.org/abs/2310.05690v1)  

---


**ABSTRACT**  
This paper proposes a method of abstractive summarization designed to scale to document collections instead of individual documents. Our approach applies a combination of semantic clustering, document size reduction within topic clusters, semantic chunking of a cluster's documents, GPT-based summarization and concatenation, and a combined sentiment and text visualization of each topic to support exploratory data analysis. Statistical comparison of our results to existing state-of-the-art systems BART, BRIO, PEGASUS, and MoCa using ROGUE summary scores showed statistically equivalent performance with BART and PEGASUS on the CNN/Daily Mail test dataset, and with BART on the Gigaword test dataset. This finding is promising since we view document collection summarization as more challenging than individual document summarization. We conclude with a discussion of how issues of scale are

{{</citation>}}


### (100/183) Automated Argument Generation from Legal Facts (Oscar Tuvey et al., 2023)

{{<citation>}}

Oscar Tuvey, Procheta Sen. (2023)  
**Automated Argument Generation from Legal Facts**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Legal  
[Paper Link](http://arxiv.org/abs/2310.05680v2)  

---


**ABSTRACT**  
The count of pending cases has shown an exponential rise across nations (e.g., with more than 10 million pending cases in India alone). The main issue lies in the fact that the number of cases submitted to the law system is far greater than the available number of legal professionals present in a country. Given this worldwide context, the utilization of AI technology has gained paramount importance to enhance the efficiency and speed of legal procedures. In this study we partcularly focus on helping legal professionals in the process of analyzing a legal case. Our specific investigation delves into harnessing the generative capabilities of open-sourced large language models to create arguments derived from the facts present in legal cases. Experimental results show that the generated arguments from the best performing method have on average 63% overlap with the benchmark set gold standard annotations.

{{</citation>}}


### (101/183) STREAM: Social data and knowledge collective intelligence platform for TRaining Ethical AI Models (Yuwei Wang et al., 2023)

{{<citation>}}

Yuwei Wang, Enmeng Lu, Zizhe Ruan, Yao Liang, Yi Zeng. (2023)  
**STREAM: Social data and knowledge collective intelligence platform for TRaining Ethical AI Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05563v1)  

---


**ABSTRACT**  
This paper presents Social data and knowledge collective intelligence platform for TRaining Ethical AI Models (STREAM) to address the challenge of aligning AI models with human moral values, and to provide ethics datasets and knowledge bases to help promote AI models "follow good advice as naturally as a stream follows its course". By creating a comprehensive and representative platform that accurately mirrors the moral judgments of diverse groups including humans and AIs, we hope to effectively portray cultural and group variations, and capture the dynamic evolution of moral judgments over time, which in turn will facilitate the Establishment, Evaluation, Embedding, Embodiment, Ensemble, and Evolvement (6Es) of the moral capabilities of AI models. Currently, STREAM has already furnished a comprehensive collection of ethical scenarios, and amassed substantial moral judgment data annotated by volunteers and various popular Large Language Models (LLMs), collectively portraying the moral preferences and performances of both humans and AIs across a range of moral contexts. This paper will outline the current structure and construction of STREAM, explore its potential applications, and discuss its future prospects.

{{</citation>}}


### (102/183) Integrating Graphs with Large Language Models: Methods and Prospects (Shirui Pan et al., 2023)

{{<citation>}}

Shirui Pan, Yizhen Zheng, Yixin Liu. (2023)  
**Integrating Graphs with Large Language Models: Methods and Prospects**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05499v1)  

---


**ABSTRACT**  
Large language models (LLMs) such as GPT-4 have emerged as frontrunners, showcasing unparalleled prowess in diverse applications, including answering queries, code generation, and more. Parallelly, graph-structured data, an intrinsic data type, is pervasive in real-world scenarios. Merging the capabilities of LLMs with graph-structured data has been a topic of keen interest. This paper bifurcates such integrations into two predominant categories. The first leverages LLMs for graph learning, where LLMs can not only augment existing graph algorithms but also stand as prediction models for various graph tasks. Conversely, the second category underscores the pivotal role of graphs in advancing LLMs. Mirroring human cognition, we solve complex tasks by adopting graphs in either reasoning or collaboration. Integrating with such structures can significantly boost the performance of LLMs in various complicated tasks. We also discuss and propose open questions for integrating LLMs with graph-structured data for the future direction of the field.

{{</citation>}}


### (103/183) Deep Optimal Timing Strategies for Time Series (Chen Pan et al., 2023)

{{<citation>}}

Chen Pan, Fan Zhou, Xuanwei Hu, Xinxin Zhu, Wenxin Ning, Zi Zhuang, Siqiao Xue, James Zhang, Yunhua Hu. (2023)  
**Deep Optimal Timing Strategies for Time Series**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CE, cs.AI  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.05479v1)  

---


**ABSTRACT**  
Deciding the best future execution time is a critical task in many business activities while evolving time series forecasting, and optimal timing strategy provides such a solution, which is driven by observed data. This solution has plenty of valuable applications to reduce the operation costs. In this paper, we propose a mechanism that combines a probabilistic time series forecasting task and an optimal timing decision task as a first systematic attempt to tackle these practical problems with both solid theoretical foundation and real-world flexibility. Specifically, it generates the future paths of the underlying time series via probabilistic forecasting algorithms, which does not need a sophisticated mathematical dynamic model relying on strong prior knowledge as most other common practices. In order to find the optimal execution time, we formulate the decision task as an optimal stopping problem, and employ a recurrent neural network structure (RNN) to approximate the optimal times. Github repository: \url{github.com/ChenPopper/optimal_timing_TSF}.

{{</citation>}}


### (104/183) Explaining the Complex Task Reasoning of Large Language Models with Template-Content Structure (Haotong Yang et al., 2023)

{{<citation>}}

Haotong Yang, Fanxu Meng, Zhouchen Lin, Muhan Zhang. (2023)  
**Explaining the Complex Task Reasoning of Large Language Models with Template-Content Structure**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.05452v1)  

---


**ABSTRACT**  
The continuous evolution of pre-trained large language models with ever-growing parameters and corpus sizes has augmented their capacity to solve complex tasks. This ability, which obviates the necessity for task-specific training or fine-tuning, relies on providing the model with a language description or some task exemplars -- referred to the prompt -- that guide the desired autoregressive generation. Despite the remarkable success, the underlying mechanisms that facilitate such exceptional generalization abilities remain an open question. In this paper, we present a novel framework that formally conceptualizes answer generation for complex natural language tasks as a hierarchical ``template-content'' structure. According to our modeling, there exist pre-trained models that can automatically decompose tasks into constituent steps during autoregressive generation, through language modeling on a sufficiently large corpus, thereby solving them. Our framework offers an explanatory tool for the complex reasoning abilities of large language models from the perspective of modeling autoregressive generation tasks. Our experiments show that practical models exhibit different behaviors for ``template'' and ``content'' providing support for our modeling.

{{</citation>}}


### (105/183) Replication of Multi-agent Reinforcement Learning for the 'Hide and Seek' Problem (Haider Kamal et al., 2023)

{{<citation>}}

Haider Kamal, Muaz A. Niazi, Hammad Afzal. (2023)  
**Replication of Multi-agent Reinforcement Learning for the 'Hide and Seek' Problem**  

---
Primary Category: cs.AI  
Categories: 68T42, 93A16, 68T05, 68T07, I-2-11; I-6-5; I-6-6; I-2-6, cs-AI, cs-LG, cs-MA, cs-RO, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05430v1)  

---


**ABSTRACT**  
Reinforcement learning generates policies based on reward functions and hyperparameters. Slight changes in these can significantly affect results. The lack of documentation and reproducibility in Reinforcement learning research makes it difficult to replicate once-deduced strategies. While previous research has identified strategies using grounded maneuvers, there is limited work in more complex environments. The agents in this study are simulated similarly to Open Al's hider and seek agents, in addition to a flying mechanism, enhancing their mobility, and expanding their range of possible actions and strategies. This added functionality improves the Hider agents to develop a chasing strategy from approximately 2 million steps to 1.6 million steps and hiders

{{</citation>}}


### (106/183) Causal Reasoning through Two Layers of Cognition for Improving Generalization in Visual Question Answering (Trang Nguyen et al., 2023)

{{<citation>}}

Trang Nguyen, Naoaki Okazaki. (2023)  
**Causal Reasoning through Two Layers of Cognition for Improving Generalization in Visual Question Answering**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: QA, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.05410v1)  

---


**ABSTRACT**  
Generalization in Visual Question Answering (VQA) requires models to answer questions about images with contexts beyond the training distribution. Existing attempts primarily refine unimodal aspects, overlooking enhancements in multimodal aspects. Besides, diverse interpretations of the input lead to various modes of answer generation, highlighting the role of causal reasoning between interpreting and answering steps in VQA. Through this lens, we propose Cognitive pathways VQA (CopVQA) improving the multimodal predictions by emphasizing causal reasoning factors. CopVQA first operates a pool of pathways that capture diverse causal reasoning flows through interpreting and answering stages. Mirroring human cognition, we decompose the responsibility of each stage into distinct experts and a cognition-enabled component (CC). The two CCs strategically execute one expert for each stage at a time. Finally, we prioritize answer predictions governed by pathways involving both CCs while disregarding answers produced by either CC, thereby emphasizing causal reasoning and supporting generalization. Our experiments on real-life and medical data consistently verify that CopVQA improves VQA performance and generalization across baselines and domains. Notably, CopVQA achieves a new state-of-the-art (SOTA) on PathVQA dataset and comparable accuracy to the current SOTA on VQA-CPv2, VQAv2, and VQA RAD, with one-fourth of the model size.

{{</citation>}}


## cs.LG (38)



### (107/183) Memory-Consistent Neural Networks for Imitation Learning (Kaustubh Sridhar et al., 2023)

{{<citation>}}

Kaustubh Sridhar, Souradeep Dutta, Dinesh Jayaraman, James Weimer, Insup Lee. (2023)  
**Memory-Consistent Neural Networks for Imitation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.06171v1)  

---


**ABSTRACT**  
Imitation learning considerably simplifies policy synthesis compared to alternative approaches by exploiting access to expert demonstrations. For such imitation policies, errors away from the training samples are particularly critical. Even rare slip-ups in the policy action outputs can compound quickly over time, since they lead to unfamiliar future states where the policy is still more likely to err, eventually causing task failures. We revisit simple supervised ``behavior cloning'' for conveniently training the policy from nothing more than pre-recorded demonstrations, but carefully design the model class to counter the compounding error phenomenon. Our ``memory-consistent neural network'' (MCNN) outputs are hard-constrained to stay within clearly specified permissible regions anchored to prototypical ``memory'' training samples. We provide a guaranteed upper bound for the sub-optimality gap induced by MCNN policies. Using MCNNs on 9 imitation learning tasks, with MLP, Transformer, and Diffusion backbones, spanning dexterous robotic manipulation and driving, proprioceptive inputs and visual inputs, and varying sizes and types of demonstration data, we find large and consistent gains in performance, validating that MCNNs are better-suited than vanilla deep neural networks for imitation learning applications. Website: https://sites.google.com/view/mcnn-imitation

{{</citation>}}


### (108/183) Mitigating Simplicity Bias in Deep Learning for Improved OOD Generalization and Robustness (Bhavya Vasudeva et al., 2023)

{{<citation>}}

Bhavya Vasudeva, Kameron Shahabi, Vatsal Sharan. (2023)  
**Mitigating Simplicity Bias in Deep Learning for Improved OOD Generalization and Robustness**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.06161v1)  

---


**ABSTRACT**  
Neural networks (NNs) are known to exhibit simplicity bias where they tend to prefer learning 'simple' features over more 'complex' ones, even when the latter may be more informative. Simplicity bias can lead to the model making biased predictions which have poor out-of-distribution (OOD) generalization. To address this, we propose a framework that encourages the model to use a more diverse set of features to make predictions. We first train a simple model, and then regularize the conditional mutual information with respect to it to obtain the final model. We demonstrate the effectiveness of this framework in various problem settings and real-world applications, showing that it effectively addresses simplicity bias and leads to more features being used, enhances OOD generalization, and improves subgroup robustness and fairness. We complement these results with theoretical analyses of the effect of the regularization and its OOD generalization properties.

{{</citation>}}


### (109/183) Latent Diffusion Model for DNA Sequence Generation (Zehui Li et al., 2023)

{{<citation>}}

Zehui Li, Yuhao Ni, Tim August B. Huygelen, Akashaditya Das, Guoxuan Xia, Guy-Bart Stan, Yiren Zhao. (2023)  
**Latent Diffusion Model for DNA Sequence Generation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.06150v1)  

---


**ABSTRACT**  
The harnessing of machine learning, especially deep generative models, has opened up promising avenues in the field of synthetic DNA sequence generation. Whilst Generative Adversarial Networks (GANs) have gained traction for this application, they often face issues such as limited sample diversity and mode collapse. On the other hand, Diffusion Models are a promising new class of generative models that are not burdened with these problems, enabling them to reach the state-of-the-art in domains such as image generation. In light of this, we propose a novel latent diffusion model, DiscDiff, tailored for discrete DNA sequence generation. By simply embedding discrete DNA sequences into a continuous latent space using an autoencoder, we are able to leverage the powerful generative abilities of continuous diffusion models for the generation of discrete data. Additionally, we introduce Fr\'echet Reconstruction Distance (FReD) as a new metric to measure the sample quality of DNA sequence generations. Our DiscDiff model demonstrates an ability to generate synthetic DNA sequences that align closely with real DNA in terms of Motif Distribution, Latent Embedding Distribution (FReD), and Chromatin Profiles. Additionally, we contribute a comprehensive cross-species dataset of 150K unique promoter-gene sequences from 15 species, enriching resources for future generative modelling in genomics. We will make our code public upon publication.

{{</citation>}}


### (110/183) Reinforcement Learning in the Era of LLMs: What is Essential? What is needed? An RL Perspective on RLHF, Prompting, and Beyond (Hao Sun, 2023)

{{<citation>}}

Hao Sun. (2023)  
**Reinforcement Learning in the Era of LLMs: What is Essential? What is needed? An RL Perspective on RLHF, Prompting, and Beyond**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: ChatGPT, GPT, GPT-4, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.06147v1)  

---


**ABSTRACT**  
Recent advancements in Large Language Models (LLMs) have garnered wide attention and led to successful products such as ChatGPT and GPT-4. Their proficiency in adhering to instructions and delivering harmless, helpful, and honest (3H) responses can largely be attributed to the technique of Reinforcement Learning from Human Feedback (RLHF). In this paper, we aim to link the research in conventional RL to RL techniques used in LLM research. Demystify this technique by discussing why, when, and how RL excels. Furthermore, we explore potential future avenues that could either benefit from or contribute to RLHF research.   Highlighted Takeaways:   1. RLHF is Online Inverse RL with Offline Demonstration Data.   2. RLHF $>$ SFT because Imitation Learning (and Inverse RL) $>$ Behavior Cloning (BC) by alleviating the problem of compounding error.   3. The RM step in RLHF generates a proxy of the expensive human feedback, such an insight can be generalized to other LLM tasks such as prompting evaluation and optimization where feedback is also expensive.   4. The policy learning in RLHF is more challenging than conventional problems studied in IRL due to their high action dimensionality and feedback sparsity.   5. The main superiority of PPO over off-policy value-based methods is its stability gained from (almost) on-policy data and conservative policy updates.

{{</citation>}}


### (111/183) Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis (Zezhi Shao et al., 2023)

{{<citation>}}

Zezhi Shao, Fei Wang, Yongjun Xu, Wei Wei, Chengqing Yu, Zhao Zhang, Di Yao, Guangyin Jin, Xin Cao, Gao Cong, Christian S. Jensen, Xueqi Cheng. (2023)  
**Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.06119v1)  

---


**ABSTRACT**  
Multivariate Time Series (MTS) widely exists in real-word complex systems, such as traffic and energy systems, making their forecasting crucial for understanding and influencing these systems. Recently, deep learning-based approaches have gained much popularity for effectively modeling temporal and spatial dependencies in MTS, specifically in Long-term Time Series Forecasting (LTSF) and Spatial-Temporal Forecasting (STF). However, the fair benchmarking issue and the choice of technical approaches have been hotly debated in related work. Such controversies significantly hinder our understanding of progress in this field. Thus, this paper aims to address these controversies to present insights into advancements achieved. To resolve benchmarking issues, we introduce BasicTS, a benchmark designed for fair comparisons in MTS forecasting. BasicTS establishes a unified training pipeline and reasonable evaluation settings, enabling an unbiased evaluation of over 30 popular MTS forecasting models on more than 18 datasets. Furthermore, we highlight the heterogeneity among MTS datasets and classify them based on temporal and spatial characteristics. We further prove that neglecting heterogeneity is the primary reason for generating controversies in technical approaches. Moreover, based on the proposed BasicTS and rich heterogeneous MTS datasets, we conduct an exhaustive and reproducible performance and efficiency comparison of popular models, providing insights for researchers in selecting and designing MTS forecasting models.

{{</citation>}}


### (112/183) Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models (Huaixiu Steven Zheng et al., 2023)

{{<citation>}}

Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen, Heng-Tze Cheng, Ed H. Chi, Quoc V Le, Denny Zhou. (2023)  
**Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model, PaLM, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.06117v1)  

---


**ABSTRACT**  
We present Step-Back Prompting, a simple prompting technique that enables LLMs to do abstractions to derive high-level concepts and first principles from instances containing specific details. Using the concepts and principles to guide the reasoning steps, LLMs significantly improve their abilities in following a correct reasoning path towards the solution. We conduct experiments of Step-Back Prompting with PaLM-2L models and observe substantial performance gains on a wide range of challenging reasoning-intensive tasks including STEM, Knowledge QA, and Multi-Hop Reasoning. For instance, Step-Back Prompting improves PaLM-2L performance on MMLU Physics and Chemistry by 7% and 11%, TimeQA by 27%, and MuSiQue by 7%.

{{</citation>}}


### (113/183) When is Agnostic Reinforcement Learning Statistically Tractable? (Zeyu Jia et al., 2023)

{{<citation>}}

Zeyu Jia, Gene Li, Alexander Rakhlin, Ayush Sekhari, Nathan Srebro. (2023)  
**When is Agnostic Reinforcement Learning Statistically Tractable?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-ST, stat-ML, stat-TH  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.06113v1)  

---


**ABSTRACT**  
We study the problem of agnostic PAC reinforcement learning (RL): given a policy class $\Pi$, how many rounds of interaction with an unknown MDP (with a potentially large state and action space) are required to learn an $\epsilon$-suboptimal policy with respect to $\Pi$? Towards that end, we introduce a new complexity measure, called the \emph{spanning capacity}, that depends solely on the set $\Pi$ and is independent of the MDP dynamics. With a generative model, we show that for any policy class $\Pi$, bounded spanning capacity characterizes PAC learnability. However, for online RL, the situation is more subtle. We show there exists a policy class $\Pi$ with a bounded spanning capacity that requires a superpolynomial number of samples to learn. This reveals a surprising separation for agnostic learnability between generative access and online access models (as well as between deterministic/stochastic MDPs under online access). On the positive side, we identify an additional \emph{sunflower} structure, which in conjunction with bounded spanning capacity enables statistically efficient online RL via a new algorithm called POPLER, which takes inspiration from classical importance sampling methods as well as techniques for reachable-state identification and policy evaluation in reward-free exploration.

{{</citation>}}


### (114/183) Transformers and Large Language Models for Chemistry and Drug Discovery (Andres M Bran et al., 2023)

{{<citation>}}

Andres M Bran, Philippe Schwaller. (2023)  
**Transformers and Large Language Models for Chemistry and Drug Discovery**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-chem-ph  
Keywords: Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.06083v1)  

---


**ABSTRACT**  
Language modeling has seen impressive progress over the last years, mainly prompted by the invention of the Transformer architecture, sparking a revolution in many fields of machine learning, with breakthroughs in chemistry and biology. In this chapter, we explore how analogies between chemical and natural language have inspired the use of Transformers to tackle important bottlenecks in the drug discovery process, such as retrosynthetic planning and chemical space exploration. The revolution started with models able to perform particular tasks with a single type of data, like linearised molecular graphs, which then evolved to include other types of data, like spectra from analytical instruments, synthesis actions, and human language. A new trend leverages recent developments in large language models, giving rise to a wave of models capable of solving generic tasks in chemistry, all facilitated by the flexibility of natural language. As we continue to explore and harness these capabilities, we can look forward to a future where machine learning plays an even more integral role in accelerating scientific discovery.

{{</citation>}}


### (115/183) Early Warning via tipping-preserving latent stochastic dynamical system and meta label correcting (Peng Zhang et al., 2023)

{{<citation>}}

Peng Zhang, Ting Gao, Jin Guo, Jinqiao Duan. (2023)  
**Early Warning via tipping-preserving latent stochastic dynamical system and meta label correcting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-DS  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.06059v1)  

---


**ABSTRACT**  
Early warning for epilepsy patients is crucial for their safety and well-being, in terms of preventing or minimizing the severity of seizures. Through the patients' EEG data, we propose a meta learning framework for improving prediction on early ictal signals. To better utilize the meta label corrector method, we fuse the information from both the real data and the augmented data from the latent Stochastic differential equation(SDE). Besides, we also optimally select the latent dynamical system via distribution of transition time between real data and that from the latent SDE. In this way, the extracted tipping dynamical feature is also integrated into the meta network to better label the noisy data. To validate our method, LSTM is implemented as the baseline model. We conduct a series of experiments to predict seizure in various long-term window from 1-2 seconds input data and find surprisingly increment of prediction accuracy.

{{</citation>}}


### (116/183) Knowledge Distillation for Anomaly Detection (Adrian Alan Pol et al., 2023)

{{<citation>}}

Adrian Alan Pol, Ekaterina Govorkova, Sonja Gronroos, Nadezda Chernyavskaya, Philip Harris, Maurizio Pierini, Isobel Ojalvo, Peter Elmer. (2023)  
**Knowledge Distillation for Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2310.06047v1)  

---


**ABSTRACT**  
Unsupervised deep learning techniques are widely used to identify anomalous behaviour. The performance of such methods is a product of the amount of training data and the model size. However, the size is often a limiting factor for the deployment on resource-constrained devices. We present a novel procedure based on knowledge distillation for compressing an unsupervised anomaly detection model into a supervised deployable one and we suggest a set of techniques to improve the detection sensitivity. Compressed models perform comparably to their larger counterparts while significantly reducing the size and memory footprint.

{{</citation>}}


### (117/183) TAIL: Task-specific Adapters for Imitation Learning with Large Pretrained Models (Zuxin Liu et al., 2023)

{{<citation>}}

Zuxin Liu, Jesse Zhang, Kavosh Asadi, Yao Liu, Ding Zhao, Shoham Sabach, Rasool Fakoor. (2023)  
**TAIL: Task-specific Adapters for Imitation Learning with Large Pretrained Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05905v1)  

---


**ABSTRACT**  
The full potential of large pretrained models remains largely untapped in control domains like robotics. This is mainly because of the scarcity of data and the computational challenges associated with training or fine-tuning these large models for such applications. Prior work mainly emphasizes effective pretraining of large models for decision-making, with little exploration into how to perform data-efficient continual adaptation of these models for new tasks. Recognizing these constraints, we introduce TAIL (Task-specific Adapters for Imitation Learning), a framework for efficient adaptation to new control tasks. Inspired by recent advancements in parameter-efficient fine-tuning in language domains, we explore efficient fine-tuning techniques -- e.g., Bottleneck Adapters, P-Tuning, and Low-Rank Adaptation (LoRA) -- in TAIL to adapt large pretrained models for new tasks with limited demonstration data. Our extensive experiments in large-scale language-conditioned manipulation tasks comparing prevalent parameter-efficient fine-tuning techniques and adaptation baselines suggest that TAIL with LoRA can achieve the best post-adaptation performance with only 1\% of the trainable parameters of full fine-tuning, while avoiding catastrophic forgetting and preserving adaptation plasticity in continual learning settings.

{{</citation>}}


### (118/183) Lion Secretly Solves Constrained Optimization: As Lyapunov Predicts (Lizhang Chen et al., 2023)

{{<citation>}}

Lizhang Chen, Bo Liu, Kaizhao Liang, Qiang Liu. (2023)  
**Lion Secretly Solves Constrained Optimization: As Lyapunov Predicts**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-OC, stat-AP, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05898v1)  

---


**ABSTRACT**  
Lion (Evolved Sign Momentum), a new optimizer discovered through program search, has shown promising results in training large AI models. It performs comparably or favorably to AdamW but with greater memory efficiency. As we can expect from the results of a random search program, Lion incorporates elements from several existing algorithms, including signed momentum, decoupled weight decay, Polak, and Nesterov momentum, but does not fit into any existing category of theoretically grounded optimizers. Thus, even though Lion appears to perform well as a general-purpose optimizer for a wide range of tasks, its theoretical basis remains uncertain. This lack of theoretical clarity limits opportunities to further enhance and expand Lion's efficacy.   This work aims to demystify Lion. Based on both continuous-time and discrete-time analysis, we demonstrate that Lion is a theoretically novel and principled approach for minimizing a general loss function $f(x)$ while enforcing a bound constraint $\|x\|_\infty \leq 1/\lambda$. Lion achieves this through the incorporation of decoupled weight decay, where $\lambda$ represents the weight decay coefficient. Our analysis is made possible by the development of a new Lyapunov function for the Lion updates. It applies to a broader family of Lion-$\kappa$ algorithms, where the $\text{sign}(\cdot)$ operator in Lion is replaced by the subgradient of a convex function $\kappa$, leading to the solution of a general composite optimization problem of $\min_x f(x) + \kappa^*(x)$. Our findings provide valuable insights into the dynamics of Lion and pave the way for further improvements and extensions of Lion-related algorithms.

{{</citation>}}


### (119/183) A Meta-Learning Perspective on Transformers for Causal Language Modeling (Xinbo Wu et al., 2023)

{{<citation>}}

Xinbo Wu, Lav R. Varshney. (2023)  
**A Meta-Learning Perspective on Transformers for Causal Language Modeling**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.05884v1)  

---


**ABSTRACT**  
The Transformer architecture has become prominent in developing large causal language models. However, mechanisms to explain its capabilities are not well understood. Focused on the training process, here we establish a meta-learning view of the Transformer architecture when trained for the causal language modeling task, by explicating an inner optimization process that may happen within the Transformer. Further, from within the inner optimization, we discover and theoretically analyze a special characteristic of the norms of learned token representations within Transformer-based causal language models. Our analysis is supported by experiments conducted on pre-trained large language models and real-world data.

{{</citation>}}


### (120/183) HyperAttention: Long-context Attention in Near-Linear Time (Insu Han et al., 2023)

{{<citation>}}

Insu Han, Rajesh Jayaram, Amin Karbasi, Vahab Mirrokni, David P. Woodruff, Amir Zandieh. (2023)  
**HyperAttention: Long-context Attention in Near-Linear Time**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, GLM, Language Model  
[Paper Link](http://arxiv.org/abs/2310.05869v2)  

---


**ABSTRACT**  
We present an approximate attention mechanism named HyperAttention to address the computational challenges posed by the growing complexity of long contexts used in Large Language Models (LLMs). Recent work suggests that in the worst-case scenario, quadratic time is necessary unless the entries of the attention matrix are bounded or the matrix has low stable rank. We introduce two parameters which measure: (1) the max column norm in the normalized attention matrix, and (2) the ratio of row norms in the unnormalized attention matrix after detecting and removing large entries. We use these fine-grained parameters to capture the hardness of the problem. Despite previous lower bounds, we are able to achieve a linear time sampling algorithm even when the matrix has unbounded entries or a large stable rank, provided the above parameters are small. HyperAttention features a modular design that easily accommodates integration of other fast low-level implementations, particularly FlashAttention. Empirically, employing Locality Sensitive Hashing (LSH) to identify large entries, HyperAttention outperforms existing methods, giving significant speed improvements compared to state-of-the-art solutions like FlashAttention. We validate the empirical performance of HyperAttention on a variety of different long-context length datasets. For example, HyperAttention makes the inference time of ChatGLM2 50\% faster on 32k context length while perplexity increases from 5.6 to 6.3. On larger context length, e.g., 131k, with causal masking, HyperAttention offers 5-fold speedup on a single attention layer.

{{</citation>}}


### (121/183) Robust Angular Synchronization via Directed Graph Neural Networks (Yixuan He et al., 2023)

{{<citation>}}

Yixuan He, Gesine Reinert, David Wipf, Mihai Cucuringu. (2023)  
**Robust Angular Synchronization via Directed Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.05842v1)  

---


**ABSTRACT**  
The angular synchronization problem aims to accurately estimate (up to a constant additive phase) a set of unknown angles $\theta_1, \dots, \theta_n\in[0, 2\pi)$ from $m$ noisy measurements of their offsets $\theta_i-\theta_j \;\mbox{mod} \; 2\pi.$ Applications include, for example, sensor network localization, phase retrieval, and distributed clock synchronization. An extension of the problem to the heterogeneous setting (dubbed $k$-synchronization) is to estimate $k$ groups of angles simultaneously, given noisy observations (with unknown group assignment) from each group. Existing methods for angular synchronization usually perform poorly in high-noise regimes, which are common in applications. In this paper, we leverage neural networks for the angular synchronization problem, and its heterogeneous extension, by proposing GNNSync, a theoretically-grounded end-to-end trainable framework using directed graph neural networks. In addition, new loss functions are devised to encode synchronization objectives. Experimental results on extensive data sets demonstrate that GNNSync attains competitive, and often superior, performance against a comprehensive set of baselines for the angular synchronization problem and its extension, validating the robustness of GNNSync even at high noise levels.

{{</citation>}}


### (122/183) A Bias-Variance-Covariance Decomposition of Kernel Scores for Generative Models (Sebastian G. Gruber et al., 2023)

{{<citation>}}

Sebastian G. Gruber, Florian Buettner. (2023)  
**A Bias-Variance-Covariance Decomposition of Kernel Scores for Generative Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Bias, QA  
[Paper Link](http://arxiv.org/abs/2310.05833v1)  

---


**ABSTRACT**  
Generative models, like large language models, are becoming increasingly relevant in our daily lives, yet a theoretical framework to assess their generalization behavior and uncertainty does not exist. Particularly, the problem of uncertainty estimation is commonly solved in an ad-hoc manner and task dependent. For example, natural language approaches cannot be transferred to image generation. In this paper we introduce the first bias-variance-covariance decomposition for kernel scores and their associated entropy. We propose unbiased and consistent estimators for each quantity which only require generated samples but not the underlying model itself. As an application, we offer a generalization evaluation of diffusion models and discover how mode collapse of minority groups is a contrary phenomenon to overfitting. Further, we demonstrate that variance and predictive kernel entropy are viable measures of uncertainty for image, audio, and language generation. Specifically, our approach for uncertainty estimation is more predictive of performance on CoQA and TriviaQA question answering datasets than existing baselines and can also be applied to closed-source models.

{{</citation>}}


### (123/183) DiffuSeq-v2: Bridging Discrete and Continuous Text Spaces for Accelerated Seq2Seq Diffusion Models (Shansan Gong et al., 2023)

{{<citation>}}

Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, Lingpeng Kong. (2023)  
**DiffuSeq-v2: Bridging Discrete and Continuous Text Spaces for Accelerated Seq2Seq Diffusion Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: NLP, Seq2Seq  
[Paper Link](http://arxiv.org/abs/2310.05793v1)  

---


**ABSTRACT**  
Diffusion models have gained prominence in generating high-quality sequences of text. Nevertheless, current approaches predominantly represent discrete text within a continuous diffusion space, which incurs substantial computational overhead during training and results in slower sampling speeds. In this paper, we introduce a soft absorbing state that facilitates the diffusion model in learning to reconstruct discrete mutations based on the underlying Gaussian space, thereby enhancing its capacity to recover conditional signals. During the sampling phase, we employ state-of-the-art ODE solvers within the continuous space to expedite the sampling process. Comprehensive experimental evaluations reveal that our proposed method effectively accelerates the training convergence by 4x and generates samples of similar quality 800x faster, rendering it significantly closer to practical application. \footnote{The code is released at \url{https://github.com/Shark-NLP/DiffuSeq}

{{</citation>}}


### (124/183) Why Should This Article Be Deleted? Transparent Stance Detection in Multilingual Wikipedia Editor Discussions (Lucie-Aimée Kaffee et al., 2023)

{{<citation>}}

Lucie-Aimée Kaffee, Arnav Arora, Isabelle Augenstein. (2023)  
**Why Should This Article Be Deleted? Transparent Stance Detection in Multilingual Wikipedia Editor Discussions**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Multilingual, Stance Detection  
[Paper Link](http://arxiv.org/abs/2310.05779v1)  

---


**ABSTRACT**  
The moderation of content on online platforms is usually non-transparent. On Wikipedia, however, this discussion is carried out publicly and the editors are encouraged to use the content moderation policies as explanations for making moderation decisions. Currently, only a few comments explicitly mention those policies -- 20% of the English ones, but as few as 2% of the German and Turkish comments. To aid in this process of understanding how content is moderated, we construct a novel multilingual dataset of Wikipedia editor discussions along with their reasoning in three languages. The dataset contains the stances of the editors (keep, delete, merge, comment), along with the stated reason, and a content moderation policy, for each edit decision. We demonstrate that stance and corresponding reason (policy) can be predicted jointly with a high degree of accuracy, adding transparency to the decision-making process. We release both our joint prediction models and the multilingual content moderation dataset for further research on automated transparent content moderation.

{{</citation>}}


### (125/183) Rethinking Memory and Communication Cost for Efficient Large Language Model Training (Chan Wu et al., 2023)

{{<citation>}}

Chan Wu, Hanxiao Zhang, Lin Ju, Jinjing Huang, Youshao Xiao, Zhaoxin Huan, Siyuan Li, Fanzhuang Meng, Lei Liang, Xiaolu Zhang, Jun Zhou. (2023)  
**Rethinking Memory and Communication Cost for Efficient Large Language Model Training**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.06003v1)  

---


**ABSTRACT**  
As model sizes and training datasets continue to increase, large-scale model training frameworks reduce memory consumption by various sharding techniques. However, the huge communication overhead reduces the training efficiency, especially in public cloud environments with varying network bandwidths. In this paper, we rethink the impact of memory consumption and communication overhead on the training speed of large language model, and propose a memory-communication balanced \underline{Pa}rtial \underline{R}edundancy \underline{O}ptimizer (PaRO). PaRO reduces the amount and frequency of inter-group communication by grouping GPU clusters and introducing minor intra-group memory redundancy, thereby improving the training efficiency of the model. Additionally, we propose a Hierarchical Overlapping Ring (HO-Ring) communication topology to enhance communication efficiency between nodes or across switches in large model training. Our experiments demonstrate that the HO-Ring algorithm improves communication efficiency by 32.6\% compared to the traditional Ring algorithm. Compared to the baseline ZeRO, PaRO significantly improves training throughput by 1.2x-2.6x and achieves a near-linear scalability. Therefore, the PaRO strategy provides more fine-grained options for the trade-off between memory consumption and communication overhead in different training scenarios.

{{</citation>}}


### (126/183) Foundation Models Meet Visualizations: Challenges and Opportunities (Weikai Yang et al., 2023)

{{<citation>}}

Weikai Yang, Mengchen Liu, Zheng Wang, Shixia Liu. (2023)  
**Foundation Models Meet Visualizations: Challenges and Opportunities**  

---
Primary Category: cs.LG  
Categories: cs-HC, cs-LG, cs.LG  
Keywords: AI, BERT, GPT  
[Paper Link](http://arxiv.org/abs/2310.05771v1)  

---


**ABSTRACT**  
Recent studies have indicated that foundation models, such as BERT and GPT, excel in adapting to a variety of downstream tasks. This adaptability has established them as the dominant force in building artificial intelligence (AI) systems. As visualization techniques intersect with these models, a new research paradigm emerges. This paper divides these intersections into two main areas: visualizations for foundation models (VIS4FM) and foundation models for visualizations (FM4VIS). In VIS4FM, we explore the primary role of visualizations in understanding, refining, and evaluating these intricate models. This addresses the pressing need for transparency, explainability, fairness, and robustness. Conversely, within FM4VIS, we highlight how foundation models can be utilized to advance the visualization field itself. The confluence of foundation models and visualizations holds great promise, but it also comes with its own set of challenges. By highlighting these challenges and the growing opportunities, this paper seeks to provide a starting point for continued exploration in this promising avenue.

{{</citation>}}


### (127/183) Nonlinear Correct and Smooth for Semi-Supervised Learning (Yuanhang Shao et al., 2023)

{{<citation>}}

Yuanhang Shao, Xiuwen Liu. (2023)  
**Nonlinear Correct and Smooth for Semi-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.05757v1)  

---


**ABSTRACT**  
Graph-based semi-supervised learning (GSSL) has been used successfully in various applications. Existing methods leverage the graph structure and labeled samples for classification. Label Propagation (LP) and Graph Neural Networks (GNNs) both iteratively pass messages on graphs, where LP propagates node labels through edges and GNN aggregates node features from the neighborhood. Recently, combining LP and GNN has led to improved performance. However, utilizing labels and features jointly in higher-order graphs has not been explored. Therefore, we propose Nonlinear Correct and Smooth (NLCS), which improves the existing post-processing approach by incorporating non-linearity and higher-order representation into the residual propagation to handle intricate node relationships effectively. Systematic evaluations show that our method achieves remarkable average improvements of 13.71% over base prediction and 2.16% over the state-of-the-art post-processing method on six commonly used datasets. Comparisons and analyses show our method effectively utilizes labels and features jointly in higher-order graphs to resolve challenging graph relationships.

{{</citation>}}


### (128/183) Planning to Go Out-of-Distribution in Offline-to-Online Reinforcement Learning (Trevor McInroe et al., 2023)

{{<citation>}}

Trevor McInroe, Stefano V. Albrecht, Amos Storkey. (2023)  
**Planning to Go Out-of-Distribution in Offline-to-Online Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05723v1)  

---


**ABSTRACT**  
Offline pretraining with a static dataset followed by online fine-tuning (offline-to-online, or OtO) is a paradigm that is well matched to a real-world RL deployment process: in few real settings would one deploy an offline policy with no test runs and tuning. In this scenario, we aim to find the best-performing policy within a limited budget of online interactions. Previous work in the OtO setting has focused on correcting for bias introduced by the policy-constraint mechanisms of offline RL algorithms. Such constraints keep the learned policy close to the behavior policy that collected the dataset, but this unnecessarily limits policy performance if the behavior policy is far from optimal. Instead, we forgo policy constraints and frame OtO RL as an exploration problem: we must maximize the benefit of the online data-collection. We study major online RL exploration paradigms, adapting them to work well with the OtO setting. These adapted methods contribute several strong baselines. Also, we introduce an algorithm for planning to go out of distribution (PTGOOD), which targets online exploration in relatively high-reward regions of the state-action space unlikely to be visited by the behavior policy. By leveraging concepts from the Conditional Entropy Bottleneck, PTGOOD encourages data collected online to provide new information relevant to improving the final deployment policy. In that way the limited interaction budget is used effectively. We show that PTGOOD significantly improves agent returns during online fine-tuning and finds the optimal policy in as few as 10k online steps in Walker and in as few as 50k in complex control tasks like Humanoid. Also, we find that PTGOOD avoids the suboptimal policy convergence that many of our baselines exhibit in several environments.

{{</citation>}}


### (129/183) Transformer Fusion with Optimal Transport (Moritz Imfeld et al., 2023)

{{<citation>}}

Moritz Imfeld, Jacopo Graldi, Marco Giordano, Thomas Hofmann, Sotiris Anagnostidis, Sidak Pal Singh. (2023)  
**Transformer Fusion with Optimal Transport**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.05719v1)  

---


**ABSTRACT**  
Fusion is a technique for merging multiple independently-trained neural networks in order to combine their capabilities. Past attempts have been restricted to the case of fully-connected, convolutional, and residual networks. In this paper, we present a systematic approach for fusing two or more transformer-based networks exploiting Optimal Transport to (soft-)align the various architectural components. We flesh out an abstraction for layer alignment, that can generalize to arbitrary architectures -- in principle -- and we apply this to the key ingredients of Transformers such as multi-head self-attention, layer-normalization, and residual connections, and we discuss how to handle them via various ablation studies. Furthermore, our method allows the fusion of models of different sizes (heterogeneous fusion), providing a new and efficient way for compression of Transformers. The proposed approach is evaluated on both image classification tasks via Vision Transformer and natural language modeling tasks using BERT. Our approach consistently outperforms vanilla fusion, and, after a surprisingly short finetuning, also outperforms the individual converged parent models. In our analysis, we uncover intriguing insights about the significant role of soft alignment in the case of Transformers. Our results showcase the potential of fusing multiple Transformers, thus compounding their expertise, in the budding paradigm of model fusion and recombination.

{{</citation>}}


### (130/183) Imitator Learning: Achieve Out-of-the-Box Imitation Ability in Variable Environments (Xiong-Hui Chen et al., 2023)

{{<citation>}}

Xiong-Hui Chen, Junyin Ye, Hang Zhao, Yi-Chen Li, Haoran Shi, Yu-Yan Xu, Zhihao Ye, Si-Hang Yang, Anqi Huang, Kai Xu, Zongzhang Zhang, Yang Yu. (2023)  
**Imitator Learning: Achieve Out-of-the-Box Imitation Ability in Variable Environments**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.05712v1)  

---


**ABSTRACT**  
Imitation learning (IL) enables agents to mimic expert behaviors. Most previous IL techniques focus on precisely imitating one policy through mass demonstrations. However, in many applications, what humans require is the ability to perform various tasks directly through a few demonstrations of corresponding tasks, where the agent would meet many unexpected changes when deployed. In this scenario, the agent is expected to not only imitate the demonstration but also adapt to unforeseen environmental changes.   This motivates us to propose a new topic called imitator learning (ItorL), which aims to derive an imitator module that can on-the-fly reconstruct the imitation policies based on very limited expert demonstrations for different unseen tasks, without any extra adjustment. In this work, we focus on imitator learning based on only one expert demonstration. To solve ItorL, we propose Demo-Attention Actor-Critic (DAAC), which integrates IL into a reinforcement-learning paradigm that can regularize policies' behaviors in unexpected situations. Besides, for autonomous imitation policy building, we design a demonstration-based attention architecture for imitator policy that can effectively output imitated actions by adaptively tracing the suitable states in demonstrations. We develop a new navigation benchmark and a robot environment for \topic~and show that DAAC~outperforms previous imitation methods \textit{with large margins} both on seen and unseen tasks.

{{</citation>}}


### (131/183) Hierarchical Reinforcement Learning for Temporal Pattern Prediction (Faith Johnson et al., 2023)

{{<citation>}}

Faith Johnson, Kristin Dana. (2023)  
**Hierarchical Reinforcement Learning for Temporal Pattern Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05695v1)  

---


**ABSTRACT**  
In this work, we explore the use of hierarchical reinforcement learning (HRL) for the task of temporal sequence prediction. Using a combination of deep learning and HRL, we develop a stock agent to predict temporal price sequences from historical stock price data and a vehicle agent to predict steering angles from first person, dash cam images. Our results in both domains indicate that a type of HRL, called feudal reinforcement learning, provides significant improvements to training speed and stability and prediction accuracy over standard RL. A key component to this success is the multi-resolution structure that introduces both temporal and spatial abstraction into the network hierarchy.

{{</citation>}}


### (132/183) Analysis of Rainfall Variability and Water Extent of Selected Hydropower Reservoir Using Google Earth Engine (GEE): A Case Study from Two Tropical Countries, Sri Lanka and Vietnam (Punsisi Rajakaruna et al., 2023)

{{<citation>}}

Punsisi Rajakaruna, Surajit Ghosh, Bunyod Holmatov. (2023)  
**Analysis of Rainfall Variability and Water Extent of Selected Hydropower Reservoir Using Google Earth Engine (GEE): A Case Study from Two Tropical Countries, Sri Lanka and Vietnam**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.05682v2)  

---


**ABSTRACT**  
This study presents a comprehensive remote sensing analysis of rainfall patterns and selected hydropower reservoir water extent in two tropical monsoon countries, Vietnam and Sri Lanka. The aim is to understand the relationship between remotely sensed rainfall data and the dynamic changes (monthly) in reservoir water extent. The analysis utilizes high-resolution optical imagery and Sentinel-1 Synthetic Aperture Radar (SAR) data to observe and monitor water bodies during different weather conditions, especially during the monsoon season. The average annual rainfall for both countries is determined, and spatiotemporal variations in monthly average rainfall are examined at regional and reservoir basin levels using the Climate Hazards Group InfraRed Precipitation with Station (CHIRPS) dataset from 1981 to 2022. Water extents are derived for selected reservoirs using Sentinel-1 SAR Ground Range Detected (GRD) images in Vietnam and Sri Lanka from 2017 to 2022. The images are pre-processed and corrected using terrain correction and refined Lee filter. An automated thresholding algorithm, OTSU, distinguishes water and land, taking advantage of both VV and VH polarization data. The connected pixel count threshold is applied to enhance result accuracy. The results indicate a clear relationship between rainfall patterns and reservoir water extent, with increased precipitation during the monsoon season leading to higher water extents in the later months. This study contributes to understanding how rainfall variability impacts reservoir water resources in tropical monsoon regions. The preliminary findings can inform water resource management strategies and support these countries' decision-making processes related to hydropower generation, flood management, and irrigation.

{{</citation>}}


### (133/183) Making Scalable Meta Learning Practical (Sang Keun Choe et al., 2023)

{{<citation>}}

Sang Keun Choe, Sanket Vaibhav Mehta, Hwijeen Ahn, Willie Neiswanger, Pengtao Xie, Emma Strubell, Eric Xing. (2023)  
**Making Scalable Meta Learning Practical**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.05674v1)  

---


**ABSTRACT**  
Despite its flexibility to learn diverse inductive biases in machine learning programs, meta learning (i.e., learning to learn) has long been recognized to suffer from poor scalability due to its tremendous compute/memory costs, training instability, and a lack of efficient distributed training support. In this work, we focus on making scalable meta learning practical by introducing SAMA, which combines advances in both implicit differentiation algorithms and systems. Specifically, SAMA is designed to flexibly support a broad range of adaptive optimizers in the base level of meta learning programs, while reducing computational burden by avoiding explicit computation of second-order gradient information, and exploiting efficient distributed training techniques implemented for first-order gradients. Evaluated on multiple large-scale meta learning benchmarks, SAMA showcases up to 1.7/4.8x increase in throughput and 2.0/3.8x decrease in memory consumption respectively on single-/multi-GPU setups compared to other baseline meta learning algorithms. Furthermore, we show that SAMA-based data optimization leads to consistent improvements in text classification accuracy with BERT and RoBERTa large language models, and achieves state-of-the-art results in both small- and large-scale data pruning on image classification tasks, demonstrating the practical applicability of scalable meta learning across language and vision domains.

{{</citation>}}


### (134/183) Multi-timestep models for Model-based Reinforcement Learning (Abdelhakim Benechehab et al., 2023)

{{<citation>}}

Abdelhakim Benechehab, Giuseppe Paolo, Albert Thomas, Maurizio Filippone, Balázs Kégl. (2023)  
**Multi-timestep models for Model-based Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05672v2)  

---


**ABSTRACT**  
In model-based reinforcement learning (MBRL), most algorithms rely on simulating trajectories from one-step dynamics models learned on data. A critical challenge of this approach is the compounding of one-step prediction errors as length of the trajectory grows. In this paper we tackle this issue by using a multi-timestep objective to train one-step models. Our objective is a weighted sum of a loss function (e.g., negative log-likelihood) at various future horizons. We explore and test a range of weights profiles. We find that exponentially decaying weights lead to models that significantly improve the long-horizon R2 score. This improvement is particularly noticeable when the models were evaluated on noisy data. Finally, using a soft actor-critic (SAC) agent in pure batch reinforcement learning (RL) and iterated batch RL scenarios, we found that our multi-timestep models outperform or match standard one-step models. This was especially evident in a noisy variant of the considered environment, highlighting the potential of our approach in real-world applications.

{{</citation>}}


### (135/183) LARA: A Light and Anti-overfitting Retraining Approach for Unsupervised Anomaly Detection (Feiyi Chen et al., 2023)

{{<citation>}}

Feiyi Chen, Zhen Qing, Yingying Zhang, Shuiguang Deng, Yi Xiao, Guansong Pang, Qingsong Wen. (2023)  
**LARA: A Light and Anti-overfitting Retraining Approach for Unsupervised Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.05668v1)  

---


**ABSTRACT**  
Most of current anomaly detection models assume that the normal pattern remains same all the time. However, the normal patterns of Web services change dramatically and frequently. The model trained on old-distribution data is outdated after such changes. Retraining the whole model every time is expensive. Besides, at the beginning of normal pattern changes, there is not enough observation data from the new distribution. Retraining a large neural network model with limited data is vulnerable to overfitting. Thus, we propose a Light and Anti-overfitting Retraining Approach (LARA) for deep variational auto-encoder based time series anomaly detection methods (VAEs). This work aims to make three novel contributions: 1) the retraining process is formulated as a convex problem and can converge at a fast rate as well as prevent overfitting; 2) designing a ruminate block, which leverages the historical data without the need to store them; 3) mathematically proving that when fine-tuning the latent vector and reconstructed data, the linear formations can achieve the least adjusting errors between the ground truths and the fine-tuned ones.   Moreover, we have performed many experiments to verify that retraining LARA with even 43 time slots of data from new distribution can result in its competitive F1 Score in comparison with the state-of-the-art anomaly detection models trained with sufficient data. Besides, we verify its light overhead.

{{</citation>}}


### (136/183) ODEFormer: Symbolic Regression of Dynamical Systems with Transformers (Stéphane d'Ascoli et al., 2023)

{{<citation>}}

Stéphane d'Ascoli, Sören Becker, Alexander Mathis, Philippe Schwaller, Niki Kilbertus. (2023)  
**ODEFormer: Symbolic Regression of Dynamical Systems with Transformers**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.05573v1)  

---


**ABSTRACT**  
We introduce ODEFormer, the first transformer able to infer multidimensional ordinary differential equation (ODE) systems in symbolic form from the observation of a single solution trajectory. We perform extensive evaluations on two datasets: (i) the existing "Strogatz" dataset featuring two-dimensional systems; (ii) ODEBench, a collection of one- to four-dimensional systems that we carefully curated from the literature to provide a more holistic benchmark. ODEFormer consistently outperforms existing methods while displaying substantially improved robustness to noisy and irregularly sampled observations, as well as faster inference. We release our code, model and benchmark dataset publicly.

{{</citation>}}


### (137/183) A novel Network Science Algorithm for Improving Triage of Patients (Pietro Hiram Guzzi et al., 2023)

{{<citation>}}

Pietro Hiram Guzzi, Annamaria De Filippo, Pierangelo Veltri. (2023)  
**A novel Network Science Algorithm for Improving Triage of Patients**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05996v1)  

---


**ABSTRACT**  
Patient triage plays a crucial role in healthcare, ensuring timely and appropriate care based on the urgency of patient conditions. Traditional triage methods heavily rely on human judgment, which can be subjective and prone to errors. Recently, a growing interest has been in leveraging artificial intelligence (AI) to develop algorithms for triaging patients. This paper presents the development of a novel algorithm for triaging patients. It is based on the analysis of patient data to produce decisions regarding their prioritization. The algorithm was trained on a comprehensive data set containing relevant patient information, such as vital signs, symptoms, and medical history. The algorithm was designed to accurately classify patients into triage categories through rigorous preprocessing and feature engineering. Experimental results demonstrate that our algorithm achieved high accuracy and performance, outperforming traditional triage methods. By incorporating computer science into the triage process, healthcare professionals can benefit from improved efficiency, accuracy, and consistency, prioritizing patients effectively and optimizing resource allocation. Although further research is needed to address challenges such as biases in training data and model interpretability, the development of AI-based algorithms for triaging patients shows great promise in enhancing healthcare delivery and patient outcomes.

{{</citation>}}


### (138/183) On Double-Descent in Reinforcement Learning with LSTD and Random Features (David Brellmann et al., 2023)

{{<citation>}}

David Brellmann, Eloïse Berthier, David Filliat, Goran Frehse. (2023)  
**On Double-Descent in Reinforcement Learning with LSTD and Random Features**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05518v1)  

---


**ABSTRACT**  
Temporal Difference (TD) algorithms are widely used in Deep Reinforcement Learning (RL). Their performance is heavily influenced by the size of the neural network. While in supervised learning, the regime of over-parameterization and its benefits are well understood, the situation in RL is much less clear. In this paper, we present a theoretical analysis of the influence of network size and $l_2$-regularization on performance. We identify the ratio between the number of parameters and the number of visited states as a crucial factor and define over-parameterization as the regime when it is larger than one. Furthermore, we observe a double-descent phenomenon, i.e., a sudden drop in performance around the parameter/state ratio of one. Leveraging random features and the lazy training regime, we study the regularized Least-Square Temporal Difference (LSTD) algorithm in an asymptotic regime, as both the number of parameters and states go to infinity, maintaining a constant ratio. We derive deterministic limits of both the empirical and the true Mean-Square Bellman Error (MSBE) that feature correction terms responsible for the double-descent. Correction terms vanish when the $l_2$-regularization is increased or the number of unvisited states goes to zero. Numerical experiments with synthetic and small real-world environments closely match the theoretical predictions.

{{</citation>}}


### (139/183) WeatherGNN: Exploiting Complicated Relationships in Numerical Weather Prediction Bias Correction (Binqing Wu et al., 2023)

{{<citation>}}

Binqing Wu, Weiqi Chen, Wengwei Wang, Bingqing Peng, Liang Sun, Ling Chen. (2023)  
**WeatherGNN: Exploiting Complicated Relationships in Numerical Weather Prediction Bias Correction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.05517v1)  

---


**ABSTRACT**  
Numerical weather prediction (NWP) may be inaccurate or biased due to incomplete atmospheric physical processes, insufficient spatial-temporal resolution, and inherent uncertainty of weather. Previous studies have attempted to correct biases by using handcrafted features and domain knowledge, or by applying general machine learning models naively. They do not fully explore the complicated meteorologic interactions and spatial dependencies in the atmosphere dynamically, which limits their applicability in NWP bias-correction. Specifically, weather factors interact with each other in complex ways, and these interactions can vary regionally. In addition, the interactions between weather factors are further complicated by the spatial dependencies between regions, which are influenced by varied terrain and atmospheric motions. To address these issues, we propose WeatherGNN, an NWP bias-correction method that utilizes Graph Neural Networks (GNN) to learn meteorologic and geographic relationships in a unified framework. Our approach includes a factor-wise GNN that captures meteorological interactions within each grid (a specific location) adaptively, and a fast hierarchical GNN that captures spatial dependencies between grids dynamically. Notably, the fast hierarchical GNN achieves linear complexity with respect to the number of grids, enhancing model efficiency and scalability. Our experimental results on two real-world datasets demonstrate the superiority of WeatherGNN in comparison with other SOTA methods, with an average improvement of 40.50\% on RMSE compared to the original NWP.

{{</citation>}}


### (140/183) Temporal Convolutional Explorer Helps Understand 1D-CNN's Learning Behavior in Time Series Classification from Frequency Domain (Junru Zhang et al., 2023)

{{<citation>}}

Junru Zhang, Lang Feng, Yang He, Yuhan Wu, Yabo Dong. (2023)  
**Temporal Convolutional Explorer Helps Understand 1D-CNN's Learning Behavior in Time Series Classification from Frequency Domain**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.05467v1)  

---


**ABSTRACT**  
While one-dimensional convolutional neural networks (1D-CNNs) have been empirically proven effective in time series classification tasks, we find that there remain undesirable outcomes that could arise in their application, motivating us to further investigate and understand their underlying mechanisms. In this work, we propose a Temporal Convolutional Explorer (TCE) to empirically explore the learning behavior of 1D-CNNs from the perspective of the frequency domain. Our TCE analysis highlights that deeper 1D-CNNs tend to distract the focus from the low-frequency components leading to the accuracy degradation phenomenon, and the disturbing convolution is the driving factor. Then, we leverage our findings to the practical application and propose a regulatory framework, which can easily be integrated into existing 1D-CNNs. It aims to rectify the suboptimal learning behavior by enabling the network to selectively bypass the specified disturbing convolutions. Finally, through comprehensive experiments on widely-used UCR, UEA, and UCI benchmarks, we demonstrate that 1) TCE's insight into 1D-CNN's learning behavior; 2) our regulatory framework enables state-of-the-art 1D-CNNs to get improved performances with less consumption of memory and computational overhead.

{{</citation>}}


### (141/183) Reward-Consistent Dynamics Models are Strongly Generalizable for Offline Reinforcement Learning (Fan-Ming Luo et al., 2023)

{{<citation>}}

Fan-Ming Luo, Tian Xu, Xingchen Cao, Yang Yu. (2023)  
**Reward-Consistent Dynamics Models are Strongly Generalizable for Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05422v1)  

---


**ABSTRACT**  
Learning a precise dynamics model can be crucial for offline reinforcement learning, which, unfortunately, has been found to be quite challenging. Dynamics models that are learned by fitting historical transitions often struggle to generalize to unseen transitions. In this study, we identify a hidden but pivotal factor termed dynamics reward that remains consistent across transitions, offering a pathway to better generalization. Therefore, we propose the idea of reward-consistent dynamics models: any trajectory generated by the dynamics model should maximize the dynamics reward derived from the data. We implement this idea as the MOREC (Model-based Offline reinforcement learning with Reward Consistency) method, which can be seamlessly integrated into previous offline model-based reinforcement learning (MBRL) methods. MOREC learns a generalizable dynamics reward function from offline data, which is subsequently employed as a transition filter in any offline MBRL method: when generating transitions, the dynamics model generates a batch of transitions and selects the one with the highest dynamics reward value. On a synthetic task, we visualize that MOREC has a strong generalization ability and can surprisingly recover some distant unseen transitions. On 21 offline tasks in D4RL and NeoRL benchmarks, MOREC improves the previous state-of-the-art performance by a significant margin, i.e., 4.6% on D4RL tasks and 25.9% on NeoRL tasks. Notably, MOREC is the first method that can achieve above 95% online RL performance in 6 out of 12 D4RL tasks and 3 out of 9 NeoRL tasks.

{{</citation>}}


### (142/183) Molecular De Novo Design through Transformer-based Reinforcement Learning (Tao Feng et al., 2023)

{{<citation>}}

Tao Feng, Pengcheng Xu, Tianfan Fu, Siddhartha Laghuvarapu, Jimeng Sun. (2023)  
**Molecular De Novo Design through Transformer-based Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.05365v2)  

---


**ABSTRACT**  
In this work, we introduce a method to fine-tune a Transformer-based generative model for molecular de novo design. Leveraging the superior sequence learning capacity of Transformers over Recurrent Neural Networks (RNNs), our model can generate molecular structures with desired properties effectively. In contrast to the traditional RNN-based models, our proposed method exhibits superior performance in generating compounds predicted to be active against various biological targets, capturing long-term dependencies in the molecular structure sequence. The model's efficacy is demonstrated across numerous tasks, including generating analogues to a query structure and producing compounds with particular attributes, outperforming the baseline RNN-based methods. Our approach can be used for scaffold hopping, library expansion starting from a single molecule, and generating compounds with high predicted activity against biological targets.

{{</citation>}}


### (143/183) GReAT: A Graph Regularized Adversarial Training Method (Samet Bayram et al., 2023)

{{<citation>}}

Samet Bayram, Kenneth Barner. (2023)  
**GReAT: A Graph Regularized Adversarial Training Method**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2310.05336v1)  

---


**ABSTRACT**  
This paper proposes a regularization method called GReAT, Graph Regularized Adversarial Training, to improve deep learning models' classification performance. Adversarial examples are a well-known challenge in machine learning, where small, purposeful perturbations to input data can mislead models. Adversarial training, a powerful and one of the most effective defense strategies, involves training models with both regular and adversarial examples. However, it often neglects the underlying structure of the data. In response, we propose GReAT, a method that leverages data graph structure to enhance model robustness. GReAT deploys the graph structure of the data into the adversarial training process, resulting in more robust models that better generalize its testing performance and defend against adversarial attacks. Through extensive evaluation on benchmark datasets, we demonstrate GReAT's effectiveness compared to state-of-the-art classification methods, highlighting its potential in improving deep learning models' classification performance.

{{</citation>}}


### (144/183) DiffCPS: Diffusion Model based Constrained Policy Search for Offline Reinforcement Learning (Longxiang He et al., 2023)

{{<citation>}}

Longxiang He, Linrui Zhang, Junbo Tan, Xueqian Wang. (2023)  
**DiffCPS: Diffusion Model based Constrained Policy Search for Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05333v1)  

---


**ABSTRACT**  
Constrained policy search (CPS) is a fundamental problem in offline reinforcement learning, which is generally solved by advantage weighted regression (AWR). However, previous methods may still encounter out-of-distribution actions due to the limited expressivity of Gaussian-based policies. On the other hand, directly applying the state-of-the-art models with distribution expression capabilities (i.e., diffusion models) in the AWR framework is insufficient since AWR requires exact policy probability densities, which is intractable in diffusion models. In this paper, we propose a novel approach called $\textbf{Diffusion Model based Constrained Policy Search (DiffCPS)}$, which tackles the diffusion-based constrained policy search without resorting to AWR. The theoretical analysis reveals our key insights by leveraging the action distribution of the diffusion model to eliminate the policy distribution constraint in the CPS and then utilizing the Evidence Lower Bound (ELBO) of diffusion-based policy to approximate the KL constraint. Consequently, DiffCPS admits the high expressivity of diffusion models while circumventing the cumbersome density calculation brought by AWR. Extensive experimental results based on the D4RL benchmark demonstrate the efficacy of our approach. We empirically show that DiffCPS achieves better or at least competitive performance compared to traditional AWR-based baselines as well as recent diffusion-based offline RL methods. The code is now available at $\href{https://github.com/felix-thu/DiffCPS}{https://github.com/felix-thu/DiffCPS}$.

{{</citation>}}


## cs.HC (2)



### (145/183) How AI Processing Delays Foster Creativity: Exploring Research Question Co-Creation with an LLM-based Agent (Yiren Liu et al., 2023)

{{<citation>}}

Yiren Liu, Si Chen, Haocong Chen, Mengxia Yu, Xiao Ran, Andrew Mo, Yiliu Tang, Yun Huang. (2023)  
**How AI Processing Delays Foster Creativity: Exploring Research Question Co-Creation with an LLM-based Agent**  

---
Primary Category: cs.HC  
Categories: cs-CE, cs-HC, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.06155v1)  

---


**ABSTRACT**  
Developing novel research questions (RQs) often requires extensive literature reviews, especially for interdisciplinary fields. Leveraging Large Language Models (LLMs), we built an LLM-based agent system, called CoQuest, supporting RQ development through human-AI co-creation. We conducted an experimental design with 20 participants to examine the effect of two interaction designs: breadth-first and depth-first RQ generation. The results showed that participants found the breadth-first approach more creative and trustworthy upon task completion. However, during the task, they rated the RQs generated through the depth-first approach as more creative. We also discovered that AI processing delays allowed users to contemplate multiple RQs simultaneously, resulting in more generated RQs and an increased sense of perceived control. Our work makes both theoretical and practical contributions by proposing and assessing a mental model for human-AI co-creation RQs.

{{</citation>}}


### (146/183) 'Mango Mango, How to Let The Lettuce Dry Without A Spinner?'': Exploring User Perceptions of Using An LLM-Based Conversational Assistant Toward Cooking Partner (Szeyi Chan et al., 2023)

{{<citation>}}

Szeyi Chan, Jiachen Li, Bingsheng Yao, Amama Mahmood, Chien-Ming Huang, Holly Jimison, Elizabeth D Mynatt, Dakuo Wang. (2023)  
**'Mango Mango, How to Let The Lettuce Dry Without A Spinner?'': Exploring User Perceptions of Using An LLM-Based Conversational Assistant Toward Cooking Partner**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.05853v1)  

---


**ABSTRACT**  
The rapid advancement of the Large Language Model (LLM) has created numerous potentials for integration with conversational assistants (CAs) assisting people in their daily tasks, particularly due to their extensive flexibility. However, users' real-world experiences interacting with these assistants remain unexplored. In this research, we chose cooking, a complex daily task, as a scenario to investigate people's successful and unsatisfactory experiences while receiving assistance from an LLM-based CA, Mango Mango. We discovered that participants value the system's ability to provide extensive information beyond the recipe, offer customized instructions based on context, and assist them in dynamically planning the task. However, they expect the system to be more adaptive to oral conversation and provide more suggestive responses to keep users actively involved. Recognizing that users began treating our LLM-CA as a personal assistant or even a partner rather than just a recipe-reading tool, we propose several design considerations for future development.

{{</citation>}}


## eess.IV (6)



### (147/183) HydraViT: Adaptive Multi-Branch Transformer for Multi-Label Disease Classification from Chest X-ray Images (Şaban Öztürk et al., 2023)

{{<citation>}}

Şaban Öztürk, M. Yiğit Turalı, Tolga Çukur. (2023)  
**HydraViT: Adaptive Multi-Branch Transformer for Multi-Label Disease Classification from Chest X-ray Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.06143v1)  

---


**ABSTRACT**  
Chest X-ray is an essential diagnostic tool in the identification of chest diseases given its high sensitivity to pathological abnormalities in the lungs. However, image-driven diagnosis is still challenging due to heterogeneity in size and location of pathology, as well as visual similarities and co-occurrence of separate pathology. Since disease-related regions often occupy a relatively small portion of diagnostic images, classification models based on traditional convolutional neural networks (CNNs) are adversely affected given their locality bias. While CNNs were previously augmented with attention maps or spatial masks to guide focus on potentially critical regions, learning localization guidance under heterogeneity in the spatial distribution of pathology is challenging. To improve multi-label classification performance, here we propose a novel method, HydraViT, that synergistically combines a transformer backbone with a multi-branch output module with learned weighting. The transformer backbone enhances sensitivity to long-range context in X-ray images, while using the self-attention mechanism to adaptively focus on task-critical regions. The multi-branch output module dedicates an independent branch to each disease label to attain robust learning across separate disease classes, along with an aggregated branch across labels to maintain sensitivity to co-occurrence relationships among pathology. Experiments demonstrate that, on average, HydraViT outperforms competing attention-guided methods by 1.2%, region-guided methods by 1.4%, and semantic-guided methods by 1.0% in multi-label classification performance.

{{</citation>}}


### (148/183) High Accuracy and Cost-Saving Active Learning 3D WD-UNet for Airway Segmentation (Shiyi Wang et al., 2023)

{{<citation>}}

Shiyi Wang, Yang Nan, Simon Walsh, Guang Yang. (2023)  
**High Accuracy and Cost-Saving Active Learning 3D WD-UNet for Airway Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2310.05638v1)  

---


**ABSTRACT**  
We propose a novel Deep Active Learning (DeepAL) model-3D Wasserstein Discriminative UNet (WD-UNet) for reducing the annotation effort of medical 3D Computed Tomography (CT) segmentation. The proposed WD-UNet learns in a semi-supervised way and accelerates learning convergence to meet or exceed the prediction metrics of supervised learning models. Our method can be embedded with different Active Learning (AL) strategies and different network structures. The model is evaluated on 3D lung airway CT scans for medical segmentation and show that the use of uncertainty metric, which is parametrized as an input of query strategy, leads to more accurate prediction results than some state-of-the-art Deep Learning (DL) supervised models, e.g.,3DUNet and 3D CEUNet. Compared to the above supervised DL methods, our WD-UNet not only saves the cost of annotation for radiologists but also saves computational resources. WD-UNet uses a limited amount of annotated data (35% of the total) to achieve better predictive metrics with a more efficient deep learning model algorithm.

{{</citation>}}


### (149/183) A Simple and Robust Framework for Cross-Modality Medical Image Segmentation applied to Vision Transformers (Matteo Bastico et al., 2023)

{{<citation>}}

Matteo Bastico, David Ryckelynck, Laurent Corté, Yannick Tillier, Etienne Decencière. (2023)  
**A Simple and Robust Framework for Cross-Modality Medical Image Segmentation applied to Vision Transformers**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.05572v1)  

---


**ABSTRACT**  
When it comes to clinical images, automatic segmentation has a wide variety of applications and a considerable diversity of input domains, such as different types of Magnetic Resonance Images (MRIs) and Computerized Tomography (CT) scans. This heterogeneity is a challenge for cross-modality algorithms that should equally perform independently of the input image type fed to them. Often, segmentation models are trained using a single modality, preventing generalization to other types of input data without resorting to transfer learning techniques. Furthermore, the multi-modal or cross-modality architectures proposed in the literature frequently require registered images, which are not easy to collect in clinical environments, or need additional processing steps, such as synthetic image generation. In this work, we propose a simple framework to achieve fair image segmentation of multiple modalities using a single conditional model that adapts its normalization layers based on the input type, trained with non-registered interleaved mixed data. We show that our framework outperforms other cross-modality segmentation methods, when applied to the same 3D UNet baseline model, on the Multi-Modality Whole Heart Segmentation Challenge. Furthermore, we define the Conditional Vision Transformer (C-ViT) encoder, based on the proposed cross-modality framework, and we show that it brings significant improvements to the resulting segmentation, up to 6.87\% of Dice accuracy, with respect to its baseline reference. The code to reproduce our experiments and the trained model weights are available at https://github.com/matteo-bastico/MI-Seg.

{{</citation>}}


### (150/183) M3FPolypSegNet: Segmentation Network with Multi-frequency Feature Fusion for Polyp Localization in Colonoscopy Images (Ju-Hyeon Nam et al., 2023)

{{<citation>}}

Ju-Hyeon Nam, Seo-Hyeong Park, Nur Suriza Syazwany, Yerim Jung, Yu-Han Im, Sang-Chul Lee. (2023)  
**M3FPolypSegNet: Segmentation Network with Multi-frequency Feature Fusion for Polyp Localization in Colonoscopy Images**  

---
Primary Category: eess.IV  
Categories: 92C55, cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2310.05538v2)  

---


**ABSTRACT**  
Polyp segmentation is crucial for preventing colorectal cancer a common type of cancer. Deep learning has been used to segment polyps automatically, which reduces the risk of misdiagnosis. Localizing small polyps in colonoscopy images is challenging because of its complex characteristics, such as color, occlusion, and various shapes of polyps. To address this challenge, a novel frequency-based fully convolutional neural network, Multi-Frequency Feature Fusion Polyp Segmentation Network (M3FPolypSegNet) was proposed to decompose the input image into low/high/full-frequency components to use the characteristics of each component. We used three independent multi-frequency encoders to map multiple input images into a high-dimensional feature space. In the Frequency-ASPP Scalable Attention Module (F-ASPP SAM), ASPP was applied between each frequency component to preserve scale information. Subsequently, scalable attention was applied to emphasize polyp regions in a high-dimensional feature space. Finally, we designed three multi-task learning (i.e., region, edge, and distance) in four decoder blocks to learn the structural characteristics of the region. The proposed model outperformed various segmentation models with performance gains of 6.92% and 7.52% on average for all metrics on CVC-ClinicDB and BKAI-IGH-NeoPolyp, respectively.

{{</citation>}}


### (151/183) RetSeg: Retention-based Colorectal Polyps Segmentation Network (Khaled ELKarazle et al., 2023)

{{<citation>}}

Khaled ELKarazle, Valliappan Raman, Caslon Chua, Patrick Then. (2023)  
**RetSeg: Retention-based Colorectal Polyps Segmentation Network**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.05446v2)  

---


**ABSTRACT**  
Vision Transformers (ViTs) have revolutionized medical imaging analysis, showcasing superior efficacy compared to conventional Convolutional Neural Networks (CNNs) in vital tasks such as polyp classification, detection, and segmentation. Leveraging attention mechanisms to focus on specific image regions, ViTs exhibit contextual awareness in processing visual data, culminating in robust and precise predictions, even for intricate medical images. Moreover, the inherent self-attention mechanism in Transformers accommodates varying input sizes and resolutions, granting an unprecedented flexibility absent in traditional CNNs. However, Transformers grapple with challenges like excessive memory usage and limited training parallelism due to self-attention, rendering them impractical for real-time disease detection on resource-constrained devices. In this study, we address these hurdles by investigating the integration of the recently introduced retention mechanism into polyp segmentation, introducing RetSeg, an encoder-decoder network featuring multi-head retention blocks. Drawing inspiration from Retentive Networks (RetNet), RetSeg is designed to bridge the gap between precise polyp segmentation and resource utilization, particularly tailored for colonoscopy images. We train and validate RetSeg for polyp segmentation employing two publicly available datasets: Kvasir-SEG and CVC-ClinicDB. Additionally, we showcase RetSeg's promising performance across diverse public datasets, including CVC-ColonDB, ETIS-LaribPolypDB, CVC-300, and BKAI-IGH NeoPolyp. While our work represents an early-stage exploration, further in-depth studies are imperative to advance these promising findings.

{{</citation>}}


### (152/183) Enhancing Prostate Cancer Diagnosis with Deep Learning: A Study using mpMRI Segmentation and Classification (Anil B. Gavade et al., 2023)

{{<citation>}}

Anil B. Gavade, Neel Kanwal, Priyanka A. Gavade, Rajendra Nerli. (2023)  
**Enhancing Prostate Cancer Diagnosis with Deep Learning: A Study using mpMRI Segmentation and Classification**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.05371v2)  

---


**ABSTRACT**  
Prostate cancer (PCa) is a severe disease among men globally. It is important to identify PCa early and make a precise diagnosis for effective treatment. For PCa diagnosis, Multi-parametric magnetic resonance imaging (mpMRI) emerged as an invaluable imaging modality that offers a precise anatomical view of the prostate gland and its tissue structure. Deep learning (DL) models can enhance existing clinical systems and improve patient care by locating regions of interest for physicians. Recently, DL techniques have been employed to develop a pipeline for segmenting and classifying different cancer types. These studies show that DL can be used to increase diagnostic precision and give objective results without variability. This work uses well-known DL models for the classification and segmentation of mpMRI images to detect PCa. Our implementation involves four pipelines; Semantic DeepSegNet with ResNet50, DeepSegNet with recurrent neural network (RNN), U-Net with RNN, and U-Net with a long short-term memory (LSTM). Each segmentation model is paired with a different classifier to evaluate the performance using different metrics. The results of our experiments show that the pipeline that uses the combination of U-Net and the LSTM model outperforms all other combinations, excelling in both segmentation and classification tasks.

{{</citation>}}


## stat.ML (3)



### (153/183) Grokking as the Transition from Lazy to Rich Training Dynamics (Tanishq Kumar et al., 2023)

{{<citation>}}

Tanishq Kumar, Blake Bordelon, Samuel J. Gershman, Cengiz Pehlevan. (2023)  
**Grokking as the Transition from Lazy to Rich Training Dynamics**  

---
Primary Category: stat.ML  
Categories: cond-mat-dis-nn, cs-LG, stat-ML, stat.ML  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.06110v1)  

---


**ABSTRACT**  
We propose that the grokking phenomenon, where the train loss of a neural network decreases much earlier than its test loss, can arise due to a neural network transitioning from lazy training dynamics to a rich, feature learning regime. To illustrate this mechanism, we study the simple setting of vanilla gradient descent on a polynomial regression problem with a two layer neural network which exhibits grokking without regularization in a way that cannot be explained by existing theories. We identify sufficient statistics for the test loss of such a network, and tracking these over training reveals that grokking arises in this setting when the network first attempts to fit a kernel regression solution with its initial features, followed by late-time feature learning where a generalizing solution is identified after train loss is already low. We find that the key determinants of grokking are the rate of feature learning -- which can be controlled precisely by parameters that scale the network output -- and the alignment of the initial features with the target function $y(x)$. We argue this delayed generalization arises when (1) the top eigenvectors of the initial neural tangent kernel and the task labels $y(x)$ are misaligned, but (2) the dataset size is large enough so that it is possible for the network to generalize eventually, but not so large that train loss perfectly tracks test loss at all epochs, and (3) the network begins training in the lazy regime so does not learn features immediately. We conclude with evidence that this transition from lazy (linear model) to rich training (feature learning) can control grokking in more general settings, like on MNIST, one-layer Transformers, and student-teacher networks.

{{</citation>}}


### (154/183) Post-hoc Bias Scoring Is Optimal For Fair Classification (Wenlong Chen et al., 2023)

{{<citation>}}

Wenlong Chen, Yegor Klochkov, Yang Liu. (2023)  
**Post-hoc Bias Scoring Is Optimal For Fair Classification**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.05725v1)  

---


**ABSTRACT**  
We consider a binary classification problem under group fairness constraints, which can be one of Demographic Parity (DP), Equalized Opportunity (EOp), or Equalized Odds (EO). We propose an explicit characterization of Bayes optimal classifier under the fairness constraints, which turns out to be a simple modification rule of the unconstrained classifier. Namely, we introduce a novel instance-level measure of bias, which we call bias score, and the modification rule is a simple linear rule on top of the finite amount of bias scores. Based on this characterization, we develop a post-hoc approach that allows us to adapt to fairness constraints while maintaining high accuracy. In the case of DP and EOp constraints, the modification rule is thresholding a single bias score, while in the case of EO constraints we are required to fit a linear modification rule with 2 parameters. The method can also be applied for composite group-fairness criteria, such as ones involving several sensitive attributes. We achieve competitive or better performance compared to both in-processing and post-processing methods across three datasets: Adult, COMPAS, and CelebA. Unlike most post-processing methods, we do not require access to sensitive attributes during the inference time.

{{</citation>}}


### (155/183) ExIFFI and EIF+: Interpretability and Enhanced Generalizability to Extend the Extended Isolation Forest (Alessio Arcudi et al., 2023)

{{<citation>}}

Alessio Arcudi, Davide Frizzo, Chiara Masiero, Gian Antonio Susto. (2023)  
**ExIFFI and EIF+: Interpretability and Enhanced Generalizability to Extend the Extended Isolation Forest**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-AP, stat-ML, stat.ML  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.05468v1)  

---


**ABSTRACT**  
Anomaly detection, an essential unsupervised machine learning task, involves identifying unusual behaviors within complex datasets and systems. While Machine Learning algorithms and decision support systems (DSSs) offer effective solutions for this task, simply pinpointing anomalies often falls short in real-world applications. Users of these systems often require insight into the underlying reasons behind predictions to facilitate Root Cause Analysis and foster trust in the model. However, due to the unsupervised nature of anomaly detection, creating interpretable tools is challenging. This work introduces EIF+, an enhanced variant of Extended Isolation Forest (EIF), designed to enhance generalization capabilities. Additionally, we present ExIFFI, a novel approach that equips Extended Isolation Forest with interpretability features, specifically feature rankings. Experimental results provide a comprehensive comparative analysis of Isolation-based approaches for Anomaly Detection, including synthetic and real dataset evaluations that demonstrate ExIFFI's effectiveness in providing explanations. We also illustrate how ExIFFI serves as a valid feature selection technique in unsupervised settings. To facilitate further research and reproducibility, we also provide open-source code to replicate the results.

{{</citation>}}


## cs.DC (4)



### (156/183) CFPB Consumer Complaints Analysis Using Hadoop (Dhwani Vaishnav et al., 2023)

{{<citation>}}

Dhwani Vaishnav, Manimozhi Neethinayagam, Akanksha S Khaire, Mansi Vivekanand Dhoke, Jongwook Woo. (2023)  
**CFPB Consumer Complaints Analysis Using Hadoop**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2310.06076v1)  

---


**ABSTRACT**  
Consumer complaints are a crucial source of information for companies, policymakers, and consumers alike. They provide insight into the problems faced by consumers and help identify areas for improvement in products, services, and regulatory frameworks. This paper aims to analyze Consumer Complaints Dataset provided by Consumer Financial Protection Bureau (CFPB) and provide insights into the nature and patterns of consumer complaints in the USA. We begin by describing the dataset and its features, including the types of complaints, companies involved, and geographic distribution. We then conduct exploratory data analysis to identify trends and patterns in the data, such as the most common types of complaints, the companies with the highest number of complaints, and the states with the most complaints. We have also performed descriptive and inferential statistics to test hypotheses and draw conclusions about the data. We have investigated whether there are significant differences in the types of complaints or companies involved based on geographic location. Overall, our analysis provides valuable insights into the nature of consumer complaints in the USA and helps stakeholders make informed decisions to improve the consumer experience.

{{</citation>}}


### (157/183) CLAID: Closing the Loop on AI & Data Collection -- A Cross-Platform Transparent Computing Middleware Framework for Smart Edge-Cloud and Digital Biomarker Applications (Patrick Langer et al., 2023)

{{<citation>}}

Patrick Langer, Elgar Fleisch, Filipe Barata. (2023)  
**CLAID: Closing the Loop on AI & Data Collection -- A Cross-Platform Transparent Computing Middleware Framework for Smart Edge-Cloud and Digital Biomarker Applications**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-SE, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05643v1)  

---


**ABSTRACT**  
The increasing number of edge devices with enhanced sensing capabilities, such as smartphones, wearables, and IoT devices equipped with sensors, holds the potential for innovative smart-edge applications in healthcare. These devices generate vast amounts of multimodal data, enabling the implementation of digital biomarkers which can be leveraged by machine learning solutions to derive insights, predict health risks, and allow personalized interventions. Training these models requires collecting data from edge devices and aggregating it in the cloud. To validate and verify those models, it is essential to utilize them in real-world scenarios and subject them to testing using data from diverse cohorts. Since some models are too computationally expensive to be run on edge devices directly, a collaborative framework between the edge and cloud becomes necessary. In this paper, we present CLAID, an open-source cross-platform middleware framework based on transparent computing compatible with Android, iOS, WearOS, Linux, macOS, and Windows. CLAID enables logical integration of devices running different operating systems into an edge-cloud system, facilitating communication and offloading between them, with bindings available in different programming languages. We provide Modules for data collection from various sensors as well as for the deployment of machine-learning models. Furthermore, we propose a novel methodology, "ML-Model in the Loop" for verifying deployed machine learning models, which helps to analyze problems that may occur during the migration of models from cloud to edge devices. We verify our framework in three different experiments and achieve 100% sampling coverage for data collection across different sensors as well as an equal performance of a cough detection model deployed on both Android and iOS devices. We evaluate the memory and battery consumption of our framework.

{{</citation>}}


### (158/183) EdgeAISim: A Toolkit for Simulation and Modelling of AI Models in Edge Computing Environments (Aadharsh Roshan Nandhakumar et al., 2023)

{{<citation>}}

Aadharsh Roshan Nandhakumar, Ayush Baranwal, Priyanshukumar Choudhary, Muhammed Golec, Sukhpal Singh Gill. (2023)  
**EdgeAISim: A Toolkit for Simulation and Modelling of AI Models in Edge Computing Environments**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.05605v1)  

---


**ABSTRACT**  
To meet next-generation IoT application demands, edge computing moves processing power and storage closer to the network edge to minimise latency and bandwidth utilisation. Edge computing is becoming popular as a result of these benefits, but resource management is still challenging. Researchers are utilising AI models to solve the challenge of resource management in edge computing systems. However, existing simulation tools are only concerned with typical resource management policies, not the adoption and implementation of AI models for resource management, especially. Consequently, researchers continue to face significant challenges, making it hard and time-consuming to use AI models when designing novel resource management policies for edge computing with existing simulation tools. To overcome these issues, we propose a lightweight Python-based toolkit called EdgeAISim for the simulation and modelling of AI models for designing resource management policies in edge computing environments. In EdgeAISim, we extended the basic components of the EdgeSimPy framework and developed new AI-based simulation models for task scheduling, energy management, service migration, network flow scheduling, and mobility support for edge computing environments. In EdgeAISim, we have utilised advanced AI models such as Multi-Armed Bandit with Upper Confidence Bound, Deep Q-Networks, Deep Q-Networks with Graphical Neural Network, and ActorCritic Network to optimize power usage while efficiently managing task migration within the edge computing environment. The performance of these proposed models of EdgeAISim is compared with the baseline, which uses a worst-fit algorithm-based resource management policy in different settings. Experimental results indicate that EdgeAISim exhibits a substantial reduction in power consumption, highlighting the compelling success of power optimization strategies in EdgeAISim.

{{</citation>}}


### (159/183) Scaling Studies for Efficient Parameter Search and Parallelism for Large Language Model Pre-training (Michael Benington et al., 2023)

{{<citation>}}

Michael Benington, Leo Phan, Chris Pierre Paul, Evan Shoemaker, Priyanka Ranade, Torstein Collett, Grant Hodgson Perez, Christopher Krieger. (2023)  
**Scaling Studies for Efficient Parameter Search and Parallelism for Large Language Model Pre-training**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-LG, cs.DC  
Keywords: AI, Language Model, Microsoft  
[Paper Link](http://arxiv.org/abs/2310.05350v2)  

---


**ABSTRACT**  
AI accelerator processing capabilities and memory constraints largely dictate the scale in which machine learning workloads (e.g., training and inference) can be executed within a desirable time frame. Training a state of the art, transformer-based model today requires use of GPU-accelerated high performance computers with high-speed interconnects. As datasets and models continue to increase in size, computational requirements and memory demands for AI also continue to grow. These challenges have inspired the development of distributed algorithm and circuit-based optimization techniques that enable the ability to progressively scale models in multi-node environments, efficiently minimize neural network cost functions for faster convergence, and store more parameters into a set number of available resources. In our research project, we focus on parallel and distributed machine learning algorithm development, specifically for optimizing the data processing and pre-training of a set of 5 encoder-decoder LLMs, ranging from 580 million parameters to 13 billion parameters. We performed a fine-grained study to quantify the relationships between three ML parallelism methods, specifically exploring Microsoft DeepSpeed Zero Redundancy Optimizer (ZeRO) stages.

{{</citation>}}


## cs.SD (5)



### (160/183) JVNV: A Corpus of Japanese Emotional Speech with Verbal Content and Nonverbal Expressions (Detai Xin et al., 2023)

{{<citation>}}

Detai Xin, Junfeng Jiang, Shinnosuke Takamichi, Yuki Saito, Akiko Aizawa, Hiroshi Saruwatari. (2023)  
**JVNV: A Corpus of Japanese Emotional Speech with Verbal Content and Nonverbal Expressions**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.06072v1)  

---


**ABSTRACT**  
We present the JVNV, a Japanese emotional speech corpus with verbal content and nonverbal vocalizations whose scripts are generated by a large-scale language model. Existing emotional speech corpora lack not only proper emotional scripts but also nonverbal vocalizations (NVs) that are essential expressions in spoken language to express emotions. We propose an automatic script generation method to produce emotional scripts by providing seed words with sentiment polarity and phrases of nonverbal vocalizations to ChatGPT using prompt engineering. We select 514 scripts with balanced phoneme coverage from the generated candidate scripts with the assistance of emotion confidence scores and language fluency scores. We demonstrate the effectiveness of JVNV by showing that JVNV has better phoneme coverage and emotion recognizability than previous Japanese emotional speech corpora. We then benchmark JVNV on emotional text-to-speech synthesis using discrete codes to represent NVs. We show that there still exists a gap between the performance of synthesizing read-aloud speech and emotional speech, and adding NVs in the speech makes the task even harder, which brings new challenges for this task and makes JVNV a valuable resource for relevant works in the future. To our best knowledge, JVNV is the first speech corpus that generates scripts automatically using large language models.

{{</citation>}}


### (161/183) Audio compression-assisted feature extraction for voice replay attack detection (Xiangyu Shi et al., 2023)

{{<citation>}}

Xiangyu Shi, Yuhao Luo, Li Wang, Haorui He, Hao Li, Lei Wang, Zhizheng Wu. (2023)  
**Audio compression-assisted feature extraction for voice replay attack detection**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Speaker Verification  
[Paper Link](http://arxiv.org/abs/2310.05813v2)  

---


**ABSTRACT**  
Replay attack is one of the most effective and simplest voice spoofing attacks. Detecting replay attacks is challenging, according to the Automatic Speaker Verification Spoofing and Countermeasures Challenge 2021 (ASVspoof 2021), because they involve a loudspeaker, a microphone, and acoustic conditions (e.g., background noise). One obstacle to detecting replay attacks is finding robust feature representations that reflect the channel noise information added to the replayed speech. This study proposes a feature extraction approach that uses audio compression for assistance. Audio compression compresses audio to preserve content and speaker information for transmission. The missed information after decompression is expected to contain content- and speaker-independent information (e.g., channel noise added during the replay process). We conducted a comprehensive experiment with a few data augmentation techniques and 3 classifiers on the ASVspoof 2021 physical access (PA) set and confirmed the effectiveness of the proposed feature extraction approach. To the best of our knowledge, the proposed approach achieves the lowest EER at 22.71% on the ASVspoof 2021 PA evaluation set.

{{</citation>}}


### (162/183) Findings of the 2023 ML-SUPERB Challenge: Pre-Training and Evaluation over More Languages and Beyond (Jiatong Shi et al., 2023)

{{<citation>}}

Jiatong Shi, William Chen, Dan Berrebbi, Hsiu-Hsuan Wang, Wei-Ping Huang, En-Pei Hu, Ho-Lam Chuang, Xuankai Chang, Yuxun Tang, Shang-Wen Li, Abdelrahman Mohamed, Hung-yi Lee, Shinji Watanabe. (2023)  
**Findings of the 2023 ML-SUPERB Challenge: Pre-Training and Evaluation over More Languages and Beyond**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2310.05513v1)  

---


**ABSTRACT**  
The 2023 Multilingual Speech Universal Performance Benchmark (ML-SUPERB) Challenge expands upon the acclaimed SUPERB framework, emphasizing self-supervised models in multilingual speech recognition and language identification. The challenge comprises a research track focused on applying ML-SUPERB to specific multilingual subjects, a Challenge Track for model submissions, and a New Language Track where language resource researchers can contribute and evaluate their low-resource language data in the context of the latest progress in multilingual speech recognition. The challenge garnered 12 model submissions and 54 language corpora, resulting in a comprehensive benchmark encompassing 154 languages. The findings indicate that merely scaling models is not the definitive solution for multilingual speech tasks, and a variety of speech/voice types present significant challenges in multilingual speech processing.

{{</citation>}}


### (163/183) AdvSV: An Over-the-Air Adversarial Attack Dataset for Speaker Verification (Li Wang et al., 2023)

{{<citation>}}

Li Wang, Jiaqi Li, Yuhao Luo, Jiahao Zheng, Lei Wang, Hao Li, Ke Xu, Chengfang Fang, Jie Shi, Zhizheng Wu. (2023)  
**AdvSV: An Over-the-Air Adversarial Attack Dataset for Speaker Verification**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Adversarial Attack, Speaker Verification  
[Paper Link](http://arxiv.org/abs/2310.05369v1)  

---


**ABSTRACT**  
It is known that deep neural networks are vulnerable to adversarial attacks. Although Automatic Speaker Verification (ASV) built on top of deep neural networks exhibits robust performance in controlled scenarios, many studies confirm that ASV is vulnerable to adversarial attacks. The lack of a standard dataset is a bottleneck for further research, especially reproducible research. In this study, we developed an open-source adversarial attack dataset for speaker verification research. As an initial step, we focused on the over-the-air attack. An over-the-air adversarial attack involves a perturbation generation algorithm, a loudspeaker, a microphone, and an acoustic environment. The variations in the recording configurations make it very challenging to reproduce previous research. The AdvSV dataset is constructed using the Voxceleb1 Verification test set as its foundation. This dataset employs representative ASV models subjected to adversarial attacks and records adversarial samples to simulate over-the-air attack settings. The scope of the dataset can be easily extended to include more types of adversarial attacks. The dataset will be released to the public under the CC-BY license. In addition, we also provide a detection baseline for reproducible research.

{{</citation>}}


### (164/183) An Initial Investigation of Neural Replay Simulator for Over-the-Air Adversarial Perturbations to Automatic Speaker Verification (Jiaqi Li et al., 2023)

{{<citation>}}

Jiaqi Li, Li Wang, Liumeng Xue, Lei Wang, Zhizheng Wu. (2023)  
**An Initial Investigation of Neural Replay Simulator for Over-the-Air Adversarial Perturbations to Automatic Speaker Verification**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Speaker Verification  
[Paper Link](http://arxiv.org/abs/2310.05354v1)  

---


**ABSTRACT**  
Deep Learning has advanced Automatic Speaker Verification (ASV) in the past few years. Although it is known that deep learning-based ASV systems are vulnerable to adversarial examples in digital access, there are few studies on adversarial attacks in the context of physical access, where a replay process (i.e., over the air) is involved. An over-the-air attack involves a loudspeaker, a microphone, and a replaying environment that impacts the movement of the sound wave. Our initial experiment confirms that the replay process impacts the effectiveness of the over-the-air attack performance. This study performs an initial investigation towards utilizing a neural replay simulator to improve over-the-air adversarial attack robustness. This is achieved by using a neural waveform synthesizer to simulate the replay process when estimating the adversarial perturbations. Experiments conducted on the ASVspoof2019 dataset confirm that the neural replay simulator can considerably increase the success rates of over-the-air adversarial attacks. This raises the concern for adversarial attacks on speaker verification in physical access applications.

{{</citation>}}


## cs.CY (3)



### (165/183) Auditing Gender Analyzers on Text Data (Siddharth D Jaiswal et al., 2023)

{{<citation>}}

Siddharth D Jaiswal, Ankit Kumar Verma, Animesh Mukherjee. (2023)  
**Auditing Gender Analyzers on Text Data**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs.CY  
Keywords: AI, BERT, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.06061v1)  

---


**ABSTRACT**  
AI models have become extremely popular and accessible to the general public. However, they are continuously under the scanner due to their demonstrable biases toward various sections of the society like people of color and non-binary people. In this study, we audit three existing gender analyzers -- uClassify, Readable and HackerFactor, for biases against non-binary individuals. These tools are designed to predict only the cisgender binary labels, which leads to discrimination against non-binary members of the society. We curate two datasets -- Reddit comments (660k) and, Tumblr posts (2.05M) and our experimental evaluation shows that the tools are highly inaccurate with the overall accuracy being ~50% on all platforms. Predictions for non-binary comments on all platforms are mostly female, thus propagating the societal bias that non-binary individuals are effeminate. To address this, we fine-tune a BERT multi-label classifier on the two datasets in multiple combinations, observe an overall performance of ~77% on the most realistically deployable setting and a surprisingly higher performance of 90% for the non-binary class. We also audit ChatGPT using zero-shot prompts on a small dataset (due to high pricing) and observe an average accuracy of 58% for Reddit and Tumblr combined (with overall better results for Reddit).   Thus, we show that existing systems, including highly advanced ones like ChatGPT are biased, and need better audits and moderation and, that such societal biases can be addressed and alleviated through simple off-the-shelf models like BERT trained on more gender inclusive datasets.

{{</citation>}}


### (166/183) An Automated Tool to Detect Suicidal Susceptibility from Social Media Posts (Yasin Dus et al., 2023)

{{<citation>}}

Yasin Dus, Georgiy Nefedov. (2023)  
**An Automated Tool to Detect Suicidal Susceptibility from Social Media Posts**  

---
Primary Category: cs.CY  
Categories: ACM-class: K-4-2, cs-CY, cs.CY  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2310.06056v1)  

---


**ABSTRACT**  
According to the World Health Organization (WHO), approximately 1.4 million individuals died by suicide in 2022. This means that one person dies by suicide every 20 seconds. Globally, suicide ranks as the 10th leading cause of death, while it ranks second for young people aged 15-29. In the year 2022, it was estimated that about 10.5 million suicide attempts occurred. The WHO suggests that alongside each completed suicide, there are many individuals who make attempts. Today, social media is a place where people share their feelings, such as happiness, sadness, anger, and love. This helps us understand how they are thinking or what they might do. This study takes advantage of this opportunity and focuses on developing an automated tool to find if someone may be thinking about harming themselves. It is developed based on the Suicidal-Electra model. We collected datasets of social media posts, processed them, and used them to train and fine-tune the model. Upon evaluating the refined model with a testing dataset, we consistently observed outstanding results. The model demonstrated an impressive accuracy rate of 93% and a commendable F1 score of 0.93. Additionally, we developed an API enabling seamless integration with third-party platforms, enhancing its potential for implementation to address the growing concern of rising suicide rates.

{{</citation>}}


### (167/183) Divide-and-Conquer Dynamics in AI-Driven Disempowerment (Peter S. Park et al., 2023)

{{<citation>}}

Peter S. Park, Max Tegmark. (2023)  
**Divide-and-Conquer Dynamics in AI-Driven Disempowerment**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-LG, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.06009v1)  

---


**ABSTRACT**  
AI companies are attempting to create AI systems that outperform humans at most economically valuable work. Current AI models are already automating away the livelihoods of some artists, actors, and writers. But there is infighting between those who prioritize current harms and future harms. We construct a game-theoretic model of conflict to study the causes and consequences of this disunity. Our model also helps explain why throughout history, stakeholders sharing a common threat have found it advantageous to unite against it, and why the common threat has in turn found it advantageous to divide and conquer.   Under realistic parameter assumptions, our model makes several predictions that find preliminary corroboration in the historical-empirical record. First, current victims of AI-driven disempowerment need the future victims to realize that their interests are also under serious and imminent threat, so that future victims are incentivized to support current victims in solidarity. Second, the movement against AI-driven disempowerment can become more united, and thereby more likely to prevail, if members believe that their efforts will be successful as opposed to futile. Finally, the movement can better unite and prevail if its members are less myopic. Myopic members prioritize their future well-being less than their present well-being, and are thus disinclined to solidarily support current victims today at personal cost, even if this is necessary to counter the shared threat of AI-driven disempowerment.

{{</citation>}}


## cs.CR (2)



### (168/183) LLM for SoC Security: A Paradigm Shift (Dipayan Saha et al., 2023)

{{<citation>}}

Dipayan Saha, Shams Tarek, Katayoon Yahyaei, Sujan Kumar Saha, Jingbo Zhou, Mark Tehranipoor, Farimah Farahmandi. (2023)  
**LLM for SoC Security: A Paradigm Shift**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs.CR  
Keywords: GPT, Language Model, Security, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.06046v1)  

---


**ABSTRACT**  
As the ubiquity and complexity of system-on-chip (SoC) designs increase across electronic devices, the task of incorporating security into an SoC design flow poses significant challenges. Existing security solutions are inadequate to provide effective verification of modern SoC designs due to their limitations in scalability, comprehensiveness, and adaptability. On the other hand, Large Language Models (LLMs) are celebrated for their remarkable success in natural language understanding, advanced reasoning, and program synthesis tasks. Recognizing an opportunity, our research delves into leveraging the emergent capabilities of Generative Pre-trained Transformers (GPTs) to address the existing gaps in SoC security, aiming for a more efficient, scalable, and adaptable methodology. By integrating LLMs into the SoC security verification paradigm, we open a new frontier of possibilities and challenges to ensure the security of increasingly complex SoCs. This paper offers an in-depth analysis of existing works, showcases practical case studies, demonstrates comprehensive experiments, and provides useful promoting guidelines. We also present the achievements, prospects, and challenges of employing LLM in different SoC security verification tasks.

{{</citation>}}


### (169/183) Decoding the Threat Landscape : ChatGPT, FraudGPT, and WormGPT in Social Engineering Attacks (Polra Victor Falade, 2023)

{{<citation>}}

Polra Victor Falade. (2023)  
**Decoding the Threat Landscape : ChatGPT, FraudGPT, and WormGPT in Social Engineering Attacks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.05595v1)  

---


**ABSTRACT**  
In the ever-evolving realm of cybersecurity, the rise of generative AI models like ChatGPT, FraudGPT, and WormGPT has introduced both innovative solutions and unprecedented challenges. This research delves into the multifaceted applications of generative AI in social engineering attacks, offering insights into the evolving threat landscape using the blog mining technique. Generative AI models have revolutionized the field of cyberattacks, empowering malicious actors to craft convincing and personalized phishing lures, manipulate public opinion through deepfakes, and exploit human cognitive biases. These models, ChatGPT, FraudGPT, and WormGPT, have augmented existing threats and ushered in new dimensions of risk. From phishing campaigns that mimic trusted organizations to deepfake technology impersonating authoritative figures, we explore how generative AI amplifies the arsenal of cybercriminals. Furthermore, we shed light on the vulnerabilities that AI-driven social engineering exploits, including psychological manipulation, targeted phishing, and the crisis of authenticity. To counter these threats, we outline a range of strategies, including traditional security measures, AI-powered security solutions, and collaborative approaches in cybersecurity. We emphasize the importance of staying vigilant, fostering awareness, and strengthening regulations in the battle against AI-enhanced social engineering attacks. In an environment characterized by the rapid evolution of AI models and a lack of training data, defending against generative AI threats requires constant adaptation and the collective efforts of individuals, organizations, and governments. This research seeks to provide a comprehensive understanding of the dynamic interplay between generative AI and social engineering attacks, equipping stakeholders with the knowledge to navigate this intricate cybersecurity landscape.

{{</citation>}}


## quant-ph (1)



### (170/183) Learning to Decode the Surface Code with a Recurrent, Transformer-Based Neural Network (Johannes Bausch et al., 2023)

{{<citation>}}

Johannes Bausch, Andrew W Senior, Francisco J H Heras, Thomas Edlich, Alex Davies, Michael Newman, Cody Jones, Kevin Satzinger, Murphy Yuezhen Niu, Sam Blackwell, George Holland, Dvir Kafri, Juan Atalaya, Craig Gidney, Demis Hassabis, Sergio Boixo, Hartmut Neven, Pushmeet Kohli. (2023)  
**Learning to Decode the Surface Code with a Recurrent, Transformer-Based Neural Network**  

---
Primary Category: quant-ph  
Categories: 81P73, 68T07, I-2-0; J-2, cs-LG, quant-ph, quant-ph  
Keywords: Google, Transformer  
[Paper Link](http://arxiv.org/abs/2310.05900v1)  

---


**ABSTRACT**  
Quantum error-correction is a prerequisite for reliable quantum computation. Towards this goal, we present a recurrent, transformer-based neural network which learns to decode the surface code, the leading quantum error-correction code. Our decoder outperforms state-of-the-art algorithmic decoders on real-world data from Google's Sycamore quantum processor for distance 3 and 5 surface codes. On distances up to 11, the decoder maintains its advantage on simulated data with realistic noise including cross-talk, leakage, and analog readout signals, and sustains its accuracy far beyond the 25 cycles it was trained on. Our work illustrates the ability of machine learning to go beyond human-designed algorithms by learning from data directly, highlighting machine learning as a strong contender for decoding in quantum computers.

{{</citation>}}


## eess.AS (1)



### (171/183) Fine-grained Audio-Visual Joint Representations for Multimodal Large Language Models (Guangzhi Sun et al., 2023)

{{<citation>}}

Guangzhi Sun, Wenyi Yu, Changli Tang, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Chao Zhang. (2023)  
**Fine-grained Audio-Visual Joint Representations for Multimodal Large Language Models**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-CV, cs-SD, eess-AS, eess.AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.05863v2)  

---


**ABSTRACT**  
Audio-visual large language models (LLM) have drawn significant attention, yet the fine-grained combination of both input streams is rather under-explored, which is challenging but necessary for LLMs to understand general video inputs. To this end, a fine-grained audio-visual joint representation (FAVOR) learning framework for multimodal LLMs is proposed in this paper, which extends a text-based LLM to simultaneously perceive speech and audio events in the audio input stream and images or videos in the visual input stream, at the frame level. To fuse the audio and visual feature streams into joint representations and to align the joint space with the LLM input embedding space, we propose a causal Q-Former structure with a causal attention module to enhance the capture of causal relations of the audio-visual frames across time. An audio-visual evaluation benchmark (AVEB) is also proposed which comprises six representative single-modal tasks with five cross-modal tasks reflecting audio-visual co-reasoning abilities. While achieving competitive single-modal performance on audio, speech and image tasks in AVEB, FAVOR achieved over 20% accuracy improvements on the video question-answering task when fine-grained information or temporal causal reasoning is required. FAVOR, in addition, demonstrated remarkable video comprehension and reasoning abilities on tasks that are unprecedented by other multimodal LLMs. An interactive demo of FAVOR is available at https://github.com/BriansIDP/AudioVisualLLM.git, and the training code and model checkpoints will be released soon.

{{</citation>}}


## eess.SY (1)



### (172/183) Deep Learning-Based Hurricane Resilient Co-planning of Transmission Lines, Battery Energy Storages and Wind Farms (Mojtaba Moradi-Sepahvand et al., 2023)

{{<citation>}}

Mojtaba Moradi-Sepahvand, Turaj Amraee, Saleh Sadeghi Gougheri. (2023)  
**Deep Learning-Based Hurricane Resilient Co-planning of Transmission Lines, Battery Energy Storages and Wind Farms**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.05814v1)  

---


**ABSTRACT**  
In this paper, a multi-stage model for expansion co-planning of transmission lines, Battery Energy Storages (BESs), and Wind Farms (WFs) is presented considering resilience against extreme weather events. In addition to High Voltage Alternating Current (HVAC) lines, Multi-Terminal Voltage Source Converter (MTVSC) based High Voltage Direct Current (HVDC) lines are planned to reduce the impact of high-risk events. To evaluate the system resilience against hurricanes, probable hurricane speed (HS) scenarios are generated using Monte Carlo Simulation (MCS). The Fragility Curve (FC) concept is utilized for calculating the failure probability of lines due to extreme hurricanes. Based on each hurricane damage, the probable scenarios are incorporated in the proposed model. Renewable Portfolio Standard (RPS) policy is modeled to integrate high penetration of WFs. To deal with the wind power and load demand uncertainties, a Chronological Time-Period Clustering (CTPC) algorithm is introduced for extracting representative hours in each planning stage. A deep learning approach based on Bi-directional Long Short-Term Memory (B-LSTM) networks is presented to forecast the yearly peak loads. The Mixed-Integer Linear Programming (MILP) formulation of the proposed model is solved using a Benders Decomposition (BD) algorithm. A modified IEEE RTS test system is used to evaluate the proposed model effectiveness.

{{</citation>}}


## cs.DM (1)



### (173/183) The Parameterised Complexity of Integer Multicommodity Flow (Hans L. Bodlaender et al., 2023)

{{<citation>}}

Hans L. Bodlaender, Isja Mannens, Jelle J. Oostveen, Sukanya Pandey, Erik Jan van Leeuwen. (2023)  
**The Parameterised Complexity of Integer Multicommodity Flow**  

---
Primary Category: cs.DM  
Categories: cs-CC, cs-DM, cs-DS, cs.DM  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.05784v1)  

---


**ABSTRACT**  
The Integer Multicommodity Flow problem has been studied extensively in the literature. However, from a parameterised perspective, mostly special cases, such as the Disjoint Paths problem, have been considered. Therefore, we investigate the parameterised complexity of the general Integer Multicommodity Flow problem. We show that the decision version of this problem on directed graphs for a constant number of commodities, when the capacities are given in unary, is XNLP-complete with pathwidth as parameter and XALP-complete with treewidth as parameter. When the capacities are given in binary, the problem is NP-complete even for graphs of pathwidth at most 13. We give related results for undirected graphs. These results imply that the problem is unlikely to be fixed-parameter tractable by these parameters.   In contrast, we show that the problem does become fixed-parameter tractable when weighted tree partition width (a variant of tree partition width for edge weighted graphs) is used as parameter.

{{</citation>}}


## cs.NI (2)



### (174/183) RateRL: A Framework for Developing RL-Based Rate Adaptation Algorithms in ns-3 (Ruben Queiros et al., 2023)

{{<citation>}}

Ruben Queiros, Luis Ferreira, Helder Fontes, Rui Campos. (2023)  
**RateRL: A Framework for Developing RL-Based Rate Adaptation Algorithms in ns-3**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05772v1)  

---


**ABSTRACT**  
The increasing complexity of recent Wi-Fi amendments is making the use of traditional algorithms and heuristics unfeasible to address the Rate Adaptation (RA) problem. This is due to the large combination of configuration parameters along with the high variability of the wireless channel. Recently, several works have proposed the usage of Reinforcement Learning (RL) techniques to address the problem. However, the proposed solutions lack sufficient technical explanation. Also, the lack of standard frameworks enabling the reproducibility of results and the limited availability of source code, makes the fair comparison with state of the art approaches a challenge. This paper proposes a framework, named RateRL, that integrates state of the art libraries with the well-known Network Simulator 3 (ns-3) to enable the implementation and evaluation of RL-based RA algorithms. To the best of our knowledge, RateRL is the first tool available to assist researchers during the implementation, validation and evaluation phases of RL-based RA algorithms and enable the fair comparison between competing algorithms.

{{</citation>}}


### (175/183) NetTiSA: Extended IP Flow with Time-series Features for Universal Bandwidth-constrained High-speed Network Traffic Classification (Josef Koumar et al., 2023)

{{<citation>}}

Josef Koumar, Karel Hynek, Jaroslav Pešek, Tomáš Čejka. (2023)  
**NetTiSA: Extended IP Flow with Time-series Features for Universal Bandwidth-constrained High-speed Network Traffic Classification**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.05530v1)  

---


**ABSTRACT**  
Network traffic monitoring based on IP Flows is a standard monitoring approach that can be deployed to various network infrastructures, even the large IPS-based networks connecting millions of people. Since flow records traditionally contain only limited information (addresses, transport ports, and amount of exchanged data), they are also commonly extended for additional features that enable network traffic analysis with high accuracy. Nevertheless, the flow extensions are often too large or hard to compute, which limits their deployment only to smaller-sized networks. This paper proposes a novel extended IP flow called NetTiSA (Network Time Series Analysed), which is based on the analysis of the time series of packet sizes. By thoroughly testing 25 different network classification tasks, we show the broad applicability and high usability of NetTiSA, which often outperforms the best-performing related works. For practical deployment, we also consider the sizes of flows extended for NetTiSA and evaluate the performance impacts of its computation in the flow exporter. The novel feature set proved universal and deployable to high-speed ISP networks with 100\,Gbps lines; thus, it enables accurate and widespread network security protection.

{{</citation>}}


## cs.SE (2)



### (176/183) What Skills Do You Need When Developing Software Using ChatGPT? (Discussion Paper) (Johan Jeuring et al., 2023)

{{<citation>}}

Johan Jeuring, Roel Groot, Hieke Keuning. (2023)  
**What Skills Do You Need When Developing Software Using ChatGPT? (Discussion Paper)**  

---
Primary Category: cs.SE  
Categories: cs-CY, cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.05998v1)  

---


**ABSTRACT**  
Since the release of LLM-based tools such as GitHub Copilot and ChatGPT the media and popular scientific literature, but also journals such as the Communications of the ACM, have been flooded with opinions how these tools will change programming. The opinions range from ``machines will program themselves'', to ``AI does not help programmers''. Of course, these statements are meant to to stir up a discussion, and should be taken with a grain of salt, but we argue that such unfounded statements are potentially harmful. Instead, we propose to investigate which skills are required to develop software using LLM-based tools.   In this paper we report on an experiment in which we explore if Computational Thinking (CT) skills predict the ability to develop software using LLM-based tools. Our results show that the ability to develop software using LLM-based tools can indeed be predicted by the score on a CT assessment. There are many limitations to our experiment, and this paper is also a call to discuss how to approach, preferably experimentally, the question of which skills are required to develop software using LLM-based tools. We propose to rephrase this question to include by what kind of people/programmers, to develop what kind of software using what kind of LLM-based tools.

{{</citation>}}


### (177/183) Quality Assurance of A GPT-based Sentiment Analysis System: Adversarial Review Data Generation and Detection (Tinghui Ouyang et al., 2023)

{{<citation>}}

Tinghui Ouyang, Hoang-Quoc Nguyen-Son, Huy H. Nguyen, Isao Echizen, Yoshiki Seo. (2023)  
**Quality Assurance of A GPT-based Sentiment Analysis System: Adversarial Review Data Generation and Detection**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, Amazon, ChatGPT, GPT, Language Model, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2310.05312v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have been garnering significant attention of AI researchers, especially following the widespread popularity of ChatGPT. However, due to LLMs' intricate architecture and vast parameters, several concerns and challenges regarding their quality assurance require to be addressed. In this paper, a fine-tuned GPT-based sentiment analysis model is first constructed and studied as the reference in AI quality analysis. Then, the quality analysis related to data adequacy is implemented, including employing the content-based approach to generate reasonable adversarial review comments as the wrongly-annotated data, and developing surprise adequacy (SA)-based techniques to detect these abnormal data. Experiments based on Amazon.com review data and a fine-tuned GPT model were implemented. Results were thoroughly discussed from the perspective of AI quality assurance to present the quality analysis of an LLM model on generated adversarial textual data and the effectiveness of using SA on anomaly detection in data quality assurance.

{{</citation>}}


## cs.CE (1)



### (178/183) Logic-guided Deep Reinforcement Learning for Stock Trading (Zhiming Li et al., 2023)

{{<citation>}}

Zhiming Li, Junzhe Jiang, Yushi Cao, Aixin Cui, Bozhi Wu, Bo Li, Yang Liu. (2023)  
**Logic-guided Deep Reinforcement Learning for Stock Trading**  

---
Primary Category: cs.CE  
Categories: cs-AI, cs-CE, cs-PL, cs.CE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.05551v1)  

---


**ABSTRACT**  
Deep reinforcement learning (DRL) has revolutionized quantitative finance by achieving excellent performance without significant manual effort. Whereas we observe that the DRL models behave unstably in a dynamic stock market due to the low signal-to-noise ratio nature of the financial data. In this paper, we propose a novel logic-guided trading framework, termed as SYENS (Program Synthesis-based Ensemble Strategy). Different from the previous state-of-the-art ensemble reinforcement learning strategy which arbitrarily selects the best-performing agent for testing based on a single measurement, our framework proposes regularizing the model's behavior in a hierarchical manner using the program synthesis by sketching paradigm. First, we propose a high-level, domain-specific language (DSL) that is used for the depiction of the market environment and action. Then based on the DSL, a novel program sketch is introduced, which embeds human expert knowledge in a logical manner. Finally, based on the program sketch, we adopt the program synthesis by sketching a paradigm and synthesizing a logical, hierarchical trading strategy. We evaluate SYENS on the 30 Dow Jones stocks under the cash trading and the margin trading settings. Experimental results demonstrate that our proposed framework can significantly outperform the baselines with much higher cumulative return and lower maximum drawdown under both settings.

{{</citation>}}


## cs.SI (1)



### (179/183) Harmful Conspiracies in Temporal Interaction Networks: Understanding the Dynamics of Digital Wildfires through Phase Transitions (Kaspara Skovli Gåsvær et al., 2023)

{{<citation>}}

Kaspara Skovli Gåsvær, Pedro G. Lind, Johannes Langguth, Morten Hjorth-Jensen, Michael Kreil, Daniel Thilo Schroeder. (2023)  
**Harmful Conspiracies in Temporal Interaction Networks: Understanding the Dynamics of Digital Wildfires through Phase Transitions**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.05542v1)  

---


**ABSTRACT**  
Shortly after the first COVID-19 cases became apparent in December 2020, rumors spread on social media suggesting a connection between the virus and the 5G radiation emanating from the recently deployed telecommunications network. In the course of the following weeks, this idea gained increasing popularity, and various alleged explanations for how such a connection manifests emerged. Ultimately, after being amplified by prominent conspiracy theorists, a series of arson attacks on telecommunication equipment follows, concluding with the kidnapping of telecommunication technicians in Peru. In this paper, we study the spread of content related to a conspiracy theory with harmful consequences, a so-called digital wildfire. In particular, we investigate the 5G and COVID-19 misinformation event on Twitter before, during, and after its peak in April and May 2020. For this purpose, we examine the community dynamics in complex temporal interaction networks underlying Twitter user activity. We assess the evolution of such digital wildfires by appropriately defining the temporal dynamics of communication in communities within social networks. We show that, for this specific misinformation event, the number of interactions of the users participating in a digital wildfire, as well as the size of the engaged communities, both follow a power-law distribution. Moreover, our research elucidates the possibility of quantifying the phases of a digital wildfire, as per established literature. We identify one such phase as a critical transition, marked by a shift from sporadic tweets to a global spread event, highlighting the dramatic scaling of misinformation propagation.

{{</citation>}}


## cs.IT (1)



### (180/183) Physical Layer Security in a Private 5G Network for Industrial and Mobility Application (Shivraj Hanumant Gonde et al., 2023)

{{<citation>}}

Shivraj Hanumant Gonde, Christoph Frisch, Svetoslav Duhovnikov, Martin Kubisch, Thomas Meyerhoff, Dominic Schupke. (2023)  
**Physical Layer Security in a Private 5G Network for Industrial and Mobility Application**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.05525v1)  

---


**ABSTRACT**  
Cellular communication technologies such as 5G are deployed on a large scale around the world. Compared to other communication technologies such as WiFi, Bluetooth, or Ultra Wideband, the 5G communication standard describes support for a large variety of use cases, e.g., Internet of Things, vehicular, industrial, and campus-wide communications. An organization can operate a Private 5G network to provide connectivity to devices in their manufacturing environment. Physical Layer Key Generation (PLKG) is a method to generate a symmetric secret on two nodes despite the presence of a potential passive eavesdropper. To the best of our knowledge, this work is one of the first to implement PLKG in a real Private 5G network. Therefore, it highlights the possibility of integrating PLKG in the communication technology highly relevant for industrial applications. This paper exemplifies the establishment of a long-term symmetric key between an aerial vehicle and IT infrastructure both located in a manufacturing environment and communicating via the radio interface of the Private 5G network.

{{</citation>}}


## cs.MM (1)



### (181/183) Robust Image Watermarking based on Cross-Attention and Invariant Domain Learning (Agnibh Dasgupta et al., 2023)

{{<citation>}}

Agnibh Dasgupta, Xin Zhong. (2023)  
**Robust Image Watermarking based on Cross-Attention and Invariant Domain Learning**  

---
Primary Category: cs.MM  
Categories: cs-LG, cs-MM, cs.MM  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.05395v1)  

---


**ABSTRACT**  
Image watermarking involves embedding and extracting watermarks within a cover image, with deep learning approaches emerging to bolster generalization and robustness. Predominantly, current methods employ convolution and concatenation for watermark embedding, while also integrating conceivable augmentation in the training process. This paper explores a robust image watermarking methodology by harnessing cross-attention and invariant domain learning, marking two novel, significant advancements. First, we design a watermark embedding technique utilizing a multi-head cross attention mechanism, enabling information exchange between the cover image and watermark to identify semantically suitable embedding locations. Second, we advocate for learning an invariant domain representation that encapsulates both semantic and noise-invariant information concerning the watermark, shedding light on promising avenues for enhancing image watermarking techniques.

{{</citation>}}


## cs.IR (1)



### (182/183) Augmented Embeddings for Custom Retrievals (Anirudh Khatry et al., 2023)

{{<citation>}}

Anirudh Khatry, Yasharth Bajpai, Priyanshu Gupta, Sumit Gulwani, Ashish Tiwari. (2023)  
**Augmented Embeddings for Custom Retrievals**  

---
Primary Category: cs.IR  
Categories: I-2-6, cs-IR, cs-LG, cs.IR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.05380v1)  

---


**ABSTRACT**  
Information retrieval involves selecting artifacts from a corpus that are most relevant to a given search query. The flavor of retrieval typically used in classical applications can be termed as homogeneous and relaxed, where queries and corpus elements are both natural language (NL) utterances (homogeneous) and the goal is to pick most relevant elements from the corpus in the Top-K, where K is large, such as 10, 25, 50 or even 100 (relaxed). Recently, retrieval is being used extensively in preparing prompts for large language models (LLMs) to enable LLMs to perform targeted tasks. These new applications of retrieval are often heterogeneous and strict -- the queries and the corpus contain different kinds of entities, such as NL and code, and there is a need for improving retrieval at Top-K for small values of K, such as K=1 or 3 or 5. Current dense retrieval techniques based on pretrained embeddings provide a general-purpose and powerful approach for retrieval, but they are oblivious to task-specific notions of similarity of heterogeneous artifacts. We introduce Adapted Dense Retrieval, a mechanism to transform embeddings to enable improved task-specific, heterogeneous and strict retrieval. Adapted Dense Retrieval works by learning a low-rank residual adaptation of the pretrained black-box embedding. We empirically validate our approach by showing improvements over the state-of-the-art general-purpose embeddings-based baseline.

{{</citation>}}


## cs.DB (1)



### (183/183) ALECE: An Attention-based Learned Cardinality Estimator for SPJ Queries on Dynamic Workloads (Extended) (Pengfei Li et al., 2023)

{{<citation>}}

Pengfei Li, Wenqing Wei, Rong Zhu, Bolin Ding, Jingren Zhou, Hua Lu. (2023)  
**ALECE: An Attention-based Learned Cardinality Estimator for SPJ Queries on Dynamic Workloads (Extended)**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.05349v2)  

---


**ABSTRACT**  
For efficient query processing, DBMS query optimizers have for decades relied on delicate cardinality estimation methods. In this work, we propose an Attention-based LEarned Cardinality Estimator (ALECE for short) for SPJ queries. The core idea is to discover the implicit relationships between queries and underlying dynamic data using attention mechanisms in ALECE's two modules that are built on top of carefully designed featurizations for data and queries. In particular, from all attributes in the database, the data-encoder module obtains organic and learnable aggregations which implicitly represent correlations among the attributes, whereas the query-analyzer module builds a bridge between the query featurizations and the data aggregations to predict the query's cardinality. We experimentally evaluate ALECE on multiple dynamic workloads. The results show that ALECE enables PostgreSQL's optimizer to achieve nearly optimal performance, clearly outperforming its built-in cardinality estimator and other alternatives.

{{</citation>}}
