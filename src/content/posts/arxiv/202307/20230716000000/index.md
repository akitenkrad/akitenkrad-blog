---
draft: false
title: "arXiv @ 2023.07.16"
date: 2023-07-16
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.16"
    identifier: arxiv_20230716
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.AI (2)](#csai-2)
- [cs.LG (9)](#cslg-9)
- [cs.CL (11)](#cscl-11)
- [cs.CC (1)](#cscc-1)
- [cs.CV (19)](#cscv-19)
- [eess.IV (2)](#eessiv-2)
- [cs.SI (3)](#cssi-3)
- [cs.DC (1)](#csdc-1)
- [econ.GN (1)](#econgn-1)
- [quant-ph (1)](#quant-ph-1)
- [cs.IT (1)](#csit-1)
- [physics.geo-ph (1)](#physicsgeo-ph-1)
- [cs.IR (1)](#csir-1)
- [cs.SE (1)](#csse-1)
- [cs.HC (1)](#cshc-1)
- [cs.AR (1)](#csar-1)
- [cs.CR (1)](#cscr-1)

## cs.AI (2)



### (1/57) Credit Assignment: Challenges and Opportunities in Developing Human-like AI Agents (Thuy Ngoc Nguyen et al., 2023)

{{<citation>}}

Thuy Ngoc Nguyen, Chase McDonald, Cleotilde Gonzalez. (2023)  
**Credit Assignment: Challenges and Opportunities in Developing Human-like AI Agents**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08171v1)  

---


**ABSTRACT**  
Temporal credit assignment is crucial for learning and skill development in natural and artificial intelligence. While computational methods like the TD approach in reinforcement learning have been proposed, it's unclear if they accurately represent how humans handle feedback delays. Cognitive models intend to represent the mental steps by which humans solve problems and perform a number of tasks, but limited research in cognitive science has addressed the credit assignment problem in humans and cognitive models. Our research uses a cognitive model based on a theory of decisions from experience, Instance-Based Learning Theory (IBLT), to test different credit assignment mechanisms in a goal-seeking navigation task with varying levels of decision complexity. Instance-Based Learning (IBL) models simulate the process of making sequential choices with different credit assignment mechanisms, including a new IBL-TD model that combines the IBL decision mechanism with the TD approach. We found that (1) An IBL model that gives equal credit assignment to all decisions is able to match human performance better than other models, including IBL-TD and Q-learning; (2) IBL-TD and Q-learning models underperform compared to humans initially, but eventually, they outperform humans; (3) humans are influenced by decision complexity, while models are not. Our study provides insights into the challenges of capturing human behavior and the potential opportunities to use these models in future AI systems to support human activities.

{{</citation>}}


### (2/57) MinT: Boosting Generalization in Mathematical Reasoning via Multi-View Fine-Tuning (Zhenwen Liang et al., 2023)

{{<citation>}}

Zhenwen Liang, Dian Yu, Xiaoman Pan, Wenlin Yao, Qingkai Zeng, Xiangliang Zhang, Dong Yu. (2023)  
**MinT: Boosting Generalization in Mathematical Reasoning via Multi-View Fine-Tuning**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-CL, cs.AI  
Keywords: LLaMA, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.07951v1)  

---


**ABSTRACT**  
Reasoning in mathematical domains remains a significant challenge for relatively small language models (LMs). Many current methods focus on specializing LMs in mathematical reasoning and rely heavily on knowledge distillation from powerful but inefficient large LMs (LLMs). In this work, we explore a new direction that avoids over-reliance on LLM teachers, introducing a multi-view fine-tuning method that efficiently exploits existing mathematical problem datasets with diverse annotation styles. Our approach uniquely considers the various annotation formats as different "views" and leverages them in training the model. By postpending distinct instructions to input questions, models can learn to generate solutions in diverse formats in a flexible manner. Experimental results show that our strategy enables a LLaMA-7B model to outperform prior approaches that utilize knowledge distillation, as well as carefully established baselines. Additionally, the proposed method grants the models promising generalization ability across various views and datasets, and the capability to learn from inaccurate or incomplete noisy data. We hope our multi-view training paradigm could inspire future studies in other machine reasoning domains.

{{</citation>}}


## cs.LG (9)



### (3/57) Discovering User Types: Mapping User Traits by Task-Specific Behaviors in Reinforcement Learning (L. L. Ankile et al., 2023)

{{<citation>}}

L. L. Ankile, B. S. Ham, K. Mao, E. Shin, S. Swaroop, F. Doshi-Velez, W. Pan. (2023)  
**Discovering User Types: Mapping User Traits by Task-Specific Behaviors in Reinforcement Learning**  

---
Primary Category: cs.LG
Categories: cs-HC, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08169v1)  

---


**ABSTRACT**  
When assisting human users in reinforcement learning (RL), we can represent users as RL agents and study key parameters, called \emph{user traits}, to inform intervention design. We study the relationship between user behaviors (policy classes) and user traits. Given an environment, we introduce an intuitive tool for studying the breakdown of "user types": broad sets of traits that result in the same behavior. We show that seemingly different real-world environments admit the same set of user types and formalize this observation as an equivalence relation defined on environments. By transferring intervention design between environments within the same equivalence class, we can help rapidly personalize interventions.

{{</citation>}}


### (4/57) Feedback is All You Need: Real-World Reinforcement Learning with Approximate Physics-Based Models (Tyler Westenbroek et al., 2023)

{{<citation>}}

Tyler Westenbroek, Jacob Levy, David Fridovich-Keil. (2023)  
**Feedback is All You Need: Real-World Reinforcement Learning with Approximate Physics-Based Models**  

---
Primary Category: cs.LG
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08168v1)  

---


**ABSTRACT**  
We focus on developing efficient and reliable policy optimization strategies for robot learning with real-world data. In recent years, policy gradient methods have emerged as a promising paradigm for training control policies in simulation. However, these approaches often remain too data inefficient or unreliable to train on real robotic hardware. In this paper we introduce a novel policy gradient-based policy optimization framework which systematically leverages a (possibly highly simplified) first-principles model and enables learning precise control policies with limited amounts of real-world data. Our approach $1)$ uses the derivatives of the model to produce sample-efficient estimates of the policy gradient and $2)$ uses the model to design a low-level tracking controller, which is embedded in the policy class. Theoretical analysis provides insight into how the presence of this feedback controller addresses overcomes key limitations of stand-alone policy gradient methods, while hardware experiments with a small car and quadruped demonstrate that our approach can learn precise control strategies reliably and with only minutes of real-world data.

{{</citation>}}


### (5/57) Tangent Transformers for Composition, Privacy and Removal (Tian Yu Liu et al., 2023)

{{<citation>}}

Tian Yu Liu, Aditya Golatkar, Stefano Soatto. (2023)  
**Tangent Transformers for Composition, Privacy and Removal**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.08122v2)  

---


**ABSTRACT**  
We introduce Tangent Attention Fine-Tuning (TAFT), a method for fine-tuning linearized transformers obtained by computing a First-order Taylor Expansion around a pre-trained initialization. We show that the Jacobian-Vector Product resulting from linearization can be computed efficiently in a single forward pass, reducing training and inference cost to the same order of magnitude as its original non-linear counterpart, while using the same number of parameters. Furthermore, we show that, when applied to various downstream visual classification tasks, the resulting Tangent Transformer fine-tuned with TAFT can perform comparably with fine-tuning the original non-linear network. Since Tangent Transformers are linear with respect to the new set of weights, and the resulting fine-tuning loss is convex, we show that TAFT enjoys several advantages compared to non-linear fine-tuning when it comes to model composition, parallel training, machine unlearning, and differential privacy.

{{</citation>}}


### (6/57) POMDP inference and robust solution via deep reinforcement learning: An application to railway optimal maintenance (Giacomo Arcieri et al., 2023)

{{<citation>}}

Giacomo Arcieri, Cyprien Hoelzl, Oliver Schwery, Daniel Straub, Konstantinos G. Papakonstantinou, Eleni Chatzi. (2023)  
**POMDP inference and robust solution via deep reinforcement learning: An application to railway optimal maintenance**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08082v1)  

---


**ABSTRACT**  
Partially Observable Markov Decision Processes (POMDPs) can model complex sequential decision-making problems under stochastic and uncertain environments. A main reason hindering their broad adoption in real-world applications is the lack of availability of a suitable POMDP model or a simulator thereof. Available solution algorithms, such as Reinforcement Learning (RL), require the knowledge of the transition dynamics and the observation generating process, which are often unknown and non-trivial to infer. In this work, we propose a combined framework for inference and robust solution of POMDPs via deep RL. First, all transition and observation model parameters are jointly inferred via Markov Chain Monte Carlo sampling of a hidden Markov model, which is conditioned on actions, in order to recover full posterior distributions from the available data. The POMDP with uncertain parameters is then solved via deep RL techniques with the parameter distributions incorporated into the solution via domain randomization, in order to develop solutions that are robust to model uncertainty. As a further contribution, we compare the use of transformers and long short-term memory networks, which constitute model-free RL solutions, with a model-based/model-free hybrid approach. We apply these methods to the real-world problem of optimal maintenance planning for railway assets.

{{</citation>}}


### (7/57) Magnetic Field-Based Reward Shaping for Goal-Conditioned Reinforcement Learning (Hongyu Ding et al., 2023)

{{<citation>}}

Hongyu Ding, Yuanze Tang, Qing Wu, Bo Wang, Chunlin Chen, Zhi Wang. (2023)  
**Magnetic Field-Based Reward Shaping for Goal-Conditioned Reinforcement Learning**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.08033v1)  

---


**ABSTRACT**  
Goal-conditioned reinforcement learning (RL) is an interesting extension of the traditional RL framework, where the dynamic environment and reward sparsity can cause conventional learning algorithms to fail. Reward shaping is a practical approach to improving sample efficiency by embedding human domain knowledge into the learning process. Existing reward shaping methods for goal-conditioned RL are typically built on distance metrics with a linear and isotropic distribution, which may fail to provide sufficient information about the ever-changing environment with high complexity. This paper proposes a novel magnetic field-based reward shaping (MFRS) method for goal-conditioned RL tasks with dynamic target and obstacles. Inspired by the physical properties of magnets, we consider the target and obstacles as permanent magnets and establish the reward function according to the intensity values of the magnetic field generated by these magnets. The nonlinear and anisotropic distribution of the magnetic field intensity can provide more accessible and conducive information about the optimization landscape, thus introducing a more sophisticated magnetic reward compared to the distance-based setting. Further, we transform our magnetic reward to the form of potential-based reward shaping by learning a secondary potential function concurrently to ensure the optimal policy invariance of our method. Experiments results in both simulated and real-world robotic manipulation tasks demonstrate that MFRS outperforms relevant existing methods and effectively improves the sample efficiency of RL algorithms in goal-conditioned tasks with various dynamics of the target and obstacles.

{{</citation>}}


### (8/57) A Survey of Techniques for Optimizing Transformer Inference (Krishna Teja Chitty-Venkata et al., 2023)

{{<citation>}}

Krishna Teja Chitty-Venkata, Sparsh Mittal, Murali Emani, Venkatram Vishwanath, Arun K. Somani. (2023)  
**A Survey of Techniques for Optimizing Transformer Inference**  

---
Primary Category: cs.LG
Categories: cs-AR, cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: BERT, ChatGPT, Computer Vision, GPT, NLP, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2307.07982v1)  

---


**ABSTRACT**  
Recent years have seen a phenomenal rise in performance and applications of transformer neural networks. The family of transformer networks, including Bidirectional Encoder Representations from Transformer (BERT), Generative Pretrained Transformer (GPT) and Vision Transformer (ViT), have shown their effectiveness across Natural Language Processing (NLP) and Computer Vision (CV) domains. Transformer-based networks such as ChatGPT have impacted the lives of common men. However, the quest for high predictive performance has led to an exponential increase in transformers' memory and compute footprint. Researchers have proposed techniques to optimize transformer inference at all levels of abstraction. This paper presents a comprehensive survey of techniques for optimizing the inference phase of transformer networks. We survey techniques such as knowledge distillation, pruning, quantization, neural architecture search and lightweight network design at the algorithmic level. We further review hardware-level optimization techniques and the design of novel hardware accelerators for transformers. We summarize the quantitative results on the number of parameters/FLOPs and accuracy of several models/techniques to showcase the tradeoff exercised by them. We also outline future directions in this rapidly evolving field of research. We believe that this survey will educate both novice and seasoned researchers and also spark a plethora of research efforts in this field.

{{</citation>}}


### (9/57) Automated Polynomial Filter Learning for Graph Neural Networks (Wendi Yu et al., 2023)

{{<citation>}}

Wendi Yu, Zhichao Hou, Xiaorui Liu. (2023)  
**Automated Polynomial Filter Learning for Graph Neural Networks**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.07956v1)  

---


**ABSTRACT**  
Polynomial graph filters have been widely used as guiding principles in the design of Graph Neural Networks (GNNs). Recently, the adaptive learning of the polynomial graph filters has demonstrated promising performance for modeling graph signals on both homophilic and heterophilic graphs, owning to their flexibility and expressiveness. In this work, we conduct a novel preliminary study to explore the potential and limitations of polynomial graph filter learning approaches, revealing a severe overfitting issue. To improve the effectiveness of polynomial graph filters, we propose Auto-Polynomial, a novel and general automated polynomial graph filter learning framework that efficiently learns better filters capable of adapting to various complex graph signals. Comprehensive experiments and ablation studies demonstrate significant and consistent performance improvements on both homophilic and heterophilic graphs across multiple learning settings considering various labeling ratios, which unleashes the potential of polynomial filter learning.

{{</citation>}}


### (10/57) On the Robustness of Split Learning against Adversarial Attacks (Mingyuan Fan et al., 2023)

{{<citation>}}

Mingyuan Fan, Cen Chen, Chengyu Wang, Wenmeng Zhou, Jun Huang. (2023)  
**On the Robustness of Split Learning against Adversarial Attacks**  

---
Primary Category: cs.LG
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2307.07916v2)  

---


**ABSTRACT**  
Split learning enables collaborative deep learning model training while preserving data privacy and model security by avoiding direct sharing of raw data and model details (i.e., sever and clients only hold partial sub-networks and exchange intermediate computations). However, existing research has mainly focused on examining its reliability for privacy protection, with little investigation into model security. Specifically, by exploring full models, attackers can launch adversarial attacks, and split learning can mitigate this severe threat by only disclosing part of models to untrusted servers.This paper aims to evaluate the robustness of split learning against adversarial attacks, particularly in the most challenging setting where untrusted servers only have access to the intermediate layers of the model.Existing adversarial attacks mostly focus on the centralized setting instead of the collaborative setting, thus, to better evaluate the robustness of split learning, we develop a tailored attack called SPADV, which comprises two stages: 1) shadow model training that addresses the issue of lacking part of the model and 2) local adversarial attack that produces adversarial examples to evaluate.The first stage only requires a few unlabeled non-IID data, and, in the second stage, SPADV perturbs the intermediate output of natural samples to craft the adversarial ones. The overall cost of the proposed attack process is relatively low, yet the empirical attack effectiveness is significantly high, demonstrating the surprising vulnerability of split learning to adversarial attacks.

{{</citation>}}


### (11/57) Predicting mechanical properties of Carbon Nanotube (CNT) images Using Multi-Layer Synthetic Finite Element Model Simulations (Kaveh Safavigerdini et al., 2023)

{{<citation>}}

Kaveh Safavigerdini, Koundinya Nouduri, Ramakrishna Surya, Andrew Reinhard, Zach Quinlan, Filiz Bunyak, Matthew R. Maschmann, Kannappan Palaniappan. (2023)  
**Predicting mechanical properties of Carbon Nanotube (CNT) images Using Multi-Layer Synthetic Finite Element Model Simulations**  

---
Primary Category: cs.LG
Categories: cond-mat-mtrl-sci, cs-CV, cs-LG, cs.LG, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07912v1)  

---


**ABSTRACT**  
We present a pipeline for predicting mechanical properties of vertically-oriented carbon nanotube (CNT) forest images using a deep learning model for artificial intelligence (AI)-based materials discovery. Our approach incorporates an innovative data augmentation technique that involves the use of multi-layer synthetic (MLS) or quasi-2.5D images which are generated by blending 2D synthetic images. The MLS images more closely resemble 3D synthetic and real scanning electron microscopy (SEM) images of CNTs but without the computational cost of performing expensive 3D simulations or experiments. Mechanical properties such as stiffness and buckling load for the MLS images are estimated using a physics-based model. The proposed deep learning architecture, CNTNeXt, builds upon our previous CNTNet neural network, using a ResNeXt feature representation followed by random forest regression estimator. Our machine learning approach for predicting CNT physical properties by utilizing a blended set of synthetic images is expected to outperform single synthetic image-based learning when it comes to predicting mechanical properties of real scanning electron microscopy images. This has the potential to accelerate understanding and control of CNT forest self-assembly for diverse applications.

{{</citation>}}


## cs.CL (11)



### (12/57) Assessing the Quality of Multiple-Choice Questions Using GPT-4 and Rule-Based Methods (Steven Moore et al., 2023)

{{<citation>}}

Steven Moore, Huy A. Nguyen, Tianying Chen, John Stamper. (2023)  
**Assessing the Quality of Multiple-Choice Questions Using GPT-4 and Rule-Based Methods**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2307.08161v1)  

---


**ABSTRACT**  
Multiple-choice questions with item-writing flaws can negatively impact student learning and skew analytics. These flaws are often present in student-generated questions, making it difficult to assess their quality and suitability for classroom usage. Existing methods for evaluating multiple-choice questions often focus on machine readability metrics, without considering their intended use within course materials and their pedagogical implications. In this study, we compared the performance of a rule-based method we developed to a machine-learning based method utilizing GPT-4 for the task of automatically assessing multiple-choice questions based on 19 common item-writing flaws. By analyzing 200 student-generated questions from four different subject areas, we found that the rule-based method correctly detected 91% of the flaws identified by human annotators, as compared to 79% by GPT-4. We demonstrated the effectiveness of the two methods in identifying common item-writing flaws present in the student-generated questions across different subject areas. The rule-based method can accurately and efficiently evaluate multiple-choice questions from multiple domains, outperforming GPT-4 and going beyond existing metrics that do not account for the educational use of such questions. Finally, we discuss the potential for using these automated methods to improve the quality of questions based on the identified flaws.

{{</citation>}}


### (13/57) The Potential and Pitfalls of using a Large Language Model such as ChatGPT or GPT-4 as a Clinical Assistant (Jingqing Zhang et al., 2023)

{{<citation>}}

Jingqing Zhang, Kai Sun, Akshay Jagadeesh, Mahta Ghahfarokhi, Deepa Gupta, Ashok Gupta, Vibhor Gupta, Yike Guo. (2023)  
**The Potential and Pitfalls of using a Large Language Model such as ChatGPT or GPT-4 as a Clinical Assistant**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Clinical, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.08152v1)  

---


**ABSTRACT**  
Recent studies have demonstrated promising performance of ChatGPT and GPT-4 on several medical domain tasks. However, none have assessed its performance using a large-scale real-world electronic health record database, nor have evaluated its utility in providing clinical diagnostic assistance for patients across a full range of disease presentation. We performed two analyses using ChatGPT and GPT-4, one to identify patients with specific medical diagnoses using a real-world large electronic health record database and the other, in providing diagnostic assistance to healthcare workers in the prospective evaluation of hypothetical patients. Our results show that GPT-4 across disease classification tasks with chain of thought and few-shot prompting can achieve performance as high as 96% F1 scores. For patient assessment, GPT-4 can accurately diagnose three out of four times. However, there were mentions of factually incorrect statements, overlooking crucial medical findings, recommendations for unnecessary investigations and overtreatment. These issues coupled with privacy concerns, make these models currently inadequate for real world clinical use. However, limited data and time needed for prompt engineering in comparison to configuration of conventional machine learning workflows highlight their potential for scalability across healthcare applications.

{{</citation>}}


### (14/57) It's All Relative: Interpretable Models for Scoring Bias in Documents (Aswin Suresh et al., 2023)

{{<citation>}}

Aswin Suresh, Chi-Hsuan Wu, Matthias Grossglauser. (2023)  
**It's All Relative: Interpretable Models for Scoring Bias in Documents**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.08139v1)  

---


**ABSTRACT**  
We propose an interpretable model to score the bias present in web documents, based only on their textual content. Our model incorporates assumptions reminiscent of the Bradley-Terry axioms and is trained on pairs of revisions of the same Wikipedia article, where one version is more biased than the other. While prior approaches based on absolute bias classification have struggled to obtain a high accuracy for the task, we are able to develop a useful model for scoring bias by learning to perform pairwise comparisons of bias accurately. We show that we can interpret the parameters of the trained model to discover the words most indicative of bias. We also apply our model in three different settings - studying the temporal evolution of bias in Wikipedia articles, comparing news sources based on bias, and scoring bias in law amendments. In each case, we demonstrate that the outputs of the model can be explained and validated, even for the two domains that are outside the training-data domain. We also use the model to compare the general level of bias between domains, where we see that legal texts are the least biased and news media are the most biased, with Wikipedia articles in between. Given its high performance, simplicity, interpretability, and wide applicability, we hope the model will be useful for a large community, including Wikipedia and news editors, political and social scientists, and the general public.

{{</citation>}}


### (15/57) Disco-Bench: A Discourse-Aware Evaluation Benchmark for Language Modelling (Longyue Wang et al., 2023)

{{<citation>}}

Longyue Wang, Zefeng Du, Donghuai Liu, Cai Deng, Dian Yu, Haiyun Jiang, Yan Wang, Leyang Cui, Shuming Shi, Zhaopeng Tu. (2023)  
**Disco-Bench: A Discourse-Aware Evaluation Benchmark for Language Modelling**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2307.08074v1)  

---


**ABSTRACT**  
Modeling discourse -- the linguistic phenomena that go beyond individual sentences, is a fundamental yet challenging aspect of natural language processing (NLP). However, existing evaluation benchmarks primarily focus on the evaluation of inter-sentence properties and overlook critical discourse phenomena that cross sentences. To bridge the gap, we propose Disco-Bench, a benchmark that can evaluate intra-sentence discourse properties across a diverse set of NLP tasks, covering understanding, translation, and generation. Disco-Bench consists of 9 document-level testsets in the literature domain, which contain rich discourse phenomena (e.g. cohesion and coherence) in Chinese and/or English. For linguistic analysis, we also design a diagnostic test suite that can examine whether the target models learn discourse knowledge. We totally evaluate 20 general-, in-domain and commercial models based on Transformer, advanced pretraining architectures and large language models (LLMs). Our results show (1) the challenge and necessity of our evaluation benchmark; (2) fine-grained pretraining based on literary document-level training data consistently improves the modeling of discourse information. We will release the datasets, pretrained models, and leaderboard, which we hope can significantly facilitate research in this field: https://github.com/longyuewangdcu/Disco-Bench.

{{</citation>}}


### (16/57) Do Emergent Abilities Exist in Quantized Large Language Models: An Empirical Study (Peiyu Liu et al., 2023)

{{<citation>}}

Peiyu Liu, Zikang Liu, Ze-Feng Gao, Dawei Gao, Wayne Xin Zhao, Yaliang Li, Bolin Ding, Ji-Rong Wen. (2023)  
**Do Emergent Abilities Exist in Quantized Large Language Models: An Empirical Study**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.08072v1)  

---


**ABSTRACT**  
Despite the superior performance, Large Language Models~(LLMs) require significant computational resources for deployment and use. To overcome this issue, quantization methods have been widely applied to reduce the memory footprint of LLMs as well as increasing the inference rate. However, a major challenge is that low-bit quantization methods often lead to performance degradation. It is important to understand how quantization impacts the capacity of LLMs. Different from previous studies focused on overall performance, this work aims to investigate the impact of quantization on \emph{emergent abilities}, which are important characteristics that distinguish LLMs from small language models. Specially, we examine the abilities of in-context learning, chain-of-thought reasoning, and instruction-following in quantized LLMs. Our empirical experiments show that these emergent abilities still exist in 4-bit quantization models, while 2-bit models encounter severe performance degradation on the test of these abilities. To improve the performance of low-bit models, we conduct two special experiments: (1) fine-gained impact analysis that studies which components (or substructures) are more sensitive to quantization, and (2) performance compensation through model fine-tuning. Our work derives a series of important findings to understand the impact of quantization on emergent abilities, and sheds lights on the possibilities of extremely low-bit quantization for LLMs.

{{</citation>}}


### (17/57) A Neural-Symbolic Approach Towards Identifying Grammatically Correct Sentences (Nicos Isaak, 2023)

{{<citation>}}

Nicos Isaak. (2023)  
**A Neural-Symbolic Approach Towards Identifying Grammatically Correct Sentences**  

---
Primary Category: cs.CL
Categories: I-2-0; I-2-3; I-2-7; I-5-1, cs-AI, cs-CL, cs.CL  
Keywords: AI, NLP  
[Paper Link](http://arxiv.org/abs/2307.08036v1)  

---


**ABSTRACT**  
Textual content around us is growing on a daily basis. Numerous articles are being written as we speak on online newspapers, blogs, or social media. Similarly, recent advances in the AI field, like language models or traditional classic AI approaches, are utilizing all the above to improve their learned representation to tackle NLP challenges with human-like accuracy. It is commonly accepted that it is crucial to have access to well-written text from valid sources to tackle challenges like text summarization, question-answering, machine translation, or even pronoun resolution. For instance, to summarize well, one needs to select the most important sentences in order to concatenate them to form the summary. However, what happens if we do not have access to well-formed English sentences or even non-valid sentences? Despite the importance of having access to well-written sentences, figuring out ways to validate them is still an open area of research. To address this problem, we present a simplified way to validate English sentences through a novel neural-symbolic approach. Lately, neural-symbolic approaches have triggered an increasing interest towards tackling various NLP challenges, as they are demonstrating their effectiveness as a central component in various AI systems. Through combining Classic with Modern AI, which involves the blending of grammatical and syntactical rules with language models, we effectively tackle the Corpus of Linguistic Acceptability (COLA), a task that shows whether or not a sequence of words is an English grammatical sentence. Among others, undertaken experiments effectively show that blending symbolic and non-symbolic systems helps the former provide insights about the latter's accuracy results.

{{</citation>}}


### (18/57) Facilitating Multi-turn Emotional Support Conversation with Positive Emotion Elicitation: A Reinforcement Learning Approach (Jinfeng Zhou et al., 2023)

{{<citation>}}

Jinfeng Zhou, Zhuang Chen, Bo Wang, Minlie Huang. (2023)  
**Facilitating Multi-turn Emotional Support Conversation with Positive Emotion Elicitation: A Reinforcement Learning Approach**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.07994v1)  

---


**ABSTRACT**  
Emotional support conversation (ESC) aims to provide emotional support (ES) to improve one's mental state. Existing works stay at fitting grounded responses and responding strategies (e.g., question), which ignore the effect on ES and lack explicit goals to guide emotional positive transition. To this end, we introduce a new paradigm to formalize multi-turn ESC as a process of positive emotion elicitation. Addressing this task requires finely adjusting the elicitation intensity in ES as the conversation progresses while maintaining conversational goals like coherence. In this paper, we propose Supporter, a mixture-of-expert-based reinforcement learning model, and well design ES and dialogue coherence rewards to guide policy's learning for responding. Experiments verify the superiority of Supporter in achieving positive emotion elicitation during responding while maintaining conversational goals including coherence.

{{</citation>}}


### (19/57) SentimentGPT: Exploiting GPT for Advanced Sentiment Analysis and its Departure from Current Machine Learning (Kiana Kheiri et al., 2023)

{{<citation>}}

Kiana Kheiri, Hamid Karimi. (2023)  
**SentimentGPT: Exploiting GPT for Advanced Sentiment Analysis and its Departure from Current Machine Learning**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs-LG, cs-SI, cs.CL  
Keywords: GPT, GPT-3.5, Sentiment Analysis, Transformer  
[Paper Link](http://arxiv.org/abs/2307.10234v1)  

---


**ABSTRACT**  
This study presents a thorough examination of various Generative Pretrained Transformer (GPT) methodologies in sentiment analysis, specifically in the context of Task 4 on the SemEval 2017 dataset. Three primary strategies are employed: 1) prompt engineering using the advanced GPT-3.5 Turbo, 2) fine-tuning GPT models, and 3) an inventive approach to embedding classification. The research yields detailed comparative insights among these strategies and individual GPT models, revealing their unique strengths and potential limitations. Additionally, the study compares these GPT-based methodologies with other contemporary, high-performing models previously used with the same dataset. The results illustrate the significant superiority of the GPT approaches in terms of predictive performance, more than 22% in F1-score compared to the state-of-the-art. Further, the paper addresses common challenges in sentiment analysis tasks, such as understanding context and detecting sarcasm. It underscores the enhanced capabilities of the GPT models to effectively navigate these complexities. Collectively, these findings highlight the promising potential of GPT models in sentiment analysis, setting the stage for future research in this field. The code can be found at https://github.com/DSAatUSU/SentimentGPT.

{{</citation>}}


### (20/57) Unifying Token and Span Level Supervisions for Few-Shot Sequence Labeling (Zifeng Cheng et al., 2023)

{{<citation>}}

Zifeng Cheng, Qingyu Zhou, Zhiwei Jiang, Xuemin Zhao, Yunbo Cao, Qing Gu. (2023)  
**Unifying Token and Span Level Supervisions for Few-Shot Sequence Labeling**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2307.07946v2)  

---


**ABSTRACT**  
Few-shot sequence labeling aims to identify novel classes based on only a few labeled samples. Existing methods solve the data scarcity problem mainly by designing token-level or span-level labeling models based on metric learning. However, these methods are only trained at a single granularity (i.e., either token level or span level) and have some weaknesses of the corresponding granularity. In this paper, we first unify token and span level supervisions and propose a Consistent Dual Adaptive Prototypical (CDAP) network for few-shot sequence labeling. CDAP contains the token-level and span-level networks, jointly trained at different granularities. To align the outputs of two networks, we further propose a consistent loss to enable them to learn from each other. During the inference phase, we propose a consistent greedy inference algorithm that first adjusts the predicted probability and then greedily selects non-overlapping spans with maximum probability. Extensive experiments show that our model achieves new state-of-the-art results on three benchmark datasets.

{{</citation>}}


### (21/57) GeoGPT: Understanding and Processing Geospatial Tasks through An Autonomous GPT (Yifan Zhang et al., 2023)

{{<citation>}}

Yifan Zhang, Cheng Wei, Shangyou Wu, Zhengting He, Wenhao Yu. (2023)  
**GeoGPT: Understanding and Processing Geospatial Tasks through An Autonomous GPT**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2307.07930v1)  

---


**ABSTRACT**  
Decision-makers in GIS need to combine a series of spatial algorithms and operations to solve geospatial tasks. For example, in the task of facility siting, the Buffer tool is usually first used to locate areas close or away from some specific entities; then, the Intersect or Erase tool is used to select candidate areas satisfied multiple requirements. Though professionals can easily understand and solve these geospatial tasks by sequentially utilizing relevant tools, it is difficult for non-professionals to handle these problems. Recently, Generative Pre-trained Transformer (e.g., ChatGPT) presents strong performance in semantic understanding and reasoning. Especially, AutoGPT can further extend the capabilities of large language models (LLMs) by automatically reasoning and calling externally defined tools. Inspired by these studies, we attempt to lower the threshold of non-professional users to solve geospatial tasks by integrating the semantic understanding ability inherent in LLMs with mature tools within the GIS community. Specifically, we develop a new framework called GeoGPT that can conduct geospatial data collection, processing, and analysis in an autonomous manner with the instruction of only natural language. In other words, GeoGPT is used to understand the demands of non-professional users merely based on input natural language descriptions, and then think, plan, and execute defined GIS tools to output final effective results. Several cases including geospatial data crawling, spatial query, facility siting, and mapping validate the effectiveness of our framework. Though limited cases are presented in this paper, GeoGPT can be further extended to various tasks by equipping with more GIS tools, and we think the paradigm of "foundational plus professional" implied in GeoGPT provides an effective way to develop next-generation GIS in this era of large foundation models.

{{</citation>}}


### (22/57) Cross-Lingual NER for Financial Transaction Data in Low-Resource Languages (Sunisth Kumar et al., 2023)

{{<citation>}}

Sunisth Kumar, Davide Liu, Alexandre Boulenger. (2023)  
**Cross-Lingual NER for Financial Transaction Data in Low-Resource Languages**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: BERT, Financial, Low-Resource, NER  
[Paper Link](http://arxiv.org/abs/2307.08714v1)  

---


**ABSTRACT**  
We propose an efficient modeling framework for cross-lingual named entity recognition in semi-structured text data. Our approach relies on both knowledge distillation and consistency training. The modeling framework leverages knowledge from a large language model (XLMRoBERTa) pre-trained on the source language, with a student-teacher relationship (knowledge distillation). The student model incorporates unsupervised consistency training (with KL divergence loss) on the low-resource target language.   We employ two independent datasets of SMSs in English and Arabic, each carrying semi-structured banking transaction information, and focus on exhibiting the transfer of knowledge from English to Arabic. With access to only 30 labeled samples, our model can generalize the recognition of merchants, amounts, and other fields from English to Arabic. We show that our modeling approach, while efficient, performs best overall when compared to state-of-the-art approaches like DistilBERT pre-trained on the target language or a supervised model directly trained on labeled data in the target language.   Our experiments show that it is enough to learn to recognize entities in English to reach reasonable performance in a low-resource language in the presence of a few labeled samples of semi-structured data. The proposed framework has implications for developing multi-lingual applications, especially in geographies where digital endeavors rely on both English and one or more low-resource language(s), sometimes mixed with English or employed singly.

{{</citation>}}


## cs.CC (1)



### (23/57) Tight (Double) Exponential Bounds for NP-Complete Problems: Treewidth and Vertex Cover Parameterizations (Florent Foucaud et al., 2023)

{{<citation>}}

Florent Foucaud, Esther Galby, Liana Khazaliya, Shaohua Li, Fionn Mc Inerney, Roohani Sharma, Prafullkumar Tale. (2023)  
**Tight (Double) Exponential Bounds for NP-Complete Problems: Treewidth and Vertex Cover Parameterizations**  

---
Primary Category: cs.CC
Categories: cs-CC, cs-DM, cs-DS, cs.CC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08149v1)  

---


**ABSTRACT**  
Treewidth is as an important parameter that yields tractability for many problems. For example, graph problems expressible in Monadic Second Order (MSO) logic and QUANTIFIED SAT or, more generally, QUANTIFIED CSP, are fixed-parameter tractable parameterized by the treewidth of the input's (primal) graph plus the length of the MSO-formula [Courcelle, Information & Computation 1990] and the quantifier rank [Chen, ECAI 2004], respectively. The algorithms generated by these (meta-)results have running times whose dependence on treewidth is a tower of exponents. A conditional lower bound by Fichte et al. [LICS 2020] shows that, for QUANTIFIED SAT, the height of this tower is equal to the number of quantifier alternations. Lower bounds showing that at least double-exponential factors in the running time are necessary, exhibit the extraordinary computational hardness of such problems, and are rare: there are very few (for treewidth tw and vertex cover vc parameterizations) and they are for $\Sigma_2^p$-, $\Sigma_3^p$- or #NP-complete problems.   We show, for the first time, that it is not necessary to go higher up in the polynomial hierarchy to obtain such lower bounds. Specifically, for the well-studied NP-complete metric graph problems METRIC DIMENSION, STRONG METRIC DIMENSION, and GEODETIC SET, we prove that they do not admit $2^{2^{o(tw)}} \cdot n^{O(1)}$-time algorithms, even on bounded diameter graphs, unless the ETH fails. For STRONG METRIC DIMENSION, this lower bound holds even for vc. This is impossible for the other two as they admit $2^{O({vc}^2)} \cdot n^{O(1)}$-time algorithms. We show that, unless the ETH fails, they do not admit $2^{o({vc}^2)}\cdot n^{O(1)}$-time algorithms, thereby adding to the short list of problems admitting such lower bounds. The latter results also yield lower bounds on the vertex-kernel sizes. We complement all our lower bounds with matching upper bounds.

{{</citation>}}


## cs.CV (19)



### (24/57) Self-Attention Based Generative Adversarial Networks For Unsupervised Video Summarization (Maria Nektaria Minaidi et al., 2023)

{{<citation>}}

Maria Nektaria Minaidi, Charilaos Papaioannou, Alexandros Potamianos. (2023)  
**Self-Attention Based Generative Adversarial Networks For Unsupervised Video Summarization**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Attention, LSTM, Self-Attention, Summarization  
[Paper Link](http://arxiv.org/abs/2307.08145v1)  

---


**ABSTRACT**  
In this paper, we study the problem of producing a comprehensive video summary following an unsupervised approach that relies on adversarial learning. We build on a popular method where a Generative Adversarial Network (GAN) is trained to create representative summaries, indistinguishable from the originals. The introduction of the attention mechanism into the architecture for the selection, encoding and decoding of video frames, shows the efficacy of self-attention and transformer in modeling temporal relationships for video summarization. We propose the SUM-GAN-AED model that uses a self-attention mechanism for frame selection, combined with LSTMs for encoding and decoding. We evaluate the performance of the SUM-GAN-AED model on the SumMe, TVSum and COGNIMUSE datasets. Experimental results indicate that using a self-attention mechanism as the frame selection mechanism outperforms the state-of-the-art on SumMe and leads to comparable to state-of-the-art performance on TVSum and COGNIMUSE.

{{</citation>}}


### (25/57) Heterogeneous graphs model spatial relationships between biological entities for breast cancer diagnosis (Akhila Krishna K et al., 2023)

{{<citation>}}

Akhila Krishna K, Ravi Kant Gupta, Nikhil Cherian Kurian, Pranav Jeevan, Amit Sethi. (2023)  
**Heterogeneous graphs model spatial relationships between biological entities for breast cancer diagnosis**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2307.08132v1)  

---


**ABSTRACT**  
The heterogeneity of breast cancer presents considerable challenges for its early detection, prognosis, and treatment selection. Convolutional neural networks often neglect the spatial relationships within histopathological images, which can limit their accuracy. Graph neural networks (GNNs) offer a promising solution by coding the spatial relationships within images. Prior studies have investigated the modeling of histopathological images as cell and tissue graphs, but they have not fully tapped into the potential of extracting interrelationships between these biological entities. In this paper, we present a novel approach using a heterogeneous GNN that captures the spatial and hierarchical relations between cell and tissue graphs to enhance the extraction of useful information from histopathological images. We also compare the performance of a cross-attention-based network and a transformer architecture for modeling the intricate relationships within tissue and cell graphs. Our model demonstrates superior efficiency in terms of parameter count and achieves higher accuracy compared to the transformer-based state-of-the-art approach on three publicly available breast cancer datasets -- BRIGHT, BreakHis, and BACH.

{{</citation>}}


### (26/57) Domain Generalisation with Bidirectional Encoder Representations from Vision Transformers (Hamza Riaz et al., 2023)

{{<citation>}}

Hamza Riaz, Alan F. Smeaton. (2023)  
**Domain Generalisation with Bidirectional Encoder Representations from Vision Transformers**  

---
Primary Category: cs.CV
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.08117v1)  

---


**ABSTRACT**  
Domain generalisation involves pooling knowledge from source domain(s) into a single model that can generalise to unseen target domain(s). Recent research in domain generalisation has faced challenges when using deep learning models as they interact with data distributions which differ from those they are trained on. Here we perform domain generalisation on out-of-distribution (OOD) vision benchmarks using vision transformers. Initially we examine four vision transformer architectures namely ViT, LeViT, DeiT, and BEIT on out-of-distribution data. As the bidirectional encoder representation from image transformers (BEIT) architecture performs best, we use it in further experiments on three benchmarks PACS, Home-Office and DomainNet. Our results show significant improvements in validation and test accuracy and our implementation significantly overcomes gaps between within-distribution and OOD data.

{{</citation>}}


### (27/57) Semi-DETR: Semi-Supervised Object Detection with Detection Transformers (Jiacheng Zhang et al., 2023)

{{<citation>}}

Jiacheng Zhang, Xiangru Lin, Wei Zhang, Kuo Wang, Xiao Tan, Junyu Han, Errui Ding, Jingdong Wang, Guanbin Li. (2023)  
**Semi-DETR: Semi-Supervised Object Detection with Detection Transformers**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Semi-Supervised, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.08095v1)  

---


**ABSTRACT**  
We analyze the DETR-based framework on semi-supervised object detection (SSOD) and observe that (1) the one-to-one assignment strategy generates incorrect matching when the pseudo ground-truth bounding box is inaccurate, leading to training inefficiency; (2) DETR-based detectors lack deterministic correspondence between the input query and its prediction output, which hinders the applicability of the consistency-based regularization widely used in current SSOD methods. We present Semi-DETR, the first transformer-based end-to-end semi-supervised object detector, to tackle these problems. Specifically, we propose a Stage-wise Hybrid Matching strategy that combines the one-to-many assignment and one-to-one assignment strategies to improve the training efficiency of the first stage and thus provide high-quality pseudo labels for the training of the second stage. Besides, we introduce a Crossview Query Consistency method to learn the semantic feature invariance of object queries from different views while avoiding the need to find deterministic query correspondence. Furthermore, we propose a Cost-based Pseudo Label Mining module to dynamically mine more pseudo boxes based on the matching cost of pseudo ground truth bounding boxes for consistency training. Extensive experiments on all SSOD settings of both COCO and Pascal VOC benchmark datasets show that our Semi-DETR method outperforms all state-of-the-art methods by clear margins. The PaddlePaddle version code1 is at https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/semi_det/semi_detr.

{{</citation>}}


### (28/57) Gait Data Augmentation using Physics-Based Biomechanical Simulation (Mritula Chandrasekaran et al., 2023)

{{<citation>}}

Mritula Chandrasekaran, Jarek Francik, Dimitrios Makris. (2023)  
**Gait Data Augmentation using Physics-Based Biomechanical Simulation**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.08092v1)  

---


**ABSTRACT**  
This paper focuses on addressing the problem of data scarcity for gait analysis. Standard augmentation methods may produce gait sequences that are not consistent with the biomechanical constraints of human walking. To address this issue, we propose a novel framework for gait data augmentation by using OpenSIM, a physics-based simulator, to synthesize biomechanically plausible walking sequences. The proposed approach is validated by augmenting the WBDS and CASIA-B datasets and then training gait-based classifiers for 3D gender gait classification and 2D gait person identification respectively. Experimental results indicate that our augmentation approach can improve the performance of model-based gait classifiers and deliver state-of-the-art results for gait-based person identification with an accuracy of up to 96.11% on the CASIA-B dataset.

{{</citation>}}


### (29/57) LafitE: Latent Diffusion Model with Feature Editing for Unsupervised Multi-class Anomaly Detection (Haonan Yin et al., 2023)

{{<citation>}}

Haonan Yin, Guanlong Jiao, Qianhui Wu, Borje F. Karlsson, Biqing Huang, Chin Yew Lin. (2023)  
**LafitE: Latent Diffusion Model with Feature Editing for Unsupervised Multi-class Anomaly Detection**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.08059v1)  

---


**ABSTRACT**  
In the context of flexible manufacturing systems that are required to produce different types and quantities of products with minimal reconfiguration, this paper addresses the problem of unsupervised multi-class anomaly detection: develop a unified model to detect anomalies from objects belonging to multiple classes when only normal data is accessible. We first explore the generative-based approach and investigate latent diffusion models for reconstruction to mitigate the notorious ``identity shortcut'' issue in auto-encoder based methods. We then introduce a feature editing strategy that modifies the input feature space of the diffusion model to further alleviate ``identity shortcuts'' and meanwhile improve the reconstruction quality of normal regions, leading to fewer false positive predictions. Moreover, we are the first who pose the problem of hyperparameter selection in unsupervised anomaly detection, and propose a solution of synthesizing anomaly data for a pseudo validation set to address this problem. Extensive experiments on benchmark datasets MVTec-AD and MPDD show that the proposed LafitE, \ie, Latent Diffusion Model with Feature Editing, outperforms state-of-art methods by a significant margin in terms of average AUROC. The hyperparamters selected via our pseudo validation set are well-matched to the real test set.

{{</citation>}}


### (30/57) Planting a SEED of Vision in Large Language Model (Yuying Ge et al., 2023)

{{<citation>}}

Yuying Ge, Yixiao Ge, Ziyun Zeng, Xintao Wang, Ying Shan. (2023)  
**Planting a SEED of Vision in Large Language Model**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.08041v1)  

---


**ABSTRACT**  
We present SEED, an elaborate image tokenizer that empowers Large Language Models (LLMs) with the emergent ability to SEE and Draw at the same time. Research on image tokenizers has previously reached an impasse, as frameworks employing quantized visual tokens have lost prominence due to subpar performance and convergence in multimodal comprehension (compared to BLIP-2, etc.) or generation (compared to Stable Diffusion, etc.). Despite the limitations, we remain confident in its natural capacity to unify visual and textual representations, facilitating scalable multimodal training with LLM's original recipe. In this study, we identify two crucial principles for the architecture and training of SEED that effectively ease subsequent alignment with LLMs. (1) Image tokens should be independent of 2D physical patch positions and instead be produced with a 1D causal dependency, exhibiting intrinsic interdependence that aligns with the left-to-right autoregressive prediction mechanism in LLMs. (2) Image tokens should capture high-level semantics consistent with the degree of semantic abstraction in words, and be optimized for both discriminativeness and reconstruction during the tokenizer training phase. As a result, the off-the-shelf LLM is able to perform both image-to-text and text-to-image generation by incorporating our SEED through efficient LoRA tuning. Comprehensive multimodal pretraining and instruction tuning, which may yield improved results, are reserved for future investigation. This version of SEED was trained in 5.7 days using only 64 V100 GPUs and 5M publicly available image-text pairs. Our preliminary study emphasizes the great potential of discrete visual tokens in versatile multimodal LLMs and the importance of proper image tokenizers in broader research.

{{</citation>}}


### (31/57) Analysing Gender Bias in Text-to-Image Models using Object Detection (Harvey Mannering, 2023)

{{<citation>}}

Harvey Mannering. (2023)  
**Analysing Gender Bias in Text-to-Image Models using Object Detection**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Bias, Object Detection  
[Paper Link](http://arxiv.org/abs/2307.08025v1)  

---


**ABSTRACT**  
This work presents a novel strategy to measure bias in text-to-image models. Using paired prompts that specify gender and vaguely reference an object (e.g. "a man/woman holding an item") we can examine whether certain objects are associated with a certain gender. In analysing results from Stable Diffusion, we observed that male prompts generated objects such as ties, knives, trucks, baseball bats, and bicycles more frequently. On the other hand, female prompts were more likely to generate objects such as handbags, umbrellas, bowls, bottles, and cups. We hope that the method outlined here will be a useful tool for examining bias in text-to-image models.

{{</citation>}}


### (32/57) Breaking Down the Task: A Unit-Grained Hybrid Training Framework for Vision and Language Decision Making (Ruipu Luo et al., 2023)

{{<citation>}}

Ruipu Luo, Jiwen Zhang, Zhongyu Wei. (2023)  
**Breaking Down the Task: A Unit-Grained Hybrid Training Framework for Vision and Language Decision Making**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.08016v1)  

---


**ABSTRACT**  
Vision language decision making (VLDM) is a challenging multimodal task. The agent have to understand complex human instructions and complete compositional tasks involving environment navigation and object manipulation. However, the long action sequences involved in VLDM make the task difficult to learn. From an environment perspective, we find that task episodes can be divided into fine-grained \textit{units}, each containing a navigation phase and an interaction phase. Since the environment within a unit stays unchanged, we propose a novel hybrid-training framework that enables active exploration in the environment and reduces the exposure bias. Such framework leverages the unit-grained configurations and is model-agnostic. Specifically, we design a Unit-Transformer (UT) with an intrinsic recurrent state that maintains a unit-scale cross-modal memory. Through extensive experiments on the TEACH benchmark, we demonstrate that our proposed framework outperforms existing state-of-the-art methods in terms of all evaluation metrics. Overall, our work introduces a novel approach to tackling the VLDM task by breaking it down into smaller, manageable units and utilizing a hybrid-training framework. By doing so, we provide a more flexible and effective solution for multimodal decision making.

{{</citation>}}


### (33/57) Boosting 3-DoF Ground-to-Satellite Camera Localization Accuracy via Geometry-Guided Cross-View Transformer (Yujiao Shi et al., 2023)

{{<citation>}}

Yujiao Shi, Fei Wu, Akhil Perincherry, Ankit Vora, Hongdong Li. (2023)  
**Boosting 3-DoF Ground-to-Satellite Camera Localization Accuracy via Geometry-Guided Cross-View Transformer**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.08015v3)  

---


**ABSTRACT**  
Image retrieval-based cross-view localization methods often lead to very coarse camera pose estimation, due to the limited sampling density of the database satellite images. In this paper, we propose a method to increase the accuracy of a ground camera's location and orientation by estimating the relative rotation and translation between the ground-level image and its matched/retrieved satellite image. Our approach designs a geometry-guided cross-view transformer that combines the benefits of conventional geometry and learnable cross-view transformers to map the ground-view observations to an overhead view. Given the synthesized overhead view and observed satellite feature maps, we construct a neural pose optimizer with strong global information embedding ability to estimate the relative rotation between them. After aligning their rotations, we develop an uncertainty-guided spatial correlation to generate a probability map of the vehicle locations, from which the relative translation can be determined. Experimental results demonstrate that our method significantly outperforms the state-of-the-art. Notably, the likelihood of restricting the vehicle lateral pose to be within 1m of its Ground Truth (GT) value on the cross-view KITTI dataset has been improved from $35.54\%$ to $76.44\%$, and the likelihood of restricting the vehicle orientation to be within $1^{\circ}$ of its GT value has been improved from $19.64\%$ to $99.10\%$.

{{</citation>}}


### (34/57) CoNAN: Conditional Neural Aggregation Network For Unconstrained Face Feature Fusion (Bhavin Jawade et al., 2023)

{{<citation>}}

Bhavin Jawade, Deen Dayal Mohan, Dennis Fedorishin, Srirangaraj Setlur, Venu Govindaraju. (2023)  
**CoNAN: Conditional Neural Aggregation Network For Unconstrained Face Feature Fusion**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2307.10237v1)  

---


**ABSTRACT**  
Face recognition from image sets acquired under unregulated and uncontrolled settings, such as at large distances, low resolutions, varying viewpoints, illumination, pose, and atmospheric conditions, is challenging. Face feature aggregation, which involves aggregating a set of N feature representations present in a template into a single global representation, plays a pivotal role in such recognition systems. Existing works in traditional face feature aggregation either utilize metadata or high-dimensional intermediate feature representations to estimate feature quality for aggregation. However, generating high-quality metadata or style information is not feasible for extremely low-resolution faces captured in long-range and high altitude settings. To overcome these limitations, we propose a feature distribution conditioning approach called CoNAN for template aggregation. Specifically, our method aims to learn a context vector conditioned over the distribution information of the incoming feature set, which is utilized to weigh the features based on their estimated informativeness. The proposed method produces state-of-the-art results on long-range unconstrained face recognition datasets such as BTS, and DroneSURF, validating the advantages of such an aggregation strategy.

{{</citation>}}


### (35/57) Towards Viewpoint-Invariant Visual Recognition via Adversarial Training (Shouwei Ruan et al., 2023)

{{<citation>}}

Shouwei Ruan, Yinpeng Dong, Hang Su, Jianteng Peng, Ning Chen, Xingxing Wei. (2023)  
**Towards Viewpoint-Invariant Visual Recognition via Adversarial Training**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2307.10235v1)  

---


**ABSTRACT**  
Visual recognition models are not invariant to viewpoint changes in the 3D world, as different viewing directions can dramatically affect the predictions given the same object. Although many efforts have been devoted to making neural networks invariant to 2D image translations and rotations, viewpoint invariance is rarely investigated. As most models process images in the perspective view, it is challenging to impose invariance to 3D viewpoint changes based only on 2D inputs. Motivated by the success of adversarial training in promoting model robustness, we propose Viewpoint-Invariant Adversarial Training (VIAT) to improve viewpoint robustness of common image classifiers. By regarding viewpoint transformation as an attack, VIAT is formulated as a minimax optimization problem, where the inner maximization characterizes diverse adversarial viewpoints by learning a Gaussian mixture distribution based on a new attack GMVFool, while the outer minimization trains a viewpoint-invariant classifier by minimizing the expected loss over the worst-case adversarial viewpoint distributions. To further improve the generalization performance, a distribution sharing strategy is introduced leveraging the transferability of adversarial viewpoints across objects. Experiments validate the effectiveness of VIAT in improving the viewpoint robustness of various image classifiers based on the diversity of adversarial viewpoints generated by GMVFool.

{{</citation>}}


### (36/57) Dual-level Interaction for Domain Adaptive Semantic Segmentation (Dongyu Yao et al., 2023)

{{<citation>}}

Dongyu Yao, Boheng Li, Run Wang, Lina Wang. (2023)  
**Dual-level Interaction for Domain Adaptive Semantic Segmentation**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.07972v1)  

---


**ABSTRACT**  
To circumvent the costly pixel-wise annotations of real-world images in the semantic segmentation task, the Unsupervised Domain Adaptation (UDA) is explored to firstly train a model with the labeled source data (synthetic images) and then adapt it to the unlabeled target data (real images). Among all the techniques being studied, the self-training approach recently secures its position in domain adaptive semantic segmentation, where a model is trained with target domain pseudo-labels. Current advances have mitigated noisy pseudo-labels resulting from the domain gap. However, they still struggle with erroneous pseudo-labels near the decision boundaries of the semantic classifier. In this paper, we tackle this issue by proposing a dual-level interaction for domain adaptation (DIDA) in semantic segmentation. Explicitly, we encourage the different augmented views of the same pixel to have not only similar class prediction (semantic-level) but also akin similarity relationship respected to other pixels (instance-level). As it is impossible to keep features of all pixel instances for a dataset, we novelly design and maintain a labeled instance bank with dynamic updating strategies to selectively store the informative features of instances. Further, DIDA performs cross-level interaction with scattering and gathering techniques to regenerate more reliable pseudolabels. Our method outperforms the state-of-the-art by a notable margin, especially on confusing and long-tailed classes. Code is available at https://github.com/RainJamesY/DIDA.

{{</citation>}}


### (37/57) Revisiting Domain-Adaptive 3D Object Detection by Reliable, Diverse and Class-balanced Pseudo-Labeling (Zhuoxiao Chen et al., 2023)

{{<citation>}}

Zhuoxiao Chen, Yadan Luo, Zi Huang, Zheng Wang, Mahsa Baktashmotlagh. (2023)  
**Revisiting Domain-Adaptive 3D Object Detection by Reliable, Diverse and Class-balanced Pseudo-Labeling**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.07944v1)  

---


**ABSTRACT**  
Unsupervised domain adaptation (DA) with the aid of pseudo labeling techniques has emerged as a crucial approach for domain-adaptive 3D object detection. While effective, existing DA methods suffer from a substantial drop in performance when applied to a multi-class training setting, due to the co-existence of low-quality pseudo labels and class imbalance issues. In this paper, we address this challenge by proposing a novel ReDB framework tailored for learning to detect all classes at once. Our approach produces Reliable, Diverse, and class-Balanced pseudo 3D boxes to iteratively guide the self-training on a distributionally different target domain. To alleviate disruptions caused by the environmental discrepancy (e.g., beam numbers), the proposed cross-domain examination (CDE) assesses the correctness of pseudo labels by copy-pasting target instances into a source environment and measuring the prediction consistency. To reduce computational overhead and mitigate the object shift (e.g., scales and point densities), we design an overlapped boxes counting (OBC) metric that allows to uniformly downsample pseudo-labeled objects across different geometric characteristics. To confront the issue of inter-class imbalance, we progressively augment the target point clouds with a class-balanced set of pseudo-labeled target instances and source objects, which boosts recognition accuracies on both frequently appearing and rare classes. Experimental results on three benchmark datasets using both voxel-based (i.e., SECOND) and point-based 3D detectors (i.e., PointRCNN) demonstrate that our proposed ReDB approach outperforms existing 3D domain adaptation methods by a large margin, improving 23.15% mAP on the nuScenes $\rightarrow$ KITTI task.

{{</citation>}}


### (38/57) KECOR: Kernel Coding Rate Maximization for Active 3D Object Detection (Yadan Luo et al., 2023)

{{<citation>}}

Yadan Luo, Zhuoxiao Chen, Zhen Fang, Zheng Zhang, Zi Huang, Mahsa Baktashmotlagh. (2023)  
**KECOR: Kernel Coding Rate Maximization for Active 3D Object Detection**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.07942v1)  

---


**ABSTRACT**  
Achieving a reliable LiDAR-based object detector in autonomous driving is paramount, but its success hinges on obtaining large amounts of precise 3D annotations. Active learning (AL) seeks to mitigate the annotation burden through algorithms that use fewer labels and can attain performance comparable to fully supervised learning. Although AL has shown promise, current approaches prioritize the selection of unlabeled point clouds with high uncertainty and/or diversity, leading to the selection of more instances for labeling and reduced computational efficiency. In this paper, we resort to a novel kernel coding rate maximization (KECOR) strategy which aims to identify the most informative point clouds to acquire labels through the lens of information theory. Greedy search is applied to seek desired point clouds that can maximize the minimal number of bits required to encode the latent features. To determine the uniqueness and informativeness of the selected samples from the model perspective, we construct a proxy network of the 3D detector head and compute the outer product of Jacobians from all proxy layers to form the empirical neural tangent kernel (NTK) matrix. To accommodate both one-stage (i.e., SECOND) and two-stage detectors (i.e., PVRCNN), we further incorporate the classification entropy maximization and well trade-off between detection performance and the total number of bounding boxes selected for annotation. Extensive experiments conducted on two 3D benchmarks and a 2D detection dataset evidence the superiority and versatility of the proposed approach. Our results show that approximately 44% box-level annotation costs and 26% computational time are reduced compared to the state-of-the-art AL method, without compromising detection performance.

{{</citation>}}


### (39/57) CVSformer: Cross-View Synthesis Transformer for Semantic Scene Completion (Haotian Dong et al., 2023)

{{<citation>}}

Haotian Dong, Enhui Ma, Lubo Wang, Miaohui Wang, Wuyuan Xie, Qing Guo, Ping Li, Lingyu Liang, Kairui Yang, Di Lin. (2023)  
**CVSformer: Cross-View Synthesis Transformer for Semantic Scene Completion**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.07938v1)  

---


**ABSTRACT**  
Semantic scene completion (SSC) requires an accurate understanding of the geometric and semantic relationships between the objects in the 3D scene for reasoning the occluded objects. The popular SSC methods voxelize the 3D objects, allowing the deep 3D convolutional network (3D CNN) to learn the object relationships from the complex scenes. However, the current networks lack the controllable kernels to model the object relationship across multiple views, where appropriate views provide the relevant information for suggesting the existence of the occluded objects. In this paper, we propose Cross-View Synthesis Transformer (CVSformer), which consists of Multi-View Feature Synthesis and Cross-View Transformer for learning cross-view object relationships. In the multi-view feature synthesis, we use a set of 3D convolutional kernels rotated differently to compute the multi-view features for each voxel. In the cross-view transformer, we employ the cross-view fusion to comprehensively learn the cross-view relationships, which form useful information for enhancing the features of individual views. We use the enhanced features to predict the geometric occupancies and semantic labels of all voxels. We evaluate CVSformer on public datasets, where CVSformer yields state-of-the-art results.

{{</citation>}}


### (40/57) S2R-ViT for Multi-Agent Cooperative Perception: Bridging the Gap from Simulation to Reality (Jinlong Li et al., 2023)

{{<citation>}}

Jinlong Li, Runsheng Xu, Xinyu Liu, Baolu Li, Qin Zou, Jiaqi Ma, Hongkai Yu. (2023)  
**S2R-ViT for Multi-Agent Cooperative Perception: Bridging the Gap from Simulation to Reality**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.07935v2)  

---


**ABSTRACT**  
Due to the lack of real multi-agent data and time-consuming of labeling, existing multi-agent cooperative perception algorithms usually select the simulated sensor data for training and validating. However, the perception performance is degraded when these simulation-trained models are deployed to the real world, due to the significant domain gap between the simulated and real data. In this paper, we propose the first Simulation-to-Reality transfer learning framework for multi-agent cooperative perception using a novel Vision Transformer, named as S2R-ViT, which considers both the Implementation Gap and Feature Gap between simulated and real data. We investigate the effects of these two types of domain gaps and propose a novel uncertainty-aware vision transformer to effectively relief the Implementation Gap and an agent-based feature adaptation module with inter-agent and ego-agent discriminators to reduce the Feature Gap. Our intensive experiments on the public multi-agent cooperative perception datasets OPV2V and V2V4Real demonstrate that the proposed S2R-ViT can effectively bridge the gap from simulation to reality and outperform other methods significantly for point cloud-based 3D object detection.

{{</citation>}}


### (41/57) Holistic Prototype Attention Network for Few-Shot VOS (Yin Tang et al., 2023)

{{<citation>}}

Yin Tang, Tao Chen, Xiruo Jiang, Yazhou Yao, Guo-Sen Xie, Heng-Tao Shen. (2023)  
**Holistic Prototype Attention Network for Few-Shot VOS**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Attention, Few-Shot  
[Paper Link](http://arxiv.org/abs/2307.07933v1)  

---


**ABSTRACT**  
Few-shot video object segmentation (FSVOS) aims to segment dynamic objects of unseen classes by resorting to a small set of support images that contain pixel-level object annotations. Existing methods have demonstrated that the domain agent-based attention mechanism is effective in FSVOS by learning the correlation between support images and query frames. However, the agent frame contains redundant pixel information and background noise, resulting in inferior segmentation performance. Moreover, existing methods tend to ignore inter-frame correlations in query videos. To alleviate the above dilemma, we propose a holistic prototype attention network (HPAN) for advancing FSVOS. Specifically, HPAN introduces a prototype graph attention module (PGAM) and a bidirectional prototype attention module (BPAM), transferring informative knowledge from seen to unseen classes. PGAM generates local prototypes from all foreground features and then utilizes their internal correlations to enhance the representation of the holistic prototypes. BPAM exploits the holistic information from support images and video frames by fusing co-attention and self-attention to achieve support-query semantic consistency and inner-frame temporal consistency. Extensive experiments on YouTube-FSVOS have been provided to demonstrate the effectiveness and superiority of our proposed HPAN method.

{{</citation>}}


### (42/57) DocTr: Document Transformer for Structured Information Extraction in Documents (Haofu Liao et al., 2023)

{{<citation>}}

Haofu Liao, Aruni RoyChowdhury, Weijian Li, Ankan Bansal, Yuting Zhang, Zhuowen Tu, Ravi Kumar Satzoda, R. Manmatha, Vijay Mahadevan. (2023)  
**DocTr: Document Transformer for Structured Information Extraction in Documents**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Information Extraction, Transformer  
[Paper Link](http://arxiv.org/abs/2307.07929v1)  

---


**ABSTRACT**  
We present a new formulation for structured information extraction (SIE) from visually rich documents. It aims to address the limitations of existing IOB tagging or graph-based formulations, which are either overly reliant on the correct ordering of input text or struggle with decoding a complex graph. Instead, motivated by anchor-based object detectors in vision, we represent an entity as an anchor word and a bounding box, and represent entity linking as the association between anchor words. This is more robust to text ordering, and maintains a compact graph for entity linking. The formulation motivates us to introduce 1) a DOCument TRansformer (DocTr) that aims at detecting and associating entity bounding boxes in visually rich documents, and 2) a simple pre-training strategy that helps learn entity detection in the context of language. Evaluations on three SIE benchmarks show the effectiveness of the proposed formulation, and the overall approach outperforms existing solutions.

{{</citation>}}


## eess.IV (2)



### (43/57) GastroVision: A Multi-class Endoscopy Image Dataset for Computer Aided Gastrointestinal Disease Detection (Debesh Jha et al., 2023)

{{<citation>}}

Debesh Jha, Vanshali Sharma, Neethi Dasu, Nikhil Kumar Tomar, Steven Hicks, M. K. Bhuyan, Pradip K. Das, Michael A. Riegler, Pål Halvorsen, Thomas de Lange, Ulas Bagci. (2023)  
**GastroVision: A Multi-class Endoscopy Image Dataset for Computer Aided Gastrointestinal Disease Detection**  

---
Primary Category: eess.IV
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08140v1)  

---


**ABSTRACT**  
Integrating real-time artificial intelligence (AI) systems in clinical practices faces challenges such as scalability and acceptance. These challenges include data availability, biased outcomes, data quality, lack of transparency, and underperformance on unseen datasets from different distributions. The scarcity of large-scale, precisely labeled, and diverse datasets are the major challenge for clinical integration. This scarcity is also due to the legal restrictions and extensive manual efforts required for accurate annotations from clinicians. To address these challenges, we present GastroVision, a multi-center open-access gastrointestinal (GI) endoscopy dataset that includes different anatomical landmarks, pathological abnormalities, polyp removal cases and normal findings (a total of 24 classes) from the GI tract. The dataset comprises 8,000 images acquired from B{\ae}rum Hospital in Norway and Karolinska University in Sweden and was annotated and verified by experienced GI endoscopists. Furthermore, we validate the significance of our dataset with extensive benchmarking based on the popular deep learning based baseline models. We believe our dataset can facilitate the development of AI-based algorithms for GI disease detection and classification. Our dataset is available at https://osf.io/84e7f/.

{{</citation>}}


### (44/57) TransNuSeg: A Lightweight Multi-Task Transformer for Nuclei Segmentation (Zhenqi He et al., 2023)

{{<citation>}}

Zhenqi He, Mathias Unberath, Jing Ke, Yiqing Shen. (2023)  
**TransNuSeg: A Lightweight Multi-Task Transformer for Nuclei Segmentation**  

---
Primary Category: eess.IV
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.08051v1)  

---


**ABSTRACT**  
Nuclei appear small in size, yet, in real clinical practice, the global spatial information and correlation of the color or brightness contrast between nuclei and background, have been considered a crucial component for accurate nuclei segmentation. However, the field of automatic nuclei segmentation is dominated by Convolutional Neural Networks (CNNs), meanwhile, the potential of the recently prevalent Transformers has not been fully explored, which is powerful in capturing local-global correlations. To this end, we make the first attempt at a pure Transformer framework for nuclei segmentation, called TransNuSeg. Different from prior work, we decouple the challenging nuclei segmentation task into an intrinsic multi-task learning task, where a tri-decoder structure is employed for nuclei instance, nuclei edge, and clustered edge segmentation respectively. To eliminate the divergent predictions from different branches in previous work, a novel self distillation loss is introduced to explicitly impose consistency regulation between branches. Moreover, to formulate the high correlation between branches and also reduce the number of parameters, an efficient attention sharing scheme is proposed by partially sharing the self-attention heads amongst the tri-decoders. Finally, a token MLP bottleneck replaces the over-parameterized Transformer bottleneck for a further reduction in model complexity. Experiments on two datasets of different modalities, including MoNuSeg have shown that our methods can outperform state-of-the-art counterparts such as CA2.5-Net by 2-3% Dice with 30% fewer parameters. In conclusion, TransNuSeg confirms the strength of Transformer in the context of nuclei segmentation, which thus can serve as an efficient solution for real clinical practice. Code is available at https://github.com/zhenqi-he/transnuseg.

{{</citation>}}


## cs.SI (3)



### (45/57) INFLECT-DGNN: Influencer Prediction with Dynamic Graph Neural Networks (Elena Tiukhova et al., 2023)

{{<citation>}}

Elena Tiukhova, Emiliano Penaloza, María Óskarsdóttir, Bart Baesens, Monique Snoeck, Cristián Bravo. (2023)  
**INFLECT-DGNN: Influencer Prediction with Dynamic Graph Neural Networks**  

---
Primary Category: cs.SI
Categories: cs-AI, cs-LG, cs-SI, cs.SI  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.08131v1)  

---


**ABSTRACT**  
Leveraging network information for predictive modeling has become widespread in many domains. Within the realm of referral and targeted marketing, influencer detection stands out as an area that could greatly benefit from the incorporation of dynamic network representation due to the ongoing development of customer-brand relationships. To elaborate this idea, we introduce INFLECT-DGNN, a new framework for INFLuencer prEdiCTion with Dynamic Graph Neural Networks that combines Graph Neural Networks (GNN) and Recurrent Neural Networks (RNN) with weighted loss functions, the Synthetic Minority Oversampling TEchnique (SMOTE) adapted for graph data, and a carefully crafted rolling-window strategy. To evaluate predictive performance, we utilize a unique corporate data set with networks of three cities and derive a profit-driven evaluation methodology for influencer prediction. Our results show how using RNN to encode temporal attributes alongside GNNs significantly improves predictive performance. We compare the results of various models to demonstrate the importance of capturing graph representation, temporal dependencies, and using a profit-driven methodology for evaluation.

{{</citation>}}


### (46/57) The Roll-Out of Community Notes Did Not Reduce Engagement With Misinformation on Twitter (Yuwei Chuai et al., 2023)

{{<citation>}}

Yuwei Chuai, Haoye Tian, Nicolas Pröllochs, Gabriele Lenzini. (2023)  
**The Roll-Out of Community Notes Did Not Reduce Engagement With Misinformation on Twitter**  

---
Primary Category: cs.SI
Categories: cs-HC, cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2307.07960v1)  

---


**ABSTRACT**  
Developing interventions that successfully reduce engagement with misinformation on social media is challenging. One intervention that has recently gained great attention is Twitter's Community Notes (previously known as "Birdwatch"). Community Notes is a crowdsourced fact-checking approach that allows users to write textual notes to inform others about potentially misleading posts on Twitter. Yet, empirical evidence regarding its effectiveness in reducing engagement with misinformation on social media is missing. In this paper, we perform a large-scale empirical study to analyze whether the introduction of the Community Notes feature and its roll-out to users in the U. S. and around the world have reduced engagement with misinformation on Twitter in terms of retweet volume and likes. We employ Difference-in-Difference (DiD) models and Regression Discontinuity Design (RDD) to analyze a comprehensive dataset consisting of all fact-checking notes and corresponding source tweets since the launch of Community Notes in early 2021. Although we observe a significant increase in the volume of fact-checks carried out via Community Notes, particularly for tweets from verified users with many followers, we find no evidence that the introduction of Community Notes significantly reduced engagement with misleading tweets on Twitter. Rather, our findings suggest that Community Notes might be too slow to effectively reduce engagement with misinformation in the early (and most viral) stage of diffusion. Our work emphasizes the importance of evaluating fact-checking interventions in the field and offers important implications to enhance crowdsourced fact-checking strategies on social media.

{{</citation>}}


### (47/57) A structural study of Big Tech firm-switching of inventors in the post-recession era (Yidan Sun et al., 2023)

{{<citation>}}

Yidan Sun, Mayank Kejriwal. (2023)  
**A structural study of Big Tech firm-switching of inventors in the post-recession era**  

---
Primary Category: cs.SI
Categories: cs-SI, cs.SI  
Keywords: Amazon, Google, Microsoft  
[Paper Link](http://arxiv.org/abs/2307.07920v1)  

---


**ABSTRACT**  
Complex systems research and network science have recently been used to provide novel insights into economic phenomena such as patenting behavior and innovation in firms. Several studies have found that increased mobility of inventors, manifested through firm switching or transitioning, is associated with increased overall productivity. This paper proposes a novel structural study of such transitioning inventors, and the role they play in patent co-authorship networks, in a cohort of highly innovative and economically influential companies such as the five Big Tech firms (Apple, Microsoft, Google, Amazon and Meta) in the post-recession period (2010-2022). We formulate and empirically investigate three research questions using Big Tech patent data. Our results show that transitioning inventors tend to have higher degree centrality than the average Big Tech inventor, and that their removal can lead to greater network fragmentation than would be expected by chance. The rate of transition over the 12-year period of study was found to be highest between 2015-2017, suggesting that the Big Tech innovation ecosystem underwent non-trivial shifts during this time. Finally, transition was associated with higher estimated impact of co-authored patents post-transition.

{{</citation>}}


## cs.DC (1)



### (48/57) MaGNAS: A Mapping-Aware Graph Neural Architecture Search Framework for Heterogeneous MPSoC Deployment (Mohanad Odema et al., 2023)

{{<citation>}}

Mohanad Odema, Halima Bouzidi, Hamza Ouarnoughi, Smail Niar, Mohammad Abdullah Al Faruque. (2023)  
**MaGNAS: A Mapping-Aware Graph Neural Architecture Search Framework for Heterogeneous MPSoC Deployment**  

---
Primary Category: cs.DC
Categories: cs-CV, cs-DC, cs-LG, cs.DC  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.08065v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) are becoming increasingly popular for vision-based applications due to their intrinsic capacity in modeling structural and contextual relations between various parts of an image frame. On another front, the rising popularity of deep vision-based applications at the edge has been facilitated by the recent advancements in heterogeneous multi-processor Systems on Chips (MPSoCs) that enable inference under real-time, stringent execution requirements. By extension, GNNs employed for vision-based applications must adhere to the same execution requirements. Yet contrary to typical deep neural networks, the irregular flow of graph learning operations poses a challenge to running GNNs on such heterogeneous MPSoC platforms. In this paper, we propose a novel unified design-mapping approach for efficient processing of vision GNN workloads on heterogeneous MPSoC platforms. Particularly, we develop MaGNAS, a mapping-aware Graph Neural Architecture Search framework. MaGNAS proposes a GNN architectural design space coupled with prospective mapping options on a heterogeneous SoC to identify model architectures that maximize on-device resource efficiency. To achieve this, MaGNAS employs a two-tier evolutionary search to identify optimal GNNs and mapping pairings that yield the best performance trade-offs. Through designing a supernet derived from the recent Vision GNN (ViG) architecture, we conducted experiments on four (04) state-of-the-art vision datasets using both (i) a real hardware SoC platform (NVIDIA Xavier AGX) and (ii) a performance/cost model simulator for DNN accelerators. Our experimental results demonstrate that MaGNAS is able to provide 1.57x latency speedup and is 3.38x more energy-efficient for several vision datasets executed on the Xavier MPSoC vs. the GPU-only deployment while sustaining an average 0.11% accuracy reduction from the baseline.

{{</citation>}}


## econ.GN (1)



### (49/57) Datalism and Data Monopolies in the Era of A.I.: A Research Agenda (Catherine E. A. Mulligan et al., 2023)

{{<citation>}}

Catherine E. A. Mulligan, Phil Godsiff. (2023)  
**Datalism and Data Monopolies in the Era of A.I.: A Research Agenda**  

---
Primary Category: econ.GN
Categories: cs-HC, econ-GN, econ.GN, q-fin-EC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.08049v1)  

---


**ABSTRACT**  
The increasing use of data in various parts of the economic and social systems is creating a new form of monopoly: data monopolies. We illustrate that the companies using these strategies, Datalists, are challenging the existing definitions used within Monopoly Capital Theory (MCT). Datalists are pursuing a different type of monopoly control than traditional multinational corporations. They are pursuing monopolistic control over data to feed their productive processes, increasingly controlled by algorithms and Artificial Intelligence (AI). These productive processes use information about humans and the creative outputs of humans as the inputs but do not classify those humans as employees, so they are not paid or credited for their labour. This paper provides an overview of this evolution and its impact on monopoly theory. It concludes with an outline for a research agenda for economics in this space.

{{</citation>}}


## quant-ph (1)



### (50/57) Fast Quantum Algorithm for Attention Computation (Yeqi Gao et al., 2023)

{{<citation>}}

Yeqi Gao, Zhao Song, Xin Yang, Ruizhe Zhang. (2023)  
**Fast Quantum Algorithm for Attention Computation**  

---
Primary Category: quant-ph
Categories: cs-LG, quant-ph, quant-ph  
Keywords: Attention, NLP  
[Paper Link](http://arxiv.org/abs/2307.08045v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated exceptional performance across a wide range of tasks. These models, powered by advanced deep learning techniques, have revolutionized the field of natural language processing (NLP) and have achieved remarkable results in various language-related tasks.   LLMs have excelled in tasks such as machine translation, sentiment analysis, question answering, text generation, text classification, language modeling, and more. They have proven to be highly effective in capturing complex linguistic patterns, understanding context, and generating coherent and contextually relevant text. The attention scheme plays a crucial role in the architecture of large language models (LLMs). It is a fundamental component that enables the model to capture and utilize contextual information during language processing tasks effectively. Making the attention scheme computation faster is one of the central questions to speed up the LLMs computation. It is well-known that quantum machine has certain computational advantages compared to the classical machine. However, it is currently unknown whether quantum computing can aid in LLM.   In this work, we focus on utilizing Grover's Search algorithm to compute a sparse attention computation matrix efficiently. We achieve a polynomial quantum speed-up over the classical method. Moreover, the attention matrix outputted by our quantum algorithm exhibits an extra low-rank structure that will be useful in obtaining a faster training algorithm for LLMs. Additionally, we present a detailed analysis of the algorithm's error analysis and time complexity within the context of computing the attention matrix.

{{</citation>}}


## cs.IT (1)



### (51/57) STAR-RIS Enhanced Joint Physical Layer Security and Covert Communications for Multi-antenna mmWave Systems (Han Xiao et al., 2023)

{{<citation>}}

Han Xiao, Xiaoyan Hu, Ang Li, Wenjie Wang, Zhou Su, Kai-Kit Wong, Kun Yang. (2023)  
**STAR-RIS Enhanced Joint Physical Layer Security and Covert Communications for Multi-antenna mmWave Systems**  

---
Primary Category: cs.IT
Categories: cs-IT, cs.IT, math-IT  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.08043v1)  

---


**ABSTRACT**  
This paper investigates the utilization of simultaneously transmitting and reflecting RIS (STAR-RIS) in supporting joint physical layer security (PLS) and covert communications (CCs) in a multi-antenna millimeter wave (mmWave) system, where the base station (BS) communicates with both covert and security users while defeating eavesdropping by wardens with the help of a STAR-RIS. Specifically, analytical derivations are performed to obtain the closed-form expression of warden's minimum detection error probability (DEP). Furthermore, the asymptotic result of the minimum DEP and the lower bound of the secure rates are derived, considering the practical assumption that BS only knows the statistical channel state information (CSI) between STAR-RIS and the wardens. Subsequently, an optimization problem is formulated with the aim of maximizing the average sum of the covert rate and the minimum secure rate while ensuring the covert requirement and quality of service (QoS) for legal users by jointly optimizing the active and passive beamformers. Due to the strong coupling among variables, an iterative algorithm based on the alternating strategy and the semi-definite relaxation (SDR) method is proposed to solve the non-convex optimization problem. Simulation results indicate that the performance of the proposed STAR-RIS-assisted scheme greatly surpasses that of the conventional RIS scheme, which validates the superiority of STAR-RIS in simultaneously implementing PLS and CCs.

{{</citation>}}


## physics.geo-ph (1)



### (52/57) Joint Microseismic Event Detection and Location with a Detection Transformer (Yuanyuan Yang et al., 2023)

{{<citation>}}

Yuanyuan Yang, Claire Birnie, Tariq Alkhalifah. (2023)  
**Joint Microseismic Event Detection and Location with a Detection Transformer**  

---
Primary Category: physics.geo-ph
Categories: cs-LG, eess-SP, physics-geo-ph, physics.geo-ph  
Keywords: Event Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2307.09207v1)  

---


**ABSTRACT**  
Microseismic event detection and location are two primary components in microseismic monitoring, which offers us invaluable insights into the subsurface during reservoir stimulation and evolution. Conventional approaches for event detection and location often suffer from manual intervention and/or heavy computation, while current machine learning-assisted approaches typically address detection and location separately; such limitations hinder the potential for real-time microseismic monitoring. We propose an approach to unify event detection and source location into a single framework by adapting a Convolutional Neural Network backbone and an encoder-decoder Transformer with a set-based Hungarian loss, which is applied directly to recorded waveforms. The proposed network is trained on synthetic data simulating multiple microseismic events corresponding to random source locations in the area of suspected microseismic activities. A synthetic test on a 2D profile of the SEAM Time Lapse model illustrates the capability of the proposed method in detecting the events properly and locating them in the subsurface accurately; while, a field test using the Arkoma Basin data further proves its practicability, efficiency, and its potential in paving the way for real-time monitoring of microseismic events.

{{</citation>}}


## cs.IR (1)



### (53/57) Data Discovery for the SDGs: A Systematic Rule-based Approach (Yuwei Jiang et al., 2023)

{{<citation>}}

Yuwei Jiang, David Johnson. (2023)  
**Data Discovery for the SDGs: A Systematic Rule-based Approach**  

---
Primary Category: cs.IR
Categories: cs-IR, cs.IR  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2307.07983v1)  

---


**ABSTRACT**  
In 2015, the United Nations put forward 17 Sustainable Development Goals (SDGs) to be achieved by 2030, where data has been promoted as a focus to innovating sustainable development and as a means to measuring progress towards achieving the SDGs. In this study, we propose a systematic approach towards discovering data types and sources that can be used for SDG research. The proposed method integrates a systematic mapping approach using manual qualitative coding over a corpus of SDG-related research literature followed by an automated process that applies rules to perform data entity extraction computationally. This approach is exemplified by an analysis of literature relating to SDG 7, the results of which are also presented in this paper. The paper concludes with a discussion of the approach and suggests future work to extend the method with more advance NLP and machine learning techniques.

{{</citation>}}


## cs.SE (1)



### (54/57) Look Before You Leap: An Exploratory Study of Uncertainty Measurement for Large Language Models (Yuheng Huang et al., 2023)

{{<citation>}}

Yuheng Huang, Jiayang Song, Zhijie Wang, Huaming Chen, Lei Ma. (2023)  
**Look Before You Leap: An Exploratory Study of Uncertainty Measurement for Large Language Models**  

---
Primary Category: cs.SE
Categories: cs-AI, cs-CL, cs-SE, cs.SE  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2307.10236v1)  

---


**ABSTRACT**  
The recent performance leap of Large Language Models (LLMs) opens up new opportunities across numerous industrial applications and domains. However, erroneous generations, such as false predictions, misinformation, and hallucination made by LLMs, have also raised severe concerns for the trustworthiness of LLMs', especially in safety-, security- and reliability-sensitive scenarios, potentially hindering real-world adoptions. While uncertainty estimation has shown its potential for interpreting the prediction risks made by general machine learning (ML) models, little is known about whether and to what extent it can help explore an LLM's capabilities and counteract its undesired behavior. To bridge the gap, in this paper, we initiate an exploratory study on the risk assessment of LLMs from the lens of uncertainty. In particular, we experiment with twelve uncertainty estimation methods and four LLMs on four prominent natural language processing (NLP) tasks to investigate to what extent uncertainty estimation techniques could help characterize the prediction risks of LLMs. Our findings validate the effectiveness of uncertainty estimation for revealing LLMs' uncertain/non-factual predictions. In addition to general NLP tasks, we extensively conduct experiments with four LLMs for code generation on two datasets. We find that uncertainty estimation can potentially uncover buggy programs generated by LLMs. Insights from our study shed light on future design and development for reliable LLMs, facilitating further research toward enhancing the trustworthiness of LLMs.

{{</citation>}}


## cs.HC (1)



### (55/57) InkSight: Leveraging Sketch Interaction for Documenting Chart Findings in Computational Notebooks (Yanna Lin et al., 2023)

{{<citation>}}

Yanna Lin, Haotian Li, Leni Yang, Aoyu Wu, Huamin Qu. (2023)  
**InkSight: Leveraging Sketch Interaction for Documenting Chart Findings in Computational Notebooks**  

---
Primary Category: cs.HC
Categories: cs-HC, cs.HC  
Keywords: GPT, GPT-3.5, Sketch  
[Paper Link](http://arxiv.org/abs/2307.07922v1)  

---


**ABSTRACT**  
Computational notebooks have become increasingly popular for exploratory data analysis due to their ability to support data exploration and explanation within a single document. Effective documentation for explaining chart findings during the exploration process is essential as it helps recall and share data analysis. However, documenting chart findings remains a challenge due to its time-consuming and tedious nature. While existing automatic methods alleviate some of the burden on users, they often fail to cater to users' specific interests. In response to these limitations, we present InkSight, a mixed-initiative computational notebook plugin that generates finding documentation based on the user's intent. InkSight allows users to express their intent in specific data subsets through sketching atop visualizations intuitively. To facilitate this, we designed two types of sketches, i.e., open-path and closed-path sketch. Upon receiving a user's sketch, InkSight identifies the sketch type and corresponding selected data items. Subsequently, it filters data fact types based on the sketch and selected data items before employing existing automatic data fact recommendation algorithms to infer data facts. Using large language models (GPT-3.5), InkSight converts data facts into effective natural language documentation. Users can conveniently fine-tune the generated documentation within InkSight. A user study with 12 participants demonstrated the usability and effectiveness of InkSight in expressing user intent and facilitating chart finding documentation.

{{</citation>}}


## cs.AR (1)



### (56/57) Exploiting FPGA Capabilities for Accelerated Biomedical Computing (Kayode Inadagbo et al., 2023)

{{<citation>}}

Kayode Inadagbo, Baran Arig, Nisanur Alici, Murat Isik. (2023)  
**Exploiting FPGA Capabilities for Accelerated Biomedical Computing**  

---
Primary Category: cs.AR
Categories: cs-AR, cs-LG, cs.AR, eess-SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.07914v1)  

---


**ABSTRACT**  
This study presents advanced neural network architectures including Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Long Short-Term Memory Networks (LSTMs), and Deep Belief Networks (DBNs) for enhanced ECG signal analysis using Field Programmable Gate Arrays (FPGAs). We utilize the MIT-BIH Arrhythmia Database for training and validation, introducing Gaussian noise to improve algorithm robustness. The implemented models feature various layers for distinct processing and classification tasks and techniques like EarlyStopping callback and Dropout layer are used to mitigate overfitting. Our work also explores the development of a custom Tensor Compute Unit (TCU) accelerator for the PYNQ Z1 board, offering comprehensive steps for FPGA-based machine learning, including setting up the Tensil toolchain in Docker, selecting architecture, configuring PS-PL, and compiling and executing models. Performance metrics such as latency and throughput are calculated for practical insights, demonstrating the potential of FPGAs in high-performance biomedical computing. The study ultimately offers a guide for optimizing neural network performance on FPGAs for various applications.

{{</citation>}}


## cs.CR (1)



### (57/57) Jailbreaker: Automated Jailbreak Across Multiple Large Language Model Chatbots (Gelei Deng et al., 2023)

{{<citation>}}

Gelei Deng, Yi Liu, Yuekang Li, Kailong Wang, Ying Zhang, Zefeng Li, Haoyu Wang, Tianwei Zhang, Yang Liu. (2023)  
**Jailbreaker: Automated Jailbreak Across Multiple Large Language Model Chatbots**  

---
Primary Category: cs.CR
Categories: cs-CR, cs.CR  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.08715v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have revolutionized Artificial Intelligence (AI) services due to their exceptional proficiency in understanding and generating human-like text. LLM chatbots, in particular, have seen widespread adoption, transforming human-machine interactions. However, these LLM chatbots are susceptible to "jailbreak" attacks, where malicious users manipulate prompts to elicit inappropriate or sensitive responses, contravening service policies. Despite existing attempts to mitigate such threats, our research reveals a substantial gap in our understanding of these vulnerabilities, largely due to the undisclosed defensive measures implemented by LLM service providers.   In this paper, we present Jailbreaker, a comprehensive framework that offers an in-depth understanding of jailbreak attacks and countermeasures. Our work makes a dual contribution. First, we propose an innovative methodology inspired by time-based SQL injection techniques to reverse-engineer the defensive strategies of prominent LLM chatbots, such as ChatGPT, Bard, and Bing Chat. This time-sensitive approach uncovers intricate details about these services' defenses, facilitating a proof-of-concept attack that successfully bypasses their mechanisms. Second, we introduce an automatic generation method for jailbreak prompts. Leveraging a fine-tuned LLM, we validate the potential of automated jailbreak generation across various commercial LLM chatbots. Our method achieves a promising average success rate of 21.58%, significantly outperforming the effectiveness of existing techniques. We have responsibly disclosed our findings to the concerned service providers, underscoring the urgent need for more robust defenses. Jailbreaker thus marks a significant step towards understanding and mitigating jailbreak threats in the realm of LLM chatbots.

{{</citation>}}
