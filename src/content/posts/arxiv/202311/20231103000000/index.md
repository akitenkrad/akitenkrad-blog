---
draft: false
title: "arXiv @ 2023.11.03"
date: 2023-11-03
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.03"
    identifier: arxiv_20231103
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.RO (2)](#csro-2)
- [cs.SE (2)](#csse-2)
- [cs.LG (24)](#cslg-24)
- [cs.SD (3)](#cssd-3)
- [eess.AS (1)](#eessas-1)
- [cs.AI (15)](#csai-15)
- [cs.CR (6)](#cscr-6)
- [cs.CL (27)](#cscl-27)
- [cs.CY (1)](#cscy-1)
- [cs.CV (25)](#cscv-25)
- [cs.DL (1)](#csdl-1)
- [cs.MA (1)](#csma-1)
- [cs.IR (5)](#csir-5)
- [stat.ML (1)](#statml-1)
- [eess.IV (3)](#eessiv-3)
- [cs.HC (1)](#cshc-1)
- [cs.GT (1)](#csgt-1)
- [cs.DC (1)](#csdc-1)
- [eess.SP (1)](#eesssp-1)

## cs.RO (2)



### (1/121) RoboVQA: Multimodal Long-Horizon Reasoning for Robotics (Pierre Sermanet et al., 2023)

{{<citation>}}

Pierre Sermanet, Tianli Ding, Jeffrey Zhao, Fei Xia, Debidatta Dwibedi, Keerthana Gopalakrishnan, Christine Chan, Gabriel Dulac-Arnold, Sharath Maddineni, Nikhil J Joshi, Pete Florence, Wei Han, Robert Baruch, Yao Lu, Suvir Mirchandani, Peng Xu, Pannag Sanketi, Karol Hausman, Izhak Shafran, Brian Ichter, Yuan Cao. (2023)  
**RoboVQA: Multimodal Long-Horizon Reasoning for Robotics**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.00899v1)  

---


**ABSTRACT**  
We present a scalable, bottom-up and intrinsically diverse data collection scheme that can be used for high-level reasoning with long and medium horizons and that has 2.2x higher throughput compared to traditional narrow top-down step-by-step collection. We collect realistic data by performing any user requests within the entirety of 3 office buildings and using multiple robot and human embodiments. With this data, we show that models trained on all embodiments perform better than ones trained on the robot data only, even when evaluated solely on robot episodes. We find that for a fixed collection budget it is beneficial to take advantage of cheaper human collection along with robot collection. We release a large and highly diverse (29,520 unique instructions) dataset dubbed RoboVQA containing 829,502 (video, text) pairs for robotics-focused visual question answering. We also demonstrate how evaluating real robot experiments with an intervention mechanism enables performing tasks to completion, making it deployable with human oversight even if imperfect while also providing a single performance metric. We demonstrate a single video-conditioned model named RoboVQA-VideoCoCa trained on our dataset that is capable of performing a variety of grounded high-level reasoning tasks in broad realistic settings with a cognitive intervention rate 46% lower than the zero-shot state of the art visual language model (VLM) baseline and is able to guide real robots through long-horizon tasks. The performance gap with zero-shot state-of-the-art models indicates that a lot of grounded data remains to be collected for real-world deployment, emphasizing the critical need for scalable data collection approaches. Finally, we show that video VLMs significantly outperform single-image VLMs with an average error rate reduction of 19% across all VQA tasks. Data and videos available at https://robovqa.github.io

{{</citation>}}


### (2/121) PIAug -- Physics Informed Augmentation for Learning Vehicle Dynamics for Off-Road Navigation (Parv Maheshwari et al., 2023)

{{<citation>}}

Parv Maheshwari, Wenshan Wang, Samuel Triest, Matthew Sivaprakasam, Shubhra Aich, John G. Rogers III, Jason M. Gregory, Sebastian Scherer. (2023)  
**PIAug -- Physics Informed Augmentation for Learning Vehicle Dynamics for Off-Road Navigation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.00815v1)  

---


**ABSTRACT**  
Modeling the precise dynamics of off-road vehicles is a complex yet essential task due to the challenging terrain they encounter and the need for optimal performance and safety. Recently, there has been a focus on integrating nominal physics-based models alongside data-driven neural networks using Physics Informed Neural Networks. These approaches often assume the availability of a well-distributed dataset; however, this assumption may not hold due to regions in the physical distribution that are hard to collect, such as high-speed motions and rare terrains. Therefore, we introduce a physics-informed data augmentation methodology called PIAug. We show an example use case of the same by modeling high-speed and aggressive motion predictions, given a dataset with only low-speed data. During the training phase, we leverage the nominal model for generating target domain (medium and high velocity) data using the available source data (low velocity). Subsequently, we employ a physics-inspired loss function with this augmented dataset to incorporate prior knowledge of physics into the neural network. Our methodology results in up to 67% less mean error in trajectory prediction in comparison to a standalone nominal model, especially during aggressive maneuvers at speeds outside the training domain. In real-life navigation experiments, our model succeeds in 4x tighter waypoint tracking constraints than the Kinematic Bicycle Model (KBM) at out-of-domain velocities.

{{</citation>}}


## cs.SE (2)



### (3/121) Generate and Pray: Using SALLMS to Evaluate the Security of LLM Generated Code (Mohammed Latif Siddiq et al., 2023)

{{<citation>}}

Mohammed Latif Siddiq, Joanna C. S. Santos. (2023)  
**Generate and Pray: Using SALLMS to Evaluate the Security of LLM Generated Code**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: BLEU, ChatGPT, GPT, Language Model, Security  
[Paper Link](http://arxiv.org/abs/2311.00889v1)  

---


**ABSTRACT**  
With the growing popularity of Large Language Models (e.g. GitHub Copilot, ChatGPT, etc.) in software engineers' daily practices, it is important to ensure that the code generated by these tools is not only functionally correct but also free of vulnerabilities. Although LLMs can help developers to be more productive, prior empirical studies have shown that LLMs can generate insecure code. There are two contributing factors to the insecure code generation. First, existing datasets used to evaluate Large Language Models (LLMs) do not adequately represent genuine software engineering tasks sensitive to security. Instead, they are often based on competitive programming challenges or classroom-type coding tasks. In real-world applications, the code produced is integrated into larger codebases, introducing potential security risks. There's a clear absence of benchmarks that focus on evaluating the security of the generated code. Second, existing evaluation metrics primarily focus on the functional correctness of the generated code while ignoring security considerations. Metrics such as pass@k gauge the probability of obtaining the correct code in the top k suggestions. Other popular metrics like BLEU, CodeBLEU, ROUGE, and METEOR similarly emphasize functional accuracy, neglecting security implications. In light of these research gaps, in this paper, we described SALLM, a framework to benchmark LLMs' abilities to generate secure code systematically. This framework has three major components: a novel dataset of security-centric Python prompts, an evaluation environment to test the generated code, and novel metrics to evaluate the models' performance from the perspective of secure code generation.

{{</citation>}}


### (4/121) Software Repositories and Machine Learning Research in Cyber Security (Mounika Vanamala et al., 2023)

{{<citation>}}

Mounika Vanamala, Keith Bryant, Alex Caravella. (2023)  
**Software Repositories and Machine Learning Research in Cyber Security**  

---
Primary Category: cs.SE  
Categories: cs-CR, cs-LG, cs-SE, cs.SE  
Keywords: Cyber Security, Security  
[Paper Link](http://arxiv.org/abs/2311.00691v1)  

---


**ABSTRACT**  
In today's rapidly evolving technological landscape and advanced software development, the rise in cyber security attacks has become a pressing concern. The integration of robust cyber security defenses has become essential across all phases of software development. It holds particular significance in identifying critical cyber security vulnerabilities at the initial stages of the software development life cycle, notably during the requirement phase. Through the utilization of cyber security repositories like The Common Attack Pattern Enumeration and Classification (CAPEC) from MITRE and the Common Vulnerabilities and Exposures (CVE) databases, attempts have been made to leverage topic modeling and machine learning for the detection of these early-stage vulnerabilities in the software requirements process. Past research themes have returned successful outcomes in attempting to automate vulnerability identification for software developers, employing a mixture of unsupervised machine learning methodologies such as LDA and topic modeling. Looking ahead, in our pursuit to improve automation and establish connections between software requirements and vulnerabilities, our strategy entails adopting a variety of supervised machine learning techniques. This array encompasses Support Vector Machines (SVM), Na\"ive Bayes, random forest, neural networking and eventually transitioning into deep learning for our investigation. In the face of the escalating complexity of cyber security, the question of whether machine learning can enhance the identification of vulnerabilities in diverse software development scenarios is a paramount consideration, offering crucial assistance to software developers in developing secure software.

{{</citation>}}


## cs.LG (24)



### (5/121) COSTAR: Improved Temporal Counterfactual Estimation with Self-Supervised Learning (Chuizheng Meng et al., 2023)

{{<citation>}}

Chuizheng Meng, Yihe Dong, Sercan Ö. Arık, Yan Liu, Tomas Pfister. (2023)  
**COSTAR: Improved Temporal Counterfactual Estimation with Self-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.00886v1)  

---


**ABSTRACT**  
Estimation of temporal counterfactual outcomes from observed history is crucial for decision-making in many domains such as healthcare and e-commerce, particularly when randomized controlled trials (RCTs) suffer from high cost or impracticality. For real-world datasets, modeling time-dependent confounders is challenging due to complex dynamics, long-range dependencies and both past treatments and covariates affecting the future outcomes. In this paper, we introduce COunterfactual Self-supervised TrAnsformeR (COSTAR), a novel approach that integrates self-supervised learning for improved historical representations. The proposed framework combines temporal and feature-wise attention with a component-wise contrastive loss tailored for temporal treatment outcome observations, yielding superior performance in estimation accuracy and generalization to out-of-distribution data compared to existing models, as validated by empirical results on both synthetic and real-world datasets.

{{</citation>}}


### (6/121) SCPO: Safe Reinforcement Learning with Safety Critic Policy Optimization (Jaafar Mhamed et al., 2023)

{{<citation>}}

Jaafar Mhamed, Shangding Gu. (2023)  
**SCPO: Safe Reinforcement Learning with Safety Critic Policy Optimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.00880v1)  

---


**ABSTRACT**  
Incorporating safety is an essential prerequisite for broadening the practical applications of reinforcement learning in real-world scenarios. To tackle this challenge, Constrained Markov Decision Processes (CMDPs) are leveraged, which introduce a distinct cost function representing safety violations. In CMDPs' settings, Lagrangian relaxation technique has been employed in previous algorithms to convert constrained optimization problems into unconstrained dual problems. However, these algorithms may inaccurately predict unsafe behavior, resulting in instability while learning the Lagrange multiplier. This study introduces a novel safe reinforcement learning algorithm, Safety Critic Policy Optimization (SCPO). In this study, we define the safety critic, a mechanism that nullifies rewards obtained through violating safety constraints. Furthermore, our theoretical analysis indicates that the proposed algorithm can automatically balance the trade-off between adhering to safety constraints and maximizing rewards. The effectiveness of the SCPO algorithm is empirically validated by benchmarking it against strong baselines.

{{</citation>}}


### (7/121) Pretraining Data Mixtures Enable Narrow Model Selection Capabilities in Transformer Models (Steve Yadlowsky et al., 2023)

{{<citation>}}

Steve Yadlowsky, Lyric Doshi, Nilesh Tripuraneni. (2023)  
**Pretraining Data Mixtures Enable Narrow Model Selection Capabilities in Transformer Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.00871v1)  

---


**ABSTRACT**  
Transformer models, notably large language models (LLMs), have the remarkable ability to perform in-context learning (ICL) -- to perform new tasks when prompted with unseen input-output examples without any explicit model training. In this work, we study how effectively transformers can bridge between their pretraining data mixture, comprised of multiple distinct task families, to identify and learn new tasks in-context which are both inside and outside the pretraining distribution. Building on previous work, we investigate this question in a controlled setting, where we study transformer models trained on sequences of $(x, f(x))$ pairs rather than natural language. Our empirical results show transformers demonstrate near-optimal unsupervised model selection capabilities, in their ability to first in-context identify different task families and in-context learn within them when the task families are well-represented in their pretraining data. However when presented with tasks or functions which are out-of-domain of their pretraining data, we demonstrate various failure modes of transformers and degradation of their generalization for even simple extrapolation tasks. Together our results highlight that the impressive ICL abilities of high-capacity sequence models may be more closely tied to the coverage of their pretraining data mixtures than inductive biases that create fundamental generalization capabilities.

{{</citation>}}


### (8/121) Selectively Sharing Experiences Improves Multi-Agent Reinforcement Learning (Matthias Gerstgrasser et al., 2023)

{{<citation>}}

Matthias Gerstgrasser, Tom Danino, Sarah Keren. (2023)  
**Selectively Sharing Experiences Improves Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.00865v1)  

---


**ABSTRACT**  
We present a novel multi-agent RL approach, Selective Multi-Agent Prioritized Experience Relay, in which agents share with other agents a limited number of transitions they observe during training. The intuition behind this is that even a small number of relevant experiences from other agents could help each agent learn. Unlike many other multi-agent RL algorithms, this approach allows for largely decentralized training, requiring only a limited communication channel between agents. We show that our approach outperforms baseline no-sharing decentralized training and state-of-the art multi-agent RL algorithms. Further, sharing only a small number of highly relevant experiences outperforms sharing all experiences between agents, and the performance uplift from selective experience sharing is robust across a range of hyperparameters and DQN variants. A reference implementation of our algorithm is available at https://github.com/mgerstgrasser/super.

{{</citation>}}


### (9/121) Training Dynamics of Contextual N-Grams in Language Models (Lucia Quirke et al., 2023)

{{<citation>}}

Lucia Quirke, Lovis Heindrich, Wes Gurnee, Neel Nanda. (2023)  
**Training Dynamics of Contextual N-Grams in Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.00863v1)  

---


**ABSTRACT**  
Prior work has shown the existence of contextual neurons in language models, including a neuron that activates on German text. We show that this neuron exists within a broader contextual n-gram circuit: we find late layer neurons which recognize and continue n-grams common in German text, but which only activate if the German neuron is active. We investigate the formation of this circuit throughout training and find that it is an example of what we call a second-order circuit. In particular, both the constituent n-gram circuits and the German detection circuit which culminates in the German neuron form with independent functions early in training - the German detection circuit partially through modeling German unigram statistics, and the n-grams by boosting appropriate completions. Only after both circuits have already formed do they fit together into a second-order circuit. Contrary to the hypotheses presented in prior work, we find that the contextual n-gram circuit forms gradually rather than in a sudden phase transition. We further present a range of anomalous observations such as a simultaneous phase transition in many tasks coinciding with the learning rate warm-up, and evidence that many context neurons form simultaneously early in training but are later unlearned.

{{</citation>}}


### (10/121) Optimal Cost Constrained Adversarial Attacks For Multiple Agent Systems (Ziqing Lu et al., 2023)

{{<citation>}}

Ziqing Lu, Guanlin Liu, Lifeng Cai, Weiyu Xu. (2023)  
**Optimal Cost Constrained Adversarial Attacks For Multiple Agent Systems**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs-MA, cs.LG  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2311.00859v1)  

---


**ABSTRACT**  
Finding optimal adversarial attack strategies is an important topic in reinforcement learning and the Markov decision process. Previous studies usually assume one all-knowing coordinator (attacker) for whom attacking different recipient (victim) agents incurs uniform costs. However, in reality, instead of using one limitless central attacker, the attacks often need to be performed by distributed attack agents. We formulate the problem of performing optimal adversarial agent-to-agent attacks using distributed attack agents, in which we impose distinct cost constraints on each different attacker-victim pair. We propose an optimal method integrating within-step static constrained attack-resource allocation optimization and between-step dynamic programming to achieve the optimal adversarial attack in a multi-agent system. Our numerical results show that the proposed attacks can significantly reduce the rewards received by the attacked agents.

{{</citation>}}


### (11/121) Language Model Training Paradigms for Clinical Feature Embeddings (Yurong Hu et al., 2023)

{{<citation>}}

Yurong Hu, Manuel Burger, Gunnar Rätsch, Rita Kuznetsova. (2023)  
**Language Model Training Paradigms for Clinical Feature Embeddings**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Clinical, Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2311.00768v1)  

---


**ABSTRACT**  
In research areas with scarce data, representation learning plays a significant role. This work aims to enhance representation learning for clinical time series by deriving universal embeddings for clinical features, such as heart rate and blood pressure. We use self-supervised training paradigms for language models to learn high-quality clinical feature embeddings, achieving a finer granularity than existing time-step and patient-level representation learning. We visualize the learnt embeddings via unsupervised dimension reduction techniques and observe a high degree of consistency with prior clinical knowledge. We also evaluate the model performance on the MIMIC-III benchmark and demonstrate the effectiveness of using clinical feature embeddings. We publish our code online for replication.

{{</citation>}}


### (12/121) FAIRLABEL: Correcting Bias in Labels (Srinivasan H Sengamedu et al., 2023)

{{<citation>}}

Srinivasan H Sengamedu, Hien Pham. (2023)  
**FAIRLABEL: Correcting Bias in Labels**  

---
Primary Category: cs.LG  
Categories: 68T07, I-2-6, cs-AI, cs-LG, cs.LG  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2311.00638v1)  

---


**ABSTRACT**  
There are several algorithms for measuring fairness of ML models. A fundamental assumption in these approaches is that the ground truth is fair or unbiased. In real-world datasets, however, the ground truth often contains data that is a result of historical and societal biases and discrimination. Models trained on these datasets will inherit and propagate the biases to the model outputs. We propose FAIRLABEL, an algorithm which detects and corrects biases in labels. The goal of FAIRLABELis to reduce the Disparate Impact (DI) across groups while maintaining high accuracy in predictions. We propose metrics to measure the quality of bias correction and validate FAIRLABEL on synthetic datasets and show that the label correction is correct 86.7% of the time vs. 71.9% for a baseline model. We also apply FAIRLABEL on benchmark datasets such as UCI Adult, German Credit Risk, and Compas datasets and show that the Disparate Impact Ratio increases by as much as 54.2%.

{{</citation>}}


### (13/121) Learning to optimize by multi-gradient for multi-objective optimization (Linxi Yang et al., 2023)

{{<citation>}}

Linxi Yang, Xinmin Yang, Liping Tang. (2023)  
**Learning to optimize by multi-gradient for multi-objective optimization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.00559v1)  

---


**ABSTRACT**  
The development of artificial intelligence (AI) for science has led to the emergence of learning-based research paradigms, necessitating a compelling reevaluation of the design of multi-objective optimization (MOO) methods. The new generation MOO methods should be rooted in automated learning rather than manual design. In this paper, we introduce a new automatic learning paradigm for optimizing MOO problems, and propose a multi-gradient learning to optimize (ML2O) method, which automatically learns a generator (or mappings) from multiple gradients to update directions. As a learning-based method, ML2O acquires knowledge of local landscapes by leveraging information from the current step and incorporates global experience extracted from historical iteration trajectory data. By introducing a new guarding mechanism, we propose a guarded multi-gradient learning to optimize (GML2O) method, and prove that the iterative sequence generated by GML2O converges to a Pareto critical point. The experimental results demonstrate that our learned optimizer outperforms hand-designed competitors on training multi-task learning (MTL) neural network.

{{</citation>}}


### (14/121) Learning impartial policies for sequential counterfactual explanations using Deep Reinforcement Learning (E. Panagiotou et al., 2023)

{{<citation>}}

E. Panagiotou, E. Ntoutsi. (2023)  
**Learning impartial policies for sequential counterfactual explanations using Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.00523v1)  

---


**ABSTRACT**  
In the field of explainable Artificial Intelligence (XAI), sequential counterfactual (SCF) examples are often used to alter the decision of a trained classifier by implementing a sequence of modifications to the input instance. Although certain test-time algorithms aim to optimize for each new instance individually, recently Reinforcement Learning (RL) methods have been proposed that seek to learn policies for discovering SCFs, thereby enhancing scalability. As is typical in RL, the formulation of the RL problem, including the specification of state space, actions, and rewards, can often be ambiguous. In this work, we identify shortcomings in existing methods that can result in policies with undesired properties, such as a bias towards specific actions. We propose to use the output probabilities of the classifier to create a more informative reward, to mitigate this effect.

{{</citation>}}


### (15/121) Retrieval-Based Reconstruction For Time-series Contrastive Learning (Maxwell A. Xu et al., 2023)

{{<citation>}}

Maxwell A. Xu, Alexander Moreno, Hui Wei, Benjamin M. Marlin, James M. Rehg. (2023)  
**Retrieval-Based Reconstruction For Time-series Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.00519v1)  

---


**ABSTRACT**  
The success of self-supervised contrastive learning hinges on identifying positive data pairs that, when pushed together in embedding space, encode useful information for subsequent downstream tasks. However, in time-series, this is challenging because creating positive pairs via augmentations may break the original semantic meaning. We hypothesize that if we can retrieve information from one subsequence to successfully reconstruct another subsequence, then they should form a positive pair. Harnessing this intuition, we introduce our novel approach: REtrieval-BAsed Reconstruction (REBAR) contrastive learning. First, we utilize a convolutional cross-attention architecture to calculate the REBAR error between two different time-series. Then, through validation experiments, we show that the REBAR error is a predictor of mutual class membership, justifying its usage as a positive/negative labeler. Finally, once integrated into a contrastive learning framework, our REBAR method can learn an embedding that achieves state-of-the-art performance on downstream tasks across various modalities.

{{</citation>}}


### (16/121) Efficient LLM Inference on CPUs (Haihao Shen et al., 2023)

{{<citation>}}

Haihao Shen, Hanwen Chang, Bo Dong, Yu Luo, Hengyu Meng. (2023)  
**Efficient LLM Inference on CPUs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.00502v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated remarkable performance and tremendous potential across a wide range of tasks. However, deploying these models has been challenging due to the astronomical amount of model parameters, which requires a demand for large memory capacity and high memory bandwidth. In this paper, we propose an effective approach that can make the deployment of LLMs more efficiently. We support an automatic INT4 weight-only quantization flow and design a special LLM runtime with highly-optimized kernels to accelerate the LLM inference on CPUs. We demonstrate the general applicability of our approach on popular LLMs including Llama2, Llama, GPT-NeoX, and showcase the extreme inference efficiency on CPUs. The code is publicly available at: https://github.com/intel/intel-extension-for-transformers.

{{</citation>}}


### (17/121) Fixed-Budget Best-Arm Identification in Sparse Linear Bandits (Recep Can Yavas et al., 2023)

{{<citation>}}

Recep Can Yavas, Vincent Y. F. Tan. (2023)  
**Fixed-Budget Best-Arm Identification in Sparse Linear Bandits**  

---
Primary Category: cs.LG  
Categories: I-2-6, cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.00481v1)  

---


**ABSTRACT**  
We study the best-arm identification problem in sparse linear bandits under the fixed-budget setting. In sparse linear bandits, the unknown feature vector $\theta^*$ may be of large dimension $d$, but only a few, say $s \ll d$ of these features have non-zero values. We design a two-phase algorithm, Lasso and Optimal-Design- (Lasso-OD) based linear best-arm identification. The first phase of Lasso-OD leverages the sparsity of the feature vector by applying the thresholded Lasso introduced by Zhou (2009), which estimates the support of $\theta^*$ correctly with high probability using rewards from the selected arms and a judicious choice of the design matrix. The second phase of Lasso-OD applies the OD-LinBAI algorithm by Yang and Tan (2022) on that estimated support. We derive a non-asymptotic upper bound on the error probability of Lasso-OD by carefully choosing hyperparameters (such as Lasso's regularization parameter) and balancing the error probabilities of both phases. For fixed sparsity $s$ and budget $T$, the exponent in the error probability of Lasso-OD depends on $s$ but not on the dimension $d$, yielding a significant performance improvement for sparse and high-dimensional linear bandits. Furthermore, we show that Lasso-OD is almost minimax optimal in the exponent. Finally, we provide numerical examples to demonstrate the significant performance improvement over the existing algorithms for non-sparse linear bandits such as OD-LinBAI, BayesGap, Peace, LinearExploration, and GSE.

{{</citation>}}


### (18/121) NEO-KD: Knowledge-Distillation-Based Adversarial Training for Robust Multi-Exit Neural Networks (Seokil Ham et al., 2023)

{{<citation>}}

Seokil Ham, Jungwuk Park, Dong-Jun Han, Jaekyun Moon. (2023)  
**NEO-KD: Knowledge-Distillation-Based Adversarial Training for Robust Multi-Exit Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Adversarial Training, Knowledge-Distillation  
[Paper Link](http://arxiv.org/abs/2311.00428v1)  

---


**ABSTRACT**  
While multi-exit neural networks are regarded as a promising solution for making efficient inference via early exits, combating adversarial attacks remains a challenging problem. In multi-exit networks, due to the high dependency among different submodels, an adversarial example targeting a specific exit not only degrades the performance of the target exit but also reduces the performance of all other exits concurrently. This makes multi-exit networks highly vulnerable to simple adversarial attacks. In this paper, we propose NEO-KD, a knowledge-distillation-based adversarial training strategy that tackles this fundamental challenge based on two key contributions. NEO-KD first resorts to neighbor knowledge distillation to guide the output of the adversarial examples to tend to the ensemble outputs of neighbor exits of clean data. NEO-KD also employs exit-wise orthogonal knowledge distillation for reducing adversarial transferability across different submodels. The result is a significantly improved robustness against adversarial attacks. Experimental results on various datasets/models show that our method achieves the best adversarial accuracy with reduced computation budgets, compared to the baselines relying on existing adversarial training or knowledge distillation techniques for multi-exit networks.

{{</citation>}}


### (19/121) Enhanced Generalization through Prioritization and Diversity in Self-Imitation Reinforcement Learning over Procedural Environments with Sparse Rewards (Alain Andres et al., 2023)

{{<citation>}}

Alain Andres, Daochen Zha, Javier Del Ser. (2023)  
**Enhanced Generalization through Prioritization and Diversity in Self-Imitation Reinforcement Learning over Procedural Environments with Sparse Rewards**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.00426v1)  

---


**ABSTRACT**  
Exploration poses a fundamental challenge in Reinforcement Learning (RL) with sparse rewards, limiting an agent's ability to learn optimal decision-making due to a lack of informative feedback signals. Self-Imitation Learning (self-IL) has emerged as a promising approach for exploration, leveraging a replay buffer to store and reproduce successful behaviors. However, traditional self-IL methods, which rely on high-return transitions and assume singleton environments, face challenges in generalization, especially in procedurally-generated (PCG) environments. Therefore, new self-IL methods have been proposed to rank which experiences to persist, but they replay transitions uniformly regardless of their significance, and do not address the diversity of the stored demonstrations. In this work, we propose tailored self-IL sampling strategies by prioritizing transitions in different ways and extending prioritization techniques to PCG environments. We also address diversity loss through modifications to counteract the impact of generalization requirements and bias introduced by prioritization techniques. Our experimental analysis, conducted over three PCG sparse reward environments, including MiniGrid and ProcGen, highlights the benefits of our proposed modifications, achieving a new state-of-the-art performance in the MiniGrid-MultiRoom-N12-S10 environment.

{{</citation>}}


### (20/121) Efficient Human-AI Coordination via Preparatory Language-based Convention (Cong Guan et al., 2023)

{{<citation>}}

Cong Guan, Lichao Zhang, Chunpeng Fan, Yichen Li, Feng Chen, Lihe Li, Yunjia Tian, Lei Yuan, Yang Yu. (2023)  
**Efficient Human-AI Coordination via Preparatory Language-based Convention**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.00416v1)  

---


**ABSTRACT**  
Developing intelligent agents capable of seamless coordination with humans is a critical step towards achieving artificial general intelligence. Existing methods for human-AI coordination typically train an agent to coordinate with a diverse set of policies or with human models fitted from real human data. However, the massively diverse styles of human behavior present obstacles for AI systems with constrained capacity, while high quality human data may not be readily available in real-world scenarios. In this study, we observe that prior to coordination, humans engage in communication to establish conventions that specify individual roles and actions, making their coordination proceed in an orderly manner. Building upon this observation, we propose employing the large language model (LLM) to develop an action plan (or equivalently, a convention) that effectively guides both human and AI. By inputting task requirements, human preferences, the number of agents, and other pertinent information into the LLM, it can generate a comprehensive convention that facilitates a clear understanding of tasks and responsibilities for all parties involved. Furthermore, we demonstrate that decomposing the convention formulation problem into sub-problems with multiple new sessions being sequentially employed and human feedback, will yield a more efficient coordination convention. Experimental evaluations conducted in the Overcooked-AI environment, utilizing a human proxy model, highlight the superior performance of our proposed method compared to existing learning-based approaches. When coordinating with real humans, our method achieves better alignment with human preferences and an average performance improvement of 15% compared to the state-of-the-art.

{{</citation>}}


### (21/121) Multi-task Representation Learning for Pure Exploration in Bilinear Bandits (Subhojyoti Mukherjee et al., 2023)

{{<citation>}}

Subhojyoti Mukherjee, Qiaomin Xie, Josiah P. Hanna, Robert Nowak. (2023)  
**Multi-task Representation Learning for Pure Exploration in Bilinear Bandits**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.00327v1)  

---


**ABSTRACT**  
We study multi-task representation learning for the problem of pure exploration in bilinear bandits. In bilinear bandits, an action takes the form of a pair of arms from two different entity types and the reward is a bilinear function of the known feature vectors of the arms. In the \textit{multi-task bilinear bandit problem}, we aim to find optimal actions for multiple tasks that share a common low-dimensional linear representation. The objective is to leverage this characteristic to expedite the process of identifying the best pair of arms for all tasks. We propose the algorithm GOBLIN that uses an experimental design approach to optimize sample allocations for learning the global representation as well as minimize the number of samples needed to identify the optimal pair of arms in individual tasks. To the best of our knowledge, this is the first study to give sample complexity analysis for pure exploration in bilinear bandits with shared representation. Our results demonstrate that by learning the shared representation across tasks, we achieve significantly improved sample complexity compared to the traditional approach of solving tasks independently.

{{</citation>}}


### (22/121) Robust Graph Clustering via Meta Weighting for Noisy Graphs (Hyeonsoo Jo et al., 2023)

{{<citation>}}

Hyeonsoo Jo, Fanchen Bu, Kijung Shin. (2023)  
**Robust Graph Clustering via Meta Weighting for Noisy Graphs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.00322v1)  

---


**ABSTRACT**  
How can we find meaningful clusters in a graph robustly against noise edges? Graph clustering (i.e., dividing nodes into groups of similar ones) is a fundamental problem in graph analysis with applications in various fields. Recent studies have demonstrated that graph neural network (GNN) based approaches yield promising results for graph clustering. However, we observe that their performance degenerates significantly on graphs with noise edges, which are prevalent in practice. In this work, we propose MetaGC for robust GNN-based graph clustering. MetaGC employs a decomposable clustering loss function, which can be rephrased as a sum of losses over node pairs. We add a learnable weight to each node pair, and MetaGC adaptively adjusts the weights of node pairs using meta-weighting so that the weights of meaningful node pairs increase and the weights of less-meaningful ones (e.g., noise edges) decrease. We show empirically that MetaGC learns weights as intended and consequently outperforms the state-of-the-art GNN-based competitors, even when they are equipped with separate denoising schemes, on five real-world graphs under varying levels of noise. Our code and datasets are available at https://github.com/HyeonsooJo/MetaGC.

{{</citation>}}


### (23/121) Federated Topic Model and Model Pruning Based on Variational Autoencoder (Chengjie Ma et al., 2023)

{{<citation>}}

Chengjie Ma, Yawen Li, Meiyu Liang, Ang Li. (2023)  
**Federated Topic Model and Model Pruning Based on Variational Autoencoder**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: Pruning, Topic Model  
[Paper Link](http://arxiv.org/abs/2311.00314v1)  

---


**ABSTRACT**  
Topic modeling has emerged as a valuable tool for discovering patterns and topics within large collections of documents. However, when cross-analysis involves multiple parties, data privacy becomes a critical concern. Federated topic modeling has been developed to address this issue, allowing multiple parties to jointly train models while protecting pri-vacy. However, there are communication and performance challenges in the federated sce-nario. In order to solve the above problems, this paper proposes a method to establish a federated topic model while ensuring the privacy of each node, and use neural network model pruning to accelerate the model, where the client periodically sends the model neu-ron cumulative gradients and model weights to the server, and the server prunes the model. To address different requirements, two different methods are proposed to determine the model pruning rate. The first method involves slow pruning throughout the entire model training process, which has limited acceleration effect on the model training process, but can ensure that the pruned model achieves higher accuracy. This can significantly reduce the model inference time during the inference process. The second strategy is to quickly reach the target pruning rate in the early stage of model training in order to accelerate the model training speed, and then continue to train the model with a smaller model size after reaching the target pruning rate. This approach may lose more useful information but can complete the model training faster. Experimental results show that the federated topic model pruning based on the variational autoencoder proposed in this paper can greatly accelerate the model training speed while ensuring the model's performance.

{{</citation>}}


### (24/121) Rethinking Decision Transformer via Hierarchical Reinforcement Learning (Yi Ma et al., 2023)

{{<citation>}}

Yi Ma, Chenjun Xiao, Hebin Liang, Jianye Hao. (2023)  
**Rethinking Decision Transformer via Hierarchical Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2311.00267v1)  

---


**ABSTRACT**  
Decision Transformer (DT) is an innovative algorithm leveraging recent advances of the transformer architecture in reinforcement learning (RL). However, a notable limitation of DT is its reliance on recalling trajectories from datasets, losing the capability to seamlessly stitch sub-optimal trajectories together. In this work we introduce a general sequence modeling framework for studying sequential decision making through the lens of Hierarchical RL. At the time of making decisions, a high-level policy first proposes an ideal prompt for the current state, a low-level policy subsequently generates an action conditioned on the given prompt. We show DT emerges as a special case of this framework with certain choices of high-level and low-level policies, and discuss the potential failure of these choices. Inspired by these observations, we study how to jointly optimize the high-level and low-level policies to enable the stitching ability, which further leads to the development of new offline RL algorithms. Our empirical results clearly show that the proposed algorithms significantly surpass DT on several control and navigation benchmarks. We hope our contributions can inspire the integration of transformer architectures within the field of RL.

{{</citation>}}


### (25/121) StableFDG: Style and Attention Based Learning for Federated Domain Generalization (Jungwuk Park et al., 2023)

{{<citation>}}

Jungwuk Park, Dong-Jun Han, Jinho Kim, Shiqiang Wang, Christopher G. Brinton, Jaekyun Moon. (2023)  
**StableFDG: Style and Attention Based Learning for Federated Domain Generalization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.00227v1)  

---


**ABSTRACT**  
Traditional federated learning (FL) algorithms operate under the assumption that the data distributions at training (source domains) and testing (target domain) are the same. The fact that domain shifts often occur in practice necessitates equipping FL methods with a domain generalization (DG) capability. However, existing DG algorithms face fundamental challenges in FL setups due to the lack of samples/domains in each client's local dataset. In this paper, we propose StableFDG, a style and attention based learning strategy for accomplishing federated domain generalization, introducing two key contributions. The first is style-based learning, which enables each client to explore novel styles beyond the original source domains in its local dataset, improving domain diversity based on the proposed style sharing, shifting, and exploration strategies. Our second contribution is an attention-based feature highlighter, which captures the similarities between the features of data samples in the same class, and emphasizes the important/common characteristics to better learn the domain-invariant characteristics of each class in data-poor FL scenarios. Experimental results show that StableFDG outperforms existing baselines on various DG benchmark datasets, demonstrating its efficacy.

{{</citation>}}


### (26/121) WinNet:time series forecasting with a window-enhanced period extracting and interacting (Wenjie Ou et al., 2023)

{{<citation>}}

Wenjie Ou, Dongyue Guo, Zheng Zhang, Zhishuo Zhao, Yi Lin. (2023)  
**WinNet:time series forecasting with a window-enhanced period extracting and interacting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.00214v1)  

---


**ABSTRACT**  
Recently, Transformer-based methods have significantly improved state-of-the-art time series forecasting results, but they suffer from high computational costs and the inability to capture the long and short periodicity of time series. We present a highly accurate and simply structured CNN-based model for long-term time series forecasting tasks, called WinNet, including (i) Inter-Intra Period Encoder (I2PE) to transform 1D sequence into 2D tensor with long and short periodicity according to the predefined periodic window, (ii) Two-Dimensional Period Decomposition (TDPD) to model period-trend and oscillation terms, and (iii) Decomposition Correlation Block (DCB) to leverage the correlations of the period-trend and oscillation terms to support the prediction tasks by CNNs. Results on nine benchmark datasets show that the WinNet can achieve SOTA performance and lower computational complexity over CNN-, MLP-, Transformer-based approaches. The WinNet provides potential for the CNN-based methods in the time series forecasting tasks, with perfect tradeoff between performance and efficiency.

{{</citation>}}


### (27/121) Transformers as Recognizers of Formal Languages: A Survey on Expressivity (Lena Strobl et al., 2023)

{{<citation>}}

Lena Strobl, William Merrill, Gail Weiss, David Chiang, Dana Angluin. (2023)  
**Transformers as Recognizers of Formal Languages: A Survey on Expressivity**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-FL, cs-LG, cs-LO, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.00208v1)  

---


**ABSTRACT**  
As transformers have gained prominence in natural language processing, some researchers have investigated theoretically what problems they can and cannot solve, by treating problems as formal languages. Exploring questions such as this will help to compare transformers with other models, and transformer variants with one another, for various tasks. Work in this subarea has made considerable progress in recent years. Here, we undertake a comprehensive survey of this work, documenting the diverse assumptions that underlie different results and providing a unified framework for harmonizing seemingly contradictory findings.

{{</citation>}}


### (28/121) Federated Natural Policy Gradient Methods for Multi-task Reinforcement Learning (Tong Yang et al., 2023)

{{<citation>}}

Tong Yang, Shicong Cen, Yuting Wei, Yuxin Chen, Yuejie Chi. (2023)  
**Federated Natural Policy Gradient Methods for Multi-task Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.00201v1)  

---


**ABSTRACT**  
Federated reinforcement learning (RL) enables collaborative decision making of multiple distributed agents without sharing local data trajectories. In this work, we consider a multi-task setting, in which each agent has its own private reward function corresponding to different tasks, while sharing the same transition kernel of the environment. Focusing on infinite-horizon tabular Markov decision processes, the goal is to learn a globally optimal policy that maximizes the sum of the discounted total rewards of all the agents in a decentralized manner, where each agent only communicates with its neighbors over some prescribed graph topology. We develop federated vanilla and entropy-regularized natural policy gradient (NPG) methods under softmax parameterization, where gradient tracking is applied to the global Q-function to mitigate the impact of imperfect information sharing. We establish non-asymptotic global convergence guarantees under exact policy evaluation, which are nearly independent of the size of the state-action space and illuminate the impacts of network size and connectivity. To the best of our knowledge, this is the first time that global convergence is established for federated multi-task RL using policy optimization. Moreover, the convergence behavior of the proposed algorithms is robust against inexactness of policy evaluation.

{{</citation>}}


## cs.SD (3)



### (29/121) Low-latency Real-time Voice Conversion on CPU (Konstantine Sadov et al., 2023)

{{<citation>}}

Konstantine Sadov, Matthew Hutter, Asara Near. (2023)  
**Low-latency Real-time Voice Conversion on CPU**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.00873v1)  

---


**ABSTRACT**  
We adapt the architectures of previous audio manipulation and generation neural networks to the task of real-time any-to-one voice conversion. Our resulting model, LLVC ($\textbf{L}$ow-latency $\textbf{L}$ow-resource $\textbf{V}$oice $\textbf{C}$onversion), has a latency of under 20ms at a bitrate of 16kHz and runs nearly 2.8x faster than real-time on a consumer CPU. LLVC uses both a generative adversarial architecture as well as knowledge distillation in order to attain this performance. To our knowledge LLVC achieves both the lowest resource usage as well as the lowest latency of any open-source voice conversion model. We provide open-source samples, code, and pretrained model weights at https://github.com/KoeAI/LLVC.

{{</citation>}}


### (30/121) Investigating Self-Supervised Deep Representations for EEG-based Auditory Attention Decoding (Karan Thakkar et al., 2023)

{{<citation>}}

Karan Thakkar, Jiarui Hai, Mounya Elhilali. (2023)  
**Investigating Self-Supervised Deep Representations for EEG-based Auditory Attention Decoding**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Attention, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.00814v1)  

---


**ABSTRACT**  
Auditory Attention Decoding (AAD) algorithms play a crucial role in isolating desired sound sources within challenging acoustic environments directly from brain activity. Although recent research has shown promise in AAD using shallow representations such as auditory envelope and spectrogram, there has been limited exploration of deep Self-Supervised (SS) representations on a larger scale. In this study, we undertake a comprehensive investigation into the performance of linear decoders across 12 deep and 2 shallow representations, applied to EEG data from multiple studies spanning 57 subjects and multiple languages. Our experimental results consistently reveal the superiority of deep features for AAD at decoding background speakers, regardless of the datasets and analysis windows. This result indicates possible nonlinear encoding of unattended signals in the brain that are revealed using deep nonlinear features. Additionally, we analyze the impact of different layers of SS representations and window sizes on AAD performance. These findings underscore the potential for enhancing EEG-based AAD systems through the integration of deep feature representations.

{{</citation>}}


### (31/121) Detecting Syllable-Level Pronunciation Stress with A Self-Attention Model (Wang Weiying et al., 2023)

{{<citation>}}

Wang Weiying, Nakajima Akinori. (2023)  
**Detecting Syllable-Level Pronunciation Stress with A Self-Attention Model**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2311.00301v1)  

---


**ABSTRACT**  
One precondition of effective oral communication is that words should be pronounced clearly, especially for non-native speakers. Word stress is the key to clear and correct English, and misplacement of syllable stress may lead to misunderstandings. Thus, knowing the stress level is important for English speakers and learners. This paper presents a self-attention model to identify the stress level for each syllable of spoken English. Various prosodic and categorical features, including the pitch level, intensity, duration and type of the syllable and its nuclei (the vowel of the syllable), are explored. These features are input to the self-attention model, and syllable-level stresses are predicted. The simplest model yields an accuracy of over 88% and 93% on different datasets, while more advanced models provide higher accuracy. Our study suggests that the self-attention model can be promising in stress-level detection. These models could be applied to various scenarios, such as online meetings and English learning.

{{</citation>}}


## eess.AS (1)



### (32/121) Automatic Disfluency Detection from Untranscribed Speech (Amrit Romana et al., 2023)

{{<citation>}}

Amrit Romana, Kazuhito Koishida, Emily Mower Provost. (2023)  
**Automatic Disfluency Detection from Untranscribed Speech**  

---
Primary Category: eess.AS  
Categories: cs-CL, eess-AS, eess.AS  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2311.00867v1)  

---


**ABSTRACT**  
Speech disfluencies, such as filled pauses or repetitions, are disruptions in the typical flow of speech. Stuttering is a speech disorder characterized by a high rate of disfluencies, but all individuals speak with some disfluencies and the rates of disfluencies may by increased by factors such as cognitive load. Clinically, automatic disfluency detection may help in treatment planning for individuals who stutter. Outside of the clinic, automatic disfluency detection may serve as a pre-processing step to improve natural language understanding in downstream applications. With this wide range of applications in mind, we investigate language, acoustic, and multimodal methods for frame-level automatic disfluency detection and categorization. Each of these methods relies on audio as an input. First, we evaluate several automatic speech recognition (ASR) systems in terms of their ability to transcribe disfluencies, measured using disfluency error rates. We then use these ASR transcripts as input to a language-based disfluency detection model. We find that disfluency detection performance is largely limited by the quality of transcripts and alignments. We find that an acoustic-based approach that does not require transcription as an intermediate step outperforms the ASR language approach. Finally, we present multimodal architectures which we find improve disfluency detection performance over the unimodal approaches. Ultimately, this work introduces novel approaches for automatic frame-level disfluency and categorization. In the long term, this will help researchers incorporate automatic disfluency detection into a range of applications.

{{</citation>}}


## cs.AI (15)



### (33/121) A Multi-Agent Reinforcement Learning Framework for Evaluating the U.S. Ending the HIV Epidemic Plan (Dinesh Sharma et al., 2023)

{{<citation>}}

Dinesh Sharma, Ankit Shah, Chaitra Gopalappa. (2023)  
**A Multi-Agent Reinforcement Learning Framework for Evaluating the U.S. Ending the HIV Epidemic Plan**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-MA, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.00855v1)  

---


**ABSTRACT**  
Human immunodeficiency virus (HIV) is a major public health concern in the United States, with about 1.2 million people living with HIV and 35,000 newly infected each year. There are considerable geographical disparities in HIV burden and care access across the U.S. The 2019 Ending the HIV Epidemic (EHE) initiative aims to reduce new infections by 90% by 2030, by improving coverage of diagnoses, treatment, and prevention interventions and prioritizing jurisdictions with high HIV prevalence. Identifying optimal scale-up of intervention combinations will help inform resource allocation. Existing HIV decision analytic models either evaluate specific cities or the overall national population, thus overlooking jurisdictional interactions or differences. In this paper, we propose a multi-agent reinforcement learning (MARL) model, that enables jurisdiction-specific decision analyses but in an environment with cross-jurisdictional epidemiological interactions. In experimental analyses, conducted on jurisdictions within California and Florida, optimal policies from MARL were significantly different than those generated from single-agent RL, highlighting the influence of jurisdictional variations and interactions. By using comprehensive modeling of HIV and formulations of state space, action space, and reward functions, this work helps demonstrate the strengths and applicability of MARL for informing public health policies, and provides a framework for expanding to the national-level to inform the EHE.

{{</citation>}}


### (34/121) Hand Gesture Classification on Praxis Dataset: Trading Accuracy for Expense (Rahat Islam et al., 2023)

{{<citation>}}

Rahat Islam, Kenneth Lai, Svetlana Yanushkevich. (2023)  
**Hand Gesture Classification on Praxis Dataset: Trading Accuracy for Expense**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.00767v1)  

---


**ABSTRACT**  
In this paper, we investigate hand gesture classifiers that rely upon the abstracted 'skeletal' data recorded using the RGB-Depth sensor. We focus on 'skeletal' data represented by the body joint coordinates, from the Praxis dataset. The PRAXIS dataset contains recordings of patients with cortical pathologies such as Alzheimer's disease, performing a Praxis test under the direction of a clinician. In this paper, we propose hand gesture classifiers that are more effective with the PRAXIS dataset than previously proposed models. Body joint data offers a compressed form of data that can be analyzed specifically for hand gesture recognition. Using a combination of windowing techniques with deep learning architecture such as a Recurrent Neural Network (RNN), we achieved an overall accuracy of 70.8% using only body joint data. In addition, we investigated a long-short-term-memory (LSTM) to extract and analyze the movement of the joints through time to recognize the hand gestures being performed and achieved a gesture recognition rate of 74.3% and 67.3% for static and dynamic gestures, respectively. The proposed approach contributed to the task of developing an automated, accurate, and inexpensive approach to diagnosing cortical pathologies for multiple healthcare applications.

{{</citation>}}


### (35/121) Unleashing the Creative Mind: Language Model As Hierarchical Policy For Improved Exploration on Challenging Problem Solving (Zhan Ling et al., 2023)

{{<citation>}}

Zhan Ling, Yunhao Fang, Xuanlin Li, Tongzhou Mu, Mingu Lee, Reza Pourreza, Roland Memisevic, Hao Su. (2023)  
**Unleashing the Creative Mind: Language Model As Hierarchical Policy For Improved Exploration on Challenging Problem Solving**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.00694v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have achieved tremendous progress, yet they still often struggle with challenging reasoning problems. Current approaches address this challenge by sampling or searching detailed and low-level reasoning chains. However, these methods are still limited in their exploration capabilities, making it challenging for correct solutions to stand out in the huge solution space. In this work, we unleash LLMs' creative potential for exploring multiple diverse problem solving strategies by framing an LLM as a hierarchical policy via in-context learning. This policy comprises of a visionary leader that proposes multiple diverse high-level problem-solving tactics as hints, accompanied by a follower that executes detailed problem-solving processes following each of the high-level instruction. The follower uses each of the leader's directives as a guide and samples multiple reasoning chains to tackle the problem, generating a solution group for each leader proposal. Additionally, we propose an effective and efficient tournament-based approach to select among these explored solution groups to reach the final answer. Our approach produces meaningful and inspiring hints, enhances problem-solving strategy exploration, and improves the final answer accuracy on challenging problems in the MATH dataset. Code will be released at https://github.com/lz1oceani/LLM-As-Hierarchical-Policy.

{{</citation>}}


### (36/121) On Task-personalized Multimodal Few-shot Learning for Visually-rich Document Entity Retrieval (Jiayi Chen et al., 2023)

{{<citation>}}

Jiayi Chen, Hanjun Dai, Bo Dai, Aidong Zhang, Wei Wei. (2023)  
**On Task-personalized Multimodal Few-shot Learning for Visually-rich Document Entity Retrieval**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.00693v1)  

---


**ABSTRACT**  
Visually-rich document entity retrieval (VDER), which extracts key information (e.g. date, address) from document images like invoices and receipts, has become an important topic in industrial NLP applications. The emergence of new document types at a constant pace, each with its unique entity types, presents a unique challenge: many documents contain unseen entity types that occur only a couple of times. Addressing this challenge requires models to have the ability of learning entities in a few-shot manner. However, prior works for Few-shot VDER mainly address the problem at the document level with a predefined global entity space, which doesn't account for the entity-level few-shot scenario: target entity types are locally personalized by each task and entity occurrences vary significantly among documents. To address this unexplored scenario, this paper studies a novel entity-level few-shot VDER task. The challenges lie in the uniqueness of the label space for each task and the increased complexity of out-of-distribution (OOD) contents. To tackle this novel task, we present a task-aware meta-learning based framework, with a central focus on achieving effective task personalization that distinguishes between in-task and out-of-task distribution. Specifically, we adopt a hierarchical decoder (HC) and employ contrastive learning (ContrastProtoNet) to achieve this goal. Furthermore, we introduce a new dataset, FewVEX, to boost future research in the field of entity-level few-shot VDER. Experimental results demonstrate our approaches significantly improve the robustness of popular meta-learning baselines.

{{</citation>}}


### (37/121) Improving Interpersonal Communication by Simulating Audiences with Language Models (Ryan Liu et al., 2023)

{{<citation>}}

Ryan Liu, Howard Yen, Raja Marjieh, Thomas L. Griffiths, Ranjay Krishna. (2023)  
**Improving Interpersonal Communication by Simulating Audiences with Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-HC, cs-LG, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.00687v2)  

---


**ABSTRACT**  
How do we communicate with others to achieve our goals? We use our prior experience or advice from others, or construct a candidate utterance by predicting how it will be received. However, our experiences are limited and biased, and reasoning about potential outcomes can be difficult and cognitively challenging. In this paper, we explore how we can leverage Large Language Model (LLM) simulations to help us communicate better. We propose the Explore-Generate-Simulate (EGS) framework, which takes as input any scenario where an individual is communicating to an audience with a goal they want to achieve. EGS (1) explores the solution space by producing a diverse set of advice relevant to the scenario, (2) generates communication candidates conditioned on subsets of the advice, and (3) simulates the reactions from various audiences to determine both the best candidate and advice to use. We evaluate the framework on eight scenarios spanning the ten fundamental processes of interpersonal communication. For each scenario, we collect a dataset of human evaluations across candidates and baselines, and showcase that our framework's chosen candidate is preferred over popular generation mechanisms including Chain-of-Thought. We also find that audience simulations achieve reasonably high agreement with human raters across 5 of the 8 scenarios. Finally, we demonstrate the generality of our framework by applying it to real-world scenarios described by users on web forums. Through evaluations and demonstrations, we show that EGS enhances the effectiveness and outcomes of goal-oriented communication across a variety of situations, thus opening up new possibilities for the application of large language models in revolutionizing communication and decision-making processes.

{{</citation>}}


### (38/121) Can Foundation Models Watch, Talk and Guide You Step by Step to Make a Cake? (Yuwei Bao et al., 2023)

{{<citation>}}

Yuwei Bao, Keunwoo Peter Yu, Yichi Zhang, Shane Storks, Itamar Bar-Yossef, Alexander De La Iglesia, Megan Su, Xiao Lin Zheng, Joyce Chai. (2023)  
**Can Foundation Models Watch, Talk and Guide You Step by Step to Make a Cake?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.00738v1)  

---


**ABSTRACT**  
Despite tremendous advances in AI, it remains a significant challenge to develop interactive task guidance systems that can offer situated, personalized guidance and assist humans in various tasks. These systems need to have a sophisticated understanding of the user as well as the environment, and make timely accurate decisions on when and what to say. To address this issue, we created a new multimodal benchmark dataset, Watch, Talk and Guide (WTaG) based on natural interaction between a human user and a human instructor. We further proposed two tasks: User and Environment Understanding, and Instructor Decision Making. We leveraged several foundation models to study to what extent these models can be quickly adapted to perceptually enabled task guidance. Our quantitative, qualitative, and human evaluation results show that these models can demonstrate fair performances in some cases with no task-specific training, but a fast and reliable adaptation remains a significant challenge. Our benchmark and baselines will provide a stepping stone for future work on situated task guidance.

{{</citation>}}


### (39/121) Tackling the Abstraction and Reasoning Corpus (ARC) with Object-centric Models and the MDL Principle (Sébastien Ferré, 2023)

{{<citation>}}

Sébastien Ferré. (2023)  
**Tackling the Abstraction and Reasoning Corpus (ARC) with Object-centric Models and the MDL Principle**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.00545v1)  

---


**ABSTRACT**  
The Abstraction and Reasoning Corpus (ARC) is a challenging benchmark, introduced to foster AI research towards human-level intelligence. It is a collection of unique tasks about generating colored grids, specified by a few examples only. In contrast to the transformation-based programs of existing work, we introduce object-centric models that are in line with the natural programs produced by humans. Our models can not only perform predictions, but also provide joint descriptions for input/output pairs. The Minimum Description Length (MDL) principle is used to efficiently search the large model space. A diverse range of tasks are solved, and the learned models are similar to the natural programs. We demonstrate the generality of our approach by applying it to a different domain.

{{</citation>}}


### (40/121) The Development of LLMs for Embodied Navigation (Jinzhou Lin et al., 2023)

{{<citation>}}

Jinzhou Lin, Han Gao, Rongtao Xu, Changwei Wang, Li Guo, Shibiao Xu. (2023)  
**The Development of LLMs for Embodied Navigation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2311.00530v1)  

---


**ABSTRACT**  
In recent years, the rapid advancement of Large Language Models (LLMs) such as the Generative Pre-trained Transformer (GPT) has attracted increasing attention due to their potential in a variety of practical applications. The application of LLMs with Embodied Intelligence has emerged as a significant area of focus. Among the myriad applications of LLMs, navigation tasks are particularly noteworthy because they demand a deep understanding of the environment and quick, accurate decision-making. LLMs can augment embodied intelligence systems with sophisticated environmental perception and decision-making support, leveraging their robust language and image-processing capabilities. This article offers an exhaustive summary of the symbiosis between LLMs and embodied intelligence with a focus on navigation. It reviews state-of-the-art models, research methodologies, and assesses the advantages and disadvantages of existing embodied navigation models and datasets. Finally, the article elucidates the role of LLMs in embodied intelligence, based on current research, and forecasts future directions in the field. A comprehensive list of studies in this survey is available at https://github.com/Rongtao-Xu/Awesome-LLM-EN

{{</citation>}}


### (41/121) Leveraging Hyperbolic Embeddings for Coarse-to-Fine Robot Design (Heng Dong et al., 2023)

{{<citation>}}

Heng Dong, Junyu Zhang, Chongjie Zhang. (2023)  
**Leveraging Hyperbolic Embeddings for Coarse-to-Fine Robot Design**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.00462v2)  

---


**ABSTRACT**  
Multi-cellular robot design aims to create robots comprised of numerous cells that can be efficiently controlled to perform diverse tasks. Previous research has demonstrated the ability to generate robots for various tasks, but these approaches often optimize robots directly in the vast design space, resulting in robots with complicated morphologies that are hard to control. In response, this paper presents a novel coarse-to-fine method for designing multi-cellular robots. Initially, this strategy seeks optimal coarse-grained robots and progressively refines them. To mitigate the challenge of determining the precise refinement juncture during the coarse-to-fine transition, we introduce the Hyperbolic Embeddings for Robot Design (HERD) framework. HERD unifies robots of various granularity within a shared hyperbolic space and leverages a refined Cross-Entropy Method for optimization. This framework enables our method to autonomously identify areas of exploration in hyperbolic space and concentrate on regions demonstrating promise. Finally, the extensive empirical studies on various challenging tasks sourced from EvoGym show our approach's superior efficiency and generalization capability.

{{</citation>}}


### (42/121) On the Opportunities of Green Computing: A Survey (You Zhou et al., 2023)

{{<citation>}}

You Zhou, Xiujing Lin, Xiang Zhang, Maolin Wang, Gangwei Jiang, Huakang Lu, Yupeng Wu, Kai Zhang, Zhe Yang, Kehang Wang, Yongduo Sui, Fengwei Jia, Zuoli Tang, Yao Zhao, Hongxuan Zhang, Tiannuo Yang, Weibo Chen, Yunong Mao, Yi Li, De Bao, Yu Li, Hongrui Liao, Ting Liu, Jingwen Liu, Jinchi Guo, Jin Zhao, Xiangyu Zhao, Ying WEI, Hong Qian, Qi Liu, Xiang Wang, Wai Kin, Chan, Chenliang Li, Yusen Li, Shiyu Yang, Jining Yan, Chao Mou, Shuai Han, Wuxia Jin, Guannan Zhang, Xiaodong Zeng. (2023)  
**On the Opportunities of Green Computing: A Survey**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.00447v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) has achieved significant advancements in technology and research with the development over several decades, and is widely used in many areas including computing vision, natural language processing, time-series analysis, speech synthesis, etc. During the age of deep learning, especially with the arise of Large Language Models, a large majority of researchers' attention is paid on pursuing new state-of-the-art (SOTA) results, resulting in ever increasing of model size and computational complexity. The needs for high computing power brings higher carbon emission and undermines research fairness by preventing small or medium-sized research institutions and companies with limited funding in participating in research. To tackle the challenges of computing resources and environmental impact of AI, Green Computing has become a hot research topic. In this survey, we give a systematic overview of the technologies used in Green Computing. We propose the framework of Green Computing and devide it into four key components: (1) Measures of Greenness, (2) Energy-Efficient AI, (3) Energy-Efficient Computing Systems and (4) AI Use Cases for Sustainability. For each components, we discuss the research progress made and the commonly used techniques to optimize the AI efficiency. We conclude that this new research direction has the potential to address the conflicts between resource constraints and AI development. We encourage more researchers to put attention on this direction and make AI more environmental friendly.

{{</citation>}}


### (43/121) Augmenting deep neural networks with symbolic knowledge: Towards trustworthy and interpretable AI for education (Danial Hooshyar et al., 2023)

{{<citation>}}

Danial Hooshyar, Roger Azevedo, Yeongwook Yang. (2023)  
**Augmenting deep neural networks with symbolic knowledge: Towards trustworthy and interpretable AI for education**  

---
Primary Category: cs.AI  
Categories: I-2-0, I-2-1, I-2-6, cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.00393v1)  

---


**ABSTRACT**  
Artificial neural networks (ANNs) have shown to be amongst the most important artificial intelligence (AI) techniques in educational applications, providing adaptive educational services. However, their educational potential is limited in practice due to three major challenges: i) difficulty in incorporating symbolic educational knowledge (e.g., causal relationships, and practitioners' knowledge) in their development, ii) learning and reflecting biases, and iii) lack of interpretability. Given the high-risk nature of education, the integration of educational knowledge into ANNs becomes crucial for developing AI applications that adhere to essential educational restrictions, and provide interpretability over the predictions. This research argues that the neural-symbolic family of AI has the potential to address the named challenges. To this end, it adapts a neural-symbolic AI framework and accordingly develops an approach called NSAI, that injects and extracts educational knowledge into and from deep neural networks, for modelling learners computational thinking. Our findings reveal that the NSAI approach has better generalizability compared to deep neural networks trained merely on training data, as well as training data augmented by SMOTE and autoencoder methods. More importantly, unlike the other models, the NSAI approach prioritises robust representations that capture causal relationships between input features and output labels, ensuring safety in learning to avoid spurious correlations and control biases in training data. Furthermore, the NSAI approach enables the extraction of rules from the learned network, facilitating interpretation and reasoning about the path to predictions, as well as refining the initial educational knowledge. These findings imply that neural-symbolic AI can overcome the limitations of ANNs in education, enabling trustworthy and interpretable applications.

{{</citation>}}


### (44/121) QFree: A Universal Value Function Factorization for Multi-Agent Reinforcement Learning (Rizhong Wang et al., 2023)

{{<citation>}}

Rizhong Wang, Huiping Li, Di Cui, Demin Xu. (2023)  
**QFree: A Universal Value Function Factorization for Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.00356v1)  

---


**ABSTRACT**  
Centralized training is widely utilized in the field of multi-agent reinforcement learning (MARL) to assure the stability of training process. Once a joint policy is obtained, it is critical to design a value function factorization method to extract optimal decentralized policies for the agents, which needs to satisfy the individual-global-max (IGM) principle. While imposing additional limitations on the IGM function class can help to meet the requirement, it comes at the cost of restricting its application to more complex multi-agent environments. In this paper, we propose QFree, a universal value function factorization method for MARL. We start by developing mathematical equivalent conditions of the IGM principle based on the advantage function, which ensures that the principle holds without any compromise, removing the conservatism of conventional methods. We then establish a more expressive mixing network architecture that can fulfill the equivalent factorization. In particular, the novel loss function is developed by considering the equivalent conditions as regularization term during policy evaluation in the MARL algorithm. Finally, the effectiveness of the proposed method is verified in a nonmonotonic matrix game scenario. Moreover, we show that QFree achieves the state-of-the-art performance in a general-purpose complex MARL benchmark environment, Starcraft Multi-Agent Challenge (SMAC).

{{</citation>}}


### (45/121) A Definition of Open-Ended Learning Problems for Goal-Conditioned Agents (Olivier Sigaud et al., 2023)

{{<citation>}}

Olivier Sigaud, Gianluca Baldassarre, Cedric Colas, Stephane Doncieux, Richard Duro, Nicolas Perrin-Gilbert, Vieri Giuliano Santucci. (2023)  
**A Definition of Open-Ended Learning Problems for Goal-Conditioned Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.00344v2)  

---


**ABSTRACT**  
A lot of recent machine learning research papers have "Open-ended learning" in their title. But very few of them attempt to define what they mean when using the term. Even worse, when looking more closely there seems to be no consensus on what distinguishes open-ended learning from related concepts such as continual learning, lifelong learning or autotelic learning. In this paper, we contribute to fixing this situation. After illustrating the genealogy of the concept and more recent perspectives about what it truly means, we outline that open-ended learning is generally conceived as a composite notion encompassing a set of diverse properties. In contrast with these previous approaches, we propose to isolate a key elementary property of open-ended processes, which is to always produce novel elements from time to time over an infinite horizon. From there, we build the notion of open-ended learning problems and focus in particular on the subset of open-ended goal-conditioned reinforcement learning problems, as this framework facilitates the definition of learning a growing repertoire of skills. Finally, we highlight the work that remains to be performed to fill the gap between our elementary definition and the more involved notions of open-ended learning that developmental AI researchers may have in mind.

{{</citation>}}


### (46/121) Can Large Language Models Capture Public Opinion about Global Warming? An Empirical Assessment of Algorithmic Fidelity and Bias (S. Lee et al., 2023)

{{<citation>}}

S. Lee, T. Q. Peng, M. H. Goldberg, S. A. Rosenthal, J. E. Kotcher, E. W. Maibach, A. Leiserowitz. (2023)  
**Can Large Language Models Capture Public Opinion about Global Warming? An Empirical Assessment of Algorithmic Fidelity and Bias**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: Bias, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.00217v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated their potential in social science research by emulating human perceptions and behaviors, a concept referred to as algorithmic fidelity. This study assesses the algorithmic fidelity and bias of LLMs by utilizing two nationally representative climate change surveys. The LLMs were conditioned on demographics and/or psychological covariates to simulate survey responses. The findings indicate that LLMs can effectively capture presidential voting behaviors but encounter challenges in accurately representing global warming perspectives when relevant covariates are not included. GPT-4 exhibits improved performance when conditioned on both demographics and covariates. However, disparities emerge in LLM estimations of the views of certain groups, with LLMs tending to underestimate worry about global warming among Black Americans. While highlighting the potential of LLMs to aid social science research, these results underscore the importance of meticulous conditioning, model selection, survey question format, and bias assessment when employing LLMs for survey simulation. Further investigation into prompt engineering and algorithm auditing is essential to harness the power of LLMs while addressing their inherent limitations.

{{</citation>}}


### (47/121) Modeling subjectivity (by Mimicking Annotator Annotation) in toxic comment identification across diverse communities (Senjuti Dutta et al., 2023)

{{<citation>}}

Senjuti Dutta, Sid Mittal, Sherol Chen, Deepak Ramachandran, Ravi Rajakumar, Ian Kivlichan, Sunny Mak, Alena Butryna, Praveen Paritosh. (2023)  
**Modeling subjectivity (by Mimicking Annotator Annotation) in toxic comment identification across diverse communities**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.00203v1)  

---


**ABSTRACT**  
The prevalence and impact of toxic discussions online have made content moderation crucial.Automated systems can play a vital role in identifying toxicity, and reducing the reliance on human moderation.Nevertheless, identifying toxic comments for diverse communities continues to present challenges that are addressed in this paper.The two-part goal of this study is to(1)identify intuitive variances from annotator disagreement using quantitative analysis and (2)model the subjectivity of these viewpoints.To achieve our goal, we published a new dataset\footnote{\url{https://github.com/XXX}} with expert annotators' annotations and used two other public datasets to identify the subjectivity of toxicity.Then leveraging the Large Language Model(LLM),we evaluate the model's ability to mimic diverse viewpoints on toxicity by varying size of the training data and utilizing same set of annotators as the test set used during model training and a separate set of annotators as the test set.We conclude that subjectivity is evident across all annotator groups, demonstrating the shortcomings of majority-rule voting. Moving forward, subjective annotations should serve as ground truth labels for training models for domains like toxicity in diverse communities.

{{</citation>}}


## cs.CR (6)



### (48/121) healthAIChain: Improving security and safety using Blockchain Technology applications in AI-based healthcare systems (Naresh Kshetri et al., 2023)

{{<citation>}}

Naresh Kshetri, James Hutson, Revathy G. (2023)  
**healthAIChain: Improving security and safety using Blockchain Technology applications in AI-based healthcare systems**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.00842v1)  

---


**ABSTRACT**  
Blockchain as a digital ledger for keeping records of digital transactions and other information, it is secure and decentralized technology. The globally growing number of digital population every day possesses a significant threat to online data including the medical and patients data. After bitcoin, blockchain technology has emerged into a general-purpose technology with applications in medical industries and healthcare. Blockchain can promote highly configurable openness while retaining the highest security standards for critical data of medical patients. Referred to as distributed record keeping for healthcare systems which makes digital assets unalterable and transparent via a cryptographic hash and decentralized network. The study delves into the security and safety improvement associated with implementing blockchain in AI-based healthcare systems. Blockchain-enabled AI tackles the existing issues related to security, performance efficiencies, and safety in healthcare systems. We have also examined the Artificial Intelligence in healthcare and medical industry, potential areas, open questions concerning the blockchain in healthcare systems. Finally, the article proposed an AI-based healthcare blockchain model (healthAIChain) to improve patients data and security.

{{</citation>}}


### (49/121) On the Integration of Self-Sovereign Identity with TLS 1.3 Handshake to Build Trust in IoT Systems (Leonardo Perugini et al., 2023)

{{<citation>}}

Leonardo Perugini, Andrea Vesco. (2023)  
**On the Integration of Self-Sovereign Identity with TLS 1.3 Handshake to Build Trust in IoT Systems**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.00386v1)  

---


**ABSTRACT**  
The centralized PKI is not a suitable solution to provide identities in large-scale IoT systems. The main problem is the high cost of managing X.509 certificates throughout their lifecycle, from installation to regular updates and revocation. The Self-Sovereign Identity (SSI) is a decentralised option that reduces the need for human intervention, and therefore has the potential to significantly reduce the complexity and cost associated to identity management in large-scale IoT systems. However, to leverage the full potential of SSI, the authentication of IoT nodes needs to be moved from the application to the Transport Layer Security (TLS) level. This paper contributes to the adoption of SSI in large-scale IoT systems by addressing, for the first time, the extension of the original TLS 1.3 handshake to support two new SSI authentication modes while maintaining the interoperability with nodes implementing the original handshake protocol. The open source implementation of the new TLS 1.3 handshake protocol in OpenSSL is used to experimentally prove the feasibility of the approach.

{{</citation>}}


### (50/121) Architecture of Data Anomaly Detection-Enhanced Decentralized Expert System for Early-Stage Alzheimer's Disease Prediction (Stefan Kambiz Behfar et al., 2023)

{{<citation>}}

Stefan Kambiz Behfar, Qumars Behfar, Marzie Hosseinpour. (2023)  
**Architecture of Data Anomaly Detection-Enhanced Decentralized Expert System for Early-Stage Alzheimer's Disease Prediction**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI, Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.00373v1)  

---


**ABSTRACT**  
Alzheimer's Disease is a global health challenge that requires early and accurate detection to improve patient outcomes. Magnetic Resonance Imaging (MRI) holds significant diagnostic potential, but its effective analysis remains a formidable task. This study introduces a groundbreaking decentralized expert system that cleverly combines blockchain technology with Artificial Intelligence (AI) to integrate robust anomaly detection for patient-submitted data.   Traditional diagnostic methods often lead to delayed and imprecise predictions, especially in the early stages of the disease. Centralized data repositories struggle to manage the immense volumes of MRI data, and persistent privacy concerns hinder collaborative efforts. Our innovative solution harnesses decentralization to protect data integrity and patient privacy, facilitated by blockchain technology. It not only emphasizes AI-driven MRI analysis but also incorporates a sophisticated data anomaly detection architecture. These mechanisms scrutinize patient-contributed data for various issues, including data quality problems and atypical findings within MRI images.   Conducting an exhaustive check of MRI image correctness and quality directly on the blockchain is impractical due to computational complexity and cost constraints. Typically, such checks are performed off-chain, and the blockchain securely records the results. This comprehensive approach empowers our decentralized app to provide more precise early-stage Alzheimer's Disease predictions. By merging the strengths of blockchain, AI, and anomaly detection, our system represents a pioneering step towards revolutionizing disease diagnostics.

{{</citation>}}


### (51/121) Stacking an autoencoder for feature selection of zero-day threats (Mahmut Tokmak et al., 2023)

{{<citation>}}

Mahmut Tokmak, Mike Nkongolo. (2023)  
**Stacking an autoencoder for feature selection of zero-day threats**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.00304v1)  

---


**ABSTRACT**  
Zero-day attack detection plays a critical role in mitigating risks, protecting assets, and staying ahead in the evolving threat landscape. This study explores the application of stacked autoencoder (SAE), a type of artificial neural network, for feature selection and zero-day threat classification using a Long Short-Term Memory (LSTM) scheme. The process involves preprocessing the UGRansome dataset and training an unsupervised SAE for feature extraction. Finetuning with supervised learning is then performed to enhance the discriminative capabilities of this model. The learned weights and activations of the autoencoder are analyzed to identify the most important features for discriminating between zero-day threats and normal system behavior. These selected features form a reduced feature set that enables accurate classification. The results indicate that the SAE-LSTM performs well across all three attack categories by showcasing high precision, recall, and F1 score values, emphasizing the model's strong predictive capabilities in identifying various types of zero-day attacks. Additionally, the balanced average scores of the SAE-LSTM suggest that the model generalizes effectively and consistently across different attack categories.

{{</citation>}}


### (52/121) Intell-dragonfly: A Cybersecurity Attack Surface Generation Engine Based On Artificial Intelligence-generated Content Technology (Xingchen Wu et al., 2023)

{{<citation>}}

Xingchen Wu, Qin Qiu, Jiaqi Li, Yang Zhao. (2023)  
**Intell-dragonfly: A Cybersecurity Attack Surface Generation Engine Based On Artificial Intelligence-generated Content Technology**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.00240v1)  

---


**ABSTRACT**  
With the rapid development of the Internet, cyber security issues have become increasingly prominent. Traditional cyber security defense methods are limited in the face of ever-changing threats, so it is critical to seek innovative attack surface generation methods. This study proposes Intell-dragonfly, a cyber security attack surface generation engine based on artificial intelligence generation technology, to meet the challenges of cyber security. Based on ChatGPT technology, this paper designs an automated attack surface generation process, which can generate diversified and personalized attack scenarios, targets, elements and schemes. Through experiments in a real network environment, the effect of the engine is verified and compared with traditional methods, which improves the authenticity and applicability of the attack surface. The experimental results show that the ChatGPT-based method has significant advantages in the accuracy, diversity and operability of attack surface generation. Furthermore, we explore the strengths and limitations of the engine and discuss its potential applications in the field of cyber security. This research provides a novel approach to the field of cyber security that is expected to have a positive impact on defense and prevention of cyberthreats.

{{</citation>}}


### (53/121) Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems (Jung-Woo Chang et al., 2023)

{{<citation>}}

Jung-Woo Chang, Ke Sun, Nasimeh Heydaribeni, Seira Hidano, Xinyu Zhang, Farinaz Koushanfar. (2023)  
**Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2311.00207v1)  

---


**ABSTRACT**  
Machine Learning (ML) has been instrumental in enabling joint transceiver optimization by merging all physical layer blocks of the end-to-end wireless communication systems. Although there have been a number of adversarial attacks on ML-based wireless systems, the existing methods do not provide a comprehensive view including multi-modality of the source data, common physical layer components, and wireless domain constraints. This paper proposes Magmaw, the first black-box attack methodology capable of generating universal adversarial perturbations for any multimodal signal transmitted over a wireless channel. We further introduce new objectives for adversarial attacks on ML-based downstream applications. The resilience of the attack to the existing widely used defense methods of adversarial training and perturbation signal subtraction is experimentally verified. For proof-of-concept evaluation, we build a real-time wireless attack platform using a software-defined radio system. Experimental results demonstrate that Magmaw causes significant performance degradation even in the presence of the defense mechanisms. Surprisingly, Magmaw is also effective against encrypted communication channels and conventional communications.

{{</citation>}}


## cs.CL (27)



### (54/121) Calibrated Seq2seq Models for Efficient and Generalizable Ultra-fine Entity Typing (Yanlin Feng et al., 2023)

{{<citation>}}

Yanlin Feng, Adithya Pratapa, David R Mortensen. (2023)  
**Calibrated Seq2seq Models for Efficient and Generalizable Ultra-fine Entity Typing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.00835v1)  

---


**ABSTRACT**  
Ultra-fine entity typing plays a crucial role in information extraction by predicting fine-grained semantic types for entity mentions in text. However, this task poses significant challenges due to the massive number of entity types in the output space. The current state-of-the-art approaches, based on standard multi-label classifiers or cross-encoder models, suffer from poor generalization performance or inefficient inference. In this paper, we present CASENT, a seq2seq model designed for ultra-fine entity typing that predicts ultra-fine types with calibrated confidence scores. Our model takes an entity mention as input and employs constrained beam search to generate multiple types autoregressively. The raw sequence probabilities associated with the predicted types are then transformed into confidence scores using a novel calibration method. We conduct extensive experiments on the UFET dataset which contains over 10k types. Our method outperforms the previous state-of-the-art in terms of F1 score and calibration error, while achieving an inference speedup of over 50 times. Additionally, we demonstrate the generalization capabilities of our model by evaluating it in zero-shot and few-shot settings on five specialized domain entity typing datasets that are unseen during training. Remarkably, our model outperforms large language models with 10 times more parameters in the zero-shot setting, and when fine-tuned on 50 examples, it significantly outperforms ChatGPT on all datasets. Our code, models and demo are available at https://github.com/yanlinf/CASENT.

{{</citation>}}


### (55/121) Little Giants: Exploring the Potential of Small LLMs as Evaluation Metrics in Summarization in the Eval4NLP 2023 Shared Task (Neema Kotonya et al., 2023)

{{<citation>}}

Neema Kotonya, Saran Krishnasamy, Joel Tetreault, Alejandro Jaimes. (2023)  
**Little Giants: Exploring the Potential of Small LLMs as Evaluation Metrics in Summarization in the Eval4NLP 2023 Shared Task**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP, Summarization  
[Paper Link](http://arxiv.org/abs/2311.00686v1)  

---


**ABSTRACT**  
This paper describes and analyzes our participation in the 2023 Eval4NLP shared task, which focuses on assessing the effectiveness of prompt-based techniques to empower Large Language Models to handle the task of quality estimation, particularly in the context of evaluating machine translations and summaries. We conducted systematic experiments with various prompting techniques, including standard prompting, prompts informed by annotator instructions, and innovative chain-of-thought prompting. In addition, we integrated these approaches with zero-shot and one-shot learning methods to maximize the efficacy of our evaluation procedures. Our work reveals that combining these approaches using a "small", open source model (orca_mini_v3_7B) yields competitive results.

{{</citation>}}


### (56/121) Attention Alignment and Flexible Positional Embeddings Improve Transformer Length Extrapolation (Ta-Chung Chi et al., 2023)

{{<citation>}}

Ta-Chung Chi, Ting-Han Fan, Alexander I. Rudnicky. (2023)  
**Attention Alignment and Flexible Positional Embeddings Improve Transformer Length Extrapolation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Attention, Embedding, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2311.00684v1)  

---


**ABSTRACT**  
An ideal length-extrapolatable Transformer language model can handle sequences longer than the training length without any long sequence fine-tuning. Such long-context utilization capability highly relies on a flexible positional embedding design. Upon investigating the flexibility of existing large pre-trained Transformer language models, we find that the T5 family deserves a closer look, as its positional embeddings capture rich and flexible attention patterns. However, T5 suffers from the dispersed attention issue: the longer the input sequence, the flatter the attention distribution. To alleviate the issue, we propose two attention alignment strategies via temperature scaling. Our findings improve the long-context utilization capability of T5 on language modeling, retrieval, and multi-document question answering without any fine-tuning, suggesting that a flexible positional embedding design and attention alignment go a long way toward Transformer length extrapolation.\footnote{\url{https://github.com/chijames/Attention-Alignment-Transformer-Length-Extrapolation}}

{{</citation>}}


### (57/121) Are Large Language Models Reliable Judges? A Study on the Factuality Evaluation Capabilities of LLMs (Xue-Yong Fu et al., 2023)

{{<citation>}}

Xue-Yong Fu, Md Tahmid Rahman Laskar, Cheng Chen, Shashi Bhushan TN. (2023)  
**Are Large Language Models Reliable Judges? A Study on the Factuality Evaluation Capabilities of LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2311.00681v1)  

---


**ABSTRACT**  
In recent years, Large Language Models (LLMs) have gained immense attention due to their notable emergent capabilities, surpassing those seen in earlier language models. A particularly intriguing application of LLMs is their role as evaluators for texts produced by various generative models.   In this study, we delve into the potential of LLMs as reliable assessors of factual consistency in summaries generated by text-generation models. Initially, we introduce an innovative approach for factuality assessment using LLMs. This entails employing a singular LLM for the entirety of the question-answering-based factuality scoring process. Following this, we examine the efficacy of various LLMs in direct factuality scoring, benchmarking them against traditional measures and human annotations.   Contrary to initial expectations, our results indicate a lack of significant correlations between factuality metrics and human evaluations, specifically for GPT-4 and PaLM-2. Notable correlations were only observed with GPT-3.5 across two factuality subcategories. These consistent findings across various factual error categories suggest a fundamental limitation in the current LLMs' capability to accurately gauge factuality.   This version presents the information more concisely while maintaining the main points and findings of the original text.

{{</citation>}}


### (58/121) Explicit Morphological Knowledge Improves Pre-training of Language Models for Hebrew (Eylon Gueta et al., 2023)

{{<citation>}}

Eylon Gueta, Omer Goldman, Reut Tsarfaty. (2023)  
**Explicit Morphological Knowledge Improves Pre-training of Language Models for Hebrew**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.00658v1)  

---


**ABSTRACT**  
Pre-trained language models (PLMs) have shown remarkable successes in acquiring a wide range of linguistic knowledge, relying solely on self-supervised training on text streams. Nevertheless, the effectiveness of this language-agnostic approach has been frequently questioned for its sub-optimal performance when applied to morphologically-rich languages (MRLs). We investigate the hypothesis that incorporating explicit morphological knowledge in the pre-training phase can improve the performance of PLMs for MRLs. We propose various morphologically driven tokenization methods enabling the model to leverage morphological cues beyond raw text. We pre-train multiple language models utilizing the different methods and evaluate them on Hebrew, a language with complex and highly ambiguous morphology. Our experiments show that morphologically driven tokenization demonstrates improved results compared to a standard language-agnostic tokenization, on a benchmark of both semantic and morphologic tasks. These findings suggest that incorporating morphological knowledge holds the potential for further improving PLMs for morphologically rich languages.

{{</citation>}}


### (59/121) Boosting Summarization with Normalizing Flows and Aggressive Training (Yu Yang et al., 2023)

{{<citation>}}

Yu Yang, Xiaotong Shen. (2023)  
**Boosting Summarization with Normalizing Flows and Aggressive Training**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Summarization, Transformer  
[Paper Link](http://arxiv.org/abs/2311.00588v1)  

---


**ABSTRACT**  
This paper presents FlowSUM, a normalizing flows-based variational encoder-decoder framework for Transformer-based summarization. Our approach tackles two primary challenges in variational summarization: insufficient semantic information in latent representations and posterior collapse during training. To address these challenges, we employ normalizing flows to enable flexible latent posterior modeling, and we propose a controlled alternate aggressive training (CAAT) strategy with an improved gate mechanism. Experimental results show that FlowSUM significantly enhances the quality of generated summaries and unleashes the potential for knowledge distillation with minimal impact on inference time. Furthermore, we investigate the issue of posterior collapse in normalizing flows and analyze how the summary quality is affected by the training strategy, gate initialization, and the type and number of normalizing flows used, offering valuable insights for future research.

{{</citation>}}


### (60/121) Crosslingual Retrieval Augmented In-context Learning for Bangla (Xiaoqian Li et al., 2023)

{{<citation>}}

Xiaoqian Li, Ercong Nie, Sheng Liang. (2023)  
**Crosslingual Retrieval Augmented In-context Learning for Bangla**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLOOM, Language Model, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.00587v1)  

---


**ABSTRACT**  
The promise of Large Language Models (LLMs) in Natural Language Processing has often been overshadowed by their limited performance in low-resource languages such as Bangla. To address this, our paper presents a pioneering approach that utilizes cross-lingual retrieval augmented in-context learning. By strategically sourcing semantically similar prompts from high-resource language, we enable multilingual pretrained language models (MPLMs), especially the generative model BLOOMZ, to successfully boost performance on Bangla tasks. Our extensive evaluation highlights that the cross-lingual retrieval augmented prompts bring steady improvements to MPLMs over the zero-shot performance.

{{</citation>}}


### (61/121) Can Large Language Models Design Accurate Label Functions? (Naiqing Guan et al., 2023)

{{<citation>}}

Naiqing Guan, Kaiwen Chen, Nick Koudas. (2023)  
**Can Large Language Models Design Accurate Label Functions?**  

---
Primary Category: cs.CL  
Categories: H-2-8; I-5-4, cs-CL, cs-DB, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.00739v1)  

---


**ABSTRACT**  
Programmatic weak supervision methodologies facilitate the expedited labeling of extensive datasets through the use of label functions (LFs) that encapsulate heuristic data sources. Nonetheless, the creation of precise LFs necessitates domain expertise and substantial endeavors. Recent advances in pre-trained language models (PLMs) have exhibited substantial potential across diverse tasks. However, the capacity of PLMs to autonomously formulate accurate LFs remains an underexplored domain. In this research, we address this gap by introducing DataSculpt, an interactive framework that harnesses PLMs for the automated generation of LFs. Within DataSculpt, we incorporate an array of prompting techniques, instance selection strategies, and LF filtration methods to explore the expansive design landscape. Ultimately, we conduct a thorough assessment of DataSculpt's performance on 12 real-world datasets, encompassing a range of tasks. This evaluation unveils both the strengths and limitations of contemporary PLMs in LF design.

{{</citation>}}


### (62/121) Text Rendering Strategies for Pixel Language Models (Jonas F. Lotz et al., 2023)

{{<citation>}}

Jonas F. Lotz, Elizabeth Salesky, Phillip Rust, Desmond Elliott. (2023)  
**Text Rendering Strategies for Pixel Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.00522v1)  

---


**ABSTRACT**  
Pixel-based language models process text rendered as images, which allows them to handle any script, making them a promising approach to open vocabulary language modelling. However, recent approaches use text renderers that produce a large set of almost-equivalent input patches, which may prove sub-optimal for downstream tasks, due to redundancy in the input representations. In this paper, we investigate four approaches to rendering text in the PIXEL model (Rust et al., 2023), and find that simple character bigram rendering brings improved performance on sentence-level tasks without compromising performance on token-level or multilingual tasks. This new rendering strategy also makes it possible to train a more compact model with only 22M parameters that performs on par with the original 86M parameter model. Our analyses show that character bigram rendering leads to a consistently better model but with an anisotropic patch embedding space, driven by a patch frequency bias, highlighting the connections between image patch- and tokenization-based language models.

{{</citation>}}


### (63/121) Robustness Tests for Automatic Machine Translation Metrics with Adversarial Attacks (Yichen Huang et al., 2023)

{{<citation>}}

Yichen Huang, Timothy Baldwin. (2023)  
**Robustness Tests for Automatic Machine Translation Metrics with Adversarial Attacks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Adversarial Attack, BERT, BLEU, Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.00508v1)  

---


**ABSTRACT**  
We investigate MT evaluation metric performance on adversarially-synthesized texts, to shed light on metric robustness. We experiment with word- and character-level attacks on three popular machine translation metrics: BERTScore, BLEURT, and COMET. Our human experiments validate that automatic metrics tend to overpenalize adversarially-degraded translations. We also identify inconsistencies in BERTScore ratings, where it judges the original sentence and the adversarially-degraded one as similar, while judging the degraded translation as notably worse than the original with respect to the reference. We identify patterns of brittleness that motivate more robust metric development.

{{</citation>}}


### (64/121) Style Locality for Controllable Generation with kNN Language Models (Gilles Nawezi et al., 2023)

{{<citation>}}

Gilles Nawezi, Lucie Flek, Charles Welch. (2023)  
**Style Locality for Controllable Generation with kNN Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.00475v1)  

---


**ABSTRACT**  
Recent language models have been improved by the addition of external memory. Nearest neighbor language models retrieve similar contexts to assist in word prediction. The addition of locality levels allows a model to learn how to weight neighbors based on their relative location to the current text in source documents, and have been shown to further improve model performance. Nearest neighbor models have been explored for controllable generation but have not examined the use of locality levels. We present a novel approach for this purpose and evaluate it using automatic and human evaluation on politeness, formality, supportiveness, and toxicity textual data. We find that our model is successfully able to control style and provides a better fluency-style trade-off than previous work.

{{</citation>}}


### (65/121) A Systematic Comparison of Syllogistic Reasoning in Humans and Language Models (Tiwalayo Eisape et al., 2023)

{{<citation>}}

Tiwalayo Eisape, MH Tessler, Ishita Dasgupta, Fei Sha, Sjoerd van Steenkiste, Tal Linzen. (2023)  
**A Systematic Comparison of Syllogistic Reasoning in Humans and Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.00445v1)  

---


**ABSTRACT**  
A central component of rational behavior is logical inference: the process of determining which conclusions follow from a set of premises. Psychologists have documented several ways in which humans' inferences deviate from the rules of logic. Do language models, which are trained on text generated by humans, replicate these biases, or are they able to overcome them? Focusing on the case of syllogisms -- inferences from two simple premises, which have been studied extensively in psychology -- we show that larger models are more logical than smaller ones, and also more logical than humans. At the same time, even the largest models make systematic errors, some of which mirror human reasoning biases such as ordering effects and logical fallacies. Overall, we find that language models mimic the human biases included in their training data, but are able to overcome them in some cases.

{{</citation>}}


### (66/121) Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling (Sanchit Gandhi et al., 2023)

{{<citation>}}

Sanchit Gandhi, Patrick von Platen, Alexander M. Rush. (2023)  
**Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2311.00430v1)  

---


**ABSTRACT**  
As the size of pre-trained speech recognition models increases, running these large models in low-latency or resource-constrained environments becomes challenging. In this work, we leverage pseudo-labelling to assemble a large-scale open-source dataset which we use to distill the Whisper model into a smaller variant, called Distil-Whisper. Using a simple word error rate (WER) heuristic, we select only the highest quality pseudo-labels for training. The distilled model is 5.8 times faster with 51% fewer parameters, while performing to within 1% WER on out-of-distribution test data in a zero-shot transfer setting. Distil-Whisper maintains the robustness of the Whisper model to difficult acoustic conditions, while being less prone to hallucination errors on long-form audio. Distil-Whisper is designed to be paired with Whisper for speculative decoding, yielding a 2 times speed-up while mathematically ensuring the same outputs as the original model. To facilitate further research in this domain, we make our training code, inference code and models publicly accessible.

{{</citation>}}


### (67/121) AdaSent: Efficient Domain-Adapted Sentence Embeddings for Few-Shot Classification (Yongxin Huang et al., 2023)

{{<citation>}}

Yongxin Huang, Kexin Wang, Sourav Dutta, Raj Nath Patel, Goran Glavaš, Iryna Gurevych. (2023)  
**AdaSent: Efficient Domain-Adapted Sentence Embeddings for Few-Shot Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, Few-Shot, Language Model, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2311.00408v1)  

---


**ABSTRACT**  
Recent work has found that few-shot sentence classification based on pre-trained Sentence Encoders (SEs) is efficient, robust, and effective. In this work, we investigate strategies for domain-specialization in the context of few-shot sentence classification with SEs. We first establish that unsupervised Domain-Adaptive Pre-Training (DAPT) of a base Pre-trained Language Model (PLM) (i.e., not an SE) substantially improves the accuracy of few-shot sentence classification by up to 8.4 points. However, applying DAPT on SEs, on the one hand, disrupts the effects of their (general-domain) Sentence Embedding Pre-Training (SEPT). On the other hand, applying general-domain SEPT on top of a domain-adapted base PLM (i.e., after DAPT) is effective but inefficient, since the computationally expensive SEPT needs to be executed on top of a DAPT-ed PLM of each domain. As a solution, we propose AdaSent, which decouples SEPT from DAPT by training a SEPT adapter on the base PLM. The adapter can be inserted into DAPT-ed PLMs from any domain. We demonstrate AdaSent's effectiveness in extensive experiments on 17 different few-shot sentence classification datasets. AdaSent matches or surpasses the performance of full SEPT on DAPT-ed PLM, while substantially reducing the training costs. The code for AdaSent is available.

{{</citation>}}


### (68/121) HARE: Explainable Hate Speech Detection with Step-by-Step Reasoning (Yongjin Yang et al., 2023)

{{<citation>}}

Yongjin Yang, Joonkee Kim, Yujin Kim, Namgyu Ho, James Thorne, Se-young Yun. (2023)  
**HARE: Explainable Hate Speech Detection with Step-by-Step Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Hate Speech Detection, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.00321v1)  

---


**ABSTRACT**  
With the proliferation of social media, accurate detection of hate speech has become critical to ensure safety online. To combat nuanced forms of hate speech, it is important to identify and thoroughly explain hate speech to help users understand its harmful effects. Recent benchmarks have attempted to tackle this issue by training generative models on free-text annotations of implications in hateful text. However, we find significant reasoning gaps in the existing annotations schemes, which may hinder the supervision of detection models. In this paper, we introduce a hate speech detection framework, HARE, which harnesses the reasoning capabilities of large language models (LLMs) to fill these gaps in explanations of hate speech, thus enabling effective supervision of detection models. Experiments on SBIC and Implicit Hate benchmarks show that our method, using model-generated data, consistently outperforms baselines, using existing free-text human annotations. Analysis demonstrates that our method enhances the explanation quality of trained models and improves generalization to unseen datasets. Our code is available at https://github.com/joonkeekim/hare-hate-speech.git.

{{</citation>}}


### (69/121) Data Augmentation for Code Translation with Comparable Corpora and Multiple References (Yiqing Xie et al., 2023)

{{<citation>}}

Yiqing Xie, Atharva Naik, Daniel Fried, Carolyn Rose. (2023)  
**Data Augmentation for Code Translation with Comparable Corpora and Multiple References**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SE, cs.CL  
Keywords: Augmentation, T5  
[Paper Link](http://arxiv.org/abs/2311.00317v1)  

---


**ABSTRACT**  
One major challenge of translating code between programming languages is that parallel training data is often limited. To overcome this challenge, we present two data augmentation techniques, one that builds comparable corpora (i.e., code pairs with similar functionality), and another that augments existing parallel data with multiple reference translations. Specifically, we build and analyze multiple types of comparable corpora, including programs generated from natural language documentation using a code generation model. Furthermore, to reduce overfitting to a single reference translation, we automatically generate additional translation references for available parallel data and filter the translations by unit tests, which increases variation in target translations. Experiments show that our data augmentation techniques significantly improve CodeT5 for translation between Java, Python, and C++ by an average of 7.5% Computational Accuracy (CA@1), which verifies the correctness of translations by execution. The code is available at https://github.com/Veronicium/CMTrans.

{{</citation>}}


### (70/121) Unsupervised Lexical Simplification with Context Augmentation (Takashi Wada et al., 2023)

{{<citation>}}

Takashi Wada, Timothy Baldwin, Jey Han Lau. (2023)  
**Unsupervised Lexical Simplification with Context Augmentation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Augmentation, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2311.00310v1)  

---


**ABSTRACT**  
We propose a new unsupervised lexical simplification method that uses only monolingual data and pre-trained language models. Given a target word and its context, our method generates substitutes based on the target context and also additional contexts sampled from monolingual data. We conduct experiments in English, Portuguese, and Spanish on the TSAR-2022 shared task, and show that our model substantially outperforms other unsupervised systems across all languages. We also establish a new state-of-the-art by ensembling our model with GPT-3.5. Lastly, we evaluate our model on the SWORDS lexical substitution data set, achieving a state-of-the-art result.

{{</citation>}}


### (71/121) Probing Explicit and Implicit Gender Bias through LLM Conditional Text Generation (Xiangjue Dong et al., 2023)

{{<citation>}}

Xiangjue Dong, Yibo Wang, Philip S. Yu, James Caverlee. (2023)  
**Probing Explicit and Implicit Gender Bias through LLM Conditional Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model, Text Generation  
[Paper Link](http://arxiv.org/abs/2311.00306v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) can generate biased and toxic responses. Yet most prior work on LLM gender bias evaluation requires predefined gender-related phrases or gender stereotypes, which are challenging to be comprehensively collected and are limited to explicit bias evaluation. In addition, we believe that instances devoid of gender-related language or explicit stereotypes in inputs can still induce gender bias in LLMs. Thus, in this work, we propose a conditional text generation mechanism without the need for predefined gender phrases and stereotypes. This approach employs three types of inputs generated through three distinct strategies to probe LLMs, aiming to show evidence of explicit and implicit gender biases in LLMs. We also utilize explicit and implicit evaluation metrics to evaluate gender bias in LLMs under different strategies. Our experiments demonstrate that an increased model size does not consistently lead to enhanced fairness and all tested LLMs exhibit explicit and/or implicit gender bias, even when explicit gender stereotypes are absent in the inputs.

{{</citation>}}


### (72/121) Entity Alignment Method of Science and Technology Patent based on Graph Convolution Network and Information Fusion (Runze Fang et al., 2023)

{{<citation>}}

Runze Fang, Yawen Li, Yingxia Shao, Zeli Guan, Zhe Xue. (2023)  
**Entity Alignment Method of Science and Technology Patent based on Graph Convolution Network and Information Fusion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Entity Alignment  
[Paper Link](http://arxiv.org/abs/2311.00300v1)  

---


**ABSTRACT**  
The entity alignment of science and technology patents aims to link the equivalent entities in the knowledge graph of different science and technology patent data sources. Most entity alignment methods only use graph neural network to obtain the embedding of graph structure or use attribute text description to obtain semantic representation, ignoring the process of multi-information fusion in science and technology patents. In order to make use of the graphic structure and auxiliary information such as the name, description and attribute of the patent entity, this paper proposes an entity alignment method based on the graph convolution network for science and technology patent information fusion. Through the graph convolution network and BERT model, the structure information and entity attribute information of the science and technology patent knowledge graph are embedded and represented to achieve multi-information fusion, thus improving the performance of entity alignment. Experiments on three benchmark data sets show that the proposed method Hit@K The evaluation indicators are better than the existing methods.

{{</citation>}}


### (73/121) Semantic Representation Learning of Scientific Literature based on Adaptive Feature and Graph Neural Network (Hongrui Gao et al., 2023)

{{<citation>}}

Hongrui Gao, Yawen Li, Meiyu Liang, Zeli Guan, Zhe Xue. (2023)  
**Semantic Representation Learning of Scientific Literature based on Adaptive Feature and Graph Neural Network**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Graph Neural Network, Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.00296v1)  

---


**ABSTRACT**  
Because most of the scientific literature data is unmarked, it makes semantic representation learning based on unsupervised graph become crucial. At the same time, in order to enrich the features of scientific literature, a learning method of semantic representation of scientific literature based on adaptive features and graph neural network is proposed. By introducing the adaptive feature method, the features of scientific literature are considered globally and locally. The graph attention mechanism is used to sum the features of scientific literature with citation relationship, and give each scientific literature different feature weights, so as to better express the correlation between the features of different scientific literature. In addition, an unsupervised graph neural network semantic representation learning method is proposed. By comparing the mutual information between the positive and negative local semantic representation of scientific literature and the global graph semantic representation in the potential space, the graph neural network can capture the local and global information, thus improving the learning ability of the semantic representation of scientific literature. The experimental results show that the proposed learning method of semantic representation of scientific literature based on adaptive feature and graph neural network is competitive on the basis of scientific literature classification, and has achieved good results.

{{</citation>}}


### (74/121) IBADR: an Iterative Bias-Aware Dataset Refinement Framework for Debiasing NLU models (Xiaoyue Wang et al., 2023)

{{<citation>}}

Xiaoyue Wang, Xin Liu, Lijie Wang, Yaoxiang Wang, Jinsong Su, Hua Wu. (2023)  
**IBADR: an Iterative Bias-Aware Dataset Refinement Framework for Debiasing NLU models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, NLU  
[Paper Link](http://arxiv.org/abs/2311.00292v1)  

---


**ABSTRACT**  
As commonly-used methods for debiasing natural language understanding (NLU) models, dataset refinement approaches heavily rely on manual data analysis, and thus maybe unable to cover all the potential biased features. In this paper, we propose IBADR, an Iterative Bias-Aware Dataset Refinement framework, which debiases NLU models without predefining biased features. We maintain an iteratively expanded sample pool. Specifically, at each iteration, we first train a shallow model to quantify the bias degree of samples in the pool. Then, we pair each sample with a bias indicator representing its bias degree, and use these extended samples to train a sample generator. In this way, this generator can effectively learn the correspondence relationship between bias indicators and samples. Furthermore, we employ the generator to produce pseudo samples with fewer biased features by feeding specific bias indicators. Finally, we incorporate the generated pseudo samples into the pool. Experimental results and in-depth analyses on two NLU tasks show that IBADR not only significantly outperforms existing dataset refinement approaches, achieving SOTA, but also is compatible with model-centric methods.

{{</citation>}}


### (75/121) Knowledge-Infused Prompting: Assessing and Advancing Clinical Text Data Generation with Large Language Models (Ran Xu et al., 2023)

{{<citation>}}

Ran Xu, Hejie Cui, Yue Yu, Xuan Kan, Wenqi Shi, Yuchen Zhuang, Wei Jin, Joyce Ho, Carl Yang. (2023)  
**Knowledge-Infused Prompting: Assessing and Advancing Clinical Text Data Generation with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL, q-bio-QM  
Keywords: Clinical, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.00287v1)  

---


**ABSTRACT**  
Clinical natural language processing requires methods that can address domain-specific challenges, such as complex medical terminology and clinical contexts. Recently, large language models (LLMs) have shown promise in this domain. Yet, their direct deployment can lead to privacy issues and are constrained by resources. To address this challenge, we delve into synthetic clinical text generation using LLMs for clinical NLP tasks. We propose an innovative, resource-efficient approach, ClinGen, which infuses knowledge into the process. Our model involves clinical knowledge extraction and context-informed LLM prompting. Both clinical topics and writing styles are drawn from external domain-specific knowledge graphs and LLMs to guide data generation. Our extensive empirical study across 7 clinical NLP tasks and 16 datasets reveals that ClinGen consistently enhances performance across various tasks, effectively aligning the distribution of real datasets and significantly enriching the diversity of generated training instances. We will publish our code and all the generated data in \url{https://github.com/ritaranx/ClinGen}.

{{</citation>}}


### (76/121) Syntactic Inductive Bias in Transformer Language Models: Especially Helpful for Low-Resource Languages? (Luke Gessler et al., 2023)

{{<citation>}}

Luke Gessler, Nathan Schneider. (2023)  
**Syntactic Inductive Bias in Transformer Language Models: Especially Helpful for Low-Resource Languages?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Bias, Language Model, Low-Resource, Transformer  
[Paper Link](http://arxiv.org/abs/2311.00268v1)  

---


**ABSTRACT**  
A line of work on Transformer-based language models such as BERT has attempted to use syntactic inductive bias to enhance the pretraining process, on the theory that building syntactic structure into the training process should reduce the amount of data needed for training. But such methods are often tested for high-resource languages such as English. In this work, we investigate whether these methods can compensate for data sparseness in low-resource languages, hypothesizing that they ought to be more effective for low-resource languages. We experiment with five low-resource languages: Uyghur, Wolof, Maltese, Coptic, and Ancient Greek. We find that these syntactic inductive bias methods produce uneven results in low-resource settings, and provide surprisingly little benefit in most cases.

{{</citation>}}


### (77/121) Plug-and-Play Policy Planner for Large Language Model Powered Dialogue Agents (Yang Deng et al., 2023)

{{<citation>}}

Yang Deng, Wenxuan Zhang, Wai Lam, See-Kiong Ng, Tat-Seng Chua. (2023)  
**Plug-and-Play Policy Planner for Large Language Model Powered Dialogue Agents**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2311.00262v1)  

---


**ABSTRACT**  
Proactive dialogues serve as a practical yet challenging dialogue problem in the era of large language models (LLMs), where the dialogue policy planning is the key to improving the proactivity of LLMs. Most existing studies enable the dialogue policy planning of LLMs using various prompting schemes or iteratively enhance this capability in handling the given case with verbal AI feedback. However, these approaches are either bounded by the policy planning capability of the frozen LLMs or hard to be transferred to new cases. In this work, we introduce a new dialogue policy planning paradigm to strategize LLMs for proactive dialogue problems with a tunable language model plug-in as a plug-and-play dialogue policy planner, named PPDPP. Specifically, we develop a novel training framework to facilitate supervised fine-tuning over available human-annotated data as well as reinforcement learning from goal-oriented AI feedback with dynamic interaction data collected by the LLM-based self-play simulation. In this manner, the LLM-powered dialogue agent can not only be generalized to different cases after the training, but also be applicable to different applications by just substituting the learned plug-in. In addition, we propose to evaluate the policy planning capability of dialogue systems under the interactive setting. Experimental results demonstrate that PPDPP consistently and substantially outperforms existing approaches on three different proactive dialogue applications, including negotiation, emotional support, and tutoring dialogues.

{{</citation>}}


### (78/121) Noisy Exemplars Make Large Language Models More Robust: A Domain-Agnostic Behavioral Analysis (Hongyi Zheng et al., 2023)

{{<citation>}}

Hongyi Zheng, Abulhair Saparov. (2023)  
**Noisy Exemplars Make Large Language Models More Robust: A Domain-Agnostic Behavioral Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.00258v1)  

---


**ABSTRACT**  
Recent advances in prompt engineering enable large language models (LLMs) to solve multi-hop logical reasoning problems with impressive accuracy. However, there is little existing work investigating the robustness of LLMs with few-shot prompting techniques. Therefore, we introduce a systematic approach to test the robustness of LLMs in multi-hop reasoning tasks via domain-agnostic perturbations. We include perturbations at multiple levels of abstractions (e.g. lexical perturbations such as typos, and semantic perturbations such as the inclusion of intermediate reasoning steps in the questions) to conduct behavioral analysis on the LLMs. Throughout our experiments, we find that models are more sensitive to certain perturbations such as replacing words with their synonyms. We also demonstrate that increasing the proportion of perturbed exemplars in the prompts improves the robustness of few-shot prompting methods.

{{</citation>}}


### (79/121) Is GPT Powerful Enough to Analyze the Emotions of Memes? (Jingjing Wang et al., 2023)

{{<citation>}}

Jingjing Wang, Joshua Luo, Grace Yang, Allen Hong, Feng Luo. (2023)  
**Is GPT Powerful Enough to Analyze the Emotions of Memes?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-MM, cs.CL  
Keywords: AI, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2311.00223v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), representing a significant achievement in artificial intelligence (AI) research, have demonstrated their ability in a multitude of tasks. This project aims to explore the capabilities of GPT-3.5, a leading example of LLMs, in processing the sentiment analysis of Internet memes. Memes, which include both verbal and visual aspects, act as a powerful yet complex tool for expressing ideas and sentiments, demanding an understanding of societal norms and cultural contexts. Notably, the detection and moderation of hateful memes pose a significant challenge due to their implicit offensive nature. This project investigates GPT's proficiency in such subjective tasks, revealing its strengths and potential limitations. The tasks include the classification of meme sentiment, determination of humor type, and detection of implicit hate in memes. The performance evaluation, using datasets from SemEval-2020 Task 8 and Facebook hateful memes, offers a comparative understanding of GPT responses against human annotations. Despite GPT's remarkable progress, our findings underscore the challenges faced by these models in handling subjective tasks, which are rooted in their inherent limitations including contextual understanding, interpretation of implicit meanings, and data biases. This research contributes to the broader discourse on the applicability of AI in handling complex, context-dependent tasks, and offers valuable insights for future advancements.

{{</citation>}}


### (80/121) Continuous Training and Fine-tuning for Domain-Specific Language Models in Medical Question Answering (Zhen Guo et al., 2023)

{{<citation>}}

Zhen Guo, Yining Hua. (2023)  
**Continuous Training and Fine-tuning for Domain-Specific Language Models in Medical Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.00204v1)  

---


**ABSTRACT**  
Large language models exhibit promising general capabilities but often lack specialized knowledge for domain-specific tasks. Developing domain experts from a base model enables a range of applications without prohibitive training costs. This work demonstrates a method using continuous training and instruction fine-tuning to rapidly adapt Llama 2 base models to the Chinese medical domain. We first conduct continuous training on 1B tokens from Chinese medical references to teach relevant vocabulary and knowledge. The models are then fine-tuned on 54K examples sourced from the Chinese National Medical Licensing Examination. Experiments on Chinese medical data confirm the effectiveness of this approach, producing a model comparable to GPT-3.5-turbo while using way less computational resource. The resulting domain-specific model could be useful for various Chinese medical applications. More broadly, this provides a template for domain-specific training of large language models in areas where pre-trained models lack the required expertise, such as law, science, and engineering.

{{</citation>}}


## cs.CY (1)



### (81/121) A Call to Arms: AI Should be Critical for Social Media Analysis of Conflict Zones (Afia Abedin et al., 2023)

{{<citation>}}

Afia Abedin, Abdul Bais, Cody Buntain, Laura Courchesne, Brian McQuinn, Matthew E. Taylor, Muhib Ullah. (2023)  
**A Call to Arms: AI Should be Critical for Social Media Analysis of Conflict Zones**  

---
Primary Category: cs.CY  
Categories: cs-CV, cs-CY, cs-HC, cs.CY  
Keywords: AI, Social Media  
[Paper Link](http://arxiv.org/abs/2311.00810v1)  

---


**ABSTRACT**  
The massive proliferation of social media data represents a transformative moment in conflict studies. This data can provide unique insights into the spread and use of weaponry, but the scale and types of data are problematic for traditional open-source intelligence. This paper presents preliminary, transdisciplinary work using computer vision to identify specific weapon systems and the insignias of the armed groups using them. There is potential to not only track how weapons are distributed through networks of armed units but also to track which types of weapons are being used by the different types of state and non-state military actors in Ukraine. Such a system could ultimately be used to understand conflicts in real-time, including where humanitarian and medical aid is most needed. We believe that using AI to help automate such processes should be a high-priority goal for our community, with near-term real-world payoffs.

{{</citation>}}


## cs.CV (25)



### (82/121) VQA-GEN: A Visual Question Answering Benchmark for Domain Generalization (Suraj Jyothi Unni et al., 2023)

{{<citation>}}

Suraj Jyothi Unni, Raha Moraffah, Huan Liu. (2023)  
**VQA-GEN: A Visual Question Answering Benchmark for Domain Generalization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.00807v1)  

---


**ABSTRACT**  
Visual question answering (VQA) models are designed to demonstrate visual-textual reasoning capabilities. However, their real-world applicability is hindered by a lack of comprehensive benchmark datasets. Existing domain generalization datasets for VQA exhibit a unilateral focus on textual shifts while VQA being a multi-modal task contains shifts across both visual and textual domains. We propose VQA-GEN, the first ever multi-modal benchmark dataset for distribution shift generated through a shift induced pipeline. Experiments demonstrate VQA-GEN dataset exposes the vulnerability of existing methods to joint multi-modal distribution shifts. validating that comprehensive multi-modal shifts are critical for robust VQA generalization. Models trained on VQA-GEN exhibit improved cross-domain and in-domain performance, confirming the value of VQA-GEN. Further, we analyze the importance of each shift technique of our pipeline contributing to the generalization of the model.

{{</citation>}}


### (83/121) TPSeNCE: Towards Artifact-Free Realistic Rain Generation for Deraining and Object Detection in Rain (Shen Zheng et al., 2023)

{{<citation>}}

Shen Zheng, Changjie Lu, Srinivasa G. Narasimhan. (2023)  
**TPSeNCE: Towards Artifact-Free Realistic Rain Generation for Deraining and Object Detection in Rain**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.00660v1)  

---


**ABSTRACT**  
Rain generation algorithms have the potential to improve the generalization of deraining methods and scene understanding in rainy conditions. However, in practice, they produce artifacts and distortions and struggle to control the amount of rain generated due to a lack of proper constraints. In this paper, we propose an unpaired image-to-image translation framework for generating realistic rainy images. We first introduce a Triangular Probability Similarity (TPS) constraint to guide the generated images toward clear and rainy images in the discriminator manifold, thereby minimizing artifacts and distortions during rain generation. Unlike conventional contrastive learning approaches, which indiscriminately push negative samples away from the anchors, we propose a Semantic Noise Contrastive Estimation (SeNCE) strategy and reassess the pushing force of negative samples based on the semantic similarity between the clear and the rainy images and the feature similarity between the anchor and the negative samples. Experiments demonstrate realistic rain generation with minimal artifacts and distortions, which benefits image deraining and object detection in rain. Furthermore, the method can be used to generate realistic snowy and night images, underscoring its potential for broader applicability. Code is available at https://github.com/ShenZheng2000/TPSeNCE.

{{</citation>}}


### (84/121) PAUMER: Patch Pausing Transformer for Semantic Segmentation (Evann Courdier et al., 2023)

{{<citation>}}

Evann Courdier, Prabhu Teja Sivaprasad, François Fleuret. (2023)  
**PAUMER: Patch Pausing Transformer for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2311.00586v1)  

---


**ABSTRACT**  
We study the problem of improving the efficiency of segmentation transformers by using disparate amounts of computation for different parts of the image. Our method, PAUMER, accomplishes this by pausing computation for patches that are deemed to not need any more computation before the final decoder. We use the entropy of predictions computed from intermediate activations as the pausing criterion, and find this aligns well with semantics of the image. Our method has a unique advantage that a single network trained with the proposed strategy can be effortlessly adapted at inference to various run-time requirements by modulating its pausing parameters. On two standard segmentation datasets, Cityscapes and ADE20K, we show that our method operates with about a $50\%$ higher throughput with an mIoU drop of about $0.65\%$ and $4.6\%$ respectively.

{{</citation>}}


### (85/121) LLaVA-Interactive: An All-in-One Demo for Image Chat, Segmentation, Generation and Editing (Wei-Ge Chen et al., 2023)

{{<citation>}}

Wei-Ge Chen, Irina Spiridonova, Jianwei Yang, Jianfeng Gao, Chunyuan Li. (2023)  
**LLaVA-Interactive: An All-in-One Demo for Image Chat, Segmentation, Generation and Editing**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-HC, cs-MM, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.00571v1)  

---


**ABSTRACT**  
LLaVA-Interactive is a research prototype for multimodal human-AI interaction. The system can have multi-turn dialogues with human users by taking multimodal user inputs and generating multimodal responses. Importantly, LLaVA-Interactive goes beyond language prompt, where visual prompt is enabled to align human intents in the interaction. The development of LLaVA-Interactive is extremely cost-efficient as the system combines three multimodal skills of pre-built AI models without additional model training: visual chat of LLaVA, image segmentation from SEEM, as well as image generation and editing from GLIGEN. A diverse set of application scenarios is presented to demonstrate the promises of LLaVA-Interactive and to inspire future research in multimodal interactive systems.

{{</citation>}}


### (86/121) Detecting Visual Cues in the Intensive Care Unit and Association with Patient Clinical Status (Subhash Nerella et al., 2023)

{{<citation>}}

Subhash Nerella, Ziyuan Guan, Andrea Davidson, Yuanfang Ren, Tezcan Baslanti, Brooke Armfield, Patrick Tighe, Azra Bihorac, Parisa Rashidi. (2023)  
**Detecting Visual Cues in the Intensive Care Unit and Association with Patient Clinical Status**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Clinical, Transformer  
[Paper Link](http://arxiv.org/abs/2311.00565v1)  

---


**ABSTRACT**  
Intensive Care Units (ICU) provide close supervision and continuous care to patients with life-threatening conditions. However, continuous patient assessment in the ICU is still limited due to time constraints and the workload on healthcare providers. Existing patient assessments in the ICU such as pain or mobility assessment are mostly sporadic and administered manually, thus introducing the potential for human errors. Developing Artificial intelligence (AI) tools that can augment human assessments in the ICU can be beneficial for providing more objective and granular monitoring capabilities. For example, capturing the variations in a patient's facial cues related to pain or agitation can help in adjusting pain-related medications or detecting agitation-inducing conditions such as delirium. Additionally, subtle changes in visual cues during or prior to adverse clinical events could potentially aid in continuous patient monitoring when combined with high-resolution physiological signals and Electronic Health Record (EHR) data. In this paper, we examined the association between visual cues and patient condition including acuity status, acute brain dysfunction, and pain. We leveraged our AU-ICU dataset with 107,064 frames collected in the ICU annotated with facial action units (AUs) labels by trained annotators. We developed a new "masked loss computation" technique that addresses the data imbalance problem by maximizing data resource utilization. We trained the model using our AU-ICU dataset in conjunction with three external datasets to detect 18 AUs. The SWIN Transformer model achieved 0.57 mean F1-score and 0.89 mean accuracy on the test set. Additionally, we performed AU inference on 634,054 frames to evaluate the association between facial AUs and clinically important patient conditions such as acuity status, acute brain dysfunction, and pain.

{{</citation>}}


### (87/121) MNN: Mixed Nearest-Neighbors for Self-Supervised Learning (Chen Peng et al., 2023)

{{<citation>}}

Chen Peng, Xianzhong Long, Yun Li. (2023)  
**MNN: Mixed Nearest-Neighbors for Self-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.00562v1)  

---


**ABSTRACT**  
In contrastive self-supervised learning, positive samples are typically drawn from the same image but in different augmented views, resulting in a relatively limited source of positive samples. An effective way to alleviate this problem is to incorporate the relationship between samples, which involves including the top-k nearest neighbors of positive samples in the framework. However, the problem of false neighbors (i.e., neighbors that do not belong to the same category as the positive sample) is an objective but often overlooked challenge due to the query of neighbor samples without human supervision. In this paper, we present a simple Self-supervised learning framework called Mixed Nearest-Neighbors for Self-Supervised Learning (MNN). MNN optimizes the influence of neighbor samples on the semantics of positive samples through an intuitive weighting approach and image mixture operations. The results of our study demonstrate that MNN exhibits exceptional generalization performance and training efficiency on four benchmark datasets.

{{</citation>}}


### (88/121) ProBio: A Protocol-guided Multimodal Dataset for Molecular Biology Lab (Jieming Cui et al., 2023)

{{<citation>}}

Jieming Cui, Ziren Gong, Baoxiong Jia, Siyuan Huang, Zilong Zheng, Jianzhu Ma, Yixin Zhu. (2023)  
**ProBio: A Protocol-guided Multimodal Dataset for Molecular Biology Lab**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.00556v1)  

---


**ABSTRACT**  
The challenge of replicating research results has posed a significant impediment to the field of molecular biology. The advent of modern intelligent systems has led to notable progress in various domains. Consequently, we embarked on an investigation of intelligent monitoring systems as a means of tackling the issue of the reproducibility crisis. Specifically, we first curate a comprehensive multimodal dataset, named ProBio, as an initial step towards this objective. This dataset comprises fine-grained hierarchical annotations intended for the purpose of studying activity understanding in BioLab. Next, we devise two challenging benchmarks, transparent solution tracking and multimodal action recognition, to emphasize the unique characteristics and difficulties associated with activity understanding in BioLab settings. Finally, we provide a thorough experimental evaluation of contemporary video understanding models and highlight their limitations in this specialized domain to identify potential avenues for future research. We hope ProBio with associated benchmarks may garner increased focus on modern AI techniques in the realm of molecular biology.

{{</citation>}}


### (89/121) Group Distributionally Robust Knowledge Distillation (Konstantinos Vilouras et al., 2023)

{{<citation>}}

Konstantinos Vilouras, Xiao Liu, Pedro Sanchez, Alison Q. O'Neil, Sotirios A. Tsaftaris. (2023)  
**Group Distributionally Robust Knowledge Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2311.00476v1)  

---


**ABSTRACT**  
Knowledge distillation enables fast and effective transfer of features learned from a bigger model to a smaller one. However, distillation objectives are susceptible to sub-population shifts, a common scenario in medical imaging analysis which refers to groups/domains of data that are underrepresented in the training set. For instance, training models on health data acquired from multiple scanners or hospitals can yield subpar performance for minority groups. In this paper, inspired by distributionally robust optimization (DRO) techniques, we address this shortcoming by proposing a group-aware distillation loss. During optimization, a set of weights is updated based on the per-group losses at a given iteration. This way, our method can dynamically focus on groups that have low performance during training. We empirically validate our method, GroupDistil on two benchmark datasets (natural images and cardiac MRIs) and show consistent improvement in terms of worst-group accuracy.

{{</citation>}}


### (90/121) CLIP-AD: A Language-Guided Staged Dual-Path Model for Zero-shot Anomaly Detection (Xuhai Chen et al., 2023)

{{<citation>}}

Xuhai Chen, Jiangning Zhang, Guanzhong Tian, Haoyang He, Wuhao Zhang, Yabiao Wang, Chengjie Wang, Yunsheng Wu, Yong Liu. (2023)  
**CLIP-AD: A Language-Guided Staged Dual-Path Model for Zero-shot Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.00453v1)  

---


**ABSTRACT**  
This paper considers zero-shot Anomaly Detection (AD), a valuable yet under-studied task, which performs AD without any reference images of the test objects. Specifically, we employ a language-guided strategy and propose a simple-yet-effective architecture CLIP-AD, leveraging the superior zero-shot classification capabilities of the large vision-language model CLIP. A natural idea for anomaly segmentation is to directly calculate the similarity between text/image features, but we observe opposite predictions and irrelevant highlights in the results. Inspired by the phenomena, we introduce a Staged Dual-Path model (SDP) that effectively uses features from various levels and applies architecture and feature surgery to address these issues. Furthermore, delving beyond surface phenomena, we identify the problem arising from misalignment of text/image features in the joint embedding space. Thus, we introduce a fine-tuning strategy by adding linear layers and construct an extended model SDP+, further enhancing the performance. Abundant experiments demonstrate the effectiveness of our approach, e.g., on VisA, SDP outperforms SOTA by +1.0/+1.2 in classification/segmentation F1 scores, while SDP+ achieves +1.9/+11.7 improvements.

{{</citation>}}


### (91/121) On Manipulating Scene Text in the Wild with Diffusion Models (Joshua Santoso et al., 2023)

{{<citation>}}

Joshua Santoso, Christian Simon, Williem Pao. (2023)  
**On Manipulating Scene Text in the Wild with Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2311.00734v1)  

---


**ABSTRACT**  
Diffusion models have gained attention for image editing yielding impressive results in text-to-image tasks. On the downside, one might notice that generated images of stable diffusion models suffer from deteriorated details. This pitfall impacts image editing tasks that require information preservation e.g., scene text editing. As a desired result, the model must show the capability to replace the text on the source image to the target text while preserving the details e.g., color, font size, and background. To leverage the potential of diffusion models, in this work, we introduce Diffusion-BasEd Scene Text manipulation Network so-called DBEST. Specifically, we design two adaptation strategies, namely one-shot style adaptation and text-recognition guidance. In experiments, we thoroughly assess and compare our proposed method against state-of-the-arts on various scene text datasets, then provide extensive ablation studies for each granularity to analyze our performance gain. Also, we demonstrate the effectiveness of our proposed method to synthesize scene text indicated by competitive Optical Character Recognition (OCR) accuracy. Our method achieves 94.15% and 98.12% on COCO-text and ICDAR2013 datasets for character-level evaluation.

{{</citation>}}


### (92/121) Improving Robustness for Vision Transformer with a Simple Dynamic Scanning Augmentation (Shashank Kotyan et al., 2023)

{{<citation>}}

Shashank Kotyan, Danilo Vasconcellos Vargas. (2023)  
**Improving Robustness for Vision Transformer with a Simple Dynamic Scanning Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Augmentation, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.00441v1)  

---


**ABSTRACT**  
Vision Transformer (ViT) has demonstrated promising performance in computer vision tasks, comparable to state-of-the-art neural networks. Yet, this new type of deep neural network architecture is vulnerable to adversarial attacks limiting its capabilities in terms of robustness. This article presents a novel contribution aimed at further improving the accuracy and robustness of ViT, particularly in the face of adversarial attacks. We propose an augmentation technique called `Dynamic Scanning Augmentation' that leverages dynamic input sequences to adaptively focus on different patches, thereby maintaining performance and robustness. Our detailed investigations reveal that this adaptability to the input sequence induces significant changes in the attention mechanism of ViT, even for the same image. We introduce four variations of Dynamic Scanning Augmentation, outperforming ViT in terms of both robustness to adversarial attacks and accuracy against natural images, with one variant showing comparable results. By integrating our augmentation technique, we observe a substantial increase in ViT's robustness, improving it from $17\%$ to $92\%$ measured across different types of adversarial attacks. These findings, together with other comprehensive tests, indicate that Dynamic Scanning Augmentation enhances accuracy and robustness by promoting a more adaptive type of attention. In conclusion, this work contributes to the ongoing research on Vision Transformers by introducing Dynamic Scanning Augmentation as a technique for improving the accuracy and robustness of ViT. The observed results highlight the potential of this approach in advancing computer vision tasks and merit further exploration in future studies.

{{</citation>}}


### (93/121) Enhancing Traffic Object Detection in Variable Illumination with RGB-Event Fusion (Zhanwen Liu et al., 2023)

{{<citation>}}

Zhanwen Liu, Nan Yang, Yang Wang, Yuke Li, Xiangmo Zhao, Fei-Yue Wang. (2023)  
**Enhancing Traffic Object Detection in Variable Illumination with RGB-Event Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.00436v1)  

---


**ABSTRACT**  
Traffic object detection under variable illumination is challenging due to the information loss caused by the limited dynamic range of conventional frame-based cameras. To address this issue, we introduce bio-inspired event cameras and propose a novel Structure-aware Fusion Network (SFNet) that extracts sharp and complete object structures from the event stream to compensate for the lost information in images through cross-modality fusion, enabling the network to obtain illumination-robust representations for traffic object detection. Specifically, to mitigate the sparsity or blurriness issues arising from diverse motion states of traffic objects in fixed-interval event sampling methods, we propose the Reliable Structure Generation Network (RSGNet) to generate Speed Invariant Frames (SIF), ensuring the integrity and sharpness of object structures. Next, we design a novel Adaptive Feature Complement Module (AFCM) which guides the adaptive fusion of two modality features to compensate for the information loss in the images by perceiving the global lightness distribution of the images, thereby generating illumination-robust representations. Finally, considering the lack of large-scale and high-quality annotations in the existing event-based object detection datasets, we build a DSEC-Det dataset, which consists of 53 sequences with 63,931 images and more than 208,000 labels for 8 classes. Extensive experimental results demonstrate that our proposed SFNet can overcome the perceptual boundaries of conventional cameras and outperform the frame-based method by 8.0% in mAP50 and 5.9% in mAP50:95. Our code and dataset will be available at https://github.com/YN-Yang/SFNet.

{{</citation>}}


### (94/121) A Spatial-Temporal Transformer based Framework For Human Pose Assessment And Correction in Education Scenarios (Wenyang Hu et al., 2023)

{{<citation>}}

Wenyang Hu, Kai Liu, Libin Liu, Huiliang Shang. (2023)  
**A Spatial-Temporal Transformer based Framework For Human Pose Assessment And Correction in Education Scenarios**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.00401v1)  

---


**ABSTRACT**  
Human pose assessment and correction play a crucial role in applications across various fields, including computer vision, robotics, sports analysis, healthcare, and entertainment. In this paper, we propose a Spatial-Temporal Transformer based Framework (STTF) for human pose assessment and correction in education scenarios such as physical exercises and science experiment. The framework comprising skeletal tracking, pose estimation, posture assessment, and posture correction modules to educate students with professional, quick-to-fix feedback. We also create a pose correction method to provide corrective feedback in the form of visual aids. We test the framework with our own dataset. It comprises (a) new recordings of five exercises, (b) existing recordings found on the internet of the same exercises, and (c) corrective feedback on the recordings by professional athletes and teachers. Results show that our model can effectively measure and comment on the quality of students' actions. The STTF leverages the power of transformer models to capture spatial and temporal dependencies in human poses, enabling accurate assessment and effective correction of students' movements.

{{</citation>}}


### (95/121) Learning Cooperative Trajectory Representations for Motion Forecasting (Hongzhi Ruan et al., 2023)

{{<citation>}}

Hongzhi Ruan, Haibao Yu, Wenxian Yang, Siqi Fan, Yingjuan Tang, Zaiqing Nie. (2023)  
**Learning Cooperative Trajectory Representations for Motion Forecasting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.00371v1)  

---


**ABSTRACT**  
Motion forecasting is an essential task for autonomous driving, and the effective information utilization from infrastructure and other vehicles can enhance motion forecasting capabilities. Existing research have primarily focused on leveraging single-frame cooperative information to enhance the limited perception capability of the ego vehicle, while underutilizing the motion and interaction information of traffic participants observed from cooperative devices. In this paper, we first propose the cooperative trajectory representations learning paradigm. Specifically, we present V2X-Graph, the first interpretable and end-to-end learning framework for cooperative motion forecasting. V2X-Graph employs an interpretable graph to fully leverage the cooperative motion and interaction contexts. Experimental results on the vehicle-to-infrastructure (V2I) motion forecasting dataset, V2X-Seq, demonstrate the effectiveness of V2X-Graph. To further evaluate on V2X scenario, we construct the first real-world vehicle-to-everything (V2X) motion forecasting dataset V2X-Traj, and the performance shows the advantage of our method. We hope both V2X-Graph and V2X-Traj can facilitate the further development of cooperative motion forecasting. Find project at https://github.com/AIR-THU/V2X-Graph, find data at https://github.com/AIR-THU/DAIR-V2X-Seq.

{{</citation>}}


### (96/121) Rethinking Samples Selection for Contrastive Learning: Mining of Potential Samples (Hengkui Dong et al., 2023)

{{<citation>}}

Hengkui Dong, Xianzhong Long, Yun Li. (2023)  
**Rethinking Samples Selection for Contrastive Learning: Mining of Potential Samples**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.00358v1)  

---


**ABSTRACT**  
Contrastive learning predicts whether two images belong to the same category by training a model to make their feature representations as close or as far away as possible. In this paper, we rethink how to mine samples in contrastive learning, unlike other methods, our approach is more comprehensive, taking into account both positive and negative samples, and mining potential samples from two aspects: First, for positive samples, we consider both the augmented sample views obtained by data augmentation and the mined sample views through data mining. Then, we weight and combine them using both soft and hard weighting strategies. Second, considering the existence of uninformative negative samples and false negative samples in the negative samples, we analyze the negative samples from the gradient perspective and finally mine negative samples that are neither too hard nor too easy as potential negative samples, i.e., those negative samples that are close to positive samples. The experiments show the obvious advantages of our method compared with some traditional self-supervised methods. Our method achieves 88.57%, 61.10%, and 36.69% top-1 accuracy on CIFAR10, CIFAR100, and TinyImagenet, respectively.

{{</citation>}}


### (97/121) LatentWarp: Consistent Diffusion Latents for Zero-Shot Video-to-Video Translation (Yuxiang Bao et al., 2023)

{{<citation>}}

Yuxiang Bao, Di Qiu, Guoliang Kang, Baochang Zhang, Bo Jin, Kaiye Wang, Pengfei Yan. (2023)  
**LatentWarp: Consistent Diffusion Latents for Zero-Shot Video-to-Video Translation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.00353v1)  

---


**ABSTRACT**  
Leveraging the generative ability of image diffusion models offers great potential for zero-shot video-to-video translation. The key lies in how to maintain temporal consistency across generated video frames by image diffusion models. Previous methods typically adopt cross-frame attention, \emph{i.e.,} sharing the \textit{key} and \textit{value} tokens across attentions of different frames, to encourage the temporal consistency. However, in those works, temporal inconsistency issue may not be thoroughly solved, rendering the fidelity of generated videos limited.%The current state of the art cross-frame attention method aims at maintaining fine-grained visual details across frames, but it is still challenged by the temporal coherence problem. In this paper, we find the bottleneck lies in the unconstrained query tokens and propose a new zero-shot video-to-video translation framework, named \textit{LatentWarp}. Our approach is simple: to constrain the query tokens to be temporally consistent, we further incorporate a warping operation in the latent space to constrain the query tokens. Specifically, based on the optical flow obtained from the original video, we warp the generated latent features of last frame to align with the current frame during the denoising process. As a result, the corresponding regions across the adjacent frames can share closely-related query tokens and attention outputs, which can further improve latent-level consistency to enhance visual temporal coherence of generated videos. Extensive experiment results demonstrate the superiority of \textit{LatentWarp} in achieving video-to-video translation with temporal coherence.

{{</citation>}}


### (98/121) fMRI-PTE: A Large-scale fMRI Pretrained Transformer Encoder for Multi-Subject Brain Activity Decoding (Xuelin Qian et al., 2023)

{{<citation>}}

Xuelin Qian, Yun Wang, Jingyang Huo, Jianfeng Feng, Yanwei Fu. (2023)  
**fMRI-PTE: A Large-scale fMRI Pretrained Transformer Encoder for Multi-Subject Brain Activity Decoding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.00342v1)  

---


**ABSTRACT**  
The exploration of brain activity and its decoding from fMRI data has been a longstanding pursuit, driven by its potential applications in brain-computer interfaces, medical diagnostics, and virtual reality. Previous approaches have primarily focused on individual subject analysis, highlighting the need for a more universal and adaptable framework, which is the core motivation behind our work. In this work, we propose fMRI-PTE, an innovative auto-encoder approach for fMRI pre-training, with a focus on addressing the challenges of varying fMRI data dimensions due to individual brain differences. Our approach involves transforming fMRI signals into unified 2D representations, ensuring consistency in dimensions and preserving distinct brain activity patterns. We introduce a novel learning strategy tailored for pre-training 2D fMRI images, enhancing the quality of reconstruction. fMRI-PTE's adaptability with image generators enables the generation of well-represented fMRI features, facilitating various downstream tasks, including within-subject and cross-subject brain activity decoding. Our contributions encompass introducing fMRI-PTE, innovative data transformation, efficient training, a novel learning strategy, and the universal applicability of our approach. Extensive experiments validate and support our claims, offering a promising foundation for further research in this domain.

{{</citation>}}


### (99/121) Space Narrative: Generating Images and 3D Scenes of Chinese Garden from Text using Deep Learning (Jiaxi Shi1 et al., 2023)

{{<citation>}}

Jiaxi Shi1, Hao Hua1. (2023)  
**Space Narrative: Generating Images and 3D Scenes of Chinese Garden from Text using Deep Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, stat-ML  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.00339v1)  

---


**ABSTRACT**  
The consistent mapping from poems to paintings is essential for the research and restoration of traditional Chinese gardens. But the lack of firsthand ma-terial is a great challenge to the reconstruction work. In this paper, we pro-pose a method to generate garden paintings based on text descriptions using deep learning method. Our image-text pair dataset consists of more than one thousand Ming Dynasty Garden paintings and their inscriptions and post-scripts. A latent text-to-image diffusion model learns the mapping from de-scriptive texts to garden paintings of the Ming Dynasty, and then the text description of Jichang Garden guides the model to generate new garden paintings. The cosine similarity between the guide text and the generated image is the evaluation criterion for the generated images. Our dataset is used to fine-tune the pre-trained diffusion model using Low-Rank Adapta-tion of Large Language Models (LoRA). We also transformed the generated images into a panorama and created a free-roam scene in Unity 3D. Our post-trained model is capable of generating garden images in the style of Ming Dynasty landscape paintings based on textual descriptions. The gener-ated images are compatible with three-dimensional presentation in Unity 3D.

{{</citation>}}


### (100/121) From Image to Language: A Critical Analysis of Visual Question Answering (VQA) Approaches, Challenges, and Opportunities (Md Farhan Ishmam et al., 2023)

{{<citation>}}

Md Farhan Ishmam, Md Sakib Hossain Shovon, M. F. Mridha, Nilanjan Dey. (2023)  
**From Image to Language: A Critical Analysis of Visual Question Answering (VQA) Approaches, Challenges, and Opportunities**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Computer Vision, NLP, Natural Language Processing, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.00308v1)  

---


**ABSTRACT**  
The multimodal task of Visual Question Answering (VQA) encompassing elements of Computer Vision (CV) and Natural Language Processing (NLP), aims to generate answers to questions on any visual input. Over time, the scope of VQA has expanded from datasets focusing on an extensive collection of natural images to datasets featuring synthetic images, video, 3D environments, and various other visual inputs. The emergence of large pre-trained networks has shifted the early VQA approaches relying on feature extraction and fusion schemes to vision language pre-training (VLP) techniques. However, there is a lack of comprehensive surveys that encompass both traditional VQA architectures and contemporary VLP-based methods. Furthermore, the VLP challenges in the lens of VQA haven't been thoroughly explored, leaving room for potential open problems to emerge. Our work presents a survey in the domain of VQA that delves into the intricacies of VQA datasets and methods over the field's history, introduces a detailed taxonomy to categorize the facets of VQA, and highlights the recent trends, challenges, and scopes for improvement. We further generalize VQA to multimodal question answering, explore tasks related to VQA, and present a set of open problems for future investigation. The work aims to navigate both beginners and experts by shedding light on the potential avenues of research and expanding the boundaries of the field.

{{</citation>}}


### (101/121) Graph Representation Learning for Infrared and Visible Image Fusion (Jing Li et al., 2023)

{{<citation>}}

Jing Li, Lu Bai, Bin Yang, Chang Li, Lingfei Ma, Edwin R. Hancock. (2023)  
**Graph Representation Learning for Infrared and Visible Image Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.00291v1)  

---


**ABSTRACT**  
Infrared and visible image fusion aims to extract complementary features to synthesize a single fused image. Many methods employ convolutional neural networks (CNNs) to extract local features due to its translation invariance and locality. However, CNNs fail to consider the image's non-local self-similarity (NLss), though it can expand the receptive field by pooling operations, it still inevitably leads to information loss. In addition, the transformer structure extracts long-range dependence by considering the correlativity among all image patches, leading to information redundancy of such transformer-based methods. However, graph representation is more flexible than grid (CNN) or sequence (transformer structure) representation to address irregular objects, and graph can also construct the relationships among the spatially repeatable details or texture with far-space distance. Therefore, to address the above issues, it is significant to convert images into the graph space and thus adopt graph convolutional networks (GCNs) to extract NLss. This is because the graph can provide a fine structure to aggregate features and propagate information across the nearest vertices without introducing redundant information. Concretely, we implement a cascaded NLss extraction pattern to extract NLss of intra- and inter-modal by exploring interactions of different image pixels in intra- and inter-image positional distance. We commence by preforming GCNs on each intra-modal to aggregate features and propagate information to extract independent intra-modal NLss. Then, GCNs are performed on the concatenate intra-modal NLss features of infrared and visible images, which can explore the cross-domain NLss of inter-modal to reconstruct the fused image. Ablation studies and extensive experiments illustrates the effectiveness and superiority of the proposed method on three datasets.

{{</citation>}}


### (102/121) Re-Scoring Using Image-Language Similarity for Few-Shot Object Detection (Min Jae Jung et al., 2023)

{{<citation>}}

Min Jae Jung, Seung Dae Han, Joohee Kim. (2023)  
**Re-Scoring Using Image-Language Similarity for Few-Shot Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Few-Shot, Object Detection  
[Paper Link](http://arxiv.org/abs/2311.00278v1)  

---


**ABSTRACT**  
Few-shot object detection, which focuses on detecting novel objects with few labels, is an emerging challenge in the community. Recent studies show that adapting a pre-trained model or modified loss function can improve performance. In this paper, we explore leveraging the power of Contrastive Language-Image Pre-training (CLIP) and hard negative classification loss in low data setting. Specifically, we propose Re-scoring using Image-language Similarity for Few-shot object detection (RISF) which extends Faster R-CNN by introducing Calibration Module using CLIP (CM-CLIP) and Background Negative Re-scale Loss (BNRL). The former adapts CLIP, which performs zero-shot classification, to re-score the classification scores of a detector using image-class similarities, the latter is modified classification loss considering the punishment for fake backgrounds as well as confusing categories on a generalized few-shot object detection dataset. Extensive experiments on MS-COCO and PASCAL VOC show that the proposed RISF substantially outperforms the state-of-the-art approaches. The code will be available.

{{</citation>}}


### (103/121) RAUNE-Net: A Residual and Attention-Driven Underwater Image Enhancement Method (Wangzhen Peng et al., 2023)

{{<citation>}}

Wangzhen Peng, Chenghao Zhou, Runze Hu, Jingchao Cao, Yutao Liu. (2023)  
**RAUNE-Net: A Residual and Attention-Driven Underwater Image Enhancement Method**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.00246v1)  

---


**ABSTRACT**  
Underwater image enhancement (UIE) poses challenges due to distinctive properties of the underwater environment, including low contrast, high turbidity, visual blurriness, and color distortion. In recent years, the application of deep learning has quietly revolutionized various areas of scientific research, including UIE. However, existing deep learning-based UIE methods generally suffer from issues of weak robustness and limited adaptability. In this paper, inspired by residual and attention mechanisms, we propose a more reliable and reasonable UIE network called RAUNE-Net by employing residual learning of high-level features at the network's bottle-neck and two aspects of attention manipulations in the down-sampling procedure. Furthermore, we collect and create two datasets specifically designed for evaluating UIE methods, which contains different types of underwater distortions and degradations. The experimental validation demonstrates that our method obtains promising objective performance and consistent visual results across various real-world underwater images compared to other eight UIE methods. Our example code and datasets are publicly available at https://github.com/fansuregrin/RAUNE-Net.

{{</citation>}}


### (104/121) 1DFormer: Learning 1D Landmark Representations via Transformer for Facial Landmark Tracking (Shi Yin et al., 2023)

{{<citation>}}

Shi Yin, Shijie Huan, Defu Lian, Shangfei Wang, Jinshui Hu, Tao Guo, Bing Yin, Baocai Yin, Cong Liu. (2023)  
**1DFormer: Learning 1D Landmark Representations via Transformer for Facial Landmark Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.00241v1)  

---


**ABSTRACT**  
Recently, heatmap regression methods based on 1D landmark representations have shown prominent performance on locating facial landmarks. However, previous methods ignored to make deep explorations on the good potentials of 1D landmark representations for sequential and structural modeling of multiple landmarks to track facial landmarks. To address this limitation, we propose a Transformer architecture, namely 1DFormer, which learns informative 1D landmark representations by capturing the dynamic and the geometric patterns of landmarks via token communications in both temporal and spatial dimensions for facial landmark tracking. For temporal modeling, we propose a recurrent token mixing mechanism, an axis-landmark-positional embedding mechanism, as well as a confidence-enhanced multi-head attention mechanism to adaptively and robustly embed long-term landmark dynamics into their 1D representations; for structure modeling, we design intra-group and inter-group structure modeling mechanisms to encode the component-level as well as global-level facial structure patterns as a refinement for the 1D representations of landmarks through token communications in the spatial dimension via 1D convolutional layers. Experimental results on the 300VW and the TF databases show that 1DFormer successfully models the long-range sequential patterns as well as the inherent facial structures to learn informative 1D representations of landmark sequences, and achieves state-of-the-art performance on facial landmark tracking.

{{</citation>}}


### (105/121) ChatGPT-Powered Hierarchical Comparisons for Image Classification (Zhiyuan Ren et al., 2023)

{{<citation>}}

Zhiyuan Ren, Yiyang Su, Xiaoming Liu. (2023)  
**ChatGPT-Powered Hierarchical Comparisons for Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ChatGPT, GPT, Image Classification  
[Paper Link](http://arxiv.org/abs/2311.00206v1)  

---


**ABSTRACT**  
The zero-shot open-vocabulary challenge in image classification is tackled by pretrained vision-language models like CLIP, which benefit from incorporating class-specific knowledge from large language models (LLMs) like ChatGPT. However, biases in CLIP lead to similar descriptions for distinct but related classes, prompting our novel image classification framework via hierarchical comparisons: using LLMs to recursively group classes into hierarchies and classifying images by comparing image-text embeddings at each hierarchy level, resulting in an intuitive, effective, and explainable approach.

{{</citation>}}


### (106/121) ZEETAD: Adapting Pretrained Vision-Language Model for Zero-Shot End-to-End Temporal Action Detection (Thinh Phan et al., 2023)

{{<citation>}}

Thinh Phan, Khoa Vo, Duy Le, Gianfranco Doretto, Donald Adjeroh, Ngan Le. (2023)  
**ZEETAD: Adapting Pretrained Vision-Language Model for Zero-Shot End-to-End Temporal Action Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model, Transformer, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.00729v1)  

---


**ABSTRACT**  
Temporal action detection (TAD) involves the localization and classification of action instances within untrimmed videos. While standard TAD follows fully supervised learning with closed-set setting on large training data, recent zero-shot TAD methods showcase the promising of open-set setting by leveraging large-scale contrastive visual-language (ViL) pretrained models. However, existing zero-shot TAD methods have limitations on how to properly construct the strong relationships between two interdependent tasks of localization and classification and adapt ViL model to video understanding. In this work, we present ZEETAD, featuring two modules: dual-localization and zero-shot proposal classification. The former is a Transformer-based module that detects action events while selectively collecting crucial semantic embeddings for later recognition. The latter one, CLIP-based module, generates semantic embeddings from text and frame inputs for each temporal unit. Additionally, we enhance discriminative capability on unseen classes by minimally updating the frozen CLIP encoder with lightweight adapters. Extensive experiments on THUMOS14 and ActivityNet-1.3 datasets demonstrate our approach's superior performance in zero-shot TAD and effective knowledge transfer from ViL models to unseen action categories.

{{</citation>}}


## cs.DL (1)



### (107/121) Integrating measures of replicability into literature search: Challenges and opportunities (Chuhao Wu et al., 2023)

{{<citation>}}

Chuhao Wu, Tatiana Chakravorti, John Carroll, Sarah Rajtmajer. (2023)  
**Integrating measures of replicability into literature search: Challenges and opportunities**  

---
Primary Category: cs.DL  
Categories: cs-DL, cs-HC, cs.DL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.00653v1)  

---


**ABSTRACT**  
Challenges to reproducibility and replicability have gained widespread attention over the past decade, driven by a number of large replication projects with lukewarm success rates. A nascent work has emerged developing algorithms to estimate, or predict, the replicability of published findings. The current study explores ways in which AI-enabled signals of confidence in research might be integrated into literature search. We interview 17 PhD researchers about their current processes for literature search and ask them to provide feedback on a prototype replicability estimation tool. Our findings suggest that information about replicability can support researchers throughout literature review and research design processes. However, explainability and interpretability of system outputs is critical, and potential drawbacks of AI-enabled confidence assessment need to be further studied before such tools could be widely accepted and deployed. We discuss implications for the design of technological tools to support scholarly activities and advance reproducibility and replicability.

{{</citation>}}


## cs.MA (1)



### (108/121) Emergence of Collective Open-Ended Exploration from Decentralized Meta-Reinforcement Learning (Richard Bornemann et al., 2023)

{{<citation>}}

Richard Bornemann, Gautier Hamon, Eleni Nisioti, Clément Moulin-Frier. (2023)  
**Emergence of Collective Open-Ended Exploration from Decentralized Meta-Reinforcement Learning**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.00651v2)  

---


**ABSTRACT**  
Recent works have proven that intricate cooperative behaviors can emerge in agents trained using meta reinforcement learning on open ended task distributions using self-play. While the results are impressive, we argue that self-play and other centralized training techniques do not accurately reflect how general collective exploration strategies emerge in the natural world: through decentralized training and over an open-ended distribution of tasks. In this work we therefore investigate the emergence of collective exploration strategies, where several agents meta-learn independent recurrent policies on an open ended distribution of tasks. To this end we introduce a novel environment with an open ended procedurally generated task space which dynamically combines multiple subtasks sampled from five diverse task types to form a vast distribution of task trees. We show that decentralized agents trained in our environment exhibit strong generalization abilities when confronted with novel objects at test time. Additionally, despite never being forced to cooperate during training the agents learn collective exploration strategies which allow them to solve novel tasks never encountered during training. We further find that the agents learned collective exploration strategies extend to an open ended task setting, allowing them to solve task trees of twice the depth compared to the ones seen during training. Our open source code as well as videos of the agents can be found on our companion website.

{{</citation>}}


## cs.IR (5)



### (109/121) GATSY: Graph Attention Network for Music Artist Similarity (Andrea Giuseppe Di Francesco et al., 2023)

{{<citation>}}

Andrea Giuseppe Di Francesco, Giuliano Giampietro, Indro Spinelli, Danilo Comminiello. (2023)  
**GATSY: Graph Attention Network for Music Artist Similarity**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Attention, Graph Attention Network  
[Paper Link](http://arxiv.org/abs/2311.00635v1)  

---


**ABSTRACT**  
The artist similarity quest has become a crucial subject in social and scientific contexts. Modern research solutions facilitate music discovery according to user tastes. However, defining similarity among artists may involve several aspects, even related to a subjective perspective, and it often affects a recommendation. This paper presents GATSY, a recommendation system built upon graph attention networks and driven by a clusterized embedding of artists. The proposed framework takes advantage of a graph topology of the input data to achieve outstanding performance results without relying heavily on hand-crafted features. This flexibility allows us to introduce fictitious artists in a music dataset, create bridges to previously unrelated artists, and get recommendations conditioned by possibly heterogeneous sources. Experimental results prove the effectiveness of the proposed method with respect to state-of-the-art solutions.

{{</citation>}}


### (110/121) Bayes-enhanced Multi-view Attention Networks for Robust POI Recommendation (Jiangnan Xia et al., 2023)

{{<citation>}}

Jiangnan Xia, Yu Yang, Senzhang Wang, Hongzhi Yin, Jiannong Cao, Philip S. Yu. (2023)  
**Bayes-enhanced Multi-view Attention Networks for Robust POI Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-SI, cs.IR  
Keywords: Attention, Social Network  
[Paper Link](http://arxiv.org/abs/2311.00491v1)  

---


**ABSTRACT**  
POI recommendation is practically important to facilitate various Location-Based Social Network services, and has attracted rising research attention recently. Existing works generally assume the available POI check-ins reported by users are the ground-truth depiction of user behaviors. However, in real application scenarios, the check-in data can be rather unreliable due to both subjective and objective causes including positioning error and user privacy concerns, leading to significant negative impacts on the performance of the POI recommendation. To this end, we investigate a novel problem of robust POI recommendation by considering the uncertainty factors of the user check-ins, and proposes a Bayes-enhanced Multi-view Attention Network. Specifically, we construct personal POI transition graph, the semantic-based POI graph and distance-based POI graph to comprehensively model the dependencies among the POIs. As the personal POI transition graph is usually sparse and sensitive to noise, we design a Bayes-enhanced spatial dependency learning module for data augmentation from the local view. A Bayesian posterior guided graph augmentation approach is adopted to generate a new graph with collaborative signals to increase the data diversity. Then both the original and the augmented graphs are used for POI representation learning to counteract the data uncertainty issue. Next, the POI representations of the three view graphs are input into the proposed multi-view attention-based user preference learning module. By incorporating the semantic and distance correlations of POIs, the user preference can be effectively refined and finally robust recommendation results are achieved. The results of extensive experiments show that BayMAN significantly outperforms the state-of-the-art methods in POI recommendation when the available check-ins are incomplete and noisy.

{{</citation>}}


### (111/121) LLMRec: Large Language Models with Graph Augmentation for Recommendation (Wei Wei et al., 2023)

{{<citation>}}

Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, Chao Huang. (2023)  
**LLMRec: Large Language Models with Graph Augmentation for Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Augmentation, Language Model  
[Paper Link](http://arxiv.org/abs/2311.00423v1)  

---


**ABSTRACT**  
The problem of data sparsity has long been a challenge in recommendation systems, and previous studies have attempted to address this issue by incorporating side information. However, this approach often introduces side effects such as noise, availability issues, and low data quality, which in turn hinder the accurate modeling of user preferences and adversely impact recommendation performance. In light of the recent advancements in large language models (LLMs), which possess extensive knowledge bases and strong reasoning capabilities, we propose a novel framework called LLMRec that enhances recommender systems by employing three simple yet effective LLM-based graph augmentation strategies. Our approach leverages the rich content available within online platforms (e.g., Netflix, MovieLens) to augment the interaction graph in three ways: (i) reinforcing user-item interaction egde, (ii) enhancing the understanding of item node attributes, and (iii) conducting user node profiling, intuitively from the natural language perspective. By employing these strategies, we address the challenges posed by sparse implicit feedback and low-quality side information in recommenders. Besides, to ensure the quality of the augmentation, we develop a denoised data robustification mechanism that includes techniques of noisy implicit feedback pruning and MAE-based feature enhancement that help refine the augmented data and improve its reliability. Furthermore, we provide theoretical analysis to support the effectiveness of LLMRec and clarify the benefits of our method in facilitating model optimization. Experimental results on benchmark datasets demonstrate the superiority of our LLM-based augmentation approach over state-of-the-art techniques. To ensure reproducibility, we have made our code and augmented data publicly available at: https://github.com/HKUDS/LLMRec.git

{{</citation>}}


### (112/121) Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems (Hao Zhang et al., 2023)

{{<citation>}}

Hao Zhang, Mingyue Cheng, Qi Liu, Zhiding Liu, Enhong Chen. (2023)  
**Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Perplexity  
[Paper Link](http://arxiv.org/abs/2311.00388v1)  

---


**ABSTRACT**  
Sequential recommender systems (SRS) have gained widespread popularity in recommendation due to their ability to effectively capture dynamic user preferences. One default setting in the current SRS is to uniformly consider each historical behavior as a positive interaction. Actually, this setting has the potential to yield sub-optimal performance, as each item makes a distinct contribution to the user's interest. For example, purchased items should be given more importance than clicked ones. Hence, we propose a general automatic sampling framework, named AutoSAM, to non-uniformly treat historical behaviors. Specifically, AutoSAM augments the standard sequential recommendation architecture with an additional sampler layer to adaptively learn the skew distribution of the raw input, and then sample informative sub-sets to build more generalizable SRS. To overcome the challenges of non-differentiable sampling actions and also introduce multiple decision factors for sampling, we further introduce a novel reinforcement learning based method to guide the training of the sampler. We theoretically design multi-objective sampling rewards including Future Prediction and Sequence Perplexity, and then optimize the whole framework in an end-to-end manner by combining the policy gradient. We conduct extensive experiments on benchmark recommender models and four real-world datasets. The experimental results demonstrate the effectiveness of the proposed approach. We will make our code publicly available after the acceptance.

{{</citation>}}


### (113/121) Caseformer: Pre-training for Legal Case Retrieval (Weihang Su et al., 2023)

{{<citation>}}

Weihang Su, Qingyao Ai, Yueyue Wu, Yixiao Ma, Haitao Li, Yiqun Liu. (2023)  
**Caseformer: Pre-training for Legal Case Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2311.00333v1)  

---


**ABSTRACT**  
Legal case retrieval aims to help legal workers find relevant cases related to their cases at hand, which is important for the guarantee of fairness and justice in legal judgments. While recent advances in neural retrieval methods have significantly improved the performance of open-domain retrieval tasks (e.g., Web search), their advantages have not been observed in legal case retrieval due to their thirst for annotated data. As annotating large-scale training data in legal domains is prohibitive due to the need for domain expertise, traditional search techniques based on lexical matching such as TF-IDF, BM25, and Query Likelihood are still prevalent in legal case retrieval systems. While previous studies have designed several pre-training methods for IR models in open-domain tasks, these methods are usually suboptimal in legal case retrieval because they cannot understand and capture the key knowledge and data structures in the legal corpus. To this end, we propose a novel pre-training framework named Caseformer that enables the pre-trained models to learn legal knowledge and domain-specific relevance information in legal case retrieval without any human-labeled data. Through three unsupervised learning tasks, Caseformer is able to capture the special language, document structure, and relevance patterns of legal case documents, making it a strong backbone for downstream legal case retrieval tasks. Experimental results show that our model has achieved state-of-the-art performance in both zero-shot and full-data fine-tuning settings. Also, experiments on both Chinese and English legal datasets demonstrate that the effectiveness of Caseformer is language-independent in legal case retrieval.

{{</citation>}}


## stat.ML (1)



### (114/121) Flexible Tails for Normalising Flows, with Application to the Modelling of Financial Return Data (Tennessee Hickling et al., 2023)

{{<citation>}}

Tennessee Hickling, Dennis Prangle. (2023)  
**Flexible Tails for Normalising Flows, with Application to the Modelling of Financial Return Data**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2311.00580v1)  

---


**ABSTRACT**  
We propose a transformation capable of altering the tail properties of a distribution, motivated by extreme value theory, which can be used as a layer in a normalizing flow to approximate multivariate heavy tailed distributions. We apply this approach to model financial returns, capturing potentially extreme shocks that arise in such data. The trained models can be used directly to generate new synthetic sets of potentially extreme returns

{{</citation>}}


## eess.IV (3)



### (115/121) A Robust Deep Learning Method with Uncertainty Estimation for the Pathological Classification of Renal Cell Carcinoma based on CT Images (Ni Yao et al., 2023)

{{<citation>}}

Ni Yao, Hang Hu, Kaicong Chen, Chen Zhao, Yuan Guo, Boya Li, Jiaofen Nan, Yanting Li, Chuang Han, Fubao Zhu, Weihua Zhou, Li Tian. (2023)  
**A Robust Deep Learning Method with Uncertainty Estimation for the Pathological Classification of Renal Cell Carcinoma based on CT Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV, physics-med-ph, q-bio-QM  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2311.00567v1)  

---


**ABSTRACT**  
Objectives To develop and validate a deep learning-based diagnostic model incorporating uncertainty estimation so as to facilitate radiologists in the preoperative differentiation of the pathological subtypes of renal cell carcinoma (RCC) based on CT images. Methods Data from 668 consecutive patients, pathologically proven RCC, were retrospectively collected from Center 1. By using five-fold cross-validation, a deep learning model incorporating uncertainty estimation was developed to classify RCC subtypes into clear cell RCC (ccRCC), papillary RCC (pRCC), and chromophobe RCC (chRCC). An external validation set of 78 patients from Center 2 further evaluated the model's performance. Results In the five-fold cross-validation, the model's area under the receiver operating characteristic curve (AUC) for the classification of ccRCC, pRCC, and chRCC was 0.868 (95% CI: 0.826-0.923), 0.846 (95% CI: 0.812-0.886), and 0.839 (95% CI: 0.802-0.88), respectively. In the external validation set, the AUCs were 0.856 (95% CI: 0.838-0.882), 0.787 (95% CI: 0.757-0.818), and 0.793 (95% CI: 0.758-0.831) for ccRCC, pRCC, and chRCC, respectively. Conclusions The developed deep learning model demonstrated robust performance in predicting the pathological subtypes of RCC, while the incorporated uncertainty emphasized the importance of understanding model confidence, which is crucial for assisting clinical decision-making for patients with renal tumors. Clinical relevance statement Our deep learning approach, integrated with uncertainty estimation, offers clinicians a dual advantage: accurate RCC subtype predictions complemented by diagnostic confidence references, promoting informed decision-making for patients with RCC.

{{</citation>}}


### (116/121) DEFN: Dual-Encoder Fourier Group Harmonics Network for Three-Dimensional Macular Hole Reconstruction with Stochastic Retinal Defect Augmentation and Dynamic Weight Composition (Xingru Huang et al., 2023)

{{<citation>}}

Xingru Huang, Yihao Guo, Jian Huang, Zhi Li, Tianyun Zhang, Kunyan Cai, Gaopeng Huang, Wenhao Chen, Zhaoyang Xu, Liangqiong Qu, Ji Hu, Tinyu Wang, Shaowei Jiang, Chenggang Yan, Yaoqi Sun, Xin Ye, Yaqi Wang. (2023)  
**DEFN: Dual-Encoder Fourier Group Harmonics Network for Three-Dimensional Macular Hole Reconstruction with Stochastic Retinal Defect Augmentation and Dynamic Weight Composition**  

---
Primary Category: eess.IV  
Categories: 68, 92, I-4; J-3, cs-CV, eess-IV, eess.IV  
Keywords: Attention, Augmentation  
[Paper Link](http://arxiv.org/abs/2311.00483v1)  

---


**ABSTRACT**  
The spatial and quantitative parameters of macular holes are vital for diagnosis, surgical choices, and post-op monitoring. Macular hole diagnosis and treatment rely heavily on spatial and quantitative data, yet the scarcity of such data has impeded the progress of deep learning techniques for effective segmentation and real-time 3D reconstruction. To address this challenge, we assembled the world's largest macular hole dataset, Retinal OCTfor Macular Hole Enhancement (ROME-3914), and a Comprehensive Archive for Retinal Segmentation (CARS-30k), both expertly annotated. In addition, we developed an innovative 3D segmentation network, the Dual-Encoder FuGH Network (DEFN), which integrates three innovative modules: Fourier Group Harmonics (FuGH), Simplified 3D Spatial Attention (S3DSA) and Harmonic Squeeze-and-Excitation Module (HSE). These three modules synergistically filter noise, reduce computational complexity, emphasize detailed features, and enhance the network's representation ability. We also proposed a novel data augmentation method, Stochastic Retinal Defect Injection (SRDI), and a network optimization strategy DynamicWeightCompose (DWC), to further improve the performance of DEFN. Compared with 13 baselines, our DEFN shows the best performance. We also offer precise 3D retinal reconstruction and quantitative metrics, bringing revolutionary diagnostic and therapeutic decision-making tools for ophthalmologists, and is expected to completely reshape the diagnosis and treatment patterns of difficult-to-treat macular degeneration. The source code is publicly available at: https://github.com/IIPL-HangzhouDianUniversity/DEFN-Pytorch.

{{</citation>}}


### (117/121) Crop Disease Classification using Support Vector Machines with Green Chromatic Coordinate (GCC) and Attention based feature extraction for IoT based Smart Agricultural Applications (Shashwat Jha et al., 2023)

{{<citation>}}

Shashwat Jha, Vishvaditya Luhach, Gauri Shanker Gupta, Beependra Singh. (2023)  
**Crop Disease Classification using Support Vector Machines with Green Chromatic Coordinate (GCC) and Attention based feature extraction for IoT based Smart Agricultural Applications**  

---
Primary Category: eess.IV  
Categories: cs-LG, eess-IV, eess.IV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2311.00429v1)  

---


**ABSTRACT**  
Crops hold paramount significance as they serve as the primary provider of energy, nutrition, and medicinal benefits for the human population. Plant diseases, however, can negatively affect leaves during agricultural cultivation, resulting in significant losses in crop output and economic value. Therefore, it is crucial for farmers to identify crop diseases. However, this method frequently necessitates hard work, a lot of planning, and in-depth familiarity with plant pathogens. Given these numerous obstacles, it is essential to provide solutions that can easily interface with mobile and IoT devices so that our farmers can guarantee the best possible crop development. Various machine learning (ML) as well as deep learning (DL) algorithms have been created & studied for the identification of plant disease detection, yielding substantial and promising results. This article presents a novel classification method that builds on prior work by utilising attention-based feature extraction, RGB channel-based chromatic analysis, Support Vector Machines (SVM) for improved performance, and the ability to integrate with mobile applications and IoT devices after quantization of information. Several disease classification algorithms were compared with the suggested model, and it was discovered that, in terms of accuracy, Vision Transformer-based feature extraction and additional Green Chromatic Coordinate feature with SVM classification achieved an accuracy of (GCCViT-SVM) - 99.69%, whereas after quantization for IoT device integration achieved an accuracy of - 97.41% while almost reducing 4x in size. Our findings have profound implications because they have the potential to transform how farmers identify crop illnesses with precise and fast information, thereby preserving agricultural output and ensuring food security.

{{</citation>}}


## cs.HC (1)



### (118/121) Will Code Remain a Relevant User Interface for End-User Programming with Generative AI Models? (Advait Sarkar, 2023)

{{<citation>}}

Advait Sarkar. (2023)  
**Will Code Remain a Relevant User Interface for End-User Programming with Generative AI Models?**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-PL, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.00382v1)  

---


**ABSTRACT**  
The research field of end-user programming has largely been concerned with helping non-experts learn to code sufficiently well in order to achieve their tasks. Generative AI stands to obviate this entirely by allowing users to generate code from naturalistic language prompts. In this essay, we explore the extent to which "traditional" programming languages remain relevant for non-expert end-user programmers in a world with generative AI. We posit the "generative shift hypothesis": that generative AI will create qualitative and quantitative expansions in the traditional scope of end-user programming. We outline some reasons that traditional programming languages may still be relevant and useful for end-user programmers. We speculate whether each of these reasons might be fundamental and enduring, or whether they may disappear with further improvements and innovations in generative AI. Finally, we articulate a set of implications for end-user programming research, including the possibility of needing to revisit many well-established core concepts, such as Ko's learning barriers and Blackwell's attention investment model.

{{</citation>}}


## cs.GT (1)



### (119/121) Incentivized Collaboration in Active Learning (Lee Cohen et al., 2023)

{{<citation>}}

Lee Cohen, Han Shao. (2023)  
**Incentivized Collaboration in Active Learning**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs-LG, cs.GT  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2311.00260v1)  

---


**ABSTRACT**  
In collaborative active learning, where multiple agents try to learn labels from a common hypothesis, we introduce an innovative framework for incentivized collaboration. Here, rational agents aim to obtain labels for their data sets while keeping label complexity at a minimum. We focus on designing (strict) individually rational (IR) collaboration protocols, ensuring that agents cannot reduce their expected label complexity by acting individually. We first show that given any optimal active learning algorithm, the collaboration protocol that runs the algorithm as is over the entire data is already IR. However, computing the optimal algorithm is NP-hard. We therefore provide collaboration protocols that achieve (strict) IR and are comparable with the best known tractable approximation algorithm in terms of label complexity.

{{</citation>}}


## cs.DC (1)



### (120/121) AMSP: Super-Scaling LLM Training via Advanced Model States Partitioning (Qiaoling Chen et al., 2023)

{{<citation>}}

Qiaoling Chen, Qinghao Hu, Zhisheng Ye, Guoteng Wang, Peng Sun, Yonggang Wen, Tianwei Zhang. (2023)  
**AMSP: Super-Scaling LLM Training via Advanced Model States Partitioning**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.00257v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated impressive performance across various downstream tasks. When training these models, there is a growing inclination to process more tokens on larger training scales but with relatively smaller model sizes. Zero Redundancy Optimizer (ZeRO), although effective in conventional training environments, grapples with scaling challenges when confronted with this emerging paradigm. To this end, we propose a novel LLM training framework AMSP, which undertakes a granular partitioning of model states, encompassing parameters ($P$), gradient ($G$), and optimizer states ($OS$). Specifically, AMSP(1) builds a unified partitioning space, enabling independent partitioning strategies for $P$, $G$, and $OS$; (2) incorporates a scale-aware partitioner to autonomously search for optimal partitioning strategies: (3) designs a dedicated communication optimizer to ensure proficient management of data placement discrepancies arising from diverse partitioning strategies. Our evaluations show that AMSP achieves up to 90.3% scaling efficiency across 1024 GPUs.

{{</citation>}}


## eess.SP (1)



### (121/121) Transformers are Efficient In-Context Estimators for Wireless Communication (Vicram Rajagopalan et al., 2023)

{{<citation>}}

Vicram Rajagopalan, Vishnu Teja Kunde, Chandra Shekhara Kaushik Valmeekam, Krishna Narayanan, Srinivas Shakkottai, Dileep Kalathil, Jean-Francois Chamberland. (2023)  
**Transformers are Efficient In-Context Estimators for Wireless Communication**  

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.00226v1)  

---


**ABSTRACT**  
Pre-trained transformers can perform in-context learning, where they adapt to a new task using only a small number of prompts without any explicit model optimization. Inspired by this attribute, we propose a novel approach, called in-context estimation, for the canonical communication problem of estimating transmitted symbols from received symbols. A communication channel is essentially a noisy function that maps transmitted symbols to received symbols, and this function can be represented by an unknown parameter whose statistics depend on an (also unknown) latent context. Conventional approaches ignore this hierarchical structure and simply attempt to use known transmissions, called pilots, to perform a least-squares estimate of the channel parameter, which is then used to estimate successive, unknown transmitted symbols. We make the basic connection that transformers show excellent contextual sequence completion with a few prompts, and so they should be able to implicitly determine the latent context from pilot symbols to perform end-to-end in-context estimation of transmitted symbols. Furthermore, the transformer should use information efficiently, i.e., it should utilize any pilots received to attain the best possible symbol estimates. Through extensive simulations, we show that in-context estimation not only significantly outperforms standard approaches, but also achieves the same performance as an estimator with perfect knowledge of the latent context within a few context examples. Thus, we make a strong case that transformers are efficient in-context estimators in the communication setting.

{{</citation>}}
