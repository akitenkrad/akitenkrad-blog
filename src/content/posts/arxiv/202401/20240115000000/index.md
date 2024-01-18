---
draft: false
title: "arXiv @ 2024.01.15"
date: 2024-01-15
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.15"
    identifier: arxiv_20240115
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (8)](#cslg-8)
- [cs.CR (2)](#cscr-2)
- [cs.CL (14)](#cscl-14)
- [cs.HC (5)](#cshc-5)
- [cs.NI (2)](#csni-2)
- [cs.AI (3)](#csai-3)
- [cs.NE (1)](#csne-1)
- [cs.CV (11)](#cscv-11)
- [cs.MM (1)](#csmm-1)
- [stat.ML (1)](#statml-1)
- [cs.CY (1)](#cscy-1)
- [quant-ph (2)](#quant-ph-2)
- [cs.SE (1)](#csse-1)
- [eess.IV (1)](#eessiv-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.RO (1)](#csro-1)

## cs.LG (8)



### (1/55) Quantized Side Tuning: Fast and Memory-Efficient Tuning of Quantized Large Language Models (Zhengxin Zhang et al., 2024)

{{<citation>}}

Zhengxin Zhang, Dan Zhao, Xupeng Miao, Gabriele Oliaro, Qing Li, Yong Jiang, Zhihao Jia. (2024)  
**Quantized Side Tuning: Fast and Memory-Efficient Tuning of Quantized Large Language Models**  

---
Primary Category: cs.LG  
Categories: I-2-7, cs-AI, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07159v1)  

---


**ABSTRACT**  
Finetuning large language models (LLMs) has been empirically effective on a variety of downstream tasks. Existing approaches to finetuning an LLM either focus on parameter-efficient finetuning, which only updates a small number of trainable parameters, or attempt to reduce the memory footprint during the training phase of the finetuning. Typically, the memory footprint during finetuning stems from three contributors: model weights, optimizer states, and intermediate activations. However, existing works still require considerable memory and none can simultaneously mitigate memory footprint for all three sources. In this paper, we present Quantized Side Tuing (QST), which enables memory-efficient and fast finetuning of LLMs by operating through a dual-stage process. First, QST quantizes an LLM's model weights into 4-bit to reduce the memory footprint of the LLM's original weights; QST also introduces a side network separated from the LLM, which utilizes the hidden states of the LLM to make task-specific predictions. Using a separate side network avoids performing backpropagation through the LLM, thus reducing the memory requirement of the intermediate activations. Furthermore, QST leverages several low-rank adaptors and gradient-free downsample modules to significantly reduce the trainable parameters, so as to save the memory footprint of the optimizer states. Experiments show that QST can reduce the total memory footprint by up to 2.3 $\times$ and speed up the finetuning process by up to 3 $\times$ while achieving competent performance compared with the state-of-the-art. When it comes to full finetuning, QST can reduce the total memory footprint up to 7 $\times$.

{{</citation>}}


### (2/55) Tensor Graph Convolutional Network for Dynamic Graph Representation Learning (Ling Wang et al., 2024)

{{<citation>}}

Ling Wang, Ye Yuan. (2024)  
**Tensor Graph Convolutional Network for Dynamic Graph Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Convolutional Network, Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.07065v1)  

---


**ABSTRACT**  
Dynamic graphs (DG) describe dynamic interactions between entities in many practical scenarios. Most existing DG representation learning models combine graph convolutional network and sequence neural network, which model spatial-temporal dependencies through two different types of neural networks. However, this hybrid design cannot well capture the spatial-temporal continuity of a DG. In this paper, we propose a tensor graph convolutional network to learn DG representations in one convolution framework based on the tensor product with the following two-fold ideas: a) representing the information of DG by tensor form; b) adopting tensor product to design a tensor graph convolutional network modeling spatial-temporal feature simultaneously. Experiments on real-world DG datasets demonstrate that our model obtains state-of-the-art performance.

{{</citation>}}


### (3/55) Contrastive Learning with Negative Sampling Correction (Lu Wang et al., 2024)

{{<citation>}}

Lu Wang, Chao Du, Pu Zhao, Chuan Luo, Zhangchi Zhu, Bo Qiao, Wei Zhang, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, Qi Zhang. (2024)  
**Contrastive Learning with Negative Sampling Correction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.08690v1)  

---


**ABSTRACT**  
As one of the most effective self-supervised representation learning methods, contrastive learning (CL) relies on multiple negative pairs to contrast against each positive pair. In the standard practice of contrastive learning, data augmentation methods are utilized to generate both positive and negative pairs. While existing works have been focusing on improving the positive sampling, the negative sampling process is often overlooked. In fact, the generated negative samples are often polluted by positive samples, which leads to a biased loss and performance degradation. To correct the negative sampling bias, we propose a novel contrastive learning method named Positive-Unlabeled Contrastive Learning (PUCL). PUCL treats the generated negative samples as unlabeled samples and uses information from positive samples to correct bias in contrastive loss. We prove that the corrected loss used in PUCL only incurs a negligible bias compared to the unbiased contrastive loss. PUCL can be applied to general contrastive learning problems and outperforms state-of-the-art methods on various image and graph classification tasks. The code of PUCL is in the supplementary file.

{{</citation>}}


### (4/55) BP(λ): Online Learning via Synthetic Gradients (Joseph Pemberton et al., 2024)

{{<citation>}}

Joseph Pemberton, Rui Ponte Costa. (2024)  
**BP(λ): Online Learning via Synthetic Gradients**  

---
Primary Category: cs.LG  
Categories: 68T07, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.07044v1)  

---


**ABSTRACT**  
Training recurrent neural networks typically relies on backpropagation through time (BPTT). BPTT depends on forward and backward passes to be completed, rendering the network locked to these computations before loss gradients are available. Recently, Jaderberg et al. proposed synthetic gradients to alleviate the need for full BPTT. In their implementation synthetic gradients are learned through a mixture of backpropagated gradients and bootstrapped synthetic gradients, analogous to the temporal difference (TD) algorithm in Reinforcement Learning (RL). However, as in TD learning, heavy use of bootstrapping can result in bias which leads to poor synthetic gradient estimates. Inspired by the accumulate $\mathrm{TD}(\lambda)$ in RL, we propose a fully online method for learning synthetic gradients which avoids the use of BPTT altogether: accumulate $BP(\lambda)$. As in accumulate $\mathrm{TD}(\lambda)$, we show analytically that accumulate $\mathrm{BP}(\lambda)$ can control the level of bias by using a mixture of temporal difference errors and recursively defined eligibility traces. We next demonstrate empirically that our model outperforms the original implementation for learning synthetic gradients in a variety of tasks, and is particularly suited for capturing longer timescales. Finally, building on recent work we reflect on accumulate $\mathrm{BP}(\lambda)$ as a principle for learning in biological circuits. In summary, inspired by RL principles we introduce an algorithm capable of bias-free online learning via synthetic gradients.

{{</citation>}}


### (5/55) Edge-Enabled Anomaly Detection and Information Completion for Social Network Knowledge Graphs (Fan Lu et al., 2024)

{{<citation>}}

Fan Lu, Quan Qi, Huaibin Qin. (2024)  
**Edge-Enabled Anomaly Detection and Information Completion for Social Network Knowledge Graphs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Anomaly Detection, Knowledge Graph, QA, Social Network  
[Paper Link](http://arxiv.org/abs/2401.07022v1)  

---


**ABSTRACT**  
In the rapidly advancing information era, various human behaviors are being precisely recorded in the form of data, including identity information, criminal records, and communication data. Law enforcement agencies can effectively maintain social security and precisely combat criminal activities by analyzing the aforementioned data. In comparison to traditional data analysis methods, deep learning models, relying on the robust computational power in cloud centers, exhibit higher accuracy in extracting data features and inferring data. However, within the architecture of cloud centers, the transmission of data from end devices introduces significant latency, hindering real-time inference of data. Furthermore, low-latency edge computing architectures face limitations in direct deployment due to relatively weak computing and storage capacities of nodes. To address these challenges, a lightweight distributed knowledge graph completion architecture is proposed. Firstly, we introduce a lightweight distributed knowledge graph completion architecture that utilizes knowledge graph embedding for data analysis. Subsequently, to filter out substandard data, a personnel data quality assessment method named PDQA is proposed. Lastly, we present a model pruning algorithm that significantly reduces the model size while maximizing performance, enabling lightweight deployment. In experiments, we compare the effects of 11 advanced models on completing the knowledge graph of public security personnel information. The results indicate that the RotatE model outperforms other models significantly in knowledge graph completion, with the pruned model size reduced by 70\%, and hits@10 reaching 86.97\%.}

{{</citation>}}


### (6/55) TemporalAugmenter: An Ensemble Recurrent Based Deep Learning Approach for Signal Classification (Nelly Elsayed et al., 2024)

{{<citation>}}

Nelly Elsayed, Constantinos L. Zekios, Navid Asadizanjani, Zag ElSayed. (2024)  
**TemporalAugmenter: An Ensemble Recurrent Based Deep Learning Approach for Signal Classification**  

---
Primary Category: cs.LG  
Categories: cs-HC, cs-LG, cs.LG, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06970v1)  

---


**ABSTRACT**  
Ensemble modeling has been widely used to solve complex problems as it helps to improve overall performance and generalization. In this paper, we propose a novel TemporalAugmenter approach based on ensemble modeling for augmenting the temporal information capturing for long-term and short-term dependencies in data integration of two variations of recurrent neural networks in two learning streams to obtain the maximum possible temporal extraction. Thus, the proposed model augments the extraction of temporal dependencies. In addition, the proposed approach reduces the preprocessing and prior stages of feature extraction, which reduces the required energy to process the models built upon the proposed TemporalAugmenter approach, contributing towards green AI. Moreover, the proposed model can be simply integrated into various domains including industrial, medical, and human-computer interaction applications. Our proposed approach empirically evaluated the speech emotion recognition, electrocardiogram signal, and signal quality examination tasks as three different signals with varying complexity and different temporal dependency features.

{{</citation>}}


### (7/55) Reinforcement Learning for Scalable Train Timetable Rescheduling with Graph Representation (Peng Yue et al., 2024)

{{<citation>}}

Peng Yue, Yaochu Jin, Xuewu Dai, Zhenhua Feng, Dongliang Cui. (2024)  
**Reinforcement Learning for Scalable Train Timetable Rescheduling with Graph Representation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.06952v1)  

---


**ABSTRACT**  
Train timetable rescheduling (TTR) aims to promptly restore the original operation of trains after unexpected disturbances or disruptions. Currently, this work is still done manually by train dispatchers, which is challenging to maintain performance under various problem instances. To mitigate this issue, this study proposes a reinforcement learning-based approach to TTR, which makes the following contributions compared to existing work. First, we design a simple directed graph to represent the TTR problem, enabling the automatic extraction of informative states through graph neural networks. Second, we reformulate the construction process of TTR's solution, not only decoupling the decision model from the problem size but also ensuring the generated scheme's feasibility. Third, we design a learning curriculum for our model to handle the scenarios with different levels of delay. Finally, a simple local search method is proposed to assist the learned decision model, which can significantly improve solution quality with little additional computation cost, further enhancing the practical value of our method. Extensive experimental results demonstrate the effectiveness of our method. The learned decision model can achieve better performance for various problems with varying degrees of train delay and different scales when compared to handcrafted rules and state-of-the-art solvers.

{{</citation>}}


### (8/55) Accelerated Sampling of Rare Events using a Neural Network Bias Potential (Xinru Hua et al., 2024)

{{<citation>}}

Xinru Hua, Rasool Ahmad, Jose Blanchet, Wei Cai. (2024)  
**Accelerated Sampling of Rare Events using a Neural Network Bias Potential**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-comp-ph  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.06936v1)  

---


**ABSTRACT**  
In the field of computational physics and material science, the efficient sampling of rare events occurring at atomic scale is crucial. It aids in understanding mechanisms behind a wide range of important phenomena, including protein folding, conformal changes, chemical reactions and materials diffusion and deformation. Traditional simulation methods, such as Molecular Dynamics and Monte Carlo, often prove inefficient in capturing the timescale of these rare events by brute force. In this paper, we introduce a practical approach by combining the idea of importance sampling with deep neural networks (DNNs) that enhance the sampling of these rare events. In particular, we approximate the variance-free bias potential function with DNNs which is trained to maximize the probability of rare event transition under the importance potential function. This method is easily scalable to high-dimensional problems and provides robust statistical guarantees on the accuracy of the estimated probability of rare event transition. Furthermore, our algorithm can actively generate and learn from any successful samples, which is a novel improvement over existing methods. Using a 2D system as a test bed, we provide comparisons between results obtained from different training strategies, traditional Monte Carlo sampling and numerically solved optimal bias potential function under different temperatures. Our numerical results demonstrate the efficacy of the DNN-based importance sampling of rare events.

{{</citation>}}


## cs.CR (2)



### (9/55) Discovering Command and Control Channels Using Reinforcement Learning (Cheng Wang et al., 2024)

{{<citation>}}

Cheng Wang, Akshay Kakkar, Christopher Redino, Abdul Rahman, Ajinsyam S, Ryan Clark, Daniel Radke, Tyler Cody, Lanxiao Huang, Edward Bowen. (2024)  
**Discovering Command and Control Channels Using Reinforcement Learning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.07154v1)  

---


**ABSTRACT**  
Command and control (C2) paths for issuing commands to malware are sometimes the only indicators of its existence within networks. Identifying potential C2 channels is often a manually driven process that involves a deep understanding of cyber tradecraft. Efforts to improve discovery of these channels through using a reinforcement learning (RL) based approach that learns to automatically carry out C2 attack campaigns on large networks, where multiple defense layers are in place serves to drive efficiency for network operators. In this paper, we model C2 traffic flow as a three-stage process and formulate it as a Markov decision process (MDP) with the objective to maximize the number of valuable hosts whose data is exfiltrated. The approach also specifically models payload and defense mechanisms such as firewalls which is a novel contribution. The attack paths learned by the RL agent can in turn help the blue team identify high-priority vulnerabilities and develop improved defense strategies. The method is evaluated on a large network with more than a thousand hosts and the results demonstrate that the agent can effectively learn attack paths while avoiding firewalls.

{{</citation>}}


### (10/55) Code Security Vulnerability Repair Using Reinforcement Learning with Large Language Models (Nafis Tanveer Islam et al., 2024)

{{<citation>}}

Nafis Tanveer Islam, Peyman Najafirad. (2024)  
**Code Security Vulnerability Repair Using Reinforcement Learning with Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-SE, cs.CR  
Keywords: Language Model, Reinforcement Learning, Security  
[Paper Link](http://arxiv.org/abs/2401.07031v1)  

---


**ABSTRACT**  
With the recent advancement of Large Language Models (LLMs), generating functionally correct code has become less complicated for a wide array of developers. While using LLMs has sped up the functional development process, it poses a heavy risk to code security. Code generation with proper security measures using LLM is a significantly more challenging task than functional code generation. Security measures may include adding a pair of lines of code with the original code, consisting of null pointer checking or prepared statements for SQL injection prevention. Currently, available code repair LLMs generate code repair by supervised fine-tuning, where the model looks at cross-entropy loss. However, the original and repaired codes are mostly similar in functionality and syntactically, except for a few (1-2) lines, which act as security measures. This imbalance between the lines needed for security measures and the functional code enforces the supervised fine-tuned model to prioritize generating functional code without adding proper security measures, which also benefits the model by resulting in minimal loss. Therefore, in this work, for security hardening and strengthening of generated code from LLMs, we propose a reinforcement learning-based method for program-specific repair with the combination of semantic and syntactic reward mechanisms that focus heavily on adding security and functional measures in the code, respectively.

{{</citation>}}


## cs.CL (14)



### (11/55) EHRAgent: Code Empowers Large Language Models for Complex Tabular Reasoning on Electronic Health Records (Wenqi Shi et al., 2024)

{{<citation>}}

Wenqi Shi, Ran Xu, Yuchen Zhuang, Yue Yu, Jieyu Zhang, Hang Wu, Yuanda Zhu, Joyce Ho, Carl Yang, May D. Wang. (2024)  
**EHRAgent: Code Empowers Large Language Models for Complex Tabular Reasoning on Electronic Health Records**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.07128v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated exceptional capabilities in planning and tool utilization as autonomous agents, but few have been developed for medical problem-solving. We propose EHRAgent1, an LLM agent empowered with a code interface, to autonomously generate and execute code for complex clinical tasks within electronic health records (EHRs). First, we formulate an EHR question-answering task into a tool-use planning process, efficiently decomposing a complicated task into a sequence of manageable actions. By integrating interactive coding and execution feedback, EHRAgent learns from error messages and improves the originally generated code through iterations. Furthermore, we enhance the LLM agent by incorporating long-term memory, which allows EHRAgent to effectively select and build upon the most relevant successful cases from past experiences. Experiments on two real-world EHR datasets show that EHRAgent outperforms the strongest LLM agent baseline by 36.48% and 12.41%, respectively. EHRAgent leverages the emerging few-shot learning capabilities of LLMs, enabling autonomous code generation and execution to tackle complex clinical tasks with minimal demonstrations.

{{</citation>}}


### (12/55) Combining Confidence Elicitation and Sample-based Methods for Uncertainty Quantification in Misinformation Mitigation (Mauricio Rivera et al., 2024)

{{<citation>}}

Mauricio Rivera, Jean-François Godbout, Reihaneh Rabbany, Kellin Pelrine. (2024)  
**Combining Confidence Elicitation and Sample-based Methods for Uncertainty Quantification in Misinformation Mitigation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.08694v1)  

---


**ABSTRACT**  
Large Language Models have emerged as prime candidates to tackle misinformation mitigation. However, existing approaches struggle with hallucinations and overconfident predictions. We propose an uncertainty quantification framework that leverages both direct confidence elicitation and sampled-based consistency methods to provide better calibration for NLP misinformation mitigation solutions. We first investigate the calibration of sample-based consistency methods that exploit distinct features of consistency across sample sizes and stochastic levels. Next, we evaluate the performance and distributional shift of a robust numeric verbalization prompt across single vs. two-step confidence elicitation procedure. We also compare the performance of the same prompt with different versions of GPT and different numerical scales. Finally, we combine the sample-based consistency and verbalized methods to propose a hybrid framework that yields a better uncertainty estimation for GPT models. Overall, our work proposes novel uncertainty quantification methods that will improve the reliability of Large Language Models in misinformation mitigation applications.

{{</citation>}}


### (13/55) Graph Language Models (Moritz Plenz et al., 2024)

{{<citation>}}

Moritz Plenz, Anette Frank. (2024)  
**Graph Language Models**  

---
Primary Category: cs.CL  
Categories: I-2-0; I-2-4; I-2-7, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GLM, GNN, Graph Neural Network, Graph Neural Networks, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.07105v1)  

---


**ABSTRACT**  
While Language Models have become workhorses for NLP, their interplay with textual knowledge graphs (KGs) - structured memories of general or domain knowledge - is actively researched. Current embedding methodologies for such graphs typically either (i) linearize graphs for embedding them using sequential Language Models (LMs), which underutilize structural information, or (ii) use Graph Neural Networks (GNNs) to preserve graph structure, while GNNs cannot represent textual features as well as a pre-trained LM could. In this work we introduce a novel language model, the Graph Language Model (GLM), that integrates the strengths of both approaches, while mitigating their weaknesses. The GLM parameters are initialized from a pretrained LM, to facilitate nuanced understanding of individual concepts and triplets. Simultaneously, its architectural design incorporates graph biases, thereby promoting effective knowledge distribution within the graph. Empirical evaluations on relation classification tasks on ConceptNet subgraphs reveal that GLM embeddings surpass both LM- and GNN-based baselines in supervised and zero-shot settings.

{{</citation>}}


### (14/55) Leveraging Large Language Models for NLG Evaluation: A Survey (Zhen Li et al., 2024)

{{<citation>}}

Zhen Li, Xiaohan Xu, Tao Shen, Can Xu, Jia-Chen Gu, Chongyang Tao. (2024)  
**Leveraging Large Language Models for NLG Evaluation: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2401.07103v1)  

---


**ABSTRACT**  
In the rapidly evolving domain of Natural Language Generation (NLG) evaluation, introducing Large Language Models (LLMs) has opened new avenues for assessing generated content quality, e.g., coherence, creativity, and context relevance. This survey aims to provide a thorough overview of leveraging LLMs for NLG evaluation, a burgeoning area that lacks a systematic analysis. We propose a coherent taxonomy for organizing existing LLM-based evaluation metrics, offering a structured framework to understand and compare these methods. Our detailed exploration includes critically assessing various LLM-based methodologies, as well as comparing their strengths and limitations in evaluating NLG outputs. By discussing unresolved challenges, including bias, robustness, domain-specificity, and unified evaluation, this survey seeks to offer insights to researchers and advocate for fairer and more advanced NLG evaluation techniques.

{{</citation>}}


### (15/55) A Novel Multi-Stage Prompting Approach for Language Agnostic MCQ Generation using GPT (Subhankar Maity et al., 2024)

{{<citation>}}

Subhankar Maity, Aniket Deroy, Sudeshna Sarkar. (2024)  
**A Novel Multi-Stage Prompting Approach for Language Agnostic MCQ Generation using GPT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, NLP  
[Paper Link](http://arxiv.org/abs/2401.07098v1)  

---


**ABSTRACT**  
We introduce a multi-stage prompting approach (MSP) for the generation of multiple choice questions (MCQs), harnessing the capabilities of GPT models such as text-davinci-003 and GPT-4, renowned for their excellence across various NLP tasks. Our approach incorporates the innovative concept of chain-of-thought prompting, a progressive technique in which the GPT model is provided with a series of interconnected cues to guide the MCQ generation process. Automated evaluations consistently demonstrate the superiority of our proposed MSP method over the traditional single-stage prompting (SSP) baseline, resulting in the production of high-quality distractors. Furthermore, the one-shot MSP technique enhances automatic evaluation results, contributing to improved distractor generation in multiple languages, including English, German, Bengali, and Hindi. In human evaluations, questions generated using our approach exhibit superior levels of grammaticality, answerability, and difficulty, highlighting its efficacy in various languages.

{{</citation>}}


### (16/55) PUB: A Pragmatics Understanding Benchmark for Assessing LLMs' Pragmatics Capabilities (Settaluri Lakshmi Sravanthi et al., 2024)

{{<citation>}}

Settaluri Lakshmi Sravanthi, Meet Doshi, Tankala Pavan Kalyan, Rudra Murthy, Pushpak Bhattacharyya, Raj Dabre. (2024)  
**PUB: A Pragmatics Understanding Benchmark for Assessing LLMs' Pragmatics Capabilities**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.07078v1)  

---


**ABSTRACT**  
LLMs have demonstrated remarkable capability for understanding semantics, but they often struggle with understanding pragmatics. To demonstrate this fact, we release a Pragmatics Understanding Benchmark (PUB) dataset consisting of fourteen tasks in four pragmatics phenomena, namely, Implicature, Presupposition, Reference, and Deixis. We curated high-quality test sets for each task, consisting of Multiple Choice Question Answers (MCQA). PUB includes a total of 28k data points, 6.1k of which have been created by us, and the rest are adapted from existing datasets. We evaluated nine models varying in the number of parameters and type of training. Our study indicates that fine-tuning for instruction-following and chat significantly enhances the pragmatics capabilities of smaller language models. However, for larger models, the base versions perform comparably with their chat-adapted counterparts. Additionally, there is a noticeable performance gap between human capabilities and model capabilities. Furthermore, unlike the consistent performance of humans across various tasks, the models demonstrate variability in their proficiency, with performance levels fluctuating due to different hints and the complexities of tasks within the same dataset. Overall, the benchmark aims to provide a comprehensive evaluation of LLM's ability to handle real-world language tasks that require pragmatic reasoning.

{{</citation>}}


### (17/55) xCoT: Cross-lingual Instruction Tuning for Cross-lingual Chain-of-Thought Reasoning (Linzheng Chai et al., 2024)

{{<citation>}}

Linzheng Chai, Jian Yang, Tao Sun, Hongcheng Guo, Jiaheng Liu, Bing Wang, Xiannian Liang, Jiaqi Bai, Tongliang Li, Qiyao Peng, Zhoujun Li. (2024)  
**xCoT: Cross-lingual Instruction Tuning for Cross-lingual Chain-of-Thought Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.07037v1)  

---


**ABSTRACT**  
Chain-of-thought (CoT) has emerged as a powerful technique to elicit reasoning in large language models and improve a variety of downstream tasks. CoT mainly demonstrates excellent performance in English, but its usage in low-resource languages is constrained due to poor language generalization. To bridge the gap among different languages, we propose a cross-lingual instruction fine-tuning framework (xCOT) to transfer knowledge from high-resource languages to low-resource languages. Specifically, the multilingual instruction training data (xCOT-INSTRUCT) is created to encourage the semantic alignment of multiple languages. We introduce cross-lingual in-context few-shot learning (xICL)) to accelerate multilingual agreement in instruction tuning, where some fragments of source languages in examples are randomly substituted by their counterpart translations of target languages. During multilingual instruction tuning, we adopt the randomly online CoT strategy to enhance the multilingual reasoning ability of the large language model by first translating the query to another language and then answering in English. To further facilitate the language transfer, we leverage the high-resource CoT to supervise the training of low-resource languages with cross-lingual distillation. Experimental results on previous benchmarks demonstrate the superior performance of xCoT in reducing the gap among different languages, highlighting its potential to reduce the cross-lingual gap.

{{</citation>}}


### (18/55) Knowledge Distillation for Closed-Source Language Models (Hongzhan Chen et al., 2024)

{{<citation>}}

Hongzhan Chen, Xiaojun Quan, Hehong Chen, Ming Yan, Ji Zhang. (2024)  
**Knowledge Distillation for Closed-Source Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Knowledge Distillation, Language Model  
[Paper Link](http://arxiv.org/abs/2401.07013v1)  

---


**ABSTRACT**  
Closed-source language models such as GPT-4 have achieved remarkable performance. Many recent studies focus on enhancing the capabilities of smaller models through knowledge distillation from closed-source language models. However, due to the incapability to directly access the weights, hidden states, and output distributions of these closed-source models, the distillation can only be performed by fine-tuning smaller models with data samples generated by closed-source language models, which constrains the effectiveness of knowledge distillation. In this paper, we propose to estimate the output distributions of closed-source language models within a Bayesian estimation framework, involving both prior and posterior estimation. The prior estimation aims to derive a prior distribution by utilizing the corpus generated by closed-source language models, while the posterior estimation employs a proxy model to update the prior distribution and derive a posterior distribution. By leveraging the estimated output distribution of closed-source language models, traditional knowledge distillation can be executed. Experimental results demonstrate that our method surpasses the performance of current models directly fine-tuned on data generated by closed-source language models.

{{</citation>}}


### (19/55) Extending LLMs' Context Window with 100 Samples (Yikai Zhang et al., 2024)

{{<citation>}}

Yikai Zhang, Junlong Li, Pengfei Liu. (2024)  
**Extending LLMs' Context Window with 100 Samples**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, LLaMA, Language Model, NLP, PaLM  
[Paper Link](http://arxiv.org/abs/2401.07004v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are known to have limited extrapolation ability beyond their pre-trained context window, constraining their application in downstream tasks with lengthy inputs. Recent studies have sought to extend LLMs' context window by modifying rotary position embedding (RoPE), a popular position encoding method adopted by well-known LLMs such as LLaMA, PaLM, and GPT-NeoX. However, prior works like Position Interpolation (PI) and YaRN are resource-intensive and lack comparative experiments to assess their applicability. In this work, we identify the inherent need for LLMs' attention entropy (i.e. the information entropy of attention scores) to maintain stability and introduce a novel extension to RoPE which combines adjusting RoPE's base frequency and scaling the attention logits to help LLMs efficiently adapt to a larger context window. We validate the superiority of our method in both fine-tuning performance and robustness across different context window sizes on various context-demanding tasks. Notably, our method extends the context window of LLaMA-2-7B-Chat to 16,384 with only 100 samples and 6 training steps, showcasing extraordinary efficiency. Finally, we also explore how data compositions and training curricula affect context window extension for specific downstream tasks, suggesting fine-tuning LLMs with lengthy conversations as a good starting point. We release our code and SFT data at https://github.com/GAIR-NLP/Entropy-ABF.

{{</citation>}}


### (20/55) Joint Unsupervised and Supervised Training for Automatic Speech Recognition via Bilevel Optimization (A F M Saif et al., 2024)

{{<citation>}}

A F M Saif, Xiaodong Cui, Han Shen, Songtao Lu, Brian Kingsbury, Tianyi Chen. (2024)  
**Joint Unsupervised and Supervised Training for Automatic Speech Recognition via Bilevel Optimization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.06980v1)  

---


**ABSTRACT**  
In this paper, we present a novel bilevel optimization-based training approach to training acoustic models for automatic speech recognition (ASR) tasks that we term {bi-level joint unsupervised and supervised training (BL-JUST)}. {BL-JUST employs a lower and upper level optimization with an unsupervised loss and a supervised loss respectively, leveraging recent advances in penalty-based bilevel optimization to solve this challenging ASR problem with affordable complexity and rigorous convergence guarantees.} To evaluate BL-JUST, extensive experiments on the LibriSpeech and TED-LIUM v2 datasets have been conducted. BL-JUST achieves superior performance over the commonly used pre-training followed by fine-tuning strategy.

{{</citation>}}


### (21/55) CHAMP: A Competition-level Dataset for Fine-Grained Analyses of LLMs' Mathematical Reasoning Capabilities (Yujun Mao et al., 2024)

{{<citation>}}

Yujun Mao, Yoon Kim, Yilun Zhou. (2024)  
**CHAMP: A Competition-level Dataset for Fine-Grained Analyses of LLMs' Mathematical Reasoning Capabilities**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.06961v1)  

---


**ABSTRACT**  
Recent large language models (LLMs) have shown indications of mathematical reasoning ability. However it has not been clear how they would fare on more challenging competition-level problems. And while self-generated verbalizations of intermediate reasoning steps (i.e., chain-of-thought prompting) have been shown to be helpful, whether LLMs can make use of helpful side information such as problem-specific hints has not been investigated before. In this paper, we propose a challenging benchmark dataset for enabling such analyses. The Concept and Hint-Annotated Math Problems (CHAMP) consists of high school math competition problems, annotated with concepts, or general math facts, and hints, or problem-specific tricks. These annotations allow us to explore the effects of additional information, such as relevant hints, misleading concepts, or related problems. This benchmark is difficult, with the best model only scoring 58.1% in standard settings. With concepts and hints, performance sometimes improves, indicating that some models can make use of such side information. We further annotate model-generated solutions for their correctness. Using this corpus, we find that models often arrive at the correct final answer through wrong reasoning steps. In addition, we test whether models are able to verify these solutions, and find that most models struggle. The dataset and code are available on the project website.

{{</citation>}}


### (22/55) Bridging the Preference Gap between Retrievers and LLMs (Zixuan Ke et al., 2024)

{{<citation>}}

Zixuan Ke, Weize Kong, Cheng Li, Mingyang Zhang, Qiaozhu Mei, Michael Bendersky. (2024)  
**Bridging the Preference Gap between Retrievers and LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06954v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated superior results across a wide range of tasks, while retrieval has long been established as an effective means of obtaining task-relevant information for humans. Retrieval-augmented Generation (RAG) are known for their effectiveness in knowledge-intensive tasks by locating relevant information and placing it within the context window of the LLM. However, the relationship between retrievers and LLMs is still under-investigated. Most existing work treats the retriever and the LLM as independent components and leaves a gap between retrieving human-friendly information and assembling a LLM-friendly context. In this work, we examine a novel bridge model, validate the ranking and selection assumptions in retrievers in the context of RAG, and propose a training framework that chains together supervised and reinforcement learning to learn a bridge model. Empirical results demonstrate the effectiveness of our method in both question-answering and personalized generation tasks.

{{</citation>}}


### (23/55) E^2-LLM: Efficient and Extreme Length Extension of Large Language Models (Jiaheng Liu et al., 2024)

{{<citation>}}

Jiaheng Liu, Zhiqi Bai, Yuanxing Zhang, Chenchen Zhang, Yu Zhang, Ge Zhang, Jiakai Wang, Haoran Que, Yukang Chen, Wenbo Su, Tiezheng Ge, Jie Fu, Wenhu Chen, Bo Zheng. (2024)  
**E^2-LLM: Efficient and Extreme Length Extension of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06951v1)  

---


**ABSTRACT**  
Typically, training LLMs with long context sizes is computationally expensive, requiring extensive training hours and GPU resources. Existing long-context extension methods usually need additional training procedures to support corresponding long-context windows, where the long-context training data (e.g., 32k) is needed, and high GPU training costs are assumed. To address the aforementioned issues, we propose an Efficient and Extreme length extension method for Large Language Models, called E 2 -LLM, with only one training procedure and dramatically reduced computation cost, which also removes the need to collect long-context data. Concretely, first, the training data of our E 2 -LLM only requires a short length (e.g., 4k), which reduces the tuning cost greatly. Second, the training procedure on the short training context window is performed only once time, and we can support different evaluation context windows at inference. Third, in E 2 - LLM, based on RoPE position embeddings, we introduce two different augmentation methods on the scale and position index parameters for different samples in training. It aims to make the model more robust to the different relative differences when directly interpolating the arbitrary context length at inference. Comprehensive experimental results on multiple benchmark datasets demonstrate the effectiveness of our E 2 -LLM on challenging long-context tasks.

{{</citation>}}


### (24/55) Knowledge-Centric Templatic Views of Documents (Isabel Cachola et al., 2024)

{{<citation>}}

Isabel Cachola, Silviu Cucerzan, Allen Herring, Vuksan Mijovic, Erik Oveson, Sujay Kumar Jauhar. (2024)  
**Knowledge-Centric Templatic Views of Documents**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.06945v1)  

---


**ABSTRACT**  
Authors seeking to communicate with broader audiences often compose their ideas about the same underlying knowledge in different documents and formats -- for example, as slide decks, newsletters, reports, brochures, etc. Prior work in document generation has generally considered the creation of each separate format to be different a task, developing independent methods for generation and evaluation. This approach is suboptimal for the advancement of AI-supported content authoring from both research and application perspectives because it leads to fragmented learning processes, redundancy in models and methods, and disjointed evaluation. Thus, in our work, we consider each of these documents to be templatic views of the same underlying knowledge, and we aim to unify the generation and evaluation of these templatic views of documents. We begin by introducing an LLM-powered method to extract the most important information from an input document and represent this information in a structured format. We show that this unified representation can be used to generate multiple templatic views with no supervision and with very little guidance, improving over strong baselines. We additionally introduce a unified evaluation method that is template agnostic, and can be adapted to building document generators for heterogeneous downstream applications. Finally, we conduct a human evaluation, which shows that humans prefer 82% of the downstream documents generated with our method. Furthermore, the newly proposed evaluation metric correlates more highly with human judgement than prior metrics, while providing a unified evaluation method.

{{</citation>}}


## cs.HC (5)



### (25/55) One Agent Too Many: User Perspectives on Approaches to Multi-agent Conversational AI (Christopher Clarke et al., 2024)

{{<citation>}}

Christopher Clarke, Karthik Krishnamurthy, Walter Talamonti, Yiping Kang, Lingjia Tang, Jason Mars. (2024)  
**One Agent Too Many: User Perspectives on Approaches to Multi-agent Conversational AI**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-HC, cs.HC  
Keywords: AI, Amazon  
[Paper Link](http://arxiv.org/abs/2401.07123v1)  

---


**ABSTRACT**  
Conversational agents have been gaining increasing popularity in recent years. Influenced by the widespread adoption of task-oriented agents such as Apple Siri and Amazon Alexa, these agents are being deployed into various applications to enhance user experience. Although these agents promote "ask me anything" functionality, they are typically built to focus on a single or finite set of expertise. Given that complex tasks often require more than one expertise, this results in the users needing to learn and adopt multiple agents. One approach to alleviate this is to abstract the orchestration of agents in the background. However, this removes the option of choice and flexibility, potentially harming the ability to complete tasks. In this paper, we explore these different interaction experiences (one agent for all) vs (user choice of agents) for conversational AI. We design prototypes for each, systematically evaluating their ability to facilitate task completion. Through a series of conducted user studies, we show that users have a significant preference for abstracting agent orchestration in both system usability and system performance. Additionally, we demonstrate that this mode of interaction is able to provide quality responses that are rated within 1% of human-selected answers.

{{</citation>}}


### (26/55) Exploring of Discrete and Continuous Input Control for AI-enhanced Assistive Robotic Arms (Max Pascher et al., 2024)

{{<citation>}}

Max Pascher, Kevin Zinta, Jens Gerken. (2024)  
**Exploring of Discrete and Continuous Input Control for AI-enhanced Assistive Robotic Arms**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-RO, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07118v1)  

---


**ABSTRACT**  
Robotic arms, integral in domestic care for individuals with motor impairments, enable them to perform Activities of Daily Living (ADLs) independently, reducing dependence on human caregivers. These collaborative robots require users to manage multiple Degrees-of-Freedom (DoFs) for tasks like grasping and manipulating objects. Conventional input devices, typically limited to two DoFs, necessitate frequent and complex mode switches to control individual DoFs. Modern adaptive controls with feed-forward multi-modal feedback reduce the overall task completion time, number of mode switches, and cognitive load. Despite the variety of input devices available, their effectiveness in adaptive settings with assistive robotics has yet to be thoroughly assessed. This study explores three different input devices by integrating them into an established XR framework for assistive robotics, evaluating them and providing empirical insights through a preliminary study for future developments.

{{</citation>}}


### (27/55) Does More Advice Help? The Effects of Second Opinions in AI-Assisted Decision Making (Zhuoran Lu et al., 2024)

{{<citation>}}

Zhuoran Lu, Dakuo Wang, Ming Yin. (2024)  
**Does More Advice Help? The Effects of Second Opinions in AI-Assisted Decision Making**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07058v1)  

---


**ABSTRACT**  
AI assistance in decision-making has become popular, yet people's inappropriate reliance on AI often leads to unsatisfactory human-AI collaboration performance. In this paper, through three pre-registered, randomized human subject experiments, we explore whether and how the provision of {second opinions} may affect decision-makers' behavior and performance in AI-assisted decision-making. We find that if both the AI model's decision recommendation and a second opinion are always presented together, decision-makers reduce their over-reliance on AI while increase their under-reliance on AI, regardless whether the second opinion is generated by a peer or another AI model. However, if decision-makers have the control to decide when to solicit a peer's second opinion, we find that their active solicitations of second opinions have the potential to mitigate over-reliance on AI without inducing increased under-reliance in some cases. We conclude by discussing the implications of our findings for promoting effective human-AI collaborations in decision-making.

{{</citation>}}


### (28/55) Risk-aware Adaptive Virtual CPU Oversubscription in Microsoft Cloud via Prototypical Human-in-the-loop Imitation Learning (Lu Wang et al., 2024)

{{<citation>}}

Lu Wang, Mayukh Das, Fangkai Yang, Junjie Sheng, Bo Qiao, Hang Dong, Si Qin, Victor Rühle, Chetan Bansal, Eli Cortez, Íñigo Goiri, Saravan Rajmohan, Qingwei Lin, Dongmei Zhang. (2024)  
**Risk-aware Adaptive Virtual CPU Oversubscription in Microsoft Cloud via Prototypical Human-in-the-loop Imitation Learning**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Microsoft  
[Paper Link](http://arxiv.org/abs/2401.07033v1)  

---


**ABSTRACT**  
Oversubscription is a prevalent practice in cloud services where the system offers more virtual resources, such as virtual cores in virtual machines, to users or applications than its available physical capacity for reducing revenue loss due to unused/redundant capacity. While oversubscription can potentially lead to significant enhancement in efficient resource utilization, the caveat is that it comes with the risks of overloading and introducing jitter at the level of physical nodes if all the co-located virtual machines have high utilization. Thus suitable oversubscription policies which maximize utilization while mitigating risks are paramount for cost-effective seamless cloud experiences. Most cloud platforms presently rely on static heuristics-driven decisions about oversubscription activation and limits, which either leads to overloading or stranded resources. Designing an intelligent oversubscription policy that can adapt to resource utilization patterns and jointly optimizes benefits and risks is, largely, an unsolved problem. We address this challenge with our proposed novel HuMan-in-the-loop Protoypical Imitation Learning (ProtoHAIL) framework that exploits approximate symmetries in utilization patterns to learn suitable policies. Also, our human-in-the-loop (knowledge-infused) training allows for learning safer policies that are robust to noise and sparsity. Our empirical investigations on real data show orders of magnitude reduction in risk and significant increase in benefits (saving stranded cores) in Microsoft cloud platform for 1st party (internal services).

{{</citation>}}


### (29/55) Apple Vision Pro: Comments in Healthcare (Ezequiel Santos et al., 2024)

{{<citation>}}

Ezequiel Santos, Vanessa Castillo. (2024)  
**Apple Vision Pro: Comments in Healthcare**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.08685v1)  

---


**ABSTRACT**  
This paper objectively analyzes the emerging discourse surrounding Apple Vision Pro's application in healthcare and medical education. Released in June 2023, Apple Vision Pro represents a significant advancement in spatial computing, combining augmented and virtual reality to create new possibilities in digital interaction. We aim to compile and present recent articles. We used PubMed, IEEE Xplore, Google Scholar, and JSTOR. Non-academic publications were excluded. The results were six commentaries, one a pre-print. All were majorly optimistic, with one mentioning VR/AR sickness. For future research directions, we stress the need for continued exploration of Apple Vision Pro's capabilities and limitations and expect expert opinions to englobe this discussion.

{{</citation>}}


## cs.NI (2)



### (30/55) Generative AI-enabled Quantum Computing Networks and Intelligent Resource Allocation (Minrui Xu et al., 2024)

{{<citation>}}

Minrui Xu, Dusit Niyato, Jiawen Kang, Zehui Xiong, Yuan Cao, Yulan Gao, Chao Ren, Han Yu. (2024)  
**Generative AI-enabled Quantum Computing Networks and Intelligent Resource Allocation**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP, quant-ph  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.07120v1)  

---


**ABSTRACT**  
Quantum computing networks enable scalable collaboration and secure information exchange among multiple classical and quantum computing nodes while executing large-scale generative AI computation tasks and advanced quantum algorithms. Quantum computing networks overcome limitations such as the number of qubits and coherence time of entangled pairs and offer advantages for generative AI infrastructure, including enhanced noise reduction through distributed processing and improved scalability by connecting multiple quantum devices. However, efficient resource allocation in quantum computing networks is a critical challenge due to factors including qubit variability and network complexity. In this article, we propose an intelligent resource allocation framework for quantum computing networks to improve network scalability with minimized resource costs. To achieve scalability in quantum computing networks, we formulate the resource allocation problem as stochastic programming, accounting for the uncertain fidelities of qubits and entangled pairs. Furthermore, we introduce state-of-the-art reinforcement learning (RL) algorithms, from generative learning to quantum machine learning for optimal quantum resource allocation to resolve the proposed stochastic resource allocation problem efficiently. Finally, we optimize the resource allocation in heterogeneous quantum computing networks supporting quantum generative learning applications and propose a multi-agent RL-based algorithm to learn the optimal resource allocation policies without prior knowledge.

{{</citation>}}


### (31/55) 6Rover: Leveraging Reinforcement Learning-based Address Pattern Mining Approach for Discovering Active Targets in IPv6 Unseeded Space (Zhichao Zhang et al., 2024)

{{<citation>}}

Zhichao Zhang, Zhaoxin Zhang, Yanan Cheng, Ning Li. (2024)  
**6Rover: Leveraging Reinforcement Learning-based Address Pattern Mining Approach for Discovering Active Targets in IPv6 Unseeded Space**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.07081v1)  

---


**ABSTRACT**  
The discovery of active IPv6 addresses represents a pivotal challenge in IPv6 network survey, as it is a prerequisite for downstream tasks such as network topology measurements and security analysis. With the rapid spread of IPv6 networks in recent years, many researchers have focused on improving the hit rate, efficiency, and coverage of IPv6 scanning methods, resulting in considerable advancements. However, existing approaches remain heavily dependent on seed addresses, thereby limiting their effectiveness in unseeded prefixes. Consequently, this paper proposes 6Rover, a reinforcement learning-based model for active address discovery in unseeded environments. To overcome the reliance on seeded addresses, 6Rover constructs patterns with higher generality that reflects the actual address allocation strategies of network administrators, thereby avoiding biased transfers of patterns from seeded to unseeded prefixes. After that, 6Rover employs a multi-armed bandit model to optimize the probing resource allocation when applying patterns to unseeded spaces. It models the challenge of discovering optimal patterns in unseeded spaces as an exploration-exploitation dilemma, and progressively uncover the potential patterns applied in unseeded spaces, leading to the efficient discovery of active addresses without seed address as the prior knowledge. Experiments on large-scale unseeded datasets show that 6Rover has a higher hit rate than existing methods in the absence of any seed addresses as prior knowledge. In real network environments, 6Rover achieved a 5% - 8% hit rate in seedless spaces with 100 million budget scale, representing an approximate 200\% improvement over the existing state-of-the-art methods.

{{</citation>}}


## cs.AI (3)



### (32/55) Open Models, Closed Minds? On Agents Capabilities in Mimicking Human Personalities through Open Large Language Models (Lucio La Cava et al., 2024)

{{<citation>}}

Lucio La Cava, Davide Costa, Andrea Tagarelli. (2024)  
**Open Models, Closed Minds? On Agents Capabilities in Mimicking Human Personalities through Open Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CY, cs-HC, cs.AI, physics-soc-ph  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.07115v1)  

---


**ABSTRACT**  
The emergence of unveiling human-like behaviors in Large Language Models (LLMs) has led to a closer connection between NLP and human psychology, leading to a proliferation of computational agents. Scholars have been studying the inherent personalities displayed by LLM agents and attempting to incorporate human traits and behaviors into them. However, these efforts have primarily focused on commercially-licensed LLMs, neglecting the widespread use and notable advancements seen in Open LLMs. This work aims to address this gap by conducting a comprehensive examination of the ability of agents to emulate human personalities using Open LLMs. To achieve this, we generate a set of ten LLM Agents based on the most representative Open models and subject them to a series of assessments concerning the Myers-Briggs Type Indicator (MBTI) test. Our approach involves evaluating the intrinsic personality traits of Open LLM agents and determining the extent to which these agents can mimic human personalities when conditioned by specific personalities and roles. Our findings unveil that: $(i)$ each Open LLM agent showcases distinct human personalities; $(ii)$ personality-conditioned prompting produces varying effects on the agents, with only few successfully mirroring the imposed personality, while most of them being ``closed-minded'' (i.e., they retain their intrinsic traits); $(iii)$ combining role and personality conditioning can enhance the agents' ability to mimic human personalities; and $(iv)$ personalities typically associated with the role of teacher tend to be emulated with greater accuracy. Our work represents a step up in understanding the dense relationship between NLP and human psychology through the lens of Open LLMs.

{{</citation>}}


### (33/55) Aquarium: A Comprehensive Framework for Exploring Predator-Prey Dynamics through Multi-Agent Reinforcement Learning Algorithms (Michael Kölle et al., 2024)

{{<citation>}}

Michael Kölle, Yannick Erpelding, Fabian Ritz, Thomy Phan, Steffen Illium, Claudia Linnhoff-Popien. (2024)  
**Aquarium: A Comprehensive Framework for Exploring Predator-Prey Dynamics through Multi-Agent Reinforcement Learning Algorithms**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-MA, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.07056v1)  

---


**ABSTRACT**  
Recent advances in Multi-Agent Reinforcement Learning have prompted the modeling of intricate interactions between agents in simulated environments. In particular, the predator-prey dynamics have captured substantial interest and various simulations been tailored to unique requirements. To prevent further time-intensive developments, we introduce Aquarium, a comprehensive Multi-Agent Reinforcement Learning environment for predator-prey interaction, enabling the study of emergent behavior. Aquarium is open source and offers a seamless integration of the PettingZoo framework, allowing a quick start with proven algorithm implementations. It features physics-based agent movement on a two-dimensional, edge-wrapping plane. The agent-environment interaction (observations, actions, rewards) and the environment settings (agent speed, prey reproduction, predator starvation, and others) are fully customizable. Besides a resource-efficient visualization, Aquarium supports to record video files, providing a visual comprehension of agent behavior. To demonstrate the environment's capabilities, we conduct preliminary studies which use PPO to train multiple prey agents to evade a predator. In accordance to the literature, we find Individual Learning to result in worse performance than Parameter Sharing, which significantly improves coordination and sample-efficiency.

{{</citation>}}


### (34/55) Distance-aware Attention Reshaping: Enhance Generalization of Neural Solver for Large-scale Vehicle Routing Problems (Yang Wang et al., 2024)

{{<citation>}}

Yang Wang, Ya-Hui Jia, Wei-Neng Chen, Yi Mei. (2024)  
**Distance-aware Attention Reshaping: Enhance Generalization of Neural Solver for Large-scale Vehicle Routing Problems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.06979v1)  

---


**ABSTRACT**  
Neural solvers based on attention mechanism have demonstrated remarkable effectiveness in solving vehicle routing problems. However, in the generalization process from small scale to large scale, we find a phenomenon of the dispersion of attention scores in existing neural solvers, which leads to poor performance. To address this issue, this paper proposes a distance-aware attention reshaping method, assisting neural solvers in solving large-scale vehicle routing problems. Specifically, without the need for additional training, we utilize the Euclidean distance information between current nodes to adjust attention scores. This enables a neural solver trained on small-scale instances to make rational choices when solving a large-scale problem. Experimental results show that the proposed method significantly outperforms existing state-of-the-art neural solvers on the large-scale CVRPLib dataset.

{{</citation>}}


## cs.NE (1)



### (35/55) Evolving Code with A Large Language Model (Erik Hemberg et al., 2024)

{{<citation>}}

Erik Hemberg, Stephen Moskal, Una-May O'Reilly. (2024)  
**Evolving Code with A Large Language Model**  

---
Primary Category: cs.NE  
Categories: I-2-8, cs-AI, cs-NE, cs.NE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07102v1)  

---


**ABSTRACT**  
Algorithms that use Large Language Models (LLMs) to evolve code arrived on the Genetic Programming (GP) scene very recently. We present LLM GP, a formalized LLM-based evolutionary algorithm designed to evolve code. Like GP, it uses evolutionary operators, but its designs and implementations of those operators radically differ from GP's because they enlist an LLM, using prompting and the LLM's pre-trained pattern matching and sequence completion capability. We also present a demonstration-level variant of LLM GP and share its code. By addressing algorithms that range from the formal to hands-on, we cover design and LLM-usage considerations as well as the scientific challenges that arise when using an LLM for genetic programming.

{{</citation>}}


## cs.CV (11)



### (36/55) Exploring Adversarial Attacks against Latent Diffusion Model from the Perspective of Adversarial Transferability (Junxi Chen et al., 2024)

{{<citation>}}

Junxi Chen, Junhao Dong, Xiaohua Xie. (2024)  
**Exploring Adversarial Attacks against Latent Diffusion Model from the Perspective of Adversarial Transferability**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2401.07087v1)  

---


**ABSTRACT**  
Recently, many studies utilized adversarial examples (AEs) to raise the cost of malicious image editing and copyright violation powered by latent diffusion models (LDMs). Despite their successes, a few have studied the surrogate model they used to generate AEs. In this paper, from the perspective of adversarial transferability, we investigate how the surrogate model's property influences the performance of AEs for LDMs. Specifically, we view the time-step sampling in the Monte-Carlo-based (MC-based) adversarial attack as selecting surrogate models. We find that the smoothness of surrogate models at different time steps differs, and we substantially improve the performance of the MC-based AEs by selecting smoother surrogate models. In the light of the theoretical framework on adversarial transferability in image classification, we also conduct a theoretical analysis to explain why smooth surrogate models can also boost AEs for LDMs.

{{</citation>}}


### (37/55) GoMatching: A Simple Baseline for Video Text Spotting via Long and Short Term Matching (Haibin He et al., 2024)

{{<citation>}}

Haibin He, Maoyuan Ye, Jing Zhang, Juhua Liu, Dacheng Tao. (2024)  
**GoMatching: A Simple Baseline for Video Text Spotting via Long and Short Term Matching**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.07080v1)  

---


**ABSTRACT**  
Beyond the text detection and recognition tasks in image text spotting, video text spotting presents an augmented challenge with the inclusion of tracking. While advanced end-to-end trainable methods have shown commendable performance, the pursuit of multi-task optimization may pose the risk of producing sub-optimal outcomes for individual tasks. In this paper, we highlight a main bottleneck in the state-of-the-art video text spotter: the limited recognition capability. In response to this issue, we propose to efficiently turn an off-the-shelf query-based image text spotter into a specialist on video and present a simple baseline termed GoMatching, which focuses the training efforts on tracking while maintaining strong recognition performance. To adapt the image text spotter to video datasets, we add a rescoring head to rescore each detected instance's confidence via efficient tuning, leading to a better tracking candidate pool. Additionally, we design a long-short term matching module, termed LST-Matcher, to enhance the spotter's tracking capability by integrating both long- and short-term matching results via Transformer. Based on the above simple designs, GoMatching achieves impressive performance on two public benchmarks, e.g., setting a new record on the ICDAR15-video dataset, and one novel test set with arbitrary-shaped text, while saving considerable training budgets. The code will be released at https://github.com/Hxyz-123/GoMatching.

{{</citation>}}


### (38/55) Dual-View Data Hallucination with Semantic Relation Guidance for Few-Shot Image Recognition (Hefeng Wu et al., 2024)

{{<citation>}}

Hefeng Wu, Guangzhi Ye, Ziyang Zhou, Ling Tian, Qing Wang, Liang Lin. (2024)  
**Dual-View Data Hallucination with Semantic Relation Guidance for Few-Shot Image Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2401.07061v1)  

---


**ABSTRACT**  
Learning to recognize novel concepts from just a few image samples is very challenging as the learned model is easily overfitted on the few data and results in poor generalizability. One promising but underexplored solution is to compensate the novel classes by generating plausible samples. However, most existing works of this line exploit visual information only, rendering the generated data easy to be distracted by some challenging factors contained in the few available samples. Being aware of the semantic information in the textual modality that reflects human concepts, this work proposes a novel framework that exploits semantic relations to guide dual-view data hallucination for few-shot image recognition. The proposed framework enables generating more diverse and reasonable data samples for novel classes through effective information transfer from base classes. Specifically, an instance-view data hallucination module hallucinates each sample of a novel class to generate new data by employing local semantic correlated attention and global semantic feature fusion derived from base classes. Meanwhile, a prototype-view data hallucination module exploits semantic-aware measure to estimate the prototype of a novel class and the associated distribution from the few samples, which thereby harvests the prototype as a more stable sample and enables resampling a large number of samples. We conduct extensive experiments and comparisons with state-of-the-art methods on several popular few-shot benchmarks to verify the effectiveness of the proposed framework.

{{</citation>}}


### (39/55) NODI: Out-Of-Distribution Detection with Noise from Diffusion (Jingqiu Zhou et al., 2024)

{{<citation>}}

Jingqiu Zhou, Aojun Zou, Hongshen Li. (2024)  
**NODI: Out-Of-Distribution Detection with Noise from Diffusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.08689v1)  

---


**ABSTRACT**  
Out-of-distribution (OOD) detection is a crucial part of deploying machine learning models safely. It has been extensively studied with a plethora of methods developed in the literature. This problem is tackled with an OOD score computation, however, previous methods compute the OOD scores with limited usage of the in-distribution dataset. For instance, the OOD scores are computed with information from a small portion of the in-distribution data. Furthermore, these methods encode images with a neural image encoder. The robustness of these methods is rarely checked with respect to image encoders of different training methods and architectures. In this work, we introduce the diffusion process into the OOD task. The diffusion model integrates information on the whole training set into the predicted noise vectors. What's more, we deduce a closed-form solution for the noise vector (stable point). Then the noise vector is converted into our OOD score, we test both the deep model predicted noise vector and the closed-form noise vector on the OOD benchmarks \cite{openood}. Our method outperforms previous OOD methods across all types of image encoders (Table. \ref{main}). A $3.5\%$ performance gain is achieved with the MAE-based image encoder. Moreover, we studied the robustness of OOD methods by applying different types of image encoders. Some OOD methods failed to generalize well when switching image encoders from ResNet to Vision Transformers, our method performs exhibits good robustness with all the image encoders.

{{</citation>}}


### (40/55) DA-BEV: Unsupervised Domain Adaptation for Bird's Eye View Perception (Kai Jiang et al., 2024)

{{<citation>}}

Kai Jiang, Jiaxing Huang, Weiying Xie, Yunsong Li, Ling Shao, Shijian Lu. (2024)  
**DA-BEV: Unsupervised Domain Adaptation for Bird's Eye View Perception**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.08687v1)  

---


**ABSTRACT**  
Camera-only Bird's Eye View (BEV) has demonstrated great potential in environment perception in a 3D space. However, most existing studies were conducted under a supervised setup which cannot scale well while handling various new data. Unsupervised domain adaptive BEV, which effective learning from various unlabelled target data, is far under-explored. In this work, we design DA-BEV, the first domain adaptive camera-only BEV framework that addresses domain adaptive BEV challenges by exploiting the complementary nature of image-view features and BEV features. DA-BEV introduces the idea of query into the domain adaptation framework to derive useful information from image-view and BEV features. It consists of two query-based designs, namely, query-based adversarial learning (QAL) and query-based self-training (QST), which exploits image-view features or BEV features to regularize the adaptation of the other. Extensive experiments show that DA-BEV achieves superior domain adaptive BEV perception performance consistently across multiple datasets and tasks such as 3D object detection and 3D scene segmentation.

{{</citation>}}


### (41/55) Class-Imbalanced Semi-Supervised Learning for Large-Scale Point Cloud Semantic Segmentation via Decoupling Optimization (Mengtian Li et al., 2024)

{{<citation>}}

Mengtian Li, Shaohui Lin, Zihan Wang, Yunhang Shen, Baochang Zhang, Lizhuang Ma. (2024)  
**Class-Imbalanced Semi-Supervised Learning for Large-Scale Point Cloud Semantic Segmentation via Decoupling Optimization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2401.06975v1)  

---


**ABSTRACT**  
Semi-supervised learning (SSL), thanks to the significant reduction of data annotation costs, has been an active research topic for large-scale 3D scene understanding. However, the existing SSL-based methods suffer from severe training bias, mainly due to class imbalance and long-tail distributions of the point cloud data. As a result, they lead to a biased prediction for the tail class segmentation. In this paper, we introduce a new decoupling optimization framework, which disentangles feature representation learning and classifier in an alternative optimization manner to shift the bias decision boundary effectively. In particular, we first employ two-round pseudo-label generation to select unlabeled points across head-to-tail classes. We further introduce multi-class imbalanced focus loss to adaptively pay more attention to feature learning across head-to-tail classes. We fix the backbone parameters after feature learning and retrain the classifier using ground-truth points to update its parameters. Extensive experiments demonstrate the effectiveness of our method outperforming previous state-of-the-art methods on both indoor and outdoor 3D point cloud datasets (i.e., S3DIS, ScanNet-V2, Semantic3D, and SemanticKITTI) using 1% and 1pt evaluation.

{{</citation>}}


### (42/55) Domain Adaptation for Large-Vocabulary Object Detectors (Kai Jiang et al., 2024)

{{<citation>}}

Kai Jiang, Jiaxing Huang, Weiying Xie, Yunsong Li, Ling Shao, Shijian Lu. (2024)  
**Domain Adaptation for Large-Vocabulary Object Detectors**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.06969v1)  

---


**ABSTRACT**  
Large-vocabulary object detectors (LVDs) aim to detect objects of many categories, which learn super objectness features and can locate objects accurately while applied to various downstream data. However, LVDs often struggle in recognizing the located objects due to domain discrepancy in data distribution and object vocabulary. At the other end, recent vision-language foundation models such as CLIP demonstrate superior open-vocabulary recognition capability. This paper presents KGD, a Knowledge Graph Distillation technique that exploits the implicit knowledge graphs (KG) in CLIP for effectively adapting LVDs to various downstream domains. KGD consists of two consecutive stages: 1) KG extraction that employs CLIP to encode downstream domain data as nodes and their feature distances as edges, constructing KG that inherits the rich semantic relations in CLIP explicitly; and 2) KG encapsulation that transfers the extracted KG into LVDs to enable accurate cross-domain object classification. In addition, KGD can extract both visual and textual KG independently, providing complementary vision and language knowledge for object localization and object classification in detection tasks over various downstream domains. Experiments over multiple widely adopted detection benchmarks show that KGD outperforms the state-of-the-art consistently by large margins.

{{</citation>}}


### (43/55) Transformer for Object Re-Identification: A Survey (Mang Ye et al., 2024)

{{<citation>}}

Mang Ye, Shuoyi Chen, Chenyue Li, Wei-Shi Zheng, David Crandall, Bo Du. (2024)  
**Transformer for Object Re-Identification: A Survey**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.06960v1)  

---


**ABSTRACT**  
Object Re-Identification (Re-ID) aims to identify and retrieve specific objects from varying viewpoints. For a prolonged period, this field has been predominantly driven by deep convolutional neural networks. In recent years, the Transformer has witnessed remarkable advancements in computer vision, prompting an increasing body of research to delve into the application of Transformer in Re-ID. This paper provides a comprehensive review and in-depth analysis of the Transformer-based Re-ID. In categorizing existing works into Image/Video-Based Re-ID, Re-ID with limited data/annotations, Cross-Modal Re-ID, and Special Re-ID Scenarios, we thoroughly elucidate the advantages demonstrated by the Transformer in addressing a multitude of challenges across these domains. Considering the trending unsupervised Re-ID, we propose a new Transformer baseline, UntransReID, achieving state-of-the-art performance on both single-/cross modal tasks. Besides, this survey also covers a wide range of Re-ID research objects, including progress in animal Re-ID. Given the diversity of species in animal Re-ID, we devise a standardized experimental benchmark and conduct extensive experiments to explore the applicability of Transformer for this task to facilitate future research. Finally, we discuss some important yet under-investigated open issues in the big foundation model era, we believe it will serve as a new handbook for researchers in this field.

{{</citation>}}


### (44/55) Attention Modules Improve Modern Image-Level Anomaly Detection: A DifferNet Case Study (André Luiz B. Vieira e Silva et al., 2024)

{{<citation>}}

André Luiz B. Vieira e Silva, Francisco Simões, Danny Kowerko, Tobias Schlosser, Felipe Battisti, Veronica Teichrieb. (2024)  
**Attention Modules Improve Modern Image-Level Anomaly Detection: A DifferNet Case Study**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection, Attention  
[Paper Link](http://arxiv.org/abs/2401.08686v1)  

---


**ABSTRACT**  
Within (semi-)automated visual inspection, learning-based approaches for assessing visual defects, including deep neural networks, enable the processing of otherwise small defect patterns in pixel size on high-resolution imagery. The emergence of these often rarely occurring defect patterns explains the general need for labeled data corpora. To not only alleviate this issue but to furthermore advance the current state of the art in unsupervised visual inspection, this contribution proposes a DifferNet-based solution enhanced with attention modules utilizing SENet and CBAM as backbone - AttentDifferNet - to improve the detection and classification capabilities on three different visual inspection and anomaly detection datasets: MVTec AD, InsPLAD-fault, and Semiconductor Wafer. In comparison to the current state of the art, it is shown that AttentDifferNet achieves improved results, which are, in turn, highlighted throughout our quantitative as well as qualitative evaluation, indicated by a general improvement in AUC of 94.34 vs. 92.46, 96.67 vs. 94.69, and 90.20 vs. 88.74%. As our variants to AttentDifferNet show great prospects in the context of currently investigated approaches, a baseline is formulated, emphasizing the importance of attention for anomaly detection.

{{</citation>}}


### (45/55) EVOKE: Emotion Enabled Virtual Avatar Mapping Using Optimized Knowledge Distillation (Maryam Nadeem et al., 2024)

{{<citation>}}

Maryam Nadeem, Raza Imam, Rouqaiah Al-Refai, Meriem Chkir, Mohamad Hoda, Abdulmotaleb El Saddik. (2024)  
**EVOKE: Emotion Enabled Virtual Avatar Mapping Using Optimized Knowledge Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2401.06957v1)  

---


**ABSTRACT**  
As virtual environments continue to advance, the demand for immersive and emotionally engaging experiences has grown. Addressing this demand, we introduce Emotion enabled Virtual avatar mapping using Optimized KnowledgE distillation (EVOKE), a lightweight emotion recognition framework designed for the seamless integration of emotion recognition into 3D avatars within virtual environments. Our approach leverages knowledge distillation involving multi-label classification on the publicly available DEAP dataset, which covers valence, arousal, and dominance as primary emotional classes. Remarkably, our distilled model, a CNN with only two convolutional layers and 18 times fewer parameters than the teacher model, achieves competitive results, boasting an accuracy of 87% while demanding far less computational resources. This equilibrium between performance and deployability positions our framework as an ideal choice for virtual environment systems. Furthermore, the multi-label classification outcomes are utilized to map emotions onto custom-designed 3D avatars.

{{</citation>}}


### (46/55) 3D Object Detection and High-Resolution Traffic Parameters Extraction Using Low-Resolution LiDAR Data (Linlin Zhang et al., 2024)

{{<citation>}}

Linlin Zhang, Xiang Yu, Armstrong Aboah, Yaw Adu-Gyamfi. (2024)  
**3D Object Detection and High-Resolution Traffic Parameters Extraction Using Low-Resolution LiDAR Data**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.06946v1)  

---


**ABSTRACT**  
Traffic volume data collection is a crucial aspect of transportation engineering and urban planning, as it provides vital insights into traffic patterns, congestion, and infrastructure efficiency. Traditional manual methods of traffic data collection are both time-consuming and costly. However, the emergence of modern technologies, particularly Light Detection and Ranging (LiDAR), has revolutionized the process by enabling efficient and accurate data collection. Despite the benefits of using LiDAR for traffic data collection, previous studies have identified two major limitations that have impeded its widespread adoption. These are the need for multiple LiDAR systems to obtain complete point cloud information of objects of interest, as well as the labor-intensive process of annotating 3D bounding boxes for object detection tasks. In response to these challenges, the current study proposes an innovative framework that alleviates the need for multiple LiDAR systems and simplifies the laborious 3D annotation process. To achieve this goal, the study employed a single LiDAR system, that aims at reducing the data acquisition cost and addressed its accompanying limitation of missing point cloud information by developing a Point Cloud Completion (PCC) framework to fill in missing point cloud information using point density. Furthermore, we also used zero-shot learning techniques to detect vehicles and pedestrians, as well as proposed a unique framework for extracting low to high features from the object of interest, such as height, acceleration, and speed. Using the 2D bounding box detection and extracted height information, this study is able to generate 3D bounding boxes automatically without human intervention.

{{</citation>}}


## cs.MM (1)



### (47/55) ScripTONES: Sentiment-Conditioned Music Generation for Movie Scripts (Vishruth Veerendranath et al., 2024)

{{<citation>}}

Vishruth Veerendranath, Vibha Masti, Utkarsh Gupta, Hrishit Chaudhuri, Gowri Srinivasa. (2024)  
**ScripTONES: Sentiment-Conditioned Music Generation for Movie Scripts**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs-SD, cs.MM, eess-AS  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2401.07084v1)  

---


**ABSTRACT**  
Film scores are considered an essential part of the film cinematic experience, but the process of film score generation is often expensive and infeasible for small-scale creators. Automating the process of film score composition would provide useful starting points for music in small projects. In this paper, we propose a two-stage pipeline for generating music from a movie script. The first phase is the Sentiment Analysis phase where the sentiment of a scene from the film script is encoded into the valence-arousal continuous space. The second phase is the Conditional Music Generation phase which takes as input the valence-arousal vector and conditionally generates piano MIDI music to match the sentiment. We study the efficacy of various music generation architectures by performing a qualitative user survey and propose methods to improve sentiment-conditioning in VAE architectures.

{{</citation>}}


## stat.ML (1)



### (48/55) Towards Responsible AI in Banking: Addressing Bias for Fair Decision-Making (Alessandro Castelnovo, 2024)

{{<citation>}}

Alessandro Castelnovo. (2024)  
**Towards Responsible AI in Banking: Addressing Bias for Fair Decision-Making**  

---
Primary Category: stat.ML  
Categories: cs-CY, cs-LG, stat-AP, stat-ML, stat.ML  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2401.08691v1)  

---


**ABSTRACT**  
In an era characterized by the pervasive integration of artificial intelligence into decision-making processes across diverse industries, the demand for trust has never been more pronounced. This thesis embarks on a comprehensive exploration of bias and fairness, with a particular emphasis on their ramifications within the banking sector, where AI-driven decisions bear substantial societal consequences. In this context, the seamless integration of fairness, explainability, and human oversight is of utmost importance, culminating in the establishment of what is commonly referred to as "Responsible AI". This emphasizes the critical nature of addressing biases within the development of a corporate culture that aligns seamlessly with both AI regulations and universal human rights standards, particularly in the realm of automated decision-making systems. Nowadays, embedding ethical principles into the development, training, and deployment of AI models is crucial for compliance with forthcoming European regulations and for promoting societal good. This thesis is structured around three fundamental pillars: understanding bias, mitigating bias, and accounting for bias. These contributions are validated through their practical application in real-world scenarios, in collaboration with Intesa Sanpaolo. This collaborative effort not only contributes to our understanding of fairness but also provides practical tools for the responsible implementation of AI-based decision-making systems. In line with open-source principles, we have released Bias On Demand and FairView as accessible Python packages, further promoting progress in the field of AI fairness.

{{</citation>}}


## cs.CY (1)



### (49/55) Classifying Proposals of Decentralized Autonomous Organizations Using Large Language Models (Christian Ziegler et al., 2024)

{{<citation>}}

Christian Ziegler, Marcos Miranda, Guangye Cao, Gustav Arentoft, Doo Wan Nam. (2024)  
**Classifying Proposals of Decentralized Autonomous Organizations Using Large Language Models**  

---
Primary Category: cs.CY  
Categories: H-0, cs-CY, cs.CY  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07059v1)  

---


**ABSTRACT**  
Our study demonstrates the effective use of Large Language Models (LLMs) for automating the classification of complex datasets. We specifically target proposals of Decentralized Autonomous Organizations (DAOs), as the classification of this data requires the understanding of context and, therefore, depends on human expertise, leading to high costs associated with the task. The study applies an iterative approach to specify categories and further refine them and the prompt in each iteration, which led to an accuracy rate of 95% in classifying a set of 100 proposals. With this, we demonstrate the potential of LLMs to automate data labeling tasks that depend on textual context effectively.

{{</citation>}}


## quant-ph (2)



### (50/55) A Reinforcement Learning Environment for Directed Quantum Circuit Synthesis (Michael Kölle et al., 2024)

{{<citation>}}

Michael Kölle, Tom Schubert, Philipp Altmann, Maximilian Zorn, Jonas Stein, Claudia Linnhoff-Popien. (2024)  
**A Reinforcement Learning Environment for Directed Quantum Circuit Synthesis**  

---
Primary Category: quant-ph  
Categories: cs-AI, quant-ph, quant-ph  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.07054v1)  

---


**ABSTRACT**  
With recent advancements in quantum computing technology, optimizing quantum circuits and ensuring reliable quantum state preparation have become increasingly vital. Traditional methods often demand extensive expertise and manual calculations, posing challenges as quantum circuits grow in qubit- and gate-count. Therefore, harnessing machine learning techniques to handle the growing variety of gate-to-qubit combinations is a promising approach. In this work, we introduce a comprehensive reinforcement learning environment for quantum circuit synthesis, where circuits are constructed utilizing gates from the the Clifford+T gate set to prepare specific target states. Our experiments focus on exploring the relationship between the depth of synthesized quantum circuits and the circuit depths used for target initialization, as well as qubit count. We organize the environment configurations into multiple evaluation levels and include a range of well-known quantum states for benchmarking purposes. We also lay baselines for evaluating the environment using Proximal Policy Optimization. By applying the trained agents to benchmark tests, we demonstrated their ability to reliably design minimal quantum circuits for a selection of 2-qubit Bell states.

{{</citation>}}


### (51/55) Quantum Advantage Actor-Critic for Reinforcement Learning (Michael Kölle et al., 2024)

{{<citation>}}

Michael Kölle, Mohamad Hgog, Fabian Ritz, Philipp Altmann, Maximilian Zorn, Jonas Stein, Claudia Linnhoff-Popien. (2024)  
**Quantum Advantage Actor-Critic for Reinforcement Learning**  

---
Primary Category: quant-ph  
Categories: cs-AI, cs-LG, quant-ph, quant-ph  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.07043v1)  

---


**ABSTRACT**  
Quantum computing offers efficient encapsulation of high-dimensional states. In this work, we propose a novel quantum reinforcement learning approach that combines the Advantage Actor-Critic algorithm with variational quantum circuits by substituting parts of the classical components. This approach addresses reinforcement learning's scalability concerns while maintaining high performance. We empirically test multiple quantum Advantage Actor-Critic configurations with the well known Cart Pole environment to evaluate our approach in control tasks with continuous state spaces. Our results indicate that the hybrid strategy of using either a quantum actor or quantum critic with classical post-processing yields a substantial performance increase compared to pure classical and pure quantum variants with similar parameter counts. They further reveal the limits of current quantum approaches due to the hardware constraints of noisy intermediate-scale quantum computers, suggesting further research to scale hybrid approaches for larger and more complex control tasks.

{{</citation>}}


## cs.SE (1)



### (52/55) Causative Insights into Open Source Software Security using Large Language Code Embeddings and Semantic Vulnerability Graph (Nafis Tanveer Islam et al., 2024)

{{<citation>}}

Nafis Tanveer Islam, Gonzalo De La Torre Parra, Dylan Manual, Murtuza Jadliwala, Peyman Najafirad. (2024)  
**Causative Insights into Open Source Software Security using Large Language Code Embeddings and Semantic Vulnerability Graph**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Embedding, Security  
[Paper Link](http://arxiv.org/abs/2401.07035v1)  

---


**ABSTRACT**  
Open Source Software (OSS) security and resilience are worldwide phenomena hampering economic and technological innovation. OSS vulnerabilities can cause unauthorized access, data breaches, network disruptions, and privacy violations, rendering any benefits worthless. While recent deep-learning techniques have shown great promise in identifying and localizing vulnerabilities in source code, it is unclear how effective these research techniques are from a usability perspective due to a lack of proper methodological analysis. Usually, these methods offload a developer's task of classifying and localizing vulnerable code; still, a reasonable study to measure the actual effectiveness of these systems to the end user has yet to be conducted. To address the challenge of proper developer training from the prior methods, we propose a system to link vulnerabilities to their root cause, thereby intuitively educating the developers to code more securely. Furthermore, we provide a comprehensive usability study to test the effectiveness of our system in fixing vulnerabilities and its capability to assist developers in writing more secure code. We demonstrate the effectiveness of our system by showing its efficacy in helping developers fix source code with vulnerabilities. Our study shows a 24% improvement in code repair capabilities compared to previous methods. We also show that, when trained by our system, on average, approximately 9% of the developers naturally tend to write more secure code with fewer vulnerabilities.

{{</citation>}}


## eess.IV (1)



### (53/55) Empowering Medical Imaging with Artificial Intelligence: A Review of Machine Learning Approaches for the Detection, and Segmentation of COVID-19 Using Radiographic and Tomographic Images (Sayed Amir Mousavi Mobarakeh et al., 2024)

{{<citation>}}

Sayed Amir Mousavi Mobarakeh, Kamran Kazemi, Ardalan Aarabi, Habibollah Danyal. (2024)  
**Empowering Medical Imaging with Artificial Intelligence: A Review of Machine Learning Approaches for the Detection, and Segmentation of COVID-19 Using Radiographic and Tomographic Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07020v1)  

---


**ABSTRACT**  
Since 2019, the global dissemination of the Coronavirus and its novel strains has resulted in a surge of new infections. The use of X-ray and computed tomography (CT) imaging techniques is critical in diagnosing and managing COVID-19. Incorporating artificial intelligence (AI) into the field of medical imaging is a powerful combination that can provide valuable support to healthcare professionals.This paper focuses on the methodological approach of using machine learning (ML) to enhance medical imaging for COVID-19 diagnosis.For example, deep learning can accurately distinguish lesions from other parts of the lung without human intervention in a matter of minutes.Moreover, ML can enhance performance efficiency by assisting radiologists in making more precise clinical decisions, such as detecting and distinguishing Covid-19 from different respiratory infections and segmenting infections in CT and X-ray images, even when the lesions have varying sizes and shapes.This article critically assesses machine learning methodologies utilized for the segmentation, classification, and detection of Covid-19 within CT and X-ray images, which are commonly employed tools in clinical and hospital settings to represent the lung in various aspects and extensive detail.There is a widespread expectation that this technology will continue to hold a central position within the healthcare sector, driving further progress in the management of the pandemic.

{{</citation>}}


## q-bio.QM (1)



### (54/55) NHANES-GCP: Leveraging the Google Cloud Platform and BigQuery ML for reproducible machine learning with data from the National Health and Nutrition Examination Survey (B. Ross Katz et al., 2024)

{{<citation>}}

B. Ross Katz, Abdul Khan, James York-Winegar, Alexander J. Titus. (2024)  
**NHANES-GCP: Leveraging the Google Cloud Platform and BigQuery ML for reproducible machine learning with data from the National Health and Nutrition Examination Survey**  

---
Primary Category: q-bio.QM  
Categories: cs-LG, q-bio-QM, q-bio.QM, stat-AP  
Keywords: GCP, Google, Language Model  
[Paper Link](http://arxiv.org/abs/2401.06967v1)  

---


**ABSTRACT**  
Summary: NHANES, the National Health and Nutrition Examination Survey, is a program of studies led by the Centers for Disease Control and Prevention (CDC) designed to assess the health and nutritional status of adults and children in the United States (U.S.). NHANES data is frequently used by biostatisticians and clinical scientists to study health trends across the U.S., but every analysis requires extensive data management and cleaning before use and this repetitive data engineering collectively costs valuable research time and decreases the reproducibility of analyses. Here, we introduce NHANES-GCP, a Cloud Development Kit for Terraform (CDKTF) Infrastructure-as-Code (IaC) and Data Build Tool (dbt) resources built on the Google Cloud Platform (GCP) that automates the data engineering and management aspects of working with NHANES data. With current GCP pricing, NHANES-GCP costs less than $2 to run and less than $15/yr of ongoing costs for hosting the NHANES data, all while providing researchers with clean data tables that can readily be integrated for large-scale analyses. We provide examples of leveraging BigQuery ML to carry out the process of selecting data, integrating data, training machine learning and statistical models, and generating results all from a single SQL-like query. NHANES-GCP is designed to enhance the reproducibility of analyses and create a well-engineered NHANES data resource for statistics, machine learning, and fine-tuning Large Language Models (LLMs).   Availability and implementation" NHANES-GCP is available at https://github.com/In-Vivo-Group/NHANES-GCP

{{</citation>}}


## cs.RO (1)



### (55/55) ORGANA: A Robotic Assistant for Automated Chemistry Experimentation and Characterization (Kourosh Darvish et al., 2024)

{{<citation>}}

Kourosh Darvish, Marta Skreta, Yuchi Zhao, Naruki Yoshikawa, Sagnik Som, Miroslav Bogdanovic, Yang Cao, Han Hao, Haoping Xu, Alán Aspuru-Guzik, Animesh Garg, Florian Shkurti. (2024)  
**ORGANA: A Robotic Assistant for Automated Chemistry Experimentation and Characterization**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.06949v1)  

---


**ABSTRACT**  
Chemistry experimentation is often resource- and labor-intensive. Despite the many benefits incurred by the integration of advanced and special-purpose lab equipment, many aspects of experimentation are still manually conducted by chemists, for example, polishing an electrode in electrochemistry experiments. Traditional lab automation infrastructure faces challenges when it comes to flexibly adapting to new chemistry experiments. To address this issue, we propose a human-friendly and flexible robotic system, ORGANA, that automates a diverse set of chemistry experiments. It is capable of interacting with chemists in the lab through natural language, using Large Language Models (LLMs). ORGANA keeps scientists informed by providing timely reports that incorporate statistical analyses. Additionally, it actively engages with users when necessary for disambiguation or troubleshooting. ORGANA can reason over user input to derive experiment goals, and plan long sequences of both high-level tasks and low-level robot actions while using feedback from the visual perception of the environment. It also supports scheduling and parallel execution for experiments that require resource allocation and coordination between multiple robots and experiment stations. We show that ORGANA successfully conducts a diverse set of chemistry experiments, including solubility assessment, pH measurement, recrystallization, and electrochemistry experiments. For the latter, we show that ORGANA robustly executes a long-horizon plan, comprising 19 steps executed in parallel, to characterize the electrochemical properties of quinone derivatives, a class of molecules used in rechargeable flow batteries. Our user study indicates that ORGANA significantly improves many aspects of user experience while reducing their physical workload. More details about ORGANA can be found at https://ac-rad.github.io/organa/.

{{</citation>}}
