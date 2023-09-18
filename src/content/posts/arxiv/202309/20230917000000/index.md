---
draft: false
title: "arXiv @ 2023.09.17"
date: 2023-09-17
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.17"
    identifier: arxiv_20230917
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (13)](#cslg-13)
- [cs.CL (27)](#cscl-27)
- [cs.CV (17)](#cscv-17)
- [cs.IT (2)](#csit-2)
- [cs.RO (5)](#csro-5)
- [cs.CR (2)](#cscr-2)
- [eess.AS (5)](#eessas-5)
- [eess.IV (4)](#eessiv-4)
- [cs.SD (5)](#cssd-5)
- [cs.AI (2)](#csai-2)
- [cs.CY (2)](#cscy-2)
- [stat.ML (1)](#statml-1)
- [cs.SE (2)](#csse-2)
- [cs.HC (2)](#cshc-2)
- [math.NA (1)](#mathna-1)
- [cs.SI (1)](#cssi-1)

## cs.LG (13)



### (1/91) Sparse Autoencoders Find Highly Interpretable Features in Language Models (Hoagy Cunningham et al., 2023)

{{<citation>}}

Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, Lee Sharkey. (2023)  
**Sparse Autoencoders Find Highly Interpretable Features in Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.08600v1)  

---


**ABSTRACT**  
One of the roadblocks to a better understanding of neural networks' internals is \textit{polysemanticity}, where neurons appear to activate in multiple, semantically distinct contexts. Polysemanticity prevents us from identifying concise, human-understandable explanations for what neural networks are doing internally. One hypothesised cause of polysemanticity is \textit{superposition}, where neural networks represent more features than they have neurons by assigning features to an overcomplete set of directions in activation space, rather than to individual neurons. Here, we attempt to identify those directions, using sparse autoencoders to reconstruct the internal activations of a language model. These autoencoders learn sets of sparsely activating features that are more interpretable and monosemantic than directions identified by alternative approaches, where interpretability is measured by automated methods. Ablating these features enables precise model editing, for example, by removing capabilities such as pronoun prediction, while disrupting model behaviour less than prior techniques. This work indicates that it is possible to resolve superposition in language models using a scalable, unsupervised method. Our method may serve as a foundation for future mechanistic interpretability work, which we hope will enable greater model transparency and steerability.

{{</citation>}}


### (2/91) Attention-Only Transformers and Implementing MLPs with Attention Heads (Robert Huben et al., 2023)

{{<citation>}}

Robert Huben, Valerie Morris. (2023)  
**Attention-Only Transformers and Implementing MLPs with Attention Heads**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.08593v1)  

---


**ABSTRACT**  
The transformer architecture is widely used in machine learning models and consists of two alternating sublayers: attention heads and MLPs. We prove that an MLP neuron can be implemented by a masked attention head with internal dimension 1 so long as the MLP's activation function comes from a restricted class including SiLU and close approximations of ReLU and GeLU. This allows one to convert an MLP-and-attention transformer into an attention-only transformer at the cost of greatly increasing the number of attention heads. We also prove that attention heads can perform the components of an MLP (linear transformations and activation functions) separately. Finally, we prove that attention heads can encode arbitrary masking patterns in their weight matrices to within arbitrarily small error.

{{</citation>}}


### (3/91) Chain-of-Thought Reasoning is a Policy Improvement Operator (Hugh Zhang et al., 2023)

{{<citation>}}

Hugh Zhang, David C. Parkes. (2023)  
**Chain-of-Thought Reasoning is a Policy Improvement Operator**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2309.08589v1)  

---


**ABSTRACT**  
Large language models have astounded the world with fascinating new capabilities. However, they currently lack the ability to teach themselves new skills, relying instead on being trained on large amounts of human-generated data. We introduce SECToR (Self-Education via Chain-of-Thought Reasoning), a proof-of-concept demonstration that language models can successfully teach themselves new skills using chain-of-thought reasoning. Inspired by previous work in both reinforcement learning (Silver et al., 2017) and human cognition (Kahneman, 2011), SECToR first uses chain-of-thought reasoning to slowly think its way through problems. SECToR then fine-tunes the model to generate those same answers, this time without using chain-of-thought reasoning. Language models trained via SECToR autonomously learn to add up to 29-digit numbers without any access to any ground truth examples beyond an initial supervised fine-tuning phase consisting only of numbers with 6 or fewer digits. Our central hypothesis is that chain-of-thought reasoning can act as a policy improvement operator, analogously to how Monte-Carlo Tree Search is used in AlphaZero. We hope that this research can lead to new directions in which language models can learn to teach themselves without the need for human demonstrations.

{{</citation>}}


### (4/91) A Bayesian Approach to Robust Inverse Reinforcement Learning (Ran Wei et al., 2023)

{{<citation>}}

Ran Wei, Siliang Zeng, Chenliang Li, Alfredo Garcia, Anthony McDonald, Mingyi Hong. (2023)  
**A Bayesian Approach to Robust Inverse Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.08571v1)  

---


**ABSTRACT**  
We consider a Bayesian approach to offline model-based inverse reinforcement learning (IRL). The proposed framework differs from existing offline model-based IRL approaches by performing simultaneous estimation of the expert's reward function and subjective model of environment dynamics. We make use of a class of prior distributions which parameterizes how accurate the expert's model of the environment is to develop efficient algorithms to estimate the expert's reward and subjective dynamics in high-dimensional settings. Our analysis reveals a novel insight that the estimated policy exhibits robust performance when the expert is believed (a priori) to have a highly accurate model of the environment. We verify this observation in the MuJoCo environments and show that our algorithms outperform state-of-the-art offline IRL algorithms.

{{</citation>}}


### (5/91) Local Differential Privacy in Graph Neural Networks: a Reconstruction Approach (Karuna Bhaila et al., 2023)

{{<citation>}}

Karuna Bhaila, Wen Huang, Yongkai Wu, Xintao Wu. (2023)  
**Local Differential Privacy in Graph Neural Networks: a Reconstruction Approach**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.08569v1)  

---


**ABSTRACT**  
Graph Neural Networks have achieved tremendous success in modeling complex graph data in a variety of applications. However, there are limited studies investigating privacy protection in GNNs. In this work, we propose a learning framework that can provide node privacy at the user level, while incurring low utility loss. We focus on a decentralized notion of Differential Privacy, namely Local Differential Privacy, and apply randomization mechanisms to perturb both feature and label data at the node level before the data is collected by a central server for model training. Specifically, we investigate the application of randomization mechanisms in high-dimensional feature settings and propose an LDP protocol with strict privacy guarantees. Based on frequency estimation in statistical analysis of randomized data, we develop reconstruction methods to approximate features and labels from perturbed data. We also formulate this learning framework to utilize frequency estimates of graph clusters to supervise the training procedure at a sub-graph level. Extensive experiments on real-world and semi-synthetic datasets demonstrate the validity of our proposed model.

{{</citation>}}


### (6/91) Deep Reinforcement Learning for Efficient and Fair Allocation of Health Care Resources (Yikuan Li et al., 2023)

{{<citation>}}

Yikuan Li, Chengsheng Mao, Kaixuan Huang, Hanyin Wang, Zheng Yu, Mengdi Wang, Yuan Luo. (2023)  
**Deep Reinforcement Learning for Efficient and Fair Allocation of Health Care Resources**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.08560v1)  

---


**ABSTRACT**  
Scarcity of health care resources could result in the unavoidable consequence of rationing. For example, ventilators are often limited in supply, especially during public health emergencies or in resource-constrained health care settings, such as amid the pandemic of COVID-19. Currently, there is no universally accepted standard for health care resource allocation protocols, resulting in different governments prioritizing patients based on various criteria and heuristic-based protocols. In this study, we investigate the use of reinforcement learning for critical care resource allocation policy optimization to fairly and effectively ration resources. We propose a transformer-based deep Q-network to integrate the disease progression of individual patients and the interaction effects among patients during the critical care resource allocation. We aim to improve both fairness of allocation and overall patient outcomes. Our experiments demonstrate that our method significantly reduces excess deaths and achieves a more equitable distribution under different levels of ventilator shortage, when compared to existing severity-based and comorbidity-based methods in use by different governments. Our source code is included in the supplement and will be released on Github upon publication.

{{</citation>}}


### (7/91) Scaling Laws for Sparsely-Connected Foundation Models (Elias Frantar et al., 2023)

{{<citation>}}

Elias Frantar, Carlos Riquelme, Neil Houlsby, Dan Alistarh, Utku Evci. (2023)  
**Scaling Laws for Sparsely-Connected Foundation Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: T5, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.08520v1)  

---


**ABSTRACT**  
We explore the impact of parameter sparsity on the scaling behavior of Transformers trained on massive datasets (i.e., "foundation models"), in both vision and language domains. In this setting, we identify the first scaling law describing the relationship between weight sparsity, number of non-zero parameters, and amount of training data, which we validate empirically across model and data scales; on ViT/JFT-4B and T5/C4. These results allow us to characterize the "optimal sparsity", the sparsity level which yields the best performance for a given effective model size and training budget. For a fixed number of non-zero parameters, we identify that the optimal sparsity increases with the amount of data used for training. We also extend our study to different sparsity structures (such as the hardware-friendly n:m pattern) and strategies (such as starting from a pretrained dense model). Our findings shed light on the power and limitations of weight sparsity across various parameter and computational settings, offering both theoretical understanding and practical implications for leveraging sparsity towards computational efficiency improvements.

{{</citation>}}


### (8/91) P-ROCKET: Pruning Random Convolution Kernels for Time Series Classification (Shaowu Chen et al., 2023)

{{<citation>}}

Shaowu Chen, Weize Sun, Lei Huang, Xiaopeng Li, Qingyuan Wang, Deepu John. (2023)  
**P-ROCKET: Pruning Random Convolution Kernels for Time Series Classification**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Pruning, Time Series  
[Paper Link](http://arxiv.org/abs/2309.08499v1)  

---


**ABSTRACT**  
In recent years, two time series classification models, ROCKET and MINIROCKET, have attracted much attention for their low training cost and state-of-the-art accuracy. Utilizing random 1-D convolutional kernels without training, ROCKET and MINIROCKET can rapidly extract features from time series data, allowing for the efficient fitting of linear classifiers. However, to comprehensively capture useful features, a large number of random kernels are required, which is incompatible for resource-constrained devices. Therefore, a heuristic evolutionary algorithm named S-ROCKET is devised to recognize and prune redundant kernels. Nevertheless, the inherent nature of evolutionary algorithms renders the evaluation of kernels within S-ROCKET an unacceptable time-consuming process. In this paper, diverging from S-ROCKET, which directly evaluates random kernels with nonsignificant differences, we remove kernels from a feature selection perspective by eliminating associating connections in the sequential classification layer. To this end, we start by formulating the pruning challenge as a Group Elastic Net classification problem and employ the ADMM method to arrive at a solution. Sequentially, we accelerate the aforementioned time-consuming solving process by bifurcating the $l_{2,1}$ and $l_2$ regularizations into two sequential stages and solve them separately, which ultimately forms our core algorithm, named P-ROCKET. Stage 1 of P-ROCKET employs group-wise regularization similarly to our initial ADMM-based Algorithm, but introduces dynamically varying penalties to greatly accelerate the process. To mitigate overfitting, Stage 2 of P-ROCKET implements element-wise regularization to refit a linear classifier, utilizing the retained features.

{{</citation>}}


### (9/91) FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning (Hongyu Zhang et al., 2023)

{{<citation>}}

Hongyu Zhang, Dongyi Zheng, Xu Yang, Jiyuan Feng, Qing Liao. (2023)  
**FedDCSR: Federated Cross-domain Sequential Recommendation via Disentangled Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.08420v1)  

---


**ABSTRACT**  
Cross-domain Sequential Recommendation (CSR) which leverages user sequence data from multiple domains has received extensive attention in recent years. However, the existing CSR methods require sharing origin user data across domains, which violates the General Data Protection Regulation (GDPR). Thus, it is necessary to combine federated learning (FL) and CSR to fully utilize knowledge from different domains while preserving data privacy. Nonetheless, the sequence feature heterogeneity across different domains significantly impacts the overall performance of FL. In this paper, we propose FedDCSR, a novel federated cross-domain sequential recommendation framework via disentangled representation learning. Specifically, to address the sequence feature heterogeneity across domains, we introduce an approach called inter-intra domain sequence representation disentanglement (SRD) to disentangle the user sequence features into domain-shared and domain-exclusive features. In addition, we design an intra domain contrastive infomax (CIM) strategy to learn richer domain-exclusive features of users by performing data augmentation on user sequences. Extensive experiments on three real-world scenarios demonstrate that FedDCSR achieves significant improvements over existing baselines.

{{</citation>}}


### (10/91) A Unified View Between Tensor Hypergraph Neural Networks And Signal Denoising (Fuli Wang et al., 2023)

{{<citation>}}

Fuli Wang, Karelia Pena-Pena, Wei Qian, Gonzalo R. Arce. (2023)  
**A Unified View Between Tensor Hypergraph Neural Networks And Signal Denoising**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2309.08385v1)  

---


**ABSTRACT**  
Hypergraph Neural networks (HyperGNNs) and hypergraph signal denoising (HyperGSD) are two fundamental topics in higher-order network modeling. Understanding the connection between these two domains is particularly useful for designing novel HyperGNNs from a HyperGSD perspective, and vice versa. In particular, the tensor-hypergraph convolutional network (T-HGCN) has emerged as a powerful architecture for preserving higher-order interactions on hypergraphs, and this work shows an equivalence relation between a HyperGSD problem and the T-HGCN. Inspired by this intriguing result, we further design a tensor-hypergraph iterative network (T-HGIN) based on the HyperGSD problem, which takes advantage of a multi-step updating scheme in every single layer. Numerical experiments are conducted to show the promising applications of the proposed T-HGIN approach.

{{</citation>}}


### (11/91) VERSE: Virtual-Gradient Aware Streaming Lifelong Learning with Anytime Inference (Soumya Banerjee et al., 2023)

{{<citation>}}

Soumya Banerjee, Vinay K. Verma, Avideep Mukherjee, Deepak Gupta, Vinay P. Namboodiri, Piyush Rai. (2023)  
**VERSE: Virtual-Gradient Aware Streaming Lifelong Learning with Anytime Inference**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.08227v1)  

---


**ABSTRACT**  
Lifelong learning, also referred to as continual learning, is the problem of training an AI agent continuously while also preventing it from forgetting its previously acquired knowledge. Most of the existing methods primarily focus on lifelong learning within a static environment and lack the ability to mitigate forgetting in a quickly-changing dynamic environment. Streaming lifelong learning is a challenging setting of lifelong learning with the goal of continuous learning in a dynamic non-stationary environment without forgetting. We introduce a novel approach to lifelong learning, which is streaming, requires a single pass over the data, can learn in a class-incremental manner, and can be evaluated on-the-fly (anytime inference). To accomplish these, we propose virtual gradients for continual representation learning to prevent catastrophic forgetting and leverage an exponential-moving-average-based semantic memory to further enhance performance. Extensive experiments on diverse datasets demonstrate our method's efficacy and superior performance over existing methods.

{{</citation>}}


### (12/91) Unveiling Invariances via Neural Network Pruning (Derek Xu et al., 2023)

{{<citation>}}

Derek Xu, Yizhou Sun, Wei Wang. (2023)  
**Unveiling Invariances via Neural Network Pruning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2309.08171v1)  

---


**ABSTRACT**  
Invariance describes transformations that do not alter data's underlying semantics. Neural networks that preserve natural invariance capture good inductive biases and achieve superior performance. Hence, modern networks are handcrafted to handle well-known invariances (ex. translations). We propose a framework to learn novel network architectures that capture data-dependent invariances via pruning. Our learned architectures consistently outperform dense neural networks on both vision and tabular datasets in both efficiency and effectiveness. We demonstrate our framework on multiple deep learning models across 3 vision and 40 tabular datasets.

{{</citation>}}


### (13/91) Supervised Stochastic Neighbor Embedding Using Contrastive Learning (Yi Zhang, 2023)

{{<citation>}}

Yi Zhang. (2023)  
**Supervised Stochastic Neighbor Embedding Using Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning, Embedding  
[Paper Link](http://arxiv.org/abs/2309.08077v1)  

---


**ABSTRACT**  
Stochastic neighbor embedding (SNE) methods $t$-SNE, UMAP are two most popular dimensionality reduction methods for data visualization. Contrastive learning, especially self-supervised contrastive learning (SSCL), has showed great success in embedding features from unlabeled data. The conceptual connection between SNE and SSCL has been exploited. In this work, within the scope of preserving neighboring information of a dataset, we extend the self-supervised contrastive approach to the fully-supervised setting, allowing us to effectively leverage label information. Clusters of samples belonging to the same class are pulled together in low-dimensional embedding space, while simultaneously pushing apart clusters of samples from different classes.

{{</citation>}}


## cs.CL (27)



### (14/91) 'Merge Conflicts!' Exploring the Impacts of External Distractors to Parametric Knowledge Graphs (Cheng Qian et al., 2023)

{{<citation>}}

Cheng Qian, Xinran Zhao, Sherry Tongshuang Wu. (2023)  
**'Merge Conflicts!' Exploring the Impacts of External Distractors to Parametric Knowledge Graphs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.08594v1)  

---


**ABSTRACT**  
Large language models (LLMs) acquire extensive knowledge during pre-training, known as their parametric knowledge. However, in order to remain up-to-date and align with human instructions, LLMs inevitably require external knowledge during their interactions with users. This raises a crucial question: How will LLMs respond when external knowledge interferes with their parametric knowledge? To investigate this question, we propose a framework that systematically elicits LLM parametric knowledge and introduces external knowledge. Specifically, we uncover the impacts by constructing a parametric knowledge graph to reveal the different knowledge structures of LLMs, and introduce external knowledge through distractors of varying degrees, methods, positions, and formats. Our experiments on both black-box and open-source models demonstrate that LLMs tend to produce responses that deviate from their parametric knowledge, particularly when they encounter direct conflicts or confounding changes of information within detailed contexts. We also find that while LLMs are sensitive to the veracity of external knowledge, they can still be distracted by unrelated information. These findings highlight the risk of hallucination when integrating external knowledge, even indirectly, during interactions with current LLMs. All the data and results are publicly available.

{{</citation>}}


### (15/91) Are Multilingual LLMs Culturally-Diverse Reasoners? An Investigation into Multicultural Proverbs and Sayings (Chen Cecilia Liu et al., 2023)

{{<citation>}}

Chen Cecilia Liu, Fajri Koto, Timothy Baldwin, Iryna Gurevych. (2023)  
**Are Multilingual LLMs Culturally-Diverse Reasoners? An Investigation into Multicultural Proverbs and Sayings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2309.08591v1)  

---


**ABSTRACT**  
Large language models (LLMs) are highly adept at question answering and reasoning tasks, but when reasoning in situational context, human expectations vary depending on the relevant cultural common ground. As human languages are associated with diverse cultures, LLMs should also be culturally-diverse reasoners. In this paper, we study the ability of a wide range of state-of-the-art multilingual LLMs (mLLMs) to reason with proverbs and sayings in a conversational context. Our experiments reveal that: (1) mLLMs 'knows' limited proverbs and memorizing proverbs does not mean understanding them within a conversational context; (2) mLLMs struggle to reason with figurative proverbs and sayings, and when asked to select the wrong answer (instead of asking it to select the correct answer); and (3) there is a "culture gap" in mLLMs when reasoning about proverbs and sayings translated from other languages. We construct and release our evaluation dataset MAPS (MulticultrAl Proverbs and Sayings) for proverb understanding with conversational context for six different languages.

{{</citation>}}


### (16/91) Neural Machine Translation Models Can Learn to be Few-shot Learners (Raphael Reinauer et al., 2023)

{{<citation>}}

Raphael Reinauer, Patrick Simianer, Kaden Uhlig, Johannes E. M. Mosig, Joern Wuebker. (2023)  
**Neural Machine Translation Models Can Learn to be Few-shot Learners**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2309.08590v1)  

---


**ABSTRACT**  
The emergent ability of Large Language Models to use a small number of examples to learn to perform in novel domains and tasks, also called in-context learning (ICL). In this work, we show that a much smaller model can be trained to perform ICL by fine-tuning towards a specialized training objective, exemplified on the task of domain adaptation for neural machine translation. With this capacity for ICL, the model can take advantage of relevant few-shot examples to adapt its output towards the domain. We compare the quality of this domain adaptation to traditional supervised techniques and ICL with a 40B-parameter Large Language Model. Our approach allows efficient batch inference on a mix of domains and outperforms state-of-the-art baselines in terms of both translation quality and immediate adaptation rate, i.e. the ability to reproduce a specific term after being shown a single example.

{{</citation>}}


### (17/91) ICLEF: In-Context Learning with Expert Feedback for Explainable Style Transfer (Arkadiy Saakyan et al., 2023)

{{<citation>}}

Arkadiy Saakyan, Smaranda Muresan. (2023)  
**ICLEF: In-Context Learning with Expert Feedback for Explainable Style Transfer**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-3.5, GPT-4, Style Transfer  
[Paper Link](http://arxiv.org/abs/2309.08583v1)  

---


**ABSTRACT**  
While state-of-the-art language models excel at the style transfer task, current work does not address explainability of style transfer systems. Explanations could be generated using large language models such as GPT-3.5 and GPT-4, but the use of such complex systems is inefficient when smaller, widely distributed, and transparent alternatives are available. We propose a framework to augment and improve a formality style transfer dataset with explanations via model distillation from ChatGPT. To further refine the generated explanations, we propose a novel way to incorporate scarce expert human feedback using in-context learning (ICLEF: In-Context Learning from Expert Feedback) by prompting ChatGPT to act as a critic to its own outputs. We use the resulting dataset of 9,960 explainable formality style transfer instances (e-GYAFC) to show that current openly distributed instruction-tuned models (and, in some settings, ChatGPT) perform poorly on the task, and that fine-tuning on our high-quality dataset leads to significant improvements as shown by automatic evaluation. In human evaluation, we show that models much smaller than ChatGPT fine-tuned on our data align better with expert preferences. Finally, we discuss two potential applications of models fine-tuned on the explainable style transfer task: interpretable authorship verification and interpretable adversarial attacks on AI-generated text detectors.

{{</citation>}}


### (18/91) Casteist but Not Racist? Quantifying Disparities in Large Language Model Bias between India and the West (Khyati Khandelwal et al., 2023)

{{<citation>}}

Khyati Khandelwal, Manuel Tonneau, Andrew M. Bean, Hannah Rose Kirk, Scott A. Hale. (2023)  
**Casteist but Not Racist? Quantifying Disparities in Large Language Model Bias between India and the West**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: Bias, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2309.08573v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), now used daily by millions of users, can encode societal biases, exposing their users to representational harms. A large body of scholarship on LLM bias exists but it predominantly adopts a Western-centric frame and attends comparatively less to bias levels and potential harms in the Global South. In this paper, we quantify stereotypical bias in popular LLMs according to an Indian-centric frame and compare bias levels between the Indian and Western contexts. To do this, we develop a novel dataset which we call Indian-BhED (Indian Bias Evaluation Dataset), containing stereotypical and anti-stereotypical examples for caste and religion contexts. We find that the majority of LLMs tested are strongly biased towards stereotypes in the Indian context, especially as compared to the Western context. We finally investigate Instruction Prompting as a simple intervention to mitigate such bias and find that it significantly reduces both stereotypical and anti-stereotypical biases in the majority of cases for GPT-3.5. The findings of this work highlight the need for including more diverse voices when evaluating LLMs.

{{</citation>}}


### (19/91) How Transferable are Attribute Controllers on Pretrained Multilingual Translation Models? (Danni Liu et al., 2023)

{{<citation>}}

Danni Liu, Jan Niehues. (2023)  
**How Transferable are Attribute Controllers on Pretrained Multilingual Translation Models?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2309.08565v1)  

---


**ABSTRACT**  
Customizing machine translation models to comply with fine-grained attributes such as formality has seen tremendous progress recently. However, current approaches mostly rely on at least some supervised data with attribute annotation. Data scarcity therefore remains a bottleneck to democratizing such customization possibilities to a wider range of languages, lower-resource ones in particular. Given recent progress in pretrained massively multilingual translation models, we use them as a foundation to transfer the attribute controlling capabilities to languages without supervised data. In this work, we present a comprehensive analysis of transferring attribute controllers based on a pretrained NLLB-200 model. We investigate both training- and inference-time control techniques under various data scenarios, and uncover their relative strengths and weaknesses in zero-shot performance and domain robustness. We show that both paradigms are complementary, as shown by consistent improvements on 5 zero-shot directions. Moreover, a human evaluation on a real low-resource language, Bengali, confirms our findings on zero-shot transfer to new target languages. The code is $\href{https://github.com/dannigt/attribute-controller-transfer}{\text{here}}$.

{{</citation>}}


### (20/91) Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers (Qingyan Guo et al., 2023)

{{<citation>}}

Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, Yujiu Yang. (2023)  
**Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2309.08532v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) excel in various tasks, but they rely on carefully crafted prompts that often demand substantial human effort. To automate this process, in this paper, we propose a novel framework for discrete prompt optimization, called EvoPrompt, which borrows the idea of evolutionary algorithms (EAs) as they exhibit good performance and fast convergence. To enable EAs to work on discrete prompts, which are natural language expressions that need to be coherent and human-readable, we connect LLMs with EAs. This approach allows us to simultaneously leverage the powerful language processing capabilities of LLMs and the efficient optimization performance of EAs. Specifically, abstaining from any gradients or parameters, EvoPrompt starts from a population of prompts and iteratively generates new prompts with LLMs based on the evolutionary operators, improving the population based on the development set. We optimize prompts for both closed- and open-source LLMs including GPT-3.5 and Alpaca, on 9 datasets spanning language understanding and generation tasks. EvoPrompt significantly outperforms human-engineered prompts and existing methods for automatic prompt generation by up to 25% and 14% respectively. Furthermore, EvoPrompt demonstrates that connecting LLMs with EAs creates synergies, which could inspire further research on the combination of LLMs and conventional algorithms.

{{</citation>}}


### (21/91) HealthFC: A Dataset of Health Claims for Evidence-Based Medical Fact-Checking (Juraj Vladika et al., 2023)

{{<citation>}}

Juraj Vladika, Phillip Schneider, Florian Matthes. (2023)  
**HealthFC: A Dataset of Health Claims for Evidence-Based Medical Fact-Checking**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Fact-Checking  
[Paper Link](http://arxiv.org/abs/2309.08503v1)  

---


**ABSTRACT**  
Seeking health-related advice on the internet has become a common practice in the digital era. Determining the trustworthiness of medical claims found online and finding appropriate evidence for this information is increasingly challenging. Fact-checking has emerged as an approach to assess the veracity of factual claims using evidence from credible knowledge sources. To help advance the automation of this task, in this paper, we introduce a novel dataset of 750 health-related claims, labeled for veracity by medical experts and backed with evidence from appropriate clinical studies. We provide an analysis of the dataset, highlighting its characteristics and challenges. The dataset can be used for Machine Learning tasks related to automated fact-checking such as evidence retrieval, veracity prediction, and explanation generation. For this purpose, we provide baseline models based on different approaches, examine their performance, and discuss the findings.

{{</citation>}}


### (22/91) Using Large Language Models for Knowledge Engineering (LLMKE): A Case Study on Wikidata (Bohui Zhang et al., 2023)

{{<citation>}}

Bohui Zhang, Ioannis Reklos, Nitisha Jain, Albert Meroño Peñuela, Elena Simperl. (2023)  
**Using Large Language Models for Knowledge Engineering (LLMKE): A Case Study on Wikidata**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.08491v1)  

---


**ABSTRACT**  
In this work, we explore the use of Large Language Models (LLMs) for knowledge engineering tasks in the context of the ISWC 2023 LM-KBC Challenge. For this task, given subject and relation pairs sourced from Wikidata, we utilize pre-trained LLMs to produce the relevant objects in string format and link them to their respective Wikidata QIDs. We developed a pipeline using LLMs for Knowledge Engineering (LLMKE), combining knowledge probing and Wikidata entity mapping. The method achieved a macro-averaged F1-score of 0.701 across the properties, with the scores varying from 1.00 to 0.328. These results demonstrate that the knowledge of LLMs varies significantly depending on the domain and that further experimentation is required to determine the circumstances under which LLMs can be used for automatic Knowledge Base (e.g., Wikidata) completion and correction. The investigation of the results also suggests the promising contribution of LLMs in collaborative knowledge engineering. LLMKE won Track 2 of the challenge. The implementation is available at https://github.com/bohuizhang/LLMKE.

{{</citation>}}


### (23/91) SilverRetriever: Advancing Neural Passage Retrieval for Polish Question Answering (Piotr Rybak et al., 2023)

{{<citation>}}

Piotr Rybak, Maciej Ogrodniczuk. (2023)  
**SilverRetriever: Advancing Neural Passage Retrieval for Polish Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Question Answering  
[Paper Link](http://arxiv.org/abs/2309.08469v1)  

---


**ABSTRACT**  
Modern open-domain question answering systems often rely on accurate and efficient retrieval components to find passages containing the facts necessary to answer the question. Recently, neural retrievers have gained popularity over lexical alternatives due to their superior performance. However, most of the work concerns popular languages such as English or Chinese. For others, such as Polish, few models are available. In this work, we present SilverRetriever, a neural retriever for Polish trained on a diverse collection of manually or weakly labeled datasets. SilverRetriever achieves much better results than other Polish models and is competitive with larger multilingual models. Together with the model, we open-source five new passage retrieval datasets.

{{</citation>}}


### (24/91) Advancing the Evaluation of Traditional Chinese Language Models: Towards a Comprehensive Benchmark Suite (Chan-Jan Hsu et al., 2023)

{{<citation>}}

Chan-Jan Hsu, Chang-Le Liu, Feng-Ting Liao, Po-Chun Hsu, Yi-Chang Chen, Da-shan Shiu. (2023)  
**Advancing the Evaluation of Traditional Chinese Language Models: Towards a Comprehensive Benchmark Suite**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2309.08448v1)  

---


**ABSTRACT**  
The evaluation of large language models is an essential task in the field of language understanding and generation. As language models continue to advance, the need for effective benchmarks to assess their performance has become imperative. In the context of Traditional Chinese, there is a scarcity of comprehensive and diverse benchmarks to evaluate the capabilities of language models, despite the existence of certain benchmarks such as DRCD, TTQA, CMDQA, and FGC dataset. To address this gap, we propose a novel set of benchmarks that leverage existing English datasets and are tailored to evaluate language models in Traditional Chinese. These benchmarks encompass a wide range of tasks, including contextual question-answering, summarization, classification, and table understanding. The proposed benchmarks offer a comprehensive evaluation framework, enabling the assessment of language models' capabilities across different tasks. In this paper, we evaluate the performance of GPT-3.5, Taiwan-LLaMa-v1.0, and Model 7-C, our proprietary model, on these benchmarks. The evaluation results highlight that our model, Model 7-C, achieves performance comparable to GPT-3.5 with respect to a part of the evaluated capabilities. In an effort to advance the evaluation of language models in Traditional Chinese and stimulate further research in this field, we have open-sourced our benchmark and opened the model for trial.

{{</citation>}}


### (25/91) Unleashing Potential of Evidence in Knowledge-Intensive Dialogue Generation (Xianjie Wu et al., 2023)

{{<citation>}}

Xianjie Wu, Jian Yang, Tongliang Li, Di Liang, Shiwei Zhang, Yiyang Du, Zhoujun Li. (2023)  
**Unleashing Potential of Evidence in Knowledge-Intensive Dialogue Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2309.08380v1)  

---


**ABSTRACT**  
Incorporating external knowledge into dialogue generation (KIDG) is crucial for improving the correctness of response, where evidence fragments serve as knowledgeable snippets supporting the factual dialogue replies. However, introducing irrelevant content often adversely impacts reply quality and easily leads to hallucinated responses. Prior work on evidence retrieval and integration in dialogue systems falls short of fully leveraging existing evidence since the model fails to locate useful fragments accurately and overlooks hidden evidence labels within the KIDG dataset. To fully Unleash the potential of evidence, we propose a framework to effectively incorporate Evidence in knowledge-Intensive Dialogue Generation (u-EIDG). Specifically, we introduce an automatic evidence generation framework that harnesses the power of Large Language Models (LLMs) to mine reliable evidence veracity labels from unlabeled data. By utilizing these evidence labels, we train a reliable evidence indicator to effectively identify relevant evidence from retrieved passages. Furthermore, we propose an evidence-augmented generator with an evidence-focused attention mechanism, which allows the model to concentrate on evidenced segments. Experimental results on MultiDoc2Dial demonstrate the efficacy of evidential label augmentation and refined attention mechanisms in improving model performance. Further analysis confirms that the proposed method outperforms other baselines (+3~+5 points) regarding coherence and factual consistency.

{{</citation>}}


### (26/91) Headless Language Models: Learning without Predicting with Contrastive Weight Tying (Nathan Godey et al., 2023)

{{<citation>}}

Nathan Godey, Éric de la Clergerie, Benoît Sagot. (2023)  
**Headless Language Models: Learning without Predicting with Contrastive Weight Tying**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLUE, Language Model  
[Paper Link](http://arxiv.org/abs/2309.08351v1)  

---


**ABSTRACT**  
Self-supervised pre-training of language models usually consists in predicting probability distributions over extensive token vocabularies. In this study, we propose an innovative method that shifts away from probability prediction and instead focuses on reconstructing input embeddings in a contrastive fashion via Constrastive Weight Tying (CWT). We apply this approach to pretrain Headless Language Models in both monolingual and multilingual contexts. Our method offers practical advantages, substantially reducing training computational requirements by up to 20 times, while simultaneously enhancing downstream performance and data efficiency. We observe a significant +1.6 GLUE score increase and a notable +2.7 LAMBADA accuracy improvement compared to classical LMs within similar compute budgets.

{{</citation>}}


### (27/91) Data Distribution Bottlenecks in Grounding Language Models to Knowledge Bases (Yiheng Shu et al., 2023)

{{<citation>}}

Yiheng Shu, Zhiwei Yu. (2023)  
**Data Distribution Bottlenecks in Grounding Language Models to Knowledge Bases**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2309.08345v1)  

---


**ABSTRACT**  
Language models (LMs) have already demonstrated remarkable abilities in understanding and generating both natural and formal language. Despite these advances, their integration with real-world environments such as large-scale knowledge bases (KBs) remains an underdeveloped area, affecting applications such as semantic parsing and indulging in "hallucinated" information. This paper is an experimental investigation aimed at uncovering the robustness challenges that LMs encounter when tasked with knowledge base question answering (KBQA). The investigation covers scenarios with inconsistent data distribution between training and inference, such as generalization to unseen domains, adaptation to various language variations, and transferability across different datasets. Our comprehensive experiments reveal that even when employed with our proposed data augmentation techniques, advanced small and large language models exhibit poor performance in various dimensions. While the LM is a promising technology, the robustness of the current form in dealing with complex environments is fragile and of limited practicality because of the data distribution issue. This calls for future research on data collection and LM learning paradims.

{{</citation>}}


### (28/91) Self-Consistent Narrative Prompts on Abductive Natural Language Inference (Chunkit Chan et al., 2023)

{{<citation>}}

Chunkit Chan, Xin Liu, Tsz Ho Chan, Jiayang Cheng, Yangqiu Song, Ginny Wong, Simon See. (2023)  
**Self-Consistent Narrative Prompts on Abductive Natural Language Inference**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLI, Natural Language Inference  
[Paper Link](http://arxiv.org/abs/2309.08303v1)  

---


**ABSTRACT**  
Abduction has long been seen as crucial for narrative comprehension and reasoning about everyday situations. The abductive natural language inference ($\alpha$NLI) task has been proposed, and this narrative text-based task aims to infer the most plausible hypothesis from the candidates given two observations. However, the inter-sentential coherence and the model consistency have not been well exploited in the previous works on this task. In this work, we propose a prompt tuning model $\alpha$-PACE, which takes self-consistency and inter-sentential coherence into consideration. Besides, we propose a general self-consistent framework that considers various narrative sequences (e.g., linear narrative and reverse chronology) for guiding the pre-trained language model in understanding the narrative context of input. We conduct extensive experiments and thorough ablation studies to illustrate the necessity and effectiveness of $\alpha$-PACE. The performance of our method shows significant improvement against extensive competitive baselines.

{{</citation>}}


### (29/91) Structural Self-Supervised Objectives for Transformers (Luca Di Liello, 2023)

{{<citation>}}

Luca Di Liello. (2023)  
**Structural Self-Supervised Objectives for Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: BERT, Fact Verification, Language Model, NLP, QA, Self-Supervised, Summarization, T5, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.08272v1)  

---


**ABSTRACT**  
This thesis focuses on improving the pre-training of natural language models using unsupervised raw data to make them more efficient and aligned with downstream applications.   In the first part, we introduce three alternative pre-training objectives to BERT's Masked Language Modeling (MLM), namely Random Token Substitution (RTS), Cluster-based Random Token Substitution (C-RTS), and Swapped Language Modeling (SLM). These objectives involve token swapping instead of masking, with RTS and C-RTS aiming to predict token originality and SLM predicting the original token values. Results show that RTS and C-RTS require less pre-training time while maintaining performance comparable to MLM. Surprisingly, SLM outperforms MLM on certain tasks despite using the same computational budget.   In the second part, we proposes self-supervised pre-training tasks that align structurally with downstream applications, reducing the need for labeled data. We use large corpora like Wikipedia and CC-News to train models to recognize if text spans originate from the same paragraph or document in several ways. By doing continuous pre-training, starting from existing models like RoBERTa, ELECTRA, DeBERTa, BART, and T5, we demonstrate significant performance improvements in tasks like Fact Verification, Answer Sentence Selection, and Summarization. These improvements are especially pronounced when limited annotation data is available. The proposed objectives also achieve state-of-the-art results on various benchmark datasets, including FEVER (dev set), ASNQ, WikiQA, and TREC-QA, as well as enhancing the quality of summaries. Importantly, these techniques can be easily integrated with other methods without altering the internal structure of Transformer models, making them versatile for various NLP applications.

{{</citation>}}


### (30/91) Investigating Answerability of LLMs for Long-Form Question Answering (Meghana Moorthy Bhat et al., 2023)

{{<citation>}}

Meghana Moorthy Bhat, Rui Meng, Ye Liu, Yingbo Zhou, Semih Yavuz. (2023)  
**Investigating Answerability of LLMs for Long-Form Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.08210v1)  

---


**ABSTRACT**  
As we embark on a new era of LLMs, it becomes increasingly crucial to understand their capabilities, limitations, and differences. Toward making further progress in this direction, we strive to build a deeper understanding of the gaps between massive LLMs (e.g., ChatGPT) and smaller yet effective open-source LLMs and their distilled counterparts. To this end, we specifically focus on long-form question answering (LFQA) because it has several practical and impactful applications (e.g., troubleshooting, customer service, etc.) yet is still understudied and challenging for LLMs. We propose a question-generation method from abstractive summaries and show that generating follow-up questions from summaries of long documents can create a challenging setting for LLMs to reason and infer from long contexts. Our experimental results confirm that: (1) our proposed method of generating questions from abstractive summaries pose a challenging setup for LLMs and shows performance gaps between LLMs like ChatGPT and open-source LLMs (Alpaca, Llama) (2) open-source LLMs exhibit decreased reliance on context for generated questions from the original document, but their generation capabilities drop significantly on generated questions from summaries -- especially for longer contexts (>1024 tokens)

{{</citation>}}


### (31/91) Encoded Summarization: Summarizing Documents into Continuous Vector Space for Legal Case Retrieval (Vu Tran et al., 2023)

{{<citation>}}

Vu Tran, Minh Le Nguyen, Satoshi Tojo, Ken Satoh. (2023)  
**Encoded Summarization: Summarizing Documents into Continuous Vector Space for Legal Case Retrieval**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Legal, Summarization  
[Paper Link](http://arxiv.org/abs/2309.08187v1)  

---


**ABSTRACT**  
We present our method for tackling a legal case retrieval task by introducing our method of encoding documents by summarizing them into continuous vector space via our phrase scoring framework utilizing deep neural networks. On the other hand, we explore the benefits from combining lexical features and latent features generated with neural networks. Our experiments show that lexical features and latent features generated with neural networks complement each other to improve the retrieval system performance. Furthermore, our experimental results suggest the importance of case summarization in different aspects: using provided summaries and performing encoded summarization. Our approach achieved F1 of 65.6% and 57.6% on the experimental datasets of legal case retrieval tasks.

{{</citation>}}


### (32/91) Multilingual Sentence-Level Semantic Search using Meta-Distillation Learning (Meryem M'hamdi et al., 2023)

{{<citation>}}

Meryem M'hamdi, Jonathan May, Franck Dernoncourt, Trung Bui, Seunghyun Yoon. (2023)  
**Multilingual Sentence-Level Semantic Search using Meta-Distillation Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2309.08185v1)  

---


**ABSTRACT**  
Multilingual semantic search is the task of retrieving relevant contents to a query expressed in different language combinations. This requires a better semantic understanding of the user's intent and its contextual meaning. Multilingual semantic search is less explored and more challenging than its monolingual or bilingual counterparts, due to the lack of multilingual parallel resources for this task and the need to circumvent "language bias". In this work, we propose an alignment approach: MAML-Align, specifically for low-resource scenarios. Our approach leverages meta-distillation learning based on MAML, an optimization-based Model-Agnostic Meta-Learner. MAML-Align distills knowledge from a Teacher meta-transfer model T-MAML, specialized in transferring from monolingual to bilingual semantic search, to a Student model S-MAML, which meta-transfers from bilingual to multilingual semantic search. To the best of our knowledge, we are the first to extend meta-distillation to a multilingual search application. Our empirical results show that on top of a strong baseline based on sentence transformers, our meta-distillation approach boosts the gains provided by MAML and significantly outperforms naive fine-tuning methods. Furthermore, multilingual meta-distillation learning improves generalization even to unseen languages.

{{</citation>}}


### (33/91) Using Large Language Model to Solve and Explain Physics Word Problems Approaching Human Level (Jingzhe Ding et al., 2023)

{{<citation>}}

Jingzhe Ding, Yan Cen, Xinyuan Wei. (2023)  
**Using Large Language Model to Solve and Explain Physics Word Problems Approaching Human Level**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2309.08182v1)  

---


**ABSTRACT**  
Our work demonstrates that large language model (LLM) pre-trained on texts can not only solve pure math word problems, but also physics word problems-problems to be solved by calculation and inference based on some prior physical knowledge. We collect and annotate the first physics word problem dataset-PhysQA, which contains over 1000 junior high school physics word problems (on Kinematics, Mass&Density, Mechanics, Heat, Electricity). Then we use OpenAI' s GPT3.5 to generate the answer of these problems and found that GPT3.5 could automatically solve 49.3% of the problems on zero-shot learning and 73.2% on few-shot learning. This result show that by using similar problem and its answer as prompt, LLM could solve elementary physics word problems approaching human level. Besides automatically solving problems, GPT3.5 could also summarize the knowledge or topic examined by the problem, generate the relevant explanation, and synthesis new physics word problems according tothe input problems.Our work is the first research on automatically solving, explaining and generating physics word problems of multiple types and scenes, and we gain an acceptable and state-of-art accuracy, which demonstrates the potential of LLM's further application in the field of secondary education.

{{</citation>}}


### (34/91) Large Language Models for Failure Mode Classification: An Investigation (Michael Stewart et al., 2023)

{{<citation>}}

Michael Stewart, Melinda Hodkiewicz, Sirui Li. (2023)  
**Large Language Models for Failure Mode Classification: An Investigation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2309.08181v1)  

---


**ABSTRACT**  
In this paper we present the first investigation into the effectiveness of Large Language Models (LLMs) for Failure Mode Classification (FMC). FMC, the task of automatically labelling an observation with a corresponding failure mode code, is a critical task in the maintenance domain as it reduces the need for reliability engineers to spend their time manually analysing work orders. We detail our approach to prompt engineering to enable an LLM to predict the failure mode of a given observation using a restricted code list. We demonstrate that the performance of a GPT-3.5 model (F1=0.80) fine-tuned on annotated data is a significant improvement over a currently available text classification model (F1=0.60) trained on the same annotated data set. The fine-tuned model also outperforms the out-of-the box GPT-3.5 (F1=0.46). This investigation reinforces the need for high quality fine-tuning data sets for domain-specific tasks using LLMs.

{{</citation>}}


### (35/91) FedJudge: Federated Legal Large Language Model (Linan Yue et al., 2023)

{{<citation>}}

Linan Yue, Qi Liu, Yichao Du, Weibo Gao, Ye Liu, Fangzhou Yao. (2023)  
**FedJudge: Federated Legal Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2309.08173v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have gained prominence in the field of Legal Intelligence, offering potential applications in assisting legal professionals and laymen. However, the centralized training of these Legal LLMs raises data privacy concerns, as legal data is distributed among various institutions containing sensitive individual information. This paper addresses this challenge by exploring the integration of Legal LLMs with Federated Learning (FL) methodologies. By employing FL, Legal LLMs can be fine-tuned locally on devices or clients, and their parameters are aggregated and distributed on a central server, ensuring data privacy without directly sharing raw data. However, computation and communication overheads hinder the full fine-tuning of LLMs under the FL setting. Moreover, the distribution shift of legal data reduces the effectiveness of FL methods. To this end, in this paper, we propose the first Federated Legal Large Language Model (FedJudge) framework, which fine-tunes Legal LLMs efficiently and effectively. Specifically, FedJudge utilizes parameter-efficient fine-tuning methods to update only a few additional parameters during the FL training. Besides, we explore the continual learning methods to preserve the global model's important parameters when training local clients to mitigate the problem of data shifts. Extensive experimental results on three real-world datasets clearly validate the effectiveness of FedJudge. Code is released at https://github.com/yuelinan/FedJudge.

{{</citation>}}


### (36/91) Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding (Jun Zhang et al., 2023)

{{<citation>}}

Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, Sharad Mehrotra. (2023)  
**Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2309.08168v1)  

---


**ABSTRACT**  
We present a novel inference scheme, self-speculative decoding, for accelerating Large Language Models (LLMs) without the need for an auxiliary model. This approach is characterized by a two-stage process: drafting and verification. The drafting stage generates draft tokens at a slightly lower quality but more quickly, which is achieved by selectively skipping certain intermediate layers during drafting Subsequently, the verification stage employs the original LLM to validate those draft output tokens in one forward pass. This process ensures the final output remains identical to that produced by the unaltered LLM, thereby maintaining output quality. The proposed method requires no additional neural network training and no extra memory footprint, making it a plug-and-play and cost-effective solution for inference acceleration. Benchmarks with LLaMA-2 and its fine-tuned models demonstrated a speedup up to 1.73$\times$.

{{</citation>}}


### (37/91) Investigating the Applicability of Self-Assessment Tests for Personality Measurement of Large Language Models (Akshat Gupta et al., 2023)

{{<citation>}}

Akshat Gupta, Xiaoyang Song, Gopala Anumanchipalli. (2023)  
**Investigating the Applicability of Self-Assessment Tests for Personality Measurement of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.08163v1)  

---


**ABSTRACT**  
As large language models (LLM) evolve in their capabilities, various recent studies have tried to quantify their behavior using psychological tools created to study human behavior. One such example is the measurement of "personality" of LLMs using personality self-assessment tests. In this paper, we take three such studies on personality measurement of LLMs that use personality self-assessment tests created to study human behavior. We use the prompts used in these three different papers to measure the personality of the same LLM. We find that all three prompts lead very different personality scores. This simple test reveals that personality self-assessment scores in LLMs depend on the subjective choice of the prompter. Since we don't know the ground truth value of personality scores for LLMs as there is no correct answer to such questions, there's no way of claiming if one prompt is more or less correct than the other. We then introduce the property of option order symmetry for personality measurement of LLMs. Since most of the self-assessment tests exist in the form of multiple choice question (MCQ) questions, we argue that the scores should also be robust to not just the prompt template but also the order in which the options are presented. This test unsurprisingly reveals that the answers to the self-assessment tests are not robust to the order of the options. These simple tests, done on ChatGPT and Llama2 models show that self-assessment personality tests created for humans are not appropriate for measuring personality in LLMs.

{{</citation>}}


### (38/91) RADE: Reference-Assisted Dialogue Evaluation for Open-Domain Dialogue (Zhengliang Shi et al., 2023)

{{<citation>}}

Zhengliang Shi, Weiwei Sun, Shuo Zhang, Zhen Zhang, Pengjie Ren, Zhaochun Ren. (2023)  
**RADE: Reference-Assisted Dialogue Evaluation for Open-Domain Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2309.08156v1)  

---


**ABSTRACT**  
Evaluating open-domain dialogue systems is challenging for reasons such as the one-to-many problem, i.e., many appropriate responses other than just the golden response. As of now, automatic evaluation methods need better consistency with humans, while reliable human evaluation can be time- and cost-intensive. To this end, we propose the Reference-Assisted Dialogue Evaluation (RADE) approach under the multi-task learning framework, which leverages the pre-created utterance as reference other than the gold response to relief the one-to-many problem. Specifically, RADE explicitly compares reference and the candidate response to predict their overall scores. Moreover, an auxiliary response generation task enhances prediction via a shared encoder. To support RADE, we extend three datasets with additional rated responses other than just a golden response by human annotation. Experiments on our three datasets and two existing benchmarks demonstrate the effectiveness of our method, where Pearson, Spearman, and Kendall correlations with human evaluation outperform state-of-the-art baselines.

{{</citation>}}


### (39/91) Unimodal Aggregation for CTC-based Speech Recognition (Ying Fang et al., 2023)

{{<citation>}}

Ying Fang, Xiaofei Li. (2023)  
**Unimodal Aggregation for CTC-based Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.08150v1)  

---


**ABSTRACT**  
This paper works on non-autoregressive automatic speech recognition. A unimodal aggregation (UMA) is proposed to segment and integrate the feature frames that belong to the same text token, and thus to learn better feature representations for text tokens. The frame-wise features and weights are both derived from an encoder. Then, the feature frames with unimodal weights are integrated and further processed by a decoder. Connectionist temporal classification (CTC) loss is applied for training. Compared to the regular CTC, the proposed method learns better feature representations and shortens the sequence length, resulting in lower recognition error and computational complexity. Experiments on three Mandarin datasets show that UMA demonstrates superior or comparable performance to other advanced non-autoregressive methods, such as self-conditioned CTC. Moreover, by integrating self-conditioned CTC into the proposed framework, the performance can be further noticeably improved.

{{</citation>}}


### (40/91) Research on Joint Representation Learning Methods for Entity Neighborhood Information and Description Information (Le Xiao et al., 2023)

{{<citation>}}

Le Xiao, Xin Shan, Yuhua Wang, Miaolei Deng. (2023)  
**Research on Joint Representation Learning Methods for Entity Neighborhood Information and Description Information**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.08100v1)  

---


**ABSTRACT**  
To address the issue of poor embedding performance in the knowledge graph of a programming design course, a joint represen-tation learning model that combines entity neighborhood infor-mation and description information is proposed. Firstly, a graph at-tention network is employed to obtain the features of entity neigh-boring nodes, incorporating relationship features to enrich the structural information. Next, the BERT-WWM model is utilized in conjunction with attention mechanisms to obtain the representation of entity description information. Finally, the final entity vector representation is obtained by combining the vector representations of entity neighborhood information and description information. Experimental results demonstrate that the proposed model achieves favorable performance on the knowledge graph dataset of the pro-gramming design course, outperforming other baseline models.

{{</citation>}}


## cs.CV (17)



### (41/91) Replacing softmax with ReLU in Vision Transformers (Mitchell Wortsman et al., 2023)

{{<citation>}}

Mitchell Wortsman, Jaehoon Lee, Justin Gilmer, Simon Kornblith. (2023)  
**Replacing softmax with ReLU in Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.08586v1)  

---


**ABSTRACT**  
Previous research observed accuracy degradation when replacing the attention softmax with a point-wise activation such as ReLU. In the context of vision transformers, we find that this degradation is mitigated when dividing by sequence length. Our experiments training small to large vision transformers on ImageNet-21k indicate that ReLU-attention can approach or match the performance of softmax-attention in terms of scaling behavior as a function of compute.

{{</citation>}}


### (42/91) Visual Speech Recognition for Low-resource Languages with Automatic Labels From Whisper Model (Jeong Hun Yeo et al., 2023)

{{<citation>}}

Jeong Hun Yeo, Minsu Kim, Shinji Watanabe, Yong Man Ro. (2023)  
**Visual Speech Recognition for Low-resource Languages with Automatic Labels From Whisper Model**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV, eess-AS  
Keywords: Low-Resource, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.08535v1)  

---


**ABSTRACT**  
This paper proposes a powerful Visual Speech Recognition (VSR) method for multiple languages, especially for low-resource languages that have a limited number of labeled data. Different from previous methods that tried to improve the VSR performance for the target language by using knowledge learned from other languages, we explore whether we can increase the amount of training data itself for the different languages without human intervention. To this end, we employ a Whisper model which can conduct both language identification and audio-based speech recognition. It serves to filter data of the desired languages and transcribe labels from the unannotated, multilingual audio-visual data pool. By comparing the performances of VSR models trained on automatic labels and the human-annotated labels, we show that we can achieve similar VSR performance to that of human-annotated labels even without utilizing human annotations. Through the automated labeling process, we label large-scale unlabeled multilingual databases, VoxCeleb2 and AVSpeech, producing 1,002 hours of data for four low VSR resource languages, French, Italian, Spanish, and Portuguese. With the automatic labels, we achieve new state-of-the-art performance on mTEDx in four languages, significantly surpassing the previous methods. The automatic labels are available online: https://github.com/JeongHun0716/Visual-Speech-Recognition-for-Low-Resource-Languages

{{</citation>}}


### (43/91) Beyond Domain Gap: Exploiting Subjectivity in Sketch-Based Person Retrieval (Kejun Lin et al., 2023)

{{<citation>}}

Kejun Lin, Zhixiang Wang, Zheng Wang, Yinqiang Zheng, Shin'ichi Satoh. (2023)  
**Beyond Domain Gap: Exploiting Subjectivity in Sketch-Based Person Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2309.08372v1)  

---


**ABSTRACT**  
Person re-identification (re-ID) requires densely distributed cameras. In practice, the person of interest may not be captured by cameras and, therefore, needs to be retrieved using subjective information (e.g., sketches from witnesses). Previous research defines this case using the sketch as sketch re-identification (Sketch re-ID) and focuses on eliminating the domain gap. Actually, subjectivity is another significant challenge. We model and investigate it by posing a new dataset with multi-witness descriptions. It features two aspects. 1) Large-scale. It contains over 4,763 sketches and 32,668 photos, making it the largest Sketch re-ID dataset. 2) Multi-perspective and multi-style. Our dataset offers multiple sketches for each identity. Witnesses' subjective cognition provides multiple perspectives on the same individual, while different artists' drawing styles provide variation in sketch styles. We further have two novel designs to alleviate the challenge of subjectivity. 1) Fusing subjectivity. We propose a non-local (NL) fusion module that gathers sketches from different witnesses for the same identity. 2) Introducing objectivity. An AttrAlign module utilizes attributes as an implicit mask to align cross-domain features. To push forward the advance of Sketch re-ID, we set three benchmarks (large-scale, multi-style, cross-style). Extensive experiments demonstrate our leading performance in these benchmarks. Dataset and Codes are publicly available at: https://github.com/Lin-Kayla/subjectivity-sketch-reid

{{</citation>}}


### (44/91) M$^3$Net: Multilevel, Mixed and Multistage Attention Network for Salient Object Detection (Yao Yuan et al., 2023)

{{<citation>}}

Yao Yuan, Pan Gao, XiaoYang Tan. (2023)  
**M$^3$Net: Multilevel, Mixed and Multistage Attention Network for Salient Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2309.08365v1)  

---


**ABSTRACT**  
Most existing salient object detection methods mostly use U-Net or feature pyramid structure, which simply aggregates feature maps of different scales, ignoring the uniqueness and interdependence of them and their respective contributions to the final prediction. To overcome these, we propose the M$^3$Net, i.e., the Multilevel, Mixed and Multistage attention network for Salient Object Detection (SOD). Firstly, we propose Multiscale Interaction Block which innovatively introduces the cross-attention approach to achieve the interaction between multilevel features, allowing high-level features to guide low-level feature learning and thus enhancing salient regions. Secondly, considering the fact that previous Transformer based SOD methods locate salient regions only using global self-attention while inevitably overlooking the details of complex objects, we propose the Mixed Attention Block. This block combines global self-attention and window self-attention, aiming at modeling context at both global and local levels to further improve the accuracy of the prediction map. Finally, we proposed a multilevel supervision strategy to optimize the aggregated feature stage-by-stage. Experiments on six challenging datasets demonstrate that the proposed M$^3$Net surpasses recent CNN and Transformer-based SOD arts in terms of four metrics. Codes are available at https://github.com/I2-Multimedia-Lab/M3Net.

{{</citation>}}


### (45/91) Continual Learning with Deep Streaming Regularized Discriminant Analysis (Joe Khawand et al., 2023)

{{<citation>}}

Joe Khawand, Peter Hanappe, David Colliaux. (2023)  
**Continual Learning with Deep Streaming Regularized Discriminant Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.08353v1)  

---


**ABSTRACT**  
Continual learning is increasingly sought after in real world machine learning applications, as it enables learning in a more human-like manner. Conventional machine learning approaches fail to achieve this, as incrementally updating the model with non-identically distributed data leads to catastrophic forgetting, where existing representations are overwritten. Although traditional continual learning methods have mostly focused on batch learning, which involves learning from large collections of labeled data sequentially, this approach is not well-suited for real-world applications where we would like new data to be integrated directly. This necessitates a paradigm shift towards streaming learning. In this paper, we propose a streaming version of regularized discriminant analysis as a solution to this challenge. We combine our algorithm with a convolutional neural network and demonstrate that it outperforms both batch learning and existing streaming learning algorithms on the ImageNet ILSVRC-2012 dataset.

{{</citation>}}


### (46/91) Edge Based Oriented Object Detection (Jianghu Shen et al., 2023)

{{<citation>}}

Jianghu Shen, Xiaojun Wu. (2023)  
**Edge Based Oriented Object Detection**  

---
Primary Category: cs.CV  
Categories: I-2-10; I-4-8, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.08265v1)  

---


**ABSTRACT**  
In the field of remote sensing, we often utilize oriented bounding boxes (OBB) to bound the objects. This approach significantly reduces the overlap among dense detection boxes and minimizes the inclusion of background content within the bounding boxes. To enhance the detection accuracy of oriented objects, we propose a unique loss function based on edge gradients, inspired by the similarity measurement function used in template matching task. During this process, we address the issues of non-differentiability of the function and the semantic alignment between gradient vectors in ground truth (GT) boxes and predicted boxes (PB). Experimental results show that our proposed loss function achieves $0.6\%$ mAP improvement compared to the commonly used Smooth L1 loss in the baseline algorithm. Additionally, we design an edge-based self-attention module to encourage the detection network to focus more on the object edges. Leveraging these two innovations, we achieve a mAP increase of 1.3% on the DOTA dataset.

{{</citation>}}


### (47/91) Leveraging the Power of Data Augmentation for Transformer-based Tracking (Jie Zhao et al., 2023)

{{<citation>}}

Jie Zhao, Johan Edstedt, Michael Felsberg, Dong Wang, Huchuan Lu. (2023)  
**Leveraging the Power of Data Augmentation for Transformer-based Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2309.08264v1)  

---


**ABSTRACT**  
Due to long-distance correlation and powerful pretrained models, transformer-based methods have initiated a breakthrough in visual object tracking performance. Previous works focus on designing effective architectures suited for tracking, but ignore that data augmentation is equally crucial for training a well-performing model. In this paper, we first explore the impact of general data augmentations on transformer-based trackers via systematic experiments, and reveal the limited effectiveness of these common strategies. Motivated by experimental observations, we then propose two data augmentation methods customized for tracking. First, we optimize existing random cropping via a dynamic search radius mechanism and simulation for boundary samples. Second, we propose a token-level feature mixing augmentation strategy, which enables the model against challenges like background interference. Extensive experiments on two transformer-based trackers and six benchmarks demonstrate the effectiveness and data efficiency of our methods, especially under challenging settings, like one-shot tracking and small image resolutions.

{{</citation>}}


### (48/91) Cartoondiff: Training-free Cartoon Image Generation with Diffusion Transformer Models (Feihong He et al., 2023)

{{<citation>}}

Feihong He, Gang Li, Lingyu Si, Leilei Yan, Shimeng Hou, Hongwei Dong, Fanzhang Li. (2023)  
**Cartoondiff: Training-free Cartoon Image Generation with Diffusion Transformer Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.08251v1)  

---


**ABSTRACT**  
Image cartoonization has attracted significant interest in the field of image generation. However, most of the existing image cartoonization techniques require re-training models using images of cartoon style. In this paper, we present CartoonDiff, a novel training-free sampling approach which generates image cartoonization using diffusion transformer models. Specifically, we decompose the reverse process of diffusion models into the semantic generation phase and the detail generation phase. Furthermore, we implement the image cartoonization process by normalizing high-frequency signal of the noisy image in specific denoising steps. CartoonDiff doesn't require any additional reference images, complex model designs, or the tedious adjustment of multiple parameters. Extensive experimental results show the powerful ability of our CartoonDiff. The project page is available at: https://cartoondiff.github.io/

{{</citation>}}


### (49/91) Optimization of Rank Losses for Image Retrieval (Elias Ramzi et al., 2023)

{{<citation>}}

Elias Ramzi, Nicolas Audebert, Clément Rambour, André Araujo, Xavier Bitot, Nicolas Thome. (2023)  
**Optimization of Rank Losses for Image Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2309.08250v1)  

---


**ABSTRACT**  
In image retrieval, standard evaluation metrics rely on score ranking, \eg average precision (AP), recall at k (R@k), normalized discounted cumulative gain (NDCG). In this work we introduce a general framework for robust and decomposable rank losses optimization. It addresses two major challenges for end-to-end training of deep neural networks with rank losses: non-differentiability and non-decomposability. Firstly we propose a general surrogate for ranking operator, SupRank, that is amenable to stochastic gradient descent. It provides an upperbound for rank losses and ensures robust training. Secondly, we use a simple yet effective loss function to reduce the decomposability gap between the averaged batch approximation of ranking losses and their values on the whole training set. We apply our framework to two standard metrics for image retrieval: AP and R@k. Additionally we apply our framework to hierarchical image retrieval. We introduce an extension of AP, the hierarchical average precision $\mathcal{H}$-AP, and optimize it as well as the NDCG. Finally we create the first hierarchical landmarks retrieval dataset. We use a semi-automatic pipeline to create hierarchical labels, extending the large scale Google Landmarks v2 dataset. The hierarchical dataset is publicly available at https://github.com/cvdfoundation/google-landmark. Code will be released at https://github.com/elias-ramzi/SupRank.

{{</citation>}}


### (50/91) UniST: Towards Unifying Saliency Transformer for Video Saliency Prediction and Detection (Junwen Xiong et al., 2023)

{{<citation>}}

Junwen Xiong, Peng Zhang, Chuanyue Li, Wei Huang, Yufei Zha, Tao You. (2023)  
**UniST: Towards Unifying Saliency Transformer for Video Saliency Prediction and Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.08220v1)  

---


**ABSTRACT**  
Video saliency prediction and detection are thriving research domains that enable computers to simulate the distribution of visual attention akin to how humans perceiving dynamic scenes. While many approaches have crafted task-specific training paradigms for either video saliency prediction or video salient object detection tasks, few attention has been devoted to devising a generalized saliency modeling framework that seamlessly bridges both these distinct tasks. In this study, we introduce the Unified Saliency Transformer (UniST) framework, which comprehensively utilizes the essential attributes of video saliency prediction and video salient object detection. In addition to extracting representations of frame sequences, a saliency-aware transformer is designed to learn the spatio-temporal representations at progressively increased resolutions, while incorporating effective cross-scale saliency information to produce a robust representation. Furthermore, a task-specific decoder is proposed to perform the final prediction for each task. To the best of our knowledge, this is the first work that explores designing a transformer structure for both saliency modeling tasks. Convincible experiments demonstrate that the proposed UniST achieves superior performance across seven challenging benchmarks for two tasks, and significantly outperforms the other state-of-the-art methods.

{{</citation>}}


### (51/91) Salient Object Detection in Optical Remote Sensing Images Driven by Transformer (Gongyang Li et al., 2023)

{{<citation>}}

Gongyang Li, Zhen Bai, Zhi Liu, Xinpeng Zhang, Haibin Ling. (2023)  
**Salient Object Detection in Optical Remote Sensing Images Driven by Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2309.08206v1)  

---


**ABSTRACT**  
Existing methods for Salient Object Detection in Optical Remote Sensing Images (ORSI-SOD) mainly adopt Convolutional Neural Networks (CNNs) as the backbone, such as VGG and ResNet. Since CNNs can only extract features within certain receptive fields, most ORSI-SOD methods generally follow the local-to-contextual paradigm. In this paper, we propose a novel Global Extraction Local Exploration Network (GeleNet) for ORSI-SOD following the global-to-local paradigm. Specifically, GeleNet first adopts a transformer backbone to generate four-level feature embeddings with global long-range dependencies. Then, GeleNet employs a Direction-aware Shuffle Weighted Spatial Attention Module (D-SWSAM) and its simplified version (SWSAM) to enhance local interactions, and a Knowledge Transfer Module (KTM) to further enhance cross-level contextual interactions. D-SWSAM comprehensively perceives the orientation information in the lowest-level features through directional convolutions to adapt to various orientations of salient objects in ORSIs, and effectively enhances the details of salient objects with an improved attention mechanism. SWSAM discards the direction-aware part of D-SWSAM to focus on localizing salient objects in the highest-level features. KTM models the contextual correlation knowledge of two middle-level features of different scales based on the self-attention mechanism, and transfers the knowledge to the raw features to generate more discriminative features. Finally, a saliency predictor is used to generate the saliency map based on the outputs of the above three modules. Extensive experiments on three public datasets demonstrate that the proposed GeleNet outperforms relevant state-of-the-art methods. The code and results of our method are available at https://github.com/MathLee/GeleNet.

{{</citation>}}


### (52/91) ECEA: Extensible Co-Existing Attention for Few-Shot Object Detection (Zhimeng Xin et al., 2023)

{{<citation>}}

Zhimeng Xin, Tianxu Wu, Shiming Chen, Yixiong Zou, Ling Shao, Xinge You. (2023)  
**ECEA: Extensible Co-Existing Attention for Few-Shot Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Few-Shot, Object Detection  
[Paper Link](http://arxiv.org/abs/2309.08196v1)  

---


**ABSTRACT**  
Few-shot object detection (FSOD) identifies objects from extremely few annotated samples. Most existing FSOD methods, recently, apply the two-stage learning paradigm, which transfers the knowledge learned from abundant base classes to assist the few-shot detectors by learning the global features. However, such existing FSOD approaches seldom consider the localization of objects from local to global. Limited by the scarce training data in FSOD, the training samples of novel classes typically capture part of objects, resulting in such FSOD methods cannot detect the completely unseen object during testing. To tackle this problem, we propose an Extensible Co-Existing Attention (ECEA) module to enable the model to infer the global object according to the local parts. Essentially, the proposed module continuously learns the extensible ability on the base stage with abundant samples and transfers it to the novel stage, which can assist the few-shot model to quickly adapt in extending local regions to co-existing regions. Specifically, we first devise an extensible attention mechanism that starts with a local region and extends attention to co-existing regions that are similar and adjacent to the given local region. We then implement the extensible attention mechanism in different feature scales to progressively discover the full object in various receptive fields. Extensive experiments on the PASCAL VOC and COCO datasets show that our ECEA module can assist the few-shot detector to completely predict the object despite some regions failing to appear in the training samples and achieve the new state of the art compared with existing FSOD methods.

{{</citation>}}


### (53/91) Differentiable Resolution Compression and Alignment for Efficient Video Classification and Retrieval (Rui Deng et al., 2023)

{{<citation>}}

Rui Deng, Qian Wu, Yuke Li, Haoran Fu. (2023)  
**Differentiable Resolution Compression and Alignment for Efficient Video Classification and Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.08167v1)  

---


**ABSTRACT**  
Optimizing video inference efficiency has become increasingly important with the growing demand for video analysis in various fields. Some existing methods achieve high efficiency by explicit discard of spatial or temporal information, which poses challenges in fast-changing and fine-grained scenarios. To address these issues, we propose an efficient video representation network with Differentiable Resolution Compression and Alignment mechanism, which compresses non-essential information in the early stage of the network to reduce computational costs while maintaining consistent temporal correlations. Specifically, we leverage a Differentiable Context-aware Compression Module to encode the saliency and non-saliency frame features, refining and updating the features into a high-low resolution video sequence. To process the new sequence, we introduce a new Resolution-Align Transformer Layer to capture global temporal correlations among frame features with different resolutions, while reducing spatial computation costs quadratically by utilizing fewer spatial tokens in low-resolution non-saliency frames. The entire network can be end-to-end optimized via the integration of the differentiable compression module. Experimental results show that our method achieves the best trade-off between efficiency and performance on near-duplicate video retrieval and competitive results on dynamic video classification compared to state-of-the-art methods. Code:https://github.com/dun-research/DRCA

{{</citation>}}


### (54/91) Uncertainty-Aware Multi-View Visual Semantic Embedding (Wenzhang Wei et al., 2023)

{{<citation>}}

Wenzhang Wei, Zhipeng Gui, Changguang Wu, Anqi Zhao, Xingguang Wang, Huayi Wu. (2023)  
**Uncertainty-Aware Multi-View Visual Semantic Embedding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-IR, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.08154v1)  

---


**ABSTRACT**  
The key challenge in image-text retrieval is effectively leveraging semantic information to measure the similarity between vision and language data. However, using instance-level binary labels, where each image is paired with a single text, fails to capture multiple correspondences between different semantic units, leading to uncertainty in multi-modal semantic understanding. Although recent research has captured fine-grained information through more complex model structures or pre-training techniques, few studies have directly modeled uncertainty of correspondence to fully exploit binary labels. To address this issue, we propose an Uncertainty-Aware Multi-View Visual Semantic Embedding (UAMVSE)} framework that decomposes the overall image-text matching into multiple view-text matchings. Our framework introduce an uncertainty-aware loss function (UALoss) to compute the weighting of each view-text loss by adaptively modeling the uncertainty in each view-text correspondence. Different weightings guide the model to focus on different semantic information, enhancing the model's ability to comprehend the correspondence of images and texts. We also design an optimized image-text matching strategy by normalizing the similarity matrix to improve model performance. Experimental results on the Flicker30k and MS-COCO datasets demonstrate that UAMVSE outperforms state-of-the-art models.

{{</citation>}}


### (55/91) DA-RAW: Domain Adaptive Object Detection for Real-World Adverse Weather Conditions (Minsik Jeon et al., 2023)

{{<citation>}}

Minsik Jeon, Junwon Seo, Jihong Min. (2023)  
**DA-RAW: Domain Adaptive Object Detection for Real-World Adverse Weather Conditions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.08152v1)  

---


**ABSTRACT**  
Despite the success of deep learning-based object detection methods in recent years, it is still challenging to make the object detector reliable in adverse weather conditions such as rain and snow. For the robust performance of object detectors, unsupervised domain adaptation has been utilized to adapt the detection network trained on clear weather images to adverse weather images. While previous methods do not explicitly address weather corruption during adaptation, the domain gap between clear and adverse weather can be decomposed into two factors with distinct characteristics: a style gap and a weather gap. In this paper, we present an unsupervised domain adaptation framework for object detection that can more effectively adapt to real-world environments with adverse weather conditions by addressing these two gaps separately. Our method resolves the style gap by concentrating on style-related information of high-level features using an attention module. Using self-supervised contrastive learning, our framework then reduces the weather gap and acquires instance features that are robust to weather corruption. Extensive experiments demonstrate that our method outperforms other methods for object detection in adverse weather conditions.

{{</citation>}}


### (56/91) Multi-Scale Estimation for Omni-Directional Saliency Maps Using Learnable Equator Bias (Takao Yamanaka et al., 2023)

{{<citation>}}

Takao Yamanaka, Tatsuya Suzuki, Taiki Nobutsune, Chenjunlin Wu. (2023)  
**Multi-Scale Estimation for Omni-Directional Saliency Maps Using Learnable Equator Bias**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.08139v1)  

---


**ABSTRACT**  
Omni-directional images have been used in wide range of applications. For the applications, it would be useful to estimate saliency maps representing probability distributions of gazing points with a head-mounted display, to detect important regions in the omni-directional images. This paper proposes a novel saliency-map estimation model for the omni-directional images by extracting overlapping 2-dimensional (2D) plane images from omni-directional images at various directions and angles of view. While 2D saliency maps tend to have high probability at the center of images (center bias), the high-probability region appears at horizontal directions in omni-directional saliency maps when a head-mounted display is used (equator bias). Therefore, the 2D saliency model with a center-bias layer was fine-tuned with an omni-directional dataset by replacing the center-bias layer to an equator-bias layer conditioned on the elevation angle for the extraction of the 2D plane image. The limited availability of omni-directional images in saliency datasets can be compensated by using the well-established 2D saliency model pretrained by a large number of training images with the ground truth of 2D saliency maps. In addition, this paper proposes a multi-scale estimation method by extracting 2D images in multiple angles of view to detect objects of various sizes with variable receptive fields. The saliency maps estimated from the multiple angles of view were integrated by using pixel-wise attention weights calculated in an integration layer for weighting the optimal scale to each object. The proposed method was evaluated using a publicly available dataset with evaluation metrics for omni-directional saliency maps. It was confirmed that the accuracy of the saliency maps was improved by the proposed method.

{{</citation>}}


### (57/91) Detail Reinforcement Diffusion Model: Augmentation Fine-Grained Visual Categorization in Few-Shot Conditions (Tianxu Wu et al., 2023)

{{<citation>}}

Tianxu Wu, Shuo Ye, Shuhuang Chen, Qinmu Peng, Xinge You. (2023)  
**Detail Reinforcement Diffusion Model: Augmentation Fine-Grained Visual Categorization in Few-Shot Conditions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.08097v1)  

---


**ABSTRACT**  
The challenge in fine-grained visual categorization lies in how to explore the subtle differences between different subclasses and achieve accurate discrimination. Previous research has relied on large-scale annotated data and pre-trained deep models to achieve the objective. However, when only a limited amount of samples is available, similar methods may become less effective. Diffusion models have been widely adopted in data augmentation due to their outstanding diversity in data generation. However, the high level of detail required for fine-grained images makes it challenging for existing methods to be directly employed. To address this issue, we propose a novel approach termed the detail reinforcement diffusion model~(DRDM), which leverages the rich knowledge of large models for fine-grained data augmentation and comprises two key components including discriminative semantic recombination (DSR) and spatial knowledge reference~(SKR). Specifically, DSR is designed to extract implicit similarity relationships from the labels and reconstruct the semantic mapping between labels and instances, which enables better discrimination of subtle differences between different subclasses. Furthermore, we introduce the SKR module, which incorporates the distributions of different datasets as references in the feature space. This allows the SKR to aggregate the high-dimensional distribution of subclass features in few-shot FGVC tasks, thus expanding the decision boundary. Through these two critical components, we effectively utilize the knowledge from large models to address the issue of data scarcity, resulting in improved performance for fine-grained visual recognition tasks. Extensive experiments demonstrate the consistent performance gain offered by our DRDM.

{{</citation>}}


## cs.IT (2)



### (58/91) Denoising Diffusion Probabilistic Models for Hardware-Impaired Communications (Mehdi Letafati et al., 2023)

{{<citation>}}

Mehdi Letafati, Samad Ali, Matti Latva-aho. (2023)  
**Denoising Diffusion Probabilistic Models for Hardware-Impaired Communications**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: AI, GPT, Generative AI, Google  
[Paper Link](http://arxiv.org/abs/2309.08568v1)  

---


**ABSTRACT**  
Generative AI has received significant attention among a spectrum of diverse industrial and academic domains, thanks to the magnificent results achieved from deep generative models such as generative pre-trained transformers (GPT) and diffusion models. In this paper, we explore the applications of denoising diffusion probabilistic models (DDPMs) in wireless communication systems under practical assumptions such as hardware impairments (HWI), low-SNR regime, and quantization error. Diffusion models are a new class of state-of-the-art generative models that have already showcased notable success with some of the popular examples by OpenAI and Google Brain. The intuition behind DDPM is to decompose the data generation process over small "denoising" steps. Inspired by this, we propose using denoising diffusion model-based receiver for a practical wireless communication scheme, while providing network resilience in low-SNR regimes, non-Gaussian noise, different HWI levels, and quantization error. We evaluate the reconstruction performance of our scheme in terms of bit error rate (BER) and mean-squared error (MSE). Our results show that 30% and 20% improvement in BER could be achieved compared to deep neural network (DNN)-based receivers in AWGN and non-Gaussian scenarios, respectively.

{{</citation>}}


### (59/91) Bayes-Optimal Estimation in Generalized Linear Models via Spatial Coupling (Pablo Pascual Cobo et al., 2023)

{{<citation>}}

Pablo Pascual Cobo, Kuan Hsieh, Ramji Venkataramanan. (2023)  
**Bayes-Optimal Estimation in Generalized Linear Models via Spatial Coupling**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2309.08404v1)  

---


**ABSTRACT**  
We consider the problem of signal estimation in a generalized linear model (GLM). GLMs include many canonical problems in statistical estimation, such as linear regression, phase retrieval, and 1-bit compressed sensing. Recent work has precisely characterized the asymptotic minimum mean-squared error (MMSE) for GLMs with i.i.d. Gaussian sensing matrices. However, in many models there is a significant gap between the MMSE and the performance of the best known feasible estimators. In this work, we address this issue by considering GLMs defined via spatially coupled sensing matrices. We propose an efficient approximate message passing (AMP) algorithm for estimation and prove that with a simple choice of spatially coupled design, the MSE of a carefully tuned AMP estimator approaches the asymptotic MMSE in the high-dimensional limit. To prove the result, we first rigorously characterize the asymptotic performance of AMP for a GLM with a generic spatially coupled design. This characterization is in terms of a deterministic recursion (`state evolution') that depends on the parameters defining the spatial coupling. Then, using a simple spatially coupled design and judicious choice of functions defining the AMP, we analyze the fixed points of the resulting state evolution and show that it achieves the asymptotic MMSE. Numerical results for phase retrieval and rectified linear regression show that spatially coupled designs can yield substantially lower MSE than i.i.d. Gaussian designs at finite dimensions when used with AMP algorithms.

{{</citation>}}


## cs.RO (5)



### (60/91) MOSAIC: Learning Unified Multi-Sensory Object Property Representations for Robot Perception (Gyan Tatiya et al., 2023)

{{<citation>}}

Gyan Tatiya, Jonathan Francis, Ho-Hsiang Wu, Yonatan Bisk, Jivko Sinapov. (2023)  
**MOSAIC: Learning Unified Multi-Sensory Object Property Representations for Robot Perception**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI, Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2309.08508v1)  

---


**ABSTRACT**  
A holistic understanding of object properties across diverse sensory modalities (e.g., visual, audio, and haptic) is essential for tasks ranging from object categorization to complex manipulation. Drawing inspiration from cognitive science studies that emphasize the significance of multi-sensory integration in human perception, we introduce MOSAIC (Multi-modal Object property learning with Self-Attention and Integrated Comprehension), a novel framework designed to facilitate the learning of unified multi-sensory object property representations. While it is undeniable that visual information plays a prominent role, we acknowledge that many fundamental object properties extend beyond the visual domain to encompass attributes like texture, mass distribution, or sounds, which significantly influence how we interact with objects. In MOSAIC, we leverage this profound insight by distilling knowledge from the extensive pre-trained Contrastive Language-Image Pre-training (CLIP) model, aligning these representations not only across vision but also haptic and auditory sensory modalities. Through extensive experiments on a dataset where a humanoid robot interacts with 100 objects across 10 exploratory behaviors, we demonstrate the versatility of MOSAIC in two task families: object categorization and object-fetching tasks. Our results underscore the efficacy of MOSAIC's unified representations, showing competitive performance in category recognition through a simple linear probe setup and excelling in the fetch object task under zero-shot transfer conditions. This work pioneers the application of CLIP-based sensory grounding in robotics, promising a significant leap in multi-sensory perception capabilities for autonomous systems. We have released the code, datasets, and additional results: https://github.com/gtatiya/MOSAIC.

{{</citation>}}


### (61/91) OccupancyDETR: Making Semantic Scene Completion as Straightforward as Object Detection (Yupeng Jia et al., 2023)

{{<citation>}}

Yupeng Jia, Jie He, Runze Chen, Fang Zhao, Haiyong Luo. (2023)  
**OccupancyDETR: Making Semantic Scene Completion as Straightforward as Object Detection**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.08504v1)  

---


**ABSTRACT**  
Visual-based 3D semantic occupancy perception (also known as 3D semantic scene completion) is a new perception paradigm for robotic applications like autonomous driving. Compared with Bird's Eye View (BEV) perception, it extends the vertical dimension, significantly enhancing the ability of robots to understand their surroundings. However, due to this very reason, the computational demand for current 3D semantic occupancy perception methods generally surpasses that of BEV perception methods and 2D perception methods. We propose a novel 3D semantic occupancy perception method, OccupancyDETR, which consists of a DETR-like object detection module and a 3D occupancy decoder module. The integration of object detection simplifies our method structurally - instead of predicting the semantics of each voxels, it identifies objects in the scene and their respective 3D occupancy grids. This speeds up our method, reduces required resources, and leverages object detection algorithm, giving our approach notable performance on small objects. We demonstrate the effectiveness of our proposed method on the SemanticKITTI dataset, showcasing an mIoU of 23 and a processing speed of 6 frames per second, thereby presenting a promising solution for real-time 3D semantic scene completion.

{{</citation>}}


### (62/91) Sim-to-Real Brush Manipulation using Behavior Cloning and Reinforcement Learning (Biao Jia et al., 2023)

{{<citation>}}

Biao Jia, Dinesh Manocha. (2023)  
**Sim-to-Real Brush Manipulation using Behavior Cloning and Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.08457v1)  

---


**ABSTRACT**  
Developing proficient brush manipulation capabilities in real-world scenarios is a complex and challenging endeavor, with wide-ranging applications in fields such as art, robotics, and digital design. In this study, we introduce an approach designed to bridge the gap between simulated environments and real-world brush manipulation. Our framework leverages behavior cloning and reinforcement learning to train a painting agent, seamlessly integrating it into both virtual and real-world environments. Additionally, we employ a real painting environment featuring a robotic arm and brush, mirroring the MyPaint virtual environment. Our results underscore the agent's effectiveness in acquiring policies for high-dimensional continuous action spaces, facilitating the smooth transfer of brush manipulation techniques from simulation to practical, real-world applications.

{{</citation>}}


### (63/91) Find What You Want: Learning Demand-conditioned Object Attribute Space for Demand-driven Navigation (Hongcheng Wang et al., 2023)

{{<citation>}}

Hongcheng Wang, Andy Guan Hong Chen, Xiaoqi Li, Mingdong Wu, Hao Dong. (2023)  
**Find What You Want: Learning Demand-conditioned Object Attribute Space for Demand-driven Navigation**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.08138v1)  

---


**ABSTRACT**  
The task of Visual Object Navigation (VON) involves an agent's ability to locate a particular object within a given scene. In order to successfully accomplish the VON task, two essential conditions must be fulfilled:1) the user must know the name of the desired object; and 2) the user-specified object must actually be present within the scene. To meet these conditions, a simulator can incorporate pre-defined object names and positions into the metadata of the scene. However, in real-world scenarios, it is often challenging to ensure that these conditions are always met. Human in an unfamiliar environment may not know which objects are present in the scene, or they may mistakenly specify an object that is not actually present. Nevertheless, despite these challenges, human may still have a demand for an object, which could potentially be fulfilled by other objects present within the scene in an equivalent manner. Hence, we propose Demand-driven Navigation (DDN), which leverages the user's demand as the task instruction and prompts the agent to find the object matches the specified demand. DDN aims to relax the stringent conditions of VON by focusing on fulfilling the user's demand rather than relying solely on predefined object categories or names. We propose a method first acquire textual attribute features of objects by extracting common knowledge from a large language model. These textual attribute features are subsequently aligned with visual attribute features using Contrastive Language-Image Pre-training (CLIP). By incorporating the visual attribute features as prior knowledge, we enhance the navigation process. Experiments on AI2Thor with the ProcThor dataset demonstrate the visual attribute features improve the agent's navigation performance and outperform the baseline methods commonly used in VON.

{{</citation>}}


### (64/91) RELAX: Reinforcement Learning Enabled 2D-LiDAR Autonomous System for Parsimonious UAVs (Guanlin Wu et al., 2023)

{{<citation>}}

Guanlin Wu, Zhuokai Zhao, Yutao He. (2023)  
**RELAX: Reinforcement Learning Enabled 2D-LiDAR Autonomous System for Parsimonious UAVs**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.08095v1)  

---


**ABSTRACT**  
Unmanned Aerial Vehicles (UAVs) have gained significant prominence in recent years for areas including surveillance, search, rescue, and package delivery. One key aspect in UAV operations shared across all these tasks is the autonomous path planning, which enables UAV to navigate through complex, unknown, and dynamic environments while avoiding obstacles without human control. Despite countless efforts having been devoted to this subject, new challenges are constantly arisen due to the persistent trade-off between performance and cost. And new studies are more urgently needed to develop autonomous system for UAVs with parsimonious sensor setup, which is a major need for wider adoptions. To this end, we propose an end-to-end autonomous framework to enable UAVs with only one single 2D-LiDAR sensor to operate in unknown dynamic environments. More specifically, we break our approach into three stages: a pre-processing Map Constructor; an offline Mission Planner; and an online reinforcement learning (RL)-based Dynamic Obstacle Handler. Experiments show that our approach provides robust and reliable dynamic path planning and obstacle avoidance with only 1/10 of the cost in sensor configuration. The code will be made public upon acceptance.

{{</citation>}}


## cs.CR (2)



### (65/91) XFedHunter: An Explainable Federated Learning Framework for Advanced Persistent Threat Detection in SDN (Huynh Thai Thi et al., 2023)

{{<citation>}}

Huynh Thai Thi, Ngo Duc Hoang Son, Phan The Duy, Nghi Hoang Khoa, Khoa Ngo-Khanh, Van-Hau Pham. (2023)  
**XFedHunter: An Explainable Federated Learning Framework for Advanced Persistent Threat Detection in SDN**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2309.08485v1)  

---


**ABSTRACT**  
Advanced Persistent Threat (APT) attacks are highly sophisticated and employ a multitude of advanced methods and techniques to target organizations and steal sensitive and confidential information. APT attacks consist of multiple stages and have a defined strategy, utilizing new and innovative techniques and technologies developed by hackers to evade security software monitoring. To effectively protect against APTs, detecting and predicting APT indicators with an explanation from Machine Learning (ML) prediction is crucial to reveal the characteristics of attackers lurking in the network system. Meanwhile, Federated Learning (FL) has emerged as a promising approach for building intelligent applications without compromising privacy. This is particularly important in cybersecurity, where sensitive data and high-quality labeling play a critical role in constructing effective machine learning models for detecting cyber threats. Therefore, this work proposes XFedHunter, an explainable federated learning framework for APT detection in Software-Defined Networking (SDN) leveraging local cyber threat knowledge from many training collaborators. In XFedHunter, Graph Neural Network (GNN) and Deep Learning model are utilized to reveal the malicious events effectively in the large number of normal ones in the network system. The experimental results on NF-ToN-IoT and DARPA TCE3 datasets indicate that our framework can enhance the trust and accountability of ML-based systems utilized for cybersecurity purposes without privacy leakage.

{{</citation>}}


### (66/91) VulnSense: Efficient Vulnerability Detection in Ethereum Smart Contracts by Multimodal Learning with Graph Neural Network and Language Model (Phan The Duy et al., 2023)

{{<citation>}}

Phan The Duy, Nghi Hoang Khoa, Nguyen Huu Quyen, Le Cong Trinh, Vu Trung Kien, Trinh Minh Hoang, Van-Hau Pham. (2023)  
**VulnSense: Efficient Vulnerability Detection in Ethereum Smart Contracts by Multimodal Learning with Graph Neural Network and Language Model**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: BERT, GNN, Graph Neural Network, LSTM, Language Model, NLP, Transformer, Transformers, Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2309.08474v1)  

---


**ABSTRACT**  
This paper presents VulnSense framework, a comprehensive approach to efficiently detect vulnerabilities in Ethereum smart contracts using a multimodal learning approach on graph-based and natural language processing (NLP) models. Our proposed framework combines three types of features from smart contracts comprising source code, opcode sequences, and control flow graph (CFG) extracted from bytecode. We employ Bidirectional Encoder Representations from Transformers (BERT), Bidirectional Long Short-Term Memory (BiLSTM) and Graph Neural Network (GNN) models to extract and analyze these features. The final layer of our multimodal approach consists of a fully connected layer used to predict vulnerabilities in Ethereum smart contracts. Addressing limitations of existing vulnerability detection methods relying on single-feature or single-model deep learning techniques, our method surpasses accuracy and effectiveness constraints. We assess VulnSense using a collection of 1.769 smart contracts derived from the combination of three datasets: Curated, SolidiFI-Benchmark, and Smartbugs Wild. We then make a comparison with various unimodal and multimodal learning techniques contributed by GNN, BiLSTM and BERT architectures. The experimental outcomes demonstrate the superior performance of our proposed approach, achieving an average accuracy of 77.96\% across all three categories of vulnerable smart contracts.

{{</citation>}}


## eess.AS (5)



### (67/91) Chunked Attention-based Encoder-Decoder Model for Streaming Speech Recognition (Mohammad Zeineldeen et al., 2023)

{{<citation>}}

Mohammad Zeineldeen, Albert Zeyer, Ralf Schlüter, Hermann Ney. (2023)  
**Chunked Attention-based Encoder-Decoder Model for Streaming Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS, stat-ML  
Keywords: Attention, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.08436v1)  

---


**ABSTRACT**  
We study a streamable attention-based encoder-decoder model in which either the decoder, or both the encoder and decoder, operate on pre-defined, fixed-size windows called chunks. A special end-of-chunk (EOC) symbol advances from one chunk to the next chunk, effectively replacing the conventional end-of-sequence symbol. This modification, while minor, situates our model as equivalent to a transducer model that operates on chunks instead of frames, where EOC corresponds to the blank symbol. We further explore the remaining differences between a standard transducer and our model. Additionally, we examine relevant aspects such as long-form speech generalization, beam size, and length normalization. Through experiments on Librispeech and TED-LIUM-v2, and by concatenating consecutive sequences for long-form trials, we find that our streamable model maintains competitive performance compared to the non-streamable variant and generalizes very well to long-form speech.

{{</citation>}}


### (68/91) Semi-supervised Sound Event Detection with Local and Global Consistency Regularization (Yiming Li et al., 2023)

{{<citation>}}

Yiming Li, Xiangdong Wang, Hong Liu, Rui Tao, Long Yan, Kazushige Ouchi. (2023)  
**Semi-supervised Sound Event Detection with Local and Global Consistency Regularization**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Event Detection  
[Paper Link](http://arxiv.org/abs/2309.08355v1)  

---


**ABSTRACT**  
Learning meaningful frame-wise features on a partially labeled dataset is crucial to semi-supervised sound event detection. Prior works either maintain consistency on frame-level predictions or seek feature-level similarity among neighboring frames, which cannot exploit the potential of unlabeled data. In this work, we design a Local and Global Consistency (LGC) regularization scheme to enhance the model on both label- and feature-level. The audio CutMix is introduced to change the contextual information of clips. Then, the local consistency is adopted to encourage the model to leverage local features for frame-level predictions, and the global consistency is applied to force features to align with global prototypes through a specially designed contrastive loss. Experiments on the DESED dataset indicate the superiority of LGC, surpassing its respective competitors largely with the same settings as the baseline system. Besides, combining LGC with existing methods can obtain further improvements. The code will be released soon.

{{</citation>}}


### (69/91) One-Class Knowledge Distillation for Spoofing Speech Detection (Jingze Lu et al., 2023)

{{<citation>}}

Jingze Lu, Yuxiang Zhang, Wenchao Wang, Zengqiang Shang, Pengyuan Zhang. (2023)  
**One-Class Knowledge Distillation for Spoofing Speech Detection**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.08285v1)  

---


**ABSTRACT**  
The detection of spoofing speech generated by unseen algorithms remains an unresolved challenge. One reason for the lack of generalization ability is traditional detecting systems follow the binary classification paradigm, which inherently assumes the possession of prior knowledge of spoofing speech. One-class methods attempt to learn the distribution of bonafide speech and are inherently suited to the task where spoofing speech exhibits significant differences. However, training a one-class system using only bonafide speech is challenging. In this paper, we introduce a teacher-student framework to provide guidance for the training of a one-class model. The proposed one-class knowledge distillation method outperforms other state-of-the-art methods on the ASVspoof 21DF dataset and InTheWild dataset, which demonstrates its superior generalization ability.

{{</citation>}}


### (70/91) Cross-lingual Knowledge Distillation via Flow-based Voice Conversion for Robust Polyglot Text-To-Speech (Dariusz Piotrowski et al., 2023)

{{<citation>}}

Dariusz Piotrowski, Renard Korzeniowski, Alessio Falai, Sebastian Cygert, Kamil Pokora, Georgi Tinchev, Ziyao Zhang, Kayoko Yanagisawa. (2023)  
**Cross-lingual Knowledge Distillation via Flow-based Voice Conversion for Robust Polyglot Text-To-Speech**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.08255v1)  

---


**ABSTRACT**  
In this work, we introduce a framework for cross-lingual speech synthesis, which involves an upstream Voice Conversion (VC) model and a downstream Text-To-Speech (TTS) model. The proposed framework consists of 4 stages. In the first two stages, we use a VC model to convert utterances in the target locale to the voice of the target speaker. In the third stage, the converted data is combined with the linguistic features and durations from recordings in the target language, which are then used to train a single-speaker acoustic model. Finally, the last stage entails the training of a locale-independent vocoder. Our evaluations show that the proposed paradigm outperforms state-of-the-art approaches which are based on training a large multilingual TTS model. In addition, our experiments demonstrate the robustness of our approach with different model architectures, languages, speakers and amounts of data. Moreover, our solution is especially beneficial in low-resource settings.

{{</citation>}}


### (71/91) Libriheavy: a 50,000 hours ASR corpus with punctuation casing and context (Wei Kang et al., 2023)

{{<citation>}}

Wei Kang, Xiaoyu Yang, Zengwei Yao, Fangjun Kuang, Yifan Yang, Liyong Guo, Long Lin, Daniel Povey. (2023)  
**Libriheavy: a 50,000 hours ASR corpus with punctuation casing and context**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.08105v1)  

---


**ABSTRACT**  
In this paper, we introduce Libriheavy, a large-scale ASR corpus consisting of 50,000 hours of read English speech derived from LibriVox. To the best of our knowledge, Libriheavy is the largest freely-available corpus of speech with supervisions. Different from other open-sourced datasets that only provide normalized transcriptions, Libriheavy contains richer information such as punctuation, casing and text context, which brings more flexibility for system building. Specifically, we propose a general and efficient pipeline to locate, align and segment the audios in previously published Librilight to its corresponding texts. The same as Librilight, Libriheavy also has three training subsets small, medium, large of the sizes 500h, 5000h, 50000h respectively. We also extract the dev and test evaluation sets from the aligned audios and guarantee there is no overlapping speakers and books in training sets. Baseline systems are built on the popular CTC-Attention and transducer models. Additionally, we open-source our dataset creatation pipeline which can also be used to other audio alignment tasks.

{{</citation>}}


## eess.IV (4)



### (72/91) Segment Anything Model for Brain Tumor Segmentation (Peng Zhang et al., 2023)

{{<citation>}}

Peng Zhang, Yaping Wang. (2023)  
**Segment Anything Model for Brain Tumor Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.08434v1)  

---


**ABSTRACT**  
Glioma is a prevalent brain tumor that poses a significant health risk to individuals. Accurate segmentation of brain tumor is essential for clinical diagnosis and treatment. The Segment Anything Model(SAM), released by Meta AI, is a fundamental model in image segmentation and has excellent zero-sample generalization capabilities. Thus, it is interesting to apply SAM to the task of brain tumor segmentation. In this study, we evaluated the performance of SAM on brain tumor segmentation and found that without any model fine-tuning, there is still a gap between SAM and the current state-of-the-art(SOTA) model.

{{</citation>}}


### (73/91) 3D SA-UNet: 3D Spatial Attention UNet with 3D ASPP for White Matter Hyperintensities Segmentation (Changlu Guo, 2023)

{{<citation>}}

Changlu Guo. (2023)  
**3D SA-UNet: 3D Spatial Attention UNet with 3D ASPP for White Matter Hyperintensities Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2309.08402v1)  

---


**ABSTRACT**  
White Matter Hyperintensity (WMH) is an imaging feature related to various diseases such as dementia and stroke. Accurately segmenting WMH using computer technology is crucial for early disease diagnosis. However, this task remains challenging due to the small lesions with low contrast and high discontinuity in the images, which contain limited contextual and spatial information. To address this challenge, we propose a deep learning model called 3D Spatial Attention U-Net (3D SA-UNet) for automatic WMH segmentation using only Fluid Attenuation Inversion Recovery (FLAIR) scans. The 3D SA-UNet introduces a 3D Spatial Attention Module that highlights important lesion features, such as WMH, while suppressing unimportant regions. Additionally, to capture features at different scales, we extend the Atrous Spatial Pyramid Pooling (ASPP) module to a 3D version, enhancing the segmentation performance of the network. We evaluate our method on publicly available dataset and demonstrate the effectiveness of 3D spatial attention module and 3D ASPP in WMH segmentation. Through experimental results, it has been demonstrated that our proposed 3D SA-UNet model achieves higher accuracy compared to other state-of-the-art 3D convolutional neural networks.

{{</citation>}}


### (74/91) Reconsidering evaluation practices in modular systems: On the propagation of errors in MRI prostate cancer detection (Erlend Sortland Rolfsnes et al., 2023)

{{<citation>}}

Erlend Sortland Rolfsnes, Philip Thangngat, Trygve Eftestøl, Tobias Nordström, Fredrik Jäderling, Martin Eklund, Alvaro Fernandez-Quilez. (2023)  
**Reconsidering evaluation practices in modular systems: On the propagation of errors in MRI prostate cancer detection**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV, physics-med-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.08381v1)  

---


**ABSTRACT**  
Magnetic resonance imaging has evolved as a key component for prostate cancer (PCa) detection, substantially increasing the radiologist workload. Artificial intelligence (AI) systems can support radiological assessment by segmenting and classifying lesions in clinically significant (csPCa) and non-clinically significant (ncsPCa). Commonly, AI systems for PCa detection involve an automatic prostate segmentation followed by the lesion detection using the extracted prostate. However, evaluation reports are typically presented in terms of detection under the assumption of the availability of a highly accurate segmentation and an idealistic scenario, omitting the propagation of errors between modules. For that purpose, we evaluate the effect of two different segmentation networks (s1 and s2) with heterogeneous performances in the detection stage and compare it with an idealistic setting (s1:89.90+-2.23 vs 88.97+-3.06 ncsPCa, P<.001, 89.30+-4.07 and 88.12+-2.71 csPCa, P<.001). Our results depict the relevance of a holistic evaluation, accounting for all the sub-modules involved in the system.

{{</citation>}}


### (75/91) Cross-Modal Synthesis of Structural MRI and Functional Connectivity Networks via Conditional ViT-GANs (Yuda Bi et al., 2023)

{{<citation>}}

Yuda Bi, Anees Abrol, Jing Sui, Vince Calhoun. (2023)  
**Cross-Modal Synthesis of Structural MRI and Functional Connectivity Networks via Conditional ViT-GANs**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.08160v1)  

---


**ABSTRACT**  
The cross-modal synthesis between structural magnetic resonance imaging (sMRI) and functional network connectivity (FNC) is a relatively unexplored area in medical imaging, especially with respect to schizophrenia. This study employs conditional Vision Transformer Generative Adversarial Networks (cViT-GANs) to generate FNC data based on sMRI inputs. After training on a comprehensive dataset that included both individuals with schizophrenia and healthy control subjects, our cViT-GAN model effectively synthesized the FNC matrix for each subject, and then formed a group difference FNC matrix, obtaining a Pearson correlation of 0.73 with the actual FNC matrix. In addition, our FNC visualization results demonstrate significant correlations in particular subcortical brain regions, highlighting the model's capability of capturing detailed structural-functional associations. This performance distinguishes our model from conditional CNN-based GAN alternatives such as Pix2Pix. Our research is one of the first attempts to link sMRI and FNC synthesis, setting it apart from other cross-modal studies that concentrate on T1- and T2-weighted MR images or the fusion of MRI and CT scans.

{{</citation>}}


## cs.SD (5)



### (76/91) Exploring Meta Information for Audio-based Zero-shot Bird Classification (Alexander Gebhard et al., 2023)

{{<citation>}}

Alexander Gebhard, Andreas Triantafyllopoulos, Teresa Bez, Lukas Christ, Alexander Kathan, Björn W. Schuller. (2023)  
**Exploring Meta Information for Audio-based Zero-shot Bird Classification**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.08398v1)  

---


**ABSTRACT**  
Advances in passive acoustic monitoring and machine learning have led to the procurement of vast datasets for computational bioacoustic research. Nevertheless, data scarcity is still an issue for rare and underrepresented species. This study investigates how meta-information can improve zero-shot audio classification, utilising bird species as an example case study due to the availability of rich and diverse metadata. We investigate three different sources of metadata: textual bird sound descriptions encoded via (S)BERT, functional traits (AVONET), and bird life-history (BLH) characteristics. As audio features, we extract audio spectrogram transformer (AST) embeddings and project them to the dimension of the auxiliary information by adopting a single linear layer. Then, we employ the dot product as compatibility function and a standard zero-shot learning ranking hinge loss to determine the correct class. The best results are achieved by concatenating the AVONET and BLH features attaining a mean F1-score of .233 over five different test sets with 8 to 10 classes.

{{</citation>}}


### (77/91) HM-Conformer: A Conformer-based audio deepfake detection system with hierarchical pooling and multi-level classification token aggregation methods (Hyun-seo Shin et al., 2023)

{{<citation>}}

Hyun-seo Shin, Jungwoo Heo, Ju-ho Kim, Chan-yeong Lim, Wonbin Kim, Ha-Jin Yu. (2023)  
**HM-Conformer: A Conformer-based audio deepfake detection system with hierarchical pooling and multi-level classification token aggregation methods**  

---
Primary Category: cs.SD  
Categories: cs-CR, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.08208v1)  

---


**ABSTRACT**  
Audio deepfake detection (ADD) is the task of detecting spoofing attacks generated by text-to-speech or voice conversion systems. Spoofing evidence, which helps to distinguish between spoofed and bona-fide utterances, might exist either locally or globally in the input features. To capture these, the Conformer, which consists of Transformers and CNN, possesses a suitable structure. However, since the Conformer was designed for sequence-to-sequence tasks, its direct application to ADD tasks may be sub-optimal. To tackle this limitation, we propose HM-Conformer by adopting two components: (1) Hierarchical pooling method progressively reducing the sequence length to eliminate duplicated information (2) Multi-level classification token aggregation method utilizing classification tokens to gather information from different blocks. Owing to these components, HM-Conformer can efficiently detect spoofing evidence by processing various sequence lengths and aggregating them. In experimental results on the ASVspoof 2021 Deepfake dataset, HM-Conformer achieved a 15.71% EER, showing competitive performance compared to recent systems.

{{</citation>}}


### (78/91) Syn-Att: Synthetic Speech Attribution via Semi-Supervised Unknown Multi-Class Ensemble of CNNs (Md Awsafur Rahman et al., 2023)

{{<citation>}}

Md Awsafur Rahman, Bishmoy Paul, Najibul Haque Sarker, Zaber Ibn Abdul Hakim, Shaikh Anowarul Fattah, Mohammad Saquib. (2023)  
**Syn-Att: Synthetic Speech Attribution via Semi-Supervised Unknown Multi-Class Ensemble of CNNs**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-SD, cs.SD, eess-AS  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2309.08146v1)  

---


**ABSTRACT**  
With the huge technological advances introduced by deep learning in audio & speech processing, many novel synthetic speech techniques achieved incredible realistic results. As these methods generate realistic fake human voices, they can be used in malicious acts such as people imitation, fake news, spreading, spoofing, media manipulations, etc. Hence, the ability to detect synthetic or natural speech has become an urgent necessity. Moreover, being able to tell which algorithm has been used to generate a synthetic speech track can be of preeminent importance to track down the culprit. In this paper, a novel strategy is proposed to attribute a synthetic speech track to the generator that is used to synthesize it. The proposed detector transforms the audio into log-mel spectrogram, extracts features using CNN, and classifies it between five known and unknown algorithms, utilizing semi-supervision and ensemble to improve its robustness and generalizability significantly. The proposed detector is validated on two evaluation datasets consisting of a total of 18,000 weakly perturbed (Eval 1) & 10,000 strongly perturbed (Eval 2) synthetic speeches. The proposed method outperforms other top teams in accuracy by 12-13% on Eval 2 and 1-2% on Eval 1, in the IEEE SP Cup challenge at ICASSP 2022.

{{</citation>}}


### (79/91) Two-Step Knowledge Distillation for Tiny Speech Enhancement (Rayan Daod Nathoo et al., 2023)

{{<citation>}}

Rayan Daod Nathoo, Mikolaj Kegler, Marko Stamenovic. (2023)  
**Two-Step Knowledge Distillation for Tiny Speech Enhancement**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.08144v1)  

---


**ABSTRACT**  
Tiny, causal models are crucial for embedded audio machine learning applications. Model compression can be achieved via distilling knowledge from a large teacher into a smaller student model. In this work, we propose a novel two-step approach for tiny speech enhancement model distillation. In contrast to the standard approach of a weighted mixture of distillation and supervised losses, we firstly pre-train the student using only the knowledge distillation (KD) objective, after which we switch to a fully supervised training regime. We also propose a novel fine-grained similarity-preserving KD loss, which aims to match the student's intra-activation Gram matrices to that of the teacher. Our method demonstrates broad improvements, but particularly shines in adverse conditions including high compression and low signal to noise ratios (SNR), yielding signal to distortion ratio gains of 0.9 dB and 1.1 dB, respectively, at -5 dB input SNR and 63x compression compared to baseline.

{{</citation>}}


### (80/91) Foundation Model Assisted Automatic Speech Emotion Recognition: Transcribing, Annotating, and Augmenting (Tiantian Feng et al., 2023)

{{<citation>}}

Tiantian Feng, Shrikanth Narayanan. (2023)  
**Foundation Model Assisted Automatic Speech Emotion Recognition: Transcribing, Annotating, and Augmenting**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: ChatGPT, Emotion Recognition, GPT  
[Paper Link](http://arxiv.org/abs/2309.08108v1)  

---


**ABSTRACT**  
Significant advances are being made in speech emotion recognition (SER) using deep learning models. Nonetheless, training SER systems remains challenging, requiring both time and costly resources. Like many other machine learning tasks, acquiring datasets for SER requires substantial data annotation efforts, including transcription and labeling. These annotation processes present challenges when attempting to scale up conventional SER systems. Recent developments in foundational models have had a tremendous impact, giving rise to applications such as ChatGPT. These models have enhanced human-computer interactions including bringing unique possibilities for streamlining data collection in fields like SER. In this research, we explore the use of foundational models to assist in automating SER from transcription and annotation to augmentation. Our study demonstrates that these models can generate transcriptions to enhance the performance of SER systems that rely solely on speech data. Furthermore, we note that annotating emotions from transcribed speech remains a challenging task. However, combining outputs from multiple LLMs enhances the quality of annotations. Lastly, our findings suggest the feasibility of augmenting existing speech emotion datasets by annotating unlabeled speech samples.

{{</citation>}}


## cs.AI (2)



### (81/91) Learning by Self-Explaining (Wolfgang Stammer et al., 2023)

{{<citation>}}

Wolfgang Stammer, Felix Friedrich, David Steinmann, Hikaru Shindo, Kristian Kersting. (2023)  
**Learning by Self-Explaining**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.08395v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) research has a long track record of drawing inspirations from findings from biology, in particular human intelligence. In contrast to current AI research that mainly treats explanations as a means for model inspection, a somewhat neglected finding from human psychology is the benefit of self-explaining in an agents' learning process. Motivated by this, we introduce a novel learning paradigm, termed Learning by Self-Explaining (LSX). The underlying idea is that a learning module (learner) performs a base task, e.g. image classification, and provides explanations to its decisions. An internal critic module next evaluates the quality of these explanations given the original task. Finally, the learner is refined with the critic's feedback and the loop is repeated as required. The intuition behind this is that an explanation is considered "good" if the critic can perform the same task given the respective explanation. Despite many implementation possibilities the structure of any LSX instantiation can be taxonomized based on four learning modules which we identify as: Fit, Explain, Reflect and Revise. In our work, we provide distinct instantiations of LSX for two different learner models, each illustrating different choices for the various LSX components. We broadly evaluate these on several datasets and show that Learning by Self-Explaining not only boosts the generalization abilities of AI models, particularly in small-data regimes, but also aids in mitigating the influence of confounding factors, as well as leading to more task specific and faithful model explanations. Overall, our results provide experimental evidence of the potential of self-explaining within the learning phase of an AI model.

{{</citation>}}


### (82/91) Quantitative and Qualitative Evaluation of Reinforcement Learning Policies for Autonomous Vehicles (Laura Ferrarotti et al., 2023)

{{<citation>}}

Laura Ferrarotti, Massimiliano Luca, Gabriele Santin, Giorgio Previati, Gianpiero Mastinu, Elena Campi, Lorenzo Uccello, Antonino Albanese, Praveen Zalaya, Alessandro Roccasalva, Bruno Lepri. (2023)  
**Quantitative and Qualitative Evaluation of Reinforcement Learning Policies for Autonomous Vehicles**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-RO, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.08254v1)  

---


**ABSTRACT**  
Optimizing traffic dynamics in an evolving transportation landscape is crucial, particularly in scenarios where autonomous vehicles (AVs) with varying levels of autonomy coexist with human-driven cars. This paper presents a novel approach to optimizing choices of AVs using Proximal Policy Optimization (PPO), a reinforcement learning algorithm. We learned a policy to minimize traffic jams (i.e., minimize the time to cross the scenario) and to minimize pollution in a roundabout in Milan, Italy. Through empirical analysis, we demonstrate that our approach can reduce time and pollution levels. Furthermore, we qualitatively evaluate the learned policy using a cutting-edge cockpit to assess its performance in near-real-world conditions. To gauge the practicality and acceptability of the policy, we conducted evaluations with human participants using the simulator, focusing on a range of metrics like traffic smoothness and safety perception. In general, our findings show that human-driven vehicles benefit from optimizing AVs dynamics. Also, participants in the study highlighted that the scenario with 80\% AVs is perceived as safer than the scenario with 20\%. The same result is obtained for traffic smoothness perception.

{{</citation>}}


## cs.CY (2)



### (83/91) Narratives of War: Ukrainian Memetic Warfare on Twitter (Yelena Mejova et al., 2023)

{{<citation>}}

Yelena Mejova, Arthur Capozzi, Corrado Monti, Gianmarco De Francisci Morales. (2023)  
**Narratives of War: Ukrainian Memetic Warfare on Twitter**  

---
Primary Category: cs.CY  
Categories: J-4; K-4, cs-CY, cs-HC, cs-SI, cs.CY  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2309.08363v1)  

---


**ABSTRACT**  
The 2022 Russian invasion of Ukraine has seen an intensification in the use of social media by governmental actors in cyber warfare. Wartime communication via memes has been a successful strategy used not only by independent accounts such as @uamemesforces, but also-for the first time in a full-scale interstate war-by official Ukrainian government accounts such as @Ukraine and @DefenceU. We study this prominent example of memetic warfare through the lens of its narratives, and find them to be a key component of success: tweets with a 'victim' narrative garner twice as many retweets. However, malevolent narratives focusing on the enemy resonate more than those about heroism or victims with countries providing more assistance to Ukraine. Our findings present a nuanced examination of Ukraine's influence operations and of the worldwide response to it, thus contributing new insights into the evolution of socio-technical systems in times of war.

{{</citation>}}


### (84/91) Talkin' 'Bout AI Generation: Copyright and the Generative-AI Supply Chain (Katherine Lee et al., 2023)

{{<citation>}}

Katherine Lee, A. Feder Cooper, James Grimmelmann. (2023)  
**Talkin' 'Bout AI Generation: Copyright and the Generative-AI Supply Chain**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.08133v1)  

---


**ABSTRACT**  
"Does generative AI infringe copyright?" is an urgent question. It is also a difficult question, for two reasons. First, "generative AI" is not just one product from one company. It is a catch-all name for a massive ecosystem of loosely related technologies, including conversational text chatbots like ChatGPT, image generators like Midjourney and DALL-E, coding assistants like GitHub Copilot, and systems that compose music and create videos. These systems behave differently and raise different legal issues. The second problem is that copyright law is notoriously complicated, and generative-AI systems manage to touch on a great many corners of it: authorship, similarity, direct and indirect liability, fair use, and licensing, among much else. These issues cannot be analyzed in isolation, because there are connections everywhere.   In this Article, we aim to bring order to the chaos. To do so, we introduce the generative-AI supply chain: an interconnected set of stages that transform training data (millions of pictures of cats) into generations (a new, potentially never-seen-before picture of a cat that has never existed). Breaking down generative AI into these constituent stages reveals all of the places at which companies and users make choices that have copyright consequences. It enables us to trace the effects of upstream technical designs on downstream uses, and to assess who in these complicated sociotechnical systems bears responsibility for infringement when it happens. Because we engage so closely with the technology of generative AI, we are able to shed more light on the copyright questions. We do not give definitive answers as to who should and should not be held liable. Instead, we identify the key decisions that courts will need to make as they grapple with these issues, and point out the consequences that would likely flow from different liability regimes.

{{</citation>}}


## stat.ML (1)



### (85/91) Topological Node2vec: Enhanced Graph Embedding via Persistent Homology (Yasuaki Hiraoka et al., 2023)

{{<citation>}}

Yasuaki Hiraoka, Yusuke Imoto, Killian Meehan, Théo Lacombe, Toshiaki Yachimura. (2023)  
**Topological Node2vec: Enhanced Graph Embedding via Persistent Homology**  

---
Primary Category: stat.ML  
Categories: cs-LG, math-AT, math-OC, stat-ML, stat.ML  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.08241v1)  

---


**ABSTRACT**  
Node2vec is a graph embedding method that learns a vector representation for each node of a weighted graph while seeking to preserve relative proximity and global structure. Numerical experiments suggest Node2vec struggles to recreate the topology of the input graph. To resolve this we introduce a topological loss term to be added to the training loss of Node2vec which tries to align the persistence diagram (PD) of the resulting embedding as closely as possible to that of the input graph. Following results in computational optimal transport, we carefully adapt entropic regularization to PD metrics, allowing us to measure the discrepancy between PDs in a differentiable way. Our modified loss function can then be minimized through gradient descent to reconstruct both the geometry and the topology of the input graph. We showcase the benefits of this approach using demonstrative synthetic examples.

{{</citation>}}


## cs.SE (2)



### (86/91) Silent Vulnerability-fixing Commit Identification Based on Graph Neural Networks (Hieu Dinh Vo et al., 2023)

{{<citation>}}

Hieu Dinh Vo, Thanh Trong Vu, Son Nguyen. (2023)  
**Silent Vulnerability-fixing Commit Identification Based on Graph Neural Networks**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.08225v1)  

---


**ABSTRACT**  
The growing dependence of software projects on external libraries has generated apprehensions regarding the security of these libraries because of concealed vulnerabilities. Handling these vulnerabilities presents difficulties due to the temporal delay between remediation and public exposure. Furthermore, a substantial fraction of open-source projects covertly address vulnerabilities without any formal notification, influencing vulnerability management. Established solutions like OWASP predominantly hinge on public announcements, limiting their efficacy in uncovering undisclosed vulnerabilities. To address this challenge, the automated identification of vulnerability-fixing commits has come to the forefront. In this paper, we present VFFINDER, a novel graph-based approach for automated silent vulnerability fix identification. VFFINDER captures structural changes using Abstract Syntax Trees (ASTs) and represents them in annotated ASTs. To precisely capture the meaning of code changes, the changed code is represented in connection with the related unchanged code. In VFFINDER, the structure of the changed code and related unchanged code are captured and the structural changes are represented in annotated Abstract Syntax Trees (aAST). VFFINDER distinguishes vulnerability-fixing commits from non-fixing ones using attention-based graph neural network models to extract structural features expressed in aASTs. We conducted experiments to evaluate VFFINDER on a dataset of 11K+ vulnerability fixing commits in 507 real-world C/C++ projects. Our results show that VFFINDER significantly improves the state-of-the-art methods by 272-420% in Precision, 22-70% in Recall, and 3.2X-8.2X in F1. Especially, VFFINDER speeds up the silent fix identification process by up to 121% with the same effort reviewing 50K LOC compared to the existing approaches.

{{</citation>}}


### (87/91) Exploring the Potential of ChatGPT in Automated Code Refinement: An Empirical Study (Qi Guo et al., 2023)

{{<citation>}}

Qi Guo, Junming Cao, Xiaofei Xie, Shangqing Liu, Xiaohong Li, Bihuan Chen, Xin Peng. (2023)  
**Exploring the Potential of ChatGPT in Automated Code Refinement: An Empirical Study**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BLEU, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.08221v1)  

---


**ABSTRACT**  
Code review is an essential activity for ensuring the quality and maintainability of software projects. However, it is a time-consuming and often error-prone task that can significantly impact the development process. Recently, ChatGPT, a cutting-edge language model, has demonstrated impressive performance in various natural language processing tasks, suggesting its potential to automate code review processes. However, it is still unclear how well ChatGPT performs in code review tasks. To fill this gap, in this paper, we conduct the first empirical study to understand the capabilities of ChatGPT in code review tasks, specifically focusing on automated code refinement based on given code reviews. To conduct the study, we select the existing benchmark CodeReview and construct a new code review dataset with high quality. We use CodeReviewer, a state-of-the-art code review tool, as a baseline for comparison with ChatGPT. Our results show that ChatGPT outperforms CodeReviewer in code refinement tasks. Specifically, our results show that ChatGPT achieves higher EM and BLEU scores of 22.78 and 76.44 respectively, while the state-of-the-art method achieves only 15.50 and 62.88 on a high-quality code review dataset. We further identify the root causes for ChatGPT's underperformance and propose several strategies to mitigate these challenges. Our study provides insights into the potential of ChatGPT in automating the code review process, and highlights the potential research directions.

{{</citation>}}


## cs.HC (2)



### (88/91) 'I'm Not Confident in Debiasing AI Systems Since I Know Too Little': Teaching AI Creators About Gender Bias Through Hands-on Tutorials (Kyrie Zhixuan Zhou et al., 2023)

{{<citation>}}

Kyrie Zhixuan Zhou, Jiaxun Cao, Xiaowen Yuan, Daniel E. Weissglass, Zachary Kilhoffer, Madelyn Rose Sanfilippo, Xin Tong. (2023)  
**'I'm Not Confident in Debiasing AI Systems Since I Know Too Little': Teaching AI Creators About Gender Bias Through Hands-on Tutorials**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CY, cs-HC, cs.HC  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2309.08121v1)  

---


**ABSTRACT**  
Gender bias is rampant in AI systems, causing bad user experience, injustices, and mental harm to women. School curricula fail to educate AI creators on this topic, leaving them unprepared to mitigate gender bias in AI. In this paper, we designed hands-on tutorials to raise AI creators' awareness of gender bias in AI and enhance their knowledge of sources of gender bias and debiasing techniques. The tutorials were evaluated with 18 AI creators, including AI researchers, AI industrial practitioners (i.e., developers and product managers), and students who had learned AI. Their improved awareness and knowledge demonstrated the effectiveness of our tutorials, which have the potential to complement the insufficient AI gender bias education in CS/AI courses. Based on the findings, we synthesize design implications and a rubric to guide future research, education, and design efforts.

{{</citation>}}


### (89/91) Empowering Private Tutoring by Chaining Large Language Models (Yulin Chen et al., 2023)

{{<citation>}}

Yulin Chen, Ning Ding, Hai-Tao Zheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou. (2023)  
**Empowering Private Tutoring by Chaining Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.08112v1)  

---


**ABSTRACT**  
Artificial intelligence has been applied in various aspects of online education to facilitate teaching and learning. However, few approaches has been made toward a complete AI-powered tutoring system. In this work, we explore the development of a full-fledged intelligent tutoring system powered by state-of-the-art large language models (LLMs), covering automatic course planning and adjusting, tailored instruction, and flexible quiz evaluation. To make the system robust to prolonged interaction and cater to individualized education, the system is decomposed into three inter-connected core processes-interaction, reflection, and reaction. Each process is implemented by chaining LLM-powered tools along with dynamically updated memory modules. Tools are LLMs prompted to execute one specific task at a time, while memories are data storage that gets updated during education process. Statistical results from learning logs demonstrate the effectiveness and mechanism of each tool usage. Subjective feedback from human users reveal the usability of each function, and comparison with ablation systems further testify the benefits of the designed processes in long-term interaction.

{{</citation>}}


## math.NA (1)



### (90/91) Low-rank Tensor Train Decomposition Using TensorSketch (Zhongming Chen et al., 2023)

{{<citation>}}

Zhongming Chen, Huilin Jiang, Gaohang Yu, Liqun Qi. (2023)  
**Low-rank Tensor Train Decomposition Using TensorSketch**  

---
Primary Category: math.NA  
Categories: cs-NA, math-NA, math.NA  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2309.08093v1)  

---


**ABSTRACT**  
Tensor train decomposition is one of the most powerful approaches for processing high-dimensional data. For low-rank tensor train decomposition of large tensors, the alternating least squares (ALS) algorithm is widely used by updating each core tensor alternatively. However, it may suffer from the curse of dimensionality due to the large scale of subproblems. In this paper, a novel randomized proximal ALS algorithm is proposed for low-rank tensor train decomposition by using TensorSketch, which allows for efficient implementation via fast Fourier transform. The theoretical lower bounds of sketch size are estimated for approximating the optimal value of subproblems. Numerical experiments on synthetic and real-world data also demonstrate the effectiveness and efficiency of the proposed algorithm.

{{</citation>}}


## cs.SI (1)



### (91/91) Social media polarization reflects shifting political alliances in Pakistan (Anees Baqir et al., 2023)

{{<citation>}}

Anees Baqir, Alessandro Galeazzi, Andrea Drocco, Fabiana Zollo. (2023)  
**Social media polarization reflects shifting political alliances in Pakistan**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2309.08075v1)  

---


**ABSTRACT**  
The rise of ideological divides in public discourse has received considerable attention in recent years. However, much of this research has been concentrated on Western democratic nations, leaving other regions largely unexplored. Here, we delve into the political landscape of Pakistan, a nation marked by intricate political dynamics and persistent turbulence. Spanning from 2018 to 2022, our analysis of Twitter data allows us to capture pivotal shifts and developments in Pakistan's political arena. By examining interactions and content generated by politicians affiliated with major political parties, we reveal a consistent and active presence of politicians on Twitter, with opposition parties exhibiting particularly robust engagement. We explore the alignment of party audiences, highlighting a notable convergence among opposition factions over time. Our analysis also uncovers significant shifts in political affiliations, including the transition of politicians to the opposition alliance. Quantitatively, we assess evolving interaction patterns, showcasing the prevalence of homophilic connections while identifying a growing interconnection among audiences of opposition parties. Our study, by accurately reflecting shifts in the political landscape, underscores the reliability of our methodology and social media data as a valuable tool for monitoring political polarization and providing a nuanced understanding of macro-level trends and individual-level transformations.

{{</citation>}}
