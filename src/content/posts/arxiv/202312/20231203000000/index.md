---
draft: false
title: "arXiv @ 2023.12.03"
date: 2023-12-03
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.03"
    identifier: arxiv_20231203
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (32)](#cslg-32)
- [cs.NE (1)](#csne-1)
- [cs.CL (22)](#cscl-22)
- [cs.NI (1)](#csni-1)
- [cs.CV (31)](#cscv-31)
- [cs.CR (8)](#cscr-8)
- [cs.IT (1)](#csit-1)
- [cs.IR (1)](#csir-1)
- [cs.SE (1)](#csse-1)
- [cs.RO (2)](#csro-2)
- [cs.HC (4)](#cshc-4)
- [cs.AI (4)](#csai-4)
- [eess.IV (2)](#eessiv-2)
- [eess.SP (1)](#eesssp-1)
- [cs.PL (1)](#cspl-1)
- [math.NA (1)](#mathna-1)
- [cs.SD (1)](#cssd-1)
- [quant-ph (3)](#quant-ph-3)
- [q-bio.QM (1)](#q-bioqm-1)
- [astro-ph.IM (1)](#astro-phim-1)

## cs.LG (32)



### (1/119) Spectral Temporal Contrastive Learning (Sacha Morin et al., 2023)

{{<citation>}}

Sacha Morin, Somjit Nath, Samira Ebrahimi Kahou, Guy Wolf. (2023)  
**Spectral Temporal Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.00966v1)  

---


**ABSTRACT**  
Learning useful data representations without requiring labels is a cornerstone of modern deep learning. Self-supervised learning methods, particularly contrastive learning (CL), have proven successful by leveraging data augmentations to define positive pairs. This success has prompted a number of theoretical studies to better understand CL and investigate theoretical bounds for downstream linear probing tasks. This work is concerned with the temporal contrastive learning (TCL) setting where the sequential structure of the data is used instead to define positive pairs, which is more commonly used in RL and robotics contexts. In this paper, we adapt recent work on Spectral CL to formulate Spectral Temporal Contrastive Learning (STCL). We discuss a population loss based on a state graph derived from a time-homogeneous reversible Markov chain with uniform stationary distribution. The STCL loss enables to connect the linear probing performance to the spectral properties of the graph, and can be estimated by considering previously observed data sequences as an ensemble of MCMC chains.

{{</citation>}}


### (2/119) Spatiotemporal Transformer for Imputing Sparse Data: A Deep Learning Approach (Kehui Yao et al., 2023)

{{<citation>}}

Kehui Yao, Jingyi Huang, Jun Zhu. (2023)  
**Spatiotemporal Transformer for Imputing Sparse Data: A Deep Learning Approach**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ME  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.00963v1)  

---


**ABSTRACT**  
Effective management of environmental resources and agricultural sustainability heavily depends on accurate soil moisture data. However, datasets like the SMAP/Sentinel-1 soil moisture product often contain missing values across their spatiotemporal grid, which poses a significant challenge. This paper introduces a novel Spatiotemporal Transformer model (ST-Transformer) specifically designed to address the issue of missing values in sparse spatiotemporal datasets, particularly focusing on soil moisture data. The ST-Transformer employs multiple spatiotemporal attention layers to capture the complex spatiotemporal correlations in the data and can integrate additional spatiotemporal covariates during the imputation process, thereby enhancing its accuracy. The model is trained using a self-supervised approach, enabling it to autonomously predict missing values from observed data points. Our model's efficacy is demonstrated through its application to the SMAP 1km soil moisture data over a 36 x 36 km grid in Texas. It showcases superior accuracy compared to well-known imputation methods. Additionally, our simulation studies on other datasets highlight the model's broader applicability in various spatiotemporal imputation tasks.

{{</citation>}}


### (3/119) A Theory of Unimodal Bias in Multimodal Learning (Yedi Zhang et al., 2023)

{{<citation>}}

Yedi Zhang, Peter E. Latham, Andrew Saxe. (2023)  
**A Theory of Unimodal Bias in Multimodal Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.00935v1)  

---


**ABSTRACT**  
Using multiple input streams simultaneously in training multimodal neural networks is intuitively advantageous, but practically challenging. A key challenge is unimodal bias, where a network overly relies on one modality and ignores others during joint training. While unimodal bias is well-documented empirically, our theoretical understanding of how architecture and data statistics influence this bias remains incomplete. Here we develop a theory of unimodal bias with deep multimodal linear networks. We calculate the duration of the unimodal phase in learning as a function of the depth at which modalities are fused within the network, dataset statistics, and initialization. We find that the deeper the layer at which fusion occurs, the longer the unimodal phase. A long unimodal phase can lead to a generalization deficit and permanent unimodal bias in the overparametrized regime. In addition, our theory reveals the modality learned first is not necessarily the modality that contributes more to the output. Our results, derived for multimodal linear networks, extend to ReLU networks in certain settings. Taken together, this work illuminates pathologies of multimodal learning under joint training, showing that late and intermediate fusion architectures can give rise to long unimodal phases and permanent unimodal bias.

{{</citation>}}


### (4/119) Extreme Event Prediction with Multi-agent Reinforcement Learning-based Parametrization of Atmospheric and Oceanic Turbulence (Rambod Mojgani et al., 2023)

{{<citation>}}

Rambod Mojgani, Daniel Waelchli, Yifei Guan, Petros Koumoutsakos, Pedram Hassanzadeh. (2023)  
**Extreme Event Prediction with Multi-agent Reinforcement Learning-based Parametrization of Atmospheric and Oceanic Turbulence**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs.LG, physics-ao-ph, physics-comp-ph, physics-flu-dyn  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.00907v1)  

---


**ABSTRACT**  
Global climate models (GCMs) are the main tools for understanding and predicting climate change. However, due to limited numerical resolutions, these models suffer from major structural uncertainties; e.g., they cannot resolve critical processes such as small-scale eddies in atmospheric and oceanic turbulence. Thus, such small-scale processes have to be represented as a function of the resolved scales via closures (parametrization). The accuracy of these closures is particularly important for capturing climate extremes. Traditionally, such closures are based on heuristics and simplifying assumptions about the unresolved physics. Recently, supervised-learned closures, trained offline on high-fidelity data, have been shown to outperform the classical physics-based closures. However, this approach requires a significant amount of high-fidelity training data and can also lead to instabilities. Reinforcement learning is emerging as a potent alternative for developing such closures as it requires only low-order statistics and leads to stable closures. In Scientific Multi-Agent Reinforcement Learning (SMARL) computational elements serve a dual role of discretization points and learning agents. We leverage SMARL and fundamentals of turbulence physics to learn closures for prototypes of atmospheric and oceanic turbulence. The policy is trained using only the enstrophy spectrum, which is nearly invariant and can be estimated from a few high-fidelity samples (these few samples are far from enough for supervised/offline learning). We show that these closures lead to stable low-resolution simulations that, at a fraction of the cost, can reproduce the high-fidelity simulations' statistics, including the tails of the probability density functions. The results demonstrate the high potential of SMARL for closure modeling for GCMs, especially in the regime of scarce data and indirect observations.

{{</citation>}}


### (5/119) Explaining Knock-on Effects of Bias Mitigation (Svetoslav Nizhnichenkov et al., 2023)

{{<citation>}}

Svetoslav Nizhnichenkov, Rahul Nair, Elizabeth Daly, Brian Mac Namee. (2023)  
**Explaining Knock-on Effects of Bias Mitigation**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.00765v1)  

---


**ABSTRACT**  
In machine learning systems, bias mitigation approaches aim to make outcomes fairer across privileged and unprivileged groups. Bias mitigation methods work in different ways and have known "waterfall" effects, e.g., mitigating bias at one place may manifest bias elsewhere. In this paper, we aim to characterise impacted cohorts when mitigation interventions are applied. To do so, we treat intervention effects as a classification task and learn an explainable meta-classifier to identify cohorts that have altered outcomes. We examine a range of bias mitigation strategies that work at various stages of the model life cycle. We empirically demonstrate that our meta-classifier is able to uncover impacted cohorts. Further, we show that all tested mitigation strategies negatively impact a non-trivial fraction of cases, i.e., people who receive unfavourable outcomes solely on account of mitigation efforts. This is despite improvement in fairness metrics. We use these results as a basis to argue for more careful audits of static mitigation interventions that go beyond aggregate metrics.

{{</citation>}}


### (6/119) Deep Unlearning: Fast and Efficient Training-free Approach to Controlled Forgetting (Sangamesh Kodge et al., 2023)

{{<citation>}}

Sangamesh Kodge, Gobinda Saha, Kaushik Roy. (2023)  
**Deep Unlearning: Fast and Efficient Training-free Approach to Controlled Forgetting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG, stat-ML  
Keywords: ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2312.00761v2)  

---


**ABSTRACT**  
Machine unlearning has emerged as a prominent and challenging area of interest, driven in large part by the rising regulatory demands for industries to delete user data upon request and the heightened awareness of privacy. Existing approaches either retrain models from scratch or use several finetuning steps for every deletion request, often constrained by computational resource limitations and restricted access to the original training data. In this work, we introduce a novel class unlearning algorithm designed to strategically eliminate an entire class or a group of classes from the learned model. To that end, our algorithm first estimates the Retain Space and the Forget Space, representing the feature or activation spaces for samples from classes to be retained and unlearned, respectively. To obtain these spaces, we propose a novel singular value decomposition-based technique that requires layer wise collection of network activations from a few forward passes through the network. We then compute the shared information between these spaces and remove it from the forget space to isolate class-discriminatory feature space for unlearning. Finally, we project the model weights in the orthogonal direction of the class-discriminatory space to obtain the unlearned model. We demonstrate our algorithm's efficacy on ImageNet using a Vision Transformer with only $\sim$1.5% drop in retain accuracy compared to the original model while maintaining under 1% accuracy on the unlearned class samples. Further, our algorithm consistently performs well when subject to Membership Inference Attacks showing 7.8% improvement on average across a variety of image classification datasets and network architectures, as compared to other baselines while being $\sim$6x more computationally efficient.

{{</citation>}}


### (7/119) Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu et al., 2023)

{{<citation>}}

Albert Gu, Tri Dao. (2023)  
**Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.00752v1)  

---


**ABSTRACT**  
Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language. We identify that a key weakness of such models is their inability to perform content-based reasoning, and make several improvements. First, simply letting the SSM parameters be functions of the input addresses their weakness with discrete modalities, allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token. Second, even though this change prevents the use of efficient convolutions, we design a hardware-aware parallel algorithm in recurrent mode. We integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba). Mamba enjoys fast inference (5$\times$ higher throughput than Transformers) and linear scaling in sequence length, and its performance improves on real data up to million-length sequences. As a general sequence model backbone, Mamba achieves state-of-the-art performance across several modalities such as language, audio, and genomics. On language modeling, our Mamba-3B model outperforms Transformers of the same size and matches Transformers twice its size, both in pretraining and downstream evaluation.

{{</citation>}}


### (8/119) Safe Reinforcement Learning in Tensor Reproducing Kernel Hilbert Space (Xiaoyuan Cheng et al., 2023)

{{<citation>}}

Xiaoyuan Cheng, Boli Chen, Liz Varga, Yukun Hu. (2023)  
**Safe Reinforcement Learning in Tensor Reproducing Kernel Hilbert Space**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.00727v1)  

---


**ABSTRACT**  
This paper delves into the problem of safe reinforcement learning (RL) in a partially observable environment with the aim of achieving safe-reachability objectives. In traditional partially observable Markov decision processes (POMDP), ensuring safety typically involves estimating the belief in latent states. However, accurately estimating an optimal Bayesian filter in POMDP to infer latent states from observations in a continuous state space poses a significant challenge, largely due to the intractable likelihood. To tackle this issue, we propose a stochastic model-based approach that guarantees RL safety almost surely in the face of unknown system dynamics and partial observation environments. We leveraged the Predictive State Representation (PSR) and Reproducing Kernel Hilbert Space (RKHS) to represent future multi-step observations analytically, and the results in this context are provable. Furthermore, we derived essential operators from the kernel Bayes' rule, enabling the recursive estimation of future observations using various operators. Under the assumption of \textit{undercompleness}, a polynomial sample complexity is established for the RL algorithm for the infinite size of observation and action spaces, ensuring an $\epsilon-$suboptimal safe policy guarantee.

{{</citation>}}


### (9/119) Removing Biases from Molecular Representations via Information Maximization (Chenyu Wang et al., 2023)

{{<citation>}}

Chenyu Wang, Sharut Gupta, Caroline Uhler, Tommi Jaakkola. (2023)  
**Removing Biases from Molecular Representations via Information Maximization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-bio-BM  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.00718v1)  

---


**ABSTRACT**  
High-throughput drug screening -- using cell imaging or gene expression measurements as readouts of drug effect -- is a critical tool in biotechnology to assess and understand the relationship between the chemical structure and biological activity of a drug. Since large-scale screens have to be divided into multiple experiments, a key difficulty is dealing with batch effects, which can introduce systematic errors and non-biological associations in the data. We propose InfoCORE, an Information maximization approach for COnfounder REmoval, to effectively deal with batch effects and obtain refined molecular representations. InfoCORE establishes a variational lower bound on the conditional mutual information of the latent representations given a batch identifier. It adaptively reweighs samples to equalize their implied batch distribution. Extensive experiments on drug screening data reveal InfoCORE's superior performance in a multitude of tasks including molecular property prediction and molecule-phenotype retrieval. Additionally, we show results for how InfoCORE offers a versatile framework and resolves general distribution shifts and issues of data fairness by minimizing correlation with spurious features or removing sensitive attributes. The code is available at https://github.com/uhlerlab/InfoCORE.

{{</citation>}}


### (10/119) Nonparametric Variational Regularisation of Pretrained Transformers (Fabio Fehr et al., 2023)

{{<citation>}}

Fabio Fehr, James Henderson. (2023)  
**Nonparametric Variational Regularisation of Pretrained Transformers**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.00662v1)  

---


**ABSTRACT**  
The current paradigm of large-scale pre-training and fine-tuning Transformer large language models has lead to significant improvements across the board in natural language processing. However, such large models are susceptible to overfitting to their training data, and as a result the models perform poorly when the domain changes. Also, due to the model's scale, the cost of fine-tuning the model to the new domain is large. Nonparametric Variational Information Bottleneck (NVIB) has been proposed as a regulariser for training cross-attention in Transformers, potentially addressing the overfitting problem. We extend the NVIB framework to replace all types of attention functions in Transformers, and show that existing pretrained Transformers can be reinterpreted as Nonparametric Variational (NV) models using a proposed identity initialisation. We then show that changing the initialisation introduces a novel, information-theoretic post-training regularisation in the attention mechanism, which improves out-of-domain generalisation without any training. This success supports the hypothesis that pretrained Transformers are implicitly NV Bayesian models.

{{</citation>}}


### (11/119) Hashmarks: Privacy-Preserving Benchmarks for High-Stakes AI Evaluation (Paul Bricman, 2023)

{{<citation>}}

Paul Bricman. (2023)  
**Hashmarks: Privacy-Preserving Benchmarks for High-Stakes AI Evaluation**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs-SE, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00645v1)  

---


**ABSTRACT**  
There is a growing need to gain insight into language model capabilities that relate to sensitive topics, such as bioterrorism or cyberwarfare. However, traditional open source benchmarks are not fit for the task, due to the associated practice of publishing the correct answers in human-readable form. At the same time, enforcing mandatory closed-quarters evaluations might stifle development and erode trust. In this context, we propose hashmarking, a protocol for evaluating language models in the open without having to disclose the correct answers. In its simplest form, a hashmark is a benchmark whose reference solutions have been cryptographically hashed prior to publication. Following an overview of the proposed evaluation protocol, we go on to assess its resilience against traditional attack vectors (e.g. rainbow table attacks), as well as against failure modes unique to increasingly capable generative models.

{{</citation>}}


### (12/119) Forecasting Trends in Food Security: a Reservoir Computing Approach (Joschka Herteux et al., 2023)

{{<citation>}}

Joschka Herteux, Christoph Räth, Amine Baha, Giulia Martini, Duccio Piovani. (2023)  
**Forecasting Trends in Food Security: a Reservoir Computing Approach**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-soc-ph, stat-ML  
Keywords: LSTM, Security  
[Paper Link](http://arxiv.org/abs/2312.00626v1)  

---


**ABSTRACT**  
Early warning systems are an essential tool for effective humanitarian action. Advance warnings on impending disasters facilitate timely and targeted response which help save lives, livelihoods, and scarce financial resources. In this work we present a new quantitative methodology to forecast levels of food consumption for 60 consecutive days, at the sub-national level, in four countries: Mali, Nigeria, Syria, and Yemen. The methodology is built on publicly available data from the World Food Programme's integrated global hunger monitoring system which collects, processes, and displays daily updates on key food security metrics, conflict, weather events, and other drivers of food insecurity across 90 countries (https://hungermap.wfp.org/). In this study, we assessed the performance of various models including ARIMA, XGBoost, LSTMs, CNNs, and Reservoir Computing (RC), by comparing their Root Mean Squared Error (RMSE) metrics. This comprehensive analysis spanned classical statistical, machine learning, and deep learning approaches. Our findings highlight Reservoir Computing as a particularly well-suited model in the field of food security given both its notable resistance to over-fitting on limited data samples and its efficient training capabilities. The methodology we introduce establishes the groundwork for a global, data-driven early warning system designed to anticipate and detect food insecurity.

{{</citation>}}


### (13/119) Beyond First-Order Tweedie: Solving Inverse Problems using Latent Diffusion (Litu Rout et al., 2023)

{{<citation>}}

Litu Rout, Yujia Chen, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, Wen-Sheng Chu. (2023)  
**Beyond First-Order Tweedie: Solving Inverse Problems using Latent Diffusion**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG, stat-ML  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.00852v1)  

---


**ABSTRACT**  
Sampling from the posterior distribution poses a major computational challenge in solving inverse problems using latent diffusion models. Common methods rely on Tweedie's first-order moments, which are known to induce a quality-limiting bias. Existing second-order approximations are impractical due to prohibitive computational costs, making standard reverse diffusion processes intractable for posterior sampling. This paper introduces Second-order Tweedie sampler from Surrogate Loss (STSL), a novel sampler that offers efficiency comparable to first-order Tweedie with a tractable reverse process using second-order approximation. Our theoretical results reveal that the second-order approximation is lower bounded by our surrogate loss that only requires $O(1)$ compute using the trace of the Hessian, and by the lower bound we derive a new drift term to make the reverse process tractable. Our method surpasses SoTA solvers PSLD and P2L, achieving 4X and 8X reduction in neural function evaluations, respectively, while notably enhancing sampling quality on FFHQ, ImageNet, and COCO benchmarks. In addition, we show STSL extends to text-guided image editing and addresses residual distortions present from corrupted images in leading text-guided image editing methods. To our best knowledge, this is the first work to offer an efficient second-order approximation in solving inverse problems using latent diffusion and editing real-world images with corruptions.

{{</citation>}}


### (14/119) Tracking Object Positions in Reinforcement Learning: A Metric for Keypoint Detection (extended version) (Emma Cramer et al., 2023)

{{<citation>}}

Emma Cramer, Jonas Reiher, Sebastian Trimpe. (2023)  
**Tracking Object Positions in Reinforcement Learning: A Metric for Keypoint Detection (extended version)**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.00592v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) for robot control typically requires a detailed representation of the environment state, including information about task-relevant objects not directly measurable. Keypoint detectors, such as spatial autoencoders (SAEs), are a common approach to extracting a low-dimensional representation from high-dimensional image data. SAEs aim at spatial features such as object positions, which are often useful representations in robotic RL. However, whether an SAE is actually able to track objects in the scene and thus yields a spatial state representation well suited for RL tasks has rarely been examined due to a lack of established metrics. In this paper, we propose to assess the performance of an SAE instance by measuring how well keypoints track ground truth objects in images. We present a computationally lightweight metric and use it to evaluate common baseline SAE architectures on image data from a simulated robot task. We find that common SAEs differ substantially in their spatial extraction capability. Furthermore, we validate that SAEs that perform well in our metric achieve superior performance when used in downstream RL. Thus, our metric is an effective and lightweight indicator of RL performance before executing expensive RL training. Building on these insights, we identify three key modifications of SAE architectures to improve tracking performance. We make our code available at anonymous.4open.science/r/sae-rl.

{{</citation>}}


### (15/119) Explainable Fraud Detection with Deep Symbolic Classification (Samantha Visbeek et al., 2023)

{{<citation>}}

Samantha Visbeek, Erman Acar, Floris den Hengst. (2023)  
**Explainable Fraud Detection with Deep Symbolic Classification**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Fraud Detection  
[Paper Link](http://arxiv.org/abs/2312.00586v1)  

---


**ABSTRACT**  
There is a growing demand for explainable, transparent, and data-driven models within the domain of fraud detection. Decisions made by fraud detection models need to be explainable in the event of a customer dispute. Additionally, the decision-making process in the model must be transparent to win the trust of regulators and business stakeholders. At the same time, fraud detection solutions can benefit from data due to the noisy, dynamic nature of fraud and the availability of large historical data sets. Finally, fraud detection is notorious for its class imbalance: there are typically several orders of magnitude more legitimate transactions than fraudulent ones. In this paper, we present Deep Symbolic Classification (DSC), an extension of the Deep Symbolic Regression framework to classification problems. DSC casts classification as a search problem in the space of all analytic functions composed of a vocabulary of variables, constants, and operations and optimizes for an arbitrary evaluation metric directly. The search is guided by a deep neural network trained with reinforcement learning. Because the functions are mathematical expressions that are in closed-form and concise, the model is inherently explainable both at the level of a single classification decision and the model's decision process. Furthermore, the class imbalance problem is successfully addressed by optimizing for metrics that are robust to class imbalance such as the F1 score. This eliminates the need for oversampling and undersampling techniques that plague traditional approaches. Finally, the model allows to explicitly balance between the prediction accuracy and the explainability. An evaluation on the PaySim data set demonstrates competitive predictive performance with state-of-the-art models, while surpassing them in terms of explainability. This establishes DSC as a promising model for fraud detection systems.

{{</citation>}}


### (16/119) Physics Inspired Criterion for Pruning-Quantization Joint Learning (Weiying Xie et al., 2023)

{{<citation>}}

Weiying Xie, Xiaoyi Fan, Xin Zhang, Yunsong Li, Jie Lei, Leyuan Fang. (2023)  
**Physics Inspired Criterion for Pruning-Quantization Joint Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet, Pruning, Quantization  
[Paper Link](http://arxiv.org/abs/2312.00851v1)  

---


**ABSTRACT**  
Pruning-quantization joint learning always facilitates the deployment of deep neural networks (DNNs) on resource-constrained edge devices. However, most existing methods do not jointly learn a global criterion for pruning and quantization in an interpretable way. In this paper, we propose a novel physics inspired criterion for pruning-quantization joint learning (PIC-PQ), which is explored from an analogy we first draw between elasticity dynamics (ED) and model compression (MC). Specifically, derived from Hooke's law in ED, we establish a linear relationship between the filters' importance distribution and the filter property (FP) by a learnable deformation scale in the physics inspired criterion (PIC). Furthermore, we extend PIC with a relative shift variable for a global view. To ensure feasibility and flexibility, available maximum bitwidth and penalty factor are introduced in quantization bitwidth assignment. Experiments on benchmarks of image classification demonstrate that PIC-PQ yields a good trade-off between accuracy and bit-operations (BOPs) compression ratio e.g., 54.96X BOPs compression ratio in ResNet56 on CIFAR10 with 0.10% accuracy drop and 53.24X in ResNet18 on ImageNet with 0.61% accuracy drop). The code will be available at https://github.com/fanxxxxyi/PIC-PQ.

{{</citation>}}


### (17/119) Interior Point Constrained Reinforcement Learning with Global Convergence Guarantees (Tingting Ni et al., 2023)

{{<citation>}}

Tingting Ni, Maryam Kamgarpour. (2023)  
**Interior Point Constrained Reinforcement Learning with Global Convergence Guarantees**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.00561v1)  

---


**ABSTRACT**  
We consider discounted infinite horizon constrained Markov decision processes (CMDPs) where the goal is to find an optimal policy that maximizes the expected cumulative reward subject to expected cumulative constraints. Motivated by the application of CMDPs in online learning of safety-critical systems, we focus on developing an algorithm that ensures constraint satisfaction during learning. To this end, we develop a zeroth-order interior point approach based on the log barrier function of the CMDP. Under the commonly assumed conditions of Fisher non-degeneracy and bounded transfer error of the policy parameterization, we establish the theoretical properties of the algorithm. In particular, in contrast to existing CMDP approaches that ensure policy feasibility only upon convergence, our algorithm guarantees feasibility of the policies during the learning process and converges to the optimal policy with a sample complexity of $O(\varepsilon^{-6})$. In comparison to the state-of-the-art policy gradient-based algorithm, C-NPG-PDA, our algorithm requires an additional $O(\varepsilon^{-2})$ samples to ensure policy feasibility during learning with same Fisher-non-degenerate parameterization.

{{</citation>}}


### (18/119) On the Out-Of-Distribution Robustness of Self-Supervised Representation Learning for Phonocardiogram Signals (Aristotelis Ballas et al., 2023)

{{<citation>}}

Aristotelis Ballas, Vasileios Papapanagiotou, Christos Diou. (2023)  
**On the Out-Of-Distribution Robustness of Self-Supervised Representation Learning for Phonocardiogram Signals**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SD, cs.LG, q-bio-QM  
Keywords: Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.00502v1)  

---


**ABSTRACT**  
Objective: Despite the recent increase in research activity, deep-learning models have not yet been widely accepted in medicine. The shortage of high-quality annotated data often hinders the development of robust and generalizable models, which do not suffer from degraded effectiveness when presented with newly-collected, out-of-distribution (OOD) datasets. Methods: Contrastive Self-Supervised Learning (SSL) offers a potential solution to the scarcity of labeled data as it takes advantage of unlabeled data to increase model effectiveness and robustness. In this research, we propose applying contrastive SSL for detecting abnormalities in phonocardiogram (PCG) samples by learning a generalized representation of the signal. Specifically, we perform an extensive comparative evaluation of a wide range of audio-based augmentations and evaluate trained classifiers on multiple datasets across different downstream tasks. Results: We experimentally demonstrate that, depending on its training distribution, the effectiveness of a fully-supervised model can degrade up to 32% when evaluated on unseen data, while SSL models only lose up to 10% or even improve in some cases. Conclusions: Contrastive SSL pretraining can assist in providing robust classifiers which can generalize to unseen, OOD data, without relying on time- and labor-intensive annotation processes by medical experts. Furthermore, the proposed extensive evaluation protocol sheds light on the most promising and appropriate augmentations for robust PCG signal processing. Significance: We provide researchers and practitioners with a roadmap towards producing robust models for PCG classification, in addition to an open-source codebase for developing novel approaches.

{{</citation>}}


### (19/119) A Bayesian approach for prompt optimization in pre-trained language models (Antonio Sabbatella et al., 2023)

{{<citation>}}

Antonio Sabbatella, Andrea Ponti, Antonio Candelieri, Ilaria Giordani, Francesco Archetti. (2023)  
**A Bayesian approach for prompt optimization in pre-trained language models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.00471v1)  

---


**ABSTRACT**  
A prompt is a sequence of symbol or tokens, selected from a vocabulary according to some rule, which is prepended/concatenated to a textual query. A key problem is how to select the sequence of tokens: in this paper we formulate it as a combinatorial optimization problem. The high dimensionality of the token space com-pounded by the length of the prompt sequence requires a very efficient solution. In this paper we propose a Bayesian optimization method, executed in a continuous em-bedding of the combinatorial space. In this paper we focus on hard prompt tuning (HPT) which directly searches for discrete tokens to be added to the text input with-out requiring access to the large language model (LLM) and can be used also when LLM is available only as a black-box. This is critically important if LLMs are made available in the Model as a Service (MaaS) manner as in GPT-4. The current manu-script is focused on the optimization of discrete prompts for classification tasks. The discrete prompts give rise to difficult combinatorial optimization problem which easily become intractable given the dimension of the token space in realistic applications. The optimization method considered in this paper is Bayesian optimization (BO) which has become the dominant approach in black-box optimization for its sample efficiency along with its modular structure and versatility. In this paper we use BoTorch, a library for Bayesian optimization research built on top of pyTorch. Albeit preliminary and obtained using a 'vanilla' version of BO, the experiments on RoB-ERTa on six benchmarks, show a good performance across a variety of tasks and enable an analysis of the tradeoff between size of the search space, accuracy and wall clock time.

{{</citation>}}


### (20/119) LinguaLinked: A Distributed Large Language Model Inference System for Mobile Devices (Junchen Zhao et al., 2023)

{{<citation>}}

Junchen Zhao, Yurun Song, Simeng Liu, Ian G. Harris, Sangeetha Abdu Jyothi. (2023)  
**LinguaLinked: A Distributed Large Language Model Inference System for Mobile Devices**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs-NI, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00388v1)  

---


**ABSTRACT**  
Deploying Large Language Models (LLMs) locally on mobile devices presents a significant challenge due to their extensive memory requirements. In this paper, we introduce LinguaLinked, a system for decentralized, distributed LLM inference on mobile devices. LinguaLinked enables collaborative execution of the inference task across multiple trusted devices. LinguaLinked ensures data privacy by processing information locally. LinguaLinked uses three key strategies. First, an optimized model assignment technique segments LLMs and uses linear optimization to align segments with each device's capabilities. Second, an optimized data transmission mechanism ensures efficient and structured data flow between model segments while also maintaining the integrity of the original model structure. Finally, LinguaLinked incorporates a runtime load balancer that actively monitors and redistributes tasks among mobile devices to prevent bottlenecks, enhancing the system's overall efficiency and responsiveness. We demonstrate that LinguaLinked facilitates efficient LLM inference while maintaining consistent throughput and minimal latency through extensive testing across various mobile devices, from high-end to low-end Android devices. In our evaluations, compared to the baseline, LinguaLinked achieves an inference performance acceleration of $1.11\times$ to $1.61\times$ in single-threaded settings, $1.73\times$ to $2.65\times$ with multi-threading. Additionally, runtime load balancing yields an overall inference acceleration of $1.29\times$ to $1.32\times$.

{{</citation>}}


### (21/119) Optimal Sample Complexity of Contrastive Learning (Noga Alon et al., 2023)

{{<citation>}}

Noga Alon, Dmitrii Avdiukhin, Dor Elboim, Orr Fischer, Grigory Yaroslavtsev. (2023)  
**Optimal Sample Complexity of Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.00379v1)  

---


**ABSTRACT**  
Contrastive learning is a highly successful technique for learning representations of data from labeled tuples, specifying the distance relations within the tuple. We study the sample complexity of contrastive learning, i.e. the minimum number of labeled tuples sufficient for getting high generalization accuracy. We give tight bounds on the sample complexity in a variety of settings, focusing on arbitrary distance functions, both general $\ell_p$-distances, and tree metrics. Our main result is an (almost) optimal bound on the sample complexity of learning $\ell_p$-distances for integer $p$. For any $p \ge 1$ we show that $\tilde \Theta(\min(nd,n^2))$ labeled tuples are necessary and sufficient for learning $d$-dimensional representations of $n$-point datasets. Our results hold for an arbitrary distribution of the input samples and are based on giving the corresponding bounds on the Vapnik-Chervonenkis/Natarajan dimension of the associated problems. We further show that the theoretical bounds on sample complexity obtained via VC/Natarajan dimension can have strong predictive power for experimental results, in contrast with the folklore belief about a substantial gap between the statistical learning theory and the practice of deep learning.

{{</citation>}}


### (22/119) Streaming Bayesian Modeling for predicting Fat-Tailed Customer Lifetime Value (Alexey V. Calabourdin et al., 2023)

{{<citation>}}

Alexey V. Calabourdin, Konstantin A. Aksenov. (2023)  
**Streaming Bayesian Modeling for predicting Fat-Tailed Customer Lifetime Value**  

---
Primary Category: cs.LG  
Categories: 62C10, 62F15, cs-LG, cs.LG, stat-AP, stat-ME  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2312.00373v1)  

---


**ABSTRACT**  
We develop an online learning MCMC approach applicable for hierarchical bayesian models and GLMS. We also develop a fat-tailed LTV model that generalizes over several kinds of fat and thin tails. We demonstrate both developments on commercial LTV data from a large mobile app.

{{</citation>}}


### (23/119) Benchmarking Multi-Domain Active Learning on Image Classification (Jiayi Li et al., 2023)

{{<citation>}}

Jiayi Li, Rohan Taori, Tatsunori B. Hashimoto. (2023)  
**Benchmarking Multi-Domain Active Learning on Image Classification**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Active Learning, Image Classification  
[Paper Link](http://arxiv.org/abs/2312.00364v1)  

---


**ABSTRACT**  
Active learning aims to enhance model performance by strategically labeling informative data points. While extensively studied, its effectiveness on large-scale, real-world datasets remains underexplored. Existing research primarily focuses on single-source data, ignoring the multi-domain nature of real-world data. We introduce a multi-domain active learning benchmark to bridge this gap. Our benchmark demonstrates that traditional single-domain active learning strategies are often less effective than random selection in multi-domain scenarios. We also introduce CLIP-GeoYFCC, a novel large-scale image dataset built around geographical domains, in contrast to existing genre-based domain datasets. Analysis on our benchmark shows that all multi-domain strategies exhibit significant tradeoffs, with no strategy outperforming across all datasets or all metrics, emphasizing the need for future research.

{{</citation>}}


### (24/119) Temperature Balancing, Layer-wise Weight Analysis, and Neural Network Training (Yefan Zhou et al., 2023)

{{<citation>}}

Yefan Zhou, Tianyu Pang, Keqin Liu, Charles H. Martin, Michael W. Mahoney, Yaoqing Yang. (2023)  
**Temperature Balancing, Layer-wise Weight Analysis, and Neural Network Training**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.00359v1)  

---


**ABSTRACT**  
Regularization in modern machine learning is crucial, and it can take various forms in algorithmic design: training set, model family, error function, regularization terms, and optimizations. In particular, the learning rate, which can be interpreted as a temperature-like parameter within the statistical mechanics of learning, plays a crucial role in neural network training. Indeed, many widely adopted training strategies basically just define the decay of the learning rate over time. This process can be interpreted as decreasing a temperature, using either a global learning rate (for the entire model) or a learning rate that varies for each parameter. This paper proposes TempBalance, a straightforward yet effective layer-wise learning rate method. TempBalance is based on Heavy-Tailed Self-Regularization (HT-SR) Theory, an approach which characterizes the implicit self-regularization of different layers in trained models. We demonstrate the efficacy of using HT-SR-motivated metrics to guide the scheduling and balancing of temperature across all network layers during model training, resulting in improved performance during testing. We implement TempBalance on CIFAR10, CIFAR100, SVHN, and TinyImageNet datasets using ResNets, VGGs, and WideResNets with various depths and widths. Our results show that TempBalance significantly outperforms ordinary SGD and carefully-tuned spectral norm regularization. We also show that TempBalance outperforms a number of state-of-the-art optimizers and learning rate schedulers.

{{</citation>}}


### (25/119) Efficient Off-Policy Safe Reinforcement Learning Using Trust Region Conditional Value at Risk (Dohyeong Kim et al., 2023)

{{<citation>}}

Dohyeong Kim, Songhwai Oh. (2023)  
**Efficient Off-Policy Safe Reinforcement Learning Using Trust Region Conditional Value at Risk**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.00342v1)  

---


**ABSTRACT**  
This paper aims to solve a safe reinforcement learning (RL) problem with risk measure-based constraints. As risk measures, such as conditional value at risk (CVaR), focus on the tail distribution of cost signals, constraining risk measures can effectively prevent a failure in the worst case. An on-policy safe RL method, called TRC, deals with a CVaR-constrained RL problem using a trust region method and can generate policies with almost zero constraint violations with high returns. However, to achieve outstanding performance in complex environments and satisfy safety constraints quickly, RL methods are required to be sample efficient. To this end, we propose an off-policy safe RL method with CVaR constraints, called off-policy TRC. If off-policy data from replay buffers is directly used to train TRC, the estimation error caused by the distributional shift results in performance degradation. To resolve this issue, we propose novel surrogate functions, in which the effect of the distributional shift can be reduced, and introduce an adaptive trust-region constraint to ensure a policy not to deviate far from replay buffers. The proposed method has been evaluated in simulation and real-world environments and satisfied safety constraints within a few steps while achieving high returns even in complex robotic tasks.

{{</citation>}}


### (26/119) Hypergraph Node Representation Learning with One-Stage Message Passing (Shilin Qu et al., 2023)

{{<citation>}}

Shilin Qu, Weiqing Wang, Yuan-Fang Li, Xin Zhou, Fajie Yuan. (2023)  
**Hypergraph Node Representation Learning with One-Stage Message Passing**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: Representation Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.00336v1)  

---


**ABSTRACT**  
Hypergraphs as an expressive and general structure have attracted considerable attention from various research domains. Most existing hypergraph node representation learning techniques are based on graph neural networks, and thus adopt the two-stage message passing paradigm (i.e. node -> hyperedge -> node). This paradigm only focuses on local information propagation and does not effectively take into account global information, resulting in less optimal representations. Our theoretical analysis of representative two-stage message passing methods shows that, mathematically, they model different ways of local message passing through hyperedges, and can be unified into one-stage message passing (i.e. node -> node). However, they still only model local information. Motivated by this theoretical analysis, we propose a novel one-stage message passing paradigm to model both global and local information propagation for hypergraphs. We integrate this paradigm into HGraphormer, a Transformer-based framework for hypergraph node representation learning. HGraphormer injects the hypergraph structure information (local information) into Transformers (global information) by combining the attention matrix and hypergraph Laplacian. Extensive experiments demonstrate that HGraphormer outperforms recent hypergraph learning methods on five representative benchmark datasets on the semi-supervised hypernode classification task, setting new state-of-the-art performance, with accuracy improvements between 2.52% and 6.70%. Our code and datasets are available.

{{</citation>}}


### (27/119) Exploring the Robustness of Decentralized Training for Large Language Models (Lin Lu et al., 2023)

{{<citation>}}

Lin Lu, Chenxi Dai, Wangcheng Tao, Binhang Yuan, Yanan Sun, Pan Zhou. (2023)  
**Exploring the Robustness of Decentralized Training for Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00843v1)  

---


**ABSTRACT**  
Decentralized training of large language models has emerged as an effective way to democratize this technology. However, the potential threats associated with this approach have not been carefully discussed, which would hinder the development of decentralized training infrastructures. This paper aims to initiate discussion towards this end by exploring the robustness of decentralized training from three main perspectives. First, we demonstrate the vulnerabilities inherent in decentralized training frameworks in terms of hardware, data, and models. Second, we highlight the fundamental difference between decentralized foundation model training and vanilla federated learning, where the security techniques employed in federated learning cannot be applied directly. Third, we discuss the essential components required for a robust and efficient decentralized training framework and present a case study by modeling a concrete threat model. Our objective in this vision paper is to emphasize the importance of addressing security concerns in the context of decentralized training for large language models.

{{</citation>}}


### (28/119) Developmental Pretraining (DPT) for Image Classification Networks (Niranjan Rajesh et al., 2023)

{{<citation>}}

Niranjan Rajesh, Debayan Gupta. (2023)  
**Developmental Pretraining (DPT) for Image Classification Networks**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2312.00304v1)  

---


**ABSTRACT**  
In the backdrop of increasing data requirements of Deep Neural Networks for object recognition that is growing more untenable by the day, we present Developmental PreTraining (DPT) as a possible solution. DPT is designed as a curriculum-based pre-training approach designed to rival traditional pre-training techniques that are data-hungry. These training approaches also introduce unnecessary features that could be misleading when the network is employed in a downstream classification task where the data is sufficiently different from the pre-training data and is scarce. We design the curriculum for DPT by drawing inspiration from human infant visual development. DPT employs a phased approach where carefully-selected primitive and universal features like edges and shapes are taught to the network participating in our pre-training regime. A model that underwent the DPT regime is tested against models with randomised weights to evaluate the viability of DPT.

{{</citation>}}


### (29/119) Age-Based Scheduling for Mobile Edge Computing: A Deep Reinforcement Learning Approach (Xingqiu He et al., 2023)

{{<citation>}}

Xingqiu He, Chaoqun You, Tony Q. S. Quek. (2023)  
**Age-Based Scheduling for Mobile Edge Computing: A Deep Reinforcement Learning Approach**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NI, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.00279v1)  

---


**ABSTRACT**  
With the rapid development of Mobile Edge Computing (MEC), various real-time applications have been deployed to benefit people's daily lives. The performance of these applications relies heavily on the freshness of collected environmental information, which can be quantified by its Age of Information (AoI). In the traditional definition of AoI, it is assumed that the status information can be actively sampled and directly used. However, for many MEC-enabled applications, the desired status information is updated in an event-driven manner and necessitates data processing. To better serve these applications, we propose a new definition of AoI and, based on the redefined AoI, we formulate an online AoI minimization problem for MEC systems. Notably, the problem can be interpreted as a Markov Decision Process (MDP), thus enabling its solution through Reinforcement Learning (RL) algorithms. Nevertheless, the traditional RL algorithms are designed for MDPs with completely unknown system dynamics and hence usually suffer long convergence times. To accelerate the learning process, we introduce Post-Decision States (PDSs) to exploit the partial knowledge of the system's dynamics. We also combine PDSs with deep RL to further improve the algorithm's applicability, scalability, and robustness. Numerical results demonstrate that our algorithm outperforms the benchmarks under various scenarios.

{{</citation>}}


### (30/119) Text Attribute Control via Closed-Loop Disentanglement (Lei Sha et al., 2023)

{{<citation>}}

Lei Sha, Thomas Lukasiewicz. (2023)  
**Text Attribute Control via Closed-Loop Disentanglement**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2312.00277v1)  

---


**ABSTRACT**  
Changing an attribute of a text without changing the content usually requires to first disentangle the text into irrelevant attributes and content representations. After that, in the inference phase, the representation of one attribute is tuned to a different value, expecting that the corresponding attribute of the text can also be changed accordingly. The usual way of disentanglement is to add some constraints on the latent space of an encoder-decoder architecture, including adversarial-based constraints and mutual-information-based constraints. However, the previous semi-supervised processes of attribute change are usually not enough to guarantee the success of attribute change and content preservation. In this paper, we propose a novel approach to achieve a robust control of attributes while enhancing content preservation. In this approach, we use a semi-supervised contrastive learning method to encourage the disentanglement of attributes in latent spaces. Differently from previous works, we re-disentangle the reconstructed sentence and compare the re-disentangled latent space with the original latent space, which makes a closed-loop disentanglement process. This also helps content preservation. In addition, the contrastive learning method is also able to replace the role of minimizing mutual information and adversarial training in the disentanglement process, which alleviates the computation cost. We conducted experiments on three text datasets, including the Yelp Service review dataset, the Amazon Product review dataset, and the GoEmotions dataset. The experimental results show the effectiveness of our model.

{{</citation>}}


### (31/119) Towards Clinical Prediction with Transparency: An Explainable AI Approach to Survival Modelling in Residential Aged Care (Teo Susnjak et al., 2023)

{{<citation>}}

Teo Susnjak, Elise Griffin, Mitchell McCutcheon, Kathleen Potter. (2023)  
**Towards Clinical Prediction with Transparency: An Explainable AI Approach to Survival Modelling in Residential Aged Care**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Clinical  
[Paper Link](http://arxiv.org/abs/2312.00271v1)  

---


**ABSTRACT**  
Background: Accurate survival time estimates aid end-of-life medical decision-making. Objectives: Develop an interpretable survival model for elderly residential aged care residents using advanced machine learning. Setting: A major Australasian residential aged care provider. Participants: Residents aged 65+ admitted for long-term care from July 2017 to August 2023. Sample size: 11,944 residents across 40 facilities. Predictors: Factors include age, gender, health status, co-morbidities, cognitive function, mood, nutrition, mobility, smoking, sleep, skin integrity, and continence. Outcome: Probability of survival post-admission, specifically calibrated for 6-month survival estimates. Statistical Analysis: Tested CoxPH, EN, RR, Lasso, GB, XGB, and RF models in 20 experiments with a 90/10 train/test split. Evaluated accuracy using C-index, Harrell's C-index, dynamic AUROC, IBS, and calibrated ROC. Chose XGB for its performance and calibrated it for 1, 3, 6, and 12-month predictions using Platt scaling. Employed SHAP values to analyze predictor impacts. Results: GB, XGB, and RF models showed the highest C-Index values (0.714, 0.712, 0.712). The optimal XGB model demonstrated a 6-month survival prediction AUROC of 0.746 (95% CI 0.744-0.749). Key mortality predictors include age, male gender, mobility, health status, pressure ulcer risk, and appetite. Conclusions: The study successfully applies machine learning to create a survival model for aged care, aligning with clinical insights on mortality risk factors and enhancing model interpretability and clinical utility through explainable AI.

{{</citation>}}


### (32/119) Sample Efficient Reinforcement Learning from Human Feedback via Active Exploration (Viraj Mehta et al., 2023)

{{<citation>}}

Viraj Mehta, Vikramjeet Das, Ojash Neopane, Yijia Dai, Ilija Bogunovic, Jeff Schneider, Willie Neiswanger. (2023)  
**Sample Efficient Reinforcement Learning from Human Feedback via Active Exploration**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.00267v1)  

---


**ABSTRACT**  
Preference-based feedback is important for many applications in reinforcement learning where direct evaluation of a reward function is not feasible. A notable recent example arises in reinforcement learning from human feedback (RLHF) on large language models. For many applications of RLHF, the cost of acquiring the human feedback can be substantial. In this work, we take advantage of the fact that one can often choose contexts at which to obtain human feedback in order to most efficiently identify a good policy, and formalize this as an offline contextual dueling bandit problem. We give an upper-confidence-bound style algorithm for this problem and prove a polynomial worst-case regret bound. We then provide empirical confirmation in a synthetic setting that our approach outperforms existing methods. After, we extend the setting and methodology for practical use in RLHF training of large language models. Here, our method is able to reach better performance with fewer samples of human preferences than multiple baselines on three real-world datasets.

{{</citation>}}


## cs.NE (1)



### (33/119) Biased Random-Key Genetic Algorithms: A Review (Mariana A. Londe et al., 2023)

{{<citation>}}

Mariana A. Londe, Luciana S. Pessoa, Carlos E. Andrade, Mauricio G. C. Resende. (2023)  
**Biased Random-Key Genetic Algorithms: A Review**  

---
Primary Category: cs.NE  
Categories: 90, 68, F-2-2; G-2-3, cs-NE, cs.NE  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.00961v1)  

---


**ABSTRACT**  
This paper is a comprehensive literature review of Biased Random-Key Genetic Algorithms (BRKGA). BRKGA is a metaheuristic that employs random-key-based chromosomes with biased, uniform, and elitist mating strategies in a genetic algorithm framework. The review encompasses over 150 papers with a wide range of applications, including classical combinatorial optimization problems, real-world industrial use cases, and non-orthodox applications such as neural network hyperparameter tuning in machine learning. Scheduling is by far the most prevalent application area in this review, followed by network design and location problems. The most frequent hybridization method employed is local search, and new features aim to increase population diversity. Overall, this survey provides a comprehensive overview of the BRKGA metaheuristic and its applications and highlights important areas for future research.

{{</citation>}}


## cs.CL (22)



### (34/119) The Cost of Compression: Investigating the Impact of Compression on Parametric Knowledge in Language Models (Satya Sai Srinath Namburi et al., 2023)

{{<citation>}}

Satya Sai Srinath Namburi, Makesh Sreedhar, Srinath Srinivasan, Frederic Sala. (2023)  
**The Cost of Compression: Investigating the Impact of Compression on Parametric Knowledge in Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00960v1)  

---


**ABSTRACT**  
Compressing large language models (LLMs), often consisting of billions of parameters, provides faster inference, smaller memory footprints, and enables local deployment. Two standard compression techniques are pruning and quantization, with the former eliminating redundant connections in model layers and the latter representing model parameters with fewer bits. The key tradeoff is between the degree of compression and the impact on the quality of the compressed model. Existing research on LLM compression primarily focuses on performance in terms of general metrics like perplexity or downstream task accuracy. More fine-grained metrics, such as those measuring parametric knowledge, remain significantly underexplored. To help bridge this gap, we present a comprehensive analysis across multiple model families (ENCODER, ENCODER-DECODER, and DECODER) using the LAMA and LM-HARNESS benchmarks in order to systematically quantify the effect of commonly employed compression techniques on model performance. A particular focus is on tradeoffs involving parametric knowledge, with the goal of providing practitioners with practical insights to help make informed decisions on compression. We release our codebase1 to enable further research.

{{</citation>}}


### (35/119) Hyperparameter Optimization for Large Language Model Instruction-Tuning (Christophe Tribes et al., 2023)

{{<citation>}}

Christophe Tribes, Sacha Benarroch-Lelong, Peng Lu, Ivan Kobyzev. (2023)  
**Hyperparameter Optimization for Large Language Model Instruction-Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, math-OC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00949v1)  

---


**ABSTRACT**  
The fine-tuning of Large Language Models (LLMs) has enabled them to recently achieve milestones in natural language processing applications. The emergence of ever larger LLMs has paved the way for more efficient fine-tuning methods. Among these, the Low-Rank Adaptation (LoRA) method keeps most of the weights of the pre-trained LLM frozen while introducing a low-rank decomposition of the weight matrix, enabling the tuning of only a very small proportion of the network. The performance on downstream tasks of models fine-tuned with LoRA heavily relies on a set of hyperparameters including the rank of the decomposition. In this work, we investigate the choice of these hyperparameters through two main blackbox optimization (BBO) techniques. We examine the whole pipeline of performing fine-tuning and validation on a pre-trained LLM as a blackbox and efficiently explore the space of hyperparameters with the \nomad algorithm, achieving a boost in performance and human alignment of the tuned model.

{{</citation>}}


### (36/119) Quick Back-Translation for Unsupervised Machine Translation (Benjamin Brimacombe et al., 2023)

{{<citation>}}

Benjamin Brimacombe, Jiawei Zhou. (2023)  
**Quick Back-Translation for Unsupervised Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-PL, cs.CL  
Keywords: Machine Translation, Transformer  
[Paper Link](http://arxiv.org/abs/2312.00912v1)  

---


**ABSTRACT**  
The field of unsupervised machine translation has seen significant advancement from the marriage of the Transformer and the back-translation algorithm. The Transformer is a powerful generative model, and back-translation leverages Transformer's high-quality translations for iterative self-improvement. However, the Transformer is encumbered by the run-time of autoregressive inference during back-translation, and back-translation is limited by a lack of synthetic data efficiency. We propose a two-for-one improvement to Transformer back-translation: Quick Back-Translation (QBT). QBT re-purposes the encoder as a generative model, and uses encoder-generated sequences to train the decoder in conjunction with the original autoregressive back-translation step, improving data throughput and utilization. Experiments on various WMT benchmarks demonstrate that a relatively small number of refining steps of QBT improve current unsupervised machine translation models, and that QBT dramatically outperforms standard back-translation only method in terms of training efficiency for comparable translation qualities.

{{</citation>}}


### (37/119) Context Retrieval via Normalized Contextual Latent Interaction for Conversational Agent (Junfeng Liu et al., 2023)

{{<citation>}}

Junfeng Liu, Zhuocheng Mei, Kewen Peng, Ranga Raju Vatsavai. (2023)  
**Context Retrieval via Normalized Contextual Latent Interaction for Conversational Agent**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00774v1)  

---


**ABSTRACT**  
Conversational agents leveraging AI, particularly deep learning, are emerging in both academic research and real-world applications. However, these applications still face challenges, including disrespecting knowledge and facts, not personalizing to user preferences, and enormous demand for computational resources during training and inference. Recent research efforts have been focused on addressing these challenges from various aspects, including supplementing various types of auxiliary information to the conversational agents. However, existing methods are still not able to effectively and efficiently exploit relevant information from these auxiliary supplements to further unleash the power of the conversational agents and the language models they use. In this paper, we present a novel method, PK-NCLI, that is able to accurately and efficiently identify relevant auxiliary information to improve the quality of conversational responses by learning the relevance among persona, chat history, and knowledge background through low-level normalized contextual latent interaction. Our experimental results indicate that PK-NCLI outperforms the state-of-the-art method, PK-FoCus, by 47.80%/30.61%/24.14% in terms of perplexity, knowledge grounding, and training efficiency, respectively, and maintained the same level of persona grounding performance. We also provide a detailed analysis of how different factors, including language model choices and trade-offs on training weights, would affect the performance of PK-NCLI.

{{</citation>}}


### (38/119) Mitigating Over-smoothing in Transformers via Regularized Nonlocal Functionals (Tam Nguyen et al., 2023)

{{<citation>}}

Tam Nguyen, Tan M. Nguyen, Richard G. Baraniuk. (2023)  
**Mitigating Over-smoothing in Transformers via Regularized Nonlocal Functionals**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.00751v1)  

---


**ABSTRACT**  
Transformers have achieved remarkable success in a wide range of natural language processing and computer vision applications. However, the representation capacity of a deep transformer model is degraded due to the over-smoothing issue in which the token representations become identical when the model's depth grows. In this work, we show that self-attention layers in transformers minimize a functional which promotes smoothness, thereby causing token uniformity. We then propose a novel regularizer that penalizes the norm of the difference between the smooth output tokens from self-attention and the input tokens to preserve the fidelity of the tokens. Minimizing the resulting regularized energy functional, we derive the Neural Transformer with a Regularized Nonlocal Functional (NeuTRENO), a novel class of transformer models that can mitigate the over-smoothing issue. We empirically demonstrate the advantages of NeuTRENO over the baseline transformers and state-of-the-art methods in reducing the over-smoothing of token representations on various practical tasks, including object classification, image segmentation, and language modeling.

{{</citation>}}


### (39/119) SeaLLMs -- Large Language Models for Southeast Asia (Xuan-Phi Nguyen et al., 2023)

{{<citation>}}

Xuan-Phi Nguyen, Wenxuan Zhang, Xin Li, Mahani Aljunied, Qingyu Tan, Liying Cheng, Guanzheng Chen, Yue Deng, Sen Yang, Chaoqun Liu, Hang Zhang, Lidong Bing. (2023)  
**SeaLLMs -- Large Language Models for Southeast Asia**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2312.00738v1)  

---


**ABSTRACT**  
Despite the remarkable achievements of large language models (LLMs) in various tasks, there remains a linguistic bias that favors high-resource languages, such as English, often at the expense of low-resource and regional languages. To address this imbalance, we introduce SeaLLMs, an innovative series of language models that specifically focuses on Southeast Asian (SEA) languages. SeaLLMs are built upon the Llama-2 model and further advanced through continued pre-training with an extended vocabulary, specialized instruction and alignment tuning to better capture the intricacies of regional languages. This allows them to respect and reflect local cultural norms, customs, stylistic preferences, and legal considerations. Our comprehensive evaluation demonstrates that SeaLLM-13b models exhibit superior performance across a wide spectrum of linguistic tasks and assistant-style instruction-following capabilities relative to comparable open-source models. Moreover, they outperform ChatGPT-3.5 in non-Latin languages, such as Thai, Khmer, Lao, and Burmese, by large margins while remaining lightweight and cost-effective to operate.

{{</citation>}}


### (40/119) Towards Transparency in Coreference Resolution: A Quantum-Inspired Approach (Hadi Wazni et al., 2023)

{{<citation>}}

Hadi Wazni, Mehrnoosh Sadrzadeh. (2023)  
**Towards Transparency in Coreference Resolution: A Quantum-Inspired Approach**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-LO, cs.CL  
Keywords: BERT, NLP, Natural Language Processing, Quantum Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.00688v1)  

---


**ABSTRACT**  
Guided by grammatical structure, words compose to form sentences, and guided by discourse structure, sentences compose to form dialogues and documents. The compositional aspect of sentence and discourse units is often overlooked by machine learning algorithms. A recent initiative called Quantum Natural Language Processing (QNLP) learns word meanings as points in a Hilbert space and acts on them via a translation of grammatical structure into Parametrised Quantum Circuits (PQCs). Previous work extended the QNLP translation to discourse structure using points in a closure of Hilbert spaces. In this paper, we evaluate this translation on a Winograd-style pronoun resolution task. We train a Variational Quantum Classifier (VQC) for binary classification and implement an end-to-end pronoun resolution system. The simulations executed on IBMQ software converged with an F1 score of 87.20%. The model outperformed two out of three classical coreference resolution systems and neared state-of-the-art SpanBERT. A mixed quantum-classical model yet improved these results with an F1 score increase of around 6%.

{{</citation>}}


### (41/119) Contextualized word senses: from attention to compositionality (Pablo Gamallo, 2023)

{{<citation>}}

Pablo Gamallo. (2023)  
**Contextualized word senses: from attention to compositionality**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.00680v1)  

---


**ABSTRACT**  
The neural architectures of language models are becoming increasingly complex, especially that of Transformers, based on the attention mechanism. Although their application to numerous natural language processing tasks has proven to be very fruitful, they continue to be models with little or no interpretability and explainability. One of the tasks for which they are best suited is the encoding of the contextual sense of words using contextualized embeddings. In this paper we propose a transparent, interpretable, and linguistically motivated strategy for encoding the contextual sense of words by modeling semantic compositionality. Particular attention is given to dependency relations and semantic notions such as selection preferences and paradigmatic classes. A partial implementation of the proposed model is carried out and compared with Transformer-based architectures for a given semantic task, namely the similarity calculation of word senses in context. The results obtained show that it is possible to be competitive with linguistically motivated models instead of using the black boxes underlying complex neural architectures.

{{</citation>}}


### (42/119) The Efficiency Spectrum of Large Language Models: An Algorithmic Survey (Tianyu Ding et al., 2023)

{{<citation>}}

Tianyu Ding, Tianyi Chen, Haidong Zhu, Jiachen Jiang, Yiqi Zhong, Jinxin Zhou, Guangzhi Wang, Zhihui Zhu, Ilya Zharkov, Luming Liang. (2023)  
**The Efficiency Spectrum of Large Language Models: An Algorithmic Survey**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00678v1)  

---


**ABSTRACT**  
The rapid growth of Large Language Models (LLMs) has been a driving force in transforming various domains, reshaping the artificial general intelligence landscape. However, the increasing computational and memory demands of these models present substantial challenges, hindering both academic research and practical applications. To address these issues, a wide array of methods, including both algorithmic and hardware solutions, have been developed to enhance the efficiency of LLMs. This survey delivers a comprehensive review of algorithmic advancements aimed at improving LLM efficiency. Unlike other surveys that typically focus on specific areas such as training or model compression, this paper examines the multi-faceted dimensions of efficiency essential for the end-to-end algorithmic development of LLMs. Specifically, it covers various topics related to efficiency, including scaling laws, data utilization, architectural innovations, training and tuning strategies, and inference techniques. This paper aims to serve as a valuable resource for researchers and practitioners, laying the groundwork for future innovations in this critical research area. Our repository of relevant references is maintained at url{https://github.com/tding1/Efficient-LLM-Survey}.

{{</citation>}}


### (43/119) The Ethics of Automating Legal Actors (Josef Valvoda et al., 2023)

{{<citation>}}

Josef Valvoda, Alec Thompson, Ryan Cotterell, Simone Teufel. (2023)  
**The Ethics of Automating Legal Actors**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Legal, NLP  
[Paper Link](http://arxiv.org/abs/2312.00584v1)  

---


**ABSTRACT**  
The introduction of large public legal datasets has brought about a renaissance in legal NLP. Many of these datasets are comprised of legal judgements - the product of judges deciding cases. This fact, together with the way machine learning works, means that several legal NLP models are models of judges. While some have argued for the automation of judges, in this position piece, we argue that automating the role of the judge raises difficult ethical challenges, in particular for common law legal systems. Our argument follows from the social role of the judge in actively shaping the law, rather than merely applying it. Since current NLP models come nowhere close to having the facilities necessary for this task, they should not be used to automate judges. Furthermore, even in the case the models could achieve human-level capabilities, there would still be remaining ethical concerns inherent in the automation of the legal process.

{{</citation>}}


### (44/119) Explanatory Argument Extraction of Correct Answers in Resident Medical Exams (Iakes Goenaga et al., 2023)

{{<citation>}}

Iakes Goenaga, Aitziber Atutxa, Koldo Gojenola, Maite Oronoz, Rodrigo Agerri. (2023)  
**Explanatory Argument Extraction of Correct Answers in Resident Medical Exams**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2312.00567v1)  

---


**ABSTRACT**  
Developing the required technology to assist medical experts in their everyday activities is currently a hot topic in the Artificial Intelligence research field. Thus, a number of large language models (LLMs) and automated benchmarks have recently been proposed with the aim of facilitating information extraction in Evidence-Based Medicine (EBM) using natural language as a tool for mediating in human-AI interaction. The most representative benchmarks are limited to either multiple-choice or long-form answers and are available only in English. In order to address these shortcomings, in this paper we present a new dataset which, unlike previous work: (i) includes not only explanatory arguments for the correct answer, but also arguments to reason why the incorrect answers are not correct; (ii) the explanations are written originally by medical doctors to answer questions from the Spanish Residency Medical Exams. Furthermore, this new benchmark allows us to setup a novel extractive task which consists of identifying the explanation of the correct answer written by medical doctors. An additional benefit of our setting is that we can leverage the extractive QA paradigm to automatically evaluate performance of LLMs without resorting to costly manual evaluation by medical experts. Comprehensive experimentation with language models for Spanish shows that sometimes multilingual models fare better than monolingual ones, even outperforming models which have been adapted to the medical domain. Furthermore, results across the monolingual models are mixed, with supposedly smaller and inferior models performing competitively. In any case, the obtained results show that our novel dataset and approach can be an effective technique to help medical practitioners in identifying relevant evidence-based explanations for medical questions.

{{</citation>}}


### (45/119) Questioning Biases in Case Judgment Summaries: Legal Datasets or Large Language Models? (Aniket Deroy et al., 2023)

{{<citation>}}

Aniket Deroy, Subhankar Maity. (2023)  
**Questioning Biases in Case Judgment Summaries: Legal Datasets or Large Language Models?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias, Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2312.00554v1)  

---


**ABSTRACT**  
The evolution of legal datasets and the advent of large language models (LLMs) have significantly transformed the legal field, particularly in the generation of case judgment summaries. However, a critical concern arises regarding the potential biases embedded within these summaries. This study scrutinizes the biases present in case judgment summaries produced by legal datasets and large language models. The research aims to analyze the impact of biases on legal decision making. By interrogating the accuracy, fairness, and implications of biases in these summaries, this study contributes to a better understanding of the role of technology in legal contexts and the implications for justice systems worldwide. In this study, we investigate biases wrt Gender-related keywords, Race-related keywords, Keywords related to crime against women, Country names and religious keywords. The study shows interesting evidences of biases in the outputs generated by the large language models and pre-trained abstractive summarization models. The reasoning behind these biases needs further studies.

{{</citation>}}


### (46/119) Improving Unsupervised Relation Extraction by Augmenting Diverse Sentence Pairs (Qing Wang et al., 2023)

{{<citation>}}

Qing Wang, Kang Zhou, Qiao Qiao, Yuepei Li, Qi Li. (2023)  
**Improving Unsupervised Relation Extraction by Augmenting Diverse Sentence Pairs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Relation Extraction  
[Paper Link](http://arxiv.org/abs/2312.00552v1)  

---


**ABSTRACT**  
Unsupervised relation extraction (URE) aims to extract relations between named entities from raw text without requiring manual annotations or pre-existing knowledge bases. In recent studies of URE, researchers put a notable emphasis on contrastive learning strategies for acquiring relation representations. However, these studies often overlook two important aspects: the inclusion of diverse positive pairs for contrastive learning and the exploration of appropriate loss functions. In this paper, we propose AugURE with both within-sentence pairs augmentation and augmentation through cross-sentence pairs extraction to increase the diversity of positive pairs and strengthen the discriminative power of contrastive learning. We also identify the limitation of noise-contrastive estimation (NCE) loss for relation representation learning and propose to apply margin loss for sentence pairs. Experiments on NYT-FB and TACRED datasets demonstrate that the proposed relation representation learning and a simple K-Means clustering achieves state-of-the-art performance.

{{</citation>}}


### (47/119) SurreyAI 2023 Submission for the Quality Estimation Shared Task (Archchana Sindhujan et al., 2023)

{{<citation>}}

Archchana Sindhujan, Diptesh Kanojia, Constantin Orasan, Tharindu Ranasinghe. (2023)  
**SurreyAI 2023 Submission for the Quality Estimation Shared Task**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00525v1)  

---


**ABSTRACT**  
Quality Estimation (QE) systems are important in situations where it is necessary to assess the quality of translations, but there is no reference available. This paper describes the approach adopted by the SurreyAI team for addressing the Sentence-Level Direct Assessment shared task in WMT23. The proposed approach builds upon the TransQuest framework, exploring various autoencoder pre-trained language models within the MonoTransQuest architecture using single and ensemble settings. The autoencoder pre-trained language models employed in the proposed systems are XLMV, InfoXLM-large, and XLMR-large. The evaluation utilizes Spearman and Pearson correlation coefficients, assessing the relationship between machine-predicted quality scores and human judgments for 5 language pairs (English-Gujarati, English-Hindi, English-Marathi, English-Tamil and English-Telugu). The MonoTQ-InfoXLM-large approach emerges as a robust strategy, surpassing all other individual models proposed in this study by significantly improving over the baseline for the majority of the language pairs.

{{</citation>}}


### (48/119) RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback (Tianyu Yu et al., 2023)

{{<citation>}}

Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, Tat-Seng Chua. (2023)  
**RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.00849v1)  

---


**ABSTRACT**  
Multimodal Large Language Models (MLLMs) have recently demonstrated impressive capabilities in multimodal understanding, reasoning, and interaction. However, existing MLLMs prevalently suffer from serious hallucination problems, generating text that is not factually grounded in associated images. The problem makes existing MLLMs untrustworthy and thus impractical in real-world (especially high-stakes) applications. To address the challenge, we present RLHF-V, which enhances MLLM trustworthiness via behavior alignment from fine-grained correctional human feedback. Specifically, RLHF-V collects human preference in the form of segment-level corrections on hallucinations, and performs dense direct preference optimization over the human feedback. Comprehensive experiments on five benchmarks in both automatic and human evaluation show that, RLHF-V can enable substantially more trustworthy MLLM behaviors with promising data and computation efficiency. Remarkably, using 1.4k annotated data samples, RLHF-V significantly reduces the hallucination rate of the base MLLM by 34.8%, outperforming the concurrent LLaVA-RLHF trained on 10k annotated data. The final model achieves state-of-the-art performance in trustworthiness among open-source MLLMs, and shows better robustness than GPT-4V in preventing hallucinations aroused from over-generalization. We open-source our code, model, and data at https://github.com/RLHF-V/RLHF-V.

{{</citation>}}


### (49/119) Summarization-based Data Augmentation for Document Classification (Yueguan Wang et al., 2023)

{{<citation>}}

Yueguan Wang, Naoki Yoshinaga. (2023)  
**Summarization-based Data Augmentation for Document Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, Summarization  
[Paper Link](http://arxiv.org/abs/2312.00513v1)  

---


**ABSTRACT**  
Despite the prevalence of pretrained language models in natural language understanding tasks, understanding lengthy text such as document is still challenging due to the data sparseness problem. Inspired by that humans develop their ability of understanding lengthy text from reading shorter text, we propose a simple yet effective summarization-based data augmentation, SUMMaug, for document classification. We first obtain easy-to-learn examples for the target document classification task by summarizing the input of the original training examples, while optionally merging the original labels to conform to the summarized input. We then use the generated pseudo examples to perform curriculum learning. Experimental results on two datasets confirmed the advantage of our method compared to existing baseline methods in terms of robustness and accuracy. We release our code and data at https://github.com/etsurin/summaug.

{{</citation>}}


### (50/119) Japanese Tort-case Dataset for Rationale-supported Legal Judgment Prediction (Hiroaki Yamada et al., 2023)

{{<citation>}}

Hiroaki Yamada, Takenobu Tokunaga, Ryutaro Ohara, Akira Tokutsu, Keisuke Takeshita, Mihoko Sumida. (2023)  
**Japanese Tort-case Dataset for Rationale-supported Legal Judgment Prediction**  

---
Primary Category: cs.CL  
Categories: 68T50, cs-AI, cs-CL, cs.CL  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2312.00480v1)  

---


**ABSTRACT**  
This paper presents the first dataset for Japanese Legal Judgment Prediction (LJP), the Japanese Tort-case Dataset (JTD), which features two tasks: tort prediction and its rationale extraction. The rationale extraction task identifies the court's accepting arguments from alleged arguments by plaintiffs and defendants, which is a novel task in the field. JTD is constructed based on annotated 3,477 Japanese Civil Code judgments by 41 legal experts, resulting in 7,978 instances with 59,697 of their alleged arguments from the involved parties. Our baseline experiments show the feasibility of the proposed two tasks, and our error analysis by legal experts identifies sources of errors and suggests future directions of the LJP research.

{{</citation>}}


### (51/119) CoLLiE: Collaborative Training of Large Language Models in an Efficient Way (Kai Lv et al., 2023)

{{<citation>}}

Kai Lv, Shuo Zhang, Tianle Gu, Shuhao Xing, Jiawei Hong, Keyu Chen, Xiaoran Liu, Yuqing Yang, Honglin Guo, Tengxiao Liu, Yu Sun, Qipeng Guo, Hang Yan, Xipeng Qiu. (2023)  
**CoLLiE: Collaborative Training of Large Language Models in an Efficient Way**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00407v1)  

---


**ABSTRACT**  
Large language models (LLMs) are increasingly pivotal in a wide range of natural language processing tasks. Access to pre-trained models, courtesy of the open-source community, has made it possible to adapt these models to specific applications for enhanced performance. However, the substantial resources required for training these models necessitate efficient solutions. This paper introduces CoLLiE, an efficient library that facilitates collaborative training of large language models using 3D parallelism, parameter-efficient fine-tuning (PEFT) methods, and optimizers such as Lion, Adan, Sophia, LOMO and AdaLomo. With its modular design and comprehensive functionality, CoLLiE offers a balanced blend of efficiency, ease of use, and customization. CoLLiE has proven superior training efficiency in comparison with prevalent solutions in pre-training and fine-tuning scenarios. Furthermore, we provide an empirical evaluation of the correlation between model size and GPU memory consumption under different optimization methods, as well as an analysis of the throughput. Lastly, we carry out a comprehensive comparison of various optimizers and PEFT methods within the instruction-tuning context. CoLLiE is available at https://github.com/OpenLMLab/collie.

{{</citation>}}


### (52/119) On Exploring the Reasoning Capability of Large Language Models with Knowledge Graphs (Pei-Chi Lo et al., 2023)

{{<citation>}}

Pei-Chi Lo, Yi-Hang Tsai, Ee-Peng Lim, San-Yih Hwang. (2023)  
**On Exploring the Reasoning Capability of Large Language Models with Knowledge Graphs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.00353v1)  

---


**ABSTRACT**  
This paper examines the capacity of LLMs to reason with knowledge graphs using their internal knowledge graph, i.e., the knowledge graph they learned during pre-training. Two research questions are formulated to investigate the accuracy of LLMs in recalling information from pre-training knowledge graphs and their ability to infer knowledge graph relations from context. To address these questions, we employ LLMs to perform four distinct knowledge graph reasoning tasks. Furthermore, we identify two types of hallucinations that may occur during knowledge reasoning with LLMs: content and ontology hallucination. Our experimental results demonstrate that LLMs can successfully tackle both simple and complex knowledge graph reasoning tasks from their own memory, as well as infer from input context.

{{</citation>}}


### (53/119) The Case for Scalable, Data-Driven Theory: A Paradigm for Scientific Progress in NLP (Julian Michael, 2023)

{{<citation>}}

Julian Michael. (2023)  
**The Case for Scalable, Data-Driven Theory: A Paradigm for Scientific Progress in NLP**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: AI, NLP, QA  
[Paper Link](http://arxiv.org/abs/2312.00349v1)  

---


**ABSTRACT**  
I propose a paradigm for scientific progress in NLP centered around developing scalable, data-driven theories of linguistic structure. The idea is to collect data in tightly scoped, carefully defined ways which allow for exhaustive annotation of behavioral phenomena of interest, and then use machine learning to construct explanatory theories of these phenomena which can form building blocks for intelligible AI systems. After laying some conceptual groundwork, I describe several investigations into data-driven theories of shallow semantic structure using Question-Answer driven Semantic Role Labeling (QA-SRL), a schema for annotating verbal predicate-argument relations using highly constrained question-answer pairs. While this only scratches the surface of the complex language behaviors of interest in AI, I outline principles for data collection and theoretical modeling which can inform future scientific progress. This note summarizes and draws heavily on my PhD thesis.

{{</citation>}}


### (54/119) PsyAttention: Psychological Attention Model for Personality Detection (Baohua Zhang et al., 2023)

{{<citation>}}

Baohua Zhang, Yongyi Huang, Wenyao Cui, Huaping Zhang, Jianyun Shang. (2023)  
**PsyAttention: Psychological Attention Model for Personality Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Personality Detection  
[Paper Link](http://arxiv.org/abs/2312.00293v1)  

---


**ABSTRACT**  
Work on personality detection has tended to incorporate psychological features from different personality models, such as BigFive and MBTI. There are more than 900 psychological features, each of which is helpful for personality detection. However, when used in combination, the application of different calculation standards among these features may result in interference between features calculated using distinct systems, thereby introducing noise and reducing performance. This paper adapts different psychological models in the proposed PsyAttention for personality detection, which can effectively encode psychological features, reducing their number by 85%. In experiments on the BigFive and MBTI models, PysAttention achieved average accuracy of 65.66% and 86.30%, respectively, outperforming state-of-the-art methods, indicating that it is effective at encoding psychological features.

{{</citation>}}


### (55/119) SEPSIS: I Can Catch Your Lies -- A New Paradigm for Deception Detection (Anku Rani et al., 2023)

{{<citation>}}

Anku Rani, Dwip Dalal, Shreya Gautam, Pankaj Gupta, Vinija Jain, Aman Chadha, Amit Sheth, Amitava Das. (2023)  
**SEPSIS: I Can Catch Your Lies -- A New Paradigm for Deception Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Twitter  
[Paper Link](http://arxiv.org/abs/2312.00292v1)  

---


**ABSTRACT**  
Deception is the intentional practice of twisting information. It is a nuanced societal practice deeply intertwined with human societal evolution, characterized by a multitude of facets. This research explores the problem of deception through the lens of psychology, employing a framework that categorizes deception into three forms: lies of omission, lies of commission, and lies of influence. The primary focus of this study is specifically on investigating only lies of omission. We propose a novel framework for deception detection leveraging NLP techniques. We curated an annotated dataset of 876,784 samples by amalgamating a popular large-scale fake news dataset and scraped news headlines from the Twitter handle of Times of India, a well-known Indian news media house. Each sample has been labeled with four layers, namely: (i) the type of omission (speculation, bias, distortion, sounds factual, and opinion), (ii) colors of lies(black, white, etc), and (iii) the intention of such lies (to influence, etc) (iv) topic of lies (political, educational, religious, etc). We present a novel multi-task learning pipeline that leverages the dataless merging of fine-tuned language models to address the deception detection task mentioned earlier. Our proposed model achieved an F1 score of 0.87, demonstrating strong performance across all layers including the type, color, intent, and topic aspects of deceptive content. Finally, our research explores the relationship between lies of omission and propaganda techniques. To accomplish this, we conducted an in-depth analysis, uncovering compelling findings. For instance, our analysis revealed a significant correlation between loaded language and opinion, shedding light on their interconnectedness. To encourage further research in this field, we will be making the models and dataset available with the MIT License, making it favorable for open-source research.

{{</citation>}}


## cs.NI (1)



### (56/119) A Comprehensive Real-World Evaluation of 5G Improvements over 4G in Low- and Mid-Bands (Muhammad Iqbal Rochman et al., 2023)

{{<citation>}}

Muhammad Iqbal Rochman, Wei Ye, Zhi-Li Zhang, Monisha Ghosh. (2023)  
**A Comprehensive Real-World Evaluation of 5G Improvements over 4G in Low- and Mid-Bands**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.00957v1)  

---


**ABSTRACT**  
As discussions around 6G begin, it is important to carefully quantify the spectral efficiency gains actually realized by deployed 5G networks as compared to 4G through various enhancements such as higher modulation, beamforming, and MIMO. This will inform the design of future cellular systems, especially in the mid-bands, which provide a good balance between bandwidth and propagation. Similar to 4G, 5G also utilizes low-band (<1 GHz) and mid-band spectrum (1 to 6 GHz), and hence comparing the performance of 4G and 5G in these bands will provide insights into how further improvements can be attained. In this work, we address a crucial question: is the performance boost in 5G compared to 4G primarily a result of increased bandwidth, or do the other enhancements play significant roles, and if so, under what circumstances? Hence, we conduct city-wide measurements of 4G and 5G cellular networks deployed in low- and mid-bands in Chicago and Minneapolis, and carefully quantify the contributions of different aspects of 5G advancements to its improved throughput performance. Our analyses show that (i) compared to 4G, the throughput improvement in 5G today is mainly influenced by the wider channel bandwidth, both from single channels and channel aggregation, (ii) in addition to wider channels, improved 5G throughput requires better signal conditions, which can be delivered by denser deployment and/or use of beamforming in mid-bands, (iii) the channel rank in real-world environments rarely supports the full 4 layers of 4x4 MIMO and (iv) advanced features such as MU-MIMO and higher order modulation such as 1024-QAM have yet to be widely deployed. These observations and conclusions lead one to consider designing the next generation of cellular systems to have wider channels, perhaps with improved channel aggregation, dense deployment with more beams.

{{</citation>}}


## cs.CV (31)



### (57/119) Improve Supervised Representation Learning with Masked Image Modeling (Kaifeng Chen et al., 2023)

{{<citation>}}

Kaifeng Chen, Daniel Salz, Huiwen Chang, Kihyuk Sohn, Dilip Krishnan, Mojtaba Seyedhosseini. (2023)  
**Improve Supervised Representation Learning with Masked Image Modeling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.00950v1)  

---


**ABSTRACT**  
Training visual embeddings with labeled data supervision has been the de facto setup for representation learning in computer vision. Inspired by recent success of adopting masked image modeling (MIM) in self-supervised representation learning, we propose a simple yet effective setup that can easily integrate MIM into existing supervised training paradigms. In our design, in addition to the original classification task applied to a vision transformer image encoder, we add a shallow transformer-based decoder on top of the encoder and introduce an MIM task which tries to reconstruct image tokens based on masked image inputs. We show with minimal change in architecture and no overhead in inference that this setup is able to improve the quality of the learned representations for downstream tasks such as classification, image retrieval, and semantic segmentation. We conduct a comprehensive study and evaluation of our setup on public benchmarks. On ImageNet-1k, our ViT-B/14 model achieves 81.72% validation accuracy, 2.01% higher than the baseline model. On K-Nearest-Neighbor image retrieval evaluation with ImageNet-1k, the same model outperforms the baseline by 1.32%. We also show that this setup can be easily scaled to larger models and datasets. Code and checkpoints will be released.

{{</citation>}}


### (58/119) Zero-Shot Video Question Answering with Procedural Programs (Rohan Choudhury et al., 2023)

{{<citation>}}

Rohan Choudhury, Koichiro Niinuma, Kris M. Kitani, László A. Jeni. (2023)  
**Zero-Shot Video Question Answering with Procedural Programs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Question Answering, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.00937v1)  

---


**ABSTRACT**  
We propose to answer zero-shot questions about videos by generating short procedural programs that derive a final answer from solving a sequence of visual subtasks. We present Procedural Video Querying (ProViQ), which uses a large language model to generate such programs from an input question and an API of visual modules in the prompt, then executes them to obtain the output. Recent similar procedural approaches have proven successful for image question answering, but videos remain challenging: we provide ProViQ with modules intended for video understanding, allowing it to generalize to a wide variety of videos. This code generation framework additionally enables ProViQ to perform other video tasks in addition to question answering, such as multi-object tracking or basic video editing. ProViQ achieves state-of-the-art results on a diverse range of benchmarks, with improvements of up to 25% on short, long, open-ended, and multimodal video question-answering datasets. Our project page is at https://rccchoudhury.github.io/proviq2023.

{{</citation>}}


### (59/119) Grounding Everything: Emerging Localization Properties in Vision-Language Transformers (Walid Bousselham et al., 2023)

{{<citation>}}

Walid Bousselham, Felix Petersen, Vittorio Ferrari, Hilde Kuehne. (2023)  
**Grounding Everything: Emerging Localization Properties in Vision-Language Transformers**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.00878v1)  

---


**ABSTRACT**  
Vision-language foundation models have shown remarkable performance in various zero-shot settings such as image retrieval, classification, or captioning. But so far, those models seem to fall behind when it comes to zero-shot localization of referential expressions and objects in images. As a result, they need to be fine-tuned for this task. In this paper, we show that pretrained vision-language (VL) models allow for zero-shot open-vocabulary object localization without any fine-tuning. To leverage those capabilities, we propose a Grounding Everything Module (GEM) that generalizes the idea of value-value attention introduced by CLIPSurgery to a self-self attention path. We show that the concept of self-self attention corresponds to clustering, thus enforcing groups of tokens arising from the same object to be similar while preserving the alignment with the language space. To further guide the group formation, we propose a set of regularizations that allows the model to finally generalize across datasets and backbones. We evaluate the proposed GEM framework on various benchmark tasks and datasets for semantic segmentation. It shows that GEM not only outperforms other training-free open-vocabulary localization methods, but also achieves state-of-the-art results on the recently proposed OpenImagesV7 large-scale segmentation benchmark.

{{</citation>}}


### (60/119) Making Large Multimodal Models Understand Arbitrary Visual Prompts (Mu Cai et al., 2023)

{{<citation>}}

Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P. Meyer, Yuning Chai, Dennis Park, Yong Jae Lee. (2023)  
**Making Large Multimodal Models Understand Arbitrary Visual Prompts**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.00784v1)  

---


**ABSTRACT**  
While existing large vision-language multimodal models focus on whole image understanding, there is a prominent gap in achieving region-specific comprehension. Current approaches that use textual coordinates or spatial encodings often fail to provide a user-friendly interface for visual prompting. To address this challenge, we introduce a novel multimodal model capable of decoding arbitrary visual prompts. This allows users to intuitively mark images and interact with the model using natural cues like a "red bounding box" or "pointed arrow". Our simple design directly overlays visual markers onto the RGB image, eliminating the need for complex region encodings, yet achieves state-of-the-art performance on region-understanding tasks like Visual7W, PointQA, and Visual Commonsense Reasoning benchmark. Furthermore, we present ViP-Bench, a comprehensive benchmark to assess the capability of models in understanding visual prompts across multiple dimensions, enabling future research in this domain. Code, data, and model are publicly available.

{{</citation>}}


### (61/119) EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything (Yunyang Xiong et al., 2023)

{{<citation>}}

Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu, Xiaoliang Dai, Dilin Wang, Fei Sun, Forrest Iandola, Raghuraman Krishnamoorthi, Vikas Chandra. (2023)  
**EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.00863v1)  

---


**ABSTRACT**  
Segment Anything Model (SAM) has emerged as a powerful tool for numerous vision applications. A key component that drives the impressive performance for zero-shot transfer and high versatility is a super large Transformer model trained on the extensive high-quality SA-1B dataset. While beneficial, the huge computation cost of SAM model has limited its applications to wider real-world applications. To address this limitation, we propose EfficientSAMs, light-weight SAM models that exhibits decent performance with largely reduced complexity. Our idea is based on leveraging masked image pretraining, SAMI, which learns to reconstruct features from SAM image encoder for effective visual representation learning. Further, we take SAMI-pretrained light-weight image encoders and mask decoder to build EfficientSAMs, and finetune the models on SA-1B for segment anything task. We perform evaluations on multiple vision tasks including image classification, object detection, instance segmentation, and semantic object detection, and find that our proposed pretraining method, SAMI, consistently outperforms other masked image pretraining methods. On segment anything task such as zero-shot instance segmentation, our EfficientSAMs with SAMI-pretrained lightweight image encoders perform favorably with a significant gain (e.g., ~4 AP on COCO/LVIS) over other fast SAM models.

{{</citation>}}


### (62/119) DeepCache: Accelerating Diffusion Models for Free (Xinyin Ma et al., 2023)

{{<citation>}}

Xinyin Ma, Gongfan Fang, Xinchao Wang. (2023)  
**DeepCache: Accelerating Diffusion Models for Free**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.00858v1)  

---


**ABSTRACT**  
Diffusion models have recently gained unprecedented attention in the field of image synthesis due to their remarkable generative capabilities. Notwithstanding their prowess, these models often incur substantial computational costs, primarily attributed to the sequential denoising process and cumbersome model size. Traditional methods for compressing diffusion models typically involve extensive retraining, presenting cost and feasibility challenges. In this paper, we introduce DeepCache, a novel training-free paradigm that accelerates diffusion models from the perspective of model architecture. DeepCache capitalizes on the inherent temporal redundancy observed in the sequential denoising steps of diffusion models, which caches and retrieves features across adjacent denoising stages, thereby curtailing redundant computations. Utilizing the property of the U-Net, we reuse the high-level features while updating the low-level features in a very cheap way. This innovative strategy, in turn, enables a speedup factor of 2.3$\times$ for Stable Diffusion v1.5 with only a 0.05 decline in CLIP Score, and 4.1$\times$ for LDM-4-G with a slight decrease of 0.22 in FID on ImageNet. Our experiments also demonstrate DeepCache's superiority over existing pruning and distillation methods that necessitate retraining and its compatibility with current sampling techniques. Furthermore, we find that under the same throughput, DeepCache effectively achieves comparable or even marginally improved results with DDIM or PLMS. The code is available at https://github.com/horseee/DeepCache

{{</citation>}}


### (63/119) PointBeV: A Sparse Approach to BeV Predictions (Loick Chambon et al., 2023)

{{<citation>}}

Loick Chambon, Eloi Zablocki, Mickael Chen, Florent Bartoccioni, Patrick Perez, Matthieu Cord. (2023)  
**PointBeV: A Sparse Approach to BeV Predictions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.00703v1)  

---


**ABSTRACT**  
Bird's-eye View (BeV) representations have emerged as the de-facto shared space in driving applications, offering a unified space for sensor data fusion and supporting various downstream tasks. However, conventional models use grids with fixed resolution and range and face computational inefficiencies due to the uniform allocation of resources across all cells. To address this, we propose PointBeV, a novel sparse BeV segmentation model operating on sparse BeV cells instead of dense grids. This approach offers precise control over memory usage, enabling the use of long temporal contexts and accommodating memory-constrained platforms. PointBeV employs an efficient two-pass strategy for training, enabling focused computation on regions of interest. At inference time, it can be used with various memory/performance trade-offs and flexibly adjusts to new specific use cases. PointBeV achieves state-of-the-art results on the nuScenes dataset for vehicle, pedestrian, and lane segmentation, showcasing superior performance in static and temporal settings despite being trained solely with sparse signals. We will release our code along with two new efficient modules used in the architecture: Sparse Feature Pulling, designed for the effective extraction of features from images to BeV, and Submanifold Attention, which enables efficient temporal modeling. Our code is available at https://github.com/valeoai/PointBeV.

{{</citation>}}


### (64/119) GIFT: Generative Interpretable Fine-Tuning Transformers (Chinmay Savadikar et al., 2023)

{{<citation>}}

Chinmay Savadikar, Xi Song, Tianfu Wu. (2023)  
**GIFT: Generative Interpretable Fine-Tuning Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.00700v1)  

---


**ABSTRACT**  
We present GIFT (Generative Interpretable Fine-tuning Transformers) for fine-tuning pretrained (often large) Transformer models at downstream tasks in a parameter-efficient way with built-in interpretability. Our GIFT is a deep parameter-residual learning method, which addresses two problems in fine-tuning a pretrained Transformer model: Where to apply the parameter-efficient fine-tuning (PEFT) to be extremely lightweight yet sufficiently expressive, and How to learn the PEFT to better exploit the knowledge of the pretrained model in a direct way? For the former, we select the final projection (linear) layer in the multi-head self-attention of a Transformer model, and verify its effectiveness. For the latter, in contrast to the prior art that directly introduce new model parameters (often in low-rank approximation form) to be learned in fine-tuning with downstream data, we propose a method for learning to generate the fine-tuning parameters. Our GIFT is a hyper-Transformer which take as input the pretrained parameters of the projection layer to generate its fine-tuning parameters using a proposed Parameter-to-Cluster Attention (PaCa). The PaCa results in a simple clustering-based forward explainer that plays the role of semantic segmentation in testing. In experiments, our proposed GIFT is tested on the VTAB benchmark and the fine-grained visual classification (FGVC) benchmark. It obtains significantly better performance than the prior art. Our code is available at https://github.com/savadikarc/gift

{{</citation>}}


### (65/119) Open-vocabulary object 6D pose estimation (Jaime Corsetti et al., 2023)

{{<citation>}}

Jaime Corsetti, Davide Boscaini, Changjae Oh, Andrea Cavallaro, Fabio Poiesi. (2023)  
**Open-vocabulary object 6D pose estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00690v1)  

---


**ABSTRACT**  
We introduce the new setting of open-vocabulary object 6D pose estimation, in which a textual prompt is used to specify the object of interest. In contrast to existing approaches, in our setting (i) the object of interest is specified solely through the textual prompt, (ii) no object model (e.g. CAD or video sequence) is required at inference, (iii) the object is imaged from two different viewpoints of two different scenes, and (iv) the object was not observed during the training phase. To operate in this setting, we introduce a novel approach that leverages a Vision-Language Model to segment the object of interest from two distinct scenes and to estimate its relative 6D pose. The key of our approach is a carefully devised strategy to fuse object-level information provided by the prompt with local image features, resulting in a feature space that can generalize to novel concepts. We validate our approach on a new benchmark based on two popular datasets, REAL275 and Toyota-Light, which collectively encompass 39 object instances appearing in four thousand image pairs. The results demonstrate that our approach outperforms both a well-established hand-crafted method and a recent deep learning-based baseline in estimating the relative 6D pose of objects in different scenes. Project website: https://jcorsetti.github.io/oryon-website/.

{{</citation>}}


### (66/119) LightCLIP: Learning Multi-Level Interaction for Lightweight Vision-Language Models (Ying Nie et al., 2023)

{{<citation>}}

Ying Nie, Wei He, Kai Han, Yehui Tang, Tianyu Guo, Fanyi Du, Yunhe Wang. (2023)  
**LightCLIP: Learning Multi-Level Interaction for Lightweight Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00674v1)  

---


**ABSTRACT**  
Vision-language pre-training like CLIP has shown promising performance on various downstream tasks such as zero-shot image classification and image-text retrieval. Most of the existing CLIP-alike works usually adopt relatively large image encoders like ResNet50 and ViT, while the lightweight counterparts are rarely discussed. In this paper, we propose a multi-level interaction paradigm for training lightweight CLIP models. Firstly, to mitigate the problem that some image-text pairs are not strictly one-to-one correspondence, we improve the conventional global instance-level alignment objective by softening the label of negative samples progressively. Secondly, a relaxed bipartite matching based token-level alignment objective is introduced for finer-grained alignment between image patches and textual words. Moreover, based on the observation that the accuracy of CLIP model does not increase correspondingly as the parameters of text encoder increase, an extra objective of masked language modeling (MLM) is leveraged for maximizing the potential of the shortened text encoder. In practice, an auxiliary fusion module injecting unmasked image embedding into masked text embedding at different network stages is proposed for enhancing the MLM. Extensive experiments show that without introducing additional computational cost during inference, the proposed method achieves a higher performance on multiple downstream tasks.

{{</citation>}}


### (67/119) SPOT: Self-Training with Patch-Order Permutation for Object-Centric Learning with Autoregressive Transformers (Ioannis Kakogeorgiou et al., 2023)

{{<citation>}}

Ioannis Kakogeorgiou, Spyros Gidaris, Konstantinos Karantzalos, Nikos Komodakis. (2023)  
**SPOT: Self-Training with Patch-Order Permutation for Object-Centric Learning with Autoregressive Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.00648v1)  

---


**ABSTRACT**  
Unsupervised object-centric learning aims to decompose scenes into interpretable object entities, termed slots. Slot-based auto-encoders stand out as a prominent method for this task. Within them, crucial aspects include guiding the encoder to generate object-specific slots and ensuring the decoder utilizes them during reconstruction. This work introduces two novel techniques, (i) an attention-based self-training approach, which distills superior slot-based attention masks from the decoder to the encoder, enhancing object segmentation, and (ii) an innovative patch-order permutation strategy for autoregressive transformers that strengthens the role of slot vectors in reconstruction. The effectiveness of these strategies is showcased experimentally. The combined approach significantly surpasses prior slot-based autoencoder methods in unsupervised object segmentation, especially with complex real-world images. We provide the implementation code at https://github.com/gkakogeorgiou/spot .

{{</citation>}}


### (68/119) QAFE-Net: Quality Assessment of Facial Expressions with Landmark Heatmaps (Shuchao Duan et al., 2023)

{{<citation>}}

Shuchao Duan, Amirhossein Dadashzadeh, Alan Whone, Majid Mirmehdi. (2023)  
**QAFE-Net: Quality Assessment of Facial Expressions with Landmark Heatmaps**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.00856v1)  

---


**ABSTRACT**  
Facial expression recognition (FER) methods have made great inroads in categorising moods and feelings in humans. Beyond FER, pain estimation methods assess levels of intensity in pain expressions, however assessing the quality of all facial expressions is of critical value in health-related applications. In this work, we address the quality of five different facial expressions in patients affected by Parkinson's disease. We propose a novel landmark-guided approach, QAFE-Net, that combines temporal landmark heatmaps with RGB data to capture small facial muscle movements that are encoded and mapped to severity scores. The proposed approach is evaluated on a new Parkinson's Disease Facial Expression dataset (PFED5), as well as on the pain estimation benchmark, the UNBC-McMaster Shoulder Pain Expression Archive Database. Our comparative experiments demonstrate that the proposed method outperforms SOTA action quality assessment works on PFED5 and achieves lower mean absolute error than the SOTA pain estimation methods on UNBC-McMaster. Our code and the new PFED5 dataset are available at https://github.com/shuchaoduan/QAFE-Net.

{{</citation>}}


### (69/119) Towards Efficient 3D Object Detection in Bird's-Eye-View Space for Autonomous Driving: A Convolutional-Only Approach (Yuxin Li et al., 2023)

{{<citation>}}

Yuxin Li, Qiang Han, Mengying Yu, Yuxin Jiang, Chaikiat Yeo, Yiheng Li, Zihang Huang, Nini Liu, Hsuanhan Chen, Xiaojun Wu. (2023)  
**Towards Efficient 3D Object Detection in Bird's-Eye-View Space for Autonomous Driving: A Convolutional-Only Approach**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.00633v1)  

---


**ABSTRACT**  
3D object detection in Bird's-Eye-View (BEV) space has recently emerged as a prevalent approach in the field of autonomous driving. Despite the demonstrated improvements in accuracy and velocity estimation compared to perspective view methods, the deployment of BEV-based techniques in real-world autonomous vehicles remains challenging. This is primarily due to their reliance on vision-transformer (ViT) based architectures, which introduce quadratic complexity with respect to the input resolution. To address this issue, we propose an efficient BEV-based 3D detection framework called BEVENet, which leverages a convolutional-only architectural design to circumvent the limitations of ViT models while maintaining the effectiveness of BEV-based methods. Our experiments show that BEVENet is 3$\times$ faster than contemporary state-of-the-art (SOTA) approaches on the NuScenes challenge, achieving a mean average precision (mAP) of 0.456 and a nuScenes detection score (NDS) of 0.555 on the NuScenes validation dataset, with an inference speed of 47.6 frames per second. To the best of our knowledge, this study stands as the first to achieve such significant efficiency improvements for BEV-based methods, highlighting their enhanced feasibility for real-world autonomous driving applications.

{{</citation>}}


### (70/119) BCN: Batch Channel Normalization for Image Classification (Afifa Khaled et al., 2023)

{{<citation>}}

Afifa Khaled, Chao Li, Jia Ning, Kun He. (2023)  
**BCN: Batch Channel Normalization for Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Image Classification, Transformer  
[Paper Link](http://arxiv.org/abs/2312.00596v1)  

---


**ABSTRACT**  
Normalization techniques have been widely used in the field of deep learning due to their capability of enabling higher learning rates and are less careful in initialization. However, the effectiveness of popular normalization technologies is typically limited to specific areas. Unlike the standard Batch Normalization (BN) and Layer Normalization (LN), where BN computes the mean and variance along the (N,H,W) dimensions and LN computes the mean and variance along the (C,H,W) dimensions (N, C, H and W are the batch, channel, spatial height and width dimension, respectively), this paper presents a novel normalization technique called Batch Channel Normalization (BCN). To exploit both the channel and batch dependence and adaptively and combine the advantages of BN and LN based on specific datasets or tasks, BCN separately normalizes inputs along the (N, H, W) and (C, H, W) axes, then combines the normalized outputs based on adaptive parameters. As a basic block, BCN can be easily integrated into existing models for various applications in the field of computer vision. Empirical results show that the proposed technique can be seamlessly applied to various versions of CNN or Vision Transformer architecture. The code is publicly available at https://github.com/AfifaKhaled/BatchChannel-Normalization

{{</citation>}}


### (71/119) Event Recognition in Laparoscopic Gynecology Videos with Hybrid Transformers (Sahar Nasirihaghighi et al., 2023)

{{<citation>}}

Sahar Nasirihaghighi, Negin Ghamsarian, Heinrich Husslein, Klaus Schoeffmann. (2023)  
**Event Recognition in Laparoscopic Gynecology Videos with Hybrid Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Event Recognition, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.00593v1)  

---


**ABSTRACT**  
Analyzing laparoscopic surgery videos presents a complex and multifaceted challenge, with applications including surgical training, intra-operative surgical complication prediction, and post-operative surgical assessment. Identifying crucial events within these videos is a significant prerequisite in a majority of these applications. In this paper, we introduce a comprehensive dataset tailored for relevant event recognition in laparoscopic gynecology videos. Our dataset includes annotations for critical events associated with major intra-operative challenges and post-operative complications. To validate the precision of our annotations, we assess event recognition performance using several CNN-RNN architectures. Furthermore, we introduce and evaluate a hybrid transformer architecture coupled with a customized training-inference framework to recognize four specific events in laparoscopic surgery videos. Leveraging the Transformer networks, our proposed architecture harnesses inter-frame dependencies to counteract the adverse effects of relevant content occlusion, motion blur, and surgical scene variation, thus significantly enhancing event recognition accuracy. Moreover, we present a frame sampling strategy designed to manage variations in surgical scenes and the surgeons' skill level, resulting in event recognition with high temporal resolution. We empirically demonstrate the superiority of our proposed methodology in event recognition compared to conventional CNN-RNN architectures through a series of extensive experiments.

{{</citation>}}


### (72/119) Less is More: Learning Reference Knowledge Using No-Reference Image Quality Assessment (Xudong Li et al., 2023)

{{<citation>}}

Xudong Li, Jingyuan Zheng, Xiawu Zheng, Runze Hu, Enwei Zhang, Yuting Gao, Yunhang Shen, Ke Li, Yutao Liu, Pingyang Dai, Yan Zhang, Rongrong Ji. (2023)  
**Less is More: Learning Reference Knowledge Using No-Reference Image Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.00591v1)  

---


**ABSTRACT**  
Image Quality Assessment (IQA) with reference images have achieved great success by imitating the human vision system, in which the image quality is effectively assessed by comparing the query image with its pristine reference image. However, for the images in the wild, it is quite difficult to access accurate reference images. We argue that it is possible to learn reference knowledge under the No-Reference Image Quality Assessment (NR-IQA) setting, which is effective and efficient empirically. Concretely, by innovatively introducing a novel feature distillation method in IQA, we propose a new framework to learn comparative knowledge from non-aligned reference images. And then, to achieve fast convergence and avoid overfitting, we further propose an inductive bias regularization. Such a framework not only solves the congenital defects of NR-IQA but also improves the feature extraction framework, enabling it to express more abundant quality information. Surprisingly, our method utilizes less input while obtaining a more significant improvement compared to the teacher models. Extensive experiments on eight standard NR-IQA datasets demonstrate the superior performance to the state-of-the-art NR-IQA methods, i.e., achieving the PLCC values of 0.917 (vs. 0.884 in LIVEC) and 0.686 (vs. 0.661 in LIVEFB).

{{</citation>}}


### (73/119) Generative models for visualising abstract social processes: Guiding streetview image synthesis of StyleGAN2 with indices of deprivation (Aleksi Knuutila, 2023)

{{<citation>}}

Aleksi Knuutila. (2023)  
**Generative models for visualising abstract social processes: Guiding streetview image synthesis of StyleGAN2 with indices of deprivation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2312.00570v1)  

---


**ABSTRACT**  
This paper presents a novel application of Generative Adverserial Networks (GANs) to study visual aspects of social processes. I train a a StyleGAN2-model on a custom dataset of 14,564 images of London, sourced from Google Streetview taken in London. After training, I invert the images in the training set, finding points in the model's latent space that correspond to them, and compare results from three inversion techniques. I connect each data point with metadata from the Indices of Multiple Deprivation, describing income, health and environmental quality in the area where the photographs were taken. It is then possible to map which parts of the model's latent space encode visual features that are distinctive for health, income and environmental quality, and condition the synthesis of new images based on these factors. The synthetic images created reflect visual features of social processes that were previously unknown and difficult to study, describing recurring visual differences between deprived and privileged areas in London. GANs are known for their capability to produce a continuous range of images that exhibit visual differences. The paper tests how to exploit this ability through visual comparisons in still images as well as through an interactive website where users can guide image synthesis with sliders. Though conditioned synthesis has its limitations and the results are difficult to validate, the paper points to the potential for generative models to be repurposed to be parts of social scientific methods.

{{</citation>}}


### (74/119) Explainable AI in Diagnosing and Anticipating Leukemia Using Transfer Learning Method (Wahidul Hasan Abir et al., 2023)

{{<citation>}}

Wahidul Hasan Abir, Md. Fahim Uddin, Faria Rahman Khanam, Mohammad Monirujjaman Khan. (2023)  
**Explainable AI in Diagnosing and Anticipating Leukemia Using Transfer Learning Method**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00487v1)  

---


**ABSTRACT**  
This research paper focuses on Acute Lymphoblastic Leukemia (ALL), a form of blood cancer prevalent in children and teenagers, characterized by the rapid proliferation of immature white blood cells (WBCs). These atypical cells can overwhelm healthy cells, leading to severe health consequences. Early and accurate detection of ALL is vital for effective treatment and improving survival rates. Traditional diagnostic methods are time-consuming, costly, and prone to errors. The paper proposes an automated detection approach using computer-aided diagnostic (CAD) models, leveraging deep learning techniques to enhance the accuracy and efficiency of leukemia diagnosis. The study utilizes various transfer learning models like ResNet101V2, VGG19, InceptionV3, and InceptionResNetV2 for classifying ALL. The methodology includes using the Local Interpretable Model-Agnostic Explanations (LIME) for ensuring the validity and reliability of the AI system's predictions. This approach is critical for overcoming the "black box" nature of AI, where decisions made by models are often opaque and unaccountable. The paper highlights that the proposed method using the InceptionV3 model achieved an impressive 98.38% accuracy, outperforming other tested models. The results, verified by the LIME algorithm, showcase the potential of this method in accurately identifying ALL, providing a valuable tool for medical practitioners. The research underscores the impact of explainable artificial intelligence (XAI) in medical diagnostics, paving the way for more transparent and trustworthy AI applications in healthcare.

{{</citation>}}


### (75/119) Dolphins: Multimodal Language Model for Driving (Yingzi Ma et al., 2023)

{{<citation>}}

Yingzi Ma, Yulong Cao, Jiachen Sun, Marco Pavone, Chaowei Xiao. (2023)  
**Dolphins: Multimodal Language Model for Driving**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00438v1)  

---


**ABSTRACT**  
The quest for fully autonomous vehicles (AVs) capable of navigating complex real-world scenarios with human-like understanding and responsiveness. In this paper, we introduce Dolphins, a novel vision-language model architected to imbibe human-like abilities as a conversational driving assistant. Dolphins is adept at processing multimodal inputs comprising video (or image) data, text instructions, and historical control signals to generate informed outputs corresponding to the provided instructions. Building upon the open-sourced pretrained Vision-Language Model, OpenFlamingo, we first enhance Dolphins's reasoning capabilities through an innovative Grounded Chain of Thought (GCoT) process. Then we tailored Dolphins to the driving domain by constructing driving-specific instruction data and conducting instruction tuning. Through the utilization of the BDD-X dataset, we designed and consolidated four distinct AV tasks into Dolphins to foster a holistic understanding of intricate driving scenarios. As a result, the distinctive features of Dolphins are characterized into two dimensions: (1) the ability to provide a comprehensive understanding of complex and long-tailed open-world driving scenarios and solve a spectrum of AV tasks, and (2) the emergence of human-like capabilities including gradient-free instant adaptation via in-context learning and error recovery via reflection.

{{</citation>}}


### (76/119) Enhancing Image Captioning with Neural Models (Pooja Bhatnagar et al., 2023)

{{<citation>}}

Pooja Bhatnagar, Sai Mrunaal, Sachin Kamnure. (2023)  
**Enhancing Image Captioning with Neural Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-NE, cs.CV  
Keywords: Image Captioning  
[Paper Link](http://arxiv.org/abs/2312.00435v1)  

---


**ABSTRACT**  
This research explores the realm of neural image captioning using deep learning models. The study investigates the performance of different neural architecture configurations, focusing on the inject architecture, and proposes a novel quality metric for evaluating caption generation. Through extensive experimentation and analysis, this work sheds light on the challenges and opportunities in image captioning, providing insights into model behavior and overfitting. The results reveal that while the merge models exhibit a larger vocabulary and higher ROUGE scores, the inject architecture generates relevant and concise image captions. The study also highlights the importance of refining training data and optimizing hyperparameters for improved model performance. This research contributes to the growing body of knowledge in neural image captioning and encourages further exploration in the field, emphasizing the democratization of artificial intelligence.

{{</citation>}}


### (77/119) Large-scale Vision-Language Models Learn Super Images for Efficient and High-Performance Partially Relevant Video Retrieval (Taichi Nishimura et al., 2023)

{{<citation>}}

Taichi Nishimura, Shota Nakada, Masayoshi Kondo. (2023)  
**Large-scale Vision-Language Models Learn Super Images for Efficient and High-Performance Partially Relevant Video Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00414v1)  

---


**ABSTRACT**  
In this paper, we propose an efficient and high-performance method for partially relevant video retrieval (PRVR), which aims to retrieve untrimmed long videos that contain at least one relevant moment to the input text query. In terms of both efficiency and performance, the overlooked bottleneck of previous studies is the visual encoding of dense frames. This guides researchers to choose lightweight visual backbones, yielding sub-optimal retrieval performance due to their limited capabilities of learned visual representations. However, it is undesirable to simply replace them with high-performance large-scale vision-and-language models (VLMs) due to their low efficiency. To address these issues, instead of dense frames, we focus on super images, which are created by rearranging the video frames in a $N \times N$ grid layout. This reduces the number of visual encodings to $\frac{1}{N^2}$ and compensates for the low efficiency of large-scale VLMs, allowing us to adopt them as powerful encoders. Surprisingly, we discover that with a simple query-image attention trick, VLMs generalize well to super images effectively and demonstrate promising zero-shot performance against SOTA methods efficiently. In addition, we propose a fine-tuning approach by incorporating a few trainable modules into the VLM backbones. The experimental results demonstrate that our approaches efficiently achieve the best performance on ActivityNet Captions and TVR.

{{</citation>}}


### (78/119) SCHEME: Scalable Channer Mixer for Vision Transformers (Deepak Sridhar et al., 2023)

{{<citation>}}

Deepak Sridhar, Yunsheng Li, Nuno Vasconcelos. (2023)  
**SCHEME: Scalable Channer Mixer for Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.00412v1)  

---


**ABSTRACT**  
Vision Transformers have received significant attention due to their impressive performance in many vision tasks. While the token mixer or attention block has been studied in great detail, the channel mixer or feature mixing block (FFN or MLP) has not been explored in depth albeit it accounts for a bulk of the parameters and computation in a model. In this work, we study whether sparse feature mixing can replace the dense connections and confirm this with a block diagonal MLP structure that improves the accuracy by supporting larger expansion ratios. To improve the feature clusters formed by this structure and thereby further improve the accuracy, a lightweight, parameter-free, channel covariance attention (CCA) mechanism is introduced as a parallel branch during training. This design of CCA enables gradual feature mixing across channel groups during training whose contribution decays to zero as the training progresses to convergence. This allows the CCA block to be discarded during inference, thus enabling enhanced performance with no additional computational cost. The resulting $\textit{Scalable CHannEl MixEr}$ (SCHEME) can be plugged into any ViT architecture to obtain a gamut of models with different trade-offs between complexity and performance by controlling the block diagonal structure size in the MLP. This is shown by the introduction of a new family of SCHEMEformer models. Experiments on image classification, object detection, and semantic segmentation, with different ViT backbones, consistently demonstrate substantial accuracy gains over existing designs, especially under lower FLOPs regimes. For example, the SCHEMEformer establishes a new SOTA of 79.7% accuracy for ViTs using pure attention mixers on ImageNet-1K at 1.77G FLOPs.

{{</citation>}}


### (79/119) VIoTGPT: Learning to Schedule Vision Tools towards Intelligent Video Internet of Things (Yaoyao Zhong et al., 2023)

{{<citation>}}

Yaoyao Zhong, Mengshi Qi, Rui Wang, Yuhan Qiu, Yang Zhang, Huadong Ma. (2023)  
**VIoTGPT: Learning to Schedule Vision Tools towards Intelligent Video Internet of Things**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2312.00401v1)  

---


**ABSTRACT**  
Video Internet of Things (VIoT) has shown full potential in collecting an unprecedented volume of video data. Learning to schedule perceiving models and analyzing the collected videos intelligently will be potential sparks for VIoT. In this paper, to address the challenges posed by the fine-grained and interrelated vision tool usage of VIoT, we build VIoTGPT, the framework based on LLMs to correctly interact with humans, query knowledge videos, and invoke vision models to accomplish complicated tasks. To support VIoTGPT and related future works, we meticulously crafted the training dataset and established benchmarks involving 11 representative vision models across three categories based on semi-automatic annotations. To guide LLM to act as the intelligent agent towards intelligent VIoT, we resort to ReAct instruction tuning based on the collected VIoT dataset to learn the tool capability. Quantitative and qualitative experimental results and analyses demonstrate the effectiveness of VIoTGPT.

{{</citation>}}


### (80/119) Learning to Estimate Critical Gait Parameters from Single-View RGB Videos with Transformer-Based Attention Network (Quoc Hung T. Le et al., 2023)

{{<citation>}}

Quoc Hung T. Le, Hieu H. Pham. (2023)  
**Learning to Estimate Critical Gait Parameters from Single-View RGB Videos with Transformer-Based Attention Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Clinical, Transformer  
[Paper Link](http://arxiv.org/abs/2312.00398v1)  

---


**ABSTRACT**  
Musculoskeletal diseases and cognitive impairments in patients lead to difficulties in movement as well as negative effects on their psychological health. Clinical gait analysis, a vital tool for early diagnosis and treatment, traditionally relies on expensive optical motion capture systems. Recent advances in computer vision and deep learning have opened the door to more accessible and cost-effective alternatives. This paper introduces a novel spatio-temporal Transformer network to estimate critical gait parameters from RGB videos captured by a single-view camera. Empirical evaluations on a public dataset of cerebral palsy patients indicate that the proposed framework surpasses current state-of-the-art approaches and show significant improvements in predicting general gait parameters (including Walking Speed, Gait Deviation Index - GDI, and Knee Flexion Angle at Maximum Extension), while utilizing fewer parameters and alleviating the need for manual feature extraction.

{{</citation>}}


### (81/119) Study and Survey on Gesture Recognition Systems (Kshitij Deshpande et al., 2023)

{{<citation>}}

Kshitij Deshpande, Varad Mashalkar, Kaustubh Mhaisekar, Amaan Naikwadi, Archana Ghotkar. (2023)  
**Study and Survey on Gesture Recognition Systems**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.00392v1)  

---


**ABSTRACT**  
In recent years, there has been a considerable amount of research in the Gesture Recognition domain, mainly owing to the technological advancements in Computer Vision. Various new applications have been conceptualised and developed in this field. This paper discusses the implementation of gesture recognition systems in multiple sectors such as gaming, healthcare, home appliances, industrial robots, and virtual reality. Different methodologies for capturing gestures are compared and contrasted throughout this survey. Various data sources and data acquisition techniques have been discussed. The role of gestures in sign language has been studied and existing approaches have been reviewed. Common challenges faced while building gesture recognition systems have also been explored.

{{</citation>}}


### (82/119) VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models (Hyeonho Jeong et al., 2023)

{{<citation>}}

Hyeonho Jeong, Geon Yeong Park, Jong Chul Ye. (2023)  
**VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.00845v1)  

---


**ABSTRACT**  
Text-to-video diffusion models have advanced video generation significantly. However, customizing these models to generate videos with tailored motions presents a substantial challenge. In specific, they encounter hurdles in (a) accurately reproducing motion from a target video, and (b) creating diverse visual variations. For example, straightforward extensions of static image customization methods to video often lead to intricate entanglements of appearance and motion data. To tackle this, here we present the Video Motion Customization (VMC) framework, a novel one-shot tuning approach crafted to adapt temporal attention layers within video diffusion models. Our approach introduces a novel motion distillation objective using residual vectors between consecutive frames as a motion reference. The diffusion process then preserves low-frequency motion trajectories while mitigating high-frequency motion-unrelated noise in image space. We validate our method against state-of-the-art video generative models across diverse real-world motions and contexts. Our codes, data and the project demo can be found at https://video-motion-customization.github.io

{{</citation>}}


### (83/119) SynFundus: Generating a synthetic fundus images dataset with millions of samples and multi-disease annotations (Fangxin Shang et al., 2023)

{{<citation>}}

Fangxin Shang, Jie Fu, Yehui Yang, Lei Ma. (2023)  
**SynFundus: Generating a synthetic fundus images dataset with millions of samples and multi-disease annotations**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.00377v1)  

---


**ABSTRACT**  
In the field of medical imaging, the scarcity of large-scale datasets due to privacy restrictions stands as a significant barrier to develop large models for medical. To address this issue, we introduce SynFundus-1M, a high-quality synthetic dataset with over 1 million retinal fundus images and extensive disease and pathologies annotations, which is generated by a Denoising Diffusion Probabilistic Model. The SynFundus-Generator and SynFundus-1M achieve superior Frechet Inception Distance (FID) scores compared to existing methods on main-stream public real datasets. Furthermore, the ophthalmologists evaluation validate the difficulty in discerning these synthetic images from real ones, confirming the SynFundus-1M's authenticity. Through extensive experiments, we demonstrate that both CNN and ViT can benifit from SynFundus-1M by pretraining or training directly. Compared to datasets like ImageNet or EyePACS, models train on SynFundus-1M not only achieve better performance but also faster convergence on various downstream tasks.

{{</citation>}}


### (84/119) Efficient Multimodal Semantic Segmentation via Dual-Prompt Learning (Shaohua Dong et al., 2023)

{{<citation>}}

Shaohua Dong, Yunhe Feng, Qing Yang, Yan Huang, Dongfang Liu, Heng Fan. (2023)  
**Efficient Multimodal Semantic Segmentation via Dual-Prompt Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.00360v2)  

---


**ABSTRACT**  
Multimodal (e.g., RGB-Depth/RGB-Thermal) fusion has shown great potential for improving semantic segmentation in complex scenes (e.g., indoor/low-light conditions). Existing approaches often fully fine-tune a dual-branch encoder-decoder framework with a complicated feature fusion strategy for achieving multimodal semantic segmentation, which is training-costly due to the massive parameter updates in feature extraction and fusion. To address this issue, we propose a surprisingly simple yet effective dual-prompt learning network (dubbed DPLNet) for training-efficient multimodal (e.g., RGB-D/T) semantic segmentation. The core of DPLNet is to directly adapt a frozen pre-trained RGB model to multimodal semantic segmentation, reducing parameter updates. For this purpose, we present two prompt learning modules, comprising multimodal prompt generator (MPG) and multimodal feature adapter (MFA). MPG works to fuse the features from different modalities in a compact manner and is inserted from shadow to deep stages to generate the multi-level multimodal prompts that are injected into the frozen backbone, while MPG adapts prompted multimodal features in the frozen backbone for better multimodal semantic segmentation. Since both the MPG and MFA are lightweight, only a few trainable parameters (3.88M, 4.4% of the pre-trained backbone parameters) are introduced for multimodal feature fusion and learning. Using a simple decoder (3.27M parameters), DPLNet achieves new state-of-the-art performance or is on a par with other complex approaches on four RGB-D/T semantic segmentation datasets while satisfying parameter efficiency. Moreover, we show that DPLNet is general and applicable to other multimodal tasks such as salient object detection and video semantic segmentation. Without special design, DPLNet outperforms many complicated models. Our code will be available at github.com/ShaohuaDong2021/DPLNet.

{{</citation>}}


### (85/119) Manipulating the Label Space for In-Context Classification (Haokun Chen et al., 2023)

{{<citation>}}

Haokun Chen, Xu Yang, Yuhang Huang, Zihan Wu, Jing Wang, Xin Geng. (2023)  
**Manipulating the Label Space for In-Context Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Language Model  
[Paper Link](http://arxiv.org/abs/2312.00351v1)  

---


**ABSTRACT**  
After pre-training by generating the next word conditional on previous words, the Language Model (LM) acquires the ability of In-Context Learning (ICL) that can learn a new task conditional on the context of the given in-context examples (ICEs). Similarly, visually-conditioned Language Modelling is also used to train Vision-Language Models (VLMs) with ICL ability. However, such VLMs typically exhibit weaker classification abilities compared to contrastive learning-based models like CLIP, since the Language Modelling objective does not directly contrast whether an object is paired with a text. To improve the ICL of classification, using more ICEs to provide more knowledge is a straightforward way. However, this may largely increase the selection time, and more importantly, the inclusion of additional in-context images tends to extend the length of the in-context sequence beyond the processing capacity of a VLM. To alleviate these limitations, we propose to manipulate the label space of each ICE to increase its knowledge density, allowing for fewer ICEs to convey as much information as a larger set would. Specifically, we propose two strategies which are Label Distribution Enhancement and Visual Descriptions Enhancement to improve In-context classification performance on diverse datasets, including the classic ImageNet and more fine-grained datasets like CUB-200. Specifically, using our approach on ImageNet, we increase accuracy from 74.70\% in a 4-shot setting to 76.21\% with just 2 shots. surpassing CLIP by 0.67\%. On CUB-200, our method raises 1-shot accuracy from 48.86\% to 69.05\%, 12.15\% higher than CLIP. The code is given in https://anonymous.4open.science/r/MLS_ICC.

{{</citation>}}


### (86/119) Learning Anatomically Consistent Embedding for Chest Radiography (Ziyu Zhou et al., 2023)

{{<citation>}}

Ziyu Zhou, Haozhe Luo, Jiaxuan Pang, Xiaowei Ding, Michael Gotway, Jianming Liang. (2023)  
**Learning Anatomically Consistent Embedding for Chest Radiography**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.00335v1)  

---


**ABSTRACT**  
Self-supervised learning (SSL) approaches have recently shown substantial success in learning visual representations from unannotated images. Compared with photographic images, medical images acquired with the same imaging protocol exhibit high consistency in anatomy. To exploit this anatomical consistency, this paper introduces a novel SSL approach, called PEAC (patch embedding of anatomical consistency), for medical image analysis. Specifically, in this paper, we propose to learn global and local consistencies via stable grid-based matching, transfer pre-trained PEAC models to diverse downstream tasks, and extensively demonstrate that (1) PEAC achieves significantly better performance than the existing state-of-the-art fully/self-supervised methods, and (2) PEAC captures the anatomical structure consistency across views of the same patient and across patients of different genders, weights, and healthy statuses, which enhances the interpretability of our method for medical image analysis.

{{</citation>}}


### (87/119) Adaptability of Computer Vision at the Tactical Edge: Addressing Environmental Uncertainty (Hayden Moore, 2023)

{{<citation>}}

Hayden Moore. (2023)  
**Adaptability of Computer Vision at the Tactical Edge: Addressing Environmental Uncertainty**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.00269v1)  

---


**ABSTRACT**  
Computer Vision (CV) systems are increasingly being adopted into Command and Control (C2) systems to improve intelligence analysis on the battlefield, the tactical edge. CV systems leverage Artificial Intelligence (AI) algorithms to help visualize and interpret the environment, enhancing situational awareness. However, the adaptability of CV systems at the tactical edge remains challenging due to rapidly changing environments and objects which can confuse the deployed models. A CV model leveraged in this environment can become uncertain in its predictions, as the environment and the objects existing in the environment begin to change. Additionally, mission objectives can rapidly change leading to adjustments in technology, camera angles, and image resolutions. All of which can negatively affect the performance of and potentially introduce uncertainty into the system. When the training environment and/or technology differs from the deployment environment, CV models can perform unexpectedly. Unfortunately, most scenarios at the tactical edge do not incorporate Uncertainty Quantification (UQ) into their deployed C2 and CV systems. This concept paper explores the idea of synchronizing robust data operations and model fine-tuning driven by UQ all at the tactical edge. Specifically, curating datasets and training child models based on the residuals of predictions, using these child models to calculate prediction intervals (PI), and then using these PI to calibrate the deployed models. By incorporating UQ into the core operations surrounding C2 and CV systems at the tactical edge, we can help drive purposeful adaptability on the battlefield.

{{</citation>}}


## cs.CR (8)



### (88/119) Survey of Security Issues in Memristor-based Machine Learning Accelerators for RF Analysis (William Lillis et al., 2023)

{{<citation>}}

William Lillis, Max Cohen Hoffing, Wayne Burleson. (2023)  
**Survey of Security Issues in Memristor-based Machine Learning Accelerators for RF Analysis**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SY, cs.CR, eess-SY  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.00942v1)  

---


**ABSTRACT**  
We explore security aspects of a new computing paradigm that combines novel memristors and traditional Complimentary Metal Oxide Semiconductor (CMOS) to construct a highly efficient analog and/or digital fabric that is especially well-suited to Machine Learning (ML) inference processors for Radio Frequency (RF) signals. Memristors have different properties than traditional CMOS which can potentially be exploited by attackers. In addition, the mixed signal approximate computing model has different vulnerabilities than traditional digital implementations. However both the memristor and the ML computation can be leveraged to create security mechanisms and countermeasures ranging from lightweight cryptography, identifiers (e.g. Physically Unclonable Functions (PUFs), fingerprints, and watermarks), entropy sources, hardware obfuscation and leakage/attack detection methods. Three different threat models are proposed: 1) Supply Chain, 2) Physical Attacks, and 3) Remote Attacks. For each threat model, potential vulnerabilities and defenses are identified. This survey reviews a variety of recent work from the hardware and ML security literature and proposes open problems for both attack and defense. The survey emphasizes the growing area of RF signal analysis and identification in terms of the commercial space, as well as military applications and threat models. We differ from other other recent surveys that target ML in general, neglecting RF applications.

{{</citation>}}


### (89/119) Using Honeybuckets to Characterize Cloud Storage Scanning in the Wild (Katherine Izhikevich et al., 2023)

{{<citation>}}

Katherine Izhikevich, Geoff Voelker, Stefan Savage, Liz Izhikevich. (2023)  
**Using Honeybuckets to Characterize Cloud Storage Scanning in the Wild**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR  
Keywords: AWS  
[Paper Link](http://arxiv.org/abs/2312.00580v1)  

---


**ABSTRACT**  
In this work, we analyze to what extent actors target poorly-secured cloud storage buckets for attack. We deployed hundreds of AWS S3 honeybuckets with different names and content to lure and measure different scanning strategies. Actors exhibited clear preferences for scanning buckets that appeared to belong to organizations, especially commercial entities in the technology sector with a vulnerability disclosure program. Actors continuously engaged with the content of buckets by downloading, uploading, and deleting files. Most alarmingly, we recorded multiple instances in which malicious actors downloaded, read, and understood a document from our honeybucket, leading them to attempt to gain unauthorized server access.

{{</citation>}}


### (90/119) Hiding in text/plain sight: Security defences of Tor Onion Services (Q Misell, 2023)

{{<citation>}}

Q Misell. (2023)  
**Hiding in text/plain sight: Security defences of Tor Onion Services**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.00545v1)  

---


**ABSTRACT**  
Tor Onion Services are a way to host websites and other internet services anonymously. Onion Services are often used to bypass internet censorship and provide information services to users in oppressive regimes. This paper presents an analysis of the security defences deployed on these Onion Services. Onion Services tend to have better security policy than sites on the clear web. However they lag behind in the deployment of HTTPS, a key defence to ensuring the security of users of such services.

{{</citation>}}


### (91/119) The Impact of Privacy and Security Attitudes and Concerns of Travellers on Their Willingness to Use Mobility-as-a-Service Systems (Maria Sophia Heering et al., 2023)

{{<citation>}}

Maria Sophia Heering, Haiyue Yuan, Shujun Li. (2023)  
**The Impact of Privacy and Security Attitudes and Concerns of Travellers on Their Willingness to Use Mobility-as-a-Service Systems**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CY, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.00519v1)  

---


**ABSTRACT**  
This paper reports results from an online survey on the impact of travellers' privacy and security attitudes and concerns on their willingness to use mobility-as-a-service (MaaS) systems. This study is part of a larger project that aims at investigating barriers to potential MaaS uptake. The online survey was designed to cover data privacy and security attitudes and concerns as well as a variety of socio-psychological and socio-demographic variables associated with travellers' intentions to use MaaS systems. The study involved $n=320$ UK participants recruited via the Prolific survey platform. Overall, correlation analysis and a multiple regression model indicated that, neither attitudes nor concerns of participants over the privacy and security of personal data would significantly impact their decisions to use MaaS systems, which was an unexpected result, however, their trust in (commercial and governmental) websites would. Another surprising result is that, having been a victim of improper invasion of privacy did not appear to affect individuals' intentions to use MaaS systems, whereas frequency with which one heard about misuse of personal data did. Implications of the results and future directions are also discussed, e.g., MaaS providers are encouraged to work on improving the trustworthiness of their corporate image.

{{</citation>}}


### (92/119) PyraTrans: Learning Attention-Enriched Multi-Scale Pyramid Network from Pre-Trained Transformers for Effective Malicious URL Detection (Ruitong Liu et al., 2023)

{{<citation>}}

Ruitong Liu, Yanbin Wang, Zhenhao Guo, Haitao Xu, Zhan Qin, Wenrui Ma, Fan Zhang. (2023)  
**PyraTrans: Learning Attention-Enriched Multi-Scale Pyramid Network from Pre-Trained Transformers for Effective Malicious URL Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Attention, BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.00508v1)  

---


**ABSTRACT**  
Detecting malicious URLs is a crucial aspect of web search and mining, significantly impacting internet security. Though advancements in machine learning have improved the effectiveness of detection methods, these methods still face significant challenges in their capacity to generalize and their resilience against evolving threats. In this paper, we propose PyraTrans, an approach that combines the strengths of pretrained Transformers and pyramid feature learning for improving malicious URL detection. We implement PyraTrans by leveraging a pretrained CharBERT as the base and augmenting it with 3 connected feature modules: 1) The Encoder Feature Extraction module, which extracts representations from each encoder layer of CharBERT to obtain multi-order features; 2) The Multi-Scale Feature Learning Module, which captures multi-scale local contextual insights and aggregate information across different layer-levels; and 3) The Pyramid Spatial Attention Module, which learns hierarchical and spatial feature attentions, highlighting critical classification signals while reducing noise. The proposed approach addresses the limitations of the Transformer in local feature learning and spatial awareness, and enabling us to extract multi-order, multi-scale URL feature representations with enhanced attentional focus. PyraTrans is evaluated using 4 benchmark datasets, where it demonstrated significant advancements over prior baseline methods. Particularly, on the imbalanced dataset, our method, with just 10% of the data for training, the TPR is 3.3-6.5 times and the F1-score is 2.9-4.5 times that of the baseline. Our approach also demonstrates robustness against adversarial attacks. Codes and data are available at https://github.com/Alixyvtte/PyraTrans.

{{</citation>}}


### (93/119) MalDicom: A Memory Forensic Framework for Detecting Malicious Payload in DICOM Files (Ayushi Mishra et al., 2023)

{{<citation>}}

Ayushi Mishra, Priyanka Bagade. (2023)  
**MalDicom: A Memory Forensic Framework for Detecting Malicious Payload in DICOM Files**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.00483v1)  

---


**ABSTRACT**  
Digital Imaging and Communication System (DICOM) is widely used throughout the public health sector for portability in medical imaging. However, these DICOM files have vulnerabilities present in the preamble section. Successful exploitation of these vulnerabilities can allow attackers to embed executable codes in the 128-Byte preamble of DICOM files. Embedding the malicious executable will not interfere with the readability or functionality of DICOM imagery. However, it will affect the underline system silently upon viewing these files. This paper shows the infiltration of Windows malware executables into DICOM files. On viewing the files, the malicious DICOM will get executed and eventually infect the entire hospital network through the radiologist's workstation. The code injection process of executing malware in DICOM files affects the hospital networks and workstations' memory. Memory forensics for the infected radiologist's workstation is crucial as it can detect which malware disrupts the hospital environment, and future detection methods can be deployed. In this paper, we consider the machine learning (ML) algorithms to conduct memory forensics on three memory dump categories: Trojan, Spyware, and Ransomware, taken from the CIC-MalMem-2022 dataset. We obtain the highest accuracy of 75\% with the Random Forest model. For estimating the feature importance for ML model prediction, we leveraged the concept of Shapley values.

{{</citation>}}


### (94/119) Unleashing Cheapfakes through Trojan Plugins of Large Language Models (Tian Dong et al., 2023)

{{<citation>}}

Tian Dong, Guoxing Chen, Shaofeng Li, Minhui Xue, Rayne Holland, Yan Meng, Zhen Liu, Haojin Zhu. (2023)  
**Unleashing Cheapfakes through Trojan Plugins of Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00374v1)  

---


**ABSTRACT**  
Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers, an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses LLM-enhanced paraphrasing to polish benchmark poisoned datasets. In contrast, in the absence of a dataset, FUSION leverages an over-poisoning procedure to transform a benign adaptor. Our experiments validate that our attacks provide higher attack effectiveness than the baseline and, for the purpose of attracting downloads, preserves or improves the adapter's utility. Finally, we provide two case studies to demonstrate that the Trojan adapter can lead a LLM-powered autonomous agent to execute unintended scripts or send phishing emails. Our novel attacks represent the first study of supply chain threats for LLMs through the lens of Trojan plugins.

{{</citation>}}


### (95/119) Mark My Words: Analyzing and Evaluating Language Model Watermarks (Julien Piet et al., 2023)

{{<citation>}}

Julien Piet, Chawin Sitawarin, Vivian Fang, Norman Mu, David Wagner. (2023)  
**Mark My Words: Analyzing and Evaluating Language Model Watermarks**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00273v1)  

---


**ABSTRACT**  
The capabilities of large language models have grown significantly in recent years and so too have concerns about their misuse. In this context, the ability to distinguish machine-generated text from human-authored content becomes important. Prior works have proposed numerous schemes to watermark text, which would benefit from a systematic evaluation framework. This work focuses on text watermarking techniques - as opposed to image watermarks - and proposes a comprehensive benchmark for them under different tasks as well as practical attacks. We focus on three main metrics: quality, size (e.g. the number of tokens needed to detect a watermark), and tamper-resistance. Current watermarking techniques are good enough to be deployed: Kirchenbauer et al. can watermark Llama2-7B-chat with no perceivable loss in quality in under 100 tokens, and with good tamper-resistance to simple attacks, regardless of temperature. We argue that watermark indistinguishability is too strong a requirement: schemes that slightly modify logit distributions outperform their indistinguishable counterparts with no noticeable loss in generation quality. We publicly release our benchmark.

{{</citation>}}


## cs.IT (1)



### (96/119) Privacy Preserving Event Detection (Xiaoshan Wang et al., 2023)

{{<citation>}}

Xiaoshan Wang, Tan F. Wong. (2023)  
**Privacy Preserving Event Detection**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Event Detection  
[Paper Link](http://arxiv.org/abs/2312.00933v1)  

---


**ABSTRACT**  
This paper presents a privacy-preserving event detection scheme based on measurements made by a network of sensors. A diameter-like decision statistic made up of the marginal types of the measurements observed by the sensors is employed. The proposed detection scheme can achieve the best type-I error exponent as the type-II error rate is required to be negligible. Detection performance with finite-length observations is also demonstrated through a simulation example of spectrum sensing. Privacy protection is achieved by obfuscating the type data with random zero-modulo-sum numbers that are generated and distributed via the exchange of encrypted messages among the sensors. The privacy-preserving performance against ``honest but curious'' adversaries, including colluding sensors, the fusion center, and external eavesdroppers, is analyzed through a series of cryptographic games. It is shown that the probability that any probabilistic polynomial time adversary successfully estimates the sensors' measured types can not be much better than independent guessing, when there are at least two non-colluding sensors.

{{</citation>}}


## cs.IR (1)



### (97/119) LLM-TAKE: Theme Aware Keyword Extraction Using Large Language Models (Reza Yousefi Maragheh et al., 2023)

{{<citation>}}

Reza Yousefi Maragheh, Chenhao Fang, Charan Chand Irugu, Parth Parikh, Jason Cho, Jianpeng Xu, Saranyan Sukumar, Malay Patel, Evren Korpeoglu, Sushant Kumar, Kannan Achan. (2023)  
**LLM-TAKE: Theme Aware Keyword Extraction Using Large Language Models**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00909v1)  

---


**ABSTRACT**  
Keyword extraction is one of the core tasks in natural language processing. Classic extraction models are notorious for having a short attention span which make it hard for them to conclude relational connections among the words and sentences that are far from each other. This, in turn, makes their usage prohibitive for generating keywords that are inferred from the context of the whole text. In this paper, we explore using Large Language Models (LLMs) in generating keywords for items that are inferred from the items textual metadata. Our modeling framework includes several stages to fine grain the results by avoiding outputting keywords that are non informative or sensitive and reduce hallucinations common in LLM. We call our LLM-based framework Theme-Aware Keyword Extraction (LLM TAKE). We propose two variations of framework for generating extractive and abstractive themes for products in an E commerce setting. We perform an extensive set of experiments on three real data sets and show that our modeling framework can enhance accuracy based and diversity based metrics when compared with benchmark models.

{{</citation>}}


## cs.SE (1)



### (98/119) Leveraging Large Language Models to Improve REST API Testing (Myeongsoo Kim et al., 2023)

{{<citation>}}

Myeongsoo Kim, Tyler Stennett, Dhruv Shah, Saurabh Sinha, Alessandro Orso. (2023)  
**Leveraging Large Language Models to Improve REST API Testing**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.00894v1)  

---


**ABSTRACT**  
The widespread adoption of REST APIs, coupled with their growing complexity and size, has led to the need for automated REST API testing tools. Current testing tools focus on the structured data in REST API specifications but often neglect valuable insights available in unstructured natural-language descriptions in the specifications, which leads to suboptimal test coverage. Recently, to address this gap, researchers have developed techniques that extract rules from these human-readable descriptions and query knowledge bases to derive meaningful input values. However, these techniques are limited in the types of rules they can extract and can produce inaccurate results. This paper presents RESTGPT, an innovative approach that leverages the power and intrinsic context-awareness of Large Language Models (LLMs) to improve REST API testing. RESTGPT takes as input an API specification, extracts machine-interpretable rules, and generates example parameter values from natural-language descriptions in the specification. It then augments the original specification with these rules and values. Our preliminary evaluation suggests that RESTGPT outperforms existing techniques in both rule extraction and value generation. Given these encouraging results, we outline future research directions for leveraging LLMs more broadly for improving REST API testing.

{{</citation>}}


## cs.RO (2)



### (99/119) Towards Generalizable Zero-Shot Manipulation via Translating Human Interaction Plans (Homanga Bharadhwaj et al., 2023)

{{<citation>}}

Homanga Bharadhwaj, Abhinav Gupta, Vikash Kumar, Shubham Tulsiani. (2023)  
**Towards Generalizable Zero-Shot Manipulation via Translating Human Interaction Plans**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.00775v1)  

---


**ABSTRACT**  
We pursue the goal of developing robots that can interact zero-shot with generic unseen objects via a diverse repertoire of manipulation skills and show how passive human videos can serve as a rich source of data for learning such generalist robots. Unlike typical robot learning approaches which directly learn how a robot should act from interaction data, we adopt a factorized approach that can leverage large-scale human videos to learn how a human would accomplish a desired task (a human plan), followed by translating this plan to the robots embodiment. Specifically, we learn a human plan predictor that, given a current image of a scene and a goal image, predicts the future hand and object configurations. We combine this with a translation module that learns a plan-conditioned robot manipulation policy, and allows following humans plans for generic manipulation tasks in a zero-shot manner with no deployment-time training. Importantly, while the plan predictor can leverage large-scale human videos for learning, the translation module only requires a small amount of in-domain data, and can generalize to tasks not seen during training. We show that our learned system can perform over 16 manipulation skills that generalize to 40 objects, encompassing 100 real-world tasks for table-top manipulation and diverse in-the-wild manipulation. https://homangab.github.io/hopman/

{{</citation>}}


### (100/119) TRC: Trust Region Conditional Value at Risk for Safe Reinforcement Learning (Dohyeong Kim et al., 2023)

{{<citation>}}

Dohyeong Kim, Songhwai Oh. (2023)  
**TRC: Trust Region Conditional Value at Risk for Safe Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.00344v1)  

---


**ABSTRACT**  
As safety is of paramount importance in robotics, reinforcement learning that reflects safety, called safe RL, has been studied extensively. In safe RL, we aim to find a policy which maximizes the desired return while satisfying the defined safety constraints. There are various types of constraints, among which constraints on conditional value at risk (CVaR) effectively lower the probability of failures caused by high costs since CVaR is a conditional expectation obtained above a certain percentile. In this paper, we propose a trust region-based safe RL method with CVaR constraints, called TRC. We first derive the upper bound on CVaR and then approximate the upper bound in a differentiable form in a trust region. Using this approximation, a subproblem to get policy gradients is formulated, and policies are trained by iteratively solving the subproblem. TRC is evaluated through safe navigation tasks in simulations with various robots and a sim-to-real environment with a Jackal robot from Clearpath. Compared to other safe RL methods, the performance is improved by 1.93 times while the constraints are satisfied in all experiments.

{{</citation>}}


## cs.HC (4)



### (101/119) Beyond ChatBots: ExploreLLM for Structured Thoughts and Personalized Model Responses (Xiao Ma et al., 2023)

{{<citation>}}

Xiao Ma, Swaroop Mishra, Ariel Liu, Sophie Su, Jilin Chen, Chinmay Kulkarni, Heng-Tze Cheng, Quoc Le, Ed Chi. (2023)  
**Beyond ChatBots: ExploreLLM for Structured Thoughts and Personalized Model Responses**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CL, cs-HC, cs-LG, cs.HC  
Keywords: ChatBot  
[Paper Link](http://arxiv.org/abs/2312.00763v1)  

---


**ABSTRACT**  
Large language model (LLM) powered chatbots are primarily text-based today, and impose a large interactional cognitive load, especially for exploratory or sensemaking tasks such as planning a trip or learning about a new city. Because the interaction is textual, users have little scaffolding in the way of structure, informational "scent", or ability to specify high-level preferences or goals. We introduce ExploreLLM that allows users to structure thoughts, help explore different options, navigate through the choices and recommendations, and to more easily steer models to generate more personalized responses. We conduct a user study and show that users find it helpful to use ExploreLLM for exploratory or planning tasks, because it provides a useful schema-like structure to the task, and guides users in planning. The study also suggests that users can more easily personalize responses with high-level preferences with ExploreLLM. Together, ExploreLLM points to a future where users interact with LLMs beyond the form of chatbots, and instead designed to support complex user tasks with a tighter integration between natural language and graphical user interfaces.

{{</citation>}}


### (102/119) Experiment on Gender and Racial/Ethnic Bias Against Video Game Streamers: Comparing Perceived Gameplay Skill and Viewer Engagement (David V. Nguyen et al., 2023)

{{<citation>}}

David V. Nguyen, Edward F. Melcer, Deanne Adams. (2023)  
**Experiment on Gender and Racial/Ethnic Bias Against Video Game Streamers: Comparing Perceived Gameplay Skill and Viewer Engagement**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.00610v1)  

---


**ABSTRACT**  
Research suggests there is a perception that females and underrepresented racial/ethnic minorities have worse gameplay skills and produce less engaging video game streaming content. This bias might impact streamers' audience size, viewers' financial patronage of a streamer, streamers' sponsorship offers, etc. However, few studies on this topic use experimental methods. To fill this gap, we conducted a between-subjects survey experiment to examine if viewers are biased against video game streamers based on the streamer's gender or race/ethnicity. 200 survey participants rated the gameplay skill and viewer engagement of an identical gameplay recording. The only change between experimental conditions was the streamer's name who purportedly created the recording. The Dunnett's test found no statistically significant differences in viewer engagement ratings when comparing White male streamers to either White female (p = 0.37), Latino male (p = 0.66), or Asian male (p = 0.09) streamers. Similarly, there were no statistically significant differences in gameplay skill ratings when comparing White male streamers to either White female (p = 0.10), Latino male (p = 1.00), or Asian male (p = 0.59) streamers. Potential contributors to statistically non-significant results and counter-intuitive results (i.e., White females received non-significantly higher ratings than White males) are discussed.

{{</citation>}}


### (103/119) A Spatio-Temporal Graph Convolutional Network for Gesture Recognition from High-Density Electromyography (Wenjuan Zhong et al., 2023)

{{<citation>}}

Wenjuan Zhong, Yuyang Zhang, Peiwen Fu, Wenxuan Xiong, Mingming Zhang. (2023)  
**A Spatio-Temporal Graph Convolutional Network for Gesture Recognition from High-Density Electromyography**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC, eess-SP  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2312.00553v1)  

---


**ABSTRACT**  
Accurate hand gesture prediction is crucial for effective upper-limb prosthetic limbs control. As the high flexibility and multiple degrees of freedom exhibited by human hands, there has been a growing interest in integrating deep networks with high-density surface electromyography (HD-sEMG) grids to enhance gesture recognition capabilities. However, many existing methods fall short in fully exploit the specific spatial topology and temporal dependencies present in HD-sEMG data. Additionally, these studies are often limited number of gestures and lack generality. Hence, this study introduces a novel gesture recognition method, named STGCN-GR, which leverages spatio-temporal graph convolution networks for HD-sEMG-based human-machine interfaces. Firstly, we construct muscle networks based on functional connectivity between channels, creating a graph representation of HD-sEMG recordings. Subsequently, a temporal convolution module is applied to capture the temporal dependences in the HD-sEMG series and a spatial graph convolution module is employed to effectively learn the intrinsic spatial topology information among distinct HD-sEMG channels. We evaluate our proposed model on a public HD-sEMG dataset comprising a substantial number of gestures (i.e., 65). Our results demonstrate the remarkable capability of the STGCN-GR method, achieving an impressive accuracy of 91.07% in predicting gestures, which surpasses state-of-the-art deep learning methods applied to the same dataset.

{{</citation>}}


### (104/119) Generative artificial intelligence enhances individual creativity but reduces the collective diversity of novel content (Anil R. Doshi et al., 2023)

{{<citation>}}

Anil R. Doshi, Oliver P. Hauser. (2023)  
**Generative artificial intelligence enhances individual creativity but reduces the collective diversity of novel content**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC, econ-GN, q-fin-EC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00506v1)  

---


**ABSTRACT**  
Creativity is core to being human. Generative artificial intelligence (GenAI) holds promise for humans to be more creative by offering new ideas, or less creative by anchoring on GenAI ideas. We study the causal impact of GenAI ideas on the production of an unstructured creative output in an online experimental study where some writers could obtain ideas for a story from a GenAI platform. We find that access to GenAI ideas causes stories to be evaluated as more creative, better written and more enjoyable, especially among less creative writers. However, objective measures of story similarity within each condition reveal that GenAI-enabled stories are more similar to each other than stories by humans alone. These results point to an increase in individual creativity, but at the same time there is a risk of losing collective novelty: this dynamic resembles a social dilemma where individual writers are better off using GenAI to improve their own writing, but collectively a narrower scope of novel content may be produced with GenAI. Our results have implications for researchers, policy-makers and practitioners interested in bolstering creativity, but point to potential downstream consequences from over-reliance.

{{</citation>}}


## cs.AI (4)



### (105/119) Deciphering Digital Detectives: Understanding LLM Behaviors and Capabilities in Multi-Agent Mystery Games (Dekun Wu et al., 2023)

{{<citation>}}

Dekun Wu, Haochen Shi, Zhiyuan Sun, Bang Liu. (2023)  
**Deciphering Digital Detectives: Understanding LLM Behaviors and Capabilities in Multi-Agent Mystery Games**  

---
Primary Category: cs.AI  
Categories: I-2-0; I-2-1; I-2-7, cs-AI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.00746v1)  

---


**ABSTRACT**  
In this study, we explore the application of Large Language Models (LLMs) in "Jubensha" (Chinese murder mystery role-playing games), a novel area in AI-driven gaming. We introduce the first Chinese dataset specifically for Jubensha, including character scripts and game rules, to foster AI agent development in this complex narrative environment. Our work also presents a unique multi-agent interaction framework using LLMs, allowing AI agents to autonomously engage in the game, enhancing the dynamics of Jubensha gameplay. To evaluate these AI agents, we developed specialized methods targeting their mastery of case information and reasoning skills. Furthermore, we incorporated the latest advancements in in-context learning to improve the agents' performance in critical aspects like information gathering, murderer detection, and logical reasoning. The experimental results validate the effectiveness of our proposed methods. This work aims to offer a fresh perspective on understanding LLM capabilities and establish a new benchmark for evaluating large language model-based agents to researchers in the field.

{{</citation>}}


### (106/119) Enhancing Explainability in Mobility Data Science through a combination of methods (Georgios Makridis et al., 2023)

{{<citation>}}

Georgios Makridis, Vasileios Koukos, Georgios Fatouros, Dimosthenis Kyriazis. (2023)  
**Enhancing Explainability in Mobility Data Science through a combination of methods**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00380v1)  

---


**ABSTRACT**  
In the domain of Mobility Data Science, the intricate task of interpreting models trained on trajectory data, and elucidating the spatio-temporal movement of entities, has persistently posed significant challenges. Conventional XAI techniques, although brimming with potential, frequently overlook the distinct structure and nuances inherent within trajectory data. Observing this deficiency, we introduced a comprehensive framework that harmonizes pivotal XAI techniques: LIME (Local Interpretable Model-agnostic Explanations), SHAP (SHapley Additive exPlanations), Saliency maps, attention mechanisms, direct trajectory visualization, and Permutation Feature Importance (PFI). Unlike conventional strategies that deploy these methods singularly, our unified approach capitalizes on the collective efficacy of these techniques, yielding deeper and more granular insights for models reliant on trajectory data. In crafting this synthesis, we effectively address the multifaceted essence of trajectories, achieving not only amplified interpretability but also a nuanced, contextually rich comprehension of model decisions. To validate and enhance our framework, we undertook a survey to gauge preferences and reception among various user demographics. Our findings underscored a dichotomy: professionals with academic orientations, particularly those in roles like Data Scientist, IT Expert, and ML Engineer, showcased a profound, technical understanding and often exhibited a predilection for amalgamated methods for interpretability. Conversely, end-users or individuals less acquainted with AI and Data Science showcased simpler inclinations, such as bar plots indicating timestep significance or visual depictions pinpointing pivotal segments of a vessel's trajectory.

{{</citation>}}


### (107/119) Green Edge AI: A Contemporary Survey (Yuyi Mao et al., 2023)

{{<citation>}}

Yuyi Mao, Xianghao Yu, Kaibin Huang, Ying-Jun Angela Zhang, Jun Zhang. (2023)  
**Green Edge AI: A Contemporary Survey**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IT, cs-NI, cs.AI, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00333v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) technologies have emerged as pivotal enablers across a multitude of industries, including consumer electronics, healthcare, and manufacturing, largely due to their resurgence over the past decade. The transformative power of AI is primarily derived from the utilization of deep neural networks (DNNs), which require extensive data for training and substantial computational resources for processing. Consequently, DNN models are typically trained and deployed on resource-rich cloud servers. However, due to potential latency issues associated with cloud communications, deep learning (DL) workflows are increasingly being transitioned to wireless edge networks near end-user devices (EUDs). This shift is designed to support latency-sensitive applications and has given rise to a new paradigm of edge AI, which will play a critical role in upcoming 6G networks to support ubiquitous AI applications. Despite its potential, edge AI faces substantial challenges, mostly due to the dichotomy between the resource limitations of wireless edge networks and the resource-intensive nature of DL. Specifically, the acquisition of large-scale data, as well as the training and inference processes of DNNs, can rapidly deplete the battery energy of EUDs. This necessitates an energy-conscious approach to edge AI to ensure both optimal and sustainable performance. In this paper, we present a contemporary survey on green edge AI. We commence by analyzing the principal energy consumption components of edge AI systems to identify the fundamental design principles of green edge AI. Guided by these principles, we then explore energy-efficient design methodologies for the three critical tasks in edge AI systems, including training data acquisition, edge training, and edge inference. Finally, we underscore potential future research directions to further enhance the energy efficiency of edge AI.

{{</citation>}}


### (108/119) Agent-OM: Leveraging Large Language Models for Ontology Matching (Zhangcheng Qiang et al., 2023)

{{<citation>}}

Zhangcheng Qiang, Weiqing Wang, Kerry Taylor. (2023)  
**Agent-OM: Leveraging Large Language Models for Ontology Matching**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-IR, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00326v1)  

---


**ABSTRACT**  
Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM-based agents have become revolutionary in data engineering and have been applied creatively in various domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With thoughtful consideration of several specific challenges to leverage LLMs for OM, we propose a generic framework, namely Agent-OM, consisting of two Siamese agents for retrieval and matching, with a set of simple prompt-based OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve very close results to the best long-standing performance on simple OM tasks and significantly improve the performance on complex and few-shot OM tasks.

{{</citation>}}


## eess.IV (2)



### (109/119) Unsupervised Adaptive Implicit Neural Representation Learning for Scan-Specific MRI Reconstruction (Junwei Yang et al., 2023)

{{<citation>}}

Junwei Yang, Pietro Liò. (2023)  
**Unsupervised Adaptive Implicit Neural Representation Learning for Scan-Specific MRI Reconstruction**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.00677v1)  

---


**ABSTRACT**  
In recent studies on MRI reconstruction, advances have shown significant promise for further accelerating the MRI acquisition. Most state-of-the-art methods require a large amount of fully-sampled data to optimise reconstruction models, which is impractical and expensive under certain clinical settings. On the other hand, for unsupervised scan-specific reconstruction methods, overfitting is likely to happen due to insufficient supervision, while restrictions on acceleration rates and under-sampling patterns further limit their applicability. To this end, we propose an unsupervised, adaptive coarse-to-fine framework that enhances reconstruction quality without being constrained by the sparsity levels or patterns in under-sampling. The framework employs an implicit neural representation for scan-specific MRI reconstruction, learning a mapping from multi-dimensional coordinates to their corresponding signal intensities. Moreover, we integrate a novel learning strategy that progressively refines the use of acquired k-space signals for self-supervision. This approach effectively adjusts the proportion of supervising signals from unevenly distributed information across different frequency bands, thus mitigating the issue of overfitting while improving the overall reconstruction. Comprehensive evaluation on a public dataset, including both 2D and 3D data, has shown that our method outperforms current state-of-the-art scan-specific MRI reconstruction techniques, for up to 8-fold under-sampling.

{{</citation>}}


### (110/119) A Recent Survey of Vision Transformers for Medical Image Segmentation (Asifullah Khan et al., 2023)

{{<citation>}}

Asifullah Khan, Zunaira Rauf, Abdul Rehman Khan, Saima Rathore, Saddam Hussain Khan, Sahar Shah, Umair Farooq, Hifsa Asif, Aqsa Asif, Umme Zahoora, Rafi Ullah Khalil, Suleman Qamar, Umme Hani Asif, Faiza Babar Khan, Abdul Majid, Jeonghwan Gwak. (2023)  
**A Recent Survey of Vision Transformers for Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.00634v1)  

---


**ABSTRACT**  
Medical image segmentation plays a crucial role in various healthcare applications, enabling accurate diagnosis, treatment planning, and disease monitoring. In recent years, Vision Transformers (ViTs) have emerged as a promising technique for addressing the challenges in medical image segmentation. In medical images, structures are usually highly interconnected and globally distributed. ViTs utilize their multi-scale attention mechanism to model the long-range relationships in the images. However, they do lack image-related inductive bias and translational invariance, potentially impacting their performance. Recently, researchers have come up with various ViT-based approaches that incorporate CNNs in their architectures, known as Hybrid Vision Transformers (HVTs) to capture local correlation in addition to the global information in the images. This survey paper provides a detailed review of the recent advancements in ViTs and HVTs for medical image segmentation. Along with the categorization of ViT and HVT-based medical image segmentation approaches we also present a detailed overview of their real-time applications in several medical image modalities. This survey may serve as a valuable resource for researchers, healthcare practitioners, and students in understanding the state-of-the-art approaches for ViT-based medical image segmentation.

{{</citation>}}


## eess.SP (1)



### (111/119) RIS-Based On-the-Air Semantic Communications -- a Diffractional Deep Neural Network Approach (Shuyi Chen et al., 2023)

{{<citation>}}

Shuyi Chen, Yingzhe Hui, Yifan Qin, Yueyi Yuan, Weixiao Meng, Xuewen Luo, Hsiao-Hwa Chen. (2023)  
**RIS-Based On-the-Air Semantic Communications -- a Diffractional Deep Neural Network Approach**  

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00535v1)  

---


**ABSTRACT**  
Semantic communication has gained significant attention recently due to its advantages in achieving higher transmission efficiency by focusing on semantic information instead of bit-level information. However, current AI-based semantic communication methods require digital hardware for implementation. With the rapid advancement on reconfigurable intelligence surfaces (RISs), a new approach called on-the-air diffractional deep neural networks (D$^2$NN) can be utilized to enable semantic communications on the wave domain. This paper proposes a new paradigm of RIS-based on-the-air semantic communications, where the computational process occurs inherently as wireless signals pass through RISs. We present the system model and discuss the data and control flows of this scheme, followed by a performance analysis using image transmission as an example. In comparison to traditional hardware-based approaches, RIS-based semantic communications offer appealing features, such as light-speed computation, low computational power requirements, and the ability to handle multiple tasks simultaneously.

{{</citation>}}


## cs.PL (1)



### (112/119) VEXIR2Vec: An Architecture-Neutral Embedding Framework for Binary Similarity (S. VenkataKeerthy et al., 2023)

{{<citation>}}

S. VenkataKeerthy, Yashas Andaluri, Sayan Dey, Soumya Banerjee, Ramakrishna Upadrasta. (2023)  
**VEXIR2Vec: An Architecture-Neutral Embedding Framework for Binary Similarity**  

---
Primary Category: cs.PL  
Categories: cs-CR, cs-LG, cs-PL, cs.PL  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.00507v1)  

---


**ABSTRACT**  
We propose VEXIR2Vec, a code embedding framework for finding similar functions in binaries. Our representations rely on VEX IR, the intermediate representation used by binary analysis tools like Valgrind and angr. Our proposed embeddings encode both syntactic and semantic information to represent a function, and is both application and architecture independent. We also propose POV, a custom Peephole Optimization engine that normalizes the VEX IR for effective similarity analysis. We design several optimizations like copy/constant propagation, constant folding, common subexpression elimination and load-store elimination in POV.   We evaluate our framework on two experiments -- diffing and searching -- involving binaries targeting different architectures, compiled using different compilers and versions, optimization sequences, and obfuscations. We show results on several standard projects and on real-world vulnerabilities. Our results show that VEXIR2Vec achieves superior precision and recall values compared to the state-of-the-art works. Our framework is highly scalable and is built as a multi-threaded, parallel library by only using open-source tools. VEXIR2Vec achieves about $3.2 \times$ speedup on the closest competitor, and orders-of-magnitude speedup on other tools.

{{</citation>}}


## math.NA (1)



### (113/119) Optimal complexity of goal-oriented adaptive FEM for nonsymmetric linear elliptic PDEs (Philipp Bringmann et al., 2023)

{{<citation>}}

Philipp Bringmann, Maximilian Brunner, Dirk Praetorius, Julian Streitberger. (2023)  
**Optimal complexity of goal-oriented adaptive FEM for nonsymmetric linear elliptic PDEs**  

---
Primary Category: math.NA  
Categories: cs-NA, math-NA, math.NA  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00489v1)  

---


**ABSTRACT**  
We analyze a goal-oriented adaptive algorithm that aims to efficiently compute the quantity of interest $G(u^\star)$ with a linear goal functional $G$ and the solution $u^\star$ to a general second-order nonsymmetric linear elliptic partial differential equation. The current state of the analysis of iterative algebraic solvers for nonsymmetric systems lacks the contraction property in the norms that are prescribed by the functional analytic setting. This seemingly prevents their application in the optimality analysis of goal-oriented adaptivity. As a remedy, this paper proposes a goal-oriented adaptive iteratively symmetrized finite element method (GOAISFEM). It employs a nested loop with a contractive symmetrization procedure, e.g., the Zarantonello iteration, and a contractive algebraic solver, e.g., an optimal multigrid solver. The various iterative procedures require well-designed stopping criteria such that the adaptive algorithm can effectively steer the local mesh refinement and the computation of the inexact discrete approximations. The main results consist of full linear convergence of the proposed adaptive algorithm and the proof of optimal convergence rates with respect to both degrees of freedom and total computational cost (i.e., optimal complexity). Numerical experiments confirm the theoretical results and investigate the selection of the parameters.

{{</citation>}}


## cs.SD (1)



### (114/119) Self-Supervised Learning of Spatial Acoustic Representation with Cross-Channel Signal Reconstruction and Multi-Channel Conformer (Bing Yang et al., 2023)

{{<citation>}}

Bing Yang, Xiaofei Li. (2023)  
**Self-Supervised Learning of Spatial Acoustic Representation with Cross-Channel Signal Reconstruction and Multi-Channel Conformer**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.00476v1)  

---


**ABSTRACT**  
Supervised learning methods have shown effectiveness in estimating spatial acoustic parameters such as time difference of arrival, direct-to-reverberant ratio and reverberation time. However, they still suffer from the simulation-to-reality generalization problem due to the mismatch between simulated and real-world acoustic characteristics and the deficiency of annotated real-world data. To this end, this work proposes a self-supervised method that takes full advantage of unlabeled data for spatial acoustic parameter estimation. First, a new pretext task, i.e. cross-channel signal reconstruction (CCSR), is designed to learn a universal spatial acoustic representation from unlabeled multi-channel microphone signals. We mask partial signals of one channel and ask the model to reconstruct them, which makes it possible to learn spatial acoustic information from unmasked signals and extract source information from the other microphone channel. An encoder-decoder structure is used to disentangle the two kinds of information. By fine-tuning the pre-trained spatial encoder with a small annotated dataset, this encoder can be used to estimate spatial acoustic parameters. Second, a novel multi-channel audio Conformer (MC-Conformer) is adopted as the encoder model architecture, which is suitable for both the pretext and downstream tasks. It is carefully designed to be able to capture the local and global characteristics of spatial acoustics exhibited in the time-frequency domain. Experimental results of five acoustic parameter estimation tasks on both simulated and real-world data show the effectiveness of the proposed method. To the best of our knowledge, this is the first self-supervised learning method in the field of spatial acoustic representation learning and multi-channel audio signal processing.

{{</citation>}}


## quant-ph (3)



### (115/119) Impact of Data Augmentation on QCNNs (Leting Zhouli et al., 2023)

{{<citation>}}

Leting Zhouli, Peiyong Wang, Udaya Parampalli. (2023)  
**Impact of Data Augmentation on QCNNs**  

---
Primary Category: quant-ph  
Categories: cs-CV, cs-LG, quant-ph, quant-ph  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2312.00358v1)  

---


**ABSTRACT**  
In recent years, Classical Convolutional Neural Networks (CNNs) have been applied for image recognition successfully. Quantum Convolutional Neural Networks (QCNNs) are proposed as a novel generalization to CNNs by using quantum mechanisms. The quantum mechanisms lead to an efficient training process in QCNNs by reducing the size of input from $N$ to $log_2N$. This paper implements and compares both CNNs and QCNNs by testing losses and prediction accuracy on three commonly used datasets. The datasets include the MNIST hand-written digits, Fashion MNIST and cat/dog face images. Additionally, data augmentation (DA), a technique commonly used in CNNs to improve the performance of classification by generating similar images based on original inputs, is also implemented in QCNNs. Surprisingly, the results showed that data augmentation didn't improve QCNNs performance. The reasons and logic behind this result are discussed, hoping to expand our understanding of Quantum machine learning theory.

{{</citation>}}


### (116/119) Quantum Kernel t-Distributed Stochastic Neighbor Embedding (Yoshiaki Kawase et al., 2023)

{{<citation>}}

Yoshiaki Kawase, Kosuke Mitarai, Keisuke Fujii. (2023)  
**Quantum Kernel t-Distributed Stochastic Neighbor Embedding**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.00352v1)  

---


**ABSTRACT**  
Data visualization is important in understanding the characteristics of data that are difficult to see directly. It is used to visualize loss landscapes and optimization trajectories to analyze optimization performance. Popular optimization analysis is performed by visualizing a loss landscape around the reached local or global minimum using principal component analysis. However, this visualization depends on the variational parameters of a quantum circuit rather than quantum states, which makes it difficult to understand the mechanism of optimization process through the property of quantum states. Here, we propose a quantum data visualization method using quantum kernels, which enables us to offer fast and highly accurate visualization of quantum states. In our numerical experiments, we visualize hand-written digits dataset and apply $k$-nearest neighbor algorithm to the low-dimensional data to quantitatively evaluate our proposed method compared with a classical kernel method. As a result, our proposed method achieves comparable accuracy to the state-of-the-art classical kernel method, meaning that the proposed visualization method based on quantum machine learning does not degrade the separability of the input higher dimensional data. Furthermore, we visualize the optimization trajectories of finding the ground states of transverse field Ising model and successfully find the trajectory characteristics. Since quantum states are higher dimensional objects that can only be seen via observables, our visualization method, which inherits the similarity of quantum data, would be useful in understanding the behavior of quantum circuits and algorithms.

{{</citation>}}


### (117/119) Skipper: Improving the Reach and Fidelity of Quantum Annealers by Skipping Long Chains (Ramin Ayanzadeh et al., 2023)

{{<citation>}}

Ramin Ayanzadeh, Moinuddin Qureshi. (2023)  
**Skipper: Improving the Reach and Fidelity of Quantum Annealers by Skipping Long Chains**  

---
Primary Category: quant-ph  
Categories: cs-AI, cs-AR, cs-ET, cs-PF, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.00264v1)  

---


**ABSTRACT**  
Quantum Annealers (QAs) operate as single-instruction machines, lacking a SWAP operation to overcome limited qubit connectivity. Consequently, multiple physical qubits are chained to form a program qubit with higher connectivity, resulting in a drastically diminished effective QA capacity by up to 33x. We observe that in QAs: (a) chain lengths exhibit a power-law distribution, a few dominant chains holding substantially more qubits than others; and (b) about 25% of physical qubits remain unused, getting isolated between these chains. We propose Skipper, a software technique that enhances the capacity and fidelity of QAs by skipping dominant chains and substituting their program qubit with two readout results. Using a 5761-qubit QA, we demonstrate that Skipper can tackle up to 59% (Avg. 28%) larger problems when eleven chains are skipped. Additionally, Skipper can improve QA fidelity by up to 44% (Avg. 33%) when cutting five chains (32 runs). Users can specify up to eleven chain cuts in Skipper, necessitating about 2,000 distinct quantum executable runs. To mitigate this, we introduce Skipper-G, a greedy scheme that skips sub-problems less likely to hold the global optimum, executing a maximum of 23 quantum executables with eleven chain trims. Skipper-G can boost QA fidelity by up to 41% (Avg. 29%) when cutting five chains (11 runs).

{{</citation>}}


## q-bio.QM (1)



### (118/119) ESM-NBR: fast and accurate nucleic acid-binding residue prediction via protein language model feature representation and multi-task learning (Wenwu Zeng et al., 2023)

{{<citation>}}

Wenwu Zeng, Dafeng Lv, Wenjuan Liu, Shaoliang Peng. (2023)  
**ESM-NBR: fast and accurate nucleic acid-binding residue prediction via protein language model feature representation and multi-task learning**  

---
Primary Category: q-bio.QM  
Categories: cs-LG, q-bio-QM, q-bio.QM  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.00842v1)  

---


**ABSTRACT**  
Protein-nucleic acid interactions play a very important role in a variety of biological activities. Accurate identification of nucleic acid-binding residues is a critical step in understanding the interaction mechanisms. Although many computationally based methods have been developed to predict nucleic acid-binding residues, challenges remain. In this study, a fast and accurate sequence-based method, called ESM-NBR, is proposed. In ESM-NBR, we first use the large protein language model ESM2 to extract discriminative biological properties feature representation from protein primary sequences; then, a multi-task deep learning model composed of stacked bidirectional long short-term memory (BiLSTM) and multi-layer perceptron (MLP) networks is employed to explore common and private information of DNA- and RNA-binding residues with ESM2 feature as input. Experimental results on benchmark data sets demonstrate that the prediction performance of ESM2 feature representation comprehensively outperforms evolutionary information-based hidden Markov model (HMM) features. Meanwhile, the ESM-NBR obtains the MCC values for DNA-binding residues prediction of 0.427 and 0.391 on two independent test sets, which are 18.61 and 10.45% higher than those of the second-best methods, respectively. Moreover, by completely discarding the time-cost multiple sequence alignment process, the prediction speed of ESM-NBR far exceeds that of existing methods (5.52s for a protein sequence of length 500, which is about 16 times faster than the second-fastest method). A user-friendly standalone package and the data of ESM-NBR are freely available for academic use at: https://github.com/wwzll123/ESM-NBR.

{{</citation>}}


## astro-ph.IM (1)



### (119/119) RadioGalaxyNET: Dataset and Novel Computer Vision Algorithms for the Detection of Extended Radio Galaxies and Infrared Hosts (Nikhel Gupta et al., 2023)

{{<citation>}}

Nikhel Gupta, Zeeshan Hayder, Ray P. Norris, Minh Huynh, Lars Petersson. (2023)  
**RadioGalaxyNET: Dataset and Novel Computer Vision Algorithms for the Detection of Extended Radio Galaxies and Infrared Hosts**  

---
Primary Category: astro-ph.IM  
Categories: astro-ph-CO, astro-ph-GA, astro-ph-IM, astro-ph.IM, cs-CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.00306v1)  

---


**ABSTRACT**  
Creating radio galaxy catalogues from next-generation deep surveys requires automated identification of associated components of extended sources and their corresponding infrared hosts. In this paper, we introduce RadioGalaxyNET, a multimodal dataset, and a suite of novel computer vision algorithms designed to automate the detection and localization of multi-component extended radio galaxies and their corresponding infrared hosts. The dataset comprises 4,155 instances of galaxies in 2,800 images with both radio and infrared channels. Each instance provides information about the extended radio galaxy class, its corresponding bounding box encompassing all components, the pixel-level segmentation mask, and the keypoint position of its corresponding infrared host galaxy. RadioGalaxyNET is the first dataset to include images from the highly sensitive Australian Square Kilometre Array Pathfinder (ASKAP) radio telescope, corresponding infrared images, and instance-level annotations for galaxy detection. We benchmark several object detection algorithms on the dataset and propose a novel multimodal approach to simultaneously detect radio galaxies and the positions of infrared hosts.

{{</citation>}}
