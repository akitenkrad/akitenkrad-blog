---
draft: false
title: "arXiv @ 2023.11.11"
date: 2023-11-11
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.11"
    identifier: arxiv_20231111
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (24)](#cslg-24)
- [cs.CV (26)](#cscv-26)
- [cs.DC (1)](#csdc-1)
- [eess.SY (1)](#eesssy-1)
- [cs.CL (25)](#cscl-25)
- [cs.CY (2)](#cscy-2)
- [cs.NI (3)](#csni-3)
- [cs.CR (4)](#cscr-4)
- [cs.SI (2)](#cssi-2)
- [cs.SD (1)](#cssd-1)
- [cs.HC (2)](#cshc-2)
- [quant-ph (2)](#quant-ph-2)
- [stat.ML (1)](#statml-1)
- [cs.AI (4)](#csai-4)
- [eess.IV (2)](#eessiv-2)
- [cs.CC (1)](#cscc-1)
- [cs.RO (1)](#csro-1)
- [cs.SE (1)](#csse-1)
- [eess.AS (1)](#eessas-1)
- [cs.NE (2)](#csne-2)
- [math.OC (1)](#mathoc-1)
- [cs.IT (1)](#csit-1)

## cs.LG (24)



### (1/108) The Paradox of Noise: An Empirical Study of Noise-Infusion Mechanisms to Improve Generalization, Stability, and Privacy in Federated Learning (Elaheh Jafarigol et al., 2023)

{{<citation>}}

Elaheh Jafarigol, Theodore Trafalis. (2023)  
**The Paradox of Noise: An Empirical Study of Noise-Infusion Mechanisms to Improve Generalization, Stability, and Privacy in Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05790v1)  

---


**ABSTRACT**  
In a data-centric era, concerns regarding privacy and ethical data handling grow as machine learning relies more on personal information. This empirical study investigates the privacy, generalization, and stability of deep learning models in the presence of additive noise in federated learning frameworks. Our main objective is to provide strategies to measure the generalization, stability, and privacy-preserving capabilities of these models and further improve them. To this end, five noise infusion mechanisms at varying noise levels within centralized and federated learning settings are explored. As model complexity is a key component of the generalization and stability of deep learning models during training and evaluation, a comparative analysis of three Convolutional Neural Network (CNN) architectures is provided. The paper introduces Signal-to-Noise Ratio (SNR) as a quantitative measure of the trade-off between privacy and training accuracy of noise-infused models, aiming to find the noise level that yields optimal privacy and accuracy. Moreover, the Price of Stability and Price of Anarchy are defined in the context of privacy-preserving deep learning, contributing to the systematic investigation of the noise infusion strategies to enhance privacy without compromising performance. Our research sheds light on the delicate balance between these critical factors, fostering a deeper understanding of the implications of noise-based regularization in machine learning. By leveraging noise as a tool for regularization and privacy enhancement, we aim to contribute to the development of robust, privacy-aware algorithms, ensuring that AI-driven solutions prioritize both utility and privacy.

{{</citation>}}


### (2/108) Dirichlet Energy Enhancement of Graph Neural Networks by Framelet Augmentation (Jialin Chen et al., 2023)

{{<citation>}}

Jialin Chen, Yuelin Wang, Cristian Bodnar, Rex Ying, Pietro Lio, Yu Guang Wang. (2023)  
**Dirichlet Energy Enhancement of Graph Neural Networks by Framelet Augmentation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Augmentation, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.05767v1)  

---


**ABSTRACT**  
Graph convolutions have been a pivotal element in learning graph representations. However, recursively aggregating neighboring information with graph convolutions leads to indistinguishable node features in deep layers, which is known as the over-smoothing issue. The performance of graph neural networks decays fast as the number of stacked layers increases, and the Dirichlet energy associated with the graph decreases to zero as well. In this work, we introduce a framelet system into the analysis of Dirichlet energy and take a multi-scale perspective to leverage the Dirichlet energy and alleviate the over-smoothing issue. Specifically, we develop a Framelet Augmentation strategy by adjusting the update rules with positive and negative increments for low-pass and high-passes respectively. Based on that, we design the Energy Enhanced Convolution (EEConv), which is an effective and practical operation that is proved to strictly enhance Dirichlet energy. From a message-passing perspective, EEConv inherits multi-hop aggregation property from the framelet transform and takes into account all hops in the multi-scale representation, which benefits the node classification tasks over heterophilous graphs. Experiments show that deep GNNs with EEConv achieve state-of-the-art performance over various node classification datasets, especially for heterophilous graphs, while also lifting the Dirichlet energy as the network goes deeper.

{{</citation>}}


### (3/108) Generative Explanations for Graph Neural Network: Methods and Evaluations (Jialin Chen et al., 2023)

{{<citation>}}

Jialin Chen, Kenza Amara, Junchi Yu, Rex Ying. (2023)  
**Generative Explanations for Graph Neural Network: Methods and Evaluations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.05764v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) achieve state-of-the-art performance in various graph-related tasks. However, the black-box nature often limits their interpretability and trustworthiness. Numerous explainability methods have been proposed to uncover the decision-making logic of GNNs, by generating underlying explanatory substructures. In this paper, we conduct a comprehensive review of the existing explanation methods for GNNs from the perspective of graph generation. Specifically, we propose a unified optimization objective for generative explanation methods, comprising two sub-objectives: Attribution and Information constraints. We further demonstrate their specific manifestations in various generative model architectures and different explanation scenarios. With the unified objective of the explanation problem, we reveal the shared characteristics and distinctions among current methods, laying the foundation for future methodological advancements. Empirical results demonstrate the advantages and limitations of different explainability approaches in terms of explanation performance, efficiency, and generalizability.

{{</citation>}}


### (4/108) Verilog-to-PyG -- A Framework for Graph Learning and Augmentation on RTL Designs (Yingjie Li et al., 2023)

{{<citation>}}

Yingjie Li, Mingju Liu, Alan Mishchenko, Cunxi Yu. (2023)  
**Verilog-to-PyG -- A Framework for Graph Learning and Augmentation on RTL Designs**  

---
Primary Category: cs.LG  
Categories: cs-AR, cs-LG, cs-LO, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.05722v1)  

---


**ABSTRACT**  
The complexity of modern hardware designs necessitates advanced methodologies for optimizing and analyzing modern digital systems. In recent times, machine learning (ML) methodologies have emerged as potent instruments for assessing design quality-of-results at the Register-Transfer Level (RTL) or Boolean level, aiming to expedite design exploration of advanced RTL configurations. In this presentation, we introduce an innovative open-source framework that translates RTL designs into graph representation foundations, which can be seamlessly integrated with the PyTorch Geometric graph learning platform. Furthermore, the Verilog-to-PyG (V2PYG) framework is compatible with the open-source Electronic Design Automation (EDA) toolchain OpenROAD, facilitating the collection of labeled datasets in an utterly open-source manner. Additionally, we will present novel RTL data augmentation methods (incorporated in our framework) that enable functional equivalent design augmentation for the construction of an extensive graph-based RTL design database. Lastly, we will showcase several using cases of V2PYG with detailed scripting examples. V2PYG can be found at \url{https://yu-maryland.github.io/Verilog-to-PyG/}.

{{</citation>}}


### (5/108) Efficient Parallelization Layouts for Large-Scale Distributed Model Training (Johannes Hagemann et al., 2023)

{{<citation>}}

Johannes Hagemann, Samuel Weinbach, Konstantin Dobler, Maximilian Schall, Gerard de Melo. (2023)  
**Efficient Parallelization Layouts for Large-Scale Distributed Model Training**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.05610v1)  

---


**ABSTRACT**  
Efficiently training large language models requires parallelizing across hundreds of hardware accelerators and invoking various compute and memory optimizations. When combined, many of these strategies have complex interactions regarding the final training efficiency. Prior work tackling this problem did not have access to the latest set of optimizations, such as FlashAttention or sequence parallelism. In this work, we conduct a comprehensive ablation study of possible training configurations for large language models. We distill this large study into several key recommendations for the most efficient training. For instance, we find that using a micro-batch size of 1 usually enables the most efficient training layouts. Larger micro-batch sizes necessitate activation checkpointing or higher degrees of model parallelism and also lead to larger pipeline bubbles. Our most efficient configurations enable us to achieve state-of-the-art training efficiency results over a range of model sizes, most notably a Model FLOPs utilization of 70.5% when training a 13B model.

{{</citation>}}


### (6/108) LLM Augmented Hierarchical Agents (Bharat Prakash et al., 2023)

{{<citation>}}

Bharat Prakash, Tim Oates, Tinoosh Mohsenin. (2023)  
**LLM Augmented Hierarchical Agents**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.05596v1)  

---


**ABSTRACT**  
Solving long-horizon, temporally-extended tasks using Reinforcement Learning (RL) is challenging, compounded by the common practice of learning without prior knowledge (or tabula rasa learning). Humans can generate and execute plans with temporally-extended actions and quickly learn to perform new tasks because we almost never solve problems from scratch. We want autonomous agents to have this same ability. Recently, LLMs have been shown to encode a tremendous amount of knowledge about the world and to perform impressive in-context learning and reasoning. However, using LLMs to solve real world problems is hard because they are not grounded in the current task. In this paper we exploit the planning capabilities of LLMs while using RL to provide learning from the environment, resulting in a hierarchical agent that uses LLMs to solve long-horizon tasks. Instead of completely relying on LLMs, they guide a high-level policy, making learning significantly more sample efficient. This approach is evaluated in simulation environments such as MiniGrid, SkillHack, and Crafter, and on a real robot arm in block manipulation tasks. We show that agents trained using our approach outperform other baselines methods and, once trained, don't need access to LLMs during deployment.

{{</citation>}}


### (7/108) Bayesian Methods for Media Mix Modelling with shape and funnel effects (Javier Marin, 2023)

{{<citation>}}

Javier Marin. (2023)  
**Bayesian Methods for Media Mix Modelling with shape and funnel effects**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05587v1)  

---


**ABSTRACT**  
In recent years, significant progress in generative AI has highlighted the important role of physics-inspired models that utilize advanced mathematical concepts based on fundamental physics principles to enhance artificial intelligence capabilities. Among these models, those based on diffusion equations have greatly improved image quality. This study aims to explore the potential uses of Maxwell-Boltzmann equation, which forms the basis of the kinetic theory of gases, and the Michaelis-Menten model in Marketing Mix Modelling (MMM) applications. We propose incorporating these equations into Hierarchical Bayesian models to analyse consumer behaviour in the context of advertising. These equation sets excel in accurately describing the random dynamics in complex systems like social interactions and consumer-advertising interactions.

{{</citation>}}


### (8/108) Zero-Shot Goal-Directed Dialogue via RL on Imagined Conversations (Joey Hong et al., 2023)

{{<citation>}}

Joey Hong, Sergey Levine, Anca Dragan. (2023)  
**Zero-Shot Goal-Directed Dialogue via RL on Imagined Conversations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Dialog, Dialogue, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.05584v1)  

---


**ABSTRACT**  
Large language models (LLMs) have emerged as powerful and general solutions to many natural language tasks. However, many of the most important applications of language generation are interactive, where an agent has to talk to a person to reach a desired outcome. For example, a teacher might try to understand their student's current comprehension level to tailor their instruction accordingly, and a travel agent might ask questions of their customer to understand their preferences in order to recommend activities they might enjoy. LLMs trained with supervised fine-tuning or "single-step" RL, as with standard RLHF, might struggle which tasks that require such goal-directed behavior, since they are not trained to optimize for overall conversational outcomes after multiple turns of interaction. In this work, we explore a new method for adapting LLMs with RL for such goal-directed dialogue. Our key insight is that, though LLMs might not effectively solve goal-directed dialogue tasks out of the box, they can provide useful data for solving such tasks by simulating suboptimal but human-like behaviors. Given a textual description of a goal-directed dialogue task, we leverage LLMs to sample diverse synthetic rollouts of hypothetical in-domain human-human interactions. Our algorithm then utilizes this dataset with offline reinforcement learning to train an interactive conversational agent that can optimize goal-directed objectives over multiple turns. In effect, the LLM produces examples of possible interactions, and RL then processes these examples to learn to perform more optimal interactions. Empirically, we show that our proposed approach achieves state-of-the-art performance in various goal-directed dialogue tasks that include teaching and preference elicitation.

{{</citation>}}


### (9/108) Exploiting Neural-Network Statistics for Low-Power DNN Inference (Lennart Bamberg et al., 2023)

{{<citation>}}

Lennart Bamberg, Ardalan Najafi, Alberto Garcia-Ortiz. (2023)  
**Exploiting Neural-Network Statistics for Low-Power DNN Inference**  

---
Primary Category: cs.LG  
Categories: cs-AR, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05557v1)  

---


**ABSTRACT**  
Specialized compute blocks have been developed for efficient DNN execution. However, due to the vast amount of data and parameter movements, the interconnects and on-chip memories form another bottleneck, impairing power and performance. This work addresses this bottleneck by contributing a low-power technique for edge-AI inference engines that combines overhead-free coding with a statistical analysis of the data and parameters of neural networks. Our approach reduces the interconnect and memory power consumption by up to 80% for state-of-the-art benchmarks while providing additional power savings for the compute blocks by up to 39%. These power improvements are achieved with no loss of accuracy and negligible hardware cost.

{{</citation>}}


### (10/108) Embedding Space Interpolation Beyond Mini-Batch, Beyond Pairs and Beyond Examples (Shashanka Venkataramanan et al., 2023)

{{<citation>}}

Shashanka Venkataramanan, Ewa Kijak, Laurent Amsaleg, Yannis Avrithis. (2023)  
**Embedding Space Interpolation Beyond Mini-Batch, Beyond Pairs and Beyond Examples**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.05538v1)  

---


**ABSTRACT**  
Mixup refers to interpolation-based data augmentation, originally motivated as a way to go beyond empirical risk minimization (ERM). Its extensions mostly focus on the definition of interpolation and the space (input or feature) where it takes place, while the augmentation process itself is less studied. In most methods, the number of generated examples is limited to the mini-batch size and the number of examples being interpolated is limited to two (pairs), in the input space.   We make progress in this direction by introducing MultiMix, which generates an arbitrarily large number of interpolated examples beyond the mini-batch size and interpolates the entire mini-batch in the embedding space. Effectively, we sample on the entire convex hull of the mini-batch rather than along linear segments between pairs of examples.   On sequence data, we further extend to Dense MultiMix. We densely interpolate features and target labels at each spatial location and also apply the loss densely. To mitigate the lack of dense labels, we inherit labels from examples and weight interpolation factors by attention as a measure of confidence.   Overall, we increase the number of loss terms per mini-batch by orders of magnitude at little additional cost. This is only possible because of interpolating in the embedding space. We empirically show that our solutions yield significant improvement over state-of-the-art mixup methods on four different benchmarks, despite interpolation being only linear. By analyzing the embedding space, we show that the classes are more tightly clustered and uniformly spread over the embedding space, thereby explaining the improved behavior.

{{</citation>}}


### (11/108) Anytime-Constrained Reinforcement Learning (Jeremy McMahan et al., 2023)

{{<citation>}}

Jeremy McMahan, Xiaojin Zhu. (2023)  
**Anytime-Constrained Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DS, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.05511v1)  

---


**ABSTRACT**  
We introduce and study constrained Markov Decision Processes (cMDPs) with anytime constraints. An anytime constraint requires the agent to never violate its budget at any point in time, almost surely. Although Markovian policies are no longer sufficient, we show that there exist optimal deterministic policies augmented with cumulative costs. In fact, we present a fixed-parameter tractable reduction from anytime-constrained cMDPs to unconstrained MDPs. Our reduction yields planning and learning algorithms that are time and sample-efficient for tabular cMDPs so long as the precision of the costs is logarithmic in the size of the cMDP. However, we also show that computing non-trivial approximately optimal policies is NP-hard in general. To circumvent this bottleneck, we design provable approximation algorithms that efficiently compute or learn an approximately feasible policy with optimal value so long as the maximum supported cost is bounded by a polynomial in the cMDP or by the absolute budget. Given our hardness results, our approximation guarantees are the best possible in terms of tractability under worst-case analysis.

{{</citation>}}


### (12/108) Diffusion Based Causal Representation Learning (Amir Mohammad Karimi Mamaghan et al., 2023)

{{<citation>}}

Amir Mohammad Karimi Mamaghan, Andrea Dittadi, Stefan Bauer, Karl Henrik Johansson, Francesco Quinzan. (2023)  
**Diffusion Based Causal Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ME  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.05421v1)  

---


**ABSTRACT**  
Causal reasoning can be considered a cornerstone of intelligent systems. Having access to an underlying causal graph comes with the promise of cause-effect estimation and the identification of efficient and safe interventions. However, learning causal representations remains a major challenge, due to the complexity of many real-world systems. Previous works on causal representation learning have mostly focused on Variational Auto-Encoders (VAE). These methods only provide representations from a point estimate, and they are unsuitable to handle high dimensions. To overcome these problems, we proposed a new Diffusion-based Causal Representation Learning (DCRL) algorithm. This algorithm uses diffusion-based representations for causal discovery. DCRL offers access to infinite dimensional latent codes, which encode different levels of information in the latent code. In a first proof of principle, we investigate the use of DCRL for causal representation learning. We further demonstrate experimentally that this approach performs comparably well in identifying the causal structure and causal variables.

{{</citation>}}


### (13/108) Generalization in medical AI: a perspective on developing scalable models (Joachim A. Behar et al., 2023)

{{<citation>}}

Joachim A. Behar, Jeremy Levy, Leo Anthony Celi. (2023)  
**Generalization in medical AI: a perspective on developing scalable models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05418v1)  

---


**ABSTRACT**  
Over the past few years, research has witnessed the advancement of deep learning models trained on large datasets, some even encompassing millions of examples. While these impressive performance on their hidden test sets, they often underperform when assessed on external datasets. Recognizing the critical role of generalization in medical AI development, many prestigious journals now require reporting results both on the local hidden test set as well as on external datasets before considering a study for publication. Effectively, the field of medical AI has transitioned from the traditional usage of a single dataset that is split into train and test to a more comprehensive framework using multiple datasets, some of which are used for model development (source domain) and others for testing (target domains). However, this new experimental setting does not necessarily resolve the challenge of generalization. This is because of the variability encountered in intended use and specificities across hospital cultures making the idea of universally generalizable systems a myth. On the other hand, the systematic, and a fortiori recurrent re-calibration, of models at the individual hospital level, although ideal, may be overoptimistic given the legal, regulatory and technical challenges that are involved. Re-calibration using transfer learning may not even be possible in some instances where reference labels of target domains are not available. In this perspective we establish a hierarchical three-level scale system reflecting the generalization level of a medical AI algorithm. This scale better reflects the diversity of real-world medical scenarios per which target domain data for re-calibration of models may or not be available and if it is, may or not have reference labels systematically available.

{{</citation>}}


### (14/108) RepQ: Generalizing Quantization-Aware Training for Re-Parametrized Architectures (Anastasiia Prutianova et al., 2023)

{{<citation>}}

Anastasiia Prutianova, Alexey Zaytsev, Chung-Kuei Lee, Fengyu Sun, Ivan Koryakovskiy. (2023)  
**RepQ: Generalizing Quantization-Aware Training for Re-Parametrized Architectures**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2311.05317v1)  

---


**ABSTRACT**  
Existing neural networks are memory-consuming and computationally intensive, making deploying them challenging in resource-constrained environments. However, there are various methods to improve their efficiency. Two such methods are quantization, a well-known approach for network compression, and re-parametrization, an emerging technique designed to improve model performance. Although both techniques have been studied individually, there has been limited research on their simultaneous application. To address this gap, we propose a novel approach called RepQ, which applies quantization to re-parametrized networks. Our method is based on the insight that the test stage weights of an arbitrary re-parametrized layer can be presented as a differentiable function of trainable parameters. We enable quantization-aware training by applying quantization on top of this function. RepQ generalizes well to various re-parametrized models and outperforms the baseline method LSQ quantization scheme in all experiments.

{{</citation>}}


### (15/108) Explainable artificial intelligence for Healthcare applications using Random Forest Classifier with LIME and SHAP (Mrutyunjaya Panda et al., 2023)

{{<citation>}}

Mrutyunjaya Panda, Soumya Ranjan Mahanta. (2023)  
**Explainable artificial intelligence for Healthcare applications using Random Forest Classifier with LIME and SHAP**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05665v1)  

---


**ABSTRACT**  
With the advances in computationally efficient artificial Intelligence (AI) techniques and their numerous applications in our everyday life, there is a pressing need to understand the computational details hidden in black box AI techniques such as most popular machine learning and deep learning techniques; through more detailed explanations. The origin of explainable AI (xAI) is coined from these challenges and recently gained more attention by the researchers by adding explainability comprehensively in traditional AI systems. This leads to develop an appropriate framework for successful applications of xAI in real life scenarios with respect to innovations, risk mitigation, ethical issues and logical values to the users. In this book chapter, an in-depth analysis of several xAI frameworks and methods including LIME (Local Interpretable Model-agnostic Explanations) and SHAP (SHapley Additive exPlanations) are provided. Random Forest Classifier as black box AI is used on a publicly available Diabetes symptoms dataset with LIME and SHAP for better interpretations. The results obtained are interesting in terms of transparency, valid and trustworthiness in diabetes disease prediction.

{{</citation>}}


### (16/108) Mixture of Weak & Strong Experts on Graphs (Hanqing Zeng et al., 2023)

{{<citation>}}

Hanqing Zeng, Hanjia Lyu, Diyi Hu, Yinglong Xia, Jiebo Luo. (2023)  
**Mixture of Weak & Strong Experts on Graphs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2311.05185v1)  

---


**ABSTRACT**  
Realistic graphs contain both rich self-features of nodes and informative structures of neighborhoods, jointly handled by a GNN in the typical setup. We propose to decouple the two modalities by mixture of weak and strong experts (Mowst), where the weak expert is a light-weight Multi-layer Perceptron (MLP), and the strong expert is an off-the-shelf Graph Neural Network (GNN). To adapt the experts' collaboration to different target nodes, we propose a "confidence" mechanism based on the dispersion of the weak expert's prediction logits. The strong expert is conditionally activated when either the node's classification relies on neighborhood information, or the weak expert has low model quality. We reveal interesting training dynamics by analyzing the influence of the confidence function on loss: our training algorithm encourages the specialization of each expert by effectively generating soft splitting of the graph. In addition, our "confidence" design imposes a desirable bias toward the strong expert to benefit from GNN's better generalization capability. Mowst is easy to optimize and achieves strong expressive power, with a computation cost comparable to a single GNN. Empirically, Mowst shows significant accuracy improvement on 6 standard node classification benchmarks (including both homophilous and heterophilous graphs).

{{</citation>}}


### (17/108) RAPID: Training-free Retrieval-based Log Anomaly Detection with PLM considering Token-level information (Gunho No et al., 2023)

{{<citation>}}

Gunho No, Yukyung Lee, Hyeongwon Kang, Pilsung Kang. (2023)  
**RAPID: Training-free Retrieval-based Log Anomaly Detection with PLM considering Token-level information**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.05160v1)  

---


**ABSTRACT**  
As the IT industry advances, system log data becomes increasingly crucial. Many computer systems rely on log texts for management due to restricted access to source code. The need for log anomaly detection is growing, especially in real-world applications, but identifying anomalies in rapidly accumulating logs remains a challenging task. Traditional deep learning-based anomaly detection models require dataset-specific training, leading to corresponding delays. Notably, most methods only focus on sequence-level log information, which makes the detection of subtle anomalies harder, and often involve inference processes that are difficult to utilize in real-time. We introduce RAPID, a model that capitalizes on the inherent features of log data to enable anomaly detection without training delays, ensuring real-time capability. RAPID treats logs as natural language, extracting representations using pre-trained language models. Given that logs can be categorized based on system context, we implement a retrieval-based technique to contrast test logs with the most similar normal logs. This strategy not only obviates the need for log-specific training but also adeptly incorporates token-level information, ensuring refined and robust detection, particularly for unseen logs. We also propose the core set technique, which can reduce the computational cost needed for comparison. Experimental results show that even without training on log data, RAPID demonstrates competitive performance compared to prior models and achieves the best performance on certain datasets. Through various research questions, we verified its capability for real-time detection without delay.

{{</citation>}}


### (18/108) Cross-modal Prompts: Adapting Large Pre-trained Models for Audio-Visual Downstream Tasks (Haoyi Duan et al., 2023)

{{<citation>}}

Haoyi Duan, Yan Xia, Mingze Zhou, Li Tang, Jieming Zhu, Zhou Zhao. (2023)  
**Cross-modal Prompts: Adapting Large Pre-trained Models for Audio-Visual Downstream Tasks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs.LG  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.05152v1)  

---


**ABSTRACT**  
In recent years, the deployment of large-scale pre-trained models in audio-visual downstream tasks has yielded remarkable outcomes. However, these models, primarily trained on single-modality unconstrained datasets, still encounter challenges in feature extraction for multi-modal tasks, leading to suboptimal performance. This limitation arises due to the introduction of irrelevant modality-specific information during encoding, which adversely affects the performance of downstream tasks. To address this challenge, this paper proposes a novel Dual-Guided Spatial-Channel-Temporal (DG-SCT) attention mechanism. This mechanism leverages audio and visual modalities as soft prompts to dynamically adjust the parameters of pre-trained models based on the current multi-modal input features. Specifically, the DG-SCT module incorporates trainable cross-modal interaction layers into pre-trained audio-visual encoders, allowing adaptive extraction of crucial information from the current modality across spatial, channel, and temporal dimensions, while preserving the frozen parameters of large-scale pre-trained models. Experimental evaluations demonstrate that our proposed model achieves state-of-the-art results across multiple downstream tasks, including AVE, AVVP, AVS, and AVQA. Furthermore, our model exhibits promising performance in challenging few-shot and zero-shot scenarios. The source code and pre-trained models are available at https://github.com/haoyi-duan/DG-SCT.

{{</citation>}}


### (19/108) Counter-Empirical Attacking based on Adversarial Reinforcement Learning for Time-Relevant Scoring System (Xiangguo Sun et al., 2023)

{{<citation>}}

Xiangguo Sun, Hong Cheng, Hang Dong, Bo Qiao, Si Qin, Qingwei Lin. (2023)  
**Counter-Empirical Attacking based on Adversarial Reinforcement Learning for Time-Relevant Scoring System**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SE, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.05144v1)  

---


**ABSTRACT**  
Scoring systems are commonly seen for platforms in the era of big data. From credit scoring systems in financial services to membership scores in E-commerce shopping platforms, platform managers use such systems to guide users towards the encouraged activity pattern, and manage resources more effectively and more efficiently thereby. To establish such scoring systems, several "empirical criteria" are firstly determined, followed by dedicated top-down design for each factor of the score, which usually requires enormous effort to adjust and tune the scoring function in the new application scenario. What's worse, many fresh projects usually have no ground-truth or any experience to evaluate a reasonable scoring system, making the designing even harder. To reduce the effort of manual adjustment of the scoring function in every new scoring system, we innovatively study the scoring system from the preset empirical criteria without any ground truth, and propose a novel framework to improve the system from scratch. In this paper, we propose a "counter-empirical attacking" mechanism that can generate "attacking" behavior traces and try to break the empirical rules of the scoring system. Then an adversarial "enhancer" is applied to evaluate the scoring system and find the improvement strategy. By training the adversarial learning problem, a proper scoring function can be learned to be robust to the attacking activity traces that are trying to violate the empirical criteria. Extensive experiments have been conducted on two scoring systems including a shared computing resource platform and a financial credit system. The experimental results have validated the effectiveness of our proposed framework.

{{</citation>}}


### (20/108) On neural and dimensional collapse in supervised and unsupervised contrastive learning with hard negative sampling (Ruijie Jiang et al., 2023)

{{<citation>}}

Ruijie Jiang, Thuan Nguyen, Shuchin Aeron, Prakash Ishwar. (2023)  
**On neural and dimensional collapse in supervised and unsupervised contrastive learning with hard negative sampling**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.05139v1)  

---


**ABSTRACT**  
For a widely-studied data model and general loss and sample-hardening functions we prove that the Supervised Contrastive Learning (SCL), Hard-SCL (HSCL), and Unsupervised Contrastive Learning (UCL) risks are minimized by representations that exhibit Neural Collapse (NC), i.e., the class means form an Equianglular Tight Frame (ETF) and data from the same class are mapped to the same representation. We also prove that for any representation mapping, the HSCL and Hard-UCL (HUCL) risks are lower bounded by the corresponding SCL and UCL risks. Although the optimality of ETF is known for SCL, albeit only for InfoNCE loss, its optimality for HSCL and UCL under general loss and hardening functions is novel. Moreover, our proofs are much simpler, compact, and transparent. We empirically demonstrate, for the first time, that ADAM optimization of HSCL and HUCL risks with random initialization and suitable hardness levels can indeed converge to the NC geometry if we incorporate unit-ball or unit-sphere feature normalization. Without incorporating hard negatives or feature normalization, however, the representations learned via ADAM suffer from dimensional collapse (DC) and fail to attain the NC geometry.

{{</citation>}}


### (21/108) Enhancing Instance-Level Image Classification with Set-Level Labels (Renyu Zhang et al., 2023)

{{<citation>}}

Renyu Zhang, Aly A. Khan, Yuxin Chen, Robert L. Grossman. (2023)  
**Enhancing Instance-Level Image Classification with Set-Level Labels**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2311.05659v1)  

---


**ABSTRACT**  
Instance-level image classification tasks have traditionally relied on single-instance labels to train models, e.g., few-shot learning and transfer learning. However, set-level coarse-grained labels that capture relationships among instances can provide richer information in real-world scenarios. In this paper, we present a novel approach to enhance instance-level image classification by leveraging set-level labels. We provide a theoretical analysis of the proposed method, including recognition conditions for fast excess risk rate, shedding light on the theoretical foundations of our approach. We conducted experiments on two distinct categories of datasets: natural image datasets and histopathology image datasets. Our experimental results demonstrate the effectiveness of our approach, showcasing improved classification performance compared to traditional single-instance label-based methods. Notably, our algorithm achieves 13% improvement in classification accuracy compared to the strongest baseline on the histopathology image classification benchmarks. Importantly, our experimental findings align with the theoretical analysis, reinforcing the robustness and reliability of our proposed method. This work bridges the gap between instance-level and set-level image classification, offering a promising avenue for advancing the capabilities of image classification models with set-level coarse-grained labels.

{{</citation>}}


### (22/108) GeoFormer: Predicting Human Mobility using Generative Pre-trained Transformer (GPT) (Aivin V. Solatorio, 2023)

{{<citation>}}

Aivin V. Solatorio. (2023)  
**GeoFormer: Predicting Human Mobility using Generative Pre-trained Transformer (GPT)**  

---
Primary Category: cs.LG  
Categories: I-2-7; I-2-4; I-2-0; I-6-3; I-6-4; I-6-5, cs-CY, cs-LG, cs.LG  
Keywords: BLEU, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2311.05092v1)  

---


**ABSTRACT**  
Predicting human mobility holds significant practical value, with applications ranging from enhancing disaster risk planning to simulating epidemic spread. In this paper, we present the GeoFormer, a decoder-only transformer model adapted from the GPT architecture to forecast human mobility. Our proposed model is rigorously tested in the context of the HuMob Challenge 2023 -- a competition designed to evaluate the performance of prediction models on standardized datasets to predict human mobility. The challenge leverages two datasets encompassing urban-scale data of 25,000 and 100,000 individuals over a longitudinal period of 75 days. GeoFormer stands out as a top performer in the competition, securing a place in the top-3 ranking. Its success is underscored by performing well on both performance metrics chosen for the competition -- the GEO-BLEU and the Dynamic Time Warping (DTW) measures. The performance of the GeoFormer on the HuMob Challenge 2023 underscores its potential to make substantial contributions to the field of human mobility prediction, with far-reaching implications for disaster preparedness, epidemic control, and beyond.

{{</citation>}}


### (23/108) Social Media Bot Detection using Dropout-GAN (Anant Shukla et al., 2023)

{{<citation>}}

Anant Shukla, Martin Jurecek, Mark Stamp. (2023)  
**Social Media Bot Detection using Dropout-GAN**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2311.05079v1)  

---


**ABSTRACT**  
Bot activity on social media platforms is a pervasive problem, undermining the credibility of online discourse and potentially leading to cybercrime. We propose an approach to bot detection using Generative Adversarial Networks (GAN). We discuss how we overcome the issue of mode collapse by utilizing multiple discriminators to train against one generator, while decoupling the discriminator to perform social media bot detection and utilizing the generator for data augmentation. In terms of classification accuracy, our approach outperforms the state-of-the-art techniques in this field. We also show how the generator in the GAN can be used to evade such a classification technique.

{{</citation>}}


### (24/108) Mental Health Diagnosis in the Digital Age: Harnessing Sentiment Analysis on Social Media Platforms upon Ultra-Sparse Feature Content (Haijian Shao et al., 2023)

{{<citation>}}

Haijian Shao, Ming Zhu, Shengjie Zhai. (2023)  
**Mental Health Diagnosis in the Digital Age: Harnessing Sentiment Analysis on Social Media Platforms upon Ultra-Sparse Feature Content**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Sentiment Analysis, Social Media  
[Paper Link](http://arxiv.org/abs/2311.05075v1)  

---


**ABSTRACT**  
Amid growing global mental health concerns, particularly among vulnerable groups, natural language processing offers a tremendous potential for early detection and intervention of people's mental disorders via analyzing their postings and discussions on social media platforms. However, ultra-sparse training data, often due to vast vocabularies and low-frequency words, hinders the analysis accuracy. Multi-labeling and Co-occurrences of symptoms may also blur the boundaries in distinguishing similar/co-related disorders. To address these issues, we propose a novel semantic feature preprocessing technique with a three-folded structure: 1) mitigating the feature sparsity with a weak classifier, 2) adaptive feature dimension with modulus loops, and 3) deep-mining and extending features among the contexts. With enhanced semantic features, we train a machine learning model to predict and classify mental disorders. We utilize the Reddit Mental Health Dataset 2022 to examine conditions such as Anxiety, Borderline Personality Disorder (BPD), and Bipolar-Disorder (BD) and present solutions to the data sparsity challenge, highlighted by 99.81% non-zero elements. After applying our preprocessing technique, the feature sparsity decreases to 85.4%. Overall, our methods, when compared to seven benchmark models, demonstrate significant performance improvements: 8.0% in accuracy, 0.069 in precision, 0.093 in recall, 0.102 in F1 score, and 0.059 in AUC. This research provides foundational insights for mental health prediction and monitoring, providing innovative solutions to navigate challenges associated with ultra-sparse data feature and intricate multi-label classification in the domain of mental health analysis.

{{</citation>}}


## cs.CV (26)



### (25/108) Are 'Hierarchical' Visual Representations Hierarchical? (Ethan Shen et al., 2023)

{{<citation>}}

Ethan Shen, Ali Farhadi, Aditya Kusupati. (2023)  
**Are 'Hierarchical' Visual Representations Hierarchical?**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.05784v1)  

---


**ABSTRACT**  
Learned visual representations often capture large amounts of semantic information for accurate downstream applications. Human understanding of the world is fundamentally grounded in hierarchy. To mimic this and further improve representation capabilities, the community has explored "hierarchical" visual representations that aim at modeling the underlying hierarchy of the visual world. In this work, we set out to investigate if hierarchical visual representations truly capture the human perceived hierarchy better than standard learned representations. To this end, we create HierNet, a suite of 12 datasets spanning 3 kinds of hierarchy from the BREEDs subset of ImageNet. After extensive evaluation of Hyperbolic and Matryoshka Representations across training setups, we conclude that they do not capture hierarchy any better than the standard representations but can assist in other aspects like search efficiency and interpretability. Our benchmark and the datasets are open-sourced at https://github.com/ethanlshen/HierNet.

{{</citation>}}


### (26/108) DONUT-hole: DONUT Sparsification by Harnessing Knowledge and Optimizing Learning Efficiency (Azhar Shaikh et al., 2023)

{{<citation>}}

Azhar Shaikh, Michael Cochez, Denis Diachkov, Michiel de Rijcke, Sahar Yousefi. (2023)  
**DONUT-hole: DONUT Sparsification by Harnessing Knowledge and Optimizing Learning Efficiency**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2311.05778v1)  

---


**ABSTRACT**  
This paper introduces DONUT-hole, a sparse OCR-free visual document understanding (VDU) model that addresses the limitations of its predecessor model, dubbed DONUT. The DONUT model, leveraging a transformer architecture, overcoming the challenges of separate optical character recognition (OCR) and visual semantic understanding (VSU) components. However, its deployment in production environments and edge devices is hindered by high memory and computational demands, particularly in large-scale request services. To overcome these challenges, we propose an optimization strategy based on knowledge distillation and model pruning. Our paradigm to produce DONUT-hole, reduces the model denisty by 54\% while preserving performance. We also achieve a global representational similarity index between DONUT and DONUT-hole based on centered kernel alignment (CKA) metric of 0.79. Moreover, we evaluate the effectiveness of DONUT-hole in the document image key information extraction (KIE) task, highlighting its potential for developing more efficient VDU systems for logistic companies.

{{</citation>}}


### (27/108) PolyMaX: General Dense Prediction with Mask Transformer (Xuan Yang et al., 2023)

{{<citation>}}

Xuan Yang, Liangzhe Yuan, Kimberly Wilber, Astuti Sharma, Xiuye Gu, Siyuan Qiao, Stephanie Debats, Huisheng Wang, Hartwig Adam, Mikhail Sirotenko, Liang-Chieh Chen. (2023)  
**PolyMaX: General Dense Prediction with Mask Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.05770v1)  

---


**ABSTRACT**  
Dense prediction tasks, such as semantic segmentation, depth estimation, and surface normal prediction, can be easily formulated as per-pixel classification (discrete outputs) or regression (continuous outputs). This per-pixel prediction paradigm has remained popular due to the prevalence of fully convolutional networks. However, on the recent frontier of segmentation task, the community has been witnessing a shift of paradigm from per-pixel prediction to cluster-prediction with the emergence of transformer architectures, particularly the mask transformers, which directly predicts a label for a mask instead of a pixel. Despite this shift, methods based on the per-pixel prediction paradigm still dominate the benchmarks on the other dense prediction tasks that require continuous outputs, such as depth estimation and surface normal prediction. Motivated by the success of DORN and AdaBins in depth estimation, achieved by discretizing the continuous output space, we propose to generalize the cluster-prediction based method to general dense prediction tasks. This allows us to unify dense prediction tasks with the mask transformer framework. Remarkably, the resulting model PolyMaX demonstrates state-of-the-art performance on three benchmarks of NYUD-v2 dataset. We hope our simple yet effective design can inspire more research on exploiting mask transformers for more dense prediction tasks. Code and model will be made available.

{{</citation>}}


### (28/108) GIPCOL: Graph-Injected Soft Prompting for Compositional Zero-Shot Learning (Guangyue Xu et al., 2023)

{{<citation>}}

Guangyue Xu, Joyce Chai, Parisa Kordjamshidi. (2023)  
**GIPCOL: Graph-Injected Soft Prompting for Compositional Zero-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.05729v1)  

---


**ABSTRACT**  
Pre-trained vision-language models (VLMs) have achieved promising success in many fields, especially with prompt learning paradigm. In this work, we propose GIP-COL (Graph-Injected Soft Prompting for COmpositional Learning) to better explore the compositional zero-shot learning (CZSL) ability of VLMs within the prompt-based learning framework. The soft prompt in GIPCOL is structured and consists of the prefix learnable vectors, attribute label and object label. In addition, the attribute and object labels in the soft prompt are designated as nodes in a compositional graph. The compositional graph is constructed based on the compositional structure of the objects and attributes extracted from the training data and consequently feeds the updated concept representation into the soft prompt to capture this compositional structure for a better prompting for CZSL. With the new prompting strategy, GIPCOL achieves state-of-the-art AUC results on all three CZSL benchmarks, including MIT-States, UT-Zappos, and C-GQA datasets in both closed and open settings compared to previous non-CLIP as well as CLIP-based methods. We analyze when and why GIPCOL operates well given the CLIP backbone and its training data limitations, and our findings shed light on designing more effective prompts for CZSL

{{</citation>}}


### (29/108) Intelligent Cervical Spine Fracture Detection Using Deep Learning Methods (Reza Behbahani Nejad et al., 2023)

{{<citation>}}

Reza Behbahani Nejad, Amir Hossein Komijani, Esmaeil Najafi. (2023)  
**Intelligent Cervical Spine Fracture Detection Using Deep Learning Methods**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.05708v1)  

---


**ABSTRACT**  
Cervical spine fractures constitute a critical medical emergency, with the potential for lifelong paralysis or even fatality if left untreated or undetected. Over time, these fractures can deteriorate without intervention. To address the lack of research on the practical application of deep learning techniques for the detection of spine fractures, this study leverages a dataset containing both cervical spine fractures and non-fractured computed tomography images. This paper introduces a two-stage pipeline designed to identify the presence of cervical vertebrae in each image slice and pinpoint the location of fractures. In the first stage, a multi-input network, incorporating image and image metadata, is trained. This network is based on the Global Context Vision Transformer, and its performance is benchmarked against popular deep learning image classification model. In the second stage, a YOLOv8 model is trained to detect fractures within the images, and its effectiveness is compared to YOLOv5. The obtained results indicate that the proposed algorithm significantly reduces the workload of radiologists and enhances the accuracy of fracture detection.

{{</citation>}}


### (30/108) FMViT: A multiple-frequency mixing Vision Transformer (Wei Tan et al., 2023)

{{<citation>}}

Wei Tan, Yifeng Geng, Xuansong Xie. (2023)  
**FMViT: A multiple-frequency mixing Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, ImageNet, Self-Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.05707v1)  

---


**ABSTRACT**  
The transformer model has gained widespread adoption in computer vision tasks in recent times. However, due to the quadratic time and memory complexity of self-attention, which is proportional to the number of input tokens, most existing Vision Transformers (ViTs) encounter challenges in achieving efficient performance in practical industrial deployment scenarios, such as TensorRT and CoreML, where traditional CNNs excel. Although some recent attempts have been made to design CNN-Transformer hybrid architectures to tackle this problem, their overall performance has not met expectations. To tackle these challenges, we propose an efficient hybrid ViT architecture named FMViT. This approach enhances the model's expressive power by blending high-frequency features and low-frequency features with varying frequencies, enabling it to capture both local and global information effectively. Additionally, we introduce deploy-friendly mechanisms such as Convolutional Multigroup Reparameterization (gMLP), Lightweight Multi-head Self-Attention (RLMHSA), and Convolutional Fusion Block (CFB) to further improve the model's performance and reduce computational overhead. Our experiments demonstrate that FMViT surpasses existing CNNs, ViTs, and CNNTransformer hybrid architectures in terms of latency/accuracy trade-offs for various vision tasks. On the TensorRT platform, FMViT outperforms Resnet101 by 2.5% (83.3% vs. 80.8%) in top-1 accuracy on the ImageNet dataset while maintaining similar inference latency. Moreover, FMViT achieves comparable performance with EfficientNet-B5, but with a 43% improvement in inference speed. On CoreML, FMViT outperforms MobileOne by 2.6% in top-1 accuracy on the ImageNet dataset, with inference latency comparable to MobileOne (78.5% vs. 75.9%). Our code can be found at https://github.com/tany0699/FMViT.

{{</citation>}}


### (31/108) Window Attention is Bugged: How not to Interpolate Position Embeddings (Daniel Bolya et al., 2023)

{{<citation>}}

Daniel Bolya, Chaitanya Ryali, Judy Hoffman, Christoph Feichtenhofer. (2023)  
**Window Attention is Bugged: How not to Interpolate Position Embeddings**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Embedding, ImageNet, Position Embedding  
[Paper Link](http://arxiv.org/abs/2311.05613v1)  

---


**ABSTRACT**  
Window attention, position embeddings, and high resolution finetuning are core concepts in the modern transformer era of computer vision. However, we find that naively combining these near ubiquitous components can have a detrimental effect on performance. The issue is simple: interpolating position embeddings while using window attention is wrong. We study two state-of-the-art methods that have these three components, namely Hiera and ViTDet, and find that both do indeed suffer from this bug. To fix it, we introduce a simple absolute window position embedding strategy, which solves the bug outright in Hiera and allows us to increase both speed and performance of the model in ViTDet. We finally combine the two to obtain HieraDet, which achieves 61.7 box mAP on COCO, making it state-of-the-art for models that only use ImageNet-1k pretraining. This all stems from what is essentially a 3 line bug fix, which we name "absolute win".

{{</citation>}}


### (32/108) 3D-QAE: Fully Quantum Auto-Encoding of 3D Point Clouds (Lakshika Rathi et al., 2023)

{{<citation>}}

Lakshika Rathi, Edith Tretschk, Christian Theobalt, Rishabh Dabral, Vladislav Golyanik. (2023)  
**3D-QAE: Fully Quantum Auto-Encoding of 3D Point Clouds**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.05604v1)  

---


**ABSTRACT**  
Existing methods for learning 3D representations are deep neural networks trained and tested on classical hardware. Quantum machine learning architectures, despite their theoretically predicted advantages in terms of speed and the representational capacity, have so far not been considered for this problem nor for tasks involving 3D data in general. This paper thus introduces the first quantum auto-encoder for 3D point clouds. Our 3D-QAE approach is fully quantum, i.e. all its data processing components are designed for quantum hardware. It is trained on collections of 3D point clouds to produce their compressed representations. Along with finding a suitable architecture, the core challenges in designing such a fully quantum model include 3D data normalisation and parameter optimisation, and we propose solutions for both these tasks. Experiments on simulated gate-based quantum hardware demonstrate that our method outperforms simple classical baselines, paving the way for a new research direction in 3D computer vision. The source code is available at https://4dqv.mpi-inf.mpg.de/QAE3D/.

{{</citation>}}


### (33/108) Accuracy of a Vision-Language Model on Challenging Medical Cases (Thomas Buckley et al., 2023)

{{<citation>}}

Thomas Buckley, James A. Diao, Adam Rodman, Arjun K. Manrai. (2023)  
**Accuracy of a Vision-Language Model on Challenging Medical Cases**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: AI, GPT, GPT-4, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2311.05591v1)  

---


**ABSTRACT**  
Background: General-purpose large language models that utilize both text and images have not been evaluated on a diverse array of challenging medical cases.   Methods: Using 934 cases from the NEJM Image Challenge published between 2005 and 2023, we evaluated the accuracy of the recently released Generative Pre-trained Transformer 4 with Vision model (GPT-4V) compared to human respondents overall and stratified by question difficulty, image type, and skin tone. We further conducted a physician evaluation of GPT-4V on 69 NEJM clinicopathological conferences (CPCs). Analyses were conducted for models utilizing text alone, images alone, and both text and images.   Results: GPT-4V achieved an overall accuracy of 61% (95% CI, 58 to 64%) compared to 49% (95% CI, 49 to 50%) for humans. GPT-4V outperformed humans at all levels of difficulty and disagreement, skin tones, and image types; the exception was radiographic images, where performance was equivalent between GPT-4V and human respondents. Longer, more informative captions were associated with improved performance for GPT-4V but similar performance for human respondents. GPT-4V included the correct diagnosis in its differential for 80% (95% CI, 68 to 88%) of CPCs when using text alone, compared to 58% (95% CI, 45 to 70%) of CPCs when using both images and text.   Conclusions: GPT-4V outperformed human respondents on challenging medical cases and was able to synthesize information from both images and text, but performance deteriorated when images were added to highly informative text. Overall, our results suggest that multimodal AI models may be useful in medical diagnostic reasoning but that their accuracy may depend heavily on context.

{{</citation>}}


### (34/108) High-Performance Transformers for Table Structure Recognition Need Early Convolutions (ShengYun Peng et al., 2023)

{{<citation>}}

ShengYun Peng, Seongmin Lee, Xiaojing Wang, Rajarajeswari Balasubramaniyan, Duen Horng Chau. (2023)  
**High-Performance Transformers for Table Structure Recognition Need Early Convolutions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.05565v1)  

---


**ABSTRACT**  
Table structure recognition (TSR) aims to convert tabular images into a machine-readable format, where a visual encoder extracts image features and a textual decoder generates table-representing tokens. Existing approaches use classic convolutional neural network (CNN) backbones for the visual encoder and transformers for the textual decoder. However, this hybrid CNN-Transformer architecture introduces a complex visual encoder that accounts for nearly half of the total model parameters, markedly reduces both training and inference speed, and hinders the potential for self-supervised learning in TSR. In this work, we design a lightweight visual encoder for TSR without sacrificing expressive power. We discover that a convolutional stem can match classic CNN backbone performance, with a much simpler model. The convolutional stem strikes an optimal balance between two crucial factors for high-performance TSR: a higher receptive field (RF) ratio and a longer sequence length. This allows it to "see" an appropriate portion of the table and "store" the complex table structure within sufficient context length for the subsequent transformer. We conducted reproducible ablation studies and open-sourced our code at https://github.com/poloclub/tsr-convstem to enhance transparency, inspire innovations, and facilitate fair comparisons in our domain as tables are a promising modality for representation learning.

{{</citation>}}


### (35/108) Object-centric Cross-modal Feature Distillation for Event-based Object Detection (Lei Li et al., 2023)

{{<citation>}}

Lei Li, Alexander Liniger, Mario Millhaeusler, Vagia Tsiminaki, Yuanyou Li, Dengxin Dai. (2023)  
**Object-centric Cross-modal Feature Distillation for Event-based Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.05494v1)  

---


**ABSTRACT**  
Event cameras are gaining popularity due to their unique properties, such as their low latency and high dynamic range. One task where these benefits can be crucial is real-time object detection. However, RGB detectors still outperform event-based detectors due to the sparsity of the event data and missing visual details. In this paper, we develop a novel knowledge distillation approach to shrink the performance gap between these two modalities. To this end, we propose a cross-modality object detection distillation method that by design can focus on regions where the knowledge distillation works best. We achieve this by using an object-centric slot attention mechanism that can iteratively decouple features maps into object-centric features and corresponding pixel-features used for distillation. We evaluate our novel distillation approach on a synthetic and a real event dataset with aligned grayscale images as a teacher modality. We show that object-centric distillation allows to significantly improve the performance of the event-based student object detector, nearly halving the performance gap with respect to the teacher.

{{</citation>}}


### (36/108) LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents (Shilong Liu et al., 2023)

{{<citation>}}

Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang, Jianfeng Gao, Chunyuan Li. (2023)  
**LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs-MM, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05437v1)  

---


**ABSTRACT**  
LLaVA-Plus is a general-purpose multimodal assistant that expands the capabilities of large multimodal models. It maintains a skill repository of pre-trained vision and vision-language models and can activate relevant tools based on users' inputs to fulfill real-world tasks. LLaVA-Plus is trained on multimodal instruction-following data to acquire the ability to use tools, covering visual understanding, generation, external knowledge retrieval, and compositions. Empirical results show that LLaVA-Plus outperforms LLaVA in existing capabilities and exhibits new ones. It is distinct in that the image query is directly grounded and actively engaged throughout the entire human-AI interaction sessions, significantly improving tool use performance and enabling new scenarios.

{{</citation>}}


### (37/108) Dual Pipeline Style Transfer with Input Distribution Differentiation (ShiQi Jiang et al., 2023)

{{<citation>}}

ShiQi Jiang, JunJie Kang, YuJian Li. (2023)  
**Dual Pipeline Style Transfer with Input Distribution Differentiation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2311.05432v1)  

---


**ABSTRACT**  
The color and texture dual pipeline architecture (CTDP) suppresses texture representation and artifacts through masked total variation loss (Mtv), and further experiments have shown that smooth input can almost completely eliminate texture representation. We have demonstrated through experiments that smooth input is not the key reason for removing texture representations, but rather the distribution differentiation of the training dataset. Based on this, we propose an input distribution differentiation training strategy (IDD), which forces the generation of textures to be completely dependent on the noise distribution, while the smooth distribution will not produce textures at all. Overall, our proposed distribution differentiation training strategy allows for two pre-defined input distributions to be responsible for two generation tasks, with noise distribution responsible for texture generation and smooth distribution responsible for color smooth transfer. Finally, we choose a smooth distribution as the input for the forward inference stage to completely eliminate texture representations and artifacts in color transfer tasks.

{{</citation>}}


### (38/108) Linear Gaussian Bounding Box Representation and Ring-Shaped Rotated Convolution for Oriented Object Detection (Zhen Zhou et al., 2023)

{{<citation>}}

Zhen Zhou, Yunkai Ma, Junfeng Fan, Zhaoyang Liu, Fengshui Jing, Min Tan. (2023)  
**Linear Gaussian Bounding Box Representation and Ring-Shaped Rotated Convolution for Oriented Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.05410v1)  

---


**ABSTRACT**  
Due to the frequent variability of object orientation, accurate prediction of orientation information remains a challenge in oriented object detection. To better extract orientation-related information, current methods primarily focus on the design of reasonable representations of oriented bounding box (OBB) and rotation-sensitive feature extraction. However, existing OBB representations often suffer from boundary discontinuity and representation ambiguity problems. Methods of designing continuous and unambiguous regression losses do not essentially solve such problems. Gaussian bounding box (GBB) avoids these OBB representation problems, but directly regressing GBB is susceptible to numerical instability. In this paper, we propose linear GBB (LGBB), a novel OBB representation. By linearly transforming the elements of GBB, LGBB does not have the boundary discontinuity and representation ambiguity problems, and have high numerical stability. On the other hand, current rotation-sensitive feature extraction methods based on convolutions can only extract features under a local receptive field, which is slow in aggregating rotation-sensitive features. To address this issue, we propose ring-shaped rotated convolution (RRC). By adaptively rotating feature maps to arbitrary orientations, RRC extracts rotation-sensitive features under a ring-shaped receptive field, rapidly aggregating rotation-sensitive features and contextual information. RRC can be applied to various models in a plug-and-play manner. Experimental results demonstrate that the proposed LGBB and RRC are effective and achieve state-of-the-art (SOTA) performance. By integrating LGBB and RRC into various models, the detection accuracy is effectively improved on DOTA and HRSC2016 datasets.

{{</citation>}}


### (39/108) Improving Hand Recognition in Uncontrolled and Uncooperative Environments using Multiple Spatial Transformers and Loss Functions (Wojciech Michal Matkowski et al., 2023)

{{<citation>}}

Wojciech Michal Matkowski, Xiaojie Li, Adams Wai Kin Kong. (2023)  
**Improving Hand Recognition in Uncontrolled and Uncooperative Environments using Multiple Spatial Transformers and Loss Functions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.05383v1)  

---


**ABSTRACT**  
The prevalence of smartphone and consumer camera has led to more evidence in the form of digital images, which are mostly taken in uncontrolled and uncooperative environments. In these images, criminals likely hide or cover their faces while their hands are observable in some cases, creating a challenging use case for forensic investigation. Many existing hand-based recognition methods perform well for hand images collected in controlled environments with user cooperation. However, their performance deteriorates significantly in uncontrolled and uncooperative environments. A recent work has exposed the potential of hand recognition in these environments. However, only the palmar regions were considered, and the recognition performance is still far from satisfactory. To improve the recognition accuracy, an algorithm integrating a multi-spatial transformer network (MSTN) and multiple loss functions is proposed to fully utilize information in full hand images. MSTN is firstly employed to localize the palms and fingers and estimate the alignment parameters. Then, the aligned images are further fed into pretrained convolutional neural networks, where features are extracted. Finally, a training scheme with multiple loss functions is used to train the network end-to-end. To demonstrate the effectiveness of the proposed algorithm, the trained model is evaluated on NTU-PI-v1 database and six benchmark databases from different domains. Experimental results show that the proposed algorithm performs significantly better than the existing methods in these uncontrolled and uncooperative environments and has good generalization capabilities to samples from different domains.

{{</citation>}}


### (40/108) u-LLaVA: Unifying Multi-Modal Tasks via Large Language Model (Jinjin Xu et al., 2023)

{{<citation>}}

Jinjin Xu, Liwu Xu, Yuzhe Yang, Xiang Li, Yanchun Xie, Yi-Jie Huang, Yaqian Li. (2023)  
**u-LLaVA: Unifying Multi-Modal Tasks via Large Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.05348v1)  

---


**ABSTRACT**  
Recent advances such as LLaVA and Mini-GPT4 have successfully integrated visual information into LLMs, yielding inspiring outcomes and giving rise to a new generation of multi-modal LLMs, or MLLMs. Nevertheless, these methods struggle with hallucinations and the mutual interference between tasks. To tackle these problems, we propose an efficient and accurate approach to adapt to downstream tasks by utilizing LLM as a bridge to connect multiple expert models, namely u-LLaVA. Firstly, we incorporate the modality alignment module and multi-task modules into LLM. Then, we reorganize or rebuild multi-type public datasets to enable efficient modality alignment and instruction following. Finally, task-specific information is extracted from the trained LLM and provided to different modules for solving downstream tasks. The overall framework is simple, effective, and achieves state-of-the-art performance across multiple benchmarks. We also release our model, the generated data, and the code base publicly available.

{{</citation>}}


### (41/108) On the Road with GPT-4V(ision): Early Explorations of Visual-Language Model on Autonomous Driving (Licheng Wen et al., 2023)

{{<citation>}}

Licheng Wen, Xuemeng Yang, Daocheng Fu, Xiaofeng Wang, Pinlong Cai, Xin Li, Tao Ma, Yingxuan Li, Linran Xu, Dengke Shang, Zheng Zhu, Shaoyan Sun, Yeqi Bai, Xinyu Cai, Min Dou, Shuanglu Hu, Botian Shi. (2023)  
**On the Road with GPT-4V(ision): Early Explorations of Visual-Language Model on Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-RO, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.05332v1)  

---


**ABSTRACT**  
The pursuit of autonomous driving technology hinges on the sophisticated integration of perception, decision-making, and control systems. Traditional approaches, both data-driven and rule-based, have been hindered by their inability to grasp the nuance of complex driving environments and the intentions of other road users. This has been a significant bottleneck, particularly in the development of common sense reasoning and nuanced scene understanding necessary for safe and reliable autonomous driving. The advent of Visual Language Models (VLM) represents a novel frontier in realizing fully autonomous vehicle driving. This report provides an exhaustive evaluation of the latest state-of-the-art VLM, \modelnamefull, and its application in autonomous driving scenarios. We explore the model's abilities to understand and reason about driving scenes, make decisions, and ultimately act in the capacity of a driver. Our comprehensive tests span from basic scene recognition to complex causal reasoning and real-time decision-making under varying conditions. Our findings reveal that \modelname demonstrates superior performance in scene understanding and causal reasoning compared to existing autonomous systems. It showcases the potential to handle out-of-distribution scenarios, recognize intentions, and make informed decisions in real driving contexts. However, challenges remain, particularly in direction discernment, traffic light recognition, vision grounding, and spatial reasoning tasks. These limitations underscore the need for further research and development. Project is now available on GitHub for interested parties to access and utilize: \url{https://github.com/PJLab-ADG/GPT4V-AD-Exploration}

{{</citation>}}


### (42/108) Spatial Attention-based Distribution Integration Network for Human Pose Estimation (Sihan Gao et al., 2023)

{{<citation>}}

Sihan Gao, Jing Zhu, Xiaoxuan Zhuang, Zhaoyue Wang, Qijin Li. (2023)  
**Spatial Attention-based Distribution Integration Network for Human Pose Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.05323v1)  

---


**ABSTRACT**  
In recent years, human pose estimation has made significant progress through the implementation of deep learning techniques. However, these techniques still face limitations when confronted with challenging scenarios, including occlusion, diverse appearances, variations in illumination, and overlap. To cope with such drawbacks, we present the Spatial Attention-based Distribution Integration Network (SADI-NET) to improve the accuracy of localization in such situations. Our network consists of three efficient models: the receptive fortified module (RFM), spatial fusion module (SFM), and distribution learning module (DLM). Building upon the classic HourglassNet architecture, we replace the basic block with our proposed RFM. The RFM incorporates a dilated residual block and attention mechanism to expand receptive fields while enhancing sensitivity to spatial information. In addition, the SFM incorporates multi-scale characteristics by employing both global and local attention mechanisms. Furthermore, the DLM, inspired by residual log-likelihood estimation (RLE), reconfigures a predicted heatmap using a trainable distribution weight. For the purpose of determining the efficacy of our model, we conducted extensive experiments on the MPII and LSP benchmarks. Particularly, our model obtained a remarkable $92.10\%$ percent accuracy on the MPII test dataset, demonstrating significant improvements over existing models and establishing state-of-the-art performance.

{{</citation>}}


### (43/108) Improving Vision-and-Language Reasoning via Spatial Relations Modeling (Cheng Yang et al., 2023)

{{<citation>}}

Cheng Yang, Rui Xu, Ye Guo, Peixiang Huang, Yiru Chen, Wenkui Ding, Zhongyuan Wang, Hong Zhou. (2023)  
**Improving Vision-and-Language Reasoning via Spatial Relations Modeling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BERT, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.05298v1)  

---


**ABSTRACT**  
Visual commonsense reasoning (VCR) is a challenging multi-modal task, which requires high-level cognition and commonsense reasoning ability about the real world. In recent years, large-scale pre-training approaches have been developed and promoted the state-of-the-art performance of VCR. However, the existing approaches almost employ the BERT-like objectives to learn multi-modal representations. These objectives motivated from the text-domain are insufficient for the excavation on the complex scenario of visual modality. Most importantly, the spatial distribution of the visual objects is basically neglected. To address the above issue, we propose to construct the spatial relation graph based on the given visual scenario. Further, we design two pre-training tasks named object position regression (OPR) and spatial relation classification (SRC) to learn to reconstruct the spatial relation graph respectively. Quantitative analysis suggests that the proposed method can guide the representations to maintain more spatial context and facilitate the attention on the essential visual regions for reasoning. We achieve the state-of-the-art results on VCR and two other vision-and-language reasoning tasks VQA, and NLVR.

{{</citation>}}


### (44/108) BrainNetDiff: Generative AI Empowers Brain Network Generation via Multimodal Diffusion Model (Yongcheng Zong et al., 2023)

{{<citation>}}

Yongcheng Zong, Shuqiang Wang. (2023)  
**BrainNetDiff: Generative AI Empowers Brain Network Generation via Multimodal Diffusion Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Generative AI, Transformer  
[Paper Link](http://arxiv.org/abs/2311.05199v1)  

---


**ABSTRACT**  
Brain network analysis has emerged as pivotal method for gaining a deeper understanding of brain functions and disease mechanisms. Despite the existence of various network construction approaches, shortcomings persist in the learning of correlations between structural and functional brain imaging data. In light of this, we introduce a novel method called BrainNetDiff, which combines a multi-head Transformer encoder to extract relevant features from fMRI time series and integrates a conditional latent diffusion model for brain network generation. Leveraging a conditional prompt and a fusion attention mechanism, this method significantly improves the accuracy and stability of brain network generation. To the best of our knowledge, this represents the first framework that employs diffusion for the fusion of the multimodal brain imaging and brain network generation from images to graphs. We validate applicability of this framework in the construction of brain network across healthy and neurologically impaired cohorts using the authentic dataset. Experimental results vividly demonstrate the significant effectiveness of the proposed method across the downstream disease classification tasks. These findings convincingly emphasize the prospective value in the field of brain network research, particularly its key significance in neuroimaging analysis and disease diagnosis. This research provides a valuable reference for the processing of multimodal brain imaging data and introduces a novel, efficient solution to the field of neuroimaging.

{{</citation>}}


### (45/108) Deep Learning in Computed Tomography Pulmonary Angiography Imaging: A Dual-Pronged Approach for Pulmonary Embolism Detection (Fabiha Bushra et al., 2023)

{{<citation>}}

Fabiha Bushra, Muhammad E. H. Chowdhury, Rusab Sarmun, Saidul Kabir, Menatalla Said, Sohaib Bassam Zoghoul, Adam Mushtak, Israa Al-Hashimi, Abdulrahman Alqahtani, Anwarul Hasan. (2023)  
**Deep Learning in Computed Tomography Pulmonary Angiography Imaging: A Dual-Pronged Approach for Pulmonary Embolism Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2311.05197v1)  

---


**ABSTRACT**  
Pulmonary Embolism (PE) is a critical medical condition characterized by obstructions in the pulmonary arteries. Despite being a major health concern, it often goes underdiagnosed leading to detrimental clinical outcomes. The increasing reliance on Computed Tomography Pulmonary Angiography for diagnosis presents challenges and a pressing need for enhanced diagnostic solutions. The primary objective of this study is to leverage deep learning techniques to enhance the Computer Assisted Diagnosis of PE. This study presents a comprehensive dual-pronged approach combining classification and detection for PE diagnosis. We introduce an Attention-Guided Convolutional Neural Network (AG-CNN) for classification, addressing both global and local lesion region. For detection, state-of-the-art models are employed to pinpoint potential PE regions. Different ensembling techniques further improve detection accuracy by combining predictions from different models. Finally, a heuristic strategy integrates classifier outputs with detection results, ensuring robust and accurate PE identification. Our attention-guided classification approach, tested on the Ferdowsi University of Mashhad's Pulmonary Embolism (FUMPE) dataset, outperformed the baseline model DenseNet-121 by achieving an 8.1% increase in the Area Under the Receiver Operating Characteristic. By employing ensemble techniques with detection models, the mean average precision (mAP) was considerably enhanced by a 4.7% increase. The classifier-guided framework further refined the mAP and F1 scores over the ensemble models. Our research offers a comprehensive approach to PE diagnostics using deep learning, addressing the prevalent issues of underdiagnosis and misdiagnosis. We aim to improve PE patient care by integrating AI solutions into clinical workflows, highlighting the potential of human-AI collaboration in medical diagnostics.

{{</citation>}}


### (46/108) FireMatch: A Semi-Supervised Video Fire Detection Network Based on Consistency and Distribution Alignment (Qinghua Lin et al., 2023)

{{<citation>}}

Qinghua Lin, Zuoyong Li, Kun Zeng, Haoyi Fan, Wei Li, Xiaoguang Zhou. (2023)  
**FireMatch: A Semi-Supervised Video Fire Detection Network Based on Consistency and Distribution Alignment**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2311.05168v1)  

---


**ABSTRACT**  
Deep learning techniques have greatly enhanced the performance of fire detection in videos. However, video-based fire detection models heavily rely on labeled data, and the process of data labeling is particularly costly and time-consuming, especially when dealing with videos. Considering the limited quantity of labeled video data, we propose a semi-supervised fire detection model called FireMatch, which is based on consistency regularization and adversarial distribution alignment. Specifically, we first combine consistency regularization with pseudo-label. For unlabeled data, we design video data augmentation to obtain corresponding weakly augmented and strongly augmented samples. The proposed model predicts weakly augmented samples and retains pseudo-label above a threshold, while training on strongly augmented samples to predict these pseudo-labels for learning more robust feature representations. Secondly, we generate video cross-set augmented samples by adversarial distribution alignment to expand the training data and alleviate the decline in classification performance caused by insufficient labeled data. Finally, we introduce a fairness loss to help the model produce diverse predictions for input samples, thereby addressing the issue of high confidence with the non-fire class in fire classification scenarios. The FireMatch achieved an accuracy of 76.92% and 91.81% on two real-world fire datasets, respectively. The experimental results demonstrate that the proposed method outperforms the current state-of-the-art semi-supervised classification methods.

{{</citation>}}


### (47/108) Dynamic Association Learning of Self-Attention and Convolution in Image Restoration (Kui Jiang et al., 2023)

{{<citation>}}

Kui Jiang, Xuemei Jia, Wenxin Huang, Wenbin Wang, Zheng Wang, Junjun Jiang. (2023)  
**Dynamic Association Learning of Self-Attention and Convolution in Image Restoration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Self-Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2311.05147v1)  

---


**ABSTRACT**  
CNNs and Self attention have achieved great success in multimedia applications for dynamic association learning of self-attention and convolution in image restoration. However, CNNs have at least two shortcomings: 1) limited receptive field; 2) static weight of sliding window at inference, unable to cope with the content diversity.In view of the advantages and disadvantages of CNNs and Self attention, this paper proposes an association learning method to utilize the advantages and suppress their shortcomings, so as to achieve high-quality and efficient inpainting. We regard rain distribution reflects the degradation location and degree, in addition to the rain distribution prediction. Thus, we propose to refine background textures with the predicted degradation prior in an association learning manner. As a result, we accomplish image deraining by associating rain streak removal and background recovery, where an image deraining network and a background recovery network are designed for two subtasks. The key part of association learning is a novel multi-input attention module. It generates the degradation prior and produces the degradation mask according to the predicted rainy distribution. Benefited from the global correlation calculation of SA, MAM can extract the informative complementary components from the rainy input with the degradation mask, and then help accurate texture restoration. Meanwhile, SA tends to aggregate feature maps with self-attention importance, but convolution diversifies them to focus on the local textures. A hybrid fusion network involves one residual Transformer branch and one encoder-decoder branch. The former takes a few learnable tokens as input and stacks multi-head attention and feed-forward networks to encode global features of the image. The latter, conversely, leverages the multi-scale encoder-decoder to represent contexture knowledge.

{{</citation>}}


### (48/108) OW-SLR: Overlapping Windows on Semi-Local Region for Image Super-Resolution (Rishav Bhardwaj et al., 2023)

{{<citation>}}

Rishav Bhardwaj, Janarthanam Jothi Balaji, Vasudevan Lakshminarayanan. (2023)  
**OW-SLR: Overlapping Windows on Semi-Local Region for Image Super-Resolution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: T5  
[Paper Link](http://arxiv.org/abs/2311.05146v1)  

---


**ABSTRACT**  
There has been considerable progress in implicit neural representation to upscale an image to any arbitrary resolution. However, existing methods are based on defining a function to predict the Red, Green and Blue (RGB) value from just four specific loci. Relying on just four loci is insufficient as it leads to losing fine details from the neighboring region(s). We show that by taking into account the semi-local region leads to an improvement in performance. In this paper, we propose applying a new technique called Overlapping Windows on Semi-Local Region (OW-SLR) to an image to obtain any arbitrary resolution by taking the coordinates of the semi-local region around a point in the latent space. This extracted detail is used to predict the RGB value of a point. We illustrate the technique by applying the algorithm to the Optical Coherence Tomography-Angiography (OCT-A) images and show that it can upscale them to random resolution. This technique outperforms the existing state-of-the-art methods when applied to the OCT500 dataset. OW-SLR provides better results for classifying healthy and diseased retinal images such as diabetic retinopathy and normals from the given set of OCT-A images. The project page is available at https://rishavbb.github.io/ow-slr/index.html

{{</citation>}}


### (49/108) SCAAT: Improving Neural Network Interpretability via Saliency Constrained Adaptive Adversarial Training (Rui Xu et al., 2023)

{{<citation>}}

Rui Xu, Wenkang Qin, Peixiang Huang, Hao Wang, Lin Luo. (2023)  
**SCAAT: Improving Neural Network Interpretability via Saliency Constrained Adaptive Adversarial Training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2311.05143v2)  

---


**ABSTRACT**  
Deep Neural Networks (DNNs) are expected to provide explanation for users to understand their black-box predictions. Saliency map is a common form of explanation illustrating the heatmap of feature attributions, but it suffers from noise in distinguishing important features. In this paper, we propose a model-agnostic learning method called Saliency Constrained Adaptive Adversarial Training (SCAAT) to improve the quality of such DNN interpretability. By constructing adversarial samples under the guidance of saliency map, SCAAT effectively eliminates most noise and makes saliency maps sparser and more faithful without any modification to the model architecture. We apply SCAAT to multiple DNNs and evaluate the quality of the generated saliency maps on various natural and pathological image datasets. Evaluations on different domains and metrics show that SCAAT significantly improves the interpretability of DNNs by providing more faithful saliency maps without sacrificing their predictive power.

{{</citation>}}


### (50/108) Reducing the Side-Effects of Oscillations in Training of Quantized YOLO Networks (Kartik Gupta et al., 2023)

{{<citation>}}

Kartik Gupta, Akshay Asthana. (2023)  
**Reducing the Side-Effects of Oscillations in Training of Quantized YOLO Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.05109v1)  

---


**ABSTRACT**  
Quantized networks use less computational and memory resources and are suitable for deployment on edge devices. While quantization-aware training QAT is the well-studied approach to quantize the networks at low precision, most research focuses on over-parameterized networks for classification with limited studies on popular and edge device friendly single-shot object detection and semantic segmentation methods like YOLO. Moreover, majority of QAT methods rely on Straight-through Estimator (STE) approximation which suffers from an oscillation phenomenon resulting in sub-optimal network quantization. In this paper, we show that it is difficult to achieve extremely low precision (4-bit and lower) for efficient YOLO models even with SOTA QAT methods due to oscillation issue and existing methods to overcome this problem are not effective on these models. To mitigate the effect of oscillation, we first propose Exponentially Moving Average (EMA) based update to the QAT model. Further, we propose a simple QAT correction method, namely QC, that takes only a single epoch of training after standard QAT procedure to correct the error induced by oscillating weights and activations resulting in a more accurate quantized model. With extensive evaluation on COCO dataset using various YOLO5 and YOLO7 variants, we show that our correction method improves quantized YOLO networks consistently on both object detection and segmentation tasks at low-precision (4-bit and 3-bit).

{{</citation>}}


## cs.DC (1)



### (51/108) MPGemmFI: A Fault Injection Technique for Mixed Precision GEMM in ML Applications (Bo Fang et al., 2023)

{{<citation>}}

Bo Fang, Xinyi Li, Harvey Dam, Cheng Tan, Siva Kumar Sastry Hari, Timothy Tsai, Ignacio Laguna, Dingwen Tao, Ganesh Gopalakrishnan, Prashant Nair, Kevin Barker, Ang Li. (2023)  
**MPGemmFI: A Fault Injection Technique for Mixed Precision GEMM in ML Applications**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2311.05782v1)  

---


**ABSTRACT**  
Emerging deep learning workloads urgently need fast general matrix multiplication (GEMM). To meet such demand, one of the critical features of machine-learning-specific accelerators such as NVIDIA Tensor Cores, AMD Matrix Cores, and Google TPUs is the support of mixed-precision enabled GEMM. For DNN models, lower-precision FP data formats and computation offer acceptable correctness but significant performance, area, and memory footprint improvement. While promising, the mixed-precision computation on error resilience remains unexplored. To this end, we develop a fault injection framework that systematically injects fault into the mixed-precision computation results. We investigate how the faults affect the accuracy of machine learning applications. Based on the error resilience characteristics, we offer lightweight error detection and correction solutions that significantly improve the overall model accuracy if the models experience hardware faults. The solutions can be efficiently integrated into the accelerator's pipelines.

{{</citation>}}


## eess.SY (1)



### (52/108) Real-time Control of Electric Autonomous Mobility-on-Demand Systems via Graph Reinforcement Learning (Aaryan Singhal et al., 2023)

{{<citation>}}

Aaryan Singhal, Daniele Gammelli, Justin Luke, Karthik Gopalakrishnan, Dominik Helmreich, Marco Pavone. (2023)  
**Real-time Control of Electric Autonomous Mobility-on-Demand Systems via Graph Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-RO, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.05780v1)  

---


**ABSTRACT**  
Operators of Electric Autonomous Mobility-on-Demand (E-AMoD) fleets need to make several real-time decisions such as matching available cars to ride requests, rebalancing idle cars to areas of high demand, and charging vehicles to ensure sufficient range. While this problem can be posed as a linear program that optimizes flows over a space-charge-time graph, the size of the resulting optimization problem does not allow for real-time implementation in realistic settings. In this work, we present the E-AMoD control problem through the lens of reinforcement learning and propose a graph network-based framework to achieve drastically improved scalability and superior performance over heuristics. Specifically, we adopt a bi-level formulation where we (1) leverage a graph network-based RL agent to specify a desired next state in the space-charge graph, and (2) solve more tractable linear programs to best achieve the desired state while ensuring feasibility. Experiments using real-world data from San Francisco and New York City show that our approach achieves up to 89% of the profits of the theoretically-optimal solution while achieving more than a 100x speedup in computational time. Furthermore, our approach outperforms the best domain-specific heuristics with comparable runtimes, with an increase in profits by up to 3x. Finally, we highlight promising zero-shot transfer capabilities of our learned policy on tasks such as inter-city generalization and service area expansion, thus showing the utility, scalability, and flexibility of our framework.

{{</citation>}}


## cs.CL (25)



### (53/108) Chatbots Are Not Reliable Text Annotators (Ross Deans Kristensen-McLachlan et al., 2023)

{{<citation>}}

Ross Deans Kristensen-McLachlan, Miceal Canavan, Márton Kardos, Mia Jacobsen, Lene Aarøe. (2023)  
**Chatbots Are Not Reliable Text Annotators**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.05769v1)  

---


**ABSTRACT**  
Recent research highlights the significant potential of ChatGPT for text annotation in social science research. However, ChatGPT is a closed-source product which has major drawbacks with regards to transparency, reproducibility, cost, and data protection. Recent advances in open-source (OS) large language models (LLMs) offer alternatives which remedy these challenges. This means that it is important to evaluate the performance of OS LLMs relative to ChatGPT and standard approaches to supervised machine learning classification. We conduct a systematic comparative evaluation of the performance of a range of OS LLM models alongside ChatGPT, using both zero- and few-shot learning as well as generic and custom prompts, with results compared to more traditional supervised classification models. Using a new dataset of Tweets from US news media, and focusing on simple binary text annotation tasks for standard social science concepts, we find significant variation in the performance of ChatGPT and OS models across the tasks, and that supervised classifiers consistently outperform both. Given the unreliable performance of ChatGPT and the significant challenges it poses to Open Science we advise against using ChatGPT for substantive text annotation tasks in social science research.

{{</citation>}}


### (54/108) Deep Natural Language Feature Learning for Interpretable Prediction (Felipe Urrutia et al., 2023)

{{<citation>}}

Felipe Urrutia, Cristian Buc, Valentin Barriere. (2023)  
**Deep Natural Language Feature Learning for Interpretable Prediction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model, NLI, Natural Language Inference  
[Paper Link](http://arxiv.org/abs/2311.05754v1)  

---


**ABSTRACT**  
We propose a general method to break down a main complex task into a set of intermediary easier sub-tasks, which are formulated in natural language as binary questions related to the final target task. Our method allows for representing each example by a vector consisting of the answers to these questions. We call this representation Natural Language Learned Features (NLLF). NLLF is generated by a small transformer language model (e.g., BERT) that has been trained in a Natural Language Inference (NLI) fashion, using weak labels automatically obtained from a Large Language Model (LLM). We show that the LLM normally struggles for the main task using in-context learning, but can handle these easiest subtasks and produce useful weak labels to train a BERT. The NLI-like training of the BERT allows for tackling zero-shot inference with any binary question, and not necessarily the ones seen during the training. We show that this NLLF vector not only helps to reach better performances by enhancing any classifier, but that it can be used as input of an easy-to-interpret machine learning model like a decision tree. This decision tree is interpretable but also reaches high performances, surpassing those of a pre-trained transformer in some cases.We have successfully applied this method to two completely different tasks: detecting incoherence in students' answers to open-ended mathematics exam questions, and screening abstracts for a systematic literature review of scientific papers on climate change and agroecology.

{{</citation>}}


### (55/108) Efficiently Adapting Pretrained Language Models To New Languages (Zoltan Csaki et al., 2023)

{{<citation>}}

Zoltan Csaki, Pian Pawakapan, Urmish Thakker, Qiantong Xu. (2023)  
**Efficiently Adapting Pretrained Language Models To New Languages**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2311.05741v1)  

---


**ABSTRACT**  
Recent large language models (LLM) exhibit sub-optimal performance on low-resource languages, as the training data of these models is usually dominated by English and other high-resource languages. Furthermore, it is challenging to train models for low-resource languages, especially from scratch, due to a lack of high quality training data. Adapting pretrained LLMs reduces the need for data in the new language while also providing cross lingual transfer capabilities. However, naively adapting to new languages leads to catastrophic forgetting and poor tokenizer efficiency. In this work, we study how to efficiently adapt any existing pretrained LLM to a new language without running into these issues. In particular, we improve the encoding efficiency of the tokenizer by adding new tokens from the target language and study the data mixing recipe to mitigate forgetting. Our experiments on adapting an English LLM to Hungarian and Thai show that our recipe can reach better performance than open source models on the target language, with minimal regressions on English.

{{</citation>}}


### (56/108) Long-Horizon Dialogue Understanding for Role Identification in the Game of Avalon with Large Language Models (Simon Stepputtis et al., 2023)

{{<citation>}}

Simon Stepputtis, Joseph Campbell, Yaqi Xie, Zhengyang Qi, Wenxin Sharon Zhang, Ruiyi Wang, Sanketh Rangreji, Michael Lewis, Katia Sycara. (2023)  
**Long-Horizon Dialogue Understanding for Role Identification in the Game of Avalon with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Dialog, Dialogue, Language Model, NLU  
[Paper Link](http://arxiv.org/abs/2311.05720v1)  

---


**ABSTRACT**  
Deception and persuasion play a critical role in long-horizon dialogues between multiple parties, especially when the interests, goals, and motivations of the participants are not aligned. Such complex tasks pose challenges for current Large Language Models (LLM) as deception and persuasion can easily mislead them, especially in long-horizon multi-party dialogues. To this end, we explore the game of Avalon: The Resistance, a social deduction game in which players must determine each other's hidden identities to complete their team's objective. We introduce an online testbed and a dataset containing 20 carefully collected and labeled games among human players that exhibit long-horizon deception in a cooperative-competitive setting. We discuss the capabilities of LLMs to utilize deceptive long-horizon conversations between six human players to determine each player's goal and motivation. Particularly, we discuss the multimodal integration of the chat between the players and the game's state that grounds the conversation, providing further insights into the true player identities. We find that even current state-of-the-art LLMs do not reach human performance, making our dataset a compelling benchmark to investigate the decision-making and language-processing capabilities of LLMs. Our dataset and online testbed can be found at our project website: https://sstepput.github.io/Avalon-NLU/

{{</citation>}}


### (57/108) Removing RLHF Protections in GPT-4 via Fine-Tuning (Qiusi Zhan et al., 2023)

{{<citation>}}

Qiusi Zhan, Richard Fang, Rohan Bindu, Akul Gupta, Tatsunori Hashimoto, Daniel Kang. (2023)  
**Removing RLHF Protections in GPT-4 via Fine-Tuning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.05553v1)  

---


**ABSTRACT**  
As large language models (LLMs) have increased in their capabilities, so does their potential for dual use. To reduce harmful outputs, produces and vendors of LLMs have used reinforcement learning with human feedback (RLHF). In tandem, LLM vendors have been increasingly enabling fine-tuning of their most powerful models. However, concurrent work has shown that fine-tuning can remove RLHF protections. We may expect that the most powerful models currently available (GPT-4) are less susceptible to fine-tuning attacks.   In this work, we show the contrary: fine-tuning allows attackers to remove RLHF protections with as few as 340 examples and a 95% success rate. These training examples can be automatically generated with weaker models. We further show that removing RLHF protections does not decrease usefulness on non-censored outputs, providing evidence that our fine-tuning strategy does not decrease usefulness despite using weaker models to generate training data. Our results show the need for further research on protections on LLMs.

{{</citation>}}


### (58/108) The Iron(ic) Melting Pot: Reviewing Human Evaluation in Humour, Irony and Sarcasm Generation (Tyler Loakman et al., 2023)

{{<citation>}}

Tyler Loakman, Aaron Maladry, Chenghua Lin. (2023)  
**The Iron(ic) Melting Pot: Reviewing Human Evaluation in Humour, Irony and Sarcasm Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2311.05552v1)  

---


**ABSTRACT**  
Human evaluation is often considered to be the gold standard method of evaluating a Natural Language Generation system. However, whilst its importance is accepted by the community at large, the quality of its execution is often brought into question. In this position paper, we argue that the generation of more esoteric forms of language - humour, irony and sarcasm - constitutes a subdomain where the characteristics of selected evaluator panels are of utmost importance, and every effort should be made to report demographic characteristics wherever possible, in the interest of transparency and replicability. We support these claims with an overview of each language form and an analysis of examples in terms of how their interpretation is affected by different participant variables. We additionally perform a critical survey of recent works in NLG to assess how well evaluation procedures are reported in this subdomain, and note a severe lack of open reporting of evaluator demographic information, and a significant reliance on crowdsourcing platforms for recruitment.

{{</citation>}}


### (59/108) Text Representation Distillation via Information Bottleneck Principle (Yanzhao Zhang et al., 2023)

{{<citation>}}

Yanzhao Zhang, Dingkun Long, Zehan Li, Pengjun Xie. (2023)  
**Text Representation Distillation via Information Bottleneck Principle**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Knowledge Distillation, Textual Similarity  
[Paper Link](http://arxiv.org/abs/2311.05472v1)  

---


**ABSTRACT**  
Pre-trained language models (PLMs) have recently shown great success in text representation field. However, the high computational cost and high-dimensional representation of PLMs pose significant challenges for practical applications. To make models more accessible, an effective method is to distill large models into smaller representation models. In order to relieve the issue of performance degradation after distillation, we propose a novel Knowledge Distillation method called IBKD. This approach is motivated by the Information Bottleneck principle and aims to maximize the mutual information between the final representation of the teacher and student model, while simultaneously reducing the mutual information between the student model's representation and the input data. This enables the student model to preserve important learned information while avoiding unnecessary information, thus reducing the risk of over-fitting. Empirical studies on two main downstream applications of text representation (Semantic Textual Similarity and Dense Retrieval tasks) demonstrate the effectiveness of our proposed approach.

{{</citation>}}


### (60/108) All Should Be Equal in the Eyes of Language Models: Counterfactually Aware Fair Text Generation (Pragyan Banerjee et al., 2023)

{{<citation>}}

Pragyan Banerjee, Abhinav Java, Surgan Jandial, Simra Shahid, Shaz Furniturewala, Balaji Krishnamurthy, Sumit Bhatia. (2023)  
**All Should Be Equal in the Eyes of Language Models: Counterfactually Aware Fair Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: Language Model, Text Generation  
[Paper Link](http://arxiv.org/abs/2311.05451v1)  

---


**ABSTRACT**  
Fairness in Language Models (LMs) remains a longstanding challenge, given the inherent biases in training data that can be perpetuated by models and affect the downstream tasks. Recent methods employ expensive retraining or attempt debiasing during inference by constraining model outputs to contrast from a reference set of biased templates or exemplars. Regardless, they dont address the primary goal of fairness to maintain equitability across different demographic groups. In this work, we posit that inferencing LMs to generate unbiased output for one demographic under a context ensues from being aware of outputs for other demographics under the same context. To this end, we propose Counterfactually Aware Fair InferencE (CAFIE), a framework that dynamically compares the model understanding of diverse demographics to generate more equitable sentences. We conduct an extensive empirical evaluation using base LMs of varying sizes and across three diverse datasets and found that CAFIE outperforms strong baselines. CAFIE produces fairer text and strikes the best balance between fairness and language modeling capability

{{</citation>}}


### (61/108) Cognitively Inspired Components for Social Conversational Agents (Alex Clay et al., 2023)

{{<citation>}}

Alex Clay, Eduardo Alonso, Esther Mondragón. (2023)  
**Cognitively Inspired Components for Social Conversational Agents**  

---
Primary Category: cs.CL  
Categories: I-2-0, cs-AI, cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.05450v1)  

---


**ABSTRACT**  
Current conversational agents (CA) have seen improvement in conversational quality in recent years due to the influence of large language models (LLMs) like GPT3. However, two key categories of problem remain. Firstly there are the unique technical problems resulting from the approach taken in creating the CA, such as scope with retrieval agents and the often nonsensical answers of former generative agents. Secondly, humans perceive CAs as social actors, and as a result expect the CA to adhere to social convention. Failure on the part of the CA in this respect can lead to a poor interaction and even the perception of threat by the user. As such, this paper presents a survey highlighting a potential solution to both categories of problem through the introduction of cognitively inspired additions to the CA. Through computational facsimiles of semantic and episodic memory, emotion, working memory, and the ability to learn, it is possible to address both the technical and social problems encountered by CAs.

{{</citation>}}


### (62/108) Mirror: A Universal Framework for Various Information Extraction Tasks (Tong Zhu et al., 2023)

{{<citation>}}

Tong Zhu, Junfei Ren, Zijian Yu, Mengsong Wu, Guoliang Zhang, Xiaoye Qu, Wenliang Chen, Zhefeng Wang, Baoxing Huai, Min Zhang. (2023)  
**Mirror: A Universal Framework for Various Information Extraction Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2311.05419v1)  

---


**ABSTRACT**  
Sharing knowledge between information extraction tasks has always been a challenge due to the diverse data formats and task variations. Meanwhile, this divergence leads to information waste and increases difficulties in building complex applications in real scenarios. Recent studies often formulate IE tasks as a triplet extraction problem. However, such a paradigm does not support multi-span and n-ary extraction, leading to weak versatility. To this end, we reorganize IE problems into unified multi-slot tuples and propose a universal framework for various IE tasks, namely Mirror. Specifically, we recast existing IE tasks as a multi-span cyclic graph extraction problem and devise a non-autoregressive graph decoding algorithm to extract all spans in a single step. It is worth noting that this graph structure is incredibly versatile, and it supports not only complex IE tasks, but also machine reading comprehension and classification tasks. We manually construct a corpus containing 57 datasets for model pretraining, and conduct experiments on 30 datasets across 8 downstream tasks. The experimental results demonstrate that our model has decent compatibility and outperforms or reaches competitive performance with SOTA systems under few-shot and zero-shot settings. The code, model weights, and pretraining corpus are available at https://github.com/Spico197/Mirror .

{{</citation>}}


### (63/108) Memorisation Cartography: Mapping out the Memorisation-Generalisation Continuum in Neural Machine Translation (Verna Dankers et al., 2023)

{{<citation>}}

Verna Dankers, Ivan Titov, Dieuwke Hupkes. (2023)  
**Memorisation Cartography: Mapping out the Memorisation-Generalisation Continuum in Neural Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.05379v1)  

---


**ABSTRACT**  
When training a neural network, it will quickly memorise some source-target mappings from your dataset but never learn some others. Yet, memorisation is not easily expressed as a binary feature that is good or bad: individual datapoints lie on a memorisation-generalisation continuum. What determines a datapoint's position on that spectrum, and how does that spectrum influence neural models' performance? We address these two questions for neural machine translation (NMT) models. We use the counterfactual memorisation metric to (1) build a resource that places 5M NMT datapoints on a memorisation-generalisation map, (2) illustrate how the datapoints' surface-level characteristics and a models' per-datum training signals are predictive of memorisation in NMT, (3) and describe the influence that subsets of that map have on NMT systems' performance.

{{</citation>}}


### (64/108) TencentLLMEval: A Hierarchical Evaluation of Real-World Capabilities for Human-Aligned LLMs (Shuyi Xie et al., 2023)

{{<citation>}}

Shuyi Xie, Wenlin Yao, Yong Dai, Shaobo Wang, Donlin Zhou, Lifeng Jin, Xinhua Feng, Pengzhi Wei, Yujie Lin, Zhichao Hu, Dong Yu, Zhengyou Zhang, Jing Nie, Yuhong Liu. (2023)  
**TencentLLMEval: A Hierarchical Evaluation of Real-World Capabilities for Human-Aligned LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.05374v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown impressive capabilities across various natural language tasks. However, evaluating their alignment with human preferences remains a challenge. To this end, we propose a comprehensive human evaluation framework to assess LLMs' proficiency in following instructions on diverse real-world tasks. We construct a hierarchical task tree encompassing 7 major areas covering over 200 categories and over 800 tasks, which covers diverse capabilities such as question answering, reasoning, multiturn dialogue, and text generation, to evaluate LLMs in a comprehensive and in-depth manner. We also design detailed evaluation standards and processes to facilitate consistent, unbiased judgments from human evaluators. A test set of over 3,000 instances is released, spanning different difficulty levels and knowledge domains. Our work provides a standardized methodology to evaluate human alignment in LLMs for both English and Chinese. We also analyze the feasibility of automating parts of evaluation with a strong LLM (GPT-4). Our framework supports a thorough assessment of LLMs as they are integrated into real-world applications. We have made publicly available the task tree, TencentLLMEval dataset, and evaluation methodology which have been demonstrated as effective in assessing the performance of Tencent Hunyuan LLMs. By doing so, we aim to facilitate the benchmarking of advances in the development of safe and human-aligned LLMs.

{{</citation>}}


### (65/108) Do personality tests generalize to Large Language Models? (Florian E. Dorner et al., 2023)

{{<citation>}}

Florian E. Dorner, Tom Sühr, Samira Samadi, Augustin Kelava. (2023)  
**Do personality tests generalize to Large Language Models?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.05297v1)  

---


**ABSTRACT**  
With large language models (LLMs) appearing to behave increasingly human-like in text-based interactions, it has become popular to attempt to evaluate various properties of these models using tests originally designed for humans. While re-using existing tests is a resource-efficient way to evaluate LLMs, careful adjustments are usually required to ensure that test results are even valid across human sub-populations. Thus, it is not clear to what extent different tests' validity generalizes to LLMs. In this work, we provide evidence that LLMs' responses to personality tests systematically deviate from typical human responses, implying that these results cannot be interpreted in the same way as human test results. Concretely, reverse-coded items (e.g. "I am introverted" vs "I am extraverted") are often both answered affirmatively by LLMs. In addition, variation across different prompts designed to "steer" LLMs to simulate particular personality types does not follow the clear separation into five independent personality factors from human samples. In light of these results, we believe it is important to pay more attention to tests' validity for LLMs before drawing strong conclusions about potentially ill-defined concepts like LLMs' "personality".

{{</citation>}}


### (66/108) DeeLM: Dependency-enhanced Large Language Model for Sentence Embeddings (Xianming Li et al., 2023)

{{<citation>}}

Xianming Li, Jing Li. (2023)  
**DeeLM: Dependency-enhanced Large Language Model for Sentence Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, Language Model, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2311.05296v1)  

---


**ABSTRACT**  
Recent studies have proposed using large language models (LLMs) for sentence embeddings. However, most existing LLMs are built with an autoregressive architecture that primarily captures forward dependencies while neglecting backward dependencies. Previous work has highlighted the importance of backward dependencies in improving sentence embeddings. To address this issue, in this paper, we first present quantitative evidence demonstrating the limited learning of backward dependencies in LLMs. Then, we propose a novel approach called Dependency-Enhanced Large Language Model (DeeLM) to improve sentence embeddings. Specifically, we found a turning point in LLMs, where surpassing specific LLM layers leads to a significant performance drop in the semantic textual similarity (STS) task. STS is a crucial task for evaluating sentence embeddings. We then extract the layers after the turning point to make them bidirectional, allowing for the learning of backward dependencies. Extensive experiments demonstrate that DeeLM outperforms baselines and achieves state-of-the-art performance across various STS tasks.

{{</citation>}}


### (67/108) A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions (Lei Huang et al., 2023)

{{<citation>}}

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, Ting Liu. (2023)  
**A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.05232v1)  

---


**ABSTRACT**  
The emergence of large language models (LLMs) has marked a significant breakthrough in natural language processing (NLP), leading to remarkable advancements in text understanding and generation. Nevertheless, alongside these strides, LLMs exhibit a critical tendency to produce hallucinations, resulting in content that is inconsistent with real-world facts or user inputs. This phenomenon poses substantial challenges to their practical deployment and raises concerns over the reliability of LLMs in real-world scenarios, which attracts increasing attention to detect and mitigate these hallucinations. In this survey, we aim to provide a thorough and in-depth overview of recent advances in the field of LLM hallucinations. We begin with an innovative taxonomy of LLM hallucinations, then delve into the factors contributing to hallucinations. Subsequently, we present a comprehensive overview of hallucination detection methods and benchmarks. Additionally, representative approaches designed to mitigate hallucinations are introduced accordingly. Finally, we analyze the challenges that highlight the current limitations and formulate open questions, aiming to delineate pathways for future research on hallucinations in LLMs.

{{</citation>}}


### (68/108) Large Language Models and Prompt Engineering for Biomedical Query Focused Multi-Document Summarisation (Diego Mollá, 2023)

{{<citation>}}

Diego Mollá. (2023)  
**Large Language Models and Prompt Engineering for Biomedical Query Focused Multi-Document Summarisation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2311.05169v1)  

---


**ABSTRACT**  
This paper reports on the use of prompt engineering and GPT-3.5 for biomedical query-focused multi-document summarisation. Using GPT-3.5 and appropriate prompts, our system achieves top ROUGE-F1 results in the task of obtaining short-paragraph-sized answers to biomedical questions in the 2023 BioASQ Challenge (BioASQ 11b). This paper confirms what has been observed in other domains: 1) Prompts that incorporated few-shot samples generally improved on their counterpart zero-shot variants; 2) The largest improvement was achieved by retrieval augmented generation. The fact that these prompts allow our top runs to rank within the top two runs of BioASQ 11b demonstrate the power of using adequate prompts for Large Language Models in general, and GPT-3.5 in particular, for query-focused summarisation.

{{</citation>}}


### (69/108) Enhancing Computation Efficiency in Large Language Models through Weight and Activation Quantization (Jangwhan Lee et al., 2023)

{{<citation>}}

Jangwhan Lee, Minsoo Kim, Seungcheol Baek, Seok Joong Hwang, Wonyong Sung, Jungwook Choi. (2023)  
**Enhancing Computation Efficiency in Large Language Models through Weight and Activation Quantization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model, QA, Quantization  
[Paper Link](http://arxiv.org/abs/2311.05161v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are proficient in natural language processing tasks, but their deployment is often restricted by extensive parameter sizes and computational demands. This paper focuses on post-training quantization (PTQ) in LLMs, specifically 4-bit weight and 8-bit activation (W4A8) quantization, to enhance computational efficiency -- a topic less explored compared to weight-only quantization. We present two innovative techniques: activation-quantization-aware scaling (AQAS) and sequence-length-aware calibration (SLAC) to enhance PTQ by considering the combined effects on weights and activations and aligning calibration sequence lengths to target tasks. Moreover, we introduce dINT, a hybrid data format combining integer and denormal representations, to address the underflow issue in W4A8 quantization, where small values are rounded to zero. Through rigorous evaluations of LLMs, including OPT and LLaMA, we demonstrate that our techniques significantly boost task accuracies to levels comparable with full-precision models. By developing arithmetic units compatible with dINT, we further confirm that our methods yield a 2$\times$ hardware efficiency improvement compared to 8-bit integer MAC unit.

{{</citation>}}


### (70/108) Weakly-supervised Deep Cognate Detection Framework for Low-Resourced Languages Using Morphological Knowledge of Closely-Related Languages (Koustava Goswami et al., 2023)

{{<citation>}}

Koustava Goswami, Priya Rani, Theodorus Fransen, John P. McCrae. (2023)  
**Weakly-supervised Deep Cognate Detection Framework for Low-Resourced Languages Using Morphological Knowledge of Closely-Related Languages**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Low-Resource  
[Paper Link](http://arxiv.org/abs/2311.05155v1)  

---


**ABSTRACT**  
Exploiting cognates for transfer learning in under-resourced languages is an exciting opportunity for language understanding tasks, including unsupervised machine translation, named entity recognition and information retrieval. Previous approaches mainly focused on supervised cognate detection tasks based on orthographic, phonetic or state-of-the-art contextual language models, which under-perform for most under-resourced languages. This paper proposes a novel language-agnostic weakly-supervised deep cognate detection framework for under-resourced languages using morphological knowledge from closely related languages. We train an encoder to gain morphological knowledge of a language and transfer the knowledge to perform unsupervised and weakly-supervised cognate detection tasks with and without the pivot language for the closely-related languages. While unsupervised, it overcomes the need for hand-crafted annotation of cognates. We performed experiments on different published cognate detection datasets across language families and observed not only significant improvement over the state-of-the-art but also our method outperformed the state-of-the-art supervised and unsupervised methods. Our model can be extended to a wide range of languages from any language family as it overcomes the requirement of the annotation of the cognate pairs for training. The code and dataset building scripts can be found at https://github.com/koustavagoswami/Weakly_supervised-Cognate_Detection

{{</citation>}}


### (71/108) Quranic Conversations: Developing a Semantic Search tool for the Quran using Arabic NLP Techniques (Yasser Shohoud et al., 2023)

{{<citation>}}

Yasser Shohoud, Maged Shoman, Sarah Abdelazim. (2023)  
**Quranic Conversations: Developing a Semantic Search tool for the Quran using Arabic NLP Techniques**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.05120v1)  

---


**ABSTRACT**  
The Holy Book of Quran is believed to be the literal word of God (Allah) as revealed to the Prophet Muhammad (PBUH) over a period of approximately 23 years. It is the book where God provides guidance on how to live a righteous and just life, emphasizing principles like honesty, compassion, charity and justice, as well as providing rules for personal conduct, family matters, business ethics and much more. However, due to constraints related to the language and the Quran organization, it is challenging for Muslims to get all relevant ayahs (verses) pertaining to a matter or inquiry of interest. Hence, we developed a Quran semantic search tool which finds the verses pertaining to the user inquiry or prompt. To achieve this, we trained several models on a large dataset of over 30 tafsirs, where typically each tafsir corresponds to one verse in the Quran and, using cosine similarity, obtained the tafsir tensor which is most similar to the prompt tensor of interest, which was then used to index for the corresponding ayah in the Quran. Using the SNxLM model, we were able to achieve a cosine similarity score as high as 0.97 which corresponds to the abdu tafsir for a verse relating to financial matters.

{{</citation>}}


### (72/108) Unsupervised Translation Quality Estimation Exploiting Synthetic Data and Pre-trained Multilingual Encoder (Yuto Kuroda et al., 2023)

{{<citation>}}

Yuto Kuroda, Atsushi Fujita, Tomoyuki Kajiwara, Takashi Ninomiya. (2023)  
**Unsupervised Translation Quality Estimation Exploiting Synthetic Data and Pre-trained Multilingual Encoder**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2311.05117v1)  

---


**ABSTRACT**  
Translation quality estimation (TQE) is the task of predicting translation quality without reference translations. Due to the enormous cost of creating training data for TQE, only a few translation directions can benefit from supervised training. To address this issue, unsupervised TQE methods have been studied. In this paper, we extensively investigate the usefulness of synthetic TQE data and pre-trained multilingual encoders in unsupervised sentence-level TQE, both of which have been proven effective in the supervised training scenarios. Our experiment on WMT20 and WMT21 datasets revealed that this approach can outperform other unsupervised TQE methods on high- and low-resource translation directions in predicting post-editing effort and human evaluation score, and some zero-resource translation directions in predicting post-editing effort.

{{</citation>}}


### (73/108) Conic10K: A Challenging Math Problem Understanding and Reasoning Dataset (Haoyi Wu et al., 2023)

{{<citation>}}

Haoyi Wu, Wenyang Hui, Yezeng Chen, Weiqi Wu, Kewei Tu, Yi Zhou. (2023)  
**Conic10K: A Challenging Math Problem Understanding and Reasoning Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4, NLP, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.05113v1)  

---


**ABSTRACT**  
Mathematical understanding and reasoning are crucial tasks for assessing the capabilities of artificial intelligence (AI). However, existing benchmarks either require just a few steps of reasoning, or only contain a small amount of data in one specific topic, making it hard to analyse AI's behaviour with reference to different problems within a specific topic in detail. In this work, we propose Conic10K, a challenging math problem dataset on conic sections in Chinese senior high school education. Our dataset contains various problems with different reasoning depths, while only the knowledge from conic sections is required. Since the dataset only involves a narrow range of knowledge, it is easy to separately analyse the knowledge a model possesses and the reasoning ability it has. For each problem, we provide a high-quality formal representation, the reasoning steps, and the final solution. Experiments show that existing large language models, including GPT-4, exhibit weak performance on complex reasoning. We hope that our findings could inspire more advanced techniques for precise natural language understanding and reasoning. Our dataset and codes are available at https://github.com/whyNLP/Conic10K.

{{</citation>}}


### (74/108) A Survey of Large Language Models in Medicine: Progress, Application, and Challenge (Hongjian Zhou et al., 2023)

{{<citation>}}

Hongjian Zhou, Boyang Gu, Xinyu Zou, Yiru Li, Sam S. Chen, Peilin Zhou, Junling Liu, Yining Hua, Chengfeng Mao, Xian Wu, Zheng Li, Fenglin Liu. (2023)  
**A Survey of Large Language Models in Medicine: Progress, Application, and Challenge**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.05112v1)  

---


**ABSTRACT**  
Large language models (LLMs), such as ChatGPT, have achieved substantial attention due to their impressive human language understanding and generation capabilities. Therefore, the application of LLMs in medicine to assist physicians and patient care emerges as a promising research direction in both artificial intelligence and clinical medicine. To this end, this survey provides a comprehensive overview of the current progress, applications, and challenges faced by LLMs in medicine. Specifically, we aim to address the following questions: 1) What are LLMs and how can medical LLMs be built? 2) What are the downstream performances of medical LLMs? 3) How can medical LLMs be utilized in real-world clinical practice? 4) What challenges arise from the use of medical LLMs? 5) How can we better construct and utilize medical LLMs? As a result, this survey aims to provide insights into the opportunities and challenges of LLMs in medicine and serve as a valuable resource for constructing practical and effective medical LLMs. A regularly updated list of practical guide resources of medical LLMs can be found at https://github.com/AI-in-Health/MedLLMsPracticalGuide.

{{</citation>}}


### (75/108) Legal-HNet: Mixing Legal Long-Context Tokens with Hartley Transform (Daniele Giofré et al., 2023)

{{<citation>}}

Daniele Giofré, Sneha Ghantasala. (2023)  
**Legal-HNet: Mixing Legal Long-Context Tokens with Hartley Transform**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Legal, NLP, Seq2Seq  
[Paper Link](http://arxiv.org/abs/2311.05089v1)  

---


**ABSTRACT**  
Since its introduction, the transformers architecture has seen great adoption in NLP applications, but it also has limitations. Although the self-attention mechanism allows for generating very rich representations of the input text, its effectiveness may be limited in specialized domains such as legal, where, for example, language models often have to process very long texts. In this paper, we explore alternatives to replace the attention-based layers with simpler token-mixing mechanisms: Hartley and Fourier transforms. Using these non-parametric techniques, we train models with long input documents from scratch in the legal domain setting. We also introduce a new hybrid Seq2Seq architecture, a no-attention-based encoder connected with an attention-based decoder, which performs quite well on existing summarization tasks with much less compute and memory requirements. We believe that similar, if not better performance, as in the case of long correlations of abstractive text summarization tasks, can be achieved by adopting these simpler infrastructures. This not only makes training models from scratch accessible to more people, but also contributes to the reduction of the carbon footprint during training.

{{</citation>}}


### (76/108) Characterizing Large Language Models as Rationalizers of Knowledge-intensive Tasks (Aditi Mishra et al., 2023)

{{<citation>}}

Aditi Mishra, Sajjadur Rahman, Hannah Kim, Kushan Mitra, Estevam Hruschka. (2023)  
**Characterizing Large Language Models as Rationalizers of Knowledge-intensive Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.05085v1)  

---


**ABSTRACT**  
Large language models (LLMs) are proficient at generating fluent text with minimal task-specific supervision. Yet, their ability to provide well-grounded rationalizations for knowledge-intensive tasks remains under-explored. Such tasks, like commonsense multiple-choice questions, require rationales based on world knowledge to support predictions and refute alternate options. We consider the task of generating knowledge-guided rationalization in natural language by using expert-written examples in a few-shot manner. Surprisingly, crowd-workers preferred knowledge-grounded rationales over crowdsourced rationalizations, citing their factuality, sufficiency, and comprehensive refutations. Although LLMs-generated rationales were preferable, further improvements in conciseness and novelty are required. In another study, we show how rationalization of incorrect model predictions erodes humans' trust in LLM-generated rationales. Motivated by these observations, we create a two-stage pipeline to review task predictions and eliminate potential incorrect decisions before rationalization, enabling trustworthy rationale generation.

{{</citation>}}


### (77/108) A Framework to Assess (Dis)agreement Among Diverse Rater Groups (Vinodkumar Prabhakaran et al., 2023)

{{<citation>}}

Vinodkumar Prabhakaran, Christopher Homan, Lora Aroyo, Alicia Parrish, Alex Taylor, Mark Díaz, Ding Wang. (2023)  
**A Framework to Assess (Dis)agreement Among Diverse Rater Groups**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05074v1)  

---


**ABSTRACT**  
Recent advancements in conversational AI have created an urgent need for safety guardrails that prevent users from being exposed to offensive and dangerous content. Much of this work relies on human ratings and feedback, but does not account for the fact that perceptions of offense and safety are inherently subjective and that there may be systematic disagreements between raters that align with their socio-demographic identities. Instead, current machine learning approaches largely ignore rater subjectivity and use gold standards that obscure disagreements (e.g., through majority voting). In order to better understand the socio-cultural leanings of such tasks, we propose a comprehensive disagreement analysis framework to measure systematic diversity in perspectives among different rater subgroups. We then demonstrate its utility by applying this framework to a dataset of human-chatbot conversations rated by a demographically diverse pool of raters. Our analysis reveals specific rater groups that have more diverse perspectives than the rest, and informs demographic axes that are crucial to consider for safety annotations.

{{</citation>}}


## cs.CY (2)



### (78/108) Bridging the Digital Divide: Performance Variation across Socio-Economic Factors in Vision-Language Models (Joan Nwatu et al., 2023)

{{<citation>}}

Joan Nwatu, Oana Ignat, Rada Mihalcea. (2023)  
**Bridging the Digital Divide: Performance Variation across Socio-Economic Factors in Vision-Language Models**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CL, cs-CV, cs-CY, cs.CY  
Keywords: AI, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.05746v1)  

---


**ABSTRACT**  
Despite the impressive performance of current AI models reported across various tasks, performance reports often do not include evaluations of how these models perform on the specific groups that will be impacted by these technologies. Among the minority groups under-represented in AI, data from low-income households are often overlooked in data collection and model evaluation. We evaluate the performance of a state-of-the-art vision-language model (CLIP) on a geo-diverse dataset containing household images associated with different income values (Dollar Street) and show that performance inequality exists among households of different income levels. Our results indicate that performance for the poorer groups is consistently lower than the wealthier groups across various topics and countries. We highlight insights that can help mitigate these issues and propose actionable steps for economic-level inclusive AI development. Code is available at https://github.com/MichiganNLP/Bridging_the_Digital_Divide.

{{</citation>}}


### (79/108) Combating Misinformation in the Age of LLMs: Opportunities and Challenges (Canyu Chen et al., 2023)

{{<citation>}}

Canyu Chen, Kai Shu. (2023)  
**Combating Misinformation in the Age of LLMs: Opportunities and Challenges**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.05656v1)  

---


**ABSTRACT**  
Misinformation such as fake news and rumors is a serious threat on information ecosystems and public trust. The emergence of Large Language Models (LLMs) has great potential to reshape the landscape of combating misinformation. Generally, LLMs can be a double-edged sword in the fight. On the one hand, LLMs bring promising opportunities for combating misinformation due to their profound world knowledge and strong reasoning abilities. Thus, one emergent question is: how to utilize LLMs to combat misinformation? On the other hand, the critical challenge is that LLMs can be easily leveraged to generate deceptive misinformation at scale. Then, another important question is: how to combat LLM-generated misinformation? In this paper, we first systematically review the history of combating misinformation before the advent of LLMs. Then we illustrate the current efforts and present an outlook for these two fundamental questions respectively. The goal of this survey paper is to facilitate the progress of utilizing LLMs for fighting misinformation and call for interdisciplinary efforts from different stakeholders for combating LLM-generated misinformation.

{{</citation>}}


## cs.NI (3)



### (80/108) Deep Learning Architecture for Network-Efficiency at the Edge (Akrit Mudvari et al., 2023)

{{<citation>}}

Akrit Mudvari, Antero Vainio, Iason Ofeidis, Sasu Tarkoma, Leandros Tassiulas. (2023)  
**Deep Learning Architecture for Network-Efficiency at the Edge**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05739v1)  

---


**ABSTRACT**  
The growing number of AI-driven applications in the mobile devices has led to solutions that integrate deep learning models with the available edge-cloud resources; due to multiple benefits such as reduction in on-device energy consumption, improved latency, improved network usage, and certain privacy improvements, split learning, where deep learning models are split away from the mobile device and computed in a distributed manner, has become an extensively explored topic. Combined with compression-aware methods where learning adapts to compression of communicated data, the benefits of this approach have further improved and could serve as an alternative to established approaches like federated learning methods. In this work, we develop an adaptive compression-aware split learning method ('deprune') to improve and train deep learning models so that they are much more network-efficient (use less network resources and are faster), which would make them ideal to deploy in weaker devices with the help of edge-cloud resources. This method is also extended ('prune') to very quickly train deep learning models, through a transfer learning approach, that trades off little accuracy for much more network-efficient inference abilities. We show that the 'deprune' method can reduce network usage by 4x when compared with a split-learning approach (that does not use our method) without loss of accuracy, while also improving accuracy over compression-aware split-learning by 4 percent. Lastly, we show that the 'prune' method can reduce the training time for certain models by up to 6x without affecting the accuracy when compared against a compression-aware split-learning approach.

{{</citation>}}


### (81/108) Joint SDN Synchronization and Controller Placement in Wireless Networks using Deep Reinforcement Learning (Akrit Mudvari et al., 2023)

{{<citation>}}

Akrit Mudvari, Leandros Tassiulas. (2023)  
**Joint SDN Synchronization and Controller Placement in Wireless Networks using Deep Reinforcement Learning**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.05582v1)  

---


**ABSTRACT**  
Software Defined Networking has afforded numerous benefits to the network users but there are certain persisting issues with this technology, two of which are scalability and privacy. The natural solution to overcoming these limitations is a distributed SDN controller architecture where multiple controllers are deployed over the network, with each controller orchestrating a certain segment of the network. However, since the centralized control is the key attribute of SDN that allows it to be so beneficial, a centralized logical view of the network will have to be maintained by each of these controllers; this can be done through synchronization of the distributed controllers, where each controller communicates with the others to ensure that they remain informed about the entire network. There is however a network cost associated with constantly having to update each others about different aspects of the network, which will become a greater issue in dynamic wireless networks. To minimize this network cost, there is a need to consider not only when to get the update information from the neighboring controllers, but also where to dynamically place the controllers such that the network costs may be minimized. The placement should take into consideration both communication for synchronization among the distributed controllers and communication of the controllers with the network devices that they manage. In this work, we show that our multi-objective deep reinforcement learning-based method performs the best at achieving different application goals by developing policy for controller synchronization as well as placement, outperforming different other possible approaches, under a wide variety of network conditions.

{{</citation>}}


### (82/108) Atom: Neural Traffic Compression with Spatio-Temporal Graph Neural Networks (Paul Almasan et al., 2023)

{{<citation>}}

Paul Almasan, Krzysztof Rusek, Shihan Xiao, Xiang Shi, Xiangle Cheng, Albert Cabellos-Aparicio, Pere Barlet-Ros. (2023)  
**Atom: Neural Traffic Compression with Spatio-Temporal Graph Neural Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.05337v1)  

---


**ABSTRACT**  
Storing network traffic data is key to efficient network management; however, it is becoming more challenging and costly due to the ever-increasing data transmission rates, traffic volumes, and connected devices. In this paper, we explore the use of neural architectures for network traffic compression. Specifically, we consider a network scenario with multiple measurement points in a network topology. Such measurements can be interpreted as multiple time series that exhibit spatial and temporal correlations induced by network topology, routing, or user behavior. We present \textit{Atom}, a neural traffic compression method that leverages spatial and temporal correlations present in network traffic. \textit{Atom} implements a customized spatio-temporal graph neural network design that effectively exploits both types of correlations simultaneously. The experimental results show that \textit{Atom} can outperform GZIP's compression ratios by 50\%-65\% on three real-world networks.

{{</citation>}}


## cs.CR (4)



### (83/108) LogShield: A Transformer-based APT Detection System Leveraging Self-Attention (Sihat Afnan et al., 2023)

{{<citation>}}

Sihat Afnan, Mushtari Sadia, Shahrear Iqbal, Anindya Iqbal. (2023)  
**LogShield: A Transformer-based APT Detection System Leveraging Self-Attention**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Attention, BERT, LSTM, Self-Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2311.05733v1)  

---


**ABSTRACT**  
Cyber attacks are often identified using system and network logs. There have been significant prior works that utilize provenance graphs and ML techniques to detect attacks, specifically advanced persistent threats, which are very difficult to detect. Lately, there have been studies where transformer-based language models are being used to detect various types of attacks from system logs. However, no such attempts have been made in the case of APTs. In addition, existing state-of-the-art techniques that use system provenance graphs, lack a data processing framework generalized across datasets for optimal performance. For mitigating this limitation as well as exploring the effectiveness of transformer-based language models, this paper proposes LogShield, a framework designed to detect APT attack patterns leveraging the power of self-attention in transformers. We incorporate customized embedding layers to effectively capture the context of event sequences derived from provenance graphs. While acknowledging the computational overhead associated with training transformer networks, our framework surpasses existing LSTM and Language models regarding APT detection. We integrated the model parameters and training procedure from the RoBERTa model and conducted extensive experiments on well-known APT datasets (DARPA OpTC and DARPA TC E3). Our framework achieved superior F1 scores of 98% and 95% on the two datasets respectively, surpassing the F1 scores of 96% and 94% obtained by LSTM models. Our findings suggest that LogShield's performance benefits from larger datasets and demonstrates its potential for generalization across diverse domains. These findings contribute to the advancement of APT attack detection methods and underscore the significance of transformer-based architectures in addressing security challenges in computer systems.

{{</citation>}}


### (84/108) FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts (Yichen Gong et al., 2023)

{{<citation>}}

Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, Xiaoyun Wang. (2023)  
**FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs.CR  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.05608v1)  

---


**ABSTRACT**  
Large vision-language models (VLMs) like GPT-4V represent an unprecedented revolution in the field of artificial intelligence (AI). Compared to single-modal large language models (LLMs), VLMs possess more versatile capabilities by incorporating additional modalities (e.g., images). Meanwhile, there's a rising enthusiasm in the AI community to develop open-source VLMs, such as LLaVA and MiniGPT4, which, however, have not undergone rigorous safety assessment. In this paper, to demonstrate that more modalities lead to unforeseen AI safety issues, we propose FigStep, a novel jailbreaking framework against VLMs. FigStep feeds harmful instructions into VLMs through the image channel and then uses benign text prompts to induce VLMs to output contents that violate common AI safety policies. Our experimental results show that FigStep can achieve an average attack success rate of 94.8% across 2 families of popular open-source VLMs, LLaVA and MiniGPT4 (a total of 5 VLMs). Moreover, we demonstrate that the methodology of FigStep can even jailbreak GPT-4V, which already leverages several system-level mechanisms to filter harmful queries. Above all, our experimental results reveal that VLMs are vulnerable to jailbreaking attacks, which highlights the necessity of novel safety alignments between visual and textual modalities.

{{</citation>}}


### (85/108) ChatGPT and other Large Language Models for Cybersecurity of Smart Grid Applications (Aydin Zaboli et al., 2023)

{{<citation>}}

Aydin Zaboli, Seong Lok Choi, Tai-Jin Song, Junho Hong. (2023)  
**ChatGPT and other Large Language Models for Cybersecurity of Smart Grid Applications**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SY, cs.CR, eess-SY  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.05462v1)  

---


**ABSTRACT**  
Cybersecurity breaches targeting electrical substations constitute a significant threat to the integrity of the power grid, necessitating comprehensive defense and mitigation strategies. Any anomaly in information and communication technology (ICT) should be detected for secure communications between devices in digital substations. This paper proposes large language models (LLM), e.g., ChatGPT, for the cybersecurity of IEC 61850-based digital substation communications. Multicast messages such as generic object oriented substation event (GOOSE) and sampled value (SV) are used for case studies. The proposed LLM-based cybersecurity framework includes for the first time data pre-processing of communication systems and human-in-the-loop (HITL) training (considering the cybersecurity guidelines recommended by humans). The results show a comparative analysis of detected anomaly data carried out based on the performance evaluation metrics for different LLMs. A hardware-in-the-loop (HIL) testbed is used to generate and extract a dataset of IEC 61850 communications.

{{</citation>}}


### (86/108) RAGLog: Log Anomaly Detection using Retrieval Augmented Generation (Jonathan Pan et al., 2023)

{{<citation>}}

Jonathan Pan, Swee Liang Wong, Yidi Yuan. (2023)  
**RAGLog: Log Anomaly Detection using Retrieval Augmented Generation**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Anomaly Detection, Language Model  
[Paper Link](http://arxiv.org/abs/2311.05261v1)  

---


**ABSTRACT**  
The ability to detect log anomalies from system logs is a vital activity needed to ensure cyber resiliency of systems. It is applied for fault identification or facilitate cyber investigation and digital forensics. However, as logs belonging to different systems and components differ significantly, the challenge to perform such analysis is humanly challenging from the volume, variety and velocity of logs. This is further complicated by the lack or unavailability of anomalous log entries to develop trained machine learning or artificial intelligence models for such purposes. In this research work, we explore the use of a Retrieval Augmented Large Language Model that leverages a vector database to detect anomalies from logs. We used a Question and Answer configuration pipeline. To the best of our knowledge, our experiment which we called RAGLog is a novel one and the experimental results show much promise.

{{</citation>}}


## cs.SI (2)



### (87/108) Susceptibility to Unreliable Information Sources: Swift Adoption with Minimal Exposure (Jinyi Ye et al., 2023)

{{<citation>}}

Jinyi Ye, Luca Luceri, Julie Jiang, Emilio Ferrara. (2023)  
**Susceptibility to Unreliable Information Sources: Swift Adoption with Minimal Exposure**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2311.05724v1)  

---


**ABSTRACT**  
Misinformation proliferation on social media platforms is a pervasive threat to the integrity of online public discourse. Genuine users, susceptible to others' influence, often unknowingly engage with, endorse, and re-share questionable pieces of information, collectively amplifying the spread of misinformation. In this study, we introduce an empirical framework to investigate users' susceptibility to influence when exposed to unreliable and reliable information sources. Leveraging two datasets on political and public health discussions on Twitter, we analyze the impact of exposure on the adoption of information sources, examining how the reliability of the source modulates this relationship. Our findings provide evidence that increased exposure augments the likelihood of adoption. Users tend to adopt low-credibility sources with fewer exposures than high-credibility sources, a trend that persists even among non-partisan users. Furthermore, the number of exposures needed for adoption varies based on the source credibility, with extreme ends of the spectrum (very high or low credibility) requiring fewer exposures for adoption. Additionally, we reveal that the adoption of information sources often mirrors users' prior exposure to sources with comparable credibility levels. Our research offers critical insights for mitigating the endorsement of misinformation by vulnerable users, offering a framework to study the dynamics of content exposure and adoption on social media platforms.

{{</citation>}}


### (88/108) News and Misinformation Consumption in Europe: A Longitudinal Cross-Country Perspective (Anees Baqir et al., 2023)

{{<citation>}}

Anees Baqir, Alessandro Galeazzi, Fabiana Zollo. (2023)  
**News and Misinformation Consumption in Europe: A Longitudinal Cross-Country Perspective**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2311.05487v2)  

---


**ABSTRACT**  
The Internet and social media have transformed news availability and accessibility, reshaping information consumption and production. However, they can also facilitate the rapid spread of misinformation, posing significant societal challenges. To combat misinformation effectively, it is crucial to understand the online information environment and news consumption patterns. Most existing research has primarily focused on single topics or individual countries, lacking cross-country comparisons. This study investigated information consumption in four European countries, analyzing three years of Twitter activity from news outlet accounts in France, Germany, Italy, and the UK and focusing on the role of misinformation sources. Our work offers a perspective on how topics of European significance are interpreted across various countries. Results indicate that reliable sources dominate the information landscape, although unreliable content is still present across all countries and topics. While most users engage with reliable sources, a small percentage consume questionable content. Interestingly, few users have a mixed information diet, bridging the gap between questionable and reliable news in the similarity network. Cross-country comparisons revealed differences in audience overlap of news sources, offering valuable guidance for policymakers and scholars in developing effective and tailored solutions to combat misinformation.

{{</citation>}}


## cs.SD (1)



### (89/108) What Do I Hear? Generating Sounds for Visuals with ChatGPT (David Chuan-En Lin et al., 2023)

{{<citation>}}

David Chuan-En Lin, Nikolas Martelaro. (2023)  
**What Do I Hear? Generating Sounds for Visuals with ChatGPT**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.05609v1)  

---


**ABSTRACT**  
This short paper introduces a workflow for generating realistic soundscapes for visual media. In contrast to prior work, which primarily focus on matching sounds for on-screen visuals, our approach extends to suggesting sounds that may not be immediately visible but are essential to crafting a convincing and immersive auditory environment. Our key insight is leveraging the reasoning capabilities of language models, such as ChatGPT. In this paper, we describe our workflow, which includes creating a scene context, brainstorming sounds, and generating the sounds.

{{</citation>}}


## cs.HC (2)



### (90/108) Conversational AI Threads for Visualizing Multidimensional Datasets (Matt-Heun Hong et al., 2023)

{{<citation>}}

Matt-Heun Hong, Anamaria Crisan. (2023)  
**Conversational AI Threads for Visualizing Multidimensional Datasets**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.05590v1)  

---


**ABSTRACT**  
Generative Large Language Models (LLMs) show potential in data analysis, yet their full capabilities remain uncharted. Our work explores the capabilities of LLMs for creating and refining visualizations via conversational interfaces. We used an LLM to conduct a re-analysis of a prior Wizard-of-Oz study examining the use of chatbots for conducting visual analysis. We surfaced the strengths and weaknesses of LLM-driven analytic chatbots, finding that they fell short in supporting progressive visualization refinements. From these findings, we developed AI Threads, a multi-threaded analytic chatbot that enables analysts to proactively manage conversational context and improve the efficacy of its outputs. We evaluate its usability through a crowdsourced study (n=40) and in-depth interviews with expert analysts (n=10). We further demonstrate the capabilities of AI Threads on a dataset outside the LLM's training corpus. Our findings show the potential of LLMs while also surfacing challenges and fruitful avenues for future research.

{{</citation>}}


### (91/108) What is prompt literacy? An exploratory study of language learners' development of new literacy skill using generative AI (Yohan Hwang et al., 2023)

{{<citation>}}

Yohan Hwang, Jang Ho Lee, Dongkwang Shin. (2023)  
**What is prompt literacy? An exploratory study of language learners' development of new literacy skill using generative AI**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05373v1)  

---


**ABSTRACT**  
In the current study,we propose that, in the era of generative AI, there is now a new form of literacy called "prompt literacy," which refers to the ability to generate precise prompts as input for AI systems, interpret the outputs, and iteratively refine prompts to achieve desired results. To explore the emergence and development of this literacy skill, the current study examined 30 EFL students' engagement in an AI-powered image creation project, through which they created artworks representing the socio-cultural meanings of English words by iteratively drafting and refining prompts in generative AI tools. By examining AI-generated images and the participants' drafting and revision of their prompts, this study demonstrated the emergence of learners' prompt literacy skills. The survey data further showed the participants' perceived improvement in their vocabulary learning strategies as a result of engaging in the target AI-powered project. In addition, the participants' post-project reflection revealed three benefits of developing prompt literacy: enjoyment from manifesting imagined outcomes; recognition of its importance for communication, problem-solving and career development; and the enhanced understanding of the collaborative nature of human-AI interaction. These findings suggest that prompt literacy is an increasingly crucial literacy for the AI era.

{{</citation>}}


## quant-ph (2)



### (92/108) Multi-Agent Quantum Reinforcement Learning using Evolutionary Optimization (Michael Kölle et al., 2023)

{{<citation>}}

Michael Kölle, Felix Topp, Thomy Phan, Philipp Altmann, Jonas Nüßlein, Claudia Linnhoff-Popien. (2023)  
**Multi-Agent Quantum Reinforcement Learning using Evolutionary Optimization**  

---
Primary Category: quant-ph  
Categories: cs-AI, cs-MA, quant-ph, quant-ph  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.05546v1)  

---


**ABSTRACT**  
Multi-Agent Reinforcement Learning is becoming increasingly more important in times of autonomous driving and other smart industrial applications. Simultaneously a promising new approach to Reinforcement Learning arises using the inherent properties of quantum mechanics, reducing the trainable parameters of a model significantly. However, gradient-based Multi-Agent Quantum Reinforcement Learning methods often have to struggle with barren plateaus, holding them back from matching the performance of classical approaches. We build upon a existing approach for gradient free Quantum Reinforcement Learning and propose tree approaches with Variational Quantum Circuits for Multi-Agent Reinforcement Learning using evolutionary optimization. We evaluate our approach in the Coin Game environment and compare them to classical approaches. We showed that our Variational Quantum Circuit approaches perform significantly better compared to a neural network with a similar amount of trainable parameters. Compared to the larger neural network, our approaches archive similar results using $97.88\%$ less parameters.

{{</citation>}}


### (93/108) Towards Quantum-Native Communication Systems: New Developments, Trends, and Challenges (Xiaolin Zhou et al., 2023)

{{<citation>}}

Xiaolin Zhou, Anqi Shen, Shuyan Hu, Wei Ni, Xin Wang, Ekram Hossain, Lajos Hanzo. (2023)  
**Towards Quantum-Native Communication Systems: New Developments, Trends, and Challenges**  

---
Primary Category: quant-ph  
Categories: cs-IT, math-IT, quant-ph, quant-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05239v1)  

---


**ABSTRACT**  
The potential synergy between quantum communications and future wireless communication systems is explored. By proposing a quantum-native or quantum-by-design philosophy, the survey examines technologies such as quantum-domain (QD) multi-input multi-output (MIMO), QD non-orthogonal multiple access (NOMA), quantum secure direct communication (QSDC), QD resource allocation, QD routing, and QD artificial intelligence (AI). The recent research advances in these areas are summarized. Given the behavior of photonic and particle-like Terahertz (THz) systems, a comprehensive system-oriented perspective is adopted to assess the feasibility of using quantum communications in future systems. This survey also reviews quantum optimization algorithms and quantum neural networks to explore the potential integration of quantum communication and quantum computing in future systems. Additionally, the current status of quantum sensing, quantum radar, and quantum timing is briefly reviewed in support of future applications. The associated research gaps and future directions are identified, including extending the entanglement coherence time, developing THz quantum communications devices, addressing challenges in channel estimation and tracking, and establishing the theoretical bounds and performance trade-offs of quantum communication, computing, and sensing. This survey offers a unique perspective on the potential for quantum communications to revolutionize future systems and pave the way for even more advanced technologies.

{{</citation>}}


## stat.ML (1)



### (94/108) Dirichlet Active Learning (Kevin Miller et al., 2023)

{{<citation>}}

Kevin Miller, Ryan Murray. (2023)  
**Dirichlet Active Learning**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2311.05501v1)  

---


**ABSTRACT**  
This work introduces Dirichlet Active Learning (DiAL), a Bayesian-inspired approach to the design of active learning algorithms. Our framework models feature-conditional class probabilities as a Dirichlet random field and lends observational strength between similar features in order to calibrate the random field. This random field can then be utilized in learning tasks: in particular, we can use current estimates of mean and variance to conduct classification and active learning in the context where labeled data is scarce. We demonstrate the applicability of this model to low-label rate graph learning by constructing ``propagation operators'' based upon the graph Laplacian, and offer computational studies demonstrating the method's competitiveness with the state of the art. Finally, we provide rigorous guarantees regarding the ability of this approach to ensure both exploration and exploitation, expressed respectively in terms of cluster exploration and increased attention to decision boundaries.

{{</citation>}}


## cs.AI (4)



### (95/108) General Policies, Subgoal Structure, and Planning Width (Blai Bonet et al., 2023)

{{<citation>}}

Blai Bonet, Hector Geffner. (2023)  
**General Policies, Subgoal Structure, and Planning Width**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2311.05490v1)  

---


**ABSTRACT**  
It has been observed that many classical planning domains with atomic goals can be solved by means of a simple polynomial exploration procedure, called IW, that runs in time exponential in the problem width, which in these cases is bounded and small. Yet, while the notion of width has become part of state-of-the-art planning algorithms such as BFWS, there is no good explanation for why so many benchmark domains have bounded width when atomic goals are considered. In this work, we address this question by relating bounded width with the existence of general optimal policies that in each planning instance are represented by tuples of atoms of bounded size. We also define the notions of (explicit) serializations and serialized width that have a broader scope as many domains have a bounded serialized width but no bounded width. Such problems are solved non-optimally in polynomial time by a suitable variant of the Serialized IW algorithm. Finally, the language of general policies and the semantics of serializations are combined to yield a simple, meaningful, and expressive language for specifying serializations in compact form in the form of sketches, which can be used for encoding domain control knowledge by hand or for learning it from small examples. Sketches express general problem decompositions in terms of subgoals, and sketches of bounded width express problem decompositions that can be solved in polynomial time.

{{</citation>}}


### (96/108) Kantian Deontology Meets AI Alignment: Towards Morally Robust Fairness Metrics (Carlos Mougan et al., 2023)

{{<citation>}}

Carlos Mougan, Joshua Brand. (2023)  
**Kantian Deontology Meets AI Alignment: Towards Morally Robust Fairness Metrics**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05227v1)  

---


**ABSTRACT**  
Deontological ethics, specifically understood through Immanuel Kant, provides a moral framework that emphasizes the importance of duties and principles, rather than the consequences of action. Understanding that despite the prominence of deontology, it is currently an overlooked approach in fairness metrics, this paper explores the compatibility of a Kantian deontological framework in fairness metrics, part of the AI alignment field. We revisit Kant's critique of utilitarianism, which is the primary approach in AI fairness metrics and argue that fairness principles should align with the Kantian deontological framework. By integrating Kantian ethics into AI alignment, we not only bring in a widely-accepted prominent moral theory but also strive for a more morally grounded AI landscape that better balances outcomes and procedures in pursuit of fairness and justice.

{{</citation>}}


### (97/108) An Experiment in Retrofitting Competency Questions for Existing Ontologies (Reham Alharbi et al., 2023)

{{<citation>}}

Reham Alharbi, Valentina Tamma, Floriana Grasso, Terry Payne. (2023)  
**An Experiment in Retrofitting Competency Questions for Existing Ontologies**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Generative AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.05662v1)  

---


**ABSTRACT**  
Competency Questions (CQs) are a form of ontology functional requirements expressed as natural language questions. Inspecting CQs together with the axioms in an ontology provides critical insights into the intended scope and applicability of the ontology. CQs also underpin a number of tasks in the development of ontologies e.g. ontology reuse, ontology testing, requirement specification, and the definition of patterns that implement such requirements. Although CQs are integral to the majority of ontology engineering methodologies, the practice of publishing CQs alongside the ontological artefacts is not widely observed by the community. In this context, we present an experiment in retrofitting CQs from existing ontologies. We propose RETROFIT-CQs, a method to extract candidate CQs directly from ontologies using Generative AI. In the paper we present the pipeline that facilitates the extraction of CQs by leveraging Large Language Models (LLMs) and we discuss its application to a number of existing ontologies.

{{</citation>}}


### (98/108) Lumos: Learning Agents with Unified Data, Modular Design, and Open-Source LLMs (Da Yin et al., 2023)

{{<citation>}}

Da Yin, Faeze Brahman, Abhilasha Ravichander, Khyathi Chandu, Kai-Wei Chang, Yejin Choi, Bill Yuchen Lin. (2023)  
**Lumos: Learning Agents with Unified Data, Modular Design, and Open-Source LLMs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.05657v1)  

---


**ABSTRACT**  
We introduce Lumos, a novel framework for training language agents that employs a unified data format and a modular architecture based on open-source large language models (LLMs). Lumos consists of three distinct modules: planning, grounding, and execution. The planning module breaks down a task into a series of high-level, tool-agnostic subgoals, which are then made specific by the grounding module through a set of low-level actions. These actions are subsequently executed by the execution module, utilizing a range of off-the-shelf tools and APIs. In order to train these modules effectively, high-quality annotations of subgoals and actions were collected and are made available for fine-tuning open-source LLMs for various tasks such as complex question answering, web tasks, and math problems. Leveraging this unified data and modular design, Lumos not only achieves comparable or superior performance to current, state-of-the-art agents, but also exhibits several key advantages: (1) Lumos surpasses GPT-4/3.5-based agents in complex question answering and web tasks, while equalling the performance of significantly larger LLM agents on math tasks; (2) Lumos outperforms open-source agents created through conventional training methods and those using chain-of-thoughts training; and (3) Lumos is capable of effectively generalizing to unseen interactive tasks, outperforming larger LLM-based agents and even exceeding performance of specialized agents.

{{</citation>}}


## eess.IV (2)



### (99/108) Using ResNet to Utilize 4-class T2-FLAIR Slice Classification Based on the Cholinergic Pathways Hyperintensities Scale for Pathological Aging (Wei-Chun Kevin Tsai et al., 2023)

{{<citation>}}

Wei-Chun Kevin Tsai, Yi-Chien Liu, Ming-Chun Yu, Chia-Ju Chou, Sui-Hing Yan, Yang-Teng Fan, Yan-Hsiang Huang, Yen-Ling Chiu, Yi-Fang Chuang, Ran-Zan Wang, Yao-Chia Shih. (2023)  
**Using ResNet to Utilize 4-class T2-FLAIR Slice Classification Based on the Cholinergic Pathways Hyperintensities Scale for Pathological Aging**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05477v1)  

---


**ABSTRACT**  
The Cholinergic Pathways Hyperintensities Scale (CHIPS) is a visual rating scale used to assess the extent of cholinergic white matter hyperintensities in T2-FLAIR images, serving as an indicator of dementia severity. However, the manual selection of four specific slices for rating throughout the entire brain is a time-consuming process. Our goal was to develop a deep learning-based model capable of automatically identifying the four slices relevant to CHIPS. To achieve this, we trained a 4-class slice classification model (BSCA) using the ADNI T2-FLAIR dataset (N=150) with the assistance of ResNet. Subsequently, we tested the model's performance on a local dataset (N=30). The results demonstrated the efficacy of our model, with an accuracy of 99.82% and an F1-score of 99.83%. This achievement highlights the potential impact of BSCA as an automatic screening tool, streamlining the selection of four specific T2-FLAIR slices that encompass white matter landmarks along the cholinergic pathways. Clinicians can leverage this tool to assess the risk of clinical dementia development efficiently.

{{</citation>}}


### (100/108) Transformer-based Model for Oral Epithelial Dysplasia Segmentation (Adam J Shephard et al., 2023)

{{<citation>}}

Adam J Shephard, Hanya Mahmood, Shan E Ahmed Raza, Anna Luiza Damaceno Araujo, Alan Roger Santos-Silva, Marcio Ajudarte Lopes, Pablo Agustin Vargas, Kris McCombe, Stephanie Craig, Jacqueline James, Jill Brooks, Paul Nankivell, Hisham Mehanna, Syed Ali Khurram, Nasir M Rajpoot. (2023)  
**Transformer-based Model for Oral Epithelial Dysplasia Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.05452v1)  

---


**ABSTRACT**  
Oral epithelial dysplasia (OED) is a premalignant histopathological diagnosis given to lesions of the oral cavity. OED grading is subject to large inter/intra-rater variability, resulting in the under/over-treatment of patients. We developed a new Transformer-based pipeline to improve detection and segmentation of OED in haematoxylin and eosin (H&E) stained whole slide images (WSIs). Our model was trained on OED cases (n = 260) and controls (n = 105) collected using three different scanners, and validated on test data from three external centres in the United Kingdom and Brazil (n = 78). Our internal experiments yield a mean F1-score of 0.81 for OED segmentation, which reduced slightly to 0.71 on external testing, showing good generalisability, and gaining state-of-the-art results. This is the first externally validated study to use Transformers for segmentation in precancerous histology images. Our publicly available model shows great promise to be the first step of a fully-integrated pipeline, allowing earlier and more efficient OED diagnosis, ultimately benefiting patient outcomes.

{{</citation>}}


## cs.CC (1)



### (101/108) On the Complexity of the Virtual Network Embedding in Specific Tree Topologies (Sergey Pankratov et al., 2023)

{{<citation>}}

Sergey Pankratov, Vitaly Aksenov, Stefan Schmid. (2023)  
**On the Complexity of the Virtual Network Embedding in Specific Tree Topologies**  

---
Primary Category: cs.CC  
Categories: cs-CC, cs-NI, cs.CC  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.05474v1)  

---


**ABSTRACT**  
Virtual networks are an innovative abstraction that extends cloud computing concepts to the network: by supporting bandwidth reservations between compute nodes (e.g., virtual machines), virtual networks can provide a predictable performance to distributed and communication-intensive cloud applications. However, in order to make the most efficient use of the shared resources, the Virtual Network Embedding (VNE) problem has to be solved: a virtual network should be mapped onto the given physical network so that resource reservations are minimized. The problem has been studied intensively already and is known to be NP-hard in general. In this paper, we revisit this problem and consider it on specific topologies, as they often arise in practice. To be more precise, we study the weighted version of the VNE problem: we consider a virtual weighted network of a specific topology which we want to embed onto a weighted network with capacities and specific topology. As for topologies, we consider most fundamental and commonly used ones: line, star, $2$-tiered star, oversubscribed $2$-tiered star, and tree, in addition to also considering arbitrary topologies. We show that typically the VNE problem is NP-hard even in more specialized cases, however, sometimes there exists a polynomial algorithm: for example, an embedding of the oversubscribed $2$-tiered star onto the tree is polynomial while an embedding of an arbitrary $2$-tiered star is not.

{{</citation>}}


## cs.RO (1)



### (102/108) TLCFuse: Temporal Multi-Modality Fusion Towards Occlusion-Aware Semantic Segmentation-Aided Motion Planning (Gustavo Salazar-Gomez et al., 2023)

{{<citation>}}

Gustavo Salazar-Gomez, Wenqian Liu, Manuel Diaz-Zapata, David Sierra-Gonzalez, Christian Laugier. (2023)  
**TLCFuse: Temporal Multi-Modality Fusion Towards Occlusion-Aware Semantic Segmentation-Aided Motion Planning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.05319v1)  

---


**ABSTRACT**  
In autonomous driving, addressing occlusion scenarios is crucial yet challenging. Robust surrounding perception is essential for handling occlusions and aiding motion planning. State-of-the-art models fuse Lidar and Camera data to produce impressive perception results, but detecting occluded objects remains challenging. In this paper, we emphasize the crucial role of temporal cues by integrating them alongside these modalities to address this challenge. We propose a novel approach for bird's eye view semantic grid segmentation, that leverages sequential sensor data to achieve robustness against occlusions. Our model extracts information from the sensor readings using attention operations and aggregates this information into a lower-dimensional latent representation, enabling thus the processing of multi-step inputs at each prediction step. Moreover, we show how it can also be directly applied to forecast the development of traffic scenes and be seamlessly integrated into a motion planner for trajectory planning. On the semantic segmentation tasks, we evaluate our model on the nuScenes dataset and show that it outperforms other baselines, with particularly large differences when evaluating on occluded and partially-occluded vehicles. Additionally, on motion planning task we are among the early teams to train and evaluate on nuPlan, a cutting-edge large-scale dataset for motion planning.

{{</citation>}}


## cs.SE (1)



### (103/108) Green Resilience of Cyber-Physical Systems (Diaeddin Rimawi, 2023)

{{<citation>}}

Diaeddin Rimawi. (2023)  
**Green Resilience of Cyber-Physical Systems**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05201v1)  

---


**ABSTRACT**  
Cyber-Physical System (CPS) represents systems that join both hardware and software components to perform real-time services. Maintaining the system's reliability is critical to the continuous delivery of these services. However, the CPS running environment is full of uncertainties and can easily lead to performance degradation. As a result, the need for a recovery technique is highly needed to achieve resilience in the system, with keeping in mind that this technique should be as green as possible. This early doctorate proposal, suggests a game theory solution to achieve resilience and green in CPS. Game theory has been known for its fast performance in decision-making, helping the system to choose what maximizes its payoffs. The proposed game model is described over a real-life collaborative artificial intelligence system (CAIS), that involves robots with humans to achieve a common goal. It shows how the expected results of the system will achieve the resilience of CAIS with minimized CO2 footprint.

{{</citation>}}


## eess.AS (1)



### (104/108) Improving Whispered Speech Recognition Performance using Pseudo-whispered based Data Augmentation (Zhaofeng Lin et al., 2023)

{{<citation>}}

Zhaofeng Lin, Tanvina Patel, Odette Scharenborg. (2023)  
**Improving Whispered Speech Recognition Performance using Pseudo-whispered based Data Augmentation**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Augmentation, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2311.05179v1)  

---


**ABSTRACT**  
Whispering is a distinct form of speech known for its soft, breathy, and hushed characteristics, often used for private communication. The acoustic characteristics of whispered speech differ substantially from normally phonated speech and the scarcity of adequate training data leads to low automatic speech recognition (ASR) performance. To address the data scarcity issue, we use a signal processing-based technique that transforms the spectral characteristics of normal speech to those of pseudo-whispered speech. We augment an End-to-End ASR with pseudo-whispered speech and achieve an 18.2% relative reduction in word error rate for whispered speech compared to the baseline. Results for the individual speaker groups in the wTIMIT database show the best results for US English. Further investigation showed that the lack of glottal information in whispered speech has the largest impact on whispered speech ASR performance.

{{</citation>}}


## cs.NE (2)



### (105/108) Rethinking Residual Connection in Training Large-Scale Spiking Neural Networks (Yudong Li et al., 2023)

{{<citation>}}

Yudong Li, Yunlin Lei, Xu Yang. (2023)  
**Rethinking Residual Connection in Training Large-Scale Spiking Neural Networks**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.05171v1)  

---


**ABSTRACT**  
Spiking Neural Network (SNN) is known as the most famous brain-inspired model, but the non-differentiable spiking mechanism makes it hard to train large-scale SNNs. To facilitate the training of large-scale SNNs, many training methods are borrowed from Artificial Neural Networks (ANNs), among which deep residual learning is the most commonly used. But the unique features of SNNs make prior intuition built upon ANNs not available for SNNs. Although there are a few studies that have made some pioneer attempts on the topology of Spiking ResNet, the advantages of different connections remain unclear. To tackle this issue, we analyze the merits and limitations of various residual connections and empirically demonstrate our ideas with extensive experiments. Then, based on our observations, we abstract the best-performing connections into densely additive (DA) connection, extend such a concept to other topologies, and propose four architectures for training large-scale SNNs, termed DANet, which brings up to 13.24% accuracy gain on ImageNet. Besides, in order to present a detailed methodology for designing the topology of large-scale SNNs, we further conduct in-depth discussions on their applicable scenarios in terms of their performance on various scales of datasets and demonstrate their advantages over prior architectures. At a low training expense, our best-performing ResNet-50/101/152 obtain 73.71%/76.13%/77.22% top-1 accuracy on ImageNet with 4 time steps. We believe that this work shall give more insights for future works to design the topology of their networks and promote the development of large-scale SNNs. The code will be publicly available.

{{</citation>}}


### (106/108) A differentiable brain simulator bridging brain simulation and brain-inspired computing (Chaoming Wang et al., 2023)

{{<citation>}}

Chaoming Wang, Tianqiu Zhang, Sichao He, Yifeng Gong, Hongyaoxing Gu, Shangyang Li, Si Wu. (2023)  
**A differentiable brain simulator bridging brain simulation and brain-inspired computing**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-NE, cs.NE, q-bio-NC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.05106v1)  

---


**ABSTRACT**  
Brain simulation builds dynamical models to mimic the structure and functions of the brain, while brain-inspired computing (BIC) develops intelligent systems by learning from the structure and functions of the brain. The two fields are intertwined and should share a common programming framework to facilitate each other's development. However, none of the existing software in the fields can achieve this goal, because traditional brain simulators lack differentiability for training, while existing deep learning (DL) frameworks fail to capture the biophysical realism and complexity of brain dynamics. In this paper, we introduce BrainPy, a differentiable brain simulator developed using JAX and XLA, with the aim of bridging the gap between brain simulation and BIC. BrainPy expands upon the functionalities of JAX, a powerful AI framework, by introducing complete capabilities for flexible, efficient, and scalable brain simulation. It offers a range of sparse and event-driven operators for efficient and scalable brain simulation, an abstraction for managing the intricacies of synaptic computations, a modular and flexible interface for constructing multi-scale brain models, and an object-oriented just-in-time compilation approach to handle the memory-intensive nature of brain dynamics. We showcase the efficiency and scalability of BrainPy on benchmark tasks, highlight its differentiable simulation for biologically plausible spiking models, and discuss its potential to support research at the intersection of brain simulation and BIC.

{{</citation>}}


## math.OC (1)



### (107/108) Improving Computational Efficiency for Powered Descent Guidance via Transformer-based Tight Constraint Prediction (Julia Briden et al., 2023)

{{<citation>}}

Julia Briden, Trey Gurga, Breanna Johnson, Abhishek Cauligi, Richard Linares. (2023)  
**Improving Computational Efficiency for Powered Descent Guidance via Transformer-based Tight Constraint Prediction**  

---
Primary Category: math.OC  
Categories: cs-LG, cs-RO, cs-SY, eess-SY, math-OC, math.OC  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.05135v1)  

---


**ABSTRACT**  
In this work, we present Transformer-based Powered Descent Guidance (T-PDG), a scalable algorithm for reducing the computational complexity of the direct optimization formulation of the spacecraft powered descent guidance problem. T-PDG uses data from prior runs of trajectory optimization algorithms to train a transformer neural network, which accurately predicts the relationship between problem parameters and the globally optimal solution for the powered descent guidance problem. The solution is encoded as the set of tight constraints corresponding to the constrained minimum-cost trajectory and the optimal final time of landing. By leveraging the attention mechanism of transformer neural networks, large sequences of time series data can be accurately predicted when given only the spacecraft state and landing site parameters. When applied to the real problem of Mars powered descent guidance, T-PDG reduces the time for computing the 3 degree of freedom fuel-optimal trajectory, when compared to lossless convexification, from an order of 1-8 seconds to less than 500 milliseconds. A safe and optimal solution is guaranteed by including a feasibility check in T-PDG before returning the final trajectory.

{{</citation>}}


## cs.IT (1)



### (108/108) Vector Approximate Survey Propagation for Model-Mismatched Estimation (Or: How to Achieve Kabashima's 1RSB Prediction) (Qun Chen et al., 2023)

{{<citation>}}

Qun Chen, Haochuan Zhang, Huimin Zhu. (2023)  
**Vector Approximate Survey Propagation for Model-Mismatched Estimation (Or: How to Achieve Kabashima's 1RSB Prediction)**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2311.05111v1)  

---


**ABSTRACT**  
For approximate inference in high-dimensional generalized linear models (GLMs), the performance of an estimator may significantly degrade when mismatch exists between the postulated model and the ground truth. In mismatched GLMs with rotation-invariant measurement matrices, Kabashima et al. proved vector approximate message passing (VAMP) computes exactly the optimal estimator if the replica symmetry (RS) ansatz is valid, but it becomes inappropriate if RS breaking (RSB) appears. Although the one-step RSB (1RSB) saddle point equations were given for the optimal estimator, the question remains: how to achieve the 1RSB prediction? This paper answers the question by proposing a new algorithm, vector approximate survey propagation (VASP). VASP derives from a reformulation of Kabashima's extremum conditions, which later links the theoretical equations to survey propagation in vector form and finally the algorithm. VASP has a complexity as low as VAMP, while embracing VAMP as a special case. The SE derived for VASP can capture precisely the per-iteration behavior of the simulated algorithm, and the SE's fixed point equations perfectly match Kabashima's 1RSB prediction, which indicates VASP can achieve the optimal performance even in a model-mismatched setting with 1RSB. Simulation results confirm VASP outperforms many state-of-the-art algorithms.

{{</citation>}}
