---
draft: false
title: "arXiv @ 2023.12.07"
date: 2023-12-07
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.07"
    identifier: arxiv_20231207
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.HC (3)](#cshc-3)
- [math.NA (1)](#mathna-1)
- [cs.LG (30)](#cslg-30)
- [cs.CY (1)](#cscy-1)
- [cs.IR (6)](#csir-6)
- [cs.AR (1)](#csar-1)
- [eess.IV (5)](#eessiv-5)
- [cs.CL (18)](#cscl-18)
- [cs.CV (37)](#cscv-37)
- [cond-mat.mes-hall (1)](#cond-matmes-hall-1)
- [stat.ML (3)](#statml-3)
- [cs.SI (2)](#cssi-2)
- [cond-mat.mtrl-sci (1)](#cond-matmtrl-sci-1)
- [cs.NI (2)](#csni-2)
- [cs.DC (1)](#csdc-1)
- [cs.AI (6)](#csai-6)
- [cs.CR (2)](#cscr-2)
- [cs.IT (1)](#csit-1)
- [cs.RO (2)](#csro-2)
- [quant-ph (1)](#quant-ph-1)
- [eess.SY (1)](#eesssy-1)
- [q-bio.QM (1)](#q-bioqm-1)

## cs.HC (3)



### (1/126) Conceptualizing the Relationship between AI Explanations and User Agency (Iyadunni Adenuga et al., 2023)

{{<citation>}}

Iyadunni Adenuga, Jonathan Dodge. (2023)  
**Conceptualizing the Relationship between AI Explanations and User Agency**  

---
Primary Category: cs.HC  
Categories: cs-CY, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03193v1)  

---


**ABSTRACT**  
We grapple with the question: How, for whom and why should explainable artificial intelligence (XAI) aim to support the user goal of agency? In particular, we analyze the relationship between agency and explanations through a user-centric lens through case studies and thought experiments. We find that explanation serves as one of several possible first steps for agency by allowing the user convert forethought to outcome in a more effective manner in future interactions. Also, we observe that XAI systems might better cater to laypersons, particularly "tinkerers", when combining explanations and user control, so they can make meaningful changes.

{{</citation>}}


### (2/126) RESIN-EDITOR: A Schema-guided Hierarchical Event Graph Visualizer and Editor (Khanh Duy Nguyen et al., 2023)

{{<citation>}}

Khanh Duy Nguyen, Zixuan Zhang, Reece Suchocki, Sha Li, Martha Palmer, Susan Brown, Jiawei Han, Heng Ji. (2023)  
**RESIN-EDITOR: A Schema-guided Hierarchical Event Graph Visualizer and Editor**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CL, cs-HC, cs.HC  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2312.03093v1)  

---


**ABSTRACT**  
In this paper, we present RESIN-EDITOR, an interactive event graph visualizer and editor designed for analyzing complex events. Our RESIN-EDITOR system allows users to render and freely edit hierarchical event graphs extracted from multimedia and multi-document news clusters with guidance from human-curated event schemas. RESIN-EDITOR's unique features include hierarchical graph visualization, comprehensive source tracing, and interactive user editing, which is more powerful and versatile than existing Information Extraction (IE) visualization tools. In our evaluation of RESIN-EDITOR, we demonstrate ways in which our tool is effective in understanding complex events and enhancing system performance. The source code, a video demonstration, and a live website for RESIN-EDITOR have been made publicly available.

{{</citation>}}


### (3/126) BOgen: Generating Part-Level 3D Designs Based on User Intention Inference through Bayesian Optimization and Variational Autoencoder (Seung Won Lee et al., 2023)

{{<citation>}}

Seung Won Lee, Jiin Choi, Kyung Hoon Hyun. (2023)  
**BOgen: Generating Part-Level 3D Designs Based on User Intention Inference through Bayesian Optimization and Variational Autoencoder**  

---
Primary Category: cs.HC  
Categories: H-5-2; I-2-1, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.02557v1)  

---


**ABSTRACT**  
Advancements in generative artificial intelligence (AI) have introduced various AI models capable of producing impressive visual design outputs. However, when it comes to AI models in the design process, prioritizing outputs that align with designers' needs over mere visual craftsmanship becomes even more crucial. Furthermore, designers often intricately combine parts of various designs to create novel designs. The ability to generate designs that align with the designers' intentions at the part level is pivotal for assisting designers. Hence, we introduced BOgen, which empowers designers to proactively generate and explore part-level designs through Bayesian optimization and variational autoencoders, thereby enhancing their overall user experience. We assessed BOgen's performance using a study involving 30 designers. The results revealed that, compared to the baseline, BOgen fulfilled the designer requirements for part recommendations and design exploration space guidance. BOgen assists designers in navigation and development, offering valuable design suggestions and fosters proactive design exploration and creation.

{{</citation>}}


## math.NA (1)



### (4/126) Image reconstructions using sparse dictionary representations and implicit, non-negative mappings (Elizabeth Newman et al., 2023)

{{<citation>}}

Elizabeth Newman, Jack Michael Solomon, Matthias Chung. (2023)  
**Image reconstructions using sparse dictionary representations and implicit, non-negative mappings**  

---
Primary Category: math.NA  
Categories: 65F10, 65F22, G-1-3, cs-NA, math-NA, math.NA  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.03180v1)  

---


**ABSTRACT**  
Many imaging science tasks can be modeled as a discrete linear inverse problem. Solving linear inverse problems is often challenging, with ill-conditioned operators and potentially non-unique solutions. Embedding prior knowledge, such as smoothness, into the solution can overcome these challenges. In this work, we encode prior knowledge using a non-negative patch dictionary, which effectively learns a basis from a training set of natural images. In this dictionary basis, we desire solutions that are non-negative and sparse (i.e., contain many zero entries). With these constraints, standard methods for solving discrete linear inverse problems are not directly applicable. One such approach is the modified residual norm steepest descent (MRNSD), which produces non-negative solutions but does not induce sparsity. In this paper, we provide two methods based on MRNSD that promote sparsity. In our first method, we add an $\ell_1$-regularization term with a new, optimal step size. In our second method, we propose a new non-negative, sparsity-promoting mapping of the solution. We compare the performance of our proposed methods on a number of numerical experiments, including deblurring, image completion, computer tomography, and superresolution. Our results show that these methods effectively solve discrete linear inverse problems with non-negativity and sparsity constraints.

{{</citation>}}


## cs.LG (30)



### (5/126) Using Curiosity for an Even Representation of Tasks in Continual Offline Reinforcement Learning (Pankayaraj Pathmanathan et al., 2023)

{{<citation>}}

Pankayaraj Pathmanathan, Natalia Díaz-Rodríguez, Javier Del Ser. (2023)  
**Using Curiosity for an Even Representation of Tasks in Continual Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.03177v1)  

---


**ABSTRACT**  
In this work, we investigate the means of using curiosity on replay buffers to improve offline multi-task continual reinforcement learning when tasks, which are defined by the non-stationarity in the environment, are non labeled and not evenly exposed to the learner in time. In particular, we investigate the use of curiosity both as a tool for task boundary detection and as a priority metric when it comes to retaining old transition tuples, which we respectively use to propose two different buffers. Firstly, we propose a Hybrid Reservoir Buffer with Task Separation (HRBTS), where curiosity is used to detect task boundaries that are not known due to the task agnostic nature of the problem. Secondly, by using curiosity as a priority metric when it comes to retaining old transition tuples, a Hybrid Curious Buffer (HCB) is proposed. We ultimately show that these buffers, in conjunction with regular reinforcement learning algorithms, can be used to alleviate the catastrophic forgetting issue suffered by the state of the art on replay buffers when the agent's exposure to tasks is not equal along time. We evaluate catastrophic forgetting and the efficiency of our proposed buffers against the latest works such as the Hybrid Reservoir Buffer (HRB) and the Multi-Time Scale Replay Buffer (MTR) in three different continual reinforcement learning settings. Experiments were done on classical control tasks and Metaworld environment. Experiments show that our proposed replay buffers display better immunity to catastrophic forgetting compared to existing works in most of the settings.

{{</citation>}}


### (6/126) Active Learning for Abrupt Shifts Change-point Detection via Derivative-Aware Gaussian Processes (Hao Zhao et al., 2023)

{{<citation>}}

Hao Zhao, Rong Pan. (2023)  
**Active Learning for Abrupt Shifts Change-point Detection via Derivative-Aware Gaussian Processes**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2312.03176v1)  

---


**ABSTRACT**  
Change-point detection (CPD) is crucial for identifying abrupt shifts in data, which influence decision-making and efficient resource allocation across various domains. To address the challenges posed by the costly and time-intensive data acquisition in CPD, we introduce the Derivative-Aware Change Detection (DACD) method. It leverages the derivative process of a Gaussian process (GP) for Active Learning (AL), aiming to pinpoint change-point locations effectively. DACD balances the exploitation and exploration of derivative processes through multiple data acquisition functions (AFs). By utilizing GP derivative mean and variance as criteria, DACD sequentially selects the next sampling data point, thus enhancing algorithmic efficiency and ensuring reliable and accurate results. We investigate the effectiveness of DACD method in diverse scenarios and show it outperforms other active learning change-point detection approaches.

{{</citation>}}


### (7/126) FlexModel: A Framework for Interpretability of Distributed Large Language Models (Matthew Choi et al., 2023)

{{<citation>}}

Matthew Choi, Muhammad Adil Asif, John Willes, David Emerson. (2023)  
**FlexModel: A Framework for Interpretability of Distributed Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-DC, cs-LG, cs.LG  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.03140v1)  

---


**ABSTRACT**  
With the growth of large language models, now incorporating billions of parameters, the hardware prerequisites for their training and deployment have seen a corresponding increase. Although existing tools facilitate model parallelization and distributed training, deeper model interactions, crucial for interpretability and responsible AI techniques, still demand thorough knowledge of distributed computing. This often hinders contributions from researchers with machine learning expertise but limited distributed computing background. Addressing this challenge, we present FlexModel, a software package providing a streamlined interface for engaging with models distributed across multi-GPU and multi-node configurations. The library is compatible with existing model distribution libraries and encapsulates PyTorch models. It exposes user-registerable HookFunctions to facilitate straightforward interaction with distributed model internals, bridging the gap between distributed and single-device model paradigms. Primarily, FlexModel enhances accessibility by democratizing model interactions and promotes more inclusive research in the domain of large-scale neural networks. The package is found at https://github.com/VectorInstitute/flex_model.

{{</citation>}}


### (8/126) Incidental Polysemanticity (Victor Lecomte et al., 2023)

{{<citation>}}

Victor Lecomte, Kushal Thaman, Trevor Chow, Rylan Schaeffer, Sanmi Koyejo. (2023)  
**Incidental Polysemanticity**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NE, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03096v1)  

---


**ABSTRACT**  
Polysemantic neurons (neurons that activate for a set of unrelated features) have been seen as a significant obstacle towards interpretability of task-optimized deep networks, with implications for AI safety. The classic origin story of polysemanticity is that the data contains more "features" than neurons, such that learning to perform a task forces the network to co-allocate multiple unrelated features to the same neuron, endangering our ability to understand the network's internal processing. In this work, we present a second and non-mutually exclusive origin story of polysemanticity. We show that polysemanticity can arise incidentally, even when there are ample neurons to represent all features in the data, using a combination of theory and experiments. This second type of polysemanticity occurs because random initialization can, by chance alone, initially assign multiple features to the same neuron, and the training dynamics then strengthen such overlap. Due to its origin, we term this \textit{incidental polysemanticity}.

{{</citation>}}


### (9/126) Toward autocorrection of chemical process flowsheets using large language models (Lukas Schulze Balhorn et al., 2023)

{{<citation>}}

Lukas Schulze Balhorn, Marc Caballero, Artur M. Schweidtmann. (2023)  
**Toward autocorrection of chemical process flowsheets using large language models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.02873v1)  

---


**ABSTRACT**  
The process engineering domain widely uses Process Flow Diagrams (PFDs) and Process and Instrumentation Diagrams (P&IDs) to represent process flows and equipment configurations. However, the P&IDs and PFDs, hereafter called flowsheets, can contain errors causing safety hazards, inefficient operation, and unnecessary expenses. Correcting and verifying flowsheets is a tedious, manual process. We propose a novel generative AI methodology for automatically identifying errors in flowsheets and suggesting corrections to the user, i.e., autocorrecting flowsheets. Inspired by the breakthrough of Large Language Models (LLMs) for grammatical autocorrection of human language, we investigate LLMs for the autocorrection of flowsheets. The input to the model is a potentially erroneous flowsheet and the output of the model are suggestions for a corrected flowsheet. We train our autocorrection model on a synthetic dataset in a supervised manner. The model achieves a top-1 accuracy of 80% and a top-5 accuracy of 84% on an independent test dataset of synthetically generated flowsheets. The results suggest that the model can learn to autocorrect the synthetic flowsheets. We envision that flowsheet autocorrection will become a useful tool for chemical engineers.

{{</citation>}}


### (10/126) Attention-enhanced neural differential equations for physics-informed deep learning of ion transport (Danyal Rehman et al., 2023)

{{<citation>}}

Danyal Rehman, John H. Lienhard. (2023)  
**Attention-enhanced neural differential equations for physics-informed deep learning of ion transport**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-MP, math-ph, physics-comp-ph  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.02871v1)  

---


**ABSTRACT**  
Species transport models typically combine partial differential equations (PDEs) with relations from hindered transport theory to quantify electromigrative, convective, and diffusive transport through complex nanoporous systems; however, these formulations are frequently substantial simplifications of the governing dynamics, leading to the poor generalization performance of PDE-based models. Given the growing interest in deep learning methods for the physical sciences, we develop a machine learning-based approach to characterize ion transport across nanoporous membranes. Our proposed framework centers around attention-enhanced neural differential equations that incorporate electroneutrality-based inductive biases to improve generalization performance relative to conventional PDE-based methods. In addition, we study the role of the attention mechanism in illuminating physically-meaningful ion-pairing relationships across diverse mixture compositions. Further, we investigate the importance of pre-training on simulated data from PDE-based models, as well as the performance benefits from hard vs. soft inductive biases. Our results indicate that physics-informed deep learning solutions can outperform their classical PDE-based counterparts and provide promising avenues for modelling complex transport phenomena across diverse applications.

{{</citation>}}


### (11/126) Semi-Supervised Health Index Monitoring with Feature Generation and Fusion (Gaëtan Frusque et al., 2023)

{{<citation>}}

Gaëtan Frusque, Ismail Nejjar, Majid Nabavi, Olga Fink. (2023)  
**Semi-Supervised Health Index Monitoring with Feature Generation and Fusion**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ME  
Keywords: Anomaly Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.02867v1)  

---


**ABSTRACT**  
The Health Index (HI) is crucial for evaluating system health, aiding tasks like anomaly detection and predicting remaining useful life for systems demanding high safety and reliability. Tight monitoring is crucial for achieving high precision at a lower cost, with applications such as spray coating. Obtaining HI labels in real-world applications is often cost-prohibitive, requiring continuous, precise health measurements. Therefore, it is more convenient to leverage run-to failure datasets that may provide potential indications of machine wear condition, making it necessary to apply semi-supervised tools for HI construction. In this study, we adapt the Deep Semi-supervised Anomaly Detection (DeepSAD) method for HI construction. We use the DeepSAD embedding as a condition indicators to address interpretability challenges and sensitivity to system-specific factors. Then, we introduce a diversity loss to enrich condition indicators. We employ an alternating projection algorithm with isotonic constraints to transform the DeepSAD embedding into a normalized HI with an increasing trend. Validation on the PHME 2010 milling dataset, a recognized benchmark with ground truth HIs demonstrates meaningful HIs estimations. Our methodology is then applied to monitor wear states of thermal spray coatings using high-frequency voltage. Our contributions create opportunities for more accessible and reliable HI estimation, particularly in cases where obtaining ground truth HI labels is unfeasible.

{{</citation>}}


### (12/126) Transformer-Based Deep Learning Model for Bored Pile Load-Deformation Prediction in Bangkok Subsoil (Sompote Youwai et al., 2023)

{{<citation>}}

Sompote Youwai, Chissanupong Thongnoo. (2023)  
**Transformer-Based Deep Learning Model for Bored Pile Load-Deformation Prediction in Bangkok Subsoil**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.03041v1)  

---


**ABSTRACT**  
This paper presents a novel deep learning model based on the transformer architecture to predict the load-deformation behavior of large bored piles in Bangkok subsoil. The model encodes the soil profile and pile features as tokenization input, and generates the load-deformation curve as output. The model also incorporates the previous sequential data of load-deformation curve into the decoder to improve the prediction accuracy. The model also incorporates the previous sequential data of load-deformation curve into the decoder. The model shows a satisfactory accuracy and generalization ability for the load-deformation curve prediction, with a mean absolute error of 5.72% for the test data. The model could also be used for parametric analysis and design optimization of piles under different soil and pile conditions, pile cross section, pile length and type of pile.

{{</citation>}}


### (13/126) MIMONets: Multiple-Input-Multiple-Output Neural Networks Exploiting Computation in Superposition (Nicolas Menet et al., 2023)

{{<citation>}}

Nicolas Menet, Michael Hersche, Geethan Karunaratne, Luca Benini, Abu Sebastian, Abbas Rahimi. (2023)  
**MIMONets: Multiple-Input-Multiple-Output Neural Networks Exploiting Computation in Superposition**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.02829v1)  

---


**ABSTRACT**  
With the advent of deep learning, progressively larger neural networks have been designed to solve complex tasks. We take advantage of these capacity-rich models to lower the cost of inference by exploiting computation in superposition. To reduce the computational burden per input, we propose Multiple-Input-Multiple-Output Neural Networks (MIMONets) capable of handling many inputs at once. MIMONets augment various deep neural network architectures with variable binding mechanisms to represent an arbitrary number of inputs in a compositional data structure via fixed-width distributed representations. Accordingly, MIMONets adapt nonlinear neural transformations to process the data structure holistically, leading to a speedup nearly proportional to the number of superposed input items in the data structure. After processing in superposition, an unbinding mechanism recovers each transformed input of interest. MIMONets also provide a dynamic trade-off between accuracy and throughput by an instantaneous on-demand switching between a set of accuracy-throughput operating points, yet within a single set of fixed parameters. We apply the concept of MIMONets to both CNN and Transformer architectures resulting in MIMOConv and MIMOFormer, respectively. Empirical evaluations show that MIMOConv achieves about 2-4 x speedup at an accuracy delta within [+0.68, -3.18]% compared to WideResNet CNNs on CIFAR10 and CIFAR100. Similarly, MIMOFormer can handle 2-4 inputs at once while maintaining a high average accuracy within a [-1.07, -3.43]% delta on the long range arena benchmark. Finally, we provide mathematical bounds on the interference between superposition channels in MIMOFormer. Our code is available at https://github.com/IBM/multiple-input-multiple-output-nets.

{{</citation>}}


### (14/126) Sample-based Dynamic Hierarchical Transformer with Layer and Head Flexibility via Contextual Bandit (Fanfei Meng et al., 2023)

{{<citation>}}

Fanfei Meng, Lele Zhang, Yu Chen, Yuxin Wang. (2023)  
**Sample-based Dynamic Hierarchical Transformer with Layer and Head Flexibility via Contextual Bandit**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NE, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.03038v1)  

---


**ABSTRACT**  
Transformer requires a fixed number of layers and heads which makes them inflexible to the complexity of individual samples and expensive in training and inference. To address this, we propose a sample-based Dynamic Hierarchical Transformer (DHT) model whose layers and heads can be dynamically configured with single data samples via solving contextual bandit problems. To determine the number of layers and heads, we use the Uniform Confidence Bound while we deploy combinatorial Thompson Sampling in order to select specific head combinations given their number. Different from previous work that focuses on compressing trained networks for inference only, DHT is not only advantageous for adaptively optimizing the underlying network architecture during training but also has a flexible network for efficient inference. To the best of our knowledge, this is the first comprehensive data-driven dynamic transformer without any additional auxiliary neural networks that implement the dynamic system. According to the experiment results, we achieve up to 74% computational savings for both training and inference with a minimal loss of accuracy.

{{</citation>}}


### (15/126) Weakly Supervised Detection of Hallucinations in LLM Activations (Miriam Rateike et al., 2023)

{{<citation>}}

Miriam Rateike, Celia Cintas, John Wamburu, Tanya Akumu, Skyler Speakman. (2023)  
**Weakly Supervised Detection of Hallucinations in LLM Activations**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.02798v1)  

---


**ABSTRACT**  
We propose an auditing method to identify whether a large language model (LLM) encodes patterns such as hallucinations in its internal states, which may propagate to downstream tasks. We introduce a weakly supervised auditing technique using a subset scanning approach to detect anomalous patterns in LLM activations from pre-trained models. Importantly, our method does not need knowledge of the type of patterns a-priori. Instead, it relies on a reference dataset devoid of anomalies during testing. Further, our approach enables the identification of pivotal nodes responsible for encoding these patterns, which may offer crucial insights for fine-tuning specific sub-networks for bias mitigation. We introduce two new scanning methods to handle LLM activations for anomalous sentences that may deviate from the expected distribution in either direction. Our results confirm prior findings of BERT's limited internal capacity for encoding hallucinations, while OPT appears capable of encoding hallucination information internally. Importantly, our scanning approach, without prior exposure to false statements, performs comparably to a fully supervised out-of-distribution classifier.

{{</citation>}}


### (16/126) Scaling Laws for Adversarial Attacks on Language Model Activations (Stanislav Fort, 2023)

{{<citation>}}

Stanislav Fort. (2023)  
**Scaling Laws for Adversarial Attacks on Language Model Activations**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: Adversarial Attack, Language Model  
[Paper Link](http://arxiv.org/abs/2312.02780v1)  

---


**ABSTRACT**  
We explore a class of adversarial attacks targeting the activations of language models. By manipulating a relatively small subset of model activations, $a$, we demonstrate the ability to control the exact prediction of a significant number (in some cases up to 1000) of subsequent tokens $t$. We empirically verify a scaling law where the maximum number of target tokens $t_\mathrm{max}$ predicted depends linearly on the number of tokens $a$ whose activations the attacker controls as $t_\mathrm{max} = \kappa a$. We find that the number of bits of control in the input space needed to control a single bit in the output space (what we call attack resistance $\chi$) is remarkably constant between $\approx 16$ and $\approx 25$ over 2 orders of magnitude of model sizes for different language models. Compared to attacks on tokens, attacks on activations are predictably much stronger, however, we identify a surprising regularity where one bit of input steered either via activations or via tokens is able to exert control over a similar amount of output bits. This gives support for the hypothesis that adversarial attacks are a consequence of dimensionality mismatch between the input and output spaces. A practical implication of the ease of attacking language model activations instead of tokens is for multi-modal and selected retrieval models, where additional data sources are added as activations directly, sidestepping the tokenized input. This opens up a new, broad attack surface. By using language models as a controllable test-bed to study adversarial attacks, we were able to experiment with input-output dimensions that are inaccessible in computer vision, especially where the output dimension dominates.

{{</citation>}}


### (17/126) LExCI: A Framework for Reinforcement Learning with Embedded Systems (Kevin Badalian et al., 2023)

{{<citation>}}

Kevin Badalian, Lucas Koch, Tobias Brinkmann, Mario Picerno, Marius Wegener, Sung-Yong Lee, Jakob Andert. (2023)  
**LExCI: A Framework for Reinforcement Learning with Embedded Systems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.02739v1)  

---


**ABSTRACT**  
Advances in artificial intelligence (AI) have led to its application in many areas of everyday life. In the context of control engineering, reinforcement learning (RL) represents a particularly promising approach as it is centred around the idea of allowing an agent to freely interact with its environment to find an optimal strategy. One of the challenges professionals face when training and deploying RL agents is that the latter often have to run on dedicated embedded devices. This could be to integrate them into an existing toolchain or to satisfy certain performance criteria like real-time constraints. Conventional RL libraries, however, cannot be easily utilised in conjunction with that kind of hardware. In this paper, we present a framework named LExCI, the Learning and Experiencing Cycle Interface, which bridges this gap and provides end-users with a free and open-source tool for training agents on embedded systems using the open-source library RLlib. Its operability is demonstrated with two state-of-the-art RL-algorithms and a rapid control prototyping system.

{{</citation>}}


### (18/126) Towards Measuring Representational Similarity of Large Language Models (Max Klabunde et al., 2023)

{{<citation>}}

Max Klabunde, Mehdi Ben Amor, Michael Granitzer, Florian Lemmerich. (2023)  
**Towards Measuring Representational Similarity of Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.02730v1)  

---


**ABSTRACT**  
Understanding the similarity of the numerous released large language models (LLMs) has many uses, e.g., simplifying model selection, detecting illegal model reuse, and advancing our understanding of what makes LLMs perform well. In this work, we measure the similarity of representations of a set of LLMs with 7B parameters. Our results suggest that some LLMs are substantially different from others. We identify challenges of using representational similarity measures that suggest the need of careful study of similarity scores to avoid false conclusions.

{{</citation>}}


### (19/126) A Self-Commissioning Edge Computing Method for Data-Driven Anomaly Detection in Power Electronic Systems (Pere Izquierdo Gomez et al., 2023)

{{<citation>}}

Pere Izquierdo Gomez, Miguel E. Lopez Gajardo, Nenad Mijatovic, Tomislav Dragicevic. (2023)  
**A Self-Commissioning Edge Computing Method for Data-Driven Anomaly Detection in Power Electronic Systems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.02661v1)  

---


**ABSTRACT**  
Ensuring the reliability of power electronic converters is a matter of great importance, and data-driven condition monitoring techniques are cementing themselves as an important tool for this purpose. However, translating methods that work well in controlled lab environments to field applications presents significant challenges, notably because of the limited diversity and accuracy of the lab training data. By enabling the use of field data, online machine learning can be a powerful tool to overcome this problem, but it introduces additional challenges in ensuring the stability and predictability of the training processes. This work presents an edge computing method that mitigates these shortcomings with minimal additional memory usage, by employing an autonomous algorithm that prioritizes the storage of training samples with larger prediction errors. The method is demonstrated on the use case of a self-commissioning condition monitoring system, in the form of a thermal anomaly detection scheme for a variable frequency motor drive, where the algorithm self-learned to distinguish normal and anomalous operation with minimal prior knowledge. The obtained results, based on experimental data, show a significant improvement in prediction accuracy and training speed, when compared to equivalent models trained online without the proposed data selection process.

{{</citation>}}


### (20/126) Do AI models produce better weather forecasts than physics-based models? A quantitative evaluation case study of Storm Ciarán (Andrew J. Charlton-Perez et al., 2023)

{{<citation>}}

Andrew J. Charlton-Perez, Helen F. Dacre, Simon Driscoll, Suzanne L. Gray, Ben Harvey, Natalie J. Harvey, Kieran M. R. Hunt, Robert W. Lee, Ranjini Swaminathan, Remy Vandaele, Ambrogio Volonté. (2023)  
**Do AI models produce better weather forecasts than physics-based models? A quantitative evaluation case study of Storm Ciarán**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-ao-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.02658v1)  

---


**ABSTRACT**  
There has been huge recent interest in the potential of making operational weather forecasts using machine learning techniques. As they become a part of the weather forecasting toolbox, there is a pressing need to understand how well current machine learning models can simulate high-impactweather events. We compare forecasts of Storm Ciar\'an, a European windstorm that caused sixteen deaths and extensive damage in Northern Europe, made by machine learning and numericalweather prediction models. The four machine learning models considered (FourCastNet, Pangu-Weather, GraphCast and FourCastNet-v2) produce forecasts that accurately capture the synoptic-scale structure of the cyclone including the position of the cloud head, shape of the warm sector and location of warm conveyor belt jet, and the large-scale dynamical drivers important for the rapid storm development such as the position of the storm relative to the upper-level jet exit. However, their ability to resolve the more detailed structures important for issuing weather warnings is more mixed. All of the machine learning models underestimate the peak amplitude of winds associated with the storm, only some machine learning models resolve the warm core seclusion and none of the machine learning models capture the sharp bent-back warm frontal gradient. Our study shows there is a great deal about the performance and properties of machine learning weather forecasts that can be derived from case studies of high-impact weather events such as Storm Ciar\'an.

{{</citation>}}


### (21/126) On the Initialization of Graph Neural Networks (Jiahang Li et al., 2023)

{{<citation>}}

Jiahang Li, Yakun Song, Xiang Song, David Paul Wipf. (2023)  
**On the Initialization of Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.02622v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have displayed considerable promise in graph representation learning across various applications. The core learning process requires the initialization of model weight matrices within each GNN layer, which is typically accomplished via classic initialization methods such as Xavier initialization. However, these methods were originally motivated to stabilize the variance of hidden embeddings and gradients across layers of Feedforward Neural Networks (FNNs) and Convolutional Neural Networks (CNNs) to avoid vanishing gradients and maintain steady information flow. In contrast, within the GNN context classical initializations disregard the impact of the input graph structure and message passing on variance. In this paper, we analyze the variance of forward and backward propagation across GNN layers and show that the variance instability of GNN initializations comes from the combined effect of the activation function, hidden dimension, graph structure and message passing. To better account for these influence factors, we propose a new initialization method for Variance Instability Reduction within GNN Optimization (Virgo), which naturally tends to equate forward and backward variances across successive layers. We conduct comprehensive experiments on 15 datasets to show that Virgo can lead to superior model performance and more stable variance at initialization on node classification, link prediction and graph classification tasks. Codes are in https://github.com/LspongebobJH/virgo_icml2023.

{{</citation>}}


### (22/126) Projection Regret: Reducing Background Bias for Novelty Detection via Diffusion Models (Sungik Choi et al., 2023)

{{<citation>}}

Sungik Choi, Hankook Lee, Honglak Lee, Moontae Lee. (2023)  
**Projection Regret: Reducing Background Bias for Novelty Detection via Diffusion Models**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.02615v1)  

---


**ABSTRACT**  
Novelty detection is a fundamental task of machine learning which aims to detect abnormal ($\textit{i.e.}$ out-of-distribution (OOD)) samples. Since diffusion models have recently emerged as the de facto standard generative framework with surprising generation results, novelty detection via diffusion models has also gained much attention. Recent methods have mainly utilized the reconstruction property of in-distribution samples. However, they often suffer from detecting OOD samples that share similar background information to the in-distribution data. Based on our observation that diffusion models can \emph{project} any sample to an in-distribution sample with similar background information, we propose \emph{Projection Regret (PR)}, an efficient novelty detection method that mitigates the bias of non-semantic information. To be specific, PR computes the perceptual distance between the test image and its diffusion-based projection to detect abnormality. Since the perceptual distance often fails to capture semantic changes when the background information is dominant, we cancel out the background bias by comparing it against recursive projections. Extensive experiments demonstrate that PR outperforms the prior art of generative-model-based novelty detection methods by a significant margin.

{{</citation>}}


### (23/126) Structured World Representations in Maze-Solving Transformers (Michael Igorevich Ivanitskiy et al., 2023)

{{<citation>}}

Michael Igorevich Ivanitskiy, Alex F. Spies, Tilman Räuker, Guillaume Corlouer, Chris Mathwin, Lucia Quirke, Can Rager, Rusheb Shah, Dan Valentine, Cecilia Diniz Behn, Katsumi Inoue, Samy Wu Fung. (2023)  
**Structured World Representations in Maze-Solving Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.02566v1)  

---


**ABSTRACT**  
Transformer models underpin many recent advances in practical machine learning applications, yet understanding their internal behavior continues to elude researchers. Given the size and complexity of these models, forming a comprehensive picture of their inner workings remains a significant challenge. To this end, we set out to understand small transformer models in a more tractable setting: that of solving mazes. In this work, we focus on the abstractions formed by these models and find evidence for the consistent emergence of structured internal representations of maze topology and valid paths. We demonstrate this by showing that the residual stream of only a single token can be linearly decoded to faithfully reconstruct the entire maze. We also find that the learned embeddings of individual tokens have spatial structure. Furthermore, we take steps towards deciphering the circuity of path-following by identifying attention heads (dubbed $\textit{adjacency heads}$), which are implicated in finding valid subsequent tokens.

{{</citation>}}


### (24/126) ULMA: Unified Language Model Alignment with Demonstration and Point-wise Human Preference (Tianchi Cai et al., 2023)

{{<citation>}}

Tianchi Cai, Xierui Song, Jiyan Jiang, Fei Teng, Jinjie Gu, Guannan Zhang. (2023)  
**ULMA: Unified Language Model Alignment with Demonstration and Point-wise Human Preference**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.02554v1)  

---


**ABSTRACT**  
Language model alignment is a cutting-edge technique in large language model training to align the model output to user's intent, e.g., being helpful and harmless. Recent alignment framework consists of two steps: supervised fine-tuning with demonstration data and preference learning with human preference data. Previous preference learning methods, such as RLHF and DPO, mainly focus on pair-wise preference data. However, in many real-world scenarios where human feedbacks are intrinsically point-wise, these methods will suffer from information loss or even fail. To fill this gap, in this paper, we first develop a preference learning method called point-wise DPO to tackle point-wise preference data. Further revelation on the connection between supervised fine-tuning and point-wise preference learning enables us to develop a unified framework for both human demonstration and point-wise preference data, which sheds new light on the construction of preference dataset. Extensive experiments on point-wise datasets with binary or continuous labels demonstrate the superior performance and efficiency of our proposed methods. A new dataset with high-quality demonstration samples on harmlessness is constructed and made publicly available.

{{</citation>}}


### (25/126) MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection (Junho Song et al., 2023)

{{<citation>}}

Junho Song, Keonwoo Kim, Jeonglyul Oh, Sungzoon Cho. (2023)  
**MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2312.02530v1)  

---


**ABSTRACT**  
Detecting anomalies in real-world multivariate time series data is challenging due to complex temporal dependencies and inter-variable correlations. Recently, reconstruction-based deep models have been widely used to solve the problem. However, these methods still suffer from an over-generalization issue and fail to deliver consistently high performance. To address this issue, we propose the MEMTO, a memory-guided Transformer using a reconstruction-based approach. It is designed to incorporate a novel memory module that can learn the degree to which each memory item should be updated in response to the input data. To stabilize the training procedure, we use a two-phase training paradigm which involves using K-means clustering for initializing memory items. Additionally, we introduce a bi-dimensional deviation-based detection criterion that calculates anomaly scores considering both input space and latent space. We evaluate our proposed method on five real-world datasets from diverse domains, and it achieves an average anomaly detection F1-score of 95.74%, significantly outperforming the previous state-of-the-art methods. We also conduct extensive experiments to empirically validate the effectiveness of our proposed model's key components.

{{</citation>}}


### (26/126) MASP: Scalable GNN-based Planning for Multi-Agent Navigation (Xinyi Yang et al., 2023)

{{<citation>}}

Xinyi Yang, Xinting Yang, Chao Yu, Jiayu Chen, Huazhong Yang, Yu Wang. (2023)  
**MASP: Scalable GNN-based Planning for Multi-Agent Navigation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Drone, GNN  
[Paper Link](http://arxiv.org/abs/2312.02522v1)  

---


**ABSTRACT**  
We investigate the problem of decentralized multi-agent navigation tasks, where multiple agents need to reach initially unassigned targets in a limited time. Classical planning-based methods suffer from expensive computation overhead at each step and offer limited expressiveness for complex cooperation strategies. In contrast, reinforcement learning (RL) has recently become a popular paradigm for addressing this issue. However, RL struggles with low data efficiency and cooperation when directly exploring (nearly) optimal policies in the large search space, especially with an increased agent number (e.g., 10+ agents) or in complex environments (e.g., 3D simulators). In this paper, we propose Multi-Agent Scalable GNN-based P lanner (MASP), a goal-conditioned hierarchical planner for navigation tasks with a substantial number of agents. MASP adopts a hierarchical framework to divide a large search space into multiple smaller spaces, thereby reducing the space complexity and accelerating training convergence. We also leverage graph neural networks (GNN) to model the interaction between agents and goals, improving goal achievement. Besides, to enhance generalization capabilities in scenarios with unseen team sizes, we divide agents into multiple groups, each with a previously trained number of agents. The results demonstrate that MASP outperforms classical planning-based competitors and RL baselines, achieving a nearly 100% success rate with minimal training data in both multi-agent particle environments (MPE) with 50 agents and a quadrotor 3-dimensional environment (OmniDrones) with 20 agents. Furthermore, the learned policy showcases zero-shot generalization across unseen team sizes.

{{</citation>}}


### (27/126) ASPEN: High-Throughput LoRA Fine-Tuning of Large Language Models with a Single GPU (Zhengmao Ye et al., 2023)

{{<citation>}}

Zhengmao Ye, Dengchun Li, Jingqi Tian, Tingfeng Lan, Jie Zuo, Lei Duan, Hui Lu, Yexi Jiang, Jian Sha, Ke Zhang, Mingjie Tang. (2023)  
**ASPEN: High-Throughput LoRA Fine-Tuning of Large Language Models with a Single GPU**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GLM, LLaMA, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2312.02515v1)  

---


**ABSTRACT**  
Transformer-based large language models (LLMs) have demonstrated outstanding performance across diverse domains, particularly when fine-turned for specific domains. Recent studies suggest that the resources required for fine-tuning LLMs can be economized through parameter-efficient methods such as Low-Rank Adaptation (LoRA). While LoRA effectively reduces computational burdens and resource demands, it currently supports only a single-job fine-tuning setup.   In this paper, we present ASPEN, a high-throughput framework for fine-tuning LLMs. ASPEN efficiently trains multiple jobs on a single GPU using the LoRA method, leveraging shared pre-trained model and adaptive scheduling. ASPEN is compatible with transformer-based language models like LLaMA and ChatGLM, etc. Experiments show that ASPEN saves 53% of GPU memory when training multiple LLaMA-7B models on NVIDIA A100 80GB GPU and boosts training throughput by about 17% compared to existing methods when training with various pre-trained models on different GPUs. The adaptive scheduling algorithm reduces turnaround time by 24%, end-to-end training latency by 12%, prioritizing jobs and preventing out-of-memory issues.

{{</citation>}}


### (28/126) Pseudo Replay-based Class Continual Learning for Online New Category Anomaly Detection in Additive Manufacturing (Zhangyue Shi et al., 2023)

{{<citation>}}

Zhangyue Shi, Tianxin Xie, Chenang Liu, Yuxuan Li. (2023)  
**Pseudo Replay-based Class Continual Learning for Online New Category Anomaly Detection in Additive Manufacturing**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.02491v1)  

---


**ABSTRACT**  
The incorporation of advanced sensors and machine learning techniques has enabled modern manufacturing enterprises to perform data-driven in-situ quality monitoring based on the sensor data collected in manufacturing processes. However, one critical challenge is that newly presented defect category may manifest as the manufacturing process continues, resulting in monitoring performance deterioration of previously trained machine learning models. Hence, there is an increasing need for empowering machine learning model to learn continually. Among all continual learning methods, memory-based continual learning has the best performance but faces the constraints of data storage capacity. To address this issue, this paper develops a novel pseudo replay-based continual learning by integrating class incremental learning and oversampling-based data generation. Without storing all the data, the developed framework could generate high-quality data representing previous classes to train machine learning model incrementally when new category anomaly occurs. In addition, it could even enhance the monitoring performance since it also effectively improves the data quality. The effectiveness of the proposed framework is validated in an additive manufacturing process, which leverages supervised classification problem for anomaly detection. The experimental results show that the developed method is very promising in detecting novel anomaly while maintaining a good performance on the previous task and brings up more flexibility in model architecture.

{{</citation>}}


### (29/126) Constrained Twin Variational Auto-Encoder for Intrusion Detection in IoT Systems (Phai Vu Dinh et al., 2023)

{{<citation>}}

Phai Vu Dinh, Quang Uy Nguyen, Dinh Thai Hoang, Diep N. Nguyen, Son Pham Bao, Eryk Dutkiewicz. (2023)  
**Constrained Twin Variational Auto-Encoder for Intrusion Detection in IoT Systems**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2312.02490v1)  

---


**ABSTRACT**  
Intrusion detection systems (IDSs) play a critical role in protecting billions of IoT devices from malicious attacks. However, the IDSs for IoT devices face inherent challenges of IoT systems, including the heterogeneity of IoT data/devices, the high dimensionality of training data, and the imbalanced data. Moreover, the deployment of IDSs on IoT systems is challenging, and sometimes impossible, due to the limited resources such as memory/storage and computing capability of typical IoT devices. To tackle these challenges, this article proposes a novel deep neural network/architecture called Constrained Twin Variational Auto-Encoder (CTVAE) that can feed classifiers of IDSs with more separable/distinguishable and lower-dimensional representation data. Additionally, in comparison to the state-of-the-art neural networks used in IDSs, CTVAE requires less memory/storage and computing power, hence making it more suitable for IoT IDS systems. Extensive experiments with the 11 most popular IoT botnet datasets show that CTVAE can boost around 1% in terms of accuracy and Fscore in detection attack compared to the state-of-the-art machine learning and representation learning methods, whilst the running time for attack detection is lower than 2E-6 seconds and the model size is lower than 1 MB. We also further investigate various characteristics of CTVAE in the latent space and in the reconstruction representation to demonstrate its efficacy compared with current well-known methods.

{{</citation>}}


### (30/126) NeutronStream: A Dynamic GNN Training Framework with Sliding Window for Graph Streams (Chaoyi Chen et al., 2023)

{{<citation>}}

Chaoyi Chen, Dechao Gao, Yanfeng Zhang, Qiange Wang, Zhenbo Fu, Xuecang Zhang, Junhua Zhu, Yu Gu, Ge Yu. (2023)  
**NeutronStream: A Dynamic GNN Training Framework with Sliding Window for Graph Streams**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2312.02473v1)  

---


**ABSTRACT**  
Existing Graph Neural Network (GNN) training frameworks have been designed to help developers easily create performant GNN implementations. However, most existing GNN frameworks assume that the input graphs are static, but ignore that most real-world graphs are constantly evolving. Though many dynamic GNN models have emerged to learn from evolving graphs, the training process of these dynamic GNNs is dramatically different from traditional GNNs in that it captures both the spatial and temporal dependencies of graph updates. This poses new challenges for designing dynamic GNN training frameworks. First, the traditional batched training method fails to capture real-time structural evolution information. Second, the time-dependent nature makes parallel training hard to design. Third, it lacks system supports for users to efficiently implement dynamic GNNs. In this paper, we present NeutronStream, a framework for training dynamic GNN models. NeutronStream abstracts the input dynamic graph into a chronologically updated stream of events and processes the stream with an optimized sliding window to incrementally capture the spatial-temporal dependencies of events. Furthermore, NeutronStream provides a parallel execution engine to tackle the sequential event processing challenge to achieve high performance. NeutronStream also integrates a built-in graph storage structure that supports dynamic updates and provides a set of easy-to-use APIs that allow users to express their dynamic GNNs. Our experimental results demonstrate that, compared to state-of-the-art dynamic GNN implementations, NeutronStream achieves speedups ranging from 1.48X to 5.87X and an average accuracy improvement of 3.97%.

{{</citation>}}


### (31/126) Generator Born from Classifier (Runpeng Yu et al., 2023)

{{<citation>}}

Runpeng Yu, Xinchao Wang. (2023)  
**Generator Born from Classifier**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.02470v1)  

---


**ABSTRACT**  
In this paper, we make a bold attempt toward an ambitious task: given a pre-trained classifier, we aim to reconstruct an image generator, without relying on any data samples. From a black-box perspective, this challenge seems intractable, since it inevitably involves identifying the inverse function for a classifier, which is, by nature, an information extraction process. As such, we resort to leveraging the knowledge encapsulated within the parameters of the neural network. Grounded on the theory of Maximum-Margin Bias of gradient descent, we propose a novel learning paradigm, in which the generator is trained to ensure that the convergence conditions of the network parameters are satisfied over the generated distribution of the samples. Empirical validation from various image generation tasks substantiates the efficacy of our strategy.

{{</citation>}}


### (32/126) Dimensionality Reduction and Dynamical Mode Recognition of Circular Arrays of Flame Oscillators Using Deep Neural Network (Weiming Xu et al., 2023)

{{<citation>}}

Weiming Xu, Tao Yang, Peng Zhang. (2023)  
**Dimensionality Reduction and Dynamical Mode Recognition of Circular Arrays of Flame Oscillators Using Deep Neural Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-flu-dyn  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.02462v1)  

---


**ABSTRACT**  
Oscillatory combustion in aero engines and modern gas turbines often has significant adverse effects on their operation, and accurately recognizing various oscillation modes is the prerequisite for understanding and controlling combustion instability. However, the high-dimensional spatial-temporal data of a complex combustion system typically poses considerable challenges to the dynamical mode recognition. Based on a two-layer bidirectional long short-term memory variational autoencoder (Bi-LSTM-VAE) dimensionality reduction model and a two-dimensional Wasserstein distance-based classifier (WDC), this study proposes a promising method (Bi-LSTM-VAE-WDC) for recognizing dynamical modes in oscillatory combustion systems. Specifically, the Bi-LSTM-VAE dimension reduction model was introduced to reduce the high-dimensional spatial-temporal data of the combustion system to a low-dimensional phase space; Gaussian kernel density estimates (GKDE) were computed based on the distribution of phase points in a grid; two-dimensional WD values were calculated from the GKDE maps to recognize the oscillation modes. The time-series data used in this study were obtained from numerical simulations of circular arrays of laminar flame oscillators. The results show that the novel Bi-LSTM-VAE method can produce a non-overlapping distribution of phase points, indicating an effective unsupervised mode recognition and classification. Furthermore, the present method exhibits a more prominent performance than VAE and PCA (principal component analysis) for distinguishing dynamical modes in complex flame systems, implying its potential in studying turbulent combustion.

{{</citation>}}


### (33/126) AI-driven emergence of frequency information non-uniform distribution via THz metasurface spectrum prediction (Xiaohua Xing et al., 2023)

{{<citation>}}

Xiaohua Xing, Yuqi Ren, Die Zou, Qiankun Zhang, Bingxuan Mao, Jianquan Yao, Deyi Xiong, Shuang Zhang, Liang Wu. (2023)  
**AI-driven emergence of frequency information non-uniform distribution via THz metasurface spectrum prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-optics  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03017v1)  

---


**ABSTRACT**  
Recently, artificial intelligence has been extensively deployed across various scientific disciplines, optimizing and guiding the progression of experiments through the integration of abundant datasets, whilst continuously probing the vast theoretical space encapsulated within the data. Particularly, deep learning models, due to their end-to-end adaptive learning capabilities, are capable of autonomously learning intrinsic data features, thereby transcending the limitations of traditional experience to a certain extent. Here, we unveil previously unreported information characteristics pertaining to different frequencies emerged during our work on predicting the terahertz spectral modulation effects of metasurfaces based on AI-prediction. Moreover, we have substantiated that our proposed methodology of simply adding supplementary multi-frequency inputs to the existing dataset during the target spectral prediction process can significantly enhance the predictive accuracy of the network. This approach effectively optimizes the utilization of existing datasets and paves the way for interdisciplinary research and applications in artificial intelligence, chemistry, composite material design, biomedicine, and other fields.

{{</citation>}}


### (34/126) Foundation Models for Weather and Climate Data Understanding: A Comprehensive Survey (Shengchao Chen et al., 2023)

{{<citation>}}

Shengchao Chen, Guodong Long, Jing Jiang, Dikai Liu, Chengqi Zhang. (2023)  
**Foundation Models for Weather and Climate Data Understanding: A Comprehensive Survey**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG, physics-ao-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03014v1)  

---


**ABSTRACT**  
As artificial intelligence (AI) continues to rapidly evolve, the realm of Earth and atmospheric sciences is increasingly adopting data-driven models, powered by progressive developments in deep learning (DL). Specifically, DL techniques are extensively utilized to decode the chaotic and nonlinear aspects of Earth systems, and to address climate challenges via understanding weather and climate data. Cutting-edge performance on specific tasks within narrower spatio-temporal scales has been achieved recently through DL. The rise of large models, specifically large language models (LLMs), has enabled fine-tuning processes that yield remarkable outcomes across various downstream tasks, thereby propelling the advancement of general AI. However, we are still navigating the initial stages of crafting general AI for weather and climate. In this survey, we offer an exhaustive, timely overview of state-of-the-art AI methodologies specifically engineered for weather and climate data, with a special focus on time series and text data. Our primary coverage encompasses four critical aspects: types of weather and climate data, principal model architectures, model scopes and applications, and datasets for weather and climate. Furthermore, in relation to the creation and application of foundation models for weather and climate data understanding, we delve into the field's prevailing challenges, offer crucial insights, and propose detailed avenues for future research. This comprehensive approach equips practitioners with the requisite knowledge to make substantial progress in this domain. Our survey encapsulates the most recent breakthroughs in research on large, data-driven models for weather and climate data understanding, emphasizing robust foundations, current advancements, practical applications, crucial resources, and prospective research opportunities.

{{</citation>}}


## cs.CY (1)



### (35/126) A Comparative Study of AI-Generated (GPT-4) and Human-crafted MCQs in Programming Education (Jacob Doughty et al., 2023)

{{<citation>}}

Jacob Doughty, Zipiao Wan, Anishka Bompelli, Jubahed Qayum, Taozhi Wang, Juran Zhang, Yujia Zheng, Aidan Doyle, Pragnya Sridhar, Arav Agarwal, Christopher Bogart, Eric Keylor, Can Kultur, Jaromir Savelka, Majd Sakr. (2023)  
**A Comparative Study of AI-Generated (GPT-4) and Human-crafted MCQs in Programming Education**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CL, cs-CY, cs.CY  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.03173v1)  

---


**ABSTRACT**  
There is a constant need for educators to develop and maintain effective up-to-date assessments. While there is a growing body of research in computing education on utilizing large language models (LLMs) in generation and engagement with coding exercises, the use of LLMs for generating programming MCQs has not been extensively explored. We analyzed the capability of GPT-4 to produce multiple-choice questions (MCQs) aligned with specific learning objectives (LOs) from Python programming classes in higher education. Specifically, we developed an LLM-powered (GPT-4) system for generation of MCQs from high-level course context and module-level LOs. We evaluated 651 LLM-generated and 449 human-crafted MCQs aligned to 246 LOs from 6 Python courses. We found that GPT-4 was capable of producing MCQs with clear language, a single correct choice, and high-quality distractors. We also observed that the generated MCQs appeared to be well-aligned with the LOs. Our findings can be leveraged by educators wishing to take advantage of the state-of-the-art generative models to support MCQ authoring efforts.

{{</citation>}}


## cs.IR (6)



### (36/126) Combining Counting Processes and Classification Improves a Stopping Rule for Technology Assisted Review (Reem Bin-Hezam et al., 2023)

{{<citation>}}

Reem Bin-Hezam, Mark Stevenson. (2023)  
**Combining Counting Processes and Classification Improves a Stopping Rule for Technology Assisted Review**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2312.03171v1)  

---


**ABSTRACT**  
Technology Assisted Review (TAR) stopping rules aim to reduce the cost of manually assessing documents for relevance by minimising the number of documents that need to be examined to ensure a desired level of recall. This paper extends an effective stopping rule using information derived from a text classifier that can be trained without the need for any additional annotation. Experiments on multiple data sets (CLEF e-Health, TREC Total Recall, TREC Legal and RCV1) showed that the proposed approach consistently improves performance and outperforms several alternative methods.

{{</citation>}}


### (37/126) RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze! (Ronak Pradeep et al., 2023)

{{<citation>}}

Ronak Pradeep, Sahel Sharifymoghaddam, Jimmy Lin. (2023)  
**RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: GPT, GPT-4, LLaMA, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.02724v1)  

---


**ABSTRACT**  
In information retrieval, proprietary large language models (LLMs) such as GPT-4 and open-source counterparts such as LLaMA and Vicuna have played a vital role in reranking. However, the gap between open-source and closed models persists, with reliance on proprietary, non-transparent models constraining reproducibility. Addressing this gap, we introduce RankZephyr, a state-of-the-art, open-source LLM for listwise zero-shot reranking. RankZephyr not only bridges the effectiveness gap with GPT-4 but in some cases surpasses the proprietary model. Our comprehensive evaluations across several datasets (TREC Deep Learning Tracks; NEWS and COVID from BEIR) showcase this ability. RankZephyr benefits from strategic training choices and is resilient against variations in initial document ordering and the number of documents reranked. Additionally, our model outperforms GPT-4 on the NovelEval test set, comprising queries and passages past its training period, which addresses concerns about data contamination. To foster further research in this rapidly evolving field, we provide all code necessary to reproduce our results at https://github.com/castorini/rank_llm.

{{</citation>}}


### (38/126) DRAFT: Dense Retrieval Augmented Few-shot Topic classifier Framework (Keonwoo Kim et al., 2023)

{{<citation>}}

Keonwoo Kim, Younggun Lee. (2023)  
**DRAFT: Dense Retrieval Augmented Few-shot Topic classifier Framework**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2312.02532v1)  

---


**ABSTRACT**  
With the growing volume of diverse information, the demand for classifying arbitrary topics has become increasingly critical. To address this challenge, we introduce DRAFT, a simple framework designed to train a classifier for few-shot topic classification. DRAFT uses a few examples of a specific topic as queries to construct Customized dataset with a dense retriever model. Multi-query retrieval (MQR) algorithm, which effectively handles multiple queries related to a specific topic, is applied to construct the Customized dataset. Subsequently, we fine-tune a classifier using the Customized dataset to identify the topic. To demonstrate the efficacy of our proposed approach, we conduct evaluations on both widely used classification benchmark datasets and manually constructed datasets with 291 diverse topics, which simulate diverse contents encountered in real-world applications. DRAFT shows competitive or superior performance compared to baselines that use in-context learning, such as GPT-3 175B and InstructGPT 175B, on few-shot topic classification tasks despite having 177 times fewer parameters, demonstrating its effectiveness.

{{</citation>}}


### (39/126) LLaRA: Aligning Large Language Models with Sequential Recommenders (Jiayi Liao et al., 2023)

{{<citation>}}

Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, Xiang Wang, Xiangnan He. (2023)  
**LLaRA: Aligning Large Language Models with Sequential Recommenders**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.02445v1)  

---


**ABSTRACT**  
Sequential recommendation aims to predict the subsequent items matching user preference based on her/his historical interactions. With the development of Large Language Models (LLMs), there is growing interest in exploring the potential of LLMs for sequential recommendation by framing it as a language modeling task. Prior works represent items in the textual prompts using either ID indexing or text indexing and feed the prompts into LLMs, but falling short of either encapsulating comprehensive world knowledge or exhibiting sufficient sequential understanding. To harness the complementary strengths of traditional recommenders (which encode user behavioral knowledge) and LLMs (which possess world knowledge about items), we propose LLaRA -- a Large Language and Recommendation Assistant framework. Specifically, LLaRA represents items in LLM's input prompts using a novel hybrid approach that integrates ID-based item embeddings from traditional recommenders with textual item features. Viewing the ``sequential behavior of the user'' as a new modality in recommendation, we employ an adapter to bridge the modality gap between ID embeddings of the traditional recommenders and the input space of LLMs. Furthermore, instead of directly exposing the hybrid prompt to LLMs, we apply a curriculum learning approach to gradually ramp up training complexity. We first warm up the LLM with text-only prompting, which aligns more naturally with the LLM's language modeling capabilities. Thereafter, we progressively transition to hybrid prompting, training the adapter to incorporate behavioral knowledge from the traditional sequential recommender into the LLM. Extensive experiments demonstrate the efficacy of LLaRA framework. Our code and data are available at https://github.com/ljy0ustc/LLaRA .

{{</citation>}}


### (40/126) E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation (Xinhang Li et al., 2023)

{{<citation>}}

Xinhang Li, Chong Chen, Xiangyu Zhao, Yong Zhang, Chunxiao Xing. (2023)  
**E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.02443v1)  

---


**ABSTRACT**  
The recent advancements in Large Language Models (LLMs) have sparked interest in harnessing their potential within recommender systems. Since LLMs are designed for natural language tasks, existing recommendation approaches have predominantly transformed recommendation tasks into open-domain natural language generation tasks. However, this approach necessitates items to possess rich semantic information, often generates out-of-range results, and suffers from notably low efficiency and limited extensibility. Furthermore, practical ID-based recommendation strategies, reliant on a huge number of unique identities (IDs) to represent users and items, have gained prominence in real-world recommender systems due to their effectiveness and efficiency. Nevertheless, the incapacity of LLMs to model IDs presents a formidable challenge when seeking to leverage LLMs for personalized recommendations. In this paper, we introduce an Elegant Effective Efficient Extensible solution for large language models for Sequential Recommendation (E4SRec), which seamlessly integrates LLMs with traditional recommender systems that exclusively utilize IDs to represent items. Specifically, E4SRec takes ID sequences as inputs, ensuring that the generated outputs fall within the candidate lists. Furthermore, E4SRec possesses the capability to generate the entire ranking list in a single forward process, and demands only a minimal set of pluggable parameters, which are trained for each dataset while keeping the entire LLM frozen. We substantiate the effectiveness, efficiency, and extensibility of our proposed E4SRec through comprehensive experiments conducted on four widely-used real-world datasets. The implementation code is accessible at https://github.com/HestiaSky/E4SRec/.

{{</citation>}}


### (41/126) PEFA: Parameter-Free Adapters for Large-scale Embedding-based Retrieval Models (Wei-Cheng Chang et al., 2023)

{{<citation>}}

Wei-Cheng Chang, Jyun-Yu Jiang, Jiong Zhang, Mutasem Al-Darabsah, Choon Hui Teo, Cho-Jui Hsieh, Hsiang-Fu Yu, S. V. N. Vishwanathan. (2023)  
**PEFA: Parameter-Free Adapters for Large-scale Embedding-based Retrieval Models**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Embedding, QA  
[Paper Link](http://arxiv.org/abs/2312.02429v2)  

---


**ABSTRACT**  
Embedding-based Retrieval Models (ERMs) have emerged as a promising framework for large-scale text retrieval problems due to powerful large language models. Nevertheless, fine-tuning ERMs to reach state-of-the-art results can be expensive due to the extreme scale of data as well as the complexity of multi-stages pipelines (e.g., pre-training, fine-tuning, distillation). In this work, we propose the PEFA framework, namely ParamEter-Free Adapters, for fast tuning of ERMs without any backward pass in the optimization. At index building stage, PEFA equips the ERM with a non-parametric k-nearest neighbor (kNN) component. At inference stage, PEFA performs a convex combination of two scoring functions, one from the ERM and the other from the kNN. Based on the neighborhood definition, PEFA framework induces two realizations, namely PEFA-XL (i.e., extra large) using double ANN indices and PEFA-XS (i.e., extra small) using a single ANN index. Empirically, PEFA achieves significant improvement on two retrieval applications. For document retrieval, regarding Recall@100 metric, PEFA improves not only pre-trained ERMs on Trivia-QA by an average of 13.2%, but also fine-tuned ERMs on NQ-320K by an average of 5.5%, respectively. For product search, PEFA improves the Recall@100 of the fine-tuned ERMs by an average of 5.3% and 14.5%, for PEFA-XS and PEFA-XL, respectively. Our code is available at https://github.com/amzn/pecos/tree/mainline/examples/pefa-wsdm24.

{{</citation>}}


## cs.AR (1)



### (42/126) A Hardware Evaluation Framework for Large Language Model Inference (Hengrui Zhang et al., 2023)

{{<citation>}}

Hengrui Zhang, August Ning, Rohan Prabhakar, David Wentzlaff. (2023)  
**A Hardware Evaluation Framework for Large Language Model Inference**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-DC, cs-LG, cs.AR  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.03134v1)  

---


**ABSTRACT**  
The past year has witnessed the increasing popularity of Large Language Models (LLMs). Their unprecedented scale and associated high hardware cost have impeded their broader adoption, calling for efficient hardware designs. With the large hardware needed to simply run LLM inference, evaluating different hardware designs becomes a new bottleneck.   This work introduces LLMCompass, a hardware evaluation framework for LLM inference workloads. LLMCompass is fast, accurate, versatile, and able to describe and evaluate different hardware designs. LLMCompass includes a mapper to automatically find performance-optimal mapping and scheduling. It also incorporates an area-based cost model to help architects reason about their design choices. Compared to real-world hardware, LLMCompass' estimated latency achieves an average 10.4% error rate across various operators with various input sizes and an average 4.1% error rate for LLM inference. With LLMCompass, simulating a 4-NVIDIA A100 GPU node running GPT-3 175B inference can be done within 16 minutes on commodity hardware, including 26,400 rounds of the mapper's parameter search.   With the aid of LLMCompass, this work draws architectural implications and explores new cost-effective hardware designs. By reducing the compute capability or replacing High Bandwidth Memory (HBM) with traditional DRAM, these new designs can achieve as much as 3.41x improvement in performance/cost compared to an NVIDIA A100, making them promising choices for democratizing LLMs.   LLMCompass is planned to be fully open-source.

{{</citation>}}


## eess.IV (5)



### (43/126) Predicting Bone Degradation Using Vision Transformer and Synthetic Cellular Microstructures Dataset (Mohammad Saber Hashemi et al., 2023)

{{<citation>}}

Mohammad Saber Hashemi, Azadeh Sheidaei. (2023)  
**Predicting Bone Degradation Using Vision Transformer and Synthetic Cellular Microstructures Dataset**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV, physics-med-ph  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.03133v1)  

---


**ABSTRACT**  
Bone degradation, especially for astronauts in microgravity conditions, is crucial for space exploration missions since the lower applied external forces accelerate the diminution in bone stiffness and strength substantially. Although existing computational models help us understand this phenomenon and possibly restrict its effect in the future, they are time-consuming to simulate the changes in the bones, not just the bone microstructures, of each individual in detail. In this study, a robust yet fast computational method to predict and visualize bone degradation has been developed. Our deep-learning method, TransVNet, can take in different 3D voxelized images and predict their evolution throughout months utilizing a hybrid 3D-CNN-VisionTransformer autoencoder architecture. Because of limited available experimental data and challenges of obtaining new samples, a digital twin dataset of diverse and initial bone-like microstructures was generated to train our TransVNet on the evolution of the 3D images through a previously developed degradation model for microgravity.

{{</citation>}}


### (44/126) Learning Cortical Anomaly through Masked Encoding for Unsupervised Heterogeneity Mapping (Hao-Chun Yang et al., 2023)

{{<citation>}}

Hao-Chun Yang, Ole Andreassen, Lars Tjelta Westlye, Andre F. Marquand, Christian F. Beckmann, Thomas Wolfers. (2023)  
**Learning Cortical Anomaly through Masked Encoding for Unsupervised Heterogeneity Mapping**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.02762v1)  

---


**ABSTRACT**  
The detection of heterogeneous mental disorders based on brain readouts remains challenging due to the complexity of symptoms and the absence of reliable biomarkers. This paper introduces CAM (Cortical Anomaly Detection through Masked Image Modeling), a novel self-supervised framework designed for the unsupervised detection of complex brain disorders using cortical surface features. We employ this framework for the detection of individuals on the psychotic spectrum and demonstrate its capabilities compared to state-ofthe-art methods, achieving an AUC of 0.696 for Schizoaffective and 0.769 for Schizophreniform, without the need for any labels. Furthermore, the analysis of atypical cortical regions includes Pars Triangularis and several frontal areas, often implicated in schizophrenia, provide further confidence in our approach. Altogether, we demonstrate a scalable approach for anomaly detection of complex brain disorders based on cortical abnormalities.

{{</citation>}}


### (45/126) C3: High-performance and low-complexity neural compression from a single image or video (Hyunjik Kim et al., 2023)

{{<citation>}}

Hyunjik Kim, Matthias Bauer, Lucas Theis, Jonathan Richard Schwarz, Emilien Dupont. (2023)  
**C3: High-performance and low-complexity neural compression from a single image or video**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV, stat-ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.02753v1)  

---


**ABSTRACT**  
Most neural compression models are trained on large datasets of images or videos in order to generalize to unseen data. Such generalization typically requires large and expressive architectures with a high decoding complexity. Here we introduce C3, a neural compression method with strong rate-distortion (RD) performance that instead overfits a small model to each image or video separately. The resulting decoding complexity of C3 can be an order of magnitude lower than neural baselines with similar RD performance. C3 builds on COOL-CHIC (Ladune et al.) and makes several simple and effective improvements for images. We further develop new methodology to apply C3 to videos. On the CLIC2020 image benchmark, we match the RD performance of VTM, the reference implementation of the H.266 codec, with less than 3k MACs/pixel for decoding. On the UVG video benchmark, we match the RD performance of the Video Compression Transformer (Mentzer et al.), a well-established neural video codec, with less than 5k MACs/pixel for decoding.

{{</citation>}}


### (46/126) Enhanced Breast Cancer Tumor Classification using MobileNetV2: A Detailed Exploration on Image Intensity, Error Mitigation, and Streamlit-driven Real-time Deployment (Aaditya Surya et al., 2023)

{{<citation>}}

Aaditya Surya, Aditya Shah, Jarnell Kabore, Subash Sasikumar. (2023)  
**Enhanced Breast Cancer Tumor Classification using MobileNetV2: A Detailed Exploration on Image Intensity, Error Mitigation, and Streamlit-driven Real-time Deployment**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2312.03020v1)  

---


**ABSTRACT**  
This research introduces a sophisticated transfer learning model based on Google's MobileNetV2 for breast cancer tumor classification into normal, benign, and malignant categories, utilizing a dataset of 1576 ultrasound images (265 normal, 891 benign, 420 malignant). The model achieves an accuracy of 0.82, precision of 0.83, recall of 0.81, ROC-AUC of 0.94, PR-AUC of 0.88, and MCC of 0.74. It examines image intensity distributions and misclassification errors, offering improvements for future applications. Addressing dataset imbalances, the study ensures a generalizable model. This work, using a dataset from Baheya Hospital, Cairo, Egypt, compiled by Walid Al-Dhabyani et al., emphasizes MobileNetV2's potential in medical imaging, aiming to improve diagnostic precision in oncology. Additionally, the paper explores Streamlit-based deployment for real-time tumor classification, demonstrating MobileNetV2's applicability in medical imaging and setting a benchmark for future research in oncology diagnostics.

{{</citation>}}


### (47/126) Breast Ultrasound Report Generation using LangChain (Jaeyoung Huh et al., 2023)

{{<citation>}}

Jaeyoung Huh, Hyun Jeong Park, Jong Chul Ye. (2023)  
**Breast Ultrasound Report Generation using LangChain**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.03013v1)  

---


**ABSTRACT**  
Breast ultrasound (BUS) is a critical diagnostic tool in the field of breast imaging, aiding in the early detection and characterization of breast abnormalities. Interpreting breast ultrasound images commonly involves creating comprehensive medical reports, containing vital information to promptly assess the patient's condition. However, the ultrasound imaging system necessitates capturing multiple images of various parts to compile a single report, presenting a time-consuming challenge. To address this problem, we propose the integration of multiple image analysis tools through a LangChain using Large Language Models (LLM), into the breast reporting process. Through a combination of designated tools and text generation through LangChain, our method can accurately extract relevant features from ultrasound images, interpret them in a clinical context, and produce comprehensive and standardized reports. This approach not only reduces the burden on radiologists and healthcare professionals but also enhances the consistency and quality of reports. The extensive experiments shows that each tools involved in the proposed method can offer qualitatively and quantitatively significant results. Furthermore, clinical evaluation on the generated reports demonstrates that the proposed method can make report in clinically meaningful way.

{{</citation>}}


## cs.CL (18)



### (48/126) Assertion Enhanced Few-Shot Learning: Instructive Technique for Large Language Models to Generate Educational Explanations (Tasmia Shahriar et al., 2023)

{{<citation>}}

Tasmia Shahriar, Noboru Matsuda, Kelly Ramos. (2023)  
**Assertion Enhanced Few-Shot Learning: Instructive Technique for Large Language Models to Generate Educational Explanations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Few-Shot, Language Model  
[Paper Link](http://arxiv.org/abs/2312.03122v1)  

---


**ABSTRACT**  
Human educators possess an intrinsic ability to anticipate and seek educational explanations from students, which drives them to pose thought-provoking questions when students cannot articulate these explanations independently. We aim to imbue Intelligent Tutoring Systems with this ability using few-shot learning capability of Large Language Models. Our work proposes a novel prompting technique, Assertion Enhanced Few-Shot Learning, to facilitate the generation of accurate, detailed oriented educational explanations. Our central hypothesis is that, in educational domain, few-shot demonstrations are necessary but not a sufficient condition for quality explanation generation. We conducted a study involving 12 in-service teachers, comparing our approach to Traditional Few-Shot Learning. The results show that Assertion Enhanced Few-Shot Learning improves explanation accuracy by 15% and yields higher-quality explanations, as evaluated by teachers. We also conduct a qualitative ablation study to factor the impact of assertions to provide educator-friendly prompting guidelines for generating explanations in their domain of interest.

{{</citation>}}


### (49/126) Understanding Environmental Posts: Sentiment and Emotion Analysis of Social Media Data (Daniyar Amangeldi et al., 2023)

{{<citation>}}

Daniyar Amangeldi, Aida Usmanova, Pakizar Shamoi. (2023)  
**Understanding Environmental Posts: Sentiment and Emotion Analysis of Social Media Data**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2312.03095v1)  

---


**ABSTRACT**  
Social media is now the predominant source of information due to the availability of immediate public response. As a result, social media data has become a valuable resource for comprehending public sentiments. Studies have shown that it can amplify ideas and influence public sentiments. This study analyzes the public perception of climate change and the environment over a decade from 2014 to 2023. Using the Pointwise Mutual Information (PMI) algorithm, we identify sentiment and explore prevailing emotions expressed within environmental tweets across various social media platforms, namely Twitter, Reddit, and YouTube. Accuracy on a human-annotated dataset was 0.65, higher than Vader score but lower than that of an expert rater (0.90). Our findings suggest that negative environmental tweets are far more common than positive or neutral ones. Climate change, air quality, emissions, plastic, and recycling are the most discussed topics on all social media platforms, highlighting its huge global concern. The most common emotions in environmental tweets are fear, trust, and anticipation, demonstrating public reactions wide and complex nature. By identifying patterns and trends in opinions related to the environment, we hope to provide insights that can help raise awareness regarding environmental issues, inform the development of interventions, and adapt further actions to meet environmental challenges.

{{</citation>}}


### (50/126) LLMs for Multi-Modal Knowledge Extraction and Analysis in Intelligence/Safety-Critical Applications (Brett Israelsen et al., 2023)

{{<citation>}}

Brett Israelsen, Soumalya Sarkar. (2023)  
**LLMs for Multi-Modal Knowledge Extraction and Analysis in Intelligence/Safety-Critical Applications**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.03088v1)  

---


**ABSTRACT**  
Large Language Models have seen rapid progress in capability in recent years; this progress has been accelerating and their capabilities, measured by various benchmarks, are beginning to approach those of humans. There is a strong demand to use such models in a wide variety of applications but, due to unresolved vulnerabilities and limitations, great care needs to be used before applying them to intelligence and safety-critical applications. This paper reviews recent literature related to LLM assessment and vulnerabilities to synthesize the current research landscape and to help understand what advances are most critical to enable use of of these technologies in intelligence and safety-critical applications. The vulnerabilities are broken down into ten high-level categories and overlaid onto a high-level life cycle of an LLM. Some general categories of mitigations are reviewed.

{{</citation>}}


### (51/126) Clinical Notes Reveal Physician Fatigue (Chao-Chun Hsu et al., 2023)

{{<citation>}}

Chao-Chun Hsu, Ziad Obermeyer, Chenhao Tan. (2023)  
**Clinical Notes Reveal Physician Fatigue**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2312.03077v1)  

---


**ABSTRACT**  
Physicians write notes about patients. In doing so, they reveal much about themselves. Using data from 129,228 emergency room visits, we train a model to identify notes written by fatigued physicians -- those who worked 5 or more of the prior 7 days. In a hold-out set, the model accurately identifies notes written by these high-workload physicians, and also flags notes written in other high-fatigue settings: on overnight shifts, and after high patient volumes. Model predictions also correlate with worse decision-making on at least one important metric: yield of testing for heart attack is 18% lower with each standard deviation increase in model-predicted fatigue. Finally, the model indicates that notes written about Black and Hispanic patients have 12% and 21% higher predicted fatigue than Whites -- larger than overnight vs. daytime differences. These results have an important implication for large language models (LLMs). Our model indicates that fatigued doctors write more predictable notes. Perhaps unsurprisingly, because word prediction is the core of how LLMs work, we find that LLM-written notes have 17% higher predicted fatigue than real physicians' notes. This indicates that LLMs may introduce distortions in generated text that are not yet fully understood.

{{</citation>}}


### (52/126) Rank-without-GPT: Building GPT-Independent Listwise Rerankers on Open-Source Large Language Models (Xinyu Zhang et al., 2023)

{{<citation>}}

Xinyu Zhang, Sebastian Hofstätter, Patrick Lewis, Raphael Tang, Jimmy Lin. (2023)  
**Rank-without-GPT: Building GPT-Independent Listwise Rerankers on Open-Source Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.02969v1)  

---


**ABSTRACT**  
Listwise rerankers based on large language models (LLM) are the zero-shot state-of-the-art. However, current works in this direction all depend on the GPT models, making it a single point of failure in scientific reproducibility. Moreover, it raises the concern that the current research findings only hold for GPT models but not LLM in general. In this work, we lift this pre-condition and build for the first time effective listwise rerankers without any form of dependency on GPT. Our passage retrieval experiments show that our best list se reranker surpasses the listwise rerankers based on GPT-3.5 by 13% and achieves 97% effectiveness of the ones built on GPT-4. Our results also show that the existing training datasets, which were expressly constructed for pointwise ranking, are insufficient for building such listwise rerankers. Instead, high-quality listwise ranking data is required and crucial, calling for further work on building human-annotated listwise data resources.

{{</citation>}}


### (53/126) WhisBERT: Multimodal Text-Audio Language Modeling on 100M Words (Lukas Wolf et al., 2023)

{{<citation>}}

Lukas Wolf, Klemen Kotar, Greta Tuckute, Eghbal Hosseini, Tamar Regev, Ethan Wilcox, Alex Warstadt. (2023)  
**WhisBERT: Multimodal Text-Audio Language Modeling on 100M Words**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.02931v1)  

---


**ABSTRACT**  
Training on multiple modalities of input can augment the capabilities of a language model. Here, we ask whether such a training regime can improve the quality and efficiency of these systems as well. We focus on text--audio and introduce Whisbert, which is inspired by the text--image approach of FLAVA \citep{singh_flava_2022}. In accordance with Babylm \citep{warstadt2023papers} guidelines, we pretrain Whisbert on a dataset comprising only 100 million words plus their corresponding speech from the word-aligned version of the People's Speech dataset \citep{galvez_peoples_2021}. To assess the impact of multimodality, we compare versions of the model that are trained on text only and on both audio and text simultaneously. We find that while Whisbert is able to perform well on multimodal masked modeling and surpasses the Babylm baselines in most benchmark tasks, it struggles to optimize its complex objective and outperform its text-only Whisbert baseline.

{{</citation>}}


### (54/126) Let the LLMs Talk: Simulating Human-to-Human Conversational QA via Zero-Shot LLM-to-LLM Interactions (Zahra Abbasiantaeb et al., 2023)

{{<citation>}}

Zahra Abbasiantaeb, Yifei Yuan, Evangelos Kanoulas, Mohammad Aliannejadi. (2023)  
**Let the LLMs Talk: Simulating Human-to-Human Conversational QA via Zero-Shot LLM-to-LLM Interactions**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: GPT, GPT-4, QA, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.02913v1)  

---


**ABSTRACT**  
Conversational question-answering (CQA) systems aim to create interactive search systems that effectively retrieve information by interacting with users. To replicate human-to-human conversations, existing work uses human annotators to play the roles of the questioner (student) and the answerer (teacher). Despite its effectiveness, challenges exist as human annotation is time-consuming, inconsistent, and not scalable. To address this issue and investigate the applicability of large language models (LLMs) in CQA simulation, we propose a simulation framework that employs zero-shot learner LLMs for simulating teacher-student interactions. Our framework involves two LLMs interacting on a specific topic, with the first LLM acting as a student, generating questions to explore a given search topic. The second LLM plays the role of a teacher by answering questions and is equipped with additional information, including a text on the given topic. We implement both the student and teacher by zero-shot prompting the GPT-4 model. To assess the effectiveness of LLMs in simulating CQA interactions and understand the disparities between LLM- and human-generated conversations, we evaluate the simulated data from various perspectives. We begin by evaluating the teacher's performance through both automatic and human assessment. Next, we evaluate the performance of the student, analyzing and comparing the disparities between questions generated by the LLM and those generated by humans. Furthermore, we conduct extensive analyses to thoroughly examine the LLM performance by benchmarking state-of-the-art reading comprehension models on both datasets. Our results reveal that the teacher LLM generates lengthier answers that tend to be more accurate and complete. The student LLM generates more diverse questions, covering more aspects of a given topic.

{{</citation>}}


### (55/126) Inherent limitations of LLMs regarding spatial information (He Yan et al., 2023)

{{<citation>}}

He Yan, Xinyao Hu, Xiangpeng Wan, Chengyu Huang, Kai Zou, Shiqi Xu. (2023)  
**Inherent limitations of LLMs regarding spatial information**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.03042v1)  

---


**ABSTRACT**  
Despite the significant advancements in natural language processing capabilities demonstrated by large language models such as ChatGPT, their proficiency in comprehending and processing spatial information, especially within the domains of 2D and 3D route planning, remains notably underdeveloped. This paper investigates the inherent limitations of ChatGPT and similar models in spatial reasoning and navigation-related tasks, an area critical for applications ranging from autonomous vehicle guidance to assistive technologies for the visually impaired. In this paper, we introduce a novel evaluation framework complemented by a baseline dataset, meticulously crafted for this study. This dataset is structured around three key tasks: plotting spatial points, planning routes in two-dimensional (2D) spaces, and devising pathways in three-dimensional (3D) environments. We specifically developed this dataset to assess the spatial reasoning abilities of ChatGPT. Our evaluation reveals key insights into the model's capabilities and limitations in spatial understanding.

{{</citation>}}


### (56/126) Clustering Pseudo Language Family in Multilingual Translation Models with Fisher Information Matrix (Xinyu Ma et al., 2023)

{{<citation>}}

Xinyu Ma, Xuebo Liu, Min Zhang. (2023)  
**Clustering Pseudo Language Family in Multilingual Translation Models with Fisher Information Matrix**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2312.02820v1)  

---


**ABSTRACT**  
In multilingual translation research, the comprehension and utilization of language families are of paramount importance. Nevertheless, clustering languages based solely on their ancestral families can yield suboptimal results due to variations in the datasets employed during the model's training phase. To mitigate this challenge, we introduce an innovative method that leverages the fisher information matrix (FIM) to cluster language families, anchored on the multilingual translation model's characteristics. We hypothesize that language pairs with similar effects on model parameters exhibit a considerable degree of linguistic congruence and should thus be grouped cohesively. This concept has led us to define pseudo language families. We provide an in-depth discussion regarding the inception and application of these pseudo language families. Empirical evaluations reveal that employing these pseudo language families enhances performance over conventional language families in adapting a multilingual translation model to unfamiliar language pairs. The proposed methodology may also be extended to scenarios requiring language similarity measurements. The source code and associated scripts can be accessed at https://github.com/ecoli-hit/PseudoFamily.

{{</citation>}}


### (57/126) Leveraging Domain Adaptation and Data Augmentation to Improve Qur'anic IR in English and Arabic (Vera Pavlova, 2023)

{{<citation>}}

Vera Pavlova. (2023)  
**Leveraging Domain Adaptation and Data Augmentation to Improve Qur'anic IR in English and Arabic**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2312.02803v1)  

---


**ABSTRACT**  
In this work, we approach the problem of Qur'anic information retrieval (IR) in Arabic and English. Using the latest state-of-the-art methods in neural IR, we research what helps to tackle this task more efficiently. Training retrieval models requires a lot of data, which is difficult to obtain for training in-domain. Therefore, we commence with training on a large amount of general domain data and then continue training on in-domain data. To handle the lack of in-domain data, we employed a data augmentation technique, which considerably improved results in MRR@10 and NDCG@5 metrics, setting the state-of-the-art in Qur'anic IR for both English and Arabic. The absence of an Islamic corpus and domain-specific model for IR task in English motivated us to address this lack of resources and take preliminary steps of the Islamic corpus compilation and domain-specific language model (LM) pre-training, which helped to improve the performance of the retrieval models that use the domain-specific LM as the shared backbone. We examined several language models (LMs) in Arabic to select one that efficiently deals with the Qur'anic IR task. Besides transferring successful experiments from English to Arabic, we conducted additional experiments with retrieval task in Arabic to amortize the scarcity of general domain datasets used to train the retrieval models. Handling Qur'anic IR task combining English and Arabic allowed us to enhance the comparison and share valuable insights across models and languages.

{{</citation>}}


### (58/126) Large Language Models on Graphs: A Comprehensive Survey (Bowen Jin et al., 2023)

{{<citation>}}

Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, Jiawei Han. (2023)  
**Large Language Models on Graphs: A Comprehensive Survey**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2312.02783v1)  

---


**ABSTRACT**  
Large language models (LLMs), such as ChatGPT and LLaMA, are creating significant advancements in natural language processing, due to their strong text encoding/decoding ability and newly found emergent capability (e.g., reasoning). While LLMs are mainly designed to process pure texts, there are many real-world scenarios where text data are associated with rich structure information in the form of graphs (e.g., academic networks, and e-commerce networks) or scenarios where graph data are paired with rich textual information (e.g., molecules with descriptions). Besides, although LLMs have shown their pure text-based reasoning ability, it is underexplored whether such ability can be generalized to graph scenarios (i.e., graph-based reasoning). In this paper, we provide a systematic review of scenarios and techniques related to large language models on graphs. We first summarize potential scenarios of adopting LLMs on graphs into three categories, namely pure graphs, text-rich graphs, and text-paired graphs. We then discuss detailed techniques for utilizing LLMs on graphs, including LLM as Predictor, LLM as Encoder, and LLM as Aligner, and compare the advantages and disadvantages of different schools of models. Furthermore, we mention the real-world applications of such methods and summarize open-source codes and benchmark datasets. Finally, we conclude with potential future research directions in this fast-growing field. The related source can be found at https://github.com/PeterGriffinJin/Awesome-Language-Model-on-Graphs.

{{</citation>}}


### (59/126) Compositional Generalization for Data-to-Text Generation (Xinnuo Xu et al., 2023)

{{<citation>}}

Xinnuo Xu, Ivan Titov, Mirella Lapata. (2023)  
**Compositional Generalization for Data-to-Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: T5, Text Generation  
[Paper Link](http://arxiv.org/abs/2312.02748v1)  

---


**ABSTRACT**  
Data-to-text generation involves transforming structured data, often represented as predicate-argument tuples, into coherent textual descriptions. Despite recent advances, systems still struggle when confronted with unseen combinations of predicates, producing unfaithful descriptions (e.g. hallucinations or omissions). We refer to this issue as compositional generalisation, and it encouraged us to create a benchmark for assessing the performance of different approaches on this specific problem. Furthermore, we propose a novel model that addresses compositional generalization by clustering predicates into groups. Our model generates text in a sentence-by-sentence manner, relying on one cluster of predicates at a time. This approach significantly outperforms T5~baselines across all evaluation metrics.Notably, it achieved a 31% improvement over T5 in terms of a metric focused on maintaining faithfulness to the input.

{{</citation>}}


### (60/126) Text Intimacy Analysis using Ensembles of Multilingual Transformers (Tanmay Chavan et al., 2023)

{{<citation>}}

Tanmay Chavan, Ved Patwardhan. (2023)  
**Text Intimacy Analysis using Ensembles of Multilingual Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.02590v1)  

---


**ABSTRACT**  
Intimacy estimation of a given text has recently gained importance due to the increase in direct interaction of NLP systems with humans. Intimacy is an important aspect of natural language and has a substantial impact on our everyday communication. Thus the level of intimacy can provide us with deeper insights and richer semantics of conversations. In this paper, we present our work on the SemEval shared task 9 on predicting the level of intimacy for the given text. The dataset consists of tweets in ten languages, out of which only six are available in the training dataset. We conduct several experiments and show that an ensemble of multilingual models along with a language-specific monolingual model has the best performance. We also evaluate other data augmentation methods such as translation and present the results. Lastly, we study the results thoroughly and present some noteworthy insights into this problem.

{{</citation>}}


### (61/126) Empathy and Distress Detection using Ensembles of Transformer Models (Tanmay Chavan et al., 2023)

{{<citation>}}

Tanmay Chavan, Kshitij Deshpande, Sheetal Sonawane. (2023)  
**Empathy and Distress Detection using Ensembles of Transformer Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2312.02578v1)  

---


**ABSTRACT**  
This paper presents our approach for the WASSA 2023 Empathy, Emotion and Personality Shared Task. Empathy and distress are human feelings that are implicitly expressed in natural discourses. Empathy and distress detection are crucial challenges in Natural Language Processing that can aid our understanding of conversations. The provided dataset consists of several long-text examples in the English language, with each example associated with a numeric score for empathy and distress. We experiment with several BERT-based models as a part of our approach. We also try various ensemble methods. Our final submission has a Pearson's r score of 0.346, placing us third in the empathy and distress detection subtask.

{{</citation>}}


### (62/126) MKA: A Scalable Medical Knowledge Assisted Mechanism for Generative Models on Medical Conversation Tasks (Ke Liang et al., 2023)

{{<citation>}}

Ke Liang, Sifan Wu, Jiayi Gu. (2023)  
**MKA: A Scalable Medical Knowledge Assisted Mechanism for Generative Models on Medical Conversation Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Dialog, Dialogue, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2312.02496v1)  

---


**ABSTRACT**  
Using natural language processing (NLP) technologies to develop medical chatbots makes the diagnosis of the patient more convenient and efficient, which is a typical application in healthcare AI. Because of its importance, lots of research have been come out. Recently, the neural generative models have shown their impressive ability as the core of chatbot, while it cannot scale well when directly applied to medical conversation due to the lack of medical-specific knowledge. To address the limitation, a scalable Medical Knowledge Assisted mechanism, MKA, is proposed in this paper. The mechanism aims to assist general neural generative models to achieve better performance on the medical conversation task. The medical-specific knowledge graph is designed within the mechanism, which contains 6 types of medical-related information, including department, drug, check, symptom, disease, food. Besides, the specific token concatenation policy is defined to effectively inject medical information into the input data. Evaluation of our method is carried out on two typical medical datasets, MedDG and MedDialog-CN. The evaluation results demonstrate that models combined with our mechanism outperform original methods in multiple automatic evaluation metrics. Besides, MKA-Bert-GPT achieves state-of-the-art performance. The open-sourced codes are public: https://github.com/LIANGKE23/Knowledge_Assisted_Medical_Dialogue_Generation_Mechanism

{{</citation>}}


### (63/126) Visually Grounded Language Learning: a review of language games, datasets, tasks, and models (Alessandro Suglia et al., 2023)

{{<citation>}}

Alessandro Suglia, Ioannis Konstas, Oliver Lemon. (2023)  
**Visually Grounded Language Learning: a review of language games, datasets, tasks, and models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Natural Language Understanding  
[Paper Link](http://arxiv.org/abs/2312.02431v1)  

---


**ABSTRACT**  
In recent years, several machine learning models have been proposed. They are trained with a language modelling objective on large-scale text-only data. With such pretraining, they can achieve impressive results on many Natural Language Understanding and Generation tasks. However, many facets of meaning cannot be learned by ``listening to the radio" only. In the literature, many Vision+Language (V+L) tasks have been defined with the aim of creating models that can ground symbols in the visual modality. In this work, we provide a systematic literature review of several tasks and models proposed in the V+L field. We rely on Wittgenstein's idea of `language games' to categorise such tasks into 3 different families: 1) discriminative games, 2) generative games, and 3) interactive games. Our analysis of the literature provides evidence that future work should be focusing on interactive games where communication in Natural Language is important to resolve ambiguities about object referents and action plans and that physical embodiment is essential to understand the semantics of situations and events. Overall, these represent key requirements for developing grounded meanings in neural models.

{{</citation>}}


### (64/126) Decoding Data Quality via Synthetic Corruptions: Embedding-guided Pruning of Code Data (Yu Yang et al., 2023)

{{<citation>}}

Yu Yang, Aaditya K. Singh, Mostafa Elhoushi, Anas Mahmoud, Kushal Tirumala, Fabian Gloeckle, Baptiste Rozière, Carole-Jean Wu, Ari S. Morcos, Newsha Ardalani. (2023)  
**Decoding Data Quality via Synthetic Corruptions: Embedding-guided Pruning of Code Data**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Embedding, Language Model, Pruning  
[Paper Link](http://arxiv.org/abs/2312.02418v1)  

---


**ABSTRACT**  
Code datasets, often collected from diverse and uncontrolled sources such as GitHub, potentially suffer from quality issues, thereby affecting the performance and training efficiency of Large Language Models (LLMs) optimized for code generation. Previous studies demonstrated the benefit of using embedding spaces for data pruning, but they mainly focused on duplicate removal or increasing variety, and in other modalities, such as images. Our work focuses on using embeddings to identify and remove "low-quality" code data. First, we explore features of "low-quality" code in embedding space, through the use of synthetic corruptions. Armed with this knowledge, we devise novel pruning metrics that operate in embedding space to identify and remove low-quality entries in the Stack dataset. We demonstrate the benefits of this synthetic corruption informed pruning (SCIP) approach on the well-established HumanEval and MBPP benchmarks, outperforming existing embedding-based methods. Importantly, we achieve up to a 3% performance improvement over no pruning, thereby showing the promise of insights from synthetic corruptions for data pruning.

{{</citation>}}


### (65/126) Efficient Online Data Mixing For Language Model Pre-Training (Alon Albalak et al., 2023)

{{<citation>}}

Alon Albalak, Liangming Pan, Colin Raffel, William Yang Wang. (2023)  
**Efficient Online Data Mixing For Language Model Pre-Training**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.02406v1)  

---


**ABSTRACT**  
The data used to pretrain large language models has a decisive impact on a model's downstream performance, which has led to a large body of work on data selection methods that aim to automatically determine the most suitable data to use for pretraining. Existing data selection methods suffer from slow and computationally expensive processes, a problem amplified by the increasing size of models and of pretraining datasets. Data mixing, on the other hand, reduces the complexity of data selection by grouping data points together and determining sampling probabilities across entire groups. However, data mixing proportions are typically fixed before training and therefore cannot adapt to changing training dynamics. To address these limitations, we develop an efficient algorithm for Online Data Mixing (ODM) that combines elements from both data selection and data mixing. Based on multi-armed bandit algorithms, our online approach optimizes the data mixing proportions during training. Remarkably, our method trains a model that reaches the final perplexity of the next best method with 19\% fewer training iterations, and improves performance on the 5-shot MMLU benchmark by 1.9% relative accuracy, while adding negligible wall-clock time during pretraining.

{{</citation>}}


## cs.CV (37)



### (66/126) AI-SAM: Automatic and Interactive Segment Anything Model (Yimu Pan et al., 2023)

{{<citation>}}

Yimu Pan, Sitao Zhang, Alison D. Gernand, Jeffery A. Goldstein, James Z. Wang. (2023)  
**AI-SAM: Automatic and Interactive Segment Anything Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03119v1)  

---


**ABSTRACT**  
Semantic segmentation is a core task in computer vision. Existing methods are generally divided into two categories: automatic and interactive. Interactive approaches, exemplified by the Segment Anything Model (SAM), have shown promise as pre-trained models. However, current adaptation strategies for these models tend to lean towards either automatic or interactive approaches. Interactive methods depend on prompts user input to operate, while automatic ones bypass the interactive promptability entirely. Addressing these limitations, we introduce a novel paradigm and its first model: the Automatic and Interactive Segment Anything Model (AI-SAM). In this paradigm, we conduct a comprehensive analysis of prompt quality and introduce the pioneering Automatic and Interactive Prompter (AI-Prompter) that automatically generates initial point prompts while accepting additional user inputs. Our experimental results demonstrate AI-SAM's effectiveness in the automatic setting, achieving state-of-the-art performance. Significantly, it offers the flexibility to incorporate additional user prompts, thereby further enhancing its performance. The project page is available at https://github.com/ymp5078/AI-SAM.

{{</citation>}}


### (67/126) ScAR: Scaling Adversarial Robustness for LiDAR Object Detection (Xiaohu Lu et al., 2023)

{{<citation>}}

Xiaohu Lu, Hayder Radha. (2023)  
**ScAR: Scaling Adversarial Robustness for LiDAR Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.03085v1)  

---


**ABSTRACT**  
The adversarial robustness of a model is its ability to resist adversarial attacks in the form of small perturbations to input data. Universal adversarial attack methods such as Fast Sign Gradient Method (FSGM) and Projected Gradient Descend (PGD) are popular for LiDAR object detection, but they are often deficient compared to task-specific adversarial attacks. Additionally, these universal methods typically require unrestricted access to the model's information, which is difficult to obtain in real-world applications. To address these limitations, we present a black-box Scaling Adversarial Robustness (ScAR) method for LiDAR object detection. By analyzing the statistical characteristics of 3D object detection datasets such as KITTI, Waymo, and nuScenes, we have found that the model's prediction is sensitive to scaling of 3D instances. We propose three black-box scaling adversarial attack methods based on the available information: model-aware attack, distribution-aware attack, and blind attack. We also introduce a strategy for generating scaling adversarial examples to improve the model's robustness against these three scaling adversarial attacks. Comparison with other methods on public datasets under different 3D object detection architectures demonstrates the effectiveness of our proposed method.

{{</citation>}}


### (68/126) GPT4Point: A Unified Framework for Point-Language Understanding and Generation (Zhangyang Qi et al., 2023)

{{<citation>}}

Zhangyang Qi, Ye Fang, Zeyi Sun, Xiaoyang Wu, Tong Wu, Jiaqi Wang, Dahua Lin, Hengshuang Zhao. (2023)  
**GPT4Point: A Unified Framework for Point-Language Understanding and Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.02980v1)  

---


**ABSTRACT**  
Multimodal Large Language Models (MLLMs) have excelled in 2D image-text comprehension and image generation, but their understanding of the 3D world is notably deficient, limiting progress in 3D language understanding and generation. To solve this problem, we introduce GPT4Point, an innovative groundbreaking point-language multimodal model designed specifically for unified 3D object understanding and generation within the MLLM framework. GPT4Point as a powerful 3D MLLM seamlessly can execute a variety of point-text reference tasks such as point-cloud captioning and Q&A. Additionally, GPT4Point is equipped with advanced capabilities for controllable 3D generation, it can get high-quality results through a low-quality point-text feature maintaining the geometric shapes and colors. To support the expansive needs of 3D object-text pairs, we develop Pyramid-XL, a point-language dataset annotation engine. It constructs a large-scale database over 1M objects of varied text granularity levels from the Objaverse-XL dataset, essential for training GPT4Point. A comprehensive benchmark has been proposed to evaluate 3D point-language understanding capabilities. In extensive evaluations, GPT4Point has demonstrated superior performance in understanding and generation.

{{</citation>}}


### (69/126) Describing Differences in Image Sets with Natural Language (Lisa Dunlap et al., 2023)

{{<citation>}}

Lisa Dunlap, Yuhui Zhang, Xiaohan Wang, Ruiqi Zhong, Trevor Darrell, Jacob Steinhardt, Joseph E. Gonzalez, Serena Yeung-Levy. (2023)  
**Describing Differences in Image Sets with Natural Language**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-CY, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.02974v1)  

---


**ABSTRACT**  
How do two sets of images differ? Discerning set-level differences is crucial for understanding model behaviors and analyzing datasets, yet manually sifting through thousands of images is impractical. To aid in this discovery process, we explore the task of automatically describing the differences between two $\textbf{sets}$ of images, which we term Set Difference Captioning. This task takes in image sets $D_A$ and $D_B$, and outputs a description that is more often true on $D_A$ than $D_B$. We outline a two-stage approach that first proposes candidate difference descriptions from image sets and then re-ranks the candidates by checking how well they can differentiate the two sets. We introduce VisDiff, which first captions the images and prompts a language model to propose candidate descriptions, then re-ranks these descriptions using CLIP. To evaluate VisDiff, we collect VisDiffBench, a dataset with 187 paired image sets with ground truth difference descriptions. We apply VisDiff to various domains, such as comparing datasets (e.g., ImageNet vs. ImageNetV2), comparing classification models (e.g., zero-shot CLIP vs. supervised ResNet), summarizing model failure modes (supervised ResNet), characterizing differences between generative models (e.g., StableDiffusionV1 and V2), and discovering what makes images memorable. Using VisDiff, we are able to find interesting and previously unknown differences in datasets and models, demonstrating its utility in revealing nuanced insights.

{{</citation>}}


### (70/126) Visual Program Distillation: Distilling Tools and Programmatic Reasoning into Vision-Language Models (Yushi Hu et al., 2023)

{{<citation>}}

Yushi Hu, Otilia Stretcu, Chun-Ta Lu, Krishnamurthy Viswanathan, Kenji Hata, Enming Luo, Ranjay Krishna, Ariel Fuxman. (2023)  
**Visual Program Distillation: Distilling Tools and Programmatic Reasoning into Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.03052v1)  

---


**ABSTRACT**  
Solving complex visual tasks such as "Who invented the musical instrument on the right?" involves a composition of skills: understanding space, recognizing instruments, and also retrieving prior knowledge. Recent work shows promise by decomposing such tasks using a large language model (LLM) into an executable program that invokes specialized vision models. However, generated programs are error-prone: they omit necessary steps, include spurious ones, and are unable to recover when the specialized models give incorrect outputs. Moreover, they require loading multiple models, incurring high latency and computation costs. We propose Visual Program Distillation (VPD), an instruction tuning framework that produces a vision-language model (VLM) capable of solving complex visual tasks with a single forward pass. VPD distills the reasoning ability of LLMs by using them to sample multiple candidate programs, which are then executed and verified to identify a correct one. It translates each correct program into a language description of the reasoning steps, which are then distilled into a VLM. Extensive experiments show that VPD improves the VLM's ability to count, understand spatial relations, and reason compositionally. Our VPD-trained PaLI-X outperforms all prior VLMs, achieving state-of-the-art performance across complex vision tasks, including MMBench, OK-VQA, A-OKVQA, TallyQA, POPE, and Hateful Memes. An evaluation with human annotators also confirms that VPD improves model response factuality and consistency. Finally, experiments on content moderation demonstrate that VPD is also helpful for adaptation to real-world applications with limited data.

{{</citation>}}


### (71/126) Diffusion-SS3D: Diffusion Model for Semi-supervised 3D Object Detection (Cheng-Ju Ho et al., 2023)

{{<citation>}}

Cheng-Ju Ho, Chen-Hsuan Tai, Yen-Yu Lin, Ming-Hsuan Yang, Yi-Hsuan Tsai. (2023)  
**Diffusion-SS3D: Diffusion Model for Semi-supervised 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.02966v1)  

---


**ABSTRACT**  
Semi-supervised object detection is crucial for 3D scene understanding, efficiently addressing the limitation of acquiring large-scale 3D bounding box annotations. Existing methods typically employ a teacher-student framework with pseudo-labeling to leverage unlabeled point clouds. However, producing reliable pseudo-labels in a diverse 3D space still remains challenging. In this work, we propose Diffusion-SS3D, a new perspective of enhancing the quality of pseudo-labels via the diffusion model for semi-supervised 3D object detection. Specifically, we include noises to produce corrupted 3D object size and class label distributions, and then utilize the diffusion model as a denoising process to obtain bounding box outputs. Moreover, we integrate the diffusion model into the teacher-student framework, so that the denoised bounding boxes can be used to improve pseudo-label generation, as well as the entire semi-supervised learning process. We conduct experiments on the ScanNet and SUN RGB-D benchmark datasets to demonstrate that our approach achieves state-of-the-art performance against existing methods. We also present extensive analysis to understand how our diffusion model design affects performance in semi-supervised learning.

{{</citation>}}


### (72/126) Classification for everyone : Building geography agnostic models for fairer recognition (Akshat Jindal et al., 2023)

{{<citation>}}

Akshat Jindal, Shreya Singh, Soham Gadgil. (2023)  
**Classification for everyone : Building geography agnostic models for fairer recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-CY, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.02957v1)  

---


**ABSTRACT**  
In this paper, we analyze different methods to mitigate inherent geographical biases present in state of the art image classification models. We first quantitatively present this bias in two datasets - The Dollar Street Dataset and ImageNet, using images with location information. We then present different methods which can be employed to reduce this bias. Finally, we analyze the effectiveness of the different techniques on making these models more robust to geographical locations of the images.

{{</citation>}}


### (73/126) DGInStyle: Domain-Generalizable Semantic Segmentation with Image Diffusion Models and Stylized Semantic Control (Yuru Jia et al., 2023)

{{<citation>}}

Yuru Jia, Lukas Hoyer, Shengyu Huang, Tianfu Wang, Luc Van Gool, Konrad Schindler, Anton Obukhov. (2023)  
**DGInStyle: Domain-Generalizable Semantic Segmentation with Image Diffusion Models and Stylized Semantic Control**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.03048v1)  

---


**ABSTRACT**  
Large, pretrained latent diffusion models (LDMs) have demonstrated an extraordinary ability to generate creative content, specialize to user data through few-shot fine-tuning, and condition their output on other modalities, such as semantic maps. However, are they usable as large-scale data generators, e.g., to improve tasks in the perception stack, like semantic segmentation? We investigate this question in the context of autonomous driving, and answer it with a resounding "yes". We propose an efficient data generation pipeline termed DGInStyle. First, we examine the problem of specializing a pretrained LDM to semantically-controlled generation within a narrow domain. Second, we design a Multi-resolution Latent Fusion technique to overcome the bias of LDMs towards dominant objects. Third, we propose a Style Swap technique to endow the rich generative prior with the learned semantic control. Using DGInStyle, we generate a diverse dataset of street scenes, train a domain-agnostic semantic segmentation model on it, and evaluate the model on multiple popular autonomous driving datasets. Our approach consistently increases the performance of several domain generalization methods, in some cases by +2.5 mIoU compared to the previous state-of-the-art method without our generative augmentation scheme. Source code and dataset are available at https://dginstyle.github.io .

{{</citation>}}


### (74/126) MIND: Multi-Task Incremental Network Distillation (Jacopo Bonato et al., 2023)

{{<citation>}}

Jacopo Bonato, Francesco Pelosin, Luigi Sabetta, Alessandro Nicolosi. (2023)  
**MIND: Multi-Task Incremental Network Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Network Distillation  
[Paper Link](http://arxiv.org/abs/2312.02916v1)  

---


**ABSTRACT**  
The recent surge in pervasive devices generating dynamic data streams has underscored the necessity for learning systems to adapt to data distributional shifts continually. To tackle this challenge, the research community has put forth a spectrum of methodologies, including the demanding pursuit of class-incremental learning without replay data. In this study, we present MIND, a parameter isolation method that aims to significantly enhance the performance of replay-free solutions and achieve state-of-the-art results on several widely studied datasets. Our approach introduces two main contributions: two alternative distillation procedures that significantly improve the efficiency of MIND increasing the accumulated knowledge of each sub-network, and the optimization of the BachNorm layers across tasks inside the sub-networks. Overall, MIND outperforms all the state-of-the-art methods for rehearsal-free Class-Incremental learning (with an increment in classification accuracy of approx. +6% on CIFAR-100/10 and +10% on TinyImageNet/10) reaching up to approx. +40% accuracy in Domain-Incremental scenarios. Moreover, we ablated each contribution to demonstrate its impact on performance improvement. Our results showcase the superior performance of MIND indicating its potential for addressing the challenges posed by Class-incremental and Domain-Incremental learning in resource-constrained environments.

{{</citation>}}


### (75/126) Realistic Scatterer Based Adversarial Attacks on SAR Image Classifiers (Tian Ye et al., 2023)

{{<citation>}}

Tian Ye, Rajgopal Kannan, Viktor Prasanna, Carl Busart, Lance Kaplan. (2023)  
**Realistic Scatterer Based Adversarial Attacks on SAR Image Classifiers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2312.02912v1)  

---


**ABSTRACT**  
Adversarial attacks have highlighted the vulnerability of classifiers based on machine learning for Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) tasks. An adversarial attack perturbs SAR images of on-ground targets such that the classifiers are misled into making incorrect predictions. However, many existing attacking techniques rely on arbitrary manipulation of SAR images while overlooking the feasibility of executing the attacks on real-world SAR imagery. Instead, adversarial attacks should be able to be implemented by physical actions, for example, placing additional false objects as scatterers around the on-ground target to perturb the SAR image and fool the SAR ATR.   In this paper, we propose the On-Target Scatterer Attack (OTSA), a scatterer-based physical adversarial attack. To ensure the feasibility of its physical execution, we enforce a constraint on the positioning of the scatterers. Specifically, we restrict the scatterers to be placed only on the target instead of in the shadow regions or the background. To achieve this, we introduce a positioning score based on Gaussian kernels and formulate an optimization problem for our OTSA attack. Using a gradient ascent method to solve the optimization problem, the OTSA can generate a vector of parameters describing the positions, shapes, sizes and amplitudes of the scatterers to guide the physical execution of the attack that will mislead SAR image classifiers. The experimental results show that our attack obtains significantly higher success rates under the positioning constraint compared with the existing method.

{{</citation>}}


### (76/126) Diversified in-domain synthesis with efficient fine-tuning for few-shot classification (Victor G. Turrisi da Costa et al., 2023)

{{<citation>}}

Victor G. Turrisi da Costa, Nicola Dall'Asen, Yiming Wang, Nicu Sebe, Elisa Ricci. (2023)  
**Diversified in-domain synthesis with efficient fine-tuning for few-shot classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.03046v1)  

---


**ABSTRACT**  
Few-shot image classification aims to learn an image classifier using only a small set of labeled examples per class. A recent research direction for improving few-shot classifiers involves augmenting the labelled samples with synthetic images created by state-of-the-art text-to-image generation models. Following this trend, we propose Diversified in-domain synthesis with efficient fine-tuning (DISEF), a novel approach which addresses the generalization challenge in few-shot learning using synthetic data. DISEF consists of two main components. First, we propose a novel text-to-image augmentation pipeline that, by leveraging the real samples and their rich semantics coming from an advanced captioning model, promotes in-domain sample diversity for better generalization. Second, we emphasize the importance of effective model fine-tuning in few-shot recognition, proposing to use Low-Rank Adaptation (LoRA) for joint adaptation of the text and image encoders in a Vision Language Model. We validate our method in ten different benchmarks, consistently outperforming baselines and establishing a new state-of-the-art for few-shot classification. Code is available at \url{https://github.com/vturrisi/disef}

{{</citation>}}


### (77/126) BenchLMM: Benchmarking Cross-style Visual Capability of Large Multimodal Models (Rizhao Cai et al., 2023)

{{<citation>}}

Rizhao Cai, Zirui Song, Dayan Guan, Zhenhao Chen, Xing Luo, Chenyu Yi, Alex Kot. (2023)  
**BenchLMM: Benchmarking Cross-style Visual Capability of Large Multimodal Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.02896v2)  

---


**ABSTRACT**  
Large Multimodal Models (LMMs) such as GPT-4V and LLaVA have shown remarkable capabilities in visual reasoning with common image styles. However, their robustness against diverse style shifts, crucial for practical applications, remains largely unexplored. In this paper, we propose a new benchmark, BenchLMM, to assess the robustness of LMMs against three different styles: artistic image style, imaging sensor style, and application style, where each style has five sub-styles. Utilizing BenchLMM, we comprehensively evaluate state-of-the-art LMMs and reveal: 1) LMMs generally suffer performance degradation when working with other styles; 2) An LMM performs better than another model in common style does not guarantee its superior performance in other styles; 3) LMMs' reasoning capability can be enhanced by prompting LMMs to predict the style first, based on which we propose a versatile and training-free method for improving LMMs; 4) An intelligent LMM is expected to interpret the causes of its errors when facing stylistic variations. We hope that our benchmark and analysis can shed new light on developing more intelligent and versatile LMMs.

{{</citation>}}


### (78/126) Are Vision Transformers More Data Hungry Than Newborn Visual Systems? (Lalit Pandey et al., 2023)

{{<citation>}}

Lalit Pandey, Samantha M. W. Wood, Justin N. Wood. (2023)  
**Are Vision Transformers More Data Hungry Than Newborn Visual Systems?**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-NE, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.02843v1)  

---


**ABSTRACT**  
Vision transformers (ViTs) are top performing models on many computer vision benchmarks and can accurately predict human behavior on object recognition tasks. However, researchers question the value of using ViTs as models of biological learning because ViTs are thought to be more data hungry than brains, with ViTs requiring more training data to reach similar levels of performance. To test this assumption, we directly compared the learning abilities of ViTs and animals, by performing parallel controlled rearing experiments on ViTs and newborn chicks. We first raised chicks in impoverished visual environments containing a single object, then simulated the training data available in those environments by building virtual animal chambers in a video game engine. We recorded the first-person images acquired by agents moving through the virtual chambers and used those images to train self supervised ViTs that leverage time as a teaching signal, akin to biological visual systems. When ViTs were trained through the eyes of newborn chicks, the ViTs solved the same view invariant object recognition tasks as the chicks. Thus, ViTs were not more data hungry than newborn visual systems: both learned view invariant object representations in impoverished visual environments. The flexible and generic attention based learning mechanism in ViTs combined with the embodied data streams available to newborn animals appears sufficient to drive the development of animal-like object recognition.

{{</citation>}}


### (79/126) RotaTR: Detection Transformer for Dense and Rotated Object (Zhu Yuke et al., 2023)

{{<citation>}}

Zhu Yuke, Ruan Yumeng, Yang Lei, Guo Sheng. (2023)  
**RotaTR: Detection Transformer for Dense and Rotated Object**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.02821v1)  

---


**ABSTRACT**  
Detecting the objects in dense and rotated scenes is a challenging task. Recent works on this topic are mostly based on Faster RCNN or Retinanet. As they are highly dependent on the pre-set dense anchors and the NMS operation, the approach is indirect and suboptimal.The end-to-end DETR-based detectors have achieved great success in horizontal object detection and many other areas like segmentation, tracking, action recognition and etc.However, the DETR-based detectors perform poorly on dense rotated target tasks and perform worse than most modern CNN-based detectors. In this paper, we find the most significant reason for the poor performance is that the original attention can not accurately focus on the oriented targets. Accordingly, we propose Rotated object detection TRansformer (RotaTR) as an extension of DETR to oriented detection. Specifically, we design Rotation Sensitive deformable (RSDeform) attention to enhance the DETR's ability to detect oriented targets. It is used to build the feature alignment module and rotation-sensitive decoder for our model. We test RotaTR on four challenging-oriented benchmarks. It shows a great advantage in detecting dense and oriented objects compared to the original DETR. It also achieves competitive results when compared to the state-of-the-art.

{{</citation>}}


### (80/126) Generating Fine-Grained Human Motions Using ChatGPT-Refined Descriptions (Xu Shi et al., 2023)

{{<citation>}}

Xu Shi, Chuanchen Luo, Junran Peng, Hongwen Zhang, Yunlian Sun. (2023)  
**Generating Fine-Grained Human Motions Using ChatGPT-Refined Descriptions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ChatGPT, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2312.02772v1)  

---


**ABSTRACT**  
Recently, significant progress has been made in text-based motion generation, enabling the generation of diverse and high-quality human motions that conform to textual descriptions. However, it remains challenging to generate fine-grained or stylized motions due to the lack of datasets annotated with detailed textual descriptions. By adopting a divide-and-conquer strategy, we propose a new framework named Fine-Grained Human Motion Diffusion Model (FG-MDM) for human motion generation. Specifically, we first parse previous vague textual annotation into fine-grained description of different body parts by leveraging a large language model (GPT-3.5). We then use these fine-grained descriptions to guide a transformer-based diffusion model. FG-MDM can generate fine-grained and stylized motions even outside of the distribution of the training data. Our experimental results demonstrate the superiority of FG-MDM over previous methods, especially the strong generalization capability. We will release our fine-grained textual annotations for HumanML3D and KIT.

{{</citation>}}


### (81/126) SEVA: Leveraging sketches to evaluate alignment between human and machine visual abstraction (Kushin Mukherjee et al., 2023)

{{<citation>}}

Kushin Mukherjee, Holly Huey, Xuanchen Lu, Yael Vinker, Rio Aguina-Kang, Ariel Shamir, Judith E. Fan. (2023)  
**SEVA: Leveraging sketches to evaluate alignment between human and machine visual abstraction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2312.03035v1)  

---


**ABSTRACT**  
Sketching is a powerful tool for creating abstract images that are sparse but meaningful. Sketch understanding poses fundamental challenges for general-purpose vision algorithms because it requires robustness to the sparsity of sketches relative to natural visual inputs and because it demands tolerance for semantic ambiguity, as sketches can reliably evoke multiple meanings. While current vision algorithms have achieved high performance on a variety of visual tasks, it remains unclear to what extent they understand sketches in a human-like way. Here we introduce SEVA, a new benchmark dataset containing approximately 90K human-generated sketches of 128 object concepts produced under different time constraints, and thus systematically varying in sparsity. We evaluated a suite of state-of-the-art vision algorithms on their ability to correctly identify the target concept depicted in these sketches and to generate responses that are strongly aligned with human response patterns on the same sketch recognition task. We found that vision algorithms that better predicted human sketch recognition performance also better approximated human uncertainty about sketch meaning, but there remains a sizable gap between model and human response patterns. To explore the potential of models that emulate human visual abstraction in generative tasks, we conducted further evaluations of a recently developed sketch generation algorithm (Vinker et al., 2022) capable of generating sketches that vary in sparsity. We hope that public release of this dataset and evaluation protocol will catalyze progress towards algorithms with enhanced capacities for human-like visual abstraction.

{{</citation>}}


### (82/126) C-NERF: Representing Scene Changes as Directional Consistency Difference-based NeRF (Rui Huang et al., 2023)

{{<citation>}}

Rui Huang, Binbin Jiang, Qingyi Zhao, William Wang, Yuxiang Zhang, Qing Guo. (2023)  
**C-NERF: Representing Scene Changes as Directional Consistency Difference-based NeRF**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2312.02751v1)  

---


**ABSTRACT**  
In this work, we aim to detect the changes caused by object variations in a scene represented by the neural radiance fields (NeRFs). Given an arbitrary view and two sets of scene images captured at different timestamps, we can predict the scene changes in that view, which has significant potential applications in scene monitoring and measuring. We conducted preliminary studies and found that such an exciting task cannot be easily achieved by utilizing existing NeRFs and 2D change detection methods with many false or missing detections. The main reason is that the 2D change detection is based on the pixel appearance difference between spatial-aligned image pairs and neglects the stereo information in the NeRF. To address the limitations, we propose the C-NERF to represent scene changes as directional consistency difference-based NeRF, which mainly contains three modules. We first perform the spatial alignment of two NeRFs captured before and after changes. Then, we identify the change points based on the direction-consistent constraint; that is, real change points have similar change representations across view directions, but fake change points do not. Finally, we design the change map rendering process based on the built NeRFs and can generate the change map of an arbitrarily specified view direction. To validate the effectiveness, we build a new dataset containing ten scenes covering diverse scenarios with different changing objects. Our approach surpasses state-of-the-art 2D change detection and NeRF-based methods by a significant margin.

{{</citation>}}


### (83/126) R3D-SWIN:Use Shifted Window Attention for Single-View 3D Reconstruction (Chenhuan Li et al., 2023)

{{<citation>}}

Chenhuan Li, Meihua Xiao, zehuan li, Mengxi Gao. (2023)  
**R3D-SWIN:Use Shifted Window Attention for Single-View 3D Reconstruction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.02725v1)  

---


**ABSTRACT**  
Recently, vision transformers have performed well in various computer vision tasks, including voxel 3D reconstruction. However, the windows of the vision transformer are not multi-scale, and there is no connection between the windows, which limits the accuracy of voxel 3D reconstruction . Therefore, we propose a shifted windows attention voxel 3D reconstruction network. To the best of our knowledge, this is the first work to apply shifted window attention to voxel 3D reconstruction. Experimental results on ShapeNet verify our method achieves SOTA accuracy in single-view reconstruction.

{{</citation>}}


### (84/126) Enhancing Vehicle Entrance and Parking Management: Deep Learning Solutions for Efficiency and Security (Muhammad Umer Ramzan et al., 2023)

{{<citation>}}

Muhammad Umer Ramzan, Usman Ali, Syed Haider Abbas Naqvi, Zeeshan Aslam, Tehseen, Husnain Ali, Muhammad Faheem. (2023)  
**Enhancing Vehicle Entrance and Parking Management: Deep Learning Solutions for Efficiency and Security**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Google, OCR, Security  
[Paper Link](http://arxiv.org/abs/2312.02699v1)  

---


**ABSTRACT**  
The auto-management of vehicle entrance and parking in any organization is a complex challenge encompassing record-keeping, efficiency, and security concerns. Manual methods for tracking vehicles and finding parking spaces are slow and a waste of time. To solve the problem of auto management of vehicle entrance and parking, we have utilized state-of-the-art deep learning models and automated the process of vehicle entrance and parking into any organization. To ensure security, our system integrated vehicle detection, license number plate verification, and face detection and recognition models to ensure that the person and vehicle are registered with the organization. We have trained multiple deep-learning models for vehicle detection, license number plate detection, face detection, and recognition, however, the YOLOv8n model outperformed all the other models. Furthermore, License plate recognition is facilitated by Google's Tesseract-OCR Engine. By integrating these technologies, the system offers efficient vehicle detection, precise identification, streamlined record keeping, and optimized parking slot allocation in buildings, thereby enhancing convenience, accuracy, and security. Future research opportunities lie in fine-tuning system performance for a wide range of real-world applications.

{{</citation>}}


### (85/126) Analyzing and Improving the Training Dynamics of Diffusion Models (Tero Karras et al., 2023)

{{<citation>}}

Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, Samuli Laine. (2023)  
**Analyzing and Improving the Training Dynamics of Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-NE, cs.CV, stat-ML  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.02696v1)  

---


**ABSTRACT**  
Diffusion models currently dominate the field of data-driven image synthesis with their unparalleled scaling to large datasets. In this paper, we identify and rectify several causes for uneven and ineffective training in the popular ADM diffusion model architecture, without altering its high-level structure. Observing uncontrolled magnitude changes and imbalances in both the network activations and weights over the course of training, we redesign the network layers to preserve activation, weight, and update magnitudes on expectation. We find that systematic application of this philosophy eliminates the observed drifts and imbalances, resulting in considerably better networks at equal computational complexity. Our modifications improve the previous record FID of 2.41 in ImageNet-512 synthesis to 1.81, achieved using fast deterministic sampling.   As an independent contribution, we present a method for setting the exponential moving average (EMA) parameters post-hoc, i.e., after completing the training run. This allows precise tuning of EMA length without the cost of performing several training runs, and reveals its surprising interactions with network architecture, training time, and guidance.

{{</citation>}}


### (86/126) UPOCR: Towards Unified Pixel-Level OCR Interface (Dezhi Peng et al., 2023)

{{<citation>}}

Dezhi Peng, Zhenhua Yang, Jiaxin Zhang, Chongyu Liu, Yongxin Shi, Kai Ding, Fengjun Guo, Lianwen Jin. (2023)  
**UPOCR: Towards Unified Pixel-Level OCR Interface**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR, Transformer  
[Paper Link](http://arxiv.org/abs/2312.02694v1)  

---


**ABSTRACT**  
In recent years, the optical character recognition (OCR) field has been proliferating with plentiful cutting-edge approaches for a wide spectrum of tasks. However, these approaches are task-specifically designed with divergent paradigms, architectures, and training strategies, which significantly increases the complexity of research and maintenance and hinders the fast deployment in applications. To this end, we propose UPOCR, a simple-yet-effective generalist model for Unified Pixel-level OCR interface. Specifically, the UPOCR unifies the paradigm of diverse OCR tasks as image-to-image transformation and the architecture as a vision Transformer (ViT)-based encoder-decoder. Learnable task prompts are introduced to push the general feature representations extracted by the encoder toward task-specific spaces, endowing the decoder with task awareness. Moreover, the model training is uniformly aimed at minimizing the discrepancy between the generated and ground-truth images regardless of the inhomogeneity among tasks. Experiments are conducted on three pixel-level OCR tasks including text removal, text segmentation, and tampered text detection. Without bells and whistles, the experimental results showcase that the proposed method can simultaneously achieve state-of-the-art performance on three tasks with a unified single model, which provides valuable strategies and insights for future research on generalist OCR models. Code will be publicly available.

{{</citation>}}


### (87/126) Zero-Shot Point Cloud Registration (Weijie Wang et al., 2023)

{{<citation>}}

Weijie Wang, Guofeng Mei, Bin Ren, Xiaoshui Huang, Fabio Poiesi, Luc Van Gool, Nicu Sebe, Bruno Lepri. (2023)  
**Zero-Shot Point Cloud Registration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.03032v1)  

---


**ABSTRACT**  
Learning-based point cloud registration approaches have significantly outperformed their traditional counterparts. However, they typically require extensive training on specific datasets. In this paper, we propose , the first zero-shot point cloud registration approach that eliminates the need for training on point cloud datasets. The cornerstone of ZeroReg is the novel transfer of image features from keypoints to the point cloud, enriched by aggregating information from 3D geometric neighborhoods. Specifically, we extract keypoints and features from 2D image pairs using a frozen pretrained 2D backbone. These features are then projected in 3D, and patches are constructed by searching for neighboring points. We integrate the geometric and visual features of each point using our novel parameter-free geometric decoder. Subsequently, the task of determining correspondences between point clouds is formulated as an optimal transport problem. Extensive evaluations of ZeroReg demonstrate its competitive performance against both traditional and learning-based methods. On benchmarks such as 3DMatch, 3DLoMatch, and ScanNet, ZeroReg achieves impressive Recall Ratios (RR) of over 84%, 46%, and 75%, respectively.

{{</citation>}}


### (88/126) Generating Visually Realistic Adversarial Patch (Xiaosen Wang et al., 2023)

{{<citation>}}

Xiaosen Wang, Kunyu Wang. (2023)  
**Generating Visually Realistic Adversarial Patch**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.03030v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) are vulnerable to various types of adversarial examples, bringing huge threats to security-critical applications. Among these, adversarial patches have drawn increasing attention due to their good applicability to fool DNNs in the physical world. However, existing works often generate patches with meaningless noise or patterns, making it conspicuous to humans. To address this issue, we explore how to generate visually realistic adversarial patches to fool DNNs. Firstly, we analyze that a high-quality adversarial patch should be realistic, position irrelevant, and printable to be deployed in the physical world. Based on this analysis, we propose an effective attack called VRAP, to generate visually realistic adversarial patches. Specifically, VRAP constrains the patch in the neighborhood of a real image to ensure the visual reality, optimizes the patch at the poorest position for position irrelevance, and adopts Total Variance loss as well as gamma transformation to make the generated patch printable without losing information. Empirical evaluations on the ImageNet dataset demonstrate that the proposed VRAP exhibits outstanding attack performance in the digital world. Moreover, the generated adversarial patches can be disguised as the scrawl or logo in the physical world to fool the deep models without being detected, bringing significant threats to DNNs-enabled applications.

{{</citation>}}


### (89/126) TPA3D: Triplane Attention for Fast Text-to-3D Generation (Hong-En Chen et al., 2023)

{{<citation>}}

Hong-En Chen, Bin-Shih Wu, Sheng-Yu Huang, Yu-Chiang Frank Wang. (2023)  
**TPA3D: Triplane Attention for Fast Text-to-3D Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.02647v1)  

---


**ABSTRACT**  
Due to the lack of large-scale text-3D correspondence data, recent text-to-3D generation works mainly rely on utilizing 2D diffusion models for synthesizing 3D data. Since diffusion-based methods typically require significant optimization time for both training and inference, the use of GAN-based models would still be desirable for fast 3D generation. In this work, we propose Triplane Attention for text-guided 3D generation (TPA3D), an end-to-end trainable GAN-based deep learning model for fast text-to-3D generation. With only 3D shape data and their rendered 2D images observed during training, our TPA3D is designed to retrieve detailed visual descriptions for synthesizing the corresponding 3D mesh data. This is achieved by the proposed attention mechanisms on the extracted sentence and word-level text features. In our experiments, we show that TPA3D generates high-quality 3D textured shapes aligned with fine-grained descriptions, while impressive computation efficiency can be observed.

{{</citation>}}


### (90/126) Stable Diffusion Exposed: Gender Bias from Prompt to Image (Yankun Wu et al., 2023)

{{<citation>}}

Yankun Wu, Yuta Nakashima, Noa Garcia. (2023)  
**Stable Diffusion Exposed: Gender Bias from Prompt to Image**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.03027v1)  

---


**ABSTRACT**  
Recent studies have highlighted biases in generative models, shedding light on their predisposition towards gender-based stereotypes and imbalances. This paper contributes to this growing body of research by introducing an evaluation protocol designed to automatically analyze the impact of gender indicators on Stable Diffusion images. Leveraging insights from prior work, we explore how gender indicators not only affect gender presentation but also the representation of objects and layouts within the generated images. Our findings include the existence of differences in the depiction of objects, such as instruments tailored for specific genders, and shifts in overall layouts. We also reveal that neutral prompts tend to produce images more aligned with masculine prompts than their feminine counterparts, providing valuable insights into the nuanced gender biases inherent in Stable Diffusion.

{{</citation>}}


### (91/126) Facilitating the Production of Well-tailored Video Summaries for Sharing on Social Media (Evlampios Apostolidis et al., 2023)

{{<citation>}}

Evlampios Apostolidis, Konstantinos Apostolidis, Vasileios Mezaris. (2023)  
**Facilitating the Production of Well-tailored Video Summaries for Sharing on Social Media**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Social Media  
[Paper Link](http://arxiv.org/abs/2312.02616v1)  

---


**ABSTRACT**  
This paper presents a web-based tool that facilitates the production of tailored summaries for online sharing on social media. Through an interactive user interface, it supports a ``one-click'' video summarization process. Based on the integrated AI models for video summarization and aspect ratio transformation, it facilitates the generation of multiple summaries of a full-length video according to the needs of target platforms with regard to the video's length and aspect ratio.

{{</citation>}}


### (92/126) An Integrated System for Spatio-Temporal Summarization of 360-degrees Videos (Ioannis Kontostathis et al., 2023)

{{<citation>}}

Ioannis Kontostathis, Evlampios Apostolidis, Vasileios Mezaris. (2023)  
**An Integrated System for Spatio-Temporal Summarization of 360-degrees Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2312.02576v1)  

---


**ABSTRACT**  
In this work, we present an integrated system for spatiotemporal summarization of 360-degrees videos. The video summary production mainly involves the detection of salient events and their synopsis into a concise summary. The analysis relies on state-of-the-art methods for saliency detection in 360-degrees video (ATSal and SST-Sal) and video summarization (CA-SUM). It also contains a mechanism that classifies a 360-degrees video based on the use of static or moving camera during recording and decides which saliency detection method will be used, as well as a 2D video production component that is responsible to create a conventional 2D video containing the salient events in the 360-degrees video. Quantitative evaluations using two datasets for 360-degrees video saliency detection (VR-EyeTracking, Sports-360) show the accuracy and positive impact of the developed decision mechanism, and justify our choice to use two different methods for detecting the salient events. A qualitative analysis using content from these datasets, gives further insights about the functionality of the decision mechanism, shows the pros and cons of each used saliency detection method and demonstrates the advanced performance of the trained summarization method against a more conventional approach.

{{</citation>}}


### (93/126) Think Twice Before Selection: Federated Evidential Active Learning for Medical Image Analysis with Domain Shifts (Jiayi Chen et al., 2023)

{{<citation>}}

Jiayi Chen, Benteng Ma, Hengfei Cui, Yong Xia, Kwang-Ting Cheng. (2023)  
**Think Twice Before Selection: Federated Evidential Active Learning for Medical Image Analysis with Domain Shifts**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2312.02567v1)  

---


**ABSTRACT**  
Federated learning facilitates the collaborative learning of a global model across multiple distributed medical institutions without centralizing data. Nevertheless, the expensive cost of annotation on local clients remains an obstacle to effectively utilizing local data. To mitigate this issue, federated active learning methods suggest leveraging local and global model predictions to select a relatively small amount of informative local data for annotation. However, existing methods mainly focus on all local data sampled from the same domain, making them unreliable in realistic medical scenarios with domain shifts among different clients. In this paper, we make the first attempt to assess the informativeness of local data derived from diverse domains and propose a novel methodology termed Federated Evidential Active Learning (FEAL) to calibrate the data evaluation under domain shift. Specifically, we introduce a Dirichlet prior distribution in both local and global models to treat the prediction as a distribution over the probability simplex and capture both aleatoric and epistemic uncertainties by using the Dirichlet-based evidential model. Then we employ the epistemic uncertainty to calibrate the aleatoric uncertainty. Afterward, we design a diversity relaxation strategy to reduce data redundancy and maintain data diversity. Extensive experiments and analyses are conducted to show the superiority of FEAL over the state-of-the-art active learning methods and the efficiency of FEAL under the federated active learning framework.

{{</citation>}}


### (94/126) DemaFormer: Damped Exponential Moving Average Transformer with Energy-Based Modeling for Temporal Language Grounding (Thong Nguyen et al., 2023)

{{<citation>}}

Thong Nguyen, Xiaobao Wu, Xinshuai Dong, Cong-Duy Nguyen, See-Kiong Ng, Luu Anh Tuan. (2023)  
**DemaFormer: Damped Exponential Moving Average Transformer with Energy-Based Modeling for Temporal Language Grounding**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.02549v1)  

---


**ABSTRACT**  
Temporal Language Grounding seeks to localize video moments that semantically correspond to a natural language query. Recent advances employ the attention mechanism to learn the relations between video moments and the text query. However, naive attention might not be able to appropriately capture such relations, resulting in ineffective distributions where target video moments are difficult to separate from the remaining ones. To resolve the issue, we propose an energy-based model framework to explicitly learn moment-query distributions. Moreover, we propose DemaFormer, a novel Transformer-based architecture that utilizes exponential moving average with a learnable damping factor to effectively encode moment-query inputs. Comprehensive experiments on four public temporal language grounding datasets showcase the superiority of our methods over the state-of-the-art baselines.

{{</citation>}}


### (95/126) GeNIe: Generative Hard Negative Images Through Diffusion (Soroush Abbasi Koohpayegani et al., 2023)

{{<citation>}}

Soroush Abbasi Koohpayegani, Anuj Singh, K L Navaneet, Hadi Jamali-Rad, Hamed Pirsiavash. (2023)  
**GeNIe: Generative Hard Negative Images Through Diffusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.02548v1)  

---


**ABSTRACT**  
Data augmentation is crucial in training deep models, preventing them from overfitting to limited data. Common data augmentation methods are effective, but recent advancements in generative AI, such as diffusion models for image generation, enable more sophisticated augmentation techniques that produce data resembling natural images. We recognize that augmented samples closer to the ideal decision boundary of a classifier are particularly effective and efficient in guiding the learning process. We introduce GeNIe which leverages a diffusion model conditioned on a text prompt to merge contrasting data points (an image from the source category and a text prompt from the target category) to generate challenging samples for the target category. Inspired by recent image editing methods, we limit the number of diffusion iterations and the amount of noise. This ensures that the generated image retains low-level and contextual features from the source image, potentially conflicting with the target category. Our extensive experiments, in few-shot and also long-tail distribution settings, demonstrate the effectiveness of our novel augmentation method, especially benefiting categories with a limited number of examples.

{{</citation>}}


### (96/126) Machine Vision Therapy: Multimodal Large Language Models Can Enhance Visual Robustness via Denoising In-Context Learning (Zhuo Huang et al., 2023)

{{<citation>}}

Zhuo Huang, Chang Liu, Yinpeng Dong, Hang Su, Shibao Zheng, Tongliang Liu. (2023)  
**Machine Vision Therapy: Multimodal Large Language Models Can Enhance Visual Robustness via Denoising In-Context Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Language Model  
[Paper Link](http://arxiv.org/abs/2312.02546v1)  

---


**ABSTRACT**  
Although vision models such as Contrastive Language-Image Pre-Training (CLIP) show impressive generalization performance, their zero-shot robustness is still limited under Out-of-Distribution (OOD) scenarios without fine-tuning. Instead of undesirably providing human supervision as commonly done, it is possible to take advantage of Multi-modal Large Language Models (MLLMs) that hold powerful visual understanding abilities. However, MLLMs are shown to struggle with vision problems due to the incompatibility of tasks, thus hindering their utilization. In this paper, we propose to effectively leverage MLLMs to conduct Machine Vision Therapy which aims to rectify the noisy predictions from vision models. By fine-tuning with the denoised labels, the learning model performance can be boosted in an unsupervised manner. To solve the incompatibility issue, we propose a novel Denoising In-Context Learning (DICL) strategy to align vision tasks with MLLMs. Concretely, by estimating a transition matrix that captures the probability of one class being confused with another, an instruction containing a correct exemplar and an erroneous one from the most probable noisy class can be constructed. Such an instruction can help any MLLMs with ICL ability to detect and rectify incorrect predictions of vision models. Through extensive experiments on ImageNet, WILDS, DomainBed, and other OOD datasets, we carefully validate the quantitative and qualitative effectiveness of our method. Our code is available at https://github.com/tmllab/Machine_Vision_Therapy.

{{</citation>}}


### (97/126) Graph Information Bottleneck for Remote Sensing Segmentation (Yuntao Shou et al., 2023)

{{<citation>}}

Yuntao Shou, Wei Ai, Tao Meng. (2023)  
**Graph Information Bottleneck for Remote Sensing Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: GNN, Transformer  
[Paper Link](http://arxiv.org/abs/2312.02545v1)  

---


**ABSTRACT**  
Remote sensing segmentation has a wide range of applications in environmental protection, and urban change detection, etc. Despite the success of deep learning-based remote sensing segmentation methods (e.g., CNN and Transformer), they are not flexible enough to model irregular objects. In addition, existing graph contrastive learning methods usually adopt the way of maximizing mutual information to keep the node representations consistent between different graph views, which may cause the model to learn task-independent redundant information. To tackle the above problems, this paper treats images as graph structures and introduces a simple contrastive vision GNN (SC-ViG) architecture for remote sensing segmentation. Specifically, we construct a node-masked and edge-masked graph view to obtain an optimal graph structure representation, which can adaptively learn whether to mask nodes and edges. Furthermore, this paper innovatively introduces information bottleneck theory into graph contrastive learning to maximize task-related information while minimizing task-independent redundant information. Finally, we replace the convolutional module in UNet with the SC-ViG module to complete the segmentation and classification tasks of remote sensing images. Extensive experiments on publicly available real datasets demonstrate that our method outperforms state-of-the-art remote sensing image segmentation methods.

{{</citation>}}


### (98/126) EtC: Temporal Boundary Expand then Clarify for Weakly Supervised Video Grounding with Multimodal Large Language Model (Guozhang Li et al., 2023)

{{<citation>}}

Guozhang Li, Xinpeng Ding, De Cheng, Jie Li, Nannan Wang, Xinbo Gao. (2023)  
**EtC: Temporal Boundary Expand then Clarify for Weakly Supervised Video Grounding with Multimodal Large Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.02483v1)  

---


**ABSTRACT**  
Early weakly supervised video grounding (WSVG) methods often struggle with incomplete boundary detection due to the absence of temporal boundary annotations. To bridge the gap between video-level and boundary-level annotation, explicit-supervision methods, i.e., generating pseudo-temporal boundaries for training, have achieved great success. However, data augmentations in these methods might disrupt critical temporal information, yielding poor pseudo boundaries. In this paper, we propose a new perspective that maintains the integrity of the original temporal content while introducing more valuable information for expanding the incomplete boundaries. To this end, we propose EtC (Expand then Clarify), first use the additional information to expand the initial incomplete pseudo boundaries, and subsequently refine these expanded ones to achieve precise boundaries. Motivated by video continuity, i.e., visual similarity across adjacent frames, we use powerful multimodal large language models (MLLMs) to annotate each frame within initial pseudo boundaries, yielding more comprehensive descriptions for expanded boundaries. To further clarify the noise of expanded boundaries, we combine mutual learning with a tailored proposal-level contrastive objective to use a learnable approach to harmonize a balance between incomplete yet clean (initial) and comprehensive yet noisy (expanded) boundaries for more precise ones. Experiments demonstrate the superiority of our method on two challenging WSVG datasets.

{{</citation>}}


### (99/126) SAM-Assisted Remote Sensing Imagery Semantic Segmentation with Object and Boundary Constraints (Xianping Ma et al., 2023)

{{<citation>}}

Xianping Ma, Qianqian Wu, Xingyu Zhao, Xiaokang Zhang, Man-On Pun, Bo Huang. (2023)  
**SAM-Assisted Remote Sensing Imagery Semantic Segmentation with Object and Boundary Constraints**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.02464v1)  

---


**ABSTRACT**  
Semantic segmentation of remote sensing imagery plays a pivotal role in extracting precise information for diverse down-stream applications. Recent development of the Segment Anything Model (SAM), an advanced general-purpose segmentation model, has revolutionized this field, presenting new avenues for accurate and efficient segmentation. However, SAM is limited to generating segmentation results without class information. Consequently, the utilization of such a powerful general vision model for semantic segmentation in remote sensing images has become a focal point of research. In this paper, we present a streamlined framework aimed at leveraging the raw output of SAM by exploiting two novel concepts called SAM-Generated Object (SGO) and SAM-Generated Boundary (SGB). More specifically, we propose a novel object loss and further introduce a boundary loss as augmentative components to aid in model optimization in a general semantic segmentation framework. Taking into account the content characteristics of SGO, we introduce the concept of object consistency to leverage segmented regions lacking semantic information. By imposing constraints on the consistency of predicted values within objects, the object loss aims to enhance semantic segmentation performance. Furthermore, the boundary loss capitalizes on the distinctive features of SGB by directing the model's attention to the boundary information of the object. Experimental results on two well-known datasets, namely ISPRS Vaihingen and LoveDA Urban, demonstrate the effectiveness of our proposed method. The source code for this work will be accessible at https://github.com/sstary/SSRS.

{{</citation>}}


### (100/126) FINER: Flexible spectral-bias tuning in Implicit NEural Representation by Variable-periodic Activation Functions (Zhen Liu et al., 2023)

{{<citation>}}

Zhen Liu, Hao Zhu, Qi Zhang, Jingde Fu, Weibing Deng, Zhan Ma, Yanwen Guo, Xun Cao. (2023)  
**FINER: Flexible spectral-bias tuning in Implicit NEural Representation by Variable-periodic Activation Functions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2312.02434v1)  

---


**ABSTRACT**  
Implicit Neural Representation (INR), which utilizes a neural network to map coordinate inputs to corresponding attributes, is causing a revolution in the field of signal processing. However, current INR techniques suffer from a restricted capability to tune their supported frequency set, resulting in imperfect performance when representing complex signals with multiple frequencies. We have identified that this frequency-related problem can be greatly alleviated by introducing variable-periodic activation functions, for which we propose FINER. By initializing the bias of the neural network within different ranges, sub-functions with various frequencies in the variable-periodic function are selected for activation. Consequently, the supported frequency set of FINER can be flexibly tuned, leading to improved performance in signal representation. We demonstrate the capabilities of FINER in the contexts of 2D image fitting, 3D signed distance field representation, and 5D neural radiance fields optimization, and we show that it outperforms existing INRs.

{{</citation>}}


### (101/126) Lenna: Language Enhanced Reasoning Detection Assistant (Fei Wei et al., 2023)

{{<citation>}}

Fei Wei, Xinyu Zhang, Ailing Zhang, Bo Zhang, Xiangxiang Chu. (2023)  
**Lenna: Language Enhanced Reasoning Detection Assistant**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.02433v1)  

---


**ABSTRACT**  
With the fast-paced development of multimodal large language models (MLLMs), we can now converse with AI systems in natural languages to understand images. However, the reasoning power and world knowledge embedded in the large language models have been much less investigated and exploited for image perception tasks. In this paper, we propose Lenna, a language-enhanced reasoning detection assistant, which utilizes the robust multimodal feature representation of MLLMs, while preserving location information for detection. This is achieved by incorporating an additional <DET> token in the MLLM vocabulary that is free of explicit semantic context but serves as a prompt for the detector to identify the corresponding position. To evaluate the reasoning capability of Lenna, we construct a ReasonDet dataset to measure its performance on reasoning-based detection. Remarkably, Lenna demonstrates outstanding performance on ReasonDet and comes with significantly low training costs. It also incurs minimal transferring overhead when extended to other tasks. Our code and model will be available at https://git.io/Lenna.

{{</citation>}}


### (102/126) MGTR: Multi-Granular Transformer for Motion Prediction with LiDAR (Yiqian Gan et al., 2023)

{{<citation>}}

Yiqian Gan, Hao Xiao, Yizhe Zhao, Ethan Zhang, Zhe Huang, Xin Ye, Lingting Ge. (2023)  
**MGTR: Multi-Granular Transformer for Motion Prediction with LiDAR**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.02409v1)  

---


**ABSTRACT**  
Motion prediction has been an essential component of autonomous driving systems since it handles highly uncertain and complex scenarios involving moving agents of different types. In this paper, we propose a Multi-Granular TRansformer (MGTR) framework, an encoder-decoder network that exploits context features in different granularities for different kinds of traffic agents. To further enhance MGTR's capabilities, we leverage LiDAR point cloud data by incorporating LiDAR semantic features from an off-the-shelf LiDAR feature extractor. We evaluate MGTR on Waymo Open Dataset motion prediction benchmark and show that the proposed method achieved state-of-the-art performance, ranking 1st on its leaderboard (https://waymo.com/open/challenges/2023/motion-prediction/).

{{</citation>}}


## cond-mat.mes-hall (1)



### (103/126) The Automated Bias Triangle Feature Extraction Framework (Madeleine Kotzagiannidis et al., 2023)

{{<citation>}}

Madeleine Kotzagiannidis, Jonas Schuff, Nathan Korda. (2023)  
**The Automated Bias Triangle Feature Extraction Framework**  

---
Primary Category: cond-mat.mes-hall  
Categories: cond-mat-mes-hall, cond-mat.mes-hall, cs-CV, quant-ph  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.03110v1)  

---


**ABSTRACT**  
Bias triangles represent features in stability diagrams of Quantum Dot (QD) devices, whose occurrence and property analysis are crucial indicators for spin physics. Nevertheless, challenges associated with quality and availability of data as well as the subtlety of physical phenomena of interest have hindered an automatic and bespoke analysis framework, often still relying (in part) on human labelling and verification. We introduce a feature extraction framework for bias triangles, built from unsupervised, segmentation-based computer vision methods, which facilitates the direct identification and quantification of physical properties of the former. Thereby, the need for human input or large training datasets to inform supervised learning approaches is circumvented, while additionally enabling the automation of pixelwise shape and feature labeling. In particular, we demonstrate that Pauli Spin Blockade (PSB) detection can be conducted effectively, efficiently and without any training data as a direct result of this approach.

{{</citation>}}


## stat.ML (3)



### (104/126) Detecting algorithmic bias in medical AI-models (Jeffrey Smith et al., 2023)

{{<citation>}}

Jeffrey Smith, Andre Holder, Rishikesan Kamaleswaran, Yao Xie. (2023)  
**Detecting algorithmic bias in medical AI-models**  

---
Primary Category: stat.ML  
Categories: cs-CY, cs-LG, stat-AP, stat-ML, stat.ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.02959v1)  

---


**ABSTRACT**  
With the growing prevalence of machine learning and artificial intelligence-based medical decision support systems, it is equally important to ensure that these systems provide patient outcomes in a fair and equitable fashion. This paper presents an innovative framework for detecting areas of algorithmic bias in medical-AI decision support systems. Our approach efficiently identifies potential biases in medical-AI models, specifically in the context of sepsis prediction, by employing the Classification and Regression Trees (CART) algorithm. We verify our methodology by conducting a series of synthetic data experiments, showcasing its ability to estimate areas of bias in controlled settings precisely. The effectiveness of the concept is further validated by experiments using electronic medical records from Grady Memorial Hospital in Atlanta, Georgia. These tests demonstrate the practical implementation of our strategy in a clinical environment, where it can function as a vital instrument for guaranteeing fairness and equity in AI-based medical decisions.

{{</citation>}}


### (105/126) A Kernel-Based Neural Network Test for High-dimensional Sequencing Data Analysis (Tingting Hou et al., 2023)

{{<citation>}}

Tingting Hou, Chang Jiang, Qing Lu. (2023)  
**A Kernel-Based Neural Network Test for High-dimensional Sequencing Data Analysis**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ME, stat-ML, stat.ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.02850v2)  

---


**ABSTRACT**  
The recent development of artificial intelligence (AI) technology, especially the advance of deep neural network (DNN) technology, has revolutionized many fields. While DNN plays a central role in modern AI technology, it has been rarely used in sequencing data analysis due to challenges brought by high-dimensional sequencing data (e.g., overfitting). Moreover, due to the complexity of neural networks and their unknown limiting distributions, building association tests on neural networks for genetic association analysis remains a great challenge. To address these challenges and fill the important gap of using AI in high-dimensional sequencing data analysis, we introduce a new kernel-based neural network (KNN) test for complex association analysis of sequencing data. The test is built on our previously developed KNN framework, which uses random effects to model the overall effects of high-dimensional genetic data and adopts kernel-based neural network structures to model complex genotype-phenotype relationships. Based on KNN, a Wald-type test is then introduced to evaluate the joint association of high-dimensional genetic data with a disease phenotype of interest, considering non-linear and non-additive effects (e.g., interaction effects). Through simulations, we demonstrated that our proposed method attained higher power compared to the sequence kernel association test (SKAT), especially in the presence of non-linear and interaction effects. Finally, we apply the methods to the whole genome sequencing (WGS) dataset from the Alzheimer's Disease Neuroimaging Initiative (ADNI) study, investigating new genes associated with the hippocampal volume change over time.

{{</citation>}}


### (106/126) Convergence Rates for Stochastic Approximation: Biased Noise with Unbounded Variance, and Applications (Rajeeva L. Karandikar et al., 2023)

{{<citation>}}

Rajeeva L. Karandikar, M. Vidyasagar. (2023)  
**Convergence Rates for Stochastic Approximation: Biased Noise with Unbounded Variance, and Applications**  

---
Primary Category: stat.ML  
Categories: 62L20, 60G17, 93D05, cs-LG, math-OC, math-PR, stat-ML, stat.ML  
Keywords: Bias, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.02828v1)  

---


**ABSTRACT**  
The Stochastic Approximation (SA) algorithm introduced by Robbins and Monro in 1951 has been a standard method for solving equations of the form $\mathbf{f}({\boldsymbol {\theta}}) = \mathbf{0}$, when only noisy measurements of $\mathbf{f}(\cdot)$ are available. If $\mathbf{f}({\boldsymbol {\theta}}) = \nabla J({\boldsymbol {\theta}})$ for some function $J(\cdot)$, then SA can also be used to find a stationary point of $J(\cdot)$. In much of the literature, it is assumed that the error term ${\boldsymbol {xi}}_{t+1}$ has zero conditional mean, and that its conditional variance is bounded as a function of $t$ (though not necessarily with respect to ${\boldsymbol {\theta}}_t$). Also, for the most part, the emphasis has been on ``synchronous'' SA, whereby, at each time $t$, \textit{every} component of ${\boldsymbol {\theta}}_t$ is updated. Over the years, SA has been applied to a variety of areas, out of which two are the focus in this paper: Convex and nonconvex optimization, and Reinforcement Learning (RL). As it turns out, in these applications, the above-mentioned assumptions do not always hold. In zero-order methods, the error neither has zero mean nor bounded conditional variance. In the present paper, we extend SA theory to encompass errors with nonzero conditional mean and/or unbounded conditional variance, and also asynchronous SA. In addition, we derive estimates for the rate of convergence of the algorithm. Then we apply the new results to problems in nonconvex optimization, and to Markovian SA, a recently emerging area in RL. We prove that SA converges in these situations, and compute the ``optimal step size sequences'' to maximize the estimated rate of convergence.

{{</citation>}}


## cs.SI (2)



### (107/126) Using the SP!CE Framework to Code Influence Campaign Activity on Social Media: Case Study on the 2022 Brazilian Presidential Election (Alexander Gocso et al., 2023)

{{<citation>}}

Alexander Gocso, Claudia Perez Brito, Bryan Ruesca, Allen Mendes, Mark A. Finlayson. (2023)  
**Using the SP!CE Framework to Code Influence Campaign Activity on Social Media: Case Study on the 2022 Brazilian Presidential Election**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-soc-ph  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2312.02810v2)  

---


**ABSTRACT**  
We describe a case study in the use of the Structured Process for Information Campaign Enhancement (SP!CE, version 2.1) to evaluate influence campaigns present in the 2nd round of the Brazilian presidential election in 2022 October. SP!CE is a US-military focused framework for describing both friendly and adversary actions in influence campaigns, and is inter-operable with the Disinformation Analysis and Risk Management (DISARM) framework. The purpose of the case study is to demonstrate how SP!CE can be used to describe influence campaign behaviors. We selected the Brazilian election as the target of the case study as it is known that there were significant amounts of mis- and disinformation present on social media during the campaigns. Our goal was to demonstrate how SP!CE could be applied in such a context, showing how social media content could be aligned with information campaign behaviors and how such an alignment can be used to analyze which mis- and disinformation narratives were in play. Additionally, we aim to provide insights on best practices regarding how to apply the framework in further research. We release the coding and screenshots of the relevant social media posts to support future research.

{{</citation>}}


### (108/126) A Low-cost, High-impact Node Injection Approach for Attacking Social Network Alignment (Shuyu Jiang et al., 2023)

{{<citation>}}

Shuyu Jiang, Yunxiang Qiu, Xian Mo, Rui Tang, Wei Wang. (2023)  
**A Low-cost, High-impact Node Injection Approach for Attacking Social Network Alignment**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2312.02790v1)  

---


**ABSTRACT**  
Social network alignment (SNA) holds significant importance for various downstream applications, prompting numerous professionals to develop and share SNA tools. Unfortunately, these tools can be exploited by malicious actors to integrate sensitive user information, posing cybersecurity risks. While many researchers have explored attacking SNA (ASNA) through a network modification attack way, practical feasibility remains a challenge. This paper introduces a novel approach, the node injection attack. To overcome the problem of modeling and solving within a limited time and balancing costs and benefits, we propose a low-cost, high-impact node injection attack via dynamic programming (DPNIA) framework. DPNIA models ASNA as a problem of maximizing the number of confirmed incorrect correspondent node pairs who have a greater similarity scores than the pairs between existing nodes, making ASNA solvable. Meanwhile, it employs a cross-network evaluation method to identify node vulnerability, facilitating a progressive attack from easy to difficult. Additionally, it utilizes an optimal injection strategy searching method, based on dynamic programming, to determine which links should be added between injected nodes and existing nodes, thereby achieving a high impact for attack effectiveness at a low cost. Experiments on four real-world datasets consistently demonstrate that DPNIA consistently and significantly outperforms various attack baselines.

{{</citation>}}


## cond-mat.mtrl-sci (1)



### (109/126) Materials Expert-Artificial Intelligence for Materials Discovery (Yanjun Liu et al., 2023)

{{<citation>}}

Yanjun Liu, Milena Jovanovic, Krishnanand Mallayya, Wesley J. Maddox, Andrew Gordon Wilson, Sebastian Klemenz, Leslie M. Schoop, Eun-Ah Kim. (2023)  
**Materials Expert-Artificial Intelligence for Materials Discovery**  

---
Primary Category: cond-mat.mtrl-sci  
Categories: cond-mat-mtrl-sci, cond-mat-str-el, cond-mat.mtrl-sci, cs-LG, physics-data-an  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.02796v1)  

---


**ABSTRACT**  
The advent of material databases provides an unprecedented opportunity to uncover predictive descriptors for emergent material properties from vast data space. However, common reliance on high-throughput ab initio data necessarily inherits limitations of such data: mismatch with experiments. On the other hand, experimental decisions are often guided by an expert's intuition honed from experiences that are rarely articulated. We propose using machine learning to "bottle" such operational intuition into quantifiable descriptors using expertly curated measurement-based data. We introduce "Materials Expert-Artificial Intelligence" (ME-AI) to encapsulate and articulate this human intuition. As a first step towards such a program, we focus on the topological semimetal (TSM) among square-net materials as the property inspired by the expert-identified descriptor based on structural information: the tolerance factor. We start by curating a dataset encompassing 12 primary features of 879 square-net materials, using experimental data whenever possible. We then use Dirichlet-based Gaussian process regression using a specialized kernel to reveal composite descriptors for square-net topological semimetals. The ME-AI learned descriptors independently reproduce expert intuition and expand upon it. Specifically, new descriptors point to hypervalency as a critical chemical feature predicting TSM within square-net compounds. Our success with a carefully defined problem points to the "machine bottling human insight" approach as promising for machine learning-aided material discovery.

{{</citation>}}


## cs.NI (2)



### (110/126) Empowering the 6G Cellular Architecture with Open RAN (Michele Polese et al., 2023)

{{<citation>}}

Michele Polese, Mischa Dohler, Falko Dressler, Melike Erol-Kantarci, Rittwik Jana, Raymond Knopp, Tommaso Melodia. (2023)  
**Empowering the 6G Cellular Architecture with Open RAN**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.02746v1)  

---


**ABSTRACT**  
Innovation and standardization in 5G have brought advancements to every facet of the cellular architecture. This ranges from the introduction of new frequency bands and signaling technologies for the radio access network (RAN), to a core network underpinned by micro-services and network function virtualization (NFV). However, like any emerging technology, the pace of real-world deployments does not instantly match the pace of innovation. To address this discrepancy, one of the key aspects under continuous development is the RAN with the aim of making it more open, adaptive, functional, and easy to manage. In this paper, we highlight the transformative potential of embracing novel cellular architectures by transitioning from conventional systems to the progressive principles of Open RAN. This promises to make 6G networks more agile, cost-effective, energy-efficient, and resilient. It opens up a plethora of novel use cases, ranging from ubiquitous support for autonomous devices to cost-effective expansions in regions previously underserved. The principles of Open RAN encompass: (i) a disaggregated architecture with modular and standardized interfaces; (ii) cloudification, programmability and orchestration; and (iii) AI-enabled data-centric closed-loop control and automation. We first discuss the transformative role Open RAN principles have played in the 5G era. Then, we adopt a system-level approach and describe how these Open RAN principles will support 6G RAN and architecture innovation. We qualitatively discuss potential performance gains that Open RAN principles yield for specific 6G use cases. For each principle, we outline the steps that research, development and standardization communities ought to take to make Open RAN principles central to next-generation cellular network designs.

{{</citation>}}


### (111/126) Congestion-aware Distributed Task Offloading in Wireless Multi-hop Networks Using Graph Neural Networks (Zhongyuan Zhao et al., 2023)

{{<citation>}}

Zhongyuan Zhao, Jake Perazzone, Gunjan Verma, Santiago Segarra. (2023)  
**Congestion-aware Distributed Task Offloading in Wireless Multi-hop Networks Using Graph Neural Networks**  

---
Primary Category: cs.NI  
Categories: 05C90, C-2-1; C-2-2, cs-LG, cs-NI, cs.NI, eess-SP  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.02471v1)  

---


**ABSTRACT**  
Computational offloading has become an enabling component for edge intelligence in mobile and smart devices. Existing offloading schemes mainly focus on mobile devices and servers, while ignoring the potential network congestion caused by tasks from multiple mobile devices, especially in wireless multi-hop networks. To fill this gap, we propose a low-overhead, congestion-aware distributed task offloading scheme by augmenting a distributed greedy framework with graph-based machine learning. In simulated wireless multi-hop networks with 20-110 nodes and a resource allocation scheme based on shortest path routing and contention-based link scheduling, our approach is demonstrated to be effective in reducing congestion or unstable queues under the context-agnostic baseline, while improving the execution latency over local computing.

{{</citation>}}


## cs.DC (1)



### (112/126) Part-time Power Measurements: nvidia-smi's Lack of Attention (Zeyu Yang et al., 2023)

{{<citation>}}

Zeyu Yang, Karel Adamek, Wesley Armour. (2023)  
**Part-time Power Measurements: nvidia-smi's Lack of Attention**  

---
Primary Category: cs.DC  
Categories: cs-AR, cs-DC, cs.DC  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2312.02741v1)  

---


**ABSTRACT**  
The GPU has emerged as the go-to accelerator for high throughput and parallel workloads, spanning scientific simulations to AI, thanks to its performance and power efficiency. Given that 6 out of the top 10 fastest supercomputers in the world use NVIDIA GPUs and many AI companies each employ 10,000's of NVIDIA GPUs, an accurate understanding of GPU power consumption is essential for making progress to further improve its efficiency. Despite the limited documentation and the lack of understanding of its mechanisms, NVIDIA GPUs' built-in power sensor, providing easily accessible power readings via the nvidia-smi interface, is widely used in energy efficient computing research on GPUs. Our study seeks to elucidate the internal mechanisms of the power readings provided by nvidia-smi and assess the accuracy of the power and energy consumption data. We have developed a suite of micro-benchmarks to profile the behaviour of nvidia-smi power readings and have evaluated them on over 70 different GPUs from all architectural generations since power measurement was first introduced in the 'Fermi' generation. We have identified several unforeseen problems in terms of power/energy measurement using nvidia-smi, for example on the A100 and H100 GPUs only 25% of the runtime is sampled for power consumption, during the other 75% of the time, the GPU can be using drastically different power and nvidia-smi and results presented by it are unaware of this. This along with other findings can lead to a drastic under/overestimation of energy consumed, especially when considering data centres housing tens of thousands of GPUs. We proposed several good practices that help to mitigate these problems. By comparing our results to those measured from an external power-meter, we have reduced the error in the energy measurement by an average of 35% and in some cases by as much as 65% in the test cases we present.

{{</citation>}}


## cs.AI (6)



### (113/126) Large Knowledge Model: Perspectives and Challenges (Huajun Chen, 2023)

{{<citation>}}

Huajun Chen. (2023)  
**Large Knowledge Model: Perspectives and Challenges**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, ChatGPT, GPT, Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2312.02706v1)  

---


**ABSTRACT**  
Humankind's understanding of the world is fundamentally linked to our perception and cognition, with \emph{human languages} serving as one of the major carriers of \emph{world knowledge}. In this vein, \emph{Large Language Models} (LLMs) like ChatGPT epitomize the pre-training of extensive, sequence-based world knowledge into neural networks, facilitating the processing and manipulation of this knowledge in a parametric space. This article explores large models through the lens of ``knowledge''. We initially investigate the role of symbolic knowledge such as Knowledge Graphs (KGs) in enhancing LLMs, covering aspects like knowledge-augmented language model, structure-inducing pre-training, knowledgeable prompts, structured CoT, knowledge editing, semantic tools for LLM and knowledgeable AI agents. Subsequently, we examine how LLMs can amplify traditional symbolic knowledge bases, encompassing aspects like using LLM as KG builder and controller, structured knowledge pretraining, LLM-enhanced symbolic reasoning, and the amalgamation of perception with cognition. Considering the intricate nature of human knowledge, we advocate for the creation of \emph{Large Knowledge Models} (LKM), specifically engineered to manage diversified spectrum of knowledge structures. This ambitious undertaking could entail several key challenges, such as disentangling knowledge representation from language models, restructuring pre-training with structured knowledge, and building large commonsense models, among others. We finally propose a five-``A'' principle to distinguish the concept of LKM.

{{</citation>}}


### (114/126) Training on Synthetic Data Beats Real Data in Multimodal Relation Extraction (Zilin Du et al., 2023)

{{<citation>}}

Zilin Du, Haoxin Li, Xu Guo, Boyang Li. (2023)  
**Training on Synthetic Data Beats Real Data in Multimodal Relation Extraction**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.AI  
Keywords: Relation Extraction  
[Paper Link](http://arxiv.org/abs/2312.03025v1)  

---


**ABSTRACT**  
The task of multimodal relation extraction has attracted significant research attention, but progress is constrained by the scarcity of available training data. One natural thought is to extend existing datasets with cross-modal generative models. In this paper, we consider a novel problem setting, where only unimodal data, either text or image, are available during training. We aim to train a multimodal classifier from synthetic data that perform well on real multimodal test data. However, training with synthetic data suffers from two obstacles: lack of data diversity and label information loss. To alleviate the issues, we propose Mutual Information-aware Multimodal Iterated Relational dAta GEneration (MI2RAGE), which applies Chained Cross-modal Generation (CCG) to promote diversity in the generated data and exploits a teacher network to select valuable training samples with high mutual information with the ground-truth labels. Comparing our method to direct training on synthetic data, we observed a significant improvement of 24.06% F1 with synthetic text and 26.42% F1 with synthetic images. Notably, our best model trained on completely synthetic images outperforms prior state-of-the-art models trained on real multimodal data by a margin of 3.76% in F1. Our codebase will be made available upon acceptance.

{{</citation>}}


### (115/126) DanZero+: Dominating the GuanDan Game through Reinforcement Learning (Youpeng Zhao et al., 2023)

{{<citation>}}

Youpeng Zhao, Yudong Lu, Jian Zhao, Wengang Zhou, Houqiang Li. (2023)  
**DanZero+: Dominating the GuanDan Game through Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.02561v1)  

---


**ABSTRACT**  
The utilization of artificial intelligence (AI) in card games has been a well-explored subject within AI research for an extensive period. Recent advancements have propelled AI programs to showcase expertise in intricate card games such as Mahjong, DouDizhu, and Texas Hold'em. In this work, we aim to develop an AI program for an exceptionally complex and popular card game called GuanDan. This game involves four players engaging in both competitive and cooperative play throughout a long process to upgrade their level, posing great challenges for AI due to its expansive state and action space, long episode length, and complex rules. Employing reinforcement learning techniques, specifically Deep Monte Carlo (DMC), and a distributed training framework, we first put forward an AI program named DanZero for this game. Evaluation against baseline AI programs based on heuristic rules highlights the outstanding performance of our bot. Besides, in order to further enhance the AI's capabilities, we apply policy-based reinforcement learning algorithm to GuanDan. To address the challenges arising from the huge action space, which will significantly impact the performance of policy-based algorithms, we adopt the pre-trained model to facilitate the training process and the achieved AI program manages to achieve a superior performance.

{{</citation>}}


### (116/126) Beyond Isolation: Multi-Agent Synergy for Improving Knowledge Graph Construction (Hongbin Ye et al., 2023)

{{<citation>}}

Hongbin Ye, Honghao Gui, Aijia Zhang, Tong Liu, Wei Hua, Weiqiang Jia. (2023)  
**Beyond Isolation: Multi-Agent Synergy for Improving Knowledge Graph Construction**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.03022v1)  

---


**ABSTRACT**  
Knowledge graph construction (KGC) is a multifaceted undertaking involving the extraction of entities, relations, and events. Traditionally, large language models (LLMs) have been viewed as solitary task-solving agents in this complex landscape. However, this paper challenges this paradigm by introducing a novel framework, CooperKGC. Departing from the conventional approach, CooperKGC establishes a collaborative processing network, assembling a KGC collaboration team capable of concurrently addressing entity, relation, and event extraction tasks. Our experiments unequivocally demonstrate that fostering collaboration and information interaction among diverse agents within CooperKGC yields superior results compared to individual cognitive processes operating in isolation. Importantly, our findings reveal that the collaboration facilitated by CooperKGC enhances knowledge selection, correction, and aggregation capabilities across multiple rounds of interactions.

{{</citation>}}


### (117/126) Creative Agents: Empowering Agents with Imagination for Creative Tasks (Chi Zhang et al., 2023)

{{<citation>}}

Chi Zhang, Penglin Cai, Yuhui Fu, Haoqi Yuan, Zongqing Lu. (2023)  
**Creative Agents: Empowering Agents with Imagination for Creative Tasks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.02519v1)  

---


**ABSTRACT**  
We study building embodied agents for open-ended creative tasks. While existing methods build instruction-following agents that can perform diverse open-ended tasks, none of them demonstrates creativity -- the ability to give novel and diverse task solutions implicit in the language instructions. This limitation comes from their inability to convert abstract language instructions into concrete task goals in the environment and perform long-horizon planning for such complicated goals. Given the observation that humans perform creative tasks with the help of imagination, we propose a class of solutions for creative agents, where the controller is enhanced with an imaginator that generates detailed imaginations of task outcomes conditioned on language instructions. We introduce several approaches to implementing the components of creative agents. We implement the imaginator with either a large language model for textual imagination or a diffusion model for visual imagination. The controller can either be a behavior-cloning policy learned from data or a pre-trained foundation model generating executable codes in the environment. We benchmark creative tasks with the challenging open-world game Minecraft, where the agents are asked to create diverse buildings given free-form language instructions. In addition, we propose novel evaluation metrics for open-ended creative tasks utilizing GPT-4V, which holds many advantages over existing metrics. We perform a detailed experimental analysis of creative agents, showing that creative agents are the first AI agents accomplishing diverse building creation in the survival mode of Minecraft. Our benchmark and models are open-source for future research on creative agents (https://github.com/PKU-RL/Creative-Agents).

{{</citation>}}


### (118/126) Let's Think Outside the Box: Exploring Leap-of-Thought in Large Language Models with Creative Humor Generation (Shanshan Zhong et al., 2023)

{{<citation>}}

Shanshan Zhong, Zhongzhan Huang, Shanghua Gao, Wushao Wen, Liang Lin, Marinka Zitnik, Pan Zhou. (2023)  
**Let's Think Outside the Box: Exploring Leap-of-Thought in Large Language Models with Creative Humor Generation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.02439v2)  

---


**ABSTRACT**  
Chain-of-Thought (CoT) guides large language models (LLMs) to reason step-by-step, and can motivate their logical reasoning ability. While effective for logical tasks, CoT is not conducive to creative problem-solving which often requires out-of-box thoughts and is crucial for innovation advancements. In this paper, we explore the Leap-of-Thought (LoT) abilities within LLMs -- a non-sequential, creative paradigm involving strong associations and knowledge leaps. To this end, we study LLMs on the popular Oogiri game which needs participants to have good creativity and strong associative thinking for responding unexpectedly and humorously to the given image, text, or both, and thus is suitable for LoT study. Then to investigate LLMs' LoT ability in the Oogiri game, we first build a multimodal and multilingual Oogiri-GO dataset which contains over 130,000 samples from the Oogiri game, and observe the insufficient LoT ability or failures of most existing LLMs on the Oogiri game. Accordingly, we introduce a creative Leap-of-Thought (CLoT) paradigm to improve LLM's LoT ability. CLoT first formulates the Oogiri-GO dataset into LoT-oriented instruction tuning data to train pretrained LLM for achieving certain LoT humor generation and discrimination abilities. Then CLoT designs an explorative self-refinement that encourages the LLM to generate more creative LoT data via exploring parallels between seemingly unrelated concepts and selects high-quality data to train itself for self-refinement. CLoT not only excels in humor generation in the Oogiri game but also boosts creative abilities in various tasks like cloud guessing game and divergent association task. These findings advance our understanding and offer a pathway to improve LLMs' creative capacities for innovative applications across domains. The dataset, code, and models will be released online. https://zhongshsh.github.io/CLoT/.

{{</citation>}}


## cs.CR (2)



### (119/126) Understanding Ethereum Mempool Security under Asymmetric DoS by Symbolic Fuzzing (Yibo Wang et al., 2023)

{{<citation>}}

Yibo Wang, Wanning Ding, Kai Li, Yuzhe Tang. (2023)  
**Understanding Ethereum Mempool Security under Asymmetric DoS by Symbolic Fuzzing**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.02642v1)  

---


**ABSTRACT**  
In blockchains, mempool controls transaction flow before consensus, denial of whose service hurts the health and security of blockchain networks. This paper presents MPFUZZ, the first mempool fuzzer to find asymmetric DoS bugs by symbolically exploring mempool state space and optimistically estimating the promisingness an intermediate state is in reaching bug oracles. Compared to the baseline blockchain fuzzers, MPFUZZ achieves a > 100x speedup in finding known DETER exploits. Running MPFUZZ on six major Ethereum clients leads to the discovering of new mempool vulnerabilities, which exhibit a wide variety of sophisticated patterns including stealthy mempool eviction and mempool locking. Rule-based mitigation schemes are proposed against newly discovered vulnerabilities.

{{</citation>}}


### (120/126) Skipping Scheme for Gate-hiding Garbled Circuits (Ke Lin, 2023)

{{<citation>}}

Ke Lin. (2023)  
**Skipping Scheme for Gate-hiding Garbled Circuits**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2312.02514v1)  

---


**ABSTRACT**  
In classic settings of garbled circuits, each gate type is leaked to improve both space and speed optimization. Zahur et al. have shown in EUROCRYPT 2015 that a typical linear garbling scheme requires at least two $\lambda$-bit elements per gate with a security parameter of $\lambda$, which limits their efficiency. In contrast to typical garbled circuits, gate-hiding garbled circuits have the potential to drastically reduce time costs, although they have been underappreciated.   We propose the first skipping scheme for gate-hiding garbled circuits to enhance the efficiency of evaluation by observing prime implicants. Our scheme introduces skip gates to eliminate the need to calculate the entire circuit, enabling unnecessary execution paths to be avoided. We also introduce two variants of our scheme that balance security with parallelism. A proof of hybrid security that combines simulation-based and symmetry-based security in semi-honest scenarios is presented to demonstrate its security under gate-hiding conditions. Our scheme will inspire new directions to improve the general garbling scheme and lead to more practical ones.

{{</citation>}}


## cs.IT (1)



### (121/126) A Neural Receiver for 5G NR Multi-user MIMO (Sebastian Cammerer et al., 2023)

{{<citation>}}

Sebastian Cammerer, Fayçal Aït Aoudia, Jakob Hoydis, Andreas Oeldemann, Andreas Roessler, Timo Mayer, Alexander Keller. (2023)  
**A Neural Receiver for 5G NR Multi-user MIMO**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.02601v1)  

---


**ABSTRACT**  
We introduce a neural network (NN)-based multiuser multiple-input multiple-output (MU-MIMO) receiver with 5G New Radio (5G NR) physical uplink shared channel (PUSCH) compatibility. The NN architecture is based on convolution layers to exploit the time and frequency correlation of the channel and a graph neural network (GNN) to handle multiple users. The proposed architecture adapts to an arbitrary number of sub-carriers and supports a varying number of multiple-input multiple-output (MIMO) layers and users without the need for any retraining. The receiver operates on an entire 5G NR slot, i.e., processes the entire received orthogonal frequency division multiplexing (OFDM) time-frequency resource grid by jointly performing channel estimation, equalization, and demapping. The proposed architecture operates less than 1 dB away from a baseline using linear minimum mean square error (LMMSE) channel estimation with K-best detection but benefits from a significantly lower computational complexity. We show the importance of a carefully designed training process such that the trained receiver is universal for a wide range of different unseen channel conditions. Finally, we demonstrate the results of a hardware-in-the-loop verification based on 3GPP compliant conformance test scenarios.

{{</citation>}}


## cs.RO (2)



### (122/126) MAINS: A Magnetic Field Aided Inertial Navigation System for Indoor Positioning (Chuan Huang et al., 2023)

{{<citation>}}

Chuan Huang, Gustaf Hendeby, Hassen Fourati, Christophe Prieur, Isaac Skog. (2023)  
**MAINS: A Magnetic Field Aided Inertial Navigation System for Indoor Positioning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.02599v1)  

---


**ABSTRACT**  
A Magnetic field Aided Inertial Navigation System (MAINS) for indoor navigation is proposed in this paper. MAINS leverages an array of magnetometers to measure spatial variations in the magnetic field, which are then used to estimate the displacement and orientation changes of the system, thereby aiding the inertial navigation system (INS). Experiments show that MAINS significantly outperforms the stand-alone INS, demonstrating a remarkable two orders of magnitude reduction in position error. Furthermore, when compared to the state-of-the-art magnetic-field-aided navigation approach, the proposed method exhibits slightly improved horizontal position accuracy. On the other hand, it has noticeably larger vertical error on datasets with large magnetic field variations. However, one of the main advantages of MAINS compared to the state-of-the-art is that it enables flexible sensor configurations. The experimental results show that the position error after 2 minutes of navigation in most cases is less than 3 meters when using an array of 30 magnetometers. Thus, the proposed navigation solution has the potential to solve one of the key challenges faced with current magnetic-field simultaneous localization and mapping (SLAM) solutions: the very limited allowable length of the exploration phase during which unvisited areas are mapped.

{{</citation>}}


### (123/126) Object Importance Estimation using Counterfactual Reasoning for Intelligent Driving (Pranay Gupta et al., 2023)

{{<citation>}}

Pranay Gupta, Abhijat Biswas, Henny Admoni, David Held. (2023)  
**Object Importance Estimation using Counterfactual Reasoning for Intelligent Driving**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2312.02467v1)  

---


**ABSTRACT**  
The ability to identify important objects in a complex and dynamic driving environment is essential for autonomous driving agents to make safe and efficient driving decisions. It also helps assistive driving systems decide when to alert drivers. We tackle object importance estimation in a data-driven fashion and introduce HOIST - Human-annotated Object Importance in Simulated Traffic. HOIST contains driving scenarios with human-annotated importance labels for vehicles and pedestrians. We additionally propose a novel approach that relies on counterfactual reasoning to estimate an object's importance. We generate counterfactual scenarios by modifying the motion of objects and ascribe importance based on how the modifications affect the ego vehicle's driving. Our approach outperforms strong baselines for the task of object importance estimation on HOIST. We also perform ablation studies to justify our design choices and show the significance of the different components of our proposed approach.

{{</citation>}}


## quant-ph (1)



### (124/126) Towards Optimizations of Quantum Circuit Simulation for Solving Max-Cut Problems with QAOA (Yu-Cheng Lin et al., 2023)

{{<citation>}}

Yu-Cheng Lin, Chuan-Chi Wang, Chia-Heng Tu, Shih-Hao Hung. (2023)  
**Towards Optimizations of Quantum Circuit Simulation for Solving Max-Cut Problems with QAOA**  

---
Primary Category: quant-ph  
Categories: cs-DC, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.03019v1)  

---


**ABSTRACT**  
Quantum approximate optimization algorithm (QAOA) is one of the popular quantum algorithms that are used to solve combinatorial optimization problems via approximations. QAOA is able to be evaluated on both physical and virtual quantum computers simulated by classical computers, with virtual ones being favored for their noise-free feature and availability. Nevertheless, performing QAOA on virtual quantum computers suffers from a slow simulation speed for solving combinatorial optimization problems which require large-scale quantum circuit simulation (QCS). In this paper, we propose techniques to accelerate QCS for QAOA using mathematical optimizations to compress quantum operations, incorporating efficient bitwise operations to further lower the computational complexity, and leveraging different levels of parallelisms from modern multi-core processors, with a study case to show the effectiveness on solving max-cut problems.

{{</citation>}}


## eess.SY (1)



### (125/126) Provable Reinforcement Learning for Networked Control Systems with Stochastic Packet Disordering (Wenqian Xue et al., 2023)

{{<citation>}}

Wenqian Xue, Yi Jiang, Frank L. Lewis, Bosen Lian. (2023)  
**Provable Reinforcement Learning for Networked Control Systems with Stochastic Packet Disordering**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.02498v1)  

---


**ABSTRACT**  
This paper formulates a stochastic optimal control problem for linear networked control systems featuring stochastic packet disordering with a unique stabilizing solution certified. The problem is solved by proposing reinforcement learning algorithms. A measurement method is first presented to deal with PD and calculate the newest control input. The NCSs with stochastic PD are modeled as stochastic NCSs. Then, given a cost function, a modified algebraic Riccati equation is derived within the formulation. We propose offline policy iteration and value iteration algorithms to solve the MARE associated with provable convergence. These two algorithms require knowledge of NCS dynamics and PD probabilities. To release that, we further design online model-free off-policy and Q-learning algorithms with an online estimation method for PD probability. Both model-free algorithms solve the optimal control problem using real-time system states, control inputs, and PD probability estimates. Simulation results verify the proposed formulation and algorithms at last.

{{</citation>}}


## q-bio.QM (1)



### (126/126) Protein Language Model-Powered 3D Ligand Binding Site Prediction from Protein Sequence (Shuo Zhang et al., 2023)

{{<citation>}}

Shuo Zhang, Lei Xie. (2023)  
**Protein Language Model-Powered 3D Ligand Binding Site Prediction from Protein Sequence**  

---
Primary Category: q-bio.QM  
Categories: cs-CL, cs-LG, q-bio-QM, q-bio.QM  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.03016v1)  

---


**ABSTRACT**  
Prediction of ligand binding sites of proteins is a fundamental and important task for understanding the function of proteins and screening potential drugs. Most existing methods require experimentally determined protein holo-structures as input. However, such structures can be unavailable on novel or less-studied proteins. To tackle this limitation, we propose LaMPSite, which only takes protein sequences and ligand molecular graphs as input for ligand binding site predictions. The protein sequences are used to retrieve residue-level embeddings and contact maps from the pre-trained ESM-2 protein language model. The ligand molecular graphs are fed into a graph neural network to compute atom-level embeddings. Then we compute and update the protein-ligand interaction embedding based on the protein residue-level embeddings and ligand atom-level embeddings, and the geometric constraints in the inferred protein contact map and ligand distance map. A final pooling on protein-ligand interaction embedding would indicate which residues belong to the binding sites. Without any 3D coordinate information of proteins, our proposed model achieves competitive performance compared to baseline methods that require 3D protein structures when predicting binding sites. Given that less than 50% of proteins have reliable structure information in the current stage, LaMPSite will provide new opportunities for drug discovery.

{{</citation>}}
