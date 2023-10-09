---
draft: false
title: "arXiv @ 2023.10.05"
date: 2023-10-05
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.05"
    identifier: arxiv_20231005
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [eess.SY (2)](#eesssy-2)
- [cs.LG (31)](#cslg-31)
- [cs.CL (41)](#cscl-41)
- [eess.IV (3)](#eessiv-3)
- [cs.RO (9)](#csro-9)
- [cs.SE (11)](#csse-11)
- [cs.SI (2)](#cssi-2)
- [cs.MA (1)](#csma-1)
- [cs.HC (5)](#cshc-5)
- [cs.CR (5)](#cscr-5)
- [math.NA (1)](#mathna-1)
- [cs.CV (26)](#cscv-26)
- [cs.AI (10)](#csai-10)
- [cs.LO (2)](#cslo-2)
- [quant-ph (1)](#quant-ph-1)
- [q-bio.NC (1)](#q-bionc-1)
- [physics.ao-ph (1)](#physicsao-ph-1)
- [cs.SD (2)](#cssd-2)
- [cs.MM (1)](#csmm-1)
- [cs.IR (1)](#csir-1)

## eess.SY (2)



### (1/156) On the Financial Consequences of Simplified Battery Sizing Models without Considering Operational Details (Nam Trong Dinh et al., 2023)

{{<citation>}}

Nam Trong Dinh, Sahand Karimi-Arpanahi, S. Ali Pourmousavi, Mingyu Guo, Julian Lemos-Vinasco, Jon A. R. Liisberg. (2023)  
**On the Financial Consequences of Simplified Battery Sizing Models without Considering Operational Details**  

---
Primary Category: eess.SY  
Categories: cs-IT, cs-SY, eess-SY, eess.SY, math-IT  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2310.02494v1)  

---


**ABSTRACT**  
Optimal battery sizing studies tend to overly simplify the practical aspects of battery operation within the battery sizing framework. Such assumptions may lead to a suboptimal battery capacity, resulting in significant financial losses for a battery project that could last more than a decade. In this paper, we compare the most common existing sizing methods in the literature with a battery sizing model that incorporates the practical operation of a battery, that is, receding horizon operation. Consequently, we quantify the financial losses caused by the suboptimal capacities obtained by these models for a realistic case study related to community battery storage (CBS). We develop the case study by constructing a mathematical framework for the CBS and local end users. Our results show that existing sizing methods can lead to financial losses of up to 22%.

{{</citation>}}


### (2/156) Health Guardian: Using Multi-modal Data to Understand Individual Health (Vince S. Siu et al., 2023)

{{<citation>}}

Vince S. Siu, Kuan Yu Hsieh, Italo Buleje, Takashi Itoh, Tian Hao, Ben Civjan, Nigel Hinds, Bing Dang, Jeffrey L. Rogers, Bo Wen. (2023)  
**Health Guardian: Using Multi-modal Data to Understand Individual Health**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: AI, Clinical  
[Paper Link](http://arxiv.org/abs/2310.01733v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) has shown great promise in revolutionizing the field of digital health by improving disease diagnosis, treatment, and prevention. This paper describes the Health Guardian platform, a non-commercial, scientific research-based platform developed by the IBM Digital Health team to rapidly translate AI research into cloud-based microservices. The platform can collect health-related data from various digital devices, including wearables and mobile applications. Its flexible architecture supports microservices that accept diverse data types such as text, audio, and video, expanding the range of digital health assessments and enabling holistic health evaluations by capturing voice, facial, and motion bio-signals. These microservices can be deployed to a clinical cohort specified through the Clinical Task Manager (CTM). The CTM then collects multi-modal, clinical data that can iteratively improve the accuracy of AI predictive models, discover new disease mechanisms, or identify novel biomarkers. This paper highlights three microservices with different input data types, including a text-based microservice for depression assessment, a video-based microservice for sit-to-stand mobility assessment, and a wearable-based microservice for functional mobility assessment. The CTM is also discussed as a tool to help design and set up clinical studies to unlock the full potential of the platform. Today, the Health Guardian platform is being leveraged in collaboration with research partners to optimize the development of AI models by utilizing a multitude of input sources. This approach streamlines research efforts, enhances efficiency, and facilitates the development and validation of digital health applications.

{{</citation>}}


## cs.LG (31)



### (3/156) DON-LSTM: Multi-Resolution Learning with DeepONets and Long Short-Term Memory Neural Networks (Katarzyna Michałowska et al., 2023)

{{<citation>}}

Katarzyna Michałowska, Somdatta Goswami, George Em Karniadakis, Signe Riemer-Sørensen. (2023)  
**DON-LSTM: Multi-Resolution Learning with DeepONets and Long Short-Term Memory Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.02491v1)  

---


**ABSTRACT**  
Deep operator networks (DeepONets, DONs) offer a distinct advantage over traditional neural networks in their ability to be trained on multi-resolution data. This property becomes especially relevant in real-world scenarios where high-resolution measurements are difficult to obtain, while low-resolution data is more readily available. Nevertheless, DeepONets alone often struggle to capture and maintain dependencies over long sequences compared to other state-of-the-art algorithms. We propose a novel architecture, named DON-LSTM, which extends the DeepONet with a long short-term memory network (LSTM). Combining these two architectures, we equip the network with explicit mechanisms to leverage multi-resolution data, as well as capture temporal dependencies in long sequences. We test our method on long-time-evolution modeling of multiple non-linear systems and show that the proposed multi-resolution DON-LSTM achieves significantly lower generalization error and requires fewer high-resolution samples compared to its vanilla counterparts.

{{</citation>}}


### (4/156) Splitting the Difference on Adversarial Training (Matan Levi et al., 2023)

{{<citation>}}

Matan Levi, Aryeh Kontorovich. (2023)  
**Splitting the Difference on Adversarial Training**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2310.02480v1)  

---


**ABSTRACT**  
The existence of adversarial examples points to a basic weakness of deep neural networks. One of the most effective defenses against such examples, adversarial training, entails training models with some degree of robustness, usually at the expense of a degraded natural accuracy. Most adversarial training methods aim to learn a model that finds, for each class, a common decision boundary encompassing both the clean and perturbed examples. In this work, we take a fundamentally different approach by treating the perturbed examples of each class as a separate class to be learned, effectively splitting each class into two classes: "clean" and "adversarial." This split doubles the number of classes to be learned, but at the same time considerably simplifies the decision boundaries. We provide a theoretical plausibility argument that sheds some light on the conditions under which our approach can be expected to be beneficial. Likewise, we empirically demonstrate that our method learns robust models while attaining optimal or near-optimal natural accuracy, e.g., on CIFAR-10 we obtain near-optimal natural accuracy of $95.01\%$ alongside significant robustness across multiple tasks. The ability to achieve such near-optimal natural accuracy, while maintaining a significant level of robustness, makes our method applicable to real-world applications where natural accuracy is at a premium. As a whole, our main contribution is a general method that confers a significant level of robustness upon classifiers with only minor or negligible degradation of their natural accuracy.

{{</citation>}}


### (5/156) Distributionally Safe Reinforcement Learning under Model Uncertainty: A Single-Level Approach by Differentiable Convex Programming (Alaa Eddine Chriat et al., 2023)

{{<citation>}}

Alaa Eddine Chriat, Chuangchuang Sun. (2023)  
**Distributionally Safe Reinforcement Learning under Model Uncertainty: A Single-Level Approach by Differentiable Convex Programming**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.02459v1)  

---


**ABSTRACT**  
Safety assurance is uncompromisable for safety-critical environments with the presence of drastic model uncertainties (e.g., distributional shift), especially with humans in the loop. However, incorporating uncertainty in safe learning will naturally lead to a bi-level problem, where at the lower level the (worst-case) safety constraint is evaluated within the uncertainty ambiguity set. In this paper, we present a tractable distributionally safe reinforcement learning framework to enforce safety under a distributional shift measured by a Wasserstein metric. To improve the tractability, we first use duality theory to transform the lower-level optimization from infinite-dimensional probability space where distributional shift is measured, to a finite-dimensional parametric space. Moreover, by differentiable convex programming, the bi-level safe learning problem is further reduced to a single-level one with two sequential computationally efficient modules: a convex quadratic program to guarantee safety followed by a projected gradient ascent to simultaneously find the worst-case uncertainty. This end-to-end differentiable framework with safety constraints, to the best of our knowledge, is the first tractable single-level solution to address distributional safety. We test our approach on first and second-order systems with varying complexities and compare our results with the uncertainty-agnostic policies, where our approach demonstrates a significant improvement on safety guarantees.

{{</citation>}}


### (6/156) Feather: An Elegant Solution to Effective DNN Sparsification (Athanasios Glentis Georgoulakis et al., 2023)

{{<citation>}}

Athanasios Glentis Georgoulakis, George Retsinas, Petros Maragos. (2023)  
**Feather: An Elegant Solution to Effective DNN Sparsification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.02448v1)  

---


**ABSTRACT**  
Neural Network pruning is an increasingly popular way for producing compact and efficient models, suitable for resource-limited environments, while preserving high performance. While the pruning can be performed using a multi-cycle training and fine-tuning process, the recent trend is to encompass the sparsification process during the standard course of training. To this end, we introduce Feather, an efficient sparse training module utilizing the powerful Straight-Through Estimator as its core, coupled with a new thresholding operator and a gradient scaling technique, enabling robust, out-of-the-box sparsification performance. Feather's effectiveness and adaptability is demonstrated using various architectures on the CIFAR dataset, while on ImageNet it achieves state-of-the-art Top-1 validation accuracy using the ResNet-50 architecture, surpassing existing methods, including more complex and computationally heavy ones, by a considerable margin. Code is publicly available at https://github.com/athglentis/feather .

{{</citation>}}


### (7/156) EGraFFBench: Evaluation of Equivariant Graph Neural Network Force Fields for Atomistic Simulations (Vaibhav Bihani et al., 2023)

{{<citation>}}

Vaibhav Bihani, Utkarsh Pratiush, Sajid Mannan, Tao Du, Zhimin Chen, Santiago Miret, Matthieu Micoulaut, Morten M Smedskjaer, Sayan Ranu, N M Anoop Krishnan. (2023)  
**EGraFFBench: Evaluation of Equivariant Graph Neural Network Force Fields for Atomistic Simulations**  

---
Primary Category: cs.LG  
Categories: cond-mat-mtrl-sci, cs-LG, cs.LG  
Keywords: Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2310.02428v1)  

---


**ABSTRACT**  
Equivariant graph neural networks force fields (EGraFFs) have shown great promise in modelling complex interactions in atomic systems by exploiting the graphs' inherent symmetries. Recent works have led to a surge in the development of novel architectures that incorporate equivariance-based inductive biases alongside architectural innovations like graph transformers and message passing to model atomic interactions. However, thorough evaluations of these deploying EGraFFs for the downstream task of real-world atomistic simulations, is lacking. To this end, here we perform a systematic benchmarking of 6 EGraFF algorithms (NequIP, Allegro, BOTNet, MACE, Equiformer, TorchMDNet), with the aim of understanding their capabilities and limitations for realistic atomistic simulations. In addition to our thorough evaluation and analysis on eight existing datasets based on the benchmarking literature, we release two new benchmark datasets, propose four new metrics, and three new challenging tasks. The new datasets and tasks evaluate the performance of EGraFF to out-of-distribution data, in terms of different crystal structures, temperatures, and new molecules. Interestingly, evaluation of the EGraFF models based on dynamic simulations reveals that having a lower error on energy or force does not guarantee stable or reliable simulation or faithful replication of the atomic structures. Moreover, we find that no model clearly outperforms other models on all datasets and tasks. Importantly, we show that the performance of all the models on out-of-distribution datasets is unreliable, pointing to the need for the development of a foundation model for force fields that can be used in real-world simulations. In summary, this work establishes a rigorous framework for evaluating machine learning force fields in the context of atomic simulations and points to open research challenges within this domain.

{{</citation>}}


### (8/156) Delta-AI: Local objectives for amortized inference in sparse graphical models (Jean-Pierre Falet et al., 2023)

{{<citation>}}

Jean-Pierre Falet, Hae Beom Lee, Nikolay Malkin, Chen Sun, Dragos Secrieru, Dinghuai Zhang, Guillaume Lajoie, Yoshua Bengio. (2023)  
**Delta-AI: Local objectives for amortized inference in sparse graphical models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02423v1)  

---


**ABSTRACT**  
We present a new algorithm for amortized inference in sparse probabilistic graphical models (PGMs), which we call $\Delta$-amortized inference ($\Delta$-AI). Our approach is based on the observation that when the sampling of variables in a PGM is seen as a sequence of actions taken by an agent, sparsity of the PGM enables local credit assignment in the agent's policy learning objective. This yields a local constraint that can be turned into a local loss in the style of generative flow networks (GFlowNets) that enables off-policy training but avoids the need to instantiate all the random variables for each parameter update, thus speeding up training considerably. The $\Delta$-AI objective matches the conditional distribution of a variable given its Markov blanket in a tractable learned sampler, which has the structure of a Bayesian network, with the same conditional distribution under the target PGM. As such, the trained sampler recovers marginals and conditional distributions of interest and enables inference of partial subsets of variables. We illustrate $\Delta$-AI's effectiveness for sampling from synthetic PGMs and training latent variable models with sparse factor structure.

{{</citation>}}


### (9/156) Can a student Large Language Model perform as well as it's teacher? (Sia Gholami et al., 2023)

{{<citation>}}

Sia Gholami, Marwan Omar. (2023)  
**Can a student Large Language Model perform as well as it's teacher?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02421v1)  

---


**ABSTRACT**  
The burgeoning complexity of contemporary deep learning models, while achieving unparalleled accuracy, has inadvertently introduced deployment challenges in resource-constrained environments. Knowledge distillation, a technique aiming to transfer knowledge from a high-capacity "teacher" model to a streamlined "student" model, emerges as a promising solution to this dilemma. This paper provides a comprehensive overview of the knowledge distillation paradigm, emphasizing its foundational principles such as the utility of soft labels and the significance of temperature scaling. Through meticulous examination, we elucidate the critical determinants of successful distillation, including the architecture of the student model, the caliber of the teacher, and the delicate balance of hyperparameters. While acknowledging its profound advantages, we also delve into the complexities and challenges inherent in the process. Our exploration underscores knowledge distillation's potential as a pivotal technique in optimizing the trade-off between model performance and deployment efficiency.

{{</citation>}}


### (10/156) Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness (Young Jin Kim et al., 2023)

{{<citation>}}

Young Jin Kim, Raffy Fahim, Hany Hassan Awadalla. (2023)  
**Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2310.02410v1)  

---


**ABSTRACT**  
Large Mixture of Experts (MoE) models could achieve state-of-the-art quality on various language tasks, including machine translation task, thanks to the efficient model scaling capability with expert parallelism. However, it has brought a fundamental issue of larger memory consumption and increased memory bandwidth bottleneck at deployment time. In this paper, we propose Mixture of Quantized Experts (MoQE) which is a simple weight-only quantization method applying ultra low-bit down to 2-bit quantizations only to expert weights for mitigating the increased memory and latency issues of MoE models. We show that low-bit quantization together with the MoE architecture delivers a reliable model performance while reducing the memory size significantly even without any additional training in most cases. In particular, expert layers in MoE models are much more robust to the quantization than conventional feedforward networks (FFN) layers. In our comprehensive analysis, we show that MoE models with 2-bit expert weights can deliver better model performance than the dense model trained on the same dataset. As a result of low-bit quantization, we show the model size can be reduced by 79.6% of the original half precision floating point (fp16) MoE model. Combined with an optimized GPU runtime implementation, it also achieves 1.24X speed-up on A100 GPUs.

{{</citation>}}


### (11/156) PCGPT: Procedural Content Generation via Transformers (Sajad Mohaghegh et al., 2023)

{{<citation>}}

Sajad Mohaghegh, Mohammad Amin Ramezan Dehnavi, Golnoosh Abdollahinejad, Matin Hashemi. (2023)  
**PCGPT: Procedural Content Generation via Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.02405v1)  

---


**ABSTRACT**  
The paper presents the PCGPT framework, an innovative approach to procedural content generation (PCG) using offline reinforcement learning and transformer networks. PCGPT utilizes an autoregressive model based on transformers to generate game levels iteratively, addressing the challenges of traditional PCG methods such as repetitive, predictable, or inconsistent content. The framework models trajectories of actions, states, and rewards, leveraging the transformer's self-attention mechanism to capture temporal dependencies and causal relationships. The approach is evaluated in the Sokoban puzzle game, where the model predicts items that are needed with their corresponding locations. Experimental results on the game Sokoban demonstrate that PCGPT generates more complex and diverse game content. Interestingly, it achieves these results in significantly fewer steps compared to existing methods, showcasing its potential for enhancing game design and online content generation. Our model represents a new PCG paradigm which outperforms previous methods.

{{</citation>}}


### (12/156) Secure and Effective Data Appraisal for Machine Learning (Xu Ouyang et al., 2023)

{{<citation>}}

Xu Ouyang, Changhong Yang, Felix Xiaozhu Lin, Yangfeng Ji. (2023)  
**Secure and Effective Data Appraisal for Machine Learning**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2310.02373v2)  

---


**ABSTRACT**  
Essential for an unfettered data market is the ability to discreetly select and evaluate training data before finalizing a transaction between the data owner and model owner. To safeguard the privacy of both data and model, this process involves scrutinizing the target model through Multi-Party Computation (MPC). While prior research has posited that the MPC-based evaluation of Transformer models is excessively resource-intensive, this paper introduces an innovative approach that renders data selection practical. The contributions of this study encompass three pivotal elements: (1) a groundbreaking pipeline for confidential data selection using MPC, (2) replicating intricate high-dimensional operations with simplified low-dimensional MLPs trained on a limited subset of pertinent data, and (3) implementing MPC in a concurrent, multi-phase manner. The proposed method is assessed across an array of Transformer models and NLP/CV benchmarks. In comparison to the direct MPC-based evaluation of the target model, our approach substantially reduces the time required, from thousands of hours to mere tens of hours, with only a nominal 0.20% dip in accuracy when training with the selected data.

{{</citation>}}


### (13/156) A Deep Reinforcement Learning Approach for Interactive Search with Sentence-level Feedback (Jianghong Zhou et al., 2023)

{{<citation>}}

Jianghong Zhou, Joyce C. Ho, Chen Lin, Eugene Agichtein. (2023)  
**A Deep Reinforcement Learning Approach for Interactive Search with Sentence-level Feedback**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-HC, cs-IR, cs-LG, cs.LG  
Keywords: BERT, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03043v1)  

---


**ABSTRACT**  
Interactive search can provide a better experience by incorporating interaction feedback from the users. This can significantly improve search accuracy as it helps avoid irrelevant information and captures the users' search intents. Existing state-of-the-art (SOTA) systems use reinforcement learning (RL) models to incorporate the interactions but focus on item-level feedback, ignoring the fine-grained information found in sentence-level feedback. Yet such feedback requires extensive RL action space exploration and large amounts of annotated data. This work addresses these challenges by proposing a new deep Q-learning (DQ) approach, DQrank. DQrank adapts BERT-based models, the SOTA in natural language processing, to select crucial sentences based on users' engagement and rank the items to obtain more satisfactory responses. We also propose two mechanisms to better explore optimal actions. DQrank further utilizes the experience replay mechanism in DQ to store the feedback sentences to obtain a better initial ranking performance. We validate the effectiveness of DQrank on three search datasets. The results show that DQRank performs at least 12% better than the previous SOTA RL approaches. We also conduct detailed ablation studies. The ablation results demonstrate that each model component can efficiently extract and accumulate long-term engagement effects from the users' sentence-level feedback. This structure offers new technologies with promised performance to construct a search system with sentence-level interaction.

{{</citation>}}


### (14/156) Language Models Represent Space and Time (Wes Gurnee et al., 2023)

{{<citation>}}

Wes Gurnee, Max Tegmark. (2023)  
**Language Models Represent Space and Time**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02207v1)  

---


**ABSTRACT**  
The capabilities of large language models (LLMs) have sparked debate over whether such systems just learn an enormous collection of superficial statistics or a coherent model of the data generating process -- a world model. We find evidence for the latter by analyzing the learned representations of three spatial datasets (world, US, NYC places) and three temporal datasets (historical figures, artworks, news headlines) in the Llama-2 family of models. We discover that LLMs learn linear representations of space and time across multiple scales. These representations are robust to prompting variations and unified across different entity types (e.g. cities and landmarks). In addition, we identify individual ``space neurons'' and ``time neurons'' that reliably encode spatial and temporal coordinates. Our analysis demonstrates that modern LLMs acquire structured knowledge about fundamental dimensions such as space and time, supporting the view that they learn not merely superficial statistics, but literal world models.

{{</citation>}}


### (15/156) 1D-CapsNet-LSTM: A Deep Learning-Based Model for Multi-Step Stock Index Forecasting (Cheng Zhang et al., 2023)

{{<citation>}}

Cheng Zhang, Nilam Nur Amir Sjarif, Roslina Ibrahim. (2023)  
**1D-CapsNet-LSTM: A Deep Learning-Based Model for Multi-Step Stock Index Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.02090v1)  

---


**ABSTRACT**  
Multi-step forecasting of stock market index prices is a crucial task in the financial sector, playing a pivotal role in decision-making across various financial activities. However, forecasting results are often unsatisfactory owing to the stochastic and volatile nature of the data. Researchers have made various attempts, and this process is ongoing. Inspired by convolutional neural network long short-term memory (CNN-LSTM) networks that utilize a 1D CNN for feature extraction to boost model performance, this study explores the use of a capsule network (CapsNet) as an advanced feature extractor in an LSTM-based forecasting model to enhance multi-step predictions. To this end, a novel neural architecture called 1D-CapsNet-LSTM was introduced, which combines a 1D CapsNet to extract high-level features from 1D sequential data and an LSTM layer to capture the temporal dependencies between the previously extracted features and uses a multi-input multi-output (MIMO) strategy to maintain the stochastic dependencies between the predicted values at different time steps. The proposed model was evaluated based on several real-world stock market indices, including Standard & Poor's 500 (S&P 500), Dow Jones Industrial Average (DJIA), Nasdaq Composite Index (IXIC), and New York Stock Exchange (NYSE), and was compared with baseline models such as LSTM, recurrent neural network (RNN), and CNN-LSTM in terms of various evaluation metrics. The comparison results suggest that the 1D-CapsNet-LSTM model outperforms the baseline models and has immense potential for the effective handling of complex prediction tasks.

{{</citation>}}


### (16/156) De Novo Drug Design with Joint Transformers (Adam Izdebski et al., 2023)

{{<citation>}}

Adam Izdebski, Ewelina Weglarz-Tomczak, Ewa Szczurek, Jakub M. Tomczak. (2023)  
**De Novo Drug Design with Joint Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.02066v1)  

---


**ABSTRACT**  
De novo drug design requires simultaneously generating novel molecules outside of training data and predicting their target properties, making it a hard task for generative models. To address this, we propose Joint Transformer that combines a Transformer decoder, a Transformer encoder, and a predictor in a joint generative model with shared weights. We show that training the model with a penalized log-likelihood objective results in state-of-the-art performance in molecule generation, while decreasing the prediction error on newly sampled molecules, as compared to a fine-tuned decoder-only Transformer, by 42%. Finally, we propose a probabilistic black-box optimization algorithm that employs Joint Transformer to generate novel molecules with improved target properties, as compared to the training data, outperforming other SMILES-based optimization methods in de novo drug design.

{{</citation>}}


### (17/156) The Inhibitor: ReLU and Addition-Based Attention for Efficient Transformers (Rickard Brännvall, 2023)

{{<citation>}}

Rickard Brännvall. (2023)  
**The Inhibitor: ReLU and Addition-Based Attention for Efficient Transformers**  

---
Primary Category: cs.LG  
Categories: 68T07 68T07, I-2-6, cs-LG, cs.LG  
Keywords: AI, Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.02041v1)  

---


**ABSTRACT**  
To enhance the computational efficiency of quantized Transformers, we replace the dot-product and Softmax-based attention with an alternative mechanism involving addition and ReLU activation only. This side-steps the expansion to double precision often required by matrix multiplication and avoids costly Softmax evaluations but maintains much of the core functionality of conventional dot-product attention. It can enable more efficient execution and support larger quantized Transformer models on resource-constrained hardware or alternative arithmetic systems like homomorphic encryption. Training experiments on four common benchmark tasks show test set prediction scores comparable to those of conventional Transformers with dot-product attention. Our scaling experiments also suggest significant computational savings, both in plaintext and under encryption. In particular, we believe that the ReLU and addition-based attention mechanism introduced in this paper may enable privacy-preserving AI applications operating under homomorphic encryption by avoiding the costly multiplication of encrypted variables.

{{</citation>}}


### (18/156) Between accurate prediction and poor decision making: the AI/ML gap (Gianluca Bontempi, 2023)

{{<citation>}}

Gianluca Bontempi. (2023)  
**Between accurate prediction and poor decision making: the AI/ML gap**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02029v1)  

---


**ABSTRACT**  
Intelligent agents rely on AI/ML functionalities to predict the consequence of possible actions and optimise the policy. However, the effort of the research community in addressing prediction accuracy has been so intense (and successful) that it created the illusion that the more accurate the learner prediction (or classification) the better would have been the final decision. Now, such an assumption is valid only if the (human or artificial) decision maker has complete knowledge of the utility of the possible actions. This paper argues that AI/ML community has taken so far a too unbalanced approach by devoting excessive attention to the estimation of the state (or target) probability to the detriment of accurate and reliable estimations of the utility. In particular, few evidence exists about the impact of a wrong utility assessment on the resulting expected utility of the decision strategy. This situation is creating a substantial gap between the expectations and the effective impact of AI solutions, as witnessed by recent criticisms and emphasised by the regulatory legislative efforts. This paper aims to study this gap by quantifying the sensitivity of the expected utility to the utility uncertainty and comparing it to the one due to probability estimation. Theoretical and simulated results show that an inaccurate utility assessment may as (and sometimes) more harmful than a poor probability estimation. The final recommendation to the community is then to undertake a focus shift from a pure accuracy-driven (or obsessed) approach to a more utility-aware methodology.

{{</citation>}}


### (19/156) DeepHGCN: Toward Deeper Hyperbolic Graph Convolutional Networks (Jiaxu Liu et al., 2023)

{{<citation>}}

Jiaxu Liu, Xinping Yi, Xiaowei Huang. (2023)  
**DeepHGCN: Toward Deeper Hyperbolic Graph Convolutional Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2310.02027v2)  

---


**ABSTRACT**  
Hyperbolic graph convolutional networks (HGCN) have demonstrated significant potential in extracting information from hierarchical graphs. However, existing HGCNs are limited to shallow architectures, due to the expensive hyperbolic operations and the over-smoothing issue as depth increases. Although in GCNs, treatments have been applied to alleviate over-smoothing, developing a hyperbolic therapy presents distinct challenges since operations should be carefully designed to fit the hyperbolic nature. Addressing the above challenges, in this work, we propose DeepHGCN, the first deep multi-layer HGCN architecture with dramatically improved computational efficiency and substantially alleviated over-smoothing effect. DeepHGCN presents two key enablers of deep HGCNs: (1) a novel hyperbolic feature transformation layer that enables fast and accurate linear maps; and (2) Techniques such as hyperbolic residual connections and regularization for both weights and features facilitated by an efficient hyperbolic midpoint method. Extensive experiments demonstrate that DeepHGCN obtains significant improvements in link prediction and node classification tasks compared to both Euclidean and shallow hyperbolic GCN variants.

{{</citation>}}


### (20/156) OOD Aware Supervised Contrastive Learning (Soroush Seifi et al., 2023)

{{<citation>}}

Soroush Seifi, Daniel Olmeda Reino, Nikolay Chumerin, Rahaf Aljundi. (2023)  
**OOD Aware Supervised Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.01942v1)  

---


**ABSTRACT**  
Out-of-Distribution (OOD) detection is a crucial problem for the safe deployment of machine learning models identifying samples that fall outside of the training distribution, i.e. in-distribution data (ID). Most OOD works focus on the classification models trained with Cross Entropy (CE) and attempt to fix its inherent issues. In this work we leverage powerful representation learned with Supervised Contrastive (SupCon) training and propose a holistic approach to learn a classifier robust to OOD data. We extend SupCon loss with two additional contrast terms. The first term pushes auxiliary OOD representations away from ID representations without imposing any constraints on similarities among auxiliary data. The second term pushes OOD features far from the existing class prototypes, while pushing ID representations closer to their corresponding class prototype. When auxiliary OOD data is not available, we propose feature mixing techniques to efficiently generate pseudo-OOD features. Our solution is simple and efficient and acts as a natural extension of the closed-set supervised contrastive representation learning. We compare against different OOD detection methods on the common benchmarks and show state-of-the-art results.

{{</citation>}}


### (21/156) FiGURe: Simple and Efficient Unsupervised Node Representations with Filter Augmentations (Chanakya Ekbote et al., 2023)

{{<citation>}}

Chanakya Ekbote, Ajinkya Pankaj Deshpande, Arun Iyer, Ramakrishna Bairi, Sundararajan Sellamanickam. (2023)  
**FiGURe: Simple and Efficient Unsupervised Node Representations with Filter Augmentations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.01892v2)  

---


**ABSTRACT**  
Unsupervised node representations learnt using contrastive learning-based methods have shown good performance on downstream tasks. However, these methods rely on augmentations that mimic low-pass filters, limiting their performance on tasks requiring different eigen-spectrum parts. This paper presents a simple filter-based augmentation method to capture different parts of the eigen-spectrum. We show significant improvements using these augmentations. Further, we show that sharing the same weights across these different filter augmentations is possible, reducing the computational load. In addition, previous works have shown that good performance on downstream tasks requires high dimensional representations. Working with high dimensions increases the computations, especially when multiple augmentations are involved. We mitigate this problem and recover good performance through lower dimensional embeddings using simple random Fourier feature projections. Our method, FiGURe achieves an average gain of up to 4.4%, compared to the state-of-the-art unsupervised models, across all datasets in consideration, both homophilic and heterophilic. Our code can be found at: https://github.com/microsoft/figure.

{{</citation>}}


### (22/156) Towards Stable Backdoor Purification through Feature Shift Tuning (Rui Min et al., 2023)

{{<citation>}}

Rui Min, Zeyu Qin, Li Shen, Minhao Cheng. (2023)  
**Towards Stable Backdoor Purification through Feature Shift Tuning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01875v1)  

---


**ABSTRACT**  
It has been widely observed that deep neural networks (DNN) are vulnerable to backdoor attacks where attackers could manipulate the model behavior maliciously by tampering with a small set of training samples. Although a line of defense methods is proposed to mitigate this threat, they either require complicated modifications to the training process or heavily rely on the specific model architecture, which makes them hard to deploy into real-world applications. Therefore, in this paper, we instead start with fine-tuning, one of the most common and easy-to-deploy backdoor defenses, through comprehensive evaluations against diverse attack scenarios. Observations made through initial experiments show that in contrast to the promising defensive results on high poisoning rates, vanilla tuning methods completely fail at low poisoning rate scenarios. Our analysis shows that with the low poisoning rate, the entanglement between backdoor and clean features undermines the effect of tuning-based defenses. Therefore, it is necessary to disentangle the backdoor and clean features in order to improve backdoor purification. To address this, we introduce Feature Shift Tuning (FST), a method for tuning-based backdoor purification. Specifically, FST encourages feature shifts by actively deviating the classifier weights from the originally compromised weights. Extensive experiments demonstrate that our FST provides consistently stable performance under different attack settings. Additionally, it is also convenient to deploy in real-world scenarios with significantly reduced computation costs. Our codes are available at \url{https://github.com/AISafety-HKUST/stable_backdoor_purification}.

{{</citation>}}


### (23/156) DeepDecipher: Accessing and Investigating Neuron Activation in Large Language Models (Albert Garde et al., 2023)

{{<citation>}}

Albert Garde, Esben Kran, Fazl Barez. (2023)  
**DeepDecipher: Accessing and Investigating Neuron Activation in Large Language Models**  

---
Primary Category: cs.LG  
Categories: 68T50 (Primary) 68T05 (Secondary), I-2-7, cs-LG, cs.LG  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.01870v1)  

---


**ABSTRACT**  
As large language models (LLMs) become more capable, there is an urgent need for interpretable and transparent tools. Current methods are difficult to implement, and accessible tools to analyze model internals are lacking. To bridge this gap, we present DeepDecipher - an API and interface for probing neurons in transformer models' MLP layers. DeepDecipher makes the outputs of advanced interpretability techniques for LLMs readily available. The easy-to-use interface also makes inspecting these complex models more intuitive. This paper outlines DeepDecipher's design and capabilities. We demonstrate how to analyze neurons, compare models, and gain insights into model behavior. For example, we contrast DeepDecipher's functionality with similar tools like Neuroscope and OpenAI's Neuron Explainer. DeepDecipher enables efficient, scalable analysis of LLMs. By granting access to state-of-the-art interpretability methods, DeepDecipher makes LLMs more transparent, trustworthy, and safe. Researchers, engineers, and developers can quickly diagnose issues, audit systems, and advance the field.

{{</citation>}}


### (24/156) Conditional Instrumental Variable Regression with Representation Learning for Causal Inference (Debo Cheng et al., 2023)

{{<citation>}}

Debo Cheng, Ziqi Xu, Jiuyong Li, Lin Liu, Jixue Liu, Thuc Duy Le. (2023)  
**Conditional Instrumental Variable Regression with Representation Learning for Causal Inference**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.01865v1)  

---


**ABSTRACT**  
This paper studies the challenging problem of estimating causal effects from observational data, in the presence of unobserved confounders. The two-stage least square (TSLS) method and its variants with a standard instrumental variable (IV) are commonly used to eliminate confounding bias, including the bias caused by unobserved confounders, but they rely on the linearity assumption. Besides, the strict condition of unconfounded instruments posed on a standard IV is too strong to be practical. To address these challenging and practical problems of the standard IV method (linearity assumption and the strict condition), in this paper, we use a conditional IV (CIV) to relax the unconfounded instrument condition of standard IV and propose a non-linear CIV regression with Confounding Balancing Representation Learning, CBRL.CIV, for jointly eliminating the confounding bias from unobserved confounders and balancing the observed confounders, without the linearity assumption. We theoretically demonstrate the soundness of CBRL.CIV. Extensive experiments on synthetic and two real-world datasets show the competitive performance of CBRL.CIV against state-of-the-art IV-based estimators and superiority in dealing with the non-linear situation.

{{</citation>}}


### (25/156) Towards Robust Fidelity for Evaluating Explainability of Graph Neural Networks (Xu Zheng et al., 2023)

{{<citation>}}

Xu Zheng, Farhad Shirani, Tianchun Wang, Wei Cheng, Zhuomin Chen, Haifeng Chen, Hua Wei, Dongsheng Luo. (2023)  
**Towards Robust Fidelity for Evaluating Explainability of Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.01820v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) are neural models that leverage the dependency structure in graphical data via message passing among the graph nodes. GNNs have emerged as pivotal architectures in analyzing graph-structured data, and their expansive application in sensitive domains requires a comprehensive understanding of their decision-making processes -- necessitating a framework for GNN explainability. An explanation function for GNNs takes a pre-trained GNN along with a graph as input, to produce a `sufficient statistic' subgraph with respect to the graph label. A main challenge in studying GNN explainability is to provide fidelity measures that evaluate the performance of these explanation functions. This paper studies this foundational challenge, spotlighting the inherent limitations of prevailing fidelity metrics, including $Fid_+$, $Fid_-$, and $Fid_\Delta$. Specifically, a formal, information-theoretic definition of explainability is introduced and it is shown that existing metrics often fail to align with this definition across various statistical scenarios. The reason is due to potential distribution shifts when subgraphs are removed in computing these fidelity measures. Subsequently, a robust class of fidelity measures are introduced, and it is shown analytically that they are resilient to distribution shift issues and are applicable in a wide range of scenarios. Extensive empirical analysis on both synthetic and real datasets are provided to illustrate that the proposed metrics are more coherent with gold standard metrics.

{{</citation>}}


### (26/156) GNNX-BENCH: Unravelling the Utility of Perturbation-based GNN Explainers through In-depth Benchmarking (Mert Kosan et al., 2023)

{{<citation>}}

Mert Kosan, Samidha Verma, Burouj Armgaan, Khushbu Pahwa, Ambuj Singh, Sourav Medya, Sayan Ranu. (2023)  
**GNNX-BENCH: Unravelling the Utility of Perturbation-based GNN Explainers through In-depth Benchmarking**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.01794v1)  

---


**ABSTRACT**  
Numerous explainability methods have been proposed to shed light on the inner workings of GNNs. Despite the inclusion of empirical evaluations in all the proposed algorithms, the interrogative aspects of these evaluations lack diversity. As a result, various facets of explainability pertaining to GNNs, such as a comparative analysis of counterfactual reasoners, their stability to variational factors such as different GNN architectures, noise, stochasticity in non-convex loss surfaces, feasibility amidst domain constraints, and so forth, have yet to be formally investigated. Motivated by this need, we present a benchmarking study on perturbation-based explainability methods for GNNs, aiming to systematically evaluate and compare a wide range of explainability techniques. Among the key findings of our study, we identify the Pareto-optimal methods that exhibit superior efficacy and stability in the presence of noise. Nonetheless, our study reveals that all algorithms are affected by stability issues when faced with noisy data. Furthermore, we have established that the current generation of counterfactual explainers often fails to provide feasible recourses due to violations of topological constraints encoded by domain-specific considerations. Overall, this benchmarking study empowers stakeholders in the field of GNNs with a comprehensive understanding of the state-of-the-art explainability methods, potential research problems for further enhancement, and the implications of their application in real-world scenarios.

{{</citation>}}


### (27/156) Can large language models provide useful feedback on research papers? A large-scale empirical analysis (Weixin Liang et al., 2023)

{{<citation>}}

Weixin Liang, Yuhui Zhang, Hancheng Cao, Binglu Wang, Daisy Ding, Xinyu Yang, Kailas Vodrahalli, Siyu He, Daniel Smith, Yian Yin, Daniel McFarland, James Zou. (2023)  
**Can large language models provide useful feedback on research papers? A large-scale empirical analysis**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-HC, cs-LG, cs.LG  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.01783v1)  

---


**ABSTRACT**  
Expert feedback lays the foundation of rigorous research. However, the rapid growth of scholarly production and intricate knowledge specialization challenge the conventional scientific feedback mechanisms. High-quality peer reviews are increasingly difficult to obtain. Researchers who are more junior or from under-resourced settings have especially hard times getting timely feedback. With the breakthrough of large language models (LLM) such as GPT-4, there is growing interest in using LLMs to generate scientific feedback on research manuscripts. However, the utility of LLM-generated feedback has not been systematically studied. To address this gap, we created an automated pipeline using GPT-4 to provide comments on the full PDFs of scientific papers. We evaluated the quality of GPT-4's feedback through two large-scale studies. We first quantitatively compared GPT-4's generated feedback with human peer reviewer feedback in 15 Nature family journals (3,096 papers in total) and the ICLR machine learning conference (1,709 papers). The overlap in the points raised by GPT-4 and by human reviewers (average overlap 30.85% for Nature journals, 39.23% for ICLR) is comparable to the overlap between two human reviewers (average overlap 28.58% for Nature journals, 35.25% for ICLR). The overlap between GPT-4 and human reviewers is larger for the weaker papers. We then conducted a prospective user study with 308 researchers from 110 US institutions in the field of AI and computational biology to understand how researchers perceive feedback generated by our GPT-4 system on their own papers. Overall, more than half (57.4%) of the users found GPT-4 generated feedback helpful/very helpful and 82.4% found it more beneficial than feedback from at least some human reviewers. While our findings show that LLM-generated feedback can help researchers, we also identify several limitations.

{{</citation>}}


### (28/156) Exploring Counterfactual Alignment Loss towards Human-centered AI (Mingzhou Liu et al., 2023)

{{<citation>}}

Mingzhou Liu, Xinwei Sun, Ching-Wen Lee, Yu Qiao, Yizhou Wang. (2023)  
**Exploring Counterfactual Alignment Loss towards Human-centered AI**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01766v1)  

---


**ABSTRACT**  
Deep neural networks have demonstrated impressive accuracy in supervised learning tasks. However, their lack of transparency makes it hard for humans to trust their results, especially in safe-critic domains such as healthcare. To address this issue, recent explanation-guided learning approaches proposed to align the gradient-based attention map to image regions annotated by human experts, thereby obtaining an intrinsically human-centered model. However, the attention map these methods are based on may fail to causally attribute the model predictions, thus compromising their validity for alignment. To address this issue, we propose a novel human-centered framework based on counterfactual generation. In particular, we utilize the counterfactual generation's ability for causal attribution to introduce a novel loss called the CounterFactual Alignment (CF-Align) loss. This loss guarantees that the features attributed by the counterfactual generation for the classifier align with the human annotations. To optimize the proposed loss that entails a counterfactual generation with an implicit function form, we leverage the implicit function theorem for backpropagation. Our method is architecture-agnostic and, therefore can be applied to any neural network. We demonstrate the effectiveness of our method on a lung cancer diagnosis dataset, showcasing faithful alignment to humans.

{{</citation>}}


### (29/156) Blending Imitation and Reinforcement Learning for Robust Policy Improvement (Xuefeng Liu et al., 2023)

{{<citation>}}

Xuefeng Liu, Takuma Yoneda, Rick L. Stevens, Matthew R. Walter, Yuxin Chen. (2023)  
**Blending Imitation and Reinforcement Learning for Robust Policy Improvement**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.01737v2)  

---


**ABSTRACT**  
While reinforcement learning (RL) has shown promising performance, its sample complexity continues to be a substantial hurdle, restricting its broader application across a variety of domains. Imitation learning (IL) utilizes oracles to improve sample efficiency, yet it is often constrained by the quality of the oracles deployed. which actively interleaves between IL and RL based on an online estimate of their performance. RPI draws on the strengths of IL, using oracle queries to facilitate exploration, an aspect that is notably challenging in sparse-reward RL, particularly during the early stages of learning. As learning unfolds, RPI gradually transitions to RL, effectively treating the learned policy as an improved oracle. This algorithm is capable of learning from and improving upon a diverse set of black-box oracles. Integral to RPI are Robust Active Policy Selection (RAPS) and Robust Policy Gradient (RPG), both of which reason over whether to perform state-wise imitation from the oracles or learn from its own value function when the learner's performance surpasses that of the oracles in a specific state. Empirical evaluations and theoretical analysis validate that RPI excels in comparison to existing state-of-the-art methodologies, demonstrating superior performance across various benchmark domains.

{{</citation>}}


### (30/156) Time-LLM: Time Series Forecasting by Reprogramming Large Language Models (Ming Jin et al., 2023)

{{<citation>}}

Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, Qingsong Wen. (2023)  
**Time-LLM: Time Series Forecasting by Reprogramming Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Language Model, NLP, Time Series  
[Paper Link](http://arxiv.org/abs/2310.01728v1)  

---


**ABSTRACT**  
Time series forecasting holds significant importance in many real-world dynamic systems and has been extensively studied. Unlike natural language process (NLP) and computer vision (CV), where a single large model can tackle multiple tasks, models for time series forecasting are often specialized, necessitating distinct designs for different tasks and applications. While pre-trained foundation models have made impressive strides in NLP and CV, their development in time series domains has been constrained by data sparsity. Recent studies have revealed that large language models (LLMs) possess robust pattern recognition and reasoning abilities over complex sequences of tokens. However, the challenge remains in effectively aligning the modalities of time series data and natural language to leverage these capabilities. In this work, we present Time-LLM, a reprogramming framework to repurpose LLMs for general time series forecasting with the backbone language models kept intact. We begin by reprogramming the input time series with text prototypes before feeding it into the frozen LLM to align the two modalities. To augment the LLM's ability to reason with time series data, we propose Prompt-as-Prefix (PaP), which enriches the input context and directs the transformation of reprogrammed input patches. The transformed time series patches from the LLM are finally projected to obtain the forecasts. Our comprehensive evaluations demonstrate that Time-LLM is a powerful time series learner that outperforms state-of-the-art, specialized forecasting models. Moreover, Time-LLM excels in both few-shot and zero-shot learning scenarios.

{{</citation>}}


### (31/156) PrACTiS: Perceiver-Attentional Copulas for Time Series (Cat P. Le et al., 2023)

{{<citation>}}

Cat P. Le, Chris Cannella, Ali Hasan, Yuting Ng, Vahid Tarokh. (2023)  
**PrACTiS: Perceiver-Attentional Copulas for Time Series**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, Time Series, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.01720v1)  

---


**ABSTRACT**  
Transformers incorporating copula structures have demonstrated remarkable performance in time series prediction. However, their heavy reliance on self-attention mechanisms demands substantial computational resources, thus limiting their practical utility across a wide range of tasks. In this work, we present a model that combines the perceiver architecture with a copula structure to enhance time-series forecasting. By leveraging the perceiver as the encoder, we efficiently transform complex, high-dimensional, multimodal data into a compact latent space, thereby significantly reducing computational demands. To further reduce complexity, we introduce midpoint inference and local attention mechanisms, enabling the model to capture dependencies within imputed samples effectively. Subsequently, we deploy the copula-based attention and output variance testing mechanism to capture the joint distribution of missing data, while simultaneously mitigating error propagation during prediction. Our experimental results on the unimodal and multimodal benchmarks showcase a consistent 20\% improvement over the state-of-the-art methods, while utilizing less than half of available memory resources.

{{</citation>}}


### (32/156) Large Language Models as Analogical Reasoners (Michihiro Yasunaga et al., 2023)

{{<citation>}}

Michihiro Yasunaga, Xinyun Chen, Yujia Li, Panupong Pasupat, Jure Leskovec, Percy Liang, Ed H. Chi, Denny Zhou. (2023)  
**Large Language Models as Analogical Reasoners**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.01714v1)  

---


**ABSTRACT**  
Chain-of-thought (CoT) prompting for language models demonstrates impressive performance across reasoning tasks, but typically needs labeled exemplars of the reasoning process. In this work, we introduce a new prompting approach, Analogical Prompting, designed to automatically guide the reasoning process of large language models. Inspired by analogical reasoning, a cognitive process in which humans draw from relevant past experiences to tackle new problems, our approach prompts language models to self-generate relevant exemplars or knowledge in the context, before proceeding to solve the given problem. This method presents several advantages: it obviates the need for labeling or retrieving exemplars, offering generality and convenience; it can also tailor the generated exemplars and knowledge to each problem, offering adaptability. Experimental results show that our approach outperforms 0-shot CoT and manual few-shot CoT in a variety of reasoning tasks, including math problem solving in GSM8K and MATH, code generation in Codeforces, and other reasoning tasks in BIG-Bench.

{{</citation>}}


### (33/156) On Representation Complexity of Model-based and Model-free Reinforcement Learning (Hanlin Zhu et al., 2023)

{{<citation>}}

Hanlin Zhu, Baihe Huang, Stuart Russell. (2023)  
**On Representation Complexity of Model-based and Model-free Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.01706v1)  

---


**ABSTRACT**  
We study the representation complexity of model-based and model-free reinforcement learning (RL) in the context of circuit complexity. We prove theoretically that there exists a broad class of MDPs such that their underlying transition and reward functions can be represented by constant depth circuits with polynomial size, while the optimal $Q$-function suffers an exponential circuit complexity in constant-depth circuits. By drawing attention to the approximation errors and building connections to complexity theory, our theory provides unique insights into why model-based algorithms usually enjoy better sample complexity than model-free algorithms from a novel representation complexity perspective: in some cases, the ground-truth rule (model) of the environment is simple to represent, while other quantities, such as $Q$-function, appear complex. We empirically corroborate our theory by comparing the approximation error of the transition kernel, reward function, and optimal $Q$-function in various Mujoco environments, which demonstrates that the approximation errors of the transition kernel and reward function are consistently lower than those of the optimal $Q$-function. To the best of our knowledge, this work is the first to study the circuit complexity of RL, which also provides a rigorous framework for future research.

{{</citation>}}


## cs.CL (41)



### (34/156) ResidualTransformer: Residual Low-rank Learning with Weight-sharing for Transformer Layers (Yiming Wang et al., 2023)

{{<citation>}}

Yiming Wang, Jinyu Li. (2023)  
**ResidualTransformer: Residual Low-rank Learning with Weight-sharing for Transformer Layers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.02489v1)  

---


**ABSTRACT**  
Memory constraint of always-on devices is one of the major concerns when deploying speech processing models on these devices. While larger models trained with sufficiently large amount of data generally perform better, making them fit in the device memory is a demanding challenge. In this paper, we aim to reduce model size by reparameterizing model weights across Transformer encoder layers and assuming a special weight composition and structure. More specifically, inspired by ResNet and the more recent LoRA work, we propose an approach named ResidualTransformer, where each weight matrix in a Transformer layer comprises 1) a shared full-rank component with its adjacent layers, and 2) a unique low-rank component to itself. The low-rank matrices only account for a small amount of model size increase. In addition, we add diagonal weight matrices to improve modeling capacity of the low-rank matrices. Experiments of our 10k-hour speech recognition and speech translation tasks show that the Transformer encoder size can be reduced by ~3X with very slight performance degradation.

{{</citation>}}


### (35/156) Large Language Models Can Be Good Privacy Protection Learners (Yijia Xiao et al., 2023)

{{<citation>}}

Yijia Xiao, Yiqiao Jin, Yushi Bai, Yue Wu, Xianjun Yang, Xiao Luo, Wenchao Yu, Xujiang Zhao, Yanchi Liu, Haifeng Chen, Wei Wang, Wei Cheng. (2023)  
**Large Language Models Can Be Good Privacy Protection Learners**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02469v1)  

---


**ABSTRACT**  
The proliferation of Large Language Models (LLMs) has driven considerable interest in fine-tuning them with domain-specific data to create specialized language models. Nevertheless, such domain-specific fine-tuning data often contains sensitive personally identifiable information (PII). Direct fine-tuning LLMs on this data without privacy protection poses a risk of leakage. To address this challenge, we introduce Privacy Protection Language Models (PPLM), a novel paradigm for fine-tuning LLMs that effectively injects domain-specific knowledge while safeguarding data privacy. Our work offers a theoretical analysis for model design and delves into various techniques such as corpus curation, penalty-based unlikelihood in training loss, and instruction-based tuning, etc. Extensive experiments across diverse datasets and scenarios demonstrate the effectiveness of our approaches. In particular, instruction tuning with both positive and negative examples, stands out as a promising method, effectively protecting private data while enhancing the model's knowledge. Our work underscores the potential for Large Language Models as robust privacy protection learners.

{{</citation>}}


### (36/156) The Empty Signifier Problem: Towards Clearer Paradigms for Operationalising 'Alignment' in Large Language Models (Hannah Rose Kirk et al., 2023)

{{<citation>}}

Hannah Rose Kirk, Bertie Vidgen, Paul Röttger, Scott A. Hale. (2023)  
**The Empty Signifier Problem: Towards Clearer Paradigms for Operationalising 'Alignment' in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02457v1)  

---


**ABSTRACT**  
In this paper, we address the concept of "alignment" in large language models (LLMs) through the lens of post-structuralist socio-political theory, specifically examining its parallels to empty signifiers. To establish a shared vocabulary around how abstract concepts of alignment are operationalised in empirical datasets, we propose a framework that demarcates: 1) which dimensions of model behaviour are considered important, then 2) how meanings and definitions are ascribed to these dimensions, and by whom. We situate existing empirical literature and provide guidance on deciding which paradigm to follow. Through this framework, we aim to foster a culture of transparency and critical evaluation, aiding the community in navigating the complexities of aligning LLMs with human populations.

{{</citation>}}


### (37/156) Backdoor Adjustment of Confounding by Provenance for Robust Text Classification of Multi-institutional Clinical Notes (Xiruo Ding et al., 2023)

{{<citation>}}

Xiruo Ding, Zhecheng Sheng, Meliha Yetişgen, Serguei Pakhomov, Trevor Cohen. (2023)  
**Backdoor Adjustment of Confounding by Provenance for Robust Text Classification of Multi-institutional Clinical Notes**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Clinical, NLP, Natural Language Processing, Text Classification  
[Paper Link](http://arxiv.org/abs/2310.02451v1)  

---


**ABSTRACT**  
Natural Language Processing (NLP) methods have been broadly applied to clinical tasks. Machine learning and deep learning approaches have been used to improve the performance of clinical NLP. However, these approaches require sufficiently large datasets for training, and trained models have been shown to transfer poorly across sites. These issues have led to the promotion of data collection and integration across different institutions for accurate and portable models. However, this can introduce a form of bias called confounding by provenance. When source-specific data distributions differ at deployment, this may harm model performance. To address this issue, we evaluate the utility of backdoor adjustment for text classification in a multi-site dataset of clinical notes annotated for mentions of substance abuse. Using an evaluation framework devised to measure robustness to distributional shifts, we assess the utility of backdoor adjustment. Our results indicate that backdoor adjustment can effectively mitigate for confounding shift.

{{</citation>}}


### (38/156) Low-Resource Languages Jailbreak GPT-4 (Zheng-Xin Yong et al., 2023)

{{<citation>}}

Zheng-Xin Yong, Cristina Menghini, Stephen H. Bach. (2023)  
**Low-Resource Languages Jailbreak GPT-4**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: AI, GPT, GPT-4, Low-Resource  
[Paper Link](http://arxiv.org/abs/2310.02446v1)  

---


**ABSTRACT**  
AI safety training and red-teaming of large language models (LLMs) are measures to mitigate the generation of unsafe content. Our work exposes the inherent cross-lingual vulnerability of these safety mechanisms, resulting from the linguistic inequality of safety training data, by successfully circumventing GPT-4's safeguard through translating unsafe English inputs into low-resource languages. On the AdvBenchmark, GPT-4 engages with the unsafe translated inputs and provides actionable items that can get the users towards their harmful goals 79% of the time, which is on par with or even surpassing state-of-the-art jailbreaking attacks. Other high-/mid-resource languages have significantly lower attack success rate, which suggests that the cross-lingual vulnerability mainly applies to low-resource languages. Previously, limited training on low-resource languages primarily affects speakers of those languages, causing technological disparities. However, our work highlights a crucial shift: this deficiency now poses a risk to all LLMs users. Publicly available translation APIs enable anyone to exploit LLMs' safety vulnerabilities. Therefore, our work calls for a more holistic red-teaming efforts to develop robust multilingual safeguards with wide language coverage.

{{</citation>}}


### (39/156) Novice Learner and Expert Tutor: Evaluating Math Reasoning Abilities of Large Language Models with Misconceptions (Naiming Liu et al., 2023)

{{<citation>}}

Naiming Liu, Shashank Sonkar, Zichao Wang, Simon Woodhead, Richard G. Baraniuk. (2023)  
**Novice Learner and Expert Tutor: Evaluating Math Reasoning Abilities of Large Language Models with Misconceptions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.02439v1)  

---


**ABSTRACT**  
We propose novel evaluations for mathematical reasoning capabilities of Large Language Models (LLMs) based on mathematical misconceptions. Our primary approach is to simulate LLMs as a novice learner and an expert tutor, aiming to identify the incorrect answer to math question resulted from a specific misconception and to recognize the misconception(s) behind an incorrect answer, respectively. Contrary to traditional LLMs-based mathematical evaluations that focus on answering math questions correctly, our approach takes inspirations from principles in educational learning sciences. We explicitly ask LLMs to mimic a novice learner by answering questions in a specific incorrect manner based on incomplete knowledge; and to mimic an expert tutor by identifying misconception(s) corresponding to an incorrect answer to a question. Using simple grade-school math problems, our experiments reveal that, while LLMs can easily answer these questions correctly, they struggle to identify 1) the incorrect answer corresponding to specific incomplete knowledge (misconceptions); 2) the misconceptions that explain particular incorrect answers. Our study indicates new opportunities for enhancing LLMs' math reasoning capabilities, especially on developing robust student simulation and expert tutoring models in the educational applications such as intelligent tutoring systems.

{{</citation>}}


### (40/156) Nugget 2D: Dynamic Contextual Compression for Scaling Decoder-only Language Models (Guanghui Qin et al., 2023)

{{<citation>}}

Guanghui Qin, Corby Rosset, Ethan C. Chau, Nikhil Rao, Benjamin Van Durme. (2023)  
**Nugget 2D: Dynamic Contextual Compression for Scaling Decoder-only Language Models**  

---
Primary Category: cs.CL  
Categories: I-2-7; I-2-6, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, BLEU, LLaMA, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.02409v1)  

---


**ABSTRACT**  
Standard Transformer-based language models (LMs) scale poorly to long contexts. We propose a solution based on dynamic contextual compression, which extends the Nugget approach of Qin & Van Durme (2023) from BERT-like frameworks to decoder-only LMs. Our method models history as compressed "nuggets" which are trained to allow for reconstruction, and it can be initialized with off-the-shelf models such as LLaMA. We demonstrate through experiments in language modeling, question answering, and summarization that Nugget2D retains capabilities in these tasks, while drastically reducing the overhead during decoding in terms of time and space. For example, in the experiments of autoencoding, Nugget2D can shrink context at a 20x compression ratio with a BLEU score of 98% for reconstruction, achieving nearly lossless encoding.

{{</citation>}}


### (41/156) Unsupervised Speech Recognition with N-Skipgram and Positional Unigram Matching (Liming Wang et al., 2023)

{{<citation>}}

Liming Wang, Mark Hasegawa-Johnson, Chang D. Yoo. (2023)  
**Unsupervised Speech Recognition with N-Skipgram and Positional Unigram Matching**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.02382v1)  

---


**ABSTRACT**  
Training unsupervised speech recognition systems presents challenges due to GAN-associated instability, misalignment between speech and text, and significant memory demands. To tackle these challenges, we introduce a novel ASR system, ESPUM. This system harnesses the power of lower-order N-skipgrams (up to N=3) combined with positional unigram statistics gathered from a small batch of samples. Evaluated on the TIMIT benchmark, our model showcases competitive performance in ASR and phoneme segmentation tasks. Access our publicly available code at https://github.com/lwang114/GraphUnsupASR.

{{</citation>}}


### (42/156) Conversational Health Agents: A Personalized LLM-Powered Agent Framework (Mahyar Abbasian et al., 2023)

{{<citation>}}

Mahyar Abbasian, Iman Azimi, Amir M. Rahmani, Ramesh Jain. (2023)  
**Conversational Health Agents: A Personalized LLM-Powered Agent Framework**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02374v1)  

---


**ABSTRACT**  
Conversational Health Agents (CHAs) are interactive systems designed to enhance personal healthcare services by engaging in empathetic conversations and processing multimodal data. While current CHAs, especially those utilizing Large Language Models (LLMs), primarily focus on conversation, they often lack comprehensive agent capabilities. This includes the ability to access personal user health data from wearables, 24/7 data collection sources, and electronic health records, as well as integrating the latest published health insights and connecting with established multimodal data analysis tools. We are developing a framework to empower CHAs by equipping them with critical thinking, knowledge acquisition, and problem-solving abilities. Our CHA platform, powered by LLMs, seamlessly integrates healthcare tools, enables multilingual and multimodal conversations, and interfaces with a variety of user data analysis tools. We illustrate its proficiency in handling complex healthcare tasks, such as stress level estimation, showcasing the agent's cognitive and operational capabilities.

{{</citation>}}


### (43/156) ProtoNER: Few shot Incremental Learning for Named Entity Recognition using Prototypical Networks (Ritesh Kumar et al., 2023)

{{<citation>}}

Ritesh Kumar, Saurabh Goyal, Ashish Verma, Vatche Isahagian. (2023)  
**ProtoNER: Few shot Incremental Learning for Named Entity Recognition using Prototypical Networks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2310.02372v1)  

---


**ABSTRACT**  
Key value pair (KVP) extraction or Named Entity Recognition(NER) from visually rich documents has been an active area of research in document understanding and data extraction domain. Several transformer based models such as LayoutLMv2, LayoutLMv3, and LiLT have emerged achieving state of the art results. However, addition of even a single new class to the existing model requires (a) re-annotation of entire training dataset to include this new class and (b) retraining the model again. Both of these issues really slow down the deployment of updated model. \\ We present \textbf{ProtoNER}: Prototypical Network based end-to-end KVP extraction model that allows addition of new classes to an existing model while requiring minimal number of newly annotated training samples. The key contributions of our model are: (1) No dependency on dataset used for initial training of the model, which alleviates the need to retain original training dataset for longer duration as well as data re-annotation which is very time consuming task, (2) No intermediate synthetic data generation which tends to add noise and results in model's performance degradation, and (3) Hybrid loss function which allows model to retain knowledge about older classes as well as learn about newly added classes.\\ Experimental results show that ProtoNER finetuned with just 30 samples is able to achieve similar results for the newly added classes as that of regular model finetuned with 2600 samples.

{{</citation>}}


### (44/156) On the definition of toxicity in NLP (Sergey Berezin et al., 2023)

{{<citation>}}

Sergey Berezin, Reza Farahbakhsh, Noel Crespi. (2023)  
**On the definition of toxicity in NLP**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.02357v2)  

---


**ABSTRACT**  
The fundamental problem in toxicity detection task lies in the fact that the toxicity is ill-defined. This causes us to rely on subjective and vague data in models' training, which results in non-robust and non-accurate results: garbage in - garbage out.   This work suggests a new, stress-level-based definition of toxicity designed to be objective and context-aware. On par with it, we also describe possible ways of applying this new definition to dataset creation and model training.

{{</citation>}}


### (45/156) Contrastive Post-training Large Language Models on Data Curriculum (Canwen Xu et al., 2023)

{{<citation>}}

Canwen Xu, Corby Rosset, Luciano Del Corro, Shweti Mahajan, Julian McAuley, Jennifer Neville, Ahmed Hassan Awadallah, Nikhil Rao. (2023)  
**Contrastive Post-training Large Language Models on Data Curriculum**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.02263v1)  

---


**ABSTRACT**  
Alignment serves as an important step to steer large language models (LLMs) towards human preferences. In this paper, we explore contrastive post-training techniques for alignment by automatically constructing preference pairs from multiple models of varying strengths (e.g., InstructGPT, ChatGPT and GPT-4). We carefully compare the contrastive techniques of SLiC and DPO to SFT baselines and find that DPO provides a step-function improvement even after continueing SFT saturates. We also explore a data curriculum learning scheme for contrastive post-training, which starts by learning from "easier" pairs and transitioning to "harder" ones, which further improves alignment. Finally, we scale up our experiments to train with more data and larger models like Orca. Remarkably, contrastive post-training further improves the performance of Orca, already a state-of-the-art instruction learning model tuned with GPT-4 outputs, to exceed that of ChatGPT.

{{</citation>}}


### (46/156) Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation (Eric Zelikman et al., 2023)

{{<citation>}}

Eric Zelikman, Eliana Lorch, Lester Mackey, Adam Tauman Kalai. (2023)  
**Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL, stat-ML  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.02304v1)  

---


**ABSTRACT**  
Several recent advances in AI systems (e.g., Tree-of-Thoughts and Program-Aided Language Models) solve problems by providing a "scaffolding" program that structures multiple calls to language models to generate better outputs. A scaffolding program is written in a programming language such as Python. In this work, we use a language-model-infused scaffolding program to improve itself. We start with a seed "improver" that improves an input program according to a given utility function by querying a language model several times and returning the best solution. We then run this seed improver to improve itself. Across a small set of downstream tasks, the resulting improved improver generates programs with significantly better performance than its seed improver. Afterward, we analyze the variety of self-improvement strategies proposed by the language model, including beam search, genetic algorithms, and simulated annealing. Since the language models themselves are not altered, this is not full recursive self-improvement. Nonetheless, it demonstrates that a modern language model, GPT-4 in our proof-of-concept experiments, is capable of writing code that can call itself to improve itself. We critically consider concerns around the development of self-improving technologies and evaluate the frequency with which the generated code bypasses a sandbox.

{{</citation>}}


### (47/156) Harnessing Pre-Trained Sentence Transformers for Offensive Language Detection in Indian Languages (Ananya Joshi et al., 2023)

{{<citation>}}

Ananya Joshi, Raviraj Joshi. (2023)  
**Harnessing Pre-Trained Sentence Transformers for Offensive Language Detection in Indian Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.02249v1)  

---


**ABSTRACT**  
In our increasingly interconnected digital world, social media platforms have emerged as powerful channels for the dissemination of hate speech and offensive content. This work delves into the domain of hate speech detection, placing specific emphasis on three low-resource Indian languages: Bengali, Assamese, and Gujarati. The challenge is framed as a text classification task, aimed at discerning whether a tweet contains offensive or non-offensive content. Leveraging the HASOC 2023 datasets, we fine-tuned pre-trained BERT and SBERT models to evaluate their effectiveness in identifying hate speech. Our findings underscore the superiority of monolingual sentence-BERT models, particularly in the Bengali language, where we achieved the highest ranking. However, the performance in Assamese and Gujarati languages signifies ongoing opportunities for enhancement. Our goal is to foster inclusive online spaces by countering hate speech proliferation.

{{</citation>}}


### (48/156) Extraction of Medication and Temporal Relation from Clinical Text by Harnessing Different Deep Learning Models (Hangyu Tu et al., 2023)

{{<citation>}}

Hangyu Tu, Lifeng Han, Goran Nenadic. (2023)  
**Extraction of Medication and Temporal Relation from Clinical Text by Harnessing Different Deep Learning Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Clinical, LSTM, NER  
[Paper Link](http://arxiv.org/abs/2310.02229v1)  

---


**ABSTRACT**  
Clinical texts, represented in electronic medical records (EMRs), contain rich medical information and are essential for disease prediction, personalised information recommendation, clinical decision support, and medication pattern mining and measurement. Relation extractions between medication mentions and temporal information can further help clinicians better understand the patients' treatment history. To evaluate the performances of deep learning (DL) and large language models (LLMs) in medication extraction and temporal relations classification, we carry out an empirical investigation of \textbf{MedTem} project using several advanced learning structures including BiLSTM-CRF and CNN-BiLSTM for a clinical domain named entity recognition (NER), and BERT-CNN for temporal relation extraction (RE), in addition to the exploration of different word embedding techniques. Furthermore, we also designed a set of post-processing roles to generate structured output on medications and the temporal relation. Our experiments show that CNN-BiLSTM slightly wins the BiLSTM-CRF model on the i2b2-2009 clinical NER task yielding 75.67, 77.83, and 78.17 for precision, recall, and F1 scores using Macro Average. BERT-CNN model also produced reasonable evaluation scores 64.48, 67.17, and 65.03 for P/R/F1 using Macro Avg on the temporal relation extraction test set from i2b2-2012 challenges. Code and Tools from MedTem will be hosted at \url{https://github.com/HECTA-UoM/MedTem}

{{</citation>}}


### (49/156) Think before you speak: Training Language Models With Pause Tokens (Sachin Goyal et al., 2023)

{{<citation>}}

Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, Vaishnavh Nagarajan. (2023)  
**Think before you speak: Training Language Models With Pause Tokens**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.02226v1)  

---


**ABSTRACT**  
Language models generate responses by producing a series of tokens in immediate succession: the $(K+1)^{th}$ token is an outcome of manipulating $K$ hidden vectors per layer, one vector per preceding token. What if instead we were to let the model manipulate say, $K+10$ hidden vectors, before it outputs the $(K+1)^{th}$ token? We operationalize this idea by performing training and inference on language models with a (learnable) $\textit{pause}$ token, a sequence of which is appended to the input prefix. We then delay extracting the model's outputs until the last pause token is seen, thereby allowing the model to process extra computation before committing to an answer. We empirically evaluate $\textit{pause-training}$ on decoder-only models of 1B and 130M parameters with causal pretraining on C4, and on downstream tasks covering reasoning, question-answering, general understanding and fact recall. Our main finding is that inference-time delays show gains when the model is both pre-trained and finetuned with delays. For the 1B model, we witness gains on 8 of 9 tasks, most prominently, a gain of $18\%$ EM score on the QA task of SQuAD, $8\%$ on CommonSenseQA and $1\%$ accuracy on the reasoning task of GSM8k. Our work raises a range of conceptual and practical future research questions on making delayed next-token prediction a widely applicable new paradigm.

{{</citation>}}


### (50/156) Can Language Models be Instructed to Protect Personal Information? (Yang Chen et al., 2023)

{{<citation>}}

Yang Chen, Ethan Mendes, Sauvik Das, Wei Xu, Alan Ritter. (2023)  
**Can Language Models be Instructed to Protect Personal Information?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.02224v1)  

---


**ABSTRACT**  
Large multimodal language models have proven transformative in numerous applications. However, these models have been shown to memorize and leak pre-training data, raising serious user privacy and information security concerns. While data leaks should be prevented, it is also crucial to examine the trade-off between the privacy protection and model utility of proposed approaches. In this paper, we introduce PrivQA -- a multimodal benchmark to assess this privacy/utility trade-off when a model is instructed to protect specific categories of personal information in a simulated scenario. We also propose a technique to iteratively self-moderate responses, which significantly improves privacy. However, through a series of red-teaming experiments, we find that adversaries can also easily circumvent these protections with simple jailbreaking methods through textual and/or image inputs. We believe PrivQA has the potential to support the development of new models with improved privacy protections, as well as the adversarial robustness of these protections. We release the entire PrivQA dataset at https://llm-access-control.github.io/.

{{</citation>}}


### (51/156) Ask Again, Then Fail: Large Language Models' Vacillations in Judgement (Qiming Xie et al., 2023)

{{<citation>}}

Qiming Xie, Zengzhi Wang, Yi Feng, Rui Xia. (2023)  
**Ask Again, Then Fail: Large Language Models' Vacillations in Judgement**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2310.02174v1)  

---


**ABSTRACT**  
With the emergence of generative conversational large language models (LLMs) like ChatGPT, serving as virtual assistants in various fields, the stability and reliability of their responses have become crucial. However, during usage, it has been observed that these models tend to waver in their judgements when confronted with follow-up questions from users expressing skepticism or disagreement. In this work, we draw inspiration from questioning strategies in education and propose a \textsc{Follow-up Questioning Mechanism} along with two evaluation metrics to assess the judgement consistency of LLMs before and after exposure to disturbances. We evaluate the judgement consistency of ChatGPT, PaLM2-Bison, and Vicuna-13B under this mechanism across eight reasoning benchmarks. Empirical results show that even when the initial answers are correct, judgement consistency sharply decreases when LLMs face disturbances such as questioning, negation, or misleading. Additionally, we study these models' judgement consistency under various settings (sampling temperature and prompts) to validate this issue further, observing the impact of prompt tone and conducting an in-depth error analysis for deeper behavioral insights. Furthermore, we also explore several prompting methods to mitigate this issue and demonstrate their effectiveness\footnote{\url{https://github.com/NUSTM/LLMs-Waver-In-Judgements}}.

{{</citation>}}


### (52/156) Dynamic LLM-Agent Network: An LLM-agent Collaboration Framework with Agent Team Optimization (Zijun Liu et al., 2023)

{{<citation>}}

Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, Diyi Yang. (2023)  
**Dynamic LLM-Agent Network: An LLM-agent Collaboration Framework with Agent Team Optimization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-MA, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.02170v1)  

---


**ABSTRACT**  
Large language model (LLM) agents have been shown effective on a wide range of tasks, and by ensembling multiple LLM agents, their performances could be further improved. Existing approaches employ a fixed set of agents to interact with each other in a static architecture, which limits their generalizability to various tasks and requires strong human prior in designing these agents. In this work, we propose to construct a strategic team of agents communicating in a dynamic interaction architecture based on the task query. Specifically, we build a framework named Dynamic LLM-Agent Network ($\textbf{DyLAN}$) for LLM-agent collaboration on complicated tasks like reasoning and code generation. DyLAN enables agents to interact for multiple rounds in a dynamic architecture with inference-time agent selection and an early-stopping mechanism to improve performance and efficiency. We further design an automatic agent team optimization algorithm based on an unsupervised metric termed $\textit{Agent Importance Score}$, enabling the selection of best agents based on the contribution each agent makes. Empirically, we demonstrate that DyLAN performs well in both reasoning and code generation tasks with reasonable computational cost. DyLAN achieves 13.0% and 13.3% improvement on MATH and HumanEval, respectively, compared to a single execution on GPT-35-turbo. On specific subjects of MMLU, agent team optimization in DyLAN increases accuracy by up to 25.0%.

{{</citation>}}


### (53/156) Editing Personality for LLMs (Shengyu Mao et al., 2023)

{{<citation>}}

Shengyu Mao, Ningyu Zhang, Xiaohan Wang, Mengru Wang, Yunzhi Yao, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen. (2023)  
**Editing Personality for LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs-MA, cs.CL  
Keywords: GPT, GPT-4, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.02168v1)  

---


**ABSTRACT**  
This paper introduces an innovative task focused on editing the personality traits of Large Language Models (LLMs). This task seeks to adjust the models' responses to opinion-related questions on specified topics since an individual's personality often manifests in the form of their expressed opinions, thereby showcasing different personality traits. Specifically, we construct a new benchmark dataset PersonalityEdit to address this task. Drawing on the theory in Social Psychology, we isolate three representative traits, namely Neuroticism, Extraversion, and Agreeableness, as the foundation for our benchmark. We then gather data using GPT-4, generating responses that not only align with a specified topic but also embody the targeted personality trait. We conduct comprehensive experiments involving various baselines and discuss the representation of personality behavior in LLMs. Our intriguing findings uncover potential challenges of the proposed task, illustrating several remaining issues. We anticipate that our work can provide the NLP community with insights. Code and datasets will be released at https://github.com/zjunlp/EasyEdit.

{{</citation>}}


### (54/156) Large Language Models Meet Knowledge Graphs to Answer Factoid Questions (Mikhail Salnikov et al., 2023)

{{<citation>}}

Mikhail Salnikov, Hai Le, Prateek Rajput, Irina Nikishina, Pavel Braslavski, Valentin Malykh, Alexander Panchenko. (2023)  
**Large Language Models Meet Knowledge Graphs to Answer Factoid Questions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2310.02166v1)  

---


**ABSTRACT**  
Recently, it has been shown that the incorporation of structured knowledge into Large Language Models significantly improves the results for a variety of NLP tasks. In this paper, we propose a method for exploring pre-trained Text-to-Text Language Models enriched with additional information from Knowledge Graphs for answering factoid questions. More specifically, we propose an algorithm for subgraphs extraction from a Knowledge Graph based on question entities and answer candidates. Then, we procure easily interpreted information with Transformer-based models through the linearization of the extracted subgraphs. Final re-ranking of the answer candidates with the extracted information boosts Hits@1 scores of the pre-trained text-to-text language models by 4-6%.

{{</citation>}}


### (55/156) Unveiling the Pitfalls of Knowledge Editing for Large Language Models (Zhoubo Li et al., 2023)

{{<citation>}}

Zhoubo Li, Ningyu Zhang, Yunzhi Yao, Mengru Wang, Xi Chen, Huajun Chen. (2023)  
**Unveiling the Pitfalls of Knowledge Editing for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-DB, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02129v1)  

---


**ABSTRACT**  
As the cost associated with fine-tuning Large Language Models (LLMs) continues to rise, recent research efforts have pivoted towards developing methodologies to edit implicit knowledge embedded within LLMs. Yet, there's still a dark cloud lingering overhead -- will knowledge editing trigger butterfly effect? since it is still unclear whether knowledge editing might introduce side effects that pose potential risks or not. This paper pioneers the investigation into the potential pitfalls associated with knowledge editing for LLMs. To achieve this, we introduce new benchmark datasets and propose innovative evaluation metrics. Our results underline two pivotal concerns: (1) Knowledge Conflict: Editing groups of facts that logically clash can magnify the inherent inconsistencies in LLMs-a facet neglected by previous methods. (2) Knowledge Distortion: Altering parameters with the aim of editing factual knowledge can irrevocably warp the innate knowledge structure of LLMs. Experimental results vividly demonstrate that knowledge editing might inadvertently cast a shadow of unintended consequences on LLMs, which warrant attention and efforts for future works. Code will be released at https://github.com/zjunlp/PitfallsKnowledgeEditing.

{{</citation>}}


### (56/156) Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View (Jintian Zhang et al., 2023)

{{<citation>}}

Jintian Zhang, Xin Xu, Shumin Deng. (2023)  
**Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs-MA, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.02124v1)  

---


**ABSTRACT**  
As Natural Language Processing (NLP) systems are increasingly employed in intricate social environments, a pressing query emerges: Can these NLP systems mirror human-esque collaborative intelligence, in a multi-agent society consisting of multiple large language models (LLMs)? This paper probes the collaboration mechanisms among contemporary NLP systems by melding practical experiments with theoretical insights. We fabricate four unique `societies' comprised of LLM agents, where each agent is characterized by a specific `trait' (easy-going or overconfident) and engages in collaboration with a distinct `thinking pattern' (debate or reflection). Evaluating these multi-agent societies on three benchmark datasets, we discern that LLM agents navigate tasks by leveraging diverse social behaviors, from active debates to introspective reflections. Notably, certain collaborative strategies only optimize efficiency (using fewer API tokens), but also outshine previous top-tier approaches. Moreover, our results further illustrate that LLM agents manifest human-like social behaviors, such as conformity or majority rule, mirroring foundational Social Psychology theories. In conclusion, we integrate insights from Social Psychology to contextualize the collaboration of LLM agents, inspiring further investigations into the collaboration mechanism for LLMs. We commit to sharing our code and datasets (already submitted in supplementary materials), hoping to catalyze further research in this promising avenue (All code and data are available at \url{https://github.com/zjunlp/MachineSoM}.).

{{</citation>}}


### (57/156) TWIZ: The Wizard of Multimodal Conversational-Stimulus (Rafael Ferreira et al., 2023)

{{<citation>}}

Rafael Ferreira, Diogo Tavares, Diogo Silva, Rodrigo Valério, João Bordalo, Inês Simões, Vasco Ramos, David Semedo, João Magalhães. (2023)  
**TWIZ: The Wizard of Multimodal Conversational-Stimulus**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02118v1)  

---


**ABSTRACT**  
In this report, we describe the vision, challenges, and scientific contributions of the Task Wizard team, TWIZ, in the Alexa Prize TaskBot Challenge 2022. Our vision, is to build TWIZ bot as an helpful, multimodal, knowledgeable, and engaging assistant that can guide users towards the successful completion of complex manual tasks. To achieve this, we focus our efforts on three main research questions: (1) Humanly-Shaped Conversations, by providing information in a knowledgeable way; (2) Multimodal Stimulus, making use of various modalities including voice, images, and videos; and (3) Zero-shot Conversational Flows, to improve the robustness of the interaction to unseen scenarios. TWIZ is an assistant capable of supporting a wide range of tasks, with several innovative features such as creative cooking, video navigation through voice, and the robust TWIZ-LLM, a Large Language Model trained for dialoguing about complex manual tasks. Given ratings and feedback provided by users, we observed that TWIZ bot is an effective and robust system, capable of guiding users through tasks while providing several multimodal stimuli.

{{</citation>}}


### (58/156) Instance Needs More Care: Rewriting Prompts for Instances Yields Better Zero-Shot Performance (Saurabh Srivastava et al., 2023)

{{<citation>}}

Saurabh Srivastava, Chengyue Huang, Weiguo Fan, Ziyu Yao. (2023)  
**Instance Needs More Care: Rewriting Prompts for Instances Yields Better Zero-Shot Performance**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.02107v2)  

---


**ABSTRACT**  
Enabling large language models (LLMs) to perform tasks in zero-shot has been an appealing goal owing to its labor-saving (i.e., requiring no task-specific annotations); as such, zero-shot prompting approaches also enjoy better task generalizability. To improve LLMs' zero-shot performance, prior work has focused on devising more effective task instructions (e.g., ``let's think step by step'' ). However, we argue that, in order for an LLM to solve them correctly in zero-shot, individual test instances need more carefully designed and customized instructions. To this end, we propose PRoMPTd, an approach that rewrites the task prompt for each individual test input to be more specific, unambiguous, and complete, so as to provide better guidance to the task LLM. We evaluated PRoMPTd on eight datasets covering tasks including arithmetics, logical reasoning, and code generation, using GPT-4 as the task LLM. Notably, PRoMPTd achieves an absolute improvement of around 10% on the complex MATH dataset and 5% on the code generation task on HumanEval, outperforming conventional zero-shot methods. In addition, we also showed that the rewritten prompt can provide better interpretability of how the LLM resolves each test instance, which can potentially be leveraged as a defense mechanism against adversarial prompting. The source code and dataset can be obtained from https://github.com/salokr/PRoMPTd

{{</citation>}}


### (59/156) Controlling Topic-Focus Articulation in Meaning-to-Text Generation using Graph Neural Networks (Chunliu Wang et al., 2023)

{{<citation>}}

Chunliu Wang, Rik van Noord, Johan Bos. (2023)  
**Controlling Topic-Focus Articulation in Meaning-to-Text Generation using Graph Neural Networks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Graph Neural Network, Graph Neural Networks, Text Generation  
[Paper Link](http://arxiv.org/abs/2310.02053v1)  

---


**ABSTRACT**  
A bare meaning representation can be expressed in various ways using natural language, depending on how the information is structured on the surface level. We are interested in finding ways to control topic-focus articulation when generating text from meaning. We focus on distinguishing active and passive voice for sentences with transitive verbs. The idea is to add pragmatic information such as topic to the meaning representation, thereby forcing either active or passive voice when given to a natural language generation system. We use graph neural models because there is no explicit information about word order in a meaning represented by a graph. We try three different methods for topic-focus articulation (TFA) employing graph neural models for a meaning-to-text generation task. We propose a novel encoding strategy about node aggregation in graph neural models, which instead of traditional encoding by aggregating adjacent node information, learns node representations by using depth-first search. The results show our approach can get competitive performance with state-of-art graph models on general text generation, and lead to significant improvements on the task of active-passive conversion compared to traditional adjacency-based aggregation strategies. Different types of TFA can have a huge impact on the performance of the graph models.

{{</citation>}}


### (60/156) Tuning Large language model for End-to-end Speech Translation (Hao Zhang et al., 2023)

{{<citation>}}

Hao Zhang, Nianwen Si, Yaqi Chen, Wenlin Zhang, Xukui Yang, Dan Qu, Xiaolin Jiao. (2023)  
**Tuning Large language model for End-to-end Speech Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: BLEU, GPT  
[Paper Link](http://arxiv.org/abs/2310.02050v1)  

---


**ABSTRACT**  
With the emergence of large language models (LLMs), multimodal models based on LLMs have demonstrated significant potential. Models such as LLaSM, X-LLM, and SpeechGPT exhibit an impressive ability to comprehend and generate human instructions. However, their performance often falters when faced with complex tasks like end-to-end speech translation (E2E-ST), a cross-language and cross-modal translation task. In comparison to single-modal models, multimodal models lag behind in these scenarios. This paper introduces LST, a Large multimodal model designed to excel at the E2E-ST task. LST consists of a speech frontend, an adapter, and a LLM backend. The training of LST consists of two stages: (1) Modality adjustment, where the adapter is tuned to align speech representation with text embedding space, and (2) Downstream task fine-tuning, where both the adapter and LLM model are trained to optimize performance on the E2EST task. Experimental results on the MuST-C speech translation benchmark demonstrate that LST-13B achieves BLEU scores of 30.39/41.55/35.33 on En-De/En-Fr/En-Es language pairs, surpassing previous models and establishing a new state-of-the-art. Additionally, we conduct an in-depth analysis of single-modal model selection and the impact of training strategies, which lays the foundation for future research. We will open up our code and models after review.

{{</citation>}}


### (61/156) Jury: A Comprehensive Evaluation Toolkit (Devrim Cavusoglu et al., 2023)

{{<citation>}}

Devrim Cavusoglu, Ulas Sert, Secil Sen, Sinan Altinuc. (2023)  
**Jury: A Comprehensive Evaluation Toolkit**  

---
Primary Category: cs.CL  
Categories: I-2-7; D-1-3, cs-AI, cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.02040v1)  

---


**ABSTRACT**  
Evaluation plays a critical role in deep learning as a fundamental block of any prediction-based system. However, the vast number of Natural Language Processing (NLP) tasks and the development of various metrics have led to challenges in evaluating different systems with different metrics. To address these challenges, we introduce jury, a toolkit that provides a unified evaluation framework with standardized structures for performing evaluation across different tasks and metrics. The objective of jury is to standardize and improve metric evaluation for all systems and aid the community in overcoming the challenges in evaluation. Since its open-source release, jury has reached a wide audience and is available at https://github.com/obss/jury.

{{</citation>}}


### (62/156) OceanGPT: A Large Language Model for Ocean Science Tasks (Zhen Bi et al., 2023)

{{<citation>}}

Zhen Bi, Ningyu Zhang, Yida Xue, Yixin Ou, Daxiong Ji, Guozhou Zheng, Huajun Chen. (2023)  
**OceanGPT: A Large Language Model for Ocean Science Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CE, cs-CL, cs-LG, cs-RO, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.02031v2)  

---


**ABSTRACT**  
Ocean science, which delves into the oceans that are reservoirs of life and biodiversity, is of great significance given that oceans cover over 70% of our planet's surface. Recently, advances in Large Language Models (LLMs) have transformed the paradigm in science. Despite the success in other domains, current LLMs often fall short in catering to the needs of domain experts like oceanographers, and the potential of LLMs for ocean science is under-explored. The intrinsic reason may be the immense and intricate nature of ocean data as well as the necessity for higher granularity and richness in knowledge. To alleviate these issues, we introduce OceanGPT, the first-ever LLM in the ocean domain, which is expert in various ocean science tasks. We propose DoInstruct, a novel framework to automatically obtain a large volume of ocean domain instruction data, which generates instructions based on multi-agent collaboration. Additionally, we construct the first oceanography benchmark, OceanBench, to evaluate the capabilities of LLMs in the ocean domain. Though comprehensive experiments, OceanGPT not only shows a higher level of knowledge expertise for oceans science tasks but also gains preliminary embodied intelligence capabilities in ocean technology. Codes, data and checkpoints will soon be available at https://github.com/zjunlp/KnowLM.

{{</citation>}}


### (63/156) Fill in the Blank: Exploring and Enhancing LLM Capabilities for Backward Reasoning in Math Word Problems (Aniruddha Deb et al., 2023)

{{<citation>}}

Aniruddha Deb, Neeva Oza, Sarthak Singla, Dinesh Khandelwal, Dinesh Garg, Parag Singla. (2023)  
**Fill in the Blank: Exploring and Enhancing LLM Capabilities for Backward Reasoning in Math Word Problems**  

---
Primary Category: cs.CL  
Categories: I-2-3, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, PaLM, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.01991v1)  

---


**ABSTRACT**  
While forward reasoning (i.e. find the answer given the question) has been explored extensively in the recent literature, backward reasoning is relatively unexplored. We examine the backward reasoning capabilities of LLMs on Math Word Problems (MWPs): given a mathematical question and its answer, with some details omitted from the question, can LLMs effectively retrieve the missing information?   In this paper, we formally define the backward reasoning task on math word problems and modify three datasets to evaluate this task: GSM8k, SVAMP and MultiArith. Our findings show a significant drop in the accuracy of models on backward reasoning compared to forward reasoning across four SOTA LLMs (GPT4, GPT3.5, PaLM-2, and LLaMa-2). Utilizing the specific format of this task, we propose three novel techniques that improve performance: Rephrase reformulates the given problem into a forward reasoning problem, PAL-Tools combines the idea of Program-Aided LLMs to produce a set of equations that can be solved by an external solver, and Check your Work exploits the availability of natural verifier of high accuracy in the forward direction, interleaving solving and verification steps. Finally, realizing that each of our base methods correctly solves a different set of problems, we propose a novel Bayesian formulation for creating an ensemble over these base methods aided by a verifier to further boost the accuracy by a significant margin. Extensive experimentation demonstrates that our techniques successively improve the performance of LLMs on the backward reasoning task, with the final ensemble-based method resulting in a substantial performance gain compared to the raw LLMs with standard prompting techniques such as chain-of-thought.

{{</citation>}}


### (64/156) Language Models as Knowledge Bases for Visual Word Sense Disambiguation (Anastasia Kritharoula et al., 2023)

{{<citation>}}

Anastasia Kritharoula, Maria Lymperaiou, Giorgos Stamou. (2023)  
**Language Models as Knowledge Bases for Visual Word Sense Disambiguation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA, Word Sense Disambiguation  
[Paper Link](http://arxiv.org/abs/2310.01960v1)  

---


**ABSTRACT**  
Visual Word Sense Disambiguation (VWSD) is a novel challenging task that lies between linguistic sense disambiguation and fine-grained multimodal retrieval. The recent advancements in the development of visiolinguistic (VL) transformers suggest some off-the-self implementations with encouraging results, which however we argue that can be further improved. To this end, we propose some knowledge-enhancement techniques towards improving the retrieval performance of VL transformers via the usage of Large Language Models (LLMs) as Knowledge Bases. More specifically, knowledge stored in LLMs is retrieved with the help of appropriate prompts in a zero-shot manner, achieving performance advancements. Moreover, we convert VWSD to a purely textual question-answering (QA) problem by considering generated image captions as multiple-choice candidate answers. Zero-shot and few-shot prompting strategies are leveraged to explore the potential of such a transformation, while Chain-of-Thought (CoT) prompting in the zero-shot setting is able to reveal the internal reasoning steps an LLM follows to select the appropriate candidate. In total, our presented approach is the first one to analyze the merits of exploiting knowledge stored in LLMs in different ways to solve WVSD.

{{</citation>}}


### (65/156) Navigating Cultural Chasms: Exploring and Unlocking the Cultural POV of Text-To-Image Models (Mor Ventura et al., 2023)

{{<citation>}}

Mor Ventura, Eyal Ben-David, Anna Korhonen, Roi Reichart. (2023)  
**Navigating Cultural Chasms: Exploring and Unlocking the Cultural POV of Text-To-Image Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.01929v1)  

---


**ABSTRACT**  
Text-To-Image (TTI) models, exemplified by DALL-E and StableDiffusion, have recently gained prominence for their remarkable zero-shot capabilities in generating images guided by textual prompts. Language, as a conduit of culture, plays a pivotal role in these models' multilingual capabilities, which in turn shape their cultural agency. In this study, we explore the cultural perception embedded in TTI models by characterizing culture across three hierarchical tiers: cultural dimensions, cultural domains, and cultural concepts. We propose a comprehensive suite of evaluation techniques, including intrinsic evaluations using the CLIP space, extrinsic evaluations with a Visual-Question-Answer (VQA) model, and human assessments, to discern TTI cultural perceptions. To facilitate our research, we introduce the CulText2I dataset, derived from four diverse TTI models and spanning ten languages. Our experiments reveal insights into these models' cultural awareness, cultural distinctions, and the unlocking of cultural features, releasing the potential for cross-cultural applications.

{{</citation>}}


### (66/156) Hierarchical Evaluation Framework: Best Practices for Human Evaluation (Iva Bojic et al., 2023)

{{<citation>}}

Iva Bojic, Jessica Chen, Si Yuan Chang, Qi Chwen Ong, Shafiq Joty, Josip Car. (2023)  
**Hierarchical Evaluation Framework: Best Practices for Human Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: AI, Machine Reading Comprehension, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.01917v1)  

---


**ABSTRACT**  
Human evaluation plays a crucial role in Natural Language Processing (NLP) as it assesses the quality and relevance of developed systems, thereby facilitating their enhancement. However, the absence of widely accepted human evaluation metrics in NLP hampers fair comparisons among different systems and the establishment of universal assessment standards. Through an extensive analysis of existing literature on human evaluation metrics, we identified several gaps in NLP evaluation methodologies. These gaps served as motivation for developing our own hierarchical evaluation framework. The proposed framework offers notable advantages, particularly in providing a more comprehensive representation of the NLP system's performance. We applied this framework to evaluate the developed Machine Reading Comprehension system, which was utilized within a human-AI symbiosis model. The results highlighted the associations between the quality of inputs and outputs, underscoring the necessity to evaluate both components rather than solely focusing on outputs. In future work, we will investigate the potential time-saving benefits of our proposed framework for evaluators assessing NLP systems.

{{</citation>}}


### (67/156) Ring Attention with Blockwise Transformers for Near-Infinite Context (Hao Liu et al., 2023)

{{<citation>}}

Hao Liu, Matei Zaharia, Pieter Abbeel. (2023)  
**Ring Attention with Blockwise Transformers for Near-Infinite Context**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.01889v2)  

---


**ABSTRACT**  
Transformers have emerged as the architecture of choice for many state-of-the-art AI models, showcasing exceptional performance across a wide range of AI applications. However, the memory demands imposed by Transformers limit their ability to handle long sequences, thereby creating challenges for tasks involving extended sequences or long-term dependencies. We present a distinct approach, Ring Attention, which leverages blockwise computation of self-attention to distribute long sequences across multiple devices while concurrently overlapping the communication of key-value blocks with the computation of blockwise attention. By processing longer input sequences while maintaining memory efficiency, Ring Attention enables training and inference of sequences that are device count times longer than those of prior memory-efficient Transformers, effectively eliminating the memory constraints imposed by individual devices. Extensive experiments on language modeling tasks demonstrate the effectiveness of Ring Attention in allowing large sequence input size and improving performance.

{{</citation>}}


### (68/156) Benchmarking and Improving Generator-Validator Consistency of Language Models (Xiang Lisa Li et al., 2023)

{{<citation>}}

Xiang Lisa Li, Vaishnavi Shrivastava, Siyan Li, Tatsunori Hashimoto, Percy Liang. (2023)  
**Benchmarking and Improving Generator-Validator Consistency of Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.01846v1)  

---


**ABSTRACT**  
As of September 2023, ChatGPT correctly answers "what is 7+8" with 15, but when asked "7+8=15, True or False" it responds with "False". This inconsistency between generating and validating an answer is prevalent in language models (LMs) and erodes trust. In this paper, we propose a framework for measuring the consistency between generation and validation (which we call generator-validator consistency, or GV-consistency), finding that even GPT-4, a state-of-the-art LM, is GV-consistent only 76% of the time. To improve the consistency of LMs, we propose to finetune on the filtered generator and validator responses that are GV-consistent, and call this approach consistency fine-tuning. We find that this approach improves GV-consistency of Alpaca-30B from 60% to 93%, and the improvement extrapolates to unseen tasks and domains (e.g., GV-consistency for positive style transfers extrapolates to unseen styles like humor). In addition to improving consistency, consistency fine-tuning improves both generator quality and validator accuracy without using any labeled data. Evaluated across 6 tasks, including math questions, knowledge-intensive QA, and instruction following, our method improves the generator quality by 16% and the validator accuracy by 6.3% across all tasks.

{{</citation>}}


### (69/156) Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs (Suyu Ge et al., 2023)

{{<citation>}}

Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, Jianfeng Gao. (2023)  
**Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.01801v1)  

---


**ABSTRACT**  
In this study, we introduce adaptive KV cache compression, a plug-and-play method that reduces the memory footprint of generative inference for Large Language Models (LLMs). Different from the conventional KV cache that retains key and value vectors for all context tokens, we conduct targeted profiling to discern the intrinsic structure of attention modules. Based on the recognized structure, we then construct the KV cache in an adaptive manner: evicting long-range contexts on attention heads emphasizing local contexts, discarding non-special tokens on attention heads centered on special tokens, and only employing the standard KV cache for attention heads that broadly attend to all tokens. Moreover, with the lightweight attention profiling used to guide the construction of the adaptive KV cache, FastGen can be deployed without resource-intensive fine-tuning or re-training. In our experiments across various asks, FastGen demonstrates substantial reduction on GPU memory consumption with negligible generation quality loss. We will release our code and the compatible CUDA kernel for reproducibility.

{{</citation>}}


### (70/156) Large Language Models Cannot Self-Correct Reasoning Yet (Jie Huang et al., 2023)

{{<citation>}}

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, Denny Zhou. (2023)  
**Large Language Models Cannot Self-Correct Reasoning Yet**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.01798v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have emerged as a groundbreaking technology with their unparalleled text generation capabilities across various applications. Nevertheless, concerns persist regarding the accuracy and appropriateness of their generated content. A contemporary methodology, self-correction, has been proposed as a remedy to these issues. Building upon this premise, this paper critically examines the role and efficacy of self-correction within LLMs, shedding light on its true potential and limitations. Central to our investigation is the notion of intrinsic self-correction, whereby an LLM attempts to correct its initial responses based solely on its inherent capabilities, without the crutch of external feedback. In the context of reasoning, our research indicates that LLMs struggle to self-correct their responses without external feedback, and at times, their performance might even degrade post self-correction. Drawing from these insights, we offer suggestions for future research and practical applications in this field.

{{</citation>}}


### (71/156) SEA: Sparse Linear Attention with Estimated Attention Mask (Heejun Lee et al., 2023)

{{<citation>}}

Heejun Lee, Jina Kim, Jeffrey Willette, Sung Ju Hwang. (2023)  
**SEA: Sparse Linear Attention with Estimated Attention Mask**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.01777v1)  

---


**ABSTRACT**  
The transformer architecture has made breakthroughs in recent years on tasks which require modeling pairwise relationships between sequential elements, as is the case in natural language understanding. However, transformers struggle with long sequences due to the quadratic complexity of the attention operation, and previous research has aimed to lower the complexity by sparsifying or linearly approximating the attention matrix. Yet, these approaches cannot straightforwardly distill knowledge from a teacher's attention matrix, and often require complete retraining from scratch. Furthermore, previous sparse and linear approaches may also lose interpretability if they do not produce full quadratic attention matrices. To address these challenges, we propose SEA: Sparse linear attention with an Estimated Attention mask. SEA estimates the attention matrix with linear complexity via kernel-based linear attention, then creates a sparse approximation to the full attention matrix with a top-k selection to perform a sparse attention operation. For language modeling tasks (Wikitext2), previous linear and sparse attention methods show a roughly two-fold worse perplexity scores over the quadratic OPT-125M baseline, while SEA achieves an even better perplexity than OPT-125M, using roughly half as much memory as OPT-125M. Moreover, SEA maintains an interpretable attention matrix and can utilize knowledge distillation to lower the complexity of existing pretrained transformers. We believe that our work will have a large practical impact, as it opens the possibility of running large transformers on resource-limited devices with less memory.

{{</citation>}}


### (72/156) Stack Attention: Improving the Ability of Transformers to Model Hierarchical Patterns (Brian DuSell et al., 2023)

{{<citation>}}

Brian DuSell, David Chiang. (2023)  
**Stack Attention: Improving the Ability of Transformers to Model Hierarchical Patterns**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.01749v1)  

---


**ABSTRACT**  
Attention, specifically scaled dot-product attention, has proven effective for natural language, but it does not have a mechanism for handling hierarchical patterns of arbitrary nesting depth, which limits its ability to recognize certain syntactic structures. To address this shortcoming, we propose stack attention: an attention operator that incorporates stacks, inspired by their theoretical connections to context-free languages (CFLs). We show that stack attention is analogous to standard attention, but with a latent model of syntax that requires no syntactic supervision. We propose two variants: one related to deterministic pushdown automata (PDAs) and one based on nondeterministic PDAs, which allows transformers to recognize arbitrary CFLs. We show that transformers with stack attention are very effective at learning CFLs that standard transformers struggle on, achieving strong results on a CFL with theoretically maximal parsing difficulty. We also show that stack attention is more effective at natural language modeling under a constrained parameter budget, and we include results on machine translation.

{{</citation>}}


### (73/156) Nugget: Neural Agglomerative Embeddings of Text (Guanghui Qin et al., 2023)

{{<citation>}}

Guanghui Qin, Benjamin Van Durme. (2023)  
**Nugget: Neural Agglomerative Embeddings of Text**  

---
Primary Category: cs.CL  
Categories: I-2-7; I-2-6, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.01732v1)  

---


**ABSTRACT**  
Embedding text sequences is a widespread requirement in modern language understanding. Existing approaches focus largely on constant-size representations. This is problematic, as the amount of information contained in text often varies with the length of the input. We propose a solution called Nugget, which encodes language into a representation based on a dynamically selected subset of input tokens. These nuggets are learned through tasks like autoencoding and machine translation, and intuitively segment language into meaningful units. We demonstrate Nugget outperforms related approaches in tasks involving semantic comparison. Finally, we illustrate these compact units allow for expanding the contextual window of a language model (LM), suggesting new future LMs that can condition on significantly larger amounts of content.

{{</citation>}}


### (74/156) Deciphering Diagnoses: How Large Language Models Explanations Influence Clinical Decision Making (D. Umerenkov et al., 2023)

{{<citation>}}

D. Umerenkov, G. Zubkova, A. Nesterov. (2023)  
**Deciphering Diagnoses: How Large Language Models Explanations Influence Clinical Decision Making**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Clinical, Language Model  
[Paper Link](http://arxiv.org/abs/2310.01708v1)  

---


**ABSTRACT**  
Clinical Decision Support Systems (CDSS) utilize evidence-based knowledge and patient data to offer real-time recommendations, with Large Language Models (LLMs) emerging as a promising tool to generate plain-text explanations for medical decisions. This study explores the effectiveness and reliability of LLMs in generating explanations for diagnoses based on patient complaints. Three experienced doctors evaluated LLM-generated explanations of the connection between patient complaints and doctor and model-assigned diagnoses across several stages. Experimental results demonstrated that LLM explanations significantly increased doctors' agreement rates with given diagnoses and highlighted potential errors in LLM outputs, ranging from 5% to 30%. The study underscores the potential and challenges of LLMs in healthcare and emphasizes the need for careful integration and evaluation to ensure patient safety and optimal clinical utility.

{{</citation>}}


## eess.IV (3)



### (75/156) OCU-Net: A Novel U-Net Architecture for Enhanced Oral Cancer Segmentation (Ahmed Albishri et al., 2023)

{{<citation>}}

Ahmed Albishri, Syed Jawad Hussain Shah, Yugyung Lee, Rong Wang. (2023)  
**OCU-Net: A Novel U-Net Architecture for Enhanced Oral Cancer Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.02486v1)  

---


**ABSTRACT**  
Accurate detection of oral cancer is crucial for improving patient outcomes. However, the field faces two key challenges: the scarcity of deep learning-based image segmentation research specifically targeting oral cancer and the lack of annotated data. Our study proposes OCU-Net, a pioneering U-Net image segmentation architecture exclusively designed to detect oral cancer in hematoxylin and eosin (H&E) stained image datasets. OCU-Net incorporates advanced deep learning modules, such as the Channel and Spatial Attention Fusion (CSAF) module, a novel and innovative feature that emphasizes important channel and spatial areas in H&E images while exploring contextual information. In addition, OCU-Net integrates other innovative components such as Squeeze-and-Excite (SE) attention module, Atrous Spatial Pyramid Pooling (ASPP) module, residual blocks, and multi-scale fusion. The incorporation of these modules showed superior performance for oral cancer segmentation for two datasets used in this research. Furthermore, we effectively utilized the efficient ImageNet pre-trained MobileNet-V2 model as a backbone of our OCU-Net to create OCU-Netm, an enhanced version achieving state-of-the-art results. Comprehensive evaluation demonstrates that OCU-Net and OCU-Netm outperformed existing segmentation methods, highlighting their precision in identifying cancer cells in H&E images from OCDC and ORCA datasets.

{{</citation>}}


### (76/156) Improving style transfer in dynamic contrast enhanced MRI using a spatio-temporal approach (Adam G. Tattersall et al., 2023)

{{<citation>}}

Adam G. Tattersall, Keith A. Goatman, Lucy E. Kershaw, Scott I. K. Semple, Sonia Dahdouh. (2023)  
**Improving style transfer in dynamic contrast enhanced MRI using a spatio-temporal approach**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.01908v1)  

---


**ABSTRACT**  
Style transfer in DCE-MRI is a challenging task due to large variations in contrast enhancements across different tissues and time. Current unsupervised methods fail due to the wide variety of contrast enhancement and motion between the images in the series. We propose a new method that combines autoencoders to disentangle content and style with convolutional LSTMs to model predicted latent spaces along time and adaptive convolutions to tackle the localised nature of contrast enhancement. To evaluate our method, we propose a new metric that takes into account the contrast enhancement. Qualitative and quantitative analyses show that the proposed method outperforms the state of the art on two different datasets.

{{</citation>}}


### (77/156) Shifting More Attention to Breast Lesion Segmentation in Ultrasound Videos (Junhao Lin et al., 2023)

{{<citation>}}

Junhao Lin, Qian Dai, Lei Zhu, Huazhu Fu, Qiong Wang, Weibin Li, Wenhao Rao, Xiaoyang Huang, Liansheng Wang. (2023)  
**Shifting More Attention to Breast Lesion Segmentation in Ultrasound Videos**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-GR, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.01861v1)  

---


**ABSTRACT**  
Breast lesion segmentation in ultrasound (US) videos is essential for diagnosing and treating axillary lymph node metastasis. However, the lack of a well-established and large-scale ultrasound video dataset with high-quality annotations has posed a persistent challenge for the research community. To overcome this issue, we meticulously curated a US video breast lesion segmentation dataset comprising 572 videos and 34,300 annotated frames, covering a wide range of realistic clinical scenarios. Furthermore, we propose a novel frequency and localization feature aggregation network (FLA-Net) that learns temporal features from the frequency domain and predicts additional lesion location positions to assist with breast lesion segmentation. We also devise a localization-based contrastive loss to reduce the lesion location distance between neighboring video frames within the same video and enlarge the location distances between frames from different ultrasound videos. Our experiments on our annotated dataset and two public video polyp segmentation datasets demonstrate that our proposed FLA-Net achieves state-of-the-art performance in breast lesion segmentation in US videos and video polyp segmentation while significantly reducing time and space complexity. Our model and dataset are available at https://github.com/jhl-Det/FLA-Net.

{{</citation>}}


## cs.RO (9)



### (78/156) Human-Like Autonomous Driving on Dense Traffic (Mustafa Yildirim et al., 2023)

{{<citation>}}

Mustafa Yildirim, Saber Fallah, Alireza Tamaddoni-Nezhad. (2023)  
**Human-Like Autonomous Driving on Dense Traffic**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02477v1)  

---


**ABSTRACT**  
This paper proposes a imitation learning model for autonomous driving on highway traffic by mimicking human drivers' driving behaviours. The study utilizes the HighD traffic dataset, which is complex, high-dimensional, and diverse in vehicle variations. Imitation learning is an alternative solution to autonomous highway driving that reduces the sample complexity of learning a challenging task compared to reinforcement learning. However, imitation learning has limitations such as vulnerability to compounding errors in unseen states, poor generalization, and inability to predict outlier driver profiles. To address these issues, the paper proposes mixture density network behaviour cloning model to manage complex and non-linear relationships between inputs and outputs and make more informed decisions about the vehicle's actions. Additional improvement is using collision penalty based on the GAIL model. The paper includes a simulated driving test to demonstrate the effectiveness of the proposed method based on real traffic scenarios and provides conclusions on its potential impact on autonomous driving.

{{</citation>}}


### (79/156) Improved Inference of Human Intent by Combining Plan Recognition and Language Feedback (Ifrah Idrees et al., 2023)

{{<citation>}}

Ifrah Idrees, Tian Yun, Naveen Sharma, Yunxin Deng, Nakul Gopalan, George Konidaris, Stefanie Tellex. (2023)  
**Improved Inference of Human Intent by Combining Plan Recognition and Language Feedback**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-HC, cs-RO, cs.RO  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2310.02462v1)  

---


**ABSTRACT**  
Conversational assistive robots can aid people, especially those with cognitive impairments, to accomplish various tasks such as cooking meals, performing exercises, or operating machines. However, to interact with people effectively, robots must recognize human plans and goals from noisy observations of human actions, even when the user acts sub-optimally. Previous works on Plan and Goal Recognition (PGR) as planning have used hierarchical task networks (HTN) to model the actor/human. However, these techniques are insufficient as they do not have user engagement via natural modes of interaction such as language. Moreover, they have no mechanisms to let users, especially those with cognitive impairments, know of a deviation from their original plan or about any sub-optimal actions taken towards their goal. We propose a novel framework for plan and goal recognition in partially observable domains -- Dialogue for Goal Recognition (D4GR) enabling a robot to rectify its belief in human progress by asking clarification questions about noisy sensor data and sub-optimal human actions. We evaluate the performance of D4GR over two simulated domains -- kitchen and blocks domain. With language feedback and the world state information in a hierarchical task model, we show that D4GR framework for the highest sensor noise performs 1% better than HTN in goal accuracy in both domains. For plan accuracy, D4GR outperforms by 4% in the kitchen domain and 2% in the blocks domain in comparison to HTN. The ALWAYS-ASK oracle outperforms our policy by 3% in goal recognition and 7%in plan recognition. D4GR does so by asking 68% fewer questions than an oracle baseline. We also demonstrate a real-world robot scenario in the kitchen domain, validating the improved plan and goal recognition of D4GR in a realistic setting.

{{</citation>}}


### (80/156) Autonomous Systems' Safety Cases for use in UK Nuclear Environments (Christopher R. Anderson et al., 2023)

{{<citation>}}

Christopher R. Anderson, Louise A. Dennis. (2023)  
**Autonomous Systems' Safety Cases for use in UK Nuclear Environments**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02344v1)  

---


**ABSTRACT**  
An overview of the process to develop a safety case for an autonomous robot deployment on a nuclear site in the UK is described and a safety case for a hypothetical robot incorporating AI is presented. This forms a first step towards a deployment, showing what is possible now and what may be possible with development of tools. It forms the basis for further discussion between nuclear site licensees, the Office for Nuclear Regulation (ONR), industry and academia.

{{</citation>}}


### (81/156) Generalizable Long-Horizon Manipulations with Large Language Models (Haoyu Zhou et al., 2023)

{{<citation>}}

Haoyu Zhou, Mingyu Ding, Weikun Peng, Masayoshi Tomizuka, Lin Shao, Chuang Gan. (2023)  
**Generalizable Long-Horizon Manipulations with Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-CL, cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02264v1)  

---


**ABSTRACT**  
This work introduces a framework harnessing the capabilities of Large Language Models (LLMs) to generate primitive task conditions for generalizable long-horizon manipulations with novel objects and unseen tasks. These task conditions serve as guides for the generation and adjustment of Dynamic Movement Primitives (DMP) trajectories for long-horizon task execution. We further create a challenging robotic manipulation task suite based on Pybullet for long-horizon task evaluation. Extensive experiments in both simulated and real-world environments demonstrate the effectiveness of our framework on both familiar tasks involving new objects and novel but related tasks, highlighting the potential of LLMs in enhancing robotic system versatility and adaptability. Project website: https://object814.github.io/Task-Condition-With-LLM/

{{</citation>}}


### (82/156) Video Transformers under Occlusion: How Physics and Background Attributes Impact Large Models for Robotic Manipulation (Shutong Jin et al., 2023)

{{<citation>}}

Shutong Jin, Ruiyu Wang, Muhammad Zahid, Florian T. Pokorny. (2023)  
**Video Transformers under Occlusion: How Physics and Background Attributes Impact Large Models for Robotic Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.02044v1)  

---


**ABSTRACT**  
As transformer architectures and dataset sizes continue to scale, the need to understand the specific dataset factors affecting model performance becomes increasingly urgent. This paper investigates how object physics attributes (color, friction coefficient, shape) and background characteristics (static, dynamic, background complexity) influence the performance of Video Transformers in trajectory prediction tasks under occlusion. Beyond mere occlusion challenges, this study aims to investigate three questions: How do object physics attributes and background characteristics influence the model performance? What kinds of attributes are most influential to the model generalization? Is there a data saturation point for large transformer model performance within a single task? To facilitate this research, we present OccluManip, a real-world video-based robot pushing dataset comprising 460,000 consistent recordings of objects with different physics and varying backgrounds. 1.4 TB and in total 1278 hours of high-quality videos of flexible temporal length along with target object trajectories are collected, accommodating tasks with different temporal requirements. Additionally, we propose Video Occlusion Transformer (VOT), a generic video-transformer-based network achieving an average 96% accuracy across all 18 sub-datasets provided in OccluManip. OccluManip and VOT will be released at: https://github.com/ShutongJIN/OccluManip.git

{{</citation>}}


### (83/156) Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving (Long Chen et al., 2023)

{{<citation>}}

Long Chen, Oleg Sinavski, Jan Hünermann, Alice Karnsund, Andrew James Willmott, Danny Birch, Daniel Maund, Jamie Shotton. (2023)  
**Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CL, cs-CV, cs-RO, cs.RO  
Keywords: GPT, GPT-3.5, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.01957v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown promise in the autonomous driving sector, particularly in generalization and interpretability. We introduce a unique object-level multimodal LLM architecture that merges vectorized numeric modalities with a pre-trained LLM to improve context understanding in driving situations. We also present a new dataset of 160k QA pairs derived from 10k driving scenarios, paired with high quality control commands collected with RL agent and question answer pairs generated by teacher LLM (GPT-3.5). A distinct pretraining strategy is devised to align numeric vector modalities with static LLM representations using vector captioning language data. We also introduce an evaluation metric for Driving QA and demonstrate our LLM-driver's proficiency in interpreting driving scenarios, answering questions, and decision-making. Our findings highlight the potential of LLM-based driving action generation in comparison to traditional behavioral cloning. We make our benchmark, datasets, and model available for further exploration.

{{</citation>}}


### (84/156) Semi-Aerodynamic Model Aided Invariant Kalman Filtering for UAV Full-State Estimation (Xiaoyu Ye et al., 2023)

{{<citation>}}

Xiaoyu Ye, Fujun Song, Zongyu Zhang, Rui Zhang, Qinghua Zeng. (2023)  
**Semi-Aerodynamic Model Aided Invariant Kalman Filtering for UAV Full-State Estimation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.01844v1)  

---


**ABSTRACT**  
Due to the state trajectory-independent features of invariant Kalman filtering (InEKF), it has attracted widespread attention in the research community for its significantly improved state estimation accuracy and convergence under disturbance. In this paper, we formulate the full-source data fusion navigation problem for fixed-wing unmanned aerial vehicle (UAV) within a framework based on error state right-invariant extended Kalman filtering (ES-RIEKF) on Lie groups. We merge measurements from a multi-rate onboard sensor network on UAVs to achieve real-time estimation of pose, air flow angles, and wind speed. Detailed derivations are provided, and the algorithm's convergence and accuracy improvements over established methods like Error State EKF (ES-EKF) and Nonlinear Complementary Filter (NCF) are demonstrated using real-flight data from UAVs. Additionally, we introduce a semi-aerodynamic model fusion framework that relies solely on ground-measurable parameters. We design and train an Long Short Term Memory (LSTM) deep network to achieve drift-free prediction of the UAV's angle of attack (AOA) and side-slip angle (SA) using easily obtainable onboard data like control surface deflections, thereby significantly reducing dependency on GNSS or complicated aerodynamic model parameters. Further, we validate the algorithm's robust advantages under GNSS denied, where flight data shows that the maximum positioning error stays within 30 meters over a 130-second denial period. To the best of our knowledge, this study is the first to apply ES-RIEKF to full-source navigation applications for fixed-wing UAVs, aiming to provide engineering references for designers. Our implementations using MATLAB/Simulink will open source.

{{</citation>}}


### (85/156) TempoNet: Empowering long-term Knee Joint Angle Prediction with Dynamic Temporal Attention in Exoskeleton Control (Lyes Saad Saoud et al., 2023)

{{<citation>}}

Lyes Saad Saoud, Irfan Hussain. (2023)  
**TempoNet: Empowering long-term Knee Joint Angle Prediction with Dynamic Temporal Attention in Exoskeleton Control**  

---
Primary Category: cs.RO  
Categories: cs-HC, cs-RO, cs.RO  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2310.01795v1)  

---


**ABSTRACT**  
In the realm of exoskeleton control, achieving precise control poses challenges due to the mechanical delay of exoskeletons. To address this, incorporating future gait trajectories as feed-forward input has been proposed. However, existing deep learning models for gait prediction mainly focus on short-term predictions, leaving the long-term performance of these models relatively unexplored. In this study, we present TempoNet, a novel model specifically designed for precise knee joint angle prediction. By harnessing dynamic temporal attention within the Transformer-based architecture, TempoNet surpasses existing models in forecasting knee joint angles over extended time horizons. Notably, our model achieves a remarkable reduction of 10\% to 185\% in Mean Absolute Error (MAE) for 100 ms ahead forecasting compared to other transformer-based models, demonstrating its effectiveness. Furthermore, TempoNet exhibits further reliability and superiority over the baseline Transformer model, outperforming it by 14\% in MAE for the 200 ms prediction horizon. These findings highlight the efficacy of TempoNet in accurately predicting knee joint angles and emphasize the importance of incorporating dynamic temporal attention. TempoNet's capability to enhance knee joint angle prediction accuracy opens up possibilities for precise control, improved rehabilitation outcomes, advanced sports performance analysis, and deeper insights into biomechanical research. Code implementation for the TempoNet model can be found in the GitHub repository: https://github.com/LyesSaadSaoud/TempoNet.

{{</citation>}}


### (86/156) Differentially Encoded Observation Spaces for Perceptive Reinforcement Learning (Lev Grossman et al., 2023)

{{<citation>}}

Lev Grossman, Brian Plancher. (2023)  
**Differentially Encoded Observation Spaces for Perceptive Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.01767v1)  

---


**ABSTRACT**  
Perceptive deep reinforcement learning (DRL) has lead to many recent breakthroughs for complex AI systems leveraging image-based input data. Applications of these results range from super-human level video game agents to dexterous, physically intelligent robots. However, training these perceptive DRL-enabled systems remains incredibly compute and memory intensive, often requiring huge training datasets and large experience replay buffers. This poses a challenge for the next generation of field robots that will need to be able to learn on the edge in order to adapt to their environments. In this paper, we begin to address this issue through differentially encoded observation spaces. By reinterpreting stored image-based observations as a video, we leverage lossless differential video encoding schemes to compress the replay buffer without impacting training performance. We evaluate our approach with three state-of-the-art DRL algorithms and find that differential image encoding reduces the memory footprint by as much as 14.2x and 16.7x across tasks from the Atari 2600 benchmark and the DeepMind Control Suite (DMC) respectively. These savings also enable large-scale perceptive DRL that previously required paging between flash and RAM to be run entirely in RAM, improving the latency of DMC tasks by as much as 32%.

{{</citation>}}


## cs.SE (11)



### (87/156) EcoAssistant: Using LLM Assistant More Affordably and Accurately (Jieyu Zhang et al., 2023)

{{<citation>}}

Jieyu Zhang, Ranjay Krishna, Ahmed H. Awadallah, Chi Wang. (2023)  
**EcoAssistant: Using LLM Assistant More Affordably and Accurately**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.03046v1)  

---


**ABSTRACT**  
Today, users ask Large language models (LLMs) as assistants to answer queries that require external knowledge; they ask about the weather in a specific city, about stock prices, and even about where specific locations are within their neighborhood. These queries require the LLM to produce code that invokes external APIs to answer the user's question, yet LLMs rarely produce correct code on the first try, requiring iterative code refinement upon execution results. In addition, using LLM assistants to support high query volumes can be expensive. In this work, we contribute a framework, EcoAssistant, that enables LLMs to answer code-driven queries more affordably and accurately. EcoAssistant contains three components. First, it allows the LLM assistants to converse with an automatic code executor to iteratively refine code or to produce answers based on the execution results. Second, we use a hierarchy of LLM assistants, which attempts to answer the query with weaker, cheaper LLMs before backing off to stronger, expensive ones. Third, we retrieve solutions from past successful queries as in-context demonstrations to help subsequent queries. Empirically, we show that EcoAssistant offers distinct advantages for affordability and accuracy, surpassing GPT-4 by 10 points of success rate with less than 50% of GPT-4's cost.

{{</citation>}}


### (88/156) Automated Bug Generation in the era of Large Language Models (Ali Reza Ibrahimzada et al., 2023)

{{<citation>}}

Ali Reza Ibrahimzada, Yang Chen, Ryan Rong, Reyhaneh Jabbarvand. (2023)  
**Automated Bug Generation in the era of Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02407v1)  

---


**ABSTRACT**  
Bugs are essential in software engineering; many research studies in the past decades have been proposed to detect, localize, and repair bugs in software systems. Effectiveness evaluation of such techniques requires complex bugs, i.e., those that are hard to detect through testing and hard to repair through debugging. From the classic software engineering point of view, a hard-to-repair bug differs from the correct code in multiple locations, making it hard to localize and repair. Hard-to-detect bugs, on the other hand, manifest themselves under specific test inputs and reachability conditions. These two objectives, i.e., generating hard-to-detect and hard-to-repair bugs, are mostly aligned; a bug generation technique can change multiple statements to be covered only under a specific set of inputs. However, these two objectives are conflicting for learning-based techniques: A bug should have a similar code representation to the correct code in the training data to challenge a bug prediction model to distinguish them. The hard-to-repair bug definition remains the same but with a caveat: the more a bug differs from the original code (at multiple locations), the more distant their representations are and easier to be detected. We propose BugFarm, to transform arbitrary code into multiple complex bugs. BugFarm leverages LLMs to mutate code in multiple locations (hard-to-repair). To ensure that multiple modifications do not notably change the code representation, BugFarm analyzes the attention of the underlying model and instructs LLMs to only change the least attended locations (hard-to-detect). Our comprehensive evaluation of 320k+ bugs from over 2.5M mutants generated by BugFarm and two alternative approaches demonstrates our superiority in generating bugs that are hard to detect by learning-based bug prediction approaches and hard to repair by SOTA learning-based program repair technique.

{{</citation>}}


### (89/156) Reinforcement Learning from Automatic Feedback for High-Quality Unit Test Generation (Benjamin Steenhoek et al., 2023)

{{<citation>}}

Benjamin Steenhoek, Michele Tufano, Neel Sundaresan, Alexey Svyatkovskiy. (2023)  
**Reinforcement Learning from Automatic Feedback for High-Quality Unit Test Generation**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: GPT, GPT-4, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.02368v1)  

---


**ABSTRACT**  
Software testing is a crucial aspect of software development, and the creation of high-quality tests that adhere to best practices is essential for effective maintenance. Recently, Large Language Models (LLMs) have gained popularity for code generation, including the automated creation of test cases. However, these LLMs are often trained on vast amounts of publicly available code, which may include test cases that do not adhere to best practices and may even contain test smells (anti-patterns). To address this issue, we propose a novel technique called Reinforcement Learning from Static Quality Metrics (RLSQM). To begin, we analyze the anti-patterns generated by the LLM and show that LLMs can generate undesirable test smells. Thus, we train specific reward models for each static quality metric, then utilize Proximal Policy Optimization (PPO) to train models for optimizing a single quality metric at a time. Furthermore, we amalgamate these rewards into a unified reward model aimed at capturing different best practices and quality aspects of tests. By comparing RL-trained models with those trained using supervised learning, we provide insights into how reliably utilize RL to improve test generation quality and into the effects of various training strategies. Our experimental results demonstrate that the RL-optimized model consistently generated high-quality test cases compared to the base LLM, improving the model by up to 21%, and successfully generates nearly 100% syntactically correct code. RLSQM also outperformed GPT-4 on four out of seven metrics. This represents a significant step towards enhancing the overall efficiency and reliability of software testing through Reinforcement Learning and static quality metrics. Our data are available at this link: https://figshare.com/s/ded476c8d4c221222849.

{{</citation>}}


### (90/156) An empirical study of ChatGPT-3.5 on question answering and code maintenance (Md Mahir Asef Kabir et al., 2023)

{{<citation>}}

Md Mahir Asef Kabir, Sk Adnan Hassan, Xiaoyin Wang, Ying Wang, Hai Yu, Na Meng. (2023)  
**An empirical study of ChatGPT-3.5 on question answering and code maintenance**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2310.02104v1)  

---


**ABSTRACT**  
Ever since the launch of ChatGPT in 2022, a rising concern is whether ChatGPT will replace programmers and kill jobs. Motivated by this widespread concern, we conducted an empirical study to systematically compare ChatGPT against programmers in question-answering and software-maintaining. We reused a dataset introduced by prior work, which includes 130 StackOverflow (SO) discussion threads referred to by the Java developers of 357 GitHub projects. We mainly investigated three research questions (RQs). First, how does ChatGPT compare with programmers when answering technical questions? Second, how do developers perceive the differences between ChatGPT's answers and SO answers? Third, how does ChatGPT compare with humans when revising code for maintenance requests?   For RQ1, we provided the 130 SO questions to ChatGPT, and manually compared ChatGPT answers with the accepted/most popular SO answers in terms of relevance, readability, informativeness, comprehensiveness, and reusability. For RQ2, we conducted a user study with 30 developers, asking each developer to assess and compare 10 pairs of answers, without knowing the information source (i.e., ChatGPT or SO). For RQ3, we distilled 48 software maintenance tasks from 48 GitHub projects citing the studied SO threads. We queried ChatGPT to revise a given Java file, and to incorporate the code implementation for any prescribed maintenance requirement. Our study reveals interesting phenomena: For the majority of SO questions (97/130), ChatGPT provided better answers; in 203 of 300 ratings, developers preferred ChatGPT answers to SO answers; ChatGPT revised code correctly for 22 of the 48 tasks. Our research will expand people's knowledge of ChatGPT capabilities, and shed light on future adoption of ChatGPT by the software industry.

{{</citation>}}


### (91/156) dFlow: A Domain Specific Language for the Rapid Development of open-source Virtual Assistants (Nikolaos Malamas et al., 2023)

{{<citation>}}

Nikolaos Malamas, Konstantinos Panayiotou, Andreas L. Symeonidis. (2023)  
**dFlow: A Domain Specific Language for the Rapid Development of open-source Virtual Assistants**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: NLP, NLU, Natural Language Processing, Natural Language Understanding  
[Paper Link](http://arxiv.org/abs/2310.02102v1)  

---


**ABSTRACT**  
An increasing number of models and frameworks for Virtual Assistant (VA) development exist nowadays, following the progress in the Natural Language Processing (NLP) and Natural Language Understanding (NLU) fields. Regardless of their performance, popularity, and ease of use, these frameworks require at least basic expertise in NLP and software engineering, even for simple and repetitive processes, limiting their use only to the domain and programming experts. However, since the current state of practice of VA development is a straightforward process, Model-Driven Engineering approaches can be utilized to achieve automation and rapid development in a more convenient manner. To this end, we present \textit{dFlow}, a textual Domain-Specific Language (DSL) that offers a simplified, reusable, and framework-agnostic language for creating task-specific VAs in a low-code manner. We describe a system-agnostic VA meta-model, the developed grammar, and all essential processes for developing and deploying smart VAs. For further convenience, we create a cloud-native architecture and expose it through the Discord platform. We conducted a large-scale empirical evaluation with more than 200 junior software developers and collected positive feedback, indicating that dFlow can accelerate the entire VA development process, while also enabling citizen and software developers with minimum experience to participate.

{{</citation>}}


### (92/156) Security Weaknesses of Copilot Generated Code in GitHub (Yujia Fu et al., 2023)

{{<citation>}}

Yujia Fu, Peng Liang, Amjed Tahir, Zengyang Li, Mojtaba Shahin, Jiaxin Yu. (2023)  
**Security Weaknesses of Copilot Generated Code in GitHub**  

---
Primary Category: cs.SE  
Categories: cs-CR, cs-SE, cs.SE  
Keywords: AI, Language Model, Security  
[Paper Link](http://arxiv.org/abs/2310.02059v1)  

---


**ABSTRACT**  
Modern code generation tools use AI models, particularly Large Language Models (LLMs), to generate functional and complete code. While such tools are becoming popular and widely available for developers, using these tools is often accompanied by security challenges. Therefore, it is important to assess the quality of the generated code, especially in terms of its security. Researchers have recently explored various aspects of code generation tools, including security. However, many open questions about the security of the generated code require further investigation, especially the security issues of automatically generated code in the wild. To this end, we conducted an empirical study by analyzing the security weaknesses in code snippets generated by GitHub Copilot that are found as part of publicly available projects hosted on GitHub. The goal is to investigate the types of security issues and their scale in real-world scenarios (rather than crafted scenarios). To this end, we identified 435 code snippets generated by Copilot from publicly available projects. We then conducted extensive security analysis to identify Common Weakness Enumeration (CWE) instances in these code snippets. The results show that (1) 35.8% of Copilot generated code snippets contain CWEs, and those issues are spread across multiple languages, (2) the security weaknesses are diverse and related to 42 different CWEs, in which CWE-78: OS Command Injection, CWE-330: Use of Insufficiently Random Values, and CWE-703: Improper Check or Handling of Exceptional Conditions occurred the most frequently, and (3) among the 42 CWEs identified, 11 of those belong to the currently recognized 2022 CWE Top-25. Our findings confirm that developers should be careful when adding code generated by Copilot (and similar AI code generation tools) and should also run appropriate security checks as they accept the suggested code.

{{</citation>}}


### (93/156) Improving web element localization by using a large language model (Michel Nass et al., 2023)

{{<citation>}}

Michel Nass, Emil Alegroth, Robert Feldt. (2023)  
**Improving web element localization by using a large language model**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.02046v1)  

---


**ABSTRACT**  
Web-based test automation heavily relies on accurately finding web elements. Traditional methods compare attributes but don't grasp the context and meaning of elements and words. The emergence of Large Language Models (LLMs) like GPT-4, which can show human-like reasoning abilities on some tasks, offers new opportunities for software engineering and web element localization. This paper introduces and evaluates VON Similo LLM, an enhanced web element localization approach. Using an LLM, it selects the most likely web element from the top-ranked ones identified by the existing VON Similo method, ideally aiming to get closer to human-like selection accuracy. An experimental study was conducted using 804 web element pairs from 48 real-world web applications. We measured the number of correctly identified elements as well as the execution times, comparing the effectiveness and efficiency of VON Similo LLM against the baseline algorithm. In addition, motivations from the LLM were recorded and analyzed for all instances where the original approach failed to find the right web element. VON Similo LLM demonstrated improved performance, reducing failed localizations from 70 to 39 (out of 804), a 44 percent reduction. Despite its slower execution time and additional costs of using the GPT-4 model, the LLMs human-like reasoning showed promise in enhancing web element localization. LLM technology can enhance web element identification in GUI test automation, reducing false positives and potentially lowering maintenance costs. However, further research is necessary to fully understand LLMs capabilities, limitations, and practical use in GUI testing.

{{</citation>}}


### (94/156) Formalizing Natural Language Intent into Program Specifications via Large Language Models (Madeline Endres et al., 2023)

{{<citation>}}

Madeline Endres, Sarah Fakhoury, Saikat Chakraborty, Shuvendu K. Lahiri. (2023)  
**Formalizing Natural Language Intent into Program Specifications via Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-PL, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.01831v1)  

---


**ABSTRACT**  
Informal natural language that describes code functionality, such as code comments or function documentation, may contain substantial information about a programs intent. However, there is typically no guarantee that a programs implementation and natural language documentation are aligned. In the case of a conflict, leveraging information in code-adjacent natural language has the potential to enhance fault localization, debugging, and code trustworthiness. In practice, however, this information is often underutilized due to the inherent ambiguity of natural language which makes natural language intent challenging to check programmatically. The "emergent abilities" of Large Language Models (LLMs) have the potential to facilitate the translation of natural language intent to programmatically checkable assertions. However, it is unclear if LLMs can correctly translate informal natural language specifications into formal specifications that match programmer intent. Additionally, it is unclear if such translation could be useful in practice. In this paper, we describe LLM4nl2post, the problem leveraging LLMs for transforming informal natural language to formal method postconditions, expressed as program assertions. We introduce and validate metrics to measure and compare different LLM4nl2post approaches, using the correctness and discriminative power of generated postconditions. We then perform qualitative and quantitative methods to assess the quality of LLM4nl2post postconditions, finding that they are generally correct and able to discriminate incorrect code. Finally, we find that LLM4nl2post via LLMs has the potential to be helpful in practice; specifications generated from natural language were able to catch 70 real-world historical bugs from Defects4J.

{{</citation>}}


### (95/156) Can GPT-4 Replicate Empirical Software Engineering Research? (Jenny T. Liang et al., 2023)

{{<citation>}}

Jenny T. Liang, Carmen Badea, Christian Bird, Robert DeLine, Denae Ford, Nicole Forsgren, Thomas Zimmermann. (2023)  
**Can GPT-4 Replicate Empirical Software Engineering Research?**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.01727v1)  

---


**ABSTRACT**  
Empirical software engineering research on production systems has brought forth a better understanding of the software engineering process for practitioners and researchers alike. However, only a small subset of production systems is studied, limiting the impact of this research. While software engineering practitioners benefit from replicating research on their own data, this poses its own set of challenges, since performing replications requires a deep understanding of research methodologies and subtle nuances in software engineering data. Given that large language models (LLMs), such as GPT-4, show promise in tackling both software engineering- and science-related tasks, these models could help democratize empirical software engineering research.   In this paper, we examine LLMs' abilities to perform replications of empirical software engineering research on new data. We specifically study their ability to surface assumptions made in empirical software engineering research methodologies, as well as their ability to plan and generate code for analysis pipelines on seven empirical software engineering papers. We perform a user study with 14 participants with software engineering research expertise, who evaluate GPT-4-generated assumptions and analysis plans (i.e., a list of module specifications) from the papers. We find that GPT-4 is able to surface correct assumptions, but struggle to generate ones that reflect common knowledge about software engineering data. In a manual analysis of the generated code, we find that the GPT-4-generated code contains the correct high-level logic, given a subset of the methodology. However, the code contains many small implementation-level errors, reflecting a lack of software engineering knowledge. Our findings have implications for leveraging LLMs for software engineering research as well as practitioner data scientists in software teams.

{{</citation>}}


### (96/156) Large Language Models for Test-Free Fault Localization (Aidan Z. H. Yang et al., 2023)

{{<citation>}}

Aidan Z. H. Yang, Ruben Martins, Claire Le Goues, Vincent J. Hellendoorn. (2023)  
**Large Language Models for Test-Free Fault Localization**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.01726v1)  

---


**ABSTRACT**  
Fault Localization (FL) aims to automatically localize buggy lines of code, a key first step in many manual and automatic debugging tasks. Previous FL techniques assume the provision of input tests, and often require extensive program analysis, program instrumentation, or data preprocessing. Prior work on deep learning for APR struggles to learn from small datasets and produces limited results on real-world programs. Inspired by the ability of large language models (LLMs) of code to adapt to new tasks based on very few examples, we investigate the applicability of LLMs to line level fault localization. Specifically, we propose to overcome the left-to-right nature of LLMs by fine-tuning a small set of bidirectional adapter layers on top of the representations learned by LLMs to produce LLMAO, the first language model based fault localization approach that locates buggy lines of code without any test coverage information. We fine-tune LLMs with 350 million, 6 billion, and 16 billion parameters on small, manually curated corpora of buggy programs such as the Defects4J corpus. We observe that our technique achieves substantially more confidence in fault localization when built on the larger models, with bug localization performance scaling consistently with the LLM size. Our empirical evaluation shows that LLMAO improves the Top-1 results over the state-of-the-art machine learning fault localization (MLFL) baselines by 2.3%-54.4%, and Top-5 results by 14.4%-35.6%. LLMAO is also the first FL technique trained using a language model architecture that can detect security vulnerabilities down to the code line level.

{{</citation>}}


### (97/156) Software Testing and Code Refactoring: A Survey with Practitioners (Danilo Leandro Lima et al., 2023)

{{<citation>}}

Danilo Leandro Lima, Ronnie de Souza Santos, Guilherme Pires Garcia, Sildemir S. da Silva, Cesar Franca, Luiz Fernando Capretz. (2023)  
**Software Testing and Code Refactoring: A Survey with Practitioners**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.01719v1)  

---


**ABSTRACT**  
Nowadays, software testing professionals are commonly required to develop coding skills to work on test automation. One essential skill required from those who code is the ability to implement code refactoring, a valued quality aspect of software development; however, software developers usually encounter obstacles in successfully applying this practice. In this scenario, the present study aims to explore how software testing professionals (e.g., software testers, test engineers, test analysts, and software QAs) deal with code refactoring to understand the benefits and limitations of this practice in the context of software testing. We followed the guidelines to conduct surveys in software engineering and applied three sampling techniques, namely convenience sampling, purposive sampling, and snowballing sampling, to collect data from testing professionals. We received answers from 80 individuals reporting their experience refactoring the code of automated tests. We concluded that in the context of software testing, refactoring offers several benefits, such as supporting the maintenance of automated tests and improving the performance of the testing team. However, practitioners might encounter barriers in effectively implementing this practice, in particular, the lack of interest from managers and leaders. Our study raises discussions on the importance of having testing professionals implement refactoring in the code of automated tests, allowing them to improve their coding abilities.

{{</citation>}}


## cs.SI (2)



### (98/156) Machine learning assist nyc subway navigation safer and faster (Wencheng Bao et al., 2023)

{{<citation>}}

Wencheng Bao, Shi Feng. (2023)  
**Machine learning assist nyc subway navigation safer and faster**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-LG, cs-SI, cs.SI  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.02447v1)  

---


**ABSTRACT**  
Mainstream navigation software, like Google and Apple Maps, often lacks the ability to provide routes prioritizing safety. However, safety remains a paramount concern for many. Our aim is to strike a balance between safety and efficiency. To achieve this, we're devising an Integer Programming model that takes into account both the shortest path and the safest route. We will harness machine learning to derive safety coefficients, employing methodologies such as generalized linear models, linear regression, and recurrent neural networks. Our evaluation will be based on the Root Mean Square Error (RMSE) across various subway stations, helping us identify the most accurate model for safety coefficient estimation. Furthermore, we'll conduct a comprehensive review of different shortest-path algorithms, assessing them based on time complexity and real-world data to determine their appropriateness in merging both safety and time efficiency.

{{</citation>}}


### (99/156) RouteKG: A knowledge graph-based framework for route prediction on road networks (Yihong Tang et al., 2023)

{{<citation>}}

Yihong Tang, Weipeng Deng, Shuyu Lei, Yuebing Liang, Zhenliang Ma, Zhan Zhao. (2023)  
**RouteKG: A knowledge graph-based framework for route prediction on road networks**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.03617v1)  

---


**ABSTRACT**  
Short-term route prediction on road networks allows us to anticipate the future trajectories of road users, enabling a plethora of intelligent transportation applications such as dynamic traffic control or personalized route recommendation. Despite the recent advances in this area, existing methods focus primarily on learning sequential patterns, neglecting the inherent spatial structure in road networks that can affect human routing decisions. To fill the gap, this paper introduces RouteKG, a novel Knowledge Graph-based framework for route prediction. Specifically, we construct a Knowledge Graph on the road network, thereby learning and leveraging spatial relations, especially moving directions, which are crucial for human navigation. Moreover, an n-ary tree-based algorithm is introduced to efficiently generate top-K routes in a batch mode, enhancing scalability and computational efficiency. To further optimize the prediction performance, a rank refinement module is incorporated to fine-tune the candidate route rankings. The model performance is evaluated using two real-world vehicle trajectory datasets from two Chinese cities, Chengdu and Shanghai, under various practical scenarios. The results demonstrate a significant improvement in accuracy over baseline methods, with an average increase of 6.2%, 7.8%, and 6.1% in top-1, 5, and 10 routes predictions, respectively. We further validate our model through a case study that utilizes the pretrained model as a simulator for real-time traffic flow estimation at the link level. The proposed RouteKG promises wide-ranging applications in vehicle navigation, traffic management, and other intelligent transportation tasks.

{{</citation>}}


## cs.MA (1)



### (100/156) Multi-Agent Reinforcement Learning Based on Representational Communication for Large-Scale Traffic Signal Control (Rohit Bokade et al., 2023)

{{<citation>}}

Rohit Bokade, Xiaoning Jin, Christopher Amato. (2023)  
**Multi-Agent Reinforcement Learning Based on Representational Communication for Large-Scale Traffic Signal Control**  

---
Primary Category: cs.MA  
Categories: I-2-11, cs-AI, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.02435v1)  

---


**ABSTRACT**  
Traffic signal control (TSC) is a challenging problem within intelligent transportation systems and has been tackled using multi-agent reinforcement learning (MARL). While centralized approaches are often infeasible for large-scale TSC problems, decentralized approaches provide scalability but introduce new challenges, such as partial observability. Communication plays a critical role in decentralized MARL, as agents must learn to exchange information using messages to better understand the system and achieve effective coordination. Deep MARL has been used to enable inter-agent communication by learning communication protocols in a differentiable manner. However, many deep MARL communication frameworks proposed for TSC allow agents to communicate with all other agents at all times, which can add to the existing noise in the system and degrade overall performance. In this study, we propose a communication-based MARL framework for large-scale TSC. Our framework allows each agent to learn a communication policy that dictates "which" part of the message is sent "to whom". In essence, our framework enables agents to selectively choose the recipients of their messages and exchange variable length messages with them. This results in a decentralized and flexible communication mechanism in which agents can effectively use the communication channel only when necessary. We designed two networks, a synthetic $4 \times 4$ grid network and a real-world network based on the Pasubio neighborhood in Bologna. Our framework achieved the lowest network congestion compared to related methods, with agents utilizing $\sim 47-65 \%$ of the communication channel. Ablation studies further demonstrated the effectiveness of the communication policies learned within our framework.

{{</citation>}}


## cs.HC (5)



### (101/156) Can Large Language Models Provide Security & Privacy Advice? Measuring the Ability of LLMs to Refute Misconceptions (Yufan Chen et al., 2023)

{{<citation>}}

Yufan Chen, Arjun Arunasalam, Z. Berkay Celik. (2023)  
**Can Large Language Models Provide Security & Privacy Advice? Measuring the Ability of LLMs to Refute Misconceptions**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-HC, cs.HC  
Keywords: ChatGPT, GPT, Language Model, Security  
[Paper Link](http://arxiv.org/abs/2310.02431v1)  

---


**ABSTRACT**  
Users seek security & privacy (S&P) advice from online resources, including trusted websites and content-sharing platforms. These resources help users understand S&P technologies and tools and suggest actionable strategies. Large Language Models (LLMs) have recently emerged as trusted information sources. However, their accuracy and correctness have been called into question. Prior research has outlined the shortcomings of LLMs in answering multiple-choice questions and user ability to inadvertently circumvent model restrictions (e.g., to produce toxic content). Yet, the ability of LLMs to provide reliable S&P advice is not well-explored. In this paper, we measure their ability to refute popular S&P misconceptions that the general public holds. We first study recent academic literature to curate a dataset of over a hundred S&P-related misconceptions across six different topics. We then query two popular LLMs (Bard and ChatGPT) and develop a labeling guide to evaluate their responses to these misconceptions. To comprehensively evaluate their responses, we further apply three strategies: query each misconception multiple times, generate and query their paraphrases, and solicit source URLs of the responses. Both models demonstrate, on average, a 21.3% non-negligible error rate, incorrectly supporting popular S&P misconceptions. The error rate increases to 32.6% when we repeatedly query LLMs with the same or paraphrased misconceptions. We also expose that models may partially support a misconception or remain noncommittal, refusing a firm stance on misconceptions. Our exploration of information sources for responses revealed that LLMs are susceptible to providing invalid URLs (21.2% for Bard and 67.7% for ChatGPT) or point to unrelated sources (44.2% returned by Bard and 18.3% by ChatGPT).

{{</citation>}}


### (102/156) AXNav: Replaying Accessibility Tests from Natural Language (Maryam Taeb et al., 2023)

{{<citation>}}

Maryam Taeb, Amanda Swearngin, Eldon School, Ruijia Cheng, Yue Jiang, Jeffrey Nichols. (2023)  
**AXNav: Replaying Accessibility Tests from Natural Language**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.02424v1)  

---


**ABSTRACT**  
Developers and quality assurance testers often rely on manual testing to test accessibility features throughout the product lifecycle. Unfortunately, manual testing can be tedious, often has an overwhelming scope, and can be difficult to schedule amongst other development milestones. Recently, Large Language Models (LLMs) have been used for a variety of tasks including automation of UIs, however to our knowledge no one has yet explored their use in controlling assistive technologies for the purposes of supporting accessibility testing. In this paper, we explore the requirements of a natural language based accessibility testing workflow, starting with a formative study. From this we build a system that takes as input a manual accessibility test (e.g., ``Search for a show in VoiceOver'') and uses an LLM combined with pixel-based UI Understanding models to execute the test and produce a chaptered, navigable video. In each video, to help QA testers we apply heuristics to detect and flag accessibility issues (e.g., Text size not increasing with Large Text enabled, VoiceOver navigation loops). We evaluate this system through a 10 participant user study with accessibility QA professionals who indicated that the tool would be very useful in their current work and performed tests similarly to how they would manually test the features. The study also reveals insights for future work on using LLMs for accessibility testing.

{{</citation>}}


### (103/156) Selenite: Scaffolding Decision Making with Comprehensive Overviews Elicited from Large Language Models (Michael Xieyang Liu et al., 2023)

{{<citation>}}

Michael Xieyang Liu, Tongshuang Wu, Tianying Chen, Franklin Mingzhe Li, Aniket Kittur, Brad A. Myers. (2023)  
**Selenite: Scaffolding Decision Making with Comprehensive Overviews Elicited from Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02161v1)  

---


**ABSTRACT**  
Decision-making in unfamiliar domains can be challenging, demanding considerable user effort to compare different options with respect to various criteria. Prior research and our formative study found that people would benefit from seeing an overview of the information space upfront, such as the criteria that others have previously found useful. However, existing sensemaking tools struggle with the "cold-start" problem -- it not only requires significant input from previous users to generate and share these overviews, but such overviews may also be biased and incomplete. In this work, we introduce a novel system, Selenite, which leverages LLMs as reasoning machines and knowledge retrievers to automatically produce a comprehensive overview of options and criteria to jumpstart users' sensemaking processes. Subsequently, Selenite also adapts as people use it, helping users find, read, and navigate unfamiliar information in a systematic yet personalized manner. Through three studies, we found that Selenite produced accurate and high-quality overviews reliably, significantly accelerated users' information processing, and effectively improved their overall comprehension and sensemaking experience.

{{</citation>}}


### (104/156) HPCClusterScape: Increasing Transparency and Efficiency of Shared High-Performance Computing Clusters for Large-scale AI Models (Heungseok Park et al., 2023)

{{<citation>}}

Heungseok Park, Aeree Cho, Hyojun Jeon, Hayoung Lee, Youngil Yang, Sungjae Lee, Heungsub Lee, Jaegul Choo. (2023)  
**HPCClusterScape: Increasing Transparency and Efficiency of Shared High-Performance Computing Clusters for Large-scale AI Models**  

---
Primary Category: cs.HC  
Categories: cs-DC, cs-HC, cs.HC  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.02120v1)  

---


**ABSTRACT**  
The emergence of large-scale AI models, like GPT-4, has significantly impacted academia and industry, driving the demand for high-performance computing (HPC) to accelerate workloads. To address this, we present HPCClusterScape, a visualization system that enhances the efficiency and transparency of shared HPC clusters for large-scale AI models. HPCClusterScape provides a comprehensive overview of system-level (e.g., partitions, hosts, and workload status) and application-level (e.g., identification of experiments and researchers) information, allowing HPC operators and machine learning researchers to monitor resource utilization and identify issues through customizable violation rules. The system includes diagnostic tools to investigate workload imbalances and synchronization bottlenecks in large-scale distributed deep learning experiments. Deployed in industrial-scale HPC clusters, HPCClusterScape incorporates user feedback and meets specific requirements. This paper outlines the challenges and prerequisites for efficient HPC operation, introduces the interactive visualization system, and highlights its contributions in addressing pain points and optimizing resource utilization in shared HPC clusters.

{{</citation>}}


### (105/156) A method of generating bespoke optimised keyboard layouts that significantly reduce typing effort for Twitter users (E. Elson, 2023)

{{<citation>}}

E. Elson. (2023)  
**A method of generating bespoke optimised keyboard layouts that significantly reduce typing effort for Twitter users**  

---
Primary Category: cs.HC  
Categories: cs-CY, cs-HC, cs-SI, cs.HC  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.01858v1)  

---


**ABSTRACT**  
This study addresses the problem of generating an optimised keyboard layout for single-finger typing on a smartphone. It offers Twitter users a tweet-typing experience that requires less effort and time. Bodies of tweet text for 85 popular Twitter users are used. While existing studies have produced optimisations that may generally benefit a variety of users, this study is unique in the sense that a bespoke optimised keyboard layout is generated for each Twitter user based on their own tweets, thereby uniquely benefiting them more than other users. The optimisation process is based on moving only six letter keys from their positions on the QWERTY keyboard, and therefore strikes an effective balance between the typing efficiency improvements offered by an optimised keyboard layout and the effort required to learn to use it. It is shown that a Twitter user will enjoy a reduction in typing effort of at least 13.4%. The typical user will benefit from a 15.8% reduction, while the highest typing effort reduction is nearly 25%. The method presented in this study could therefore be used in practical ways to offer any Twitter user a uniquely-improved tweeting experience.

{{</citation>}}


## cs.CR (5)



### (106/156) Jailbreaker in Jail: Moving Target Defense for Large Language Models (Bocheng Chen et al., 2023)

{{<citation>}}

Bocheng Chen, Advait Paliwal, Qiben Yan. (2023)  
**Jailbreaker in Jail: Moving Target Defense for Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02417v1)  

---


**ABSTRACT**  
Large language models (LLMs), known for their capability in understanding and following instructions, are vulnerable to adversarial attacks. Researchers have found that current commercial LLMs either fail to be "harmless" by presenting unethical answers, or fail to be "helpful" by refusing to offer meaningful answers when faced with adversarial queries. To strike a balance between being helpful and harmless, we design a moving target defense (MTD) enhanced LLM system. The system aims to deliver non-toxic answers that align with outputs from multiple model candidates, making them more robust against adversarial attacks. We design a query and output analysis model to filter out unsafe or non-responsive answers. %to achieve the two objectives of randomly selecting outputs from different LLMs. We evaluate over 8 most recent chatbot models with state-of-the-art adversarial queries. Our MTD-enhanced LLM system reduces the attack success rate from 37.5\% to 0\%. Meanwhile, it decreases the response refusal rate from 50\% to 0\%.

{{</citation>}}


### (107/156) Gotta Catch 'em All: Aggregating CVSS Scores (Angel Longueira-Romero et al., 2023)

{{<citation>}}

Angel Longueira-Romero, Jose Luis Flores, Rosa Iglesias, Iñaki Garitano. (2023)  
**Gotta Catch 'em All: Aggregating CVSS Scores**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.02062v1)  

---


**ABSTRACT**  
Security metrics are not standardized, but inter-national proposals such as the Common Vulnerability ScoringSystem (CVSS) for quantifying the severity of known vulnerabil-ities are widely used. Many CVSS aggregation mechanisms havebeen proposed in the literature. Nevertheless, factors related tothe context of the System Under Test (SUT) are not taken intoaccount in the aggregation process; vulnerabilities that in theoryaffect the SUT, but are not exploitable in reality. We propose aCVSS aggregation algorithm that integrates information aboutthe functionality disruption of the SUT, exploitation difficulty,existence of exploits, and the context where the SUT operates.The aggregation algorithm was applied to OpenPLC V3, showingthat it is capable of filtering out vulnerabilities that cannot beexploited in the real conditions of deployment of the particularsystem. Finally, because of the nature of the proposed algorithm,the result can be interpreted in the same way as a normal CVSS.

{{</citation>}}


### (108/156) Steganalysis of AI Models LSB Attacks (Daniel Gilkarov et al., 2023)

{{<citation>}}

Daniel Gilkarov, Ran Dubin. (2023)  
**Steganalysis of AI Models LSB Attacks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01969v1)  

---


**ABSTRACT**  
Artificial intelligence has made significant progress in the last decade, leading to a rise in the popularity of model sharing. The model zoo ecosystem, a repository of pre-trained AI models, has advanced the AI open-source community and opened new avenues for cyber risks. Malicious attackers can exploit shared models to launch cyber-attacks. This work focuses on the steganalysis of injected malicious Least Significant Bit (LSB) steganography into AI models, and it is the first work focusing on AI model attacks. In response to this threat, this paper presents a steganalysis method specifically tailored to detect and mitigate malicious LSB steganography attacks based on supervised and unsupervised AI detection steganalysis methods. Our proposed technique aims to preserve the integrity of shared models, protect user trust, and maintain the momentum of open collaboration within the AI community. In this work, we propose 3 steganalysis methods and open source our code. We found that the success of the steganalysis depends on the LSB attack location. If the attacker decides to exploit the least significant bits in the LSB, the ability to detect the attacks is low. However, if the attack is in the most significant LSB bits, the attack can be detected with almost perfect accuracy.

{{</citation>}}


### (109/156) Enhancing Workflow Security in Multi-Cloud Environments through Monitoring and Adaptation upon Cloud Service and Network Security Violations (Nafiseh Soveizi et al., 2023)

{{<citation>}}

Nafiseh Soveizi, Dimka Karastoyanova. (2023)  
**Enhancing Workflow Security in Multi-Cloud Environments through Monitoring and Adaptation upon Cloud Service and Network Security Violations**  

---
Primary Category: cs.CR  
Categories: H-4; D-2-9; K-6-5, cs-CR, cs-SE, cs.CR  
Keywords: Network Security, Security  
[Paper Link](http://arxiv.org/abs/2310.01878v1)  

---


**ABSTRACT**  
Cloud computing has emerged as a crucial solution for handling data- and compute-intensive workflows, offering scalability to address dynamic demands. However, ensuring the secure execution of workflows in the untrusted multi-cloud environment poses significant challenges, given the sensitive nature of the involved data and tasks. The lack of comprehensive approaches for detecting attacks during workflow execution, coupled with inadequate measures for reacting to security and privacy breaches has been identified in the literature. To close this gap, in this work, we propose an approach that focuses on monitoring cloud services and networks to detect security violations during workflow executions. Upon detection, our approach selects the optimal adaptation action to minimize the impact on the workflow. To mitigate the uncertain cost associated with such adaptations and their potential impact on other tasks in the workflow, we employ adaptive learning to determine the most suitable adaptation action. Our approach is evaluated based on the performance of the detection procedure and the impact of the selected adaptations on the workflows.

{{</citation>}}


### (110/156) Multi-class Network Intrusion Detection with Class Imbalance via LSTM & SMOTE (Muhammad Wasim Nawaz et al., 2023)

{{<citation>}}

Muhammad Wasim Nawaz, Rashid Munawar, Ahsan Mehmood, Muhammad Mahboob Ur Rahman, Qammer H. Abbasi. (2023)  
**Multi-class Network Intrusion Detection with Class Imbalance via LSTM & SMOTE**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Intrusion Detection, LSTM  
[Paper Link](http://arxiv.org/abs/2310.01850v1)  

---


**ABSTRACT**  
Monitoring network traffic to maintain the quality of service (QoS) and to detect network intrusions in a timely and efficient manner is essential. As network traffic is sequential, recurrent neural networks (RNNs) such as long short-term memory (LSTM) are suitable for building network intrusion detection systems. However, in the case of a few dataset examples of the rare attack types, even these networks perform poorly. This paper proposes to use oversampling techniques along with appropriate loss functions to handle class imbalance for the detection of various types of network intrusions. Our deep learning model employs LSTM with fully connected layers to perform multi-class classification of network attacks. We enhance the representation of minority classes: i) through the application of the Synthetic Minority Over-sampling Technique (SMOTE), and ii) by employing categorical focal cross-entropy loss to apply a focal factor to down-weight examples of the majority classes and focus more on hard examples of the minority classes. Extensive experiments on KDD99 and CICIDS2017 datasets show promising results in detecting network intrusions (with many rare attack types, e.g., Probe, R2L, DDoS, PortScan, etc.).

{{</citation>}}


## math.NA (1)



### (111/156) Computing a Sparse Approximate Inverse on Quantum Annealing Machines (Sanjay Suresh et al., 2023)

{{<citation>}}

Sanjay Suresh, Krishnan Suresh. (2023)  
**Computing a Sparse Approximate Inverse on Quantum Annealing Machines**  

---
Primary Category: math.NA  
Categories: cs-ET, cs-NA, math-NA, math.NA  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02388v1)  

---


**ABSTRACT**  
Many engineering problems involve solving large linear systems of equations. Conjugate gradient (CG) is one of the most popular iterative methods for solving such systems. However, CG typically requires a good preconditioner to speed up convergence. One such preconditioner is the sparse approximate inverse (SPAI).   In this paper, we explore the computation of an SPAI on quantum annealing machines by solving a series of quadratic unconstrained binary optimization (QUBO) problems. Numerical experiments are conducted using both well-conditioned and poorly-conditioned linear systems arising from a 2D finite difference formulation of the Poisson problem.

{{</citation>}}


## cs.CV (26)



### (112/156) ScaleNet: An Unsupervised Representation Learning Method for Limited Information (Huili Huang et al., 2023)

{{<citation>}}

Huili Huang, M. Mahdi Roozbahani. (2023)  
**ScaleNet: An Unsupervised Representation Learning Method for Limited Information**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.02386v1)  

---


**ABSTRACT**  
Although large-scale labeled data are essential for deep convolutional neural networks (ConvNets) to learn high-level semantic visual representations, it is time-consuming and impractical to collect and annotate large-scale datasets. A simple and efficient unsupervised representation learning method named ScaleNet based on multi-scale images is proposed in this study to enhance the performance of ConvNets when limited information is available. The input images are first resized to a smaller size and fed to the ConvNet to recognize the rotation degree. Next, the ConvNet learns the rotation-prediction task for the original size images based on the parameters transferred from the previous model. The CIFAR-10 and ImageNet datasets are examined on different architectures such as AlexNet and ResNet50 in this study. The current study demonstrates that specific image features, such as Harris corner information, play a critical role in the efficiency of the rotation-prediction task. The ScaleNet supersedes the RotNet by ~7% in the limited CIFAR-10 dataset. The transferred parameters from a ScaleNet model with limited data improve the ImageNet Classification task by about 6% compared to the RotNet model. This study shows the capability of the ScaleNet method to improve other cutting-edge models such as SimCLR by learning effective features for classification tasks.

{{</citation>}}


### (113/156) TransRadar: Adaptive-Directional Transformer for Real-Time Multi-View Radar Semantic Segmentation (Yahia Dalbah et al., 2023)

{{<citation>}}

Yahia Dalbah, Jean Lahoud, Hisham Cholakkal. (2023)  
**TransRadar: Adaptive-Directional Transformer for Real-Time Multi-View Radar Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2310.02260v1)  

---


**ABSTRACT**  
Scene understanding plays an essential role in enabling autonomous driving and maintaining high standards of performance and safety. To address this task, cameras and laser scanners (LiDARs) have been the most commonly used sensors, with radars being less popular. Despite that, radars remain low-cost, information-dense, and fast-sensing techniques that are resistant to adverse weather conditions. While multiple works have been previously presented for radar-based scene semantic segmentation, the nature of the radar data still poses a challenge due to the inherent noise and sparsity, as well as the disproportionate foreground and background. In this work, we propose a novel approach to the semantic segmentation of radar scenes using a multi-input fusion of radar data through a novel architecture and loss functions that are tailored to tackle the drawbacks of radar perception. Our novel architecture includes an efficient attention block that adaptively captures important feature information. Our method, TransRadar, outperforms state-of-the-art methods on the CARRADA and RADIal datasets while having smaller model sizes. https://github.com/YahiDar/TransRadar

{{</citation>}}


### (114/156) MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts (Pan Lu et al., 2023)

{{<citation>}}

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, Jianfeng Gao. (2023)  
**MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: AI, GPT, GPT-4, Language Model, OCR, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.02255v1)  

---


**ABSTRACT**  
Although Large Language Models (LLMs) and Large Multimodal Models (LMMs) exhibit impressive skills in various domains, their ability for mathematical reasoning within visual contexts has not been formally examined. Equipping LLMs and LMMs with this capability is vital for general-purpose AI assistants and showcases promising potential in education, data analysis, and scientific discovery. To bridge this gap, we present MathVista, a benchmark designed to amalgamate challenges from diverse mathematical and visual tasks. We first taxonomize the key task types, reasoning skills, and visual contexts from the literature to guide our selection from 28 existing math-focused and visual question answering datasets. Then, we construct three new datasets, IQTest, FunctionQA, and PaperQA, to accommodate for missing types of visual contexts. The problems featured often require deep visual understanding beyond OCR or image captioning, and compositional reasoning with rich domain-specific tools, thus posing a notable challenge to existing models. We conduct a comprehensive evaluation of 11 prominent open-source and proprietary foundation models (LLMs, LLMs augmented with tools, and LMMs), and early experiments with GPT-4V. The best-performing model, Multimodal Bard, achieves only 58% of human performance (34.8% vs 60.3%), indicating ample room for further improvement. Given this significant gap, MathVista fuels future research in the development of general-purpose AI agents capable of tackling mathematically intensive and visually rich real-world tasks. Preliminary tests show that MathVista also presents challenges to GPT-4V, underscoring the benchmark's importance. The project is available at https://mathvista.github.io/.

{{</citation>}}


### (115/156) MiniGPT-5: Interleaved Vision-and-Language Generation via Generative Vokens (Kaizhi Zheng et al., 2023)

{{<citation>}}

Kaizhi Zheng, Xuehai He, Xin Eric Wang. (2023)  
**MiniGPT-5: Interleaved Vision-and-Language Generation via Generative Vokens**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Dialog, GPT, GPT-5, Language Model  
[Paper Link](http://arxiv.org/abs/2310.02239v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) have garnered significant attention for their advancements in natural language processing, demonstrating unparalleled prowess in text comprehension and generation. Yet, the simultaneous generation of images with coherent textual narratives remains an evolving frontier. In response, we introduce an innovative interleaved vision-and-language generation technique anchored by the concept of "generative vokens," acting as the bridge for harmonized image-text outputs. Our approach is characterized by a distinctive two-staged training strategy focusing on description-free multimodal generation, where the training requires no comprehensive descriptions of images. To bolster model integrity, classifier-free guidance is incorporated, enhancing the effectiveness of vokens on image generation. Our model, MiniGPT-5, exhibits substantial improvement over the baseline Divter model on the MMDialog dataset and consistently delivers superior or comparable multimodal outputs in human evaluations on the VIST dataset, highlighting its efficacy across diverse benchmarks.

{{</citation>}}


### (116/156) Learnable Data Augmentation for One-Shot Unsupervised Domain Adaptation (Julio Ivan Davila Carrazco et al., 2023)

{{<citation>}}

Julio Ivan Davila Carrazco, Pietro Morerio, Alessio Del Bue, Vittorio Murino. (2023)  
**Learnable Data Augmentation for One-Shot Unsupervised Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.02201v1)  

---


**ABSTRACT**  
This paper presents a classification framework based on learnable data augmentation to tackle the One-Shot Unsupervised Domain Adaptation (OS-UDA) problem. OS-UDA is the most challenging setting in Domain Adaptation, as only one single unlabeled target sample is assumed to be available for model adaptation. Driven by such single sample, our method LearnAug-UDA learns how to augment source data, making it perceptually similar to the target. As a result, a classifier trained on such augmented data will generalize well for the target domain. To achieve this, we designed an encoder-decoder architecture that exploits a perceptual loss and style transfer strategies to augment the source data. Our method achieves state-of-the-art performance on two well-known Domain Adaptation benchmarks, DomainNet and VisDA. The project code is available at https://github.com/IIT-PAVIS/LearnAug-UDA

{{</citation>}}


### (117/156) SIEVE: Multimodal Dataset Pruning Using Image Captioning Models (Anas Mahmoud et al., 2023)

{{<citation>}}

Anas Mahmoud, Mostafa Elhoushi, Amro Abbas, Yu Yang, Newsha Ardalani, Hugh Leather, Ari Morcos. (2023)  
**SIEVE: Multimodal Dataset Pruning Using Image Captioning Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Captioning, Language Model, Pruning  
[Paper Link](http://arxiv.org/abs/2310.02110v1)  

---


**ABSTRACT**  
Vision-Language Models (VLMs) are pretrained on large, diverse, and noisy web-crawled datasets. This underscores the critical need for dataset pruning, as the quality of these datasets is strongly correlated with the performance of VLMs on downstream tasks. Using CLIPScore from a pretrained model to only train models using highly-aligned samples is one of the most successful methods for pruning.We argue that this approach suffers from multiple limitations including: 1) false positives due to spurious correlations captured by the pretrained CLIP model, 2) false negatives due to poor discrimination between hard and bad samples, and 3) biased ranking towards samples similar to the pretrained CLIP dataset. We propose a pruning method, SIEVE, that employs synthetic captions generated by image-captioning models pretrained on small, diverse, and well-aligned image-text pairs to evaluate the alignment of noisy image-text pairs. To bridge the gap between the limited diversity of generated captions and the high diversity of alternative text (alt-text), we estimate the semantic textual similarity in the embedding space of a language model pretrained on billions of sentences. Using DataComp, a multimodal dataset filtering benchmark, we achieve state-of-the-art performance on the large scale pool, and competitive results on the medium scale pool, surpassing CLIPScore-based filtering by 1.7% and 2.6% on average, on 38 downstream tasks.

{{</citation>}}


### (118/156) Leveraging Classic Deconvolution and Feature Extraction in Zero-Shot Image Restoration (Tomáš Chobola et al., 2023)

{{<citation>}}

Tomáš Chobola, Gesine Müller, Veit Dausmann, Anton Theileis, Jan Taucher, Jan Huisken, Tingying Peng. (2023)  
**Leveraging Classic Deconvolution and Feature Extraction in Zero-Shot Image Restoration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.02097v1)  

---


**ABSTRACT**  
Non-blind deconvolution aims to restore a sharp image from its blurred counterpart given an obtained kernel. Existing deep neural architectures are often built based on large datasets of sharp ground truth images and trained with supervision. Sharp, high quality ground truth images, however, are not always available, especially for biomedical applications. This severely hampers the applicability of current approaches in practice. In this paper, we propose a novel non-blind deconvolution method that leverages the power of deep learning and classic iterative deconvolution algorithms. Our approach combines a pre-trained network to extract deep features from the input image with iterative Richardson-Lucy deconvolution steps. Subsequently, a zero-shot optimisation process is employed to integrate the deconvolved features, resulting in a high-quality reconstructed image. By performing the preliminary reconstruction with the classic iterative deconvolution method, we can effectively utilise a smaller network to produce the final image, thus accelerating the reconstruction whilst reducing the demand for valuable computational resources. Our method demonstrates significant improvements in various real-world applications non-blind deconvolution tasks.

{{</citation>}}


### (119/156) Point Neighborhood Embeddings (Pedro Hermosilla, 2023)

{{<citation>}}

Pedro Hermosilla. (2023)  
**Point Neighborhood Embeddings**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.02083v1)  

---


**ABSTRACT**  
Point convolution operations rely on different embedding mechanisms to encode the neighborhood information of each point in order to detect patterns in 3D space. However, as convolutions are usually evaluated as a whole, not much work has been done to investigate which is the ideal mechanism to encode such neighborhood information. In this paper, we provide the first extensive study that analyzes such Point Neighborhood Embeddings (PNE) alone in a controlled experimental setup. From our experiments, we derive a set of recommendations for PNE that can help to improve future designs of neural network architectures for point clouds. Our most surprising finding shows that the most commonly used embedding based on a Multi-layer Perceptron (MLP) with ReLU activation functions provides the lowest performance among all embeddings, even being surpassed on some tasks by a simple linear combination of the point coordinates. Additionally, we show that a neural network architecture using simple convolutions based on such embeddings is able to achieve state-of-the-art results on several tasks, outperforming recent and more complex operations. Lastly, we show that these findings extrapolate to other more complex convolution operations, where we show how following our recommendations we are able to improve recent state-of-the-art architectures.

{{</citation>}}


### (120/156) Content Bias in Deep Learning Age Approximation: A new Approach Towards more Explainability (Robert Jöchl et al., 2023)

{{<citation>}}

Robert Jöchl, Andreas Uhl. (2023)  
**Content Bias in Deep Learning Age Approximation: A new Approach Towards more Explainability**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.02067v1)  

---


**ABSTRACT**  
In the context of temporal image forensics, it is not evident that a neural network, trained on images from different time-slots (classes), exploit solely age related features. Usually, images taken in close temporal proximity (e.g., belonging to the same age class) share some common content properties. Such content bias can be exploited by a neural network. In this work, a novel approach that evaluates the influence of image content is proposed. This approach is verified using synthetic images (where content bias can be ruled out) with an age signal embedded. Based on the proposed approach, it is shown that a `standard' neural network trained in the context of age classification is strongly dependent on image content. As a potential countermeasure, two different techniques are applied to mitigate the influence of the image content during training, and they are also evaluated by the proposed method.

{{</citation>}}


### (121/156) Constructing Image-Text Pair Dataset from Books (Yamato Okamoto et al., 2023)

{{<citation>}}

Yamato Okamoto, Haruto Toyonaga, Yoshihisa Ijiri, Hirokatsu Kataoka. (2023)  
**Constructing Image-Text Pair Dataset from Books**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2310.01936v1)  

---


**ABSTRACT**  
Digital archiving is becoming widespread owing to its effectiveness in protecting valuable books and providing knowledge to many people electronically. In this paper, we propose a novel approach to leverage digital archives for machine learning. If we can fully utilize such digitized data, machine learning has the potential to uncover unknown insights and ultimately acquire knowledge autonomously, just like humans read books. As a first step, we design a dataset construction pipeline comprising an optical character reader (OCR), an object detector, and a layout analyzer for the autonomous extraction of image-text pairs. In our experiments, we apply our pipeline on old photo books to construct an image-text pair dataset, showing its effectiveness in image-text retrieval and insight extraction.

{{</citation>}}


### (122/156) MarineDet: Towards Open-Marine Object Detection (Liang Haixin et al., 2023)

{{<citation>}}

Liang Haixin, Zheng Ziqiang, Ma Zeyu, Sai-Kit Yeung. (2023)  
**MarineDet: Towards Open-Marine Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.01931v1)  

---


**ABSTRACT**  
Marine object detection has gained prominence in marine research, driven by the pressing need to unravel oceanic mysteries and enhance our understanding of invaluable marine ecosystems. There is a profound requirement to efficiently and accurately identify and localize diverse and unseen marine entities within underwater imagery. The open-marine object detection (OMOD for short) is required to detect diverse and unseen marine objects, performing categorization and localization simultaneously. To achieve OMOD, we present \textbf{MarineDet}. We formulate a joint visual-text semantic space through pre-training and then perform marine-specific training to achieve in-air-to-marine knowledge transfer. Considering there is no specific dataset designed for OMOD, we construct a \textbf{MarineDet dataset} consisting of 821 marine-relative object categories to promote and measure OMOD performance. The experimental results demonstrate the superior performance of MarineDet over existing generalist and specialist object detection algorithms. To the best of our knowledge, we are the first to present OMOD, which holds a more valuable and practical setting for marine ecosystem monitoring and management. Our research not only pushes the boundaries of marine understanding but also offers a standard pipeline for OMOD.

{{</citation>}}


### (123/156) RoFormer for Position Aware Multiple Instance Learning in Whole Slide Image Classification (Etienne Pochet et al., 2023)

{{<citation>}}

Etienne Pochet, Rami Maroun, Roger Trullo. (2023)  
**RoFormer for Position Aware Multiple Instance Learning in Whole Slide Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2310.01924v1)  

---


**ABSTRACT**  
Whole slide image (WSI) classification is a critical task in computational pathology. However, the gigapixel-size of such images remains a major challenge for the current state of deep-learning. Current methods rely on multiple-instance learning (MIL) models with frozen feature extractors. Given the the high number of instances in each image, MIL methods have long assumed independence and permutation-invariance of patches, disregarding the tissue structure and correlation between patches. Recent works started studying this correlation between instances but the computational workload of such a high number of tokens remained a limiting factor. In particular, relative position of patches remains unaddressed. We propose to apply a straightforward encoding module, namely a RoFormer layer , relying on memory-efficient exact self-attention and relative positional encoding. This module can perform full self-attention with relative position encoding on patches of large and arbitrary shaped WSIs, solving the need for correlation between instances and spatial modeling of tissues. We demonstrate that our method outperforms state-of-the-art MIL models on three commonly used public datasets (TCGA-NSCLC, BRACS and Camelyon16)) on weakly supervised classification tasks. Code is available at https://github.com/Sanofi-Public/DDS-RoFormerMIL

{{</citation>}}


### (124/156) CLIP Is Also a Good Teacher: A New Learning Framework for Inductive Zero-shot Semantic Segmentation (Jialei Chen et al., 2023)

{{<citation>}}

Jialei Chen, Daisuke Deguchi, Chenkai Zhang, Xu Zheng, Hiroshi Murase. (2023)  
**CLIP Is Also a Good Teacher: A New Learning Framework for Inductive Zero-shot Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GLM, Language Model, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.02296v1)  

---


**ABSTRACT**  
Existing Generalized Zero-shot Semantic Segmentation (GZLSS) methods apply either finetuning the CLIP paradigm or formulating it as a mask classification task, benefiting from the Vision-Language Models (VLMs). However, the fine-tuning methods are restricted with fixed backbone models which are not flexible for segmentation, and mask classification methods heavily rely on additional explicit mask proposers. Meanwhile, prevalent methods utilize only seen categories which is a great waste, i.e., neglecting the area exists but not annotated. To this end, we propose CLIPTeacher, a new learning framework that can be applied to various per-pixel classification segmentation models without introducing any explicit mask proposer or changing the structure of CLIP, and utilize both seen and ignoring areas. Specifically, CLIPTeacher consists of two key modules: Global Learning Module (GLM) and Pixel Learning Module (PLM). Specifically, GLM aligns the dense features from an image encoder with the CLS token, i.e., the only token trained in CLIP, which is a simple but effective way to probe global information from the CLIP models. In contrast, PLM only leverages dense tokens from CLIP to produce high-level pseudo annotations for ignoring areas without introducing any extra mask proposer. Meanwhile, PLM can fully take advantage of the whole image based on the pseudo annotations. Experimental results on three benchmark datasets: PASCAL VOC 2012, COCO-Stuff 164k, and PASCAL Context show large performance gains, i.e., 2.2%, 1.3%, and 8.8%

{{</citation>}}


### (125/156) Beyond the Benchmark: Detecting Diverse Anomalies in Videos (Yoav Arad et al., 2023)

{{<citation>}}

Yoav Arad, Michael Werman. (2023)  
**Beyond the Benchmark: Detecting Diverse Anomalies in Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI, Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.01904v1)  

---


**ABSTRACT**  
Video Anomaly Detection (VAD) plays a crucial role in modern surveillance systems, aiming to identify various anomalies in real-world situations. However, current benchmark datasets predominantly emphasize simple, single-frame anomalies such as novel object detection. This narrow focus restricts the advancement of VAD models. In this research, we advocate for an expansion of VAD investigations to encompass intricate anomalies that extend beyond conventional benchmark boundaries. To facilitate this, we introduce two datasets, HMDB-AD and HMDB-Violence, to challenge models with diverse action-based anomalies. These datasets are derived from the HMDB51 action recognition dataset. We further present Multi-Frame Anomaly Detection (MFAD), a novel method built upon the AI-VAD framework. AI-VAD utilizes single-frame features such as pose estimation and deep image encoding, and two-frame features such as object velocity. They then apply a density estimation algorithm to compute anomaly scores. To address complex multi-frame anomalies, we add a deep video encoding features capturing long-range temporal dependencies, and logistic regression to enhance final score calculation. Experimental results confirm our assumptions, highlighting existing models limitations with new anomaly types. MFAD excels in both simple and complex anomaly detection scenarios.

{{</citation>}}


### (126/156) Zero-Shot Refinement of Buildings' Segmentation Models using SAM (Ali Mayladan et al., 2023)

{{<citation>}}

Ali Mayladan, Hasan Nasrallah, Hasan Moughnieh, Mustafa Shukor, Ali J. Ghandour. (2023)  
**Zero-Shot Refinement of Buildings' Segmentation Models using SAM**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: AI, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.01845v1)  

---


**ABSTRACT**  
Foundation models have excelled in various tasks but are often evaluated on general benchmarks. The adaptation of these models for specific domains, such as remote sensing imagery, remains an underexplored area. In remote sensing, precise building instance segmentation is vital for applications like urban planning. While Convolutional Neural Networks (CNNs) perform well, their generalization can be limited. For this aim, we present a novel approach to adapt foundation models to address existing models' generalization dropback. Among several models, our focus centers on the Segment Anything Model (SAM), a potent foundation model renowned for its prowess in class-agnostic image segmentation capabilities. We start by identifying the limitations of SAM, revealing its suboptimal performance when applied to remote sensing imagery. Moreover, SAM does not offer recognition abilities and thus fails to classify and tag localized objects. To address these limitations, we introduce different prompting strategies, including integrating a pre-trained CNN as a prompt generator. This novel approach augments SAM with recognition abilities, a first of its kind. We evaluated our method on three remote sensing datasets, including the WHU Buildings dataset, the Massachusetts Buildings dataset, and the AICrowd Mapping Challenge. For out-of-distribution performance on the WHU dataset, we achieve a 5.47% increase in IoU and a 4.81% improvement in F1-score. For in-distribution performance on the WHU dataset, we observe a 2.72% and 1.58% increase in True-Positive-IoU and True-Positive-F1 score, respectively. We intend to release our code repository, hoping to inspire further exploration of foundation models for domain-specific tasks within the remote sensing community.

{{</citation>}}


### (127/156) Selective Feature Adapter for Dense Vision Transformers (Xueqing Deng et al., 2023)

{{<citation>}}

Xueqing Deng, Qi Fan, Xiaojie Jin, Linjie Yang, Peng Wang. (2023)  
**Selective Feature Adapter for Dense Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.01843v1)  

---


**ABSTRACT**  
Fine-tuning pre-trained transformer models, e.g., Swin Transformer, are successful in numerous downstream for dense prediction vision tasks. However, one major issue is the cost/storage of their huge amount of parameters, which becomes increasingly challenging to handle with the growing amount of vision tasks. In this paper, we propose an effective approach to alleviate the issue, namely selective feature adapter (SFA). It achieves state-of-the-art (SoTA) performance under any given budget of trainable parameters, and demonstrates comparable or better performance than fully fine-tuned models across various dense tasks. Specifically, SFA consists of external adapters and internal adapters which are sequentially operated over a transformer model. For external adapters, we properly select the places and amount of additional multilayer perception (MLP). For internal adapters, we transform a few task-important parameters inside the transformer, which are automatically discovered through a simple yet effective lottery ticket algorithm. Our experiments show that the dual adapter module, a.k.a SFA, is essential to achieve the best trade-off on dense vision tasks, such as segmentation, detection and depth-estimation, outperforming other adapters with a single module.

{{</citation>}}


### (128/156) SelfGraphVQA: A Self-Supervised Graph Neural Network for Scene-based Question Answering (Bruno Souza et al., 2023)

{{<citation>}}

Bruno Souza, Marius Aasan, Helio Pedrini, Adín Ramírez Rivera. (2023)  
**SelfGraphVQA: A Self-Supervised Graph Neural Network for Scene-based Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Graph Neural Network, QA, Question Answering, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.01842v1)  

---


**ABSTRACT**  
The intersection of vision and language is of major interest due to the increased focus on seamless integration between recognition and reasoning. Scene graphs (SGs) have emerged as a useful tool for multimodal image analysis, showing impressive performance in tasks such as Visual Question Answering (VQA). In this work, we demonstrate that despite the effectiveness of scene graphs in VQA tasks, current methods that utilize idealized annotated scene graphs struggle to generalize when using predicted scene graphs extracted from images. To address this issue, we introduce the SelfGraphVQA framework. Our approach extracts a scene graph from an input image using a pre-trained scene graph generator and employs semantically-preserving augmentation with self-supervised techniques. This method improves the utilization of graph representations in VQA tasks by circumventing the need for costly and potentially biased annotated data. By creating alternative views of the extracted graphs through image augmentations, we can learn joint embeddings by optimizing the informational content in their representations using an un-normalized contrastive approach. As we work with SGs, we experiment with three distinct maximization strategies: node-wise, graph-wise, and permutation-equivariant regularization. We empirically showcase the effectiveness of the extracted scene graph for VQA and demonstrate that these approaches enhance overall performance by highlighting the significance of visual information. This offers a more practical solution for VQA tasks that rely on SGs for complex reasoning questions.

{{</citation>}}


### (129/156) Self-Supervised High Dynamic Range Imaging with Multi-Exposure Images in Dynamic Scenes (Zhilu Zhang et al., 2023)

{{<citation>}}

Zhilu Zhang, Haoyu Wang, Shuai Liu, Xiaotao Wang, Lei Lei, Wangmeng Zuo. (2023)  
**Self-Supervised High Dynamic Range Imaging with Multi-Exposure Images in Dynamic Scenes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.01840v1)  

---


**ABSTRACT**  
Merging multi-exposure images is a common approach for obtaining high dynamic range (HDR) images, with the primary challenge being the avoidance of ghosting artifacts in dynamic scenes. Recent methods have proposed using deep neural networks for deghosting. However, the methods typically rely on sufficient data with HDR ground-truths, which are difficult and costly to collect. In this work, to eliminate the need for labeled data, we propose SelfHDR, a self-supervised HDR reconstruction method that only requires dynamic multi-exposure images during training. Specifically, SelfHDR learns a reconstruction network under the supervision of two complementary components, which can be constructed from multi-exposure images and focus on HDR color as well as structure, respectively. The color component is estimated from aligned multi-exposure images, while the structure one is generated through a structure-focused network that is supervised by the color component and an input reference (\eg, medium-exposure) image. During testing, the learned reconstruction network is directly deployed to predict an HDR image. Experiments on real-world images demonstrate our SelfHDR achieves superior results against the state-of-the-art self-supervised methods, and comparable performance to supervised ones. Codes are available at https://github.com/cszhilu1998/SelfHDR

{{</citation>}}


### (130/156) Extending CAM-based XAI methods for Remote Sensing Imagery Segmentation (Abdul Karim Gizzini et al., 2023)

{{<citation>}}

Abdul Karim Gizzini, Mustafa Shukor, Ali J. Ghandour. (2023)  
**Extending CAM-based XAI methods for Remote Sensing Imagery Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01837v1)  

---


**ABSTRACT**  
Current AI-based methods do not provide comprehensible physical interpretations of the utilized data, extracted features, and predictions/inference operations. As a result, deep learning models trained using high-resolution satellite imagery lack transparency and explainability and can be merely seen as a black box, which limits their wide-level adoption. Experts need help understanding the complex behavior of AI models and the underlying decision-making process. The explainable artificial intelligence (XAI) field is an emerging field providing means for robust, practical, and trustworthy deployment of AI models. Several XAI techniques have been proposed for image classification tasks, whereas the interpretation of image segmentation remains largely unexplored. This paper offers to bridge this gap by adapting the recent XAI classification algorithms and making them usable for muti-class image segmentation, where we mainly focus on buildings' segmentation from high-resolution satellite images. To benchmark and compare the performance of the proposed approaches, we introduce a new XAI evaluation methodology and metric based on "Entropy" to measure the model uncertainty. Conventional XAI evaluation methods rely mainly on feeding area-of-interest regions from the image back to the pre-trained (utility) model and then calculating the average change in the probability of the target class. Those evaluation metrics lack the needed robustness, and we show that using Entropy to monitor the model uncertainty in segmenting the pixels within the target class is more suitable. We hope this work will pave the way for additional XAI research for image segmentation and applications in the remote sensing discipline.

{{</citation>}}


### (131/156) AI-Generated Images as Data Source: The Dawn of Synthetic Era (Zuhao Yang et al., 2023)

{{<citation>}}

Zuhao Yang, Fangneng Zhan, Kunhao Liu, Muyu Xu, Shijian Lu. (2023)  
**AI-Generated Images as Data Source: The Dawn of Synthetic Era**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01830v1)  

---


**ABSTRACT**  
The advancement of visual intelligence is intrinsically tethered to the availability of data. In parallel, generative Artificial Intelligence (AI) has unlocked the potential to create synthetic images that closely resemble real-world photographs, which prompts a compelling inquiry: how visual intelligence benefit from the advance of generative AI? This paper explores the innovative concept of harnessing these AI-generated images as a new data source, reshaping traditional model paradigms in visual intelligence. In contrast to real data, AI-generated data sources exhibit remarkable advantages, including unmatched abundance and scalability, the rapid generation of vast datasets, and the effortless simulation of edge cases. Built on the success of generative AI models, we examines the potential of their generated data in a range of applications, from training machine learning models to simulating scenarios for computational modelling, testing, and validation. We probe the technological foundations that support this groundbreaking use of generative AI, engaging in an in-depth discussion on the ethical, legal, and practical considerations that accompany this transformative paradigm shift. Through an exhaustive survey of current technologies and applications, this paper presents a comprehensive view of the synthetic era in visual intelligence. A project with this paper can be found at https://github.com/mwxely/AIGS .

{{</citation>}}


### (132/156) Trainable Noise Model as an XAI evaluation method: application on Sobol for remote sensing image segmentation (Hossein Shreim et al., 2023)

{{<citation>}}

Hossein Shreim, Abdul Karim Gizzini, Ali J. Ghandour. (2023)  
**Trainable Noise Model as an XAI evaluation method: application on Sobol for remote sensing image segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01828v1)  

---


**ABSTRACT**  
eXplainable Artificial Intelligence (XAI) has emerged as an essential requirement when dealing with mission-critical applications, ensuring transparency and interpretability of the employed black box AI models. The significance of XAI spans various domains, from healthcare to finance, where understanding the decision-making process of deep learning algorithms is essential. Most AI-based computer vision models are often black boxes; hence, providing explainability of deep neural networks in image processing is crucial for their wide adoption and deployment in medical image analysis, autonomous driving, and remote sensing applications. Recently, several XAI methods for image classification tasks have been introduced. On the contrary, image segmentation has received comparatively less attention in the context of explainability, although it is a fundamental task in computer vision applications, especially in remote sensing. Only some research proposes gradient-based XAI algorithms for image segmentation. This paper adapts the recent gradient-free Sobol XAI method for semantic segmentation. To measure the performance of the Sobol method for segmentation, we propose a quantitative XAI evaluation method based on a learnable noise model. The main objective of this model is to induce noise on the explanation maps, where higher induced noise signifies low accuracy and vice versa. A benchmark analysis is conducted to evaluate and compare performance of three XAI methods, including Seg-Grad-CAM, Seg-Grad-CAM++ and Seg-Sobol using the proposed noise-based evaluation technique. This constitutes the first attempt to run and evaluate XAI methods using high-resolution satellite images.

{{</citation>}}


### (133/156) Amazing Combinatorial Creation: Acceptable Swap-Sampling for Text-to-Image Generation (Jun Li et al., 2023)

{{<citation>}}

Jun Li, Zedong Zhang, Jian Yang. (2023)  
**Amazing Combinatorial Creation: Acceptable Swap-Sampling for Text-to-Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.01819v1)  

---


**ABSTRACT**  
Exploring a machine learning system to generate meaningful combinatorial object images from multiple textual descriptions, emulating human creativity, is a significant challenge as humans are able to construct amazing combinatorial objects, but machines strive to emulate data distribution. In this paper, we develop a straightforward yet highly effective technique called acceptable swap-sampling to generate a combinatorial object image that exhibits novelty and surprise, utilizing text concepts of different objects. Initially, we propose a swapping mechanism that constructs a novel embedding by exchanging column vectors of two text embeddings for generating a new combinatorial image through a cutting-edge diffusion model. Furthermore, we design an acceptable region by managing suitable CLIP distances between the new image and the original concept generations, increasing the likelihood of accepting the new image with a high-quality combination. This region allows us to efficiently sample a small subset from a new image pool generated by using randomly exchanging column vectors. Lastly, we employ a segmentation method to compare CLIP distances among the segmented components, ultimately selecting the most promising object image from the sampled subset. Our experiments focus on text pairs of objects from ImageNet, and our results demonstrate that our approach outperforms recent methods such as Stable-Diffusion2, DALLE2, ERNIE-ViLG2 and Bing in generating novel and surprising object images, even when the associated concepts appear to be implausible, such as lionfish-abacus. Furthermore, during the sampling process, our approach without training and human preference is also comparable to PickScore and HPSv2 trained using human preference datasets.

{{</citation>}}


### (134/156) PPT: Token Pruning and Pooling for Efficient Vision Transformers (Xinjian Wu et al., 2023)

{{<citation>}}

Xinjian Wu, Fanhu Zeng, Xiudong Wang, Yunhe Wang, Xinghao Chen. (2023)  
**PPT: Token Pruning and Pooling for Efficient Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Pruning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.01812v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) have emerged as powerful models in the field of computer vision, delivering superior performance across various vision tasks. However, the high computational complexity poses a significant barrier to their practical applications in real-world scenarios. Motivated by the fact that not all tokens contribute equally to the final predictions and fewer tokens bring less computational cost, reducing redundant tokens has become a prevailing paradigm for accelerating vision transformers. However, we argue that it is not optimal to either only reduce inattentive redundancy by token pruning, or only reduce duplicative redundancy by token merging. To this end, in this paper we propose a novel acceleration framework, namely token Pruning & Pooling Transformers (PPT), to adaptively tackle these two types of redundancy in different layers. By heuristically integrating both token pruning and token pooling techniques in ViTs without additional trainable parameters, PPT effectively reduces the model complexity while maintaining its predictive accuracy. For example, PPT reduces over 37% FLOPs and improves the throughput by over 45% for DeiT-S without any accuracy drop on the ImageNet dataset.

{{</citation>}}


### (135/156) Improvement and Enhancement of YOLOv5 Small Target Recognition Based on Multi-module Optimization (Qingyang Li et al., 2023)

{{<citation>}}

Qingyang Li, Yuchen Li, Hongyi Duan, JiaLiang Kang, Jianan Zhang, Xueqian Gan, Ruotong Xu. (2023)  
**Improvement and Enhancement of YOLOv5 Small Target Recognition Based on Multi-module Optimization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.01806v1)  

---


**ABSTRACT**  
In this paper, the limitations of YOLOv5s model on small target detection task are deeply studied and improved. The performance of the model is successfully enhanced by introducing GhostNet-based convolutional module, RepGFPN-based Neck module optimization, CA and Transformer's attention mechanism, and loss function improvement using NWD. The experimental results validate the positive impact of these improvement strategies on model precision, recall and mAP. In particular, the improved model shows significant superiority in dealing with complex backgrounds and tiny targets in real-world application tests. This study provides an effective optimization strategy for the YOLOv5s model on small target detection, and lays a solid foundation for future related research and applications.

{{</citation>}}


### (136/156) HallE-Switch: Rethinking and Controlling Object Existence Hallucinations in Large Vision Language Models for Detailed Caption (Bohan Zhai et al., 2023)

{{<citation>}}

Bohan Zhai, Shijia Yang, Xiangchen Zhao, Chenfeng Xu, Sheng Shen, Dongdi Zhao, Kurt Keutzer, Manling Li, Tan Yan, Xiangjun Fan. (2023)  
**HallE-Switch: Rethinking and Controlling Object Existence Hallucinations in Large Vision Language Models for Detailed Caption**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.01779v1)  

---


**ABSTRACT**  
Current large vision-language models (LVLMs) achieve remarkable progress, yet there remains significant uncertainty regarding their ability to accurately apprehend visual details, that is, in performing detailed captioning. To address this, we introduce \textit{CCEval}, a GPT-4 assisted evaluation method tailored for detailed captioning. Interestingly, while LVLMs demonstrate minimal object existence hallucination in existing VQA benchmarks, our proposed evaluation reveals continued susceptibility to such hallucinations. In this paper, we make the first attempt to investigate and attribute such hallucinations, including image resolution, the language decoder size, and instruction data amount, quality, granularity. Our findings underscore the unwarranted inference when the language description includes details at a finer object granularity than what the vision module can ground or verify, thus inducing hallucination. To control such hallucinations, we further attribute the reliability of captioning to contextual knowledge (involving only contextually grounded objects) and parametric knowledge (containing inferred objects by the model). Thus, we introduce $\textit{HallE-Switch}$, a controllable LVLM in terms of $\textbf{Hall}$ucination in object $\textbf{E}$xistence. HallE-Switch can condition the captioning to shift between (i) exclusively depicting contextual knowledge for grounded objects and (ii) blending it with parametric knowledge to imagine inferred objects. Our method reduces hallucination by 44% compared to LLaVA$_{7B}$ and maintains the same object coverage.

{{</citation>}}


### (137/156) ImageNet-OOD: Deciphering Modern Out-of-Distribution Detection Algorithms (William Yang et al., 2023)

{{<citation>}}

William Yang, Byron Zhang, Olga Russakovsky. (2023)  
**ImageNet-OOD: Deciphering Modern Out-of-Distribution Detection Algorithms**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.01755v1)  

---


**ABSTRACT**  
The task of out-of-distribution (OOD) detection is notoriously ill-defined. Earlier works focused on new-class detection, aiming to identify label-altering data distribution shifts, also known as "semantic shift." However, recent works argue for a focus on failure detection, expanding the OOD evaluation framework to account for label-preserving data distribution shifts, also known as "covariate shift." Intriguingly, under this new framework, complex OOD detectors that were previously considered state-of-the-art now perform similarly to, or even worse than the simple maximum softmax probability baseline. This raises the question: what are the latest OOD detectors actually detecting? Deciphering the behavior of OOD detection algorithms requires evaluation datasets that decouples semantic shift and covariate shift. To aid our investigations, we present ImageNet-OOD, a clean semantic shift dataset that minimizes the interference of covariate shift. Through comprehensive experiments, we show that OOD detectors are more sensitive to covariate shift than to semantic shift, and the benefits of recent OOD detection algorithms on semantic shift detection is minimal. Our dataset and analyses provide important insights for guiding the design of future OOD detectors.

{{</citation>}}


## cs.AI (10)



### (138/156) Prioritized Soft Q-Decomposition for Lexicographic Reinforcement Learning (Finn Rietz et al., 2023)

{{<citation>}}

Finn Rietz, Stefan Heinrich, Erik Schaffernicht, Johannes Andreas Stork. (2023)  
**Prioritized Soft Q-Decomposition for Lexicographic Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.02360v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) for complex tasks remains a challenge, primarily due to the difficulties of engineering scalar reward functions and the inherent inefficiency of training models from scratch. Instead, it would be better to specify complex tasks in terms of elementary subtasks and to reuse subtask solutions whenever possible. In this work, we address continuous space lexicographic multi-objective RL problems, consisting of prioritized subtasks, which are notoriously difficult to solve. We show that these can be scalarized with a subtask transformation and then solved incrementally using value decomposition. Exploiting this insight, we propose prioritized soft Q-decomposition (PSQD), a novel algorithm for learning and adapting subtask solutions under lexicographic priorities in continuous state-action spaces. PSQD offers the ability to reuse previously learned subtask solutions in a zero-shot composition, followed by an adaptation step. Its ability to use retained subtask training data for offline learning eliminates the need for new environment interaction during adaptation. We demonstrate the efficacy of our approach by presenting successful learning, reuse, and adaptation results for both low- and high-dimensional simulated robot control tasks, as well as offline learning results. In contrast to baseline approaches, PSQD does not trade off between conflicting subtasks or priority constraints and satisfies subtask priorities during learning. PSQD provides an intuitive framework for tackling complex RL problems, offering insights into the inner workings of the subtask composition.

{{</citation>}}


### (139/156) Towards a Unified Framework for Sequential Decision Making (Carlos Núñez-Molina et al., 2023)

{{<citation>}}

Carlos Núñez-Molina, Pablo Mesejo, Juan Fernández-Olivares. (2023)  
**Towards a Unified Framework for Sequential Decision Making**  

---
Primary Category: cs.AI  
Categories: I-2-8, cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.02167v1)  

---


**ABSTRACT**  
In recent years, the integration of Automated Planning (AP) and Reinforcement Learning (RL) has seen a surge of interest. To perform this integration, a general framework for Sequential Decision Making (SDM) would prove immensely useful, as it would help us understand how AP and RL fit together. In this preliminary work, we attempt to provide such a framework, suitable for any method ranging from Classical Planning to Deep RL, by drawing on concepts from Probability Theory and Bayesian inference. We formulate an SDM task as a set of training and test Markov Decision Processes (MDPs), to account for generalization. We provide a general algorithm for SDM which we hypothesize every SDM method is based on. According to it, every SDM algorithm can be seen as a procedure that iteratively improves its solution estimate by leveraging the task knowledge available. Finally, we derive a set of formulas and algorithms for calculating interesting properties of SDM tasks and methods, which make possible their empirical evaluation and comparison.

{{</citation>}}


### (140/156) Learning Reliable Logical Rules with SATNet (Zhaoyu Li et al., 2023)

{{<citation>}}

Zhaoyu Li, Jinpei Guo, Yuhe Jiang, Xujie Si. (2023)  
**Learning Reliable Logical Rules with SATNet**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02133v1)  

---


**ABSTRACT**  
Bridging logical reasoning and deep learning is crucial for advanced AI systems. In this work, we present a new framework that addresses this goal by generating interpretable and verifiable logical rules through differentiable learning, without relying on pre-specified logical structures. Our approach builds upon SATNet, a differentiable MaxSAT solver that learns the underlying rules from input-output examples. Despite its efficacy, the learned weights in SATNet are not straightforwardly interpretable, failing to produce human-readable rules. To address this, we propose a novel specification method called "maximum equality", which enables the interchangeability between the learned weights of SATNet and a set of propositional logical rules in weighted MaxSAT form. With the decoded weighted MaxSAT formula, we further introduce several effective verification techniques to validate it against the ground truth rules. Experiments on stream transformations and Sudoku problems show that our decoded rules are highly reliable: using exact solvers on them could achieve 100% accuracy, whereas the original SATNet fails to give correct solutions in many cases. Furthermore, we formally verify that our decoded logical rules are functionally equivalent to the ground truth ones.

{{</citation>}}


### (141/156) Towards Effective Human-AI Decision-Making: The Role of Human Learning in Appropriate Reliance on AI Advice (Max Schemmer et al., 2023)

{{<citation>}}

Max Schemmer, Andrea Bartos, Philipp Spitzer, Patrick Hemmer, Niklas Kühl, Jonas Liebschner, Gerhard Satzger. (2023)  
**Towards Effective Human-AI Decision-Making: The Role of Human Learning in Appropriate Reliance on AI Advice**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02108v1)  

---


**ABSTRACT**  
The true potential of human-AI collaboration lies in exploiting the complementary capabilities of humans and AI to achieve a joint performance superior to that of the individual AI or human, i.e., to achieve complementary team performance (CTP). To realize this complementarity potential, humans need to exercise discretion in following AI 's advice, i.e., appropriately relying on the AI's advice. While previous work has focused on building a mental model of the AI to assess AI recommendations, recent research has shown that the mental model alone cannot explain appropriate reliance. We hypothesize that, in addition to the mental model, human learning is a key mediator of appropriate reliance and, thus, CTP. In this study, we demonstrate the relationship between learning and appropriate reliance in an experiment with 100 participants. This work provides fundamental concepts for analyzing reliance and derives implications for the effective design of human-AI decision-making.

{{</citation>}}


### (142/156) Towards End-to-End Embodied Decision Making via Multi-modal Large Language Model: Explorations with GPT4-Vision and Beyond (Liang Chen et al., 2023)

{{<citation>}}

Liang Chen, Yichi Zhang, Shuhuai Ren, Haozhe Zhao, Zefan Cai, Yuchi Wang, Tianyu Liu, Baobao Chang. (2023)  
**Towards End-to-End Embodied Decision Making via Multi-modal Large Language Model: Explorations with GPT4-Vision and Beyond**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs-RO, cs.AI  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.02071v1)  

---


**ABSTRACT**  
In this study, we explore the potential of Multimodal Large Language Models (MLLMs) in improving embodied decision-making processes for agents. While Large Language Models (LLMs) have been widely used due to their advanced reasoning skills and vast world knowledge, MLLMs like GPT4-Vision offer enhanced visual understanding and reasoning capabilities. We investigate whether state-of-the-art MLLMs can handle embodied decision-making in an end-to-end manner and whether collaborations between LLMs and MLLMs can enhance decision-making. To address these questions, we introduce a new benchmark called PCA-EVAL, which evaluates embodied decision-making from the perspectives of Perception, Cognition, and Action. Additionally, we propose HOLMES, a multi-agent cooperation framework that allows LLMs to leverage MLLMs and APIs to gather multimodal information for informed decision-making. We compare end-to-end embodied decision-making and HOLMES on our benchmark and find that the GPT4-Vision model demonstrates strong end-to-end embodied decision-making abilities, outperforming GPT4-HOLMES in terms of average decision accuracy (+3%). However, this performance is exclusive to the latest GPT4-Vision model, surpassing the open-source state-of-the-art MLLM by 26%. Our results indicate that powerful MLLMs like GPT4-Vision hold promise for decision-making in embodied agents, offering new avenues for MLLM research.

{{</citation>}}


### (143/156) AlignDiff: Aligning Diverse Human Preferences via Behavior-Customisable Diffusion Model (Zibin Dong et al., 2023)

{{<citation>}}

Zibin Dong, Yifu Yuan, Jianye Hao, Fei Ni, Yao Mu, Yan Zheng, Yujing Hu, Tangjie Lv, Changjie Fan, Zhipeng Hu. (2023)  
**AlignDiff: Aligning Diverse Human Preferences via Behavior-Customisable Diffusion Model**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02054v1)  

---


**ABSTRACT**  
Aligning agent behaviors with diverse human preferences remains a challenging problem in reinforcement learning (RL), owing to the inherent abstractness and mutability of human preferences. To address these issues, we propose AlignDiff, a novel framework that leverages RL from Human Feedback (RLHF) to quantify human preferences, covering abstractness, and utilizes them to guide diffusion planning for zero-shot behavior customizing, covering mutability. AlignDiff can accurately match user-customized behaviors and efficiently switch from one to another. To build the framework, we first establish the multi-perspective human feedback datasets, which contain comparisons for the attributes of diverse behaviors, and then train an attribute strength model to predict quantified relative strengths. After relabeling behavioral datasets with relative strengths, we proceed to train an attribute-conditioned diffusion model, which serves as a planner with the attribute strength model as a director for preference aligning at the inference phase. We evaluate AlignDiff on various locomotion tasks and demonstrate its superior performance on preference matching, switching, and covering compared to other baselines. Its capability of completing unseen downstream tasks under human instructions also showcases the promising potential for human-AI collaboration. More visualization videos are released on https://aligndiff.github.io/.

{{</citation>}}


### (144/156) Towards Feasible Counterfactual Explanations: A Taxonomy Guided Template-based NLG Method (Pedram Salimi et al., 2023)

{{<citation>}}

Pedram Salimi, Nirmalie Wiratunga, David Corsar, Anjana Wijekoon. (2023)  
**Towards Feasible Counterfactual Explanations: A Taxonomy Guided Template-based NLG Method**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02019v1)  

---


**ABSTRACT**  
Counterfactual Explanations (cf-XAI) describe the smallest changes in feature values necessary to change an outcome from one class to another. However, many cf-XAI methods neglect the feasibility of those changes. In this paper, we introduce a novel approach for presenting cf-XAI in natural language (Natural-XAI), giving careful consideration to actionable and comprehensible aspects while remaining cognizant of immutability and ethical concerns. We present three contributions to this endeavor. Firstly, through a user study, we identify two types of themes present in cf-XAI composed by humans: content-related, focusing on how features and their values are included from both the counterfactual and the query perspectives; and structure-related, focusing on the structure and terminology used for describing necessary value changes. Secondly, we introduce a feature actionability taxonomy with four clearly defined categories, to streamline the explanation presentation process. Using insights from the user study and our taxonomy, we created a generalisable template-based natural language generation (NLG) method compatible with existing explainers like DICE, NICE, and DisCERN, to produce counterfactuals that address the aforementioned limitations of existing approaches. Finally, we conducted a second user study to assess the performance of our taxonomy-guided NLG templates on three domains. Our findings show that the taxonomy-guided Natural-XAI approach (n-XAI^T) received higher user ratings across all dimensions, with significantly improved results in the majority of the domains assessed for articulation, acceptability, feasibility, and sensitivity dimensions.

{{</citation>}}


### (145/156) Fine-tuned vs. Prompt-tuned Supervised Representations: Which Better Account for Brain Language Representations? (Jingyuan Sun et al., 2023)

{{<citation>}}

Jingyuan Sun, Marie-Francine Moens. (2023)  
**Fine-tuned vs. Prompt-tuned Supervised Representations: Which Better Account for Brain Language Representations?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: NLU  
[Paper Link](http://arxiv.org/abs/2310.01854v1)  

---


**ABSTRACT**  
To decipher the algorithm underlying the human brain's language representation, previous work probed brain responses to language input with pre-trained artificial neural network (ANN) models fine-tuned on NLU tasks. However, full fine-tuning generally updates the entire parametric space and distorts pre-trained features, cognitively inconsistent with the brain's robust multi-task learning ability. Prompt-tuning, in contrast, protects pre-trained weights and learns task-specific embeddings to fit a task. Could prompt-tuning generate representations that better account for the brain's language representations than fine-tuning? If so, what kind of NLU task leads a pre-trained model to better decode the information represented in the human brain? We investigate these questions by comparing prompt-tuned and fine-tuned representations in neural decoding, that is predicting the linguistic stimulus from the brain activities evoked by the stimulus. We find that on none of the 10 NLU tasks, full fine-tuning significantly outperforms prompt-tuning in neural decoding, implicating that a more brain-consistent tuning method yields representations that better correlate with brain data. Moreover, we identify that tasks dealing with fine-grained concept meaning yield representations that better decode brain activation patterns than other tasks, especially the syntactic chunking task. This indicates that our brain encodes more fine-grained concept information than shallow syntactic information when representing languages.

{{</citation>}}


### (146/156) Mini-BEHAVIOR: A Procedurally Generated Benchmark for Long-horizon Decision-Making in Embodied AI (Emily Jin et al., 2023)

{{<citation>}}

Emily Jin, Jiaheng Hu, Zhuoyi Huang, Ruohan Zhang, Jiajun Wu, Li Fei-Fei, Roberto Martín-Martín. (2023)  
**Mini-BEHAVIOR: A Procedurally Generated Benchmark for Long-horizon Decision-Making in Embodied AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-RO, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01824v1)  

---


**ABSTRACT**  
We present Mini-BEHAVIOR, a novel benchmark for embodied AI that challenges agents to use reasoning and decision-making skills to solve complex activities that resemble everyday human challenges. The Mini-BEHAVIOR environment is a fast, realistic Gridworld environment that offers the benefits of rapid prototyping and ease of use while preserving a symbolic level of physical realism and complexity found in complex embodied AI benchmarks. We introduce key features such as procedural generation, to enable the creation of countless task variations and support open-ended learning. Mini-BEHAVIOR provides implementations of various household tasks from the original BEHAVIOR benchmark, along with starter code for data collection and reinforcement learning agent training. In essence, Mini-BEHAVIOR offers a fast, open-ended benchmark for evaluating decision-making and planning solutions in embodied AI. It serves as a user-friendly entry point for research and facilitates the evaluation and development of solutions, simplifying their assessment and development while advancing the field of embodied AI. Code is publicly available at https://github.com/StanfordVL/mini_behavior.

{{</citation>}}


### (147/156) Discrete, compositional, and symbolic representations through attractor dynamics (Andrew Nam et al., 2023)

{{<citation>}}

Andrew Nam, Eric Elmoznino, Nikolay Malkin, Chen Sun, Yoshua Bengio, Guillaume Lajoie. (2023)  
**Discrete, compositional, and symbolic representations through attractor dynamics**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01807v1)  

---


**ABSTRACT**  
Compositionality is an important feature of discrete symbolic systems, such as language and programs, as it enables them to have infinite capacity despite a finite symbol set. It serves as a useful abstraction for reasoning in both cognitive science and in AI, yet the interface between continuous and symbolic processing is often imposed by fiat at the algorithmic level, such as by means of quantization or a softmax sampling step. In this work, we explore how discretization could be implemented in a more neurally plausible manner through the modeling of attractor dynamics that partition the continuous representation space into basins that correspond to sequences of symbols. Building on established work in attractor networks and introducing novel training methods, we show that imposing structure in the symbolic space can produce compositionality in the attractor-supported representation space of rich sensory inputs. Lastly, we argue that our model exhibits the process of an information bottleneck that is thought to play a role in conscious experience, decomposing the rich information of a sensory input into stable components encoding symbolic information.

{{</citation>}}


## cs.LO (2)



### (148/156) Reasoning about Intuitionistic Computation Tree Logic (Davide Catta et al., 2023)

{{<citation>}}

Davide Catta, Vadim Malvone, Aniello Murano. (2023)  
**Reasoning about Intuitionistic Computation Tree Logic**  

---
Primary Category: cs.LO  
Categories: cs-AI, cs-LO, cs.LO  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2310.02355v1)  

---


**ABSTRACT**  
In this paper, we define an intuitionistic version of Computation Tree Logic. After explaining the semantic features of intuitionistic logic, we examine how these characteristics can be interesting for formal verification purposes. Subsequently, we define the syntax and semantics of our intuitionistic version of CTL and study some simple properties of the so obtained logic. We conclude by demonstrating that some fixed-point axioms of CTL are not valid in the intuitionistic version of CTL we have defined.

{{</citation>}}


### (149/156) Strong Faithfulness for ELH Ontology Embeddings (Victor Lacerda et al., 2023)

{{<citation>}}

Victor Lacerda, Ana Ozaki, Ricardo Guimarães. (2023)  
**Strong Faithfulness for ELH Ontology Embeddings**  

---
Primary Category: cs.LO  
Categories: cs-LO, cs.LO  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.02198v1)  

---


**ABSTRACT**  
Ontology embedding methods are powerful approaches to represent and reason over structured knowledge in various domains. One advantage of ontology embeddings over knowledge graph embeddings is their ability to capture and impose an underlying schema to which the model must conform. Despite advances, most current approaches do not guarantee that the resulting embedding respects the axioms the ontology entails. In this work, we formally prove that normalized ${\cal ELH}$ has the strong faithfulness property on convex geometric models, which means that there is an embedding that precisely captures the original ontology. We present a region-based geometric model for embedding normalized ${\cal ELH}$ ontologies into a continuous vector space. To prove strong faithfulness, our construction takes advantage of the fact that normalized ${\cal ELH}$ has a finite canonical model. We first prove the statement assuming (possibly) non-convex regions, allowing us to keep the required dimensions low. Then, we impose convexity on the regions and show the property still holds. Finally, we consider reasoning tasks on geometric models and analyze the complexity in the class of convex geometric models used for proving strong faithfulness.

{{</citation>}}


## quant-ph (1)



### (150/156) Approximately Equivariant Quantum Neural Network for $p4m$ Group Symmetries in Images (Su Yeon Chang et al., 2023)

{{<citation>}}

Su Yeon Chang, Michele Grossi, Bertrand Le Saux, Sofia Vallecorsa. (2023)  
**Approximately Equivariant Quantum Neural Network for $p4m$ Group Symmetries in Images**  

---
Primary Category: quant-ph  
Categories: cs-AI, cs-LG, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.02323v1)  

---


**ABSTRACT**  
Quantum Neural Networks (QNNs) are suggested as one of the quantum algorithms which can be efficiently simulated with a low depth on near-term quantum hardware in the presence of noises. However, their performance highly relies on choosing the most suitable architecture of Variational Quantum Algorithms (VQAs), and the problem-agnostic models often suffer issues regarding trainability and generalization power. As a solution, the most recent works explore Geometric Quantum Machine Learning (GQML) using QNNs equivariant with respect to the underlying symmetry of the dataset. GQML adds an inductive bias to the model by incorporating the prior knowledge on the given dataset and leads to enhancing the optimization performance while constraining the search space. This work proposes equivariant Quantum Convolutional Neural Networks (EquivQCNNs) for image classification under planar $p4m$ symmetry, including reflectional and $90^\circ$ rotational symmetry. We present the results tested in different use cases, such as phase detection of the 2D Ising model and classification of the extended MNIST dataset, and compare them with those obtained with the non-equivariant model, proving that the equivariance fosters better generalization of the model.

{{</citation>}}


## q-bio.NC (1)



### (151/156) Graph Neural Network-based EEG Classification: A Survey (Dominik Klepl et al., 2023)

{{<citation>}}

Dominik Klepl, Min Wu, Fei He. (2023)  
**Graph Neural Network-based EEG Classification: A Survey**  

---
Primary Category: q-bio.NC  
Categories: cs-LG, q-bio-NC, q-bio-QM, q-bio.NC  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2310.02152v1)  

---


**ABSTRACT**  
Graph neural networks (GNN) are increasingly used to classify EEG for tasks such as emotion recognition, motor imagery and neurological diseases and disorders. A wide range of methods have been proposed to design GNN-based classifiers. Therefore, there is a need for a systematic review and categorisation of these approaches. We exhaustively search the published literature on this topic and derive several categories for comparison. These categories highlight the similarities and differences among the methods. The results suggest a prevalence of spectral graph convolutional layers over spatial. Additionally, we identify standard forms of node features, with the most popular being the raw EEG signal and differential entropy. Our results summarise the emerging trends in GNN-based approaches for EEG classification. Finally, we discuss several promising research directions, such as exploring the potential of transfer learning methods and appropriate modelling of cross-frequency interactions.

{{</citation>}}


## physics.ao-ph (1)



### (152/156) ACE: A fast, skillful learned global atmospheric model for climate prediction (Oliver Watt-Meyer et al., 2023)

{{<citation>}}

Oliver Watt-Meyer, Gideon Dresdner, Jeremy McGibbon, Spencer K. Clark, Brian Henn, James Duncan, Noah D. Brenowitz, Karthik Kashinath, Michael S. Pritchard, Boris Bonev, Matthew E. Peters, Christopher S. Bretherton. (2023)  
**ACE: A fast, skillful learned global atmospheric model for climate prediction**  

---
Primary Category: physics.ao-ph  
Categories: cs-LG, physics-ao-ph, physics.ao-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02074v1)  

---


**ABSTRACT**  
Existing ML-based atmospheric models are not suitable for climate prediction, which requires long-term stability and physical consistency. We present ACE (AI2 Climate Emulator), a 200M-parameter, autoregressive machine learning emulator of an existing comprehensive 100-km resolution global atmospheric model. The formulation of ACE allows evaluation of physical laws such as the conservation of mass and moisture. The emulator is stable for 10 years, nearly conserves column moisture without explicit constraints and faithfully reproduces the reference model's climate, outperforming a challenging baseline on over 80% of tracked variables. ACE requires nearly 100x less wall clock time and is 100x more energy efficient than the reference model using typically available resources.

{{</citation>}}


## cs.SD (2)



### (153/156) Prompting Audios Using Acoustic Properties For Emotion Representation (Hira Dhamyal et al., 2023)

{{<citation>}}

Hira Dhamyal, Benjamin Elizalde, Soham Deshmukh, Huaming Wang, Bhiksha Raj, Rita Singh. (2023)  
**Prompting Audios Using Acoustic Properties For Emotion Representation**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2310.02298v2)  

---


**ABSTRACT**  
Emotions lie on a continuum, but current models treat emotions as a finite valued discrete variable. This representation does not capture the diversity in the expression of emotion. To better represent emotions we propose the use of natural language descriptions (or prompts). In this work, we address the challenge of automatically generating these prompts and training a model to better learn emotion representations from audio and prompt pairs. We use acoustic properties that are correlated to emotion like pitch, intensity, speech rate, and articulation rate to automatically generate prompts i.e. 'acoustic prompts'. We use a contrastive learning objective to map speech to their respective acoustic prompts. We evaluate our model on Emotion Audio Retrieval and Speech Emotion Recognition. Our results show that the acoustic prompts significantly improve the model's performance in EAR, in various Precision@K metrics. In SER, we observe a 3.8% relative accuracy improvement on the Ravdess dataset.

{{</citation>}}


### (154/156) Mel-Band RoFormer for Music Source Separation (Ju-Chiang Wang et al., 2023)

{{<citation>}}

Ju-Chiang Wang, Wei-Tsung Lu, Minz Won. (2023)  
**Mel-Band RoFormer for Music Source Separation**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Embedding, Position Embedding, Transformer  
[Paper Link](http://arxiv.org/abs/2310.01809v1)  

---


**ABSTRACT**  
Recently, multi-band spectrogram-based approaches such as Band-Split RNN (BSRNN) have demonstrated promising results for music source separation. In our recent work, we introduce the BS-RoFormer model which inherits the idea of band-split scheme in BSRNN at the front-end, and then uses the hierarchical Transformer with Rotary Position Embedding (RoPE) to model the inner-band and inter-band sequences for multi-band mask estimation. This model has achieved state-of-the-art performance, but the band-split scheme is defined empirically, without analytic supports from the literature. In this paper, we propose Mel-RoFormer, which adopts the Mel-band scheme that maps the frequency bins into overlapped subbands according to the mel scale. In contract, the band-split mapping in BSRNN and BS-RoFormer is non-overlapping and designed based on heuristics. Using the MUSDB18HQ dataset for experiments, we demonstrate that Mel-RoFormer outperforms BS-RoFormer in the separation tasks of vocals, drums, and other stems.

{{</citation>}}


## cs.MM (1)



### (155/156) Online Multimedia Verification with Computational Tools and OSINT: Russia-Ukraine Conflict Case Studies (Sohail Ahmed Khan et al., 2023)

{{<citation>}}

Sohail Ahmed Khan, Jan Gunnar Furuly, Henrik Brattli Vold, Rano Tahseen, Duc-Tien Dang-Nguyen. (2023)  
**Online Multimedia Verification with Computational Tools and OSINT: Russia-Ukraine Conflict Case Studies**  

---
Primary Category: cs.MM  
Categories: cs-CY, cs-IR, cs-MM, cs.MM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01978v1)  

---


**ABSTRACT**  
This paper investigates the use of computational tools and Open-Source Intelligence (OSINT) techniques for verifying online multimedia content, with a specific focus on real-world cases from the Russia-Ukraine conflict. Over a nine-month period from April to December 2022, we examine verification workflows, tools, and case studies published by \faktiskbar. Our study showcases the effectiveness of diverse resources, including AI tools, geolocation tools, internet archives, and social media monitoring platforms, in enabling journalists and fact-checkers to efficiently process and corroborate evidence, ensuring the dissemination of accurate information. This research underscores the vital role of computational tools and OSINT techniques in promoting evidence-based reporting and combatting misinformation. We also touch on the current limitations of available tools and prospects for future developments in multimedia verification.

{{</citation>}}


## cs.IR (1)



### (156/156) Beyond-Accuracy: A Review on Diversity, Serendipity and Fairness in Recommender Systems Based on Graph Neural Networks (Tomislav Duricic et al., 2023)

{{<citation>}}

Tomislav Duricic, Dominik Kowald, Emanuel Lacic, Elisabeth Lex. (2023)  
**Beyond-Accuracy: A Review on Diversity, Serendipity and Fairness in Recommender Systems Based on Graph Neural Networks**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.02294v1)  

---


**ABSTRACT**  
By providing personalized suggestions to users, recommender systems have become essential to numerous online platforms. Collaborative filtering, particularly graph-based approaches using Graph Neural Networks (GNNs), have demonstrated great results in terms of recommendation accuracy. However, accuracy may not always be the most important criterion for evaluating recommender systems' performance, since beyond-accuracy aspects such as recommendation diversity, serendipity, and fairness can strongly influence user engagement and satisfaction. This review paper focuses on addressing these dimensions in GNN-based recommender systems, going beyond the conventional accuracy-centric perspective. We begin by reviewing recent developments in approaches that improve not only the accuracy-diversity trade-off but also promote serendipity and fairness in GNN-based recommender systems. We discuss different stages of model development including data preprocessing, graph construction, embedding initialization, propagation layers, embedding fusion, score computation, and training methodologies. Furthermore, we present a look into the practical difficulties encountered in assuring diversity, serendipity, and fairness, while retaining high accuracy. Finally, we discuss potential future research directions for developing more robust GNN-based recommender systems that go beyond the unidimensional perspective of focusing solely on accuracy. This review aims to provide researchers and practitioners with an in-depth understanding of the multifaceted issues that arise when designing GNN-based recommender systems, setting our work apart by offering a comprehensive exploration of beyond-accuracy dimensions.

{{</citation>}}
