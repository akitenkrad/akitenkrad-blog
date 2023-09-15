---
draft: false
title: "arXiv @ 2023.09.14"
date: 2023-09-14
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.14"
    identifier: arxiv_20230914
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (26)](#cslg-26)
- [cs.AI (5)](#csai-5)
- [cs.CV (29)](#cscv-29)
- [cs.RO (1)](#csro-1)
- [cs.SE (6)](#csse-6)
- [cs.CL (24)](#cscl-24)
- [cs.IR (3)](#csir-3)
- [cs.LO (1)](#cslo-1)
- [cs.HC (3)](#cshc-3)
- [cs.DB (2)](#csdb-2)
- [cs.NI (1)](#csni-1)
- [eess.IV (2)](#eessiv-2)
- [physics.flu-dyn (1)](#physicsflu-dyn-1)
- [cs.CY (1)](#cscy-1)
- [eess.AS (1)](#eessas-1)
- [cs.DL (1)](#csdl-1)
- [cs.AR (1)](#csar-1)
- [astro-ph.IM (1)](#astro-phim-1)
- [cs.CR (4)](#cscr-4)
- [cs.SI (1)](#cssi-1)
- [cs.GR (1)](#csgr-1)
- [cond-mat.mes-hall (1)](#cond-matmes-hall-1)
- [cs.GT (1)](#csgt-1)

## cs.LG (26)



### (1/117) Bregman Graph Neural Network (Jiayu Zhai et al., 2023)

{{<citation>}}

Jiayu Zhai, Lequan Lin, Dai Shi, Junbin Gao. (2023)  
**Bregman Graph Neural Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2309.06645v1)  

---


**ABSTRACT**  
Numerous recent research on graph neural networks (GNNs) has focused on formulating GNN architectures as an optimization problem with the smoothness assumption. However, in node classification tasks, the smoothing effect induced by GNNs tends to assimilate representations and over-homogenize labels of connected nodes, leading to adverse effects such as over-smoothing and misclassification. In this paper, we propose a novel bilevel optimization framework for GNNs inspired by the notion of Bregman distance. We demonstrate that the GNN layer proposed accordingly can effectively mitigate the over-smoothing issue by introducing a mechanism reminiscent of the "skip connection". We validate our theoretical results through comprehensive empirical studies in which Bregman-enhanced GNNs outperform their original counterparts in both homophilic and heterophilic graphs. Furthermore, our experiments also show that Bregman GNNs can produce more robust learning accuracy even when the number of layers is high, suggesting the effectiveness of the proposed method in alleviating the over-smoothing issue.

{{</citation>}}


### (2/117) RT-LM: Uncertainty-Aware Resource Management for Real-Time Inference of Language Models (Yufei Li et al., 2023)

{{<citation>}}

Yufei Li, Zexin Li, Wei Yang, Cong Liu. (2023)  
**RT-LM: Uncertainty-Aware Resource Management for Real-Time Inference of Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.06619v1)  

---


**ABSTRACT**  
Recent advancements in language models (LMs) have gained substantial attentions on their capability to generate human-like responses. Though exhibiting a promising future for various applications such as conversation AI, these LMs face deployment challenges on various devices due to their extreme computational cost and unpredictable inference latency. Such varied inference latency, identified as a consequence of uncertainty intrinsic to the nature of language, can lead to computational inefficiency and degrade the overall performance of LMs, especially under high-traffic workloads. Unfortunately, the bandwidth of these uncertainty sources is extensive, complicating the prediction of latency and the effects emanating from such uncertainties. To understand and mitigate the impact of uncertainty on real-time response-demanding systems, we take the first step to comprehend, quantify and optimize these uncertainty-induced latency performance variations in LMs. Specifically, we present RT-LM, an uncertainty-aware resource management ecosystem for real-time inference of LMs. RT-LM innovatively quantifies how specific input uncertainties, adversely affect latency, often leading to an increased output length. Exploiting these insights, we devise a lightweight yet effective method to dynamically correlate input text uncertainties with output length at runtime. Utilizing this quantification as a latency heuristic, we integrate the uncertainty information into a system-level scheduler which explores several uncertainty-induced optimization opportunities, including uncertainty-aware prioritization, dynamic consolidation, and strategic CPU offloading. Quantitative experiments across five state-of-the-art LMs on two hardware platforms demonstrates that RT-LM can significantly reduce the average response time and improve throughput while incurring a rather small runtime overhead.

{{</citation>}}


### (3/117) Harmonic-NAS: Hardware-Aware Multimodal Neural Architecture Search on Resource-constrained Devices (Mohamed Imed Eddine Ghebriout et al., 2023)

{{<citation>}}

Mohamed Imed Eddine Ghebriout, Halima Bouzidi, Smail Niar, Hamza Ouarnoughi. (2023)  
**Harmonic-NAS: Hardware-Aware Multimodal Neural Architecture Search on Resource-constrained Devices**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.06612v1)  

---


**ABSTRACT**  
The recent surge of interest surrounding Multimodal Neural Networks (MM-NN) is attributed to their ability to effectively process and integrate information from diverse data sources. In MM-NN, features are extracted and fused from multiple modalities using adequate unimodal backbones and specific fusion networks. Although this helps strengthen the multimodal information representation, designing such networks is labor-intensive. It requires tuning the architectural parameters of the unimodal backbones, choosing the fusing point, and selecting the operations for fusion. Furthermore, multimodality AI is emerging as a cutting-edge option in Internet of Things (IoT) systems where inference latency and energy consumption are critical metrics in addition to accuracy. In this paper, we propose Harmonic-NAS, a framework for the joint optimization of unimodal backbones and multimodal fusion networks with hardware awareness on resource-constrained devices. Harmonic-NAS involves a two-tier optimization approach for the unimodal backbone architectures and fusion strategy and operators. By incorporating the hardware dimension into the optimization, evaluation results on various devices and multimodal datasets have demonstrated the superiority of Harmonic-NAS over state-of-the-art approaches achieving up to 10.9% accuracy improvement, 1.91x latency reduction, and 2.14x energy efficiency gain.

{{</citation>}}


### (4/117) Reasoning with Latent Diffusion in Offline Reinforcement Learning (Siddarth Venkatraman et al., 2023)

{{<citation>}}

Siddarth Venkatraman, Shivesh Khaitan, Ravi Tej Akella, John Dolan, Jeff Schneider, Glen Berseth. (2023)  
**Reasoning with Latent Diffusion in Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reasoning, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.06599v1)  

---


**ABSTRACT**  
Offline reinforcement learning (RL) holds promise as a means to learn high-reward policies from a static dataset, without the need for further environment interactions. However, a key challenge in offline RL lies in effectively stitching portions of suboptimal trajectories from the static dataset while avoiding extrapolation errors arising due to a lack of support in the dataset. Existing approaches use conservative methods that are tricky to tune and struggle with multi-modal data (as we show) or rely on noisy Monte Carlo return-to-go samples for reward conditioning. In this work, we propose a novel approach that leverages the expressiveness of latent diffusion to model in-support trajectory sequences as compressed latent skills. This facilitates learning a Q-function while avoiding extrapolation error via batch-constraining. The latent space is also expressive and gracefully copes with multi-modal data. We show that the learned temporally-abstract latent space encodes richer task-specific information for offline RL tasks as compared to raw state-actions. This improves credit assignment and facilitates faster reward propagation during Q-learning. Our method demonstrates state-of-the-art performance on the D4RL benchmarks, particularly excelling in long-horizon, sparse-reward tasks.

{{</citation>}}


### (5/117) Explainable Graph Neural Network for Alzheimer's Disease And Related Dementias Risk Prediction (Xinyue Hu et al., 2023)

{{<citation>}}

Xinyue Hu, Zenan Sun, Yi Nian, Yifang Dang, Fang Li, Jingna Feng, Evan Yu, Cui Tao. (2023)  
**Explainable Graph Neural Network for Alzheimer's Disease And Related Dementias Risk Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.06584v1)  

---


**ABSTRACT**  
Alzheimer's disease and related dementias (ADRD) ranks as the sixth leading cause of death in the US, underlining the importance of accurate ADRD risk prediction. While recent advancement in ADRD risk prediction have primarily relied on imaging analysis, yet not all patients undergo medical imaging before an ADRD diagnosis. Merging machine learning with claims data can reveal additional risk factors and uncover interconnections among diverse medical codes. Our goal is to utilize Graph Neural Networks (GNNs) with claims data for ADRD risk prediction. Addressing the lack of human-interpretable reasons behind these predictions, we introduce an innovative method to evaluate relationship importance and its influence on ADRD risk prediction, ensuring comprehensive interpretation.   We employed Variationally Regularized Encoder-decoder Graph Neural Network (VGNN) for estimating ADRD likelihood. We created three scenarios to assess the model's efficiency, using Random Forest and Light Gradient Boost Machine as baselines. We further used our relation importance method to clarify the key relationships for ADRD risk prediction. VGNN surpassed other baseline models by 10% in the area under the receiver operating characteristic. The integration of the GNN model and relation importance interpretation could potentially play an essential role in providing valuable insight into factors that may contribute to or delay ADRD progression.   Employing a GNN approach with claims data enhances ADRD risk prediction and provides insights into the impact of interconnected medical code relationships. This methodology not only enables ADRD risk modeling but also shows potential for other image analysis predictions using claims data.

{{</citation>}}


### (6/117) Exploring the Benefits of Differentially Private Pre-training and Parameter-Efficient Fine-tuning for Table Transformers (Xilong Wang et al., 2023)

{{<citation>}}

Xilong Wang, Chia-Mu Yu, Pin-Yu Chen. (2023)  
**Exploring the Benefits of Differentially Private Pre-training and Parameter-Efficient Fine-tuning for Table Transformers**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.06526v1)  

---


**ABSTRACT**  
For machine learning with tabular data, Table Transformer (TabTransformer) is a state-of-the-art neural network model, while Differential Privacy (DP) is an essential component to ensure data privacy. In this paper, we explore the benefits of combining these two aspects together in the scenario of transfer learning -- differentially private pre-training and fine-tuning of TabTransformers with a variety of parameter-efficient fine-tuning (PEFT) methods, including Adapter, LoRA, and Prompt Tuning. Our extensive experiments on the ACSIncome dataset show that these PEFT methods outperform traditional approaches in terms of the accuracy of the downstream task and the number of trainable parameters, thus achieving an improved trade-off among parameter efficiency, privacy, and accuracy. Our code is available at github.com/IBM/DP-TabTransformer.

{{</citation>}}


### (7/117) A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale (Hao-Jun Michael Shi et al., 2023)

{{<citation>}}

Hao-Jun Michael Shi, Tsung-Hsien Lee, Shintaro Iwasaki, Jose Gallego-Posada, Zhijing Li, Kaushik Rangadurai, Dheevatsa Mudigere, Michael Rabbat. (2023)  
**A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs-MS, cs.LG, math-OC  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.06497v1)  

---


**ABSTRACT**  
Shampoo is an online and stochastic optimization algorithm belonging to the AdaGrad family of methods for training neural networks. It constructs a block-diagonal preconditioner where each block consists of a coarse Kronecker product approximation to full-matrix AdaGrad for each parameter of the neural network. In this work, we provide a complete description of the algorithm as well as the performance optimizations that our implementation leverages to train deep networks at-scale in PyTorch. Our implementation enables fast multi-GPU distributed data-parallel training by distributing the memory and computation associated with blocks of each parameter via PyTorch's DTensor data structure and performing an AllGather primitive on the computed search directions at each iteration. This major performance enhancement enables us to achieve at most a 10% performance reduction in per-step wall-clock time compared against standard diagonal-scaling-based adaptive gradient methods. We validate our implementation by performing an ablation study on training ImageNet ResNet50, demonstrating Shampoo's superiority over standard training recipes with minimal hyperparameter tuning.

{{</citation>}}


### (8/117) Learning Minimalistic Tsetlin Machine Clauses with Markov Boundary-Guided Pruning (Ole-Christoffer Granmo et al., 2023)

{{<citation>}}

Ole-Christoffer Granmo, Per-Arne Andersen, Lei Jiao, Xuan Zhang, Christian Blakely, Tor Tveit. (2023)  
**Learning Minimalistic Tsetlin Machine Clauses with Markov Boundary-Guided Pruning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2309.06315v1)  

---


**ABSTRACT**  
A set of variables is the Markov blanket of a random variable if it contains all the information needed for predicting the variable. If the blanket cannot be reduced without losing useful information, it is called a Markov boundary. Identifying the Markov boundary of a random variable is advantageous because all variables outside the boundary are superfluous. Hence, the Markov boundary provides an optimal feature set. However, learning the Markov boundary from data is challenging for two reasons. If one or more variables are removed from the Markov boundary, variables outside the boundary may start providing information. Conversely, variables within the boundary may stop providing information. The true role of each candidate variable is only manifesting when the Markov boundary has been identified. In this paper, we propose a new Tsetlin Machine (TM) feedback scheme that supplements Type I and Type II feedback. The scheme introduces a novel Finite State Automaton - a Context-Specific Independence Automaton. The automaton learns which features are outside the Markov boundary of the target, allowing them to be pruned from the TM during learning. We investigate the new scheme empirically, showing how it is capable of exploiting context-specific independence to find Markov boundaries. Further, we provide a theoretical analysis of convergence. Our approach thus connects the field of Bayesian networks (BN) with TMs, potentially opening up for synergies when it comes to inference and learning, including TM-produced Bayesian knowledge bases and TM-based Bayesian inference.

{{</citation>}}


### (9/117) Speciality vs Generality: An Empirical Study on Catastrophic Forgetting in Fine-tuning Foundation Models (Yong Lin et al., 2023)

{{<citation>}}

Yong Lin, Lu Tan, Hangyu Lin, Zeming Zheng, Renjie Pi, Jipeng Zhang, Shizhe Diao, Haoxiang Wang, Han Zhao, Yuan Yao, Tong Zhang. (2023)  
**Speciality vs Generality: An Empirical Study on Catastrophic Forgetting in Fine-tuning Foundation Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet, Language Model  
[Paper Link](http://arxiv.org/abs/2309.06256v1)  

---


**ABSTRACT**  
Foundation models, including Vision Language Models (VLMs) and Large Language Models (LLMs), possess the $generality$ to handle diverse distributions and tasks, which stems from their extensive pre-training datasets. The fine-tuning of foundation models is a common practice to enhance task performance or align the model's behavior with human expectations, allowing them to gain $speciality$. However, the small datasets used for fine-tuning may not adequately cover the diverse distributions and tasks encountered during pre-training. Consequently, the pursuit of speciality during fine-tuning can lead to a loss of {generality} in the model, which is related to catastrophic forgetting (CF) in deep learning. In this study, we demonstrate this phenomenon in both VLMs and LLMs. For instance, fine-tuning VLMs like CLIP on ImageNet results in a loss of generality in handling diverse distributions, and fine-tuning LLMs like Galactica in the medical domain leads to a loss in following instructions and common sense.   To address the trade-off between the speciality and generality, we investigate multiple regularization methods from continual learning, the weight averaging method (Wise-FT) from out-of-distributional (OOD) generalization, which interpolates parameters between pre-trained and fine-tuned models, and parameter-efficient fine-tuning methods like Low-Rank Adaptation (LoRA). Our findings show that both continual learning and Wise-ft methods effectively mitigate the loss of generality, with Wise-FT exhibiting the strongest performance in balancing speciality and generality.

{{</citation>}}


### (10/117) Risk-Aware Reinforcement Learning through Optimal Transport Theory (Ali Baheri, 2023)

{{<citation>}}

Ali Baheri. (2023)  
**Risk-Aware Reinforcement Learning through Optimal Transport Theory**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.06239v1)  

---


**ABSTRACT**  
In the dynamic and uncertain environments where reinforcement learning (RL) operates, risk management becomes a crucial factor in ensuring reliable decision-making. Traditional RL approaches, while effective in reward optimization, often overlook the landscape of potential risks. In response, this paper pioneers the integration of Optimal Transport (OT) theory with RL to create a risk-aware framework. Our approach modifies the objective function, ensuring that the resulting policy not only maximizes expected rewards but also respects risk constraints dictated by OT distances between state visitation distributions and the desired risk profiles. By leveraging the mathematical precision of OT, we offer a formulation that elevates risk considerations alongside conventional RL objectives. Our contributions are substantiated with a series of theorems, mapping the relationships between risk distributions, optimal value functions, and policy behaviors. Through the lens of OT, this work illuminates a promising direction for RL, ensuring a balanced fusion of reward pursuit and risk awareness.

{{</citation>}}


### (11/117) The first step is the hardest: Pitfalls of Representing and Tokenizing Temporal Data for Large Language Models (Dimitris Spathis et al., 2023)

{{<citation>}}

Dimitris Spathis, Fahim Kawsar. (2023)  
**The first step is the hardest: Pitfalls of Representing and Tokenizing Temporal Data for Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.06236v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable generalization across diverse tasks, leading individuals to increasingly use them as personal assistants and universal computing engines. Nevertheless, a notable obstacle emerges when feeding numerical/temporal data into these models, such as data sourced from wearables or electronic health records. LLMs employ tokenizers in their input that break down text into smaller units. However, tokenizers are not designed to represent numerical values and might struggle to understand repetitive patterns and context, treating consecutive values as separate tokens and disregarding their temporal relationships. Here, we discuss recent works that employ LLMs for human-centric tasks such as in mobile health sensing and present a case study showing that popular LLMs tokenize temporal data incorrectly. To address that, we highlight potential solutions such as prompt tuning with lightweight embedding layers as well as multimodal adapters, that can help bridge this "modality gap". While the capability of language models to generalize to other modalities with minimal or no finetuning is exciting, this paper underscores the fact that their outputs cannot be meaningful if they stumble over input nuances.

{{</citation>}}


### (12/117) Long-term drought prediction using deep neural networks based on geospatial weather data (Vsevolod Grabar et al., 2023)

{{<citation>}}

Vsevolod Grabar, Alexander Marusov, Alexey Zaytsev, Yury Maximov, Nazar Sotiriadi, Alexander Bulkin. (2023)  
**Long-term drought prediction using deep neural networks based on geospatial weather data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.06212v1)  

---


**ABSTRACT**  
The accurate prediction of drought probability in specific regions is crucial for informed decision-making in agricultural practices. It is important to make predictions one year in advance, particularly for long-term decisions. However, forecasting this probability presents challenges due to the complex interplay of various factors within the region of interest and neighboring areas. In this study, we propose an end-to-end solution to address this issue based on various spatiotemporal neural networks. The models considered focus on predicting the drought intensity based on the Palmer Drought Severity Index (PDSI) for subregions of interest, leveraging intrinsic factors and insights from climate models to enhance drought predictions.   Comparative evaluations demonstrate the superior accuracy of Convolutional LSTM (ConvLSTM) and transformer models compared to baseline gradient boosting and logistic regression solutions. The two former models achieved impressive ROC AUC scores from 0.90 to 0.70 for forecast horizons from one to six months, outperforming baseline models. The transformer showed superiority for shorter horizons, while ConvLSTM did so for longer horizons. Thus, we recommend selecting the models accordingly for long-term drought forecasting.   To ensure the broad applicability of the considered models, we conduct extensive validation across regions worldwide, considering different environmental conditions. We also run several ablation and sensitivity studies to challenge our findings and provide additional information on how to solve the problem.

{{</citation>}}


### (13/117) Efficient Memory Management for Large Language Model Serving with PagedAttention (Woosuk Kwon et al., 2023)

{{<citation>}}

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. (2023)  
**Efficient Memory Management for Large Language Model Serving with PagedAttention**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Attention, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2309.06180v1)  

---


**ABSTRACT**  
High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size. To address this problem, we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems. On top of it, we build vLLM, an LLM serving system that achieves (1) near-zero waste in KV cache memory and (2) flexible sharing of KV cache within and across requests to further reduce memory usage. Our evaluations show that vLLM improves the throughput of popular LLMs by 2-4$\times$ with the same level of latency compared to the state-of-the-art systems, such as FasterTransformer and Orca. The improvement is more pronounced with longer sequences, larger models, and more complex decoding algorithms. vLLM's source code is publicly available at https://github.com/vllm-project/vllm

{{</citation>}}


### (14/117) Elucidating the solution space of extended reverse-time SDE for diffusion models (Qinpeng Cui et al., 2023)

{{<citation>}}

Qinpeng Cui, Xinyi Zhang, Zongqing Lu, Qingmin Liao. (2023)  
**Elucidating the solution space of extended reverse-time SDE for diffusion models**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.06169v1)  

---


**ABSTRACT**  
Diffusion models (DMs) demonstrate potent image generation capabilities in various generative modeling tasks. Nevertheless, their primary limitation lies in slow sampling speed, requiring hundreds or thousands of sequential function evaluations through large neural networks to generate high-quality images. Sampling from DMs can be seen as solving corresponding stochastic differential equations (SDEs) or ordinary differential equations (ODEs). In this work, we formulate the sampling process as an extended reverse-time SDE (ER SDE), unifying prior explorations into ODEs and SDEs. Leveraging the semi-linear structure of ER SDE solutions, we offer exact solutions and arbitrarily high-order approximate solutions for VP SDE and VE SDE, respectively. Based on the solution space of the ER SDE, we yield mathematical insights elucidating the superior performance of ODE solvers over SDE solvers in terms of fast sampling. Additionally, we unveil that VP SDE solvers stand on par with their VE SDE counterparts. Finally, we devise fast and training-free samplers, ER-SDE Solvers, elevating the efficiency of stochastic samplers to unprecedented levels. Experimental results demonstrate achieving 3.45 FID in 20 function evaluations and 2.24 FID in 50 function evaluations on the ImageNet 64$\times$64 dataset.

{{</citation>}}


### (15/117) Certified Robust Models with Slack Control and Large Lipschitz Constants (Max Losch et al., 2023)

{{<citation>}}

Max Losch, David Stutz, Bernt Schiele, Mario Fritz. (2023)  
**Certified Robust Models with Slack Control and Large Lipschitz Constants**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG, stat-ML  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.06166v1)  

---


**ABSTRACT**  
Despite recent success, state-of-the-art learning-based models remain highly vulnerable to input changes such as adversarial examples. In order to obtain certifiable robustness against such perturbations, recent work considers Lipschitz-based regularizers or constraints while at the same time increasing prediction margin. Unfortunately, this comes at the cost of significantly decreased accuracy. In this paper, we propose a Calibrated Lipschitz-Margin Loss (CLL) that addresses this issue and improves certified robustness by tackling two problems: Firstly, commonly used margin losses do not adjust the penalties to the shrinking output distribution; caused by minimizing the Lipschitz constant $K$. Secondly, and most importantly, we observe that minimization of $K$ can lead to overly smooth decision functions. This limits the model's complexity and thus reduces accuracy. Our CLL addresses these issues by explicitly calibrating the loss w.r.t. margin and Lipschitz constant, thereby establishing full control over slack and improving robustness certificates even with larger Lipschitz constants. On CIFAR-10, CIFAR-100 and Tiny-ImageNet, our models consistently outperform losses that leave the constant unattended. On CIFAR-100 and Tiny-ImageNet, CLL improves upon state-of-the-art deterministic $L_2$ robust accuracies. In contrast to current trends, we unlock potential of much smaller models without $K=1$ constraints.

{{</citation>}}


### (16/117) Robust-MBDL: A Robust Multi-branch Deep Learning Based Model for Remaining Useful Life Prediction and Operational Condition Identification of Rotating Machines (Khoa Tran et al., 2023)

{{<citation>}}

Khoa Tran, Hai-Canh Vu, Lam Pham, Nassim Boudaoud. (2023)  
**Robust-MBDL: A Robust Multi-branch Deep Learning Based Model for Remaining Useful Life Prediction and Operational Condition Identification of Rotating Machines**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.06157v1)  

---


**ABSTRACT**  
In this paper, a Robust Multi-branch Deep learning-based system for remaining useful life (RUL) prediction and condition operations (CO) identification of rotating machines is proposed. In particular, the proposed system comprises main components: (1) an LSTM-Autoencoder to denoise the vibration data; (2) a feature extraction to generate time-domain, frequency-domain, and time-frequency based features from the denoised data; (3) a novel and robust multi-branch deep learning network architecture to exploit the multiple features. The performance of our proposed system was evaluated and compared to the state-of-the-art systems on two benchmark datasets of XJTU-SY and PRONOSTIA. The experimental results prove that our proposed system outperforms the state-of-the-art systems and presents potential for real-life applications on bearing machines.

{{</citation>}}


### (17/117) A Machine Learning Framework to Deconstruct the Primary Drivers for Electricity Market Price Events (Milan Jain et al., 2023)

{{<citation>}}

Milan Jain, Xueqing Sun, Sohom Datta, Abhishek Somani. (2023)  
**A Machine Learning Framework to Deconstruct the Primary Drivers for Electricity Market Price Events**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.06082v1)  

---


**ABSTRACT**  
Power grids are moving towards 100% renewable energy source bulk power grids, and the overall dynamics of power system operations and electricity markets are changing. The electricity markets are not only dispatching resources economically but also taking into account various controllable actions like renewable curtailment, transmission congestion mitigation, and energy storage optimization to ensure grid reliability. As a result, price formations in electricity markets have become quite complex. Traditional root cause analysis and statistical approaches are rendered inapplicable to analyze and infer the main drivers behind price formation in the modern grid and markets with variable renewable energy (VRE). In this paper, we propose a machine learning-based analysis framework to deconstruct the primary drivers for price spike events in modern electricity markets with high renewable energy. The outcomes can be utilized for various critical aspects of market design, renewable dispatch and curtailment, operations, and cyber-security applications. The framework can be applied to any ISO or market data; however, in this paper, it is applied to open-source publicly available datasets from California Independent System Operator (CAISO) and ISO New England (ISO-NE).

{{</citation>}}


### (18/117) Information Flow in Graph Neural Networks: A Clinical Triage Use Case (Víctor Valls et al., 2023)

{{<citation>}}

Víctor Valls, Mykhaylo Zayats, Alessandra Pascale. (2023)  
**Information Flow in Graph Neural Networks: A Clinical Triage Use Case**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Clinical, GNN, Graph Neural Network, Graph Neural Networks, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.06081v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have gained popularity in healthcare and other domains due to their ability to process multi-modal and multi-relational graphs. However, efficient training of GNNs remains challenging, with several open research questions. In this paper, we investigate how the flow of embedding information within GNNs affects the prediction of links in Knowledge Graphs (KGs). Specifically, we propose a mathematical model that decouples the GNN connectivity from the connectivity of the graph data and evaluate the performance of GNNs in a clinical triage use case. Our results demonstrate that incorporating domain knowledge into the GNN connectivity leads to better performance than using the same connectivity as the KG or allowing unconstrained embedding propagation. Moreover, we show that negative edges play a crucial role in achieving good predictions, and that using too many GNN layers can degrade performance.

{{</citation>}}


### (19/117) Selection of contributing factors for predicting landslide susceptibility using machine learning and deep learning models (Cheng Chen et al., 2023)

{{<citation>}}

Cheng Chen, Lei Fan. (2023)  
**Selection of contributing factors for predicting landslide susceptibility using machine learning and deep learning models**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG, physics-geo-ph  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.06062v2)  

---


**ABSTRACT**  
Landslides are a common natural disaster that can cause casualties, property safety threats and economic losses. Therefore, it is important to understand or predict the probability of landslide occurrence at potentially risky sites. A commonly used means is to carry out a landslide susceptibility assessment based on a landslide inventory and a set of landslide contributing factors. This can be readily achieved using machine learning (ML) models such as logistic regression (LR), support vector machine (SVM), random forest (RF), extreme gradient boosting (Xgboost), or deep learning (DL) models such as convolutional neural network (CNN) and long short time memory (LSTM). As the input data for these models, landslide contributing factors have varying influences on landslide occurrence. Therefore, it is logically feasible to select more important contributing factors and eliminate less relevant ones, with the aim of increasing the prediction accuracy of these models. However, selecting more important factors is still a challenging task and there is no generally accepted method. Furthermore, the effects of factor selection using various methods on the prediction accuracy of ML and DL models are unclear. In this study, the impact of the selection of contributing factors on the accuracy of landslide susceptibility predictions using ML and DL models was investigated. Four methods for selecting contributing factors were considered for all the aforementioned ML and DL models, which included Information Gain Ratio (IGR), Recursive Feature Elimination (RFE), Particle Swarm Optimization (PSO), Least Absolute Shrinkage and Selection Operators (LASSO) and Harris Hawk Optimization (HHO). In addition, autoencoder-based factor selection methods for DL models were also investigated. To assess their performances, an exhaustive approach was adopted,...

{{</citation>}}


### (20/117) How does representation impact in-context learning: A exploration on a synthetic task (Jingwen Fu et al., 2023)

{{<citation>}}

Jingwen Fu, Tao Yang, Yuwang Wang, Yan Lu, Nanning Zheng. (2023)  
**How does representation impact in-context learning: A exploration on a synthetic task**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.06054v1)  

---


**ABSTRACT**  
In-context learning, i.e., learning from in-context samples, is an impressive ability of Transformer. However, the mechanism driving the in-context learning is not yet fully understood. In this study, we aim to investigate from an underexplored perspective of representation learning. The representation is more complex for in-context learning senario, where the representation can be impacted by both model weights and in-context samples. We refer the above two conceptually aspects of representation as in-weight component and in-context component, respectively. To study how the two components affect in-context learning capabilities, we construct a novel synthetic task, making it possible to device two probes, in-weights probe and in-context probe, to evaluate the two components, respectively. We demonstrate that the goodness of in-context component is highly related to the in-context learning performance, which indicates the entanglement between in-context learning and representation learning. Furthermore, we find that a good in-weights component can actually benefit the learning of the in-context component, indicating that in-weights learning should be the foundation of in-context learning. To further understand the the in-context learning mechanism and importance of the in-weights component, we proof by construction that a simple Transformer, which uses pattern matching and copy-past mechanism to perform in-context learning, can match the in-context learning performance with more complex, best tuned Transformer under the perfect in-weights component assumption. In short, those discoveries from representation learning perspective shed light on new approaches to improve the in-context capacity.

{{</citation>}}


### (21/117) Normality Learning-based Graph Anomaly Detection via Multi-Scale Contrastive Learning (Jingcan Duan et al., 2023)

{{<citation>}}

Jingcan Duan, Pei Zhang, Siwei Wang, Jingtao Hu, Hu Jin, Jiaxin Zhang, Haifang Zhou, Haifang Zhou. (2023)  
**Normality Learning-based Graph Anomaly Detection via Multi-Scale Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.06034v1)  

---


**ABSTRACT**  
Graph anomaly detection (GAD) has attracted increasing attention in machine learning and data mining. Recent works have mainly focused on how to capture richer information to improve the quality of node embeddings for GAD. Despite their significant advances in detection performance, there is still a relative dearth of research on the properties of the task. GAD aims to discern the anomalies that deviate from most nodes. However, the model is prone to learn the pattern of normal samples which make up the majority of samples. Meanwhile, anomalies can be easily detected when their behaviors differ from normality. Therefore, the performance can be further improved by enhancing the ability to learn the normal pattern. To this end, we propose a normality learning-based GAD framework via multi-scale contrastive learning networks (NLGAD for abbreviation). Specifically, we first initialize the model with the contrastive networks on different scales. To provide sufficient and reliable normal nodes for normality learning, we design an effective hybrid strategy for normality selection. Finally, the model is refined with the only input of reliable normal nodes and learns a more accurate estimate of normality so that anomalous nodes can be more easily distinguished. Eventually, extensive experiments on six benchmark graph datasets demonstrate the effectiveness of our normality learning-based scheme on GAD. Notably, the proposed algorithm improves the detection performance (up to 5.89% AUC gain) compared with the state-of-the-art methods. The source code is released at https://github.com/FelixDJC/NLGAD.

{{</citation>}}


### (22/117) Emergent Communication in Multi-Agent Reinforcement Learning for Future Wireless Networks (Marwa Chafii et al., 2023)

{{<citation>}}

Marwa Chafii, Salmane Naoumi, Reda Alami, Ebtesam Almazrouei, Mehdi Bennis, Merouane Debbah. (2023)  
**Emergent Communication in Multi-Agent Reinforcement Learning for Future Wireless Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MA, cs.LG, eess-SP  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.06021v1)  

---


**ABSTRACT**  
In different wireless network scenarios, multiple network entities need to cooperate in order to achieve a common task with minimum delay and energy consumption. Future wireless networks mandate exchanging high dimensional data in dynamic and uncertain environments, therefore implementing communication control tasks becomes challenging and highly complex. Multi-agent reinforcement learning with emergent communication (EC-MARL) is a promising solution to address high dimensional continuous control problems with partially observable states in a cooperative fashion where agents build an emergent communication protocol to solve complex tasks. This paper articulates the importance of EC-MARL within the context of future 6G wireless networks, which imbues autonomous decision-making capabilities into network entities to solve complex tasks such as autonomous driving, robot navigation, flying base stations network planning, and smart city applications. An overview of EC-MARL algorithms and their design criteria are provided while presenting use cases and research opportunities on this emerging topic.

{{</citation>}}


### (23/117) Goal Space Abstraction in Hierarchical Reinforcement Learning via Reachability Analysis (Mehdi Zadem et al., 2023)

{{<citation>}}

Mehdi Zadem, Sergio Mover, Sao Mai Nguyen. (2023)  
**Goal Space Abstraction in Hierarchical Reinforcement Learning via Reachability Analysis**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-FL, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.07168v1)  

---


**ABSTRACT**  
Open-ended learning benefits immensely from the use of symbolic methods for goal representation as they offer ways to structure knowledge for efficient and transferable learning. However, the existing Hierarchical Reinforcement Learning (HRL) approaches relying on symbolic reasoning are often limited as they require a manual goal representation. The challenge in autonomously discovering a symbolic goal representation is that it must preserve critical information, such as the environment dynamics. In this work, we propose a developmental mechanism for subgoal discovery via an emergent representation that abstracts (i.e., groups together) sets of environment states that have similar roles in the task. We create a HRL algorithm that gradually learns this representation along with the policies and evaluate it on navigation tasks to show the learned representation is interpretable and results in data efficiency.

{{</citation>}}


### (24/117) Neural Network Layer Matrix Decomposition reveals Latent Manifold Encoding and Memory Capacity (Ng Shyh-Chang et al., 2023)

{{<citation>}}

Ng Shyh-Chang, A-Li Luo, Bo Qiu. (2023)  
**Neural Network Layer Matrix Decomposition reveals Latent Manifold Encoding and Memory Capacity**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG, physics-bio-ph  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.05968v1)  

---


**ABSTRACT**  
We prove the converse of the universal approximation theorem, i.e. a neural network (NN) encoding theorem which shows that for every stably converged NN of continuous activation functions, its weight matrix actually encodes a continuous function that approximates its training dataset to within a finite margin of error over a bounded domain. We further show that using the Eckart-Young theorem for truncated singular value decomposition of the weight matrix for every NN layer, we can illuminate the nature of the latent space manifold of the training dataset encoded and represented by every NN layer, and the geometric nature of the mathematical operations performed by each NN layer. Our results have implications for understanding how NNs break the curse of dimensionality by harnessing memory capacity for expressivity, and that the two are complementary. This Layer Matrix Decomposition (LMD) further suggests a close relationship between eigen-decomposition of NN layers and the latest advances in conceptualizations of Hopfield networks and Transformer NN models.

{{</citation>}}


### (25/117) GLAD: Content-aware Dynamic Graphs For Log Anomaly Detection (Yufei Li et al., 2023)

{{<citation>}}

Yufei Li, Yanchi Liu, Haoyu Wang, Zhengzhang Chen, Wei Cheng, Yuncong Chen, Wenchao Yu, Haifeng Chen, Cong Liu. (2023)  
**GLAD: Content-aware Dynamic Graphs For Log Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: Anomaly Detection, GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2309.05953v1)  

---


**ABSTRACT**  
Logs play a crucial role in system monitoring and debugging by recording valuable system information, including events and states. Although various methods have been proposed to detect anomalies in log sequences, they often overlook the significance of considering relations among system components, such as services and users, which can be identified from log contents. Understanding these relations is vital for detecting anomalies and their underlying causes. To address this issue, we introduce GLAD, a Graph-based Log Anomaly Detection framework designed to detect relational anomalies in system logs. GLAD incorporates log semantics, relational patterns, and sequential patterns into a unified framework for anomaly detection. Specifically, GLAD first introduces a field extraction module that utilizes prompt-based few-shot learning to identify essential fields from log contents. Then GLAD constructs dynamic log graphs for sliding windows by interconnecting extracted fields and log events parsed from the log parser. These graphs represent events and fields as nodes and their relations as edges. Subsequently, GLAD utilizes a temporal-attentive graph edge anomaly detection model for identifying anomalous relations in these dynamic log graphs. This model employs a Graph Neural Network (GNN)-based encoder enhanced with transformers to capture content, structural and temporal features. We evaluate our proposed method on three datasets, and the results demonstrate the effectiveness of GLAD in detecting anomalies indicated by varying relational patterns.

{{</citation>}}


### (26/117) ACT: Empowering Decision Transformer with Dynamic Programming via Advantage Conditioning (Chenxiao Gao et al., 2023)

{{<citation>}}

Chenxiao Gao, Chenyang Wu, Mingjun Cao, Rui Kong, Zongzhang Zhang, Yang Yu. (2023)  
**ACT: Empowering Decision Transformer with Dynamic Programming via Advantage Conditioning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.05915v1)  

---


**ABSTRACT**  
Decision Transformer (DT), which employs expressive sequence modeling techniques to perform action generation, has emerged as a promising approach to offline policy optimization. However, DT generates actions conditioned on a desired future return, which is known to bear some weaknesses such as the susceptibility to environmental stochasticity. To overcome DT's weaknesses, we propose to empower DT with dynamic programming. Our method comprises three steps. First, we employ in-sample value iteration to obtain approximated value functions, which involves dynamic programming over the MDP structure. Second, we evaluate action quality in context with estimated advantages. We introduce two types of advantage estimators, IAE and GAE, which are suitable for different tasks. Third, we train an Advantage-Conditioned Transformer (ACT) to generate actions conditioned on the estimated advantages. Finally, during testing, ACT generates actions conditioned on a desired advantage. Our evaluation results validate that, by leveraging the power of dynamic programming, ACT demonstrates effective trajectory stitching and robust action generation in spite of the environmental stochasticity, outperforming baseline methods across various benchmarks. Additionally, we conduct an in-depth analysis of ACT's various design choices through ablation studies.

{{</citation>}}


## cs.AI (5)



### (27/117) The Relational Bottleneck as an Inductive Bias for Efficient Abstraction (Taylor W. Webb et al., 2023)

{{<citation>}}

Taylor W. Webb, Steven M. Frankland, Awni Altabaa, Kamesh Krishnamurthy, Declan Campbell, Jacob Russin, Randall O'Reilly, John Lafferty, Jonathan D. Cohen. (2023)  
**The Relational Bottleneck as an Inductive Bias for Efficient Abstraction**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-NE, cs.AI  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.06629v1)  

---


**ABSTRACT**  
A central challenge for cognitive science is to explain how abstract concepts are acquired from limited experience. This effort has often been framed in terms of a dichotomy between empiricist and nativist approaches, most recently embodied by debates concerning deep neural networks and symbolic cognitive models. Here, we highlight a recently emerging line of work that suggests a novel reconciliation of these approaches, by exploiting an inductive bias that we term the relational bottleneck. We review a family of models that employ this approach to induce abstractions in a data-efficient manner, emphasizing their potential as candidate models for the acquisition of abstract concepts in the human mind and brain.

{{</citation>}}


### (28/117) Exploring Large Language Models for Ontology Alignment (Yuan He et al., 2023)

{{<citation>}}

Yuan He, Jiaoyan Chen, Hang Dong, Ian Horrocks. (2023)  
**Exploring Large Language Models for Ontology Alignment**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: BERT, GPT, GPT-3.5, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2309.07172v1)  

---


**ABSTRACT**  
This work investigates the applicability of recent generative Large Language Models (LLMs), such as the GPT series and Flan-T5, to ontology alignment for identifying concept equivalence mappings across ontologies. To test the zero-shot performance of Flan-T5-XXL and GPT-3.5-turbo, we leverage challenging subsets from two equivalence matching datasets of the OAEI Bio-ML track, taking into account concept labels and structural contexts. Preliminary findings suggest that LLMs have the potential to outperform existing ontology alignment systems like BERTMap, given careful framework and prompt design.

{{</citation>}}


### (29/117) Transferability analysis of data-driven additive manufacturing knowledge: a case study between powder bed fusion and directed energy deposition (Mutahar Safdar et al., 2023)

{{<citation>}}

Mutahar Safdar, Jiarui Xie, Hyunwoong Ko, Yan Lu, Guy Lamouche, Yaoyao Fiona Zhao. (2023)  
**Transferability analysis of data-driven additive manufacturing knowledge: a case study between powder bed fusion and directed energy deposition**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.06286v1)  

---


**ABSTRACT**  
Data-driven research in Additive Manufacturing (AM) has gained significant success in recent years. This has led to a plethora of scientific literature to emerge. The knowledge in these works consists of AM and Artificial Intelligence (AI) contexts that have not been mined and formalized in an integrated way. Moreover, no tools or guidelines exist to support data-driven knowledge transfer from one context to another. As a result, data-driven solutions using specific AI techniques are being developed and validated only for specific AM process technologies. There is a potential to exploit the inherent similarities across various AM technologies and adapt the existing solutions from one process or problem to another using AI, such as Transfer Learning. We propose a three-step knowledge transferability analysis framework in AM to support data-driven AM knowledge transfer. As a prerequisite to transferability analysis, AM knowledge is featurized into identified knowledge components. The framework consists of pre-transfer, transfer, and post-transfer steps to accomplish knowledge transfer. A case study is conducted between flagship metal AM processes. Laser Powder Bed Fusion (LPBF) is the source of knowledge motivated by its relative matureness in applying AI over Directed Energy Deposition (DED), which drives the need for knowledge transfer as the less explored target process. We show successful transfer at different levels of the data-driven solution, including data representation, model architecture, and model parameters. The pipeline of AM knowledge transfer can be automated in the future to allow efficient cross-context or cross-process knowledge exchange.

{{</citation>}}


### (30/117) Fidelity-Induced Interpretable Policy Extraction for Reinforcement Learning (Xiao Liu et al., 2023)

{{<citation>}}

Xiao Liu, Wubing Chen, Mao Tan. (2023)  
**Fidelity-Induced Interpretable Policy Extraction for Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.06097v1)  

---


**ABSTRACT**  
Deep Reinforcement Learning (DRL) has achieved remarkable success in sequential decision-making problems. However, existing DRL agents make decisions in an opaque fashion, hindering the user from establishing trust and scrutinizing weaknesses of the agents. While recent research has developed Interpretable Policy Extraction (IPE) methods for explaining how an agent takes actions, their explanations are often inconsistent with the agent's behavior and thus, frequently fail to explain. To tackle this issue, we propose a novel method, Fidelity-Induced Policy Extraction (FIPE). Specifically, we start by analyzing the optimization mechanism of existing IPE methods, elaborating on the issue of ignoring consistency while increasing cumulative rewards. We then design a fidelity-induced mechanism by integrate a fidelity measurement into the reinforcement learning feedback. We conduct experiments in the complex control environment of StarCraft II, an arena typically avoided by current IPE methods. The experiment results demonstrate that FIPE outperforms the baselines in terms of interaction performance and consistency, meanwhile easy to understand.

{{</citation>}}


### (31/117) Life-inspired Interoceptive Artificial Intelligence for Autonomous and Adaptive Agents (Sungwoo Lee et al., 2023)

{{<citation>}}

Sungwoo Lee, Younghyun Oh, Hyunhoe An, Hyebhin Yoon, Karl J. Friston, Seok Jun Hong, Choong-Wan Woo. (2023)  
**Life-inspired Interoceptive Artificial Intelligence for Autonomous and Adaptive Agents**  

---
Primary Category: cs.AI  
Categories: I-2-0, cs-AI, cs-NE, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.05999v1)  

---


**ABSTRACT**  
Building autonomous --- i.e., choosing goals based on one's needs -- and adaptive -- i.e., surviving in ever-changing environments -- agents has been a holy grail of artificial intelligence (AI). A living organism is a prime example of such an agent, offering important lessons about adaptive autonomy. Here, we focus on interoception, a process of monitoring one's internal environment to keep it within certain bounds, which underwrites the survival of an organism. To develop AI with interoception, we need to factorize the state variables representing internal environments from external environments and adopt life-inspired mathematical properties of internal environment states. This paper offers a new perspective on how interoception can help build autonomous and adaptive agents by integrating the legacy of cybernetics with recent advances in theories of life, reinforcement learning, and neuroscience.

{{</citation>}}


## cs.CV (29)



### (32/117) Accelerating Deep Neural Networks via Semi-Structured Activation Sparsity (Matteo Grimaldi et al., 2023)

{{<citation>}}

Matteo Grimaldi, Darshan C. Ganji, Ivan Lazarevich, Sudhakar Sah. (2023)  
**Accelerating Deep Neural Networks via Semi-Structured Activation Sparsity**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.06626v1)  

---


**ABSTRACT**  
The demand for efficient processing of deep neural networks (DNNs) on embedded devices is a significant challenge limiting their deployment. Exploiting sparsity in the network's feature maps is one of the ways to reduce its inference latency. It is known that unstructured sparsity results in lower accuracy degradation with respect to structured sparsity but the former needs extensive inference engine changes to get latency benefits. To tackle this challenge, we propose a solution to induce semi-structured activation sparsity exploitable through minor runtime modifications. To attain high speedup levels at inference time, we design a sparse training procedure with awareness of the final position of the activations while computing the General Matrix Multiplication (GEMM). We extensively evaluate the proposed solution across various models for image classification and object detection tasks. Remarkably, our approach yields a speed improvement of $1.25 \times$ with a minimal accuracy drop of $1.1\%$ for the ResNet18 model on the ImageNet dataset. Furthermore, when combined with a state-of-the-art structured pruning method, the resulting models provide a good latency-accuracy trade-off, outperforming models that solely employ structured pruning techniques.

{{</citation>}}


### (33/117) Rank2Tell: A Multimodal Driving Dataset for Joint Importance Ranking and Reasoning (Enna Sachdeva et al., 2023)

{{<citation>}}

Enna Sachdeva, Nakul Agarwal, Suhas Chundi, Sean Roelofs, Jiachen Li, Behzad Dariush, Chiho Choi, Mykel Kochenderfer. (2023)  
**Rank2Tell: A Multimodal Driving Dataset for Joint Importance Ranking and Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2309.06597v1)  

---


**ABSTRACT**  
The widespread adoption of commercial autonomous vehicles (AVs) and advanced driver assistance systems (ADAS) may largely depend on their acceptance by society, for which their perceived trustworthiness and interpretability to riders are crucial. In general, this task is challenging because modern autonomous systems software relies heavily on black-box artificial intelligence models. Towards this goal, this paper introduces a novel dataset, Rank2Tell, a multi-modal ego-centric dataset for Ranking the importance level and Telling the reason for the importance. Using various close and open-ended visual question answering, the dataset provides dense annotations of various semantic, spatial, temporal, and relational attributes of various important objects in complex traffic scenarios. The dense annotations and unique attributes of the dataset make it a valuable resource for researchers working on visual scene understanding and related fields. Further, we introduce a joint model for joint importance level ranking and natural language captions generation to benchmark our dataset and demonstrate performance with quantitative evaluations.

{{</citation>}}


### (34/117) Zero-Shot Visual Classification with Guided Cropping (Piyapat Saranrittichai et al., 2023)

{{<citation>}}

Piyapat Saranrittichai, Mauricio Munoz, Volker Fischer, Chaithanya Kumar Mummadi. (2023)  
**Zero-Shot Visual Classification with Guided Cropping**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.06581v1)  

---


**ABSTRACT**  
Pretrained vision-language models, such as CLIP, show promising zero-shot performance across a wide variety of datasets. For closed-set classification tasks, however, there is an inherent limitation: CLIP image encoders are typically designed to extract generic image-level features that summarize superfluous or confounding information for the target tasks. This results in degradation of classification performance, especially when objects of interest cover small areas of input images. In this work, we propose CLIP with Guided Cropping (GC-CLIP), where we use an off-the-shelf zero-shot object detection model in a preprocessing step to increase focus of zero-shot classifier to the object of interest and minimize influence of extraneous image regions. We empirically show that our approach improves zero-shot classification results across architectures and datasets, favorably for small objects.

{{</citation>}}


### (35/117) DF-TransFusion: Multimodal Deepfake Detection via Lip-Audio Cross-Attention and Facial Self-Attention (Aaditya Kharel et al., 2023)

{{<citation>}}

Aaditya Kharel, Manas Paranjape, Aniket Bera. (2023)  
**DF-TransFusion: Multimodal Deepfake Detection via Lip-Audio Cross-Attention and Facial Self-Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2309.06511v1)  

---


**ABSTRACT**  
With the rise in manipulated media, deepfake detection has become an imperative task for preserving the authenticity of digital content. In this paper, we present a novel multi-modal audio-video framework designed to concurrently process audio and video inputs for deepfake detection tasks. Our model capitalizes on lip synchronization with input audio through a cross-attention mechanism while extracting visual cues via a fine-tuned VGG-16 network. Subsequently, a transformer encoder network is employed to perform facial self-attention. We conduct multiple ablation studies highlighting different strengths of our approach. Our multi-modal methodology outperforms state-of-the-art multi-modal deepfake detection techniques in terms of F-1 and per-video AUC scores.

{{</citation>}}


### (36/117) Attention De-sparsification Matters: Inducing Diversity in Digital Pathology Representation Learning (Saarthak Kapse et al., 2023)

{{<citation>}}

Saarthak Kapse, Srijan Das, Jingwei Zhang, Rajarsi R. Gupta, Joel Saltz, Dimitris Samaras, Prateek Prasanna. (2023)  
**Attention De-sparsification Matters: Inducing Diversity in Digital Pathology Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.06439v1)  

---


**ABSTRACT**  
We propose DiRL, a Diversity-inducing Representation Learning technique for histopathology imaging. Self-supervised learning techniques, such as contrastive and non-contrastive approaches, have been shown to learn rich and effective representations of digitized tissue samples with limited pathologist supervision. Our analysis of vanilla SSL-pretrained models' attention distribution reveals an insightful observation: sparsity in attention, i.e, models tends to localize most of their attention to some prominent patterns in the image. Although attention sparsity can be beneficial in natural images due to these prominent patterns being the object of interest itself, this can be sub-optimal in digital pathology; this is because, unlike natural images, digital pathology scans are not object-centric, but rather a complex phenotype of various spatially intermixed biological components. Inadequate diversification of attention in these complex images could result in crucial information loss. To address this, we leverage cell segmentation to densely extract multiple histopathology-specific representations, and then propose a prior-guided dense pretext task for SSL, designed to match the multiple corresponding representations between the views. Through this, the model learns to attend to various components more closely and evenly, thus inducing adequate diversification in attention for capturing context rich representations. Through quantitative and qualitative analysis on multiple tasks across cancer types, we demonstrate the efficacy of our method and observe that the attention is more globally distributed.

{{</citation>}}


### (37/117) Exploring Non-additive Randomness on ViT against Query-Based Black-Box Attacks (Jindong Gu et al., 2023)

{{<citation>}}

Jindong Gu, Fangyun Wei, Philip Torr, Han Hu. (2023)  
**Exploring Non-additive Randomness on ViT against Query-Based Black-Box Attacks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.06438v1)  

---


**ABSTRACT**  
Deep Neural Networks can be easily fooled by small and imperceptible perturbations. The query-based black-box attack (QBBA) is able to create the perturbations using model output probabilities of image queries requiring no access to the underlying models. QBBA poses realistic threats to real-world applications. Recently, various types of robustness have been explored to defend against QBBA. In this work, we first taxonomize the stochastic defense strategies against QBBA. Following our taxonomy, we propose to explore non-additive randomness in models to defend against QBBA. Specifically, we focus on underexplored Vision Transformers based on their flexible architectures. Extensive experiments show that the proposed defense approach achieves effective defense, without much sacrifice in performance.

{{</citation>}}


### (38/117) Action Segmentation Using 2D Skeleton Heatmaps (Syed Waleed Hyder et al., 2023)

{{<citation>}}

Syed Waleed Hyder, Muhammad Usama, Anas Zafar, Muhammad Naufil, Andrey Konin, M. Zeeshan Zia, Quoc-Huy Tran. (2023)  
**Action Segmentation Using 2D Skeleton Heatmaps**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2309.06462v1)  

---


**ABSTRACT**  
This paper presents a 2D skeleton-based action segmentation method with applications in fine-grained human activity recognition. In contrast with state-of-the-art methods which directly take sequences of 3D skeleton coordinates as inputs and apply Graph Convolutional Networks (GCNs) for spatiotemporal feature learning, our main idea is to use sequences of 2D skeleton heatmaps as inputs and employ Temporal Convolutional Networks (TCNs) to extract spatiotemporal features. Despite lacking 3D information, our approach yields comparable/superior performances and better robustness against missing keypoints than previous methods on action segmentation datasets. Moreover, we improve the performances further by using both 2D skeleton heatmaps and RGB videos as inputs. To our best knowledge, this is the first work to utilize 2D skeleton heatmap inputs and the first work to explore 2D skeleton+RGB fusion for action segmentation.

{{</citation>}}


### (39/117) Grounded Language Acquisition From Object and Action Imagery (James Robert Kubricht et al., 2023)

{{<citation>}}

James Robert Kubricht, Zhaoyuan Yang, Jianwei Qiu, Peter Henry Tu. (2023)  
**Grounded Language Acquisition From Object and Action Imagery**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2309.06335v1)  

---


**ABSTRACT**  
Deep learning approaches to natural language processing have made great strides in recent years. While these models produce symbols that convey vast amounts of diverse knowledge, it is unclear how such symbols are grounded in data from the world. In this paper, we explore the development of a private language for visual data representation by training emergent language (EL) encoders/decoders in both i) a traditional referential game environment and ii) a contrastive learning environment utilizing a within-class matching training paradigm. An additional classification layer utilizing neural machine translation and random forest classification was used to transform symbolic representations (sequences of integer symbols) to class labels. These methods were applied in two experiments focusing on object recognition and action recognition. For object recognition, a set of sketches produced by human participants from real imagery was used (Sketchy dataset) and for action recognition, 2D trajectories were generated from 3D motion capture systems (MOVI dataset). In order to interpret the symbols produced for data in each experiment, gradient-weighted class activation mapping (Grad-CAM) methods were used to identify pixel regions indicating semantic features which contribute evidence towards symbols in learned languages. Additionally, a t-distributed stochastic neighbor embedding (t-SNE) method was used to investigate embeddings learned by CNN feature extractors.

{{</citation>}}


### (40/117) AI4Food-NutritionFW: A Novel Framework for the Automatic Synthesis and Analysis of Eating Behaviours (Sergio Romero-Tapiador et al., 2023)

{{<citation>}}

Sergio Romero-Tapiador, Ruben Tolosana, Aythami Morales, Isabel Espinosa-Salinas, Gala Freixer, Julian Fierrez, Ruben Vera-Rodriguez, Enrique Carrillo de Santa Pau, Ana Ramírez de Molina, Javier Ortega-Garcia. (2023)  
**AI4Food-NutritionFW: A Novel Framework for the Automatic Synthesis and Analysis of Eating Behaviours**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-DB, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.06308v1)  

---


**ABSTRACT**  
Nowadays millions of images are shared on social media and web platforms. In particular, many of them are food images taken from a smartphone over time, providing information related to the individual's diet. On the other hand, eating behaviours are directly related to some of the most prevalent diseases in the world. Exploiting recent advances in image processing and Artificial Intelligence (AI), this scenario represents an excellent opportunity to: i) create new methods that analyse the individuals' health from what they eat, and ii) develop personalised recommendations to improve nutrition and diet under specific circumstances (e.g., obesity or COVID). Having tunable tools for creating food image datasets that facilitate research in both lines is very much needed.   This paper proposes AI4Food-NutritionFW, a framework for the creation of food image datasets according to configurable eating behaviours. AI4Food-NutritionFW simulates a user-friendly and widespread scenario where images are taken using a smartphone. In addition to the framework, we also provide and describe a unique food image dataset that includes 4,800 different weekly eating behaviours from 15 different profiles and 1,200 subjects. Specifically, we consider profiles that comply with actual lifestyles from healthy eating behaviours (according to established knowledge), variable profiles (e.g., eating out, holidays), to unhealthy ones (e.g., excess of fast food or sweets). Finally, we automatically evaluate a healthy index of the subject's eating behaviours using multidimensional metrics based on guidelines for healthy diets proposed by international organisations, achieving promising results (99.53% and 99.60% accuracy and sensitivity, respectively). We also release to the research community a software implementation of our proposed AI4Food-NutritionFW and the mentioned food image dataset created with it.

{{</citation>}}


### (41/117) Self-Training and Multi-Task Learning for Limited Data: Evaluation Study on Object Detection (Hoàng-Ân Lê et al., 2023)

{{<citation>}}

Hoàng-Ân Lê, Minh-Tan Pham. (2023)  
**Self-Training and Multi-Task Learning for Limited Data: Evaluation Study on Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.06288v1)  

---


**ABSTRACT**  
Self-training allows a network to learn from the predictions of a more complicated model, thus often requires well-trained teacher models and mixture of teacher-student data while multi-task learning jointly optimizes different targets to learn salient interrelationship and requires multi-task annotations for each training example. These frameworks, despite being particularly data demanding have potentials for data exploitation if such assumptions can be relaxed. In this paper, we compare self-training object detection under the deficiency of teacher training data where students are trained on unseen examples by the teacher, and multi-task learning with partially annotated data, i.e. single-task annotation per training example. Both scenarios have their own limitation but potentially helpful with limited annotated data. Experimental results show the improvement of performance when using a weak teacher with unseen data for training a multi-task student. Despite the limited setup we believe the experimental results show the potential of multi-task knowledge distillation and self-training, which could be beneficial for future study. Source code is at https://lhoangan.github.io/multas.

{{</citation>}}


### (42/117) Jersey Number Recognition using Keyframe Identification from Low-Resolution Broadcast Videos (Bavesh Balaji et al., 2023)

{{<citation>}}

Bavesh Balaji, Jerrin Bright, Harish Prakash, Yuhao Chen, David A Clausi, John Zelek. (2023)  
**Jersey Number Recognition using Keyframe Identification from Low-Resolution Broadcast Videos**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.06285v1)  

---


**ABSTRACT**  
Player identification is a crucial component in vision-driven soccer analytics, enabling various downstream tasks such as player assessment, in-game analysis, and broadcast production. However, automatically detecting jersey numbers from player tracklets in videos presents challenges due to motion blur, low resolution, distortions, and occlusions. Existing methods, utilizing Spatial Transformer Networks, CNNs, and Vision Transformers, have shown success in image data but struggle with real-world video data, where jersey numbers are not visible in most of the frames. Hence, identifying frames that contain the jersey number is a key sub-problem to tackle. To address these issues, we propose a robust keyframe identification module that extracts frames containing essential high-level information about the jersey number. A spatio-temporal network is then employed to model spatial and temporal context and predict the probabilities of jersey numbers in the video. Additionally, we adopt a multi-task loss function to predict the probability distribution of each digit separately. Extensive evaluations on the SoccerNet dataset demonstrate that incorporating our proposed keyframe identification module results in a significant 37.81% and 37.70% increase in the accuracies of 2 different test sets with domain gaps. These results highlight the effectiveness and importance of our approach in tackling the challenges of automatic jersey number detection in sports videos.

{{</citation>}}


### (43/117) IBAFormer: Intra-batch Attention Transformer for Domain Generalized Semantic Segmentation (Qiyu Sun et al., 2023)

{{<citation>}}

Qiyu Sun, Huilin Chen, Meng Zheng, Ziyan Wu, Michael Felsberg, Yang Tang. (2023)  
**IBAFormer: Intra-batch Attention Transformer for Domain Generalized Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Semantic Segmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2309.06282v1)  

---


**ABSTRACT**  
Domain generalized semantic segmentation (DGSS) is a critical yet challenging task, where the model is trained only on source data without access to any target data. Despite the proposal of numerous DGSS strategies, the generalization capability remains limited in CNN architectures. Though some Transformer-based segmentation models show promising performance, they primarily focus on capturing intra-sample attentive relationships, disregarding inter-sample correlations which can potentially benefit DGSS. To this end, we enhance the attention modules in Transformer networks for improving DGSS by incorporating information from other independent samples in the same batch, enriching contextual information, and diversifying the training data for each attention block. Specifically, we propose two alternative intra-batch attention mechanisms, namely mean-based intra-batch attention (MIBA) and element-wise intra-batch attention (EIBA), to capture correlations between different samples, enhancing feature representation and generalization capabilities. Building upon intra-batch attention, we introduce IBAFormer, which integrates self-attention modules with the proposed intra-batch attention for DGSS. Extensive experiments demonstrate that IBAFormer achieves SOTA performance in DGSS, and ablation studies further confirm the effectiveness of each introduced component.

{{</citation>}}


### (44/117) Human Action Co-occurrence in Lifestyle Vlogs using Graph Link Prediction (Oana Ignat et al., 2023)

{{<citation>}}

Oana Ignat, Santiago Castro, Weiji Li, Rada Mihalcea. (2023)  
**Human Action Co-occurrence in Lifestyle Vlogs using Graph Link Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-CY, cs-IR, cs.CV  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.06219v1)  

---


**ABSTRACT**  
We introduce the task of automatic human action co-occurrence identification, i.e., determine whether two human actions can co-occur in the same interval of time. We create and make publicly available the ACE (Action Co-occurrencE) dataset, consisting of a large graph of ~12k co-occurring pairs of visual actions and their corresponding video clips. We describe graph link prediction models that leverage visual and textual information to automatically infer if two actions are co-occurring. We show that graphs are particularly well suited to capture relations between human actions, and the learned graph representations are effective for our task and capture novel and relevant information across different data domains. The ACE dataset and the code introduced in this paper are publicly available at https://github.com/MichiganNLP/vlog_action_co-occurrence.

{{</citation>}}


### (45/117) SCP: Scene Completion Pre-training for 3D Object Detection (Yiming Shan et al., 2023)

{{<citation>}}

Yiming Shan, Yan Xia, Yuhong Chen, Daniel Cremers. (2023)  
**SCP: Scene Completion Pre-training for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.06199v1)  

---


**ABSTRACT**  
3D object detection using LiDAR point clouds is a fundamental task in the fields of computer vision, robotics, and autonomous driving. However, existing 3D detectors heavily rely on annotated datasets, which are both time-consuming and prone to errors during the process of labeling 3D bounding boxes. In this paper, we propose a Scene Completion Pre-training (SCP) method to enhance the performance of 3D object detectors with less labeled data. SCP offers three key advantages: (1) Improved initialization of the point cloud model. By completing the scene point clouds, SCP effectively captures the spatial and semantic relationships among objects within urban environments. (2) Elimination of the need for additional datasets. SCP serves as a valuable auxiliary network that does not impose any additional efforts or data requirements on the 3D detectors. (3) Reduction of the amount of labeled data for detection. With the help of SCP, the existing state-of-the-art 3D detectors can achieve comparable performance while only relying on 20% labeled data.

{{</citation>}}


### (46/117) 360$^\circ$ from a Single Camera: A Few-Shot Approach for LiDAR Segmentation (Laurenz Reichardt et al., 2023)

{{<citation>}}

Laurenz Reichardt, Nikolas Ebert, Oliver Wasenmüller. (2023)  
**360$^\circ$ from a Single Camera: A Few-Shot Approach for LiDAR Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.06197v1)  

---


**ABSTRACT**  
Deep learning applications on LiDAR data suffer from a strong domain gap when applied to different sensors or tasks. In order for these methods to obtain similar accuracy on different data in comparison to values reported on public benchmarks, a large scale annotated dataset is necessary. However, in practical applications labeled data is costly and time consuming to obtain. Such factors have triggered various research in label-efficient methods, but a large gap remains to their fully-supervised counterparts. Thus, we propose ImageTo360, an effective and streamlined few-shot approach to label-efficient LiDAR segmentation. Our method utilizes an image teacher network to generate semantic predictions for LiDAR data within a single camera view. The teacher is used to pretrain the LiDAR segmentation student network, prior to optional fine-tuning on 360$^\circ$ data. Our method is implemented in a modular manner on the point level and as such is generalizable to different architectures. We improve over the current state-of-the-art results for label-efficient methods and even surpass some traditional fully-supervised segmentation networks.

{{</citation>}}


### (47/117) A 3M-Hybrid Model for the Restoration of Unique Giant Murals: A Case Study on the Murals of Yongle Palace (Jing Yang et al., 2023)

{{<citation>}}

Jing Yang, Nur Intan Raihana Ruhaiyem, Chichun Zhou. (2023)  
**A 3M-Hybrid Model for the Restoration of Unique Giant Murals: A Case Study on the Murals of Yongle Palace**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.06194v1)  

---


**ABSTRACT**  
The Yongle Palace murals, as valuable cultural heritage, have suffered varying degrees of damage, making their restoration of significant importance. However, the giant size and unique data of Yongle Palace murals present challenges for existing deep-learning based restoration methods: 1) The distinctive style introduces domain bias in traditional transfer learning-based restoration methods, while the scarcity of mural data further limits the applicability of these methods. 2) Additionally, the giant size of these murals results in a wider range of defect types and sizes, necessitating models with greater adaptability. Consequently, there is a lack of focus on deep learning-based restoration methods for the unique giant murals of Yongle Palace. Here, a 3M-Hybrid model is proposed to address these challenges. Firstly, based on the characteristic that the mural data frequency is prominent in the distribution of low and high frequency features, high and low frequency features are separately abstracted for complementary learning. Furthermore, we integrate a pre-trained Vision Transformer model (VIT) into the CNN module, allowing us to leverage the benefits of a large model while mitigating domain bias. Secondly, we mitigate seam and structural distortion issues resulting from the restoration of large defects by employing a multi-scale and multi-perspective strategy, including data segmentation and fusion. Experimental results demonstrate the efficacy of our proposed model. In regular-sized mural restoration, it improves SSIM and PSNR by 14.61% and 4.73%, respectively, compared to the best model among four representative CNN models. Additionally, it achieves favorable results in the final restoration of giant murals.

{{</citation>}}


### (48/117) Computer Vision Pipeline for Automated Antarctic Krill Analysis (Mazvydas Gudelis et al., 2023)

{{<citation>}}

Mazvydas Gudelis, Michal Mackiewicz, Julie Bremner, Sophie Fielding. (2023)  
**Computer Vision Pipeline for Automated Antarctic Krill Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2309.06188v1)  

---


**ABSTRACT**  
British Antarctic Survey (BAS) researchers launch annual expeditions to the Antarctic in order to estimate Antarctic Krill biomass and assess the change from previous years. These comparisons provide insight into the effects of the current environment on this key component of the marine food chain. In this work we have developed tools for automating the data collection and analysis process, using web-based image annotation tools and deep learning image classification and regression models. We achieve highly accurate krill instance segmentation results with an average 77.28% AP score, as well as separate maturity stage and length estimation of krill specimens with 62.99% accuracy and a 1.96 mm length error respectively.

{{</citation>}}


### (49/117) Active Label Refinement for Semantic Segmentation of Satellite Images (Tuan Pham Minh et al., 2023)

{{<citation>}}

Tuan Pham Minh, Jayan Wijesingha, Daniel Kottke, Marek Herde, Denis Huseljic, Bernhard Sick, Michael Wachendorf, Thomas Esch. (2023)  
**Active Label Refinement for Semantic Segmentation of Satellite Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.06159v1)  

---


**ABSTRACT**  
Remote sensing through semantic segmentation of satellite images contributes to the understanding and utilisation of the earth's surface. For this purpose, semantic segmentation networks are typically trained on large sets of labelled satellite images. However, obtaining expert labels for these images is costly. Therefore, we propose to rely on a low-cost approach, e.g. crowdsourcing or pretrained networks, to label the images in the first step. Since these initial labels are partially erroneous, we use active learning strategies to cost-efficiently refine the labels in the second step. We evaluate the active learning strategies using satellite images of Bengaluru in India, labelled with land cover and land use labels. Our experimental results suggest that an active label refinement to improve the semantic segmentation network's performance is beneficial.

{{</citation>}}


### (50/117) Towards Visual Taxonomy Expansion (Tinghui Zhu et al., 2023)

{{<citation>}}

Tinghui Zhu, Jingping Liu, Jiaqing Liang, Haiyun Jiang, Yanghua Xiao, Zongyu Wang, Rui Xie, Yunsen Xian. (2023)  
**Towards Visual Taxonomy Expansion**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.06105v1)  

---


**ABSTRACT**  
Taxonomy expansion task is essential in organizing the ever-increasing volume of new concepts into existing taxonomies. Most existing methods focus exclusively on using textual semantics, leading to an inability to generalize to unseen terms and the "Prototypical Hypernym Problem." In this paper, we propose Visual Taxonomy Expansion (VTE), introducing visual features into the taxonomy expansion task. We propose a textual hypernymy learning task and a visual prototype learning task to cluster textual and visual semantics. In addition to the tasks on respective modalities, we introduce a hyper-proto constraint that integrates textual and visual semantics to produce fine-grained visual semantics. Our method is evaluated on two datasets, where we obtain compelling results. Specifically, on the Chinese taxonomy dataset, our method significantly improves accuracy by 8.75 %. Additionally, our approach performs better than ChatGPT on the Chinese taxonomy dataset.

{{</citation>}}


### (51/117) Real-Time Semantic Segmentation: A Brief Survey & Comparative Study in Remote Sensing (Clifford Broni-Bediako et al., 2023)

{{<citation>}}

Clifford Broni-Bediako, Junshi Xia, Naoto Yokoya. (2023)  
**Real-Time Semantic Segmentation: A Brief Survey & Comparative Study in Remote Sensing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.06047v1)  

---


**ABSTRACT**  
Real-time semantic segmentation of remote sensing imagery is a challenging task that requires a trade-off between effectiveness and efficiency. It has many applications including tracking forest fires, detecting changes in land use and land cover, crop health monitoring, and so on. With the success of efficient deep learning methods (i.e., efficient deep neural networks) for real-time semantic segmentation in computer vision, researchers have adopted these efficient deep neural networks in remote sensing image analysis. This paper begins with a summary of the fundamental compression methods for designing efficient deep neural networks and provides a brief but comprehensive survey, outlining the recent developments in real-time semantic segmentation of remote sensing imagery. We examine several seminal efficient deep learning methods, placing them in a taxonomy based on the network architecture design approach. Furthermore, we evaluate the quality and efficiency of some existing efficient deep neural networks on a publicly available remote sensing semantic segmentation benchmark dataset, the OpenEarthMap. The experimental results of an extensive comparative study demonstrate that most of the existing efficient deep neural networks have good segmentation quality, but they suffer low inference speed (i.e., high latency rate), which may limit their capability of deployment in real-time applications of remote sensing image segmentation. We provide some insights into the current trend and future research directions for real-time semantic segmentation of remote sensing imagery.

{{</citation>}}


### (52/117) Learning from History: Task-agnostic Model Contrastive Learning for Image Restoration (Gang Wu et al., 2023)

{{<citation>}}

Gang Wu, Junjun Jiang, Kui Jiang, Xianming Liu. (2023)  
**Learning from History: Task-agnostic Model Contrastive Learning for Image Restoration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.06023v1)  

---


**ABSTRACT**  
Contrastive learning has emerged as a prevailing paradigm for high-level vision tasks, which, by introducing properly negative samples, has also been exploited for low-level vision tasks to achieve a compact optimization space to account for their ill-posed nature. However, existing methods rely on manually predefined, task-oriented negatives, which often exhibit pronounced task-specific biases. In this paper, we propose a innovative approach for the adaptive generation of negative samples directly from the target model itself, called ``learning from history``. We introduce the Self-Prior guided Negative loss for image restoration (SPNIR) to enable this approach. Our approach is task-agnostic and generic, making it compatible with any existing image restoration method or task. We demonstrate the effectiveness of our approach by retraining existing models with SPNIR. The results show significant improvements in image restoration across various tasks and architectures. For example, models retrained with SPNIR outperform the original FFANet and DehazeFormer by 3.41 dB and 0.57 dB on the RESIDE indoor dataset for image dehazing. Similarly, they achieve notable improvements of 0.47 dB on SPA-Data over IDT for image deraining and 0.12 dB on Manga109 for a 4x scale super-resolution over lightweight SwinIR, respectively. Code and retrained models are available at https://github.com/Aitical/Task-agnostic_Model_Contrastive_Learning_Image_Restoration.

{{</citation>}}


### (53/117) Feature Aggregation Network for Building Extraction from High-resolution Remote Sensing Images (Xuan Zhou et al., 2023)

{{<citation>}}

Xuan Zhou, Xuefeng Wei. (2023)  
**Feature Aggregation Network for Building Extraction from High-resolution Remote Sensing Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2309.06017v1)  

---


**ABSTRACT**  
The rapid advancement in high-resolution satellite remote sensing data acquisition, particularly those achieving submeter precision, has uncovered the potential for detailed extraction of surface architectural features. However, the diversity and complexity of surface distributions frequently lead to current methods focusing exclusively on localized information of surface features. This often results in significant intraclass variability in boundary recognition and between buildings. Therefore, the task of fine-grained extraction of surface features from high-resolution satellite imagery has emerged as a critical challenge in remote sensing image processing. In this work, we propose the Feature Aggregation Network (FANet), concentrating on extracting both global and local features, thereby enabling the refined extraction of landmark buildings from high-resolution satellite remote sensing imagery. The Pyramid Vision Transformer captures these global features, which are subsequently refined by the Feature Aggregation Module and merged into a cohesive representation by the Difference Elimination Module. In addition, to ensure a comprehensive feature map, we have incorporated the Receptive Field Block and Dual Attention Module, expanding the receptive field and intensifying attention across spatial and channel dimensions. Extensive experiments on multiple datasets have validated the outstanding capability of FANet in extracting features from high-resolution satellite images. This signifies a major breakthrough in the field of remote sensing image processing. We will release our code soon.

{{</citation>}}


### (54/117) TSSAT: Two-Stage Statistics-Aware Transformation for Artistic Style Transfer (Haibo Chen et al., 2023)

{{<citation>}}

Haibo Chen, Lei Zhao, Jun Li, Jian Yang. (2023)  
**TSSAT: Two-Stage Statistics-Aware Transformation for Artistic Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2309.06004v1)  

---


**ABSTRACT**  
Artistic style transfer aims to create new artistic images by rendering a given photograph with the target artistic style. Existing methods learn styles simply based on global statistics or local patches, lacking careful consideration of the drawing process in practice. Consequently, the stylization results either fail to capture abundant and diversified local style patterns, or contain undesired semantic information of the style image and deviate from the global style distribution. To address this issue, we imitate the drawing process of humans and propose a Two-Stage Statistics-Aware Transformation (TSSAT) module, which first builds the global style foundation by aligning the global statistics of content and style features and then further enriches local style details by swapping the local statistics (instead of local features) in a patch-wise manner, significantly improving the stylization effects. Moreover, to further enhance both content and style representations, we introduce two novel losses: an attention-based content loss and a patch-based style loss, where the former enables better content preservation by enforcing the semantic relation in the content image to be retained during stylization, and the latter focuses on increasing the local style similarity between the style and stylized images. Extensive qualitative and quantitative experiments verify the effectiveness of our method.

{{</citation>}}


### (55/117) FLDNet: A Foreground-Aware Network for Polyp Segmentation Leveraging Long-Distance Dependencies (Xuefeng Wei et al., 2023)

{{<citation>}}

Xuefeng Wei, Xuan Zhou. (2023)  
**FLDNet: A Foreground-Aware Network for Polyp Segmentation Leveraging Long-Distance Dependencies**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.05987v1)  

---


**ABSTRACT**  
Given the close association between colorectal cancer and polyps, the diagnosis and identification of colorectal polyps play a critical role in the detection and surgical intervention of colorectal cancer. In this context, the automatic detection and segmentation of polyps from various colonoscopy images has emerged as a significant problem that has attracted broad attention. Current polyp segmentation techniques face several challenges: firstly, polyps vary in size, texture, color, and pattern; secondly, the boundaries between polyps and mucosa are usually blurred, existing studies have focused on learning the local features of polyps while ignoring the long-range dependencies of the features, and also ignoring the local context and global contextual information of the combined features. To address these challenges, we propose FLDNet (Foreground-Long-Distance Network), a Transformer-based neural network that captures long-distance dependencies for accurate polyp segmentation. Specifically, the proposed model consists of three main modules: a pyramid-based Transformer encoder, a local context module, and a foreground-Aware module. Multilevel features with long-distance dependency information are first captured by the pyramid-based transformer encoder. On the high-level features, the local context module obtains the local characteristics related to the polyps by constructing different local context information. The coarse map obtained by decoding the reconstructed highest-level features guides the feature fusion process in the foreground-Aware module of the high-level features to achieve foreground enhancement of the polyps. Our proposed method, FLDNet, was evaluated using seven metrics on common datasets and demonstrated superiority over state-of-the-art methods on widely-used evaluation measures.

{{</citation>}}


### (56/117) Beyond Generation: Harnessing Text to Image Models for Object Detection and Segmentation (Yunhao Ge et al., 2023)

{{<citation>}}

Yunhao Ge, Jiashu Xu, Brian Nlong Zhao, Neel Joshi, Laurent Itti, Vibhav Vineet. (2023)  
**Beyond Generation: Harnessing Text to Image Models for Object Detection and Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.05956v1)  

---


**ABSTRACT**  
We propose a new paradigm to automatically generate training data with accurate labels at scale using the text-to-image synthesis frameworks (e.g., DALL-E, Stable Diffusion, etc.). The proposed approach1 decouples training data generation into foreground object generation, and contextually coherent background generation. To generate foreground objects, we employ a straightforward textual template, incorporating the object class name as input prompts. This is fed into a text-to-image synthesis framework, producing various foreground images set against isolated backgrounds. A foreground-background segmentation algorithm is then used to generate foreground object masks. To generate context images, we begin by creating language descriptions of the context. This is achieved by applying an image captioning method to a small set of images representing the desired context. These textual descriptions are then transformed into a diverse array of context images via a text-to-image synthesis framework. Subsequently, we composite these with the foreground object masks produced in the initial step, utilizing a cut-and-paste method, to formulate the training data. We demonstrate the advantages of our approach on five object detection and segmentation datasets, including Pascal VOC and COCO. We found that detectors trained solely on synthetic data produced by our method achieve performance comparable to those trained on real data (Fig. 1). Moreover, a combination of real and synthetic data yields even much better results. Further analysis indicates that the synthetic data distribution complements the real data distribution effectively. Additionally, we emphasize the compositional nature of our data generation approach in out-of-distribution and zero-shot data generation scenarios. We open-source our code at https://github.com/gyhandy/Text2Image-for-Detection

{{</citation>}}


### (57/117) Combining deep learning and street view imagery to map smallholder crop types (Jordi Laguarta et al., 2023)

{{<citation>}}

Jordi Laguarta, Thomas Friedel, Sherrie Wang. (2023)  
**Combining deep learning and street view imagery to map smallholder crop types**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2309.05930v1)  

---


**ABSTRACT**  
Accurate crop type maps are an essential source of information for monitoring yield progress at scale, projecting global crop production, and planning effective policies. To date, however, crop type maps remain challenging to create in low and middle-income countries due to a lack of ground truth labels for training machine learning models. Field surveys are the gold standard in terms of accuracy but require an often-prohibitively large amount of time, money, and statistical capacity. In recent years, street-level imagery, such as Google Street View, KartaView, and Mapillary, has become available around the world. Such imagery contains rich information about crop types grown at particular locations and times. In this work, we develop an automated system to generate crop type ground references using deep learning and Google Street View imagery. The method efficiently curates a set of street view images containing crop fields, trains a model to predict crop type by utilizing weakly-labelled images from disparate out-of-domain sources, and combines predicted labels with remote sensing time series to create a wall-to-wall crop type map. We show that, in Thailand, the resulting country-wide map of rice, cassava, maize, and sugarcane achieves an accuracy of 93%. As the availability of roadside imagery expands, our pipeline provides a way to map crop types at scale around the globe, especially in underserved smallholder regions.

{{</citation>}}


### (58/117) Quality-Agnostic Deepfake Detection with Intra-model Collaborative Learning (Binh M. Le et al., 2023)

{{<citation>}}

Binh M. Le, Simon S. Woo. (2023)  
**Quality-Agnostic Deepfake Detection with Intra-model Collaborative Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.05911v1)  

---


**ABSTRACT**  
Deepfake has recently raised a plethora of societal concerns over its possible security threats and dissemination of fake information. Much research on deepfake detection has been undertaken. However, detecting low quality as well as simultaneously detecting different qualities of deepfakes still remains a grave challenge. Most SOTA approaches are limited by using a single specific model for detecting certain deepfake video quality type. When constructing multiple models with prior information about video quality, this kind of strategy incurs significant computational cost, as well as model and training data overhead. Further, it cannot be scalable and practical to deploy in real-world settings. In this work, we propose a universal intra-model collaborative learning framework to enable the effective and simultaneous detection of different quality of deepfakes. That is, our approach is the quality-agnostic deepfake detection method, dubbed QAD . In particular, by observing the upper bound of general error expectation, we maximize the dependency between intermediate representations of images from different quality levels via Hilbert-Schmidt Independence Criterion. In addition, an Adversarial Weight Perturbation module is carefully devised to enable the model to be more robust against image corruption while boosting the overall model's performance. Extensive experiments over seven popular deepfake datasets demonstrate the superiority of our QAD model over prior SOTA benchmarks.

{{</citation>}}


### (59/117) Enhancing Representation in Radiography-Reports Foundation Model: A Granular Alignment Algorithm Using Masked Contrastive Learning (Weijian Huang et al., 2023)

{{<citation>}}

Weijian Huang, Hongyu Zhou, Cheng Li, Hao Yang, Jiarun Liu, Shanshan Wang. (2023)  
**Enhancing Representation in Radiography-Reports Foundation Model: A Granular Alignment Algorithm Using Masked Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.05904v1)  

---


**ABSTRACT**  
Recently, multi-modal vision-language foundation models have gained significant attention in the medical field. While these models offer great opportunities, they still face a number of challenges, such as the requirement for fine-grained knowledge understanding in computer-aided diagnosis and capability of utilizing very limited or no task-specific labeled data in real-world clinical applications. In this study, we present MaCo, a novel multi-modal medical foundation model that explores masked contrastive learning to achieve granular alignment and zero-shot learning for a variety of medical imaging tasks. MaCo incorporates a correlation weighting mechanism to adjust the correlation between masked image patches and their corresponding reports, thereby enhancing the representation learning capabilities. We evaluate MaCo on six well-known open-source X-ray datasets, and the experimental results show it outperforms seven state-of-the-art approaches for classification, segmentation, and zero-shot phase grounding, demonstrating its great potential to promote a wide range of medical image analysis tasks.

{{</citation>}}


### (60/117) Adversarial Attacks Assessment of Salient Object Detection via Symbolic Learning (Gustavo Olague et al., 2023)

{{<citation>}}

Gustavo Olague, Roberto Pineda, Gerardo Ibarra-Vazquez, Matthieu Olague, Axel Martinez, Sambit Bakshi, Jonathan Vargas, Isnardo Reducindo. (2023)  
**Adversarial Attacks Assessment of Salient Object Detection via Symbolic Learning**  

---
Primary Category: cs.CV  
Categories: 68T45, 68T05, 68T07, I-4-6; I-1-2, cs-CR, cs-CV, cs-LG, cs-NE, cs.CV  
Keywords: Adversarial Attack, Object Detection  
[Paper Link](http://arxiv.org/abs/2309.05900v1)  

---


**ABSTRACT**  
Machine learning is at the center of mainstream technology and outperforms classical approaches to handcrafted feature design. Aside from its learning process for artificial feature extraction, it has an end-to-end paradigm from input to output, reaching outstandingly accurate results. However, security concerns about its robustness to malicious and imperceptible perturbations have drawn attention since its prediction can be changed entirely. Salient object detection is a research area where deep convolutional neural networks have proven effective but whose trustworthiness represents a significant issue requiring analysis and solutions to hackers' attacks. Brain programming is a kind of symbolic learning in the vein of good old-fashioned artificial intelligence. This work provides evidence that symbolic learning robustness is crucial in designing reliable visual attention systems since it can withstand even the most intense perturbations. We test this evolutionary computation methodology against several adversarial attacks and noise perturbations using standard databases and a real-world problem of a shorebird called the Snowy Plover portraying a visual attention task. We compare our methodology with five different deep learning approaches, proving that they do not match the symbolic paradigm regarding robustness. All neural networks suffer significant performance losses, while brain programming stands its ground and remains unaffected. Also, by studying the Snowy Plover, we remark on the importance of security in surveillance activities regarding wildlife protection and conservation.

{{</citation>}}


## cs.RO (1)



### (61/117) A Reinforcement Learning Approach for Robotic Unloading from Visual Observations (Vittorio Giammarino et al., 2023)

{{<citation>}}

Vittorio Giammarino, Alberto Giammarino, Matthew Pearce. (2023)  
**A Reinforcement Learning Approach for Robotic Unloading from Visual Observations**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.06621v1)  

---


**ABSTRACT**  
In this work, we focus on a robotic unloading problem from visual observations, where robots are required to autonomously unload stacks of parcels using RGB-D images as their primary input source. While supervised and imitation learning have accomplished good results in these types of tasks, they heavily rely on labeled data, which are challenging to obtain in realistic scenarios. Our study aims to develop a sample efficient controller framework that can learn unloading tasks without the need for labeled data during the learning process. To tackle this challenge, we propose a hierarchical controller structure that combines a high-level decision-making module with classical motion control. The high-level module is trained using Deep Reinforcement Learning (DRL), wherein we incorporate a safety bias mechanism and design a reward function tailored to this task. Our experiments demonstrate that both these elements play a crucial role in achieving improved learning performance. Furthermore, to ensure reproducibility and establish a benchmark for future research, we provide free access to our code and simulation.

{{</citation>}}


## cs.SE (6)



### (62/117) The Grand Illusion: The Myth of Software Portability and Implications for ML Progress (Fraser Mince et al., 2023)

{{<citation>}}

Fraser Mince, Dzung Dinh, Jonas Kgomo, Neil Thompson, Sara Hooker. (2023)  
**The Grand Illusion: The Myth of Software Portability and Implications for ML Progress**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.07181v1)  

---


**ABSTRACT**  
Pushing the boundaries of machine learning often requires exploring different hardware and software combinations. However, the freedom to experiment across different tooling stacks can be at odds with the drive for efficiency, which has produced increasingly specialized AI hardware and incentivized consolidation around a narrow set of ML frameworks. Exploratory research can be restricted if software and hardware are co-evolving, making it even harder to stray away from mainstream ideas that work well with popular tooling stacks. While this friction increasingly impacts the rate of innovation in machine learning, to our knowledge the lack of portability in tooling has not been quantified. In this work, we ask: How portable are popular ML software frameworks? We conduct a large-scale study of the portability of mainstream ML frameworks across different hardware types. Our findings paint an uncomfortable picture -- frameworks can lose more than 40% of their key functions when ported to other hardware. Worse, even when functions are portable, the slowdown in their performance can be extreme and render performance untenable. Collectively, our results reveal how costly straying from a narrow set of hardware-software combinations can be - and suggest that specialization of hardware impedes innovation in machine learning research.

{{</citation>}}


### (63/117) Commands as AI Conversations (Diomidis Spinellis, 2023)

{{<citation>}}

Diomidis Spinellis. (2023)  
**Commands as AI Conversations**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.06551v1)  

---


**ABSTRACT**  
Developers and data scientists often struggle to write command-line inputs, even though graphical interfaces or tools like ChatGPT can assist. The solution? "ai-cli," an open-source system inspired by GitHub Copilot that converts natural language prompts into executable commands for various Linux command-line tools. By tapping into OpenAI's API, which allows interaction through JSON HTTP requests, "ai-cli" transforms user queries into actionable command-line instructions. However, integrating AI assistance across multiple command-line tools, especially in open source settings, can be complex. Historically, operating systems could mediate, but individual tool functionality and the lack of a unified approach have made centralized integration challenging. The "ai-cli" tool, by bridging this gap through dynamic loading and linking with each program's Readline library API, makes command-line interfaces smarter and more user-friendly, opening avenues for further enhancement and cross-platform applicability.

{{</citation>}}


### (64/117) Unveiling the potential of large language models in generating semantic and cross-language clones (Palash R. Roy et al., 2023)

{{<citation>}}

Palash R. Roy, Ajmain I. Alam, Farouq Al-omari, Banani Roy, Chanchal K. Roy, Kevin A. Schneider. (2023)  
**Unveiling the potential of large language models in generating semantic and cross-language clones**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-LG, cs-SE, cs.SE  
Keywords: AI, BLEU, GPT  
[Paper Link](http://arxiv.org/abs/2309.06424v1)  

---


**ABSTRACT**  
Semantic and Cross-language code clone generation may be useful for code reuse, code comprehension, refactoring and benchmarking. OpenAI's GPT model has potential in such clone generation as GPT is used for text generation. When developers copy/paste codes from Stack Overflow (SO) or within a system, there might be inconsistent changes leading to unexpected behaviours. Similarly, if someone possesses a code snippet in a particular programming language but seeks equivalent functionality in a different language, a semantic cross-language code clone generation approach could provide valuable assistance. In this study, using SemanticCloneBench as a vehicle, we evaluated how well the GPT-3 model could help generate semantic and cross-language clone variants for a given fragment.We have comprised a diverse set of code fragments and assessed GPT-3s performance in generating code variants.Through extensive experimentation and analysis, where 9 judges spent 158 hours to validate, we investigate the model's ability to produce accurate and semantically correct variants. Our findings shed light on GPT-3's strengths in code generation, offering insights into the potential applications and challenges of using advanced language models in software development. Our quantitative analysis yields compelling results. In the realm of semantic clones, GPT-3 attains an impressive accuracy of 62.14% and 0.55 BLEU score, achieved through few-shot prompt engineering. Furthermore, the model shines in transcending linguistic confines, boasting an exceptional 91.25% accuracy in generating cross-language clones

{{</citation>}}


### (65/117) RAP-Gen: Retrieval-Augmented Patch Generation with CodeT5 for Automatic Program Repair (Weishi Wang et al., 2023)

{{<citation>}}

Weishi Wang, Yue Wang, Shafiq Joty, Steven C. H. Hoi. (2023)  
**RAP-Gen: Retrieval-Augmented Patch Generation with CodeT5 for Automatic Program Repair**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: T5  
[Paper Link](http://arxiv.org/abs/2309.06057v1)  

---


**ABSTRACT**  
Automatic program repair (APR) is crucial to reduce manual debugging efforts for developers and improve software reliability. While conventional search-based techniques typically rely on heuristic rules or a redundancy assumption to mine fix patterns, recent years have witnessed the surge of deep learning (DL) based approaches to automate the program repair process in a data-driven manner. However, their performance is often limited by a fixed set of parameters to model the highly complex search space of APR. To ease such burden on the parametric models, in this work, we propose a novel Retrieval-Augmented Patch Generation framework (RAP-Gen) by explicitly leveraging relevant fix patterns retrieved from a codebase of previous bug-fix pairs. Specifically, we build a hybrid patch retriever to account for both lexical and semantic matching based on the raw source code in a language-agnostic manner, which does not rely on any code-specific features. In addition, we adapt a code-aware language model CodeT5 as our foundation model to facilitate both patch retrieval and generation tasks in a unified manner. We adopt a stage-wise approach where the patch retriever first retrieves a relevant external bug-fix pair to augment the buggy input for the CodeT5 patch generator, which synthesizes a ranked list of repair patch candidates. Notably, RAP-Gen is a generic APR framework that can flexibly integrate different patch retrievers and generators to repair various types of bugs. We thoroughly evaluate RAP-Gen on three benchmarks in two programming languages, including the TFix benchmark in JavaScript, and Code Refinement and Defects4J benchmarks in Java, where the bug localization information may or may not be provided. Experimental results show that RAP-Gen significantly outperforms previous state-of-the-art approaches on all benchmarks, e.g., repairing 15 more bugs on 818 Defects4J bugs.

{{</citation>}}


### (66/117) Automatically Estimating the Effort Required to Repay Self-Admitted Technical Debt (Yikun Li et al., 2023)

{{<citation>}}

Yikun Li, Mohamed Soliman, Paris Avgeriou. (2023)  
**Automatically Estimating the Effort Required to Repay Self-Admitted Technical Debt**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.06020v1)  

---


**ABSTRACT**  
Technical debt refers to the consequences of sub-optimal decisions made during software development that prioritize short-term benefits over long-term maintainability. Self-Admitted Technical Debt (SATD) is a specific form of technical debt, explicitly documented by developers within software artifacts such as source code comments and commit messages. As SATD can hinder software development and maintenance, it is crucial to address and prioritize it effectively. However, current methodologies lack the ability to automatically estimate the repayment effort of SATD based on its textual descriptions. To address this limitation, we propose a novel approach for automatically estimating SATD repayment effort, utilizing a comprehensive dataset comprising 341,740 SATD items from 2,568,728 commits across 1,060 Apache repositories. Our findings show that different types of SATD require varying levels of repayment effort, with code/design, requirement, and test debt demanding greater effort compared to non-SATD items, while documentation debt requires less. We introduce and evaluate machine learning methodologies, particularly BERT and TextCNN, which outperforms classic machine learning methods and the naive baseline in estimating repayment effort. Additionally, we summarize keywords associated with varying levels of repayment effort that occur during SATD repayment. Our contributions aim to enhance the prioritization of SATD repayment effort and resource allocation efficiency, ultimately benefiting software development and maintainability.

{{</citation>}}


### (67/117) Comparing Llama-2 and GPT-3 LLMs for HPC kernels generation (Pedro Valero-Lara et al., 2023)

{{<citation>}}

Pedro Valero-Lara, Alexis Huante, Mustafa Al Lail, William F. Godoy, Keita Teranishi, Prasanna Balaprakash, Jeffrey S. Vetter. (2023)  
**Comparing Llama-2 and GPT-3 LLMs for HPC kernels generation**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-DC, cs-PL, cs-SE, cs.SE  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2309.07103v1)  

---


**ABSTRACT**  
We evaluate the use of the open-source Llama-2 model for generating well-known, high-performance computing kernels (e.g., AXPY, GEMV, GEMM) on different parallel programming models and languages (e.g., C++: OpenMP, OpenMP Offload, OpenACC, CUDA, HIP; Fortran: OpenMP, OpenMP Offload, OpenACC; Python: numpy, Numba, pyCUDA, cuPy; and Julia: Threads, CUDA.jl, AMDGPU.jl). We built upon our previous work that is based on the OpenAI Codex, which is a descendant of GPT-3, to generate similar kernels with simple prompts via GitHub Copilot. Our goal is to compare the accuracy of Llama-2 and our original GPT-3 baseline by using a similar metric. Llama-2 has a simplified model that shows competitive or even superior accuracy. We also report on the differences between these foundational large language models as generative AI continues to redefine human-computer interactions. Overall, Copilot generates codes that are more reliable but less optimized, whereas codes generated by Llama-2 are less reliable but more optimized when correct.

{{</citation>}}


## cs.CL (24)



### (68/117) Do Generative Large Language Models need billions of parameters? (Sia Gholami et al., 2023)

{{<citation>}}

Sia Gholami, Marwan Omar. (2023)  
**Do Generative Large Language Models need billions of parameters?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.06589v1)  

---


**ABSTRACT**  
This paper presents novel systems and methodologies for the development of efficient large language models (LLMs). It explores the trade-offs between model size, performance, and computational resources, with the aim of maximizing the efficiency of these AI systems. The research explores novel methods that allow different parts of the model to share parameters, reducing the total number of unique parameters required. This approach ensures that the model remains compact without sacrificing its ability to learn and represent complex language structures. This study provides valuable insights and tools for creating more efficient and effective LLMs, contributing to a more sustainable and accessible future for AI language modeling.

{{</citation>}}


### (69/117) Text Encoders Lack Knowledge: Leveraging Generative LLMs for Domain-Specific Semantic Textual Similarity (Joseph Gatto et al., 2023)

{{<citation>}}

Joseph Gatto, Omar Sharif, Parker Seegmiller, Philip Bohlman, Sarah Masud Preum. (2023)  
**Text Encoders Lack Knowledge: Leveraging Generative LLMs for Domain-Specific Semantic Textual Similarity**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Textual Similarity  
[Paper Link](http://arxiv.org/abs/2309.06541v1)  

---


**ABSTRACT**  
Amidst the sharp rise in the evaluation of large language models (LLMs) on various tasks, we find that semantic textual similarity (STS) has been under-explored. In this study, we show that STS can be cast as a text generation problem while maintaining strong performance on multiple STS benchmarks. Additionally, we show generative LLMs significantly outperform existing encoder-based STS models when characterizing the semantic similarity between two texts with complex semantic relationships dependent on world knowledge. We validate this claim by evaluating both generative LLMs and existing encoder-based STS models on three newly collected STS challenge sets which require world knowledge in the domains of Health, Politics, and Sports. All newly collected data is sourced from social media content posted after May 2023 to ensure the performance of closed-source models like ChatGPT cannot be credited to memorization. Our results show that, on average, generative LLMs outperform the best encoder-only baselines by an average of 22.3% on STS tasks requiring world knowledge. Our results suggest generative language models with STS-specific prompting strategies achieve state-of-the-art performance in complex, domain-specific STS tasks.

{{</citation>}}


### (70/117) Overview of Memotion 3: Sentiment and Emotion Analysis of Codemixed Hinglish Memes (Shreyash Mishra et al., 2023)

{{<citation>}}

Shreyash Mishra, S Suryavardan, Megha Chakraborty, Parth Patwa, Anku Rani, Aman Chadha, Aishwarya Reganti, Amitava Das, Amit Sheth, Manoj Chinnakotla, Asif Ekbal, Srijan Kumar. (2023)  
**Overview of Memotion 3: Sentiment and Emotion Analysis of Codemixed Hinglish Memes**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, BERT  
[Paper Link](http://arxiv.org/abs/2309.06517v1)  

---


**ABSTRACT**  
Analyzing memes on the internet has emerged as a crucial endeavor due to the impact this multi-modal form of content wields in shaping online discourse. Memes have become a powerful tool for expressing emotions and sentiments, possibly even spreading hate and misinformation, through humor and sarcasm. In this paper, we present the overview of the Memotion 3 shared task, as part of the DeFactify 2 workshop at AAAI-23. The task released an annotated dataset of Hindi-English code-mixed memes based on their Sentiment (Task A), Emotion (Task B), and Emotion intensity (Task C). Each of these is defined as an individual task and the participants are ranked separately for each task. Over 50 teams registered for the shared task and 5 made final submissions to the test set of the Memotion 3 dataset. CLIP, BERT modifications, ViT etc. were the most popular models among the participants along with approaches such as Student-Teacher model, Fusion, and Ensembling. The best final F1 score for Task A is 34.41, Task B is 79.77 and Task C is 59.82.

{{</citation>}}


### (71/117) Leveraging Large Language Models and Weak Supervision for Social Media data annotation: an evaluation using COVID-19 self-reported vaccination tweets (Ramya Tekumalla et al., 2023)

{{<citation>}}

Ramya Tekumalla, Juan M. Banda. (2023)  
**Leveraging Large Language Models and Weak Supervision for Social Media data annotation: an evaluation using COVID-19 self-reported vaccination tweets**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SI, cs.CL  
Keywords: GPT, GPT-4, Language Model, Social Media  
[Paper Link](http://arxiv.org/abs/2309.06503v1)  

---


**ABSTRACT**  
The COVID-19 pandemic has presented significant challenges to the healthcare industry and society as a whole. With the rapid development of COVID-19 vaccines, social media platforms have become a popular medium for discussions on vaccine-related topics. Identifying vaccine-related tweets and analyzing them can provide valuable insights for public health research-ers and policymakers. However, manual annotation of a large number of tweets is time-consuming and expensive. In this study, we evaluate the usage of Large Language Models, in this case GPT-4 (March 23 version), and weak supervision, to identify COVID-19 vaccine-related tweets, with the purpose of comparing performance against human annotators. We leveraged a manu-ally curated gold-standard dataset and used GPT-4 to provide labels without any additional fine-tuning or instructing, in a single-shot mode (no additional prompting).

{{</citation>}}


### (72/117) Leveraging Large Language Models for Automated Dialogue Analysis (Sarah E. Finch et al., 2023)

{{<citation>}}

Sarah E. Finch, Ellie S. Paek, Jinho D. Choi. (2023)  
**Leveraging Large Language Models for Automated Dialogue Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Dialog, Dialogue, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2309.06490v1)  

---


**ABSTRACT**  
Developing high-performing dialogue systems benefits from the automatic identification of undesirable behaviors in system responses. However, detecting such behaviors remains challenging, as it draws on a breadth of general knowledge and understanding of conversational practices. Although recent research has focused on building specialized classifiers for detecting specific dialogue behaviors, the behavior coverage is still incomplete and there is a lack of testing on real-world human-bot interactions. This paper investigates the ability of a state-of-the-art large language model (LLM), ChatGPT-3.5, to perform dialogue behavior detection for nine categories in real human-bot dialogues. We aim to assess whether ChatGPT can match specialized models and approximate human performance, thereby reducing the cost of behavior detection tasks. Our findings reveal that neither specialized models nor ChatGPT have yet achieved satisfactory results for this task, falling short of human performance. Nevertheless, ChatGPT shows promising potential and often outperforms specialized detection models. We conclude with an in-depth examination of the prevalent shortcomings of ChatGPT, offering guidance for future research to enhance LLM capabilities.

{{</citation>}}


### (73/117) Widely Interpretable Semantic Representation: Frameless Meaning Representation for Broader Applicability (Lydia Feng et al., 2023)

{{<citation>}}

Lydia Feng, Gregor Williamson, Han He, Jinho D. Choi. (2023)  
**Widely Interpretable Semantic Representation: Frameless Meaning Representation for Broader Applicability**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Abstract Meaning Representation  
[Paper Link](http://arxiv.org/abs/2309.06460v1)  

---


**ABSTRACT**  
This paper presents a novel semantic representation, WISeR, that overcomes challenges for Abstract Meaning Representation (AMR). Despite its strengths, AMR is not easily applied to languages or domains without predefined semantic frames, and its use of numbered arguments results in semantic role labels, which are not directly interpretable and are semantically overloaded for parsers. We examine the numbered arguments of predicates in AMR and convert them to thematic roles that do not require reference to semantic frames. We create a new corpus of 1K English dialogue sentences annotated in both WISeR and AMR. WISeR shows stronger inter-annotator agreement for beginner and experienced annotators, with beginners becoming proficient in WISeR annotation more quickly. Finally, we train a state-of-the-art parser on the AMR 3.0 corpus and a WISeR corpus converted from AMR 3.0. The parser is evaluated on these corpora and our dialogue corpus. The WISeR model exhibits higher accuracy than its AMR counterpart across the board, demonstrating that WISeR is easier for parsers to learn.

{{</citation>}}


### (74/117) Cited Text Spans for Citation Text Generation (Xiangci Li et al., 2023)

{{<citation>}}

Xiangci Li, Yi-Hui Lee, Jessica Ouyang. (2023)  
**Cited Text Spans for Citation Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2309.06365v1)  

---


**ABSTRACT**  
Automatic related work generation must ground their outputs to the content of the cited papers to avoid non-factual hallucinations, but due to the length of scientific documents, existing abstractive approaches have conditioned only on the cited paper \textit{abstracts}. We demonstrate that the abstract is not always the most appropriate input for citation generation and that models trained in this way learn to hallucinate. We propose to condition instead on the \textit{cited text span} (CTS) as an alternative to the abstract. Because manual CTS annotation is extremely time- and labor-intensive, we experiment with automatic, ROUGE-based labeling of candidate CTS sentences, achieving sufficiently strong performance to substitute for expensive human annotations, and we propose a human-in-the-loop, keyword-based CTS retrieval approach that makes generating citation texts grounded in the full text of cited papers both promising and practical.

{{</citation>}}


### (75/117) Learning to Predict Concept Ordering for Common Sense Generation (Tianhui Zhang et al., 2023)

{{<citation>}}

Tianhui Zhang, Danushka Bollegala, Bei Peng. (2023)  
**Learning to Predict Concept Ordering for Common Sense Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2309.06363v1)  

---


**ABSTRACT**  
Prior work has shown that the ordering in which concepts are shown to a commonsense generator plays an important role, affecting the quality of the generated sentence. However, it remains a challenge to determine the optimal ordering of a given set of concepts such that a natural sentence covering all the concepts could be generated from a pretrained generator. To understand the relationship between the ordering of the input concepts and the quality of the generated sentences, we conduct a systematic study considering multiple language models (LMs) and concept ordering strategies. We find that BART-large model consistently outperforms all other LMs considered in this study when fine-tuned using the ordering of concepts as they appear in CommonGen training data as measured using multiple evaluation metrics. Moreover, the larger GPT3-based large language models (LLMs) variants do not necessarily outperform much smaller LMs on this task, even when fine-tuned on task-specific training data. Interestingly, human annotators significantly reorder input concept sets when manually writing sentences covering those concepts, and this ordering provides the best sentence generations independently of the LM used for the generation, outperforming a probabilistic concept ordering baseline

{{</citation>}}


### (76/117) Re-Reading Improves Reasoning in Language Models (Xiaohan Xu et al., 2023)

{{<citation>}}

Xiaohan Xu, Chongyang Tao, Tao Shen, Can Xu, Hongbo Xu, Guodong Long, Jian-guang Lou. (2023)  
**Re-Reading Improves Reasoning in Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.06275v1)  

---


**ABSTRACT**  
Reasoning presents a significant and challenging issue for Large Language Models (LLMs). The predominant focus of research has revolved around developing diverse prompting strategies to guide and structure the reasoning processes of LLMs. However, these approaches based on decoder-only causal language models often operate the input question in a single forward pass, potentially missing the rich, back-and-forth interactions inherent in human reasoning. Scant attention has been paid to a critical dimension, i.e., the input question itself embedded within the prompts. In response, we introduce a deceptively simple yet highly effective prompting strategy, termed question "re-reading". Drawing inspiration from human learning and problem-solving, re-reading entails revisiting the question information embedded within input prompts. This approach aligns seamlessly with the cognitive principle of reinforcement, enabling LLMs to extract deeper insights, identify intricate patterns, establish more nuanced connections, and ultimately enhance their reasoning capabilities across various tasks. Experiments conducted on a series of reasoning benchmarks serve to underscore the effectiveness and generality of our method. Moreover, our findings demonstrate that our approach seamlessly integrates with various language models, though-eliciting prompting methods, and ensemble techniques, further underscoring its versatility and compatibility in the realm of LLMs.

{{</citation>}}


### (77/117) Improving and Evaluating the Detection of Fragmentation in News Recommendations with the Clustering of News Story Chains (Alessandra Polimeno et al., 2023)

{{<citation>}}

Alessandra Polimeno, Myrthe Reuver, Sanne Vrijenhoek, Antske Fokkens. (2023)  
**Improving and Evaluating the Detection of Fragmentation in News Recommendations with the Clustering of News Story Chains**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-IR, cs.CL  
Keywords: BERT, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.06192v1)  

---


**ABSTRACT**  
News recommender systems play an increasingly influential role in shaping information access within democratic societies. However, tailoring recommendations to users' specific interests can result in the divergence of information streams. Fragmented access to information poses challenges to the integrity of the public sphere, thereby influencing democracy and public discourse. The Fragmentation metric quantifies the degree of fragmentation of information streams in news recommendations. Accurate measurement of this metric requires the application of Natural Language Processing (NLP) to identify distinct news events, stories, or timelines. This paper presents an extensive investigation of various approaches for quantifying Fragmentation in news recommendations. These approaches are evaluated both intrinsically, by measuring performance on news story clustering, and extrinsically, by assessing the Fragmentation scores of different simulated news recommender scenarios. Our findings demonstrate that agglomerative hierarchical clustering coupled with SentenceBERT text representation is substantially better at detecting Fragmentation than earlier implementations. Additionally, the analysis of simulated scenarios yields valuable insights and recommendations for stakeholders concerning the measurement and interpretation of Fragmentation.

{{</citation>}}


### (78/117) Glancing Future for Simultaneous Machine Translation (Shoutao Guo et al., 2023)

{{<citation>}}

Shoutao Guo, Shaolei Zhang, Yang Feng. (2023)  
**Glancing Future for Simultaneous Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2309.06179v1)  

---


**ABSTRACT**  
Simultaneous machine translation (SiMT) outputs translation while reading the source sentence. Unlike conventional sequence-to-sequence (seq2seq) training, existing SiMT methods adopt the prefix-to-prefix (prefix2prefix) training, where the model predicts target tokens based on partial source tokens. However, the prefix2prefix training diminishes the ability of the model to capture global information and introduces forced predictions due to the absence of essential source information. Consequently, it is crucial to bridge the gap between the prefix2prefix training and seq2seq training to enhance the translation capability of the SiMT model. In this paper, we propose a novel method that glances future in curriculum learning to achieve the transition from the seq2seq training to prefix2prefix training. Specifically, we gradually reduce the available source information from the whole sentence to the prefix corresponding to that latency. Our method is applicable to a wide range of SiMT methods and experiments demonstrate that our method outperforms strong baselines.

{{</citation>}}


### (79/117) AKEM: Aligning Knowledge Base to Queries with Ensemble Model for Entity Recognition and Linking (Di Lu et al., 2023)

{{<citation>}}

Di Lu, Zhongping Liang, Caixia Yuan, Xiaojie Wang. (2023)  
**AKEM: Aligning Knowledge Base to Queries with Ensemble Model for Entity Recognition and Linking**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.06175v2)  

---


**ABSTRACT**  
This paper presents a novel approach to address the Entity Recognition and Linking Challenge at NLPCC 2015. The task involves extracting named entity mentions from short search queries and linking them to entities within a reference Chinese knowledge base. To tackle this problem, we first expand the existing knowledge base and utilize external knowledge to identify candidate entities, thereby improving the recall rate. Next, we extract features from the candidate entities and utilize Support Vector Regression and Multiple Additive Regression Tree as scoring functions to filter the results. Additionally, we apply rules to further refine the results and enhance precision. Our method is computationally efficient and achieves an F1 score of 0.535.

{{</citation>}}


### (80/117) Overview of GUA-SPA at IberLEF 2023: Guarani-Spanish Code Switching Analysis (Luis Chiruzzo et al., 2023)

{{<citation>}}

Luis Chiruzzo, Marvin Agüero-Torales, Gustavo Giménez-Lugo, Aldo Alvarez, Yliana Rodríguez, Santiago Góngora, Thamar Solorio. (2023)  
**Overview of GUA-SPA at IberLEF 2023: Guarani-Spanish Code Switching Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2309.06163v1)  

---


**ABSTRACT**  
We present the first shared task for detecting and analyzing code-switching in Guarani and Spanish, GUA-SPA at IberLEF 2023. The challenge consisted of three tasks: identifying the language of a token, NER, and a novel task of classifying the way a Spanish span is used in the code-switched context. We annotated a corpus of 1500 texts extracted from news articles and tweets, around 25 thousand tokens, with the information for the tasks. Three teams took part in the evaluation phase, obtaining in general good results for Task 1, and more mixed results for Tasks 2 and 3.

{{</citation>}}


### (81/117) Prompting4Debugging: Red-Teaming Text-to-Image Diffusion Models by Finding Problematic Prompts (Zhi-Yi Chin et al., 2023)

{{<citation>}}

Zhi-Yi Chin, Chieh-Ming Jiang, Ching-Chun Huang, Pin-Yu Chen, Wei-Chen Chiu. (2023)  
**Prompting4Debugging: Red-Teaming Text-to-Image Diffusion Models by Finding Problematic Prompts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.06135v1)  

---


**ABSTRACT**  
Text-to-image diffusion models, e.g. Stable Diffusion (SD), lately have shown remarkable ability in high-quality content generation, and become one of the representatives for the recent wave of transformative AI. Nevertheless, such advance comes with an intensifying concern about the misuse of this generative technology, especially for producing copyrighted or NSFW (i.e. not safe for work) images. Although efforts have been made to filter inappropriate images/prompts or remove undesirable concepts/styles via model fine-tuning, the reliability of these safety mechanisms against diversified problematic prompts remains largely unexplored. In this work, we propose Prompting4Debugging (P4D) as a debugging and red-teaming tool that automatically finds problematic prompts for diffusion models to test the reliability of a deployed safety mechanism. We demonstrate the efficacy of our P4D tool in uncovering new vulnerabilities of SD models with safety mechanisms. Particularly, our result shows that around half of prompts in existing safe prompting benchmarks which were originally considered "safe" can actually be manipulated to bypass many deployed safety mechanisms, including concept removal, negative prompt, and safety guidance. Our findings suggest that, without comprehensive testing, the evaluations on limited safe prompting benchmarks can lead to a false sense of safety for text-to-image models.

{{</citation>}}


### (82/117) Measuring vagueness and subjectivity in texts: from symbolic to neural VAGO (Benjamin Icard et al., 2023)

{{<citation>}}

Benjamin Icard, Vincent Claveau, Ghislain Atemezing, Paul Égré. (2023)  
**Measuring vagueness and subjectivity in texts: from symbolic to neural VAGO**  

---
Primary Category: cs.CL  
Categories: 68T07, 68T50, cs-AI, cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.06132v1)  

---


**ABSTRACT**  
We present a hybrid approach to the automated measurement of vagueness and subjectivity in texts. We first introduce the expert system VAGO, we illustrate it on a small benchmark of fact vs. opinion sentences, and then test it on the larger French press corpus FreSaDa to confirm the higher prevalence of subjective markers in satirical vs. regular texts. We then build a neural clone of VAGO, based on a BERT-like architecture, trained on the symbolic VAGO scores obtained on FreSaDa. Using explainability tools (LIME), we show the interest of this neural version for the enrichment of the lexicons of the symbolic version, and for the production of versions in other languages.

{{</citation>}}


### (83/117) Characterizing Latent Perspectives of Media Houses Towards Public Figures (Sharath Srivatsa et al., 2023)

{{<citation>}}

Sharath Srivatsa, Srinath Srinivasa. (2023)  
**Characterizing Latent Perspectives of Media Houses Towards Public Figures**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2309.06112v1)  

---


**ABSTRACT**  
Media houses reporting on public figures, often come with their own biases stemming from their respective worldviews. A characterization of these underlying patterns helps us in better understanding and interpreting news stories. For this, we need diverse or subjective summarizations, which may not be amenable for classifying into predefined class labels. This work proposes a zero-shot approach for non-extractive or generative characterizations of person entities from a corpus using GPT-2. We use well-articulated articles from several well-known news media houses as a corpus to build a sound argument for this approach. First, we fine-tune a GPT-2 pre-trained language model with a corpus where specific person entities are characterized. Second, we further fine-tune this with demonstrations of person entity characterizations, created from a corpus of programmatically constructed characterizations. This twice fine-tuned model is primed with manual prompts consisting of entity names that were not previously encountered in the second fine-tuning, to generate a simple sentence about the entity. The results were encouraging, when compared against actual characterizations from the corpus.

{{</citation>}}


### (84/117) BHASA: A Holistic Southeast Asian Linguistic and Cultural Evaluation Suite for Large Language Models (Wei Qi Leong et al., 2023)

{{<citation>}}

Wei Qi Leong, Jian Gang Ngui, Yosephine Susanto, Hamsawardhini Rengarajan, Kengatharaiyer Sarveswaran, William Chandra Tjhi. (2023)  
**BHASA: A Holistic Southeast Asian Linguistic and Cultural Evaluation Suite for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, NLP, NLU, Natural Language Understanding, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.06085v1)  

---


**ABSTRACT**  
The rapid development of Large Language Models (LLMs) and the emergence of novel abilities with scale have necessitated the construction of holistic, diverse and challenging benchmarks such as HELM and BIG-bench. However, at the moment, most of these benchmarks focus only on performance in English and evaluations that include Southeast Asian (SEA) languages are few in number. We therefore propose BHASA, a holistic linguistic and cultural evaluation suite for LLMs in SEA languages. It comprises three components: (1) a NLP benchmark covering eight tasks across Natural Language Understanding (NLU), Generation (NLG) and Reasoning (NLR) tasks, (2) LINDSEA, a linguistic diagnostic toolkit that spans the gamut of linguistic phenomena including syntax, semantics and pragmatics, and (3) a cultural diagnostics dataset that probes for both cultural representation and sensitivity. For this preliminary effort, we implement the NLP benchmark only for Indonesian, Vietnamese, Thai and Tamil, and we only include Indonesian and Tamil for LINDSEA and the cultural diagnostics dataset. As GPT-4 is purportedly one of the best-performing multilingual LLMs at the moment, we use it as a yardstick to gauge the capabilities of LLMs in the context of SEA languages. Our initial experiments on GPT-4 with BHASA find it lacking in various aspects of linguistic capabilities, cultural representation and sensitivity in the targeted SEA languages. BHASA is a work in progress and will continue to be improved and expanded in the future.

{{</citation>}}


### (85/117) Narrowing the Gap between Supervised and Unsupervised Sentence Representation Learning with Large Language Model (Mingxin Li et al., 2023)

{{<citation>}}

Mingxin Li, Richong Zhang, Zhijie Nie, Yongyi Mao. (2023)  
**Narrowing the Gap between Supervised and Unsupervised Sentence Representation Learning with Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Embedding, Language Model, NLP, Natural Language Processing, Representation Learning, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2309.06453v1)  

---


**ABSTRACT**  
Sentence Representation Learning (SRL) is a fundamental task in Natural Language Processing (NLP), with Contrastive learning of Sentence Embeddings (CSE) as the mainstream technique due to its superior performance. An intriguing phenomenon in CSE is the significant performance gap between supervised and unsupervised methods, even when their sentence encoder and loss function are the same. Previous works attribute this performance gap to differences in two representation properties (alignment and uniformity). However, alignment and uniformity only measure the results, which means they cannot answer "What happens during the training process that leads to the performance gap?" and "How can the performance gap be narrowed?". In this paper, we conduct empirical experiments to answer these "What" and "How" questions. We first answer the "What" question by thoroughly comparing the behavior of supervised and unsupervised CSE during their respective training processes. From the comparison, We observe a significant difference in fitting difficulty. Thus, we introduce a metric, called Fitting Difficulty Increment (FDI), to measure the fitting difficulty gap between the evaluation dataset and the held-out training dataset, and use the metric to answer the "What" question. Then, based on the insights gained from the "What" question, we tackle the "How" question by increasing the fitting difficulty of the training dataset. We achieve this by leveraging the In-Context Learning (ICL) capability of the Large Language Model (LLM) to generate data that simulates complex patterns. By utilizing the hierarchical patterns in the LLM-generated data, we effectively narrow the gap between supervised and unsupervised CSE.

{{</citation>}}


### (86/117) Circuit Breaking: Removing Model Behaviors with Targeted Ablation (Maximilian Li et al., 2023)

{{<citation>}}

Maximilian Li, Xander Davies, Max Nadeau. (2023)  
**Circuit Breaking: Removing Model Behaviors with Targeted Ablation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2309.05973v1)  

---


**ABSTRACT**  
Language models often exhibit behaviors that improve performance on a pre-training objective but harm performance on downstream tasks. We propose a novel approach to removing undesirable behaviors by ablating a small number of causal pathways between model components, with the intention of disabling the computational circuit responsible for the bad behavior. Given a small dataset of inputs where the model behaves poorly, we learn to ablate a small number of important causal pathways. In the setting of reducing GPT-2 toxic language generation, we find ablating just 12 of the 11.6K causal edges mitigates toxic generation with minimal degradation of performance on other inputs.

{{</citation>}}


### (87/117) The Moral Machine Experiment on Large Language Models (Kazuhiro Takemoto, 2023)

{{<citation>}}

Kazuhiro Takemoto. (2023)  
**The Moral Machine Experiment on Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-HC, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2309.05958v1)  

---


**ABSTRACT**  
As large language models (LLMs) become more deeply integrated into various sectors, understanding how they make moral judgments has become crucial, particularly in the realm of autonomous driving. This study utilized the Moral Machine framework to investigate the ethical decision-making tendencies of prominent LLMs, including GPT-3.5, GPT-4, PaLM 2, and Llama 2, comparing their responses to human preferences. While LLMs' and humans' preferences such as prioritizing humans over pets and favoring saving more lives are broadly aligned, PaLM 2 and Llama 2, especially, evidence distinct deviations. Additionally, despite the qualitative similarities between the LLM and human preferences, there are significant quantitative disparities, suggesting that LLMs might lean toward more uncompromising decisions, compared to the milder inclinations of humans. These insights elucidate the ethical frameworks of LLMs and their potential implications for autonomous driving.

{{</citation>}}


### (88/117) Balanced and Explainable Social Media Analysis for Public Health with Large Language Models (Yan Jiang et al., 2023)

{{<citation>}}

Yan Jiang, Ruihong Qiu, Yi Zhang, Peng-Fei Zhang. (2023)  
**Balanced and Explainable Social Media Analysis for Public Health with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model, Social Media  
[Paper Link](http://arxiv.org/abs/2309.05951v1)  

---


**ABSTRACT**  
As social media becomes increasingly popular, more and more public health activities emerge, which is worth noting for pandemic monitoring and government decision-making. Current techniques for public health analysis involve popular models such as BERT and large language models (LLMs). Although recent progress in LLMs has shown a strong ability to comprehend knowledge by being fine-tuned on specific domain datasets, the costs of training an in-domain LLM for every specific public health task are especially expensive. Furthermore, such kinds of in-domain datasets from social media are generally highly imbalanced, which will hinder the efficiency of LLMs tuning. To tackle these challenges, the data imbalance issue can be overcome by sophisticated data augmentation methods for social media datasets. In addition, the ability of the LLMs can be effectively utilised by prompting the model properly. In light of the above discussion, in this paper, a novel ALEX framework is proposed for social media analysis on public health. Specifically, an augmentation pipeline is developed to resolve the data imbalance issue. Furthermore, an LLMs explanation mechanism is proposed by prompting an LLM with the predicted results from BERT models. Extensive experiments conducted on three tasks at the Social Media Mining for Health 2023 (SMM4H) competition with the first ranking in two tasks demonstrate the superior performance of the proposed ALEX method. Our code has been released in https://github.com/YanJiangJerry/ALEX.

{{</citation>}}


### (89/117) Language Models as Black-Box Optimizers for Vision-Language Models (Samuel Yu et al., 2023)

{{<citation>}}

Samuel Yu, Shihong Liu, Zhiqiu Lin, Deepak Pathak, Deva Ramanan. (2023)  
**Language Models as Black-Box Optimizers for Vision-Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs-LG, cs-MM, cs.CL  
Keywords: AI, ChatGPT, GPT, ImageNet, Language Model  
[Paper Link](http://arxiv.org/abs/2309.05950v1)  

---


**ABSTRACT**  
Vision-language models (VLMs) pre-trained on web-scale datasets have demonstrated remarkable capabilities across a variety of vision and multimodal tasks. Currently, fine-tuning methods for VLMs mainly operate in a white-box setting, requiring access to model parameters for backpropagation. However, many VLMs rely on proprietary data and are not open-source, which restricts the use of white-box approaches for fine-tuning. Given that popular private large language models (LLMs) like ChatGPT still offer a language-based user interface, we aim to develop a novel fine-tuning approach for VLMs through natural language prompts, thereby avoiding the need to access model parameters, feature embeddings, or output logits. In this setup, we propose employing chat-based LLMs as black-box optimizers to search for the best text prompt on the illustrative task of few-shot image classification using CLIP. Specifically, we adopt an automatic "hill-climbing" procedure that converges on an effective prompt by evaluating the accuracy of current prompts and asking LLMs to refine them based on textual feedback, all within a conversational process without human-in-the-loop. In a challenging 1-shot learning setup, our simple approach surpasses the white-box continuous prompting method CoOp by an average of 1.5% across 11 datasets including ImageNet. Our approach also outperforms OpenAI's manually crafted prompts and is more efficient than other black-box methods like iterative APE. Additionally, we highlight the advantage of conversational feedback incorporating both positive and negative prompts, suggesting that LLMs can utilize the implicit "gradient" direction in textual feedback for a more efficient search. Lastly, we find that the text prompts generated through our strategy are not only more interpretable but also transfer well across different CLIP architectures in a black-box manner.

{{</citation>}}


### (90/117) Answering Subjective Induction Questions on Products by Summarizing Multi-sources Multi-viewpoints Knowledge (Yufeng Zhang et al., 2023)

{{<citation>}}

Yufeng Zhang, Meng-xiang Wang, Jianxing Yu. (2023)  
**Answering Subjective Induction Questions on Products by Summarizing Multi-sources Multi-viewpoints Knowledge**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.05938v1)  

---


**ABSTRACT**  
This paper proposes a new task in the field of Answering Subjective Induction Question on Products (SUBJPQA). The answer to this kind of question is non-unique, but can be interpreted from many perspectives. For example, the answer to 'whether the phone is heavy' has a variety of different viewpoints. A satisfied answer should be able to summarize these subjective opinions from multiple sources and provide objective knowledge, such as the weight of a phone. That is quite different from the traditional QA task, in which the answer to a factoid question is unique and can be found from a single data source. To address this new task, we propose a three-steps method. We first retrieve all answer-related clues from multiple knowledge sources on facts and opinions. The implicit commonsense facts are also collected to supplement the necessary but missing contexts. We then capture their relevance with the questions by interactive attention. Next, we design a reinforcement-based summarizer to aggregate all these knowledgeable clues. Based on a template-controlled decoder, we can output a comprehensive and multi-perspective answer. Due to the lack of a relevant evaluated benchmark set for the new task, we construct a large-scale dataset, named SupQA, consisting of 48,352 samples across 15 product domains. Evaluation results show the effectiveness of our approach.

{{</citation>}}


### (91/117) Do PLMs Know and Understand Ontological Knowledge? (Weiqi Wu et al., 2023)

{{<citation>}}

Weiqi Wu, Chengyue Jiang, Yong Jiang, Pengjun Xie, Kewei Tu. (2023)  
**Do PLMs Know and Understand Ontological Knowledge?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2309.05936v1)  

---


**ABSTRACT**  
Ontological knowledge, which comprises classes and properties and their relationships, is integral to world knowledge. It is significant to explore whether Pretrained Language Models (PLMs) know and understand such knowledge. However, existing PLM-probing studies focus mainly on factual knowledge, lacking a systematic probing of ontological knowledge. In this paper, we focus on probing whether PLMs store ontological knowledge and have a semantic understanding of the knowledge rather than rote memorization of the surface form. To probe whether PLMs know ontological knowledge, we investigate how well PLMs memorize: (1) types of entities; (2) hierarchical relationships among classes and properties, e.g., Person is a subclass of Animal and Member of Sports Team is a subproperty of Member of ; (3) domain and range constraints of properties, e.g., the subject of Member of Sports Team should be a Person and the object should be a Sports Team. To further probe whether PLMs truly understand ontological knowledge beyond memorization, we comprehensively study whether they can reliably perform logical reasoning with given knowledge according to ontological entailment rules. Our probing results show that PLMs can memorize certain ontological knowledge and utilize implicit knowledge in reasoning. However, both the memorizing and reasoning performances are less than perfect, indicating incomplete knowledge and understanding.

{{</citation>}}


## cs.IR (3)



### (92/117) Hierarchical Multi-Task Learning Framework for Session-based Recommendations (Sejoon Oh et al., 2023)

{{<citation>}}

Sejoon Oh, Walid Shalaby, Amir Afsharinejad, Xiquan Cui. (2023)  
**Hierarchical Multi-Task Learning Framework for Session-based Recommendations**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.06533v1)  

---


**ABSTRACT**  
While session-based recommender systems (SBRSs) have shown superior recommendation performance, multi-task learning (MTL) has been adopted by SBRSs to enhance their prediction accuracy and generalizability further. Hierarchical MTL (H-MTL) sets a hierarchical structure between prediction tasks and feeds outputs from auxiliary tasks to main tasks. This hierarchy leads to richer input features for main tasks and higher interpretability of predictions, compared to existing MTL frameworks. However, the H-MTL framework has not been investigated in SBRSs yet. In this paper, we propose HierSRec which incorporates the H-MTL architecture into SBRSs. HierSRec encodes a given session with a metadata-aware Transformer and performs next-category prediction (i.e., auxiliary task) with the session encoding. Next, HierSRec conducts next-item prediction (i.e., main task) with the category prediction result and session encoding. For scalable inference, HierSRec creates a compact set of candidate items (e.g., 4% of total items) per test example using the category prediction. Experiments show that HierSRec outperforms existing SBRSs as per next-item prediction accuracy on two session-based recommendation datasets. The accuracy of HierSRec measured with the carefully-curated candidate items aligns with the accuracy of HierSRec calculated with all items, which validates the usefulness of our candidate generation scheme via H-MTL.

{{</citation>}}


### (93/117) Annotating Data for Fine-Tuning a Neural Ranker? Current Active Learning Strategies are not Better than Random Selection (Sophia Althammer et al., 2023)

{{<citation>}}

Sophia Althammer, Guido Zuccon, Sebastian Hofstätter, Suzan Verberne, Allan Hanbury. (2023)  
**Annotating Data for Fine-Tuning a Neural Ranker? Current Active Learning Strategies are not Better than Random Selection**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Active Learning, Language Model, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2309.06131v1)  

---


**ABSTRACT**  
Search methods based on Pretrained Language Models (PLM) have demonstrated great effectiveness gains compared to statistical and early neural ranking models. However, fine-tuning PLM-based rankers requires a great amount of annotated training data. Annotating data involves a large manual effort and thus is expensive, especially in domain specific tasks. In this paper we investigate fine-tuning PLM-based rankers under limited training data and budget. We investigate two scenarios: fine-tuning a ranker from scratch, and domain adaptation starting with a ranker already fine-tuned on general data, and continuing fine-tuning on a target dataset. We observe a great variability in effectiveness when fine-tuning on different randomly selected subsets of training data. This suggests that it is possible to achieve effectiveness gains by actively selecting a subset of the training data that has the most positive effect on the rankers. This way, it would be possible to fine-tune effective PLM rankers at a reduced annotation budget. To investigate this, we adapt existing Active Learning (AL) strategies to the task of fine-tuning PLM rankers and investigate their effectiveness, also considering annotation and computational costs. Our extensive analysis shows that AL strategies do not significantly outperform random selection of training subsets in terms of effectiveness. We further find that gains provided by AL strategies come at the expense of more assessments (thus higher annotation costs) and AL strategies underperform random selection when comparing effectiveness given a fixed annotation cost. Our results highlight that ``optimal'' subsets of training data that provide high effectiveness at low annotation cost do exist, but current mainstream AL strategies applied to PLM rankers are not capable of identifying them.

{{</citation>}}


### (94/117) SAGE: Structured Attribute Value Generation for Billion-Scale Product Catalogs (Athanasios N. Nikolakopoulos et al., 2023)

{{<citation>}}

Athanasios N. Nikolakopoulos, Swati Kaul, Siva Karthik Gade, Bella Dubrov, Umit Batur, Suleiman Ali Khan. (2023)  
**SAGE: Structured Attribute Value Generation for Billion-Scale Product Catalogs**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs.IR  
Keywords: Seq2Seq  
[Paper Link](http://arxiv.org/abs/2309.05920v1)  

---


**ABSTRACT**  
We introduce SAGE; a Generative LLM for inferring attribute values for products across world-wide e-Commerce catalogs. We introduce a novel formulation of the attribute-value prediction problem as a Seq2Seq summarization task, across languages, product types and target attributes. Our novel modeling approach lifts the restriction of predicting attribute values within a pre-specified set of choices, as well as, the requirement that the sought attribute values need to be explicitly mentioned in the text. SAGE can infer attribute values even when such values are mentioned implicitly using periphrastic language, or not-at-all-as is the case for common-sense defaults. Additionally, SAGE is capable of predicting whether an attribute is inapplicable for the product at hand, or non-obtainable from the available information. SAGE is the first method able to tackle all aspects of the attribute-value-prediction task as they arise in practical settings in e-Commerce catalogs. A comprehensive set of experiments demonstrates the effectiveness of the proposed approach, as well as, its superiority against state-of-the-art competing alternatives. Moreover, our experiments highlight SAGE's ability to tackle the task of predicting attribute values in zero-shot setting; thereby, opening up opportunities for significantly reducing the overall number of labeled examples required for training.

{{</citation>}}


## cs.LO (1)



### (95/117) Compositional Separation of Control Flow and Data Flow (Damian Arellanes, 2023)

{{<citation>}}

Damian Arellanes. (2023)  
**Compositional Separation of Control Flow and Data Flow**  

---
Primary Category: cs.LO  
Categories: cs-LO, cs.LO  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2309.06397v1)  

---


**ABSTRACT**  
Every constructive model of computation (CMC) has an underlying composition mechanism for combining simple computation devices into more complex ones. Composition can be done by (explicitly or implicitly) defining control flow, data flow or any combination thereof. Control flow specifies the order in which individual computation devices are activated, whereas data flow defines how data is exchanged among them. Unfortunately, traditional CMCs either mix data and control or only consider one dimension explicitly, which makes it difficult to reason about data flow and control flow separately. Reasoning about these dimensions orthogonally is a crucial desideratum for optimisation, maintainability and verification purposes. In this paper, we introduce a novel model that explicitly treats data flow and control flow as separate dimensions, while providing modularity. As the model is rooted in category theory, it provides category-theoretic operations for compositionally constructing sequential or parallel composites. Compositionality entails that a composite exhibits the same properties as its respective constituents, including separation of concerns and modularity.

{{</citation>}}


## cs.HC (3)



### (96/117) Style2Fab: Functionality-Aware Segmentation for Fabricating Personalized 3D Models with Generative AI (Faraz Faruqi et al., 2023)

{{<citation>}}

Faraz Faruqi, Ahmed Katary, Tarik Hasic, Amira Abdel-Rahman, Nayeemur Rahman, Leandra Tejedor, Mackenzie Leake, Megan Hofmann, Stefanie Mueller. (2023)  
**Style2Fab: Functionality-Aware Segmentation for Fabricating Personalized 3D Models with Generative AI**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2309.06379v1)  

---


**ABSTRACT**  
With recent advances in Generative AI, it is becoming easier to automatically manipulate 3D models. However, current methods tend to apply edits to models globally, which risks compromising the intended functionality of the 3D model when fabricated in the physical world. For example, modifying functional segments in 3D models, such as the base of a vase, could break the original functionality of the model, thus causing the vase to fall over. We introduce a method for automatically segmenting 3D models into functional and aesthetic elements. This method allows users to selectively modify aesthetic segments of 3D models, without affecting the functional segments. To develop this method we first create a taxonomy of functionality in 3D models by qualitatively analyzing 1000 models sourced from a popular 3D printing repository, Thingiverse. With this taxonomy, we develop a semi-automatic classification method to decompose 3D models into functional and aesthetic elements. We propose a system called Style2Fab that allows users to selectively stylize 3D models without compromising their functionality. We evaluate the effectiveness of our classification method compared to human-annotated data, and demonstrate the utility of Style2Fab with a user study to show that functionality-aware segmentation helps preserve model functionality.

{{</citation>}}


### (97/117) Modeling Cognitive-Affective Processes with Appraisal and Reinforcement Learning (Jiayi Zhang et al., 2023)

{{<citation>}}

Jiayi Zhang, Joost Broekens, Jussi Jokinen. (2023)  
**Modeling Cognitive-Affective Processes with Appraisal and Reinforcement Learning**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.06367v1)  

---


**ABSTRACT**  
Computational models can advance affective science by shedding light onto the interplay between cognition and emotion from an information processing point of view. We propose a computational model of emotion that integrates reinforcement learning (RL) and appraisal theory, establishing a formal relationship between reward processing, goal-directed task learning, cognitive appraisal and emotional experiences. The model achieves this by formalizing evaluative checks from the component process model (CPM) in terms of temporal difference learning updates. We formalized novelty, goal relevance, goal conduciveness, and power. The formalization is task independent and can be applied to any task that can be represented as a Markov decision problem (MDP) and solved using RL. We investigated to what extent CPM-RL enables simulation of emotional responses cased by interactive task events. We evaluate the model by predicting a range of human emotions based on a series of vignette studies, highlighting its potential in improving our understanding of the role of reward processing in affective experiences.

{{</citation>}}


### (98/117) On the Injunction of XAIxArt (Cheshta Arora et al., 2023)

{{<citation>}}

Cheshta Arora, Debarun Sarkar. (2023)  
**On the Injunction of XAIxArt**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.06227v1)  

---


**ABSTRACT**  
The position paper highlights the range of concerns that are engulfed in the injunction of explainable artificial intelligence in art (XAIxArt). Through a series of quick sub-questions, it points towards the ambiguities concerning 'explanation' and the postpositivist tradition of 'relevant explanation'. Rejecting both 'explanation' and 'relevant explanation', the paper takes a stance that XAIxArt is a symptom of insecurity of the anthropocentric notion of art and a nostalgic desire to return to outmoded notions of authorship and human agency. To justify this stance, the paper makes a distinction between an ornamentation model of explanation to a model of explanation as sense-making.

{{</citation>}}


## cs.DB (2)



### (99/117) Enhancing In-Memory Spatial Indexing with Learned Search (Varun Pandey et al., 2023)

{{<citation>}}

Varun Pandey, Alexander van Renen, Eleni Tzirita Zacharatou, Andreas Kipf, Ibrahim Sabek, Jialin Ding, Volker Markl, Alfons Kemper. (2023)  
**Enhancing In-Memory Spatial Indexing with Learned Search**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2309.06354v1)  

---


**ABSTRACT**  
Spatial data is ubiquitous. Massive amounts of data are generated every day from a plethora of sources such as billions of GPS-enabled devices (e.g., cell phones, cars, and sensors), consumer-based applications (e.g., Uber and Strava), and social media platforms (e.g., location-tagged posts on Facebook, Twitter, and Instagram). This exponential growth in spatial data has led the research community to build systems and applications for efficient spatial data processing.   In this study, we apply a recently developed machine-learned search technique for single-dimensional sorted data to spatial indexing. Specifically, we partition spatial data using six traditional spatial partitioning techniques and employ machine-learned search within each partition to support point, range, distance, and spatial join queries. Adhering to the latest research trends, we tune the partitioning techniques to be instance-optimized. By tuning each partitioning technique for optimal performance, we demonstrate that: (i) grid-based index structures outperform tree-based index structures (from 1.23$\times$ to 2.47$\times$), (ii) learning-enhanced variants of commonly used spatial index structures outperform their original counterparts (from 1.44$\times$ to 53.34$\times$ faster), (iii) machine-learned search within a partition is faster than binary search by 11.79% - 39.51% when filtering on one dimension, (iv) the benefit of machine-learned search diminishes in the presence of other compute-intensive operations (e.g. scan costs in higher selectivity queries, Haversine distance computation, and point-in-polygon tests), and (v) index lookup is the bottleneck for tree-based structures, which could potentially be reduced by linearizing the indexed partitions.

{{</citation>}}


### (100/117) OmniSketch: Efficient Multi-Dimensional High-Velocity Stream Analytics with Arbitrary Predicates (Wieger R. Punter et al., 2023)

{{<citation>}}

Wieger R. Punter, Odysseas Papapetrou, Minos Garofalakis. (2023)  
**OmniSketch: Efficient Multi-Dimensional High-Velocity Stream Analytics with Arbitrary Predicates**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2309.06051v1)  

---


**ABSTRACT**  
A key need in different disciplines is to perform analytics over fast-paced data streams, similar in nature to the traditional OLAP analytics in relational databases i.e., with filters and aggregates. Storing unbounded streams, however, is not a realistic, or desired approach due to the high storage requirements, and the delays introduced when storing massive data. Accordingly, many synopses/sketches have been proposed that can summarize the stream in small memory (usually sufficiently small to be stored in RAM), such that aggregate queries can be efficiently approximated, without storing the full stream. However, past synopses predominantly focus on summarizing single-attribute streams, and cannot handle filters and constraints on arbitrary subsets of multiple attributes efficiently. In this work, we propose OmniSketch, the first sketch that scales to fast-paced and complex data streams (with many attributes), and supports aggregates with filters on multiple attributes, dynamically chosen at query time. The sketch offers probabilistic guarantees, a favorable space-accuracy tradeoff, and a worst-case logarithmic complexity for updating and for query execution. We demonstrate experimentally with both real and synthetic data that the sketch outperforms the state-of-the-art, and that it can approximate complex ad-hoc queries within the configured accuracy guarantees, with small memory requirements.

{{</citation>}}


## cs.NI (1)



### (101/117) Making Network Configuration Human Friendly (Changjie Wang et al., 2023)

{{<citation>}}

Changjie Wang, Mariano Scazzariello, Alireza Farshin, Dejan Kostic, Marco Chiesa. (2023)  
**Making Network Configuration Human Friendly**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.06342v1)  

---


**ABSTRACT**  
This paper explores opportunities to utilize Large Language Models (LLMs) to make network configuration human-friendly, simplifying the configuration of network devices and minimizing errors. We examine the effectiveness of these models in translating high-level policies and requirements (i.e., specified in natural language) into low-level network APIs, which requires understanding the hardware and protocols. More specifically, we propose NETBUDDY for generating network configurations from scratch and modifying them at runtime. NETBUDDY splits the generation of network configurations into fine-grained steps and relies on self-healing code-generation approaches to better take advantage of the full potential of LLMs. We first thoroughly examine the challenges of using these models to produce a fully functional & correct configuration, and then evaluate the feasibility of realizing NETBUDDY by building a proof-of-concept solution using GPT-4 to translate a set of high-level requirements into P4 and BGP configurations and run them using the Kathar\'a network emulator.

{{</citation>}}


## eess.IV (2)



### (102/117) ssVERDICT: Self-Supervised VERDICT-MRI for Enhanced Prostate Tumour Characterisation (Snigdha Sen et al., 2023)

{{<citation>}}

Snigdha Sen, Saurabh Singh, Hayley Pye, Caroline Moore, Hayley Whitaker, Shonit Punwani, David Atkinson, Eleftheria Panagiotaki, Paddy J. Slator. (2023)  
**ssVERDICT: Self-Supervised VERDICT-MRI for Enhanced Prostate Tumour Characterisation**  

---
Primary Category: eess.IV  
Categories: cs-LG, eess-IV, eess.IV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.06268v1)  

---


**ABSTRACT**  
MRI is increasingly being used in the diagnosis of prostate cancer (PCa), with diffusion MRI (dMRI) playing an integral role. When combined with computational models, dMRI can estimate microstructural information such as cell size. Conventionally, such models are fit with a nonlinear least squares (NLLS) curve fitting approach, associated with a high computational cost. Supervised deep neural networks (DNNs) are an efficient alternative, however their performance is significantly affected by the underlying distribution of the synthetic training data. Self-supervised learning is an attractive alternative, where instead of using a separate training dataset, the network learns the features of the input data itself. This approach has only been applied to fitting of trivial dMRI models thus far. Here, we introduce a self-supervised DNN to estimate the parameters of the VERDICT (Vascular, Extracellular and Restricted DIffusion for Cytometry in Tumours) model for prostate. We demonstrate, for the first time, fitting of a complex three-compartment biophysical model with machine learning without the requirement of explicit training labels. We compare the estimation performance to baseline NLLS and supervised DNN methods, observing improvement in estimation accuracy and reduction in bias with respect to ground truth values. Our approach also achieves a higher confidence level for discrimination between cancerous and benign prostate tissue in comparison to the other methods on a dataset of 20 PCa patients, indicating potential for accurate tumour characterisation.

{{</citation>}}


### (103/117) A2V: A Semi-Supervised Domain Adaptation Framework for Brain Vessel Segmentation via Two-Phase Training Angiography-to-Venography Translation (Francesco Galati et al., 2023)

{{<citation>}}

Francesco Galati, Daniele Falcetta, Rosa Cortese, Barbara Casolla, Ferran Prados, Ninon Burgos, Maria A. Zuluaga. (2023)  
**A2V: A Semi-Supervised Domain Adaptation Framework for Brain Vessel Segmentation via Two-Phase Training Angiography-to-Venography Translation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2309.06075v1)  

---


**ABSTRACT**  
We present a semi-supervised domain adaptation framework for brain vessel segmentation from different image modalities. Existing state-of-the-art methods focus on a single modality, despite the wide range of available cerebrovascular imaging techniques. This can lead to significant distribution shifts that negatively impact the generalization across modalities. By relying on annotated angiographies and a limited number of annotated venographies, our framework accomplishes image-to-image translation and semantic segmentation, leveraging a disentangled and semantically rich latent space to represent heterogeneous data and perform image-level adaptation from source to target domains. Moreover, we reduce the typical complexity of cycle-based architectures and minimize the use of adversarial training, which allows us to build an efficient and intuitive model with stable training. We evaluate our method on magnetic resonance angiographies and venographies. While achieving state-of-the-art performance in the source domain, our method attains a Dice score coefficient in the target domain that is only 8.9% lower, highlighting its promising potential for robust cerebrovascular image segmentation across different modalities.

{{</citation>}}


## physics.flu-dyn (1)



### (104/117) Toward Discretization-Consistent Closure Schemes for Large Eddy Simulation Using Reinforcement Learning (Andrea Beck et al., 2023)

{{<citation>}}

Andrea Beck, Marius Kurz. (2023)  
**Toward Discretization-Consistent Closure Schemes for Large Eddy Simulation Using Reinforcement Learning**  

---
Primary Category: physics.flu-dyn  
Categories: cs-LG, physics-flu-dyn, physics.flu-dyn  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.06260v1)  

---


**ABSTRACT**  
We propose a novel method for developing discretization-consistent closure schemes for implicitly filtered Large Eddy Simulation (LES). In implicitly filtered LES, the induced filter kernel, and thus the closure terms, are determined by the properties of the grid and the discretization operator, leading to additional computational subgrid terms that are generally unknown in a priori analysis. Therefore, the task of adapting the coefficients of LES closure models is formulated as a Markov decision process and solved in an a posteriori manner with Reinforcement Learning (RL). This allows to adjust the model to the actual discretization as it also incorporates the interaction between the discretization and the model itself. This optimization framework is applied to both explicit and implicit closure models. An element-local eddy viscosity model is optimized as the explicit model. For the implicit modeling, RL is applied to identify an optimal blending strategy for a hybrid discontinuous Galerkin (DG) and finite volume scheme. All newly derived models achieve accurate and consistent results, either matching or outperforming classical state-of-the-art models for different discretizations and resolutions. Moreover, the explicit model is demonstrated to adapt its distribution of viscosity within the DG elements to the inhomogeneous discretization properties of the operator. In the implicit case, the optimized hybrid scheme renders itself as a viable modeling ansatz that could initiate a new class of high order schemes for compressible turbulence. Overall, the results demonstrate that the proposed RL optimization can provide discretization-consistent closures that could reduce the uncertainty in implicitly filtered LES.

{{</citation>}}


## cs.CY (1)



### (105/117) Cookiescanner: An Automated Tool for Detecting and Evaluating GDPR Consent Notices on Websites (Ralf Gundelach et al., 2023)

{{<citation>}}

Ralf Gundelach, Dominik Herrmann. (2023)  
**Cookiescanner: An Automated Tool for Detecting and Evaluating GDPR Consent Notices on Websites**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.06196v1)  

---


**ABSTRACT**  
The enforcement of the GDPR led to the widespread adoption of consent notices, colloquially known as cookie banners. Studies have shown that many website operators do not comply with the law and track users prior to any interaction with the consent notice, or attempt to trick users into giving consent through dark patterns. Previous research has relied on manually curated filter lists or automated detection methods limited to a subset of websites, making research on GDPR compliance of consent notices tedious or limited. We present \emph{cookiescanner}, an automated scanning tool that detects and extracts consent notices via various methods and checks if they offer a decline option or use color diversion. We evaluated cookiescanner on a random sample of the top 10,000 websites listed by Tranco. We found that manually curated filter lists have the highest precision but recall fewer consent notices than our keyword-based methods. Our BERT model achieves high precision for English notices, which is in line with previous work, but suffers from low recall due to insufficient candidate extraction. While the automated detection of decline options proved to be challenging due to the dynamic nature of many sites, detecting instances of different colors of the buttons was successful in most cases. Besides systematically evaluating our various detection techniques, we have manually annotated 1,000 websites to provide a ground-truth baseline, which has not existed previously. Furthermore, we release our code and the annotated dataset in the interest of reproducibility and repeatability.

{{</citation>}}


## eess.AS (1)



### (106/117) Assessing the Generalization Gap of Learning-Based Speech Enhancement Systems in Noisy and Reverberant Environments (Philippe Gonzalez et al., 2023)

{{<citation>}}

Philippe Gonzalez, Tommy Sonne Alstrøm, Tobias May. (2023)  
**Assessing the Generalization Gap of Learning-Based Speech Enhancement Systems in Noisy and Reverberant Environments**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2309.06183v1)  

---


**ABSTRACT**  
The acoustic variability of noisy and reverberant speech mixtures is influenced by multiple factors, such as the spectro-temporal characteristics of the target speaker and the interfering noise, the signal-to-noise ratio (SNR) and the room characteristics. This large variability poses a major challenge for learning-based speech enhancement systems, since a mismatch between the training and testing conditions can substantially reduce the performance of the system. Generalization to unseen conditions is typically assessed by testing the system with a new speech, noise or binaural room impulse response (BRIR) database different from the one used during training. However, the difficulty of the speech enhancement task can change across databases, which can substantially influence the results. The present study introduces a generalization assessment framework that uses a reference model trained on the test condition, such that it can be used as a proxy for the difficulty of the test condition. This allows to disentangle the effect of the change in task difficulty from the effect of dealing with new data, and thus to define a new measure of generalization performance termed the generalization gap. The procedure is repeated in a cross-validation fashion by cycling through multiple speech, noise, and BRIR databases to accurately estimate the generalization gap. The proposed framework is applied to evaluate the generalization potential of a feedforward neural network (FFNN), Conv-TasNet, DCCRN and MANNER. We find that for all models, the performance degrades the most in speech mismatches, while good noise and room generalization can be achieved by training on multiple databases. Moreover, while recent models show higher performance in matched conditions, their performance substantially decreases in mismatched conditions and can become inferior to that of the FFNN-based system.

{{</citation>}}


## cs.DL (1)



### (107/117) A comparison of citation-based clustering and topic modeling for science mapping (Qianqian Xie et al., 2023)

{{<citation>}}

Qianqian Xie, Ludo Waltman. (2023)  
**A comparison of citation-based clustering and topic modeling for science mapping**  

---
Primary Category: cs.DL  
Categories: cs-DL, cs.DL  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2309.06160v1)  

---


**ABSTRACT**  
Science mapping is an important tool to gain insight into scientific fields, to identify emerging research trends, and to support science policy. Understanding the different ways in which different science mapping approaches capture the structure of scientific fields is critical. This paper presents a comparative analysis of two commonly used approaches, topic modeling (TM) and citation-based clustering (CC), to assess their respective strengths, weaknesses, and the characteristics of their results. We compare the two approaches using cluster-to-topic and topic-to-cluster mappings based on science maps of cardiovascular research (CVR) generated by TM and CC. Our findings reveal that relations between topics and clusters are generally weak, with limited overlap between topics and clusters. Only in a few exceptional cases do more than one-third of the documents in a topic belong to the same cluster, or vice versa. CC excels at identifying diseases and generating specialized clusters in Clinical Treatment & Surgical Procedures, while TM focuses on sub-techniques within diagnostic techniques, provides a general perspective on Clinical Treatment & Surgical Procedures, and identifies distinct topics related to practical guidelines. Our work enhances the understanding of science mapping approaches based on TM and CC and delivers practical guidance for scientometricians on how to apply these approaches effectively.

{{</citation>}}


## cs.AR (1)



### (108/117) Accelerating Edge AI with Morpher: An Integrated Design, Compilation and Simulation Framework for CGRAs (Dhananjaya Wijerathne et al., 2023)

{{<citation>}}

Dhananjaya Wijerathne, Zhaoying Li, Tulika Mitra. (2023)  
**Accelerating Edge AI with Morpher: An Integrated Design, Compilation and Simulation Framework for CGRAs**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-LG, cs-PF, cs.AR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.06127v1)  

---


**ABSTRACT**  
Coarse-Grained Reconfigurable Arrays (CGRAs) hold great promise as power-efficient edge accelerator, offering versatility beyond AI applications. Morpher, an open-source, architecture-adaptive CGRA design framework, is specifically designed to explore the vast design space of CGRAs. The comprehensive ecosystem of Morpher includes a tailored compiler, simulator, accelerator synthesis, and validation framework. This study provides an overview of Morpher, highlighting its capabilities in automatically compiling AI application kernels onto user-defined CGRA architectures and verifying their functionality. Through the Morpher framework, the versatility of CGRAs is harnessed to facilitate efficient compilation and verification of edge AI applications, covering important kernels representative of a wide range of embedded AI workloads. Morpher is available online at https://github.com/ecolab-nus/morpher-v2.

{{</citation>}}


## astro-ph.IM (1)



### (109/117) AstroLLaMA: Towards Specialized Foundation Models in Astronomy (Tuan Dung Nguyen et al., 2023)

{{<citation>}}

Tuan Dung Nguyen, Yuan-Sen Ting, Ioana Ciucă, Charlie O'Neill, Ze-Chang Sun, Maja Jabłońska, Sandor Kruk, Ernest Perkowski, Jack Miller, Jason Li, Josh Peek, Kartheik Iyer, Tomasz Różański, Pranav Khetarpal, Sharaf Zaman, David Brodrick, Sergio J. Rodríguez Méndez, Thang Bui, Alyssa Goodman, Alberto Accomazzi, Jill Naiman, Jesse Cranney, Kevin Schawinski, UniverseTBD. (2023)  
**AstroLLaMA: Towards Specialized Foundation Models in Astronomy**  

---
Primary Category: astro-ph.IM  
Categories: astro-ph-CO, astro-ph-GA, astro-ph-HE, astro-ph-IM, astro-ph.IM, cs-CL, cs-LG  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2309.06126v1)  

---


**ABSTRACT**  
Large language models excel in many human-language tasks but often falter in highly specialized domains like scholarly astronomy. To bridge this gap, we introduce AstroLLaMA, a 7-billion-parameter model fine-tuned from LLaMA-2 using over 300,000 astronomy abstracts from arXiv. Optimized for traditional causal language modeling, AstroLLaMA achieves a 30% lower perplexity than Llama-2, showing marked domain adaptation. Our model generates more insightful and scientifically relevant text completions and embedding extraction than state-of-the-arts foundation models despite having significantly fewer parameters. AstroLLaMA serves as a robust, domain-specific model with broad fine-tuning potential. Its public release aims to spur astronomy-focused research, including automatic paper summarization and conversational agent development.

{{</citation>}}


## cs.CR (4)



### (110/117) Backdoor Attacks and Countermeasures in Natural Language Processing Models: A Comprehensive Security Review (Pengzhou Cheng et al., 2023)

{{<citation>}}

Pengzhou Cheng, Zongru Wu, Wei Du, Gongshen Liu. (2023)  
**Backdoor Attacks and Countermeasures in Natural Language Processing Models: A Comprehensive Security Review**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: NLP, Natural Language Processing, Security  
[Paper Link](http://arxiv.org/abs/2309.06055v2)  

---


**ABSTRACT**  
Deep Neural Networks (DNNs) have led to unprecedented progress in various natural language processing (NLP) tasks. Owing to limited data and computation resources, using third-party data and models has become a new paradigm for adapting various tasks. However, research shows that it has some potential security vulnerabilities because attackers can manipulate the training process and data source. Such a way can set specific triggers, making the model exhibit expected behaviors that have little inferior influence on the model's performance for primitive tasks, called backdoor attacks. Hence, it could have dire consequences, especially considering that the backdoor attack surfaces are broad.   To get a precise grasp and understanding of this problem, a systematic and comprehensive review is required to confront various security challenges from different phases and attack purposes. Additionally, there is a dearth of analysis and comparison of the various emerging backdoor countermeasures in this situation. In this paper, we conduct a timely review of backdoor attacks and countermeasures to sound the red alarm for the NLP security community. According to the affected stage of the machine learning pipeline, the attack surfaces are recognized to be wide and then formalized into three categorizations: attacking pre-trained model with fine-tuning (APMF) or prompt-tuning (APMP), and attacking final model with training (AFMT), where AFMT can be subdivided into different attack aims. Thus, attacks under each categorization are combed. The countermeasures are categorized into two general classes: sample inspection and model inspection. Overall, the research on the defense side is far behind the attack side, and there is no single defense that can prevent all types of backdoor attacks. An attacker can intelligently bypass existing defenses with a more invisible attack. ......

{{</citation>}}


### (111/117) Catch You Everything Everywhere: Guarding Textual Inversion via Concept Watermarking (Weitao Feng et al., 2023)

{{<citation>}}

Weitao Feng, Jiyan He, Jie Zhang, Tianwei Zhang, Wenbo Zhou, Weiming Zhang, Nenghai Yu. (2023)  
**Catch You Everything Everywhere: Guarding Textual Inversion via Concept Watermarking**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.05940v1)  

---


**ABSTRACT**  
AIGC (AI-Generated Content) has achieved tremendous success in many applications such as text-to-image tasks, where the model can generate high-quality images with diverse prompts, namely, different descriptions in natural languages. More surprisingly, the emerging personalization techniques even succeed in describing unseen concepts with only a few personal images as references, and there have been some commercial platforms for sharing the valuable personalized concept. However, such an advanced technique also introduces a severe threat, where malicious users can misuse the target concept to generate highly-realistic illegal images. Therefore, it becomes necessary for the platform to trace malicious users and hold them accountable.   In this paper, we focus on guarding the most popular lightweight personalization model, ie, Textual Inversion (TI). To achieve it, we propose the novel concept watermarking, where watermark information is embedded into the target concept and then extracted from generated images based on the watermarked concept. Specifically, we jointly train a watermark encoder and a watermark decoder with the sampler in the loop.   It shows great resilience to different diffusion sampling processes possibly chosen by malicious users, meanwhile preserving utility for normal use. In practice, the concept owner can upload his concept with different watermarks (ie, serial numbers) to the platform, and the platform allocates different users with different serial numbers for subsequent tracing and forensics.

{{</citation>}}


### (112/117) Behind The Wings: The Case of Reverse Engineering and Drone Hijacking in DJI Enhanced Wi-Fi Protocol (Derry Pratama et al., 2023)

{{<citation>}}

Derry Pratama, Jaegeun Moon, Agus Mahardika Ari Laksmono, Dongwook Yun, Iqbal Muhammad, Byeonguk Jeong, Janghyun Ji, Howon Kim. (2023)  
**Behind The Wings: The Case of Reverse Engineering and Drone Hijacking in DJI Enhanced Wi-Fi Protocol**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2309.05913v1)  

---


**ABSTRACT**  
This research paper entails an examination of the Enhanced Wi-Fi protocol, focusing on its control command reverse-engineering analysis and subsequent demonstration of a hijacking attack. Our investigation discovered vulnerabilities in the Enhanced Wi-Fi control commands, rendering them susceptible to hijacking attacks. Notably, the study established that even readily available and cost-effective commercial off-the-shelf Wi-Fi routers could be leveraged as effective tools for executing such attacks. To illustrate this vulnerability, a proof-of-concept remote hijacking attack was carried out on a DJI Mini SE drone, whereby we intercepted the control commands to manipulate the drone's flight trajectory. The findings of this research emphasize the critical necessity of implementing robust security measures to safeguard unmanned aerial vehicles against potential hijacking threats. Considering that civilian drones are now used as war weapons, the study underscores the urgent need for further exploration and advancement in the domain of civilian drone security.

{{</citation>}}


### (113/117) Generalized Attacks on Face Verification Systems (Ehsan Nazari et al., 2023)

{{<citation>}}

Ehsan Nazari, Paula Branco, Guy-Vincent Jourdan. (2023)  
**Generalized Attacks on Face Verification Systems**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs-LG, cs.CR  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2309.05879v1)  

---


**ABSTRACT**  
Face verification (FV) using deep neural network models has made tremendous progress in recent years, surpassing human accuracy and seeing deployment in various applications such as border control and smartphone unlocking. However, FV systems are vulnerable to Adversarial Attacks, which manipulate input images to deceive these systems in ways usually unnoticeable to humans. This paper provides an in-depth study of attacks on FV systems. We introduce the DodgePersonation Attack that formulates the creation of face images that impersonate a set of given identities while avoiding being identified as any of the identities in a separate, disjoint set. A taxonomy is proposed to provide a unified view of different types of Adversarial Attacks against FV systems, including Dodging Attacks, Impersonation Attacks, and Master Face Attacks. Finally, we propose the ''One Face to Rule Them All'' Attack which implements the DodgePersonation Attack with state-of-the-art performance on a well-known scenario (Master Face Attack) and which can also be used for the new scenarios introduced in this paper. While the state-of-the-art Master Face Attack can produce a set of 9 images to cover 43.82% of the identities in their test database, with 9 images our attack can cover 57.27% to 58.5% of these identifies while giving the attacker the choice of the identity to use to create the impersonation. Moreover, the 9 generated attack images appear identical to a casual observer.

{{</citation>}}


## cs.SI (1)



### (114/117) Evaluating the Ebb and Flow: An In-depth Analysis of Question-Answering Trends across Diverse Platforms (Rima Hazra et al., 2023)

{{<citation>}}

Rima Hazra, Agnik Saha, Somnath Banerjee, Animesh Mukherjee. (2023)  
**Evaluating the Ebb and Flow: An In-depth Analysis of Question-Answering Trends across Diverse Platforms**  

---
Primary Category: cs.SI  
Categories: cs-CL, cs-IR, cs-LG, cs-SI, cs.SI  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.05961v1)  

---


**ABSTRACT**  
Community Question Answering (CQA) platforms steadily gain popularity as they provide users with fast responses to their queries. The swiftness of these responses is contingent on a mixture of query-specific and user-related elements. This paper scrutinizes these contributing factors within the context of six highly popular CQA platforms, identified through their standout answering speed. Our investigation reveals a correlation between the time taken to yield the first response to a question and several variables: the metadata, the formulation of the questions, and the level of interaction among users. Additionally, by employing conventional machine learning models to analyze these metadata and patterns of user interaction, we endeavor to predict which queries will receive their initial responses promptly.

{{</citation>}}


## cs.GR (1)



### (115/117) GA-Sketching: Shape Modeling from Multi-View Sketching with Geometry-Aligned Deep Implicit Functions (Jie Zhou et al., 2023)

{{<citation>}}

Jie Zhou, Zhongjin Luo, Qian Yu, Xiaoguang Han, Hongbo Fu. (2023)  
**GA-Sketching: Shape Modeling from Multi-View Sketching with Geometry-Aligned Deep Implicit Functions**  

---
Primary Category: cs.GR  
Categories: cs-GR, cs.GR  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2309.05946v1)  

---


**ABSTRACT**  
Sketch-based shape modeling aims to bridge the gap between 2D drawing and 3D modeling by providing an intuitive and accessible approach to create 3D shapes from 2D sketches. However, existing methods still suffer from limitations in reconstruction quality and multi-view interaction friendliness, hindering their practical application. This paper proposes a faithful and user-friendly iterative solution to tackle these limitations by learning geometry-aligned deep implicit functions from one or multiple sketches. Our method lifts 2D sketches to volume-based feature tensors, which align strongly with the output 3D shape, enabling accurate reconstruction and faithful editing. Such a geometry-aligned feature encoding technique is well-suited to iterative modeling since features from different viewpoints can be easily memorized or aggregated. Based on these advantages, we design a unified interactive system for sketch-based shape modeling. It enables users to generate the desired geometry iteratively by drawing sketches from any number of viewpoints. In addition, it allows users to edit the generated surface by making a few local modifications. We demonstrate the effectiveness and practicality of our method with extensive experiments and user studies, where we found that our method outperformed existing methods in terms of accuracy, efficiency, and user satisfaction. The source code of this project is available at https://github.com/LordLiang/GA-Sketching.

{{</citation>}}


## cond-mat.mes-hall (1)



### (116/117) Quantized Non-Volatile Nanomagnetic Synapse based Autoencoder for Efficient Unsupervised Network Anomaly Detection (Muhammad Sabbir Alam et al., 2023)

{{<citation>}}

Muhammad Sabbir Alam, Walid Al Misba, Jayasimha Atulasimha. (2023)  
**Quantized Non-Volatile Nanomagnetic Synapse based Autoencoder for Efficient Unsupervised Network Anomaly Detection**  

---
Primary Category: cond-mat.mes-hall  
Categories: cond-mat-mes-hall, cond-mat.mes-hall, cs-LG, cs-NE  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2309.06449v1)  

---


**ABSTRACT**  
In the autoencoder based anomaly detection paradigm, implementing the autoencoder in edge devices capable of learning in real-time is exceedingly challenging due to limited hardware, energy, and computational resources. We show that these limitations can be addressed by designing an autoencoder with low-resolution non-volatile memory-based synapses and employing an effective quantized neural network learning algorithm. We propose a ferromagnetic racetrack with engineered notches hosting a magnetic domain wall (DW) as the autoencoder synapses, where limited state (5-state) synaptic weights are manipulated by spin orbit torque (SOT) current pulses. The performance of anomaly detection of the proposed autoencoder model is evaluated on the NSL-KDD dataset. Limited resolution and DW device stochasticity aware training of the autoencoder is performed, which yields comparable anomaly detection performance to the autoencoder having floating-point precision weights. While the limited number of quantized states and the inherent stochastic nature of DW synaptic weights in nanoscale devices are known to negatively impact the performance, our hardware-aware training algorithm is shown to leverage these imperfect device characteristics to generate an improvement in anomaly detection accuracy (90.98%) compared to accuracy obtained with floating-point trained weights. Furthermore, our DW-based approach demonstrates a remarkable reduction of at least three orders of magnitude in weight updates during training compared to the floating-point approach, implying substantial energy savings for our method. This work could stimulate the development of extremely energy efficient non-volatile multi-state synapse-based processors that can perform real-time training and inference on the edge with unsupervised data.

{{</citation>}}


## cs.GT (1)



### (117/117) Strategic Behavior of Large Language Models: Game Structure vs. Contextual Framing (Nunzio Lorè et al., 2023)

{{<citation>}}

Nunzio Lorè, Babak Heydari. (2023)  
**Strategic Behavior of Large Language Models: Game Structure vs. Contextual Framing**  

---
Primary Category: cs.GT  
Categories: 91C99 (Primary), 91A05, 91A10, 91F99 (Secondary), I-2-8; J-4; K-4-m, cs-AI, cs-CY, cs-GT, cs-HC, cs.GT, econ-TH  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.05898v1)  

---


**ABSTRACT**  
This paper investigates the strategic decision-making capabilities of three Large Language Models (LLMs): GPT-3.5, GPT-4, and LLaMa-2, within the framework of game theory. Utilizing four canonical two-player games -- Prisoner's Dilemma, Stag Hunt, Snowdrift, and Prisoner's Delight -- we explore how these models navigate social dilemmas, situations where players can either cooperate for a collective benefit or defect for individual gain. Crucially, we extend our analysis to examine the role of contextual framing, such as diplomatic relations or casual friendships, in shaping the models' decisions. Our findings reveal a complex landscape: while GPT-3.5 is highly sensitive to contextual framing, it shows limited ability to engage in abstract strategic reasoning. Both GPT-4 and LLaMa-2 adjust their strategies based on game structure and context, but LLaMa-2 exhibits a more nuanced understanding of the games' underlying mechanics. These results highlight the current limitations and varied proficiencies of LLMs in strategic decision-making, cautioning against their unqualified use in tasks requiring complex strategic reasoning.

{{</citation>}}
