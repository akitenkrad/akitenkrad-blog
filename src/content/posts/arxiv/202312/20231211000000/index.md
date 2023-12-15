---
draft: false
title: "arXiv @ 2023.12.11"
date: 2023-12-11
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.11"
    identifier: arxiv_20231211
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.IR (2)](#csir-2)
- [eess.IV (2)](#eessiv-2)
- [cs.LG (16)](#cslg-16)
- [cs.SE (3)](#csse-3)
- [cs.CV (11)](#cscv-11)
- [cs.DL (1)](#csdl-1)
- [cs.AI (15)](#csai-15)
- [cs.CY (2)](#cscy-2)
- [eess.SY (1)](#eesssy-1)
- [cs.CL (15)](#cscl-15)
- [cs.SI (2)](#cssi-2)
- [cs.CR (3)](#cscr-3)
- [stat.ML (2)](#statml-2)
- [cs.NE (1)](#csne-1)
- [cs.SD (1)](#cssd-1)
- [cs.DC (1)](#csdc-1)
- [cs.NI (1)](#csni-1)
- [eess.SP (1)](#eesssp-1)

## cs.IR (2)



### (1/80) Context Tuning for Retrieval Augmented Generation (Raviteja Anantha et al., 2023)

{{<citation>}}

Raviteja Anantha, Tharun Bethi, Danil Vodianik, Srinivas Chappidi. (2023)  
**Context Tuning for Retrieval Augmented Generation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.05708v1)  

---


**ABSTRACT**  
Large language models (LLMs) have the remarkable ability to solve new tasks with just a few examples, but they need access to the right tools. Retrieval Augmented Generation (RAG) addresses this problem by retrieving a list of relevant tools for a given task. However, RAG's tool retrieval step requires all the required information to be explicitly present in the query. This is a limitation, as semantic search, the widely adopted tool retrieval method, can fail when the query is incomplete or lacks context. To address this limitation, we propose Context Tuning for RAG, which employs a smart context retrieval system to fetch relevant information that improves both tool retrieval and plan generation. Our lightweight context retrieval model uses numerical, categorical, and habitual usage signals to retrieve and rank context items. Our empirical results demonstrate that context tuning significantly enhances semantic search, achieving a 3.5-fold and 1.5-fold improvement in Recall@K for context retrieval and tool retrieval tasks respectively, and resulting in an 11.6% increase in LLM-based planner accuracy. Additionally, we show that our proposed lightweight model using Reciprocal Rank Fusion (RRF) with LambdaMART outperforms GPT-4 based retrieval. Moreover, we observe context augmentation at plan generation, even after tool retrieval, reduces hallucination.

{{</citation>}}


### (2/80) ESPN: Memory-Efficient Multi-Vector Information Retrieval (Susav Shrestha et al., 2023)

{{<citation>}}

Susav Shrestha, Narasimha Reddy, Zongwang Li. (2023)  
**ESPN: Memory-Efficient Multi-Vector Information Retrieval**  

---
Primary Category: cs.IR  
Categories: H-3-2; H-3-3; H-3-4; I-7-0; I-2-7, cs-IR, cs-LG, cs.IR  
Keywords: Embedding, Information Retrieval  
[Paper Link](http://arxiv.org/abs/2312.05417v1)  

---


**ABSTRACT**  
Recent advances in large language models have demonstrated remarkable effectiveness in information retrieval (IR) tasks. While many neural IR systems encode queries and documents into single-vector representations, multi-vector models elevate the retrieval quality by producing multi-vector representations and facilitating similarity searches at the granularity of individual tokens. However, these models significantly amplify memory and storage requirements for retrieval indices by an order of magnitude. This escalation in index size renders the scalability of multi-vector IR models progressively challenging due to their substantial memory demands. We introduce Embedding from Storage Pipelined Network (ESPN) where we offload the entire re-ranking embedding tables to SSDs and reduce the memory requirements by 5-16x. We design a software prefetcher with hit rates exceeding 90%, improving SSD based retrieval up to 6.4x, and demonstrate that we can maintain near memory levels of query latency even for large query batch sizes.

{{</citation>}}


## eess.IV (2)



### (3/80) Non-Cartesian Self-Supervised Physics-Driven Deep Learning Reconstruction for Highly-Accelerated Multi-Echo Spiral fMRI (Hongyi Gu et al., 2023)

{{<citation>}}

Hongyi Gu, Chi Zhang, Zidan Yu, Christoph Rettenmeier, V. Andrew Stenger, Mehmet Akçakaya. (2023)  
**Non-Cartesian Self-Supervised Physics-Driven Deep Learning Reconstruction for Highly-Accelerated Multi-Echo Spiral fMRI**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess-SP, eess.IV, physics-med-ph  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.05707v1)  

---


**ABSTRACT**  
Functional MRI (fMRI) is an important tool for non-invasive studies of brain function. Over the past decade, multi-echo fMRI methods that sample multiple echo times has become popular with potential to improve quantification. While these acquisitions are typically performed with Cartesian trajectories, non-Cartesian trajectories, in particular spiral acquisitions, hold promise for denser sampling of echo times. However, such acquisitions require very high acceleration rates for sufficient spatiotemporal resolutions. In this work, we propose to use a physics-driven deep learning (PD-DL) reconstruction to accelerate multi-echo spiral fMRI by 10-fold. We modify a self-supervised learning algorithm for optimized training with non-Cartesian trajectories and use it to train the PD-DL network. Results show that the proposed self-supervised PD-DL reconstruction achieves high spatio-temporal resolution with meaningful BOLD analysis.

{{</citation>}}


### (4/80) Exploring 3D U-Net Training Configurations and Post-Processing Strategies for the MICCAI 2023 Kidney and Tumor Segmentation Challenge (Kwang-Hyun Uhm et al., 2023)

{{<citation>}}

Kwang-Hyun Uhm, Hyunjun Cho, Zhixin Xu, Seohoon Lim, Seung-Won Jung, Sung-Hoo Hong, Sung-Jea Ko. (2023)  
**Exploring 3D U-Net Training Configurations and Post-Processing Strategies for the MICCAI 2023 Kidney and Tumor Segmentation Challenge**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.05528v1)  

---


**ABSTRACT**  
In 2023, it is estimated that 81,800 kidney cancer cases will be newly diagnosed, and 14,890 people will die from this cancer in the United States. Preoperative dynamic contrast-enhanced abdominal computed tomography (CT) is often used for detecting lesions. However, there exists inter-observer variability due to subtle differences in the imaging features of kidney and kidney tumors. In this paper, we explore various 3D U-Net training configurations and effective post-processing strategies for accurate segmentation of kidneys, cysts, and kidney tumors in CT images. We validated our model on the dataset of the 2023 Kidney and Kidney Tumor Segmentation (KiTS23) challenge. Our method took second place in the final ranking of the KiTS23 challenge on unseen test data with an average Dice score of 0.820 and an average Surface Dice of 0.712.

{{</citation>}}


## cs.LG (16)



### (5/80) Unsupervised Multi-modal Feature Alignment for Time Series Representation Learning (Chen Liang et al., 2023)

{{<citation>}}

Chen Liang, Donghua Yang, Zhiyu Liang, Hongzhi Wang, Zheng Liang, Xiyang Zhang, Jianfeng Huang. (2023)  
**Unsupervised Multi-modal Feature Alignment for Time Series Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2312.05698v1)  

---


**ABSTRACT**  
In recent times, the field of unsupervised representation learning (URL) for time series data has garnered significant interest due to its remarkable adaptability across diverse downstream applications. Unsupervised learning goals differ from downstream tasks, making it tricky to ensure downstream task utility by focusing only on temporal feature characterization. Researchers have proposed multiple transformations to extract discriminative patterns implied in informative time series, trying to fill the gap. Despite the introduction of a variety of feature engineering techniques, e.g. spectral domain, wavelet transformed features, features in image form and symbolic features etc. the utilization of intricate feature fusion methods and dependence on heterogeneous features during inference hampers the scalability of the solutions. To address this, our study introduces an innovative approach that focuses on aligning and binding time series representations encoded from different modalities, inspired by spectral graph theory, thereby guiding the neural encoder to uncover latent pattern associations among these multi-modal features. In contrast to conventional methods that fuse features from multiple modalities, our proposed approach simplifies the neural architecture by retaining a single time series encoder, consequently leading to preserved scalability. We further demonstrate and prove mechanisms for the encoder to maintain better inductive bias. In our experimental evaluation, we validated the proposed method on a diverse set of time series datasets from various domains. Our approach outperforms existing state-of-the-art URL methods across diverse downstream tasks.

{{</citation>}}


### (6/80) Agile-Quant: Activation-Guided Quantization for Faster Inference of LLMs on the Edge (Xuan Shen et al., 2023)

{{<citation>}}

Xuan Shen, Peiyan Dong, Lei Lu, Zhenglun Kong, Zhengang Li, Ming Lin, Chao Wu, Yanzhi Wang. (2023)  
**Agile-Quant: Activation-Guided Quantization for Faster Inference of LLMs on the Edge**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: BLOOM, LLaMA, Language Model, Quantization  
[Paper Link](http://arxiv.org/abs/2312.05693v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) stand out for their impressive performance in intricate language modeling tasks. However, their demanding computational and memory needs pose obstacles for broad use on edge devices. Quantization is then introduced to boost LLMs' on-device efficiency. Recent works show that 8-bit or lower weight quantization is feasible with minimal impact on end-to-end task performance, while the activation is still not quantized. On the other hand, mainstream commodity edge devices still struggle to execute these sub-8-bit quantized networks effectively. In this paper, we propose Agile-Quant, an activation-guided quantization framework for popular Large Language Models (LLMs), and implement an end-to-end accelerator on multiple edge devices for faster inference. Considering the hardware profiling and activation analysis, we first introduce a basic activation quantization strategy to balance the trade-off of task performance and real inference speed. Then we leverage the activation-aware token pruning technique to reduce the outliers and the adverse impact on attentivity. Ultimately, we utilize the SIMD-based 4-bit multiplier and our efficient TRIP matrix multiplication to implement the accelerator for LLMs on the edge. We apply our framework on different scales of LLMs including LLaMA, OPT, and BLOOM with 4-bit or 8-bit for the activation and 4-bit for the weight quantization. Experiments show that Agile-Quant achieves simultaneous quantization of model weights and activations while maintaining task performance comparable to existing weight-only quantization methods. Moreover, in the 8- and 4-bit scenario, Agile-Quant achieves an on-device speedup of up to 2.55x compared to its FP16 counterparts across multiple edge devices, marking a pioneering advancement in this domain.

{{</citation>}}


### (7/80) Leveraging Reinforcement Learning and Large Language Models for Code Optimization (Shukai Duan et al., 2023)

{{<citation>}}

Shukai Duan, Nikos Kanakaris, Xiongye Xiao, Heng Ping, Chenyu Zhou, Nesreen K. Ahmed, Guixiang Ma, Mihai Capota, Theodore L. Willke, Shahin Nazarian, Paul Bogdan. (2023)  
**Leveraging Reinforcement Learning and Large Language Models for Code Optimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-PL, cs-SE, cs.LG  
Keywords: Language Model, Reinforcement Learning, T5  
[Paper Link](http://arxiv.org/abs/2312.05657v1)  

---


**ABSTRACT**  
Code optimization is a daunting task that requires a significant level of expertise from experienced programmers. This level of expertise is not sufficient when compared to the rapid development of new hardware architectures. Towards advancing the whole code optimization process, recent approaches rely on machine learning and artificial intelligence techniques. This paper introduces a new framework to decrease the complexity of code optimization. The proposed framework builds on large language models (LLMs) and reinforcement learning (RL) and enables LLMs to receive feedback from their environment (i.e., unit tests) during the fine-tuning process. We compare our framework with existing state-of-the-art models and show that it is more efficient with respect to speed and computational usage, as a result of the decrement in training steps and its applicability to models with fewer parameters. Additionally, our framework reduces the possibility of logical and syntactical errors. Toward evaluating our approach, we run several experiments on the PIE dataset using a CodeT5 language model and RRHF, a new reinforcement learning algorithm. We adopt a variety of evaluation metrics with regards to optimization quality, and speedup. The evaluation results demonstrate that the proposed framework has similar results in comparison with existing models using shorter training times and smaller pre-trained models. In particular, we accomplish an increase of 5.6% and 2.2 over the baseline models concerning the %OP T and SP metrics.

{{</citation>}}


### (8/80) Triplet Edge Attention for Algorithmic Reasoning (Yeonjoon Jung et al., 2023)

{{<citation>}}

Yeonjoon Jung, Sungsoo Ahn. (2023)  
**Triplet Edge Attention for Algorithmic Reasoning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.05611v1)  

---


**ABSTRACT**  
This work investigates neural algorithmic reasoning to develop neural networks capable of learning from classical algorithms. The main challenge is to develop graph neural networks that are expressive enough to predict the given algorithm outputs while generalizing well to out-of-distribution data. In this work, we introduce a new graph neural network layer called Triplet Edge Attention (TEA), an edge-aware graph attention layer. Our algorithm works by precisely computing edge latent, aggregating multiple triplet messages using edge-based attention. We empirically validate our TEA layer in the CLRS benchmark and demonstrate a $5%$ improvement on average. In particular, we achieve a $30%$ improvement for the string algorithms compared to the state-of-the-art model.

{{</citation>}}


### (9/80) TCNCA: Temporal Convolution Network with Chunked Attention for Scalable Sequence Processing (Aleksandar Terzic et al., 2023)

{{<citation>}}

Aleksandar Terzic, Michael Hersche, Geethan Karunaratne, Luca Benini, Abu Sebastian, Abbas Rahimi. (2023)  
**TCNCA: Temporal Convolution Network with Chunked Attention for Scalable Sequence Processing**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.05605v1)  

---


**ABSTRACT**  
MEGA is a recent transformer-based architecture, which utilizes a linear recurrent operator whose parallel computation, based on the FFT, scales as $O(LlogL)$, with $L$ being the sequence length. We build upon their approach by replacing the linear recurrence with a special temporal convolutional network which permits larger receptive field size with shallower networks, and reduces the computational complexity to $O(L)$. The resulting model is called TCNCA, a Temporal Convolutional Network with Chunked Attention. We evaluate TCNCA on EnWik8 language modeling, long-range-arena (LRA) sequence classification, as well as a synthetic reasoning benchmark associative recall. On EnWik8, TCNCA outperforms MEGA, reaching a lower loss with $1.37\times$/$1.24\times$ faster forward/backward pass during training. The dilated convolutions used in TCNCA are consistently and significantly faster operations than the FFT-based parallelized recurrence in GPUs, making them a scalable candidate for handling very large sequence lengths: they are up to $7.07\times$/$2.86\times$ faster in the forward/backward pass for sequences up to 131k. Further on LRA, TCNCA achieves, on average, $1.28\times$ speed-up during inference with similar accuracy to what MEGA achieves. On associative recall, we find that even a simplified version of TCNCA, without excessive multiplicative and additive interactions, remains superior or competitive to MEGA on a range of sequence lengths and vocabulary sizes.

{{</citation>}}


### (10/80) Evolving Reservoirs for Meta Reinforcement Learning (Corentin Léger et al., 2023)

{{<citation>}}

Corentin Léger, Gautier Hamon, Eleni Nisioti, Xavier Hinaut, Clément Moulin-Frier. (2023)  
**Evolving Reservoirs for Meta Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NE, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.06695v1)  

---


**ABSTRACT**  
Animals often demonstrate a remarkable ability to adapt to their environments during their lifetime. They do so partly due to the evolution of morphological and neural structures. These structures capture features of environments shared between generations to bias and speed up lifetime learning. In this work, we propose a computational model for studying a mechanism that can enable such a process. We adopt a computational framework based on meta reinforcement learning as a model of the interplay between evolution and development. At the evolutionary scale, we evolve reservoirs, a family of recurrent neural networks that differ from conventional networks in that one optimizes not the weight values but hyperparameters of the architecture: the later control macro-level properties, such as memory and dynamics. At the developmental scale, we employ these evolved reservoirs to facilitate the learning of a behavioral policy through Reinforcement Learning (RL). Within an RL agent, a reservoir encodes the environment state before providing it to an action policy. We evaluate our approach on several 2D and 3D simulated environments. Our results show that the evolution of reservoirs can improve the learning of diverse challenging tasks. We study in particular three hypotheses: the use of an architecture combining reservoirs and reinforcement learning could enable (1) solving tasks with partial observability, (2) generating oscillatory dynamics that facilitate the learning of locomotion tasks, and (3) facilitating the generalization of learned behaviors to new tasks unknown during the evolution phase.

{{</citation>}}


### (11/80) Factorized Explainer for Graph Neural Networks (Rundong Huang et al., 2023)

{{<citation>}}

Rundong Huang, Farhad Shirani, Dongsheng Luo. (2023)  
**Factorized Explainer for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.05596v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have received increasing attention due to their ability to learn from graph-structured data. To open the black-box of these deep learning models, post-hoc instance-level explanation methods have been proposed to understand GNN predictions. These methods seek to discover substructures that explain the prediction behavior of a trained GNN. In this paper, we show analytically that for a large class of explanation tasks, conventional approaches, which are based on the principle of graph information bottleneck (GIB), admit trivial solutions that do not align with the notion of explainability. Instead, we argue that a modified GIB principle may be used to avoid the aforementioned trivial solutions. We further introduce a novel factorized explanation model with theoretical performance guarantees. The modified GIB is used to analyze the structural properties of the proposed factorized explainer. We conduct extensive experiments on both synthetic and real-world datasets to validate the effectiveness of our proposed factorized explainer over existing approaches.

{{</citation>}}


### (12/80) Deeper Understanding of Black-box Predictions via Generalized Influence Functions (Hyeonsu Lyu et al., 2023)

{{<citation>}}

Hyeonsu Lyu, Jonggyu Jang, Sehyun Ryu, Hyun Jong Yang. (2023)  
**Deeper Understanding of Black-box Predictions via Generalized Influence Functions**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.05586v1)  

---


**ABSTRACT**  
Influence functions (IFs) elucidate how learning data affects model behavior. However, growing non-convexity and the number of parameters in modern large-scale models lead to imprecise influence approximation and instability in computations. We highly suspect that the first-order approximation in large models causes such fragility, as IFs change all parameters including possibly nuisance parameters that are irrelevant to the examined data. Thus, we attempt to selectively analyze parameters associated with the data. However, simply computing influence from the chosen parameters can be misleading, as it fails to nullify the subliminal impact of unselected parameters. Our approach introduces generalized IFs, precisely estimating target parameters' influence while considering fixed parameters' effects. Unlike the classic IFs, we newly adopt a method to identify pertinent target parameters closely associated with the analyzed data. Furthermore, we tackle computational instability with a robust inverse-Hessian-vector product approximation. Remarkably, the proposed approximation algorithm guarantees convergence regardless of the network configurations. We evaluated our approach on ResNet-18 and VGG-11 for class removal and backdoor model recovery. Modifying just 10\% of the network yields results comparable to the network retrained from scratch. Aligned with our first guess, we also confirm that modifying an excessive number of parameters results in a decline in network utility. We believe our proposal can become a versatile tool for model analysis across various AI domains, appealing to both specialists and general readers. Codes are available at https://github.com/hslyu/GIF.

{{</citation>}}


### (13/80) Reinforcement Neighborhood Selection for Unsupervised Graph Anomaly Detection (Yuanchen Bei et al., 2023)

{{<citation>}}

Yuanchen Bei, Sheng Zhou, Qiaoyu Tan, Hao Xu, Hao Chen, Zhao Li, Jiajun Bu. (2023)  
**Reinforcement Neighborhood Selection for Unsupervised Graph Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.05526v1)  

---


**ABSTRACT**  
Unsupervised graph anomaly detection is crucial for various practical applications as it aims to identify anomalies in a graph that exhibit rare patterns deviating significantly from the majority of nodes. Recent advancements have utilized Graph Neural Networks (GNNs) to learn high-quality node representations for anomaly detection by aggregating information from neighborhoods. However, the presence of anomalies may render the observed neighborhood unreliable and result in misleading information aggregation for node representation learning. Selecting the proper neighborhood is critical for graph anomaly detection but also challenging due to the absence of anomaly-oriented guidance and the interdependence with representation learning. To address these issues, we utilize the advantages of reinforcement learning in adaptively learning in complex environments and propose a novel method that incorporates Reinforcement neighborhood selection for unsupervised graph ANomaly Detection (RAND). RAND begins by enriching the candidate neighbor pool of the given central node with multiple types of indirect neighbors. Next, RAND designs a tailored reinforcement anomaly evaluation module to assess the reliability and reward of considering the given neighbor. Finally, RAND selects the most reliable subset of neighbors based on these rewards and introduces an anomaly-aware aggregator to amplify messages from reliable neighbors while diminishing messages from unreliable ones. Extensive experiments on both three synthetic and two real-world datasets demonstrate that RAND outperforms the state-of-the-art methods.

{{</citation>}}


### (14/80) Isomorphic-Consistent Variational Graph Auto-Encoders for Multi-Level Graph Representation Learning (Hanxuan Yang et al., 2023)

{{<citation>}}

Hanxuan Yang, Qingchao Kong, Wenji Mao. (2023)  
**Isomorphic-Consistent Variational Graph Auto-Encoders for Multi-Level Graph Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.05519v1)  

---


**ABSTRACT**  
Graph representation learning is a fundamental research theme and can be generalized to benefit multiple downstream tasks from the node and link levels to the higher graph level. In practice, it is desirable to develop task-agnostic general graph representation learning methods that are typically trained in an unsupervised manner. Related research reveals that the power of graph representation learning methods depends on whether they can differentiate distinct graph structures as different embeddings and map isomorphic graphs to consistent embeddings (i.e., the isomorphic consistency of graph models). However, for task-agnostic general graph representation learning, existing unsupervised graph models, represented by the variational graph auto-encoders (VGAEs), can only keep the isomorphic consistency within the subgraphs of 1-hop neighborhoods and thus usually manifest inferior performance on the more difficult higher-level tasks. To overcome the limitations of existing unsupervised methods, in this paper, we propose the Isomorphic-Consistent VGAE (IsoC-VGAE) for multi-level task-agnostic graph representation learning. We first devise a decoding scheme to provide a theoretical guarantee of keeping the isomorphic consistency under the settings of unsupervised learning. We then propose the Inverse Graph Neural Network (Inv-GNN) decoder as its intuitive realization, which trains the model via reconstructing the GNN node embeddings with multi-hop neighborhood information, so as to maintain the high-order isomorphic consistency within the VGAE framework. We conduct extensive experiments on the representative graph learning tasks at different levels, including node classification, link prediction and graph classification, and the results verify that our proposed model generally outperforms both the state-of-the-art unsupervised methods and representative supervised methods.

{{</citation>}}


### (15/80) Stateful Large Language Model Serving with Pensieve (Lingfan Yu et al., 2023)

{{<citation>}}

Lingfan Yu, Jinyang Li. (2023)  
**Stateful Large Language Model Serving with Pensieve**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Attention, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.05516v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have recently experienced great success, as evident in the widespread popularity of ChatGPT. Existing LLM serving systems are stateless across requests. Consequently, when LLMs are used in the common setting of multi-turn conversations, a growing log of the conversation history must be processed alongside any request by the serving system at each turn, resulting in repeated history processing. In this paper, we design $Pensieve$, a system optimized for multi-turn conversation LLM serving. $Pensieve$ maintains the conversation state across requests by caching previously processed history to avoid duplicate processing. $Pensieve$'s multi-tier caching strategy can utilize both GPU and CPU memory to efficiently store and retrieve cached data. $Pensieve$ also generalizes the recent PagedAttention kernel to support attention between multiple input tokens with a GPU cache spread over non-contiguous memory. Our evaluation shows that $Pensieve$ is able to achieve 1.51-1.95x throughput compared to vLLM and reduce latency by 60-75%.

{{</citation>}}


### (16/80) Improving Adversarial Robust Fairness via Anti-Bias Soft Label Distillation (Shiji Zhao et al., 2023)

{{<citation>}}

Shiji Zhao, Xizhe Wang, Xingxing Wei. (2023)  
**Improving Adversarial Robust Fairness via Anti-Bias Soft Label Distillation**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-CY, cs-LG, cs.LG  
Keywords: Adversarial Training, Bias, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2312.05508v1)  

---


**ABSTRACT**  
Adversarial Training (AT) has been widely proved to be an effective method to improve the adversarial robustness against adversarial examples for Deep Neural Networks (DNNs). As a variant of AT, Adversarial Robustness Distillation (ARD) has demonstrated its superior performance in improving the robustness of small student models with the guidance of large teacher models. However, both AT and ARD encounter the robust fairness problem: these models exhibit strong robustness when facing part of classes (easy class), but weak robustness when facing others (hard class). In this paper, we give an in-depth analysis of the potential factors and argue that the smoothness degree of samples' soft labels for different classes (i.e., hard class or easy class) will affect the robust fairness of DNN models from both empirical observation and theoretical analysis. Based on the above finding, we propose an Anti-Bias Soft Label Distillation (ABSLD) method to mitigate the adversarial robust fairness problem within the framework of Knowledge Distillation (KD). Specifically, ABSLD adaptively reduces the student's error risk gap between different classes to achieve fairness by adjusting the class-wise smoothness degree of samples' soft labels during the training process, and the smoothness degree of soft labels is controlled by assigning different temperatures in KD to different classes. Extensive experiments demonstrate that ABSLD outperforms state-of-the-art AT, ARD, and robust fairness methods in terms of overall performance of robustness and fairness.

{{</citation>}}


### (17/80) Poisoning $\times$ Evasion: Symbiotic Adversarial Robustness for Graph Neural Networks (Ege Erdogan et al., 2023)

{{<citation>}}

Ege Erdogan, Simon Geisler, Stephan Günnemann. (2023)  
**Poisoning $\times$ Evasion: Symbiotic Adversarial Robustness for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.05502v1)  

---


**ABSTRACT**  
It is well-known that deep learning models are vulnerable to small input perturbations. Such perturbed instances are called adversarial examples. Adversarial examples are commonly crafted to fool a model either at training time (poisoning) or test time (evasion). In this work, we study the symbiosis of poisoning and evasion. We show that combining both threat models can substantially improve the devastating efficacy of adversarial attacks. Specifically, we study the robustness of Graph Neural Networks (GNNs) under structure perturbations and devise a memory-efficient adaptive end-to-end attack for the novel threat model using first-order optimization.

{{</citation>}}


### (18/80) Exploring Sparsity in Graph Transformers (Chuang Liu et al., 2023)

{{<citation>}}

Chuang Liu, Yibing Zhan, Xueqi Ma, Liang Ding, Dapeng Tao, Jia Wu, Wenbin Hu, Bo Du. (2023)  
**Exploring Sparsity in Graph Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.05479v1)  

---


**ABSTRACT**  
Graph Transformers (GTs) have achieved impressive results on various graph-related tasks. However, the huge computational cost of GTs hinders their deployment and application, especially in resource-constrained environments. Therefore, in this paper, we explore the feasibility of sparsifying GTs, a significant yet under-explored topic. We first discuss the redundancy of GTs based on the characteristics of existing GT models, and then propose a comprehensive \textbf{G}raph \textbf{T}ransformer \textbf{SP}arsification (GTSP) framework that helps to reduce the computational complexity of GTs from four dimensions: the input graph data, attention heads, model layers, and model weights. Specifically, GTSP designs differentiable masks for each individual compressible component, enabling effective end-to-end pruning. We examine our GTSP through extensive experiments on prominent GTs, including GraphTrans, Graphormer, and GraphGPS. The experimental results substantiate that GTSP effectively cuts computational costs, accompanied by only marginal decreases in accuracy or, in some cases, even improvements. For instance, GTSP yields a reduction of 30\% in Floating Point Operations while contributing to a 1.8\% increase in Area Under the Curve accuracy on OGBG-HIV dataset. Furthermore, we provide several insights on the characteristics of attention heads and the behavior of attention mechanisms, all of which have immense potential to inspire future research endeavors in this domain.

{{</citation>}}


### (19/80) On Task-Relevant Loss Functions in Meta-Reinforcement Learning and Online LQR (Jaeuk Shin et al., 2023)

{{<citation>}}

Jaeuk Shin, Giho Kim, Howon Lee, Joonho Han, Insoon Yang. (2023)  
**On Task-Relevant Loss Functions in Meta-Reinforcement Learning and Online LQR**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05465v1)  

---


**ABSTRACT**  
Designing a competent meta-reinforcement learning (meta-RL) algorithm in terms of data usage remains a central challenge to be tackled for its successful real-world applications. In this paper, we propose a sample-efficient meta-RL algorithm that learns a model of the system or environment at hand in a task-directed manner. As opposed to the standard model-based approaches to meta-RL, our method exploits the value information in order to rapidly capture the decision-critical part of the environment. The key component of our method is the loss function for learning the task inference module and the system model that systematically couples the model discrepancy and the value estimate, thereby facilitating the learning of the policy and the task inference module with a significantly smaller amount of data compared to the existing meta-RL algorithms. The idea is also extended to a non-meta-RL setting, namely an online linear quadratic regulator (LQR) problem, where our method can be simplified to reveal the essence of the strategy. The proposed method is evaluated in high-dimensional robotic control and online LQR problems, empirically verifying its effectiveness in extracting information indispensable for solving the tasks from observations in a sample efficient manner.

{{</citation>}}


### (20/80) Mitigating Nonlinear Algorithmic Bias in Binary Classification (Wendy Hui et al., 2023)

{{<citation>}}

Wendy Hui, Wai Kwong Lau. (2023)  
**Mitigating Nonlinear Algorithmic Bias in Binary Classification**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG, stat-AP  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2312.05429v1)  

---


**ABSTRACT**  
This paper proposes the use of causal modeling to detect and mitigate algorithmic bias that is nonlinear in the protected attribute. We provide a general overview of our approach. We use the German Credit data set, which is available for download from the UC Irvine Machine Learning Repository, to develop (1) a prediction model, which is treated as a black box, and (2) a causal model for bias mitigation. In this paper, we focus on age bias and the problem of binary classification. We show that the probability of getting correctly classified as "low risk" is lowest among young people. The probability increases with age nonlinearly. To incorporate the nonlinearity into the causal model, we introduce a higher order polynomial term. Based on the fitted causal model, the de-biased probability estimates are computed, showing improved fairness with little impact on overall classification accuracy. Causal modeling is intuitive and, hence, its use can enhance explicability and promotes trust among different stakeholders of AI.

{{</citation>}}


## cs.SE (3)



### (21/80) GPT-4 and Safety Case Generation: An Exploratory Analysis (Mithila Sivakumar et al., 2023)

{{<citation>}}

Mithila Sivakumar, Alvine Boaye Belle, Jinjun Shan, Kimya Khakzad Shahandashti. (2023)  
**GPT-4 and Safety Case Generation: An Exploratory Analysis**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.05696v1)  

---


**ABSTRACT**  
In the ever-evolving landscape of software engineering, the emergence of large language models (LLMs) and conversational interfaces, exemplified by ChatGPT, is nothing short of revolutionary. While their potential is undeniable across various domains, this paper sets out on a captivating expedition to investigate their uncharted territory, the exploration of generating safety cases. In this paper, our primary objective is to delve into the existing knowledge base of GPT-4, focusing specifically on its understanding of the Goal Structuring Notation (GSN), a well-established notation allowing to visually represent safety cases. Subsequently, we perform four distinct experiments with GPT-4. These experiments are designed to assess its capacity for generating safety cases within a defined system and application domain. To measure the performance of GPT-4 in this context, we compare the results it generates with ground-truth safety cases created for an X-ray system system and a Machine-Learning (ML)-enabled component for tire noise recognition (TNR) in a vehicle. This allowed us to gain valuable insights into the model's generative capabilities. Our findings indicate that GPT-4 demonstrates the capacity to produce safety arguments that are moderately accurate and reasonable. Furthermore, it exhibits the capability to generate safety cases that closely align with the semantic content of the reference safety cases used as ground-truths in our experiments.

{{</citation>}}


### (22/80) Redefining Developer Assistance: Through Large Language Models in Software Ecosystem (Somnath Banerjee et al., 2023)

{{<citation>}}

Somnath Banerjee, Avik Dutta, Sayan Layek, Amruit Sahoo, Sam Conrad Joyce, Rima Hazra. (2023)  
**Redefining Developer Assistance: Through Large Language Models in Software Ecosystem**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Language Model, NER, Named Entity Recognition, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2312.05626v1)  

---


**ABSTRACT**  
In this paper, we delve into the advancement of domain-specific Large Language Models (LLMs) with a focus on their application in software development. We introduce DevAssistLlama, a model developed through instruction tuning, to assist developers in processing software-related natural language queries. This model, a variant of instruction tuned LLM, is particularly adept at handling intricate technical documentation, enhancing developer capability in software specific tasks. The creation of DevAssistLlama involved constructing an extensive instruction dataset from various software systems, enabling effective handling of Named Entity Recognition (NER), Relation Extraction (RE), and Link Prediction (LP). Our results demonstrate DevAssistLlama's superior capabilities in these tasks, in comparison with other models including ChatGPT. This research not only highlights the potential of specialized LLMs in software development also the pioneer LLM for this domain.

{{</citation>}}


### (23/80) Chain-of-Thought in Neural Code Generation: From and For Lightweight Language Models (Guang Yang et al., 2023)

{{<citation>}}

Guang Yang, Yu Zhou, Xiang Chen, Xiangyu Zhang, Terry Yue Zhuo, Taolue Chen. (2023)  
**Chain-of-Thought in Neural Code Generation: From and For Lightweight Language Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: GLM, Language Model  
[Paper Link](http://arxiv.org/abs/2312.05562v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable potential in code generation. The integration of Chain of Thought (CoT) reasoning can further boost their performance. However, current CoT methods often require manual writing or LLMs with over 100 billion parameters to generate, impeding their applicability in resource-constrained scenarios. In this study, we investigate lightweight Language Models (lLMs), which are defined to have fewer than 10 billion parameters. Empirically, we find that most lLMs cannot generate high-quality CoTs when prompted by the few-shot method, but can take advantage of high-quality CoTs generated elsewhere to improve their performance in code generation. Based on these findings, we design a novel approach COTTON which can leverage lLMs to automatically generate CoTs for code generation. We synthesize new datasets and conduct extensive experiments on various benchmarks. The results show that the CoTs generated by COTTON outperform the baselines in terms of automated and human evaluation metrics. In particular, the CoTs generated by COTTON boost various lLMs to achieve higher performance gains than those generated by LLMs such as ChatGLM (130B), and are competitive with those generated by gpt-3.5-turbo (175B). Our study also showcases the potential of lLMs in software engineering applications.

{{</citation>}}


## cs.CV (11)



### (24/80) The Counterattack of CNNs in Self-Supervised Learning: Larger Kernel Size might be All You Need (Tianjin Huang et al., 2023)

{{<citation>}}

Tianjin Huang, Tianlong Chen, Zhangyang Wang, Shiwei Liu. (2023)  
**The Counterattack of CNNs in Self-Supervised Learning: Larger Kernel Size might be All You Need**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Self-Supervised, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.05695v2)  

---


**ABSTRACT**  
Vision Transformers have been rapidly uprising in computer vision thanks to their outstanding scaling trends, and gradually replacing convolutional neural networks (CNNs). Recent works on self-supervised learning (SSL) introduce siamese pre-training tasks, on which Transformer backbones continue to demonstrate ever stronger results than CNNs. People come to believe that Transformers or self-attention modules are inherently more suitable than CNNs in the context of SSL. However, it is noteworthy that most if not all prior arts of SSL with CNNs chose the standard ResNets as their backbones, whose architecture effectiveness is known to already lag behind advanced Vision Transformers. Therefore, it remains unclear whether the self-attention operation is crucial for the recent advances in SSL - or CNNs can deliver the same excellence with more advanced designs, too? Can we close the SSL performance gap between Transformers and CNNs? To answer these intriguing questions, we apply self-supervised pre-training to the recently proposed, stronger lager-kernel CNN architecture and conduct an apple-to-apple comparison with Transformers, in their SSL performance. Our results show that we are able to build pure CNN SSL architectures that perform on par with or better than the best SSL-trained Transformers, by just scaling up convolutional kernel sizes besides other small tweaks. Impressively, when transferring to the downstream tasks \texttt{MS COCO} detection and segmentation, our SSL pre-trained CNN model (trained in 100 epochs) achieves the same good performance as the 300-epoch pre-trained Transformer counterpart. We hope this work can help to better understand what is essential (or not) for self-supervised learning backbones.

{{</citation>}}


### (25/80) Performance of externally validated machine learning models based on histopathology images for the diagnosis, classification, prognosis, or treatment outcome prediction in female breast cancer: A systematic review (Ricardo Gonzalez et al., 2023)

{{<citation>}}

Ricardo Gonzalez, Peyman Nejat, Ashirbani Saha, Clinton J. V. Campbell, Andrew P. Norgan, Cynthia Lokker. (2023)  
**Performance of externally validated machine learning models based on histopathology images for the diagnosis, classification, prognosis, or treatment outcome prediction in female breast cancer: A systematic review**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2312.06697v1)  

---


**ABSTRACT**  
Numerous machine learning (ML) models have been developed for breast cancer using various types of data. Successful external validation (EV) of ML models is important evidence of their generalizability. The aim of this systematic review was to assess the performance of externally validated ML models based on histopathology images for diagnosis, classification, prognosis, or treatment outcome prediction in female breast cancer. A systematic search of MEDLINE, EMBASE, CINAHL, IEEE, MICCAI, and SPIE conferences was performed for studies published between January 2010 and February 2022. The Prediction Model Risk of Bias Assessment Tool (PROBAST) was employed, and the results were narratively described. Of the 2011 non-duplicated citations, 8 journal articles and 2 conference proceedings met inclusion criteria. Three studies externally validated ML models for diagnosis, 4 for classification, 2 for prognosis, and 1 for both classification and prognosis. Most studies used Convolutional Neural Networks and one used logistic regression algorithms. For diagnostic/classification models, the most common performance metrics reported in the EV were accuracy and area under the curve, which were greater than 87% and 90%, respectively, using pathologists' annotations as ground truth. The hazard ratios in the EV of prognostic ML models were between 1.7 (95% CI, 1.2-2.6) and 1.8 (95% CI, 1.3-2.7) to predict distant disease-free survival; 1.91 (95% CI, 1.11-3.29) for recurrence, and between 0.09 (95% CI, 0.01-0.70) and 0.65 (95% CI, 0.43-0.98) for overall survival, using clinical data as ground truth. Despite EV being an important step before the clinical application of a ML model, it hasn't been performed routinely. The large variability in the training/validation datasets, methods, performance metrics, and reported information limited the comparison of the models and the analysis of their results (...)

{{</citation>}}


### (26/80) EipFormer: Emphasizing Instance Positions in 3D Instance Segmentation (Mengnan Zhao et al., 2023)

{{<citation>}}

Mengnan Zhao, Lihe Zhang, Yuqiu Kong, Baocai Yin. (2023)  
**EipFormer: Emphasizing Instance Positions in 3D Instance Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.05602v1)  

---


**ABSTRACT**  
3D instance segmentation plays a crucial role in comprehending 3D scenes. Despite recent advancements in this field, existing approaches exhibit certain limitations. These methods often rely on fixed instance positions obtained from sampled representative points in vast 3D point clouds, using center prediction or farthest point sampling. However, these selected positions may deviate from actual instance centers, posing challenges in precisely grouping instances. Moreover, the common practice of grouping candidate instances from a single type of coordinates introduces difficulties in identifying neighboring instances or incorporating edge points. To tackle these issues, we present a novel Transformer-based architecture, EipFormer, which comprises progressive aggregation and dual position embedding. The progressive aggregation mechanism leverages instance positions to refine instance proposals. It enhances the initial instance positions through weighted farthest point sampling and further refines the instance positions and proposals using aggregation averaging and center matching. Additionally, dual position embedding superposes the original and centralized position embeddings, thereby enhancing the model performance in distinguishing adjacent instances. Extensive experiments on popular datasets demonstrate that EipFormer achieves superior or comparable performance compared to state-of-the-art approaches.

{{</citation>}}


### (27/80) CSL: Class-Agnostic Structure-Constrained Learning for Segmentation Including the Unseen (Hao Zhang et al., 2023)

{{<citation>}}

Hao Zhang, Fang Li, Lu Qi, Ming-Hsuan Yang, Narendra Ahuja. (2023)  
**CSL: Class-Agnostic Structure-Constrained Learning for Segmentation Including the Unseen**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.05538v1)  

---


**ABSTRACT**  
Addressing Out-Of-Distribution (OOD) Segmentation and Zero-Shot Semantic Segmentation (ZS3) is challenging, necessitating segmenting unseen classes. Existing strategies adapt the class-agnostic Mask2Former (CA-M2F) tailored to specific tasks. However, these methods cater to singular tasks, demand training from scratch, and we demonstrate certain deficiencies in CA-M2F, which affect performance. We propose the Class-Agnostic Structure-Constrained Learning (CSL), a plug-in framework that can integrate with existing methods, thereby embedding structural constraints and achieving performance gain, including the unseen, specifically OOD, ZS3, and domain adaptation (DA) tasks. There are two schemes for CSL to integrate with existing methods (1) by distilling knowledge from a base teacher network, enforcing constraints across training and inference phrases, or (2) by leveraging established models to obtain per-pixel distributions without retraining, appending constraints during the inference phase. We propose soft assignment and mask split methodologies that enhance OOD object segmentation. Empirical evaluations demonstrate CSL's prowess in boosting the performance of existing algorithms spanning OOD segmentation, ZS3, and DA segmentation, consistently transcending the state-of-art across all three tasks.

{{</citation>}}


### (28/80) Shapley Values-enabled Progressive Pseudo Bag Augmentation for Whole Slide Image Classification (Renao Yan et al., 2023)

{{<citation>}}

Renao Yan, Qiehe Sun, Cheng Jin, Yiqing Liu, Yonghong He, Tian Guan, Hao Chen. (2023)  
**Shapley Values-enabled Progressive Pseudo Bag Augmentation for Whole Slide Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Image Classification  
[Paper Link](http://arxiv.org/abs/2312.05490v1)  

---


**ABSTRACT**  
In computational pathology, whole slide image (WSI) classification presents a formidable challenge due to its gigapixel resolution and limited fine-grained annotations. Multiple instance learning (MIL) offers a weakly supervised solution, yet refining instance-level information from bag-level labels remains complex. While most of the conventional MIL methods use attention scores to estimate instance importance scores (IIS) which contribute to the prediction of the slide labels, these often lead to skewed attention distributions and inaccuracies in identifying crucial instances. To address these issues, we propose a new approach inspired by cooperative game theory: employing Shapley values to assess each instance's contribution, thereby improving IIS estimation. The computation of the Shapley value is then accelerated using attention, meanwhile retaining the enhanced instance identification and prioritization. We further introduce a framework for the progressive assignment of pseudo bags based on estimated IIS, encouraging more balanced attention distributions in MIL models. Our extensive experiments on CAMELYON-16, BRACS, and TCGA-LUNG datasets show our method's superiority over existing state-of-the-art approaches, offering enhanced interpretability and class-wise insights. We will release the code upon acceptance.

{{</citation>}}


### (29/80) BARET : Balanced Attention based Real image Editing driven by Target-text Inversion (Yuming Qiao et al., 2023)

{{<citation>}}

Yuming Qiao, Fanyi Wang, Jingwen Su, Yanhao Zhang, Yunjie Yu, Siyu Wu, Guo-Jun Qi. (2023)  
**BARET : Balanced Attention based Real image Editing driven by Target-text Inversion**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.05482v1)  

---


**ABSTRACT**  
Image editing approaches with diffusion models have been rapidly developed, yet their applicability are subject to requirements such as specific editing types (e.g., foreground or background object editing, style transfer), multiple conditions (e.g., mask, sketch, caption), and time consuming fine-tuning of diffusion models. For alleviating these limitations and realizing efficient real image editing, we propose a novel editing technique that only requires an input image and target text for various editing types including non-rigid edits without fine-tuning diffusion model. Our method contains three novelties:(I) Target-text Inversion Schedule (TTIS) is designed to fine-tune the input target text embedding to achieve fast image reconstruction without image caption and acceleration of convergence.(II) Progressive Transition Scheme applies progressive linear interpolation between target text embedding and its fine-tuned version to generate transition embedding for maintaining non-rigid editing capability.(III) Balanced Attention Module (BAM) balances the tradeoff between textual description and image semantics.By the means of combining self-attention map from reconstruction process and cross-attention map from transition process, the guidance of target text embeddings in diffusion process is optimized.In order to demonstrate editing capability, effectiveness and efficiency of the proposed BARET, we have conducted extensive qualitative and quantitative experiments. Moreover, results derived from user study and ablation study further prove the superiority over other methods.

{{</citation>}}


### (30/80) Exploring the Naturalness of AI-Generated Images (Zijian Chen et al., 2023)

{{<citation>}}

Zijian Chen, Wei Sun, Haoning Wu, Zicheng Zhang, Jun Jia, Xiongkuo Min, Guangtao Zhai, Wenjun Zhang. (2023)  
**Exploring the Naturalness of AI-Generated Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.05476v2)  

---


**ABSTRACT**  
The proliferation of Artificial Intelligence-Generated Images (AGIs) has greatly expanded the Image Naturalness Assessment (INA) problem. Different from early definitions that mainly focus on tone-mapped images with limited distortions (e.g., exposure, contrast, and color reproduction), INA on AI-generated images is especially challenging as it has more diverse contents and could be affected by factors from multiple perspectives, including low-level technical distortions and high-level rationality distortions. In this paper, we take the first step to benchmark and assess the visual naturalness of AI-generated images. First, we construct the AI-Generated Image Naturalness (AGIN) database by conducting a large-scale subjective study to collect human opinions on the overall naturalness as well as perceptions from technical and rationality perspectives. AGIN verifies that naturalness is universally and disparately affected by both technical and rationality distortions. Second, we propose the Joint Objective Image Naturalness evaluaTor (JOINT), to automatically learn the naturalness of AGIs that aligns human ratings. Specifically, JOINT imitates human reasoning in naturalness evaluation by jointly learning both technical and rationality perspectives. Experimental results show our proposed JOINT significantly surpasses baselines for providing more subjectively consistent results on naturalness assessment. Our database and code will be released in https://github.com/zijianchen98/AGIN.

{{</citation>}}


### (31/80) Identifying and Mitigating Model Failures through Few-shot CLIP-aided Diffusion Generation (Atoosa Chegini et al., 2023)

{{<citation>}}

Atoosa Chegini, Soheil Feizi. (2023)  
**Identifying and Mitigating Model Failures through Few-shot CLIP-aided Diffusion Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ChatGPT, GPT, ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2312.05464v1)  

---


**ABSTRACT**  
Deep learning models can encounter unexpected failures, especially when dealing with challenging sub-populations. One common reason for these failures is the occurrence of objects in backgrounds that are rarely seen during training. To gain a better understanding of these failure modes, human-interpretable descriptions are crucial for further analysis and improvement which is expensive. In this study, we propose an end-to-end framework that utilizes the capabilities of large language models (ChatGPT) and vision-language deep models (CLIP) to generate text descriptions of failure modes associated with spurious correlations (e.g. rarely seen backgrounds) without human-in-the-loop intervention. These descriptions can be used to generate synthetic data using generative models, such as diffusion models. The model can now use this generated data to learn from its weaknesses and enhance its performance on backgrounds that are uncommon for each class of data. Our approach serves as a broad solution, promising progress in comprehending model failure modes and strengthening deep learning models across a wide range of failure scenarios (e.g. bacckgrounds, colors) automatically in a few-shot manner. Our experiments have shown remarkable \textbf{improvements in accuracy ($\sim \textbf{21%}$)} on hard sub-populations (particularly for wrong background association) across $40$ different models, such as ResNets, EfficientNets, DenseNets, Vision Transformer (ViT), SwAVs, MoCos, DINOs, and CLIPs on various datasets such as ImageNet-1000, CIFAR-10, and CIFAR-100.

{{</citation>}}


### (32/80) TALDS-Net: Task-Aware Adaptive Local Descriptors Selection for Few-shot Image Classification (Qian Qiao et al., 2023)

{{<citation>}}

Qian Qiao, Yu Xie, Ziyin Zeng, Fanzhang Li. (2023)  
**TALDS-Net: Task-Aware Adaptive Local Descriptors Selection for Few-shot Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2312.05449v1)  

---


**ABSTRACT**  
Few-shot image classification aims to classify images from unseen novel classes with few samples. Recent works demonstrate that deep local descriptors exhibit enhanced representational capabilities compared to image-level features. However, most existing methods solely rely on either employing all local descriptors or directly utilizing partial descriptors, potentially resulting in the loss of crucial information. Moreover, these methods primarily emphasize the selection of query descriptors while overlooking support descriptors. In this paper, we propose a novel Task-Aware Adaptive Local Descriptors Selection Network (TALDS-Net), which exhibits the capacity for adaptive selection of task-aware support descriptors and query descriptors. Specifically, we compare the similarity of each local support descriptor with other local support descriptors to obtain the optimal support descriptor subset and then compare the query descriptors with the optimal support subset to obtain discriminative query descriptors. Extensive experiments demonstrate that our TALDS-Net outperforms state-of-the-art methods on both general and fine-grained datasets.

{{</citation>}}


### (33/80) From Static to Dynamic: Adapting Landmark-Aware Image Models for Facial Expression Recognition in Videos (Yin Chen et al., 2023)

{{<citation>}}

Yin Chen, Jia Li, Shiguang Shan, Meng Wang, Richang Hong. (2023)  
**From Static to Dynamic: Adapting Landmark-Aware Image Models for Facial Expression Recognition in Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.05447v1)  

---


**ABSTRACT**  
Dynamic facial expression recognition (DFER) in the wild is still hindered by data limitations, e.g., insufficient quantity and diversity of pose, occlusion and illumination, as well as the inherent ambiguity of facial expressions. In contrast, static facial expression recognition (SFER) currently shows much higher performance and can benefit from more abundant high-quality training data. Moreover, the appearance features and dynamic dependencies of DFER remain largely unexplored. To tackle these challenges, we introduce a novel Static-to-Dynamic model (S2D) that leverages existing SFER knowledge and dynamic information implicitly encoded in extracted facial landmark-aware features, thereby significantly improving DFER performance. Firstly, we build and train an image model for SFER, which incorporates a standard Vision Transformer (ViT) and Multi-View Complementary Prompters (MCPs) only. Then, we obtain our video model (i.e., S2D), for DFER, by inserting Temporal-Modeling Adapters (TMAs) into the image model. MCPs enhance facial expression features with landmark-aware features inferred by an off-the-shelf facial landmark detector. And the TMAs capture and model the relationships of dynamic changes in facial expressions, effectively extending the pre-trained image model for videos. Notably, MCPs and TMAs only increase a fraction of trainable parameters (less than +10\%) to the original image model. Moreover, we present a novel Emotion-Anchors (i.e., reference samples for each emotion category) based Self-Distillation Loss to reduce the detrimental influence of ambiguous emotion labels, further enhancing our S2D. Experiments conducted on popular SFER and DFER datasets show that we achieve the state of the art.

{{</citation>}}


### (34/80) Efficient Quantization Strategies for Latent Diffusion Models (Yuewei Yang et al., 2023)

{{<citation>}}

Yuewei Yang, Xiaoliang Dai, Jialiang Wang, Peizhao Zhang, Hongbo Zhang. (2023)  
**Efficient Quantization Strategies for Latent Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2312.05431v1)  

---


**ABSTRACT**  
Latent Diffusion Models (LDMs) capture the dynamic evolution of latent variables over time, blending patterns and multimodality in a generative system. Despite the proficiency of LDM in various applications, such as text-to-image generation, facilitated by robust text encoders and a variational autoencoder, the critical need to deploy large generative models on edge devices compels a search for more compact yet effective alternatives. Post Training Quantization (PTQ), a method to compress the operational size of deep learning models, encounters challenges when applied to LDM due to temporal and structural complexities. This study proposes a quantization strategy that efficiently quantize LDMs, leveraging Signal-to-Quantization-Noise Ratio (SQNR) as a pivotal metric for evaluation. By treating the quantization discrepancy as relative noise and identifying sensitive part(s) of a model, we propose an efficient quantization approach encompassing both global and local strategies. The global quantization process mitigates relative quantization noise by initiating higher-precision quantization on sensitive blocks, while local treatments address specific challenges in quantization-sensitive and time-sensitive modules. The outcomes of our experiments reveal that the implementation of both global and local treatments yields a highly efficient and effective Post Training Quantization (PTQ) of LDMs.

{{</citation>}}


## cs.DL (1)



### (35/80) NLLG Quarterly arXiv Report 09/23: What are the most influential current AI Papers? (Ran Zhang et al., 2023)

{{<citation>}}

Ran Zhang, Aida Kostikova, Christoph Leiter, Jonas Belouadi, Daniil Larionov, Yanran Chen, Vivian Fresen, Steffen Eger. (2023)  
**NLLG Quarterly arXiv Report 09/23: What are the most influential current AI Papers?**  

---
Primary Category: cs.DL  
Categories: cs-AI, cs-CL, cs-CV, cs-CY, cs-DL, cs-LG, cs.DL  
Keywords: AI, Computer Vision, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.05688v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) has witnessed rapid growth, especially in the subfields Natural Language Processing (NLP), Machine Learning (ML) and Computer Vision (CV). Keeping pace with this rapid progress poses a considerable challenge for researchers and professionals in the field. In this arXiv report, the second of its kind, which covers the period from January to September 2023, we aim to provide insights and analysis that help navigate these dynamic areas of AI. We accomplish this by 1) identifying the top-40 most cited papers from arXiv in the given period, comparing the current top-40 papers to the previous report, which covered the period January to June; 2) analyzing dataset characteristics and keyword popularity; 3) examining the global sectoral distribution of institutions to reveal differences in engagement across geographical areas. Our findings highlight the continued dominance of NLP: while only 16% of all submitted papers have NLP as primary category (more than 25% have CV and ML as primary category), 50% of the most cited papers have NLP as primary category, 90% of which target LLMs. Additionally, we show that i) the US dominates among both top-40 and top-9k papers, followed by China; ii) Europe clearly lags behind and is hardly represented in the top-40 most cited papers; iii) US industry is largely overrepresented in the top-40 most influential papers.

{{</citation>}}


## cs.AI (15)



### (36/80) Privacy Preserving Multi-Agent Reinforcement Learning in Supply Chains (Ananta Mukherjee et al., 2023)

{{<citation>}}

Ananta Mukherjee, Peeyush Kumar, Boling Yang, Nishanth Chandran, Divya Gupta. (2023)  
**Privacy Preserving Multi-Agent Reinforcement Learning in Supply Chains**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05686v1)  

---


**ABSTRACT**  
This paper addresses privacy concerns in multi-agent reinforcement learning (MARL), specifically within the context of supply chains where individual strategic data must remain confidential. Organizations within the supply chain are modeled as agents, each seeking to optimize their own objectives while interacting with others. As each organization's strategy is contingent on neighboring strategies, maintaining privacy of state and action-related information is crucial. To tackle this challenge, we propose a game-theoretic, privacy-preserving mechanism, utilizing a secure multi-party computation (MPC) framework in MARL settings. Our major contribution is the successful implementation of a secure MPC framework, SecFloat on EzPC, to solve this problem. However, simply implementing policy gradient methods such as MADDPG operations using SecFloat, while conceptually feasible, would be programmatically intractable. To overcome this hurdle, we devise a novel approach that breaks down the forward and backward pass of the neural network into elementary operations compatible with SecFloat , creating efficient and secure versions of the MADDPG algorithm. Furthermore, we present a learning mechanism that carries out floating point operations in a privacy-preserving manner, an important feature for successful learning in MARL framework. Experiments reveal that there is on average 68.19% less supply chain wastage in 2 PC compared to no data share, while also giving on average 42.27% better average cumulative revenue for each player. This work paves the way for practical, privacy-preserving MARL, promising significant improvements in secure computation within supply chain contexts and broadly.

{{</citation>}}


### (37/80) Transformer as Linear Expansion of Learngene (Shiyu Xia et al., 2023)

{{<citation>}}

Shiyu Xia, Miaosen Zhang, Xu Yang, Ruiming Chen, Haokun Chen, Xin Geng. (2023)  
**Transformer as Linear Expansion of Learngene**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.05614v1)  

---


**ABSTRACT**  
We propose expanding the shared Transformer module to produce and initialize Transformers with diverse depths, enabling adaptation to dynamic resource constraints. Drawing an analogy to genetic expansibility, we term such module as learngene. To identify the expansion mechanism, we delve into the relationship between the layer position and its corresponding weight value, and find that linear function appropriately approximates this relationship. Building on this insight, we present Transformer as Linear Expansion of learnGene (TLEG), a novel approach for flexibly producing and initializing Transformers of diverse depths. Specifically, to learn learngene, we firstly construct an auxiliary Transformer linearly expanded from learngene, after which we train it through employing soft distillation. Subsequently, we can produce and initialize Transformers of varying depths via linearly expanding the well-trained learngene, thereby supporting diverse downstream scenarios. Extensive experiments on ImageNet-1K classification demonstrate that TLEG achieves comparable or better performance compared to many individual models trained from scratch, while reducing around 2$\times$ training cost. When transferring one model to several downstream classification datasets, TLEG surpasses existing initialization methods by a large margin (e.g., +6.87% on iNat 2019 and +7.66% on CIFAR-100). Under the situation where we need to produce models of different scales adapting for different resource constraints, TLEG achieves comparable results while reducing around 19$\times$ parameters stored to initialize these models and around 5$\times$ training costs, in contrast to the pre-training and fine-tuning approach.

{{</citation>}}


### (38/80) Not All Data Matters: An End-to-End Adaptive Dataset Pruning Framework for Enhancing Model Performance and Efficiency (Suorong Yang et al., 2023)

{{<citation>}}

Suorong Yang, Hongchao Yang, Suhan Guo, Furao Shen, Jian Zhao. (2023)  
**Not All Data Matters: An End-to-End Adaptive Dataset Pruning Framework for Enhancing Model Performance and Efficiency**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2312.05599v1)  

---


**ABSTRACT**  
While deep neural networks have demonstrated remarkable performance across various tasks, they typically require massive training data. Due to the presence of redundancies and biases in real-world datasets, not all data in the training dataset contributes to the model performance. To address this issue, dataset pruning techniques have been introduced to enhance model performance and efficiency by eliminating redundant training samples and reducing computational and memory overhead. However, previous works most rely on manually crafted scalar scores, limiting their practical performance and scalability across diverse deep networks and datasets. In this paper, we propose AdaPruner, an end-to-end Adaptive DAtaset PRUNing framEwoRk. AdaPruner can perform effective dataset pruning without the need for explicitly defined metrics. Our framework jointly prunes training data and fine-tunes models with task-specific optimization objectives. AdaPruner leverages (1) An adaptive dataset pruning (ADP) module, which iteratively prunes redundant samples to an expected pruning ratio; and (2) A pruning performance controller (PPC) module, which optimizes the model performance for accurate pruning. Therefore, AdaPruner exhibits high scalability and compatibility across various datasets and deep networks, yielding improved dataset distribution and enhanced model performance. AdaPruner can still significantly enhance model performance even after pruning up to 10-30\% of the training data. Notably, these improvements are accompanied by substantial savings in memory and computation costs. Qualitative and quantitative experiments suggest that AdaPruner outperforms other state-of-the-art dataset pruning methods by a large margin.

{{</citation>}}


### (39/80) Artificial Intelligence in the automatic coding of interviews on Landscape Quality Objectives. Comparison and case study (Mario Burgui-Burgui, 2023)

{{<citation>}}

Mario Burgui-Burgui. (2023)  
**Artificial Intelligence in the automatic coding of interviews on Landscape Quality Objectives. Comparison and case study**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, ChatGPT, GPT, Google  
[Paper Link](http://arxiv.org/abs/2312.05597v1)  

---


**ABSTRACT**  
In this study, we conducted a comparative analysis of the automated coding provided by three Artificial Intelligence functionalities (At-las.ti, ChatGPT and Google Bard) in relation to the manual coding of 12 research interviews focused on Landscape Quality Objectives for a small island in the north of Cuba (Cayo Santa Mar\'ia). For this purpose, the following comparison criteria were established: Accuracy, Comprehensiveness, Thematic Coherence, Redundancy, Clarity, Detail and Regularity. The analysis showed the usefulness of AI for the intended purpose, albeit with numerous flaws and shortcomings. In summary, today the automatic coding of AIs can be considered useful as a guide towards a subsequent in-depth and meticulous analysis of the information by the researcher. However, as this is such a recently developed field, rapid evolution is expected to bring the necessary improvements to these tools.

{{</citation>}}


### (40/80) A Review of Hybrid and Ensemble in Deep Learning for Natural Language Processing (Jianguo Jia et al., 2023)

{{<citation>}}

Jianguo Jia, Wen Liang, Youzhi Liang. (2023)  
**A Review of Hybrid and Ensemble in Deep Learning for Natural Language Processing**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: BERT, Language Model, Machine Translation, NLP, Named Entity Recognition, Natural Language Processing, Question Answering, Sentiment Analysis, Speech Recognition, Summarization, Text Classification, Transformer  
[Paper Link](http://arxiv.org/abs/2312.05589v1)  

---


**ABSTRACT**  
This review presents a comprehensive exploration of hybrid and ensemble deep learning models within Natural Language Processing (NLP), shedding light on their transformative potential across diverse tasks such as Sentiment Analysis, Named Entity Recognition, Machine Translation, Question Answering, Text Classification, Generation, Speech Recognition, Summarization, and Language Modeling. The paper systematically introduces each task, delineates key architectures from Recurrent Neural Networks (RNNs) to Transformer-based models like BERT, and evaluates their performance, challenges, and computational demands. The adaptability of ensemble techniques is emphasized, highlighting their capacity to enhance various NLP applications. Challenges in implementation, including computational overhead, overfitting, and model interpretation complexities, are addressed alongside the trade-off between interpretability and performance. Serving as a concise yet invaluable guide, this review synthesizes insights into tasks, architectures, and challenges, offering a holistic perspective for researchers and practitioners aiming to advance language-driven applications through ensemble deep learning in NLP.

{{</citation>}}


### (41/80) Language-assisted Vision Model Debugger: A Sample-Free Approach to Finding Bugs (Chaoquan Jiang et al., 2023)

{{<citation>}}

Chaoquan Jiang, Jinqiang Wang, Rui Hu, Jitao Sang. (2023)  
**Language-assisted Vision Model Debugger: A Sample-Free Approach to Finding Bugs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.05588v1)  

---


**ABSTRACT**  
Vision models with high overall accuracy often exhibit systematic errors in specific scenarios, posing potential serious safety concerns. Diagnosing bugs of vision models is gaining increased attention, however traditional diagnostic approaches require annotation efforts (\eg rich metadata accompanying each samples of CelebA). To address this issue,We propose a language-assisted diagnostic method that uses texts instead of images to diagnose bugs in vision models based on multi-modal models (\eg CLIP). Our approach connects the embedding space of CLIP with the buggy vision model to be diagnosed; meanwhile, utilizing a shared classifier and the cross-modal transferability of embedding space from CLIP, the text-branch of CLIP become a proxy model to find bugs in the buggy model. The proxy model can classify texts paired with images. During the diagnosis, a Large Language Model (LLM) is employed to obtain task-relevant corpora, and this corpora is used to extract keywords. Descriptions constructed with templates containing these keywords serve as input text to probe errors in the proxy model. Finally, we validate the ability to diagnose existing visual models using language on the Waterbirds and CelebA datasets, we can identify bugs comprehensible to human experts, uncovering not only known bugs but also previously unknown ones.

{{</citation>}}


### (42/80) Dynamic Adjustment of Matching Radii under the Broadcasting Mode: A Novel Multitask Learning Strategy and Temporal Modeling Approach (Taijie Chen et al., 2023)

{{<citation>}}

Taijie Chen, Zijian Shen, Siyuan Feng, Linchuan Yang, Jintao Ke. (2023)  
**Dynamic Adjustment of Matching Radii under the Broadcasting Mode: A Novel Multitask Learning Strategy and Temporal Modeling Approach**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.05576v1)  

---


**ABSTRACT**  
As ride-hailing services have experienced significant growth, the majority of research has concentrated on the dispatching mode, where drivers must adhere to the platform's assigned routes. However, the broadcasting mode, in which drivers can freely choose their preferred orders from those broadcast by the platform, has received less attention. One important but challenging task in such a system is the determination of the optimal matching radius, which usually varies across space, time, and real-time supply/demand characteristics. This study develops a Transformer-Encoder-Based (TEB) model that predicts key system performance metrics for a range of matching radii, which enables the ride-hailing platform to select an optimal matching radius that maximizes overall system performance according to real-time supply and demand information. To simultaneously maximize multiple system performance metrics for matching radius determination, we devise a novel multi-task learning algorithm that enhances convergence speed of each task (corresponding to the optimization of one metric) and delivers more accurate overall predictions. We evaluate our methods in a simulation environment specifically designed for broadcasting-mode-based ride-hailing service. Our findings reveal that dynamically adjusting matching radii based on our proposed predict-then-optimize approach significantly improves system performance, e.g., increasing platform revenue by 7.55% and enhancing order fulfillment rate by 13% compared to benchmark algorithms.

{{</citation>}}


### (43/80) Frugal LMs Trained to Invoke Symbolic Solvers Achieve Parameter-Efficient Arithmetic Reasoning (Subhabrata Dutta et al., 2023)

{{<citation>}}

Subhabrata Dutta, Joykirat Singh, Ishan Pandey, Sunny Manchanda, Soumen Chakrabarti, Tanmoy Chakraborty. (2023)  
**Frugal LMs Trained to Invoke Symbolic Solvers Achieve Parameter-Efficient Arithmetic Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: GPT, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.05571v1)  

---


**ABSTRACT**  
Large Language Models (LLM) exhibit zero-shot mathematical reasoning capacity as a behavior emergent with scale, commonly manifesting as chain-of-thoughts (CoT) reasoning. However, multiple empirical findings suggest that this prowess is exclusive to LLMs with exorbitant sizes (beyond 50 billion parameters). Meanwhile, educational neuroscientists suggest that symbolic algebraic manipulation be introduced around the same time as arithmetic word problems to modularize language-to-formulation, symbolic manipulation of the formulation, and endgame arithmetic. In this paper, we start with the hypothesis that much smaller LMs, which are weak at multi-step reasoning, can achieve reasonable arithmetic reasoning if arithmetic word problems are posed as a formalize-then-solve task. In our architecture, which we call SYRELM, the LM serves the role of a translator to map natural language arithmetic questions into a formal language (FL) description. A symbolic solver then evaluates the FL expression to obtain the answer. A small frozen LM, equipped with an efficient low-rank adapter, is capable of generating FL expressions that incorporate natural language descriptions of the arithmetic problem (e.g., variable names and their purposes, formal expressions combining variables, etc.). We adopt policy-gradient reinforcement learning to train the adapted LM, informed by the non-differentiable symbolic solver. This marks a sharp departure from the recent development in tool-augmented LLMs, in which the external tools (e.g., calculator, Web search, etc.) are essentially detached from the learning phase of the LM. SYRELM shows massive improvements (e.g., +30.65 absolute point improvement in accuracy on the SVAMP dataset using GPT-J 6B model) over base LMs, while keeping our testbed easy to diagnose, interpret and within reach of most researchers.

{{</citation>}}


### (44/80) D3A-TS: Denoising-Driven Data Augmentation in Time Series (David Solis-Martin et al., 2023)

{{<citation>}}

David Solis-Martin, Juan Galan-Paez, Joaquin Borrego-Diaz. (2023)  
**D3A-TS: Denoising-Driven Data Augmentation in Time Series**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Augmentation, Computer Vision, Natural Language Processing, Time Series  
[Paper Link](http://arxiv.org/abs/2312.05550v1)  

---


**ABSTRACT**  
It has been demonstrated that the amount of data is crucial in data-driven machine learning methods. Data is always valuable, but in some tasks, it is almost like gold. This occurs in engineering areas where data is scarce or very expensive to obtain, such as predictive maintenance, where faults are rare. In this context, a mechanism to generate synthetic data can be very useful. While in fields such as Computer Vision or Natural Language Processing synthetic data generation has been extensively explored with promising results, in other domains such as time series it has received less attention. This work specifically focuses on studying and analyzing the use of different techniques for data augmentation in time series for classification and regression problems. The proposed approach involves the use of diffusion probabilistic models, which have recently achieved successful results in the field of Image Processing, for data augmentation in time series. Additionally, the use of meta-attributes to condition the data augmentation process is investigated. The results highlight the high utility of this methodology in creating synthetic data to train classification and regression models. To assess the results, six different datasets from diverse domains were employed, showcasing versatility in terms of input size and output types. Finally, an extensive ablation study is conducted to further support the obtained outcomes.

{{</citation>}}


### (45/80) Causal-CoG: A Causal-Effect Look at Context Generation for Boosting Multi-modal Language Models (Shitian Zhao et al., 2023)

{{<citation>}}

Shitian Zhao, Zhuowan Li, Yadong Lu, Alan Yuille, Yan Wang. (2023)  
**Causal-CoG: A Causal-Effect Look at Context Generation for Boosting Multi-modal Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2312.06685v1)  

---


**ABSTRACT**  
While Multi-modal Language Models (MLMs) demonstrate impressive multimodal ability, they still struggle on providing factual and precise responses for tasks like visual question answering (VQA). In this paper, we address this challenge from the perspective of contextual information. We propose Causal Context Generation, Causal-CoG, which is a prompting strategy that engages contextual information to enhance precise VQA during inference. Specifically, we prompt MLMs to generate contexts, i.e, text description of an image, and engage the generated contexts for question answering. Moreover, we investigate the advantage of contexts on VQA from a causality perspective, introducing causality filtering to select samples for which contextual information is helpful. To show the effectiveness of Causal-CoG, we run extensive experiments on 10 multimodal benchmarks and show consistent improvements, e.g., +6.30% on POPE, +13.69% on Vizwiz and +6.43% on VQAv2 compared to direct decoding, surpassing existing methods. We hope Casual-CoG inspires explorations of context knowledge in multimodal models, and serves as a plug-and-play strategy for MLM decoding.

{{</citation>}}


### (46/80) Enhanced E-Commerce Attribute Extraction: Innovating with Decorative Relation Correction and LLAMA 2.0-Based Annotation (Jianghong Zhou et al., 2023)

{{<citation>}}

Jianghong Zhou, Weizhi Du, Md Omar Faruk Rokon, Zhaodong Wang, Jiaxuan Xu, Isha Shah, Kuang-chih Lee, Musen Wen. (2023)  
**Enhanced E-Commerce Attribute Extraction: Innovating with Decorative Relation Correction and LLAMA 2.0-Based Annotation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: BERT, Language Model, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2312.06684v1)  

---


**ABSTRACT**  
The rapid proliferation of e-commerce platforms accentuates the need for advanced search and retrieval systems to foster a superior user experience. Central to this endeavor is the precise extraction of product attributes from customer queries, enabling refined search, comparison, and other crucial e-commerce functionalities. Unlike traditional Named Entity Recognition (NER) tasks, e-commerce queries present a unique challenge owing to the intrinsic decorative relationship between product types and attributes. In this study, we propose a pioneering framework that integrates BERT for classification, a Conditional Random Fields (CRFs) layer for attribute value extraction, and Large Language Models (LLMs) for data annotation, significantly advancing attribute recognition from customer inquiries. Our approach capitalizes on the robust representation learning of BERT, synergized with the sequence decoding prowess of CRFs, to adeptly identify and extract attribute values. We introduce a novel decorative relation correction mechanism to further refine the extraction process based on the nuanced relationships between product types and attributes inherent in e-commerce data. Employing LLMs, we annotate additional data to expand the model's grasp and coverage of diverse attributes. Our methodology is rigorously validated on various datasets, including Walmart, BestBuy's e-commerce NER dataset, and the CoNLL dataset, demonstrating substantial improvements in attribute recognition performance. Particularly, the model showcased promising results during a two-month deployment in Walmart's Sponsor Product Search, underscoring its practical utility and effectiveness.

{{</citation>}}


### (47/80) Can Large Language Models Serve as Rational Players in Game Theory? A Systematic Analysis (Caoyun Fan et al., 2023)

{{<citation>}}

Caoyun Fan, Jindou Chen, Yaohui Jin, Hao He. (2023)  
**Can Large Language Models Serve as Rational Players in Game Theory? A Systematic Analysis**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-GT, cs.AI  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.05488v2)  

---


**ABSTRACT**  
Game theory, as an analytical tool, is frequently utilized to analyze human behavior in social science research. With the high alignment between the behavior of Large Language Models (LLMs) and humans, a promising research direction is to employ LLMs as substitutes for humans in game experiments, enabling social science research. However, despite numerous empirical researches on the combination of LLMs and game theory, the capability boundaries of LLMs in game theory remain unclear. In this research, we endeavor to systematically analyze LLMs in the context of game theory. Specifically, rationality, as the fundamental principle of game theory, serves as the metric for evaluating players' behavior -- building a clear desire, refining belief about uncertainty, and taking optimal actions. Accordingly, we select three classical games (dictator game, Rock-Paper-Scissors, and ring-network game) to analyze to what extent LLMs can achieve rationality in these three aspects. The experimental results indicate that even the current state-of-the-art LLM (GPT-4) exhibits substantial disparities compared to humans in game theory. For instance, LLMs struggle to build desires based on uncommon preferences, fail to refine belief from many simple patterns, and may overlook or modify refined belief when taking actions. Therefore, we consider that introducing LLMs into game experiments in the field of social science should be approached with greater caution.

{{</citation>}}


### (48/80) Learning to Denoise Unreliable Interactions for Link Prediction on Biomedical Knowledge Graph (Tengfei Ma et al., 2023)

{{<citation>}}

Tengfei Ma, Yujie Chen, Wen Tao, Dashun Zheng, Xuan Lin, Patrick Cheong-lao Pang, Yiping Liu, Yijun Wang, Bosheng Song, Xiangxiang Zeng. (2023)  
**Learning to Denoise Unreliable Interactions for Link Prediction on Biomedical Knowledge Graph**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.06682v1)  

---


**ABSTRACT**  
Link prediction in biomedical knowledge graphs (KGs) aims at predicting unknown interactions between entities, including drug-target interaction (DTI) and drug-drug interaction (DDI), which is critical for drug discovery and therapeutics. Previous methods prefer to utilize the rich semantic relations and topological structure of the KG to predict missing links, yielding promising outcomes. However, all these works only focus on improving the predictive performance without considering the inevitable noise and unreliable interactions existing in the KGs, which limits the development of KG-based computational methods. To address these limitations, we propose a Denoised Link Prediction framework, called DenoisedLP. DenoisedLP obtains reliable interactions based on the local subgraph by denoising noisy links in a learnable way, providing a universal module for mining underlying task-relevant relations. To collaborate with the smoothed semantic information, DenoisedLP introduces the semantic subgraph by blurring conflict relations around the predicted link. By maximizing the mutual information between the reliable structure and smoothed semantic relations, DenoisedLP emphasizes the informative interactions for predicting relation-specific links. Experimental results on real-world datasets demonstrate that DenoisedLP outperforms state-of-the-art methods on DTI and DDI prediction tasks, and verify the effectiveness and robustness of denoising unreliable interactions on the contaminated KGs.

{{</citation>}}


### (49/80) Image and Data Mining in Reticular Chemistry Using GPT-4V (Zhiling Zheng et al., 2023)

{{<citation>}}

Zhiling Zheng, Zhiguo He, Omar Khattab, Nakul Rampal, Matei A. Zaharia, Christian Borgs, Jennifer T. Chayes, Omar M. Yaghi. (2023)  
**Image and Data Mining in Reticular Chemistry Using GPT-4V**  

---
Primary Category: cs.AI  
Categories: cond-mat-mtrl-sci, cs-AI, cs-CV, cs-IR, cs.AI  
Keywords: AI, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.05468v1)  

---


**ABSTRACT**  
The integration of artificial intelligence into scientific research has reached a new pinnacle with GPT-4V, a large language model featuring enhanced vision capabilities, accessible through ChatGPT or an API. This study demonstrates the remarkable ability of GPT-4V to navigate and obtain complex data for metal-organic frameworks, especially from graphical sources. Our approach involved an automated process of converting 346 scholarly articles into 6240 images, which represents a benchmark dataset in this task, followed by deploying GPT-4V to categorize and analyze these images using natural language prompts. This methodology enabled GPT-4V to accurately identify and interpret key plots integral to MOF characterization, such as nitrogen isotherms, PXRD patterns, and TGA curves, among others, with accuracy and recall above 93%. The model's proficiency in extracting critical information from these plots not only underscores its capability in data mining but also highlights its potential in aiding the creation of comprehensive digital databases for reticular chemistry. In addition, the extracted nitrogen isotherm data from the selected literature allowed for a comparison between theoretical and experimental porosity values for over 200 compounds, highlighting certain discrepancies and underscoring the importance of integrating computational and experimental data. This work highlights the potential of AI in accelerating scientific discovery and innovation, bridging the gap between computational tools and experimental research, and paving the way for more efficient, inclusive, and comprehensive scientific inquiry.

{{</citation>}}


### (50/80) Stochastic Directly-Follows Process Discovery Using Grammatical Inference (Hanan Alkhammash et al., 2023)

{{<citation>}}

Hanan Alkhammash, Artem Polyvyanyy, Alistair Moffat. (2023)  
**Stochastic Directly-Follows Process Discovery Using Grammatical Inference**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-FL, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2312.05433v1)  

---


**ABSTRACT**  
Starting with a collection of traces generated by process executions, process discovery is the task of constructing a simple model that describes the process, where simplicity is often measured in terms of model size. The challenge of process discovery is that the process of interest is unknown, and that while the input traces constitute positive examples of process executions, no negative examples are available. Many commercial tools discover Directly-Follows Graphs, in which nodes represent the observable actions of the process, and directed arcs indicate execution order possibilities over the actions. We propose a new approach for discovering sound Directly-Follows Graphs that is grounded in grammatical inference over the input traces. To promote the discovery of small graphs that also describe the process accurately we design and evaluate a genetic algorithm that supports the convergence of the inference parameters to the areas that lead to the discovery of interesting models. Experiments over real-world datasets confirm that our new approach can construct smaller models that represent the input traces and their frequencies more accurately than the state-of-the-art technique. Reasoning over the frequencies of encoded traces also becomes possible, due to the stochastic semantics of the action graphs we propose, which, for the first time, are interpreted as models that describe the stochastic languages of action traces.

{{</citation>}}


## cs.CY (2)



### (51/80) Using Think-Aloud Data to Understand Relations between Self-Regulation Cycle Characteristics and Student Performance in Intelligent Tutoring Systems (Conrad Borchers et al., 2023)

{{<citation>}}

Conrad Borchers, Jiayi Zhang, Ryan S. Baker, Vincent Aleven. (2023)  
**Using Think-Aloud Data to Understand Relations between Self-Regulation Cycle Characteristics and Student Performance in Intelligent Tutoring Systems**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.05675v1)  

---


**ABSTRACT**  
Numerous studies demonstrate the importance of self-regulation during learning by problem-solving. Recent work in learning analytics has largely examined students' use of SRL concerning overall learning gains. Limited research has related SRL to in-the-moment performance differences among learners. The present study investigates SRL behaviors in relationship to learners' moment-by-moment performance while working with intelligent tutoring systems for stoichiometry chemistry. We demonstrate the feasibility of labeling SRL behaviors based on AI-generated think-aloud transcripts, identifying the presence or absence of four SRL categories (processing information, planning, enacting, and realizing errors) in each utterance. Using the SRL codes, we conducted regression analyses to examine how the use of SRL in terms of presence, frequency, cyclical characteristics, and recency relate to student performance on subsequent steps in multi-step problems. A model considering students' SRL cycle characteristics outperformed a model only using in-the-moment SRL assessment. In line with theoretical predictions, students' actions during earlier, process-heavy stages of SRL cycles exhibited lower moment-by-moment correctness during problem-solving than later SRL cycle stages. We discuss system re-design opportunities to add SRL support during stages of processing and paths forward for using machine learning to speed research depending on the assessment of SRL based on transcription of think-aloud data.

{{</citation>}}


### (52/80) Enhancing Situational Awareness in Surveillance: Leveraging Data Visualization Techniques for Machine Learning-based Video Analytics Outcomes (Babak Rahimi Ardabili et al., 2023)

{{<citation>}}

Babak Rahimi Ardabili, Shanle Yao, Armin Danesh Pazho, Lauren Bourque, Hamed Tabkhi. (2023)  
**Enhancing Situational Awareness in Surveillance: Leveraging Data Visualization Techniques for Machine Learning-based Video Analytics Outcomes**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.05629v1)  

---


**ABSTRACT**  
The pervasive deployment of surveillance cameras produces a massive volume of data, requiring nuanced interpretation. This study thoroughly examines data representation and visualization techniques tailored for AI surveillance data within current infrastructures. It delves into essential data metrics, methods for situational awareness, and various visualization techniques, highlighting their potential to enhance safety and guide urban development. This study is built upon real-world research conducted in a community college environment, utilizing eight cameras over eight days. This study presents tools like the Occupancy Indicator, Statistical Anomaly Detection, Bird's Eye View, and Heatmaps to elucidate pedestrian behaviors, surveillance, and public safety. Given the intricate data from smart video surveillance, such as bounding boxes and segmented images, we aim to convert these computer vision results into intuitive visualizations and actionable insights for stakeholders, including law enforcement, urban planners, and social scientists. The results emphasize the crucial impact of visualizing AI surveillance data on emergency handling, public health protocols, crowd control, resource distribution, predictive modeling, city planning, and informed decision-making.

{{</citation>}}


## eess.SY (1)



### (53/80) Position control of an acoustic cavitation bubble by reinforcement learning (Kálmán Klapcsik et al., 2023)

{{<citation>}}

Kálmán Klapcsik, Bálint Gyires-Tóth, Juan Manuel Rosselló, Ferenc Hegedűs. (2023)  
**Position control of an acoustic cavitation bubble by reinforcement learning**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05674v1)  

---


**ABSTRACT**  
A control technique is developed via Reinforcement Learning that allows arbitrary controlling of the position of an acoustic cavitation bubble in a dual-frequency standing acoustic wave field. The agent must choose the optimal pressure amplitude values to manipulate the bubble position in the range of $x/\lambda_0\in[0.05, 0.25]$. To train the agent an actor-critic off-policy algorithm (Deep Deterministic Policy Gradient) was used that supports continuous action space, which allows setting the pressure amplitude values continuously within $0$ and $1\, \mathrm{bar}$. A shaped reward function is formulated that minimizes the distance between the bubble and the target position and implicitly encourages the agent to perform the position control within the shortest amount of time. In some cases, the optimal control can be 7 times faster than the solution expected from the linear theory.

{{</citation>}}


## cs.CL (15)



### (54/80) Hate Speech and Offensive Content Detection in Indo-Aryan Languages: A Battle of LSTM and Transformers (Nikhil Narayan et al., 2023)

{{<citation>}}

Nikhil Narayan, Mrutyunjay Biswal, Pramod Goyal, Abhranta Panigrahi. (2023)  
**Hate Speech and Offensive Content Detection in Indo-Aryan Languages: A Battle of LSTM and Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LSTM, Multilingual, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.05671v1)  

---


**ABSTRACT**  
Social media platforms serve as accessible outlets for individuals to express their thoughts and experiences, resulting in an influx of user-generated data spanning all age groups. While these platforms enable free expression, they also present significant challenges, including the proliferation of hate speech and offensive content. Such objectionable language disrupts objective discourse and can lead to radicalization of debates, ultimately threatening democratic values. Consequently, organizations have taken steps to monitor and curb abusive behavior, necessitating automated methods for identifying suspicious posts. This paper contributes to Hate Speech and Offensive Content Identification in English and Indo-Aryan Languages (HASOC) 2023 shared tasks track. We, team Z-AGI Labs, conduct a comprehensive comparative analysis of hate speech classification across five distinct languages: Bengali, Assamese, Bodo, Sinhala, and Gujarati. Our study encompasses a wide range of pre-trained models, including Bert variants, XLM-R, and LSTM models, to assess their performance in identifying hate speech across these languages. Results reveal intriguing variations in model performance. Notably, Bert Base Multilingual Cased emerges as a strong performer across languages, achieving an F1 score of 0.67027 for Bengali and 0.70525 for Assamese. At the same time, it significantly outperforms other models with an impressive F1 score of 0.83009 for Bodo. In Sinhala, XLM-R stands out with an F1 score of 0.83493, whereas for Gujarati, a custom LSTM-based model outshined with an F1 score of 0.76601. This study offers valuable insights into the suitability of various pre-trained models for hate speech detection in multilingual settings. By considering the nuances of each, our research contributes to an informed model selection for building robust hate speech detection systems.

{{</citation>}}


### (55/80) Understanding the Effect of Model Compression on Social Bias in Large Language Models (Gustavo Gonçalves et al., 2023)

{{<citation>}}

Gustavo Gonçalves, Emma Strubell. (2023)  
**Understanding the Effect of Model Compression on Social Bias in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2312.05662v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) trained with self-supervision on vast corpora of web text fit to the social biases of that text. Without intervention, these social biases persist in the model's predictions in downstream tasks, leading to representational harm. Many strategies have been proposed to mitigate the effects of inappropriate social biases learned during pretraining. Simultaneously, methods for model compression have become increasingly popular to reduce the computational burden of LLMs. Despite the popularity and need for both approaches, little work has been done to explore the interplay between these two. We perform a carefully controlled study of the impact of model compression via quantization and knowledge distillation on measures of social bias in LLMs. Longer pretraining and larger models led to higher social bias, and quantization showed a regularizer effect with its best trade-off around 20% of the original pretraining time.

{{</citation>}}


### (56/80) PILLOW: Enhancing Efficient Instruction Fine-tuning via Prompt Matching (Zhenting Qi et al., 2023)

{{<citation>}}

Zhenting Qi, Xiaoyu Tan, Shaojie Shi, Chao Qu, Yinghui Xu, Yuan Qi. (2023)  
**PILLOW: Enhancing Efficient Instruction Fine-tuning via Prompt Matching**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05621v1)  

---


**ABSTRACT**  
Instruction fine-tuning has conventionally been employed to adapt Large Language Models (LLMs) to a variety of tasks. Nonetheless, this technique often necessitates substantial computational resources, making it impractical for deployment by individuals or small-scale entities. Recently, Low-Rank Adaptation (LoRA) has become a promising alternative, offering high capabilities on par with full tuning with reduced resource overhead. However, attaining satisfactory performance through the fine-tuning of LoRA is a non-trivial challenge. In this paper, we propose PILLOW, which aims to improve LoRA's performance by a discrimination-based prompting method, leveraging LLMs' In-Context Learning ability. PILLOW incorporates a matching network that selects prompts from a user-defined prompt pool, concatenates the selected prompts with the user instruction as input, and performs inference using the LoRA-fine-tuned LLMs. Trained with Reinforcement Learning, PILLOW exhibits commensurate performance on various evaluation metrics compared with typical instruction fine-tuning methods, utilizing only consumer-grade GPU resources and exhibiting a large reduction in computational costs.

{{</citation>}}


### (57/80) Sim-GPT: Text Similarity via GPT Annotated Data (Shuhe Wang et al., 2023)

{{<citation>}}

Shuhe Wang, Beiming Cao, Shengyu Zhang, Xiaoya Li, Jiwei Li, Fei Wu, Guoyin Wang, Eduard Hovy. (2023)  
**Sim-GPT: Text Similarity via GPT Annotated Data**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, GPT, GPT-4, NLI, Textual Similarity  
[Paper Link](http://arxiv.org/abs/2312.05603v2)  

---


**ABSTRACT**  
Due to the lack of a large collection of high-quality labeled sentence pairs with textual similarity scores, existing approaches for Semantic Textual Similarity (STS) mostly rely on unsupervised techniques or training signals that are only partially correlated with textual similarity, e.g., NLI-based datasets. To tackle this issue, in this paper, we propose the strategy of measuring text similarity via GPT annotated data (Sim-GPT for short). The core idea of Sim-GPT is to generate data with STS labels using GPT-4, based on which an STS model is trained. Sim-GPT framework utilizes LLMs to provide a substantial amount of reliable annotated data filling the gap of the lack of training signals for STS. Sim-GPT is trained on a one-time generated dataset using BERT or RoBERTa as the backbone, which offers long-term savings in cost and speed compared to repeatedly invoking LLMs for each sentence pair. Trained on the examples from GPT-4 (371K), Sim-GPT yields SOTA performances on the widely-used seven STS benchmarks: +0.99 over supervised-SimCSE, and +0.42 over the current SOTA PromCSE model. To encourage further advancements of the field, we release both models and the 371K annotated examples from GPT-4. Code, models and annotated data are available at: https://github.com/ShuheWang1998/Sim-GPT.

{{</citation>}}


### (58/80) Enhancing Medical Specialty Assignment to Patients using NLP Techniques (Chris Solomou, 2023)

{{<citation>}}

Chris Solomou. (2023)  
**Enhancing Medical Specialty Assignment to Patients using NLP Techniques**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.05585v1)  

---


**ABSTRACT**  
The introduction of Large Language Models (LLMs), and the vast volume of publicly available medical data, amplified the application of NLP to the medical domain. However, LLMs are pretrained on data that are not explicitly relevant to the domain that are applied to and are often biased towards the original data they were pretrained upon. Even when pretrained on domainspecific data, these models typically require time-consuming fine-tuning to achieve good performance for a specific task. To address these limitations, we propose an alternative approach that achieves superior performance while being computationally efficient. Specifically, we utilize keywords to train a deep learning architecture that outperforms a language model pretrained on a large corpus of text. Our proposal does not require pretraining nor fine-tuning and can be applied directly to a specific setting for performing multi-label classification. Our objective is to automatically assign a new patient to the specialty of the medical professional they require, using a dataset that contains medical transcriptions and relevant keywords. To this end, we fine-tune the PubMedBERT model on this dataset, which serves as the baseline for our experiments. We then twice train/fine-tune a DNN and the RoBERTa language model, using both the keywords and the full transcriptions as input. We compare the performance of these approaches using relevant metrics. Our results demonstrate that utilizing keywords for text classification significantly improves classification performance, for both a basic DL architecture and a large language model. Our approach represents a promising and efficient alternative to traditional methods for finetuning language models on domain-specific data and has potential applications in various medical domains

{{</citation>}}


### (59/80) Augmenty: A Python Library for Structured Text Augmentation (Kenneth Enevoldsen, 2023)

{{<citation>}}

Kenneth Enevoldsen. (2023)  
**Augmenty: A Python Library for Structured Text Augmentation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, NER  
[Paper Link](http://arxiv.org/abs/2312.05520v1)  

---


**ABSTRACT**  
Augmnety is a Python library for structured text augmentation. It is built on top of spaCy and allows for augmentation of both the text and its annotations. Augmenty provides a wide range of augmenters which can be combined in a flexible manner to create complex augmentation pipelines. It also includes a set of primitives that can be used to create custom augmenters such as word replacement augmenters. This functionality allows for augmentations within a range of applications such as named entity recognition (NER), part-of-speech tagging, and dependency parsing.

{{</citation>}}


### (60/80) Aligner: One Global Token is Worth Millions of Parameters When Aligning Large Language Models (Zhou Ziheng et al., 2023)

{{<citation>}}

Zhou Ziheng, Yingnian Wu, Song-Chun Zhu, Demetri Terzopoulos. (2023)  
**Aligner: One Global Token is Worth Millions of Parameters When Aligning Large Language Models**  

---
Primary Category: cs.CL  
Categories: I-2; I-2-6; I-2-7, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.05503v1)  

---


**ABSTRACT**  
We introduce Aligner, a novel Parameter-Efficient Fine-Tuning (PEFT) method for aligning multi-billion-parameter-sized Large Language Models (LLMs). Aligner employs a unique design that constructs a globally shared set of tunable tokens that modify the attention of every layer. Remarkably with this method, even when using one token accounting for a mere 5,000 parameters, Aligner can still perform comparably well to state-of-the-art LLM adaptation methods like LoRA that require millions of parameters. This capacity is substantiated in both instruction following and value alignment tasks. Besides the multiple order-of-magnitude improvement in parameter efficiency, the insight Aligner provides into the internal mechanisms of LLMs is also valuable. The architectural features and efficacy of our method, in addition to our experiments demonstrate that an LLM separates its internal handling of "form" and "knowledge" in a somewhat orthogonal manner. This finding promises to motivate new research into LLM mechanism understanding and value alignment.

{{</citation>}}


### (61/80) History Matters: Temporal Knowledge Editing in Large Language Model (Xunjian Yin et al., 2023)

{{<citation>}}

Xunjian Yin, Jin Jiang, Liming Yang, Xiaojun Wan. (2023)  
**History Matters: Temporal Knowledge Editing in Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.05497v3)  

---


**ABSTRACT**  
The imperative task of revising or updating the knowledge stored within large language models arises from two distinct sources: intrinsic errors inherent in the model which should be corrected and outdated knowledge due to external shifts in the real world which should be updated. Prevailing efforts in model editing conflate these two distinct categories of edits arising from distinct reasons and directly modify the original knowledge in models into new knowledge. However, we argue that preserving the model's original knowledge remains pertinent. Specifically, if a model's knowledge becomes outdated due to evolving worldly dynamics, it should retain recollection of the historical knowledge while integrating the newfound knowledge. In this work, we introduce the task of Temporal Knowledge Editing (TKE) and establish a benchmark AToKe (Assessment of TempOral Knowledge Editing) to evaluate current model editing methods. We find that while existing model editing methods are effective at making models remember new knowledge, the edited model catastrophically forgets historical knowledge. To address this gap, we propose a simple and general framework termed Multi-Editing with Time Objective (METO) for enhancing existing editing models, which edits both historical and new knowledge concurrently and optimizes the model's prediction for the time of each fact. Our assessments demonstrate that while AToKe is still difficult, METO maintains the effectiveness of learning new knowledge and meanwhile substantially improves the performance of edited models on utilizing historical knowledge.

{{</citation>}}


### (62/80) Using Captum to Explain Generative Language Models (Vivek Miglani et al., 2023)

{{<citation>}}

Vivek Miglani, Aobo Yang, Aram H. Markosyan, Diego Garcia-Olano, Narine Kokhlikyan. (2023)  
**Using Captum to Explain Generative Language Models**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.05491v1)  

---


**ABSTRACT**  
Captum is a comprehensive library for model explainability in PyTorch, offering a range of methods from the interpretability literature to enhance users' understanding of PyTorch models. In this paper, we introduce new features in Captum that are specifically designed to analyze the behavior of generative language models. We provide an overview of the available functionalities and example applications of their potential for understanding learned associations within generative language models.

{{</citation>}}


### (63/80) Teamwork Dimensions Classification Using BERT (Junyoung Lee et al., 2023)

{{<citation>}}

Junyoung Lee, Elizabeth Koh. (2023)  
**Teamwork Dimensions Classification Using BERT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.05483v1)  

---


**ABSTRACT**  
Teamwork is a necessary competency for students that is often inadequately assessed. Towards providing a formative assessment of student teamwork, an automated natural language processing approach was developed to identify teamwork dimensions of students' online team chat. Developments in the field of natural language processing and artificial intelligence have resulted in advanced deep transfer learning approaches namely the Bidirectional Encoder Representations from Transformers (BERT) model that allow for more in-depth understanding of the context of the text. While traditional machine learning algorithms were used in the previous work for the automatic classification of chat messages into the different teamwork dimensions, our findings have shown that classifiers based on the pre-trained language model BERT provides improved classification performance, as well as much potential for generalizability in the language use of varying team chat contexts and team member demographics. This model will contribute towards an enhanced learning analytics tool for teamwork assessment and feedback.

{{</citation>}}


### (64/80) Fine-Grained Analysis of Team Collaborative Dialogue (Ian Perera et al., 2023)

{{<citation>}}

Ian Perera, Matthew Johnson, Carson Wilber. (2023)  
**Fine-Grained Analysis of Team Collaborative Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.05471v1)  

---


**ABSTRACT**  
Natural language analysis of human collaborative chat dialogues is an understudied domain with many unique challenges: a large number of dialogue act labels, underspecified and dynamic tasks, interleaved topics, and long-range contextual dependence. While prior work has studied broad metrics of team dialogue and associated performance using methods such as LSA, there has been little effort in generating fine-grained descriptions of team dynamics and individual performance from dialogue. We describe initial work towards developing an explainable analytics tool in the software development domain using Slack chats mined from our organization, including generation of a novel, hierarchical labeling scheme; design of descriptive metrics based on the frequency of occurrence of dialogue acts; and initial results using a transformer + CRF architecture to incorporate long-range context.

{{</citation>}}


### (65/80) Textual Toxicity in Social Media: Understanding the Bangla Toxic Language Expressed in Facebook Comment (Mohammad Mamun Or Rashid, 2023)

{{<citation>}}

Mohammad Mamun Or Rashid. (2023)  
**Textual Toxicity in Social Media: Understanding the Bangla Toxic Language Expressed in Facebook Comment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2312.05467v1)  

---


**ABSTRACT**  
Social Media is a repository of digital literature including user-generated content. The users of social media are expressing their opinion with diverse mediums such as text, emojis, memes, and also through other visual and textual mediums. A major portion of these media elements could be treated as harmful to others and they are known by many words including Cyberbullying and Toxic Language . The goal of this research paper is to analyze a curated and value-added dataset of toxic language titled ToxLex_bn . It is an exhaustive wordlist that can be used as classifier material to detect toxicity in social media. The toxic language/script used by the Bengali community as cyberbullying, hate speech and moral policing became major trends in social media culture in Bangladesh and West Bengal. The toxicity became so high that the victims has to post as a counter or release explanation video for the haters. Most cases are pointed to women celebrity and their relation, dress, lifestyle are became trolled and toxicity flooded in comments boxes. Not only celebrity bashing but also hates occurred between Hindu Muslims, India-Bangladesh, Two opponents of 1971 and these are very common for virtual conflict in the comment thread. Even many times facebook comment causes sue and legal matters in Bangladesh and thus it requires more study. In this study, a Bangla toxic language dataset has been analyzed which was inputted by the user in Bengali script & language. For this, about 1968 unique bigrams or phrases as wordlists have been analyzed which are derived from 2207590 comments. It is assumed that this analysis will reinforce the detection of Bangla's toxic language used in social media and thus cure this virtual disease.

{{</citation>}}


### (66/80) Steering Llama 2 via Contrastive Activation Addition (Nina Rimsky et al., 2023)

{{<citation>}}

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner. (2023)  
**Steering Llama 2 via Contrastive Activation Addition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.06681v1)  

---


**ABSTRACT**  
We introduce Contrastive Activation Addition (CAA), an innovative method for steering language models by modifying activations during their forward passes. CAA computes ``steering vectors'' by averaging the difference in residual stream activations between pairs of positive and negative examples of a particular behavior such as factual versus hallucinatory responses. During inference, these steering vectors are added at all token positions after the user's prompt with either a positive or negative coefficient, allowing precise control over the degree of the targeted behavior. We evaluate CAA's effectiveness on Llama 2 Chat using both multiple-choice behavioral question datasets and open-ended generation tasks. We demonstrate that CAA significantly alters model behavior, outperforms traditional methods like finetuning and few-shot prompting, and minimally reduces capabilities. Moreover, by employing various activation space interpretation methods, we gain deeper insights into CAA's mechanisms. CAA both accurately steers model outputs and also sheds light on how high-level concepts are represented in Large Language Models (LLMs).

{{</citation>}}


### (67/80) Domain Adaptation of a State of the Art Text-to-SQL Model: Lessons Learned and Challenges Found (Irene Manotas et al., 2023)

{{<citation>}}

Irene Manotas, Octavian Popescu, Ngoc Phuoc An Vo, Vadim Sheinin. (2023)  
**Domain Adaptation of a State of the Art Text-to-SQL Model: Lessons Learned and Challenges Found**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: Language Model, T5  
[Paper Link](http://arxiv.org/abs/2312.05448v1)  

---


**ABSTRACT**  
There are many recent advanced developments for the Text-to-SQL task, where the Picard model is one of the the top performing models as measured by the Spider dataset competition. However, bringing Text-to-SQL systems to realistic use-cases through domain adaptation remains a tough challenge. We analyze how well the base T5 Language Model and Picard perform on query structures different from the Spider dataset, we fine-tuned the base model on the Spider data and on independent databases (DB). To avoid accessing the DB content online during inference, we also present an alternative way to disambiguate the values in an input question using a rule-based approach that relies on an intermediate representation of the semantic concepts of an input question. In our results we show in what cases T5 and Picard can deliver good performance, we share the lessons learned, and discuss current domain adaptation challenges.

{{</citation>}}


### (68/80) Beneath the Surface: Unveiling Harmful Memes with Multimodal Reasoning Distilled from Large Language Models (Hongzhan Lin et al., 2023)

{{<citation>}}

Hongzhan Lin, Ziyang Luo, Jing Ma, Long Chen. (2023)  
**Beneath the Surface: Unveiling Harmful Memes with Multimodal Reasoning Distilled from Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.05434v1)  

---


**ABSTRACT**  
The age of social media is rife with memes. Understanding and detecting harmful memes pose a significant challenge due to their implicit meaning that is not explicitly conveyed through the surface text and image. However, existing harmful meme detection approaches only recognize superficial harm-indicative signals in an end-to-end classification manner but ignore in-depth cognition of the meme text and image. In this paper, we attempt to detect harmful memes based on advanced reasoning over the interplay of multimodal information in memes. Inspired by the success of Large Language Models (LLMs) on complex reasoning, we first conduct abductive reasoning with LLMs. Then we propose a novel generative framework to learn reasonable thoughts from LLMs for better multimodal fusion and lightweight fine-tuning, which consists of two training stages: 1) Distill multimodal reasoning knowledge from LLMs; and 2) Fine-tune the generative framework to infer harmfulness. Extensive experiments conducted on three meme datasets demonstrate that our proposed approach achieves superior performance than state-of-the-art methods on the harmful meme detection task.

{{</citation>}}


## cs.SI (2)



### (69/80) Polarization in Decentralized Online Social Networks (Lucio La Cava et al., 2023)

{{<citation>}}

Lucio La Cava, Domenico Mandaglio, Andrea Tagarelli. (2023)  
**Polarization in Decentralized Online Social Networks**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI, physics-soc-ph  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2312.05668v1)  

---


**ABSTRACT**  
Centralized social media platforms are currently experiencing a shift in user engagement, drawing attention to alternative paradigms like Decentralized Online Social Networks (DOSNs). The rising popularity of DOSNs finds its root in the accessibility of open-source software, enabling anyone to create a new instance (i.e., server) and participate in a decentralized network known as Fediverse. Despite this growing momentum, there has been a lack of studies addressing the effect of positive and negative interactions among instances within DOSNs. This work aims to fill this gap by presenting a preliminary examination of instances' polarization in DOSNs, focusing on Mastodon -- the most widely recognized decentralized social media platform, boasting over 10M users and nearly 20K instances to date. Our results suggest that polarization in the Fediverse emerges in unique ways, influenced by the desire to foster a federated environment between instances, also facilitating the isolation of instances that may pose potential risks to the Fediverse.

{{</citation>}}


### (70/80) A Hybrid Method of Sentiment Analysis and Machine Learning Algorithm for the U.S. Presidential Election Forecasting (Guocheng Feng et al., 2023)

{{<citation>}}

Guocheng Feng, Huaiyu Cai, Kaihao Chen, Zhijian Li. (2023)  
**A Hybrid Method of Sentiment Analysis and Machine Learning Algorithm for the U.S. Presidential Election Forecasting**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Sentiment Analysis, Twitter  
[Paper Link](http://arxiv.org/abs/2312.05584v1)  

---


**ABSTRACT**  
U.S. Presidential Election forecasting has been a research interest for several decades. Currently, election prediction consists of two main approaches: traditional models that incorporate economic data and poll surveys, and models that leverage Twitter (or X) and other social media platforms due to their increasing popularity in the past decade. However, traditional approaches have predominantly focused on national-level predictions, while social media-based approaches often oversimplify the nuanced differences between online discourse and the broader voting population's political landscape.   In this work, we perform a hybrid method of both the machine learning algorithm and the sentiment analysis on the state level with various independent variables including census data, economic indicators, polling averages, and the newly defined average sentiment scores from Twitter. Our prediction for the 2020 U.S. Presidential Election yielded promising results. Most of our models successfully predicted a victory for the Democratic candidate with 96% accuracy using Gradient Boosting Trees and Multi-Layer Perceptron algorithms. This novel prediction framework addresses the limitations of existing U.S. Presidential Election forecasting approaches, particularly in terms of state-level predictions. It provides a valuable foundation for future research in this field and contributes to advancing our understanding of election dynamics.

{{</citation>}}


## cs.CR (3)



### (71/80) Towards a Graph Neural Network-Based Approach for Estimating Hidden States in Cyber Attack Simulations (Pontus Johnson et al., 2023)

{{<citation>}}

Pontus Johnson, Mathias Ekstedt. (2023)  
**Towards a Graph Neural Network-Based Approach for Estimating Hidden States in Cyber Attack Simulations**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2312.05666v1)  

---


**ABSTRACT**  
This work-in-progress paper introduces a prototype for a novel Graph Neural Network (GNN) based approach to estimate hidden states in cyber attack simulations. Utilizing the Meta Attack Language (MAL) in conjunction with Relational Dynamic Decision Language (RDDL) conformant simulations, our framework aims to map the intricate complexity of cyber attacks with a vast number of possible vectors in the simulations. While the prototype is yet to be completed and validated, we discuss its foundational concepts, the architecture, and the potential implications for the field of computer security.

{{</citation>}}


### (72/80) Enhancing Modbus TCP Protocol Security with eBPF Technology (Jia-Yi Jhan et al., 2023)

{{<citation>}}

Jia-Yi Jhan, Hung-Min Sun. (2023)  
**Enhancing Modbus TCP Protocol Security with eBPF Technology**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.05665v1)  

---


**ABSTRACT**  
The core component of an Industrial Control System (ICS) is often a Programmable Logic Controller (PLC) combined with various modules. In such systems, the communication between devices is mainly based on the Modbus protocol, which was developed by Modicon (now Schneider Electric) in 1979 as an application-level communication protocol and has become a de facto standard for ICS for the past 40 years. Modbus TCP is a variant of this protocol for communications over the TCP/IP network. However, the Modbus protocol was not designed with security in mind, and the use of plaintext transmissions during communication makes information easily accessible to the attackers, while the lack of an authentication mechanism gives any protocol-compliant device the ability to take over control. In this study, we use the eBPF technology to shift the process of protocol change to the lower level of the operating system, making the change transparent to the existing software, and enhancing the security of the Modbus TCP protocol without affecting the existing software ecosystem as much as possible.

{{</citation>}}


### (73/80) Trade-off of Security, Latency, and Throughput of the Nakamoto Consensus (Shujie Cao et al., 2023)

{{<citation>}}

Shujie Cao, Dongning Guo. (2023)  
**Trade-off of Security, Latency, and Throughput of the Nakamoto Consensus**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.05506v1)  

---


**ABSTRACT**  
This paper delves into the fundamental trade-off between security, latency, and throughput in proof-of-work longest-chain-wins protocols, also known as the Nakamoto consensus. New upper and lower bounds on the probability of violating transaction safety are derived as a function of honest and adversarial mining rates, an upper bound on block propagation delays, and transaction confirmation latency, both in time and in block depth. The results include a first closed-form finite-latency bound applicable to all delays and mining rates up to the ultimate fault tolerance. Notably, for most parameters relevant to Bitcoin and proof-of-work Ethereum, the gap between the upper and lower bounds is significantly narrower than the best gaps previously established in the literature. Furthermore, the paper reveals a fundamental trade-off between transaction throughput and confirmation latency, ultimately determined by the desired fault tolerance and the growth of block propagation delay as block size increases.

{{</citation>}}


## stat.ML (2)



### (74/80) Sample-Optimal Locally Private Hypothesis Selection and the Provable Benefits of Interactivity (Alireza F. Pour et al., 2023)

{{<citation>}}

Alireza F. Pour, Hassan Ashtiani, Shahab Asoodeh. (2023)  
**Sample-Optimal Locally Private Hypothesis Selection and the Provable Benefits of Interactivity**  

---
Primary Category: stat.ML  
Categories: cs-CR, cs-IT, cs-LG, math-IT, stat-ML, stat.ML  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.05645v1)  

---


**ABSTRACT**  
We study the problem of hypothesis selection under the constraint of local differential privacy. Given a class $\mathcal{F}$ of $k$ distributions and a set of i.i.d. samples from an unknown distribution $h$, the goal of hypothesis selection is to pick a distribution $\hat{f}$ whose total variation distance to $h$ is comparable with the best distribution in $\mathcal{F}$ (with high probability). We devise an $\varepsilon$-locally-differentially-private ($\varepsilon$-LDP) algorithm that uses $\Theta\left(\frac{k}{\alpha^2\min \{\varepsilon^2,1\}}\right)$ samples to guarantee that $d_{TV}(h,\hat{f})\leq \alpha + 9 \min_{f\in \mathcal{F}}d_{TV}(h,f)$ with high probability. This sample complexity is optimal for $\varepsilon<1$, matching the lower bound of Gopi et al. (2020). All previously known algorithms for this problem required $\Omega\left(\frac{k\log k}{\alpha^2\min \{ \varepsilon^2 ,1\}} \right)$ samples to work.   Moreover, our result demonstrates the power of interaction for $\varepsilon$-LDP hypothesis selection. Namely, it breaks the known lower bound of $\Omega\left(\frac{k\log k}{\alpha^2\min \{ \varepsilon^2 ,1\}} \right)$ for the sample complexity of non-interactive hypothesis selection. Our algorithm breaks this barrier using only $\Theta(\log \log k)$ rounds of interaction.   To prove our results, we define the notion of \emph{critical queries} for a Statistical Query Algorithm (SQA) which may be of independent interest. Informally, an SQA is said to use a small number of critical queries if its success relies on the accuracy of only a small number of queries it asks. We then design an LDP algorithm that uses a smaller number of critical queries.

{{</citation>}}


### (75/80) Distributional Bellman Operators over Mean Embeddings (Li Kevin Wenliang et al., 2023)

{{<citation>}}

Li Kevin Wenliang, Grégoire Déletang, Matthew Aitchison, Marcus Hutter, Anian Ruoss, Arthur Gretton, Mark Rowland. (2023)  
**Distributional Bellman Operators over Mean Embeddings**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.07358v1)  

---


**ABSTRACT**  
We propose a novel algorithmic framework for distributional reinforcement learning, based on learning finite-dimensional mean embeddings of return distributions. We derive several new algorithms for dynamic programming and temporal-difference learning based on this framework, provide asymptotic convergence theory, and examine the empirical performance of the algorithms on a suite of tabular tasks. Further, we show that this approach can be straightforwardly combined with deep reinforcement learning, and obtain a new deep RL agent that improves over baseline distributional approaches on the Arcade Learning Environment.

{{</citation>}}


## cs.NE (1)



### (76/80) NiSNN-A: Non-iterative Spiking Neural Networks with Attention with Application to Motor Imagery EEG Classification (Chuhan Zhang et al., 2023)

{{<citation>}}

Chuhan Zhang, Wei Pan, Cosimo Della Santina. (2023)  
**NiSNN-A: Non-iterative Spiking Neural Networks with Attention with Application to Motor Imagery EEG Classification**  

---
Primary Category: cs.NE  
Categories: cs-LG, cs-NE, cs.NE  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.05643v1)  

---


**ABSTRACT**  
Motor imagery, an important category in electroencephalogram (EEG) research, often intersects with scenarios demanding low energy consumption, such as portable medical devices and isolated environment operations. Traditional deep learning algorithms, despite their effectiveness, are characterized by significant computational demands accompanied by high energy usage. As an alternative, spiking neural networks (SNNs), inspired by the biological functions of the brain, emerge as a promising energy-efficient solution. However, SNNs typically exhibit lower accuracy than their counterpart convolutional neural networks (CNNs). Although attention mechanisms successfully increase network accuracy by focusing on relevant features, their integration in the SNN framework remains an open question. In this work, we combine the SNN and the attention mechanisms for the EEG classification, aiming to improve precision and reduce energy consumption. To this end, we first propose a Non-iterative Leaky Integrate-and-Fire (LIF) neuron model, overcoming the gradient issues in the traditional SNNs using the Iterative LIF neurons. Then, we introduce the sequence-based attention mechanisms to refine the feature map. We evaluated the proposed Non-iterative SNN with Attention (NiSNN-A) model on OpenBMI, a large-scale motor imagery dataset. Experiment results demonstrate that 1) our model outperforms other SNN models by achieving higher accuracy, 2) our model increases energy efficiency compared to the counterpart CNN models (i.e., by 2.27 times) while maintaining comparable accuracy.

{{</citation>}}


## cs.SD (1)



### (77/80) Keyword spotting -- Detecting commands in speech using deep learning (Sumedha Rai et al., 2023)

{{<citation>}}

Sumedha Rai, Tong Li, Bella Lyu. (2023)  
**Keyword spotting -- Detecting commands in speech using deep learning**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-CL, cs-HC, cs-SD, cs.SD, eess-AS  
Keywords: Attention, LSTM  
[Paper Link](http://arxiv.org/abs/2312.05640v1)  

---


**ABSTRACT**  
Speech recognition has become an important task in the development of machine learning and artificial intelligence. In this study, we explore the important task of keyword spotting using speech recognition machine learning and deep learning techniques. We implement feature engineering by converting raw waveforms to Mel Frequency Cepstral Coefficients (MFCCs), which we use as inputs to our models. We experiment with several different algorithms such as Hidden Markov Model with Gaussian Mixture, Convolutional Neural Networks and variants of Recurrent Neural Networks including Long Short-Term Memory and the Attention mechanism. In our experiments, RNN with BiLSTM and Attention achieves the best performance with an accuracy of 93.9 %

{{</citation>}}


## cs.DC (1)



### (78/80) JITSPMM: Just-in-Time Instruction Generation for Accelerated Sparse Matrix-Matrix Multiplication (Qiang Fu et al., 2023)

{{<citation>}}

Qiang Fu, Thomas B. Rolinger, H. Howie Huang. (2023)  
**JITSPMM: Just-in-Time Instruction Generation for Accelerated Sparse Matrix-Matrix Multiplication**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-PF, cs-PL, cs.DC  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.05639v1)  

---


**ABSTRACT**  
Achieving high performance for Sparse MatrixMatrix Multiplication (SpMM) has received increasing research attention, especially on multi-core CPUs, due to the large input data size in applications such as graph neural networks (GNNs). Most existing solutions for SpMM computation follow the aheadof-time (AOT) compilation approach, which compiles a program entirely before it is executed. AOT compilation for SpMM faces three key limitations: unnecessary memory access, additional branch overhead, and redundant instructions. These limitations stem from the fact that crucial information pertaining to SpMM is not known until runtime. In this paper, we propose JITSPMM, a just-in-time (JIT) assembly code generation framework to accelerated SpMM computation on multi-core CPUs with SIMD extensions. First, JITSPMM integrates the JIT assembly code generation technique into three widely-used workload division methods for SpMM to achieve balanced workload distribution among CPU threads. Next, with the availability of runtime information, JITSPMM employs a novel technique, coarse-grain column merging, to maximize instruction-level parallelism by unrolling the performance-critical loop. Furthermore, JITSPMM intelligently allocates registers to cache frequently accessed data to minimizing memory accesses, and employs selected SIMD instructions to enhance arithmetic throughput. We conduct a performance evaluation of JITSPMM and compare it two AOT baselines. The first involves existing SpMM implementations compiled using the Intel icc compiler with auto-vectorization. The second utilizes the highly-optimized SpMM routine provided by Intel MKL. Our results show that JITSPMM provides an average improvement of 3.8x and 1.4x, respectively.

{{</citation>}}


## cs.NI (1)



### (79/80) Generative AI for Physical Layer Communications: A Survey (Nguyen Van Huynh et al., 2023)

{{<citation>}}

Nguyen Van Huynh, Jiacheng Wang, Hongyang Du, Dinh Thai Hoang, Dusit Niyato, Diep N. Nguyen, Dong In Kim, Khaled B. Letaief. (2023)  
**Generative AI for Physical Layer Communications: A Survey**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-NI, cs.NI  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.05594v1)  

---


**ABSTRACT**  
The recent evolution of generative artificial intelligence (GAI) leads to the emergence of groundbreaking applications such as ChatGPT, which not only enhances the efficiency of digital content production, such as text, audio, video, or even network traffic data, but also enriches its diversity. Beyond digital content creation, GAI's capability in analyzing complex data distributions offers great potential for wireless communications, particularly amidst a rapid expansion of new physical layer communication technologies. For example, the diffusion model can learn input signal distributions and use them to improve the channel estimation accuracy, while the variational autoencoder can model channel distribution and infer latent variables for blind channel equalization. Therefore, this paper presents a comprehensive investigation of GAI's applications for communications at the physical layer, ranging from traditional issues, including signal classification, channel estimation, and equalization, to emerging topics, such as intelligent reflecting surfaces and joint source channel coding. We also compare GAI-enabled physical layer communications with those supported by traditional AI, highlighting GAI's inherent capabilities and unique contributions in these areas. Finally, the paper discusses open issues and proposes several future research directions, laying a foundation for further exploration and advancement of GAI in physical layer communications.

{{</citation>}}


## eess.SP (1)



### (80/80) Annotating sleep states in children from wrist-worn accelerometer data using Machine Learning (Ashwin Ram et al., 2023)

{{<citation>}}

Ashwin Ram, Sundar Sripada V. S., Shuvam Keshari, Zizhe Jiang. (2023)  
**Annotating sleep states in children from wrist-worn accelerometer data using Machine Learning**  

---
Primary Category: eess.SP  
Categories: cs-CV, cs-CY, cs-LG, eess-SP, eess.SP  
Keywords: Event Detection, LSTM  
[Paper Link](http://arxiv.org/abs/2312.07561v1)  

---


**ABSTRACT**  
Sleep detection and annotation are crucial for researchers to understand sleep patterns, especially in children. With modern wrist-worn watches comprising built-in accelerometers, sleep logs can be collected. However, the annotation of these logs into distinct sleep events: onset and wakeup, proves to be challenging. These annotations must be automated, precise, and scalable. We propose to model the accelerometer data using different machine learning (ML) techniques such as support vectors, boosting, ensemble methods, and more complex approaches involving LSTMs and Region-based CNNs. Later, we aim to evaluate these approaches using the Event Detection Average Precision (EDAP) score (similar to the IOU metric) to eventually compare the predictive power and model performance.

{{</citation>}}
