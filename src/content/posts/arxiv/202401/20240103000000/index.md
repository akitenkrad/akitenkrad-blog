---
draft: false
title: "arXiv @ 2024.01.03"
date: 2024-01-03
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.03"
    identifier: arxiv_20240103
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (9)](#cslg-9)
- [cs.CV (12)](#cscv-12)
- [cs.SE (4)](#csse-4)
- [cs.AI (1)](#csai-1)
- [cs.CL (10)](#cscl-10)
- [cs.HC (1)](#cshc-1)
- [cs.CR (3)](#cscr-3)
- [cs.RO (1)](#csro-1)
- [cs.SI (2)](#cssi-2)
- [cs.GR (1)](#csgr-1)
- [physics.comp-ph (1)](#physicscomp-ph-1)
- [eess.IV (3)](#eessiv-3)
- [q-bio.NC (1)](#q-bionc-1)
- [math.DS (1)](#mathds-1)
- [eess.SP (1)](#eesssp-1)
- [cs.SD (1)](#cssd-1)
- [cs.NI (1)](#csni-1)

## cs.LG (9)



### (1/53) Downstream Task-Oriented Generative Model Selections on Synthetic Data Training for Fraud Detection Models (Yinan Cheng et al., 2024)

{{<citation>}}

Yinan Cheng, Chi-Hua Wang, Vamsi K. Potluru, Tucker Balch, Guang Cheng. (2024)  
**Downstream Task-Oriented Generative Model Selections on Synthetic Data Training for Fraud Detection Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Fraud Detection  
[Paper Link](http://arxiv.org/abs/2401.00974v1)  

---


**ABSTRACT**  
Devising procedures for downstream task-oriented generative model selections is an unresolved problem of practical importance. Existing studies focused on the utility of a single family of generative models. They provided limited insights on how synthetic data practitioners select the best family generative models for synthetic training tasks given a specific combination of machine learning model class and performance metric. In this paper, we approach the downstream task-oriented generative model selections problem in the case of training fraud detection models and investigate the best practice given different combinations of model interpretability and model performance constraints. Our investigation supports that, while both Neural Network(NN)-based and Bayesian Network(BN)-based generative models are both good to complete synthetic training task under loose model interpretability constrain, the BN-based generative models is better than NN-based when synthetic training fraud detection model under strict model interpretability constrain. Our results provides practical guidance for machine learning practitioner who is interested in replacing their training dataset from real to synthetic, and shed lights on more general downstream task-oriented generative model selection problems.

{{</citation>}}


### (2/53) Improve Fidelity and Utility of Synthetic Credit Card Transaction Time Series from Data-centric Perspective (Din-Yin Hsieh et al., 2024)

{{<citation>}}

Din-Yin Hsieh, Chi-Hua Wang, Guang Cheng. (2024)  
**Improve Fidelity and Utility of Synthetic Credit Card Transaction Time Series from Data-centric Perspective**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.00965v1)  

---


**ABSTRACT**  
Exploring generative model training for synthetic tabular data, specifically in sequential contexts such as credit card transaction data, presents significant challenges. This paper addresses these challenges, focusing on attaining both high fidelity to actual data and optimal utility for machine learning tasks. We introduce five pre-processing schemas to enhance the training of the Conditional Probabilistic Auto-Regressive Model (CPAR), demonstrating incremental improvements in the synthetic data's fidelity and utility. Upon achieving satisfactory fidelity levels, our attention shifts to training fraud detection models tailored for time-series data, evaluating the utility of the synthetic data. Our findings offer valuable insights and practical guidelines for synthetic data practitioners in the finance sector, transitioning from real to synthetic datasets for training purposes, and illuminating broader methodologies for synthesizing credit card transaction time series.

{{</citation>}}


### (3/53) SecFormer: Towards Fast and Accurate Privacy-Preserving Inference for Large Language Models (Jinglong Luo et al., 2024)

{{<citation>}}

Jinglong Luo, Yehong Zhang, Jiaqi Zhang, Xin Mu, Hui Wang, Yue Yu, Zenglin Xu. (2024)  
**SecFormer: Towards Fast and Accurate Privacy-Preserving Inference for Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: BERT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.00793v1)  

---


**ABSTRACT**  
With the growing use of large language models hosted on cloud platforms to offer inference services, privacy concerns are escalating, especially concerning sensitive data like investment plans and bank account details. Secure Multi-Party Computing (SMPC) emerges as a promising solution to protect the privacy of inference data and model parameters. However, the application of SMPC in Privacy-Preserving Inference (PPI) for large language models, particularly those based on the Transformer architecture, often leads to considerable slowdowns or declines in performance. This is largely due to the multitude of nonlinear operations in the Transformer architecture, which are not well-suited to SMPC and are difficult to circumvent or optimize effectively. To address this concern, we introduce an advanced optimization framework called SecFormer, designed to strike an optimal balance between performance and efficiency in PPI for Transformer models. By implementing knowledge distillation techniques, we successfully eliminate the high-cost exponential and maximum operations in PPI without sacrificing model performance. Additionally, we have developed a suite of efficient SMPC protocols that utilize segmented polynomials and Goldschmidt's method to handle other complex nonlinear functions within PPI, such as GeLU, LayerNorm, and Softmax. Our extensive experiments reveal that SecFormer outperforms MPCFormer in performance, showing improvements of $5.6\%$ and $24.2\%$ for BERT$_{\text{BASE}}$ and BERT$_{\text{LARGE}}$, respectively. In terms of efficiency, SecFormer is 3.4 and 3.2 times faster than Puma, demonstrating its effectiveness and speed.

{{</citation>}}


### (4/53) MPRE: Multi-perspective Patient Representation Extractor for Disease Prediction (Ziyue Yu et al., 2024)

{{<citation>}}

Ziyue Yu, Jiayi Wang, Wuman Luo, Rita Tse, Giovanni Pau. (2024)  
**MPRE: Multi-perspective Patient Representation Extractor for Disease Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.00756v1)  

---


**ABSTRACT**  
Patient representation learning based on electronic health records (EHR) is a critical task for disease prediction. This task aims to effectively extract useful information on dynamic features. Although various existing works have achieved remarkable progress, the model performance can be further improved by fully extracting the trends, variations, and the correlation between the trends and variations in dynamic features. In addition, sparse visit records limit the performance of deep learning models. To address these issues, we propose the Multi-perspective Patient Representation Extractor (MPRE) for disease prediction. Specifically, we propose Frequency Transformation Module (FTM) to extract the trend and variation information of dynamic features in the time-frequency domain, which can enhance the feature representation. In the 2D Multi-Extraction Network (2D MEN), we form the 2D temporal tensor based on trend and variation. Then, the correlations between trend and variation are captured by the proposed dilated operation. Moreover, we propose the First-Order Difference Attention Mechanism (FODAM) to calculate the contributions of differences in adjacent variations to the disease diagnosis adaptively. To evaluate the performance of MPRE and baseline methods, we conduct extensive experiments on two real-world public datasets. The experiment results show that MPRE outperforms state-of-the-art baseline methods in terms of AUROC and AUPRC.

{{</citation>}}


### (5/53) Saliency-Aware Regularized Graph Neural Network (Wenjie Pei et al., 2024)

{{<citation>}}

Wenjie Pei, Weina Xu, Zongze Wu, Weichao Li, Jinfan Wang, Guangming Lu, Xiangrong Wang. (2024)  
**Saliency-Aware Regularized Graph Neural Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2401.00755v1)  

---


**ABSTRACT**  
The crux of graph classification lies in the effective representation learning for the entire graph. Typical graph neural networks focus on modeling the local dependencies when aggregating features of neighboring nodes, and obtain the representation for the entire graph by aggregating node features. Such methods have two potential limitations: 1) the global node saliency w.r.t. graph classification is not explicitly modeled, which is crucial since different nodes may have different semantic relevance to graph classification; 2) the graph representation directly aggregated from node features may have limited effectiveness to reflect graph-level information. In this work, we propose the Saliency-Aware Regularized Graph Neural Network (SAR-GNN) for graph classification, which consists of two core modules: 1) a traditional graph neural network serving as the backbone for learning node features and 2) the Graph Neural Memory designed to distill a compact graph representation from node features of the backbone. We first estimate the global node saliency by measuring the semantic similarity between the compact graph representation and node features. Then the learned saliency distribution is leveraged to regularize the neighborhood aggregation of the backbone, which facilitates the message passing of features for salient nodes and suppresses the less relevant nodes. Thus, our model can learn more effective graph representation. We demonstrate the merits of SAR-GNN by extensive experiments on seven datasets across various types of graph data. Code will be released.

{{</citation>}}


### (6/53) A Survey on Graph Neural Networks in Intelligent Transportation Systems (Hourun Li et al., 2024)

{{<citation>}}

Hourun Li, Yusheng Zhao, Zhengyang Mao, Yifang Qin, Zhiping Xiao, Jiaqi Feng, Yiyang Gu, Wei Ju, Xiao Luo, Ming Zhang. (2024)  
**A Survey on Graph Neural Networks in Intelligent Transportation Systems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.00713v2)  

---


**ABSTRACT**  
Intelligent Transportation System (ITS) is vital in improving traffic congestion, reducing traffic accidents, optimizing urban planning, etc. However, due to the complexity of the traffic network, traditional machine learning and statistical methods are relegated to the background. With the advent of the artificial intelligence era, many deep learning frameworks have made remarkable progress in various fields and are now considered effective methods in many areas. As a deep learning method, Graph Neural Networks (GNNs) have emerged as a highly competitive method in the ITS field since 2019 due to their strong ability to model graph-related problems. As a result, more and more scholars pay attention to the applications of GNNs in transportation domains, which have shown excellent performance. However, most of the research in this area is still concentrated on traffic forecasting, while other ITS domains, such as autonomous vehicles and urban planning, still require more attention. This paper aims to review the applications of GNNs in six representative and emerging ITS domains: traffic forecasting, autonomous vehicles, traffic signal control, transportation safety, demand prediction, and parking management. We have reviewed extensive graph-related studies from 2018 to 2023, summarized their methods, features, and contributions, and presented them in informative tables or lists. Finally, we have identified the challenges of applying GNNs to ITS and suggested potential future directions.

{{</citation>}}


### (7/53) Communication-Efficient Federated Learning for LEO Constellations Integrated with HAPs Using Hybrid NOMA-OFDM (Mohamed Elmahallawy et al., 2024)

{{<citation>}}

Mohamed Elmahallawy, Tie Luo, Khaled Ramadan. (2024)  
**Communication-Efficient Federated Learning for LEO Constellations Integrated with HAPs Using Hybrid NOMA-OFDM**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.00685v1)  

---


**ABSTRACT**  
Space AI has become increasingly important and sometimes even necessary for government, businesses, and society. An active research topic under this mission is integrating federated learning (FL) with satellite communications (SatCom) so that numerous low Earth orbit (LEO) satellites can collaboratively train a machine learning model. However, the special communication environment of SatCom leads to a very slow FL training process up to days and weeks. This paper proposes NomaFedHAP, a novel FL-SatCom approach tailored to LEO satellites, that (1) utilizes high-altitude platforms (HAPs) as distributed parameter servers (PS) to enhance satellite visibility, and (2) introduces non-orthogonal multiple access (NOMA) into LEO to enable fast and bandwidth-efficient model transmissions. In addition, NomaFedHAP includes (3) a new communication topology that exploits HAPs to bridge satellites among different orbits to mitigate the Doppler shift, and (4) a new FL model aggregation scheme that optimally balances models between different orbits and shells. Moreover, we (5) derive a closed-form expression of the outage probability for satellites in near and far shells, as well as for the entire system. Our extensive simulations have validated the mathematical analysis and demonstrated the superior performance of NomaFedHAP in achieving fast and efficient FL model convergence with high accuracy as compared to the state-of-the-art.

{{</citation>}}


### (8/53) On Discprecncies between Perturbation Evaluations of Graph Neural Network Attributions (Razieh Rezaei et al., 2024)

{{<citation>}}

Razieh Rezaei, Alireza Dizaji, Ashkan Khakzar, Anees Kazi, Nassir Navab, Daniel Rueckert. (2024)  
**On Discprecncies between Perturbation Evaluations of Graph Neural Network Attributions**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2401.00633v1)  

---


**ABSTRACT**  
Neural networks are increasingly finding their way into the realm of graphs and modeling relationships between features. Concurrently graph neural network explanation approaches are being invented to uncover relationships between the nodes of the graphs. However, there is a disparity between the existing attribution methods, and it is unclear which attribution to trust. Therefore research has introduced evaluation experiments that assess them from different perspectives. In this work, we assess attribution methods from a perspective not previously explored in the graph domain: retraining. The core idea is to retrain the network on important (or not important) relationships as identified by the attributions and evaluate how networks can generalize based on these relationships. We reformulate the retraining framework to sidestep issues lurking in the previous formulation and propose guidelines for correct analysis. We run our analysis on four state-of-the-art GNN attribution methods and five synthetic and real-world graph classification datasets. The analysis reveals that attributions perform variably depending on the dataset and the network. Most importantly, we observe that the famous GNNExplainer performs similarly to an arbitrary designation of edge importance. The study concludes that the retraining evaluation cannot be used as a generalized benchmark and recommends it as a toolset to evaluate attributions on a specifically addressed network, dataset, and sparsity.

{{</citation>}}


### (9/53) Beyond Efficiency: A Systematic Survey of Resource-Efficient Large Language Models (Guangji Bai et al., 2024)

{{<citation>}}

Guangji Bai, Zheng Chai, Chen Ling, Shiyu Wang, Jiaying Lu, Nan Zhang, Tingwei Shi, Ziyang Yu, Mengdan Zhu, Yifei Zhang, Carl Yang, Yue Cheng, Liang Zhao. (2024)  
**Beyond Efficiency: A Systematic Survey of Resource-Efficient Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.00625v2)  

---


**ABSTRACT**  
The burgeoning field of Large Language Models (LLMs), exemplified by sophisticated models like OpenAI's ChatGPT, represents a significant advancement in artificial intelligence. These models, however, bring forth substantial challenges in the high consumption of computational, memory, energy, and financial resources, especially in environments with limited resource capabilities. This survey aims to systematically address these challenges by reviewing a broad spectrum of techniques designed to enhance the resource efficiency of LLMs. We categorize methods based on their optimization focus: computational, memory, energy, financial, and network resources and their applicability across various stages of an LLM's lifecycle, including architecture design, pretraining, finetuning, and system design. Additionally, the survey introduces a nuanced categorization of resource efficiency techniques by their specific resource types, which uncovers the intricate relationships and mappings between various resources and corresponding optimization techniques. A standardized set of evaluation metrics and datasets is also presented to facilitate consistent and fair comparisons across different models and techniques. By offering a comprehensive overview of the current sota and identifying open research avenues, this survey serves as a foundational reference for researchers and practitioners, aiding them in developing more sustainable and efficient LLMs in a rapidly evolving landscape.

{{</citation>}}


## cs.CV (12)



### (10/53) Efficient Multi-domain Text Recognition Deep Neural Network Parameterization with Residual Adapters (Jiayou Chao et al., 2024)

{{<citation>}}

Jiayou Chao, Wei Zhu. (2024)  
**Efficient Multi-domain Text Recognition Deep Neural Network Parameterization with Residual Adapters**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2401.00971v1)  

---


**ABSTRACT**  
Recent advancements in deep neural networks have markedly enhanced the performance of computer vision tasks, yet the specialized nature of these networks often necessitates extensive data and high computational power. Addressing these requirements, this study presents a novel neural network model adept at optical character recognition (OCR) across diverse domains, leveraging the strengths of multi-task learning to improve efficiency and generalization. The model is designed to achieve rapid adaptation to new domains, maintain a compact size conducive to reduced computational resource demand, ensure high accuracy, retain knowledge from previous learning experiences, and allow for domain-specific performance improvements without the need to retrain entirely. Rigorous evaluation on open datasets has validated the model's ability to significantly lower the number of trainable parameters without sacrificing performance, indicating its potential as a scalable and adaptable solution in the field of computer vision, particularly for applications in optical text recognition.

{{</citation>}}


### (11/53) Data Augmentation Techniques for Cross-Domain WiFi CSI-based Human Activity Recognition (Julian Strohmayer et al., 2024)

{{<citation>}}

Julian Strohmayer, Martin Kampel. (2024)  
**Data Augmentation Techniques for Cross-Domain WiFi CSI-based Human Activity Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2401.00964v1)  

---


**ABSTRACT**  
The recognition of human activities based on WiFi Channel State Information (CSI) enables contactless and visual privacy-preserving sensing in indoor environments. However, poor model generalization, due to varying environmental conditions and sensing hardware, is a well-known problem in this space. To address this issue, in this work, data augmentation techniques commonly used in image-based learning are applied to WiFi CSI to investigate their effects on model generalization performance in cross-scenario and cross-system settings. In particular, we focus on the generalization between line-of-sight (LOS) and non-line-of-sight (NLOS) through-wall scenarios, as well as on the generalization between different antenna systems, which remains under-explored. We collect and make publicly available a dataset of CSI amplitude spectrograms of human activities. Utilizing this data, an ablation study is conducted in which activity recognition models based on the EfficientNetV2 architecture are trained, allowing us to assess the effects of each augmentation on model generalization performance. The gathered results show that specific combinations of simple data augmentation techniques applied to CSI amplitude data can significantly improve cross-scenario and cross-system generalization.

{{</citation>}}


### (12/53) DiffAugment: Diffusion based Long-Tailed Visual Relationship Recognition (Parul Gupta et al., 2024)

{{<citation>}}

Parul Gupta, Tuan Nguyen, Abhinav Dhall, Munawar Hayat, Trung Le, Thanh-Toan Do. (2024)  
**DiffAugment: Diffusion based Long-Tailed Visual Relationship Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.01387v1)  

---


**ABSTRACT**  
The task of Visual Relationship Recognition (VRR) aims to identify relationships between two interacting objects in an image and is particularly challenging due to the widely-spread and highly imbalanced distribution of <subject, relation, object> triplets. To overcome the resultant performance bias in existing VRR approaches, we introduce DiffAugment -- a method which first augments the tail classes in the linguistic space by making use of WordNet and then utilizes the generative prowess of Diffusion Models to expand the visual space for minority classes. We propose a novel hardness-aware component in diffusion which is based upon the hardness of each <S,R,O> triplet and demonstrate the effectiveness of hardness-aware diffusion in generating visual embeddings for the tail classes. We also propose a novel subject and object based seeding strategy for diffusion sampling which improves the discriminative capability of the generated visual embeddings. Extensive experimentation on the GQA-LT dataset shows favorable gains in the subject/object and relation average per-class accuracy using Diffusion augmented samples.

{{</citation>}}


### (13/53) Boundary Attention: Learning to Find Faint Boundaries at Any Resolution (Mia Gaia Polansky et al., 2024)

{{<citation>}}

Mia Gaia Polansky, Charles Herrmann, Junhwa Hur, Deqing Sun, Dor Verbin, Todd Zickler. (2024)  
**Boundary Attention: Learning to Find Faint Boundaries at Any Resolution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.00935v1)  

---


**ABSTRACT**  
We present a differentiable model that explicitly models boundaries -- including contours, corners and junctions -- using a new mechanism that we call boundary attention. We show that our model provides accurate results even when the boundary signal is very weak or is swamped by noise. Compared to previous classical methods for finding faint boundaries, our model has the advantages of being differentiable; being scalable to larger images; and automatically adapting to an appropriate level of geometric detail in each part of an image. Compared to previous deep methods for finding boundaries via end-to-end training, it has the advantages of providing sub-pixel precision, being more resilient to noise, and being able to process any image at its native resolution and aspect ratio.

{{</citation>}}


### (14/53) COSMO: COntrastive Streamlined MultimOdal Model with Interleaved Pre-Training (Alex Jinpeng Wang et al., 2024)

{{<citation>}}

Alex Jinpeng Wang, Linjie Li, Kevin Qinghong Lin, Jianfeng Wang, Kevin Lin, Zhengyuan Yang, Lijuan Wang, Mike Zheng Shou. (2024)  
**COSMO: COntrastive Streamlined MultimOdal Model with Interleaved Pre-Training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.00849v1)  

---


**ABSTRACT**  
In the evolution of Vision-Language Pre-training, shifting from short-text comprehension to encompassing extended textual contexts is pivotal. Recent autoregressive vision-language models like \cite{flamingo, palme}, leveraging the long-context capability of Large Language Models, have excelled in few-shot text generation tasks but face challenges in alignment tasks. Addressing this gap, we introduce the contrastive loss into text generation models, presenting the COntrastive-Streamlined MultimOdal framework (\ModelName), strategically partitioning the language model into dedicated unimodal text processing and adept multimodal data handling components. \ModelName, our unified framework, merges unimodal and multimodal elements, enhancing model performance for tasks involving textual and visual data while notably reducing learnable parameters. However, these models demand extensive long-text datasets, yet the availability of high-quality long-text video datasets remains limited. To bridge this gap, this work introduces \VideoDatasetName, an inaugural interleaved video-text dataset featuring comprehensive captions, marking a significant step forward. Demonstrating its impact, we illustrate how \VideoDatasetName{} enhances model performance in image-text tasks. With 34% learnable parameters and utilizing 72\% of the available data, our model demonstrates significant superiority over OpenFlamingo~\cite{openflamingo}. For instance, in the 4-shot flickr captioning task, performance notably improves from 57.2% to 65.\%. The contributions of \ModelName{} and \VideoDatasetName{} are underscored by notable performance gains across 14 diverse downstream datasets encompassing both image-text and video-text tasks.

{{</citation>}}


### (15/53) Mocap Everyone Everywhere: Lightweight Motion Capture With Smartwatches and a Head-Mounted Camera (Jiye Lee et al., 2024)

{{<citation>}}

Jiye Lee, Hanbyul Joo. (2024)  
**Mocap Everyone Everywhere: Lightweight Motion Capture With Smartwatches and a Head-Mounted Camera**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.00847v1)  

---


**ABSTRACT**  
We present a lightweight and affordable motion capture method based on two smartwatches and a head-mounted camera. In contrast to the existing approaches that use six or more expert-level IMU devices, our approach is much more cost-effective and convenient. Our method can make wearable motion capture accessible to everyone everywhere, enabling 3D full-body motion capture in diverse environments. As a key idea to overcome the extreme sparsity and ambiguities of sensor inputs, we integrate 6D head poses obtained from the head-mounted cameras for motion estimation. To enable capture in expansive indoor and outdoor scenes, we propose an algorithm to track and update floor level changes to define head poses, coupled with a multi-stage Transformer-based regression module. We also introduce novel strategies leveraging visual cues of egocentric images to further enhance the motion capture quality while reducing ambiguities. We demonstrate the performance of our method on various challenging scenarios, including complex outdoor environments and everyday motions including object interactions and social interactions among multiple individuals.

{{</citation>}}


### (16/53) Rethinking RAFT for Efficient Optical Flow (Navid Eslami et al., 2024)

{{<citation>}}

Navid Eslami, Farnoosh Arefi, Amir M. Mansourian, Shohreh Kasaei. (2024)  
**Rethinking RAFT for Efficient Optical Flow**  

---
Primary Category: cs.CV  
Categories: ACM-class: F-2-2, I-2-7, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.00833v1)  

---


**ABSTRACT**  
Despite significant progress in deep learning-based optical flow methods, accurately estimating large displacements and repetitive patterns remains a challenge. The limitations of local features and similarity search patterns used in these algorithms contribute to this issue. Additionally, some existing methods suffer from slow runtime and excessive graphic memory consumption. To address these problems, this paper proposes a novel approach based on the RAFT framework. The proposed Attention-based Feature Localization (AFL) approach incorporates the attention mechanism to handle global feature extraction and address repetitive patterns. It introduces an operator for matching pixels with corresponding counterparts in the second frame and assigning accurate flow values. Furthermore, an Amorphous Lookup Operator (ALO) is proposed to enhance convergence speed and improve RAFTs ability to handle large displacements by reducing data redundancy in its search operator and expanding the search space for similarity extraction. The proposed method, Efficient RAFT (Ef-RAFT),achieves significant improvements of 10% on the Sintel dataset and 5% on the KITTI dataset over RAFT. Remarkably, these enhancements are attained with a modest 33% reduction in speed and a mere 13% increase in memory usage. The code is available at: https://github.com/n3slami/Ef-RAFT

{{</citation>}}


### (17/53) DiffMorph: Text-less Image Morphing with Diffusion Models (Shounak Chatterjee, 2024)

{{<citation>}}

Shounak Chatterjee. (2024)  
**DiffMorph: Text-less Image Morphing with Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.00739v1)  

---


**ABSTRACT**  
Text-conditioned image generation models are a prevalent use of AI image synthesis, yet intuitively controlling output guided by an artist remains challenging. Current methods require multiple images and textual prompts for each object to specify them as concepts to generate a single customized image.   On the other hand, our work, \verb|DiffMorph|, introduces a novel approach that synthesizes images that mix concepts without the use of textual prompts. Our work integrates a sketch-to-image module to incorporate user sketches as input. \verb|DiffMorph| takes an initial image with conditioning artist-drawn sketches to generate a morphed image.   We employ a pre-trained text-to-image diffusion model and fine-tune it to reconstruct each image faithfully. We seamlessly merge images and concepts from sketches into a cohesive composition. The image generation capability of our work is demonstrated through our results and a comparison of these with prompt-based image generation.

{{</citation>}}


### (18/53) BRAU-Net++: U-Shaped Hybrid CNN-Transformer Network for Medical Image Segmentation (Libin Lan et al., 2024)

{{<citation>}}

Libin Lan, Pengzhou Cai, Lu Jiang, Xiaojuan Liu, Yongmei Li, Yudong Zhang. (2024)  
**BRAU-Net++: U-Shaped Hybrid CNN-Transformer Network for Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.00722v1)  

---


**ABSTRACT**  
Accurate medical image segmentation is essential for clinical quantification, disease diagnosis, treatment planning and many other applications. Both convolution-based and transformer-based u-shaped architectures have made significant success in various medical image segmentation tasks. The former can efficiently learn local information of images while requiring much more image-specific inductive biases inherent to convolution operation. The latter can effectively capture long-range dependency at different feature scales using self-attention, whereas it typically encounters the challenges of quadratic compute and memory requirements with sequence length increasing. To address this problem, through integrating the merits of these two paradigms in a well-designed u-shaped architecture, we propose a hybrid yet effective CNN-Transformer network, named BRAU-Net++, for an accurate medical image segmentation task. Specifically, BRAU-Net++ uses bi-level routing attention as the core building block to design our u-shaped encoder-decoder structure, in which both encoder and decoder are hierarchically constructed, so as to learn global semantic information while reducing computational complexity. Furthermore, this network restructures skip connection by incorporating channel-spatial attention which adopts convolution operations, aiming to minimize local spatial information loss and amplify global dimension-interaction of multi-scale features. Extensive experiments on three public benchmark datasets demonstrate that our proposed approach surpasses other state-of-the-art methods including its baseline: BRAU-Net under almost all evaluation metrics. We achieve the average Dice-Similarity Coefficient (DSC) of 82.47, 90.10, and 92.94 on Synapse multi-organ segmentation, ISIC-2018 Challenge, and CVC-ClinicDB, as well as the mIoU of 84.01 and 88.17 on ISIC-2018 Challenge and CVC-ClinicDB, respectively.

{{</citation>}}


### (19/53) Towards Efficient and Effective Text-to-Video Retrieval with Coarse-to-Fine Visual Representation Learning (Kaibin Tian et al., 2024)

{{<citation>}}

Kaibin Tian, Yanhua Cheng, Yi Liu, Xinglin Hou, Quan Chen, Han Li. (2024)  
**Towards Efficient and Effective Text-to-Video Retrieval with Coarse-to-Fine Visual Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.00701v1)  

---


**ABSTRACT**  
In recent years, text-to-video retrieval methods based on CLIP have experienced rapid development. The primary direction of evolution is to exploit the much wider gamut of visual and textual cues to achieve alignment. Concretely, those methods with impressive performance often design a heavy fusion block for sentence (words)-video (frames) interaction, regardless of the prohibitive computation complexity. Nevertheless, these approaches are not optimal in terms of feature utilization and retrieval efficiency. To address this issue, we adopt multi-granularity visual feature learning, ensuring the model's comprehensiveness in capturing visual content features spanning from abstract to detailed levels during the training phase. To better leverage the multi-granularity features, we devise a two-stage retrieval architecture in the retrieval phase. This solution ingeniously balances the coarse and fine granularity of retrieval content. Moreover, it also strikes a harmonious equilibrium between retrieval effectiveness and efficiency. Specifically, in training phase, we design a parameter-free text-gated interaction block (TIB) for fine-grained video representation learning and embed an extra Pearson Constraint to optimize cross-modal representation learning. In retrieval phase, we use coarse-grained video representations for fast recall of top-k candidates, which are then reranked by fine-grained video representations. Extensive experiments on four benchmarks demonstrate the efficiency and effectiveness. Notably, our method achieves comparable performance with the current state-of-the-art methods while being nearly 50 times faster.

{{</citation>}}


### (20/53) Credible Teacher for Semi-Supervised Object Detection in Open Scene (Jingyu Zhuang et al., 2024)

{{<citation>}}

Jingyu Zhuang, Kuo Wang, Liang Lin, Guanbin Li. (2024)  
**Credible Teacher for Semi-Supervised Object Detection in Open Scene**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2401.00695v2)  

---


**ABSTRACT**  
Semi-Supervised Object Detection (SSOD) has achieved resounding success by leveraging unlabeled data to improve detection performance. However, in Open Scene Semi-Supervised Object Detection (O-SSOD), unlabeled data may contains unknown objects not observed in the labeled data, which will increase uncertainty in the model's predictions for known objects. It is detrimental to the current methods that mainly rely on self-training, as more uncertainty leads to the lower localization and classification precision of pseudo labels. To this end, we propose Credible Teacher, an end-to-end framework. Credible Teacher adopts an interactive teaching mechanism using flexible labels to prevent uncertain pseudo labels from misleading the model and gradually reduces its uncertainty through the guidance of other credible pseudo labels. Empirical results have demonstrated our method effectively restrains the adverse effect caused by O-SSOD and significantly outperforms existing counterparts.

{{</citation>}}


### (21/53) ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention (Chenhang He et al., 2024)

{{<citation>}}

Chenhang He, Ruihuang Li, Guowen Zhang, Lei Zhang. (2024)  
**ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2401.00912v1)  

---


**ABSTRACT**  
Window-based transformers have demonstrated strong ability in large-scale point cloud understanding by capturing context-aware representations with affordable attention computation in a more localized manner. However, because of the sparse nature of point clouds, the number of voxels per window varies significantly. Current methods partition the voxels in each window into multiple subsets of equal size, which cost expensive overhead in sorting and padding the voxels, making them run slower than sparse convolution based methods. In this paper, we present ScatterFormer, which, for the first time to our best knowledge, could directly perform attention on voxel sets with variable length. The key of ScatterFormer lies in the innovative Scatter Linear Attention (SLA) module, which leverages the linear attention mechanism to process in parallel all voxels scattered in different windows. Harnessing the hierarchical computation units of the GPU and matrix blocking algorithm, we reduce the latency of the proposed SLA module to less than 1 ms on moderate GPUs. Besides, we develop a cross-window interaction module to simultaneously enhance the local representation and allow the information flow across windows, eliminating the need for window shifting. Our proposed ScatterFormer demonstrates 73 mAP (L2) on the large-scale Waymo Open Dataset and 70.5 NDS on the NuScenes dataset, running at an outstanding detection rate of 28 FPS. Code is available at https://github.com/skyhehe123/ScatterFormer

{{</citation>}}


## cs.SE (4)



### (22/53) Leveraging Large Language Models to Boost Dafny's Developers Productivity (Álvaro Silva et al., 2024)

{{<citation>}}

Álvaro Silva, Alexandra Mendes, João F. Ferreira. (2024)  
**Leveraging Large Language Models to Boost Dafny's Developers Productivity**  

---
Primary Category: cs.SE  
Categories: cs-LO, cs-PL, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.00963v1)  

---


**ABSTRACT**  
This research idea paper proposes leveraging Large Language Models (LLMs) to enhance the productivity of Dafny developers. Although the use of verification-aware languages, such as Dafny, has increased considerably in the last decade, these are still not widely adopted. Often the cost of using such languages is too high, due to the level of expertise required from the developers and challenges that they often face when trying to prove a program correct. Even though Dafny automates a lot of the verification process, sometimes there are steps that are too complex for Dafny to perform on its own. One such case is that of missing lemmas, i.e. Dafny is unable to prove a result without being given further help in the form of a theorem that can assist it in the proof of the step.   In this paper, we describe preliminary work on a new Dafny plugin that leverages LLMs to assist developers by generating suggestions for relevant lemmas that Dafny is unable to discover and use. Moreover, for the lemmas that cannot be proved automatically, the plugin also attempts to provide accompanying calculational proofs. We also discuss ideas for future work by describing a research agenda on using LLMs to increase the adoption of verification-aware languages in general, by increasing developers productivity and by reducing the level of expertise required for crafting formal specifications and proving program properties.

{{</citation>}}


### (23/53) New Job, New Gender? Measuring the Social Bias in Image Generation Models (Wenxuan Wang et al., 2024)

{{<citation>}}

Wenxuan Wang, Haonan Bai, Jen-tse Huang, Yuxuan Wan, Youliang Yuan, Haoyi Qiu, Nanyun Peng, Michael R. Lyu. (2024)  
**New Job, New Gender? Measuring the Social Bias in Image Generation Models**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-CV, cs-MM, cs-SE, cs.SE  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.00763v1)  

---


**ABSTRACT**  
Image generation models can generate or edit images from a given text. Recent advancements in image generation technology, exemplified by DALL-E and Midjourney, have been groundbreaking. These advanced models, despite their impressive capabilities, are often trained on massive Internet datasets, making them susceptible to generating content that perpetuates social stereotypes and biases, which can lead to severe consequences. Prior research on assessing bias within image generation models suffers from several shortcomings, including limited accuracy, reliance on extensive human labor, and lack of comprehensive analysis. In this paper, we propose BiasPainter, a novel metamorphic testing framework that can accurately, automatically and comprehensively trigger social bias in image generation models. BiasPainter uses a diverse range of seed images of individuals and prompts the image generation models to edit these images using gender, race, and age-neutral queries. These queries span 62 professions, 39 activities, 57 types of objects, and 70 personality traits. The framework then compares the edited images to the original seed images, focusing on any changes related to gender, race, and age. BiasPainter adopts a testing oracle that these characteristics should not be modified when subjected to neutral prompts. Built upon this design, BiasPainter can trigger the social bias and evaluate the fairness of image generation models. To evaluate the effectiveness of BiasPainter, we use BiasPainter to test five widely-used commercial image generation software and models, such as stable diffusion and Midjourney. Experimental results show that 100\% of the generated test cases can successfully trigger social bias in image generation models.

{{</citation>}}


### (24/53) The Earth is Flat? Unveiling Factual Errors in Large Language Models (Wenxuan Wang et al., 2024)

{{<citation>}}

Wenxuan Wang, Juluan Shi, Zhaopeng Tu, Youliang Yuan, Jen-tse Huang, Wenxiang Jiao, Michael R. Lyu. (2024)  
**The Earth is Flat? Unveiling Factual Errors in Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2401.00761v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) like ChatGPT are foundational in various applications due to their extensive knowledge from pre-training and fine-tuning. Despite this, they are prone to generating factual and commonsense errors, raising concerns in critical areas like healthcare, journalism, and education to mislead users. Current methods for evaluating LLMs' veracity are limited by test data leakage or the need for extensive human labor, hindering efficient and accurate error detection. To tackle this problem, we introduce a novel, automatic testing framework, FactChecker, aimed at uncovering factual inaccuracies in LLMs. This framework involves three main steps: First, it constructs a factual knowledge graph by retrieving fact triplets from a large-scale knowledge database. Then, leveraging the knowledge graph, FactChecker employs a rule-based approach to generates three types of questions (Yes-No, Multiple-Choice, and WH questions) that involve single-hop and multi-hop relations, along with correct answers. Lastly, it assesses the LLMs' responses for accuracy using tailored matching strategies for each question type. Our extensive tests on six prominent LLMs, including text-davinci-002, text-davinci-003, ChatGPT~(gpt-3.5-turbo, gpt-4), Vicuna, and LLaMA-2, reveal that FactChecker can trigger factual errors in up to 45\% of questions in these models. Moreover, we demonstrate that FactChecker's test cases can improve LLMs' factual accuracy through in-context learning and fine-tuning (e.g., llama-2-13b-chat's accuracy increase from 35.3\% to 68.5\%). We are making all code, data, and results available for future research endeavors.

{{</citation>}}


### (25/53) A & B == B & A: Triggering Logical Reasoning Failures in Large Language Models (Yuxuan Wan et al., 2024)

{{<citation>}}

Yuxuan Wan, Wenxuan Wang, Yiliu Yang, Youliang Yuan, Jen-tse Huang, Pinjia He, Wenxiang Jiao, Michael R. Lyu. (2024)  
**A & B == B & A: Triggering Logical Reasoning Failures in Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-LO, cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.00757v1)  

---


**ABSTRACT**  
Recent advancements in large language models (LLMs) have propelled Artificial Intelligence (AI) to new heights, enabling breakthroughs in various tasks such as writing assistance, code generation, and machine translation. A significant distinction of advanced LLMs, such as ChatGPT, is their demonstrated ability to "reason." However, evaluating the reasoning ability of LLMs remains a challenge as most existing evaluations focus on their accuracy on the downstream tasks rather than directly assessing their reasoning processes. Efforts have been made to develop benchmarks and metrics to assess reasoning in LLMs, but they suffer from data leakage or limited scope. In this paper, we introduce LogicAsker, an automatic approach that comprehensively evaluates and improves the logical reasoning abilities of LLMs under a set of atomic reasoning skills based on propositional and predicate logic. The results provide insights into LLMs' reasoning abilities and reveal the logical rules the LLMs did not learn well. We evaluate LogicAsker on six widely deployed LLMs, including GPT-3, ChatGPT, GPT-4, Bard, Vicuna, and Guanaco. The results show that test cases from LogicAsker can find logical reasoning failures in different LLMs with a rate of 25\% - 94\%. In addition, the test cases of LogicAsker can be further used to design demonstration examples for in-context learning, which effectively improves the logical reasoning ability of LLMs, e.g., 10\% for GPT-4. As far as we know, our work is the first to create prompts based on testing results to improve LLMs' formal reasoning ability effectively. All the code, data, and results will be released for reproduction and future research.

{{</citation>}}


## cs.AI (1)



### (26/53) Taking the Next Step with Generative Artificial Intelligence: The Transformative Role of Multimodal Large Language Models in Science Education (Arne Bewersdorff et al., 2024)

{{<citation>}}

Arne Bewersdorff, Christian Hartmann, Marie Hornberger, Kathrin Seßler, Maria Bannert, Enkelejda Kasneci, Gjergji Kasneci, Xiaoming Zhai, Claudia Nerdel. (2024)  
**Taking the Next Step with Generative Artificial Intelligence: The Transformative Role of Multimodal Large Language Models in Science Education**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.00832v1)  

---


**ABSTRACT**  
The integration of Artificial Intelligence (AI), particularly Large Language Model (LLM)-based systems, in education has shown promise in enhancing teaching and learning experiences. However, the advent of Multimodal Large Language Models (MLLMs) like GPT-4 with vision (GPT-4V), capable of processing multimodal data including text, sound, and visual inputs, opens a new era of enriched, personalized, and interactive learning landscapes in education. Grounded in theory of multimedia learning, this paper explores the transformative role of MLLMs in central aspects of science education by presenting exemplary innovative learning scenarios. Possible applications for MLLMs could range from content creation to tailored support for learning, fostering competencies in scientific practices, and providing assessment and feedback. These scenarios are not limited to text-based and uni-modal formats but can be multimodal, increasing thus personalization, accessibility, and potential learning effectiveness. Besides many opportunities, challenges such as data protection and ethical considerations become more salient, calling for robust frameworks to ensure responsible integration. This paper underscores the necessity for a balanced approach in implementing MLLMs, where the technology complements rather than supplants the educator's role, ensuring thus an effective and ethical use of AI in science education. It calls for further research to explore the nuanced implications of MLLMs on the evolving role of educators and to extend the discourse beyond science education to other disciplines. Through the exploration of potentials, challenges, and future implications, we aim to contribute to a preliminary understanding of the transformative trajectory of MLLMs in science education and beyond.

{{</citation>}}


## cs.CL (10)



### (27/53) A Computational Framework for Behavioral Assessment of LLM Therapists (Yu Ying Chiu et al., 2024)

{{<citation>}}

Yu Ying Chiu, Ashish Sharma, Inna Wanyin Lin, Tim Althoff. (2024)  
**A Computational Framework for Behavioral Assessment of LLM Therapists**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.00820v1)  

---


**ABSTRACT**  
The emergence of ChatGPT and other large language models (LLMs) has greatly increased interest in utilizing LLMs as therapists to support individuals struggling with mental health challenges. However, due to the lack of systematic studies, our understanding of how LLM therapists behave, i.e., ways in which they respond to clients, is significantly limited. Understanding their behavior across a wide range of clients and situations is crucial to accurately assess their capabilities and limitations in the high-risk setting of mental health, where undesirable behaviors can lead to severe consequences. In this paper, we propose BOLT, a novel computational framework to study the conversational behavior of LLMs when employed as therapists. We develop an in-context learning method to quantitatively measure the behavior of LLMs based on 13 different psychotherapy techniques including reflections, questions, solutions, normalizing, and psychoeducation. Subsequently, we compare the behavior of LLM therapists against that of high- and low-quality human therapy, and study how their behavior can be modulated to better reflect behaviors observed in high-quality therapy. Our analysis of GPT and Llama-variants reveals that these LLMs often resemble behaviors more commonly exhibited in low-quality therapy rather than high-quality therapy, such as offering a higher degree of problem-solving advice when clients share emotions, which is against typical recommendations. At the same time, unlike low-quality therapy, LLMs reflect significantly more upon clients' needs and strengths. Our analysis framework suggests that despite the ability of LLMs to generate anecdotal examples that appear similar to human therapists, LLM therapists are currently not fully consistent with high-quality care, and thus require additional research to ensure quality care.

{{</citation>}}


### (28/53) If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code Empowers Large Language Models to Serve as Intelligent Agents (Ke Yang et al., 2024)

{{<citation>}}

Ke Yang, Jiateng Liu, John Wu, Chaoqi Yang, Yi R. Fung, Sha Li, Zixuan Huang, Xu Cao, Xingyao Wang, Yiquan Wang, Heng Ji, Chengxiang Zhai. (2024)  
**If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code Empowers Large Language Models to Serve as Intelligent Agents**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.00812v1)  

---


**ABSTRACT**  
The prominent large language models (LLMs) of today differ from past language models not only in size, but also in the fact that they are trained on a combination of natural language and formal language (code). As a medium between humans and computers, code translates high-level goals into executable steps, featuring standard syntax, logical consistency, abstraction, and modularity. In this survey, we present an overview of the various benefits of integrating code into LLMs' training data. Specifically, beyond enhancing LLMs in code generation, we observe that these unique properties of code help (i) unlock the reasoning ability of LLMs, enabling their applications to a range of more complex natural language tasks; (ii) steer LLMs to produce structured and precise intermediate steps, which can then be connected to external execution ends through function calls; and (iii) take advantage of code compilation and execution environment, which also provides diverse feedback for model improvement. In addition, we trace how these profound capabilities of LLMs, brought by code, have led to their emergence as intelligent agents (IAs) in situations where the ability to understand instructions, decompose goals, plan and execute actions, and refine from feedback are crucial to their success on downstream tasks. Finally, we present several key challenges and future directions of empowering LLMs with code.

{{</citation>}}


### (29/53) PerSHOP -- A Persian dataset for shopping dialogue systems modeling (Keyvan Mahmoudi et al., 2024)

{{<citation>}}

Keyvan Mahmoudi, Heshaam Faili. (2024)  
**PerSHOP -- A Persian dataset for shopping dialogue systems modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: Google, NLU  
[Paper Link](http://arxiv.org/abs/2401.00811v1)  

---


**ABSTRACT**  
Nowadays, dialogue systems are used in many fields of industry and research. There are successful instances of these systems, such as Apple Siri, Google Assistant, and IBM Watson. Task-oriented dialogue system is a category of these, that are used in specific tasks. They can perform tasks such as booking plane tickets or making restaurant reservations. Shopping is one of the most popular areas on these systems. The bot replaces the human salesperson and interacts with the customers by speaking. To train the models behind the scenes of these systems, annotated data is needed. In this paper, we developed a dataset of dialogues in the Persian language through crowd-sourcing. We annotated these dialogues to train a model. This dataset contains nearly 22k utterances in 15 different domains and 1061 dialogues. This is the largest Persian dataset in this field, which is provided freely so that future researchers can use it. Also, we proposed some baseline models for natural language understanding (NLU) tasks. These models perform two tasks for NLU: intent classification and entity extraction. The F-1 score metric obtained for intent classification is around 91% and for entity extraction is around 93%, which can be a baseline for future research.

{{</citation>}}


### (30/53) Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models (Terry Yue Zhuo et al., 2024)

{{<citation>}}

Terry Yue Zhuo, Armel Zebaze, Nitchakarn Suppattarachai, Leandro von Werra, Harm de Vries, Qian Liu, Niklas Muennighoff. (2024)  
**Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-SE, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.00788v1)  

---


**ABSTRACT**  
The high cost of full-parameter fine-tuning (FFT) of Large Language Models (LLMs) has led to a series of parameter-efficient fine-tuning (PEFT) methods. However, it remains unclear which methods provide the best cost-performance trade-off at different model scales. We introduce Astraios, a suite of 28 instruction-tuned OctoCoder models using 7 tuning methods and 4 model sizes up to 16 billion parameters. Through investigations across 5 tasks and 8 different datasets encompassing both code comprehension and code generation tasks, we find that FFT generally leads to the best downstream performance across all scales, and PEFT methods differ significantly in their efficacy based on the model scale. LoRA usually offers the most favorable trade-off between cost and performance. Further investigation into the effects of these methods on both model robustness and code security reveals that larger models tend to demonstrate reduced robustness and less security. At last, we explore the relationships among updated parameters, cross-entropy loss, and task performance. We find that the tuning effectiveness observed in small models generalizes well to larger models, and the validation loss in instruction tuning can be a reliable indicator of overall downstream performance.

{{</citation>}}


### (31/53) Temporal Validity Change Prediction (Georg Wenzel et al., 2024)

{{<citation>}}

Georg Wenzel, Adam Jatowt. (2024)  
**Temporal Validity Change Prediction**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: AI, Twitter  
[Paper Link](http://arxiv.org/abs/2401.00779v1)  

---


**ABSTRACT**  
Temporal validity is an important property of text that is useful for many downstream applications, such as recommender systems, conversational AI, or story understanding. Existing benchmarking tasks often require models to identify the temporal validity duration of a single statement. However, in many cases, additional contextual information, such as sentences in a story or posts on a social media profile, can be collected from the available text stream. This contextual information may greatly alter the duration for which a statement is expected to be valid. We propose Temporal Validity Change Prediction, a natural language processing task benchmarking the capability of machine learning models to detect contextual statements that induce such change. We create a dataset consisting of temporal target statements sourced from Twitter and crowdsource sample context statements. We then benchmark a set of transformer-based language models on our dataset. Finally, we experiment with temporal validity duration prediction as an auxiliary task to improve the performance of the state-of-the-art model.

{{</citation>}}


### (32/53) Machine Translation Testing via Syntactic Tree Pruning (Quanjun Zhang et al., 2024)

{{<citation>}}

Quanjun Zhang, Juan Zhai, Chunrong Fang, Jiawei Liu, Weisong Sun, Haichuan Hu, Qingyu Wang. (2024)  
**Machine Translation Testing via Syntactic Tree Pruning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SE, cs.CL  
Keywords: Google, Machine Translation, Microsoft, Pruning  
[Paper Link](http://arxiv.org/abs/2401.00751v1)  

---


**ABSTRACT**  
Machine translation systems have been widely adopted in our daily life, making life easier and more convenient. Unfortunately, erroneous translations may result in severe consequences, such as financial losses. This requires to improve the accuracy and the reliability of machine translation systems. However, it is challenging to test machine translation systems because of the complexity and intractability of the underlying neural models. To tackle these challenges, we propose a novel metamorphic testing approach by syntactic tree pruning (STP) to validate machine translation systems. Our key insight is that a pruned sentence should have similar crucial semantics compared with the original sentence. Specifically, STP (1) proposes a core semantics-preserving pruning strategy by basic sentence structure and dependency relations on the level of syntactic tree representation; (2) generates source sentence pairs based on the metamorphic relation; (3) reports suspicious issues whose translations break the consistency property by a bag-of-words model. We further evaluate STP on two state-of-the-art machine translation systems (i.e., Google Translate and Bing Microsoft Translator) with 1,200 source sentences as inputs. The results show that STP can accurately find 5,073 unique erroneous translations in Google Translate and 5,100 unique erroneous translations in Bing Microsoft Translator (400% more than state-of-the-art techniques), with 64.5% and 65.4% precision, respectively. The reported erroneous translations vary in types and more than 90% of them cannot be found by state-of-the-art techniques. There are 9,393 erroneous translations unique to STP, which is 711.9% more than state-of-the-art techniques. Moreover, STP is quite effective to detect translation errors for the original sentences with a recall reaching 74.0%, improving state-of-the-art techniques by 55.1% on average.

{{</citation>}}


### (33/53) ToolEyes: Fine-Grained Evaluation for Tool Learning Capabilities of Large Language Models in Real-world Scenarios (Junjie Ye et al., 2024)

{{<citation>}}

Junjie Ye, Guanyu Li, Songyang Gao, Caishuang Huang, Yilong Wu, Sixian Li, Xiaoran Fan, Shihan Dou, Qi Zhang, Tao Gui, Xuanjing Huang. (2024)  
**ToolEyes: Fine-Grained Evaluation for Tool Learning Capabilities of Large Language Models in Real-world Scenarios**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.00741v1)  

---


**ABSTRACT**  
Existing evaluations of tool learning primarily focus on validating the alignment of selected tools for large language models (LLMs) with expected outcomes. However, these approaches rely on a limited set of scenarios where answers can be pre-determined, diverging from genuine needs. Furthermore, a sole emphasis on outcomes disregards the intricate capabilities essential for LLMs to effectively utilize tools. To tackle this issue, we propose ToolEyes, a fine-grained system tailored for the evaluation of the LLMs' tool learning capabilities in authentic scenarios. The system meticulously examines seven real-world scenarios, analyzing five dimensions crucial to LLMs in tool learning: format alignment, intent comprehension, behavior planning, tool selection, and answer organization. Additionally, ToolEyes incorporates a tool library boasting approximately 600 tools, serving as an intermediary between LLMs and the physical world. Evaluations involving ten LLMs across three categories reveal a preference for specific scenarios and limited cognitive abilities in tool learning. Intriguingly, expanding the model size even exacerbates the hindrance to tool learning. These findings offer instructive insights aimed at advancing the field of tool learning. The data is available att https://github.com/Junjie-Ye/ToolEyes.git.

{{</citation>}}


### (34/53) Large Language Models aren't all that you need (Kiran Voderhobli Holla et al., 2024)

{{<citation>}}

Kiran Voderhobli Holla, Chaithanya Kumar, Aryan Singh. (2024)  
**Large Language Models aren't all that you need**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model, Multilingual, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2401.00698v1)  

---


**ABSTRACT**  
This paper describes the architecture and systems built towards solving the SemEval 2023 Task 2: MultiCoNER II (Multilingual Complex Named Entity Recognition) [1]. We evaluate two approaches (a) a traditional Conditional Random Fields model and (b) a Large Language Model (LLM) fine-tuned with a customized head and compare the two approaches. The novel ideas explored are: 1) Decaying auxiliary loss (with residual) - where we train the model on an auxiliary task of Coarse-Grained NER and include this task as a part of the loss function 2) Triplet token blending - where we explore ways of blending the embeddings of neighboring tokens in the final NER layer prior to prediction 3) Task-optimal heads - where we explore a variety of custom heads and learning rates for the final layer of the LLM. We also explore multiple LLMs including GPT-3 and experiment with a variety of dropout and other hyperparameter settings before arriving at our final model which achieves micro & macro f1 of 0.85/0.84 (on dev) and 0.67/0.61 on the test data . We show that while pre-trained LLMs, by themselves, bring about a large improvement in scores as compared to traditional models, we also demonstrate that tangible improvements to the Macro-F1 score can be made by augmenting the LLM with additional feature/loss/model engineering techniques described above.

{{</citation>}}


### (35/53) Benchmarking Large Language Models on Controllable Generation under Diversified Instructions (Yihan Chen et al., 2024)

{{<citation>}}

Yihan Chen, Benfeng Xu, Quan Wang, Yi Liu, Zhendong Mao. (2024)  
**Benchmarking Large Language Models on Controllable Generation under Diversified Instructions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.00690v1)  

---


**ABSTRACT**  
While large language models (LLMs) have exhibited impressive instruction-following capabilities, it is still unclear whether and to what extent they can respond to explicit constraints that might be entailed in various instructions. As a significant aspect of LLM alignment, it is thus important to formulate such a specialized set of instructions as well as investigate the resulting behavior of LLMs. To address this vacancy, we propose a new benchmark CoDI-Eval to systematically and comprehensively evaluate LLMs' responses to instructions with various constraints. We construct a large collection of constraints-attributed instructions as a test suite focused on both generalization and coverage. Specifically, we advocate an instruction diversification process to synthesize diverse forms of constraint expression and also deliberate the candidate task taxonomy with even finer-grained sub-categories. Finally, we automate the entire evaluation process to facilitate further developments. Different from existing studies on controllable text generation, CoDI-Eval extends the scope to the prevalent instruction-following paradigm for the first time. We provide extensive evaluations of representative LLMs (e.g., ChatGPT, Vicuna) on CoDI-Eval, revealing their limitations in following instructions with specific constraints and there is still a significant gap between open-source and commercial closed-source LLMs. We believe this benchmark will facilitate research into improving the controllability of LLMs' responses to instructions. Our data and code are available at https://github.com/Xt-cyh/CoDI-Eval.

{{</citation>}}


### (36/53) Predicting Anti-microbial Resistance using Large Language Models (Hyunwoo Yoo et al., 2024)

{{<citation>}}

Hyunwoo Yoo, Bahrad Sokhansanj, James R. Brown, Gail Rosen. (2024)  
**Predicting Anti-microbial Resistance using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.00642v1)  

---


**ABSTRACT**  
During times of increasing antibiotic resistance and the spread of infectious diseases like COVID-19, it is important to classify genes related to antibiotic resistance. As natural language processing has advanced with transformer-based language models, many language models that learn characteristics of nucleotide sequences have also emerged. These models show good performance in classifying various features of nucleotide sequences. When classifying nucleotide sequences, not only the sequence itself, but also various background knowledge is utilized. In this study, we use not only a nucleotide sequence-based language model but also a text language model based on PubMed articles to reflect more biological background knowledge in the model. We propose a method to fine-tune the nucleotide sequence language model and the text language model based on various databases of antibiotic resistance genes. We also propose an LLM-based augmentation technique to supplement the data and an ensemble method to effectively combine the two models. We also propose a benchmark for evaluating the model. Our method achieved better performance than the nucleotide sequence language model in the drug resistance class prediction.

{{</citation>}}


## cs.HC (1)



### (37/53) Agricultural 4.0 Leveraging on Technological Solutions: Study for Smart Farming Sector (Emmanuel Kojo Gyamfi et al., 2024)

{{<citation>}}

Emmanuel Kojo Gyamfi, Zag ElSayed, Jess Kropczynski, Mustapha Awinsongya Yakubu, Nelly Elsayed. (2024)  
**Agricultural 4.0 Leveraging on Technological Solutions: Study for Smart Farming Sector**  

---
Primary Category: cs.HC  
Categories: cs-CY, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.00814v1)  

---


**ABSTRACT**  
By 2050, it is predicted that there will be 9 billion people on the planet, which will call for more production, lower costs, and the preservation of natural resources. It is anticipated that atypical occurrences and climate change will pose severe risks to agricultural output. It follows that a 70% or more significant rise in food output is anticipated. Smart farming, often known as agriculture 4.0, is a tech-driven revolution in agriculture with the goal of raising industry production and efficiency. Four primary trends are responsible for it: food waste, climate change, population shifts, and resource scarcity. The agriculture industry is changing as a result of the adoption of emerging technologies. Using cutting-edge technology like IoT, AI, and other sensors, smart farming transforms traditional production methods and international agricultural policies. The objective is to establish a value chain that is optimized to facilitate enhanced monitoring and decreased labor expenses. The agricultural sector has seen tremendous transformation as a result of the fourth industrial revolution, which has combined traditional farming methods with cutting-edge technology to increase productivity, sustainability, and efficiency. To effectively utilize the potential of technology gadgets in the agriculture sector, collaboration between governments, private sector entities, and other stakeholders is necessary. This paper covers Agriculture 4.0, looks at its possible benefits and drawbacks of the implementation methodologies, compatibility, reliability, and investigates the several digital tools that are being utilized to change the agriculture industry and how to mitigate the challenges.

{{</citation>}}


## cs.CR (3)



### (38/53) Privacy-Preserving Data in IoT-based Cloud Systems: A Comprehensive Survey with AI Integration (D. Dhinakaran et al., 2024)

{{<citation>}}

D. Dhinakaran, S. M. Udhaya Sankar, D. Selvaraj, S. Edwin Raja. (2024)  
**Privacy-Preserving Data in IoT-based Cloud Systems: A Comprehensive Survey with AI Integration**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.00794v1)  

---


**ABSTRACT**  
As the integration of Internet of Things devices with cloud computing proliferates, the paramount importance of privacy preservation comes to the forefront. This survey paper meticulously explores the landscape of privacy issues in the dynamic intersection of IoT and cloud systems. The comprehensive literature review synthesizes existing research, illuminating key challenges and discerning emerging trends in privacy preserving techniques. The categorization of diverse approaches unveils a nuanced understanding of encryption techniques, anonymization strategies, access control mechanisms, and the burgeoning integration of artificial intelligence. Notable trends include the infusion of machine learning for dynamic anonymization, homomorphic encryption for secure computation, and AI-driven access control systems. The culmination of this survey contributes a holistic view, laying the groundwork for understanding the multifaceted strategies employed in securing sensitive data within IoT-based cloud environments. The insights garnered from this survey provide a valuable resource for researchers, practitioners, and policymakers navigating the complex terrain of privacy preservation in the evolving landscape of IoT and cloud computing

{{</citation>}}


### (39/53) Digger: Detecting Copyright Content Mis-usage in Large Language Model Training (Haodong Li et al., 2024)

{{<citation>}}

Haodong Li, Gelei Deng, Yi Liu, Kailong Wang, Yuekang Li, Tianwei Zhang, Yang Liu, Guoai Xu, Guosheng Xu, Haoyu Wang. (2024)  
**Digger: Detecting Copyright Content Mis-usage in Large Language Model Training**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs-LG, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.00676v1)  

---


**ABSTRACT**  
Pre-training, which utilizes extensive and varied datasets, is a critical factor in the success of Large Language Models (LLMs) across numerous applications. However, the detailed makeup of these datasets is often not disclosed, leading to concerns about data security and potential misuse. This is particularly relevant when copyrighted material, still under legal protection, is used inappropriately, either intentionally or unintentionally, infringing on the rights of the authors.   In this paper, we introduce a detailed framework designed to detect and assess the presence of content from potentially copyrighted books within the training datasets of LLMs. This framework also provides a confidence estimation for the likelihood of each content sample's inclusion. To validate our approach, we conduct a series of simulated experiments, the results of which affirm the framework's effectiveness in identifying and addressing instances of content misuse in LLM training processes. Furthermore, we investigate the presence of recognizable quotes from famous literary works within these datasets. The outcomes of our study have significant implications for ensuring the ethical use of copyrighted materials in the development of LLMs, highlighting the need for more transparent and responsible data management practices in this field.

{{</citation>}}


### (40/53) TBDD: A New Trust-based, DRL-driven Framework for Blockchain Sharding in IoT (Zixu Zhang et al., 2024)

{{<citation>}}

Zixu Zhang, Guangsheng Yu, Caijun Sun, Xu Wang, Ying Wang, Ming Zhang, Wei Ni, Ren Ping Liu, Andrew Reeves, Nektarios Georgalas. (2024)  
**TBDD: A New Trust-based, DRL-driven Framework for Blockchain Sharding in IoT**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.00632v1)  

---


**ABSTRACT**  
Integrating sharded blockchain with IoT presents a solution for trust issues and optimized data flow. Sharding boosts blockchain scalability by dividing its nodes into parallel shards, yet it's vulnerable to the $1\%$ attacks where dishonest nodes target a shard to corrupt the entire blockchain. Balancing security with scalability is pivotal for such systems. Deep Reinforcement Learning (DRL) adeptly handles dynamic, complex systems and multi-dimensional optimization. This paper introduces a Trust-based and DRL-driven (\textsc{TbDd}) framework, crafted to counter shard collusion risks and dynamically adjust node allocation, enhancing throughput while maintaining network security. With a comprehensive trust evaluation mechanism, \textsc{TbDd} discerns node types and performs targeted resharding against potential threats. The model maximizes tolerance for dishonest nodes, optimizes node movement frequency, ensures even node distribution in shards, and balances sharding risks. Rigorous evaluations prove \textsc{TbDd}'s superiority over conventional random-, community-, and trust-based sharding methods in shard risk equilibrium and reducing cross-shard transactions.

{{</citation>}}


## cs.RO (1)



### (41/53) Edge Computing based Human-Robot Cognitive Fusion: A Medical Case Study in the Autism Spectrum Disorder Therapy (Qin Yang, 2024)

{{<citation>}}

Qin Yang. (2024)  
**Edge Computing based Human-Robot Cognitive Fusion: A Medical Case Study in the Autism Spectrum Disorder Therapy**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-DC, cs-LG, cs-MA, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.00776v1)  

---


**ABSTRACT**  
In recent years, edge computing has served as a paradigm that enables many future technologies like AI, Robotics, IoT, and high-speed wireless sensor networks (like 5G) by connecting cloud computing facilities and services to the end users. Especially in medical and healthcare applications, it provides remote patient monitoring and increases voluminous multimedia. From the robotics angle, robot-assisted therapy (RAT) is an active-assistive robotic technology in rehabilitation robotics, attracting many researchers to study and benefit people with disability like autism spectrum disorder (ASD) children. However, the main challenge of RAT is that the model capable of detecting the affective states of ASD people exists and can recall individual preferences. Moreover, involving expert diagnosis and recommendations to guide robots in updating the therapy approach to adapt to different statuses and scenarios is a crucial part of the ASD therapy process. This paper proposes the architecture of edge cognitive computing by combining human experts and assisted robots collaborating in the same framework to help ASD patients with long-term support. By integrating the real-time computing and analysis of a new cognitive robotic model for ASD therapy, the proposed architecture can achieve a seamless remote diagnosis, round-the-clock symptom monitoring, emergency warning, therapy alteration, and advanced assistance.

{{</citation>}}


## cs.SI (2)



### (42/53) Strong Transitivity Relations and Graph Neural Networks (Yassin Mohamadi et al., 2024)

{{<citation>}}

Yassin Mohamadi, Mostafa Haghir Chehreghani. (2024)  
**Strong Transitivity Relations and Graph Neural Networks**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-LG, cs-SI, cs.SI  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.01384v1)  

---


**ABSTRACT**  
Local neighborhoods play a crucial role in embedding generation in graph-based learning. It is commonly believed that nodes ought to have embeddings that resemble those of their neighbors. In this research, we try to carefully expand the concept of similarity from nearby neighborhoods to the entire graph. We provide an extension of similarity that is based on transitivity relations, which enables Graph Neural Networks (GNNs) to capture both global similarities and local similarities over the whole graph. We introduce Transitivity Graph Neural Network (TransGNN), which more than local node similarities, takes into account global similarities by distinguishing strong transitivity relations from weak ones and exploiting them. We evaluate our model over several real-world datasets and showed that it considerably improves the performance of several well-known GNN models, for tasks such as node classification.

{{</citation>}}


### (43/53) IRWE: Inductive Random Walk for Joint Inference of Identity and Position Network Embedding (Meng Qin et al., 2024)

{{<citation>}}

Meng Qin, Dit-Yan Yeung. (2024)  
**IRWE: Inductive Random Walk for Joint Inference of Identity and Position Network Embedding**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.00651v1)  

---


**ABSTRACT**  
Network embedding, which maps graphs to distributed representations, is a unified framework for various graph inference tasks. According to the topology properties (e.g., structural roles and community memberships of nodes) to be preserved, it can be categorized into the identity and position embedding. However, existing methods can only capture one type of property. Some approaches can support the inductive inference that generalizes the embedding model to new nodes or graphs but relies on the availability of attributes. Due to the complicated correlations between topology and attributes, it is unclear for some inductive methods which type of property they can capture. In this study, we explore a unified framework for the joint inductive inference of identity and position embeddings without attributes. An inductive random walk embedding (IRWE) method is proposed, which combines multiple attention units to handle the random walk on graph topology and simultaneously derives identity and position embeddings that are jointly optimized. In particular, we demonstrate that some random walk statistics can be informative features to characterize node identities and positions while supporting the inductive embedding inference. Experiments validate the superior performance of IRWE beyond various baselines for the transductive and inductive inference of identity and position embeddings.

{{</citation>}}


## cs.GR (1)



### (44/53) Free-form Shape Modeling in XR: A Systematic Review (Shounak Chatterjee, 2024)

{{<citation>}}

Shounak Chatterjee. (2024)  
**Free-form Shape Modeling in XR: A Systematic Review**  

---
Primary Category: cs.GR  
Categories: cs-GR, cs.GR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.00924v1)  

---


**ABSTRACT**  
Shape modeling research in Computer Graphics has been an active area for decades. The ability to create and edit complex 3D shapes has been of key importance in Computer-Aided Design, Animation, Architecture, and Entertainment. With the growing popularity of Virtual and Augmented Reality, new applications and tools have been developed for artistic content creation; real-time interactive shape modeling has become increasingly important for a continuum of virtual and augmented reality environments (eXtended Reality (XR)). Shape modeling in XR opens new possibilities for intuitive design and shape modeling in an accessible way. Artificial Intelligence (AI) approaches generating shape information from text prompts are set to change how artists create and edit 3D models. There has been a substantial body of research on interactive 3D shape modeling. However, there is no recent extensive review of the existing techniques and what AI shape generation means for shape modeling in interactive XR environments. In this state-of-the-art paper, we fill this research gap in the literature by surveying free-form shape modeling work in XR, with a focus on sculpting and 3D sketching, the most intuitive forms of free-form shape modeling. We classify and discuss these works across five dimensions: contribution of the articles, domain setting, interaction tool, auto-completion, and collaborative designing. The paper concludes by discussing the disconnect between interactive 3D sculpting and sketching and how this will likely evolve with the prevalence of AI shape-generation tools in the future.

{{</citation>}}


## physics.comp-ph (1)



### (45/53) Harmonizing Covariance and Expressiveness for Deep Hamiltonian Regression in Crystalline Material Research: a Hybrid Cascaded Regression Framework (Shi Yin et al., 2024)

{{<citation>}}

Shi Yin, Xudong Zhu, Tianyu Gao, Haochong Zhang, Feng Wu, Lixin He. (2024)  
**Harmonizing Covariance and Expressiveness for Deep Hamiltonian Regression in Crystalline Material Research: a Hybrid Cascaded Regression Framework**  

---
Primary Category: physics.comp-ph  
Categories: cond-mat-mtrl-sci, cs-LG, physics-comp-ph, physics.comp-ph  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.00744v3)  

---


**ABSTRACT**  
Deep learning for Hamiltonian regression of quantum systems in material research necessitates satisfying the covariance laws, among which achieving SO(3)-equivariance without sacrificing the expressiveness of networks remains an elusive challenge due to the restriction to non-linear mappings on guaranteeing theoretical equivariance. To alleviate the covariance-expressiveness dilemma, we propose a hybrid framework with two cascaded regression stages. The first stage, with a theoretically-guaranteed covariant neural network modeling symmetry properties of 3D atom systems, yields theoretically covariant features and baseline Hamiltonian predictions, assisting the second stage in learning covariance. Meanwhile, the second stage, powered by a non-linear 3D graph Transformer network we propose for structural modeling of 3D atomic systems, refines the first stage's output as a fine-grained prediction of Hamiltonians with better expressiveness capability. The combination of a theoretically covariant yet inevitably less expressive model with a highly expressive non-linear network enables precise, generalizable predictions while maintaining robust covariance under coordinate transformations. Our method achieves state-of-the-art performance in Hamiltonian prediction for electronic structure calculations, confirmed through experiments on five crystalline material databases.

{{</citation>}}


## eess.IV (3)



### (46/53) Beyond Subspace Isolation: Many-to-Many Transformer for Light Field Image Super-resolution (Zeke Zexi Hu et al., 2024)

{{<citation>}}

Zeke Zexi Hu, Xiaoming Chen, Vera Yuk Ying Chung, Yiran Shen. (2024)  
**Beyond Subspace Isolation: Many-to-Many Transformer for Light Field Image Super-resolution**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.00740v1)  

---


**ABSTRACT**  
The effective extraction of spatial-angular features plays a crucial role in light field image super-resolution (LFSR) tasks, and the introduction of convolution and Transformers leads to significant improvement in this area. Nevertheless, due to the large 4D data volume of light field images, many existing methods opted to decompose the data into a number of lower-dimensional subspaces and perform Transformers in each sub-space individually. As a side effect, these methods inadvertently restrict the self-attention mechanisms to a One-to-One scheme accessing only a limited subset of LF data, explicitly preventing comprehensive optimization on all spatial and angular cues. In this paper, we identify this limitation as subspace isolation and introduce a novel Many-to-Many Transformer (M2MT) to address it. M2MT aggregates angular information in the spatial subspace before performing the self-attention mechanism. It enables complete access to all information across all sub-aperture images (SAIs) in a light field image. Consequently, M2MT is enabled to comprehensively capture long-range correlation dependencies. With M2MT as the pivotal component, we develop a simple yet effective M2MT network for LFSR. Our experimental results demonstrate that M2MT achieves state-of-the-art performance across various public datasets. We further conduct in-depth analysis using local attribution maps (LAM) to obtain visual interpretability, and the results validate that M2MT is empowered with a truly non-local context in both spatial and angular subspaces to mitigate subspace isolation and acquire effective spatial-angular representation.

{{</citation>}}


### (47/53) MultiFusionNet: Multilayer Multimodal Fusion of Deep Neural Networks for Chest X-Ray Image Classification (Saurabh Agarwal et al., 2024)

{{<citation>}}

Saurabh Agarwal, K. V. Arya, Yogesh Kumar Meena. (2024)  
**MultiFusionNet: Multilayer Multimodal Fusion of Deep Neural Networks for Chest X-Ray Image Classification**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2401.00728v1)  

---


**ABSTRACT**  
Chest X-ray imaging is a critical diagnostic tool for identifying pulmonary diseases. However, manual interpretation of these images is time-consuming and error-prone. Automated systems utilizing convolutional neural networks (CNNs) have shown promise in improving the accuracy and efficiency of chest X-ray image classification. While previous work has mainly focused on using feature maps from the final convolution layer, there is a need to explore the benefits of leveraging additional layers for improved disease classification. Extracting robust features from limited medical image datasets remains a critical challenge. In this paper, we propose a novel deep learning-based multilayer multimodal fusion model that emphasizes extracting features from different layers and fusing them. Our disease detection model considers the discriminatory information captured by each layer. Furthermore, we propose the fusion of different-sized feature maps (FDSFM) module to effectively merge feature maps from diverse layers. The proposed model achieves a significantly higher accuracy of 97.21% and 99.60% for both three-class and two-class classifications, respectively. The proposed multilayer multimodal fusion model, along with the FDSFM module, holds promise for accurate disease classification and can also be extended to other disease classifications in chest X-ray images.

{{</citation>}}


### (48/53) Self-supervised learning for skin cancer diagnosis with limited training data (Hamish Haggerty et al., 2024)

{{<citation>}}

Hamish Haggerty, Rohitash Chandra. (2024)  
**Self-supervised learning for skin cancer diagnosis with limited training data**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.00692v1)  

---


**ABSTRACT**  
Cancer diagnosis is a well-studied problem in machine learning since early detection of cancer is often the determining factor in prognosis. Supervised deep learning achieves excellent results in cancer image classification, usually through transfer learning. However, these models require large amounts of labelled data and for several types of cancer, large labelled datasets do not exist. In this paper, we demonstrate that a model pre-trained using a self-supervised learning algorithm known as Barlow Twins can outperform the conventional supervised transfer learning pipeline. We juxtapose two base models: i) pretrained in a supervised fashion on ImageNet; ii) pretrained in a self-supervised fashion on ImageNet. Both are subsequently fine tuned on a small labelled skin lesion dataset and evaluated on a large test set. We achieve a mean test accuracy of 70\% for self-supervised transfer in comparison to 66\% for supervised transfer. Interestingly, boosting performance further is possible by self-supervised pretraining a second time (on unlabelled skin lesion images) before subsequent fine tuning. This hints at an alternative path to collecting more labelled data in settings where this is challenging - namely just collecting more unlabelled images. Our framework is applicable to cancer image classification models in the low-labelled data regime.

{{</citation>}}


## q-bio.NC (1)



### (49/53) Predicting Infant Brain Connectivity with Federated Multi-Trajectory GNNs using Scarce Data (Michalis Pistos et al., 2024)

{{<citation>}}

Michalis Pistos, Islem Rekik. (2024)  
**Predicting Infant Brain Connectivity with Federated Multi-Trajectory GNNs using Scarce Data**  

---
Primary Category: q-bio.NC  
Categories: cs-AI, cs-CV, cs-LG, q-bio-NC, q-bio.NC  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.01383v1)  

---


**ABSTRACT**  
The understanding of the convoluted evolution of infant brain networks during the first postnatal year is pivotal for identifying the dynamics of early brain connectivity development. Existing deep learning solutions suffer from three major limitations. First, they cannot generalize to multi-trajectory prediction tasks, where each graph trajectory corresponds to a particular imaging modality or connectivity type (e.g., T1-w MRI). Second, existing models require extensive training datasets to achieve satisfactory performance which are often challenging to obtain. Third, they do not efficiently utilize incomplete time series data. To address these limitations, we introduce FedGmTE-Net++, a federated graph-based multi-trajectory evolution network. Using the power of federation, we aggregate local learnings among diverse hospitals with limited datasets. As a result, we enhance the performance of each hospital's local generative model, while preserving data privacy. The three key innovations of FedGmTE-Net++ are: (i) presenting the first federated learning framework specifically designed for brain multi-trajectory evolution prediction in a data-scarce environment, (ii) incorporating an auxiliary regularizer in the local objective function to exploit all the longitudinal brain connectivity within the evolution trajectory and maximize data utilization, (iii) introducing a two-step imputation process, comprising a preliminary KNN-based precompletion followed by an imputation refinement step that employs regressors to improve similarity scores and refine imputations. Our comprehensive experimental results showed the outperformance of FedGmTE-Net++ in brain multi-trajectory prediction from a single baseline graph in comparison with benchmark methods.

{{</citation>}}


## math.DS (1)



### (50/53) Data Assimilation in Chaotic Systems Using Deep Reinforcement Learning (Mohamad Abed El Rahman Hammoud et al., 2024)

{{<citation>}}

Mohamad Abed El Rahman Hammoud, Naila Raboudi, Edriss S. Titi, Omar Knio, Ibrahim Hoteit. (2024)  
**Data Assimilation in Chaotic Systems Using Deep Reinforcement Learning**  

---
Primary Category: math.DS  
Categories: cs-AI, cs-LG, math-DS, math.DS, physics-ao-ph  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.00916v1)  

---


**ABSTRACT**  
Data assimilation (DA) plays a pivotal role in diverse applications, ranging from climate predictions and weather forecasts to trajectory planning for autonomous vehicles. A prime example is the widely used ensemble Kalman filter (EnKF), which relies on linear updates to minimize variance among the ensemble of forecast states. Recent advancements have seen the emergence of deep learning approaches in this domain, primarily within a supervised learning framework. However, the adaptability of such models to untrained scenarios remains a challenge. In this study, we introduce a novel DA strategy that utilizes reinforcement learning (RL) to apply state corrections using full or partial observations of the state variables. Our investigation focuses on demonstrating this approach to the chaotic Lorenz '63 system, where the agent's objective is to minimize the root-mean-squared error between the observations and corresponding forecast states. Consequently, the agent develops a correction strategy, enhancing model forecasts based on available system state observations. Our strategy employs a stochastic action policy, enabling a Monte Carlo-based DA framework that relies on randomly sampling the policy to generate an ensemble of assimilated realizations. Results demonstrate that the developed RL algorithm performs favorably when compared to the EnKF. Additionally, we illustrate the agent's capability to assimilate non-Gaussian data, addressing a significant limitation of the EnKF.

{{</citation>}}


## eess.SP (1)



### (51/53) The Smooth Trajectory Estimator for LMB Filters (Hoa Van Nguyen et al., 2024)

{{<citation>}}

Hoa Van Nguyen, Tran Thien Dat Nguyen, Changbeom Shim, Marzhar Anuar. (2024)  
**The Smooth Trajectory Estimator for LMB Filters**  

---
Primary Category: eess.SP  
Categories: cs-SY, eess-SP, eess-SY, eess.SP  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2401.00682v1)  

---


**ABSTRACT**  
This paper proposes a smooth-trajectory estimator for the labelled multi-Bernoulli (LMB) filter by exploiting the special structure of the generalised labelled multi-Bernoulli (GLMB) filter. We devise a simple and intuitive approach to store the best association map when approximating the GLMB random finite set (RFS) to the LMB RFS. In particular, we construct a smooth-trajectory estimator (i.e., an estimator over the entire trajectories of labelled estimates) for the LMB filter based on the history of the best association map and all of the measurements up to the current time. Experimental results under two challenging scenarios demonstrate significant tracking accuracy improvements with negligible additional computational time compared to the conventional LMB filter. The source code is publicly available at https://tinyurl.com/ste-lmb, aimed at promoting advancements in MOT algorithms.

{{</citation>}}


## cs.SD (1)



### (52/53) Enhancing Pre-trained ASR System Fine-tuning for Dysarthric Speech Recognition using Adversarial Data Augmentation (Huimeng Wang et al., 2024)

{{<citation>}}

Huimeng Wang, Zengrui Jin, Mengzhe Geng, Shujie Hu, Guinan Li, Tianzi Wang, Haoning Xu, Xunying Liu. (2024)  
**Enhancing Pre-trained ASR System Fine-tuning for Dysarthric Speech Recognition using Adversarial Data Augmentation**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Augmentation, BERT, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.00662v1)  

---


**ABSTRACT**  
Automatic recognition of dysarthric speech remains a highly challenging task to date. Neuro-motor conditions and co-occurring physical disabilities create difficulty in large-scale data collection for ASR system development. Adapting SSL pre-trained ASR models to limited dysarthric speech via data-intensive parameter fine-tuning leads to poor generalization. To this end, this paper presents an extensive comparative study of various data augmentation approaches to improve the robustness of pre-trained ASR model fine-tuning to dysarthric speech. These include: a) conventional speaker-independent perturbation of impaired speech; b) speaker-dependent speed perturbation, or GAN-based adversarial perturbation of normal, control speech based on their time alignment against parallel dysarthric speech; c) novel Spectral basis GAN-based adversarial data augmentation operating on non-parallel data. Experiments conducted on the UASpeech corpus suggest GAN-based data augmentation consistently outperforms fine-tuned Wav2vec2.0 and HuBERT models using no data augmentation and speed perturbation across different data expansion operating points by statistically significant word error rate (WER) reductions up to 2.01% and 0.96% absolute (9.03% and 4.63% relative) respectively on the UASpeech test set of 16 dysarthric speakers. After cross-system outputs rescoring, the best system produced the lowest published WER of 16.53% (46.47% on very low intelligibility) on UASpeech.

{{</citation>}}


## cs.NI (1)



### (53/53) Coordinated Deep Neural Networks: A Versatile Edge Offloading Algorithm (Alireza Maleki et al., 2024)

{{<citation>}}

Alireza Maleki, Hamed Shah-Mansouri, Babak H. Khalaj. (2024)  
**Coordinated Deep Neural Networks: A Versatile Edge Offloading Algorithm**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-NI, cs.NI, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.00631v1)  

---


**ABSTRACT**  
As artificial intelligence (AI) applications continue to expand, there is a growing need for deep neural network (DNN) models. Although DNN models deployed at the edge are promising to provide AI as a service with low latency, their cooperation is yet to be explored. In this paper, we consider the DNN service providers share their computing resources as well as their models' parameters and allow other DNNs to offload their computations without mirroring. We propose a novel algorithm called coordinated DNNs on edge (\textbf{CoDE}) that facilitates coordination among DNN services by creating multi-task DNNs out of individual models. CoDE aims to find the optimal path that results in the lowest possible cost, where the cost reflects the inference delay, model accuracy, and local computation workload. With CoDE, DNN models can make new paths for inference by using their own or other models' parameters. We then evaluate the performance of CoDE through numerical experiments. The results demonstrate a $75\%$ reduction in the local service computation workload while degrading the accuracy by only $2\%$ and having the same inference time in a balanced load condition. Under heavy load, CoDE can further decrease the inference time by $30\%$ while the accuracy is reduced by only $4\%$.

{{</citation>}}
