---
draft: false
title: "arXiv @ 2023.08.21"
date: 2023-08-21
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.21"
    identifier: arxiv_20230821
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.MA (1)](#csma-1)
- [cs.CV (17)](#cscv-17)
- [cs.MM (1)](#csmm-1)
- [cs.LG (15)](#cslg-15)
- [cs.CL (9)](#cscl-9)
- [cs.DC (1)](#csdc-1)
- [cs.SI (2)](#cssi-2)
- [cs.CR (2)](#cscr-2)
- [cs.IR (5)](#csir-5)
- [cs.RO (3)](#csro-3)
- [cs.SE (4)](#csse-4)
- [cs.CY (1)](#cscy-1)
- [cs.SD (1)](#cssd-1)
- [cs.IT (1)](#csit-1)
- [cs.HC (3)](#cshc-3)
- [cs.PL (1)](#cspl-1)
- [eess.SY (1)](#eesssy-1)

## cs.MA (1)



### (1/68) Intelligent Communication Planning for Constrained Environmental IoT Sensing with Reinforcement Learning (Yi Hu et al., 2023)

{{<citation>}}

Yi Hu, Jinhang Zuo, Bob Iannucci, Carlee Joe-Wong. (2023)  
**Intelligent Communication Planning for Constrained Environmental IoT Sensing with Reinforcement Learning**  

---
Primary Category: cs.MA  
Categories: cs-LG, cs-MA, cs-SY, cs.MA, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.10124v1)  

---


**ABSTRACT**  
Internet of Things (IoT) technologies have enabled numerous data-driven mobile applications and have the potential to significantly improve environmental monitoring and hazard warnings through the deployment of a network of IoT sensors. However, these IoT devices are often power-constrained and utilize wireless communication schemes with limited bandwidth. Such power constraints limit the amount of information each device can share across the network, while bandwidth limitations hinder sensors' coordination of their transmissions. In this work, we formulate the communication planning problem of IoT sensors that track the state of the environment. We seek to optimize sensors' decisions in collecting environmental data under stringent resource constraints. We propose a multi-agent reinforcement learning (MARL) method to find the optimal communication policies for each sensor that maximize the tracking accuracy subject to the power and bandwidth limitations. MARL learns and exploits the spatial-temporal correlation of the environmental data at each sensor's location to reduce the redundant reports from the sensors. Experiments on wildfire spread with LoRA wireless network simulators show that our MARL method can learn to balance the need to collect enough data to predict wildfire spread with unknown bandwidth limitations.

{{</citation>}}


## cs.CV (17)



### (2/68) HollowNeRF: Pruning Hashgrid-Based NeRFs with Trainable Collision Mitigation (Xiufeng Xie et al., 2023)

{{<citation>}}

Xiufeng Xie, Riccardo Gherardi, Zhihong Pan, Stephen Huang. (2023)  
**HollowNeRF: Pruning Hashgrid-Based NeRFs with Trainable Collision Mitigation**  

---
Primary Category: cs.CV  
Categories: I-4-5, cs-CV, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2308.10122v1)  

---


**ABSTRACT**  
Neural radiance fields (NeRF) have garnered significant attention, with recent works such as Instant-NGP accelerating NeRF training and evaluation through a combination of hashgrid-based positional encoding and neural networks. However, effectively leveraging the spatial sparsity of 3D scenes remains a challenge. To cull away unnecessary regions of the feature grid, existing solutions rely on prior knowledge of object shape or periodically estimate object shape during training by repeated model evaluations, which are costly and wasteful.   To address this issue, we propose HollowNeRF, a novel compression solution for hashgrid-based NeRF which automatically sparsifies the feature grid during the training phase. Instead of directly compressing dense features, HollowNeRF trains a coarse 3D saliency mask that guides efficient feature pruning, and employs an alternating direction method of multipliers (ADMM) pruner to sparsify the 3D saliency mask during training. By exploiting the sparsity in the 3D scene to redistribute hash collisions, HollowNeRF improves rendering quality while using a fraction of the parameters of comparable state-of-the-art solutions, leading to a better cost-accuracy trade-off. Our method delivers comparable rendering quality to Instant-NGP, while utilizing just 31% of the parameters. In addition, our solution can achieve a PSNR accuracy gain of up to 1dB using only 56% of the parameters.

{{</citation>}}


### (3/68) ASPIRE: Language-Guided Augmentation for Robust Image Classification (Sreyan Ghosh et al., 2023)

{{<citation>}}

Sreyan Ghosh, Chandra Kiran Reddy Evuru, Sonal Kumar, Utkarsh Tyagi, Sakshi Singh, Sanjoy Chowdhury, Dinesh Manocha. (2023)  
**ASPIRE: Language-Guided Augmentation for Robust Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Augmentation, Image Classification, ImageNet  
[Paper Link](http://arxiv.org/abs/2308.10103v1)  

---


**ABSTRACT**  
Neural image classifiers can often learn to make predictions by overly relying on non-predictive features that are spuriously correlated with the class labels in the training data. This leads to poor performance in real-world atypical scenarios where such features are absent. Supplementing the training dataset with images without such spurious features can aid robust learning against spurious correlations via better generalization. This paper presents ASPIRE (Language-guided data Augmentation for SPurIous correlation REmoval), a simple yet effective solution for expanding the training dataset with synthetic images without spurious features. ASPIRE, guided by language, generates these images without requiring any form of additional supervision or existing examples. Precisely, we employ LLMs to first extract foreground and background features from textual descriptions of an image, followed by advanced language-guided image editing to discover the features that are spuriously correlated with the class label. Finally, we personalize a text-to-image generation model to generate diverse in-domain images without spurious features. We demonstrate the effectiveness of ASPIRE on 4 datasets, including the very challenging Hard ImageNet dataset, and 9 baselines and show that ASPIRE improves the classification accuracy of prior methods by 1% - 38%. Code soon at: https://github.com/Sreyan88/ASPIRE.

{{</citation>}}


### (4/68) DPL: Decoupled Prompt Learning for Vision-Language Models (Chen Xu et al., 2023)

{{<citation>}}

Chen Xu, Yuhan Zhu, Guozhen Zhang, Haocheng Shen, Yixuan Liao, Xiaoxin Chen, Gangshan Wu, Limin Wang. (2023)  
**DPL: Decoupled Prompt Learning for Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.10061v1)  

---


**ABSTRACT**  
Prompt learning has emerged as an efficient and effective approach for transferring foundational Vision-Language Models (e.g., CLIP) to downstream tasks. However, current methods tend to overfit to seen categories, thereby limiting their generalization ability for unseen classes. In this paper, we propose a new method, Decoupled Prompt Learning (DPL), which reformulates the attention in prompt learning to alleviate this problem. Specifically, we theoretically investigate the collaborative process between prompts and instances (i.e., image patches/text tokens) by reformulating the original self-attention into four separate sub-processes. Through detailed analysis, we observe that certain sub-processes can be strengthened to bolster robustness and generalizability by some approximation techniques. Furthermore, we introduce language-conditioned textual prompting based on decoupled attention to naturally preserve the generalization of text input. Our approach is flexible for both visual and textual modalities, making it easily extendable to multi-modal prompt learning. By combining the proposed techniques, our approach achieves state-of-the-art performance on three representative benchmarks encompassing 15 image recognition datasets, while maintaining parameter-efficient. Moreover, our DPL does not rely on any auxiliary regularization task or extra training data, further demonstrating its remarkable generalization ability.

{{</citation>}}


### (5/68) Pseudo Flow Consistency for Self-Supervised 6D Object Pose Estimation (Yang Hai et al., 2023)

{{<citation>}}

Yang Hai, Rui Song, Jiaojiao Li, David Ferstl, Yinlin Hu. (2023)  
**Pseudo Flow Consistency for Self-Supervised 6D Object Pose Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.10016v1)  

---


**ABSTRACT**  
Most self-supervised 6D object pose estimation methods can only work with additional depth information or rely on the accurate annotation of 2D segmentation masks, limiting their application range. In this paper, we propose a 6D object pose estimation method that can be trained with pure RGB images without any auxiliary information. We first obtain a rough pose initialization from networks trained on synthetic images rendered from the target's 3D mesh. Then, we introduce a refinement strategy leveraging the geometry constraint in synthetic-to-real image pairs from multiple different views. We formulate this geometry constraint as pixel-level flow consistency between the training images with dynamically generated pseudo labels. We evaluate our method on three challenging datasets and demonstrate that it outperforms state-of-the-art self-supervised methods significantly, with neither 2D annotations nor additional depth images.

{{</citation>}}


### (6/68) Partition-and-Debias: Agnostic Biases Mitigation via A Mixture of Biases-Specific Experts (Jiaxuan Li et al., 2023)

{{<citation>}}

Jiaxuan Li, Duc Minh Vo, Hideki Nakayama. (2023)  
**Partition-and-Debias: Agnostic Biases Mitigation via A Mixture of Biases-Specific Experts**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.10005v1)  

---


**ABSTRACT**  
Bias mitigation in image classification has been widely researched, and existing methods have yielded notable results. However, most of these methods implicitly assume that a given image contains only one type of known or unknown bias, failing to consider the complexities of real-world biases. We introduce a more challenging scenario, agnostic biases mitigation, aiming at bias removal regardless of whether the type of bias or the number of types is unknown in the datasets. To address this difficult task, we present the Partition-and-Debias (PnD) method that uses a mixture of biases-specific experts to implicitly divide the bias space into multiple subspaces and a gating module to find a consensus among experts to achieve debiased classification. Experiments on both public and constructed benchmarks demonstrated the efficacy of the PnD. Code is available at: https://github.com/Jiaxuan-Li/PnD.

{{</citation>}}


### (7/68) AltDiffusion: A Multilingual Text-to-Image Diffusion Model (Fulong Ye et al., 2023)

{{<citation>}}

Fulong Ye, Guang Liu, Xinya Wu, Ledell Wu. (2023)  
**AltDiffusion: A Multilingual Text-to-Image Diffusion Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2308.09991v1)  

---


**ABSTRACT**  
Large Text-to-Image(T2I) diffusion models have shown a remarkable capability to produce photorealistic and diverse images based on text inputs. However, existing works only support limited language input, e.g., English, Chinese, and Japanese, leaving users beyond these languages underserved and blocking the global expansion of T2I models. Therefore, this paper presents AltDiffusion, a novel multilingual T2I diffusion model that supports eighteen different languages. Specifically, we first train a multilingual text encoder based on the knowledge distillation. Then we plug it into a pretrained English-only diffusion model and train the model with a two-stage schema to enhance the multilingual capability, including concept alignment and quality improvement stage on a large-scale multilingual dataset. Furthermore, we introduce a new benchmark, which includes Multilingual-General-18(MG-18) and Multilingual-Cultural-18(MC-18) datasets, to evaluate the capabilities of T2I diffusion models for generating high-quality images and capturing culture-specific concepts in different languages. Experimental results on both MG-18 and MC-18 demonstrate that AltDiffusion outperforms current state-of-the-art T2I models, e.g., Stable Diffusion in multilingual understanding, especially with respect to culture-specific concepts, while still having comparable capability for generating high-quality images.

{{</citation>}}


### (8/68) Anomaly-Aware Semantic Segmentation via Style-Aligned OoD Augmentation (Dan Zhang et al., 2023)

{{<citation>}}

Dan Zhang, Kaspar Sakmann, William Beluch, Robin Hutmacher, Yumeng Li. (2023)  
**Anomaly-Aware Semantic Segmentation via Style-Aligned OoD Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Augmentation, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.09965v1)  

---


**ABSTRACT**  
Within the context of autonomous driving, encountering unknown objects becomes inevitable during deployment in the open world. Therefore, it is crucial to equip standard semantic segmentation models with anomaly awareness. Many previous approaches have utilized synthetic out-of-distribution (OoD) data augmentation to tackle this problem. In this work, we advance the OoD synthesis process by reducing the domain gap between the OoD data and driving scenes, effectively mitigating the style difference that might otherwise act as an obvious shortcut during training. Additionally, we propose a simple fine-tuning loss that effectively induces a pre-trained semantic segmentation model to generate a ``none of the given classes" prediction, leveraging per-pixel OoD scores for anomaly segmentation. With minimal fine-tuning effort, our pipeline enables the use of pre-trained models for anomaly segmentation while maintaining the performance on the original task.

{{</citation>}}


### (9/68) UniAP: Towards Universal Animal Perception in Vision via Few-shot Learning (Meiqi Sun et al., 2023)

{{<citation>}}

Meiqi Sun, Zhonghan Zhao, Wenhao Chai, Hanjun Luo, Shidong Cao, Yanting Zhang, Jenq-Neng Hwang, Gaoang Wang. (2023)  
**UniAP: Towards Universal Animal Perception in Vision via Few-shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.09953v1)  

---


**ABSTRACT**  
Animal visual perception is an important technique for automatically monitoring animal health, understanding animal behaviors, and assisting animal-related research. However, it is challenging to design a deep learning-based perception model that can freely adapt to different animals across various perception tasks, due to the varying poses of a large diversity of animals, lacking data on rare species, and the semantic inconsistency of different tasks. We introduce UniAP, a novel Universal Animal Perception model that leverages few-shot learning to enable cross-species perception among various visual tasks. Our proposed model takes support images and labels as prompt guidance for a query image. Images and labels are processed through a Transformer-based encoder and a lightweight label encoder, respectively. Then a matching module is designed for aggregating information between prompt guidance and the query image, followed by a multi-head label decoder to generate outputs for various tasks. By capitalizing on the shared visual characteristics among different animals and tasks, UniAP enables the transfer of knowledge from well-studied species to those with limited labeled data or even unseen species. We demonstrate the effectiveness of UniAP through comprehensive experiments in pose estimation, segmentation, and classification tasks on diverse animal species, showcasing its ability to generalize and adapt to new classes with minimal labeled examples.

{{</citation>}}


### (10/68) Weakly-Supervised Action Localization by Hierarchically-structured Latent Attention Modeling (Guiqin Wang et al., 2023)

{{<citation>}}

Guiqin Wang, Peng Zhao, Cong Zhao, Shusen Yang, Jie Cheng, Luziwei Leng, Jianxing Liao, Qinghai Guo. (2023)  
**Weakly-Supervised Action Localization by Hierarchically-structured Latent Attention Modeling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.09946v1)  

---


**ABSTRACT**  
Weakly-supervised action localization aims to recognize and localize action instancese in untrimmed videos with only video-level labels. Most existing models rely on multiple instance learning(MIL), where the predictions of unlabeled instances are supervised by classifying labeled bags. The MIL-based methods are relatively well studied with cogent performance achieved on classification but not on localization. Generally, they locate temporal regions by the video-level classification but overlook the temporal variations of feature semantics. To address this problem, we propose a novel attention-based hierarchically-structured latent model to learn the temporal variations of feature semantics. Specifically, our model entails two components, the first is an unsupervised change-points detection module that detects change-points by learning the latent representations of video features in a temporal hierarchy based on their rates of change, and the second is an attention-based classification model that selects the change-points of the foreground as the boundaries. To evaluate the effectiveness of our model, we conduct extensive experiments on two benchmark datasets, THUMOS-14 and ActivityNet-v1.3. The experiments show that our method outperforms current state-of-the-art methods, and even achieves comparable performance with fully-supervised methods.

{{</citation>}}


### (11/68) BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions (Wenbo Hu et al., 2023)

{{<citation>}}

Wenbo Hu, Yifan Xu, Li Yi, Weiyue Li, Zeyuan Chen, Zhuowen Tu. (2023)  
**BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Language Model, OCR, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.09936v1)  

---


**ABSTRACT**  
Vision Language Models (VLMs), which extend Large Language Models (LLM) by incorporating visual understanding capability, have demonstrated significant advancements in addressing open-ended visual question-answering (VQA) tasks. However, these models cannot accurately interpret images infused with text, a common occurrence in real-world scenarios. Standard procedures for extracting information from images often involve learning a fixed set of query embeddings. These embeddings are designed to encapsulate image contexts and are later used as soft prompt inputs in LLMs. Yet, this process is limited to the token count, potentially curtailing the recognition of scenes with text-rich context. To improve upon them, the present study introduces BLIVA: an augmented version of InstructBLIP with Visual Assistant. BLIVA incorporates the query embeddings from InstructBLIP and also directly projects encoded patch embeddings into the LLM, a technique inspired by LLaVA. This approach assists the model to capture intricate details potentially missed during the query decoding process. Empirical evidence demonstrates that our model, BLIVA, significantly enhances performance in processing text-rich VQA benchmarks (up to 17.76\% in OCR-VQA benchmark) and in undertaking typical VQA benchmarks (up to 7.9\% in Visual Spatial Reasoning benchmark), comparing to our baseline InstructBLIP. BLIVA demonstrates significant capability in decoding real-world images, irrespective of text presence. To demonstrate the broad industry applications enabled by BLIVA, we evaluate the model using a new dataset comprising YouTube thumbnails paired with question-answer sets across 13 diverse categories. For researchers interested in further exploration, our code and models are freely accessible at https://github.com/mlpc-ucsd/BLIVA.git

{{</citation>}}


### (12/68) MDCS: More Diverse Experts with Consistency Self-distillation for Long-tailed Recognition (Qihao Zhao et al., 2023)

{{<citation>}}

Qihao Zhao, Chen Jiang, Wei Hu, Fan Zhang, Jun Liu. (2023)  
**MDCS: More Diverse Experts with Consistency Self-distillation for Long-tailed Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.09922v1)  

---


**ABSTRACT**  
Recently, multi-expert methods have led to significant improvements in long-tail recognition (LTR). We summarize two aspects that need further enhancement to contribute to LTR boosting: (1) More diverse experts; (2) Lower model variance. However, the previous methods didn't handle them well. To this end, we propose More Diverse experts with Consistency Self-distillation (MDCS) to bridge the gap left by earlier methods. Our MDCS approach consists of two core components: Diversity Loss (DL) and Consistency Self-distillation (CS). In detail, DL promotes diversity among experts by controlling their focus on different categories. To reduce the model variance, we employ KL divergence to distill the richer knowledge of weakly augmented instances for the experts' self-distillation. In particular, we design Confident Instance Sampling (CIS) to select the correctly classified instances for CS to avoid biased/noisy knowledge. In the analysis and ablation study, we demonstrate that our method compared with previous work can effectively increase the diversity of experts, significantly reduce the variance of the model, and improve recognition accuracy. Moreover, the roles of our DL and CS are mutually reinforcing and coupled: the diversity of experts benefits from the CS, and the CS cannot achieve remarkable results without the DL. Experiments show our MDCS outperforms the state-of-the-art by 1% $\sim$ 2% on five popular long-tailed benchmarks, including CIFAR10-LT, CIFAR100-LT, ImageNet-LT, Places-LT, and iNaturalist 2018. The code is available at https://github.com/fistyee/MDCS.

{{</citation>}}


### (13/68) EGANS: Evolutionary Generative Adversarial Network Search for Zero-Shot Learning (Shiming Chen et al., 2023)

{{<citation>}}

Shiming Chen, Shihuang Chen, Wenjin Hou, Weiping Ding, Xinge You. (2023)  
**EGANS: Evolutionary Generative Adversarial Network Search for Zero-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.09915v1)  

---


**ABSTRACT**  
Zero-shot learning (ZSL) aims to recognize the novel classes which cannot be collected for training a prediction model. Accordingly, generative models (e.g., generative adversarial network (GAN)) are typically used to synthesize the visual samples conditioned by the class semantic vectors and achieve remarkable progress for ZSL. However, existing GAN-based generative ZSL methods are based on hand-crafted models, which cannot adapt to various datasets/scenarios and fails to model instability. To alleviate these challenges, we propose evolutionary generative adversarial network search (termed EGANS) to automatically design the generative network with good adaptation and stability, enabling reliable visual feature sample synthesis for advancing ZSL. Specifically, we adopt cooperative dual evolution to conduct a neural architecture search for both generator and discriminator under a unified evolutionary adversarial framework. EGANS is learned by two stages: evolution generator architecture search and evolution discriminator architecture search. During the evolution generator architecture search, we adopt a many-to-one adversarial training strategy to evolutionarily search for the optimal generator. Then the optimal generator is further applied to search for the optimal discriminator in the evolution discriminator architecture search with a similar evolution search algorithm. Once the optimal generator and discriminator are searched, we entail them into various generative ZSL baselines for ZSL classification. Extensive experiments show that EGANS consistently improve existing generative ZSL methods on the standard CUB, SUN, AWA2 and FLO datasets. The significant performance gains indicate that the evolutionary neural architecture search explores a virgin field in ZSL.

{{</citation>}}


### (14/68) Noisy-Correspondence Learning for Text-to-Image Person Re-identification (Yang Qin et al., 2023)

{{<citation>}}

Yang Qin, Yingke Chen, Dezhong Peng, Xi Peng, Joey Tianyi Zhou, Peng Hu. (2023)  
**Noisy-Correspondence Learning for Text-to-Image Person Re-identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.09911v1)  

---


**ABSTRACT**  
Text-to-image person re-identification (TIReID) is a compelling topic in the cross-modal community, which aims to retrieve the target person based on a textual query. Although numerous TIReID methods have been proposed and achieved promising performance, they implicitly assume the training image-text pairs are correctly aligned, which is not always the case in real-world scenarios. In practice, the image-text pairs inevitably exist under-correlated or even false-correlated, a.k.a noisy correspondence (NC), due to the low quality of the images and annotation errors. To address this problem, we propose a novel Robust Dual Embedding method (RDE) that can learn robust visual-semantic associations even with NC. Specifically, RDE consists of two main components: 1) A Confident Consensus Division (CCD) module that leverages the dual-grained decisions of dual embedding modules to obtain a consensus set of clean training data, which enables the model to learn correct and reliable visual-semantic associations. 2) A Triplet-Alignment Loss (TAL) relaxes the conventional triplet-ranking loss with hardest negatives, which tends to rapidly overfit NC, to a log-exponential upper bound over all negatives, thus preventing the model from overemphasizing false image-text pairs. We conduct extensive experiments on three public benchmarks, namely CUHK-PEDES, ICFG-PEDES, and RSTPReID, to evaluate the performance and robustness of our RDE. Our method achieves state-of-the-art results both with and without synthetic noisy correspondences on all three datasets.

{{</citation>}}


### (15/68) Towards a High-Performance Object Detector: Insights from Drone Detection Using ViT and CNN-based Deep Learning Models (Junyang Zhang, 2023)

{{<citation>}}

Junyang Zhang. (2023)  
**Towards a High-Performance Object Detector: Insights from Drone Detection Using ViT and CNN-based Deep Learning Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Drone, Transformer  
[Paper Link](http://arxiv.org/abs/2308.09899v1)  

---


**ABSTRACT**  
Accurate drone detection is strongly desired in drone collision avoidance, drone defense and autonomous Unmanned Aerial Vehicle (UAV) self-landing. With the recent emergence of the Vision Transformer (ViT), this critical task is reassessed in this paper using a UAV dataset composed of 1359 drone photos. We construct various CNN and ViT-based models, demonstrating that for single-drone detection, a basic ViT can achieve performance 4.6 times more robust than our best CNN-based transfer learning models. By implementing the state-of-the-art You Only Look Once (YOLO v7, 200 epochs) and the experimental ViT-based You Only Look At One Sequence (YOLOS, 20 epochs) in multi-drone detection, we attain impressive 98% and 96% mAP values, respectively. We find that ViT outperforms CNN at the same epoch, but also requires more training data, computational power, and sophisticated, performance-oriented designs to fully surpass the capabilities of cutting-edge CNN detectors. We summarize the distinct characteristics of ViT and CNN models to aid future researchers in developing more efficient deep learning models.

{{</citation>}}


### (16/68) SwinLSTM:Improving Spatiotemporal Prediction Accuracy using Swin Transformer and LSTM (Song Tang et al., 2023)

{{<citation>}}

Song Tang, Chuang Li, Pu Zhang, RongNian Tang. (2023)  
**SwinLSTM:Improving Spatiotemporal Prediction Accuracy using Swin Transformer and LSTM**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2308.09891v1)  

---


**ABSTRACT**  
Integrating CNNs and RNNs to capture spatiotemporal dependencies is a prevalent strategy for spatiotemporal prediction tasks. However, the property of CNNs to learn local spatial information decreases their efficiency in capturing spatiotemporal dependencies, thereby limiting their prediction accuracy. In this paper, we propose a new recurrent cell, SwinLSTM, which integrates Swin Transformer blocks and the simplified LSTM, an extension that replaces the convolutional structure in ConvLSTM with the self-attention mechanism. Furthermore, we construct a network with SwinLSTM cell as the core for spatiotemporal prediction. Without using unique tricks, SwinLSTM outperforms state-of-the-art methods on Moving MNIST, Human3.6m, TaxiBJ, and KTH datasets. In particular, it exhibits a significant improvement in prediction accuracy compared to ConvLSTM. Our competitive experimental results demonstrate that learning global spatial dependencies is more advantageous for models to capture spatiotemporal dependencies. We hope that SwinLSTM can serve as a solid baseline to promote the advancement of spatiotemporal prediction accuracy. The codes are publicly available at https://github.com/SongTang-x/SwinLSTM.

{{</citation>}}


### (17/68) DUAW: Data-free Universal Adversarial Watermark against Stable Diffusion Customization (Xiaoyu Ye et al., 2023)

{{<citation>}}

Xiaoyu Ye, Hao Huang, Jiaqi An, Yongtao Wang. (2023)  
**DUAW: Data-free Universal Adversarial Watermark against Stable Diffusion Customization**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.09889v1)  

---


**ABSTRACT**  
Stable Diffusion (SD) customization approaches enable users to personalize SD model outputs, greatly enhancing the flexibility and diversity of AI art. However, they also allow individuals to plagiarize specific styles or subjects from copyrighted images, which raises significant concerns about potential copyright infringement. To address this issue, we propose an invisible data-free universal adversarial watermark (DUAW), aiming to protect a myriad of copyrighted images from different customization approaches across various versions of SD models. First, DUAW is designed to disrupt the variational autoencoder during SD customization. Second, DUAW operates in a data-free context, where it is trained on synthetic images produced by a Large Language Model (LLM) and a pretrained SD model. This approach circumvents the necessity of directly handling copyrighted images, thereby preserving their confidentiality. Once crafted, DUAW can be imperceptibly integrated into massive copyrighted images, serving as a protective measure by inducing significant distortions in the images generated by customized SD models. Experimental results demonstrate that DUAW can effectively distort the outputs of fine-tuned SD models, rendering them discernible to both human observers and a simple classifier.

{{</citation>}}


### (18/68) Calibrating Uncertainty for Semi-Supervised Crowd Counting (Chen Li et al., 2023)

{{<citation>}}

Chen Li, Xiaoling Hu, Shahira Abousamra, Chao Chen. (2023)  
**Calibrating Uncertainty for Semi-Supervised Crowd Counting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.09887v1)  

---


**ABSTRACT**  
Semi-supervised crowd counting is an important yet challenging task. A popular approach is to iteratively generate pseudo-labels for unlabeled data and add them to the training set. The key is to use uncertainty to select reliable pseudo-labels. In this paper, we propose a novel method to calibrate model uncertainty for crowd counting. Our method takes a supervised uncertainty estimation strategy to train the model through a surrogate function. This ensures the uncertainty is well controlled throughout the training. We propose a matching-based patch-wise surrogate function to better approximate uncertainty for crowd counting tasks. The proposed method pays a sufficient amount of attention to details, while maintaining a proper granularity. Altogether our method is able to generate reliable uncertainty estimation, high quality pseudolabels, and achieve state-of-the-art performance in semisupervised crowd counting.

{{</citation>}}


## cs.MM (1)



### (19/68) Dronevision: An Experimental 3D Testbed for Flying Light Specks (Hamed Alimohammadzadeh et al., 2023)

{{<citation>}}

Hamed Alimohammadzadeh, Rohit Bernard, Yang Chen, Trung Phan, Prashant Singh, Shuqin Zhu, Heather Culbertson, Shahram Ghandeharizadeh. (2023)  
**Dronevision: An Experimental 3D Testbed for Flying Light Specks**  

---
Primary Category: cs.MM  
Categories: cs-GR, cs-MM, cs-RO, cs.MM  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2308.10121v1)  

---


**ABSTRACT**  
Today's robotic laboratories for drones are housed in a large room. At times, they are the size of a warehouse. These spaces are typically equipped with permanent devices to localize the drones, e.g., Vicon Infrared cameras. Significant time is invested to fine-tune the localization apparatus to compute and control the position of the drones. One may use these laboratories to develop a 3D multimedia system with miniature sized drones configured with light sources. As an alternative, this brave new idea paper envisions shrinking these room-sized laboratories to the size of a cube or cuboid that sits on a desk and costs less than 10K dollars. The resulting Dronevision (DV) will be the size of a 1990s Television. In addition to light sources, its Flying Light Specks (FLSs) will be network-enabled drones with storage and processing capability to implement decentralized algorithms. The DV will include a localization technique to expedite development of 3D displays. It will act as a haptic interface for a user to interact with and manipulate the 3D virtual illuminations. It will empower an experimenter to design, implement, test, debug, and maintain software and hardware that realize novel algorithms in the comfort of their office without having to reserve a laboratory. In addition to enhancing productivity, it will improve safety of the experimenter by minimizing the likelihood of accidents. This paper introduces the concept of a DV, the research agenda one may pursue using this device, and our plans to realize one.

{{</citation>}}


## cs.LG (15)



### (20/68) Deep Generative Modeling-based Data Augmentation with Demonstration using the BFBT Benchmark Void Fraction Datasets (Farah Alsafadi et al., 2023)

{{<citation>}}

Farah Alsafadi, Xu Wu. (2023)  
**Deep Generative Modeling-based Data Augmentation with Demonstration using the BFBT Benchmark Void Fraction Datasets**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.10120v1)  

---


**ABSTRACT**  
Deep learning (DL) has achieved remarkable successes in many disciplines such as computer vision and natural language processing due to the availability of ``big data''. However, such success cannot be easily replicated in many nuclear engineering problems because of the limited amount of training data, especially when the data comes from high-cost experiments. To overcome such a data scarcity issue, this paper explores the applications of deep generative models (DGMs) that have been widely used for image data generation to scientific data augmentation. DGMs, such as generative adversarial networks (GANs), normalizing flows (NFs), variational autoencoders (VAEs), and conditional VAEs (CVAEs), can be trained to learn the underlying probabilistic distribution of the training dataset. Once trained, they can be used to generate synthetic data that are similar to the training data and significantly expand the dataset size. By employing DGMs to augment TRACE simulated data of the steady-state void fractions based on the NUPEC Boiling Water Reactor Full-size Fine-mesh Bundle Test (BFBT) benchmark, this study demonstrates that VAEs, CVAEs, and GANs have comparable generative performance with similar errors in the synthetic data, with CVAEs achieving the smallest errors. The findings shows that DGMs have a great potential to augment scientific data in nuclear engineering, which proves effective for expanding the training dataset and enabling other DL models to be trained more accurately.

{{</citation>}}


### (21/68) Geometric instability of graph neural networks on large graphs (Emily Morris et al., 2023)

{{<citation>}}

Emily Morris, Haotian Shen, Weiling Du, Muhammad Hamza Sajjad, Borun Shi. (2023)  
**Geometric instability of graph neural networks on large graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.10099v1)  

---


**ABSTRACT**  
We analyse the geometric instability of embeddings produced by graph neural networks (GNNs). Existing methods are only applicable for small graphs and lack context in the graph domain. We propose a simple, efficient and graph-native Graph Gram Index (GGI) to measure such instability which is invariant to permutation, orthogonal transformation, translation and order of evaluation. This allows us to study the varying instability behaviour of GNN embeddings on large graphs for both node classification and link prediction.

{{</citation>}}


### (22/68) Contrastive Learning for Non-Local Graphs with Multi-Resolution Structural Views (Asif Khan et al., 2023)

{{<citation>}}

Asif Khan, Amos Storkey. (2023)  
**Contrastive Learning for Non-Local Graphs with Multi-Resolution Structural Views**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.10077v1)  

---


**ABSTRACT**  
Learning node-level representations of heterophilic graphs is crucial for various applications, including fraudster detection and protein function prediction. In such graphs, nodes share structural similarity identified by the equivalence of their connectivity which is implicitly encoded in the form of higher-order hierarchical information in the graphs. The contrastive methods are popular choices for learning the representation of nodes in a graph. However, existing contrastive methods struggle to capture higher-order graph structures. To address this limitation, we propose a novel multiview contrastive learning approach that integrates diffusion filters on graphs. By incorporating multiple graph views as augmentations, our method captures the structural equivalence in heterophilic graphs, enabling the discovery of hidden relationships and similarities not apparent in traditional node representations. Our approach outperforms baselines on synthetic and real structural datasets, surpassing the best baseline by $16.06\%$ on Cornell, $3.27\%$ on Texas, and $8.04\%$ on Wisconsin. Additionally, it consistently achieves superior performance on proximal tasks, demonstrating its effectiveness in uncovering structural information and improving downstream applications.

{{</citation>}}


### (23/68) Efficient Representation Learning for Healthcare with Cross-Architectural Self-Supervision (Pranav Singh et al., 2023)

{{<citation>}}

Pranav Singh, Jacopo Cirrone. (2023)  
**Efficient Representation Learning for Healthcare with Cross-Architectural Self-Supervision**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Representation Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.10064v1)  

---


**ABSTRACT**  
In healthcare and biomedical applications, extreme computational requirements pose a significant barrier to adopting representation learning. Representation learning can enhance the performance of deep learning architectures by learning useful priors from limited medical data. However, state-of-the-art self-supervised techniques suffer from reduced performance when using smaller batch sizes or shorter pretraining epochs, which are more practical in clinical settings. We present Cross Architectural - Self Supervision (CASS) in response to this challenge. This novel siamese self-supervised learning approach synergistically leverages Transformer and Convolutional Neural Networks (CNN) for efficient learning. Our empirical evaluation demonstrates that CASS-trained CNNs and Transformers outperform existing self-supervised learning methods across four diverse healthcare datasets. With only 1% labeled data for finetuning, CASS achieves a 3.8% average improvement; with 10% labeled data, it gains 5.9%; and with 100% labeled data, it reaches a remarkable 10.13% enhancement. Notably, CASS reduces pretraining time by 69% compared to state-of-the-art methods, making it more amenable to clinical implementation. We also demonstrate that CASS is considerably more robust to variations in batch size and pretraining epochs, making it a suitable candidate for machine learning in healthcare applications.

{{</citation>}}


### (24/68) The Snowflake Hypothesis: Training Deep GNN with One Node One Receptive field (Kun Wang et al., 2023)

{{<citation>}}

Kun Wang, Guohao Li, Shilong Wang, Guibin Zhang, Kai Wang, Yang You, Xiaojiang Peng, Yuxuan Liang, Yang Wang. (2023)  
**The Snowflake Hypothesis: Training Deep GNN with One Node One Receptive field**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.10051v1)  

---


**ABSTRACT**  
Despite Graph Neural Networks demonstrating considerable promise in graph representation learning tasks, GNNs predominantly face significant issues with over-fitting and over-smoothing as they go deeper as models of computer vision realm. In this work, we conduct a systematic study of deeper GNN research trajectories. Our findings indicate that the current success of deep GNNs primarily stems from (I) the adoption of innovations from CNNs, such as residual/skip connections, or (II) the tailor-made aggregation algorithms like DropEdge. However, these algorithms often lack intrinsic interpretability and indiscriminately treat all nodes within a given layer in a similar manner, thereby failing to capture the nuanced differences among various nodes. To this end, we introduce the Snowflake Hypothesis -- a novel paradigm underpinning the concept of ``one node, one receptive field''. The hypothesis draws inspiration from the unique and individualistic patterns of each snowflake, proposing a corresponding uniqueness in the receptive fields of nodes in the GNNs.   We employ the simplest gradient and node-level cosine distance as guiding principles to regulate the aggregation depth for each node, and conduct comprehensive experiments including: (1) different training schemes; (2) various shallow and deep GNN backbones, and (3) various numbers of layers (8, 16, 32, 64) on multiple benchmarks (six graphs including dense graphs with millions of nodes); (4) compare with different aggregation strategies. The observational results demonstrate that our hypothesis can serve as a universal operator for a range of tasks, and it displays tremendous potential on deep GNNs. It can be applied to various GNN frameworks, enhancing its effectiveness when operating in-depth, and guiding the selection of the optimal network depth in an explainable and generalizable way.

{{</citation>}}


### (25/68) Semi-Supervised Anomaly Detection for the Determination of Vehicle Hijacking Tweets (Taahir Aiyoob Patel et al., 2023)

{{<citation>}}

Taahir Aiyoob Patel, Clement N. Nyirenda. (2023)  
**Semi-Supervised Anomaly Detection for the Determination of Vehicle Hijacking Tweets**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.10036v1)  

---


**ABSTRACT**  
In South Africa, there is an ever-growing issue of vehicle hijackings. This leads to travellers constantly being in fear of becoming a victim to such an incident. This work presents a new semi-supervised approach to using tweets to identify hijacking incidents by using unsupervised anomaly detection algorithms. Tweets consisting of the keyword "hijacking" are obtained, stored, and processed using the term frequency-inverse document frequency (TF-IDF) and further analyzed by using two anomaly detection algorithms: 1) K-Nearest Neighbour (KNN); 2) Cluster Based Outlier Factor (CBLOF). The comparative evaluation showed that the KNN method produced an accuracy of 89%, whereas the CBLOF produced an accuracy of 90%. The CBLOF method was also able to obtain a F1-Score of 0.8, whereas the KNN produced a 0.78. Therefore, there is a slight difference between the two approaches, in favour of CBLOF, which has been selected as a preferred unsupervised method for the determination of relevant hijacking tweets. In future, a comparison will be done between supervised learning methods and the unsupervised methods presented in this work on larger dataset. Optimisation mechanisms will also be employed in order to increase the overall performance.

{{</citation>}}


### (26/68) To prune or not to prune : A chaos-causality approach to principled pruning of dense neural networks (Rajan Sahu et al., 2023)

{{<citation>}}

Rajan Sahu, Shivam Chadha, Nithin Nagaraj, Archana Mathur, Snehanshu Saha. (2023)  
**To prune or not to prune : A chaos-causality approach to principled pruning of dense neural networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2308.09955v1)  

---


**ABSTRACT**  
Reducing the size of a neural network (pruning) by removing weights without impacting its performance is an important problem for resource-constrained devices. In the past, pruning was typically accomplished by ranking or penalizing weights based on criteria like magnitude and removing low-ranked weights before retraining the remaining ones. Pruning strategies may also involve removing neurons from the network in order to achieve the desired reduction in network size. We formulate pruning as an optimization problem with the objective of minimizing misclassifications by selecting specific weights. To accomplish this, we have introduced the concept of chaos in learning (Lyapunov exponents) via weight updates and exploiting causality to identify the causal weights responsible for misclassification. Such a pruned network maintains the original performance and retains feature explainability.

{{</citation>}}


### (27/68) Study on the effectiveness of AutoML in detecting cardiovascular disease (T. V. Afanasieva et al., 2023)

{{<citation>}}

T. V. Afanasieva, A. P. Kuzlyakin, A. V. Komolov. (2023)  
**Study on the effectiveness of AutoML in detecting cardiovascular disease**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09947v1)  

---


**ABSTRACT**  
Cardiovascular diseases are widespread among patients with chronic noncommunicable diseases and are one of the leading causes of death, including in the working age. The article presents the relevance of the development and application of patient-oriented systems, in which machine learning (ML) is a promising technology that allows predicting cardiovascular diseases. Automated machine learning (AutoML) makes it possible to simplify and speed up the process of developing AI/ML applications, which is key in the development of patient-oriented systems by application users, in particular medical specialists. The authors propose a framework for the application of automatic machine learning and three scenarios that allowed for data combining five data sets of cardiovascular disease indicators from the UCI Machine Learning Repository to investigate the effectiveness in detecting this class of diseases. The study investigated one AutoML model that used and optimized the hyperparameters of thirteen basic ML models (KNeighborsUnif, KNeighborsDist, LightGBMXT, LightGBM, RandomForestGini, RandomForestEntr, CatBoost, ExtraTreesGini, ExtraTreesEntr, NeuralNetFastA, XGBoost, NeuralNetTorch, LightGBMLarge) and included the most accurate models in the weighted ensemble. The results of the study showed that the structure of the AutoML model for detecting cardiovascular diseases depends not only on the efficiency and accuracy of the basic models used, but also on the scenarios for preprocessing the initial data, in particular, on the technique of data normalization. The comparative analysis showed that the accuracy of the AutoML model in detecting cardiovascular disease varied in the range from 87.41% to 92.3%, and the maximum accuracy was obtained when normalizing the source data into binary values, and the minimum was obtained when using the built-in AutoML technique.

{{</citation>}}


### (28/68) Never Explore Repeatedly in Multi-Agent Reinforcement Learning (Chenghao Li et al., 2023)

{{<citation>}}

Chenghao Li, Tonghan Wang, Chongjie Zhang, Qianchuan Zhao. (2023)  
**Never Explore Repeatedly in Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Google, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.09909v1)  

---


**ABSTRACT**  
In the realm of multi-agent reinforcement learning, intrinsic motivations have emerged as a pivotal tool for exploration. While the computation of many intrinsic rewards relies on estimating variational posteriors using neural network approximators, a notable challenge has surfaced due to the limited expressive capability of these neural statistics approximators. We pinpoint this challenge as the "revisitation" issue, where agents recurrently explore confined areas of the task space. To combat this, we propose a dynamic reward scaling approach. This method is crafted to stabilize the significant fluctuations in intrinsic rewards in previously explored areas and promote broader exploration, effectively curbing the revisitation phenomenon. Our experimental findings underscore the efficacy of our approach, showcasing enhanced performance in demanding environments like Google Research Football and StarCraft II micromanagement tasks, especially in sparse reward settings.

{{</citation>}}


### (29/68) Imputing Brain Measurements Across Data Sets via Graph Neural Networks (Yixin Wang et al., 2023)

{{<citation>}}

Yixin Wang, Wei Peng, Susan F. Tapert, Qingyu Zhao, Kilian M. Pohl. (2023)  
**Imputing Brain Measurements Across Data Sets via Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.09907v1)  

---


**ABSTRACT**  
Publicly available data sets of structural MRIs might not contain specific measurements of brain Regions of Interests (ROIs) that are important for training machine learning models. For example, the curvature scores computed by Freesurfer are not released by the Adolescent Brain Cognitive Development (ABCD) Study. One can address this issue by simply reapplying Freesurfer to the data set. However, this approach is generally computationally and labor intensive (e.g., requiring quality control). An alternative is to impute the missing measurements via a deep learning approach. However, the state-of-the-art is designed to estimate randomly missing values rather than entire measurements. We therefore propose to re-frame the imputation problem as a prediction task on another (public) data set that contains the missing measurements and shares some ROI measurements with the data sets of interest. A deep learning model is then trained to predict the missing measurements from the shared ones and afterwards is applied to the other data sets. Our proposed algorithm models the dependencies between ROI measurements via a graph neural network (GNN) and accounts for demographic differences in brain measurements (e.g. sex) by feeding the graph encoding into a parallel architecture. The architecture simultaneously optimizes a graph decoder to impute values and a classifier in predicting demographic factors. We test the approach, called Demographic Aware Graph-based Imputation (DAGI), on imputing those missing Freesurfer measurements of ABCD (N=3760) by training the predictor on those publicly released by the National Consortium on Alcohol and Neurodevelopment in Adolescence (NCANDA, N=540)...

{{</citation>}}


### (30/68) DPMAC: Differentially Private Communication for Cooperative Multi-Agent Reinforcement Learning (Canzhe Zhao et al., 2023)

{{<citation>}}

Canzhe Zhao, Yanjie Ze, Jing Dong, Baoxiang Wang, Shuai Li. (2023)  
**DPMAC: Differentially Private Communication for Cooperative Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.09902v1)  

---


**ABSTRACT**  
Communication lays the foundation for cooperation in human society and in multi-agent reinforcement learning (MARL). Humans also desire to maintain their privacy when communicating with others, yet such privacy concern has not been considered in existing works in MARL. To this end, we propose the \textit{differentially private multi-agent communication} (DPMAC) algorithm, which protects the sensitive information of individual agents by equipping each agent with a local message sender with rigorous $(\epsilon, \delta)$-differential privacy (DP) guarantee. In contrast to directly perturbing the messages with predefined DP noise as commonly done in privacy-preserving scenarios, we adopt a stochastic message sender for each agent respectively and incorporate the DP requirement into the sender, which automatically adjusts the learned message distribution to alleviate the instability caused by DP noise. Further, we prove the existence of a Nash equilibrium in cooperative MARL with privacy-preserving communication, which suggests that this problem is game-theoretically learnable. Extensive experiments demonstrate a clear advantage of DPMAC over baseline methods in privacy-preserving scenarios.

{{</citation>}}


### (31/68) Contrastive Learning-based Imputation-Prediction Networks for In-hospital Mortality Risk Modeling using EHRs (Yuxi Liu et al., 2023)

{{<citation>}}

Yuxi Liu, Zhenhao Zhang, Shaowen Qin, Flora D. Salim, Antonio Jimeno Yepes. (2023)  
**Contrastive Learning-based Imputation-Prediction Networks for In-hospital Mortality Risk Modeling using EHRs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.09896v1)  

---


**ABSTRACT**  
Predicting the risk of in-hospital mortality from electronic health records (EHRs) has received considerable attention. Such predictions will provide early warning of a patient's health condition to healthcare professionals so that timely interventions can be taken. This prediction task is challenging since EHR data are intrinsically irregular, with not only many missing values but also varying time intervals between medical records. Existing approaches focus on exploiting the variable correlations in patient medical records to impute missing values and establishing time-decay mechanisms to deal with such irregularity. This paper presents a novel contrastive learning-based imputation-prediction network for predicting in-hospital mortality risks using EHR data. Our approach introduces graph analysis-based patient stratification modeling in the imputation process to group similar patients. This allows information of similar patients only to be used, in addition to personal contextual information, for missing value imputation. Moreover, our approach can integrate contrastive learning into the proposed network architecture to enhance patient representation learning and predictive performance on the classification task. Experiments on two real-world EHR datasets show that our approach outperforms the state-of-the-art approaches in both imputation and prediction tasks.

{{</citation>}}


### (32/68) Inductive-bias Learning: Generating Code Models with Large Language Model (Toma Tanaka et al., 2023)

{{<citation>}}

Toma Tanaka, Naofumi Emoto, Tsukasa Yumibayashi. (2023)  
**Inductive-bias Learning: Generating Code Models with Large Language Model**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2308.09890v1)  

---


**ABSTRACT**  
Large Language Models(LLMs) have been attracting attention due to a ability called in-context learning(ICL). ICL, without updating the parameters of a LLM, it is possible to achieve highly accurate inference based on rules ``in the context'' by merely inputting a training data into the prompt. Although ICL is a developing field with many unanswered questions, LLMs themselves serves as a inference model, seemingly realizing inference without explicitly indicate ``inductive bias''. On the other hand, a code generation is also a highlighted application of LLMs. The accuracy of code generation has dramatically improved, enabling even non-engineers to generate code to perform the desired tasks by crafting appropriate prompts. In this paper, we propose a novel ``learning'' method called an ``Inductive-Bias Learning (IBL)'', which combines the techniques of ICL and code generation. An idea of IBL is straightforward. Like ICL, IBL inputs a training data into the prompt and outputs a code with a necessary structure for inference (we referred to as ``Code Model'') from a ``contextual understanding''. Despite being a seemingly simple approach, IBL encompasses both a ``property of inference without explicit inductive bias'' inherent in ICL and a ``readability and explainability'' of the code generation. Surprisingly, generated Code Models have been found to achieve predictive accuracy comparable to, and in some cases surpassing, ICL and representative machine learning models. Our IBL code is open source: https://github.com/fuyu-quant/IBLM

{{</citation>}}


### (33/68) A Transformer-based Framework For Multi-variate Time Series: A Remaining Useful Life Prediction Use Case (Oluwaseyi Ogunfowora et al., 2023)

{{<citation>}}

Oluwaseyi Ogunfowora, Homayoun Najjaran. (2023)  
**A Transformer-based Framework For Multi-variate Time Series: A Remaining Useful Life Prediction Use Case**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Language Model, Natural Language Processing, Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2308.09884v1)  

---


**ABSTRACT**  
In recent times, Large Language Models (LLMs) have captured a global spotlight and revolutionized the field of Natural Language Processing. One of the factors attributed to the effectiveness of LLMs is the model architecture used for training, transformers. Transformer models excel at capturing contextual features in sequential data since time series data are sequential, transformer models can be leveraged for more efficient time series data prediction. The field of prognostics is vital to system health management and proper maintenance planning. A reliable estimation of the remaining useful life (RUL) of machines holds the potential for substantial cost savings. This includes avoiding abrupt machine failures, maximizing equipment usage, and serving as a decision support system (DSS). This work proposed an encoder-transformer architecture-based framework for multivariate time series prediction for a prognostics use case. We validated the effectiveness of the proposed framework on all four sets of the C-MAPPS benchmark dataset for the remaining useful life prediction task. To effectively transfer the knowledge and application of transformers from the natural language domain to time series, three model-specific experiments were conducted. Also, to enable the model awareness of the initial stages of the machine life and its degradation path, a novel expanding window method was proposed for the first time in this work, it was compared with the sliding window method, and it led to a large improvement in the performance of the encoder transformer model. Finally, the performance of the proposed encoder-transformer model was evaluated on the test dataset and compared with the results from 13 other state-of-the-art (SOTA) models in the literature and it outperformed them all with an average performance increase of 137.65% over the next best model across all the datasets.

{{</citation>}}


### (34/68) Skill Transformer: A Monolithic Policy for Mobile Manipulation (Xiaoyu Huang et al., 2023)

{{<citation>}}

Xiaoyu Huang, Dhruv Batra, Akshara Rai, Andrew Szot. (2023)  
**Skill Transformer: A Monolithic Policy for Mobile Manipulation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.09873v1)  

---


**ABSTRACT**  
We present Skill Transformer, an approach for solving long-horizon robotic tasks by combining conditional sequence modeling and skill modularity. Conditioned on egocentric and proprioceptive observations of a robot, Skill Transformer is trained end-to-end to predict both a high-level skill (e.g., navigation, picking, placing), and a whole-body low-level action (e.g., base and arm motion), using a transformer architecture and demonstration trajectories that solve the full task. It retains the composability and modularity of the overall task through a skill predictor module while reasoning about low-level actions and avoiding hand-off errors, common in modular approaches. We test Skill Transformer on an embodied rearrangement benchmark and find it performs robust task planning and low-level control in new scenarios, achieving a 2.5x higher success rate than baselines in hard rearrangement problems.

{{</citation>}}


## cs.CL (9)



### (35/68) Open, Closed, or Small Language Models for Text Classification? (Hao Yu et al., 2023)

{{<citation>}}

Hao Yu, Zachary Yang, Kellin Pelrine, Jean Francois Godbout, Reihaneh Rabbany. (2023)  
**Open, Closed, or Small Language Models for Text Classification?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model, NLP, Text Classification  
[Paper Link](http://arxiv.org/abs/2308.10092v1)  

---


**ABSTRACT**  
Recent advancements in large language models have demonstrated remarkable capabilities across various NLP tasks. But many questions remain, including whether open-source models match closed ones, why these models excel or struggle with certain tasks, and what types of practical procedures can improve performance. We address these questions in the context of classification by evaluating three classes of models using eight datasets across three distinct tasks: named entity recognition, political party prediction, and misinformation detection. While larger LLMs often lead to improved performance, open-source models can rival their closed-source counterparts by fine-tuning. Moreover, supervised smaller models, like RoBERTa, can achieve similar or even greater performance in many datasets compared to generative LLMs. On the other hand, closed models maintain an advantage in hard tasks that demand the most generalizability. This study underscores the importance of model selection based on task requirements

{{</citation>}}


### (36/68) PACE: Improving Prompt with Actor-Critic Editing for Large Language Model (Yihong Dong et al., 2023)

{{<citation>}}

Yihong Dong, Kangcheng Luo, Xue Jiang, Zhi Jin, Ge Li. (2023)  
**PACE: Improving Prompt with Actor-Critic Editing for Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SE, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.10088v1)  

---


**ABSTRACT**  
Large language models (LLMs) have showcased remarkable potential across various tasks by conditioning on prompts. However, the quality of different human-written prompts leads to substantial discrepancies in LLMs' performance, and improving prompts usually necessitates considerable human effort and expertise. To this end, this paper proposes Prompt with Actor-Critic Editing (PACE) for LLMs to enable automatic prompt editing. Drawing inspiration from the actor-critic algorithm in reinforcement learning, PACE leverages LLMs as the dual roles of actors and critics, conceptualizing prompt as a type of policy. PACE refines prompt, taking into account the feedback from both actors performing prompt and critics criticizing response. This process helps LLMs better align prompt to a specific task, thanks to real responses and thinking from LLMs. We conduct extensive experiments on 24 instruction induction tasks and 21 big-bench tasks. Experimental results indicate that PACE elevates the relative performance of medium/low-quality human-written prompts by up to 98\%, which has comparable performance to high-quality human-written prompts. Moreover, PACE also exhibits notable efficacy for prompt generation.

{{</citation>}}


### (37/68) HICL: Hashtag-Driven In-Context Learning for Social Media Natural Language Understanding (Hanzhuo Tan et al., 2023)

{{<citation>}}

Hanzhuo Tan, Chunpu Xu, Jing Li, Yuqun Zhang, Zeyang Fang, Zeyu Chen, Baohua Lai. (2023)  
**HICL: Hashtag-Driven In-Context Learning for Social Media Natural Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NLU, Natural Language Understanding, Social Media  
[Paper Link](http://arxiv.org/abs/2308.09985v1)  

---


**ABSTRACT**  
Natural language understanding (NLU) is integral to various social media applications. However, existing NLU models rely heavily on context for semantic learning, resulting in compromised performance when faced with short and noisy social media content. To address this issue, we leverage in-context learning (ICL), wherein language models learn to make inferences by conditioning on a handful of demonstrations to enrich the context and propose a novel hashtag-driven in-context learning (HICL) framework. Concretely, we pre-train a model #Encoder, which employs #hashtags (user-annotated topic labels) to drive BERT-based pre-training through contrastive learning. Our objective here is to enable #Encoder to gain the ability to incorporate topic-related semantic information, which allows it to retrieve topic-related posts to enrich contexts and enhance social media NLU with noisy contexts. To further integrate the retrieved context with the source text, we employ a gradient-based method to identify trigger terms useful in fusing information from both sources. For empirical studies, we collected 45M tweets to set up an in-context NLU benchmark, and the experimental results on seven downstream tasks show that HICL substantially advances the previous state-of-the-art results. Furthermore, we conducted extensive analyzes and found that: (1) combining source input with a top-retrieved post from #Encoder is more effective than using semantically similar posts; (2) trigger words can largely benefit in merging context from the source and retrieved posts.

{{</citation>}}


### (38/68) FinEval: A Chinese Financial Domain Knowledge Evaluation Benchmark for Large Language Models (Liwen Zhang et al., 2023)

{{<citation>}}

Liwen Zhang, Weige Cai, Zhaowei Liu, Zhi Yang, Wei Dai, Yujie Liao, Qianru Qin, Yifei Li, Xingyu Liu, Zhiqiang Liu, Zhoufan Zhu, Anbo Wu, Xin Guo, Yun Chen. (2023)  
**FinEval: A Chinese Financial Domain Knowledge Evaluation Benchmark for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Financial, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.09975v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated exceptional performance in various natural language processing tasks, yet their efficacy in more challenging and domain-specific tasks remains largely unexplored. This paper presents FinEval, a benchmark specifically designed for the financial domain knowledge in the LLMs. FinEval is a collection of high-quality multiple-choice questions covering Finance, Economy, Accounting, and Certificate. It includes 4,661 questions spanning 34 different academic subjects. To ensure a comprehensive model performance evaluation, FinEval employs a range of prompt types, including zero-shot and few-shot prompts, as well as answer-only and chain-of-thought prompts. Evaluating state-of-the-art Chinese and English LLMs on FinEval, the results show that only GPT-4 achieved an accuracy close to 70% in different prompt settings, indicating significant growth potential for LLMs in the financial domain knowledge. Our work offers a more comprehensive financial knowledge evaluation benchmark, utilizing data of mock exams and covering a wide range of evaluated LLMs.

{{</citation>}}


### (39/68) Tackling Vision Language Tasks Through Learning Inner Monologues (Diji Yang et al., 2023)

{{<citation>}}

Diji Yang, Kezhen Chen, Jinmeng Rao, Xiaoyuan Guo, Yawen Zhang, Jie Yang, Yi Zhang. (2023)  
**Tackling Vision Language Tasks Through Learning Inner Monologues**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.09970v1)  

---


**ABSTRACT**  
Visual language tasks require AI models to comprehend and reason with both visual and textual content. Driven by the power of Large Language Models (LLMs), two prominent methods have emerged: (1) the hybrid integration between LLMs and Vision-Language Models (VLMs), where visual inputs are firstly converted into language descriptions by VLMs, serving as inputs for LLMs to generate final answer(s); (2) visual feature alignment in language space, where visual inputs are encoded as embeddings and projected to LLMs' language space via further supervised fine-tuning. The first approach provides light training costs and interpretability but is hard to be optimized in an end-to-end fashion. The second approach presents decent performance, but feature alignment usually requires large amounts of training data and lacks interpretability. To tackle this dilemma, we propose a novel approach, Inner Monologue Multi-Modal Optimization (IMMO), to solve complex vision language problems by simulating inner monologue processes, a cognitive process in which an individual engages in silent verbal communication with themselves. We enable LLMs and VLMs to interact through natural language conversation and propose to use a two-stage training process to learn how to do the inner monologue (self-asking questions and answering questions). IMMO is evaluated on two popular tasks and the results suggest by emulating the cognitive phenomenon of internal dialogue, our approach can enhance reasoning and explanation abilities, contributing to the more effective fusion of vision and language models. More importantly, instead of using predefined human-crafted monologues, IMMO learns this process within the deep learning models, promising wider applicability to many different AI problems beyond vision language tasks.

{{</citation>}}


### (40/68) Data-to-text Generation for Severely Under-Resourced Languages with GPT-3.5: A Bit of Help Needed from Google Translate (Michela Lorandi et al., 2023)

{{<citation>}}

Michela Lorandi, Anya Belz. (2023)  
**Data-to-text Generation for Severely Under-Resourced Languages with GPT-3.5: A Bit of Help Needed from Google Translate**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Google  
[Paper Link](http://arxiv.org/abs/2308.09957v1)  

---


**ABSTRACT**  
LLMs like GPT are great at tasks involving English which dominates in their training data. In this paper, we look at how they cope with tasks involving languages that are severely under-represented in their training data, in the context of data-to-text generation for Irish, Maltese, Welsh and Breton. During the prompt-engineering phase we tested a range of prompt types and formats on GPT-3.5 and~4 with a small sample of example input/output pairs. We then fully evaluated the two most promising prompts in two scenarios: (i) direct generation into the under-resourced language, and (ii) generation into English followed by translation into the under-resourced language. We find that few-shot prompting works better for direct generation into under-resourced languages, but that the difference disappears when pivoting via English. The few-shot + translation system variants were submitted to the WebNLG 2023 shared task where they outperformed competitor systems by substantial margins in all languages on all metrics. We conclude that good performance on under-resourced languages can be achieved out-of-the box with state-of-the-art LLMs. However, our best results (for Welsh) remain well below the lowest ranked English system at WebNLG'20.

{{</citation>}}


### (41/68) Eva-KELLM: A New Benchmark for Evaluating Knowledge Editing of LLMs (Suhang Wu et al., 2023)

{{<citation>}}

Suhang Wu, Minlong Peng, Yue Chen, Jinsong Su, Mingming Sun. (2023)  
**Eva-KELLM: A New Benchmark for Evaluating Knowledge Editing of LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.09954v1)  

---


**ABSTRACT**  
Large language models (LLMs) possess a wealth of knowledge encoded in their parameters. However, this knowledge may become outdated or unsuitable over time. As a result, there has been a growing interest in knowledge editing for LLMs and evaluating its effectiveness. Existing studies primarily focus on knowledge editing using factual triplets, which not only incur high costs for collection but also struggle to express complex facts. Furthermore, these studies are often limited in their evaluation perspectives. In this paper, we propose Eva-KELLM, a new benchmark for evaluating knowledge editing of LLMs. This benchmark includes an evaluation framework and a corresponding dataset. Under our framework, we first ask the LLM to perform knowledge editing using raw documents, which provides a more convenient and universal approach compared to using factual triplets. We then evaluate the updated LLM from multiple perspectives. In addition to assessing the effectiveness of knowledge editing and the retention of unrelated knowledge from conventional studies, we further test the LLM's ability in two aspects: 1) Reasoning with the altered knowledge, aiming for the LLM to genuinely learn the altered knowledge instead of simply memorizing it. 2) Cross-lingual knowledge transfer, where the LLM updated with raw documents in one language should be capable of handling queries from another language. To facilitate further research, we construct and release the corresponding dataset. Using this benchmark, we investigate the effectiveness of several commonly-used knowledge editing methods. Experimental results indicate that the current methods for knowledge editing using raw documents are not effective in yielding satisfactory results, particularly when it comes to reasoning with altered knowledge and cross-lingual knowledge transfer.

{{</citation>}}


### (42/68) Utilizing Semantic Textual Similarity for Clinical Survey Data Feature Selection (Benjamin C. Warner et al., 2023)

{{<citation>}}

Benjamin C. Warner, Ziqi Xu, Simon Haroutounian, Thomas Kannampallil, Chenyang Lu. (2023)  
**Utilizing Semantic Textual Similarity for Clinical Survey Data Feature Selection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Clinical, Textual Similarity  
[Paper Link](http://arxiv.org/abs/2308.09892v1)  

---


**ABSTRACT**  
Survey data can contain a high number of features while having a comparatively low quantity of examples. Machine learning models that attempt to predict outcomes from survey data under these conditions can overfit and result in poor generalizability. One remedy to this issue is feature selection, which attempts to select an optimal subset of features to learn upon. A relatively unexplored source of information in the feature selection process is the usage of textual names of features, which may be semantically indicative of which features are relevant to a target outcome. The relationships between feature names and target names can be evaluated using language models (LMs) to produce semantic textual similarity (STS) scores, which can then be used to select features. We examine the performance using STS to select features directly and in the minimal-redundancy-maximal-relevance (mRMR) algorithm. The performance of STS as a feature selection metric is evaluated against preliminary survey data collected as a part of a clinical study on persistent post-surgical pain (PPSP). The results suggest that features selected with STS can result in higher performance models compared to traditional feature selection algorithms.

{{</citation>}}


### (43/68) Breaking Language Barriers: A Question Answering Dataset for Hindi and Marathi (Maithili Sabane et al., 2023)

{{<citation>}}

Maithili Sabane, Onkar Litake, Aman Chadha. (2023)  
**Breaking Language Barriers: A Question Answering Dataset for Hindi and Marathi**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Question Answering  
[Paper Link](http://arxiv.org/abs/2308.09862v1)  

---


**ABSTRACT**  
The recent advances in deep-learning have led to the development of highly sophisticated systems with an unquenchable appetite for data. On the other hand, building good deep-learning models for low-resource languages remains a challenging task. This paper focuses on developing a Question Answering dataset for two such languages- Hindi and Marathi. Despite Hindi being the 3rd most spoken language worldwide, with 345 million speakers, and Marathi being the 11th most spoken language globally, with 83.2 million speakers, both languages face limited resources for building efficient Question Answering systems. To tackle the challenge of data scarcity, we have developed a novel approach for translating the SQuAD 2.0 dataset into Hindi and Marathi. We release the largest Question-Answering dataset available for these languages, with each dataset containing 28,000 samples. We evaluate the dataset on various architectures and release the best-performing models for both Hindi and Marathi, which will facilitate further research in these languages. Leveraging similarity tools, our method holds the potential to create datasets in diverse languages, thereby enhancing the understanding of natural language across varied linguistic contexts. Our fine-tuned models, code, and dataset will be made publicly available.

{{</citation>}}


## cs.DC (1)



### (44/68) GNNPipe: Accelerating Distributed Full-Graph GNN Training with Pipelined Model Parallelism (Jingji Chen et al., 2023)

{{<citation>}}

Jingji Chen, Zhuoming Chen, Xuehai Qian. (2023)  
**GNNPipe: Accelerating Distributed Full-Graph GNN Training with Pipelined Model Parallelism**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs.DC  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.10087v1)  

---


**ABSTRACT**  
Current distributed full-graph GNN training methods adopt a variant of data parallelism, namely graph parallelism, in which the whole graph is divided into multiple partitions (subgraphs) and each GPU processes one of them. This incurs high communication overhead because of the inter-partition message passing at each layer. To this end, we proposed a new training method named GNNPipe that adopts model parallelism instead, which has a lower worst-case asymptotic communication complexity than graph parallelism. To ensure high GPU utilization, we proposed to combine model parallelism with a chunk-based pipelined training method, in which each GPU processes a different chunk of graph data at different layers concurrently. We further proposed hybrid parallelism that combines model and graph parallelism when the model-level parallelism is insufficient. We also introduced several tricks to ensure convergence speed and model accuracies to accommodate embedding staleness introduced by pipelining. Extensive experiments show that our method reduces the per-epoch training time by up to 2.45x (on average 2.03x) and reduces the communication volume and overhead by up to 22.51x and 27.21x (on average 10.27x and 14.96x), respectively, while achieving a comparable level of model accuracy and convergence speed compared to graph parallelism.

{{</citation>}}


## cs.SI (2)



### (45/68) Connecting the Dots: Leveraging Social Network Analysis to Understand and Optimize Collaborative Dynamics Within the Global Film Production Network (Mehrdad Maghsoudi et al., 2023)

{{<citation>}}

Mehrdad Maghsoudi, Saeid Aliakbar, Sajjad HabibiPour. (2023)  
**Connecting the Dots: Leveraging Social Network Analysis to Understand and Optimize Collaborative Dynamics Within the Global Film Production Network**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2308.10086v1)  

---


**ABSTRACT**  
In recent years, the global film industry has observed a notable surge in international cooperation and cross-border investments. However, a comprehensive overview of these collaborative investments within the industry is lacking. This study employs social network analysis to delve into the possibilities that lie in collaborative efforts and joint investments within the film sector. The research constructs a network of 150 countries based on shared creative elements in their film productions, comprising over 7800 interconnected links. Employing measures of centrality, certain pivotal nations such as the United States, China, and England emerge as influential nodes, showcasing a strong potential to steer industry growth through collaborative engagement. Through a more detailed exploration involving community identification, distinct clusters centered around thematic commonalities that have converged through joint creative endeavors become evident. For example, the "Global Thrill Seekers" community focuses on action films, whereas the "Cultural-Social Cinema Group" addresses worldwide cultural and social issues. Each of these communities presents distinctive perspectives for international cooperation and the collaborative creation of content. This analysis significantly enhances our understanding of the global film network's structure and dynamics, while concurrently highlighting promising pathways for future investment and collaborative initiatives. The research underscores the critical role of leveraging social network analysis methodologies to optimize informed decision-making concerning collaborative investments, thereby paving the way for anticipatory outcomes. This study not only contributes insights but also serves as a model for investigating data-centric participation within the creative industries.

{{</citation>}}


### (46/68) Explicit Time Embedding Based Cascade Attention Network for Information Popularity Prediction (Xigang Sun et al., 2023)

{{<citation>}}

Xigang Sun, Jingya Zhou, Ling Liu, Wenqi Wei. (2023)  
**Explicit Time Embedding Based Cascade Attention Network for Information Popularity Prediction**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-IR, cs-SI, cs.SI  
Keywords: Attention, Embedding  
[Paper Link](http://arxiv.org/abs/2308.09976v1)  

---


**ABSTRACT**  
Predicting information cascade popularity is a fundamental problem in social networks. Capturing temporal attributes and cascade role information (e.g., cascade graphs and cascade sequences) is necessary for understanding the information cascade. Current methods rarely focus on unifying this information for popularity predictions, which prevents them from effectively modeling the full properties of cascades to achieve satisfactory prediction performances. In this paper, we propose an explicit Time embedding based Cascade Attention Network (TCAN) as a novel popularity prediction architecture for large-scale information networks. TCAN integrates temporal attributes (i.e., periodicity, linearity, and non-linear scaling) into node features via a general time embedding approach (TE), and then employs a cascade graph attention encoder (CGAT) and a cascade sequence attention encoder (CSAT) to fully learn the representation of cascade graphs and cascade sequences. We use two real-world datasets (i.e., Weibo and APS) with tens of thousands of cascade samples to validate our methods. Experimental results show that TCAN obtains mean logarithm squared errors of 2.007 and 1.201 and running times of 1.76 hours and 0.15 hours on both datasets, respectively. Furthermore, TCAN outperforms other representative baselines by 10.4%, 3.8%, and 10.4% in terms of MSLE, MAE, and R-squared on average while maintaining good interpretability.

{{</citation>}}


## cs.CR (2)



### (47/68) Robust Fraud Detection via Supervised Contrastive Learning (Vinay M. S. et al., 2023)

{{<citation>}}

Vinay M. S., Shuhan Yuan, Xintao Wu. (2023)  
**Robust Fraud Detection via Supervised Contrastive Learning**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Contrastive Learning, Fraud Detection  
[Paper Link](http://arxiv.org/abs/2308.10055v1)  

---


**ABSTRACT**  
Deep learning models have recently become popular for detecting malicious user activity sessions in computing platforms. In many real-world scenarios, only a few labeled malicious and a large amount of normal sessions are available. These few labeled malicious sessions usually do not cover the entire diversity of all possible malicious sessions. In many scenarios, possible malicious sessions can be highly diverse. As a consequence, learned session representations of deep learning models can become ineffective in achieving a good generalization performance for unseen malicious sessions. To tackle this open-set fraud detection challenge, we propose a robust supervised contrastive learning based framework called ConRo, which specifically operates in the scenario where only a few malicious sessions having limited diversity is available. ConRo applies an effective data augmentation strategy to generate diverse potential malicious sessions. By employing these generated and available training set sessions, ConRo derives separable representations w.r.t open-set fraud detection task by leveraging supervised contrastive learning. We empirically evaluate our ConRo framework and other state-of-the-art baselines on benchmark datasets. Our ConRo framework demonstrates noticeable performance improvement over state-of-the-art baselines.

{{</citation>}}


### (48/68) East: Efficient and Accurate Secure Transformer Framework for Inference (Yuanchao Ding et al., 2023)

{{<citation>}}

Yuanchao Ding, Hua Guo, Yewei Guan, Weixin Liu, Jiarong Huo, Zhenyu Guan, Xiyong Zhang. (2023)  
**East: Efficient and Accurate Secure Transformer Framework for Inference**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs.CR  
Keywords: BERT, ChatGPT, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2308.09923v1)  

---


**ABSTRACT**  
Transformer has been successfully used in practical applications, such as ChatGPT, due to its powerful advantages. However, users' input is leaked to the model provider during the service. With people's attention to privacy, privacy-preserving Transformer inference is on the demand of such services. Secure protocols for non-linear functions are crucial in privacy-preserving Transformer inference, which are not well studied. Thus, designing practical secure protocols for non-linear functions is hard but significant to model performance. In this work, we propose a framework \emph{East} to enable efficient and accurate secure Transformer inference. Firstly, we propose a new oblivious piecewise polynomial evaluation algorithm and apply it to the activation functions, which reduces the runtime and communication of GELU by over 1.5$\times$ and 2.5$\times$, compared to prior arts. Secondly, the secure protocols for softmax and layer normalization are carefully designed to faithfully maintain the desired functionality. Thirdly, several optimizations are conducted in detail to enhance the overall efficiency. We applied \emph{East} to BERT and the results show that the inference accuracy remains consistent with the plaintext inference without fine-tuning. Compared to Iron, we achieve about 1.8$\times$ lower communication within 1.2$\times$ lower runtime.

{{</citation>}}


## cs.IR (5)



### (49/68) Large Language Models as Zero-Shot Conversational Recommenders (Zhankui He et al., 2023)

{{<citation>}}

Zhankui He, Zhouhang Xie, Rahul Jha, Harald Steck, Dawen Liang, Yesu Feng, Bodhisattwa Prasad Majumder, Nathan Kallus, Julian McAuley. (2023)  
**Large Language Models as Zero-Shot Conversational Recommenders**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.10053v1)  

---


**ABSTRACT**  
In this paper, we present empirical studies on conversational recommendation tasks using representative large language models in a zero-shot setting with three primary contributions. (1) Data: To gain insights into model behavior in "in-the-wild" conversational recommendation scenarios, we construct a new dataset of recommendation-related conversations by scraping a popular discussion website. This is the largest public real-world conversational recommendation dataset to date. (2) Evaluation: On the new dataset and two existing conversational recommendation datasets, we observe that even without fine-tuning, large language models can outperform existing fine-tuned conversational recommendation models. (3) Analysis: We propose various probing tasks to investigate the mechanisms behind the remarkable performance of large language models in conversational recommendation. We analyze both the large language models' behaviors and the characteristics of the datasets, providing a holistic understanding of the models' effectiveness, limitations and suggesting directions for the design of future conversational recommenders

{{</citation>}}


### (50/68) Voucher Abuse Detection with Prompt-based Fine-tuning on Graph Neural Networks (Zhihao Wen et al., 2023)

{{<citation>}}

Zhihao Wen, Yuan Fang, Yihan Liu, Yang Guo, Shuji Hao. (2023)  
**Voucher Abuse Detection with Prompt-based Fine-tuning on Graph Neural Networks**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.10028v1)  

---


**ABSTRACT**  
Voucher abuse detection is an important anomaly detection problem in E-commerce. While many GNN-based solutions have emerged, the supervised paradigm depends on a large quantity of labeled data. A popular alternative is to adopt self-supervised pre-training using label-free data, and further fine-tune on a downstream task with limited labels. Nevertheless, the "pre-train, fine-tune" paradigm is often plagued by the objective gap between pre-training and downstream tasks. Hence, we propose VPGNN, a prompt-based fine-tuning framework on GNNs for voucher abuse detection. We design a novel graph prompting function to reformulate the downstream task into a similar template as the pretext task in pre-training, thereby narrowing the objective gap. Extensive experiments on both proprietary and public datasets demonstrate the strength of VPGNN in both few-shot and semi-supervised scenarios. Moreover, an online deployment of VPGNN in a production environment shows a 23.4% improvement over two existing deployed models.

{{</citation>}}


### (51/68) printf: Preference Modeling Based on User Reviews with Item Images and Textual Information via Graph Learning (Hao-Lun Lin et al., 2023)

{{<citation>}}

Hao-Lun Lin, Jyun-Yu Jiang, Ming-Hao Juan, Pu-Jen Cheng. (2023)  
**printf: Preference Modeling Based on User Reviews with Item Images and Textual Information via Graph Learning**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2308.09943v1)  

---


**ABSTRACT**  
Nowadays, modern recommender systems usually leverage textual and visual contents as auxiliary information to predict user preference. For textual information, review texts are one of the most popular contents to model user behaviors. Nevertheless, reviews usually lose their shine when it comes to top-N recommender systems because those that solely utilize textual reviews as features struggle to adequately capture the interaction relationships between users and items. For visual one, it is usually modeled with naive convolutional networks and also hard to capture high-order relationships between users and items. Moreover, previous works did not collaboratively use both texts and images in a proper way. In this paper, we propose printf, preference modeling based on user reviews with item images and textual information via graph learning, to address the above challenges. Specifically, the dimension-based attention mechanism directs relations between user reviews and interacted items, allowing each dimension to contribute different importance weights to derive user representations. Extensive experiments are conducted on three publicly available datasets. The experimental results demonstrate that our proposed printf consistently outperforms baseline methods with the relative improvements for NDCG@5 of 26.80%, 48.65%, and 25.74% on Amazon-Grocery, Amazon-Tools, and Amazon-Electronics datasets, respectively. The in-depth analysis also indicates the dimensions of review representations definitely have different topics and aspects, assisting the validity of our model design.

{{</citation>}}


### (52/68) RAH! RecSys-Assistant-Human: A Human-Central Recommendation Framework with Large Language Models (Yubo Shu et al., 2023)

{{<citation>}}

Yubo Shu, Hansu Gu, Peng Zhang, Haonan Zhang, Tun Lu, Dongsheng Li, Ning Gu. (2023)  
**RAH! RecSys-Assistant-Human: A Human-Central Recommendation Framework with Large Language Models**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.09904v1)  

---


**ABSTRACT**  
The recommendation ecosystem involves interactions between recommender systems(Computer) and users(Human). Orthogonal to the perspective of recommender systems, we attempt to utilize LLMs from the perspective of users and propose a more human-central recommendation framework named RAH, which consists of Recommender system, Assistant and Human. The assistant is a LLM-based and personal proxy for a human to achieve user satisfaction. The assistant plays a non-invasion role and the RAH framework can adapt to different recommender systems and user groups. Subsequently, we implement and evaluate the RAH framework for learning user personalities and proxy human feedback. The experiment shows that (1) using learn-action-critic and reflection mechanisms can lead more aligned personality and (2) our assistant can effectively proxy human feedback and help adjust recommender systems. Finally, we discuss further strategies in the RAH framework to address human-central concerns including user control, privacy and fairness.

{{</citation>}}


### (53/68) Black-box Adversarial Attacks against Dense Retrieval Models: A Multi-view Contrastive Learning Method (Yu-An Liu et al., 2023)

{{<citation>}}

Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Yixing Fan, Xueqi Cheng. (2023)  
**Black-box Adversarial Attacks against Dense Retrieval Models: A Multi-view Contrastive Learning Method**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Adversarial Attack, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.09861v1)  

---


**ABSTRACT**  
Neural ranking models (NRMs) and dense retrieval (DR) models have given rise to substantial improvements in overall retrieval performance. In addition to their effectiveness, and motivated by the proven lack of robustness of deep learning-based approaches in other areas, there is growing interest in the robustness of deep learning-based approaches to the core retrieval problem. Adversarial attack methods that have so far been developed mainly focus on attacking NRMs, with very little attention being paid to the robustness of DR models. In this paper, we introduce the adversarial retrieval attack (AREA) task. The AREA task is meant to trick DR models into retrieving a target document that is outside the initial set of candidate documents retrieved by the DR model in response to a query. We consider the decision-based black-box adversarial setting, which is realistic in real-world search engines. To address the AREA task, we first employ existing adversarial attack methods designed for NRMs. We find that the promising results that have previously been reported on attacking NRMs, do not generalize to DR models: these methods underperform a simple term spamming method. We attribute the observed lack of generalizability to the interaction-focused architecture of NRMs, which emphasizes fine-grained relevance matching. DR models follow a different representation-focused architecture that prioritizes coarse-grained representations. We propose to formalize attacks on DR models as a contrastive learning problem in a multi-view representation space. The core idea is to encourage the consistency between each view representation of the target document and its corresponding viewer via view-wise supervision signals. Experimental results demonstrate that the proposed method can significantly outperform existing attack strategies in misleading the DR model with small indiscernible text perturbations.

{{</citation>}}


## cs.RO (3)



### (54/68) Towards Probabilistic Causal Discovery, Inference & Explanations for Autonomous Drones in Mine Surveying Tasks (Ricardo Cannizzaro et al., 2023)

{{<citation>}}

Ricardo Cannizzaro, Rhys Howard, Paulina Lewinska, Lars Kunze. (2023)  
**Towards Probabilistic Causal Discovery, Inference & Explanations for Autonomous Drones in Mine Surveying Tasks**  

---
Primary Category: cs.RO  
Categories: I-2-9; I-2-6; G-3; I-6-3; I-2-8; J-7, cs-AI, cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2308.10047v1)  

---


**ABSTRACT**  
Causal modelling offers great potential to provide autonomous agents the ability to understand the data-generation process that governs their interactions with the world. Such models capture formal knowledge as well as probabilistic representations of noise and uncertainty typically encountered by autonomous robots in real-world environments. Thus, causality can aid autonomous agents in making decisions and explaining outcomes, but deploying causality in such a manner introduces new challenges. Here we identify challenges relating to causality in the context of a drone system operating in a salt mine. Such environments are challenging for autonomous agents because of the presence of confounders, non-stationarity, and a difficulty in building complete causal models ahead of time. To address these issues, we propose a probabilistic causal framework consisting of: causally-informed POMDP planning, online SCM adaptation, and post-hoc counterfactual explanations. Further, we outline planned experimentation to evaluate the framework integrated with a drone system in simulated mine environments and on a real-world mine dataset.

{{</citation>}}


### (55/68) Forecast-MAE: Self-supervised Pre-training for Motion Forecasting with Masked Autoencoders (Jie Cheng et al., 2023)

{{<citation>}}

Jie Cheng, Xiaodong Mei, Ming Liu. (2023)  
**Forecast-MAE: Self-supervised Pre-training for Motion Forecasting with Masked Autoencoders**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.09882v1)  

---


**ABSTRACT**  
This study explores the application of self-supervised learning (SSL) to the task of motion forecasting, an area that has not yet been extensively investigated despite the widespread success of SSL in computer vision and natural language processing. To address this gap, we introduce Forecast-MAE, an extension of the mask autoencoders framework that is specifically designed for self-supervised learning of the motion forecasting task. Our approach includes a novel masking strategy that leverages the strong interconnections between agents' trajectories and road networks, involving complementary masking of agents' future or history trajectories and random masking of lane segments. Our experiments on the challenging Argoverse 2 motion forecasting benchmark show that Forecast-MAE, which utilizes standard Transformer blocks with minimal inductive bias, achieves competitive performance compared to state-of-the-art methods that rely on supervised learning and sophisticated designs. Moreover, it outperforms the previous self-supervised learning method by a significant margin. Code is available at https://github.com/jchengai/forecast-mae.

{{</citation>}}


### (56/68) Learning Soft Robot Dynamics using Differentiable Kalman Filters and Spatio-Temporal Embeddings (Xiao Liu et al., 2023)

{{<citation>}}

Xiao Liu, Shuhei Ikemoto, Yuhei Yoshimitsu, Heni Ben Amor. (2023)  
**Learning Soft Robot Dynamics using Differentiable Kalman Filters and Spatio-Temporal Embeddings**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.09868v1)  

---


**ABSTRACT**  
This paper introduces a novel approach for modeling the dynamics of soft robots, utilizing a differentiable filter architecture. The proposed approach enables end-to-end training to learn system dynamics, noise characteristics, and temporal behavior of the robot. A novel spatio-temporal embedding process is discussed to handle observations with varying sensor placements and sampling frequencies. The efficacy of this approach is demonstrated on a tensegrity robot arm by learning end-effector dynamics from demonstrations with complex bending motions. The model is proven to be robust against missing modalities, diverse sensor placement, and varying sampling rates. Additionally, the proposed framework is shown to identify physical interactions with humans during motion. The utilization of a differentiable filter presents a novel solution to the difficulties of modeling soft robot dynamics. Our approach shows substantial improvement in accuracy compared to state-of-the-art filtering methods, with at least a 24% reduction in mean absolute error (MAE) observed. Furthermore, the predicted end-effector positions show an average MAE of 25.77mm from the ground truth, highlighting the advantage of our approach. The code is available at https://github.com/ir-lab/soft_robot_DEnKF.

{{</citation>}}


## cs.SE (4)



### (57/68) Cupid: Leveraging ChatGPT for More Accurate Duplicate Bug Report Detection (Ting Zhang et al., 2023)

{{<citation>}}

Ting Zhang, Ivana Clairine Irsan, Ferdian Thung, David Lo. (2023)  
**Cupid: Leveraging ChatGPT for More Accurate Duplicate Bug Report Detection**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.10022v1)  

---


**ABSTRACT**  
Duplicate bug report detection (DBRD) is a long-standing challenge in both academia and industry. Over the past decades, researchers have proposed various approaches to detect duplicate bug reports more accurately. With the recent advancement of deep learning, researchers have also proposed several approaches that leverage deep learning models to detect duplicate bug reports. A recent benchmarking study on DBRD also reveals that the performance of deep learning-based approaches is not always better than the traditional approaches. However, traditional approaches have limitations, e.g., they are usually based on the bag-of-words model, which cannot capture the semantics of bug reports. To address these aforementioned challenges, we seek to leverage state-of-the-art large language model to improve the performance of the traditional DBRD approach.   In this paper, we propose an approach called Cupid, which combines the best-performing traditional DBRD approach REP with the state-of-the-art large language model ChatGPT. Specifically, we first leverage ChatGPT under the zero-shot setting to get essential information on bug reports. We then use the essential information as the input of REP to detect duplicate bug reports. We conducted an evaluation on comparing Cupid with three existing approaches on three datasets. The experimental results show that Cupid achieves new state-of-the-art results, reaching Recall Rate@10 scores ranging from 0.59 to 0.67 across all the datasets analyzed. Our work highlights the potential of combining large language models to improve the performance of software engineering tasks.

{{</citation>}}


### (58/68) Evaluating Transfer Learning for Simplifying GitHub READMEs (Haoyu Gao et al., 2023)

{{<citation>}}

Haoyu Gao, Christoph Treude, Mansooreh Zahedi. (2023)  
**Evaluating Transfer Learning for Simplifying GitHub READMEs**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BLEU, Transformer  
[Paper Link](http://arxiv.org/abs/2308.09940v1)  

---


**ABSTRACT**  
Software documentation captures detailed knowledge about a software product, e.g., code, technologies, and design. It plays an important role in the coordination of development teams and in conveying ideas to various stakeholders. However, software documentation can be hard to comprehend if it is written with jargon and complicated sentence structure. In this study, we explored the potential of text simplification techniques in the domain of software engineering to automatically simplify GitHub README files. We collected software-related pairs of GitHub README files consisting of 14,588 entries, aligned difficult sentences with their simplified counterparts, and trained a Transformer-based model to automatically simplify difficult versions. To mitigate the sparse and noisy nature of the software-related simplification dataset, we applied general text simplification knowledge to this field. Since many general-domain difficult-to-simple Wikipedia document pairs are already publicly available, we explored the potential of transfer learning by first training the model on the Wikipedia data and then fine-tuning it on the README data. Using automated BLEU scores and human evaluation, we compared the performance of different transfer learning schemes and the baseline models without transfer learning. The transfer learning model using the best checkpoint trained on a general topic corpus achieved the best performance of 34.68 BLEU score and statistically significantly higher human annotation scores compared to the rest of the schemes and baselines. We conclude that using transfer learning is a promising direction to circumvent the lack of data and drift style problem in software README files simplification and achieved a better trade-off between simplification and preservation of meaning.

{{</citation>}}


### (59/68) Practical Anomaly Detection over Multivariate Monitoring Metrics for Online Services (Jinyang Liu et al., 2023)

{{<citation>}}

Jinyang Liu, Tianyi Yang, Zhuangbin Chen, Yuxin Su, Cong Feng, Zengyin Yang, Michael R. Lyu. (2023)  
**Practical Anomaly Detection over Multivariate Monitoring Metrics for Online Services**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.09937v1)  

---


**ABSTRACT**  
As modern software systems continue to grow in terms of complexity and volume, anomaly detection on multivariate monitoring metrics, which profile systems' health status, becomes more and more critical and challenging. In particular, the dependency between different metrics and their historical patterns plays a critical role in pursuing prompt and accurate anomaly detection. Existing approaches fall short of industrial needs for being unable to capture such information efficiently. To fill this significant gap, in this paper, we propose CMAnomaly, an anomaly detection framework on multivariate monitoring metrics based on collaborative machine. The proposed collaborative machine is a mechanism to capture the pairwise interactions along with feature and temporal dimensions with linear time complexity. Cost-effective models can then be employed to leverage both the dependency between monitoring metrics and their historical patterns for anomaly detection. The proposed framework is extensively evaluated with both public data and industrial data collected from a large-scale online service system of Huawei Cloud. The experimental results demonstrate that compared with state-of-the-art baseline models, CMAnomaly achieves an average F1 score of 0.9494, outperforming baselines by 6.77% to 10.68%, and runs 10X to 20X faster. Furthermore, we also share our experience of deploying CMAnomaly in Huawei Cloud.

{{</citation>}}


### (60/68) What Do Code Models Memorize? An Empirical Study on Large Language Models of Code (Zhou Yang et al., 2023)

{{<citation>}}

Zhou Yang, Zhipeng Zhao, Chenyu Wang, Jieke Shi, Dongsun Kim, DongGyun Han, David Lo. (2023)  
**What Do Code Models Memorize? An Empirical Study on Large Language Models of Code**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.09932v1)  

---


**ABSTRACT**  
The availability of large-scale datasets, advanced architectures, and powerful computational resources have led to effective code models that automate diverse software engineering activities. The datasets usually consist of billions of lines of code from both open-source and private repositories. A code model memorizes and produces source code verbatim, which potentially contains vulnerabilities, sensitive information, or code with strict licenses, leading to potential security and privacy issues.   This paper investigates an important problem: to what extent do code models memorize their training data? We conduct an empirical study to explore memorization in large pre-trained code models. Our study highlights that simply extracting 20,000 outputs (each having 512 tokens) from a code model can produce over 40,125 code snippets that are memorized from the training data. To provide a better understanding, we build a taxonomy of memorized contents with 3 categories and 14 subcategories. The results show that the prompts sent to the code models affect the distribution of memorized contents. We identify several key factors of memorization. Specifically, given the same architecture, larger models suffer more from memorization problems. A code model produces more memorization when it is allowed to generate longer outputs. We also find a strong positive correlation between the number of an output's occurrences in the training data and that in the generated outputs, which indicates that a potential way to reduce memorization is to remove duplicates in the training data. We then identify effective metrics that infer whether an output contains memorization accurately. We also make some suggestions regarding dealing with memorization in code models.

{{</citation>}}


## cs.CY (1)



### (61/68) Artificial Intelligence across Europe: A Study on Awareness, Attitude and Trust (Teresa Scantamburlo et al., 2023)

{{<citation>}}

Teresa Scantamburlo, Atia Cortés, Francesca Foffano, Cristian Barrué, Veronica Distefano, Long Pham, Alessandro Fabris. (2023)  
**Artificial Intelligence across Europe: A Study on Awareness, Attitude and Trust**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09979v1)  

---


**ABSTRACT**  
This paper presents the results of an extensive study investigating the opinions on Artificial Intelligence (AI) of a sample of 4,006 European citizens from eight distinct countries (France, Germany, Italy, Netherlands, Poland, Romania, Spain, and Sweden). The aim of the study is to gain a better understanding of people's views and perceptions within the European context, which is already marked by important policy actions and regulatory processes. To survey the perceptions of the citizens of Europe we design and validate a new questionnaire (PAICE) structured around three dimensions: people's awareness, attitude, and trust. We observe that while awareness is characterized by a low level of self-assessed competency, the attitude toward AI is very positive for more than half of the population. Reflecting upon the collected results, we highlight implicit contradictions and identify trends that may interfere with the creation of an ecosystem of trust and the development of inclusive AI policies. The introduction of rules that ensure legal and ethical standards, along with the activity of high-level educational entities, and the promotion of AI literacy are identified as key factors in supporting a trustworthy AI ecosystem. We make some recommendations for AI governance focused on the European context and conclude with suggestions for future work.

{{</citation>}}


## cs.SD (1)



### (62/68) Spatial Reconstructed Local Attention Res2Net with F0 Subband for Fake Speech Detection (Cunhang Fan et al., 2023)

{{<citation>}}

Cunhang Fan, Jun Xue, Jianhua Tao, Jiangyan Yi, Chenglong Wang, Chengshi Zheng, Zhao Lv. (2023)  
**Spatial Reconstructed Local Attention Res2Net with F0 Subband for Fake Speech Detection**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.09944v1)  

---


**ABSTRACT**  
The rhythm of synthetic speech is usually too smooth, which causes that the fundamental frequency (F0) of synthetic speech is significantly different from that of real speech. It is expected that the F0 feature contains the discriminative information for the fake speech detection (FSD) task. In this paper, we propose a novel F0 subband for FSD. In addition, to effectively model the F0 subband so as to improve the performance of FSD, the spatial reconstructed local attention Res2Net (SR-LA Res2Net) is proposed. Specifically, Res2Net is used as a backbone network to obtain multiscale information, and enhanced with a spatial reconstruction mechanism to avoid losing important information when the channel group is constantly superimposed. In addition, local attention is designed to make the model focus on the local information of the F0 subband. Experimental results on the ASVspoof 2019 LA dataset show that our proposed method obtains an equal error rate (EER) of 0.47% and a minimum tandem detection cost function (min t-DCF) of 0.0159, achieving the state-of-the-art performance among all of the single systems.

{{</citation>}}


## cs.IT (1)



### (63/68) Robust Train-to-Train Transmission Scheduling in mmWave Band for High Speed Train Communication Systems (Yunhan Ma et al., 2023)

{{<citation>}}

Yunhan Ma, Yong Niu, Shiwen Mao, Zhu Han, Ruisi He, Zhangdui Zhong, Ning Wang, Bo Ai. (2023)  
**Robust Train-to-Train Transmission Scheduling in mmWave Band for High Speed Train Communication Systems**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-NI, cs.IT, math-IT  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.09926v1)  

---


**ABSTRACT**  
Demands for data traffic in high-speed railway (HSR) has increased drastically. The increasing entertainment needs of passengers, safety control information exchanges of trains, etc., make train-to-train (T2T) communications face the challenge of achieving high-capacity and high-quality data transmissions. In order to greatly increase the communication capacity, it is urgent to introduce millimeter wave (mmWave) technology. Faced with the problem that mmWave link is easy to be blocked, this paper leverages the existing equipment to assist relay, and proposes an effective transmission scheduling scheme to improve the robustness of T2T communication systems. First of all, we formulate a mixed integer nonlinear programming (MINLP) optimization problem the transmission scheduling in T2T communication systems where mobile relays (MRs) are all working in the full-duplex (FD) mode. Then we propose a low complexity heuristic algorithm to solve the optimization problem, which consists of three components: relay selection, transmission mode selection, and transmission scheduling. The simulation results show that the proposed algorithm can greatly improve the number of completed flows and system throughput. Finally, we analyze the influence of different design parameters on the system performance. The results show that the proposed algorithm can achieve more data flows and system throughput within a reasonable communication distance threshold in T2T communication with obstacles in different orbits. It can balance the computational complexity and system performance to achieve an efficient and robust data transmission.

{{</citation>}}


## cs.HC (3)



### (64/68) An Autoethnographic Case Study of Generative Artificial Intelligence's Utility for Accessibility (Kate S Glazko et al., 2023)

{{<citation>}}

Kate S Glazko, Momona Yamagami, Aashaka Desai, Kelly Avery Mack, Venkatesh Potluri, Xuhai Xu, Jennifer Mankoff. (2023)  
**An Autoethnographic Case Study of Generative Artificial Intelligence's Utility for Accessibility**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09924v1)  

---


**ABSTRACT**  
With the recent rapid rise in Generative Artificial Intelligence (GAI) tools, it is imperative that we understand their impact on people with disabilities, both positive and negative. However, although we know that AI in general poses both risks and opportunities for people with disabilities, little is known specifically about GAI in particular. To address this, we conducted a three-month autoethnography of our use of GAI to meet personal and professional needs as a team of researchers with and without disabilities. Our findings demonstrate a wide variety of potential accessibility-related uses for GAI while also highlighting concerns around verifiability, training data, ableism, and false promises.

{{</citation>}}


### (65/68) User-centric AIGC products: Explainable Artificial Intelligence and AIGC products (Hanjie Yu et al., 2023)

{{<citation>}}

Hanjie Yu, Yan Dong, Qiong Wu. (2023)  
**User-centric AIGC products: Explainable Artificial Intelligence and AIGC products**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2308.09877v1)  

---


**ABSTRACT**  
Generative AI tools, such as ChatGPT and Midjourney, are transforming artistic creation as AI-art integration advances. However, Artificial Intelligence Generated Content (AIGC) tools face user experience challenges, necessitating a human-centric design approach. This paper offers a brief overview of research on explainable AI (XAI) and user experience, examining factors leading to suboptimal experiences with AIGC tools. Our proposed solution integrates interpretable AI methodologies into the input and adjustment feedback stages of AIGC products. We underscore XAI's potential to enhance the user experience for ordinary users and present a conceptual framework for improving AIGC user experience.

{{</citation>}}


### (66/68) Characterizing Usability Issue Discussions in OSS Projects (Arghavan Sanei et al., 2023)

{{<citation>}}

Arghavan Sanei, Jinghui Cheng. (2023)  
**Characterizing Usability Issue Discussions in OSS Projects**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-SE, cs.HC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.09876v1)  

---


**ABSTRACT**  
Usability is a crucial factor but one of the most neglected concerns in open source software (OSS). While far from an ideal approach, a common practice that OSS communities adopt to collaboratively address usability is through discussions on issue tracking systems (ITSs). However, there is little knowledge about the extent to which OSS community members engage in usability issue discussions, the aspects of usability they frequently target, and the characteristics of their collaboration around usability issue discussions. This knowledge is important for providing practical recommendations and research directions to better support OSS communities in addressing this important topic and improve OSS usability in general. To help achieve this goal, we performed an extensive empirical study on issues discussed in five popular OSS applications: three data science notebook projects (Jupyter Lab, Google Colab, and CoCalc) and two code editor projects (VSCode and Atom). Our results indicated that while usability issues are extensively discussed in the OSS projects, their scope tended to be limited to efficiency and aesthetics. Additionally, these issues are more frequently posted by experienced community members and display distinguishable characteristics, such as involving more visual communication and more participants. Our results provide important implications that can inform the OSS practitioners to better engage the community in usability issue discussion and shed light on future research efforts toward collaboration techniques and tools for discussing niche topics in diverse communities, such as the usability issues in the OSS context.

{{</citation>}}


## cs.PL (1)



### (67/68) Knowledge Transfer from High-Resource to Low-Resource Programming Languages for Code LLMs (Federico Cassano et al., 2023)

{{<citation>}}

Federico Cassano, John Gouwar, Francesca Lucchetti, Claire Schlesinger, Carolyn Jane Anderson, Michael Greenberg, Abhinav Jangda, Arjun Guha. (2023)  
**Knowledge Transfer from High-Resource to Low-Resource Programming Languages for Code LLMs**  

---
Primary Category: cs.PL  
Categories: cs-LG, cs-PL, cs.PL  
Keywords: Language Model, Low-Resource  
[Paper Link](http://arxiv.org/abs/2308.09895v1)  

---


**ABSTRACT**  
Over the past few years, Large Language Models of Code (Code LLMs) have started to have a significant impact on programming practice. Code LLMs are also emerging as a building block for research in programming languages and software engineering. However, the quality of code produced by a Code LLM varies significantly by programming languages. Code LLMs produce impressive results on programming languages that are well represented in their training data (e.g., Java, Python, or JavaScript), but struggle with low-resource languages, like OCaml and Racket.   This paper presents an effective approach for boosting the performance of Code LLMs on low-resource languages using semi-synthetic data. Our approach generates high-quality datasets for low-resource languages, which can then be used to fine-tune any pretrained Code LLM. Our approach, called MultiPL-T, translates training data from high-resource languages into training data for low-resource languages. We apply our approach to generate tens of thousands of new, validated training items for Racket, OCaml, and Lua from Python. Moreover, we use an open dataset (The Stack) and model (StarCoderBase), which allow us to decontaminate benchmarks and train models on this data without violating the model license.   With MultiPL-T generated data, we present fine-tuned versions of StarCoderBase that achieve state-of-the-art performance for Racket, OCaml, and Lua on benchmark problems. For Lua, our fine-tuned model achieves the same performance as StarCoderBase as Python -- a very high-resource language -- on the MultiPL-E benchmarks. For Racket and OCaml, we double their performance on MultiPL-E, bringing their performance close to higher-resource languages such as Ruby and C#.

{{</citation>}}


## eess.SY (1)



### (68/68) An Observer-Based Reinforcement Learning Solution for Model-Following Problems (Mohammed I. Abouheaf et al., 2023)

{{<citation>}}

Mohammed I. Abouheaf, Kyriakos G. Vamvoudakis, Mohammad A. Mayyas, Hashim A. Hashim. (2023)  
**An Observer-Based Reinforcement Learning Solution for Model-Following Problems**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.09872v1)  

---


**ABSTRACT**  
In this paper, a multi-objective model-following control problem is solved using an observer-based adaptive learning scheme. The overall goal is to regulate the model-following error dynamics along with optimizing the dynamic variables of a process in a model-free fashion. This solution employs an integral reinforcement learning approach to adapt three strategies. The first strategy observes the states of desired process dynamics, while the second one stabilizes and optimizes the closed-loop system. The third strategy allows the process to follow a desired reference-trajectory. The adaptive learning scheme is implemented using an approximate projection estimation approach under mild conditions about the learning parameters.

{{</citation>}}
