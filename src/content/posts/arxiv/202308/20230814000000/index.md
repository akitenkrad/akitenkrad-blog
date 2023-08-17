---
draft: false
title: "arXiv @ 2023.08.14"
date: 2023-08-14
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.14"
    identifier: arxiv_20230814
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.SE (2)](#csse-2)
- [cs.CV (10)](#cscv-10)
- [quant-ph (1)](#quant-ph-1)
- [cs.CR (2)](#cscr-2)
- [cs.LG (9)](#cslg-9)
- [cs.CL (11)](#cscl-11)
- [cs.RO (1)](#csro-1)
- [cs.NE (1)](#csne-1)
- [cs.IR (1)](#csir-1)
- [eess.AS (1)](#eessas-1)
- [cs.AI (2)](#csai-2)
- [cs.IT (1)](#csit-1)
- [cs.NI (1)](#csni-1)
- [cs.SD (1)](#cssd-1)
- [cs.SI (1)](#cssi-1)

## cs.SE (2)



### (1/45) Smart Knowledge Transfer using Google-like Search (Srijoni Majumdar et al., 2023)

{{<citation>}}

Srijoni Majumdar, Partha Pratim Das. (2023)  
**Smart Knowledge Transfer using Google-like Search**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.06653v1)  

---


**ABSTRACT**  
To address the issue of rising software maintenance cost due to program comprehension challenges, we propose SMARTKT (Smart Knowledge Transfer), a search framework, which extracts and integrates knowledge related to various aspects of an application in form of a semantic graph. This graph supports syntax and semantic queries and converts the process of program comprehension into a {\em google-like} search problem.

{{</citation>}}


### (2/45) Copilot Security: A User Study (Owura Asare et al., 2023)

{{<citation>}}

Owura Asare, Meiyappan Nagappan, N. Asokan. (2023)  
**Copilot Security: A User Study**  

---
Primary Category: cs.SE  
Categories: cs-CR, cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.06587v1)  

---


**ABSTRACT**  
Code generation tools driven by artificial intelligence have recently become more popular due to advancements in deep learning and natural language processing that have increased their capabilities. The proliferation of these tools may be a double-edged sword because while they can increase developer productivity by making it easier to write code, research has shown that they can also generate insecure code. In this paper, we perform a user-centered evaluation GitHub's Copilot to better understand its strengths and weaknesses with respect to code security. We conduct a user study where participants solve programming problems, which have potentially vulnerable solutions, with and without Copilot assistance. The main goal of the user study is to determine how the use of Copilot affects participants' security performance. In our set of participants (n=25), we find that access to Copilot accompanies a more secure solution when tackling harder problems. For the easier problem, we observe no effect of Copilot access on the security of solutions. We also observe no disproportionate impact of Copilot use on particular kinds of vulnerabilities.

{{</citation>}}


## cs.CV (10)



### (3/45) 3DMOTFormer: Graph Transformer for Online 3D Multi-Object Tracking (Shuxiao Ding et al., 2023)

{{<citation>}}

Shuxiao Ding, Eike Rehder, Lukas Schneider, Marius Cordts, Juergen Gall. (2023)  
**3DMOTFormer: Graph Transformer for Online 3D Multi-Object Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.06635v1)  

---


**ABSTRACT**  
Tracking 3D objects accurately and consistently is crucial for autonomous vehicles, enabling more reliable downstream tasks such as trajectory prediction and motion planning. Based on the substantial progress in object detection in recent years, the tracking-by-detection paradigm has become a popular choice due to its simplicity and efficiency. State-of-the-art 3D multi-object tracking (MOT) approaches typically rely on non-learned model-based algorithms such as Kalman Filter but require many manually tuned parameters. On the other hand, learning-based approaches face the problem of adapting the training to the online setting, leading to inevitable distribution mismatch between training and inference as well as suboptimal performance. In this work, we propose 3DMOTFormer, a learned geometry-based 3D MOT framework building upon the transformer architecture. We use an Edge-Augmented Graph Transformer to reason on the track-detection bipartite graph frame-by-frame and conduct data association via edge classification. To reduce the distribution mismatch between training and inference, we propose a novel online training strategy with an autoregressive and recurrent forward pass as well as sequential batch optimization. Using CenterPoint detections, our approach achieves 71.2% and 68.2% AMOTA on the nuScenes validation and test split, respectively. In addition, a trained 3DMOTFormer model generalizes well across different object detectors. Code is available at: https://github.com/dsx0511/3DMOTFormer.

{{</citation>}}


### (4/45) DFM-X: Augmentation by Leveraging Prior Knowledge of Shortcut Learning (Shunxin Wang et al., 2023)

{{<citation>}}

Shunxin Wang, Christoph Brune, Raymond Veldhuis, Nicola Strisciuglio. (2023)  
**DFM-X: Augmentation by Leveraging Prior Knowledge of Shortcut Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.06622v1)  

---


**ABSTRACT**  
Neural networks are prone to learn easy solutions from superficial statistics in the data, namely shortcut learning, which impairs generalization and robustness of models. We propose a data augmentation strategy, named DFM-X, that leverages knowledge about frequency shortcuts, encoded in Dominant Frequencies Maps computed for image classification models. We randomly select X% training images of certain classes for augmentation, and process them by retaining the frequencies included in the DFMs of other classes. This strategy compels the models to leverage a broader range of frequencies for classification, rather than relying on specific frequency sets. Thus, the models learn more deep and task-related semantics compared to their counterpart trained with standard setups. Unlike other commonly used augmentation techniques which focus on increasing the visual variations of training data, our method targets exploiting the original data efficiently, by distilling prior knowledge about destructive learning behavior of models from data. Our experimental results demonstrate that DFM-X improves robustness against common corruptions and adversarial attacks. It can be seamlessly integrated with other augmentation techniques to further enhance the robustness of models.

{{</citation>}}


### (5/45) LadleNet: Translating Thermal Infrared Images to Visible Light Images Using A Scalable Two-stage U-Net (Tonghui Zou, 2023)

{{<citation>}}

Tonghui Zou. (2023)  
**LadleNet: Translating Thermal Infrared Images to Visible Light Images Using A Scalable Two-stage U-Net**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.06603v1)  

---


**ABSTRACT**  
The translation of thermal infrared (TIR) images to visible light (VI) images presents a challenging task with potential applications spanning various domains such as TIR-VI image registration and fusion. Leveraging supplementary information derived from TIR image conversions can significantly enhance model performance and generalization across these applications. However, prevailing issues within this field include suboptimal image fidelity and limited model scalability. In this paper, we introduce an algorithm, LadleNet, based on the U-Net architecture. LadleNet employs a two-stage U-Net concatenation structure, augmented with skip connections and refined feature aggregation techniques, resulting in a substantial enhancement in model performance. Comprising 'Handle' and 'Bowl' modules, LadleNet's Handle module facilitates the construction of an abstract semantic space, while the Bowl module decodes this semantic space to yield mapped VI images. The Handle module exhibits extensibility by allowing the substitution of its network architecture with semantic segmentation networks, thereby establishing more abstract semantic spaces to bolster model performance. Consequently, we propose LadleNet+, which replaces LadleNet's Handle module with the pre-trained DeepLabv3+ network, thereby endowing the model with enhanced semantic space construction capabilities. The proposed method is evaluated and tested on the KAIST dataset, accompanied by quantitative and qualitative analyses. Compared to existing methodologies, our approach achieves state-of-the-art performance in terms of image clarity and perceptual quality. The source code will be made available at https://github.com/Ach-1914/LadleNet/tree/main/.

{{</citation>}}


### (6/45) Revisiting Vision Transformer from the View of Path Ensemble (Shuning Chang et al., 2023)

{{<citation>}}

Shuning Chang, Pichao Wang, Hao Luo, Fan Wang, Mike Zheng Shou. (2023)  
**Revisiting Vision Transformer from the View of Path Ensemble**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.06548v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) are normally regarded as a stack of transformer layers. In this work, we propose a novel view of ViTs showing that they can be seen as ensemble networks containing multiple parallel paths with different lengths. Specifically, we equivalently transform the traditional cascade of multi-head self-attention (MSA) and feed-forward network (FFN) into three parallel paths in each transformer layer. Then, we utilize the identity connection in our new transformer form and further transform the ViT into an explicit multi-path ensemble network. From the new perspective, these paths perform two functions: the first is to provide the feature for the classifier directly, and the second is to provide the lower-level feature representation for subsequent longer paths. We investigate the influence of each path for the final prediction and discover that some paths even pull down the performance. Therefore, we propose the path pruning and EnsembleScale skills for improvement, which cut out the underperforming paths and re-weight the ensemble components, respectively, to optimize the path combination and make the short paths focus on providing high-quality representation for subsequent paths. We also demonstrate that our path combination strategies can help ViTs go deeper and act as high-pass filters to filter out partial low-frequency signals. To further enhance the representation of paths served for subsequent paths, self-distillation is applied to transfer knowledge from the long paths to the short paths. This work calls for more future research to explain and design ViTs from new perspectives.

{{</citation>}}


### (7/45) Dealing with Small Annotated Datasets for Deep Learning in Medical Imaging: An Evaluation of Self-Supervised Pre-Training on CT Scans Comparing Contrastive and Masked Autoencoder Methods for Convolutional Models (Daniel Wolf et al., 2023)

{{<citation>}}

Daniel Wolf, Tristan Payer, Catharina Silvia Lisson, Christoph Gerhard Lisson, Meinrad Beer, Timo Ropinski, Michael Götz. (2023)  
**Dealing with Small Annotated Datasets for Deep Learning in Medical Imaging: An Evaluation of Self-Supervised Pre-Training on CT Scans Comparing Contrastive and Masked Autoencoder Methods for Convolutional Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.06534v1)  

---


**ABSTRACT**  
Deep learning in medical imaging has the potential to minimize the risk of diagnostic errors, reduce radiologist workload, and accelerate diagnosis. Training such deep learning models requires large and accurate datasets, with annotations for all training samples. However, in the medical imaging domain, annotated datasets for specific tasks are often small due to the high complexity of annotations, limited access, or the rarity of diseases. To address this challenge, deep learning models can be pre-trained on large image datasets without annotations using methods from the field of self-supervised learning. After pre-training, small annotated datasets are sufficient to fine-tune the models for a specific task, the so-called ``downstream task". The most popular self-supervised pre-training approaches in medical imaging are based on contrastive learning. However, recent studies in natural image processing indicate a strong potential for masked autoencoder approaches. Our work compares state-of-the-art contrastive learning methods with the recently introduced masked autoencoder approach "SparK" for convolutional neural networks (CNNs) on medical images. Therefore we pre-train on a large unannotated CT image dataset and fine-tune on several downstream CT classification tasks. Due to the challenge of obtaining sufficient annotated training data in the medical imaging domain, it is of particular interest to evaluate how the self-supervised pre-training methods perform on small downstream datasets. By experimenting with gradually reducing the training dataset size of our downstream tasks, we find that the reduction has different effects depending on the type of pre-training chosen. The SparK pre-training method is more robust to the training dataset size than the contrastive methods. Based on our results, we propose the SparK pre-training for medical downstream tasks with small datasets.

{{</citation>}}


### (8/45) BEV-DG: Cross-Modal Learning under Bird's-Eye View for Domain Generalization of 3D Semantic Segmentation (Miaoyu Li et al., 2023)

{{<citation>}}

Miaoyu Li, Yachao Zhang, Xu MA, Yanyun Qu, Yun Fu. (2023)  
**BEV-DG: Cross-Modal Learning under Bird's-Eye View for Domain Generalization of 3D Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.06530v1)  

---


**ABSTRACT**  
Cross-modal Unsupervised Domain Adaptation (UDA) aims to exploit the complementarity of 2D-3D data to overcome the lack of annotation in a new domain. However, UDA methods rely on access to the target domain during training, meaning the trained model only works in a specific target domain. In light of this, we propose cross-modal learning under bird's-eye view for Domain Generalization (DG) of 3D semantic segmentation, called BEV-DG. DG is more challenging because the model cannot access the target domain during training, meaning it needs to rely on cross-modal learning to alleviate the domain gap. Since 3D semantic segmentation requires the classification of each point, existing cross-modal learning is directly conducted point-to-point, which is sensitive to the misalignment in projections between pixels and points. To this end, our approach aims to optimize domain-irrelevant representation modeling with the aid of cross-modal learning under bird's-eye view. We propose BEV-based Area-to-area Fusion (BAF) to conduct cross-modal learning under bird's-eye view, which has a higher fault tolerance for point-level misalignment. Furthermore, to model domain-irrelevant representations, we propose BEV-driven Domain Contrastive Learning (BDCL) with the help of cross-modal learning under bird's-eye view. We design three domain generalization settings based on three 3D datasets, and BEV-DG significantly outperforms state-of-the-art competitors with tremendous margins in all settings.

{{</citation>}}


### (9/45) Text-to-Video: a Two-stage Framework for Zero-shot Identity-agnostic Talking-head Generation (Zhichao Wang et al., 2023)

{{<citation>}}

Zhichao Wang, Mengyu Dai, Keld Lundgaard. (2023)  
**Text-to-Video: a Two-stage Framework for Zero-shot Identity-agnostic Talking-head Generation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.06457v1)  

---


**ABSTRACT**  
The advent of ChatGPT has introduced innovative methods for information gathering and analysis. However, the information provided by ChatGPT is limited to text, and the visualization of this information remains constrained. Previous research has explored zero-shot text-to-video (TTV) approaches to transform text into videos. However, these methods lacked control over the identity of the generated audio, i.e., not identity-agnostic, hindering their effectiveness. To address this limitation, we propose a novel two-stage framework for person-agnostic video cloning, specifically focusing on TTV generation. In the first stage, we leverage pretrained zero-shot models to achieve text-to-speech (TTS) conversion. In the second stage, an audio-driven talking head generation method is employed to produce compelling videos privided the audio generated in the first stage. This paper presents a comparative analysis of different TTS and audio-driven talking head generation methods, identifying the most promising approach for future research and development. Some audio and videos samples can be found in the following link: https://github.com/ZhichaoWang970201/Text-to-Video/tree/main.

{{</citation>}}


### (10/45) Improved YOLOv8 Detection Algorithm in Security Inspection Image (Liyao Lu, 2023)

{{<citation>}}

Liyao Lu. (2023)  
**Improved YOLOv8 Detection Algorithm in Security Inspection Image**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.06452v2)  

---


**ABSTRACT**  
Security inspection is the first line of defense to ensure the safety of people's lives and property, and intelligent security inspection is an inevitable trend in the future development of the security inspection industry. Aiming at the problems of overlapping detection objects, false detection of contraband, and missed detection in the process of X-ray image detection, an improved X-ray contraband detection algorithm CSS-YOLO based on YOLOv8s is proposed.

{{</citation>}}


### (11/45) TongueSAM: An Universal Tongue Segmentation Model Based on SAM with Zero-Shot (Shan Cao et al., 2023)

{{<citation>}}

Shan Cao, Qunsheng Ruan, Qingfeng Wu. (2023)  
**TongueSAM: An Universal Tongue Segmentation Model Based on SAM with Zero-Shot**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.06444v1)  

---


**ABSTRACT**  
Tongue segmentation serves as the primary step in automated TCM tongue diagnosis, which plays a significant role in the diagnostic results. Currently, numerous deep learning based methods have achieved promising results. However, most of these methods exhibit mediocre performance on tongues different from the training set. To address this issue, this paper proposes a universal tongue segmentation model named TongueSAM based on SAM (Segment Anything Model). SAM is a large-scale pretrained interactive segmentation model known for its powerful zero-shot generalization capability. Applying SAM to tongue segmentation enables the segmentation of various types of tongue images with zero-shot. In this study, a Prompt Generator based on object detection is integrated into SAM to enable an end-to-end automated tongue segmentation method. Experiments demonstrate that TongueSAM achieves exceptional performance across various of tongue segmentation datasets, particularly under zero-shot. TongueSAM can be directly applied to other datasets without fine-tuning. As far as we know, this is the first application of large-scale pretrained model for tongue segmentation. The project and pretrained model of TongueSAM be publiced in :https://github.com/cshan-github/TongueSAM.

{{</citation>}}


### (12/45) Distributionally Robust Optimization and Invariant Representation Learning for Addressing Subgroup Underrepresentation: Mechanisms and Limitations (Nilesh Kumar et al., 2023)

{{<citation>}}

Nilesh Kumar, Ruby Shrestha, Zhiyuan Li, Linwei Wang. (2023)  
**Distributionally Robust Optimization and Invariant Representation Learning for Addressing Subgroup Underrepresentation: Mechanisms and Limitations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.06434v1)  

---


**ABSTRACT**  
Spurious correlation caused by subgroup underrepresentation has received increasing attention as a source of bias that can be perpetuated by deep neural networks (DNNs). Distributionally robust optimization has shown success in addressing this bias, although the underlying working mechanism mostly relies on upweighting under-performing samples as surrogates for those underrepresented in data. At the same time, while invariant representation learning has been a powerful choice for removing nuisance-sensitive features, it has been little considered in settings where spurious correlations are caused by significant underrepresentation of subgroups. In this paper, we take the first step to better understand and improve the mechanisms for debiasing spurious correlation due to subgroup underrepresentation in medical image classification. Through a comprehensive evaluation study, we first show that 1) generalized reweighting of under-performing samples can be problematic when bias is not the only cause for poor performance, while 2) naive invariant representation learning suffers from spurious correlations itself. We then present a novel approach that leverages robust optimization to facilitate the learning of invariant representations at the presence of spurious correlations. Finetuned classifiers utilizing such representation demonstrated improved abilities to reduce subgroup performance disparity, while maintaining high average and worst-group performance.

{{</citation>}}


## quant-ph (1)



### (13/45) DISQ: Dynamic Iteration Skipping for Variational Quantum Algorithms (Junyao Zhang et al., 2023)

{{<citation>}}

Junyao Zhang, Hanrui Wang, Gokul Subramanian Ravi, Frederic T. Chong, Song Han, Frank Mueller, Yiran Chen. (2023)  
**DISQ: Dynamic Iteration Skipping for Variational Quantum Algorithms**  

---
Primary Category: quant-ph  
Categories: cs-SY, eess-SY, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.06634v1)  

---


**ABSTRACT**  
This paper proposes DISQ to craft a stable landscape for VQA training and tackle the noise drift challenge. DISQ adopts a "drift detector" with a reference circuit to identify and skip iterations that are severely affected by noise drift errors. Specifically, the circuits from the previous training iteration are re-executed as a reference circuit in the current iteration to estimate noise drift impacts. The iteration is deemed compromised by noise drift errors and thus skipped if noise drift flips the direction of the ideal optimization gradient. To enhance noise drift detection reliability, we further propose to leverage multiple reference circuits from previous iterations to provide a well founded judge of current noise drift. Nevertheless, multiple reference circuits also introduce considerable execution overhead. To mitigate extra overhead, we propose Pauli-term subsetting (prime and minor subsets) to execute only observable circuits with large coefficient magnitudes (prime subset) during drift detection. Only this minor subset is executed when the current iteration is drift-free. Evaluations across various applications and QPUs demonstrate that DISQ can mitigate a significant portion of the noise drift impact on VQAs and achieve 1.51-2.24x fidelity improvement over the traditional baseline. DISQ's benefit is 1.1-1.9x over the best alternative approach while boosting average noise detection speed by 2.07x

{{</citation>}}


## cs.CR (2)



### (14/45) PQC-HA: A Framework for Prototyping and In-Hardware Evaluation of Post-Quantum Cryptography Hardware Accelerators (Richard Sattel et al., 2023)

{{<citation>}}

Richard Sattel, Christoph Spang, Carsten Heinz, Andreas Koch. (2023)  
**PQC-HA: A Framework for Prototyping and In-Hardware Evaluation of Post-Quantum Cryptography Hardware Accelerators**  

---
Primary Category: cs.CR  
Categories: cs-AR, cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.06621v1)  

---


**ABSTRACT**  
In the third round of the NIST Post-Quantum Cryptography standardization project, the focus is on optimizing software and hardware implementations of candidate schemes. The winning schemes are CRYSTALS Kyber and CRYSTALS Dilithium, which serve as a Key Encapsulation Mechanism (KEM) and Digital Signature Algorithm (DSA), respectively. This study utilizes the TaPaSCo open-source framework to create hardware building blocks for both schemes using High-level Synthesis (HLS) from minimally modified ANSI C software reference implementations across all security levels. Additionally, a generic TaPaSCo host runtime application is developed in Rust to verify their functionality through the standard NIST interface, utilizing the corresponding Known Answer Test mechanism on actual hardware. Building on this foundation, the communication overhead for TaPaSCo hardware accelerators on PCIe-connected FPGA devices is evaluated and compared with previous work and optimized AVX2 software reference implementations. The results demonstrate the feasibility of verifying and evaluating the performance of Post-Quantum Cryptography accelerators on real hardware using TaPaSCo. Furthermore, the off-chip accelerator communication overhead of the NIST standard interface is measured, which, on its own, outweighs the execution wall clock time of the optimized software reference implementation of Kyber at Security Level 1.

{{</citation>}}


### (15/45) On the Security Bootstrapping in Named Data Networking (Tianyuan Yu et al., 2023)

{{<citation>}}

Tianyuan Yu, Xinyu Ma, Hongcheng Xie, Xiaohua Jia, Lixia Zhang. (2023)  
**On the Security Bootstrapping in Named Data Networking**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.06490v1)  

---


**ABSTRACT**  
By requiring all data packets been cryptographically authenticatable, the Named Data Networking (NDN) architecture design provides a basic building block for secured networking. This basic NDN function requires that all entities in an NDN network go through a security bootstrapping process to obtain the initial security credentials. Recent years have witnessed a number of proposed solutions for NDN security bootstrapping protocols. Built upon the existing results, in this paper we take the next step to develop a systematic model of security bootstrapping: Trust-domain Entity Bootstrapping (TEB). This model is based on the emerging concept of trust domain and describes the steps and their dependencies in the bootstrapping process. We evaluate the expressiveness and sufficiency of this model by using it to describe several current bootstrapping protocols.

{{</citation>}}


## cs.LG (9)



### (16/45) Can Unstructured Pruning Reduce the Depth in Deep Neural Networks? (Liao Zhu et al., 2023)

{{<citation>}}

Liao Zhu, Victor Quétu, Van-Tam Nguyen, Enzo Tartaglione. (2023)  
**Can Unstructured Pruning Reduce the Depth in Deep Neural Networks?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2308.06619v1)  

---


**ABSTRACT**  
Pruning is a widely used technique for reducing the size of deep neural networks while maintaining their performance. However, such a technique, despite being able to massively compress deep models, is hardly able to remove entire layers from a model (even when structured): is this an addressable task? In this study, we introduce EGP, an innovative Entropy Guided Pruning algorithm aimed at reducing the size of deep neural networks while preserving their performance. The key focus of EGP is to prioritize pruning connections in layers with low entropy, ultimately leading to their complete removal. Through extensive experiments conducted on popular models like ResNet-18 and Swin-T, our findings demonstrate that EGP effectively compresses deep neural networks while maintaining competitive performance levels. Our results not only shed light on the underlying mechanism behind the advantages of unstructured pruning, but also pave the way for further investigations into the intricate relationship between entropy, pruning techniques, and deep learning performance. The EGP algorithm and its insights hold great promise for advancing the field of network compression and optimization. The source code for EGP is released open-source.

{{</citation>}}


### (17/45) Value-Distributional Model-Based Reinforcement Learning (Carlos E. Luis et al., 2023)

{{<citation>}}

Carlos E. Luis, Alessandro G. Bottero, Julia Vinogradska, Felix Berkenkamp, Jan Peters. (2023)  
**Value-Distributional Model-Based Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.06590v1)  

---


**ABSTRACT**  
Quantifying uncertainty about a policy's long-term performance is important to solve sequential decision-making tasks. We study the problem from a model-based Bayesian reinforcement learning perspective, where the goal is to learn the posterior distribution over value functions induced by parameter (epistemic) uncertainty of the Markov decision process. Previous work restricts the analysis to a few moments of the distribution over values or imposes a particular distribution shape, e.g., Gaussians. Inspired by distributional reinforcement learning, we introduce a Bellman operator whose fixed-point is the value distribution function. Based on our theory, we propose Epistemic Quantile-Regression (EQR), a model-based algorithm that learns a value distribution function that can be used for policy optimization. Evaluation across several continuous-control tasks shows performance benefits with respect to established model-based and model-free algorithms.

{{</citation>}}


### (18/45) EquiDiff: A Conditional Equivariant Diffusion Model For Trajectory Prediction (Kehua Chen et al., 2023)

{{<citation>}}

Kehua Chen, Xianda Chen, Zihan Yu, Meixin Zhu, Hai Yang. (2023)  
**EquiDiff: A Conditional Equivariant Diffusion Model For Trajectory Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Attention, Graph Attention Network  
[Paper Link](http://arxiv.org/abs/2308.06564v1)  

---


**ABSTRACT**  
Accurate trajectory prediction is crucial for the safe and efficient operation of autonomous vehicles. The growing popularity of deep learning has led to the development of numerous methods for trajectory prediction. While deterministic deep learning models have been widely used, deep generative models have gained popularity as they learn data distributions from training data and account for trajectory uncertainties. In this study, we propose EquiDiff, a deep generative model for predicting future vehicle trajectories. EquiDiff is based on the conditional diffusion model, which generates future trajectories by incorporating historical information and random Gaussian noise. The backbone model of EquiDiff is an SO(2)-equivariant transformer that fully utilizes the geometric properties of location coordinates. In addition, we employ Recurrent Neural Networks and Graph Attention Networks to extract social interactions from historical trajectories. To evaluate the performance of EquiDiff, we conduct extensive experiments on the NGSIM dataset. Our results demonstrate that EquiDiff outperforms other baseline models in short-term prediction, but has slightly higher errors for long-term prediction. Furthermore, we conduct an ablation study to investigate the contribution of each component of EquiDiff to the prediction accuracy. Additionally, we present a visualization of the generation process of our diffusion model, providing insights into the uncertainty of the prediction.

{{</citation>}}


### (19/45) SLoRA: Federated Parameter Efficient Fine-Tuning of Language Models (Sara Babakniya et al., 2023)

{{<citation>}}

Sara Babakniya, Ahmed Roushdy Elkordy, Yahya H. Ezzeldin, Qingfeng Liu, Kee-Bong Song, Mostafa El-Khamy, Salman Avestimehr. (2023)  
**SLoRA: Federated Parameter Efficient Fine-Tuning of Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.06522v1)  

---


**ABSTRACT**  
Transfer learning via fine-tuning pre-trained transformer models has gained significant success in delivering state-of-the-art results across various NLP tasks. In the absence of centralized data, Federated Learning (FL) can benefit from distributed and private data of the FL edge clients for fine-tuning. However, due to the limited communication, computation, and storage capabilities of edge devices and the huge sizes of popular transformer models, efficient fine-tuning is crucial to make federated training feasible. This work explores the opportunities and challenges associated with applying parameter efficient fine-tuning (PEFT) methods in different FL settings for language tasks. Specifically, our investigation reveals that as the data across users becomes more diverse, the gap between fully fine-tuning the model and employing PEFT methods widens. To bridge this performance gap, we propose a method called SLoRA, which overcomes the key limitations of LoRA in high heterogeneous data scenarios through a novel data-driven initialization technique. Our experimental results demonstrate that SLoRA achieves performance comparable to full fine-tuning, with significant sparse updates with approximately $\sim 1\%$ density while reducing training time by up to $90\%$.

{{</citation>}}


### (20/45) Performance Analysis for Resource Constrained Decentralized Federated Learning Over Wireless Networks (Zhigang Yan et al., 2023)

{{<citation>}}

Zhigang Yan, Dong Li. (2023)  
**Performance Analysis for Resource Constrained Decentralized Federated Learning Over Wireless Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-PF, cs.LG, eess-SP  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.06496v1)  

---


**ABSTRACT**  
Federated learning (FL) can lead to significant communication overhead and reliance on a central server. To address these challenges, decentralized federated learning (DFL) has been proposed as a more resilient framework. DFL involves parameter exchange between devices through a wireless network. This study analyzes the performance of resource-constrained DFL using different communication schemes (digital and analog) over wireless networks to optimize communication efficiency. Specifically, we provide convergence bounds for both digital and analog transmission approaches, enabling analysis of the model performance trained on DFL. Furthermore, for digital transmission, we investigate and analyze resource allocation between computation and communication and convergence rates, obtaining its communication complexity and the minimum probability of correction communication required for convergence guarantee. For analog transmission, we discuss the impact of channel fading and noise on the model performance and the maximum errors accumulation with convergence guarantee over fading channels. Finally, we conduct numerical simulations to evaluate the performance and convergence rate of convolutional neural networks (CNNs) and Vision Transformer (ViT) trained in the DFL framework on fashion-MNIST and CIFAR-10 datasets. Our simulation results validate our analysis and discussion, revealing how to improve performance by optimizing system parameters under different communication conditions.

{{</citation>}}


### (21/45) Volterra Accentuated Non-Linear Dynamical Admittance (VANYA) to model Deforestation: An Exemplification from the Amazon Rainforest (Karthik R. et al., 2023)

{{<citation>}}

Karthik R., Ramamoorthy A.. (2023)  
**Volterra Accentuated Non-Linear Dynamical Admittance (VANYA) to model Deforestation: An Exemplification from the Amazon Rainforest**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Amazon  
[Paper Link](http://arxiv.org/abs/2308.06471v1)  

---


**ABSTRACT**  
Intelligent automation supports us against cyclones, droughts, and seismic events with recent technology advancements. Algorithmic learning has advanced fields like neuroscience, genetics, and human-computer interaction. Time-series data boosts progress. Challenges persist in adopting these approaches in traditional fields. Neural networks face comprehension and bias issues. AI's expansion across scientific areas is due to adaptable descriptors and combinatorial argumentation. This article focuses on modeling Forest loss using the VANYA Model, incorporating Prey Predator Dynamics. VANYA predicts forest cover, demonstrated on Amazon Rainforest data against other forecasters like Long Short-Term Memory, N-BEATS, RCN.

{{</citation>}}


### (22/45) Not So Robust After All: Evaluating the Robustness of Deep Neural Networks to Unseen Adversarial Attacks (Roman Garaev et al., 2023)

{{<citation>}}

Roman Garaev, Bader Rasheed, Adil Khan. (2023)  
**Not So Robust After All: Evaluating the Robustness of Deep Neural Networks to Unseen Adversarial Attacks**  

---
Primary Category: cs.LG  
Categories: I-2-10, cs-AI, cs-LG, cs.LG  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2308.06467v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) have gained prominence in various applications, such as classification, recognition, and prediction, prompting increased scrutiny of their properties. A fundamental attribute of traditional DNNs is their vulnerability to modifications in input data, which has resulted in the investigation of adversarial attacks. These attacks manipulate the data in order to mislead a DNN. This study aims to challenge the efficacy and generalization of contemporary defense mechanisms against adversarial attacks. Specifically, we explore the hypothesis proposed by Ilyas et. al, which posits that DNN image features can be either robust or non-robust, with adversarial attacks targeting the latter. This hypothesis suggests that training a DNN on a dataset consisting solely of robust features should produce a model resistant to adversarial attacks. However, our experiments demonstrate that this is not universally true. To gain further insights into our findings, we analyze the impact of adversarial attack norms on DNN representations, focusing on samples subjected to $L_2$ and $L_{\infty}$ norm attacks. Further, we employ canonical correlation analysis, visualize the representations, and calculate the mean distance between these representations and various DNN decision boundaries. Our results reveal a significant difference between $L_2$ and $L_{\infty}$ norms, which could provide insights into the potential dangers posed by $L_{\infty}$ norm attacks, previously underestimated by the research community.

{{</citation>}}


### (23/45) Multi-Label Knowledge Distillation (Penghui Yang et al., 2023)

{{<citation>}}

Penghui Yang, Ming-Kun Xie, Chen-Chen Zong, Lei Feng, Gang Niu, Masashi Sugiyama, Sheng-Jun Huang. (2023)  
**Multi-Label Knowledge Distillation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2308.06453v1)  

---


**ABSTRACT**  
Existing knowledge distillation methods typically work by imparting the knowledge of output logits or intermediate feature maps from the teacher network to the student network, which is very successful in multi-class single-label learning. However, these methods can hardly be extended to the multi-label learning scenario, where each instance is associated with multiple semantic labels, because the prediction probabilities do not sum to one and feature maps of the whole example may ignore minor classes in such a scenario. In this paper, we propose a novel multi-label knowledge distillation method. On one hand, it exploits the informative semantic knowledge from the logits by dividing the multi-label learning problem into a set of binary classification problems; on the other hand, it enhances the distinctiveness of the learned feature representations by leveraging the structural information of label-wise embeddings. Experimental results on multiple benchmark datasets validate that the proposed method can avoid knowledge counteraction among labels, thus achieving superior performance against diverse comparing methods. Our code is available at: https://github.com/penghui-yang/L2D

{{</citation>}}


### (24/45) Sensitivity-Aware Mixed-Precision Quantization and Width Optimization of Deep Neural Networks Through Cluster-Based Tree-Structured Parzen Estimation (Seyedarmin Azizi et al., 2023)

{{<citation>}}

Seyedarmin Azizi, Mahdi Nazemi, Arash Fayyazi, Massoud Pedram. (2023)  
**Sensitivity-Aware Mixed-Precision Quantization and Width Optimization of Deep Neural Networks Through Cluster-Based Tree-Structured Parzen Estimation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NE, cs.LG, stat-ML  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2308.06422v2)  

---


**ABSTRACT**  
As the complexity and computational demands of deep learning models rise, the need for effective optimization methods for neural network designs becomes paramount. This work introduces an innovative search mechanism for automatically selecting the best bit-width and layer-width for individual neural network layers. This leads to a marked enhancement in deep neural network efficiency. The search domain is strategically reduced by leveraging Hessian-based pruning, ensuring the removal of non-crucial parameters. Subsequently, we detail the development of surrogate models for favorable and unfavorable outcomes by employing a cluster-based tree-structured Parzen estimator. This strategy allows for a streamlined exploration of architectural possibilities and swift pinpointing of top-performing designs. Through rigorous testing on well-known datasets, our method proves its distinct advantage over existing methods. Compared to leading compression strategies, our approach records an impressive 20% decrease in model size without compromising accuracy. Additionally, our method boasts a 12x reduction in search time relative to the best search-focused strategies currently available. As a result, our proposed method represents a leap forward in neural network design optimization, paving the way for quick model design and implementation in settings with limited resources, thereby propelling the potential of scalable deep learning solutions.

{{</citation>}}


## cs.CL (11)



### (25/45) Bio-SIEVE: Exploring Instruction Tuning Large Language Models for Systematic Review Automation (Ambrose Robinson et al., 2023)

{{<citation>}}

Ambrose Robinson, William Thorne, Ben P. Wu, Abdullah Pandor, Munira Essat, Mark Stevenson, Xingyi Song. (2023)  
**Bio-SIEVE: Exploring Instruction Tuning Large Language Models for Systematic Review Automation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2308.06610v1)  

---


**ABSTRACT**  
Medical systematic reviews can be very costly and resource intensive. We explore how Large Language Models (LLMs) can support and be trained to perform literature screening when provided with a detailed set of selection criteria. Specifically, we instruction tune LLaMA and Guanaco models to perform abstract screening for medical systematic reviews. Our best model, Bio-SIEVE, outperforms both ChatGPT and trained traditional approaches, and generalises better across medical domains. However, there remains the challenge of adapting the model to safety-first scenarios. We also explore the impact of multi-task training with Bio-SIEVE-Multi, including tasks such as PICO extraction and exclusion reasoning, but find that it is unable to match single-task Bio-SIEVE's performance. We see Bio-SIEVE as an important step towards specialising LLMs for the biomedical systematic review process and explore its future developmental opportunities. We release our models, code and a list of DOIs to reconstruct our dataset for reproducibility.

{{</citation>}}


### (26/45) VisIT-Bench: A Benchmark for Vision-Language Instruction Following Inspired by Real-World Use (Yonatan Bitton et al., 2023)

{{<citation>}}

Yonatan Bitton, Hritik Bansal, Jack Hessel, Rulin Shao, Wanrong Zhu, Anas Awadalla, Josh Gardner, Rohan Taori, Ludwig Schimdt. (2023)  
**VisIT-Bench: A Benchmark for Vision-Language Instruction Following Inspired by Real-World Use**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: GPT, GPT-4, QA  
[Paper Link](http://arxiv.org/abs/2308.06595v1)  

---


**ABSTRACT**  
We introduce VisIT-Bench (Visual InsTruction Benchmark), a benchmark for evaluation of instruction-following vision-language models for real-world use. Our starting point is curating 70 'instruction families' that we envision instruction tuned vision-language models should be able to address. Extending beyond evaluations like VQAv2 and COCO, tasks range from basic recognition to game playing and creative generation. Following curation, our dataset comprises 592 test queries, each with a human-authored instruction-conditioned caption. These descriptions surface instruction-specific factors, e.g., for an instruction asking about the accessibility of a storefront for wheelchair users, the instruction-conditioned caption describes ramps/potential obstacles. These descriptions enable 1) collecting human-verified reference outputs for each instance; and 2) automatic evaluation of candidate multimodal generations using a text-only LLM, aligning with human judgment. We quantify quality gaps between models and references using both human and automatic evaluations; e.g., the top-performing instruction-following model wins against the GPT-4 reference in just 27% of the comparison. VisIT-Bench is dynamic to participate, practitioners simply submit their model's response on the project website; Data, code and leaderboard is available at visit-bench.github.io.

{{</citation>}}


### (27/45) MT4CrossOIE: Multi-stage Tuning for Cross-lingual Open Information Extraction (Zixiang Wang et al., 2023)

{{<citation>}}

Zixiang Wang, Linzheng Chai, Jian Yang, Jiaqi Bai, Yuwei Yin, Jiaheng Liu, Hongcheng Guo, Tongliang Li, Liqun Yang, Hebboul Zine el-abidine, Zhoujun Li. (2023)  
**MT4CrossOIE: Multi-stage Tuning for Cross-lingual Open Information Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Information Extraction, Multilingual, NLP  
[Paper Link](http://arxiv.org/abs/2308.06552v1)  

---


**ABSTRACT**  
Cross-lingual open information extraction aims to extract structured information from raw text across multiple languages. Previous work uses a shared cross-lingual pre-trained model to handle the different languages but underuses the potential of the language-specific representation. In this paper, we propose an effective multi-stage tuning framework called MT4CrossIE, designed for enhancing cross-lingual open information extraction by injecting language-specific knowledge into the shared model. Specifically, the cross-lingual pre-trained model is first tuned in a shared semantic space (e.g., embedding matrix) in the fixed encoder and then other components are optimized in the second stage. After enough training, we freeze the pre-trained model and tune the multiple extra low-rank language-specific modules using mixture-of-LoRAs for model-based cross-lingual transfer. In addition, we leverage two-stage prompting to encourage the large language model (LLM) to annotate the multi-lingual raw data for data-based cross-lingual transfer. The model is trained with multi-lingual objectives on our proposed dataset OpenIE4++ by combing the model-based and data-based transfer techniques. Experimental results on various benchmarks emphasize the importance of aggregating multiple plug-in-and-play language-specific modules and demonstrate the effectiveness of MT4CrossIE in cross-lingual OIE\footnote{\url{https://github.com/CSJianYang/Multilingual-Multimodal-NLP}}.

{{</citation>}}


### (28/45) AutoConv: Automatically Generating Information-seeking Conversations with Large Language Models (Siheng Li et al., 2023)

{{<citation>}}

Siheng Li, Cheng Yang, Yichun Yin, Xinyu Zhu, Zesen Cheng, Lifeng Shang, Xin Jiang, Qun Liu, Yujiu Yang. (2023)  
**AutoConv: Automatically Generating Information-seeking Conversations with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.06507v1)  

---


**ABSTRACT**  
Information-seeking conversation, which aims to help users gather information through conversation, has achieved great progress in recent years. However, the research is still stymied by the scarcity of training data. To alleviate this problem, we propose AutoConv for synthetic conversation generation, which takes advantage of the few-shot learning ability and generation capacity of large language models (LLM). Specifically, we formulate the conversation generation problem as a language modeling task, then finetune an LLM with a few human conversations to capture the characteristics of the information-seeking process and use it for generating synthetic conversations with high quality. Experimental results on two frequently-used datasets verify that AutoConv has substantial improvements over strong baselines and alleviates the dependence on human annotation. In addition, we also provide several analysis studies to promote future research.

{{</citation>}}


### (29/45) Three Ways of Using Large Language Models to Evaluate Chat (Ondřej Plátek et al., 2023)

{{<citation>}}

Ondřej Plátek, Vojtěch Hudeček, Patricia Schmidtová, Mateusz Lango, Ondřej Dušek. (2023)  
**Three Ways of Using Large Language Models to Evaluate Chat**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.06502v1)  

---


**ABSTRACT**  
This paper describes the systems submitted by team6 for ChatEval, the DSTC 11 Track 4 competition. We present three different approaches to predicting turn-level qualities of chatbot responses based on large language models (LLMs). We report improvement over the baseline using dynamic few-shot examples from a vector store for the prompts for ChatGPT. We also analyze the performance of the other two approaches and report needed improvements for future work. We developed the three systems over just two weeks, showing the potential of LLMs for this task. An ablation study conducted after the challenge deadline shows that the new Llama 2 models are closing the performance gap between ChatGPT and open-source LLMs. However, we find that the Llama 2 models do not benefit from few-shot examples in the same way as ChatGPT.

{{</citation>}}


### (30/45) NewsDialogues: Towards Proactive News Grounded Conversation (Siheng Li et al., 2023)

{{<citation>}}

Siheng Li, Yichun Yin, Cheng Yang, Wangjie Jiang, Yiwei Li, Zesen Cheng, Lifeng Shang, Xin Jiang, Qun Liu, Yujiu Yang. (2023)  
**NewsDialogues: Towards Proactive News Grounded Conversation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2308.06501v1)  

---


**ABSTRACT**  
Hot news is one of the most popular topics in daily conversations. However, news grounded conversation has long been stymied by the lack of well-designed task definition and scarce data. In this paper, we propose a novel task, Proactive News Grounded Conversation, in which a dialogue system can proactively lead the conversation based on some key topics of the news. In addition, both information-seeking and chit-chat scenarios are included realistically, where the user may ask a series of questions about the news details or express their opinions and be eager to chat. To further develop this novel task, we collect a human-to-human Chinese dialogue dataset \ts{NewsDialogues}, which includes 1K conversations with a total of 14.6K utterances and detailed annotations for target topics and knowledge spans. Furthermore, we propose a method named Predict-Generate-Rank, consisting of a generator for grounded knowledge prediction and response generation, and a ranker for the ranking of multiple responses to alleviate the exposure bias. We conduct comprehensive experiments to demonstrate the effectiveness of the proposed method and further present several key findings and challenges to prompt future research.

{{</citation>}}


### (31/45) Generating Faithful Text From a Knowledge Graph with Noisy Reference Text (Tahsina Hashem et al., 2023)

{{<citation>}}

Tahsina Hashem, Weiqing Wang, Derry Tanti Wijaya, Mohammed Eunus Ali, Yuan-Fang Li. (2023)  
**Generating Faithful Text From a Knowledge Graph with Noisy Reference Text**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.06488v1)  

---


**ABSTRACT**  
Knowledge Graph (KG)-to-Text generation aims at generating fluent natural-language text that accurately represents the information of a given knowledge graph. While significant progress has been made in this task by exploiting the power of pre-trained language models (PLMs) with appropriate graph structure-aware modules, existing models still fall short of generating faithful text, especially when the ground-truth natural-language text contains additional information that is not present in the graph. In this paper, we develop a KG-to-text generation model that can generate faithful natural-language text from a given graph, in the presence of noisy reference text. Our framework incorporates two core ideas: Firstly, we utilize contrastive learning to enhance the model's ability to differentiate between faithful and hallucinated information in the text, thereby encouraging the decoder to generate text that aligns with the input graph. Secondly, we empower the decoder to control the level of hallucination in the generated text by employing a controllable text generation technique. We evaluate our model's performance through the standard quantitative metrics as well as a ChatGPT-based quantitative and qualitative analysis. Our evaluation demonstrates the superior performance of our model over state-of-the-art KG-to-text models on faithfulness.

{{</citation>}}


### (32/45) GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher (Youliang Yuan et al., 2023)

{{<citation>}}

Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Pinjia He, Shuming Shi, Zhaopeng Tu. (2023)  
**GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.06463v1)  

---


**ABSTRACT**  
Safety lies at the core of the development of Large Language Models (LLMs). There is ample work on aligning LLMs with human ethics and preferences, including data filtering in pretraining, supervised fine-tuning, reinforcement learning from human feedback, and red teaming, etc. In this study, we discover that chat in cipher can bypass the safety alignment techniques of LLMs, which are mainly conducted in natural languages. We propose a novel framework CipherChat to systematically examine the generalizability of safety alignment to non-natural languages -- ciphers. CipherChat enables humans to chat with LLMs through cipher prompts topped with system role descriptions and few-shot enciphered demonstrations. We use CipherChat to assess state-of-the-art LLMs, including ChatGPT and GPT-4 for different representative human ciphers across 11 safety domains in both English and Chinese. Experimental results show that certain ciphers succeed almost 100% of the time to bypass the safety alignment of GPT-4 in several safety domains, demonstrating the necessity of developing safety alignment for non-natural languages. Notably, we identify that LLMs seem to have a ''secret cipher'', and propose a novel SelfCipher that uses only role play and several demonstrations in natural language to evoke this capability. SelfCipher surprisingly outperforms existing human ciphers in almost all cases. Our code and data will be released at https://github.com/RobustNLP/CipherChat.

{{</citation>}}


### (33/45) Demonstration-based learning for few-shot biomedical named entity recognition under machine reading comprehension (Leilei Su et al., 2023)

{{<citation>}}

Leilei Su, Jian Chen, Yifan Peng, Cong Sun. (2023)  
**Demonstration-based learning for few-shot biomedical named entity recognition under machine reading comprehension**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, NLP  
[Paper Link](http://arxiv.org/abs/2308.06454v1)  

---


**ABSTRACT**  
Although deep learning techniques have shown significant achievements, they frequently depend on extensive amounts of hand-labeled data and tend to perform inadequately in few-shot scenarios. The objective of this study is to devise a strategy that can improve the model's capability to recognize biomedical entities in scenarios of few-shot learning. By redefining biomedical named entity recognition (BioNER) as a machine reading comprehension (MRC) problem, we propose a demonstration-based learning method to address few-shot BioNER, which involves constructing appropriate task demonstrations. In assessing our proposed method, we compared the proposed method with existing advanced methods using six benchmark datasets, including BC4CHEMD, BC5CDR-Chemical, BC5CDR-Disease, NCBI-Disease, BC2GM, and JNLPBA. We examined the models' efficacy by reporting F1 scores from both the 25-shot and 50-shot learning experiments. In 25-shot learning, we observed 1.1% improvements in the average F1 scores compared to the baseline method, reaching 61.7%, 84.1%, 69.1%, 70.1%, 50.6%, and 59.9% on six datasets, respectively. In 50-shot learning, we further improved the average F1 scores by 1.0% compared to the baseline method, reaching 73.1%, 86.8%, 76.1%, 75.6%, 61.7%, and 65.4%, respectively. We reported that in the realm of few-shot learning BioNER, MRC-based language models are much more proficient in recognizing biomedical entities compared to the sequence labeling approach. Furthermore, our MRC-language models can compete successfully with fully-supervised learning methodologies that rely heavily on the availability of abundant annotated data. These results highlight possible pathways for future advancements in few-shot BioNER methodologies.

{{</citation>}}


### (34/45) Simple Model Also Works: A Novel Emotion Recognition Network in Textual Conversation Based on Curriculum Learning Strategy (Jiang Li et al., 2023)

{{<citation>}}

Jiang Li, Xiaoping Wang, Yingjian Liu, Qing Zhou, Zhigang Zeng. (2023)  
**Simple Model Also Works: A Novel Emotion Recognition Network in Textual Conversation Based on Curriculum Learning Strategy**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2308.06450v1)  

---


**ABSTRACT**  
Emotion Recognition in Conversation (ERC) has emerged as a research hotspot in domains such as conversational robots and question-answer systems. How to efficiently and adequately retrieve contextual emotional cues has been one of the key challenges in the ERC task. Existing efforts do not fully model the context and employ complex network structures, resulting in excessive computational resource overhead without substantial performance improvement. In this paper, we propose a novel Emotion Recognition Network based on Curriculum Learning strategy (ERNetCL). The proposed ERNetCL primarily consists of Temporal Encoder (TE), Spatial Encoder (SE), and Curriculum Learning (CL) loss. We utilize TE and SE to combine the strengths of previous methods in a simplistic manner to efficiently capture temporal and spatial contextual information in the conversation. To simulate the way humans learn curriculum from easy to hard, we apply the idea of CL to the ERC task to progressively optimize the network parameters of ERNetCL. At the beginning of training, we assign lower learning weights to difficult samples. As the epoch increases, the learning weights for these samples are gradually raised. Extensive experiments on four datasets exhibit that our proposed method is effective and dramatically beats other baseline models.

{{</citation>}}


### (35/45) Performance Prediction for Multi-hop Questions (Mohammadreza Samadi et al., 2023)

{{<citation>}}

Mohammadreza Samadi, Davood Rafiei. (2023)  
**Performance Prediction for Multi-hop Questions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-DB, cs-IR, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.06431v1)  

---


**ABSTRACT**  
We study the problem of Query Performance Prediction (QPP) for open-domain multi-hop Question Answering (QA), where the task is to estimate the difficulty of evaluating a multi-hop question over a corpus. Despite the extensive research on predicting the performance of ad-hoc and QA retrieval models, there has been a lack of study on the estimation of the difficulty of multi-hop questions. The problem is challenging due to the multi-step nature of the retrieval process, potential dependency of the steps and the reasoning involved. To tackle this challenge, we propose multHP, a novel pre-retrieval method for predicting the performance of open-domain multi-hop questions. Our extensive evaluation on the largest multi-hop QA dataset using several modern QA systems shows that the proposed model is a strong predictor of the performance, outperforming traditional single-hop QPP models. Additionally, we demonstrate that our approach can be effectively used to optimize the parameters of QA systems, such as the number of documents to be retrieved, resulting in improved overall retrieval performance.

{{</citation>}}


## cs.RO (1)



### (36/45) CoverNav: Cover Following Navigation Planning in Unstructured Outdoor Environment with Deep Reinforcement Learning (Jumman Hossain et al., 2023)

{{<citation>}}

Jumman Hossain, Abu-Zaher Faridee, Nirmalya Roy, Anjan Basak, Derrik E. Asher. (2023)  
**CoverNav: Cover Following Navigation Planning in Unstructured Outdoor Environment with Deep Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.06594v1)  

---


**ABSTRACT**  
Autonomous navigation in offroad environments has been extensively studied in the robotics field. However, navigation in covert situations where an autonomous vehicle needs to remain hidden from outside observers remains an underexplored area. In this paper, we propose a novel Deep Reinforcement Learning (DRL) based algorithm, called CoverNav, for identifying covert and navigable trajectories with minimal cost in offroad terrains and jungle environments in the presence of observers. CoverNav focuses on unmanned ground vehicles seeking shelters and taking covers while safely navigating to a predefined destination. Our proposed DRL method computes a local cost map that helps distinguish which path will grant the maximal covertness while maintaining a low cost trajectory using an elevation map generated from 3D point cloud data, the robot's pose, and directed goal information. CoverNav helps robot agents to learn the low elevation terrain using a reward function while penalizing it proportionately when it experiences high elevation. If an observer is spotted, CoverNav enables the robot to select natural obstacles (e.g., rocks, houses, disabled vehicles, trees, etc.) and use them as shelters to hide behind. We evaluate CoverNav using the Unity simulation environment and show that it guarantees dynamically feasible velocities in the terrain when fed with an elevation map generated by another DRL based navigation algorithm. Additionally, we evaluate CoverNav's effectiveness in achieving a maximum goal distance of 12 meters and its success rate in different elevation scenarios with and without cover objects. We observe competitive performance comparable to state of the art (SOTA) methods without compromising accuracy.

{{</citation>}}


## cs.NE (1)



### (37/45) Gated Attention Coding for Training High-performance and Efficient Spiking Neural Networks (Xuerui Qiu et al., 2023)

{{<citation>}}

Xuerui Qiu, Rui-Jie Zhu, Yuhong Chou, Zhaorui Wang, Liang-jian Deng, Guoqi Li. (2023)  
**Gated Attention Coding for Training High-performance and Efficient Spiking Neural Networks**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: Attention, ImageNet  
[Paper Link](http://arxiv.org/abs/2308.06582v1)  

---


**ABSTRACT**  
Spiking neural networks (SNNs) are emerging as an energy-efficient alternative to traditional artificial neural networks (ANNs) due to their unique spike-based event-driven nature. Coding is crucial in SNNs as it converts external input stimuli into spatio-temporal feature sequences. However, most existing deep SNNs rely on direct coding that generates powerless spike representation and lacks the temporal dynamics inherent in human vision. Hence, we introduce Gated Attention Coding (GAC), a plug-and-play module that leverages the multi-dimensional gated attention unit to efficiently encode inputs into powerful representations before feeding them into the SNN architecture. GAC functions as a preprocessing layer that does not disrupt the spike-driven nature of the SNN, making it amenable to efficient neuromorphic hardware implementation with minimal modifications. Through an observer model theoretical analysis, we demonstrate GAC's attention mechanism improves temporal dynamics and coding efficiency. Experiments on CIFAR10/100 and ImageNet datasets demonstrate that GAC achieves state-of-the-art accuracy with remarkable efficiency. Notably, we improve top-1 accuracy by 3.10\% on CIFAR100 with only 6-time steps and 1.07\% on ImageNet while reducing energy usage to 66.9\% of the previous works. To our best knowledge, it is the first time to explore the attention-based dynamic coding scheme in deep SNNs, with exceptional effectiveness and efficiency on large-scale datasets.

{{</citation>}}


## cs.IR (1)



### (38/45) Contrastive Learning for Cross-modal Artist Retrieval (Andres Ferraro et al., 2023)

{{<citation>}}

Andres Ferraro, Jaehun Kim, Sergio Oramas, Andreas Ehmann, Fabien Gouyon. (2023)  
**Contrastive Learning for Cross-modal Artist Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.06556v1)  

---


**ABSTRACT**  
Music retrieval and recommendation applications often rely on content features encoded as embeddings, which provide vector representations of items in a music dataset. Numerous complementary embeddings can be derived from processing items originally represented in several modalities, e.g., audio signals, user interaction data, or editorial data. However, data of any given modality might not be available for all items in any music dataset. In this work, we propose a method based on contrastive learning to combine embeddings from multiple modalities and explore the impact of the presence or absence of embeddings from diverse modalities in an artist similarity task. Experiments on two datasets suggest that our contrastive method outperforms single-modality embeddings and baseline algorithms for combining modalities, both in terms of artist retrieval accuracy and coverage. Improvements with respect to other methods are particularly significant for less popular query artists. We demonstrate our method successfully combines complementary information from diverse modalities, and is more robust to missing modality data (i.e., it better handles the retrieval of artists with different modality embeddings than the query artist's).

{{</citation>}}


## eess.AS (1)



### (39/45) Alternative Pseudo-Labeling for Semi-Supervised Automatic Speech Recognition (Han Zhu et al., 2023)

{{<citation>}}

Han Zhu, Dongji Gao, Gaofeng Cheng, Daniel Povey, Pengyuan Zhang, Yonghong Yan. (2023)  
**Alternative Pseudo-Labeling for Semi-Supervised Automatic Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Semi-Supervised, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2308.06547v1)  

---


**ABSTRACT**  
When labeled data is insufficient, semi-supervised learning with the pseudo-labeling technique can significantly improve the performance of automatic speech recognition. However, pseudo-labels are often noisy, containing numerous incorrect tokens. Taking noisy labels as ground-truth in the loss function results in suboptimal performance. Previous works attempted to mitigate this issue by either filtering out the nosiest pseudo-labels or improving the overall quality of pseudo-labels. While these methods are effective to some extent, it is unrealistic to entirely eliminate incorrect tokens in pseudo-labels. In this work, we propose a novel framework named alternative pseudo-labeling to tackle the issue of noisy pseudo-labels from the perspective of the training objective. The framework comprises several components. Firstly, a generalized CTC loss function is introduced to handle noisy pseudo-labels by accepting alternative tokens in the positions of incorrect tokens. Applying this loss function in pseudo-labeling requires detecting incorrect tokens in the predicted pseudo-labels. In this work, we adopt a confidence-based error detection method that identifies the incorrect tokens by comparing their confidence scores with a given threshold, thus necessitating the confidence score to be discriminative. Hence, the second proposed technique is the contrastive CTC loss function that widens the confidence gap between the correctly and incorrectly predicted tokens, thereby improving the error detection ability. Additionally, obtaining satisfactory performance with confidence-based error detection typically requires extensive threshold tuning. Instead, we propose an automatic thresholding method that uses labeled data as a proxy for determining the threshold, thus saving the pain of manual tuning.

{{</citation>}}


## cs.AI (2)



### (40/45) Learning Abstract Visual Reasoning via Task Decomposition: A Case Study in Raven Progressive Matrices (Jakub Kwiatkowski et al., 2023)

{{<citation>}}

Jakub Kwiatkowski, Krzysztof Krawiec. (2023)  
**Learning Abstract Visual Reasoning via Task Decomposition: A Case Study in Raven Progressive Matrices**  

---
Primary Category: cs.AI  
Categories: 68T20 (Primary) 68T05 (Secondary), I-2-8; I-2-6; I-2-10, cs-AI, cs-CV, cs-LG, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.06528v1)  

---


**ABSTRACT**  
One of the challenges in learning to perform abstract reasoning is that problems are often posed as monolithic tasks, with no intermediate subgoals. In Raven Progressive Matrices (RPM), the task is to choose one of the available answers given a context, where both contexts and answers are composite images featuring multiple objects in various spatial arrangements. As this high-level goal is the only guidance available, learning is challenging and most contemporary solvers tend to be opaque. In this study, we propose a deep learning architecture based on the transformer blueprint which, rather than directly making the above choice, predicts the visual properties of individual objects and their arrangements. The multidimensional predictions obtained in this way are then directly juxtaposed to choose the answer. We consider a few ways in which the model parses the visual input into tokens and several regimes of masking parts of the input in self-supervised training. In experimental assessment, the models not only outperform state-of-the-art methods but also provide interesting insights and partial explanations about the inference. The design of the method also makes it immune to biases that are known to exist in some RPM benchmarks.

{{</citation>}}


### (41/45) HyperFormer: Enhancing Entity and Relation Interaction for Hyper-Relational Knowledge Graph Completion (Zhiwei Hu et al., 2023)

{{<citation>}}

Zhiwei Hu, Víctor Gutiérrez-Basulto, Zhiliang Xiang, Ru Li, Jeff Z. Pan. (2023)  
**HyperFormer: Enhancing Entity and Relation Interaction for Hyper-Relational Knowledge Graph Completion**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.06512v1)  

---


**ABSTRACT**  
Hyper-relational knowledge graphs (HKGs) extend standard knowledge graphs by associating attribute-value qualifiers to triples, which effectively represent additional fine-grained information about its associated triple. Hyper-relational knowledge graph completion (HKGC) aims at inferring unknown triples while considering its qualifiers. Most existing approaches to HKGC exploit a global-level graph structure to encode hyper-relational knowledge into the graph convolution message passing process. However, the addition of multi-hop information might bring noise into the triple prediction process. To address this problem, we propose HyperFormer, a model that considers local-level sequential information, which encodes the content of the entities, relations and qualifiers of a triple. More precisely, HyperFormer is composed of three different modules: an entity neighbor aggregator module allowing to integrate the information of the neighbors of an entity to capture different perspectives of it; a relation qualifier aggregator module to integrate hyper-relational knowledge into the corresponding relation to refine the representation of relational content; a convolution-based bidirectional interaction module based on a convolutional operation, capturing pairwise bidirectional interactions of entity-relation, entity-qualifier, and relation-qualifier. realize the depth perception of the content related to the current statement. Furthermore, we introduce a Mixture-of-Experts strategy into the feed-forward layers of HyperFormer to strengthen its representation capabilities while reducing the amount of model parameters and computation. Extensive experiments on three well-known datasets with four different conditions demonstrate HyperFormer's effectiveness. Datasets and code are available at https://github.com/zhiweihu1103/HKGC-HyperFormer.

{{</citation>}}


## cs.IT (1)



### (42/45) Integrated Sensing-Communication-Computation for Over-the-Air Edge AI Inference (Zeming Zhuang et al., 2023)

{{<citation>}}

Zeming Zhuang, Dingzhu Wen, Yuanming Shi, Guangxu Zhu, Sheng Wu, Dusit Niyato. (2023)  
**Integrated Sensing-Communication-Computation for Over-the-Air Edge AI Inference**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.06503v1)  

---


**ABSTRACT**  
Edge-device co-inference refers to deploying well-trained artificial intelligent (AI) models at the network edge under the cooperation of devices and edge servers for providing ambient intelligent services. For enhancing the utilization of limited network resources in edge-device co-inference tasks from a systematic view, we propose a task-oriented scheme of integrated sensing, computation and communication (ISCC) in this work. In this system, all devices sense a target from the same wide view to obtain homogeneous noise-corrupted sensory data, from which the local feature vectors are extracted. All local feature vectors are aggregated at the server using over-the-air computation (AirComp) in a broadband channel with the orthogonal-frequency-division-multiplexing technique for suppressing the sensing and channel noise. The aggregated denoised global feature vector is further input to a server-side AI model for completing the downstream inference task. A novel task-oriented design criterion, called maximum minimum pair-wise discriminant gain, is adopted for classification tasks. It extends the distance of the closest class pair in the feature space, leading to a balanced and enhanced inference accuracy. Under this criterion, a problem of joint sensing power assignment, transmit precoding and receive beamforming is formulated. The challenge lies in three aspects: the coupling between sensing and AirComp, the joint optimization of all feature dimensions' AirComp aggregation over a broadband channel, and the complicated form of the maximum minimum pair-wise discriminant gain. To solve this problem, a task-oriented ISCC scheme with AirComp is proposed. Experiments based on a human motion recognition task are conducted to verify the advantages of the proposed scheme over the existing scheme and a baseline.

{{</citation>}}


## cs.NI (1)



### (43/45) mmHawkeye: Passive UAV Detection with a COTS mmWave Radar (Jia Zhang et al., 2023)

{{<citation>}}

Jia Zhang, Xin Na, Rui Xi, Yimiao Sun, Yuan He. (2023)  
**mmHawkeye: Passive UAV Detection with a COTS mmWave Radar**  

---
Primary Category: cs.NI  
Categories: C-2; J-3, cs-NI, cs.NI, eess-SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2308.06479v1)  

---


**ABSTRACT**  
Small Unmanned Aerial Vehicles (UAVs) are becoming potential threats to security-sensitive areas and personal privacy. A UAV can shoot photos at height, but how to detect such an uninvited intruder is an open problem. This paper presents mmHawkeye, a passive approach for UAV detection with a COTS millimeter wave (mmWave) radar. mmHawkeye doesn't require prior knowledge of the type, motions, and flight trajectory of the UAV, while exploiting the signal feature induced by the UAV's periodic micro-motion (PMM) for long-range accurate detection. The design is therefore effective in dealing with low-SNR and uncertain reflected signals from the UAV. mmHawkeye can further track the UAV's position with dynamic programming and particle filtering, and identify it with a Long Short-Term Memory (LSTM) based detector. We implement mmHawkeye on a commercial mmWave radar and evaluate its performance under varied settings. The experimental results show that mmHawkeye has a detection accuracy of 95.8% and can realize detection at a range up to 80m.

{{</citation>}}


## cs.SD (1)



### (44/45) Flexible Keyword Spotting based on Homogeneous Audio-Text Embedding (Kumari Nishu et al., 2023)

{{<citation>}}

Kumari Nishu, Minsik Cho, Paul Dixon, Devang Naik. (2023)  
**Flexible Keyword Spotting based on Homogeneous Audio-Text Embedding**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.06472v1)  

---


**ABSTRACT**  
Spotting user-defined/flexible keywords represented in text frequently uses an expensive text encoder for joint analysis with an audio encoder in an embedding space, which can suffer from heterogeneous modality representation (i.e., large mismatch) and increased complexity. In this work, we propose a novel architecture to efficiently detect arbitrary keywords based on an audio-compliant text encoder which inherently has homogeneous representation with audio embedding, and it is also much smaller than a compatible text encoder. Our text encoder converts the text to phonemes using a grapheme-to-phoneme (G2P) model, and then to an embedding using representative phoneme vectors, extracted from the paired audio encoder on rich speech datasets. We further augment our method with confusable keyword generation to develop an audio-text embedding verifier with strong discriminative power. Experimental results show that our scheme outperforms the state-of-the-art results on Libriphrase hard dataset, increasing Area Under the ROC Curve (AUC) metric from 84.21% to 92.7% and reducing Equal-Error-Rate (EER) metric from 23.36% to 14.4%.

{{</citation>}}


## cs.SI (1)



### (45/45) Mainstream News Articles Co-Shared with Fake News Buttress Misinformation Narratives (Pranav Goel et al., 2023)

{{<citation>}}

Pranav Goel, Jon Green, David Lazer, Philip Resnik. (2023)  
**Mainstream News Articles Co-Shared with Fake News Buttress Misinformation Narratives**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Fake News, NLP, Twitter  
[Paper Link](http://arxiv.org/abs/2308.06459v1)  

---


**ABSTRACT**  
Most prior and current research examining misinformation spread on social media focuses on reports published by 'fake' news sources. These approaches fail to capture another potential form of misinformation with a much larger audience: factual news from mainstream sources ('real' news) repurposed to promote false or misleading narratives. We operationalize narratives using an existing unsupervised NLP technique and examine the narratives present in misinformation content. We find that certain articles from reliable outlets are shared by a disproportionate number of users who also shared fake news on Twitter. We consider these 'real' news articles to be co-shared with fake news. We show that co-shared articles contain existing misinformation narratives at a significantly higher rate than articles from the same reliable outlets that are not co-shared with fake news. This holds true even when articles are chosen following strict criteria of reliability for the outlets and after accounting for the alternative explanation of partisan curation of articles. For example, we observe that a recent article published by The Washington Post titled "Vaccinated people now make up a majority of COVID deaths" was disproportionately shared by Twitter users with a history of sharing anti-vaccine false news reports. Our findings suggest a strategic repurposing of mainstream news by conveyors of misinformation as a way to enhance the reach and persuasiveness of misleading narratives. We also conduct a comprehensive case study to help highlight how such repurposing can happen on Twitter as a consequence of the inclusion of particular narratives in the framing of mainstream news.

{{</citation>}}
