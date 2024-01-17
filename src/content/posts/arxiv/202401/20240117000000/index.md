---
draft: false
title: "arXiv @ 2024.01.17"
date: 2024-01-17
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.17"
    identifier: arxiv_20240117
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (23)](#cscv-23)
- [cs.SE (8)](#csse-8)
- [cs.LG (11)](#cslg-11)
- [cs.CL (29)](#cscl-29)
- [cs.AI (7)](#csai-7)
- [cs.CR (2)](#cscr-2)
- [eess.IV (1)](#eessiv-1)
- [cs.RO (4)](#csro-4)
- [cs.HC (1)](#cshc-1)
- [cs.SD (1)](#cssd-1)
- [cs.CY (2)](#cscy-2)
- [quant-ph (1)](#quant-ph-1)
- [cs.IR (2)](#csir-2)
- [cs.NI (1)](#csni-1)
- [eess.SY (1)](#eesssy-1)
- [eess.AS (1)](#eessas-1)
- [cs.GT (1)](#csgt-1)
- [cs.MM (1)](#csmm-1)
- [cs.SI (1)](#cssi-1)
- [cs.CE (1)](#csce-1)

## cs.CV (23)



### (1/99) Convolutional Neural Network Compression via Dynamic Parameter Rank Pruning (Manish Sharma et al., 2024)

{{<citation>}}

Manish Sharma, Jamison Heard, Eli Saber, Panos P. Markopoulos. (2024)  
**Convolutional Neural Network Compression via Dynamic Parameter Rank Pruning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2401.08014v1)  

---


**ABSTRACT**  
While Convolutional Neural Networks (CNNs) excel at learning complex latent-space representations, their over-parameterization can lead to overfitting and reduced performance, particularly with limited data. This, alongside their high computational and memory demands, limits the applicability of CNNs for edge deployment. Low-rank matrix approximation has emerged as a promising approach to reduce CNN parameters, but its application presents challenges including rank selection and performance loss. To address these issues, we propose an efficient training method for CNN compression via dynamic parameter rank pruning. Our approach integrates efficient matrix factorization and novel regularization techniques, forming a robust framework for dynamic rank reduction and model compression. We use Singular Value Decomposition (SVD) to model low-rank convolutional filters and dense weight matrices and we achieve model compression by training the SVD factors with back-propagation in an end-to-end way. We evaluate our method on an array of modern CNNs, including ResNet-18, ResNet-20, and ResNet-32, and datasets like CIFAR-10, CIFAR-100, and ImageNet (2012), showcasing its applicability in computer vision. Our experiments show that the proposed method can yield substantial storage savings while maintaining or even enhancing classification performance.

{{</citation>}}


### (2/99) Transformer-based Video Saliency Prediction with High Temporal Dimension Decoding (Morteza Moradi et al., 2024)

{{<citation>}}

Morteza Moradi, Simone Palazzo, Concetto Spampinato. (2024)  
**Transformer-based Video Saliency Prediction with High Temporal Dimension Decoding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2401.07942v1)  

---


**ABSTRACT**  
In recent years, finding an effective and efficient strategy for exploiting spatial and temporal information has been a hot research topic in video saliency prediction (VSP). With the emergence of spatio-temporal transformers, the weakness of the prior strategies, e.g., 3D convolutional networks and LSTM-based networks, for capturing long-range dependencies has been effectively compensated. While VSP has drawn benefits from spatio-temporal transformers, finding the most effective way for aggregating temporal features is still challenging. To address this concern, we propose a transformer-based video saliency prediction approach with high temporal dimension decoding network (THTD-Net). This strategy accounts for the lack of complex hierarchical interactions between features that are extracted from the transformer-based spatio-temporal encoder: in particular, it does not require multiple decoders and aims at gradually reducing temporal features' dimensions in the decoder. This decoder-based architecture yields comparable performance to multi-branch and over-complicated models on common benchmarks such as DHF1K, UCF-sports and Hollywood-2.

{{</citation>}}


### (3/99) Vertical Federated Image Segmentation (Paul K. Mandal et al., 2024)

{{<citation>}}

Paul K. Mandal, Cole Leo. (2024)  
**Vertical Federated Image Segmentation**  

---
Primary Category: cs.CV  
Categories: C-2-4; I-2-8; I-4; I-4-8, cs-AI, cs-CV, cs-DC, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07931v1)  

---


**ABSTRACT**  
With the popularization of AI solutions for image based problems, there has been a growing concern for both data privacy and acquisition. In a large number of cases, information is located on separate data silos and it can be difficult for a developer to consolidate all of it in a fashion that is appropriate for machine learning model development. Alongside this, a portion of these localized data regions may not have access to a labelled ground truth. This indicates that they have the capacity to reach conclusions numerically, but are not able to assign classifications amid a lack of pertinent information. Such a determination is often negligible, especially when attempting to develop image based solutions that often necessitate this capability. With this being the case, we propose an innovative vertical federated learning (VFL) model architecture that can operate under this common set of conditions. This is the first (and currently the only) implementation of a system that can work under the constraints of a VFL environment and perform image segmentation while maintaining nominal accuracies. We achieved this by utilizing an FCN that boasts the ability to operate on federates that lack labelled data and privately share the respective weights with a central server, that of which hosts the necessary features for classification. Tests were conducted on the CamVid dataset in order to determine the impact of heavy feature compression required for the transfer of information between federates, as well as to reach nominal conclusions about the overall performance metrics when working under such constraints.

{{</citation>}}


### (4/99) Machine Learning Based Object Tracking (Md Rakibul Karim Akanda et al., 2024)

{{<citation>}}

Md Rakibul Karim Akanda, Joshua Reynolds, Treylin Jackson, Milijah Gray. (2024)  
**Machine Learning Based Object Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, physics-app-ph  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2401.07929v1)  

---


**ABSTRACT**  
Machine learning based object detection as well as tracking that object have been performed in this paper. The authors were able to set a range of interest (ROI) around an object using Open Computer Vision, better known as OpenCV. Next a tracking algorithm has been used to maintain tracking on an object while simultaneously operating two servo motors to keep the object centered in the frame. Detailed procedure and code are included in this paper.

{{</citation>}}


### (5/99) $M^{2}$Fusion: Bayesian-based Multimodal Multi-level Fusion on Colorectal Cancer Microsatellite Instability Prediction (Quan Liu et al., 2024)

{{<citation>}}

Quan Liu, Jiawen Yao, Lisha Yao, Xin Chen, Jingren Zhou, Le Lu, Ling Zhang, Zaiyi Liu, Yuankai Huo. (2024)  
**$M^{2}$Fusion: Bayesian-based Multimodal Multi-level Fusion on Colorectal Cancer Microsatellite Instability Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.07854v1)  

---


**ABSTRACT**  
Colorectal cancer (CRC) micro-satellite instability (MSI) prediction on histopathology images is a challenging weakly supervised learning task that involves multi-instance learning on gigapixel images. To date, radiology images have proven to have CRC MSI information and efficient patient imaging techniques. Different data modalities integration offers the opportunity to increase the accuracy and robustness of MSI prediction. Despite the progress in representation learning from the whole slide images (WSI) and exploring the potential of making use of radiology data, CRC MSI prediction remains a challenge to fuse the information from multiple data modalities (e.g., pathology WSI and radiology CT image). In this paper, we propose $M^{2}$Fusion: a Bayesian-based multimodal multi-level fusion pipeline for CRC MSI. The proposed fusion model $M^{2}$Fusion is capable of discovering more novel patterns within and across modalities that are beneficial for predicting MSI than using a single modality alone, as well as other fusion methods. The contribution of the paper is three-fold: (1) $M^{2}$Fusion is the first pipeline of multi-level fusion on pathology WSI and 3D radiology CT image for MSI prediction; (2) CT images are the first time integrated into multimodal fusion for CRC MSI prediction; (3) feature-level fusion strategy is evaluated on both Transformer-based and CNN-based method. Our approach is validated on cross-validation of 352 cases and outperforms either feature-level (0.8177 vs. 0.7908) or decision-level fusion strategy (0.8177 vs. 0.7289) on AUC score.

{{</citation>}}


### (6/99) VeCAF: VLM-empowered Collaborative Active Finetuning with Training Objective Awareness (Rongyu Zhang et al., 2024)

{{<citation>}}

Rongyu Zhang, Zefan Cai, Huanrui Yang, Zidong Liu, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, Baobao Chang, Yuan Du, Li Du, Shanghang Zhang. (2024)  
**VeCAF: VLM-empowered Collaborative Active Finetuning with Training Objective Awareness**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.07853v1)  

---


**ABSTRACT**  
Finetuning a pretrained vision model (PVM) is a common technique for learning downstream vision tasks. The conventional finetuning process with the randomly sampled data points results in diminished training efficiency. To address this drawback, we propose a novel approach, VLM-empowered Collaborative Active Finetuning (VeCAF). VeCAF optimizes a parametric data selection model by incorporating the training objective of the model being tuned. Effectively, this guides the PVM towards the performance goal with improved data and computational efficiency. As vision-language models (VLMs) have achieved significant advancements by establishing a robust connection between image and language domains, we exploit the inherent semantic richness of the text embedding space and utilize text embedding of pretrained VLM models to augment PVM image features for better data selection and finetuning. Furthermore, the flexibility of text-domain augmentation gives VeCAF a unique ability to handle out-of-distribution scenarios without external augmented data. Extensive experiments show the leading performance and high efficiency of VeCAF that is superior to baselines in both in-distribution and out-of-distribution image classification tasks. On ImageNet, VeCAF needs up to 3.3x less training batches to reach the target performance compared to full finetuning and achieves 2.8% accuracy improvement over SOTA methods with the same number of batches.

{{</citation>}}


### (7/99) Uncovering the Full Potential of Visual Grounding Methods in VQA (Daniel Reich et al., 2024)

{{<citation>}}

Daniel Reich, Tanja Schultz. (2024)  
**Uncovering the Full Potential of Visual Grounding Methods in VQA**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.07803v1)  

---


**ABSTRACT**  
Visual Grounding (VG) methods in Visual Question Answering (VQA) attempt to improve VQA performance by strengthening a model's reliance on question-relevant visual information. The presence of such relevant information in the visual input is typically assumed in training and testing. This assumption, however, is inherently flawed when dealing with imperfect image representations common in large-scale VQA, where the information carried by visual features frequently deviates from expected ground-truth contents. As a result, training and testing of VG-methods is performed with largely inaccurate data, which obstructs proper assessment of their potential benefits.   In this work, we demonstrate that current evaluation schemes for VG-methods are problematic due to the flawed assumption of availability of relevant visual information. Our experiments show that the potential benefits of these methods are severely underestimated as a result.

{{</citation>}}


### (8/99) Pedestrian Detection in Low-Light Conditions: A Comprehensive Survey (Bahareh Ghari et al., 2024)

{{<citation>}}

Bahareh Ghari, Ali Tourani, Asadollah Shahbahrami, Georgi Gaydadjiev. (2024)  
**Pedestrian Detection in Low-Light Conditions: A Comprehensive Survey**  

---
Primary Category: cs.CV  
Categories: I-4-8; I-2-10, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07801v1)  

---


**ABSTRACT**  
Pedestrian detection remains a critical problem in various domains, such as computer vision, surveillance, and autonomous driving. In particular, accurate and instant detection of pedestrians in low-light conditions and reduced visibility is of utmost importance for autonomous vehicles to prevent accidents and save lives. This paper aims to comprehensively survey various pedestrian detection approaches, baselines, and datasets that specifically target low-light conditions. The survey discusses the challenges faced in detecting pedestrians at night and explores state-of-the-art methodologies proposed in recent years to address this issue. These methodologies encompass a diverse range, including deep learning-based, feature-based, and hybrid approaches, which have shown promising results in enhancing pedestrian detection performance under challenging lighting conditions. Furthermore, the paper highlights current research directions in the field and identifies potential solutions that merit further investigation by researchers. By thoroughly examining pedestrian detection techniques in low-light conditions, this survey seeks to contribute to the advancement of safer and more reliable autonomous driving systems and other applications related to pedestrian safety. Accordingly, most of the current approaches in the field use deep learning-based image fusion methodologies (i.e., early, halfway, and late fusion) for accurate and reliable pedestrian detection. Moreover, the majority of the works in the field (approximately 48%) have been evaluated on the KAIST dataset, while the real-world video feeds recorded by authors have been used in less than six percent of the works.

{{</citation>}}


### (9/99) Improving OCR Quality in 19th Century Historical Documents Using a Combined Machine Learning Based Approach (David Fleischhacker et al., 2024)

{{<citation>}}

David Fleischhacker, Wolfgang Goederle, Roman Kern. (2024)  
**Improving OCR Quality in 19th Century Historical Documents Using a Combined Machine Learning Based Approach**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2401.07787v1)  

---


**ABSTRACT**  
This paper addresses a major challenge to historical research on the 19th century. Large quantities of sources have become digitally available for the first time, while extraction techniques are lagging behind. Therefore, we researched machine learning (ML) models to recognise and extract complex data structures in a high-value historical primary source, the Schematismus. It records every single person in the Habsburg civil service above a certain hierarchical level between 1702 and 1918 and documents the genesis of the central administration over two centuries. Its complex and intricate structure as well as its enormous size have so far made any more comprehensive analysis of the administrative and social structure of the later Habsburg Empire on the basis of this source impossible. We pursued two central objectives: Primarily, the improvement of the OCR quality, for which we considered an improved structure recognition to be essential; in the further course, it turned out that this also made the extraction of the data structure possible. We chose Faster R-CNN as base for the ML architecture for structure recognition. In order to obtain the required amount of training data quickly and economically, we synthesised Hof- und Staatsschematismus-style data, which we used to train our model. The model was then fine-tuned with a smaller set of manually annotated historical source data. We then used Tesseract-OCR, which was further optimised for the style of our documents, to complete the combined structure extraction and OCR process. Results show a significant decrease in the two standard parameters of OCR-performance, WER and CER (where lower values are better). Combined structure detection and fine-tuned OCR improved CER and WER values by remarkable 71.98 percent (CER) respectively 52.49 percent (WER).

{{</citation>}}


### (10/99) Graph Transformer GANs with Graph Masked Modeling for Architectural Layout Generation (Hao Tang et al., 2024)

{{<citation>}}

Hao Tang, Ling Shao, Nicu Sebe, Luc Van Gool. (2024)  
**Graph Transformer GANs with Graph Masked Modeling for Architectural Layout Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.07721v1)  

---


**ABSTRACT**  
We present a novel graph Transformer generative adversarial network (GTGAN) to learn effective graph node relations in an end-to-end fashion for challenging graph-constrained architectural layout generation tasks. The proposed graph-Transformer-based generator includes a novel graph Transformer encoder that combines graph convolutions and self-attentions in a Transformer to model both local and global interactions across connected and non-connected graph nodes. Specifically, the proposed connected node attention (CNA) and non-connected node attention (NNA) aim to capture the global relations across connected nodes and non-connected nodes in the input graph, respectively. The proposed graph modeling block (GMB) aims to exploit local vertex interactions based on a house layout topology. Moreover, we propose a new node classification-based discriminator to preserve the high-level semantic and discriminative node features for different house components. To maintain the relative spatial relationships between ground truth and predicted graphs, we also propose a novel graph-based cycle-consistency loss. Finally, we propose a novel self-guided pre-training method for graph representation learning. This approach involves simultaneous masking of nodes and edges at an elevated mask ratio (i.e., 40%) and their subsequent reconstruction using an asymmetric graph-centric autoencoder architecture. This method markedly improves the model's learning proficiency and expediency. Experiments on three challenging graph-constrained architectural layout generation tasks (i.e., house layout generation, house roof generation, and building layout generation) with three public datasets demonstrate the effectiveness of the proposed method in terms of objective quantitative scores and subjective visual realism. New state-of-the-art results are established by large margins on these three tasks.

{{</citation>}}


### (11/99) Towards Efficient Diffusion-Based Image Editing with Instant Attention Masks (Siyu Zou et al., 2024)

{{<citation>}}

Siyu Zou, Jiji Tang, Yiyi Zhou, Jing He, Chaoyi Zhao, Rongsheng Zhang, Zhipeng Hu, Xiaoshuai Sun. (2024)  
**Towards Efficient Diffusion-Based Image Editing with Instant Attention Masks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, ImageNet  
[Paper Link](http://arxiv.org/abs/2401.07709v1)  

---


**ABSTRACT**  
Diffusion-based Image Editing (DIE) is an emerging research hot-spot, which often applies a semantic mask to control the target area for diffusion-based editing. However, most existing solutions obtain these masks via manual operations or off-line processing, greatly reducing their efficiency. In this paper, we propose a novel and efficient image editing method for Text-to-Image (T2I) diffusion models, termed Instant Diffusion Editing(InstDiffEdit). In particular, InstDiffEdit aims to employ the cross-modal attention ability of existing diffusion models to achieve instant mask guidance during the diffusion steps. To reduce the noise of attention maps and realize the full automatics, we equip InstDiffEdit with a training-free refinement scheme to adaptively aggregate the attention distributions for the automatic yet accurate mask generation. Meanwhile, to supplement the existing evaluations of DIE, we propose a new benchmark called Editing-Mask to examine the mask accuracy and local editing ability of existing methods. To validate InstDiffEdit, we also conduct extensive experiments on ImageNet and Imagen, and compare it with a bunch of the SOTA methods. The experimental results show that InstDiffEdit not only outperforms the SOTA methods in both image quality and editing results, but also has a much faster inference speed, i.e., +5 to +6 times. Our code available at https://anonymous.4open.science/r/InstDiffEdit-C306/

{{</citation>}}


### (12/99) Fine-Grained Prototypes Distillation for Few-Shot Object Detection (Zichen Wang et al., 2024)

{{<citation>}}

Zichen Wang, Bo Yang, Haonan Yue, Zhenghao Ma. (2024)  
**Fine-Grained Prototypes Distillation for Few-Shot Object Detection**  

---
Primary Category: cs.CV  
Categories: 68T45, I-2-10; I-4-10, cs-CV, cs.CV  
Keywords: Few-Shot, Object Detection  
[Paper Link](http://arxiv.org/abs/2401.07629v1)  

---


**ABSTRACT**  
Few-shot object detection (FSOD) aims at extending a generic detector for novel object detection with only a few training examples. It attracts great concerns recently due to the practical meanings. Meta-learning has been demonstrated to be an effective paradigm for this task. In general, methods based on meta-learning employ an additional support branch to encode novel examples (a.k.a. support images) into class prototypes, which are then fused with query branch to facilitate the model prediction. However, the class-level prototypes are difficult to precisely generate, and they also lack detailed information, leading to instability in performance.New methods are required to capture the distinctive local context for more robust novel object detection. To this end, we propose to distill the most representative support features into fine-grained prototypes. These prototypes are then assigned into query feature maps based on the matching results, modeling the detailed feature relations between two branches. This process is realized by our Fine-Grained Feature Aggregation (FFA) module. Moreover, in terms of high-level feature fusion, we propose Balanced Class-Agnostic Sampling (B-CAS) strategy and Non-Linear Fusion (NLF) module from differenct perspectives. They are complementary to each other and depict the high-level feature relations more effectively. Extensive experiments on PASCAL VOC and MS COCO benchmarks show that our method sets a new state-of-the-art performance in most settings. Our code is available at https://github.com/wangchen1801/FPD.

{{</citation>}}


### (13/99) Collaboratively Self-supervised Video Representation Learning for Action Recognition (Jie Zhang et al., 2024)

{{<citation>}}

Jie Zhang, Zhifan Wan, Lanqing Hu, Stephen Lin, Shuzhe Wu, Shiguang Shan. (2024)  
**Collaboratively Self-supervised Video Representation Learning for Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.07584v1)  

---


**ABSTRACT**  
Considering the close connection between action recognition and human pose estimation, we design a Collaboratively Self-supervised Video Representation (CSVR) learning framework specific to action recognition by jointly considering generative pose prediction and discriminative context matching as pretext tasks. Specifically, our CSVR consists of three branches: a generative pose prediction branch, a discriminative context matching branch, and a video generating branch. Among them, the first one encodes dynamic motion feature by utilizing Conditional-GAN to predict the human poses of future frames, and the second branch extracts static context features by pulling the representations of clips and compressed key frames from the same video together while pushing apart the pairs from different videos. The third branch is designed to recover the current video frames and predict the future ones, for the purpose of collaboratively improving dynamic motion features and static context features. Extensive experiments demonstrate that our method achieves state-of-the-art performance on the UCF101 and HMDB51 datasets.

{{</citation>}}


### (14/99) Exploiting GPT-4 Vision for Zero-shot Point Cloud Understanding (Qi Sun et al., 2024)

{{<citation>}}

Qi Sun, Xiao Cui, Wengang Zhou, Houqiang Li. (2024)  
**Exploiting GPT-4 Vision for Zero-shot Point Cloud Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.07572v1)  

---


**ABSTRACT**  
In this study, we tackle the challenge of classifying the object category in point clouds, which previous works like PointCLIP struggle to address due to the inherent limitations of the CLIP architecture. Our approach leverages GPT-4 Vision (GPT-4V) to overcome these challenges by employing its advanced generative abilities, enabling a more adaptive and robust classification process. We adapt the application of GPT-4V to process complex 3D data, enabling it to achieve zero-shot recognition capabilities without altering the underlying model architecture. Our methodology also includes a systematic strategy for point cloud image visualization, mitigating domain gap and enhancing GPT-4V's efficiency. Experimental validation demonstrates our approach's superiority in diverse scenarios, setting a new benchmark in zero-shot point cloud classification.

{{</citation>}}


### (15/99) Bias-Conflict Sample Synthesis and Adversarial Removal Debias Strategy for Temporal Sentence Grounding in Video (Zhaobo Qi et al., 2024)

{{<citation>}}

Zhaobo Qi, Yibo Yuan, Xiaowen Ruan, Shuhui Wang, Weigang Zhang, Qingming Huang. (2024)  
**Bias-Conflict Sample Synthesis and Adversarial Removal Debias Strategy for Temporal Sentence Grounding in Video**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.07567v1)  

---


**ABSTRACT**  
Temporal Sentence Grounding in Video (TSGV) is troubled by dataset bias issue, which is caused by the uneven temporal distribution of the target moments for samples with similar semantic components in input videos or query texts. Existing methods resort to utilizing prior knowledge about bias to artificially break this uneven distribution, which only removes a limited amount of significant language biases. In this work, we propose the bias-conflict sample synthesis and adversarial removal debias strategy (BSSARD), which dynamically generates bias-conflict samples by explicitly leveraging potentially spurious correlations between single-modality features and the temporal position of the target moments. Through adversarial training, its bias generators continuously introduce biases and generate bias-conflict samples to deceive its grounding model. Meanwhile, the grounding model continuously eliminates the introduced biases, which requires it to model multi-modality alignment information. BSSARD will cover most kinds of coupling relationships and disrupt language and visual biases simultaneously. Extensive experiments on Charades-CD and ActivityNet-CD demonstrate the promising debiasing capability of BSSARD. Source codes are available at https://github.com/qzhb/BSSARD.

{{</citation>}}


### (16/99) Combining Image- and Geometric-based Deep Learning for Shape Regression: A Comparison to Pixel-level Methods for Segmentation in Chest X-Ray (Ron Keuth et al., 2024)

{{<citation>}}

Ron Keuth, Mattias Heinrich. (2024)  
**Combining Image- and Geometric-based Deep Learning for Shape Regression: A Comparison to Pixel-level Methods for Segmentation in Chest X-Ray**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.07542v1)  

---


**ABSTRACT**  
When solving a segmentation task, shaped-base methods can be beneficial compared to pixelwise classification due to geometric understanding of the target object as shape, preventing the generation of anatomical implausible predictions in particular for corrupted data. In this work, we propose a novel hybrid method that combines a lightweight CNN backbone with a geometric neural network (Point Transformer) for shape regression. Using the same CNN encoder, the Point Transformer reaches segmentation quality on per with current state-of-the-art convolutional decoders ($4\pm1.9$ vs $3.9\pm2.9$ error in mm and $85\pm13$ vs $88\pm10$ Dice), but crucially, is more stable w.r.t image distortion, starting to outperform them at a corruption level of 30%. Furthermore, we include the nnU-Net as an upper baseline, which has $3.7\times$ more trainable parameters than our proposed method.

{{</citation>}}


### (17/99) MM-SAP: A Comprehensive Benchmark for Assessing Self-Awareness of Multimodal Large Language Models in Perception (Yuhao Wang et al., 2024)

{{<citation>}}

Yuhao Wang, Yusheng Liao, Heyang Liu, Hongcheng Liu, Yu Wang, Yanfeng Wang. (2024)  
**MM-SAP: A Comprehensive Benchmark for Assessing Self-Awareness of Multimodal Large Language Models in Perception**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07529v1)  

---


**ABSTRACT**  
Multimodal Large Language Models (MLLMs) have shown their remarkable abilities in visual perception and understanding recently. However, how to comprehensively evaluate the capabilities of MLLMs remains a challenge. Most of the existing benchmarks predominantly focus on assessing perception, cognition, and reasoning, neglecting the abilities of self-awareness, referring to the model's recognition of its own capability boundary. In our study, we focus on self-awareness in image perception and introduce the knowledge quadrant for MLLMs, which clearly defines the knowns and unknowns in perception. Based on this, we propose a novel benchmark specifically designed to evaluate the Self-Aware capabilities in Perception for MLLMs(MM-SAP). MM-SAP encompasses three distinct sub-datasets, each focusing on different aspects of self-awareness. We evaluated eight well-known MLLMs using MM-SAP, analyzing their self-awareness and providing detailed insights. Code and data are available at https://github.com/YHWmz/MM-SAP

{{</citation>}}


### (18/99) One for All: Toward Unified Foundation Models for Earth Vision (Zhitong Xiong et al., 2024)

{{<citation>}}

Zhitong Xiong, Yi Wang, Fahong Zhang, Xiao Xiang Zhu. (2024)  
**One for All: Toward Unified Foundation Models for Earth Vision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.07527v1)  

---


**ABSTRACT**  
Foundation models characterized by extensive parameters and trained on large-scale datasets have demonstrated remarkable efficacy across various downstream tasks for remote sensing data. Current remote sensing foundation models typically specialize in a single modality or a specific spatial resolution range, limiting their versatility for downstream datasets. While there have been attempts to develop multi-modal remote sensing foundation models, they typically employ separate vision encoders for each modality or spatial resolution, necessitating a switch in backbones contingent upon the input data. To address this issue, we introduce a simple yet effective method, termed OFA-Net (One-For-All Network): employing a single, shared Transformer backbone for multiple data modalities with different spatial resolutions. Using the masked image modeling mechanism, we pre-train a single Transformer backbone on a curated multi-modal dataset with this simple design. Then the backbone model can be used in different downstream tasks, thus forging a path towards a unified foundation backbone model in Earth vision. The proposed method is evaluated on 12 distinct downstream tasks and demonstrates promising performance.

{{</citation>}}


### (19/99) PolMERLIN: Self-Supervised Polarimetric Complex SAR Image Despeckling with Masked Networks (Shunya Kato et al., 2024)

{{<citation>}}

Shunya Kato, Masaki Saito, Katsuhiko Ishiguro, Sol Cummings. (2024)  
**PolMERLIN: Self-Supervised Polarimetric Complex SAR Image Despeckling with Masked Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.07503v1)  

---


**ABSTRACT**  
Despeckling is a crucial noise reduction task in improving the quality of synthetic aperture radar (SAR) images. Directly obtaining noise-free SAR images is a challenging task that has hindered the development of accurate despeckling algorithms. The advent of deep learning has facilitated the study of denoising models that learn from only noisy SAR images. However, existing methods deal solely with single-polarization images and cannot handle the multi-polarization images captured by modern satellites. In this work, we present an extension of the existing model for generating single-polarization SAR images to handle multi-polarization SAR images. Specifically, we propose a novel self-supervised despeckling approach called channel masking, which exploits the relationship between polarizations. Additionally, we utilize a spatial masking method that addresses pixel-to-pixel correlations to further enhance the performance of our approach. By effectively incorporating multiple polarization information, our method surpasses current state-of-the-art methods in quantitative evaluation in both synthetic and real-world scenarios.

{{</citation>}}


### (20/99) Harnessing Deep Learning and Satellite Imagery for Post-Buyout Land Cover Mapping (Hakan T. Otal et al., 2024)

{{<citation>}}

Hakan T. Otal, Elyse Zavar, Sherri B. Binder, Alex Greer, M. Abdullah Canbaz. (2024)  
**Harnessing Deep Learning and Satellite Imagery for Post-Buyout Land Cover Mapping**  

---
Primary Category: cs.CV  
Categories: 68T45, I-4-9, cs-CV, cs-CY, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.07500v1)  

---


**ABSTRACT**  
Environmental disasters such as floods, hurricanes, and wildfires have increasingly threatened communities worldwide, prompting various mitigation strategies. Among these, property buyouts have emerged as a prominent approach to reducing vulnerability to future disasters. This strategy involves governments purchasing at-risk properties from willing sellers and converting the land into open space, ostensibly reducing future disaster risk and impact. However, the aftermath of these buyouts, particularly concerning land-use patterns and community impacts, remains under-explored. This research aims to fill this gap by employing innovative techniques like satellite imagery analysis and deep learning to study these patterns. To achieve this goal, we employed FEMA's Hazard Mitigation Grant Program (HMGP) buyout dataset, encompassing over 41,004 addresses of these buyout properties from 1989 to 2017. Leveraging Google's Maps Static API, we gathered 40,053 satellite images corresponding to these buyout lands. Subsequently, we implemented five cutting-edge machine learning models to evaluate their performance in classifying land cover types. Notably, this task involved multi-class classification, and our model achieved an outstanding ROC-AUC score of 98.86%

{{</citation>}}


### (21/99) CascadeV-Det: Cascade Point Voting for 3D Object Detection (Yingping Liang et al., 2024)

{{<citation>}}

Yingping Liang, Ying Fu. (2024)  
**CascadeV-Det: Cascade Point Voting for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.07477v1)  

---


**ABSTRACT**  
Anchor-free object detectors are highly efficient in performing point-based prediction without the need for extra post-processing of anchors. However, different from the 2D grids, the 3D points used in these detectors are often far from the ground truth center, making it challenging to accurately regress the bounding boxes. To address this issue, we propose a Cascade Voting (CascadeV) strategy that provides high-quality 3D object detection with point-based prediction. Specifically, CascadeV performs cascade detection using a novel Cascade Voting decoder that combines two new components: Instance Aware Voting (IA-Voting) and a Cascade Point Assignment (CPA) module. The IA-Voting module updates the object features of updated proposal points within the bounding box using conditional inverse distance weighting. This approach prevents features from being aggregated outside the instance and helps improve the accuracy of object detection. Additionally, since model training can suffer from a lack of proposal points with high centerness, we have developed the CPA module to narrow down the positive assignment threshold with cascade stages. This approach relaxes the dependence on proposal centerness in the early stages while ensuring an ample quantity of positives with high centerness in the later stages. Experiments show that FCAF3D with our CascadeV achieves state-of-the-art 3D object detection results with 70.4\% mAP@0.25 and 51.6\% mAP@0.5 on SUN RGB-D and competitive results on ScanNet. Code will be released at https://github.com/Sharpiless/CascadeV-Det

{{</citation>}}


### (22/99) Semantic Segmentation in Multiple Adverse Weather Conditions with Domain Knowledge Retention (Xin Yang et al., 2024)

{{<citation>}}

Xin Yang, Wending Yan, Yuan Yuan, Michael Bi Mi, Robby T. Tan. (2024)  
**Semantic Segmentation in Multiple Adverse Weather Conditions with Domain Knowledge Retention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.07459v1)  

---


**ABSTRACT**  
Semantic segmentation's performance is often compromised when applied to unlabeled adverse weather conditions. Unsupervised domain adaptation is a potential approach to enhancing the model's adaptability and robustness to adverse weather. However, existing methods encounter difficulties when sequentially adapting the model to multiple unlabeled adverse weather conditions. They struggle to acquire new knowledge while also retaining previously learned knowledge.To address these problems, we propose a semantic segmentation method for multiple adverse weather conditions that incorporates adaptive knowledge acquisition, pseudolabel blending, and weather composition replay. Our adaptive knowledge acquisition enables the model to avoid learning from extreme images that could potentially cause the model to forget. In our approach of blending pseudo-labels, we not only utilize the current model but also integrate the previously learned model into the ongoing learning process. This collaboration between the current teacher and the previous model enhances the robustness of the pseudo-labels for the current target. Our weather composition replay mechanism allows the model to continuously refine its previously learned weather information while simultaneously learning from the new target domain. Our method consistently outperforms the stateof-the-art methods, and obtains the best performance with averaged mIoU (%) of 65.7 and the lowest forgetting (%) of 3.6 against 60.1 and 11.3, on the ACDC datasets for a four-target continual multi-target domain adaptation.

{{</citation>}}


### (23/99) Concept-Guided Prompt Learning for Generalization in Vision-Language Models (Yi Zhang et al., 2024)

{{<citation>}}

Yi Zhang, Ce Zhang, Ke Yu, Yushun Tang, Zhihai He. (2024)  
**Concept-Guided Prompt Learning for Generalization in Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07457v1)  

---


**ABSTRACT**  
Contrastive Language-Image Pretraining (CLIP) model has exhibited remarkable efficacy in establishing cross-modal connections between texts and images, yielding impressive performance across a broad spectrum of downstream applications through fine-tuning. However, for generalization tasks, the current fine-tuning methods for CLIP, such as CoOp and CoCoOp, demonstrate relatively low performance on some fine-grained datasets. We recognize the underlying reason is that these previous methods only projected global features into the prompt, neglecting the various visual concepts, such as colors, shapes, and sizes, which are naturally transferable across domains and play a crucial role in generalization tasks. To address this issue, in this work, we propose Concept-Guided Prompt Learning (CPL) for vision-language models. Specifically, we leverage the well-learned knowledge of CLIP to create a visual concept cache to enable concept-guided prompting. In order to refine the text features, we further develop a projector that transforms multi-level visual features into text features. We observe that this concept-guided prompt learning approach is able to achieve enhanced consistency between visual and linguistic modalities. Extensive experimental results demonstrate that our CPL method significantly improves generalization capabilities compared to the current state-of-the-art methods.

{{</citation>}}


## cs.SE (8)



### (24/99) A Novel Approach for Automatic Program Repair using Round-Trip Translation with Large Language Models (Fernando Vallecillos Ruiz et al., 2024)

{{<citation>}}

Fernando Vallecillos Ruiz, Anastasiia Grishina, Max Hort, Leon Moonen. (2024)  
**A Novel Approach for Automatic Program Repair using Round-Trip Translation with Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-LG, cs-SE, cs.SE  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.07994v1)  

---


**ABSTRACT**  
Research shows that grammatical mistakes in a sentence can be corrected by translating it to another language and back using neural machine translation with language models. We investigate whether this correction capability of Large Language Models (LLMs) extends to Automatic Program Repair (APR). Current generative models for APR are pre-trained on source code and fine-tuned for repair. This paper proposes bypassing the fine-tuning step and using Round-Trip Translation (RTT): translation of code from one programming language to another programming or natural language, and back. We hypothesize that RTT with LLMs restores the most commonly seen patterns in code during pre-training, i.e., performs a regression toward the mean, which removes bugs as they are a form of noise w.r.t. the more frequent, natural, bug-free code in the training data. To test this hypothesis, we employ eight recent LLMs pre-trained on code, including the latest GPT versions, and four common program repair benchmarks in Java. We find that RTT with English as an intermediate language repaired 101 of 164 bugs with GPT-4 on the HumanEval-Java dataset. Moreover, 46 of these are unique bugs that are not repaired by other LLMs fine-tuned for APR. Our findings highlight the viability of round-trip translation with LLMs as a technique for automated program repair and its potential for research in software engineering.   Keywords: automated program repair, large language model, machine translation

{{</citation>}}


### (25/99) On Inter-dataset Code Duplication and Data Leakage in Large Language Models (José Antonio Hernández López et al., 2024)

{{<citation>}}

José Antonio Hernández López, Boqi Chen, Tushar Sharma, Dániel Varró. (2024)  
**On Inter-dataset Code Duplication and Data Leakage in Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07930v1)  

---


**ABSTRACT**  
Motivation. Large language models (LLMs) have exhibited remarkable proficiency in diverse software engineering (SE) tasks. Handling such tasks typically involves acquiring foundational coding knowledge on large, general-purpose datasets during a pre-training phase, and subsequently refining on smaller, task-specific datasets as part of a fine-tuning phase.   Problem statement. Data leakage is a well-known issue in training of machine learning models. A manifestation of this issue is the intersection of the training and testing splits. While intra-dataset code duplication examines this intersection within a given dataset and has been addressed in prior research, inter-dataset code duplication, which gauges the overlap between different datasets, remains largely unexplored. If this phenomenon exists, it could compromise the integrity of LLM evaluations because of the inclusion of fine-tuning test samples that were already encountered during pre-training, resulting in inflated performance metrics.   Contribution. This paper explores the phenomenon of inter-dataset code duplication and its impact on evaluating LLMs across diverse SE tasks.   Study design. We conduct an empirical study using the CSN dataset, a widely adopted pre-training dataset, and five fine-tuning datasets used for various SE tasks. We first identify the intersection between the pre-training and fine-tuning datasets using a deduplication process. Then, we fine-tune four models pre-trained on CSN to evaluate their performance on samples encountered during pre-training and those unseen during that phase.   Results. Our findings reveal a potential threat to the evaluation of various LLMs across multiple SE tasks, stemming from the inter-dataset code duplication phenomenon. Moreover, we demonstrate that this threat is accentuated by factors like the LLM's size and the chosen fine-tuning technique.

{{</citation>}}


### (26/99) Towards Automatic Translation of Machine Learning Visual Insights to Analytical Assertions (Arumoy Shome et al., 2024)

{{<citation>}}

Arumoy Shome, Luis Cruz, Arie van Deursen. (2024)  
**Towards Automatic Translation of Machine Learning Visual Insights to Analytical Assertions**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.07696v1)  

---


**ABSTRACT**  
We present our vision for developing an automated tool capable of translating visual properties observed in Machine Learning (ML) visualisations into Python assertions. The tool aims to streamline the process of manually verifying these visualisations in the ML development cycle, which is critical as real-world data and assumptions often change post-deployment. In a prior study, we mined $54,070$ Jupyter notebooks from Github and created a catalogue of $269$ semantically related visualisation-assertion (VA) pairs. Building on this catalogue, we propose to build a taxonomy that organises the VA pairs based on ML verification tasks. The input feature space comprises of a rich source of information mined from the Jupyter notebooks -- visualisations, Python source code, and associated markdown text. The effectiveness of various AI models, including traditional NLP4Code models and modern Large Language Models, will be compared using established machine translation metrics and evaluated through a qualitative study with human participants. The paper also plans to address the challenge of extending the existing VA pair dataset with additional pairs from Kaggle and to compare the tool's effectiveness with commercial generative AI models like ChatGPT. This research not only contributes to the field of ML system validation but also explores novel ways to leverage AI for automating and enhancing software engineering practices in ML.

{{</citation>}}


### (27/99) Selene: Pioneering Automated Proof in Software Verification (Lichen Zhang et al., 2024)

{{<citation>}}

Lichen Zhang, Shuai Lu, Nan Duan. (2024)  
**Selene: Pioneering Automated Proof in Software Verification**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.07663v1)  

---


**ABSTRACT**  
Ensuring correctness is a pivotal aspect of software engineering. Among the various strategies available, software verification offers a definitive assurance of correctness. Nevertheless, writing verification proofs is resource-intensive and manpower-consuming, and there is a great need to automate this process. We introduce Selene in this paper, which is the first project-level automated proof benchmark constructed based on the real-world industrial-level project of the seL4 operating system microkernel. Selene provides a comprehensive framework for end-to-end evaluation and a lightweight verification environment. Our experimental results with advanced LLMs, such as GPT-3.5-turbo and GPT-4, highlight the capabilities of large language models (LLMs) in the domain of automated proof generation. Additionally, our further proposed augmentations indicate that the challenges presented by Selene can be mitigated in future research endeavors.

{{</citation>}}


### (28/99) MLAD: A Unified Model for Multi-system Log Anomaly Detection (Runqiang Zang et al., 2024)

{{<citation>}}

Runqiang Zang, Hongcheng Guo, Jian Yang, Jiaheng Liu, Zhoujun Li, Tieqiao Zheng, Xu Shi, Liangfan Zheng, Bo Zhang. (2024)  
**MLAD: A Unified Model for Multi-system Log Anomaly Detection**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-LG, cs-SE, cs.SE  
Keywords: Anomaly Detection, Attention  
[Paper Link](http://arxiv.org/abs/2401.07655v1)  

---


**ABSTRACT**  
In spite of the rapid advancements in unsupervised log anomaly detection techniques, the current mainstream models still necessitate specific training for individual system datasets, resulting in costly procedures and limited scalability due to dataset size, thereby leading to performance bottlenecks. Furthermore, numerous models lack cognitive reasoning capabilities, posing challenges in direct transferability to similar systems for effective anomaly detection. Additionally, akin to reconstruction networks, these models often encounter the "identical shortcut" predicament, wherein the majority of system logs are classified as normal, erroneously predicting normal classes when confronted with rare anomaly logs due to reconstruction errors.   To address the aforementioned issues, we propose MLAD, a novel anomaly detection model that incorporates semantic relational reasoning across multiple systems. Specifically, we employ Sentence-bert to capture the similarities between log sequences and convert them into highly-dimensional learnable semantic vectors. Subsequently, we revamp the formulas of the Attention layer to discern the significance of each keyword in the sequence and model the overall distribution of the multi-system dataset through appropriate vector space diffusion. Lastly, we employ a Gaussian mixture model to highlight the uncertainty of rare words pertaining to the "identical shortcut" problem, optimizing the vector space of the samples using the maximum expectation model. Experiments on three real-world datasets demonstrate the superiority of MLAD.

{{</citation>}}


### (29/99) TDD Without Tears: Towards Test Case Generation from Requirements through Deep Reinforcement Learning (Wannita Takerngsaksiri et al., 2024)

{{<citation>}}

Wannita Takerngsaksiri, Rujikorn Charakorn, Chakkrit Tantithamthavorn, Yuan-Fang Li. (2024)  
**TDD Without Tears: Towards Test Case Generation from Requirements through Deep Reinforcement Learning**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: GPT, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.07576v1)  

---


**ABSTRACT**  
Test-driven development (TDD) is a widely-employed software development practice that mandates writing test cases based on requirements before writing the actual code. While writing test cases is the centerpiece of TDD, it is time-consuming, expensive, and often shunned by developers. To address these issues associated with TDD, automated test case generation approaches have recently been investigated. Such approaches take source code as input, but not the requirements. Therefore, existing work does not fully support true TDD, as actual code is required to generate test cases. In addition, current deep learning-based test case generation approaches are trained with one learning objective, i.e., to generate test cases that are exactly matched with the ground-truth test cases. However, such approaches may limit the model's ability to generate different yet correct test cases. In this paper, we introduce PyTester, a Text-to-Testcase generation approach that can automatically generate syntactically correct, executable, complete, and effective test cases while being aligned with a given natural language requirement. We evaluate PyTester on the public APPS benchmark dataset, and the results show that our Deep RL approach enables PyTester, a small language model, to outperform much larger language models like GPT3.5, StarCoder, and InCoder. Our findings suggest that future research could consider improving small over large LMs for better resource efficiency by integrating the SE domain knowledge into the design of reinforcement learning architecture.

{{</citation>}}


### (30/99) Exploring the Potential of Large Language Models in Self-adaptive Systems (Jialong Li et al., 2024)

{{<citation>}}

Jialong Li, Mingyue Zhang, Nianyu Li, Danny Weyns, Zhi Jin, Kenji Tei. (2024)  
**Exploring the Potential of Large Language Models in Self-adaptive Systems**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07534v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), with their abilities in knowledge acquisition and reasoning, can potentially enhance the various aspects of Self-adaptive Systems (SAS). Yet, the potential of LLMs in SAS remains largely unexplored and ambiguous, due to the lack of literature from flagship conferences or journals in the field, such as SEAMS and TAAS. The interdisciplinary nature of SAS suggests that drawing and integrating ideas from related fields, such as software engineering and autonomous agents, could unveil innovative research directions for LLMs within SAS. To this end, this paper reports the results of a literature review of studies in relevant fields, summarizes and classifies the studies relevant to SAS, and outlines their potential to specific aspects of SAS.

{{</citation>}}


### (31/99) Your Instructions Are Not Always Helpful: Assessing the Efficacy of Instruction Fine-tuning for Software Vulnerability Detection (Imam Nur Bani Yusuf et al., 2024)

{{<citation>}}

Imam Nur Bani Yusuf, Lingxiao Jiang. (2024)  
**Your Instructions Are Not Always Helpful: Assessing the Efficacy of Instruction Fine-tuning for Software Vulnerability Detection**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2401.07466v1)  

---


**ABSTRACT**  
Software, while beneficial, poses potential cybersecurity risks due to inherent vulnerabilities. Detecting these vulnerabilities is crucial, and deep learning has shown promise as an effective tool for this task due to its ability to perform well without extensive feature engineering. However, a challenge in deploying deep learning for vulnerability detection is the limited availability of training data. Recent research highlights the deep learning efficacy in diverse tasks. This success is attributed to instruction fine-tuning, a technique that remains under-explored in the context of vulnerability detection. This paper investigates the capability of models, specifically a recent language model, to generalize beyond the programming languages used in their training data. It also examines the role of natural language instructions in enhancing this generalization. Our study evaluates the model performance on a real-world dataset to predict vulnerable code. We present key insights and lessons learned, contributing to understanding the deep learning application in software vulnerability detection.

{{</citation>}}


## cs.LG (11)



### (32/99) Robustness Against Adversarial Attacks via Learning Confined Adversarial Polytopes (Shayan Mohajer Hamidi et al., 2024)

{{<citation>}}

Shayan Mohajer Hamidi, Linfeng Ye. (2024)  
**Robustness Against Adversarial Attacks via Learning Confined Adversarial Polytopes**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2401.07991v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) could be deceived by generating human-imperceptible perturbations of clean samples. Therefore, enhancing the robustness of DNNs against adversarial attacks is a crucial task. In this paper, we aim to train robust DNNs by limiting the set of outputs reachable via a norm-bounded perturbation added to a clean sample. We refer to this set as adversarial polytope, and each clean sample has a respective adversarial polytope. Indeed, if the respective polytopes for all the samples are compact such that they do not intersect the decision boundaries of the DNN, then the DNN is robust against adversarial samples. Hence, the inner-working of our algorithm is based on learning \textbf{c}onfined \textbf{a}dversarial \textbf{p}olytopes (CAP). By conducting a thorough set of experiments, we demonstrate the effectiveness of CAP over existing adversarial robustness methods in improving the robustness of models against state-of-the-art attacks including AutoAttack.

{{</citation>}}


### (33/99) GD-CAF: Graph Dual-stream Convolutional Attention Fusion for Precipitation Nowcasting (Lorand Vatamany et al., 2024)

{{<citation>}}

Lorand Vatamany, Siamak Mehrkanoon. (2024)  
**GD-CAF: Graph Dual-stream Convolutional Attention Fusion for Precipitation Nowcasting**  

---
Primary Category: cs.LG  
Categories: I-2; I-5, cs-CV, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.07958v1)  

---


**ABSTRACT**  
Accurate precipitation nowcasting is essential for various purposes, including flood prediction, disaster management, optimizing agricultural activities, managing transportation routes and renewable energy. While several studies have addressed this challenging task from a sequence-to-sequence perspective, most of them have focused on a single area without considering the existing correlation between multiple disjoint regions. In this paper, we formulate precipitation nowcasting as a spatiotemporal graph sequence nowcasting problem. In particular, we introduce Graph Dual-stream Convolutional Attention Fusion (GD-CAF), a novel approach designed to learn from historical spatiotemporal graph of precipitation maps and nowcast future time step ahead precipitation at different spatial locations. GD-CAF consists of spatio-temporal convolutional attention as well as gated fusion modules which are equipped with depthwise-separable convolutional operations. This enhancement enables the model to directly process the high-dimensional spatiotemporal graph of precipitation maps and exploits higher-order correlations between the data dimensions. We evaluate our model on seven years of precipitation maps across Europe and its neighboring areas collected from the ERA5 dataset, provided by Copernicus. The model receives a fully connected graph in which each node represents historical observations from a specific region on the map. Consequently, each node contains a 3D tensor with time, height, and width dimensions. Experimental results demonstrate that the proposed GD-CAF model outperforms the other examined models. Furthermore, the averaged seasonal spatial and temporal attention scores over the test set are visualized to provide additional insights about the strongest connections between different regions or time steps. These visualizations shed light on the decision-making process of our model.

{{</citation>}}


### (34/99) The Chronicles of RAG: The Retriever, the Chunk and the Generator (Paulo Finardi et al., 2024)

{{<citation>}}

Paulo Finardi, Leonardo Avila, Rodrigo Castaldoni, Pedro Gengo, Celio Larcher, Marcos Piau, Pablo Costa, Vinicius Caridá. (2024)  
**The Chronicles of RAG: The Retriever, the Chunk and the Generator**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.LG  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2401.07883v1)  

---


**ABSTRACT**  
Retrieval Augmented Generation (RAG) has become one of the most popular paradigms for enabling LLMs to access external data, and also as a mechanism for grounding to mitigate against hallucinations. When implementing RAG you can face several challenges like effective integration of retrieval models, efficient representation learning, data diversity, computational efficiency optimization, evaluation, and quality of text generation. Given all these challenges, every day a new technique to improve RAG appears, making it unfeasible to experiment with all combinations for your problem. In this context, this paper presents good practices to implement, optimize, and evaluate RAG for the Brazilian Portuguese language, focusing on the establishment of a simple pipeline for inference and experiments. We explored a diverse set of methods to answer questions about the first Harry Potter book. To generate the answers we used the OpenAI's gpt-4, gpt-4-1106-preview, gpt-3.5-turbo-1106, and Google's Gemini Pro. Focusing on the quality of the retriever, our approach achieved an improvement of MRR@10 by 35.4% compared to the baseline. When optimizing the input size in the application, we observed that it is possible to further enhance it by 2.4%. Finally, we present the complete architecture of the RAG with our recommendations. As result, we moved from a baseline of 57.88% to a maximum relative score of 98.61%.

{{</citation>}}


### (35/99) Do stable neural networks exist for classification problems? -- A new view on stability in AI (Z. N. D. Liu et al., 2024)

{{<citation>}}

Z. N. D. Liu, A. C. Hansen. (2024)  
**Do stable neural networks exist for classification problems? -- A new view on stability in AI**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-FA  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07874v1)  

---


**ABSTRACT**  
In deep learning (DL) the instability phenomenon is widespread and well documented, most commonly using the classical measure of stability, the Lipschitz constant. While a small Lipchitz constant is traditionally viewed as guarantying stability, it does not capture the instability phenomenon in DL for classification well. The reason is that a classification function -- which is the target function to be approximated -- is necessarily discontinuous, thus having an 'infinite' Lipchitz constant. As a result, the classical approach will deem every classification function unstable, yet basic classification functions a la 'is there a cat in the image?' will typically be locally very 'flat' -- and thus locally stable -- except at the decision boundary. The lack of an appropriate measure of stability hinders a rigorous theory for stability in DL, and consequently, there are no proper approximation theoretic results that can guarantee the existence of stable networks for classification functions. In this paper we introduce a novel stability measure $\mathscr{S}(f)$, for any classification function $f$, appropriate to study the stability of discontinuous functions and their approximations. We further prove two approximation theorems: First, for any $\epsilon > 0$ and any classification function $f$ on a \emph{compact set}, there is a neural network (NN) $\psi$, such that $\psi - f \neq 0$ only on a set of measure $< \epsilon$, moreover, $\mathscr{S}(\psi) \geq \mathscr{S}(f) - \epsilon$ (as accurate and stable as $f$ up to $\epsilon$). Second, for any classification function $f$ and $\epsilon > 0$, there exists a NN $\psi$ such that $\psi = f$ on the set of points that are at least $\epsilon$ away from the decision boundary.

{{</citation>}}


### (36/99) The ODE Method for Stochastic Approximation and Reinforcement Learning with Markovian Noise (Shuze Liu et al., 2024)

{{<citation>}}

Shuze Liu, Shuhang Chen, Shangtong Zhang. (2024)  
**The ODE Method for Stochastic Approximation and Reinforcement Learning with Markovian Noise**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.07844v1)  

---


**ABSTRACT**  
Stochastic approximation is a class of algorithms that update a vector iteratively, incrementally, and stochastically, including, e.g., stochastic gradient descent and temporal difference learning. One fundamental challenge in analyzing a stochastic approximation algorithm is to establish its stability, i.e., to show that the stochastic vector iterates are bounded almost surely. In this paper, we extend the celebrated Borkar-Meyn theorem for stability from the Martingale difference noise setting to the Markovian noise setting, which greatly improves its applicability in reinforcement learning, especially in those off-policy reinforcement learning algorithms with linear function approximation and eligibility traces. Central to our analysis is the diminishing asymptotic rate of change of a few functions, which is implied by both a form of strong law of large numbers and a commonly used V4 Lyapunov drift condition and trivially holds if the Markov chain is finite and irreducible.

{{</citation>}}


### (37/99) Empirical Evidence for the Fragment level Understanding on Drug Molecular Structure of LLMs (Xiuyuan Hu et al., 2024)

{{<citation>}}

Xiuyuan Hu, Guoqing Liu, Yang Zhao, Hao Zhang. (2024)  
**Empirical Evidence for the Fragment level Understanding on Drug Molecular Structure of LLMs**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs.LG, q-bio-BM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07657v1)  

---


**ABSTRACT**  
AI for drug discovery has been a research hotspot in recent years, and SMILES-based language models has been increasingly applied in drug molecular design. However, no work has explored whether and how language models understand the chemical spatial structure from 1D sequences. In this work, we pre-train a transformer model on chemical language and fine-tune it toward drug design objectives, and investigate the correspondence between high-frequency SMILES substrings and molecular fragments. The results indicate that language models can understand chemical structures from the perspective of molecular fragments, and the structural knowledge learned through fine-tuning is reflected in the high-frequency SMILES substrings generated by the model.

{{</citation>}}


### (38/99) Compute-Efficient Active Learning (Gábor Németh et al., 2024)

{{<citation>}}

Gábor Németh, Tamás Matuszka. (2024)  
**Compute-Efficient Active Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2401.07639v1)  

---


**ABSTRACT**  
Active learning, a powerful paradigm in machine learning, aims at reducing labeling costs by selecting the most informative samples from an unlabeled dataset. However, the traditional active learning process often demands extensive computational resources, hindering scalability and efficiency. In this paper, we address this critical issue by presenting a novel method designed to alleviate the computational burden associated with active learning on massive datasets. To achieve this goal, we introduce a simple, yet effective method-agnostic framework that outlines how to strategically choose and annotate data points, optimizing the process for efficiency while maintaining model performance. Through case studies, we demonstrate the effectiveness of our proposed method in reducing computational costs while maintaining or, in some cases, even surpassing baseline model outcomes. Code is available at https://github.com/aimotive/Compute-Efficient-Active-Learning.

{{</citation>}}


### (39/99) Safe Reinforcement Learning with Free-form Natural Language Constraints and Pre-Trained Language Models (Xingzhou Lou et al., 2024)

{{<citation>}}

Xingzhou Lou, Junge Zhang, Ziyan Wang, Kaiqi Huang, Yali Du. (2024)  
**Safe Reinforcement Learning with Free-form Natural Language Constraints and Pre-Trained Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.07553v1)  

---


**ABSTRACT**  
Safe reinforcement learning (RL) agents accomplish given tasks while adhering to specific constraints. Employing constraints expressed via easily-understandable human language offers considerable potential for real-world applications due to its accessibility and non-reliance on domain expertise. Previous safe RL methods with natural language constraints typically adopt a recurrent neural network, which leads to limited capabilities when dealing with various forms of human language input. Furthermore, these methods often require a ground-truth cost function, necessitating domain expertise for the conversion of language constraints into a well-defined cost function that determines constraint violation. To address these issues, we proposes to use pre-trained language models (LM) to facilitate RL agents' comprehension of natural language constraints and allow them to infer costs for safe policy learning. Through the use of pre-trained LMs and the elimination of the need for a ground-truth cost, our method enhances safe policy learning under a diverse set of human-derived free-form natural language constraints. Experiments on grid-world navigation and robot control show that the proposed method can achieve strong performance while adhering to given constraints. The usage of pre-trained LMs allows our method to comprehend complicated constraints and learn safe policies without the need for ground-truth cost at any stage of training or evaluation. Extensive ablation studies are conducted to demonstrate the efficacy of each part of our method.

{{</citation>}}


### (40/99) Robust Semi-Supervised Learning for Self-learning Open-World Classes (Wenjuan Xi et al., 2024)

{{<citation>}}

Wenjuan Xi, Xin Song, Weili Guo, Yang Yang. (2024)  
**Robust Semi-Supervised Learning for Self-learning Open-World Classes**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2401.07551v1)  

---


**ABSTRACT**  
Existing semi-supervised learning (SSL) methods assume that labeled and unlabeled data share the same class space. However, in real-world applications, unlabeled data always contain classes not present in the labeled set, which may cause classification performance degradation of known classes. Therefore, open-world SSL approaches are researched to handle the presence of multiple unknown classes in the unlabeled data, which aims to accurately classify known classes while fine-grained distinguishing different unknown classes. To address this challenge, in this paper, we propose an open-world SSL method for Self-learning Open-world Classes (SSOC), which can explicitly self-learn multiple unknown classes. Specifically, SSOC first defines class center tokens for both known and unknown classes and autonomously learns token representations according to all samples with the cross-attention mechanism. To effectively discover novel classes, SSOC further designs a pairwise similarity loss in addition to the entropy loss, which can wisely exploit the information available in unlabeled data from instances' predictions and relationships. Extensive experiments demonstrate that SSOC outperforms the state-of-the-art baselines on multiple popular classification benchmarks. Specifically, on the ImageNet-100 dataset with a novel ratio of 90%, SSOC achieves a remarkable 22% improvement.

{{</citation>}}


### (41/99) Temporal Link Prediction Using Graph Embedding Dynamics (Sanaz Hasanzadeh Fard et al., 2024)

{{<citation>}}

Sanaz Hasanzadeh Fard, Mohammad Ghassemi. (2024)  
**Temporal Link Prediction Using Graph Embedding Dynamics**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.07516v1)  

---


**ABSTRACT**  
Graphs are a powerful representation tool in machine learning applications, with link prediction being a key task in graph learning. Temporal link prediction in dynamic networks is of particular interest due to its potential for solving complex scientific and real-world problems. Traditional approaches to temporal link prediction have focused on finding the aggregation of dynamics of the network as a unified output. In this study, we propose a novel perspective on temporal link prediction by defining nodes as Newtonian objects and incorporating the concept of velocity to predict network dynamics. By computing more specific dynamics of each node, rather than overall dynamics, we improve both accuracy and explainability in predicting future connections. We demonstrate the effectiveness of our approach using two datasets, including 17 years of co-authorship data from PubMed. Experimental results show that our temporal graph embedding dynamics approach improves downstream classification models' ability to predict future collaboration efficacy in co-authorship networks by 17.34% (AUROC improvement relative to the baseline model). Furthermore, our approach offers an interpretable layer over traditional approaches to address the temporal link prediction problem.

{{</citation>}}


### (42/99) Harnessing the Power of Beta Scoring in Deep Active Learning for Multi-Label Text Classification (Wei Tan et al., 2024)

{{<citation>}}

Wei Tan, Ngoc Dang Nguyen, Lan Du, Wray Buntine. (2024)  
**Harnessing the Power of Beta Scoring in Deep Active Learning for Multi-Label Text Classification**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Active Learning, Text Classification  
[Paper Link](http://arxiv.org/abs/2401.07395v1)  

---


**ABSTRACT**  
Within the scope of natural language processing, the domain of multi-label text classification is uniquely challenging due to its expansive and uneven label distribution. The complexity deepens due to the demand for an extensive set of annotated data for training an advanced deep learning model, especially in specialized fields where the labeling task can be labor-intensive and often requires domain-specific knowledge. Addressing these challenges, our study introduces a novel deep active learning strategy, capitalizing on the Beta family of proper scoring rules within the Expected Loss Reduction framework. It computes the expected increase in scores using the Beta Scoring Rules, which are then transformed into sample vector representations. These vector representations guide the diverse selection of informative samples, directly linking this process to the model's expected proper score. Comprehensive evaluations across both synthetic and real datasets reveal our method's capability to often outperform established acquisition techniques in multi-label text classification, presenting encouraging outcomes across various architectural and dataset scenarios.

{{</citation>}}


## cs.CL (29)



### (43/99) Leveraging External Knowledge Resources to Enable Domain-Specific Comprehension (Saptarshi Sengupta et al., 2024)

{{<citation>}}

Saptarshi Sengupta, Connor Heaton, Prasenjit Mitra, Soumalya Sarkar. (2024)  
**Leveraging External Knowledge Resources to Enable Domain-Specific Comprehension**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Machine Reading Comprehension, NLP, QA  
[Paper Link](http://arxiv.org/abs/2401.07977v1)  

---


**ABSTRACT**  
Machine Reading Comprehension (MRC) has been a long-standing problem in NLP and, with the recent introduction of the BERT family of transformer based language models, it has come a long way to getting solved. Unfortunately, however, when BERT variants trained on general text corpora are applied to domain-specific text, their performance inevitably degrades on account of the domain shift i.e. genre/subject matter discrepancy between the training and downstream application data. Knowledge graphs act as reservoirs for either open or closed domain information and prior studies have shown that they can be used to improve the performance of general-purpose transformers in domain-specific applications. Building on existing work, we introduce a method using Multi-Layer Perceptrons (MLPs) for aligning and integrating embeddings extracted from knowledge graphs with the embeddings spaces of pre-trained language models (LMs). We fuse the aligned embeddings with open-domain LMs BERT and RoBERTa, and fine-tune them for two MRC tasks namely span detection (COVID-QA) and multiple-choice questions (PubMedQA). On the COVID-QA dataset, we see that our approach allows these models to perform similar to their domain-specific counterparts, Bio/Sci-BERT, as evidenced by the Exact Match (EM) metric. With regards to PubMedQA, we observe an overall improvement in accuracy while the F1 stays relatively the same over the domain-specific models.

{{</citation>}}


### (44/99) A Study on Large Language Models' Limitations in Multiple-Choice Question Answering (Aisha Khatun et al., 2024)

{{<citation>}}

Aisha Khatun, Daniel G. Brown. (2024)  
**A Study on Large Language Models' Limitations in Multiple-Choice Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.07955v1)  

---


**ABSTRACT**  
The widespread adoption of Large Language Models (LLMs) has become commonplace, particularly with the emergence of open-source models. More importantly, smaller models are well-suited for integration into consumer devices and are frequently employed either as standalone solutions or as subroutines in various AI tasks. Despite their ubiquitous use, there is no systematic analysis of their specific capabilities and limitations. In this study, we tackle one of the most widely used tasks - answering Multiple Choice Question (MCQ). We analyze 26 small open-source models and find that 65% of the models do not understand the task, only 4 models properly select an answer from the given choices, and only 5 of these models are choice order independent. These results are rather alarming given the extensive use of MCQ tests with these models. We recommend exercising caution and testing task understanding before using MCQ to evaluate LLMs in any field whatsoever.

{{</citation>}}


### (45/99) SciGLM: Training Scientific Language Models with Self-Reflective Instruction Annotation and Tuning (Dan Zhang et al., 2024)

{{<citation>}}

Dan Zhang, Ziniu Hu, Sining Zhoubian, Zhengxiao Du, Kaiyu Yang, Zihan Wang, Yisong Yue, Yuxiao Dong, Jie Tang. (2024)  
**SciGLM: Training Scientific Language Models with Self-Reflective Instruction Annotation and Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLM, Language Model  
[Paper Link](http://arxiv.org/abs/2401.07950v1)  

---


**ABSTRACT**  
\label{sec:abstract} Large Language Models (LLMs) have shown promise in assisting scientific discovery. However, such applications are currently limited by LLMs' deficiencies in understanding intricate scientific concepts, deriving symbolic equations, and solving advanced numerical calculations. To bridge these gaps, we introduce SciGLM, a suite of scientific language models able to conduct college-level scientific reasoning. Central to our approach is a novel self-reflective instruction annotation framework to address the data scarcity challenge in the science domain. This framework leverages existing LLMs to generate step-by-step reasoning for unlabelled scientific questions, followed by a process of self-reflective critic-and-revise. Applying this framework, we curated SciInstruct, a diverse and high-quality dataset encompassing mathematics, physics, chemistry, and formal proofs. We fine-tuned the ChatGLM family of language models with SciInstruct, enhancing their capabilities in scientific and mathematical reasoning. Remarkably, SciGLM consistently improves both the base model (ChatGLM3-6B-Base) and larger-scale models (12B and 32B), without sacrificing the language understanding capabilities of the base model. This makes SciGLM a suitable foundational model to facilitate diverse scientific discovery tasks. For the benefit of the wider research community, we release SciInstruct, SciGLM, alongside a self-reflective framework and fine-tuning code at \url{https://github.com/THUDM/SciGLM}.

{{</citation>}}


### (46/99) SemEval-2017 Task 4: Sentiment Analysis in Twitter using BERT (Rupak Kumar Das et al., 2024)

{{<citation>}}

Rupak Kumar Das, Dr. Ted Pedersen. (2024)  
**SemEval-2017 Task 4: Sentiment Analysis in Twitter using BERT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Sentiment Analysis, Twitter  
[Paper Link](http://arxiv.org/abs/2401.07944v1)  

---


**ABSTRACT**  
This paper uses the BERT model, which is a transformer-based architecture, to solve task 4A, English Language, Sentiment Analysis in Twitter of SemEval2017. BERT is a very powerful large language model for classification tasks when the amount of training data is small. For this experiment, we have used the BERT{\textsubscript{\tiny BASE}} model, which has 12 hidden layers. This model provides better accuracy, precision, recall, and f1 score than the Naive Bayes baseline model. It performs better in binary classification subtasks than the multi-class classification subtasks. We also considered all kinds of ethical issues during this experiment, as Twitter data contains personal and sensible information. The dataset and code used in our experiment can be found in this GitHub repository.

{{</citation>}}


### (47/99) Can Large Language Models Explain Themselves? (Andreas Madsen et al., 2024)

{{<citation>}}

Andreas Madsen, Sarath Chandar, Siva Reddy. (2024)  
**Can Large Language Models Explain Themselves?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Falcon, Language Model  
[Paper Link](http://arxiv.org/abs/2401.07927v1)  

---


**ABSTRACT**  
Instruction-tuned large language models (LLMs) excel at many tasks, and will even provide explanations for their behavior. Since these models are directly accessible to the public, there is a risk that convincing and wrong explanations can lead to unsupported confidence in LLMs. Therefore, interpretability-faithfulness of self-explanations is an important consideration for AI Safety. Assessing the interpretability-faithfulness of these explanations, termed self-explanations, is challenging as the models are too complex for humans to annotate what is a correct explanation. To address this, we propose employing self-consistency checks as a measure of faithfulness. For example, if an LLM says a set of words is important for making a prediction, then it should not be able to make the same prediction without these words. While self-consistency checks are a common approach to faithfulness, they have not previously been applied to LLM's self-explanations. We apply self-consistency checks to three types of self-explanations: counterfactuals, importance measures, and redactions. Our work demonstrate that faithfulness is both task and model dependent, e.g., for sentiment classification, counterfactual explanations are more faithful for Llama2, importance measures for Mistral, and redaction for Falcon 40B. Finally, our findings are robust to prompt-variations.

{{</citation>}}


### (48/99) Word Boundary Information Isn't Useful for Encoder Language Models (Edward Gow-Smith et al., 2024)

{{<citation>}}

Edward Gow-Smith, Dylan Phelps, Harish Tayyar Madabushi, Carolina Scarton, Aline Villavicencio. (2024)  
**Word Boundary Information Isn't Useful for Encoder Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLUE, Language Model, NER, NLP  
[Paper Link](http://arxiv.org/abs/2401.07923v1)  

---


**ABSTRACT**  
All existing transformer-based approaches to NLP using subword tokenisation algorithms encode whitespace (word boundary information) through the use of special space symbols (such as \#\# or \_) forming part of tokens. These symbols have been shown to a) lead to reduced morphological validity of tokenisations, and b) give substantial vocabulary redundancy. As such, removing these symbols has been shown to have a beneficial effect on the processing of morphologically complex words for transformer encoders in the pretrain-finetune paradigm. In this work, we explore whether word boundary information is at all useful to such models. In particular, we train transformer encoders across four different training scales, and investigate several alternative approaches to including word boundary information, evaluating on a range of tasks across different domains and problem set-ups: GLUE (for sentence-level classification), NER (for token-level classification), and two classification datasets involving complex words (Superbizarre and FLOTA). Overall, through an extensive experimental setup that includes the pre-training of 29 models, we find no substantial improvements from our alternative approaches, suggesting that modifying tokenisers to remove word boundary information isn't leading to a loss of useful information.

{{</citation>}}


### (49/99) The Pitfalls of Defining Hallucination (Kees van Deemter, 2024)

{{<citation>}}

Kees van Deemter. (2024)  
**The Pitfalls of Defining Hallucination**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2401.07897v1)  

---


**ABSTRACT**  
Despite impressive advances in Natural Language Generation (NLG) and Large Language Models (LLMs), researchers are still unclear about important aspects of NLG evaluation. To substantiate this claim, I examine current classifications of hallucination and omission in Data-text NLG, and I propose a logic-based synthesis of these classfications. I conclude by highlighting some remaining limitations of all current thinking about hallucination and by discussing implications for LLMs.

{{</citation>}}


### (50/99) EMBRE: Entity-aware Masking for Biomedical Relation Extraction (Mingjie Li et al., 2024)

{{<citation>}}

Mingjie Li, Karin Verspoor. (2024)  
**EMBRE: Entity-aware Masking for Biomedical Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NER, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2401.07877v1)  

---


**ABSTRACT**  
Information extraction techniques, including named entity recognition (NER) and relation extraction (RE), are crucial in many domains to support making sense of vast amounts of unstructured text data by identifying and connecting relevant information. Such techniques can assist researchers in extracting valuable insights. In this paper, we introduce the Entity-aware Masking for Biomedical Relation Extraction (EMBRE) method for biomedical relation extraction, as applied in the context of the BioRED challenge Task 1, in which human-annotated entities are provided as input. Specifically, we integrate entity knowledge into a deep neural network by pretraining the backbone model with an entity masking objective. We randomly mask named entities for each instance and let the model identify the masked entity along with its type. In this way, the model is capable of learning more specific knowledge and more robust representations. Then, we utilize the pre-trained model as our backbone to encode language representations and feed these representations into two multilayer perceptron (MLPs) to predict the logits for relation and novelty, respectively. The experimental results demonstrate that our proposed method can improve the performances of entity pair, relation and novelty extraction over our baseline.

{{</citation>}}


### (51/99) The What, Why, and How of Context Length Extension Techniques in Large Language Models -- A Detailed Survey (Saurav Pawar et al., 2024)

{{<citation>}}

Saurav Pawar, S. M Towhidul Islam Tonmoy, S M Mehedi Zaman, Vinija Jain, Aman Chadha, Amitava Das. (2024)  
**The What, Why, and How of Context Length Extension Techniques in Large Language Models -- A Detailed Survey**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.07872v1)  

---


**ABSTRACT**  
The advent of Large Language Models (LLMs) represents a notable breakthrough in Natural Language Processing (NLP), contributing to substantial progress in both text comprehension and generation. However, amidst these advancements, it is noteworthy that LLMs often face a limitation in terms of context length extrapolation. Understanding and extending the context length for LLMs is crucial in enhancing their performance across various NLP applications. In this survey paper, we delve into the multifaceted aspects of exploring why it is essential, and the potential transformations that superior techniques could bring to NLP applications. We study the inherent challenges associated with extending context length and present an organized overview of the existing strategies employed by researchers. Additionally, we discuss the intricacies of evaluating context extension techniques and highlight the open challenges that researchers face in this domain. Furthermore, we explore whether there is a consensus within the research community regarding evaluation standards and identify areas where further agreement is needed. This comprehensive survey aims to serve as a valuable resource for researchers, guiding them through the nuances of context length extension techniques and fostering discussions on future advancements in this evolving field.

{{</citation>}}


### (52/99) Authorship Obfuscation in Multilingual Machine-Generated Text Detection (Dominik Macko et al., 2024)

{{<citation>}}

Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova. (2024)  
**Authorship Obfuscation in Multilingual Machine-Generated Text Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2401.07867v1)  

---


**ABSTRACT**  
High-quality text generation capability of latest Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause detection evasion in all tested languages, where homoglyph attacks are especially successful.

{{</citation>}}


### (53/99) Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding (Heming Xia et al., 2024)

{{<citation>}}

Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, Zhifang Sui. (2024)  
**Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07851v1)  

---


**ABSTRACT**  
To mitigate the high inference latency stemming from autoregressive decoding in Large Language Models (LLMs), Speculative Decoding has emerged as a novel decoding paradigm for LLM inference. In each decoding step, this method first efficiently drafts several future tokens and then verifies them in parallel. Unlike autoregressive decoding, Speculative Decoding facilitates the simultaneous decoding of multiple tokens per step, thereby accelerating inference. This paper presents a comprehensive overview and analysis of this promising decoding paradigm. We begin by providing a formal definition and formulation of Speculative Decoding. Then, we organize in-depth discussions on its key facets, including current leading techniques, the challenges faced, and potential future directions in this field. We aim for this work to serve as a catalyst for further research on Speculative Decoding, ultimately contributing to more efficient LLM inference.

{{</citation>}}


### (54/99) Milestones in Bengali Sentiment Analysis leveraging Transformer-models: Fundamentals, Challenges and Future Directions (Saptarshi Sengupta et al., 2024)

{{<citation>}}

Saptarshi Sengupta, Shreya Ghosh, Prasenjit Mitra, Tarikul Islam Tamiti. (2024)  
**Milestones in Bengali Sentiment Analysis leveraging Transformer-models: Fundamentals, Challenges and Future Directions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Sentiment Analysis, Transformer  
[Paper Link](http://arxiv.org/abs/2401.07847v1)  

---


**ABSTRACT**  
Sentiment Analysis (SA) refers to the task of associating a view polarity (usually, positive, negative, or neutral; or even fine-grained such as slightly angry, sad, etc.) to a given text, essentially breaking it down to a supervised (since we have the view labels apriori) classification task. Although heavily studied in resource-rich languages such as English thus pushing the SOTA by leaps and bounds, owing to the arrival of the Transformer architecture, the same cannot be said for resource-poor languages such as Bengali (BN). For a language spoken by roughly 300 million people, the technology enabling them to run trials on their favored tongue is severely lacking. In this paper, we analyze the SOTA for SA in Bengali, particularly, Transformer-based models. We discuss available datasets, their drawbacks, the nuances associated with Bengali i.e. what makes this a challenging language to apply SA on, and finally provide insights for future direction to mitigate the limitations in the field.

{{</citation>}}


### (55/99) Question Translation Training for Better Multilingual Reasoning (Wenhao Zhu et al., 2024)

{{<citation>}}

Wenhao Zhu, Shujian Huang, Fei Yuan, Shuaijie She, Jiajun Chen, Alexandra Birch. (2024)  
**Question Translation Training for Better Multilingual Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Multilingual, NLP, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.07817v1)  

---


**ABSTRACT**  
Large language models show compelling performance on reasoning tasks but they tend to perform much worse in languages other than English. This is unsurprising given that their training data largely consists of English text and instructions. A typical solution is to translate instruction data into all languages of interest, and then train on the resulting multilingual data, which is called translate-training. This approach not only incurs high cost, but also results in poorly translated data due to the non-standard formatting of chain-of-thought and mathematical reasoning instructions. In this paper, we explore the benefits of question alignment, where we train the model to translate reasoning questions into English by finetuning on X-English question data. In this way we perform targetted, in-domain language alignment which makes best use of English instruction data to unlock the LLMs' multilingual reasoning abilities. Experimental results on LLaMA2-13B show that question alignment leads to consistent improvements over the translate-training approach: an average improvement of 11.3\% and 16.1\% accuracy across ten languages on the MGSM and MSVAMP maths reasoning benchmarks (The project will be available at: https://github.com/NJUNLP/QAlign).

{{</citation>}}


### (56/99) Consolidating Strategies for Countering Hate Speech Using Persuasive Dialogues (Sougata Saha et al., 2024)

{{<citation>}}

Sougata Saha, Rohini Srihari. (2024)  
**Consolidating Strategies for Countering Hate Speech Using Persuasive Dialogues**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2401.07810v1)  

---


**ABSTRACT**  
Hateful comments are prevalent on social media platforms. Although tools for automatically detecting, flagging, and blocking such false, offensive, and harmful content online have lately matured, such reactive and brute force methods alone provide short-term and superficial remedies while the perpetrators persist. With the public availability of large language models which can generate articulate synthetic and engaging content at scale, there are concerns about the rapid growth of dissemination of such malicious content on the web. There is now a need to focus on deeper, long-term solutions that involve engaging with the human perpetrator behind the source of the content to change their viewpoint or at least bring down the rhetoric using persuasive means. To do that, we propose defining and experimenting with controllable strategies for generating counter-arguments to hateful comments in online conversations. We experiment with controlling response generation using features based on (i) argument structure and reasoning-based Walton argument schemes, (ii) counter-argument speech acts, and (iii) human characteristics-based qualities such as Big-5 personality traits and human values. Using automatic and human evaluations, we determine the best combination of features that generate fluent, argumentative, and logically sound arguments for countering hate. We further share the developed computational models for automatically annotating text with such features, and a silver-standard annotated version of an existing hate speech dialog corpora.

{{</citation>}}


### (57/99) Flexibly Scaling Large Language Models Contexts Through Extensible Tokenization (Ninglu Shao et al., 2024)

{{<citation>}}

Ninglu Shao, Shitao Xiao, Zheng Liu, Peitian Zhang. (2024)  
**Flexibly Scaling Large Language Models Contexts Through Extensible Tokenization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07793v1)  

---


**ABSTRACT**  
Large language models (LLMs) are in need of sufficient contexts to handle many critical applications, such as retrieval augmented generation and few-shot learning. However, due to the constrained window size, the LLMs can only access to the information within a limited context. Although the size of context window can be extended by fine-tuning, it will result in a substantial cost in both training and inference stage. In this paper, we present Extensible Tokenization as an alternative method which realizes the flexible scaling of LLMs' context. Extensible Tokenization stands as a midware in between of the tokenized context and the LLM, which transforms the raw token embeddings into the extensible embeddings. Such embeddings provide a more compact representation for the long context, on top of which the LLM is able to perceive more information with the same context window. Extensible Tokenization is also featured by its flexibility: the scaling factor can be flexibly determined within a feasible scope, leading to the extension of an arbitrary context length at the inference time. Besides, Extensible Tokenization is introduced as a drop-in component, which can be seamlessly plugged into not only the LLM itself and but also its fine-tuned derivatives, bringing in the extended contextual information while fully preserving the LLM's existing capabilities. We perform comprehensive experiments on long-context language modeling and understanding tasks, which verify Extensible Tokenization as an effective, efficient, flexible, and compatible method to extend LLM's context. Our model and source code will be made publicly available.

{{</citation>}}


### (58/99) Quantum Transfer Learning for Acceptability Judgements (Giuseppe Buonaiuto et al., 2024)

{{<citation>}}

Giuseppe Buonaiuto, Raffaele Guarasci, Aniello Minutolo, Giuseppe De Pietro, Massimo Esposito. (2024)  
**Quantum Transfer Learning for Acceptability Judgements**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, physics-comp-ph, quant-ph  
Keywords: AI, NLP  
[Paper Link](http://arxiv.org/abs/2401.07777v1)  

---


**ABSTRACT**  
Hybrid quantum-classical classifiers promise to positively impact critical aspects of natural language processing tasks, particularly classification-related ones. Among the possibilities currently investigated, quantum transfer learning, i.e., using a quantum circuit for fine-tuning pre-trained classical models for a specific task, is attracting significant attention as a potential platform for proving quantum advantage.   This work shows potential advantages, both in terms of performance and expressiveness, of quantum transfer learning algorithms trained on embedding vectors extracted from a large language model to perform classification on a classical Linguistics task: acceptability judgments. Acceptability judgment is the ability to determine whether a sentence is considered natural and well-formed by a native speaker. The approach has been tested on sentences extracted from ItaCoLa, a corpus that collects Italian sentences labeled with their acceptability judgment. The evaluation phase shows results for the quantum transfer learning pipeline comparable to state-of-the-art classical transfer learning algorithms, proving current quantum computers' capabilities to tackle NLP tasks for ready-to-use applications. Furthermore, a qualitative linguistic analysis, aided by explainable AI methods, reveals the capabilities of quantum transfer learning algorithms to correctly classify complex and more structured sentences, compared to their classical counterpart. This finding sets the ground for a quantifiable quantum advantage in NLP in the near future.

{{</citation>}}


### (59/99) On the importance of Data Scale in Pretraining Arabic Language Models (Abbas Ghaddar et al., 2024)

{{<citation>}}

Abbas Ghaddar, Philippe Langlais, Mehdi Rezagholizadeh, Boxing Chen. (2024)  
**On the importance of Data Scale in Pretraining Arabic Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model, NLP, Natural Language Processing, Pretrained Language Models, T5  
[Paper Link](http://arxiv.org/abs/2401.07760v1)  

---


**ABSTRACT**  
Pretraining monolingual language models have been proven to be vital for performance in Arabic Natural Language Processing (NLP) tasks. In this paper, we conduct a comprehensive study on the role of data in Arabic Pretrained Language Models (PLMs). More precisely, we reassess the performance of a suite of state-of-the-art Arabic PLMs by retraining them on massive-scale, high-quality Arabic corpora. We have significantly improved the performance of the leading Arabic encoder-only BERT-base and encoder-decoder T5-base models on the ALUE and ORCA leaderboards, thereby reporting state-of-the-art results in their respective model categories. In addition, our analysis strongly suggests that pretraining data by far is the primary contributor to performance, surpassing other factors. Our models and source code are publicly available at https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/JABER-PyTorch.

{{</citation>}}


### (60/99) Prompting open-source and commercial language models for grammatical error correction of English learner text (Christopher Davis et al., 2024)

{{<citation>}}

Christopher Davis, Andrew Caines, Øistein Andersen, Shiva Taslimipoor, Helen Yannakoudakis, Zheng Yuan, Christopher Bryant, Marek Rei, Paula Buttery. (2024)  
**Prompting open-source and commercial language models for grammatical error correction of English learner text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2401.07702v1)  

---


**ABSTRACT**  
Thanks to recent advances in generative AI, we are able to prompt large language models (LLMs) to produce texts which are fluent and grammatical. In addition, it has been shown that we can elicit attempts at grammatical error correction (GEC) from LLMs when prompted with ungrammatical input sentences. We evaluate how well LLMs can perform at GEC by measuring their performance on established benchmark datasets. We go beyond previous studies, which only examined GPT* models on a selection of English GEC datasets, by evaluating seven open-source and three commercial LLMs on four established GEC benchmarks. We investigate model performance and report results against individual error types. Our results indicate that LLMs do not always outperform supervised English GEC models except in specific contexts -- namely commercial LLMs on benchmarks annotated with fluency corrections as opposed to minimal edits. We find that several open-source models outperform commercial ones on minimal edit benchmarks, and that in some settings zero-shot prompting is just as competitive as few-shot prompting.

{{</citation>}}


### (61/99) Assisted Knowledge Graph Authoring: Human-Supervised Knowledge Graph Construction from Natural Language (Marcel Gohsen et al., 2024)

{{<citation>}}

Marcel Gohsen, Benno Stein. (2024)  
**Assisted Knowledge Graph Authoring: Human-Supervised Knowledge Graph Construction from Natural Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.07683v1)  

---


**ABSTRACT**  
Encyclopedic knowledge graphs, such as Wikidata, host an extensive repository of millions of knowledge statements. However, domain-specific knowledge from fields such as history, physics, or medicine is significantly underrepresented in those graphs. Although few domain-specific knowledge graphs exist (e.g., Pubmed for medicine), developing specialized retrieval applications for many domains still requires constructing knowledge graphs from scratch. To facilitate knowledge graph construction, we introduce WAKA: a Web application that allows domain experts to create knowledge graphs through the medium with which they are most familiar: natural language.

{{</citation>}}


### (62/99) MAPLE: Multilingual Evaluation of Parameter Efficient Finetuning of Large Language Models (Divyanshu Aggarwal et al., 2024)

{{<citation>}}

Divyanshu Aggarwal, Ashutosh Sathe, Sunayana Sitaram. (2024)  
**MAPLE: Multilingual Evaluation of Parameter Efficient Finetuning of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2401.07598v1)  

---


**ABSTRACT**  
Parameter efficient finetuning has emerged as a viable solution for improving the performance of Large Language Models without requiring massive resources and compute. Prior work on multilingual evaluation has shown that there is a large gap between the performance of LLMs on English and other languages. Further, there is also a large gap between the performance of smaller open-source models and larger LLMs. Finetuning can be an effective way to bridge this gap and make language models more equitable. In this work, we finetune the LLaMA-7B and Mistral-7B models on synthetic multilingual instruction tuning data to determine its effect on model performance on five downstream tasks covering twenty three languages in all. Additionally, we experiment with various parameters, such as rank for low-rank adaptation and values of quantisation to determine their effects on downstream performance and find that higher rank and higher quantisation values benefit low-resource languages. We find that parameter efficient finetuning of smaller open source models sometimes bridges the gap between the performance of these models and the larger ones, however, English performance can take a hit. We also find that finetuning sometimes improves performance on low-resource languages, while degrading performance on high-resource languages.

{{</citation>}}


### (63/99) Cascaded Cross-Modal Transformer for Audio-Textual Classification (Nicolae-Catalin Ristea et al., 2024)

{{<citation>}}

Nicolae-Catalin Ristea, Andrei Anghel, Radu Tudor Ionescu. (2024)  
**Cascaded Cross-Modal Transformer for Audio-Textual Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.07575v1)  

---


**ABSTRACT**  
Speech classification tasks often require powerful language understanding models to grasp useful features, which becomes problematic when limited training data is available. To attain superior classification performance, we propose to harness the inherent value of multimodal representations by transcribing speech using automatic speech recognition (ASR) models and translating the transcripts into different languages via pretrained translation models. We thus obtain an audio-textual (multimodal) representation for each data sample. Subsequently, we combine language-specific Bidirectional Encoder Representations from Transformers (BERT) with Wav2Vec2.0 audio features via a novel cascaded cross-modal transformer (CCMT). Our model is based on two cascaded transformer blocks. The first one combines text-specific features from distinct languages, while the second one combines acoustic features with multilingual features previously learned by the first transformer block. We employed our system in the Requests Sub-Challenge of the ACM Multimedia 2023 Computational Paralinguistics Challenge. CCMT was declared the winning solution, obtaining an unweighted average recall (UAR) of 65.41% and 85.87% for complaint and request detection, respectively. Moreover, we applied our framework on the Speech Commands v2 and HarperValleyBank dialog data sets, surpassing previous studies reporting results on these benchmarks. Our code is freely available for download at: https://github.com/ristea/ccmt.

{{</citation>}}


### (64/99) Editing Arbitrary Propositions in LLMs without Subject Labels (Itai Feigenbaum et al., 2024)

{{<citation>}}

Itai Feigenbaum, Devansh Arpit, Huan Wang, Shelby Heinecke, Juan Carlos Niebles, Weiran Yao, Caiming Xiong, Silvio Savarese. (2024)  
**Editing Arbitrary Propositions in LLMs without Subject Labels**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07526v1)  

---


**ABSTRACT**  
Large Language Model (LLM) editing modifies factual information in LLMs. Locate-and-Edit (L\&E) methods accomplish this by finding where relevant information is stored within the neural network, and editing the weights at that location. The goal of editing is to modify the response of an LLM to a proposition independently of its phrasing, while not modifying its response to other related propositions. Existing methods are limited to binary propositions, which represent straightforward binary relations between a subject and an object. Furthermore, existing methods rely on semantic subject labels, which may not be available or even be well-defined in practice. In this paper, we show that both of these issues can be effectively skirted with a simple and fast localization method called Gradient Tracing (GT). This localization method allows editing arbitrary propositions instead of just binary ones, and does so without the need for subject labels. As propositions always have a truth value, our experiments prompt an LLM as a boolean classifier, and edit its T/F response to propositions. Our method applies GT for location tracing, and then edit the model at that location using a mild variant of Rank-One Model Editing (ROME). On datasets of binary propositions derived from the CounterFact dataset, we show that our method -- without access to subject labels -- performs close to state-of-the-art L\&E methods which has access subject labels. We then introduce a new dataset, Factual Accuracy Classification Test (FACT), which includes non-binary propositions and for which subject labels are not generally applicable, and therefore is beyond the scope of existing L\&E methods. Nevertheless, we show that with our method editing is possible on FACT.

{{</citation>}}


### (65/99) Survey of Natural Language Processing for Education: Taxonomy, Systematic Review, and Future Trends (Yunshi Lan et al., 2024)

{{<citation>}}

Yunshi Lan, Xinyuan Li, Hanyue Du, Xuesong Lu, Ming Gao, Weining Qian, Aoying Zhou. (2024)  
**Survey of Natural Language Processing for Education: Taxonomy, Systematic Review, and Future Trends**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.07518v1)  

---


**ABSTRACT**  
Natural Language Processing (NLP) aims to analyze the text via techniques in the computer science field. It serves the applications in healthcare, commerce, and education domains. Particularly, NLP has been applied to the education domain to help teaching and learning. In this survey, we review recent advances in NLP with a focus on solving problems related to the education domain. In detail, we begin with introducing the relevant background. Then, we present the taxonomy of NLP in the education domain. Next, we illustrate the task definition, challenges, and corresponding techniques based on the above taxonomy. After that, we showcase some off-the-shelf demonstrations in this domain and conclude with future directions.

{{</citation>}}


### (66/99) Developing ChatGPT for Biology and Medicine: A Complete Review of Biomedical Question Answering (Qing Li et al., 2024)

{{<citation>}}

Qing Li, Lei Li, Yu Li. (2024)  
**Developing ChatGPT for Biology and Medicine: A Complete Review of Biomedical Question Answering**  

---
Primary Category: cs.CL  
Categories: I-2-1, cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, NLP, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.07510v1)  

---


**ABSTRACT**  
ChatGPT explores a strategic blueprint of question answering (QA) in delivering medical diagnosis, treatment recommendations, and other healthcare support. This is achieved through the increasing incorporation of medical domain data via natural language processing (NLP) and multimodal paradigms. By transitioning the distribution of text, images, videos, and other modalities from the general domain to the medical domain, these techniques have expedited the progress of medical domain question answering (MDQA). They bridge the gap between human natural language and sophisticated medical domain knowledge or expert manual annotations, handling large-scale, diverse, unbalanced, or even unlabeled data analysis scenarios in medical contexts. Central to our focus is the utilizing of language models and multimodal paradigms for medical question answering, aiming to guide the research community in selecting appropriate mechanisms for their specific medical research requirements. Specialized tasks such as unimodal-related question answering, reading comprehension, reasoning, diagnosis, relation extraction, probability modeling, and others, as well as multimodal-related tasks like vision question answering, image caption, cross-modal retrieval, report summarization, and generation, are discussed in detail. Each section delves into the intricate specifics of the respective method under consideration. This paper highlights the structures and advancements of medical domain explorations against general domain methods, emphasizing their applications across different tasks and datasets. It also outlines current challenges and opportunities for future medical domain research, paving the way for continued innovation and application in this rapidly evolving field.

{{</citation>}}


### (67/99) GWPT: A Green Word-Embedding-based POS Tagger (Chengwei Wei et al., 2024)

{{<citation>}}

Chengwei Wei, Runqi Pang, C. -C. Jay Kuo. (2024)  
**GWPT: A Green Word-Embedding-based POS Tagger**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, NLP  
[Paper Link](http://arxiv.org/abs/2401.07475v1)  

---


**ABSTRACT**  
As a fundamental tool for natural language processing (NLP), the part-of-speech (POS) tagger assigns the POS label to each word in a sentence. A novel lightweight POS tagger based on word embeddings is proposed and named GWPT (green word-embedding-based POS tagger) in this work. Following the green learning (GL) methodology, GWPT contains three modules in cascade: 1) representation learning, 2) feature learning, and 3) decision learning modules. The main novelty of GWPT lies in representation learning. It uses non-contextual or contextual word embeddings, partitions embedding dimension indices into low-, medium-, and high-frequency sets, and represents them with different N-grams. It is shown by experimental results that GWPT offers state-of-the-art accuracies with fewer model parameters and significantly lower computational complexity in both training and inference as compared with deep-learning-based methods.

{{</citation>}}


### (68/99) Utilizing deep learning models for the identification of enhancers and super-enhancers based on genomic and epigenomic features (Zahra Ahani et al., 2024)

{{<citation>}}

Zahra Ahani, Moein Shahiki Tash, Yoel Ledo Mezquita, Jason Angel. (2024)  
**Utilizing deep learning models for the identification of enhancers and super-enhancers based on genomic and epigenomic features**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.07470v1)  

---


**ABSTRACT**  
This paper provides an extensive examination of a sizable dataset of English tweets focusing on nine widely recognized cryptocurrencies, specifically Cardano, Binance, Bitcoin, Dogecoin, Ethereum, Fantom, Matic, Shiba, and Ripple. Our primary objective was to conduct a psycholinguistic and emotion analysis of social media content associated with these cryptocurrencies. To enable investigators to make more informed decisions. The study involved comparing linguistic characteristics across the diverse digital coins, shedding light on the distinctive linguistic patterns that emerge within each coin's community. To achieve this, we utilized advanced text analysis techniques. Additionally, our work unveiled an intriguing Understanding of the interplay between these digital assets within the cryptocurrency community. By examining which coin pairs are mentioned together most frequently in the dataset, we established correlations between different cryptocurrencies. To ensure the reliability of our findings, we initially gathered a total of 832,559 tweets from Twitter. These tweets underwent a rigorous preprocessing stage, resulting in a refined dataset of 115,899 tweets that were used for our analysis. Overall, our research offers valuable Perception into the linguistic nuances of various digital coins' online communities and provides a deeper understanding of their interactions in the cryptocurrency space.

{{</citation>}}


### (69/99) Only Send What You Need: Learning to Communicate Efficiently in Federated Multilingual Machine Translation (Yun-Wei Chu et al., 2024)

{{<citation>}}

Yun-Wei Chu, Dong-Jun Han, Christopher G. Brinton. (2024)  
**Only Send What You Need: Learning to Communicate Efficiently in Federated Multilingual Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Machine Translation, Multilingual  
[Paper Link](http://arxiv.org/abs/2401.07456v1)  

---


**ABSTRACT**  
Federated learning (FL) is a promising approach for solving multilingual tasks, potentially enabling clients with their own language-specific data to collaboratively construct a high-quality neural machine translation (NMT) model. However, communication constraints in practical network systems present challenges for exchanging large-scale NMT engines between FL parties. In this paper, we propose a meta-learning-based adaptive parameter selection methodology, MetaSend, that improves the communication efficiency of model transmissions from clients during FL-based multilingual NMT training. Our approach learns a dynamic threshold for filtering parameters prior to transmission without compromising the NMT model quality, based on the tensor deviations of clients between different FL rounds. Through experiments on two NMT datasets with different language distributions, we demonstrate that MetaSend obtains substantial improvements over baselines in translation quality in the presence of a limited communication budget.

{{</citation>}}


### (70/99) Stability Analysis of ChatGPT-based Sentiment Analysis in AI Quality Assurance (Tinghui Ouyang et al., 2024)

{{<citation>}}

Tinghui Ouyang, AprilPyone MaungMaung, Koichi Konishi, Yoshiki Seo, Isao Echizen. (2024)  
**Stability Analysis of ChatGPT-based Sentiment Analysis in AI Quality Assurance**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2401.07441v1)  

---


**ABSTRACT**  
In the era of large AI models, the complex architecture and vast parameters present substantial challenges for effective AI quality management (AIQM), e.g. large language model (LLM). This paper focuses on investigating the quality assurance of a specific LLM-based AI product--a ChatGPT-based sentiment analysis system. The study delves into stability issues related to both the operation and robustness of the expansive AI model on which ChatGPT is based. Experimental analysis is conducted using benchmark datasets for sentiment analysis. The results reveal that the constructed ChatGPT-based sentiment analysis system exhibits uncertainty, which is attributed to various operational factors. It demonstrated that the system also exhibits stability issues in handling conventional small text attacks involving robustness.

{{</citation>}}


### (71/99) Leveraging the power of transformers for guilt detection in text (Abdul Gafar Manuel Meque et al., 2024)

{{<citation>}}

Abdul Gafar Manuel Meque, Jason Angel, Grigori Sidorov, Alexander Gelbukh. (2024)  
**Leveraging the power of transformers for guilt detection in text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.07414v1)  

---


**ABSTRACT**  
In recent years, language models and deep learning techniques have revolutionized natural language processing tasks, including emotion detection. However, the specific emotion of guilt has received limited attention in this field. In this research, we explore the applicability of three transformer-based language models for detecting guilt in text and compare their performance for general emotion detection and guilt detection. Our proposed model outformed BERT and RoBERTa models by two and one points respectively. Additionally, we analyze the challenges in developing accurate guilt-detection models and evaluate our model's effectiveness in detecting related emotions like "shame" through qualitative analysis of results.

{{</citation>}}


## cs.AI (7)



### (72/99) AI-as-exploration: Navigating intelligence space (Dimitri Coelho Mollo, 2024)

{{<citation>}}

Dimitri Coelho Mollo. (2024)  
**AI-as-exploration: Navigating intelligence space**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.07964v1)  

---


**ABSTRACT**  
Artificial Intelligence is a field that lives many lives, and the term has come to encompass a motley collection of scientific and commercial endeavours. In this paper, I articulate the contours of a rather neglected but central scientific role that AI has to play, which I dub `AI-as-exploration'.The basic thrust of AI-as-exploration is that of creating and studying systems that can reveal candidate building blocks of intelligence that may differ from the forms of human and animal intelligence we are familiar with. In other words, I suggest that AI is one of the best tools we have for exploring intelligence space, namely the space of possible intelligent systems. I illustrate the value of AI-as-exploration by focusing on a specific case study, i.e., recent work on the capacity to combine novel and invented concepts in humans and Large Language Models. I show that the latter, despite showing human-level accuracy in such a task, most probably solve it in ways radically different, but no less relevant to intelligence research, to those hypothesised for humans.

{{</citation>}}


### (73/99) A Strategy for Implementing description Temporal Dynamic Algorithms in Dynamic Knowledge Graphs by SPIN (Alireza Shahbazi et al., 2024)

{{<citation>}}

Alireza Shahbazi, Seyyed Ahmad Mirsanei, Malikeh Haj Khan Mirzaye Sarraf, Behrouz Minaei Bidgoli. (2024)  
**A Strategy for Implementing description Temporal Dynamic Algorithms in Dynamic Knowledge Graphs by SPIN**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LO, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.07890v1)  

---


**ABSTRACT**  
Planning and reasoning about actions and processes, in addition to reasoning about propositions, are important issues in recent logical and computer science studies. The widespread use of actions in everyday life such as IoT, semantic web services, etc., and the limitations and issues in the action formalisms are two factors that lead us to study about how actions are represented.   Since 2007, there was some ideas to integrate Description Logic (DL) and action formalisms for representing both static and dynamic knowledge. In meanwhile, time is an important factor in dynamic situations, and actions change states over time. In this study, on the one hand, we examined related logical structures such as extensions of description logics (DLs), temporal formalisms, and action formalisms. On the other hand, we analyzed possible tools for designing and developing the Knowledge and Action Base (KAB).   For representation and reasoning about actions, we embedded actions into DLs (such as Dynamic-ALC and its extensions). We propose a terminable algorithm for action projection, planning, checking the satisfiability, consistency, realizability, and executability, and also querying from KAB. Actions in this framework were modeled with SPIN and added to state space. This framework has also been implemented as a plugin for the Prot\'eg\'e ontology editor.   During the last two decades, various algorithms have been presented, but due to the high computational complexity, we face many problems in implementing dynamic ontologies. In addition, an algorithm to detect the inconsistency of actions' effects was not explicitly stated. In the proposed strategy, the interactions of actions with other parts of modeled knowledge, and a method to check consistency between the effects of actions are presented. With this framework, the ramification problem can be well handled in future works.

{{</citation>}}


### (74/99) Explainable Predictive Maintenance: A Survey of Current Methods, Challenges and Opportunities (Logan Cummins et al., 2024)

{{<citation>}}

Logan Cummins, Alex Sommers, Somayeh Bakhtiari Ramezani, Sudip Mittal, Joseph Jabour, Maria Seale, Shahram Rahimi. (2024)  
**Explainable Predictive Maintenance: A Survey of Current Methods, Challenges and Opportunities**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07871v1)  

---


**ABSTRACT**  
Predictive maintenance is a well studied collection of techniques that aims to prolong the life of a mechanical system by using artificial intelligence and machine learning to predict the optimal time to perform maintenance. The methods allow maintainers of systems and hardware to reduce financial and time costs of upkeep. As these methods are adopted for more serious and potentially life-threatening applications, the human operators need trust the predictive system. This attracts the field of Explainable AI (XAI) to introduce explainability and interpretability into the predictive system. XAI brings methods to the field of predictive maintenance that can amplify trust in the users while maintaining well-performing systems. This survey on explainable predictive maintenance (XPM) discusses and presents the current methods of XAI as applied to predictive maintenance while following the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) 2020 guidelines. We categorize the different XPM methods into groups that follow the XAI literature. Additionally, we include current challenges and a discussion on future research directions in XPM.

{{</citation>}}


### (75/99) When Large Language Model Agents Meet 6G Networks: Perception, Grounding, and Alignment (Minrui Xu et al., 2024)

{{<citation>}}

Minrui Xu, Niyato Dusit, Jiawen Kang, Zehui Xiong, Shiwen Mao, Zhu Han, Dong In Kim, Khaled B. Letaief. (2024)  
**When Large Language Model Agents Meet 6G Networks: Perception, Grounding, and Alignment**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-NI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.07764v1)  

---


**ABSTRACT**  
AI agents based on multimodal large language models (LLMs) are expected to revolutionize human-computer interaction and offer more personalized assistant services across various domains like healthcare, education, manufacturing, and entertainment. Deploying LLM agents in 6G networks enables users to access previously expensive AI assistant services via mobile devices democratically, thereby reducing interaction latency and better preserving user privacy. Nevertheless, the limited capacity of mobile devices constrains the effectiveness of deploying and executing local LLMs, which necessitates offloading complex tasks to global LLMs running on edge servers during long-horizon interactions. In this article, we propose a split learning system for LLM agents in 6G networks leveraging the collaboration between mobile devices and edge servers, where multiple LLMs with different roles are distributed across mobile devices and edge servers to perform user-agent interactive tasks collaboratively. In the proposed system, LLM agents are split into perception, grounding, and alignment modules, facilitating inter-module communications to meet extended user requirements on 6G network functions, including integrated sensing and communication, digital twins, and task-oriented communications. Furthermore, we introduce a novel model caching algorithm for LLMs within the proposed system to improve model utilization in context, thus reducing network costs of the collaborative mobile and edge LLM agents.

{{</citation>}}


### (76/99) Combining Machine Learning and Ontology: A Systematic Literature Review (Sarah Ghidalia et al., 2024)

{{<citation>}}

Sarah Ghidalia, Ouassila Labbani Narsis, Aurélie Bertaux, Christophe Nicolle. (2024)  
**Combining Machine Learning and Ontology: A Systematic Literature Review**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07744v1)  

---


**ABSTRACT**  
Motivated by the desire to explore the process of combining inductive and deductive reasoning, we conducted a systematic literature review of articles that investigate the integration of machine learning and ontologies. The objective was to identify diverse techniques that incorporate both inductive reasoning (performed by machine learning) and deductive reasoning (performed by ontologies) into artificial intelligence systems. Our review, which included the analysis of 128 studies, allowed us to identify three main categories of hybridization between machine learning and ontologies: learning-enhanced ontologies, semantic data mining, and learning and reasoning systems. We provide a comprehensive examination of all these categories, emphasizing the various machine learning algorithms utilized in the studies. Furthermore, we compared our classification with similar recent work in the field of hybrid AI and neuro-symbolic approaches.

{{</citation>}}


### (77/99) Formal Logic Enabled Personalized Federated Learning Through Property Inference (Ziyan An et al., 2024)

{{<citation>}}

Ziyan An, Taylor T. Johnson, Meiyi Ma. (2024)  
**Formal Logic Enabled Personalized Federated Learning Through Property Inference**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07448v1)  

---


**ABSTRACT**  
Recent advancements in federated learning (FL) have greatly facilitated the development of decentralized collaborative applications, particularly in the domain of Artificial Intelligence of Things (AIoT). However, a critical aspect missing from the current research landscape is the ability to enable data-driven client models with symbolic reasoning capabilities. Specifically, the inherent heterogeneity of participating client devices poses a significant challenge, as each client exhibits unique logic reasoning properties. Failing to consider these device-specific specifications can result in critical properties being missed in the client predictions, leading to suboptimal performance. In this work, we propose a new training paradigm that leverages temporal logic reasoning to address this issue. Our approach involves enhancing the training process by incorporating mechanically generated logic expressions for each FL client. Additionally, we introduce the concept of aggregation clusters and develop a partitioning algorithm to effectively group clients based on the alignment of their temporal reasoning properties. We evaluate the proposed method on two tasks: a real-world traffic volume prediction task consisting of sensory data from fifteen states and a smart city multi-task prediction utilizing synthetic data. The evaluation results exhibit clear improvements, with performance accuracy improved by up to 54% across all sequential prediction models.

{{</citation>}}


### (78/99) Generalized Planning for the Abstraction and Reasoning Corpus (Chao Lei et al., 2024)

{{<citation>}}

Chao Lei, Nir Lipovetzky, Krista A. Ehinger. (2024)  
**Generalized Planning for the Abstraction and Reasoning Corpus**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.07426v1)  

---


**ABSTRACT**  
The Abstraction and Reasoning Corpus (ARC) is a general artificial intelligence benchmark that poses difficulties for pure machine learning methods due to its requirement for fluid intelligence with a focus on reasoning and abstraction. In this work, we introduce an ARC solver, Generalized Planning for Abstract Reasoning (GPAR). It casts an ARC problem as a generalized planning (GP) problem, where a solution is formalized as a planning program with pointers. We express each ARC problem using the standard Planning Domain Definition Language (PDDL) coupled with external functions representing object-centric abstractions. We show how to scale up GP solvers via domain knowledge specific to ARC in the form of restrictions over the actions model, predicates, arguments and valid structure of planning programs. Our experiments demonstrate that GPAR outperforms the state-of-the-art solvers on the object-centric tasks of the ARC, showing the effectiveness of GP and the expressiveness of PDDL to model ARC problems. The challenges provided by the ARC benchmark motivate research to advance existing GP solvers and understand new relations with other planning computational models. Code is available at github.com/you68681/GPAR.

{{</citation>}}


## cs.CR (2)



### (79/99) ADMIn: Attacks on Dataset, Model and Input. A Threat Model for AI Based Software (Vimal Kumar et al., 2024)

{{<citation>}}

Vimal Kumar, Juliette Mayo, Khadija Bahiss. (2024)  
**ADMIn: Attacks on Dataset, Model and Input. A Threat Model for AI Based Software**  

---
Primary Category: cs.CR  
Categories: K-6-5, cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07960v1)  

---


**ABSTRACT**  
Machine learning (ML) and artificial intelligence (AI) techniques have now become commonplace in software products and services. When threat modelling a system, it is therefore important that we consider threats unique to ML and AI techniques, in addition to threats to our software. In this paper, we present a threat model that can be used to systematically uncover threats to AI based software. The threat model consists of two main parts, a model of the software development process for AI based software and an attack taxonomy that has been developed using attacks found in adversarial AI research. We apply the threat model to two real life AI based software and discuss the process and the threats found.

{{</citation>}}


### (80/99) Signed-Prompt: A New Approach to Prevent Prompt Injection Attacks Against LLM-Integrated Applications (Xuchen Suo, 2024)

{{<citation>}}

Xuchen Suo. (2024)  
**Signed-Prompt: A New Approach to Prevent Prompt Injection Attacks Against LLM-Integrated Applications**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.07612v1)  

---


**ABSTRACT**  
The critical challenge of prompt injection attacks in Large Language Models (LLMs) integrated applications, a growing concern in the Artificial Intelligence (AI) field. Such attacks, which manipulate LLMs through natural language inputs, pose a significant threat to the security of these applications. Traditional defense strategies, including output and input filtering, as well as delimiter use, have proven inadequate. This paper introduces the 'Signed-Prompt' method as a novel solution. The study involves signing sensitive instructions within command segments by authorized users, enabling the LLM to discern trusted instruction sources. The paper presents a comprehensive analysis of prompt injection attack patterns, followed by a detailed explanation of the Signed-Prompt concept, including its basic architecture and implementation through both prompt engineering and fine-tuning of LLMs. Experiments demonstrate the effectiveness of the Signed-Prompt method, showing substantial resistance to various types of prompt injection attacks, thus validating its potential as a robust defense strategy in AI security.

{{</citation>}}


## eess.IV (1)



### (81/99) Machine Perceptual Quality: Evaluating the Impact of Severe Lossy Compression on Audio and Image Models (Dan Jacobellis et al., 2024)

{{<citation>}}

Dan Jacobellis, Daniel Cummings, Neeraja J. Yadwadkar. (2024)  
**Machine Perceptual Quality: Evaluating the Impact of Severe Lossy Compression on Audio and Image Models**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, cs-SD, eess-AS, eess-IV, eess.IV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.07957v1)  

---


**ABSTRACT**  
In the field of neural data compression, the prevailing focus has been on optimizing algorithms for either classical distortion metrics, such as PSNR or SSIM, or human perceptual quality. With increasing amounts of data consumed by machines rather than humans, a new paradigm of machine-oriented compression$\unicode{x2013}$which prioritizes the retention of features salient for machine perception over traditional human-centric criteria$\unicode{x2013}$has emerged, creating several new challenges to the development, evaluation, and deployment of systems utilizing lossy compression. In particular, it is unclear how different approaches to lossy compression will affect the performance of downstream machine perception tasks. To address this under-explored area, we evaluate various perception models$\unicode{x2013}$including image classification, image segmentation, speech recognition, and music source separation$\unicode{x2013}$under severe lossy compression. We utilize several popular codecs spanning conventional, neural, and generative compression architectures. Our results indicate three key findings: (1) using generative compression, it is feasible to leverage highly compressed data while incurring a negligible impact on machine perceptual quality; (2) machine perceptual quality correlates strongly with deep similarity metrics, indicating a crucial role of these metrics in the development of machine-oriented codecs; and (3) using lossy compressed datasets, (e.g. ImageNet) for pre-training can lead to counter-intuitive scenarios where lossy compression increases machine perceptual quality rather than degrading it. To encourage engagement on this growing area of research, our code and experiments are available at: https://github.com/danjacobellis/MPQ.

{{</citation>}}


## cs.RO (4)



### (82/99) Survey of Learning Approaches for Robotic In-Hand Manipulation (Abraham Itzhak Weinberg et al., 2024)

{{<citation>}}

Abraham Itzhak Weinberg, Alon Shirizly, Osher Azulay, Avishai Sintov. (2024)  
**Survey of Learning Approaches for Robotic In-Hand Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.07915v1)  

---


**ABSTRACT**  
Human dexterity is an invaluable capability for precise manipulation of objects in complex tasks. The capability of robots to similarly grasp and perform in-hand manipulation of objects is critical for their use in the ever changing human environment, and for their ability to replace manpower. In recent decades, significant effort has been put in order to enable in-hand manipulation capabilities to robotic systems. Initial robotic manipulators followed carefully programmed paths, while later attempts provided a solution based on analytical modeling of motion and contact. However, these have failed to provide practical solutions due to inability to cope with complex environments and uncertainties. Therefore, the effort has shifted to learning-based approaches where data is collected from the real world or through a simulation, during repeated attempts to complete various tasks. The vast majority of learning approaches focused on learning data-based models that describe the system to some extent or Reinforcement Learning (RL). RL, in particular, has seen growing interest due to the remarkable ability to generate solutions to problems with minimal human guidance. In this survey paper, we track the developments of learning approaches for in-hand manipulations and, explore the challenges and opportunities. This survey is designed both as an introduction for novices in the field with a glossary of terms as well as a guide of novel advances for advanced practitioners.

{{</citation>}}


### (83/99) Consolidating Trees of Robotic Plans Generated Using Large Language Models to Improve Reliability (Md Sadman Sakib et al., 2024)

{{<citation>}}

Md Sadman Sakib, Yu Sun. (2024)  
**Consolidating Trees of Robotic Plans Generated Using Large Language Models to Improve Reliability**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CL, cs-RO, cs.RO  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.07868v1)  

---


**ABSTRACT**  
The inherent probabilistic nature of Large Language Models (LLMs) introduces an element of unpredictability, raising concerns about potential discrepancies in their output. This paper introduces an innovative approach aims to generate correct and optimal robotic task plans for diverse real-world demands and scenarios. LLMs have been used to generate task plans, but they are unreliable and may contain wrong, questionable, or high-cost steps. The proposed approach uses LLM to generate a number of task plans as trees and amalgamates them into a graph by removing questionable paths. Then an optimal task tree can be retrieved to circumvent questionable and high-cost nodes, thereby improving planning accuracy and execution efficiency. The approach is further improved by incorporating a large knowledge network. Leveraging GPT-4 further, the high-level task plan is converted into a low-level Planning Domain Definition Language (PDDL) plan executable by a robot. Evaluation results highlight the superior accuracy and efficiency of our approach compared to previous methodologies in the field of task planning.

{{</citation>}}


### (84/99) Online Learning of Piecewise Polynomial Signed Distance Fields for Manipulation Tasks (Ante Marić et al., 2024)

{{<citation>}}

Ante Marić, Yiming Li, Sylvain Calinon. (2024)  
**Online Learning of Piecewise Polynomial Signed Distance Fields for Manipulation Tasks**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.07698v1)  

---


**ABSTRACT**  
Reasoning about distance is indispensable for establishing or avoiding contact in manipulation tasks. To this end, we present an online method for learning implicit representations of signed distance using piecewise polynomial basis functions. Starting from an arbitrary prior shape, our approach incrementally constructs a continuous representation from incoming point cloud data. It offers fast access to distance and analytical gradients without the need to store training data. We assess the accuracy of our model on a diverse set of household objects and compare it to neural network and Gaussian process counterparts. Distance reconstruction and real-time updates are further evaluated in a physical experiment by simultaneously collecting sparse point cloud data and using the evolving model to control a manipulator.

{{</citation>}}


### (85/99) Multi-task robot data for dual-arm fine manipulation (Heecheol Kim et al., 2024)

{{<citation>}}

Heecheol Kim, Yoshiyuki Ohmura, Yasuo Kuniyoshi. (2024)  
**Multi-task robot data for dual-arm fine manipulation**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.07603v1)  

---


**ABSTRACT**  
In the field of robotic manipulation, deep imitation learning is recognized as a promising approach for acquiring manipulation skills. Additionally, learning from diverse robot datasets is considered a viable method to achieve versatility and adaptability. In such research, by learning various tasks, robots achieved generality across multiple objects. However, such multi-task robot datasets have mainly focused on single-arm tasks that are relatively imprecise, not addressing the fine-grained object manipulation that robots are expected to perform in the real world. This paper introduces a dataset of diverse object manipulations that includes dual-arm tasks and/or tasks requiring fine manipulation. To this end, we have generated dataset with 224k episodes (150 hours, 1,104 language instructions) which includes dual-arm fine tasks such as bowl-moving, pencil-case opening or banana-peeling, and this data is publicly available. Additionally, this dataset includes visual attention signals as well as dual-action labels, a signal that separates actions into a robust reaching trajectory and precise interaction with objects, and language instructions to achieve robust and precise object manipulation. We applied the dataset to our Dual-Action and Attention (DAA), a model designed for fine-grained dual arm manipulation tasks and robust against covariate shifts. The model was tested with over 7k total trials in real robot manipulation tasks, demonstrating its capability in fine manipulation.

{{</citation>}}


## cs.HC (1)



### (86/99) Deep Fuzzy Framework for Emotion Recognition using EEG Signals and Emotion Representation in Type-2 Fuzzy VAD Space (Mohammad Asif et al., 2024)

{{<citation>}}

Mohammad Asif, Noman Ali, Sudhakar Mishra, Anushka Dandawate, Uma Shanker Tiwary. (2024)  
**Deep Fuzzy Framework for Emotion Recognition using EEG Signals and Emotion Representation in Type-2 Fuzzy VAD Space**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2401.07892v1)  

---


**ABSTRACT**  
Recently, the representation of emotions in the Valence, Arousal and Dominance (VAD) space has drawn enough attention. However, the complex nature of emotions and the subjective biases in self-reported values of VAD make the emotion model too specific to a particular experiment. This study aims to develop a generic model representing emotions using a fuzzy VAD space and improve emotion recognition by utilizing this representation. We partitioned the crisp VAD space into a fuzzy VAD space using low, medium and high type-2 fuzzy dimensions to represent emotions. A framework that integrates fuzzy VAD space with EEG data has been developed to recognize emotions. The EEG features were extracted using spatial and temporal feature vectors from time-frequency spectrograms, while the subject-reported values of VAD were also considered. The study was conducted on the DENS dataset, which includes a wide range of twenty-four emotions, along with EEG data and subjective ratings. The study was validated using various deep fuzzy framework models based on type-2 fuzzy representation, cuboid probabilistic lattice representation and unsupervised fuzzy emotion clusters. These models resulted in emotion recognition accuracy of 96.09\%, 95.75\% and 95.31\%, respectively, for the classes of 24 emotions. The study also included an ablation study, one with crisp VAD space and the other without VAD space. The result with crisp VAD space performed better, while the deep fuzzy framework outperformed both models. The model was extended to predict cross-subject cases of emotions, and the results with 78.37\% accuracy are promising, proving the generality of our model. The generic nature of the developed model, along with its successful cross-subject predictions, gives direction for real-world applications in the areas such as affective computing, human-computer interaction, and mental health monitoring.

{{</citation>}}


## cs.SD (1)



### (87/99) Decoupled Spatial and Temporal Processing for Resource Efficient Multichannel Speech Enhancement (Ashutosh Pandey et al., 2024)

{{<citation>}}

Ashutosh Pandey, Buye Xu. (2024)  
**Decoupled Spatial and Temporal Processing for Resource Efficient Multichannel Speech Enhancement**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.07879v1)  

---


**ABSTRACT**  
We present a novel model designed for resource-efficient multichannel speech enhancement in the time domain, with a focus on low latency, lightweight, and low computational requirements. The proposed model incorporates explicit spatial and temporal processing within deep neural network (DNN) layers. Inspired by frequency-dependent multichannel filtering, our spatial filtering process applies multiple trainable filters to each hidden unit across the spatial dimension, resulting in a multichannel output. The temporal processing is applied over a single-channel output stream from the spatial processing using a Long Short-Term Memory (LSTM) network. The output from the temporal processing stage is then further integrated into the spatial dimension through elementwise multiplication. This explicit separation of spatial and temporal processing results in a resource-efficient network design. Empirical findings from our experiments show that our proposed model significantly outperforms robust baseline models while demanding far fewer parameters and computations, while achieving an ultra-low algorithmic latency of just 2 milliseconds.

{{</citation>}}


## cs.CY (2)



### (88/99) Two Types of AI Existential Risk: Decisive and Accumulative (Atoosa Kasirzadeh, 2024)

{{<citation>}}

Atoosa Kasirzadeh. (2024)  
**Two Types of AI Existential Risk: Decisive and Accumulative**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-LG, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07836v1)  

---


**ABSTRACT**  
The conventional discourse on existential risks (x-risks) from AI typically focuses on abrupt, dire events caused by advanced AI systems, particularly those that might achieve or surpass human-level intelligence. These events have severe consequences that either lead to human extinction or irreversibly cripple human civilization to a point beyond recovery. This discourse, however, often neglects the serious possibility of AI x-risks manifesting incrementally through a series of smaller yet interconnected disruptions, gradually crossing critical thresholds over time. This paper contrasts the conventional "decisive AI x-risk hypothesis" with an "accumulative AI x-risk hypothesis." While the former envisions an overt AI takeover pathway, characterized by scenarios like uncontrollable superintelligence, the latter suggests a different causal pathway to existential catastrophes. This involves a gradual accumulation of critical AI-induced threats such as severe vulnerabilities and systemic erosion of econopolitical structures. The accumulative hypothesis suggests a boiling frog scenario where incremental AI risks slowly converge, undermining resilience until a triggering event results in irreversible collapse. Through systems analysis, this paper examines the distinct assumptions differentiating these two hypotheses. It is then argued that the accumulative view reconciles seemingly incompatible perspectives on AI risks. The implications of differentiating between these causal pathways -- the decisive and the accumulative -- for the governance of AI risks as well as long-term AI safety are discussed.

{{</citation>}}


### (89/99) How Social Media Big Data Can Improve Suicide Prevention (Anastasia Peshkovskaya et al., 2024)

{{<citation>}}

Anastasia Peshkovskaya, Yu-Tao Xiang. (2024)  
**How Social Media Big Data Can Improve Suicide Prevention**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-HC, cs-SI, cs.CY  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2401.07718v1)  

---


**ABSTRACT**  
In the light of increasing clues on social media impact on self-harm and suicide risks, there is still no evidence on who are and how factually engaged in suicide-related online behaviors. This study reports new findings of high-performance supercomputing investigation of publicly accessible big data sourced from one of the world-largest social networking site. Three-month supercomputer searching resulted in 570,156 young adult users who consumed suicide-related information on social media. Most of them were 21-24 year olds with higher share of females (58%) of predominantly younger age. Every eight user was alarmingly engrossed with up to 15 suicide-related online groups. Evidently, suicide groups on social media are highly underrated public health issue that might weaken the prevention efforts. Suicide prevention strategies that target social media users must be implemented extensively. While major gap in functional understanding of technologies relevance for use in public mental health still exists, current findings act for better understanding digital technologies utility for translational advance and offer relevant evidence-based framework for improving suicide prevention in general population.

{{</citation>}}


## quant-ph (1)



### (90/99) Predominant Aspects on Security for Quantum Machine Learning: Literature Review (Nicola Franco et al., 2024)

{{<citation>}}

Nicola Franco, Alona Sakhnenko, Leon Stolpmann, Daniel Thuerck, Fabian Petsch, Annika Rüll, Jeanette Miriam Lorenz. (2024)  
**Predominant Aspects on Security for Quantum Machine Learning: Literature Review**  

---
Primary Category: quant-ph  
Categories: cs-CR, quant-ph, quant-ph  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.07774v1)  

---


**ABSTRACT**  
Quantum Machine Learning (QML) has emerged as a promising intersection of quantum computing and classical machine learning, anticipated to drive breakthroughs in computational tasks. This paper discusses the question which security concerns and strengths are connected to QML by means of a systematic literature review. We categorize and review the security of QML models, their vulnerabilities inherent to quantum architectures, and the mitigation strategies proposed. The survey reveals that while QML possesses unique strengths, it also introduces novel attack vectors not seen in classical systems. Techniques like adversarial training, quantum noise exploitation, and quantum differential privacy have shown potential in enhancing QML robustness. Our review discuss the need for continued and rigorous research to ensure the secure deployment of QML in real-world applications. This work serves as a foundational reference for researchers and practitioners aiming to navigate the security aspects of QML.

{{</citation>}}


## cs.IR (2)



### (91/99) Deep Evolutional Instant Interest Network for CTR Prediction in Trigger-Induced Recommendation (Zhibo Xiao et al., 2024)

{{<citation>}}

Zhibo Xiao, Luwei Yang, Tao Zhang, Wen Jiang, Wei Ning, Yujiu Yang. (2024)  
**Deep Evolutional Instant Interest Network for CTR Prediction in Trigger-Induced Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2401.07769v1)  

---


**ABSTRACT**  
The recommendation has been playing a key role in many industries, e.g., e-commerce, streaming media, social media, etc. Recently, a new recommendation scenario, called Trigger-Induced Recommendation (TIR), where users are able to explicitly express their instant interests via trigger items, is emerging as an essential role in many e-commerce platforms, e.g., Alibaba.com and Amazon. Without explicitly modeling the user's instant interest, traditional recommendation methods usually obtain sub-optimal results in TIR. Even though there are a few methods considering the trigger and target items simultaneously to solve this problem, they still haven't taken into account temporal information of user behaviors, the dynamic change of user instant interest when the user scrolls down and the interactions between the trigger and target items. To tackle these problems, we propose a novel method -- Deep Evolutional Instant Interest Network (DEI2N), for click-through rate prediction in TIR scenarios. Specifically, we design a User Instant Interest Modeling Layer to predict the dynamic change of the intensity of instant interest when the user scrolls down. Temporal information is utilized in user behavior modeling. Moreover, an Interaction Layer is introduced to learn better interactions between the trigger and target items. We evaluate our method on several offline and real-world industrial datasets. Experimental results show that our proposed DEI2N outperforms state-of-the-art baselines. In addition, online A/B testing demonstrates the superiority over the existing baseline in real-world production environments.

{{</citation>}}


### (92/99) GACE: Learning Graph-Based Cross-Page Ads Embedding For Click-Through Rate Prediction (Haowen Wang et al., 2024)

{{<citation>}}

Haowen Wang, Yuliang Du, Congyun Jin, Yujiao Li, Yingbo Wang, Tao Sun, Piqi Qin, Cong Fan. (2024)  
**GACE: Learning Graph-Based Cross-Page Ads Embedding For Click-Through Rate Prediction**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR, stat-ME  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.07445v1)  

---


**ABSTRACT**  
Predicting click-through rate (CTR) is the core task of many ads online recommendation systems, which helps improve user experience and increase platform revenue. In this type of recommendation system, we often encounter two main problems: the joint usage of multi-page historical advertising data and the cold start of new ads. In this paper, we proposed GACE, a graph-based cross-page ads embedding generation method. It can warm up and generate the representation embedding of cold-start and existing ads across various pages. Specifically, we carefully build linkages and a weighted undirected graph model considering semantic and page-type attributes to guide the direction of feature fusion and generation. We designed a variational auto-encoding task as pre-training module and generated embedding representations for new and old ads based on this task. The results evaluated in the public dataset AliEC from RecBole and the real-world industry dataset from Alipay show that our GACE method is significantly superior to the SOTA method. In the online A/B test, the click-through rate on three real-world pages from Alipay has increased by 3.6%, 2.13%, and 3.02%, respectively. Especially in the cold-start task, the CTR increased by 9.96%, 7.51%, and 8.97%, respectively.

{{</citation>}}


## cs.NI (1)



### (93/99) A Pragmatical Approach to Anomaly Detection Evaluation in Edge Cloud Systems (Sotiris Skaperas et al., 2024)

{{<citation>}}

Sotiris Skaperas, George Koukis, Ioanna Angeliki Kapetanidou, Vassilis Tsaoussidis, Lefteris Mamatas. (2024)  
**A Pragmatical Approach to Anomaly Detection Evaluation in Edge Cloud Systems**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2401.07717v1)  

---


**ABSTRACT**  
Anomaly detection (AD) has been recently employed in the context of edge cloud computing, e.g., for intrusion detection and identification of performance issues. However, state-of-the-art anomaly detection procedures do not systematically consider restrictions and performance requirements inherent to the edge, such as system responsiveness and resource consumption. In this paper, we attempt to investigate the performance of change-point based detectors, i.e., a class of lightweight and accurate AD methods, in relation to the requirements of edge cloud systems. Firstly, we review the theoretical properties of two major categories of change point approaches, i.e., Bayesian and cumulative sum (CUSUM), also discussing their suitability for edge systems. Secondly, we introduce a novel experimental methodology and apply it over two distinct edge cloud test-beds to evaluate the performance of such mechanisms in real-world edge environments. Our experimental results reveal important insights and trade-offs for the applicability and the online performance of the selected change point detectors.

{{</citation>}}


## eess.SY (1)



### (94/99) Eco-driving Intelligent Systems and Algorithms: A Patent Review (Zhipeng Ma et al., 2024)

{{<citation>}}

Zhipeng Ma, Bo Nørregaard Jørgensen, Zheng Grace Ma. (2024)  
**Eco-driving Intelligent Systems and Algorithms: A Patent Review**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.07559v1)  

---


**ABSTRACT**  
The transportation industry remains a significant contributor to greenhouse gas emissions, highlighting the requirement for intelligent systems to enhance vehicle energy efficiency. The intellectual property rights of developed systems should be protected by patents. However, there is no patent overview of eco-driving intelligent systems. Unlike a scientific article, a patent documentation indicates both novelty and commercialization potential of an inventor. To address this research gap, this paper provides a patent overview of eco-driving intelligent systems and algorithms. 424 patents in the Google Patent database are analyzed. The patent analysis results show that the top three Cooperative Patent Classifications are: Y02T - climate change mitigation technologies related to transportation (50.7%), B60W - Conjoint control of vehicle subunits of different types or different functions (34.4%) and B60L - Propulsion of electrically-propelled vehicles (20.2%). 219 patents were filed after 2016 when deep learning became popular and can be categorized into five groups: vehicle energy management, smart driving, ecological and sustainable driving, fuel consumption reduction, and driving behavior optimization. Furthermore, all 219 patents involve the physical components of the intelligent system and/or novel machine learning/deep learning algorithms. Moreover, over 70% of them are granted by the China National Intellectual Property Administration.

{{</citation>}}


## eess.AS (1)



### (95/99) SeMaScore : a new evaluation metric for automatic speech recognition tasks (Zitha Sasindran et al., 2024)

{{<citation>}}

Zitha Sasindran, Harsha Yelchuri, T. V. Prabhakar. (2024)  
**SeMaScore : a new evaluation metric for automatic speech recognition tasks**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.07506v1)  

---


**ABSTRACT**  
In this study, we present SeMaScore, generated using a segment-wise mapping and scoring algorithm that serves as an evaluation metric for automatic speech recognition tasks. SeMaScore leverages both the error rate and a more robust similarity score. We show that our algorithm's score generation improves upon the state-of-the-art BERTscore. Our experimental results show that SeMaScore corresponds well with expert human assessments, signal-to-noise ratio levels, and other natural language metrics. We outperform BERTscore by 41x in metric computation speed. Overall, we demonstrate that SeMaScore serves as a more dependable evaluation metric, particularly in real-world situations involving atypical speech patterns.

{{</citation>}}


## cs.GT (1)



### (96/99) Existence of MMS Allocations with Mixed Manna (Kevin Hsu, 2024)

{{<citation>}}

Kevin Hsu. (2024)  
**Existence of MMS Allocations with Mixed Manna**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs.GT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07490v1)  

---


**ABSTRACT**  
Maximin share (MMS) allocations are a popular relaxation of envy-free allocations that have received wide attention in the context of the fair division of indivisible items. Although MMS allocations can fail to exist [1], previous work has found conditions under which they exist. Specifically, MMS allocations exist whenever $m \leq n+5$ in the context of goods allocation, and this bound is tight in the sense that MMS allocations can fail to exist when $m = n+6$ [2]. Unfortunately, the technique used to establish this result does not generalize readily to the chores and mixed manna settings. This paper generalizes this result to the chores setting and provides a partial solution for the mixed manna setting. Our results depend on the presence of certain types of agents. Specifically, an agent $i$ is a goods agent (resp. chores agent) if every item is a good (resp. chore) to $i$, and a non-negative mixed agent if $i$ is neither a goods nor a chores agent and the MMS guarantee of $i$ is non-negative. In this paper, we prove that an MMS allocation exists if $m \leq n+5$ and there exists a goods agent, a non-negative mixed agent, or only chores agents.   [1] David Kurokawa, Ariel D Procaccia, and Junxing Wang. When can the maximin share guarantee be guaranteed? In Thirtieth AAAI Conference on Artificial Intelligence, 2016.   [2] Uriel Feige, Ariel Sapir, and Laliv Tauber. A tight negative example for mms fair allocations. In International Conference on Web and Internet Economics, pages 355-372. Springer, 2021.

{{</citation>}}


## cs.MM (1)



### (97/99) Startup Delay Aware Short Video Ordering: Problem, Model, and A Reinforcement Learning based Algorithm (Zhipeng Gao et al., 2024)

{{<citation>}}

Zhipeng Gao, Chunxi Li, Yongxiang Zhao, Baoxian Zhang. (2024)  
**Startup Delay Aware Short Video Ordering: Problem, Model, and A Reinforcement Learning based Algorithm**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.07411v1)  

---


**ABSTRACT**  
Short video applications have attracted billions of users on the Internet and can satisfy diverse users' fragmented spare time with content-rich and duration-short videos. To achieve fast playback at user side, existing short video systems typically enforce burst transmission of initial segment of each video when being requested for improved quality of user experiences. However, such a way of burst transmissions can cause unexpected large startup delays at user side. This is because users may frequently switch videos when sequentially watching a list of short videos recommended by the server side, which can cause excessive burst transmissions of initial segments of different short videos and thus quickly deplete the network transmission capacity. In this paper, we adopt token bucket to characterize the video transmission path between video server and each user, and accordingly study how to effectively reduce the startup delay of short videos by effectively arranging the viewing order of a video list at the server side. We formulate the optimal video ordering problem for minimizing the maximum video startup delay as a combinatorial optimization problem and prove its NP-hardness. We accordingly propose a Partially Shared Actor Critic reinforcement learning algorithm (PSAC) to learn optimized video ordering strategy. Numerical results based on a real dataset provided by a large-scale short video service provider demonstrate that the proposed PSAC algorithm can significantly reduce the video startup delay compared to baseline algorithms.

{{</citation>}}


## cs.SI (1)



### (98/99) Multi-Task DNS Security Analysis via High-Order Heterogeneous Graph Embedding (Meng Qin, 2024)

{{<citation>}}

Meng Qin. (2024)  
**Multi-Task DNS Security Analysis via High-Order Heterogeneous Graph Embedding**  

---
Primary Category: cs.SI  
Categories: cs-CR, cs-SI, cs.SI  
Keywords: Embedding, Security  
[Paper Link](http://arxiv.org/abs/2401.07410v1)  

---


**ABSTRACT**  
DNS is an essential Internet infrastructure to support network applications and services, but is also a significant tool exploited by various cyberattacks. Existing DNS security analysis techniques mostly focus on one specific task associated with one single entity (e.g., domain) via conventional feature engineering. They rely heavily on the labor-intensive feature selection and largely ignore the intrinsic correlations among the heterogeneous DNS entities (e.g., domain and IP). In this paper, I explore the potential of heterogeneous graph embedding to automatically learn the behavior features of multiple DNS entities, and to simultaneously support more than one security tasks. Considering the joint optimization of malicious domain detection and IP reputation evaluation as an example, I propose a novel joint DNS embedding (JDE) model to formulate the DNS query behavior via a similarity-enhanced graph with heterogeneous entities. The random walk technique is applied to the heterogeneous graph to comprehensively explore the hidden homogeneous and heterogeneous high-order proximities among domains and IPs. Extensive experiments on real DNS traffic demonstrate that the joint optimization of multiple tasks with the latent high-order proximities can lead to better security analysis performance for all the tasks than respectively optimizing each single task with the observable low-order proximity.

{{</citation>}}


## cs.CE (1)



### (99/99) Multimodal Language and Graph Learning of Adsorption Configuration in Catalysis (Janghoon Ock et al., 2024)

{{<citation>}}

Janghoon Ock, Rishikesh Magar, Akshay Antony, Amir Barati Farimani. (2024)  
**Multimodal Language and Graph Learning of Adsorption Configuration in Catalysis**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: GNN, Transformer  
[Paper Link](http://arxiv.org/abs/2401.07408v1)  

---


**ABSTRACT**  
Adsorption energy, a reactivity descriptor, should be accurately assessed for efficient catalyst screening. This evaluation requires determining the lowest energy across various adsorption configurations on the catalytic surface. While graph neural networks (GNNs) have gained popularity as a machine learning approach for computing the energy of catalyst systems, they rely heavily on atomic spatial coordinates and often lack clarity in their interpretations. Recent advancements in language models have broadened their applicability to predicting catalytic properties, allowing us to bypass the complexities of graph representation. These models are adept at handling textual data, making it possible to incorporate observable features in a human-readable format. However, language models encounter challenges in accurately predicting the energy of adsorption configurations, typically showing a high mean absolute error (MAE) of about 0.71 eV. Our study addresses this limitation by introducing a self-supervised multi-modal learning approach, termed graph-assisted pretraining. This method significantly reduces the MAE to 0.35 eV through a combination of data augmentation, achieving comparable accuracy with DimeNet++ while using 0.4% of its training data size. Furthermore, the Transformer encoder at the core of the language model can provide insights into the feature focus through its attention scores. This analysis shows that our multimodal training effectively redirects the model's attention toward relevant adsorption configurations from adsorbate-related features, enhancing prediction accuracy and interpretability.

{{</citation>}}
