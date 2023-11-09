---
draft: false
title: "arXiv @ 2023.11.09"
date: 2023-11-09
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.09"
    identifier: arxiv_20231109
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (29)](#cscv-29)
- [cs.NE (1)](#csne-1)
- [cs.RO (7)](#csro-7)
- [cs.CL (45)](#cscl-45)
- [cs.CY (2)](#cscy-2)
- [cs.HC (4)](#cshc-4)
- [eess.SY (3)](#eesssy-3)
- [cs.LG (29)](#cslg-29)
- [cs.IR (3)](#csir-3)
- [cs.CR (3)](#cscr-3)
- [math.NA (1)](#mathna-1)
- [physics.chem-ph (1)](#physicschem-ph-1)
- [cs.AI (6)](#csai-6)
- [cs.SE (2)](#csse-2)
- [eess.SP (1)](#eesssp-1)
- [cs.NI (2)](#csni-2)
- [cs.FL (1)](#csfl-1)
- [quant-ph (1)](#quant-ph-1)
- [cs.PF (1)](#cspf-1)
- [physics.ao-ph (1)](#physicsao-ph-1)
- [cs.CG (1)](#cscg-1)

## cs.CV (29)



### (1/144) 3DiffTection: 3D Object Detection with Geometry-Aware Diffusion Features (Chenfeng Xu et al., 2023)

{{<citation>}}

Chenfeng Xu, Huan Ling, Sanja Fidler, Or Litany. (2023)  
**3DiffTection: 3D Object Detection with Geometry-Aware Diffusion Features**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.04391v1)  

---


**ABSTRACT**  
We present 3DiffTection, a state-of-the-art method for 3D object detection from single images, leveraging features from a 3D-aware diffusion model. Annotating large-scale image data for 3D detection is resource-intensive and time-consuming. Recently, pretrained large image diffusion models have become prominent as effective feature extractors for 2D perception tasks. However, these features are initially trained on paired text and image data, which are not optimized for 3D tasks, and often exhibit a domain gap when applied to the target data. Our approach bridges these gaps through two specialized tuning strategies: geometric and semantic. For geometric tuning, we fine-tune a diffusion model to perform novel view synthesis conditioned on a single image, by introducing a novel epipolar warp operator. This task meets two essential criteria: the necessity for 3D awareness and reliance solely on posed image data, which are readily available (e.g., from videos) and does not require manual annotation. For semantic refinement, we further train the model on target data with detection supervision. Both tuning phases employ ControlNet to preserve the integrity of the original feature capabilities. In the final step, we harness these enhanced capabilities to conduct a test-time prediction ensemble across multiple virtual viewpoints. Through our methodology, we obtain 3D-aware features that are tailored for 3D detection and excel in identifying cross-view point correspondences. Consequently, our model emerges as a powerful 3D detector, substantially surpassing previous benchmarks, e.g., Cube-RCNN, a precedent in single-view 3D detection by 9.43\% in AP3D on the Omni3D-ARkitscene dataset. Furthermore, 3DiffTection showcases robust data efficiency and generalization to cross-domain data.

{{</citation>}}


### (2/144) A Deep Learning Approach to Video Anomaly Detection using Convolutional Autoencoders (Gopikrishna Pavuluri et al., 2023)

{{<citation>}}

Gopikrishna Pavuluri, Gayathri Annem. (2023)  
**A Deep Learning Approach to Video Anomaly Detection using Convolutional Autoencoders**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.04351v1)  

---


**ABSTRACT**  
In this research we propose a deep learning approach for detecting anomalies in videos using convolutional autoencoder and decoder neural networks on the UCSD dataset.Our method utilizes a convolutional autoencoder to learn the spatiotemporal patterns of normal videos and then compares each frame of a test video to this learned representation. We evaluated our approach on the UCSD dataset and achieved an overall accuracy of 99.35% on the Ped1 dataset and 99.77% on the Ped2 dataset, demonstrating the effectiveness of our method for detecting anomalies in surveillance videos. The results show that our method outperforms other state-of-the-art methods, and it can be used in real-world applications for video anomaly detection.

{{</citation>}}


### (3/144) Towards Garment Sewing Pattern Reconstruction from a Single Image (Lijuan Liu et al., 2023)

{{<citation>}}

Lijuan Liu, Xiangyu Xu, Zhijie Lin, Jiabin Liang, Shuicheng Yan. (2023)  
**Towards Garment Sewing Pattern Reconstruction from a Single Image**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-GR, cs-LG, cs-MM, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.04218v1)  

---


**ABSTRACT**  
Garment sewing pattern represents the intrinsic rest shape of a garment, and is the core for many applications like fashion design, virtual try-on, and digital avatars. In this work, we explore the challenging problem of recovering garment sewing patterns from daily photos for augmenting these applications. To solve the problem, we first synthesize a versatile dataset, named SewFactory, which consists of around 1M images and ground-truth sewing patterns for model training and quantitative evaluation. SewFactory covers a wide range of human poses, body shapes, and sewing patterns, and possesses realistic appearances thanks to the proposed human texture synthesis network. Then, we propose a two-level Transformer network called Sewformer, which significantly improves the sewing pattern prediction performance. Extensive experiments demonstrate that the proposed framework is effective in recovering sewing patterns and well generalizes to casually-taken human photos. Code, dataset, and pre-trained models are available at: https://sewformer.github.io.

{{</citation>}}


### (4/144) Deep Hashing via Householder Quantization (Lucas R. Schwengber et al., 2023)

{{<citation>}}

Lucas R. Schwengber, Lucas Resende, Paulo Orenstein, Roberto I. Oliveira. (2023)  
**Deep Hashing via Householder Quantization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-IR, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2311.04207v1)  

---


**ABSTRACT**  
Hashing is at the heart of large-scale image similarity search, and recent methods have been substantially improved through deep learning techniques. Such algorithms typically learn continuous embeddings of the data. To avoid a subsequent costly binarization step, a common solution is to employ loss functions that combine a similarity learning term (to ensure similar images are grouped to nearby embeddings) and a quantization penalty term (to ensure that the embedding entries are close to binarized entries, e.g., -1 or 1). Still, the interaction between these two terms can make learning harder and the embeddings worse. We propose an alternative quantization strategy that decomposes the learning problem in two stages: first, perform similarity learning over the embedding space with no quantization; second, find an optimal orthogonal transformation of the embeddings so each coordinate of the embedding is close to its sign, and then quantize the transformed embedding through the sign function. In the second step, we parametrize orthogonal transformations using Householder matrices to efficiently leverage stochastic gradient descent. Since similarity measures are usually invariant under orthogonal transformations, this quantization strategy comes at no cost in terms of performance. The resulting algorithm is unsupervised, fast, hyperparameter-free and can be run on top of any existing deep hashing or metric learning algorithm. We provide extensive experimental results showing that this approach leads to state-of-the-art performance on widely used image datasets, and, unlike other quantization strategies, brings consistent improvements in performance to existing deep hashing algorithms.

{{</citation>}}


### (5/144) Selective Visual Representations Improve Convergence and Generalization for Embodied AI (Ainaz Eftekhar et al., 2023)

{{<citation>}}

Ainaz Eftekhar, Kuo-Hao Zeng, Jiafei Duan, Ali Farhadi, Ani Kembhavi, Ranjay Krishna. (2023)  
**Selective Visual Representations Improve Convergence and Generalization for Embodied AI**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.04193v1)  

---


**ABSTRACT**  
Embodied AI models often employ off the shelf vision backbones like CLIP to encode their visual observations. Although such general purpose representations encode rich syntactic and semantic information about the scene, much of this information is often irrelevant to the specific task at hand. This introduces noise within the learning process and distracts the agent's focus from task-relevant visual cues. Inspired by selective attention in humans-the process through which people filter their perception based on their experiences, knowledge, and the task at hand-we introduce a parameter-efficient approach to filter visual stimuli for embodied AI. Our approach induces a task-conditioned bottleneck using a small learnable codebook module. This codebook is trained jointly to optimize task reward and acts as a task-conditioned selective filter over the visual observation. Our experiments showcase state-of-the-art performance for object goal navigation and object displacement across 5 benchmarks, ProcTHOR, ArchitecTHOR, RoboTHOR, AI2-iTHOR, and ManipulaTHOR. The filtered representations produced by the codebook are also able generalize better and converge faster when adapted to other simulation environments such as Habitat. Our qualitative analyses show that agents explore their environments more effectively and their representations retain task-relevant information like target object recognition while ignoring superfluous information about other objects. Code and pretrained models are available at our project website: https://embodied-codebook.github.io.

{{</citation>}}


### (6/144) JaSPICE: Automatic Evaluation Metric Using Predicate-Argument Structures for Image Captioning Models (Yuiga Wada et al., 2023)

{{<citation>}}

Yuiga Wada, Kanta Kaneda, Komei Sugiura. (2023)  
**JaSPICE: Automatic Evaluation Metric Using Predicate-Argument Structures for Image Captioning Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: AI, BLEU, Image Captioning  
[Paper Link](http://arxiv.org/abs/2311.04192v1)  

---


**ABSTRACT**  
Image captioning studies heavily rely on automatic evaluation metrics such as BLEU and METEOR. However, such n-gram-based metrics have been shown to correlate poorly with human evaluation, leading to the proposal of alternative metrics such as SPICE for English; however, no equivalent metrics have been established for other languages. Therefore, in this study, we propose an automatic evaluation metric called JaSPICE, which evaluates Japanese captions based on scene graphs. The proposed method generates a scene graph from dependencies and the predicate-argument structure, and extends the graph using synonyms. We conducted experiments employing 10 image captioning models trained on STAIR Captions and PFN-PIC and constructed the Shichimi dataset, which contains 103,170 human evaluations. The results showed that our metric outperformed the baseline metrics for the correlation coefficient with the human evaluation.

{{</citation>}}


### (7/144) A Simple Interpretable Transformer for Fine-Grained Image Classification and Analysis (Dipanjyoti Paul et al., 2023)

{{<citation>}}

Dipanjyoti Paul, Arpita Chowdhury, Xinqi Xiong, Feng-Ju Chang, David Carlyn, Samuel Stevens, Kaiya Provost, Anuj Karpatne, Bryan Carstens, Daniel Rubenstein, Charles Stewart, Tanya Berger-Wolf, Yu Su, Wei-Lun Chao. (2023)  
**A Simple Interpretable Transformer for Fine-Grained Image Classification and Analysis**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Image Classification, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.04157v1)  

---


**ABSTRACT**  
We present a novel usage of Transformers to make image classification interpretable. Unlike mainstream classifiers that wait until the last fully-connected layer to incorporate class information to make predictions, we investigate a proactive approach, asking each class to search for itself in an image. We realize this idea via a Transformer encoder-decoder inspired by DEtection TRansformer (DETR). We learn ``class-specific'' queries (one for each class) as input to the decoder, enabling each class to localize its patterns in an image via cross-attention. We name our approach INterpretable TRansformer (INTR), which is fairly easy to implement and exhibits several compelling properties. We show that INTR intrinsically encourages each class to attend distinctively; the cross-attention weights thus provide a faithful interpretation of the prediction. Interestingly, via ``multi-head'' cross-attention, INTR could identify different ``attributes'' of a class, making it particularly suitable for fine-grained classification and analysis, which we demonstrate on eight datasets. Our code and pre-trained model are publicly accessible at https://github.com/Imageomics/INTR.

{{</citation>}}


### (8/144) Image-Pointcloud Fusion based Anomaly Detection using PD-REAL Dataset (Jianjian Qin et al., 2023)

{{<citation>}}

Jianjian Qin, Chunzhi Gu, Jun Yu, Chao Zhang. (2023)  
**Image-Pointcloud Fusion based Anomaly Detection using PD-REAL Dataset**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.04095v1)  

---


**ABSTRACT**  
We present PD-REAL, a novel large-scale dataset for unsupervised anomaly detection (AD) in the 3D domain. It is motivated by the fact that 2D-only representations in the AD task may fail to capture the geometric structures of anomalies due to uncertainty in lighting conditions or shooting angles. PD-REAL consists entirely of Play-Doh models for 15 object categories and focuses on the analysis of potential benefits from 3D information in a controlled environment. Specifically, objects are first created with six types of anomalies, such as dent, crack, or perforation, and then photographed under different lighting conditions to mimic real-world inspection scenarios. To demonstrate the usefulness of 3D information, we use a commercially available RealSense camera to capture RGB and depth images. Compared to the existing 3D dataset for AD tasks, the data acquisition of PD-REAL is significantly cheaper, easily scalable and easier to control variables. Extensive evaluations with state-of-the-art AD algorithms on our dataset demonstrate the benefits as well as challenges of using 3D information. Our dataset can be downloaded from https://github.com/Andy-cs008/PD-REAL

{{</citation>}}


### (9/144) Proceedings of the 5th International Workshop on Reading Music Systems (Jorge Calvo-Zaragoza et al., 2023)

{{<citation>}}

Jorge Calvo-Zaragoza, Alexander Pacha, Elona Shatri. (2023)  
**Proceedings of the 5th International Workshop on Reading Music Systems**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-IR, cs-LG, cs-SD, cs.CV, eess-AS  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2311.04091v1)  

---


**ABSTRACT**  
The International Workshop on Reading Music Systems (WoRMS) is a workshop that tries to connect researchers who develop systems for reading music, such as in the field of Optical Music Recognition, with other researchers and practitioners that could benefit from such systems, like librarians or musicologists. The relevant topics of interest for the workshop include, but are not limited to: Music reading systems; Optical music recognition; Datasets and performance evaluation; Image processing on music scores; Writer identification; Authoring, editing, storing and presentation systems for music scores; Multi-modal systems; Novel input-methods for music to produce written music; Web-based Music Information Retrieval services; Applications and projects; Use-cases related to written music.   These are the proceedings of the 5th International Workshop on Reading Music Systems, held in Milan, Italy on Nov. 4th 2023.

{{</citation>}}


### (10/144) Augmenting Lane Perception and Topology Understanding with Standard Definition Navigation Maps (Katie Z Luo et al., 2023)

{{<citation>}}

Katie Z Luo, Xinshuo Weng, Yan Wang, Shuang Wu, Jie Li, Kilian Q Weinberger, Yue Wang, Marco Pavone. (2023)  
**Augmenting Lane Perception and Topology Understanding with Standard Definition Navigation Maps**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.04079v1)  

---


**ABSTRACT**  
Autonomous driving has traditionally relied heavily on costly and labor-intensive High Definition (HD) maps, hindering scalability. In contrast, Standard Definition (SD) maps are more affordable and have worldwide coverage, offering a scalable alternative. In this work, we systematically explore the effect of SD maps for real-time lane-topology understanding. We propose a novel framework to integrate SD maps into online map prediction and propose a Transformer-based encoder, SD Map Encoder Representations from transFormers, to leverage priors in SD maps for the lane-topology prediction task. This enhancement consistently and significantly boosts (by up to 60%) lane detection and topology prediction on current state-of-the-art online map prediction methods without bells and whistles and can be immediately incorporated into any Transformer-based lane-topology method. Code is available at https://github.com/NVlabs/SMERF.

{{</citation>}}


### (11/144) LISBET: a self-supervised Transformer model for the automatic segmentation of social behavior motifs (Giuseppe Chindemi et al., 2023)

{{<citation>}}

Giuseppe Chindemi, Benoit Girard, Camilla Bellone. (2023)  
**LISBET: a self-supervised Transformer model for the automatic segmentation of social behavior motifs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, q-bio-QM, stat-ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.04069v1)  

---


**ABSTRACT**  
Social behavior, defined as the process by which individuals act and react in response to others, is crucial for the function of societies and holds profound implications for mental health. To fully grasp the intricacies of social behavior and identify potential therapeutic targets for addressing social deficits, it is essential to understand its core principles. Although machine learning algorithms have made it easier to study specific aspects of complex behavior, current methodologies tend to focus primarily on single-animal behavior. In this study, we introduce LISBET (seLf-supervIsed Social BEhavioral Transformer), a model designed to detect and segment social interactions. Our model eliminates the need for feature selection and extensive human annotation by using self-supervised learning to detect and quantify social behaviors from dynamic body parts tracking data. LISBET can be used in hypothesis-driven mode to automate behavior classification using supervised finetuning, and in discovery-driven mode to segment social behavior motifs using unsupervised learning. We found that motifs recognized using the discovery-driven approach not only closely match the human annotations but also correlate with the electrophysiological activity of dopaminergic neurons in the Ventral Tegmental Area (VTA). We hope LISBET will help the community improve our understanding of social behaviors and their neural underpinnings.

{{</citation>}}


### (12/144) Bias and Diversity in Synthetic-based Face Recognition (Marco Huber et al., 2023)

{{<citation>}}

Marco Huber, Anh Thi Luu, Fadi Boutros, Arjan Kuijper, Naser Damer. (2023)  
**Bias and Diversity in Synthetic-based Face Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.03970v1)  

---


**ABSTRACT**  
Synthetic data is emerging as a substitute for authentic data to solve ethical and legal challenges in handling authentic face data. The current models can create real-looking face images of people who do not exist. However, it is a known and sensitive problem that face recognition systems are susceptible to bias, i.e. performance differences between different demographic and non-demographics attributes, which can lead to unfair decisions. In this work, we investigate how the diversity of synthetic face recognition datasets compares to authentic datasets, and how the distribution of the training data of the generative models affects the distribution of the synthetic data. To do this, we looked at the distribution of gender, ethnicity, age, and head position. Furthermore, we investigated the concrete bias of three recent synthetic-based face recognition models on the studied attributes in comparison to a baseline model trained on authentic data. Our results show that the generator generate a similar distribution as the used training data in terms of the different attributes. With regard to bias, it can be seen that the synthetic-based models share a similar bias behavior with the authentic-based models. However, with the uncovered lower intra-identity attribute consistency seems to be beneficial in reducing bias.

{{</citation>}}


### (13/144) Enhancing Multimodal Compositional Reasoning of Visual Language Models with Generative Negative Mining (Ugur Sahin et al., 2023)

{{<citation>}}

Ugur Sahin, Hang Li, Qadeer Khan, Daniel Cremers, Volker Tresp. (2023)  
**Enhancing Multimodal Compositional Reasoning of Visual Language Models with Generative Negative Mining**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.03964v1)  

---


**ABSTRACT**  
Contemporary large-scale visual language models (VLMs) exhibit strong representation capacities, making them ubiquitous for enhancing image and text understanding tasks. They are often trained in a contrastive manner on a large and diverse corpus of images and corresponding text captions scraped from the internet. Despite this, VLMs often struggle with compositional reasoning tasks which require a fine-grained understanding of the complex interactions of objects and their attributes. This failure can be attributed to two main factors: 1) Contrastive approaches have traditionally focused on mining negative examples from existing datasets. However, the mined negative examples might not be difficult for the model to discriminate from the positive. An alternative to mining would be negative sample generation 2) But existing generative approaches primarily focus on generating hard negative texts associated with a given image. Mining in the other direction, i.e., generating negative image samples associated with a given text has been ignored. To overcome both these limitations, we propose a framework that not only mines in both directions but also generates challenging negative samples in both modalities, i.e., images and texts. Leveraging these generative hard negative samples, we significantly enhance VLMs' performance in tasks involving multimodal compositional reasoning. Our code and dataset are released at https://ugorsahin.github.io/enhancing-multimodal-compositional-reasoning-of-vlm.html.

{{</citation>}}


### (14/144) Improving the Effectiveness of Deep Generative Data (Ruyu Wang et al., 2023)

{{<citation>}}

Ruyu Wang, Sabrina Schmedding, Marco F. Huber. (2023)  
**Improving the Effectiveness of Deep Generative Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.03959v2)  

---


**ABSTRACT**  
Recent deep generative models (DGMs) such as generative adversarial networks (GANs) and diffusion probabilistic models (DPMs) have shown their impressive ability in generating high-fidelity photorealistic images. Although looking appealing to human eyes, training a model on purely synthetic images for downstream image processing tasks like image classification often results in an undesired performance drop compared to training on real data. Previous works have demonstrated that enhancing a real dataset with synthetic images from DGMs can be beneficial. However, the improvements were subjected to certain circumstances and yet were not comparable to adding the same number of real images. In this work, we propose a new taxonomy to describe factors contributing to this commonly observed phenomenon and investigate it on the popular CIFAR-10 dataset. We hypothesize that the Content Gap accounts for a large portion of the performance drop when using synthetic images from DGM and propose strategies to better utilize them in downstream tasks. Extensive experiments on multiple datasets showcase that our method outperforms baselines on downstream classification tasks both in case of training on synthetic only (Synthetic-to-Real) and training on a mix of real and synthetic data (Data Augmentation), particularly in the data-scarce scenario.

{{</citation>}}


### (15/144) FLORA: Fine-grained Low-Rank Architecture Search for Vision Transformer (Chi-Chih Chang et al., 2023)

{{<citation>}}

Chi-Chih Chang, Yuan-Yao Sung, Shixing Yu, Ning-Chi Huang, Diana Marculescu, Kai-Chiang Wu. (2023)  
**FLORA: Fine-grained Low-Rank Architecture Search for Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.03912v1)  

---


**ABSTRACT**  
Vision Transformers (ViT) have recently demonstrated success across a myriad of computer vision tasks. However, their elevated computational demands pose significant challenges for real-world deployment. While low-rank approximation stands out as a renowned method to reduce computational loads, efficiently automating the target rank selection in ViT remains a challenge. Drawing from the notable similarity and alignment between the processes of rank selection and One-Shot NAS, we introduce FLORA, an end-to-end automatic framework based on NAS. To overcome the design challenge of supernet posed by vast search space, FLORA employs a low-rank aware candidate filtering strategy. This method adeptly identifies and eliminates underperforming candidates, effectively alleviating potential undertraining and interference among subnetworks. To further enhance the quality of low-rank supernets, we design a low-rank specific training paradigm. First, we propose weight inheritance to construct supernet and enable gradient sharing among low-rank modules. Secondly, we adopt low-rank aware sampling to strategically allocate training resources, taking into account inherited information from pre-trained models. Empirical results underscore FLORA's efficacy. With our method, a more fine-grained rank configuration can be generated automatically and yield up to 33% extra FLOPs reduction compared to a simple uniform configuration. More specific, FLORA-DeiT-B/FLORA-Swin-B can save up to 55%/42% FLOPs almost without performance degradtion. Importantly, FLORA boasts both versatility and orthogonality, offering an extra 21%-26% FLOPs reduction when integrated with leading compression techniques or compact hybrid structures. Our code is publicly available at https://github.com/shadowpa0327/FLORA.

{{</citation>}}


### (16/144) Mini but Mighty: Finetuning ViTs with Mini Adapters (Imad Eddine Marouf et al., 2023)

{{<citation>}}

Imad Eddine Marouf, Enzo Tartaglione, Stéphane Lathuilière. (2023)  
**Mini but Mighty: Finetuning ViTs with Mini Adapters**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.03873v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) have become one of the dominant architectures in computer vision, and pre-trained ViT models are commonly adapted to new tasks via fine-tuning. Recent works proposed several parameter-efficient transfer learning methods, such as adapters, to avoid the prohibitive training and storage cost of finetuning. In this work, we observe that adapters perform poorly when the dimension of adapters is small, and we propose MiMi, a training framework that addresses this issue. We start with large adapters which can reach high performance, and iteratively reduce their size. To enable automatic estimation of the hidden dimension of every adapter, we also introduce a new scoring function, specifically designed for adapters, that compares the neuron importance across layers. Our method outperforms existing methods in finding the best trade-off between accuracy and trained parameters across the three dataset benchmarks DomainNet, VTAB, and Multi-task, for a total of 29 datasets.

{{</citation>}}


### (17/144) A Comparative Study of Knowledge Transfer Methods for Misaligned Urban Building Labels (Bipul Neupane et al., 2023)

{{<citation>}}

Bipul Neupane, Jagannath Aryal, Abbas Rajabifard. (2023)  
**A Comparative Study of Knowledge Transfer Methods for Misaligned Urban Building Labels**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2311.03867v1)  

---


**ABSTRACT**  
Misalignment in Earth observation (EO) images and building labels impact the training of accurate convolutional neural networks (CNNs) for semantic segmentation of building footprints. Recently, three Teacher-Student knowledge transfer methods have been introduced to address this issue: supervised domain adaptation (SDA), knowledge distillation (KD), and deep mutual learning (DML). However, these methods are merely studied for different urban buildings (low-rise, mid-rise, high-rise, and skyscrapers), where misalignment increases with building height and spatial resolution. In this study, we present a workflow for the systematic comparative study of the three methods. The workflow first identifies the best (with the highest evaluation scores) hyperparameters, lightweight CNNs for the Student (among 43 CNNs from Computer Vision), and encoder-decoder networks (EDNs) for both Teachers and Students. Secondly, three building footprint datasets are developed to train and evaluate the identified Teachers and Students in the three transfer methods. The results show that U-Net with VGG19 (U-VGG19) is the best Teacher, and U-EfficientNetv2B3 and U-EfficientNet-lite0 are among the best Students. With these Teacher-Student pairs, SDA could yield upto 0.943, 0.868, 0.912, and 0.697 F1 scores in the low-rise, mid-rise, high-rise, and skyscrapers respectively. KD and DML provide model compression of upto 82%, despite marginal loss in performance. This new comparison concludes that SDA is the most effective method to address the misalignment problem, while KD and DML can efficiently compress network size without significant loss in performance. The 158 experiments and datasets developed in this study will be valuable to minimise the misaligned labels.

{{</citation>}}


### (18/144) Reducing Spatial Fitting Error in Distillation of Denoising Diffusion Models (Shengzhe Zhou et al., 2023)

{{<citation>}}

Shengzhe Zhou, Zejian Lee, Shengyuan Zhang, Lefan Hou, Changyuan Yang, Guang Yang, Lingyun Sun. (2023)  
**Reducing Spatial Fitting Error in Distillation of Denoising Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.03830v1)  

---


**ABSTRACT**  
Denoising Diffusion models have exhibited remarkable capabilities in image generation. However, generating high-quality samples requires a large number of iterations. Knowledge distillation for diffusion models is an effective method to address this limitation with a shortened sampling process but causes degraded generative quality. Based on our analysis with bias-variance decomposition and experimental observations, we attribute the degradation to the spatial fitting error occurring in the training of both the teacher and student model. Accordingly, we propose $\textbf{S}$patial $\textbf{F}$itting-$\textbf{E}$rror $\textbf{R}$eduction $\textbf{D}$istillation model ($\textbf{SFERD}$). SFERD utilizes attention guidance from the teacher model and a designed semantic gradient predictor to reduce the student's fitting error. Empirically, our proposed model facilitates high-quality sample generation in a few function evaluations. We achieve an FID of 5.31 on CIFAR-10 and 9.39 on ImageNet 64$\times$64 with only one step, outperforming existing diffusion methods. Our study provides a new perspective on diffusion distillation by highlighting the intrinsic denoising ability of models.

{{</citation>}}


### (19/144) Detecting Any Human-Object Interaction Relationship: Universal HOI Detector with Spatial Prompt Learning on Foundation Models (Yichao Cao et al., 2023)

{{<citation>}}

Yichao Cao, Qingfei Tang, Xiu Su, Chen Song, Shan You, Xiaobo Lu, Chang Xu. (2023)  
**Detecting Any Human-Object Interaction Relationship: Universal HOI Detector with Spatial Prompt Learning on Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.03799v1)  

---


**ABSTRACT**  
Human-object interaction (HOI) detection aims to comprehend the intricate relationships between humans and objects, predicting $<human, action, object>$ triplets, and serving as the foundation for numerous computer vision tasks. The complexity and diversity of human-object interactions in the real world, however, pose significant challenges for both annotation and recognition, particularly in recognizing interactions within an open world context. This study explores the universal interaction recognition in an open-world setting through the use of Vision-Language (VL) foundation models and large language models (LLMs). The proposed method is dubbed as \emph{\textbf{UniHOI}}. We conduct a deep analysis of the three hierarchical features inherent in visual HOI detectors and propose a method for high-level relation extraction aimed at VL foundation models, which we call HO prompt-based learning. Our design includes an HO Prompt-guided Decoder (HOPD), facilitates the association of high-level relation representations in the foundation model with various HO pairs within the image. Furthermore, we utilize a LLM (\emph{i.e.} GPT) for interaction interpretation, generating a richer linguistic understanding for complex HOIs. For open-category interaction recognition, our method supports either of two input types: interaction phrase or interpretive sentence. Our efficient architecture design and learning methods effectively unleash the potential of the VL foundation models and LLMs, allowing UniHOI to surpass all existing methods with a substantial margin, under both supervised and zero-shot settings. The code and pre-trained weights are available at: \url{https://github.com/Caoyichao/UniHOI}.

{{</citation>}}


### (20/144) Self-MI: Efficient Multimodal Fusion via Self-Supervised Multi-Task Learning with Auxiliary Mutual Information Maximization (Cam-Van Thi Nguyen et al., 2023)

{{<citation>}}

Cam-Van Thi Nguyen, Ngoc-Hoa Thi Nguyen, Duc-Trong Le, Quang-Thuy Ha. (2023)  
**Self-MI: Efficient Multimodal Fusion via Self-Supervised Multi-Task Learning with Auxiliary Mutual Information Maximization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.03785v1)  

---


**ABSTRACT**  
Multimodal representation learning poses significant challenges in capturing informative and distinct features from multiple modalities. Existing methods often struggle to exploit the unique characteristics of each modality due to unified multimodal annotations. In this study, we propose Self-MI in the self-supervised learning fashion, which also leverage Contrastive Predictive Coding (CPC) as an auxiliary technique to maximize the Mutual Information (MI) between unimodal input pairs and the multimodal fusion result with unimodal inputs. Moreover, we design a label generation module, $ULG_{MI}$ for short, that enables us to create meaningful and informative labels for each modality in a self-supervised manner. By maximizing the Mutual Information, we encourage better alignment between the multimodal fusion and the individual modalities, facilitating improved multimodal fusion. Extensive experiments on three benchmark datasets including CMU-MOSI, CMU-MOSEI, and SIMS, demonstrate the effectiveness of Self-MI in enhancing the multimodal fusion task.

{{</citation>}}


### (21/144) Meta-Adapter: An Online Few-shot Learner for Vision-Language Model (Cheng Cheng et al., 2023)

{{<citation>}}

Cheng Cheng, Lin Song, Ruoyi Xue, Hang Wang, Hongbin Sun, Yixiao Ge, Ying Shan. (2023)  
**Meta-Adapter: An Online Few-shot Learner for Vision-Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.03774v1)  

---


**ABSTRACT**  
The contrastive vision-language pre-training, known as CLIP, demonstrates remarkable potential in perceiving open-world visual concepts, enabling effective zero-shot image recognition. Nevertheless, few-shot learning methods based on CLIP typically require offline fine-tuning of the parameters on few-shot samples, resulting in longer inference time and the risk of over-fitting in certain domains. To tackle these challenges, we propose the Meta-Adapter, a lightweight residual-style adapter, to refine the CLIP features guided by the few-shot samples in an online manner. With a few training samples, our method can enable effective few-shot learning capabilities and generalize to unseen data or tasks without additional fine-tuning, achieving competitive performance and high efficiency. Without bells and whistles, our approach outperforms the state-of-the-art online few-shot learning method by an average of 3.6\% on eight image classification datasets with higher inference speed. Furthermore, our model is simple and flexible, serving as a plug-and-play module directly applicable to downstream tasks. Without further fine-tuning, Meta-Adapter obtains notable performance improvements in open-vocabulary object detection and segmentation tasks.

{{</citation>}}


### (22/144) Lightweight Portrait Matting via Regional Attention and Refinement (Yatao Zhong et al., 2023)

{{<citation>}}

Yatao Zhong, Ilya Zharkov. (2023)  
**Lightweight Portrait Matting via Regional Attention and Refinement**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.03770v1)  

---


**ABSTRACT**  
We present a lightweight model for high resolution portrait matting. The model does not use any auxiliary inputs such as trimaps or background captures and achieves real time performance for HD videos and near real time for 4K. Our model is built upon a two-stage framework with a low resolution network for coarse alpha estimation followed by a refinement network for local region improvement. However, a naive implementation of the two-stage model suffers from poor matting quality if not utilizing any auxiliary inputs. We address the performance gap by leveraging the vision transformer (ViT) as the backbone of the low resolution network, motivated by the observation that the tokenization step of ViT can reduce spatial resolution while retain as much pixel information as possible. To inform local regions of the context, we propose a novel cross region attention (CRA) module in the refinement network to propagate the contextual information across the neighboring regions. We demonstrate that our method achieves superior results and outperforms other baselines on three benchmark datasets while only uses $1/20$ of the FLOPS compared to the existing state-of-the-art model.

{{</citation>}}


### (23/144) Multiclass Segmentation using Teeth Attention Modules for Dental X-ray Images (Afnan Ghafoor et al., 2023)

{{<citation>}}

Afnan Ghafoor, Seong-Yong Moon, Bumshik Lee. (2023)  
**Multiclass Segmentation using Teeth Attention Modules for Dental X-ray Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.03749v1)  

---


**ABSTRACT**  
This paper proposed a cutting-edge multiclass teeth segmentation architecture that integrates an M-Net-like structure with Swin Transformers and a novel component named Teeth Attention Block (TAB). Existing teeth image segmentation methods have issues with less accurate and unreliable segmentation outcomes due to the complex and varying morphology of teeth, although teeth segmentation in dental panoramic images is essential for dental disease diagnosis. We propose a novel teeth segmentation model incorporating an M-Net-like structure with Swin Transformers and TAB. The proposed TAB utilizes a unique attention mechanism that focuses specifically on the complex structures of teeth. The attention mechanism in TAB precisely highlights key elements of teeth features in panoramic images, resulting in more accurate segmentation outcomes. The proposed architecture effectively captures local and global contextual information, accurately defining each tooth and its surrounding structures. Furthermore, we employ a multiscale supervision strategy, which leverages the left and right legs of the U-Net structure, boosting the performance of the segmentation with enhanced feature representation. The squared Dice loss is utilized to tackle the class imbalance issue, ensuring accurate segmentation across all classes. The proposed method was validated on a panoramic teeth X-ray dataset, which was taken in a real-world dental diagnosis. The experimental results demonstrate the efficacy of our proposed architecture for tooth segmentation on multiple benchmark dental image datasets, outperforming existing state-of-the-art methods in objective metrics and visual examinations. This study has the potential to significantly enhance dental image analysis and contribute to advances in dental applications.

{{</citation>}}


### (24/144) SBCFormer: Lightweight Network Capable of Full-size ImageNet Classification at 1 FPS on Single Board Computers (Xiangyong Lu et al., 2023)

{{<citation>}}

Xiangyong Lu, Masanori Suganuma, Takayuki Okatani. (2023)  
**SBCFormer: Lightweight Network Capable of Full-size ImageNet Classification at 1 FPS on Single Board Computers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2311.03747v1)  

---


**ABSTRACT**  
Computer vision has become increasingly prevalent in solving real-world problems across diverse domains, including smart agriculture, fishery, and livestock management. These applications may not require processing many image frames per second, leading practitioners to use single board computers (SBCs). Although many lightweight networks have been developed for mobile/edge devices, they primarily target smartphones with more powerful processors and not SBCs with the low-end CPUs. This paper introduces a CNN-ViT hybrid network called SBCFormer, which achieves high accuracy and fast computation on such low-end CPUs. The hardware constraints of these CPUs make the Transformer's attention mechanism preferable to convolution. However, using attention on low-end CPUs presents a challenge: high-resolution internal feature maps demand excessive computational resources, but reducing their resolution results in the loss of local image details. SBCFormer introduces an architectural design to address this issue. As a result, SBCFormer achieves the highest trade-off between accuracy and speed on a Raspberry Pi 4 Model B with an ARM-Cortex A72 CPU. For the first time, it achieves an ImageNet-1K top-1 accuracy of around 80% at a speed of 1.0 frame/sec on the SBC. Code is available at https://github.com/xyongLu/SBCFormer.

{{</citation>}}


### (25/144) Unsupervised Video Summarization (Hanqing Li et al., 2023)

{{<citation>}}

Hanqing Li, Diego Klabjan, Jean Utke. (2023)  
**Unsupervised Video Summarization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2311.03745v1)  

---


**ABSTRACT**  
This paper introduces a new, unsupervised method for automatic video summarization using ideas from generative adversarial networks but eliminating the discriminator, having a simple loss function, and separating training of different parts of the model. An iterative training strategy is also applied by alternately training the reconstructor and the frame selector for multiple iterations. Furthermore, a trainable mask vector is added to the model in summary generation during training and evaluation. The method also includes an unsupervised model selection algorithm. Results from experiments on two public datasets (SumMe and TVSum) and four datasets we created (Soccer, LoL, MLB, and ShortMLB) demonstrate the effectiveness of each component on the model performance, particularly the iterative training strategy. Evaluations and comparisons with the state-of-the-art methods highlight the advantages of the proposed method in performance, stability, and training efficiency.

{{</citation>}}


### (26/144) 3DifFusionDet: Diffusion Model for 3D Object Detection with Robust LiDAR-Camera Fusion (Xinhao Xiang et al., 2023)

{{<citation>}}

Xinhao Xiang, Simon Dräger, Jiawei Zhang. (2023)  
**3DifFusionDet: Diffusion Model for 3D Object Detection with Robust LiDAR-Camera Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.03742v1)  

---


**ABSTRACT**  
Good 3D object detection performance from LiDAR-Camera sensors demands seamless feature alignment and fusion strategies. We propose the 3DifFusionDet framework in this paper, which structures 3D object detection as a denoising diffusion process from noisy 3D boxes to target boxes. In this framework, ground truth boxes diffuse in a random distribution for training, and the model learns to reverse the noising process. During inference, the model gradually refines a set of boxes that were generated at random to the outcomes. Under the feature align strategy, the progressive refinement method could make a significant contribution to robust LiDAR-Camera fusion. The iterative refinement process could also demonstrate great adaptability by applying the framework to various detecting circumstances where varying levels of accuracy and speed are required. Extensive experiments on KITTI, a benchmark for real-world traffic object identification, revealed that 3DifFusionDet is able to perform favorably in comparison to earlier, well-respected detectors.

{{</citation>}}


### (27/144) DeepInspect: An AI-Powered Defect Detection for Manufacturing Industries (Arti Kumbhar et al., 2023)

{{<citation>}}

Arti Kumbhar, Amruta Chougule, Priya Lokhande, Saloni Navaghane, Aditi Burud, Saee Nimbalkar. (2023)  
**DeepInspect: An AI-Powered Defect Detection for Manufacturing Industries**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.03725v2)  

---


**ABSTRACT**  
Utilizing Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Generative Adversarial Networks (GANs), our system introduces an innovative approach to defect detection in manufacturing. This technology excels in precisely identifying faults by extracting intricate details from product photographs, utilizing RNNs to detect evolving errors and generating synthetic defect data to bolster the model's robustness and adaptability across various defect scenarios. The project leverages a deep learning framework to automate real-time flaw detection in the manufacturing process. It harnesses extensive datasets of annotated images to discern complex defect patterns. This integrated system seamlessly fits into production workflows, thereby boosting efficiency and elevating product quality. As a result, it reduces waste and operational costs, ultimately enhancing market competitiveness.

{{</citation>}}


### (28/144) Random Field Augmentations for Self-Supervised Representation Learning (Philip Andrew Mansfield et al., 2023)

{{<citation>}}

Philip Andrew Mansfield, Arash Afkanpour, Warren Richard Morningstar, Karan Singhal. (2023)  
**Random Field Augmentations for Self-Supervised Representation Learning**  

---
Primary Category: cs.CV  
Categories: I-2-6; I-2-10; I-5-1, cs-CV, cs-LG, cs.CV  
Keywords: Augmentation, ImageNet, Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.03629v1)  

---


**ABSTRACT**  
Self-supervised representation learning is heavily dependent on data augmentations to specify the invariances encoded in representations. Previous work has shown that applying diverse data augmentations is crucial to downstream performance, but augmentation techniques remain under-explored. In this work, we propose a new family of local transformations based on Gaussian random fields to generate image augmentations for self-supervised representation learning. These transformations generalize the well-established affine and color transformations (translation, rotation, color jitter, etc.) and greatly increase the space of augmentations by allowing transformation parameter values to vary from pixel to pixel. The parameters are treated as continuous functions of spatial coordinates, and modeled as independent Gaussian random fields. Empirical results show the effectiveness of the new transformations for self-supervised representation learning. Specifically, we achieve a 1.7% top-1 accuracy improvement over baseline on ImageNet downstream classification, and a 3.6% improvement on out-of-distribution iNaturalist downstream classification. However, due to the flexibility of the new transformations, learned representations are sensitive to hyperparameters. While mild transformations improve representations, we observe that strong transformations can degrade the structure of an image, indicating that balancing the diversity and strength of augmentations is important for improving generalization of learned representations.

{{</citation>}}


### (29/144) FusionViT: Hierarchical 3D Object Detection via LiDAR-Camera Vision Transformer Fusion (Xinhao Xiang et al., 2023)

{{<citation>}}

Xinhao Xiang, Jiawei Zhang. (2023)  
**FusionViT: Hierarchical 3D Object Detection via LiDAR-Camera Vision Transformer Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2311.03620v1)  

---


**ABSTRACT**  
For 3D object detection, both camera and lidar have been demonstrated to be useful sensory devices for providing complementary information about the same scenery with data representations in different modalities, e.g., 2D RGB image vs 3D point cloud. An effective representation learning and fusion of such multi-modal sensor data is necessary and critical for better 3D object detection performance. To solve the problem, in this paper, we will introduce a novel vision transformer-based 3D object detection model, namely FusionViT. Different from the existing 3D object detection approaches, FusionViT is a pure-ViT based framework, which adopts a hierarchical architecture by extending the transformer model to embed both images and point clouds for effective representation learning. Such multi-modal data embedding representations will be further fused together via a fusion vision transformer model prior to feeding the learned features to the object detection head for both detection and localization of the 3D objects in the input scenery. To demonstrate the effectiveness of FusionViT, extensive experiments have been done on real-world traffic object detection benchmark datasets KITTI and Waymo Open. Notably, our FusionViT model can achieve state-of-the-art performance and outperforms not only the existing baseline methods that merely rely on camera images or lidar point clouds, but also the latest multi-modal image-point cloud deep fusion approaches.

{{</citation>}}


## cs.NE (1)



### (30/144) Harnessing Manycore Processors with Distributed Memory for Accelerated Training of Sparse and Recurrent Models (Jan Finkbeiner et al., 2023)

{{<citation>}}

Jan Finkbeiner, Thomas Gmeinder, Mark Pupilli, Alexander Titterton, Emre Neftci. (2023)  
**Harnessing Manycore Processors with Distributed Memory for Accelerated Training of Sparse and Recurrent Models**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-NE, cs.NE  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.04386v1)  

---


**ABSTRACT**  
Current AI training infrastructure is dominated by single instruction multiple data (SIMD) and systolic array architectures, such as Graphics Processing Units (GPUs) and Tensor Processing Units (TPUs), that excel at accelerating parallel workloads and dense vector matrix multiplications. Potentially more efficient neural network models utilizing sparsity and recurrence cannot leverage the full power of SIMD processor and are thus at a severe disadvantage compared to today's prominent parallel architectures like Transformers and CNNs, thereby hindering the path towards more sustainable AI. To overcome this limitation, we explore sparse and recurrent model training on a massively parallel multiple instruction multiple data (MIMD) architecture with distributed local memory. We implement a training routine based on backpropagation through time (BPTT) for the brain-inspired class of Spiking Neural Networks (SNNs) that feature binary sparse activations. We observe a massive advantage in using sparse activation tensors with a MIMD processor, the Intelligence Processing Unit (IPU) compared to GPUs. On training workloads, our results demonstrate 5-10x throughput gains compared to A100 GPUs and up to 38x gains for higher levels of activation sparsity, without a significant slowdown in training convergence or reduction in final model performance. Furthermore, our results show highly promising trends for both single and multi IPU configurations as we scale up to larger model sizes. Our work paves the way towards more efficient, non-standard models via AI training hardware beyond GPUs, and competitive large scale SNN models.

{{</citation>}}


## cs.RO (7)



### (31/144) Active Collision Avoidance System for E-Scooters in Pedestrian Environment (Xuke Yan et al., 2023)

{{<citation>}}

Xuke Yan, Dan Shen. (2023)  
**Active Collision Avoidance System for E-Scooters in Pedestrian Environment**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.04383v1)  

---


**ABSTRACT**  
In the dense fabric of urban areas, electric scooters have rapidly become a preferred mode of transportation. As they cater to modern mobility demands, they present significant safety challenges, especially when interacting with pedestrians. In general, e-scooters are suggested to be ridden in bike lanes/sidewalks or share the road with cars at the maximum speed of about 15-20 mph, which is more flexible and much faster than pedestrians and bicyclists. Accurate prediction of pedestrian movement, coupled with assistant motion control of scooters, is essential in minimizing collision risks and seamlessly integrating scooters in areas dense with pedestrians. Addressing these safety concerns, our research introduces a novel e-Scooter collision avoidance system (eCAS) with a method for predicting pedestrian trajectories, employing an advanced LSTM network integrated with a state refinement module. This proactive model is designed to ensure unobstructed movement in areas with substantial pedestrian traffic without collisions. Results are validated on two public datasets, ETH and UCY, providing encouraging outcomes. Our model demonstrated proficiency in anticipating pedestrian paths and augmented scooter path planning, allowing for heightened adaptability in densely populated locales. This study shows the potential of melding pedestrian trajectory prediction with scooter motion planning. With the ubiquity of electric scooters in urban environments, such advancements have become crucial to safeguard all participants in urban transit.

{{</citation>}}


### (32/144) From Diagram to Deployment: Translating BPMN Collaborations into X-Klaim for Efficient Multi-Robot System Programming (Khalid Bourr et al., 2023)

{{<citation>}}

Khalid Bourr, Francesco Tiezzi. (2023)  
**From Diagram to Deployment: Translating BPMN Collaborations into X-Klaim for Efficient Multi-Robot System Programming**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SE, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.04126v1)  

---


**ABSTRACT**  
This paper introduces a novel method for translating Business Process Model and Notation (BPMN) diagrams into executable X-Klaim code for Multi-Robot Systems (MRSs). Merging the clarity of BPMN with the operational strength of X-Klaim, we enable the design and execution of complex robotic interactions without requiring in-depth knowledge of the underlying programming language from the users. Our approach maintains the BPMN model's core design principles and logic in the translation to X-Klaim, thus enhancing the readability and maintainability of MRS applications. We offer a series of translated examples, address optimization strategies, and introduce the B2XKLAIM tool, which automates the conversion process. This method aims to streamline MRS programming and improve collaboration between roboticists and domain experts throughout the design and implementation stages.

{{</citation>}}


### (33/144) Interactive Semantic Map Representation for Skill-based Visual Object Navigation (Tatiana Zemskova et al., 2023)

{{<citation>}}

Tatiana Zemskova, Aleksei Staroverov, Kirill Muravyev, Dmitry Yudin, Aleksandr Panov. (2023)  
**Interactive Semantic Map Representation for Skill-based Visual Object Navigation**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.04107v1)  

---


**ABSTRACT**  
Visual object navigation using learning methods is one of the key tasks in mobile robotics. This paper introduces a new representation of a scene semantic map formed during the embodied agent interaction with the indoor environment. It is based on a neural network method that adjusts the weights of the segmentation model with backpropagation of the predicted fusion loss values during inference on a regular (backward) or delayed (forward) image sequence. We have implemented this representation into a full-fledged navigation approach called SkillTron, which can select robot skills from end-to-end policies based on reinforcement learning and classic map-based planning methods. The proposed approach makes it possible to form both intermediate goals for robot exploration and the final goal for object navigation. We conducted intensive experiments with the proposed approach in the Habitat environment, which showed a significant superiority in navigation quality metrics compared to state-of-the-art approaches. The developed code and used custom datasets are publicly available at github.com/AIRI-Institute/skill-fusion.

{{</citation>}}


### (34/144) Estimator-Coupled Reinforcement Learning for Robust Purely Tactile In-Hand Manipulation (Lennart Röstel et al., 2023)

{{<citation>}}

Lennart Röstel, Johannes Pitz, Leon Sievers, Berthold Bäuml. (2023)  
**Estimator-Coupled Reinforcement Learning for Robust Purely Tactile In-Hand Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.04060v1)  

---


**ABSTRACT**  
This paper identifies and addresses the problems with naively combining (reinforcement) learning-based controllers and state estimators for robotic in-hand manipulation. Specifically, we tackle the challenging task of purely tactile, goal-conditioned, dextrous in-hand reorientation with the hand pointing downwards. Due to the limited sensing available, many control strategies that are feasible in simulation when having full knowledge of the object's state do not allow for accurate state estimation. Hence, separately training the controller and the estimator and combining the two at test time leads to poor performance. We solve this problem by coupling the control policy to the state estimator already during training in simulation. This approach leads to more robust state estimation and overall higher performance on the task while maintaining an interpretability advantage over end-to-end policy learning. With our GPU-accelerated implementation, learning from scratch takes a median training time of only 6.5 hours on a single, low-cost GPU. In simulation experiments with the DLR-Hand II and for four significantly different object shapes, we provide an in-depth analysis of the performance of our approach. We demonstrate the successful sim2real transfer by rotating the four objects to all 24 orientations in the $\pi/2$ discretization of SO(3), which has never been achieved for such a diverse set of shapes. Finally, our method can reorient a cube consecutively to nine goals (median), which was beyond the reach of previous methods in this challenging setting.

{{</citation>}}


### (35/144) Enhanced Information Extraction from Cylindrical Visual-Tactile Sensors via Image Fusion (Zilan Li et al., 2023)

{{<citation>}}

Zilan Li, Zhibin Zou, Weiliang Xu, Yuanzhi Zhou, Guoyuan Zhou, Xuan Huang, Xinming Li. (2023)  
**Enhanced Information Extraction from Cylindrical Visual-Tactile Sensors via Image Fusion**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2311.04002v1)  

---


**ABSTRACT**  
Vision-based tactile sensors equipped with planar contact structures acquire the shape, force, and motion states of objects in contact. The limited planar contact area presents a challenge in acquiring information about larger target objects. In contrast, vision-based tactile sensors with cylindrical contact structures could extend the contact area by rolling, which can acquire much tactile information that exceeds the sensing projection area in a single contact. However, the tactile data acquired by cylindrical structures does not consistently correspond to the same depth level. Therefore, stitching and analyzing the data in an extended contact area is a challenging problem. In this work, we propose an image fusion method based on cylindrical vision-based tactile sensors. The method takes advantage of the changing characteristics of the contact depth of cylindrical structures, extracts the effective information of different contact depths in the frequency domain, and performs differential fusion for the information characteristics. The results show that in object contact confronting an area larger than single sensing, the images fused with our proposed method have higher information and structural similarity compared with the method of stitching based on motion distance sampling. Meanwhile, it is robust to sampling time. We complement this method with a deep neural network to illustrate its potential for fusing and recognizing object contact information using cylindrical vision-based tactile sensors.

{{</citation>}}


### (36/144) Deep Bayesian Reinforcement Learning for Spacecraft Proximity Maneuvers and Docking (Desong Du et al., 2023)

{{<citation>}}

Desong Du, Naiming Qi, Yanfang Liu, Wei Pan. (2023)  
**Deep Bayesian Reinforcement Learning for Spacecraft Proximity Maneuvers and Docking**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.03680v1)  

---


**ABSTRACT**  
In the pursuit of autonomous spacecraft proximity maneuvers and docking(PMD), we introduce a novel Bayesian actor-critic reinforcement learning algorithm to learn a control policy with the stability guarantee. The PMD task is formulated as a Markov decision process that reflects the relative dynamic model, the docking cone and the cost function. Drawing from the principles of Lyapunov theory, we frame the temporal difference learning as a constrained Gaussian process regression problem. This innovative approach allows the state-value function to be expressed as a Lyapunov function, leveraging the Gaussian process and deep kernel learning. We develop a novel Bayesian quadrature policy optimization procedure to analytically compute the policy gradient while integrating Lyapunov-based stability constraints. This integration is pivotal in satisfying the rigorous safety demands of spaceflight missions. The proposed algorithm has been experimentally evaluated on a spacecraft air-bearing testbed and shows impressive and promising performance.

{{</citation>}}


### (37/144) TWIST: Teacher-Student World Model Distillation for Efficient Sim-to-Real Transfer (Jun Yamada et al., 2023)

{{<citation>}}

Jun Yamada, Marc Rigter, Jack Collins, Ingmar Posner. (2023)  
**TWIST: Teacher-Student World Model Distillation for Efficient Sim-to-Real Transfer**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Model Distillation  
[Paper Link](http://arxiv.org/abs/2311.03622v1)  

---


**ABSTRACT**  
Model-based RL is a promising approach for real-world robotics due to its improved sample efficiency and generalization capabilities compared to model-free RL. However, effective model-based RL solutions for vision-based real-world applications require bridging the sim-to-real gap for any world model learnt. Due to its significant computational cost, standard domain randomisation does not provide an effective solution to this problem. This paper proposes TWIST (Teacher-Student World Model Distillation for Sim-to-Real Transfer) to achieve efficient sim-to-real transfer of vision-based model-based RL using distillation. Specifically, TWIST leverages state observations as readily accessible, privileged information commonly garnered from a simulator to significantly accelerate sim-to-real transfer. Specifically, a teacher world model is trained efficiently on state information. At the same time, a matching dataset is collected of domain-randomised image observations. The teacher world model then supervises a student world model that takes the domain-randomised image observations as input. By distilling the learned latent dynamics model from the teacher to the student model, TWIST achieves efficient and effective sim-to-real transfer for vision-based model-based RL tasks. Experiments in simulated and real robotics tasks demonstrate that our approach outperforms naive domain randomisation and model-free methods in terms of sample efficiency and task performance of sim-to-real transfer.

{{</citation>}}


## cs.CL (45)



### (38/144) Evaluating multiple large language models in pediatric ophthalmology (Jason Holmes et al., 2023)

{{<citation>}}

Jason Holmes, Rui Peng, Yiwei Li, Jinyu Hu, Zhengliang Liu, Zihao Wu, Huan Zhao, Xi Jiang, Wei Liu, Hong Wei, Jie Zou, Tianming Liu, Yi Shao. (2023)  
**Evaluating multiple large language models in pediatric ophthalmology**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-3.5, GPT-4, PaLM  
[Paper Link](http://arxiv.org/abs/2311.04368v1)  

---


**ABSTRACT**  
IMPORTANCE The response effectiveness of different large language models (LLMs) and various individuals, including medical students, graduate students, and practicing physicians, in pediatric ophthalmology consultations, has not been clearly established yet. OBJECTIVE Design a 100-question exam based on pediatric ophthalmology to evaluate the performance of LLMs in highly specialized scenarios and compare them with the performance of medical students and physicians at different levels. DESIGN, SETTING, AND PARTICIPANTS This survey study assessed three LLMs, namely ChatGPT (GPT-3.5), GPT-4, and PaLM2, were assessed alongside three human cohorts: medical students, postgraduate students, and attending physicians, in their ability to answer questions related to pediatric ophthalmology. It was conducted by administering questionnaires in the form of test papers through the LLM network interface, with the valuable participation of volunteers. MAIN OUTCOMES AND MEASURES Mean scores of LLM and humans on 100 multiple-choice questions, as well as the answer stability, correlation, and response confidence of each LLM. RESULTS GPT-4 performed comparably to attending physicians, while ChatGPT (GPT-3.5) and PaLM2 outperformed medical students but slightly trailed behind postgraduate students. Furthermore, GPT-4 exhibited greater stability and confidence when responding to inquiries compared to ChatGPT (GPT-3.5) and PaLM2. CONCLUSIONS AND RELEVANCE Our results underscore the potential for LLMs to provide medical assistance in pediatric ophthalmology and suggest significant capacity to guide the education of medical students.

{{</citation>}}


### (39/144) Syntax-Guided Transformers: Elevating Compositional Generalization and Grounding in Multimodal Environments (Danial Kamali et al., 2023)

{{<citation>}}

Danial Kamali, Parisa Kordjamshidi. (2023)  
**Syntax-Guided Transformers: Elevating Compositional Generalization and Grounding in Multimodal Environments**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-RO, cs.CL  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.04364v1)  

---


**ABSTRACT**  
Compositional generalization, the ability of intelligent models to extrapolate understanding of components to novel compositions, is a fundamental yet challenging facet in AI research, especially within multimodal environments. In this work, we address this challenge by exploiting the syntactic structure of language to boost compositional generalization. This paper elevates the importance of syntactic grounding, particularly through attention masking techniques derived from text input parsing. We introduce and evaluate the merits of using syntactic information in the multimodal grounding problem. Our results on grounded compositional generalization underscore the positive impact of dependency parsing across diverse tasks when utilized with Weight Sharing across the Transformer encoder. The results push the state-of-the-art in multimodal grounding and parameter-efficient modeling and provide insights for future research.

{{</citation>}}


### (40/144) Uncovering Causal Variables in Transformers using Circuit Probing (Michael A. Lepori et al., 2023)

{{<citation>}}

Michael A. Lepori, Thomas Serre, Ellie Pavlick. (2023)  
**Uncovering Causal Variables in Transformers using Circuit Probing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.04354v1)  

---


**ABSTRACT**  
Neural network models have achieved high performance on a wide variety of complex tasks, but the algorithms that they implement are notoriously difficult to interpret. In order to understand these algorithms, it is often necessary to hypothesize intermediate variables involved in the network's computation. For example, does a language model depend on particular syntactic properties when generating a sentence? However, existing analysis tools make it difficult to test hypotheses of this type. We propose a new analysis technique -- circuit probing -- that automatically uncovers low-level circuits that compute hypothesized intermediate variables. This enables causal analysis through targeted ablation at the level of model parameters. We apply this method to models trained on simple arithmetic tasks, demonstrating its effectiveness at (1) deciphering the algorithms that models have learned, (2) revealing modular structure within a model, and (3) tracking the development of circuits over training. We compare circuit probing to other methods across these three experiments, and find it on par or more effective than existing analysis methods. Finally, we demonstrate circuit probing on a real-world use case, uncovering circuits that are responsible for subject-verb agreement and reflexive anaphora in GPT2-Small and Medium.

{{</citation>}}


### (41/144) Evaluating the Effectiveness of Retrieval-Augmented Large Language Models in Scientific Document Reasoning (Sai Munikoti et al., 2023)

{{<citation>}}

Sai Munikoti, Anurag Acharya, Sridevi Wagle, Sameera Horawalavithana. (2023)  
**Evaluating the Effectiveness of Retrieval-Augmented Large Language Models in Scientific Document Reasoning**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.04348v1)  

---


**ABSTRACT**  
Despite the dramatic progress in Large Language Model (LLM) development, LLMs often provide seemingly plausible but not factual information, often referred to as hallucinations. Retrieval-augmented LLMs provide a non-parametric approach to solve these issues by retrieving relevant information from external data sources and augment the training process. These models help to trace evidence from an externally provided knowledge base allowing the model predictions to be better interpreted and verified. In this work, we critically evaluate these models in their ability to perform in scientific document reasoning tasks. To this end, we tuned multiple such model variants with science-focused instructions and evaluated them on a scientific document reasoning benchmark for the usefulness of the retrieved document passages. Our findings suggest that models justify predictions in science tasks with fabricated evidence and leveraging scientific corpus as pretraining data does not alleviate the risk of evidence fabrication.

{{</citation>}}


### (42/144) Sub-Sentence Encoder: Contrastive Learning of Propositional Semantic Representations (Sihao Chen et al., 2023)

{{<citation>}}

Sihao Chen, Hongming Zhang, Tong Chen, Ben Zhou, Wenhao Yu, Dian Yu, Baolin Peng, Hongwei Wang, Dan Roth, Dong Yu. (2023)  
**Sub-Sentence Encoder: Contrastive Learning of Propositional Semantic Representations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.04335v1)  

---


**ABSTRACT**  
We introduce sub-sentence encoder, a contrastively-learned contextual embedding model for fine-grained semantic representation of text. In contrast to the standard practice with sentence embeddings, where the meaning of an entire sequence of text is encoded into a fixed-length vector, the sub-sentence encoder learns to produce distinct contextual embeddings corresponding to different atomic propositions, i.e. atomic units of meaning expressed within a text sequence. The sub-sentence embeddings are contrastively learned to recognize (inferred) semantic equivalence between propositions across different text sequences. Our experiments show the effectiveness of sub-sentence encoders in applications, such as retrieving supporting facts for fine-grained text attribution or recognizing the conditional semantic similarity between texts. In practice, we demonstrate that sub-sentence encoders keep the same level of inference cost and space complexity compared to sentence encoders.

{{</citation>}}


### (43/144) Formal Aspects of Language Modeling (Ryan Cotterell et al., 2023)

{{<citation>}}

Ryan Cotterell, Anej Svete, Clara Meister, Tianyu Liu, Li Du. (2023)  
**Formal Aspects of Language Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.04329v1)  

---


**ABSTRACT**  
Large language models have become one of the most commonly deployed NLP inventions. In the past half-decade, their integration into core natural language processing tools has dramatically increased the performance of such tools, and they have entered the public discourse surrounding artificial intelligence. Consequently, it is important for both developers and researchers alike to understand the mathematical foundations of large language models, as well as how to implement them. These notes are the accompaniment to the theoretical portion of the ETH Z\"urich course on large language models, covering what constitutes a language model from a formal, theoretical perspective.

{{</citation>}}


### (44/144) Aspect-based Meeting Transcript Summarization: A Two-Stage Approach with Weak Supervision on Sentence Classification (Zhongfen Deng et al., 2023)

{{<citation>}}

Zhongfen Deng, Seunghyun Yoon, Trung Bui, Franck Dernoncourt, Quan Hung Tran, Shuaiqi Liu, Wenting Zhao, Tao Zhang, Yibo Wang, Philip S. Yu. (2023)  
**Aspect-based Meeting Transcript Summarization: A Two-Stage Approach with Weak Supervision on Sentence Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2311.04292v1)  

---


**ABSTRACT**  
Aspect-based meeting transcript summarization aims to produce multiple summaries, each focusing on one aspect of content in a meeting transcript. It is challenging as sentences related to different aspects can mingle together, and those relevant to a specific aspect can be scattered throughout the long transcript of a meeting. The traditional summarization methods produce one summary mixing information of all aspects, which cannot deal with the above challenges of aspect-based meeting transcript summarization. In this paper, we propose a two-stage method for aspect-based meeting transcript summarization. To select the input content related to specific aspects, we train a sentence classifier on a dataset constructed from the AMI corpus with pseudo-labeling. Then we merge the sentences selected for a specific aspect as the input for the summarizer to produce the aspect-based summary. Experimental results on the AMI corpus outperform many strong baselines, which verifies the effectiveness of our proposed method.

{{</citation>}}


### (45/144) CRAB: Assessing the Strength of Causal Relationships Between Real-world Events (Angelika Romanou et al., 2023)

{{<citation>}}

Angelika Romanou, Syrielle Montariol, Debjit Paul, Leo Laugier, Karl Aberer, Antoine Bosselut. (2023)  
**CRAB: Assessing the Strength of Causal Relationships Between Real-world Events**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP, Reasoning, Twitter  
[Paper Link](http://arxiv.org/abs/2311.04284v1)  

---


**ABSTRACT**  
Understanding narratives requires reasoning about the cause-and-effect relationships between events mentioned in the text. While existing foundation models yield impressive results in many NLP tasks requiring reasoning, it is unclear whether they understand the complexity of the underlying network of causal relationships of events in narratives. In this work, we present CRAB, a new Causal Reasoning Assessment Benchmark designed to evaluate causal understanding of events in real-world narratives. CRAB contains fine-grained, contextual causality annotations for ~2.7K pairs of real-world events that describe various newsworthy event timelines (e.g., the acquisition of Twitter by Elon Musk). Using CRAB, we measure the performance of several large language models, demonstrating that most systems achieve poor performance on the task. Motivated by classical causal principles, we also analyze the causal structures of groups of events in CRAB, and find that models perform worse on causal reasoning when events are derived from complex causal structures compared to simple linear causal chains. We make our dataset and code available to the research community.

{{</citation>}}


### (46/144) Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves (Yihe Deng et al., 2023)

{{<citation>}}

Yihe Deng, Weitong Zhang, Zixiang Chen, Quanquan Gu. (2023)  
**Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.04205v1)  

---


**ABSTRACT**  
Misunderstandings arise not only in interpersonal communication but also between humans and Large Language Models (LLMs). Such discrepancies can make LLMs interpret seemingly unambiguous questions in unexpected ways, yielding incorrect responses. While it is widely acknowledged that the quality of a prompt, such as a question, significantly impacts the quality of the response provided by LLMs, a systematic method for crafting questions that LLMs can better comprehend is still underdeveloped. In this paper, we present a method named `Rephrase and Respond' (RaR), which allows LLMs to rephrase and expand questions posed by humans and provide responses in a single prompt. This approach serves as a simple yet effective prompting method for improving performance. We also introduce a two-step variant of RaR, where a rephrasing LLM first rephrases the question and then passes the original and rephrased questions together to a different responding LLM. This facilitates the effective utilization of rephrased questions generated by one LLM with another. Our experiments demonstrate that our methods significantly improve the performance of different models across a wide range to tasks. We further provide a comprehensive comparison between RaR and the popular Chain-of-Thought (CoT) methods, both theoretically and empirically. We show that RaR is complementary to CoT and can be combined with CoT to achieve even better performance. Our work not only contributes to enhancing LLM performance efficiently and effectively but also sheds light on a fair evaluation of LLM capabilities. Data and codes are available at https://github.com/uclaml/Rephrase-and-Respond.

{{</citation>}}


### (47/144) SpaDeLeF: A Dataset for Hierarchical Classification of Lexical Functions for Collocations in Spanish (Yevhen Kostiuk et al., 2023)

{{<citation>}}

Yevhen Kostiuk, Grigori Sidorov, Olga Kolesnikova. (2023)  
**SpaDeLeF: A Dataset for Hierarchical Classification of Lexical Functions for Collocations in Spanish**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.04189v1)  

---


**ABSTRACT**  
In natural language processing (NLP), lexical function is a concept to unambiguously represent semantic and syntactic features of words and phrases in text first crafted in the Meaning-Text Theory. Hierarchical classification of lexical functions involves organizing these features into a tree-like hierarchy of categories or labels. This is a challenging task as it requires a good understanding of the context and the relationships among words and phrases in text. It also needs large amounts of labeled data to train language models effectively. In this paper, we present a dataset of most frequent Spanish verb-noun collocations and sentences where they occur, each collocation is assigned to one of 37 lexical functions defined as classes for a hierarchical classification task. Each class represents a relation between the noun and the verb in a collocation involving their semantic and syntactic features. We combine the classes in a tree-based structure, and introduce classification objectives for each level of the structure. The dataset was created by dependency tree parsing and matching of the phrases in Spanish news. We provide baselines and data splits for each objective.

{{</citation>}}


### (48/144) Enhancing LLM Intelligence with ARM-RAG: Auxiliary Rationale Memory for Retrieval Augmented Generation (Eric Melz, 2023)

{{<citation>}}

Eric Melz. (2023)  
**Enhancing LLM Intelligence with ARM-RAG: Auxiliary Rationale Memory for Retrieval Augmented Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.04177v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are smart but forgetful. Recent studies, (e.g., (Bubeck et al., 2023)) on modern LLMs have shown that they are capable of performing amazing tasks typically necessitating human-level intelligence. However, unlike humans, frozen LLMs do not improve over time; they neither acquire new knowledge nor learn from their successes or failures. Some approaches to improving the intelligence of LLMs include fine-tuning models based on problem-solving performance (Zelikman et al., 2022), and building bigger and more sophisticated models (Bubeck et al., 2023). However, these methods have the drawback of requiring substantial data and computational resources to retrain existing models. In this paper, we explore the use of Retrieval Augmented Generation, also known as RAG (Lewis et al., 2021) to improve problem-solving performance. We propose ARM-RAG (Auxiliary Rationale Memory for Retrieval Augmented Generation), a system that learns from its successes without incurring high training costs. We demonstrate that the storage and subsequent retrieval of reasoning chains have a positive influence on performance in grade-school math problems.

{{</citation>}}


### (49/144) Perturbed examples reveal invariances shared by language models (Ruchit Rawal et al., 2023)

{{<citation>}}

Ruchit Rawal, Mariya Toneva. (2023)  
**Perturbed examples reveal invariances shared by language models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.04166v1)  

---


**ABSTRACT**  
An explosion of work in language is leading to ever-increasing numbers of available natural language processing models, with little understanding of how new models compare to better-understood models. One major reason for this difficulty is saturating benchmark datasets, which may not reflect well differences in model performance in the wild. In this work, we propose a novel framework for comparing two natural language processing models by revealing their shared invariance to interpretable input perturbations that are designed to target a specific linguistic capability (e.g., Synonym-Invariance, Typo-Invariance). Via experiments on models from within the same and across different architecture families, this framework offers a number of insights about how changes in models (e.g., distillation, increase in size, amount of pre-training) affect multiple well-defined linguistic capabilities. Furthermore, we also demonstrate how our framework can enable evaluation of the invariances shared between models that are available as commercial black-box APIs (e.g., InstructGPT family) and models that are relatively better understood (e.g., GPT-2). Across several experiments, we observe that large language models share many of the invariances encoded by models of various sizes, whereas the invariances encoded by large language models are only shared by other large models. Possessing a wide variety of invariances may be a key reason for the recent successes of large language models, and our framework can shed light on the types of invariances that are retained by or emerge in new models.

{{</citation>}}


### (50/144) Black-Box Prompt Optimization: Aligning Large Language Models without Model Training (Jiale Cheng et al., 2023)

{{<citation>}}

Jiale Cheng, Xiao Liu, Kehan Zheng, Pei Ke, Hongning Wang, Yuxiao Dong, Jie Tang, Minlie Huang. (2023)  
**Black-Box Prompt Optimization: Aligning Large Language Models without Model Training**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.04155v2)  

---


**ABSTRACT**  
Large language models (LLMs) have shown impressive success in various applications. However, these models are often not well aligned with human intents, which calls for additional treatments on them, that is, the alignment problem. To make LLMs better follow user instructions, existing alignment methods mostly focus on further training them. However, the extra training of LLMs are usually expensive in terms of GPU compute; worse still, LLMs of interest are oftentimes not accessible for user-demanded training, such as GPTs. In this work, we take a different perspective -- Black-Box Prompt Optimization (BPO) -- to perform alignments. The idea is to optimize user prompts to suit LLMs' input understanding, so as to best realize users' intents without updating LLMs' parameters. BPO is model-agnostic and the empirical results demonstrate that the BPO-aligned ChatGPT yields a 22% increase in the win rate against its original version, and 10% for GPT-4. Importantly, the BPO-aligned LLMs can outperform the same models aligned by PPO and DPO, and it also brings additional performance gains when combining BPO with PPO or DPO. Code and datasets are released at https://github.com/thu-coai/BPO.

{{</citation>}}


### (51/144) What is Lost in Knowledge Distillation? (Manas Mohanty et al., 2023)

{{<citation>}}

Manas Mohanty, Tanya Roosta, Peyman Passban. (2023)  
**What is Lost in Knowledge Distillation?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Distillation, NLP  
[Paper Link](http://arxiv.org/abs/2311.04142v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) have improved NLP tasks significantly, but training and maintaining such networks could be costly. Model compression techniques, such as, knowledge distillation (KD), have been proposed to address the issue; however, the compression process could be lossy. Motivated by this, our work investigates how a distilled student model differs from its teacher, if the distillation process causes any information losses, and if the loss follows a specific pattern. Our experiments aim to shed light on the type of tasks might be less or more sensitive to KD by reporting data points on the contribution of different factors, such as the number of layers or attention heads. Results such as ours could be utilized when determining effective and efficient configurations to achieve optimal information transfers between larger (teacher) and smaller (student) models.

{{</citation>}}


### (52/144) Modelling Sentiment Analysis: LLMs and data augmentation techniques (Guillem Senabre Prades, 2023)

{{<citation>}}

Guillem Senabre Prades. (2023)  
**Modelling Sentiment Analysis: LLMs and data augmentation techniques**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2311.04139v1)  

---


**ABSTRACT**  
This paper provides different approaches for a binary sentiment classification on a small training dataset. LLMs that provided state-of-the-art results in sentiment analysis and similar domains are being used, such as BERT, RoBERTa and XLNet.

{{</citation>}}


### (53/144) Locating Cross-Task Sequence Continuation Circuits in Transformers (Michael Lan et al., 2023)

{{<citation>}}

Michael Lan, Fazl Barez. (2023)  
**Locating Cross-Task Sequence Continuation Circuits in Transformers**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.04131v1)  

---


**ABSTRACT**  
While transformer models exhibit strong capabilities on linguistic tasks, their complex architectures make them difficult to interpret. Recent work has aimed to reverse engineer transformer models into human-readable representations called circuits that implement algorithmic functions. We extend this research by analyzing and comparing circuits for similar sequence continuation tasks, which include increasing sequences of digits, number words, and months. Through the application of circuit analysis techniques, we identify key sub-circuits responsible for detecting sequence members and for predicting the next member in a sequence. Our analysis reveals that semantically related sequences rely on shared circuit subgraphs with analogous roles. Overall, documenting shared computational structures enables better prediction of model behaviors, identification of errors, and safer editing procedures. This mechanistic understanding of transformers is a critical step towards building more robust, aligned, and interpretable language models.

{{</citation>}}


### (54/144) Unveiling Safety Vulnerabilities of Large Language Models (George Kour et al., 2023)

{{<citation>}}

George Kour, Marcel Zalmanovici, Naama Zwerdling, Esther Goldbraich, Ora Nova Fandina, Ateret Anaby-Tavor, Orna Raz, Eitan Farchi. (2023)  
**Unveiling Safety Vulnerabilities of Large Language Models**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.04124v1)  

---


**ABSTRACT**  
As large language models become more prevalent, their possible harmful or inappropriate responses are a cause for concern. This paper introduces a unique dataset containing adversarial examples in the form of questions, which we call AttaQ, designed to provoke such harmful or inappropriate responses. We assess the efficacy of our dataset by analyzing the vulnerabilities of various models when subjected to it. Additionally, we introduce a novel automatic approach for identifying and naming vulnerable semantic regions - input semantic areas for which the model is likely to produce harmful outputs. This is achieved through the application of specialized clustering techniques that consider both the semantic similarity of the input attacks and the harmfulness of the model's responses. Automatically identifying vulnerable semantic regions enhances the evaluation of model weaknesses, facilitating targeted improvements to its safety mechanisms and overall reliability.

{{</citation>}}


### (55/144) Personality Style Recognition via Machine Learning: Identifying Anaclitic and Introjective Personality Styles from Patients' Speech (Semere Kiros Bitew et al., 2023)

{{<citation>}}

Semere Kiros Bitew, Vincent Schelstraete, Klim Zaporojets, Kimberly Van Nieuwenhove, Reitske Meganck, Chris Develder. (2023)  
**Personality Style Recognition via Machine Learning: Identifying Anaclitic and Introjective Personality Styles from Patients' Speech**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NLP  
[Paper Link](http://arxiv.org/abs/2311.04088v1)  

---


**ABSTRACT**  
In disentangling the heterogeneity observed in psychopathology, personality of the patients is considered crucial. While it has been demonstrated that personality traits are reflected in the language used by a patient, we hypothesize that this enables automatic inference of the personality type directly from speech utterances, potentially more accurately than through a traditional questionnaire-based approach explicitly designed for personality classification. To validate this hypothesis, we adopt natural language processing (NLP) and standard machine learning tools for classification. We test this on a dataset of recorded clinical diagnostic interviews (CDI) on a sample of 79 patients diagnosed with major depressive disorder (MDD) -- a condition for which differentiated treatment based on personality styles has been advocated -- and classified into anaclitic and introjective personality styles. We start by analyzing the interviews to see which linguistic features are associated with each style, in order to gain a better understanding of the styles. Then, we develop automatic classifiers based on (a) standardized questionnaire responses; (b) basic text features, i.e., TF-IDF scores of words and word sequences; (c) more advanced text features, using LIWC (linguistic inquiry and word count) and context-aware features using BERT (bidirectional encoder representations from transformers); (d) audio features. We find that automated classification with language-derived features (i.e., based on LIWC) significantly outperforms questionnaire-based classification models. Furthermore, the best performance is achieved by combining LIWC with the questionnaire features. This suggests that more work should be put into developing linguistically based automated techniques for characterizing personality, however questionnaires still to some extent complement such methods.

{{</citation>}}


### (56/144) Do LLMs exhibit human-like response biases? A case study in survey design (Lindia Tjuatja et al., 2023)

{{<citation>}}

Lindia Tjuatja, Valerie Chen, Sherry Tongshuang Wu, Ameet Talwalkar, Graham Neubig. (2023)  
**Do LLMs exhibit human-like response biases? A case study in survey design**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.04076v1)  

---


**ABSTRACT**  
As large language models (LLMs) become more capable, there is growing excitement about the possibility of using LLMs as proxies for humans in real-world tasks where subjective labels are desired, such as in surveys and opinion polling. One widely-cited barrier to the adoption of LLMs is their sensitivity to prompt wording -- but interestingly, humans also display sensitivities to instruction changes in the form of response biases. As such, we argue that if LLMs are going to be used to approximate human opinions, it is necessary to investigate the extent to which LLMs also reflect human response biases, if at all. In this work, we use survey design as a case study, where human response biases caused by permutations in wordings of ``prompts'' have been extensively studied. Drawing from prior work in social psychology, we design a dataset and propose a framework to evaluate whether LLMs exhibit human-like response biases in survey questionnaires. Our comprehensive evaluation of nine models shows that popular open and commercial LLMs generally fail to reflect human-like behavior. These inconsistencies tend to be more prominent in models that have been instruction fine-tuned. Furthermore, even if a model shows a significant change in the same direction as humans, we find that perturbations that are not meant to elicit significant changes in humans may also result in a similar change, suggesting that such a result could be partially due to other spurious correlations. These results highlight the potential pitfalls of using LLMs to substitute humans in parts of the annotation pipeline, and further underscore the importance of finer-grained characterizations of model behavior. Our code, dataset, and collected samples are available at https://github.com/lindiatjuatja/BiasMonkey

{{</citation>}}


### (57/144) Implementation and Comparison of Methods to Extract Reliability KPIs out of Textual Wind Turbine Maintenance Work Orders (Marc-Alexander Lutz et al., 2023)

{{<citation>}}

Marc-Alexander Lutz, Bastian Schäfermeier, Rachael Sexton, Michael Sharp, Alden Dima, Stefan Faulstich, Jagan Mohini Aluri. (2023)  
**Implementation and Comparison of Methods to Extract Reliability KPIs out of Textual Wind Turbine Maintenance Work Orders**  

---
Primary Category: cs.CL  
Categories: I-2-7; I-7, cs-CL, cs-LG, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.04064v1)  

---


**ABSTRACT**  
Maintenance work orders are commonly used to document information about wind turbine operation and maintenance. This includes details about proactive and reactive wind turbine downtimes, such as preventative and corrective maintenance. However, the information contained in maintenance work orders is often unstructured and difficult to analyze, making it challenging for decision-makers to use this information for optimizing operation and maintenance. To address this issue, this work presents three different approaches to calculate reliability key performance indicators from maintenance work orders. The first approach involves manual labeling of the maintenance work orders by domain experts, using the schema defined in an industrial guideline to assign the label accordingly. The second approach involves the development of a model that automatically labels the maintenance work orders using text classification methods. The third technique uses an AI-assisted tagging tool to tag and structure the raw maintenance information contained in the maintenance work orders. The resulting calculated reliability key performance indicator of the first approach are used as a benchmark for comparison with the results of the second and third approaches. The quality and time spent are considered as criteria for evaluation. Overall, these three methods make extracting maintenance information from maintenance work orders more efficient, enable the assessment of reliability key performance indicators and therefore support the optimization of wind turbine operation and maintenance.

{{</citation>}}


### (58/144) P-Bench: A Multi-level Privacy Evaluation Benchmark for Language Models (Haoran Li et al., 2023)

{{<citation>}}

Haoran Li, Dadi Guo, Donghao Li, Wei Fan, Qi Hu, Xin Liu, Chunkit Chan, Duanyi Yao, Yangqiu Song. (2023)  
**P-Bench: A Multi-level Privacy Evaluation Benchmark for Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs.CL  
Keywords: GLUE, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.04044v1)  

---


**ABSTRACT**  
The rapid development of language models (LMs) brings unprecedented accessibility and usage for both models and users. On the one hand, powerful LMs, trained with massive textual data, achieve state-of-the-art performance over numerous downstream NLP tasks. On the other hand, more and more attention is paid to unrestricted model accesses that may bring malicious privacy risks of data leakage. To address these issues, many recent works propose privacy-preserving language models (PPLMs) with differential privacy (DP). Unfortunately, different DP implementations make it challenging for a fair comparison among existing PPLMs. In this paper, we present P-Bench, a multi-perspective privacy evaluation benchmark to empirically and intuitively quantify the privacy leakage of LMs. Instead of only protecting and measuring the privacy of protected data with DP parameters, P-Bench sheds light on the neglected inference data privacy during actual usage. P-Bench first clearly defines multi-faceted privacy objectives during private fine-tuning. Then, P-Bench constructs a unified pipeline to perform private fine-tuning. Lastly, P-Bench performs existing privacy attacks on LMs with pre-defined privacy objectives as the empirical evaluation results. The empirical attack results are used to fairly and intuitively evaluate the privacy leakage of various PPLMs. We conduct extensive experiments on three datasets of GLUE for mainstream LMs.

{{</citation>}}


### (59/144) mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration (Qinghao Ye et al., 2023)

{{<citation>}}

Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, Jingren Zhou. (2023)  
**mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.04257v1)  

---


**ABSTRACT**  
Multi-modal Large Language Models (MLLMs) have demonstrated impressive instruction abilities across various open-ended tasks. However, previous methods primarily focus on enhancing multi-modal capabilities. In this work, we introduce a versatile multi-modal large language model, mPLUG-Owl2, which effectively leverages modality collaboration to improve performance in both text and multi-modal tasks. mPLUG-Owl2 utilizes a modularized network design, with the language decoder acting as a universal interface for managing different modalities. Specifically, mPLUG-Owl2 incorporates shared functional modules to facilitate modality collaboration and introduces a modality-adaptive module that preserves modality-specific features. Extensive experiments reveal that mPLUG-Owl2 is capable of generalizing both text tasks and multi-modal tasks and achieving state-of-the-art performances with a single generic model. Notably, mPLUG-Owl2 is the first MLLM model that demonstrates the modality collaboration phenomenon in both pure-text and multi-modal scenarios, setting a pioneering path in the development of future multi-modal foundation models.

{{</citation>}}


### (60/144) Analyzing Film Adaptation through Narrative Alignment (Tanzir Pial et al., 2023)

{{<citation>}}

Tanzir Pial, Shahreen Salim, Charuta Pethe, Allen Kim, Steven Skiena. (2023)  
**Analyzing Film Adaptation through Narrative Alignment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2311.04020v1)  

---


**ABSTRACT**  
Novels are often adapted into feature films, but the differences between the two media usually require dropping sections of the source text from the movie script. Here we study this screen adaptation process by constructing narrative alignments using the Smith-Waterman local alignment algorithm coupled with SBERT embedding distance to quantify text similarity between scenes and book units. We use these alignments to perform an automated analysis of 40 adaptations, revealing insights into the screenwriting process concerning (i) faithfulness of adaptation, (ii) importance of dialog, (iii) preservation of narrative order, and (iv) gender representation issues reflective of the Bechdel test.

{{</citation>}}


### (61/144) Factoring Hate Speech: A New Annotation Framework to Study Hate Speech in Social Media (Gal Ron et al., 2023)

{{<citation>}}

Gal Ron, Effi Levi, Odelia Oshri, Shaul R. Shenhav. (2023)  
**Factoring Hate Speech: A New Annotation Framework to Study Hate Speech in Social Media**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2311.03969v1)  

---


**ABSTRACT**  
In this work we propose a novel annotation scheme which factors hate speech into five separate discursive categories. To evaluate our scheme, we construct a corpus of over 2.9M Twitter posts containing hateful expressions directed at Jews, and annotate a sample dataset of 1,050 tweets. We present a statistical analysis of the annotated dataset as well as discuss annotation examples, and conclude by discussing promising directions for future work.

{{</citation>}}


### (62/144) An Analysis of Dialogue Repair in Voice Assistants (Matthew Galbraith, 2023)

{{<citation>}}

Matthew Galbraith. (2023)  
**An Analysis of Dialogue Repair in Voice Assistants**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs-RO, cs.CL  
Keywords: Dialog, Dialogue, Google  
[Paper Link](http://arxiv.org/abs/2311.03952v1)  

---


**ABSTRACT**  
Spoken dialogue systems have transformed human-machine interaction by providing real-time responses to queries. However, misunderstandings between the user and system persist. This study explores the significance of interactional language in dialogue repair between virtual assistants and users by analyzing interactions with Google Assistant and Siri, focusing on their utilization and response to the other-initiated repair strategy "huh?" prevalent in human-human interaction. Findings reveal several assistant-generated strategies but an inability to replicate human-like repair strategies such as "huh?". English and Spanish user acceptability surveys show differences in users' repair strategy preferences and assistant usage, with both similarities and disparities among the two surveyed languages. These results shed light on inequalities between interactional language in human-human interaction and human-machine interaction, underscoring the need for further research on the impact of interactional language in human-machine interaction in English and beyond.

{{</citation>}}


### (63/144) Improving Korean NLP Tasks with Linguistically Informed Subword Tokenization and Sub-character Decomposition (Taehee Jeon et al., 2023)

{{<citation>}}

Taehee Jeon, Bongseok Yang, Changhwan Kim, Yoonseob Lim. (2023)  
**Improving Korean NLP Tasks with Linguistically Informed Subword Tokenization and Sub-character Decomposition**  

---
Primary Category: cs.CL  
Categories: 68T50, cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.03928v1)  

---


**ABSTRACT**  
We introduce a morpheme-aware subword tokenization method that utilizes sub-character decomposition to address the challenges of applying Byte Pair Encoding (BPE) to Korean, a language characterized by its rich morphology and unique writing system. Our approach balances linguistic accuracy with computational efficiency in Pre-trained Language Models (PLMs). Our evaluations show that this technique achieves good performances overall, notably improving results in the syntactic task of NIKL-CoLA. This suggests that integrating morpheme type information can enhance language models' syntactic and semantic capabilities, indicating that adopting more linguistic insights can further improve performance beyond standard morphological analysis.

{{</citation>}}


### (64/144) Sparse Contrastive Learning of Sentence Embeddings (Ruize An et al., 2023)

{{<citation>}}

Ruize An, Chen Zhang, Dawei Song. (2023)  
**Sparse Contrastive Learning of Sentence Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning, Embedding, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2311.03881v1)  

---


**ABSTRACT**  
Recently, SimCSE has shown the feasibility of contrastive learning in training sentence embeddings and illustrates its expressiveness in spanning an aligned and uniform embedding space. However, prior studies have shown that dense models could contain harmful parameters that affect the model performance, and it is no wonder that SimCSE can as well be invented with such parameters. Driven by this, parameter sparsification is applied, where alignment and uniformity scores are used to measure the contribution of each parameter to the overall quality of sentence embeddings. Drawing from a preliminary study, we consider parameters with minimal contributions to be detrimental, as their sparsification results in improved model performance. To discuss the ubiquity of detrimental parameters and remove them, more experiments on the standard semantic textual similarity (STS) tasks and transfer learning tasks are conducted, and the results show that the proposed sparsified SimCSE (SparseCSE) has excellent performance in comparison with SimCSE. Furthermore, through in-depth analysis, we establish the validity and stability of our sparsification method, showcasing that the embedding space generated by SparseCSE exhibits improved alignment compared to that produced by SimCSE. Importantly, the uniformity yet remains uncompromised.

{{</citation>}}


### (65/144) Aspects of human memory and Large Language Models (Romuald A. Janik, 2023)

{{<citation>}}

Romuald A. Janik. (2023)  
**Aspects of human memory and Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL, q-bio-NC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.03839v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are huge artificial neural networks which primarily serve to generate text, but also provide a very sophisticated probabilistic model of language use. Since generating a semantically consistent text requires a form of effective memory, we investigate the memory properties of LLMs and find surprising similarities with key characteristics of human memory. This result strongly suggests that the biological features of human memory leave an imprint on the way that we structure our textual narratives.

{{</citation>}}


### (66/144) Conversations in Galician: a Large Language Model for an Underrepresented Language (Eliseo Bao et al., 2023)

{{<citation>}}

Eliseo Bao, Anxo Pérez, Javier Parapar. (2023)  
**Conversations in Galician: a Large Language Model for an Underrepresented Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, LLaMA, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.03812v1)  

---


**ABSTRACT**  
The recent proliferation of Large Conversation Language Models has highlighted the economic significance of widespread access to this type of AI technologies in the current information age. Nevertheless, prevailing models have primarily been trained on corpora consisting of documents written in popular languages. The dearth of such cutting-edge tools for low-resource languages further exacerbates their underrepresentation in the current economic landscape, thereby impacting their native speakers. This paper introduces two novel resources designed to enhance Natural Language Processing (NLP) for the Galician language. We present a Galician adaptation of the Alpaca dataset, comprising 52,000 instructions and demonstrations. This dataset proves invaluable for enhancing language models by fine-tuning them to more accurately adhere to provided instructions. Additionally, as a demonstration of the dataset utility, we fine-tuned LLaMA-7B to comprehend and respond in Galician, a language not originally supported by the model, by following the Alpaca format. This work contributes to the research on multilingual models tailored for low-resource settings, a crucial endeavor in ensuring the inclusion of all linguistic communities in the development of Large Language Models. Another noteworthy aspect of this research is the exploration of how knowledge of a closely related language, in this case, Portuguese, can assist in generating coherent text when training resources are scarce. Both the Galician Alpaca dataset and Cabuxa-7B are publicly accessible on our Huggingface Hub, and we have made the source code available to facilitate replication of this experiment and encourage further advancements for underrepresented languages.

{{</citation>}}


### (67/144) Noisy Pair Corrector for Dense Retrieval (Hang Zhang et al., 2023)

{{<citation>}}

Hang Zhang, Yeyun Gong, Xingwei He, Dayiheng Liu, Daya Guo, Jiancheng Lv, Jian Guo. (2023)  
**Noisy Pair Corrector for Dense Retrieval**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.03798v1)  

---


**ABSTRACT**  
Most dense retrieval models contain an implicit assumption: the training query-document pairs are exactly matched. Since it is expensive to annotate the corpus manually, training pairs in real-world applications are usually collected automatically, which inevitably introduces mismatched-pair noise. In this paper, we explore an interesting and challenging problem in dense retrieval, how to train an effective model with mismatched-pair noise. To solve this problem, we propose a novel approach called Noisy Pair Corrector (NPC), which consists of a detection module and a correction module. The detection module estimates noise pairs by calculating the perplexity between annotated positive and easy negative documents. The correction module utilizes an exponential moving average (EMA) model to provide a soft supervised signal, aiding in mitigating the effects of noise. We conduct experiments on text-retrieval benchmarks Natural Question and TriviaQA, code-search benchmarks StaQC and SO-DS. Experimental results show that NPC achieves excellent performance in handling both synthetic and realistic noise.

{{</citation>}}


### (68/144) Character-Level Bangla Text-to-IPA Transcription Using Transformer Architecture with Sequence Alignment (Jakir Hasan et al., 2023)

{{<citation>}}

Jakir Hasan, Shrestha Datta, Ameya Debnath. (2023)  
**Character-Level Bangla Text-to-IPA Transcription Using Transformer Architecture with Sequence Alignment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.03792v1)  

---


**ABSTRACT**  
The International Phonetic Alphabet (IPA) is indispensable in language learning and understanding, aiding users in accurate pronunciation and comprehension. Additionally, it plays a pivotal role in speech therapy, linguistic research, accurate transliteration, and the development of text-to-speech systems, making it an essential tool across diverse fields. Bangla being 7th as one of the widely used languages, gives rise to the need for IPA in its domain. Its IPA mapping is too diverse to be captured manually giving the need for Artificial Intelligence and Machine Learning in this field. In this study, we have utilized a transformer-based sequence-to-sequence model at the letter and symbol level to get the IPA of each Bangla word as the variation of IPA in association of different words is almost null. Our transformer model only consisted of 8.5 million parameters with only a single decoder and encoder layer. Additionally, to handle the punctuation marks and the occurrence of foreign languages in the text, we have utilized manual mapping as the model won't be able to learn to separate them from Bangla words while decreasing our required computational resources. Finally, maintaining the relative position of the sentence component IPAs and generation of the combined IPA has led us to achieve the top position with a word error rate of 0.10582 in the public ranking of DataVerse Challenge - ITVerse 2023 (https://www.kaggle.com/competitions/dataverse_2023/).

{{</citation>}}


### (69/144) Language Representation Projection: Can We Transfer Factual Knowledge across Languages in Multilingual Language Models? (Shaoyang Xu et al., 2023)

{{<citation>}}

Shaoyang Xu, Junzhuo Li, Deyi Xiong. (2023)  
**Language Representation Projection: Can We Transfer Factual Knowledge across Languages in Multilingual Language Models?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2311.03788v1)  

---


**ABSTRACT**  
Multilingual pretrained language models serve as repositories of multilingual factual knowledge. Nevertheless, a substantial performance gap of factual knowledge probing exists between high-resource languages and low-resource languages, suggesting limited implicit factual knowledge transfer across languages in multilingual pretrained language models. This paper investigates the feasibility of explicitly transferring relatively rich factual knowledge from English to non-English languages. To accomplish this, we propose two parameter-free $\textbf{L}$anguage $\textbf{R}$epresentation $\textbf{P}$rojection modules (LRP2). The first module converts non-English representations into English-like equivalents, while the second module reverts English-like representations back into representations of the corresponding non-English language. Experimental results on the mLAMA dataset demonstrate that LRP2 significantly improves factual knowledge retrieval accuracy and facilitates knowledge transferability across diverse non-English languages. We further investigate the working mechanism of LRP2 from the perspectives of representation space and cross-lingual knowledge neuron.

{{</citation>}}


### (70/144) Ensembling Textual and Structure-Based Models for Knowledge Graph Completion (Ananjan Nandi et al., 2023)

{{<citation>}}

Ananjan Nandi, Navdeep Kaur, Parag Singla, Mausam. (2023)  
**Ensembling Textual and Structure-Based Models for Knowledge Graph Completion**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2311.03780v1)  

---


**ABSTRACT**  
We consider two popular approaches to Knowledge Graph Completion (KGC): textual models that rely on textual entity descriptions, and structure-based models that exploit the connectivity structure of the Knowledge Graph (KG). Preliminary experiments show that these approaches have complementary strengths: structure-based models perform well when the gold answer is easily reachable from the query head in the KG, while textual models exploit descriptions to give good performance even when the gold answer is not reachable. In response, we explore ensembling as a way of combining the best of both approaches. We propose a novel method for learning query-dependent ensemble weights by using the distributions of scores assigned by individual models to all candidate entities. Our ensemble baseline achieves state-of-the-art results on three standard KGC datasets, with up to 6.8 pt MRR and 8.3 pt Hits@1 gains over best individual models.

{{</citation>}}


### (71/144) Gender Inflected or Bias Inflicted: On Using Grammatical Gender Cues for Bias Evaluation in Machine Translation (Pushpdeep Singh, 2023)

{{<citation>}}

Pushpdeep Singh. (2023)  
**Gender Inflected or Bias Inflicted: On Using Grammatical Gender Cues for Bias Evaluation in Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.03767v1)  

---


**ABSTRACT**  
Neural Machine Translation (NMT) models are state-of-the-art for machine translation. However, these models are known to have various social biases, especially gender bias. Most of the work on evaluating gender bias in NMT has focused primarily on English as the source language. For source languages different from English, most of the studies use gender-neutral sentences to evaluate gender bias. However, practically, many sentences that we encounter do have gender information. Therefore, it makes more sense to evaluate for bias using such sentences. This allows us to determine if NMT models can identify the correct gender based on the grammatical gender cues in the source sentence rather than relying on biased correlations with, say, occupation terms. To demonstrate our point, in this work, we use Hindi as the source language and construct two sets of gender-specific sentences: OTSC-Hindi and WinoMT-Hindi that we use to evaluate different Hindi-English (HI-EN) NMT systems automatically for gender bias. Our work highlights the importance of considering the nature of language when designing such extrinsic bias evaluation datasets.

{{</citation>}}


### (72/144) Multilingual Mathematical Autoformalization (Albert Q. Jiang et al., 2023)

{{<citation>}}

Albert Q. Jiang, Wenda Li, Mateja Jamnik. (2023)  
**Multilingual Mathematical Autoformalization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2311.03755v1)  

---


**ABSTRACT**  
Autoformalization is the task of translating natural language materials into machine-verifiable formalisations. Progress in autoformalization research is hindered by the lack of a sizeable dataset consisting of informal-formal pairs expressing the same essence. Existing methods tend to circumvent this challenge by manually curating small corpora or using few-shot learning with large language models. But these methods suffer from data scarcity and formal language acquisition difficulty. In this work, we create $\texttt{MMA}$, a large, flexible, multilingual, and multi-domain dataset of informal-formal pairs, by using a language model to translate in the reverse direction, that is, from formal mathematical statements into corresponding informal ones. Experiments show that language models fine-tuned on $\texttt{MMA}$ produce $16-18\%$ of statements acceptable with minimal corrections on the $\texttt{miniF2F}$ and $\texttt{ProofNet}$ benchmarks, up from $0\%$ with the base model. We demonstrate that fine-tuning on multilingual formal data results in more capable autoformalization models even when deployed on monolingual tasks.

{{</citation>}}


### (73/144) Which is better? Exploring Prompting Strategy For LLM-based Metrics (Joonghoon Kim et al., 2023)

{{<citation>}}

Joonghoon Kim, Saeran Park, Kiyoon Jeong, Sangmin Lee, Seung Hun Han, Jiyoon Lee, Pilsung Kang. (2023)  
**Which is better? Exploring Prompting Strategy For LLM-based Metrics**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, GPT, GPT-4, Language Model, Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2311.03754v1)  

---


**ABSTRACT**  
This paper describes the DSBA submissions to the Prompting Large Language Models as Explainable Metrics shared task, where systems were submitted to two tracks: small and large summarization tracks. With advanced Large Language Models (LLMs) such as GPT-4, evaluating the quality of Natural Language Generation (NLG) has become increasingly paramount. Traditional similarity-based metrics such as BLEU and ROUGE have shown to misalign with human evaluation and are ill-suited for open-ended generation tasks. To address this issue, we explore the potential capability of LLM-based metrics, especially leveraging open-source LLMs. In this study, wide range of prompts and prompting techniques are systematically analyzed with three approaches: prompting strategy, score aggregation, and explainability. Our research focuses on formulating effective prompt templates, determining the granularity of NLG quality scores and assessing the impact of in-context examples on LLM-based evaluation. Furthermore, three aggregation strategies are compared to identify the most reliable method for aggregating NLG quality scores. To examine explainability, we devise a strategy that generates rationales for the scores and analyzes the characteristics of the explanation produced by the open-source LLMs. Extensive experiments provide insights regarding evaluation capabilities of open-source LLMs and suggest effective prompting strategies.

{{</citation>}}


### (74/144) Unified Low-Resource Sequence Labeling by Sample-Aware Dynamic Sparse Finetuning (Sarkar Snigdha Sarathi Das et al., 2023)

{{<citation>}}

Sarkar Snigdha Sarathi Das, Ranran Haoran Zhang, Peng Shi, Wenpeng Yin, Rui Zhang. (2023)  
**Unified Low-Resource Sequence Labeling by Sample-Aware Dynamic Sparse Finetuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Low-Resource, Named Entity Recognition, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2311.03748v1)  

---


**ABSTRACT**  
Unified Sequence Labeling that articulates different sequence labeling problems such as Named Entity Recognition, Relation Extraction, Semantic Role Labeling, etc. in a generalized sequence-to-sequence format opens up the opportunity to make the maximum utilization of large language model knowledge toward structured prediction. Unfortunately, this requires formatting them into specialized augmented format unknown to the base pretrained language model (PLMs) necessitating finetuning to the target format. This significantly bounds its usefulness in data-limited settings where finetuning large models cannot properly generalize to the target format. To address this challenge and leverage PLM knowledge effectively, we propose FISH-DIP, a sample-aware dynamic sparse finetuning strategy that selectively focuses on a fraction of parameters, informed by feedback from highly regressing examples, during the fine-tuning process. By leveraging the dynamism of sparsity, our approach mitigates the impact of well-learned samples and prioritizes underperforming instances for improvement in generalization. Across five tasks of sequence labeling, we demonstrate that FISH-DIP can smoothly optimize the model in low resource settings offering upto 40% performance improvements over full fine-tuning depending on target evaluation settings. Also, compared to in-context learning and other parameter-efficient fine-tuning approaches, FISH-DIP performs comparably or better, notably in extreme low-resource settings.

{{</citation>}}


### (75/144) Leveraging Structured Information for Explainable Multi-hop Question Answering and Reasoning (Ruosen Li et al., 2023)

{{<citation>}}

Ruosen Li, Xinya Du. (2023)  
**Leveraging Structured Information for Explainable Multi-hop Question Answering and Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.03734v1)  

---


**ABSTRACT**  
Neural models, including large language models (LLMs), achieve superior performance on multi-hop question-answering. To elicit reasoning capabilities from LLMs, recent works propose using the chain-of-thought (CoT) mechanism to generate both the reasoning chain and the answer, which enhances the model's capabilities in conducting multi-hop reasoning. However, several challenges still remain: such as struggling with inaccurate reasoning, hallucinations, and lack of interpretability. On the other hand, information extraction (IE) identifies entities, relations, and events grounded to the text. The extracted structured information can be easily interpreted by humans and machines (Grishman, 2019). In this work, we investigate constructing and leveraging extracted semantic structures (graphs) for multi-hop question answering, especially the reasoning process. Empirical results and human evaluations show that our framework: generates more faithful reasoning chains and substantially improves the QA performance on two benchmark datasets. Moreover, the extracted structures themselves naturally provide grounded explanations that are preferred by humans, as compared to the generated reasoning chains and saliency-based explanations.

{{</citation>}}


### (76/144) A Survey of Large Language Models Attribution (Dongfang Li et al., 2023)

{{<citation>}}

Dongfang Li, Zetian Sun, Xinshuo Hu, Zhenyu Liu, Ziyang Chen, Baotian Hu, Aiguo Wu, Min Zhang. (2023)  
**A Survey of Large Language Models Attribution**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.03731v1)  

---


**ABSTRACT**  
Open-domain generative systems have gained significant attention in the field of conversational AI (e.g., generative search engines). This paper presents a comprehensive review of the attribution mechanisms employed by these systems, particularly large language models. Though attribution or citation improve the factuality and verifiability, issues like ambiguous knowledge reservoirs, inherent biases, and the drawbacks of excessive attribution can hinder the effectiveness of these systems. The aim of this survey is to provide valuable insights for researchers, aiding in the refinement of attribution methodologies to enhance the reliability and veracity of responses generated by open-domain generative systems. We believe that this field is still in its early stages; hence, we maintain a repository to keep track of ongoing studies at https://github.com/HITsz-TMG/awesome-llm-attributions.

{{</citation>}}


### (77/144) LLM as an Art Director (LaDi): Using LLMs to improve Text-to-Media Generators (Allen Roush et al., 2023)

{{<citation>}}

Allen Roush, Emil Zakirov, Artemiy Shirokov, Polina Lunina, Jack Gane, Alexander Duffy, Charlie Basil, Aber Whitcomb, Jim Benedetto, Chris DeWolfe. (2023)  
**LLM as an Art Director (LaDi): Using LLMs to improve Text-to-Media Generators**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.03716v1)  

---


**ABSTRACT**  
Recent advancements in text-to-image generation have revolutionized numerous fields, including art and cinema, by automating the generation of high-quality, context-aware images and video. However, the utility of these technologies is often limited by the inadequacy of text prompts in guiding the generator to produce artistically coherent and subject-relevant images. In this paper, We describe the techniques that can be used to make Large Language Models (LLMs) act as Art Directors that enhance image and video generation. We describe our unified system for this called "LaDi". We explore how LaDi integrates multiple techniques for augmenting the capabilities of text-to-image generators (T2Is) and text-to-video generators (T2Vs), with a focus on constrained decoding, intelligent prompting, fine-tuning, and retrieval. LaDi and these techniques are being used today in apps and platforms developed by Plai Labs.

{{</citation>}}


### (78/144) Bilingual Corpus Mining and Multistage Fine-Tuning for Improving Machine Translation of Lecture Transcripts (Haiyue Song et al., 2023)

{{<citation>}}

Haiyue Song, Raj Dabre, Chenhui Chu, Atsushi Fujita, Sadao Kurohashi. (2023)  
**Bilingual Corpus Mining and Multistage Fine-Tuning for Improving Machine Translation of Lecture Transcripts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.03696v1)  

---


**ABSTRACT**  
Lecture transcript translation helps learners understand online courses, however, building a high-quality lecture machine translation system lacks publicly available parallel corpora. To address this, we examine a framework for parallel corpus mining, which provides a quick and effective way to mine a parallel corpus from publicly available lectures on Coursera. To create the parallel corpora, we propose a dynamic programming based sentence alignment algorithm which leverages the cosine similarity of machine-translated sentences. The sentence alignment F1 score reaches 96%, which is higher than using the BERTScore, LASER, or sentBERT methods. For both English--Japanese and English--Chinese lecture translations, we extracted parallel corpora of approximately 50,000 lines and created development and test sets through manual filtering for benchmarking translation performance. Through machine translation experiments, we show that the mined corpora enhance the quality of lecture transcript translation when used in conjunction with out-of-domain parallel corpora via multistage fine-tuning. Furthermore, this study also suggests guidelines for gathering and cleaning corpora, mining parallel sentences, cleaning noise in the mined data, and creating high-quality evaluation splits. For the sake of reproducibility, we have released the corpora as well as the code to create them. The dataset is available at https://github.com/shyyhs/CourseraParallelCorpusMining.

{{</citation>}}


### (79/144) CBSiMT: Mitigating Hallucination in Simultaneous Machine Translation with Weighted Prefix-to-Prefix Training (Mengge Liu et al., 2023)

{{<citation>}}

Mengge Liu, Wen Zhang, Xiang Li, Yanzhi Tian, Yuhang Guo, Jian Luan, Bin Wang, Shuoying Chen. (2023)  
**CBSiMT: Mitigating Hallucination in Simultaneous Machine Translation with Weighted Prefix-to-Prefix Training**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.03672v1)  

---


**ABSTRACT**  
Simultaneous machine translation (SiMT) is a challenging task that requires starting translation before the full source sentence is available. Prefix-to-prefix framework is often applied to SiMT, which learns to predict target tokens using only a partial source prefix. However, due to the word order difference between languages, misaligned prefix pairs would make SiMT models suffer from serious hallucination problems, i.e. target outputs that are unfaithful to source inputs. Such problems can not only produce target tokens that are not supported by the source prefix, but also hinder generating the correct translation by receiving more source words. In this work, we propose a Confidence-Based Simultaneous Machine Translation (CBSiMT) framework, which uses model confidence to perceive hallucination tokens and mitigates their negative impact with weighted prefix-to-prefix training. Specifically, token-level and sentence-level weights are calculated based on model confidence and acted on the loss function. We explicitly quantify the faithfulness of the generated target tokens using the token-level weight, and employ the sentence-level weight to alleviate the disturbance of sentence pairs with serious word order differences on the model. Experimental results on MuST-C English-to-Chinese and WMT15 German-to-English SiMT tasks demonstrate that our method can consistently improve translation quality at most latency regimes, with up to 2 BLEU scores improvement at low latency.

{{</citation>}}


### (80/144) Generalization of NLP Models: Notion and Causation (Aparna Elangovan et al., 2023)

{{<citation>}}

Aparna Elangovan, Jiayuan He, Yuan Li, Karin Verspoor. (2023)  
**Generalization of NLP Models: Notion and Causation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.03663v1)  

---


**ABSTRACT**  
The NLP community typically relies on performance of a model on a held-out test set to assess generalization. Performance drops observed in datasets outside of official test sets are generally attributed to "out-of-distribution'' effects. Here, we explore the foundations of generalizability and study the various factors that affect it, articulating generalizability lessons from clinical studies. In clinical research generalizability depends on (a) internal validity of experiments to ensure controlled measurement of cause and effect, and (b) external validity or transportability of the results to the wider population. We present the need to ensure internal validity when building machine learning models in natural language processing, especially where results may be impacted by spurious correlations in the data. We demonstrate how spurious factors, such as the distance between entities in relation extraction tasks, can affect model internal validity and in turn adversely impact generalization. We also offer guidance on how to analyze generalization failures.

{{</citation>}}


### (81/144) The Linear Representation Hypothesis and the Geometry of Large Language Models (Kiho Park et al., 2023)

{{<citation>}}

Kiho Park, Yo Joong Choe, Victor Veitch. (2023)  
**The Linear Representation Hypothesis and the Geometry of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL, stat-ML  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2311.03658v1)  

---


**ABSTRACT**  
Informally, the 'linear representation hypothesis' is the idea that high-level concepts are represented linearly as directions in some representation space. In this paper, we address two closely related questions: What does "linear representation" actually mean? And, how do we make sense of geometric notions (e.g., cosine similarity or projection) in the representation space? To answer these, we use the language of counterfactuals to give two formalizations of "linear representation", one in the output (word) representation space, and one in the input (sentence) space. We then prove these connect to linear probing and model steering, respectively. To make sense of geometric notions, we use the formalization to identify a particular (non-Euclidean) inner product that respects language structure in a sense we make precise. Using this causal inner product, we show how to unify all notions of linear representation. In particular, this allows the construction of probes and steering vectors using counterfactual pairs. Experiments with LLaMA-2 demonstrate the existence of linear representations of concepts, the connection to interpretation and control, and the fundamental role of the choice of inner product.

{{</citation>}}


### (82/144) GNAT: A General Narrative Alignment Tool (Tanzir Pial et al., 2023)

{{<citation>}}

Tanzir Pial, Steven Skiena. (2023)  
**GNAT: A General Narrative Alignment Tool**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.03627v1)  

---


**ABSTRACT**  
Algorithmic sequence alignment identifies similar segments shared between pairs of documents, and is fundamental to many NLP tasks. But it is difficult to recognize similarities between distant versions of narratives such as translations and retellings, particularly for summaries and abridgements which are much shorter than the original novels.   We develop a general approach to narrative alignment coupling the Smith-Waterman algorithm from bioinformatics with modern text similarity metrics. We show that the background of alignment scores fits a Gumbel distribution, enabling us to define rigorous p-values on the significance of any alignment. We apply and evaluate our general narrative alignment tool (GNAT) on four distinct problem domains differing greatly in both the relative and absolute length of documents, namely summary-to-book alignment, translated book alignment, short story alignment, and plagiarism detection -- demonstrating the power and performance of our methods.

{{</citation>}}


## cs.CY (2)



### (83/144) Educating for AI Cybersecurity Work and Research: Ethics, Systems Thinking, and Communication Requirements (Sorin Adam Matei et al., 2023)

{{<citation>}}

Sorin Adam Matei, Elisa Bertino. (2023)  
**Educating for AI Cybersecurity Work and Research: Ethics, Systems Thinking, and Communication Requirements**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CR, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.04326v1)  

---


**ABSTRACT**  
The present study explored managerial and instructor perceptions of their freshly employed cybersecurity workers' or students' preparedness to work effectively in a changing cybersecurity environment that includes AI tools. Specifically, we related perceptions of technical preparedness to ethical, systems thinking, and communication skills. We found that managers and professors perceive preparedness to use AI tools in cybersecurity to be significantly associated with all three non-technical skill sets. Most important, ethics is a clear leader in the network of relationships. Contrary to expectations that ethical concerns are left behind in the rush to adopt the most advanced AI tools in security, both higher education instructors and managers appreciate their role and see them closely associated with technical prowess. Another significant finding is that professors over-estimate students' preparedness for ethical, system thinking, and communication abilities compared to IT managers' perceptions of their newly employed IT workers.

{{</citation>}}


### (84/144) Healthcare Security Breaches in the United States: Insights and their Socio-Technical Implications (Megha M. Moncy et al., 2023)

{{<citation>}}

Megha M. Moncy, Sadia Afreen, Saptarshi Purkayastha. (2023)  
**Healthcare Security Breaches in the United States: Insights and their Socio-Technical Implications**  

---
Primary Category: cs.CY  
Categories: cs-CR, cs-CY, cs.CY  
Keywords: Clinical, Security  
[Paper Link](http://arxiv.org/abs/2311.03664v1)  

---


**ABSTRACT**  
This research examines the pivotal role of human behavior in the realm of healthcare data management, situated at the confluence of technological advancements and human conduct. An in-depth analysis of security breaches in the United States from 2009 to the present elucidates the dominance of human-induced security breaches. While technological weak points are certainly a concern, our study highlights that a significant proportion of breaches are precipitated by human errors and practices, thus pinpointing a conspicuous deficiency in training, awareness, and organizational architecture. In spite of stringent federal mandates, such as the Health Insurance Portability and Accountability Act (HIPAA) and the Health Information Technology for Economic and Clinical Health (HITECH) Act, breaches persist, emphasizing the indispensable role of human factors within this domain. Such oversights not only jeopardize patient data confidentiality but also undermine the foundational trust inherent in the healthcare infrastructure. By probing the socio-technical facets of healthcare security infringements, this article advocates for an integrated, dynamic, and holistic approach to healthcare data security. The findings underscore the imperative of augmenting technological defenses while concurrently elevating human conduct and institutional ethos, thereby cultivating a robust and impervious healthcare data management environment.

{{</citation>}}


## cs.HC (4)



### (85/144) KNIMEZoBot: Enhancing Literature Review with Zotero and KNIME OpenAI Integration using Retrieval-Augmented Generation (Suad Alshammari et al., 2023)

{{<citation>}}

Suad Alshammari, Lama Basalelah, Walaa Abu Rukbah, Ali Alsuhibani, Dayanjan S. Wijesinghe. (2023)  
**KNIMEZoBot: Enhancing Literature Review with Zotero and KNIME OpenAI Integration using Retrieval-Augmented Generation**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.04310v1)  

---


**ABSTRACT**  
Academic researchers face challenges keeping up with exponentially growing published findings in their field. Performing comprehensive literature reviews to synthesize knowledge is time-consuming and labor-intensive using manual approaches. Recent advances in artificial intelligence provide promising solutions, yet many require coding expertise, limiting accessibility. KNIMEZoBot represents an innovative integration of Zotero, OpenAI, and the KNIME visual programming platform to automate literature review tasks for users with no coding experience. By leveraging KNIME's intuitive graphical interface, researchers can create workflows to search their Zotero libraries and utilize OpenAI models to extract key information without coding. Users simply provide API keys and configure settings through a user-friendly interface in a locally stored copy of the workflow. KNIMEZoBot then allows asking natural language questions via a chatbot and retrieves relevant passages from papers to generate synthesized answers. This system has significant potential to expedite literature reviews for researchers unfamiliar with coding by automating retrieval and analysis of publications in personal Zotero libraries. KNIMEZoBot demonstrates how thoughtfully designed AI tools can expand accessibility and accelerate knowledge building across diverse research domains.

{{</citation>}}


### (86/144) Human-AI Collaboration in Thematic Analysis using ChatGPT: A User Study and Design Recommendations (Lixiang Yan et al., 2023)

{{<citation>}}

Lixiang Yan, Vanessa Echeverria, Gloria Fernandez Nieto, Yueqiao Jin, Zachari Swiecki, Linxuan Zhao, Dragan Gašević, Roberto Martinez-Maldonado. (2023)  
**Human-AI Collaboration in Thematic Analysis using ChatGPT: A User Study and Design Recommendations**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.03999v1)  

---


**ABSTRACT**  
Generative artificial intelligence (GenAI) offers promising potential for advancing human-AI collaboration in qualitative research. However, existing works focused on conventional machine-learning and pattern-based AI systems, and little is known about how researchers interact with GenAI in qualitative research. This work delves into researchers' perceptions of their collaboration with GenAI, specifically ChatGPT. Through a user study involving ten qualitative researchers, we found ChatGPT to be a valuable collaborator for thematic analysis, enhancing coding efficiency, aiding initial data exploration, offering granular quantitative insights, and assisting comprehension for non-native speakers and non-experts. Yet, concerns about its trustworthiness and accuracy, reliability and consistency, limited contextual understanding, and broader acceptance within the research community persist. We contribute five actionable design recommendations to foster effective human-AI collaboration. These include incorporating transparent explanatory mechanisms, enhancing interface and integration capabilities, prioritising contextual understanding and customisation, embedding human-AI feedback loops and iterative functionality, and strengthening trust through validation mechanisms.

{{</citation>}}


### (87/144) 6DVF: Data Visualisation Framework for mHealth Apps (Yasmeen Anjeer Alshehhi et al., 2023)

{{<citation>}}

Yasmeen Anjeer Alshehhi, Khlood Ahmad, Mohamed Abdelrazek, Alessio Bonti. (2023)  
**6DVF: Data Visualisation Framework for mHealth Apps**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2311.03657v1)  

---


**ABSTRACT**  
The widespread of data visualisation tools on smartphones has provided end users an easy way to track their health data, leading designers to put more effort into delivering suitable visualisations. Both academia and industry have developed several frameworks to guide the creation of informative and well-designed charts, such as the visualisation and design framework and Google Material Design.   Despite the typical focus on design and chart types in these existing frameworks, our study highlights the need to incorporate additional components when developing data visualisations. The needs of non-expert users, the nature of the data being represented, and the mobile environment are often not prioritised in these frameworks, leading to visualisations that do not meet user needs and expectations. To address these issues, we propose our Six-Dimensions Data Visualisation Framework (6DVF) to assist in the design and evaluation of visualisations on mobile devices. Finally, we present our initial findings from a designer evaluation experiment.

{{</citation>}}


### (88/144) Analysis of the User Perception of Chatbots in Education Using A Partial Least Squares Structural Equation Modeling Approach (Md Rabiul Hasan et al., 2023)

{{<citation>}}

Md Rabiul Hasan, Nahian Ismail Chowdhury, Md Hadisur Rahman, Md Asif Bin Syed, JuHyeong Ryu. (2023)  
**Analysis of the User Perception of Chatbots in Education Using A Partial Least Squares Structural Equation Modeling Approach**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-LG, cs.HC  
Keywords: AI, ChatGPT, GPT, Google, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2311.03636v1)  

---


**ABSTRACT**  
The integration of Artificial Intelligence (AI) into education is a recent development, with chatbots emerging as a noteworthy addition to this transformative landscape. As online learning platforms rapidly advance, students need to adapt swiftly to excel in this dynamic environment. Consequently, understanding the acceptance of chatbots, particularly those employing Large Language Model (LLM) such as Chat Generative Pretrained Transformer (ChatGPT), Google Bard, and other interactive AI technologies, is of paramount importance. However, existing research on chatbots in education has overlooked key behavior-related aspects, such as Optimism, Innovativeness, Discomfort, Insecurity, Transparency, Ethics, Interaction, Engagement, and Accuracy, creating a significant literature gap. To address this gap, this study employs Partial Least Squares Structural Equation Modeling (PLS-SEM) to investigate the determinant of chatbots adoption in education among students, considering the Technology Readiness Index (TRI) and Technology Acceptance Model (TAM). Utilizing a five-point Likert scale for data collection, we gathered a total of 185 responses, which were analyzed using R-Studio software. We established 12 hypotheses to achieve its objectives. The results showed that Optimism and Innovativeness are positively associated with Perceived Ease of Use (PEOU) and Perceived Usefulness (PU). Conversely, Discomfort and Insecurity negatively impact PEOU, with only Insecurity negatively affecting PU. These findings provide insights for future technology designers, elucidating critical user behavior factors influencing chatbots adoption and utilization in educational contexts.

{{</citation>}}


## eess.SY (3)



### (89/144) Adaptive Stochastic Nonlinear Model Predictive Control with Look-ahead Deep Reinforcement Learning for Autonomous Vehicle Motion Control (Baha Zarrouki et al., 2023)

{{<citation>}}

Baha Zarrouki, Chenyang Wang, Johannes Betz. (2023)  
**Adaptive Stochastic Nonlinear Model Predictive Control with Look-ahead Deep Reinforcement Learning for Autonomous Vehicle Motion Control**  

---
Primary Category: eess.SY  
Categories: cs-RO, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.04303v1)  

---


**ABSTRACT**  
In this paper, we present a Deep Reinforcement Learning (RL)-driven Adaptive Stochastic Nonlinear Model Predictive Control (SNMPC) to optimize uncertainty handling, constraints robustification, feasibility, and closed-loop performance. To this end, we conceive an RL agent to proactively anticipate upcoming control tasks and to dynamically determine the most suitable combination of key SNMPC parameters - foremost the robustification factor $\kappa$ and the Uncertainty Propagation Horizon (UPH) $T_u$. We analyze the trained RL agent's decision-making process and highlight its ability to learn context-dependent optimal parameters. One key finding is that adapting the constraints robustification factor with the learned policy reduces conservatism and improves closed-loop performance while adapting UPH renders previously infeasible SNMPC problems feasible when faced with severe disturbances. We showcase the enhanced robustness and feasibility of our Adaptive SNMPC (aSNMPC) through the real-time motion control task of an autonomous passenger vehicle to follow an optimal race line when confronted with significant time-variant disturbances. Experimental findings demonstrate that our look-ahead RL-driven aSNMPC outperforms its Static SNMPC (sSNMPC) counterpart in minimizing the lateral deviation both with accurate and inaccurate disturbance assumptions and even when driving in previously unexplored environments.

{{</citation>}}


### (90/144) Graph Neural Networks for Power Grid Operational Risk Assessment (Yadong Zhang et al., 2023)

{{<citation>}}

Yadong Zhang, Pranav M Karve, Sankaran Mahadevan. (2023)  
**Graph Neural Networks for Power Grid Operational Risk Assessment**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.03661v1)  

---


**ABSTRACT**  
In this article, the utility of graph neural network (GNN) surrogates for Monte Carlo (MC) sampling-based risk quantification in daily operations of power grid is investigated. The MC simulation process necessitates solving a large number of optimal power flow (OPF) problems corresponding to the sample values of stochastic grid variables (power demand and renewable generation), which is computationally prohibitive. Computationally inexpensive surrogates of the OPF problem provide an attractive alternative for expedited MC simulation. GNN surrogates are especially suitable due to their superior ability to handle graph-structured data. Therefore, GNN surrogates of OPF problem are trained using supervised learning. They are then used to obtain Monte Carlo (MC) samples of the quantities of interest (operating reserve, transmission line flow) given the (hours-ahead) probabilistic wind generation and load forecast. The utility of GNN surrogates is evaluated by comparing OPF-based and GNN-based grid reliability and risk for IEEE Case118 synthetic grid. It is shown that the GNN surrogates are sufficiently accurate for predicting the (bus-level, branch-level and system-level) grid state and enable fast as well as accurate operational risk quantification for power grids. The article thus develops various tools for fast reliability and risk quantification for real-world power grids using GNNs.

{{</citation>}}


### (91/144) GNN-Based Beamforming for Sum-Rate Maximization in MU-MISO Networks (Yuhang Li et al., 2023)

{{<citation>}}

Yuhang Li, Yang Lu, Bo Ai, Octavia A. Dobre, Zhiguo Ding, Dusit Niyato. (2023)  
**GNN-Based Beamforming for Sum-Rate Maximization in MU-MISO Networks**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.03659v1)  

---


**ABSTRACT**  
The advantages of graph neural networks (GNNs) in leveraging the graph topology of wireless networks have drawn increasing attentions. This paper studies the GNN-based learning approach for the sum-rate maximization in multiple-user multiple-input single-output (MU-MISO) networks subject to the users' individual data rate requirements and the power budget of the base station. By modeling the MU-MISO network as a graph, a GNN-based architecture named CRGAT is proposed to directly map the channel state information to the beamforming vectors. The attention-enabled aggregation and the residual-assisted combination are adopted to enhance the learning capability and avoid the oversmoothing issue. Furthermore, a novel activation function is proposed for the constraint due to the limited power budget at the base station. The CRGAT is trained in an unsupervised learning manner with two proposed loss functions. An evaluation method is proposed for the learning-based approach, based on which the effectiveness of the proposed CRGAT is validated in comparison with several convex optimization and learning based approaches. Numerical results are provided to reveal the advantages of the CRGAT including the millisecond-level response with limited optimality performance loss, the scalability to different number of users and power budgets, and the adaptability to different system settings.

{{</citation>}}


## cs.LG (29)



### (92/144) Class-Incremental Continual Learning for General Purpose Healthcare Models (Amritpal Singh et al., 2023)

{{<citation>}}

Amritpal Singh, Mustafa Burak Gurbuz, Shiva Souhith Gantha, Prahlad Jasti. (2023)  
**Class-Incremental Continual Learning for General Purpose Healthcare Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.04301v1)  

---


**ABSTRACT**  
Healthcare clinics regularly encounter dynamic data that changes due to variations in patient populations, treatment policies, medical devices, and emerging disease patterns. Deep learning models can suffer from catastrophic forgetting when fine-tuned in such scenarios, causing poor performance on previously learned tasks. Continual learning allows learning on new tasks without performance drop on previous tasks. In this work, we investigate the performance of continual learning models on four different medical imaging scenarios involving ten classification datasets from diverse modalities, clinical specialties, and hospitals. We implement various continual learning approaches and evaluate their performance in these scenarios. Our results demonstrate that a single model can sequentially learn new tasks from different specialties and achieve comparable performance to naive methods. These findings indicate the feasibility of recycling or sharing models across the same or different medical specialties, offering another step towards the development of general-purpose medical imaging AI that can be shared across institutions.

{{</citation>}}


### (93/144) Wearable data from subjects playing Super Mario, sitting university exams, or performing physical exercise help detect acute mood episodes via self-supervised learning (Filippo Corponi et al., 2023)

{{<citation>}}

Filippo Corponi, Bryan M. Li, Gerard Anmella, Clàudia Valenzuela-Pascual, Ariadna Mas, Isabella Pacchiarotti, Marc Valentí, Iria Grande, Antonio Benabarre, Marina Garriga, Eduard Vieta, Allan H Young, Stephen M. Lawrie, Heather C. Whalley, Diego Hidalgo-Mazzei, Antonio Vergari. (2023)  
**Wearable data from subjects playing Super Mario, sitting university exams, or performing physical exercise help detect acute mood episodes via self-supervised learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, eess-SP  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.04215v1)  

---


**ABSTRACT**  
Personal sensing, leveraging data passively and near-continuously collected with wearables from patients in their ecological environment, is a promising paradigm to monitor mood disorders (MDs), a major determinant of worldwide disease burden. However, collecting and annotating wearable data is very resource-intensive. Studies of this kind can thus typically afford to recruit only a couple dozens of patients. This constitutes one of the major obstacles to applying modern supervised machine learning techniques to MDs detection. In this paper, we overcome this data bottleneck and advance the detection of MDs acute episode vs stable state from wearables data on the back of recent advances in self-supervised learning (SSL). This leverages unlabelled data to learn representations during pre-training, subsequently exploited for a supervised task. First, we collected open-access datasets recording with an Empatica E4 spanning different, unrelated to MD monitoring, personal sensing tasks -- from emotion recognition in Super Mario players to stress detection in undergraduates -- and devised a pre-processing pipeline performing on-/off-body detection, sleep-wake detection, segmentation, and (optionally) feature extraction. With 161 E4-recorded subjects, we introduce E4SelfLearning, the largest to date open access collection, and its pre-processing pipeline. Second, we show that SSL confidently outperforms fully-supervised pipelines using either our novel E4-tailored Transformer architecture (E4mer) or classical baseline XGBoost: 81.23% against 75.35% (E4mer) and 72.02% (XGBoost) correctly classified recording segments from 64 (half acute, half stable) patients. Lastly, we illustrate that SSL performance is strongly associated with the specific surrogate task employed for pre-training as well as with unlabelled data availability.

{{</citation>}}


### (94/144) Spatio-Temporal Anomaly Detection with Graph Networks for Data Quality Monitoring of the Hadron Calorimeter (Mulugeta Weldezgina Asres et al., 2023)

{{<citation>}}

Mulugeta Weldezgina Asres, Christian Walter Omlin, Long Wang, David Yu, Pavel Parygin, Jay Dittmann, Georgia Karapostoli, Markus Seidel, Rosamaria Venditti, Luka Lambrecht, Emanuele Usai, Muhammad Ahmad, Javier Fernandez Menendez, Kaori Maeshima, the CMS-HCAL Collaboration. (2023)  
**Spatio-Temporal Anomaly Detection with Graph Networks for Data Quality Monitoring of the Hadron Calorimeter**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.04190v1)  

---


**ABSTRACT**  
The compact muon solenoid (CMS) experiment is a general-purpose detector for high-energy collision at the large hadron collider (LHC) at CERN. It employs an online data quality monitoring (DQM) system to promptly spot and diagnose particle data acquisition problems to avoid data quality loss. In this study, we present semi-supervised spatio-temporal anomaly detection (AD) monitoring for the physics particle reading channels of the hadronic calorimeter (HCAL) of the CMS using three-dimensional digi-occupancy map data of the DQM. We propose the GraphSTAD system, which employs convolutional and graph neural networks to learn local spatial characteristics induced by particles traversing the detector, and global behavior owing to shared backend circuit connections and housing boxes of the channels, respectively. Recurrent neural networks capture the temporal evolution of the extracted spatial features. We have validated the accuracy of the proposed AD system in capturing diverse channel fault types using the LHC Run-2 collision data sets. The GraphSTAD system has achieved production-level accuracy and is being integrated into the CMS core production system--for real-time monitoring of the HCAL. We have also provided a quantitative performance comparison with alternative benchmark models to demonstrate the promising leverage of the presented system.

{{</citation>}}


### (95/144) Multi-resolution Time-Series Transformer for Long-term Forecasting (Yitian Zhang et al., 2023)

{{<citation>}}

Yitian Zhang, Liheng Ma, Soumyasundar Pal, Yingxue Zhang, Mark Coates. (2023)  
**Multi-resolution Time-Series Transformer for Long-term Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.04147v1)  

---


**ABSTRACT**  
The performance of transformers for time-series forecasting has improved significantly. Recent architectures learn complex temporal patterns by segmenting a time-series into patches and using the patches as tokens. The patch size controls the ability of transformers to learn the temporal patterns at different frequencies: shorter patches are effective for learning localized, high-frequency patterns, whereas mining long-term seasonalities and trends requires longer patches. Inspired by this observation, we propose a novel framework, Multi-resolution Time-Series Transformer (MTST), which consists of a multi-branch architecture for simultaneous modeling of diverse temporal patterns at different resolutions. In contrast to many existing time-series transformers, we employ relative positional encoding, which is better suited for extracting periodic components at different scales. Extensive experiments on several real-world datasets demonstrate the effectiveness of MTST in comparison to state-of-the-art forecasting techniques.

{{</citation>}}


### (96/144) Do Language Models Learn Semantics of Code? A Case Study in Vulnerability Detection (Benjamin Steenhoek et al., 2023)

{{<citation>}}

Benjamin Steenhoek, Md Mahbubur Rahman, Shaila Sharmin, Wei Le. (2023)  
**Do Language Models Learn Semantics of Code? A Case Study in Vulnerability Detection**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Language Model, Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2311.04109v1)  

---


**ABSTRACT**  
Recently, pretrained language models have shown state-of-the-art performance on the vulnerability detection task. These models are pretrained on a large corpus of source code, then fine-tuned on a smaller supervised vulnerability dataset. Due to the different training objectives and the performance of the models, it is interesting to consider whether the models have learned the semantics of code relevant to vulnerability detection, namely bug semantics, and if so, how the alignment to bug semantics relates to model performance. In this paper, we analyze the models using three distinct methods: interpretability tools, attention analysis, and interaction matrix analysis. We compare the models' influential feature sets with the bug semantic features which define the causes of bugs, including buggy paths and Potentially Vulnerable Statements (PVS). We find that (1) better-performing models also aligned better with PVS, (2) the models failed to align strongly to PVS, and (3) the models failed to align at all to buggy paths. Based on our analysis, we developed two annotation methods which highlight the bug semantics inside the model's inputs. We evaluated our approach on four distinct transformer models and four vulnerability datasets and found that our annotations improved the models' performance in the majority of settings - 11 out of 16, with up to 9.57 points improvement in F1 score compared to conventional fine-tuning. We further found that with our annotations, the models aligned up to 232% better to potentially vulnerable statements. Our findings indicate that it is helpful to provide the model with information of the bug semantics, that the model can attend to it, and motivate future work in learning more complex path-based bug semantics. Our code and data are available at https://figshare.com/s/4a16a528d6874aad51a0.

{{</citation>}}


### (97/144) Time-Efficient Reinforcement Learning with Stochastic Stateful Policies (Firas Al-Hafez et al., 2023)

{{<citation>}}

Firas Al-Hafez, Guoping Zhao, Jan Peters, Davide Tateo. (2023)  
**Time-Efficient Reinforcement Learning with Stochastic Stateful Policies**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.04082v1)  

---


**ABSTRACT**  
Stateful policies play an important role in reinforcement learning, such as handling partially observable environments, enhancing robustness, or imposing an inductive bias directly into the policy structure. The conventional method for training stateful policies is Backpropagation Through Time (BPTT), which comes with significant drawbacks, such as slow training due to sequential gradient propagation and the occurrence of vanishing or exploding gradients. The gradient is often truncated to address these issues, resulting in a biased policy update. We present a novel approach for training stateful policies by decomposing the latter into a stochastic internal state kernel and a stateless policy, jointly optimized by following the stateful policy gradient. We introduce different versions of the stateful policy gradient theorem, enabling us to easily instantiate stateful variants of popular reinforcement learning and imitation learning algorithms. Furthermore, we provide a theoretical analysis of our new gradient estimator and compare it with BPTT. We evaluate our approach on complex continuous control tasks, e.g., humanoid locomotion, and demonstrate that our gradient estimator scales effectively with task complexity while offering a faster and simpler alternative to BPTT.

{{</citation>}}


### (98/144) Multitask Multimodal Prompted Training for Interactive Embodied Task Completion (Georgios Pantazopoulos et al., 2023)

{{<citation>}}

Georgios Pantazopoulos, Malvina Nikandrou, Amit Parekh, Bhathiya Hemanthage, Arash Eshghi, Ioannis Konstas, Verena Rieser, Oliver Lemon, Alessandro Suglia. (2023)  
**Multitask Multimodal Prompted Training for Interactive Embodied Task Completion**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Dialog  
[Paper Link](http://arxiv.org/abs/2311.04067v1)  

---


**ABSTRACT**  
Interactive and embodied tasks pose at least two fundamental challenges to existing Vision & Language (VL) models, including 1) grounding language in trajectories of actions and observations, and 2) referential disambiguation. To tackle these challenges, we propose an Embodied MultiModal Agent (EMMA): a unified encoder-decoder model that reasons over images and trajectories, and casts action prediction as multimodal text generation. By unifying all tasks as text generation, EMMA learns a language of actions which facilitates transfer across tasks. Different to previous modular approaches with independently trained components, we use a single multitask model where each task contributes to goal completion. EMMA performs on par with similar models on several VL benchmarks and sets a new state-of-the-art performance (36.81% success rate) on the Dialog-guided Task Completion (DTC), a benchmark to evaluate dialog-guided agents in the Alexa Arena

{{</citation>}}


### (99/144) Multi-View Causal Representation Learning with Partial Observability (Dingling Yao et al., 2023)

{{<citation>}}

Dingling Yao, Danru Xu, Sébastien Lachapelle, Sara Magliacane, Perouz Taslakian, Georg Martius, Julius von Kügelgen, Francesco Locatello. (2023)  
**Multi-View Causal Representation Learning with Partial Observability**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.04056v1)  

---


**ABSTRACT**  
We present a unified framework for studying the identifiability of representations learned from simultaneously observed views, such as different data modalities. We allow a partially observed setting in which each view constitutes a nonlinear mixture of a subset of underlying latent variables, which can be causally related. We prove that the information shared across all subsets of any number of views can be learned up to a smooth bijection using contrastive learning and a single encoder per view. We also provide graphical criteria indicating which latent variables can be identified through a simple set of rules, which we refer to as identifiability algebra. Our general framework and theoretical results unify and extend several previous works on multi-view nonlinear ICA, disentanglement, and causal representation learning. We experimentally validate our claims on numerical, image, and multi-modal data sets. Further, we demonstrate that the performance of prior methods is recovered in different special cases of our setup. Overall, we find that access to multiple partial views enables us to identify a more fine-grained representation, under the generally milder assumption of partial observability.

{{</citation>}}


### (100/144) Generative Structural Design Integrating BIM and Diffusion Model (Zhili He et al., 2023)

{{<citation>}}

Zhili He, Yu-Hsing Wang, Jian Zhang. (2023)  
**Generative Structural Design Integrating BIM and Diffusion Model**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.04052v1)  

---


**ABSTRACT**  
Intelligent structural design using AI can effectively reduce time overhead and increase efficiency. It has potential to become the new design paradigm in the future to assist and even replace engineers, and so it has become a research hotspot in the academic community. However, current methods have some limitations to be addressed, whether in terms of application scope, visual quality of generated results, or evaluation metrics of results. This study proposes a comprehensive solution. Firstly, we introduce building information modeling (BIM) into intelligent structural design and establishes a structural design pipeline integrating BIM and generative AI, which is a powerful supplement to the previous frameworks that only considered CAD drawings. In order to improve the perceptual quality and details of generations, this study makes 3 contributions. Firstly, in terms of generation framework, inspired by the process of human drawing, a novel 2-stage generation framework is proposed to replace the traditional end-to-end framework to reduce the generation difficulty for AI models. Secondly, in terms of generative AI tools adopted, diffusion models (DMs) are introduced to replace widely used generative adversarial network (GAN)-based models, and a novel physics-based conditional diffusion model (PCDM) is proposed to consider different design prerequisites. Thirdly, in terms of neural networks, an attention block (AB) consisting of a self-attention block (SAB) and a parallel cross-attention block (PCAB) is designed to facilitate cross-domain data fusion. The quantitative and qualitative results demonstrate the powerful generation and representation capabilities of PCDM. Necessary ablation studies are conducted to examine the validity of the methods. This study also shows that DMs have the potential to replace GANs and become the new benchmark for generative problems in civil engineering.

{{</citation>}}


### (101/144) Reinforcement Learning Fine-tuning of Language Models is Biased Towards More Extractable Features (Diogo Cruz et al., 2023)

{{<citation>}}

Diogo Cruz, Edoardo Pona, Alex Holness-Tofts, Elias Schmied, Víctor Abia Alonso, Charlie Griffin, Bogdan-Ionut Cirstea. (2023)  
**Reinforcement Learning Fine-tuning of Language Models is Biased Towards More Extractable Features**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: AI, Bias, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.04046v1)  

---


**ABSTRACT**  
Many capable large language models (LLMs) are developed via self-supervised pre-training followed by a reinforcement-learning fine-tuning phase, often based on human or AI feedback. During this stage, models may be guided by their inductive biases to rely on simpler features which may be easier to extract, at a cost to robustness and generalisation. We investigate whether principles governing inductive biases in the supervised fine-tuning of LLMs also apply when the fine-tuning process uses reinforcement learning. Following Lovering et al (2021), we test two hypotheses: that features more $\textit{extractable}$ after pre-training are more likely to be utilised by the final policy, and that the evidence for/against a feature predicts whether it will be utilised. Through controlled experiments on synthetic and natural language tasks, we find statistically significant correlations which constitute strong evidence for these hypotheses.

{{</citation>}}


### (102/144) Impact of HPO on AutoML Forecasting Ensembles (David Hoffmann, 2023)

{{<citation>}}

David Hoffmann. (2023)  
**Impact of HPO on AutoML Forecasting Ensembles**  

---
Primary Category: cs.LG  
Categories: I-2-6; I-2-2, cs-AI, cs-LG, cs.LG  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2311.04034v1)  

---


**ABSTRACT**  
A forecasting ensemble consisting of a diverse range of estimators for both local and global univariate forecasting, in particular MQ-CNN,DeepAR, Prophet, NPTS, ARIMA and ETS, can be used to make forecasts for a variety of problems. This paper delves into the aspect of adding different hyperparameter optimization strategies to the deep learning models in such a setup (DeepAR and MQ-CNN), exploring the trade-off between added training cost and the increase in accuracy for different configurations. It shows that in such a setup, adding hyperparameter optimization can lead to performance improvements, with the final setup having a 9.9 % percent accuracy improvement with respect to the avg-wQL over the baseline ensemble without HPO, accompanied by a 65.8 % increase in end-to-end ensemble latency. This improvement is based on an empirical analysis of combining the ensemble pipeline with different tuning strategies, namely Bayesian Optimisation and Hyperband and different configurations of those strategies. In the final configuration, the proposed combination of ensemble learning and HPO outperforms the state of the art commercial AutoML forecasting solution, Amazon Forecast, with a 3.5 % lower error and 16.0 % lower end-to-end ensemble latency.

{{</citation>}}


### (103/144) AGNES: Abstraction-guided Framework for Deep Neural Networks Security (Akshay Dhonthi et al., 2023)

{{<citation>}}

Akshay Dhonthi, Marcello Eiermann, Ernst Moritz Hahn, Vahid Hashemi. (2023)  
**AGNES: Abstraction-guided Framework for Deep Neural Networks Security**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.04009v1)  

---


**ABSTRACT**  
Deep Neural Networks (DNNs) are becoming widespread, particularly in safety-critical areas. One prominent application is image recognition in autonomous driving, where the correct classification of objects, such as traffic signs, is essential for safe driving. Unfortunately, DNNs are prone to backdoors, meaning that they concentrate on attributes of the image that should be irrelevant for their correct classification. Backdoors are integrated into a DNN during training, either with malicious intent (such as a manipulated training process, because of which a yellow sticker always leads to a traffic sign being recognised as a stop sign) or unintentional (such as a rural background leading to any traffic sign being recognised as animal crossing, because of biased training data).   In this paper, we introduce AGNES, a tool to detect backdoors in DNNs for image recognition. We discuss the principle approach on which AGNES is based. Afterwards, we show that our tool performs better than many state-of-the-art methods for multiple relevant case studies.

{{</citation>}}


### (104/144) The Energy Prediction Smart-Meter Dataset: Analysis of Previous Competitions and Beyond (Direnc Pekaslan et al., 2023)

{{<citation>}}

Direnc Pekaslan, Jose Maria Alonso-Moral, Kasun Bandara, Christoph Bergmeir, Juan Bernabe-Moreno, Robert Eigenmann, Nils Einecke, Selvi Ergen, Rakshitha Godahewa, Hansika Hewamalage, Jesus Lago, Steffen Limmer, Sven Rebhan, Boris Rabinovich, Dilini Rajapasksha, Heda Song, Christian Wagner, Wenlong Wu, Luis Magdalena, Isaac Triguero. (2023)  
**The Energy Prediction Smart-Meter Dataset: Analysis of Previous Competitions and Beyond**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.04007v1)  

---


**ABSTRACT**  
This paper presents the real-world smart-meter dataset and offers an analysis of solutions derived from the Energy Prediction Technical Challenges, focusing primarily on two key competitions: the IEEE Computational Intelligence Society (IEEE-CIS) Technical Challenge on Energy Prediction from Smart Meter data in 2020 (named EP) and its follow-up challenge at the IEEE International Conference on Fuzzy Systems (FUZZ-IEEE) in 2021 (named as XEP). These competitions focus on accurate energy consumption forecasting and the importance of interpretability in understanding the underlying factors. The challenge aims to predict monthly and yearly estimated consumption for households, addressing the accurate billing problem with limited historical smart meter data. The dataset comprises 3,248 smart meters, with varying data availability ranging from a minimum of one month to a year. This paper delves into the challenges, solutions and analysing issues related to the provided real-world smart meter data, developing accurate predictions at the household level, and introducing evaluation criteria for assessing interpretability. Additionally, this paper discusses aspects beyond the competitions: opportunities for energy disaggregation and pattern detection applications at the household level, significance of communicating energy-driven factors for optimised billing, and emphasising the importance of responsible AI and data privacy considerations. These aspects provide insights into the broader implications and potential advancements in energy consumption prediction. Overall, these competitions provide a dataset for residential energy research and serve as a catalyst for exploring accurate forecasting, enhancing interpretability, and driving progress towards the discussion of various aspects such as energy disaggregation, demand response programs or behavioural interventions.

{{</citation>}}


### (105/144) Its All Graph To Me: Foundational Topology Models with Contrastive Learning on Multiple Domains (Alex O. Davies et al., 2023)

{{<citation>}}

Alex O. Davies, Riku W. Green, Nirav S. Ajmeri, Telmo M. Silva Filho. (2023)  
**Its All Graph To Me: Foundational Topology Models with Contrastive Learning on Multiple Domains**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.03976v1)  

---


**ABSTRACT**  
Representations and embeddings of graph data have been essential in many domains of research.   The principle benefit of learning such representations is that the pre-trained model can be fine-tuned on smaller datasets where data or labels are scarse.   Existing models, however, are domain specific; for example a model trained on molecular graphs is fine-tuned on other molecular graphs.   This means that in many application cases the choice of pre-trained model can be arbitrary, and novel domains may lack an appropriate pre-trained model.   This is of particular issue where data is scarse, precluding traditional supervised methods.   In this work we use adversarial contrastive learning to present a \method, a model pre-trained on many graph domains.   We train the model only on topologies but include node labels in evaluation.   We evaluate the efficacy of its learnt representations on various downstream tasks.   Against baseline models pre-trained on single domains, as well as un-trained models and non-transferred models, we show that performance is equal or better using our single model.   This includes when node labels are used in evaluation, where performance is consistently superior to single-domain or non-pre-trained models.

{{</citation>}}


### (106/144) MixtureGrowth: Growing Neural Networks by Recombining Learned Parameters (Chau Pham et al., 2023)

{{<citation>}}

Chau Pham, Piotr Teterwak, Soren Nelson, Bryan A. Plummer. (2023)  
**MixtureGrowth: Growing Neural Networks by Recombining Learned Parameters**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.04251v1)  

---


**ABSTRACT**  
Most deep neural networks are trained under fixed network architectures and require retraining when the architecture changes. If expanding the network's size is needed, it is necessary to retrain from scratch, which is expensive. To avoid this, one can grow from a small network by adding random weights over time to gradually achieve the target network size. However, this naive approach falls short in practice as it brings too much noise to the growing process. Prior work tackled this issue by leveraging the already learned weights and training data for generating new weights through conducting a computationally expensive analysis step. In this paper, we introduce MixtureGrowth, a new approach to growing networks that circumvents the initialization overhead in prior work. Before growing, each layer in our model is generated with a linear combination of parameter templates. Newly grown layer weights are generated by using a new linear combination of existing templates for a layer. On one hand, these templates are already trained for the task, providing a strong initialization. On the other, the new coefficients provide flexibility for the added layer weights to learn something new. We show that our approach boosts top-1 accuracy over the state-of-the-art by 2-2.5% on CIFAR-100 and ImageNet datasets, while achieving comparable performance with fewer FLOPs to a larger network trained from scratch. Code is available at https://github.com/chaudatascience/mixturegrowth.

{{</citation>}}


### (107/144) Temporal Graph Representation Learning with Adaptive Augmentation Contrastive (Hongjiang Chen et al., 2023)

{{<citation>}}

Hongjiang Chen, Pengfei Jiao, Huijun Tang, Huaming Wu. (2023)  
**Temporal Graph Representation Learning with Adaptive Augmentation Contrastive**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Augmentation, Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.03897v1)  

---


**ABSTRACT**  
Temporal graph representation learning aims to generate low-dimensional dynamic node embeddings to capture temporal information as well as structural and property information. Current representation learning methods for temporal networks often focus on capturing fine-grained information, which may lead to the model capturing random noise instead of essential semantic information. While graph contrastive learning has shown promise in dealing with noise, it only applies to static graphs or snapshots and may not be suitable for handling time-dependent noise. To alleviate the above challenge, we propose a novel Temporal Graph representation learning with Adaptive augmentation Contrastive (TGAC) model. The adaptive augmentation on the temporal graph is made by combining prior knowledge with temporal information, and the contrastive objective function is constructed by defining the augmented inter-view contrast and intra-view contrast. To complement TGAC, we propose three adaptive augmentation strategies that modify topological features to reduce noise from the network. Our extensive experiments on various real networks demonstrate that the proposed model outperforms other temporal graph representation learning methods.

{{</citation>}}


### (108/144) PT-Tuning: Bridging the Gap between Time Series Masked Reconstruction and Forecasting via Prompt Token Tuning (Hao Liu et al., 2023)

{{<citation>}}

Hao Liu, Jinrui Gan, Xiaoxuan Fan, Yi Zhang, Chuanxian Luo, Jing Zhang, Guangxin Jiang, Yucheng Qian, Changwei Zhao, Huan Ma, Zhenyu Guo. (2023)  
**PT-Tuning: Bridging the Gap between Time Series Masked Reconstruction and Forecasting via Prompt Token Tuning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.03768v1)  

---


**ABSTRACT**  
Self-supervised learning has been actively studied in time series domain recently, especially for masked reconstruction. Most of these methods follow the "Pre-training + Fine-tuning" paradigm in which a new decoder replaces the pre-trained decoder to fit for a specific downstream task, leading to inconsistency of upstream and downstream tasks. In this paper, we first point out that the unification of task objectives and adaptation for task difficulty are critical for bridging the gap between time series masked reconstruction and forecasting. By reserving the pre-trained mask token during fine-tuning stage, the forecasting task can be taken as a special case of masked reconstruction, where the future values are masked and reconstructed based on history values. It guarantees the consistency of task objectives but there is still a gap in task difficulty. Because masked reconstruction can utilize contextual information while forecasting can only use historical information to reconstruct. To further mitigate the existed gap, we propose a simple yet effective prompt token tuning (PT-Tuning) paradigm, in which all pre-trained parameters are frozen and only a few trainable prompt tokens are added to extended mask tokens in element-wise manner. Extensive experiments on real-world datasets demonstrate the superiority of our proposed paradigm with state-of-the-art performance compared to representation learning and end-to-end supervised forecasting methods.

{{</citation>}}


### (109/144) Neuro-GPT: Developing A Foundation Model for EEG (Wenhui Cui et al., 2023)

{{<citation>}}

Wenhui Cui, Woojae Jeong, Philipp Thölke, Takfarinas Medani, Karim Jerbi, Anand A. Joshi, Richard M. Leahy. (2023)  
**Neuro-GPT: Developing A Foundation Model for EEG**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.03764v1)  

---


**ABSTRACT**  
To handle the scarcity and heterogeneity of electroencephalography (EEG) data in Brain-Computer Interface (BCI) tasks, and to harness the vast public data, we propose Neuro-GPT, a foundation model consisting of an EEG encoder and a GPT model. The foundation model is pre-trained on a large-scale public EEG dataset, using a self-supervised task which learns how to reconstruct the masked chunk in EEG. We then fine-tune the foundation model on a Motor Imagery Classification task where only 9 subjects are available. Experiments demonstrated that applying foundation model can significantly improve classification performance compared to the model trained from scratch, which provides evidence for the advanced generalizability of foundation model and the ability to address the challenges of data scarcity and heterogeneity.

{{</citation>}}


### (110/144) Learning Decentralized Traffic Signal Controllers with Multi-Agent Graph Reinforcement Learning (Yao Zhang et al., 2023)

{{<citation>}}

Yao Zhang, Zhiwen Yu, Jun Zhang, Liang Wang, Tom H. Luan, Bin Guo, Chau Yuen. (2023)  
**Learning Decentralized Traffic Signal Controllers with Multi-Agent Graph Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.03756v1)  

---


**ABSTRACT**  
This paper considers optimal traffic signal control in smart cities, which has been taken as a complex networked system control problem. Given the interacting dynamics among traffic lights and road networks, attaining controller adaptivity and scalability stands out as a primary challenge. Capturing the spatial-temporal correlation among traffic lights under the framework of Multi-Agent Reinforcement Learning (MARL) is a promising solution. Nevertheless, existing MARL algorithms ignore effective information aggregation which is fundamental for improving the learning capacity of decentralized agents. In this paper, we design a new decentralized control architecture with improved environmental observability to capture the spatial-temporal correlation. Specifically, we first develop a topology-aware information aggregation strategy to extract correlation-related information from unstructured data gathered in the road network. Particularly, we transfer the road network topology into a graph shift operator by forming a diffusion process on the topology, which subsequently facilitates the construction of graph signals. A diffusion convolution module is developed, forming a new MARL algorithm, which endows agents with the capabilities of graph learning. Extensive experiments based on both synthetic and real-world datasets verify that our proposal outperforms existing decentralized algorithms.

{{</citation>}}


### (111/144) Analysis and Applications of Deep Learning with Finite Samples in Full Life-Cycle Intelligence of Nuclear Power Generation (Chenwei Tang et al., 2023)

{{<citation>}}

Chenwei Tang, Wenqiang Zhou, Dong Wang, Caiyang Yu, Zhenan He, Jizhe Zhou, Shudong Huang, Yi Gao, Jianming Chen, Wentao Feng, Jiancheng Lv. (2023)  
**Analysis and Applications of Deep Learning with Finite Samples in Full Life-Cycle Intelligence of Nuclear Power Generation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.04247v1)  

---


**ABSTRACT**  
The advent of Industry 4.0 has precipitated the incorporation of Artificial Intelligence (AI) methods within industrial contexts, aiming to realize intelligent manufacturing, operation as well as maintenance, also known as industrial intelligence. However, intricate industrial milieus, particularly those relating to energy exploration and production, frequently encompass data characterized by long-tailed class distribution, sample imbalance, and domain shift. These attributes pose noteworthy challenges to data-centric Deep Learning (DL) techniques, crucial for the realization of industrial intelligence. The present study centers on the intricate and distinctive industrial scenarios of Nuclear Power Generation (NPG), meticulously scrutinizing the application of DL techniques under the constraints of finite data samples. Initially, the paper expounds on potential employment scenarios for AI across the full life-cycle of NPG. Subsequently, we delve into an evaluative exposition of DL's advancement, grounded in the finite sample perspective. This encompasses aspects such as small-sample learning, few-shot learning, zero-shot learning, and open-set recognition, also referring to the unique data characteristics of NPG. The paper then proceeds to present two specific case studies. The first revolves around the automatic recognition of zirconium alloy metallography, while the second pertains to open-set recognition for signal diagnosis of machinery sensors. These cases, spanning the entirety of NPG's life-cycle, are accompanied by constructive outcomes and insightful deliberations. By exploring and applying DL methodologies within the constraints of finite sample availability, this paper not only furnishes a robust technical foundation but also introduces a fresh perspective toward the secure and efficient advancement and exploitation of this advanced energy source.

{{</citation>}}


### (112/144) Learning to Learn for Few-shot Continual Active Learning (Stella Ho et al., 2023)

{{<citation>}}

Stella Ho, Ming Liu, Shang Gao, Longxiang Gao. (2023)  
**Learning to Learn for Few-shot Continual Active Learning**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Active Learning, NLP  
[Paper Link](http://arxiv.org/abs/2311.03732v1)  

---


**ABSTRACT**  
Continual learning strives to ensure stability in solving previously seen tasks while demonstrating plasticity in a novel domain. Recent advances in CL are mostly confined to a supervised learning setting, especially in NLP domain. In this work, we consider a few-shot continual active learning (CAL) setting where labeled data is inadequate, and unlabeled data is abundant but with a limited annotation budget. We propose a simple but efficient method, called Meta-Continual Active Learning. Specifically, we employ meta-learning and experience replay to address the trade-off between stability and plasticity. As a result, it finds an optimal initialization that efficiently utilizes annotated information for fast adaptation while preventing catastrophic forgetting of past tasks. We conduct extensive experiments to validate the effectiveness of the proposed method and analyze the effect of various active learning strategies and memory sample selection methods in a few-shot CAL setup. Our experiment results demonstrate that random sampling is the best default strategy for both active learning and memory sample selection to solve few-shot CAL problems.

{{</citation>}}


### (113/144) Mitigating Estimation Errors by Twin TD-Regularized Actor and Critic for Deep Reinforcement Learning (Junmin Zhong et al., 2023)

{{<citation>}}

Junmin Zhong, Ruofan Wu, Jennie Si. (2023)  
**Mitigating Estimation Errors by Twin TD-Regularized Actor and Critic for Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.03711v1)  

---


**ABSTRACT**  
We address the issue of estimation bias in deep reinforcement learning (DRL) by introducing solution mechanisms that include a new, twin TD-regularized actor-critic (TDR) method. It aims at reducing both over and under-estimation errors. With TDR and by combining good DRL improvements, such as distributional learning and long N-step surrogate stage reward (LNSS) method, we show that our new TDR-based actor-critic learning has enabled DRL methods to outperform their respective baselines in challenging environments in DeepMind Control Suite. Furthermore, they elevate TD3 and SAC respectively to a level of performance comparable to that of D4PG (the current SOTA), and they also improve the performance of D4PG to a new SOTA level measured by mean reward, convergence speed, learning success rate, and learning variance.

{{</citation>}}


### (114/144) A Novel Variational Lower Bound for Inverse Reinforcement Learning (Yikang Gui et al., 2023)

{{<citation>}}

Yikang Gui, Prashant Doshi. (2023)  
**A Novel Variational Lower Bound for Inverse Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.03698v1)  

---


**ABSTRACT**  
Inverse reinforcement learning (IRL) seeks to learn the reward function from expert trajectories, to understand the task for imitation or collaboration thereby removing the need for manual reward engineering. However, IRL in the context of large, high-dimensional problems with unknown dynamics has been particularly challenging. In this paper, we present a new Variational Lower Bound for IRL (VLB-IRL), which is derived under the framework of a probabilistic graphical model with an optimality node. Our method simultaneously learns the reward function and policy under the learned reward function by maximizing the lower bound, which is equivalent to minimizing the reverse Kullback-Leibler divergence between an approximated distribution of optimality given the reward function and the true distribution of optimality given trajectories. This leads to a new IRL method that learns a valid reward function such that the policy under the learned reward achieves expert-level performance on several known domains. Importantly, the method outperforms the existing state-of-the-art IRL algorithms on these domains by demonstrating better reward from the learned policy.

{{</citation>}}


### (115/144) Context Shift Reduction for Offline Meta-Reinforcement Learning (Yunkai Gao et al., 2023)

{{<citation>}}

Yunkai Gao, Rui Zhang, Jiaming Guo, Fan Wu, Qi Yi, Shaohui Peng, Siming Lan, Ruizhi Chen, Zidong Du, Xing Hu, Qi Guo, Ling Li, Yunji Chen. (2023)  
**Context Shift Reduction for Offline Meta-Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.03695v1)  

---


**ABSTRACT**  
Offline meta-reinforcement learning (OMRL) utilizes pre-collected offline datasets to enhance the agent's generalization ability on unseen tasks. However, the context shift problem arises due to the distribution discrepancy between the contexts used for training (from the behavior policy) and testing (from the exploration policy). The context shift problem leads to incorrect task inference and further deteriorates the generalization ability of the meta-policy. Existing OMRL methods either overlook this problem or attempt to mitigate it with additional information. In this paper, we propose a novel approach called Context Shift Reduction for OMRL (CSRO) to address the context shift problem with only offline datasets. The key insight of CSRO is to minimize the influence of policy in context during both the meta-training and meta-test phases. During meta-training, we design a max-min mutual information representation learning mechanism to diminish the impact of the behavior policy on task representation. In the meta-test phase, we introduce the non-prior context collection strategy to reduce the effect of the exploration policy. Experimental results demonstrate that CSRO significantly reduces the context shift and improves the generalization ability, surpassing previous methods across various challenging domains.

{{</citation>}}


### (116/144) Stable Modular Control via Contraction Theory for Reinforcement Learning (Bing Song et al., 2023)

{{<citation>}}

Bing Song, Jean-Jacques Slotine, Quang-Cuong Pham. (2023)  
**Stable Modular Control via Contraction Theory for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.03669v1)  

---


**ABSTRACT**  
We propose a novel way to integrate control techniques with reinforcement learning (RL) for stability, robustness, and generalization: leveraging contraction theory to realize modularity in neural control, which ensures that combining stable subsystems can automatically preserve the stability. We realize such modularity via signal composition and dynamic decomposition. Signal composition creates the latent space, within which RL applies to maximizing rewards. Dynamic decomposition is realized by coordinate transformation that creates an auxiliary space, within which the latent signals are coupled in the way that their combination can preserve stability provided each signal, that is, each subsystem, has stable self-feedbacks. Leveraging modularity, the nonlinear stability problem is deconstructed into algebraically solvable ones, the stability of the subsystems in the auxiliary space, yielding linear constraints on the input gradients of control networks that can be as simple as switching the signs of network weights. This minimally invasive method for stability allows arguably easy integration into the modular neural architectures in machine learning, like hierarchical RL, and improves their performance. We demonstrate in simulation the necessity and the effectiveness of our method: the necessity for robustness and generalization, and the effectiveness in improving hierarchical RL for manipulation learning.

{{</citation>}}


### (117/144) GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks (Zhonghang Li et al., 2023)

{{<citation>}}

Zhonghang Li, Lianghao Xia, Yong Xu, Chao Huang. (2023)  
**GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: GPT, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.04245v1)  

---


**ABSTRACT**  
In recent years, there has been a rapid development of spatio-temporal prediction techniques in response to the increasing demands of traffic management and travel planning. While advanced end-to-end models have achieved notable success in improving predictive performance, their integration and expansion pose significant challenges. This work aims to address these challenges by introducing a spatio-temporal pre-training framework that seamlessly integrates with downstream baselines and enhances their performance. The framework is built upon two key designs: (i) We propose a spatio-temporal mask autoencoder as a pre-training model for learning spatio-temporal dependencies. The model incorporates customized parameter learners and hierarchical spatial pattern encoding networks. These modules are specifically designed to capture spatio-temporal customized representations and intra- and inter-cluster region semantic relationships, which have often been neglected in existing approaches. (ii) We introduce an adaptive mask strategy as part of the pre-training mechanism. This strategy guides the mask autoencoder in learning robust spatio-temporal representations and facilitates the modeling of different relationships, ranging from intra-cluster to inter-cluster, in an easy-to-hard training manner. Extensive experiments conducted on representative benchmarks demonstrate the effectiveness of our proposed method. We have made our model implementation publicly available at https://github.com/HKUDS/GPT-ST.

{{</citation>}}


### (118/144) SeRO: Self-Supervised Reinforcement Learning for Recovery from Out-of-Distribution Situations (Chan Kim et al., 2023)

{{<citation>}}

Chan Kim, Jaekyung Cho, Christophe Bobda, Seung-Woo Seo, Seong-Woo Kim. (2023)  
**SeRO: Self-Supervised Reinforcement Learning for Recovery from Out-of-Distribution Situations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.03651v1)  

---


**ABSTRACT**  
Robotic agents trained using reinforcement learning have the problem of taking unreliable actions in an out-of-distribution (OOD) state. Agents can easily become OOD in real-world environments because it is almost impossible for them to visit and learn the entire state space during training. Unfortunately, unreliable actions do not ensure that agents perform their original tasks successfully. Therefore, agents should be able to recognize whether they are in OOD states and learn how to return to the learned state distribution rather than continue to take unreliable actions. In this study, we propose a novel method for retraining agents to recover from OOD situations in a self-supervised manner when they fall into OOD states. Our in-depth experimental results demonstrate that our method substantially improves the agent's ability to recover from OOD situations in terms of sample efficiency and restoration of the performance for the original tasks. Moreover, we show that our method can retrain the agent to recover from OOD situations even when in-distribution states are difficult to visit through exploration.

{{</citation>}}


### (119/144) HKTGNN: Hierarchical Knowledge Transferable Graph Neural Network-based Supply Chain Risk Assessment (Zhanting Zhou et al., 2023)

{{<citation>}}

Zhanting Zhou, Kejun Bi, Yuyanzhen Zhong, Chao Tang, Dongfen Li, Shi Ying, Ruijin Wang. (2023)  
**HKTGNN: Hierarchical Knowledge Transferable Graph Neural Network-based Supply Chain Risk Assessment**  

---
Primary Category: cs.LG  
Categories: I-2-4, cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2311.04244v1)  

---


**ABSTRACT**  
The strength of a supply chain is an important measure of a country's or region's technical advancement and overall competitiveness. Establishing supply chain risk assessment models for effective management and mitigation of potential risks has become increasingly crucial. As the number of businesses grows, the important relationships become more complicated and difficult to measure. This emphasizes the need of extracting relevant information from graph data. Previously, academics mostly employed knowledge inference to increase the visibility of links between nodes in the supply chain. However, they have not solved the data hunger problem of single node feature characteristics. We propose a hierarchical knowledge transferable graph neural network-based (HKTGNN) supply chain risk assessment model to address these issues. Our approach is based on current graph embedding methods for assessing corporate investment risk assessment. We embed the supply chain network corresponding to individual goods in the supply chain using the graph embedding module, resulting in a directed homogeneous graph with just product nodes. This reduces the complicated supply chain network into a basic product network. It addresses difficulties using the domain difference knowledge transferable module based on centrality, which is presented by the premise that supply chain feature characteristics may be biased in the actual world. Meanwhile, the feature complement and message passing will alleviate the data hunger problem, which is driven by domain differences. Our model outperforms in experiments on a real-world supply chain dataset. We will give an equation to prove that our comparative experiment is both effective and fair.

{{</citation>}}


### (120/144) Counterfactual Data Augmentation with Contrastive Learning (Ahmed Aloui et al., 2023)

{{<citation>}}

Ahmed Aloui, Juncheng Dong, Cat P. Le, Vahid Tarokh. (2023)  
**Counterfactual Data Augmentation with Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ME, stat-ML  
Keywords: Augmentation, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.03630v1)  

---


**ABSTRACT**  
Statistical disparity between distinct treatment groups is one of the most significant challenges for estimating Conditional Average Treatment Effects (CATE). To address this, we introduce a model-agnostic data augmentation method that imputes the counterfactual outcomes for a selected subset of individuals. Specifically, we utilize contrastive learning to learn a representation space and a similarity measure such that in the learned representation space close individuals identified by the learned similarity measure have similar potential outcomes. This property ensures reliable imputation of counterfactual outcomes for the individuals with close neighbors from the alternative treatment group. By augmenting the original dataset with these reliable imputations, we can effectively reduce the discrepancy between different treatment groups, while inducing minimal imputation error. The augmented dataset is subsequently employed to train CATE estimation models. Theoretical analysis and experimental studies on synthetic and semi-synthetic benchmarks demonstrate that our method achieves significant improvements in both performance and robustness to overfitting across state-of-the-art models.

{{</citation>}}


## cs.IR (3)



### (121/144) Exploring Recommendation Capabilities of GPT-4V(ision): A Preliminary Case Study (Peilin Zhou et al., 2023)

{{<citation>}}

Peilin Zhou, Meng Cao, You-Liang Huang, Qichen Ye, Peiyan Zhang, Junling Liu, Yueqi Xie, Yining Hua, Jaeboum Kim. (2023)  
**Exploring Recommendation Capabilities of GPT-4V(ision): A Preliminary Case Study**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.04199v1)  

---


**ABSTRACT**  
Large Multimodal Models (LMMs) have demonstrated impressive performance across various vision and language tasks, yet their potential applications in recommendation tasks with visual assistance remain unexplored. To bridge this gap, we present a preliminary case study investigating the recommendation capabilities of GPT-4V(ison), a recently released LMM by OpenAI. We construct a series of qualitative test samples spanning multiple domains and employ these samples to assess the quality of GPT-4V's responses within recommendation scenarios. Evaluation results on these test samples prove that GPT-4V has remarkable zero-shot recommendation abilities across diverse domains, thanks to its robust visual-text comprehension capabilities and extensive general knowledge. However, we have also identified some limitations in using GPT-4V for recommendations, including a tendency to provide similar responses when given similar inputs. This report concludes with an in-depth discussion of the challenges and research opportunities associated with utilizing GPT-4V in recommendation scenarios. Our objective is to explore the potential of extending LMMs from vision and language tasks to recommendation tasks. We hope to inspire further research into next-generation multimodal generative recommendation models, which can enhance user experiences by offering greater diversity and interactivity. All images and prompts used in this report will be accessible at https://github.com/PALIN2018/Evaluate_GPT-4V_Rec.

{{</citation>}}


### (122/144) OLaLa: Ontology Matching with Large Language Models (Sven Hertling et al., 2023)

{{<citation>}}

Sven Hertling, Heiko Paulheim. (2023)  
**OLaLa: Ontology Matching with Large Language Models**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2311.03837v1)  

---


**ABSTRACT**  
Ontology (and more generally: Knowledge Graph) Matching is a challenging task where information in natural language is one of the most important signals to process. With the rise of Large Language Models, it is possible to incorporate this knowledge in a better way into the matching pipeline. A number of decisions still need to be taken, e.g., how to generate a prompt that is useful to the model, how information in the KG can be formulated in prompts, which Large Language Model to choose, how to provide existing correspondences to the model, how to generate candidates, etc. In this paper, we present a prototype that explores these questions by applying zero-shot and few-shot prompting with multiple open Large Language Models to different tasks of the Ontology Alignment Evaluation Initiative (OAEI). We show that with only a handful of examples and a well-designed prompt, it is possible to achieve results that are en par with supervised matching systems which use a much larger portion of the ground truth.

{{</citation>}}


### (123/144) Large Language Model based Long-tail Query Rewriting in Taobao Search (Wenjun Peng et al., 2023)

{{<citation>}}

Wenjun Peng, Guiyang Li, Yue Jiang, Zilong Wang, Dan Ou, Xiaoyi Zeng, Tongxu, Enhong Chen. (2023)  
**Large Language Model based Long-tail Query Rewriting in Taobao Search**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.03758v1)  

---


**ABSTRACT**  
In the realm of e-commerce search, the significance of semantic matching cannot be overstated, as it directly impacts both user experience and company revenue. Query rewriting serves as an important technique to bridge semantic gaps inherent in the semantic matching process. However, existing query rewriting methods often struggle to effectively optimize long-tail queries and alleviate the phenomenon of \textit{``\nothing''} caused by semantic gap. In this paper, we present \textbf{\method}, a comprehensive framework that \textbf{B}ridges the s\textbf{E}mantic gap for long-tail \textbf{QUE}ries. \method comprises three stages: multi-instruction supervised fine tuning (SFT), offline feedback, and objective alignment. Specifically, we first construct a rewriting dataset based on rejection sampling, and mix it with multiple auxiliary tasks data to fine tune our large language model (LLM) in a supervised fashion during the first stage. Subsequently, with the well-trained LLM, we employ beam search to generate multiple candidate rewrites, which would be fed into Taobao offline system to simulate the retrieval process and obtain the partial order. Leveraging the partial order of candidate rewrites, we introduce a contrastive learning method to highlight the distinctions between rewrites and align the model with the Taobao online objectives. Offline experiments prove the effectiveness of our method in enhancing retrieval performance. Online A/B tests reveal that our method can significantly boost gross merchandise volume (GMV), number of transaction (\#Trans) and unique visitor (UV) for long-tail queries. \method has been deployed on Taobao, one of most popular online shopping platforms in China, since October 2023.

{{</citation>}}


## cs.CR (3)



### (124/144) Quantization-aware Neural Architectural Search for Intrusion Detection (Rabin Yu Acharya et al., 2023)

{{<citation>}}

Rabin Yu Acharya, Laurens Le Jeune, Nele Mentens, Fatemeh Ganji, Domenic Forte. (2023)  
**Quantization-aware Neural Architectural Search for Intrusion Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Intrusion Detection, Quantization  
[Paper Link](http://arxiv.org/abs/2311.04194v1)  

---


**ABSTRACT**  
Deploying machine learning-based intrusion detection systems (IDSs) on hardware devices is challenging due to their limited computational resources, power consumption, and network connectivity. Hence, there is a significant need for robust, deep learning models specifically designed with such constraints in mind. In this paper, we present a design methodology that automatically trains and evolves quantized neural network (NN) models that are a thousand times smaller than state-of-the-art NNs but can efficiently analyze network data for intrusion at high accuracy. In this regard, the number of LUTs utilized by this network when deployed to an FPGA is between 2.3x and 8.5x smaller with performance comparable to prior work.

{{</citation>}}


### (125/144) IC-SECURE: Intelligent System for Assisting Security Experts in Generating Playbooks for Automated Incident Response (Ryuta Kremer et al., 2023)

{{<citation>}}

Ryuta Kremer, Prasanna N. Wudali, Satoru Momiyama, Toshinori Araki, Jun Furukawa, Yuval Elovici, Asaf Shabtai. (2023)  
**IC-SECURE: Intelligent System for Assisting Security Experts in Generating Playbooks for Automated Incident Response**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.03825v1)  

---


**ABSTRACT**  
Security orchestration, automation, and response (SOAR) systems ingest alerts from security information and event management (SIEM) system, and then trigger relevant playbooks that automate and orchestrate the execution of a sequence of security activities. SOAR systems have two major limitations: (i) security analysts need to define, create and change playbooks manually, and (ii) the choice between multiple playbooks that could be triggered is based on rules defined by security analysts. To address these limitations, recent studies in the field of artificial intelligence for cybersecurity suggested the task of interactive playbook creation. In this paper, we propose IC-SECURE, an interactive playbook creation solution based on a novel deep learning-based approach that provides recommendations to security analysts during the playbook creation process. IC-SECURE captures the context in the form of alert data and current status of incomplete playbook, required to make reasonable recommendation for next module that should be included in the new playbook being created. We created three evaluation datasets, each of which involved a combination of a set of alert rules and a set of playbooks from a SOAR platform. We evaluated IC-SECURE under various settings, and compared our results with two state-of-the-art recommender system methods. In our evaluation IC-SECURE demonstrated superior performance compared to other methods by consistently recommending the correct security module, achieving precision@1 > 0.8 and recall@3 > 0.92

{{</citation>}}


### (126/144) SoK: Security Below the OS -- A Security Analysis of UEFI (Priyanka Prakash Surve et al., 2023)

{{<citation>}}

Priyanka Prakash Surve, Oleg Brodt, Mark Yampolskiy, Yuval Elovici, Asaf Shabtai. (2023)  
**SoK: Security Below the OS -- A Security Analysis of UEFI**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.03809v1)  

---


**ABSTRACT**  
The Unified Extensible Firmware Interface (UEFI) is a linchpin of modern computing systems, governing secure system initialization and booting. This paper is urgently needed because of the surge in UEFI-related attacks and vulnerabilities in recent years. Motivated by this urgent concern, we undertake an extensive exploration of the UEFI landscape, dissecting its distribution supply chain, booting process, and security features. We carefully study a spectrum of UEFI-targeted attacks and proofs of concept (PoCs) for exploiting UEFI-related vulnerabilities. Building upon these insights, we construct a comprehensive attack threat model encompassing threat actors, attack vectors, attack types, vulnerabilities, attack capabilities, and attacker objectives. Drawing inspiration from the MITRE ATT&CK framework, we present a MITRE ATT&CK-like taxonomy delineating tactics, techniques, and sub-techniques in the context of UEFI attacks. This taxonomy can provide a road map for identifying existing gaps and developing new techniques for rootkit prevention, detection, and removal. Finally, the paper discusses existing countermeasures against UEFI attacks including a variety of technical and operational measures that can be implemented to lower the risk of UEFI attacks to an acceptable level. This paper seeks to clarify the complexities of UEFI and equip the cybersecurity community with the necessary knowledge to strengthen the security of this critical component against a growing threat landscape.

{{</citation>}}


## math.NA (1)



### (127/144) Formulating and Heuristic Solving of Contact Problems in Hybrid Data-Driven Computational Mechanics (Cristian Guillermo Gebhardt et al., 2023)

{{<citation>}}

Cristian Guillermo Gebhardt, Senta Lange, Marc Christian Steinbach. (2023)  
**Formulating and Heuristic Solving of Contact Problems in Hybrid Data-Driven Computational Mechanics**  

---
Primary Category: math.NA  
Categories: 74B20, 74M15, 90C30, 90C33, 90C59, cs-NA, math-NA, math.NA  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.04083v1)  

---


**ABSTRACT**  
In this work we consider the hybrid Data-Driven Computational Mechanics (DDCM) approach, in which a smooth constitutive manifold is reconstructed to obtain a well-behaved nonlinear optimization problem (NLP) rather than the much harder discrete-continous NLP (DCNLP) of the direct DDCM approach. The key focus is on the addition of geometric inequality constraints to the hybrid DDCM formulation. Therein, the required constraint force leads to a contact problem in the form of a mathematical program with complementarity constraints (MPCC), a problem class that is still less complex than the DCNLP. For this MPCC we propose a heuristic quick-shot solution approach, which can produce verifiable solutions by solving up to four NLPs. We perform various numerical experiments on three different contact problems of increasing difficulty to demonstrate the potential and limitations of this approach.

{{</citation>}}


## physics.chem-ph (1)



### (128/144) Extracting human interpretable structure-property relationships in chemistry using XAI and large language models (Geemi P. Wellawatte et al., 2023)

{{<citation>}}

Geemi P. Wellawatte, Philippe Schwaller. (2023)  
**Extracting human interpretable structure-property relationships in chemistry using XAI and large language models**  

---
Primary Category: physics.chem-ph  
Categories: cs-LG, physics-chem-ph, physics.chem-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.04047v1)  

---


**ABSTRACT**  
Explainable Artificial Intelligence (XAI) is an emerging field in AI that aims to address the opaque nature of machine learning models. Furthermore, it has been shown that XAI can be used to extract input-output relationships, making them a useful tool in chemistry to understand structure-property relationships. However, one of the main limitations of XAI methods is that they are developed for technically oriented users. We propose the XpertAI framework that integrates XAI methods with large language models (LLMs) accessing scientific literature to generate accessible natural language explanations of raw chemical data automatically. We conducted 5 case studies to evaluate the performance of XpertAI. Our results show that XpertAI combines the strengths of LLMs and XAI tools in generating specific, scientific, and interpretable explanations.

{{</citation>}}


## cs.AI (6)



### (129/144) A Method to Improve the Performance of Reinforcement Learning Based on the Y Operator for a Class of Stochastic Differential Equation-Based Child-Mother Systems (Cheng Yin et al., 2023)

{{<citation>}}

Cheng Yin, Yi Chen. (2023)  
**A Method to Improve the Performance of Reinforcement Learning Based on the Y Operator for a Class of Stochastic Differential Equation-Based Child-Mother Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.04014v1)  

---


**ABSTRACT**  
This paper introduces a novel operator, termed the Y operator, to elevate control performance in Actor-Critic(AC) based reinforcement learning for systems governed by stochastic differential equations(SDEs). The Y operator ingeniously integrates the stochasticity of a class of child-mother system into the Critic network's loss function, yielding substantial advancements in the control performance of RL algorithms.Additionally, the Y operator elegantly reformulates the challenge of solving partial differential equations for the state-value function into a parallel problem for the drift and diffusion functions within the system's SDEs.A rigorous mathematical proof confirms the operator's validity.This transformation enables the Y Operator-based Reinforcement Learning(YORL) framework to efficiently tackle optimal control problems in both model-based and data-driven systems.The superiority of YORL is demonstrated through linear and nonlinear numerical examples showing its enhanced performance over existing methods post convergence.

{{</citation>}}


### (130/144) Everything of Thoughts: Defying the Law of Penrose Triangle for Thought Generation (Ruomeng Ding et al., 2023)

{{<citation>}}

Ruomeng Ding, Chaoyun Zhang, Lu Wang, Yong Xu, Minghua Ma, Wei Zhang, Si Qin, Saravan Rajmohan, Qingwei Lin, Dongmei Zhang. (2023)  
**Everything of Thoughts: Defying the Law of Penrose Triangle for Thought Generation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.04254v1)  

---


**ABSTRACT**  
Recent advancements in Large Language Models (LLMs) have revolutionized decision-making by breaking down complex problems into more manageable language sequences referred to as ``thoughts''. An effective thought design should consider three key perspectives: performance, efficiency, and flexibility. However, existing thought can at most exhibit two of these attributes. To address these limitations, we introduce a novel thought prompting approach called ``Everything of Thoughts'' (XoT) to defy the law of ``Penrose triangle of existing thought paradigms. XoT leverages pretrained reinforcement learning and Monte Carlo Tree Search (MCTS) to incorporate external domain knowledge into thoughts, thereby enhancing LLMs' capabilities and enabling them to generalize to unseen problems efficiently. Through the utilization of the MCTS-LLM collaborative thought revision framework, this approach autonomously produces high-quality comprehensive cognitive mappings with minimal LLM interactions. Additionally, XoT empowers LLMs to engage in unconstrained thinking, allowing for flexible cognitive mappings for problems with multiple solutions.

{{</citation>}}


### (131/144) Unifying Structure and Language Semantic for Efficient Contrastive Knowledge Graph Completion with Structured Entity Anchors (Sang-Hyun Je et al., 2023)

{{<citation>}}

Sang-Hyun Je, Wontae Choi, Kwangjin Oh. (2023)  
**Unifying Structure and Language Semantic for Efficient Contrastive Knowledge Graph Completion with Structured Entity Anchors**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2311.04250v1)  

---


**ABSTRACT**  
The goal of knowledge graph completion (KGC) is to predict missing links in a KG using trained facts that are already known. In recent, pre-trained language model (PLM) based methods that utilize both textual and structural information are emerging, but their performances lag behind state-of-the-art (SOTA) structure-based methods or some methods lose their inductive inference capabilities in the process of fusing structure embedding to text encoder. In this paper, we propose a novel method to effectively unify structure information and language semantics without losing the power of inductive reasoning. We adopt entity anchors and these anchors and textual description of KG elements are fed together into the PLM-based encoder to learn unified representations. In addition, the proposed method utilizes additional random negative samples which can be reused in the each mini-batch during contrastive learning to learn a generalized entity representations. We verify the effectiveness of the our proposed method through various experiments and analysis. The experimental results on standard benchmark widely used in link prediction task show that the proposed model outperforms existing the SOTA KGC models. Especially, our method show the largest performance improvement on FB15K-237, which is competitive to the SOTA of structure-based KGC methods.

{{</citation>}}


### (132/144) Scene-Driven Multimodal Knowledge Graph Construction for Embodied AI (Song Yaoxian et al., 2023)

{{<citation>}}

Song Yaoxian, Sun Penglei, Liu Haoyu, Li Zhixu, Song Wei, Xiao Yanghua, Zhou Xiaofang. (2023)  
**Scene-Driven Multimodal Knowledge Graph Construction for Embodied AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-RO, cs-SC, cs.AI  
Keywords: AI, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2311.03783v1)  

---


**ABSTRACT**  
Embodied AI is one of the most popular studies in artificial intelligence and robotics, which can effectively improve the intelligence of real-world agents (i.e. robots) serving human beings. Scene knowledge is important for an agent to understand the surroundings and make correct decisions in the varied open world. Currently, knowledge base for embodied tasks is missing and most existing work use general knowledge base or pre-trained models to enhance the intelligence of an agent. For conventional knowledge base, it is sparse, insufficient in capacity and cost in data collection. For pre-trained models, they face the uncertainty of knowledge and hard maintenance. To overcome the challenges of scene knowledge, we propose a scene-driven multimodal knowledge graph (Scene-MMKG) construction method combining conventional knowledge engineering and large language models. A unified scene knowledge injection framework is introduced for knowledge representation. To evaluate the advantages of our proposed method, we instantiate Scene-MMKG considering typical indoor robotic functionalities (Manipulation and Mobility), named ManipMob-MMKG. Comparisons in characteristics indicate our instantiated ManipMob-MMKG has broad superiority in data-collection efficiency and knowledge quality. Experimental results on typical embodied tasks show that knowledge-enhanced methods using our instantiated ManipMob-MMKG can improve the performance obviously without re-designing model structures complexly. Our project can be found at https://sites.google.com/view/manipmob-mmkg

{{</citation>}}


### (133/144) The NeurIPS 2022 Neural MMO Challenge: A Massively Multiagent Competition with Specialization and Trade (Enhong Liu et al., 2023)

{{<citation>}}

Enhong Liu, Joseph Suarez, Chenhui You, Bo Wu, Bingcheng Chen, Jun Hu, Jiaxin Chen, Xiaolong Zhu, Clare Zhu, Julian Togelius, Sharada Mohanty, Weijun Hong, Rui Du, Yibing Zhang, Qinwen Wang, Xinhang Li, Zheng Yuan, Xiang Li, Yuejia Huang, Kun Zhang, Hanhui Yang, Shiqi Tang, Phillip Isola. (2023)  
**The NeurIPS 2022 Neural MMO Challenge: A Massively Multiagent Competition with Specialization and Trade**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-MA, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.03707v1)  

---


**ABSTRACT**  
In this paper, we present the results of the NeurIPS-2022 Neural MMO Challenge, which attracted 500 participants and received over 1,600 submissions. Like the previous IJCAI-2022 Neural MMO Challenge, it involved agents from 16 populations surviving in procedurally generated worlds by collecting resources and defeating opponents. This year's competition runs on the latest v1.6 Neural MMO, which introduces new equipment, combat, trading, and a better scoring system. These elements combine to pose additional robustness and generalization challenges not present in previous competitions. This paper summarizes the design and results of the challenge, explores the potential of this environment as a benchmark for learning methods, and presents some practical reinforcement learning training approaches for complex tasks with sparse rewards. Additionally, we have open-sourced our baselines, including environment wrappers, benchmarks, and visualization tools for future research.

{{</citation>}}


### (134/144) Hypothesis Network Planned Exploration for Rapid Meta-Reinforcement Learning Adaptation (Maxwell Joseph Jacobson et al., 2023)

{{<citation>}}

Maxwell Joseph Jacobson, Yexiang Xue. (2023)  
**Hypothesis Network Planned Exploration for Rapid Meta-Reinforcement Learning Adaptation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.03701v1)  

---


**ABSTRACT**  
Meta Reinforcement Learning (Meta RL) trains agents that adapt to fast-changing environments and tasks. Current strategies often lose adaption efficiency due to the passive nature of model exploration, causing delayed understanding of new transition dynamics. This results in particularly fast-evolving tasks being impossible to solve. We propose a novel approach, Hypothesis Network Planned Exploration (HyPE), that integrates an active and planned exploration process via the hypothesis network to optimize adaptation speed. HyPE uses a generative hypothesis network to form potential models of state transition dynamics, then eliminates incorrect models through strategically devised experiments. Evaluated on a symbolic version of the Alchemy game, HyPE outpaces baseline methods in adaptation speed and model accuracy, validating its potential in enhancing reinforcement learning adaptation in rapidly evolving settings.

{{</citation>}}


## cs.SE (2)



### (135/144) Multimodal extended reality applications offer benefits for volumetric biomedical image analysis in research and medicine (Kathrin Krieger et al., 2023)

{{<citation>}}

Kathrin Krieger, Jan Egger, Jens Kleesiek, Matthias Gunzer, Jianxu Chen. (2023)  
**Multimodal extended reality applications offer benefits for volumetric biomedical image analysis in research and medicine**  

---
Primary Category: cs.SE  
Categories: cs-GR, cs-HC, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.03986v1)  

---


**ABSTRACT**  
3D data from high-resolution volumetric imaging is a central resource for diagnosis and treatment in modern medicine. While the fast development of AI enhances imaging and analysis, commonly used visualization methods lag far behind. Recent research used extended reality (XR) for perceiving 3D images with visual depth perception and touch, but used restricting haptic devices. While unrestricted touch is beneficial for volumetric data examination, implementing natural haptic interaction with XR is challenging. The research question is whether a multimodal XR application with intutitive haptic interaction adds value and should be pursued. In a study, 24 expterts for biomedical images in research and medicine. explored 3D anatomical medical shapes with 3 applications: a multimodal virtual reality (VR) prototype using haptic gloves, a simple VR prototype using VR controllers, and a commonly used standard PC application. Results of the standardized questionnaires showed no significant differences between the three application types regarding usability and no significant difference between both VR applications regarding presence. Participants agreed to statements that VR visualizations provide better depth information, that using the hands instead of controllers simplifies data exploration, that the multimodal VR prototype allows intuitive data exploration, and that it is beneficial over traditional data examination methods. While most participants mentioned the manual interaction as best aspect, they also found it the most improvable. We conclude that a multimodal XR application with improved manual interaction adds value for volumetric biomedical data examination. We will proceed with our open-source research project ISH3DE (Intuitive Stereoptic Haptic 3D Data Exploration) to serve medical education, therapeutic decisions, surgery preparations, or research data analysis.

{{</citation>}}


### (136/144) Requirements Engineering using Generative AI: Prompts and Prompting Patterns (Krishna Ronanki et al., 2023)

{{<citation>}}

Krishna Ronanki, Beatriz Cabrero-Daniel, Jennifer Horkoff, Christian Berger. (2023)  
**Requirements Engineering using Generative AI: Prompts and Prompting Patterns**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, GPT, GPT-3.5, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.03832v1)  

---


**ABSTRACT**  
[Context]: Companies are increasingly recognizing the importance of automating Requirements Engineering (RE) tasks due to their resource-intensive nature. The advent of GenAI has made these tasks more amenable to automation, thanks to its ability to understand and interpret context effectively. [Problem]: However, in the context of GenAI, prompt engineering is a critical factor for success. Despite this, we currently lack tools and methods to systematically assess and determine the most effective prompt patterns to employ for a particular RE task. [Method]: Two tasks related to requirements, specifically requirement classification and tracing, were automated using the GPT-3.5 turbo API. The performance evaluation involved assessing various prompts created using 5 prompt patterns and implemented programmatically to perform the selected RE tasks, focusing on metrics such as precision, recall, accuracy, and F-Score. [Results]: This paper evaluates the effectiveness of the 5 prompt patterns' ability to make GPT-3.5 turbo perform the selected RE tasks and offers recommendations on which prompt pattern to use for a specific RE task. Additionally, it also provides an evaluation framework as a reference for researchers and practitioners who want to evaluate different prompt patterns for different RE tasks.

{{</citation>}}


## eess.SP (1)



### (137/144) Blind Federated Learning via Over-the-Air q-QAM (Saeed Razavikia et al., 2023)

{{<citation>}}

Saeed Razavikia, José Mairton Barros Da Silva Júnior, Carlo Fischione. (2023)  
**Blind Federated Learning via Over-the-Air q-QAM**  

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.04253v1)  

---


**ABSTRACT**  
In this work, we investigate federated edge learning over a fading multiple access channel. To alleviate the communication burden between the edge devices and the access point, we introduce a pioneering digital over-the-air computation strategy employing q-ary quadrature amplitude modulation, culminating in a low latency communication scheme. Indeed, we propose a new federated edge learning framework in which edge devices use digital modulation for over-the-air uplink transmission to the edge server while they have no access to the channel state information. Furthermore, we incorporate multiple antennas at the edge server to overcome the fading inherent in wireless communication. We analyze the number of antennas required to mitigate the fading impact effectively. We prove a non-asymptotic upper bound for the mean squared error for the proposed federated learning with digital over-the-air uplink transmissions under both noisy and fading conditions. Leveraging the derived upper bound, we characterize the convergence rate of the learning process of a non-convex loss function in terms of the mean square error of gradients due to the fading channel. Furthermore, we substantiate the theoretical assurances through numerical experiments concerning mean square error and the convergence efficacy of the digital federated edge learning framework. Notably, the results demonstrate that augmenting the number of antennas at the edge server and adopting higher-order modulations improve the model accuracy up to 60\%.

{{</citation>}}


## cs.NI (2)



### (138/144) On Deep Reinforcement Learning for Traffic Steering Intelligent ORAN (Fatemeh Kavehmadavani et al., 2023)

{{<citation>}}

Fatemeh Kavehmadavani, Van-Dinh Nguyen, Thang X. Vu, Symeon Chatzinotas. (2023)  
**On Deep Reinforcement Learning for Traffic Steering Intelligent ORAN**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.03853v1)  

---


**ABSTRACT**  
This paper aims to develop the intelligent traffic steering (TS) framework, which has recently been considered as one of the key developments of 3GPP for advanced 5G. Since achieving key performance indicators (KPIs) for heterogeneous services may not be possible in the monolithic architecture, a novel deep reinforcement learning (DRL)-based TS algorithm is proposed at the non-real-time (non-RT) RAN intelligent controller (RIC) within the open radio access network (ORAN) architecture. To enable ORAN's intelligence, we distribute traffic load onto appropriate paths, which helps efficiently allocate resources to end users in a downlink multi-service scenario. Our proposed approach employs a three-step hierarchical process that involves heuristics, machine learning, and convex optimization to steer traffic flows. Through system-level simulations, we show the superior performance of the proposed intelligent TS scheme, surpassing established benchmark systems by 45.50%.

{{</citation>}}


### (139/144) Integrated Sensing, Communication, and Computing for Cost-effective Multimodal Federated Perception (Ning Chen et al., 2023)

{{<citation>}}

Ning Chen, Zhipeng Cheng, Xuwei Fan, Bangzhen Huang, Yifeng Zhao, Lianfen Huang, Xiaojiang Du, Mohsen Guizani. (2023)  
**Integrated Sensing, Communication, and Computing for Cost-effective Multimodal Federated Perception**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.03815v1)  

---


**ABSTRACT**  
Federated learning (FL) is a classic paradigm of 6G edge intelligence (EI), which alleviates privacy leaks and high communication pressure caused by traditional centralized data processing in the artificial intelligence of things (AIoT). The implementation of multimodal federated perception (MFP) services involves three sub-processes, including sensing-based multimodal data generation, communication-based model transmission, and computing-based model training, ultimately relying on available underlying multi-domain physical resources such as time, frequency, and computing power. How to reasonably coordinate the multi-domain resources scheduling among sensing, communication, and computing, therefore, is crucial to the MFP networks. To address the above issues, this paper investigates service-oriented resource management with integrated sensing, communication, and computing (ISCC). With the incentive mechanism of the MFP service market, the resources management problem is redefined as a social welfare maximization problem, where the idea of "expanding resources" and "reducing costs" is used to improve learning performance gain and reduce resource costs. Experimental results demonstrate the effectiveness and robustness of the proposed resource scheduling mechanisms.

{{</citation>}}


## cs.FL (1)



### (140/144) Leveraging Large Language Models for Automated Proof Synthesis in Rust (Jianan Yao et al., 2023)

{{<citation>}}

Jianan Yao, Ziqiao Zhou, Weiteng Chen, Weidong Cui. (2023)  
**Leveraging Large Language Models for Automated Proof Synthesis in Rust**  

---
Primary Category: cs.FL  
Categories: cs-AI, cs-FL, cs.FL  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.03739v1)  

---


**ABSTRACT**  
Formal verification can provably guarantee the correctness of critical system software, but the high proof burden has long hindered its wide adoption. Recently, Large Language Models (LLMs) have shown success in code analysis and synthesis. In this paper, we present a combination of LLMs and static analysis to synthesize invariants, assertions, and other proof structures for a Rust-based formal verification framework called Verus. In a few-shot setting, LLMs demonstrate impressive logical ability in generating postconditions and loop invariants, especially when analyzing short code snippets. However, LLMs lack the ability to retain and propagate context information, a strength of traditional static analysis. Based on these observations, we developed a prototype based on OpenAI's GPT-4 model. Our prototype decomposes the verification task into multiple smaller ones, iteratively queries GPT-4, and combines its output with lightweight static analysis. We evaluated the prototype with a developer in the automation loop on 20 vector-manipulating programs. The results demonstrate that it significantly reduces human effort in writing entry-level proof code.

{{</citation>}}


## quant-ph (1)



### (141/144) Generalized Hybrid Search and Applications to Blockchain and Hash Function Security (Alexandru Cojocaru et al., 2023)

{{<citation>}}

Alexandru Cojocaru, Juan Garay, Fang Song. (2023)  
**Generalized Hybrid Search and Applications to Blockchain and Hash Function Security**  

---
Primary Category: quant-ph  
Categories: cs-CR, quant-ph, quant-ph  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.03723v1)  

---


**ABSTRACT**  
In this work we first examine the hardness of solving various search problems by hybrid quantum-classical strategies, namely, by algorithms that have both quantum and classical capabilities. We then construct a hybrid quantum-classical search algorithm and analyze its success probability. Regarding the former, for search problems that are allowed to have multiple solutions and in which the input is sampled according to arbitrary distributions we establish their hybrid quantum-classical query complexities -- i.e., given a fixed number of classical and quantum queries, determine what is the probability of solving the search task. At a technical level, our results generalize the framework for hybrid quantum-classical search algorithms proposed by Rosmanis. Namely, for an arbitrary distribution $D$ on Boolean functions, the probability an algorithm equipped with $\tau_c$ classical and $\tau_q$ quantum queries succeeds in finding a preimage of $1$ for a function sampled from $D$ is at most $\nu_D \cdot(2\sqrt{\tau_c} + 2\tau_q + 1)^2$, where $\nu_D$ captures the average (over $D$) fraction of preimages of $1$. As applications of our hardness results, we first revisit and generalize the security of the Bitcoin protocol called the Bitcoin backbone, to a setting where the adversary has both quantum and classical capabilities, presenting a new hybrid honest majority condition necessary for the protocol to properly operate. Secondly, we examine the generic security of hash functions against hybrid adversaries. Regarding our second contribution, we design a hybrid algorithm which first spends all of its classical queries and in the second stage runs a ``modified Grover'' where the initial state depends on the distribution $D$. We show how to analyze its success probability for arbitrary target distributions and, importantly, its optimality for the uniform and the Bernoulli distribution cases.

{{</citation>}}


## cs.PF (1)



### (142/144) Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models (Longteng Zhang et al., 2023)

{{<citation>}}

Longteng Zhang, Xiang Liu, Zeyu Li, Xinglin Pan, Peijie Dong, Ruibo Fan, Rui Guo, Xin Wang, Qiong Luo, Shaohuai Shi, Xiaowen Chu. (2023)  
**Dissecting the Runtime Performance of the Training, Fine-tuning, and Inference of Large Language Models**  

---
Primary Category: cs.PF  
Categories: cs-CL, cs-LG, cs-PF, cs.PF  
Keywords: Attention, Language Model  
[Paper Link](http://arxiv.org/abs/2311.03687v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have seen great advance in both academia and industry, and their popularity results in numerous open-source frameworks and techniques in accelerating LLM pre-training, fine-tuning, and inference. Training and deploying LLMs are expensive as it requires considerable computing resources and memory, hence many efficient approaches have been developed for improving system pipelines as well as operators. However, the runtime performance can vary significantly across hardware and software stacks, which makes it difficult to choose the best configuration. In this work, we aim to benchmark the performance from both macro and micro perspectives. First, we benchmark the end-to-end performance of pre-training, fine-tuning, and serving LLMs in different sizes , i.e., 7, 13, and 70 billion parameters (7B, 13B, and 70B) on three 8-GPU platforms with and without individual optimization techniques, including ZeRO, quantization, recomputation, FlashAttention. Then, we dive deeper to provide a detailed runtime analysis of the sub-modules, including computing and communication operators in LLMs. For end users, our benchmark and findings help better understand different optimization techniques, training and inference frameworks, together with hardware platforms in choosing configurations for deploying LLMs. For researchers, our in-depth module-wise analyses discover potential opportunities for future work to further optimize the runtime performance of LLMs.

{{</citation>}}


## physics.ao-ph (1)



### (143/144) Machine Learning Parameterization of the Multi-scale Kain-Fritsch (MSKF) Convection Scheme (Xiaohui Zhong et al., 2023)

{{<citation>}}

Xiaohui Zhong, Xing Yu, Hao Li. (2023)  
**Machine Learning Parameterization of the Multi-scale Kain-Fritsch (MSKF) Convection Scheme**  

---
Primary Category: physics.ao-ph  
Categories: cs-AI, physics-ao-ph, physics.ao-ph  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.03652v1)  

---


**ABSTRACT**  
Warm-sector heavy rainfall often occurs along the coast of South China, and it is usually localized and long-lasting, making it challenging to predict. High-resolution numerical weather prediction (NWP) models are increasingly used to better resolve topographic features and forecast such high-impact weather events. However, when the grid spacing becomes comparable to the length scales of convection, known as the gray zone, the turbulent eddies in the atmospheric boundary layer are only partially resolved and parameterized to some extent. Whether using a convection parameterization (CP) scheme in the gray zone remains controversial. Scale-aware CP schemes are developed to enhance the representation of convective transport within the gray zone. The multi-scale Kain-Fritsch (MSKF) scheme includes modifications that allow for its effective implementation at a grid resolution as high as 2 km. In recent years, there has been an increasing application of machine learning (ML) models to various domains of atmospheric sciences, including the replacement of physical parameterizations with ML models. This work proposes a multi-output bidirectional long short-term memory (Bi-LSTM) model as a replace the scale-aware MSKF CP scheme. The Weather Research and Forecast (WRF) model is used to generate training and testing data over South China at a horizontal resolution of 5 km. Furthermore, the WRF model is coupled with the ML based CP scheme and compared with WRF simulations with original MSKF scheme. The results demonstrate that the Bi-LSTM model can achieve high accuracy, indicating the potential use of ML models to substitute the MSKF scheme in the gray zone.

{{</citation>}}


## cs.CG (1)



### (144/144) Minimizing Pentagons in the Plane through Automated Reasoning (Bernardo Subercaseaux et al., 2023)

{{<citation>}}

Bernardo Subercaseaux, John Mackey, Marijn J. H. Heule, Ruben Martins. (2023)  
**Minimizing Pentagons in the Plane through Automated Reasoning**  

---
Primary Category: cs.CG  
Categories: cs-CG, cs-DM, cs.CG  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2311.03645v1)  

---


**ABSTRACT**  
We study $\mu_5(n)$, the minimum number of convex pentagons induced by $n$ points in the plane in general position. Despite a significant body of research in understanding $\mu_4(n)$, the variant concerning convex quadrilaterals, not much is known about $\mu_5(n)$. We present two explicit constructions, inspired by point placements obtained through a combination of Stochastic Local Search and a program for realizability of point sets, that provide $\mu_5(n) \leq \binom{\lfloor n/2 \rfloor}{5} + \binom{\lceil n/2 \rceil}{5}$. Furthermore, we conjecture this bound to be optimal, and provide partial evidence by leveraging a MaxSAT encoding that allows us to verify our conjecture for $n \leq 16$.

{{</citation>}}
