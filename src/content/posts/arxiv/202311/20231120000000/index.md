---
draft: false
title: "arXiv @ 2023.11.20"
date: 2023-11-20
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.20"
    identifier: arxiv_20231120
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (15)](#cscv-15)
- [cs.SE (3)](#csse-3)
- [cs.AI (8)](#csai-8)
- [cs.LG (8)](#cslg-8)
- [cs.CL (14)](#cscl-14)
- [cs.SI (3)](#cssi-3)
- [cs.DL (1)](#csdl-1)
- [cs.NI (1)](#csni-1)
- [eess.IV (3)](#eessiv-3)
- [cs.IR (2)](#csir-2)
- [cond-mat.mtrl-sci (1)](#cond-matmtrl-sci-1)
- [cs.HC (2)](#cshc-2)
- [cs.CR (1)](#cscr-1)

## cs.CV (15)



### (1/62) Active Prompt Learning in Vision Language Models (Jihwan Bang et al., 2023)

{{<citation>}}

Jihwan Bang, Sumyeong Ahn, Jae-Gil Lee. (2023)  
**Active Prompt Learning in Vision Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11178v1)  

---


**ABSTRACT**  
Pre-trained Vision Language Models (VLMs) have demonstrated notable progress in various zero-shot tasks, such as classification and retrieval. Despite their performance, because improving performance on new tasks requires task-specific knowledge, their adaptation is essential. While labels are needed for the adaptation, acquiring them is typically expensive. To overcome this challenge, active learning, a method of achieving a high performance by obtaining labels for a small number of samples from experts, has been studied. Active learning primarily focuses on selecting unlabeled samples for labeling and leveraging them to train models. In this study, we pose the question, "how can the pre-trained VLMs be adapted under the active learning framework?" In response to this inquiry, we observe that (1) simply applying a conventional active learning framework to pre-trained VLMs even may degrade performance compared to random selection because of the class imbalance in labeling candidates, and (2) the knowledge of VLMs can provide hints for achieving the balance before labeling. Based on these observations, we devise a novel active learning framework for VLMs, denoted as PCB. To assess the effectiveness of our approach, we conduct experiments on seven different real-world datasets, and the results demonstrate that PCB surpasses conventional active learning and random sampling methods.

{{</citation>}}


### (2/62) Security Fence Inspection at Airports Using Object Detection (Nils Friederich et al., 2023)

{{<citation>}}

Nils Friederich, Andreas Specker, Jürgen Beyerer. (2023)  
**Security Fence Inspection at Airports Using Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Security  
[Paper Link](http://arxiv.org/abs/2311.12064v1)  

---


**ABSTRACT**  
To ensure the security of airports, it is essential to protect the airside from unauthorized access. For this purpose, security fences are commonly used, but they require regular inspection to detect damages. However, due to the growing shortage of human specialists and the large manual effort, there is the need for automated methods. The aim is to automatically inspect the fence for damage with the help of an autonomous robot. In this work, we explore object detection methods to address the fence inspection task and localize various types of damages. In addition to evaluating four State-of-the-Art (SOTA) object detection models, we analyze the impact of several design criteria, aiming at adapting to the task-specific challenges. This includes contrast adjustment, optimization of hyperparameters, and utilization of modern backbones. The experimental results indicate that our optimized You Only Look Once v5 (YOLOv5) model achieves the highest accuracy of the four methods with an increase of 6.9% points in Average Precision (AP) compared to the baseline. Moreover, we show the real-time capability of the model. The trained models are published on GitHub: https://github.com/N-Friederich/airport_fence_inspection.

{{</citation>}}


### (3/62) Mitigating Exposure Bias in Discriminator Guided Diffusion Models (Eleftherios Tsonis et al., 2023)

{{<citation>}}

Eleftherios Tsonis, Paraskevi Tzouveli, Athanasios Voulodimos. (2023)  
**Mitigating Exposure Bias in Discriminator Guided Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.11164v1)  

---


**ABSTRACT**  
Diffusion Models have demonstrated remarkable performance in image generation. However, their demanding computational requirements for training have prompted ongoing efforts to enhance the quality of generated images through modifications in the sampling process. A recent approach, known as Discriminator Guidance, seeks to bridge the gap between the model score and the data score by incorporating an auxiliary term, derived from a discriminator network. We show that despite significantly improving sample quality, this technique has not resolved the persistent issue of Exposure Bias and we propose SEDM-G++, which incorporates a modified sampling approach, combining Discriminator Guidance and Epsilon Scaling. Our proposed approach outperforms the current state-of-the-art, by achieving an FID score of 1.73 on the unconditional CIFAR-10 dataset.

{{</citation>}}


### (4/62) Benchmarking Feature Extractors for Reinforcement Learning-Based Semiconductor Defect Localization (Enrique Dehaerne et al., 2023)

{{<citation>}}

Enrique Dehaerne, Bappaditya Dey, Sandip Halder, Stefan De Gendt. (2023)  
**Benchmarking Feature Extractors for Reinforcement Learning-Based Semiconductor Defect Localization**  

---
Primary Category: cs.CV  
Categories: I-4-9, cs-CV, cs.CV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.11145v1)  

---


**ABSTRACT**  
As semiconductor patterning dimensions shrink, more advanced Scanning Electron Microscopy (SEM) image-based defect inspection techniques are needed. Recently, many Machine Learning (ML)-based approaches have been proposed for defect localization and have shown impressive results. These methods often rely on feature extraction from a full SEM image and possibly a number of regions of interest. In this study, we propose a deep Reinforcement Learning (RL)-based approach to defect localization which iteratively extracts features from increasingly smaller regions of the input image. We compare the results of 18 agents trained with different feature extractors. We discuss the advantages and disadvantages of different feature extractors as well as the RL-based framework in general for semiconductor defect localization.

{{</citation>}}


### (5/62) Estimating Uncertainty in Landslide Segmentation Models (Savinay Nagendra et al., 2023)

{{<citation>}}

Savinay Nagendra, Chaopeng Shen, Daniel Kifer. (2023)  
**Estimating Uncertainty in Landslide Segmentation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.11138v1)  

---


**ABSTRACT**  
Landslides are a recurring, widespread hazard. Preparation and mitigation efforts can be aided by a high-quality, large-scale dataset that covers global at-risk areas. Such a dataset currently does not exist and is impossible to construct manually. Recent automated efforts focus on deep learning models for landslide segmentation (pixel labeling) from satellite imagery. However, it is also important to characterize the uncertainty or confidence levels of such segmentations. Accurate and robust uncertainty estimates can enable low-cost (in terms of manual labor) oversight of auto-generated landslide databases to resolve errors, identify hard negative examples, and increase the size of labeled training data. In this paper, we evaluate several methods for assessing pixel-level uncertainty of the segmentation. Three methods that do not require architectural changes were compared, including Pre-Threshold activations, Monte-Carlo Dropout and Test-Time Augmentation -- a method that measures the robustness of predictions in the face of data augmentation. Experimentally, the quality of the latter method was consistently higher than the others across a variety of models and metrics in our dataset.

{{</citation>}}


### (6/62) ShapeMaker: Self-Supervised Joint Shape Canonicalization, Segmentation, Retrieval and Deformation (Yan Di et al., 2023)

{{<citation>}}

Yan Di, Chenyangguang Zhang, Chaowei Wang, Ruida Zhang, Guangyao Zhai, Yanyan Li, Bowen Fu, Xiangyang Ji, Shan Gao. (2023)  
**ShapeMaker: Self-Supervised Joint Shape Canonicalization, Segmentation, Retrieval and Deformation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.11106v1)  

---


**ABSTRACT**  
In this paper, we present ShapeMaker, a unified self-supervised learning framework for joint shape canonicalization, segmentation, retrieval and deformation. Given a partially-observed object in an arbitrary pose, we first canonicalize the object by extracting point-wise affine-invariant features, disentangling inherent structure of the object with its pose and size. These learned features are then leveraged to predict semantically consistent part segmentation and corresponding part centers. Next, our lightweight retrieval module aggregates the features within each part as its retrieval token and compare all the tokens with source shapes from a pre-established database to identify the most geometrically similar shape. Finally, we deform the retrieved shape in the deformation module to tightly fit the input object by harnessing part center guided neural cage deformation. The key insight of ShapeMaker is the simultaneous training of the four highly-associated processes: canonicalization, segmentation, retrieval, and deformation, leveraging cross-task consistency losses for mutual supervision. Extensive experiments on synthetic datasets PartNet, ComplementMe, and real-world dataset Scan2CAD demonstrate that ShapeMaker surpasses competitors by a large margin. Codes will be released soon.

{{</citation>}}


### (7/62) Radiology Report Generation Using Transformers Conditioned with Non-imaging Data (Nurbanu Aksoy et al., 2023)

{{<citation>}}

Nurbanu Aksoy, Nishant Ravikumar, Alejandro F Frangi. (2023)  
**Radiology Report Generation Using Transformers Conditioned with Non-imaging Data**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.11097v1)  

---


**ABSTRACT**  
Medical image interpretation is central to most clinical applications such as disease diagnosis, treatment planning, and prognostication. In clinical practice, radiologists examine medical images and manually compile their findings into reports, which can be a time-consuming process. Automated approaches to radiology report generation, therefore, can reduce radiologist workload and improve efficiency in the clinical pathway. While recent deep-learning approaches for automated report generation from medical images have seen some success, most studies have relied on image-derived features alone, ignoring non-imaging patient data. Although a few studies have included the word-level contexts along with the image, the use of patient demographics is still unexplored. This paper proposes a novel multi-modal transformer network that integrates chest x-ray (CXR) images and associated patient demographic information, to synthesise patient-specific radiology reports. The proposed network uses a convolutional neural network to extract visual features from CXRs and a transformer-based encoder-decoder network that combines the visual features with semantic text embeddings of patient demographic information, to synthesise full-text radiology reports. Data from two public databases were used to train and evaluate the proposed approach. CXRs and reports were extracted from the MIMIC-CXR database and combined with corresponding patients' data MIMIC-IV. Based on the evaluation metrics used including patient demographic information was found to improve the quality of reports generated using the proposed approach, relative to a baseline network trained using CXRs alone. The proposed approach shows potential for enhancing radiology report generation by leveraging rich patient metadata and combining semantic text embeddings derived thereof, with medical image-derived visual features.

{{</citation>}}


### (8/62) Kuro Siwo: 12.1 billion $m^2$ under the water. A global multi-temporal satellite dataset for rapid flood mapping (Nikolaos Ioannis Bountos et al., 2023)

{{<citation>}}

Nikolaos Ioannis Bountos, Maria Sdraka, Angelos Zavras, Ilektra Karasante, Andreas Karavias, Themistocles Herekakis, Angeliki Thanasou, Dimitrios Michail, Ioannis Papoutsis. (2023)  
**Kuro Siwo: 12.1 billion $m^2$ under the water. A global multi-temporal satellite dataset for rapid flood mapping**  

---
Primary Category: cs.CV  
Categories: I-2; I-4; I-5-4, cs-AI, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.12056v1)  

---


**ABSTRACT**  
Global floods, exacerbated by climate change, pose severe threats to human life, infrastructure, and the environment. This urgency is highlighted by recent catastrophic events in Pakistan and New Zealand, underlining the critical need for precise flood mapping for guiding restoration efforts, understanding vulnerabilities, and preparing for future events. While Synthetic Aperture Radar (SAR) offers day-and-night, all-weather imaging capabilities, harnessing it for deep learning is hindered by the absence of a large annotated dataset. To bridge this gap, we introduce Kuro Siwo, a meticulously curated multi-temporal dataset, spanning 32 flood events globally. Our dataset maps more than 63 billion m2 of land, with 12.1 billion of them being either a flooded area or a permanent water body. Kuro Siwo stands out for its unparalleled annotation quality to facilitate rapid flood mapping in a supervised setting. We also augment learning by including a large unlabeled set of SAR samples, aimed at self-supervised pretraining. We provide an extensive benchmark and strong baselines for a diverse set of flood events from Europe, America, Africa and Australia. Our benchmark demonstrates the quality of Kuro Siwo annotations, training models that can achieve $\approx$ 85% and $\approx$ 87% in F1-score for flooded areas and general water detection respectively. This work calls on the deep learning community to develop solution-driven algorithms for rapid flood mapping, with the potential to aid civil protection and humanitarian agencies amid climate change challenges. Our code and data will be made available at https://github.com/Orion-AI-Lab/KuroSiwo

{{</citation>}}


### (9/62) HIDRO-VQA: High Dynamic Range Oracle for Video Quality Assessment (Shreshth Saini et al., 2023)

{{<citation>}}

Shreshth Saini, Avinab Saha, Alan C. Bovik. (2023)  
**HIDRO-VQA: High Dynamic Range Oracle for Video Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV, eess-IV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.11059v1)  

---


**ABSTRACT**  
We introduce HIDRO-VQA, a no-reference (NR) video quality assessment model designed to provide precise quality evaluations of High Dynamic Range (HDR) videos. HDR videos exhibit a broader spectrum of luminance, detail, and color than Standard Dynamic Range (SDR) videos. As HDR content becomes increasingly popular, there is a growing demand for video quality assessment (VQA) algorithms that effectively address distortions unique to HDR content. To address this challenge, we propose a self-supervised contrastive fine-tuning approach to transfer quality-aware features from the SDR to the HDR domain, utilizing unlabeled HDR videos. Our findings demonstrate that self-supervised pre-trained neural networks on SDR content can be further fine-tuned in a self-supervised setting using limited unlabeled HDR videos to achieve state-of-the-art performance on the only publicly available VQA database for HDR content, the LIVE-HDR VQA database. Moreover, our algorithm can be extended to the Full Reference VQA setting, also achieving state-of-the-art performance. Our code is available publicly at https://github.com/avinabsaha/HIDRO-VQA.

{{</citation>}}


### (10/62) Geometric Data Augmentations to Mitigate Distribution Shifts in Pollen Classification from Microscopic Images (Nam Cao et al., 2023)

{{<citation>}}

Nam Cao, Olga Saukh. (2023)  
**Geometric Data Augmentations to Mitigate Distribution Shifts in Pollen Classification from Microscopic Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Augmentation, Sketch  
[Paper Link](http://arxiv.org/abs/2311.11029v1)  

---


**ABSTRACT**  
Distribution shifts are characterized by differences between the training and test data distributions. They can significantly reduce the accuracy of machine learning models deployed in real-world scenarios. This paper explores the distribution shift problem when classifying pollen grains from microscopic images collected in the wild with a low-cost camera sensor. We leverage the domain knowledge that geometric features are highly important for accurate pollen identification and introduce two novel geometric image augmentation techniques to significantly narrow the accuracy gap between the model performance on the train and test datasets. In particular, we show that Tenengrad and ImageToSketch filters are highly effective to balance the shape and texture information while leaving out unimportant details that may confuse the model. Extensive evaluations on various model architectures demonstrate a consistent improvement of the model generalization to field data of up to 14% achieved by the geometric augmentation techniques when compared to a wide range of standard image augmentations. The approach is validated through an ablation study using pollen hydration tests to recover the shape of dry pollen grains. The proposed geometric augmentations also receive the highest scores according to the affinity and diversity measures from the literature.

{{</citation>}}


### (11/62) Boost Adversarial Transferability by Uniform Scale and Mix Mask Method (Tao Wang et al., 2023)

{{<citation>}}

Tao Wang, Zijian Ying, Qianmu Li, zhichao Lian. (2023)  
**Boost Adversarial Transferability by Uniform Scale and Mix Mask Method**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.12051v1)  

---


**ABSTRACT**  
Adversarial examples generated from surrogate models often possess the ability to deceive other black-box models, a property known as transferability. Recent research has focused on enhancing adversarial transferability, with input transformation being one of the most effective approaches. However, existing input transformation methods suffer from two issues. Firstly, certain methods, such as the Scale-Invariant Method, employ exponentially decreasing scale invariant parameters that decrease the adaptability in generating effective adversarial examples across multiple scales. Secondly, most mixup methods only linearly combine candidate images with the source image, leading to reduced features blending effectiveness. To address these challenges, we propose a framework called Uniform Scale and Mix Mask Method (US-MM) for adversarial example generation. The Uniform Scale approach explores the upper and lower boundaries of perturbation with a linear factor, minimizing the negative impact of scale copies. The Mix Mask method introduces masks into the mixing process in a nonlinear manner, significantly improving the effectiveness of mixing strategies. Ablation experiments are conducted to validate the effectiveness of each component in US-MM and explore the effect of hyper-parameters. Empirical evaluations on standard ImageNet datasets demonstrate that US-MM achieves an average of 7% better transfer attack success rate compared to state-of-the-art methods.

{{</citation>}}


### (12/62) Energizing Federated Learning via Filter-Aware Attention (Ziyuan Yang et al., 2023)

{{<citation>}}

Ziyuan Yang, Zerui Shao, Huijie Huangfu, Hui Yu, Andrew Beng Jin Teoh, Xiaoxiao Li, Hongming Shan, Yi Zhang. (2023)  
**Energizing Federated Learning via Filter-Aware Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Pruning  
[Paper Link](http://arxiv.org/abs/2311.12049v1)  

---


**ABSTRACT**  
Federated learning (FL) is a promising distributed paradigm, eliminating the need for data sharing but facing challenges from data heterogeneity. Personalized parameter generation through a hypernetwork proves effective, yet existing methods fail to personalize local model structures. This leads to redundant parameters struggling to adapt to diverse data distributions. To address these limitations, we propose FedOFA, utilizing personalized orthogonal filter attention for parameter recalibration. The core is the Two-stream Filter-aware Attention (TFA) module, meticulously designed to extract personalized filter-aware attention maps, incorporating Intra-Filter Attention (IntraFa) and Inter-Filter Attention (InterFA) streams. These streams enhance representation capability and explore optimal implicit structures for local models. Orthogonal regularization minimizes redundancy by averting inter-correlation between filters. Furthermore, we introduce an Attention-Guided Pruning Strategy (AGPS) for communication efficiency. AGPS selectively retains crucial neurons while masking redundant ones, reducing communication costs without performance sacrifice. Importantly, FedOFA operates on the server side, incurring no additional computational cost on the client, making it advantageous in communication-constrained scenarios. Extensive experiments validate superior performance over state-of-the-art approaches, with code availability upon paper acceptance.

{{</citation>}}


### (13/62) Learning Scene Context Without Images (Amirreza Rouhi et al., 2023)

{{<citation>}}

Amirreza Rouhi, David Han. (2023)  
**Learning Scene Context Without Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.10998v1)  

---


**ABSTRACT**  
Teaching machines of scene contextual knowledge would enable them to interact more effectively with the environment and to anticipate or predict objects that may not be immediately apparent in their perceptual field. In this paper, we introduce a novel transformer-based approach called $LMOD$ ( Label-based Missing Object Detection) to teach scene contextual knowledge to machines using an attention mechanism. A distinctive aspect of the proposed approach is its reliance solely on labels from image datasets to teach scene context, entirely eliminating the need for the actual image itself. We show how scene-wide relationships among different objects can be learned using a self-attention mechanism. We further show that the contextual knowledge gained from label based learning can enhance performance of other visual based object detection algorithm.

{{</citation>}}


### (14/62) Behavior Optimized Image Generation (Varun Khurana et al., 2023)

{{<citation>}}

Varun Khurana, Yaman K Singla, Jayakumar Subramanian, Rajiv Ratn Shah, Changyou Chen, Zhiqiang Xu, Balaji Krishnamurthy. (2023)  
**Behavior Optimized Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.10995v1)  

---


**ABSTRACT**  
The last few years have witnessed great success on image generation, which has crossed the acceptance thresholds of aesthetics, making it directly applicable to personal and commercial applications. However, images, especially in marketing and advertising applications, are often created as a means to an end as opposed to just aesthetic concerns. The goal can be increasing sales, getting more clicks, likes, or image sales (in the case of stock businesses). Therefore, the generated images need to perform well on these key performance indicators (KPIs), in addition to being aesthetically good. In this paper, we make the first endeavor to answer the question of "How can one infuse the knowledge of the end-goal within the image generation process itself to create not just better-looking images but also "better-performing'' images?''. We propose BoigLLM, an LLM that understands both image content and user behavior. BoigLLM knows how an image should look to get a certain required KPI. We show that BoigLLM outperforms 13x larger models such as GPT-3.5 and GPT-4 in this task, demonstrating that while these state-of-the-art models can understand images, they lack information on how these images perform in the real world. To generate actual pixels of behavior-conditioned images, we train a diffusion-based model (BoigSD) to align with a proposed BoigLLM-defined reward. We show the performance of the overall pipeline on two datasets covering two different behaviors: a stock dataset with the number of forward actions as the KPI and a dataset containing tweets with the total likes as the KPI, denoted as BoigBench. To advance research in the direction of utility-driven image generation and understanding, we release BoigBench, a benchmark dataset containing 168 million enterprise tweets with their media, brand account names, time of post, and total likes.

{{</citation>}}


### (15/62) Multiple View Geometry Transformers for 3D Human Pose Estimation (Ziwei Liao et al., 2023)

{{<citation>}}

Ziwei Liao, Jialiang Zhu, Chunyu Wang, Han Hu, Steven L. Waslander. (2023)  
**Multiple View Geometry Transformers for 3D Human Pose Estimation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.10983v1)  

---


**ABSTRACT**  
In this work, we aim to improve the 3D reasoning ability of Transformers in multi-view 3D human pose estimation. Recent works have focused on end-to-end learning-based transformer designs, which struggle to resolve geometric information accurately, particularly during occlusion. Instead, we propose a novel hybrid model, MVGFormer, which has a series of geometric and appearance modules organized in an iterative manner. The geometry modules are learning-free and handle all viewpoint-dependent 3D tasks geometrically which notably improves the model's generalization ability. The appearance modules are learnable and are dedicated to estimating 2D poses from image signals end-to-end which enables them to achieve accurate estimates even when occlusion occurs, leading to a model that is both accurate and generalizable to new cameras and geometries. We evaluate our approach for both in-domain and out-of-domain settings, where our model consistently outperforms state-of-the-art methods, and especially does so by a significant margin in the out-of-domain setting. We will release the code and models: https://github.com/XunshanMan/MVGFormer.

{{</citation>}}


## cs.SE (3)



### (16/62) Assessing the Security of GitHub Copilot Generated Code -- A Targeted Replication Study (Vahid Majdinasab et al., 2023)

{{<citation>}}

Vahid Majdinasab, Michael Joshua Bishop, Shawn Rasheed, Arghavan Moradidakhel, Amjed Tahir, Foutse Khomh. (2023)  
**Assessing the Security of GitHub Copilot Generated Code -- A Targeted Replication Study**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, Amazon, Security  
[Paper Link](http://arxiv.org/abs/2311.11177v1)  

---


**ABSTRACT**  
AI-powered code generation models have been developing rapidly, allowing developers to expedite code generation and thus improve their productivity. These models are trained on large corpora of code (primarily sourced from public repositories), which may contain bugs and vulnerabilities. Several concerns have been raised about the security of the code generated by these models. Recent studies have investigated security issues in AI-powered code generation tools such as GitHub Copilot and Amazon CodeWhisperer, revealing several security weaknesses in the code generated by these tools. As these tools evolve, it is expected that they will improve their security protocols to prevent the suggestion of insecure code to developers. This paper replicates the study of Pearce et al., which investigated security weaknesses in Copilot and uncovered several weaknesses in the code suggested by Copilot across diverse scenarios and languages (Python, C and Verilog). Our replication examines Copilot security weaknesses using newer versions of Copilot and CodeQL (the security analysis framework). The replication focused on the presence of security vulnerabilities in Python code. Our results indicate that, even with the improvements in newer versions of Copilot, the percentage of vulnerable code suggestions has reduced from 36.54% to 27.25%. Nonetheless, it remains evident that the model still suggests insecure code.

{{</citation>}}


### (17/62) (Why) Is My Prompt Getting Worse? Rethinking Regression Testing for Evolving LLM APIs (Wanqin Ma et al., 2023)

{{<citation>}}

Wanqin Ma, Chenyang Yang, Christian Kästner. (2023)  
**(Why) Is My Prompt Getting Worse? Rethinking Regression Testing for Evolving LLM APIs**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11123v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are increasingly integrated into software applications. Downstream application developers often access LLMs through APIs provided as a service. However, LLM APIs are often updated silently and scheduled to be deprecated, forcing users to continuously adapt to evolving models. This can cause performance regression and affect prompt design choices, as evidenced by our case study on toxicity detection. Based on our case study, we emphasize the need for and re-examine the concept of regression testing for evolving LLM APIs. We argue that regression testing LLMs requires fundamental changes to traditional testing approaches, due to different correctness notions, prompting brittleness, and non-determinism in LLM APIs.

{{</citation>}}


### (18/62) Can AI Serve as a Substitute for Human Subjects in Software Engineering Research? (Marco A. Gerosa et al., 2023)

{{<citation>}}

Marco A. Gerosa, Bianca Trinkenreich, Igor Steinmacher, Anita Sarma. (2023)  
**Can AI Serve as a Substitute for Human Subjects in Software Engineering Research?**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.11081v1)  

---


**ABSTRACT**  
Research within sociotechnical domains, such as Software Engineering, fundamentally requires a thorough consideration of the human perspective. However, traditional qualitative data collection methods suffer from challenges related to scale, labor intensity, and the increasing difficulty of participant recruitment. This vision paper proposes a novel approach to qualitative data collection in software engineering research by harnessing the capabilities of artificial intelligence (AI), especially large language models (LLMs) like ChatGPT. We explore the potential of AI-generated synthetic text as an alternative source of qualitative data, by discussing how LLMs can replicate human responses and behaviors in research settings. We examine the application of AI in automating data collection across various methodologies, including persona-based prompting for interviews, multi-persona dialogue for focus groups, and mega-persona responses for surveys. Additionally, we discuss the prospective development of new foundation models aimed at emulating human behavior in observational studies and user evaluations. By simulating human interaction and feedback, these AI models could offer scalable and efficient means of data generation, while providing insights into human attitudes, experiences, and performance. We discuss several open problems and research opportunities to implement this vision and conclude that while AI could augment aspects of data gathering in software engineering research, it cannot replace the nuanced, empathetic understanding inherent in human subjects in some cases, and an integrated approach where both AI and human-generated data coexist will likely yield the most effective outcomes.

{{</citation>}}


## cs.AI (8)



### (19/62) Best uses of ChatGPT and Generative AI for computer science research (Eduardo C. Garrido-Merchan, 2023)

{{<citation>}}

Eduardo C. Garrido-Merchan. (2023)  
**Best uses of ChatGPT and Generative AI for computer science research**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.11175v1)  

---


**ABSTRACT**  
Generative Artificial Intelligence (AI), particularly tools like OpenAI's popular ChatGPT, is reshaping the landscape of computer science research. Used wisely, these tools can boost the productivity of a computer research scientist. This paper provides an exploration of the diverse applications of ChatGPT and other generative AI technologies in computer science academic research, making recommendations about the use of Generative AI to make more productive the role of the computer research scientist, with the focus of writing new research papers. We highlight innovative uses such as brainstorming research ideas, aiding in the drafting and styling of academic papers and assisting in the synthesis of state-of-the-art section. Further, we delve into using these technologies in understanding interdisciplinary approaches, making complex texts simpler, and recommending suitable academic journals for publication. Significant focus is placed on generative AI's contributions to synthetic data creation, research methodology, and mentorship, as well as in task organization and article quality assessment. The paper also addresses the utility of AI in article review, adapting texts to length constraints, constructing counterarguments, and survey development. Moreover, we explore the capabilities of these tools in disseminating ideas, generating images and audio, text transcription, and engaging with editors. We also describe some non-recommended uses of generative AI for computer science research, mainly because of the limitations of this technology.

{{</citation>}}


### (20/62) An Improved Neural Network Model Based On CNN Using For Fruit Sugar Degree Detection (Boyang Deng et al., 2023)

{{<citation>}}

Boyang Deng, Xin Wen, Zhan Gao. (2023)  
**An Improved Neural Network Model Based On CNN Using For Fruit Sugar Degree Detection**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Image Classification, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.11120v1)  

---


**ABSTRACT**  
Artificial Intelligence(AI) widely applies in Image Classification and Recognition, Text Understanding and Natural Language Processing, which makes great progress. In this paper, we introduced AI into the fruit quality detection field. We designed a fruit sugar degree regression model using an Artificial Neural Network based on spectra of fruits within the visible/near-infrared(V/NIR)range. After analysis of fruit spectra, we innovatively proposed a new neural network structure: low layers consist of a Multilayer Perceptron(MLP), a middle layer is a 2-dimensional correlation matrix layer, and high layers consist of several Convolutional Neural Network(CNN) layers. In this study, we used fruit sugar value as a detection target, collecting two fruits called Gan Nan Navel and Tian Shan Pear as samples, doing experiments respectively, and comparing their results. We used Analysis of Variance(ANOVA) to evaluate the reliability of the dataset we collected. Then, we tried multiple strategies to process spectrum data, evaluating their effects. In this paper, we tried to add Wavelet Decomposition(WD) to reduce feature dimensions and a Genetic Algorithm(GA) to find excellent features. Then, we compared Neural Network models with traditional Partial Least Squares(PLS) based models. We also compared the neural network structure we designed(MLP-CNN) with other traditional neural network structures. In this paper, we proposed a new evaluation standard derived from dataset standard deviation(STD) for evaluating detection performance, validating the viability of using an artificial neural network model to do fruit sugar degree nondestructive detection.

{{</citation>}}


### (21/62) Designing Interpretable ML System to Enhance Trustworthy AI in Healthcare: A Systematic Review of the Last Decade to A Proposed Robust Framework (Elham Nasarian et al., 2023)

{{<citation>}}

Elham Nasarian, Roohallah Alizadehsani, U. Rajendra Acharyac, d Kwok-Leung Tsui. (2023)  
**Designing Interpretable ML System to Enhance Trustworthy AI in Healthcare: A Systematic Review of the Last Decade to A Proposed Robust Framework**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11055v1)  

---


**ABSTRACT**  
AI-based medical technologies, including wearables, telemedicine, LLMs, and digital care twins, significantly impact healthcare. Ensuring AI results are accurate and interpretable is crucial, especially for clinicians. This paper reviews processes and challenges of interpretable ML (IML) and explainable AI (XAI) in healthcare. Objectives include reviewing XAI processes, methods, applications, and challenges, with a focus on quality control. The IML process is classified into data pre-processing interpretability, interpretable modeling, and post-processing interpretability. The paper aims to establish the importance of robust interpretability in healthcare through experimental results, providing insights for creating communicable clinician-AI tools. Research questions, eligibility criteria, and goals were identified following PRISMA and PICO methods. PubMed, Scopus, and Web of Science were systematically searched using specific strings. The survey introduces a step-by-step roadmap for implementing XAI in clinical applications, addressing existing gaps and acknowledging XAI model limitations.

{{</citation>}}


### (22/62) Orca 2: Teaching Small Language Models How to Reason (Arindam Mitra et al., 2023)

{{<citation>}}

Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agrawal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour, Ahmed Awadallah. (2023)  
**Orca 2: Teaching Small Language Models How to Reason**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11045v1)  

---


**ABSTRACT**  
Orca 1 learns from rich signals, such as explanation traces, allowing it to outperform conventional instruction-tuned models on benchmarks like BigBench Hard and AGIEval. In Orca 2, we continue exploring how improved training signals can enhance smaller LMs' reasoning abilities. Research on training small LMs has often relied on imitation learning to replicate the output of more capable models. We contend that excessive emphasis on imitation may restrict the potential of smaller models. We seek to teach small LMs to employ different solution strategies for different tasks, potentially different from the one used by the larger model. For example, while larger models might provide a direct answer to a complex task, smaller models may not have the same capacity. In Orca 2, we teach the model various reasoning techniques (step-by-step, recall then generate, recall-reason-generate, direct answer, etc.). More crucially, we aim to help the model learn to determine the most effective solution strategy for each task. We evaluate Orca 2 using a comprehensive set of 15 diverse benchmarks (corresponding to approximately 100 tasks and over 36,000 unique prompts). Orca 2 significantly surpasses models of similar size and attains performance levels similar or better to those of models 5-10x larger, as assessed on complex tasks that test advanced reasoning abilities in zero-shot settings. We open-source Orca 2 to encourage further research on the development, evaluation, and alignment of smaller LMs.

{{</citation>}}


### (23/62) Data Center Audio/Video Intelligence on Device (DAVID) -- An Edge-AI Platform for Smart-Toys (Gabriel Cosache et al., 2023)

{{<citation>}}

Gabriel Cosache, Francisco Salgado, Cosmin Rotariu, George Sterpu, Rishabh Jain, Peter Corcoran. (2023)  
**Data Center Audio/Video Intelligence on Device (DAVID) -- An Edge-AI Platform for Smart-Toys**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11030v1)  

---


**ABSTRACT**  
An overview is given of the DAVID Smart-Toy platform, one of the first Edge AI platform designs to incorporate advanced low-power data processing by neural inference models co-located with the relevant image or audio sensors. There is also on-board capability for in-device text-to-speech generation. Two alternative embodiments are presented: a smart Teddy-bear, and a roving dog-like robot. The platform offers a speech-driven user interface and can observe and interpret user actions and facial expressions via its computer vision sensor node. A particular benefit of this design is that no personally identifiable information passes beyond the neural inference nodes thus providing inbuilt compliance with data protection regulations.

{{</citation>}}


### (24/62) HungerGist: An Interpretable Predictive Model for Food Insecurity (Yongsu Ahn et al., 2023)

{{<citation>}}

Yongsu Ahn, Muheng Yan, Yu-Ru Lin, Zian Wang. (2023)  
**HungerGist: An Interpretable Predictive Model for Food Insecurity**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.10953v1)  

---


**ABSTRACT**  
The escalating food insecurity in Africa, caused by factors such as war, climate change, and poverty, demonstrates the critical need for advanced early warning systems. Traditional methodologies, relying on expert-curated data encompassing climate, geography, and social disturbances, often fall short due to data limitations, hindering comprehensive analysis and potential discovery of new predictive factors. To address this, this paper introduces "HungerGist", a multi-task deep learning model utilizing news texts and NLP techniques. Using a corpus of over 53,000 news articles from nine African countries over four years, we demonstrate that our model, trained solely on news data, outperforms the baseline method trained on both traditional risk factors and human-curated keywords. In addition, our method has the ability to detect critical texts that contain interpretable signals known as "gists." Moreover, our examination of these gists indicates that this approach has the potential to reveal latent factors that would otherwise remain concealed in unstructured texts.

{{</citation>}}


### (25/62) Case Repositories: Towards Case-Based Reasoning for AI Alignment (K. J. Kevin Feng et al., 2023)

{{<citation>}}

K. J. Kevin Feng, Quan Ze Chen, Inyoung Cheong, King Xia, Amy X. Zhang. (2023)  
**Case Repositories: Towards Case-Based Reasoning for AI Alignment**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-HC, cs.AI  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.10934v2)  

---


**ABSTRACT**  
Case studies commonly form the pedagogical backbone in law, ethics, and many other domains that face complex and ambiguous societal questions informed by human values. Similar complexities and ambiguities arise when we consider how AI should be aligned in practice: when faced with vast quantities of diverse (and sometimes conflicting) values from different individuals and communities, with whose values is AI to align, and how should AI do so? We propose a complementary approach to constitutional AI alignment, grounded in ideas from case-based reasoning (CBR), that focuses on the construction of policies through judgments on a set of cases. We present a process to assemble such a case repository by: 1) gathering a set of ``seed'' cases -- questions one may ask an AI system -- in a particular domain from discussions in online communities, 2) eliciting domain-specific key dimensions for cases through workshops with domain experts, 3) using LLMs to generate variations of cases not seen in the wild, and 4) engaging with the public to judge and improve cases. We then discuss how such a case repository could assist in AI alignment, both through directly acting as precedents to ground acceptable behaviors, and as a medium for individuals and communities to engage in moral reasoning around AI.

{{</citation>}}


### (26/62) Representing visual classification as a linear combination of words (Shobhit Agarwal et al., 2023)

{{<citation>}}

Shobhit Agarwal, Yevgeniy R. Semenov, William Lotter. (2023)  
**Representing visual classification as a linear combination of words**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.10933v1)  

---


**ABSTRACT**  
Explainability is a longstanding challenge in deep learning, especially in high-stakes domains like healthcare. Common explainability methods highlight image regions that drive an AI model's decision. Humans, however, heavily rely on language to convey explanations of not only "where" but "what". Additionally, most explainability approaches focus on explaining individual AI predictions, rather than describing the features used by an AI model in general. The latter would be especially useful for model and dataset auditing, and potentially even knowledge generation as AI is increasingly being used in novel tasks. Here, we present an explainability strategy that uses a vision-language model to identify language-based descriptors of a visual classification task. By leveraging a pre-trained joint embedding space between images and text, our approach estimates a new classification task as a linear combination of words, resulting in a weight for each word that indicates its alignment with the vision-based classifier. We assess our approach using two medical imaging classification tasks, where we find that the resulting descriptors largely align with clinical knowledge despite a lack of domain-specific language training. However, our approach also identifies the potential for 'shortcut connections' in the public datasets used. Towards a functional measure of explainability, we perform a pilot reader study where we find that the AI-identified words can enable non-expert humans to perform a specialized medical task at a non-trivial level. Altogether, our results emphasize the potential of using multimodal foundational models to deliver intuitive, language-based explanations of visual tasks.

{{</citation>}}


## cs.LG (8)



### (27/62) Low-Precision Floating-Point for Efficient On-Board Deep Neural Network Processing (Cédric Gernigon et al., 2023)

{{<citation>}}

Cédric Gernigon, Silviu-Ioan Filip, Olivier Sentieys, Clément Coggiola, Mickaël Bruno. (2023)  
**Low-Precision Floating-Point for Efficient On-Board Deep Neural Network Processing**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2311.11172v1)  

---


**ABSTRACT**  
One of the major bottlenecks in high-resolution Earth Observation (EO) space systems is the downlink between the satellite and the ground. Due to hardware limitations, on-board power limitations or ground-station operation costs, there is a strong need to reduce the amount of data transmitted. Various processing methods can be used to compress the data. One of them is the use of on-board deep learning to extract relevant information in the data. However, most ground-based deep neural network parameters and computations are performed using single-precision floating-point arithmetic, which is not adapted to the context of on-board processing. We propose to rely on quantized neural networks and study how to combine low precision (mini) floating-point arithmetic with a Quantization-Aware Training methodology. We evaluate our approach with a semantic segmentation task for ship detection using satellite images from the Airbus Ship dataset. Our results show that 6-bit floating-point quantization for both weights and activations can compete with single-precision without significant accuracy degradation. Using a Thin U-Net 32 model, only a 0.3% accuracy degradation is observed with 6-bit minifloat quantization (a 6-bit equivalent integer-based approach leads to a 0.5% degradation). An initial hardware study also confirms the potential impact of such low-precision floating-point designs, but further investigation at the scale of a full inference accelerator is needed before concluding whether they are relevant in a practical on-board scenario.

{{</citation>}}


### (28/62) Environment-Aware Dynamic Graph Learning for Out-of-Distribution Generalization (Haonan Yuan et al., 2023)

{{<citation>}}

Haonan Yuan, Qingyun Sun, Xingcheng Fu, Ziwei Zhang, Cheng Ji, Hao Peng, Jianxin Li. (2023)  
**Environment-Aware Dynamic Graph Learning for Out-of-Distribution Generalization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.11114v1)  

---


**ABSTRACT**  
Dynamic graph neural networks (DGNNs) are increasingly pervasive in exploiting spatio-temporal patterns on dynamic graphs. However, existing works fail to generalize under distribution shifts, which are common in real-world scenarios. As the generation of dynamic graphs is heavily influenced by latent environments, investigating their impacts on the out-of-distribution (OOD) generalization is critical. However, it remains unexplored with the following two major challenges: (1) How to properly model and infer the complex environments on dynamic graphs with distribution shifts? (2) How to discover invariant patterns given inferred spatio-temporal environments? To solve these challenges, we propose a novel Environment-Aware dynamic Graph LEarning (EAGLE) framework for OOD generalization by modeling complex coupled environments and exploiting spatio-temporal invariant patterns. Specifically, we first design the environment-aware EA-DGNN to model environments by multi-channel environments disentangling. Then, we propose an environment instantiation mechanism for environment diversification with inferred distributions. Finally, we discriminate spatio-temporal invariant patterns for out-of-distribution prediction by the invariant pattern recognition mechanism and perform fine-grained causal interventions node-wisely with a mixture of instantiated environment samples. Experiments on real-world and synthetic dynamic graph datasets demonstrate the superiority of our method against state-of-the-art baselines under distribution shifts. To the best of our knowledge, we are the first to study OOD generalization on dynamic graphs from the environment learning perspective.

{{</citation>}}


### (29/62) Deep Tensor Network (Yifan Zhang, 2023)

{{<citation>}}

Yifan Zhang. (2023)  
**Deep Tensor Network**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG, quant-ph  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.11091v1)  

---


**ABSTRACT**  
In this paper, we delve into the foundational principles of tensor categories, harnessing the universal property of the tensor product to pioneer novel methodologies in deep network architectures. Our primary contribution is the introduction of the Tensor Attention and Tensor Interaction Mechanism, a groundbreaking approach that leverages the tensor category to enhance the computational efficiency and the expressiveness of deep networks, and can even be generalized into the quantum realm.

{{</citation>}}


### (30/62) Compositional Fusion of Signals in Data Embedding (Zhijin Guo et al., 2023)

{{<citation>}}

Zhijin Guo, Zhaozhen Xu, Martha Lewis, Nello Cristianini. (2023)  
**Compositional Fusion of Signals in Data Embedding**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, BERT, Embedding  
[Paper Link](http://arxiv.org/abs/2311.11085v1)  

---


**ABSTRACT**  
Embeddings in AI convert symbolic structures into fixed-dimensional vectors, effectively fusing multiple signals. However, the nature of this fusion in real-world data is often unclear. To address this, we introduce two methods: (1) Correlation-based Fusion Detection, measuring correlation between known attributes and embeddings, and (2) Additive Fusion Detection, viewing embeddings as sums of individual vectors representing attributes.   Applying these methods, word embeddings were found to combine semantic and morphological signals. BERT sentence embeddings were decomposed into individual word vectors of subject, verb and object. In the knowledge graph-based recommender system, user embeddings, even without training on demographic data, exhibited signals of demographics like age and gender.   This study highlights that embeddings are fusions of multiple signals, from Word2Vec components to demographic hints in graph embeddings.

{{</citation>}}


### (31/62) ECLM: Efficient Edge-Cloud Collaborative Learning with Continuous Environment Adaptation (Yan Zhuang et al., 2023)

{{<citation>}}

Yan Zhuang, Zhenzhe Zheng, Yunfeng Shao, Bingshuai Li, Fan Wu, Guihai Chen. (2023)  
**ECLM: Efficient Edge-Cloud Collaborative Learning with Continuous Environment Adaptation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11083v1)  

---


**ABSTRACT**  
Pervasive mobile AI applications primarily employ one of the two learning paradigms: cloud-based learning (with powerful large models) or on-device learning (with lightweight small models). Despite their own advantages, neither paradigm can effectively handle dynamic edge environments with frequent data distribution shifts and on-device resource fluctuations, inevitably suffering from performance degradation. In this paper, we propose ECLM, an edge-cloud collaborative learning framework for rapid model adaptation for dynamic edge environments. We first propose a novel block-level model decomposition design to decompose the original large cloud model into multiple combinable modules. By flexibly combining a subset of the modules, this design enables the derivation of compact, task-specific sub-models for heterogeneous edge devices from the large cloud model, and the seamless integration of new knowledge learned on these devices into the cloud model periodically. As such, ECLM ensures that the cloud model always provides up-to-date sub-models for edge devices. We further propose an end-to-end learning framework that incorporates the modular model design into an efficient model adaptation pipeline including an offline on-cloud model prototyping and training stage, and an online edge-cloud collaborative adaptation stage. Extensive experiments over various datasets demonstrate that ECLM significantly improves model performance (e.g., 18.89% accuracy increase) and resource efficiency (e.g., 7.12x communication cost reduction) in adapting models to dynamic edge environments by efficiently collaborating the edge and the cloud models.

{{</citation>}}


### (32/62) Tactics2D: A Multi-agent Reinforcement Learning Environment for Driving Decision-making (Yueyuan Li et al., 2023)

{{<citation>}}

Yueyuan Li, Songan Zhang, Mingyang Jiang, Xingyuan Chen, Ming Yang. (2023)  
**Tactics2D: A Multi-agent Reinforcement Learning Environment for Driving Decision-making**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.11058v1)  

---


**ABSTRACT**  
Tactics2D is an open-source multi-agent reinforcement learning library with a Python backend. Its goal is to provide a convenient toolset for researchers to develop decision-making algorithms for autonomous driving. The library includes diverse traffic scenarios implemented as gym-based environments equipped with multi-sensory capabilities and violation detection for traffic rules. Additionally, it features a reinforcement learning baseline tested with reasonable evaluation metrics. Tactics2D is highly modular and customizable. The source code of Tactics2D is available at https://github.com/WoodOxen/Tactics2D.

{{</citation>}}


### (33/62) SORTAD: Self-Supervised Optimized Random Transformations for Anomaly Detection in Tabular Data (Guy Hay et al., 2023)

{{<citation>}}

Guy Hay, Pablo Liberman. (2023)  
**SORTAD: Self-Supervised Optimized Random Transformations for Anomaly Detection in Tabular Data**  

---
Primary Category: cs.LG  
Categories: I-5-1, cs-LG, cs.LG  
Keywords: Anomaly Detection, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.11018v1)  

---


**ABSTRACT**  
We consider a self-supervised approach to anomaly detection in tabular data. Random transformations are applied to the data, and then each transformation is identified based on its output. These predicted transformations are used to identify anomalies. In tabular data this approach faces many challenges that are related to the uncorrelated nature of the data. These challenges affect the transformations that should be used, as well as the use of their predictions. To this end, we propose SORTAD, a novel algorithm that is tailor-made to solve these challenges. SORTAD optimally chooses random transformations that help the classification process, and have a scoring function that is more sensitive to the changes in the transformations classification prediction encountered in tabular data. SORTAD achieved state-of-the-art results on multiple commonly used anomaly detection data sets, as well as in the overall results across all data sets tested.

{{</citation>}}


### (34/62) BrainZ-BP: A Non-invasive Cuff-less Blood Pressure Estimation Approach Leveraging Brain Bio-impedance and Electrocardiogram (Bufang Yang et al., 2023)

{{<citation>}}

Bufang Yang, Le Liu, Wenxuan Wu, Mengliang Zhou, Hongxing Liu, Xinbao Ning. (2023)  
**BrainZ-BP: A Non-invasive Cuff-less Blood Pressure Estimation Approach Leveraging Brain Bio-impedance and Electrocardiogram**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2311.10996v1)  

---


**ABSTRACT**  
Accurate and continuous blood pressure (BP) monitoring is essential to the early prevention of cardiovascular diseases. Non-invasive and cuff-less BP estimation algorithm has gained much attention in recent years. Previous studies have demonstrated that brain bio-impedance (BIOZ) is a promising technique for non-invasive intracranial pressure (ICP) monitoring. Clinically, treatment for patients with traumatic brain injuries (TBI) requires monitoring the ICP and BP of patients simultaneously. Estimating BP by brain BIOZ directly can reduce the number of sensors attached to the patients, thus improving their comfort. To address the issues, in this study, we explore the feasibility of leveraging brain BIOZ for BP estimation and propose a novel cuff-less BP estimation approach called BrainZ-BP. Two electrodes are placed on the forehead and occipital bone of the head in the anterior-posterior direction for brain BIOZ measurement. Various features including pulse transit time and morphological features of brain BIOZ are extracted and fed into four regression models for BP estimation. Results show that the mean absolute error, root mean square error, and correlation coefficient of random forest regression model are 2.17 mmHg, 3.91 mmHg, and 0.90 for systolic pressure estimation, and are 1.71 mmHg, 3.02 mmHg, and 0.89 for diastolic pressure estimation. The presented BrainZ-BP can be applied in the brain BIOZ-based ICP monitoring scenario to monitor BP simultaneously.

{{</citation>}}


## cs.CL (14)



### (35/62) Experts-in-the-Loop: Establishing an Effective Workflow in Crafting Privacy Q&A (Zahra Kolagar et al., 2023)

{{<citation>}}

Zahra Kolagar, Anna Katharina Leschanowsky, Birgit Popp. (2023)  
**Experts-in-the-Loop: Establishing an Effective Workflow in Crafting Privacy Q&A**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11161v1)  

---


**ABSTRACT**  
Privacy policies play a vital role in safeguarding user privacy as legal jurisdictions worldwide emphasize the need for transparent data processing. While the suitability of privacy policies to enhance transparency has been critically discussed, employing conversational AI systems presents unique challenges in informing users effectively. In this position paper, we propose a dynamic workflow for transforming privacy policies into privacy question-and-answer (Q&A) pairs to make privacy policies easily accessible through conversational AI. Thereby, we facilitate interdisciplinary collaboration among legal experts and conversation designers, while also considering the utilization of large language models' generative capabilities and addressing associated challenges. Our proposed workflow underscores continuous improvement and monitoring throughout the construction of privacy Q&As, advocating for comprehensive review and refinement through an experts-in-the-loop approach.

{{</citation>}}


### (36/62) Vashantor: A Large-scale Multilingual Benchmark Dataset for Automated Translation of Bangla Regional Dialects to Bangla Language (Fatema Tuj Johora Faria et al., 2023)

{{<citation>}}

Fatema Tuj Johora Faria, Mukaffi Bin Moin, Ahmed Al Wase, Mehidi Ahmmed, Md. Rabius Sani, Tashreef Muhammad. (2023)  
**Vashantor: A Large-scale Multilingual Benchmark Dataset for Automated Translation of Bangla Regional Dialects to Bangla Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, BLEU, Multilingual, T5  
[Paper Link](http://arxiv.org/abs/2311.11142v1)  

---


**ABSTRACT**  
The Bangla linguistic variety is a fascinating mix of regional dialects that adds to the cultural diversity of the Bangla-speaking community. Despite extensive study into translating Bangla to English, English to Bangla, and Banglish to Bangla in the past, there has been a noticeable gap in translating Bangla regional dialects into standard Bangla. In this study, we set out to fill this gap by creating a collection of 32,500 sentences, encompassing Bangla, Banglish, and English, representing five regional Bangla dialects. Our aim is to translate these regional dialects into standard Bangla and detect regions accurately. To achieve this, we proposed models known as mT5 and BanglaT5 for translating regional dialects into standard Bangla. Additionally, we employed mBERT and Bangla-bert-base to determine the specific regions from where these dialects originated. Our experimental results showed the highest BLEU score of 69.06 for Mymensingh regional dialects and the lowest BLEU score of 36.75 for Chittagong regional dialects. We also observed the lowest average word error rate of 0.1548 for Mymensingh regional dialects and the highest of 0.3385 for Chittagong regional dialects. For region detection, we achieved an accuracy of 85.86% for Bangla-bert-base and 84.36% for mBERT. This is the first large-scale investigation of Bangla regional dialects to Bangla machine translation. We believe our findings will not only pave the way for future work on Bangla regional dialects to Bangla machine translation, but will also be useful in solving similar language-related challenges in low-resource language conditions.

{{</citation>}}


### (37/62) A Principled Framework for Knowledge-enhanced Large Language Model (Saizhuo Wang et al., 2023)

{{<citation>}}

Saizhuo Wang, Zhihan Liu, Zhaoran Wang, Jian Guo. (2023)  
**A Principled Framework for Knowledge-enhanced Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11135v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are versatile, yet they often falter in tasks requiring deep and reliable reasoning due to issues like hallucinations, limiting their applicability in critical scenarios. This paper introduces a rigorously designed framework for creating LLMs that effectively anchor knowledge and employ a closed-loop reasoning process, enhancing their capability for in-depth analysis. We dissect the framework to illustrate the contribution of each component to the LLMs' performance, offering a theoretical assurance of improved reasoning under well-defined assumptions.

{{</citation>}}


### (38/62) Utilizing Speech Emotion Recognition and Recommender Systems for Negative Emotion Handling in Therapy Chatbots (Farideh Majidi et al., 2023)

{{<citation>}}

Farideh Majidi, Marzieh Bahrami. (2023)  
**Utilizing Speech Emotion Recognition and Recommender Systems for Negative Emotion Handling in Therapy Chatbots**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Emotion Recognition, LSTM  
[Paper Link](http://arxiv.org/abs/2311.11116v1)  

---


**ABSTRACT**  
Emotional well-being significantly influences mental health and overall quality of life. As therapy chatbots become increasingly prevalent, their ability to comprehend and respond empathetically to users' emotions remains limited. This paper addresses this limitation by proposing an approach to enhance therapy chatbots with auditory perception, enabling them to understand users' feelings and provide human-like empathy. The proposed method incorporates speech emotion recognition (SER) techniques using Convolutional Neural Network (CNN) models and the ShEMO dataset to accurately detect and classify negative emotions, including anger, fear, and sadness. The SER model achieves a validation accuracy of 88%, demonstrating its effectiveness in recognizing emotional states from speech signals. Furthermore, a recommender system is developed, leveraging the SER model's output to generate personalized recommendations for managing negative emotions, for which a new bilingual dataset was generated as well since there is no such dataset available for this task. The recommender model achieves an accuracy of 98% by employing a combination of global vectors for word representation (GloVe) and LSTM models. To provide a more immersive and empathetic user experience, a text-to-speech model called GlowTTS is integrated, enabling the therapy chatbot to audibly communicate the generated recommendations to users in both English and Persian. The proposed approach offers promising potential to enhance therapy chatbots by providing them with the ability to recognize and respond to users' emotions, ultimately improving the delivery of mental health support for both English and Persian-speaking users.

{{</citation>}}


### (39/62) Responsible AI Considerations in Text Summarization Research: A Review of Current Practices (Yu Lu Liu et al., 2023)

{{<citation>}}

Yu Lu Liu, Meng Cao, Su Lin Blodgett, Jackie Chi Kit Cheung, Alexandra Olteanu, Adam Trischler. (2023)  
**Responsible AI Considerations in Text Summarization Research: A Review of Current Practices**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, NLP, Summarization, Text Summarization  
[Paper Link](http://arxiv.org/abs/2311.11103v1)  

---


**ABSTRACT**  
AI and NLP publication venues have increasingly encouraged researchers to reflect on possible ethical considerations, adverse impacts, and other responsible AI issues their work might engender. However, for specific NLP tasks our understanding of how prevalent such issues are, or when and why these issues are likely to arise, remains limited. Focusing on text summarization -- a common NLP task largely overlooked by the responsible AI community -- we examine research and reporting practices in the current literature. We conduct a multi-round qualitative analysis of 333 summarization papers from the ACL Anthology published between 2020-2022. We focus on how, which, and when responsible AI issues are covered, which relevant stakeholders are considered, and mismatches between stated and realized research goals. We also discuss current evaluation practices and consider how authors discuss the limitations of both prior work and their own work. Overall, we find that relatively few papers engage with possible stakeholders or contexts of use, which limits their consideration of potential downstream adverse impacts or other responsible AI issues. Based on our findings, we make recommendations on concrete practices and research directions.

{{</citation>}}


### (40/62) Combining EEG and NLP Features for Predicting Students' Lecture Comprehension using Ensemble Classification (Phantharach Natnithikarat et al., 2023)

{{<citation>}}

Phantharach Natnithikarat, Theerawit Wilaiprasitporn, Supavit Kongwudhikunakorn. (2023)  
**Combining EEG and NLP Features for Predicting Students' Lecture Comprehension using Ensemble Classification**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.11088v1)  

---


**ABSTRACT**  
Electroencephalography (EEG) and Natural Language Processing (NLP) can be applied for education to measure students' comprehension in classroom lectures; currently, the two measures have been used separately. In this work, we propose a classification framework for predicting students' lecture comprehension in two tasks: (i) students' confusion after listening to the simulated lecture and (ii) the correctness of students' responses to the post-lecture assessment. The proposed framework includes EEG and NLP feature extraction, processing, and classification. EEG and NLP features are extracted to construct integrated features obtained from recorded EEG signals and sentence-level syntactic analysis, which provide information about specific biomarkers and sentence structures. An ensemble stacking classification method -- a combination of multiple individual models that produces an enhanced predictive model -- is studied to learn from the features to make predictions accurately. Furthermore, we also utilized subjective confusion ratings as another integrated feature to enhance classification performance. By doing so, experiment results show that this framework performs better than the baselines, which achieved F1 up to 0.65 for predicting confusion and 0.78 for predicting correctness, highlighting that utilizing this has helped improve the classification performance.

{{</citation>}}


### (41/62) Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning (Clifton Poth et al., 2023)

{{<citation>}}

Clifton Poth, Hannah Sterz, Indraneil Paul, Sukannya Purkayastha, Leon Engländer, Timo Imhof, Ivan Vulić, Sebastian Ruder, Iryna Gurevych, Jonas Pfeiffer. (2023)  
**Adapters: A Unified Library for Parameter-Efficient and Modular Transfer Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.11077v1)  

---


**ABSTRACT**  
We introduce Adapters, an open-source library that unifies parameter-efficient and modular transfer learning in large language models. By integrating 10 diverse adapter methods into a unified interface, Adapters offers ease of use and flexible configuration. Our library allows researchers and practitioners to leverage adapter modularity through composition blocks, enabling the design of complex adapter setups. We demonstrate the library's efficacy by evaluating its performance against full fine-tuning on various NLP tasks. Adapters provides a powerful tool for addressing the challenges of conventional fine-tuning paradigms and promoting more efficient and modular transfer learning. The library is available via https://adapterhub.ml/adapters.

{{</citation>}}


### (42/62) Bit Cipher -- A Simple yet Powerful Word Representation System that Integrates Efficiently with Language Models (Haoran Zhao et al., 2023)

{{<citation>}}

Haoran Zhao, Jake Ryland Williams. (2023)  
**Bit Cipher -- A Simple yet Powerful Word Representation System that Integrates Efficiently with Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NER, T5, Word Representation  
[Paper Link](http://arxiv.org/abs/2311.11012v1)  

---


**ABSTRACT**  
While Large Language Models (LLMs) become ever more dominant, classic pre-trained word embeddings sustain their relevance through computational efficiency and nuanced linguistic interpretation. Drawing from recent studies demonstrating that the convergence of GloVe and word2vec optimizations all tend towards log-co-occurrence matrix variants, we construct a novel word representation system called Bit-cipher that eliminates the need of backpropagation while leveraging contextual information and hyper-efficient dimensionality reduction techniques based on unigram frequency, providing strong interpretability, alongside efficiency. We use the bit-cipher algorithm to train word vectors via a two-step process that critically relies on a hyperparameter -- bits -- that controls the vector dimension. While the first step trains the bit-cipher, the second utilizes it under two different aggregation modes -- summation or concatenation -- to produce contextually rich representations from word co-occurrences. We extend our investigation into bit-cipher's efficacy, performing probing experiments on part-of-speech (POS) tagging and named entity recognition (NER) to assess its competitiveness with classic embeddings like word2vec and GloVe. Additionally, we explore its applicability in LM training and fine-tuning. By replacing embedding layers with cipher embeddings, our experiments illustrate the notable efficiency of cipher in accelerating the training process and attaining better optima compared to conventional training paradigms. Experiments on the integration of bit-cipher embedding layers with Roberta, T5, and OPT, prior to or as a substitute for fine-tuning, showcase a promising enhancement to transfer learning, allowing rapid model convergence while preserving competitive performance.

{{</citation>}}


### (43/62) Joyful: Joint Modality Fusion and Graph Contrastive Learning for Multimodal Emotion Recognition (Dongyuan Li et al., 2023)

{{<citation>}}

Dongyuan Li, Yusong Wang, Kotaro Funakoshi, Manabu Okumura. (2023)  
**Joyful: Joint Modality Fusion and Graph Contrastive Learning for Multimodal Emotion Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning, Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2311.11009v1)  

---


**ABSTRACT**  
Multimodal emotion recognition aims to recognize emotions for each utterance of multiple modalities, which has received increasing attention for its application in human-machine interaction. Current graph-based methods fail to simultaneously depict global contextual features and local diverse uni-modal features in a dialogue. Furthermore, with the number of graph layers increasing, they easily fall into over-smoothing. In this paper, we propose a method for joint modality fusion and graph contrastive learning for multimodal emotion recognition (Joyful), where multimodality fusion, contrastive learning, and emotion recognition are jointly optimized. Specifically, we first design a new multimodal fusion mechanism that can provide deep interaction and fusion between the global contextual and uni-modal specific features. Then, we introduce a graph contrastive learning framework with inter-view and intra-view contrastive losses to learn more distinguishable representations for samples with different sentiments. Extensive experiments on three benchmark datasets indicate that Joyful achieved state-of-the-art (SOTA) performance compared to all baselines.

{{</citation>}}


### (44/62) Journey of Hallucination-minimized Generative AI Solutions for Financial Decision Makers (Sohini Roychowdhury, 2023)

{{<citation>}}

Sohini Roychowdhury. (2023)  
**Journey of Hallucination-minimized Generative AI Solutions for Financial Decision Makers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Financial, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.10961v1)  

---


**ABSTRACT**  
Generative AI has significantly reduced the entry barrier to the domain of AI owing to the ease of use and core capabilities of automation, translation, and intelligent actions in our day to day lives. Currently, Large language models (LLMs) that power such chatbots are being utilized primarily for their automation capabilities for software monitoring, report generation etc. and for specific personalized question answering capabilities, on a limited scope and scale. One major limitation of the currently evolving family of LLMs is 'hallucinations', wherein inaccurate responses are reported as factual. Hallucinations are primarily caused by biased training data, ambiguous prompts and inaccurate LLM parameters, and they majorly occur while combining mathematical facts with language-based context. Thus, monitoring and controlling for hallucinations becomes necessary when designing solutions that are meant for decision makers. In this work we present the three major stages in the journey of designing hallucination-minimized LLM-based solutions that are specialized for the decision makers of the financial domain, namely: prototyping, scaling and LLM evolution using human feedback. These three stages and the novel data to answer generation modules presented in this work are necessary to ensure that the Generative AI chatbots, autonomous reports and alerts are reliable and high-quality to aid key decision-making processes.

{{</citation>}}


### (45/62) An Empirical Bayes Framework for Open-Domain Dialogue Generation (Jing Yang Lee et al., 2023)

{{<citation>}}

Jing Yang Lee, Kong Aik Lee, Woon-Seng Gan. (2023)  
**An Empirical Bayes Framework for Open-Domain Dialogue Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2311.10945v1)  

---


**ABSTRACT**  
To engage human users in meaningful conversation, open-domain dialogue agents are required to generate diverse and contextually coherent dialogue. Despite recent advancements, which can be attributed to the usage of pretrained language models, the generation of diverse and coherent dialogue remains an open research problem. A popular approach to address this issue involves the adaptation of variational frameworks. However, while these approaches successfully improve diversity, they tend to compromise on contextual coherence. Hence, we propose the Bayesian Open-domain Dialogue with Empirical Bayes (BODEB) framework, an empirical bayes framework for constructing an Bayesian open-domain dialogue agent by leveraging pretrained parameters to inform the prior and posterior parameter distributions. Empirical results show that BODEB achieves better results in terms of both diversity and coherence compared to variational frameworks.

{{</citation>}}


### (46/62) Partially Randomizing Transformer Weights for Dialogue Response Diversity (Jing Yang Lee et al., 2023)

{{<citation>}}

Jing Yang Lee, Kong Aik Lee, Woon-Seng Gan. (2023)  
**Partially Randomizing Transformer Weights for Dialogue Response Diversity**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Transformer  
[Paper Link](http://arxiv.org/abs/2311.10943v1)  

---


**ABSTRACT**  
Despite recent progress in generative open-domain dialogue, the issue of low response diversity persists. Prior works have addressed this issue via either novel objective functions, alternative learning approaches such as variational frameworks, or architectural extensions such as the Randomized Link (RL) Transformer. However, these approaches typically entail either additional difficulties during training/inference, or a significant increase in model size and complexity. Hence, we propose the \underline{Pa}rtially \underline{Ra}ndomized trans\underline{Former} (PaRaFormer), a simple extension of the transformer which involves freezing the weights of selected layers after random initialization. Experimental results reveal that the performance of the PaRaformer is comparable to that of the aforementioned approaches, despite not entailing any additional training difficulty or increase in model complexity.

{{</citation>}}


### (47/62) CAMRA: Copilot for AMR Annotation (Jon Z. Cai et al., 2023)

{{<citation>}}

Jon Z. Cai, Shafiuddin Rehan Ahmed, Julia Bonn, Kristin Wright-Bettner, Martha Palmer, James H. Martin. (2023)  
**CAMRA: Copilot for AMR Annotation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Abstract Meaning Representation  
[Paper Link](http://arxiv.org/abs/2311.10928v1)  

---


**ABSTRACT**  
In this paper, we introduce CAMRA (Copilot for AMR Annotatations), a cutting-edge web-based tool designed for constructing Abstract Meaning Representation (AMR) from natural language text. CAMRA offers a novel approach to deep lexical semantics annotation such as AMR, treating AMR annotation akin to coding in programming languages. Leveraging the familiarity of programming paradigms, CAMRA encompasses all essential features of existing AMR editors, including example lookup, while going a step further by integrating Propbank roleset lookup as an autocomplete feature within the tool. Notably, CAMRA incorporates AMR parser models as coding co-pilots, greatly enhancing the efficiency and accuracy of AMR annotators. To demonstrate the tool's capabilities, we provide a live demo accessible at: https://camra.colorado.edu

{{</citation>}}


### (48/62) Understanding and Mitigating Classification Errors Through Interpretable Token Patterns (Michael A. Hedderich et al., 2023)

{{<citation>}}

Michael A. Hedderich, Jonas Fischer, Dietrich Klakow, Jilles Vreeken. (2023)  
**Understanding and Mitigating Classification Errors Through Interpretable Token Patterns**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NER, NLP, QA  
[Paper Link](http://arxiv.org/abs/2311.10920v1)  

---


**ABSTRACT**  
State-of-the-art NLP methods achieve human-like performance on many tasks, but make errors nevertheless. Characterizing these errors in easily interpretable terms gives insight into whether a classifier is prone to making systematic errors, but also gives a way to act and improve the classifier. We propose to discover those patterns of tokens that distinguish correct and erroneous predictions as to obtain global and interpretable descriptions for arbitrary NLP classifiers. We formulate the problem of finding a succinct and non-redundant set of such patterns in terms of the Minimum Description Length principle. Through an extensive set of experiments, we show that our method, Premise, performs well in practice. Unlike existing solutions, it recovers ground truth, even on highly imbalanced data over large vocabularies. In VQA and NER case studies, we confirm that it gives clear and actionable insight into the systematic errors made by NLP classifiers.

{{</citation>}}


## cs.SI (3)



### (49/62) Contextualizing Internet Memes Across Social Media Platforms (Saurav Joshi et al., 2023)

{{<citation>}}

Saurav Joshi, Filip Ilievski, Luca Luceri. (2023)  
**Contextualizing Internet Memes Across Social Media Platforms**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-IR, cs-SI, cs.SI  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2311.11157v1)  

---


**ABSTRACT**  
Internet memes have emerged as a novel format for communication and expressing ideas on the web. Their fluidity and creative nature are reflected in their widespread use, often across platforms and occasionally for unethical or harmful purposes. While computational work has already analyzed their high-level virality over time and developed specialized classifiers for hate speech detection, there have been no efforts to date that aim to holistically track, identify, and map internet memes posted on social media. To bridge this gap, we investigate whether internet memes across social media platforms can be contextualized by using a semantic repository of knowledge, namely, a knowledge graph. We collect thousands of potential internet meme posts from two social media platforms, namely Reddit and Discord, and perform an extract-transform-load procedure to create a data lake with candidate meme posts. By using vision transformer-based similarity, we match these candidates against the memes cataloged in a recently released knowledge graph of internet memes, IMKG. We provide evidence that memes published online can be identified by mapping them to IMKG. We leverage this grounding to study the prevalence of memes on different platforms, discover popular memes, and select common meme channels and subreddits. Finally, we illustrate how the grounding can enable users to get context about memes on social media thanks to their link to the knowledge graph.

{{</citation>}}


### (50/62) DSCom: A Data-Driven Self-Adaptive Community-Based Framework for Influence Maximization in Social Networks (Yuxin Zuo et al., 2023)

{{<citation>}}

Yuxin Zuo, Haojia Sun, Yongyi Hu, Jianxiong Guo, Xiaofeng Gao. (2023)  
**DSCom: A Data-Driven Self-Adaptive Community-Based Framework for Influence Maximization in Social Networks**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2311.11080v1)  

---


**ABSTRACT**  
Influence maximization aims to find a subset of seeds that maximize the influence spread under a given budget. In this paper, we mainly address the data-driven version of this problem, where the diffusion model is not given but needs to be inferred from the history cascades. Several previous works have addressed this topic in a statistical way and provided efficient algorithms with theoretical guarantee. However, in their settings, though the diffusion parameters are inferred, they still need users to preset the diffusion model, which can be an intractable problem in real-world practices. In this paper, we reformulate the problem on the attributed network and leverage the node attributes to estimate the closeness between the connected nodes. Specifically, we propose a machine learning-based framework, named DSCom, to address this problem in a heuristic way. Under this framework, we first infer the users' relationship from the diffusion dataset through attention mechanism and then leverage spectral clustering to overcome the influence overlap problem in the lack of exact diffusion formula. Compared to the previous theoretical works, we carefully designed empirical experiments with parameterized diffusion models based on real-world social networks, which prove the efficiency and effectiveness of our algorithm.

{{</citation>}}


### (51/62) Community-Aware Efficient Graph Contrastive Learning via Personalized Self-Training (Yuecheng Li et al., 2023)

{{<citation>}}

Yuecheng Li, Yanming Hu, Lele Fu, Chuan Chen, Lei Yang, Zibin Zheng. (2023)  
**Community-Aware Efficient Graph Contrastive Learning via Personalized Self-Training**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-LG, cs-SI, cs.SI  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.11073v1)  

---


**ABSTRACT**  
In recent years, graph contrastive learning (GCL) has emerged as one of the optimal solutions for various supervised tasks at the node level. However, for unsupervised and structure-related tasks such as community detection, current GCL algorithms face difficulties in acquiring the necessary community-level information, resulting in poor performance. In addition, general contrastive learning algorithms improve the performance of downstream tasks by increasing the number of negative samples, which leads to severe class collision and unfairness of community detection. To address above issues, we propose a novel Community-aware Efficient Graph Contrastive Learning Framework (CEGCL) to jointly learn community partition and node representations in an end-to-end manner. Specifically, we first design a personalized self-training (PeST) strategy for unsupervised scenarios, which enables our model to capture precise community-level personalized information in a graph. With the benefit of the PeST, we alleviate class collision and unfairness without sacrificing the overall model performance. Furthermore, the aligned graph clustering (AlGC) is employed to obtain the community partition. In this module, we align the clustering space of our downstream task with that in PeST to achieve more consistent node embeddings. Finally, we demonstrate the effectiveness of our model for community detection both theoretically and experimentally. Extensive experimental results also show that our CEGCL exhibits state-of-the-art performance on three benchmark datasets with different scales.

{{</citation>}}


## cs.DL (1)



### (52/62) The state of OAI-PMH repositories in Canadian Universities (Frédéric Piedboeuf et al., 2023)

{{<citation>}}

Frédéric Piedboeuf, Guillaume Le Berre, David Alfonso-Hermelo, Olivier Charbonneau, Philippe Langlais. (2023)  
**The state of OAI-PMH repositories in Canadian Universities**  

---
Primary Category: cs.DL  
Categories: cs-DL, cs.DL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11140v1)  

---


**ABSTRACT**  
This article presents a study of the current state of Universities Institutional Repositories (UIRs) in Canada. UIRs are vital to sharing information and documents, mainly Electronic Thesis and Dissertation (ETDs), and theoretically allow anyone, anywhere, to access the documents contained within the repository. Despite calls for consistent and shareable metadata in these repositories, our literature review shows inconsistencies in UIRs, including incorrect use of metadata fields and the omission of crucial information, rendering the systematic analysis of UIR complex. Nonetheless, we collected the data of 57 Canadian UIRs with the aim of analyzing Canadian data and to assess the quality of its UIRs. This was surprisingly difficult due to the lack of information about the UIRs, and we attempt to ease future collection efforts by organizing vital information which are difficult to find, starting from addresses of UIRs. We furthermore present and analyze the main characteristics of the UIRs we managed to collect, using this dataset to create recommendations for future practitioners.

{{</citation>}}


## cs.NI (1)



### (53/62) User-Centric Interactive AI for Distributed Diffusion Model-based AI-Generated Content (Hongyang Du et al., 2023)

{{<citation>}}

Hongyang Du, Ruichen Zhang, Dusit Niyato, Jiawen Kang, Zehui Xiong, Shuguang Cui, Xuemin Shen, Dong In Kim. (2023)  
**User-Centric Interactive AI for Distributed Diffusion Model-based AI-Generated Content**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.11094v1)  

---


**ABSTRACT**  
Distributed Artificial Intelligence-Generated Content (AIGC) has attracted increasing attention. However, it faces two significant challenges: how to maximize the subjective Quality of Experience (QoE) and how to enhance the energy efficiency, which are particularly pronounced in widely adopted Generative Diffusion Model (GDM)-based AIGC services for image generation. In this paper, we propose a novel user-centric Interactive AI (IAI) approach for service management, with a distributed GDM-based AIGC framework, prioritizing efficient and collaborative GDM deployment. Specifically, we restructure the GDM's inference process, i.e., the denoising chain, to enable users' semantically similar prompts to share a portion of diffusion steps. Furthermore, to maximize the users' subjective QoE, we propose an IAI approach, i.e., Reinforcement Learning With Large Language Models Interaction (RLLI), which utilizes Large Language Model (LLM)-empowered generative agents to replicate users interaction, providing real-time and subjective QoE feedback that reflects a spectrum of user personalities. Lastly, we present the GDM-based Deep Deterministic Policy Gradient (G-DDPG) algorithm, adapted to the proposed RLLI framework, for effective communication and computing resource allocation while considering user subjective personalities and dynamic wireless environments in decision-making. Simulation results show that G-DDPG can increase the sum QoE by 15%, compared with the conventional DDPG algorithm.

{{</citation>}}


## eess.IV (3)



### (54/62) LightBTSeg: A lightweight breast tumor segmentation model using ultrasound images via dual-path joint knowledge distillation (Hongjiang Guo et al., 2023)

{{<citation>}}

Hongjiang Guo, Shengwen Wang, Hao Dang, Kangle Xiao, Yaru Yang, Wenpei Liu, Tongtong Liu, Yiying Wan. (2023)  
**LightBTSeg: A lightweight breast tumor segmentation model using ultrasound images via dual-path joint knowledge distillation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.11086v1)  

---


**ABSTRACT**  
The accurate segmentation of breast tumors is an important prerequisite for lesion detection, which has significant clinical value for breast tumor research. The mainstream deep learning-based methods have achieved a breakthrough. However, these high-performance segmentation methods are formidable to implement in clinical scenarios since they always embrace high computation complexity, massive parameters, slow inference speed, and huge memory consumption. To tackle this problem, we propose LightBTSeg, a dual-path joint knowledge distillation framework, for lightweight breast tumor segmentation. Concretely, we design a double-teacher model to represent the fine-grained feature of breast ultrasound according to different semantic feature realignments of benign and malignant breast tumors. Specifically, we leverage the bottleneck architecture to reconstruct the original Attention U-Net. It is regarded as a lightweight student model named Simplified U-Net. Then, the prior knowledge of benign and malignant categories is utilized to design the teacher network combined dual-path joint knowledge distillation, which distills the knowledge from cumbersome benign and malignant teachers to a lightweight student model. Extensive experiments conducted on breast ultrasound images (Dataset BUSI) and Breast Ultrasound Dataset B (Dataset B) datasets demonstrate that LightBTSeg outperforms various counterparts.

{{</citation>}}


### (55/62) Enhancing Transformer-Based Segmentation for Breast Cancer Diagnosis using Auto-Augmentation and Search Optimisation Techniques (Leon Hamnett et al., 2023)

{{<citation>}}

Leon Hamnett, Mary Adewunmi, Modinat Abayomi, Kayode Raheem, Fahad Ahmed. (2023)  
**Enhancing Transformer-Based Segmentation for Breast Cancer Diagnosis using Auto-Augmentation and Search Optimisation Techniques**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Augmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2311.11065v1)  

---


**ABSTRACT**  
Breast cancer remains a critical global health challenge, necessitating early and accurate detection for effective treatment. This paper introduces a methodology that combines automated image augmentation selection (RandAugment) with search optimisation strategies (Tree-based Parzen Estimator) to identify optimal values for the number of image augmentations and the magnitude of their associated augmentation parameters, leading to enhanced segmentation performance. We empirically validate our approach on breast cancer histology slides, focusing on the segmentation of cancer cells. A comparative analysis of state-of-the-art transformer-based segmentation models is conducted, including SegFormer, PoolFormer, and MaskFormer models, to establish a comprehensive baseline, before applying the augmentation methodology. Our results show that the proposed methodology leads to segmentation models that are more resilient to variations in histology slides whilst maintaining high levels of segmentation performance, and show improved segmentation of the tumour class when compared to previous research. Our best result after applying the augmentations is a Dice Score of 84.08 and an IoU score of 72.54 when segmenting the tumour class. The primary contribution of this paper is the development of a methodology that enhances segmentation performance while ensuring model robustness to data variances. This has significant implications for medical practitioners, enabling the development of more effective machine learning models for clinical applications to identify breast cancer cells from histology slides. Furthermore, the codebase accompanying this research will be released upon publication. This will facilitate further research and application development based on our methodology, thereby amplifying its impact.

{{</citation>}}


### (56/62) Structure-Aware Sparse-View X-ray 3D Reconstruction (Yuanhao Cai et al., 2023)

{{<citation>}}

Yuanhao Cai, Jiahao Wang, Alan Yuille, Zongwei Zhou, Angtian Wang. (2023)  
**Structure-Aware Sparse-View X-ray 3D Reconstruction**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.10959v1)  

---


**ABSTRACT**  
X-ray, known for its ability to reveal internal structures of objects, is expected to provide richer information for 3D reconstruction than visible light. Yet, existing neural radiance fields (NeRF) algorithms overlook this important nature of X-ray, leading to their limitations in capturing structural contents of imaged objects. In this paper, we propose a framework, Structure-Aware X-ray Neural Radiodensity Fields (SAX-NeRF), for sparse-view X-ray 3D reconstruction. Firstly, we design a Line Segment-based Transformer (Lineformer) as the backbone of SAX-NeRF. Linefomer captures internal structures of objects in 3D space by modeling the dependencies within each line segment of an X-ray. Secondly, we present a Masked Local-Global (MLG) ray sampling strategy to extract contextual and geometric information in 2D projection. Plus, we collect a larger-scale dataset X3D covering wider X-ray applications. Experiments on X3D show that SAX-NeRF surpasses previous NeRF-based methods by 12.56 and 2.49 dB on novel view synthesis and CT reconstruction. Code, models, and data will be released at https://github.com/caiyuanhao1998/SAX-NeRF

{{</citation>}}


## cs.IR (2)



### (57/62) SBTRec- A Transformer Framework for Personalized Tour Recommendation Problem with Sentiment Analysis (Ngai Lam Ho et al., 2023)

{{<citation>}}

Ngai Lam Ho, Roy Ka-Wei Lee, Kwan Hui Lim. (2023)  
**SBTRec- A Transformer Framework for Personalized Tour Recommendation Problem with Sentiment Analysis**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs-SI, cs.IR  
Keywords: BERT, Sentiment Analysis, Transformer  
[Paper Link](http://arxiv.org/abs/2311.11071v1)  

---


**ABSTRACT**  
When traveling to an unfamiliar city for holidays, tourists often rely on guidebooks, travel websites, or recommendation systems to plan their daily itineraries and explore popular points of interest (POIs). However, these approaches may lack optimization in terms of time feasibility, localities, and user preferences. In this paper, we propose the SBTRec algorithm: a BERT-based Trajectory Recommendation with sentiment analysis, for recommending personalized sequences of POIs as itineraries. The key contributions of this work include analyzing users' check-ins and uploaded photos to understand the relationship between POI visits and distance. We introduce SBTRec, which encompasses sentiment analysis to improve recommendation accuracy by understanding users' preferences and satisfaction levels from reviews and comments about different POIs. Our proposed algorithms are evaluated against other sequence prediction methods using datasets from 8 cities. The results demonstrate that SBTRec achieves an average F1 score of 61.45%, outperforming baseline algorithms.   The paper further discusses the flexibility of the SBTRec algorithm, its ability to adapt to different scenarios and cities without modification, and its potential for extension by incorporating additional information for more reliable predictions. Overall, SBTRec provides personalized and relevant POI recommendations, enhancing tourists' overall trip experiences. Future work includes fine-tuning personalized embeddings for users, with evaluation of users' comments on POIs,~to further enhance prediction accuracy.

{{</citation>}}


### (58/62) RecExplainer: Aligning Large Language Models for Recommendation Model Interpretability (Yuxuan Lei et al., 2023)

{{<citation>}}

Yuxuan Lei, Jianxun Lian, Jing Yao, Xu Huang, Defu Lian, Xing Xie. (2023)  
**RecExplainer: Aligning Large Language Models for Recommendation Model Interpretability**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.10947v1)  

---


**ABSTRACT**  
Recommender systems are widely used in various online services, with embedding-based models being particularly popular due to their expressiveness in representing complex signals. However, these models often lack interpretability, making them less reliable and transparent for both users and developers. With the emergence of large language models (LLMs), we find that their capabilities in language expression, knowledge-aware reasoning, and instruction following are exceptionally powerful. Based on this, we propose a new model interpretation approach for recommender systems, by using LLMs as surrogate models and learn to mimic and comprehend target recommender models. Specifically, we introduce three alignment methods: behavior alignment, intention alignment, and hybrid alignment. Behavior alignment operates in the language space, representing user preferences and item information as text to learn the recommendation model's behavior; intention alignment works in the latent space of the recommendation model, using user and item representations to understand the model's behavior; hybrid alignment combines both language and latent spaces for alignment training. To demonstrate the effectiveness of our methods, we conduct evaluation from two perspectives: alignment effect, and explanation generation ability on three public datasets. Experimental results indicate that our approach effectively enables LLMs to comprehend the patterns of recommendation models and generate highly credible recommendation explanations.

{{</citation>}}


## cond-mat.mtrl-sci (1)



### (59/62) AIMS-EREA -- A framework for AI-accelerated Innovation of Materials for Sustainability -- for Environmental Remediation and Energy Applications (Sudarson Roy Pratihar et al., 2023)

{{<citation>}}

Sudarson Roy Pratihar, Deepesh Pai, Manaswita Nag. (2023)  
**AIMS-EREA -- A framework for AI-accelerated Innovation of Materials for Sustainability -- for Environmental Remediation and Energy Applications**  

---
Primary Category: cond-mat.mtrl-sci  
Categories: cond-mat-mtrl-sci, cond-mat.mtrl-sci, cs-AI  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.11060v1)  

---


**ABSTRACT**  
Many environmental remediation and energy applications (conversion and storage) for sustainability need design and development of green novel materials. Discovery processes of such novel materials are time taking and cumbersome due to large number of possible combinations and permutations of materials structures. Often theoretical studies based on Density Functional Theory (DFT) and other theories, coupled with Simulations are conducted to narrow down sample space of candidate materials, before conducting laboratory-based synthesis and analytical process. With the emergence of artificial intelligence (AI), AI techniques are being tried in this process too to ease out simulation time and cost. However tremendous values of previously published research from various parts of the world are still left as labor-intensive manual effort and discretion of individual researcher and prone to human omissions. AIMS-EREA is our novel framework to blend best of breed of Material Science theory with power of Generative AI to give best impact and smooth and quickest discovery of material for sustainability. This also helps to eliminate the possibility of production of hazardous residues and bye-products of the reactions. AIMS-EREA uses all available resources -- Predictive and Analytical AI on large collection of chemical databases along with automated intelligent assimilation of deep materials knowledge from previously published research works through Generative AI. We demonstrate use of our own novel framework with an example, how this framework can be successfully applied to achieve desired success in development of thermoelectric material for waste heat conversion.

{{</citation>}}


## cs.HC (2)



### (60/62) Images Connect Us Together: Navigating a COVID-19 Local Outbreak in China Through Social Media Images (Changyang He et al., 2023)

{{<citation>}}

Changyang He, Lu He, Wenjie Yang, Bo Li. (2023)  
**Images Connect Us Together: Navigating a COVID-19 Local Outbreak in China Through Social Media Images**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-SI, cs.HC  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2311.10977v1)  

---


**ABSTRACT**  
Social media images, curated or casual, have become a crucial component of communicating situational information and emotions during health crises. Despite its prevalence and significance in informational dissemination and emotional connection, there lacks a comprehensive understanding of visual crisis communication in the aftermath of a pandemic which is characterized by uncertain local situations and emotional fatigue. To fill this gap, this work collected 345,423 crisis-related posts and 65,376 original images during the Xi'an COVID-19 local outbreak in China, and adopted a mixed-methods approach to understanding themes, goals, and strategies of crisis imagery. Image clustering captured the diversity of visual themes during the outbreak, such as text images embedding authoritative guidelines and ``visual diaries'' recording and sharing the quarantine life. Through text classification of the post that visuals were situated in, we found that different visual themes highly correlated with the informational and emotional goals of the post text, such as adopting text images to convey the latest policies and sharing food images to express anxiety. We further unpacked nuanced strategies of crisis image use through inductive coding, such as signifying authority and triggering empathy. We discuss the opportunities and challenges of crisis imagery and provide design implications to facilitate effective visual crisis communication.

{{</citation>}}


### (61/62) Engage Wider Audience or Facilitate Quality Answers? a Mixed-methods Analysis of Questioning Strategies for Research Sensemaking on a Community Q&A Site (Changyang He et al., 2023)

{{<citation>}}

Changyang He, Yue Deng, Lu He, Qingyu Guo, Yu Zhang, Zhicong Lu, Bo Li. (2023)  
**Engage Wider Audience or Facilitate Quality Answers? a Mixed-methods Analysis of Questioning Strategies for Research Sensemaking on a Community Q&A Site**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.10975v1)  

---


**ABSTRACT**  
Discussing research-sensemaking questions on Community Question and Answering (CQA) platforms has been an increasingly common practice for the public to participate in science communication. Nonetheless, how users strategically craft research-sensemaking questions to engage public participation and facilitate knowledge construction is a significant yet less understood problem. To fill this gap, we collected 837 science-related questions and 157,684 answers from Zhihu, and conducted a mixed-methods study to explore user-developed strategies in proposing research-sensemaking questions, and their potential effects on public engagement and knowledge construction. Through open coding, we captured a comprehensive taxonomy of question-crafting strategies, such as eyecatching narratives with counter-intuitive claims and rigorous descriptions with data use. Regression analysis indicated that these strategies correlated with user engagement and answer construction in different ways (e.g., emotional questions attracted more views and answers), yet there existed a general divergence between wide participation and quality knowledge establishment, when most questioning strategies could not ensure both. Based on log analysis, we further found that collaborative editing afforded unique values in refining research-sensemaking questions regarding accuracy, rigor, comprehensiveness and attractiveness. We propose design implications to facilitate accessible, accurate and engaging science communication on CQA platforms.

{{</citation>}}


## cs.CR (1)



### (62/62) Reveal the Mathematical Structures of Honeyword Security Metrics (Pengcheng Su et al., 2023)

{{<citation>}}

Pengcheng Su, Haibo Cheng, Wenting Li, Ping Wang. (2023)  
**Reveal the Mathematical Structures of Honeyword Security Metrics**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.10960v1)  

---


**ABSTRACT**  
Honeyword is a representative ``honey" technique to detect intruders by luring them with decoy data. This kind of honey technique blends a primary object (from distribution $P$) with decoy samples (from distribution $Q$). In this research, we focus on two key Honeyword security metrics: the flatness function and the success-number function. Previous researchers are engaged in designing experimental methods to estimate their values. We've derived theoretical formulas on both metrics of the strongest $\mathcal{A}$ using the optimal guessing strategy, marking a first in the field.   The mathematical structures of these metrics are intriguing: the flatness function has an expression as $\epsilon(i)=\sum_{j=1}^{i}\int_{0}^{+\infty}\tbinom{k-1}{j-1} f(x)G^{k-j}(x)(1-G(x))^{j-1}dx$. In particular, the most important one, $\epsilon(1)$ is $\frac{1}{k}(M-\int_{0}^{M}G^k(x)dx)+b$, where $M=\max_{x: Q(x)\neq 0}\frac{P(x)}{Q(x)}$, $b=\sum_{x: Q(x)=0}P(x)$, and $G$ is a cumulative distribution function derived from $P$ and $Q$. This formula provides a criterion to compare different honey distributions: the one with smaller $M$ and $b$ is more satisfactory. The mathematical structure of the success-number function is a series of convolutions with beta distribution kernels: $\lambda_U(i)=U\sum_{j=1}^{i}\int_{\frac{1}{k}}^{1} \frac{\phi(x)}{1-\phi(x)} \tbinom{U-1}{j-1} x^{U-j}(1-x)^{j-1}dx$, where $U$ is the number of users in the system and $\phi(x)$ is a monotonically increasing function. For further elaboration, we made some representative calculations. Our findings offer insights into security assessments for Honeyword and similar honey techniques, contributing to enhanced security measures in these systems.

{{</citation>}}
