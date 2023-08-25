---
draft: false
title: "arXiv @ 2023.08.25"
date: 2023-08-25
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.25"
    identifier: arxiv_20230825
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (39)](#cscv-39)
- [cs.CL (18)](#cscl-18)
- [eess.SP (1)](#eesssp-1)
- [cs.CR (6)](#cscr-6)
- [cs.LG (16)](#cslg-16)
- [eess.IV (2)](#eessiv-2)
- [cs.IR (4)](#csir-4)
- [cs.SE (5)](#csse-5)
- [cs.AI (3)](#csai-3)
- [cs.IT (2)](#csit-2)
- [cs.CY (2)](#cscy-2)
- [q-bio.QM (1)](#q-bioqm-1)
- [q-fin.PM (1)](#q-finpm-1)
- [cs.HC (2)](#cshc-2)
- [physics.geo-ph (1)](#physicsgeo-ph-1)
- [cs.RO (2)](#csro-2)
- [cs.NI (1)](#csni-1)
- [q-bio.TO (1)](#q-bioto-1)
- [cs.DC (1)](#csdc-1)
- [eess.AS (1)](#eessas-1)
- [cs.SD (1)](#cssd-1)
- [cs.MA (1)](#csma-1)

## cs.CV (39)



### (1/111) Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion (Junjiao Tian et al., 2023)

{{<citation>}}

Junjiao Tian, Lavisha Aggarwal, Andrea Colaco, Zsolt Kira, Mar Gonzalez-Franco. (2023)  
**Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.12469v1)  

---


**ABSTRACT**  
Producing quality segmentation masks for images is a fundamental problem in computer vision. Recent research has explored large-scale supervised training to enable zero-shot segmentation on virtually any image style and unsupervised training to enable segmentation without dense annotations. However, constructing a model capable of segmenting anything in a zero-shot manner without any annotations is still challenging. In this paper, we propose to utilize the self-attention layers in stable diffusion models to achieve this goal because the pre-trained stable diffusion model has learned inherent concepts of objects within its attention layers. Specifically, we introduce a simple yet effective iterative merging process based on measuring KL divergence among attention maps to merge them into valid segmentation masks. The proposed method does not require any training or language dependency to extract quality segmentation for any images. On COCO-Stuff-27, our method surpasses the prior unsupervised zero-shot SOTA method by an absolute 26% in pixel accuracy and 17% in mean IoU.

{{</citation>}}


### (2/111) Augmenting medical image classifiers with synthetic data from latent diffusion models (Luke W. Sagers et al., 2023)

{{<citation>}}

Luke W. Sagers, James A. Diao, Luke Melas-Kyriazi, Matthew Groh, Pranav Rajpurkar, Adewole S. Adamson, Veronica Rotemberg, Roxana Daneshjou, Arjun K. Manrai. (2023)  
**Augmenting medical image classifiers with synthetic data from latent diffusion models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.12453v1)  

---


**ABSTRACT**  
While hundreds of artificial intelligence (AI) algorithms are now approved or cleared by the US Food and Drugs Administration (FDA), many studies have shown inconsistent generalization or latent bias, particularly for underrepresented populations. Some have proposed that generative AI could reduce the need for real data, but its utility in model development remains unclear. Skin disease serves as a useful case study in synthetic image generation due to the diversity of disease appearance, particularly across the protected attribute of skin tone. Here we show that latent diffusion models can scalably generate images of skin disease and that augmenting model training with these data improves performance in data-limited settings. These performance gains saturate at synthetic-to-real image ratios above 10:1 and are substantially smaller than the gains obtained from adding real images. As part of our analysis, we generate and analyze a new dataset of 458,920 synthetic images produced using several generation strategies. Our results suggest that synthetic data could serve as a force-multiplier for model development, but the collection of diverse real-world data remains the most important step to improve medical AI algorithms.

{{</citation>}}


### (3/111) MOFO: MOtion FOcused Self-Supervision for Video Understanding (Mona Ahmadian et al., 2023)

{{<citation>}}

Mona Ahmadian, Frank Guerin, Andrew Gilbert. (2023)  
**MOFO: MOtion FOcused Self-Supervision for Video Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.12447v1)  

---


**ABSTRACT**  
Self-supervised learning (SSL) techniques have recently produced outstanding results in learning visual representations from unlabeled videos. Despite the importance of motion in supervised learning techniques for action recognition, SSL methods often do not explicitly consider motion information in videos. To address this issue, we propose MOFO (MOtion FOcused), a novel SSL method for focusing representation learning on the motion area of a video, for action recognition. MOFO automatically detects motion areas in videos and uses these to guide the self-supervision task. We use a masked autoencoder which randomly masks out a high proportion of the input sequence; we force a specified percentage of the inside of the motion area to be masked and the remainder from outside. We further incorporate motion information into the finetuning step to emphasise motion in the downstream task. We demonstrate that our motion-focused innovations can significantly boost the performance of the currently leading SSL method (VideoMAE) for action recognition. Our method improves the recent self-supervised Vision Transformer (ViT), VideoMAE, by achieving +2.6%, +2.1%, +1.3% accuracy on Epic-Kitchens verb, noun and action classification, respectively, and +4.7% accuracy on Something-Something V2 action classification. Our proposed approach significantly improves the performance of the current SSL method for action recognition, indicating the importance of explicitly encoding motion in SSL.

{{</citation>}}


### (4/111) Self-Supervised Learning for Endoscopic Video Analysis (Roy Hirsch et al., 2023)

{{<citation>}}

Roy Hirsch, Mathilde Caron, Regev Cohen, Amir Livne, Ron Shapiro, Tomer Golany, Roman Goldenberg, Daniel Freedman, Ehud Rivlin. (2023)  
**Self-Supervised Learning for Endoscopic Video Analysis**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.12394v1)  

---


**ABSTRACT**  
Self-supervised learning (SSL) has led to important breakthroughs in computer vision by allowing learning from large amounts of unlabeled data. As such, it might have a pivotal role to play in biomedicine where annotating data requires a highly specialized expertise. Yet, there are many healthcare domains for which SSL has not been extensively explored. One such domain is endoscopy, minimally invasive procedures which are commonly used to detect and treat infections, chronic inflammatory diseases or cancer. In this work, we study the use of a leading SSL framework, namely Masked Siamese Networks (MSNs), for endoscopic video analysis such as colonoscopy and laparoscopy. To fully exploit the power of SSL, we create sizable unlabeled endoscopic video datasets for training MSNs. These strong image representations serve as a foundation for secondary training with limited annotated datasets, resulting in state-of-the-art performance in endoscopic benchmarks like surgical phase recognition during laparoscopy and colonoscopic polyp characterization. Additionally, we achieve a 50% reduction in annotated data size without sacrificing performance. Thus, our work provides evidence that SSL can dramatically reduce the need of annotated data in endoscopy.

{{</citation>}}


### (5/111) With a Little Help from your own Past: Prototypical Memory Networks for Image Captioning (Manuele Barraco et al., 2023)

{{<citation>}}

Manuele Barraco, Sara Sarto, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara. (2023)  
**With a Little Help from your own Past: Prototypical Memory Networks for Image Captioning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-MM, cs.CV  
Keywords: Image Captioning, Transformer  
[Paper Link](http://arxiv.org/abs/2308.12383v1)  

---


**ABSTRACT**  
Image captioning, like many tasks involving vision and language, currently relies on Transformer-based architectures for extracting the semantics in an image and translating it into linguistically coherent descriptions. Although successful, the attention operator only considers a weighted summation of projections of the current input sample, therefore ignoring the relevant semantic information which can come from the joint observation of other samples. In this paper, we devise a network which can perform attention over activations obtained while processing other training samples, through a prototypical memory model. Our memory models the distribution of past keys and values through the definition of prototype vectors which are both discriminative and compact. Experimentally, we assess the performance of the proposed model on the COCO dataset, in comparison with carefully designed baselines and state-of-the-art approaches, and by investigating the role of each of the proposed components. We demonstrate that our proposal can increase the performance of an encoder-decoder Transformer by 3.7 CIDEr points both when training in cross-entropy only and when fine-tuning with self-critical sequence training. Source code and trained models are available at: https://github.com/aimagelab/PMA-Net.

{{</citation>}}


### (6/111) Vision Transformer Adapters for Generalizable Multitask Learning (Deblina Bhattacharjee et al., 2023)

{{<citation>}}

Deblina Bhattacharjee, Sabine Süsstrunk, Mathieu Salzmann. (2023)  
**Vision Transformer Adapters for Generalizable Multitask Learning**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.12372v1)  

---


**ABSTRACT**  
We introduce the first multitasking vision transformer adapters that learn generalizable task affinities which can be applied to novel tasks and domains. Integrated into an off-the-shelf vision transformer backbone, our adapters can simultaneously solve multiple dense vision tasks in a parameter-efficient manner, unlike existing multitasking transformers that are parametrically expensive. In contrast to concurrent methods, we do not require retraining or fine-tuning whenever a new task or domain is added. We introduce a task-adapted attention mechanism within our adapter framework that combines gradient-based task similarities with attention-based ones. The learned task affinities generalize to the following settings: zero-shot task transfer, unsupervised domain adaptation, and generalization without fine-tuning to novel domains. We demonstrate that our approach outperforms not only the existing convolutional neural network-based multitasking methods but also the vision transformer-based ones. Our project page is at \url{https://ivrl.github.io/VTAGML}.

{{</citation>}}


### (7/111) Open-set Face Recognition with Neural Ensemble, Maximal Entropy Loss and Feature Augmentation (Rafael Henrique Vareto et al., 2023)

{{<citation>}}

Rafael Henrique Vareto, Manuel Günther, William Robson Schwartz. (2023)  
**Open-set Face Recognition with Neural Ensemble, Maximal Entropy Loss and Feature Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.12371v1)  

---


**ABSTRACT**  
Open-set face recognition refers to a scenario in which biometric systems have incomplete knowledge of all existing subjects. Therefore, they are expected to prevent face samples of unregistered subjects from being identified as previously enrolled identities. This watchlist context adds an arduous requirement that calls for the dismissal of irrelevant faces by focusing mainly on subjects of interest. As a response, this work introduces a novel method that associates an ensemble of compact neural networks with a margin-based cost function that explores additional samples. Supplementary negative samples can be obtained from external databases or synthetically built at the representation level in training time with a new mix-up feature augmentation approach. Deep neural networks pre-trained on large face datasets serve as the preliminary feature extraction module. We carry out experiments on well-known LFW and IJB-C datasets where results show that the approach is able to boost closed and open-set identification rates.

{{</citation>}}


### (8/111) Continual Zero-Shot Learning through Semantically Guided Generative Random Walks (Wenxuan Zhang et al., 2023)

{{<citation>}}

Wenxuan Zhang, Paul Janson, Kai Yi, Ivan Skorokhodov, Mohamed Elhoseiny. (2023)  
**Continual Zero-Shot Learning through Semantically Guided Generative Random Walks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.12366v1)  

---


**ABSTRACT**  
Learning novel concepts, remembering previous knowledge, and adapting it to future tasks occur simultaneously throughout a human's lifetime. To model such comprehensive abilities, continual zero-shot learning (CZSL) has recently been introduced. However, most existing methods overused unseen semantic information that may not be continually accessible in realistic settings. In this paper, we address the challenge of continual zero-shot learning where unseen information is not provided during training, by leveraging generative modeling. The heart of the generative-based methods is to learn quality representations from seen classes to improve the generative understanding of the unseen visual space. Motivated by this, we introduce generalization-bound tools and provide the first theoretical explanation for the benefits of generative modeling to CZSL tasks. Guided by the theoretical analysis, we then propose our learning algorithm that employs a novel semantically guided Generative Random Walk (GRW) loss. The GRW loss augments the training by continually encouraging the model to generate realistic and characterized samples to represent the unseen space. Our algorithm achieves state-of-the-art performance on AWA1, AWA2, CUB, and SUN datasets, surpassing existing CZSL methods by 3-7\%. The code has been made available here \url{https://github.com/wx-zhang/IGCZSL}

{{</citation>}}


### (9/111) Saliency-based Video Summarization for Face Anti-spoofing (Usman Muhammad et al., 2023)

{{<citation>}}

Usman Muhammad, Mourad Oussalah, Md Ziaul Hoque, Jorma Laaksonen. (2023)  
**Saliency-based Video Summarization for Face Anti-spoofing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2308.12364v1)  

---


**ABSTRACT**  
Due to the growing availability of face anti-spoofing databases, researchers are increasingly focusing on video-based methods that use hundreds to thousands of images to assess their impact on performance. However, there is no clear consensus on the exact number of frames in a video required to improve the performance of face anti-spoofing tasks. Inspired by the visual saliency theory, we present a video summarization method for face anti-spoofing tasks that aims to enhance the performance and efficiency of deep learning models by leveraging visual saliency. In particular, saliency information is extracted from the differences between the Laplacian and Wiener filter outputs of the source images, enabling identification of the most visually salient regions within each frame. Subsequently, the source images are decomposed into base and detail layers, enhancing representation of important information. The weighting maps are then computed based on the saliency information, indicating the importance of each pixel in the image. By linearly combining the base and detail layers using the weighting maps, the method fuses the source images to create a single representative image that summarizes the entire video. The key contribution of our proposed method lies in demonstrating how visual saliency can be used as a data-centric approach to improve the performance and efficiency of face presentation attack detection models. By focusing on the most salient images or regions within the images, a more representative and diverse training set can be created, potentially leading to more effective models. To validate the method's effectiveness, a simple deep learning architecture (CNN-RNN) was used, and the experimental results showcased state-of-the-art performance on five challenging face anti-spoofing datasets.

{{</citation>}}


### (10/111) Diffusion-based Image Translation with Label Guidance for Domain Adaptive Semantic Segmentation (Duo Peng et al., 2023)

{{<citation>}}

Duo Peng, Ping Hu, Qiuhong Ke, Jun Liu. (2023)  
**Diffusion-based Image Translation with Label Guidance for Domain Adaptive Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.12350v1)  

---


**ABSTRACT**  
Translating images from a source domain to a target domain for learning target models is one of the most common strategies in domain adaptive semantic segmentation (DASS). However, existing methods still struggle to preserve semantically-consistent local details between the original and translated images. In this work, we present an innovative approach that addresses this challenge by using source-domain labels as explicit guidance during image translation. Concretely, we formulate cross-domain image translation as a denoising diffusion process and utilize a novel Semantic Gradient Guidance (SGG) method to constrain the translation process, conditioning it on the pixel-wise source labels. Additionally, a Progressive Translation Learning (PTL) strategy is devised to enable the SGG method to work reliably across domains with large gaps. Extensive experiments demonstrate the superiority of our approach over state-of-the-art methods.

{{</citation>}}


### (11/111) A Generative Approach for Image Registration of Visible-Thermal (VT) Cancer Faces (Catherine Ordun et al., 2023)

{{<citation>}}

Catherine Ordun, Alexandra Cha, Edward Raff, Sanjay Purushotham, Karen Kwok, Mason Rule, James Gulley. (2023)  
**A Generative Approach for Image Registration of Visible-Thermal (VT) Cancer Faces**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.12271v1)  

---


**ABSTRACT**  
Since thermal imagery offers a unique modality to investigate pain, the U.S. National Institutes of Health (NIH) has collected a large and diverse set of cancer patient facial thermograms for AI-based pain research. However, differing angles from camera capture between thermal and visible sensors has led to misalignment between Visible-Thermal (VT) images. We modernize the classic computer vision task of image registration by applying and modifying a generative alignment algorithm to register VT cancer faces, without the need for a reference or alignment parameters. By registering VT faces, we demonstrate that the quality of thermal images produced in the generative AI downstream task of Visible-to-Thermal (V2T) image translation significantly improves up to 52.5\%, than without registration. Images in this paper have been approved by the NIH NCI for public dissemination.

{{</citation>}}


### (12/111) MolGrapher: Graph-based Visual Recognition of Chemical Structures (Lucas Morin et al., 2023)

{{<citation>}}

Lucas Morin, Martin Danelljan, Maria Isabel Agea, Ahmed Nassar, Valery Weber, Ingmar Meijer, Peter Staar, Fisher Yu. (2023)  
**MolGrapher: Graph-based Visual Recognition of Chemical Structures**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2308.12234v1)  

---


**ABSTRACT**  
The automatic analysis of chemical literature has immense potential to accelerate the discovery of new materials and drugs. Much of the critical information in patent documents and scientific articles is contained in figures, depicting the molecule structures. However, automatically parsing the exact chemical structure is a formidable challenge, due to the amount of detailed information, the diversity of drawing styles, and the need for training data. In this work, we introduce MolGrapher to recognize chemical structures visually. First, a deep keypoint detector detects the atoms. Second, we treat all candidate atoms and bonds as nodes and put them in a graph. This construct allows a natural graph representation of the molecule. Last, we classify atom and bond nodes in the graph with a Graph Neural Network. To address the lack of real training data, we propose a synthetic data generation pipeline producing diverse and realistic results. In addition, we introduce a large-scale benchmark of annotated real molecule images, USPTO-30K, to spur research on this critical topic. Extensive experiments on five datasets show that our approach significantly outperforms classical and learning-based methods in most settings. Code, models, and datasets are available.

{{</citation>}}


### (13/111) SG-Former: Self-guided Transformer with Evolving Token Reallocation (Sucheng Ren et al., 2023)

{{<citation>}}

Sucheng Ren, Xingyi Yang, Songhua Liu, Xinchao Wang. (2023)  
**SG-Former: Self-guided Transformer with Evolving Token Reallocation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2308.12216v1)  

---


**ABSTRACT**  
Vision Transformer has demonstrated impressive success across various vision tasks. However, its heavy computation cost, which grows quadratically with respect to the token sequence length, largely limits its power in handling large feature maps. To alleviate the computation cost, previous works rely on either fine-grained self-attentions restricted to local small regions, or global self-attentions but to shorten the sequence length resulting in coarse granularity. In this paper, we propose a novel model, termed as Self-guided Transformer~(SG-Former), towards effective global self-attention with adaptive fine granularity. At the heart of our approach is to utilize a significance map, which is estimated through hybrid-scale self-attention and evolves itself during training, to reallocate tokens based on the significance of each region. Intuitively, we assign more tokens to the salient regions for achieving fine-grained attention, while allocating fewer tokens to the minor regions in exchange for efficiency and global receptive fields. The proposed SG-Former achieves performance superior to state of the art: our base size model achieves \textbf{84.7\%} Top-1 accuracy on ImageNet-1K, \textbf{51.2mAP} bbAP on CoCo, \textbf{52.7mIoU} on ADE20K surpassing the Swin Transformer by \textbf{+1.3\% / +2.7 mAP/ +3 mIoU}, with lower computation costs and fewer parameters. The code is available at \href{https://github.com/OliverRensu/SG-Former}{https://github.com/OliverRensu/SG-Former}

{{</citation>}}


### (14/111) CLIPN for Zero-Shot OOD Detection: Teaching CLIP to Say No (Hualiang Wang et al., 2023)

{{<citation>}}

Hualiang Wang, Yi Li, Huifeng Yao, Xiaomeng Li. (2023)  
**CLIPN for Zero-Shot OOD Detection: Teaching CLIP to Say No**  

---
Primary Category: cs.CV  
Categories: 68T45, I-4-9, cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.12213v2)  

---


**ABSTRACT**  
Out-of-distribution (OOD) detection refers to training the model on an in-distribution (ID) dataset to classify whether the input images come from unknown classes. Considerable effort has been invested in designing various OOD detection methods based on either convolutional neural networks or transformers. However, zero-shot OOD detection methods driven by CLIP, which only require class names for ID, have received less attention. This paper presents a novel method, namely CLIP saying no (CLIPN), which empowers the logic of saying no within CLIP. Our key motivation is to equip CLIP with the capability of distinguishing OOD and ID samples using positive-semantic prompts and negation-semantic prompts. Specifically, we design a novel learnable no prompt and a no text encoder to capture negation semantics within images. Subsequently, we introduce two loss functions: the image-text binary-opposite loss and the text semantic-opposite loss, which we use to teach CLIPN to associate images with no prompts, thereby enabling it to identify unknown samples. Furthermore, we propose two threshold-free inference algorithms to perform OOD detection by utilizing negation semantics from no prompts and the text encoder. Experimental results on 9 benchmark datasets (3 ID datasets and 6 OOD datasets) for the OOD detection task demonstrate that CLIPN, based on ViT-B-16, outperforms 7 well-used algorithms by at least 2.34% and 11.64% in terms of AUROC and FPR95 for zero-shot OOD detection on ImageNet-1K. Our CLIPN can serve as a solid foundation for effectively leveraging CLIP in downstream OOD tasks. The code is available on https://github.com/xmed-lab/CLIPN.

{{</citation>}}


### (15/111) Multimodal Latent Emotion Recognition from Micro-expression and Physiological Signals (Liangfei Zhang et al., 2023)

{{<citation>}}

Liangfei Zhang, Yifei Qian, Ognjen Arandjelovic, Anthony Zhu. (2023)  
**Multimodal Latent Emotion Recognition from Micro-expression and Physiological Signals**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2308.12156v1)  

---


**ABSTRACT**  
This paper discusses the benefits of incorporating multimodal data for improving latent emotion recognition accuracy, focusing on micro-expression (ME) and physiological signals (PS). The proposed approach presents a novel multimodal learning framework that combines ME and PS, including a 1D separable and mixable depthwise inception network, a standardised normal distribution weighted feature fusion method, and depth/physiology guided attention modules for multimodal learning. Experimental results show that the proposed approach outperforms the benchmark method, with the weighted fusion method and guided attention modules both contributing to enhanced performance.

{{</citation>}}


### (16/111) Masking Strategies for Background Bias Removal in Computer Vision Models (Ananthu Aniraj et al., 2023)

{{<citation>}}

Ananthu Aniraj, Cassio F. Dantas, Dino Ienco, Diego Marcos. (2023)  
**Masking Strategies for Background Bias Removal in Computer Vision Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Bias, Computer Vision, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.12127v1)  

---


**ABSTRACT**  
Models for fine-grained image classification tasks, where the difference between some classes can be extremely subtle and the number of samples per class tends to be low, are particularly prone to picking up background-related biases and demand robust methods to handle potential examples with out-of-distribution (OOD) backgrounds. To gain deeper insights into this critical problem, our research investigates the impact of background-induced bias on fine-grained image classification, evaluating standard backbone models such as Convolutional Neural Network (CNN) and Vision Transformers (ViT). We explore two masking strategies to mitigate background-induced bias: Early masking, which removes background information at the (input) image level, and late masking, which selectively masks high-level spatial features corresponding to the background. Extensive experiments assess the behavior of CNN and ViT models under different masking strategies, with a focus on their generalization to OOD backgrounds. The obtained findings demonstrate that both proposed strategies enhance OOD performance compared to the baseline models, with early masking consistently exhibiting the best OOD performance. Notably, a ViT variant employing GAP-Pooled Patch token-based classification combined with early masking achieves the highest OOD robustness.

{{</citation>}}


### (17/111) Advancements in Point Cloud Data Augmentation for Deep Learning: A Survey (Qinfeng Zhu et al., 2023)

{{<citation>}}

Qinfeng Zhu, Lei Fan, Ningxin Weng. (2023)  
**Advancements in Point Cloud Data Augmentation for Deep Learning: A Survey**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.12113v1)  

---


**ABSTRACT**  
Point cloud has a wide range of applications in areas such as autonomous driving, mapping, navigation, scene reconstruction, and medical imaging. Due to its great potentials in these applications, point cloud processing has gained great attention in the field of computer vision. Among various point cloud processing techniques, deep learning (DL) has become one of the mainstream and effective methods for tasks such as detection, segmentation and classification. To reduce overfitting during training DL models and improve model performance especially when the amount and/or diversity of training data are limited, augmentation is often crucial. Although various point cloud data augmentation methods have been widely used in different point cloud processing tasks, there are currently no published systematic surveys or reviews of these methods. Therefore, this article surveys and discusses these methods and categorizes them into a taxonomy framework. Through the comprehensive evaluation and comparison of the augmentation methods, this article identifies their potentials and limitations and suggests possible future research directions. This work helps researchers gain a holistic understanding of the current status of point cloud data augmentation and promotes its wider application and development.

{{</citation>}}


### (18/111) Manipulating Embeddings of Stable Diffusion Prompts (Niklas Deckers et al., 2023)

{{<citation>}}

Niklas Deckers, Julia Peters, Martin Potthast. (2023)  
**Manipulating Embeddings of Stable Diffusion Prompts**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.12059v1)  

---


**ABSTRACT**  
Generative text-to-image models such as Stable Diffusion allow users to generate images based on a textual description, the prompt. Changing the prompt is still the primary means for the user to change a generated image as desired. However, changing the image by reformulating the prompt remains a difficult process of trial and error, which has led to the emergence of prompt engineering as a new field of research. We propose and analyze methods to change the embedding of a prompt directly instead of the prompt text. It allows for more fine-grained and targeted control that takes into account user intentions. Our approach treats the generative text-to-image model as a continuous function and passes gradients between the image space and the prompt embedding space. By addressing different user interaction problems, we can apply this idea in three scenarios: (1) Optimization of a metric defined in image space that could measure, for example, image style. (2) Assistance of users in creative tasks by enabling them to navigate the image space along a selection of directions of "near" prompt embeddings. (3) Changing the embedding of the prompt to include information that the user has seen in a particular seed but finds difficult to describe in the prompt. Our experiments demonstrate the feasibility of the described methods.

{{</citation>}}


### (19/111) Head-Tail Cooperative Learning Network for Unbiased Scene Graph Generation (Lei Wang et al., 2023)

{{<citation>}}

Lei Wang, Zejian Yuan, Yao Lu, Badong Chen. (2023)  
**Head-Tail Cooperative Learning Network for Unbiased Scene Graph Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.12048v1)  

---


**ABSTRACT**  
Scene Graph Generation (SGG) as a critical task in image understanding, facing the challenge of head-biased prediction caused by the long-tail distribution of predicates. However, current unbiased SGG methods can easily prioritize improving the prediction of tail predicates while ignoring the substantial sacrifice in the prediction of head predicates, leading to a shift from head bias to tail bias. To address this issue, we propose a model-agnostic Head-Tail Collaborative Learning (HTCL) network that includes head-prefer and tail-prefer feature representation branches that collaborate to achieve accurate recognition of both head and tail predicates. We also propose a self-supervised learning approach to enhance the prediction ability of the tail-prefer feature representation branch by constraining tail-prefer predicate features. Specifically, self-supervised learning converges head predicate features to their class centers while dispersing tail predicate features as much as possible through contrast learning and head center loss. We demonstrate the effectiveness of our HTCL by applying it to various SGG models on VG150, Open Images V6 and GQA200 datasets. The results show that our method achieves higher mean Recall with a minimal sacrifice in Recall and achieves a new state-of-the-art overall performance. Our code is available at https://github.com/wanglei0618/HTCL.

{{</citation>}}


### (20/111) CgT-GAN: CLIP-guided Text GAN for Image Captioning (Jiarui Yu et al., 2023)

{{<citation>}}

Jiarui Yu, Haoran Li, Yanbin Hao, Bin Zhu, Tong Xu, Xiangnan He. (2023)  
**CgT-GAN: CLIP-guided Text GAN for Image Captioning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-MM, cs.CV  
Keywords: Image Captioning  
[Paper Link](http://arxiv.org/abs/2308.12045v1)  

---


**ABSTRACT**  
The large-scale visual-language pre-trained model, Contrastive Language-Image Pre-training (CLIP), has significantly improved image captioning for scenarios without human-annotated image-caption pairs. Recent advanced CLIP-based image captioning without human annotations follows a text-only training paradigm, i.e., reconstructing text from shared embedding space. Nevertheless, these approaches are limited by the training/inference gap or huge storage requirements for text embeddings. Given that it is trivial to obtain images in the real world, we propose CLIP-guided text GAN (CgT-GAN), which incorporates images into the training process to enable the model to "see" real visual modality. Particularly, we use adversarial training to teach CgT-GAN to mimic the phrases of an external text corpus and CLIP-based reward to provide semantic guidance. The caption generator is jointly rewarded based on the caption naturalness to human language calculated from the GAN's discriminator and the semantic guidance reward computed by the CLIP-based reward module. In addition to the cosine similarity as the semantic guidance reward (i.e., CLIP-cos), we further introduce a novel semantic guidance reward called CLIP-agg, which aligns the generated caption with a weighted text embedding by attentively aggregating the entire corpus. Experimental results on three subtasks (ZS-IC, In-UIC and Cross-UIC) show that CgT-GAN outperforms state-of-the-art methods significantly across all metrics. Code is available at https://github.com/Lihr747/CgtGAN.

{{</citation>}}


### (21/111) Distribution-Aware Calibration for Object Detection with Noisy Bounding Boxes (Donghao Zhou et al., 2023)

{{<citation>}}

Donghao Zhou, Jialin Li, Jinpeng Li, Jiancheng Huang, Qiang Nie, Yong Liu, Bin-Bin Gao, Qiong Wang, Pheng-Ann Heng, Guangyong Chen. (2023)  
**Distribution-Aware Calibration for Object Detection with Noisy Bounding Boxes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.12017v1)  

---


**ABSTRACT**  
Large-scale well-annotated datasets are of great importance for training an effective object detector. However, obtaining accurate bounding box annotations is laborious and demanding. Unfortunately, the resultant noisy bounding boxes could cause corrupt supervision signals and thus diminish detection performance. Motivated by the observation that the real ground-truth is usually situated in the aggregation region of the proposals assigned to a noisy ground-truth, we propose DIStribution-aware CalibratiOn (DISCO) to model the spatial distribution of proposals for calibrating supervision signals. In DISCO, spatial distribution modeling is performed to statistically extract the potential locations of objects. Based on the modeled distribution, three distribution-aware techniques, i.e., distribution-aware proposal augmentation (DA-Aug), distribution-aware box refinement (DA-Ref), and distribution-aware confidence estimation (DA-Est), are developed to improve classification, localization, and interpretability, respectively. Extensive experiments on large-scale noisy image datasets (i.e., Pascal VOC and MS-COCO) demonstrate that DISCO can achieve state-of-the-art detection performance, especially at high noise levels.

{{</citation>}}


### (22/111) Multi-stage Factorized Spatio-Temporal Representation for RGB-D Action and Gesture Recognition (Yujun Ma et al., 2023)

{{<citation>}}

Yujun Ma, Benjia Zhou, Ruili Wang, Pichao Wang. (2023)  
**Multi-stage Factorized Spatio-Temporal Representation for RGB-D Action and Gesture Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.12006v1)  

---


**ABSTRACT**  
RGB-D action and gesture recognition remain an interesting topic in human-centered scene understanding, primarily due to the multiple granularities and large variation in human motion. Although many RGB-D based action and gesture recognition approaches have demonstrated remarkable results by utilizing highly integrated spatio-temporal representations across multiple modalities (i.e., RGB and depth data), they still encounter several challenges. Firstly, vanilla 3D convolution makes it hard to capture fine-grained motion differences between local clips under different modalities. Secondly, the intricate nature of highly integrated spatio-temporal modeling can lead to optimization difficulties. Thirdly, duplicate and unnecessary information can add complexity and complicate entangled spatio-temporal modeling. To address the above issues, we propose an innovative heuristic architecture called Multi-stage Factorized Spatio-Temporal (MFST) for RGB-D action and gesture recognition. The proposed MFST model comprises a 3D Central Difference Convolution Stem (CDC-Stem) module and multiple factorized spatio-temporal stages. The CDC-Stem enriches fine-grained temporal perception, and the multiple hierarchical spatio-temporal stages construct dimension-independent higher-order semantic primitives. Specifically, the CDC-Stem module captures bottom-level spatio-temporal features and passes them successively to the following spatio-temporal factored stages to capture the hierarchical spatial and temporal features through the Multi- Scale Convolution and Transformer (MSC-Trans) hybrid block and Weight-shared Multi-Scale Transformer (WMS-Trans) block. The seamless integration of these innovative designs results in a robust spatio-temporal representation that outperforms state-of-the-art approaches on RGB-D action and gesture recognition datasets.

{{</citation>}}


### (23/111) Local Distortion Aware Efficient Transformer Adaptation for Image Quality Assessment (Kangmin Xu et al., 2023)

{{<citation>}}

Kangmin Xu, Liang Liao, Jing Xiao, Chaofeng Chen, Haoning Wu, Qiong Yan, Weisi Lin. (2023)  
**Local Distortion Aware Efficient Transformer Adaptation for Image Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Transformer  
[Paper Link](http://arxiv.org/abs/2308.12001v1)  

---


**ABSTRACT**  
Image Quality Assessment (IQA) constitutes a fundamental task within the field of computer vision, yet it remains an unresolved challenge, owing to the intricate distortion conditions, diverse image contents, and limited availability of data. Recently, the community has witnessed the emergence of numerous large-scale pretrained foundation models, which greatly benefit from dramatically increased data and parameter capacities. However, it remains an open problem whether the scaling law in high-level tasks is also applicable to IQA task which is closely related to low-level clues. In this paper, we demonstrate that with proper injection of local distortion features, a larger pretrained and fixed foundation model performs better in IQA tasks. Specifically, for the lack of local distortion structure and inductive bias of vision transformer (ViT), alongside the large-scale pretrained ViT, we use another pretrained convolution neural network (CNN), which is well known for capturing the local structure, to extract multi-scale image features. Further, we propose a local distortion extractor to obtain local distortion features from the pretrained CNN and a local distortion injector to inject the local distortion features into ViT. By only training the extractor and injector, our method can benefit from the rich knowledge in the powerful foundation models and achieve state-of-the-art performance on popular IQA datasets, indicating that IQA is not only a low-level problem but also benefits from stronger high-level features drawn from large-scale pretrained models.

{{</citation>}}


### (24/111) EVE: Efficient Vision-Language Pre-training with Masked Prediction and Modality-Aware MoE (Junyi Chen et al., 2023)

{{<citation>}}

Junyi Chen, Longteng Guo, Jia Sun, Shuai Shao, Zehuan Yuan, Liang Lin, Dongyu Zhang. (2023)  
**EVE: Efficient Vision-Language Pre-training with Masked Prediction and Modality-Aware MoE**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs-MM, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.11971v1)  

---


**ABSTRACT**  
Building scalable vision-language models to learn from diverse, multimodal data remains an open challenge. In this paper, we introduce an Efficient Vision-languagE foundation model, namely EVE, which is one unified multimodal Transformer pre-trained solely by one unified pre-training task. Specifically, EVE encodes both vision and language within a shared Transformer network integrated with modality-aware sparse Mixture-of-Experts (MoE) modules, which capture modality-specific information by selectively switching to different experts. To unify pre-training tasks of vision and language, EVE performs masked signal modeling on image-text pairs to reconstruct masked signals, i.e., image pixels and text tokens, given visible signals. This simple yet effective pre-training objective accelerates training by 3.5x compared to the model pre-trained with Image-Text Contrastive and Image-Text Matching losses. Owing to the combination of the unified architecture and pre-training task, EVE is easy to scale up, enabling better downstream performance with fewer resources and faster training speed. Despite its simplicity, EVE achieves state-of-the-art performance on various vision-language downstream tasks, including visual question answering, visual reasoning, and image-text retrieval.

{{</citation>}}


### (25/111) Learning Bottleneck Transformer for Event Image-Voxel Feature Fusion based Classification (Chengguo Yuan et al., 2023)

{{<citation>}}

Chengguo Yuan, Yu Jin, Zongzhen Wu, Fanting Wei, Yangzirui Wang, Lan Chen, Xiao Wang. (2023)  
**Learning Bottleneck Transformer for Event Image-Voxel Feature Fusion based Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: GNN, Graph Neural Network, Transformer  
[Paper Link](http://arxiv.org/abs/2308.11937v1)  

---


**ABSTRACT**  
Recognizing target objects using an event-based camera draws more and more attention in recent years. Existing works usually represent the event streams into point-cloud, voxel, image, etc, and learn the feature representations using various deep neural networks. Their final results may be limited by the following factors: monotonous modal expressions and the design of the network structure. To address the aforementioned challenges, this paper proposes a novel dual-stream framework for event representation, extraction, and fusion. This framework simultaneously models two common representations: event images and event voxels. By utilizing Transformer and Structured Graph Neural Network (GNN) architectures, spatial information and three-dimensional stereo information can be learned separately. Additionally, a bottleneck Transformer is introduced to facilitate the fusion of the dual-stream information. Extensive experiments demonstrate that our proposed framework achieves state-of-the-art performance on two widely used event-based classification datasets. The source code of this work is available at: \url{https://github.com/Event-AHU/EFV_event_classification}

{{</citation>}}


### (26/111) MixNet: Toward Accurate Detection of Challenging Scene Text in the Wild (Yu-Xiang Zeng et al., 2023)

{{<citation>}}

Yu-Xiang Zeng, Jun-Wei Hsieh, Xin Li, Ming-Ching Chang. (2023)  
**MixNet: Toward Accurate Detection of Challenging Scene Text in the Wild**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.12817v1)  

---


**ABSTRACT**  
Detecting small scene text instances in the wild is particularly challenging, where the influence of irregular positions and nonideal lighting often leads to detection errors. We present MixNet, a hybrid architecture that combines the strengths of CNNs and Transformers, capable of accurately detecting small text from challenging natural scenes, regardless of the orientations, styles, and lighting conditions. MixNet incorporates two key modules: (1) the Feature Shuffle Network (FSNet) to serve as the backbone and (2) the Central Transformer Block (CTBlock) to exploit the 1D manifold constraint of the scene text. We first introduce a novel feature shuffling strategy in FSNet to facilitate the exchange of features across multiple scales, generating high-resolution features superior to popular ResNet and HRNet. The FSNet backbone has achieved significant improvements over many existing text detection methods, including PAN, DB, and FAST. Then we design a complementary CTBlock to leverage center line based features similar to the medial axis of text regions and show that it can outperform contour-based approaches in challenging cases when small scene texts appear closely. Extensive experimental results show that MixNet, which mixes FSNet with CTBlock, achieves state-of-the-art results on multiple scene text detection datasets.

{{</citation>}}


### (27/111) Concept Bottleneck with Visual Concept Filtering for Explainable Medical Image Classification (Injae Kim et al., 2023)

{{<citation>}}

Injae Kim, Jongha Kim, Joonmyung Choi, Hyunwoo J. Kim. (2023)  
**Concept Bottleneck with Visual Concept Filtering for Explainable Medical Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Image Classification, Language Model  
[Paper Link](http://arxiv.org/abs/2308.11920v1)  

---


**ABSTRACT**  
Interpretability is a crucial factor in building reliable models for various medical applications. Concept Bottleneck Models (CBMs) enable interpretable image classification by utilizing human-understandable concepts as intermediate targets. Unlike conventional methods that require extensive human labor to construct the concept set, recent works leveraging Large Language Models (LLMs) for generating concepts made automatic concept generation possible. However, those methods do not consider whether a concept is visually relevant or not, which is an important factor in computing meaningful concept scores. Therefore, we propose a visual activation score that measures whether the concept contains visual cues or not, which can be easily computed with unlabeled image data. Computed visual activation scores are then used to filter out the less visible concepts, thus resulting in a final concept set with visually meaningful concepts. Our experimental results show that adopting the proposed visual activation score for concept filtering consistently boosts performance compared to the baseline. Moreover, qualitative analyses also validate that visually relevant concepts are successfully selected with the visual activation score.

{{</citation>}}


### (28/111) AMSP-UOD: When Vortex Convolution and Stochastic Perturbation Meet Underwater Object Detection (Jingchun Zhou et al., 2023)

{{<citation>}}

Jingchun Zhou, Zongxin He, Kin-Man Lam, Yudong Wang, Weishi Zhang, ChunLe Guo, Chongyi Li. (2023)  
**AMSP-UOD: When Vortex Convolution and Stochastic Perturbation Meet Underwater Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.11918v1)  

---


**ABSTRACT**  
In this paper, we present a novel Amplitude-Modulated Stochastic Perturbation and Vortex Convolutional Network, AMSP-UOD, designed for underwater object detection. AMSP-UOD specifically addresses the impact of non-ideal imaging factors on detection accuracy in complex underwater environments. To mitigate the influence of noise on object detection performance, we propose AMSP Vortex Convolution (AMSP-VConv) to disrupt the noise distribution, enhance feature extraction capabilities, effectively reduce parameters, and improve network robustness. We design the Feature Association Decoupling Cross Stage Partial (FAD-CSP) module, which strengthens the association of long and short-range features, improving the network performance in complex underwater environments. Additionally, our sophisticated post-processing method, based on non-maximum suppression with aspect-ratio similarity thresholds, optimizes detection in dense scenes, such as waterweed and schools of fish, improving object detection accuracy. Extensive experiments on the URPC and RUOD datasets demonstrate that our method outperforms existing state-of-the-art methods in terms of accuracy and noise immunity. AMSP-UOD proposes an innovative solution with the potential for real-world applications. Code will be made publicly available.

{{</citation>}}


### (29/111) LFS-GAN: Lifelong Few-Shot Image Generation (Juwon Seo et al., 2023)

{{<citation>}}

Juwon Seo, Ji-Su Kang, Gyeong-Moon Park. (2023)  
**LFS-GAN: Lifelong Few-Shot Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.11917v1)  

---


**ABSTRACT**  
We address a challenging lifelong few-shot image generation task for the first time. In this situation, a generative model learns a sequence of tasks using only a few samples per task. Consequently, the learned model encounters both catastrophic forgetting and overfitting problems at a time. Existing studies on lifelong GANs have proposed modulation-based methods to prevent catastrophic forgetting. However, they require considerable additional parameters and cannot generate high-fidelity and diverse images from limited data. On the other hand, the existing few-shot GANs suffer from severe catastrophic forgetting when learning multiple tasks. To alleviate these issues, we propose a framework called Lifelong Few-Shot GAN (LFS-GAN) that can generate high-quality and diverse images in lifelong few-shot image generation task. Our proposed framework learns each task using an efficient task-specific modulator - Learnable Factorized Tensor (LeFT). LeFT is rank-constrained and has a rich representation ability due to its unique reconstruction technique. Furthermore, we propose a novel mode seeking loss to improve the diversity of our model in low-data circumstances. Extensive experiments demonstrate that the proposed LFS-GAN can generate high-fidelity and diverse images without any forgetting and mode collapse in various domains, achieving state-of-the-art in lifelong few-shot image generation task. Surprisingly, we find that our LFS-GAN even outperforms the existing few-shot GANs in the few-shot image generation task. The code is available at Github.

{{</citation>}}


### (30/111) ACLS: Adaptive and Conditional Label Smoothing for Network Calibration (Hyekang Park et al., 2023)

{{<citation>}}

Hyekang Park, Jongyoun Noh, Youngmin Oh, Donghyeon Baek, Bumsub Ham. (2023)  
**ACLS: Adaptive and Conditional Label Smoothing for Network Calibration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.11911v2)  

---


**ABSTRACT**  
We address the problem of network calibration adjusting miscalibrated confidences of deep neural networks. Many approaches to network calibration adopt a regularization-based method that exploits a regularization term to smooth the miscalibrated confidences. Although these approaches have shown the effectiveness on calibrating the networks, there is still a lack of understanding on the underlying principles of regularization in terms of network calibration. We present in this paper an in-depth analysis of existing regularization-based methods, providing a better understanding on how they affect to network calibration. Specifically, we have observed that 1) the regularization-based methods can be interpreted as variants of label smoothing, and 2) they do not always behave desirably. Based on the analysis, we introduce a novel loss function, dubbed ACLS, that unifies the merits of existing regularization methods, while avoiding the limitations. We show extensive experimental results for image classification and semantic segmentation on standard benchmarks, including CIFAR10, Tiny-ImageNet, ImageNet, and PASCAL VOC, demonstrating the effectiveness of our loss function.

{{</citation>}}


### (31/111) Edge-aware Hard Clustering Graph Pooling for Brain Imaging Data (Cheng Zhu et al., 2023)

{{<citation>}}

Cheng Zhu, Jiayi Zhu, Lijuan Zhang, Xi Wu, Shuqi Yang, Ping Liang, Honghan Chen, Ying Tan. (2023)  
**Edge-aware Hard Clustering Graph Pooling for Brain Imaging Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2308.11909v2)  

---


**ABSTRACT**  
Graph Convolutional Networks (GCNs) can capture non-Euclidean spatial dependence between different brain regions, and the graph pooling operator in GCNs is key to enhancing the representation learning capability and acquiring abnormal brain maps. However, the majority of existing research designs graph pooling operators only from the perspective of nodes while disregarding the original edge features, in a way that not only confines graph pooling application scenarios, but also diminishes its ability to capture critical substructures. In this study, a clustering graph pooling method that first supports multidimensional edge features, called Edge-aware hard clustering graph pooling (EHCPool), is developed. EHCPool proposes the first 'Edge-to-node' score evaluation criterion based on edge features to assess node feature significance. To more effectively capture the critical subgraphs, a novel Iteration n-top strategy is further designed to adaptively learn sparse hard clustering assignments for graphs. Subsequently, an innovative N-E Aggregation strategy is presented to aggregate node and edge feature information in each independent subgraph. The proposed model was evaluated on multi-site brain imaging public datasets and yielded state-of-the-art performance. We believe this method is the first deep learning tool with the potential to probe different types of abnormal functional brain networks from data-driven perspective.

{{</citation>}}


### (32/111) Camera-Driven Representation Learning for Unsupervised Domain Adaptive Person Re-identification (Geon Lee et al., 2023)

{{<citation>}}

Geon Lee, Sanghoon Lee, Dohyung Kim, Younghoon Shin, Yongsang Yoon, Bumsub Ham. (2023)  
**Camera-Driven Representation Learning for Unsupervised Domain Adaptive Person Re-identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.11901v1)  

---


**ABSTRACT**  
We present a novel unsupervised domain adaption method for person re-identification (reID) that generalizes a model trained on a labeled source domain to an unlabeled target domain. We introduce a camera-driven curriculum learning (CaCL) framework that leverages camera labels of person images to transfer knowledge from source to target domains progressively. To this end, we divide target domain dataset into multiple subsets based on the camera labels, and initially train our model with a single subset (i.e., images captured by a single camera). We then gradually exploit more subsets for training, according to a curriculum sequence obtained with a camera-driven scheduling rule. The scheduler considers maximum mean discrepancies (MMD) between each subset and the source domain dataset, such that the subset closer to the source domain is exploited earlier within the curriculum. For each curriculum sequence, we generate pseudo labels of person images in a target domain to train a reID model in a supervised way. We have observed that the pseudo labels are highly biased toward cameras, suggesting that person images obtained from the same camera are likely to have the same pseudo labels, even for different IDs. To address the camera bias problem, we also introduce a camera-diversity (CD) loss encouraging person images of the same pseudo label, but captured across various cameras, to involve more for discriminative feature learning, providing person representations robust to inter-camera variations. Experimental results on standard benchmarks, including real-to-real and synthetic-to-real scenarios, demonstrate the effectiveness of our framework.

{{</citation>}}


### (33/111) Exploring the Optimization Objective of One-Class Classification for Anomaly Detection (Han Gao et al., 2023)

{{<citation>}}

Han Gao, Huiyuan Luo, Fei Shen, Zhengtao Zhang. (2023)  
**Exploring the Optimization Objective of One-Class Classification for Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.11898v1)  

---


**ABSTRACT**  
One-class classification (OCC) is a longstanding method for anomaly detection. With the powerful representation capability of the pre-trained backbone, OCC methods have witnessed significant performance improvements. Typically, most of these OCC methods employ transfer learning to enhance the discriminative nature of the pre-trained backbone's features, thus achieving remarkable efficacy. While most current approaches emphasize feature transfer strategies, we argue that the optimization objective space within OCC methods could also be an underlying critical factor influencing performance. In this work, we conducted a thorough investigation into the optimization objective of OCC. Through rigorous theoretical analysis and derivation, we unveil a key insights: any space with the suitable norm can serve as an equivalent substitute for the hypersphere center, without relying on the distribution assumption of training samples. Further, we provide guidelines for determining the feasible domain of norms for the OCC optimization objective. This novel insight sparks a simple and data-agnostic deep one-class classification method. Our method is straightforward, with a single 1x1 convolutional layer as a trainable projector and any space with suitable norm as the optimization objective. Extensive experiments validate the reliability and efficacy of our findings and the corresponding methodology, resulting in state-of-the-art performance in both one-class classification and industrial vision anomaly detection and segmentation tasks.

{{</citation>}}


### (34/111) Age Prediction From Face Images Via Contrastive Learning (Yeongnam Chae et al., 2023)

{{<citation>}}

Yeongnam Chae, Poulami Raha, Mijung Kim, Bjorn Stenger. (2023)  
**Age Prediction From Face Images Via Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.11896v1)  

---


**ABSTRACT**  
This paper presents a novel approach for accurately estimating age from face images, which overcomes the challenge of collecting a large dataset of individuals with the same identity at different ages. Instead, we leverage readily available face datasets of different people at different ages and aim to extract age-related features using contrastive learning. Our method emphasizes these relevant features while suppressing identity-related features using a combination of cosine similarity and triplet margin losses. We demonstrate the effectiveness of our proposed approach by achieving state-of-the-art performance on two public datasets, FG-NET and MORPH-II.

{{</citation>}}


### (35/111) A Unified Framework for 3D Point Cloud Visual Grounding (Haojia Lin et al., 2023)

{{<citation>}}

Haojia Lin, Yongdong Luo, Xiawu Zheng, Lijiang Li, Fei Chao, Taisong Jin, Donghao Luo, Chengjie Wang, Yan Wang, Liujuan Cao. (2023)  
**A Unified Framework for 3D Point Cloud Visual Grounding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.11887v1)  

---


**ABSTRACT**  
3D point cloud visual grounding plays a critical role in 3D scene comprehension, encompassing 3D referring expression comprehension (3DREC) and segmentation (3DRES). We argue that 3DREC and 3DRES should be unified in one framework, which is also a natural progression in the community. To explain, 3DREC can help 3DRES locate the referent, while 3DRES can also facilitate 3DREC via more finegrained language-visual alignment. To achieve this, this paper takes the initiative step to integrate 3DREC and 3DRES into a unified framework, termed 3D Referring Transformer (3DRefTR). Its key idea is to build upon a mature 3DREC model and leverage ready query embeddings and visual tokens from the 3DREC model to construct a dedicated mask branch. Specially, we propose Superpoint Mask Branch, which serves a dual purpose: i) By leveraging the heterogeneous CPU-GPU parallelism, while the GPU is occupied generating visual tokens, the CPU concurrently produces superpoints, equivalently accomplishing the upsampling computation; ii) By harnessing on the inherent association between the superpoints and point cloud, it eliminates the heavy computational overhead on the high-resolution visual features for upsampling. This elegant design enables 3DRefTR to achieve both well-performing 3DRES and 3DREC capacities with only a 6% additional latency compared to the original 3DREC model. Empirical evaluations affirm the superiority of 3DRefTR. Specifically, on the ScanRefer dataset, 3DRefTR surpasses the state-of-the-art 3DRES method by 12.43% in mIoU and improves upon the SOTA 3DREC method by 0.6% Acc@0.25IoU.

{{</citation>}}


### (36/111) Integrated Image and Location Analysis for Wound Classification: A Deep Learning Approach (Yash Patel et al., 2023)

{{<citation>}}

Yash Patel, Tirth Shah, Mrinal Kanti Dhar, Taiyu Zhang, Jeffrey Niezgoda, Sandeep Gopalakrishnan, Zeyun Yu. (2023)  
**Integrated Image and Location Analysis for Wound Classification: A Deep Learning Approach**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.11877v2)  

---


**ABSTRACT**  
The global burden of acute and chronic wounds presents a compelling case for enhancing wound classification methods, a vital step in diagnosing and determining optimal treatments. Recognizing this need, we introduce an innovative multi-modal network based on a deep convolutional neural network for categorizing wounds into four categories: diabetic, pressure, surgical, and venous ulcers. Our multi-modal network uses wound images and their corresponding body locations for more precise classification. A unique aspect of our methodology is incorporating a body map system that facilitates accurate wound location tagging, improving upon traditional wound image classification techniques. A distinctive feature of our approach is the integration of models such as VGG16, ResNet152, and EfficientNet within a novel architecture. This architecture includes elements like spatial and channel-wise Squeeze-and-Excitation modules, Axial Attention, and an Adaptive Gated Multi-Layer Perceptron, providing a robust foundation for classification. Our multi-modal network was trained and evaluated on two distinct datasets comprising relevant images and corresponding location information. Notably, our proposed network outperformed traditional methods, reaching an accuracy range of 74.79% to 100% for Region of Interest (ROI) without location classifications, 73.98% to 100% for ROI with location classifications, and 78.10% to 100% for whole image classifications. This marks a significant enhancement over previously reported performance metrics in the literature. Our results indicate the potential of our multi-modal network as an effective decision-support tool for wound image classification, paving the way for its application in various clinical contexts.

{{</citation>}}


### (37/111) Semi-Supervised Learning via Weight-aware Distillation under Class Distribution Mismatch (Pan Du et al., 2023)

{{<citation>}}

Pan Du, Suyun Zhao, Zisen Sheng, Cuiping Li, Hong Chen. (2023)  
**Semi-Supervised Learning via Weight-aware Distillation under Class Distribution Mismatch**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.11874v1)  

---


**ABSTRACT**  
Semi-Supervised Learning (SSL) under class distribution mismatch aims to tackle a challenging problem wherein unlabeled data contain lots of unknown categories unseen in the labeled ones. In such mismatch scenarios, traditional SSL suffers severe performance damage due to the harmful invasion of the instances with unknown categories into the target classifier. In this study, by strict mathematical reasoning, we reveal that the SSL error under class distribution mismatch is composed of pseudo-labeling error and invasion error, both of which jointly bound the SSL population risk. To alleviate the SSL error, we propose a robust SSL framework called Weight-Aware Distillation (WAD) that, by weights, selectively transfers knowledge beneficial to the target task from unsupervised contrastive representation to the target classifier. Specifically, WAD captures adaptive weights and high-quality pseudo labels to target instances by exploring point mutual information (PMI) in representation space to maximize the role of unlabeled data and filter unknown categories. Theoretically, we prove that WAD has a tight upper bound of population risk under class distribution mismatch. Experimentally, extensive results demonstrate that WAD outperforms five state-of-the-art SSL approaches and one standard baseline on two benchmark datasets, CIFAR10 and CIFAR100, and an artificial cross-dataset. The code is available at https://github.com/RUC-DWBI-ML/research/tree/main/WAD-master.

{{</citation>}}


### (38/111) CoC-GAN: Employing Context Cluster for Unveiling a New Pathway in Image Generation (Zihao Wang et al., 2023)

{{<citation>}}

Zihao Wang, Yiming Huang, Ziyu Zhou. (2023)  
**CoC-GAN: Employing Context Cluster for Unveiling a New Pathway in Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.11857v1)  

---


**ABSTRACT**  
Image generation tasks are traditionally undertaken using Convolutional Neural Networks (CNN) or Transformer architectures for feature aggregating and dispatching. Despite the frequent application of convolution and attention structures, these structures are not fundamentally required to solve the problem of instability and the lack of interpretability in image generation. In this paper, we propose a unique image generation process premised on the perspective of converting images into a set of point clouds. In other words, we interpret an image as a set of points. As such, our methodology leverages simple clustering methods named Context Clustering (CoC) to generate images from unordered point sets, which defies the convention of using convolution or attention mechanisms. Hence, we exclusively depend on this clustering technique, combined with the multi-layer perceptron (MLP) in a generative model. Furthermore, we implement the integration of a module termed the 'Point Increaser' for the model. This module is just an MLP tasked with generating additional points for clustering, which are subsequently integrated within the paradigm of the Generative Adversarial Network (GAN). We introduce this model with the novel structure as the Context Clustering Generative Adversarial Network (CoC-GAN), which offers a distinctive viewpoint in the domain of feature aggregating and dispatching. Empirical evaluations affirm that our CoC-GAN, devoid of convolution and attention mechanisms, exhibits outstanding performance. Its interpretability, endowed by the CoC module, also allows for visualization in our experiments. The promising results underscore the feasibility of our method and thus warrant future investigations of applying Context Clustering to more novel and interpretable image generation.

{{</citation>}}


### (39/111) Compressed Models Decompress Race Biases: What Quantized Models Forget for Fair Face Recognition (Pedro C. Neto et al., 2023)

{{<citation>}}

Pedro C. Neto, Eduarda Caldeira, Jaime S. Cardoso, Ana F. Sequeira. (2023)  
**Compressed Models Decompress Race Biases: What Quantized Models Forget for Fair Face Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.11840v1)  

---


**ABSTRACT**  
With the ever-growing complexity of deep learning models for face recognition, it becomes hard to deploy these systems in real life. Researchers have two options: 1) use smaller models; 2) compress their current models. Since the usage of smaller models might lead to concerning biases, compression gains relevance. However, compressing might be also responsible for an increase in the bias of the final model. We investigate the overall performance, the performance on each ethnicity subgroup and the racial bias of a State-of-the-Art quantization approach when used with synthetic and real data. This analysis provides a few more details on potential benefits of performing quantization with synthetic data, for instance, the reduction of biases on the majority of test scenarios. We tested five distinct architectures and three different training datasets. The models were evaluated on a fourth dataset which was collected to infer and compare the performance of face recognition models on different ethnicity.

{{</citation>}}


## cs.CL (18)



### (40/111) Are ChatGPT and GPT-4 Good Poker Players? -- A Pre-Flop Analysis (Akshat Gupta, 2023)

{{<citation>}}

Akshat Gupta. (2023)  
**Are ChatGPT and GPT-4 Good Poker Players? -- A Pre-Flop Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-GT, cs.CL  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.12466v1)  

---


**ABSTRACT**  
Since the introduction of ChatGPT and GPT-4, these models have been tested across a large number of tasks. Their adeptness across domains is evident, but their aptitude in playing games and specifically their aptitude in the realm of poker has remained unexplored. Poker is a game that requires decision making under uncertainty and incomplete information. In this paper, we put ChatGPT and GPT-4 through the poker test and evaluate their poker skills. Our findings reveal that while both models display an advanced understanding of poker, encompassing concepts like the valuation of starting hands, playing positions and other intricacies of game theory optimal (GTO) poker, both ChatGPT and GPT-4 are NOT game theory optimal poker players.   Through a series of experiments, we first discover the characteristics of optimal prompts and model parameters for playing poker with these models. Our observations then unveil the distinct playing personas of the two models. We first conclude that GPT-4 is a more advanced poker player than ChatGPT. This exploration then sheds light on the divergent poker tactics of the two models: ChatGPT's conservativeness juxtaposed against GPT-4's aggression. In poker vernacular, when tasked to play GTO poker, ChatGPT plays like a Nit, which means that it has a propensity to only engage with premium hands and folds a majority of hands. When subjected to the same directive, GPT-4 plays like a maniac, showcasing a loose and aggressive style of play. Both strategies, although relatively advanced, are not game theory optimal.

{{</citation>}}


### (41/111) D4: Improving LLM Pretraining via Document De-Duplication and Diversification (Kushal Tirumala et al., 2023)

{{<citation>}}

Kushal Tirumala, Daniel Simig, Armen Aghajanyan, Ari S. Morcos. (2023)  
**D4: Improving LLM Pretraining via Document De-Duplication and Diversification**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.12284v1)  

---


**ABSTRACT**  
Over recent years, an increasing amount of compute and data has been poured into training large language models (LLMs), usually by doing one-pass learning on as many tokens as possible randomly selected from large-scale web corpora. While training on ever-larger portions of the internet leads to consistent performance improvements, the size of these improvements diminishes with scale, and there has been little work exploring the effect of data selection on pre-training and downstream performance beyond simple de-duplication methods such as MinHash. Here, we show that careful data selection (on top of de-duplicated data) via pre-trained model embeddings can speed up training (20% efficiency gains) and improves average downstream accuracy on 16 NLP tasks (up to 2%) at the 6.7B model scale. Furthermore, we show that repeating data intelligently consistently outperforms baseline training (while repeating random data performs worse than baseline training). Our results indicate that clever data selection can significantly improve LLM pre-training, calls into question the common practice of training for a single epoch on as much data as possible, and demonstrates a path to keep improving our models past the limits of randomly sampling web data.

{{</citation>}}


### (42/111) Simple is Better and Large is Not Enough: Towards Ensembling of Foundational Language Models (Nancy Tyagi et al., 2023)

{{<citation>}}

Nancy Tyagi, Aidin Shiri, Surjodeep Sarkar, Abhishek Kumar Umrawal, Manas Gaur. (2023)  
**Simple is Better and Large is Not Enough: Towards Ensembling of Foundational Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model, NLP, T5  
[Paper Link](http://arxiv.org/abs/2308.12272v1)  

---


**ABSTRACT**  
Foundational Language Models (FLMs) have advanced natural language processing (NLP) research. Current researchers are developing larger FLMs (e.g., XLNet, T5) to enable contextualized language representation, classification, and generation. While developing larger FLMs has been of significant advantage, it is also a liability concerning hallucination and predictive uncertainty. Fundamentally, larger FLMs are built on the same foundations as smaller FLMs (e.g., BERT); hence, one must recognize the potential of smaller FLMs which can be realized through an ensemble. In the current research, we perform a reality check on FLMs and their ensemble on benchmark and real-world datasets. We hypothesize that the ensembling of FLMs can influence the individualistic attention of FLMs and unravel the strength of coordination and cooperation of different FLMs. We utilize BERT and define three other ensemble techniques: {Shallow, Semi, and Deep}, wherein the Deep-Ensemble introduces a knowledge-guided reinforcement learning approach. We discovered that the suggested Deep-Ensemble BERT outperforms its large variation i.e. BERTlarge, by a factor of many times using datasets that show the usefulness of NLP in sensitive fields, such as mental health.

{{</citation>}}


### (43/111) Prompt2Model: Generating Deployable Models from Natural Language Instructions (Vijay Viswanathan et al., 2023)

{{<citation>}}

Vijay Viswanathan, Chenyang Zhao, Amanda Bertsch, Tongshuang Wu, Graham Neubig. (2023)  
**Prompt2Model: Generating Deployable Models from Natural Language Instructions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.12261v1)  

---


**ABSTRACT**  
Large language models (LLMs) enable system builders today to create competent NLP systems through prompting, where they only need to describe the task in natural language and provide a few examples. However, in other ways, LLMs are a step backward from traditional special-purpose NLP models; they require extensive computational resources for deployment and can be gated behind APIs. In this paper, we propose Prompt2Model, a general-purpose method that takes a natural language task description like the prompts provided to LLMs, and uses it to train a special-purpose model that is conducive to deployment. This is done through a multi-step process of retrieval of existing datasets and pretrained models, dataset generation using LLMs, and supervised fine-tuning on these retrieved and generated datasets. Over three tasks, we demonstrate that given the same few-shot prompt as input, Prompt2Model trains models that outperform the results of a strong LLM, gpt-3.5-turbo, by an average of 20% while being up to 700 times smaller. We also show that this data can be used to obtain reliable performance estimates of model performance, enabling model developers to assess model reliability before deployment. Prompt2Model is available open-source at https://github.com/neulab/prompt2model.

{{</citation>}}


### (44/111) Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning (Jiasheng Ye et al., 2023)

{{<citation>}}

Jiasheng Ye, Zaixiang Zheng, Yu Bao, Lihua Qian, Quanquan Gu. (2023)  
**Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.12219v1)  

---


**ABSTRACT**  
The recent surge of generative AI has been fueled by the generative power of diffusion probabilistic models and the scalable capabilities of large language models. Despite their potential, it remains elusive whether diffusion language models can solve general language tasks comparable to their autoregressive counterparts. This paper demonstrates that scaling diffusion models w.r.t. data, sizes, and tasks can effectively make them strong language learners. We build competent diffusion language models at scale by first acquiring knowledge from massive data via masked language modeling pretraining thanks to their intrinsic connections. We then reprogram pretrained masked language models into diffusion language models via diffusive adaptation, wherein task-specific finetuning and instruction finetuning are explored to unlock their versatility in solving general language tasks. Experiments show that scaling diffusion language models consistently improves performance across downstream language tasks. We further discover that instruction finetuning can elicit zero-shot and few-shot in-context learning abilities that help tackle many unseen tasks by following natural language instructions, and show promise in advanced and challenging abilities such as reasoning

{{</citation>}}


### (45/111) Evaluation of Faithfulness Using the Longest Supported Subsequence (Anirudh Mittal et al., 2023)

{{<citation>}}

Anirudh Mittal, Timo Schick, Mikel Artetxe, Jane Dwivedi-Yu. (2023)  
**Evaluation of Faithfulness Using the Longest Supported Subsequence**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.12157v1)  

---


**ABSTRACT**  
As increasingly sophisticated language models emerge, their trustworthiness becomes a pivotal issue, especially in tasks such as summarization and question-answering. Ensuring their responses are contextually grounded and faithful is challenging due to the linguistic diversity and the myriad of possible answers. In this paper, we introduce a novel approach to evaluate faithfulness of machine-generated text by computing the longest noncontinuous substring of the claim that is supported by the context, which we refer to as the Longest Supported Subsequence (LSS). Using a new human-annotated dataset, we finetune a model to generate LSS. We introduce a new method of evaluation and demonstrate that these metrics correlate better with human ratings when LSS is employed, as opposed to when it is not. Our proposed metric demonstrates an 18% enhancement over the prevailing state-of-the-art metric for faithfulness on our dataset. Our metric consistently outperforms other metrics on a summarization dataset across six different models. Finally, we compare several popular Large Language Models (LLMs) for faithfulness using this metric. We release the human-annotated dataset built for predicting LSS and our fine-tuned model for evaluating faithfulness.

{{</citation>}}


### (46/111) Instruction Position Matters in Sequence Generation with Large Language Models (Yijin Liu et al., 2023)

{{<citation>}}

Yijin Liu, Xianfeng Zeng, Fandong Meng, Jie Zhou. (2023)  
**Instruction Position Matters in Sequence Generation with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Language Model  
[Paper Link](http://arxiv.org/abs/2308.12097v1)  

---


**ABSTRACT**  
Large language models (LLMs) are capable of performing conditional sequence generation tasks, such as translation or summarization, through instruction fine-tuning. The fine-tuning data is generally sequentially concatenated from a specific task instruction, an input sentence, and the corresponding response. Considering the locality modeled by the self-attention mechanism of LLMs, these models face the risk of instruction forgetting when generating responses for long input sentences. To mitigate this issue, we propose enhancing the instruction-following capability of LLMs by shifting the position of task instructions after the input sentences. Theoretical analysis suggests that our straightforward method can alter the model's learning focus, thereby emphasizing the training of instruction-following capabilities. Concurrently, experimental results demonstrate that our approach consistently outperforms traditional settings across various model scales (1B / 7B / 13B) and different sequence generation tasks (translation and summarization), without any additional data or annotation costs. Notably, our method significantly improves the zero-shot performance on conditional sequence generation, e.g., up to 9.7 BLEU points on WMT zero-shot translation tasks.

{{</citation>}}


### (47/111) FlexKBQA: A Flexible LLM-Powered Framework for Few-Shot Knowledge Base Question Answering (Zhenyu Li et al., 2023)

{{<citation>}}

Zhenyu Li, Sunqi Fan, Yu Gu, Xiuxing Li, Zhichao Duan, Bowen Dong, Ning Liu, Jianyong Wang. (2023)  
**FlexKBQA: A Flexible LLM-Powered Framework for Few-Shot Knowledge Base Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Few-Shot, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.12060v1)  

---


**ABSTRACT**  
Knowledge base question answering (KBQA) is a critical yet challenging task due to the vast number of entities within knowledge bases and the diversity of natural language questions posed by users. Unfortunately, the performance of most KBQA models tends to decline significantly in real-world scenarios where high-quality annotated data is insufficient. To mitigate the burden associated with manual annotation, we introduce FlexKBQA by utilizing Large Language Models (LLMs) as program translators for addressing the challenges inherent in the few-shot KBQA task. Specifically, FlexKBQA leverages automated algorithms to sample diverse programs, such as SPARQL queries, from the knowledge base, which are subsequently converted into natural language questions via LLMs. This synthetic dataset facilitates training a specialized lightweight model for the KB. Additionally, to reduce the barriers of distribution shift between synthetic data and real user questions, FlexKBQA introduces an executionguided self-training method to iterative leverage unlabeled user questions. Furthermore, we explore harnessing the inherent reasoning capability of LLMs to enhance the entire framework. Consequently, FlexKBQA delivers substantial flexibility, encompassing data annotation, deployment, and being domain agnostic. Through extensive experiments on GrailQA, WebQSP, and KQA Pro, we observe that under the few-shot even the more challenging zero-shot scenarios, FlexKBQA achieves impressive results with a few annotations, surpassing all previous baselines and even approaching the performance of supervised models, achieving a remarkable 93% performance relative to the fully-supervised models. We posit that FlexKBQA represents a significant advancement towards exploring better integration of large and lightweight models. The code is open-sourced.

{{</citation>}}


### (48/111) Aligning Language Models with Offline Reinforcement Learning from Human Feedback (Jian Hu et al., 2023)

{{<citation>}}

Jian Hu, Li Tao, June Yang, Chandler Zhou. (2023)  
**Aligning Language Models with Offline Reinforcement Learning from Human Feedback**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2308.12050v1)  

---


**ABSTRACT**  
Learning from human preferences is crucial for language models (LMs) to effectively cater to human needs and societal values. Previous research has made notable progress by leveraging human feedback to follow instructions. However, these approaches rely primarily on online reinforcement learning (RL) techniques like Proximal Policy Optimization (PPO), which have been proven unstable and challenging to tune for language models. Moreover, PPO requires complex distributed system implementation, hindering the efficiency of large-scale distributed training. In this study, we propose an offline reinforcement learning from human feedback (RLHF) framework to align LMs using pre-generated samples without interacting with RL environments. Specifically, we explore maximum likelihood estimation (MLE) with filtering, reward-weighted regression (RWR), and Decision Transformer (DT) to align language models to human preferences. By employing a loss function similar to supervised fine-tuning, our methods ensure more stable model training than PPO with a simple machine learning system~(MLSys) and much fewer (around 12.3\%) computing resources. Experimental results demonstrate the DT alignment outperforms other Offline RLHF methods and is better than PPO.

{{</citation>}}


### (49/111) IncreLoRA: Incremental Parameter Allocation Method for Parameter-Efficient Fine-tuning (Feiyu Zhang et al., 2023)

{{<citation>}}

Feiyu Zhang, Liangzhi Li, Junhao Chen, Zhouqiang Jiang, Bowen Wang, Yiming Qian. (2023)  
**IncreLoRA: Incremental Parameter Allocation Method for Parameter-Efficient Fine-tuning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GLUE  
[Paper Link](http://arxiv.org/abs/2308.12043v1)  

---


**ABSTRACT**  
With the increasing size of pre-trained language models (PLMs), fine-tuning all the parameters in the model is not efficient, especially when there are a large number of downstream tasks, which incur significant training and storage costs. Many parameter-efficient fine-tuning (PEFT) approaches have been proposed, among which, Low-Rank Adaptation (LoRA) is a representative approach that injects trainable rank decomposition matrices into every target module. Yet LoRA ignores the importance of parameters in different modules. To address this problem, many works have been proposed to prune the parameters of LoRA. However, under limited training conditions, the upper bound of the rank of the pruned parameter matrix is still affected by the preset values. We, therefore, propose IncreLoRA, an incremental parameter allocation method that adaptively adds trainable parameters during training based on the importance scores of each module. This approach is different from the pruning method as it is not limited by the initial number of training parameters, and each parameter matrix has a higher rank upper bound for the same training overhead. We conduct extensive experiments on GLUE to demonstrate the effectiveness of IncreLoRA. The results show that our method owns higher parameter efficiency, especially when under the low-resource settings where our method significantly outperforms the baselines. Our code is publicly available.

{{</citation>}}


### (50/111) Large Multilingual Models Pivot Zero-Shot Multimodal Learning across Languages (Jinyi Hu et al., 2023)

{{<citation>}}

Jinyi Hu, Yuan Yao, Chongyi Wang, Shan Wang, Yinxu Pan, Qianyu Chen, Tianyu Yu, Hanghao Wu, Yue Zhao, Haoye Zhang, Xu Han, Yankai Lin, Jiao Xue, Dahai Li, Zhiyuan Liu, Maosong Sun. (2023)  
**Large Multilingual Models Pivot Zero-Shot Multimodal Learning across Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Multilingual, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.12038v1)  

---


**ABSTRACT**  
Recently there has been a significant surge in multimodal learning in terms of both image-to-text and text-to-image generation. However, the success is typically limited to English, leaving other languages largely behind. Building a competitive counterpart in other languages is highly challenging due to the low-resource nature of non-English multimodal data (i.e., lack of large-scale, high-quality image-text data). In this work, we propose MPM, an effective training paradigm for training large multimodal models in low-resource languages. MPM demonstrates that Multilingual language models can Pivot zero-shot Multimodal learning across languages. Specifically, based on a strong multilingual large language model, multimodal models pretrained on English-only image-text data can well generalize to other languages in a zero-shot manner for both image-to-text and text-to-image generation, even surpassing models trained on image-text data in native languages. Taking Chinese as a practice of MPM, we build large multimodal models VisCPM in image-to-text and text-to-image generation, which achieve state-of-the-art (open-source) performance in Chinese. To facilitate future research, we open-source codes and model weights at https://github.com/OpenBMB/VisCPM.git.

{{</citation>}}


### (51/111) PREFER: Prompt Ensemble Learning via Feedback-Reflect-Refine (Chenrui Zhang et al., 2023)

{{<citation>}}

Chenrui Zhang, Lin Liu, Jinpeng Wang, Chuyuan Wang, Xiao Sun, Hongyu Wang, Mingchen Cai. (2023)  
**PREFER: Prompt Ensemble Learning via Feedback-Reflect-Refine**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.12033v1)  

---


**ABSTRACT**  
As an effective tool for eliciting the power of Large Language Models (LLMs), prompting has recently demonstrated unprecedented abilities across a variety of complex tasks. To further improve the performance, prompt ensemble has attracted substantial interest for tackling the hallucination and instability of LLMs. However, existing methods usually adopt a two-stage paradigm, which requires a pre-prepared set of prompts with substantial manual effort, and is unable to perform directed optimization for different weak learners. In this paper, we propose a simple, universal, and automatic method named PREFER (Pompt Ensemble learning via Feedback-Reflect-Refine) to address the stated limitations. Specifically, given the fact that weak learners are supposed to focus on hard examples during boosting, PREFER builds a feedback mechanism for reflecting on the inadequacies of existing weak learners. Based on this, the LLM is required to automatically synthesize new prompts for iterative refinement. Moreover, to enhance stability of the prompt effect evaluation, we propose a novel prompt bagging method involving forward and backward thinking, which is superior to majority voting and is beneficial for both feedback and weight calculation in boosting. Extensive experiments demonstrate that our PREFER achieves state-of-the-art performance in multiple types of tasks by a significant margin. We have made our code publicly available.

{{</citation>}}


### (52/111) From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning (Ming Li et al., 2023)

{{<citation>}}

Ming Li, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang Chen, Ning Cheng, Jianzong Wang, Tianyi Zhou, Jing Xiao. (2023)  
**From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.12032v1)  

---


**ABSTRACT**  
In the realm of Large Language Models, the balance between instruction data quality and quantity has become a focal point. Recognizing this, we introduce a self-guided methodology for LLMs to autonomously discern and select cherry samples from vast open-source datasets, effectively minimizing manual curation and potential cost for instruction tuning an LLM. Our key innovation, the Instruction-Following Difficulty (IFD) metric, emerges as a pivotal tool to identify discrepancies between a model's expected responses and its autonomous generation prowess. Through the adept application of IFD, cherry samples are pinpointed, leading to a marked uptick in model training efficiency. Empirical validations on renowned datasets like Alpaca and WizardLM underpin our findings; with a mere 10% of conventional data input, our strategy showcases improved results. This synthesis of self-guided cherry-picking and the IFD metric signifies a transformative leap in the optimization of LLMs, promising both efficiency and resource-conscious advancements.

{{</citation>}}


### (53/111) Prompt-Based Length Controlled Generation with Reinforcement Learning (Renlong Jie et al., 2023)

{{<citation>}}

Renlong Jie, Xiaojun Meng, Lifeng Shang, Xin Jiang, Qun Liu. (2023)  
**Prompt-Based Length Controlled Generation with Reinforcement Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.12030v1)  

---


**ABSTRACT**  
Recently, large language models (LLMs) like ChatGPT and GPT-4 have attracted great attention given their surprising improvement and performance. Length controlled generation of LLMs emerges as an important topic, which also enables users to fully leverage the capability of LLMs in more real-world scenarios like generating a proper answer or essay of a desired length. In addition, the autoregressive generation in LLMs is extremely time-consuming, while the ability of controlling this generated length can arbitrarily reduce the inference cost by limiting the length, and thus satisfy different needs. Therefore, we aim to propose a prompt-based length control method to achieve this length controlled generation, which can also be widely applied in GPT-style LLMs. In particular, we adopt reinforcement learning with the reward signal given by either trainable or rule-based reward model, which further affects the generation of LLMs via rewarding a pre-defined target length. Experiments show that our method significantly improves the accuracy of prompt-based length control for summarization task on popular datasets like CNNDM and NYT. We believe this length-controllable ability can provide more potentials towards the era of LLMs.

{{</citation>}}


### (54/111) Graecia capta ferum victorem cepit. Detecting Latin Allusions to Ancient Greek Literature (Frederick Riemenschneider et al., 2023)

{{<citation>}}

Frederick Riemenschneider, Anette Frank. (2023)  
**Graecia capta ferum victorem cepit. Detecting Latin Allusions to Ancient Greek Literature**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: BERT, NLP  
[Paper Link](http://arxiv.org/abs/2308.12008v1)  

---


**ABSTRACT**  
Intertextual allusions hold a pivotal role in Classical Philology, with Latin authors frequently referencing Ancient Greek texts. Until now, the automatic identification of these intertextual references has been constrained to monolingual approaches, seeking parallels solely within Latin or Greek texts. In this study, we introduce SPhilBERTa, a trilingual Sentence-RoBERTa model tailored for Classical Philology, which excels at cross-lingual semantic comprehension and identification of identical sentences across Ancient Greek, Latin, and English. We generate new training data by automatically translating English texts into Ancient Greek. Further, we present a case study, demonstrating SPhilBERTa's capability to facilitate automated detection of intertextual parallels. Our models and resources are available at https://github.com/Heidelberg-NLP/ancient-language-models.

{{</citation>}}


### (55/111) Topical-Chat: Towards Knowledge-Grounded Open-Domain Conversations (Karthik Gopalakrishnan et al., 2023)

{{<citation>}}

Karthik Gopalakrishnan, Behnam Hedayatnia, Qinlang Chen, Anna Gottardi, Sanjeev Kwatra, Anu Venkatesh, Raefer Gabriel, Dilek Hakkani-Tur. (2023)  
**Topical-Chat: Towards Knowledge-Grounded Open-Domain Conversations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.11995v1)  

---


**ABSTRACT**  
Building socialbots that can have deep, engaging open-domain conversations with humans is one of the grand challenges of artificial intelligence (AI). To this end, bots need to be able to leverage world knowledge spanning several domains effectively when conversing with humans who have their own world knowledge. Existing knowledge-grounded conversation datasets are primarily stylized with explicit roles for conversation partners. These datasets also do not explore depth or breadth of topical coverage with transitions in conversations. We introduce Topical-Chat, a knowledge-grounded human-human conversation dataset where the underlying knowledge spans 8 broad topics and conversation partners don't have explicitly defined roles, to help further research in open-domain conversational AI. We also train several state-of-the-art encoder-decoder conversational models on Topical-Chat and perform automated and human evaluation for benchmarking.

{{</citation>}}


### (56/111) Bridging the Gap: Deciphering Tabular Data Using Large Language Model (Hengyuan Zhang et al., 2023)

{{<citation>}}

Hengyuan Zhang, Peng Chang, Zongcheng Ji. (2023)  
**Bridging the Gap: Deciphering Tabular Data Using Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.11891v1)  

---


**ABSTRACT**  
In the realm of natural language processing, the understanding of tabular data has perpetually stood as a focal point of scholarly inquiry. The emergence of expansive language models, exemplified by the likes of ChatGPT, has ushered in a wave of endeavors wherein researchers aim to harness these models for tasks related to table-based question answering. Central to our investigative pursuits is the elucidation of methodologies that amplify the aptitude of such large language models in discerning both the structural intricacies and inherent content of tables, ultimately facilitating their capacity to provide informed responses to pertinent queries. To this end, we have architected a distinctive module dedicated to the serialization of tables for seamless integration with expansive language models. Additionally, we've instituted a corrective mechanism within the model to rectify potential inaccuracies. Experimental results indicate that, although our proposed method trails the SOTA by approximately 11.7% in overall metrics, it surpasses the SOTA by about 1.2% in tests on specific datasets. This research marks the first application of large language models to table-based question answering tasks, enhancing the model's comprehension of both table structures and content.

{{</citation>}}


### (57/111) Cabrita: closing the gap for foreign languages (Celio Larcher et al., 2023)

{{<citation>}}

Celio Larcher, Marcos Piau, Paulo Finardi, Pedro Gengo, Piero Esposito, Vinicius Caridá. (2023)  
**Cabrita: closing the gap for foreign languages**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2308.11878v1)  

---


**ABSTRACT**  
The strategy of training the model from scratch in a specific language or domain serves two essential purposes: i) enhancing performance in the particular linguistic or domain context, and ii) ensuring effective tokenization. The main limitation inherent to this approach lies in the associated cost, which can reach six to seven-digit dollar values, depending on the model size and the number of parameters involved.   The main solution to overcome the cost challenge is to rely on available pre-trained models, which, despite recent advancements such as the LLaMA and LLaMA-2 models, still demonstrate inefficiency for certain specific domain problems or prove ineffective in scenarios involving conversational memory resources, given the large number of tokens required to represent text.   To overcome this issue, we present a methodology named Cabrita, which, as our research demonstrates, successfully addresses the performance and efficient tokenization problem, all at an affordable cost. We believe that this methodology can be applied to any transformer-like architecture model. To validate the study, we conducted continuous pre-training exclusively using Portuguese text on a 3-billion-parameter model known as OpenLLaMA, resulting in a model named openCabrita 3B. The openCabrita 3B also features a new tokenizer that results in a significant reduction in the number of tokens required to represent the text. In our assessment, for few-shot learning tasks, we achieved similar results with this 3B model compared to a traditional continuous pre-training approach as well as to 7B models English pre-trained models.

{{</citation>}}


## eess.SP (1)



### (58/111) Zero-delay Consistent Signal Reconstruction from Streamed Multivariate Time Series (Emilio Ruiz-Moreno et al., 2023)

{{<citation>}}

Emilio Ruiz-Moreno, Luis Miguel López-Ramos, Baltasar Beferull-Lozano. (2023)  
**Zero-delay Consistent Signal Reconstruction from Streamed Multivariate Time Series**  

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.12459v1)  

---


**ABSTRACT**  
Digitalizing real-world analog signals typically involves sampling in time and discretizing in amplitude. Subsequent signal reconstructions inevitably incur an error that depends on the amplitude resolution and the temporal density of the acquired samples. From an implementation viewpoint, consistent signal reconstruction methods have proven a profitable error-rate decay as the sampling rate increases. Despite that, these results are obtained under offline settings. Therefore, a research gap exists regarding methods for consistent signal reconstruction from data streams. This paper presents a method that consistently reconstructs streamed multivariate time series of quantization intervals under a zero-delay response requirement. On the other hand, previous work has shown that the temporal dependencies within univariate time series can be exploited to reduce the roughness of zero-delay signal reconstructions. This work shows that the spatiotemporal dependencies within multivariate time series can also be exploited to achieve improved results. Specifically, the spatiotemporal dependencies of the multivariate time series are learned, with the assistance of a recurrent neural network, to reduce the roughness of the signal reconstruction on average while ensuring consistency. Our experiments show that our proposed method achieves a favorable error-rate decay with the sampling rate compared to a similar but non-consistent reconstruction.

{{</citation>}}


## cs.CR (6)



### (59/111) Trend and Emerging Types of 419 Scams (Polra Victor Falade, 2023)

{{<citation>}}

Polra Victor Falade. (2023)  
**Trend and Emerging Types of 419 Scams**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.12448v1)  

---


**ABSTRACT**  
Technological advancements have revolutionized various aspects of human life, facilitating communication, business operations, healthcare, education, and environmental monitoring. However, this increased reliance on technology has also led to a surge in cybercrime, including cyber scams. The "419 scam" or Nigerian scam has been a persistent problem for decades, encompassing frauds like advance fee scams, fake lotteries, and black money scams. Initially prevalent through postal mail and later via fax, the scam has now transitioned to email. This study aims to identify recent types of 419 scam emails, particularly after the covid 19 pandemic, and explore commonly used email subjects. Analysis of the sample 419 scam emails revealed trending scams like lucky winner, threat of exposure, business/partnership proposals, investment, cancer/long-term illness, fund, and compensation scams. Emerging scams included COVID-related, cryptocurrency, marketing contact, and software development scams. Irrespective of the scam type, scammers commonly employed email subjects such as 'Re', 'Good day', 'Greetings', 'Dear friend', 'Confirm', 'Attention', and 'Hello dear'. The severity of cybercrime, especially the 419 scams, cannot be overstated, as it erodes trust, causes financial losses, and hampers Nigeria's reputation and economic progress. Combatting cyber scams and enhancing cybersecurity measures are crucial to protect individuals and organizations from falling victim to these fraudulent schemes.

{{</citation>}}


### (60/111) BaDExpert: Extracting Backdoor Functionality for Accurate Backdoor Input Detection (Tinghao Xie et al., 2023)

{{<citation>}}

Tinghao Xie, Xiangyu Qi, Ping He, Yiming Li, Jiachen T. Wang, Prateek Mittal. (2023)  
**BaDExpert: Extracting Backdoor Functionality for Accurate Backdoor Input Detection**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-CV, cs-LG, cs.CR  
Keywords: ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2308.12439v1)  

---


**ABSTRACT**  
We present a novel defense, against backdoor attacks on Deep Neural Networks (DNNs), wherein adversaries covertly implant malicious behaviors (backdoors) into DNNs. Our defense falls within the category of post-development defenses that operate independently of how the model was generated. The proposed defense is built upon a novel reverse engineering approach that can directly extract backdoor functionality of a given backdoored model to a backdoor expert model. The approach is straightforward -- finetuning the backdoored model over a small set of intentionally mislabeled clean samples, such that it unlearns the normal functionality while still preserving the backdoor functionality, and thus resulting in a model (dubbed a backdoor expert model) that can only recognize backdoor inputs. Based on the extracted backdoor expert model, we show the feasibility of devising highly accurate backdoor input detectors that filter out the backdoor inputs during model inference. Further augmented by an ensemble strategy with a finetuned auxiliary model, our defense, BaDExpert (Backdoor Input Detection with Backdoor Expert), effectively mitigates 16 SOTA backdoor attacks while minimally impacting clean utility. The effectiveness of BaDExpert has been verified on multiple datasets (CIFAR10, GTSRB and ImageNet) across various model architectures (ResNet, VGG, MobileNetV2 and Vision Transformer).

{{</citation>}}


### (61/111) Devising and Detecting Phishing: large language models vs. Smaller Human Models (Fredrik Heiding et al., 2023)

{{<citation>}}

Fredrik Heiding, Bruce Schneier, Arun Vishwanath, Jeremy Bernstein. (2023)  
**Devising and Detecting Phishing: large language models vs. Smaller Human Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI, GPT, GPT-4, LLaMA, PaLM  
[Paper Link](http://arxiv.org/abs/2308.12287v1)  

---


**ABSTRACT**  
AI programs, built using large language models, make it possible to automatically create phishing emails based on a few data points about a user. They stand in contrast to traditional phishing emails that hackers manually design using general rules gleaned from experience. The V-Triad is an advanced set of rules for manually designing phishing emails to exploit our cognitive heuristics and biases. In this study, we compare the performance of phishing emails created automatically by GPT-4 and manually using the V-Triad. We also combine GPT-4 with the V-Triad to assess their combined potential. A fourth group, exposed to generic phishing emails, was our control group. We utilized a factorial approach, sending emails to 112 randomly selected participants recruited for the study. The control group emails received a click-through rate between 19-28%, the GPT-generated emails 30-44%, emails generated by the V-Triad 69-79%, and emails generated by GPT and the V-Triad 43-81%. Each participant was asked to explain for why they pressed or did not press a link in the email. These answers often contradict each other, highlighting the need for personalized content. The cues that make one person avoid phishing emails make another person fall for them. Next, we used four popular large language models (GPT, Claude, PaLM, and LLaMA) to detect the intention of phishing emails and compare the results to human detection. The language models demonstrated a strong ability to detect malicious intent, even in non-obvious phishing emails. They sometimes surpassed human detection, although often being slightly less accurate than humans.

{{</citation>}}


### (62/111) Out of the Cage: How Stochastic Parrots Win in Cyber Security Environments (Maria Rigaki et al., 2023)

{{<citation>}}

Maria Rigaki, Ondřej Lukáš, Carlos A. Catania, Sebastian Garcia. (2023)  
**Out of the Cage: How Stochastic Parrots Win in Cyber Security Environments**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs.CR  
Keywords: Cyber Security, Language Model, Security  
[Paper Link](http://arxiv.org/abs/2308.12086v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have gained widespread popularity across diverse domains involving text generation, summarization, and various natural language processing tasks. Despite their inherent limitations, LLM-based designs have shown promising capabilities in planning and navigating open-world scenarios. This paper introduces a novel application of pre-trained LLMs as agents within cybersecurity network environments, focusing on their utility for sequential decision-making processes.   We present an approach wherein pre-trained LLMs are leveraged as attacking agents in two reinforcement learning environments. Our proposed agents demonstrate similar or better performance against state-of-the-art agents trained for thousands of episodes in most scenarios and configurations. In addition, the best LLM agents perform similarly to human testers of the environment without any additional training process. This design highlights the potential of LLMs to efficiently address complex decision-making tasks within cybersecurity.   Furthermore, we introduce a new network security environment named NetSecGame. The environment is designed to eventually support complex multi-agent scenarios within the network security domain. The proposed environment mimics real network attacks and is designed to be highly modular and adaptable for various scenarios.

{{</citation>}}


### (63/111) Unleashing IoT Security: Assessing the Effectiveness of Best Practices in Protecting Against Threats (Philipp Pütz et al., 2023)

{{<citation>}}

Philipp Pütz, Richard Mitev, Markus Miettinen, Ahmad-Reza Sadeghi. (2023)  
**Unleashing IoT Security: Assessing the Effectiveness of Best Practices in Protecting Against Threats**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.12072v1)  

---


**ABSTRACT**  
The Internet of Things (IoT) market is rapidly growing and is expected to double from 2020 to 2025. The increasing use of IoT devices, particularly in smart homes, raises crucial concerns about user privacy and security as these devices often handle sensitive and critical information. Inadequate security designs and implementations by IoT vendors can lead to significant vulnerabilities.   To address these IoT device vulnerabilities, institutions, and organizations have published IoT security best practices (BPs) to guide manufacturers in ensuring the security of their products. However, there is currently no standardized approach for evaluating the effectiveness of individual BP recommendations. This leads to manufacturers investing effort in implementing less effective BPs while potentially neglecting measures with greater impact.   In this paper, we propose a methodology for evaluating the security impact of IoT BPs and ranking them based on their effectiveness in protecting against security threats. Our approach involves translating identified BPs into concrete test cases that can be applied to real-world IoT devices to assess their effectiveness in mitigating vulnerabilities. We applied this methodology to evaluate the security impact of nine commodity IoT products, discovering 18 vulnerabilities. By empirically assessing the actual impact of BPs on device security, IoT designers and implementers can prioritize their security investments more effectively, improving security outcomes and optimizing limited security budgets.

{{</citation>}}


### (64/111) Does Physical Adversarial Example Really Matter to Autonomous Driving? Towards System-Level Effect of Adversarial Object Evasion Attack (Ningfei Wang et al., 2023)

{{<citation>}}

Ningfei Wang, Yunpeng Luo, Takami Sato, Kaidi Xu, Qi Alfred Chen. (2023)  
**Does Physical Adversarial Example Really Matter to Autonomous Driving? Towards System-Level Effect of Adversarial Object Evasion Attack**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.11894v1)  

---


**ABSTRACT**  
In autonomous driving (AD), accurate perception is indispensable to achieving safe and secure driving. Due to its safety-criticality, the security of AD perception has been widely studied. Among different attacks on AD perception, the physical adversarial object evasion attacks are especially severe. However, we find that all existing literature only evaluates their attack effect at the targeted AI component level but not at the system level, i.e., with the entire system semantics and context such as the full AD pipeline. Thereby, this raises a critical research question: can these existing researches effectively achieve system-level attack effects (e.g., traffic rule violations) in the real-world AD context? In this work, we conduct the first measurement study on whether and how effectively the existing designs can lead to system-level effects, especially for the STOP sign-evasion attacks due to their popularity and severity. Our evaluation results show that all the representative prior works cannot achieve any system-level effects. We observe two design limitations in the prior works: 1) physical model-inconsistent object size distribution in pixel sampling and 2) lack of vehicle plant model and AD system model consideration. Then, we propose SysAdv, a novel system-driven attack design in the AD context and our evaluation results show that the system-level effects can be significantly improved, i.e., the violation rate increases by around 70%.

{{</citation>}}


## cs.LG (16)



### (65/111) An Intentional Forgetting-Driven Self-Healing Method For Deep Reinforcement Learning Systems (Ahmed Haj Yahmed et al., 2023)

{{<citation>}}

Ahmed Haj Yahmed, Rached Bouchoucha, Houssem Ben Braiek, Foutse Khomh. (2023)  
**An Intentional Forgetting-Driven Self-Healing Method For Deep Reinforcement Learning Systems**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SE, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.12445v1)  

---


**ABSTRACT**  
Deep reinforcement learning (DRL) is increasingly applied in large-scale productions like Netflix and Facebook. As with most data-driven systems, DRL systems can exhibit undesirable behaviors due to environmental drifts, which often occur in constantly-changing production settings. Continual Learning (CL) is the inherent self-healing approach for adapting the DRL agent in response to the environment's conditions shifts. However, successive shifts of considerable magnitude may cause the production environment to drift from its original state. Recent studies have shown that these environmental drifts tend to drive CL into long, or even unsuccessful, healing cycles, which arise from inefficiencies such as catastrophic forgetting, warm-starting failure, and slow convergence. In this paper, we propose Dr. DRL, an effective self-healing approach for DRL systems that integrates a novel mechanism of intentional forgetting into vanilla CL to overcome its main issues. Dr. DRL deliberately erases the DRL system's minor behaviors to systematically prioritize the adaptation of the key problem-solving skills. Using well-established DRL algorithms, Dr. DRL is compared with vanilla CL on various drifted environments. Dr. DRL is able to reduce, on average, the healing time and fine-tuning episodes by, respectively, 18.74% and 17.72%. Dr. DRL successfully helps agents to adapt to 19.63% of drifted environments left unsolved by vanilla CL while maintaining and even enhancing by up to 45% the obtained rewards for drifted environments that are resolved by both approaches.

{{</citation>}}


### (66/111) Deploying Deep Reinforcement Learning Systems: A Taxonomy of Challenges (Ahmed Haj Yahmed et al., 2023)

{{<citation>}}

Ahmed Haj Yahmed, Altaf Allah Abbassi, Amin Nikanjam, Heng Li, Foutse Khomh. (2023)  
**Deploying Deep Reinforcement Learning Systems: A Taxonomy of Challenges**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SE, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.12438v1)  

---


**ABSTRACT**  
Deep reinforcement learning (DRL), leveraging Deep Learning (DL) in reinforcement learning, has shown significant potential in achieving human-level autonomy in a wide range of domains, including robotics, computer vision, and computer games. This potential justifies the enthusiasm and growing interest in DRL in both academia and industry. However, the community currently focuses mostly on the development phase of DRL systems, with little attention devoted to DRL deployment. In this paper, we propose an empirical study on Stack Overflow (SO), the most popular Q&A forum for developers, to uncover and understand the challenges practitioners faced when deploying DRL systems. Specifically, we categorized relevant SO posts by deployment platforms: server/cloud, mobile/embedded system, browser, and game engine. After filtering and manual analysis, we examined 357 SO posts about DRL deployment, investigated the current state, and identified the challenges related to deploying DRL systems. Then, we investigate the prevalence and difficulty of these challenges. Results show that the general interest in DRL deployment is growing, confirming the study's relevance and importance. Results also show that DRL deployment is more difficult than other DRL issues. Additionally, we built a taxonomy of 31 unique challenges in deploying DRL to different platforms. On all platforms, RL environment-related challenges are the most popular, and communication-related challenges are the most difficult among practitioners. We hope our study inspires future research and helps the community overcome the most common and difficult challenges practitioners face when deploying DRL systems.

{{</citation>}}


### (67/111) FOSA: Full Information Maximum Likelihood (FIML) Optimized Self-Attention Imputation for Missing Data (Ou Deng et al., 2023)

{{<citation>}}

Ou Deng, Qun Jin. (2023)  
**FOSA: Full Information Maximum Likelihood (FIML) Optimized Self-Attention Imputation for Missing Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2308.12388v1)  

---


**ABSTRACT**  
In data imputation, effectively addressing missing values is pivotal, especially in intricate datasets. This paper delves into the FIML Optimized Self-attention (FOSA) framework, an innovative approach that amalgamates the strengths of Full Information Maximum Likelihood (FIML) estimation with the capabilities of self-attention neural networks. Our methodology commences with an initial estimation of missing values via FIML, subsequently refining these estimates by leveraging the self-attention mechanism. Our comprehensive experiments on both simulated and real-world datasets underscore FOSA's pronounced advantages over traditional FIML techniques, encapsulating facets of accuracy, computational efficiency, and adaptability to diverse data structures. Intriguingly, even in scenarios where the Structural Equation Model (SEM) might be mis-specified, leading to suboptimal FIML estimates, the robust architecture of FOSA's self-attention component adeptly rectifies and optimizes the imputation outcomes. Our empirical tests reveal that FOSA consistently delivers commendable predictions, even in the face of up to 40% random missingness, highlighting its robustness and potential for wide-scale applications in data imputation.

{{</citation>}}


### (68/111) Language Reward Modulation for Pretraining Reinforcement Learning (Ademi Adeniji et al., 2023)

{{<citation>}}

Ademi Adeniji, Amber Xie, Carmelo Sferrazza, Younggyo Seo, Stephen James, Pieter Abbeel. (2023)  
**Language Reward Modulation for Pretraining Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.12270v1)  

---


**ABSTRACT**  
Using learned reward functions (LRFs) as a means to solve sparse-reward reinforcement learning (RL) tasks has yielded some steady progress in task-complexity through the years. In this work, we question whether today's LRFs are best-suited as a direct replacement for task rewards. Instead, we propose leveraging the capabilities of LRFs as a pretraining signal for RL. Concretely, we propose $\textbf{LA}$nguage Reward $\textbf{M}$odulated $\textbf{P}$retraining (LAMP) which leverages the zero-shot capabilities of Vision-Language Models (VLMs) as a $\textit{pretraining}$ utility for RL as opposed to a downstream task reward. LAMP uses a frozen, pretrained VLM to scalably generate noisy, albeit shaped exploration rewards by computing the contrastive alignment between a highly diverse collection of language instructions and the image observations of an agent in its pretraining environment. LAMP optimizes these rewards in conjunction with standard novelty-seeking exploration rewards with reinforcement learning to acquire a language-conditioned, pretrained policy. Our VLM pretraining approach, which is a departure from previous attempts to use LRFs, can warmstart sample-efficient learning on robot manipulation tasks in RLBench.

{{</citation>}}


### (69/111) How to Protect Copyright Data in Optimization of Large Language Models? (Timothy Chu et al., 2023)

{{<citation>}}

Timothy Chu, Zhao Song, Chiwun Yang. (2023)  
**How to Protect Copyright Data in Optimization of Large Language Models?**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: AI, Attention, Language Model  
[Paper Link](http://arxiv.org/abs/2308.12247v1)  

---


**ABSTRACT**  
Large language models (LLMs) and generative AI have played a transformative role in computer research and applications. Controversy has arisen as to whether these models output copyrighted data, which can occur if the data the models are trained on is copyrighted. LLMs are built on the transformer neural network architecture, which in turn relies on a mathematical computation called Attention that uses the softmax function.   In this paper, we show that large language model training and optimization can be seen as a softmax regression problem. We then establish a method of efficiently performing softmax regression, in a way that prevents the regression function from generating copyright data. This establishes a theoretical method of training large language models in a way that avoids generating copyright data.

{{</citation>}}


### (70/111) Curriculum Learning with Adam: The Devil Is in the Wrong Details (Lucas Weber et al., 2023)

{{<citation>}}

Lucas Weber, Jaap Jumelet, Paul Michel, Elia Bruni, Dieuwke Hupkes. (2023)  
**Curriculum Learning with Adam: The Devil Is in the Wrong Details**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.12202v1)  

---


**ABSTRACT**  
Curriculum learning (CL) posits that machine learning models -- similar to humans -- may learn more efficiently from data that match their current learning progress. However, CL methods are still poorly understood and, in particular for natural language processing (NLP), have achieved only limited success. In this paper, we explore why. Starting from an attempt to replicate and extend a number of recent curriculum methods, we find that their results are surprisingly brittle when applied to NLP. A deep dive into the (in)effectiveness of the curricula in some scenarios shows us why: when curricula are employed in combination with the popular Adam optimisation algorithm, they oftentimes learn to adapt to suboptimally chosen optimisation parameters for this algorithm. We present a number of different case studies with different common hand-crafted and automated CL approaches to illustrate this phenomenon, and we find that none of them outperforms optimisation with only Adam with well-chosen hyperparameters. As such, our results contribute to understanding why CL methods work, but at the same time urge caution when claiming positive results.

{{</citation>}}


### (71/111) Cached Operator Reordering: A Unified View for Fast GNN Training (Julia Bazinska et al., 2023)

{{<citation>}}

Julia Bazinska, Andrei Ivanov, Tal Ben-Nun, Nikoli Dryden, Maciej Besta, Siyuan Shen, Torsten Hoefler. (2023)  
**Cached Operator Reordering: A Unified View for Fast GNN Training**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-PF, cs.LG  
Keywords: Attention, GNN, Graph Convolutional Network, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.12093v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) are a powerful tool for handling structured graph data and addressing tasks such as node classification, graph classification, and clustering. However, the sparse nature of GNN computation poses new challenges for performance optimization compared to traditional deep neural networks. We address these challenges by providing a unified view of GNN computation, I/O, and memory. By analyzing the computational graphs of the Graph Convolutional Network (GCN) and Graph Attention (GAT) layers -- two widely used GNN layers -- we propose alternative computation strategies. We present adaptive operator reordering with caching, which achieves a speedup of up to 2.43x for GCN compared to the current state-of-the-art. Furthermore, an exploration of different caching schemes for GAT yields a speedup of up to 1.94x. The proposed optimizations save memory, are easily implemented across various hardware platforms, and have the potential to alleviate performance bottlenecks in training large-scale GNN models.

{{</citation>}}


### (72/111) InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4 (Lai Wei et al., 2023)

{{<citation>}}

Lai Wei, Zihao Jiang, Weiran Huang, Lichao Sun. (2023)  
**InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.12067v1)  

---


**ABSTRACT**  
Multimodal large language models acquire their instruction-following capabilities through a two-stage training process: pre-training on image-text pairs and fine-tuning on supervised vision-language instruction data. Recent studies have shown that large language models can achieve satisfactory results even with a limited amount of high-quality instruction-following data. In this paper, we introduce InstructionGPT-4, which is fine-tuned on a small dataset comprising only 200 examples, amounting to approximately 6% of the instruction-following data used in the alignment dataset for MiniGPT-4. We first propose several metrics to access the quality of multimodal instruction data. Based on these metrics, we present a simple and effective data selector to automatically identify and filter low-quality vision-language data. By employing this method, InstructionGPT-4 outperforms the original MiniGPT-4 on various evaluations (e.g., visual question answering, GPT-4 preference). Overall, our findings demonstrate that less but high-quality instruction tuning data is efficient to enable multimodal large language models to generate better output.

{{</citation>}}


### (73/111) Bias-Aware Minimisation: Understanding and Mitigating Estimator Bias in Private SGD (Moritz Knolle et al., 2023)

{{<citation>}}

Moritz Knolle, Robert Dorfman, Alexander Ziller, Daniel Rueckert, Georgios Kaissis. (2023)  
**Bias-Aware Minimisation: Understanding and Mitigating Estimator Bias in Private SGD**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Bias, ImageNet  
[Paper Link](http://arxiv.org/abs/2308.12018v1)  

---


**ABSTRACT**  
Differentially private SGD (DP-SGD) holds the promise of enabling the safe and responsible application of machine learning to sensitive datasets. However, DP-SGD only provides a biased, noisy estimate of a mini-batch gradient. This renders optimisation steps less effective and limits model utility as a result. With this work, we show a connection between per-sample gradient norms and the estimation bias of the private gradient oracle used in DP-SGD. Here, we propose Bias-Aware Minimisation (BAM) that allows for the provable reduction of private gradient estimator bias. We show how to efficiently compute quantities needed for BAM to scale to large neural networks and highlight similarities to closely related methods such as Sharpness-Aware Minimisation. Finally, we provide empirical evidence that BAM not only reduces bias but also substantially improves privacy-utility trade-offs on the CIFAR-10, CIFAR-100, and ImageNet-32 datasets.

{{</citation>}}


### (74/111) Graph Neural Stochastic Differential Equations (Richard Bergna et al., 2023)

{{<citation>}}

Richard Bergna, Felix Opolka, Pietro Liò, Jose Miguel Hernandez-Lobato. (2023)  
**Graph Neural Stochastic Differential Equations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2308.12316v1)  

---


**ABSTRACT**  
We present a novel model Graph Neural Stochastic Differential Equations (Graph Neural SDEs). This technique enhances the Graph Neural Ordinary Differential Equations (Graph Neural ODEs) by embedding randomness into data representation using Brownian motion. This inclusion allows for the assessment of prediction uncertainty, a crucial aspect frequently missed in current models. In our framework, we spotlight the \textit{Latent Graph Neural SDE} variant, demonstrating its effectiveness. Through empirical studies, we find that Latent Graph Neural SDEs surpass conventional models like Graph Convolutional Networks and Graph Neural ODEs, especially in confidence prediction, making them superior in handling out-of-distribution detection across both static and spatio-temporal contexts.

{{</citation>}}


### (75/111) Trustworthy Representation Learning Across Domains (Ronghang Zhu et al., 2023)

{{<citation>}}

Ronghang Zhu, Dongliang Guo, Daiqing Qi, Zhixuan Chu, Xiang Yu, Sheng Li. (2023)  
**Trustworthy Representation Learning Across Domains**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.12315v1)  

---


**ABSTRACT**  
As AI systems have obtained significant performance to be deployed widely in our daily live and human society, people both enjoy the benefits brought by these technologies and suffer many social issues induced by these systems. To make AI systems good enough and trustworthy, plenty of researches have been done to build guidelines for trustworthy AI systems. Machine learning is one of the most important parts for AI systems and representation learning is the fundamental technology in machine learning. How to make the representation learning trustworthy in real-world application, e.g., cross domain scenarios, is very valuable and necessary for both machine learning and AI system fields. Inspired by the concepts in trustworthy AI, we proposed the first trustworthy representation learning across domains framework which includes four concepts, i.e, robustness, privacy, fairness, and explainability, to give a comprehensive literature review on this research direction. Specifically, we first introduce the details of the proposed trustworthy framework for representation learning across domains. Second, we provide basic notions and comprehensively summarize existing methods for the trustworthy framework from four concepts. Finally, we conclude this survey with insights and discussions on future research directions.

{{</citation>}}


### (76/111) Will More Expressive Graph Neural Networks do Better on Generative Tasks? (Xiandong Zou et al., 2023)

{{<citation>}}

Xiandong Zou, Xiangyu Zhao, Pietro Liò, Yiren Zhao. (2023)  
**Will More Expressive Graph Neural Networks do Better on Generative Tasks?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-bio-BM, stat-ML  
Keywords: GCP, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.11978v1)  

---


**ABSTRACT**  
Graph generation poses a significant challenge as it involves predicting a complete graph with multiple nodes and edges based on simply a given label. This task also carries fundamental importance to numerous real-world applications, including de-novo drug and molecular design. In recent years, several successful methods have emerged in the field of graph generation. However, these approaches suffer from two significant shortcomings: (1) the underlying Graph Neural Network (GNN) architectures used in these methods are often underexplored; and (2) these methods are often evaluated on only a limited number of metrics. To fill this gap, we investigate the expressiveness of GNNs under the context of the molecular graph generation task, by replacing the underlying GNNs of graph generative models with more expressive GNNs. Specifically, we analyse the performance of six GNNs in two different generative frameworks (GCPN and GraphAF), on six different molecular generative objectives on the ZINC-250k dataset. Through our extensive experiments, we demonstrate that advanced GNNs can indeed improve the performance of GCPN and GraphAF on molecular generation tasks, but GNN expressiveness is not a necessary condition for a good GNN-based generative model. Moreover, we show that GCPN and GraphAF with advanced GNNs can achieve state-of-the-art results across 17 other non-GNN-based graph generative approaches, such as variational autoencoders and Bayesian optimisation models, on the proposed molecular generative objectives (DRD2, Median1, Median2), which are important metrics for de-novo molecular design.

{{</citation>}}


### (77/111) Multi-scale Transformer Pyramid Networks for Multivariate Time Series Forecasting (Yifan Zhang et al., 2023)

{{<citation>}}

Yifan Zhang, Rui Wu, Sergiu M. Dascalu, Frederick C. Harris Jr. (2023)  
**Multi-scale Transformer Pyramid Networks for Multivariate Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.11946v1)  

---


**ABSTRACT**  
Multivariate Time Series (MTS) forecasting involves modeling temporal dependencies within historical records. Transformers have demonstrated remarkable performance in MTS forecasting due to their capability to capture long-term dependencies. However, prior work has been confined to modeling temporal dependencies at either a fixed scale or multiple scales that exponentially increase (most with base 2). This limitation hinders their effectiveness in capturing diverse seasonalities, such as hourly and daily patterns. In this paper, we introduce a dimension invariant embedding technique that captures short-term temporal dependencies and projects MTS data into a higher-dimensional space, while preserving the dimensions of time steps and variables in MTS data. Furthermore, we present a novel Multi-scale Transformer Pyramid Network (MTPNet), specifically designed to effectively capture temporal dependencies at multiple unconstrained scales. The predictions are inferred from multi-scale latent representations obtained from transformers at various scales. Extensive experiments on nine benchmark datasets demonstrate that the proposed MTPNet outperforms recent state-of-the-art methods.

{{</citation>}}


### (78/111) Retail Demand Forecasting: A Comparative Study for Multivariate Time Series (Md Sabbirul Haque et al., 2023)

{{<citation>}}

Md Sabbirul Haque, Md Shahedul Amin, Jonayet Miah. (2023)  
**Retail Demand Forecasting: A Comparative Study for Multivariate Time Series**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-fin-ST  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.11939v1)  

---


**ABSTRACT**  
Accurate demand forecasting in the retail industry is a critical determinant of financial performance and supply chain efficiency. As global markets become increasingly interconnected, businesses are turning towards advanced prediction models to gain a competitive edge. However, existing literature mostly focuses on historical sales data and ignores the vital influence of macroeconomic conditions on consumer spending behavior. In this study, we bridge this gap by enriching time series data of customer demand with macroeconomic variables, such as the Consumer Price Index (CPI), Index of Consumer Sentiment (ICS), and unemployment rates. Leveraging this comprehensive dataset, we develop and compare various regression and machine learning models to predict retail demand accurately.

{{</citation>}}


### (79/111) Addressing Selection Bias in Computerized Adaptive Testing: A User-Wise Aggregate Influence Function Approach (Soonwoo Kwon et al., 2023)

{{<citation>}}

Soonwoo Kwon, Sojung Kim, Seunghyun Lee, Jin-Young Kim, Suyeong An, Kyuseok Kim. (2023)  
**Addressing Selection Bias in Computerized Adaptive Testing: A User-Wise Aggregate Influence Function Approach**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.11912v1)  

---


**ABSTRACT**  
Computerized Adaptive Testing (CAT) is a widely used, efficient test mode that adapts to the examinee's proficiency level in the test domain. CAT requires pre-trained item profiles, for CAT iteratively assesses the student real-time based on the registered items' profiles, and selects the next item to administer using candidate items' profiles. However, obtaining such item profiles is a costly process that involves gathering a large, dense item-response data, then training a diagnostic model on the collected data. In this paper, we explore the possibility of leveraging response data collected in the CAT service. We first show that this poses a unique challenge due to the inherent selection bias introduced by CAT, i.e., more proficient students will receive harder questions. Indeed, when naively training the diagnostic model using CAT response data, we observe that item profiles deviate significantly from the ground-truth. To tackle the selection bias issue, we propose the user-wise aggregate influence function method. Our intuition is to filter out users whose response data is heavily biased in an aggregate manner, as judged by how much perturbation the added data will introduce during parameter estimation. This way, we may enhance the performance of CAT while introducing minimal bias to the item profiles. We provide extensive experiments to demonstrate the superiority of our proposed method based on the three public datasets and one dataset that contains real-world CAT response data.

{{</citation>}}


### (80/111) Adversarial Training Using Feedback Loops (Ali Haisam Muhammad Rafid et al., 2023)

{{<citation>}}

Ali Haisam Muhammad Rafid, Adrian Sandu. (2023)  
**Adversarial Training Using Feedback Loops**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2308.11881v2)  

---


**ABSTRACT**  
Deep neural networks (DNN) have found wide applicability in numerous fields due to their ability to accurately learn very complex input-output relations. Despite their accuracy and extensive use, DNNs are highly susceptible to adversarial attacks due to limited generalizability. For future progress in the field, it is essential to build DNNs that are robust to any kind of perturbations to the data points. In the past, many techniques have been proposed to robustify DNNs using first-order derivative information of the network.   This paper proposes a new robustification approach based on control theory. A neural network architecture that incorporates feedback control, named Feedback Neural Networks, is proposed. The controller is itself a neural network, which is trained using regular and adversarial data such as to stabilize the system outputs. The novel adversarial training approach based on the feedback control architecture is called Feedback Looped Adversarial Training (FLAT). Numerical results on standard test problems empirically show that our FLAT method is more effective than the state-of-the-art to guard against adversarial attacks.

{{</citation>}}


## eess.IV (2)



### (81/111) TAI-GAN: Temporally and Anatomically Informed GAN for early-to-late frame conversion in dynamic cardiac PET motion correction (Xueqi Guo et al., 2023)

{{<citation>}}

Xueqi Guo, Luyao Shi, Xiongchao Chen, Bo Zhou, Qiong Liu, Huidong Xie, Yi-Hwa Liu, Richard Palyo, Edward J. Miller, Albert J. Sinusas, Bruce Spottiswoode, Chi Liu, Nicha C. Dvornek. (2023)  
**TAI-GAN: Temporally and Anatomically Informed GAN for early-to-late frame conversion in dynamic cardiac PET motion correction**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.12443v1)  

---


**ABSTRACT**  
The rapid tracer kinetics of rubidium-82 ($^{82}$Rb) and high variation of cross-frame distribution in dynamic cardiac positron emission tomography (PET) raise significant challenges for inter-frame motion correction, particularly for the early frames where conventional intensity-based image registration techniques are not applicable. Alternatively, a promising approach utilizes generative methods to handle the tracer distribution changes to assist existing registration methods. To improve frame-wise registration and parametric quantification, we propose a Temporally and Anatomically Informed Generative Adversarial Network (TAI-GAN) to transform the early frames into the late reference frame using an all-to-one mapping. Specifically, a feature-wise linear modulation layer encodes channel-wise parameters generated from temporal tracer kinetics information, and rough cardiac segmentations with local shifts serve as the anatomical information. We validated our proposed method on a clinical $^{82}$Rb PET dataset and found that our TAI-GAN can produce converted early frames with high image quality, comparable to the real reference frames. After TAI-GAN conversion, motion estimation accuracy and clinical myocardial blood flow (MBF) quantification were improved compared to using the original frames. Our code is published at https://github.com/gxq1998/TAI-GAN.

{{</citation>}}


### (82/111) Anisotropic Hybrid Networks for liver tumor segmentation with uncertainty quantification (Benjamin Lambert et al., 2023)

{{<citation>}}

Benjamin Lambert, Pauline Roca, Florence Forbes, Senan Doyle, Michel Dojat. (2023)  
**Anisotropic Hybrid Networks for liver tumor segmentation with uncertainty quantification**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV, q-bio-QM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.11969v1)  

---


**ABSTRACT**  
The burden of liver tumors is important, ranking as the fourth leading cause of cancer mortality. In case of hepatocellular carcinoma (HCC), the delineation of liver and tumor on contrast-enhanced magnetic resonance imaging (CE-MRI) is performed to guide the treatment strategy. As this task is time-consuming, needs high expertise and could be subject to inter-observer variability there is a strong need for automatic tools. However, challenges arise from the lack of available training data, as well as the high variability in terms of image resolution and MRI sequence. In this work we propose to compare two different pipelines based on anisotropic models to obtain the segmentation of the liver and tumors. The first pipeline corresponds to a baseline multi-class model that performs the simultaneous segmentation of the liver and tumor classes. In the second approach, we train two distinct binary models, one segmenting the liver only and the other the tumors. Our results show that both pipelines exhibit different strengths and weaknesses. Moreover we propose an uncertainty quantification strategy allowing the identification of potential false positive tumor lesions. Both solutions were submitted to the MICCAI 2023 Atlas challenge regarding liver and tumor segmentation.

{{</citation>}}


## cs.IR (4)



### (83/111) Evolution of ESG-focused DLT Research: An NLP Analysis of the Literature (Walter Hernandez et al., 2023)

{{<citation>}}

Walter Hernandez, Kamil Tylinski, Alastair Moore, Niall Roche, Nikhil Vadgama, Horst Treiblmaier, Jiangbo Shangguan, Paolo Tasca, Jiahua Xu. (2023)  
**Evolution of ESG-focused DLT Research: An NLP Analysis of the Literature**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: NER, NLP, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2308.12420v1)  

---


**ABSTRACT**  
Distributed Ledger Technologies (DLTs) have rapidly evolved, necessitating comprehensive insights into their diverse components. However, a systematic literature review that emphasizes the Environmental, Sustainability, and Governance (ESG) components of DLT remains lacking. To bridge this gap, we selected 107 seed papers to build a citation network of 63,083 references and refined it to a corpus of 24,539 publications for analysis. Then, we labeled the named entities in 46 papers according to twelve top-level categories derived from an established technology taxonomy and enhanced the taxonomy by pinpointing DLT's ESG elements. Leveraging transformer-based language models, we fine-tuned a pre-trained language model for a Named Entity Recognition (NER) task using our labeled dataset. We used our fine-tuned language model to distill the corpus to 505 key papers, facilitating a literature review via named entities and temporal graph analysis on DLT evolution in the context of ESG. Our contributions are a methodology to conduct a machine learning-driven systematic literature review in the DLT field, placing a special emphasis on ESG aspects. Furthermore, we present a first-of-its-kind NER dataset, composed of 54,808 named entities, designed for DLT and ESG-related explorations.

{{</citation>}}


### (84/111) LLMRec: Benchmarking Large Language Models on Recommendation Task (Junling Liu et al., 2023)

{{<citation>}}

Junling Liu, Chao Liu, Peilin Zhou, Qichen Ye, Dading Chong, Kang Zhou, Yueqi Xie, Yuwei Cao, Shoujin Wang, Chenyu You, Philip S. Yu. (2023)  
**LLMRec: Benchmarking Large Language Models on Recommendation Task**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: ChatGPT, GLM, GPT, LLaMA, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.12241v1)  

---


**ABSTRACT**  
Recently, the fast development of Large Language Models (LLMs) such as ChatGPT has significantly advanced NLP tasks by enhancing the capabilities of conversational models. However, the application of LLMs in the recommendation domain has not been thoroughly investigated. To bridge this gap, we propose LLMRec, a LLM-based recommender system designed for benchmarking LLMs on various recommendation tasks. Specifically, we benchmark several popular off-the-shelf LLMs, such as ChatGPT, LLaMA, ChatGLM, on five recommendation tasks, including rating prediction, sequential recommendation, direct recommendation, explanation generation, and review summarization. Furthermore, we investigate the effectiveness of supervised finetuning to improve LLMs' instruction compliance ability. The benchmark results indicate that LLMs displayed only moderate proficiency in accuracy-based tasks such as sequential and direct recommendation. However, they demonstrated comparable performance to state-of-the-art methods in explainability-based tasks. We also conduct qualitative evaluations to further evaluate the quality of contents generated by different models, and the results show that LLMs can truly understand the provided information and generate clearer and more reasonable results. We aspire that this benchmark will serve as an inspiration for researchers to delve deeper into the potential of LLMs in enhancing recommendation performance. Our codes, processed data and benchmark results are available at https://github.com/williamliujl/LLMRec.

{{</citation>}}


### (85/111) Counterfactual Graph Augmentation for Consumer Unfairness Mitigation in Recommender Systems (Ludovico Boratto et al., 2023)

{{<citation>}}

Ludovico Boratto, Francesco Fabbri, Gianni Fenu, Mirko Marras, Giacomo Medda. (2023)  
**Counterfactual Graph Augmentation for Consumer Unfairness Mitigation in Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.12083v1)  

---


**ABSTRACT**  
In recommendation literature, explainability and fairness are becoming two prominent perspectives to consider. However, prior works have mostly addressed them separately, for instance by explaining to consumers why a certain item was recommended or mitigating disparate impacts in recommendation utility. None of them has leveraged explainability techniques to inform unfairness mitigation. In this paper, we propose an approach that relies on counterfactual explanations to augment the set of user-item interactions, such that using them while inferring recommendations leads to fairer outcomes. Modeling user-item interactions as a bipartite graph, our approach augments the latter by identifying new user-item edges that not only can explain the original unfairness by design, but can also mitigate it. Experiments on two public data sets show that our approach effectively leads to a better trade-off between fairness and recommendation utility compared with state-of-the-art mitigation procedures. We further analyze the characteristics of added edges to highlight key unfairness patterns. Source code available at https://github.com/jackmedda/RS-BGExplainer/tree/cikm2023.

{{</citation>}}


### (86/111) LKPNR: LLM and KG for Personalized News Recommendation Framework (Chen hao et al., 2023)

{{<citation>}}

Chen hao, Xie Runfeng, Cui Xiangyang, Yan Zhou, Wang Xin, Xuan Zhanwei, Zhang Kai. (2023)  
**LKPNR: LLM and KG for Personalized News Recommendation Framework**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2308.12028v1)  

---


**ABSTRACT**  
Accurately recommending candidate news articles to users is a basic challenge faced by personalized news recommendation systems. Traditional methods are usually difficult to grasp the complex semantic information in news texts, resulting in unsatisfactory recommendation results. Besides, these traditional methods are more friendly to active users with rich historical behaviors. However, they can not effectively solve the "long tail problem" of inactive users. To address these issues, this research presents a novel general framework that combines Large Language Models (LLM) and Knowledge Graphs (KG) into semantic representations of traditional methods. In order to improve semantic understanding in complex news texts, we use LLMs' powerful text understanding ability to generate news representations containing rich semantic information. In addition, our method combines the information about news entities and mines high-order structural information through multiple hops in KG, thus alleviating the challenge of long tail distribution. Experimental results demonstrate that compared with various traditional models, the framework significantly improves the recommendation effect. The successful integration of LLM and KG in our framework has established a feasible path for achieving more accurate personalized recommendations in the news field. Our code is available at https://github.com/Xuan-ZW/LKPNR.

{{</citation>}}


## cs.SE (5)



### (87/111) Benchmarking Causal Study to Interpret Large Language Models for Source Code (Daniel Rodriguez-Cardenas et al., 2023)

{{<citation>}}

Daniel Rodriguez-Cardenas, David N. Palacio, Dipin Khati, Henry Burke, Denys Poshyvanyk. (2023)  
**Benchmarking Causal Study to Interpret Large Language Models for Source Code**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: BLEU, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.12415v1)  

---


**ABSTRACT**  
One of the most common solutions adopted by software researchers to address code generation is by training Large Language Models (LLMs) on massive amounts of source code. Although a number of studies have shown that LLMs have been effectively evaluated on popular accuracy metrics (e.g., BLEU, CodeBleu), previous research has largely overlooked the role of Causal Inference as a fundamental component of the interpretability of LLMs' performance. Existing benchmarks and datasets are meant to highlight the difference between the expected and the generated outcome, but do not take into account confounding variables (e.g., lines of code, prompt size) that equally influence the accuracy metrics. The fact remains that, when dealing with generative software tasks by LLMs, no benchmark is available to tell researchers how to quantify neither the causal effect of SE-based treatments nor the correlation of confounders to the model's performance. In an effort to bring statistical rigor to the evaluation of LLMs, this paper introduces a benchmarking strategy named Galeras comprised of curated testbeds for three SE tasks (i.e., code completion, code summarization, and commit generation) to help aid the interpretation of LLMs' performance. We illustrate the insights of our benchmarking strategy by conducting a case study on the performance of ChatGPT under distinct prompt engineering methods. The results of the case study demonstrate the positive causal influence of prompt semantics on ChatGPT's generative performance by an average treatment effect of $\approx 3\%$. Moreover, it was found that confounders such as prompt size are highly correlated with accuracy metrics ($\approx 0.412\%$). The end result of our case study is to showcase causal inference evaluations, in practice, to reduce confounding bias. By reducing the bias, we offer an interpretable solution for the accuracy metric under analysis.

{{</citation>}}


### (88/111) Bugsplainer: Leveraging Code Structures to Explain Software Bugs with Neural Machine Translation (Parvez Mahbub et al., 2023)

{{<citation>}}

Parvez Mahbub, Mohammad Masudur Rahman, Ohiduzzaman Shuvo, Avinash Gopal. (2023)  
**Bugsplainer: Leveraging Code Structures to Explain Software Bugs with Neural Machine Translation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Machine Translation, T5  
[Paper Link](http://arxiv.org/abs/2308.12267v1)  

---


**ABSTRACT**  
Software bugs cost the global economy billions of dollars each year and take up ~50% of the development time. Once a bug is reported, the assigned developer attempts to identify and understand the source code responsible for the bug and then corrects the code. Over the last five decades, there has been significant research on automatically finding or correcting software bugs. However, there has been little research on automatically explaining the bugs to the developers, which is essential but a highly challenging task. In this paper, we propose Bugsplainer, a novel web-based debugging solution that generates natural language explanations for software bugs by learning from a large corpus of bug-fix commits. Bugsplainer leverages code structures to reason about a bug and employs the fine-tuned version of a text generation model, CodeT5, to generate the explanations.   Tool video: https://youtu.be/xga-ScvULpk

{{</citation>}}


### (89/111) Resiliency Analysis of LLM generated models for Industrial Automation (Oluwatosin Ogundare et al., 2023)

{{<citation>}}

Oluwatosin Ogundare, Gustavo Quiros Araya, Ioannis Akrotirianakis, Ankit Shukla. (2023)  
**Resiliency Analysis of LLM generated models for Industrial Automation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.12129v1)  

---


**ABSTRACT**  
This paper proposes a study of the resilience and efficiency of automatically generated industrial automation and control systems using Large Language Models (LLMs). The approach involves modeling the system using percolation theory to estimate its resilience and formulating the design problem as an optimization problem subject to constraints. Techniques from stochastic optimization and regret analysis are used to find a near-optimal solution with provable regret bounds. The study aims to provide insights into the effectiveness and reliability of automatically generated systems in industrial automation and control, and to identify potential areas for improvement in their design and implementation.

{{</citation>}}


### (90/111) On Using Information Retrieval to Recommend Machine Learning Good Practices for Software Engineers (Laura Cabra-Acela et al., 2023)

{{<citation>}}

Laura Cabra-Acela, Anamaria Mojica-Hanke, Mario Linares-Vásquez, Steffen Herbold. (2023)  
**On Using Information Retrieval to Recommend Machine Learning Good Practices for Software Engineers**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2308.12095v1)  

---


**ABSTRACT**  
Machine learning (ML) is nowadays widely used for different purposes and in several disciplines. From self-driving cars to automated medical diagnosis, machine learning models extensively support users' daily activities, and software engineering tasks are no exception. Not embracing good ML practices may lead to pitfalls that hinder the performance of an ML system and potentially lead to unexpected results. Despite the existence of documentation and literature about ML best practices, many non-ML experts turn towards gray literature like blogs and Q&A systems when looking for help and guidance when implementing ML systems. To better aid users in distilling relevant knowledge from such sources, we propose a recommender system that recommends ML practices based on the user's context. As a first step in creating a recommender system for machine learning practices, we implemented Idaka. A tool that provides two different approaches for retrieving/generating ML best practices: i) an information retrieval (IR) engine and ii) a large language model. The IR-engine uses BM25 as the algorithm for retrieving the practices, and a large language model, in our case Alpaca. The platform has been designed to allow comparative studies of best practices retrieval tools. Idaka is publicly available at GitHub: https://bit.ly/idaka. Video: https://youtu.be/cEb-AhIPxnM.

{{</citation>}}


### (91/111) Integrating Large Language Models into the Debugging C Compiler for generating contextual error explanations (Andrew Taylor et al., 2023)

{{<citation>}}

Andrew Taylor, Alexandra Vassar, Jake Renzella, Hammond Pearce. (2023)  
**Integrating Large Language Models into the Debugging C Compiler for generating contextual error explanations**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-PL, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.11873v1)  

---


**ABSTRACT**  
This paper introduces a method for Large Language Models (LLM) to produce enhanced compiler error explanations, in simple language, within our Debugging C Compiler (DCC). It is well documented that compiler error messages have been known to present a barrier for novices learning how to program. Although our initial use of DCC in introductory programming (CS1) has been instrumental in teaching C to novice programmers by providing safeguards to commonly occurring errors and translating the usually cryptic compiler error messages at both compile- and run-time, we proposed that incorporating LLM-generated explanations would further enhance the learning experience for novice programmers. Through an expert evaluation, we observed that LLM-generated explanations for compiler errors were conceptually accurate in 90% of compile-time errors, and 75% of run-time errors. Additionally, the new DCC-help tool has been increasingly adopted by students, with an average of 1047 unique runs per week, demonstrating a promising initial assessment of using LLMs to complement compiler output to enhance programming education for beginners. We release our tool as open-source to the community.

{{</citation>}}


## cs.AI (3)



### (92/111) A Theory of Intelligences: Concepts, Models, Implications (Michael E. Hochberg, 2023)

{{<citation>}}

Michael E. Hochberg. (2023)  
**A Theory of Intelligences: Concepts, Models, Implications**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.12411v1)  

---


**ABSTRACT**  
Intelligence is a human construct to represent the ability to achieve goals. Given this wide berth, intelligence has been defined countless times, studied in a variety of ways and quantified using numerous measures. Understanding intelligence ultimately requires theory and quantification, both of which are elusive. My main objectives are to identify some of the central elements in and surrounding intelligence, discuss some of its challenges and propose a theory based on first principles. I focus on intelligence as defined by and for humans, frequently in comparison to machines, with the intention of setting the stage for more general characterizations in life, collectives, human designs such as AI and in non-designed physical and chemical systems. I discuss key features of intelligence, including path efficiency and goal accuracy, intelligence as a Black Box, environmental influences, flexibility to deal with surprisal, the regress of intelligence, the relativistic nature of intelligence and difficulty, and temporal changes in intelligence including its evolution. I present a framework for a first principles Theory of IntelligenceS (TIS), based on the quantifiable macro-scale system features of difficulty, surprisal and goal resolution accuracy. The proposed partitioning of uncertainty/solving and accuracy/understanding is particularly novel since it predicts that paths to a goal not only function to accurately achieve goals, but as experimentations leading to higher probabilities for future attainable goals and increased breadth to enter new goal spaces. TIS can therefore explain endeavors that do not necessarily affect Darwinian fitness, such as leisure, politics, games and art. I conclude with several conceptual advances of TIS including a compact mathematical form of surprisal and difficulty, the theoretical basis of TIS, and open questions.

{{</citation>}}


### (93/111) From Instructions to Intrinsic Human Values -- A Survey of Alignment Goals for Big Models (Jing Yao et al., 2023)

{{<citation>}}

Jing Yao, Xiaoyuan Yi, Xiting Wang, Jindong Wang, Xing Xie. (2023)  
**From Instructions to Intrinsic Human Values -- A Survey of Alignment Goals for Big Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CY, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.12014v1)  

---


**ABSTRACT**  
Big models, exemplified by Large Language Models (LLMs), are models typically pre-trained on massive data and comprised of enormous parameters, which not only obtain significantly improved performance across diverse tasks but also present emergent capabilities absent in smaller models. However, the growing intertwining of big models with everyday human lives poses potential risks and might cause serious social harm. Therefore, many efforts have been made to align LLMs with humans to make them better follow user instructions and satisfy human preferences. Nevertheless, `what to align with' has not been fully discussed, and inappropriate alignment goals might even backfire. In this paper, we conduct a comprehensive survey of different alignment goals in existing work and trace their evolution paths to help identify the most essential goal. Particularly, we investigate related works from two perspectives: the definition of alignment goals and alignment evaluation. Our analysis encompasses three distinct levels of alignment goals and reveals a goal transformation from fundamental abilities to value orientation, indicating the potential of intrinsic human values as the alignment goal for enhanced LLMs. Based on such results, we further discuss the challenges of achieving such intrinsic value alignment and provide a collection of available resources for future research on the alignment of big models.

{{</citation>}}


### (94/111) Towards CausalGPT: A Multi-Agent Approach for Faithful Knowledge Reasoning via Promoting Causal Consistency in LLMs (Ziyi Tang et al., 2023)

{{<citation>}}

Ziyi Tang, Ruilin Wang, Weixing Chen, Keze Wang, Yang Liu, Tianshui Chen, Liang Lin. (2023)  
**Towards CausalGPT: A Multi-Agent Approach for Faithful Knowledge Reasoning via Promoting Causal Consistency in LLMs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-MA, cs.AI  
Keywords: GPT, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.11914v1)  

---


**ABSTRACT**  
Despite advancements in LLMs, knowledge-based reasoning remains a longstanding issue due to the fragility of knowledge recall and inference. Existing methods primarily encourage LLMs to autonomously plan and solve problems or to extensively sample reasoning chains without addressing the conceptual and inferential fallacies. Attempting to alleviate inferential fallacies and drawing inspiration from multi-agent collaboration, we present a framework to increase faithfulness and causality for knowledge-based reasoning. Specifically, we propose to employ multiple intelligent agents (i.e., reasoner and causal evaluator) to work collaboratively in a reasoning-and-consensus paradigm for elevated reasoning faithfulness. The reasoners focus on providing solutions with human-like causality to solve open-domain problems. On the other hand, the causal evaluator agent scrutinizes if the answer in a solution is causally deducible from the question and vice versa, with a counterfactual answer replacing the original. According to the extensive and comprehensive evaluations on a variety of knowledge reasoning tasks (e.g., science question answering and commonsense reasoning), our framework outperforms all compared state-of-the-art approaches by large margins.

{{</citation>}}


## cs.IT (2)



### (95/111) Private Information Retrieval with Private Noisy Side Information (Hassan ZivariFard et al., 2023)

{{<citation>}}

Hassan ZivariFard, Remi A. Chou. (2023)  
**Private Information Retrieval with Private Noisy Side Information**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2308.12374v1)  

---


**ABSTRACT**  
Consider Private Information Retrieval (PIR), where a client wants to retrieve one file out of $K$ files that are replicated in $N$ different servers and the client selection must remain private when up to $T$ servers may collude. Additionally, suppose that the client has noisy side information about each of the $K$ files, and the side information about a specific file is obtained by passing this file through one of $D$ possible discrete memoryless test channels, where $D\le K$. While the statistics of the test channels are known by the client and by all the servers, the specific mapping $\boldsymbol{\mathcal{M}}$ between the files and the test channels is unknown to the servers. We study this problem under two different privacy metrics. Under the first privacy metric, the client wants to preserve the privacy of its desired file selection and the mapping $\boldsymbol{\mathcal{M}}$. Under the second privacy metric, the client wants to preserve the privacy of its desired file and the mapping $\boldsymbol{\mathcal{M}}$, but is willing to reveal the index of the test channel that is associated to its desired file. For both of these two privacy metrics, we derive the optimal normalized download cost. Our problem setup generalizes PIR with colluding servers, PIR with private noiseless side information, and PIR with private side information under storage constraints.

{{</citation>}}


### (96/111) An EEP-based robust beamforming approach for superdirective antenna arrays and experimental validations (Mengying Gao et al., 2023)

{{<citation>}}

Mengying Gao, Haifan Yin, Liangcheng Han. (2023)  
**An EEP-based robust beamforming approach for superdirective antenna arrays and experimental validations**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2308.11934v1)  

---


**ABSTRACT**  
A superdirective antenna array has the potential to achieve an array gain proportional to the square of the number of antennas, making it of great value for future wireless communications. However, designing the superdirective beamformer while considering the complicated mutual-coupling effect is a practical challenge. Moreover, the superdirective antenna array is highly sensitive to excitation errors, especially when the number of antennas is large or the antenna spacing is very small, necessitating demanding and precise control over excitations. To address these problems, we first propose a novel superdirective beamforming approach based on the embedded element pattern (EEP), which contains the coupling information. The closed-form solution to the beamforming vector and the corresponding directivity factor are derived. This method relies on the beam coupling factors (BCFs) between the antennas, which are provided in closed form. To address the high sensitivity problem, we formulate a constrained optimization problem and propose an EEP-aided orthogonal complement-based robust beamforming (EEP-OCRB) algorithm. Full-wave simulation results validate our proposed methods. Finally, we build a prototype of a 5-dipole superdirective antenna array and conduct real-world experiments. The measurement results demonstrate the realization of the superdirectivity with our EEP-based method, as well as the robustness of the proposed EEP-OCRB algorithm to excitation errors.

{{</citation>}}


## cs.CY (2)



### (97/111) A Model for Integrating Generative AI into Course Content Development (Ethan Dickey et al., 2023)

{{<citation>}}

Ethan Dickey, Andres Bejarano. (2023)  
**A Model for Integrating Generative AI into Course Content Development**  

---
Primary Category: cs.CY  
Categories: K-3-2, cs-CY, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2308.12276v2)  

---


**ABSTRACT**  
As Generative AI (GenAI) models continue to gain prominence, a new frontier is emerging in the field of computer science education. Results from initial anonymous surveys reveal that nearly half (48.5%) of our students now turn to GenAI for academic assignments, highlighting its growing role in modern education. With educators facing challenges in creating dynamic and unique course content, the potential of GenAI becomes evident. It offers not only a quicker method for content development but also paves the way for diversified, high-quality educational resources, countering traditional cheating methods and catering to varied student needs. Key questions thus arise: "How can GenAI assist instructors in creating meaningful content and problems quickly, and can it reduce the instructional staff's workload?"   Addressing the first question, we unveil the "GenAI Content Generation Framework". This novel tool equips educators to tap into the prowess of GenAI for course content design. The framework presents a systematic and practical blueprint for generating university-level course material through chat-based GenAI. Drawing from our first-hand experiences, we provide strategic guidance on formulating inquiries and organizing GenAI sessions to elicit quality content that aligns with specific educational goals and context.   Our work stands apart by outlining a specific workflow and offering concrete suggestions for harnessing GenAI in course material development, backed by a strong case for its adoption. Armed with the framework and insights presented in this paper, educators and course content developers can move forward with assurance, tapping into GenAI's vast potential for innovative content creation.

{{</citation>}}


### (98/111) Innovating Computer Programming Pedagogy: The AI-Lab Framework for Generative AI Adoption (Ethan Dickey et al., 2023)

{{<citation>}}

Ethan Dickey, Andres Bejarano, Chirayu Garg. (2023)  
**Innovating Computer Programming Pedagogy: The AI-Lab Framework for Generative AI Adoption**  

---
Primary Category: cs.CY  
Categories: K-3-2, cs-CY, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2308.12258v1)  

---


**ABSTRACT**  
Over the last year, the ascent of Generative AI (GenAI) has raised concerns about its impact on core skill development, such as problem-solving and algorithmic thinking, in Computer Science students. Preliminary anonymous surveys show that at least 48.5% of our students use GenAI for homework. With the proliferation of these tools, the academic community must contemplate the appropriate role of these tools in education. Neglecting this might culminate in a phenomenon we term the "Junior-Year Wall," where students struggle in advanced courses due to prior over-dependence on GenAI. Instead of discouraging GenAI use, which may unintentionally foster covert usage, our research seeks to answer: "How can educators guide students' interactions with GenAI to preserve core skill development during their foundational academic years?"   We introduce "AI-Lab," a pedagogical framework for guiding students in effectively leveraging GenAI within core collegiate programming courses. This framework accentuates GenAI's benefits and potential as a pedagogical instrument. By identifying and rectifying GenAI's errors, students enrich their learning process. Moreover, AI-Lab presents opportunities to use GenAI for tailored support such as topic introductions, detailed examples, corner case identification, rephrased explanations, and debugging assistance. Importantly, the framework highlights the risks of GenAI over-dependence, aiming to intrinsically motivate students towards balanced usage. This approach is premised on the idea that mere warnings of GenAI's potential failures may be misconstrued as instructional shortcomings rather than genuine tool limitations.   Additionally, AI-Lab offers strategies for formulating prompts to elicit high-quality GenAI responses. For educators, AI-Lab provides mechanisms to explore students' perceptions of GenAI's role in their learning experience.

{{</citation>}}


## q-bio.QM (1)



### (99/111) Enhancing cardiovascular risk prediction through AI-enabled calcium-omics (Ammar Hoori et al., 2023)

{{<citation>}}

Ammar Hoori, Sadeer Al-Kindi, Tao Hu, Yingnan Song, Hao Wu, Juhwan Lee, Nour Tashtish, Pingfu Fu, Robert Gilkeson, Sanjay Rajagopalan, David L. Wilson. (2023)  
**Enhancing cardiovascular risk prediction through AI-enabled calcium-omics**  

---
Primary Category: q-bio.QM  
Categories: cs-AI, q-bio-QM, q-bio.QM  
Keywords: AI, Clinical  
[Paper Link](http://arxiv.org/abs/2308.12224v1)  

---


**ABSTRACT**  
Background. Coronary artery calcium (CAC) is a powerful predictor of major adverse cardiovascular events (MACE). Traditional Agatston score simply sums the calcium, albeit in a non-linear way, leaving room for improved calcification assessments that will more fully capture the extent of disease.   Objective. To determine if AI methods using detailed calcification features (i.e., calcium-omics) can improve MACE prediction.   Methods. We investigated additional features of calcification including assessment of mass, volume, density, spatial distribution, territory, etc. We used a Cox model with elastic-net regularization on 2457 CT calcium score (CTCS) enriched for MACE events obtained from a large no-cost CLARIFY program (ClinicalTri-als.gov Identifier: NCT04075162). We employed sampling techniques to enhance model training. We also investigated Cox models with selected features to identify explainable high-risk characteristics.   Results. Our proposed calcium-omics model with modified synthetic down sampling and up sampling gave C-index (80.5%/71.6%) and two-year AUC (82.4%/74.8%) for (80:20, training/testing), respectively (sampling was applied to the training set only). Results compared favorably to Agatston which gave C-index (71.3%/70.3%) and AUC (71.8%/68.8%), respectively. Among calcium-omics features, numbers of calcifications, LAD mass, and diffusivity (a measure of spatial distribution) were important determinants of increased risk, with dense calcification (>1000HU) associated with lower risk. The calcium-omics model reclassified 63% of MACE patients to the high risk group in a held-out test. The categorical net-reclassification index was NRI=0.153.   Conclusions. AI analysis of coronary calcification can lead to improved results as compared to Agatston scoring. Our findings suggest the utility of calcium-omics in improved prediction of risk.

{{</citation>}}


## q-fin.PM (1)



### (100/111) Learning to Learn Financial Networks for Optimising Momentum Strategies (Xingyue Pu et al., 2023)

{{<citation>}}

Xingyue Pu, Stefan Zohren, Stephen Roberts, Xiaowen Dong. (2023)  
**Learning to Learn Financial Networks for Optimising Momentum Strategies**  

---
Primary Category: q-fin.PM  
Categories: cs-AI, cs-LG, q-fin-PM, q-fin-TR, q-fin.PM, stat-ML  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2308.12212v1)  

---


**ABSTRACT**  
Network momentum provides a novel type of risk premium, which exploits the interconnections among assets in a financial network to predict future returns. However, the current process of constructing financial networks relies heavily on expensive databases and financial expertise, limiting accessibility for small-sized and academic institutions. Furthermore, the traditional approach treats network construction and portfolio optimisation as separate tasks, potentially hindering optimal portfolio performance. To address these challenges, we propose L2GMOM, an end-to-end machine learning framework that simultaneously learns financial networks and optimises trading signals for network momentum strategies. The model of L2GMOM is a neural network with a highly interpretable forward propagation architecture, which is derived from algorithm unrolling. The L2GMOM is flexible and can be trained with diverse loss functions for portfolio performance, e.g. the negative Sharpe ratio. Backtesting on 64 continuous future contracts demonstrates a significant improvement in portfolio profitability and risk control, with a Sharpe ratio of 1.74 across a 20-year period.

{{</citation>}}


## cs.HC (2)



### (101/111) Inferring Human Intentions from Predicted Action Probabilities (Lei Shi et al., 2023)

{{<citation>}}

Lei Shi, Paul-Christian Bürkner, Andreas Bulling. (2023)  
**Inferring Human Intentions from Predicted Action Probabilities**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.12194v1)  

---


**ABSTRACT**  
Predicting the next action that a human is most likely to perform is key to human-AI collaboration and has consequently attracted increasing research interests in recent years. An important factor for next action prediction are human intentions: If the AI agent knows the intention it can predict future actions and plan collaboration more effectively. Existing Bayesian methods for this task struggle with complex visual input while deep neural network (DNN) based methods do not provide uncertainty quantifications. In this work we combine both approaches for the first time and show that the predicted next action probabilities contain information that can be used to infer the underlying intention. We propose a two-step approach to human intention prediction: While a DNN predicts the probabilities of the next action, MCMC-based Bayesian inference is used to infer the underlying intention from these predictions. This approach not only allows for independent design of the DNN architecture but also the subsequently fast, design-independent inference of human intentions. We evaluate our method using a series of experiments on the Watch-And-Help (WAH) and a keyboard and mouse interaction dataset. Our results show that our approach can accurately predict human intentions from observed actions and the implicit information contained in next action probabilities. Furthermore, we show that our approach can predict the correct intention even if only few actions have been observed.

{{</citation>}}


### (102/111) Diagnosing Infeasible Optimization Problems Using Large Language Models (Hao Chen et al., 2023)

{{<citation>}}

Hao Chen, Gonzalo E. Constante-Flores, Can Li. (2023)  
**Diagnosing Infeasible Optimization Problems Using Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-HC, cs-LG, cs.HC, math-OC  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.12923v1)  

---


**ABSTRACT**  
Decision-making problems can be represented as mathematical optimization models, finding wide applications in fields such as economics, engineering and manufacturing, transportation, and health care. Optimization models are mathematical abstractions of the problem of making the best decision while satisfying a set of requirements or constraints. One of the primary barriers to deploying these models in practice is the challenge of helping practitioners understand and interpret such models, particularly when they are infeasible, meaning no decision satisfies all the constraints. Existing methods for diagnosing infeasible optimization models often rely on expert systems, necessitating significant background knowledge in optimization. In this paper, we introduce OptiChat, a first-of-its-kind natural language-based system equipped with a chatbot GUI for engaging in interactive conversations about infeasible optimization models. OptiChat can provide natural language descriptions of the optimization model itself, identify potential sources of infeasibility, and offer suggestions to make the model feasible. The implementation of OptiChat is built on GPT-4, which interfaces with an optimization solver to identify the minimal subset of constraints that render the entire optimization problem infeasible, also known as the Irreducible Infeasible Subset (IIS). We utilize few-shot learning, expert chain-of-thought, key-retrieve, and sentiment prompts to enhance OptiChat's reliability. Our experiments demonstrate that OptiChat assists both expert and non-expert users in improving their understanding of the optimization models, enabling them to quickly identify the sources of infeasibility.

{{</citation>}}


## physics.geo-ph (1)



### (103/111) Self-Supervised Knowledge-Driven Deep Learning for 3D Magnetic Inversion (Yinshuo Li et al., 2023)

{{<citation>}}

Yinshuo Li, Zhuo Jia, Wenkai Lu, Cao Song. (2023)  
**Self-Supervised Knowledge-Driven Deep Learning for 3D Magnetic Inversion**  

---
Primary Category: physics.geo-ph  
Categories: cs-LG, physics-geo-ph, physics.geo-ph  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.12193v1)  

---


**ABSTRACT**  
The magnetic inversion method is one of the non-destructive geophysical methods, which aims to estimate the subsurface susceptibility distribution from surface magnetic anomaly data. Recently, supervised deep learning methods have been widely utilized in lots of geophysical fields including magnetic inversion. However, these methods rely heavily on synthetic training data, whose performance is limited since the synthetic data is not independently and identically distributed with the field data. Thus, we proposed to realize magnetic inversion by self-supervised deep learning. The proposed self-supervised knowledge-driven 3D magnetic inversion method (SSKMI) learns on the target field data by a closed loop of the inversion and forward models. Given that the parameters of the forward model are preset, SSKMI can optimize the inversion model by minimizing the mean absolute error between observed and re-estimated surface magnetic anomalies. Besides, there is a knowledge-driven module in the proposed inversion model, which makes the deep learning method more explicable. Meanwhile, comparative experiments demonstrate that the knowledge-driven module can accelerate the training of the proposed method and achieve better results. Since magnetic inversion is an ill-pose task, SSKMI proposed to constrain the inversion model by a guideline in the auxiliary loop. The experimental results demonstrate that the proposed method is a reliable magnetic inversion method with outstanding performance.

{{</citation>}}


## cs.RO (2)



### (104/111) In-Hand Cube Reconfiguration: Simplified (Sumit Patidar et al., 2023)

{{<citation>}}

Sumit Patidar, Adrian Sieler, Oliver Brock. (2023)  
**In-Hand Cube Reconfiguration: Simplified**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.12178v1)  

---


**ABSTRACT**  
We present a simple approach to in-hand cube reconfiguration. By simplifying planning, control, and perception as much as possible, while maintaining robust and general performance, we gain insights into the inherent complexity of in-hand cube reconfiguration. We also demonstrate the effectiveness of combining GOFAI-based planning with the exploitation of environmental constraints and inherently compliant end-effectors in the context of dexterous manipulation. The proposed system outperforms a substantially more complex system for cube reconfiguration based on deep learning and accurate physical simulation, contributing arguments to the discussion about what the most promising approach to general manipulation might be. Project website: https://rbo.gitlab-pages.tu-berlin.de/robotics/simpleIHM/

{{</citation>}}


### (105/111) Identifying Reaction-Aware Driving Styles of Stochastic Model Predictive Controlled Vehicles by Inverse Reinforcement Learning (Ni Dang et al., 2023)

{{<citation>}}

Ni Dang, Tao Shi, Zengjie Zhang, Wanxin Jin, Marion Leibold, Martin Buss. (2023)  
**Identifying Reaction-Aware Driving Styles of Stochastic Model Predictive Controlled Vehicles by Inverse Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.12069v1)  

---


**ABSTRACT**  
The driving style of an Autonomous Vehicle (AV) refers to how it behaves and interacts with other AVs. In a multi-vehicle autonomous driving system, an AV capable of identifying the driving styles of its nearby AVs can reliably evaluate the risk of collisions and make more reasonable driving decisions. However, there has not been a consistent definition of driving styles for an AV in the literature, although it is considered that the driving style is encoded in the AV's trajectories and can be identified using Maximum Entropy Inverse Reinforcement Learning (ME-IRL) methods as a cost function. Nevertheless, an important indicator of the driving style, i.e., how an AV reacts to its nearby AVs, is not fully incorporated in the feature design of previous ME-IRL methods. In this paper, we describe the driving style as a cost function of a series of weighted features. We design additional novel features to capture the AV's reaction-aware characteristics. Then, we identify the driving styles from the demonstration trajectories generated by the Stochastic Model Predictive Control (SMPC) using a modified ME-IRL method with our newly proposed features. The proposed method is validated using MATLAB simulation and an off-the-shelf experiment.

{{</citation>}}


## cs.NI (1)



### (106/111) Fine-grained Spatio-Temporal Distribution Prediction of Mobile Content Delivery in 5G Ultra-Dense Networks (Shaoyuan Huang et al., 2023)

{{<citation>}}

Shaoyuan Huang, Heng Zhang, Xiaofei Wang, Min Chen, Jianxin Li, Victor C. M. Leung. (2023)  
**Fine-grained Spatio-Temporal Distribution Prediction of Mobile Content Delivery in 5G Ultra-Dense Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2308.12322v1)  

---


**ABSTRACT**  
The 5G networks have extensively promoted the growth of mobile users and novel applications, and with the skyrocketing user requests for a large amount of popular content, the consequent content delivery services (CDSs) have been bringing a heavy load to mobile service providers. As a key mission in intelligent networks management, understanding and predicting the distribution of CDSs benefits many tasks of modern network services such as resource provisioning and proactive content caching for content delivery networks. However, the revolutions in novel ubiquitous network architectures led by ultra-dense networks (UDNs) make the task extremely challenging. Specifically, conventional methods face the challenges of insufficient spatio precision, lacking generalizability, and complex multi-feature dependencies of user requests, making their effectiveness unreliable in CDSs prediction under 5G UDNs. In this paper, we propose to adopt a series of encoding and sampling methods to model CDSs of known and unknown areas at a tailored fine-grained level. Moreover, we design a spatio-temporal-social multi-feature extraction framework for CDSs hotspots prediction, in which a novel edge-enhanced graph convolution block is proposed to encode dynamic CDSs networks based on the social relationships and the spatio features. Besides, we introduce the Long-Short Term Memory (LSTM) to further capture the temporal dependency. Extensive performance evaluations with real-world measurement data collected in two mobile content applications demonstrate the effectiveness of our proposed solution, which can improve the prediction area under the curve (AUC) by 40.5% compared to the state-of-the-art proposals at a spatio granularity of 76m, with up to 80% of the unknown areas.

{{</citation>}}


## q-bio.TO (1)



### (107/111) Critical Evaluation of Artificial Intelligence as Digital Twin of Pathologist for Prostate Cancer Pathology (Okyaz Eminaga et al., 2023)

{{<citation>}}

Okyaz Eminaga, Mahmoud Abbas, Christian Kunder, Yuri Tolkach, Ryan Han, James D. Brooks, Rosalie Nolley, Axel Semjonow, Martin Boegemann, Robert West, Jin Long, Richard Fan, Olaf Bettendorf. (2023)  
**Critical Evaluation of Artificial Intelligence as Digital Twin of Pathologist for Prostate Cancer Pathology**  

---
Primary Category: q-bio.TO  
Categories: cs-AI, q-bio-TO, q-bio.TO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.11992v1)  

---


**ABSTRACT**  
Prostate cancer pathology plays a crucial role in clinical management but is time-consuming. Artificial intelligence (AI) shows promise in detecting prostate cancer and grading patterns. We tested an AI-based digital twin of a pathologist, vPatho, on 2,603 histology images of prostate tissue stained with hematoxylin and eosin. We analyzed various factors influencing tumor-grade disagreement between vPatho and six human pathologists. Our results demonstrated that vPatho achieved comparable performance in prostate cancer detection and tumor volume estimation, as reported in the literature. Concordance levels between vPatho and human pathologists were examined. Notably, moderate to substantial agreement was observed in identifying complementary histological features such as ductal, cribriform, nerve, blood vessels, and lymph cell infiltrations. However, concordance in tumor grading showed a decline when applied to prostatectomy specimens (kappa = 0.44) compared to biopsy cores (kappa = 0.70). Adjusting the decision threshold for the secondary Gleason pattern from 5% to 10% improved the concordance level between pathologists and vPatho for tumor grading on prostatectomy specimens (kappa from 0.44 to 0.64). Potential causes of grade discordance included the vertical extent of tumors toward the prostate boundary and the proportions of slides with prostate cancer. Gleason pattern 4 was particularly associated with discordance. Notably, grade discordance with vPatho was not specific to any of the six pathologists involved in routine clinical grading. In conclusion, our study highlights the potential utility of AI in developing a digital twin of a pathologist. This approach can help uncover limitations in AI adoption and the current grading system for prostate cancer pathology.

{{</citation>}}


## cs.DC (1)



### (108/111) Federated Semi-Supervised and Semi-Asynchronous Learning for Anomaly Detection in IoT Networks (Wenbin Zhai et al., 2023)

{{<citation>}}

Wenbin Zhai, Feng Wang, Liang Liu, Youwei Ding, Wanying Lu. (2023)  
**Federated Semi-Supervised and Semi-Asynchronous Learning for Anomaly Detection in IoT Networks**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Anomaly Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.11981v1)  

---


**ABSTRACT**  
Existing FL-based approaches are based on the unrealistic assumption that the data on the client-side is fully annotated with ground truths. Furthermore, it is a great challenge how to improve the training efficiency while ensuring the detection accuracy in the highly heterogeneous and resource-constrained IoT networks. Meanwhile, the communication cost between clients and the server is also a problem that can not be ignored. Therefore, in this paper, we propose a Federated Semi-Supervised and Semi-Asynchronous (FedS3A) learning for anomaly detection in IoT networks. First, we consider a more realistic assumption that labeled data is only available at the server, and pseudo-labeling is utilized to implement federated semi-supervised learning, in which a dynamic weight of supervised learning is exploited to balance the supervised learning at the server and unsupervised learning at clients. Then, we propose a semi-asynchronous model update and staleness tolerant distribution scheme to achieve a trade-off between the round efficiency and detection accuracy. Meanwhile, the staleness of local models and the participation frequency of clients are considered to adjust their contributions to the global model. In addition, a group-based aggregation function is proposed to deal with the non-IID distribution of the data. Finally, the difference transmission based on the sparse matrix is adopted to reduce the communication cost. Extensive experimental results show that FedS3A can achieve greater than 98% accuracy even when the data is non-IID and is superior to the classic FL-based algorithms in terms of both detection performance and round efficiency, achieving a win-win situation. Meanwhile, FedS3A successfully reduces the communication cost by higher than 50%.

{{</citation>}}


## eess.AS (1)



### (109/111) Joint Prediction of Audio Event and Annoyance Rating in an Urban Soundscape by Hierarchical Graph Representation Learning (Yuanbo Hou et al., 2023)

{{<citation>}}

Yuanbo Hou, Siyang Song, Cheng Luo, Andrew Mitchell, Qiaoqiao Ren, Weicheng Xie, Jian Kang, Wenwu Wang, Dick Botteldooren. (2023)  
**Joint Prediction of Audio Event and Annoyance Rating in an Urban Soundscape by Hierarchical Graph Representation Learning**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.11980v1)  

---


**ABSTRACT**  
Sound events in daily life carry rich information about the objective world. The composition of these sounds affects the mood of people in a soundscape. Most previous approaches only focus on classifying and detecting audio events and scenes, but may ignore their perceptual quality that may impact humans' listening mood for the environment, e.g. annoyance. To this end, this paper proposes a novel hierarchical graph representation learning (HGRL) approach which links objective audio events (AE) with subjective annoyance ratings (AR) of the soundscape perceived by humans. The hierarchical graph consists of fine-grained event (fAE) embeddings with single-class event semantics, coarse-grained event (cAE) embeddings with multi-class event semantics, and AR embeddings. Experiments show the proposed HGRL successfully integrates AE with AR for AEC and ARP tasks, while coordinating the relations between cAE and fAE and further aligning the two different grains of AE information with the AR.

{{</citation>}}


## cs.SD (1)



### (110/111) CED: Consistent ensemble distillation for audio tagging (Heinrich Dinkel et al., 2023)

{{<citation>}}

Heinrich Dinkel, Yongqing Wang, Zhiyong Yan, Junbo Zhang, Yujun Wang. (2023)  
**CED: Consistent ensemble distillation for audio tagging**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.11957v1)  

---


**ABSTRACT**  
Augmentation and knowledge distillation (KD) are well-established techniques employed in the realm of audio classification tasks, aimed at enhancing performance and reducing model sizes on the widely recognized Audioset (AS) benchmark. Although both techniques are effective individually, their combined use, called consistent teaching, hasn't been explored before. This paper proposes CED, a simple training framework that distils student models from large teacher ensembles with consistent teaching. To achieve this, CED efficiently stores logits as well as the augmentation methods on disk, making it scalable to large-scale datasets. Central to CED's efficacy is its label-free nature, meaning that only the stored logits are used for the optimization of a student model only requiring 0.3\% additional disk space for AS. The study trains various transformer-based models, including a 10M parameter model achieving a 49.0 mean average precision (mAP) on AS. Pretrained models and code are available at https://github.com/RicherMans/CED.

{{</citation>}}


## cs.MA (1)



### (111/111) ${\rm E}(3)$-Equivariant Actor-Critic Methods for Cooperative Multi-Agent Reinforcement Learning (Dingyang Chen et al., 2023)

{{<citation>}}

Dingyang Chen, Qi Zhang. (2023)  
**${\rm E}(3)$-Equivariant Actor-Critic Methods for Cooperative Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-LG, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.11842v1)  

---


**ABSTRACT**  
Identification and analysis of symmetrical patterns in the natural world have led to significant discoveries across various scientific fields, such as the formulation of gravitational laws in physics and advancements in the study of chemical structures. In this paper, we focus on exploiting Euclidean symmetries inherent in certain cooperative multi-agent reinforcement learning (MARL) problems and prevalent in many applications. We begin by formally characterizing a subclass of Markov games with a general notion of symmetries that admits the existence of symmetric optimal values and policies. Motivated by these properties, we design neural network architectures with symmetric constraints embedded as an inductive bias for multi-agent actor-critic methods. This inductive bias results in superior performance in various cooperative MARL benchmarks and impressive generalization capabilities such as zero-shot learning and transfer learning in unseen scenarios with repeated symmetric patterns. The code is available at: https://github.com/dchen48/E3AC.

{{</citation>}}
