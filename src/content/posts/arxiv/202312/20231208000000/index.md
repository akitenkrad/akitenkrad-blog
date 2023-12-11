---
draft: false
title: "arXiv @ 2023.12.08"
date: 2023-12-08
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.08"
    identifier: arxiv_20231208
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (33)](#cscv-33)
- [cs.LG (23)](#cslg-23)
- [cs.HC (1)](#cshc-1)
- [cs.CL (18)](#cscl-18)
- [cs.CY (1)](#cscy-1)
- [cs.RO (4)](#csro-4)
- [cs.CR (7)](#cscr-7)
- [physics.data-an (1)](#physicsdata-an-1)
- [cs.OS (1)](#csos-1)
- [eess.AS (2)](#eessas-2)
- [cs.AI (3)](#csai-3)
- [eess.IV (2)](#eessiv-2)
- [cs.CE (1)](#csce-1)
- [stat.ML (2)](#statml-2)
- [stat.ME (1)](#statme-1)
- [cs.AR (1)](#csar-1)
- [quant-ph (1)](#quant-ph-1)
- [cs.IR (1)](#csir-1)
- [cs.SD (1)](#cssd-1)
- [cond-mat.soft (1)](#cond-matsoft-1)
- [cs.SI (2)](#cssi-2)
- [cs.DC (1)](#csdc-1)

## cs.CV (33)



### (1/108) A Layer-Wise Tokens-to-Token Transformer Network for Improved Historical Document Image Enhancement (Risab Biswas et al., 2023)

{{<citation>}}

Risab Biswas, Swalpa Kumar Roy, Umapada Pal. (2023)  
**A Layer-Wise Tokens-to-Token Transformer Network for Improved Historical Document Image Enhancement**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.03946v1)  

---


**ABSTRACT**  
Document image enhancement is a fundamental and important stage for attaining the best performance in any document analysis assignment because there are many degradation situations that could harm document images, making it more difficult to recognize and analyze them. In this paper, we propose \textbf{T2T-BinFormer} which is a novel document binarization encoder-decoder architecture based on a Tokens-to-token vision transformer. Each image is divided into a set of tokens with a defined length using the ViT model, which is then applied several times to model the global relationship between the tokens. However, the conventional tokenization of input data does not adequately reflect the crucial local structure between adjacent pixels of the input image, which results in low efficiency. Instead of using a simple ViT and hard splitting of images for the document image enhancement task, we employed a progressive tokenization technique to capture this local information from an image to achieve more effective results. Experiments on various DIBCO and H-DIBCO benchmarks demonstrate that the proposed model outperforms the existing CNN and ViT-based state-of-the-art methods. In this research, the primary area of examination is the application of the proposed architecture to the task of document binarization. The source code will be made available at https://github.com/RisabBiswas/T2T-BinFormer.

{{</citation>}}


### (2/108) The Potential of Vision-Language Models for Content Moderation of Children's Videos (Syed Hammad Ahmed et al., 2023)

{{<citation>}}

Syed Hammad Ahmed, Shengnan Hu, Gita Sukthankar. (2023)  
**The Potential of Vision-Language Models for Content Moderation of Children's Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-CY, cs-LG, cs-SI, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.03936v1)  

---


**ABSTRACT**  
Natural language supervision has been shown to be effective for zero-shot learning in many computer vision tasks, such as object detection and activity recognition. However, generating informative prompts can be challenging for more subtle tasks, such as video content moderation. This can be difficult, as there are many reasons why a video might be inappropriate, beyond violence and obscenity. For example, scammers may attempt to create junk content that is similar to popular educational videos but with no meaningful information. This paper evaluates the performance of several CLIP variations for content moderation of children's cartoons in both the supervised and zero-shot setting. We show that our proposed model (Vanilla CLIP with Projection Layer) outperforms previous work conducted on the Malicious or Benign (MOB) benchmark for video content moderation. This paper presents an in depth analysis of how context-specific language prompts affect content moderation performance. Our results indicate that it is important to include more context in content moderation prompts, particularly for cartoon videos as they are not well represented in the CLIP training data.

{{</citation>}}


### (3/108) Skeleton-in-Context: Unified Skeleton Sequence Modeling with In-Context Learning (Xinshun Wang et al., 2023)

{{<citation>}}

Xinshun Wang, Zhongbin Fang, Xia Li, Xiangtai Li, Chen Chen, Mengyuan Liu. (2023)  
**Skeleton-in-Context: Unified Skeleton Sequence Modeling with In-Context Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.03703v1)  

---


**ABSTRACT**  
In-context learning provides a new perspective for multi-task modeling for vision and NLP. Under this setting, the model can perceive tasks from prompts and accomplish them without any extra task-specific head predictions or model fine-tuning. However, Skeleton sequence modeling via in-context learning remains unexplored. Directly applying existing in-context models from other areas onto skeleton sequences fails due to the inter-frame and cross-task pose similarity that makes it outstandingly hard to perceive the task correctly from a subtle context. To address this challenge, we propose Skeleton-in-Context (SiC), an effective framework for in-context skeleton sequence modeling. Our SiC is able to handle multiple skeleton-based tasks simultaneously after a single training process and accomplish each task from context according to the given prompt. It can further generalize to new, unseen tasks according to customized prompts. To facilitate context perception, we additionally propose a task-unified prompt, which adaptively learns tasks of different natures, such as partial joint-level generation, sequence-level prediction, or 2D-to-3D motion prediction. We conduct extensive experiments to evaluate the effectiveness of our SiC on multiple tasks, including motion prediction, pose estimation, joint completion, and future pose estimation. We also evaluate its generalization capability on unseen tasks such as motion-in-between. These experiments show that our model achieves state-of-the-art multi-task performance and even outperforms single-task methods on certain tasks.

{{</citation>}}


### (4/108) Self-conditioned Image Generation via Generating Representations (Tianhong Li et al., 2023)

{{<citation>}}

Tianhong Li, Dina Katabi, Kaiming He. (2023)  
**Self-conditioned Image Generation via Generating Representations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.03701v2)  

---


**ABSTRACT**  
This paper presents $\textbf{R}$epresentation-$\textbf{C}$onditioned image $\textbf{G}$eneration (RCG), a simple yet effective image generation framework which sets a new benchmark in class-unconditional image generation. RCG does not condition on any human annotations. Instead, it conditions on a self-supervised representation distribution which is mapped from the image distribution using a pre-trained encoder. During generation, RCG samples from such representation distribution using a representation diffusion model (RDM), and employs a pixel generator to craft image pixels conditioned on the sampled representation. Such a design provides substantial guidance during the generative process, resulting in high-quality image generation. Tested on ImageNet 256$\times$256, RCG achieves a Frechet Inception Distance (FID) of 3.31 and an Inception Score (IS) of 253.4. These results not only significantly improve the state-of-the-art of class-unconditional image generation but also rival the current leading methods in class-conditional image generation, bridging the long-standing performance gap between these two tasks. Code is available at https://github.com/LTH14/rcg.

{{</citation>}}


### (5/108) Reason2Drive: Towards Interpretable and Chain-based Reasoning for Autonomous Driving (Ming Nie et al., 2023)

{{<citation>}}

Ming Nie, Renyuan Peng, Chunwei Wang, Xinyue Cai, Jianhua Han, Hang Xu, Li Zhang. (2023)  
**Reason2Drive: Towards Interpretable and Chain-based Reasoning for Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BLEU, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.03661v1)  

---


**ABSTRACT**  
Large vision-language models (VLMs) have garnered increasing interest in autonomous driving areas, due to their advanced capabilities in complex reasoning tasks essential for highly autonomous vehicle behavior. Despite their potential, research in autonomous systems is hindered by the lack of datasets with annotated reasoning chains that explain the decision-making processes in driving. To bridge this gap, we present Reason2Drive, a benchmark dataset with over 600K video-text pairs, aimed at facilitating the study of interpretable reasoning in complex driving environments. We distinctly characterize the autonomous driving process as a sequential combination of perception, prediction, and reasoning steps, and the question-answer pairs are automatically collected from a diverse range of open-source outdoor driving datasets, including nuScenes, Waymo and ONCE. Moreover, we introduce a novel aggregated evaluation metric to assess chain-based reasoning performance in autonomous systems, addressing the semantic ambiguities of existing metrics such as BLEU and CIDEr. Based on the proposed benchmark, we conduct experiments to assess various existing VLMs, revealing insights into their reasoning capabilities. Additionally, we develop an efficient approach to empower VLMs to leverage object-level perceptual elements in both feature extraction and prediction, further enhancing their reasoning accuracy. The code and dataset will be released.

{{</citation>}}


### (6/108) MOCHa: Multi-Objective Reinforcement Mitigating Caption Hallucinations (Assaf Ben-Kish et al., 2023)

{{<citation>}}

Assaf Ben-Kish, Moran Yanuka, Morris Alper, Raja Giryes, Hadar Averbuch-Elor. (2023)  
**MOCHa: Multi-Objective Reinforcement Mitigating Caption Hallucinations**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03631v1)  

---


**ABSTRACT**  
While recent years have seen rapid progress in image-conditioned text generation, image captioning still suffers from the fundamental issue of hallucinations, the generation of spurious details that cannot be inferred from the given image. Dedicated methods for reducing hallucinations in image captioning largely focus on closed-vocabulary object tokens, ignoring most types of hallucinations that occur in practice. In this work, we propose MOCHa, an approach that harnesses advancements in reinforcement learning (RL) to address the sequence-level nature of hallucinations in an open-world setup. To optimize for caption fidelity to the input image, we leverage ground-truth reference captions as proxies to measure the logical consistency of generated captions. However, optimizing for caption fidelity alone fails to preserve the semantic adequacy of generations; therefore, we propose a multi-objective reward function that jointly targets these qualities, without requiring any strong supervision. We demonstrate that these goals can be simultaneously optimized with our framework, enhancing performance for various captioning models of different scales. Our qualitative and quantitative results demonstrate MOCHa's superior performance across various established metrics. We also demonstrate the benefit of our method in the open-vocabulary setting. To this end, we contribute OpenCHAIR, a new benchmark for quantifying open-vocabulary hallucinations in image captioning models, constructed using generative foundation models. We will release our code, benchmark, and trained models.

{{</citation>}}


### (7/108) Language-Informed Visual Concept Learning (Sharon Lee et al., 2023)

{{<citation>}}

Sharon Lee, Yunzhi Zhang, Shangzhe Wu, Jiajun Wu. (2023)  
**Language-Informed Visual Concept Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.03587v1)  

---


**ABSTRACT**  
Our understanding of the visual world is centered around various concept axes, characterizing different aspects of visual entities. While different concept axes can be easily specified by language, e.g. color, the exact visual nuances along each axis often exceed the limitations of linguistic articulations, e.g. a particular style of painting. In this work, our goal is to learn a language-informed visual concept representation, by simply distilling large pre-trained vision-language models. Specifically, we train a set of concept encoders to encode the information pertinent to a set of language-informed concept axes, with an objective of reproducing the input image through a pre-trained Text-to-Image (T2I) model. To encourage better disentanglement of different concept encoders, we anchor the concept embeddings to a set of text embeddings obtained from a pre-trained Visual Question Answering (VQA) model. At inference time, the model extracts concept embeddings along various axes from new test images, which can be remixed to generate images with novel compositions of visual concepts. With a lightweight test-time finetuning procedure, it can also generalize to novel concepts unseen at training.

{{</citation>}}


### (8/108) Foundation Model Assisted Weakly Supervised Semantic Segmentation (Xiaobo Yang et al., 2023)

{{<citation>}}

Xiaobo Yang, Xiaojin Gong. (2023)  
**Foundation Model Assisted Weakly Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.03585v1)  

---


**ABSTRACT**  
This work aims to leverage pre-trained foundation models, such as contrastive language-image pre-training (CLIP) and segment anything model (SAM), to address weakly supervised semantic segmentation (WSSS) using image-level labels. To this end, we propose a coarse-to-fine framework based on CLIP and SAM for generating high-quality segmentation seeds. Specifically, we construct an image classification task and a seed segmentation task, which are jointly performed by CLIP with frozen weights and two sets of learnable task-specific prompts. A SAM-based seeding (SAMS) module is designed and applied to each task to produce either coarse or fine seed maps. Moreover, we design a multi-label contrastive loss supervised by image-level labels and a CAM activation loss supervised by the generated coarse seed map. These losses are used to learn the prompts, which are the only parts need to be learned in our framework. Once the prompts are learned, we input each image along with the learned segmentation-specific prompts into CLIP and the SAMS module to produce high-quality segmentation seeds. These seeds serve as pseudo labels to train an off-the-shelf segmentation network like other two-stage WSSS methods. Experiments show that our method achieves the state-of-the-art performance on PASCAL VOC 2012 and competitive results on MS COCO 2014.

{{</citation>}}


### (9/108) DocBinFormer: A Two-Level Transformer Network for Effective Document Image Binarization (Risab Biswas et al., 2023)

{{<citation>}}

Risab Biswas, Swalpa Kumar Roy, Ning Wang, Umapada Pal, Guang-Bin Huang. (2023)  
**DocBinFormer: A Two-Level Transformer Network for Effective Document Image Binarization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.03568v1)  

---


**ABSTRACT**  
In real life, various degradation scenarios exist that might damage document images, making it harder to recognize and analyze them, thus binarization is a fundamental and crucial step for achieving the most optimal performance in any document analysis task. We propose DocBinFormer (Document Binarization Transformer), a novel two-level vision transformer (TL-ViT) architecture based on vision transformers for effective document image binarization. The presented architecture employs a two-level transformer encoder to effectively capture both global and local feature representation from the input images. These complimentary bi-level features are exploited for efficient document image binarization, resulting in improved results for system-generated as well as handwritten document images in a comprehensive approach. With the absence of convolutional layers, the transformer encoder uses the pixel patches and sub-patches along with their positional information to operate directly on them, while the decoder generates a clean (binarized) output image from the latent representation of the patches. Instead of using a simple vision transformer block to extract information from the image patches, the proposed architecture uses two transformer blocks for greater coverage of the extracted feature space on a global and local scale. The encoded feature representation is used by the decoder block to generate the corresponding binarized output. Extensive experiments on a variety of DIBCO and H-DIBCO benchmarks show that the proposed model outperforms state-of-the-art techniques on four metrics. The source code will be made available at https://github.com/RisabBiswas/DocBinFormer.

{{</citation>}}


### (10/108) SYNC-CLIP: Synthetic Data Make CLIP Generalize Better in Data-Limited Scenarios (Mushui Liu et al., 2023)

{{<citation>}}

Mushui Liu, Weijie He, Ziqian Lu, Yunlong Yu. (2023)  
**SYNC-CLIP: Synthetic Data Make CLIP Generalize Better in Data-Limited Scenarios**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.03805v1)  

---


**ABSTRACT**  
Prompt learning is a powerful technique for transferring Vision-Language Models (VLMs) such as CLIP to downstream tasks. However, the prompt-based methods that are fine-tuned solely with base classes may struggle to generalize to novel classes in open-vocabulary scenarios, especially when data are limited. To address this issue, we propose an innovative approach called SYNC-CLIP that leverages SYNthetiC data for enhancing the generalization capability of CLIP. Based on the observation of the distribution shift between the real and synthetic samples, we treat real and synthetic samples as distinct domains and propose to optimize separate domain prompts to capture domain-specific information, along with the shared visual prompts to preserve the semantic consistency between two domains. By aligning the cross-domain features, the synthetic data from novel classes can provide implicit guidance to rebalance the decision boundaries. Experimental results on three model generalization tasks demonstrate that our method performs very competitively across various benchmarks. Notably, SYNC-CLIP outperforms the state-of-the-art competitor PromptSRC by an average improvement of 3.0% on novel classes across 11 datasets in open-vocabulary scenarios.

{{</citation>}}


### (11/108) Enhancing Kinship Verification through Multiscale Retinex and Combined Deep-Shallow features (El Ouanas Belabbaci et al., 2023)

{{<citation>}}

El Ouanas Belabbaci, Mohammed Khammari, Ammar Chouchane, Mohcene Bessaoudi, Abdelmalik Ouamane, Yassine Himeur, Shadi Atalla, Wathiq Mansoor. (2023)  
**Enhancing Kinship Verification through Multiscale Retinex and Combined Deep-Shallow features**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2312.03562v1)  

---


**ABSTRACT**  
The challenge of kinship verification from facial images represents a cutting-edge and formidable frontier in the realms of pattern recognition and computer vision. This area of study holds a myriad of potential applications, spanning from image annotation and forensic analysis to social media research. Our research stands out by integrating a preprocessing method named Multiscale Retinex (MSR), which elevates image quality and amplifies contrast, ultimately bolstering the end results. Strategically, our methodology capitalizes on the harmonious blend of deep and shallow texture descriptors, merging them proficiently at the score level through the Logistic Regression (LR) method. To elucidate, we employ the Local Phase Quantization (LPQ) descriptor to extract shallow texture characteristics. For deep feature extraction, we turn to the prowess of the VGG16 model, which is pre-trained on a convolutional neural network (CNN). The robustness and efficacy of our method have been put to the test through meticulous experiments on three rigorous kinship datasets, namely: Cornell Kin Face, UB Kin Face, and TS Kin Face.

{{</citation>}}


### (12/108) When an Image is Worth 1,024 x 1,024 Words: A Case Study in Computational Pathology (Wenhui Wang et al., 2023)

{{<citation>}}

Wenhui Wang, Shuming Ma, Hanwen Xu, Naoto Usuyama, Jiayu Ding, Hoifung Poon, Furu Wei. (2023)  
**When an Image is Worth 1,024 x 1,024 Words: A Case Study in Computational Pathology**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.03558v1)  

---


**ABSTRACT**  
This technical report presents LongViT, a vision Transformer that can process gigapixel images in an end-to-end manner. Specifically, we split the gigapixel image into a sequence of millions of patches and project them linearly into embeddings. LongNet is then employed to model the extremely long sequence, generating representations that capture both short-range and long-range dependencies. The linear computation complexity of LongNet, along with its distributed algorithm, enables us to overcome the constraints of both computation and memory. We apply LongViT in the field of computational pathology, aiming for cancer diagnosis and prognosis within gigapixel whole-slide images. Experimental results demonstrate that LongViT effectively encodes gigapixel images and outperforms previous state-of-the-art methods on cancer subtyping and survival prediction. Code and models will be available at https://aka.ms/LongViT.

{{</citation>}}


### (13/108) Personalized Face Inpainting with Diffusion Models by Parallel Visual Attention (Jianjin Xu et al., 2023)

{{<citation>}}

Jianjin Xu, Saman Motamed, Praneetha Vaddamanu, Chen Henry Wu, Christian Haene, Jean-Charles Bazin, Fernando de la Torre. (2023)  
**Personalized Face Inpainting with Diffusion Models by Parallel Visual Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.03556v1)  

---


**ABSTRACT**  
Face inpainting is important in various applications, such as photo restoration, image editing, and virtual reality. Despite the significant advances in face generative models, ensuring that a person's unique facial identity is maintained during the inpainting process is still an elusive goal. Current state-of-the-art techniques, exemplified by MyStyle, necessitate resource-intensive fine-tuning and a substantial number of images for each new identity. Furthermore, existing methods often fall short in accommodating user-specified semantic attributes, such as beard or expression. To improve inpainting results, and reduce the computational complexity during inference, this paper proposes the use of Parallel Visual Attention (PVA) in conjunction with diffusion models. Specifically, we insert parallel attention matrices to each cross-attention module in the denoising network, which attends to features extracted from reference images by an identity encoder. We train the added attention modules and identity encoder on CelebAHQ-IDI, a dataset proposed for identity-preserving face inpainting. Experiments demonstrate that PVA attains unparalleled identity resemblance in both face inpainting and face inpainting with language guidance tasks, in comparison to various benchmarks, including MyStyle, Paint by Example, and Custom Diffusion. Our findings reveal that PVA ensures good identity preservation while offering effective language-controllability. Additionally, in contrast to Custom Diffusion, PVA requires just 40 fine-tuning steps for each new identity, which translates to a significant speed increase of over 20 times.

{{</citation>}}


### (14/108) How Low Can You Go? Surfacing Prototypical In-Distribution Samples for Unsupervised Anomaly Detection (Felix Meissen et al., 2023)

{{<citation>}}

Felix Meissen, Johannes Getzner, Alexander Ziller, Georgios Kaissis, Daniel Rueckert. (2023)  
**How Low Can You Go? Surfacing Prototypical In-Distribution Samples for Unsupervised Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.03804v1)  

---


**ABSTRACT**  
Unsupervised anomaly detection (UAD) alleviates large labeling efforts by training exclusively on unlabeled in-distribution data and detecting outliers as anomalies. Generally, the assumption prevails that large training datasets allow the training of higher-performing UAD models. However, in this work, we show that using only very few training samples can already match - and in some cases even improve - anomaly detection compared to training with the whole training dataset. We propose three methods to identify prototypical samples from a large dataset of in-distribution samples. We demonstrate that by training with a subset of just ten such samples, we achieve an area under the receiver operating characteristics curve (AUROC) of $96.37 \%$ on CIFAR10, $92.59 \%$ on CIFAR100, $95.37 \%$ on MNIST, $95.38 \%$ on Fashion-MNIST, $96.37 \%$ on MVTec-AD, $98.81 \%$ on BraTS, and $81.95 \%$ on RSNA pneumonia detection, even exceeding the performance of full training in $25/67$ classes we tested. Additionally, we show that the prototypical in-distribution samples identified by our proposed methods translate well to different models and other datasets and that using their characteristics as guidance allows for successful manual selection of small subsets of high-performing samples. Our code is available at https://anonymous.4open.science/r/uad_prototypical_samples/

{{</citation>}}


### (15/108) Texture-Semantic Collaboration Network for ORSI Salient Object Detection (Gongyang Li et al., 2023)

{{<citation>}}

Gongyang Li, Zhen Bai, Zhi Liu. (2023)  
**Texture-Semantic Collaboration Network for ORSI Salient Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.03548v1)  

---


**ABSTRACT**  
Salient object detection (SOD) in optical remote sensing images (ORSIs) has become increasingly popular recently. Due to the characteristics of ORSIs, ORSI-SOD is full of challenges, such as multiple objects, small objects, low illuminations, and irregular shapes. To address these challenges, we propose a concise yet effective Texture-Semantic Collaboration Network (TSCNet) to explore the collaboration of texture cues and semantic cues for ORSI-SOD. Specifically, TSCNet is based on the generic encoder-decoder structure. In addition to the encoder and decoder, TSCNet includes a vital Texture-Semantic Collaboration Module (TSCM), which performs valuable feature modulation and interaction on basic features extracted from the encoder. The main idea of our TSCM is to make full use of the texture features at the lowest level and the semantic features at the highest level to achieve the expression enhancement of salient regions on features. In the TSCM, we first enhance the position of potential salient regions using semantic features. Then, we render and restore the object details using the texture features. Meanwhile, we also perceive regions of various scales, and construct interactions between different regions. Thanks to the perfect combination of TSCM and generic structure, our TSCNet can take care of both the position and details of salient objects, effectively handling various scenes. Extensive experiments on three datasets demonstrate that our TSCNet achieves competitive performance compared to 14 state-of-the-art methods. The code and results of our method are available at https://github.com/MathLee/TSCNet.

{{</citation>}}


### (16/108) GPT-4 Enhanced Multimodal Grounding for Autonomous Driving: Leveraging Cross-Modal Attention with Large Language Models (Haicheng Liao et al., 2023)

{{<citation>}}

Haicheng Liao, Huanming Shen, Zhenning Li, Chengyue Wang, Guofa Li, Yiming Bie, Chengzhong Xu. (2023)  
**GPT-4 Enhanced Multimodal Grounding for Autonomous Driving: Leveraging Cross-Modal Attention with Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.03543v1)  

---


**ABSTRACT**  
In the field of autonomous vehicles (AVs), accurately discerning commander intent and executing linguistic commands within a visual context presents a significant challenge. This paper introduces a sophisticated encoder-decoder framework, developed to address visual grounding in AVs.Our Context-Aware Visual Grounding (CAVG) model is an advanced system that integrates five core encoders-Text, Image, Context, and Cross-Modal-with a Multimodal decoder. This integration enables the CAVG model to adeptly capture contextual semantics and to learn human emotional features, augmented by state-of-the-art Large Language Models (LLMs) including GPT-4. The architecture of CAVG is reinforced by the implementation of multi-head cross-modal attention mechanisms and a Region-Specific Dynamic (RSD) layer for attention modulation. This architectural design enables the model to efficiently process and interpret a range of cross-modal inputs, yielding a comprehensive understanding of the correlation between verbal commands and corresponding visual scenes. Empirical evaluations on the Talk2Car dataset, a real-world benchmark, demonstrate that CAVG establishes new standards in prediction accuracy and operational efficiency. Notably, the model exhibits exceptional performance even with limited training data, ranging from 50% to 75% of the full dataset. This feature highlights its effectiveness and potential for deployment in practical AV applications. Moreover, CAVG has shown remarkable robustness and adaptability in challenging scenarios, including long-text command interpretation, low-light conditions, ambiguous command contexts, inclement weather conditions, and densely populated urban environments. The code for the proposed model is available at our Github.

{{</citation>}}


### (17/108) Low-shot Object Learning with Mutual Exclusivity Bias (Anh Thai et al., 2023)

{{<citation>}}

Anh Thai, Ahmad Humayun, Stefan Stojanov, Zixuan Huang, Bikram Boote, James M. Rehg. (2023)  
**Low-shot Object Learning with Mutual Exclusivity Bias**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.03533v1)  

---


**ABSTRACT**  
This paper introduces Low-shot Object Learning with Mutual Exclusivity Bias (LSME), the first computational framing of mutual exclusivity bias, a phenomenon commonly observed in infants during word learning. We provide a novel dataset, comprehensive baselines, and a state-of-the-art method to enable the ML community to tackle this challenging learning task. The goal of LSME is to analyze an RGB image of a scene containing multiple objects and correctly associate a previously-unknown object instance with a provided category label. This association is then used to perform low-shot learning to test category generalization. We provide a data generation pipeline for the LSME problem and conduct a thorough analysis of the factors that contribute to its difficulty. Additionally, we evaluate the performance of multiple baselines, including state-of-the-art foundation models. Finally, we present a baseline approach that outperforms state-of-the-art models in terms of low-shot accuracy.

{{</citation>}}


### (18/108) On the Diversity and Realism of Distilled Dataset: An Efficient Dataset Distillation Paradigm (Peng Sun et al., 2023)

{{<citation>}}

Peng Sun, Bei Shi, Daiwei Yu, Tao Lin. (2023)  
**On the Diversity and Realism of Distilled Dataset: An Efficient Dataset Distillation Paradigm**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.03526v1)  

---


**ABSTRACT**  
Contemporary machine learning requires training large neural networks on massive datasets and thus faces the challenges of high computational demands. Dataset distillation, as a recent emerging strategy, aims to compress real-world datasets for efficient training. However, this line of research currently struggle with large-scale and high-resolution datasets, hindering its practicality and feasibility. To this end, we re-examine the existing dataset distillation methods and identify three properties required for large-scale real-world applications, namely, realism, diversity, and efficiency. As a remedy, we propose RDED, a novel computationally-efficient yet effective data distillation paradigm, to enable both diversity and realism of the distilled data. Extensive empirical results over various neural architectures and datasets demonstrate the advancement of RDED: we can distill the full ImageNet-1K to a small dataset comprising 10 images per class within 7 minutes, achieving a notable 42% top-1 accuracy with ResNet-18 on a single RTX-4090 GPU (while the SOTA only achieves 21% but requires 6 hours).

{{</citation>}}


### (19/108) Defense Against Adversarial Attacks using Convolutional Auto-Encoders (Shreyasi Mandal, 2023)

{{<citation>}}

Shreyasi Mandal. (2023)  
**Defense Against Adversarial Attacks using Convolutional Auto-Encoders**  

---
Primary Category: cs.CV  
Categories: I-4-5; I-5-1; I-5-4, cs-AI, cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2312.03520v1)  

---


**ABSTRACT**  
Deep learning models, while achieving state-of-the-art performance on many tasks, are susceptible to adversarial attacks that exploit inherent vulnerabilities in their architectures. Adversarial attacks manipulate the input data with imperceptible perturbations, causing the model to misclassify the data or produce erroneous outputs. This work is based on enhancing the robustness of targeted classifier models against adversarial attacks. To achieve this, an convolutional autoencoder-based approach is employed that effectively counters adversarial perturbations introduced to the input images. By generating images closely resembling the input images, the proposed methodology aims to restore the model's accuracy.

{{</citation>}}


### (20/108) AnimateZero: Video Diffusion Models are Zero-Shot Image Animators (Jiwen Yu et al., 2023)

{{<citation>}}

Jiwen Yu, Xiaodong Cun, Chenyang Qi, Yong Zhang, Xintao Wang, Ying Shan, Jian Zhang. (2023)  
**AnimateZero: Video Diffusion Models are Zero-Shot Image Animators**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.03793v1)  

---


**ABSTRACT**  
Large-scale text-to-video (T2V) diffusion models have great progress in recent years in terms of visual quality, motion and temporal consistency. However, the generation process is still a black box, where all attributes (e.g., appearance, motion) are learned and generated jointly without precise control ability other than rough text descriptions. Inspired by image animation which decouples the video as one specific appearance with the corresponding motion, we propose AnimateZero to unveil the pre-trained text-to-video diffusion model, i.e., AnimateDiff, and provide more precise appearance and motion control abilities for it. For appearance control, we borrow intermediate latents and their features from the text-to-image (T2I) generation for ensuring the generated first frame is equal to the given generated image. For temporal control, we replace the global temporal attention of the original T2V model with our proposed positional-corrected window attention to ensure other frames align with the first frame well. Empowered by the proposed methods, AnimateZero can successfully control the generating progress without further training. As a zero-shot image animator for given images, AnimateZero also enables multiple new applications, including interactive video generation and real image animation. The detailed experiments demonstrate the effectiveness of the proposed method in both T2V and related applications.

{{</citation>}}


### (21/108) Memory-Efficient Optical Flow via Radius-Distribution Orthogonal Cost Volume (Gangwei Xu et al., 2023)

{{<citation>}}

Gangwei Xu, Shujun Chen, Hao Jia, Miaojie Feng, Xin Yang. (2023)  
**Memory-Efficient Optical Flow via Radius-Distribution Orthogonal Cost Volume**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.03790v1)  

---


**ABSTRACT**  
The full 4D cost volume in Recurrent All-Pairs Field Transforms (RAFT) or global matching by Transformer achieves impressive performance for optical flow estimation. However, their memory consumption increases quadratically with input resolution, rendering them impractical for high-resolution images. In this paper, we present MeFlow, a novel memory-efficient method for high-resolution optical flow estimation. The key of MeFlow is a recurrent local orthogonal cost volume representation, which decomposes the 2D search space dynamically into two 1D orthogonal spaces, enabling our method to scale effectively to very high-resolution inputs. To preserve essential information in the orthogonal space, we utilize self attention to propagate feature information from the 2D space to the orthogonal space. We further propose a radius-distribution multi-scale lookup strategy to model the correspondences of large displacements at a negligible cost. We verify the efficiency and effectiveness of our method on the challenging Sintel and KITTI benchmarks, and real-world 4K ($2160\!\times\!3840$) images. Our method achieves competitive performance on both Sintel and KITTI benchmarks, while maintaining the highest memory efficiency on high-resolution inputs.

{{</citation>}}


### (22/108) F3-Pruning: A Training-Free and Generalized Pruning Strategy towards Faster and Finer Text-to-Video Synthesis (Sitong Su et al., 2023)

{{<citation>}}

Sitong Su, Jianzhi Liu, Lianli Gao, Jingkuan Song. (2023)  
**F3-Pruning: A Training-Free and Generalized Pruning Strategy towards Faster and Finer Text-to-Video Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2312.03459v1)  

---


**ABSTRACT**  
Recently Text-to-Video (T2V) synthesis has undergone a breakthrough by training transformers or diffusion models on large-scale datasets. Nevertheless, inferring such large models incurs huge costs.Previous inference acceleration works either require costly retraining or are model-specific.To address this issue, instead of retraining we explore the inference process of two mainstream T2V models using transformers and diffusion models.The exploration reveals the redundancy in temporal attention modules of both models, which are commonly utilized to establish temporal relations among frames.Consequently, we propose a training-free and generalized pruning strategy called F3-Pruning to prune redundant temporal attention weights.Specifically, when aggregate temporal attention values are ranked below a certain ratio, corresponding weights will be pruned.Extensive experiments on three datasets using a classic transformer-based model CogVideo and a typical diffusion-based model Tune-A-Video verify the effectiveness of F3-Pruning in inference acceleration, quality assurance and broad applicability.

{{</citation>}}


### (23/108) ShareCMP: Polarization-Aware RGB-P Semantic Segmentation (Zhuoyan Liu et al., 2023)

{{<citation>}}

Zhuoyan Liu, Bo Wang, Lizhi Wang, Chenyu Mao, Ye Li. (2023)  
**ShareCMP: Polarization-Aware RGB-P Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: I-4-6, cs-CV, cs.CV  
Keywords: Attention, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.03430v1)  

---


**ABSTRACT**  
Multimodal semantic segmentation is developing rapidly, but the modality of RGB-Polarization remains underexplored. To delve into this problem, we construct a UPLight RGB-P segmentation benchmark with 12 typical underwater semantic classes which provides data support for Autonomous Underwater Vehicles (AUVs) to perform special perception tasks. In this work, we design the ShareCMP, an RGB-P semantic segmentation framework with a shared dual-branch architecture, which reduces the number of parameters by about 26-33% compared to previous dual-branch models. It encompasses a Polarization Generate Attention (PGA) module designed to generate polarization modal images with richer polarization properties for the encoder. In addition, we introduce the Class Polarization-Aware Loss (CPALoss) to improve the learning and understanding of the encoder for polarization modal information and to optimize the PGA module. With extensive experiments on a total of three RGB-P benchmarks, our ShareCMP achieves state-of-the-art performance in mIoU with fewer parameters on the UPLight (92.45%), ZJU (92.7%), and MCubeS (50.99%) datasets. The code is available at https://github.com/LEFTeyex/ShareCMP.

{{</citation>}}


### (24/108) DeepPyramid+: Medical Image Segmentation using Pyramid View Fusion and Deformable Pyramid Reception (Negin Ghamsarian et al., 2023)

{{<citation>}}

Negin Ghamsarian, Sebastian Wolf, Martin Zinkernagel, Klaus Schoeffmann, Raphael Sznitman. (2023)  
**DeepPyramid+: Medical Image Segmentation using Pyramid View Fusion and Deformable Pyramid Reception**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.03409v1)  

---


**ABSTRACT**  
Semantic Segmentation plays a pivotal role in many applications related to medical image and video analysis. However, designing a neural network architecture for medical image and surgical video segmentation is challenging due to the diverse features of relevant classes, including heterogeneity, deformability, transparency, blunt boundaries, and various distortions. We propose a network architecture, DeepPyramid+, which addresses diverse challenges encountered in medical image and surgical video segmentation. The proposed DeepPyramid+ incorporates two major modules, namely "Pyramid View Fusion" (PVF) and "Deformable Pyramid Reception," (DPR), to address the outlined challenges. PVF replicates a deduction process within the neural network, aligning with the human visual system, thereby enhancing the representation of relative information at each pixel position. Complementarily, DPR introduces shape- and scale-adaptive feature extraction techniques using dilated deformable convolutions, enhancing accuracy and robustness in handling heterogeneous classes and deformable shapes. Extensive experiments conducted on diverse datasets, including endometriosis videos, MRI images, OCT scans, and cataract and laparoscopy videos, demonstrate the effectiveness of DeepPyramid+ in handling various challenges such as shape and scale variation, reflection, and blur degradation. DeepPyramid+ demonstrates significant improvements in segmentation performance, achieving up to a 3.65% increase in Dice coefficient for intra-domain segmentation and up to a 17% increase in Dice coefficient for cross-domain segmentation. DeepPyramid+ consistently outperforms state-of-the-art networks across diverse modalities considering different backbone networks, showcasing its versatility.

{{</citation>}}


### (25/108) SVQ: Sparse Vector Quantization for Spatiotemporal Forecasting (Chao Chen et al., 2023)

{{<citation>}}

Chao Chen, Tian Zhou, Yanjun Zhao, Hui Liu, Liang Sun, Rong Jin. (2023)  
**SVQ: Sparse Vector Quantization for Spatiotemporal Forecasting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2312.03406v2)  

---


**ABSTRACT**  
Spatiotemporal forecasting tasks, such as weather forecasting and traffic prediction, offer significant societal benefits. These tasks can be effectively approached as image forecasting problems using computer vision models. Vector quantization (VQ) is a well-known method for discrete representation that improves the latent space, leading to enhanced generalization and transfer learning capabilities. One of the main challenges in using VQ for spatiotemporal forecasting is how to balance between keeping enough details and removing noises from the original patterns for better generalization. We address this challenge by developing sparse vector quantization, or {\bf SVQ} for short, that leverages sparse regression to make better trade-off between the two objectives. The main innovation of this work is to approximate sparse regression by a two-layer MLP and a randomly fixed or learnable matrix, dramatically improving its computational efficiency. Through experiments conducted on diverse datasets in multiple fields including weather forecasting, traffic flow prediction, and video forecasting, we unequivocally demonstrate that our proposed method consistently enhances the performance of base models and achieves state-of-the-art results across all benchmarks.

{{</citation>}}


### (26/108) Riemannian Complex Matrix Convolution Network for PolSAR Image Classification (Junfei Shi et al., 2023)

{{<citation>}}

Junfei Shi, Wei Wang, Haiyan Jin, Mengmeng Nie, Shanshan Ji. (2023)  
**Riemannian Complex Matrix Convolution Network for PolSAR Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2312.03378v1)  

---


**ABSTRACT**  
Recently, deep learning methods have achieved superior performance for Polarimetric Synthetic Aperture Radar(PolSAR) image classification. Existing deep learning methods learn PolSAR data by converting the covariance matrix into a feature vector or complex-valued vector as the input. However, all these methods cannot learn the structure of complex matrix directly and destroy the channel correlation. To learn geometric structure of complex matrix, we propose a Riemannian complex matrix convolution network for PolSAR image classification in Riemannian space for the first time, which directly utilizes the complex matrix as the network input and defines the Riemannian operations to learn complex matrix's features. The proposed Riemannian complex matrix convolution network considers PolSAR complex matrix endowed in Riemannian manifold, and defines a series of new Riemannian convolution, ReLu and LogEig operations in Riemannian space, which breaks through the Euclidean constraint of conventional networks. Then, a CNN module is appended to enhance contextual Riemannian features. Besides, a fast kernel learning method is developed for the proposed method to learn class-specific features and reduce the computation time effectively. Experiments are conducted on three sets of real PolSAR data with different bands and sensors. Experiments results demonstrates the proposed method can obtain superior performance than the state-of-the-art methods.

{{</citation>}}


### (27/108) PointMoment:Mixed-Moment-based Self-Supervised Representation Learning for 3D Point Clouds (Xin Cao et al., 2023)

{{<citation>}}

Xin Cao, Xinxin Han, Yifan Wang, Mengna Yang, Kang Li. (2023)  
**PointMoment:Mixed-Moment-based Self-Supervised Representation Learning for 3D Point Clouds**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.03350v1)  

---


**ABSTRACT**  
Large and rich data is a prerequisite for effective training of deep neural networks. However, the irregularity of point cloud data makes manual annotation time-consuming and laborious. Self-supervised representation learning, which leverages the intrinsic structure of large-scale unlabelled data to learn meaningful feature representations, has attracted increasing attention in the field of point cloud research. However, self-supervised representation learning often suffers from model collapse, resulting in reduced information and diversity of the learned representation, and consequently degrading the performance of downstream tasks. To address this problem, we propose PointMoment, a novel framework for point cloud self-supervised representation learning that utilizes a high-order mixed moment loss function rather than the conventional contrastive loss function. Moreover, our framework does not require any special techniques such as asymmetric network architectures, gradient stopping, etc. Specifically, we calculate the high-order mixed moment of the feature variables and force them to decompose into products of their individual moment, thereby making multiple variables more independent and minimizing the feature redundancy. We also incorporate a contrastive learning approach to maximize the feature invariance under different data augmentations of the same point cloud. Experimental results show that our approach outperforms previous unsupervised learning methods on the downstream task of 3D point cloud classification and segmentation.

{{</citation>}}


### (28/108) Building Category Graphs Representation with Spatial and Temporal Attention for Visual Navigation (Xiaobo Hu et al., 2023)

{{<citation>}}

Xiaobo Hu, Youfang Lin, HeHe Fan, Shuo Wang, Zhihao Wu, Kai Lv. (2023)  
**Building Category Graphs Representation with Spatial and Temporal Attention for Visual Navigation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2312.03327v1)  

---


**ABSTRACT**  
Given an object of interest, visual navigation aims to reach the object's location based on a sequence of partial observations. To this end, an agent needs to 1) learn a piece of certain knowledge about the relations of object categories in the world during training and 2) look for the target object based on the pre-learned object category relations and its moving trajectory in the current unseen environment. In this paper, we propose a Category Relation Graph (CRG) to learn the knowledge of object category layout relations and a Temporal-Spatial-Region (TSR) attention architecture to perceive the long-term spatial-temporal dependencies of objects helping the navigation. We learn prior knowledge of object layout, establishing a category relationship graph to deduce the positions of specific objects. Subsequently, we introduced TSR to capture the relationships of objects in temporal, spatial, and regions within the observation trajectories. Specifically, we propose a Temporal attention module (T) to model the temporal structure of the observation sequence, which implicitly encodes the historical moving or trajectory information. Then, a Spatial attention module (S) is used to uncover the spatial context of the current observation objects based on the category relation graph and past observations. Last, a Region attention module (R) shifts the attention to the target-relevant region. Based on the visual representation extracted by our method, the agent can better perceive the environment and easily learn superior navigation policy. Experiments on AI2-THOR demonstrate our CRG-TSR method significantly outperforms existing methods regarding both effectiveness and efficiency. The code has been included in the supplementary material and will be publicly available.

{{</citation>}}


### (29/108) GCFA:Geodesic Curve Feature Augmentation via Shape Space Theory (Yuexing Han et al., 2023)

{{<citation>}}

Yuexing Han, Guanxin Wan, Bing Wang. (2023)  
**GCFA:Geodesic Curve Feature Augmentation via Shape Space Theory**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2312.03325v1)  

---


**ABSTRACT**  
Deep learning has yielded remarkable outcomes in various domains. However, the challenge of requiring large-scale labeled samples still persists in deep learning. Thus, data augmentation has been introduced as a critical strategy to train deep learning models. However, data augmentation suffers from information loss and poor performance in small sample environments. To overcome these drawbacks, we propose a feature augmentation method based on shape space theory, i.e., Geodesic curve feature augmentation, called GCFA in brevity. First, we extract features from the image with the neural network model. Then, the multiple image features are projected into a pre-shape space as features. In the pre-shape space, a Geodesic curve is built to fit the features. Finally, the many generated features on the Geodesic curve are used to train the various machine learning models. The GCFA module can be seamlessly integrated with most machine learning methods. And the proposed method is simple, effective and insensitive for the small sample datasets. Several examples demonstrate that the GCFA method can greatly improve the performance of the data preprocessing model in a small sample environment.

{{</citation>}}


### (30/108) On the Robustness of Large Multimodal Models Against Image Adversarial Attacks (Xuanming Cui et al., 2023)

{{<citation>}}

Xuanming Cui, Alejandro Aparcedo, Young Kyun Jang, Ser-Nam Lim. (2023)  
**On the Robustness of Large Multimodal Models Against Image Adversarial Attacks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack, QA  
[Paper Link](http://arxiv.org/abs/2312.03777v2)  

---


**ABSTRACT**  
Recent advances in instruction tuning have led to the development of State-of-the-Art Large Multimodal Models (LMMs). Given the novelty of these models, the impact of visual adversarial attacks on LMMs has not been thoroughly examined. We conduct a comprehensive study of the robustness of various LMMs against different adversarial attacks, evaluated across tasks including image classification, image captioning, and Visual Question Answer (VQA). We find that in general LMMs are not robust to visual adversarial inputs. However, our findings suggest that context provided to the model via prompts, such as questions in a QA pair helps to mitigate the effects of visual adversarial inputs. Notably, the LMMs evaluated demonstrated remarkable resilience to such attacks on the ScienceQA task with only an 8.10% drop in performance compared to their visual counterparts which dropped 99.73%. We also propose a new approach to real-world image classification which we term query decomposition. By incorporating existence queries into our input prompt we observe diminished attack effectiveness and improvements in image classification accuracy. This research highlights a previously under-explored facet of LMM robustness and sets the stage for future work aimed at strengthening the resilience of multimodal systems in adversarial environments.

{{</citation>}}


### (31/108) Class Incremental Learning for Adversarial Robustness (Seungju Cho et al., 2023)

{{<citation>}}

Seungju Cho, Hongsin Lee, Changick Kim. (2023)  
**Class Incremental Learning for Adversarial Robustness**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.03289v2)  

---


**ABSTRACT**  
Adversarial training integrates adversarial examples during model training to enhance robustness. However, its application in fixed dataset settings differs from real-world dynamics, where data accumulates incrementally. In this study, we investigate Adversarially Robust Class Incremental Learning (ARCIL), a method that combines adversarial robustness with incremental learning. We observe that combining incremental learning with naive adversarial training easily leads to a loss of robustness. We discover that this is attributed to the disappearance of the flatness of the loss function, a characteristic of adversarial training. To address this issue, we propose the Flatness Preserving Distillation (FPD) loss that leverages the output difference between adversarial and clean examples. Additionally, we introduce the Logit Adjustment Distillation (LAD) loss, which adapts the model's knowledge to perform well on new tasks. Experimental results demonstrate the superiority of our method over approaches that apply adversarial training to existing incremental learning methods, which provides a strong baseline for incremental learning on adversarial robustness in the future. Our method achieves AutoAttack accuracy that is 5.99\%p, 5.27\%p, and 3.90\%p higher on average than the baseline on split CIFAR-10, CIFAR-100, and Tiny ImageNet, respectively. The code will be made available.

{{</citation>}}


### (32/108) STEP CATFormer: Spatial-Temporal Effective Body-Part Cross Attention Transformer for Skeleton-based Action Recognition (Nguyen Huu Bao Long, 2023)

{{<citation>}}

Nguyen Huu Bao Long. (2023)  
**STEP CATFormer: Spatial-Temporal Effective Body-Part Cross Attention Transformer for Skeleton-based Action Recognition**  

---
Primary Category: cs.CV  
Categories: I-2-10, cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.03288v1)  

---


**ABSTRACT**  
Graph convolutional networks (GCNs) have been widely used and achieved remarkable results in skeleton-based action recognition. We think the key to skeleton-based action recognition is a skeleton hanging in frames, so we focus on how the Graph Convolutional Convolution networks learn different topologies and effectively aggregate joint features in the global temporal and local temporal. In this work, we propose three Channel-wise Tolopogy Graph Convolution based on Channel-wise Topology Refinement Graph Convolution (CTR-GCN). Combining CTR-GCN with two joint cross-attention modules can capture the upper-lower body part and hand-foot relationship skeleton features. After that, to capture features of human skeletons changing in frames we design the Temporal Attention Transformers to extract skeletons effectively. The Temporal Attention Transformers can learn the temporal features of human skeleton sequences. Finally, we fuse the temporal features output scale with MLP and classification. We develop a powerful graph convolutional network named Spatial Temporal Effective Body-part Cross Attention Transformer which notably high-performance on the NTU RGB+D, NTU RGB+D 120 datasets. Our code and models are available at https://github.com/maclong01/STEP-CATFormer

{{</citation>}}


### (33/108) Satellite Imagery and AI: A New Era in Ocean Conservation, from Research to Deployment and Impact (Patrick Beukema et al., 2023)

{{<citation>}}

Patrick Beukema, Favyen Bastani, Piper Wolters, Henry Herzog, Joe Ferdinando. (2023)  
**Satellite Imagery and AI: A New Era in Ocean Conservation, from Research to Deployment and Impact**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03207v1)  

---


**ABSTRACT**  
Illegal, unreported, and unregulated (IUU) fishing poses a global threat to ocean habitats. Publicly available satellite data offered by NASA and the European Space Agency (ESA) provide an opportunity to actively monitor this activity. Effectively leveraging satellite data for maritime conservation requires highly reliable machine learning models operating globally with minimal latency. This paper introduces three specialized computer vision models designed for synthetic aperture radar (Sentinel-1), optical imagery (Sentinel-2), and nighttime lights (Suomi-NPP/NOAA-20). It also presents best practices for developing and delivering real-time computer vision services for conservation. These models have been deployed in Skylight, a real time maritime monitoring platform, which is provided at no cost to users worldwide.

{{</citation>}}


## cs.LG (23)



### (34/108) Adaptive Weighted Co-Learning for Cross-Domain Few-Shot Learning (Abdullah Alchihabi et al., 2023)

{{<citation>}}

Abdullah Alchihabi, Marzi Heidari, Yuhong Guo. (2023)  
**Adaptive Weighted Co-Learning for Cross-Domain Few-Shot Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.03928v1)  

---


**ABSTRACT**  
Due to the availability of only a few labeled instances for the novel target prediction task and the significant domain shift between the well annotated source domain and the target domain, cross-domain few-shot learning (CDFSL) induces a very challenging adaptation problem. In this paper, we propose a simple Adaptive Weighted Co-Learning (AWCoL) method to address the CDFSL challenge by adapting two independently trained source prototypical classification models to the target task in a weighted co-learning manner. The proposed method deploys a weighted moving average prediction strategy to generate probabilistic predictions from each model, and then conducts adaptive co-learning by jointly fine-tuning the two models in an alternating manner based on the pseudo-labels and instance weights produced from the predictions. Moreover, a negative pseudo-labeling regularizer is further deployed to improve the fine-tuning process by penalizing false predictions. Comprehensive experiments are conducted on multiple benchmark datasets and the empirical results demonstrate that the proposed method produces state-of-the-art CDFSL performance.

{{</citation>}}


### (35/108) A Pseudo-Semantic Loss for Autoregressive Models with Logical Constraints (Kareem Ahmed et al., 2023)

{{<citation>}}

Kareem Ahmed, Kai-Wei Chang, Guy Van den Broeck. (2023)  
**A Pseudo-Semantic Loss for Autoregressive Models with Logical Constraints**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03905v1)  

---


**ABSTRACT**  
Neuro-symbolic AI bridges the gap between purely symbolic and neural approaches to learning. This often requires maximizing the likelihood of a symbolic constraint w.r.t the neural network's output distribution. Such output distributions are typically assumed to be fully-factorized. This limits the applicability of neuro-symbolic learning to the more expressive autoregressive distributions, e.g., transformers. Under such distributions, computing the likelihood of even simple constraints is #P-hard. Instead of attempting to enforce the constraint on the entire output distribution, we propose to do so on a random, local approximation thereof. More precisely, we optimize the likelihood of the constraint under a pseudolikelihood-based approximation centered around a model sample. Our approximation is factorized, allowing the reuse of solutions to sub-problems, a main tenet for efficiently computing neuro-symbolic losses. Moreover, it is a local, high-fidelity approximation of the likelihood, exhibiting low entropy and KL-divergence around the model sample. We evaluate our approach on Sudoku and shortest-path prediction cast as autoregressive generation, and observe that we greatly improve upon the base model's ability to predict logically-consistent outputs. We also evaluate on the task of detoxifying large language models. Using a simple constraint disallowing a list of toxic words, we are able to steer the model's outputs away from toxic generations, achieving SoTA detoxification compared to previous approaches.

{{</citation>}}


### (36/108) Adaptive Dependency Learning Graph Neural Networks (Abishek Sriramulu et al., 2023)

{{<citation>}}

Abishek Sriramulu, Nicolas Fourrier, Christoph Bergmeir. (2023)  
**Adaptive Dependency Learning Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.03903v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNN) have recently gained popularity in the forecasting domain due to their ability to model complex spatial and temporal patterns in tasks such as traffic forecasting and region-based demand forecasting. Most of these methods require a predefined graph as input, whereas in real-life multivariate time series problems, a well-predefined dependency graph rarely exists. This requirement makes it harder for GNNs to be utilised widely for multivariate forecasting problems in other domains such as retail or energy. In this paper, we propose a hybrid approach combining neural networks and statistical structure learning models to self-learn the dependencies and construct a dynamically changing dependency graph from multivariate data aiming to enable the use of GNNs for multivariate forecasting even when a well-defined graph does not exist. The statistical structure modeling in conjunction with neural networks provides a well-principled and efficient approach by bringing in causal semantics to determine dependencies among the series. Finally, we demonstrate significantly improved performance using our proposed approach on real-world benchmark datasets without a pre-defined dependency graph.

{{</citation>}}


### (37/108) A Masked Pruning Approach for Dimensionality Reduction in Communication-Efficient Federated Learning Systems (Tamir L. S. Gez et al., 2023)

{{<citation>}}

Tamir L. S. Gez, Kobi Cohen. (2023)  
**A Masked Pruning Approach for Dimensionality Reduction in Communication-Efficient Federated Learning Systems**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2312.03889v1)  

---


**ABSTRACT**  
Federated Learning (FL) represents a growing machine learning (ML) paradigm designed for training models across numerous nodes that retain local datasets, all without directly exchanging the underlying private data with the parameter server (PS). Its increasing popularity is attributed to notable advantages in terms of training deep neural network (DNN) models under privacy aspects and efficient utilization of communication resources. Unfortunately, DNNs suffer from high computational and communication costs, as well as memory consumption in intricate tasks. These factors restrict the applicability of FL algorithms in communication-constrained systems with limited hardware resources.   In this paper, we develop a novel algorithm that overcomes these limitations by synergistically combining a pruning-based method with the FL process, resulting in low-dimensional representations of the model with minimal communication cost, dubbed Masked Pruning over FL (MPFL). The algorithm operates by initially distributing weights to the nodes through the PS. Subsequently, each node locally trains its model and computes pruning masks. These low-dimensional masks are then transmitted back to the PS, which generates a consensus pruning mask, broadcasted back to the nodes. This iterative process enhances the robustness and stability of the masked pruning model. The generated mask is used to train the FL model, achieving significant bandwidth savings. We present an extensive experimental study demonstrating the superior performance of MPFL compared to existing methods. Additionally, we have developed an open-source software package for the benefit of researchers and developers in related fields.

{{</citation>}}


### (38/108) Learning Genomic Sequence Representations using Graph Neural Networks over De Bruijn Graphs (Kacper Kapuśniak et al., 2023)

{{<citation>}}

Kacper Kapuśniak, Manuel Burger, Gunnar Rätsch, Amir Joudaki. (2023)  
**Learning Genomic Sequence Representations using Graph Neural Networks over De Bruijn Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-GN  
Keywords: Contrastive Learning, Graph Convolutional Network, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.03865v1)  

---


**ABSTRACT**  
The rapid expansion of genomic sequence data calls for new methods to achieve robust sequence representations. Existing techniques often neglect intricate structural details, emphasizing mainly contextual information. To address this, we developed k-mer embeddings that merge contextual and structural string information by enhancing De Bruijn graphs with structural similarity connections. Subsequently, we crafted a self-supervised method based on Contrastive Learning that employs a heterogeneous Graph Convolutional Network encoder and constructs positive pairs based on node similarities. Our embeddings consistently outperform prior techniques for Edit Distance Approximation and Closest String Retrieval tasks.

{{</citation>}}


### (39/108) Pearl: A Production-ready Reinforcement Learning Agent (Zheqing Zhu et al., 2023)

{{<citation>}}

Zheqing Zhu, Rodrigo de Salvo Braz, Jalaj Bhandari, Daniel Jiang, Yi Wan, Yonathan Efroni, Liyuan Wang, Ruiyang Xu, Hongbo Guo, Alex Nikulkov, Dmytro Korenkevych, Urun Dogan, Frank Cheng, Zheng Wu, Wanqiao Xu. (2023)  
**Pearl: A Production-ready Reinforcement Learning Agent**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.03814v1)  

---


**ABSTRACT**  
Reinforcement Learning (RL) offers a versatile framework for achieving long-term goals. Its generality allows us to formalize a wide range of problems that real-world intelligent systems encounter, such as dealing with delayed rewards, handling partial observability, addressing the exploration and exploitation dilemma, utilizing offline data to improve online performance, and ensuring safety constraints are met. Despite considerable progress made by the RL research community in addressing these issues, existing open-source RL libraries tend to focus on a narrow portion of the RL solution pipeline, leaving other aspects largely unattended. This paper introduces Pearl, a Production-ready RL agent software package explicitly designed to embrace these challenges in a modular fashion. In addition to presenting preliminary benchmark results, this paper highlights Pearl's industry adoptions to demonstrate its readiness for production usage. Pearl is open sourced on Github at github.com/facebookresearch/pearl and its official website is located at pearlagent.github.io.

{{</citation>}}


### (40/108) Interpretability Illusions in the Generalization of Simplified Models (Dan Friedman et al., 2023)

{{<citation>}}

Dan Friedman, Andrew Lampinen, Lucas Dixon, Danqi Chen, Asma Ghandeharioun. (2023)  
**Interpretability Illusions in the Generalization of Simplified Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.03656v1)  

---


**ABSTRACT**  
A common method to study deep learning systems is to use simplified model representations -- for example, using singular value decomposition to visualize the model's hidden states in a lower dimensional space. This approach assumes that the results of these simplified are faithful to the original model. Here, we illustrate an important caveat to this assumption: even if the simplified representations can accurately approximate the full model on the training set, they may fail to accurately capture the model's behavior out of distribution -- the understanding developed from simplified representations may be an illusion. We illustrate this by training Transformer models on controlled datasets with systematic generalization splits. First, we train models on the Dyck balanced-parenthesis languages. We simplify these models using tools like dimensionality reduction and clustering, and then explicitly test how these simplified proxies match the behavior of the original model on various out-of-distribution test sets. We find that the simplified proxies are generally less faithful out of distribution. In cases where the original model generalizes to novel structures or deeper depths, the simplified versions may fail, or generalize better. This finding holds even if the simplified representations do not directly depend on the training distribution. Next, we study a more naturalistic task: predicting the next character in a dataset of computer code. We find similar generalization gaps between the original model and simplified proxies, and conduct further analysis to investigate which aspects of the code completion task are associated with the largest gaps. Together, our results raise questions about the extent to which mechanistic interpretations derived using tools like SVD can reliably predict what a model will do in novel situations.

{{</citation>}}


### (41/108) MACCA: Offline Multi-agent Reinforcement Learning with Causal Credit Assignment (Ziyan Wang et al., 2023)

{{<citation>}}

Ziyan Wang, Yali Du, Yudi Zhang, Meng Fang, Biwei Huang. (2023)  
**MACCA: Offline Multi-agent Reinforcement Learning with Causal Credit Assignment**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.03644v1)  

---


**ABSTRACT**  
Offline Multi-agent Reinforcement Learning (MARL) is valuable in scenarios where online interaction is impractical or risky. While independent learning in MARL offers flexibility and scalability, accurately assigning credit to individual agents in offline settings poses challenges due to partial observability and emergent behavior. Directly transferring the online credit assignment method to offline settings results in suboptimal outcomes due to the absence of real-time feedback and intricate agent interactions. Our approach, MACCA, characterizing the generative process as a Dynamic Bayesian Network, captures relationships between environmental variables, states, actions, and rewards. Estimating this model on offline data, MACCA can learn each agent's contribution by analyzing the causal relationship of their individual rewards, ensuring accurate and interpretable credit assignment. Additionally, the modularity of our approach allows it to seamlessly integrate with various offline MARL methods. Theoretically, we proved that under the setting of the offline dataset, the underlying causal structure and the function for generating the individual rewards of agents are identifiable, which laid the foundation for the correctness of our modeling. Experimentally, we tested MACCA in two environments, including discrete and continuous action settings. The results show that MACCA outperforms SOTA methods and improves performance upon their backbones.

{{</citation>}}


### (42/108) Transformer-Powered Surrogates Close the ICF Simulation-Experiment Gap with Extremely Limited Data (Matthew L. Olson et al., 2023)

{{<citation>}}

Matthew L. Olson, Shusen Liu, Jayaraman J. Thiagarajan, Bogdan Kustowski, Weng-Keen Wong, Rushil Anirudh. (2023)  
**Transformer-Powered Surrogates Close the ICF Simulation-Experiment Gap with Extremely Limited Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.03642v1)  

---


**ABSTRACT**  
Recent advances in machine learning, specifically transformer architecture, have led to significant advancements in commercial domains. These powerful models have demonstrated superior capability to learn complex relationships and often generalize better to new data and problems. This paper presents a novel transformer-powered approach for enhancing prediction accuracy in multi-modal output scenarios, where sparse experimental data is supplemented with simulation data. The proposed approach integrates transformer-based architecture with a novel graph-based hyper-parameter optimization technique. The resulting system not only effectively reduces simulation bias, but also achieves superior prediction accuracy compared to the prior method. We demonstrate the efficacy of our approach on inertial confinement fusion experiments, where only 10 shots of real-world data are available, as well as synthetic versions of these experiments.

{{</citation>}}


### (43/108) Multi-Scale and Multi-Modal Contrastive Learning Network for Biomedical Time Series (Hongbo Guo et al., 2023)

{{<citation>}}

Hongbo Guo, Xinzi Xu, Hao Wu, Guoxing Wang. (2023)  
**Multi-Scale and Multi-Modal Contrastive Learning Network for Biomedical Time Series**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2312.03796v1)  

---


**ABSTRACT**  
Multi-modal biomedical time series (MBTS) data offers a holistic view of the physiological state, holding significant importance in various bio-medical applications. Owing to inherent noise and distribution gaps across different modalities, MBTS can be complex to model. Various deep learning models have been developed to learn representations of MBTS but still fall short in robustness due to the ignorance of modal-to-modal variations. This paper presents a multi-scale and multi-modal biomedical time series representation learning (MBSL) network with contrastive learning to migrate these variations. Firstly, MBTS is grouped based on inter-modal distances, then each group with minimum intra-modal variations can be effectively modeled by individual encoders. Besides, to enhance the multi-scale feature extraction (encoder), various patch lengths and mask ratios are designed to generate tokens with semantic information at different scales and diverse contextual perspectives respectively. Finally, cross-modal contrastive learning is proposed to maximize consistency among inter-modal groups, maintaining useful information and eliminating noises. Experiments against four bio-medical applications show that MBSL outperforms state-of-the-art models by 33.9% mean average errors (MAE) in respiration rate, by 13.8% MAE in exercise heart rate, by 1.41% accuracy in human activity recognition, and by 1.14% F1-score in obstructive sleep apnea-hypopnea syndrome.

{{</citation>}}


### (44/108) Towards Sobolev Pruning (Neil Kichler et al., 2023)

{{<citation>}}

Neil Kichler, Sher Afghan, Uwe Naumann. (2023)  
**Towards Sobolev Pruning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-fin-CP  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2312.03510v2)  

---


**ABSTRACT**  
The increasing use of stochastic models for describing complex phenomena warrants surrogate models that capture the reference model characteristics at a fraction of the computational cost, foregoing potentially expensive Monte Carlo simulation. The predominant approach of fitting a large neural network and then pruning it to a reduced size has commonly neglected shortcomings. The produced surrogate models often will not capture the sensitivities and uncertainties inherent in the original model. In particular, (higher-order) derivative information of such surrogates could differ drastically. Given a large enough network, we expect this derivative information to match. However, the pruned model will almost certainly not share this behavior.   In this paper, we propose to find surrogate models by using sensitivity information throughout the learning and pruning process. We build on work using Interval Adjoint Significance Analysis for pruning and combine it with the recent advancements in Sobolev Training to accurately model the original sensitivity information in the pruned neural network based surrogate model. We experimentally underpin the method on an example of pricing a multidimensional Basket option modelled through a stochastic differential equation with Brownian motion. The proposed method is, however, not limited to the domain of quantitative finance, which was chosen as a case study for intuitive interpretations of the sensitivities. It serves as a foundation for building further surrogate modelling techniques considering sensitivity information.

{{</citation>}}


### (45/108) SmoothQuant+: Accurate and Efficient 4-bit Post-Training WeightQuantization for LLM (Jiayi Pan et al., 2023)

{{<citation>}}

Jiayi Pan, Chengcan Wang, Kaifu Zheng, Yangguang Li, Zhenyu Wang, Bin Feng. (2023)  
**SmoothQuant+: Accurate and Efficient 4-bit Post-Training WeightQuantization for LLM**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2312.03788v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown remarkable capabilities in various tasks. However their huge model size and the consequent demand for computational and memory resources also pose challenges to model deployment. Currently, 4-bit post-training quantization (PTQ) has achieved some success in LLMs, reducing the memory footprint by approximately 75% compared to FP16 models, albeit with some accuracy loss. In this paper, we propose SmoothQuant+, an accurate and efficient 4-bit weight-only PTQ that requires no additional training, which enables lossless in accuracy for LLMs for the first time. Based on the fact that the loss of weight quantization is amplified by the activation outliers, SmoothQuant+ smoothes the activation outliers by channel before quantization, while adjusting the corresponding weights for mathematical equivalence, and then performs group-wise 4-bit weight quantization for linear layers. We have integrated SmoothQuant+ into the vLLM framework, an advanced high-throughput inference engine specially developed for LLMs, and equipped it with an efficient W4A16 CUDA kernels, so that vLLM can seamlessly support SmoothQuant+ 4-bit weight quantization. Our results show that, with SmoothQuant+, the Code Llama-34B model can be quantized and deployed on a A100 40GB GPU, achieving lossless accuracy and a throughput increase of 1.9 to 4.0 times compared to the FP16 model deployed on two A100 40GB GPUs. Moreover, the latency per token is only 68% of the FP16 model deployed on two A100 40GB GPUs. This is the state-of-the-art 4-bit weight quantization for LLMs as we know.

{{</citation>}}


### (46/108) Compressed Context Memory For Online Language Model Interaction (Jang-Hyun Kim et al., 2023)

{{<citation>}}

Jang-Hyun Kim, Junyoung Yeom, Sangdoo Yun, Hyun Oh Song. (2023)  
**Compressed Context Memory For Online Language Model Interaction**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: ChatGPT, GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2312.03414v1)  

---


**ABSTRACT**  
This paper presents a novel context compression method for Transformer language models in online scenarios such as ChatGPT, where the context continually expands. As the context lengthens, the attention process requires more memory and computational resources, which in turn reduces the throughput of the language model. To this end, we propose a compressed context memory system that continually compresses the growing context into a compact memory space. The compression process simply involves integrating a lightweight conditional LoRA into the language model's forward pass during inference. Based on the compressed context memory, the language model can perform inference with reduced memory and attention operations. Through evaluations on conversation, personalization, and multi-task learning, we demonstrate that our approach achieves the performance level of a full context model with $5\times$ smaller context memory space. Codes are available at https://github.com/snu-mllab/context-memory.

{{</citation>}}


### (47/108) Generalized Contrastive Divergence: Joint Training of Energy-Based Model and Diffusion Model through Inverse Reinforcement Learning (Sangwoong Yoon et al., 2023)

{{<citation>}}

Sangwoong Yoon, Dohyun Kwon, Himchan Hwang, Yung-Kyun Noh, Frank C. Park. (2023)  
**Generalized Contrastive Divergence: Joint Training of Energy-Based Model and Diffusion Model through Inverse Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.03397v1)  

---


**ABSTRACT**  
We present Generalized Contrastive Divergence (GCD), a novel objective function for training an energy-based model (EBM) and a sampler simultaneously. GCD generalizes Contrastive Divergence (Hinton, 2002), a celebrated algorithm for training EBM, by replacing Markov Chain Monte Carlo (MCMC) distribution with a trainable sampler, such as a diffusion model. In GCD, the joint training of EBM and a diffusion model is formulated as a minimax problem, which reaches an equilibrium when both models converge to the data distribution. The minimax learning with GCD bears interesting equivalence to inverse reinforcement learning, where the energy corresponds to a negative reward, the diffusion model is a policy, and the real data is expert demonstrations. We present preliminary yet promising results showing that joint training is beneficial for both EBM and a diffusion model. GCD enables EBM training without MCMC while improving the sample quality of a diffusion model.

{{</citation>}}


### (48/108) Complementary Benefits of Contrastive Learning and Self-Training Under Distribution Shift (Saurabh Garg et al., 2023)

{{<citation>}}

Saurabh Garg, Amrith Setlur, Zachary Chase Lipton, Sivaraman Balakrishnan, Virginia Smith, Aditi Raghunathan. (2023)  
**Complementary Benefits of Contrastive Learning and Self-Training Under Distribution Shift**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG, stat-ML  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.03318v1)  

---


**ABSTRACT**  
Self-training and contrastive learning have emerged as leading techniques for incorporating unlabeled data, both under distribution shift (unsupervised domain adaptation) and when it is absent (semi-supervised learning). However, despite the popularity and compatibility of these techniques, their efficacy in combination remains unexplored. In this paper, we undertake a systematic empirical investigation of this combination, finding that (i) in domain adaptation settings, self-training and contrastive learning offer significant complementary gains; and (ii) in semi-supervised learning settings, surprisingly, the benefits are not synergistic. Across eight distribution shift datasets (e.g., BREEDs, WILDS), we demonstrate that the combined method obtains 3--8% higher accuracy than either approach independently. We then theoretically analyze these techniques in a simplified model of distribution shift, demonstrating scenarios under which the features produced by contrastive learning can yield a good initialization for self-training to further amplify gains and achieve optimal performance, even when either method alone would fail.

{{</citation>}}


### (49/108) Enhancing Molecular Property Prediction via Mixture of Collaborative Experts (Xu Yao et al., 2023)

{{<citation>}}

Xu Yao, Shuang Liang, Songqiao Han, Hailiang Huang. (2023)  
**Enhancing Molecular Property Prediction via Mixture of Collaborative Experts**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MA, cs.LG, q-bio-QM  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.03292v1)  

---


**ABSTRACT**  
Molecular Property Prediction (MPP) task involves predicting biochemical properties based on molecular features, such as molecular graph structures, contributing to the discovery of lead compounds in drug development. To address data scarcity and imbalance in MPP, some studies have adopted Graph Neural Networks (GNN) as an encoder to extract commonalities from molecular graphs. However, these approaches often use a separate predictor for each task, neglecting the shared characteristics among predictors corresponding to different tasks. In response to this limitation, we introduce the GNN-MoCE architecture. It employs the Mixture of Collaborative Experts (MoCE) as predictors, exploiting task commonalities while confronting the homogeneity issue in the expert pool and the decision dominance dilemma within the expert group. To enhance expert diversity for collaboration among all experts, the Expert-Specific Projection method is proposed to assign a unique projection perspective to each expert. To balance decision-making influence for collaboration within the expert group, the Expert-Specific Loss is presented to integrate individual expert loss into the weighted decision loss of the group for more equitable training. Benefiting from the enhancements of MoCE in expert creation, dynamic expert group formation, and experts' collaboration, our model demonstrates superior performance over traditional methods on 24 MPP datasets, especially in tasks with limited data or high imbalance.

{{</citation>}}


### (50/108) OMNIINPUT: A Model-centric Evaluation Framework through Output Distribution (Weitang Liu et al., 2023)

{{<citation>}}

Weitang Liu, Ying Wai Li, Tianle Wang, Yi-Zhuang You, Jingbo Shang. (2023)  
**OMNIINPUT: A Model-centric Evaluation Framework through Output Distribution**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03291v1)  

---


**ABSTRACT**  
We propose a novel model-centric evaluation framework, OmniInput, to evaluate the quality of an AI/ML model's predictions on all possible inputs (including human-unrecognizable ones), which is crucial for AI safety and reliability. Unlike traditional data-centric evaluation based on pre-defined test sets, the test set in OmniInput is self-constructed by the model itself and the model quality is evaluated by investigating its output distribution. We employ an efficient sampler to obtain representative inputs and the output distribution of the trained model, which, after selective annotation, can be used to estimate the model's precision and recall at different output values and a comprehensive precision-recall curve. Our experiments demonstrate that OmniInput enables a more fine-grained comparison between models, especially when their performance is almost the same on pre-defined datasets, leading to new findings and insights for how to train more robust, generalizable models.

{{</citation>}}


### (51/108) Anomaly Detection for Scalable Task Grouping in Reinforcement Learning-based RAN Optimization (Jimmy Li et al., 2023)

{{<citation>}}

Jimmy Li, Igor Kozlov, Di Wu, Xue Liu, Gregory Dudek. (2023)  
**Anomaly Detection for Scalable Task Grouping in Reinforcement Learning-based RAN Optimization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.03277v1)  

---


**ABSTRACT**  
The use of learning-based methods for optimizing cellular radio access networks (RAN) has received increasing attention in recent years. This coincides with a rapid increase in the number of cell sites worldwide, driven largely by dramatic growth in cellular network traffic. Training and maintaining learned models that work well across a large number of cell sites has thus become a pertinent problem. This paper proposes a scalable framework for constructing a reinforcement learning policy bank that can perform RAN optimization across a large number of cell sites with varying traffic patterns. Central to our framework is a novel application of anomaly detection techniques to assess the compatibility between sites (tasks) and the policy bank. This allows our framework to intelligently identify when a policy can be reused for a task, and when a new policy needs to be trained and added to the policy bank. Our results show that our approach to compatibility assessment leads to an efficient use of computational resources, by allowing us to construct a performant policy bank without exhaustively training on all tasks, which makes it applicable under real-world constraints.

{{</citation>}}


### (52/108) CAFE: Towards Compact, Adaptive, and Fast Embedding for Large-scale Recommendation Models (Hailin Zhang et al., 2023)

{{<citation>}}

Hailin Zhang, Zirui Liu, Boxuan Chen, Yikai Zhao, Tong Zhao, Tong Yang, Bin Cui. (2023)  
**CAFE: Towards Compact, Adaptive, and Fast Embedding for Large-scale Recommendation Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding, Sketch  
[Paper Link](http://arxiv.org/abs/2312.03256v1)  

---


**ABSTRACT**  
Recently, the growing memory demands of embedding tables in Deep Learning Recommendation Models (DLRMs) pose great challenges for model training and deployment. Existing embedding compression solutions cannot simultaneously meet three key design requirements: memory efficiency, low latency, and adaptability to dynamic data distribution. This paper presents CAFE, a Compact, Adaptive, and Fast Embedding compression framework that addresses the above requirements. The design philosophy of CAFE is to dynamically allocate more memory resources to important features (called hot features), and allocate less memory to unimportant ones. In CAFE, we propose a fast and lightweight sketch data structure, named HotSketch, to capture feature importance and report hot features in real time. For each reported hot feature, we assign it a unique embedding. For the non-hot features, we allow multiple features to share one embedding by using hash embedding technique. Guided by our design philosophy, we further propose a multi-level hash embedding framework to optimize the embedding tables of non-hot features. We theoretically analyze the accuracy of HotSketch, and analyze the model convergence against deviation. Extensive experiments show that CAFE significantly outperforms existing embedding compression methods, yielding 3.92% and 3.68% superior testing AUC on Criteo Kaggle dataset and CriteoTB dataset at a compression ratio of 10000x. The source codes of CAFE are available at GitHub.

{{</citation>}}


### (53/108) Customizable Combination of Parameter-Efficient Modules for Multi-Task Learning (Haowen Wang et al., 2023)

{{<citation>}}

Haowen Wang, Tao Sun, Cong Fan, Jinjie Gu. (2023)  
**Customizable Combination of Parameter-Efficient Modules for Multi-Task Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GLUE, SuperGLUE  
[Paper Link](http://arxiv.org/abs/2312.03248v1)  

---


**ABSTRACT**  
Modular and composable transfer learning is an emerging direction in the field of Parameter Efficient Fine-Tuning, as it enables neural networks to better organize various aspects of knowledge, leading to improved cross-task generalization. In this paper, we introduce a novel approach Customized Polytropon C-Poly that combines task-common skills and task-specific skills, while the skill parameters being highly parameterized using low-rank techniques. Each task is associated with a customizable number of exclusive specialized skills and also benefits from skills shared with peer tasks. A skill assignment matrix is jointly learned. To evaluate our approach, we conducted extensive experiments on the Super-NaturalInstructions and the SuperGLUE benchmarks. Our findings demonstrate that C-Poly outperforms fully-shared, task-specific, and skill-indistinguishable baselines, significantly enhancing the sample efficiency in multi-task learning scenarios.

{{</citation>}}


### (54/108) Multicoated and Folded Graph Neural Networks with Strong Lottery Tickets (Jiale Yan et al., 2023)

{{<citation>}}

Jiale Yan, Hiroaki Ito, Ángel López García-Arias, Yasuyuki Okoshi, Hikari Otsuka, Kazushi Kawamura, Thiem Van Chu, Masato Motomura. (2023)  
**Multicoated and Folded Graph Neural Networks with Strong Lottery Tickets**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.03236v1)  

---


**ABSTRACT**  
The Strong Lottery Ticket Hypothesis (SLTH) demonstrates the existence of high-performing subnetworks within a randomly initialized model, discoverable through pruning a convolutional neural network (CNN) without any weight training. A recent study, called Untrained GNNs Tickets (UGT), expanded SLTH from CNNs to shallow graph neural networks (GNNs). However, discrepancies persist when comparing baseline models with learned dense weights. Additionally, there remains an unexplored area in applying SLTH to deeper GNNs, which, despite delivering improved accuracy with additional layers, suffer from excessive memory requirements. To address these challenges, this work utilizes Multicoated Supermasks (M-Sup), a scalar pruning mask method, and implements it in GNNs by proposing a strategy for setting its pruning thresholds adaptively. In the context of deep GNNs, this research uncovers the existence of untrained recurrent networks, which exhibit performance on par with their trained feed-forward counterparts. This paper also introduces the Multi-Stage Folding and Unshared Masks methods to expand the search space in terms of both architecture and parameters. Through the evaluation of various datasets, including the Open Graph Benchmark (OGB), this work establishes a triple-win scenario for SLTH-based GNNs: by achieving high sparsity, competitive performance, and high memory efficiency with up to 98.7\% reduction, it demonstrates suitability for energy-efficient graph processing.

{{</citation>}}


### (55/108) Bootstrap Your Own Variance (Polina Turishcheva et al., 2023)

{{<citation>}}

Polina Turishcheva, Jason Ramapuram, Sinead Williamson, Dan Busbridge, Eeshan Dhekane, Russ Webb. (2023)  
**Bootstrap Your Own Variance**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.03213v1)  

---


**ABSTRACT**  
Understanding model uncertainty is important for many applications. We propose Bootstrap Your Own Variance (BYOV), combining Bootstrap Your Own Latent (BYOL), a negative-free Self-Supervised Learning (SSL) algorithm, with Bayes by Backprop (BBB), a Bayesian method for estimating model posteriors. We find that the learned predictive std of BYOV vs. a supervised BBB model is well captured by a Gaussian distribution, providing preliminary evidence that the learned parameter posterior is useful for label free uncertainty estimation. BYOV improves upon the deterministic BYOL baseline (+2.83% test ECE, +1.03% test Brier) and presents better calibration and reliability when tested with various augmentations (eg: +2.4% test ECE, +1.2% test Brier for Salt & Pepper noise).

{{</citation>}}


### (56/108) Domain Invariant Representation Learning and Sleep Dynamics Modeling for Automatic Sleep Staging (Seungyeon Lee et al., 2023)

{{<citation>}}

Seungyeon Lee, Thai-Hoang Pham, Zhao Cheng, Ping Zhang. (2023)  
**Domain Invariant Representation Learning and Sleep Dynamics Modeling for Automatic Sleep Staging**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.03196v2)  

---


**ABSTRACT**  
Sleep staging has become a critical task in diagnosing and treating sleep disorders to prevent sleep related diseases. With rapidly growing large scale public sleep databases and advances in machine learning, significant progress has been made toward automatic sleep staging. However, previous studies face some critical problems in sleep studies; the heterogeneity of subjects' physiological signals, the inability to extract meaningful information from unlabeled sleep signal data to improve predictive performances, the difficulty in modeling correlations between sleep stages, and the lack of an effective mechanism to quantify predictive uncertainty. In this study, we propose a neural network based automatic sleep staging model, named DREAM, to learn domain generalized representations from physiological signals and models sleep dynamics. DREAM learns sleep related and subject invariant representations from diverse subjects' sleep signal segments and models sleep dynamics by capturing interactions between sequential signal segments and between sleep stages. In the experiments, we demonstrate that DREAM outperforms the existing sleep staging methods on three datasets. The case study demonstrates that our model can learn the generalized decision function resulting in good prediction performances for the new subjects, especially in case there are differences between testing and training subjects. The usage of unlabeled data shows the benefit of leveraging unlabeled EEG data. Further, uncertainty quantification demonstrates that DREAM provides prediction uncertainty, making the model reliable and helping sleep experts in real world applications.

{{</citation>}}


## cs.HC (1)



### (57/108) Data Safety vs. App Privacy: Comparing the Usability of Android and iOS Privacy Labels (Yanzi Lin et al., 2023)

{{<citation>}}

Yanzi Lin, Jaideep Juneja, Eleanor Birrell, Lorrie Cranor. (2023)  
**Data Safety vs. App Privacy: Comparing the Usability of Android and iOS Privacy Labels**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2312.03918v1)  

---


**ABSTRACT**  
Privacy labels -- standardized, compact representations of data collection and data use practices -- have frequently been recommended as a solution to the shortcomings of privacy policies. Apple introduced mandatory privacy labels for apps in its App Store in December 2020; Google introduced data safety labels for Android apps in July 2022. iOS app privacy labels have been evaluated and critiqued in prior work. In this work, we evaluated Android data safety labels and explored how differences between the two label designs impact user comprehension and label utility. We conducted a between-subjects, semi-structured interview study with 12 Android users and 12 iOS users. While some users found Android Data Safety Labels informative and helpful, other users found them too vague. Compared to iOS App Privacy Labels, Android users found the distinction between data collection groups more intuitive and found explicit inclusion of omitted data collection groups more salient. However, some users expressed skepticism regarding elided information about collected data type categories. Most users missed critical information due to not expanding the accordion interface, and they were surprised by collection practices excluded from Android's definitions. Our findings also revealed that Android users generally appreciated information about security practices included in the labels and iOS users wanted that information added.

{{</citation>}}


## cs.CL (18)



### (58/108) Collaboration or Corporate Capture? Quantifying NLP's Reliance on Industry Artifacts and Contributions (Will Aitken et al., 2023)

{{<citation>}}

Will Aitken, Mohamed Abdalla, Karen Rudie, Catherine Stinson. (2023)  
**Collaboration or Corporate Capture? Quantifying NLP's Reliance on Industry Artifacts and Contributions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.03912v1)  

---


**ABSTRACT**  
The advent of transformers, higher computational budgets, and big data has engendered remarkable progress in Natural Language Processing (NLP). Impressive performance of industry pre-trained models has garnered public attention in recent years and made news headlines. That these are industry models is noteworthy. Rarely, if ever, are academic institutes producing exciting new NLP models. Using these models is critical for competing on NLP benchmarks and correspondingly to stay relevant in NLP research. We surveyed 100 papers published at EMNLP 2022 to determine whether this phenomenon constitutes a reliance on industry for NLP publications.   We find that there is indeed a substantial reliance. Citations of industry artifacts and contributions across categories is at least three times greater than industry publication rates per year. Quantifying this reliance does not settle how we ought to interpret the results. We discuss two possible perspectives in our discussion: 1) Is collaboration with industry still collaboration in the absence of an alternative? Or 2) has free NLP inquiry been captured by the motivations and research direction of private corporations?

{{</citation>}}


### (59/108) Efficient Large Language Models: A Survey (Zhongwei Wan et al., 2023)

{{<citation>}}

Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Zhongnan Qu, Shen Yan, Yi Zhu, Quanlu Zhang, Mosharaf Chowdhury, Mi Zhang. (2023)  
**Efficient Large Language Models: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.03863v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable capabilities in important tasks such as natural language understanding, language generation, and complex reasoning and have the potential to make a substantial impact on our society. Such capabilities, however, come with the considerable resources they demand, highlighting the strong need to develop effective techniques for addressing their efficiency challenges. In this survey, we provide a systematic and comprehensive review of efficient LLMs research. We organize the literature in a taxonomy consisting of three main categories, covering distinct yet interconnected efficient LLMs topics from model-centric, data-centric, and framework-centric perspective, respectively. We have also created a GitHub repository where we compile the papers featured in this survey at https://github.com/AIoT-MLSys-Lab/EfficientLLMs, https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey, and will actively maintain this repository and incorporate new research as it emerges. We hope our survey can serve as a valuable resource to help researchers and practitioners gain a systematic understanding of the research developments in efficient LLMs and inspire them to contribute to this important and exciting field.

{{</citation>}}


### (60/108) Evaluating and Mitigating Discrimination in Language Model Decisions (Alex Tamkin et al., 2023)

{{<citation>}}

Alex Tamkin, Amanda Askell, Liane Lovitt, Esin Durmus, Nicholas Joseph, Shauna Kravec, Karina Nguyen, Jared Kaplan, Deep Ganguli. (2023)  
**Evaluating and Mitigating Discrimination in Language Model Decisions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.03689v1)  

---


**ABSTRACT**  
As language models (LMs) advance, interest is growing in applying them to high-stakes societal decisions, such as determining financing or housing eligibility. However, their potential for discrimination in such contexts raises ethical concerns, motivating the need for better methods to evaluate these risks. We present a method for proactively evaluating the potential discriminatory impact of LMs in a wide range of use cases, including hypothetical use cases where they have not yet been deployed. Specifically, we use an LM to generate a wide array of potential prompts that decision-makers may input into an LM, spanning 70 diverse decision scenarios across society, and systematically vary the demographic information in each prompt. Applying this methodology reveals patterns of both positive and negative discrimination in the Claude 2.0 model in select settings when no interventions are applied. While we do not endorse or permit the use of language models to make automated decisions for the high-risk use cases we study, we demonstrate techniques to significantly decrease both positive and negative discrimination through careful prompt engineering, providing pathways toward safer deployment in use cases where they may be appropriate. Our work enables developers and policymakers to anticipate, measure, and address discrimination as language model capabilities and applications continue to expand. We release our dataset and prompts at https://huggingface.co/datasets/Anthropic/discrim-eval

{{</citation>}}


### (61/108) Improving Activation Steering in Language Models with Mean-Centring (Ole Jorgensen et al., 2023)

{{<citation>}}

Ole Jorgensen, Dylan Cope, Nandi Schoots, Murray Shanahan. (2023)  
**Improving Activation Steering in Language Models with Mean-Centring**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.03813v1)  

---


**ABSTRACT**  
Recent work in activation steering has demonstrated the potential to better control the outputs of Large Language Models (LLMs), but it involves finding steering vectors. This is difficult because engineers do not typically know how features are represented in these models. We seek to address this issue by applying the idea of mean-centring to steering vectors. We find that taking the average of activations associated with a target dataset, and then subtracting the mean of all training activations, results in effective steering vectors. We test this method on a variety of models on natural language tasks by steering away from generating toxic text, and steering the completion of a story towards a target genre. We also apply mean-centring to extract function vectors, more effectively triggering the execution of a range of natural language tasks by a significant margin (compared to previous baselines). This suggests that mean-centring can be used to easily improve the effectiveness of activation steering in a wide range of contexts.

{{</citation>}}


### (62/108) Not All Large Language Models (LLMs) Succumb to the 'Reversal Curse': A Comparative Study of Deductive Logical Reasoning in BERT and GPT Models (Jingye Yang et al., 2023)

{{<citation>}}

Jingye Yang, Da Wu, Kai Wang. (2023)  
**Not All Large Language Models (LLMs) Succumb to the 'Reversal Curse': A Comparative Study of Deductive Logical Reasoning in BERT and GPT Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, ChatGPT, GPT, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.03633v1)  

---


**ABSTRACT**  
The "Reversal Curse" refers to the scenario where auto-regressive decoder large language models (LLMs), such as ChatGPT, trained on "A is B" fail to learn "B is A", demonstrating a basic failure of logical deduction. This raises a red flag in the use of GPT models for certain general tasks such as constructing knowledge graphs, considering their adherence to this symmetric principle. In our study, we examined a bidirectional LLM, BERT, and found that it is immune to the reversal curse. Driven by ongoing efforts to construct biomedical knowledge graphs with LLMs, we also embarked on evaluating more complex but essential deductive reasoning capabilities. This process included first training encoder and decoder language models to master the intersection ($\cap$) and union ($\cup$) operations on two sets and then moving on to assess their capability to infer different combinations of union ($\cup$) and intersection ($\cap$) operations on three newly created sets. The findings showed that while both encoder and decoder language models, trained for tasks involving two sets (union/intersection), were proficient in such scenarios, they encountered difficulties when dealing with operations that included three sets (various combinations of union and intersection). Our research highlights the distinct characteristics of encoder and decoder models in simple and complex logical reasoning. In practice, the choice between BERT and GPT should be guided by the specific requirements and nature of the task at hand, leveraging their respective strengths in bidirectional context comprehension and sequence prediction.

{{</citation>}}


### (63/108) Improving Bias Mitigation through Bias Experts in Natural Language Understanding (Eojin Jeon et al., 2023)

{{<citation>}}

Eojin Jeon, Mingyu Lee, Juhyeong Park, Yeachan Kim, Wing-Lam Mok, SangKeun Lee. (2023)  
**Improving Bias Mitigation through Bias Experts in Natural Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Natural Language Understanding  
[Paper Link](http://arxiv.org/abs/2312.03577v1)  

---


**ABSTRACT**  
Biases in the dataset often enable the model to achieve high performance on in-distribution data, while poorly performing on out-of-distribution data. To mitigate the detrimental effect of the bias on the networks, previous works have proposed debiasing methods that down-weight the biased examples identified by an auxiliary model, which is trained with explicit bias labels. However, finding a type of bias in datasets is a costly process. Therefore, recent studies have attempted to make the auxiliary model biased without the guidance (or annotation) of bias labels, by constraining the model's training environment or the capability of the model itself. Despite the promising debiasing results of recent works, the multi-class learning objective, which has been naively used to train the auxiliary model, may harm the bias mitigation effect due to its regularization effect and competitive nature across classes. As an alternative, we propose a new debiasing framework that introduces binary classifiers between the auxiliary model and the main model, coined bias experts. Specifically, each bias expert is trained on a binary classification task derived from the multi-class classification task via the One-vs-Rest approach. Experimental results demonstrate that our proposed strategy improves the bias identification ability of the auxiliary model. Consequently, our debiased model consistently outperforms the state-of-the-art on various challenge datasets.

{{</citation>}}


### (64/108) XAIQA: Explainer-Based Data Augmentation for Extractive Question Answering (Joel Stremmel et al., 2023)

{{<citation>}}

Joel Stremmel, Ardavan Saeedi, Hamid Hassanzadeh, Sanjit Batra, Jeffrey Hertzberg, Jaime Murillo, Eran Halperin. (2023)  
**XAIQA: Explainer-Based Data Augmentation for Extractive Question Answering**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: AI, Augmentation, GPT, GPT-4, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.03567v1)  

---


**ABSTRACT**  
Extractive question answering (QA) systems can enable physicians and researchers to query medical records, a foundational capability for designing clinical studies and understanding patient medical history. However, building these systems typically requires expert-annotated QA pairs. Large language models (LLMs), which can perform extractive QA, depend on high quality data in their prompts, specialized for the application domain. We introduce a novel approach, XAIQA, for generating synthetic QA pairs at scale from data naturally available in electronic health records. Our method uses the idea of a classification model explainer to generate questions and answers about medical concepts corresponding to medical codes. In an expert evaluation with two physicians, our method identifies $2.2\times$ more semantic matches and $3.8\times$ more clinical abbreviations than two popular approaches that use sentence transformers to create QA pairs. In an ML evaluation, adding our QA pairs improves performance of GPT-4 as an extractive QA model, including on difficult questions. In both the expert and ML evaluations, we examine trade-offs between our method and sentence transformers for QA pair generation depending on question difficulty.

{{</citation>}}


### (65/108) Holmes: Towards Distributed Training Across Clusters with Heterogeneous NIC Environment (Fei Yang et al., 2023)

{{<citation>}}

Fei Yang, Shuang Peng, Ning Sun, Fangyu Wang, Ke Tan, Fu Wu, Jiezhong Qiu, Aimin Pan. (2023)  
**Holmes: Towards Distributed Training Across Clusters with Heterogeneous NIC Environment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-DC, cs.CL  
Keywords: GPT, LLaMA  
[Paper Link](http://arxiv.org/abs/2312.03549v2)  

---


**ABSTRACT**  
Large language models (LLMs) such as GPT-3, OPT, and LLaMA have demonstrated remarkable accuracy in a wide range of tasks. However, training these models can incur significant expenses, often requiring tens of thousands of GPUs for months of continuous operation. Typically, this training is carried out in specialized GPU clusters equipped with homogeneous high-speed Remote Direct Memory Access (RDMA) network interface cards (NICs). The acquisition and maintenance of such dedicated clusters is challenging. Current LLM training frameworks, like Megatron-LM and Megatron-DeepSpeed, focus primarily on optimizing training within homogeneous cluster settings. In this paper, we introduce Holmes, a training framework for LLMs that employs thoughtfully crafted data and model parallelism strategies over the heterogeneous NIC environment. Our primary technical contribution lies in a novel scheduling method that intelligently allocates distinct computational tasklets in LLM training to specific groups of GPU devices based on the characteristics of their connected NICs. Furthermore, our proposed framework, utilizing pipeline parallel techniques, demonstrates scalability to multiple GPU clusters, even in scenarios without high-speed interconnects between nodes in distinct clusters. We conducted comprehensive experiments that involved various scenarios in the heterogeneous NIC environment. In most cases, our framework achieves performance levels close to those achievable with homogeneous RDMA-capable networks (InfiniBand or RoCE), significantly exceeding training efficiency within the pure Ethernet environment. Additionally, we verified that our framework outperforms other mainstream LLM frameworks under heterogeneous NIC environment in terms of training efficiency and can be seamlessly integrated with them.

{{</citation>}}


### (66/108) Sig-Networks Toolkit: Signature Networks for Longitudinal Language Modelling (Talia Tseriotou et al., 2023)

{{<citation>}}

Talia Tseriotou, Ryan Sze-Yin Chan, Adam Tsakalidis, Iman Munire Bilal, Elena Kochkina, Terry Lyons, Maria Liakata. (2023)  
**Sig-Networks Toolkit: Signature Networks for Longitudinal Language Modelling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.03523v1)  

---


**ABSTRACT**  
We present an open-source, pip installable toolkit, Sig-Networks, the first of its kind for longitudinal language modelling. A central focus is the incorporation of Signature-based Neural Network models, which have recently shown success in temporal tasks. We apply and extend published research providing a full suite of signature-based models. Their components can be used as PyTorch building blocks in future architectures. Sig-Networks enables task-agnostic dataset plug-in, seamless pre-processing for sequential data, parameter flexibility, automated tuning across a range of models. We examine signature networks under three different NLP tasks of varying temporal granularity: counselling conversations, rumour stance switch and mood changes in social media threads, showing SOTA performance in all three, and provide guidance for future tasks. We release the Toolkit as a PyTorch package with an introductory video, Git repositories for preprocessing and modelling including sample notebooks on the modeled NLP tasks.

{{</citation>}}


### (67/108) Exploring Answer Information Methods for Question Generation with Transformers (Talha Chafekar et al., 2023)

{{<citation>}}

Talha Chafekar, Aafiya Hussain, Grishma Sharma, Deepak Sharma. (2023)  
**Exploring Answer Information Methods for Question Generation with Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Question Generation, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.03483v1)  

---


**ABSTRACT**  
There has been a lot of work in question generation where different methods to provide target answers as input, have been employed. This experimentation has been mostly carried out for RNN based models. We use three different methods and their combinations for incorporating answer information and explore their effect on several automatic evaluation metrics. The methods that are used are answer prompting, using a custom product method using answer embeddings and encoder outputs, choosing sentences from the input paragraph that have answer related information, and using a separate cross-attention attention block in the decoder which attends to the answer. We observe that answer prompting without any additional modes obtains the best scores across rouge, meteor scores. Additionally, we use a custom metric to calculate how many of the generated questions have the same answer, as the answer which is used to generate them.

{{</citation>}}


### (68/108) AMR Parsing is Far from Solved: GrAPES, the Granular AMR Parsing Evaluation Suite (Jonas Groschwitz et al., 2023)

{{<citation>}}

Jonas Groschwitz, Shay B. Cohen, Lucia Donatelli, Meaghan Fowlie. (2023)  
**AMR Parsing is Far from Solved: GrAPES, the Granular AMR Parsing Evaluation Suite**  

---
Primary Category: cs.CL  
Categories: J-5, cs-CL, cs.CL  
Keywords: Abstract Meaning Representation  
[Paper Link](http://arxiv.org/abs/2312.03480v1)  

---


**ABSTRACT**  
We present the Granular AMR Parsing Evaluation Suite (GrAPES), a challenge set for Abstract Meaning Representation (AMR) parsing with accompanying evaluation metrics. AMR parsers now obtain high scores on the standard AMR evaluation metric Smatch, close to or even above reported inter-annotator agreement. But that does not mean that AMR parsing is solved; in fact, human evaluation in previous work indicates that current parsers still quite frequently make errors on node labels or graph structure that substantially distort sentence meaning. Here, we provide an evaluation suite that tests AMR parsers on a range of phenomena of practical, technical, and linguistic interest. Our 36 categories range from seen and unseen labels, to structural generalization, to coreference. GrAPES reveals in depth the abilities and shortcomings of current AMR parsers.

{{</citation>}}


### (69/108) Think from Words(TFW): Initiating Human-Like Cognition in Large Language Models Through Think from Words for Japanese Text-level Classification (Chengguang Gan et al., 2023)

{{<citation>}}

Chengguang Gan, Qinghao Zhang, Tatsunori Mori. (2023)  
**Think from Words(TFW): Initiating Human-Like Cognition in Large Language Models Through Think from Words for Japanese Text-level Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.03458v1)  

---


**ABSTRACT**  
The proliferation of Large Language Models (LLMs) has spurred extensive research into LLM-related Prompt investigations, such as Instruction Learning (IL), In-context Learning (ICL), and Chain-of-Thought (CoT). These approaches aim to improve LLMs' responses by enabling them to provide concise statements or examples for deeper contemplation when addressing questions. However, independent thinking by LLMs can introduce variability in their thought processes, leading to potential inaccuracies. In response, our study seeks to bridge the gap between LLM and human-like thinking processes, recognizing that text comprehension begins with understanding individual words. To tackle this challenge, we have expanded the CoT method to cater to a specific domain. Our approach, known as "Think from Words" (TFW), initiates the comprehension process at the word level and then extends it to encompass the entire text. We also propose "TFW with Extra word-level information" (TFW Extra), augmenting comprehension with additional word-level data. To assess our methods, we employ text classification on six Japanese datasets comprising text-level and word-level elements. Our findings not only validate the effectiveness of TFW but also shed light on the impact of various word-level information types on LLMs' text comprehension, offering insights into their potential to cause misinterpretations and errors in the overall comprehension of the final text.

{{</citation>}}


### (70/108) Comparative Analysis of Multilingual Text Classification & Identification through Deep Learning and Embedding Visualization (Arinjay Wyawhare, 2023)

{{<citation>}}

Arinjay Wyawhare. (2023)  
**Comparative Analysis of Multilingual Text Classification & Identification through Deep Learning and Embedding Visualization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, LSTM, Multilingual, Text Classification, Transformer  
[Paper Link](http://arxiv.org/abs/2312.03789v1)  

---


**ABSTRACT**  
This research conducts a comparative study on multilingual text classification methods, utilizing deep learning and embedding visualization. The study employs LangDetect, LangId, FastText, and Sentence Transformer on a dataset encompassing 17 languages. It explores dimensionality's impact on clustering, revealing FastText's clearer clustering in 2D visualization due to its extensive multilingual corpus training. Notably, the FastText multi-layer perceptron model achieved remarkable accuracy, precision, recall, and F1 score, outperforming the Sentence Transformer model. The study underscores the effectiveness of these techniques in multilingual text classification, emphasizing the importance of large multilingual corpora for training embeddings. It lays the groundwork for future research and assists practitioners in developing language detection and classification systems. Additionally, it includes the comparison of multi-layer perceptron, LSTM, and Convolution models for classification.

{{</citation>}}


### (71/108) A Text-to-Text Model for Multilingual Offensive Language Identification (Tharindu Ranasinghe et al., 2023)

{{<citation>}}

Tharindu Ranasinghe, Marcos Zampieri. (2023)  
**A Text-to-Text Model for Multilingual Offensive Language Identification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Identification, Multilingual, T5  
[Paper Link](http://arxiv.org/abs/2312.03379v1)  

---


**ABSTRACT**  
The ubiquity of offensive content on social media is a growing cause for concern among companies and government organizations. Recently, transformer-based models such as BERT, XLNET, and XLM-R have achieved state-of-the-art performance in detecting various forms of offensive content (e.g. hate speech, cyberbullying, and cyberaggression). However, the majority of these models are limited in their capabilities due to their encoder-only architecture, which restricts the number and types of labels in downstream tasks. Addressing these limitations, this study presents the first pre-trained model with encoder-decoder architecture for offensive language identification with text-to-text transformers (T5) trained on two large offensive language identification datasets; SOLID and CCTK. We investigate the effectiveness of combining two datasets and selecting an optimal threshold in semi-supervised instances in SOLID in the T5 retraining step. Our pre-trained T5 model outperforms other transformer-based models fine-tuned for offensive language detection, such as fBERT and HateBERT, in multiple English benchmarks. Following a similar approach, we also train the first multilingual pre-trained model for offensive language identification using mT5 and evaluate its performance on a set of six different languages (German, Hindi, Korean, Marathi, Sinhala, and Spanish). The results demonstrate that this multilingual model achieves a new state-of-the-art on all the above datasets, showing its usefulness in multilingual scenarios. Our proposed T5-based models will be made freely available to the community.

{{</citation>}}


### (72/108) KhabarChin: Automatic Detection of Important News in the Persian Language (Hamed Hematian Hemati et al., 2023)

{{<citation>}}

Hamed Hematian Hemati, Arash Lagzian, Moein Salimi Sartakhti, Hamid Beigy, Ehsaneddin Asgari. (2023)  
**KhabarChin: Automatic Detection of Important News in the Persian Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.03361v1)  

---


**ABSTRACT**  
Being aware of important news is crucial for staying informed and making well-informed decisions efficiently. Natural Language Processing (NLP) approaches can significantly automate this process. This paper introduces the detection of important news, in a previously unexplored area, and presents a new benchmarking dataset (Khabarchin) for detecting important news in the Persian language. We define important news articles as those deemed significant for a considerable portion of society, capable of influencing their mindset or decision-making. The news articles are obtained from seven different prominent Persian news agencies, resulting in the annotation of 7,869 samples and the creation of the dataset. Two challenges of high disagreement and imbalance between classes were faced, and solutions were provided for them. We also propose several learning-based models, ranging from conventional machine learning to state-of-the-art transformer models, to tackle this task. Furthermore, we introduce the second task of important sentence detection in news articles, as they often come with a significant contextual length that makes it challenging for readers to identify important information. We identify these sentences in a weakly supervised manner.

{{</citation>}}


### (73/108) Teaching Specific Scientific Knowledge into Large Language Models through Additional Training (Kan Hatakeyama-Sato et al., 2023)

{{<citation>}}

Kan Hatakeyama-Sato, Yasuhiko Igarashi, Shun Katakami, Yuta Nabae, Teruaki Hayakawa. (2023)  
**Teaching Specific Scientific Knowledge into Large Language Models through Additional Training**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.03360v1)  

---


**ABSTRACT**  
Through additional training, we explore embedding specialized scientific knowledge into the Llama 2 Large Language Model (LLM). Key findings reveal that effective knowledge integration requires reading texts from multiple perspectives, especially in instructional formats. We utilize text augmentation to tackle the scarcity of specialized texts, including style conversions and translations. Hyperparameter optimization proves crucial, with different size models (7b, 13b, and 70b) reasonably undergoing additional training. Validating our methods, we construct a dataset of 65,000 scientific papers. Although we have succeeded in partially embedding knowledge, the study highlights the complexities and limitations of incorporating specialized information into LLMs, suggesting areas for further improvement.

{{</citation>}}


### (74/108) Measuring Misogyny in Natural Language Generation: Preliminary Results from a Case Study on two Reddit Communities (Aaron J. Snoswell et al., 2023)

{{<citation>}}

Aaron J. Snoswell, Lucinda Nelson, Hao Xue, Flora D. Salim, Nicolas Suzor, Jean Burgess. (2023)  
**Measuring Misogyny in Natural Language Generation: Preliminary Results from a Case Study on two Reddit Communities**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2312.03330v1)  

---


**ABSTRACT**  
Generic `toxicity' classifiers continue to be used for evaluating the potential for harm in natural language generation, despite mounting evidence of their shortcomings. We consider the challenge of measuring misogyny in natural language generation, and argue that generic `toxicity' classifiers are inadequate for this task. We use data from two well-characterised `Incel' communities on Reddit that differ primarily in their degrees of misogyny to construct a pair of training corpora which we use to fine-tune two language models. We show that an open source `toxicity' classifier is unable to distinguish meaningfully between generations from these models. We contrast this with a misogyny-specific lexicon recently proposed by feminist subject-matter experts, demonstrating that, despite the limitations of simple lexicon-based approaches, this shows promise as a benchmark to evaluate language models for misogyny, and that it is sensitive enough to reveal the known differences in these Reddit communities. Our preliminary findings highlight the limitations of a generic approach to evaluating harms, and further emphasise the need for careful benchmark design and selection in natural language evaluation.

{{</citation>}}


### (75/108) Corporate Bankruptcy Prediction with Domain-Adapted BERT (Alex Kim et al., 2023)

{{<citation>}}

Alex Kim, Sangwon Yoon. (2023)  
**Corporate Bankruptcy Prediction with Domain-Adapted BERT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL, econ-GN, q-fin-EC  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.03194v1)  

---


**ABSTRACT**  
This study performs BERT-based analysis, which is a representative contextualized language model, on corporate disclosure data to predict impending bankruptcies. Prior literature on bankruptcy prediction mainly focuses on developing more sophisticated prediction methodologies with financial variables. However, in our study, we focus on improving the quality of input dataset. Specifically, we employ BERT model to perform sentiment analysis on MD&A disclosures. We show that BERT outperforms dictionary-based predictions and Word2Vec-based predictions in terms of adjusted R-square in logistic regression, k-nearest neighbor (kNN-5), and linear kernel support vector machine (SVM). Further, instead of pre-training the BERT model from scratch, we apply self-learning with confidence-based filtering to corporate disclosure data (10-K). We achieve the accuracy rate of 91.56% and demonstrate that the domain adaptation procedure brings a significant improvement in prediction accuracy.

{{</citation>}}


## cs.CY (1)



### (76/108) Deliberative Technology for Alignment (Andrew Konya et al., 2023)

{{<citation>}}

Andrew Konya, Deger Turan, Aviv Ovadya, Lina Qui, Daanish Masood, Flynn Devine, Lisa Schirch, Isabella Roberts, Deliberative Alignment Forum. (2023)  
**Deliberative Technology for Alignment**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-HC, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03893v1)  

---


**ABSTRACT**  
For humanity to maintain and expand its agency into the future, the most powerful systems we create must be those which act to align the future with the will of humanity. The most powerful systems today are massive institutions like governments, firms, and NGOs. Deliberative technology is already being used across these institutions to help align governance and diplomacy with human will, and modern AI is poised to make this technology significantly better. At the same time, the race to superhuman AGI is already underway, and the AI systems it gives rise to may become the most powerful systems of the future. Failure to align the impact of such powerful AI with the will of humanity may lead to catastrophic consequences, while success may unleash abundance. Right now, there is a window of opportunity to use deliberative technology to align the impact of powerful AI with the will of humanity. Moreover, it may be possible to engineer a symbiotic coupling between powerful AI and deliberative alignment systems such that the quality of alignment improves as AI capabilities increase.

{{</citation>}}


## cs.RO (4)



### (77/108) Geometry Matching for Multi-Embodiment Grasping (Maria Attarian et al., 2023)

{{<citation>}}

Maria Attarian, Muhammad Adil Asif, Jingzhou Liu, Ruthrash Hari, Animesh Garg, Igor Gilitschenski, Jonathan Tompson. (2023)  
**Geometry Matching for Multi-Embodiment Grasping**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.03864v1)  

---


**ABSTRACT**  
Many existing learning-based grasping approaches concentrate on a single embodiment, provide limited generalization to higher DoF end-effectors and cannot capture a diverse set of grasp modes. We tackle the problem of grasping using multiple embodiments by learning rich geometric representations for both objects and end-effectors using Graph Neural Networks. Our novel method - GeoMatch - applies supervised learning on grasping data from multiple embodiments, learning end-to-end contact point likelihood maps as well as conditional autoregressive predictions of grasps keypoint-by-keypoint. We compare our method against baselines that support multiple embodiments. Our approach performs better across three end-effectors, while also producing diverse grasps. Examples, including real robot demos, can be found at geo-match.github.io.

{{</citation>}}


### (78/108) MIRACLE: Inverse Reinforcement and Curriculum Learning Model for Human-inspired Mobile Robot Navigation (Nihal Gunukula et al., 2023)

{{<citation>}}

Nihal Gunukula, Kshitij Tiwari, Aniket Bera. (2023)  
**MIRACLE: Inverse Reinforcement and Curriculum Learning Model for Human-inspired Mobile Robot Navigation**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.03651v2)  

---


**ABSTRACT**  
In emergency scenarios, mobile robots must navigate like humans, interpreting stimuli to locate potential victims rapidly without interfering with first responders. Existing socially-aware navigation algorithms face computational and adaptability challenges. To overcome these, we propose a solution, MIRACLE -- an inverse reinforcement and curriculum learning model, that employs gamified learning to gather stimuli-driven human navigational data. This data is then used to train a Deep Inverse Maximum Entropy Reinforcement Learning model, reducing reliance on demonstrator abilities. Testing reveals a low loss of 2.7717 within a 400-sized environment, signifying human-like response replication. Current databases lack comprehensive stimuli-driven data, necessitating our approach. By doing so, we enable robots to navigate emergency situations with human-like perception, enhancing their life-saving capabilities.

{{</citation>}}


### (79/108) Optimal Wildfire Escape Route Planning for Drones under Dynamic Fire and Smoke (Chang Liu et al., 2023)

{{<citation>}}

Chang Liu, Tamas Sziranyi. (2023)  
**Optimal Wildfire Escape Route Planning for Drones under Dynamic Fire and Smoke**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2312.03521v1)  

---


**ABSTRACT**  
In recent years, the increasing prevalence and intensity of wildfires have posed significant challenges to emergency response teams. The utilization of unmanned aerial vehicles (UAVs), commonly known as drones, has shown promise in aiding wildfire management efforts. This work focuses on the development of an optimal wildfire escape route planning system specifically designed for drones, considering dynamic fire and smoke models. First, the location of the source of the wildfire can be well located by information fusion between UAV and satellite, and the road conditions in the vicinity of the fire can be assessed and analyzed using multi-channel remote sensing data. Second, the road network can be extracted and segmented in real time using UAV vision technology, and each road in the road network map can be given priority based on the results of road condition classification. Third, the spread model of dynamic fires calculates the new location of the fire source based on the fire intensity, wind speed and direction, and the radius increases as the wildfire spreads. Smoke is generated around the fire source to create a visual representation of a burning fire. Finally, based on the improved A* algorithm, which considers all the above factors, the UAV can quickly plan an escape route based on the starting and destination locations that avoid the location of the fire source and the area where it is spreading. By considering dynamic fire and smoke models, the proposed system enhances the safety and efficiency of drone operations in wildfire environments.

{{</citation>}}


### (80/108) VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation (Naoki Yokoyama et al., 2023)

{{<citation>}}

Naoki Yokoyama, Sehoon Ha, Dhruv Batra, Jiuguang Wang, Bernadette Bucher. (2023)  
**VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.03275v1)  

---


**ABSTRACT**  
Understanding how humans leverage semantic knowledge to navigate unfamiliar environments and decide where to explore next is pivotal for developing robots capable of human-like search behaviors. We introduce a zero-shot navigation approach, Vision-Language Frontier Maps (VLFM), which is inspired by human reasoning and designed to navigate towards unseen semantic objects in novel environments. VLFM builds occupancy maps from depth observations to identify frontiers, and leverages RGB observations and a pre-trained vision-language model to generate a language-grounded value map. VLFM then uses this map to identify the most promising frontier to explore for finding an instance of a given target object category. We evaluate VLFM in photo-realistic environments from the Gibson, Habitat-Matterport 3D (HM3D), and Matterport 3D (MP3D) datasets within the Habitat simulator. Remarkably, VLFM achieves state-of-the-art results on all three datasets as measured by success weighted by path length (SPL) for the Object Goal Navigation task. Furthermore, we show that VLFM's zero-shot nature enables it to be readily deployed on real-world robots such as the Boston Dynamics Spot mobile manipulation platform. We deploy VLFM on Spot and demonstrate its capability to efficiently navigate to target objects within an office building in the real world, without any prior knowledge of the environment. The accomplishments of VLFM underscore the promising potential of vision-language models in advancing the field of semantic navigation. Videos of real-world deployment can be viewed at naoki.io/vlfm.

{{</citation>}}


## cs.CR (7)



### (81/108) Dr. Jekyll and Mr. Hyde: Two Faces of LLMs (Matteo Gioele Collu et al., 2023)

{{<citation>}}

Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek. (2023)  
**Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.03853v1)  

---


**ABSTRACT**  
This year, we witnessed a rise in the use of Large Language Models, especially when combined with applications like chatbot assistants. Safety mechanisms and specialized training procedures are put in place to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with opposite characteristics as those of the truthful assistants they are supposed to be. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversation followed a role-play style to get the response the assistant was not allowed to provide. By making use of personas, we show that the response that is prohibited is actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. It also introduces several ways of activating such adversarial personas, altogether showing that both chatbots are vulnerable to this kind of attack.

{{</citation>}}


### (82/108) Fed-urlBERT: Client-side Lightweight Federated Transformers for URL Threat Analysis (Yujie Li et al., 2023)

{{<citation>}}

Yujie Li, Yanbin Wang, Haitao Xu, Zhenhao Guo, Fan Zhang, Ruitong Liu, Wenrui Ma. (2023)  
**Fed-urlBERT: Client-side Lightweight Federated Transformers for URL Threat Analysis**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.03636v1)  

---


**ABSTRACT**  
In evolving cyber landscapes, the detection of malicious URLs calls for cooperation and knowledge sharing across domains. However, collaboration is often hindered by concerns over privacy and business sensitivities. Federated learning addresses these issues by enabling multi-clients collaboration without direct data exchange. Unfortunately, if highly expressive Transformer models are used, clients may face intolerable computational burdens, and the exchange of weights could quickly deplete network bandwidth. In this paper, we propose Fed-urlBERT, a federated URL pre-trained model designed to address both privacy concerns and the need for cross-domain collaboration in cybersecurity. Fed-urlBERT leverages split learning to divide the pre-training model into client and server part, so that the client part takes up less extensive computation resources and bandwidth. Our appraoch achieves performance comparable to centralized model under both independently and identically distributed (IID) and two non-IID data scenarios. Significantly, our federated model shows about an 7% decrease in the FPR compared to the centralized model. Additionally, we implement an adaptive local aggregation strategy that mitigates heterogeneity among clients, demonstrating promising performance improvements. Overall, our study validates the applicability of the proposed Transformer federated learning for URL threat analysis, establishing a foundation for real-world collaborative cybersecurity efforts. The source code is accessible at https://github.com/Davidup1/FedURLBERT.

{{</citation>}}


### (83/108) TrustFed: A Reliable Federated Learning Framework with Malicious-Attack Resistance (Hangn Su et al., 2023)

{{<citation>}}

Hangn Su, Jianhong Zhou, Xianhua Niu, Gang Feng. (2023)  
**TrustFed: A Reliable Federated Learning Framework with Malicious-Attack Resistance**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04597v1)  

---


**ABSTRACT**  
As a key technology in 6G research, federated learning (FL) enables collaborative learning among multiple clients while ensuring individual data privacy. However, malicious attackers among the participating clients can intentionally tamper with the training data or the trained model, compromising the accuracy and trustworthiness of the system. To address this issue, in this paper, we propose a hierarchical audit-based FL (HiAudit-FL) framework, with the aim to enhance the reliability and security of the learning process. The hierarchical audit process includes two stages, namely model-audit and parameter-audit. In the model-audit stage, a low-overhead audit method is employed to identify suspicious clients. Subsequently, in the parameter-audit stage, a resource-consuming method is used to detect all malicious clients with higher accuracy among the suspicious ones. Specifically, we execute the model audit method among partial clients for multiple rounds, which is modeled as a partial observation Markov decision process (POMDP) with the aim to enhance the robustness and accountability of the decision-making in complex and uncertain environments. Meanwhile, we formulate the problem of identifying malicious attackers through a multi-round audit as an active sequential hypothesis testing problem and leverage a diffusion model-based AI-Enabled audit selection strategy (ASS) to decide which clients should be audited in each round. To accomplish efficient and effective audit selection, we design a DRL-ASS algorithm by incorporating the ASS in a deep reinforcement learning (DRL) framework. Our simulation results demonstrate that HiAudit-FL can effectively identify and handle potential malicious users accurately, with small system overhead.

{{</citation>}}


### (84/108) Behavioral Authentication for Security and Safety (Cheng Wang et al., 2023)

{{<citation>}}

Cheng Wang, Hao Tang, Hangyu Zhu, Junhan Zheng, Changjun Jiang. (2023)  
**Behavioral Authentication for Security and Safety**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.03429v1)  

---


**ABSTRACT**  
The issues of both system security and safety can be dissected integrally from the perspective of behavioral \emph{appropriateness}. That is, a system is secure or safe can be judged by whether the behavior of certain agent(s) is \emph{appropriate} or not. Specifically, a so-called \emph{appropriate behavior} involves the right agent performing the right actions at the right time under certain conditions. Then, according to different levels of appropriateness and degrees of custodies, behavioral authentication can be graded into three levels, i.e., the authentication of behavioral \emph{Identity}, \emph{Conformity}, and \emph{Benignity}. In a broad sense, for the security and safety issue, behavioral authentication is not only an innovative and promising method due to its inherent advantages but also a critical and fundamental problem due to the ubiquity of behavior generation and the necessity of behavior regulation in any system. By this classification, this review provides a comprehensive examination of the background and preliminaries of behavioral authentication. It further summarizes existing research based on their respective focus areas and characteristics. The challenges confronted by current behavioral authentication methods are analyzed, and potential research directions are discussed to promote the diversified and integrated development of behavioral authentication.

{{</citation>}}


### (85/108) Securing Data Platforms: Strategic Masking Techniques for Privacy and Security for B2B Enterprise Data (Mandar Khoje, 2023)

{{<citation>}}

Mandar Khoje. (2023)  
**Securing Data Platforms: Strategic Masking Techniques for Privacy and Security for B2B Enterprise Data**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.03293v1)  

---


**ABSTRACT**  
In today's digital age, the imperative to protect data privacy and security is a paramount concern, especially for business-to-business (B2B) enterprises that handle sensitive information. These enterprises are increasingly constructing data platforms, which are integrated suites of technology solutions architected for the efficient management, processing, storage, and data analysis. It has become critical to design these data platforms with mechanisms that inherently support data privacy and security, particularly as they encounter the added complexity of safeguarding unstructured data types such as log files and text documents. Within this context, data masking stands out as a vital feature of data platform architecture. It proactively conceals sensitive elements, ensuring data privacy while preserving the information's value for business operations and analytics. This protective measure entails a strategic two-fold process: firstly, accurately pinpointing the sensitive data that necessitates concealment, and secondly, applying sophisticated methods to disguise that data effectively within the data platform infrastructure. This research delves into the nuances of embedding advanced data masking techniques within the very fabric of data platforms and an in-depth exploration of how enterprises can adopt a comprehensive approach toward effective data masking implementation by exploring different identification and anonymization techniques.

{{</citation>}}


### (86/108) A Simple Framework to Enhance the Adversarial Robustness of Deep Learning-based Intrusion Detection System (Xinwei Yuan et al., 2023)

{{<citation>}}

Xinwei Yuan, Shu Han, Wei Huang, Hongliang Ye, Xianglong Kong, Fan Zhang. (2023)  
**A Simple Framework to Enhance the Adversarial Robustness of Deep Learning-based Intrusion Detection System**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2312.03245v1)  

---


**ABSTRACT**  
Deep learning based intrusion detection systems (DL-based IDS) have emerged as one of the best choices for providing security solutions against various network intrusion attacks. However, due to the emergence and development of adversarial deep learning technologies, it becomes challenging for the adoption of DL models into IDS. In this paper, we propose a novel IDS architecture that can enhance the robustness of IDS against adversarial attacks by combining conventional machine learning (ML) models and Deep Learning models. The proposed DLL-IDS consists of three components: DL-based IDS, adversarial example (AE) detector, and ML-based IDS. We first develop a novel AE detector based on the local intrinsic dimensionality (LID). Then, we exploit the low attack transferability between DL models and ML models to find a robust ML model that can assist us in determining the maliciousness of AEs. If the input traffic is detected as an AE, the ML-based IDS will predict the maliciousness of input traffic, otherwise the DL-based IDS will work for the prediction. The fusion mechanism can leverage the high prediction accuracy of DL models and low attack transferability between DL models and ML models to improve the robustness of the whole system. In our experiments, we observe a significant improvement in the prediction performance of the IDS when subjected to adversarial attack, achieving high accuracy with low resource consumption.

{{</citation>}}


### (87/108) FedGeo: Privacy-Preserving User Next Location Prediction with Federated Learning (Chung Park et al., 2023)

{{<citation>}}

Chung Park, Taekyoon Choi, Taesan Kim, Mincheol Cho, Junui Hong, Minsung Choi, Jaegul Choo. (2023)  
**FedGeo: Privacy-Preserving User Next Location Prediction with Federated Learning**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs.CR  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.04594v1)  

---


**ABSTRACT**  
A User Next Location Prediction (UNLP) task, which predicts the next location that a user will move to given his/her trajectory, is an indispensable task for a wide range of applications. Previous studies using large-scale trajectory datasets in a single server have achieved remarkable performance in UNLP task. However, in real-world applications, legal and ethical issues have been raised regarding privacy concerns leading to restrictions against sharing human trajectory datasets to any other server. In response, Federated Learning (FL) has emerged to address the personal privacy issue by collaboratively training multiple clients (i.e., users) and then aggregating them. While previous studies employed FL for UNLP, they are still unable to achieve reliable performance because of the heterogeneity of clients' mobility. To tackle this problem, we propose the Federated Learning for Geographic Information (FedGeo), a FL framework specialized for UNLP, which alleviates the heterogeneity of clients' mobility and guarantees personal privacy protection. Firstly, we incorporate prior global geographic adjacency information to the local client model, since the spatial correlation between locations is trained partially in each client who has only a heterogeneous subset of the overall trajectories in FL. We also introduce a novel aggregation method that minimizes the gap between client models to solve the problem of client drift caused by differences between client models when learning with their heterogeneous data. Lastly, we probabilistically exclude clients with extremely heterogeneous data from the FL process by focusing on clients who visit relatively diverse locations. We show that FedGeo is superior to other FL methods for model performance in UNLP task. We also validated our model in a real-world application using our own customers' mobile phones and the FL agent system.

{{</citation>}}


## physics.data-an (1)



### (88/108) High Pileup Particle Tracking with Object Condensation (Kilian Lieret et al., 2023)

{{<citation>}}

Kilian Lieret, Gage DeZoort, Devdoot Chatterjee, Jian Park, Siqi Miao, Pan Li. (2023)  
**High Pileup Particle Tracking with Object Condensation**  

---
Primary Category: physics.data-an  
Categories: cs-LG, hep-ex, physics-data-an, physics.data-an  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.03823v1)  

---


**ABSTRACT**  
Recent work has demonstrated that graph neural networks (GNNs) can match the performance of traditional algorithms for charged particle tracking while improving scalability to meet the computing challenges posed by the HL-LHC. Most GNN tracking algorithms are based on edge classification and identify tracks as connected components from an initial graph containing spurious connections. In this talk, we consider an alternative based on object condensation (OC), a multi-objective learning framework designed to cluster points (hits) belonging to an arbitrary number of objects (tracks) and regress the properties of each object. Building on our previous results, we present a streamlined model and show progress toward a one-shot OC tracking algorithm in a high-pileup environment.

{{</citation>}}


## cs.OS (1)



### (89/108) LLM as OS (llmao), Agents as Apps: Envisioning AIOS, Agents and the AIOS-Agent Ecosystem (Yingqiang Ge et al., 2023)

{{<citation>}}

Yingqiang Ge, Yujie Ren, Wenyue Hua, Shuyuan Xu, Juntao Tan, Yongfeng Zhang. (2023)  
**LLM as OS (llmao), Agents as Apps: Envisioning AIOS, Agents and the AIOS-Agent Ecosystem**  

---
Primary Category: cs.OS  
Categories: cs-AI, cs-CL, cs-LG, cs-OS, cs.OS  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.03815v1)  

---


**ABSTRACT**  
This paper envisions a revolutionary AIOS-Agent ecosystem, where Large Language Model (LLM) serves as the (Artificial) Intelligent Operating System (IOS, or AIOS)--an operating system ``with soul''. Upon this foundation, a diverse range of LLM-based AI Agent Applications (Agents, or AAPs) are developed, enriching the AIOS-Agent ecosystem and signaling a paradigm shift from the traditional OS-APP ecosystem. We envision that LLM's impact will not be limited to the AI application level, instead, it will in turn revolutionize the design and implementation of computer system, architecture, software, and programming language, featured by several main concepts: LLM as OS (system-level), Agents as Applications (application-level), Natural Language as Programming Interface (user-level), and Tools as Devices/Libraries (hardware/middleware-level).

{{</citation>}}


## eess.AS (2)



### (90/108) An Integration of Pre-Trained Speech and Language Models for End-to-End Speech Recognition (Yukiya Hono et al., 2023)

{{<citation>}}

Yukiya Hono, Koh Mitsuda, Tianyu Zhao, Kentaro Mitsui, Toshiaki Wakatsuki, Kei Sawada. (2023)  
**An Integration of Pre-Trained Speech and Language Models for End-to-End Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-CL, cs-LG, eess-AS, eess.AS  
Keywords: Language Model, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.03668v1)  

---


**ABSTRACT**  
Advances in machine learning have made it possible to perform various text and speech processing tasks, including automatic speech recognition (ASR), in an end-to-end (E2E) manner. Since typical E2E approaches require large amounts of training data and resources, leveraging pre-trained foundation models instead of training from scratch is gaining attention. Although there have been attempts to use pre-trained speech and language models in ASR, most of them are limited to using either. This paper explores the potential of integrating a pre-trained speech representation model with a large language model (LLM) for E2E ASR. The proposed model enables E2E ASR by generating text tokens in an autoregressive manner via speech representations as speech prompts, taking advantage of the vast knowledge provided by the LLM. Furthermore, the proposed model can incorporate remarkable developments for LLM utilization, such as inference optimization and parameter-efficient domain adaptation. Experimental results show that the proposed model achieves performance comparable to modern E2E ASR models.

{{</citation>}}


### (91/108) Golden Gemini is All You Need: Finding the Sweet Spots for Speaker Verification (Tianchi Liu et al., 2023)

{{<citation>}}

Tianchi Liu, Kong Aik Lee, Qiongqiong Wang, Haizhou Li. (2023)  
**Golden Gemini is All You Need: Finding the Sweet Spots for Speaker Verification**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Speaker Verification  
[Paper Link](http://arxiv.org/abs/2312.03620v1)  

---


**ABSTRACT**  
Previous studies demonstrate the impressive performance of residual neural networks (ResNet) in speaker verification. The ResNet models treat the time and frequency dimensions equally. They follow the default stride configuration designed for image recognition, where the horizontal and vertical axes exhibit similarities. This approach ignores the fact that time and frequency are asymmetric in speech representation. In this paper, we address this issue and look for optimal stride configurations specifically tailored for speaker verification. We represent the stride space on a trellis diagram, and conduct a systematic study on the impact of temporal and frequency resolutions on the performance and further identify two optimal points, namely Golden Gemini, which serves as a guiding principle for designing 2D ResNet-based speaker verification models. By following the principle, a state-of-the-art ResNet baseline model gains a significant performance improvement on VoxCeleb, SITW, and CNCeleb datasets with 7.70%/11.76% average EER/minDCF reductions, respectively, across different network depths (ResNet18, 34, 50, and 101), while reducing the number of parameters by 16.5% and FLOPs by 4.1%. We refer to it as Gemini ResNet. Further investigation reveals the efficacy of the proposed Golden Gemini operating points across various training conditions and architectures. Furthermore, we present a new benchmark, namely the Gemini DF-ResNet, using a cutting-edge model.

{{</citation>}}


## cs.AI (3)



### (92/108) Generative agent-based modeling with actions grounded in physical, social, or digital space using Concordia (Alexander Sasha Vezhnevets et al., 2023)

{{<citation>}}

Alexander Sasha Vezhnevets, John P. Agapiou, Avia Aharon, Ron Ziv, Jayd Matyas, Edgar A. Duéñez-Guzmán, William A. Cunningham, Simon Osindero, Danny Karmon, Joel Z. Leibo. (2023)  
**Generative agent-based modeling with actions grounded in physical, social, or digital space using Concordia**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.03664v1)  

---


**ABSTRACT**  
Agent-based modeling has been around for decades, and applied widely across the social and natural sciences. The scope of this research method is now poised to grow dramatically as it absorbs the new affordances provided by Large Language Models (LLM)s. Generative Agent-Based Models (GABM) are not just classic Agent-Based Models (ABM)s where the agents talk to one another. Rather, GABMs are constructed using an LLM to apply common sense to situations, act "reasonably", recall common semantic knowledge, produce API calls to control digital technologies like apps, and communicate both within the simulation and to researchers viewing it from the outside. Here we present Concordia, a library to facilitate constructing and working with GABMs. Concordia makes it easy to construct language-mediated simulations of physically- or digitally-grounded environments. Concordia agents produce their behavior using a flexible component system which mediates between two fundamental operations: LLM calls and associative memory retrieval. A special agent called the Game Master (GM), which was inspired by tabletop role-playing games, is responsible for simulating the environment where the agents interact. Agents take actions by describing what they want to do in natural language. The GM then translates their actions into appropriate implementations. In a simulated physical world, the GM checks the physical plausibility of agent actions and describes their effects. In digital environments simulating technologies such as apps and services, the GM may handle API calls to integrate with external tools such as general AI assistants (e.g., Bard, ChatGPT), and digital apps (e.g., Calendar, Email, Search, etc.). Concordia was designed to support a wide array of applications both in scientific research and for evaluating performance of real digital services by simulating users and/or generating synthetic data.

{{</citation>}}


### (93/108) Active Wildfires Detection and Dynamic Escape Routes Planning for Humans through Information Fusion between Drones and Satellites (Chang Liu et al., 2023)

{{<citation>}}

Chang Liu, Tamas Sziranyi. (2023)  
**Active Wildfires Detection and Dynamic Escape Routes Planning for Humans through Information Fusion between Drones and Satellites**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs.AI  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2312.03519v1)  

---


**ABSTRACT**  
UAVs are playing an increasingly important role in the field of wilderness rescue by virtue of their flexibility. This paper proposes a fusion of UAV vision technology and satellite image analysis technology for active wildfires detection and road networks extraction of wildfire areas and real-time dynamic escape route planning for people in distress. Firstly, the fire source location and the segmentation of smoke and flames are targeted based on Sentinel 2 satellite imagery. Secondly, the road segmentation and the road condition assessment are performed by D-linkNet and NDVI values in the central area of the fire source by UAV. Finally, the dynamic optimal route planning for humans in real time is performed by the weighted A* algorithm in the road network with the dynamic fire spread model. Taking the Chongqing wildfire on August 24, 2022, as a case study, the results demonstrate that the dynamic escape route planning algorithm can provide an optimal real-time navigation path for humans in the presence of fire through the information fusion of UAVs and satellites.

{{</citation>}}


### (94/108) Can language agents be alternatives to PPO? A Preliminary Empirical Study On OpenAI Gym (Junjie Sheng et al., 2023)

{{<citation>}}

Junjie Sheng, Zixiao Huang, Chuyun Shen, Wenhao Li, Yun Hua, Bo Jin, Hongyuan Zha, Xiangfeng Wang. (2023)  
**Can language agents be alternatives to PPO? A Preliminary Empirical Study On OpenAI Gym**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03290v1)  

---


**ABSTRACT**  
The formidable capacity for zero- or few-shot decision-making in language agents encourages us to pose a compelling question: Can language agents be alternatives to PPO agents in traditional sequential decision-making tasks? To investigate this, we first take environments collected in OpenAI Gym as our testbeds and ground them to textual environments that construct the TextGym simulator. This allows for straightforward and efficient comparisons between PPO agents and language agents, given the widespread adoption of OpenAI Gym. To ensure a fair and effective benchmarking, we introduce $5$ levels of scenario for accurate domain-knowledge controlling and a unified RL-inspired framework for language agents. Additionally, we propose an innovative explore-exploit-guided language (EXE) agent to solve tasks within TextGym. Through numerical experiments and ablation studies, we extract valuable insights into the decision-making capabilities of language agents and make a preliminary evaluation of their potential to be alternatives to PPO in classical sequential decision-making problems. This paper sheds light on the performance of language agents and paves the way for future research in this exciting domain. Our code is publicly available at~\url{https://github.com/mail-ecnu/Text-Gym-Agents}.

{{</citation>}}


## eess.IV (2)



### (95/108) Editable Stain Transformation Of Histological Images Using Unpaired GANs (Tibor Sloboda et al., 2023)

{{<citation>}}

Tibor Sloboda, Lukáš Hudec, Wanda Benešová. (2023)  
**Editable Stain Transformation Of Histological Images Using Unpaired GANs**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03647v1)  

---


**ABSTRACT**  
Double staining in histopathology, particularly for metaplastic breast cancer, typically employs H&E and P63 dyes. However, P63's tissue damage and high cost necessitate alternative methods. This study introduces xAI-CycleGAN, an advanced architecture combining Mask CycleGAN with explainability features and structure-preserving capabilities for transforming H&E stained breast tissue images into P63-like images. The architecture allows for output editing, enhancing resemblance to actual images and enabling further model refinement. We showcase xAI-CycleGAN's efficacy in maintaining structural integrity and generating high-quality images. Additionally, a histopathologist survey indicates the generated images' realism is often comparable to actual images, validating our model's high-quality output.

{{</citation>}}


### (96/108) PneumoLLM: Harnessing the Power of Large Language Model for Pneumoconiosis Diagnosis (Meiyue Song et al., 2023)

{{<citation>}}

Meiyue Song, Zhihua Yu, Jiaxin Wang, Jiarui Wang, Yuting Lu, Baicun Li, Xiaoxu Wang, Qinghua Huang, Zhijun Li, Nikolaos I. Kanellakis, Jiangfeng Liu, Jing Wang, Binglu Wang, Juntao Yang. (2023)  
**PneumoLLM: Harnessing the Power of Large Language Model for Pneumoconiosis Diagnosis**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.03490v2)  

---


**ABSTRACT**  
The conventional pretraining-and-finetuning paradigm, while effective for common diseases with ample data, faces challenges in diagnosing data-scarce occupational diseases like pneumoconiosis. Recently, large language models (LLMs) have exhibits unprecedented ability when conducting multiple tasks in dialogue, bringing opportunities to diagnosis. A common strategy might involve using adapter layers for vision-language alignment and diagnosis in a dialogic manner. Yet, this approach often requires optimization of extensive learnable parameters in the text branch and the dialogue head, potentially diminishing the LLMs' efficacy, especially with limited training data. In our work, we innovate by eliminating the text branch and substituting the dialogue head with a classification head. This approach presents a more effective method for harnessing LLMs in diagnosis with fewer learnable parameters. Furthermore, to balance the retention of detailed image information with progression towards accurate diagnosis, we introduce the contextual multi-token engine. This engine is specialized in adaptively generating diagnostic tokens. Additionally, we propose the information emitter module, which unidirectionally emits information from image tokens to diagnosis tokens. Comprehensive experiments validate the superiority of our methods and the effectiveness of proposed modules. Our codes can be found at https://github.com/CodeMonsterPHD/PneumoLLM/tree/main.

{{</citation>}}


## cs.CE (1)



### (97/108) Augmenting optimization-based molecular design with graph neural networks (Shiqiang Zhang et al., 2023)

{{<citation>}}

Shiqiang Zhang, Juan S. Campos, Christian Feldmann, Frederik Sandfort, Miriam Mathea, Ruth Misener. (2023)  
**Augmenting optimization-based molecular design with graph neural networks**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.03613v1)  

---


**ABSTRACT**  
Computer-aided molecular design (CAMD) studies quantitative structure-property relationships and discovers desired molecules using optimization algorithms. With the emergence of machine learning models, CAMD score functions may be replaced by various surrogates to automatically learn the structure-property relationships. Due to their outstanding performance on graph domains, graph neural networks (GNNs) have recently appeared frequently in CAMD. But using GNNs introduces new optimization challenges. This paper formulates GNNs using mixed-integer programming and then integrates this GNN formulation into the optimization and machine learning toolkit OMLT. To characterize and formulate molecules, we inherit the well-established mixed-integer optimization formulation for CAMD and propose symmetry-breaking constraints to remove symmetric solutions caused by graph isomorphism. In two case studies, we investigate fragment-based odorant molecular design with more practical requirements to test the compatibility and performance of our approaches.

{{</citation>}}


## stat.ML (2)



### (98/108) Invariance & Causal Representation Learning: Prospects and Limitations (Simon Bing et al., 2023)

{{<citation>}}

Simon Bing, Jonas Wahl, Urmi Ninad, Jakob Runge. (2023)  
**Invariance & Causal Representation Learning: Prospects and Limitations**  

---
Primary Category: stat.ML  
Categories: cs-AI, cs-LG, stat-ML, stat.ML  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.03580v1)  

---


**ABSTRACT**  
In causal models, a given mechanism is assumed to be invariant to changes of other mechanisms. While this principle has been utilized for inference in settings where the causal variables are observed, theoretical insights when the variables of interest are latent are largely missing. We assay the connection between invariance and causal representation learning by establishing impossibility results which show that invariance alone is insufficient to identify latent causal variables. Together with practical considerations, we use these theoretical findings to highlight the need for additional constraints in order to identify representations by exploiting invariance.

{{</citation>}}


### (99/108) Precision of Individual Shapley Value Explanations (Lars Henry Berge Olsen, 2023)

{{<citation>}}

Lars Henry Berge Olsen. (2023)  
**Precision of Individual Shapley Value Explanations**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-AP, stat-CO, stat-ML, stat.ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03485v1)  

---


**ABSTRACT**  
Shapley values are extensively used in explainable artificial intelligence (XAI) as a framework to explain predictions made by complex machine learning (ML) models. In this work, we focus on conditional Shapley values for predictive models fitted to tabular data and explain the prediction $f(\boldsymbol{x}^{*})$ for a single observation $\boldsymbol{x}^{*}$ at the time. Numerous Shapley value estimation methods have been proposed and empirically compared on an average basis in the XAI literature. However, less focus has been devoted to analyzing the precision of the Shapley value explanations on an individual basis. We extend our work in Olsen et al. (2023) by demonstrating and discussing that the explanations are systematically less precise for observations on the outer region of the training data distribution for all used estimation methods. This is expected from a statistical point of view, but to the best of our knowledge, it has not been systematically addressed in the Shapley value literature. This is crucial knowledge for Shapley values practitioners, who should be more careful in applying these observations' corresponding Shapley value explanations.

{{</citation>}}


## stat.ME (1)



### (100/108) Blueprinting the Future: Automatic Item Categorization using Hierarchical Zero-Shot and Few-Shot Classifiers (Ting Wang et al., 2023)

{{<citation>}}

Ting Wang, Keith Stelter, Jenn Floyd, Thomas O'Neill, Nathaniel Hendrix, Andrew Bazemore, Kevin Rode, Warren Newton. (2023)  
**Blueprinting the Future: Automatic Item Categorization using Hierarchical Zero-Shot and Few-Shot Classifiers**  

---
Primary Category: stat.ME  
Categories: cs-CY, cs-LG, stat-ME, stat.ME  
Keywords: Few-Shot, GPT, Transformer, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.03561v1)  

---


**ABSTRACT**  
In testing industry, precise item categorization is pivotal to align exam questions with the designated content domains outlined in the assessment blueprint. Traditional methods either entail manual classification, which is laborious and error-prone, or utilize machine learning requiring extensive training data, often leading to model underfit or overfit issues. This study unveils a novel approach employing the zero-shot and few-shot Generative Pretrained Transformer (GPT) classifier for hierarchical item categorization, minimizing the necessity for training data, and instead, leveraging human-like language descriptions to define categories. Through a structured python dictionary, the hierarchical nature of examination blueprints is navigated seamlessly, allowing for a tiered classification of items across multiple levels. An initial simulation with artificial data demonstrates the efficacy of this method, achieving an average accuracy of 92.91% measured by the F1 score. This method was further applied to real exam items from the 2022 In-Training Examination (ITE) conducted by the American Board of Family Medicine (ABFM), reclassifying 200 items according to a newly formulated blueprint swiftly in 15 minutes, a task that traditionally could span several days among editors and physicians. This innovative approach not only drastically cuts down classification time but also ensures a consistent, principle-driven categorization, minimizing human biases and discrepancies. The ability to refine classifications by adjusting definitions adds to its robustness and sustainability.

{{</citation>}}


## cs.AR (1)



### (101/108) MCAIMem: a Mixed SRAM and eDRAM Cell for Area and Energy-efficient on-chip AI Memory (Duy-Thanh Nguyen et al., 2023)

{{<citation>}}

Duy-Thanh Nguyen, Abhiroop Bhattacharjee, Abhishek Moitra, Priyadarshini Panda. (2023)  
**MCAIMem: a Mixed SRAM and eDRAM Cell for Area and Energy-efficient on-chip AI Memory**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03559v1)  

---


**ABSTRACT**  
AI chips commonly employ SRAM memory as buffers for their reliability and speed, which contribute to high performance. However, SRAM is expensive and demands significant area and energy consumption. Previous studies have explored replacing SRAM with emerging technologies like non-volatile memory, which offers fast-read memory access and a small cell area. Despite these advantages, non-volatile memory's slow write memory access and high write energy consumption prevent it from surpassing SRAM performance in AI applications with extensive memory access requirements. Some research has also investigated eDRAM as an area-efficient on-chip memory with similar access times as SRAM. Still, refresh power remains a concern, leaving the trade-off between performance, area, and power consumption unresolved. To address this issue, our paper presents a novel mixed CMOS cell memory design that balances performance, area, and energy efficiency for AI memory by combining SRAM and eDRAM cells. We consider the proportion ratio of one SRAM and seven eDRAM cells in the memory to achieve area reduction using mixed CMOS cell memory. Additionally, we capitalize on the characteristics of DNN data representation and integrate asymmetric eDRAM cells to lower energy consumption. To validate our proposed MCAIMem solution, we conduct extensive simulations and benchmarking against traditional SRAM. Our results demonstrate that MCAIMem significantly outperforms these alternatives in terms of area and energy efficiency. Specifically, our MCAIMem can reduce the area by 48\% and energy consumption by 3.4$\times$ compared to SRAM designs, without incurring any accuracy loss.

{{</citation>}}


## quant-ph (1)



### (102/108) Clustering by Contour coreset and variational quantum eigensolver (Canaan Yung et al., 2023)

{{<citation>}}

Canaan Yung, Muhammad Usman. (2023)  
**Clustering by Contour coreset and variational quantum eigensolver**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.03516v1)  

---


**ABSTRACT**  
Recent work has proposed solving the k-means clustering problem on quantum computers via the Quantum Approximate Optimization Algorithm (QAOA) and coreset techniques. Although the current method demonstrates the possibility of quantum k-means clustering, it does not ensure high accuracy and consistency across a wide range of datasets. The existing coreset techniques are designed for classical algorithms and there has been no quantum-tailored coreset technique which is designed to boost the accuracy of quantum algorithms. In this work, we propose solving the k-means clustering problem with the variational quantum eigensolver (VQE) and a customised coreset method, the Contour coreset, which has been formulated with specific focus on quantum algorithms. Extensive simulations with synthetic and real-life data demonstrated that our VQE+Contour Coreset approach outperforms existing QAOA+Coreset k-means clustering approaches with higher accuracy and lower standard deviation. Our work has shown that quantum tailored coreset techniques has the potential to significantly boost the performance of quantum algorithms when compared to using generic off-the-shelf coreset techniques.

{{</citation>}}


## cs.IR (1)



### (103/108) Boosting legal case retrieval by query content selection with large language models (Youchao Zhou et al., 2023)

{{<citation>}}

Youchao Zhou, Heyan Huang, Zhijing Wu. (2023)  
**Boosting legal case retrieval by query content selection with large language models**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2312.03494v1)  

---


**ABSTRACT**  
Legal case retrieval, which aims to retrieve relevant cases to a given query case, benefits judgment justice and attracts increasing attention. Unlike generic retrieval queries, legal case queries are typically long and the definition of relevance is closely related to legal-specific elements. Therefore, legal case queries may suffer from noise and sparsity of salient content, which hinders retrieval models from perceiving correct information in a query. While previous studies have paid attention to improving retrieval models and understanding relevance judgments, we focus on enhancing legal case retrieval by utilizing the salient content in legal case queries. We first annotate the salient content in queries manually and investigate how sparse and dense retrieval models attend to those content. Then we experiment with various query content selection methods utilizing large language models (LLMs) to extract or summarize salient content and incorporate it into the retrieval models. Experimental results show that reformulating long queries using LLMs improves the performance of both sparse and dense models in legal case retrieval.

{{</citation>}}


## cs.SD (1)



### (104/108) JAMMIN-GPT: Text-based Improvisation using LLMs in Ableton Live (Sven Hollowell et al., 2023)

{{<citation>}}

Sven Hollowell, Tashi Namgyal, Paul Marshall. (2023)  
**JAMMIN-GPT: Text-based Improvisation using LLMs in Ableton Live**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-HC, cs-SD, cs.SD, eess-AS  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.03479v1)  

---


**ABSTRACT**  
We introduce a system that allows users of Ableton Live to create MIDI-clips by naming them with musical descriptions. Users can compose by typing the desired musical content directly in Ableton's clip view, which is then inserted by our integrated system. This allows users to stay in the flow of their creative process while quickly generating musical ideas. The system works by prompting ChatGPT to reply using one of several text-based musical formats, such as ABC notation, chord symbols, or drum tablature. This is an important step in integrating generative AI tools into pre-existing musical workflows, and could be valuable for content makers who prefer to express their creative vision through descriptive language. Code is available at https://github.com/supersational/JAMMIN-GPT.

{{</citation>}}


## cond-mat.soft (1)



### (105/108) An AI for Scientific Discovery Route between Amorphous Networks and Mechanical Behavior (Changliang Zhu et al., 2023)

{{<citation>}}

Changliang Zhu, Chenchao Fang, Zhipeng Jin, Baowen Li, Xiangying Shen, Lei Xu. (2023)  
**An AI for Scientific Discovery Route between Amorphous Networks and Mechanical Behavior**  

---
Primary Category: cond-mat.soft  
Categories: cond-mat-soft, cond-mat.soft, cs-LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.03404v1)  

---


**ABSTRACT**  
"AI for science" is widely recognized as a future trend in the development of scientific research. Currently, although machine learning algorithms have played a crucial role in scientific research with numerous successful cases, relatively few instances exist where AI assists researchers in uncovering the underlying physical mechanisms behind a certain phenomenon and subsequently using that mechanism to improve machine learning algorithms' efficiency. This article uses the investigation into the relationship between extreme Poisson's ratio values and the structure of amorphous networks as a case study to illustrate how machine learning methods can assist in revealing underlying physical mechanisms. Upon recognizing that the Poisson's ratio relies on the low-frequency vibrational modes of dynamical matrix, we can then employ a convolutional neural network, trained on the dynamical matrix instead of traditional image recognition, to predict the Poisson's ratio of amorphous networks with a much higher efficiency. Through this example, we aim to showcase the role that artificial intelligence can play in revealing fundamental physical mechanisms, which subsequently improves the machine learning algorithms significantly.

{{</citation>}}


## cs.SI (2)



### (106/108) Public emotional dynamics toward AIGC content generation across social media platform (Qinglan Wei et al., 2023)

{{<citation>}}

Qinglan Wei, Jiayi Li, Yuan Zhang. (2023)  
**Public emotional dynamics toward AIGC content generation across social media platform**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.03779v1)  

---


**ABSTRACT**  
Given the widespread popularity of interactive AI models like ChatGPT, public opinion on emerging artificial intelligence generated content(AIGC) has been extensively debated. Pessimists believe that AIGC will replace humans in the future, and optimists think that it will further liberate productivity. Public emotions play a crucial role on social media platforms. They can provide valuable insights into the public's opinions, attitudes, and behaviors. There is a lack of research on the analysis of social group emotions triggered by AIGC content, and even more on the cross-platform differences of group emotions. This study fills the research gap by connecting the theory of group dynamics with emotions in social media. Specifically, we develop a scientific group emotion calculation and visualization system based on chains of communication. The system is capable of crawling data in real time and presenting the current state of group emotions in a fine-grained manner. We then analyze which group dynamic factors drive different public emotions towards nine AIGC products on the three most popular social media platforms in China. Finally, we obtain four main findings. First, Douyin is the only platform with negative group emotion on emerging AI technologies. Second, Weibo users prefer extreme emotions more than others. Third, the group emotion varies by education and age. It is negatively correlated with senior high school or lower and 25 or younger, and positively correlated with bachelor's degree or higher and 26-35. Fourth, the group emotion polarization increases with more posts without comments and celebrity publishers. By analyzing the key dynamic factors of group emotions to AIGC on various social media platforms, we can improve our products and services, develop more effective marketing strategies, and create more accurate and effective AI models to solve complex problems.

{{</citation>}}


### (107/108) Masking Behaviors in Epidemiological Networks with Cognitively-plausible Reinforcement Learning (Konstantinos Mitsopoulos et al., 2023)

{{<citation>}}

Konstantinos Mitsopoulos, Lawrence Baker, Christian Lebiere, Peter Pirolli, Mark Orr, Raffaele Vardavas. (2023)  
**Masking Behaviors in Epidemiological Networks with Cognitively-plausible Reinforcement Learning**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-soc-ph  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.03301v1)  

---


**ABSTRACT**  
The COVID-19 pandemic highlighted the critical role of human behavior in influencing infectious disease transmission and the need for models capturing this complex dynamic. We present an agent-based model integrating an epidemiological simulation of disease spread with a cognitive architecture driving individual mask-wearing decisions. Agents decide whether to mask based on a utility function weighting factors like peer conformity, personal risk tolerance, and mask-wearing discomfort. By conducting experiments systematically varying behavioral model parameters and social network structures, we demonstrate how adaptive decision-making interacts with network connectivity patterns to impact population-level infection outcomes. The model provides a flexible computational framework for gaining insights into how behavioral interventions like mask mandates may differentially influence disease spread across communities with diverse social structures. Findings highlight the importance of integrating realistic human decision processes in epidemiological models to inform policy decisions during public health crises.

{{</citation>}}


## cs.DC (1)



### (108/108) HEET: A Heterogeneity Measure to Quantify the Difference across Distributed Computing Systems (Ali Mokhtari et al., 2023)

{{<citation>}}

Ali Mokhtari, Saeid Ghafouri, Pooyan Jamshidi, Mohsen Amini Salehi. (2023)  
**HEET: A Heterogeneity Measure to Quantify the Difference across Distributed Computing Systems**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-PF, cs.DC  
Keywords: AWS  
[Paper Link](http://arxiv.org/abs/2312.03235v1)  

---


**ABSTRACT**  
Although system heterogeneity has been extensively studied in the past, there is yet to be a study on measuring the impact of heterogeneity on system performance. For this purpose, we propose a heterogeneity measure that can characterize the impact of the heterogeneity of a system on its performance behavior in terms of throughput or makespan. We develop a mathematical model to characterize a heterogeneous system in terms of its task and machine heterogeneity dimensions and then reduce it to a single value, called Homogeneous Equivalent Execution Time (HEET), which represents the execution time behavior of the entire system. We used AWS EC2 instances to implement a real-world machine learning inference system. Performance evaluation of the HEET score across different heterogeneous system configurations demonstrates that HEET can accurately characterize the performance behavior of these systems. In particular, the results show that our proposed method is capable of predicting the true makespan of heterogeneous systems without online evaluations with an average precision of 84%. This heterogeneity measure is instrumental for solution architects to configure their systems proactively to be sufficiently heterogeneous to meet their desired performance objectives.

{{</citation>}}
