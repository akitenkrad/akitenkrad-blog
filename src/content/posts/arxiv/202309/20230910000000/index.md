---
draft: false
title: "arXiv @ 2023.09.10"
date: 2023-09-10
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.10"
    identifier: arxiv_20230910
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (12)](#cscv-12)
- [cs.CL (12)](#cscl-12)
- [cs.LG (7)](#cslg-7)
- [math.PR (1)](#mathpr-1)
- [cs.RO (3)](#csro-3)
- [cs.AI (2)](#csai-2)
- [cs.NI (1)](#csni-1)
- [cs.SD (1)](#cssd-1)
- [cs.IT (1)](#csit-1)
- [cs.SE (1)](#csse-1)
- [eess.IV (1)](#eessiv-1)

## cs.CV (12)



### (1/42) Generalized Cross-domain Multi-label Few-shot Learning for Chest X-rays (Aroof Aimen et al., 2023)

{{<citation>}}

Aroof Aimen, Arsh Verma, Makarand Tapaswi, Narayanan C. Krishnan. (2023)  
**Generalized Cross-domain Multi-label Few-shot Learning for Chest X-rays**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.04462v1)  

---


**ABSTRACT**  
Real-world application of chest X-ray abnormality classification requires dealing with several challenges: (i) limited training data; (ii) training and evaluation sets that are derived from different domains; and (iii) classes that appear during training may have partial overlap with classes of interest during evaluation. To address these challenges, we present an integrated framework called Generalized Cross-Domain Multi-Label Few-Shot Learning (GenCDML-FSL). The framework supports overlap in classes during training and evaluation, cross-domain transfer, adopts meta-learning to learn using few training samples, and assumes each chest X-ray image is either normal or associated with one or more abnormalities. Furthermore, we propose Generalized Episodic Training (GenET), a training strategy that equips models to operate with multiple challenges observed in the GenCDML-FSL scenario. Comparisons with well-established methods such as transfer learning, hybrid transfer learning, and multi-label meta-learning on multiple datasets show the superiority of our approach.

{{</citation>}}


### (2/42) Language Prompt for Autonomous Driving (Dongming Wu et al., 2023)

{{<citation>}}

Dongming Wu, Wencheng Han, Tiancai Wang, Yingfei Liu, Xiangyu Zhang, Jianbing Shen. (2023)  
**Language Prompt for Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.04379v1)  

---


**ABSTRACT**  
A new trend in the computer vision community is to capture objects of interest following flexible human command represented by a natural language prompt. However, the progress of using language prompts in driving scenarios is stuck in a bottleneck due to the scarcity of paired prompt-instance data. To address this challenge, we propose the first object-centric language prompt set for driving scenes within 3D, multi-view, and multi-frame space, named NuPrompt. It expands Nuscenes dataset by constructing a total of 35,367 language descriptions, each referring to an average of 5.3 object tracks. Based on the object-text pairs from the new benchmark, we formulate a new prompt-based driving task, \ie, employing a language prompt to predict the described object trajectory across views and frames. Furthermore, we provide a simple end-to-end baseline model based on Transformer, named PromptTrack. Experiments show that our PromptTrack achieves impressive performance on NuPrompt. We hope this work can provide more new insights for the autonomous driving community. Dataset and Code will be made public at \href{https://github.com/wudongming97/Prompt4Driving}{https://github.com/wudongming97/Prompt4Driving}.

{{</citation>}}


### (3/42) MoEController: Instruction-based Arbitrary Image Manipulation with Mixture-of-Expert Controllers (Sijia Li et al., 2023)

{{<citation>}}

Sijia Li, Chen Chen, Haonan Lu. (2023)  
**MoEController: Instruction-based Arbitrary Image Manipulation with Mixture-of-Expert Controllers**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.04372v1)  

---


**ABSTRACT**  
Diffusion-model-based text-guided image generation has recently made astounding progress, producing fascinating results in open-domain image manipulation tasks. Few models, however, currently have complete zero-shot capabilities for both global and local image editing due to the complexity and diversity of image manipulation tasks. In this work, we propose a method with a mixture-of-expert (MOE) controllers to align the text-guided capacity of diffusion models with different kinds of human instructions, enabling our model to handle various open-domain image manipulation tasks with natural language instructions. First, we use large language models (ChatGPT) and conditional image synthesis models (ControlNet) to generate a large number of global image transfer dataset in addition to the instruction-based local image editing dataset. Then, using an MOE technique and task-specific adaptation training on a large-scale dataset, our conditional diffusion model can edit images globally and locally. Extensive experiments demonstrate that our approach performs surprisingly well on various image manipulation tasks when dealing with open-domain images and arbitrary human instructions. Please refer to our project page: [https://oppo-mente-lab.github.io/moe_controller/]

{{</citation>}}


### (4/42) CNN Injected Transformer for Image Exposure Correction (Shuning Xu et al., 2023)

{{<citation>}}

Shuning Xu, Xiangyu Chen, Binbin Song, Jiantao Zhou. (2023)  
**CNN Injected Transformer for Image Exposure Correction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.04366v1)  

---


**ABSTRACT**  
Capturing images with incorrect exposure settings fails to deliver a satisfactory visual experience. Only when the exposure is properly set, can the color and details of the images be appropriately preserved. Previous exposure correction methods based on convolutions often produce exposure deviation in images as a consequence of the restricted receptive field of convolutional kernels. This issue arises because convolutions are not capable of capturing long-range dependencies in images accurately. To overcome this challenge, we can apply the Transformer to address the exposure correction problem, leveraging its capability in modeling long-range dependencies to capture global representation. However, solely relying on the window-based Transformer leads to visually disturbing blocking artifacts due to the application of self-attention in small patches. In this paper, we propose a CNN Injected Transformer (CIT) to harness the individual strengths of CNN and Transformer simultaneously. Specifically, we construct the CIT by utilizing a window-based Transformer to exploit the long-range interactions among different regions in the entire image. Within each CIT block, we incorporate a channel attention block (CAB) and a half-instance normalization block (HINB) to assist the window-based self-attention to acquire the global statistics and refine local features. In addition to the hybrid architecture design for exposure correction, we apply a set of carefully formulated loss functions to improve the spatial coherence and rectify potential color deviations. Extensive experiments demonstrate that our image exposure correction method outperforms state-of-the-art approaches in terms of both quantitative and qualitative metrics.

{{</citation>}}


### (5/42) Mobile V-MoEs: Scaling Down Vision Transformers via Sparse Mixture-of-Experts (Erik Daxberger et al., 2023)

{{<citation>}}

Erik Daxberger, Floris Weers, Bowen Zhang, Tom Gunter, Ruoming Pang, Marcin Eichner, Michael Emmersberger, Yinfei Yang, Alexander Toshev, Xianzhi Du. (2023)  
**Mobile V-MoEs: Scaling Down Vision Transformers via Sparse Mixture-of-Experts**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, stat-ML  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.04354v1)  

---


**ABSTRACT**  
Sparse Mixture-of-Experts models (MoEs) have recently gained popularity due to their ability to decouple model size from inference efficiency by only activating a small subset of the model parameters for any given input token. As such, sparse MoEs have enabled unprecedented scalability, resulting in tremendous successes across domains such as natural language processing and computer vision. In this work, we instead explore the use of sparse MoEs to scale-down Vision Transformers (ViTs) to make them more attractive for resource-constrained vision applications. To this end, we propose a simplified and mobile-friendly MoE design where entire images rather than individual patches are routed to the experts. We also propose a stable MoE training procedure that uses super-class information to guide the router. We empirically show that our sparse Mobile Vision MoEs (V-MoEs) can achieve a better trade-off between performance and efficiency than the corresponding dense ViTs. For example, for the ViT-Tiny model, our Mobile V-MoE outperforms its dense counterpart by 3.39% on ImageNet-1k. For an even smaller ViT variant with only 54M FLOPs inference cost, our MoE achieves an improvement of 4.66%.

{{</citation>}}


### (6/42) AMLP:Adaptive Masking Lesion Patches for Self-supervised Medical Image Segmentation (Xiangtao Wang et al., 2023)

{{<citation>}}

Xiangtao Wang, Ruizhi Wang, Jie Zhou, Thomas Lukasiewicz, Zhenghua Xu. (2023)  
**AMLP:Adaptive Masking Lesion Patches for Self-supervised Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.04312v1)  

---


**ABSTRACT**  
Self-supervised masked image modeling has shown promising results on natural images. However, directly applying such methods to medical images remains challenging. This difficulty stems from the complexity and distinct characteristics of lesions compared to natural images, which impedes effective representation learning. Additionally, conventional high fixed masking ratios restrict reconstructing fine lesion details, limiting the scope of learnable information. To tackle these limitations, we propose a novel self-supervised medical image segmentation framework, Adaptive Masking Lesion Patches (AMLP). Specifically, we design a Masked Patch Selection (MPS) strategy to identify and focus learning on patches containing lesions. Lesion regions are scarce yet critical, making their precise reconstruction vital. To reduce misclassification of lesion and background patches caused by unsupervised clustering in MPS, we introduce an Attention Reconstruction Loss (ARL) to focus on hard-to-reconstruct patches likely depicting lesions. We further propose a Category Consistency Loss (CCL) to refine patch categorization based on reconstruction difficulty, strengthening distinction between lesions and background. Moreover, we develop an Adaptive Masking Ratio (AMR) strategy that gradually increases the masking ratio to expand reconstructible information and improve learning. Extensive experiments on two medical segmentation datasets demonstrate AMLP's superior performance compared to existing self-supervised approaches. The proposed strategies effectively address limitations in applying masked modeling to medical images, tailored to capturing fine lesion details vital for segmentation tasks.

{{</citation>}}


### (7/42) Have We Ever Encountered This Before? Retrieving Out-of-Distribution Road Obstacles from Driving Scenes (Youssef Shoeb et al., 2023)

{{<citation>}}

Youssef Shoeb, Robin Chan, Gesina Schwalbe, Azarm Nowzard, Fatma Güney, Hanno Gottschalk. (2023)  
**Have We Ever Encountered This Before? Retrieving Out-of-Distribution Road Obstacles from Driving Scenes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.04302v1)  

---


**ABSTRACT**  
In the life cycle of highly automated systems operating in an open and dynamic environment, the ability to adjust to emerging challenges is crucial. For systems integrating data-driven AI-based components, rapid responses to deployment issues require fast access to related data for testing and reconfiguration. In the context of automated driving, this especially applies to road obstacles that were not included in the training data, commonly referred to as out-of-distribution (OoD) road obstacles. Given the availability of large uncurated recordings of driving scenes, a pragmatic approach is to query a database to retrieve similar scenarios featuring the same safety concerns due to OoD road obstacles. In this work, we extend beyond identifying OoD road obstacles in video streams and offer a comprehensive approach to extract sequences of OoD road obstacles using text queries, thereby proposing a way of curating a collection of OoD data for subsequent analysis. Our proposed method leverages the recent advances in OoD segmentation and multi-modal foundation models to identify and efficiently extract safety-relevant scenes from unlabeled videos. We present a first approach for the novel task of text-based OoD object retrieval, which addresses the question ''Have we ever encountered this before?''.

{{</citation>}}


### (8/42) Context-Aware Prompt Tuning for Vision-Language Model with Dual-Alignment (Hongyu Hu et al., 2023)

{{<citation>}}

Hongyu Hu, Tiancheng Lin, Jie Wang, Zhenbang Sun, Yi Xu. (2023)  
**Context-Aware Prompt Tuning for Vision-Language Model with Dual-Alignment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.04158v1)  

---


**ABSTRACT**  
Large-scale vision-language models (VLMs), e.g., CLIP, learn broad visual concepts from tedious training data, showing superb generalization ability. Amount of prompt learning methods have been proposed to efficiently adapt the VLMs to downstream tasks with only a few training samples. We introduce a novel method to improve the prompt learning of vision-language models by incorporating pre-trained large language models (LLMs), called Dual-Aligned Prompt Tuning (DuAl-PT). Learnable prompts, like CoOp, implicitly model the context through end-to-end training, which are difficult to control and interpret. While explicit context descriptions generated by LLMs, like GPT-3, can be directly used for zero-shot classification, such prompts are overly relying on LLMs and still underexplored in few-shot domains. With DuAl-PT, we propose to learn more context-aware prompts, benefiting from both explicit and implicit context modeling. To achieve this, we introduce a pre-trained LLM to generate context descriptions, and we encourage the prompts to learn from the LLM's knowledge by alignment, as well as the alignment between prompts and local image features. Empirically, DuAl-PT achieves superior performance on 11 downstream datasets on few-shot recognition and base-to-new generalization. Hopefully, DuAl-PT can serve as a strong baseline. Code will be available.

{{</citation>}}


### (9/42) Representation Synthesis by Probabilistic Many-Valued Logic Operation in Self-Supervised Learning (Hiroki Nakamura et al., 2023)

{{<citation>}}

Hiroki Nakamura, Masashi Okada, Tadahiro Taniguchi. (2023)  
**Representation Synthesis by Probabilistic Many-Valued Logic Operation in Self-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.04148v1)  

---


**ABSTRACT**  
Self-supervised learning (SSL) using mixed images has been studied to learn various image representations. Existing methods using mixed images learn a representation by maximizing the similarity between the representation of the mixed image and the synthesized representation of the original images. However, few methods consider the synthesis of representations from the perspective of mathematical logic. In this study, we focused on a synthesis method of representations. We proposed a new SSL with mixed images and a new representation format based on many-valued logic. This format can indicate the feature-possession degree, that is, how much of each image feature is possessed by a representation. This representation format and representation synthesis by logic operation realize that the synthesized representation preserves the remarkable characteristics of the original representations. Our method performed competitively with previous representation synthesis methods for image classification tasks. We also examined the relationship between the feature-possession degree and the number of classes of images in the multilabel image classification dataset to verify that the intended learning was achieved. In addition, we discussed image retrieval, which is an application of our proposed representation format using many-valued logic.

{{</citation>}}


### (10/42) Robot Localization and Mapping Final Report -- Sequential Adversarial Learning for Self-Supervised Deep Visual Odometry (Akankshya Kar et al., 2023)

{{<citation>}}

Akankshya Kar, Sajal Maheshwari, Shamit Lal, Vinay Sameer Raja Kad. (2023)  
**Robot Localization and Mapping Final Report -- Sequential Adversarial Learning for Self-Supervised Deep Visual Odometry**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.04147v1)  

---


**ABSTRACT**  
Visual odometry (VO) and SLAM have been using multi-view geometry via local structure from motion for decades. These methods have a slight disadvantage in challenging scenarios such as low-texture images, dynamic scenarios, etc. Meanwhile, use of deep neural networks to extract high level features is ubiquitous in computer vision. For VO, we can use these deep networks to extract depth and pose estimates using these high level features. The visual odometry task then can be modeled as an image generation task where the pose estimation is the by-product. This can also be achieved in a self-supervised manner, thereby eliminating the data (supervised) intensive nature of training deep neural networks. Although some works tried the similar approach [1], the depth and pose estimation in the previous works are vague sometimes resulting in accumulation of error (drift) along the trajectory. The goal of this work is to tackle these limitations of past approaches and to develop a method that can provide better depths and pose estimates. To address this, a couple of approaches are explored: 1) Modeling: Using optical flow and recurrent neural networks (RNN) in order to exploit spatio-temporal correlations which can provide more information to estimate depth. 2) Loss function: Generative adversarial network (GAN) [2] is deployed to improve the depth estimation (and thereby pose too), as shown in Figure 1. This additional loss term improves the realism in generated images and reduces artifacts.

{{</citation>}}


### (11/42) From Text to Mask: Localizing Entities Using the Attention of Text-to-Image Diffusion Models (Changming Xiao et al., 2023)

{{<citation>}}

Changming Xiao, Qi Yang, Feng Zhou, Changshui Zhang. (2023)  
**From Text to Mask: Localizing Entities Using the Attention of Text-to-Image Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Microsoft  
[Paper Link](http://arxiv.org/abs/2309.04109v1)  

---


**ABSTRACT**  
Diffusion models have revolted the field of text-to-image generation recently. The unique way of fusing text and image information contributes to their remarkable capability of generating highly text-related images. From another perspective, these generative models imply clues about the precise correlation between words and pixels. In this work, a simple but effective method is proposed to utilize the attention mechanism in the denoising network of text-to-image diffusion models. Without re-training nor inference-time optimization, the semantic grounding of phrases can be attained directly. We evaluate our method on Pascal VOC 2012 and Microsoft COCO 2014 under weakly-supervised semantic segmentation setting and our method achieves superior performance to prior methods. In addition, the acquired word-pixel correlation is found to be generalizable for the learned text embedding of customized generation methods, requiring only a few modifications. To validate our discovery, we introduce a new practical task called "personalized referring image segmentation" with a new dataset. Experiments in various situations demonstrate the advantages of our method compared to strong baselines on this task. In summary, our work reveals a novel way to extract the rich multi-modal knowledge hidden in diffusion models for segmentation.

{{</citation>}}


### (12/42) Weakly Supervised Point Clouds Transformer for 3D Object Detection (Zuojin Tang et al., 2023)

{{<citation>}}

Zuojin Tang, Bo Sun, Tongwei Ma, Daosheng Li, Zhenhui Xu. (2023)  
**Weakly Supervised Point Clouds Transformer for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: ImageNet, Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2309.04105v1)  

---


**ABSTRACT**  
The annotation of 3D datasets is required for semantic-segmentation and object detection in scene understanding. In this paper we present a framework for the weakly supervision of a point clouds transformer that is used for 3D object detection. The aim is to decrease the required amount of supervision needed for training, as a result of the high cost of annotating a 3D datasets. We propose an Unsupervised Voting Proposal Module, which learns randomly preset anchor points and uses voting network to select prepared anchor points of high quality. Then it distills information into student and teacher network. In terms of student network, we apply ResNet network to efficiently extract local characteristics. However, it also can lose much global information. To provide the input which incorporates the global and local information as the input of student networks, we adopt the self-attention mechanism of transformer to extract global features, and the ResNet layers to extract region proposals. The teacher network supervises the classification and regression of the student network using the pre-trained model on ImageNet. On the challenging KITTI datasets, the experimental results have achieved the highest level of average precision compared with the most recent weakly supervised 3D object detectors.

{{</citation>}}


## cs.CL (12)



### (13/42) Measuring and Improving Chain-of-Thought Reasoning in Vision-Language Models (Yangyi Chen et al., 2023)

{{<citation>}}

Yangyi Chen, Karan Sikka, Michael Cogswell, Heng Ji, Ajay Divakaran. (2023)  
**Measuring and Improving Chain-of-Thought Reasoning in Vision-Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.04461v1)  

---


**ABSTRACT**  
Vision-language models (VLMs) have recently demonstrated strong efficacy as visual assistants that can parse natural queries about the visual content and generate human-like outputs. In this work, we explore the ability of these models to demonstrate human-like reasoning based on the perceived information. To address a crucial concern regarding the extent to which their reasoning capabilities are fully consistent and grounded, we also measure the reasoning consistency of these models. We achieve this by proposing a chain-of-thought (CoT) based consistency measure. However, such an evaluation requires a benchmark that encompasses both high-level inference and detailed reasoning chains, which is costly. We tackle this challenge by proposing a LLM-Human-in-the-Loop pipeline, which notably reduces cost while simultaneously ensuring the generation of a high-quality dataset. Based on this pipeline and the existing coarse-grained annotated dataset, we build the CURE benchmark to measure both the zero-shot reasoning performance and consistency of VLMs. We evaluate existing state-of-the-art VLMs, and find that even the best-performing model is unable to demonstrate strong visual reasoning capabilities and consistency, indicating that substantial efforts are required to enable VLMs to perform visual reasoning as systematically and consistently as humans. As an early step, we propose a two-stage training framework aimed at improving both the reasoning performance and consistency of VLMs. The first stage involves employing supervised fine-tuning of VLMs using step-by-step reasoning samples automatically generated by LLMs. In the second stage, we further augment the training process by incorporating feedback provided by LLMs to produce reasoning chains that are highly consistent and grounded. We empirically highlight the effectiveness of our framework in both reasoning performance and consistency.

{{</citation>}}


### (14/42) CSPRD: A Financial Policy Retrieval Dataset for Chinese Stock Market (Jinyuan Wang et al., 2023)

{{<citation>}}

Jinyuan Wang, Hai Zhao, Zhong Wang, Zeyang Zhu, Jinhao Xie, Yong Yu, Yongjian Fei, Yue Huang, Dawei Cheng. (2023)  
**CSPRD: A Financial Policy Retrieval Dataset for Chinese Stock Market**  

---
Primary Category: cs.CL  
Categories: cs-CE, cs-CL, cs.CL  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2309.04389v1)  

---


**ABSTRACT**  
In recent years, great advances in pre-trained language models (PLMs) have sparked considerable research focus and achieved promising performance on the approach of dense passage retrieval, which aims at retrieving relative passages from massive corpus with given questions. However, most of existing datasets mainly benchmark the models with factoid queries of general commonsense, while specialised fields such as finance and economics remain unexplored due to the deficiency of large-scale and high-quality datasets with expert annotations. In this work, we propose a new task, policy retrieval, by introducing the Chinese Stock Policy Retrieval Dataset (CSPRD), which provides 700+ prospectus passages labeled by experienced experts with relevant articles from 10k+ entries in our collected Chinese policy corpus. Experiments on lexical, embedding and fine-tuned bi-encoder models show the effectiveness of our proposed CSPRD yet also suggests ample potential for improvement. Our best performing baseline achieves 56.1% MRR@10, 28.5% NDCG@10, 37.5% Recall@10 and 80.6% Precision@10 on dev set.

{{</citation>}}


### (15/42) Beyond Static Datasets: A Deep Interaction Approach to LLM Evaluation (Jiatong Li et al., 2023)

{{<citation>}}

Jiatong Li, Rui Li, Qi Liu. (2023)  
**Beyond Static Datasets: A Deep Interaction Approach to LLM Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.04369v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have made progress in various real-world tasks, which stimulates requirements for the evaluation of LLMs. Existing LLM evaluation methods are mainly supervised signal-based which depends on static datasets and cannot evaluate the ability of LLMs in dynamic real-world scenarios where deep interaction widely exists. Other LLM evaluation methods are human-based which are costly and time-consuming and are incapable of large-scale evaluation of LLMs. To address the issues above, we propose a novel Deep Interaction-based LLM-evaluation framework. In our proposed framework, LLMs' performances in real-world domains can be evaluated from their deep interaction with other LLMs in elaborately designed evaluation tasks. Furthermore, our proposed framework is a general evaluation method that can be applied to a host of real-world tasks such as machine translation and code generation. We demonstrate the effectiveness of our proposed method through extensive experiments on four elaborately designed evaluation tasks.

{{</citation>}}


### (16/42) Encoding Multi-Domain Scientific Papers by Ensembling Multiple CLS Tokens (Ronald Seoh et al., 2023)

{{<citation>}}

Ronald Seoh, Haw-Shiuan Chang, Andrew McCallum. (2023)  
**Encoding Multi-Domain Scientific Papers by Ensembling Multiple CLS Tokens**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-DL, cs-LG, cs.CL  
Keywords: BERT, Transformer  
[Paper Link](http://arxiv.org/abs/2309.04333v1)  

---


**ABSTRACT**  
Many useful tasks on scientific documents, such as topic classification and citation prediction, involve corpora that span multiple scientific domains. Typically, such tasks are accomplished by representing the text with a vector embedding obtained from a Transformer's single CLS token. In this paper, we argue that using multiple CLS tokens could make a Transformer better specialize to multiple scientific domains. We present Multi2SPE: it encourages each of multiple CLS tokens to learn diverse ways of aggregating token embeddings, then sums them up together to create a single vector representation. We also propose our new multi-domain benchmark, Multi-SciDocs, to test scientific paper vector encoders under multi-domain settings. We show that Multi2SPE reduces error by up to 25 percent in multi-domain citation prediction, while requiring only a negligible amount of computation in addition to one BERT forward pass.

{{</citation>}}


### (17/42) Fuzzy Fingerprinting Transformer Language-Models for Emotion Recognition in Conversations (Patrícia Pereira et al., 2023)

{{<citation>}}

Patrícia Pereira, Rui Ribeiro, Helena Moniz, Luisa Coheur, Joao Paulo Carvalho. (2023)  
**Fuzzy Fingerprinting Transformer Language-Models for Emotion Recognition in Conversations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Dialog, Emotion Recognition, Language Model, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2309.04292v1)  

---


**ABSTRACT**  
Fuzzy Fingerprints have been successfully used as an interpretable text classification technique, but, like most other techniques, have been largely surpassed in performance by Large Pre-trained Language Models, such as BERT or RoBERTa. These models deliver state-of-the-art results in several Natural Language Processing tasks, namely Emotion Recognition in Conversations (ERC), but suffer from the lack of interpretability and explainability. In this paper, we propose to combine the two approaches to perform ERC, as a means to obtain simpler and more interpretable Large Language Models-based classifiers. We propose to feed the utterances and their previous conversational turns to a pre-trained RoBERTa, obtaining contextual embedding utterance representations, that are then supplied to an adapted Fuzzy Fingerprint classification module. We validate our approach on the widely used DailyDialog ERC benchmark dataset, in which we obtain state-of-the-art level results using a much lighter model.

{{</citation>}}


### (18/42) From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting (Griffin Adams et al., 2023)

{{<citation>}}

Griffin Adams, Alexander Fabbri, Faisal Ladhak, Eric Lehman, Noémie Elhadad. (2023)  
**From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Summarization  
[Paper Link](http://arxiv.org/abs/2309.04269v1)  

---


**ABSTRACT**  
Selecting the ``right'' amount of information to include in a summary is a difficult task. A good summary should be detailed and entity-centric without being overly dense and hard to follow. To better understand this tradeoff, we solicit increasingly dense GPT-4 summaries with what we refer to as a ``Chain of Density'' (CoD) prompt. Specifically, GPT-4 generates an initial entity-sparse summary before iteratively incorporating missing salient entities without increasing the length. Summaries generated by CoD are more abstractive, exhibit more fusion, and have less of a lead bias than GPT-4 summaries generated by a vanilla prompt. We conduct a human preference study on 100 CNN DailyMail articles and find that that humans prefer GPT-4 summaries that are more dense than those generated by a vanilla prompt and almost as dense as human written summaries. Qualitative analysis supports the notion that there exists a tradeoff between informativeness and readability. 500 annotated CoD summaries, as well as an extra 5,000 unannotated summaries, are freely available on HuggingFace (https://huggingface.co/datasets/griffin/chain_of_density).

{{</citation>}}


### (19/42) UQ at #SMM4H 2023: ALEX for Public Health Analysis with Social Media (Yan Jiang et al., 2023)

{{<citation>}}

Yan Jiang, Ruihong Qiu, Yi Zhang, Zi Huang. (2023)  
**UQ at #SMM4H 2023: ALEX for Public Health Analysis with Social Media**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Social Media  
[Paper Link](http://arxiv.org/abs/2309.04213v1)  

---


**ABSTRACT**  
As social media becomes increasingly popular, more and more activities related to public health emerge. Current techniques for public health analysis involve popular models such as BERT and large language models (LLMs). However, the costs of training in-domain LLMs for public health are especially expensive. Furthermore, such kinds of in-domain datasets from social media are generally imbalanced. To tackle these challenges, the data imbalance issue can be overcome by data augmentation and balanced training. Moreover, the ability of the LLMs can be effectively utilized by prompting the model properly. In this paper, a novel ALEX framework is proposed to improve the performance of public health analysis on social media by adopting an LLMs explanation mechanism. Results show that our ALEX model got the best performance among all submissions in both Task 2 and Task 4 with a high score in Task 1 in Social Media Mining for Health 2023 (SMM4H)[1]. Our code has been released at https:// github.com/YanJiangJerry/ALEX.

{{</citation>}}


### (20/42) The CALLA Dataset: Probing LLMs' Interactive Knowledge Acquisition from Chinese Medical Literature (Yanrui Du et al., 2023)

{{<citation>}}

Yanrui Du, Sendong Zhao, Yuhan Chen, Rai Bai, Jing Liu, Hua Wu, Haifeng Wang, Bing Qin. (2023)  
**The CALLA Dataset: Probing LLMs' Interactive Knowledge Acquisition from Chinese Medical Literature**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.04198v1)  

---


**ABSTRACT**  
The application of Large Language Models (LLMs) to the medical domain has stimulated the interest of researchers. Recent studies have focused on constructing Instruction Fine-Tuning (IFT) data through medical knowledge graphs to enrich the interactive medical knowledge of LLMs. However, the medical literature serving as a rich source of medical knowledge remains unexplored. Our work introduces the CALLA dataset to probe LLMs' interactive knowledge acquisition from Chinese medical literature. It assesses the proficiency of LLMs in mastering medical knowledge through a free-dialogue fact-checking task. We identify a phenomenon called the ``fact-following response``, where LLMs tend to affirm facts mentioned in questions and display a reluctance to challenge them. To eliminate the inaccurate evaluation caused by this phenomenon, for the golden fact, we artificially construct test data from two perspectives: one consistent with the fact and one inconsistent with the fact. Drawing from the probing experiment on the CALLA dataset, we conclude that IFT data highly correlated with the medical literature corpus serves as a potent catalyst for LLMs, enabling themselves to skillfully employ the medical knowledge acquired during the pre-training phase within interactive scenarios, enhancing accuracy. Furthermore, we design a framework for automatically constructing IFT data based on medical literature and discuss some real-world applications.

{{</citation>}}


### (21/42) Knowledge-tuning Large Language Models with Structured Medical Knowledge Bases for Reliable Response Generation in Chinese (Haochun Wang et al., 2023)

{{<citation>}}

Haochun Wang, Sendong Zhao, Zewen Qiang, Zijian Li, Nuwa Xi, Yanrui Du, MuZhen Cai, Haoqiang Guo, Yuhan Chen, Haoming Xu, Bing Qin, Ting Liu. (2023)  
**Knowledge-tuning Large Language Models with Structured Medical Knowledge Bases for Reliable Response Generation in Chinese**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP, QA  
[Paper Link](http://arxiv.org/abs/2309.04175v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable success in diverse natural language processing (NLP) tasks in general domains. However, LLMs sometimes generate responses with the hallucination about medical facts due to limited domain knowledge. Such shortcomings pose potential risks in the utilization of LLMs within medical contexts. To address this challenge, we propose knowledge-tuning, which leverages structured medical knowledge bases for the LLMs to grasp domain knowledge efficiently and facilitate reliable response generation. We also release cMedKnowQA, a Chinese medical knowledge question-answering dataset constructed from medical knowledge bases to assess the medical knowledge proficiency of LLMs. Experimental results show that the LLMs which are knowledge-tuned with cMedKnowQA, can exhibit higher levels of accuracy in response generation compared with vanilla instruction-tuning and offer a new reliable way for the domain adaptation of LLMs.

{{</citation>}}


### (22/42) Manifold-based Verbalizer Space Re-embedding for Tuning-free Prompt-based Classification (Haochun Wang et al., 2023)

{{<citation>}}

Haochun Wang, Sendong Zhao, Chi Liu, Nuwa Xi, Muzhen Cai, Bing Qin, Ting Liu. (2023)  
**Manifold-based Verbalizer Space Re-embedding for Tuning-free Prompt-based Classification**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Embedding, LLaMA  
[Paper Link](http://arxiv.org/abs/2309.04174v1)  

---


**ABSTRACT**  
Prompt-based classification adapts tasks to a cloze question format utilizing the [MASK] token and the filled tokens are then mapped to labels through pre-defined verbalizers. Recent studies have explored the use of verbalizer embeddings to reduce labor in this process. However, all existing studies require a tuning process for either the pre-trained models or additional trainable embeddings. Meanwhile, the distance between high-dimensional verbalizer embeddings should not be measured by Euclidean distance due to the potential for non-linear manifolds in the representation space. In this study, we propose a tuning-free manifold-based space re-embedding method called Locally Linear Embedding with Intra-class Neighborhood Constraint (LLE-INC) for verbalizer embeddings, which preserves local properties within the same class as guidance for classification. Experimental results indicate that even without tuning any parameters, our LLE-INC is on par with automated verbalizers with parameter tuning. And with the parameter updating, our approach further enhances prompt-based tuning by up to 3.2%. Furthermore, experiments with the LLaMA-7B&13B indicate that LLE-INC is an efficient tuning-free classification approach for the hyper-scale language models.

{{</citation>}}


### (23/42) NESTLE: a No-Code Tool for Statistical Analysis of Legal Corpus (Kyoungyeon Cho et al., 2023)

{{<citation>}}

Kyoungyeon Cho, Seungkum Han, Wonseok Hwang. (2023)  
**NESTLE: a No-Code Tool for Statistical Analysis of Legal Corpus**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GLUE, GPT, GPT-4, Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2309.04146v1)  

---


**ABSTRACT**  
The statistical analysis of large scale legal corpus can provide valuable legal insights. For such analysis one needs to (1) select a subset of the corpus using document retrieval tools, (2) structuralize text using information extraction (IE) systems, and (3) visualize the data for the statistical analysis. Each process demands either specialized tools or programming skills whereas no comprehensive unified "no-code" tools have been available. Especially for IE, if the target information is not predefined in the ontology of the IE system, one needs to build their own system. Here we provide NESTLE, a no code tool for large-scale statistical analysis of legal corpus. With NESTLE, users can search target documents, extract information, and visualize the structured data all via the chat interface with accompanying auxiliary GUI for the fine-level control. NESTLE consists of three main components: a search engine, an end-to-end IE system, and a Large Language Model (LLM) that glues the whole components together and provides the chat interface. Powered by LLM and the end-to-end IE system, NESTLE can extract any type of information that has not been predefined in the IE system opening up the possibility of unlimited customizable statistical analysis of the corpus without writing a single line of code. The use of the custom end-to-end IE system also enables faster and low-cost IE on large scale corpus. We validate our system on 15 Korean precedent IE tasks and 3 legal text classification tasks from LEXGLUE. The comprehensive experiments reveal NESTLE can achieve GPT-4 comparable performance by training the internal IE module with 4 human-labeled, and 192 LLM-labeled examples. The detailed analysis provides the insight on the trade-off between accuracy, time, and cost in building such system.

{{</citation>}}


### (24/42) Unsupervised Multi-document Summarization with Holistic Inference (Haopeng Zhang et al., 2023)

{{<citation>}}

Haopeng Zhang, Sangwoo Cho, Kaiqiang Song, Xiaoyang Wang, Hongwei Wang, Jiawei Zhang, Dong Yu. (2023)  
**Unsupervised Multi-document Summarization with Holistic Inference**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2309.04087v1)  

---


**ABSTRACT**  
Multi-document summarization aims to obtain core information from a collection of documents written on the same topic. This paper proposes a new holistic framework for unsupervised multi-document extractive summarization. Our method incorporates the holistic beam search inference method associated with the holistic measurements, named Subset Representative Index (SRI). SRI balances the importance and diversity of a subset of sentences from the source documents and can be calculated in unsupervised and adaptive manners. To demonstrate the effectiveness of our method, we conduct extensive experiments on both small and large-scale multi-document summarization datasets under both unsupervised and adaptive settings. The proposed method outperforms strong baselines by a significant margin, as indicated by the resulting ROUGE scores and diversity measures. Our findings also suggest that diversity is essential for improving multi-document summary performance.

{{</citation>}}


## cs.LG (7)



### (25/42) Subwords as Skills: Tokenization for Sparse-Reward Reinforcement Learning (David Yunis et al., 2023)

{{<citation>}}

David Yunis, Justin Jung, Falcon Dai, Matthew Walter. (2023)  
**Subwords as Skills: Tokenization for Sparse-Reward Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.04459v1)  

---


**ABSTRACT**  
Exploration in sparse-reward reinforcement learning is difficult due to the requirement of long, coordinated sequences of actions in order to achieve any reward. Moreover, in continuous action spaces there are an infinite number of possible actions, which only increases the difficulty of exploration. One class of methods designed to address these issues forms temporally extended actions, often called skills, from interaction data collected in the same domain, and optimizes a policy on top of this new action space. Typically such methods require a lengthy pretraining phase, especially in continuous action spaces, in order to form the skills before reinforcement learning can begin. Given prior evidence that the full range of the continuous action space is not required in such tasks, we propose a novel approach to skill-generation with two components. First we discretize the action space through clustering, and second we leverage a tokenization technique borrowed from natural language processing to generate temporally extended actions. Such a method outperforms baselines for skill-generation in several challenging sparse-reward domains, and requires orders-of-magnitude less computation in skill-generation and online rollouts.

{{</citation>}}


### (26/42) Robust Representation Learning for Privacy-Preserving Machine Learning: A Multi-Objective Autoencoder Approach (Sofiane Ouaari et al., 2023)

{{<citation>}}

Sofiane Ouaari, Ali Burak Ünal, Mete Akgün, Nico Pfeifer. (2023)  
**Robust Representation Learning for Privacy-Preserving Machine Learning: A Multi-Objective Autoencoder Approach**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.04427v1)  

---


**ABSTRACT**  
Several domains increasingly rely on machine learning in their applications. The resulting heavy dependence on data has led to the emergence of various laws and regulations around data ethics and privacy and growing awareness of the need for privacy-preserving machine learning (ppML). Current ppML techniques utilize methods that are either purely based on cryptography, such as homomorphic encryption, or that introduce noise into the input, such as differential privacy. The main criticism given to those techniques is the fact that they either are too slow or they trade off a model s performance for improved confidentiality. To address this performance reduction, we aim to leverage robust representation learning as a way of encoding our data while optimizing the privacy-utility trade-off. Our method centers on training autoencoders in a multi-objective manner and then concatenating the latent and learned features from the encoding part as the encoded form of our data. Such a deep learning-powered encoding can then safely be sent to a third party for intensive training and hyperparameter tuning. With our proposed framework, we can share our data and use third party tools without being under the threat of revealing its original form. We empirically validate our results on unimodal and multimodal settings, the latter following a vertical splitting system and show improved performance over state-of-the-art.

{{</citation>}}


### (27/42) Active Learning for Classifying 2D Grid-Based Level Completability (Mahsa Bazzaz et al., 2023)

{{<citation>}}

Mahsa Bazzaz, Seth Cooper. (2023)  
**Active Learning for Classifying 2D Grid-Based Level Completability**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2309.04367v1)  

---


**ABSTRACT**  
Determining the completability of levels generated by procedural generators such as machine learning models can be challenging, as it can involve the use of solver agents that often require a significant amount of time to analyze and solve levels. Active learning is not yet widely adopted in game evaluations, although it has been used successfully in natural language processing, image and speech recognition, and computer vision, where the availability of labeled data is limited or expensive. In this paper, we propose the use of active learning for learning level completability classification. Through an active learning approach, we train deep-learning models to classify the completability of generated levels for Super Mario Bros., Kid Icarus, and a Zelda-like game. We compare active learning for querying levels to label with completability against random queries. Our results show using an active learning approach to label levels results in better classifier performance with the same amount of labeled data.

{{</citation>}}


### (28/42) Zero-Shot Robustification of Zero-Shot Models With Foundation Models (Dyah Adila et al., 2023)

{{<citation>}}

Dyah Adila, Changho Shin, Linrong Cai, Frederic Sala. (2023)  
**Zero-Shot Robustification of Zero-Shot Models With Foundation Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: NLP, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.04344v1)  

---


**ABSTRACT**  
Zero-shot inference is a powerful paradigm that enables the use of large pretrained models for downstream classification tasks without further training. However, these models are vulnerable to inherited biases that can impact their performance. The traditional solution is fine-tuning, but this undermines the key advantage of pretrained models, which is their ability to be used out-of-the-box. We propose RoboShot, a method that improves the robustness of pretrained model embeddings in a fully zero-shot fashion. First, we use zero-shot language models (LMs) to obtain useful insights from task descriptions. These insights are embedded and used to remove harmful and boost useful components in embeddings -- without any supervision. Theoretically, we provide a simple and tractable model for biases in zero-shot embeddings and give a result characterizing under what conditions our approach can boost performance. Empirically, we evaluate RoboShot on nine image and NLP classification tasks and show an average improvement of 15.98% over several zero-shot baselines. Additionally, we demonstrate that RoboShot is compatible with a variety of pretrained and language models.

{{</citation>}}


### (29/42) Graph Neural Networks Use Graphs When They Shouldn't (Maya Bechler-Speicher et al., 2023)

{{<citation>}}

Maya Bechler-Speicher, Ido Amos, Ran Gilad-Bachrach, Amir Globerson. (2023)  
**Graph Neural Networks Use Graphs When They Shouldn't**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.04332v1)  

---


**ABSTRACT**  
Predictions over graphs play a crucial role in various domains, including social networks, molecular biology, medicine, and more. Graph Neural Networks (GNNs) have emerged as the dominant approach for learning on graph data. Instances of graph labeling problems consist of the graph-structure (i.e., the adjacency matrix), along with node-specific feature vectors. In some cases, this graph-structure is non-informative for the predictive task. For instance, molecular properties such as molar mass depend solely on the constituent atoms (node features), and not on the molecular structure. While GNNs have the ability to ignore the graph-structure in such cases, it is not clear that they will. In this work, we show that GNNs actually tend to overfit the graph-structure in the sense that they use it even when a better solution can be obtained by ignoring it. We examine this phenomenon with respect to different graph distributions and find that regular graphs are more robust to this overfitting. We then provide a theoretical explanation for this phenomenon, via analyzing the implicit bias of gradient-descent-based learning of GNNs in this setting. Finally, based on our empirical and theoretical findings, we propose a graph-editing method to mitigate the tendency of GNNs to overfit graph-structures that should be ignored. We show that this method indeed improves the accuracy of GNNs across multiple benchmarks.

{{</citation>}}


### (30/42) Curve Your Attention: Mixed-Curvature Transformers for Graph Representation Learning (Sungjun Cho et al., 2023)

{{<citation>}}

Sungjun Cho, Seunghyuk Cho, Sungwoo Park, Hankook Lee, Honglak Lee, Moontae Lee. (2023)  
**Curve Your Attention: Mixed-Curvature Transformers for Graph Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, Representation Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.04082v1)  

---


**ABSTRACT**  
Real-world graphs naturally exhibit hierarchical or cyclical structures that are unfit for the typical Euclidean space. While there exist graph neural networks that leverage hyperbolic or spherical spaces to learn representations that embed such structures more accurately, these methods are confined under the message-passing paradigm, making the models vulnerable against side-effects such as oversmoothing and oversquashing. More recent work have proposed global attention-based graph Transformers that can easily model long-range interactions, but their extensions towards non-Euclidean geometry are yet unexplored. To bridge this gap, we propose Fully Product-Stereographic Transformer, a generalization of Transformers towards operating entirely on the product of constant curvature spaces. When combined with tokenized graph Transformers, our model can learn the curvature appropriate for the input graph in an end-to-end fashion, without the need of additional tuning on different curvature initializations. We also provide a kernelized approach to non-Euclidean attention, which enables our model to run in time and memory cost linear to the number of nodes and edges while respecting the underlying geometry. Experiments on graph reconstruction and node classification demonstrate the benefits of generalizing Transformers to the non-Euclidean domain.

{{</citation>}}


### (31/42) UER: A Heuristic Bias Addressing Approach for Online Continual Learning (Huiwei Lin et al., 2023)

{{<citation>}}

Huiwei Lin, Shanshan Feng, Baoquan Zhang, Hongliang Qiao, Xutao Li, Yunming Ye. (2023)  
**UER: A Heuristic Bias Addressing Approach for Online Continual Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.04081v1)  

---


**ABSTRACT**  
Online continual learning aims to continuously train neural networks from a continuous data stream with a single pass-through data. As the most effective approach, the rehearsal-based methods replay part of previous data. Commonly used predictors in existing methods tend to generate biased dot-product logits that prefer to the classes of current data, which is known as a bias issue and a phenomenon of forgetting. Many approaches have been proposed to overcome the forgetting problem by correcting the bias; however, they still need to be improved in online fashion. In this paper, we try to address the bias issue by a more straightforward and more efficient method. By decomposing the dot-product logits into an angle factor and a norm factor, we empirically find that the bias problem mainly occurs in the angle factor, which can be used to learn novel knowledge as cosine logits. On the contrary, the norm factor abandoned by existing methods helps remember historical knowledge. Based on this observation, we intuitively propose to leverage the norm factor to balance the new and old knowledge for addressing the bias. To this end, we develop a heuristic approach called unbias experience replay (UER). UER learns current samples only by the angle factor and further replays previous samples by both the norm and angle factors. Extensive experiments on three datasets show that UER achieves superior performance over various state-of-the-art methods. The code is in https://github.com/FelixHuiweiLin/UER.

{{</citation>}}


## math.PR (1)



### (32/42) Soft Quantization using Entropic Regularization (Rajmadan Lakshmanan et al., 2023)

{{<citation>}}

Rajmadan Lakshmanan, Alois Pichler. (2023)  
**Soft Quantization using Entropic Regularization**  

---
Primary Category: math.PR  
Categories: 94A17, 81S20, 40A25, cs-LG, math-OC, math-PR, math.PR  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2309.04428v1)  

---


**ABSTRACT**  
The quantization problem aims to find the best possible approximation of probability measures on ${\mathbb{R}}^d$ using finite, discrete measures. The Wasserstein distance is a typical choice to measure the quality of the approximation. This contribution investigates the properties and robustness of the entropy-regularized quantization problem, which relaxes the standard quantization problem. The proposed approximation technique naturally adopts the softmin function, which is well known for its robustness in terms of theoretical and practicability standpoints. Moreover, we use the entropy-regularized Wasserstein distance to evaluate the quality of the soft quantization problem's approximation, and we implement a stochastic gradient approach to achieve the optimal solutions. The control parameter in our proposed method allows for the adjustment of the optimization problem's difficulty level, providing significant advantages when dealing with exceptionally challenging problems of interest. As well, this contribution empirically illustrates the performance of the method in various expositions.

{{</citation>}}


## cs.RO (3)



### (33/42) Seeing-Eye Quadruped Navigation with Force Responsive Locomotion Control (David DeFazio et al., 2023)

{{<citation>}}

David DeFazio, Eisuke Hirota, Shiqi Zhang. (2023)  
**Seeing-Eye Quadruped Navigation with Force Responsive Locomotion Control**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.04370v1)  

---


**ABSTRACT**  
Seeing-eye robots are very useful tools for guiding visually impaired people, potentially producing a huge societal impact given the low availability and high cost of real guide dogs. Although a few seeing-eye robot systems have already been demonstrated, none considered external tugs from humans, which frequently occur in a real guide dog setting. In this paper, we simultaneously train a locomotion controller that is robust to external tugging forces via Reinforcement Learning (RL), and an external force estimator via supervised learning. The controller ensures stable walking, and the force estimator enables the robot to respond to the external forces from the human. These forces are used to guide the robot to the global goal, which is unknown to the robot, while the robot guides the human around nearby obstacles via a local planner. Experimental results in simulation and on hardware show that our controller is robust to external forces, and our seeing-eye system can accurately detect force direction. We demonstrate our full seeing-eye robot system on a real quadruped robot with a blindfolded human. The video can be seen at our project page: https://bu-air-lab.github.io/guide_dog/

{{</citation>}}


### (34/42) Incremental Learning of Humanoid Robot Behavior from Natural Interaction and Large Language Models (Leonard Bärmann et al., 2023)

{{<citation>}}

Leonard Bärmann, Rainer Kartmann, Fabian Peller-Konrad, Alex Waibel, Tamim Asfour. (2023)  
**Incremental Learning of Humanoid Robot Behavior from Natural Interaction and Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.04316v1)  

---


**ABSTRACT**  
Natural-language dialog is key for intuitive human-robot interaction. It can be used not only to express humans' intents, but also to communicate instructions for improvement if a robot does not understand a command correctly. Of great importance is to endow robots with the ability to learn from such interaction experience in an incremental way to allow them to improve their behaviors or avoid mistakes in the future. In this paper, we propose a system to achieve incremental learning of complex behavior from natural interaction, and demonstrate its implementation on a humanoid robot. Building on recent advances, we present a system that deploys Large Language Models (LLMs) for high-level orchestration of the robot's behavior, based on the idea of enabling the LLM to generate Python statements in an interactive console to invoke both robot perception and action. The interaction loop is closed by feeding back human instructions, environment observations, and execution results to the LLM, thus informing the generation of the next statement. Specifically, we introduce incremental prompt learning, which enables the system to interactively learn from its mistakes. For that purpose, the LLM can call another LLM responsible for code-level improvements of the current interaction based on human feedback. The improved interaction is then saved in the robot's memory, and thus retrieved on similar requests. We integrate the system in the robot cognitive architecture of the humanoid robot ARMAR-6 and evaluate our methods both quantitatively (in simulation) and qualitatively (in simulation and real-world) by demonstrating generalized incrementally-learned knowledge.

{{</citation>}}


### (35/42) SayNav: Grounding Large Language Models for Dynamic Planning to Navigation in New Environments (Abhinav Rajvanshi et al., 2023)

{{<citation>}}

Abhinav Rajvanshi, Karan Sikka, Xiao Lin, Bhoram Lee, Han-Pang Chiu, Alvaro Velasquez. (2023)  
**SayNav: Grounding Large Language Models for Dynamic Planning to Navigation in New Environments**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.04077v1)  

---


**ABSTRACT**  
Semantic reasoning and dynamic planning capabilities are crucial for an autonomous agent to perform complex navigation tasks in unknown environments. It requires a large amount of common-sense knowledge, that humans possess, to succeed in these tasks. We present SayNav, a new approach that leverages human knowledge from Large Language Models (LLMs) for efficient generalization to complex navigation tasks in unknown large-scale environments. SayNav uses a novel grounding mechanism, that incrementally builds a 3D scene graph of the explored environment as inputs to LLMs, for generating feasible and contextually appropriate high-level plans for navigation. The LLM-generated plan is then executed by a pre-trained low-level planner, that treats each planned step as a short-distance point-goal navigation sub-task. SayNav dynamically generates step-by-step instructions during navigation and continuously refines future steps based on newly perceived information. We evaluate SayNav on a new multi-object navigation task, that requires the agent to utilize a massive amount of human knowledge to efficiently search multiple different objects in an unknown environment. SayNav outperforms an oracle based Point-nav baseline, achieving a success rate of 95.35% (vs 56.06% for the baseline), under the ideal settings on this task, highlighting its ability to generate dynamic plans for successfully locating objects in large-scale new environments.

{{</citation>}}


## cs.AI (2)



### (36/42) FIMO: A Challenge Formal Dataset for Automated Theorem Proving (Chengwu Liu et al., 2023)

{{<citation>}}

Chengwu Liu, Jianhao Shen, Huajian Xin, Zhengying Liu, Ye Yuan, Haiming Wang, Wei Ju, Chuanyang Zheng, Yichun Yin, Lin Li, Ming Zhang, Qun Liu. (2023)  
**FIMO: A Challenge Formal Dataset for Automated Theorem Proving**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.04295v1)  

---


**ABSTRACT**  
We present FIMO, an innovative dataset comprising formal mathematical problem statements sourced from the International Mathematical Olympiad (IMO) Shortlisted Problems. Designed to facilitate advanced automated theorem proving at the IMO level, FIMO is currently tailored for the Lean formal language. It comprises 149 formal problem statements, accompanied by both informal problem descriptions and their corresponding LaTeX-based informal proofs. Through initial experiments involving GPT-4, our findings underscore the existing limitations in current methodologies, indicating a substantial journey ahead before achieving satisfactory IMO-level automated theorem proving outcomes.

{{</citation>}}


### (37/42) Inferring physical laws by artificial intelligence based causal models (Jorawar Singh et al., 2023)

{{<citation>}}

Jorawar Singh, Kishor Bharti, Arvind. (2023)  
**Inferring physical laws by artificial intelligence based causal models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, physics-data-an, quant-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.04069v1)  

---


**ABSTRACT**  
The advances in Artificial Intelligence (AI) and Machine Learning (ML) have opened up many avenues for scientific research, and are adding new dimensions to the process of knowledge creation. However, even the most powerful and versatile of ML applications till date are primarily in the domain of analysis of associations and boil down to complex data fitting. Judea Pearl has pointed out that Artificial General Intelligence must involve interventions involving the acts of doing and imagining. Any machine assisted scientific discovery thus must include casual analysis and interventions. In this context, we propose a causal learning model of physical principles, which not only recognizes correlations but also brings out casual relationships. We use the principles of causal inference and interventions to study the cause-and-effect relationships in the context of some well-known physical phenomena. We show that this technique can not only figure out associations among data, but is also able to correctly ascertain the cause-and-effect relations amongst the variables, thereby strengthening (or weakening) our confidence in the proposed model of the underlying physical process.

{{</citation>}}


## cs.NI (1)



### (38/42) LLMCad: Fast and Scalable On-device Large Language Model Inference (Daliang Xu et al., 2023)

{{<citation>}}

Daliang Xu, Wangsong Yin, Xin Jin, Ying Zhang, Shiyun Wei, Mengwei Xu, Xuanzhe Liu. (2023)  
**LLMCad: Fast and Scalable On-device Large Language Model Inference**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-NI, cs.NI  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.04255v1)  

---


**ABSTRACT**  
Generative tasks, such as text generation and question answering, hold a crucial position in the realm of mobile applications. Due to their sensitivity to privacy concerns, there is a growing demand for their execution directly on mobile devices. Currently, the execution of these generative tasks heavily depends on Large Language Models (LLMs). Nevertheless, the limited memory capacity of these devices presents a formidable challenge to the scalability of such models.   In our research, we introduce LLMCad, an innovative on-device inference engine specifically designed for efficient generative Natural Language Processing (NLP) tasks. The core idea behind LLMCad revolves around model collaboration: a compact LLM, residing in memory, takes charge of generating the most straightforward tokens, while a high-precision LLM steps in to validate these tokens and rectify any identified errors. LLMCad incorporates three novel techniques: (1) Instead of generating candidate tokens in a sequential manner, LLMCad employs the smaller LLM to construct a token tree, encompassing a wider range of plausible token pathways. Subsequently, the larger LLM can efficiently validate all of these pathways simultaneously. (2) It employs a self-adjusting fallback strategy, swiftly initiating the verification process whenever the smaller LLM generates an erroneous token. (3) To ensure a continuous flow of token generation, LLMCad speculatively generates tokens during the verification process by implementing a compute-IO pipeline. Through an extensive series of experiments, LLMCad showcases an impressive token generation speed, achieving rates up to 9.3x faster than existing inference engines.

{{</citation>}}


## cs.SD (1)



### (39/42) A Long-Tail Friendly Representation Framework for Artist and Music Similarity (Haoran Xiang et al., 2023)

{{<citation>}}

Haoran Xiang, Junyu Dai, Xuchen Song, Furao Shen. (2023)  
**A Long-Tail Friendly Representation Framework for Artist and Music Similarity**  

---
Primary Category: cs.SD  
Categories: cs-IR, cs-SD, cs.SD, eess-AS  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2309.04182v1)  

---


**ABSTRACT**  
The investigation of the similarity between artists and music is crucial in music retrieval and recommendation, and addressing the challenge of the long-tail phenomenon is increasingly important. This paper proposes a Long-Tail Friendly Representation Framework (LTFRF) that utilizes neural networks to model the similarity relationship. Our approach integrates music, user, metadata, and relationship data into a unified metric learning framework, and employs a meta-consistency relationship as a regular term to introduce the Multi-Relationship Loss. Compared to the Graph Neural Network (GNN), our proposed framework improves the representation performance in long-tail scenarios, which are characterized by sparse relationships between artists and music. We conduct experiments and analysis on the AllMusic dataset, and the results demonstrate that our framework provides a favorable generalization of artist and music representation. Specifically, on similar artist/music recommendation tasks, the LTFRF outperforms the baseline by 9.69%/19.42% in Hit Ratio@10, and in long-tail cases, the framework achieves 11.05%/14.14% higher than the baseline in Consistent@10.

{{</citation>}}


## cs.IT (1)



### (40/42) Sparse-DFT and WHT Precoding with Iterative Detection for Highly Frequency-Selective Channels (Roberto Bomfin et al., 2023)

{{<citation>}}

Roberto Bomfin, Marwa Chafii. (2023)  
**Sparse-DFT and WHT Precoding with Iterative Detection for Highly Frequency-Selective Channels**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.04149v1)  

---


**ABSTRACT**  
Various precoders have been recently studied by the wireless community to combat the channel fading effects. Two prominent precoders are implemented with the discrete Fourier transform (DFT) and Walsh-Hadamard transform (WHT). The WHT precoder is implemented with less complexity since it does not need complex multiplications. Also, spreading can be applied sparsely to decrease the transceiver complexity, leading to sparse DFT (SDFT) and sparse Walsh-Hadamard (SWH). Another relevant topic is the design of iterative receivers that deal with inter-symbol-interference (ISI). In particular, many detectors based on expectation propagation (EP) have been proposed recently for channels with high levels of ISI. An alternative is the maximum a-posterior (MAP) detector, although it leads to unfeasible high complexity in many cases. In this paper, we provide a relatively low-complexity \textcolor{black}{computation} of the MAP detector for the SWH. We also propose two \textcolor{black}{feasible methods} based on the Log-MAP and Max-Log-MAP. Additionally, the DFT, SDFT and SWH precoders are compared using an EP-based receiver with one-tap FD equalization. Lastly, SWH-Max-Log-MAP is compared to the (S)DFT with EP-based receiver in terms of performance and complexity. The results show that the proposed SWH-Max-Log-MAP has a better performance and complexity trade-off for QPSK and 16-QAM under highly selective channels, but has unfeasible complexity for higher QAM orders.

{{</citation>}}


## cs.SE (1)



### (41/42) Trustworthy and Synergistic Artificial Intelligence for Software Engineering: Vision and Roadmaps (David Lo, 2023)

{{<citation>}}

David Lo. (2023)  
**Trustworthy and Synergistic Artificial Intelligence for Software Engineering: Vision and Roadmaps**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.04142v1)  

---


**ABSTRACT**  
For decades, much software engineering research has been dedicated to devising automated solutions aimed at enhancing developer productivity and elevating software quality. The past two decades have witnessed an unparalleled surge in the development of intelligent solutions tailored for software engineering tasks. This momentum established the Artificial Intelligence for Software Engineering (AI4SE) area, which has swiftly become one of the most active and popular areas within the software engineering field.   This Future of Software Engineering (FoSE) paper navigates through several focal points. It commences with a succinct introduction and history of AI4SE. Thereafter, it underscores the core challenges inherent to AI4SE, particularly highlighting the need to realize trustworthy and synergistic AI4SE. Progressing, the paper paints a vision for the potential leaps achievable if AI4SE's key challenges are surmounted, suggesting a transition towards Software Engineering 2.0. Two strategic roadmaps are then laid out: one centered on realizing trustworthy AI4SE, and the other on fostering synergistic AI4SE. While this paper may not serve as a conclusive guide, its intent is to catalyze further progress. The ultimate aspiration is to position AI4SE as a linchpin in redefining the horizons of software engineering, propelling us toward Software Engineering 2.0.

{{</citation>}}


## eess.IV (1)



### (42/42) Enhancing Hierarchical Transformers for Whole Brain Segmentation with Intracranial Measurements Integration (Xin Yu et al., 2023)

{{<citation>}}

Xin Yu, Yucheng Tang, Qi Yang, Ho Hin Lee, Shunxing Bao, Yuankai Huo, Bennett A. Landman. (2023)  
**Enhancing Hierarchical Transformers for Whole Brain Segmentation with Intracranial Measurements Integration**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.04071v1)  

---


**ABSTRACT**  
Whole brain segmentation with magnetic resonance imaging (MRI) enables the non-invasive measurement of brain regions, including total intracranial volume (TICV) and posterior fossa volume (PFV). Enhancing the existing whole brain segmentation methodology to incorporate intracranial measurements offers a heightened level of comprehensiveness in the analysis of brain structures. Despite its potential, the task of generalizing deep learning techniques for intracranial measurements faces data availability constraints due to limited manually annotated atlases encompassing whole brain and TICV/PFV labels. In this paper, we enhancing the hierarchical transformer UNesT for whole brain segmentation to achieve segmenting whole brain with 133 classes and TICV/PFV simultaneously. To address the problem of data scarcity, the model is first pretrained on 4859 T1-weighted (T1w) 3D volumes sourced from 8 different sites. These volumes are processed through a multi-atlas segmentation pipeline for label generation, while TICV/PFV labels are unavailable. Subsequently, the model is finetuned with 45 T1w 3D volumes from Open Access Series Imaging Studies (OASIS) where both 133 whole brain classes and TICV/PFV labels are available. We evaluate our method with Dice similarity coefficients(DSC). We show that our model is able to conduct precise TICV/PFV estimation while maintaining the 132 brain regions performance at a comparable level. Code and trained model are available at: https://github.com/MASILab/UNesT/wholebrainSeg.

{{</citation>}}
