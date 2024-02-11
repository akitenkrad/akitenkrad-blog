---
draft: false
title: "arXiv @ 2024.02.10"
date: 2024-02-10
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.02.10"
    identifier: arxiv_20240210
    parent: 202402_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (19)](#cscv-19)
- [cs.LG (32)](#cslg-32)
- [cs.RO (2)](#csro-2)
- [cs.AI (10)](#csai-10)
- [cs.CL (29)](#cscl-29)
- [cs.HC (6)](#cshc-6)
- [cs.SI (3)](#cssi-3)
- [stat.ML (3)](#statml-3)
- [eess.AS (1)](#eessas-1)
- [eess.IV (3)](#eessiv-3)
- [cs.CY (2)](#cscy-2)
- [cs.GT (1)](#csgt-1)
- [cs.IR (1)](#csir-1)
- [cs.IT (1)](#csit-1)
- [cs.MA (1)](#csma-1)
- [cs.CR (2)](#cscr-2)
- [cs.SE (4)](#csse-4)
- [cs.DL (1)](#csdl-1)
- [eess.SP (2)](#eesssp-2)
- [eess.SY (1)](#eesssy-1)
- [physics.flu-dyn (1)](#physicsflu-dyn-1)

## cs.CV (19)



### (1/125) InstaGen: Enhancing Object Detection by Training on Synthetic Dataset (Chengjian Feng et al., 2024)

{{<citation>}}

Chengjian Feng, Yujie Zhong, Zequn Jie, Weidi Xie, Lin Ma. (2024)  
**InstaGen: Enhancing Object Detection by Training on Synthetic Dataset**
<br/>
<button class="copy-to-clipboard" title="InstaGen: Enhancing Object Detection by Training on Synthetic Dataset" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05937v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05937v1.pdf" filename="2402.05937v1.pdf">Download PDF</button>

---


**ABSTRACT**  
In this paper, we introduce a novel paradigm to enhance the ability of object detector, e.g., expanding categories or improving detection performance, by training on synthetic dataset generated from diffusion models. Specifically, we integrate an instance-level grounding head into a pre-trained, generative diffusion model, to augment it with the ability of localising arbitrary instances in the generated images. The grounding head is trained to align the text embedding of category names with the regional visual feature of the diffusion model, using supervision from an off-the-shelf object detector, and a novel self-training scheme on (novel) categories not covered by the detector. This enhanced version of diffusion model, termed as InstaGen, can serve as a data synthesizer for object detection. We conduct thorough experiments to show that, object detector can be enhanced while training on the synthetic dataset from InstaGen, demonstrating superior performance over existing state-of-the-art methods in open-vocabulary (+4.5 AP) and data-sparse (+1.2 to 5.2 AP) scenarios.

{{</citation>}}


### (2/125) SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models (Peng Gao et al., 2024)

{{<citation>}}

Peng Gao, Renrui Zhang, Chris Liu, Longtian Qiu, Siyuan Huang, Weifeng Lin, Shitian Zhao, Shijie Geng, Ziyi Lin, Peng Jin, Kaipeng Zhang, Wenqi Shao, Chao Xu, Conghui He, Junjun He, Hao Shao, Pan Lu, Hongsheng Li, Yu Qiao. (2024)  
**SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models**
<br/>
<button class="copy-to-clipboard" title="SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: LLaMA, Language Model, OCR  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05935v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05935v1.pdf" filename="2402.05935v1.pdf">Download PDF</button>

---


**ABSTRACT**  
We propose SPHINX-X, an extensive Multimodality Large Language Model (MLLM) series developed upon SPHINX. To improve the architecture and training efficiency, we modify the SPHINX framework by removing redundant visual encoders, bypassing fully-padded sub-images with skip tokens, and simplifying multi-stage training into a one-stage all-in-one paradigm. To fully unleash the potential of MLLMs, we assemble a comprehensive multi-domain and multimodal dataset covering publicly available resources in language, vision, and vision-language tasks. We further enrich this collection with our curated OCR intensive and Set-of-Mark datasets, extending the diversity and generality. By training over different base LLMs including TinyLlama1.1B, InternLM2-7B, LLaMA2-13B, and Mixtral8x7B, we obtain a spectrum of MLLMs that vary in parameter size and multilingual capabilities. Comprehensive benchmarking reveals a strong correlation between the multi-modal performance with the data and parameter scales. Code and models are released at https://github.com/Alpha-VLLM/LLaMA2-Accessory

{{</citation>}}


### (3/125) Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data (Shufan Li et al., 2024)

{{<citation>}}

Shufan Li, Harkanwar Singh, Aditya Grover. (2024)  
**Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data**
<br/>
<button class="copy-to-clipboard" title="Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, LSTM, Transformer, Transformers  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05892v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05892v1.pdf" filename="2402.05892v1.pdf">Download PDF</button>

---


**ABSTRACT**  
In recent years, Transformers have become the de-facto architecture for sequence modeling on text and a variety of multi-dimensional data, such as images and video. However, the use of self-attention layers in a Transformer incurs prohibitive compute and memory complexity that scales quadratically w.r.t. the sequence length. A recent architecture, Mamba, based on state space models has been shown to achieve comparable performance for modeling text sequences, while scaling linearly with the sequence length. In this work, we present Mamba-ND, a generalized design extending the Mamba architecture to arbitrary multi-dimensional data. Our design alternatively unravels the input data across different dimensions following row-major orderings. We provide a systematic comparison of Mamba-ND with several other alternatives, based on prior multi-dimensional extensions such as Bi-directional LSTMs and S4ND. Empirically, we show that Mamba-ND demonstrates performance competitive with the state-of-the-art on a variety of multi-dimensional benchmarks, including ImageNet-1K classification, HMDB-51 action recognition, and ERA5 weather forecasting.

{{</citation>}}


### (4/125) CREMA: Multimodal Compositional Video Reasoning via Efficient Modular Adaptation and Fusion (Shoubin Yu et al., 2024)

{{<citation>}}

Shoubin Yu, Jaehong Yoon, Mohit Bansal. (2024)  
**CREMA: Multimodal Compositional Video Reasoning via Efficient Modular Adaptation and Fusion**
<br/>
<button class="copy-to-clipboard" title="CREMA: Multimodal Compositional Video Reasoning via Efficient Modular Adaptation and Fusion" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Reasoning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05889v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05889v1.pdf" filename="2402.05889v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Despite impressive advancements in multimodal compositional reasoning approaches, they are still limited in their flexibility and efficiency by processing fixed modality inputs while updating a lot of model parameters. This paper tackles these critical challenges and proposes CREMA, an efficient and modular modality-fusion framework for injecting any new modality into video reasoning. We first augment multiple informative modalities (such as optical flow, 3D point cloud, audio) from given videos without extra human annotation by leveraging existing pre-trained models. Next, we introduce a query transformer with multiple parameter-efficient modules associated with each accessible modality. It projects diverse modality features to the LLM token embedding space, allowing the model to integrate different data types for response generation. Furthermore, we propose a fusion module designed to compress multimodal queries, maintaining computational efficiency in the LLM while combining additional modalities. We validate our method on video-3D, video-audio, and video-language reasoning tasks and achieve better/equivalent performance against strong multimodal LLMs, including BLIP-2, 3D-LLM, and SeViLA while using 96% fewer trainable parameters. We provide extensive analyses of CREMA, including the impact of each modality on reasoning domains, the design of the fusion module, and example visualizations.

{{</citation>}}


### (5/125) Privacy-Preserving Synthetic Continual Semantic Segmentation for Robotic Surgery (Mengya Xu et al., 2024)

{{<citation>}}

Mengya Xu, Mobarakol Islam, Long Bai, Hongliang Ren. (2024)  
**Privacy-Preserving Synthetic Continual Semantic Segmentation for Robotic Surgery**
<br/>
<button class="copy-to-clipboard" title="Privacy-Preserving Synthetic Continual Semantic Segmentation for Robotic Surgery" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05860v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05860v1.pdf" filename="2402.05860v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Deep Neural Networks (DNNs) based semantic segmentation of the robotic instruments and tissues can enhance the precision of surgical activities in robot-assisted surgery. However, in biological learning, DNNs cannot learn incremental tasks over time and exhibit catastrophic forgetting, which refers to the sharp decline in performance on previously learned tasks after learning a new one. Specifically, when data scarcity is the issue, the model shows a rapid drop in performance on previously learned instruments after learning new data with new instruments. The problem becomes worse when it limits releasing the dataset of the old instruments for the old model due to privacy concerns and the unavailability of the data for the new or updated version of the instruments for the continual learning model. For this purpose, we develop a privacy-preserving synthetic continual semantic segmentation framework by blending and harmonizing (i) open-source old instruments foreground to the synthesized background without revealing real patient data in public and (ii) new instruments foreground to extensively augmented real background. To boost the balanced logit distillation from the old model to the continual learning model, we design overlapping class-aware temperature normalization (CAT) by controlling model learning utility. We also introduce multi-scale shifted-feature distillation (SD) to maintain long and short-range spatial relationships among the semantic objects where conventional short-range spatial features with limited information reduce the power of feature distillation. We demonstrate the effectiveness of our framework on the EndoVis 2017 and 2018 instrument segmentation dataset with a generalized continual learning setting. Code is available at~\url{https://github.com/XuMengyaAmy/Synthetic_CAT_SD}.

{{</citation>}}


### (6/125) You Only Need One Color Space: An Efficient Network for Low-light Image Enhancement (Yixu Feng et al., 2024)

{{<citation>}}

Yixu Feng, Cheng Zhang, Pei Wang, Peng Wu, Qingsen Yan, Yanning Zhang. (2024)  
**You Only Need One Color Space: An Efficient Network for Low-light Image Enhancement**
<br/>
<button class="copy-to-clipboard" title="You Only Need One Color Space: An Efficient Network for Low-light Image Enhancement" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05809v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05809v1.pdf" filename="2402.05809v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Low-Light Image Enhancement (LLIE) task tends to restore the details and visual information from corrupted low-light images. Most existing methods learn the mapping function between low/normal-light images by Deep Neural Networks (DNNs) on sRGB and HSV color space. Nevertheless, enhancement involves amplifying image signals, and applying these color spaces to low-light images with a low signal-to-noise ratio can introduce sensitivity and instability into the enhancement process. Consequently, this results in the presence of color artifacts and brightness artifacts in the enhanced images. To alleviate this problem, we propose a novel trainable color space, named Horizontal/Vertical-Intensity (HVI). It not only decouples brightness and color from RGB channels to mitigate the instability during enhancement but also adapts to low-light images in different illumination ranges due to the trainable parameters. Further, we design a novel Color and Intensity Decoupling Network (CIDNet) with two branches dedicated to processing the decoupled image brightness and color in the HVI space. Within CIDNet, we introduce the Lightweight Cross-Attention (LCA) module to facilitate interaction between image structure and content information in both branches, while also suppressing noise in low-light images. Finally, we conducted 22 quantitative and qualitative experiments to show that the proposed CIDNet outperforms the state-of-the-art methods on 11 datasets. The code will be available at https://github.com/Fediory/HVI-CIDNet.

{{</citation>}}


### (7/125) TaE: Task-aware Expandable Representation for Long Tail Class Incremental Learning (Linjie Li et al., 2024)

{{<citation>}}

Linjie Li, S. Liu, Zhenyu Wu, JI yang. (2024)  
**TaE: Task-aware Expandable Representation for Long Tail Class Incremental Learning**
<br/>
<button class="copy-to-clipboard" title="TaE: Task-aware Expandable Representation for Long Tail Class Incremental Learning" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05797v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05797v1.pdf" filename="2402.05797v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Class-incremental learning (CIL) aims to train classifiers that learn new classes without forgetting old ones. Most CIL methods focus on balanced data distribution for each task, overlooking real-world long-tailed distributions. Therefore, Long-Tailed Class-Incremental Learning (LT-CIL) has been introduced, which trains on data where head classes have more samples than tail classes. Existing methods mainly focus on preserving representative samples from previous classes to combat catastrophic forgetting. Recently, dynamic network algorithms frozen old network structures and expanded new ones, achieving significant performance. However, with the introduction of the long-tail problem, merely extending task-specific parameters can lead to miscalibrated predictions, while expanding the entire model results in an explosion of memory size. To address these issues, we introduce a novel Task-aware Expandable (TaE) framework, dynamically allocating and updating task-specific trainable parameters to learn diverse representations from each incremental task, while resisting forgetting through the majority of frozen model parameters. To further encourage the class-specific feature representation, we develop a Centroid-Enhanced (CEd) method to guide the update of these task-aware parameters. This approach is designed to adaptively minimize the distances between intra-class features while simultaneously maximizing the distances between inter-class features across all seen classes. The utility of this centroid-enhanced method extends to all "training from scratch" CIL algorithms. Extensive experiments were conducted on CIFAR-100 and ImageNet100 under different settings, which demonstrates that TaE achieves state-of-the-art performance.

{{</citation>}}


### (8/125) DiffSpeaker: Speech-Driven 3D Facial Animation with Diffusion Transformer (Zhiyuan Ma et al., 2024)

{{<citation>}}

Zhiyuan Ma, Xiangyu Zhu, Guojun Qi, Chen Qian, Zhaoxiang Zhang, Zhen Lei. (2024)  
**DiffSpeaker: Speech-Driven 3D Facial Animation with Diffusion Transformer**
<br/>
<button class="copy-to-clipboard" title="DiffSpeaker: Speech-Driven 3D Facial Animation with Diffusion Transformer" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05712v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05712v1.pdf" filename="2402.05712v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Speech-driven 3D facial animation is important for many multimedia applications. Recent work has shown promise in using either Diffusion models or Transformer architectures for this task. However, their mere aggregation does not lead to improved performance. We suspect this is due to a shortage of paired audio-4D data, which is crucial for the Transformer to effectively perform as a denoiser within the Diffusion framework. To tackle this issue, we present DiffSpeaker, a Transformer-based network equipped with novel biased conditional attention modules. These modules serve as substitutes for the traditional self/cross-attention in standard Transformers, incorporating thoughtfully designed biases that steer the attention mechanisms to concentrate on both the relevant task-specific and diffusion-related conditions. We also explore the trade-off between accurate lip synchronization and non-verbal facial expressions within the Diffusion paradigm. Experiments show our model not only achieves state-of-the-art performance on existing benchmarks, but also fast inference speed owing to its ability to generate facial motions in parallel.

{{</citation>}}


### (9/125) Scalable Diffusion Models with State Space Backbone (Zhengcong Fei et al., 2024)

{{<citation>}}

Zhengcong Fei, Mingyuan Fan, Changqian Yu, Junshi Huang. (2024)  
**Scalable Diffusion Models with State Space Backbone**
<br/>
<button class="copy-to-clipboard" title="Scalable Diffusion Models with State Space Backbone" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: ImageNet, Transformer  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05608v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05608v1.pdf" filename="2402.05608v1.pdf">Download PDF</button>

---


**ABSTRACT**  
This paper presents a new exploration into a category of diffusion models built upon state space architecture. We endeavor to train diffusion models for image data, wherein the traditional U-Net backbone is supplanted by a state space backbone, functioning on raw patches or latent space. Given its notable efficacy in accommodating long-range dependencies, Diffusion State Space Models (DiS) are distinguished by treating all inputs including time, condition, and noisy image patches as tokens. Our assessment of DiS encompasses both unconditional and class-conditional image generation scenarios, revealing that DiS exhibits comparable, if not superior, performance to CNN-based or Transformer-based U-Net architectures of commensurate size. Furthermore, we analyze the scalability of DiS, gauged by the forward pass complexity quantified in Gflops. DiS models with higher Gflops, achieved through augmentation of depth/width or augmentation of input tokens, consistently demonstrate lower FID. In addition to demonstrating commendable scalability characteristics, DiS-H/2 models in latent space achieve performance levels akin to prior diffusion models on class-conditional ImageNet benchmarks at the resolution of 256$\times$256 and 512$\times$512, while significantly reducing the computational burden. The code and models are available at: https://github.com/feizc/DiS.

{{</citation>}}


### (10/125) A Concept for Reconstructing Stucco Statues from historic Sketches using synthetic Data only (Thomas Pöllabauer et al., 2024)

{{<citation>}}

Thomas Pöllabauer, Julius Kühn. (2024)  
**A Concept for Reconstructing Stucco Statues from historic Sketches using synthetic Data only**
<br/>
<button class="copy-to-clipboard" title="A Concept for Reconstructing Stucco Statues from historic Sketches using synthetic Data only" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Sketch  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05593v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05593v1.pdf" filename="2402.05593v1.pdf">Download PDF</button>

---


**ABSTRACT**  
In medieval times, stuccoworkers used a red color, called sinopia, to first create a sketch of the to-be-made statue on the wall. Today, many of these statues are destroyed, but using the original drawings, deriving from the red color also called sinopia, we can reconstruct how the final statue might have looked.We propose a fully-automated approach to reconstruct a point cloud and show preliminary results by generating a color-image, a depth-map, as well as surface normals requiring only a single sketch, and without requiring a collection of other, similar samples. Our proposed solution allows real-time reconstruction on-site, for instance, within an exhibition, or to generate a useful starting point for an expert, trying to manually reconstruct the statue, all while using only synthetic data for training.

{{</citation>}}


### (11/125) RESMatch: Referring Expression Segmentation in a Semi-Supervised Manner (Ying Zang et al., 2024)

{{<citation>}}

Ying Zang, Chenglong Fu, Runlong Cao, Didi Zhu, Min Zhang, Wenjun Hu, Lanyun Zhu, Tianrun Chen. (2024)  
**RESMatch: Referring Expression Segmentation in a Semi-Supervised Manner**
<br/>
<button class="copy-to-clipboard" title="RESMatch: Referring Expression Segmentation in a Semi-Supervised Manner" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Semi-Supervised  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05589v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05589v1.pdf" filename="2402.05589v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Referring expression segmentation (RES), a task that involves localizing specific instance-level objects based on free-form linguistic descriptions, has emerged as a crucial frontier in human-AI interaction. It demands an intricate understanding of both visual and textual contexts and often requires extensive training data. This paper introduces RESMatch, the first semi-supervised learning (SSL) approach for RES, aimed at reducing reliance on exhaustive data annotation. Extensive validation on multiple RES datasets demonstrates that RESMatch significantly outperforms baseline approaches, establishing a new state-of-the-art. Although existing SSL techniques are effective in image segmentation, we find that they fall short in RES. Facing the challenges including the comprehension of free-form linguistic descriptions and the variability in object attributes, RESMatch introduces a trifecta of adaptations: revised strong perturbation, text augmentation, and adjustments for pseudo-label quality and strong-weak supervision. This pioneering work lays the groundwork for future research in semi-supervised learning for referring expression segmentation.

{{</citation>}}


### (12/125) On Convolutional Vision Transformers for Yield Prediction (Alvin Inderka et al., 2024)

{{<citation>}}

Alvin Inderka, Florian Huber, Volker Steinhage. (2024)  
**On Convolutional Vision Transformers for Yield Prediction**
<br/>
<button class="copy-to-clipboard" title="On Convolutional Vision Transformers for Yield Prediction" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05557v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05557v1.pdf" filename="2402.05557v1.pdf">Download PDF</button>

---


**ABSTRACT**  
While a variety of methods offer good yield prediction on histogrammed remote sensing data, vision Transformers are only sparsely represented in the literature. The Convolution vision Transformer (CvT) is being tested to evaluate vision Transformers that are currently achieving state-of-the-art results in many other vision tasks. CvT combines some of the advantages of convolution with the advantages of dynamic attention and global context fusion of Transformers. It performs worse than widely tested methods such as XGBoost and CNNs, but shows that Transformers have potential to improve yield prediction.

{{</citation>}}


### (13/125) Question Aware Vision Transformer for Multimodal Reasoning (Roy Ganz et al., 2024)

{{<citation>}}

Roy Ganz, Yair Kittenplon, Aviad Aberdam, Elad Ben Avraham, Oren Nuriel, Shai Mazor, Ron Litman. (2024)  
**Question Aware Vision Transformer for Multimodal Reasoning**
<br/>
<button class="copy-to-clipboard" title="Question Aware Vision Transformer for Multimodal Reasoning" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, QA, Reasoning, Transformer  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05472v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05472v1.pdf" filename="2402.05472v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Vision-Language (VL) models have gained significant research focus, enabling remarkable advances in multimodal reasoning. These architectures typically comprise a vision encoder, a Large Language Model (LLM), and a projection module that aligns visual features with the LLM's representation space. Despite their success, a critical limitation persists: the vision encoding process remains decoupled from user queries, often in the form of image-related questions. Consequently, the resulting visual features may not be optimally attuned to the query-specific elements of the image. To address this, we introduce QA-ViT, a Question Aware Vision Transformer approach for multimodal reasoning, which embeds question awareness directly within the vision encoder. This integration results in dynamic visual features focusing on relevant image aspects to the posed question. QA-ViT is model-agnostic and can be incorporated efficiently into any VL architecture. Extensive experiments demonstrate the effectiveness of applying our method to various multimodal architectures, leading to consistent improvement across diverse tasks and showcasing its potential for enhancing visual and scene-text understanding.

{{</citation>}}


### (14/125) Minecraft-ify: Minecraft Style Image Generation with Text-guided Image Editing for In-Game Application (Bumsoo Kim et al., 2024)

{{<citation>}}

Bumsoo Kim, Sanghyun Byun, Yonghoon Jung, Wonseop Shin, Sareer UI Amin, Sanghyun Seo. (2024)  
**Minecraft-ify: Minecraft Style Image Generation with Text-guided Image Editing for In-Game Application**
<br/>
<button class="copy-to-clipboard" title="Minecraft-ify: Minecraft Style Image Generation with Text-guided Image Editing for In-Game Application" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-GR, cs-LG, cs-MM, cs.CV  
Keywords: AI  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05448v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05448v1.pdf" filename="2402.05448v1.pdf">Download PDF</button>

---


**ABSTRACT**  
In this paper, we first present the character texture generation system \textit{Minecraft-ify}, specified to Minecraft video game toward in-game application. Ours can generate face-focused image for texture mapping tailored to 3D virtual character having cube manifold. While existing projects or works only generate texture, proposed system can inverse the user-provided real image, or generate average/random appearance from learned distribution. Moreover, it can be manipulated with text-guidance using StyleGAN and StyleCLIP. These features provide a more extended user experience with enlarged freedom as a user-friendly AI-tool. Project page can be found at https://gh-bumsookim.github.io/Minecraft-ify/

{{</citation>}}


### (15/125) MTSA-SNN: A Multi-modal Time Series Analysis Model Based on Spiking Neural Network (Chengzhi Liu et al., 2024)

{{<citation>}}

Chengzhi Liu, Chong Zhong, Mingyu Jin, Zheng Tao, Zihong Luo, Chenghao Liu, Shuliang Zhao. (2024)  
**MTSA-SNN: A Multi-modal Time Series Analysis Model Based on Spiking Neural Network**
<br/>
<button class="copy-to-clipboard" title="MTSA-SNN: A Multi-modal Time Series Analysis Model Based on Spiking Neural Network" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Time Series  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05423v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05423v1.pdf" filename="2402.05423v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Time series analysis and modelling constitute a crucial research area. Traditional artificial neural networks struggle with complex, non-stationary time series data due to high computational complexity, limited ability to capture temporal information, and difficulty in handling event-driven data. To address these challenges, we propose a Multi-modal Time Series Analysis Model Based on Spiking Neural Network (MTSA-SNN). The Pulse Encoder unifies the encoding of temporal images and sequential information in a common pulse-based representation. The Joint Learning Module employs a joint learning function and weight allocation mechanism to fuse information from multi-modal pulse signals complementary. Additionally, we incorporate wavelet transform operations to enhance the model's ability to analyze and evaluate temporal information. Experimental results demonstrate that our method achieved superior performance on three complex time-series tasks. This work provides an effective event-driven approach to overcome the challenges associated with analyzing intricate temporal information. Access to the source code is available at https://github.com/Chenngzz/MTSA-SNN}{https://github.com/Chenngzz/MTSA-SNN

{{</citation>}}


### (16/125) Segmentation-free Connectionist Temporal Classification loss based OCR Model for Text Captcha Classification (Vaibhav Khatavkar et al., 2024)

{{<citation>}}

Vaibhav Khatavkar, Makarand Velankar, Sneha Petkar. (2024)  
**Segmentation-free Connectionist Temporal Classification loss based OCR Model for Text Captcha Classification**
<br/>
<button class="copy-to-clipboard" title="Segmentation-free Connectionist Temporal Classification loss based OCR Model for Text Captcha Classification" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: OCR  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05417v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05417v1.pdf" filename="2402.05417v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Captcha are widely used to secure systems from automatic responses by distinguishing computer responses from human responses. Text, audio, video, picture picture-based Optical Character Recognition (OCR) are used for creating captcha. Text-based OCR captcha are the most often used captcha which faces issues namely, complex and distorted contents. There are attempts to build captcha detection and classification-based systems using machine learning and neural networks, which need to be tuned for accuracy. The existing systems face challenges in the recognition of distorted characters, handling variable-length captcha and finding sequential dependencies in captcha. In this work, we propose a segmentation-free OCR model for text captcha classification based on the connectionist temporal classification loss technique. The proposed model is trained and tested on a publicly available captcha dataset. The proposed model gives 99.80\% character level accuracy, while 95\% word level accuracy. The accuracy of the proposed model is compared with the state-of-the-art models and proves to be effective. The variable length complex captcha can be thus processed with the segmentation-free connectionist temporal classification loss technique with dependencies which will be massively used in securing the software systems.

{{</citation>}}


### (17/125) On the Effect of Image Resolution on Semantic Segmentation (Ritambhara Singh et al., 2024)

{{<citation>}}

Ritambhara Singh, Abhishek Jain, Pietro Perona, Shivani Agarwal, Junfeng Yang. (2024)  
**On the Effect of Image Resolution on Semantic Segmentation**
<br/>
<button class="copy-to-clipboard" title="On the Effect of Image Resolution on Semantic Segmentation" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05398v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05398v1.pdf" filename="2402.05398v1.pdf">Download PDF</button>

---


**ABSTRACT**  
High-resolution semantic segmentation requires substantial computational resources. Traditional approaches in the field typically downscale the input images before processing and then upscale the low-resolution outputs back to their original dimensions. While this strategy effectively identifies broad regions, it often misses finer details. In this study, we demonstrate that a streamlined model capable of directly producing high-resolution segmentations can match the performance of more complex systems that generate lower-resolution results. By simplifying the network architecture, we enable the processing of images at their native resolution. Our approach leverages a bottom-up information propagation technique across various scales, which we have empirically shown to enhance segmentation accuracy. We have rigorously tested our method using leading-edge semantic segmentation datasets. Specifically, for the Cityscapes dataset, we further boost accuracy by applying the Noisy Student Training technique.

{{</citation>}}


### (18/125) Enhancing Zero-shot Counting via Language-guided Exemplar Learning (Mingjie Wang et al., 2024)

{{<citation>}}

Mingjie Wang, Jun Zhou, Yong Dai, Eric Buys, Minglun Gong. (2024)  
**Enhancing Zero-shot Counting via Language-guided Exemplar Learning**
<br/>
<button class="copy-to-clipboard" title="Enhancing Zero-shot Counting via Language-guided Exemplar Learning" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05394v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05394v1.pdf" filename="2402.05394v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Recently, Class-Agnostic Counting (CAC) problem has garnered increasing attention owing to its intriguing generality and superior efficiency compared to Category-Specific Counting (CSC). This paper proposes a novel ExpressCount to enhance zero-shot object counting by delving deeply into language-guided exemplar learning. Specifically, the ExpressCount is comprised of an innovative Language-oriented Exemplar Perceptron and a downstream visual Zero-shot Counting pipeline. Thereinto, the perceptron hammers at exploiting accurate exemplar cues from collaborative language-vision signals by inheriting rich semantic priors from the prevailing pre-trained Large Language Models (LLMs), whereas the counting pipeline excels in mining fine-grained features through dual-branch and cross-attention schemes, contributing to the high-quality similarity learning. Apart from building a bridge between the LLM in vogue and the visual counting tasks, expression-guided exemplar estimation significantly advances zero-shot learning capabilities for counting instances with arbitrary classes. Moreover, devising a FSC-147-Express with annotations of meticulous linguistic expressions pioneers a new venue for developing and validating language-based counting models. Extensive experiments demonstrate the state-of-the-art performance of our ExpressCount, even showcasing the accuracy on par with partial CSC models.

{{</citation>}}


### (19/125) CIC: A framework for Culturally-aware Image Captioning (Youngsik Yun et al., 2024)

{{<citation>}}

Youngsik Yun, Jihie Kim. (2024)  
**CIC: A framework for Culturally-aware Image Captioning**
<br/>
<button class="copy-to-clipboard" title="CIC: A framework for Culturally-aware Image Captioning" index=0>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-0 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Image Captioning, Language Model, QA, Question Answering  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05374v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05374v1.pdf" filename="2402.05374v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Image Captioning generates descriptive sentences from images using Vision-Language Pre-trained models (VLPs) such as BLIP, which has improved greatly. However, current methods lack the generation of detailed descriptive captions for the cultural elements depicted in the images, such as the traditional clothing worn by people from Asian cultural groups. In this paper, we propose a new framework, \textbf{Culturally-aware Image Captioning (CIC)}, that generates captions and describes cultural elements extracted from cultural visual elements in images representing cultures. Inspired by methods combining visual modality and Large Language Models (LLMs) through appropriate prompts, our framework (1) generates questions based on cultural categories from images, (2) extracts cultural visual elements from Visual Question Answering (VQA) using generated questions, and (3) generates culturally-aware captions using LLMs with the prompts. Our human evaluation conducted on 45 participants from 4 different cultural groups with a high understanding of the corresponding culture shows that our proposed framework generates more culturally descriptive captions when compared to the image captioning baseline based on VLPs. Our code and dataset will be made publicly available upon acceptance.

{{</citation>}}


## cs.LG (32)



### (20/125) Classifying Nodes in Graphs without GNNs (Daniel Winter et al., 2024)

{{<citation>}}

Daniel Winter, Niv Cohen, Yedid Hoshen. (2024)  
**Classifying Nodes in Graphs without GNNs**
<br/>
<button class="copy-to-clipboard" title="Classifying Nodes in Graphs without GNNs" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05934v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05934v1.pdf" filename="2402.05934v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Graph neural networks (GNNs) are the dominant paradigm for classifying nodes in a graph, but they have several undesirable attributes stemming from their message passing architecture. Recently, distillation methods succeeded in eliminating the use of GNNs at test time but they still require them during training. We perform a careful analysis of the role that GNNs play in distillation methods. This analysis leads us to propose a fully GNN-free approach for node classification, not requiring them at train or test time. Our method consists of three key components: smoothness constraints, pseudo-labeling iterations and neighborhood-label histograms. Our final approach can match the state-of-the-art accuracy on standard popular benchmarks such as citation and co-purchase networks, without training a GNN.

{{</citation>}}


### (21/125) Time Series Diffusion in the Frequency Domain (Jonathan Crabbé et al., 2024)

{{<citation>}}

Jonathan Crabbé, Nicolas Huynh, Jan Stanczuk, Mihaela van der Schaar. (2024)  
**Time Series Diffusion in the Frequency Domain**
<br/>
<button class="copy-to-clipboard" title="Time Series Diffusion in the Frequency Domain" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05933v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05933v1.pdf" filename="2402.05933v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Fourier analysis has been an instrumental tool in the development of signal processing. This leads us to wonder whether this framework could similarly benefit generative modelling. In this paper, we explore this question through the scope of time series diffusion models. More specifically, we analyze whether representing time series in the frequency domain is a useful inductive bias for score-based diffusion models. By starting from the canonical SDE formulation of diffusion in the time domain, we show that a dual diffusion process occurs in the frequency domain with an important nuance: Brownian motions are replaced by what we call mirrored Brownian motions, characterized by mirror symmetries among their components. Building on this insight, we show how to adapt the denoising score matching approach to implement diffusion models in the frequency domain. This results in frequency diffusion models, which we compare to canonical time diffusion models. Our empirical evaluation on real-world datasets, covering various domains like healthcare and finance, shows that frequency diffusion models better capture the training distribution than time diffusion models. We explain this observation by showing that time series from these datasets tend to be more localized in the frequency domain than in the time domain, which makes them easier to model in the former case. All our observations point towards impactful synergies between Fourier analysis and diffusion models.

{{</citation>}}


### (22/125) On the Convergence of Zeroth-Order Federated Tuning in Large Language Models (Zhenqing Ling et al., 2024)

{{<citation>}}

Zhenqing Ling, Daoyuan Chen, Liuyi Yao, Yaliang Li, Ying Shen. (2024)  
**On the Convergence of Zeroth-Order Federated Tuning in Large Language Models**
<br/>
<button class="copy-to-clipboard" title="On the Convergence of Zeroth-Order Federated Tuning in Large Language Models" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05926v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05926v1.pdf" filename="2402.05926v1.pdf">Download PDF</button>

---


**ABSTRACT**  
The confluence of Federated Learning (FL) and Large Language Models (LLMs) is ushering in a new era in privacy-preserving natural language processing. However, the intensive memory requirements for fine-tuning LLMs pose significant challenges, especially when deploying on edge devices with limited computational resources. To circumvent this, we explore the novel integration of Memory-efficient Zeroth-Order Optimization within a federated setting, a synergy we denote as FedMeZO. Our study is the first to examine the theoretical underpinnings of FedMeZO in the context of LLMs, tackling key questions regarding the influence of large parameter spaces on optimization behavior, the establishment of convergence properties, and the identification of critical parameters for convergence to inform personalized federated strategies. Our extensive empirical evidence supports the theory, showing that FedMeZO not only converges faster than traditional first-order methods such as SGD but also significantly reduces GPU memory usage during training to levels comparable to those during inference. Moreover, the proposed personalized FL strategy that is built upon the theoretical insights to customize the client-wise learning rate can effectively accelerate loss reduction. We hope our work can help to bridge theoretical and practical aspects of federated fine-tuning for LLMs and facilitate further development and research.

{{</citation>}}


### (23/125) Risk-Sensitive Multi-Agent Reinforcement Learning in Network Aggregative Markov Games (Hafez Ghaemi et al., 2024)

{{<citation>}}

Hafez Ghaemi, Hamed Kebriaei, Alireza Ramezani Moghaddam, Majid Nili Ahamdabadi. (2024)  
**Risk-Sensitive Multi-Agent Reinforcement Learning in Network Aggregative Markov Games**
<br/>
<button class="copy-to-clipboard" title="Risk-Sensitive Multi-Agent Reinforcement Learning in Network Aggregative Markov Games" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: I-2-6; I-2-11, cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05906v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05906v1.pdf" filename="2402.05906v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Classical multi-agent reinforcement learning (MARL) assumes risk neutrality and complete objectivity for agents. However, in settings where agents need to consider or model human economic or social preferences, a notion of risk must be incorporated into the RL optimization problem. This will be of greater importance in MARL where other human or non-human agents are involved, possibly with their own risk-sensitive policies. In this work, we consider risk-sensitive and non-cooperative MARL with cumulative prospect theory (CPT), a non-convex risk measure and a generalization of coherent measures of risk. CPT is capable of explaining loss aversion in humans and their tendency to overestimate/underestimate small/large probabilities. We propose a distributed sampling-based actor-critic (AC) algorithm with CPT risk for network aggregative Markov games (NAMGs), which we call Distributed Nested CPT-AC. Under a set of assumptions, we prove the convergence of the algorithm to a subjective notion of Markov perfect Nash equilibrium in NAMGs. The experimental results show that subjective CPT policies obtained by our algorithm can be different from the risk-neutral ones, and agents with a higher loss aversion are more inclined to socially isolate themselves in an NAMG.

{{</citation>}}


### (24/125) Federated Offline Reinforcement Learning: Collaborative Single-Policy Coverage Suffices (Jiin Woo et al., 2024)

{{<citation>}}

Jiin Woo, Laixi Shi, Gauri Joshi, Yuejie Chi. (2024)  
**Federated Offline Reinforcement Learning: Collaborative Single-Policy Coverage Suffices**
<br/>
<button class="copy-to-clipboard" title="Federated Offline Reinforcement Learning: Collaborative Single-Policy Coverage Suffices" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MA, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05876v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05876v1.pdf" filename="2402.05876v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Offline reinforcement learning (RL), which seeks to learn an optimal policy using offline data, has garnered significant interest due to its potential in critical applications where online data collection is infeasible or expensive. This work explores the benefit of federated learning for offline RL, aiming at collaboratively leveraging offline datasets at multiple agents. Focusing on finite-horizon episodic tabular Markov decision processes (MDPs), we design FedLCB-Q, a variant of the popular model-free Q-learning algorithm tailored for federated offline RL. FedLCB-Q updates local Q-functions at agents with novel learning rate schedules and aggregates them at a central server using importance averaging and a carefully designed pessimistic penalty term. Our sample complexity analysis reveals that, with appropriately chosen parameters and synchronization schedules, FedLCB-Q achieves linear speedup in terms of the number of agents without requiring high-quality datasets at individual agents, as long as the local datasets collectively cover the state-action space visited by the optimal policy, highlighting the power of collaboration in the federated setting. In fact, the sample complexity almost matches that of the single-agent counterpart, as if all the data are stored at a central location, up to polynomial factors of the horizon length. Furthermore, FedLCB-Q is communication-efficient, where the number of communication rounds is only linear with respect to the horizon length up to logarithmic factors.

{{</citation>}}


### (25/125) Let Your Graph Do the Talking: Encoding Structured Data for LLMs (Bryan Perozzi et al., 2024)

{{<citation>}}

Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsitsulin, Mehran Kazemi, Rami Al-Rfou, Jonathan Halcrow. (2024)  
**Let Your Graph Do the Talking: Encoding Structured Data for LLMs**
<br/>
<button class="copy-to-clipboard" title="Let Your Graph Do the Talking: Encoding Structured Data for LLMs" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: I-5-1; I-2-6; I-2-7, cs-AI, cs-LG, cs-SI, cs.LG, stat-ML  
Keywords: QA  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05862v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05862v1.pdf" filename="2402.05862v1.pdf">Download PDF</button>

---


**ABSTRACT**  
How can we best encode structured data into sequential form for use in large language models (LLMs)? In this work, we introduce a parameter-efficient method to explicitly represent structured data for LLMs. Our method, GraphToken, learns an encoding function to extend prompts with explicit structured information. Unlike other work which focuses on limited domains (e.g. knowledge graph representation), our work is the first effort focused on the general encoding of structured data to be used for various reasoning tasks. We show that explicitly representing the graph structure allows significant improvements to graph reasoning tasks. Specifically, we see across the board improvements - up to 73% points - on node, edge and, graph-level tasks from the GraphQA benchmark.

{{</citation>}}


### (26/125) Learning to Route Among Specialized Experts for Zero-Shot Generalization (Mohammed Muqeeth et al., 2024)

{{<citation>}}

Mohammed Muqeeth, Haokun Liu, Yufan Liu, Colin Raffel. (2024)  
**Learning to Route Among Specialized Experts for Zero-Shot Generalization**
<br/>
<button class="copy-to-clipboard" title="Learning to Route Among Specialized Experts for Zero-Shot Generalization" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Zero-Shot  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05859v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05859v1.pdf" filename="2402.05859v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Recently, there has been a widespread proliferation of "expert" language models that are specialized to a specific task or domain through parameter-efficient fine-tuning. How can we recycle large collections of expert language models to improve zero-shot generalization to unseen tasks? In this work, we propose Post-Hoc Adaptive Tokenwise Gating Over an Ocean of Specialized Experts (PHATGOOSE), which learns to route among specialized modules that were produced through parameter-efficient fine-tuning. Unlike past methods that learn to route among specialized models, PHATGOOSE explores the possibility that zero-shot generalization will be improved if different experts can be adaptively chosen for each token and at each layer in the model. Crucially, our method is post-hoc - it does not require simultaneous access to the datasets used to create the specialized models and only requires a modest amount of additional compute after each expert model is trained. In experiments covering a range of specialized model collections and zero-shot generalization benchmarks, we find that PHATGOOSE outperforms past methods for post-hoc routing and, in some cases, outperforms explicit multitask training (which requires simultaneous data access). To better understand the routing strategy learned by PHATGOOSE, we perform qualitative experiments to validate that PHATGOOSE's performance stems from its ability to make adaptive per-token and per-module expert choices. We release all of our code to support future work on improving zero-shot generalization by recycling specialized experts.

{{</citation>}}


### (27/125) Sparse-VQ Transformer: An FFN-Free Framework with Vector Quantization for Enhanced Time Series Forecasting (Yanjun Zhao et al., 2024)

{{<citation>}}

Yanjun Zhao, Tian Zhou, Chao Chen, Liang Sun, Yi Qian, Rong Jin. (2024)  
**Sparse-VQ Transformer: An FFN-Free Framework with Vector Quantization for Enhanced Time Series Forecasting**
<br/>
<button class="copy-to-clipboard" title="Sparse-VQ Transformer: An FFN-Free Framework with Vector Quantization for Enhanced Time Series Forecasting" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, NLP, Quantization, Time Series, Transformer  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05830v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05830v1.pdf" filename="2402.05830v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Time series analysis is vital for numerous applications, and transformers have become increasingly prominent in this domain. Leading methods customize the transformer architecture from NLP and CV, utilizing a patching technique to convert continuous signals into segments. Yet, time series data are uniquely challenging due to significant distribution shifts and intrinsic noise levels. To address these two challenges,we introduce the Sparse Vector Quantized FFN-Free Transformer (Sparse-VQ). Our methodology capitalizes on a sparse vector quantization technique coupled with Reverse Instance Normalization (RevIN) to reduce noise impact and capture sufficient statistics for forecasting, serving as an alternative to the Feed-Forward layer (FFN) in the transformer architecture. Our FFN-free approach trims the parameter count, enhancing computational efficiency and reducing overfitting. Through evaluations across ten benchmark datasets, including the newly introduced CAISO dataset, Sparse-VQ surpasses leading models with a 7.84% and 4.17% decrease in MAE for univariate and multivariate time series forecasting, respectively. Moreover, it can be seamlessly integrated with existing transformer-based models to elevate their performance.

{{</citation>}}


### (28/125) Discovering Temporally-Aware Reinforcement Learning Algorithms (Matthew Thomas Jackson et al., 2024)

{{<citation>}}

Matthew Thomas Jackson, Chris Lu, Louis Kirsch, Robert Tjarko Lange, Shimon Whiteson, Jakob Nicolaus Foerster. (2024)  
**Discovering Temporally-Aware Reinforcement Learning Algorithms**
<br/>
<button class="copy-to-clipboard" title="Discovering Temporally-Aware Reinforcement Learning Algorithms" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05828v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05828v1.pdf" filename="2402.05828v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Recent advancements in meta-learning have enabled the automatic discovery of novel reinforcement learning algorithms parameterized by surrogate objective functions. To improve upon manually designed algorithms, the parameterization of this learned objective function must be expressive enough to represent novel principles of learning (instead of merely recovering already established ones) while still generalizing to a wide range of settings outside of its meta-training distribution. However, existing methods focus on discovering objective functions that, like many widely used objective functions in reinforcement learning, do not take into account the total number of steps allowed for training, or "training horizon". In contrast, humans use a plethora of different learning objectives across the course of acquiring a new ability. For instance, students may alter their studying techniques based on the proximity to exam deadlines and their self-assessed capabilities. This paper contends that ignoring the optimization time horizon significantly restricts the expressive potential of discovered learning algorithms. We propose a simple augmentation to two existing objective discovery approaches that allows the discovered algorithm to dynamically update its objective function throughout the agent's training procedure, resulting in expressive schedules and increased generalization across different training horizons. In the process, we find that commonly used meta-gradient approaches fail to discover such adaptive objective functions while evolution strategies discover highly dynamic learning rules. We demonstrate the effectiveness of our approach on a wide range of tasks and analyze the resulting learned algorithms, which we find effectively balance exploration and exploitation by modifying the structure of their learning rules throughout the agent's lifetime.

{{</citation>}}


### (29/125) Guided Evolution with Binary Discriminators for ML Program Search (John D. Co-Reyes et al., 2024)

{{<citation>}}

John D. Co-Reyes, Yingjie Miao, George Tucker, Aleksandra Faust, Esteban Real. (2024)  
**Guided Evolution with Binary Discriminators for ML Program Search**
<br/>
<button class="copy-to-clipboard" title="Guided Evolution with Binary Discriminators for ML Program Search" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: GNN  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05821v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05821v1.pdf" filename="2402.05821v1.pdf">Download PDF</button>

---


**ABSTRACT**  
How to automatically design better machine learning programs is an open problem within AutoML. While evolution has been a popular tool to search for better ML programs, using learning itself to guide the search has been less successful and less understood on harder problems but has the promise to dramatically increase the speed and final performance of the optimization process. We propose guiding evolution with a binary discriminator, trained online to distinguish which program is better given a pair of programs. The discriminator selects better programs without having to perform a costly evaluation and thus speed up the convergence of evolution. Our method can encode a wide variety of ML components including symbolic optimizers, neural architectures, RL loss functions, and symbolic regression equations with the same directed acyclic graph representation. By combining this representation with modern GNNs and an adaptive mutation strategy, we demonstrate our method can speed up evolution across a set of diverse problems including a 3.7x speedup on the symbolic search for ML optimizers and a 4x speedup for RL loss functions.

{{</citation>}}


### (30/125) Unsupervised Discovery of Clinical Disease Signatures Using Probabilistic Independence (Thomas A. Lasko et al., 2024)

{{<citation>}}

Thomas A. Lasko, John M. Still, Thomas Z. Li, Marco Barbero Mota, William W. Stead, Eric V. Strobl, Bennett A. Landman, Fabien Maldonado. (2024)  
**Unsupervised Discovery of Clinical Disease Signatures Using Probabilistic Independence**
<br/>
<button class="copy-to-clipboard" title="Unsupervised Discovery of Clinical Disease Signatures Using Probabilistic Independence" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: I-2-6; I-2-1; J-3, cs-LG, cs.LG, stat-AP, stat-ML  
Keywords: Clinical  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05802v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05802v1.pdf" filename="2402.05802v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Insufficiently precise diagnosis of clinical disease is likely responsible for many treatment failures, even for common conditions and treatments. With a large enough dataset, it may be possible to use unsupervised machine learning to define clinical disease patterns more precisely. We present an approach to learning these patterns by using probabilistic independence to disentangle the imprint on the medical record of causal latent sources of disease. We inferred a broad set of 2000 clinical signatures of latent sources from 9195 variables in 269,099 Electronic Health Records. The learned signatures produced better discrimination than the original variables in a lung cancer prediction task unknown to the inference algorithm, predicting 3-year malignancy in patients with no history of cancer before a solitary lung nodule was discovered. More importantly, the signatures' greater explanatory power identified pre-nodule signatures of apparently undiagnosed cancer in many of those patients.

{{</citation>}}


### (31/125) Limits of Transformer Language Models on Algorithmic Learning (Jonathan Thomm et al., 2024)

{{<citation>}}

Jonathan Thomm, Aleksandar Terzic, Geethan Karunaratne, Giacomo Camposampiero, Bernhard Schölkopf, Abbas Rahimi. (2024)  
**Limits of Transformer Language Models on Algorithmic Learning**
<br/>
<button class="copy-to-clipboard" title="Limits of Transformer Language Models on Algorithmic Learning" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GPT, GPT-4, LLaMA, Language Model, Transformer  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05785v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05785v1.pdf" filename="2402.05785v1.pdf">Download PDF</button>

---


**ABSTRACT**  
We analyze the capabilities of Transformer language models on learning discrete algorithms. To this end, we introduce two new tasks demanding the composition of several discrete sub-tasks. On both training LLaMA models from scratch and prompting on GPT-4 and Gemini we measure learning compositions of learned primitives. We observe that the compositional capabilities of state-of-the-art Transformer language models are very limited and sample-wise scale worse than relearning all sub-tasks for a new algorithmic composition. We also present a theorem in complexity theory, showing that gradient descent on memorizing feedforward models can be exponentially data inefficient.

{{</citation>}}


### (32/125) Implicit Bias and Fast Convergence Rates for Self-attention (Bhavya Vasudeva et al., 2024)

{{<citation>}}

Bhavya Vasudeva, Puneesh Deora, Christos Thrampoulidis. (2024)  
**Implicit Bias and Fast Convergence Rates for Self-attention**
<br/>
<button class="copy-to-clipboard" title="Implicit Bias and Fast Convergence Rates for Self-attention" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC, stat-ML  
Keywords: Bias  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05738v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05738v1.pdf" filename="2402.05738v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Self-attention, the core mechanism of transformers, distinguishes them from traditional neural networks and drives their outstanding performance. Towards developing the fundamental optimization principles of self-attention, we investigate the implicit bias of gradient descent (GD) in training a self-attention layer with fixed linear decoder in binary classification. Drawing inspiration from the study of GD in linear logistic regression over separable data, recent work demonstrates that as the number of iterations $t$ approaches infinity, the key-query matrix $W_t$ converges locally (with respect to the initialization direction) to a hard-margin SVM solution $W_{mm}$. Our work enhances this result in four aspects. Firstly, we identify non-trivial data settings for which convergence is provably global, thus shedding light on the optimization landscape. Secondly, we provide the first finite-time convergence rate for $W_t$ to $W_{mm}$, along with quantifying the rate of sparsification in the attention map. Thirdly, through an analysis of normalized GD and Polyak step-size, we demonstrate analytically that adaptive step-size rules can accelerate the convergence of self-attention. Additionally, we remove the restriction of prior work on a fixed linear decoder. Our results reinforce the implicit-bias perspective of self-attention and strengthen its connections to implicit-bias in linear logistic regression, despite the intricate non-convex nature of the former.

{{</citation>}}


### (33/125) Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations (Pranav Kulkarni et al., 2024)

{{<citation>}}

Pranav Kulkarni, Andrew Chan, Nithya Navarathna, Skylar Chan, Paul H. Yi, Vishwa S. Parekh. (2024)  
**Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations**
<br/>
<button class="copy-to-clipboard" title="Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI, Bias  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05713v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05713v1.pdf" filename="2402.05713v1.pdf">Download PDF</button>

---


**ABSTRACT**  
The proliferation of artificial intelligence (AI) in radiology has shed light on the risk of deep learning (DL) models exacerbating clinical biases towards vulnerable patient populations. While prior literature has focused on quantifying biases exhibited by trained DL models, demographically targeted adversarial bias attacks on DL models and its implication in the clinical environment remains an underexplored field of research in medical imaging. In this work, we demonstrate that demographically targeted label poisoning attacks can introduce adversarial underdiagnosis bias in DL models and degrade performance on underrepresented groups without impacting overall model performance. Moreover, our results across multiple performance metrics and demographic groups like sex, age, and their intersectional subgroups indicate that a group's vulnerability to undetectable adversarial bias attacks is directly correlated with its representation in the model's training data.

{{</citation>}}


### (34/125) Is Adversarial Training with Compressed Datasets Effective? (Tong Chen et al., 2024)

{{<citation>}}

Tong Chen, Raghavendra Selvan. (2024)  
**Is Adversarial Training with Compressed Datasets Effective?**
<br/>
<button class="copy-to-clipboard" title="Is Adversarial Training with Compressed Datasets Effective?" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Adversarial Training  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05675v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05675v1.pdf" filename="2402.05675v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Dataset Condensation (DC) refers to the recent class of dataset compression methods that generate a smaller, synthetic, dataset from a larger dataset. This synthetic dataset retains the essential information of the original dataset, enabling models trained on it to achieve performance levels comparable to those trained on the full dataset. Most current DC methods have mainly concerned with achieving high test performance with limited data budget, and have not directly addressed the question of adversarial robustness. In this work, we investigate the impact of adversarial robustness on models trained with compressed datasets. We show that the compressed datasets obtained from DC methods are not effective in transferring adversarial robustness to models. As a solution to improve dataset compression efficiency and adversarial robustness simultaneously, we propose a novel robustness-aware dataset compression method based on finding the Minimal Finite Covering (MFC) of the dataset. The proposed method is (1) obtained by one-time computation and is applicable for any model, (2) more effective than DC methods when applying adversarial training over MFC, (3) provably robust by minimizing the generalized adversarial loss. Additionally, empirical evaluation on three datasets shows that the proposed method is able to achieve better robustness and performance trade-off compared to DC methods such as distribution matching.

{{</citation>}}


### (35/125) Mesoscale Traffic Forecasting for Real-Time Bottleneck and Shockwave Prediction (Raphael Chekroun et al., 2024)

{{<citation>}}

Raphael Chekroun, Han Wang, Jonathan Lee, Marin Toromanoff, Sascha Hornauer, Fabien Moutarde, Maria Laura Delle Monache. (2024)  
**Mesoscale Traffic Forecasting for Real-Time Bottleneck and Shockwave Prediction**
<br/>
<button class="copy-to-clipboard" title="Mesoscale Traffic Forecasting for Real-Time Bottleneck and Shockwave Prediction" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Attention, LSTM, Self-Attention  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05663v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05663v1.pdf" filename="2402.05663v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Accurate real-time traffic state forecasting plays a pivotal role in traffic control research. In particular, the CIRCLES consortium project necessitates predictive techniques to mitigate the impact of data source delays. After the success of the MegaVanderTest experiment, this paper aims at overcoming the current system limitations and develop a more suited approach to improve the real-time traffic state estimation for the next iterations of the experiment. In this paper, we introduce the SA-LSTM, a deep forecasting method integrating Self-Attention (SA) on the spatial dimension with Long Short-Term Memory (LSTM) yielding state-of-the-art results in real-time mesoscale traffic forecasting. We extend this approach to multi-step forecasting with the n-step SA-LSTM, which outperforms traditional multi-step forecasting methods in the trade-off between short-term and long-term predictions, all while operating in real-time.

{{</citation>}}


### (36/125) Rethinking Propagation for Unsupervised Graph Domain Adaptation (Meihan Liu et al., 2024)

{{<citation>}}

Meihan Liu, Zeyu Fang, Zhen Zhang, Ming Gu, Sheng Zhou, Xin Wang, Jiajun Bu. (2024)  
**Rethinking Propagation for Unsupervised Graph Domain Adaptation**
<br/>
<button class="copy-to-clipboard" title="Rethinking Propagation for Unsupervised Graph Domain Adaptation" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05660v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05660v1.pdf" filename="2402.05660v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Unsupervised Graph Domain Adaptation (UGDA) aims to transfer knowledge from a labelled source graph to an unlabelled target graph in order to address the distribution shifts between graph domains. Previous works have primarily focused on aligning data from the source and target graph in the representation space learned by graph neural networks (GNNs). However, the inherent generalization capability of GNNs has been largely overlooked. Motivated by our empirical analysis, we reevaluate the role of GNNs in graph domain adaptation and uncover the pivotal role of the propagation process in GNNs for adapting to different graph domains. We provide a comprehensive theoretical analysis of UGDA and derive a generalization bound for multi-layer GNNs. By formulating GNN Lipschitz for k-layer GNNs, we show that the target risk bound can be tighter by removing propagation layers in source graph and stacking multiple propagation layers in target graph. Based on the empirical and theoretical analysis mentioned above, we propose a simple yet effective approach called A2GNN for graph domain adaptation. Through extensive experiments on real-world datasets, we demonstrate the effectiveness of our proposed A2GNN framework.

{{</citation>}}


### (37/125) Improving Token-Based World Models with Parallel Observation Prediction (Lior Cohen et al., 2024)

{{<citation>}}

Lior Cohen, Kaixin Wang, Bingyi Kang, Shie Mannor. (2024)  
**Improving Token-Based World Models with Parallel Observation Prediction**
<br/>
<button class="copy-to-clipboard" title="Improving Token-Based World Models with Parallel Observation Prediction" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05643v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05643v1.pdf" filename="2402.05643v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Motivated by the success of Transformers when applied to sequences of discrete symbols, token-based world models (TBWMs) were recently proposed as sample-efficient methods. In TBWMs, the world model consumes agent experience as a language-like sequence of tokens, where each observation constitutes a sub-sequence. However, during imagination, the sequential token-by-token generation of next observations results in a severe bottleneck, leading to long training times, poor GPU utilization, and limited representations. To resolve this bottleneck, we devise a novel Parallel Observation Prediction (POP) mechanism. POP augments a Retentive Network (RetNet) with a novel forward mode tailored to our reinforcement learning setting. We incorporate POP in a novel TBWM agent named REM (Retentive Environment Model), showcasing a 15.4x faster imagination compared to prior TBWMs. REM attains superhuman performance on 12 out of 26 games of the Atari 100K benchmark, while training in less than 12 hours. Our code is available at \url{https://github.com/leor-c/REM}.

{{</citation>}}


### (38/125) RepQuant: Towards Accurate Post-Training Quantization of Large Transformer Models via Scale Reparameterization (Zhikai Li et al., 2024)

{{<citation>}}

Zhikai Li, Xuewen Liu, Jing Zhang, Qingyi Gu. (2024)  
**RepQuant: Towards Accurate Post-Training Quantization of Large Transformer Models via Scale Reparameterization**
<br/>
<button class="copy-to-clipboard" title="RepQuant: Towards Accurate Post-Training Quantization of Large Transformer Models via Scale Reparameterization" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Quantization, Transformer  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05628v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05628v1.pdf" filename="2402.05628v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Large transformer models have demonstrated remarkable success. Post-training quantization (PTQ), which requires only a small dataset for calibration and avoids end-to-end retraining, is a promising solution for compressing these large models. Regrettably, existing PTQ methods typically exhibit non-trivial performance loss. We find that the performance bottleneck stems from over-consideration of hardware compatibility in the quantization process, compelling them to reluctantly employ simple quantizers, albeit at the expense of accuracy. With the above insights, we propose RepQuant, a novel PTQ framework with quantization-inference decoupling paradigm to address the above issues. RepQuant employs complex quantizers in the quantization process and simplified quantizers in the inference process, and performs mathematically equivalent transformations between the two through quantization scale reparameterization, thus ensuring both accurate quantization and efficient inference. More specifically, we focus on two components with extreme distributions: LayerNorm activations and Softmax activations. Initially, we apply channel-wise quantization and log$\sqrt{2}$ quantization, respectively, which are tailored to their distributions. In particular, for the former, we introduce a learnable per-channel dual clipping scheme, which is designed to efficiently identify outliers in the unbalanced activations with fine granularity. Then, we reparameterize the scales to hardware-friendly layer-wise quantization and log2 quantization for inference. Moreover, quantized weight reconstruction is seamlessly integrated into the above procedure to further push the performance limits. Extensive experiments are performed on different large-scale transformer variants on multiple tasks, including vision, language, and multi-modal transformers, and RepQuant encouragingly demonstrates significant performance advantages.

{{</citation>}}


### (39/125) The Loss Landscape of Shallow ReLU-like Neural Networks: Stationary Points, Saddle Escaping, and Network Embedding (Zhengqing Wu et al., 2024)

{{<citation>}}

Zhengqing Wu, Berfin Simsek, Francois Ged. (2024)  
**The Loss Landscape of Shallow ReLU-like Neural Networks: Stationary Points, Saddle Escaping, and Network Embedding**
<br/>
<button class="copy-to-clipboard" title="The Loss Landscape of Shallow ReLU-like Neural Networks: Stationary Points, Saddle Escaping, and Network Embedding" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05626v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05626v1.pdf" filename="2402.05626v1.pdf">Download PDF</button>

---


**ABSTRACT**  
In this paper, we investigate the loss landscape of one-hidden-layer neural networks with ReLU-like activation functions trained with the empirical squared loss. As the activation function is non-differentiable, it is so far unclear how to completely characterize the stationary points. We propose the conditions for stationarity that apply to both non-differentiable and differentiable cases. Additionally, we show that, if a stationary point does not contain "escape neurons", which are defined with first-order conditions, then it must be a local minimum. Moreover, for the scalar-output case, the presence of an escape neuron guarantees that the stationary point is not a local minimum. Our results refine the description of the saddle-to-saddle training process starting from infinitesimally small (vanishing) initialization for shallow ReLU-like networks, linking saddle escaping directly with the parameter changes of escape neurons. Moreover, we are also able to fully discuss how network embedding, which is to instantiate a narrower network within a wider network, reshapes the stationary points.

{{</citation>}}


### (40/125) Hypergraph Node Classification With Graph Neural Networks (Bohan Tang et al., 2024)

{{<citation>}}

Bohan Tang, Zexi Liu, Keyue Jiang, Siheng Chen, Xiaowen Dong. (2024)  
**Hypergraph Node Classification With Graph Neural Networks**
<br/>
<button class="copy-to-clipboard" title="Hypergraph Node Classification With Graph Neural Networks" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, eess-SP, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05569v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05569v1.pdf" filename="2402.05569v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Hypergraphs, with hyperedges connecting more than two nodes, are key for modelling higher-order interactions in real-world data. The success of graph neural networks (GNNs) reveals the capability of neural networks to process data with pairwise interactions. This inspires the usage of neural networks for data with higher-order interactions, thereby leading to the development of hypergraph neural networks (HyperGNNs). GNNs and HyperGNNs are typically considered distinct since they are designed for data on different geometric topologies. However, in this paper, we theoretically demonstrate that, in the context of node classification, most HyperGNNs can be approximated using a GNN with a weighted clique expansion of the hypergraph. This leads to WCE-GNN, a simple and efficient framework comprising a GNN and a weighted clique expansion (WCE), for hypergraph node classification. Experiments on nine real-world hypergraph node classification benchmarks showcase that WCE-GNN demonstrates not only higher classification accuracy compared to state-of-the-art HyperGNNs, but also superior memory and runtime efficiency.

{{</citation>}}


### (41/125) Offline Actor-Critic Reinforcement Learning Scales to Large Models (Jost Tobias Springenberg et al., 2024)

{{<citation>}}

Jost Tobias Springenberg, Abbas Abdolmaleki, Jingwei Zhang, Oliver Groth, Michael Bloesch, Thomas Lampe, Philemon Brakel, Sarah Bechtle, Steven Kapturowski, Roland Hafner, Nicolas Heess, Martin Riedmiller. (2024)  
**Offline Actor-Critic Reinforcement Learning Scales to Large Models**
<br/>
<button class="copy-to-clipboard" title="Offline Actor-Critic Reinforcement Learning Scales to Large Models" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05546v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05546v1.pdf" filename="2402.05546v1.pdf">Download PDF</button>

---


**ABSTRACT**  
We show that offline actor-critic reinforcement learning can scale to large models - such as transformers - and follows similar scaling laws as supervised learning. We find that offline actor-critic algorithms can outperform strong, supervised, behavioral cloning baselines for multi-task training on a large dataset containing both sub-optimal and expert behavior on 132 continuous control tasks. We introduce a Perceiver-based actor-critic model and elucidate the key model features needed to make offline RL work with self- and cross-attention modules. Overall, we find that: i) simple offline actor critic algorithms are a natural choice for gradually moving away from the currently predominant paradigm of behavioral cloning, and ii) via offline RL it is possible to learn multi-task policies that master many domains simultaneously, including real robotics tasks, from sub-optimal demonstrations or self-generated data.

{{</citation>}}


### (42/125) Reinforcement Learning as a Catalyst for Robust and Fair Federated Learning: Deciphering the Dynamics of Client Contributions (Jialuo He et al., 2024)

{{<citation>}}

Jialuo He, Wei Chen, Xiaojin Zhang. (2024)  
**Reinforcement Learning as a Catalyst for Robust and Fair Federated Learning: Deciphering the Dynamics of Client Contributions**
<br/>
<button class="copy-to-clipboard" title="Reinforcement Learning as a Catalyst for Robust and Fair Federated Learning: Deciphering the Dynamics of Client Contributions" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05541v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05541v1.pdf" filename="2402.05541v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Recent advancements in federated learning (FL) have produced models that retain user privacy by training across multiple decentralized devices or systems holding local data samples. However, these strategies often neglect the inherent challenges of statistical heterogeneity and vulnerability to adversarial attacks, which can degrade model robustness and fairness. Personalized FL strategies offer some respite by adjusting models to fit individual client profiles, yet they tend to neglect server-side aggregation vulnerabilities. To address these issues, we propose Reinforcement Federated Learning (RFL), a novel framework that leverages deep reinforcement learning to adaptively optimize client contribution during aggregation, thereby enhancing both model robustness against malicious clients and fairness across participants under non-identically distributed settings. To achieve this goal, we propose a meticulous approach involving a Deep Deterministic Policy Gradient-based algorithm for continuous control of aggregation weights, an innovative client selection method based on model parameter distances, and a reward mechanism guided by validation set performance. Empirically, extensive experiments demonstrate that, in terms of robustness, RFL outperforms the state-of-the-art methods, while maintaining comparable levels of fairness, offering a promising solution to build resilient and fair federated systems.

{{</citation>}}


### (43/125) Empowering machine learning models with contextual knowledge for enhancing the detection of eating disorders in social media posts (José Alberto Benítez-Andrades et al., 2024)

{{<citation>}}

José Alberto Benítez-Andrades, María Teresa García-Ordás, Mayra Russo, Ahmad Sakor, Luis Daniel Fernandes Rotger, Maria-Esther Vidal. (2024)  
**Empowering machine learning models with contextual knowledge for enhancing the detection of eating disorders in social media posts**
<br/>
<button class="copy-to-clipboard" title="Empowering machine learning models with contextual knowledge for enhancing the detection of eating disorders in social media posts" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: AI, BERT, Falcon  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05536v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05536v1.pdf" filename="2402.05536v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Social networks are vital for information sharing, especially in the health sector for discussing diseases and treatments. These platforms, however, often feature posts as brief texts, posing challenges for Artificial Intelligence (AI) in understanding context. We introduce a novel hybrid approach combining community-maintained knowledge graphs (like Wikidata) with deep learning to enhance the categorization of social media posts. This method uses advanced entity recognizers and linkers (like Falcon 2.0) to connect short post entities to knowledge graphs. Knowledge graph embeddings (KGEs) and contextualized word embeddings (like BERT) are then employed to create rich, context-based representations of these posts.   Our focus is on the health domain, particularly in identifying posts related to eating disorders (e.g., anorexia, bulimia) to aid healthcare providers in early diagnosis. We tested our approach on a dataset of 2,000 tweets about eating disorders, finding that merging word embeddings with knowledge graph information enhances the predictive models' reliability. This methodology aims to assist health experts in spotting patterns indicative of mental disorders, thereby improving early detection and accurate diagnosis for personalized medicine.

{{</citation>}}


### (44/125) Differentially Private Model-Based Offline Reinforcement Learning (Alexandre Rio et al., 2024)

{{<citation>}}

Alexandre Rio, Merwan Barlier, Igor Colin, Albert Thomas. (2024)  
**Differentially Private Model-Based Offline Reinforcement Learning**
<br/>
<button class="copy-to-clipboard" title="Differentially Private Model-Based Offline Reinforcement Learning" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05525v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05525v1.pdf" filename="2402.05525v1.pdf">Download PDF</button>

---


**ABSTRACT**  
We address offline reinforcement learning with privacy guarantees, where the goal is to train a policy that is differentially private with respect to individual trajectories in the dataset. To achieve this, we introduce DP-MORL, an MBRL algorithm coming with differential privacy guarantees. A private model of the environment is first learned from offline data using DP-FedAvg, a training method for neural networks that provides differential privacy guarantees at the trajectory level. Then, we use model-based policy optimization to derive a policy from the (penalized) private model, without any further interaction with the system or access to the input data. We empirically show that DP-MORL enables the training of private RL agents from offline data and we furthermore outline the price of privacy in this setting.

{{</citation>}}


### (45/125) Linearizing Models for Efficient yet Robust Private Inference (Sreetama Sarkar et al., 2024)

{{<citation>}}

Sreetama Sarkar, Souvik Kundu, Peter A. Beerel. (2024)  
**Linearizing Models for Efficient yet Robust Private Inference**
<br/>
<button class="copy-to-clipboard" title="Linearizing Models for Efficient yet Robust Private Inference" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: ImageNet  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05521v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05521v1.pdf" filename="2402.05521v1.pdf">Download PDF</button>

---


**ABSTRACT**  
The growing concern about data privacy has led to the development of private inference (PI) frameworks in client-server applications which protects both data privacy and model IP. However, the cryptographic primitives required yield significant latency overhead which limits its wide-spread application. At the same time, changing environments demand the PI service to be robust against various naturally occurring and gradient-based perturbations. Despite several works focused on the development of latency-efficient models suitable for PI, the impact of these models on robustness has remained unexplored. Towards this goal, this paper presents RLNet, a class of robust linearized networks that can yield latency improvement via reduction of high-latency ReLU operations while improving the model performance on both clean and corrupted images. In particular, RLNet models provide a "triple win ticket" of improved classification accuracy on clean, naturally perturbed, and gradient-based perturbed images using a shared-mask shared-weight architecture with over an order of magnitude fewer ReLUs than baseline models. To demonstrate the efficacy of RLNet, we perform extensive experiments with ResNet and WRN model variants on CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets. Our experimental evaluations show that RLNet can yield models with up to 11.14x fewer ReLUs, with accuracy close to the all-ReLU models, on clean, naturally perturbed, and gradient-based perturbed images. Compared with the SoTA non-robust linearized models at similar ReLU budgets, RLNet achieves an improvement in adversarial accuracy of up to ~47%, naturally perturbed accuracy up to ~16.4%, while improving clean image accuracy up to ~1.5%.

{{</citation>}}


### (46/125) Accurate LoRA-Finetuning Quantization of LLMs via Information Retention (Haotong Qin et al., 2024)

{{<citation>}}

Haotong Qin, Xudong Ma, Xingyu Zheng, Xiaoyang Li, Yang Zhang, Shouda Liu, Jie Luo, Xianglong Liu, Michele Magno. (2024)  
**Accurate LoRA-Finetuning Quantization of LLMs via Information Retention**
<br/>
<button class="copy-to-clipboard" title="Accurate LoRA-Finetuning Quantization of LLMs via Information Retention" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: LLaMA, Quantization  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05445v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05445v1.pdf" filename="2402.05445v1.pdf">Download PDF</button>

---


**ABSTRACT**  
The LoRA-finetuning quantization of LLMs has been extensively studied to obtain accurate yet compact LLMs for deployment on resource-constrained hardware. However, existing methods cause the quantized LLM to severely degrade and even fail to benefit from the finetuning of LoRA. This paper proposes a novel IR-QLoRA for pushing quantized LLMs with LoRA to be highly accurate through information retention. The proposed IR-QLoRA mainly relies on two technologies derived from the perspective of unified information: (1) statistics-based Information Calibration Quantization allows the quantized parameters of LLM to retain original information accurately; (2) finetuning-based Information Elastic Connection makes LoRA utilizes elastic representation transformation with diverse information. Comprehensive experiments show that IR-QLoRA can significantly improve accuracy across LLaMA and LLaMA2 families under 2-4 bit-widths, e.g., 4- bit LLaMA-7B achieves 1.4% improvement on MMLU compared with the state-of-the-art methods. The significant performance gain requires only a tiny 0.31% additional time consumption, revealing the satisfactory efficiency of our IRQLoRA. We highlight that IR-QLoRA enjoys excellent versatility, compatible with various frameworks (e.g., NormalFloat and Integer quantization) and brings general accuracy gains. The code is available at https://github.com/htqin/ir-qlora.

{{</citation>}}


### (47/125) Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes (Lucio Dery et al., 2024)

{{<citation>}}

Lucio Dery, Steven Kolawole, Jean-Francois Kagey, Virginia Smith, Graham Neubig, Ameet Talwalkar. (2024)  
**Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes**
<br/>
<button class="copy-to-clipboard" title="Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Pruning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05406v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05406v1.pdf" filename="2402.05406v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Given the generational gap in available hardware between lay practitioners and the most endowed institutions, LLMs are becoming increasingly inaccessible as they grow in size. Whilst many approaches have been proposed to compress LLMs to make their resource consumption manageable, these methods themselves tend to be resource intensive, putting them out of the reach of the very user groups they target. In this work, we explore the problem of structured pruning of LLMs using only forward passes. We seek to empower practitioners to prune models so large that their available hardware has just enough memory to run inference. We develop Bonsai, a gradient-free, perturbative pruning method capable of delivering small, fast, and accurate pruned models.   We observe that Bonsai outputs pruned models that (i) outperform those generated by more expensive gradient-based structured pruning methods, and (ii) are twice as fast (with comparable accuracy) as those generated by semi-structured pruning methods requiring comparable resources as Bonsai. We also leverage Bonsai to produce a new sub-2B model using a single A6000 that yields state-of-the-art performance on 4/6 tasks on the Huggingface Open LLM leaderboard.

{{</citation>}}


### (48/125) TASER: Temporal Adaptive Sampling for Fast and Accurate Dynamic Graph Representation Learning (Gangda Deng et al., 2024)

{{<citation>}}

Gangda Deng, Hongkuan Zhou, Hanqing Zeng, Yinglong Xia, Christopher Leung, Jianbo Li, Rajgopal Kannan, Viktor Prasanna. (2024)  
**TASER: Temporal Adaptive Sampling for Fast and Accurate Dynamic Graph Representation Learning**
<br/>
<button class="copy-to-clipboard" title="TASER: Temporal Adaptive Sampling for Fast and Accurate Dynamic Graph Representation Learning" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Representation Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05396v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05396v1.pdf" filename="2402.05396v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Recently, Temporal Graph Neural Networks (TGNNs) have demonstrated state-of-the-art performance in various high-impact applications, including fraud detection and content recommendation. Despite the success of TGNNs, they are prone to the prevalent noise found in real-world dynamic graphs like time-deprecated links and skewed interaction distribution. The noise causes two critical issues that significantly compromise the accuracy of TGNNs: (1) models are supervised by inferior interactions, and (2) noisy input induces high variance in the aggregated messages. However, current TGNN denoising techniques do not consider the diverse and dynamic noise pattern of each node. In addition, they also suffer from the excessive mini-batch generation overheads caused by traversing more neighbors. We believe the remedy for fast and accurate TGNNs lies in temporal adaptive sampling. In this work, we propose TASER, the first adaptive sampling method for TGNNs optimized for accuracy, efficiency, and scalability. TASER adapts its mini-batch selection based on training dynamics and temporal neighbor selection based on the contextual, structural, and temporal properties of past interactions. To alleviate the bottleneck in mini-batch generation, TASER implements a pure GPU-based temporal neighbor finder and a dedicated GPU feature cache. We evaluate the performance of TASER using two state-of-the-art backbone TGNNs. On five popular datasets, TASER outperforms the corresponding baselines by an average of 2.3% in Mean Reciprocal Rank (MRR) while achieving an average of 5.1x speedup in training time.

{{</citation>}}


### (49/125) Attention as Robust Representation for Time Series Forecasting (PeiSong Niu et al., 2024)

{{<citation>}}

PeiSong Niu, Tian Zhou, Xue Wang, Liang Sun, Rong Jin. (2024)  
**Attention as Robust Representation for Time Series Forecasting**
<br/>
<button class="copy-to-clipboard" title="Attention as Robust Representation for Time Series Forecasting" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, NLP, Time Series, Transformer, Transformers  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05370v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05370v1.pdf" filename="2402.05370v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Time series forecasting is essential for many practical applications, with the adoption of transformer-based models on the rise due to their impressive performance in NLP and CV. Transformers' key feature, the attention mechanism, dynamically fusing embeddings to enhance data representation, often relegating attention weights to a byproduct role. Yet, time series data, characterized by noise and non-stationarity, poses significant forecasting challenges. Our approach elevates attention weights as the primary representation for time series, capitalizing on the temporal relationships among data points to improve forecasting accuracy. Our study shows that an attention map, structured using global landmarks and local windows, acts as a robust kernel representation for data points, withstanding noise and shifts in distribution. Our method outperforms state-of-the-art models, reducing mean squared error (MSE) in multivariate time series forecasting by a notable 3.6% without altering the core neural network architecture. It serves as a versatile component that can readily replace recent patching based embedding schemes in transformer-based models, boosting their performance.

{{</citation>}}


### (50/125) Noise Contrastive Alignment of Language Models with Explicit Rewards (Huayu Chen et al., 2024)

{{<citation>}}

Huayu Chen, Guande He, Hang Su, Jun Zhu. (2024)  
**Noise Contrastive Alignment of Language Models with Explicit Rewards**
<br/>
<button class="copy-to-clipboard" title="Noise Contrastive Alignment of Language Models with Explicit Rewards" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: GPT, GPT-4, Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05369v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05369v1.pdf" filename="2402.05369v1.pdf">Download PDF</button>

---


**ABSTRACT**  
User intentions are typically formalized as evaluation rewards to be maximized when fine-tuning language models (LMs). Existing alignment methods, such as Direct Preference Optimization (DPO), are mainly tailored for pairwise preference data where rewards are implicitly defined rather than explicitly given. In this paper, we introduce a general framework for LM alignment, leveraging Noise Contrastive Estimation (NCE) to bridge the gap in handling reward datasets explicitly annotated with scalar evaluations. Our framework comprises two parallel algorithms, NCA and InfoNCA, both enabling the direct extraction of an LM policy from reward data as well as preference data. Notably, we show that the DPO loss is a special case of our proposed InfoNCA objective under pairwise preference settings, thereby integrating and extending current alignment theories. By contrasting NCA and InfoNCA, we show that InfoNCA and DPO adjust relative likelihood across different responses to a single instruction, while NCA optimizes absolute likelihood for each response. We apply our methods to align a 7B language model with a GPT-4 annotated reward dataset. Experimental results suggest that InfoNCA surpasses the DPO baseline in GPT-4 evaluations, while NCA enjoys better training stability with competitive performance.

{{</citation>}}


### (51/125) Exploring Learning Complexity for Downstream Data Pruning (Wenyu Jiang et al., 2024)

{{<citation>}}

Wenyu Jiang, Zhenlong Liu, Zejian Xie, Songxin Zhang, Bingyi Jing, Hongxin Wei. (2024)  
**Exploring Learning Complexity for Downstream Data Pruning**
<br/>
<button class="copy-to-clipboard" title="Exploring Learning Complexity for Downstream Data Pruning" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Pruning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05356v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05356v1.pdf" filename="2402.05356v1.pdf">Download PDF</button>

---


**ABSTRACT**  
The over-parameterized pre-trained models pose a great challenge to fine-tuning with limited computation resources. An intuitive solution is to prune the less informative samples from the fine-tuning dataset. A series of training-based scoring functions are proposed to quantify the informativeness of the data subset but the pruning cost becomes non-negligible due to the heavy parameter updating. For efficient pruning, it is viable to adapt the similarity scoring function of geometric-based methods from training-based to training-free. However, we empirically show that such adaption distorts the original pruning and results in inferior performance on the downstream tasks. In this paper, we propose to treat the learning complexity (LC) as the scoring function for classification and regression tasks. Specifically, the learning complexity is defined as the average predicted confidence of subnets with different capacities, which encapsulates data processing within a converged model. Then we preserve the diverse and easy samples for fine-tuning. Extensive experiments with vision datasets demonstrate the effectiveness and efficiency of the proposed scoring function for classification tasks. For the instruction fine-tuning of large language models, our method achieves state-of-the-art performance with stable convergence, outperforming the full training with only 10\% of the instruction dataset.

{{</citation>}}


## cs.RO (2)



### (52/125) Driving Everywhere with Large Language Model Policy Adaptation (Boyi Li et al., 2024)

{{<citation>}}

Boyi Li, Yue Wang, Jiageng Mao, Boris Ivanovic, Sushant Veer, Karen Leung, Marco Pavone. (2024)  
**Driving Everywhere with Large Language Model Policy Adaptation**
<br/>
<button class="copy-to-clipboard" title="Driving Everywhere with Large Language Model Policy Adaptation" index=2>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-2 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CL, cs-RO, cs.RO  
Keywords: Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05932v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05932v1.pdf" filename="2402.05932v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Adapting driving behavior to new environments, customs, and laws is a long-standing problem in autonomous driving, precluding the widespread deployment of autonomous vehicles (AVs). In this paper, we present LLaDA, a simple yet powerful tool that enables human drivers and autonomous vehicles alike to drive everywhere by adapting their tasks and motion plans to traffic rules in new locations. LLaDA achieves this by leveraging the impressive zero-shot generalizability of large language models (LLMs) in interpreting the traffic rules in the local driver handbook. Through an extensive user study, we show that LLaDA's instructions are useful in disambiguating in-the-wild unexpected situations. We also demonstrate LLaDA's ability to adapt AV motion planning policies in real-world datasets; LLaDA outperforms baseline planning approaches on all our metrics. Please check our website for more details: https://boyiliee.github.io/llada.

{{</citation>}}


### (53/125) Real-World Robot Applications of Foundation Models: A Review (Kento Kawaharazuka et al., 2024)

{{<citation>}}

Kento Kawaharazuka, Tatsuya Matsushima, Andrew Gambardella, Jiaxian Guo, Chris Paxton, Andy Zeng. (2024)  
**Real-World Robot Applications of Foundation Models: A Review**
<br/>
<button class="copy-to-clipboard" title="Real-World Robot Applications of Foundation Models: A Review" index=2>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-2 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05741v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05741v1.pdf" filename="2402.05741v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Recent developments in foundation models, like Large Language Models (LLMs) and Vision-Language Models (VLMs), trained on extensive data, facilitate flexible application across different tasks and modalities. Their impact spans various fields, including healthcare, education, and robotics. This paper provides an overview of the practical application of foundation models in real-world robotics, with a primary emphasis on the replacement of specific components within existing robot systems. The summary encompasses the perspective of input-output relationships in foundation models, as well as their role in perception, motion planning, and control within the field of robotics. This paper concludes with a discussion of future challenges and implications for practical robot applications.

{{</citation>}}


## cs.AI (10)



### (54/125) An Interactive Agent Foundation Model (Zane Durante et al., 2024)

{{<citation>}}

Zane Durante, Bidipta Sarkar, Ran Gong, Rohan Taori, Yusuke Noda, Paul Tang, Ehsan Adeli, Shrinidhi Kowshika Lakshmikanth, Kevin Schulman, Arnold Milstein, Demetri Terzopoulos, Ade Famoti, Noboru Kuno, Ashley Llorens, Hoi Vo, Katsu Ikeuchi, Li Fei-Fei, Jianfeng Gao, Naoki Wake, Qiuyuan Huang. (2024)  
**An Interactive Agent Foundation Model**
<br/>
<button class="copy-to-clipboard" title="An Interactive Agent Foundation Model" index=3>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-3 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-RO, cs.AI  
Keywords: AI  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05929v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05929v1.pdf" filename="2402.05929v1.pdf">Download PDF</button>

---


**ABSTRACT**  
The development of artificial intelligence systems is transitioning from creating static, task-specific models to dynamic, agent-based systems capable of performing well in a wide range of applications. We propose an Interactive Agent Foundation Model that uses a novel multi-task agent training paradigm for training AI agents across a wide range of domains, datasets, and tasks. Our training paradigm unifies diverse pre-training strategies, including visual masked auto-encoders, language modeling, and next-action prediction, enabling a versatile and adaptable AI framework. We demonstrate the performance of our framework across three separate domains -- Robotics, Gaming AI, and Healthcare. Our model demonstrates its ability to generate meaningful and contextually relevant outputs in each area. The strength of our approach lies in its generality, leveraging a variety of data sources such as robotics sequences, gameplay data, large-scale video datasets, and textual information for effective multimodal and multi-task learning. Our approach provides a promising avenue for developing generalist, action-taking, multimodal systems.

{{</citation>}}


### (55/125) Large Language Model Meets Graph Neural Network in Knowledge Distillation (Shengxiang Hu et al., 2024)

{{<citation>}}

Shengxiang Hu, Guobing Zou, Song Yang, Bofeng Zhang, Yixin Chen. (2024)  
**Large Language Model Meets Graph Neural Network in Knowledge Distillation**
<br/>
<button class="copy-to-clipboard" title="Large Language Model Meets Graph Neural Network in Knowledge Distillation" index=3>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-3 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.AI  
Categories: 68T30, 68R10, 68T05, cs-AI, cs-LG, cs.AI  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Knowledge Distillation, Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05894v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05894v1.pdf" filename="2402.05894v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Despite recent community revelations about the advancements and potential of Large Language Models (LLMs) in understanding Text-Attributed Graphs (TAG), the deployment of LLMs for production is hindered by their high computational and storage requirements, as well as long latencies during inference. Simultaneously, although traditional Graph Neural Networks (GNNs) are light weight and adept at learning structural features of graphs, their ability to grasp the complex semantics in TAGs is somewhat constrained for real applications. To address these limitations, we concentrate on the downstream task of node classification in TAG and propose a novel graph knowledge distillation framework, termed Linguistic Graph Knowledge Distillation (LinguGKD), using LLMs as teacher models and GNNs as student models for knowledge distillation. It involves TAG-oriented instruction tuning of LLM on designed node classification prompts, followed by aligning the hierarchically learned node features of the teacher LLM and the student GNN in latent space, employing a layer-adaptive contrastive learning strategy. Through extensive experiments on a variety of LLM and GNN models and multiple benchmark datasets, the proposed LinguGKD significantly boosts the student GNN's predictive accuracy and convergence rate, without the need of extra data or model parameters. Compared to teacher LLM, distilled GNN achieves superior inference speed equipped with much fewer computing and storage demands, when surpassing the teacher LLM's classification performance on some of benchmark datasets.

{{</citation>}}


### (56/125) How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis (Federico Bianchi et al., 2024)

{{<citation>}}

Federico Bianchi, Patrick John Chia, Mert Yuksekgonul, Jacopo Tagliabue, Dan Jurafsky, James Zou. (2024)  
**How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis**
<br/>
<button class="copy-to-clipboard" title="How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis" index=3>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-3 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: GPT, GPT-4  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05863v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05863v1.pdf" filename="2402.05863v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Negotiation is the basis of social interactions; humans negotiate everything from the price of cars to how to share common resources. With rapidly growing interest in using large language models (LLMs) to act as agents on behalf of human users, such LLM agents would also need to be able to negotiate. In this paper, we study how well LLMs can negotiate with each other. We develop NegotiationArena: a flexible framework for evaluating and probing the negotiation abilities of LLM agents. We implemented three types of scenarios in NegotiationArena to assess LLM's behaviors in allocating shared resources (ultimatum games), aggregate resources (trading games) and buy/sell goods (price negotiations). Each scenario allows for multiple turns of flexible dialogues between LLM agents to allow for more complex negotiations. Interestingly, LLM agents can significantly boost their negotiation outcomes by employing certain behavioral tactics. For example, by pretending to be desolate and desperate, LLMs can improve their payoffs by 20\% when negotiating against the standard GPT-4. We also quantify irrational negotiation behaviors exhibited by the LLM agents, many of which also appear in humans. Together, \NegotiationArena offers a new environment to investigate LLM interactions, enabling new insights into LLM's theory of mind, irrationality, and reasoning abilities.

{{</citation>}}


### (57/125) Limitations of Agents Simulated by Predictive Models (Raymond Douglas et al., 2024)

{{<citation>}}

Raymond Douglas, Jacek Karwowski, Chan Bae, Andis Draguns, Victoria Krakovna. (2024)  
**Limitations of Agents Simulated by Predictive Models**
<br/>
<button class="copy-to-clipboard" title="Limitations of Agents Simulated by Predictive Models" index=3>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-3 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Transformer, Transformers  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05829v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05829v1.pdf" filename="2402.05829v1.pdf">Download PDF</button>

---


**ABSTRACT**  
There is increasing focus on adapting predictive models into agent-like systems, most notably AI assistants based on language models. We outline two structural reasons for why these models can fail when turned into agents. First, we discuss auto-suggestive delusions. Prior work has shown theoretically that models fail to imitate agents that generated the training data if the agents relied on hidden observations: the hidden observations act as confounding variables, and the models treat actions they generate as evidence for nonexistent observations. Second, we introduce and formally study a related, novel limitation: predictor-policy incoherence. When a model generates a sequence of actions, the model's implicit prediction of the policy that generated those actions can serve as a confounding variable. The result is that models choose actions as if they expect future actions to be suboptimal, causing them to be overly conservative. We show that both of those failures are fixed by including a feedback loop from the environment, that is, re-training the models on their own actions. We give simple demonstrations of both limitations using Decision Transformers and confirm that empirical results agree with our conceptual and formal analysis. Our treatment provides a unifying view of those failure modes, and informs the question of why fine-tuning offline learned policies with online learning makes them more effective.

{{</citation>}}


### (58/125) Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning (Zhiheng Xi et al., 2024)

{{<citation>}}

Zhiheng Xi, Wenxiang Chen, Boyang Hong, Senjie Jin, Rui Zheng, Wei He, Yiwen Ding, Shichun Liu, Xin Guo, Junzhe Wang, Honglin Guo, Wei Shen, Xiaoran Fan, Yuhao Zhou, Shihan Dou, Xiao Wang, Xinbo Zhang, Peng Sun, Tao Gui, Qi Zhang, Xuanjing Huang. (2024)  
**Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning**
<br/>
<button class="copy-to-clipboard" title="Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning" index=3>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-3 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Language Model, Reasoning, Reinforcement Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05808v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05808v1.pdf" filename="2402.05808v1.pdf">Download PDF</button>

---


**ABSTRACT**  
In this paper, we propose R$^3$: Learning Reasoning through Reverse Curriculum Reinforcement Learning (RL), a novel method that employs only outcome supervision to achieve the benefits of process supervision for large language models. The core challenge in applying RL to complex reasoning is to identify a sequence of actions that result in positive rewards and provide appropriate supervision for optimization. Outcome supervision provides sparse rewards for final results without identifying error locations, whereas process supervision offers step-wise rewards but requires extensive manual annotation. R$^3$ overcomes these limitations by learning from correct demonstrations. Specifically, R$^3$ progressively slides the start state of reasoning from a demonstration's end to its beginning, facilitating easier model exploration at all stages. Thus, R$^3$ establishes a step-wise curriculum, allowing outcome supervision to offer step-level signals and precisely pinpoint errors. Using Llama2-7B, our method surpasses RL baseline on eight reasoning tasks by $4.1$ points on average. Notebaly, in program-based reasoning on GSM8K, it exceeds the baseline by $4.2$ points across three backbone models, and without any extra data, Codellama-7B + R$^3$ performs comparable to larger models or closed-source models.

{{</citation>}}


### (59/125) Prompting Fairness: Artificial Intelligence as Game Players (Jazmia Henry, 2024)

{{<citation>}}

Jazmia Henry. (2024)  
**Prompting Fairness: Artificial Intelligence as Game Players**
<br/>
<button class="copy-to-clipboard" title="Prompting Fairness: Artificial Intelligence as Game Players" index=3>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-3 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.AI  
Categories: cs-AI, cs-GT, cs.AI  
Keywords: AI  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05786v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05786v1.pdf" filename="2402.05786v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Utilitarian games such as dictator games to measure fairness have been studied in the social sciences for decades. These games have given us insight into not only how humans view fairness but also in what conditions the frequency of fairness, altruism and greed increase or decrease. While these games have traditionally been focused on humans, the rise of AI gives us the ability to study how these models play these games. AI is becoming a constant in human interaction and examining how these models portray fairness in game play can give us some insight into how AI makes decisions. Over 101 rounds of the dictator game, I conclude that AI has a strong sense of fairness that is dependant of it it deems the person it is playing with as trustworthy, framing has a strong effect on how much AI gives a recipient when designated the trustee, and there may be evidence that AI experiences inequality aversion just as humans.

{{</citation>}}


### (60/125) Optimizing Delegation in Collaborative Human-AI Hybrid Teams (Andrew Fuchs et al., 2024)

{{<citation>}}

Andrew Fuchs, Andrea Passarella, Marco Conti. (2024)  
**Optimizing Delegation in Collaborative Human-AI Hybrid Teams**
<br/>
<button class="copy-to-clipboard" title="Optimizing Delegation in Collaborative Human-AI Hybrid Teams" index=3>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-3 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-LG, cs.AI  
Keywords: AI, Reinforcement Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05605v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05605v1.pdf" filename="2402.05605v1.pdf">Download PDF</button>

---


**ABSTRACT**  
When humans and autonomous systems operate together as what we refer to as a hybrid team, we of course wish to ensure the team operates successfully and effectively. We refer to team members as agents. In our proposed framework, we address the case of hybrid teams in which, at any time, only one team member (the control agent) is authorized to act as control for the team. To determine the best selection of a control agent, we propose the addition of an AI manager (via Reinforcement Learning) which learns as an outside observer of the team. The manager learns a model of behavior linking observations of agent performance and the environment/world the team is operating in, and from these observations makes the most desirable selection of a control agent. We restrict the manager task by introducing a set of constraints. The manager constraints indicate acceptable team operation, so a violation occurs if the team enters a condition which is unacceptable and requires manager intervention. To ensure minimal added complexity or potential inefficiency for the team, the manager should attempt to minimize the number of times the team reaches a constraint violation and requires subsequent manager intervention. Therefore our manager is optimizing its selection of authorized agents to boost overall team performance while minimizing the frequency of manager intervention. We demonstrate our manager performance in a simulated driving scenario representing the case of a hybrid team of agents composed of a human driver and autonomous driving system. We perform experiments for our driving scenario with interfering vehicles, indicating the need for collision avoidance and proper speed control. Our results indicate a positive impact of our manager, with some cases resulting in increased team performance up to ~187% that of the best solo agent performance.

{{</citation>}}


### (61/125) Rapid Optimization for Jailbreaking LLMs via Subconscious Exploitation and Echopraxia (Guangyu Shen et al., 2024)

{{<citation>}}

Guangyu Shen, Siyuan Cheng, Kaiyuan Zhang, Guanhong Tao, Shengwei An, Lu Yan, Zhuo Zhang, Shiqing Ma, Xiangyu Zhang. (2024)  
**Rapid Optimization for Jailbreaking LLMs via Subconscious Exploitation and Echopraxia**
<br/>
<button class="copy-to-clipboard" title="Rapid Optimization for Jailbreaking LLMs via Subconscious Exploitation and Echopraxia" index=3>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-3 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CR, cs.AI  
Keywords: Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05467v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05467v1.pdf" filename="2402.05467v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Large Language Models (LLMs) have become prevalent across diverse sectors, transforming human life with their extraordinary reasoning and comprehension abilities. As they find increased use in sensitive tasks, safety concerns have gained widespread attention. Extensive efforts have been dedicated to aligning LLMs with human moral principles to ensure their safe deployment. Despite their potential, recent research indicates aligned LLMs are prone to specialized jailbreaking prompts that bypass safety measures to elicit violent and harmful content. The intrinsic discrete nature and substantial scale of contemporary LLMs pose significant challenges in automatically generating diverse, efficient, and potent jailbreaking prompts, representing a continuous obstacle. In this paper, we introduce RIPPLE (Rapid Optimization via Subconscious Exploitation and Echopraxia), a novel optimization-based method inspired by two psychological concepts: subconsciousness and echopraxia, which describe the processes of the mind that occur without conscious awareness and the involuntary mimicry of actions, respectively. Evaluations across 6 open-source LLMs and 4 commercial LLM APIs show RIPPLE achieves an average Attack Success Rate of 91.5\%, outperforming five current methods by up to 47.0\% with an 8x reduction in overhead. Furthermore, it displays significant transferability and stealth, successfully evading established detection mechanisms. The code of our work is available at \url{https://github.com/SolidShen/RIPPLE_official/tree/official}

{{</citation>}}


### (62/125) Knowledge Graphs Meet Multi-Modal Learning: A Comprehensive Survey (Zhuo Chen et al., 2024)

{{<citation>}}

Zhuo Chen, Yichi Zhang, Yin Fang, Yuxia Geng, Lingbing Guo, Xiang Chen, Qian Li, Wen Zhang, Jiaoyan Chen, Yushan Zhu, Jiaqi Li, Xiaoze Liu, Jeff Z. Pan, Ningyu Zhang, Huajun Chen. (2024)  
**Knowledge Graphs Meet Multi-Modal Learning: A Comprehensive Survey**
<br/>
<button class="copy-to-clipboard" title="Knowledge Graphs Meet Multi-Modal Learning: A Comprehensive Survey" index=3>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-3 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs-IR, cs-LG, cs.AI  
Keywords: AI, Entity Alignment, Image Classification, Knowledge Graph, Language Model, Question Answering  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05391v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05391v1.pdf" filename="2402.05391v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Knowledge Graphs (KGs) play a pivotal role in advancing various AI applications, with the semantic web community's exploration into multi-modal dimensions unlocking new avenues for innovation. In this survey, we carefully review over 300 articles, focusing on KG-aware research in two principal aspects: KG-driven Multi-Modal (KG4MM) learning, where KGs support multi-modal tasks, and Multi-Modal Knowledge Graph (MM4KG), which extends KG studies into the MMKG realm. We begin by defining KGs and MMKGs, then explore their construction progress. Our review includes two primary task categories: KG-aware multi-modal learning tasks, such as Image Classification and Visual Question Answering, and intrinsic MMKG tasks like Multi-modal Knowledge Graph Completion and Entity Alignment, highlighting specific research trajectories. For most of these tasks, we provide definitions, evaluation benchmarks, and additionally outline essential insights for conducting relevant research. Finally, we discuss current challenges and identify emerging trends, such as progress in Large Language Modeling and Multi-modal Pre-training strategies. This survey aims to serve as a comprehensive reference for researchers already involved in or considering delving into KG and multi-modal learning research, offering insights into the evolving landscape of MMKG research and supporting future work.

{{</citation>}}


### (63/125) Guiding Large Language Models with Divide-and-Conquer Program for Discerning Problem Solving (Yizhou Zhang et al., 2024)

{{<citation>}}

Yizhou Zhang, Lun Du, Defu Cao, Qiang Fu, Yan Liu. (2024)  
**Guiding Large Language Models with Divide-and-Conquer Program for Discerning Problem Solving**
<br/>
<button class="copy-to-clipboard" title="Guiding Large Language Models with Divide-and-Conquer Program for Discerning Problem Solving" index=3>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-3 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Language Model, Transformer  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05359v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05359v1.pdf" filename="2402.05359v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Foundation models, such as Large language Models (LLMs), have attracted significant amount of interest due to their large number of applications. Existing works show that appropriate prompt design, such as Chain-of-Thoughts, can unlock LLM's powerful capacity in diverse areas. However, when handling tasks involving repetitive sub-tasks and/or deceptive contents, such as arithmetic calculation and article-level fake news detection, existing prompting strategies either suffers from insufficient expressive power or intermediate errors triggered by hallucination. To make LLM more discerning to such intermediate errors, we propose to guide LLM with a Divide-and-Conquer program that simultaneously ensures superior expressive power and disentangles task decomposition, sub-task resolution, and resolution assembly process. Theoretic analysis reveals that our strategy can guide LLM to extend the expressive power of fixed-depth Transformer. Experiments indicate that our proposed method can achieve better performance than typical prompting strategies in tasks bothered by intermediate errors and deceptive contents, such as large integer multiplication, hallucination detection and misinformation detection.

{{</citation>}}


## cs.CL (29)



### (64/125) WebLINX: Real-World Website Navigation with Multi-Turn Dialogue (Xing Han Lù et al., 2024)

{{<citation>}}

Xing Han Lù, Zdeněk Kasner, Siva Reddy. (2024)  
**WebLINX: Real-World Website Navigation with Multi-Turn Dialogue**
<br/>
<button class="copy-to-clipboard" title="WebLINX: Real-World Website Navigation with Multi-Turn Dialogue" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: Dialog, Dialogue, GPT, GPT-4, Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05930v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05930v1.pdf" filename="2402.05930v1.pdf">Download PDF</button>

---


**ABSTRACT**  
We propose the problem of conversational web navigation, where a digital agent controls a web browser and follows user instructions to solve real-world tasks in a multi-turn dialogue fashion. To support this problem, we introduce WEBLINX - a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. Our benchmark covers a broad range of patterns on over 150 real-world websites and can be used to train and evaluate agents in diverse scenarios. Due to the magnitude of information present, Large Language Models (LLMs) cannot process entire web pages in real-time. To solve this bottleneck, we design a retrieval-inspired model that efficiently prunes HTML pages by ranking relevant elements. We use the selected elements, along with screenshots and action history, to assess a variety of models for their ability to replicate human behavior when navigating the web. Our experiments span from small text-only to proprietary multimodal LLMs. We find that smaller finetuned decoders surpass the best zero-shot LLMs (including GPT-4V), but also larger finetuned multimodal models which were explicitly pretrained on screenshots. However, all finetuned models struggle to generalize to unseen websites. Our findings highlight the need for large multimodal models that can generalize to novel settings. Our code, data and models are available for research: https://mcgill-nlp.github.io/weblinx

{{</citation>}}


### (65/125) Efficient Stagewise Pretraining via Progressive Subnetworks (Abhishek Panigrahi et al., 2024)

{{<citation>}}

Abhishek Panigrahi, Nikunj Saunshi, Kaifeng Lyu, Sobhan Miryoosefi, Sashank Reddi, Satyen Kale, Sanjiv Kumar. (2024)  
**Efficient Stagewise Pretraining via Progressive Subnetworks**
<br/>
<button class="copy-to-clipboard" title="Efficient Stagewise Pretraining via Progressive Subnetworks" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, GLUE, QA, SuperGLUE  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05913v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05913v1.pdf" filename="2402.05913v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Recent developments in large language models have sparked interest in efficient pretraining methods. A recent effective paradigm is to perform stage-wise training, where the size of the model is gradually increased over the course of training (e.g. gradual stacking (Reddi et al., 2023)). While the resource and wall-time savings are appealing, it has limitations, particularly the inability to evaluate the full model during earlier stages, and degradation in model quality due to smaller model capacity in the initial stages. In this work, we propose an alternative framework, progressive subnetwork training, that maintains the full model throughout training, but only trains subnetworks within the model in each step. We focus on a simple instantiation of this framework, Random Path Training (RaPTr) that only trains a sub-path of layers in each step, progressively increasing the path lengths in stages. RaPTr achieves better pre-training loss for BERT and UL2 language models while requiring 20-33% fewer FLOPs compared to standard training, and is competitive or better than other efficient training methods. Furthermore, RaPTr shows better downstream performance on UL2, improving QA tasks and SuperGLUE by 1-5% compared to standard training and stacking. Finally, we provide a theoretical basis for RaPTr to justify (a) the increasing complexity of subnetworks in stages, and (b) the stability in loss across stage transitions due to residual connections and layer norm.

{{</citation>}}


### (66/125) FACT-GPT: Fact-Checking Augmentation via Claim Matching with LLMs (Eun Cheol Choi et al., 2024)

{{<citation>}}

Eun Cheol Choi, Emilio Ferrara. (2024)  
**FACT-GPT: Fact-Checking Augmentation via Claim Matching with LLMs**
<br/>
<button class="copy-to-clipboard" title="FACT-GPT: Fact-Checking Augmentation via Claim Matching with LLMs" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-HC, cs-SI, cs.CL  
Keywords: Augmentation, Fact-Checking, GPT, Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05904v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05904v1.pdf" filename="2402.05904v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Our society is facing rampant misinformation harming public health and trust. To address the societal challenge, we introduce FACT-GPT, a system leveraging Large Language Models (LLMs) to automate the claim matching stage of fact-checking. FACT-GPT, trained on a synthetic dataset, identifies social media content that aligns with, contradicts, or is irrelevant to previously debunked claims. Our evaluation shows that our specialized LLMs can match the accuracy of larger models in identifying related claims, closely mirroring human judgment. This research provides an automated solution for efficient claim matching, demonstrates the potential of LLMs in supporting fact-checkers, and offers valuable resources for further research in the field.

{{</citation>}}


### (67/125) PromptCrypt: Prompt Encryption for Secure Communication with Large Language Models (Guo Lin et al., 2024)

{{<citation>}}

Guo Lin, Wenyue Hua, Yongfeng Zhang. (2024)  
**PromptCrypt: Prompt Encryption for Secure Communication with Large Language Models**
<br/>
<button class="copy-to-clipboard" title="PromptCrypt: Prompt Encryption for Secure Communication with Large Language Models" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs-IR, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05868v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05868v1.pdf" filename="2402.05868v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Cloud-based large language models (LLMs) such as ChatGPT have increasingly become integral to daily operations, serving as vital tools across various applications. While these models offer substantial benefits in terms of accessibility and functionality, they also introduce significant privacy concerns: the transmission and storage of user data in cloud infrastructures pose substantial risks of data breaches and unauthorized access to sensitive information; even if the transmission and storage of data is encrypted, the LLM service provider itself still knows the real contents of the data, preventing individuals or entities from confidently using such LLM services. To address these concerns, this paper proposes a simple yet effective mechanism PromptCrypt to protect user privacy. It uses Emoji to encrypt the user inputs before sending them to LLM, effectively rendering them indecipherable to human or LLM's examination while retaining the original intent of the prompt, thus ensuring the model's performance remains unaffected. We conduct experiments on three tasks, personalized recommendation, sentiment analysis, and tabular data analysis. Experiment results reveal that PromptCrypt can encrypt personal information within prompts in such a manner that not only prevents the discernment of sensitive data by humans or LLM itself, but also maintains or even improves the precision without further tuning, achieving comparable or even better task accuracy than directly prompting the LLM without prompt encryption. These results highlight the practicality of adopting encryption measures that safeguard user privacy without compromising the functional integrity and performance of LLMs. Code and dataset are available at https://github.com/agiresearch/PromptCrypt.

{{</citation>}}


### (68/125) Is it Possible to Edit Large Language Models Robustly? (Xinbei Ma et al., 2024)

{{<citation>}}

Xinbei Ma, Tianjie Ju, Jiyang Qiu, Zhuosheng Zhang, Hai Zhao, Lifeng Liu, Yulong Wang. (2024)  
**Is it Possible to Edit Large Language Models Robustly?**
<br/>
<button class="copy-to-clipboard" title="Is it Possible to Edit Large Language Models Robustly?" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05827v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05827v1.pdf" filename="2402.05827v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Large language models (LLMs) have played a pivotal role in building communicative AI to imitate human behaviors but face the challenge of efficient customization. To tackle this challenge, recent studies have delved into the realm of model editing, which manipulates specific memories of language models and changes the related language generation. However, the robustness of model editing remains an open question. This work seeks to understand the strengths and limitations of editing methods, thus facilitating robust, realistic applications of communicative AI. Concretely, we conduct extensive analysis to address the three key research questions. Q1: Can edited LLMs behave consistently resembling communicative AI in realistic situations? Q2: To what extent does the rephrasing of prompts lead LLMs to deviate from the edited knowledge memory? Q3: Which knowledge features are correlated with the performance and robustness of editing? Our experimental results uncover a substantial disparity between existing editing methods and the practical application of LLMs. On rephrased prompts that are complex and flexible but common in realistic applications, the performance of editing experiences a significant decline. Further analysis shows that more popular knowledge is memorized better, easier to recall, and more challenging to edit effectively.

{{</citation>}}


### (69/125) Selective Forgetting: Advancing Machine Unlearning Techniques and Evaluation in Language Models (Lingzhi Wang et al., 2024)

{{<citation>}}

Lingzhi Wang, Xingshan Zeng, Jinsong Guo, Kam-Fai Wong, Georg Gottlob. (2024)  
**Selective Forgetting: Advancing Machine Unlearning Techniques and Evaluation in Language Models**
<br/>
<button class="copy-to-clipboard" title="Selective Forgetting: Advancing Machine Unlearning Techniques and Evaluation in Language Models" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Information Extraction, Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05813v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05813v1.pdf" filename="2402.05813v1.pdf">Download PDF</button>

---


**ABSTRACT**  
The aim of this study is to investigate Machine Unlearning (MU), a burgeoning field focused on addressing concerns related to neural models inadvertently retaining personal or sensitive data. Here, a novel approach is introduced to achieve precise and selective forgetting within language models. Unlike previous methodologies that adopt completely opposing training objectives, this approach aims to mitigate adverse effects on language model performance, particularly in generation tasks. Furthermore, two innovative evaluation metrics are proposed: Sensitive Information Extraction Likelihood (S-EL) and Sensitive Information Memory Accuracy (S-MA), designed to gauge the effectiveness of sensitive information elimination. To reinforce the forgetting framework, an effective method for annotating sensitive scopes is presented, involving both online and offline strategies. The online selection mechanism leverages language probability scores to ensure computational efficiency, while the offline annotation entails a robust two-stage process based on Large Language Models (LLMs).

{{</citation>}}


### (70/125) FAQ-Gen: An automated system to generate domain-specific FAQs to aid content comprehension (Sahil Kale et al., 2024)

{{<citation>}}

Sahil Kale, Gautam Khaire, Jay Patankar. (2024)  
**FAQ-Gen: An automated system to generate domain-specific FAQs to aid content comprehension**
<br/>
<button class="copy-to-clipboard" title="FAQ-Gen: An automated system to generate domain-specific FAQs to aid content comprehension" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05812v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05812v1.pdf" filename="2402.05812v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Frequently Asked Questions (FAQs) refer to the most common inquiries about specific content. They serve as content comprehension aids by simplifying topics and enhancing understanding through succinct presentation of information. In this paper, we address FAQ generation as a well-defined Natural Language Processing (NLP) task through the development of an end-to-end system leveraging text-to-text transformation models. We present a literature review covering traditional question-answering systems, highlighting their limitations when applied directly to the FAQ generation task. We propose our system capable of building FAQs from textual content tailored to specific domains, enhancing their accuracy and relevance. We utilise self-curated algorithms for obtaining optimal representation of information to be provided as input and also for ranking the question-answer pairs to maximise human comprehension. Qualitative human evaluation showcases the generated FAQs to be well-constructed and readable, while also utilising domain-specific constructs to highlight domain-based nuances and jargon in the original content.

{{</citation>}}


### (71/125) SpiRit-LM: Interleaved Spoken and Written Language Model (Tu Anh Nguyen et al., 2024)

{{<citation>}}

Tu Anh Nguyen, Benjamin Muller, Bokai Yu, Marta R. Costa-jussa, Maha Elbayad, Sravya Popuri, Paul-Ambroise Duquenne, Robin Algayres, Ruslan Mavlyutov, Itai Gat, Gabriel Synnaeve, Juan Pino, Benoit Sagot, Emmanuel Dupoux. (2024)  
**SpiRit-LM: Interleaved Spoken and Written Language Model**
<br/>
<button class="copy-to-clipboard" title="SpiRit-LM: Interleaved Spoken and Written Language Model" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05755v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05755v1.pdf" filename="2402.05755v1.pdf">Download PDF</button>

---


**ABSTRACT**  
We introduce SPIRIT-LM, a foundation multimodal language model that freely mixes text and speech. Our model is based on a pretrained text language model that we extend to the speech modality by continuously training it on text and speech units. Speech and text sequences are concatenated as a single set of tokens, and trained with a word-level interleaving method using a small automatically-curated speech-text parallel corpus. SPIRIT-LM comes in two versions: a BASE version that uses speech semantic units and an EXPRESSIVE version that models expressivity using pitch and style units in addition to the semantic units. For both versions, the text is encoded with subword BPE tokens. The resulting model displays both the semantic abilities of text models and the expressive abilities of speech models. Additionally, we demonstrate that SPIRIT-LM is able to learn new tasks in a few-shot fashion across modalities (i.e. ASR, TTS, Speech Classification).

{{</citation>}}


### (72/125) TimeArena: Shaping Efficient Multitasking Language Agents in a Time-Aware Simulation (Yikai Zhang et al., 2024)

{{<citation>}}

Yikai Zhang, Siyu Yuan, Caiyu Hu, Kyle Richardson, Yanghua Xiao, Jiangjie Chen. (2024)  
**TimeArena: Shaping Efficient Multitasking Language Agents in a Time-Aware Simulation**
<br/>
<button class="copy-to-clipboard" title="TimeArena: Shaping Efficient Multitasking Language Agents in a Time-Aware Simulation" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05733v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05733v1.pdf" filename="2402.05733v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Despite remarkable advancements in emulating human-like behavior through Large Language Models (LLMs), current textual simulations do not adequately address the notion of time. To this end, we introduce TimeArena, a novel textual simulated environment that incorporates complex temporal dynamics and constraints that better reflect real-life planning scenarios. In TimeArena, agents are asked to complete multiple tasks as soon as possible, allowing for parallel processing to save time. We implement the dependency between actions, the time duration for each action, and the occupancy of the agent and the objects in the environment. TimeArena grounds to 30 real-world tasks in cooking, household activities, and laboratory work. We conduct extensive experiments with various state-of-the-art LLMs using TimeArena. Our findings reveal that even the most powerful models, e.g., GPT-4, still lag behind humans in effective multitasking, underscoring the need for enhanced temporal awareness in the development of language agents.

{{</citation>}}


### (73/125) Unified Speech-Text Pretraining for Spoken Dialog Modeling (Heeseung Kim et al., 2024)

{{<citation>}}

Heeseung Kim, Soonshin Seo, Kyeongseok Jeong, Ohsung Kwon, Jungwhan Kim, Jaehong Lee, Eunwoo Song, Myungwoo Oh, Sungroh Yoon, Kang Min Yoo. (2024)  
**Unified Speech-Text Pretraining for Spoken Dialog Modeling**
<br/>
<button class="copy-to-clipboard" title="Unified Speech-Text Pretraining for Spoken Dialog Modeling" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Dialog  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05706v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05706v1.pdf" filename="2402.05706v1.pdf">Download PDF</button>

---


**ABSTRACT**  
While recent work shows promising results in expanding the capabilities of large language models (LLM) to directly understand and synthesize speech, an LLM-based strategy for modeling spoken dialogs remains elusive and calls for further investigation. This work proposes an extensive speech-text LLM framework, named the Unified Spoken Dialog Model (USDM), to generate coherent spoken responses with organic prosodic features relevant to the given input speech without relying on automatic speech recognition (ASR) or text-to-speech (TTS) solutions. Our approach employs a multi-step speech-text inference scheme that leverages chain-of-reasoning capabilities exhibited by the underlying LLM. We also propose a generalized speech-text pretraining scheme that helps with capturing cross-modal semantics. Automatic and human evaluations show that the proposed approach is effective in generating natural-sounding spoken responses, outperforming both prior and cascaded baselines. Detailed comparative studies reveal that, despite the cascaded approach being stronger in individual components, the joint speech-text modeling improves robustness against recognition errors and speech quality. Demo is available at https://unifiedsdm.github.io.

{{</citation>}}


### (74/125) Self-Alignment of Large Language Models via Monopolylogue-based Social Scene Simulation (Xianghe Pang et al., 2024)

{{<citation>}}

Xianghe Pang, Shuo Tang, Rui Ye, Yuxin Xiong, Bolun Zhang, Yanfeng Wang, Siheng Chen. (2024)  
**Self-Alignment of Large Language Models via Monopolylogue-based Social Scene Simulation**
<br/>
<button class="copy-to-clipboard" title="Self-Alignment of Large Language Models via Monopolylogue-based Social Scene Simulation" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: AI, GPT, GPT-4, Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05699v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05699v1.pdf" filename="2402.05699v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Aligning large language models (LLMs) with human values is imperative to mitigate potential adverse effects resulting from their misuse. Drawing from the sociological insight that acknowledging all parties' concerns is a key factor in shaping human values, this paper proposes a novel direction to align LLMs by themselves: social scene simulation. To achieve this, we present MATRIX, a novel social scene simulator that emulates realistic scenes around a user's input query, enabling the LLM to take social consequences into account before responding. MATRIX serves as a virtual rehearsal space, akin to a Monopolylogue, where the LLM performs diverse roles related to the query and practice by itself. To inject this alignment, we fine-tune the LLM with MATRIX-simulated data, ensuring adherence to human values without compromising inference speed. We theoretically show that the LLM with MATRIX outperforms Constitutional AI under mild assumptions. Finally, extensive experiments validate that our method outperforms over 10 baselines across 4 benchmarks. As evidenced by 875 user ratings, our tuned 13B-size LLM exceeds GPT-4 in aligning with human values. Code is available at https://github.com/pangxianghe/MATRIX.

{{</citation>}}


### (75/125) Multilingual E5 Text Embeddings: A Technical Report (Liang Wang et al., 2024)

{{<citation>}}

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, Furu Wei. (2024)  
**Multilingual E5 Text Embeddings: A Technical Report**
<br/>
<button class="copy-to-clipboard" title="Multilingual E5 Text Embeddings: A Technical Report" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Embedding, Multilingual  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05672v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05672v1.pdf" filename="2402.05672v1.pdf">Download PDF</button>

---


**ABSTRACT**  
This technical report presents the training methodology and evaluation results of the open-source multilingual E5 text embedding models, released in mid-2023. Three embedding models of different sizes (small / base / large) are provided, offering a balance between the inference efficiency and embedding quality. The training procedure adheres to the English E5 model recipe, involving contrastive pre-training on 1 billion multilingual text pairs, followed by fine-tuning on a combination of labeled datasets. Additionally, we introduce a new instruction-tuned embedding model, whose performance is on par with state-of-the-art, English-only models of similar sizes. Information regarding the model release can be found at https://github.com/microsoft/unilm/tree/master/e5 .

{{</citation>}}


### (76/125) Efficient Models for the Detection of Hate, Abuse and Profanity (Christoph Tillmann et al., 2024)

{{<citation>}}

Christoph Tillmann, Aashka Trivedi, Bishwaranjan Bhattacharjee. (2024)  
**Efficient Models for the Detection of Hate, Abuse and Profanity**
<br/>
<button class="copy-to-clipboard" title="Efficient Models for the Detection of Hate, Abuse and Profanity" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: BERT, Language Model, NLP, Natural Language Processing, Transformer, Transformers  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05624v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05624v1.pdf" filename="2402.05624v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Large Language Models (LLMs) are the cornerstone for many Natural Language Processing (NLP) tasks like sentiment analysis, document classification, named entity recognition, question answering, summarization, etc. LLMs are often trained on data which originates from the web. This data is prone to having content with Hate, Abuse and Profanity (HAP). For a detailed definition of HAP, please refer to the Appendix. Due to the LLMs being exposed to HAP content during training, the models learn it and may then generate hateful or profane content. For example, when the open-source RoBERTa model (specifically, the RoBERTA base model) from the HuggingFace (HF) Transformers library is prompted to replace the mask token in `I do not know that Persian people are that MASK` it returns the word `stupid` with the highest score. This is unacceptable in civil discourse.The detection of Hate, Abuse and Profanity in text is a vital component of creating civil and unbiased LLMs, which is needed not only for English, but for all languages. In this article, we briefly describe the creation of HAP detectors and various ways of using them to make models civil and acceptable in the output they generate.

{{</citation>}}


### (77/125) Deep Learning-based Computational Job Market Analysis: A Survey on Skill Extraction and Classification from Job Postings (Elena Senger et al., 2024)

{{<citation>}}

Elena Senger, Mike Zhang, Rob van der Goot, Barbara Plank. (2024)  
**Deep Learning-based Computational Job Market Analysis: A Survey on Skill Extraction and Classification from Job Postings**
<br/>
<button class="copy-to-clipboard" title="Deep Learning-based Computational Job Market Analysis: A Survey on Skill Extraction and Classification from Job Postings" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05617v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05617v1.pdf" filename="2402.05617v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Recent years have brought significant advances to Natural Language Processing (NLP), which enabled fast progress in the field of computational job market analysis. Core tasks in this application domain are skill extraction and classification from job postings. Because of its quick growth and its interdisciplinary nature, there is no exhaustive assessment of this emerging field. This survey aims to fill this gap by providing a comprehensive overview of deep learning methodologies, datasets, and terminologies specific to NLP-driven skill extraction and classification. Our comprehensive cataloging of publicly available datasets addresses the lack of consolidated information on dataset creation and characteristics. Finally, the focus on terminology addresses the current lack of consistent definitions for important concepts, such as hard and soft skills, and terms relating to skill extraction and classification.

{{</citation>}}


### (78/125) Pretrained Generative Language Models as General Learning Frameworks for Sequence-Based Tasks (Ben Fauber, 2024)

{{<citation>}}

Ben Fauber. (2024)  
**Pretrained Generative Language Models as General Learning Frameworks for Sequence-Based Tasks**
<br/>
<button class="copy-to-clipboard" title="Pretrained Generative Language Models as General Learning Frameworks for Sequence-Based Tasks" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05616v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05616v1.pdf" filename="2402.05616v1.pdf">Download PDF</button>

---


**ABSTRACT**  
We propose that small pretrained foundational generative language models with millions of parameters can be utilized as a general learning framework for sequence-based tasks. Our proposal overcomes the computational resource, skill set, and timeline challenges associated with training neural networks and language models from scratch. Further, our approach focuses on creating small and highly specialized models that can accurately execute a challenging task of which the base model is incapable of performing. We demonstrate that 125M, 350M, and 1.3B parameter pretrained foundational language models can be instruction fine-tuned with 10,000-to-1,000,000 instruction examples to achieve near state-of-the-art results on challenging cheminformatics tasks. We also demonstrate the role of successive language model fine-tuning epochs on improved outcomes, as well as the importance of both data formatting and pretrained foundational language model selection for instruction fine-tuning success.

{{</citation>}}


### (79/125) AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers (Reduan Achtibat et al., 2024)

{{<citation>}}

Reduan Achtibat, Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Aakriti Jain, Thomas Wiegand, Sebastian Lapuschkin, Wojciech Samek. (2024)  
**AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers**
<br/>
<button class="copy-to-clipboard" title="AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: Attention, Language Model, T5, Transformer, Transformers  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05602v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05602v1.pdf" filename="2402.05602v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Large Language Models are prone to biased predictions and hallucinations, underlining the paramount importance of understanding their model-internal reasoning process. However, achieving faithful attributions for the entirety of a black-box transformer model and maintaining computational efficiency is an unsolved challenge. By extending the Layer-wise Relevance Propagation attribution method to handle attention layers, we address these challenges effectively. While partial solutions exist, our method is the first to faithfully and holistically attribute not only input but also latent representations of transformer models with the computational efficiency similar to a singular backward pass. Through extensive evaluations against existing methods on Llama 2, Flan-T5 and the Vision Transformer architecture, we demonstrate that our proposed approach surpasses alternative methods in terms of faithfulness and enables the understanding of latent representations, opening up the door for concept-based explanations. We provide an open-source implementation on GitHub https://github.com/rachtibat/LRP-for-Transformers.

{{</citation>}}


### (80/125) SoftEDA: Rethinking Rule-Based Data Augmentation with Soft Labels (Juhwan Choi et al., 2024)

{{<citation>}}

Juhwan Choi, Kyohoon Jin, Junho Lee, Sangmin Song, Youngbin Kim. (2024)  
**SoftEDA: Rethinking Rule-Based Data Augmentation with Soft Labels**
<br/>
<button class="copy-to-clipboard" title="SoftEDA: Rethinking Rule-Based Data Augmentation with Soft Labels" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Augmentation, NLP  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05591v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05591v1.pdf" filename="2402.05591v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Rule-based text data augmentation is widely used for NLP tasks due to its simplicity. However, this method can potentially damage the original meaning of the text, ultimately hurting the performance of the model. To overcome this limitation, we propose a straightforward technique for applying soft labels to augmented data. We conducted experiments across seven different classification tasks and empirically demonstrated the effectiveness of our proposed approach. We have publicly opened our source code for reproducibility.

{{</citation>}}


### (81/125) AutoAugment Is What You Need: Enhancing Rule-based Augmentation Methods in Low-resource Regimes (Juhwan Choi et al., 2024)

{{<citation>}}

Juhwan Choi, Kyohoon Jin, Junho Lee, Sangmin Song, Youngbin Kim. (2024)  
**AutoAugment Is What You Need: Enhancing Rule-based Augmentation Methods in Low-resource Regimes**
<br/>
<button class="copy-to-clipboard" title="AutoAugment Is What You Need: Enhancing Rule-based Augmentation Methods in Low-resource Regimes" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Augmentation  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05584v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05584v1.pdf" filename="2402.05584v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Text data augmentation is a complex problem due to the discrete nature of sentences. Although rule-based augmentation methods are widely adopted in real-world applications because of their simplicity, they suffer from potential semantic damage. Previous researchers have suggested easy data augmentation with soft labels (softEDA), employing label smoothing to mitigate this problem. However, finding the best factor for each model and dataset is challenging; therefore, using softEDA in real-world applications is still difficult. In this paper, we propose adapting AutoAugment to solve this problem. The experimental results suggest that the proposed method can boost existing augmentation methods and that rule-based methods can enhance cutting-edge pre-trained language models. We offer the source code.

{{</citation>}}


### (82/125) Establishing degrees of closeness between audio recordings along different dimensions using large-scale cross-lingual models (Maxime Fily et al., 2024)

{{<citation>}}

Maxime Fily, Guillaume Wisniewski, Severine Guillaume, Gilles Adda, Alexis Michaud. (2024)  
**Establishing degrees of closeness between audio recordings along different dimensions using large-scale cross-lingual models**
<br/>
<button class="copy-to-clipboard" title="Establishing degrees of closeness between audio recordings along different dimensions using large-scale cross-lingual models" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Embedding  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05581v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05581v1.pdf" filename="2402.05581v1.pdf">Download PDF</button>

---


**ABSTRACT**  
In the highly constrained context of low-resource language studies, we explore vector representations of speech from a pretrained model to determine their level of abstraction with regard to the audio signal. We propose a new unsupervised method using ABX tests on audio recordings with carefully curated metadata to shed light on the type of information present in the representations. ABX tests determine whether the representations computed by a multilingual speech model encode a given characteristic. Three experiments are devised: one on room acoustics aspects, one on linguistic genre, and one on phonetic aspects. The results confirm that the representations extracted from recordings with different linguistic/extra-linguistic characteristics differ along the same lines. Embedding more audio signal in one vector better discriminates extra-linguistic characteristics, whereas shorter snippets are better to distinguish segmental information. The method is fully unsupervised, potentially opening new research avenues for comparative work on under-documented languages.

{{</citation>}}


### (83/125) Traditional Machine Learning Models and Bidirectional Encoder Representations From Transformer (BERT)-Based Automatic Classification of Tweets About Eating Disorders: Algorithm Development and Validation Study (José Alberto Benítez-Andrades et al., 2024)

{{<citation>}}

José Alberto Benítez-Andrades, José-Manuel Alija-Pérez, Maria-Esther Vidal, Rafael Pastor-Vargas, María Teresa García-Ordás. (2024)  
**Traditional Machine Learning Models and Bidirectional Encoder Representations From Transformer (BERT)-Based Automatic Classification of Tweets About Eating Disorders: Algorithm Development and Validation Study**
<br/>
<button class="copy-to-clipboard" title="Traditional Machine Learning Models and Bidirectional Encoder Representations From Transformer (BERT)-Based Automatic Classification of Tweets About Eating Disorders: Algorithm Development and Validation Study" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, Transformer  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05571v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05571v1.pdf" filename="2402.05571v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Background: Eating disorders are increasingly prevalent, and social networks offer valuable information.   Objective: Our goal was to identify efficient machine learning models for categorizing tweets related to eating disorders.   Methods: Over three months, we collected tweets about eating disorders. A 2,000-tweet subset was labeled for: (1) being written by individuals with eating disorders, (2) promoting eating disorders, (3) informativeness, and (4) scientific content. Both traditional machine learning and deep learning models were employed for classification, assessing accuracy, F1 score, and computational time.   Results: From 1,058,957 collected tweets, transformer-based bidirectional encoder representations achieved the highest F1 scores (71.1%-86.4%) across all four categories.   Conclusions: Transformer-based models outperform traditional techniques in classifying eating disorder-related tweets, though they require more computational resources.

{{</citation>}}


### (84/125) Benchmarking Large Language Models on Communicative Medical Coaching: a Novel System and Dataset (Hengguan Huang et al., 2024)

{{<citation>}}

Hengguan Huang, Songtao Wang, Hongfu Liu, Hao Wang, Ye Wang. (2024)  
**Benchmarking Large Language Models on Communicative Medical Coaching: a Novel System and Dataset**
<br/>
<button class="copy-to-clipboard" title="Benchmarking Large Language Models on Communicative Medical Coaching: a Novel System and Dataset" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, Language Model, NLP  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05547v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05547v1.pdf" filename="2402.05547v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Traditional applications of natural language processing (NLP) in healthcare have predominantly focused on patient-centered services, enhancing patient interactions and care delivery, such as through medical dialogue systems. However, the potential of NLP to benefit inexperienced doctors, particularly in areas such as communicative medical coaching, remains largely unexplored. We introduce ``ChatCoach,'' an integrated human-AI cooperative framework. Within this framework, both a patient agent and a coaching agent collaboratively support medical learners in practicing their medical communication skills during consultations. Unlike traditional dialogue systems, ChatCoach provides a simulated environment where a human doctor can engage in medical dialogue with a patient agent. Simultaneously, a coaching agent provides real-time feedback to the doctor. To construct the ChatCoach system, we developed a dataset and integrated Large Language Models such as ChatGPT and Llama2, aiming to assess their effectiveness in communicative medical coaching tasks. Our comparative analysis demonstrates that instruction-tuned Llama2 significantly outperforms ChatGPT's prompting-based approaches.

{{</citation>}}


### (85/125) Named Entity Recognition for Address Extraction in Speech-to-Text Transcriptions Using Synthetic Data (Bibiána Lajčinová et al., 2024)

{{<citation>}}

Bibiána Lajčinová, Patrik Valábek, Michal Spišiak. (2024)  
**Named Entity Recognition for Address Extraction in Speech-to-Text Transcriptions Using Synthetic Data**
<br/>
<button class="copy-to-clipboard" title="Named Entity Recognition for Address Extraction in Speech-to-Text Transcriptions Using Synthetic Data" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: BERT, GPT, NER, Named Entity Recognition, Transformer, Transformers  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05545v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05545v1.pdf" filename="2402.05545v1.pdf">Download PDF</button>

---


**ABSTRACT**  
This paper introduces an approach for building a Named Entity Recognition (NER) model built upon a Bidirectional Encoder Representations from Transformers (BERT) architecture, specifically utilizing the SlovakBERT model. This NER model extracts address parts from data acquired from speech-to-text transcriptions. Due to scarcity of real data, a synthetic dataset using GPT API was generated. The importance of mimicking spoken language variability in this artificial data is emphasized. The performance of our NER model, trained solely on synthetic data, is evaluated using small real test dataset.

{{</citation>}}


### (86/125) GPTs Are Multilingual Annotators for Sequence Generation Tasks (Juhwan Choi et al., 2024)

{{<citation>}}

Juhwan Choi, Eunju Lee, Kyohoon Jin, YoungBin Kim. (2024)  
**GPTs Are Multilingual Annotators for Sequence Generation Tasks**
<br/>
<button class="copy-to-clipboard" title="GPTs Are Multilingual Annotators for Sequence Generation Tasks" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Multilingual  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05512v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05512v1.pdf" filename="2402.05512v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Data annotation is an essential step for constructing new datasets. However, the conventional approach of data annotation through crowdsourcing is both time-consuming and expensive. In addition, the complexity of this process increases when dealing with low-resource languages owing to the difference in the language pool of crowdworkers. To address these issues, this study proposes an autonomous annotation method by utilizing large language models, which have been recently demonstrated to exhibit remarkable performance. Through our experiments, we demonstrate that the proposed method is not just cost-efficient but also applicable for low-resource language annotation. Additionally, we constructed an image captioning dataset using our approach and are committed to open this dataset for future study. We have opened our source code for further study and reproducibility.

{{</citation>}}


### (87/125) It's Never Too Late: Fusing Acoustic Information into Large Language Models for Automatic Speech Recognition (Chen Chen et al., 2024)

{{<citation>}}

Chen Chen, Ruizhe Li, Yuchen Hu, Sabato Marco Siniscalchi, Pin-Yu Chen, Ensiong Chng, Chao-Han Huck Yang. (2024)  
**It's Never Too Late: Fusing Acoustic Information into Large Language Models for Automatic Speech Recognition**
<br/>
<button class="copy-to-clipboard" title="It's Never Too Late: Fusing Acoustic Information into Large Language Models for Automatic Speech Recognition" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-MM, cs-SD, cs.CL, eess-AS  
Keywords: Language Model, Speech Recognition  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05457v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05457v1.pdf" filename="2402.05457v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Recent studies have successfully shown that large language models (LLMs) can be successfully used for generative error correction (GER) on top of the automatic speech recognition (ASR) output. Specifically, an LLM is utilized to carry out a direct mapping from the N-best hypotheses list generated by an ASR system to the predicted output transcription. However, despite its effectiveness, GER introduces extra data uncertainty since the LLM is trained without taking into account acoustic information available in the speech signal. In this work, we aim to overcome such a limitation by infusing acoustic information before generating the predicted transcription through a novel late fusion solution termed Uncertainty-Aware Dynamic Fusion (UADF). UADF is a multimodal fusion approach implemented into an auto-regressive decoding process and works in two stages: (i) It first analyzes and calibrates the token-level LLM decision, and (ii) it then dynamically assimilates the information from the acoustic modality. Experimental evidence collected from various ASR tasks shows that UADF surpasses existing fusion mechanisms in several ways. It yields significant improvements in word error rate (WER) while mitigating data uncertainty issues in LLM and addressing the poor generalization relied with sole modality during fusion. We also demonstrate that UADF seamlessly adapts to audio-visual speech recognition.

{{</citation>}}


### (88/125) Large Language Models for Psycholinguistic Plausibility Pretesting (Samuel Joseph Amouyal et al., 2024)

{{<citation>}}

Samuel Joseph Amouyal, Aya Meltzer-Asscher, Jonathan Berant. (2024)  
**Large Language Models for Psycholinguistic Plausibility Pretesting**
<br/>
<button class="copy-to-clipboard" title="Large Language Models for Psycholinguistic Plausibility Pretesting" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05455v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05455v1.pdf" filename="2402.05455v1.pdf">Download PDF</button>

---


**ABSTRACT**  
In psycholinguistics, the creation of controlled materials is crucial to ensure that research outcomes are solely attributed to the intended manipulations and not influenced by extraneous factors. To achieve this, psycholinguists typically pretest linguistic materials, where a common pretest is to solicit plausibility judgments from human evaluators on specific sentences. In this work, we investigate whether Language Models (LMs) can be used to generate these plausibility judgements. We investigate a wide range of LMs across multiple linguistic structures and evaluate whether their plausibility judgements correlate with human judgements. We find that GPT-4 plausibility judgements highly correlate with human judgements across the structures we examine, whereas other LMs correlate well with humans on commonly used syntactic structures. We then test whether this correlation implies that LMs can be used instead of humans for pretesting. We find that when coarse-grained plausibility judgements are needed, this works well, but when fine-grained judgements are necessary, even GPT-4 does not provide satisfactory discriminative power.

{{</citation>}}


### (89/125) Improving Agent Interactions in Virtual Environments with Language Models (Jack Zhang, 2024)

{{<citation>}}

Jack Zhang. (2024)  
**Improving Agent Interactions in Virtual Environments with Language Models**
<br/>
<button class="copy-to-clipboard" title="Improving Agent Interactions in Virtual Environments with Language Models" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05440v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05440v1.pdf" filename="2402.05440v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Enhancing AI systems with efficient communication skills for effective human assistance necessitates proactive initiatives from the system side to discern specific circumstances and interact aptly. This research focuses on a collective building assignment in the Minecraft dataset, employing language modeling to enhance task understanding through state-of-the-art methods. These models focus on grounding multi-modal understanding and task-oriented dialogue comprehension tasks, providing insights into their interpretative and responsive capabilities. Our experimental results showcase a substantial improvement over existing methods, indicating a promising direction for future research in this domain.

{{</citation>}}


### (90/125) GPT-4 Generated Narratives of Life Events using a Structured Narrative Prompt: A Validation Study (Christopher J. Lynch et al., 2024)

{{<citation>}}

Christopher J. Lynch, Erik Jensen, Madison H. Munro, Virginia Zamponi, Joseph Martinez, Kevin O'Brien, Brandon Feldhaus, Katherine Smith, Ann Marie Reinhold, Ross Gore. (2024)  
**GPT-4 Generated Narratives of Life Events using a Structured Narrative Prompt: A Validation Study**
<br/>
<button class="copy-to-clipboard" title="GPT-4 Generated Narratives of Life Events using a Structured Narrative Prompt: A Validation Study" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: I-2-7; I-6-4, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, GPT, GPT-4, Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05435v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05435v1.pdf" filename="2402.05435v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Large Language Models (LLMs) play a pivotal role in generating vast arrays of narratives, facilitating a systematic exploration of their effectiveness for communicating life events in narrative form. In this study, we employ a zero-shot structured narrative prompt to generate 24,000 narratives using OpenAI's GPT-4. From this dataset, we manually classify 2,880 narratives and evaluate their validity in conveying birth, death, hiring, and firing events. Remarkably, 87.43% of the narratives sufficiently convey the intention of the structured prompt. To automate the identification of valid and invalid narratives, we train and validate nine Machine Learning models on the classified datasets. Leveraging these models, we extend our analysis to predict the classifications of the remaining 21,120 narratives. All the ML models excelled at classifying valid narratives as valid, but experienced challenges at simultaneously classifying invalid narratives as invalid. Our findings not only advance the study of LLM capabilities, limitations, and validity but also offer practical insights for narrative generation and natural language processing applications.

{{</citation>}}


### (91/125) In-Context Principle Learning from Mistakes (Tianjun Zhang et al., 2024)

{{<citation>}}

Tianjun Zhang, Aman Madaan, Luyu Gao, Steven Zheng, Swaroop Mishra, Yiming Yang, Niket Tandon, Uri Alon. (2024)  
**In-Context Principle Learning from Mistakes**
<br/>
<button class="copy-to-clipboard" title="In-Context Principle Learning from Mistakes" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, QA  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05403v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05403v1.pdf" filename="2402.05403v1.pdf">Download PDF</button>

---


**ABSTRACT**  
In-context learning (ICL, also known as few-shot prompting) has been the standard method of adapting LLMs to downstream tasks, by learning from a few input-output examples. Nonetheless, all ICL-based approaches only learn from correct input-output pairs. In this paper, we revisit this paradigm, by learning more from the few given input-output examples. We introduce Learning Principles (LEAP): First, we intentionally induce the model to make mistakes on these few examples; then we reflect on these mistakes, and learn explicit task-specific "principles" from them, which help solve similar problems and avoid common mistakes; finally, we prompt the model to answer unseen test questions using the original few-shot examples and these learned general principles. We evaluate LEAP on a wide range of benchmarks, including multi-hop question answering (Hotpot QA), textual QA (DROP), Big-Bench Hard reasoning, and math problems (GSM8K and MATH); in all these benchmarks, LEAP improves the strongest available LLMs such as GPT-3.5-turbo, GPT-4, GPT-4 turbo and Claude-2.1. For example, LEAP improves over the standard few-shot prompting using GPT-4 by 7.5% in DROP, and by 3.3% in HotpotQA. Importantly, LEAP does not require any more input or examples than the standard few-shot prompting settings.

{{</citation>}}


### (92/125) Zero-Shot Chain-of-Thought Reasoning Guided by Evolutionary Algorithms in Large Language Models (Feihu Jin et al., 2024)

{{<citation>}}

Feihu Jin, Yifan Liu, Ying Tan. (2024)  
**Zero-Shot Chain-of-Thought Reasoning Guided by Evolutionary Algorithms in Large Language Models**
<br/>
<button class="copy-to-clipboard" title="Zero-Shot Chain-of-Thought Reasoning Guided by Evolutionary Algorithms in Large Language Models" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, Reasoning, Zero-Shot  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05376v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05376v1.pdf" filename="2402.05376v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks and exhibited impressive reasoning abilities by applying zero-shot Chain-of-Thought (CoT) prompting. However, due to the evolving nature of sentence prefixes during the pre-training phase, existing zero-shot CoT prompting methods that employ identical CoT prompting across all task instances may not be optimal. In this paper, we introduce a novel zero-shot prompting method that leverages evolutionary algorithms to generate diverse promptings for LLMs dynamically. Our approach involves initializing two CoT promptings, performing evolutionary operations based on LLMs to create a varied set, and utilizing the LLMs to select a suitable CoT prompting for a given problem. Additionally, a rewriting operation, guided by the selected CoT prompting, enhances the understanding of the LLMs about the problem. Extensive experiments conducted across ten reasoning datasets demonstrate the superior performance of our proposed method compared to current zero-shot CoT prompting methods on GPT-3.5-turbo and GPT-4. Moreover, in-depth analytical experiments underscore the adaptability and effectiveness of our method in various reasoning tasks.

{{</citation>}}


## cs.HC (6)



### (93/125) Personalizing Driver Safety Interfaces via Driver Cognitive Factors Inference (Emily S Sumner et al., 2024)

{{<citation>}}

Emily S Sumner, Jonathan DeCastro, Jean Costa, Deepak E Gopinath, Everlyne Kimani, Shabnam Hakimi, Allison Morgan, Andrew Best, Hieu Nguyen, Daniel J Brooks, Bassam ul Haq, Andrew Patrikalakis, Hiroshi Yasuda, Kate Sieck, Avinash Balachandran, Tiffany Chen, Guy Rosman. (2024)  
**Personalizing Driver Safety Interfaces via Driver Cognitive Factors Inference**
<br/>
<button class="copy-to-clipboard" title="Personalizing Driver Safety Interfaces via Driver Cognitive Factors Inference" index=5>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-5 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05893v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05893v1.pdf" filename="2402.05893v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Recent advances in AI and intelligent vehicle technology hold promise to revolutionize mobility and transportation, in the form of advanced driving assistance (ADAS) interfaces. Although it is widely recognized that certain cognitive factors, such as impulsivity and inhibitory control, are related to risky driving behavior, play a significant role in on-road risk-taking, existing systems fail to leverage such factors. Varying levels of these cognitive factors could influence the effectiveness and acceptance of driver safety interfaces.   We demonstrate an approach for personalizing driver interaction via driver safety interfaces that are triggered based on a learned recurrent neural network. The network is trained from a population of human drivers to infer impulsivity and inhibitory control from recent driving behavior. Using a high-fidelity vehicle motion simulator, we demonstrate the ability to deduce these factors from driver behavior. We then use these inferred factors to make instantaneous determinations on whether or not to engage a driver safety interface. This interface aims to decrease a driver's speed during yellow lights and reduce their inclination to run through them.

{{</citation>}}


### (94/125) Visual Harmony: Text-Visual Interplay in Circular Infographics (Shuqi He et al., 2024)

{{<citation>}}

Shuqi He, Yuqing Chen, Yuxin Xia, Yichun Li, Hai-Ning Liang, Lingyun Yu. (2024)  
**Visual Harmony: Text-Visual Interplay in Circular Infographics**
<br/>
<button class="copy-to-clipboard" title="Visual Harmony: Text-Visual Interplay in Circular Infographics" index=5>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-5 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Embedding  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05798v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05798v1.pdf" filename="2402.05798v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Infographics are visual representations designed for efficient and effective communication of data and knowledge. One crucial aspect of infographic design is the interplay between text and visual elements, particularly in circular visualizations where the textual descriptions can either be embedded within the graphics or placed adjacent to the visual representation. While several studies have examined text layout design in visualizations in general, the text-visual interplay in infographics and its subsequent perceptual effects remain underexplored. To address this, our study investigates how varying text placement and descriptiveness impact pleasantness, comprehension and overall memorability in the infographics viewing experience. We recruited 30 participants and presented them with a collection of 15 infographics across a diverse set of topics, including media and public events, health and nutrition, science and research, and sustainability. The text placement (embed, side-to-side) and descriptiveness (simplistic, normal, descriptive) were systematically manipulated, resulting in a total of six experimental conditions. Our key findings indicate that text placement can significantly influence the memorability of infographics, whereas descriptiveness can significantly impact the pleasantness of the viewing experience. Embedding text placement and simplistic text can potentially contribute to more effective infographic designs. These results offer valuable insights for infographic designers, contributing to the creation of more effective and memorable visual representations.

{{</citation>}}


### (95/125) Evolving AI for Wellness: Dynamic and Personalized Real-time Loneliness Detection Using Passive Sensing (Malik Muhammad Qirtas et al., 2024)

{{<citation>}}

Malik Muhammad Qirtas, Evi Zafeiridi, Eleanor Bantry White, Dirk Pesch. (2024)  
**Evolving AI for Wellness: Dynamic and Personalized Real-time Loneliness Detection Using Passive Sensing**
<br/>
<button class="copy-to-clipboard" title="Evolving AI for Wellness: Dynamic and Personalized Real-time Loneliness Detection Using Passive Sensing" index=5>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-5 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05698v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05698v1.pdf" filename="2402.05698v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Loneliness is a growing health concern as it can lead to depression and other associated mental health problems for people who experience feelings of loneliness over prolonged periods of time. Utilizing passive sensing methods that use smartphone and wearable sensor data to capture daily behavioural patterns offers a promising approach for the early detection of loneliness. Given the subjective nature of loneliness and people's varying daily routines, past detection approaches using machine learning models often face challenges with effectively detecting loneliness. This paper proposes a methodologically novel approach, particularly developing a loneliness detection system that evolves over time, adapts to new data, and provides real-time detection. Our study utilized the Globem dataset, a comprehensive collection of passive sensing data acquired over 10 weeks from university students. The base of our approach is the continuous identification and refinement of similar behavioural groups among students using an incremental clustering method. As we add new data, the model improves based on changing behavioural patterns. Parallel to this, we create and update classification models to detect loneliness among the evolving behavioural groups of students. When unique behavioural patterns are observed among student data, specialized classification models have been created. For predictions of loneliness, a collaborative effort between the generalized and specialized models is employed, treating each prediction as a vote. This study's findings reveal that group-based loneliness detection models exhibit superior performance compared to generic models, underscoring the necessity for more personalized approaches tailored to specific behavioural patterns. These results pave the way for future research, emphasizing the development of finely-tuned, individualized mental health interventions.

{{</citation>}}


### (96/125) Kontextbasierte Aktivitätserkennung -- Synergie von Mensch und Technik in der Social Networked Industry (Friedrich Niemann et al., 2024)

{{<citation>}}

Friedrich Niemann, Christopher Reining. (2024)  
**Kontextbasierte Aktivitätserkennung -- Synergie von Mensch und Technik in der Social Networked Industry**
<br/>
<button class="copy-to-clipboard" title="Kontextbasierte Aktivitätserkennung -- Synergie von Mensch und Technik in der Social Networked Industry" index=5>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-5 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Social Network  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05480v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05480v1.pdf" filename="2402.05480v1.pdf">Download PDF</button>

---


**ABSTRACT**  
In a social networked industry, the focus is on collaboration between humans and technology. Communication is the basic prerequisite for synergetic collaboration between all players. It includes non-verbal as well as verbal interactions. To enable non-verbal interaction, machines must be able to detect and understand human movements. This article presents the ongoing fundamental research on the analysis of human movements using sensor-based activity recognition and identifies potential for a transfer to industrial applications. The focus is on the practical feasibility of activity recognition by adding further data streams such as the position data of logistical objects and tools, meaning the context in which a certain activity is carried out.   --   In der Social Networked Industry steht die Zusammenarbeit von Mensch und Technik im Vordergrund. Grundvoraussetzung f\"ur eine synergetische Zusammenarbeit aller Akteure ist die Kommunikation, welche neben verbalen auch nonverbale Interaktionen umfasst. Um eine nonverbale Interaktion zu erm\"oglichen, m\"ussen Maschinen in der Lage sein, menschliche Bewegungen zu erfassen und zu verstehen. Dieser Beitrag stellt die laufende Grundlagenforschung zur Analyse menschlicher Bewegungen mittels sensorgest\"utzter Aktivit\"atserkennung vor und zeigt Ankn\"upfungspunkte f\"ur einen Transfer in industrielle Anwendungen. Im Fokus steht die Praxistauglichkeit der Aktivit\"atserkennung durch die Hinzunahme weiterer Datenstr\"ome wie beispielsweise den Positionsdaten logistischer Objekte und Hilfsmitteln, d. h. dem Kontext, in dem eine gewisse Aktivit\"at ausgef\"uhrt wird.

{{</citation>}}


### (97/125) Form-From: A Design Space of Social Media Systems (Amy X. Zhang et al., 2024)

{{<citation>}}

Amy X. Zhang, Michael S. Bernstein, David R. Karger, Mark S. Ackerman. (2024)  
**Form-From: A Design Space of Social Media Systems**
<br/>
<button class="copy-to-clipboard" title="Form-From: A Design Space of Social Media Systems" index=5>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-5 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.HC  
Categories: cs-HC, cs-SI, cs.HC  
Keywords: Social Media  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05388v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05388v1.pdf" filename="2402.05388v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Social media systems are as varied as they are pervasive. They have been almost universally adopted for a broad range of purposes including work, entertainment, activism, and decision making. As a result, they have also diversified, with many distinct designs differing in content type, organization, delivery mechanism, access control, and many other dimensions. In this work, we aim to characterize and then distill a concise design space of social media systems that can help us understand similarities and differences, recognize potential consequences of design choice, and identify spaces for innovation. Our model, which we call Form-From, characterizes social media based on (1) the form of the content, either threaded or flat, and (2) from where or from whom one might receive content, ranging from spaces to networks to the commons. We derive Form-From inductively from a larger set of 62 dimensions organized into 10 categories. To demonstrate the utility of our model, we trace the history of social media systems as they traverse the Form-From space over time, and we identify common design patterns within cells of the model.

{{</citation>}}


### (98/125) Are We Asking the Right Questions?: Designing for Community Stakeholders' Interactions with AI in Policing (MD Romael Haque et al., 2024)

{{<citation>}}

MD Romael Haque, Devansh Saxena, Katy Weathington, Joseph Chudzik, Shion Guha. (2024)  
**Are We Asking the Right Questions?: Designing for Community Stakeholders' Interactions with AI in Policing**
<br/>
<button class="copy-to-clipboard" title="Are We Asking the Right Questions?: Designing for Community Stakeholders' Interactions with AI in Policing" index=5>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-5 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05348v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05348v1.pdf" filename="2402.05348v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Research into recidivism risk prediction in the criminal legal system has garnered significant attention from HCI, critical algorithm studies, and the emerging field of human-AI decision-making. This study focuses on algorithmic crime mapping, a prevalent yet underexplored form of algorithmic decision support (ADS) in this context. We conducted experiments and follow-up interviews with 60 participants, including community members, technical experts, and law enforcement agents (LEAs), to explore how lived experiences, technical knowledge, and domain expertise shape interactions with the ADS, impacting human-AI decision-making. Surprisingly, we found that domain experts (LEAs) often exhibited anchoring bias, readily accepting and engaging with the first crime map presented to them. Conversely, community members and technical experts were more inclined to engage with the tool, adjust controls, and generate different maps. Our findings highlight that all three stakeholders were able to provide critical feedback regarding AI design and use - community members questioned the core motivation of the tool, technical experts drew attention to the elastic nature of data science practice, and LEAs suggested redesign pathways such that the tool could complement their domain expertise.

{{</citation>}}


## cs.SI (3)



### (99/125) GET-Tok: A GenAI-Enriched Multimodal TikTok Dataset Documenting the 2022 Attempted Coup in Peru (Gabriela Pinto et al., 2024)

{{<citation>}}

Gabriela Pinto, Keith Burghardt, Kristina Lerman, Emilio Ferrara. (2024)  
**GET-Tok: A GenAI-Enriched Multimodal TikTok Dataset Documenting the 2022 Attempted Coup in Peru**
<br/>
<button class="copy-to-clipboard" title="GET-Tok: A GenAI-Enriched Multimodal TikTok Dataset Documenting the 2022 Attempted Coup in Peru" index=6>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-6 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.SI  
Categories: cs-CY, cs-HC, cs-SI, cs.SI  
Keywords: AI, Generative AI, OCR  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05882v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05882v1.pdf" filename="2402.05882v1.pdf">Download PDF</button>

---


**ABSTRACT**  
TikTok is one of the largest and fastest-growing social media sites in the world. TikTok features, however, such as voice transcripts, are often missing and other important features, such as OCR or video descriptions, do not exist. We introduce the Generative AI Enriched TikTok (GET-Tok) data, a pipeline for collecting TikTok videos and enriched data by augmenting the TikTok Research API with generative AI models. As a case study, we collect videos about the attempted coup in Peru initiated by its former President, Pedro Castillo, and its accompanying protests. The data includes information on 43,697 videos published from November 20, 2022 to March 1, 2023 (102 days). Generative AI augments the collected data via transcripts of TikTok videos, text descriptions of what is shown in the videos, what text is displayed within the video, and the stances expressed in the video. Overall, this pipeline will contribute to a better understanding of online discussion in a multimodal setting with applications of Generative AI, especially outlining the utility of this pipeline in non-English-language social media. Our code used to produce the pipeline is in a public Github repository: https://github.com/gabbypinto/GET-Tok-Peru.

{{</citation>}}


### (100/125) Coordinated Activity Modulates the Behavior and Emotions of Organic Users: A Case Study on Tweets about the Gaza Conflict (Priyanka Dey et al., 2024)

{{<citation>}}

Priyanka Dey, Luca Luceri, Emilio Ferrara. (2024)  
**Coordinated Activity Modulates the Behavior and Emotions of Organic Users: A Case Study on Tweets about the Gaza Conflict**
<br/>
<button class="copy-to-clipboard" title="Coordinated Activity Modulates the Behavior and Emotions of Organic Users: A Case Study on Tweets about the Gaza Conflict" index=6>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-6 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Twitter  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05873v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05873v1.pdf" filename="2402.05873v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Social media has become a crucial conduit for the swift dissemination of information during global crises. However, this also paves the way for the manipulation of narratives by malicious actors. This research delves into the interaction dynamics between coordinated (malicious) entities and organic (regular) users on Twitter amidst the Gaza conflict. Through the analysis of approximately 3.5 million tweets from over 1.3 million users, our study uncovers that coordinated users significantly impact the information landscape, successfully disseminating their content across the network: a substantial fraction of their messages is adopted and shared by organic users. Furthermore, the study documents a progressive increase in organic users' engagement with coordinated content, which is paralleled by a discernible shift towards more emotionally polarized expressions in their subsequent communications. These results highlight the critical need for vigilance and a nuanced understanding of information manipulation on social media platforms.

{{</citation>}}


### (101/125) Exploring the Nostr Ecosystem: A Study of Decentralization and Resilience (Yiluo Wei et al., 2024)

{{<citation>}}

Yiluo Wei, Gareth Tyson. (2024)  
**Exploring the Nostr Ecosystem: A Study of Decentralization and Resilience**
<br/>
<button class="copy-to-clipboard" title="Exploring the Nostr Ecosystem: A Study of Decentralization and Resilience" index=6>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-6 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05709v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05709v1.pdf" filename="2402.05709v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Nostr is an open decentralized social network launched in 2022. From a user's perspective, it is similar to a micro-blogging service like Twitter. However, the underlying infrastructure is very different, and Nostr boasts a range of unique features that set it apart. Nostr introduces the concept of relays, which act as open storage servers that receive, store, and distribute user posts. Each user is uniquely identified by a public key, ensuring authenticity of posts through digital signatures. Consequently, users are able to securely send and receive posts through various relays, which frees them from single-server reliance and enhances post availability (e.g., making it more censorship resistant). The Nostr ecosystem has garnered significant attention, boasting 4 million users and 60 million posts in just 2 years. To understand its characteristics and challenges, we conduct the first large-scale measurement of the Nostr ecosystem, spanning from July 1, 2023, to December 31, 2023. Our study focuses on two key aspects: Nostr relays and post replication strategies. We find that Nostr achieves superior decentralization compared to traditional Fediverse applications. However, relay availability remains a challenge, where financial sustainability (particularly for free-to-use relays) emerges as a contributing factor. We also find that the replication of posts across relays enhances post availability but introduces significant overhead. To address this, we propose two design innovations. One to control the number of post replications, and another to reduce the overhead during post retrieval. Via data-driven evaluations, we demonstrate their effectiveness without negatively impacting the system.

{{</citation>}}


## stat.ML (3)



### (102/125) Prior-Dependent Allocations for Bayesian Fixed-Budget Best-Arm Identification in Structured Bandits (Nicolas Nguyen et al., 2024)

{{<citation>}}

Nicolas Nguyen, Imad Aouali, András György, Claire Vernade. (2024)  
**Prior-Dependent Allocations for Bayesian Fixed-Budget Best-Arm Identification in Structured Bandits**
<br/>
<button class="copy-to-clipboard" title="Prior-Dependent Allocations for Bayesian Fixed-Budget Best-Arm Identification in Structured Bandits" index=7>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-7 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: AI  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05878v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05878v1.pdf" filename="2402.05878v1.pdf">Download PDF</button>

---


**ABSTRACT**  
We study the problem of Bayesian fixed-budget best-arm identification (BAI) in structured bandits. We propose an algorithm that uses fixed allocations based on the prior information and the structure of the environment. We provide theoretical bounds on its performance across diverse models, including the first prior-dependent upper bounds for linear and hierarchical BAI. Our key contribution is introducing new proof methods that result in tighter bounds for multi-armed BAI compared to existing methods. We extensively compare our approach to other fixed-budget BAI methods, demonstrating its consistent and robust performance in various settings. Our work improves our understanding of Bayesian fixed-budget BAI in structured bandits and highlights the effectiveness of our approach in practical scenarios.

{{</citation>}}


### (103/125) How do Transformers perform In-Context Autoregressive Learning? (Michael E. Sander et al., 2024)

{{<citation>}}

Michael E. Sander, Raja Giryes, Taiji Suzuki, Mathieu Blondel, Gabriel Peyré. (2024)  
**How do Transformers perform In-Context Autoregressive Learning?**
<br/>
<button class="copy-to-clipboard" title="How do Transformers perform In-Context Autoregressive Learning?" index=7>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-7 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Transformer, Transformers  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05787v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05787v1.pdf" filename="2402.05787v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Transformers have achieved state-of-the-art performance in language modeling tasks. However, the reasons behind their tremendous success are still unclear. In this paper, towards a better understanding, we train a Transformer model on a simple next token prediction task, where sequences are generated as a first-order autoregressive process $s_{t+1} = W s_t$. We show how a trained Transformer predicts the next token by first learning $W$ in-context, then applying a prediction mapping. We call the resulting procedure in-context autoregressive learning. More precisely, focusing on commuting orthogonal matrices $W$, we first show that a trained one-layer linear Transformer implements one step of gradient descent for the minimization of an inner objective function, when considering augmented tokens. When the tokens are not augmented, we characterize the global minima of a one-layer diagonal linear multi-head Transformer. Importantly, we exhibit orthogonality between heads and show that positional encoding captures trigonometric relations in the data. On the experimental side, we consider the general case of non-commuting orthogonal matrices and generalize our theoretical findings.

{{</citation>}}


### (104/125) A High Dimensional Model for Adversarial Training: Geometry and Trade-Offs (Kasimir Tanner et al., 2024)

{{<citation>}}

Kasimir Tanner, Matteo Vilucchio, Bruno Loureiro, Florent Krzakala. (2024)  
**A High Dimensional Model for Adversarial Training: Geometry and Trade-Offs**
<br/>
<button class="copy-to-clipboard" title="A High Dimensional Model for Adversarial Training: Geometry and Trade-Offs" index=7>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-7 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: stat.ML  
Categories: cond-mat-dis-nn, cs-LG, stat-ML, stat.ML  
Keywords: Adversarial Training  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05674v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05674v1.pdf" filename="2402.05674v1.pdf">Download PDF</button>

---


**ABSTRACT**  
This work investigates adversarial training in the context of margin-based linear classifiers in the high-dimensional regime where the dimension $d$ and the number of data points $n$ diverge with a fixed ratio $\alpha = n / d$. We introduce a tractable mathematical model where the interplay between the data and adversarial attacker geometries can be studied, while capturing the core phenomenology observed in the adversarial robustness literature. Our main theoretical contribution is an exact asymptotic description of the sufficient statistics for the adversarial empirical risk minimiser, under generic convex and non-increasing losses. Our result allow us to precisely characterise which directions in the data are associated with a higher generalisation/robustness trade-off, as defined by a robustness and a usefulness metric. In particular, we unveil the existence of directions which can be defended without penalising accuracy. Finally, we show the advantage of defending non-robust features during training, identifying a uniform protection as an inherently effective defence mechanism.

{{</citation>}}


## eess.AS (1)



### (105/125) Integrating Self-supervised Speech Model with Pseudo Word-level Targets from Visually-grounded Speech Model (Hung-Chieh Fang et al., 2024)

{{<citation>}}

Hung-Chieh Fang, Nai-Xuan Ye, Yi-Jen Shih, Puyuan Peng, Hsuan-Fu Wang, Layne Berry, Hung-yi Lee, David Harwath. (2024)  
**Integrating Self-supervised Speech Model with Pseudo Word-level Targets from Visually-grounded Speech Model**
<br/>
<button class="copy-to-clipboard" title="Integrating Self-supervised Speech Model with Pseudo Word-level Targets from Visually-grounded Speech Model" index=8>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-8 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: eess.AS  
Categories: cs-CL, cs-LG, eess-AS, eess.AS  
Keywords: BERT  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05819v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05819v1.pdf" filename="2402.05819v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Recent advances in self-supervised speech models have shown significant improvement in many downstream tasks. However, these models predominantly centered on frame-level training objectives, which can fall short in spoken language understanding tasks that require semantic comprehension. Existing works often rely on additional speech-text data as intermediate targets, which is costly in the real-world setting. To address this challenge, we propose Pseudo-Word HuBERT (PW-HuBERT), a framework that integrates pseudo word-level targets into the training process, where the targets are derived from a visually-ground speech model, notably eliminating the need for speech-text paired data. Our experimental results on four spoken language understanding (SLU) benchmarks suggest the superiority of our model in capturing semantic information.

{{</citation>}}


## eess.IV (3)



### (106/125) Using YOLO v7 to Detect Kidney in Magnetic Resonance Imaging: A Supervised Contrastive Learning (Pouria Yazdian Anari et al., 2024)

{{<citation>}}

Pouria Yazdian Anari, Fiona Obiezu, Nathan Lay, Fatemeh Dehghani Firouzabadi, Aditi Chaurasia, Mahshid Golagha, Shiva Singh, Fatemeh Homayounieh, Aryan Zahergivar, Stephanie Harmon, Evrim Turkbey, Rabindra Gautam, Kevin Ma, Maria Merino, Elizabeth C. Jones, Mark W. Ball, W. Marston Linehan, Baris Turkbey, Ashkan A. Malayeri. (2024)  
**Using YOLO v7 to Detect Kidney in Magnetic Resonance Imaging: A Supervised Contrastive Learning**
<br/>
<button class="copy-to-clipboard" title="Using YOLO v7 to Detect Kidney in Magnetic Resonance Imaging: A Supervised Contrastive Learning" index=9>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-9 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Contrastive Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05817v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05817v1.pdf" filename="2402.05817v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Introduction This study explores the use of the latest You Only Look Once (YOLO V7) object detection method to enhance kidney detection in medical imaging by training and testing a modified YOLO V7 on medical image formats. Methods Study includes 878 patients with various subtypes of renal cell carcinoma (RCC) and 206 patients with normal kidneys. A total of 5657 MRI scans for 1084 patients were retrieved. 326 patients with 1034 tumors recruited from a retrospective maintained database, and bounding boxes were drawn around their tumors. A primary model was trained on 80% of annotated cases, with 20% saved for testing (primary test set). The best primary model was then used to identify tumors in the remaining 861 patients and bounding box coordinates were generated on their scans using the model. Ten benchmark training sets were created with generated coordinates on not-segmented patients. The final model used to predict the kidney in the primary test set. We reported the positive predictive value (PPV), sensitivity, and mean average precision (mAP). Results The primary training set showed an average PPV of 0.94 +/- 0.01, sensitivity of 0.87 +/- 0.04, and mAP of 0.91 +/- 0.02. The best primary model yielded a PPV of 0.97, sensitivity of 0.92, and mAP of 0.95. The final model demonstrated an average PPV of 0.95 +/- 0.03, sensitivity of 0.98 +/- 0.004, and mAP of 0.95 +/- 0.01. Conclusion Using a semi-supervised approach with a medical image library, we developed a high-performing model for kidney detection. Further external validation is required to assess the model's generalizability.

{{</citation>}}


### (107/125) Joint End-to-End Image Compression and Denoising: Leveraging Contrastive Learning and Multi-Scale Self-ONNs (Yuxin Xie et al., 2024)

{{<citation>}}

Yuxin Xie, Li Yu, Farhad Pakdaman, Moncef Gabbouj. (2024)  
**Joint End-to-End Image Compression and Denoising: Leveraging Contrastive Learning and Multi-Scale Self-ONNs**
<br/>
<button class="copy-to-clipboard" title="Joint End-to-End Image Compression and Denoising: Leveraging Contrastive Learning and Multi-Scale Self-ONNs" index=9>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-9 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: eess.IV  
Categories: cs-CV, cs-MM, eess-IV, eess.IV  
Keywords: Contrastive Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05582v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05582v1.pdf" filename="2402.05582v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Noisy images are a challenge to image compression algorithms due to the inherent difficulty of compressing noise. As noise cannot easily be discerned from image details, such as high-frequency signals, its presence leads to extra bits needed for compression. Since the emerging learned image compression paradigm enables end-to-end optimization of codecs, recent efforts were made to integrate denoising into the compression model, relying on clean image features to guide denoising. However, these methods exhibit suboptimal performance under high noise levels, lacking the capability to generalize across diverse noise types. In this paper, we propose a novel method integrating a multi-scale denoiser comprising of Self Organizing Operational Neural Networks, for joint image compression and denoising. We employ contrastive learning to boost the network ability to differentiate noise from high frequency signal components, by emphasizing the correlation between noisy and clean counterparts. Experimental results demonstrate the effectiveness of the proposed method both in rate-distortion performance, and codec speed, outperforming the current state-of-the-art.

{{</citation>}}


### (108/125) Unleashing the Infinity Power of Geometry: A Novel Geometry-Aware Transformer (GOAT) for Whole Slide Histopathology Image Analysis (Mingxin Liu et al., 2024)

{{<citation>}}

Mingxin Liu, Yunzan Liu, Pengbo Xu, Jiquan Ma. (2024)  
**Unleashing the Infinity Power of Geometry: A Novel Geometry-Aware Transformer (GOAT) for Whole Slide Histopathology Image Analysis**
<br/>
<button class="copy-to-clipboard" title="Unleashing the Infinity Power of Geometry: A Novel Geometry-Aware Transformer (GOAT) for Whole Slide Histopathology Image Analysis" index=9>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-9 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05373v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05373v1.pdf" filename="2402.05373v1.pdf">Download PDF</button>

---


**ABSTRACT**  
The histopathology analysis is of great significance for the diagnosis and prognosis of cancers, however, it has great challenges due to the enormous heterogeneity of gigapixel whole slide images (WSIs) and the intricate representation of pathological features. However, recent methods have not adequately exploited geometrical representation in WSIs which is significant in disease diagnosis. Therefore, we proposed a novel weakly-supervised framework, Geometry-Aware Transformer (GOAT), in which we urge the model to pay attention to the geometric characteristics within the tumor microenvironment which often serve as potent indicators. In addition, a context-aware attention mechanism is designed to extract and enhance the morphological features within WSIs.

{{</citation>}}


## cs.CY (2)



### (109/125) Examining Gender and Racial Bias in Large Vision-Language Models Using a Novel Dataset of Parallel Images (Kathleen C. Fraser et al., 2024)

{{<citation>}}

Kathleen C. Fraser, Svetlana Kiritchenko. (2024)  
**Examining Gender and Racial Bias in Large Vision-Language Models Using a Novel Dataset of Parallel Images**
<br/>
<button class="copy-to-clipboard" title="Examining Gender and Racial Bias in Large Vision-Language Models Using a Novel Dataset of Parallel Images" index=10>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-10 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CV, cs-CY, cs.CY  
Keywords: AI, Bias, Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05779v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05779v1.pdf" filename="2402.05779v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Following on recent advances in large language models (LLMs) and subsequent chat models, a new wave of large vision-language models (LVLMs) has emerged. Such models can incorporate images as input in addition to text, and perform tasks such as visual question answering, image captioning, story generation, etc. Here, we examine potential gender and racial biases in such systems, based on the perceived characteristics of the people in the input images. To accomplish this, we present a new dataset PAIRS (PArallel Images for eveRyday Scenarios). The PAIRS dataset contains sets of AI-generated images of people, such that the images are highly similar in terms of background and visual content, but differ along the dimensions of gender (man, woman) and race (Black, white). By querying the LVLMs with such images, we observe significant differences in the responses according to the perceived gender or race of the person depicted.

{{</citation>}}


### (110/125) A Framework for Assessing Proportionate Intervention with Face Recognition Systems in Real-Life Scenarios (Pablo Negri et al., 2024)

{{<citation>}}

Pablo Negri, Isabelle Hupont, Emilia Gomez. (2024)  
**A Framework for Assessing Proportionate Intervention with Face Recognition Systems in Real-Life Scenarios**
<br/>
<button class="copy-to-clipboard" title="A Framework for Assessing Proportionate Intervention with Face Recognition Systems in Real-Life Scenarios" index=10>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-10 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05731v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05731v1.pdf" filename="2402.05731v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Face recognition (FR) has reached a high technical maturity. However, its use needs to be carefully assessed from an ethical perspective, especially in sensitive scenarios. This is precisely the focus of this paper: the use of FR for the identification of specific subjects in moderately to densely crowded spaces (e.g. public spaces, sports stadiums, train stations) and law enforcement scenarios. In particular, there is a need to consider the trade-off between the need to protect privacy and fundamental rights of citizens as well as their safety. Recent Artificial Intelligence (AI) policies, notably the European AI Act, propose that such FR interventions should be proportionate and deployed only when strictly necessary. Nevertheless, concrete guidelines on how to address the concept of proportional FR intervention are lacking to date. This paper proposes a framework to contribute to assessing whether an FR intervention is proportionate or not for a given context of use in the above mentioned scenarios. It also identifies the main quantitative and qualitative variables relevant to the FR intervention decision (e.g. number of people in the scene, level of harm that the person(s) in search could perpetrate, consequences to individual rights and freedoms) and propose a 2D graphical model making it possible to balance these variables in terms of ethical cost vs security gain. Finally, different FR scenarios inspired by real-world deployments validate the proposed model. The framework is conceived as a simple support tool for decision makers when confronted with the deployment of an FR system.

{{</citation>}}


## cs.GT (1)



### (111/125) When is Mean-Field Reinforcement Learning Tractable and Relevant? (Batuhan Yardim et al., 2024)

{{<citation>}}

Batuhan Yardim, Artur Goldman, Niao He. (2024)  
**When is Mean-Field Reinforcement Learning Tractable and Relevant?**
<br/>
<button class="copy-to-clipboard" title="When is Mean-Field Reinforcement Learning Tractable and Relevant?" index=11>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-11 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.GT  
Categories: cs-GT, cs-MA, cs.GT, math-OC  
Keywords: Reinforcement Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05757v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05757v1.pdf" filename="2402.05757v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Mean-field reinforcement learning has become a popular theoretical framework for efficiently approximating large-scale multi-agent reinforcement learning (MARL) problems exhibiting symmetry. However, questions remain regarding the applicability of mean-field approximations: in particular, their approximation accuracy of real-world systems and conditions under which they become computationally tractable. We establish explicit finite-agent bounds for how well the MFG solution approximates the true $N$-player game for two popular mean-field solution concepts. Furthermore, for the first time, we establish explicit lower bounds indicating that MFGs are poor or uninformative at approximating $N$-player games assuming only Lipschitz dynamics and rewards. Finally, we analyze the computational complexity of solving MFGs with only Lipschitz properties and prove that they are in the class of \textsc{PPAD}-complete problems conjectured to be intractable, similar to general sum $N$ player games. Our theoretical results underscore the limitations of MFGs and complement and justify existing work by proving difficulty in the absence of common theoretical assumptions.

{{</citation>}}


## cs.IR (1)



### (112/125) CounterCLR: Counterfactual Contrastive Learning with Non-random Missing Data in Recommendation (Jun Wang et al., 2024)

{{<citation>}}

Jun Wang, Haoxuan Li, Chi Zhang, Dongxu Liang, Enyun Yu, Wenwu Ou, Wenjia Wang. (2024)  
**CounterCLR: Counterfactual Contrastive Learning with Non-random Missing Data in Recommendation**
<br/>
<button class="copy-to-clipboard" title="CounterCLR: Counterfactual Contrastive Learning with Non-random Missing Data in Recommendation" index=12>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-12 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Contrastive Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05740v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05740v1.pdf" filename="2402.05740v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Recommender systems are designed to learn user preferences from observed feedback and comprise many fundamental tasks, such as rating prediction and post-click conversion rate (pCVR) prediction. However, the observed feedback usually suffer from two issues: selection bias and data sparsity, where biased and insufficient feedback seriously degrade the performance of recommender systems in terms of accuracy and ranking. Existing solutions for handling the issues, such as data imputation and inverse propensity score, are highly susceptible to additional trained imputation or propensity models. In this work, we propose a novel counterfactual contrastive learning framework for recommendation, named CounterCLR, to tackle the problem of non-random missing data by exploiting the advances in contrast learning. Specifically, the proposed CounterCLR employs a deep representation network, called CauNet, to infer non-random missing data in recommendations and perform user preference modeling by further introducing a self-supervised contrastive learning task. Our CounterCLR mitigates the selection bias problem without the need for additional models or estimators, while also enhancing the generalization ability in cases of sparse data. Experiments on real-world datasets demonstrate the effectiveness and superiority of our method.

{{</citation>}}


## cs.IT (1)



### (113/125) Physical Layer Security over Fluid Antenna Systems (Farshad Rostami Ghadi et al., 2024)

{{<citation>}}

Farshad Rostami Ghadi, Kai-Kit Wong, F. Javier Lopez-Martinez, Wee Kiat New, Hao Xu, Chan-Byoung Chae. (2024)  
**Physical Layer Security over Fluid Antenna Systems**
<br/>
<button class="copy-to-clipboard" title="Physical Layer Security over Fluid Antenna Systems" index=13>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-13 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: Security  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05722v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05722v1.pdf" filename="2402.05722v1.pdf">Download PDF</button>

---


**ABSTRACT**  
This paper investigates the performance of physical layer security (PLS) in fluid antenna-aided communication systems under arbitrary correlated fading channels. In particular, it is considered that a single fixed-antenna transmitter aims to send confidential information to a legitimate receiver equipped with a planar fluid antenna system (FAS), while an eavesdropper, also taking advantage of a planar FAS, attempts to decode the desired message. For this scenario, we first present analytical expressions of the equivalent channel distributions at the legitimate user and eavesdropper by using copula, so that the obtained analytical results are valid for any arbitrarily correlated fading distributions. Then, with the help of Gauss-Laguerre quadrature, we derive compact analytical expressions for the average secrecy capacity (ASC), the secrecy outage probability (SOP), and the secrecy energy efficiency (SEE) for the FAS wiretap channel. Moreover, for exemplary purposes, we also obtain the compact expression of ASC, SOP, and SEE by utilizing the Gaussian copula under correlated Rayleigh fading channels as a special case. Eventually, numerical results indicate that applying the fluid antenna with only one active port to PLS can guarantee more secure and reliable transmission, when compared to traditional antenna systems (TAS) exploiting maximal ratio combining (MRC).

{{</citation>}}


## cs.MA (1)



### (114/125) Offline Risk-sensitive RL with Partial Observability to Enhance Performance in Human-Robot Teaming (Giorgio Angelotti et al., 2024)

{{<citation>}}

Giorgio Angelotti, Caroline P. C. Chanel, Adam H. M. Pinto, Christophe Lounis, Corentin Chauffaut, Nicolas Drougard. (2024)  
**Offline Risk-sensitive RL with Partial Observability to Enhance Performance in Human-Robot Teaming**
<br/>
<button class="copy-to-clipboard" title="Offline Risk-sensitive RL with Partial Observability to Enhance Performance in Human-Robot Teaming" index=14>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-14 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.MA  
Categories: cs-AI, cs-HC, cs-LG, cs-MA, cs-RO, cs.MA  
Keywords: Reinforcement Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05703v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05703v1.pdf" filename="2402.05703v1.pdf">Download PDF</button>

---


**ABSTRACT**  
The integration of physiological computing into mixed-initiative human-robot interaction systems offers valuable advantages in autonomous task allocation by incorporating real-time features as human state observations into the decision-making system. This approach may alleviate the cognitive load on human operators by intelligently allocating mission tasks between agents. Nevertheless, accommodating a diverse pool of human participants with varying physiological and behavioral measurements presents a substantial challenge. To address this, resorting to a probabilistic framework becomes necessary, given the inherent uncertainty and partial observability on the human's state. Recent research suggests to learn a Partially Observable Markov Decision Process (POMDP) model from a data set of previously collected experiences that can be solved using Offline Reinforcement Learning (ORL) methods. In the present work, we not only highlight the potential of partially observable representations and physiological measurements to improve human operator state estimation and performance, but also enhance the overall mission effectiveness of a human-robot team. Importantly, as the fixed data set may not contain enough information to fully represent complex stochastic processes, we propose a method to incorporate model uncertainty, thus enabling risk-sensitive sequential decision-making. Experiments were conducted with a group of twenty-six human participants within a simulated robot teleoperation environment, yielding empirical evidence of the method's efficacy. The obtained adaptive task allocation policy led to statistically significant higher scores than the one that was used to collect the data set, allowing for generalization across diverse participants also taking into account risk-sensitive metrics.

{{</citation>}}


## cs.CR (2)



### (115/125) Comprehensive Assessment of Jailbreak Attacks Against LLMs (Junjie Chu et al., 2024)

{{<citation>}}

Junjie Chu, Yugeng Liu, Ziqing Yang, Xinyue Shen, Michael Backes, Yang Zhang. (2024)  
**Comprehensive Assessment of Jailbreak Attacks Against LLMs**
<br/>
<button class="copy-to-clipboard" title="Comprehensive Assessment of Jailbreak Attacks Against LLMs" index=15>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-15 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.CR  
Keywords: GLM, GPT, GPT-3.5, Language Model, PaLM  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05668v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05668v1.pdf" filename="2402.05668v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Misuse of the Large Language Models (LLMs) has raised widespread concern. To address this issue, safeguards have been taken to ensure that LLMs align with social ethics. However, recent findings have revealed an unsettling vulnerability bypassing the safeguards of LLMs, known as jailbreak attacks. By applying techniques, such as employing role-playing scenarios, adversarial examples, or subtle subversion of safety objectives as a prompt, LLMs can produce an inappropriate or even harmful response. While researchers have studied several categories of jailbreak attacks, they have done so in isolation. To fill this gap, we present the first large-scale measurement of various jailbreak attack methods. We concentrate on 13 cutting-edge jailbreak methods from four categories, 160 questions from 16 violation categories, and six popular LLMs. Our extensive experimental results demonstrate that the optimized jailbreak prompts consistently achieve the highest attack success rates, as well as exhibit robustness across different LLMs. Some jailbreak prompt datasets, available from the Internet, can also achieve high attack success rates on many LLMs, such as ChatGLM3, GPT-3.5, and PaLM2. Despite the claims from many organizations regarding the coverage of violation categories in their policies, the attack success rates from these categories remain high, indicating the challenges of effectively aligning LLM policies and the ability to counter jailbreak attacks. We also discuss the trade-off between the attack performance and efficiency, as well as show that the transferability of the jailbreak prompts is still viable, becoming an option for black-box models. Overall, our research highlights the necessity of evaluating different jailbreak methods. We hope our study can provide insights for future research on jailbreak attacks and serve as a benchmark tool for evaluating them for practitioners.

{{</citation>}}


### (116/125) Domain-Agnostic Hardware Fingerprinting-Based Device Identifier for Zero-Trust IoT Security (Abdurrahman Elmaghbub et al., 2024)

{{<citation>}}

Abdurrahman Elmaghbub, Bechir Hamdaoui. (2024)  
**Domain-Agnostic Hardware Fingerprinting-Based Device Identifier for Zero-Trust IoT Security**
<br/>
<button class="copy-to-clipboard" title="Domain-Agnostic Hardware Fingerprinting-Based Device Identifier for Zero-Trust IoT Security" index=15>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-15 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR, eess-SP  
Keywords: Security  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05332v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05332v1.pdf" filename="2402.05332v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Next-generation networks aim for comprehensive connectivity, interconnecting humans, machines, devices, and systems seamlessly. This interconnectivity raises concerns about privacy and security, given the potential network-wide impact of a single compromise. To address this challenge, the Zero Trust (ZT) paradigm emerges as a key method for safeguarding network integrity and data confidentiality. This work introduces EPS-CNN, a novel deep-learning-based wireless device identification framework designed to serve as the device authentication layer within the ZT architecture, with a focus on resource-constrained IoT devices. At the core of EPS-CNN, a Convolutional Neural Network (CNN) is utilized to generate the device identity from a unique RF signal representation, known as the Double-Sided Envelope Power Spectrum (EPS), which effectively captures the device-specific hardware characteristics while ignoring device-unrelated information. Experimental evaluations show that the proposed framework achieves over 99%, 93%, and 95% testing accuracy when tested in same-domain (day, location, and channel), cross-day, and cross-location scenarios, respectively. Our findings demonstrate the superiority of the proposed framework in enhancing the accuracy, robustness, and adaptability of deep learning-based methods, thus offering a pioneering solution for enabling ZT IoT device identification.

{{</citation>}}


## cs.SE (4)



### (117/125) Rocks Coding, Not Development--A Human-Centric, Experimental Evaluation of LLM-Supported SE Tasks (Wei Wang et al., 2024)

{{<citation>}}

Wei Wang, Huilong Ning, Gaowei Zhang, Libo Liu, Yi Wang. (2024)  
**Rocks Coding, Not Development--A Human-Centric, Experimental Evaluation of LLM-Supported SE Tasks**
<br/>
<button class="copy-to-clipboard" title="Rocks Coding, Not Development--A Human-Centric, Experimental Evaluation of LLM-Supported SE Tasks" index=16>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-16 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05650v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05650v1.pdf" filename="2402.05650v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Recently, large language models (LLM) based generative AI has been gaining momentum for their impressive high-quality performances in multiple domains, particularly after the release of the ChatGPT. Many believe that they have the potential to perform general-purpose problem-solving in software development and replace human software developers. Nevertheless, there are in a lack of serious investigation into the capability of these LLM techniques in fulfilling software development tasks. In a controlled 2 $\times$ 2 between-subject experiment with 109 participants, we examined whether and to what degree working with ChatGPT was helpful in the coding task and typical software development task and how people work with ChatGPT. We found that while ChatGPT performed well in solving simple coding problems, its performance in supporting typical software development tasks was not that good. We also observed the interactions between participants and ChatGPT and found the relations between the interactions and the outcomes. Our study thus provides first-hand insights into using ChatGPT to fulfill software engineering tasks with real-world developers and motivates the need for novel interaction mechanisms that help developers effectively work with large language models to achieve desired outcomes.

{{</citation>}}


### (118/125) The Impact of AI Tool on Engineering at ANZ Bank An Emperical Study on GitHub Copilot within Coporate Environment (Sayan Chatterjee et al., 2024)

{{<citation>}}

Sayan Chatterjee, Ching Louis Liu, Gareth Rowland, Tim Hogarth. (2024)  
**The Impact of AI Tool on Engineering at ANZ Bank An Emperical Study on GitHub Copilot within Coporate Environment**
<br/>
<button class="copy-to-clipboard" title="The Impact of AI Tool on Engineering at ANZ Bank An Emperical Study on GitHub Copilot within Coporate Environment" index=16>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-16 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI, Language Model  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05636v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05636v1.pdf" filename="2402.05636v1.pdf">Download PDF</button>

---


**ABSTRACT**  
The increasing popularity of AI, particularly Large Language Models (LLMs), has significantly impacted various domains, including Software Engineering. This study explores the integration of AI tools in software engineering practices within a large organization. We focus on ANZ Bank, which employs over 5000 engineers covering all aspects of the software development life cycle. This paper details an experiment conducted using GitHub Copilot, a notable AI tool, within a controlled environment to evaluate its effectiveness in real-world engineering tasks. Additionally, this paper shares initial findings on the productivity improvements observed after GitHub Copilot was adopted on a large scale, with about 1000 engineers using it. ANZ Bank's six-week experiment with GitHub Copilot included two weeks of preparation and four weeks of active testing. The study evaluated participant sentiment and the tool's impact on productivity, code quality, and security. Initially, participants used GitHub Copilot for proposed use-cases, with their feedback gathered through regular surveys. In the second phase, they were divided into Control and Copilot groups, each tackling the same Python challenges, and their experiences were again surveyed. Results showed a notable boost in productivity and code quality with GitHub Copilot, though its impact on code security remained inconclusive. Participant responses were overall positive, confirming GitHub Copilot's effectiveness in large-scale software engineering environments. Early data from 1000 engineers also indicated a significant increase in productivity and job satisfaction.

{{</citation>}}


### (119/125) Leveraging AI for Enhanced Software Effort Estimation: A Comprehensive Study and Framework Proposal (Nhi Tran et al., 2024)

{{<citation>}}

Nhi Tran, Tan Tran, Nam Nguyen. (2024)  
**Leveraging AI for Enhanced Software Effort Estimation: A Comprehensive Study and Framework Proposal**
<br/>
<button class="copy-to-clipboard" title="Leveraging AI for Enhanced Software Effort Estimation: A Comprehensive Study and Framework Proposal" index=16>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-16 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05484v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05484v1.pdf" filename="2402.05484v1.pdf">Download PDF</button>

---


**ABSTRACT**  
This paper presents an extensive study on the application of AI techniques for software effort estimation in the past five years from 2017 to 2023. By overcoming the limitations of traditional methods, the study aims to improve accuracy and reliability. Through performance evaluation and comparison with diverse Machine Learning models, including Artificial Neural Network (ANN), Support Vector Machine (SVM), Linear Regression, Random Forest and other techniques, the most effective method is identified. The proposed AI-based framework holds the potential to enhance project planning and resource allocation, contributing to the research area of software project effort estimation.

{{</citation>}}


### (120/125) POLARIS: A framework to guide the development of Trustworthy AI systems (Maria Teresa Baldassarre et al., 2024)

{{<citation>}}

Maria Teresa Baldassarre, Domenico Gigante, Marcos Kalinowski, Azzurra Ragone. (2024)  
**POLARIS: A framework to guide the development of Trustworthy AI systems**
<br/>
<button class="copy-to-clipboard" title="POLARIS: A framework to guide the development of Trustworthy AI systems" index=16>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-16 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05340v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05340v1.pdf" filename="2402.05340v1.pdf">Download PDF</button>

---


**ABSTRACT**  
In the ever-expanding landscape of Artificial Intelligence (AI), where innovation thrives and new products and services are continuously being delivered, ensuring that AI systems are designed and developed responsibly throughout their entire lifecycle is crucial. To this end, several AI ethics principles and guidelines have been issued to which AI systems should conform. Nevertheless, relying solely on high-level AI ethics principles is far from sufficient to ensure the responsible engineering of AI systems. In this field, AI professionals often navigate by sight. Indeed, while recommendations promoting Trustworthy AI (TAI) exist, these are often high-level statements that are difficult to translate into concrete implementation strategies. There is a significant gap between high-level AI ethics principles and low-level concrete practices for AI professionals. To address this challenge, our work presents an experience report where we develop a novel holistic framework for Trustworthy AI - designed to bridge the gap between theory and practice - and report insights from its application in an industrial case study. The framework is built on the result of a systematic review of the state of the practice, a survey, and think-aloud interviews with 34 AI practitioners. The framework, unlike most of those already in the literature, is designed to provide actionable guidelines and tools to support different types of stakeholders throughout the entire Software Development Life Cycle (SDLC). Our goal is to empower AI professionals to confidently navigate the ethical dimensions of TAI through practical insights, ensuring that the vast potential of AI is exploited responsibly for the benefit of society as a whole.

{{</citation>}}


## cs.DL (1)



### (121/125) Can ChatGPT evaluate research quality? (Mike Thelwall, 2024)

{{<citation>}}

Mike Thelwall. (2024)  
**Can ChatGPT evaluate research quality?**
<br/>
<button class="copy-to-clipboard" title="Can ChatGPT evaluate research quality?" index=17>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-17 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.DL  
Categories: cs-AI, cs-DL, cs.DL  
Keywords: ChatGPT, GPT, GPT-4  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05519v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05519v1.pdf" filename="2402.05519v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Purpose: Assess whether ChatGPT 4.0 is accurate enough to perform research evaluations on journal articles to automate this time-consuming task. Design/methodology/approach: Test the extent to which ChatGPT-4 can assess the quality of journal articles using a case study of the published scoring guidelines of the UK Research Excellence Framework (REF) 2021 to create a research evaluation ChatGPT. This was applied to 51 of my own articles and compared against my own quality judgements. Findings: ChatGPT-4 can produce plausible document summaries and quality evaluation rationales that match the REF criteria. Its overall scores have weak correlations with my self-evaluation scores of the same documents (averaging r=0.281 over 15 iterations, with 8 being statistically significantly different from 0). In contrast, the average scores from the 15 iterations produced a statistically significant positive correlation of 0.509. Thus, averaging scores from multiple ChatGPT-4 rounds seems more effective than individual scores. The positive correlation may be due to ChatGPT being able to extract the author's significance, rigour, and originality claims from inside each paper. If my weakest articles are removed, then the correlation with average scores (r=0.200) falls below statistical significance, suggesting that ChatGPT struggles to make fine-grained evaluations. Research limitations: The data is self-evaluations of a convenience sample of articles from one academic in one field. Practical implications: Overall, ChatGPT does not yet seem to be accurate enough to be trusted for any formal or informal research quality evaluation tasks. Research evaluators, including journal editors, should therefore take steps to control its use. Originality/value: This is the first published attempt at post-publication expert review accuracy testing for ChatGPT.

{{</citation>}}


## eess.SP (2)



### (122/125) A Non-Intrusive Neural Quality Assessment Model for Surface Electromyography Signals (Cho-Yuan Lee et al., 2024)

{{<citation>}}

Cho-Yuan Lee, Kuan-Chen Wang, Kai-Chun Liu, Xugang Lu, Ping-Chen Yeh, Yu Tsao. (2024)  
**A Non-Intrusive Neural Quality Assessment Model for Surface Electromyography Signals**
<br/>
<button class="copy-to-clipboard" title="A Non-Intrusive Neural Quality Assessment Model for Surface Electromyography Signals" index=18>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-18 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP  
Keywords: LSTM, QA  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05482v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05482v1.pdf" filename="2402.05482v1.pdf">Download PDF</button>

---


**ABSTRACT**  
In practical scenarios involving the measurement of surface electromyography (sEMG) in muscles, particularly those areas near the heart, one of the primary sources of contamination is the presence of electrocardiogram (ECG) signals. To assess the quality of real-world sEMG data more effectively, this study proposes QASE-net, a new non-intrusive model that predicts the SNR of sEMG signals. QASE-net combines CNN-BLSTM with attention mechanisms and follows an end-to-end training strategy. Our experimental framework utilizes real-world sEMG and ECG data from two open-access databases, the Non-Invasive Adaptive Prosthetics Database and the MIT-BIH Normal Sinus Rhythm Database, respectively. The experimental results demonstrate the superiority of QASE-net over the previous assessment model, exhibiting significantly reduced prediction errors and notably higher linear correlations with the ground truth. These findings show the potential of QASE-net to substantially enhance the reliability and precision of sEMG quality assessment in practical applications.

{{</citation>}}


### (123/125) Graph Neural Networks for Physical-Layer Security in Multi-User Flexible-Duplex Networks (Tharaka Perera et al., 2024)

{{<citation>}}

Tharaka Perera, Saman Atapattu, Yuting Fang, Jamie Evans. (2024)  
**Graph Neural Networks for Physical-Layer Security in Multi-User Flexible-Duplex Networks**
<br/>
<button class="copy-to-clipboard" title="Graph Neural Networks for Physical-Layer Security in Multi-User Flexible-Duplex Networks" index=18>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-18 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: eess.SP  
Categories: cs-AI, cs-CR, cs-LG, eess-SP, eess.SP  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Security  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05378v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05378v1.pdf" filename="2402.05378v1.pdf">Download PDF</button>

---


**ABSTRACT**  
This paper explores Physical-Layer Security (PLS) in Flexible Duplex (FlexD) networks, considering scenarios involving eavesdroppers. Our investigation revolves around the intricacies of the sum secrecy rate maximization problem, particularly when faced with coordinated and distributed eavesdroppers employing a Minimum Mean Square Error (MMSE) receiver. Our contributions include an iterative classical optimization solution and an unsupervised learning strategy based on Graph Neural Networks (GNNs). To the best of our knowledge, this work marks the initial exploration of GNNs for PLS applications. Additionally, we extend the GNN approach to address the absence of eavesdroppers' channel knowledge. Extensive numerical simulations highlight FlexD's superiority over Half-Duplex (HD) communications and the GNN approach's superiority over the classical method in both performance and time complexity.

{{</citation>}}


## eess.SY (1)



### (124/125) Multi-Network Constrained Operational Optimization in Community Integrated Energy Systems: A Safe Reinforcement Learning Approach (Ze Hu et al., 2024)

{{<citation>}}

Ze Hu, Ka Wing Chan, Ziqing Zhu, Xiang Wei, Siqi Bu. (2024)  
**Multi-Network Constrained Operational Optimization in Community Integrated Energy Systems: A Safe Reinforcement Learning Approach**
<br/>
<button class="copy-to-clipboard" title="Multi-Network Constrained Operational Optimization in Community Integrated Energy Systems: A Safe Reinforcement Learning Approach" index=19>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-19 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05412v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05412v1.pdf" filename="2402.05412v1.pdf">Download PDF</button>

---


**ABSTRACT**  
The integrated community energy system (ICES) has emerged as a promising solution for enhancing the efficiency of the distribution system by effectively coordinating multiple energy sources. However, the operational optimization of ICES is hindered by the physical constraints of heterogeneous networks including electricity, natural gas, and heat. These challenges are difficult to address due to the non-linearity of network constraints and the high complexity of multi-network coordination. This paper, therefore, proposes a novel Safe Reinforcement Learning (SRL) algorithm to optimize the multi-network constrained operation problem of ICES. Firstly, a comprehensive ICES model is established considering integrated demand response (IDR), multiple energy devices, and network constraints. The multi-network operational optimization problem of ICES is then presented and reformulated as a constrained Markov Decision Process (C-MDP) accounting for violating physical network constraints. The proposed novel SRL algorithm, named Primal-Dual Twin Delayed Deep Deterministic Policy Gradient (PD-TD3), solves the C-MDP by employing a Lagrangian multiplier to penalize the multi-network constraint violation, ensuring that violations are within a tolerated range and avoid over-conservative strategy with a low reward at the same time. The proposed algorithm accurately estimates the cumulative reward and cost of the training process, thus achieving a fair balance between improving profits and reducing constraint violations in a privacy-protected environment with only partial information. A case study comparing the proposed algorithm with benchmark RL algorithms demonstrates the computational performance in increasing total profits and alleviating the network constraint violations.

{{</citation>}}


## physics.flu-dyn (1)



### (125/125) Reduced-order modeling of unsteady fluid flow using neural network ensembles (Rakesh Halder et al., 2024)

{{<citation>}}

Rakesh Halder, Mohammadmehdi Ataei, Hesam Salehipour, Krzysztof Fidkowski, Kevin Maki. (2024)  
**Reduced-order modeling of unsteady fluid flow using neural network ensembles**
<br/>
<button class="copy-to-clipboard" title="Reduced-order modeling of unsteady fluid flow using neural network ensembles" index=20>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-20 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: physics.flu-dyn  
Categories: cs-LG, physics-flu-dyn, physics.flu-dyn  
Keywords: LSTM  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2402.05372v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="http://arxiv.org/pdf/2402.05372v1.pdf" filename="2402.05372v1.pdf">Download PDF</button>

---


**ABSTRACT**  
The use of deep learning has become increasingly popular in reduced-order models (ROMs) to obtain low-dimensional representations of full-order models. Convolutional autoencoders (CAEs) are often used to this end as they are adept at handling data that are spatially distributed, including solutions to partial differential equations. When applied to unsteady physics problems, ROMs also require a model for time-series prediction of the low-dimensional latent variables. Long short-term memory (LSTM) networks, a type of recurrent neural network useful for modeling sequential data, are frequently employed in data-driven ROMs for autoregressive time-series prediction. When making predictions at unseen design points over long time horizons, error propagation is a frequently encountered issue, where errors made early on can compound over time and lead to large inaccuracies. In this work, we propose using bagging, a commonly used ensemble learning technique, to develop a fully data-driven ROM framework referred to as the CAE-eLSTM ROM that uses CAEs for spatial reconstruction of the full-order model and LSTM ensembles for time-series prediction. When applied to two unsteady fluid dynamics problems, our results show that the presented framework effectively reduces error propagation and leads to more accurate time-series prediction of latent variables at unseen points.

{{</citation>}}
