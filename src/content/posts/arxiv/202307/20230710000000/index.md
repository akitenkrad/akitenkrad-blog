---
draft: false
title: "arXiv @ 2023.07.10"
date: 2023-07-10
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.10"
    identifier: arxiv_20230710
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (9)](#cscl-9)
- [cs.CV (10)](#cscv-10)
- [cs.SI (1)](#cssi-1)
- [stat.ML (1)](#statml-1)
- [cs.AI (4)](#csai-4)
- [cs.RO (4)](#csro-4)
- [cs.HC (2)](#cshc-2)
- [cs.SD (1)](#cssd-1)
- [physics.flu-dyn (1)](#physicsflu-dyn-1)
- [cs.SE (1)](#csse-1)
- [cs.LG (4)](#cslg-4)
- [cs.NI (1)](#csni-1)
- [cs.CY (1)](#cscy-1)
- [cs.AR (1)](#csar-1)
- [cs.CR (1)](#cscr-1)
- [eess.AS (1)](#eessas-1)
- [math.NA (1)](#mathna-1)
- [quant-ph (1)](#quant-ph-1)
- [cs.IR (1)](#csir-1)
- [cs.IT (1)](#csit-1)

## cs.CL (9)



### (1/47) Bidirectional Attention as a Mixture of Continuous Word Experts (Kevin Christian Wibisono et al., 2023)

{{<citation>}}

Kevin Christian Wibisono, Yixin Wang. (2023)  
**Bidirectional Attention as a Mixture of Continuous Word Experts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL, stat-ML  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.04057v1)  

---


**ABSTRACT**  
Bidirectional attention $\unicode{x2013}$ composed of self-attention with positional encodings and the masked language model (MLM) objective $\unicode{x2013}$ has emerged as a key component of modern large language models (LLMs). Despite its empirical success, few studies have examined its statistical underpinnings: What statistical model is bidirectional attention implicitly fitting? What sets it apart from its non-attention predecessors? We explore these questions in this paper. The key observation is that fitting a single-layer single-head bidirectional attention, upon reparameterization, is equivalent to fitting a continuous bag of words (CBOW) model with mixture-of-experts (MoE) weights. Further, bidirectional attention with multiple heads and multiple layers is equivalent to stacked MoEs and a mixture of MoEs, respectively. This statistical viewpoint reveals the distinct use of MoE in bidirectional attention, which aligns with its practical effectiveness in handling heterogeneous data. It also suggests an immediate extension to categorical tabular data, if we view each word location in a sentence as a tabular feature. Across empirical studies, we find that this extension outperforms existing tabular extensions of transformers in out-of-distribution (OOD) generalization. Finally, this statistical perspective of bidirectional attention enables us to theoretically characterize when linear word analogies are present in its word embeddings. These analyses show that bidirectional attention can require much stronger assumptions to exhibit linear word analogies than its non-attention predecessors.

{{</citation>}}


### (2/47) How is Fatherhood Framed Online in Singapore? (Tran Hien Van et al., 2023)

{{<citation>}}

Tran Hien Van, Abhay Goyal, Muhammad Siddique, Lam Yin Cheung, Nimay Parekh, Jonathan Y Huang, Keri McCrickerd, Edson C Tandoc Jr., Gerard Chung, Navin Kumar. (2023)  
**How is Fatherhood Framed Online in Singapore?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Twitter  
[Paper Link](http://arxiv.org/abs/2307.04053v1)  

---


**ABSTRACT**  
The proliferation of discussion about fatherhood in Singapore attests to its significance, indicating the need for an exploration of how fatherhood is framed, aiding policy-making around fatherhood in Singapore. Sound and holistic policy around fatherhood in Singapore may reduce stigma and apprehension around being a parent, critical to improving the nations flagging birth rate. We analyzed 15,705 articles and 56,221 posts to study how fatherhood is framed in Singapore across a range of online platforms (news outlets, parenting forums, Twitter). We used NLP techniques to understand these differences. While fatherhood was framed in a range of ways on the Singaporean online environment, it did not seem that fathers were framed as central to the Singaporean family unit. A strength of our work is how the different techniques we have applied validate each other.

{{</citation>}}


### (3/47) Can LLMs be Good Financial Advisors?: An Initial Study in Personal Decision Making for Optimized Outcomes (Kausik Lakkaraju et al., 2023)

{{<citation>}}

Kausik Lakkaraju, Sai Krishna Revanth Vuruma, Vishal Pallagani, Bharath Muppasani, Biplav Srivastava. (2023)  
**Can LLMs be Good Financial Advisors?: An Initial Study in Personal Decision Making for Optimized Outcomes**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Financial, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.07422v1)  

---


**ABSTRACT**  
Increasingly powerful Large Language Model (LLM) based chatbots, like ChatGPT and Bard, are becoming available to users that have the potential to revolutionize the quality of decision-making achieved by the public. In this context, we set out to investigate how such systems perform in the personal finance domain, where financial inclusion has been an overarching stated aim of banks for decades. We asked 13 questions representing banking products in personal finance: bank account, credit card, and certificate of deposits and their inter-product interactions, and decisions related to high-value purchases, payment of bank dues, and investment advice, and in different dialects and languages (English, African American Vernacular English, and Telugu). We find that although the outputs of the chatbots are fluent and plausible, there are still critical gaps in providing accurate and reliable financial information using LLM-based chatbots.

{{</citation>}}


### (4/47) Revisiting Cross-Lingual Summarization: A Corpus-based Study and A New Benchmark with Improved Annotation (Yulong Chen et al., 2023)

{{<citation>}}

Yulong Chen, Huajian Zhang, Yijie Zhou, Xuefeng Bai, Yueguan Wang, Ming Zhong, Jianhao Yan, Yafu Li, Judy Li, Michael Zhu, Yue Zhang. (2023)  
**Revisiting Cross-Lingual Summarization: A Corpus-based Study and A New Benchmark with Improved Annotation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2307.04018v1)  

---


**ABSTRACT**  
Most existing cross-lingual summarization (CLS) work constructs CLS corpora by simply and directly translating pre-annotated summaries from one language to another, which can contain errors from both summarization and translation processes. To address this issue, we propose ConvSumX, a cross-lingual conversation summarization benchmark, through a new annotation schema that explicitly considers source input context. ConvSumX consists of 2 sub-tasks under different real-world scenarios, with each covering 3 language directions. We conduct thorough analysis on ConvSumX and 3 widely-used manually annotated CLS corpora and empirically find that ConvSumX is more faithful towards input text. Additionally, based on the same intuition, we propose a 2-Step method, which takes both conversation and summary as input to simulate human annotation process. Experimental results show that 2-Step method surpasses strong baselines on ConvSumX under both automatic and human evaluation. Analysis shows that both source input text and summary are crucial for modeling cross-lingual summaries.

{{</citation>}}


### (5/47) Advancements in Scientific Controllable Text Generation Methods (Arnav Goel et al., 2023)

{{<citation>}}

Arnav Goel, Medha Hira, Avinash Anand, Siddhesh Bangar, Dr. Rajiv Ratn Shah. (2023)  
**Advancements in Scientific Controllable Text Generation Methods**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2307.05538v1)  

---


**ABSTRACT**  
The previous work on controllable text generation is organized using a new schema we provide in this study. Seven components make up the schema, and each one is crucial to the creation process. To accomplish controlled generation for scientific literature, we describe the various modulation strategies utilised to modulate each of the seven components. We also offer a theoretical study and qualitative examination of these methods. This insight makes possible new architectures based on combinations of these components. Future research will compare these methods empirically to learn more about their strengths and utility.

{{</citation>}}


### (6/47) A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-Confidence Generation (Neeraj Varshney et al., 2023)

{{<citation>}}

Neeraj Varshney, Wenlin Yao, Hongming Zhang, Jianshu Chen, Dong Yu. (2023)  
**A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-Confidence Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2307.03987v1)  

---


**ABSTRACT**  
Recently developed large language models have achieved remarkable success in generating fluent and coherent text. However, these models often tend to 'hallucinate' which critically hampers their reliability. In this work, we address this crucial problem and propose an approach that actively detects and mitigates hallucinations during the generation process. Specifically, we first identify the candidates of potential hallucination leveraging the model's logit output values, check their correctness through a validation procedure, mitigate the detected hallucinations, and then continue with the generation process. Through extensive experiments with the 'article generation task', we first demonstrate the individual efficacy of our detection and mitigation techniques. Specifically, the detection technique achieves a recall of 88% and the mitigation technique successfully mitigates 57.6% of the correctly detected hallucinations. Importantly, our mitigation technique does not introduce new hallucinations even in the case of incorrectly detected hallucinations, i.e., false positives. Then, we show that the proposed active detection and mitigation approach successfully reduces the hallucinations of the GPT-3 model from 47.5% to 14.5% on average. In summary, our work contributes to improving the reliability and trustworthiness of large language models, a crucial step en route to enabling their widespread adoption in real-world applications.

{{</citation>}}


### (7/47) Evaluating the Capability of Large-scale Language Models on Chinese Grammatical Error Correction Task (Fanyi Qu et al., 2023)

{{<citation>}}

Fanyi Qu, Yunfang Wu. (2023)  
**Evaluating the Capability of Large-scale Language Models on Chinese Grammatical Error Correction Task**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.03972v1)  

---


**ABSTRACT**  
Large-scale language models (LLMs) has shown remarkable capability in various of Natural Language Processing (NLP) tasks and attracted lots of attention recently. However, some studies indicated that large language models fail to achieve promising result beyond the state-of-the-art models in English grammatical error correction (GEC) tasks. In this report, we aim to explore the how large language models perform on Chinese grammatical error correction tasks and provide guidance for future work. We conduct experiments with 3 different LLMs of different model scale on 4 Chinese GEC dataset. Our experimental results indicate that the performances of LLMs on automatic evaluation metrics falls short of the previous sota models because of the problem of over-correction. Furthermore, we also discover notable variations in the performance of LLMs when evaluated on different data distributions. Our findings demonstrates that further investigation is required for the application of LLMs on Chinese GEC task.

{{</citation>}}


### (8/47) Is ChatGPT a Good Personality Recognizer? A Preliminary Study (Yu Ji et al., 2023)

{{<citation>}}

Yu Ji, Wen Wu, Hong Zheng, Yi Hu, Xi Chen, Liang He. (2023)  
**Is ChatGPT a Good Personality Recognizer? A Preliminary Study**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.03952v1)  

---


**ABSTRACT**  
In recent years, personality has been regarded as a valuable personal factor being incorporated into numerous tasks such as sentiment analysis and product recommendation. This has led to widespread attention to text-based personality recognition task, which aims to identify an individual's personality based on given text. Considering that ChatGPT has recently exhibited remarkable abilities on various natural language processing tasks, we provide a preliminary evaluation of ChatGPT on text-based personality recognition task for generating effective personality data. Concretely, we employ a variety of prompting strategies to explore ChatGPT's ability in recognizing personality from given text, especially the level-oriented prompting strategy we designed for guiding ChatGPT in analyzing given text at a specified level. We compare the performance of ChatGPT on two representative real-world datasets with traditional neural network, fine-tuned RoBERTa, and corresponding state-of-the-art task-specific model. The experimental results show that ChatGPT with zero-shot chain-of-thought prompting exhibits impressive personality recognition ability. Triggered by zero-shot chain-of-thought prompting, ChatGPT outperforms fine-tuned RoBERTa on the two datasets and is capable to provide natural language explanations through text-based logical reasoning. Furthermore, relative to zero-shot chain-of-thought prompting, zero-shot level-oriented chain-of-thought prompting enhances the personality prediction ability of ChatGPT and reduces the performance gap between ChatGPT and corresponding state-of-the-art task-specific model. Besides, we also conduct experiments to observe the fairness of ChatGPT when identifying personality and discover that ChatGPT shows unfairness to some sensitive demographic attributes such as gender and age.

{{</citation>}}


### (9/47) Opening up ChatGPT: Tracking openness, transparency, and accountability in instruction-tuned text generators (Andreas Liesenfeld et al., 2023)

{{<citation>}}

Andreas Liesenfeld, Alianda Lopez, Mark Dingemanse. (2023)  
**Opening up ChatGPT: Tracking openness, transparency, and accountability in instruction-tuned text generators**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.05532v1)  

---


**ABSTRACT**  
Large language models that exhibit instruction-following behaviour represent one of the biggest recent upheavals in conversational interfaces, a trend in large part fuelled by the release of OpenAI's ChatGPT, a proprietary large language model for text generation fine-tuned through reinforcement learning from human feedback (LLM+RLHF). We review the risks of relying on proprietary software and survey the first crop of open-source projects of comparable architecture and functionality. The main contribution of this paper is to show that openness is differentiated, and to offer scientific documentation of degrees of openness in this fast-moving field. We evaluate projects in terms of openness of code, training data, model weights, RLHF data, licensing, scientific documentation, and access methods. We find that while there is a fast-growing list of projects billing themselves as 'open source', many inherit undocumented data of dubious legality, few share the all-important instruction-tuning (a key site where human annotation labour is involved), and careful scientific documentation is exceedingly rare. Degrees of openness are relevant to fairness and accountability at all points, from data collection and curation to model architecture, and from training and fine-tuning to release and deployment.

{{</citation>}}


## cs.CV (10)



### (10/47) Deep Unsupervised Learning Using Spike-Timing-Dependent Plasticity (Sen Lu et al., 2023)

{{<citation>}}

Sen Lu, Abhronil Sengupta. (2023)  
**Deep Unsupervised Learning Using Spike-Timing-Dependent Plasticity**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.04054v1)  

---


**ABSTRACT**  
Spike-Timing-Dependent Plasticity (STDP) is an unsupervised learning mechanism for Spiking Neural Networks (SNNs) that has received significant attention from the neuromorphic hardware community. However, scaling such local learning techniques to deeper networks and large-scale tasks has remained elusive. In this work, we investigate a Deep-STDP framework where a convolutional network is trained in tandem with pseudo-labels generated by the STDP clustering process on the network outputs. We achieve $24.56\%$ higher accuracy and $3.5\times$ faster convergence speed at iso-accuracy on a 10-class subset of the Tiny ImageNet dataset in contrast to a $k$-means clustering approach.

{{</citation>}}


### (11/47) Measuring the Success of Diffusion Models at Imitating Human Artists (Stephen Casper et al., 2023)

{{<citation>}}

Stephen Casper, Zifan Guo, Shreya Mogulothu, Zachary Marinov, Chinmay Deshpande, Rui-Jie Yew, Zheng Dai, Dylan Hadfield-Menell. (2023)  
**Measuring the Success of Diffusion Models at Imitating Human Artists**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.04028v1)  

---


**ABSTRACT**  
Modern diffusion models have set the state-of-the-art in AI image generation. Their success is due, in part, to training on Internet-scale data which often includes copyrighted work. This prompts questions about the extent to which these models learn from, imitate, or copy the work of human artists. This work suggests that tying copyright liability to the capabilities of the model may be useful given the evolving ecosystem of generative models. Specifically, much of the legal analysis of copyright and generative systems focuses on the use of protected data for training. As a result, the connections between data, training, and the system are often obscured. In our approach, we consider simple image classification techniques to measure a model's ability to imitate specific artists. Specifically, we use Contrastive Language-Image Pretrained (CLIP) encoders to classify images in a zero-shot fashion. Our process first prompts a model to imitate a specific artist. Then, we test whether CLIP can be used to reclassify the artist (or the artist's work) from the imitation. If these tests match the imitation back to the original artist, this suggests the model can imitate that artist's expression. Our approach is simple and quantitative. Furthermore, it uses standard techniques and does not require additional training. We demonstrate our approach with an audit of Stable Diffusion's capacity to imitate 70 professional digital artists with copyrighted work online. When Stable Diffusion is prompted to imitate an artist from this set, we find that the artist can be identified from the imitation with an average accuracy of 81.0%. Finally, we also show that a sample of the artist's work can be matched to these imitation images with a high degree of statistical reliability. Overall, these results suggest that Stable Diffusion is broadly successful at imitating individual human artists.

{{</citation>}}


### (12/47) Stimulating the Diffusion Model for Image Denoising via Adaptive Embedding and Ensembling (Tong Li et al., 2023)

{{<citation>}}

Tong Li, Hansen Feng, Lizhi Wang, Zhiwei Xiong, Hua Huang. (2023)  
**Stimulating the Diffusion Model for Image Denoising via Adaptive Embedding and Ensembling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.03992v1)  

---


**ABSTRACT**  
Image denoising is a fundamental problem in computational photography, where achieving high-quality perceptual performance with low distortion is highly demanding. Current methods either struggle with perceptual performance or suffer from significant distortion. Recently, the emerging diffusion model achieves state-of-the-art performance in various tasks, and its denoising mechanism demonstrates great potential for image denoising. However, stimulating diffusion models for image denoising is not straightforward and requires solving several critical problems. On the one hand, the input inconsistency hinders the connection of diffusion models and image denoising. On the other hand, the content inconsistency between the generated image and the desired denoised image introduces additional distortion. To tackle these problems, we present a novel strategy called Diffusion Model for Image Denoising (DMID) by understanding and rethinking the diffusion model from a denoising perspective. Our DMID strategy includes an adaptive embedding method that embeds the noisy image into a pre-trained diffusion model, and an adaptive ensembling method that reduces distortion in the denoised image. Our DMID strategy achieves state-of-the-art performance on all distortion-based and perceptual metrics, for both Gaussian and real-world image denoising.

{{</citation>}}


### (13/47) Building and Road Segmentation Using EffUNet and Transfer Learning Approach (Sahil Gangurde, 2023)

{{<citation>}}

Sahil Gangurde. (2023)  
**Building and Road Segmentation Using EffUNet and Transfer Learning Approach**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2307.03980v1)  

---


**ABSTRACT**  
In city, information about urban objects such as water supply, railway lines, power lines, buildings, roads, etc., is necessary for city planning. In particular, information about the spread of these objects, locations and capacity is needed for the policymakers to make impactful decisions. This thesis aims to segment the building and roads from the aerial image captured by the satellites and UAVs. Many different architectures have been proposed for the semantic segmentation task and UNet being one of them. In this thesis, we propose a novel architecture based on Google's newly proposed EfficientNetV2 as an encoder for feature extraction with UNet decoder for constructing the segmentation map. Using this approach we achieved a benchmark score for the Massachusetts Building and Road dataset with an mIOU of 0.8365 and 0.9153 respectively.

{{</citation>}}


### (14/47) End-to-End Supervised Multilabel Contrastive Learning (Ahmad Sajedi et al., 2023)

{{<citation>}}

Ahmad Sajedi, Samir Khaki, Konstantinos N. Plataniotis, Mahdi S. Hosseini. (2023)  
**End-to-End Supervised Multilabel Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.03967v1)  

---


**ABSTRACT**  
Multilabel representation learning is recognized as a challenging problem that can be associated with either label dependencies between object categories or data-related issues such as the inherent imbalance of positive/negative samples. Recent advances address these challenges from model- and data-centric viewpoints. In model-centric, the label correlation is obtained by an external model designs (e.g., graph CNN) to incorporate an inductive bias for training. However, they fail to design an end-to-end training framework, leading to high computational complexity. On the contrary, in data-centric, the realistic nature of the dataset is considered for improving the classification while ignoring the label dependencies. In this paper, we propose a new end-to-end training framework -- dubbed KMCL (Kernel-based Mutlilabel Contrastive Learning) -- to address the shortcomings of both model- and data-centric designs. The KMCL first transforms the embedded features into a mixture of exponential kernels in Gaussian RKHS. It is then followed by encoding an objective loss that is comprised of (a) reconstruction loss to reconstruct kernel representation, (b) asymmetric classification loss to address the inherent imbalance problem, and (c) contrastive loss to capture label correlation. The KMCL models the uncertainty of the feature encoder while maintaining a low computational footprint. Extensive experiments are conducted on image classification tasks to showcase the consistent improvements of KMCL over the SOTA methods. PyTorch implementation is provided in \url{https://github.com/mahdihosseini/KMCL}.

{{</citation>}}


### (15/47) Reading Between the Lanes: Text VideoQA on the Road (George Tom et al., 2023)

{{<citation>}}

George Tom, Minesh Mathew, Sergi Garcia, Dimosthenis Karatzas, C. V. Jawahar. (2023)  
**Reading Between the Lanes: Text VideoQA on the Road**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.03948v1)  

---


**ABSTRACT**  
Text and signs around roads provide crucial information for drivers, vital for safe navigation and situational awareness. Scene text recognition in motion is a challenging problem, while textual cues typically appear for a short time span, and early detection at a distance is necessary. Systems that exploit such information to assist the driver should not only extract and incorporate visual and textual cues from the video stream but also reason over time. To address this issue, we introduce RoadTextVQA, a new dataset for the task of video question answering (VideoQA) in the context of driver assistance. RoadTextVQA consists of $3,222$ driving videos collected from multiple countries, annotated with $10,500$ questions, all based on text or road signs present in the driving videos. We assess the performance of state-of-the-art video question answering models on our RoadTextVQA dataset, highlighting the significant potential for improvement in this domain and the usefulness of the dataset in advancing research on in-vehicle support systems and text-aware multimodal question answering. The dataset is available at http://cvit.iiit.ac.in/research/projects/cvit-projects/roadtextvqa

{{</citation>}}


### (16/47) Camouflaged Object Detection with Feature Grafting and Distractor Aware (Yuxuan Song et al., 2023)

{{<citation>}}

Yuxuan Song, Xinyue Li, Lin Qi. (2023)  
**Camouflaged Object Detection with Feature Grafting and Distractor Aware**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2307.03943v1)  

---


**ABSTRACT**  
The task of Camouflaged Object Detection (COD) aims to accurately segment camouflaged objects that integrated into the environment, which is more challenging than ordinary detection as the texture between the target and background is visually indistinguishable. In this paper, we proposed a novel Feature Grafting and Distractor Aware network (FDNet) to handle the COD task. Specifically, we use CNN and Transformer to encode multi-scale images in parallel. In order to better explore the advantages of the two encoders, we design a cross-attention-based Feature Grafting Module to graft features extracted from Transformer branch into CNN branch, after which the features are aggregated in the Feature Fusion Module. A Distractor Aware Module is designed to explicitly model the two possible distractors in the COD task to refine the coarse camouflage map. We also proposed the largest artificial camouflaged object dataset which contains 2000 images with annotations, named ACOD2K. We conducted extensive experiments on four widely used benchmark datasets and the ACOD2K dataset. The results show that our method significantly outperforms other state-of-the-art methods. The code and the ACOD2K will be available at https://github.com/syxvision/FDNet.

{{</citation>}}


### (17/47) Edge-Aware Mirror Network for Camouflaged Object Detection (Dongyue Sun et al., 2023)

{{<citation>}}

Dongyue Sun, Shiyao Jiang, Lin Qi. (2023)  
**Edge-Aware Mirror Network for Camouflaged Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.03932v1)  

---


**ABSTRACT**  
Existing edge-aware camouflaged object detection (COD) methods normally output the edge prediction in the early stage. However, edges are important and fundamental factors in the following segmentation task. Due to the high visual similarity between camouflaged targets and the surroundings, edge prior predicted in early stage usually introduces erroneous foreground-background and contaminates features for segmentation. To tackle this problem, we propose a novel Edge-aware Mirror Network (EAMNet), which models edge detection and camouflaged object segmentation as a cross refinement process. More specifically, EAMNet has a two-branch architecture, where a segmentation-induced edge aggregation module and an edge-induced integrity aggregation module are designed to cross-guide the segmentation branch and edge detection branch. A guided-residual channel attention module which leverages the residual connection and gated convolution finally better extracts structural details from low-level features. Quantitative and qualitative experiment results show that EAMNet outperforms existing cutting-edge baselines on three widely used COD datasets. Codes are available at https://github.com/sdy1999/EAMNet.

{{</citation>}}


### (18/47) VS-TransGRU: A Novel Transformer-GRU-based Framework Enhanced by Visual-Semantic Fusion for Egocentric Action Anticipation (Congqi Cao et al., 2023)

{{<citation>}}

Congqi Cao, Ze Sun, Qinyi Lv, Lingtong Min, Yanning Zhang. (2023)  
**VS-TransGRU: A Novel Transformer-GRU-based Framework Enhanced by Visual-Semantic Fusion for Egocentric Action Anticipation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.03918v1)  

---


**ABSTRACT**  
Egocentric action anticipation is a challenging task that aims to make advanced predictions of future actions from current and historical observations in the first-person view. Most existing methods focus on improving the model architecture and loss function based on the visual input and recurrent neural network to boost the anticipation performance. However, these methods, which merely consider visual information and rely on a single network architecture, gradually reach a performance plateau. In order to fully understand what has been observed and capture the dependencies between current observations and future actions well enough, we propose a novel visual-semantic fusion enhanced and Transformer GRU-based action anticipation framework in this paper. Firstly, high-level semantic information is introduced to improve the performance of action anticipation for the first time. We propose to use the semantic features generated based on the class labels or directly from the visual observations to augment the original visual features. Secondly, an effective visual-semantic fusion module is proposed to make up for the semantic gap and fully utilize the complementarity of different modalities. Thirdly, to take advantage of both the parallel and autoregressive models, we design a Transformer based encoder for long-term sequential modeling and a GRU-based decoder for flexible iteration decoding. Extensive experiments on two large-scale first-person view datasets, i.e., EPIC-Kitchens and EGTEA Gaze+, validate the effectiveness of our proposed method, which achieves new state-of-the-art performance, outperforming previous approaches by a large margin.

{{</citation>}}


### (19/47) Sketch-A-Shape: Zero-Shot Sketch-to-3D Shape Generation (Aditya Sanghi et al., 2023)

{{<citation>}}

Aditya Sanghi, Pradeep Kumar Jayaraman, Arianna Rampini, Joseph Lambourne, Hooman Shayani, Evan Atherton, Saeid Asgari Taghanaki. (2023)  
**Sketch-A-Shape: Zero-Shot Sketch-to-3D Shape Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Sketch, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.03869v1)  

---


**ABSTRACT**  
Significant progress has recently been made in creative applications of large pre-trained models for downstream tasks in 3D vision, such as text-to-shape generation. This motivates our investigation of how these pre-trained models can be used effectively to generate 3D shapes from sketches, which has largely remained an open challenge due to the limited sketch-shape paired datasets and the varying level of abstraction in the sketches. We discover that conditioning a 3D generative model on the features (obtained from a frozen large pre-trained vision model) of synthetic renderings during training enables us to effectively generate 3D shapes from sketches at inference time. This suggests that the large pre-trained vision model features carry semantic signals that are resilient to domain shifts, i.e., allowing us to use only RGB renderings, but generalizing to sketches at inference time. We conduct a comprehensive set of experiments investigating different design factors and demonstrate the effectiveness of our straightforward approach for generation of multiple 3D shapes per each input sketch regardless of their level of abstraction without requiring any paired datasets during training.

{{</citation>}}


## cs.SI (1)



### (20/47) Social Media Analytics in Disaster Response: A Comprehensive Review (Mohammadsepehr Karimiziarani, 2023)

{{<citation>}}

Mohammadsepehr Karimiziarani. (2023)  
**Social Media Analytics in Disaster Response: A Comprehensive Review**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2307.04046v1)  

---


**ABSTRACT**  
Social media has emerged as a valuable resource for disaster management, revolutionizing the way emergency response and recovery efforts are conducted during natural disasters. This review paper aims to provide a comprehensive analysis of social media analytics for disaster management. The abstract begins by highlighting the increasing prevalence of natural disasters and the need for effective strategies to mitigate their impact. It then emphasizes the growing influence of social media in disaster situations, discussing its role in disaster detection, situational awareness, and emergency communication. The abstract explores the challenges and opportunities associated with leveraging social media data for disaster management purposes. It examines methodologies and techniques used in social media analytics, including data collection, preprocessing, and analysis, with a focus on data mining and machine learning approaches. The abstract also presents a thorough examination of case studies and best practices that demonstrate the successful application of social media analytics in disaster response and recovery. Ethical considerations and privacy concerns related to the use of social media data in disaster scenarios are addressed. The abstract concludes by identifying future research directions and potential advancements in social media analytics for disaster management. The review paper aims to provide practitioners and researchers with a comprehensive understanding of the current state of social media analytics in disaster management, while highlighting the need for continued research and innovation in this field.

{{</citation>}}


## stat.ML (1)



### (21/47) Sup-Norm Convergence of Deep Neural Network Estimator for Nonparametric Regression by Adversarial Training (Masaaki Imaizumi, 2023)

{{<citation>}}

Masaaki Imaizumi. (2023)  
**Sup-Norm Convergence of Deep Neural Network Estimator for Nonparametric Regression by Adversarial Training**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2307.04042v1)  

---


**ABSTRACT**  
We show the sup-norm convergence of deep neural network estimators with a novel adversarial training scheme. For the nonparametric regression problem, it has been shown that an estimator using deep neural networks can achieve better performances in the sense of the $L2$-norm. In contrast, it is difficult for the neural estimator with least-squares to achieve the sup-norm convergence, due to the deep structure of neural network models. In this study, we develop an adversarial training scheme and investigate the sup-norm convergence of deep neural network estimators. First, we find that ordinary adversarial training makes neural estimators inconsistent. Second, we show that a deep neural network estimator achieves the optimal rate in the sup-norm sense by the proposed adversarial training with correction. We extend our adversarial training to general setups of a loss function and a data-generating function. Our experiments support the theoretical findings.

{{</citation>}}


## cs.AI (4)



### (22/47) The Value of Chess Squares (Aditya Gupta et al., 2023)

{{<citation>}}

Aditya Gupta, Shiva Maharaj, Nicholas Polson, Vadim Sokolov. (2023)  
**The Value of Chess Squares**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.05330v1)  

---


**ABSTRACT**  
Valuing chess squares and determining the placement of pieces on the board are the main objectives of our study. With the emergence of chess AI, it has become possible to accurately assess the worth of positions in a game of chess. The conventional approach assigns fixed values to pieces $(\symking=\infty, \symqueen=9, \symrook=5, \symbishop=3, \symknight=3, \sympawn=1)$. We enhance this analysis by introducing marginal valuations for both pieces and squares. We demonstrate our method by examining the positioning of Knights and Bishops, and also provide valuable insights into the valuation of pawns. Notably, Nimzowitsch was among the pioneers in advocating for the significance of Pawn structure and valuation. Finally, we conclude by suggesting potential avenues for future research.

{{</citation>}}


### (23/47) Multi-Intent Detection in User Provided Annotations for Programming by Examples Systems (Nischal Ashok Kumar et al., 2023)

{{<citation>}}

Nischal Ashok Kumar, Nitin Gupta, Shanmukha Guttula, Hima Patel. (2023)  
**Multi-Intent Detection in User Provided Annotations for Programming by Examples Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SE, cs.AI  
Keywords: AI, Intent Detection  
[Paper Link](http://arxiv.org/abs/2307.03966v1)  

---


**ABSTRACT**  
In mapping enterprise applications, data mapping remains a fundamental part of integration development, but its time consuming. An increasing number of applications lack naming standards, and nested field structures further add complexity for the integration developers. Once the mapping is done, data transformation is the next challenge for the users since each application expects data to be in a certain format. Also, while building integration flow, developers need to understand the format of the source and target data field and come up with transformation program that can change data from source to target format. The problem of automatic generation of a transformation program through program synthesis paradigm from some specifications has been studied since the early days of Artificial Intelligence (AI). Programming by Example (PBE) is one such kind of technique that targets automatic inferencing of a computer program to accomplish a format or string conversion task from user-provided input and output samples. To learn the correct intent, a diverse set of samples from the user is required. However, there is a possibility that the user fails to provide a diverse set of samples. This can lead to multiple intents or ambiguity in the input and output samples. Hence, PBE systems can get confused in generating the correct intent program. In this paper, we propose a deep neural network based ambiguity prediction model, which analyzes the input-output strings and maps them to a different set of properties responsible for multiple intent. Users can analyze these properties and accordingly can provide new samples or modify existing samples which can help in building a better PBE system for mapping enterprise applications.

{{</citation>}}


### (24/47) Applying human-centered AI in developing effective human-AI teaming: A perspective of human-AI joint cognitive systems (Wei Xu et al., 2023)

{{<citation>}}

Wei Xu, Zaifeng Gao. (2023)  
**Applying human-centered AI in developing effective human-AI teaming: A perspective of human-AI joint cognitive systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.03913v3)  

---


**ABSTRACT**  
Research and application have used human-AI teaming (HAT) as a new paradigm to develop AI systems. HAT recognizes that AI will function as a teammate instead of simply a tool in collaboration with humans. Effective human-AI teams need to be capable of taking advantage of the unique abilities of both humans and AI while overcoming the known challenges and limitations of each member, augmenting human capabilities, and raising joint performance beyond that of either entity. The National AI Research and Strategic Plan 2023 update has recognized that research programs focusing primarily on the independent performance of AI systems generally fail to consider the functionality that AI must provide within the context of dynamic, adaptive, and collaborative teams and calls for further research on human-AI teaming and collaboration. However, there has been debate about whether AI can work as a teammate with humans. The primary concern is that adopting the "teaming" paradigm contradicts the human-centered AI (HCAI) approach, resulting in humans losing control of AI systems. This article further analyzes the HAT paradigm and the debates. Specifically, we elaborate on our proposed conceptual framework of human-AI joint cognitive systems (HAIJCS) and apply it to represent HAT under the HCAI umbrella. We believe that HAIJCS may help adopt HAI while enabling HCAI. The implications and future work for HAIJCS are also discussed.   Insights: AI has led to the emergence of a new form of human-machine relationship: human-AI teaming (HAT), a paradigmatic shift in human-AI systems; We must follow a human-centered AI (HCAI) approach when applying HAT as a new design paradigm; We propose a conceptual framework of human-AI joint cognitive systems (HAIJCS) to represent and implement HAT for developing effective human-AI teaming

{{</citation>}}


### (25/47) Large Language Models for Supply Chain Optimization (Beibin Li et al., 2023)

{{<citation>}}

Beibin Li, Konstantina Mellou, Bo Zhang, Jeevan Pathuri, Ishai Menache. (2023)  
**Large Language Models for Supply Chain Optimization**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-DM, cs-LG, cs.AI  
Keywords: Language Model, Microsoft  
[Paper Link](http://arxiv.org/abs/2307.03875v2)  

---


**ABSTRACT**  
Supply chain operations traditionally involve a variety of complex decision making problems. Over the last few decades, supply chains greatly benefited from advances in computation, which allowed the transition from manual processing to automation and cost-effective optimization. Nonetheless, business operators still need to spend substantial efforts in explaining and interpreting the optimization outcomes to stakeholders. Motivated by the recent advances in Large Language Models (LLMs), we study how this disruptive technology can help bridge the gap between supply chain automation and human comprehension and trust thereof. We design OptiGuide -- a framework that accepts as input queries in plain text, and outputs insights about the underlying optimization outcomes. Our framework does not forgo the state-of-the-art combinatorial optimization technology, but rather leverages it to quantitatively answer what-if scenarios (e.g., how would the cost change if we used supplier B instead of supplier A for a given demand?). Importantly, our design does not require sending proprietary data over to LLMs, which can be a privacy concern in some circumstances. We demonstrate the effectiveness of our framework on a real server placement scenario within Microsoft's cloud supply chain. Along the way, we develop a general evaluation benchmark, which can be used to evaluate the accuracy of the LLM output in other scenarios.

{{</citation>}}


## cs.RO (4)



### (26/47) Meta-Policy Learning over Plan Ensembles for Robust Articulated Object Manipulation (Constantinos Chamzas et al., 2023)

{{<citation>}}

Constantinos Chamzas, Caelan Garrett, Balakumar Sundaralingam, Lydia E. Kavraki, Dieter Fox. (2023)  
**Meta-Policy Learning over Plan Ensembles for Robust Articulated Object Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.04040v1)  

---


**ABSTRACT**  
Recent work has shown that complex manipulation skills, such as pushing or pouring, can be learned through state-of-the-art learning based techniques, such as Reinforcement Learning (RL). However, these methods often have high sample-complexity, are susceptible to domain changes, and produce unsafe motions that a robot should not perform. On the other hand, purely geometric model-based planning can produce complex behaviors that satisfy all the geometric constraints of the robot but might not be dynamically feasible for a given environment. In this work, we leverage a geometric model-based planner to build a mixture of path-policies on which a task-specific meta-policy can be learned to complete the task. In our results, we demonstrate that a successful meta-policy can be learned to push a door, while requiring little data and being robust to model uncertainty of the environment. We tested our method on a 7-DOF Franka-Emika Robot pushing a cabinet door in simulation.

{{</citation>}}


### (27/47) Employing Drones in Agriculture: An Exploration of Various Drone Types and Key Advantages (E. C. Nunes, 2023)

{{<citation>}}

E. C. Nunes. (2023)  
**Employing Drones in Agriculture: An Exploration of Various Drone Types and Key Advantages**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2307.04037v2)  

---


**ABSTRACT**  
This article explores the use of drones in agriculture and discusses the various types of drones employed for different agricultural applications. Drones, also known as unmanned aerial vehicles (UAVs), offer numerous advantages in farming practices. They provide real-time and high-resolution data collection, enabling farmers to make informed irrigation, fertilization, and pest management decisions. Drones assist in precision spraying and application of agricultural inputs, minimizing chemical wastage and optimizing resource utilization. They offer accessibility to inaccessible areas, reduce manual labor, and provide cost savings and increased operational efficiency. Drones also play a crucial role in mapping and surveying agricultural fields, aiding crop planning and resource allocation. However, challenges such as regulations and limited flight time need to be addressed. The advantages of using drones in agriculture include precision agriculture, cost and time savings, improved data collection and analysis, enhanced crop management, accessibility and flexibility, environmental sustainability, and increased safety for farmers. Overall, drones have the potential to revolutionize farming practices, leading to increased efficiency, productivity, and sustainability in agriculture.

{{</citation>}}


### (28/47) Autonomy 2.0: The Quest for Economies of Scale (Shuang Wu et al., 2023)

{{<citation>}}

Shuang Wu, Bo Yu, Shaoshan Liu, Yuhao Zhu. (2023)  
**Autonomy 2.0: The Quest for Economies of Scale**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CY, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.03973v1)  

---


**ABSTRACT**  
With the advancement of robotics and AI technologies in the past decade, we have now entered the age of autonomous machines. In this new age of information technology, autonomous machines, such as service robots, autonomous drones, delivery robots, and autonomous vehicles, rather than humans, will provide services. In this article, through examining the technical challenges and economic impact of the digital economy, we argue that scalability is both highly necessary from a technical perspective and significantly advantageous from an economic perspective, thus is the key for the autonomy industry to achieve its full potential. Nonetheless, the current development paradigm, dubbed Autonomy 1.0, scales with the number of engineers, instead of with the amount of data or compute resources, hence preventing the autonomy industry to fully benefit from the economies of scale, especially the exponentially cheapening compute cost and the explosion of available data. We further analyze the key scalability blockers and explain how a new development paradigm, dubbed Autonomy 2.0, can address these problems to greatly boost the autonomy industry.

{{</citation>}}


### (29/47) MARBLER: An Open Platform for Standarized Evaluation of Multi-Robot Reinforcement Learning Algorithms (Reza Torbati et al., 2023)

{{<citation>}}

Reza Torbati, Shubham Lohiya, Shivika Singh, Meher Shashwat Nigam, Harish Ravichandar. (2023)  
**MARBLER: An Open Platform for Standarized Evaluation of Multi-Robot Reinforcement Learning Algorithms**  

---
Primary Category: cs.RO  
Categories: cs-MA, cs-RO, cs.RO  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.03891v3)  

---


**ABSTRACT**  
Multi-agent reinforcement learning (MARL) has enjoyed significant recent progress, thanks to deep learning. This is naturally starting to benefit multi-robot systems (MRS) in the form of multi-robot RL (MRRL). However, existing infrastructure to train and evaluate policies predominantly focus on challenges in coordinating virtual agents, and ignore characteristics important to robotic systems. Few platforms support realistic robot dynamics, and fewer still can evaluate Sim2Real performance of learned behavior. To address these issues, we contribute MARBLER: Multi-Agent RL Benchmark and Learning Environment for the Robotarium. MARBLER offers a robust and comprehensive evaluation platform for MRRL by marrying Georgia Tech's Robotarium (which enables rapid prototyping on physical MRS) and OpenAI's Gym framework (which facilitates standardized use of modern learning algorithms). MARBLER offers a highly controllable environment with realistic dynamics, including barrier certificate-based obstacle avoidance. It allows anyone across the world to train and deploy MRRL algorithms on a physical testbed with reproducibility. Further, we introduce five novel scenarios inspired by common challenges in MRS and provide support for new custom scenarios. Finally, we use MARBLER to evaluate popular MARL algorithms and provide insights into their suitability for MRRL. In summary, MARBLER can be a valuable tool to the MRS research community by facilitating comprehensive and standardized evaluation of learning algorithms on realistic simulations and physical hardware. Links to our open-source framework and the videos of real-world experiments can be found at https://shubhlohiya.github.io/MARBLER/.

{{</citation>}}


## cs.HC (2)



### (30/47) Designing a Direct Feedback Loop between Humans and Convolutional Neural Networks through Local Explanations (Tong Steven Sun et al., 2023)

{{<citation>}}

Tong Steven Sun, Yuyang Gao, Shubham Khaladkar, Sijia Liu, Liang Zhao, Young-Ho Kim, Sungsoo Ray Hong. (2023)  
**Designing a Direct Feedback Loop between Humans and Convolutional Neural Networks through Local Explanations**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CV, cs-HC, cs-LG, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.04036v1)  

---


**ABSTRACT**  
The local explanation provides heatmaps on images to explain how Convolutional Neural Networks (CNNs) derive their output. Due to its visual straightforwardness, the method has been one of the most popular explainable AI (XAI) methods for diagnosing CNNs. Through our formative study (S1), however, we captured ML engineers' ambivalent perspective about the local explanation as a valuable and indispensable envision in building CNNs versus the process that exhausts them due to the heuristic nature of detecting vulnerability. Moreover, steering the CNNs based on the vulnerability learned from the diagnosis seemed highly challenging. To mitigate the gap, we designed DeepFuse, the first interactive design that realizes the direct feedback loop between a user and CNNs in diagnosing and revising CNN's vulnerability using local explanations. DeepFuse helps CNN engineers to systemically search "unreasonable" local explanations and annotate the new boundaries for those identified as unreasonable in a labor-efficient manner. Next, it steers the model based on the given annotation such that the model doesn't introduce similar mistakes. We conducted a two-day study (S2) with 12 experienced CNN engineers. Using DeepFuse, participants made a more accurate and "reasonable" model than the current state-of-the-art. Also, participants found the way DeepFuse guides case-based reasoning can practically improve their current practice. We provide implications for design that explain how future HCI-driven design can move our practice forward to make XAI-driven insights more actionable.

{{</citation>}}


### (31/47) Designing Mixed-Initiative Video Games (Daijin Yang, 2023)

{{<citation>}}

Daijin Yang. (2023)  
**Designing Mixed-Initiative Video Games**  

---
Primary Category: cs.HC  
Categories: J-5, cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.03877v1)  

---


**ABSTRACT**  
The development of Artificial Intelligence (AI) enables humans to co-create content with machines. The unexpectedness of AI-generated content can bring inspiration and entertainment to users. However, the co-creation interactions are always designed for content creators and have poor accessibility. To explore gamification of mixed-initiative co-creation and make human-AI interactions accessible and fun for players, I prototyped Snake Story, a mixed-initiative game where players can select AI-generated texts to write a story of a snake by playing a "Snake" like game. A controlled experiment was conducted to investigate the dynamics of player-AI interactions with and without the game component in the designed interface. As a result of a study with 11 players (n=11), I found that players utilized different strategies when playing with the two versions, game mechanics significantly affected the output stories, players' creative process, as well as role perceptions, and players with different backgrounds showed different preferences for the two versions. Based on these results, I further discussed considerations for mixed-initiative game design. This work aims to inspire the design of engaging co-creation experiences.

{{</citation>}}


## cs.SD (1)



### (32/47) Emotion-Guided Music Accompaniment Generation Based on Variational Autoencoder (Qi Wang et al., 2023)

{{<citation>}}

Qi Wang, Shubing Zhang, Li Zhou. (2023)  
**Emotion-Guided Music Accompaniment Generation Based on Variational Autoencoder**  

---
Primary Category: cs.SD  
Categories: cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.04015v1)  

---


**ABSTRACT**  
Music accompaniment generation is a crucial aspect in the composition process. Deep neural networks have made significant strides in this field, but it remains a challenge for AI to effectively incorporate human emotions to create beautiful accompaniments. Existing models struggle to effectively characterize human emotions within neural network models while composing music. To address this issue, we propose the use of an easy-to-represent emotion flow model, the Valence/Arousal Curve, which allows for the compatibility of emotional information within the model through data transformation and enhances interpretability of emotional factors by utilizing a Variational Autoencoder as the model structure. Further, we used relative self-attention to maintain the structure of the music at music phrase level and to generate a richer accompaniment when combined with the rules of music theory.

{{</citation>}}


## physics.flu-dyn (1)



### (33/47) Understanding the Efficacy of U-Net & Vision Transformer for Groundwater Numerical Modelling (Maria Luisa Taccari et al., 2023)

{{<citation>}}

Maria Luisa Taccari, Oded Ovadia, He Wang, Adar Kahana, Xiaohui Chen, Peter K. Jimack. (2023)  
**Understanding the Efficacy of U-Net & Vision Transformer for Groundwater Numerical Modelling**  

---
Primary Category: physics.flu-dyn  
Categories: cs-CE, cs-LG, physics-flu-dyn, physics.flu-dyn  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.04010v1)  

---


**ABSTRACT**  
This paper presents a comprehensive comparison of various machine learning models, namely U-Net, U-Net integrated with Vision Transformers (ViT), and Fourier Neural Operator (FNO), for time-dependent forward modelling in groundwater systems. Through testing on synthetic datasets, it is demonstrated that U-Net and U-Net + ViT models outperform FNO in accuracy and efficiency, especially in sparse data scenarios. These findings underscore the potential of U-Net-based models for groundwater modelling in real-world applications where data scarcity is prevalent.

{{</citation>}}


## cs.SE (1)



### (34/47) ReviewRanker: A Semi-Supervised Learning Based Approach for Code Review Quality Estimation (Saifullah Mahbub et al., 2023)

{{<citation>}}

Saifullah Mahbub, Md. Easin Arafat, Chowdhury Rafeed Rahman, Zannatul Ferdows, Masum Hasan. (2023)  
**ReviewRanker: A Semi-Supervised Learning Based Approach for Code Review Quality Estimation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2307.03996v1)  

---


**ABSTRACT**  
Code review is considered a key process in the software industry for minimizing bugs and improving code quality. Inspection of review process effectiveness and continuous improvement can boost development productivity. Such inspection is a time-consuming and human-bias-prone task. We propose a semi-supervised learning based system ReviewRanker which is aimed at assigning each code review a confidence score which is expected to resonate with the quality of the review. Our proposed method is trained based on simple and and well defined labels provided by developers. The labeling task requires little to no effort from the developers and has an indirect relation to the end goal (assignment of review confidence score). ReviewRanker is expected to improve industry-wide code review quality inspection through reducing human bias and effort required for such task. The system has the potential of minimizing the back-and-forth cycle existing in the development and review process. Usable code and dataset for this research can be found at: https://github.com/saifarnab/code_review

{{</citation>}}


## cs.LG (4)



### (35/47) NLP Meets RNA: Unsupervised Embedding Learning for Ribozymes with Word2Vec (Andrew Kean Gao, 2023)

{{<citation>}}

Andrew Kean Gao. (2023)  
**NLP Meets RNA: Unsupervised Embedding Learning for Ribozymes with Word2Vec**  

---
Primary Category: cs.LG  
Categories: I-2-7, cs-LG, cs.LG, q-bio-BM  
Keywords: Embedding, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2307.05537v1)  

---


**ABSTRACT**  
Ribozymes, RNA molecules with distinct 3D structures and catalytic activity, have widespread applications in synthetic biology and therapeutics. However, relatively little research has focused on leveraging deep learning to enhance our understanding of ribozymes. This study implements Word2Vec, an unsupervised learning technique for natural language processing, to learn ribozyme embeddings. Ribo2Vec was trained on over 9,000 diverse ribozymes, learning to map sequences to 128 and 256-dimensional vector spaces. Using Ribo2Vec, sequence embeddings for five classes of ribozymes (hatchet, pistol, hairpin, hovlinc, and twister sister) were calculated. Principal component analysis demonstrated the ability of these embeddings to distinguish between ribozyme classes. Furthermore, a simple SVM classifier trained on ribozyme embeddings showed promising results in accurately classifying ribozyme types. Our results suggest that the embedding vectors contained meaningful information about ribozymes. Interestingly, 256-dimensional embeddings behaved similarly to 128-dimensional embeddings, suggesting that a lower dimension vector space is generally sufficient to capture ribozyme features. This approach demonstrates the potential of Word2Vec for bioinformatics, opening new avenues for ribozyme research. Future research includes using a Transformer-based method to learn RNA embeddings, which can capture long-range interactions between nucleotides.

{{</citation>}}


### (36/47) Digital Twins for Patient Care via Knowledge Graphs and Closed-Form Continuous-Time Liquid Neural Networks (Logan Nye, 2023)

{{<citation>}}

Logan Nye. (2023)  
**Digital Twins for Patient Care via Knowledge Graphs and Closed-Form Continuous-Time Liquid Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2307.04772v1)  

---


**ABSTRACT**  
Digital twin technology has is anticipated to transform healthcare, enabling personalized medicines and support, earlier diagnoses, simulated treatment outcomes, and optimized surgical plans. Digital twins are readily gaining traction in industries like manufacturing, supply chain logistics, and civil infrastructure. Not in patient care, however. The challenge of modeling complex diseases with multimodal patient data and the computational complexities of analyzing it have stifled digital twin adoption in the biomedical vertical. Yet, these major obstacles can potentially be handled by approaching these models in a different way. This paper proposes a novel framework for addressing the barriers to clinical twin modeling created by computational costs and modeling complexities. We propose structuring patient health data as a knowledge graph and using closed-form continuous-time liquid neural networks, for real-time analytics. By synthesizing multimodal patient data and leveraging the flexibility and efficiency of closed form continuous time networks and knowledge graph ontologies, our approach enables real time insights, personalized medicine, early diagnosis and intervention, and optimal surgical planning. This novel approach provides a comprehensive and adaptable view of patient health along with real-time analytics, paving the way for digital twin simulations and other anticipated benefits in healthcare.

{{</citation>}}


### (37/47) Fairness-Aware Graph Neural Networks: A Survey (April Chen et al., 2023)

{{<citation>}}

April Chen, Ryan A. Rossi, Namyong Park, Puja Trivedi, Yu Wang, Tong Yu, Sungchul Kim, Franck Dernoncourt, Nesreen K. Ahmed. (2023)  
**Fairness-Aware Graph Neural Networks: A Survey**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.03929v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have become increasingly important due to their representational power and state-of-the-art predictive performance on many fundamental learning tasks. Despite this success, GNNs suffer from fairness issues that arise as a result of the underlying graph data and the fundamental aggregation mechanism that lies at the heart of the large class of GNN models. In this article, we examine and categorize fairness techniques for improving the fairness of GNNs. Previous work on fair GNN models and techniques are discussed in terms of whether they focus on improving fairness during a preprocessing step, during training, or in a post-processing phase. Furthermore, we discuss how such techniques can be used together whenever appropriate, and highlight the advantages and intuition as well. We also introduce an intuitive taxonomy for fairness evaluation metrics including graph-level fairness, neighborhood-level fairness, embedding-level fairness, and prediction-level fairness metrics. In addition, graph datasets that are useful for benchmarking the fairness of GNN models are summarized succinctly. Finally, we highlight key open problems and challenges that remain to be addressed.

{{</citation>}}


### (38/47) Improving Prototypical Part Networks with Reward Reweighing, Reselection, and Retraining (Robin Netzorg et al., 2023)

{{<citation>}}

Robin Netzorg, Jiaxun Li, Bin Yu. (2023)  
**Improving Prototypical Part Networks with Reward Reweighing, Reselection, and Retraining**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-HC, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.03887v1)  

---


**ABSTRACT**  
In recent years, work has gone into developing deep interpretable methods for image classification that clearly attributes a model's output to specific features of the data. One such of these methods is the prototypical part network (ProtoPNet), which attempts to classify images based on meaningful parts of the input. While this method results in interpretable classifications, this method often learns to classify from spurious or inconsistent parts of the image. Hoping to remedy this, we take inspiration from the recent developments in Reinforcement Learning with Human Feedback (RLHF) to fine-tune these prototypes. By collecting human annotations of prototypes quality via a 1-5 scale on the CUB-200-2011 dataset, we construct a reward model that learns to identify non-spurious prototypes. In place of a full RL update, we propose the reweighted, reselected, and retrained prototypical part network (R3-ProtoPNet), which adds an additional three steps to the ProtoPNet training loop. The first two steps are reward-based reweighting and reselection, which align prototypes with human feedback. The final step is retraining to realign the model's features with the updated prototypes. We find that R3-ProtoPNet improves the overall consistency and meaningfulness of the prototypes, but lower the test predictive accuracy when used independently. When multiple R3-ProtoPNets are incorporated into an ensemble, we find an increase in test predictive performance while maintaining interpretability.

{{</citation>}}


## cs.NI (1)



### (39/47) BER Analysis of Full Duplex Relay assisted BPSK-SIM based VLC System for Indoor Applications (L Bhargava Kumar et al., 2023)

{{<citation>}}

L Bhargava Kumar, Ramavath Prasad Naik, Datta Choudhari, Prabu Krishnan, Goutham Simha G D, Jagadeesh V K. (2023)  
**BER Analysis of Full Duplex Relay assisted BPSK-SIM based VLC System for Indoor Applications**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.03981v1)  

---


**ABSTRACT**  
This paper contemplates a relay-assisted visible light communication (VLC) system, where the light source (Table lamp) acts as a relay node and cooperates with the main light source. Following the IEEE 802.15.7r1 VLC reference channel model, we assume that there are two different light sources present in an office room. The first one is the source terminal present on the ceiling and another one is the desk lamp that serves as the relay station which works in full-duplex method. Because of the loop interference channel, we model VLC relay terminal using ray tracing simulations. We have analyzed bit error rate (BER) performance of the relay-assisted VLC system using binary phase shift keying-subcarrier intensity modulation (BPSK-SIM) technique. The proposed method outperforms existing phase shift keying (PSK) and square M-quadrature amplitude modulation (M-QAM) techniques. The proposed VLC system using BPSK-SIM technique achieves a BER performance of for an SNR of 20 dB. The results of proposed full duplex and half duplex relayed VLC system are evaluated using equal power allocation (EPA) and optimum power allocations (OPA) techniques over three different modulation schemes which are 2-PSK, square M-QAM, BPSK-SIM.

{{</citation>}}


## cs.CY (1)



### (40/47) Right to be Forgotten in the Era of Large Language Models: Implications, Challenges, and Solutions (Dawen Zhang et al., 2023)

{{<citation>}}

Dawen Zhang, Pamela Finckenberg-Broman, Thong Hoang, Shidong Pan, Zhenchang Xing, Mark Staples, Xiwei Xu. (2023)  
**Right to be Forgotten in the Era of Large Language Models: Implications, Challenges, and Solutions**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CL, cs-CY, cs.CY  
Keywords: Google, Language Model  
[Paper Link](http://arxiv.org/abs/2307.03941v1)  

---


**ABSTRACT**  
The Right to be Forgotten (RTBF) was first established as the result of the ruling of Google Spain SL, Google Inc. v AEPD, Mario Costeja Gonz\'alez, and was later included as the Right to Erasure under the General Data Protection Regulation (GDPR) of European Union to allow individuals the right to request personal data be deleted by organizations. Specifically for search engines, individuals can send requests to organizations to exclude their information from the query results. With the recent development of Large Language Models (LLMs) and their use in chatbots, LLM-enabled software systems have become popular. But they are not excluded from the RTBF. Compared with the indexing approach used by search engines, LLMs store, and process information in a completely different way. This poses new challenges for compliance with the RTBF. In this paper, we explore these challenges and provide our insights on how to implement technical solutions for the RTBF, including the use of machine unlearning, model editing, and prompting engineering.

{{</citation>}}


## cs.AR (1)



### (41/47) Towards Efficient In-memory Computing Hardware for Quantized Neural Networks: State-of-the-art, Open Challenges and Perspectives (Olga Krestinskaya et al., 2023)

{{<citation>}}

Olga Krestinskaya, Li Zhang, Khaled Nabil Salama. (2023)  
**Towards Efficient In-memory Computing Hardware for Quantized Neural Networks: State-of-the-art, Open Challenges and Perspectives**  

---
Primary Category: cs.AR  
Categories: cs-AI, cs-AR, cs-ET, cs.AR  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2307.03936v1)  

---


**ABSTRACT**  
The amount of data processed in the cloud, the development of Internet-of-Things (IoT) applications, and growing data privacy concerns force the transition from cloud-based to edge-based processing. Limited energy and computational resources on edge push the transition from traditional von Neumann architectures to In-memory Computing (IMC), especially for machine learning and neural network applications. Network compression techniques are applied to implement a neural network on limited hardware resources. Quantization is one of the most efficient network compression techniques allowing to reduce the memory footprint, latency, and energy consumption. This paper provides a comprehensive review of IMC-based Quantized Neural Networks (QNN) and links software-based quantization approaches to IMC hardware implementation. Moreover, open challenges, QNN design requirements, recommendations, and perspectives along with an IMC-based QNN hardware roadmap are provided.

{{</citation>}}


## cs.CR (1)



### (42/47) Enhancing Room Security and Automating Class Attendance Using ID Cards (Shravan Bhat et al., 2023)

{{<citation>}}

Shravan Bhat, Nithin R, Pranav S. (2023)  
**Enhancing Room Security and Automating Class Attendance Using ID Cards**  

---
Primary Category: cs.CR  
Categories: J-7, cs-CR, cs-HC, cs.CR, none  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.03926v1)  

---


**ABSTRACT**  
With the rapid advancements in technology, automation has emerged as the future of human endeavors. From simple tasks like attendance management to complex security systems, automation has the potential to revolutionize various aspects of our lives. This research paper explores the implementation of a method aimed at enhancing room security in hostels and automating class attendance using ID cards. In this study, we propose a system that utilizes the unique identity information stored in ID cards for various security and check-in tasks. By integrating RFID (Radio-Frequency Identification) reader technology, GSM modules, Node MCU, and Arduino, we create a comprehensive solution. The RFID reader scans the ID card, extracting the relevant information and verifying the user's identity. The data is then transmitted via the GSM module to a central database, ensuring real-time monitoring and security measures. Moreover, the system also enables the automation of class attendance. By utilizing the same ID cards, students can simply tap their cards on a reader placed in the classroom. This information is recorded automatically, eliminating the need for manual attendance taking and reducing errors and time consumption. This research project highlights the practical implementation of ID card technology to enhance room security in hostels and automate class attendance processes. By leveraging the power of automation, we aim to streamline administrative tasks, improve security measures, and optimize efficiency in educational institutions and other relevant settings.

{{</citation>}}


## eess.AS (1)



### (43/47) On decoder-only architecture for speech-to-text and large language model integration (Jian Wu et al., 2023)

{{<citation>}}

Jian Wu, Yashesh Gaur, Zhuo Chen, Long Zhou, Yimeng Zhu, Tianrui Wang, Jinyu Li, Shujie Liu, Bo Ren, Linquan Liu, Yu Wu. (2023)  
**On decoder-only architecture for speech-to-text and large language model integration**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2307.03917v2)  

---


**ABSTRACT**  
Large language models (LLMs) have achieved remarkable success in the field of natural language processing, enabling better human-computer interaction using natural language. However, the seamless integration of speech signals into LLMs has not been explored well. The "decoder-only" architecture has also not been well studied for speech processing tasks. In this research, we introduce Speech-LLaMA, a novel approach that effectively incorporates acoustic information into text-based large language models. Our method leverages Connectionist Temporal Classification and a simple audio encoder to map the compressed acoustic features to the continuous semantic space of the LLM. In addition, we further probe the decoder-only architecture for speech-to-text tasks by training a smaller scale randomly initialized speech-LLaMA model from speech-text paired data alone. We conduct experiments on multilingual speech-to-text translation tasks and demonstrate a significant improvement over strong baselines, highlighting the potential advantages of decoder-only models for speech-to-text conversion.

{{</citation>}}


## math.NA (1)



### (44/47) Mixed Precision Iterative Refinement with Adaptive Precision Sparse Approximate Inverse Preconditioning (Noaman Khan et al., 2023)

{{<citation>}}

Noaman Khan, Erin Carson. (2023)  
**Mixed Precision Iterative Refinement with Adaptive Precision Sparse Approximate Inverse Preconditioning**  

---
Primary Category: math.NA  
Categories: cs-NA, math-NA, math.NA  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.03914v1)  

---


**ABSTRACT**  
Hardware trends have motivated the development of mixed precision algo-rithms in numerical linear algebra, which aim to decrease runtime while maintaining acceptable accuracy. One recent development is the development of an adaptive precision sparse matrix-vector produce routine, which may be used to accelerate the solution of sparse linear systems by iterative methods. This approach is also applicable to the application of inexact preconditioners, such as sparse approximate inverse preconditioners used in Krylov subspace methods. In this work, we develop an adaptive precision sparse approximate inverse preconditioner and demonstrate its use within a five-precision GMRES-based iterative refinement method. We call this algorithm variant BSPAI-GMRES-IR. We then analyze the conditions for the convergence of BSPAI-GMRES-IR, and determine settings under which BSPAI-GMRES-IR will produce similar backward and forward errors as the existing SPAI-GMRES-IR method, the latter of which does not use adaptive precision in preconditioning. Our numerical experiments show that this approach can potentially lead to a reduction in the cost of storing and applying sparse approximate inverse preconditioners, although a significant reduction in cost may comes at the expense of increasing the number of GMRES iterations required for convergence.

{{</citation>}}


## quant-ph (1)



### (45/47) Active Learning in Physics: From 101, to Progress, and Perspective (Yongcheng Ding et al., 2023)

{{<citation>}}

Yongcheng Ding, José D. Martín-Guerrero, Yolanda Vives-Gilabert, Xi Chen. (2023)  
**Active Learning in Physics: From 101, to Progress, and Perspective**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2307.03899v1)  

---


**ABSTRACT**  
Active Learning (AL) is a family of machine learning (ML) algorithms that predates the current era of artificial intelligence. Unlike traditional approaches that require labeled samples for training, AL iteratively selects unlabeled samples to be annotated by an expert. This protocol aims to prioritize the most informative samples, leading to improved model performance compared to training with all labeled samples. In recent years, AL has gained increasing attention, particularly in the field of physics. This paper presents a comprehensive and accessible introduction to the theory of AL reviewing the latest advancements across various domains. Additionally, we explore the potential integration of AL with quantum ML, envisioning a synergistic fusion of these two fields rather than viewing AL as a mere extension of classical ML into the quantum realm.

{{</citation>}}


## cs.IR (1)



### (46/47) Embedding Mental Health Discourse for Community Recommendation (Hy Dang et al., 2023)

{{<citation>}}

Hy Dang, Bang Nguyen, Noah Ziems, Meng Jiang. (2023)  
**Embedding Mental Health Discourse for Community Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.03892v1)  

---


**ABSTRACT**  
Our paper investigates the use of discourse embedding techniques to develop a community recommendation system that focuses on mental health support groups on social media. Social media platforms provide a means for users to anonymously connect with communities that cater to their specific interests. However, with the vast number of online communities available, users may face difficulties in identifying relevant groups to address their mental health concerns. To address this challenge, we explore the integration of discourse information from various subreddit communities using embedding techniques to develop an effective recommendation system. Our approach involves the use of content-based and collaborative filtering techniques to enhance the performance of the recommendation system. Our findings indicate that the proposed approach outperforms the use of each technique separately and provides interpretability in the recommendation process.

{{</citation>}}


## cs.IT (1)



### (47/47) Personalized Resource Allocation in Wireless Networks: An AI-Enabled and Big Data-Driven Multi-Objective Optimization (Rawan Alkurd et al., 2023)

{{<citation>}}

Rawan Alkurd, Ibrahim Abualhaol, Halim Yanikomeroglu. (2023)  
**Personalized Resource Allocation in Wireless Networks: An AI-Enabled and Big Data-Driven Multi-Objective Optimization**  

---
Primary Category: cs.IT  
Categories: cs-AI, cs-IT, cs.IT, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.03867v1)  

---


**ABSTRACT**  
The design and optimization of wireless networks have mostly been based on strong mathematical and theoretical modeling. Nonetheless, as novel applications emerge in the era of 5G and beyond, unprecedented levels of complexity will be encountered in the design and optimization of the network. As a result, the use of Artificial Intelligence (AI) is envisioned for wireless network design and optimization due to the flexibility and adaptability it offers in solving extremely complex problems in real-time. One of the main future applications of AI is enabling user-level personalization for numerous use cases. AI will revolutionize the way we interact with computers in which computers will be able to sense commands and emotions from humans in a non-intrusive manner, making the entire process transparent to users. By leveraging this capability, and accelerated by the advances in computing technologies, wireless networks can be redesigned to enable the personalization of network services to the user level in real-time. While current wireless networks are being optimized to achieve a predefined set of quality requirements, the personalization technology advocated in this article is supported by an intelligent big data-driven layer designed to micro-manage the scarce network resources. This layer provides the intelligence required to decide the necessary service quality that achieves the target satisfaction level for each user. Due to its dynamic and flexible design, personalized networks are expected to achieve unprecedented improvements in optimizing two contradicting objectives in wireless networks: saving resources and improving user satisfaction levels.

{{</citation>}}
