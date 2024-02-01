---
draft: false
title: "arXiv @ 2024.01.29"
date: 2024-01-29
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.29"
    identifier: arxiv_20240129
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (11)](#cscv-11)
- [eess.IV (2)](#eessiv-2)
- [cs.CL (21)](#cscl-21)
- [cs.HC (1)](#cshc-1)
- [cs.CY (3)](#cscy-3)
- [cs.RO (1)](#csro-1)
- [cs.LG (7)](#cslg-7)
- [cs.IR (1)](#csir-1)
- [cs.SE (2)](#csse-2)
- [cs.AI (1)](#csai-1)
- [cs.CR (2)](#cscr-2)
- [cs.SD (1)](#cssd-1)
- [physics.ao-ph (1)](#physicsao-ph-1)
- [math.ST (1)](#mathst-1)

## cs.CV (11)



### (1/55) Exploring the Transferability of a Foundation Model for Fundus Images: Application to Hypertensive Retinopathy (Julio Silva-Rodriguez et al., 2024)

{{<citation>}}

Julio Silva-Rodriguez, Jihed Chelbi, Waziha Kabir, Hadi Chakor, Jose Dolz, Ismail Ben Ayed, Riadh Kobbi. (2024)  
**Exploring the Transferability of a Foundation Model for Fundus Images: Application to Hypertensive Retinopathy**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15526v1)  

---


**ABSTRACT**  
Using deep learning models pre-trained on Imagenet is the traditional solution for medical image classification to deal with data scarcity. Nevertheless, relevant literature supports that this strategy may offer limited gains due to the high dissimilarity between domains. Currently, the paradigm of adapting domain-specialized foundation models is proving to be a promising alternative. However, how to perform such knowledge transfer, and the benefits and limitations it presents, are under study. The CGI-HRDC challenge for Hypertensive Retinopathy diagnosis on fundus images introduces an appealing opportunity to evaluate the transferability of a recently released vision-language foundation model of the retina, FLAIR. In this work, we explore the potential of using FLAIR features as starting point for fundus image classification, and we compare its performance with regard to Imagenet initialization on two popular transfer learning methods: Linear Probing (LP) and Fine-Tuning (FP). Our empirical observations suggest that, in any case, the use of the traditional strategy provides performance gains. In contrast, direct transferability from FLAIR model allows gains of 2.5%. When fine-tuning the whole network, the performance gap increases up to 4%. In this case, we show that avoiding feature deterioration via LP initialization of the classifier allows the best re-use of the rich pre-trained features. Although direct transferability using LP still offers limited performance, we believe that foundation models such as FLAIR will drive the evolution of deep-learning-based fundus image analysis.

{{</citation>}}


### (2/55) FloodLense: A Framework for ChatGPT-based Real-time Flood Detection (Pranath Reddy Kumbam et al., 2024)

{{<citation>}}

Pranath Reddy Kumbam, Kshitij Maruti Vejre. (2024)  
**FloodLense: A Framework for ChatGPT-based Real-time Flood Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.15501v1)  

---


**ABSTRACT**  
This study addresses the vital issue of real-time flood detection and management. It innovatively combines advanced deep learning models with Large language models (LLM), enhancing flood monitoring and response capabilities. This approach addresses the limitations of current methods by offering a more accurate, versatile, user-friendly and accessible solution. The integration of UNet, RDN, and ViT models with natural language processing significantly improves flood area detection in diverse environments, including using aerial and satellite imagery. The experimental evaluation demonstrates the models' efficacy in accurately identifying and mapping flood zones, showcasing the project's potential in transforming environmental monitoring and disaster management fields.

{{</citation>}}


### (3/55) A New Method for Vehicle Logo Recognition Based on Swin Transformer (Yang Li et al., 2024)

{{<citation>}}

Yang Li, Doudou Zhang, Jianli Xiao. (2024)  
**A New Method for Vehicle Logo Recognition Based on Swin Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.15458v1)  

---


**ABSTRACT**  
Intelligent Transportation Systems (ITS) utilize sensors, cameras, and big data analysis to monitor real-time traffic conditions, aiming to improve traffic efficiency and safety. Accurate vehicle recognition is crucial in this process, and Vehicle Logo Recognition (VLR) stands as a key method. VLR enables effective management and monitoring by distinguishing vehicles on the road. Convolutional Neural Networks (CNNs) have made impressive strides in VLR research. However, achieving higher performance demands significant time and computational resources for training. Recently, the rise of Transformer models has brought new opportunities to VLR. Swin Transformer, with its efficient computation and global feature modeling capabilities, outperforms CNNs under challenging conditions. In this paper, we implement real-time VLR using Swin Transformer and fine-tune it for optimal performance. Extensive experiments conducted on three public vehicle logo datasets (HFUT-VL1, XMU, CTGU-VLD) demonstrate impressive top accuracy results of 99.28%, 100%, and 99.17%, respectively. Additionally, the use of a transfer learning strategy enables our method to be on par with state-of-the-art VLR methods. These findings affirm the superiority of our approach over existing methods. Future research can explore and optimize the application of the Swin Transformer in other vehicle vision recognition tasks to drive advancements in ITS.

{{</citation>}}


### (4/55) Face to Cartoon Incremental Super-Resolution using Knowledge Distillation (Trinetra Devkatte et al., 2024)

{{<citation>}}

Trinetra Devkatte, Shiv Ram Dubey, Satish Kumar Singh, Abdenour Hadid. (2024)  
**Face to Cartoon Incremental Super-Resolution using Knowledge Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2401.15366v1)  

---


**ABSTRACT**  
Facial super-resolution/hallucination is an important area of research that seeks to enhance low-resolution facial images for a variety of applications. While Generative Adversarial Networks (GANs) have shown promise in this area, their ability to adapt to new, unseen data remains a challenge. This paper addresses this problem by proposing an incremental super-resolution using GANs with knowledge distillation (ISR-KD) for face to cartoon. Previous research in this area has not investigated incremental learning, which is critical for real-world applications where new data is continually being generated. The proposed ISR-KD aims to develop a novel unified framework for facial super-resolution that can handle different settings, including different types of faces such as cartoon face and various levels of detail. To achieve this, a GAN-based super-resolution network was pre-trained on the CelebA dataset and then incrementally trained on the iCartoonFace dataset, using knowledge distillation to retain performance on the CelebA test set while improving the performance on iCartoonFace test set. Our experiments demonstrate the effectiveness of knowledge distillation in incrementally adding capability to the model for cartoon face super-resolution while retaining the learned knowledge for facial hallucination tasks in GANs.

{{</citation>}}


### (5/55) An open dataset for oracle bone script recognition and decipherment (Pengjie Wang et al., 2024)

{{<citation>}}

Pengjie Wang, Kaile Zhang, Yuliang Liu, Jinpeng Wan, Haisu Guan, Zhebin Kuang, Xinyu Wang, Lianwen Jin, Xiang Bai. (2024)  
**An open dataset for oracle bone script recognition and decipherment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15365v1)  

---


**ABSTRACT**  
Oracle Bone Script (OBS), one of the earliest known forms of ancient Chinese writing, holds invaluable insights into the humanities and geography of the Shang Dynasty, dating back 3,000 years. The immense historical and cultural significance of these writings cannot be overstated. However, the passage of time has obscured much of their meaning, presenting a significant challenge in deciphering these ancient texts. With the advent of Artificial Intelligence (AI), employing AI to assist in interpreting OBS has become a feasible option. Yet, progress in this area has been hindered by a lack of high-quality datasets. To address this issue, this paper details the creation of the HUST-OBS dataset. This dataset encompasses 77,064 images of 1,588 individual deciphered scripts and 62,989 images of 9,411 undeciphered characters, with a total of 140,053 images, compiled from diverse sources. Additionally, all images and labels have been reviewed and corrected by experts in oracle bone studies. The hope is that this dataset could inspire and assist future research in deciphering those unknown OBS.

{{</citation>}}


### (6/55) Transformer-based Clipped Contrastive Quantization Learning for Unsupervised Image Retrieval (Ayush Dubey et al., 2024)

{{<citation>}}

Ayush Dubey, Shiv Ram Dubey, Satish Kumar Singh, Wei-Ta Chu. (2024)  
**Transformer-based Clipped Contrastive Quantization Learning for Unsupervised Image Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Quantization, Transformer  
[Paper Link](http://arxiv.org/abs/2401.15362v1)  

---


**ABSTRACT**  
Unsupervised image retrieval aims to learn the important visual characteristics without any given level to retrieve the similar images for a given query image. The Convolutional Neural Network (CNN)-based approaches have been extensively exploited with self-supervised contrastive learning for image hashing. However, the existing approaches suffer due to lack of effective utilization of global features by CNNs and biased-ness created by false negative pairs in the contrastive learning. In this paper, we propose a TransClippedCLR model by encoding the global context of an image using Transformer having local context through patch based processing, by generating the hash codes through product quantization and by avoiding the potential false negative pairs through clipped contrastive learning. The proposed model is tested with superior performance for unsupervised image retrieval on benchmark datasets, including CIFAR10, NUS-Wide and Flickr25K, as compared to the recent state-of-the-art deep models. The results using the proposed clipped contrastive learning are greatly improved on all datasets as compared to same backbone network with vanilla contrastive learning.

{{</citation>}}


### (7/55) You Only Look Bottom-Up for Monocular 3D Object Detection (Kaixin Xiong et al., 2024)

{{<citation>}}

Kaixin Xiong, Dingyuan Zhang, Dingkang Liang, Zhe Liu, Hongcheng Yang, Wondimu Dikubab, Jianwei Cheng, Xiang Bai. (2024)  
**You Only Look Bottom-Up for Monocular 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Object Detection  
[Paper Link](http://arxiv.org/abs/2401.15319v1)  

---


**ABSTRACT**  
Monocular 3D Object Detection is an essential task for autonomous driving. Meanwhile, accurate 3D object detection from pure images is very challenging due to the loss of depth information. Most existing image-based methods infer objects' location in 3D space based on their 2D sizes on the image plane, which usually ignores the intrinsic position clues from images, leading to unsatisfactory performances. Motivated by the fact that humans could leverage the bottom-up positional clues to locate objects in 3D space from a single image, in this paper, we explore the position modeling from the image feature column and propose a new method named You Only Look Bottum-Up (YOLOBU). Specifically, our YOLOBU leverages Column-based Cross Attention to determine how much a pixel contributes to pixels above it. Next, the Row-based Reverse Cumulative Sum (RRCS) is introduced to build the connections of pixels in the bottom-up direction. Our YOLOBU fully explores the position clues for monocular 3D detection via building the relationship of pixels from the bottom-up way. Extensive experiments on the KITTI dataset demonstrate the effectiveness and superiority of our method.

{{</citation>}}


### (8/55) SkipViT: Speeding Up Vision Transformers with a Token-Level Skip Connection (Foozhan Ataiefard et al., 2024)

{{<citation>}}

Foozhan Ataiefard, Walid Ahmed, Habib Hajimolahoseini, Saina Asani, Farnoosh Javadi, Mohammad Hassanpour, Omar Mohamed Awad, Austin Wen, Kangling Liu, Yang Liu. (2024)  
**SkipViT: Speeding Up Vision Transformers with a Token-Level Skip Connection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.15293v1)  

---


**ABSTRACT**  
Vision transformers are known to be more computationally and data-intensive than CNN models. These transformer models such as ViT, require all the input image tokens to learn the relationship among them. However, many of these tokens are not informative and may contain irrelevant information such as unrelated background or unimportant scenery. These tokens are overlooked by the multi-head self-attention (MHSA), resulting in many redundant and unnecessary computations in MHSA and the feed-forward network (FFN). In this work, we propose a method to optimize the amount of unnecessary interactions between unimportant tokens by separating and sending them through a different low-cost computational path. Our method does not add any parameters to the ViT model and aims to find the best trade-off between training throughput and achieving a 0% loss in the Top-1 accuracy of the final model. Our experimental results on training ViT-small from scratch show that SkipViT is capable of effectively dropping 55% of the tokens while gaining more than 13% training throughput and maintaining classification accuracy at the level of the baseline model on Huawei Ascend910A.

{{</citation>}}


### (9/55) STAC: Leveraging Spatio-Temporal Data Associations For Efficient Cross-Camera Streaming and Analytics (Volodymyr Vakhniuk et al., 2024)

{{<citation>}}

Volodymyr Vakhniuk, Ayush Sarkar, Ragini Gupta. (2024)  
**STAC: Leveraging Spatio-Temporal Data Associations For Efficient Cross-Camera Streaming and Analytics**  

---
Primary Category: cs.CV  
Categories: I-4-2; I-4-0; C-2-2; C-2-0, cs-CV, cs-MM, cs-NI, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15288v1)  

---


**ABSTRACT**  
We propose an efficient cross-cameras surveillance system called,STAC, that leverages spatio-temporal associations between multiple cameras to provide real-time analytics and inference under constrained network environments. STAC is built using the proposed omni-scale feature learning people reidentification (reid) algorithm that allows accurate detection, tracking and re-identification of people across cameras using the spatio-temporal characteristics of video frames. We integrate STAC with frame filtering and state-of-the-art compression for streaming technique (that is, ffmpeg libx264 codec) to remove redundant information from cross-camera frames. This helps in optimizing the cost of video transmission as well as compute/processing, while maintaining high accuracy for real-time query inference. The introduction of AICity Challenge 2023 Data [1] by NVIDIA has allowed exploration of systems utilizing multi-camera people tracking algorithms. We evaluate the performance of STAC using this dataset to measure the accuracy metrics and inference rate for reid. Additionally, we quantify the reduction in video streams achieved through frame filtering and compression using FFmpeg compared to the raw camera streams. For completeness, we make available our repository to reproduce the results, available at https://github.com/VolodymyrVakhniuk/CS444_Final_Project.

{{</citation>}}


### (10/55) Dynamic Transformer Architecture for Continual Learning of Multimodal Tasks (Yuliang Cai et al., 2024)

{{<citation>}}

Yuliang Cai, Mohammad Rostami. (2024)  
**Dynamic Transformer Architecture for Continual Learning of Multimodal Tasks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.15275v1)  

---


**ABSTRACT**  
Transformer neural networks are increasingly replacing prior architectures in a wide range of applications in different data modalities. The increasing size and computational demands of fine-tuning large pre-trained transformer neural networks pose significant challenges for the widespread adoption of these models for applications that demand on-edge computing. To tackle this challenge, continual learning (CL) emerges as a solution by facilitating the transfer of knowledge across tasks that arrive sequentially for an autonomously learning agent. However, current CL methods mainly focus on learning tasks that are exclusively vision-based or language-based. We propose a transformer-based CL framework focusing on learning tasks that involve both vision and language, known as Vision-and-Language (VaL) tasks. Due to the success of transformers in other modalities, our architecture has the potential to be used in multimodal learning settings. In our framework, we benefit from introducing extra parameters to a base transformer to specialize the network for each task. As a result, we enable dynamic model expansion to learn several tasks in a sequence. We also use knowledge distillation to benefit from relevant past experiences to learn the current task more efficiently. Our proposed method, Task Attentive Multimodal Continual Learning (TAM-CL), allows for the exchange of information between tasks while mitigating the problem of catastrophic forgetting. Notably, our approach is scalable, incurring minimal memory and time overhead. TAM-CL achieves state-of-the-art (SOTA) performance on challenging multimodal tasks

{{</citation>}}


### (11/55) Vanishing-Point-Guided Video Semantic Segmentation of Driving Scenes (Diandian Guo et al., 2024)

{{<citation>}}

Diandian Guo, Deng-Ping Fan, Tongyu Lu, Christos Sakaridis, Luc Van Gool. (2024)  
**Vanishing-Point-Guided Video Semantic Segmentation of Driving Scenes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.15261v1)  

---


**ABSTRACT**  
The estimation of implicit cross-frame correspondences and the high computational cost have long been major challenges in video semantic segmentation (VSS) for driving scenes. Prior works utilize keyframes, feature propagation, or cross-frame attention to address these issues. By contrast, we are the first to harness vanishing point (VP) priors for more effective segmentation. Intuitively, objects near VPs (i.e., away from the vehicle) are less discernible. Moreover, they tend to move radially away from the VP over time in the usual case of a forward-facing camera, a straight road, and linear forward motion of the vehicle. Our novel, efficient network for VSS, named VPSeg, incorporates two modules that utilize exactly this pair of static and dynamic VP priors: sparse-to-dense feature mining (DenseVP) and VP-guided motion fusion (MotionVP). MotionVP employs VP-guided motion estimation to establish explicit correspondences across frames and help attend to the most relevant features from neighboring frames, while DenseVP enhances weak dynamic features in distant regions around VPs. These modules operate within a context-detail framework, which separates contextual features from high-resolution local features at different input resolutions to reduce computational costs. Contextual and local features are integrated through contextualized motion attention (CMA) for the final prediction. Extensive experiments on two popular driving segmentation benchmarks, Cityscapes and ACDC, demonstrate that VPSeg outperforms previous SOTA methods, with only modest computational overhead.

{{</citation>}}


## eess.IV (2)



### (12/55) MiTU-Net: A fine-tuned U-Net with SegFormer backbone for segmenting pubic symphysis-fetal head (Fangyijie Wang et al., 2024)

{{<citation>}}

Fangyijie Wang, Guenole Silvestre, Kathleen Curran. (2024)  
**MiTU-Net: A fine-tuned U-Net with SegFormer backbone for segmenting pubic symphysis-fetal head**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.15513v1)  

---


**ABSTRACT**  
Ultrasound measurements have been examined as potential tools for predicting the likelihood of successful vaginal delivery. The angle of progression (AoP) is a measurable parameter that can be obtained during the initial stage of labor. The AoP is defined as the angle between a straight line along the longitudinal axis of the pubic symphysis (PS) and a line from the inferior edge of the PS to the leading edge of the fetal head (FH). However, the process of measuring AoP on ultrasound images is time consuming and prone to errors. To address this challenge, we propose the Mix Transformer U-Net (MiTU-Net) network, for automatic fetal head-pubic symphysis segmentation and AoP measurement. The MiTU-Net model is based on an encoder-decoder framework, utilizing a pre-trained efficient transformer to enhance feature representation. Within the efficient transformer encoder, the model significantly reduces the trainable parameters of the encoder-decoder model. The effectiveness of the proposed method is demonstrated through experiments conducted on a recent transperineal ultrasound dataset. Our model achieves competitive performance, ranking 5th compared to existing approaches. The MiTU-Net presents an efficient method for automatic segmentation and AoP measurement, reducing errors and assisting sonographers in clinical practice. Reproducibility: Framework implementation and models available on https://github.com/13204942/MiTU-Net.

{{</citation>}}


### (13/55) ParaTransCNN: Parallelized TransCNN Encoder for Medical Image Segmentation (Hongkun Sun et al., 2024)

{{<citation>}}

Hongkun Sun, Jing Xu, Yuping Duan. (2024)  
**ParaTransCNN: Parallelized TransCNN Encoder for Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.15307v1)  

---


**ABSTRACT**  
The convolutional neural network-based methods have become more and more popular for medical image segmentation due to their outstanding performance. However, they struggle with capturing long-range dependencies, which are essential for accurately modeling global contextual correlations. Thanks to the ability to model long-range dependencies by expanding the receptive field, the transformer-based methods have gained prominence. Inspired by this, we propose an advanced 2D feature extraction method by combining the convolutional neural network and Transformer architectures. More specifically, we introduce a parallelized encoder structure, where one branch uses ResNet to extract local information from images, while the other branch uses Transformer to extract global information. Furthermore, we integrate pyramid structures into the Transformer to extract global information at varying resolutions, especially in intensive prediction tasks. To efficiently utilize the different information in the parallelized encoder at the decoder stage, we use a channel attention module to merge the features of the encoder and propagate them through skip connections and bottlenecks. Intensive numerical experiments are performed on both aortic vessel tree, cardiac, and multi-organ datasets. By comparing with state-of-the-art medical image segmentation methods, our method is shown with better segmentation accuracy, especially on small organs. The code is publicly available on https://github.com/HongkunSun/ParaTransCNN.

{{</citation>}}


## cs.CL (21)



### (14/55) Style-News: Incorporating Stylized News Generation and Adversarial Verification for Neural Fake News Detection (Wei-Yao Wang et al., 2024)

{{<citation>}}

Wei-Yao Wang, Yu-Chieh Chang, Wen-Chih Peng. (2024)  
**Style-News: Incorporating Stylized News Generation and Adversarial Verification for Neural Fake News Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-SI, cs.CL  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2401.15509v1)  

---


**ABSTRACT**  
With the improvements in generative models, the issues of producing hallucinations in various domains (e.g., law, writing) have been brought to people's attention due to concerns about misinformation. In this paper, we focus on neural fake news, which refers to content generated by neural networks aiming to mimic the style of real news to deceive people. To prevent harmful disinformation spreading fallaciously from malicious social media (e.g., content farms), we propose a novel verification framework, Style-News, using publisher metadata to imply a publisher's template with the corresponding text types, political stance, and credibility. Based on threat modeling aspects, a style-aware neural news generator is introduced as an adversary for generating news content conditioning for a specific publisher, and style and source discriminators are trained to defend against this attack by identifying which publisher the style corresponds with, and discriminating whether the source of the given news is human-written or machine-generated. To evaluate the quality of the generated content, we integrate various dimensional metrics (language fluency, content preservation, and style adherence) and demonstrate that Style-News significantly outperforms the previous approaches by a margin of 0.35 for fluency, 15.24 for content, and 0.38 for style at most. Moreover, our discriminative model outperforms state-of-the-art baselines in terms of publisher prediction (up to 4.64%) and neural fake news detection (+6.94% $\sim$ 31.72%).

{{</citation>}}


### (15/55) Do We Need Language-Specific Fact-Checking Models? The Case of Chinese (Caiqi Zhang et al., 2024)

{{<citation>}}

Caiqi Zhang, Zhijiang Guo, Andreas Vlachos. (2024)  
**Do We Need Language-Specific Fact-Checking Models? The Case of Chinese**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Fact-Checking, GPT  
[Paper Link](http://arxiv.org/abs/2401.15498v1)  

---


**ABSTRACT**  
This paper investigates the potential benefits of language-specific fact-checking models, focusing on the case of Chinese. We demonstrate the limitations of methods such as translating Chinese claims and evidence into English or directly using multilingual large language models (e.g. GPT4), highlighting the need for language-specific systems. We further develop a state-of-the-art Chinese fact-checking system that, in contrast to previous approaches which treat evidence selection as a pairwise sentence classification task, considers the context of sentences. We also create an adversarial dataset to identify biases in our model, and while they are present as in English language datasets and models, they are often specific to the Chinese culture. Our study emphasizes the importance of language-specific fact-checking models to effectively combat misinformation.

{{</citation>}}


### (16/55) Baichuan2-Sum: Instruction Finetune Baichuan2-7B Model for Dialogue Summarization (Jianfei Xiao et al., 2024)

{{<citation>}}

Jianfei Xiao, Yancan Chen, Yimin Ou, Hanyi Yu, Yiyong Xiao. (2024)  
**Baichuan2-Sum: Instruction Finetune Baichuan2-7B Model for Dialogue Summarization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Dialog, Dialogue, Summarization  
[Paper Link](http://arxiv.org/abs/2401.15496v2)  

---


**ABSTRACT**  
Large language models (LLMs) like Llama, Baichuan and Bloom models show remarkable ability with instruction fine-tuning in many natural language tasks. Nevertheless, for the dialogue summarization task, which aims to generate summaries for different roles in dialogue, most of the state-of-the-art methods conduct on small models (e.g Bart and Bert). Existing methods try to add task specified optimization on small models like adding global-local centrality score to models. In this paper, we propose an instruction fine-tuning model: Baichuan2-Sum, for role-oriented diaglouge summarization. By setting different instructions for different roles, the model can learn from the dialogue interactions and output the expected summaries. Furthermore, we applied NEFTune technique to add suitable noise during training to improve the results. The experiments demonstrate that the proposed model achieves the new state-of-the-art results on two public dialogue summarization datasets: CSDS and SAMSUM. We release our model and related codes to facilitate future studies on dialogue summarization task.

{{</citation>}}


### (17/55) To Burst or Not to Burst: Generating and Quantifying Improbable Text (Kuleen Sasse et al., 2024)

{{<citation>}}

Kuleen Sasse, Samuel Barham, Efsun Sarioglu Kayi, Edward W. Staley. (2024)  
**To Burst or Not to Burst: Generating and Quantifying Improbable Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2401.15476v1)  

---


**ABSTRACT**  
While large language models (LLMs) are extremely capable at text generation, their outputs are still distinguishable from human-authored text. We explore this separation across many metrics over text, many sampling techniques, many types of text data, and across two popular LLMs, LLaMA and Vicuna. Along the way, we introduce a new metric, recoverability, to highlight differences between human and machine text; and we propose a new sampling technique, burst sampling, designed to close this gap. We find that LLaMA and Vicuna have distinct distributions under many of the metrics, and that this influences our results: Recoverability separates real from fake text better than any other metric when using LLaMA. When using Vicuna, burst sampling produces text which is distributionally closer to real text compared to other sampling techniques.

{{</citation>}}


### (18/55) ConvoSense: Overcoming Monotonous Commonsense Inferences for Conversational AI (Sarah E. Finch et al., 2024)

{{<citation>}}

Sarah E. Finch, Jinho D. Choi. (2024)  
**ConvoSense: Overcoming Monotonous Commonsense Inferences for Conversational AI**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2401.15471v1)  

---


**ABSTRACT**  
Mastering commonsense understanding and reasoning is a pivotal skill essential for conducting engaging conversations. While there have been several attempts to create datasets that facilitate commonsense inferences in dialogue contexts, existing datasets tend to lack in-depth details, restate information already present in the conversation, and often fail to capture the multifaceted nature of commonsense reasoning. In response to these limitations, we compile a new synthetic dataset for commonsense reasoning in dialogue contexts using GPT, ConvoSense, that boasts greater contextual novelty, offers a higher volume of inferences per example, and substantially enriches the detail conveyed by the inferences. Our dataset contains over 500,000 inferences across 12,000 dialogues with 10 popular inference types, which empowers the training of generative commonsense models for dialogue that are superior in producing plausible inferences with high novelty when compared to models trained on the previous datasets. To the best of our knowledge, ConvoSense is the first of its kind to provide such a multitude of novel inferences at such a large scale.

{{</citation>}}


### (19/55) DataFrame QA: A Universal LLM Framework on DataFrame Question Answering Without Data Exposure (Junyi Ye et al., 2024)

{{<citation>}}

Junyi Ye, Mengnan Du, Guiling Wang. (2024)  
**DataFrame QA: A Universal LLM Framework on DataFrame Question Answering Without Data Exposure**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.15463v1)  

---


**ABSTRACT**  
This paper introduces DataFrame question answering (QA), a novel task that utilizes large language models (LLMs) to generate Pandas queries for information retrieval and data analysis on dataframes, emphasizing safe and non-revealing data handling. Our method, which solely relies on dataframe column names, not only ensures data privacy but also significantly reduces the context window in the prompt, streamlining information processing and addressing major challenges in LLM-based data analysis. We propose DataFrame QA as a comprehensive framework that includes safe Pandas query generation and code execution. Various LLMs, notably GPT-4, are evaluated using the pass@1 metric on the renowned WikiSQL and our newly developed 'UCI-DataFrameQA', tailored for complex data analysis queries. Our findings indicate that GPT-4 achieves pass@1 rates of 86% on WikiSQL and 97% on UCI-DataFrameQA, underscoring its capability in securely retrieving and aggregating dataframe values and conducting sophisticated data analyses. This approach, deployable in a zero-shot manner without prior training or adjustments, proves to be highly adaptable and secure for diverse applications.

{{</citation>}}


### (20/55) Learning to Trust Your Feelings: Leveraging Self-awareness in LLMs for Hallucination Mitigation (Yuxin Liang et al., 2024)

{{<citation>}}

Yuxin Liang, Zhuoyang Song, Hao Wang, Jiaxing Zhang. (2024)  
**Learning to Trust Your Feelings: Leveraging Self-awareness in LLMs for Hallucination Mitigation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.15449v1)  

---


**ABSTRACT**  
We evaluate the ability of Large Language Models (LLMs) to discern and express their internal knowledge state, a key factor in countering factual hallucination and ensuring reliable application of LLMs. We observe a robust self-awareness of internal knowledge state in LLMs, evidenced by over 85% accuracy in knowledge probing. However, LLMs often fail to express their internal knowledge during generation, leading to factual hallucinations. We develop an automated hallucination annotation tool, Dreamcatcher, which merges knowledge probing and consistency checking methods to rank factual preference data. Using knowledge preference as reward, We propose a Reinforcement Learning from Knowledge Feedback (RLKF) training framework, leveraging reinforcement learning to enhance the factuality and honesty of LLMs. Our experiments across multiple models show that RLKF training effectively enhances the ability of models to utilize their internal knowledge state, boosting performance in a variety of knowledge-based and honesty-related tasks.

{{</citation>}}


### (21/55) Pre-training and Diagnosing Knowledge Base Completion Models (Vid Kocijan et al., 2024)

{{<citation>}}

Vid Kocijan, Myeongjun Erik Jang, Thomas Lukasiewicz. (2024)  
**Pre-training and Diagnosing Knowledge Base Completion Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.15439v1)  

---


**ABSTRACT**  
In this work, we introduce and analyze an approach to knowledge transfer from one collection of facts to another without the need for entity or relation matching. The method works for both canonicalized knowledge bases and uncanonicalized or open knowledge bases, i.e., knowledge bases where more than one copy of a real-world entity or relation may exist. The main contribution is a method that can make use of large-scale pre-training on facts, which were collected from unstructured text, to improve predictions on structured data from a specific domain. The introduced method is most impactful on small datasets such as ReVerb20k, where a 6% absolute increase of mean reciprocal rank and 65% relative decrease of mean rank over the previously best method was achieved, despite not relying on large pre-trained models like Bert. To understand the obtained pre-trained models better, we then introduce a novel dataset for the analysis of pre-trained models for Open Knowledge Base Completion, called Doge (Diagnostics of Open knowledge Graph Embeddings). It consists of 6 subsets and is designed to measure multiple properties of a pre-trained model: robustness against synonyms, ability to perform deductive reasoning, presence of gender stereotypes, consistency with reverse relations, and coverage of different areas of general knowledge. Using the introduced dataset, we show that the existing OKBC models lack consistency in the presence of synonyms and inverse relations and are unable to perform deductive reasoning. Moreover, their predictions often align with gender stereotypes, which persist even when presented with counterevidence. We additionally investigate the role of pre-trained word embeddings and demonstrate that avoiding biased word embeddings is not a sufficient measure to prevent biased behavior of OKBC models.

{{</citation>}}


### (22/55) Indexing Portuguese NLP Resources with PT-Pump-Up (Rúben Almeida et al., 2024)

{{<citation>}}

Rúben Almeida, Ricardo Campos, Alípio Jorge, Sérgio Nunes. (2024)  
**Indexing Portuguese NLP Resources with PT-Pump-Up**  

---
Primary Category: cs.CL  
Categories: 68P20, I-7-1, cs-CL, cs-IR, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2401.15400v1)  

---


**ABSTRACT**  
The recent advances in natural language processing (NLP) are linked to training processes that require vast amounts of corpora. Access to this data is commonly not a trivial process due to resource dispersion and the need to maintain these infrastructures online and up-to-date. New developments in NLP are often compromised due to the scarcity of data or lack of a shared repository that works as an entry point to the community. This is especially true in low and mid-resource languages, such as Portuguese, which lack data and proper resource management infrastructures. In this work, we propose PT-Pump-Up, a set of tools that aim to reduce resource dispersion and improve the accessibility to Portuguese NLP resources. Our proposal is divided into four software components: a) a web platform to list the available resources; b) a client-side Python package to simplify the loading of Portuguese NLP resources; c) an administrative Python package to manage the platform and d) a public GitHub repository to foster future collaboration and contributions. All four components are accessible using: https://linktr.ee/pt_pump_up

{{</citation>}}


### (23/55) Semantics of Multiword Expressions in Transformer-Based Models: A Survey (Filip Miletić et al., 2024)

{{<citation>}}

Filip Miletić, Sabine Schulte im Walde. (2024)  
**Semantics of Multiword Expressions in Transformer-Based Models: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.15393v1)  

---


**ABSTRACT**  
Multiword expressions (MWEs) are composed of multiple words and exhibit variable degrees of compositionality. As such, their meanings are notoriously difficult to model, and it is unclear to what extent this issue affects transformer architectures. Addressing this gap, we provide the first in-depth survey of MWE processing with transformer models. We overall find that they capture MWE semantics inconsistently, as shown by reliance on surface patterns and memorized information. MWE meaning is also strongly localized, predominantly in early layers of the architecture. Representations benefit from specific linguistic properties, such as lower semantic idiosyncrasy and ambiguity of target expressions. Our findings overall question the ability of transformer models to robustly capture fine-grained semantics. Furthermore, we highlight the need for more directly comparable evaluation setups.

{{</citation>}}


### (24/55) MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries (Yixuan Tang et al., 2024)

{{<citation>}}

Yixuan Tang, Yi Yang. (2024)  
**MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, PaLM  
[Paper Link](http://arxiv.org/abs/2401.15391v1)  

---


**ABSTRACT**  
Retrieval-augmented generation (RAG) augments large language models (LLM) by retrieving relevant knowledge, showing promising potential in mitigating LLM hallucinations and enhancing response quality, thereby facilitating the great adoption of LLMs in practice. However, we find that existing RAG systems are inadequate in answering multi-hop queries, which require retrieving and reasoning over multiple pieces of supporting evidence. Furthermore, to our knowledge, no existing RAG benchmarking dataset focuses on multi-hop queries. In this paper, we develop a novel dataset, MultiHop-RAG, which consists of a knowledge base, a large collection of multi-hop queries, their ground-truth answers, and the associated supporting evidence. We detail the procedure of building the dataset, utilizing an English news article dataset as the underlying RAG knowledge base. We demonstrate the benchmarking utility of MultiHop-RAG in two experiments. The first experiment compares different embedding models for retrieving evidence for multi-hop queries. In the second experiment, we examine the capabilities of various state-of-the-art LLMs, including GPT-4, PaLM, and Llama2-70B, in reasoning and answering multi-hop queries given the evidence. Both experiments reveal that existing RAG methods perform unsatisfactorily in retrieving and answering multi-hop queries. We hope MultiHop-RAG will be a valuable resource for the community in developing effective RAG systems, thereby facilitating greater adoption of LLMs in practice. The MultiHop-RAG and implemented RAG system is publicly available at https://github.com/yixuantt/MultiHop-RAG/.

{{</citation>}}


### (25/55) Towards Event Extraction from Speech with Contextual Clues (Jingqi Kang et al., 2024)

{{<citation>}}

Jingqi Kang, Tongtong Wu, Jinming Zhao, Guitao Wang, Guilin Qi, Yuan-Fang Li, Gholamreza Haffari. (2024)  
**Towards Event Extraction from Speech with Contextual Clues**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-MM, cs.CL  
Keywords: Event Extraction  
[Paper Link](http://arxiv.org/abs/2401.15385v1)  

---


**ABSTRACT**  
While text-based event extraction has been an active research area and has seen successful application in many domains, extracting semantic events from speech directly is an under-explored problem. In this paper, we introduce the Speech Event Extraction (SpeechEE) task and construct three synthetic training sets and one human-spoken test set. Compared to event extraction from text, SpeechEE poses greater challenges mainly due to complex speech signals that are continuous and have no word boundaries. Additionally, unlike perceptible sound events, semantic events are more subtle and require a deeper understanding. To tackle these challenges, we introduce a sequence-to-structure generation paradigm that can produce events from speech signals in an end-to-end manner, together with a conditioned generation method that utilizes speech recognition transcripts as the contextual clue. We further propose to represent events with a flat format to make outputs more natural language-like. Our experimental results show that our method brings significant improvements on all datasets, achieving a maximum F1 gain of 10.7%. The code and datasets are released on https://github.com/jodie-kang/SpeechEE.

{{</citation>}}


### (26/55) A RAG-based Question Answering System Proposal for Understanding Islam: MufassirQAS LLM (Ahmet Yusuf Alan et al., 2024)

{{<citation>}}

Ahmet Yusuf Alan, Enis Karaarslan, Ömer Aydin. (2024)  
**A RAG-based Question Answering System Proposal for Understanding Islam: MufassirQAS LLM**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, NLP, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.15378v3)  

---


**ABSTRACT**  
There exist challenges in learning and understanding religions as the presence of complexity and depth of religious doctrines and teachings. Chatbots as question-answering systems can help in solving these challenges. LLM chatbots use NLP techniques to establish connections between topics and accurately respond to complex questions. These capabilities make it perfect to be used in enlightenment on religion as a question answering chatbot. However, LLMs also have a tendency to generate false information, known as hallucination. The responses of the chatbots can include content that insults personal religious beliefs, interfaith conflicts, and controversial or sensitive topics. It needs to avoid such cases without promoting hate speech or offending certain groups of people or their beliefs. This study uses a vector database-based Retrieval Augmented Generation (RAG) approach to enhance the accuracy and transparency of LLMs. Our question-answering system is called as "MufassirQAS". We created a vector database with several open-access books that include Turkish context. These are Turkish translations, and interpretations on Islam. We worked on creating system prompts with care, ensuring they provide instructions that prevent harmful, offensive, or disrespectful responses. We also tested the MufassirQAS and ChatGPT with sensitive questions. We got better performance with our system. Study and enhancements are still in progress. Results and future works are given.

{{</citation>}}


### (27/55) LegalDuet: Learning Effective Representations for Legal Judgment Prediction through a Dual-View Legal Clue Reasoning (Pengjie Liu et al., 2024)

{{<citation>}}

Pengjie Liu, Zhenghao Liu, Xiaoyuan Yi, Liner Yang, Shuo Wang, Yu Gu, Ge Yu, Xing Xie, Shuang-hua Yang. (2024)  
**LegalDuet: Learning Effective Representations for Legal Judgment Prediction through a Dual-View Legal Clue Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Legal, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.15371v1)  

---


**ABSTRACT**  
Most existing Legal Judgment Prediction (LJP) models focus on discovering the legal triggers in the criminal fact description. However, in real-world scenarios, a professional judge not only needs to assimilate the law case experience that thrives on past sentenced legal judgments but also depends on the professional legal grounded reasoning that learned from professional legal knowledge. In this paper, we propose a LegalDuet model, which pretrains language models to learn a tailored embedding space for making legal judgments. It proposes a dual-view legal clue reasoning mechanism, which derives from two reasoning chains of judges: 1) Law Case Reasoning, which makes legal judgments according to the judgment experiences learned from analogy/confusing legal cases; 2) Legal Ground Reasoning, which lies in matching the legal clues between criminal cases and legal decisions. Our experiments show that LegalDuet achieves state-of-the-art performance on the CAIL2018 dataset and outperforms baselines with about 4% improvements on average. Our dual-view reasoning based pretraining can capture critical legal clues to learn a tailored embedding space to distinguish criminal cases. It reduces LegalDuet's uncertainty during prediction and brings pretraining advances to the confusing/low frequent charges. All codes are available at https://github.com/NEUIR/LegalDuet.

{{</citation>}}


### (28/55) Importance-Aware Data Augmentation for Document-Level Neural Machine Translation (Minghao Wu et al., 2024)

{{<citation>}}

Minghao Wu, Yufei Wang, George Foster, Lizhen Qu, Gholamreza Haffari. (2024)  
**Importance-Aware Data Augmentation for Document-Level Neural Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, BLEU, Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.15360v1)  

---


**ABSTRACT**  
Document-level neural machine translation (DocNMT) aims to generate translations that are both coherent and cohesive, in contrast to its sentence-level counterpart. However, due to its longer input length and limited availability of training data, DocNMT often faces the challenge of data sparsity. To overcome this issue, we propose a novel Importance-Aware Data Augmentation (IADA) algorithm for DocNMT that augments the training data based on token importance information estimated by the norm of hidden states and training gradients. We conduct comprehensive experiments on three widely-used DocNMT benchmarks. Our empirical results show that our proposed IADA outperforms strong DocNMT baselines as well as several data augmentation approaches, with statistical significance on both sentence-level and document-level BLEU.

{{</citation>}}


### (29/55) A Survey on Neural Topic Models: Methods, Applications, and Challenges (Xiaobao Wu et al., 2024)

{{<citation>}}

Xiaobao Wu, Thong Nguyen, Anh Tuan Luu. (2024)  
**A Survey on Neural Topic Models: Methods, Applications, and Challenges**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: Topic Model  
[Paper Link](http://arxiv.org/abs/2401.15351v1)  

---


**ABSTRACT**  
Topic models have been prevalent for decades to discover latent topics and infer topic proportions of documents in an unsupervised fashion. They have been widely used in various applications like text analysis and context recommendation. Recently, the rise of neural networks has facilitated the emergence of a new research field -- Neural Topic Models (NTMs). Different from conventional topic models, NTMs directly optimize parameters without requiring model-specific derivations. This endows NTMs with better scalability and flexibility, resulting in significant research attention and plentiful new methods and applications. In this paper, we present a comprehensive survey on neural topic models concerning methods, applications, and challenges. Specifically, we systematically organize current NTM methods according to their network structures and introduce the NTMs for various scenarios like short texts and cross-lingual documents. We also discuss a wide range of popular applications built on NTMs. Finally, we highlight the challenges confronted by NTMs to inspire future research.

{{</citation>}}


### (30/55) A Comprehensive Survey of Compression Algorithms for Language Models (Seungcheol Park et al., 2024)

{{<citation>}}

Seungcheol Park, Jaehyeon Choi, Sojin Lee, U Kang. (2024)  
**A Comprehensive Survey of Compression Algorithms for Language Models**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.15347v1)  

---


**ABSTRACT**  
How can we compress language models without sacrificing accuracy? The number of compression algorithms for language models is rapidly growing to benefit from remarkable advances of recent language models without side effects due to the gigantic size of language models, such as increased carbon emissions and expensive maintenance fees. While numerous compression algorithms have shown remarkable progress in compressing language models, it ironically becomes challenging to capture emerging trends and identify the fundamental concepts underlying them due to the excessive number of algorithms. In this paper, we survey and summarize diverse compression algorithms including pruning, quantization, knowledge distillation, low-rank approximation, parameter sharing, and efficient architecture design. We not only summarize the overall trend of diverse compression algorithms but also select representative algorithms and provide in-depth analyses of them. We discuss the value of each category of compression algorithms, and the desired properties of low-cost compression algorithms which have a significant impact due to the emergence of large language models. Finally, we introduce promising future research topics based on our survey results.

{{</citation>}}


### (31/55) Equipping Language Models with Tool Use Capability for Tabular Data Analysis in Finance (Adrian Theuma et al., 2024)

{{<citation>}}

Adrian Theuma, Ehsan Shareghi. (2024)  
**Equipping Language Models with Tool Use Capability for Tabular Data Analysis in Finance**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2401.15328v2)  

---


**ABSTRACT**  
Large language models (LLMs) have exhibited an array of reasoning capabilities but face challenges like error propagation and hallucination, particularly in specialised areas like finance, where data is heterogeneous, and precision is paramount. We explore the potential of language model augmentation with external tools to mitigate these limitations and offload certain reasoning steps to external tools that are more suited for the task, instead of solely depending on the LLM's inherent abilities. More concretely, using financial domain question-answering datasets, we apply supervised fine-tuning on a LLaMA-2 13B Chat model to act both as a 'task router' and 'task solver'. The 'task router' dynamically directs a question to either be answered internally by the LLM or externally via the right tool from the tool set. Our tool-equipped SFT model, Raven, demonstrates an improvement of 35.2% and 5.06% over the base model and SFT-only baselines, respectively, and is highly competitive with strong GPT-3.5 results. To the best of our knowledge, our work is the first that investigates tool augmentation of language models for the finance domain.

{{</citation>}}


### (32/55) UNSEE: Unsupervised Non-contrastive Sentence Embeddings (Ömer Veysel Çağatan, 2024)

{{<citation>}}

Ömer Veysel Çağatan. (2024)  
**UNSEE: Unsupervised Non-contrastive Sentence Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2401.15316v1)  

---


**ABSTRACT**  
We present UNSEE: Unsupervised Non-Contrastive Sentence Embeddings, a novel approach that outperforms SimCSE in the Massive Text Embedding benchmark. Our exploration begins by addressing the challenge of representation collapse, a phenomenon observed when contrastive objectives in SimCSE are replaced with non-contrastive objectives. To counter this issue, we propose a straightforward solution known as the target network, effectively mitigating representation collapse. The introduction of the target network allows us to leverage non-contrastive objectives, maintaining training stability while achieving performance improvements comparable to contrastive objectives. Our method has achieved peak performance in non-contrastive sentence embeddings through meticulous fine-tuning and optimization. This comprehensive effort has yielded superior sentence representation models, showcasing the effectiveness of our approach.

{{</citation>}}


### (33/55) How We Refute Claims: Automatic Fact-Checking through Flaw Identification and Explanation (Wei-Yu Kao et al., 2024)

{{<citation>}}

Wei-Yu Kao, An-Zi Yen. (2024)  
**How We Refute Claims: Automatic Fact-Checking through Flaw Identification and Explanation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Fact-Checking  
[Paper Link](http://arxiv.org/abs/2401.15312v1)  

---


**ABSTRACT**  
Automated fact-checking is a crucial task in the governance of internet content. Although various studies utilize advanced models to tackle this issue, a significant gap persists in addressing complex real-world rumors and deceptive claims. To address this challenge, this paper explores the novel task of flaw-oriented fact-checking, including aspect generation and flaw identification. We also introduce RefuteClaim, a new framework designed specifically for this task. Given the absence of an existing dataset, we present FlawCheck, a dataset created by extracting and transforming insights from expert reviews into relevant aspects and identified flaws. The experimental results underscore the efficacy of RefuteClaim, particularly in classifying and elucidating false claims.

{{</citation>}}


### (34/55) Improving Medical Reasoning through Retrieval and Self-Reflection with Retrieval-Augmented Large Language Models (Minbyul Jeong et al., 2024)

{{<citation>}}

Minbyul Jeong, Jiwoong Sohn, Mujeen Sung, Jaewoo Kang. (2024)  
**Improving Medical Reasoning through Retrieval and Self-Reflection with Retrieval-Augmented Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.15269v1)  

---


**ABSTRACT**  
Recent proprietary large language models (LLMs), such as GPT-4, have achieved a milestone in tackling diverse challenges in the biomedical domain, ranging from multiple-choice questions to long-form generations. To address challenges that still cannot be handled with the encoded knowledge of LLMs, various retrieval-augmented generation (RAG) methods have been developed by searching documents from the knowledge corpus and appending them unconditionally or selectively to the input of LLMs for generation. However, when applying existing methods to different domain-specific problems, poor generalization becomes apparent, leading to fetching incorrect documents or making inaccurate judgments. In this paper, we introduce Self-BioRAG, a framework reliable for biomedical text that specializes in generating explanations, retrieving domain-specific documents, and self-reflecting generated responses. We utilize 84k filtered biomedical instruction sets to train Self-BioRAG that can assess its generated explanations with customized reflective tokens. Our work proves that domain-specific components, such as a retriever, domain-related document corpus, and instruction sets are necessary for adhering to domain-related instructions. Using three major medical question-answering benchmark datasets, experimental results of Self-BioRAG demonstrate significant performance gains by achieving a 7.2% absolute improvement on average over the state-of-the-art open-foundation model with a parameter size of 7B or less. Overall, we analyze that Self-BioRAG finds the clues in the question, retrieves relevant documents if needed, and understands how to answer with information from retrieved documents and encoded knowledge as a medical expert does. We release our data and code for training our framework components and model weights (7B and 13B) to enhance capabilities in biomedical and clinical domains.

{{</citation>}}


## cs.HC (1)



### (35/55) 'May I Speak?': Multi-modal Attention Guidance in Social VR Group Conversations (Geonsun Lee et al., 2024)

{{<citation>}}

Geonsun Lee, Dae Yeol Lee, Guan-Ming Su, Dinesh Manocha. (2024)  
**'May I Speak?': Multi-modal Attention Guidance in Social VR Group Conversations**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.15507v1)  

---


**ABSTRACT**  
In this paper, we present a novel multi-modal attention guidance method designed to address the challenges of turn-taking dynamics in meetings and enhance group conversations within virtual reality (VR) environments. Recognizing the difficulties posed by a confined field of view and the absence of detailed gesture tracking in VR, our proposed method aims to mitigate the challenges of noticing new speakers attempting to join the conversation. This approach tailors attention guidance, providing a nuanced experience for highly engaged participants while offering subtler cues for those less engaged, thereby enriching the overall meeting dynamics. Through group interview studies, we gathered insights to guide our design, resulting in a prototype that employs "light" as a diegetic guidance mechanism, complemented by spatial audio. The combination creates an intuitive and immersive meeting environment, effectively directing users' attention to new speakers. An evaluation study, comparing our method to state-of-the-art attention guidance approaches, demonstrated significantly faster response times (p < 0.001), heightened perceived conversation satisfaction (p < 0.001), and preference (p < 0.001) for our method. Our findings contribute to the understanding of design implications for VR social attention guidance, opening avenues for future research and development.

{{</citation>}}


## cs.CY (3)



### (36/55) Foregrounding Artist Opinions: A Survey Study on Transparency, Ownership, and Fairness in AI Generative Art (Juniper Lovato et al., 2024)

{{<citation>}}

Juniper Lovato, Julia Zimmerman, Isabelle Smith, Peter Dodds, Jennifer Karson. (2024)  
**Foregrounding Artist Opinions: A Survey Study on Transparency, Ownership, and Fairness in AI Generative Art**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.15497v2)  

---


**ABSTRACT**  
Generative Artificial Intelligence (AI) tools are used to create art-like outputs and aid in the creative process. While these tools have potential benefits for artists, they also have the potential to harm the art workforce and infringe upon artistic and intellectual property rights. Without explicit consent from artists, Generative AI creators scrape artists' digital work to train Generative AI models and produce art-like model outputs at scale. These outputs are now being used to compete with human artists in the marketplace as well as being used by some artists in their generative processes to create art. We surveyed 459 artists to investigate the tension between artists' opinions on Generative AI art's potential utility and harm. This study surveys artists' opinions on the utility and threat of Generative AI art models, fair practices in the disclosure of artistic works in AI art training models, ownership and rights of AI art derivatives, and fair compensation. We find that artists, by and large, think that model creators should be required to disclose in detail what art and images they use to train their AI models. We also find that artists' opinions vary by professional status and practice, demographics, whether they have purchased art, and familiarity with and use of Generative AI. We hope the results of this work will further more meaningful collaboration and alignment between the art community and Generative AI researchers and developers.

{{</citation>}}


### (37/55) Artificial Intelligence: Arguments for Catastrophic Risk (Adam Bales et al., 2024)

{{<citation>}}

Adam Bales, William D'Alessandro, Cameron Domenico Kirk-Giannini. (2024)  
**Artificial Intelligence: Arguments for Catastrophic Risk**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15487v1)  

---


**ABSTRACT**  
Recent progress in artificial intelligence (AI) has drawn attention to the technology's transformative potential, including what some see as its prospects for causing large-scale harm. We review two influential arguments purporting to show how AI could pose catastrophic risks. The first argument -- the Problem of Power-Seeking -- claims that, under certain assumptions, advanced AI systems are likely to engage in dangerous power-seeking behavior in pursuit of their goals. We review reasons for thinking that AI systems might seek power, that they might obtain it, that this could lead to catastrophe, and that we might build and deploy such systems anyway. The second argument claims that the development of human-level AI will unlock rapid further progress, culminating in AI systems far more capable than any human -- this is the Singularity Hypothesis. Power-seeking behavior on the part of such systems might be particularly dangerous. We discuss a variety of objections to both arguments and conclude by assessing the state of the debate.

{{</citation>}}


### (38/55) Building ethical guidelines for generative AI in scientific research (Zhicheng Lin, 2024)

{{<citation>}}

Zhicheng Lin. (2024)  
**Building ethical guidelines for generative AI in scientific research**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15284v1)  

---


**ABSTRACT**  
Generative artificial intelligence tools like large language models are rapidly transforming academic research and real world applications. However, discussions on ethical guidelines for generative AI in science remain fragmented, underscoring the urgent need for consensus based standards. This paper offers an initial framework by developing analyses and mitigation strategies across five key themes: understanding model limitations regarding truthfulness and bias; respecting privacy, confidentiality, and copyright; avoiding plagiarism and policy violations when incorporating model output; ensuring applications provide overall benefit; and using AI transparently and reproducibly. Common scenarios are outlined to demonstrate potential ethical violations. We argue that global consensus coupled with professional training and reasonable enforcement are critical to promoting the benefits of AI while safeguarding research integrity.

{{</citation>}}


## cs.RO (1)



### (39/55) R$\times$R: Rapid eXploration for Reinforcement Learning via Sampling-based Reset Distributions and Imitation Pre-training (Gagan Khandate et al., 2024)

{{<citation>}}

Gagan Khandate, Tristan L. Saidi, Siqi Shang, Eric T. Chang, Yang Liu, Seth Dennis, Johnson Adams, Matei Ciocarlie. (2024)  
**R$\times$R: Rapid eXploration for Reinforcement Learning via Sampling-based Reset Distributions and Imitation Pre-training**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.15484v1)  

---


**ABSTRACT**  
We present a method for enabling Reinforcement Learning of motor control policies for complex skills such as dexterous manipulation. We posit that a key difficulty for training such policies is the difficulty of exploring the problem state space, as the accessible and useful regions of this space form a complex structure along manifolds of the original high-dimensional state space. This work presents a method to enable and support exploration with Sampling-based Planning. We use a generally applicable non-holonomic Rapidly-exploring Random Trees algorithm and present multiple methods to use the resulting structure to bootstrap model-free Reinforcement Learning. Our method is effective at learning various challenging dexterous motor control skills of higher difficulty than previously shown. In particular, we achieve dexterous in-hand manipulation of complex objects while simultaneously securing the object without the use of passive support surfaces. These policies also transfer effectively to real robots. A number of example videos can also be found on the project website: https://sbrl.cs.columbia.edu

{{</citation>}}


## cs.LG (7)



### (40/55) Social Interpretable Reinforcement Learning (Leonardo Lucio Custode et al., 2024)

{{<citation>}}

Leonardo Lucio Custode, Giovanni Iacca. (2024)  
**Social Interpretable Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: I-2-6; I-2-8, cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.15480v1)  

---


**ABSTRACT**  
Reinforcement Learning (RL) bears the promise of being an enabling technology for many applications. However, since most of the literature in the field is currently focused on opaque models, the use of RL in high-stakes scenarios, where interpretability is crucial, is still limited. Recently, some approaches to interpretable RL, e.g., based on Decision Trees, have been proposed, but one of the main limitations of these techniques is their training cost. To overcome this limitation, we propose a new population-based method, called Social Interpretable RL (SIRL), inspired by social learning principles, to improve learning efficiency. Our method mimics a social learning process, where each agent in a group learns to solve a given task based both on its own individual experience as well as the experience acquired together with its peers. Our approach is divided into two phases. In the \emph{collaborative phase}, all the agents in the population interact with a shared instance of the environment, where each agent observes the state and independently proposes an action. Then, voting is performed to choose the action that will actually be performed in the environment. In the \emph{individual phase}, each agent refines its individual performance by interacting with its own instance of the environment. This mechanism makes the agents experience a larger number of episodes while simultaneously reducing the computational cost of the process. Our results on six well-known benchmarks show that SIRL reaches state-of-the-art performance w.r.t. the alternative interpretable methods from the literature.

{{</citation>}}


### (41/55) Towards Causal Classification: A Comprehensive Study on Graph Neural Networks (Simi Job et al., 2024)

{{<citation>}}

Simi Job, Xiaohui Tao, Taotao Cai, Lin Li, Haoran Xie, Jianming Yong. (2024)  
**Towards Causal Classification: A Comprehensive Study on Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.15444v1)  

---


**ABSTRACT**  
The exploration of Graph Neural Networks (GNNs) for processing graph-structured data has expanded, particularly their potential for causal analysis due to their universal approximation capabilities. Anticipated to significantly enhance common graph-based tasks such as classification and prediction, the development of a causally enhanced GNN framework is yet to be thoroughly investigated. Addressing this shortfall, our study delves into nine benchmark graph classification models, testing their strength and versatility across seven datasets spanning three varied domains to discern the impact of causality on the predictive prowess of GNNs. This research offers a detailed assessment of these models, shedding light on their efficiency, and flexibility in different data environments, and highlighting areas needing advancement. Our findings are instrumental in furthering the understanding and practical application of GNNs in diverse datacentric fields

{{</citation>}}


### (42/55) A Survey on Data Augmentation in Large Model Era (Yue Zhou et al., 2024)

{{<citation>}}

Yue Zhou, Chenlu Guo, Xu Wang, Yi Chang, Yuan Wu. (2024)  
**A Survey on Data Augmentation in Large Model Era**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2401.15422v1)  

---


**ABSTRACT**  
Large models, encompassing large language and diffusion models, have shown exceptional promise in approximating human-level intelligence, garnering significant interest from both academic and industrial spheres. However, the training of these large models necessitates vast quantities of high-quality data, and with continuous updates to these models, the existing reservoir of high-quality data may soon be depleted. This challenge has catalyzed a surge in research focused on data augmentation methods. Leveraging large models, these data augmentation techniques have outperformed traditional approaches. This paper offers an exhaustive review of large model-driven data augmentation methods, adopting a comprehensive perspective. We begin by establishing a classification of relevant studies into three main categories: image augmentation, text augmentation, and paired data augmentation. Following this, we delve into various data post-processing techniques pertinent to large model-based data augmentation. Our discussion then expands to encompass the array of applications for these data augmentation methods within natural language processing, computer vision, and audio signal processing. We proceed to evaluate the successes and limitations of large model-based data augmentation across different scenarios. Concluding our review, we highlight prospective challenges and avenues for future exploration in the field of data augmentation. Our objective is to furnish researchers with critical insights, ultimately contributing to the advancement of more sophisticated large models. We consistently maintain the related open-source materials at: https://github.com/MLGroup-JLU/LLM-data-aug-survey.

{{</citation>}}


### (43/55) FaKnow: A Unified Library for Fake News Detection (Yiyuan Zhu et al., 2024)

{{<citation>}}

Yiyuan Zhu, Yongjun Li, Jialiang Wang, Ming Gao, Jiali Wei. (2024)  
**FaKnow: A Unified Library for Fake News Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2401.16441v1)  

---


**ABSTRACT**  
Over the past years, a large number of fake news detection algorithms based on deep learning have emerged. However, they are often developed under different frameworks, each mandating distinct utilization methodologies, consequently hindering reproducibility. Additionally, a substantial amount of redundancy characterizes the code development of such fake news detection models. To address these concerns, we propose FaKnow, a unified and comprehensive fake news detection algorithm library. It encompasses a variety of widely used fake news detection models, categorized as content-based and social context-based approaches. This library covers the full spectrum of the model training and evaluation process, effectively organizing the data, models, and training procedures within a unified framework. Furthermore, it furnishes a series of auxiliary functionalities and tools, including visualization, and logging. Our work contributes to the standardization and unification of fake news detection research, concurrently facilitating the endeavors of researchers in this field. The open-source code and documentation can be accessed at https://github.com/NPURG/FaKnow and https://faknow.readthedocs.io, respectively.

{{</citation>}}


### (44/55) Adaptive Least Mean Squares Graph Neural Networks and Online Graph Signal Estimation (Yi Yan et al., 2024)

{{<citation>}}

Yi Yan, Changran Peng, Ercan Engin Kuruoglu. (2024)  
**Adaptive Least Mean Squares Graph Neural Networks and Online Graph Signal Estimation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.15304v1)  

---


**ABSTRACT**  
The online prediction of multivariate signals, existing simultaneously in space and time, from noisy partial observations is a fundamental task in numerous applications. We propose an efficient Neural Network architecture for the online estimation of time-varying graph signals named the Adaptive Least Mean Squares Graph Neural Networks (LMS-GNN). LMS-GNN aims to capture the time variation and bridge the cross-space-time interactions under the condition that signals are corrupted by noise and missing values. The LMS-GNN is a combination of adaptive graph filters and Graph Neural Networks (GNN). At each time step, the forward propagation of LMS-GNN is similar to adaptive graph filters where the output is based on the error between the observation and the prediction similar to GNN. The filter coefficients are updated via backpropagation as in GNN. Experimenting on real-world temperature data reveals that our LMS-GNN achieves more accurate online predictions compared to graph-based methods like adaptive graph filters and graph convolutional neural networks.

{{</citation>}}


### (45/55) SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks (Azmine Toushik Wasi et al., 2024)

{{<citation>}}

Azmine Toushik Wasi, MD Shafikul Islam, Adipto Raihan Akib. (2024)  
**SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: I-2-1; I-2-8; E-0; J-2; H-3-7, cs-AI, cs-LG, cs-SY, cs.LG, eess-SY, stat-AP  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.15299v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have gained traction across different domains such as transportation, bio-informatics, language processing, and computer vision. However, there is a noticeable absence of research on applying GNNs to supply chain networks. Supply chain networks are inherently graph-like in structure, making them prime candidates for applying GNN methodologies. This opens up a world of possibilities for optimizing, predicting, and solving even the most complex supply chain problems. A major setback in this approach lies in the absence of real-world benchmark datasets to facilitate the research and resolution of supply chain problems using GNNs. To address the issue, we present a real-world benchmark dataset for temporal tasks, obtained from one of the leading FMCG companies in Bangladesh, focusing on supply chain planning for production purposes. The dataset includes temporal data as node features to enable sales predictions, production planning, and the identification of factory issues. By utilizing this dataset, researchers can employ GNNs to address numerous supply chain problems, thereby advancing the field of supply chain analytics and planning. Source: https://github.com/CIOL-SUST/SupplyGraph

{{</citation>}}


### (46/55) Finite-Time Analysis of On-Policy Heterogeneous Federated Reinforcement Learning (Chenyu Zhang et al., 2024)

{{<citation>}}

Chenyu Zhang, Han Wang, Aritra Mitra, James Anderson. (2024)  
**Finite-Time Analysis of On-Policy Heterogeneous Federated Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.15273v1)  

---


**ABSTRACT**  
Federated reinforcement learning (FRL) has emerged as a promising paradigm for reducing the sample complexity of reinforcement learning tasks by exploiting information from different agents. However, when each agent interacts with a potentially different environment, little to nothing is known theoretically about the non-asymptotic performance of FRL algorithms. The lack of such results can be attributed to various technical challenges and their intricate interplay: Markovian sampling, linear function approximation, multiple local updates to save communication, heterogeneity in the reward functions and transition kernels of the agents' MDPs, and continuous state-action spaces. Moreover, in the on-policy setting, the behavior policies vary with time, further complicating the analysis. In response, we introduce FedSARSA, a novel federated on-policy reinforcement learning scheme, equipped with linear function approximation, to address these challenges and provide a comprehensive finite-time error analysis. Notably, we establish that FedSARSA converges to a policy that is near-optimal for all agents, with the extent of near-optimality proportional to the level of heterogeneity. Furthermore, we prove that FedSARSA leverages agent collaboration to enable linear speedups as the number of agents increases, which holds for both fixed and adaptive step-size configurations.

{{</citation>}}


## cs.IR (1)



### (47/55) Navigating the Post-API Dilemma Search Engine Results Pages Present a Biased View of Social Media Data (Amrit Poudel et al., 2024)

{{<citation>}}

Amrit Poudel, Tim Weninger. (2024)  
**Navigating the Post-API Dilemma Search Engine Results Pages Present a Biased View of Social Media Data**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs-SI, cs.IR  
Keywords: Bias, Google, Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2401.15479v1)  

---


**ABSTRACT**  
Recent decisions to discontinue access to social media APIs are having detrimental effects on Internet research and the field of computational social science as a whole. This lack of access to data has been dubbed the Post-API era of Internet research. Fortunately, popular search engines have the means to crawl, capture, and surface social media data on their Search Engine Results Pages (SERP) if provided the proper search query, and may provide a solution to this dilemma. In the present work we ask: does SERP provide a complete and unbiased sample of social media data? Is SERP a viable alternative to direct API-access? To answer these questions, we perform a comparative analysis between (Google) SERP results and nonsampled data from Reddit and Twitter/X. We find that SERP results are highly biased in favor of popular posts; against political, pornographic, and vulgar posts; are more positive in their sentiment; and have large topical gaps. Overall, we conclude that SERP is not a viable alternative to social media API access.

{{</citation>}}


## cs.SE (2)



### (48/55) Large Language Model for Vulnerability Detection: Emerging Results and Future Directions (Xin Zhou et al., 2024)

{{<citation>}}

Xin Zhou, Ting Zhang, David Lo. (2024)  
**Large Language Model for Vulnerability Detection: Emerging Results and Future Directions**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2401.15468v1)  

---


**ABSTRACT**  
Previous learning-based vulnerability detection methods relied on either medium-sized pre-trained models or smaller neural networks from scratch. Recent advancements in Large Pre-Trained Language Models (LLMs) have showcased remarkable few-shot learning capabilities in various tasks. However, the effectiveness of LLMs in detecting software vulnerabilities is largely unexplored. This paper aims to bridge this gap by exploring how LLMs perform with various prompts, particularly focusing on two state-of-the-art LLMs: GPT-3.5 and GPT-4. Our experimental results showed that GPT-3.5 achieves competitive performance with the prior state-of-the-art vulnerability detection approach and GPT-4 consistently outperformed the state-of-the-art.

{{</citation>}}


### (49/55) Large Language Model as Synthesizer: Fusing Diverse Inputs for Better Automatic Vulnerability Repair (Xin Zhou et al., 2024)

{{<citation>}}

Xin Zhou, Kisub Kim, Bowen Xu, DongGyun Han, David Lo. (2024)  
**Large Language Model as Synthesizer: Fusing Diverse Inputs for Better Automatic Vulnerability Repair**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BLEU, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.15459v1)  

---


**ABSTRACT**  
The advances of deep learning (DL) have paved the way for automatic software vulnerability repair approaches, which effectively learn the mapping from the vulnerable code to the fixed code. Nevertheless, existing DL-based vulnerability repair methods face notable limitations: 1) they struggle to handle lengthy vulnerable code, 2) they treat code as natural language texts, neglecting its inherent structure, and 3) they do not tap into the valuable expert knowledge present in the expert system. To address this, we propose VulMaster, a Transformer-based neural network model that excels at generating vulnerability repairs by comprehensively understanding the entire vulnerable code, irrespective of its length. This model also integrates diverse information, encompassing vulnerable code structures and expert knowledge from the CWE system. We evaluated VulMaster on a real-world C/C++ vulnerability repair dataset comprising 1,754 projects with 5,800 vulnerable functions. The experimental results demonstrated that VulMaster exhibits substantial improvements compared to the learning-based state-of-the-art vulnerability repair approach. Specifically, VulMaster improves the EM, BLEU, and CodeBLEU scores from 10.2\% to 20.0\%, 21.3\% to 29.3\%, and 32.5\% to 40.9\%, respectively.

{{</citation>}}


## cs.AI (1)



### (50/55) A Statistical Framework for Measuring AI Reliance (Ziyang Guo et al., 2024)

{{<citation>}}

Ziyang Guo, Yifan Wu, Jason Hartline, Jessica Hullman. (2024)  
**A Statistical Framework for Measuring AI Reliance**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15356v1)  

---


**ABSTRACT**  
Humans frequently make decisions with the aid of artificially intelligent (AI) systems. A common pattern is for the AI to recommend an action to the human who retains control over the final decision. Researchers have identified ensuring that a human has appropriate reliance on an AI as a critical component of achieving complementary performance. We argue that the current definition of appropriate reliance used in such research lacks formal statistical grounding and can lead to contradictions. We propose a formal definition of reliance, based on statistical decision theory, which separates the concepts of reliance as the probability the decision-maker follows the AI's prediction from challenges a human may face in differentiating the signals and forming accurate beliefs about the situation. Our definition gives rise to a framework that can be used to guide the design and interpretation of studies on human-AI complementarity and reliance. Using recent AI-advised decision making studies from literature, we demonstrate how our framework can be used to separate the loss due to mis-reliance from the loss due to not accurately differentiating the signals. We evaluate these losses by comparing to a baseline and a benchmark for complementary performance defined by the expected payoff achieved by a rational agent facing the same decision task as the behavioral agents.

{{</citation>}}


## cs.CR (2)



### (51/55) L-AutoDA: Leveraging Large Language Models for Automated Decision-based Adversarial Attacks (Ping Guo et al., 2024)

{{<citation>}}

Ping Guo, Fei Liu, Xi Lin, Qingchuan Zhao, Qingfu Zhang. (2024)  
**L-AutoDA: Leveraging Large Language Models for Automated Decision-based Adversarial Attacks**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-CV, cs-LG, cs.CR  
Keywords: AI, Adversarial Attack, Language Model  
[Paper Link](http://arxiv.org/abs/2401.15335v1)  

---


**ABSTRACT**  
In the rapidly evolving field of machine learning, adversarial attacks present a significant challenge to model robustness and security. Decision-based attacks, which only require feedback on the decision of a model rather than detailed probabilities or scores, are particularly insidious and difficult to defend against. This work introduces L-AutoDA (Large Language Model-based Automated Decision-based Adversarial Attacks), a novel approach leveraging the generative capabilities of Large Language Models (LLMs) to automate the design of these attacks. By iteratively interacting with LLMs in an evolutionary framework, L-AutoDA automatically designs competitive attack algorithms efficiently without much human effort. We demonstrate the efficacy of L-AutoDA on CIFAR-10 dataset, showing significant improvements over baseline methods in both success rate and computational efficiency. Our findings underscore the potential of language models as tools for adversarial attack generation and highlight new avenues for the development of robust AI systems.

{{</citation>}}


### (52/55) Where's the 'up'?! A Comprehensive (bottom-up) Study on the Security of Arm Cortex-M Systems (Xi Tan et al., 2024)

{{<citation>}}

Xi Tan, Zheyuan Ma, Sandro Pinto, Le Guan, Ning Zhang, Jun Xu, Zhiqiang Lin, Hongxin Hu, Ziming Zhao. (2024)  
**Where's the 'up'?! A Comprehensive (bottom-up) Study on the Security of Arm Cortex-M Systems**  

---
Primary Category: cs.CR  
Categories: C-0; K-6-5, cs-AR, cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.15289v2)  

---


**ABSTRACT**  
Arm Cortex-M processors are the most widely used 32-bit microcontrollers among embedded and Internetof-Things devices. Despite the widespread usage, there has been little effort in summarizing their hardware security features, characterizing the limitations and vulnerabilities of their hardware and software stack, and systematizing the research on securing these systems. The goals and contributions of this paper are multi-fold. First, we analyze the hardware security limitations and issues of Cortex-M systems. Second, we conducted a deep study of the software stack designed for Cortex-M and revealed its limitations, which is accompanied by an empirical analysis of 1,797 real-world firmware from seven hardware vendors. Third, we categorize the reported bugs in Cortex-M software systems. Finally, we systematize the efforts that aim at securing Cortex-M systems and evaluate them in terms of the protections they offer, run-time performance, required hardware features, etc. Based on the insights, we develop a set of recommendations for the research community and MCU software developers.

{{</citation>}}


## cs.SD (1)



### (53/55) Music Auto-Tagging with Robust Music Representation Learned via Domain Adversarial Training (Haesun Joung et al., 2024)

{{<citation>}}

Haesun Joung, Kyogu Lee. (2024)  
**Music Auto-Tagging with Robust Music Representation Learned via Domain Adversarial Training**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-IR, cs-SD, cs.SD, eess-AS  
Keywords: Adversarial Training, Information Retrieval  
[Paper Link](http://arxiv.org/abs/2401.15323v1)  

---


**ABSTRACT**  
Music auto-tagging is crucial for enhancing music discovery and recommendation. Existing models in Music Information Retrieval (MIR) struggle with real-world noise such as environmental and speech sounds in multimedia content. This study proposes a method inspired by speech-related tasks to enhance music auto-tagging performance in noisy settings. The approach integrates Domain Adversarial Training (DAT) into the music domain, enabling robust music representations that withstand noise. Unlike previous research, this approach involves an additional pretraining phase for the domain classifier, to avoid performance degradation in the subsequent phase. Adding various synthesized noisy music data improves the model's generalization across different noise levels. The proposed architecture demonstrates enhanced performance in music auto-tagging by effectively utilizing unlabeled noisy music data. Additional experiments with supplementary unlabeled data further improves the model's performance, underscoring its robust generalization capabilities and broad applicability.

{{</citation>}}


## physics.ao-ph (1)



### (54/55) A Practical Probabilistic Benchmark for AI Weather Models (Noah D. Brenowitz et al., 2024)

{{<citation>}}

Noah D. Brenowitz, Yair Cohen, Jaideep Pathak, Ankur Mahesh, Boris Bonev, Thorsten Kurth, Dale R. Durran, Peter Harrington, Michael S. Pritchard. (2024)  
**A Practical Probabilistic Benchmark for AI Weather Models**  

---
Primary Category: physics.ao-ph  
Categories: cs-LG, physics-ao-ph, physics.ao-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.15305v1)  

---


**ABSTRACT**  
Since the weather is chaotic, forecasts aim to predict the distribution of future states rather than make a single prediction. Recently, multiple data driven weather models have emerged claiming breakthroughs in skill. However, these have mostly been benchmarked using deterministic skill scores, and little is known about their probabilistic skill. Unfortunately, it is hard to fairly compare AI weather models in a probabilistic sense, since variations in choice of ensemble initialization, definition of state, and noise injection methodology become confounding. Moreover, even obtaining ensemble forecast baselines is a substantial engineering challenge given the data volumes involved. We sidestep both problems by applying a decades-old idea -- lagged ensembles -- whereby an ensemble can be constructed from a moderately-sized library of deterministic forecasts. This allows the first parameter-free intercomparison of leading AI weather models' probabilistic skill against an operational baseline. The results reveal that two leading AI weather models, i.e. GraphCast and Pangu, are tied on the probabilistic CRPS metric even though the former outperforms the latter in deterministic scoring. We also reveal how multiple time-step loss functions, which many data-driven weather models have employed, are counter-productive: they improve deterministic metrics at the cost of increased dissipation, deteriorating probabilistic skill. This is confirmed through ablations applied to a spherical Fourier Neural Operator (SFNO) approach to AI weather forecasting. Separate SFNO ablations modulating effective resolution reveal it has a useful effect on ensemble dispersion relevant to achieving good ensemble calibration. We hope these and forthcoming insights from lagged ensembles can help guide the development of AI weather forecasts and have thus shared the diagnostic code.

{{</citation>}}


## math.ST (1)



### (55/55) Asymptotic Behavior of Adversarial Training Estimator under $\ell_\infty$-Perturbation (Yiling Xie et al., 2024)

{{<citation>}}

Yiling Xie, Xiaoming Huo. (2024)  
**Asymptotic Behavior of Adversarial Training Estimator under $\ell_\infty$-Perturbation**  

---
Primary Category: math.ST  
Categories: cs-LG, math-ST, math.ST, stat-ME, stat-ML, stat-TH  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2401.15262v1)  

---


**ABSTRACT**  
Adversarial training has been proposed to hedge against adversarial attacks in machine learning and statistical models. This paper focuses on adversarial training under $\ell_\infty$-perturbation, which has recently attracted much research attention. The asymptotic behavior of the adversarial training estimator is investigated in the generalized linear model. The results imply that the limiting distribution of the adversarial training estimator under $\ell_\infty$-perturbation could put a positive probability mass at $0$ when the true parameter is $0$, providing a theoretical guarantee of the associated sparsity-recovery ability. Alternatively, a two-step procedure is proposed -- adaptive adversarial training, which could further improve the performance of adversarial training under $\ell_\infty$-perturbation. Specifically, the proposed procedure could achieve asymptotic unbiasedness and variable-selection consistency. Numerical experiments are conducted to show the sparsity-recovery ability of adversarial training under $\ell_\infty$-perturbation and to compare the empirical performance between classic adversarial training and adaptive adversarial training.

{{</citation>}}
