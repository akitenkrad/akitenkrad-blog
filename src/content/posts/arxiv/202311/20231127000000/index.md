---
draft: false
title: "arXiv @ 2023.11.27"
date: 2023-11-27
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.27"
    identifier: arxiv_20231127
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (19)](#cscv-19)
- [cs.LG (12)](#cslg-12)
- [cs.CY (1)](#cscy-1)
- [quant-ph (1)](#quant-ph-1)
- [cs.CL (10)](#cscl-10)
- [cs.RO (1)](#csro-1)
- [cs.DC (1)](#csdc-1)
- [cs.SI (1)](#cssi-1)

## cs.CV (19)



### (1/46) Can SAM recognize crops? Quantifying the zero-shot performance of a semantic segmentation foundation model on generating crop-type maps using satellite imagery for precision agriculture (Rutuja Gurav et al., 2023)

{{<citation>}}

Rutuja Gurav, Het Patel, Zhuocheng Shang, Ahmed Eldawy, Jia Chen, Elia Scudiero, Evangelos Papalexakis. (2023)  
**Can SAM recognize crops? Quantifying the zero-shot performance of a semantic segmentation foundation model on generating crop-type maps using satellite imagery for precision agriculture**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15138v1)  

---


**ABSTRACT**  
Climate change is increasingly disrupting worldwide agriculture, making global food production less reliable.To tackle the growing challenges in feeding the planet, cutting-edge management strategies, such as precision agriculture, empower farmers and decision-makers with rich and actionable information to increase the efficiency and sustainability of their farming practices.Crop-type maps are key information for decision-support tools but are challenging and costly to generate.We investigate the capabilities of Meta AI's Segment Anything Model (SAM) for crop-map prediction task, acknowledging its recent successes at zero-shot image segmentation.However, SAM being limited to up-to 3 channel inputs and its zero-shot usage being class-agnostic in nature pose unique challenges in using it directly for crop-type mapping.We propose using clustering consensus metrics to assess SAM's zero-shot performance in segmenting satellite imagery and producing crop-type maps.Although direct crop-type mapping is challenging using SAM in zero-shot setting, experiments reveal SAM's potential for swiftly and accurately outlining fields in satellite images, serving as a foundation for subsequent crop classification.This paper attempts to highlight a use-case of state-of-the-art image segmentation models like SAM for crop-type mapping and related specific needs of the agriculture industry, offering a potential avenue for automatic, efficient, and cost-effective data products for precision agriculture practices.

{{</citation>}}


### (2/46) Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets (Andreas Blattmann et al., 2023)

{{<citation>}}

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, Robin Rombach. (2023)  
**Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15127v1)  

---


**ABSTRACT**  
We present Stable Video Diffusion - a latent video diffusion model for high-resolution, state-of-the-art text-to-video and image-to-video generation. Recently, latent diffusion models trained for 2D image synthesis have been turned into generative video models by inserting temporal layers and finetuning them on small, high-quality video datasets. However, training methods in the literature vary widely, and the field has yet to agree on a unified strategy for curating video data. In this paper, we identify and evaluate three different stages for successful training of video LDMs: text-to-image pretraining, video pretraining, and high-quality video finetuning. Furthermore, we demonstrate the necessity of a well-curated pretraining dataset for generating high-quality videos and present a systematic curation process to train a strong base model, including captioning and filtering strategies. We then explore the impact of finetuning our base model on high-quality data and train a text-to-video model that is competitive with closed-source video generation. We also show that our base model provides a powerful motion representation for downstream tasks such as image-to-video generation and adaptability to camera motion-specific LoRA modules. Finally, we demonstrate that our model provides a strong multi-view 3D-prior and can serve as a base to finetune a multi-view diffusion model that jointly generates multiple views of objects in a feedforward fashion, outperforming image-based methods at a fraction of their compute budget. We release code and model weights at https://github.com/Stability-AI/generative-models .

{{</citation>}}


### (3/46) SAMv2: A Unified Framework for Learning Appearance, Semantic and Cross-Modality Anatomical Embeddings (Xiaoyu Bai et al., 2023)

{{<citation>}}

Xiaoyu Bai, Fan Bai, Xiaofei Huo, Jia Ge, Jingjing Lu, Xianghua Ye, Ke Yan, Yong Xia. (2023)  
**SAMv2: A Unified Framework for Learning Appearance, Semantic and Cross-Modality Anatomical Embeddings**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.15111v2)  

---


**ABSTRACT**  
Identifying anatomical structures (e.g., lesions or landmarks) in medical images plays a fundamental role in medical image analysis. As an exemplar-based landmark detection method, Self-supervised Anatomical eMbedding (SAM) learns a discriminative embedding for each voxel in the image and has shown promising results on various tasks. However, SAM still faces challenges in: (1) differentiating voxels with similar appearance but different semantic meanings (\textit{e.g.}, two adjacent structures without clear borders); (2) matching voxels with similar semantics but markedly different appearance (e.g., the same vessel before and after contrast injection); and (3) cross-modality matching (e.g., CT-MRI registration). To overcome these challenges, we propose SAMv2, which is a unified framework designed to learn appearance, semantic, and cross-modality anatomical embeddings. Specifically, SAMv2 incorporates three key innovations: (1) semantic embedding learning with prototypical contrastive loss; (2) a fixed-point-based matching strategy; and (3) an iterative approach for cross-modality embedding learning. We thoroughly evaluated SAMv2 across three tasks, including one-shot landmark detection, lesion tracking on longitudinal CT scans, and CT-MRI affine/rigid registration with varying field of view. Our results suggest that SAMv2 outperforms SAM and other state-of-the-art methods, offering a robust and versatile approach for landmark based medical image analysis tasks. Code and trained models are available at: https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2

{{</citation>}}


### (4/46) Leveraging Diffusion Perturbations for Measuring Fairness in Computer Vision (Nicholas Lui et al., 2023)

{{<citation>}}

Nicholas Lui, Bryan Chia, William Berrios, Candace Ross, Douwe Kiela. (2023)  
**Leveraging Diffusion Perturbations for Measuring Fairness in Computer Vision**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2311.15108v1)  

---


**ABSTRACT**  
Computer vision models have been known to encode harmful biases, leading to the potentially unfair treatment of historically marginalized groups, such as people of color. However, there remains a lack of datasets balanced along demographic traits that can be used to evaluate the downstream fairness of these models. In this work, we demonstrate that diffusion models can be leveraged to create such a dataset. We first use a diffusion model to generate a large set of images depicting various occupations. Subsequently, each image is edited using inpainting to generate multiple variants, where each variant refers to a different perceived race. Using this dataset, we benchmark several vision-language models on a multi-class occupation classification task. We find that images generated with non-Caucasian labels have a significantly higher occupation misclassification rate than images generated with Caucasian labels, and that several misclassifications are suggestive of racial biases. We measure a model's downstream fairness by computing the standard deviation in the probability of predicting the true occupation label across the different perceived identity groups. Using this fairness metric, we find significant disparities between the evaluated vision-and-language models. We hope that our work demonstrates the potential value of diffusion methods for fairness evaluations.

{{</citation>}}


### (5/46) RandMSAugment: A Mixed-Sample Augmentation for Limited-Data Scenarios (Swarna Kamlam Ravindran et al., 2023)

{{<citation>}}

Swarna Kamlam Ravindran, Carlo Tomasi. (2023)  
**RandMSAugment: A Mixed-Sample Augmentation for Limited-Data Scenarios**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.16508v1)  

---


**ABSTRACT**  
The high costs of annotating large datasets suggests a need for effectively training CNNs with limited data, and data augmentation is a promising direction. We study foundational augmentation techniques, including Mixed Sample Data Augmentations (MSDAs) and a no-parameter variant of RandAugment termed Preset-RandAugment, in the fully supervised scenario. We observe that Preset-RandAugment excels in limited-data contexts while MSDAs are moderately effective. We show that low-level feature transforms play a pivotal role in this performance difference, postulate a new property of augmentations related to their data efficiency, and propose new ways to measure the diversity and realism of augmentations. Building on these insights, we introduce a novel augmentation technique called RandMSAugment that integrates complementary strengths of existing methods. RandMSAugment significantly outperforms the competition on CIFAR-100, STL-10, and Tiny-Imagenet. With very small training sets (4, 25, 100 samples/class), RandMSAugment achieves compelling performance gains between 4.1% and 6.75%. Even with more training data (500 samples/class) we improve performance by 1.03% to 2.47%. RandMSAugment does not require hyperparameter tuning, extra validation data, or cumbersome optimizations.

{{</citation>}}


### (6/46) Double-Flow-based Steganography without Embedding for Image-to-Image Hiding (Bingbing Song et al., 2023)

{{<citation>}}

Bingbing Song, Derui Wang, Tianwei Zhang, Renyang Liu, Yu Lin, Wei Zhou. (2023)  
**Double-Flow-based Steganography without Embedding for Image-to-Image Hiding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.15027v1)  

---


**ABSTRACT**  
As an emerging concept, steganography without embedding (SWE) hides a secret message without directly embedding it into a cover. Thus, SWE has the unique advantage of being immune to typical steganalysis methods and can better protect the secret message from being exposed. However, existing SWE methods are generally criticized for their poor payload capacity and low fidelity of recovered secret messages. In this paper, we propose a novel steganography-without-embedding technique, named DF-SWE, which addresses the aforementioned drawbacks and produces diverse and natural stego images. Specifically, DF-SWE employs a reversible circulation of double flow to build a reversible bijective transformation between the secret image and the generated stego image. Hence, it provides a way to directly generate stego images from secret images without a cover image. Besides leveraging the invertible property, DF-SWE can invert a secret image from a generated stego image in a nearly lossless manner and increases the fidelity of extracted secret images. To the best of our knowledge, DF-SWE is the first SWE method that can hide large images and multiple images into one image with the same size, significantly enhancing the payload capacity. According to the experimental results, the payload capacity of DF-SWE achieves 24-72 BPP is 8000-16000 times compared to its competitors while producing diverse images to minimize the exposure risk. Importantly, DF-SWE can be applied in the steganography of secret images in various domains without requiring training data from the corresponding domains. This domain-agnostic property suggests that DF-SWE can 1) be applied to hiding private data and 2) be deployed in resource-limited systems.

{{</citation>}}


### (7/46) Occlusion Sensitivity Analysis with Augmentation Subspace Perturbation in Deep Feature Space (Pedro Valois et al., 2023)

{{<citation>}}

Pedro Valois, Koichiro Niinuma, Kazuhiro Fukui. (2023)  
**Occlusion Sensitivity Analysis with Augmentation Subspace Perturbation in Deep Feature Space**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Augmentation, ImageNet  
[Paper Link](http://arxiv.org/abs/2311.15022v1)  

---


**ABSTRACT**  
Deep Learning of neural networks has gained prominence in multiple life-critical applications like medical diagnoses and autonomous vehicle accident investigations. However, concerns about model transparency and biases persist. Explainable methods are viewed as the solution to address these challenges. In this study, we introduce the Occlusion Sensitivity Analysis with Deep Feature Augmentation Subspace (OSA-DAS), a novel perturbation-based interpretability approach for computer vision. While traditional perturbation methods make only use of occlusions to explain the model predictions, OSA-DAS extends standard occlusion sensitivity analysis by enabling the integration with diverse image augmentations. Distinctly, our method utilizes the output vector of a DNN to build low-dimensional subspaces within the deep feature vector space, offering a more precise explanation of the model prediction. The structural similarity between these subspaces encompasses the influence of diverse augmentations and occlusions. We test extensively on the ImageNet-1k, and our class- and model-agnostic approach outperforms commonly used interpreters, setting it apart in the realm of explainable AI.

{{</citation>}}


### (8/46) VSCode: General Visual Salient and Camouflaged Object Detection with 2D Prompt Learning (Ziyang Luo et al., 2023)

{{<citation>}}

Ziyang Luo, Nian Liu, Wangbo Zhao, Xuguang Yang, Dingwen Zhang, Deng-Ping Fan, Fahad Khan, Junwei Han. (2023)  
**VSCode: General Visual Salient and Camouflaged Object Detection with 2D Prompt Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.15011v1)  

---


**ABSTRACT**  
Salient object detection (SOD) and camouflaged object detection (COD) are related yet distinct binary mapping tasks. These tasks involve multiple modalities, sharing commonalities and unique cues. Existing research often employs intricate task-specific specialist models, potentially leading to redundancy and suboptimal results. We introduce VSCode, a generalist model with novel 2D prompt learning, to jointly address four SOD tasks and three COD tasks. We utilize VST as the foundation model and introduce 2D prompts within the encoder-decoder architecture to learn domain and task-specific knowledge on two separate dimensions. A prompt discrimination loss helps disentangle peculiarities to benefit model optimization. VSCode outperforms state-of-the-art methods across six tasks on 26 datasets and exhibits zero-shot generalization to unseen tasks by combining 2D prompts, such as RGB-D COD.

{{</citation>}}


### (9/46) $Z^*$: Zero-shot Style Transfer via Attention Rearrangement (Yingying Deng et al., 2023)

{{<citation>}}

Yingying Deng, Xiangyu He, Fan Tang, Weiming Dong. (2023)  
**$Z^*$: Zero-shot Style Transfer via Attention Rearrangement**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Style Transfer  
[Paper Link](http://arxiv.org/abs/2311.16491v1)  

---


**ABSTRACT**  
Despite the remarkable progress in image style transfer, formulating style in the context of art is inherently subjective and challenging. In contrast to existing learning/tuning methods, this study shows that vanilla diffusion models can directly extract style information and seamlessly integrate the generative prior into the content image without retraining. Specifically, we adopt dual denoising paths to represent content/style references in latent space and then guide the content image denoising process with style latent codes. We further reveal that the cross-attention mechanism in latent diffusion models tends to blend the content and style images, resulting in stylized outputs that deviate from the original content image. To overcome this limitation, we introduce a cross-attention rearrangement strategy. Through theoretical analysis and experiments, we demonstrate the effectiveness and superiority of the diffusion-based $\underline{Z}$ero-shot $\underline{S}$tyle $\underline{T}$ransfer via $\underline{A}$ttention $\underline{R}$earrangement, Z-STAR.

{{</citation>}}


### (10/46) Elucidating and Overcoming the Challenges of Label Noise in Supervised Contrastive Learning (Zijun Long et al., 2023)

{{<citation>}}

Zijun Long, George Killick, Lipeng Zhuang, Richard McCreadie, Gerardo Aragon Camarasa, Paul Henderson. (2023)  
**Elucidating and Overcoming the Challenges of Label Noise in Supervised Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.16481v1)  

---


**ABSTRACT**  
Image classification datasets exhibit a non-negligible fraction of mislabeled examples, often due to human error when one class superficially resembles another. This issue poses challenges in supervised contrastive learning (SCL), where the goal is to cluster together data points of the same class in the embedding space while distancing those of disparate classes. While such methods outperform those based on cross-entropy, they are not immune to labeling errors. However, while the detrimental effects of noisy labels in supervised learning are well-researched, their influence on SCL remains largely unexplored. Hence, we analyse the effect of label errors and examine how they disrupt the SCL algorithm's ability to distinguish between positive and negative sample pairs. Our analysis reveals that human labeling errors manifest as easy positive samples in around 99% of cases. We, therefore, propose D-SCL, a novel Debiased Supervised Contrastive Learning objective designed to mitigate the bias introduced by labeling errors. We demonstrate that D-SCL consistently outperforms state-of-the-art techniques for representation learning across diverse vision benchmarks, offering improved robustness to label errors.

{{</citation>}}


### (11/46) Point Cloud Pre-training with Diffusion Models (Xiao Zheng et al., 2023)

{{<citation>}}

Xiao Zheng, Xiaoshui Huang, Guofeng Mei, Yuenan Hou, Zhaoyang Lyu, Bo Dai, Wanli Ouyang, Yongshun Gong. (2023)  
**Point Cloud Pre-training with Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.14960v1)  

---


**ABSTRACT**  
Pre-training a model and then fine-tuning it on downstream tasks has demonstrated significant success in the 2D image and NLP domains. However, due to the unordered and non-uniform density characteristics of point clouds, it is non-trivial to explore the prior knowledge of point clouds and pre-train a point cloud backbone. In this paper, we propose a novel pre-training method called Point cloud Diffusion pre-training (PointDif). We consider the point cloud pre-training task as a conditional point-to-point generation problem and introduce a conditional point generator. This generator aggregates the features extracted by the backbone and employs them as the condition to guide the point-to-point recovery from the noisy point cloud, thereby assisting the backbone in capturing both local and global geometric priors as well as the global point density distribution of the object. We also present a recurrent uniform sampling optimization strategy, which enables the model to uniformly recover from various noise levels and learn from balanced supervision. Our PointDif achieves substantial improvement across various real-world datasets for diverse downstream tasks such as classification, segmentation and detection. Specifically, PointDif attains 70.0% mIoU on S3DIS Area 5 for the segmentation task and achieves an average improvement of 2.4% on ScanObjectNN for the classification task compared to TAP. Furthermore, our pre-training framework can be flexibly applied to diverse point cloud backbones and bring considerable gains.

{{</citation>}}


### (12/46) OpenNet: Incremental Learning for Autonomous Driving Object Detection with Balanced Loss (Zezhou Wang et al., 2023)

{{<citation>}}

Zezhou Wang, Guitao Cao, Xidong Xi, Jiangtao Wang. (2023)  
**OpenNet: Incremental Learning for Autonomous Driving Object Detection with Balanced Loss**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.14939v1)  

---


**ABSTRACT**  
Automated driving object detection has always been a challenging task in computer vision due to environmental uncertainties. These uncertainties include significant differences in object sizes and encountering the class unseen. It may result in poor performance when traditional object detection models are directly applied to automated driving detection. Because they usually presume fixed categories of common traffic participants, such as pedestrians and cars. Worsely, the huge class imbalance between common and novel classes further exacerbates performance degradation. To address the issues stated, we propose OpenNet to moderate the class imbalance with the Balanced Loss, which is based on Cross Entropy Loss. Besides, we adopt an inductive layer based on gradient reshaping to fast learn new classes with limited samples during incremental learning. To against catastrophic forgetting, we employ normalized feature distillation. By the way, we improve multi-scale detection robustness and unknown class recognition through FPN and energy-based detection, respectively. The Experimental results upon the CODA dataset show that the proposed method can obtain better performance than that of the existing methods.

{{</citation>}}


### (13/46) FreePIH: Training-Free Painterly Image Harmonization with Diffusion Model (Ruibin Li et al., 2023)

{{<citation>}}

Ruibin Li, Jingcai Guo, Song Guo, Qihua Zhou, Jie Zhang. (2023)  
**FreePIH: Training-Free Painterly Image Harmonization with Diffusion Model**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14926v1)  

---


**ABSTRACT**  
This paper provides an efficient training-free painterly image harmonization (PIH) method, dubbed FreePIH, that leverages only a pre-trained diffusion model to achieve state-of-the-art harmonization results. Unlike existing methods that require either training auxiliary networks or fine-tuning a large pre-trained backbone, or both, to harmonize a foreground object with a painterly-style background image, our FreePIH tames the denoising process as a plug-in module for foreground image style transfer. Specifically, we find that the very last few steps of the denoising (i.e., generation) process strongly correspond to the stylistic information of images, and based on this, we propose to augment the latent features of both the foreground and background images with Gaussians for a direct denoising-based harmonization. To guarantee the fidelity of the harmonized image, we make use of multi-scale features to enforce the consistency of the content and stability of the foreground objects in the latent space, and meanwhile, aligning both fore-/back-grounds with the same style. Moreover, to accommodate the generation with more structural and textural details, we further integrate text prompts to attend to the latent features, hence improving the generation quality. Quantitative and qualitative evaluations on COCO and LAION 5B datasets demonstrate that our method can surpass representative baselines by large margins.

{{</citation>}}


### (14/46) GPT4Video: A Unified Multimodal Large Language Model for lnstruction-Followed Understanding and Safety-Aware Generation (Zhanyu Wang et al., 2023)

{{<citation>}}

Zhanyu Wang, Longyue Wang, Zhen Zhao, Minghao Wu, Chenyang Lyu, Huayang Li, Deng Cai, Luping Zhou, Shuming Shi, Zhaopeng Tu. (2023)  
**GPT4Video: A Unified Multimodal Large Language Model for lnstruction-Followed Understanding and Safety-Aware Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Language Model, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.16511v1)  

---


**ABSTRACT**  
While the recent advances in Multimodal Large Language Models (MLLMs) constitute a significant leap forward in the field, these models are predominantly confined to the realm of input-side multimodal comprehension, lacking the capacity for multimodal content generation. To fill this gap, we present GPT4Video, a unified multi-model framework that empowers Large Language Models (LLMs) with the capability of both video understanding and generation. Specifically, we develop an instruction-following-based approach integrated with the stable diffusion generative model, which has demonstrated to effectively and securely handle video generation scenarios. GPT4Video offers the following benefits: 1) It exhibits impressive capabilities in both video understanding and generation scenarios. For example, GPT4Video outperforms Valley by 11.8\% on the Video Question Answering task, and surpasses NExt-GPT by 2.3\% on the Text to Video generation task. 2) it endows the LLM/MLLM with video generation capabilities without requiring additional training parameters and can flexibly interface with a wide range of models to perform video generation. 3) it maintains a safe and healthy conversation not only in output-side but also the input side in an end-to-end manner. Qualitative and qualitative experiments demonstrate that GPT4Video holds the potential to function as a effective, safe and Humanoid-like video assistant that can handle both video understanding and generation scenarios.

{{</citation>}}


### (15/46) CUCL: Codebook for Unsupervised Continual Learning (Chen Cheng et al., 2023)

{{<citation>}}

Chen Cheng, Jingkuan Song, Xiaosu Zhu, Junchen Zhu, Lianli Gao, Hengtao Shen. (2023)  
**CUCL: Codebook for Unsupervised Continual Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Quantization  
[Paper Link](http://arxiv.org/abs/2311.14911v1)  

---


**ABSTRACT**  
The focus of this study is on Unsupervised Continual Learning (UCL), as it presents an alternative to Supervised Continual Learning which needs high-quality manual labeled data. The experiments under the UCL paradigm indicate a phenomenon where the results on the first few tasks are suboptimal. This phenomenon can render the model inappropriate for practical applications. To address this issue, after analyzing the phenomenon and identifying the lack of diversity as a vital factor, we propose a method named Codebook for Unsupervised Continual Learning (CUCL) which promotes the model to learn discriminative features to complete the class boundary. Specifically, we first introduce a Product Quantization to inject diversity into the representation and apply a cross quantized contrastive loss between the original representation and the quantized one to capture discriminative information. Then, based on the quantizer, we propose an effective Codebook Rehearsal to address catastrophic forgetting. This study involves conducting extensive experiments on CIFAR100, TinyImageNet, and MiniImageNet benchmark datasets. Our method significantly boosts the performances of supervised and unsupervised methods. For instance, on TinyImageNet, our method led to a relative improvement of 12.76% and 7% when compared with Simsiam and BYOL, respectively.

{{</citation>}}


### (16/46) AutoEval-Video: An Automatic Benchmark for Assessing Large Vision Language Models in Open-Ended Video Question Answering (Xiuyuan Chen et al., 2023)

{{<citation>}}

Xiuyuan Chen, Yuan Lin, Yuchen Zhang, Weiran Huang. (2023)  
**AutoEval-Video: An Automatic Benchmark for Assessing Large Vision Language Models in Open-Ended Video Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.14906v1)  

---


**ABSTRACT**  
We propose a novel and challenging benchmark, AutoEval-Video, to comprehensively evaluate large vision-language models in open-ended video question answering. The comprehensiveness of AutoEval-Video is demonstrated in two aspects: 1) AutoEval-Video constructs open-ended video-questions across 9 skill dimensions, addressing capabilities of perception, comprehension, and generation. 2) AutoEval-Video contains newly collected videos that cover over 40 distinct themes. To efficiently evaluate responses to the open-ended questions, we employ an LLM-based evaluation approach, but instead of merely providing a reference answer, we annotate unique evaluation rules for every single instance (video-question pair). To maximize the robustness of these rules, we develop a novel adversarial annotation mechanism. By using instance-specific rules as prompt, GPT-4, as an automatic evaluator, can achieve a stable evaluation accuracy of around 97.0\%, comparable to the 94.9\% - 97.5\% accuracy of a human evaluator. Furthermore, we assess the performance of eight large vision-language models on AutoEval-Video. Among them, GPT-4V(ision) significantly outperforms other models, achieving an accuracy of 32.2\%. However, there is still substantial room for improvement compared to human accuracy of 72.8\%. By conducting an extensive case study, we uncover several drawbacks of GPT-4V, such as limited temporal and dynamic comprehension, and overly general responses. Code is available at \href{https://github.com/Xiuyuan-Chen/AutoEval-Video}{\color{magenta}https://github.com/Xiuyuan-Chen/AutoEval-Video}.

{{</citation>}}


### (17/46) Parkinson Disease classification Using Contrastive Graph Cross-View Learning with Multimodal Fusion of SPECT Images and Clinical Features (Jun-En Ding et al., 2023)

{{<citation>}}

Jun-En Ding, Chien-Chin Hsu, Feng Liu. (2023)  
**Parkinson Disease classification Using Contrastive Graph Cross-View Learning with Multimodal Fusion of SPECT Images and Clinical Features**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2311.14902v1)  

---


**ABSTRACT**  
Parkinson's Disease (PD) is a neurodegenerative neurological disorder that impacts movement and afflicts over 10 million people worldwide. Previous researches have come up with deep learning models for predicting Parkinson's disease primarily using medical images and didn't leverage the manifold structure in the dataset. Our study introduces a multimodal approach with both image and non-image features with a contrastive cross-view graph fusion for Parkinson's disease classification. Specifically, we designed a multimodal co-attention module to integrate embeddings from two distinct graph views derived from low dimensional representation of images and clinical features, enabling the extraction of more stable and structured features from the multiview data. Additionally, we have devised a simplified fusion method utilizing a contrastive loss for positive and negative pairs, to enhance the model's overall cross-view fusion learning capabilities. In our experiments, the graph-view multimodal approach can achieve an accuracy rate of 91% and an AUC of 92.8% in five-fold cross-validation, and it also demonstrates superior predictive capabilities on non-image data as compared to methods that rely solely on machine learning methods.

{{</citation>}}


### (18/46) HyperDID: Hyperspectral Intrinsic Image Decomposition with Deep Feature Embedding (Zhiqiang Gong et al., 2023)

{{<citation>}}

Zhiqiang Gong, Xian Zhou, Wen Yao, Xiaohu Zheng, Ping Zhong. (2023)  
**HyperDID: Hyperspectral Intrinsic Image Decomposition with Deep Feature Embedding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.14899v1)  

---


**ABSTRACT**  
The dissection of hyperspectral images into intrinsic components through hyperspectral intrinsic image decomposition (HIID) enhances the interpretability of hyperspectral data, providing a foundation for more accurate classification outcomes. However, the classification performance of HIID is constrained by the model's representational ability. To address this limitation, this study rethinks hyperspectral intrinsic image decomposition for classification tasks by introducing deep feature embedding. The proposed framework, HyperDID, incorporates the Environmental Feature Module (EFM) and Categorical Feature Module (CFM) to extract intrinsic features. Additionally, a Feature Discrimination Module (FDM) is introduced to separate environment-related and category-related features. Experimental results across three commonly used datasets validate the effectiveness of HyperDID in improving hyperspectral image classification performance. This novel approach holds promise for advancing the capabilities of hyperspectral image analysis by leveraging deep feature embedding principles. The implementation of the proposed method could be accessed soon at https://github.com/shendu-sw/HyperDID for the sake of reproducibility.

{{</citation>}}


### (19/46) Towards Scalable 3D Anomaly Detection and Localization: A Benchmark via 3D Anomaly Synthesis and A Self-Supervised Learning Network (Wenqiao Li et al., 2023)

{{<citation>}}

Wenqiao Li, Xiaohao Xu. (2023)  
**Towards Scalable 3D Anomaly Detection and Localization: A Benchmark via 3D Anomaly Synthesis and A Self-Supervised Learning Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.14897v1)  

---


**ABSTRACT**  
Recently, 3D anomaly detection, a crucial problem involving fine-grained geometry discrimination, is getting more attention. However, the lack of abundant real 3D anomaly data limits the scalability of current models. To enable scalable anomaly data collection, we propose a 3D anomaly synthesis pipeline to adapt existing large-scale 3Dmodels for 3D anomaly detection. Specifically, we construct a synthetic dataset, i.e., Anomaly-ShapeNet, basedon ShapeNet. Anomaly-ShapeNet consists of 1600 point cloud samples under 40 categories, which provides a rich and varied collection of data, enabling efficient training and enhancing adaptability to industrial scenarios. Meanwhile,to enable scalable representation learning for 3D anomaly localization, we propose a self-supervised method, i.e., Iterative Mask Reconstruction Network (IMRNet). During training, we propose a geometry-aware sample module to preserve potentially anomalous local regions during point cloud down-sampling. Then, we randomly mask out point patches and sent the visible patches to a transformer for reconstruction-based self-supervision. During testing, the point cloud repeatedly goes through the Mask Reconstruction Network, with each iteration's output becoming the next input. By merging and contrasting the final reconstructed point cloud with the initial input, our method successfully locates anomalies. Experiments show that IMRNet outperforms previous state-of-the-art methods, achieving 66.1% in I-AUC on Anomaly-ShapeNet dataset and 72.5% in I-AUC on Real3D-AD dataset. Our dataset will be released at https://github.com/Chopper-233/Anomaly-ShapeNet

{{</citation>}}


## cs.LG (12)



### (20/46) SwiftLearn: A Data-Efficient Training Method of Deep Learning Models using Importance Sampling (Habib Hajimolahoseini et al., 2023)

{{<citation>}}

Habib Hajimolahoseini, Omar Mohamed Awad, Walid Ahmed, Austin Wen, Saina Asani, Mohammad Hassanpour, Farnoosh Javadi, Mehdi Ahmadi, Foozhan Ataiefard, Kangling Liu, Yang Liu. (2023)  
**SwiftLearn: A Data-Efficient Training Method of Deep Learning Models using Importance Sampling**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: BERT, GLUE, NLP  
[Paper Link](http://arxiv.org/abs/2311.15134v1)  

---


**ABSTRACT**  
In this paper, we present SwiftLearn, a data-efficient approach to accelerate training of deep learning models using a subset of data samples selected during the warm-up stages of training. This subset is selected based on an importance criteria measured over the entire dataset during warm-up stages, aiming to preserve the model performance with fewer examples during the rest of training. The importance measure we propose could be updated during training every once in a while, to make sure that all of the data samples have a chance to return to the training loop if they show a higher importance. The model architecture is unchanged but since the number of data samples controls the number of forward and backward passes during training, we can reduce the training time by reducing the number of training samples used in each epoch of training. Experimental results on a variety of CV and NLP models during both pretraining and finetuning show that the model performance could be preserved while achieving a significant speed-up during training. More specifically, BERT finetuning on GLUE benchmark shows that almost 90% of the data can be dropped achieving an end-to-end average speedup of 3.36x while keeping the average accuracy drop less than 0.92%.

{{</citation>}}


### (21/46) Localizing Lying in Llama: Understanding Instructed Dishonesty on True-False Questions Through Prompting, Probing, and Patching (James Campbell et al., 2023)

{{<citation>}}

James Campbell, Richard Ren, Phillip Guo. (2023)  
**Localizing Lying in Llama: Understanding Instructed Dishonesty on True-False Questions Through Prompting, Probing, and Patching**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2311.15131v1)  

---


**ABSTRACT**  
Large language models (LLMs) demonstrate significant knowledge through their outputs, though it is often unclear whether false outputs are due to a lack of knowledge or dishonesty. In this paper, we investigate instructed dishonesty, wherein we explicitly prompt LLaMA-2-70b-chat to lie. We perform prompt engineering to find which prompts best induce lying behavior, and then use mechanistic interpretability approaches to localize where in the network this behavior occurs. Using linear probing and activation patching, we localize five layers that appear especially important for lying. We then find just 46 attention heads within these layers that enable us to causally intervene such that the lying model instead answers honestly. We show that these interventions work robustly across many prompts and dataset splits. Overall, our work contributes a greater understanding of dishonesty in LLMs so that we may hope to prevent it.

{{</citation>}}


### (22/46) Everybody Needs a Little HELP: Explaining Graphs via Hierarchical Concepts (Jonas Jürß et al., 2023)

{{<citation>}}

Jonas Jürß, Lucie Charlotte Magister, Pietro Barbiero, Pietro Liò, Nikola Simidjievski. (2023)  
**Everybody Needs a Little HELP: Explaining Graphs via Hierarchical Concepts**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.15112v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have led to major breakthroughs in a variety of domains such as drug discovery, social network analysis, and travel time estimation. However, they lack interpretability which hinders human trust and thereby deployment to settings with high-stakes decisions. A line of interpretable methods approach this by discovering a small set of relevant concepts as subgraphs in the last GNN layer that together explain the prediction. This can yield oversimplified explanations, failing to explain the interaction between GNN layers. To address this oversight, we provide HELP (Hierarchical Explainable Latent Pooling), a novel, inherently interpretable graph pooling approach that reveals how concepts from different GNN layers compose to new ones in later steps. HELP is more than 1-WL expressive and is the first non-spectral, end-to-end-learnable, hierarchical graph pooling method that can learn to pool a variable number of arbitrary connected components. We empirically demonstrate that it performs on-par with standard GCNs and popular pooling methods in terms of accuracy while yielding explanations that are aligned with expert knowledge in the domains of chemistry and social networks. In addition to a qualitative analysis, we employ concept completeness scores as well as concept conformity, a novel metric to measure the noise in discovered concepts, quantitatively verifying that the discovered concepts are significantly easier to fully understand than those from previous work. Our work represents a first step towards an understanding of graph neural networks that goes beyond a set of concepts from the final layer and instead explains the complex interplay of concepts on different levels.

{{</citation>}}


### (23/46) Enhancing Sentiment Analysis Results through Outlier Detection Optimization (Yuetian Chen et al., 2023)

{{<citation>}}

Yuetian Chen, Mei Si. (2023)  
**Enhancing Sentiment Analysis Results through Outlier Detection Optimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: BERT, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2311.16185v1)  

---


**ABSTRACT**  
When dealing with text data containing subjective labels like speaker emotions, inaccuracies or discrepancies among labelers are not uncommon. Such discrepancies can significantly affect the performance of machine learning algorithms. This study investigates the potential of identifying and addressing outliers in text data with subjective labels, aiming to enhance classification outcomes. We utilized the Deep SVDD algorithm, a one-class classification method, to detect outliers in nine text-based emotion and sentiment analysis datasets. By employing both a small-sized language model (DistilBERT base model with 66 million parameters) and non-deep learning machine learning algorithms (decision tree, KNN, Logistic Regression, and LDA) as the classifier, our findings suggest that the removal of outliers can lead to enhanced results in most cases. Additionally, as outliers in such datasets are not necessarily unlearnable, we experienced utilizing a large language model -- DeBERTa v3 large with 131 million parameters, which can capture very complex patterns in data. We continued to observe performance enhancements across multiple datasets.

{{</citation>}}


### (24/46) Where2Start: Leveraging initial States for Robust and Sample-Efficient Reinforcement Learning (Pouya Parsa et al., 2023)

{{<citation>}}

Pouya Parsa, Raoof Zare Moayedi, Mohammad Bornosi, Mohammad Mahdi Bejani. (2023)  
**Where2Start: Leveraging initial States for Robust and Sample-Efficient Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.15089v1)  

---


**ABSTRACT**  
The reinforcement learning algorithms that focus on how to compute the gradient and choose next actions, are effectively improved the performance of the agents. However, these algorithms are environment-agnostic. This means that the algorithms did not use the knowledge that has been captured by trajectory. This poses that the algorithms should sample many trajectories to train the model. By considering the essence of environment and how much the agent learn from each scenario in that environment, the strategy of the learning procedure can be changed. The strategy retrieves more informative trajectories, so the agent can learn with fewer trajectory sample. We propose Where2Start algorithm that selects the initial state so that the agent has more instability in vicinity of that state. We show that this kind of selection decreases number of trajectories that should be sampled that the agent reach to acceptable reward. Our experiments shows that Where2Start can improve sample efficiency up to 8 times. Also Where2Start can combined with most of state-of-the-art algorithms and improve that robustness and sample efficiency significantly.

{{</citation>}}


### (25/46) Training a Hopfield Variational Autoencoder with Equilibrium Propagation (Tom Van Der Meersch et al., 2023)

{{<citation>}}

Tom Van Der Meersch, Johannes Deleu, Thomas Demeester. (2023)  
**Training a Hopfield Variational Autoencoder with Equilibrium Propagation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15047v1)  

---


**ABSTRACT**  
On dedicated analog hardware, equilibrium propagation is an energy-efficient alternative to backpropagation. In spite of its theoretical guarantees, its application in the AI domain remains limited to the discriminative setting. Meanwhile, despite its high computational demands, generative AI is on the rise. In this paper, we demonstrate the application of Equilibrium Propagation in training a variational autoencoder (VAE) for generative modeling. Leveraging the symmetric nature of Hopfield networks, we propose using a single model to serve as both the encoder and decoder which could effectively halve the required chip size for VAE implementations, paving the way for more efficient analog hardware configurations.

{{</citation>}}


### (26/46) On-Device Soft Sensors: Real-Time Fluid Flow Estimation from Level Sensor Data (Tianheng Ling et al., 2023)

{{<citation>}}

Tianheng Ling, Chao Qian, Gregor Schiele. (2023)  
**On-Device Soft Sensors: Real-Time Fluid Flow Estimation from Level Sensor Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15036v1)  

---


**ABSTRACT**  
Soft sensors are crucial in bridging autonomous systems' physical and digital realms, enhancing sensor fusion and perception. Instead of deploying soft sensors on the Cloud, this study shift towards employing on-device soft sensors, promising heightened efficiency and bolstering data security. Our approach substantially improves energy efficiency by deploying Artificial Intelligence (AI) directly on devices within a wireless sensor network. Furthermore, the synergistic integration of the Microcontroller Unit and Field-Programmable Gate Array (FPGA) leverages the rapid AI inference capabilities of the latter. Empirical evidence from our real-world use case demonstrates that FPGA-based soft sensors achieve inference times ranging remarkably from 1.04 to 12.04 microseconds. These compelling results highlight the considerable potential of our innovative approach for executing real-time inference tasks efficiently, thereby presenting a feasible alternative that effectively addresses the latency challenges intrinsic to Cloud-based deployments.

{{</citation>}}


### (27/46) Exploring Causal Learning through Graph Neural Networks: An In-depth Review (Simi Job et al., 2023)

{{<citation>}}

Simi Job, Xiaohui Tao, Taotao Cai, Haoran Xie, Lin Li, Jianming Yong, Qing Li. (2023)  
**Exploring Causal Learning through Graph Neural Networks: An In-depth Review**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.14994v1)  

---


**ABSTRACT**  
In machine learning, exploring data correlations to predict outcomes is a fundamental task. Recognizing causal relationships embedded within data is pivotal for a comprehensive understanding of system dynamics, the significance of which is paramount in data-driven decision-making processes. Beyond traditional methods, there has been a surge in the use of graph neural networks (GNNs) for causal learning, given their capabilities as universal data approximators. Thus, a thorough review of the advancements in causal learning using GNNs is both relevant and timely. To structure this review, we introduce a novel taxonomy that encompasses various state-of-the-art GNN methods employed in studying causality. GNNs are further categorized based on their applications in the causality domain. We further provide an exhaustive compilation of datasets integral to causal learning with GNNs to serve as a resource for practical study. This review also touches upon the application of causal learning across diverse sectors. We conclude the review with insights into potential challenges and promising avenues for future exploration in this rapidly evolving field of machine learning.

{{</citation>}}


### (28/46) Eliminating Domain Bias for Federated Learning in Representation Space (Jianqing Zhang et al., 2023)

{{<citation>}}

Jianqing Zhang, Yang Hua, Jian Cao, Hao Wang, Tao Song, Zhengui Xue, Ruhui Ma, Haibing Guan. (2023)  
**Eliminating Domain Bias for Federated Learning in Representation Space**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.14975v1)  

---


**ABSTRACT**  
Recently, federated learning (FL) is popular for its privacy-preserving and collaborative learning abilities. However, under statistically heterogeneous scenarios, we observe that biased data domains on clients cause a representation bias phenomenon and further degenerate generic representations during local training, i.e., the representation degeneration phenomenon. To address these issues, we propose a general framework Domain Bias Eliminator (DBE) for FL. Our theoretical analysis reveals that DBE can promote bi-directional knowledge transfer between server and client, as it reduces the domain discrepancy between server and client in representation space. Besides, extensive experiments on four datasets show that DBE can greatly improve existing FL methods in both generalization and personalization abilities. The DBE-equipped FL method can outperform ten state-of-the-art personalized FL methods by a large margin. Our code is public at https://github.com/TsingZ0/DBE.

{{</citation>}}


### (29/46) Robust Graph Neural Networks via Unbiased Aggregation (Ruiqi Feng et al., 2023)

{{<citation>}}

Ruiqi Feng, Zhichao Hou, Tyler Derr, Xiaorui Liu. (2023)  
**Robust Graph Neural Networks via Unbiased Aggregation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.14934v1)  

---


**ABSTRACT**  
The adversarial robustness of Graph Neural Networks (GNNs) has been questioned due to the false sense of security uncovered by strong adaptive attacks despite the existence of numerous defenses. In this work, we delve into the robustness analysis of representative robust GNNs and provide a unified robust estimation point of view to understand their robustness and limitations. Our novel analysis of estimation bias motivates the design of a robust and unbiased graph signal estimator. We then develop an efficient Quasi-Newton iterative reweighted least squares algorithm to solve the estimation problem, which unfolds as robust unbiased aggregation layers in GNNs with a theoretical convergence guarantee. Our comprehensive experiments confirm the strong robustness of our proposed model, and the ablation study provides a deep understanding of its advantages.

{{</citation>}}


### (30/46) Aiming to Minimize Alcohol-Impaired Road Fatalities: Utilizing Fairness-Aware and Domain Knowledge-Infused Artificial Intelligence (Tejas Venkateswaran et al., 2023)

{{<citation>}}

Tejas Venkateswaran, Sheikh Rabiul Islam, Md Golam Moula Mehedi Hasan, Mohiuddin Ahmed. (2023)  
**Aiming to Minimize Alcohol-Impaired Road Fatalities: Utilizing Fairness-Aware and Domain Knowledge-Infused Artificial Intelligence**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.16180v1)  

---


**ABSTRACT**  
Approximately 30% of all traffic fatalities in the United States are attributed to alcohol-impaired driving. This means that, despite stringent laws against this offense in every state, the frequency of drunk driving accidents is alarming, resulting in approximately one person being killed every 45 minutes. The process of charging individuals with Driving Under the Influence (DUI) is intricate and can sometimes be subjective, involving multiple stages such as observing the vehicle in motion, interacting with the driver, and conducting Standardized Field Sobriety Tests (SFSTs). Biases have been observed through racial profiling, leading to some groups and geographical areas facing fewer DUI tests, resulting in many actual DUI incidents going undetected, ultimately leading to a higher number of fatalities. To tackle this issue, our research introduces an Artificial Intelligence-based predictor that is both fairness-aware and incorporates domain knowledge to analyze DUI-related fatalities in different geographic locations. Through this model, we gain intriguing insights into the interplay between various demographic groups, including age, race, and income. By utilizing the provided information to allocate policing resources in a more equitable and efficient manner, there is potential to reduce DUI-related fatalities and have a significant impact on road safety.

{{</citation>}}


### (31/46) Projected Off-Policy Q-Learning (POP-QL) for Stabilizing Offline Reinforcement Learning (Melrose Roderick et al., 2023)

{{<citation>}}

Melrose Roderick, Gaurav Manek, Felix Berkenkamp, J. Zico Kolter. (2023)  
**Projected Off-Policy Q-Learning (POP-QL) for Stabilizing Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.14885v1)  

---


**ABSTRACT**  
A key problem in off-policy Reinforcement Learning (RL) is the mismatch, or distribution shift, between the dataset and the distribution over states and actions visited by the learned policy. This problem is exacerbated in the fully offline setting. The main approach to correct this shift has been through importance sampling, which leads to high-variance gradients. Other approaches, such as conservatism or behavior-regularization, regularize the policy at the cost of performance. In this paper, we propose a new approach for stable off-policy Q-Learning. Our method, Projected Off-Policy Q-Learning (POP-QL), is a novel actor-critic algorithm that simultaneously reweights off-policy samples and constrains the policy to prevent divergence and reduce value-approximation error. In our experiments, POP-QL not only shows competitive performance on standard benchmarks, but also out-performs competing methods in tasks where the data-collection policy is significantly sub-optimal.

{{</citation>}}


## cs.CY (1)



### (32/46) Using Assignment Incentives to Reduce Student Procrastination and Encourage Code Review Interactions (Kevin Wang et al., 2023)

{{<citation>}}

Kevin Wang, Ramon Lawrence. (2023)  
**Using Assignment Incentives to Reduce Student Procrastination and Encourage Code Review Interactions**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.15125v1)  

---


**ABSTRACT**  
Procrastination causes student stress, reduced learning and performance, and results in very busy help sessions immediately before deadlines. A key challenge is encouraging students to complete assignments earlier rather than waiting until right before the deadline, so the focus becomes on the learning objectives rather than just meeting deadlines. This work presents an incentive system encouraging students to complete assignments many days before deadlines. Completed assignments are code reviewed by staff for correctness and providing feedback, which results in more student-instructor interactions and may help reduce student use of generative AI. The incentives result in a change in student behavior with 45% of assignments completed early and 30% up to 4 days before the deadline. Students receive real-time feedback with no increase in marking time.

{{</citation>}}


## quant-ph (1)



### (33/46) FPQA-C: A Compilation Framework for Field Programmable Qubit Array (Hanrui Wang et al., 2023)

{{<citation>}}

Hanrui Wang, Pengyu Liu, Bochen Tan, Yilian Liu, Jiaqi Gu, David Z. Pan, Jason Cong, Umut Acar, Song Han. (2023)  
**FPQA-C: A Compilation Framework for Field Programmable Qubit Array**  

---
Primary Category: quant-ph  
Categories: cs-AR, cs-DC, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.15123v1)  

---


**ABSTRACT**  
The neutral atom array has gained prominence in quantum computing for its scalability and operation fidelity. Previous works focus on \textit{fixed} atom arrays (FAA) that necessitate extensive SWAP operations for long-range interactions. This work explores a novel architecture known as \textit{field programmable qubit array (FPQA)}, which uniquely allows for coherent atom movements during circuit execution and significantly \textit{reduces the cost of long-range interactions}. However, the atom movements have multiple hardware constraints, making movement scheduling very challenging.   In this work, we introduce FPQA-C, a compilation framework tailored for qubit mapping, atom movement, and gate scheduling of FPQA. It contains a qubit-array mapper to decide the coarse-grained mapping of qubit to arrays, leveraging MAX k-Cut on a constructed gate frequency graph to minimize SWAP overhead. Subsequently, a qubit-atom mapper determines the fine-grained mapping of qubits to specific atoms in the array, and considers load balance to prevent hardware constraint violations. We further propose a high-parallelism router that iteratively identifies parallelizable 2Q gates and decide the atom movements and gate executions, thus improving the parallelism. Besides, for fault-tolerant computing with FPQA, we provide comprehensive simulations evaluating logical error rates, execution times, physical qubit requirements, code distances, and bandwidth.   We rigorously assess FPQA-C across 20+ diverse benchmarks, including generic circuits (arbitrary, QASMBench, SupermarQ), Quantum Simulation, and QAOA circuits. FPQA-C consistently outperforms the IBM Superconducting, FAA with long-range gates, FAA with rectangular and triangular topologies, achieving 2Q gate reductions by factors of 5.3x, 3.2x, 3.4x, and 2.6x, and circuit depth reductions by factors of 3.6x, 3.2x, 3.1x, and 2.2x, respectively.

{{</citation>}}


## cs.CL (10)



### (34/46) Relevance feedback strategies for recall-oriented neural information retrieval (Timo Kats et al., 2023)

{{<citation>}}

Timo Kats, Peter van der Putten, Jan Scholtes. (2023)  
**Relevance feedback strategies for recall-oriented neural information retrieval**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2311.15110v1)  

---


**ABSTRACT**  
In a number of information retrieval applications (e.g., patent search, literature review, due diligence, etc.), preventing false negatives is more important than preventing false positives. However, approaches designed to reduce review effort (like "technology assisted review") can create false negatives, since they are often based on active learning systems that exclude documents automatically based on user feedback. Therefore, this research proposes a more recall-oriented approach to reducing review effort. More specifically, through iteratively re-ranking the relevance rankings based on user feedback, which is also referred to as relevance feedback. In our proposed method, the relevance rankings are produced by a BERT-based dense-vector search and the relevance feedback is based on cumulatively summing the queried and selected embeddings. Our results show that this method can reduce review effort between 17.85% and 59.04%, compared to a baseline approach (of no feedback), given a fixed recall target

{{</citation>}}


### (35/46) Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion (Bernal Jimenez Gutierrez et al., 2023)

{{<citation>}}

Bernal Jimenez Gutierrez, Yuqing Mao, Vinh Nguyen, Kin Wah Fung, Yu Su, Olivier Bodenreider. (2023)  
**Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.15106v1)  

---


**ABSTRACT**  
As the immense opportunities enabled by large language models become more apparent, NLP systems will be increasingly expected to excel in real-world settings. However, in many instances, powerful models alone will not yield translational NLP solutions, especially if the formulated problem is not well aligned with the real-world task. In this work, we study the case of UMLS vocabulary insertion, an important real-world task in which hundreds of thousands of new terms, referred to as atoms, are added to the UMLS, one of the most comprehensive open-source biomedical knowledge bases. Previous work aimed to develop an automated NLP system to make this time-consuming, costly, and error-prone task more efficient. Nevertheless, practical progress in this direction has been difficult to achieve due to a problem formulation and evaluation gap between research output and the real-world task. In order to address this gap, we introduce a new formulation for UMLS vocabulary insertion which mirrors the real-world task, datasets which faithfully represent it and several strong baselines we developed through re-purposing existing solutions. Additionally, we propose an effective rule-enhanced biomedical language model which enables important new model behavior, outperforms all strong baselines and provides measurable qualitative improvements to editors who carry out the UVI task. We hope this case study provides insight into the considerable importance of problem formulation for the success of translational NLP solutions.

{{</citation>}}


### (36/46) Multilingual self-supervised speech representations improve the speech recognition of low-resource African languages with codeswitching (Tolúlopé Ògúnrèmí et al., 2023)

{{<citation>}}

Tolúlopé Ògúnrèmí, Christopher D. Manning, Dan Jurafsky. (2023)  
**Multilingual self-supervised speech representations improve the speech recognition of low-resource African languages with codeswitching**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2311.15077v1)  

---


**ABSTRACT**  
While many speakers of low-resource languages regularly code-switch between their languages and other regional languages or English, datasets of codeswitched speech are too small to train bespoke acoustic models from scratch or do language model rescoring. Here we propose finetuning self-supervised speech representations such as wav2vec 2.0 XLSR to recognize code-switched data. We find that finetuning self-supervised multilingual representations and augmenting them with n-gram language models trained from transcripts reduces absolute word error rates by up to 20% compared to baselines of hybrid models trained from scratch on code-switched data. Our findings suggest that in circumstances with limited training data finetuning self-supervised representations is a better performing and viable solution.

{{</citation>}}


### (37/46) nlpBDpatriots at BLP-2023 Task 2: A Transfer Learning Approach to Bangla Sentiment Analysis (Dhiman Goswami et al., 2023)

{{<citation>}}

Dhiman Goswami, Md Nishat Raihan, Sadiya Sayara Chowdhury Puspo, Marcos Zampieri. (2023)  
**nlpBDpatriots at BLP-2023 Task 2: A Transfer Learning Approach to Bangla Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Sentiment Analysis, Social Media  
[Paper Link](http://arxiv.org/abs/2311.15032v1)  

---


**ABSTRACT**  
In this paper, we discuss the nlpBDpatriots entry to the shared task on Sentiment Analysis of Bangla Social Media Posts organized at the first workshop on Bangla Language Processing (BLP) co-located with EMNLP. The main objective of this task is to identify the polarity of social media content using a Bangla dataset annotated with positive, neutral, and negative labels provided by the shared task organizers. Our best system for this task is a transfer learning approach with data augmentation which achieved a micro F1 score of 0.71. Our best system ranked 12th among 30 teams that participated in the competition.

{{</citation>}}


### (38/46) nlpBDpatriots at BLP-2023 Task 1: A Two-Step Classification for Violence Inciting Text Detection in Bangla (Md Nishat Raihan et al., 2023)

{{<citation>}}

Md Nishat Raihan, Dhiman Goswami, Sadiya Sayara Chowdhury Puspo, Marcos Zampieri. (2023)  
**nlpBDpatriots at BLP-2023 Task 1: A Two-Step Classification for Violence Inciting Text Detection in Bangla**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.15029v1)  

---


**ABSTRACT**  
In this paper, we discuss the nlpBDpatriots entry to the shared task on Violence Inciting Text Detection (VITD) organized as part of the first workshop on Bangla Language Processing (BLP) co-located with EMNLP. The aim of this task is to identify and classify the violent threats, that provoke further unlawful violent acts. Our best-performing approach for the task is two-step classification using back translation and multilinguality which ranked 6th out of 27 teams with a macro F1 score of 0.74.

{{</citation>}}


### (39/46) Offensive Language Identification in Transliterated and Code-Mixed Bangla (Md Nishat Raihan et al., 2023)

{{<citation>}}

Md Nishat Raihan, Umma Hani Tanmoy, Anika Binte Islam, Kai North, Tharindu Ranasinghe, Antonios Anastasopoulos, Marcos Zampieri. (2023)  
**Offensive Language Identification in Transliterated and Code-Mixed Bangla**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Identification, NLP  
[Paper Link](http://arxiv.org/abs/2311.15023v1)  

---


**ABSTRACT**  
Identifying offensive content in social media is vital for creating safe online communities. Several recent studies have addressed this problem by creating datasets for various languages. In this paper, we explore offensive language identification in texts with transliterations and code-mixing, linguistic phenomena common in multilingual societies, and a known challenge for NLP systems. We introduce TB-OLID, a transliterated Bangla offensive language dataset containing 5,000 manually annotated comments. We train and fine-tune machine learning models on TB-OLID, and we evaluate their results on this dataset. Our results show that English pre-trained transformer-based models, such as fBERT and HateBERT achieve the best performance on this dataset.

{{</citation>}}


### (40/46) E-CORE: Emotion Correlation Enhanced Empathetic Dialogue Generation (Fengyi Fu et al., 2023)

{{<citation>}}

Fengyi Fu, Lei Zhang, Quan Wang, Zhendong Mao. (2023)  
**E-CORE: Emotion Correlation Enhanced Empathetic Dialogue Generation**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2311.15016v1)  

---


**ABSTRACT**  
Achieving empathy is a crucial step toward humanized dialogue systems. Current approaches for empathetic dialogue generation mainly perceive an emotional label to generate an empathetic response conditioned on it, which simply treat emotions independently, but ignore the intrinsic emotion correlation in dialogues, resulting in inaccurate emotion perception and unsuitable response generation. In this paper, we propose a novel emotion correlation enhanced empathetic dialogue generation framework, which comprehensively realizes emotion correlation learning, utilization, and supervising. Specifically, a multi-resolution emotion graph is devised to capture context-based emotion interactions from different resolutions, further modeling emotion correlation. Then we propose an emotion correlation enhanced decoder, with a novel correlation-aware aggregation and soft/hard strategy, respectively improving the emotion perception and response generation. Experimental results on the benchmark dataset demonstrate the superiority of our model in both empathetic perception and expression.

{{</citation>}}


### (41/46) Walking a Tightrope -- Evaluating Large Language Models in High-Risk Domains (Chia-Chien Hung et al., 2023)

{{<citation>}}

Chia-Chien Hung, Wiem Ben Rim, Lindsay Frost, Lars Bruckner, Carolin Lawrence. (2023)  
**Walking a Tightrope -- Evaluating Large Language Models in High-Risk Domains**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.14966v1)  

---


**ABSTRACT**  
High-risk domains pose unique challenges that require language models to provide accurate and safe responses. Despite the great success of large language models (LLMs), such as ChatGPT and its variants, their performance in high-risk domains remains unclear. Our study delves into an in-depth analysis of the performance of instruction-tuned LLMs, focusing on factual accuracy and safety adherence. To comprehensively assess the capabilities of LLMs, we conduct experiments on six NLP datasets including question answering and summarization tasks within two high-risk domains: legal and medical. Further qualitative analysis highlights the existing limitations inherent in current LLMs when evaluating in high-risk domains. This underscores the essential nature of not only improving LLM capabilities but also prioritizing the refinement of domain-specific metrics, and embracing a more human-centric approach to enhance safety and factual reliability. Our findings advance the field toward the concerns of properly evaluating LLMs in high-risk domains, aiming to steer the adaptability of LLMs in fulfilling societal obligations and aligning with forthcoming regulations, such as the EU AI Act.

{{</citation>}}


### (42/46) Faster Minimum Bayes Risk Decoding with Confidence-based Pruning (Julius Cheng et al., 2023)

{{<citation>}}

Julius Cheng, Andreas Vlachos. (2023)  
**Faster Minimum Bayes Risk Decoding with Confidence-based Pruning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2311.14919v1)  

---


**ABSTRACT**  
Minimum Bayes risk (MBR) decoding outputs the hypothesis with the highest expected utility over the model distribution for some utility function. It has been shown to improve accuracy over beam search in conditional language generation problems and especially neural machine translation, in both human and automatic evaluations. However, the standard sampling-based algorithm for MBR is substantially more computationally expensive than beam search, requiring a large number of samples as well as a quadratic number of calls to the utility function, limiting its applicability. We describe an algorithm for MBR which gradually grows the number of samples used to estimate the utility while pruning hypotheses that are unlikely to have the highest utility according to confidence estimates obtained with bootstrap sampling. Our method requires fewer samples and drastically reduces the number of calls to the utility function compared to standard MBR while being statistically indistinguishable in terms of accuracy. We demonstrate the effectiveness of our approach in experiments on three language pairs, using chrF++ and COMET as utility/evaluation metrics.

{{</citation>}}


### (43/46) Code Search Debiasing:Improve Search Results beyond Overall Ranking Performance (Sheng Zhang et al., 2023)

{{<citation>}}

Sheng Zhang, Hui Li, Yanlin Wang, Zhao Wei, Yong Xiu, Juhong Wang, Rongong Ji. (2023)  
**Code Search Debiasing:Improve Search Results beyond Overall Ranking Performance**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.14901v1)  

---


**ABSTRACT**  
Code search engine is an essential tool in software development. Many code search methods have sprung up, focusing on the overall ranking performance of code search. In this paper, we study code search from another perspective by analyzing the bias of code search models. Biased code search engines provide poor user experience, even though they show promising overall performance. Due to different development conventions (e.g., prefer long queries or abbreviations), some programmers will find the engine useful, while others may find it hard to get desirable search results. To mitigate biases, we develop a general debiasing framework that employs reranking to calibrate search results. It can be easily plugged into existing engines and handle new code search biases discovered in the future. Experiments show that our framework can effectively reduce biases. Meanwhile, the overall ranking performance of code search gets improved after debiasing.

{{</citation>}}


## cs.RO (1)



### (44/46) Agent as Cerebrum, Controller as Cerebellum: Implementing an Embodied LMM-based Agent on Drones (Haoran Zhao et al., 2023)

{{<citation>}}

Haoran Zhao, Fengxing Pan, Huqiuyue Ping, Yaoming Zhou. (2023)  
**Agent as Cerebrum, Controller as Cerebellum: Implementing an Embodied LMM-based Agent on Drones**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Drone, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.15033v1)  

---


**ABSTRACT**  
In this study, we present a novel paradigm for industrial robotic embodied agents, encapsulating an 'agent as cerebrum, controller as cerebellum' architecture. Our approach harnesses the power of Large Multimodal Models (LMMs) within an agent framework known as AeroAgent, tailored for drone technology in industrial settings. To facilitate seamless integration with robotic systems, we introduce ROSchain, a bespoke linkage framework connecting LMM-based agents to the Robot Operating System (ROS). We report findings from extensive empirical research, including simulated experiments on the Airgen and real-world case study, particularly in individual search and rescue operations. The results demonstrate AeroAgent's superior performance in comparison to existing Deep Reinforcement Learning (DRL)-based agents, highlighting the advantages of the embodied LMM in complex, real-world scenarios.

{{</citation>}}


## cs.DC (1)



### (45/46) HongTu: Scalable Full-Graph GNN Training on Multiple GPUs (via communication-optimized CPU data offloading) (Qiange Wang et al., 2023)

{{<citation>}}

Qiange Wang, Yao Chen, Weng-Fai Wong, Bingsheng He. (2023)  
**HongTu: Scalable Full-Graph GNN Training on Multiple GPUs (via communication-optimized CPU data offloading)**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.14898v1)  

---


**ABSTRACT**  
Full-graph training on graph neural networks (GNN) has emerged as a promising training method for its effectiveness. Full-graph training requires extensive memory and computation resources. To accelerate this training process, researchers have proposed employing multi-GPU processing. However the scalability of existing frameworks is limited as they necessitate maintaining the training data for every layer in GPU memory. To efficiently train on large graphs, we present HongTu, a scalable full-graph GNN training system running on GPU-accelerated platforms. HongTu stores vertex data in CPU memory and offloads training to GPUs. HongTu employs a memory-efficient full-graph training framework that reduces runtime memory consumption by using partition-based training and recomputation-caching-hybrid intermediate data management. To address the issue of increased host-GPU communication caused by duplicated neighbor access among partitions, HongTu employs a deduplicated communication framework that converts the redundant host-GPU communication to efficient inter/intra-GPU data access. Further, HongTu uses a cost model-guided graph reorganization method to minimize communication overhead. Experimental results on a 4XA100 GPU server show that HongTu effectively supports billion-scale full-graph GNN training while reducing host-GPU data communication by 25%-71%. Compared to the full-graph GNN system DistGNN running on 16 CPU nodes, HongTu achieves speedups ranging from 7.8X to 20.2X. For small graphs where the training data fits into the GPUs, HongTu achieves performance comparable to existing GPU-based GNN systems.

{{</citation>}}


## cs.SI (1)



### (46/46) Predicting Potential School Shooters from Social Media Posts (Alana Cedeno et al., 2023)

{{<citation>}}

Alana Cedeno, Rachel Liang, Sheikh Rabiul Islam. (2023)  
**Predicting Potential School Shooters from Social Media Posts**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2311.14883v1)  

---


**ABSTRACT**  
The rate of terror attacks has surged over the past decade, resulting in the tragic and senseless loss or alteration of numerous lives. Offenders behind mass shootings, bombings, or other domestic terrorism incidents have historically exhibited warning signs on social media before carrying out actual incidents. However, due to inadequate and comprehensive police procedures, authorities and social media platforms are often unable to detect these early indicators of intent. To tackle this issue, we aim to create a multimodal model capable of predicting sentiments simultaneously from both images (i.e., social media photos) and text (i.e., social media posts), generating a unified prediction. The proposed method involves segregating the image and text components of an online post and utilizing a captioning model to generate sentences summarizing the image's contents. Subsequently, a sentiment analyzer evaluates this caption, or description, along with the original post's text to determine whether the post is positive (i.e., concerning) or negative (i.e., benign). This undertaking represents a significant step toward implementing the developed system in real-world scenarios.

{{</citation>}}
