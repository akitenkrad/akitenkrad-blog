---
draft: false
title: "arXiv @ 2023.11.26"
date: 2023-11-26
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.26"
    identifier: arxiv_20231126
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (25)](#cscv-25)
- [cs.LG (15)](#cslg-15)
- [cs.GT (1)](#csgt-1)
- [cs.PF (1)](#cspf-1)
- [cs.CL (8)](#cscl-8)
- [cs.CE (1)](#csce-1)
- [cs.AI (8)](#csai-8)
- [stat.ML (1)](#statml-1)
- [cs.DB (1)](#csdb-1)
- [cs.NI (3)](#csni-3)
- [cs.SD (1)](#cssd-1)
- [cs.CR (2)](#cscr-2)
- [eess.IV (2)](#eessiv-2)
- [cs.RO (2)](#csro-2)
- [eess.SY (1)](#eesssy-1)
- [cs.CY (1)](#cscy-1)
- [eess.SP (1)](#eesssp-1)
- [cs.SI (1)](#cssi-1)

## cs.CV (25)



### (1/75) Understanding Self-Supervised Features for Learning Unsupervised Instance Segmentation (Paul Engstler et al., 2023)

{{<citation>}}

Paul Engstler, Luke Melas-Kyriazi, Christian Rupprecht, Iro Laina. (2023)  
**Understanding Self-Supervised Features for Learning Unsupervised Instance Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.14665v1)  

---


**ABSTRACT**  
Self-supervised learning (SSL) can be used to solve complex visual tasks without human labels. Self-supervised representations encode useful semantic information about images, and as a result, they have already been used for tasks such as unsupervised semantic segmentation. In this paper, we investigate self-supervised representations for instance segmentation without any manual annotations. We find that the features of different SSL methods vary in their level of instance-awareness. In particular, DINO features, which are known to be excellent semantic descriptors, lack behind MAE features in their sensitivity for separating instances.

{{</citation>}}


### (2/75) Charting New Territories: Exploring the Geographic and Geospatial Capabilities of Multimodal LLMs (Jonathan Roberts et al., 2023)

{{<citation>}}

Jonathan Roberts, Timo Lüddecke, Rehan Sheikh, Kai Han, Samuel Albanie. (2023)  
**Charting New Territories: Exploring the Geographic and Geospatial Capabilities of Multimodal LLMs**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.14656v1)  

---


**ABSTRACT**  
Multimodal large language models (MLLMs) have shown remarkable capabilities across a broad range of tasks but their knowledge and abilities in the geographic and geospatial domains are yet to be explored, despite potential wide-ranging benefits to navigation, environmental research, urban development, and disaster response. We conduct a series of experiments exploring various vision capabilities of MLLMs within these domains, particularly focusing on the frontier model GPT-4V, and benchmark its performance against open-source counterparts. Our methodology involves challenging these models with a small-scale geographic benchmark consisting of a suite of visual tasks, testing their abilities across a spectrum of complexity. The analysis uncovers not only where such models excel, including instances where they outperform humans, but also where they falter, providing a balanced view of their capabilities in the geographic domain. To enable the comparison and evaluation of future models, our benchmark will be publicly released.

{{</citation>}}


### (3/75) CatVersion: Concatenating Embeddings for Diffusion-Based Text-to-Image Personalization (Ruoyu Zhao et al., 2023)

{{<citation>}}

Ruoyu Zhao, Mingrui Zhu, Shiyin Dong, Nannan Wang, Xinbo Gao. (2023)  
**CatVersion: Concatenating Embeddings for Diffusion-Based Text-to-Image Personalization**  

---
Primary Category: cs.CV  
Categories: 68U05, I-3-3, cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.14631v1)  

---


**ABSTRACT**  
We propose CatVersion, an inversion-based method that learns the personalized concept through a handful of examples. Subsequently, users can utilize text prompts to generate images that embody the personalized concept, thereby achieving text-to-image personalization. In contrast to existing approaches that emphasize word embedding learning or parameter fine-tuning for the diffusion model, which potentially causes concept dilution or overfitting, our method concatenates embeddings on the feature-dense space of the text encoder in the diffusion model to learn the gap between the personalized concept and its base class, aiming to maximize the preservation of prior knowledge in diffusion models while restoring the personalized concepts. To this end, we first dissect the text encoder's integration in the image generation process to identify the feature-dense space of the encoder. Afterward, we concatenate embeddings on the Keys and Values in this space to learn the gap between the personalized concept and its base class. In this way, the concatenated embeddings ultimately manifest as a residual on the original attention output. To more accurately and unbiasedly quantify the results of personalized image generation, we improve the CLIP image alignment score based on masks. Qualitatively and quantitatively, CatVersion helps to restore personalization concepts more faithfully and enables more robust editing.

{{</citation>}}


### (4/75) ARIA: On the interaction between Architectures, Aggregation methods and Initializations in federated visual classification (Vasilis Siomos et al., 2023)

{{<citation>}}

Vasilis Siomos, Sergio Naval-Marimont, Jonathan Passerat-Palmbach, Giacomo Tarroni. (2023)  
**ARIA: On the interaction between Architectures, Aggregation methods and Initializations in federated visual classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-DC, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.14625v1)  

---


**ABSTRACT**  
Federated Learning (FL) is a collaborative training paradigm that allows for privacy-preserving learning of cross-institutional models by eliminating the exchange of sensitive data and instead relying on the exchange of model parameters between the clients and a server. Despite individual studies on how client models are aggregated, and, more recently, on the benefits of ImageNet pre-training, there is a lack of understanding of the effect the architecture chosen for the federation has, and of how the aforementioned elements interconnect. To this end, we conduct the first joint ARchitecture-Initialization-Aggregation study and benchmark ARIAs across a range of medical image classification tasks. We find that, contrary to current practices, ARIA elements have to be chosen together to achieve the best possible performance. Our results also shed light on good choices for each element depending on the task, the effect of normalisation layers, and the utility of SSL pre-training, pointing to potential directions for designing FL-specific architectures and training pipelines.

{{</citation>}}


### (5/75) Neural Style Transfer for Computer Games (Eleftherios Ioannou et al., 2023)

{{<citation>}}

Eleftherios Ioannou, Steve Maddock. (2023)  
**Neural Style Transfer for Computer Games**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2311.14617v1)  

---


**ABSTRACT**  
Neural Style Transfer (NST) research has been applied to images, videos, 3D meshes and radiance fields, but its application to 3D computer games remains relatively unexplored. Whilst image and video NST systems can be used as a post-processing effect for a computer game, this results in undesired artefacts and diminished post-processing effects. Here, we present an approach for injecting depth-aware NST as part of the 3D rendering pipeline. Qualitative and quantitative experiments are used to validate our in-game stylisation framework. We demonstrate temporally consistent results of artistically stylised game scenes, outperforming state-of-the-art image and video NST methods.

{{</citation>}}


### (6/75) Large Language Models as Automated Aligners for benchmarking Vision-Language Models (Yuanfeng Ji et al., 2023)

{{<citation>}}

Yuanfeng Ji, Chongjian Ge, Weikai Kong, Enze Xie, Zhengying Liu, Zhengguo Li, Ping Luo. (2023)  
**Large Language Models as Automated Aligners for benchmarking Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.14580v1)  

---


**ABSTRACT**  
With the advancements in Large Language Models (LLMs), Vision-Language Models (VLMs) have reached a new level of sophistication, showing notable competence in executing intricate cognition and reasoning tasks. However, existing evaluation benchmarks, primarily relying on rigid, hand-crafted datasets to measure task-specific performance, face significant limitations in assessing the alignment of these increasingly anthropomorphic models with human intelligence. In this work, we address the limitations via Auto-Bench, which delves into exploring LLMs as proficient aligners, measuring the alignment between VLMs and human intelligence and value through automatic data curation and assessment. Specifically, for data curation, Auto-Bench utilizes LLMs (e.g., GPT-4) to automatically generate a vast set of question-answer-reasoning triplets via prompting on visual symbolic representations (e.g., captions, object locations, instance relationships, and etc.). The curated data closely matches human intent, owing to the extensive world knowledge embedded in LLMs. Through this pipeline, a total of 28.5K human-verified and 3,504K unfiltered question-answer-reasoning triplets have been curated, covering 4 primary abilities and 16 sub-abilities. We subsequently engage LLMs like GPT-3.5 to serve as judges, implementing the quantitative and qualitative automated assessments to facilitate a comprehensive evaluation of VLMs. Our validation results reveal that LLMs are proficient in both evaluation data curation and model assessment, achieving an average agreement rate of 85%. We envision Auto-Bench as a flexible, scalable, and comprehensive benchmark for evaluating the evolving sophisticated VLMs.

{{</citation>}}


### (7/75) Griffon: Spelling out All Object Locations at Any Granularity with Large Language Models (Yufei Zhan et al., 2023)

{{<citation>}}

Yufei Zhan, Yousong Zhu, Zhiyang Chen, Fan Yang, Ming Tang, Jinqiao Wang. (2023)  
**Griffon: Spelling out All Object Locations at Any Granularity with Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.14552v1)  

---


**ABSTRACT**  
Replicating the innate human ability to detect all objects based on free-form texts at any granularity remains a formidable challenge for Vision-Language models. Current Large Vision Language Models (LVLMs) are predominantly constrained to grounding a single, pre-existing object, relying solely on data from Referring Expression Comprehension tasks. The limitation leads to a compromise in model design, necessitating the introduction of visual expert models or the integration of customized head structures. Beyond these constraints, our research delves into the untapped potential of LVLMs and uncover their inherent capability for basic object perception, allowing them to accurately identify and locate objects of interest. Building on this insight, we introduce a novel language-prompted localization dataset designed to fully unleash the capabilities of LVLMs in integrating fine-grained object perception with precise location awareness. More importantly, we present $\textbf{Griffon}$, a purely LVLM-based baseline, which does not require the introduction of any special tokens, expert models, or additional detection modules. It simply maintains a consistent structure with popular LVLMs by unifying data formats across various localization-related scenarios and is trained end-to-end through a well-designed pipeline. Comprehensive experiments demonstrate that $\textbf{Griffon}$ not only achieves state-of-the-art performance on the fine-grained RefCOCO series but also approaches the capabilities of the expert model Faster RCNN on the detection benchmark MSCOCO.

{{</citation>}}


### (8/75) Inferring Latent Class Statistics from Text for Robust Visual Few-Shot Learning (Yassir Bendou et al., 2023)

{{<citation>}}

Yassir Bendou, Vincent Gripon, Bastien Pasdeloup, Giulia Lioi, Lukas Mauch, Fabien Cardinaux, Ghouthi Boukli Hacene. (2023)  
**Inferring Latent Class Statistics from Text for Robust Visual Few-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2311.14544v1)  

---


**ABSTRACT**  
In the realm of few-shot learning, foundation models like CLIP have proven effective but exhibit limitations in cross-domain robustness especially in few-shot settings. Recent works add text as an extra modality to enhance the performance of these models. Most of these approaches treat text as an auxiliary modality without fully exploring its potential to elucidate the underlying class visual features distribution. In this paper, we present a novel approach that leverages text-derived statistics to predict the mean and covariance of the visual feature distribution for each class. This predictive framework enriches the latent space, yielding more robust and generalizable few-shot learning models. We demonstrate the efficacy of incorporating both mean and covariance statistics in improving few-shot classification performance across various datasets. Our method shows that we can use text to predict the mean and covariance of the distribution offering promising improvements in few-shot learning scenarios.

{{</citation>}}


### (9/75) Multi-Class Anomaly Detection based on Regularized Discriminative Coupled hypersphere-based Feature Adaptation (Mehdi Rafiei et al., 2023)

{{<citation>}}

Mehdi Rafiei, Alexandros Iosifidis. (2023)  
**Multi-Class Anomaly Detection based on Regularized Discriminative Coupled hypersphere-based Feature Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.14506v1)  

---


**ABSTRACT**  
In anomaly detection, identification of anomalies across diverse product categories is a complex task. This paper introduces a new model by including class discriminative properties obtained by a modified Regularized Discriminative Variational Auto-Encoder (RD-VAE) in the feature extraction process of Coupled-hypersphere-based Feature Adaptation (CFA). By doing so, the proposed Regularized Discriminative Coupled-hypersphere-based Feature Adaptation (RD-CFA), forms a solution for multi-class anomaly detection. By using the discriminative power of RD-VAE to capture intricate class distributions, combined with CFA's robust anomaly detection capability, the proposed method excels in discerning anomalies across various classes. Extensive evaluations on multi-class anomaly detection and localization using the MVTec AD and BeanTech AD datasets showcase the effectiveness of RD-CFA compared to eight leading contemporary methods.

{{</citation>}}


### (10/75) MRxaI: Black-Box Explainability for Image Classifiers in a Medical Setting (Nathan Blake et al., 2023)

{{<citation>}}

Nathan Blake, Hana Chockler, David A. Kelly, Santiago Calderon Pena, Akchunya Chanchal. (2023)  
**MRxaI: Black-Box Explainability for Image Classifiers in a Medical Setting**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14471v1)  

---


**ABSTRACT**  
Existing tools for explaining the output of image classifiers can be divided into white-box, which rely on access to the model internals, and black-box, agnostic to the model. As the usage of AI in the medical domain grows, so too does the usage of explainability tools. Existing work on medical image explanations focuses on white-box tools, such as gradcam. However, there are clear advantages to switching to a black-box tool, including the ability to use it with any classifier and the wide selection of black-box tools available. On standard images, black-box tools are as precise as white-box. In this paper we compare the performance of several black-box methods against gradcam on a brain cancer MRI dataset. We demonstrate that most black-box tools are not suitable for explaining medical image classifications and present a detailed analysis of the reasons for their shortcomings. We also show that one black-box tool, a causal explainability-based rex, performs as well as \gradcam.

{{</citation>}}


### (11/75) Segment (Almost) Nothing: Prompt-Agnostic Adversarial Attacks on Segmentation Models (Francesco Croce et al., 2023)

{{<citation>}}

Francesco Croce, Matthias Hein. (2023)  
**Segment (Almost) Nothing: Prompt-Agnostic Adversarial Attacks on Segmentation Models**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2311.14450v1)  

---


**ABSTRACT**  
General purpose segmentation models are able to generate (semantic) segmentation masks from a variety of prompts, including visual (points, boxed, etc.) and textual (object names) ones. In particular, input images are pre-processed by an image encoder to obtain embedding vectors which are later used for mask predictions. Existing adversarial attacks target the end-to-end tasks, i.e. aim at altering the segmentation mask predicted for a specific image-prompt pair. However, this requires running an individual attack for each new prompt for the same image. We propose instead to generate prompt-agnostic adversarial attacks by maximizing the $\ell_2$-distance, in the latent space, between the embedding of the original and perturbed images. Since the encoding process only depends on the image, distorted image representations will cause perturbations in the segmentation masks for a variety of prompts. We show that even imperceptible $\ell_\infty$-bounded perturbations of radius $\epsilon=1/255$ are often sufficient to drastically modify the masks predicted with point, box and text prompts by recently proposed foundation models for segmentation. Moreover, we explore the possibility of creating universal, i.e. non image-specific, attacks which can be readily applied to any input without further computational cost.

{{</citation>}}


### (12/75) GCPV: Guided Concept Projection Vectors for the Explainable Inspection of CNN Feature Spaces (Georgii Mikriukov et al., 2023)

{{<citation>}}

Georgii Mikriukov, Gesina Schwalbe, Christian Hellert, Korinna Bade. (2023)  
**GCPV: Guided Concept Projection Vectors for the Explainable Inspection of CNN Feature Spaces**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, GCP  
[Paper Link](http://arxiv.org/abs/2311.14435v1)  

---


**ABSTRACT**  
For debugging and verification of computer vision convolutional deep neural networks (CNNs) human inspection of the learned latent representations is imperative. Therefore, state-of-the-art eXplainable Artificial Intelligence (XAI) methods globally associate given natural language semantic concepts with representing vectors or regions in the CNN latent space supporting manual inspection. Yet, this approach comes with two major disadvantages: They are locally inaccurate when reconstructing a concept label and discard information about the distribution of concept instance representations. The latter, though, is of particular interest for debugging, like finding and understanding outliers, learned notions of sub-concepts, and concept confusion. Furthermore, current single-layer approaches neglect that information about a concept may be spread over the CNN depth. To overcome these shortcomings, we introduce the local-to-global Guided Concept Projection Vectors (GCPV) approach: It (1) generates local concept vectors that each precisely reconstruct a concept segmentation label, and then (2) generalizes these to global concept and even sub-concept vectors by means of hiearchical clustering. Our experiments on object detectors demonstrate improved performance compared to the state-of-the-art, the benefit of multi-layer concept vectors, and robustness against low-quality concept segmentation labels. Finally, we demonstrate that GCPVs can be applied to find root causes for confusion of concepts like bus and truck, and reveal interesting concept-level outliers. Thus, GCPVs pose a promising step towards interpretable model debugging and informed data improvement.

{{</citation>}}


### (13/75) OneFormer3D: One Transformer for Unified Point Cloud Segmentation (Maxim Kolodiazhnyi et al., 2023)

{{<citation>}}

Maxim Kolodiazhnyi, Anna Vorontsova, Anton Konushin, Danila Rukhovich. (2023)  
**OneFormer3D: One Transformer for Unified Point Cloud Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.14405v1)  

---


**ABSTRACT**  
Semantic, instance, and panoptic segmentation of 3D point clouds have been addressed using task-specific models of distinct design. Thereby, the similarity of all segmentation tasks and the implicit relationship between them have not been utilized effectively. This paper presents a unified, simple, and effective model addressing all these tasks jointly. The model, named OneFormer3D, performs instance and semantic segmentation consistently, using a group of learnable kernels, where each kernel is responsible for generating a mask for either an instance or a semantic category. These kernels are trained with a transformer-based decoder with unified instance and semantic queries passed as an input. Such a design enables training a model end-to-end in a single run, so that it achieves top performance on all three segmentation tasks simultaneously. Specifically, our OneFormer3D ranks 1st and sets a new state-of-the-art (+2.1 mAP50) in the ScanNet test leaderboard. We also demonstrate the state-of-the-art results in semantic, instance, and panoptic segmentation of ScanNet (+21 PQ), ScanNet200 (+3.8 mAP50), and S3DIS (+0.8 mIoU) datasets.

{{</citation>}}


### (14/75) A Parameterized Generative Adversarial Network Using Cyclic Projection for Explainable Medical Image Classification (Xiangyu Xiong et al., 2023)

{{<citation>}}

Xiangyu Xiong, Yue Sun, Xiaohong Liu, ChanTong Lam, Tong Tong, Hao Chen, Qinquan Gao, Wei Ke, Tao Tan. (2023)  
**A Parameterized Generative Adversarial Network Using Cyclic Projection for Explainable Medical Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2311.14388v1)  

---


**ABSTRACT**  
Although current data augmentation methods are successful to alleviate the data insufficiency, conventional augmentation are primarily intra-domain while advanced generative adversarial networks (GANs) generate images remaining uncertain, particularly in small-scale datasets. In this paper, we propose a parameterized GAN (ParaGAN) that effectively controls the changes of synthetic samples among domains and highlights the attention regions for downstream classification. Specifically, ParaGAN incorporates projection distance parameters in cyclic projection and projects the source images to the decision boundary to obtain the class-difference maps. Our experiments show that ParaGAN can consistently outperform the existing augmentation methods with explainable classification on two small-scale medical datasets.

{{</citation>}}


### (15/75) Towards Concept-based Interpretability of Skin Lesion Diagnosis using Vision-Language Models (Cristiano Patrício et al., 2023)

{{<citation>}}

Cristiano Patrício, Luís F. Teixeira, João C. Neves. (2023)  
**Towards Concept-based Interpretability of Skin Lesion Diagnosis using Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.14339v1)  

---


**ABSTRACT**  
Concept-based models naturally lend themselves to the development of inherently interpretable skin lesion diagnosis, as medical experts make decisions based on a set of visual patterns of the lesion. Nevertheless, the development of these models depends on the existence of concept-annotated datasets, whose availability is scarce due to the specialized knowledge and expertise required in the annotation process. In this work, we show that vision-language models can be used to alleviate the dependence on a large number of concept-annotated samples. In particular, we propose an embedding learning strategy to adapt CLIP to the downstream task of skin lesion classification using concept-based descriptions as textual embeddings. Our experiments reveal that vision-language models not only attain better accuracy when using concepts as textual embeddings, but also require a smaller number of concept-annotated samples to attain comparable performance to approaches specifically devised for automatic concept generation.

{{</citation>}}


### (16/75) TVT: Training-Free Vision Transformer Search on Tiny Datasets (Zimian Wei et al., 2023)

{{<citation>}}

Zimian Wei, Hengyue Pan, Lujun Li, Peijie Dong, Zhiliang Tian, Xin Niu, Dongsheng Li. (2023)  
**TVT: Training-Free Vision Transformer Search on Tiny Datasets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.14337v1)  

---


**ABSTRACT**  
Training-free Vision Transformer (ViT) architecture search is presented to search for a better ViT with zero-cost proxies. While ViTs achieve significant distillation gains from CNN teacher models on small datasets, the current zero-cost proxies in ViTs do not generalize well to the distillation training paradigm according to our experimental observations. In this paper, for the first time, we investigate how to search in a training-free manner with the help of teacher models and devise an effective Training-free ViT (TVT) search framework. Firstly, we observe that the similarity of attention maps between ViT and ConvNet teachers affects distill accuracy notably. Thus, we present a teacher-aware metric conditioned on the feature attention relations between teacher and student. Additionally, TVT employs the L2-Norm of the student's weights as the student-capability metric to improve ranking consistency. Finally, TVT searches for the best ViT for distilling with ConvNet teachers via our teacher-aware metric and student-capability metric, resulting in impressive gains in efficiency and effectiveness. Extensive experiments on various tiny datasets and search spaces show that our TVT outperforms state-of-the-art training-free search methods. The code will be released.

{{</citation>}}


### (17/75) Maximizing Discrimination Capability of Knowledge Distillation with Energy-based Score (Seonghak Kim et al., 2023)

{{<citation>}}

Seonghak Kim, Gyeongdo Ham, Suin Lee, Donggon Jang, Daeshik Kim. (2023)  
**Maximizing Discrimination Capability of Knowledge Distillation with Energy-based Score**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2311.14334v1)  

---


**ABSTRACT**  
To apply the latest computer vision techniques that require a large computational cost in real industrial applications, knowledge distillation methods (KDs) are essential. Existing logit-based KDs apply the constant temperature scaling to all samples in dataset, limiting the utilization of knowledge inherent in each sample individually. In our approach, we classify the dataset into two categories (i.e., low energy and high energy samples) based on their energy score. Through experiments, we have confirmed that low energy samples exhibit high confidence scores, indicating certain predictions, while high energy samples yield low confidence scores, meaning uncertain predictions. To distill optimal knowledge by adjusting non-target class predictions, we apply a higher temperature to low energy samples to create smoother distributions and a lower temperature to high energy samples to achieve sharper distributions. When compared to previous logit-based and feature-based methods, our energy-based KD (Energy KD) achieves better performance on various datasets. Especially, Energy KD shows significant improvements on CIFAR-100-LT and ImageNet datasets, which contain many challenging samples. Furthermore, we propose high energy-based data augmentation (HE-DA) for further improving the performance. We demonstrate that meaningful performance improvement could be achieved by augmenting only 20-50% of dataset, suggesting that it can be employed on resource-limited devices. To the best of our knowledge, this paper represents the first attempt to make use of energy scores in KD and DA, and we believe it will greatly contribute to future research.

{{</citation>}}


### (18/75) Stable Cluster Discrimination for Deep Clustering (Qi Qian, 2023)

{{<citation>}}

Qi Qian. (2023)  
**Stable Cluster Discrimination for Deep Clustering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.14310v1)  

---


**ABSTRACT**  
Deep clustering can optimize representations of instances (i.e., representation learning) and explore the inherent data distribution (i.e., clustering) simultaneously, which demonstrates a superior performance over conventional clustering methods with given features. However, the coupled objective implies a trivial solution that all instances collapse to the uniform features. To tackle the challenge, a two-stage training strategy is developed for decoupling, where it introduces an additional pre-training stage for representation learning and then fine-tunes the obtained model for clustering. Meanwhile, one-stage methods are developed mainly for representation learning rather than clustering, where various constraints for cluster assignments are designed to avoid collapsing explicitly. Despite the success of these methods, an appropriate learning objective tailored for deep clustering has not been investigated sufficiently. In this work, we first show that the prevalent discrimination task in supervised learning is unstable for one-stage clustering due to the lack of ground-truth labels and positive instances for certain clusters in each mini-batch. To mitigate the issue, a novel stable cluster discrimination (SeCu) task is proposed and a new hardness-aware clustering criterion can be obtained accordingly. Moreover, a global entropy constraint for cluster assignments is studied with efficient optimization. Extensive experiments are conducted on benchmark data sets and ImageNet. SeCu achieves state-of-the-art performance on all of them, which demonstrates the effectiveness of one-stage deep clustering. Code is available at \url{https://github.com/idstcv/SeCu}.

{{</citation>}}


### (19/75) Cosine Similarity Knowledge Distillation for Individual Class Information Transfer (Gyeongdo Ham et al., 2023)

{{<citation>}}

Gyeongdo Ham, Seonghak Kim, Suin Lee, Jae-Hyeok Lee, Daeshik Kim. (2023)  
**Cosine Similarity Knowledge Distillation for Individual Class Information Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.14307v1)  

---


**ABSTRACT**  
Previous logits-based Knowledge Distillation (KD) have utilized predictions about multiple categories within each sample (i.e., class predictions) and have employed Kullback-Leibler (KL) divergence to reduce the discrepancy between the student and teacher predictions. Despite the proliferation of KD techniques, the student model continues to fall short of achieving a similar level as teachers. In response, we introduce a novel and effective KD method capable of achieving results on par with or superior to the teacher models performance. We utilize teacher and student predictions about multiple samples for each category (i.e., batch predictions) and apply cosine similarity, a commonly used technique in Natural Language Processing (NLP) for measuring the resemblance between text embeddings. This metric's inherent scale-invariance property, which relies solely on vector direction and not magnitude, allows the student to dynamically learn from the teacher's knowledge, rather than being bound by a fixed distribution of the teacher's knowledge. Furthermore, we propose a method called cosine similarity weighted temperature (CSWT) to improve the performance. CSWT reduces the temperature scaling in KD when the cosine similarity between the student and teacher models is high, and conversely, it increases the temperature scaling when the cosine similarity is low. This adjustment optimizes the transfer of information from the teacher to the student model. Extensive experimental results show that our proposed method serves as a viable alternative to existing methods. We anticipate that this approach will offer valuable insights for future research on model compression.

{{</citation>}}


### (20/75) GeoViT: A Versatile Vision Transformer Architecture for Geospatial Image Analysis (Madhav Khirwar et al., 2023)

{{<citation>}}

Madhav Khirwar, Ankur Narang. (2023)  
**GeoViT: A Versatile Vision Transformer Architecture for Geospatial Image Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.14301v1)  

---


**ABSTRACT**  
Greenhouse gases are pivotal drivers of climate change, necessitating precise quantification and source identification to foster mitigation strategies. We introduce GeoViT, a compact vision transformer model adept in processing satellite imagery for multimodal segmentation, classification, and regression tasks targeting CO2 and NO2 emissions. Leveraging GeoViT, we attain superior accuracy in estimating power generation rates, fuel type, plume coverage for CO2, and high-resolution NO2 concentration mapping, surpassing previous state-of-the-art models while significantly reducing model size. GeoViT demonstrates the efficacy of vision transformer architectures in harnessing satellite-derived data for enhanced GHG emission insights, proving instrumental in advancing climate change monitoring and emission regulation efforts globally.

{{</citation>}}


### (21/75) Image Super-Resolution with Text Prompt Diffusion (Zheng Chen et al., 2023)

{{<citation>}}

Zheng Chen, Yulun Zhang, Jinjin Gu, Xin Yuan, Linghe Kong, Guihai Chen, Xiaokang Yang. (2023)  
**Image Super-Resolution with Text Prompt Diffusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: T5  
[Paper Link](http://arxiv.org/abs/2311.14282v1)  

---


**ABSTRACT**  
Image super-resolution (SR) methods typically model degradation to improve reconstruction accuracy in complex and unknown degradation scenarios. However, extracting degradation information from low-resolution images is challenging, which limits the model performance. To boost image SR performance, one feasible approach is to introduce additional priors. Inspired by advancements in multi-modal methods and text prompt image processing, we introduce text prompts to image SR to provide degradation priors. Specifically, we first design a text-image generation pipeline to integrate text into SR dataset through the text degradation representation and degradation model. The text representation applies a discretization manner based on the binning method to describe the degradation abstractly. This representation method can also maintain the flexibility of language. Meanwhile, we propose the PromptSR to realize the text prompt SR. The PromptSR employs the diffusion model and the pre-trained language model (e.g., T5 and CLIP). We train the model on the generated text-image dataset. Extensive experiments indicate that introducing text prompts into image SR, yields excellent results on both synthetic and real-world images. Code: https://github.com/zhengchen1999/PromptSR.

{{</citation>}}


### (22/75) Cooperative Dual Attention for Audio-Visual Speech Enhancement with Facial Cues (Feixiang Wang et al., 2023)

{{<citation>}}

Feixiang Wang, Shuang Yang, Shiguang Shan, Xilin Chen. (2023)  
**Cooperative Dual Attention for Audio-Visual Speech Enhancement with Facial Cues**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-SD, cs.CV, eess-AS  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.14275v1)  

---


**ABSTRACT**  
In this work, we focus on leveraging facial cues beyond the lip region for robust Audio-Visual Speech Enhancement (AVSE). The facial region, encompassing the lip region, reflects additional speech-related attributes such as gender, skin color, nationality, etc., which contribute to the effectiveness of AVSE. However, static and dynamic speech-unrelated attributes also exist, causing appearance changes during speech. To address these challenges, we propose a Dual Attention Cooperative Framework, DualAVSE, to ignore speech-unrelated information, capture speech-related information with facial cues, and dynamically integrate it with the audio signal for AVSE. Specifically, we introduce a spatial attention-based visual encoder to capture and enhance visual speech information beyond the lip region, incorporating global facial context and automatically ignoring speech-unrelated information for robust visual feature extraction. Additionally, a dynamic visual feature fusion strategy is introduced by integrating a temporal-dimensional self-attention module, enabling the model to robustly handle facial variations. The acoustic noise in the speaking process is variable, impacting audio quality. Therefore, a dynamic fusion strategy for both audio and visual features is introduced to address this issue. By integrating cooperative dual attention in the visual encoder and audio-visual fusion strategy, our model effectively extracts beneficial speech information from both audio and visual cues for AVSE. Thorough analysis and comparison on different datasets, including normal and challenging cases with unreliable or absent visual information, consistently show our model outperforming existing methods across multiple metrics.

{{</citation>}}


### (23/75) CRISP: Hybrid Structured Sparsity for Class-aware Model Pruning (Shivam Aggarwal et al., 2023)

{{<citation>}}

Shivam Aggarwal, Kuluhan Binici, Tulika Mitra. (2023)  
**CRISP: Hybrid Structured Sparsity for Class-aware Model Pruning**  

---
Primary Category: cs.CV  
Categories: cs-AR, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2311.14272v1)  

---


**ABSTRACT**  
Machine learning pipelines for classification tasks often train a universal model to achieve accuracy across a broad range of classes. However, a typical user encounters only a limited selection of classes regularly. This disparity provides an opportunity to enhance computational efficiency by tailoring models to focus on user-specific classes. Existing works rely on unstructured pruning, which introduces randomly distributed non-zero values in the model, making it unsuitable for hardware acceleration. Alternatively, some approaches employ structured pruning, such as channel pruning, but these tend to provide only minimal compression and may lead to reduced model accuracy. In this work, we propose CRISP, a novel pruning framework leveraging a hybrid structured sparsity pattern that combines both fine-grained N:M structured sparsity and coarse-grained block sparsity. Our pruning strategy is guided by a gradient-based class-aware saliency score, allowing us to retain weights crucial for user-specific classes. CRISP achieves high accuracy with minimal memory consumption for popular models like ResNet-50, VGG-16, and MobileNetV2 on ImageNet and CIFAR-100 datasets. Moreover, CRISP delivers up to 14$\times$ reduction in latency and energy consumption compared to existing pruning methods while maintaining comparable accuracy. Our code is available at https://github.com/shivmgg/CRISP/.

{{</citation>}}


### (24/75) ZeroPS: High-quality Cross-modal Knowledge Transfer for Zero-Shot 3D Part Segmentation (Yuheng Xue et al., 2023)

{{<citation>}}

Yuheng Xue, Nenglun Chen, Jun Liu, Wenyun Sun. (2023)  
**ZeroPS: High-quality Cross-modal Knowledge Transfer for Zero-Shot 3D Part Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.14262v1)  

---


**ABSTRACT**  
Recently, many 2D pretrained foundational models have demonstrated impressive zero-shot prediction capabilities. In this work, we design a novel pipeline for zero-shot 3D part segmentation, called ZeroPS. It high-quality transfers knowledge from 2D pretrained foundational models to 3D point clouds. The main idea of our approach is to explore the natural relationship between multi-view correspondences and the prompt mechanism of foundational models and build bridges on it. Our pipeline consists of two components: 1) a self-extension component that extends 2D groups from a single viewpoint to spatial global-level 3D groups; 2) a multi-modal labeling component that introduces a two-dimensional checking mechanism to vote each 2D predicted bounding box to the best matching 3D part, and a Class Non-highest Vote Penalty function to refine the Vote Matrix. Additionally, a merging algorithm is included to merge part-level 3D groups. Extensive evaluation of three zero-shot segmentation tasks on PartnetE datasets, achieving state-of-the-art results with significant improvements (+19.6%, +5.2% and +4.9%, respectively) over existing methods. Our proposed approach does not need any training, fine-tuning or learnable parameters. It is hardly affected by domain shift. The code will be released.

{{</citation>}}


### (25/75) RSB-Pose: Robust Short-Baseline Binocular 3D Human Pose Estimation with Occlusion Handling (Xiaoyue Wan et al., 2023)

{{<citation>}}

Xiaoyue Wan, Zhuo Chen, Yiming Bao, Xu Zhao. (2023)  
**RSB-Pose: Robust Short-Baseline Binocular 3D Human Pose Estimation with Occlusion Handling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.14242v1)  

---


**ABSTRACT**  
In the domain of 3D Human Pose Estimation, which finds widespread daily applications, the requirement for convenient acquisition equipment continues to grow. To satisfy this demand, we set our sights on a short-baseline binocular setting that offers both portability and a geometric measurement property that radically mitigates depth ambiguity. However, as the binocular baseline shortens, two serious challenges emerge: first, the robustness of 3D reconstruction against 2D errors deteriorates; and second, occlusion reoccurs due to the limited visual differences between two views. To address the first challenge, we propose the Stereo Co-Keypoints Estimation module to improve the view consistency of 2D keypoints and enhance the 3D robustness. In this module, the disparity is utilized to represent the correspondence of binocular 2D points and the Stereo Volume Feature is introduced to contain binocular features across different disparities. Through the regression of SVF, two-view 2D keypoints are simultaneously estimated in a collaborative way which restricts their view consistency. Furthermore, to deal with occlusions, a Pre-trained Pose Transformer module is introduced. Through this module, 3D poses are refined by perceiving pose coherence, a representation of joint correlations. This perception is injected by the Pose Transformer network and learned through a pre-training task that recovers iterative masked joints. Comprehensive experiments carried out on H36M and MHAD datasets, complemented by visualizations, validate the effectiveness of our approach in the short-baseline binocular 3D Human Pose Estimation and occlusion handling.

{{</citation>}}


## cs.LG (15)



### (26/75) One Pass Streaming Algorithm for Super Long Token Attention Approximation in Sublinear Space (Raghav Addanki et al., 2023)

{{<citation>}}

Raghav Addanki, Chenyang Li, Zhao Song, Chiwun Yang. (2023)  
**One Pass Streaming Algorithm for Super Long Token Attention Approximation in Sublinear Space**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: AI, Attention, Language Model  
[Paper Link](http://arxiv.org/abs/2311.14652v1)  

---


**ABSTRACT**  
Deploying Large Language Models (LLMs) in streaming applications that involve long contexts, particularly for extended dialogues and text analysis, is of paramount importance but presents two significant challenges. Firstly, the memory consumption is substantial during the decoding phase due to the caching of Key and Value states (KV) of previous tokens. Secondly, attention computation is time-consuming with a time complexity of $O(n^2)$ for the generation of each token. In recent OpenAI DevDay (Nov 6, 2023), OpenAI released a new model that is able to support a 128K-long document, in our paper, we focus on the memory-efficient issue when context length $n$ is much greater than 128K ($n \gg 2^d$). Considering a single-layer self-attention with Query, Key, and Value matrices $Q, K, V \in \mathbb{R}^{n \times d}$, the polynomial method approximates the attention output $T \in \mathbb{R}^{n \times d}$. It accomplishes this by constructing $U_1, U_2 \in \mathbb{R}^{n \times t}$ to expedite attention ${\sf Attn}(Q, K, V)$ computation within $n^{1+o(1)}$ time executions. Despite this, storing the Key and Value matrices $K, V \in \mathbb{R}^{n \times d}$ still necessitates $O( n d)$ space, leading to significant memory usage. In response to these challenges, we introduce a new algorithm that only reads one pass of the data in streaming fashion. This method employs sublinear space $o(n)$ to store three sketch matrices, alleviating the need for exact $K, V$ storage. Notably, our algorithm exhibits exceptional memory-efficient performance with super-long tokens. As the token length $n$ increases, our error guarantee diminishes while the memory usage remains nearly constant. This unique attribute underscores the potential of our technique in efficiently handling LLMs in streaming applications.

{{</citation>}}


### (27/75) Differentially Private SGD Without Clipping Bias: An Error-Feedback Approach (Xinwei Zhang et al., 2023)

{{<citation>}}

Xinwei Zhang, Zhiqi Bu, Zhiwei Steven Wu, Mingyi Hong. (2023)  
**Differentially Private SGD Without Clipping Bias: An Error-Feedback Approach**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.14632v1)  

---


**ABSTRACT**  
Differentially Private Stochastic Gradient Descent with gradient clipping (DPSGD-GC) is a powerful tool for training deep learning models using sensitive data, providing both a solid theoretical privacy guarantee and high efficiency. However, using DPSGD-GC to ensure Differential Privacy (DP) comes at the cost of model performance degradation due to DP noise injection and gradient clipping. Existing research has extensively analyzed the theoretical convergence of DPSGD-GC, and has shown that it only converges when using large clipping thresholds that are dependent on problem-specific parameters. Unfortunately, these parameters are often unknown in practice, making it hard to choose the optimal clipping threshold. Therefore, in practice, DPSGD-GC suffers from degraded performance due to the {\it constant} bias introduced by the clipping.   In our work, we propose a new error-feedback (EF) DP algorithm as an alternative to DPSGD-GC, which not only offers a diminishing utility bound without inducing a constant clipping bias, but more importantly, it allows for an arbitrary choice of clipping threshold that is independent of the problem. We establish an algorithm-specific DP analysis for our proposed algorithm, providing privacy guarantees based on R{\'e}nyi DP. Additionally, we demonstrate that under mild conditions, our algorithm can achieve nearly the same utility bound as DPSGD without gradient clipping. Our empirical results on Cifar-10/100 and E2E datasets, show that the proposed algorithm achieves higher accuracies than DPSGD while maintaining the same level of DP guarantee.

{{</citation>}}


### (28/75) Finding Foundation Models for Time Series Classification with a PreText Task (Ali Ismail-Fawaz et al., 2023)

{{<citation>}}

Ali Ismail-Fawaz, Maxime Devanne, Stefano Berretti, Jonathan Weber, Germain Forestier. (2023)  
**Finding Foundation Models for Time Series Classification with a PreText Task**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.14534v1)  

---


**ABSTRACT**  
Over the past decade, Time Series Classification (TSC) has gained an increasing attention. While various methods were explored, deep learning - particularly through Convolutional Neural Networks (CNNs)-stands out as an effective approach. However, due to the limited availability of training data, defining a foundation model for TSC that overcomes the overfitting problem is still a challenging task. The UCR archive, encompassing a wide spectrum of datasets ranging from motion recognition to ECG-based heart disease detection, serves as a prime example for exploring this issue in diverse TSC scenarios. In this paper, we address the overfitting challenge by introducing pre-trained domain foundation models. A key aspect of our methodology is a novel pretext task that spans multiple datasets. This task is designed to identify the originating dataset of each time series sample, with the goal of creating flexible convolution filters that can be applied across different datasets. The research process consists of two phases: a pre-training phase where the model acquires general features through the pretext task, and a subsequent fine-tuning phase for specific dataset classifications. Our extensive experiments on the UCR archive demonstrate that this pre-training strategy significantly outperforms the conventional training approach without pre-training. This strategy effectively reduces overfitting in small datasets and provides an efficient route for adapting these models to new datasets, thus advancing the capabilities of deep learning in TSC.

{{</citation>}}


### (29/75) Fault Detection in Telecom Networks using Bi-level Federated Graph Neural Networks (R. Bourgerie et al., 2023)

{{<citation>}}

R. Bourgerie, T. Zanouda. (2023)  
**Fault Detection in Telecom Networks using Bi-level Federated Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NI, cs.LG  
Keywords: Anomaly Detection, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.14469v1)  

---


**ABSTRACT**  
5G and Beyond Networks become increasingly complex and heterogeneous, with diversified and high requirements from a wide variety of emerging applications. The complexity and diversity of Telecom networks place an increasing strain on maintenance and operation efforts. Moreover, the strict security and privacy requirements present a challenge for mobile operators to leverage network data. To detect network faults, and mitigate future failures, prior work focused on leveraging traditional ML/DL methods to locate anomalies in networks. The current approaches, although powerful, do not consider the intertwined nature of embedded and software-intensive Radio Access Network systems. In this paper, we propose a Bi-level Federated Graph Neural Network anomaly detection and diagnosis model that is able to detect anomalies in Telecom networks in a privacy-preserving manner, while minimizing communication costs. Our method revolves around conceptualizing Telecom data as a bi-level temporal Graph Neural Networks. The first graph captures the interactions between different RAN nodes that are exposed to different deployment scenarios in the network, while each individual Radio Access Network node is further elaborated into its software (SW) execution graph. Additionally, we use Federated Learning to address privacy and security limitations. Furthermore, we study the performance of anomaly detection model under three settings: (1) Centralized (2) Federated Learning and (3) Personalized Federated Learning using real-world data from an operational network. Our comprehensive experiments showed that Personalized Federated Temporal Graph Neural Networks method outperforms the most commonly used techniques for Anomaly Detection.

{{</citation>}}


### (30/75) Finite Volume Features, Global Geometry Representations, and Residual Training for Deep Learning-based CFD Simulation (Loh Sher En Jessica et al., 2023)

{{<citation>}}

Loh Sher En Jessica, Naheed Anjum Arafat, Wei Xian Lim, Wai Lee Chan, Adams Wai Kin Kong. (2023)  
**Finite Volume Features, Global Geometry Representations, and Residual Training for Deep Learning-based CFD Simulation**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs.LG, physics-flu-dyn  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.14464v1)  

---


**ABSTRACT**  
Computational fluid dynamics (CFD) simulation is an irreplaceable modelling step in many engineering designs, but it is often computationally expensive. Some graph neural network (GNN)-based CFD methods have been proposed. However, the current methods inherit the weakness of traditional numerical simulators, as well as ignore the cell characteristics in the mesh used in the finite volume method, a common method in practical CFD applications. Specifically, the input nodes in these GNN methods have very limited information about any object immersed in the simulation domain and its surrounding environment. Also, the cell characteristics of the mesh such as cell volume, face surface area, and face centroid are not included in the message-passing operations in the GNN methods. To address these weaknesses, this work proposes two novel geometric representations: Shortest Vector (SV) and Directional Integrated Distance (DID). Extracted from the mesh, the SV and DID provide global geometry perspective to each input node, thus removing the need to collect this information through message-passing. This work also introduces the use of Finite Volume Features (FVF) in the graph convolutions as node and edge attributes, enabling its message-passing operations to adjust to different nodes. Finally, this work is the first to demonstrate how residual training, with the availability of low-resolution data, can be adopted to improve the flow field prediction accuracy. Experimental results on two datasets with five different state-of-the-art GNN methods for CFD indicate that SV, DID, FVF and residual training can effectively reduce the predictive error of current GNN-based methods by as much as 41%.

{{</citation>}}


### (31/75) Unveiling The Factors of Aesthetic Preferences with Explainable AI (Derya Soydaner et al., 2023)

{{<citation>}}

Derya Soydaner, Johan Wagemans. (2023)  
**Unveiling The Factors of Aesthetic Preferences with Explainable AI**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14410v1)  

---


**ABSTRACT**  
The allure of aesthetic appeal in images captivates our senses, yet the underlying intricacies of aesthetic preferences remain elusive. In this study, we pioneer a novel perspective by utilizing machine learning models that focus on aesthetic attributes known to influence preferences. Through a data mining approach, our models process these attributes as inputs to predict the aesthetic scores of images. Moreover, to delve deeper and obtain interpretable explanations regarding the factors driving aesthetic preferences, we utilize the popular Explainable AI (XAI) technique known as SHapley Additive exPlanations (SHAP). Our methodology involves employing various machine learning models, including Random Forest, XGBoost, Support Vector Regression, and Multilayer Perceptron, to compare their performances in accurately predicting aesthetic scores, and consistently observing results in conjunction with SHAP. We conduct experiments on three image aesthetic benchmarks, providing insights into the roles of attributes and their interactions. Ultimately, our study aims to shed light on the complex nature of aesthetic preferences in images through machine learning and provides a deeper understanding of the attributes that influence aesthetic judgements.

{{</citation>}}


### (32/75) LLamol: A Dynamic Multi-Conditional Generative Transformer for De Novo Molecular Design (Niklas Dobberstein et al., 2023)

{{<citation>}}

Niklas Dobberstein, Astrid Maass, Jan Hamaekers. (2023)  
**LLamol: A Dynamic Multi-Conditional Generative Transformer for De Novo Molecular Design**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, physics-chem-ph  
Keywords: GPT, NLP, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2311.14407v1)  

---


**ABSTRACT**  
Generative models have demonstrated substantial promise in Natural Language Processing (NLP) and have found application in designing molecules, as seen in General Pretrained Transformer (GPT) models. In our efforts to develop such a tool for exploring the organic chemical space in search of potentially electro-active compounds, we present "LLamol", a single novel generative transformer model based on the LLama 2 architecture, which was trained on a 13M superset of organic compounds drawn from diverse public sources. To allow for a maximum flexibility in usage and robustness in view of potentially incomplete data, we introduce "Stochastic Context Learning" as a new training procedure. We demonstrate that the resulting model adeptly handles single- and multi-conditional organic molecule generation with up to four conditions, yet more are possible. The model generates valid molecular structures in SMILES notation while flexibly incorporating three numerical and/or one token sequence into the generative process, just as requested. The generated compounds are very satisfactory in all scenarios tested. In detail, we showcase the model's capability to utilize token sequences for conditioning, either individually or in combination with numerical properties, making LLamol a potent tool for de novo molecule design, easily expandable with new properties.

{{</citation>}}


### (33/75) BHGNN-RT: Network embedding for directed heterogeneous graphs (Xiyang Sun et al., 2023)

{{<citation>}}

Xiyang Sun, Fumiyasu Komaki. (2023)  
**BHGNN-RT: Network embedding for directed heterogeneous graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.14404v1)  

---


**ABSTRACT**  
Networks are one of the most valuable data structures for modeling problems in the real world. However, the most recent node embedding strategies have focused on undirected graphs, with limited attention to directed graphs, especially directed heterogeneous graphs. In this study, we first investigated the network properties of directed heterogeneous graphs. Based on network analysis, we proposed an embedding method, a bidirectional heterogeneous graph neural network with random teleport (BHGNN-RT), for directed heterogeneous graphs, that leverages bidirectional message-passing process and network heterogeneity. With the optimization of teleport proportion, BHGNN-RT is beneficial to overcome the over-smoothing problem. Extensive experiments on various datasets were conducted to verify the efficacy and efficiency of BHGNN-RT. Furthermore, we investigated the effects of message components, model layer, and teleport proportion on model performance. The performance comparison with all other baselines illustrates that BHGNN-RT achieves state-of-the-art performance, outperforming the benchmark methods in both node classification and unsupervised clustering tasks.

{{</citation>}}


### (34/75) Directly Attention Loss Adjusted Prioritized Experience Replay (Zhuoying Chen et al., 2023)

{{<citation>}}

Zhuoying Chen, Huiping Li, Zhaoxu Wang. (2023)  
**Directly Attention Loss Adjusted Prioritized Experience Replay**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2311.14390v1)  

---


**ABSTRACT**  
Prioritized Experience Replay (PER) enables the model to learn more about relatively important samples by artificially changing their accessed frequencies. However, this non-uniform sampling method shifts the state-action distribution that is originally used to estimate Q-value functions, which brings about the estimation deviation. In this article, an novel off policy reinforcement learning training framework called Directly Attention Loss Adjusted Prioritized Experience Replay (DALAP) is proposed, which can directly quantify the changed extent of the shifted distribution through Parallel Self-Attention network, so as to accurately compensate the error. In addition, a Priority-Encouragement mechanism is designed simultaneously to optimize the sample screening criterion, and further improve the training efficiency. In order to verify the effectiveness and generality of DALAP, we integrate it with the value-function based, the policy-gradient based and multi-agent reinforcement learning algorithm, respectively. The multiple groups of comparative experiments show that DALAP has the significant advantages of both improving the convergence rate and reducing the training variance.

{{</citation>}}


### (35/75) Deciphering and integrating invariants for neural operator learning with various physical mechanisms (Rui Zhang et al., 2023)

{{<citation>}}

Rui Zhang, Qi Meng, Zhi-Ming Ma. (2023)  
**Deciphering and integrating invariants for neural operator learning with various physical mechanisms**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NA, cs.LG, math-NA, physics-comp-ph  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.14361v1)  

---


**ABSTRACT**  
Neural operators have been explored as surrogate models for simulating physical systems to overcome the limitations of traditional partial differential equation (PDE) solvers. However, most existing operator learning methods assume that the data originate from a single physical mechanism, limiting their applicability and performance in more realistic scenarios. To this end, we propose Physical Invariant Attention Neural Operator (PIANO) to decipher and integrate the physical invariants (PI) for operator learning from the PDE series with various physical mechanisms. PIANO employs self-supervised learning to extract physical knowledge and attention mechanisms to integrate them into dynamic convolutional layers. Compared to existing techniques, PIANO can reduce the relative error by 13.6\%-82.2\% on PDE forecasting tasks across varying coefficients, forces, or boundary conditions. Additionally, varied downstream tasks reveal that the PI embeddings deciphered by PIANO align well with the underlying invariants in the PDE systems, verifying the physical significance of PIANO. The source code will be publicly available at: https://github.com/optray/PIANO.

{{</citation>}}


### (36/75) Comparative Analysis of Transformers for Modeling Tabular Data: A Casestudy using Industry Scale Dataset (Usneek Singh et al., 2023)

{{<citation>}}

Usneek Singh, Piyush Arora, Shamika Ganesan, Mohit Kumar, Siddhant Kulkarni, Salil R. Joshi. (2023)  
**Comparative Analysis of Transformers for Modeling Tabular Data: A Casestudy using Industry Scale Dataset**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.14335v1)  

---


**ABSTRACT**  
We perform a comparative analysis of transformer-based models designed for modeling tabular data, specifically on an industry-scale dataset. While earlier studies demonstrated promising outcomes on smaller public or synthetic datasets, the effectiveness did not extend to larger industry-scale datasets. The challenges identified include handling high-dimensional data, the necessity for efficient pre-processing of categorical and numerical features, and addressing substantial computational requirements.   To overcome the identified challenges, the study conducts an extensive examination of various transformer-based models using both synthetic datasets and the default prediction Kaggle dataset (2022) from American Express. The paper presents crucial insights into optimal data pre-processing, compares pre-training and direct supervised learning methods, discusses strategies for managing categorical and numerical features, and highlights trade-offs between computational resources and performance. Focusing on temporal financial data modeling, the research aims to facilitate the systematic development and deployment of transformer-based models in real-world scenarios, emphasizing scalability.

{{</citation>}}


### (37/75) Cycle Invariant Positional Encoding for Graph Representation Learning (Zuoyu Yan et al., 2023)

{{<citation>}}

Zuoyu Yan, Tengfei Ma, Liangcai Gao, Zhi Tang, Chao Chen, Yusu Wang. (2023)  
**Cycle Invariant Positional Encoding for Graph Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.14333v1)  

---


**ABSTRACT**  
Cycles are fundamental elements in graph-structured data and have demonstrated their effectiveness in enhancing graph learning models. To encode such information into a graph learning framework, prior works often extract a summary quantity, ranging from the number of cycles to the more sophisticated persistence diagram summaries. However, more detailed information, such as which edges are encoded in a cycle, has not yet been used in graph neural networks. In this paper, we make one step towards addressing this gap, and propose a structure encoding module, called CycleNet, that encodes cycle information via edge structure encoding in a permutation invariant manner. To efficiently encode the space of all cycles, we start with a cycle basis (i.e., a minimal set of cycles generating the cycle space) which we compute via the kernel of the 1-dimensional Hodge Laplacian of the input graph. To guarantee the encoding is invariant w.r.t. the choice of cycle basis, we encode the cycle information via the orthogonal projector of the cycle basis, which is inspired by BasisNet proposed by Lim et al. We also develop a more efficient variant which however requires that the input graph has a unique shortest cycle basis. To demonstrate the effectiveness of the proposed module, we provide some theoretical understandings of its expressive power. Moreover, we show via a range of experiments that networks enhanced by our CycleNet module perform better in various benchmarks compared to several existing SOTA models.

{{</citation>}}


### (38/75) GATGPT: A Pre-trained Large Language Model with Graph Attention Network for Spatiotemporal Imputation (Yakun Chen et al., 2023)

{{<citation>}}

Yakun Chen, Xianzhi Wang, Guandong Xu. (2023)  
**GATGPT: A Pre-trained Large Language Model with Graph Attention Network for Spatiotemporal Imputation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Attention, GPT, Graph Attention Network, Language Model  
[Paper Link](http://arxiv.org/abs/2311.14332v1)  

---


**ABSTRACT**  
The analysis of spatiotemporal data is increasingly utilized across diverse domains, including transportation, healthcare, and meteorology. In real-world settings, such data often contain missing elements due to issues like sensor malfunctions and data transmission errors. The objective of spatiotemporal imputation is to estimate these missing values by understanding the inherent spatial and temporal relationships in the observed multivariate time series. Traditionally, spatiotemporal imputation has relied on specific, intricate architectures designed for this purpose, which suffer from limited applicability and high computational complexity. In contrast, our approach integrates pre-trained large language models (LLMs) into spatiotemporal imputation, introducing a groundbreaking framework, GATGPT. This framework merges a graph attention mechanism with LLMs. We maintain most of the LLM parameters unchanged to leverage existing knowledge for learning temporal patterns, while fine-tuning the upper layers tailored to various applications. The graph attention component enhances the LLM's ability to understand spatial relationships. Through tests on three distinct real-world datasets, our innovative approach demonstrates comparable results to established deep learning benchmarks.

{{</citation>}}


### (39/75) AdaMedGraph: Adaboosting Graph Neural Networks for Personalized Medicine (Jie Lian et al., 2023)

{{<citation>}}

Jie Lian, Xufang Luo, Caihua Shan, Dongqi Han, Varut Vardhanabhuti, Dongsheng Li. (2023)  
**AdaMedGraph: Adaboosting Graph Neural Networks for Personalized Medicine**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.14304v1)  

---


**ABSTRACT**  
Precision medicine tailored to individual patients has gained significant attention in recent times. Machine learning techniques are now employed to process personalized data from various sources, including images, genetics, and assessments. These techniques have demonstrated good outcomes in many clinical prediction tasks. Notably, the approach of constructing graphs by linking similar patients and then applying graph neural networks (GNNs) stands out, because related information from analogous patients are aggregated and considered for prediction. However, selecting the appropriate edge feature to define patient similarity and construct the graph is challenging, given that each patient is depicted by high-dimensional features from diverse sources. Previous studies rely on human expertise to select the edge feature, which is neither scalable nor efficient in pinpointing crucial edge features for complex diseases. In this paper, we propose a novel algorithm named \ours, which can automatically select important features to construct multiple patient similarity graphs, and train GNNs based on these graphs as weak learners in adaptive boosting. \ours{} is evaluated on two real-world medical scenarios and shows superiors performance.

{{</citation>}}


### (40/75) Out-of-Distribution Generalized Dynamic Graph Neural Network with Disentangled Intervention and Invariance Promotion (Zeyang Zhang et al., 2023)

{{<citation>}}

Zeyang Zhang, Xin Wang, Ziwei Zhang, Haoyang Li, Wenwu Zhu. (2023)  
**Out-of-Distribution Generalized Dynamic Graph Neural Network with Disentangled Intervention and Invariance Promotion**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2311.14255v1)  

---


**ABSTRACT**  
Dynamic graph neural networks (DyGNNs) have demonstrated powerful predictive abilities by exploiting graph structural and temporal dynamics. However, the existing DyGNNs fail to handle distribution shifts, which naturally exist in dynamic graphs, mainly because the patterns exploited by DyGNNs may be variant with respect to labels under distribution shifts. In this paper, we propose Disentangled Intervention-based Dynamic graph Attention networks with Invariance Promotion (I-DIDA) to handle spatio-temporal distribution shifts in dynamic graphs by discovering and utilizing invariant patterns, i.e., structures and features whose predictive abilities are stable across distribution shifts. Specifically, we first propose a disentangled spatio-temporal attention network to capture the variant and invariant patterns. By utilizing the disentangled patterns, we design a spatio-temporal intervention mechanism to create multiple interventional distributions and an environment inference module to infer the latent spatio-temporal environments, and minimize the variance of predictions among these intervened distributions and environments, so that our model can make predictions based on invariant patterns with stable predictive abilities under distribution shifts. Extensive experiments demonstrate the superiority of our method over state-of-the-art baselines under distribution shifts. Our work is the first study of spatio-temporal distribution shifts in dynamic graphs, to the best of our knowledge.

{{</citation>}}


## cs.GT (1)



### (41/75) History Filtering in Imperfect Information Games: Algorithms and Complexity (Christopher Solinas et al., 2023)

{{<citation>}}

Christopher Solinas, Douglas Rebstock, Nathan R. Sturtevant, Michael Buro. (2023)  
**History Filtering in Imperfect Information Games: Algorithms and Complexity**  

---
Primary Category: cs.GT  
Categories: cs-AI, cs-GT, cs.GT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14651v1)  

---


**ABSTRACT**  
Historically applied exclusively to perfect information games, depth-limited search with value functions has been key to recent advances in AI for imperfect information games. Most prominent approaches with strong theoretical guarantees require subgame decomposition - a process in which a subgame is computed from public information and player beliefs. However, subgame decomposition can itself require non-trivial computations, and its tractability depends on the existence of efficient algorithms for either full enumeration or generation of the histories that form the root of the subgame. Despite this, no formal analysis of the tractability of such computations has been established in prior work, and application domains have often consisted of games, such as poker, for which enumeration is trivial on modern hardware. Applying these ideas to more complex domains requires understanding their cost.   In this work, we introduce and analyze the computational aspects and tractability of filtering histories for subgame decomposition. We show that constructing a single history from the root of the subgame is generally intractable, and then provide a necessary and sufficient condition for efficient enumeration. We also introduce a novel Markov Chain Monte Carlo-based generation algorithm for trick-taking card games - a domain where enumeration is often prohibitively expensive. Our experiments demonstrate its improved scalability in the trick-taking card game Oh Hell. These contributions clarify when and how depth-limited search via subgame decomposition can be an effective tool for sequential decision-making in imperfect information settings.

{{</citation>}}


## cs.PF (1)



### (42/75) GVEL: Fast Graph Loading in Edgelist and Compressed Sparse Row (CSR) formats (Subhajit Sahu, 2023)

{{<citation>}}

Subhajit Sahu. (2023)  
**GVEL: Fast Graph Loading in Edgelist and Compressed Sparse Row (CSR) formats**  

---
Primary Category: cs.PF  
Categories: B-8-2, cs-PF, cs.PF  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14650v1)  

---


**ABSTRACT**  
Efficient IO techniques are crucial in high-performance graph processing frameworks like Gunrock and Hornet, as fast graph loading is essential to minimize processing time and reduce system/cloud usage charges. This research study presents approaches for efficiently reading an Edgelist from a text file and converting it to a Compressed Sparse Row (CSR) representation. On a server with dual 16-core Intel Xeon Gold 6226R processors and MegaRAID SAS-3 storage, our approach, which we term as GVEL, outperforms Hornet, Gunrock, and PIGO by significant margins in CSR reading, exhibiting an average speedup of 78x, 112x, and 1.8x, respectively. For Edgelist reading, GVEL is 2.6x faster than PIGO on average, and achieves a Edgelist read rate of 1.9 billion edges/s. For every doubling of threads, GVEL improves performance at an average rate of 1.9x and 1.7x for reading Edgelist and reading CSR respectively.

{{</citation>}}


## cs.CL (8)



### (43/75) Calibrated Language Models Must Hallucinate (Adam Tauman Kalai et al., 2023)

{{<citation>}}

Adam Tauman Kalai, Santosh S. Vempala. (2023)  
**Calibrated Language Models Must Hallucinate**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.14648v1)  

---


**ABSTRACT**  
Recent language models have a mysterious tendency to generate false but plausible-sounding text. Such "hallucinations" are an obstacle to the usability of language-based AI systems and can harm people who rely upon their outputs. This work shows shows that there is an inherent statistical reason that pretrained language models hallucinate certain types of facts, having nothing to do with the transformer LM architecture or data quality. For "arbitrary" facts whose veracity cannot be determined from the training data, we show that hallucination is necessary for language models that satisfy a statistical calibration condition appropriate for generative language models. Specifically, if the maximum probability of any fact is bounded, we show that the probability of generating a hallucination is close to the fraction of facts that occur exactly once in the training data (a "Good-Turing" estimate), even assuming ideal training data without errors.   One conclusion is that models pretrained to be sufficiently good predictors (i.e., calibrated) may require post-training to mitigate hallucinations on the type of arbitrary facts that tend to appear once in the training set. However, our analysis also suggests that there is no statistical reason that pretraining will lead to hallucination on facts that tend to appear more than once in the training data (like references to publications such as articles and books, whose hallucinations have been particularly notable and problematic) or on systematic facts (like arithmetic calculations). Therefore, different architectures and learning algorithms may mitigate these latter types of hallucinations.

{{</citation>}}


### (44/75) GPT Struct Me: Probing GPT Models on Narrative Entity Extraction (Hugo Sousa et al., 2023)

{{<citation>}}

Hugo Sousa, Nuno Guimarães, Alípio Jorge, Ricardo Campos. (2023)  
**GPT Struct Me: Probing GPT Models on Narrative Entity Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2311.14583v1)  

---


**ABSTRACT**  
The importance of systems that can extract structured information from textual data becomes increasingly pronounced given the ever-increasing volume of text produced on a daily basis. Having a system that can effectively extract such information in an interoperable manner would be an asset for several domains, be it finance, health, or legal. Recent developments in natural language processing led to the production of powerful language models that can, to some degree, mimic human intelligence. Such effectiveness raises a pertinent question: Can these models be leveraged for the extraction of structured information? In this work, we address this question by evaluating the capabilities of two state-of-the-art language models -- GPT-3 and GPT-3.5, commonly known as ChatGPT -- in the extraction of narrative entities, namely events, participants, and temporal expressions. This study is conducted on the Text2Story Lusa dataset, a collection of 119 Portuguese news articles whose annotation framework includes a set of entity structures along with several tags and attribute values. We first select the best prompt template through an ablation study over prompt components that provide varying degrees of information on a subset of documents of the dataset. Subsequently, we use the best templates to evaluate the effectiveness of the models on the remaining documents. The results obtained indicate that GPT models are competitive with out-of-the-box baseline systems, presenting an all-in-one alternative for practitioners with limited resources. By studying the strengths and limitations of these models in the context of information extraction, we offer insights that can guide future improvements and avenues to explore in this field.

{{</citation>}}


### (45/75) Data-Efficient Alignment of Large Language Models with Human Feedback Through Natural Language (Di Jin et al., 2023)

{{<citation>}}

Di Jin, Shikib Mehri, Devamanyu Hazarika, Aishwarya Padmakumar, Sungjin Lee, Yang Liu, Mahdi Namazifar. (2023)  
**Data-Efficient Alignment of Large Language Models with Human Feedback Through Natural Language**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BARD, ChatGPT, Falcon, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.14543v1)  

---


**ABSTRACT**  
Learning from human feedback is a prominent technique to align the output of large language models (LLMs) with human expectations. Reinforcement learning from human feedback (RLHF) leverages human preference signals that are in the form of ranking of response pairs to perform this alignment. However, human preference on LLM outputs can come in much richer forms including natural language, which may provide detailed feedback on strengths and weaknesses of a given response. In this work we investigate data efficiency of modeling human feedback that is in natural language. Specifically, we fine-tune an open-source LLM, e.g., Falcon-40B-Instruct, on a relatively small amount (1000 records or even less) of human feedback in natural language in the form of critiques and revisions of responses. We show that this model is able to improve the quality of responses from even some of the strongest LLMs such as ChatGPT, BARD, and Vicuna, through critique and revision of those responses. For instance, through one iteration of revision of ChatGPT responses, the revised responses have 56.6% win rate over the original ones, and this win rate can be further improved to 65.9% after applying the revision for five iterations.

{{</citation>}}


### (46/75) CMed-GPT: Prompt Tuning for Entity-Aware Chinese Medical Dialogue Generation (Zhijie Qu et al., 2023)

{{<citation>}}

Zhijie Qu, Juan Li, Zerui Ma, Jianqiang Li. (2023)  
**CMed-GPT: Prompt Tuning for Entity-Aware Chinese Medical Dialogue Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Dialog, Dialogue, GPT  
[Paper Link](http://arxiv.org/abs/2311.14539v1)  

---


**ABSTRACT**  
Medical dialogue generation relies on natural language generation techniques to enable online medical consultations. Recently, the widespread adoption of large-scale models in the field of natural language processing has facilitated rapid advancements in this technology. Existing medical dialogue models are mostly based on BERT and pre-trained on English corpora, but there is a lack of high-performing models on the task of Chinese medical dialogue generation. To solve the above problem, this paper proposes CMed-GPT, which is the GPT pre-training language model based on Chinese medical domain text. The model is available in two versions, namely, base and large, with corresponding perplexity values of 8.64 and 8.01. Additionally, we incorporate lexical and entity embeddings into the dialogue text in a uniform manner to meet the requirements of downstream dialogue generation tasks. By applying both fine-tuning and p-tuning to CMed-GPT, we lowered the PPL from 8.44 to 7.35. This study not only confirms the exceptional performance of the CMed-GPT model in generating Chinese biomedical text but also highlights the advantages of p-tuning over traditional fine-tuning with prefix prompts. Furthermore, we validate the significance of incorporating external information in medical dialogue generation, which enhances the quality of dialogue generation.

{{</citation>}}


### (47/75) Machine Translation for Ge'ez Language (Aman Kassahun Wassie, 2023)

{{<citation>}}

Aman Kassahun Wassie. (2023)  
**Machine Translation for Ge'ez Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, GPT, GPT-3.5, Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.14530v1)  

---


**ABSTRACT**  
Machine translation (MT) for low-resource languages such as Ge'ez, an ancient language that is no longer spoken in daily life, faces challenges such as out-of-vocabulary words, domain mismatches, and lack of sufficient labeled training data. In this work, we explore various methods to improve Ge'ez MT, including transfer-learning from related languages, optimizing shared vocabulary and token segmentation approaches, finetuning large pre-trained models, and using large language models (LLMs) for few-shot translation with fuzzy matches. We develop a multilingual neural machine translation (MNMT) model based on languages relatedness, which brings an average performance improvement of about 4 BLEU compared to standard bilingual models. We also attempt to finetune the NLLB-200 model, one of the most advanced translation models available today, but find that it performs poorly with only 4k training samples for Ge'ez. Furthermore, we experiment with using GPT-3.5, a state-of-the-art LLM, for few-shot translation with fuzzy matches, which leverages embedding similarity-based retrieval to find context examples from a parallel corpus. We observe that GPT-3.5 achieves a remarkable BLEU score of 9.2 with no initial knowledge of Ge'ez, but still lower than the MNMT baseline of 15.2. Our work provides insights into the potential and limitations of different approaches for low-resource and ancient language MT.

{{</citation>}}


### (48/75) Controlled Text Generation via Language Model Arithmetic (Jasper Dekoninck et al., 2023)

{{<citation>}}

Jasper Dekoninck, Marc Fischer, Luca Beurer-Kellner, Martin Vechev. (2023)  
**Controlled Text Generation via Language Model Arithmetic**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Text Generation  
[Paper Link](http://arxiv.org/abs/2311.14479v1)  

---


**ABSTRACT**  
As Large Language Models (LLMs) are deployed more widely, customization with respect to vocabulary, style and character becomes more important. In this work we introduce model arithmetic, a novel inference framework for composing and biasing LLMs without the need for model (re)training or highly specific datasets. In addition, the framework allows for more precise control of generated text than direct prompting and prior controlled text generation (CTG) techniques. Using model arithmetic, we can express prior CTG techniques as simple formulas and naturally extend them to new and more effective formulations. Further, we show that speculative sampling, a technique for efficient LLM sampling, extends to our setting. This enables highly efficient text generation with multiple composed models with only marginal overhead over a single model. Our empirical evaluation demonstrates that model arithmetic allows fine-grained control of generated text while outperforming state-of-the-art on the task of toxicity reduction.

{{</citation>}}


### (49/75) DP-NMT: Scalable Differentially-Private Machine Translation (Timour Igamberdiev et al., 2023)

{{<citation>}}

Timour Igamberdiev, Doan Nam Long Vu, Felix Künnecke, Zhuo Yu, Jannik Holmer, Ivan Habernal. (2023)  
**DP-NMT: Scalable Differentially-Private Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.14465v1)  

---


**ABSTRACT**  
Neural machine translation (NMT) is a widely popular text generation task, yet there is a considerable research gap in the development of privacy-preserving NMT models, despite significant data privacy concerns for NMT systems. Differentially private stochastic gradient descent (DP-SGD) is a popular method for training machine learning models with concrete privacy guarantees; however, the implementation specifics of training a model with DP-SGD are not always clarified in existing models, with differing software libraries used and code bases not always being public, leading to reproducibility issues. To tackle this, we introduce DP-NMT, an open-source framework for carrying out research on privacy-preserving NMT with DP-SGD, bringing together numerous models, datasets, and evaluation metrics in one systematic software package. Our goal is to provide a platform for researchers to advance the development of privacy-preserving NMT systems, keeping the specific details of the DP-SGD algorithm transparent and intuitive to implement. We run a set of experiments on datasets from both general and privacy-related domains to demonstrate our framework in use. We make our framework publicly available and welcome feedback from the community.

{{</citation>}}


### (50/75) ÚFAL CorPipe at CRAC 2023: Larger Context Improves Multilingual Coreference Resolution (Milan Straka, 2023)

{{<citation>}}

Milan Straka. (2023)  
**ÚFAL CorPipe at CRAC 2023: Larger Context Improves Multilingual Coreference Resolution**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2311.14391v1)  

---


**ABSTRACT**  
We present CorPipe, the winning entry to the CRAC 2023 Shared Task on Multilingual Coreference Resolution. Our system is an improved version of our earlier multilingual coreference pipeline, and it surpasses other participants by a large margin of 4.5 percent points. CorPipe first performs mention detection, followed by coreference linking via an antecedent-maximization approach on the retrieved spans. Both tasks are trained jointly on all available corpora using a shared pretrained language model. Our main improvements comprise inputs larger than 512 subwords and changing the mention decoding to support ensembling. The source code is available at https://github.com/ufal/crac2023-corpipe.

{{</citation>}}


## cs.CE (1)



### (51/75) Evolution of Neural Architectures for Financial Forecasting: A Note on Data Incompatibility during Crisis Periods (Faizal Hafiz et al., 2023)

{{<citation>}}

Faizal Hafiz, Jan Broekaert, Akshya Swain. (2023)  
**Evolution of Neural Architectures for Financial Forecasting: A Note on Data Incompatibility during Crisis Periods**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs-NE, cs.CE  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2311.14604v1)  

---


**ABSTRACT**  
This note focuses on the optimization of neural architectures for stock index movement forecasting following a major market disruption or crisis. Given that such crises may introduce a shift in market dynamics, this study aims to investigate whether the training data from market dynamics prior to the crisis are compatible with the data during the crisis period. To this end, two distinct learning environments are designed to evaluate and reconcile the effects of possibly different market dynamics. These environments differ principally based on the role assigned to the pre-crisis data. In both environments, a set of non-dominated architectures are identified to satisfy the multi-criteria co-evolution problem, which simultaneously addresses the selection issues related to features and hidden layer topology. To test the hypothesis of pre-crisis data incompatibility, the day-ahead movement prediction of the NASDAQ index is considered during two recent and major market disruptions; the 2008 financial crisis and the COVID-19 pandemic. The results of a detailed comparative evaluation convincingly support the incompatibility hypothesis and highlight the need to select re-training windows carefully.

{{</citation>}}


## cs.AI (8)



### (52/75) RAISE -- Radiology AI Safety, an End-to-end lifecycle approach (M. Jorge Cardoso et al., 2023)

{{<citation>}}

M. Jorge Cardoso, Julia Moosbauer, Tessa S. Cook, B. Selnur Erdal, Brad Genereaux, Vikash Gupta, Bennett A. Landman, Tiarna Lee, Parashkev Nachev, Elanchezhian Somasundaram, Ronald M. Summers, Khaled Younis, Sebastien Ourselin, Franz MJ Pfister. (2023)  
**RAISE -- Radiology AI Safety, an End-to-end lifecycle approach**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, physics-med-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14570v1)  

---


**ABSTRACT**  
The integration of AI into radiology introduces opportunities for improved clinical care provision and efficiency but it demands a meticulous approach to mitigate potential risks as with any other new technology. Beginning with rigorous pre-deployment evaluation and validation, the focus should be on ensuring models meet the highest standards of safety, effectiveness and efficacy for their intended applications. Input and output guardrails implemented during production usage act as an additional layer of protection, identifying and addressing individual failures as they occur. Continuous post-deployment monitoring allows for tracking population-level performance (data drift), fairness, and value delivery over time. Scheduling reviews of post-deployment model performance and educating radiologists about new algorithmic-driven findings is critical for AI to be effective in clinical practice. Recognizing that no single AI solution can provide absolute assurance even when limited to its intended use, the synergistic application of quality assurance at multiple levels - regulatory, clinical, technical, and ethical - is emphasized. Collaborative efforts between stakeholders spanning healthcare systems, industry, academia, and government are imperative to address the multifaceted challenges involved. Trust in AI is an earned privilege, contingent on a broad set of goals, among them transparently demonstrating that the AI adheres to the same rigorous safety, effectiveness and efficacy standards as other established medical technologies. By doing so, developers can instil confidence among providers and patients alike, enabling the responsible scaling of AI and the realization of its potential benefits. The roadmap presented herein aims to expedite the achievement of deployable, reliable, and safe AI in radiology.

{{</citation>}}


### (53/75) Universal Jailbreak Backdoors from Poisoned Human Feedback (Javier Rando et al., 2023)

{{<citation>}}

Javier Rando, Florian Tramèr. (2023)  
**Universal Jailbreak Backdoors from Poisoned Human Feedback**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.14455v1)  

---


**ABSTRACT**  
Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.

{{</citation>}}


### (54/75) Prototype of deployment of Federated Learning with IoT devices (Pablo García Santaclara et al., 2023)

{{<citation>}}

Pablo García Santaclara, Ana Fernández Vilas, Rebeca P. Díaz Redondo. (2023)  
**Prototype of deployment of Federated Learning with IoT devices**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14401v1)  

---


**ABSTRACT**  
In the age of technology, data is an increasingly important resource. This importance is growing in the field of Artificial Intelligence (AI), where sub fields such as Machine Learning (ML) need more and more data to achieve better results. Internet of Things (IoT) is the connection of sensors and smart objects to collect and exchange data, in addition to achieving many other tasks. A huge amount of the resource desired, data, is stored in mobile devices, sensors and other Internet of Things (IoT) devices, but remains there due to data protection restrictions. At the same time these devices do not have enough data or computational capacity to train good models. Moreover, transmitting, storing and processing all this data on a centralised server is problematic. Federated Learning (FL) provides an innovative solution that allows devices to learn in a collaborative way. More importantly, it accomplishes this without violating data protection laws. FL is currently growing, and there are several solutions that implement it. This article presents a prototype of a FL solution where the IoT devices used were raspberry pi boards. The results compare the performance of a solution of this type with those obtained in traditional approaches. In addition, the FL solution performance was tested in a hostile environment. A convolutional neural network (CNN) and a image data set were used. The results show the feasibility and usability of these techniques, although in many cases they do not reach the performance of traditional approaches.

{{</citation>}}


### (55/75) Ethical implications of ChatGPT in higher education: A scoping review (Ming Li et al., 2023)

{{<citation>}}

Ming Li, Ariunaa Enkhtur, Fei Cheng, Beverley Anne Yamamoto. (2023)  
**Ethical implications of ChatGPT in higher education: A scoping review**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.14378v1)  

---


**ABSTRACT**  
This scoping review explores the ethical challenges of using ChatGPT in education, focusing particularly on issues related to higher education. By reviewing recent academic articles written in English, Chinese, and Japanese, we aimed to provide a comprehensive overview of relevant research while identifying gaps for future considerations. Drawing on Arksey and O'Malley's (2005) five-stage scoping review framework, we identified research questions, search terms, and conducted article search from four databases in the target three languages. Each article was reviewed by at least two researchers identifying the main ethical issues of utilizing AI in education, particularly higher education. Our analysis of ethical issues followed the framework developed by DeepMind (Weiginger et al., 2021) to identify six main areas of ethical concern in Language Models. The majority of papers were concerned with misinformation harms (n=25) and/or human-computer interaction related harms (n=24). Given the rapid deployment of Generative Artificial Intelligence (GAI), it is imperative for educators to conduct more empirical studies to develop sound ethical policies for the use of GAI.

{{</citation>}}


### (56/75) Large Language Models as Topological Structure Enhancers for Text-Attributed Graphs (Shengyin Sun et al., 2023)

{{<citation>}}

Shengyin Sun, Yuxiang Ren, Chen Ma, Xuecang Zhang. (2023)  
**Large Language Models as Topological Structure Enhancers for Text-Attributed Graphs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: GNN, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.14324v1)  

---


**ABSTRACT**  
The latest advancements in large language models (LLMs) have revolutionized the field of natural language processing (NLP). Inspired by the success of LLMs in NLP tasks, some recent work has begun investigating the potential of applying LLMs in graph learning tasks. However, most of the existing work focuses on utilizing LLMs as powerful node feature augmenters, leaving employing LLMs to enhance graph topological structures an understudied problem. In this work, we explore how to leverage the information retrieval and text generation capabilities of LLMs to refine/enhance the topological structure of text-attributed graphs (TAGs) under the node classification setting. First, we propose using LLMs to help remove unreliable edges and add reliable ones in the TAG. Specifically, we first let the LLM output the semantic similarity between node attributes through delicate prompt designs, and then perform edge deletion and edge addition based on the similarity. Second, we propose using pseudo-labels generated by the LLM to improve graph topology, that is, we introduce the pseudo-label propagation as a regularization to guide the graph neural network (GNN) in learning proper edge weights. Finally, we incorporate the two aforementioned LLM-based methods for graph topological refinement into the process of GNN training, and perform extensive experiments on four real-world datasets. The experimental results demonstrate the effectiveness of LLM-based graph topology refinement (achieving a 0.15%--2.47% performance gain on public benchmarks).

{{</citation>}}


### (57/75) Robust Domain Misinformation Detection via Multi-modal Feature Alignment (Hui Liu et al., 2023)

{{<citation>}}

Hui Liu, Wenya Wang, Hao Sun, Anderson Rocha, Haoliang Li. (2023)  
**Robust Domain Misinformation Detection via Multi-modal Feature Alignment**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2311.14315v1)  

---


**ABSTRACT**  
Social media misinformation harms individuals and societies and is potentialized by fast-growing multi-modal content (i.e., texts and images), which accounts for higher "credibility" than text-only news pieces. Although existing supervised misinformation detection methods have obtained acceptable performances in key setups, they may require large amounts of labeled data from various events, which can be time-consuming and tedious. In turn, directly training a model by leveraging a publicly available dataset may fail to generalize due to domain shifts between the training data (a.k.a. source domains) and the data from target domains. Most prior work on domain shift focuses on a single modality (e.g., text modality) and ignores the scenario where sufficient unlabeled target domain data may not be readily available in an early stage. The lack of data often happens due to the dynamic propagation trend (i.e., the number of posts related to fake news increases slowly before catching the public attention). We propose a novel robust domain and cross-modal approach (\textbf{RDCM}) for multi-modal misinformation detection. It reduces the domain shift by aligning the joint distribution of textual and visual modalities through an inter-domain alignment module and bridges the semantic gap between both modalities through a cross-modality alignment module. We also propose a framework that simultaneously considers application scenarios of domain generalization (in which the target domain data is unavailable) and domain adaptation (in which unlabeled target domain data is available). Evaluation results on two public multi-modal misinformation detection datasets (Pheme and Twitter Datasets) evince the superiority of the proposed model. The formal implementation of this paper can be found in this link: https://github.com/less-and-less-bugs/RDCM

{{</citation>}}


### (58/75) New Epochs in AI Supervision: Design and Implementation of an Autonomous Radiology AI Monitoring System (Vasantha Kumar Venugopal et al., 2023)

{{<citation>}}

Vasantha Kumar Venugopal, Abhishek Gupta, Rohit Takhar, Vidur Mahajan. (2023)  
**New Epochs in AI Supervision: Design and Implementation of an Autonomous Radiology AI Monitoring System**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14305v1)  

---


**ABSTRACT**  
With the increasingly widespread adoption of AI in healthcare, maintaining the accuracy and reliability of AI models in clinical practice has become crucial. In this context, we introduce novel methods for monitoring the performance of radiology AI classification models in practice, addressing the challenges of obtaining real-time ground truth for performance monitoring. We propose two metrics - predictive divergence and temporal stability - to be used for preemptive alerts of AI performance changes. Predictive divergence, measured using Kullback-Leibler and Jensen-Shannon divergences, evaluates model accuracy by comparing predictions with those of two supplementary models. Temporal stability is assessed through a comparison of current predictions against historical moving averages, identifying potential model decay or data drift. This approach was retrospectively validated using chest X-ray data from a single-center imaging clinic, demonstrating its effectiveness in maintaining AI model reliability. By providing continuous, real-time insights into model performance, our system ensures the safe and effective use of AI in clinical decision-making, paving the way for more robust AI integration in healthcare

{{</citation>}}


### (59/75) Efficient Open-world Reinforcement Learning via Knowledge Distillation and Autonomous Rule Discovery (Ekaterina Nikonova et al., 2023)

{{<citation>}}

Ekaterina Nikonova, Cheng Xue, Jochen Renz. (2023)  
**Efficient Open-world Reinforcement Learning via Knowledge Distillation and Autonomous Rule Discovery**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Knowledge Distillation, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.14270v1)  

---


**ABSTRACT**  
Deep reinforcement learning suffers from catastrophic forgetting and sample inefficiency making it less applicable to the ever-changing real world. However, the ability to use previously learned knowledge is essential for AI agents to quickly adapt to novelties. Often, certain spatial information observed by the agent in the previous interactions can be leveraged to infer task-specific rules. Inferred rules can then help the agent to avoid potentially dangerous situations in the previously unseen states and guide the learning process increasing agent's novelty adaptation speed. In this work, we propose a general framework that is applicable to deep reinforcement learning agents. Our framework provides the agent with an autonomous way to discover the task-specific rules in the novel environments and self-supervise it's learning. We provide a rule-driven deep Q-learning agent (RDQ) as one possible implementation of that framework. We show that RDQ successfully extracts task-specific rules as it interacts with the world and uses them to drastically increase its learning efficiency. In our experiments, we show that the RDQ agent is significantly more resilient to the novelties than the baseline agents, and is able to detect and adapt to novel situations faster.

{{</citation>}}


## stat.ML (1)



### (60/75) FRUITS: Feature Extraction Using Iterated Sums for Time Series Classification (Joscha Diehl et al., 2023)

{{<citation>}}

Joscha Diehl, Richard Krieg. (2023)  
**FRUITS: Feature Extraction Using Iterated Sums for Time Series Classification**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.14549v1)  

---


**ABSTRACT**  
We introduce a pipeline for time series classification that extracts features based on the iterated-sums signature (ISS) and then applies a linear classifier. These features are intrinsically nonlinear, capture chronological information, and, under certain settings, are invariant to time-warping. We are competitive with state-of-the-art methods on the UCR archive, both in terms of accuracy and speed. We make our code available at \url{https://github.com/irkri/fruits}.

{{</citation>}}


## cs.DB (1)



### (61/75) RDF Stream Taxonomy: Systematizing RDF Stream Types in Research and Practice (Piotr Sowinski et al., 2023)

{{<citation>}}

Piotr Sowinski, Pawel Szmeja, Maria Ganzha, Marcin Paprzycki. (2023)  
**RDF Stream Taxonomy: Systematizing RDF Stream Types in Research and Practice**  

---
Primary Category: cs.DB  
Categories: cs-AI, cs-DB, cs.DB  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14540v1)  

---


**ABSTRACT**  
Over the years, RDF streaming was explored in research and practice from many angles, resulting in a wide range of RDF stream definitions. This variety presents a major challenge in discussing and integrating streaming solutions, due to the lack of a common language. This work attempts to address this critical research gap, by systematizing RDF stream types present in the literature in a novel taxonomy. The proposed RDF Stream Taxonomy (RDF-STaX) is embodied in an OWL 2 DL ontology that follows the FAIR principles, making it readily applicable in practice. Extensive documentation and additional resources are provided, to foster the adoption of the ontology. Two realized use cases are presented, demonstrating the usefulness of the resource in discussing research works and annotating streaming datasets. Another result of this contribution is the novel nanopublications dataset, which serves as a collaborative, living state-of-the-art review of RDF streaming. The aim of RDF-STaX is to address a real need of the community for a better way to systematize and describe RDF streams. The resource is designed to help drive innovation in RDF streaming, by fostering scientific discussion, cooperation, and tool interoperability.

{{</citation>}}


## cs.NI (3)



### (62/75) Digital Twin-Native AI-Driven Service Architecture for Industrial Networks (Kubra Duran et al., 2023)

{{<citation>}}

Kubra Duran, Matthew Broadbent, Gokhan Yurdakul, Berk Canberk. (2023)  
**Digital Twin-Native AI-Driven Service Architecture for Industrial Networks**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-NI, cs.NI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.14532v1)  

---


**ABSTRACT**  
The dramatic increase in the connectivity demand results in an excessive amount of Internet of Things (IoT) sensors. To meet the management needs of these large-scale networks, such as accurate monitoring and learning capabilities, Digital Twin (DT) is the key enabler. However, current attempts regarding DT implementations remain insufficient due to the perpetual connectivity requirements of IoT networks. Furthermore, the sensor data streaming in IoT networks cause higher processing time than traditional methods. In addition to these, the current intelligent mechanisms cannot perform well due to the spatiotemporal changes in the implemented IoT network scenario. To handle these challenges, we propose a DT-native AI-driven service architecture in support of the concept of IoT networks. Within the proposed DT-native architecture, we implement a TCP-based data flow pipeline and a Reinforcement Learning (RL)-based learner model. We apply the proposed architecture to one of the broad concepts of IoT networks, the Internet of Vehicles (IoV). We measure the efficiency of our proposed architecture and note ~30% processing time-saving thanks to the TCP-based data flow pipeline. Moreover, we test the performance of the learner model by applying several learning rate combinations for actor and critic networks and highlight the most successive model.

{{</citation>}}


### (63/75) Benchmarking Large Language Models for Log Analysis, Security, and Interpretation (Egil Karlsen et al., 2023)

{{<citation>}}

Egil Karlsen, Xiao Luo, Nur Zincir-Heywood, Malcolm Heywood. (2023)  
**Benchmarking Large Language Models for Log Analysis, Security, and Interpretation**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: BERT, GPT, Language Model, Security  
[Paper Link](http://arxiv.org/abs/2311.14519v1)  

---


**ABSTRACT**  
Large Language Models (LLM) continue to demonstrate their utility in a variety of emergent capabilities in different fields. An area that could benefit from effective language understanding in cybersecurity is the analysis of log files. This work explores LLMs with different architectures (BERT, RoBERTa, DistilRoBERTa, GPT-2, and GPT-Neo) that are benchmarked for their capacity to better analyze application and system log files for security. Specifically, 60 fine-tuned language models for log analysis are deployed and benchmarked. The resulting models demonstrate that they can be used to perform log analysis effectively with fine-tuning being particularly important for appropriate domain adaptation to specific log types. The best-performing fine-tuned sequence classification model (DistilRoBERTa) outperforms the current state-of-the-art; with an average F1-Score of 0.998 across six datasets from both web application and system log sources. To achieve this, we propose and implement a new experimentation pipeline (LLM4Sec) which leverages LLMs for log analysis experimentation, evaluation, and analysis.

{{</citation>}}


### (64/75) Federated Transformed Learning for a Circular, Secure, and Tiny AI (Weisi Guo et al., 2023)

{{<citation>}}

Weisi Guo, Schyler Sun, Bin Li, Sam Blakeman. (2023)  
**Federated Transformed Learning for a Circular, Secure, and Tiny AI**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-LG, cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14371v1)  

---


**ABSTRACT**  
Deep Learning (DL) is penetrating into a diverse range of mass mobility, smart living, and industrial applications, rapidly transforming the way we live and work. DL is at the heart of many AI implementations. A key set of challenges is to produce AI modules that are: (1) "circular" - can solve new tasks without forgetting how to solve previous ones, (2) "secure" - have immunity to adversarial data attacks, and (3) "tiny" - implementable in low power low cost embedded hardware. Clearly it is difficult to achieve all three aspects on a single horizontal layer of platforms, as the techniques require transformed deep representations that incur different computation and communication requirements. Here we set out the vision to achieve transformed DL representations across a 5G and Beyond networked architecture. We first detail the cross-sectoral motivations for each challenge area, before demonstrating recent advances in DL research that can achieve circular, secure, and tiny AI (CST-AI). Recognising the conflicting demand of each transformed deep representation, we federate their deep learning transformations and functionalities across the network to achieve connected run-time capabilities.

{{</citation>}}


## cs.SD (1)



### (65/75) tinyCLAP: Distilling Constrastive Language-Audio Pretrained Models (Francesco Paissan et al., 2023)

{{<citation>}}

Francesco Paissan, Elisabetta Farella. (2023)  
**tinyCLAP: Distilling Constrastive Language-Audio Pretrained Models**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2311.14517v1)  

---


**ABSTRACT**  
Contrastive Language-Audio Pretraining (CLAP) became of crucial importance in the field of audio and speech processing. Its employment ranges from sound event detection to text-to-audio generation. However, one of the main limitations is the considerable amount of data required in the training process and the overall computational complexity during inference. This paper investigates how we can reduce the complexity of contrastive language-audio pre-trained models, yielding an efficient model that we call tinyCLAP. We derive an unimodal distillation loss from first principles and explore how the dimensionality of the shared, multimodal latent space can be reduced via pruning. TinyCLAP uses only 6% of the original Microsoft CLAP parameters with a minimal reduction (less than 5%) in zero-shot classification performance across the three sound event detection datasets on which it was tested

{{</citation>}}


## cs.CR (2)



### (66/75) Malware Analysis on AI Technique (Amjani Gupta et al., 2023)

{{<citation>}}

Amjani Gupta, Dr. Karan Singh. (2023)  
**Malware Analysis on AI Technique**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14501v1)  

---


**ABSTRACT**  
In today's world, we are performing our maximum work through the Internet, i.e., online payment, data transfer, etc., per day. More than thousands of users are connecting. So, it's essential to provide security to the user. It is necessary to detect and prevent malicious object from gaining persistence and causing destruction within the organization. Therefore, Malware analysis is needed in order to secure the system. This necessitates the use of effective and efficient approaches for detecting OS malware. Due to the cheap cost of technology, artificial intelligence has also become less difficult to implement in projects to analyse malware. The categorization and analysis of malware on OS using various AI-based analysis techniques are covered in detail in this paper.

{{</citation>}}


### (67/75) AI-based Attack Graph Generation (Sangbeom Park et al., 2023)

{{<citation>}}

Sangbeom Park, Jaesung Lee, Jeongdo Yoo, Min Geun Song, Hyosun Lee, Jaewoong Choi, Chaeyeon Sagong, Huy Kang Kim. (2023)  
**AI-based Attack Graph Generation**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14342v1)  

---


**ABSTRACT**  
With the advancement of IoT technology, many electronic devices are interconnected through networks, communicating with each other and performing specific roles. However, as numerous devices join networks, the threat of cyberattacks also escalates. Preventing and detecting cyber threats are crucial, and one method of preventing such threats involves using attack graphs. Attack graphs are widely used to assess security threats within networks. However, a drawback emerges as the network scales, as generating attack graphs becomes time-consuming. To overcome this limitation, artificial intelligence models can be employed. By utilizing AI models, attack graphs can be created within a short period, approximating optimal outcomes. AI models designed for attack graph generation consist of encoders and decoders, trained using reinforcement learning algorithms. After training the AI models, we confirmed the model's learning effectiveness by observing changes in loss and reward values. Additionally, we compared attack graphs generated by the AI model with those created through conventional methods.

{{</citation>}}


## eess.IV (2)



### (68/75) Sliding Window FastEdit: A Framework for Lesion Annotation in Whole-body PET Images (Matthias Hadlich et al., 2023)

{{<citation>}}

Matthias Hadlich, Zdravko Marinov, Moon Kim, Enrico Nasca, Jens Kleesiek, Rainer Stiefelhagen. (2023)  
**Sliding Window FastEdit: A Framework for Lesion Annotation in Whole-body PET Images**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, cs-HC, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14482v1)  

---


**ABSTRACT**  
Deep learning has revolutionized the accurate segmentation of diseases in medical imaging. However, achieving such results requires training with numerous manual voxel annotations. This requirement presents a challenge for whole-body Positron Emission Tomography (PET) imaging, where lesions are scattered throughout the body. To tackle this problem, we introduce SW-FastEdit - an interactive segmentation framework that accelerates the labeling by utilizing only a few user clicks instead of voxelwise annotations. While prior interactive models crop or resize PET volumes due to memory constraints, we use the complete volume with our sliding window-based interactive scheme. Our model outperforms existing non-sliding window interactive models on the AutoPET dataset and generalizes to the previously unseen HECKTOR dataset. A user study revealed that annotators achieve high-quality predictions with only 10 click iterations and a low perceived NASA-TLX workload. Our framework is implemented using MONAI Label and is available: https://github.com/matt3o/AutoPET2-Submission/

{{</citation>}}


### (69/75) CT-xCOV: a CT-scan based Explainable Framework for COVid-19 diagnosis (Ismail Elbouknify et al., 2023)

{{<citation>}}

Ismail Elbouknify, Afaf Bouhoute, Khalid Fardousse, Ismail Berrada, Abdelmajid Badri. (2023)  
**CT-xCOV: a CT-scan based Explainable Framework for COVid-19 diagnosis**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14462v1)  

---


**ABSTRACT**  
In this work, CT-xCOV, an explainable framework for COVID-19 diagnosis using Deep Learning (DL) on CT-scans is developed. CT-xCOV adopts an end-to-end approach from lung segmentation to COVID-19 detection and explanations of the detection model's prediction. For lung segmentation, we used the well-known U-Net model. For COVID-19 detection, we compared three different CNN architectures: a standard CNN, ResNet50, and DenseNet121. After the detection, visual and textual explanations are provided. For visual explanations, we applied three different XAI techniques, namely, Grad-Cam, Integrated Gradient (IG), and LIME. Textual explanations are added by computing the percentage of infection by lungs. To assess the performance of the used XAI techniques, we propose a ground-truth-based evaluation method, measuring the similarity between the visualization outputs and the ground-truth infections. The performed experiments show that the applied DL models achieved good results. The U-Net segmentation model achieved a high Dice coefficient (98%). The performance of our proposed classification model (standard CNN) was validated using 5-fold cross-validation (acc of 98.40% and f1-score 98.23%). Lastly, the results of the comparison of XAI techniques show that Grad-Cam gives the best explanations compared to LIME and IG, by achieving a Dice coefficient of 55%, on COVID-19 positive scans, compared to 29% and 24% obtained by IG and LIME respectively. The code and the dataset used in this paper are available in the GitHub repository [1].

{{</citation>}}


## cs.RO (2)



### (70/75) What you need to know about a learning robot: Identifying the enabling architecture of complex systems (Helen Beierling et al., 2023)

{{<citation>}}

Helen Beierling, Phillip Richter, Mara Brandt, Lutz Terfloth, Carsten Schulte, Heiko Wersing, Anna-Lisa Vollmer. (2023)  
**What you need to know about a learning robot: Identifying the enabling architecture of complex systems**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.14431v1)  

---


**ABSTRACT**  
Nowadays, we are dealing more and more with robots and AI in everyday life. However, their behavior is not always apparent to most lay users, especially in error situations. As a result, there can be misconceptions about the behavior of the technologies in use. This, in turn, can lead to misuse and rejection by users. Explanation, for example, through transparency, can address these misconceptions. However, it would be confusing and overwhelming for users if the entire software or hardware was explained. Therefore, this paper looks at the 'enabling' architecture. It describes those aspects of a robotic system that might need to be explained to enable someone to use the technology effectively. Furthermore, this paper is concerned with the 'explanandum', which is the corresponding misunderstanding or missing concepts of the enabling architecture that needs to be clarified. We have thus developed and present an approach for determining this 'enabling' architecture and the resulting 'explanandum' of complex technologies.

{{</citation>}}


### (71/75) Robot Learning in the Era of Foundation Models: A Survey (Xuan Xiao et al., 2023)

{{<citation>}}

Xuan Xiao, Jiahang Liu, Zhipeng Wang, Yanmin Zhou, Yong Qi, Qian Cheng, Bin He, Shuo Jiang. (2023)  
**Robot Learning in the Era of Foundation Models: A Survey**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.14379v1)  

---


**ABSTRACT**  
The proliferation of Large Language Models (LLMs) has s fueled a shift in robot learning from automation towards general embodied Artificial Intelligence (AI). Adopting foundation models together with traditional learning methods to robot learning has increasingly gained recent interest research community and showed potential for real-life application. However, there are few literatures comprehensively reviewing the relatively new technologies combined with robotics. The purpose of this review is to systematically assess the state-of-the-art foundation model techniques in the robot learning and to identify future potential areas. Specifically, we first summarized the technical evolution of robot learning and identified the necessary preliminary preparations for foundation models including the simulators, datasets, foundation model framework. In addition, we focused on the following four mainstream areas of robot learning including manipulation, navigation, planning, and reasoning and demonstrated how the foundation model techniques can be adopted in the above scenarios. Furthermore, critical issues which are neglected in the current literatures including robot hardware and software decoupling, dynamic data, generalization performance with the presence of human, etc. were discussed. This review highlights the state-of-the-art progress of foundation models in robot learning and future research should focus on multimodal interaction especially dynamics data, exclusive foundation models for robots, and AI alignment, etc.

{{</citation>}}


## eess.SY (1)



### (72/75) Approximation of Convex Envelope Using Reinforcement Learning (Vivek S. Borkar et al., 2023)

{{<citation>}}

Vivek S. Borkar, Adit Akarsh. (2023)  
**Approximation of Convex Envelope Using Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.14421v1)  

---


**ABSTRACT**  
Oberman gave a stochastic control formulation of the problem of estimating the convex envelope of a non-convex function. Based on this, we develop a reinforcement learning scheme to approximate the convex envelope, using a variant of Q-learning for controlled optimal stopping. It shows very promising results on a standard library of test problems.

{{</citation>}}


## cs.CY (1)



### (73/75) Potential Societal Biases of ChatGPT in Higher Education: A Scoping Review (Ming Li et al., 2023)

{{<citation>}}

Ming Li, Ariunaa Enkhtur, Beverley Anne Yamamoto, Fei Cheng. (2023)  
**Potential Societal Biases of ChatGPT in Higher Education: A Scoping Review**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI, Bias, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.14381v1)  

---


**ABSTRACT**  
ChatGPT and other Generative Artificial Intelligence (GAI) models tend to inherit and even amplify prevailing societal biases as they are trained on large amounts of existing data. Given the increasing usage of ChatGPT and other GAI by students, faculty members, and staff in higher education institutions (HEIs), there is an urgent need to examine the ethical issues involved such as its potential biases. In this scoping review, we clarify the ways in which biases related to GAI in higher education settings have been discussed in recent academic publications and identify what type of potential biases are commonly reported in this body of literature. We searched for academic articles written in English, Chinese, and Japanese across four main databases concerned with GAI usage in higher education and bias. Our findings show that while there is an awareness of potential biases around large language models (LLMs) and GAI, the majority of articles touch on ``bias'' at a relatively superficial level. Few identify what types of bias may occur under what circumstances. Neither do they discuss the possible implications for the higher education, staff, faculty members, or students. There is a notable lack of empirical work at this point, and we call for higher education researchers and AI experts to conduct more research in this area.

{{</citation>}}


## eess.SP (1)



### (74/75) Windformer:Bi-Directional Long-Distance Spatio-Temporal Network For Wind Speed Prediction (Xuewei Li et al., 2023)

{{<citation>}}

Xuewei Li, Zewen Shang, Zhiqiang Liu, Jian Yu, Wei Xiong, Mei Yu. (2023)  
**Windformer:Bi-Directional Long-Distance Spatio-Temporal Network For Wind Speed Prediction**  

---
Primary Category: eess.SP  
Categories: cs-AI, eess-SP, eess.SP  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2311.14316v1)  

---


**ABSTRACT**  
Wind speed prediction is critical to the management of wind power generation. Due to the large range of wind speed fluctuations and wake effect, there may also be strong correlations between long-distance wind turbines. This difficult-to-extract feature has become a bottleneck for improving accuracy. History and future time information includes the trend of airflow changes, whether this dynamic information can be utilized will also affect the prediction effect. In response to the above problems, this paper proposes Windformer. First, Windformer divides the wind turbine cluster into multiple non-overlapping windows and calculates correlations inside the windows, then shifts the windows partially to provide connectivity between windows, and finally fuses multi-channel features based on detailed and global information. To dynamically model the change process of wind speed, this paper extracts time series in both history and future directions simultaneously. Compared with other current-advanced methods, the Mean Square Error (MSE) of Windformer is reduced by 0.5\% to 15\% on two datasets from NERL.

{{</citation>}}


## cs.SI (1)



### (75/75) Fair Influence Maximization in Social Networks: A Community-Based Evolutionary Algorithm (Kaicong Ma et al., 2023)

{{<citation>}}

Kaicong Ma, Xinxiang Xu, Haipeng Yang, Renzhi Cao, Lei Zhang. (2023)  
**Fair Influence Maximization in Social Networks: A Community-Based Evolutionary Algorithm**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2311.14288v1)  

---


**ABSTRACT**  
Influence Maximization (IM) has been extensively studied in network science, which attempts to find a subset of users to maximize the influence spread. A new variant of IM, Fair Influence Maximization (FIM), which primarily enhances the fair propagation of information, attracts increasing attention in academic. However, existing algorithms for FIM suffer from a trade-off between fairness and running time. Since it is a tough task to ensure that users are fairly influenced in terms of sensitive attributes, such as race or gender, while maintaining a high influence spread. To tackle this problem, in this paper, we propose an effective and efficient Community-based Evolutionary Algorithm for FIM (named CEA-FIM). In CEA-FIM, a community-based node selection strategy is proposed to identify potential nodes, which not only considers the size of the community but also the attributes of the nodes in the community. Subsequently, we design an evolutionary algorithm based on the proposed node selection strategy to hasten the search for the optimal solution, including the novel initialization, crossover and mutation strategies. We validate the proposed algorithm CEA-FIM by performing experiments on real-world and synthetic networks. The experimental results show that the proposed CEA-FIM achieves a better balance between effectiveness and efficiency, compared to the state-of-the-art baseline algorithms.

{{</citation>}}
