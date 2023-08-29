---
draft: false
title: "arXiv @ 2023.08.29"
date: 2023-08-29
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.29"
    identifier: arxiv_20230829
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (7)](#cslg-7)
- [cs.DB (1)](#csdb-1)
- [cs.CV (26)](#cscv-26)
- [cs.AI (3)](#csai-3)
- [cs.CL (6)](#cscl-6)
- [cs.IR (3)](#csir-3)
- [eess.SY (2)](#eesssy-2)
- [eess.IV (2)](#eessiv-2)
- [cs.SD (2)](#cssd-2)

## cs.LG (7)



### (1/52) On Active Learning for Gaussian Process-based Global Sensitivity Analysis (Mohit Chauhan et al., 2023)

{{<citation>}}

Mohit Chauhan, Mariel Ojeda-Tuz, Ryan Catarelli, Kurtis Gurley, Dimitrios Tsapetis, Michael D. Shields. (2023)  
**On Active Learning for Gaussian Process-based Global Sensitivity Analysis**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2308.14220v1)  

---


**ABSTRACT**  
This paper explores the application of active learning strategies to adaptively learn Sobol indices for global sensitivity analysis. We demonstrate that active learning for Sobol indices poses unique challenges due to the definition of the Sobol index as a ratio of variances estimated from Gaussian process surrogates. Consequently, learning strategies must either focus on convergence in the numerator or the denominator of this ratio. However, rapid convergence in either one does not guarantee convergence in the Sobol index. We propose a novel strategy for active learning that focuses on resolving the main effects of the Gaussian process (associated with the numerator of the Sobol index) and compare this with existing strategies based on convergence in the total variance (the denominator of the Sobol index). The new strategy, implemented through a new learning function termed the MUSIC (minimize uncertainty in Sobol index convergence), generally converges in Sobol index error more rapidly than the existing strategies based on the Expected Improvement for Global Fit (EIGF) and the Variance Improvement for Global Fit (VIGF). Both strategies are compared with simple sequential random sampling and the MUSIC learning function generally converges most rapidly for low-dimensional problems. However, for high-dimensional problems, the performance is comparable to random sampling. The new learning strategy is demonstrated for a practical case of adaptive experimental design for large-scale Boundary Layer Wind Tunnel experiments.

{{</citation>}}


### (2/52) TimeTrail: Unveiling Financial Fraud Patterns through Temporal Correlation Analysis (Sushrut Ghimire, 2023)

{{<citation>}}

Sushrut Ghimire. (2023)  
**TimeTrail: Unveiling Financial Fraud Patterns through Temporal Correlation Analysis**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-fin-ST  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2308.14215v1)  

---


**ABSTRACT**  
In the field of financial fraud detection, understanding the underlying patterns and dynamics is important to ensure effective and reliable systems. This research introduces a new technique, "TimeTrail," which employs advanced temporal correlation analysis to explain complex financial fraud patterns. The technique leverages time-related insights to provide transparent and interpretable explanations for fraud detection decisions, enhancing accountability and trust.   The "TimeTrail" methodology consists of three key phases: temporal data enrichment, dynamic correlation analysis, and interpretable pattern visualization. Initially, raw financial transaction data is enriched with temporal attributes. Dynamic correlations between these attributes are then quantified using innovative statistical measures. Finally, a unified visualization framework presents these correlations in an interpretable manner. To validate the effectiveness of "TimeTrail," a study is conducted on a diverse financial dataset, surrounding various fraud scenarios. Results demonstrate the technique's capability to uncover hidden temporal correlations and patterns, performing better than conventional methods in both accuracy and interpretability. Moreover, a case study showcasing the application of "TimeTrail" in real-world scenarios highlights its utility for fraud detection.

{{</citation>}}


### (3/52) Topological Augmentation for Class-Imbalanced Node Classification (Zhining Liu et al., 2023)

{{<citation>}}

Zhining Liu, Zhichen Zeng, Ruizhong Qiu, Hyunsik Yoo, David Zhou, Zhe Xu, Yada Zhu, Kommy Weldemariam, Jingrui He, Hanghang Tong. (2023)  
**Topological Augmentation for Class-Imbalanced Node Classification**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.14181v1)  

---


**ABSTRACT**  
Class imbalance is prevalent in real-world node classification tasks and often biases graph learning models toward majority classes. Most existing studies root from a node-centric perspective and aim to address the class imbalance in training data by node/class-wise reweighting or resampling. In this paper, we approach the source of the class-imbalance bias from an under-explored topology-centric perspective. Our investigation reveals that beyond the inherently skewed training class distribution, the graph topology also plays an important role in the formation of predictive bias: we identify two fundamental challenges, namely ambivalent and distant message-passing, that can exacerbate the bias by aggravating majority-class over-generalization and minority-class misclassification. In light of these findings, we devise a lightweight topological augmentation method ToBA to dynamically rectify the nodes influenced by ambivalent/distant message-passing during graph learning, so as to mitigate the class-imbalance bias. We highlight that ToBA is a model-agnostic, efficient, and versatile solution that can be seamlessly combined with and further boost other imbalance-handling techniques. Systematic experiments validate the superior performance of ToBA in both promoting imbalanced node classification and mitigating the prediction bias between different classes.

{{</citation>}}


### (4/52) SPEED: Streaming Partition and Parallel Acceleration for Temporal Interaction Graph Embedding (Xi Chen et al., 2023)

{{<citation>}}

Xi Chen, Yongxiang Liao, Yun Xiong, Yao Zhang, Siwei Zhang, Jiawei Zhang, Yiheng Sun. (2023)  
**SPEED: Streaming Partition and Parallel Acceleration for Temporal Interaction Graph Embedding**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs-SI, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.14129v1)  

---


**ABSTRACT**  
Temporal Interaction Graphs (TIGs) are widely employed to model intricate real-world systems such as financial systems and social networks. To capture the dynamism and interdependencies of nodes, existing TIG embedding models need to process edges sequentially and chronologically. However, this requirement prevents it from being processed in parallel and struggle to accommodate burgeoning data volumes to GPU. Consequently, many large-scale temporal interaction graphs are confined to CPU processing. Furthermore, a generalized GPU scaling and acceleration approach remains unavailable. To facilitate large-scale TIGs' implementation on GPUs for acceleration, we introduce a novel training approach namely Streaming Edge Partitioning and Parallel Acceleration for Temporal Interaction Graph Embedding (SPEED). The SPEED is comprised of a Streaming Edge Partitioning Component (SEP) which addresses space overhead issue by assigning fewer nodes to each GPU, and a Parallel Acceleration Component (PAC) which enables simultaneous training of different sub-graphs, addressing time overhead issue. Our method can achieve a good balance in computing resources, computing time, and downstream task performance. Empirical validation across 7 real-world datasets demonstrates the potential to expedite training speeds by a factor of up to 19.29x. Simultaneously, resource consumption of a single-GPU can be diminished by up to 69%, thus enabling the multiple GPU-based training and acceleration encompassing millions of nodes and billions of edges. Furthermore, our approach also maintains its competitiveness in downstream tasks.

{{</citation>}}


### (5/52) Empowering Clinicians and Democratizing Data Science: Large Language Models Automate Machine Learning for Clinical Studies (Soroosh Tayebi Arasteh et al., 2023)

{{<citation>}}

Soroosh Tayebi Arasteh, Tianyu Han, Mahshad Lotfinia, Christiane Kuhl, Jakob Nikolas Kather, Daniel Truhn, Sven Nebelung. (2023)  
**Empowering Clinicians and Democratizing Data Science: Large Language Models Automate Machine Learning for Clinical Studies**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: ChatGPT, Clinical, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.14120v1)  

---


**ABSTRACT**  
A knowledge gap persists between Machine Learning (ML) developers (e.g., data scientists) and practitioners (e.g., clinicians), hampering the full utilization of ML for clinical data analysis. We investigated the potential of the chatGPT Code Interpreter (CI), an extension of GPT-4, to bridge this gap and perform ML analyses efficiently. Real-world clinical datasets and study details from large trials across various medical specialties were presented to chatGPT CI without specific guidance. ChatGPT CI autonomously developed state-of-the-art ML models based on the original study's training data to predict clinical outcomes such as cancer development, cancer progression, disease complications, or biomarkers such as pathogenic gene sequences. Strikingly, these ML models matched or outperformed their published counterparts. We conclude that chatGPT CI offers a promising avenue to democratize ML in medicine, making advanced analytics accessible to non-ML experts and promoting broader applications in medical research and practice.

{{</citation>}}


### (6/52) Hybrid Transformer-RNN Architecture for Household Occupancy Detection Using Low-Resolution Smart Meter Data (Xinyu Liang et al., 2023)

{{<citation>}}

Xinyu Liang, Hao Wang. (2023)  
**Hybrid Transformer-RNN Architecture for Household Occupancy Detection Using Low-Resolution Smart Meter Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.14114v1)  

---


**ABSTRACT**  
Residential occupancy detection has become an enabling technology in today's urbanized world for various smart home applications, such as building automation, energy management, and improved security and comfort. Digitalization of the energy system provides smart meter data that can be used for occupancy detection in a non-intrusive manner without causing concerns regarding privacy and data security. In particular, deep learning techniques make it possible to infer occupancy from low-resolution smart meter data, such that the need for accurate occupancy detection with privacy preservation can be achieved. Our work is thus motivated to develop a privacy-aware and effective model for residential occupancy detection in contemporary living environments. Our model aims to leverage the advantages of both recurrent neural networks (RNNs), which are adept at capturing local temporal dependencies, and transformers, which are effective at handling global temporal dependencies. Our designed hybrid transformer-RNN model detects residential occupancy using hourly smart meter data, achieving an accuracy of nearly 92\% across households with diverse profiles. We validate the effectiveness of our method using a publicly accessible dataset and demonstrate its performance by comparing it with state-of-the-art models, including attention-based occupancy detection methods.

{{</citation>}}


### (7/52) Pruning the Unlabeled Data to Improve Semi-Supervised Learning (Guy Hacohen et al., 2023)

{{<citation>}}

Guy Hacohen, Daphna Weinshall. (2023)  
**Pruning the Unlabeled Data to Improve Semi-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Pruning, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.14058v1)  

---


**ABSTRACT**  
In the domain of semi-supervised learning (SSL), the conventional approach involves training a learner with a limited amount of labeled data alongside a substantial volume of unlabeled data, both drawn from the same underlying distribution. However, for deep learning models, this standard practice may not yield optimal results. In this research, we propose an alternative perspective, suggesting that distributions that are more readily separable could offer superior benefits to the learner as compared to the original distribution. To achieve this, we present PruneSSL, a practical technique for selectively removing examples from the original unlabeled dataset to enhance its separability. We present an empirical study, showing that although PruneSSL reduces the quantity of available training data for the learner, it significantly improves the performance of various competitive SSL algorithms, thereby achieving state-of-the-art results across several image classification tasks.

{{</citation>}}


## cs.DB (1)



### (8/52) Generations of Knowledge Graphs: The Crazy Ideas and the Business Impact (Xin Luna Dong, 2023)

{{<citation>}}

Xin Luna Dong. (2023)  
**Generations of Knowledge Graphs: The Crazy Ideas and the Business Impact**  

---
Primary Category: cs.DB  
Categories: cs-AI, cs-CL, cs-DB, cs.DB  
Keywords: Amazon, Google, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.14217v1)  

---


**ABSTRACT**  
Knowledge Graphs (KGs) have been used to support a wide range of applications, from web search to personal assistant. In this paper, we describe three generations of knowledge graphs: entity-based KGs, which have been supporting general search and question answering (e.g., at Google and Bing); text-rich KGs, which have been supporting search and recommendations for products, bio-informatics, etc. (e.g., at Amazon and Alibaba); and the emerging integration of KGs and LLMs, which we call dual neural KGs. We describe the characteristics of each generation of KGs, the crazy ideas behind the scenes in constructing such KGs, and the techniques developed over time to enable industry impact. In addition, we use KGs as examples to demonstrate a recipe to evolve research ideas from innovations to production practice, and then to the next level of innovations, to advance both science and business.

{{</citation>}}


## cs.CV (26)



### (9/52) Exploring the Transfer Learning Capabilities of CLIP in Domain Generalization for Diabetic Retinopathy (Sanoojan Baliah et al., 2023)

{{<citation>}}

Sanoojan Baliah, Fadillah A. Maani, Santosh Sanjeev, Muhammad Haris Khan. (2023)  
**Exploring the Transfer Learning Capabilities of CLIP in Domain Generalization for Diabetic Retinopathy**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14212v1)  

---


**ABSTRACT**  
Diabetic Retinopathy (DR), a leading cause of vision impairment, requires early detection and treatment. Developing robust AI models for DR classification holds substantial potential, but a key challenge is ensuring their generalization in unfamiliar domains with varying data distributions. To address this, our paper investigates cross-domain generalization, also known as domain generalization (DG), within the context of DR classification. DG, a challenging problem in the medical domain, is complicated by the difficulty of gathering labeled data across different domains, such as patient demographics and disease stages. Some recent studies have shown the effectiveness of using CLIP to handle the DG problem in natural images. In this study, we investigate CLIP's transfer learning capabilities and its potential for cross-domain generalization in diabetic retinopathy (DR) classification. We carry out comprehensive experiments to assess the efficacy and potential of CLIP in addressing DG for DR classification. Further, we introduce a multi-modal fine-tuning strategy named Context Optimization with Learnable Visual Tokens (CoOpLVT), which enhances context optimization by conditioning on visual features. Our findings demonstrate that the proposed method increases the F1-score by 1.8% over the baseline, thus underlining its promise for effective DG in DR classification. Our code is publicly available at https://github.com/Sanoojan/CLIP-DRDG.

{{</citation>}}


### (10/52) SketchDreamer: Interactive Text-Augmented Creative Sketch Ideation (Zhiyu Qu et al., 2023)

{{<citation>}}

Zhiyu Qu, Tao Xiang, Yi-Zhe Song. (2023)  
**SketchDreamer: Interactive Text-Augmented Creative Sketch Ideation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Sketch  
[Paper Link](http://arxiv.org/abs/2308.14191v1)  

---


**ABSTRACT**  
Artificial Intelligence Generated Content (AIGC) has shown remarkable progress in generating realistic images. However, in this paper, we take a step "backward" and address AIGC for the most rudimentary visual modality of human sketches. Our objective is on the creative nature of sketches, and that creative sketching should take the form of an interactive process. We further enable text to drive the sketch ideation process, allowing creativity to be freely defined, while simultaneously tackling the challenge of "I can't sketch". We present a method to generate controlled sketches using a text-conditioned diffusion model trained on pixel representations of images. Our proposed approach, referred to as SketchDreamer, integrates a differentiable rasteriser of Bezier curves that optimises an initial input to distil abstract semantic knowledge from a pretrained diffusion model. We utilise Score Distillation Sampling to learn a sketch that aligns with a given caption, which importantly enable both text and sketch to interact with the ideation process. Our objective is to empower non-professional users to create sketches and, through a series of optimisation processes, transform a narrative into a storyboard by expanding the text prompt while making minor adjustments to the sketch input. Through this work, we hope to aspire the way we create visual content, democratise the creative process, and inspire further research in enhancing human creativity in AIGC. The code is available at \url{https://github.com/WinKawaks/SketchDreamer}.

{{</citation>}}


### (11/52) AIGC for Various Data Modalities: A Survey (Lin Geng Foo et al., 2023)

{{<citation>}}

Lin Geng Foo, Hossein Rahmani, Jun Liu. (2023)  
**AIGC for Various Data Modalities: A Survey**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14177v1)  

---


**ABSTRACT**  
AI-generated content (AIGC) methods aim to produce text, images, videos, 3D assets, and other media using AI algorithms. Due to its wide range of applications and the demonstrated potential of recent works, AIGC developments have been attracting a lot of attention recently, and AIGC methods have been developed for various data modalities, such as image, video, text, 3D shape (as voxels, point clouds, meshes, and neural implicit fields), 3D scene, 3D human avatar (body and head), 3D motion, and audio -- each presenting different characteristics and challenges. Furthermore, there have also been many significant developments in cross-modality AIGC methods, where generative methods can receive conditioning input in one modality and produce outputs in another. Examples include going from various modalities to image, video, 3D shape, 3D scene, 3D avatar (body and head), 3D motion (skeleton and avatar), and audio modalities. In this paper, we provide a comprehensive review of AIGC methods across different data modalities, including both single-modal and cross-modality methods, highlighting the various challenges, representative works, and recent technical directions in each setting. We also present comparative results on several benchmark datasets in various modalities. Moreover, we also discuss the challenges and potential future research directions.

{{</citation>}}


### (12/52) A Unified Transformer-based Network for multimodal Emotion Recognition (Kamran Ali et al., 2023)

{{<citation>}}

Kamran Ali, Charles E. Hughes. (2023)  
**A Unified Transformer-based Network for multimodal Emotion Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Emotion Recognition, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2308.14160v1)  

---


**ABSTRACT**  
The development of transformer-based models has resulted in significant advances in addressing various vision and NLP-based research challenges. However, the progress made in transformer-based methods has not been effectively applied to biosensing research. This paper presents a novel Unified Biosensor-Vision Multi-modal Transformer-based (UBVMT) method to classify emotions in an arousal-valence space by combining a 2D representation of an ECG/PPG signal with the face information. To achieve this goal, we first investigate and compare the unimodal emotion recognition performance of three image-based representations of the ECG/PPG signal. We then present our UBVMT network which is trained to perform emotion recognition by combining the 2D image-based representation of the ECG/PPG signal and the facial expression features. Our unified transformer model consists of homogeneous transformer blocks that take as an input the 2D representation of the ECG/PPG signal and the corresponding face frame for emotion representation learning with minimal modality-specific design. Our UBVMT model is trained by reconstructing masked patches of video frames and 2D images of ECG/PPG signals, and contrastive modeling to align face and ECG/PPG data. Extensive experiments on the MAHNOB-HCI and DEAP datasets show that our Unified UBVMT-based model produces comparable results to the state-of-the-art techniques.

{{</citation>}}


### (13/52) Sparse Sampling Transformer with Uncertainty-Driven Ranking for Unified Removal of Raindrops and Rain Streaks (Sixiang Chen et al., 2023)

{{<citation>}}

Sixiang Chen, Tian Ye, Jinbin Bai, Erkang Chen, Jun Shi, Lei Zhu. (2023)  
**Sparse Sampling Transformer with Uncertainty-Driven Ranking for Unified Removal of Raindrops and Rain Streaks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.14153v1)  

---


**ABSTRACT**  
In the real world, image degradations caused by rain often exhibit a combination of rain streaks and raindrops, thereby increasing the challenges of recovering the underlying clean image. Note that the rain streaks and raindrops have diverse shapes, sizes, and locations in the captured image, and thus modeling the correlation relationship between irregular degradations caused by rain artifacts is a necessary prerequisite for image deraining. This paper aims to present an efficient and flexible mechanism to learn and model degradation relationships in a global view, thereby achieving a unified removal of intricate rain scenes. To do so, we propose a Sparse Sampling Transformer based on Uncertainty-Driven Ranking, dubbed UDR-S2Former. Compared to previous methods, our UDR-S2Former has three merits. First, it can adaptively sample relevant image degradation information to model underlying degradation relationships. Second, explicit application of the uncertainty-driven ranking strategy can facilitate the network to attend to degradation features and understand the reconstruction process. Finally, experimental results show that our UDR-S2Former clearly outperforms state-of-the-art methods for all benchmarks.

{{</citation>}}


### (14/52) Unaligned 2D to 3D Translation with Conditional Vector-Quantized Code Diffusion using Transformers (Abril Corona-Figueroa et al., 2023)

{{<citation>}}

Abril Corona-Figueroa, Sam Bond-Taylor, Neelanjan Bhowmik, Yona Falinie A. Gaus, Toby P. Breckon, Hubert P. H. Shum, Chris G. Willcocks. (2023)  
**Unaligned 2D to 3D Translation with Conditional Vector-Quantized Code Diffusion using Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.14152v1)  

---


**ABSTRACT**  
Generating 3D images of complex objects conditionally from a few 2D views is a difficult synthesis problem, compounded by issues such as domain gap and geometric misalignment. For instance, a unified framework such as Generative Adversarial Networks cannot achieve this unless they explicitly define both a domain-invariant and geometric-invariant joint latent distribution, whereas Neural Radiance Fields are generally unable to handle both issues as they optimize at the pixel level. By contrast, we propose a simple and novel 2D to 3D synthesis approach based on conditional diffusion with vector-quantized codes. Operating in an information-rich code space enables high-resolution 3D synthesis via full-coverage attention across the views. Specifically, we generate the 3D codes (e.g. for CT images) conditional on previously generated 3D codes and the entire codebook of two 2D views (e.g. 2D X-rays). Qualitative and quantitative results demonstrate state-of-the-art performance over specialized methods across varied evaluation criteria, including fidelity metrics such as density, coverage, and distortion metrics for two complex volumetric imagery datasets from in real-world scenarios.

{{</citation>}}


### (15/52) Synergizing Contrastive Learning and Optimal Transport for 3D Point Cloud Domain Adaptation (Siddharth Katageri et al., 2023)

{{<citation>}}

Siddharth Katageri, Arkadipta De, Chaitanya Devaguptapu, VSSV Prasad, Charu Sharma, Manohar Kaul. (2023)  
**Synergizing Contrastive Learning and Optimal Transport for 3D Point Cloud Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.14126v1)  

---


**ABSTRACT**  
Recently, the fundamental problem of unsupervised domain adaptation (UDA) on 3D point clouds has been motivated by a wide variety of applications in robotics, virtual reality, and scene understanding, to name a few. The point cloud data acquisition procedures manifest themselves as significant domain discrepancies and geometric variations among both similar and dissimilar classes. The standard domain adaptation methods developed for images do not directly translate to point cloud data because of their complex geometric nature. To address this challenge, we leverage the idea of multimodality and alignment between distributions. We propose a new UDA architecture for point cloud classification that benefits from multimodal contrastive learning to get better class separation in both domains individually. Further, the use of optimal transport (OT) aims at learning source and target data distributions jointly to reduce the cross-domain shift and provide a better alignment. We conduct a comprehensive empirical study on PointDA-10 and GraspNetPC-10 and show that our method achieves state-of-the-art performance on GraspNetPC-10 (with approx 4-12% margin) and best average performance on PointDA-10. Our ablation studies and decision boundary analysis also validate the significance of our contrastive learning module and OT alignment.

{{</citation>}}


### (16/52) Semi-Supervised Learning in the Few-Shot Zero-Shot Scenario (Noam Fluss et al., 2023)

{{<citation>}}

Noam Fluss, Guy Hacohen, Daphna Weinshall. (2023)  
**Semi-Supervised Learning in the Few-Shot Zero-Shot Scenario**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Few-Shot, Semi-Supervised, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.14119v1)  

---


**ABSTRACT**  
Semi-Supervised Learning (SSL) leverages both labeled and unlabeled data to improve model performance. Traditional SSL methods assume that labeled and unlabeled data share the same label space. However, in real-world applications, especially when the labeled training set is small, there may be classes that are missing from the labeled set. Existing frameworks aim to either reject all unseen classes (open-set SSL) or to discover unseen classes by partitioning an unlabeled set during training (open-world SSL). In our work, we construct a classifier for points from both seen and unseen classes. Our approach is based on extending an existing SSL method, such as FlexMatch, by incorporating an additional entropy loss. This enhancement allows our method to improve the performance of any existing SSL method in the classification of both seen and unseen classes. We demonstrate large improvement gains over state-of-the-art SSL, open-set SSL, and open-world SSL methods, on two benchmark image classification data sets, CIFAR-100 and STL-10. The gains are most pronounced when the labeled data is severely limited (1-25 labeled examples per class).

{{</citation>}}


### (17/52) Superpixels algorithms through network community detection (Anthony Perez, 2023)

{{<citation>}}

Anthony Perez. (2023)  
**Superpixels algorithms through network community detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2308.14101v1)  

---


**ABSTRACT**  
Community detection is a powerful tool from complex networks analysis that finds applications in various research areas. Several image segmentation methods rely for instance on community detection algorithms as a black box in order to compute undersegmentations, i.e. a small number of regions that represent areas of interest of the image. However, to the best of our knowledge, the efficiency of such an approach w.r.t. superpixels, that aim at representing the image at a smaller level while preserving as much as possible original information, has been neglected so far. The only related work seems to be the one by Liu et. al. (IET Image Processing, 2022) that developed a superpixels algorithm using a so-called modularity maximization approach, leading to relevant results. We follow this line of research by studying the efficiency of superpixels computed by state-of-the-art community detection algorithms on a 4-connected pixel graph, so-called pixel-grid. We first detect communities on such a graph and then apply a simple merging procedure that allows to obtain the desired number of superpixels. As we shall see, such methods result in the computation of relevant superpixels as emphasized by both qualitative and quantitative experiments, according to different widely-used metrics based on ground-truth comparison or on superpixels only. We observe that the choice of the community detection algorithm has a great impact on the number of communities and hence on the merging procedure. Similarly, small variations on the pixel-grid may provide different results from both qualitative and quantitative viewpoints. For the sake of completeness, we compare our results with those of several state-of-the-art superpixels algorithms as computed by Stutz et al. (Computer Vision and Image Understanding, 2018).

{{</citation>}}


### (18/52) Rethinking Exemplars for Continual Semantic Segmentation in Endoscopy Scenes: Entropy-based Mini-Batch Pseudo-Replay (Guankun Wang et al., 2023)

{{<citation>}}

Guankun Wang, Long Bai, Yanan Wu, Tong Chen, Hongliang Ren. (2023)  
**Rethinking Exemplars for Continual Semantic Segmentation in Endoscopy Scenes: Entropy-based Mini-Batch Pseudo-Replay**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.14100v1)  

---


**ABSTRACT**  
Endoscopy is a widely used technique for the early detection of diseases or robotic-assisted minimally invasive surgery (RMIS). Numerous deep learning (DL)-based research works have been developed for automated diagnosis or processing of endoscopic view. However, existing DL models may suffer from catastrophic forgetting. When new target classes are introduced over time or cross institutions, the performance of old classes may suffer severe degradation. More seriously, data privacy and storage issues may lead to the unavailability of old data when updating the model. Therefore, it is necessary to develop a continual learning (CL) methodology to solve the problem of catastrophic forgetting in endoscopic image segmentation. To tackle this, we propose a Endoscopy Continual Semantic Segmentation (EndoCSS) framework that does not involve the storage and privacy issues of exemplar data. The framework includes a mini-batch pseudo-replay (MB-PR) mechanism and a self-adaptive noisy cross-entropy (SAN-CE) loss. The MB-PR strategy circumvents privacy and storage issues by generating pseudo-replay images through a generative model. Meanwhile, the MB-PR strategy can also correct the model deviation to the replay data and current training data, which is aroused by the significant difference in the amount of current and replay images. Therefore, the model can perform effective representation learning on both new and old tasks. SAN-CE loss can help model fitting by adjusting the model's output logits, and also improve the robustness of training. Extensive continual semantic segmentation (CSS) experiments on public datasets demonstrate that our method can robustly and effectively address the catastrophic forgetting brought by class increment in endoscopy scenes. The results show that our framework holds excellent potential for real-world deployment in a streaming learning manner.

{{</citation>}}


### (19/52) A comprehensive review on Plant Leaf Disease detection using Deep learning (Sumaya Mustofa et al., 2023)

{{<citation>}}

Sumaya Mustofa, Md Mehedi Hasan Munna, Yousuf Rayhan Emon, Golam Rabbany, Md Taimur Ahad. (2023)  
**A comprehensive review on Plant Leaf Disease detection using Deep learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.14087v1)  

---


**ABSTRACT**  
Leaf disease is a common fatal disease for plants. Early diagnosis and detection is necessary in order to improve the prognosis of leaf diseases affecting plant. For predicting leaf disease, several automated systems have already been developed using different plant pathology imaging modalities. This paper provides a systematic review of the literature on leaf disease-based models for the diagnosis of various plant leaf diseases via deep learning. The advantages and limitations of different deep learning models including Vision Transformer (ViT), Deep convolutional neural network (DCNN), Convolutional neural network (CNN), Residual Skip Network-based Super-Resolution for Leaf Disease Detection (RSNSR-LDD), Disease Detection Network (DDN), and YOLO (You only look once) are described in this review. The review also shows that the studies related to leaf disease detection applied different deep learning models to a number of publicly available datasets. For comparing the performance of the models, different metrics such as accuracy, precision, recall, etc. were used in the existing studies.

{{</citation>}}


### (20/52) A Novel Multi-scale Attention Feature Extraction Block for Aerial Remote Sensing Image Classification (Chiranjibi Sitaula et al., 2023)

{{<citation>}}

Chiranjibi Sitaula, Jagannath Aryal, Avik Bhattacharya. (2023)  
**A Novel Multi-scale Attention Feature Extraction Block for Aerial Remote Sensing Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Attention, Image Classification  
[Paper Link](http://arxiv.org/abs/2308.14076v1)  

---


**ABSTRACT**  
Classification of very high-resolution (VHR) aerial remote sensing (RS) images is a well-established research area in the remote sensing community as it provides valuable spatial information for decision-making. Existing works on VHR aerial RS image classification produce an excellent classification performance; nevertheless, they have a limited capability to well-represent VHR RS images having complex and small objects, thereby leading to performance instability. As such, we propose a novel plug-and-play multi-scale attention feature extraction block (MSAFEB) based on multi-scale convolution at two levels with skip connection, producing discriminative/salient information at a deeper/finer level. The experimental study on two benchmark VHR aerial RS image datasets (AID and NWPU) demonstrates that our proposal achieves a stable/consistent performance (minimum standard deviation of $0.002$) and competent overall classification performance (AID: 95.85\% and NWPU: 94.09\%).

{{</citation>}}


### (21/52) Nonrigid Object Contact Estimation With Regional Unwrapping Transformer (Wei Xie et al., 2023)

{{<citation>}}

Wei Xie, Zimeng Zhao, Shiying Li, Binghui Zuo, Yangang Wang. (2023)  
**Nonrigid Object Contact Estimation With Regional Unwrapping Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.14074v1)  

---


**ABSTRACT**  
Acquiring contact patterns between hands and nonrigid objects is a common concern in the vision and robotics community. However, existing learning-based methods focus more on contact with rigid ones from monocular images. When adopting them for nonrigid contact, a major problem is that the existing contact representation is restricted by the geometry of the object. Consequently, contact neighborhoods are stored in an unordered manner and contact features are difficult to align with image cues. At the core of our approach lies a novel hand-object contact representation called RUPs (Region Unwrapping Profiles), which unwrap the roughly estimated hand-object surfaces as multiple high-resolution 2D regional profiles. The region grouping strategy is consistent with the hand kinematic bone division because they are the primitive initiators for a composite contact pattern. Based on this representation, our Regional Unwrapping Transformer (RUFormer) learns the correlation priors across regions from monocular inputs and predicts corresponding contact and deformed transformations. Our experiments demonstrate that the proposed framework can robustly estimate the deformed degrees and deformed transformations, which makes it suitable for both nonrigid and rigid contact.

{{</citation>}}


### (22/52) DETDet: Dual Ensemble Teeth Detection (Kyoungyeon Choi et al., 2023)

{{<citation>}}

Kyoungyeon Choi, Jaewon Shin, Eunyi Lyou. (2023)  
**DETDet: Dual Ensemble Teeth Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.14070v1)  

---


**ABSTRACT**  
The field of dentistry is in the era of digital transformation. Particularly, artificial intelligence is anticipated to play a significant role in digital dentistry. AI holds the potential to significantly assist dental practitioners and elevate diagnostic accuracy. In alignment with this vision, the 2023 MICCAI DENTEX challenge aims to enhance the performance of dental panoramic X-ray diagnosis and enumeration through technological advancement. In response, we introduce DETDet, a Dual Ensemble Teeth Detection network. DETDet encompasses two distinct modules dedicated to enumeration and diagnosis. Leveraging the advantages of teeth mask data, we employ Mask-RCNN for the enumeration module. For the diagnosis module, we adopt an ensemble model comprising DiffusionDet and DINO. To further enhance precision scores, we integrate a complementary module to harness the potential of unlabeled data. The code for our approach will be made accessible at https://github.com/Bestever-choi/Evident

{{</citation>}}


### (23/52) Multi-model fusion for Aerial Vision and Dialog Navigation based on human attention aids (Xinyi Wang et al., 2023)

{{<citation>}}

Xinyi Wang, Xuan Cui, Danxu Li, Fang Liu, Licheng Jiao. (2023)  
**Multi-model fusion for Aerial Vision and Dialog Navigation based on human attention aids**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Dialog, Drone, LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2308.14064v1)  

---


**ABSTRACT**  
Drones have been widely used in many areas of our daily lives. It relieves people of the burden of holding a controller all the time and makes drone control easier to use for people with disabilities or occupied hands. However, the control of aerial robots is more complicated compared to normal robots due to factors such as uncontrollable height. Therefore, it is crucial to develop an intelligent UAV that has the ability to talk to humans and follow natural language commands. In this report, we present an aerial navigation task for the 2023 ICCV Conversation History. Based on the AVDN dataset containing more than 3k recorded navigation trajectories and asynchronous human-robot conversations, we propose an effective method of fusion training of Human Attention Aided Transformer model (HAA-Transformer) and Human Attention Aided LSTM (HAA-LSTM) model, which achieves the prediction of the navigation routing points and human attention. The method not only achieves high SR and SPL metrics, but also shows a 7% improvement in GP metrics compared to the baseline model.

{{</citation>}}


### (24/52) Hierarchical Contrastive Learning for Pattern-Generalizable Image Corruption Detection (Xin Feng et al., 2023)

{{<citation>}}

Xin Feng, Yifeng Xu, Guangming Lu, Wenjie Pei. (2023)  
**Hierarchical Contrastive Learning for Pattern-Generalizable Image Corruption Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.14061v1)  

---


**ABSTRACT**  
Effective image restoration with large-size corruptions, such as blind image inpainting, entails precise detection of corruption region masks which remains extremely challenging due to diverse shapes and patterns of corruptions. In this work, we present a novel method for automatic corruption detection, which allows for blind corruption restoration without known corruption masks. Specifically, we develop a hierarchical contrastive learning framework to detect corrupted regions by capturing the intrinsic semantic distinctions between corrupted and uncorrupted regions. In particular, our model detects the corrupted mask in a coarse-to-fine manner by first predicting a coarse mask by contrastive learning in low-resolution feature space and then refines the uncertain area of the mask by high-resolution contrastive learning. A specialized hierarchical interaction mechanism is designed to facilitate the knowledge propagation of contrastive learning in different scales, boosting the modeling performance substantially. The detected multi-scale corruption masks are then leveraged to guide the corruption restoration. Detecting corrupted regions by learning the contrastive distinctions rather than the semantic patterns of corruptions, our model has well generalization ability across different corruption patterns. Extensive experiments demonstrate following merits of our model: 1) the superior performance over other methods on both corruption detection and various image restoration tasks including blind inpainting and watermark removal, and 2) strong generalization across different corruption patterns such as graffiti, random noise or other image content. Codes and trained weights are available at https://github.com/xyfJASON/HCL .

{{</citation>}}


### (25/52) PECon: Contrastive Pretraining to Enhance Feature Alignment between CT and EHR Data for Improved Pulmonary Embolism Diagnosis (Santosh Sanjeev et al., 2023)

{{<citation>}}

Santosh Sanjeev, Salwa K. Al Khatib, Mai A. Shaaban, Ibrahim Almakky, Vijay Ram Papineni, Mohammad Yaqub. (2023)  
**PECon: Contrastive Pretraining to Enhance Feature Alignment between CT and EHR Data for Improved Pulmonary Embolism Diagnosis**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.14050v1)  

---


**ABSTRACT**  
Previous deep learning efforts have focused on improving the performance of Pulmonary Embolism(PE) diagnosis from Computed Tomography (CT) scans using Convolutional Neural Networks (CNN). However, the features from CT scans alone are not always sufficient for the diagnosis of PE. CT scans along with electronic heath records (EHR) can provide a better insight into the patients condition and can lead to more accurate PE diagnosis. In this paper, we propose Pulmonary Embolism Detection using Contrastive Learning (PECon), a supervised contrastive pretraining strategy that employs both the patients CT scans as well as the EHR data, aiming to enhance the alignment of feature representations between the two modalities and leverage information to improve the PE diagnosis. In order to achieve this, we make use of the class labels and pull the sample features of the same class together, while pushing away those of the other class. Results show that the proposed work outperforms the existing techniques and achieves state-of-the-art performance on the RadFusion dataset with an F1-score of 0.913, accuracy of 0.90 and an AUROC of 0.943. Furthermore, we also explore the explainability of our approach in comparison to other methods. Our code is publicly available at https://github.com/BioMedIA-MBZUAI/PECon.

{{</citation>}}


### (26/52) MB-TaylorFormer: Multi-branch Efficient Transformer Expanded by Taylor Formula for Image Dehazing (Yuwei Qiu et al., 2023)

{{<citation>}}

Yuwei Qiu, Kaihao Zhang, Chenxi Wang, Wenhan Luo, Hongdong Li, Zhi Jin. (2023)  
**MB-TaylorFormer: Multi-branch Efficient Transformer Expanded by Taylor Formula for Image Dehazing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.14036v1)  

---


**ABSTRACT**  
In recent years, Transformer networks are beginning to replace pure convolutional neural networks (CNNs) in the field of computer vision due to their global receptive field and adaptability to input. However, the quadratic computational complexity of softmax-attention limits the wide application in image dehazing task, especially for high-resolution images. To address this issue, we propose a new Transformer variant, which applies the Taylor expansion to approximate the softmax-attention and achieves linear computational complexity. A multi-scale attention refinement module is proposed as a complement to correct the error of the Taylor expansion. Furthermore, we introduce a multi-branch architecture with multi-scale patch embedding to the proposed Transformer, which embeds features by overlapping deformable convolution of different scales. The design of multi-scale patch embedding is based on three key ideas: 1) various sizes of the receptive field; 2) multi-level semantic information; 3) flexible shapes of the receptive field. Our model, named Multi-branch Transformer expanded by Taylor formula (MB-TaylorFormer), can embed coarse to fine features more flexibly at the patch embedding stage and capture long-distance pixel interactions with limited computational cost. Experimental results on several dehazing benchmarks show that MB-TaylorFormer achieves state-of-the-art (SOTA) performance with a light computational burden. The source code and pre-trained models are available at https://github.com/FVL2020/ICCV-2023-MB-TaylorFormer.

{{</citation>}}


### (27/52) Forensic Histopathological Recognition via a Context-Aware MIL Network Powered by Self-Supervised Contrastive Learning (Chen Shen et al., 2023)

{{<citation>}}

Chen Shen, Jun Zhang, Xinggong Liang, Zeyi Hao, Kehan Li, Fan Wang, Zhenyuan Wang, Chunfeng Lian. (2023)  
**Forensic Histopathological Recognition via a Context-Aware MIL Network Powered by Self-Supervised Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI, Contrastive Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.14030v1)  

---


**ABSTRACT**  
Forensic pathology is critical in analyzing death manner and time from the microscopic aspect to assist in the establishment of reliable factual bases for criminal investigation. In practice, even the manual differentiation between different postmortem organ tissues is challenging and relies on expertise, considering that changes like putrefaction and autolysis could significantly change typical histopathological appearance. Developing AI-based computational pathology techniques to assist forensic pathologists is practically meaningful, which requires reliable discriminative representation learning to capture tissues' fine-grained postmortem patterns. To this end, we propose a framework called FPath, in which a dedicated self-supervised contrastive learning strategy and a context-aware multiple-instance learning (MIL) block are designed to learn discriminative representations from postmortem histopathological images acquired at varying magnification scales. Our self-supervised learning step leverages multiple complementary contrastive losses and regularization terms to train a double-tier backbone for fine-grained and informative patch/instance embedding. Thereafter, the context-aware MIL adaptively distills from the local instances a holistic bag/image-level representation for the recognition task. On a large-scale database of $19,607$ experimental rat postmortem images and $3,378$ real-world human decedent images, our FPath led to state-of-the-art accuracy and promising cross-domain generalization in recognizing seven different postmortem tissues. The source code will be released on \href{https://github.com/ladderlab-xjtu/forensic_pathology}{https://github.com/ladderlab-xjtu/forensic\_pathology}.

{{</citation>}}


### (28/52) Balanced Representation Learning for Long-tailed Skeleton-based Action Recognition (Hongda Liu et al., 2023)

{{<citation>}}

Hongda Liu, Yunlong Wang, Min Ren, Junxing Hu, Zhengquan Luo, Guangqi Hou, Zhenan Sun. (2023)  
**Balanced Representation Learning for Long-tailed Skeleton-based Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.14024v1)  

---


**ABSTRACT**  
Skeleton-based action recognition has recently made significant progress. However, data imbalance is still a great challenge in real-world scenarios. The performance of current action recognition algorithms declines sharply when training data suffers from heavy class imbalance. The imbalanced data actually degrades the representations learned by these methods and becomes the bottleneck for action recognition. How to learn unbiased representations from imbalanced action data is the key to long-tailed action recognition. In this paper, we propose a novel balanced representation learning method to address the long-tailed problem in action recognition. Firstly, a spatial-temporal action exploration strategy is presented to expand the sample space effectively, generating more valuable samples in a rebalanced manner. Secondly, we design a detached action-aware learning schedule to further mitigate the bias in the representation space. The schedule detaches the representation learning of tail classes from training and proposes an action-aware loss to impose more effective constraints. Additionally, a skip-modal representation is proposed to provide complementary structural information. The proposed method is validated on four skeleton datasets, NTU RGB+D 60, NTU RGB+D 120, NW-UCLA, and Kinetics. It not only achieves consistently large improvement compared to the state-of-the-art (SOTA) methods, but also demonstrates a superior generalization capacity through extensive experiments. Our code is available at https://github.com/firework8/BRL.

{{</citation>}}


### (29/52) Domain-Specificity Inducing Transformers for Source-Free Domain Adaptation (Sunandini Sanyal et al., 2023)

{{<citation>}}

Sunandini Sanyal, Ashish Ramayee Asokan, Suvaansh Bhambri, Akshay Kulkarni, Jogendra Nath Kundu, R. Venkatesh Babu. (2023)  
**Domain-Specificity Inducing Transformers for Source-Free Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.14023v1)  

---


**ABSTRACT**  
Conventional Domain Adaptation (DA) methods aim to learn domain-invariant feature representations to improve the target adaptation performance. However, we motivate that domain-specificity is equally important since in-domain trained models hold crucial domain-specific properties that are beneficial for adaptation. Hence, we propose to build a framework that supports disentanglement and learning of domain-specific factors and task-specific factors in a unified model. Motivated by the success of vision transformers in several multi-modal vision problems, we find that queries could be leveraged to extract the domain-specific factors. Hence, we propose a novel Domain-specificity-inducing Transformer (DSiT) framework for disentangling and learning both domain-specific and task-specific factors. To achieve disentanglement, we propose to construct novel Domain-Representative Inputs (DRI) with domain-specific information to train a domain classifier with a novel domain token. We are the first to utilize vision transformers for domain adaptation in a privacy-oriented source-free setting, and our approach achieves state-of-the-art performance on single-source, multi-source, and multi-target benchmarks

{{</citation>}}


### (30/52) VQ-Font: Few-Shot Font Generation with Structure-Aware Enhancement and Quantization (Mingshuai Yao et al., 2023)

{{<citation>}}

Mingshuai Yao, Yabo Zhang, Xianhui Lin, Xiaoming Li, Wangmeng Zuo. (2023)  
**VQ-Font: Few-Shot Font Generation with Structure-Aware Enhancement and Quantization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Quantization  
[Paper Link](http://arxiv.org/abs/2308.14018v1)  

---


**ABSTRACT**  
Few-shot font generation is challenging, as it needs to capture the fine-grained stroke styles from a limited set of reference glyphs, and then transfer to other characters, which are expected to have similar styles. However, due to the diversity and complexity of Chinese font styles, the synthesized glyphs of existing methods usually exhibit visible artifacts, such as missing details and distorted strokes. In this paper, we propose a VQGAN-based framework (i.e., VQ-Font) to enhance glyph fidelity through token prior refinement and structure-aware enhancement. Specifically, we pre-train a VQGAN to encapsulate font token prior within a codebook. Subsequently, VQ-Font refines the synthesized glyphs with the codebook to eliminate the domain gap between synthesized and real-world strokes. Furthermore, our VQ-Font leverages the inherent design of Chinese characters, where structure components such as radicals and character components are combined in specific arrangements, to recalibrate fine-grained styles based on references. This process improves the matching and fusion of styles at the structure level. Both modules collaborate to enhance the fidelity of the generated fonts. Experiments on a collected font dataset show that our VQ-Font outperforms the competing methods both quantitatively and qualitatively, especially in generating challenging styles.

{{</citation>}}


### (31/52) Towards Fast and Accurate Image-Text Retrieval with Self-Supervised Fine-Grained Alignment (Jiamin Zhuang et al., 2023)

{{<citation>}}

Jiamin Zhuang, Jing Yu, Yang Ding, Xiangyan Qu, Yue Hu. (2023)  
**Towards Fast and Accurate Image-Text Retrieval with Self-Supervised Fine-Grained Alignment**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.14009v1)  

---


**ABSTRACT**  
Image-text retrieval requires the system to bridge the heterogenous gap between vision and language for accurate retrieval while keeping the network lightweight-enough for efficient retrieval. Existing trade-off solutions mainly study from the view of incorporating cross-modal interactions with the independent-embedding framework or leveraging stronger pretrained encoders, which still demand time-consuming similarity measurement or heavyweight model structure in the retrieval stage. In this work, we propose an image-text alignment module SelfAlign on top of the independent-embedding framework, which improves the retrieval accuracy while maintains the retrieval efficiency without extra supervision. SelfAlign contains two collaborative sub-modules that force image-text alignment at both concept level and context level by self-supervised contrastive learning. It does not require cross-modal embedding interactions during training while maintaining independent image and text encoders during retrieval. With comparable time cost, SelfAlign consistently boosts the accuracy of state-of-the-art non-pretraining independent-embedding models respectively by 9.1%, 4.2% and 6.6% in terms of R@sum score on Flickr30K, MSCOCO 1K and MS-COCO 5K datasets. The retrieval accuracy also outperforms most existing interactive-embedding models with orders of magnitude decrease in retrieval time. The source code is available at: https://github.com/Zjamie813/SelfAlign.

{{</citation>}}


### (32/52) Computation-efficient Deep Learning for Computer Vision: A Survey (Yulin Wang et al., 2023)

{{<citation>}}

Yulin Wang, Yizeng Han, Chaofei Wang, Shiji Song, Qi Tian, Gao Huang. (2023)  
**Computation-efficient Deep Learning for Computer Vision: A Survey**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2308.13998v1)  

---


**ABSTRACT**  
Over the past decade, deep learning models have exhibited considerable advancements, reaching or even exceeding human-level performance in a range of visual perception tasks. This remarkable progress has sparked interest in applying deep networks to real-world applications, such as autonomous vehicles, mobile devices, robotics, and edge computing. However, the challenge remains that state-of-the-art models usually demand significant computational resources, leading to impractical power consumption, latency, or carbon emissions in real-world scenarios. This trade-off between effectiveness and efficiency has catalyzed the emergence of a new research focus: computationally efficient deep learning, which strives to achieve satisfactory performance while minimizing the computational cost during inference. This review offers an extensive analysis of this rapidly evolving field by examining four key areas: 1) the development of static or dynamic light-weighted backbone models for the efficient extraction of discriminative deep representations; 2) the specialized network architectures or algorithms tailored for specific computer vision tasks; 3) the techniques employed for compressing deep learning models; and 4) the strategies for deploying efficient deep networks on hardware platforms. Additionally, we provide a systematic discussion on the critical challenges faced in this domain, such as network architecture design, training schemes, practical efficiency, and more realistic model compression approaches, as well as potential future research directions.

{{</citation>}}


### (33/52) JL-lemma derived Optimal Projections for Discriminative Dictionary Learning (G. Madhuri et al., 2023)

{{<citation>}}

G. Madhuri, Atul Negi. (2023)  
**JL-lemma derived Optimal Projections for Discriminative Dictionary Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-SP  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2308.13991v1)  

---


**ABSTRACT**  
To overcome difficulties in classifying large dimensionality data with a large number of classes, we propose a novel approach called JLSPCADL. This paper uses the Johnson-Lindenstrauss (JL) Lemma to select the dimensionality of a transformed space in which a discriminative dictionary can be learned for signal classification. Rather than reducing dimensionality via random projections, as is often done with JL, we use a projection transformation matrix derived from Modified Supervised PC Analysis (M-SPCA) with the JL-prescribed dimension.   JLSPCADL provides a heuristic to deduce suitable distortion levels and the corresponding Suitable Description Length (SDL) of dictionary atoms to derive an optimal feature space and thus the SDL of dictionary atoms for better classification. Unlike state-of-the-art dimensionality reduction-based dictionary learning methods, a projection transformation matrix derived in a single step from M-SPCA provides maximum feature-label consistency of the transformed space while preserving the cluster structure of the original data. Despite confusing pairs, the dictionary for the transformed space generates discriminative sparse coefficients, with fewer training samples. Experimentation demonstrates that JLSPCADL scales well with an increasing number of classes and dimensionality. Improved label consistency of features due to M-SPCA helps to classify better. Further, the complexity of training a discriminative dictionary is significantly reduced by using SDL. Experimentation on OCR and face recognition datasets shows relatively better classification performance than other supervised dictionary learning algorithms.

{{</citation>}}


### (34/52) Enhancing Bloodstain Analysis Through AI-Based Segmentation: Leveraging Segment Anything Model for Crime Scene Investigation (Zihan Dong et al., 2023)

{{<citation>}}

Zihan Dong, ZhengDong Zhang. (2023)  
**Enhancing Bloodstain Analysis Through AI-Based Segmentation: Leveraging Segment Anything Model for Crime Scene Investigation**  

---
Primary Category: cs.CV  
Categories: 68-xx, cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.13979v1)  

---


**ABSTRACT**  
Bloodstain pattern analysis plays a crucial role in crime scene investigations by providing valuable information through the study of unique blood patterns. Conventional image analysis methods, like Thresholding and Contrast, impose stringent requirements on the image background and is labor-intensive in the context of droplet image segmentation. The Segment Anything Model (SAM), a recently proposed method for extensive image recognition, is yet to be adequately assessed for its accuracy and efficiency on bloodstain image segmentation. This paper explores the application of pre-trained SAM and fine-tuned SAM on bloodstain image segmentation with diverse image backgrounds. Experiment results indicate that both pre-trained and fine-tuned SAM perform the bloodstain image segmentation task with satisfactory accuracy and efficiency, while fine-tuned SAM achieves an overall 2.2\% accuracy improvement than pre-trained SAM and 4.70\% acceleration in terms of speed for image recognition. Analysis of factors that influence bloodstain recognition is carried out. This research demonstrates the potential application of SAM on bloodstain image segmentation, showcasing the effectiveness of Artificial Intelligence application in criminology research. We release all code and demos at \url{https://github.com/Zdong104/Bloodstain_Analysis_Ai_Tool}

{{</citation>}}


## cs.AI (3)



### (35/52) Symbolic and Language Agnostic Large Language Models (Walid S. Saba, 2023)

{{<citation>}}

Walid S. Saba. (2023)  
**Symbolic and Language Agnostic Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.14199v1)  

---


**ABSTRACT**  
We argue that the relative success of large language models (LLMs) is not a reflection on the symbolic vs. subsymbolic debate but a reflection on employing an appropriate strategy of bottom-up reverse engineering of language at scale. However, due to the subsymbolic nature of these models whatever knowledge these systems acquire about language will always be buried in millions of microfeatures (weights) none of which is meaningful on its own. Moreover, and due to their stochastic nature, these models will often fail in capturing various inferential aspects that are prevalent in natural language. What we suggest here is employing the successful bottom-up strategy in a symbolic setting, producing symbolic, language agnostic and ontologically grounded large language models.

{{</citation>}}


### (36/52) Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum (Shen Gao et al., 2023)

{{<citation>}}

Shen Gao, Zhengliang Shi, Minghang Zhu, Bowen Fang, Xin Xin, Pengjie Ren, Zhumin Chen, Jun Ma. (2023)  
**Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.14034v1)  

---


**ABSTRACT**  
Augmenting large language models (LLMs) with external tools has emerged as a promising approach to extending the capability of LLMs. Although some works employ open-source LLMs for the tool learning task, most of them are trained in a controlled environment in which LLMs only learn to execute the human-provided tools. However, selecting proper tools from the large toolset is also a crucial ability for the tool learning model to be applied in real-world applications. Existing methods usually directly employ self-instruction methods to train the model, which ignores differences in tool complexity. In this paper, we propose the Confucius, a novel tool learning framework to train LLM to use complicated tools in real-world scenarios, which contains two main phases: (1) We first propose a multi-stage learning method to teach the LLM to use various tools from an easy-to-difficult curriculum; (2) thenceforth, we propose the Iterative Self-instruct from Introspective Feedback (ISIF) to dynamically construct the dataset to improve the ability to use the complicated tool. Extensive experiments conducted on both controlled and real-world settings demonstrate the superiority of our tool learning framework in the real-world application scenarios compared to both tuning-free (e.g. ChatGPT, Claude) and tuning-based baselines (e.g. GPT4Tools).

{{</citation>}}


### (37/52) Understanding the Usage of QUBO-based Hamiltonian Function in Combinatorial Optimization over Graphs: A Discussion Using Max Cut (MC) Problem (Redwan Ahmed Rizvee et al., 2023)

{{<citation>}}

Redwan Ahmed Rizvee, Md. Mosaddek Khan. (2023)  
**Understanding the Usage of QUBO-based Hamiltonian Function in Combinatorial Optimization over Graphs: A Discussion Using Max Cut (MC) Problem**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI, math-OC  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2308.13978v1)  

---


**ABSTRACT**  
Quadratic Unconstrained Binary Optimization (QUBO) is a generic technique to model various NP-hard combinatorial optimization problems in the form of binary variables. The Hamiltonian function is often used to formulate QUBO problems where it is used as the objective function in the context of optimization. In this study, we investigate how reinforcement learning-based (RL) paradigms with the presence of the Hamiltonian function can address combinatorial optimization problems over graphs in QUBO formulations. We use Graph Neural Network (GNN) as the message-passing architecture to convey the information among the nodes. We have centered our discussion on QUBO formulated Max-Cut problem but the intuitions can be extended to any QUBO supported canonical NP-Hard combinatorial optimization problems. We mainly investigate three formulations, Monty-Carlo Tree Search with GNN-based RL (MCTS-GNN), DQN with GNN-based RL, and a generic GNN with attention-based RL (GRL). Our findings state that in the RL-based paradigm, the Hamiltonian function-based optimization in QUBO formulation brings model convergence and can be used as a generic reward function. We also analyze and present the performance of our RL-based setups through experimenting over graphs of different densities and compare them with a simple GNN-based setup in the light of constraint violation, learning stability and computation cost. As per one of our findings, all the architectures provide a very comparable performance in sparse graphs as per the number of constraint violation whreas MCTS-GNN gives the best performance. In the similar criteria, the performance significantly starts to drop both for GRL and simple GNN-based setups whereas MCTS-GNN and DQN shines. We also present the corresponding mathematical formulations and in-depth discussion of the observed characteristics during experimentations.

{{</citation>}}


## cs.CL (6)



### (38/52) Empowering Cross-lingual Abilities of Instruction-tuned Large Language Models by Translation-following demonstrations (Leonardo Ranaldi et al., 2023)

{{<citation>}}

Leonardo Ranaldi, Giulia Pucci, Andre Freitas. (2023)  
**Empowering Cross-lingual Abilities of Instruction-tuned Large Language Models by Translation-following demonstrations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.14186v1)  

---


**ABSTRACT**  
The language ability of Large Language Models (LLMs) is often unbalanced towards English because of the imbalance in the distribution of the pre-training data. This disparity is demanded in further fine-tuning and affecting the cross-lingual abilities of LLMs. In this paper, we propose to empower Instructiontuned LLMs (It-LLMs) in languages other than English by building semantic alignment between them. Hence, we propose CrossAlpaca, an It-LLM with cross-lingual instruction-following and Translation-following demonstrations to improve semantic alignment between languages. We validate our approach on the multilingual Question Answering (QA) benchmarks XQUAD and MLQA and adapted versions of MMLU and BBH. Our models, tested over six different languages, outperform the It-LLMs tuned on monolingual data. The final results show that instruction tuning on non-English data is not enough and that semantic alignment can be further improved by Translation-following demonstrations.

{{</citation>}}


### (39/52) Generative AI for Business Strategy: Using Foundation Models to Create Business Strategy Tools (Son The Nguyen et al., 2023)

{{<citation>}}

Son The Nguyen, Theja Tulabandhula. (2023)  
**Generative AI for Business Strategy: Using Foundation Models to Create Business Strategy Tools**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, Generative AI, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2308.14182v1)  

---


**ABSTRACT**  
Generative models (foundation models) such as LLMs (large language models) are having a large impact on multiple fields. In this work, we propose the use of such models for business decision making. In particular, we combine unstructured textual data sources (e.g., news data) with multiple foundation models (namely, GPT4, transformer-based Named Entity Recognition (NER) models and Entailment-based Zero-shot Classifiers (ZSC)) to derive IT (information technology) artifacts in the form of a (sequence of) signed business networks. We posit that such artifacts can inform business stakeholders about the state of the market and their own positioning as well as provide quantitative insights into improving their future outlook.

{{</citation>}}


### (40/52) Towards Vision-Language Mechanistic Interpretability: A Causal Tracing Tool for BLIP (Vedant Palit et al., 2023)

{{<citation>}}

Vedant Palit, Rohan Pandey, Aryaman Arora, Paul Pu Liang. (2023)  
**Towards Vision-Language Mechanistic Interpretability: A Causal Tracing Tool for BLIP**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.14179v1)  

---


**ABSTRACT**  
Mechanistic interpretability seeks to understand the neural mechanisms that enable specific behaviors in Large Language Models (LLMs) by leveraging causality-based methods. While these approaches have identified neural circuits that copy spans of text, capture factual knowledge, and more, they remain unusable for multimodal models since adapting these tools to the vision-language domain requires considerable architectural changes. In this work, we adapt a unimodal causal tracing tool to BLIP to enable the study of the neural mechanisms underlying image-conditioned text generation. We demonstrate our approach on a visual question answering dataset, highlighting the causal relevance of later layer representations for all tokens. Furthermore, we release our BLIP causal tracing tool as open source to enable further experimentation in vision-language mechanistic interpretability by the community. Our code is available at https://github.com/vedantpalit/Towards-Vision-Language-Mechanistic-Interpretability.

{{</citation>}}


### (41/52) Examining User-Friendly and Open-Sourced Large GPT Models: A Survey on Language, Multimodal, and Scientific GPT Models (Kaiyuan Gao et al., 2023)

{{<citation>}}

Kaiyuan Gao, Sunan He, Zhenyu He, Jiacheng Lin, QiZhi Pei, Jie Shao, Wei Zhang. (2023)  
**Examining User-Friendly and Open-Sourced Large GPT Models: A Survey on Language, Multimodal, and Scientific GPT Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, NLP  
[Paper Link](http://arxiv.org/abs/2308.14149v1)  

---


**ABSTRACT**  
Generative pre-trained transformer (GPT) models have revolutionized the field of natural language processing (NLP) with remarkable performance in various tasks and also extend their power to multimodal domains. Despite their success, large GPT models like GPT-4 face inherent limitations such as considerable size, high computational requirements, complex deployment processes, and closed development loops. These constraints restrict their widespread adoption and raise concerns regarding their responsible development and usage. The need for user-friendly, relatively small, and open-sourced alternative GPT models arises from the desire to overcome these limitations while retaining high performance. In this survey paper, we provide an examination of alternative open-sourced models of large GPTs, focusing on user-friendly and relatively small models that facilitate easier deployment and accessibility. Through this extensive survey, we aim to equip researchers, practitioners, and enthusiasts with a thorough understanding of user-friendly and relatively small open-sourced models of large GPTs, their current state, challenges, and future research directions, inspiring the development of more efficient, accessible, and versatile GPT models that cater to the broader scientific community and advance the field of general artificial intelligence. The source contents are continuously updating in https://github.com/GPT-Alternatives/gpt_alternatives.

{{</citation>}}


### (42/52) Detecting Language Model Attacks with Perplexity (Gabriel Alon et al., 2023)

{{<citation>}}

Gabriel Alon, Michael Kamfonas. (2023)  
**Detecting Language Model Attacks with Perplexity**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: Language Model, Perplexity  
[Paper Link](http://arxiv.org/abs/2308.14132v1)  

---


**ABSTRACT**  
A novel hack involving Large Language Models (LLMs) has emerged, leveraging adversarial suffixes to trick models into generating perilous responses. This method has garnered considerable attention from reputable media outlets such as the New York Times and Wired, thereby influencing public perception regarding the security and safety of LLMs. In this study, we advocate the utilization of perplexity as one of the means to recognize such potential attacks. The underlying concept behind these hacks revolves around appending an unusually constructed string of text to a harmful query that would otherwise be blocked. This maneuver confuses the protective mechanisms and tricks the model into generating a forbidden response. Such scenarios could result in providing detailed instructions to a malicious user for constructing explosives or orchestrating a bank heist. Our investigation demonstrates the feasibility of employing perplexity, a prevalent natural language processing metric, to detect these adversarial tactics before generating a forbidden response. By evaluating the perplexity of queries with and without such adversarial suffixes using an open-source LLM, we discovered that nearly 90 percent were above a perplexity of 1000. This contrast underscores the efficacy of perplexity for detecting this type of exploit.

{{</citation>}}


### (43/52) MedAlign: A Clinician-Generated Dataset for Instruction Following with Electronic Medical Records (Scott L. Fleming et al., 2023)

{{<citation>}}

Scott L. Fleming, Alejandro Lozano, William J. Haberkorn, Jenelle A. Jindal, Eduardo P. Reis, Rahul Thapa, Louis Blankemeier, Julian Z. Genkins, Ethan Steinberg, Ashwin Nayak, Birju S. Patel, Chia-Chun Chiang, Alison Callahan, Zepeng Huo, Sergios Gatidis, Scott J. Adams, Oluseyi Fayanju, Shreya J. Shah, Thomas Savage, Ethan Goh, Akshay S. Chaudhari, Nima Aghaeepour, Christopher Sharp, Michael A. Pfeffer, Percy Liang, Jonathan H. Chen, Keith E. Morse, Emma P. Brunskill, Jason A. Fries, Nigam H. Shah. (2023)  
**MedAlign: A Clinician-Generated Dataset for Instruction Following with Electronic Medical Records**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.14089v1)  

---


**ABSTRACT**  
The ability of large language models (LLMs) to follow natural language instructions with human-level fluency suggests many opportunities in healthcare to reduce administrative burden and improve quality of care. However, evaluating LLMs on realistic text generation tasks for healthcare remains challenging. Existing question answering datasets for electronic health record (EHR) data fail to capture the complexity of information needs and documentation burdens experienced by clinicians. To address these challenges, we introduce MedAlign, a benchmark dataset of 983 natural language instructions for EHR data. MedAlign is curated by 15 clinicians (7 specialities), includes clinician-written reference responses for 303 instructions, and provides 276 longitudinal EHRs for grounding instruction-response pairs. We used MedAlign to evaluate 6 general domain LLMs, having clinicians rank the accuracy and quality of each LLM response. We found high error rates, ranging from 35% (GPT-4) to 68% (MPT-7B-Instruct), and an 8.3% drop in accuracy moving from 32k to 2k context lengths for GPT-4. Finally, we report correlations between clinician rankings and automated natural language generation metrics as a way to rank LLMs without human review. We make MedAlign available under a research data use agreement to enable LLM evaluations on tasks aligned with clinician needs and preferences.

{{</citation>}}


## cs.IR (3)



### (44/52) Only Encode Once: Making Content-based News Recommender Greener (Qijiong Liu et al., 2023)

{{<citation>}}

Qijiong Liu, Jieming Zhu, Quanyu Dai, Xiao-Ming Wu. (2023)  
**Only Encode Once: Making Content-based News Recommender Greener**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.14155v1)  

---


**ABSTRACT**  
Large pretrained language models (PLM) have become de facto news encoders in modern news recommender systems, due to their strong ability in comprehending textual content. These huge Transformer-based architectures, when finetuned on recommendation tasks, can greatly improve news recommendation performance. However, the PLM-based pretrain-finetune framework incurs high computational cost and energy consumption, primarily due to the extensive redundant processing of news encoding during each training epoch. In this paper, we propose the ``Only Encode Once'' framework for news recommendation (OLEO), by decoupling news representation learning from downstream recommendation task learning. The decoupled design makes content-based news recommender as green and efficient as id-based ones, leading to great reduction in computational cost and training resources. Extensive experiments show that our OLEO framework can reduce carbon emissions by up to 13 times compared with the state-of-the-art pretrain-finetune framework and maintain a competitive or even superior performance level. The source code is released for reproducibility.

{{</citation>}}


### (45/52) CTR is not Enough: a Novel Reinforcement Learning based Ranking Approach for Optimizing Session Clicks (Shaowei Liu et al., 2023)

{{<citation>}}

Shaowei Liu, Yangjun Liu. (2023)  
**CTR is not Enough: a Novel Reinforcement Learning based Ranking Approach for Optimizing Session Clicks**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14056v1)  

---


**ABSTRACT**  
Ranking is a crucial module using in the recommender system. In particular, the ranking module using in our YoungTao recommendation scenario is to provide an ordered list of items to users, to maximize the click number throughout the recommendation session for each user. However, we found that the traditional ranking method for optimizing Click-Through rate(CTR) cannot address our ranking scenario well, since it completely ignores user leaving, and CTR is the optimization goal for the one-step recommendation. To effectively undertake the purpose of our ranking module, we propose a long-term optimization goal, named as CTE (Click-Through quantity expectation), for explicitly taking the behavior of user leaving into account. Based on CTE, we propose an effective model trained by reinforcement learning. Moreover, we build a simulation environment from offline log data for estimating PBR and CTR. We conduct extensive experiments on offline datasets and an online e-commerce scenario in TaoBao. Experimental results show that our method can boost performance effectively

{{</citation>}}


### (46/52) Text Matching Improves Sequential Recommendation by Reducing Popularity Biases (Zhenghao Liu et al., 2023)

{{<citation>}}

Zhenghao Liu, Sen Mei, Chenyan Xiong, Xiaohua Li, Shi Yu, Zhiyuan Liu, Yu Gu, Ge Yu. (2023)  
**Text Matching Improves Sequential Recommendation by Reducing Popularity Biases**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.14029v1)  

---


**ABSTRACT**  
This paper proposes Text mAtching based SequenTial rEcommendation model (TASTE), which maps items and users in an embedding space and recommends items by matching their text representations. TASTE verbalizes items and user-item interactions using identifiers and attributes of items. To better characterize user behaviors, TASTE additionally proposes an attention sparsity method, which enables TASTE to model longer user-item interactions by reducing the self-attention computations during encoding. Our experiments show that TASTE outperforms the state-of-the-art methods on widely used sequential recommendation datasets. TASTE alleviates the cold start problem by representing long-tail items using full-text modeling and bringing the benefits of pretrained language models to recommendation systems. Our further analyses illustrate that TASTE significantly improves the recommendation accuracy by reducing the popularity bias of previous item id based recommendation models and returning more appropriate and text-relevant items to satisfy users. All codes are available at https://github.com/OpenMatch/TASTE.

{{</citation>}}


## eess.SY (2)



### (47/52) Reinforcement Learning-based Optimal Control and Software Rejuvenation for Safe and Efficient UAV Navigation (Angela Chen et al., 2023)

{{<citation>}}

Angela Chen, Konstantinos Mitsopoulos, Raffaele Romagnoli. (2023)  
**Reinforcement Learning-based Optimal Control and Software Rejuvenation for Safe and Efficient UAV Navigation**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14139v1)  

---


**ABSTRACT**  
Unmanned autonomous vehicles (UAVs) rely on effective path planning and tracking control to accomplish complex tasks in various domains. Reinforcement Learning (RL) methods are becoming increasingly popular in control applications, as they can learn from data and deal with unmodelled dynamics. Cyber-physical systems (CPSs), such as UAVs, integrate sensing, network communication, control, and computation to solve challenging problems. In this context, Software Rejuvenation (SR) is a protection mechanism that refreshes the control software to mitigate cyber-attacks, but it can affect the tracking controller's performance due to discrepancies between the control software and the physical system state. Traditional approaches to mitigate this effect are conservative, hindering the overall system performance. In this paper, we propose a novel approach that incorporates Deep Reinforcement Learning (Deep RL) into SR to design a safe and high-performing tracking controller. Our approach optimizes safety and performance, and we demonstrate its effectiveness during UAV simulations. We compare our approach with traditional methods and show that it improves the system's performance while maintaining safety constraints.

{{</citation>}}


### (48/52) MARL for Decentralized Electric Vehicle Charging Coordination with V2V Energy Exchange (Jiarong Fan et al., 2023)

{{<citation>}}

Jiarong Fan, Hao Wang, Ariel Liebman. (2023)  
**MARL for Decentralized Electric Vehicle Charging Coordination with V2V Energy Exchange**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-MA, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.14111v1)  

---


**ABSTRACT**  
Effective energy management of electric vehicle (EV) charging stations is critical to supporting the transport sector's sustainable energy transition. This paper addresses the EV charging coordination by considering vehicle-to-vehicle (V2V) energy exchange as the flexibility to harness in EV charging stations. Moreover, this paper takes into account EV user experiences, such as charging satisfaction and fairness. We propose a Multi-Agent Reinforcement Learning (MARL) approach to coordinate EV charging with V2V energy exchange while considering uncertainties in the EV arrival time, energy price, and solar energy generation. The exploration capability of MARL is enhanced by introducing parameter noise into MARL's neural network models. Experimental results demonstrate the superior performance and scalability of our proposed method compared to traditional optimization baselines. The decentralized execution of the algorithm enables it to effectively deal with partial system faults in the charging station.

{{</citation>}}


## eess.IV (2)



### (49/52) Bi-Modality Medical Image Synthesis Using Semi-Supervised Sequential Generative Adversarial Networks (Xin Yang et al., 2023)

{{<citation>}}

Xin Yang, Yi Lin, Zhiwei Wang, Xin Li, Kwang-Ting Cheng. (2023)  
**Bi-Modality Medical Image Synthesis Using Semi-Supervised Sequential Generative Adversarial Networks**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.14066v1)  

---


**ABSTRACT**  
In this paper, we propose a bi-modality medical image synthesis approach based on sequential generative adversarial network (GAN) and semi-supervised learning. Our approach consists of two generative modules that synthesize images of the two modalities in a sequential order. A method for measuring the synthesis complexity is proposed to automatically determine the synthesis order in our sequential GAN. Images of the modality with a lower complexity are synthesized first, and the counterparts with a higher complexity are generated later. Our sequential GAN is trained end-to-end in a semi-supervised manner. In supervised training, the joint distribution of bi-modality images are learned from real paired images of the two modalities by explicitly minimizing the reconstruction losses between the real and synthetic images. To avoid overfitting limited training images, in unsupervised training, the marginal distribution of each modality is learned based on unpaired images by minimizing the Wasserstein distance between the distributions of real and fake images. We comprehensively evaluate the proposed model using two synthesis tasks based on three types of evaluate metrics and user studies. Visual and quantitative results demonstrate the superiority of our method to the state-of-the-art methods, and reasonable visual quality and clinical significance. Code is made publicly available at https://github.com/hustlinyi/Multimodal-Medical-Image-Synthesis.

{{</citation>}}


### (50/52) High-risk Factor Prediction in Lung Cancer Using Thin CT Scans: An Attention-Enhanced Graph Convolutional Network Approach (Xiaotong Fu et al., 2023)

{{<citation>}}

Xiaotong Fu, Xiangyu Meng, Jing Zhou, Ying Ji. (2023)  
**High-risk Factor Prediction in Lung Cancer Using Thin CT Scans: An Attention-Enhanced Graph Convolutional Network Approach**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention, Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2308.14000v1)  

---


**ABSTRACT**  
Lung cancer, particularly in its advanced stages, remains a leading cause of death globally. Though early detection via low-dose computed tomography (CT) is promising, the identification of high-risk factors crucial for surgical mode selection remains a challenge. Addressing this, our study introduces an Attention-Enhanced Graph Convolutional Network (AE-GCN) model to classify whether there are high-risk factors in stage I lung cancer based on the preoperative CT images. This will aid surgeons in determining the optimal surgical method before the operation. Unlike previous studies that relied on 3D patch techniques to represent nodule spatial features, our method employs a GCN model to capture the spatial characteristics of pulmonary nodules. Specifically, we regard each slice of the nodule as a graph vertex, and the inherent spatial relationships between slices form the edges. Then, to enhance the expression of nodule features, we integrated both channel and spatial attention mechanisms with a pre-trained VGG model for adaptive feature extraction from pulmonary nodules. Lastly, the effectiveness of the proposed method is demonstrated using real-world data collected from the hospitals, thereby emphasizing its potential utility in the clinical practice.

{{</citation>}}


## cs.SD (2)



### (51/52) Anomalous Sound Detection Using Self-Attention-Based Frequency Pattern Analysis of Machine Sounds (Hejing Zhang et al., 2023)

{{<citation>}}

Hejing Zhang, Jian Guan, Qiaoxi Zhu, Feiyang Xiao, Youde Liu. (2023)  
**Anomalous Sound Detection Using Self-Attention-Based Frequency Pattern Analysis of Machine Sounds**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2308.14063v1)  

---


**ABSTRACT**  
Different machines can exhibit diverse frequency patterns in their emitted sound. This feature has been recently explored in anomaly sound detection and reached state-of-the-art performance. However, existing methods rely on the manual or empirical determination of the frequency filter by observing the effective frequency range in the training data, which may be impractical for general application. This paper proposes an anomalous sound detection method using self-attention-based frequency pattern analysis and spectral-temporal information fusion. Our experiments demonstrate that the self-attention module automatically and adaptively analyses the effective frequencies of a machine sound and enhances that information in the spectral feature representation. With spectral-temporal information fusion, the obtained audio feature eventually improves the anomaly detection performance on the DCASE 2020 Challenge Task 2 dataset.

{{</citation>}}


### (52/52) Multi-Subdomain Adversarial Network for Cross-Subject EEG-based Emotion Recognition (Guang Lin et al., 2023)

{{<citation>}}

Guang Lin, Jianhai Zhang. (2023)  
**Multi-Subdomain Adversarial Network for Cross-Subject EEG-based Emotion Recognition**  

---
Primary Category: cs.SD  
Categories: cs-HC, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2308.14059v1)  

---


**ABSTRACT**  
The individual difference between subjects is significant in EEG-based emotion recognition, resulting in the difficulty of sharing the model across subjects. Previous studies use domain adaptation algorithms to minimize the global domain discrepancy while ignoring the class information, which may cause misalignment of subdomains and reduce model performance. This paper proposes a multi-subdomain adversarial network (MSAN) for cross-subject EEG-based emotion recognition. MSAN uses adversarial training to model the discrepancy in the global domain and subdomain to reduce the intra-class distance and enlarge the inter-class distance. In addition, MSAN initializes parameters through a pre-trained autoencoder to ensure the stability and convertibility of the model. The experimental results show that the accuracy of MSAN is improved by 30.02\% on the SEED dataset comparing with the nontransfer method.

{{</citation>}}
