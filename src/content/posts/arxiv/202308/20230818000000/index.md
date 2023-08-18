---
draft: false
title: "arXiv @ 2023.08.18"
date: 2023-08-18
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.18"
    identifier: arxiv_20230818
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (20)](#cscv-20)
- [eess.SY (3)](#eesssy-3)
- [cs.SI (1)](#cssi-1)
- [cs.CL (11)](#cscl-11)
- [cs.LG (15)](#cslg-15)
- [cs.IR (4)](#csir-4)
- [stat.ML (1)](#statml-1)
- [cs.CR (1)](#cscr-1)
- [eess.IV (4)](#eessiv-4)
- [cs.CY (1)](#cscy-1)
- [cs.NI (1)](#csni-1)
- [cs.SE (1)](#csse-1)
- [cs.NE (2)](#csne-2)
- [cs.RO (1)](#csro-1)
- [cs.AR (1)](#csar-1)
- [cs.SD (2)](#cssd-2)
- [cs.HC (1)](#cshc-1)

## cs.CV (20)



### (1/70) TeCH: Text-guided Reconstruction of Lifelike Clothed Humans (Yangyi Huang et al., 2023)

{{<citation>}}

Yangyi Huang, Hongwei Yi, Yuliang Xiu, Tingting Liao, Jiaxiang Tang, Deng Cai, Justus Thies. (2023)  
**TeCH: Text-guided Reconstruction of Lifelike Clothed Humans**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-GR, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.08545v1)  

---


**ABSTRACT**  
Despite recent research advancements in reconstructing clothed humans from a single image, accurately restoring the "unseen regions" with high-level details remains an unsolved challenge that lacks attention. Existing methods often generate overly smooth back-side surfaces with a blurry texture. But how to effectively capture all visual attributes of an individual from a single image, which are sufficient to reconstruct unseen areas (e.g., the back view)? Motivated by the power of foundation models, TeCH reconstructs the 3D human by leveraging 1) descriptive text prompts (e.g., garments, colors, hairstyles) which are automatically generated via a garment parsing model and Visual Question Answering (VQA), 2) a personalized fine-tuned Text-to-Image diffusion model (T2I) which learns the "indescribable" appearance. To represent high-resolution 3D clothed humans at an affordable cost, we propose a hybrid 3D representation based on DMTet, which consists of an explicit body shape grid and an implicit distance field. Guided by the descriptive prompts + personalized T2I diffusion model, the geometry and texture of the 3D humans are optimized through multi-view Score Distillation Sampling (SDS) and reconstruction losses based on the original observation. TeCH produces high-fidelity 3D clothed humans with consistent & delicate texture, and detailed full-body geometry. Quantitative and qualitative experiments demonstrate that TeCH outperforms the state-of-the-art methods in terms of reconstruction accuracy and rendering quality. The code will be publicly available for research purposes at https://huangyangyi.github.io/tech

{{</citation>}}


### (2/70) Painter: Teaching Auto-regressive Language Models to Draw Sketches (Reza Pourreza et al., 2023)

{{<citation>}}

Reza Pourreza, Apratim Bhattacharyya, Sunny Panchal, Mingu Lee, Pulkit Madan, Roland Memisevic. (2023)  
**Painter: Teaching Auto-regressive Language Models to Draw Sketches**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Language Model, Sketch  
[Paper Link](http://arxiv.org/abs/2308.08520v1)  

---


**ABSTRACT**  
Large language models (LLMs) have made tremendous progress in natural language understanding and they have also been successfully adopted in other domains such as computer vision, robotics, reinforcement learning, etc. In this work, we apply LLMs to image generation tasks by directly generating the virtual brush strokes to paint an image. We present Painter, an LLM that can convert user prompts in text description format to sketches by generating the corresponding brush strokes in an auto-regressive way. We construct Painter based on off-the-shelf LLM that is pre-trained on a large text corpus, by fine-tuning it on the new task while preserving language understanding capabilities. We create a dataset of diverse multi-object sketches paired with textual prompts that covers several object types and tasks. Painter can generate sketches from text descriptions, remove objects from canvas, and detect and classify objects in sketches. Although this is an unprecedented pioneering work in using LLMs for auto-regressive image generation, the results are very encouraging.

{{</citation>}}


### (3/70) Exploiting Point-Wise Attention in 6D Object Pose Estimation Based on Bidirectional Prediction (Yuhao Yang et al., 2023)

{{<citation>}}

Yuhao Yang, Jun Wu, Guangjian Zhang, Rong Xiong. (2023)  
**Exploiting Point-Wise Attention in 6D Object Pose Estimation Based on Bidirectional Prediction**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.08518v1)  

---


**ABSTRACT**  
Traditional geometric registration based estimation methods only exploit the CAD model implicitly, which leads to their dependence on observation quality and deficiency to occlusion.To address the problem,the paper proposes a bidirectional correspondence prediction network with a point-wise attention-aware mechanism. This network not only requires the model points to predict the correspondence but also explicitly models the geometric similarities between observations and the model prior.} Our key insight is that the correlations between each model point and scene point provide essential information for learning point-pair matches. To further tackle the correlation noises brought by feature distribution divergence, we design a simple but effective pseudo-siamese network to improve feature homogeneity.Experimental results on the public datasets of LineMOD, YCB-Video, and Occ-LineMOD show that the proposed method achieves better performance than other state-of-the-art methods under the same evaluation criteria. Its robustness in estimating poses is greatly improved, especially in an environment with severe occlusions.

{{</citation>}}


### (4/70) Self-Supervised Online Camera Calibration for Automated Driving and Parking Applications (Ciarán Hogan et al., 2023)

{{<citation>}}

Ciarán Hogan, Ganesh Sistu, Ciarán Eising. (2023)  
**Self-Supervised Online Camera Calibration for Automated Driving and Parking Applications**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.08495v1)  

---


**ABSTRACT**  
Camera-based perception systems play a central role in modern autonomous vehicles. These camera based perception algorithms require an accurate calibration to map the real world distances to image pixels. In practice, calibration is a laborious procedure requiring specialised data collection and careful tuning. This process must be repeated whenever the parameters of the camera change, which can be a frequent occurrence in autonomous vehicles. Hence there is a need to calibrate at regular intervals to ensure the camera is accurate. Proposed is a deep learning framework to learn intrinsic and extrinsic calibration of the camera in real time. The framework is self-supervised and doesn't require any labelling or supervision to learn the calibration parameters. The framework learns calibration without the need for any physical targets or to drive the car on special planar surfaces.

{{</citation>}}


### (5/70) Classification Committee for Active Deep Object Detection (Lei Zhao et al., 2023)

{{<citation>}}

Lei Zhao, Bo Li, Xingxing Wei. (2023)  
**Classification Committee for Active Deep Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.08476v1)  

---


**ABSTRACT**  
In object detection, the cost of labeling is much high because it needs not only to confirm the categories of multiple objects in an image but also to accurately determine the bounding boxes of each object. Thus, integrating active learning into object detection will raise pretty positive significance. In this paper, we propose a classification committee for active deep object detection method by introducing a discrepancy mechanism of multiple classifiers for samples' selection when training object detectors. The model contains a main detector and a classification committee. The main detector denotes the target object detector trained from a labeled pool composed of the selected informative images. The role of the classification committee is to select the most informative images according to their uncertainty values from the view of classification, which is expected to focus more on the discrepancy and representative of instances. Specifically, they compute the uncertainty for a specified instance within the image by measuring its discrepancy output by the committee pre-trained via the proposed Maximum Classifiers Discrepancy Group Loss (MCDGL). The most informative images are finally determined by selecting the ones with many high-uncertainty instances. Besides, to mitigate the impact of interference instances, we design a Focus on Positive Instances Loss (FPIL) to make the committee the ability to automatically focus on the representative instances as well as precisely encode their discrepancies for the same instance. Experiments are conducted on Pascal VOC and COCO datasets versus some popular object detectors. And results show that our method outperforms the state-of-the-art active learning methods, which verifies the effectiveness of the proposed method.

{{</citation>}}


### (6/70) Integrating Visual and Semantic Similarity Using Hierarchies for Image Retrieval (Aishwarya Venkataramanan et al., 2023)

{{<citation>}}

Aishwarya Venkataramanan, Martin Laviale, Cédric Pradalier. (2023)  
**Integrating Visual and Semantic Similarity Using Hierarchies for Image Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Similarity  
[Paper Link](http://arxiv.org/abs/2308.08431v1)  

---


**ABSTRACT**  
Most of the research in content-based image retrieval (CBIR) focus on developing robust feature representations that can effectively retrieve instances from a database of images that are visually similar to a query. However, the retrieved images sometimes contain results that are not semantically related to the query. To address this, we propose a method for CBIR that captures both visual and semantic similarity using a visual hierarchy. The hierarchy is constructed by merging classes with overlapping features in the latent space of a deep neural network trained for classification, assuming that overlapping classes share high visual and semantic similarities. Finally, the constructed hierarchy is integrated into the distance calculation metric for similarity search. Experiments on standard datasets: CUB-200-2011 and CIFAR100, and a real-life use case using diatom microscopy images show that our method achieves superior performance compared to the existing methods on image retrieval.

{{</citation>}}


### (7/70) Tem-adapter: Adapting Image-Text Pretraining for Video Question Answer (Guangyi Chen et al., 2023)

{{<citation>}}

Guangyi Chen, Xiao Liu, Guangrun Wang, Kun Zhang, Philip H. S. Torr, Xiao-Ping Zhang, Yansong Tang. (2023)  
**Tem-adapter: Adapting Image-Text Pretraining for Video Question Answer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Transformer  
[Paper Link](http://arxiv.org/abs/2308.08414v1)  

---


**ABSTRACT**  
Video-language pre-trained models have shown remarkable success in guiding video question-answering (VideoQA) tasks. However, due to the length of video sequences, training large-scale video-based models incurs considerably higher costs than training image-based ones. This motivates us to leverage the knowledge from image-based pretraining, despite the obvious gaps between image and video domains. To bridge these gaps, in this paper, we propose Tem-Adapter, which enables the learning of temporal dynamics and complex semantics by a visual Temporal Aligner and a textual Semantic Aligner. Unlike conventional pretrained knowledge adaptation methods that only concentrate on the downstream task objective, the Temporal Aligner introduces an extra language-guided autoregressive task aimed at facilitating the learning of temporal dependencies, with the objective of predicting future states based on historical clues and language guidance that describes event progression. Besides, to reduce the semantic gap and adapt the textual representation for better event description, we introduce a Semantic Aligner that first designs a template to fuse question and answer pairs as event descriptions and then learns a Transformer decoder with the whole video sequence as guidance for refinement. We evaluate Tem-Adapter and different pre-train transferring methods on two VideoQA benchmarks, and the significant performance improvement demonstrates the effectiveness of our method.

{{</citation>}}


### (8/70) Agglomerative Transformer for Human-Object Interaction Detection (Danyang Tu et al., 2023)

{{<citation>}}

Danyang Tu, Wei Sun, Guangtao Zhai, Wei Shen. (2023)  
**Agglomerative Transformer for Human-Object Interaction Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.08370v1)  

---


**ABSTRACT**  
We propose an agglomerative Transformer (AGER) that enables Transformer-based human-object interaction (HOI) detectors to flexibly exploit extra instance-level cues in a single-stage and end-to-end manner for the first time. AGER acquires instance tokens by dynamically clustering patch tokens and aligning cluster centers to instances with textual guidance, thus enjoying two benefits: 1) Integrality: each instance token is encouraged to contain all discriminative feature regions of an instance, which demonstrates a significant improvement in the extraction of different instance-level cues and subsequently leads to a new state-of-the-art performance of HOI detection with 36.75 mAP on HICO-Det. 2) Efficiency: the dynamical clustering mechanism allows AGER to generate instance tokens jointly with the feature learning of the Transformer encoder, eliminating the need of an additional object detector or instance decoder in prior methods, thus allowing the extraction of desirable extra cues for HOI detection in a single-stage and end-to-end pipeline. Concretely, AGER reduces GFLOPs by 8.5% and improves FPS by 36%, even compared to a vanilla DETR-like pipeline without extra cue extraction.

{{</citation>}}


### (9/70) KernelWarehouse: Towards Parameter-Efficient Dynamic Convolution (Chao Li et al., 2023)

{{<citation>}}

Chao Li, Anbang Yao. (2023)  
**KernelWarehouse: Towards Parameter-Efficient Dynamic Convolution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.08361v1)  

---


**ABSTRACT**  
Dynamic convolution learns a linear mixture of $n$ static kernels weighted with their sample-dependent attentions, demonstrating superior performance compared to normal convolution. However, existing designs are parameter-inefficient: they increase the number of convolutional parameters by $n$ times. This and the optimization difficulty lead to no research progress in dynamic convolution that can allow us to use a significant large value of $n$ (e.g., $n>100$ instead of typical setting $n<10$) to push forward the performance boundary. In this paper, we propose $KernelWarehouse$, a more general form of dynamic convolution, which can strike a favorable trade-off between parameter efficiency and representation power. Its key idea is to redefine the basic concepts of "$kernels$" and "$assembling$ $kernels$" in dynamic convolution from the perspective of reducing kernel dimension and increasing kernel number significantly. In principle, KernelWarehouse enhances convolutional parameter dependencies within the same layer and across successive layers via tactful kernel partition and warehouse sharing, yielding a high degree of freedom to fit a desired parameter budget. We validate our method on ImageNet and MS-COCO datasets with different ConvNet architectures, and show that it attains state-of-the-art results. For instance, the ResNet18|ResNet50|MobileNetV2|ConvNeXt-Tiny model trained with KernelWarehouse on ImageNet reaches 76.05%|81.05%|75.52%|82.51% top-1 accuracy. Thanks to its flexible design, KernelWarehouse can even reduce the model size of a ConvNet while improving the accuracy, e.g., our ResNet18 model with 36.45%|65.10% parameter reduction to the baseline shows 2.89%|2.29% absolute improvement to top-1 accuracy.

{{</citation>}}


### (10/70) Improving Depth Gradient Continuity in Transformers: A Comparative Study on Monocular Depth Estimation with CNN (Jiawei Yao et al., 2023)

{{<citation>}}

Jiawei Yao, Tong Wu, Xiaofeng Zhang. (2023)  
**Improving Depth Gradient Continuity in Transformers: A Comparative Study on Monocular Depth Estimation with CNN**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.08333v1)  

---


**ABSTRACT**  
Monocular depth estimation is an ongoing challenge in computer vision. Recent progress with Transformer models has demonstrated notable advantages over conventional CNNs in this area. However, there's still a gap in understanding how these models prioritize different regions in 2D images and how these regions affect depth estimation performance. To explore the differences between Transformers and CNNs, we employ a sparse pixel approach to contrastively analyze the distinctions between the two. Our findings suggest that while Transformers excel in handling global context and intricate textures, they lag behind CNNs in preserving depth gradient continuity. To further enhance the performance of Transformer models in monocular depth estimation, we propose the Depth Gradient Refinement (DGR) module that refines depth estimation through high-order differentiation, feature fusion, and recalibration. Additionally, we leverage optimal transport theory, treating depth maps as spatial probability distributions, and employ the optimal transport distance as a loss function to optimize our model. Experimental results demonstrate that models integrated with the plug-and-play Depth Gradient Refinement (DGR) module and the proposed loss function enhance performance without increasing complexity and computational costs. This research not only offers fresh insights into the distinctions between Transformers and CNNs in depth estimation but also paves the way for novel depth estimation methodologies.

{{</citation>}}


### (11/70) Visually-Aware Context Modeling for News Image Captioning (Tingyu Qu et al., 2023)

{{<citation>}}

Tingyu Qu, Tinne Tuytelaars, Marie-Francine Moens. (2023)  
**Visually-Aware Context Modeling for News Image Captioning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Captioning  
[Paper Link](http://arxiv.org/abs/2308.08325v1)  

---


**ABSTRACT**  
The goal of News Image Captioning is to generate an image caption according to the content of both a news article and an image. To leverage the visual information effectively, it is important to exploit the connection between the context in the articles/captions and the images. Psychological studies indicate that human faces in images draw higher attention priorities. On top of that, humans often play a central role in news stories, as also proven by the face-name co-occurrence pattern we discover in existing News Image Captioning datasets. Therefore, we design a face-naming module for faces in images and names in captions/articles to learn a better name embedding. Apart from names, which can be directly linked to an image area (faces), news image captions mostly contain context information that can only be found in the article. Humans typically address this by searching for relevant information from the article based on the image. To emulate this thought process, we design a retrieval strategy using CLIP to retrieve sentences that are semantically close to the image. We conduct extensive experiments to demonstrate the efficacy of our framework. Without using additional paired data, we establish the new state-of-the-art performance on two News Image Captioning datasets, exceeding the previous state-of-the-art by 5 CIDEr points. We will release code upon acceptance.

{{</citation>}}


### (12/70) Leveraging Next-Active Objects for Context-Aware Anticipation in Egocentric Videos (Sanket Thakur et al., 2023)

{{<citation>}}

Sanket Thakur, Cigdem Beyan, Pietro Morerio, Vittorio Murino, Alessio Del Bue. (2023)  
**Leveraging Next-Active Objects for Context-Aware Anticipation in Egocentric Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.08303v1)  

---


**ABSTRACT**  
Objects are crucial for understanding human-object interactions. By identifying the relevant objects, one can also predict potential future interactions or actions that may occur with these objects. In this paper, we study the problem of Short-Term Object interaction anticipation (STA) and propose NAOGAT (Next-Active-Object Guided Anticipation Transformer), a multi-modal end-to-end transformer network, that attends to objects in observed frames in order to anticipate the next-active-object (NAO) and, eventually, to guide the model to predict context-aware future actions. The task is challenging since it requires anticipating future action along with the object with which the action occurs and the time after which the interaction will begin, a.k.a. the time to contact (TTC). Compared to existing video modeling architectures for action anticipation, NAOGAT captures the relationship between objects and the global scene context in order to predict detections for the next active object and anticipate relevant future actions given these detections, leveraging the objects' dynamics to improve accuracy. One of the key strengths of our approach, in fact, is its ability to exploit the motion dynamics of objects within a given clip, which is often ignored by other models, and separately decoding the object-centric and motion-centric information. Through our experiments, we show that our model outperforms existing methods on two separate datasets, Ego4D and EpicKitchens-100 ("Unseen Set"), as measured by several additional metrics, such as time to contact, and next-active-object localization. The code will be available upon acceptance.

{{</citation>}}


### (13/70) Computer vision-enriched discrete choice models, with an application to residential location choice (Sander van Cranenburgh et al., 2023)

{{<citation>}}

Sander van Cranenburgh, Francisco Garrido-Valenzuela. (2023)  
**Computer vision-enriched discrete choice models, with an application to residential location choice**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, econ-EM  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2308.08276v1)  

---


**ABSTRACT**  
Visual imagery is indispensable to many multi-attribute decision situations. Examples of such decision situations in travel behaviour research include residential location choices, vehicle choices, tourist destination choices, and various safety-related choices. However, current discrete choice models cannot handle image data and thus cannot incorporate information embedded in images into their representations of choice behaviour. This gap between discrete choice models' capabilities and the real-world behaviour it seeks to model leads to incomplete and, possibly, misleading outcomes. To solve this gap, this study proposes "Computer Vision-enriched Discrete Choice Models" (CV-DCMs). CV-DCMs can handle choice tasks involving numeric attributes and images by integrating computer vision and traditional discrete choice models. Moreover, because CV-DCMs are grounded in random utility maximisation principles, they maintain the solid behavioural foundation of traditional discrete choice models. We demonstrate the proposed CV-DCM by applying it to data obtained through a novel stated choice experiment involving residential location choices. In this experiment, respondents faced choice tasks with trade-offs between commute time, monthly housing cost and street-level conditions, presented using images. As such, this research contributes to the growing body of literature in the travel behaviour field that seeks to integrate discrete choice modelling and machine learning.

{{</citation>}}


### (14/70) Contrastive Learning for Lane Detection via cross-similarity (Ali Zoljodi et al., 2023)

{{<citation>}}

Ali Zoljodi, Sadegh Abadijou, Mina Alibeigi, Masoud Daneshtalab. (2023)  
**Contrastive Learning for Lane Detection via cross-similarity**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.08242v1)  

---


**ABSTRACT**  
Detecting road lanes is challenging due to intricate markings vulnerable to unfavorable conditions. Lane markings have strong shape priors, but their visibility is easily compromised. Factors like lighting, weather, vehicles, pedestrians, and aging colors challenge the detection. A large amount of data is required to train a lane detection approach that can withstand natural variations caused by low visibility. This is because there are numerous lane shapes and natural variations that exist. Our solution, Contrastive Learning for Lane Detection via cross-similarity (CLLD), is a self-supervised learning method that tackles this challenge by enhancing lane detection models resilience to real-world conditions that cause lane low visibility. CLLD is a novel multitask contrastive learning that trains lane detection approaches to detect lane markings even in low visible situations by integrating local feature contrastive learning (CL) with our new proposed operation cross-similarity. Local feature CL focuses on extracting features for small image parts, which is necessary to localize lane segments, while cross-similarity captures global features to detect obscured lane segments using their surrounding. We enhance cross-similarity by randomly masking parts of input images for augmentation. Evaluated on benchmark datasets, CLLD outperforms state-of-the-art contrastive learning, especially in visibility-impairing conditions like shadows. Compared to supervised learning, CLLD excels in scenarios like shadows and crowded scenes.

{{</citation>}}


### (15/70) Low-Light Image Enhancement with Illumination-Aware Gamma Correction and Complete Image Modelling Network (Yinglong Wang et al., 2023)

{{<citation>}}

Yinglong Wang, Zhen Liu, Jianzhuang Liu, Songcen Xu, Shuaicheng Liu. (2023)  
**Low-Light Image Enhancement with Illumination-Aware Gamma Correction and Complete Image Modelling Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.08220v1)  

---


**ABSTRACT**  
This paper presents a novel network structure with illumination-aware gamma correction and complete image modelling to solve the low-light image enhancement problem. Low-light environments usually lead to less informative large-scale dark areas, directly learning deep representations from low-light images is insensitive to recovering normal illumination. We propose to integrate the effectiveness of gamma correction with the strong modelling capacities of deep networks, which enables the correction factor gamma to be learned in a coarse to elaborate manner via adaptively perceiving the deviated illumination. Because exponential operation introduces high computational complexity, we propose to use Taylor Series to approximate gamma correction, accelerating the training and inference speed. Dark areas usually occupy large scales in low-light images, common local modelling structures, e.g., CNN, SwinIR, are thus insufficient to recover accurate illumination across whole low-light images. We propose a novel Transformer block to completely simulate the dependencies of all pixels across images via a local-to-global hierarchical attention mechanism, so that dark areas could be inferred by borrowing the information from far informative regions in a highly effective manner. Extensive experiments on several benchmark datasets demonstrate that our approach outperforms state-of-the-art methods.

{{</citation>}}


### (16/70) MEDOE: A Multi-Expert Decoder and Output Ensemble Framework for Long-tailed Semantic Segmentation (Junao Shen et al., 2023)

{{<citation>}}

Junao Shen, Long Chen, Kun Kuang, Fei Wu, Tian Feng, Wei Zhang. (2023)  
**MEDOE: A Multi-Expert Decoder and Output Ensemble Framework for Long-tailed Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.08213v1)  

---


**ABSTRACT**  
Long-tailed distribution of semantic categories, which has been often ignored in conventional methods, causes unsatisfactory performance in semantic segmentation on tail categories. In this paper, we focus on the problem of long-tailed semantic segmentation. Although some long-tailed recognition methods (e.g., re-sampling/re-weighting) have been proposed in other problems, they can probably compromise crucial contextual information and are thus hardly adaptable to the problem of long-tailed semantic segmentation. To address this issue, we propose MEDOE, a novel framework for long-tailed semantic segmentation via contextual information ensemble-and-grouping. The proposed two-sage framework comprises a multi-expert decoder (MED) and a multi-expert output ensemble (MOE). Specifically, the MED includes several "experts". Based on the pixel frequency distribution, each expert takes the dataset masked according to the specific categories as input and generates contextual information self-adaptively for classification; The MOE adopts learnable decision weights for the ensemble of the experts' outputs. As a model-agnostic framework, our MEDOE can be flexibly and efficiently coupled with various popular deep neural networks (e.g., DeepLabv3+, OCRNet, and PSPNet) to improve their performance in long-tailed semantic segmentation. Experimental results show that the proposed framework outperforms the current methods on both Cityscapes and ADE20K datasets by up to 1.78% in mIoU and 5.89% in mAcc.

{{</citation>}}


### (17/70) S2R: Exploring a Double-Win Transformer-Based Framework for Ideal and Blind Super-Resolution (Minghao She et al., 2023)

{{<citation>}}

Minghao She, Wendong Mao, Huihong Shi, Zhongfeng Wang. (2023)  
**S2R: Exploring a Double-Win Transformer-Based Framework for Ideal and Blind Super-Resolution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.08142v1)  

---


**ABSTRACT**  
Nowadays, deep learning based methods have demonstrated impressive performance on ideal super-resolution (SR) datasets, but most of these methods incur dramatically performance drops when directly applied in real-world SR reconstruction tasks with unpredictable blur kernels. To tackle this issue, blind SR methods are proposed to improve the visual results on random blur kernels, which causes unsatisfactory reconstruction effects on ideal low-resolution images similarly. In this paper, we propose a double-win framework for ideal and blind SR task, named S2R, including a light-weight transformer-based SR model (S2R transformer) and a novel coarse-to-fine training strategy, which can achieve excellent visual results on both ideal and random fuzzy conditions. On algorithm level, S2R transformer smartly combines some efficient and light-weight blocks to enhance the representation ability of extracted features with relatively low number of parameters. For training strategy, a coarse-level learning process is firstly performed to improve the generalization of the network with the help of a large-scale external dataset, and then, a fast fine-tune process is developed to transfer the pre-trained model to real-world SR tasks by mining the internal features of the image. Experimental results show that the proposed S2R outperforms other single-image SR models in ideal SR condition with only 578K parameters. Meanwhile, it can achieve better visual results than regular blind SR models in blind fuzzy conditions with only 10 gradient updates, which improve convergence speed by 300 times, significantly accelerating the transfer-learning process in real-world situations.

{{</citation>}}


### (18/70) GPA-3D: Geometry-aware Prototype Alignment for Unsupervised Domain Adaptive 3D Object Detection from Point Clouds (Ziyu Li et al., 2023)

{{<citation>}}

Ziyu Li, Jingming Guo, Tongtong Cao, Liu Bingbing, Wankou Yang. (2023)  
**GPA-3D: Geometry-aware Prototype Alignment for Unsupervised Domain Adaptive 3D Object Detection from Point Clouds**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.08140v1)  

---


**ABSTRACT**  
LiDAR-based 3D detection has made great progress in recent years. However, the performance of 3D detectors is considerably limited when deployed in unseen environments, owing to the severe domain gap problem. Existing domain adaptive 3D detection methods do not adequately consider the problem of the distributional discrepancy in feature space, thereby hindering generalization of detectors across domains. In this work, we propose a novel unsupervised domain adaptive \textbf{3D} detection framework, namely \textbf{G}eometry-aware \textbf{P}rototype \textbf{A}lignment (\textbf{GPA-3D}), which explicitly leverages the intrinsic geometric relationship from point cloud objects to reduce the feature discrepancy, thus facilitating cross-domain transferring. Specifically, GPA-3D assigns a series of tailored and learnable prototypes to point cloud objects with distinct geometric structures. Each prototype aligns BEV (bird's-eye-view) features derived from corresponding point cloud objects on source and target domains, reducing the distributional discrepancy and achieving better adaptation. The evaluation results obtained on various benchmarks, including Waymo, nuScenes and KITTI, demonstrate the superiority of our GPA-3D over the state-of-the-art approaches for different adaptation scenarios. The MindSpore version code will be publicly available at \url{https://github.com/Liz66666/GPA3D}.

{{</citation>}}


### (19/70) SYENet: A Simple Yet Effective Network for Multiple Low-Level Vision Tasks with Real-time Performance on Mobile Device (Weiran Gou et al., 2023)

{{<citation>}}

Weiran Gou, Ziyao Yi, Yan Xiang, Shaoqing Li, Zibin Liu, Dehui Kong, Ke Xu. (2023)  
**SYENet: A Simple Yet Effective Network for Multiple Low-Level Vision Tasks with Real-time Performance on Mobile Device**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.08137v1)  

---


**ABSTRACT**  
With the rapid development of AI hardware accelerators, applying deep learning-based algorithms to solve various low-level vision tasks on mobile devices has gradually become possible. However, two main problems still need to be solved: task-specific algorithms make it difficult to integrate them into a single neural network architecture, and large amounts of parameters make it difficult to achieve real-time inference. To tackle these problems, we propose a novel network, SYENet, with only $~$6K parameters, to handle multiple low-level vision tasks on mobile devices in a real-time manner. The SYENet consists of two asymmetrical branches with simple building blocks. To effectively connect the results by asymmetrical branches, a Quadratic Connection Unit(QCU) is proposed. Furthermore, to improve performance, a new Outlier-Aware Loss is proposed to process the image. The proposed method proves its superior performance with the best PSNR as compared with other networks in real-time applications such as Image Signal Processing(ISP), Low-Light Enhancement(LLE), and Super-Resolution(SR) with 2K60FPS throughput on Qualcomm 8 Gen 1 mobile SoC(System-on-Chip). Particularly, for ISP task, SYENet got the highest score in MAI 2022 Learned Smartphone ISP challenge.

{{</citation>}}


### (20/70) Pro-Cap: Leveraging a Frozen Vision-Language Model for Hateful Meme Detection (Rui Cao et al., 2023)

{{<citation>}}

Rui Cao, Ming Shan Hee, Adriel Kuek, Wen-Haw Chong, Roy Ka-Wei Lee, Jing Jiang. (2023)  
**Pro-Cap: Leveraging a Frozen Vision-Language Model for Hateful Meme Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-IR, cs-MM, cs.CV  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2308.08088v1)  

---


**ABSTRACT**  
Hateful meme detection is a challenging multimodal task that requires comprehension of both vision and language, as well as cross-modal interactions. Recent studies have tried to fine-tune pre-trained vision-language models (PVLMs) for this task. However, with increasing model sizes, it becomes important to leverage powerful PVLMs more efficiently, rather than simply fine-tuning them. Recently, researchers have attempted to convert meme images into textual captions and prompt language models for predictions. This approach has shown good performance but suffers from non-informative image captions. Considering the two factors mentioned above, we propose a probing-based captioning approach to leverage PVLMs in a zero-shot visual question answering (VQA) manner. Specifically, we prompt a frozen PVLM by asking hateful content-related questions and use the answers as image captions (which we call Pro-Cap), so that the captions contain information critical for hateful content detection. The good performance of models with Pro-Cap on three benchmarks validates the effectiveness and generalization of the proposed method.

{{</citation>}}


## eess.SY (3)



### (21/70) Can Transformers Learn Optimal Filtering for Unknown Systems? (Haldun Balim et al., 2023)

{{<citation>}}

Haldun Balim, Zhe Du, Samet Oymak, Necmiye Ozay. (2023)  
**Can Transformers Learn Optimal Filtering for Unknown Systems?**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.08536v1)  

---


**ABSTRACT**  
Transformers have demonstrated remarkable success in natural language processing; however, their potential remains mostly unexplored for problems arising in dynamical systems. In this work, we investigate the optimal output estimation problem using transformers, which generate output predictions using all the past ones. We train the transformer using various systems drawn from a prior distribution and then evaluate its performance on previously unseen systems from the same distribution. As a result, the obtained transformer acts like a prediction algorithm that learns in-context and quickly adapts to and predicts well for different systems - thus we call it meta-output-predictor (MOP). MOP matches the performance of the optimal output estimator, based on Kalman filter, for most linear dynamical systems even though it does not have access to a model. We observe via extensive numerical experiments that MOP also performs well in challenging scenarios with non-i.i.d. noise, time-varying dynamics, and nonlinear dynamics like a quadrotor system with unknown parameters. To further support this observation, in the second part of the paper, we provide statistical guarantees on the performance of MOP and quantify the required amount of training to achieve a desired excess risk during test-time. Finally, we point out some limitations of MOP by identifying two classes of problems MOP fails to perform well, highlighting the need for caution when using transformers for control and estimation.

{{</citation>}}


### (22/70) A Robust Integrated Multi-Strategy Bus Control System via Deep Reinforcement Learning (Qinghui Nie et al., 2023)

{{<citation>}}

Qinghui Nie, Jishun Ou, Haiyang Zhang, Jiawei Lu, Shen Li, Haotian Shi. (2023)  
**A Robust Integrated Multi-Strategy Bus Control System via Deep Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.08179v1)  

---


**ABSTRACT**  
An efficient urban bus control system has the potential to significantly reduce travel delays and streamline the allocation of transportation resources, thereby offering enhanced and user-friendly transit services to passengers. However, bus operation efficiency can be impacted by bus bunching. This problem is notably exacerbated when the bus system operates along a signalized corridor with unpredictable travel demand. To mitigate this challenge, we introduce a multi-strategy fusion approach for the longitudinal control of connected and automated buses. The approach is driven by a physics-informed deep reinforcement learning (DRL) algorithm and takes into account a variety of traffic conditions along urban signalized corridors. Taking advantage of connected and autonomous vehicle (CAV) technology, the proposed approach can leverage real-time information regarding bus operating conditions and road traffic environment. By integrating the aforementioned information into the DRL-based bus control framework, our designed physics-informed DRL state fusion approach and reward function efficiently embed prior physics and leverage the merits of equilibrium and consensus concepts from control theory. This integration enables the framework to learn and adapt multiple control strategies to effectively manage complex traffic conditions and fluctuating passenger demands. Three control variables, i.e., dwell time at stops, speed between stations, and signal priority, are formulated to minimize travel duration and ensure bus stability with the aim of avoiding bus bunching. We present simulation results to validate the effectiveness of the proposed approach, underlining its superior performance when subjected to sensitivity analysis, specifically considering factors such as traffic volume, desired speed, and traffic signal conditions.

{{</citation>}}


### (23/70) Real-Time Numerical Differentiation of Sampled Data Using Adaptive Input and State Estimation (Shashank Verma et al., 2023)

{{<citation>}}

Shashank Verma, Sneha Sanjeevini, E. Dogan Sumer, Dennis S. Bernstein. (2023)  
**Real-Time Numerical Differentiation of Sampled Data Using Adaptive Input and State Estimation**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SP, eess-SY, eess.SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.08074v1)  

---


**ABSTRACT**  
Real-time numerical differentiation plays a crucial role in many digital control algorithms, such as PID control, which requires numerical differentiation to implement derivative action. This paper addresses the problem of numerical differentiation for real-time implementation with minimal prior information about the signal and noise using adaptive input and state estimation. Adaptive input estimation with adaptive state estimation (AIE/ASE) is based on retrospective cost input estimation, while adaptive state estimation is based on an adaptive Kalman filter in which the input-estimation error covariance and the measurement-noise covariance are updated online. The accuracy of AIE/ASE is compared numerically to several conventional numerical differentiation methods. Finally, AIE/ASE is applied to simulated vehicle position data generated from CarSim.

{{</citation>}}


## cs.SI (1)



### (24/70) Patterns and Pathways: Applying Social Network Analysis to Understand User Behavior in the Tourism Industry Websites (Mehrdad Maghsoudi et al., 2023)

{{<citation>}}

Mehrdad Maghsoudi, Saeid Aliakbar, AmirMahdi Mohammadi. (2023)  
**Patterns and Pathways: Applying Social Network Analysis to Understand User Behavior in the Tourism Industry Websites**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2308.08527v1)  

---


**ABSTRACT**  
The contemporary tourism landscape is undergoing rapid digitization, necessitating a nuanced comprehension of online user behavior to guide data-driven decision-making. This research bridges an existing gap by investigating the tourism website ecosystem through social network analysis. It focuses specifically on inter-website communication patterns based on user navigation. Data mining facilitates the identification of 162 core Iranian tourism websites, which are visualized as an interconnected network with websites as nodes and user transitions as weighted directed edges. By implementing community detection, eight key clusters are discerned, encompassing domains like ticket/tour bookings, accommodations, location services, and cuisine. Further analysis of inter-community relationships reveals website groupings frequently accessed together by users, highlighting complementary services sought during travel planning. The research derives invaluable insights into user preferences and information propagation within the tourism ecosystem. The methodology and findings contribute original perspectives to academia while offering pragmatic strategic recommendations to industry stakeholders like service providers, investors, and policymakers. This pioneering exploration of latent user behavior patterns advances comprehension of the evolving digital tourism landscape in Iran. It contributes pathways toward a sustainable future vision of the ecosystem, guiding stakeholders in targeted decision-making based on empirical evidence derived from social network analysis of websites and consumption patterns. The innovative methodology expands the toolkit for data-driven tourism research within academia.

{{</citation>}}


## cs.CL (11)



### (25/70) Time Travel in LLMs: Tracing Data Contamination in Large Language Models (Shahriar Golchin et al., 2023)

{{<citation>}}

Shahriar Golchin, Mihai Surdeanu. (2023)  
**Time Travel in LLMs: Tracing Data Contamination in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: BLEU, GPT, GPT-4, Language Model, NLI  
[Paper Link](http://arxiv.org/abs/2308.08493v1)  

---


**ABSTRACT**  
Data contamination, i.e., the presence of test data from downstream tasks in the training data of large language models (LLMs), is a potential major issue in understanding LLMs' effectiveness on other tasks. We propose a straightforward yet effective method for identifying data contamination within LLMs. At its core, our approach starts by identifying potential contamination in individual instances that are drawn from a small random sample; using this information, our approach then assesses if an entire dataset partition is contaminated. To estimate contamination of individual instances, we employ "guided instruction:" a prompt consisting of the dataset name, partition type, and the initial segment of a reference instance, asking the LLM to complete it. An instance is flagged as contaminated if the LLM's output either exactly or closely matches the latter segment of the reference. To understand if an entire partition is contaminated, we propose two ideas. The first idea marks a dataset partition as contaminated if the average overlap score with the reference instances (as measured by ROUGE or BLEURT) is statistically significantly better with the guided instruction vs. a general instruction that does not include the dataset and partition name. The second idea marks a dataset as contaminated if a classifier based on GPT-4 with in-context learning prompting marks multiple instances as contaminated. Our best method achieves an accuracy between 92% and 100% in detecting if an LLM is contaminated with seven datasets, containing train and test/validation partitions, when contrasted with manual evaluation by human expert. Further, our findings indicate that GPT-4 is contaminated with AG News, WNLI, and XSum datasets.

{{</citation>}}


### (26/70) Mitigating the Exposure Bias in Sentence-Level Grapheme-to-Phoneme (G2P) Transduction (Eunseop Yoon et al., 2023)

{{<citation>}}

Eunseop Yoon, Hee Suk Yoon, Dhananjaya Gowda, SooHwan Eom, Daehyeok Kim, John Harvill, Heting Gao, Mark Hasegawa-Johnson, Chanwoo Kim, Chang D. Yoo. (2023)  
**Mitigating the Exposure Bias in Sentence-Level Grapheme-to-Phoneme (G2P) Transduction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Bias, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2308.08442v1)  

---


**ABSTRACT**  
Text-to-Text Transfer Transformer (T5) has recently been considered for the Grapheme-to-Phoneme (G2P) transduction. As a follow-up, a tokenizer-free byte-level model based on T5 referred to as ByT5, recently gave promising results on word-level G2P conversion by representing each input character with its corresponding UTF-8 encoding. Although it is generally understood that sentence-level or paragraph-level G2P can improve usability in real-world applications as it is better suited to perform on heteronyms and linking sounds between words, we find that using ByT5 for these scenarios is nontrivial. Since ByT5 operates on the character level, it requires longer decoding steps, which deteriorates the performance due to the exposure bias commonly observed in auto-regressive generation models. This paper shows that the performance of sentence-level and paragraph-level G2P can be improved by mitigating such exposure bias using our proposed loss-based sampling method.

{{</citation>}}


### (27/70) SummHelper: Collaborative Human-Computer Summarization (Aviv Slobodkin et al., 2023)

{{<citation>}}

Aviv Slobodkin, Niv Nachum, Shmuel Amar, Ori Shapira, Ido Dagan. (2023)  
**SummHelper: Collaborative Human-Computer Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2308.08363v1)  

---


**ABSTRACT**  
Current approaches for text summarization are predominantly automatic, with rather limited space for human intervention and control over the process. In this paper, we introduce SummHelper, a 2-phase summarization assistant designed to foster human-machine collaboration. The initial phase involves content selection, where the system recommends potential content, allowing users to accept, modify, or introduce additional selections. The subsequent phase, content consolidation, involves SummHelper generating a coherent summary from these selections, which users can then refine using visual mappings between the summary and the source text. Small-scale user studies reveal the effectiveness of our application, with participants being especially appreciative of the balance between automated guidance and opportunities for personal input.

{{</citation>}}


### (28/70) Detoxify Language Model Step-by-Step (Zecheng Tang et al., 2023)

{{<citation>}}

Zecheng Tang, Keyan Zhou, Pinzheng Wang, Yuyang Ding, Juntao Li, Minzhang. (2023)  
**Detoxify Language Model Step-by-Step**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.08295v1)  

---


**ABSTRACT**  
Detoxification for LLMs is challenging since it requires models to avoid generating harmful content while maintaining the generation capability. To ensure the safety of generations, previous detoxification methods detoxify the models by changing the data distributions or constraining the generations from different aspects in a single-step manner. However, these approaches will dramatically affect the generation quality of LLMs, e.g., discourse coherence and semantic consistency, since language models tend to generate along the toxic prompt while detoxification methods work in the opposite direction. To handle such a conflict, we decompose the detoxification process into different sub-steps, where the detoxification is concentrated in the input stage and the subsequent continual generation is based on the non-toxic prompt. Besides, we also calibrate the strong reasoning ability of LLMs by designing a Detox-Chain to connect the above sub-steps in an orderly manner, which allows LLMs to detoxify the text step-by-step. Automatic and human evaluation on two benchmarks reveals that by training with Detox-Chain, six LLMs scaling from 1B to 33B can obtain significant detoxification and generation improvement. Our code and data are available at https://github.com/CODINNLG/Detox-CoT. Warning: examples in the paper may contain uncensored offensive content.

{{</citation>}}


### (29/70) TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series (Chenxi Sun et al., 2023)

{{<citation>}}

Chenxi Sun, Yaliang Li, Hongyan Li, Shenda Hong. (2023)  
**TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Embedding, Time Series  
[Paper Link](http://arxiv.org/abs/2308.08241v1)  

---


**ABSTRACT**  
This work summarizes two strategies for completing time-series (TS) tasks using today's language model (LLM): LLM-for-TS, design and train a fundamental large model for TS data; TS-for-LLM, enable the pre-trained LLM to handle TS data. Considering the insufficient data accumulation, limited resources, and semantic context requirements, this work focuses on TS-for-LLM methods, where we aim to activate LLM's ability for TS data by designing a TS embedding method suitable for LLM. The proposed method is named TEST. It first tokenizes TS, builds an encoder to embed them by instance-wise, feature-wise, and text-prototype-aligned contrast, and then creates prompts to make LLM more open to embeddings, and finally implements TS tasks. Experiments are carried out on TS classification and forecasting tasks using 8 LLMs with different structures and sizes. Although its results cannot significantly outperform the current SOTA models customized for TS tasks, by treating LLM as the pattern machine, it can endow LLM's ability to process TS data without compromising the language ability. This paper is intended to serve as a foundational work that will inspire further research.

{{</citation>}}


### (30/70) Challenges and Opportunities of Using Transformer-Based Multi-Task Learning in NLP Through ML Lifecycle: A Survey (Lovre Torbarina et al., 2023)

{{<citation>}}

Lovre Torbarina, Tin Ferkovic, Lukasz Roguski, Velimir Mihelcic, Bruno Sarlija, Zeljko Kraljevic. (2023)  
**Challenges and Opportunities of Using Transformer-Based Multi-Task Learning in NLP Through ML Lifecycle: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2308.08234v1)  

---


**ABSTRACT**  
The increasing adoption of natural language processing (NLP) models across industries has led to practitioners' need for machine learning systems to handle these models efficiently, from training to serving them in production. However, training, deploying, and updating multiple models can be complex, costly, and time-consuming, mainly when using transformer-based pre-trained language models. Multi-Task Learning (MTL) has emerged as a promising approach to improve efficiency and performance through joint training, rather than training separate models. Motivated by this, we first provide an overview of transformer-based MTL approaches in NLP. Then, we discuss the challenges and opportunities of using MTL approaches throughout typical ML lifecycle phases, specifically focusing on the challenges related to data engineering, model development, deployment, and monitoring phases. This survey focuses on transformer-based MTL architectures and, to the best of our knowledge, is novel in that it systematically analyses how transformer-based MTL in NLP fits into ML lifecycle phases. Furthermore, we motivate research on the connection between MTL and continual learning (CL), as this area remains unexplored. We believe it would be practical to have a model that can handle both MTL and CL, as this would make it easier to periodically re-train the model, update it due to distribution shifts, and add new capabilities to meet real-world requirements.

{{</citation>}}


### (31/70) MoCoSA: Momentum Contrast for Knowledge Graph Completion with Structure-Augmented Pre-trained Language Models (Jiabang He et al., 2023)

{{<citation>}}

Jiabang He, Liu Jia, Lei Wang, Xiyao Li, Xing Xu. (2023)  
**MoCoSA: Momentum Contrast for Knowledge Graph Completion with Structure-Augmented Pre-trained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2308.08204v1)  

---


**ABSTRACT**  
Knowledge Graph Completion (KGC) aims to conduct reasoning on the facts within knowledge graphs and automatically infer missing links. Existing methods can mainly be categorized into structure-based or description-based. On the one hand, structure-based methods effectively represent relational facts in knowledge graphs using entity embeddings. However, they struggle with semantically rich real-world entities due to limited structural information and fail to generalize to unseen entities. On the other hand, description-based methods leverage pre-trained language models (PLMs) to understand textual information. They exhibit strong robustness towards unseen entities. However, they have difficulty with larger negative sampling and often lag behind structure-based methods. To address these issues, in this paper, we propose Momentum Contrast for knowledge graph completion with Structure-Augmented pre-trained language models (MoCoSA), which allows the PLM to perceive the structural information by the adaptable structure encoder. To improve learning efficiency, we proposed momentum hard negative and intra-relation negative sampling. Experimental results demonstrate that our approach achieves state-of-the-art performance in terms of mean reciprocal rank (MRR), with improvements of 2.5% on WN18RR and 21% on OpenBG500.

{{</citation>}}


### (32/70) Enhancing Performance on Seen and Unseen Dialogue Scenarios using Retrieval-Augmented End-to-End Task-Oriented System (Jianguo Zhang et al., 2023)

{{<citation>}}

Jianguo Zhang, Stephen Roller, Kun Qian, Zhiwei Liu, Rui Meng, Shelby Heinecke, Huan Wang, Silvio Savarese, Caiming Xiong. (2023)  
**Enhancing Performance on Seen and Unseen Dialogue Scenarios using Retrieval-Augmented End-to-End Task-Oriented System**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2308.08169v1)  

---


**ABSTRACT**  
End-to-end task-oriented dialogue (TOD) systems have achieved promising performance by leveraging sophisticated natural language understanding and natural language generation capabilities of pre-trained models. This work enables the TOD systems with more flexibility through a simple cache. The cache provides the flexibility to dynamically update the TOD systems and handle both existing and unseen dialogue scenarios. Towards this end, we first fine-tune a retrieval module to effectively retrieve the most relevant information entries from the cache. We then train end-to-end TOD models that can refer to and ground on both dialogue history and retrieved information during TOD generation. The cache is straightforward to construct, and the backbone models of TOD systems are compatible with existing pre-trained generative models. Extensive experiments demonstrate the superior performance of our framework, with a notable improvement in non-empty joint goal accuracy by 6.7% compared to strong baselines.

{{</citation>}}


### (33/70) Sarcasm Detection in a Disaster Context (Tiberiu Sosea et al., 2023)

{{<citation>}}

Tiberiu Sosea, Junyi Jessy Li, Cornelia Caragea. (2023)  
**Sarcasm Detection in a Disaster Context**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Sarcasm Detection, Twitter  
[Paper Link](http://arxiv.org/abs/2308.08156v1)  

---


**ABSTRACT**  
During natural disasters, people often use social media platforms such as Twitter to ask for help, to provide information about the disaster situation, or to express contempt about the unfolding event or public policies and guidelines. This contempt is in some cases expressed as sarcasm or irony. Understanding this form of speech in a disaster-centric context is essential to improving natural language understanding of disaster-related tweets. In this paper, we introduce HurricaneSARC, a dataset of 15,000 tweets annotated for intended sarcasm, and provide a comprehensive investigation of sarcasm detection using pre-trained language models. Our best model is able to obtain as much as 0.70 F1 on our dataset. We also demonstrate that the performance on HurricaneSARC can be improved by leveraging intermediate task transfer learning. We release our data and code at https://github.com/tsosea2/HurricaneSarc.

{{</citation>}}


### (34/70) Fast Training of NMT Model with Data Sorting (Daniela N. Rim et al., 2023)

{{<citation>}}

Daniela N. Rim, Kimera Richard, Heeyoul Choi. (2023)  
**Fast Training of NMT Model with Data Sorting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2308.08153v1)  

---


**ABSTRACT**  
The Transformer model has revolutionized Natural Language Processing tasks such as Neural Machine Translation, and many efforts have been made to study the Transformer architecture, which increased its efficiency and accuracy. One potential area for improvement is to address the computation of empty tokens that the Transformer computes only to discard them later, leading to an unnecessary computational burden. To tackle this, we propose an algorithm that sorts translation sentence pairs based on their length before batching, minimizing the waste of computing power. Since the amount of sorting could violate the independent and identically distributed (i.i.d) data assumption, we sort the data partially. In experiments, we apply the proposed method to English-Korean and English-Luganda language pairs for machine translation and show that there are gains in computational time while maintaining the performance. Our method is independent of architectures, so that it can be easily integrated into any training process with flexible data lengths.

{{</citation>}}


### (35/70) MDDial: A Multi-turn Differential Diagnosis Dialogue Dataset with Reliability Evaluation (Srija Macherla et al., 2023)

{{<citation>}}

Srija Macherla, Man Luo, Mihir Parmar, Chitta Baral. (2023)  
**MDDial: A Multi-turn Differential Diagnosis Dialogue Dataset with Reliability Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2308.08147v1)  

---


**ABSTRACT**  
Dialogue systems for Automatic Differential Diagnosis (ADD) have a wide range of real-life applications. These dialogue systems are promising for providing easy access and reducing medical costs. Building end-to-end ADD dialogue systems requires dialogue training datasets. However, to the best of our knowledge, there is no publicly available ADD dialogue dataset in English (although non-English datasets exist). Driven by this, we introduce MDDial, the first differential diagnosis dialogue dataset in English which can aid to build and evaluate end-to-end ADD dialogue systems. Additionally, earlier studies present the accuracy of diagnosis and symptoms either individually or as a combined weighted score. This method overlooks the connection between the symptoms and the diagnosis. We introduce a unified score for the ADD system that takes into account the interplay between symptoms and diagnosis. This score also indicates the system's reliability. To the end, we train two moderate-size of language models on MDDial. Our experiments suggest that while these language models can perform well on many natural language understanding tasks, including dialogue tasks in the general domain, they struggle to relate relevant symptoms and disease and thus have poor performance on MDDial. MDDial will be released publicly to aid the study of ADD dialogue research.

{{</citation>}}


## cs.LG (15)



### (36/70) Label Propagation Techniques for Artifact Detection in Imbalanced Classes using Photoplethysmogram Signals (Clara Macabiau et al., 2023)

{{<citation>}}

Clara Macabiau, Thanh-Dung Le, Kevin Albert, Philippe Jouvet, Rita Noumeir. (2023)  
**Label Propagation Techniques for Artifact Detection in Imbalanced Classes using Photoplethysmogram Signals**  

---
Primary Category: cs.LG  
Categories: 68T02, cs-LG, cs.LG, eess-SP  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.08480v1)  

---


**ABSTRACT**  
Photoplethysmogram (PPG) signals are widely used in healthcare for monitoring vital signs, but they are susceptible to motion artifacts that can lead to inaccurate interpretations. In this study, the use of label propagation techniques to propagate labels among PPG samples is explored, particularly in imbalanced class scenarios where clean PPG samples are significantly outnumbered by artifact-contaminated samples. With a precision of 91%, a recall of 90% and an F1 score of 90% for the class without artifacts, the results demonstrate its effectiveness in labeling a medical dataset, even when clean samples are rare. For the classification of artifacts our study compares supervised classifiers such as conventional classifiers and neural networks (MLP, Transformers, FCN) with the semi-supervised label propagation algorithm. With a precision of 89%, a recall of 95% and an F1 score of 92%, the KNN supervised model gives good results, but the semi-supervised algorithm performs better in detecting artifacts. The findings suggest that the semi-supervised algorithm label propagation hold promise for artifact detection in PPG signals, which can enhance the reliability of PPG-based health monitoring systems in real-world applications.

{{</citation>}}


### (37/70) LLM4TS: Two-Stage Fine-Tuning for Time-Series Forecasting with Pre-Trained LLMs (Ching Chang et al., 2023)

{{<citation>}}

Ching Chang, Wen-Chih Peng, Tien-Fu Chen. (2023)  
**LLM4TS: Two-Stage Fine-Tuning for Time-Series Forecasting with Pre-Trained LLMs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Computer Vision, Language Model, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.08469v1)  

---


**ABSTRACT**  
In this work, we leverage pre-trained Large Language Models (LLMs) to enhance time-series forecasting. Mirroring the growing interest in unifying models for Natural Language Processing and Computer Vision, we envision creating an analogous model for long-term time-series forecasting. Due to limited large-scale time-series data for building robust foundation models, our approach LLM4TS focuses on leveraging the strengths of pre-trained LLMs. By combining time-series patching with temporal encoding, we have enhanced the capability of LLMs to handle time-series data effectively. Inspired by the supervised fine-tuning in chatbot domains, we prioritize a two-stage fine-tuning process: first conducting supervised fine-tuning to orient the LLM towards time-series data, followed by task-specific downstream fine-tuning. Furthermore, to unlock the flexibility of pre-trained LLMs without extensive parameter adjustments, we adopt several Parameter-Efficient Fine-Tuning (PEFT) techniques. Drawing on these innovations, LLM4TS has yielded state-of-the-art results in long-term forecasting. Our model has also shown exceptional capabilities as both a robust representation learner and an effective few-shot learner, thanks to the knowledge transferred from the pre-trained LLM.

{{</citation>}}


### (38/70) Explainable AI for clinical risk prediction: a survey of concepts, methods, and modalities (Munib Mesinovic et al., 2023)

{{<citation>}}

Munib Mesinovic, Peter Watkinson, Tingting Zhu. (2023)  
**Explainable AI for clinical risk prediction: a survey of concepts, methods, and modalities**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.08407v1)  

---


**ABSTRACT**  
Recent advancements in AI applications to healthcare have shown incredible promise in surpassing human performance in diagnosis and disease prognosis. With the increasing complexity of AI models, however, concerns regarding their opacity, potential biases, and the need for interpretability. To ensure trust and reliability in AI systems, especially in clinical risk prediction models, explainability becomes crucial. Explainability is usually referred to as an AI system's ability to provide a robust interpretation of its decision-making logic or the decisions themselves to human stakeholders. In clinical risk prediction, other aspects of explainability like fairness, bias, trust, and transparency also represent important concepts beyond just interpretability. In this review, we address the relationship between these concepts as they are often used together or interchangeably. This review also discusses recent progress in developing explainable models for clinical risk prediction, highlighting the importance of quantitative and clinical evaluation and validation across multiple common modalities in clinical practice. It emphasizes the need for external validation and the combination of diverse interpretability methods to enhance trust and fairness. Adopting rigorous testing, such as using synthetic datasets with known generative factors, can further improve the reliability of explainability methods. Open access and code-sharing resources are essential for transparency and reproducibility, enabling the growth and trustworthiness of explainable research. While challenges exist, an end-to-end approach to explainability in clinical risk prediction, incorporating stakeholders from clinicians to developers, is essential for success.

{{</citation>}}


### (39/70) Independent Distribution Regularization for Private Graph Embedding (Qi Hu et al., 2023)

{{<citation>}}

Qi Hu, Yangqiu Song. (2023)  
**Independent Distribution Regularization for Private Graph Embedding**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.08360v1)  

---


**ABSTRACT**  
Learning graph embeddings is a crucial task in graph mining tasks. An effective graph embedding model can learn low-dimensional representations from graph-structured data for data publishing benefiting various downstream applications such as node classification, link prediction, etc. However, recent studies have revealed that graph embeddings are susceptible to attribute inference attacks, which allow attackers to infer private node attributes from the learned graph embeddings. To address these concerns, privacy-preserving graph embedding methods have emerged, aiming to simultaneously consider primary learning and privacy protection through adversarial learning. However, most existing methods assume that representation models have access to all sensitive attributes in advance during the training stage, which is not always the case due to diverse privacy preferences. Furthermore, the commonly used adversarial learning technique in privacy-preserving representation learning suffers from unstable training issues. In this paper, we propose a novel approach called Private Variational Graph AutoEncoders (PVGAE) with the aid of independent distribution penalty as a regularization term. Specifically, we split the original variational graph autoencoder (VGAE) to learn sensitive and non-sensitive latent representations using two sets of encoders. Additionally, we introduce a novel regularization to enforce the independence of the encoders. We prove the theoretical effectiveness of regularization from the perspective of mutual information. Experimental results on three real-world datasets demonstrate that PVGAE outperforms other baselines in private embedding learning regarding utility performance and privacy protection.

{{</citation>}}


### (40/70) Convergence of Two-Layer Regression with Nonlinear Units (Yichuan Deng et al., 2023)

{{<citation>}}

Yichuan Deng, Zhao Song, Shenghao Xie. (2023)  
**Convergence of Two-Layer Regression with Nonlinear Units**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Attention, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.08358v1)  

---


**ABSTRACT**  
Large language models (LLMs), such as ChatGPT and GPT4, have shown outstanding performance in many human life task. Attention computation plays an important role in training LLMs. Softmax unit and ReLU unit are the key structure in attention computation. Inspired by them, we put forward a softmax ReLU regression problem. Generally speaking, our goal is to find an optimal solution to the regression problem involving the ReLU unit. In this work, we calculate a close form representation for the Hessian of the loss function. Under certain assumptions, we prove the Lipschitz continuous and the PSDness of the Hessian. Then, we introduce an greedy algorithm based on approximate Newton method, which converges in the sense of the distance to optimal solution. Last, We relax the Lipschitz condition and prove the convergence in the sense of loss value.

{{</citation>}}


### (41/70) Graph Out-of-Distribution Generalization with Controllable Data Augmentation (Bin Lu et al., 2023)

{{<citation>}}

Bin Lu, Xiaoying Gan, Ze Zhao, Shiyu Liang, Luoyi Fu, Xinbing Wang, Chenghu Zhou. (2023)  
**Graph Out-of-Distribution Generalization with Controllable Data Augmentation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: Augmentation, GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2308.08344v1)  

---


**ABSTRACT**  
Graph Neural Network (GNN) has demonstrated extraordinary performance in classifying graph properties. However, due to the selection bias of training and testing data (e.g., training on small graphs and testing on large graphs, or training on dense graphs and testing on sparse graphs), distribution deviation is widespread. More importantly, we often observe \emph{hybrid structure distribution shift} of both scale and density, despite of one-sided biased data partition. The spurious correlations over hybrid distribution deviation degrade the performance of previous GNN methods and show large instability among different datasets. To alleviate this problem, we propose \texttt{OOD-GMixup} to jointly manipulate the training distribution with \emph{controllable data augmentation} in metric space. Specifically, we first extract the graph rationales to eliminate the spurious correlations due to irrelevant information. Secondly, we generate virtual samples with perturbation on graph rationale representation domain to obtain potential OOD training samples. Finally, we propose OOD calibration to measure the distribution deviation of virtual samples by leveraging Extreme Value Theory, and further actively control the training distribution by emphasizing the impact of virtual OOD samples. Extensive studies on several real-world datasets on graph classification demonstrate the superiority of our proposed method over state-of-the-art baselines.

{{</citation>}}


### (42/70) Learning Logic Programs by Discovering Higher-Order Abstractions (Céline Hocquette et al., 2023)

{{<citation>}}

Céline Hocquette, Sebastijan Dumančić, Andrew Cropper. (2023)  
**Learning Logic Programs by Discovering Higher-Order Abstractions**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-PL, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.08334v1)  

---


**ABSTRACT**  
Discovering novel abstractions is important for human-level AI. We introduce an approach to discover higher-order abstractions, such as map, filter, and fold. We focus on inductive logic programming, which induces logic programs from examples and background knowledge. We introduce the higher-order refactoring problem, where the goal is to compress a logic program by introducing higher-order abstractions. We implement our approach in STEVIE, which formulates the higher-order refactoring problem as a constraint optimisation problem. Our experimental results on multiple domains, including program synthesis and visual reasoning, show that, compared to no refactoring, STEVIE can improve predictive accuracies by 27% and reduce learning times by 47%. We also show that STEVIE can discover abstractions that transfer to different domains

{{</citation>}}


### (43/70) It Ain't That Bad: Understanding the Mysterious Performance Drop in OOD Generalization for Generative Transformer Models (Xingcheng Xu et al., 2023)

{{<citation>}}

Xingcheng Xu, Zihao Pan, Haipeng Zhang, Yanqing Yang. (2023)  
**It Ain't That Bad: Understanding the Mysterious Performance Drop in OOD Generalization for Generative Transformer Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.08268v1)  

---


**ABSTRACT**  
Generative Transformer-based models have achieved remarkable proficiency on solving diverse problems. However, their generalization ability is not fully understood and not always satisfying. Researchers take basic mathematical tasks like n-digit addition or multiplication as important perspectives for investigating their generalization behaviors. Curiously, it is observed that when training on n-digit operations (e.g., additions) in which both input operands are n-digit in length, models generalize successfully on unseen n-digit inputs (in-distribution (ID) generalization), but fail miserably and mysteriously on longer, unseen cases (out-of-distribution (OOD) generalization). Studies try to bridge this gap with workarounds such as modifying position embedding, fine-tuning, and priming with more extensive or instructive data. However, without addressing the essential mechanism, there is hardly any guarantee regarding the robustness of these solutions. We bring this unexplained performance drop into attention and ask whether it is purely from random errors. Here we turn to the mechanistic line of research which has notable successes in model interpretability. We discover that the strong ID generalization stems from structured representations, while behind the unsatisfying OOD performance, the models still exhibit clear learned algebraic structures. Specifically, these models map unseen OOD inputs to outputs with equivalence relations in the ID domain. These highlight the potential of the models to carry useful information for improved generalization.

{{</citation>}}


### (44/70) The Expressive Power of Graph Neural Networks: A Survey (Bingxu Zhang et al., 2023)

{{<citation>}}

Bingxu Zhang, Changjun Fan, Shixuan Liu, Kuihua Huang, Xiang Zhao, Jincai Huang, Zhong Liu. (2023)  
**The Expressive Power of Graph Neural Networks: A Survey**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.08235v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) are effective machine learning models for many graph-related applications. Despite their empirical success, many research efforts focus on the theoretical limitations of GNNs, i.e., the GNNs expressive power. Early works in this domain mainly focus on studying the graph isomorphism recognition ability of GNNs, and recent works try to leverage the properties such as subgraph counting and connectivity learning to characterize the expressive power of GNNs, which are more practical and closer to real-world. However, no survey papers and open-source repositories comprehensively summarize and discuss models in this important direction. To fill the gap, we conduct a first survey for models for enhancing expressive power under different forms of definition. Concretely, the models are reviewed based on three categories, i.e., Graph feature enhancement, Graph topology enhancement, and GNNs architecture enhancement.

{{</citation>}}


### (45/70) How To Overcome Confirmation Bias in Semi-Supervised Image Classification By Active Learning (Sandra Gilhuber et al., 2023)

{{<citation>}}

Sandra Gilhuber, Rasmus Hvingelby, Mang Ling Ada Fok, Thomas Seidl. (2023)  
**How To Overcome Confirmation Bias in Semi-Supervised Image Classification By Active Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Active Learning, Bias, Image Classification, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.08224v1)  

---


**ABSTRACT**  
Do we need active learning? The rise of strong deep semi-supervised methods raises doubt about the usability of active learning in limited labeled data settings. This is caused by results showing that combining semi-supervised learning (SSL) methods with a random selection for labeling can outperform existing active learning (AL) techniques. However, these results are obtained from experiments on well-established benchmark datasets that can overestimate the external validity. However, the literature lacks sufficient research on the performance of active semi-supervised learning methods in realistic data scenarios, leaving a notable gap in our understanding. Therefore we present three data challenges common in real-world applications: between-class imbalance, within-class imbalance, and between-class similarity. These challenges can hurt SSL performance due to confirmation bias. We conduct experiments with SSL and AL on simulated data challenges and find that random sampling does not mitigate confirmation bias and, in some cases, leads to worse performance than supervised learning. In contrast, we demonstrate that AL can overcome confirmation bias in SSL in these realistic settings. Our results provide insights into the potential of combining active and semi-supervised learning in the presence of common real-world challenges, which is a promising direction for robust methods when learning with limited labeled data in real-world applications.

{{</citation>}}


### (46/70) Expressivity of Graph Neural Networks Through the Lens of Adversarial Robustness (Francesco Campi et al., 2023)

{{<citation>}}

Francesco Campi, Lukas Gosch, Tom Wollschläger, Yan Scholten, Stephan Günnemann. (2023)  
**Expressivity of Graph Neural Networks Through the Lens of Adversarial Robustness**  

---
Primary Category: cs.LG  
Categories: I-2-6, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.08173v1)  

---


**ABSTRACT**  
We perform the first adversarial robustness study into Graph Neural Networks (GNNs) that are provably more powerful than traditional Message Passing Neural Networks (MPNNs). In particular, we use adversarial robustness as a tool to uncover a significant gap between their theoretically possible and empirically achieved expressive power. To do so, we focus on the ability of GNNs to count specific subgraph patterns, which is an established measure of expressivity, and extend the concept of adversarial robustness to this task. Based on this, we develop efficient adversarial attacks for subgraph counting and show that more powerful GNNs fail to generalize even to small perturbations to the graph's structure. Expanding on this, we show that such architectures also fail to count substructures on out-of-distribution graphs.

{{</citation>}}


### (47/70) Hierarchical Topological Ordering with Conditional Independence Test for Limited Time Series (Anpeng Wu et al., 2023)

{{<citation>}}

Anpeng Wu, Haoxuan Li, Kun Kuang, Keli Zhang, Fei Wu. (2023)  
**Hierarchical Topological Ordering with Conditional Independence Test for Limited Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ME  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.08148v1)  

---


**ABSTRACT**  
Learning directed acyclic graphs (DAGs) to identify causal relations underlying observational data is crucial but also poses significant challenges. Recently, topology-based methods have emerged as a two-step approach to discovering DAGs by first learning the topological ordering of variables and then eliminating redundant edges, while ensuring that the graph remains acyclic. However, one limitation is that these methods would generate numerous spurious edges that require subsequent pruning. To overcome this limitation, in this paper, we propose an improvement to topology-based methods by introducing limited time series data, consisting of only two cross-sectional records that need not be adjacent in time and are subject to flexible timing. By incorporating conditional instrumental variables as exogenous interventions, we aim to identify descendant nodes for each variable. Following this line, we propose a hierarchical topological ordering algorithm with conditional independence test (HT-CIT), which enables the efficient learning of sparse DAGs with a smaller search space compared to other popular approaches. The HT-CIT algorithm greatly reduces the number of edges that need to be pruned. Empirical results from synthetic and real-world datasets demonstrate the superiority of the proposed HT-CIT algorithm.

{{</citation>}}


### (48/70) Is Self-Supervised Pretraining Good for Extrapolation in Molecular Property Prediction? (Shun Takashige et al., 2023)

{{<citation>}}

Shun Takashige, Masatoshi Hanai, Toyotaro Suzumura, Limin Wang, Kenjiro Taura. (2023)  
**Is Self-Supervised Pretraining Good for Extrapolation in Molecular Property Prediction?**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-QM  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.08129v1)  

---


**ABSTRACT**  
The prediction of material properties plays a crucial role in the development and discovery of materials in diverse applications, such as batteries, semiconductors, catalysts, and pharmaceuticals. Recently, there has been a growing interest in employing data-driven approaches by using machine learning technologies, in combination with conventional theoretical calculations. In material science, the prediction of unobserved values, commonly referred to as extrapolation, is particularly critical for property prediction as it enables researchers to gain insight into materials beyond the limits of available data. However, even with the recent advancements in powerful machine learning models, accurate extrapolation is still widely recognized as a significantly challenging problem. On the other hand, self-supervised pretraining is a machine learning technique where a model is first trained on unlabeled data using relatively simple pretext tasks before being trained on labeled data for target tasks. As self-supervised pretraining can effectively utilize material data without observed property values, it has the potential to improve the model's extrapolation ability. In this paper, we clarify how such self-supervised pretraining can enhance extrapolation performance.We propose an experimental framework for the demonstration and empirically reveal that while models were unable to accurately extrapolate absolute property values, self-supervised pretraining enables them to learn relative tendencies of unobserved property values and improve extrapolation performance.

{{</citation>}}


### (49/70) How to Mask in Error Correction Code Transformer: Systematic and Double Masking (Seong-Joon Park et al., 2023)

{{<citation>}}

Seong-Joon Park, Hee-Youl Kwak, Sang-Hyo Kim, Sunghwan Kim, Yongjune Kim, Jong-Seon No. (2023)  
**How to Mask in Error Correction Code Transformer: Systematic and Double Masking**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-IT, cs-LG, cs.LG, math-IT  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.08128v1)  

---


**ABSTRACT**  
In communication and storage systems, error correction codes (ECCs) are pivotal in ensuring data reliability. As deep learning's applicability has broadened across diverse domains, there is a growing research focus on neural network-based decoders that outperform traditional decoding algorithms. Among these neural decoders, Error Correction Code Transformer (ECCT) has achieved the state-of-the-art performance, outperforming other methods by large margins. To further enhance the performance of ECCT, we propose two novel methods. First, leveraging the systematic encoding technique of ECCs, we introduce a new masking matrix for ECCT, aiming to improve the performance and reduce the computational complexity. Second, we propose a novel transformer architecture of ECCT called a double-masked ECCT. This architecture employs two different mask matrices in a parallel manner to learn more diverse features of the relationship between codeword bits in the masked self-attention blocks. Extensive simulation results show that the proposed double-masked ECCT outperforms the conventional ECCT, achieving the state-of-the-art decoding performance with significant margins.

{{</citation>}}


### (50/70) S-Mixup: Structural Mixup for Graph Neural Networks (Junghurn Kim et al., 2023)

{{<citation>}}

Junghurn Kim, Sukwon Yun, Chanyoung Park. (2023)  
**S-Mixup: Structural Mixup for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.08097v1)  

---


**ABSTRACT**  
Existing studies for applying the mixup technique on graphs mainly focus on graph classification tasks, while the research in node classification is still under-explored. In this paper, we propose a novel mixup augmentation for node classification called Structural Mixup (S-Mixup). The core idea is to take into account the structural information while mixing nodes. Specifically, S-Mixup obtains pseudo-labels for unlabeled nodes in a graph along with their prediction confidence via a Graph Neural Network (GNN) classifier. These serve as the criteria for the composition of the mixup pool for both inter and intra-class mixups. Furthermore, we utilize the edge gradient obtained from the GNN training and propose a gradient-based edge selection strategy for selecting edges to be attached to the nodes generated by the mixup. Through extensive experiments on real-world benchmark datasets, we demonstrate the effectiveness of S-Mixup evaluated on the node classification task. We observe that S-Mixup enhances the robustness and generalization performance of GNNs, especially in heterophilous situations. The source code of S-Mixup can be found at \url{https://github.com/SukwonYun/S-Mixup}

{{</citation>}}


## cs.IR (4)



### (51/70) A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems (Keqin Bao et al., 2023)

{{<citation>}}

Keqin Bao, Jizhi Zhang, Wenjie Wang, Yang Zhang, Zhengyi Yang, Yancheng Luo, Fuli Feng, Xiangnaan He, Qi Tian. (2023)  
**A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.08434v1)  

---


**ABSTRACT**  
As the focus on Large Language Models (LLMs) in the field of recommendation intensifies, the optimization of LLMs for recommendation purposes (referred to as LLM4Rec) assumes a crucial role in augmenting their effectiveness in providing recommendations. However, existing approaches for LLM4Rec often assess performance using restricted sets of candidates, which may not accurately reflect the models' overall ranking capabilities. In this paper, our objective is to investigate the comprehensive ranking capacity of LLMs and propose a two-step grounding framework known as BIGRec (Bi-step Grounding Paradigm for Recommendation). It initially grounds LLMs to the recommendation space by fine-tuning them to generate meaningful tokens for items and subsequently identifies appropriate actual items that correspond to the generated tokens. By conducting extensive experiments on two datasets, we substantiate the superior performance, capacity for handling few-shot scenarios, and versatility across multiple domains exhibited by BIGRec. Furthermore, we observe that the marginal benefits derived from increasing the quantity of training samples are modest for BIGRec, implying that LLMs possess the limited capability to assimilate statistical information, such as popularity and collaborative filtering, due to their robust semantic priors. These findings also underline the efficacy of integrating diverse statistical information into the LLM4Rec framework, thereby pointing towards a potential avenue for future research. Our code and data are available at https://github.com/SAI990323/Grounding4Rec.

{{</citation>}}


### (52/70) Knowledge-Enhanced Multi-Label Few-Shot Product Attribute-Value Extraction (Jiaying Gong et al., 2023)

{{<citation>}}

Jiaying Gong, Wei-Te Chen, Hoda Eldardiry. (2023)  
**Knowledge-Enhanced Multi-Label Few-Shot Product Attribute-Value Extraction**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.08413v1)  

---


**ABSTRACT**  
Existing attribute-value extraction (AVE) models require large quantities of labeled data for training. However, new products with new attribute-value pairs enter the market every day in real-world e-Commerce. Thus, we formulate AVE in multi-label few-shot learning (FSL), aiming to extract unseen attribute value pairs based on a small number of training examples. We propose a Knowledge-Enhanced Attentive Framework (KEAF) based on prototypical networks, leveraging the generated label description and category information to learn more discriminative prototypes. Besides, KEAF integrates with hybrid attention to reduce noise and capture more informative semantics for each class by calculating the label-relevant and query-related weights. To achieve multi-label inference, KEAF further learns a dynamic threshold by integrating the semantic information from both the support set and the query set. Extensive experiments with ablation studies conducted on two datasets demonstrate that KEAF outperforms other SOTA models for information extraction in FSL. The code can be found at: https://github.com/gjiaying/KEAF

{{</citation>}}


### (53/70) Pre-training with Large Language Model-based Document Expansion for Dense Passage Retrieval (Guangyuan Ma et al., 2023)

{{<citation>}}

Guangyuan Ma, Xing Wu, Peng Wang, Zijia Lin, Songlin Hu. (2023)  
**Pre-training with Large Language Model-based Document Expansion for Dense Passage Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.08285v1)  

---


**ABSTRACT**  
In this paper, we systematically study the potential of pre-training with Large Language Model(LLM)-based document expansion for dense passage retrieval. Concretely, we leverage the capabilities of LLMs for document expansion, i.e. query generation, and effectively transfer expanded knowledge to retrievers using pre-training strategies tailored for passage retrieval. These strategies include contrastive learning and bottlenecked query generation. Furthermore, we incorporate a curriculum learning strategy to reduce the reliance on LLM inferences. Experimental results demonstrate that pre-training with LLM-based document expansion significantly boosts the retrieval performance on large-scale web-search tasks. Our work shows strong zero-shot and out-of-domain retrieval abilities, making it more widely applicable for retrieval when initializing with no human-labeled data.

{{</citation>}}


### (54/70) Uncovering User Interest from Biased and Noised Watch Time in Video Recommendation (Haiyuan Zhao et al., 2023)

{{<citation>}}

Haiyuan Zhao, Lei Zhang, Jun Xu, Guohao Cai, Zhenhua Dong, Ji-Rong Wen. (2023)  
**Uncovering User Interest from Biased and Noised Watch Time in Video Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.08120v1)  

---


**ABSTRACT**  
In the video recommendation, watch time is commonly adopted as an indicator of user interest. However, watch time is not only influenced by the matching of users' interests but also by other factors, such as duration bias and noisy watching. Duration bias refers to the tendency for users to spend more time on videos with longer durations, regardless of their actual interest level. Noisy watching, on the other hand, describes users taking time to determine whether they like a video or not, which can result in users spending time watching videos they do not like. Consequently, the existence of duration bias and noisy watching make watch time an inadequate label for indicating user interest. Furthermore, current methods primarily address duration bias and ignore the impact of noisy watching, which may limit their effectiveness in uncovering user interest from watch time. In this study, we first analyze the generation mechanism of users' watch time from a unified causal viewpoint. Specifically, we considered the watch time as a mixture of the user's actual interest level, the duration-biased watch time, and the noisy watch time. To mitigate both the duration bias and noisy watching, we propose Debiased and Denoised watch time Correction (D$^2$Co), which can be divided into two steps: First, we employ a duration-wise Gaussian Mixture Model plus frequency-weighted moving average for estimating the bias and noise terms; then we utilize a sensitivity-controlled correction function to separate the user interest from the watch time, which is robust to the estimation error of bias and noise terms. The experiments on two public video recommendation datasets and online A/B testing indicate the effectiveness of the proposed method.

{{</citation>}}


## stat.ML (1)



### (55/70) Eliciting Risk Aversion with Inverse Reinforcement Learning via Interactive Questioning (Ziteng Cheng et al., 2023)

{{<citation>}}

Ziteng Cheng, Anthony Coache, Sebastian Jaimungal. (2023)  
**Eliciting Risk Aversion with Inverse Reinforcement Learning via Interactive Questioning**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.08427v1)  

---


**ABSTRACT**  
This paper proposes a novel framework for identifying an agent's risk aversion using interactive questioning. Our study is conducted in two scenarios: a one-period case and an infinite horizon case. In the one-period case, we assume that the agent's risk aversion is characterized by a cost function of the state and a distortion risk measure. In the infinite horizon case, we model risk aversion with an additional component, a discount factor. Assuming the access to a finite set of candidates containing the agent's true risk aversion, we show that asking the agent to demonstrate her optimal policies in various environment, which may depend on their previous answers, is an effective means of identifying the agent's risk aversion. Specifically, we prove that the agent's risk aversion can be identified as the number of questions tends to infinity, and the questions are randomly designed. We also develop an algorithm for designing optimal questions and provide empirical evidence that our method learns risk aversion significantly faster than randomly designed questions in simulations. Our framework has important applications in robo-advising and provides a new approach for identifying an agent's risk preferences.

{{</citation>}}


## cs.CR (1)



### (56/70) Diff-CAPTCHA: An Image-based CAPTCHA with Security Enhanced by Denoising Diffusion Model (Ran Jiang et al., 2023)

{{<citation>}}

Ran Jiang, Sanfeng Zhang, Linfeng Liu, Yanbing Peng. (2023)  
**Diff-CAPTCHA: An Image-based CAPTCHA with Security Enhanced by Denoising Diffusion Model**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.08367v1)  

---


**ABSTRACT**  
To enhance the security of text CAPTCHAs, various methods have been employed, such as adding the interference lines on the text, randomly distorting the characters, and overlapping multiple characters. These methods partly increase the difficulty of automated segmentation and recognition attacks. However, facing the rapid development of the end-to-end breaking algorithms, their security has been greatly weakened. The diffusion model is a novel image generation model that can generate the text images with deep fusion of characters and background images. In this paper, an image-click CAPTCHA scheme called Diff-CAPTCHA is proposed based on denoising diffusion models. The background image and characters of the CAPTCHA are treated as a whole to guide the generation process of a diffusion model, thus weakening the character features available for machine learning, enhancing the diversity of character features in the CAPTCHA, and increasing the difficulty of breaking algorithms. To evaluate the security of Diff-CAPTCHA, this paper develops several attack methods, including end-to-end attacks based on Faster R-CNN and two-stage attacks, and Diff-CAPTCHA is compared with three baseline schemes, including commercial CAPTCHA scheme and security-enhanced CAPTCHA scheme based on style transfer. The experimental results show that diffusion models can effectively enhance CAPTCHA security while maintaining good usability in human testing.

{{</citation>}}


## eess.IV (4)



### (57/70) GAEI-UNet: Global Attention and Elastic Interaction U-Net for Vessel Image Segmentation (Ruiqiang Xiao et al., 2023)

{{<citation>}}

Ruiqiang Xiao, Zhuoyue Wan, Yang Xiang. (2023)  
**GAEI-UNet: Global Attention and Elastic Interaction U-Net for Vessel Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.08345v1)  

---


**ABSTRACT**  
Vessel image segmentation plays a pivotal role in medical diagnostics, aiding in the early detection and treatment of vascular diseases. While segmentation based on deep learning has shown promising results, effectively segmenting small structures and maintaining connectivity between them remains challenging. To address these limitations, we propose GAEI-UNet, a novel model that combines global attention and elastic interaction-based techniques. GAEI-UNet leverages global spatial and channel context information to enhance high-level semantic understanding within the U-Net architecture, enabling precise segmentation of small vessels. Additionally, we adopt an elastic interaction-based loss function to improve connectivity among these fine structures. By capturing the forces generated by misalignment between target and predicted shapes, our model effectively learns to preserve the correct topology of vessel networks. Evaluation on retinal vessel dataset -- DRIVE demonstrates the superior performance of GAEI-UNet in terms of SE and connectivity of small structures, without significantly increasing computational complexity. This research aims to advance the field of vessel image segmentation, providing more accurate and reliable diagnostic tools for the medical community. The implementation code is available on Code.

{{</citation>}}


### (58/70) ECPC-IDS:A benchmark endometrail cancer PET/CT image dataset for evaluation of semantic segmentation and detection of hypermetabolic regions (Dechao Tang et al., 2023)

{{<citation>}}

Dechao Tang, Xuanyi Li, Tianming Du, Deguo Ma, Zhiyu Ma, Hongzan Sun, Marcin Grzegorzek, Huiyan Jiang, Chen Li. (2023)  
**ECPC-IDS:A benchmark endometrail cancer PET/CT image dataset for evaluation of semantic segmentation and detection of hypermetabolic regions**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.08313v1)  

---


**ABSTRACT**  
Endometrial cancer is one of the most common tumors in the female reproductive system and is the third most common gynecological malignancy that causes death after ovarian and cervical cancer. Early diagnosis can significantly improve the 5-year survival rate of patients. With the development of artificial intelligence, computer-assisted diagnosis plays an increasingly important role in improving the accuracy and objectivity of diagnosis, as well as reducing the workload of doctors. However, the absence of publicly available endometrial cancer image datasets restricts the application of computer-assisted diagnostic techniques.In this paper, a publicly available Endometrial Cancer PET/CT Image Dataset for Evaluation of Semantic Segmentation and Detection of Hypermetabolic Regions (ECPC-IDS) are published. Specifically, the segmentation section includes PET and CT images, with a total of 7159 images in multiple formats. In order to prove the effectiveness of segmentation methods on ECPC-IDS, five classical deep learning semantic segmentation methods are selected to test the image segmentation task. The object detection section also includes PET and CT images, with a total of 3579 images and XML files with annotation information. Six deep learning methods are selected for experiments on the detection task.This study conduct extensive experiments using deep learning-based semantic segmentation and object detection methods to demonstrate the differences between various methods on ECPC-IDS. As far as we know, this is the first publicly available dataset of endometrial cancer with a large number of multiple images, including a large amount of information required for image and target detection. ECPC-IDS can aid researchers in exploring new algorithms to enhance computer-assisted technology, benefiting both clinical doctors and patients greatly.

{{</citation>}}


### (59/70) CARE: A Large Scale CT Image Dataset and Clinical Applicable Benchmark Model for Rectal Cancer Segmentation (Hantao Zhang et al., 2023)

{{<citation>}}

Hantao Zhang, Weidong Guo, Chenyang Qiu, Shouhong Wan, Bingbing Zou, Wanqin Wang, Peiquan Jin. (2023)  
**CARE: A Large Scale CT Image Dataset and Clinical Applicable Benchmark Model for Rectal Cancer Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2308.08283v1)  

---


**ABSTRACT**  
Rectal cancer segmentation of CT image plays a crucial role in timely clinical diagnosis, radiotherapy treatment, and follow-up. Although current segmentation methods have shown promise in delineating cancerous tissues, they still encounter challenges in achieving high segmentation precision. These obstacles arise from the intricate anatomical structures of the rectum and the difficulties in performing differential diagnosis of rectal cancer. Additionally, a major obstacle is the lack of a large-scale, finely annotated CT image dataset for rectal cancer segmentation. To address these issues, this work introduces a novel large scale rectal cancer CT image dataset CARE with pixel-level annotations for both normal and cancerous rectum, which serves as a valuable resource for algorithm research and clinical application development. Moreover, we propose a novel medical cancer lesion segmentation benchmark model named U-SAM. The model is specifically designed to tackle the challenges posed by the intricate anatomical structures of abdominal organs by incorporating prompt information. U-SAM contains three key components: promptable information (e.g., points) to aid in target area localization, a convolution module for capturing low-level lesion details, and skip-connections to preserve and recover spatial information during the encoding-decoding process. To evaluate the effectiveness of U-SAM, we systematically compare its performance with several popular segmentation methods on the CARE dataset. The generalization of the model is further verified on the WORD dataset. Extensive experiments demonstrate that the proposed U-SAM outperforms state-of-the-art methods on these two datasets. These experiments can serve as the baseline for future research and clinical application development.

{{</citation>}}


### (60/70) AATCT-IDS: A Benchmark Abdominal Adipose Tissue CT Image Dataset for Image Denoising, Semantic Segmentation, and Radiomics Evaluation (Zhiyu Ma et al., 2023)

{{<citation>}}

Zhiyu Ma, Chen Li, Tianming Du, Le Zhang, Dechao Tang, Deguo Ma, Shanchuan Huang, Yan Liu, Yihao Sun, Zhihao Chen, Jin Yuan, Qianqing Nie, Marcin Grzegorzek, Hongzan Sun. (2023)  
**AATCT-IDS: A Benchmark Abdominal Adipose Tissue CT Image Dataset for Image Denoising, Semantic Segmentation, and Radiomics Evaluation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.08172v1)  

---


**ABSTRACT**  
Methods: In this study, a benchmark \emph{Abdominal Adipose Tissue CT Image Dataset} (AATTCT-IDS) containing 300 subjects is prepared and published. AATTCT-IDS publics 13,732 raw CT slices, and the researchers individually annotate the subcutaneous and visceral adipose tissue regions of 3,213 of those slices that have the same slice distance to validate denoising methods, train semantic segmentation models, and study radiomics. For different tasks, this paper compares and analyzes the performance of various methods on AATTCT-IDS by combining the visualization results and evaluation data. Thus, verify the research potential of this data set in the above three types of tasks.   Results: In the comparative study of image denoising, algorithms using a smoothing strategy suppress mixed noise at the expense of image details and obtain better evaluation data. Methods such as BM3D preserve the original image structure better, although the evaluation data are slightly lower. The results show significant differences among them. In the comparative study of semantic segmentation of abdominal adipose tissue, the segmentation results of adipose tissue by each model show different structural characteristics. Among them, BiSeNet obtains segmentation results only slightly inferior to U-Net with the shortest training time and effectively separates small and isolated adipose tissue. In addition, the radiomics study based on AATTCT-IDS reveals three adipose distributions in the subject population.   Conclusion: AATTCT-IDS contains the ground truth of adipose tissue regions in abdominal CT slices. This open-source dataset can attract researchers to explore the multi-dimensional characteristics of abdominal adipose tissue and thus help physicians and patients in clinical practice. AATCT-IDS is freely published for non-commercial purpose at: \url{https://figshare.com/articles/dataset/AATTCT-IDS/23807256}.

{{</citation>}}


## cs.CY (1)



### (61/70) Expert opinions on making GDPR usable (Johanna Johansen, 2023)

{{<citation>}}

Johanna Johansen. (2023)  
**Expert opinions on making GDPR usable**  

---
Primary Category: cs.CY  
Categories: cs-CR, cs-CY, cs-HC, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.08287v1)  

---


**ABSTRACT**  
We present the results of a study done in order to validate concepts and methods that have been introduced in (Johansen and Fischer-Hubner, 2020. "Making GDPR Usable: A Model to Support Usability Evaluations of Privacy." in IFIP AICT 576, 275-291). We use as respondents in our interviews experts working across fields of relevance to these concepts, including law and data protection/privacy, certifications and standardization, and usability (as studied in the field of Human-Computer Interaction). We study the experts' opinions about four new concepts, namely: (i) a definition of Usable Privacy, (ii) 30 Usable Privacy Goals identified as excerpts from the GDPR (European General Data Protection Regulation), (iii) a set of 25 corresponding Usable Privacy Criteria together with their multiple measurable sub-criteria, and (iv) the Usable Privacy Cube model, which puts all these together with the EuroPriSe certification criteria, with the purpose of making explicit several aspects of certification processes such as orderings of criteria, interactions between these, different stakeholder perspectives, and context of use/processing.   The expert opinions are varied, example-rich, and forward-looking, which gives a impressive list of open problems where the above four concepts can work as a foundation for further developments. We employed a critical qualitative research, using theory triangulation to analyze the data representing three groups of experts, categorized as 'certifications', 'law', and 'usability', coming both from industry and academia. The results of our analysis show agreement among the experts about the need for evaluations and measuring of usability of privacy in order to allow for exercising data subjects' rights and to evaluate the degree to which data controllers comply with the data protection principles.

{{</citation>}}


## cs.NI (1)



### (62/70) Deep Reinforcement Learning based Joint Spectrum Allocation and Configuration Design for STAR-RIS-Assisted V2X Communications (Pyae Sone Aung et al., 2023)

{{<citation>}}

Pyae Sone Aung, Loc X. Nguyen, Yan Kyaw Tun, Zhu Han, Choong Seon Hong. (2023)  
**Deep Reinforcement Learning based Joint Spectrum Allocation and Configuration Design for STAR-RIS-Assisted V2X Communications**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.08279v1)  

---


**ABSTRACT**  
Vehicle-to-Everything (V2X) communications play a crucial role in ensuring safe and efficient modern transportation systems. However, challenges arise in scenarios with buildings, leading to signal obstruction and coverage limitations. To alleviate these challenges, reconfigurable intelligent surface (RIS) is regarded as an effective solution for communication performance by tuning passive signal reflection. RIS has acquired prominence in 6G networks due to its improved spectral efficiency, simple deployment, and cost-effectiveness. Nevertheless, conventional RIS solutions have coverage limitations. Therefore, researchers have started focusing on the promising concept of simultaneously transmitting and reflecting RIS (STAR-RIS), which provides 360\degree coverage while utilizing the advantages of RIS technology. In this paper, a STAR-RIS-assisted V2X communication system is investigated. An optimization problem is formulated to maximize the achievable data rate for vehicle-to-infrastructure (V2I) users while satisfying the latency and reliability requirements of vehicle-to-vehicle (V2V) pairs by jointly optimizing the spectrum allocation, amplitudes, and phase shifts of STAR-RIS elements, digital beamforming vectors for V2I links, and transmit power for V2V pairs. Since it is challenging to solve in polynomial time, we decompose our problem into two sub-problems. For the first sub-problem, we model the control variables as a Markov Decision Process (MDP) and propose a combined double deep Q-network (DDQN) with an attention mechanism so that the model can potentially focus on relevant inputs. For the latter, a standard optimization-based approach is implemented to provide a real-time solution, reducing computational costs. Extensive numerical analysis is developed to demonstrate the superiority of our proposed algorithm compared to benchmark schemes.

{{</citation>}}


## cs.SE (1)



### (63/70) Boosting Commit Classification with Contrastive Learning (Jiajun Tong et al., 2023)

{{<citation>}}

Jiajun Tong, Zhixiao Wang, Xiaobin Rui. (2023)  
**Boosting Commit Classification with Contrastive Learning**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.08263v1)  

---


**ABSTRACT**  
Commit Classification (CC) is an important task in software maintenance, which helps software developers classify code changes into different types according to their nature and purpose. It allows developers to understand better how their development efforts are progressing, identify areas where they need improvement, and make informed decisions about when and how to release new software versions. However, existing models need lots of manually labeled data for fine-tuning processes, and ignore sentence-level semantic information, which is often essential for discovering the difference between diverse commits. Therefore, it is still challenging to solve CC in fewshot scenario.   To solve the above problems, we propose a contrastive learning-based commit classification framework. Firstly, we generate $K$ sentences and pseudo-labels according to the labels of the dataset, which aims to enhance the dataset. Secondly, we randomly group the augmented data $N$ times to compare their similarity with the positive $T_p^{|C|}$ and negative $T_n^{|C|}$ samples. We utilize individual pretrained sentence transformers (ST)s to efficiently obtain the sentence-level embeddings from different features respectively. Finally, we adopt the cosine similarity function to limit the distribution of vectors, similar vectors are more adjacent. The light fine-tuned model is then applied to the label prediction of incoming commits.   Extensive experiments on two open available datasets demonstrate that our framework can solve the CC problem simply but effectively in fewshot scenarios, while achieving state-of-the-art(SOTA) performance and improving the adaptability of the model without requiring a large number of training samples for fine-tuning. The code, data, and trained models are available at https://github.com/AppleMax1992/CommitFit.

{{</citation>}}


## cs.NE (2)



### (64/70) Inherent Redundancy in Spiking Neural Networks (Man Yao et al., 2023)

{{<citation>}}

Man Yao, Jiakui Hu, Guangshe Zhao, Yaoyuan Wang, Ziyang Zhang, Bo Xu, Guoqi Li. (2023)  
**Inherent Redundancy in Spiking Neural Networks**  

---
Primary Category: cs.NE  
Categories: cs-CV, cs-LG, cs-NE, cs.NE  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.08227v1)  

---


**ABSTRACT**  
Spiking Neural Networks (SNNs) are well known as a promising energy-efficient alternative to conventional artificial neural networks. Subject to the preconceived impression that SNNs are sparse firing, the analysis and optimization of inherent redundancy in SNNs have been largely overlooked, thus the potential advantages of spike-based neuromorphic computing in accuracy and energy efficiency are interfered. In this work, we pose and focus on three key questions regarding the inherent redundancy in SNNs. We argue that the redundancy is induced by the spatio-temporal invariance of SNNs, which enhances the efficiency of parameter utilization but also invites lots of noise spikes. Further, we analyze the effect of spatio-temporal invariance on the spatio-temporal dynamics and spike firing of SNNs. Then, motivated by these analyses, we propose an Advance Spatial Attention (ASA) module to harness SNNs' redundancy, which can adaptively optimize their membrane potential distribution by a pair of individual spatial attention sub-modules. In this way, noise spike features are accurately regulated. Experimental results demonstrate that the proposed method can significantly drop the spike firing with better performance than state-of-the-art SNN baselines. Our code is available in \url{https://github.com/BICLab/ASA-SNN}.

{{</citation>}}


### (65/70) Expressivity of Spiking Neural Networks (Manjot Singh et al., 2023)

{{<citation>}}

Manjot Singh, Adalbert Fono, Gitta Kutyniok. (2023)  
**Expressivity of Spiking Neural Networks**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.08218v1)  

---


**ABSTRACT**  
This article studies the expressive power of spiking neural networks where information is encoded in the firing time of neurons. The implementation of spiking neural networks on neuromorphic hardware presents a promising choice for future energy-efficient AI applications. However, there exist very few results that compare the computational power of spiking neurons to arbitrary threshold circuits and sigmoidal neurons. Additionally, it has also been shown that a network of spiking neurons is capable of approximating any continuous function. By using the Spike Response Model as a mathematical model of a spiking neuron and assuming a linear response function, we prove that the mapping generated by a network of spiking neurons is continuous piecewise linear. We also show that a spiking neural network can emulate the output of any multi-layer (ReLU) neural network. Furthermore, we show that the maximum number of linear regions generated by a spiking neuron scales exponentially with respect to the input dimension, a characteristic that distinguishes it significantly from an artificial (ReLU) neuron. Our results further extend the understanding of the approximation properties of spiking neural networks and open up new avenues where spiking neural networks can be deployed instead of artificial neural networks without any performance loss.

{{</citation>}}


## cs.RO (1)



### (66/70) HyperSNN: A new efficient and robust deep learning model for resource constrained control applications (Zhanglu Yan et al., 2023)

{{<citation>}}

Zhanglu Yan, Shida Wang, Kaiwen Tang, Wong-Fai Wong. (2023)  
**HyperSNN: A new efficient and robust deep learning model for resource constrained control applications**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.08222v1)  

---


**ABSTRACT**  
In light of the increasing adoption of edge computing in areas such as intelligent furniture, robotics, and smart homes, this paper introduces HyperSNN, an innovative method for control tasks that uses spiking neural networks (SNNs) in combination with hyperdimensional computing. HyperSNN substitutes expensive 32-bit floating point multiplications with 8-bit integer additions, resulting in reduced energy consumption while enhancing robustness and potentially improving accuracy. Our model was tested on AI Gym benchmarks, including Cartpole, Acrobot, MountainCar, and Lunar Lander. HyperSNN achieves control accuracies that are on par with conventional machine learning methods but with only 1.36% to 9.96% of the energy expenditure. Furthermore, our experiments showed increased robustness when using HyperSNN. We believe that HyperSNN is especially suitable for interactive, mobile, and wearable devices, promoting energy-efficient and robust system design. Furthermore, it paves the way for the practical implementation of complex algorithms like model predictive control (MPC) in real-world industrial scenarios.

{{</citation>}}


## cs.AR (1)



### (67/70) Accelerating Generic Graph Neural Networks via Architecture, Compiler, Partition Method Co-Design (Shuwen Lu et al., 2023)

{{<citation>}}

Shuwen Lu, Zhihui Zhang, Cong Guo, Jingwen Leng, Yangjie Zhou, Minyi Guo. (2023)  
**Accelerating Generic Graph Neural Networks via Architecture, Compiler, Partition Method Co-Design**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-LG, cs.AR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.08174v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have shown significant accuracy improvements in a variety of graph learning domains, sparking considerable research interest. To translate these accuracy improvements into practical applications, it is essential to develop high-performance and efficient hardware acceleration for GNN models. However, designing GNN accelerators faces two fundamental challenges: the high bandwidth requirement of GNN models and the diversity of GNN models. Previous works have addressed the first challenge by using more expensive memory interfaces to achieve higher bandwidth. For the second challenge, existing works either support specific GNN models or have generic designs with poor hardware utilization.   In this work, we tackle both challenges simultaneously. First, we identify a new type of partition-level operator fusion, which we utilize to internally reduce the high bandwidth requirement of GNNs. Next, we introduce partition-level multi-threading to schedule the concurrent processing of graph partitions, utilizing different hardware resources. To further reduce the extra on-chip memory required by multi-threading, we propose fine-grained graph partitioning to generate denser graph partitions. Importantly, these three methods make no assumptions about the targeted GNN models, addressing the challenge of model variety. We implement these methods in a framework called SwitchBlade, consisting of a compiler, a graph partitioner, and a hardware accelerator. Our evaluation demonstrates that SwitchBlade achieves an average speedup of $1.85\times$ and energy savings of $19.03\times$ compared to the NVIDIA V100 GPU. Additionally, SwitchBlade delivers performance comparable to state-of-the-art specialized accelerators.

{{</citation>}}


## cs.SD (2)



### (68/70) SCANet: A Self- and Cross-Attention Network for Audio-Visual Speech Separation (Kai Li et al., 2023)

{{<citation>}}

Kai Li, Runxuan Yang, Xiaolin Hu. (2023)  
**SCANet: A Self- and Cross-Attention Network for Audio-Visual Speech Separation**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.08143v1)  

---


**ABSTRACT**  
The integration of different modalities, such as audio and visual information, plays a crucial role in human perception of the surrounding environment. Recent research has made significant progress in designing fusion modules for audio-visual speech separation. However, they predominantly focus on multi-modal fusion architectures situated either at the top or bottom positions, rather than comprehensively considering multi-modal fusion at various hierarchical positions within the network. In this paper, we propose a novel model called self- and cross-attention network (SCANet), which leverages the attention mechanism for efficient audio-visual feature fusion. SCANet consists of two types of attention blocks: self-attention (SA) and cross-attention (CA) blocks, where the CA blocks are distributed at the top (TCA), middle (MCA) and bottom (BCA) of SCANet. These blocks maintain the ability to learn modality-specific features and enable the extraction of different semantics from audio-visual features. Comprehensive experiments on three standard audio-visual separation benchmarks (LRS2, LRS3, and VoxCeleb2) demonstrate the effectiveness of SCANet, outperforming existing state-of-the-art (SOTA) methods while maintaining comparable inference time.

{{</citation>}}


### (69/70) Radio2Text: Streaming Speech Recognition Using mmWave Radio Signals (Running Zhao et al., 2023)

{{<citation>}}

Running Zhao, Jiangtao Yu, Hang Zhao, Edith C. H. Ngai. (2023)  
**Radio2Text: Streaming Speech Recognition Using mmWave Radio Signals**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-HC, cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition, Transformer  
[Paper Link](http://arxiv.org/abs/2308.08125v1)  

---


**ABSTRACT**  
Millimeter wave (mmWave) based speech recognition provides more possibility for audio-related applications, such as conference speech transcription and eavesdropping. However, considering the practicality in real scenarios, latency and recognizable vocabulary size are two critical factors that cannot be overlooked. In this paper, we propose Radio2Text, the first mmWave-based system for streaming automatic speech recognition (ASR) with a vocabulary size exceeding 13,000 words. Radio2Text is based on a tailored streaming Transformer that is capable of effectively learning representations of speech-related features, paving the way for streaming ASR with a large vocabulary. To alleviate the deficiency of streaming networks unable to access entire future inputs, we propose the Guidance Initialization that facilitates the transfer of feature knowledge related to the global context from the non-streaming Transformer to the tailored streaming Transformer through weight inheritance. Further, we propose a cross-modal structure based on knowledge distillation (KD), named cross-modal KD, to mitigate the negative effect of low quality mmWave signals on recognition performance. In the cross-modal KD, the audio streaming Transformer provides feature and response guidance that inherit fruitful and accurate speech information to supervise the training of the tailored radio streaming Transformer. The experimental results show that our Radio2Text can achieve a character error rate of 5.7% and a word error rate of 9.4% for the recognition of a vocabulary consisting of over 13,000 words.

{{</citation>}}


## cs.HC (1)



### (70/70) ChatLogo: A Large Language Model-Driven Hybrid Natural-Programming Language Interface for Agent-based Modeling and Programming (John Chen et al., 2023)

{{<citation>}}

John Chen, Uri Wilensky. (2023)  
**ChatLogo: A Large Language Model-Driven Hybrid Natural-Programming Language Interface for Agent-based Modeling and Programming**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-SE, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.08102v1)  

---


**ABSTRACT**  
Building on Papert (1980)'s idea of children talking to computers, we propose ChatLogo, a hybrid natural-programming language interface for agent-based modeling and programming. We build upon previous efforts to scaffold ABM & P learning and recent development in leveraging large language models (LLMs) to support the learning of computational programming. ChatLogo aims to support conversations with computers in a mix of natural and programming languages, provide a more user-friendly interface for novice learners, and keep the technical system from over-reliance on any single LLM. We introduced the main elements of our design: an intelligent command center, and a conversational interface to support creative expression. We discussed the presentation format and future work. Responding to the challenges of supporting open-ended constructionist learning of ABM & P and leveraging LLMs for educational purposes, we contribute to the field by proposing the first constructionist LLM-driven interface to support computational and complex systems thinking.

{{</citation>}}
