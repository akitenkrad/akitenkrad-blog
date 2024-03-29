---
draft: false
title: "arXiv @ 2023.12.10"
date: 2023-12-10
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.10"
    identifier: arxiv_20231210
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (25)](#cscv-25)
- [cs.LG (13)](#cslg-13)
- [cs.CL (21)](#cscl-21)
- [cs.IT (2)](#csit-2)
- [cs.AI (11)](#csai-11)
- [cs.DC (2)](#csdc-2)
- [cs.SE (5)](#csse-5)
- [cs.CY (2)](#cscy-2)
- [eess.IV (4)](#eessiv-4)
- [eess.SY (4)](#eesssy-4)
- [cs.RO (4)](#csro-4)
- [cs.CC (1)](#cscc-1)
- [cs.NI (3)](#csni-3)
- [cs.CR (6)](#cscr-6)
- [cs.IR (2)](#csir-2)
- [cs.HC (1)](#cshc-1)
- [cs.ET (1)](#cset-1)
- [cs.MA (1)](#csma-1)
- [cs.GT (1)](#csgt-1)
- [hep-ex (1)](#hep-ex-1)

## cs.CV (25)



### (1/110) Active Learning Guided Federated Online Adaptation: Applications in Medical Image Segmentation (Md Shazid Islam et al., 2023)

{{<citation>}}

Md Shazid Islam, Sayak Nag, Arindam Dutta, Miraj Ahmed, Fahim Faisal Niloy, Amit K. Roy-Chowdhury. (2023)  
**Active Learning Guided Federated Online Adaptation: Applications in Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2312.05407v1)  

---


**ABSTRACT**  
Data privacy, storage, and distribution shifts are major bottlenecks in medical image analysis. Data cannot be shared across patients, physicians, and facilities due to privacy concerns, usually requiring each patient's data to be analyzed in a discreet setting at a near real-time pace. However, one would like to take advantage of the accumulated knowledge across healthcare facilities as the computational systems analyze data of more and more patients while incorporating feedback provided by physicians to improve accuracy. Motivated by these, we propose a method for medical image segmentation that adapts to each incoming data batch (online adaptation), incorporates physician feedback through active learning, and assimilates knowledge across facilities in a federated setup. Combining an online adaptation scheme at test time with an efficient sampling strategy with budgeted annotation helps bridge the gap between the source and the incoming stream of target domain data. A federated setup allows collaborative aggregation of knowledge across distinct distributed models without needing to share the data across different models. This facilitates the improvement of performance over time by accumulating knowledge across users. Towards achieving these goals, we propose a computationally amicable, privacy-preserving image segmentation technique \textbf{DrFRODA} that uses federated learning to adapt the model in an online manner with feedback from doctors in the loop. Our experiments on publicly available datasets show that the proposed distributed active learning-based online adaptation method outperforms unsupervised online adaptation methods and shows competitive results with offline active learning-based adaptation methods.

{{</citation>}}


### (2/110) Loss Functions in the Era of Semantic Segmentation: A Survey and Outlook (Reza Azad et al., 2023)

{{<citation>}}

Reza Azad, Moein Heidary, Kadir Yilmaz, Michael Hüttemann, Sanaz Karimijafarbigloo, Yuli Wu, Anke Schmeink, Dorit Merhof. (2023)  
**Loss Functions in the Era of Semantic Segmentation: A Survey and Outlook**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.05391v1)  

---


**ABSTRACT**  
Semantic image segmentation, the process of classifying each pixel in an image into a particular class, plays an important role in many visual understanding systems. As the predominant criterion for evaluating the performance of statistical models, loss functions are crucial for shaping the development of deep learning-based segmentation algorithms and improving their overall performance. To aid researchers in identifying the optimal loss function for their particular application, this survey provides a comprehensive and unified review of $25$ loss functions utilized in image segmentation. We provide a novel taxonomy and thorough review of how these loss functions are customized and leveraged in image segmentation, with a systematic categorization emphasizing their significant features and applications. Furthermore, to evaluate the efficacy of these methods in real-world scenarios, we propose unbiased evaluations of some distinct and renowned loss functions on established medical and natural image datasets. We conclude this review by identifying current challenges and unveiling future research opportunities. Finally, we have compiled the reviewed studies that have open-source implementations on our GitHub page.

{{</citation>}}


### (3/110) NoiseCLR: A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Directions in Diffusion Models (Yusuf Dalva et al., 2023)

{{<citation>}}

Yusuf Dalva, Pinar Yanardag. (2023)  
**NoiseCLR: A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Directions in Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.05390v1)  

---


**ABSTRACT**  
Generative models have been very popular in the recent years for their image generation capabilities. GAN-based models are highly regarded for their disentangled latent space, which is a key feature contributing to their success in controlled image editing. On the other hand, diffusion models have emerged as powerful tools for generating high-quality images. However, the latent space of diffusion models is not as thoroughly explored or understood. Existing methods that aim to explore the latent space of diffusion models usually relies on text prompts to pinpoint specific semantics. However, this approach may be restrictive in areas such as art, fashion, or specialized fields like medicine, where suitable text prompts might not be available or easy to conceive thus limiting the scope of existing work. In this paper, we propose an unsupervised method to discover latent semantics in text-to-image diffusion models without relying on text prompts. Our method takes a small set of unlabeled images from specific domains, such as faces or cats, and a pre-trained diffusion model, and discovers diverse semantics in unsupervised fashion using a contrastive learning objective. Moreover, the learned directions can be applied simultaneously, either within the same domain (such as various types of facial edits) or across different domains (such as applying cat and face edits within the same image) without interfering with each other. Our extensive experiments show that our method achieves highly disentangled edits, outperforming existing approaches in both diffusion-based and GAN-based latent space editing methods.

{{</citation>}}


### (4/110) AI-driven Structure Detection and Information Extraction from Historical Cadastral Maps (Early 19th Century Franciscean Cadastre in the Province of Styria) and Current High-resolution Satellite and Aerial Imagery for Remote Sensing (Wolfgang Göderle et al., 2023)

{{<citation>}}

Wolfgang Göderle, Christian Macher, Katrin Mauthner, Oliver Pimas, Fabian Rampetsreiter. (2023)  
**AI-driven Structure Detection and Information Extraction from Historical Cadastral Maps (Early 19th Century Franciscean Cadastre in the Province of Styria) and Current High-resolution Satellite and Aerial Imagery for Remote Sensing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI, Information Extraction, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.07560v1)  

---


**ABSTRACT**  
Cadastres from the 19th century are a complex as well as rich source for historians and archaeologists, whose use presents them with great challenges. For archaeological and historical remote sensing, we have trained several Deep Learning models, CNNs as well as Vision Transformers, to extract large-scale data from this knowledge representation. We present the principle results of our work here and we present a the demonstrator of our browser-based tool that allows researchers and public stakeholders to quickly identify spots that featured buildings in the 19th century Franciscean Cadastre. The tool not only supports scholars and fellow researchers in building a better understanding of the settlement history of the region of Styria, it also helps public administration and fellow citizens to swiftly identify areas of heightened sensibility with regard to the cultural heritage of the region.

{{</citation>}}


### (5/110) PixLore: A Dataset-driven Approach to Rich Image Captioning (Diego Bonilla, 2023)

{{<citation>}}

Diego Bonilla. (2023)  
**PixLore: A Dataset-driven Approach to Rich Image Captioning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ChatGPT, Computer Vision, GPT, GPT-4, Google, Image Captioning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.05349v1)  

---


**ABSTRACT**  
In the domain of vision-language integration, generating detailed image captions poses a significant challenge due to the lack of a curated and rich dataset. This study introduces PixLore, a novel method that leverages Querying Transformers through the fine-tuning of the BLIP-2 model using the LoRa method on a standard commercial GPU. Our approach, which involves training on a carefully assembled dataset from state-of-the-art Computer Vision models combined and augmented by ChatGPT, addresses the question of whether intricate image understanding can be achieved with an ensemble of smaller-scale models. Comparative evaluations against major models such as GPT-4 and Google Bard demonstrate that PixLore-2.7B, despite having considerably fewer parameters, is rated higher than the existing State-of-the-Art models in over half of the assessments. This research not only presents a groundbreaking approach but also highlights the importance of well-curated datasets in enhancing the performance of smaller models.

{{</citation>}}


### (6/110) Reconstructing Hands in 3D with Transformers (Georgios Pavlakos et al., 2023)

{{<citation>}}

Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, Jitendra Malik. (2023)  
**Reconstructing Hands in 3D with Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.05251v1)  

---


**ABSTRACT**  
We present an approach that can reconstruct hands in 3D from monocular input. Our approach for Hand Mesh Recovery, HaMeR, follows a fully transformer-based architecture and can analyze hands with significantly increased accuracy and robustness compared to previous work. The key to HaMeR's success lies in scaling up both the data used for training and the capacity of the deep network for hand reconstruction. For training data, we combine multiple datasets that contain 2D or 3D hand annotations. For the deep model, we use a large scale Vision Transformer architecture. Our final model consistently outperforms the previous baselines on popular 3D hand pose benchmarks. To further evaluate the effect of our design in non-controlled settings, we annotate existing in-the-wild datasets with 2D hand keypoint annotations. On this newly collected dataset of annotations, HInt, we demonstrate significant improvements over existing baselines. We make our code, data and models available on the project website: https://geopavlakos.github.io/hamer/.

{{</citation>}}


### (7/110) Few-Shot Class-Incremental Learning via Training-Free Prototype Calibration (Qi-Wei Wang et al., 2023)

{{<citation>}}

Qi-Wei Wang, Da-Wei Zhou, Yi-Kai Zhang, De-Chuan Zhan, Han-Jia Ye. (2023)  
**Few-Shot Class-Incremental Learning via Training-Free Prototype Calibration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.05229v1)  

---


**ABSTRACT**  
Real-world scenarios are usually accompanied by continuously appearing classes with scare labeled samples, which require the machine learning model to incrementally learn new classes and maintain the knowledge of base classes. In this Few-Shot Class-Incremental Learning (FSCIL) scenario, existing methods either introduce extra learnable components or rely on a frozen feature extractor to mitigate catastrophic forgetting and overfitting problems. However, we find a tendency for existing methods to misclassify the samples of new classes into base classes, which leads to the poor performance of new classes. In other words, the strong discriminability of base classes distracts the classification of new classes. To figure out this intriguing phenomenon, we observe that although the feature extractor is only trained on base classes, it can surprisingly represent the semantic similarity between the base and unseen new classes. Building upon these analyses, we propose a simple yet effective Training-frEE calibratioN (TEEN) strategy to enhance the discriminability of new classes by fusing the new prototypes (i.e., mean features of a class) with weighted base prototypes. In addition to standard benchmarks in FSCIL, TEEN demonstrates remarkable performance and consistent improvements over baseline methods in the few-shot learning scenario. Code is available at: https://github.com/wangkiw/TEEN

{{</citation>}}


### (8/110) Noise Adaptor in Spiking Neural Networks (Chen Li et al., 2023)

{{<citation>}}

Chen Li, Bipin Rajendran. (2023)  
**Noise Adaptor in Spiking Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.05290v1)  

---


**ABSTRACT**  
Recent strides in low-latency spiking neural network (SNN) algorithms have drawn significant interest, particularly due to their event-driven computing nature and fast inference capability. One of the most efficient ways to construct a low-latency SNN is by converting a pre-trained, low-bit artificial neural network (ANN) into an SNN. However, this conversion process faces two main challenges: First, converting SNNs from low-bit ANNs can lead to ``occasional noise" -- the phenomenon where occasional spikes are generated in spiking neurons where they should not be -- during inference, which significantly lowers SNN accuracy. Second, although low-latency SNNs initially show fast improvements in accuracy with time steps, these accuracy growths soon plateau, resulting in their peak accuracy lagging behind both full-precision ANNs and traditional ``long-latency SNNs'' that prioritize precision over speed.   In response to these two challenges, this paper introduces a novel technique named ``noise adaptor.'' Noise adaptor can model occasional noise during training and implicitly optimize SNN accuracy, particularly at high simulation times $T$. Our research utilizes the ResNet model for a comprehensive analysis of the impact of the noise adaptor on low-latency SNNs. The results demonstrate that our method outperforms the previously reported quant-ANN-to-SNN conversion technique. We achieved an accuracy of 95.95\% within 4 time steps on CIFAR-10 using ResNet-18, and an accuracy of 74.37\% within 64 time steps on ImageNet using ResNet-50. Remarkably, these results were obtained without resorting to any noise correction methods during SNN inference, such as negative spikes or two-stage SNN simulations. Our approach significantly boosts the peak accuracy of low-latency SNNs, bringing them on par with the accuracy of full-precision ANNs. Code will be open source.

{{</citation>}}


### (9/110) An adversarial attack approach for eXplainable AI evaluation on deepfake detection models (Balachandar Gowrisankar et al., 2023)

{{<citation>}}

Balachandar Gowrisankar, Vrizlynn L. L. Thing. (2023)  
**An adversarial attack approach for eXplainable AI evaluation on deepfake detection models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.06627v1)  

---


**ABSTRACT**  
With the rising concern on model interpretability, the application of eXplainable AI (XAI) tools on deepfake detection models has been a topic of interest recently. In image classification tasks, XAI tools highlight pixels influencing the decision given by a model. This helps in troubleshooting the model and determining areas that may require further tuning of parameters. With a wide range of tools available in the market, choosing the right tool for a model becomes necessary as each one may highlight different sets of pixels for a given image. There is a need to evaluate different tools and decide the best performing ones among them. Generic XAI evaluation methods like insertion or removal of salient pixels/segments are applicable for general image classification tasks but may produce less meaningful results when applied on deepfake detection models due to their functionality. In this paper, we perform experiments to show that generic removal/insertion XAI evaluation methods are not suitable for deepfake detection models. We also propose and implement an XAI evaluation approach specifically suited for deepfake detection models.

{{</citation>}}


### (10/110) MuVieCAST: Multi-View Consistent Artistic Style Transfer (Nail Ibrahimli et al., 2023)

{{<citation>}}

Nail Ibrahimli, Julian F. P. Kooij, Liangliang Nan. (2023)  
**MuVieCAST: Multi-View Consistent Artistic Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2312.05046v1)  

---


**ABSTRACT**  
We introduce MuVieCAST, a modular multi-view consistent style transfer network architecture that enables consistent style transfer between multiple viewpoints of the same scene. This network architecture supports both sparse and dense views, making it versatile enough to handle a wide range of multi-view image datasets. The approach consists of three modules that perform specific tasks related to style transfer, namely content preservation, image transformation, and multi-view consistency enforcement. We extensively evaluate our approach across multiple application domains including depth-map-based point cloud fusion, mesh reconstruction, and novel-view synthesis. Our experiments reveal that the proposed framework achieves an exceptional generation of stylized images, exhibiting consistent outcomes across perspectives. A user study focusing on novel-view synthesis further confirms these results, with approximately 68\% of cases participants expressing a preference for our generated outputs compared to the recent state-of-the-art method. Our modular framework is extensible and can easily be integrated with various backbone architectures, making it a flexible solution for multi-view style transfer. More results are demonstrated on our project page: muviecast.github.io

{{</citation>}}


### (11/110) Synthesizing Traffic Datasets using Graph Neural Networks (Daniel Rodriguez-Criado et al., 2023)

{{<citation>}}

Daniel Rodriguez-Criado, Maria Chli, Luis J. Manso, George Vogiatzis. (2023)  
**Synthesizing Traffic Datasets using Graph Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.05031v1)  

---


**ABSTRACT**  
Traffic congestion in urban areas presents significant challenges, and Intelligent Transportation Systems (ITS) have sought to address these via automated and adaptive controls. However, these systems often struggle to transfer simulated experiences to real-world scenarios. This paper introduces a novel methodology for bridging this `sim-real' gap by creating photorealistic images from 2D traffic simulations and recorded junction footage. We propose a novel image generation approach, integrating a Conditional Generative Adversarial Network with a Graph Neural Network (GNN) to facilitate the creation of realistic urban traffic images. We harness GNNs' ability to process information at different levels of abstraction alongside segmented images for preserving locality data. The presented architecture leverages the power of SPADE and Graph ATtention (GAT) network models to create images based on simulated traffic scenarios. These images are conditioned by factors such as entity positions, colors, and time of day. The uniqueness of our approach lies in its ability to effectively translate structured and human-readable conditions, encoded as graphs, into realistic images. This advancement contributes to applications requiring rich traffic image datasets, from data augmentation to urban traffic solutions. We further provide an application to test the model's capabilities, including generating images with manually defined positions for various entities.

{{</citation>}}


### (12/110) MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness (Xiaoyun Xu et al., 2023)

{{<citation>}}

Xiaoyun Xu, Shujian Yu, Jingzheng Wu, Stjepan Picek. (2023)  
**MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.04960v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) achieve superior performance on various tasks compared to convolutional neural networks (CNNs), but ViTs are also vulnerable to adversarial attacks. Adversarial training is one of the most successful methods to build robust CNN models. Thus, recent works explored new methodologies for adversarial training of ViTs based on the differences between ViTs and CNNs, such as better training strategies, preventing attention from focusing on a single block, or discarding low-attention embeddings. However, these methods still follow the design of traditional supervised adversarial training, limiting the potential of adversarial training on ViTs. This paper proposes a novel defense method, MIMIR, which aims to build a different adversarial training methodology by utilizing Masked Image Modeling at pre-training. We create an autoencoder that accepts adversarial examples as input but takes the clean examples as the modeling target. Then, we create a mutual information (MI) penalty following the idea of the Information Bottleneck. Among the two information source inputs and corresponding adversarial perturbation, the perturbation information is eliminated due to the constraint of the modeling target. Next, we provide a theoretical analysis of MIMIR using the bounds of the MI penalty. We also design two adaptive attacks when the adversary is aware of the MIMIR defense and show that MIMIR still performs well. The experimental results show that MIMIR improves (natural and adversarial) accuracy on average by 4.19\% on CIFAR-10 and 5.52\% on ImageNet-1K, compared to baselines. On Tiny-ImageNet, we obtained improved natural accuracy of 2.99\% on average and comparable adversarial accuracy. Our code and trained models are publicly available\footnote{\url{https://anonymous.4open.science/r/MIMIR-5444/README.md}}.

{{</citation>}}


### (13/110) Retrieval-based Video Language Model for Efficient Long Video Question Answering (Jiaqi Xu et al., 2023)

{{<citation>}}

Jiaqi Xu, Cuiling Lan, Wenxuan Xie, Xuejin Chen, Yan Lu. (2023)  
**Retrieval-based Video Language Model for Efficient Long Video Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.04931v1)  

---


**ABSTRACT**  
The remarkable natural language understanding, reasoning, and generation capabilities of large language models (LLMs) have made them attractive for application to video question answering (Video QA) tasks, utilizing video tokens as contextual input. However, employing LLMs for long video understanding presents significant challenges and remains under-explored. The extensive number of video tokens leads to considerable computational costs for LLMs while using aggregated tokens results in loss of vision details. Moreover, the presence of abundant question-irrelevant tokens introduces noise to the video QA process. To address these issues, we introduce a simple yet effective retrieval-based video language model (R-VLM) for efficient and interpretable long video QA. Specifically, given a question (query) and a long video, our model identifies and selects the most relevant $K$ video chunks and uses their associated visual tokens to serve as context for the LLM inference. This effectively reduces the number of video tokens, eliminates noise interference, and enhances system performance. Our experimental results validate the effectiveness of our framework for comprehending long videos. Furthermore, based on the retrieved chunks, our model is interpretable that provides the justifications on where we get the answers.

{{</citation>}}


### (14/110) Accelerating Convolutional Neural Network Pruning via Spatial Aura Entropy (Bogdan Musat et al., 2023)

{{<citation>}}

Bogdan Musat, Razvan Andonie. (2023)  
**Accelerating Convolutional Neural Network Pruning via Spatial Aura Entropy**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2312.04926v1)  

---


**ABSTRACT**  
In recent years, pruning has emerged as a popular technique to reduce the computational complexity and memory footprint of Convolutional Neural Network (CNN) models. Mutual Information (MI) has been widely used as a criterion for identifying unimportant filters to prune. However, existing methods for MI computation suffer from high computational cost and sensitivity to noise, leading to suboptimal pruning performance. We propose a novel method to improve MI computation for CNN pruning, using the spatial aura entropy. The spatial aura entropy is useful for evaluating the heterogeneity in the distribution of the neural activations over a neighborhood, providing information about local features. Our method effectively improves the MI computation for CNN pruning, leading to more robust and efficient pruning. Experimental results on the CIFAR-10 benchmark dataset demonstrate the superiority of our approach in terms of pruning performance and computational efficiency.

{{</citation>}}


### (15/110) SA-Attack: Improving Adversarial Transferability of Vision-Language Pre-training Models via Self-Augmentation (Bangyan He et al., 2023)

{{<citation>}}

Bangyan He, Xiaojun Jia, Siyuan Liang, Tianrui Lou, Yang Liu, Xiaochun Cao. (2023)  
**SA-Attack: Improving Adversarial Transferability of Vision-Language Pre-training Models via Self-Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2312.04913v1)  

---


**ABSTRACT**  
Current Visual-Language Pre-training (VLP) models are vulnerable to adversarial examples. These adversarial examples present substantial security risks to VLP models, as they can leverage inherent weaknesses in the models, resulting in incorrect predictions. In contrast to white-box adversarial attacks, transfer attacks (where the adversary crafts adversarial examples on a white-box model to fool another black-box model) are more reflective of real-world scenarios, thus making them more meaningful for research. By summarizing and analyzing existing research, we identified two factors that can influence the efficacy of transfer attacks on VLP models: inter-modal interaction and data diversity. Based on these insights, we propose a self-augment-based transfer attack method, termed SA-Attack. Specifically, during the generation of adversarial images and adversarial texts, we apply different data augmentation methods to the image modality and text modality, respectively, with the aim of improving the adversarial transferability of the generated adversarial images and texts. Experiments conducted on the FLickr30K and COCO datasets have validated the effectiveness of our method. Our code will be available after this paper is accepted.

{{</citation>}}


### (16/110) Cross-BERT for Point Cloud Pretraining (Xin Li et al., 2023)

{{<citation>}}

Xin Li, Peng Li, Zeyong Wei, Zhe Zhu, Mingqiang Wei, Junhui Hou, Liangliang Nan, Jing Qin, Haoran Xie, Fu Lee Wang. (2023)  
**Cross-BERT for Point Cloud Pretraining**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.04891v1)  

---


**ABSTRACT**  
Introducing BERT into cross-modal settings raises difficulties in its optimization for handling multiple modalities. Both the BERT architecture and training objective need to be adapted to incorporate and model information from different modalities. In this paper, we address these challenges by exploring the implicit semantic and geometric correlations between 2D and 3D data of the same objects/scenes. We propose a new cross-modal BERT-style self-supervised learning paradigm, called Cross-BERT. To facilitate pretraining for irregular and sparse point clouds, we design two self-supervised tasks to boost cross-modal interaction. The first task, referred to as Point-Image Alignment, aims to align features between unimodal and cross-modal representations to capture the correspondences between the 2D and 3D modalities. The second task, termed Masked Cross-modal Modeling, further improves mask modeling of BERT by incorporating high-dimensional semantic information obtained by cross-modal interaction. By performing cross-modal interaction, Cross-BERT can smoothly reconstruct the masked tokens during pretraining, leading to notable performance enhancements for downstream tasks. Through empirical evaluation, we demonstrate that Cross-BERT outperforms existing state-of-the-art methods in 3D downstream applications. Our work highlights the effectiveness of leveraging cross-modal 2D knowledge to strengthen 3D point cloud representation and the transferable capability of BERT across modalities.

{{</citation>}}


### (17/110) Interpretable Underwater Diver Gesture Recognition (Sudeep Mangalvedhekar et al., 2023)

{{<citation>}}

Sudeep Mangalvedhekar, Shreyas Nahar, Sudarshan Maskare, Kaushal Mahajan, Dr. Anant Bagade. (2023)  
**Interpretable Underwater Diver Gesture Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04874v1)  

---


**ABSTRACT**  
In recent years, usage and applications of Autonomous Underwater Vehicles has grown rapidly. Interaction of divers with the AUVs remains an integral part of the usage of AUVs for various applications and makes building robust and efficient underwater gesture recognition systems extremely important. In this paper, we propose an Underwater Gesture Recognition system trained on the Cognitive Autonomous Diving Buddy Underwater gesture dataset using deep learning that achieves 98.01\% accuracy on the dataset, which to the best of our knowledge is the best performance achieved on this dataset at the time of writing this paper. We also improve the Gesture Recognition System Interpretability by using XAI techniques to visualize the model's predictions.

{{</citation>}}


### (18/110) Adapting Vision Transformer for Efficient Change Detection (Yang Zhao et al., 2023)

{{<citation>}}

Yang Zhao, Yuxiang Zhang, Yanni Dong, Bo Du. (2023)  
**Adapting Vision Transformer for Efficient Change Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.04869v1)  

---


**ABSTRACT**  
Most change detection models based on vision transformers currently follow a "pretraining then fine-tuning" strategy. This involves initializing the model weights using large scale classification datasets, which can be either natural images or remote sensing images. However, fully tuning such a model requires significant time and resources. In this paper, we propose an efficient tuning approach that involves freezing the parameters of the pretrained image encoder and introducing additional training parameters. Through this approach, we have achieved competitive or even better results while maintaining extremely low resource consumption across six change detection benchmarks. For example, training time on LEVIR-CD, a change detection benchmark, is only half an hour with 9 GB memory usage, which could be very convenient for most researchers. Additionally, the decoupled tuning framework can be extended to any pretrained model for semantic change detection and multi temporal change detection as well. We hope that our proposed approach will serve as a part of foundational model to inspire more unified training approaches on change detection in the future.

{{</citation>}}


### (19/110) Learning Generalizable Perceptual Representations for Data-Efficient No-Reference Image Quality Assessment (Suhas Srinath et al., 2023)

{{<citation>}}

Suhas Srinath, Shankhanil Mitra, Shika Rao, Rajiv Soundararajan. (2023)  
**Learning Generalizable Perceptual Representations for Data-Efficient No-Reference Image Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.04838v1)  

---


**ABSTRACT**  
No-reference (NR) image quality assessment (IQA) is an important tool in enhancing the user experience in diverse visual applications. A major drawback of state-of-the-art NR-IQA techniques is their reliance on a large number of human annotations to train models for a target IQA application. To mitigate this requirement, there is a need for unsupervised learning of generalizable quality representations that capture diverse distortions. We enable the learning of low-level quality features agnostic to distortion types by introducing a novel quality-aware contrastive loss. Further, we leverage the generalizability of vision-language models by fine-tuning one such model to extract high-level image quality information through relevant text prompts. The two sets of features are combined to effectively predict quality by training a simple regressor with very few samples on a target dataset. Additionally, we design zero-shot quality predictions from both pathways in a completely blind setting. Our experiments on diverse datasets encompassing various distortions show the generalizability of the features and their superior performance in the data-efficient and zero-shot settings. Code will be made available at https://github.com/suhas-srinath/GRepQ.

{{</citation>}}


### (20/110) SiCP: Simultaneous Individual and Cooperative Perception for 3D Object Detection in Connected and Automated Vehicles (Deyuan Qu et al., 2023)

{{<citation>}}

Deyuan Qu, Qi Chen, Tianyu Bai, Andy Qin, Hongsheng Lu, Heng Fan, Song Fu, Qing Yang. (2023)  
**SiCP: Simultaneous Individual and Cooperative Perception for 3D Object Detection in Connected and Automated Vehicles**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.04822v1)  

---


**ABSTRACT**  
Cooperative perception for connected and automated vehicles is traditionally achieved through the fusion of feature maps from two or more vehicles. However, the absence of feature maps shared from other vehicles can lead to a significant decline in object detection performance for cooperative perception models compared to standalone 3D detection models. This drawback impedes the adoption of cooperative perception as vehicle resources are often insufficient to concurrently employ two perception models. To tackle this issue, we present Simultaneous Individual and Cooperative Perception (SiCP), a generic framework that supports a wide range of the state-of-the-art standalone perception backbones and enhances them with a novel Dual-Perception Network (DP-Net) designed to facilitate both individual and cooperative perception. In addition to its lightweight nature with only 0.13M parameters, DP-Net is robust and retains crucial gradient information during feature map fusion. As demonstrated in a comprehensive evaluation on the OPV2V dataset, thanks to DP-Net, SiCP surpasses state-of-the-art cooperative perception solutions while preserving the performance of standalone perception solutions.

{{</citation>}}


### (21/110) MoVQA: A Benchmark of Versatile Question-Answering for Long-Form Movie Understanding (Hongjie Zhang et al., 2023)

{{<citation>}}

Hongjie Zhang, Yi Liu, Lu Dong, Yifei Huang, Zhen-Hua Ling, Yali Wang, Limin Wang, Yu Qiao. (2023)  
**MoVQA: A Benchmark of Versatile Question-Answering for Long-Form Movie Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.04817v1)  

---


**ABSTRACT**  
While several long-form VideoQA datasets have been introduced, the length of both videos used to curate questions and sub-clips of clues leveraged to answer those questions have not yet reached the criteria for genuine long-form video understanding. Moreover, their QAs are unduly narrow and modality-biased, lacking a wider view of understanding long-term video content with rich dynamics and complex narratives. To remedy this, we introduce MoVQA, a long-form movie question-answering dataset, and benchmark to assess the diverse cognitive capabilities of multimodal systems rely on multi-level temporal lengths, with considering both video length and clue length. Additionally, to take a step towards human-level understanding in long-form video, versatile and multimodal question-answering is designed from the moviegoer-perspective to assess the model capabilities on various perceptual and cognitive axes.Through analysis involving various baselines reveals a consistent trend: the performance of all methods significantly deteriorate with increasing video and clue length. Meanwhile, our established baseline method has shown some improvements, but there is still ample scope for enhancement on our challenging MoVQA dataset. We expect our MoVQA to provide a new perspective and encourage inspiring works on long-form video understanding research.

{{</citation>}}


### (22/110) DARNet: Bridging Domain Gaps in Cross-Domain Few-Shot Segmentation with Dynamic Adaptation (Haoran Fan et al., 2023)

{{<citation>}}

Haoran Fan, Qi Fan, Maurice Pagnucco, Yang Song. (2023)  
**DARNet: Bridging Domain Gaps in Cross-Domain Few-Shot Segmentation with Dynamic Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.04813v1)  

---


**ABSTRACT**  
Few-shot segmentation (FSS) aims to segment novel classes in a query image by using only a small number of supporting images from base classes. However, in cross-domain few-shot segmentation (CD-FSS), leveraging features from label-rich domains for resource-constrained domains poses challenges due to domain discrepancies. This work presents a Dynamically Adaptive Refine (DARNet) method that aims to balance generalization and specificity for CD-FSS. Our method includes the Channel Statistics Disruption (CSD) strategy, which perturbs feature channel statistics in the source domain, bolstering generalization to unknown target domains. Moreover, recognizing the variability across target domains, an Adaptive Refine Self-Matching (ARSM) method is also proposed to adjust the matching threshold and dynamically refine the prediction result with the self-matching method, enhancing accuracy. We also present a Test-Time Adaptation (TTA) method to refine the model's adaptability to diverse feature distributions. Our approach demonstrates superior performance against state-of-the-art methods in CD-FSS tasks.

{{</citation>}}


### (23/110) MimicDiffusion: Purifying Adversarial Perturbation via Mimicking Clean Diffusion Model (Kaiyu Song et al., 2023)

{{<citation>}}

Kaiyu Song, Hanjiang Lai. (2023)  
**MimicDiffusion: Purifying Adversarial Perturbation via Mimicking Clean Diffusion Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.04802v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) are vulnerable to adversarial perturbation, where an imperceptible perturbation is added to the image that can fool the DNNs. Diffusion-based adversarial purification focuses on using the diffusion model to generate a clean image against such adversarial attacks. Unfortunately, the generative process of the diffusion model is also inevitably affected by adversarial perturbation since the diffusion model is also a deep network where its input has adversarial perturbation. In this work, we propose MimicDiffusion, a new diffusion-based adversarial purification technique, that directly approximates the generative process of the diffusion model with the clean image as input. Concretely, we analyze the differences between the guided terms using the clean image and the adversarial sample. After that, we first implement MimicDiffusion based on Manhattan distance. Then, we propose two guidance to purify the adversarial perturbation and approximate the clean diffusion model. Extensive experiments on three image datasets including CIFAR-10, CIFAR-100, and ImageNet with three classifier backbones including WideResNet-70-16, WideResNet-28-10, and ResNet50 demonstrate that MimicDiffusion significantly performs better than the state-of-the-art baselines. On CIFAR-10, CIFAR-100, and ImageNet, it achieves 92.67\%, 61.35\%, and 61.53\% average robust accuracy, which are 18.49\%, 13.23\%, and 17.64\% higher, respectively. The code is available in the supplementary material.

{{</citation>}}


### (24/110) User-Aware Prefix-Tuning is a Good Learner for Personalized Image Captioning (Xuan Wang et al., 2023)

{{<citation>}}

Xuan Wang, Guanhong Wang, Wenhao Chai, Jiayu Zhou, Gaoang Wang. (2023)  
**User-Aware Prefix-Tuning is a Good Learner for Personalized Image Captioning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BLEU, GPT, Image Captioning  
[Paper Link](http://arxiv.org/abs/2312.04793v1)  

---


**ABSTRACT**  
Image captioning bridges the gap between vision and language by automatically generating natural language descriptions for images. Traditional image captioning methods often overlook the preferences and characteristics of users. Personalized image captioning solves this problem by incorporating user prior knowledge into the model, such as writing styles and preferred vocabularies. Most existing methods emphasize the user context fusion process by memory networks or transformers. However, these methods ignore the distinct domains of each dataset. Therefore, they need to update the entire caption model parameters when meeting new samples, which is time-consuming and calculation-intensive. To address this challenge, we propose a novel personalized image captioning framework that leverages user context to consider personality factors. Additionally, our framework utilizes the prefix-tuning paradigm to extract knowledge from a frozen large language model, reducing the gap between different language domains. Specifically, we employ CLIP to extract the visual features of an image and align the semantic space using a query-guided mapping network. By incorporating the transformer layer, we merge the visual features with the user's contextual prior knowledge to generate informative prefixes. Moreover, we employ GPT-2 as the frozen large language model. With a small number of parameters to be trained, our model performs efficiently and effectively. Our model outperforms existing baseline models on Instagram and YFCC100M datasets across five evaluation metrics, demonstrating its superiority, including twofold improvements in metrics such as BLEU-4 and CIDEr.

{{</citation>}}


### (25/110) Fine-Tuning InstructPix2Pix for Advanced Image Colorization (Zifeng An et al., 2023)

{{<citation>}}

Zifeng An, Zijing Xu, Eric Fan, Qi Cao. (2023)  
**Fine-Tuning InstructPix2Pix for Advanced Image Colorization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.04780v1)  

---


**ABSTRACT**  
This paper presents a novel approach to human image colorization by fine-tuning the InstructPix2Pix model, which integrates a language model (GPT-3) with a text-to-image model (Stable Diffusion). Despite the original InstructPix2Pix model's proficiency in editing images based on textual instructions, it exhibits limitations in the focused domain of colorization. To address this, we fine-tuned the model using the IMDB-WIKI dataset, pairing black-and-white images with a diverse set of colorization prompts generated by ChatGPT. This paper contributes by (1) applying fine-tuning techniques to stable diffusion models specifically for colorization tasks, and (2) employing generative models to create varied conditioning prompts. After finetuning, our model outperforms the original InstructPix2Pix model on multiple metrics quantitatively, and we produce more realistically colored images qualitatively. The code for this project is provided on the GitHub Repository https://github.com/AllenAnZifeng/DeepLearning282.

{{</citation>}}


## cs.LG (13)



### (26/110) Disentangled Latent Representation Learning for Tackling the Confounding M-Bias Problem in Causal Inference (Debo Cheng et al., 2023)

{{<citation>}}

Debo Cheng, Yang Xie, Ziqi Xu, Jiuyong Li, Lin Liu, Jixue Liu, Yinghao Zhang, Zaiwen Feng. (2023)  
**Disentangled Latent Representation Learning for Tackling the Confounding M-Bias Problem in Causal Inference**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ME  
Keywords: Bias, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.05404v1)  

---


**ABSTRACT**  
In causal inference, it is a fundamental task to estimate the causal effect from observational data. However, latent confounders pose major challenges in causal inference in observational data, for example, confounding bias and M-bias. Recent data-driven causal effect estimators tackle the confounding bias problem via balanced representation learning, but assume no M-bias in the system, thus they fail to handle the M-bias. In this paper, we identify a challenging and unsolved problem caused by a variable that leads to confounding bias and M-bias simultaneously. To address this problem with co-occurring M-bias and confounding bias, we propose a novel Disentangled Latent Representation learning framework for learning latent representations from proxy variables for unbiased Causal effect Estimation (DLRCE) from observational data. Specifically, DLRCE learns three sets of latent representations from the measured proxy variables to adjust for the confounding bias and M-bias. Extensive experiments on both synthetic and three real-world datasets demonstrate that DLRCE significantly outperforms the state-of-the-art estimators in the case of the presence of both confounding bias and M-bias.

{{</citation>}}


### (27/110) Cross Domain Generative Augmentation: Domain Generalization with Latent Diffusion Models (Sobhan Hemati et al., 2023)

{{<citation>}}

Sobhan Hemati, Mahdi Beitollahi, Amir Hossein Estiri, Bassel Al Omari, Xi Chen, Guojun Zhang. (2023)  
**Cross Domain Generative Augmentation: Domain Generalization with Latent Diffusion Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2312.05387v1)  

---


**ABSTRACT**  
Despite the huge effort in developing novel regularizers for Domain Generalization (DG), adding simple data augmentation to the vanilla ERM which is a practical implementation of the Vicinal Risk Minimization principle (VRM) \citep{chapelle2000vicinal} outperforms or stays competitive with many of the proposed regularizers. The VRM reduces the estimation error in ERM by replacing the point-wise kernel estimates with a more precise estimation of true data distribution that reduces the gap between data points \textbf{within each domain}. However, in the DG setting, the estimation error of true data distribution by ERM is mainly caused by the distribution shift \textbf{between domains} which cannot be fully addressed by simple data augmentation techniques within each domain. Inspired by this limitation of VRM, we propose a novel data augmentation named Cross Domain Generative Augmentation (CDGA) that replaces the pointwise kernel estimates in ERM with new density estimates in the \textbf{vicinity of domain pairs} so that the gap between domains is further reduced. To this end, CDGA, which is built upon latent diffusion models (LDM), generates synthetic images to fill the gap between all domains and as a result, reduces the non-iidness. We show that CDGA outperforms SOTA DG methods under the Domainbed benchmark. To explain the effectiveness of CDGA, we generate more than 5 Million synthetic images and perform extensive ablation studies including data scaling laws, distribution visualization, domain shift quantification, adversarial robustness, and loss landscape analysis.

{{</citation>}}


### (28/110) AI Competitions and Benchmarks: The life cycle of challenges and benchmarks (Gustavo Stolovitzky et al., 2023)

{{<citation>}}

Gustavo Stolovitzky, Julio Saez-Rodriguez, Julie Bletz, Jacob Albrecht, Gaia Andreoletti, James C. Costello, Paul Boutros. (2023)  
**AI Competitions and Benchmarks: The life cycle of challenges and benchmarks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.05296v1)  

---


**ABSTRACT**  
Data Science research is undergoing a revolution fueled by the transformative power of technology, the Internet, and an ever increasing computational capacity. The rate at which sophisticated algorithms can be developed is unprecedented, yet they remain outpaced by the massive amounts of data that are increasingly available to researchers. Here we argue for the need to creatively leverage the scientific research and algorithm development community as an axis of robust innovation. Engaging these communities in the scientific discovery enterprise by critical assessments, community experiments, and/or crowdsourcing will multiply opportunities to develop new data driven, reproducible and well benchmarked algorithmic solutions to fundamental and applied problems of current interest. Coordinated community engagement in the analysis of highly complex and massive data has emerged as one approach to find robust methodologies that best address these challenges. When community engagement is done in the form of competitions, also known as challenges, the validation of the analytical methodology is inherently addressed, establishing performance benchmarks. Finally, challenges foster open innovation across multiple disciplines to create communities that collaborate directly or indirectly to address significant scientific gaps. Together, participants can solve important problems as varied as health research, climate change, and social equity. Ultimately, challenges can catalyze and accelerate the synthesis of complex data into knowledge or actionable information, and should be viewed a powerful tool to make lasting social and research contributions.

{{</citation>}}


### (29/110) Modeling Risk in Reinforcement Learning: A Literature Mapping (Leonardo Villalobos-Arias et al., 2023)

{{<citation>}}

Leonardo Villalobos-Arias, Derek Martin, Abhijeet Krishnan, Madeleine Gagné, Colin M. Potts, Arnav Jhala. (2023)  
**Modeling Risk in Reinforcement Learning: A Literature Mapping**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05231v1)  

---


**ABSTRACT**  
Safe reinforcement learning deals with mitigating or avoiding unsafe situations by reinforcement learning (RL) agents. Safe RL approaches are based on specific risk representations for particular problems or domains. In order to analyze agent behaviors, compare safe RL approaches, and effectively transfer techniques between application domains, it is necessary to understand the types of risk specific to safe RL problems. We performed a systematic literature mapping with the objective to characterize risk in safe RL. Based on the obtained results, we present definitions, characteristics, and types of risk that hold on multiple application domains. Our literature mapping covers literature from the last 5 years (2017-2022), from a variety of knowledge areas (AI, finance, engineering, medicine) where RL approaches emphasize risk representation and management. Our mapping covers 72 papers filtered systematically from over thousands of papers on the topic. Our proposed notion of risk covers a variety of representations, disciplinary differences, common training exercises, and types of techniques. We encourage researchers to include explicit and detailed accounts of risk in future safe RL research reports, using this mapping as a starting point. With this information, researchers and practitioners could draw stronger conclusions on the effectiveness of techniques on different problems.

{{</citation>}}


### (30/110) AI Competitions and Benchmarks: Competition platforms (Andrey Ustyuzhanin et al., 2023)

{{<citation>}}

Andrey Ustyuzhanin, Harald Carlens. (2023)  
**AI Competitions and Benchmarks: Competition platforms**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.05185v1)  

---


**ABSTRACT**  
The ecosystem of artificial intelligence competitions is a diverse and multifaceted landscape, encompassing a variety of platforms that each host numerous competitions annually, alongside a plethora of specialized websites dedicated to singular contests. These platforms adeptly manage the overarching administrative responsibilities inherent in orchestrating competitions, thus affording organizers the liberty to allocate greater attention to other facets of their contests. Notably, these platforms exhibit considerable diversity in their operational functionalities, economic models, and community dynamics. This chapter conducts an extensive review of the foremost services in this realm and elucidates several alternative methodologies that facilitate the independent hosting of such challenges. Keywords: competition platform, challenge hosting services, comparison.

{{</citation>}}


### (31/110) SparQ Attention: Bandwidth-Efficient LLM Inference (Luka Ribar et al., 2023)

{{<citation>}}

Luka Ribar, Ivan Chelombiev, Luke Hudlass-Galley, Charlie Blake, Carlo Luschi, Douglas Orr. (2023)  
**SparQ Attention: Bandwidth-Efficient LLM Inference**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.04985v1)  

---


**ABSTRACT**  
Generative large language models (LLMs) have opened up numerous novel possibilities, but due to their significant computational requirements their ubiquitous use remains challenging. Some of the most useful applications require processing large numbers of samples at a time and using long contexts, both significantly increasing the memory communication load of the models. We introduce SparQ Attention, a technique for increasing the inference throughput of LLMs by reducing the memory bandwidth requirements within the attention blocks through selective fetching of the cached history. Our proposed technique can be applied directly to off-the-shelf LLMs during inference, without requiring any modification to the pre-training setup or additional fine-tuning. We show how SparQ Attention can decrease the attention memory bandwidth requirements up to eight times without any loss in accuracy by evaluating Llama 2 and Pythia models on a wide range of downstream tasks.

{{</citation>}}


### (32/110) Pruning Convolutional Filters via Reinforcement Learning with Entropy Minimization (Bogdan Musat et al., 2023)

{{<citation>}}

Bogdan Musat, Razvan Andonie. (2023)  
**Pruning Convolutional Filters via Reinforcement Learning with Entropy Minimization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Pruning, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.04918v1)  

---


**ABSTRACT**  
Structural pruning has become an integral part of neural network optimization, used to achieve architectural configurations which can be deployed and run more efficiently on embedded devices. Previous results showed that pruning is possible with minimum performance loss by utilizing a reinforcement learning agent which makes decisions about the sparsity level of each neural layer by maximizing as a reward the accuracy of the network. We introduce a novel information-theoretic reward function which minimizes the spatial entropy of convolutional activations. This minimization ultimately acts as a proxy for maintaining accuracy, although these two criteria are not related in any way. Our method shows that there is another possibility to preserve accuracy without the need to directly optimize it in the agent's reward function. In our experiments, we were able to reduce the total number of FLOPS of multiple popular neural network architectures by 5-10x, incurring minimal or no performance drop and being on par with the solution found by maximizing the accuracy.

{{</citation>}}


### (33/110) EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language Models with 3D Parallelism (Yanxi Chen et al., 2023)

{{<citation>}}

Yanxi Chen, Xuchen Pan, Yaliang Li, Bolin Ding, Jingren Zhou. (2023)  
**EE-LLM: Large-Scale Training and Inference of Early-Exit Large Language Models with 3D Parallelism**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.04916v1)  

---


**ABSTRACT**  
We present EE-LLM, a framework for large-scale training and inference of early-exit large language models (LLMs). While recent works have shown preliminary evidence for the efficacy of early exiting in accelerating LLM inference, EE-LLM makes a foundational step towards scaling up early-exit LLMs by supporting their training and inference with massive 3D parallelism. Built upon Megatron-LM, EE-LLM implements a variety of algorithmic innovations and performance optimizations tailored to early exiting, including a lightweight method that facilitates backpropagation for the early-exit training objective with pipeline parallelism, techniques of leveraging idle resources in the original pipeline schedule for computation related to early-exit layers, and two approaches of early-exit inference that are compatible with KV caching for autoregressive generation. Our analytical and empirical study shows that EE-LLM achieves great training efficiency with negligible computational overhead compared to standard LLM training, as well as outstanding inference speedup without compromising output quality. To facilitate further research and adoption, we release EE-LLM at https://github.com/pan-x-c/EE-LLM.

{{</citation>}}


### (34/110) Understanding Community Bias Amplification in Graph Representation Learning (Shengzhong Zhang et al., 2023)

{{<citation>}}

Shengzhong Zhang, Wenjie Yang, Yimin Zhang, Hongwei Zhang, Divin Yan, Zengfeng Huang. (2023)  
**Understanding Community Bias Amplification in Graph Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Bias, Contrastive Learning, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.04883v1)  

---


**ABSTRACT**  
In this work, we discover a phenomenon of community bias amplification in graph representation learning, which refers to the exacerbation of performance bias between different classes by graph representation learning. We conduct an in-depth theoretical study of this phenomenon from a novel spectral perspective. Our analysis suggests that structural bias between communities results in varying local convergence speeds for node embeddings. This phenomenon leads to bias amplification in the classification results of downstream tasks. Based on the theoretical insights, we propose random graph coarsening, which is proved to be effective in dealing with the above issue. Finally, we propose a novel graph contrastive learning model called Random Graph Coarsening Contrastive Learning (RGCCL), which utilizes random coarsening as data augmentation and mitigates community bias by contrasting the coarsened graph with the original graph. Extensive experiments on various datasets demonstrate the advantage of our method when dealing with community bias amplification.

{{</citation>}}


### (35/110) HC-Ref: Hierarchical Constrained Refinement for Robust Adversarial Training of GNNs (Xiaobing Pei et al., 2023)

{{<citation>}}

Xiaobing Pei, Haoran Yang, Gang Shen. (2023)  
**HC-Ref: Hierarchical Constrained Refinement for Robust Adversarial Training of GNNs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Adversarial Training, GNN  
[Paper Link](http://arxiv.org/abs/2312.04879v1)  

---


**ABSTRACT**  
Recent studies have shown that attackers can catastrophically reduce the performance of GNNs by maliciously modifying the graph structure or node features on the graph. Adversarial training, which has been shown to be one of the most effective defense mechanisms against adversarial attacks in computer vision, holds great promise for enhancing the robustness of GNNs. There is limited research on defending against attacks by performing adversarial training on graphs, and it is crucial to delve deeper into this approach to optimize its effectiveness. Therefore, based on robust adversarial training on graphs, we propose a hierarchical constraint refinement framework (HC-Ref) that enhances the anti-perturbation capabilities of GNNs and downstream classifiers separately, ultimately leading to improved robustness. We propose corresponding adversarial regularization terms that are conducive to adaptively narrowing the domain gap between the normal part and the perturbation part according to the characteristics of different layers, promoting the smoothness of the predicted distribution of both parts. Moreover, existing research on graph robust adversarial training primarily concentrates on training from the standpoint of node feature perturbations and seldom takes into account alterations in the graph structure. This limitation makes it challenging to prevent attacks based on topological changes in the graph. This paper generates adversarial examples by utilizing graph structure perturbations, offering an effective approach to defend against attack methods that are based on topological changes. Extensive experiments on two real-world graph benchmarks show that HC-Ref successfully resists various attacks and has better node classification performance compared to several baseline methods.

{{</citation>}}


### (36/110) StructComp: Substituting propagation with Structural Compression in Training Graph Contrastive Learning (Shengzhong Zhang et al., 2023)

{{<citation>}}

Shengzhong Zhang, Wenjie Yang, Xinyuan Cao, Hongwei Zhang, Zengfeng Huang. (2023)  
**StructComp: Substituting propagation with Structural Compression in Training Graph Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.04865v1)  

---


**ABSTRACT**  
Graph contrastive learning (GCL) has become a powerful tool for learning graph data, but its scalability remains a significant challenge. In this work, we propose a simple yet effective training framework called Structural Compression (StructComp) to address this issue. Inspired by a sparse low-rank approximation on the diffusion matrix, StructComp trains the encoder with the compressed nodes. This allows the encoder not to perform any message passing during the training stage, and significantly reduces the number of sample pairs in the contrastive loss. We theoretically prove that the original GCL loss can be approximated with the contrastive loss computed by StructComp. Moreover, StructComp can be regarded as an additional regularization term for GCL models, resulting in a more robust encoder. Empirical studies on seven benchmark datasets show that StructComp greatly reduces the time and memory consumption while improving model performance compared to the vanilla GCL models and scalable training methods.

{{</citation>}}


### (37/110) Not All Negatives Are Worth Attending to: Meta-Bootstrapping Negative Sampling Framework for Link Prediction (Yakun Wang et al., 2023)

{{<citation>}}

Yakun Wang, Binbin Hu, Shuo Yang, Meiqi Zhu, Zhiqiang Zhang, Qiyang Zhang, Jun Zhou, Guo Ye, Huimei He. (2023)  
**Not All Negatives Are Worth Attending to: Meta-Bootstrapping Negative Sampling Framework for Link Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.04815v2)  

---


**ABSTRACT**  
The rapid development of graph neural networks (GNNs) encourages the rising of link prediction, achieving promising performance with various applications. Unfortunately, through a comprehensive analysis, we surprisingly find that current link predictors with dynamic negative samplers (DNSs) suffer from the migration phenomenon between "easy" and "hard" samples, which goes against the preference of DNS of choosing "hard" negatives, thus severely hindering capability. Towards this end, we propose the MeBNS framework, serving as a general plugin that can potentially improve current negative sampling based link predictors. In particular, we elaborately devise a Meta-learning Supported Teacher-student GNN (MST-GNN) that is not only built upon teacher-student architecture for alleviating the migration between "easy" and "hard" samples but also equipped with a meta learning based sample re-weighting module for helping the student GNN distinguish "hard" samples in a fine-grained manner. To effectively guide the learning of MST-GNN, we prepare a Structure enhanced Training Data Generator (STD-Generator) and an Uncertainty based Meta Data Collector (UMD-Collector) for supporting the teacher and student GNN, respectively. Extensive experiments show that the MeBNS achieves remarkable performance across six link prediction benchmark datasets.

{{</citation>}}


### (38/110) Target to Source: Guidance-Based Diffusion Model for Test-Time Adaptation (Kaiyu Song et al., 2023)

{{<citation>}}

Kaiyu Song, Hanjiang Lai. (2023)  
**Target to Source: Guidance-Based Diffusion Model for Test-Time Adaptation**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.05274v1)  

---


**ABSTRACT**  
Most recent works of test-time adaptation (TTA) aim to alleviate domain shift problems by re-training source classifiers in each domain. On the other hand, the emergence of the diffusion model provides another solution to TTA, which directly maps the test data from the target domain to the source domain based on a diffusion model pre-trained in the source domain. The source classifier does not need to be fine-tuned. However, 1) the semantic information loss from test data to the source domain and 2) the model shift between the source classifier and diffusion model would prevent the diffusion model from mapping the test data back to the source domain correctly. In this paper, we propose a novel guidance-based diffusion-driven adaptation (GDDA) to overcome the data shift and let the diffusion model find a better way to go back to the source. Concretely, we first propose detail and global guidance to better keep the common semantics of the test and source data. The two guidance include a contrastive loss and mean squared error to alleviate the information loss by fully exploring the diffusion model and the test data. Meanwhile, we propose a classifier-aware guidance to reduce the bias caused by the model shift, which can incorporate the source classifier's information into the generation process of the diffusion model. Extensive experiments on three image datasets with three classifier backbones demonstrate that GDDA significantly performs better than the state-of-the-art baselines. On CIFAR-10C, CIFAR-100C, and ImageNetC, GDDA achieves 11.54\%, 19.05\%, and 11.63\% average accuracy improvements, respectively. GDDA even achieves equal performance compared with methods of re-training classifiers. The code is available in the supplementary material.

{{</citation>}}


## cs.CL (21)



### (39/110) Towards Controlled Table-to-Text Generation with Scientific Reasoning (Zhixin Guo et al., 2023)

{{<citation>}}

Zhixin Guo, Jianping Zhou, Jiexing Qi, Mingxuan Yan, Ziwei He, Guanjie Zheng, Zhouhan Lin, Xinbing Wang, Chenghu Zhou. (2023)  
**Towards Controlled Table-to-Text Generation with Scientific Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reasoning, Text Generation  
[Paper Link](http://arxiv.org/abs/2312.05402v1)  

---


**ABSTRACT**  
The sheer volume of scientific experimental results and complex technical statements, often presented in tabular formats, presents a formidable barrier to individuals acquiring preferred information. The realms of scientific reasoning and content generation that adhere to user preferences encounter distinct challenges. In this work, we present a new task for generating fluent and logical descriptions that match user preferences over scientific tabular data, aiming to automate scientific document analysis. To facilitate research in this direction, we construct a new challenging dataset CTRLSciTab consisting of table-description pairs extracted from the scientific literature, with highlighted cells and corresponding domain-specific knowledge base. We evaluated popular pre-trained language models to establish a baseline and proposed a novel architecture outperforming competing approaches. The results showed that large models struggle to produce accurate content that aligns with user preferences. As the first of its kind, our work should motivate further research in scientific domains.

{{</citation>}}


### (40/110) PaperQA: Retrieval-Augmented Generative Agent for Scientific Research (Jakub Lála et al., 2023)

{{<citation>}}

Jakub Lála, Odhran O'Donoghue, Aleksandar Shtedritski, Sam Cox, Samuel G. Rodriques, Andrew D. White. (2023)  
**PaperQA: Retrieval-Augmented Generative Agent for Scientific Research**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2312.07559v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) generalize well across language tasks, but suffer from hallucinations and uninterpretability, making it difficult to assess their accuracy without ground-truth. Retrieval-Augmented Generation (RAG) models have been proposed to reduce hallucinations and provide provenance for how an answer was generated. Applying such models to the scientific literature may enable large-scale, systematic processing of scientific knowledge. We present PaperQA, a RAG agent for answering questions over the scientific literature. PaperQA is an agent that performs information retrieval across full-text scientific articles, assesses the relevance of sources and passages, and uses RAG to provide answers. Viewing this agent as a question answering model, we find it exceeds performance of existing LLMs and LLM agents on current science QA benchmarks. To push the field closer to how humans perform research on scientific literature, we also introduce LitQA, a more complex benchmark that requires retrieval and synthesis of information from full-text scientific papers across the literature. Finally, we demonstrate PaperQA's matches expert human researchers on LitQA.

{{</citation>}}


### (41/110) Seeing ChatGPT Through Universities' Policies, Resources and Guidelines (Hui Wang et al., 2023)

{{<citation>}}

Hui Wang, Anh Dang, Zihao Wu, Son Mac. (2023)  
**Seeing ChatGPT Through Universities' Policies, Resources and Guidelines**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.05235v1)  

---


**ABSTRACT**  
The advancements in Artificial Intelligence (AI) technologies such as ChatGPT have gained popularity in recent days. The integration of ChatGPT in educational contexts has already created attractions due to a wide range of applications. However, the automatic generation of human-like texts also poses potential risks to academic integrity, especially when faced with writing-intensive language courses. Considering the ongoing debates, this study aims to investigate the academic policies and guidelines established by US universities regarding the use of ChatGPT in teaching and learning. The data sources include academic policies, statements, guidelines as well as relevant resources that were provided by the top 50 universities in the United States, according to U.S. News. Thematic analysis and qualitative analysis were employed in the analysis and showed that most top 50 universities were open but cautious towards the integration of generative AI in teaching and learning and also expressed their concerns on ethical usage, accuracy, and data privacy. Most universities also provided a variety of resources and guidelines, including syllabus templates/samples, workshops and discussions, shared articles, and one-on-one consultations, with focuses on general technical introduction, ethical concerns, pedagogical applications, preventive strategies, data privacy, limitations, and detective tools. The findings will inform future policy-making regarding the integration of ChatGPT in college-level education and influence the provision of supportive resources by universities for the appropriate application of ChatGPT in education.

{{</citation>}}


### (42/110) DelucionQA: Detecting Hallucinations in Domain-specific Question Answering (Mobashir Sadat et al., 2023)

{{<citation>}}

Mobashir Sadat, Zhengyu Zhou, Lukas Lange, Jun Araki, Arsalan Gundroo, Bingqing Wang, Rakesh R Menon, Md Rizwan Parvez, Zhe Feng. (2023)  
**DelucionQA: Detecting Hallucinations in Domain-specific Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.05200v1)  

---


**ABSTRACT**  
Hallucination is a well-known phenomenon in text generated by large language models (LLMs). The existence of hallucinatory responses is found in almost all application scenarios e.g., summarization, question-answering (QA) etc. For applications requiring high reliability (e.g., customer-facing assistants), the potential existence of hallucination in LLM-generated text is a critical problem. The amount of hallucination can be reduced by leveraging information retrieval to provide relevant background information to the LLM. However, LLMs can still generate hallucinatory content for various reasons (e.g., prioritizing its parametric knowledge over the context, failure to capture the relevant information from the context, etc.). Detecting hallucinations through automated methods is thus paramount. To facilitate research in this direction, we introduce a sophisticated dataset, DelucionQA, that captures hallucinations made by retrieval-augmented LLMs for a domain-specific QA task. Furthermore, we propose a set of hallucination detection methods to serve as baselines for future works from the research community. Analysis and case study are also provided to share valuable insights on hallucination phenomena in the target scenario.

{{</citation>}}


### (43/110) Seamless: Multilingual Expressive and Streaming Speech Translation (Seamless Communication et al., 2023)

{{<citation>}}

Seamless Communication, Loïc Barrault, Yu-An Chung, Mariano Coria Meglioli, David Dale, Ning Dong, Mark Duppenthaler, Paul-Ambroise Duquenne, Brian Ellis, Hady Elsahar, Justin Haaheim, John Hoffman, Min-Jae Hwang, Hirofumi Inaguma, Christopher Klaiber, Ilia Kulikov, Pengwei Li, Daniel Licht, Jean Maillard, Ruslan Mavlyutov, Alice Rakotoarison, Kaushik Ram Sadagopan, Abinesh Ramakrishnan, Tuan Tran, Guillaume Wenzek, Yilin Yang, Ethan Ye, Ivan Evtimov, Pierre Fernandez, Cynthia Gao, Prangthip Hansanti, Elahe Kalbassi, Amanda Kallet, Artyom Kozhevnikov, Gabriel Mejia Gonzalez, Robin San Roman, Christophe Touret, Corinne Wong, Carleigh Wood, Bokai Yu, Pierre Andrews, Can Balioglu, Peng-Jen Chen, Marta R. Costa-jussà, Maha Elbayad, Hongyu Gong, Francisco Guzmán, Kevin Heffernan, Somya Jain, Justine Kao, Ann Lee, Xutai Ma, Alex Mourachko, Benjamin Peloquin, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Anna Sun, Paden Tomasello, Changhan Wang, Jeff Wang, Skyler Wang, Mary Williamson. (2023)  
**Seamless: Multilingual Expressive and Streaming Speech Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Attention, Multilingual  
[Paper Link](http://arxiv.org/abs/2312.05187v1)  

---


**ABSTRACT**  
Large-scale automatic speech translation systems today lack key features that help machine-mediated communication feel seamless when compared to human-to-human dialogue. In this work, we introduce a family of models that enable end-to-end expressive and multilingual translations in a streaming fashion. First, we contribute an improved version of the massively multilingual and multimodal SeamlessM4T model-SeamlessM4T v2. This newer model, incorporating an updated UnitY2 framework, was trained on more low-resource language data. SeamlessM4T v2 provides the foundation on which our next two models are initiated. SeamlessExpressive enables translation that preserves vocal styles and prosody. Compared to previous efforts in expressive speech research, our work addresses certain underexplored aspects of prosody, such as speech rate and pauses, while also preserving the style of one's voice. As for SeamlessStreaming, our model leverages the Efficient Monotonic Multihead Attention mechanism to generate low-latency target translations without waiting for complete source utterances. As the first of its kind, SeamlessStreaming enables simultaneous speech-to-speech/text translation for multiple source and target languages. To ensure that our models can be used safely and responsibly, we implemented the first known red-teaming effort for multimodal machine translation, a system for the detection and mitigation of added toxicity, a systematic evaluation of gender bias, and an inaudible localized watermarking mechanism designed to dampen the impact of deepfakes. Consequently, we bring major components from SeamlessExpressive and SeamlessStreaming together to form Seamless, the first publicly available system that unlocks expressive cross-lingual communication in real-time. The contributions to this work are publicly released and accessible at https://github.com/facebookresearch/seamless_communication

{{</citation>}}


### (44/110) PathFinder: Guided Search over Multi-Step Reasoning Paths (Olga Golovneva et al., 2023)

{{<citation>}}

Olga Golovneva, Sean O'Brien, Ramakanth Pasunuru, Tianlu Wang, Luke Zettlemoyer, Maryam Fazel-Zarandi, Asli Celikyilmaz. (2023)  
**PathFinder: Guided Search over Multi-Step Reasoning Paths**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2312.05180v2)  

---


**ABSTRACT**  
With recent advancements in large language models, methods like chain-of-thought prompting to elicit reasoning chains have been shown to improve results on reasoning tasks. However, tasks that require multiple steps of reasoning still pose significant challenges to state-of-the-art models. Drawing inspiration from the beam search algorithm, we propose PathFinder, a tree-search-based reasoning path generation approach. It enhances diverse branching and multi-hop reasoning through the integration of dynamic decoding, enabled by varying sampling methods and parameters. Using constrained reasoning, PathFinder integrates novel quality constraints, pruning, and exploration methods to enhance the efficiency and the quality of generation. Moreover, it includes scoring and ranking features to improve candidate selection. Our approach outperforms competitive baselines on three complex arithmetic and commonsense reasoning tasks by 6% on average. Our model generalizes well to longer, unseen reasoning chains, reflecting similar complexities to beam search with large branching factors.

{{</citation>}}


### (45/110) From Lengthy to Lucid: A Systematic Literature Review on NLP Techniques for Taming Long Sentences (Tatiana Passali et al., 2023)

{{<citation>}}

Tatiana Passali, Efstathios Chatzikyriakidis, Stelios Andreadis, Thanos G. Stavropoulos, Anastasia Matonaki, Anestis Fachantidis, Grigorios Tsoumakas. (2023)  
**From Lengthy to Lucid: A Systematic Literature Review on NLP Techniques for Taming Long Sentences**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.05172v1)  

---


**ABSTRACT**  
Long sentences have been a persistent issue in written communication for many years since they make it challenging for readers to grasp the main points or follow the initial intention of the writer. This survey, conducted using the PRISMA guidelines, systematically reviews two main strategies for addressing the issue of long sentences: a) sentence compression and b) sentence splitting. An increased trend of interest in this area has been observed since 2005, with significant growth after 2017. Current research is dominated by supervised approaches for both sentence compression and splitting. Yet, there is a considerable gap in weakly and self-supervised techniques, suggesting an opportunity for further research, especially in domains with limited data. In this survey, we categorize and group the most representative methods into a comprehensive taxonomy. We also conduct a comparative evaluation analysis of these methods on common sentence compression and splitting datasets. Finally, we discuss the challenges and limitations of current methods, providing valuable insights for future research directions. This survey is meant to serve as a comprehensive resource for addressing the complexities of long sentences. We aim to enable researchers to make further advancements in the field until long sentences are no longer a barrier to effective communication.

{{</citation>}}


### (46/110) LaCour!: Enabling Research on Argumentation in Hearings of the European Court of Human Rights (Lena Held et al., 2023)

{{<citation>}}

Lena Held, Ivan Habernal. (2023)  
**LaCour!: Enabling Research on Argumentation in Hearings of the European Court of Human Rights**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.05061v1)  

---


**ABSTRACT**  
Why does an argument end up in the final court decision? Was it deliberated or questioned during the oral hearings? Was there something in the hearings that triggered a particular judge to write a dissenting opinion? Despite the availability of the final judgments of the European Court of Human Rights (ECHR), none of these legal research questions can currently be answered as the ECHR's multilingual oral hearings are not transcribed, structured, or speaker-attributed. We address this fundamental gap by presenting LaCour!, the first corpus of textual oral arguments of the ECHR, consisting of 154 full hearings (2.1 million tokens from over 267 hours of video footage) in English, French, and other court languages, each linked to the corresponding final judgment documents. In addition to the transcribed and partially manually corrected text from the video, we provide sentence-level timestamps and manually annotated role and language labels. We also showcase LaCour! in a set of preliminary experiments that explore the interplay between questions and dissenting opinions. Apart from the use cases in legal NLP, we hope that law students or other interested parties will also use LaCour! as a learning resource, as it is freely available in various formats at https://huggingface.co/datasets/TrustHLT/LaCour.

{{</citation>}}


### (47/110) Converting Epics/Stories into Pseudocode using Transformers (Gaurav Kolhatkar et al., 2023)

{{<citation>}}

Gaurav Kolhatkar, Akshit Madan, Nidhi Kowtal, Satyajit Roy, Sheetal Sonawane. (2023)  
**Converting Epics/Stories into Pseudocode using Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Natural Language Processing, T5, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.05047v1)  

---


**ABSTRACT**  
The conversion of user epics or stories into their appropriate representation in pseudocode or code is a time-consuming task, which can take up a large portion of the time in an industrial project. With this research paper, we aim to present a methodology to generate pseudocode from a given agile user story of small functionalities so as to reduce the overall time spent on the industrial project. Pseudocode is a programming language agnostic representation of the steps involved in a computer program, which can be easily converted into any programming language. Leveraging the potential of Natural Language Processing, we want to simplify the development process in organizations that use the Agile Model of Software Development. We present a methodology to convert a problem described in the English language into pseudocode. This methodology divides the Text to Pseudocode conversion task into two stages or subtasks, each of which is treated like an individual machine translation task. Stage 1 is Text to Code Conversion and Stage 2 is Code to Pseudocode Conversion. We find that the CodeT5 model gives the best results in terms of BLEU score when trained separately on the two subtasks mentioned above. BLEU score is a metric that is used to measure the similarity between a machine-translated text and a set of reference translations.

{{</citation>}}


### (48/110) Boosting Prompt-Based Self-Training With Mapping-Free Automatic Verbalizer for Multi-Class Classification (Yookyung Kho et al., 2023)

{{<citation>}}

Yookyung Kho, Jaehee Kim, Pilsung Kang. (2023)  
**Boosting Prompt-Based Self-Training With Mapping-Free Automatic Verbalizer for Multi-Class Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.04982v1)  

---


**ABSTRACT**  
Recently, prompt-based fine-tuning has garnered considerable interest as a core technique for few-shot text classification task. This approach reformulates the fine-tuning objective to align with the Masked Language Modeling (MLM) objective. Leveraging unlabeled data, prompt-based self-training has shown greater effectiveness in binary and three-class classification. However, prompt-based self-training for multi-class classification has not been adequately investigated, despite its significant applicability to real-world scenarios. Moreover, extending current methods to multi-class classification suffers from the verbalizer that extracts the predicted value of manually pre-defined single label word for each class from MLM predictions. Consequently, we introduce a novel, efficient verbalizer structure, named Mapping-free Automatic Verbalizer (MAV). Comprising two fully connected layers, MAV serves as a trainable verbalizer that automatically extracts the requisite word features for classification by capitalizing on all available information from MLM predictions. Experimental results on five multi-class classification datasets indicate MAV's superior self-training efficacy.

{{</citation>}}


### (49/110) Zoology: Measuring and Improving Recall in Efficient Language Models (Simran Arora et al., 2023)

{{<citation>}}

Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, Christopher Ré. (2023)  
**Zoology: Measuring and Improving Recall in Efficient Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Attention, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2312.04927v1)  

---


**ABSTRACT**  
Attention-free language models that combine gating and convolutions are growing in popularity due to their efficiency and increasingly competitive performance. To better understand these architectures, we pretrain a suite of 17 attention and "gated-convolution" language models, finding that SoTA gated-convolution architectures still underperform attention by up to 2.1 perplexity points on the Pile. In fine-grained analysis, we find 82% of the gap is explained by each model's ability to recall information that is previously mentioned in-context, e.g. "Hakuna Matata means no worries Hakuna Matata it means no" $\rightarrow$ "??". On this task, termed "associative recall", we find that attention outperforms gated-convolutions by a large margin: a 70M parameter attention model outperforms a 1.4 billion parameter gated-convolution model on associative recall. This is surprising because prior work shows gated convolutions can perfectly solve synthetic tests for AR capability. To close the gap between synthetics and real language, we develop a new formalization of the task called multi-query associative recall (MQAR) that better reflects actual language. We perform an empirical and theoretical study of MQAR that elucidates differences in the parameter-efficiency of attention and gated-convolution recall. Informed by our analysis, we evaluate simple convolution-attention hybrids and show that hybrids with input-dependent sparse attention patterns can close 97.4% of the gap to attention, while maintaining sub-quadratic scaling. Our code is accessible at: https://github.com/HazyResearch/zoology.

{{</citation>}}


### (50/110) Lyrics: Boosting Fine-grained Language-Vision Alignment and Comprehension via Semantic-aware Visual Objects (Junyu Lu et al., 2023)

{{<citation>}}

Junyu Lu, Ruyi Gan, Dixiang Zhang, Xiaojun Wu, Ziwei Wu, Renliang Sun, Jiaxing Zhang, Pingjian Zhang, Yan Song. (2023)  
**Lyrics: Boosting Fine-grained Language-Vision Alignment and Comprehension via Semantic-aware Visual Objects**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2312.05278v1)  

---


**ABSTRACT**  
Large Vision Language Models (LVLMs) have demonstrated impressive zero-shot capabilities in various vision-language dialogue scenarios. However, the absence of fine-grained visual object detection hinders the model from understanding the details of images, leading to irreparable visual hallucinations and factual errors. In this paper, we propose Lyrics, a novel multi-modal pre-training and instruction fine-tuning paradigm that bootstraps vision-language alignment from fine-grained cross-modal collaboration. Building on the foundation of BLIP-2, Lyrics infuses local visual features extracted from a visual refiner that includes image tagging, object detection and semantic segmentation modules into the Querying Transformer, while on the text side, the language inputs equip the boundary boxes and tags derived from the visual refiner. We further introduce a two-stage training scheme, in which the pre-training stage bridges the modality gap through explicit and comprehensive vision-language alignment targets. During the instruction fine-tuning stage, we introduce semantic-aware visual feature extraction, a crucial method that enables the model to extract informative features from concrete visual objects. Our approach achieves strong performance on 13 held-out datasets across various vision-language tasks, and demonstrates promising multi-modal understanding and detailed depiction capabilities in real dialogue scenarios.

{{</citation>}}


### (51/110) Ophtha-LLaMA2: A Large Language Model for Ophthalmology (Huan Zhao et al., 2023)

{{<citation>}}

Huan Zhao, Qian Ling, Yi Pan, Tianyang Zhong, Jin-Yu Hu, Junjie Yao, Fengqian Xiao, Zhenxiang Xiao, Yutong Zhang, San-Hua Xu, Shi-Nan Wu, Min Kang, Zihao Wu, Zhengliang Liu, Xi Jiang, Tianming Liu, Yi Shao. (2023)  
**Ophtha-LLaMA2: A Large Language Model for Ophthalmology**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.04906v1)  

---


**ABSTRACT**  
In recent years, pre-trained large language models (LLMs) have achieved tremendous success in the field of Natural Language Processing (NLP). Prior studies have primarily focused on general and generic domains, with relatively less research on specialized LLMs in the medical field. The specialization and high accuracy requirements for diagnosis in the medical field, as well as the challenges in collecting large-scale data, have constrained the application and development of LLMs in medical scenarios. In the field of ophthalmology, clinical diagnosis mainly relies on doctors' interpretation of reports and making diagnostic decisions. In order to take advantage of LLMs to provide decision support for doctors, we collected three modalities of ophthalmic report data and fine-tuned the LLaMA2 model, successfully constructing an LLM termed the "Ophtha-LLaMA2" specifically tailored for ophthalmic disease diagnosis. Inference test results show that even with a smaller fine-tuning dataset, Ophtha-LLaMA2 performs significantly better in ophthalmic diagnosis compared to other LLMs. It demonstrates that the Ophtha-LLaMA2 exhibits satisfying accuracy and efficiency in ophthalmic disease diagnosis, making it a valuable tool for ophthalmologists to provide improved diagnostic support for patients. This research provides a useful reference for the application of LLMs in the field of ophthalmology, while showcasing the immense potential and prospects in this domain.

{{</citation>}}


### (52/110) Classification of Human- and AI-Generated Texts for English, French, German, and Spanish (Kristina Schaaff et al., 2023)

{{<citation>}}

Kristina Schaaff, Tim Schlippe, Lorenz Mindner. (2023)  
**Classification of Human- and AI-Generated Texts for English, French, German, and Spanish**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04882v1)  

---


**ABSTRACT**  
In this paper we analyze features to classify human- and AI-generated text for English, French, German and Spanish and compare them across languages. We investigate two scenarios: (1) The detection of text generated by AI from scratch, and (2) the detection of text rephrased by AI. For training and testing the classifiers in this multilingual setting, we created a new text corpus covering 10 topics for each language. For the detection of AI-generated text, the combination of all proposed features performs best, indicating that our features are portable to other related languages: The F1-scores are close with 99% for Spanish, 98% for English, 97% for German and 95% for French. For the detection of AI-rephrased text, the systems with all features outperform systems with other features in many cases, but using only document features performs best for German (72%) and Spanish (86%) and only text vector features leads to best results for English (78%).

{{</citation>}}


### (53/110) Generating Explanations to Understand and Repair Embedding-based Entity Alignment (Xiaobin Tian et al., 2023)

{{<citation>}}

Xiaobin Tian, Zequn Sun, Wei Hu. (2023)  
**Generating Explanations to Understand and Repair Embedding-based Entity Alignment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-DB, cs.CL  
Keywords: Embedding, Entity Alignment  
[Paper Link](http://arxiv.org/abs/2312.04877v1)  

---


**ABSTRACT**  
Entity alignment seeks identical entities in different knowledge graphs, which is a long-standing task in the database research. Recent work leverages deep learning to embed entities in vector space and align them via nearest neighbor search. Although embedding-based entity alignment has gained marked success in recent years, it lacks explanations for alignment decisions. In this paper, we present the first framework that can generate explanations for understanding and repairing embedding-based entity alignment results. Given an entity alignment pair produced by an embedding model, we first compare its neighbor entities and relations to build a matching subgraph as a local explanation. We then construct an alignment dependency graph to understand the pair from an abstract perspective. Finally, we repair the pair by resolving three types of alignment conflicts based on dependency graphs. Experiments on five datasets demonstrate the effectiveness and generalization of our framework in explaining and repairing embedding-based entity alignment results.

{{</citation>}}


### (54/110) Apollo's Oracle: Retrieval-Augmented Reasoning in Multi-Agent Debates (Haotian Wang et al., 2023)

{{<citation>}}

Haotian Wang, Xiyuan Du, Weijiang Yu, Qianglong Chen, Kun Zhu, Zheng Chu, Lian Yan, Yi Guan. (2023)  
**Apollo's Oracle: Retrieval-Augmented Reasoning in Multi-Agent Debates**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2312.04854v1)  

---


**ABSTRACT**  
Multi-agent debate systems are designed to derive accurate and consistent conclusions through adversarial interactions among agents. However, these systems often encounter challenges due to cognitive constraints, manifesting as (1) agents' obstinate adherence to incorrect viewpoints and (2) their propensity to abandon correct viewpoints. These issues are primarily responsible for the ineffectiveness of such debates. Addressing the challenge of cognitive constraints, we introduce a novel framework, the Multi-Agent Debate with Retrieval Augmented (MADRA). MADRA incorporates retrieval of prior knowledge into the debate process, effectively breaking cognitive constraints and enhancing the agents' reasoning capabilities. Furthermore, we have developed a self-selection module within this framework, enabling agents to autonomously select pertinent evidence, thereby minimizing the impact of irrelevant or noisy data. We have comprehensively tested and analyzed MADRA across six diverse datasets. The experimental results demonstrate that our approach significantly enhances performance across various tasks, proving the effectiveness of our proposed method.

{{</citation>}}


### (55/110) FREDSum: A Dialogue Summarization Corpus for French Political Debates (Virgile Rennard et al., 2023)

{{<citation>}}

Virgile Rennard, Guokan Shang, Damien Grari, Julie Hunter, Michalis Vazirgiannis. (2023)  
**FREDSum: A Dialogue Summarization Corpus for French Political Debates**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Summarization  
[Paper Link](http://arxiv.org/abs/2312.04843v1)  

---


**ABSTRACT**  
Recent advances in deep learning, and especially the invention of encoder-decoder architectures, has significantly improved the performance of abstractive summarization systems. The majority of research has focused on written documents, however, neglecting the problem of multi-party dialogue summarization. In this paper, we present a dataset of French political debates for the purpose of enhancing resources for multi-lingual dialogue summarization. Our dataset consists of manually transcribed and annotated political debates, covering a range of topics and perspectives. We highlight the importance of high quality transcription and annotations for training accurate and effective dialogue summarization models, and emphasize the need for multilingual resources to support dialogue summarization in non-English languages. We also provide baseline experiments using state-of-the-art methods, and encourage further research in this area to advance the field of dialogue summarization. Our dataset will be made publicly available for use by the research community.

{{</citation>}}


### (56/110) HuRef: HUman-REadable Fingerprint for Large Language Models (Boyi Zeng et al., 2023)

{{<citation>}}

Boyi Zeng, Chenghu Zhou, Xinbing Wang, Zhouhan Lin. (2023)  
**HuRef: HUman-REadable Fingerprint for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2312.04828v1)  

---


**ABSTRACT**  
Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations through fine-tuning or continued pretraining. In this study, we introduce HuRef, a human-readable fingerprint for LLMs that uniquely identifies the base model without exposing model parameters or interfering with training. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, showing negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning (SFT), and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension permutation or matrix rotation, which significantly change it without affecting performance. To address this, leveraging the Transformer structure, we systematically analyze potential attacks and define three invariant terms that identify an LLM's base model. We make these invariant terms human-readable by mapping them to a Gaussian vector using a convolutional encoder and then converting it into a natural image with StyleGAN2. Our method generates a dog image as an identity fingerprint for an LLM, where the dog's appearance strongly indicates the LLM's base model. Experimental results across various LLMs demonstrate the effectiveness of our method, the generated dog image remains invariant to different training steps, including SFT, RLHF, or even continued pretraining with augmented vocabulary in a new language.

{{</citation>}}


### (57/110) Improving Neural Machine Translation by Multi-Knowledge Integration with Prompting (Ke Wang et al., 2023)

{{<citation>}}

Ke Wang, Jun Xie, Yuqi Zhang, Yu Zhao. (2023)  
**Improving Neural Machine Translation by Multi-Knowledge Integration with Prompting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2312.04807v1)  

---


**ABSTRACT**  
Improving neural machine translation (NMT) systems with prompting has achieved significant progress in recent years. In this work, we focus on how to integrate multi-knowledge, multiple types of knowledge, into NMT models to enhance the performance with prompting. We propose a unified framework, which can integrate effectively multiple types of knowledge including sentences, terminologies/phrases and translation templates into NMT models. We utilize multiple types of knowledge as prefix-prompts of input for the encoder and decoder of NMT models to guide the translation process. The approach requires no changes to the model architecture and effectively adapts to domain-specific translation without retraining. The experiments on English-Chinese and English-German translation demonstrate that our approach significantly outperform strong baselines, achieving high translation quality and terminology match accuracy.

{{</citation>}}


### (58/110) How to Determine the Most Powerful Pre-trained Language Model without Brute Force Fine-tuning? An Empirical Survey (Jun Bai et al., 2023)

{{<citation>}}

Jun Bai, Xiaofeng Zhang, Chen Li, Hanhua Hong, Xi Xu, Chenghua Lin, Wenge Rong. (2023)  
**How to Determine the Most Powerful Pre-trained Language Model without Brute Force Fine-tuning? An Empirical Survey**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLUE, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04775v1)  

---


**ABSTRACT**  
Transferability estimation has been attached to great attention in the computer vision fields. Researchers try to estimate with low computational cost the performance of a model when transferred from a source task to a given target task. Considering the effectiveness of such estimations, the communities of natural language processing also began to study similar problems for the selection of pre-trained language models. However, there is a lack of a comprehensive comparison between these estimation methods yet. Also, the differences between vision and language scenarios make it doubtful whether previous conclusions can be established across fields. In this paper, we first conduct a thorough survey of existing transferability estimation methods being able to find the most suitable model, then we conduct a detailed empirical study for the surveyed methods based on the GLUE benchmark. From qualitative and quantitative analyses, we demonstrate the strengths and weaknesses of existing methods and show that H-Score generally performs well with superiorities in effectiveness and efficiency. We also outline the difficulties of consideration of training details, applicability to text generation, and consistency to certain metrics which shed light on future directions.

{{</citation>}}


### (59/110) First Attempt at Building Parallel Corpora for Machine Translation of Northeast India's Very Low-Resource Languages (Atnafu Lambebo Tonja et al., 2023)

{{<citation>}}

Atnafu Lambebo Tonja, Melkamu Mersha, Ananya Kalita, Olga Kolesnikova, Jugal Kalita. (2023)  
**First Attempt at Building Parallel Corpora for Machine Translation of Northeast India's Very Low-Resource Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Low-Resource, Machine Translation  
[Paper Link](http://arxiv.org/abs/2312.04764v1)  

---


**ABSTRACT**  
This paper presents the creation of initial bilingual corpora for thirteen very low-resource languages of India, all from Northeast India. It also presents the results of initial translation efforts in these languages. It creates the first-ever parallel corpora for these languages and provides initial benchmark neural machine translation results for these languages. We intend to extend these corpora to include a large number of low-resource Indian languages and integrate the effort with our prior work with African and American-Indian languages to create corpora covering a large number of languages from across the world.

{{</citation>}}


## cs.IT (2)



### (60/110) Generative Network Layer for Communication Systems with Artificial Intelligence (Mathias Thorsager et al., 2023)

{{<citation>}}

Mathias Thorsager, Israel Leyva-Mayorga, Beatriz Soret, Petar Popovski. (2023)  
**Generative Network Layer for Communication Systems with Artificial Intelligence**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-LG, cs.IT, math-IT  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.05398v1)  

---


**ABSTRACT**  
The traditional role of the network layer is the transfer of packet replicas from source to destination through intermediate network nodes. We present a generative network layer that uses Generative AI (GenAI) at intermediate or edge network nodes and analyze its impact on the required data rates in the network. We conduct a case study where the GenAI-aided nodes generate images from prompts that consist of substantially compressed latent representations. The results from network flow analyses under image quality constraints show that the generative network layer can achieve an improvement of more than 100% in terms of the required data rate.

{{</citation>}}


### (61/110) Active Eavesdropper Mitigation via Orthogonal Channel Estimation (Gian Marti et al., 2023)

{{<citation>}}

Gian Marti, Christoph Studer. (2023)  
**Active Eavesdropper Mitigation via Orthogonal Channel Estimation**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.05025v1)  

---


**ABSTRACT**  
Beamforming is a powerful tool for physical layer security, as it can be used for steering signals towards legitimate receivers and away from eavesdroppers. An active eavesdropper, however, can interfere with the pilot phase that the transmitter needs to acquire the channel knowledge necessary for beamforming. By doing so, the eavesdropper can make the transmitter form beams towards the eavesdropper rather than towards the legitimate receiver. To mitigate active eavesdroppers, we propose VILLAIN, a novel channel estimator that uses secret pilots. When an eavesdropper interferes with the pilot phase, VILLAIN produces a channel estimate that is orthogonal to the eavesdropper's channel (in the noiseless case). We prove that beamforming based on this channel estimate delivers the highest possible signal power to the legitimate receiver without delivering any signal power to the eavesdropper. Simulations show that VILLAIN mitigates active eavesdroppers also in the noisy case.

{{</citation>}}


## cs.AI (11)



### (62/110) The logic of NTQR evaluations of noisy AI agents: Complete postulates and logically consistent error correlations (Andrés Corrada-Emmanuel, 2023)

{{<citation>}}

Andrés Corrada-Emmanuel. (2023)  
**The logic of NTQR evaluations of noisy AI agents: Complete postulates and logically consistent error correlations**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.05392v1)  

---


**ABSTRACT**  
In his "ship of state" allegory (\textit{Republic}, Book VI, 488) Plato poses a question -- how can a crew of sailors presumed to know little about the art of navigation recognize the true pilot among them? The allegory argues that a simple majority voting procedure cannot safely determine who is most qualified to pilot a ship when the voting members are ignorant or biased. We formalize Plato's concerns by considering the problem in AI safety of monitoring noisy AI agents in unsupervised settings. An algorithm evaluating AI agents using unlabeled data would be subject to the evaluation dilemma - how would we know the evaluation algorithm was correct itself? This endless validation chain can be avoided by considering purely algebraic functions of the observed responses. We can construct complete postulates than can prove or disprove the logical consistency of any grading algorithm. A complete set of postulates exists whenever we are evaluating $N$ experts that took $T$ tests with $Q$ questions with $R$ responses each. We discuss evaluating binary classifiers that have taken a single test - the $(N,T=1,Q,R=2)$ tests. We show how some of the postulates have been previously identified in the ML literature but not recognized as such - the \textbf{agreement equations} of Platanios. The complete postulates for pair correlated binary classifiers are considered and we show how it allows for error correlations to be quickly calculated. An algebraic evaluator based on the assumption that the ensemble is error independent is compared with grading by majority voting on evaluations using the \uciadult and and \texttt{two-norm} datasets. Throughout, we demonstrate how the formalism of logical consistency via algebraic postulates of evaluation can help increase the safety of machines using AI algorithms.

{{</citation>}}


### (63/110) Exploring Parity Challenges in Reinforcement Learning through Curriculum Learning with Noisy Labels (Bei Zhou et al., 2023)

{{<citation>}}

Bei Zhou, Soren Riis. (2023)  
**Exploring Parity Challenges in Reinforcement Learning through Curriculum Learning with Noisy Labels**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05379v1)  

---


**ABSTRACT**  
This paper delves into applying reinforcement learning (RL) in strategy games, particularly those characterized by parity challenges, as seen in specific positions of Go and Chess and a broader range of impartial games. We propose a simulated learning process, structured within a curriculum learning framework and augmented with noisy labels, to mirror the intricacies of self-play learning scenarios. This approach thoroughly analyses how neural networks (NNs) adapt and evolve from elementary to increasingly complex game positions. Our empirical research indicates that even minimal label noise can significantly impede NNs' ability to discern effective strategies, a difficulty that intensifies with the growing complexity of the game positions. These findings underscore the urgent need for advanced methodologies in RL training, specifically tailored to counter the obstacles imposed by noisy evaluations. The development of such methodologies is crucial not only for enhancing NN proficiency in strategy games with significant parity elements but also for broadening the resilience and efficiency of RL systems across diverse and complex environments.

{{</citation>}}


### (64/110) Emergence and Function of Abstract Representations in Self-Supervised Transformers (Quentin RV. Ferry et al., 2023)

{{<citation>}}

Quentin RV. Ferry, Joshua Ching, Takashi Kawai. (2023)  
**Emergence and Function of Abstract Representations in Self-Supervised Transformers**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Self-Supervised, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.05361v1)  

---


**ABSTRACT**  
Human intelligence relies in part on our brains' ability to create abstract mental models that succinctly capture the hidden blueprint of our reality. Such abstract world models notably allow us to rapidly navigate novel situations by generalizing prior knowledge, a trait deep learning systems have historically struggled to replicate. However, the recent shift from supervised to self-supervised objectives, combined with expressive transformer-based architectures, have yielded powerful foundation models that appear to learn versatile representations that can support a wide range of downstream tasks. This promising development raises the intriguing possibility of such models developing in silico abstract world models. We test this hypothesis by studying the inner workings of small-scale transformers trained to reconstruct partially masked visual scenes generated from a simple blueprint. We show that the network develops intermediate abstract representations, or abstractions, that encode all semantic features of the dataset. These abstractions manifest as low-dimensional manifolds where the embeddings of semantically related tokens transiently converge, thus allowing for the generalization of downstream computations. Using precise manipulation experiments, we demonstrate that abstractions are central to the network's decision-making process. Our research also suggests that these abstractions are compositionally structured, exhibiting features like contextual independence and part-whole relationships that mirror the compositional nature of the dataset. Finally, we introduce a Language-Enhanced Architecture (LEA) designed to encourage the network to articulate its computations. We find that LEA develops an abstraction-centric language that can be easily interpreted, allowing us to more readily access and steer the network's decision-making process.

{{</citation>}}


### (65/110) Bad Students Make Great Teachers: Active Learning Accelerates Large-Scale Visual Understanding (Talfan Evans et al., 2023)

{{<citation>}}

Talfan Evans, Shreya Pathak, Hamza Merzic, Jonathan Schwarz, Ryutaro Tanno, Olivier J. Henaff. (2023)  
**Bad Students Make Great Teachers: Active Learning Accelerates Large-Scale Visual Understanding**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2312.05328v2)  

---


**ABSTRACT**  
We propose a method for accelerating large-scale pre-training with online data selection policies. For the first time, we demonstrate that model-based data selection can reduce the total computation needed to reach the performance of models trained with uniform sampling. The key insight which enables this "compute-positive" regime is that small models provide good proxies for the loss of much larger models, such that computation spent on scoring data can be drastically scaled down but still significantly accelerate training of the learner.. These data selection policies also strongly generalize across datasets and tasks, opening an avenue for further amortizing the overhead of data scoring by re-using off-the-shelf models and training sequences. Our methods, ClassAct and ActiveCLIP, require 46% and 51% fewer training updates and up to 25% less total computation when training visual classifiers on JFT and multimodal models on ALIGN, respectively. Finally, our paradigm seamlessly applies to the curation of large-scale image-text datasets, yielding a new state-of-the-art in several multimodal transfer tasks and pre-training regimes.

{{</citation>}}


### (66/110) Language Models, Agent Models, and World Models: The LAW for Machine Reasoning and Planning (Zhiting Hu et al., 2023)

{{<citation>}}

Zhiting Hu, Tianmin Shu. (2023)  
**Language Models, Agent Models, and World Models: The LAW for Machine Reasoning and Planning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs-RO, cs.AI  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.05230v1)  

---


**ABSTRACT**  
Despite their tremendous success in many applications, large language models often fall short of consistent reasoning and planning in various (language, embodied, and social) scenarios, due to inherent limitations in their inference, learning, and modeling capabilities. In this position paper, we present a new perspective of machine reasoning, LAW, that connects the concepts of Language models, Agent models, and World models, for more robust and versatile reasoning capabilities. In particular, we propose that world and agent models are a better abstraction of reasoning, that introduces the crucial elements of deliberate human-like reasoning, including beliefs about the world and other agents, anticipation of consequences, goals/rewards, and strategic planning. Crucially, language models in LAW serve as a backend to implement the system or its elements and hence provide the computational power and adaptability. We review the recent studies that have made relevant progress and discuss future research directions towards operationalizing the LAW framework.

{{</citation>}}


### (67/110) HALO: An Ontology for Representing Hallucinations in Generative Models (Navapat Nananukul et al., 2023)

{{<citation>}}

Navapat Nananukul, Mayank Kejriwal. (2023)  
**HALO: An Ontology for Representing Hallucinations in Generative Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.05209v1)  

---


**ABSTRACT**  
Recent progress in generative AI, including large language models (LLMs) like ChatGPT, has opened up significant opportunities in fields ranging from natural language processing to knowledge discovery and data mining. However, there is also a growing awareness that the models can be prone to problems such as making information up or `hallucinations', and faulty reasoning on seemingly simple problems. Because of the popularity of models like ChatGPT, both academic scholars and citizen scientists have documented hallucinations of several different types and severity. Despite this body of work, a formal model for describing and representing these hallucinations (with relevant meta-data) at a fine-grained level, is still lacking. In this paper, we address this gap by presenting the Hallucination Ontology or HALO, a formal, extensible ontology written in OWL that currently offers support for six different types of hallucinations known to arise in LLMs, along with support for provenance and experimental metadata. We also collect and publish a dataset containing hallucinations that we inductively gathered across multiple independent Web sources, and show that HALO can be successfully used to model this dataset and answer competency questions.

{{</citation>}}


### (68/110) DARLEI: Deep Accelerated Reinforcement Learning with Evolutionary Intelligence (Saeejith Nair et al., 2023)

{{<citation>}}

Saeejith Nair, Mohammad Javad Shafiee, Alexander Wong. (2023)  
**DARLEI: Deep Accelerated Reinforcement Learning with Evolutionary Intelligence**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-NE, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05171v1)  

---


**ABSTRACT**  
We present DARLEI, a framework that combines evolutionary algorithms with parallelized reinforcement learning for efficiently training and evolving populations of UNIMAL agents. Our approach utilizes Proximal Policy Optimization (PPO) for individual agent learning and pairs it with a tournament selection-based generational learning mechanism to foster morphological evolution. By building on Nvidia's Isaac Gym, DARLEI leverages GPU accelerated simulation to achieve over 20x speedup using just a single workstation, compared to previous work which required large distributed CPU clusters. We systematically characterize DARLEI's performance under various conditions, revealing factors impacting diversity of evolved morphologies. For example, by enabling inter-agent collisions within the simulator, we find that we can simulate some multi-agent interactions between the same morphology, and see how it influences individual agent capabilities and long-term evolutionary adaptation. While current results demonstrate limited diversity across generations, we hope to extend DARLEI in future work to include interactions between diverse morphologies in richer environments, and create a platform that allows for coevolving populations and investigating emergent behaviours in them. Our source code is also made publicly at https://saeejithnair.github.io/darlei.

{{</citation>}}


### (69/110) Beyond Transduction: A Survey on Inductive, Few Shot, and Zero Shot Link Prediction in Knowledge Graphs (Nicolas Hubert et al., 2023)

{{<citation>}}

Nicolas Hubert, Pierre Monnin, Heiko Paulheim. (2023)  
**Beyond Transduction: A Survey on Inductive, Few Shot, and Zero Shot Link Prediction in Knowledge Graphs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.04997v1)  

---


**ABSTRACT**  
Knowledge graphs (KGs) comprise entities interconnected by relations of different semantic meanings. KGs are being used in a wide range of applications. However, they inherently suffer from incompleteness, i.e. entities or facts about entities are missing. Consequently, a larger body of works focuses on the completion of missing information in KGs, which is commonly referred to as link prediction (LP). This task has traditionally and extensively been studied in the transductive setting, where all entities and relations in the testing set are observed during training. Recently, several works have tackled the LP task under more challenging settings, where entities and relations in the test set may be unobserved during training, or appear in only a few facts. These works are known as inductive, few-shot, and zero-shot link prediction. In this work, we conduct a systematic review of existing works in this area. A thorough analysis leads us to point out the undesirable existence of diverging terminologies and task definitions for the aforementioned settings, which further limits the possibility of comparison between recent works. We consequently aim at dissecting each setting thoroughly, attempting to reveal its intrinsic characteristics. A unifying nomenclature is ultimately proposed to refer to each of them in a simple and consistent manner.

{{</citation>}}


### (70/110) KwaiAgents: Generalized Information-seeking Agent System with Large Language Models (Haojie Pan et al., 2023)

{{<citation>}}

Haojie Pan, Zepeng Zhai, Hao Yuan, Yaojia Lv, Ruiji Fu, Ming Liu, Zhongyuan Wang, Bing Qin. (2023)  
**KwaiAgents: Generalized Information-seeking Agent System with Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04889v1)  

---


**ABSTRACT**  
Driven by curiosity, humans have continually sought to explore and understand the world around them, leading to the invention of various tools to satiate this inquisitiveness. Despite not having the capacity to process and memorize vast amounts of information in their brains, humans excel in critical thinking, planning, reflection, and harnessing available tools to interact with and interpret the world, enabling them to find answers efficiently. The recent advancements in large language models (LLMs) suggest that machines might also possess the aforementioned human-like capabilities, allowing them to exhibit powerful abilities even with a constrained parameter count. In this paper, we introduce KwaiAgents, a generalized information-seeking agent system based on LLMs. Within KwaiAgents, we propose an agent system that employs LLMs as its cognitive core, which is capable of understanding a user's query, behavior guidelines, and referencing external documents. The agent can also update and retrieve information from its internal memory, plan and execute actions using a time-aware search-browse toolkit, and ultimately provide a comprehensive response. We further investigate the system's performance when powered by LLMs less advanced than GPT-4, and introduce the Meta-Agent Tuning (MAT) framework, designed to ensure even an open-sourced 7B or 13B model performs well among many agent systems. We exploit both benchmark and human evaluations to systematically validate these capabilities. Extensive experiments show the superiority of our agent system compared to other autonomous agents and highlight the enhanced generalized agent-abilities of our fine-tuned LLMs.

{{</citation>}}


### (71/110) Localized Symbolic Knowledge Distillation for Visual Commonsense Models (Jae Sung Park et al., 2023)

{{<citation>}}

Jae Sung Park, Jack Hessel, Khyathi Raghavi Chandu, Paul Pu Liang, Ximing Lu, Peter West, Youngjae Yu, Qiuyuan Huang, Jianfeng Gao, Ali Farhadi, Yejin Choi. (2023)  
**Localized Symbolic Knowledge Distillation for Visual Commonsense Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs.AI  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2312.04837v2)  

---


**ABSTRACT**  
Instruction following vision-language (VL) models offer a flexible interface that supports a broad range of multimodal tasks in a zero-shot fashion. However, interfaces that operate on full images do not directly enable the user to "point to" and access specific regions within images. This capability is important not only to support reference-grounded VL benchmarks, but also, for practical applications that require precise within-image reasoning. We build Localized Visual Commonsense models, which allow users to specify (multiple) regions as input. We train our model by sampling localized commonsense knowledge from a large language model (LLM): specifically, we prompt an LLM to collect commonsense knowledge given a global literal image description and a local literal region description automatically generated by a set of VL models. With a separately trained critic model that selects high-quality examples, we find that training on the localized commonsense corpus can successfully distill existing VL models to support a reference-as-input interface. Empirical results and human evaluations in a zero-shot setup demonstrate that our distillation method results in more precise VL models of reasoning compared to a baseline of passing a generated referring expression to an LLM.

{{</citation>}}


### (72/110) Making Large Language Models Better Knowledge Miners for Online Marketing with Progressive Prompting Augmentation (Chunjing Gan et al., 2023)

{{<citation>}}

Chunjing Gan, Dan Yang, Binbin Hu, Ziqi Liu, Yue Shen, Zhiqiang Zhang, Jinjie Gu, Jun Zhou, Guannan Zhang. (2023)  
**Making Large Language Models Better Knowledge Miners for Online Marketing with Progressive Prompting Augmentation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, Augmentation, Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2312.05276v1)  

---


**ABSTRACT**  
Nowadays, the rapid development of mobile economy has promoted the flourishing of online marketing campaigns, whose success greatly hinges on the efficient matching between user preferences and desired marketing campaigns where a well-established Marketing-oriented Knowledge Graph (dubbed as MoKG) could serve as the critical "bridge" for preference propagation. In this paper, we seek to carefully prompt a Large Language Model (LLM) with domain-level knowledge as a better marketing-oriented knowledge miner for marketing-oriented knowledge graph construction, which is however non-trivial, suffering from several inevitable issues in real-world marketing scenarios, i.e., uncontrollable relation generation of LLMs,insufficient prompting ability of a single prompt, the unaffordable deployment cost of LLMs. To this end, we propose PAIR, a novel Progressive prompting Augmented mIning fRamework for harvesting marketing-oriented knowledge graph with LLMs. In particular, we reduce the pure relation generation to an LLM based adaptive relation filtering process through the knowledge-empowered prompting technique. Next, we steer LLMs for entity expansion with progressive prompting augmentation,followed by a reliable aggregation with comprehensive consideration of both self-consistency and semantic relatedness. In terms of online serving, we specialize in a small and white-box PAIR (i.e.,LightPAIR),which is fine-tuned with a high-quality corpus provided by a strong teacher-LLM. Extensive experiments and practical applications in audience targeting verify the effectiveness of the proposed (Light)PAIR.

{{</citation>}}


## cs.DC (2)



### (73/110) Apparate: Rethinking Early Exits to Tame Latency-Throughput Tensions in ML Serving (Yinwei Dai et al., 2023)

{{<citation>}}

Yinwei Dai, Rui Pan, Anand Iyer, Kai Li, Ravi Netravali. (2023)  
**Apparate: Rethinking Early Exits to Tame Latency-Throughput Tensions in ML Serving**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-LG, cs.DC  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.05385v1)  

---


**ABSTRACT**  
Machine learning (ML) inference platforms are tasked with balancing two competing goals: ensuring high throughput given many requests, and delivering low-latency responses to support interactive applications. Unfortunately, existing platform knobs (e.g., batch sizes) fail to ease this fundamental tension, and instead only enable users to harshly trade off one property for the other. This paper explores an alternate strategy to taming throughput-latency tradeoffs by changing the granularity at which inference is performed. We present Apparate, a system that automatically applies and manages early exits (EEs) in ML models, whereby certain inputs can exit with results at intermediate layers. To cope with the time-varying overhead and accuracy challenges that EEs bring, Apparate repurposes exits to provide continual feedback that powers several novel runtime monitoring and adaptation strategies. Apparate lowers median response latencies by 40.5-91.5% and 10.0-24.2% for diverse CV and NLP workloads, respectively, without affecting throughputs or violating tight accuracy constraints.

{{</citation>}}


### (74/110) DeltaZip: Multi-Tenant Language Model Serving via Delta Compression (Xiaozhe Yao et al., 2023)

{{<citation>}}

Xiaozhe Yao, Ana Klimovic. (2023)  
**DeltaZip: Multi-Tenant Language Model Serving via Delta Compression**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-LG, cs.DC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.05215v1)  

---


**ABSTRACT**  
Fine-tuning large language models (LLMs) for downstream tasks can greatly improve model quality, however serving many different fine-tuned LLMs concurrently for users in multi-tenant environments is challenging. Dedicating GPU memory for each model is prohibitively expensive and naively swapping large model weights in and out of GPU memory is slow. Our key insight is that fine-tuned models can be quickly swapped in and out of GPU memory by extracting and compressing the delta between each model and its pre-trained base model. We propose DeltaZip, an LLM serving system that efficiently serves multiple full-parameter fine-tuned models concurrently by aggressively compressing model deltas by a factor of $6\times$ to $8\times$ while maintaining high model quality. DeltaZip increases serving throughput by $1.5\times$ to $3\times$ and improves SLO attainment compared to a vanilla HuggingFace serving system.

{{</citation>}}


## cs.SE (5)



### (75/110) Neuron Patching: Neuron-level Model Editing on Code Generation and LLMs (Jian Gu et al., 2023)

{{<citation>}}

Jian Gu, Chunyang Chen, Aldeida Aleti. (2023)  
**Neuron Patching: Neuron-level Model Editing on Code Generation and LLMs**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-LG, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.05356v1)  

---


**ABSTRACT**  
Large Language Models are successfully adopted in software engineering, especially in code generation. Updating these models with new knowledge is very expensive, and is often required to fully realize their value. In this paper, we propose a novel and effective model editing approach, \textsc{MENT}, to patch LLMs in coding tasks. Based on the mechanism of generative LLMs, \textsc{MENT} enables model editing in next-token predictions, and further supports common coding tasks. \textsc{MENT} is effective, efficient, and reliable. It can correct a neural model by patching 1 or 2 neurons. As the pioneer work on neuron-level model editing of generative models, we formalize the editing process and introduce the involved concepts. Besides, we also introduce new measures to evaluate its generalization ability, and build a benchmark for further study. Our approach is evaluated on three coding tasks, including API-seq recommendation, line-level code generation, and pseudocode-to-code transaction. It outperforms the state-of-the-art by a significant margin on both effectiveness and efficiency measures. In addition, we demonstrate the usages of \textsc{MENT} for LLM reasoning in software engineering. By editing the LLM knowledge with \textsc{MENT}, the directly or indirectly dependent behaviors in the chain-of-thought change accordingly and automatically.

{{</citation>}}


### (76/110) INSPECT: Intrinsic and Systematic Probing Evaluation for Code Transformers (Anjan Karmakar et al., 2023)

{{<citation>}}

Anjan Karmakar, Romain Robbes. (2023)  
**INSPECT: Intrinsic and Systematic Probing Evaluation for Code Transformers**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.05092v1)  

---


**ABSTRACT**  
Pre-trained models of source code have recently been successfully applied to a wide variety of Software Engineering tasks; they have also seen some practical adoption in practice, e.g. for code completion. Yet, we still know very little about what these pre-trained models learn about source code. In this article, we use probing--simple diagnostic tasks that do not further train the models--to discover to what extent pre-trained models learn about specific aspects of source code. We use an extensible framework to define 15 probing tasks that exercise surface, syntactic, structural and semantic characteristics of source code. We probe 8 pre-trained source code models, as well as a natural language model (BERT) as our baseline. We find that models that incorporate some structural information (such as GraphCodeBERT) have a better representation of source code characteristics. Surprisingly, we find that for some probing tasks, BERT is competitive with the source code models, indicating that there are ample opportunities to improve source-code specific pre-training on the respective code characteristics. We encourage other researchers to evaluate their models with our probing task suite, so that they may peer into the hidden layers of the models and identify what intrinsic code characteristics are encoded.

{{</citation>}}


### (77/110) Challenges, Strengths, and Strategies of Software Engineers with ADHD: A Case Study (Grischa Liebel et al., 2023)

{{<citation>}}

Grischa Liebel, Noah Langlois, Kiev Gama. (2023)  
**Challenges, Strengths, and Strategies of Software Engineers with ADHD: A Case Study**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.05029v1)  

---


**ABSTRACT**  
Neurodiversity describes brain function variation in individuals, including Attention deficit hyperactivity disorder (ADHD) and Autism spectrum disorder. Neurodivergent individuals both experience challenges and exhibit strengths in the workplace. As an important disorder included under the neurodiversity term, an estimated 5.0% to 7.1% of the world population have ADHD. However, existing studies involving ADHD in the workplace are of general nature and do not focus on software engineering (SE) activities. To address this gap, we performed an exploratory qualitative case study on the experiences of people with ADHD working in SE. We find that people with ADHD struggle with several important SE-related activities, e.g., task organisation and estimation, attention to work, relation to others. Furthermore, they experience issues with physical and mental health. In terms of strengths, they exhibit, e.g., increased creative skills, perform well when solving puzzles, and have the capability to think ahead. Our findings align well with existing clinical ADHD research, and have important implications to SE practice.

{{</citation>}}


### (78/110) Out of Context: How important is Local Context in Neural Program Repair? (Julian Aron Prenner et al., 2023)

{{<citation>}}

Julian Aron Prenner, Romain Robbes. (2023)  
**Out of Context: How important is Local Context in Neural Program Repair?**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.04986v1)  

---


**ABSTRACT**  
Deep learning source code models have been applied very successfully to the problem of automated program repair. One of the standing issues is the small input window of current models which often cannot fully fit the context code required for a bug fix (e.g., method or class declarations of a project). Instead, input is often restricted to the local context, that is, the lines below and above the bug location. In this work we study the importance of this local context on repair success: how much local context is needed?; is context before or after the bug location more important? how is local context tied to the bug type? To answer these questions we train and evaluate Transformer models in many different local context configurations on three datasets and two programming languages. Our results indicate that overall repair success increases with the size of the local context (albeit not for all bug types) and confirm the common practice that roughly 50-60% of the input window should be used for context leading the bug. Our results are not only relevant for researchers working on Transformer-based APR tools but also for benchmark and dataset creators who must decide what and how much context to include in their datasets.

{{</citation>}}


### (79/110) Are We Testing or Being Tested? Exploring the Practical Applications of Large Language Models in Software Testing (Robson Santos et al., 2023)

{{<citation>}}

Robson Santos, Italo Santos, Cleyton Magalhaes, Ronnie de Souza Santos. (2023)  
**Are We Testing or Being Tested? Exploring the Practical Applications of Large Language Models in Software Testing**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.04860v1)  

---


**ABSTRACT**  
A Large Language Model (LLM) represents a cutting-edge artificial intelligence model that generates coherent content, including grammatically precise sentences, human-like paragraphs, and syntactically accurate code snippets. LLMs can play a pivotal role in software development, including software testing. LLMs go beyond traditional roles such as requirement analysis and documentation and can support test case generation, making them valuable tools that significantly enhance testing practices within the field. Hence, we explore the practical application of LLMs in software testing within an industrial setting, focusing on their current use by professional testers. In this context, rather than relying on existing data, we conducted a cross-sectional survey and collected data within real working contexts, specifically, engaging with practitioners in industrial settings. We applied quantitative and qualitative techniques to analyze and synthesize our collected data. Our findings demonstrate that LLMs effectively enhance testing documents and significantly assist testing professionals in programming tasks like debugging and test case automation. LLMs can support individuals engaged in manual testing who need to code. However, it is crucial to emphasize that, at this early stage, software testing professionals should use LLMs with caution while well-defined methods and guidelines are being built for the secure adoption of these tools.

{{</citation>}}


## cs.CY (2)



### (80/110) Contra generative AI detection in higher education assessments (Cesare G. Ardito, 2023)

{{<citation>}}

Cesare G. Ardito. (2023)  
**Contra generative AI detection in higher education assessments**  

---
Primary Category: cs.CY  
Categories: 97B40, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.05241v1)  

---


**ABSTRACT**  
This paper presents a critical analysis of generative Artificial Intelligence (AI) detection tools in higher education assessments. The rapid advancement and widespread adoption of generative AI, particularly in education, necessitates a reevaluation of traditional academic integrity mechanisms. We explore the effectiveness, vulnerabilities, and ethical implications of AI detection tools in the context of preserving academic integrity. Our study synthesises insights from various case studies, newspaper articles, and student testimonies to scrutinise the practical and philosophical challenges associated with AI detection. We argue that the reliance on detection mechanisms is misaligned with the educational landscape, where AI plays an increasingly widespread role. This paper advocates for a strategic shift towards robust assessment methods and educational policies that embrace generative AI usage while ensuring academic integrity and authenticity in assessments.

{{</citation>}}


### (81/110) Shifting Climates: Climate Change Communication from YouTube to TikTok (Arianna Pera et al., 2023)

{{<citation>}}

Arianna Pera, Luca Maria Aiello. (2023)  
**Shifting Climates: Climate Change Communication from YouTube to TikTok**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY, physics-soc-ph  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.04974v1)  

---


**ABSTRACT**  
Public discourse on critical issues such as climate change is progressively shifting to social media platforms that prioritize short-form video content. To improve our understanding of this transition, we studied the video content produced by 21 prominent YouTube creators who have expanded their influence to TikTok as information disseminators. Using dictionary-based tools and BERT-based embeddings, we analyzed the transcripts of nearly 7k climate-related videos across both platforms and the 574k comments they received. We found that, when using TikTok, creators use a more emotionally resonant, self-referential, and action-oriented language compared to YouTube. We also observed a strong semantic alignment between videos and comments, with creators who excel at diversifying their TikTok content from YouTube typically receiving responses that more closely align with their produced content. This suggests that tailored communication strategies hold greater promise in directing public discussion towards desired topics, which bears implications for the design of effective climate communication campaigns.

{{</citation>}}


## eess.IV (4)



### (82/110) MRI Scan Synthesis Methods based on Clustering and Pix2Pix (Giulia Baldini et al., 2023)

{{<citation>}}

Giulia Baldini, Melanie Schmidt, Charlotte Zäske, Liliana L. Caldeira. (2023)  
**MRI Scan Synthesis Methods based on Clustering and Pix2Pix**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.05176v1)  

---


**ABSTRACT**  
We consider a missing data problem in the context of automatic segmentation methods for Magnetic Resonance Imaging (MRI) brain scans. Usually, automated MRI scan segmentation is based on multiple scans (e.g., T1-weighted, T2-weighted, T1CE, FLAIR). However, quite often a scan is blurry, missing or otherwise unusable. We investigate the question whether a missing scan can be synthesized. We exemplify that this is in principle possible by synthesizing a T2-weighted scan from a given T1-weighted scan. Our first aim is to compute a picture that resembles the missing scan closely, measured by average mean squared error (MSE). We develop/use several methods for this, including a random baseline approach, a clustering-based method and pixel-to-pixel translation method by (Pix2Pix) which is based on conditional GANs. The lowest MSE is achieved by our clustering-based method. Our second aim is to compare the methods with respect to the affect that using the synthesized scan has on the segmentation process. For this, we use a DeepMedic model trained with the four input scan modalities named above. We replace the T2-weighted scan by the synthesized picture and evaluate the segmentations with respect to the tumor identification, using Dice scores as numerical evaluation. The evaluation shows that the segmentation works well with synthesized scans (in particular, with Pix2Pix methods) in many cases.

{{</citation>}}


### (83/110) Shape-aware Segmentation of the Placenta in BOLD Fetal MRI Time Series (S. Mazdak Abulnaga et al., 2023)

{{<citation>}}

S. Mazdak Abulnaga, Neel Dey, Sean I. Young, Eileen Pan, Katherine I. Hobgood, Clinton J. Wang, P. Ellen Grant, Esra Abaci Turk, Polina Golland. (2023)  
**Shape-aware Segmentation of the Placenta in BOLD Fetal MRI Time Series**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2312.05148v1)  

---


**ABSTRACT**  
Blood oxygen level dependent (BOLD) MRI time series with maternal hyperoxia can assess placental oxygenation and function. Measuring precise BOLD changes in the placenta requires accurate temporal placental segmentation and is confounded by fetal and maternal motion, contractions, and hyperoxia-induced intensity changes. Current BOLD placenta segmentation methods warp a manually annotated subject-specific template to the entire time series. However, as the placenta is a thin, elongated, and highly non-rigid organ subject to large deformations and obfuscated edges, existing work cannot accurately segment the placental shape, especially near boundaries. In this work, we propose a machine learning segmentation framework for placental BOLD MRI and apply it to segmenting each volume in a time series. We use a placental-boundary weighted loss formulation and perform a comprehensive evaluation across several popular segmentation objectives. Our model is trained and tested on a cohort of 91 subjects containing healthy fetuses, fetuses with fetal growth restriction, and mothers with high BMI. Biomedically, our model performs reliably in segmenting volumes in both normoxic and hyperoxic points in the BOLD time series. We further find that boundary-weighting increases placental segmentation performance by 8.3% and 6.0% Dice coefficient for the cross-entropy and signed distance transform objectives, respectively. Our code and trained model is available at https://github.com/mabulnaga/automatic-placenta-segmentation.

{{</citation>}}


### (84/110) DiffCMR: Fast Cardiac MRI Reconstruction with Diffusion Probabilistic Models (Tianqi Xiang et al., 2023)

{{<citation>}}

Tianqi Xiang, Wenjun Yue, Yiqun Lin, Jiewen Yang, Zhenkun Wang, Xiaomeng Li. (2023)  
**DiffCMR: Fast Cardiac MRI Reconstruction with Diffusion Probabilistic Models**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04853v1)  

---


**ABSTRACT**  
Performing magnetic resonance imaging (MRI) reconstruction from under-sampled k-space data can accelerate the procedure to acquire MRI scans and reduce patients' discomfort. The reconstruction problem is usually formulated as a denoising task that removes the noise in under-sampled MRI image slices. Although previous GAN-based methods have achieved good performance in image denoising, they are difficult to train and require careful tuning of hyperparameters. In this paper, we propose a novel MRI denoising framework DiffCMR by leveraging conditional denoising diffusion probabilistic models. Specifically, DiffCMR perceives conditioning signals from the under-sampled MRI image slice and generates its corresponding fully-sampled MRI image slice. During inference, we adopt a multi-round ensembling strategy to stabilize the performance. We validate DiffCMR with cine reconstruction and T1/T2 mapping tasks on MICCAI 2023 Cardiac MRI Reconstruction Challenge (CMRxRecon) dataset. Results show that our method achieves state-of-the-art performance, exceeding previous methods by a significant margin. Code is available at https://github.com/xmed-lab/DiffCMR.

{{</citation>}}


### (85/110) Image Synthesis-based Late Stage Cancer Augmentation and Semi-Supervised Segmentation for MRI Rectal Cancer Staging (Saeko Sasuga et al., 2023)

{{<citation>}}

Saeko Sasuga, Akira Kudo, Yoshiro Kitamura, Satoshi Iizuka, Edgar Simo-Serra, Atsushi Hamabe, Masayuki Ishii, Ichiro Takemasa. (2023)  
**Image Synthesis-based Late Stage Cancer Augmentation and Semi-Supervised Segmentation for MRI Rectal Cancer Staging**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Augmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.04779v1)  

---


**ABSTRACT**  
Rectal cancer is one of the most common diseases and a major cause of mortality. For deciding rectal cancer treatment plans, T-staging is important. However, evaluating the index from preoperative MRI images requires high radiologists' skill and experience. Therefore, the aim of this study is to segment the mesorectum, rectum, and rectal cancer region so that the system can predict T-stage from segmentation results. Generally, shortage of large and diverse dataset and high quality annotation are known to be the bottlenecks in computer aided diagnostics development. Regarding rectal cancer, advanced cancer images are very rare, and per-pixel annotation requires high radiologists' skill and time. Therefore, it is not feasible to collect comprehensive disease patterns in a training dataset. To tackle this, we propose two kinds of approaches of image synthesis-based late stage cancer augmentation and semi-supervised learning which is designed for T-stage prediction. In the image synthesis data augmentation approach, we generated advanced cancer images from labels. The real cancer labels were deformed to resemble advanced cancer labels by artificial cancer progress simulation. Next, we introduce a T-staging loss which enables us to train segmentation models from per-image T-stage labels. The loss works to keep inclusion/invasion relationships between rectum and cancer region consistent to the ground truth T-stage. The verification tests show that the proposed method obtains the best sensitivity (0.76) and specificity (0.80) in distinguishing between over T3 stage and underT2. In the ablation studies, our semi-supervised learning approach with the T-staging loss improved specificity by 0.13. Adding the image synthesis-based data augmentation improved the DICE score of invasion cancer area by 0.08 from baseline.

{{</citation>}}


## eess.SY (4)



### (86/110) Multi-Agent Reinforcement Learning via Distributed MPC as a Function Approximator (Samuel Mallick et al., 2023)

{{<citation>}}

Samuel Mallick, Filippo Airaldi, Azita Dabiri, Bart De Schutter. (2023)  
**Multi-Agent Reinforcement Learning via Distributed MPC as a Function Approximator**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05166v1)  

---


**ABSTRACT**  
This paper presents a novel approach to multi-agent reinforcement learning (RL) for linear systems with convex polytopic constraints. Existing works on RL has demonstrated the use of model predictive control (MPC) as a function approximator for the policy and value functions. The current paper is the first work to extend this idea to the multi-agent setting. We propose the use of a distributed MPC scheme as a function approximator, with a structure allowing for distributed learning and deployment, while maintaining the properties of centralized learning. The structured MPC scheme is introduced, and it is shown that the global policy and value functions can be approximated in a fully distributed manner. We then show that Q-learning updates can be performed distributively without introducing nonstationarity, by reconstructing a centralized learning update. The effectiveness of the approach is demonstrated on an academic example and a power systems case study.

{{</citation>}}


### (87/110) UniTSA: A Universal Reinforcement Learning Framework for V2X Traffic Signal Control (Maonan Wang et al., 2023)

{{<citation>}}

Maonan Wang, Xi Xiong, Yuheng Kan, Chengcheng Xu, Man-On Pun. (2023)  
**UniTSA: A Universal Reinforcement Learning Framework for V2X Traffic Signal Control**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05090v1)  

---


**ABSTRACT**  
Traffic congestion is a persistent problem in urban areas, which calls for the development of effective traffic signal control (TSC) systems. While existing Reinforcement Learning (RL)-based methods have shown promising performance in optimizing TSC, it is challenging to generalize these methods across intersections of different structures. In this work, a universal RL-based TSC framework is proposed for Vehicle-to-Everything (V2X) environments. The proposed framework introduces a novel agent design that incorporates a junction matrix to characterize intersection states, making the proposed model applicable to diverse intersections. To equip the proposed RL-based framework with enhanced capability of handling various intersection structures, novel traffic state augmentation methods are tailor-made for signal light control systems. Finally, extensive experimental results derived from multiple intersection configurations confirm the effectiveness of the proposed framework. The source code in this work is available at https://github.com/wmn7/Universal_Light

{{</citation>}}


### (88/110) Finite Horizon Reinforcement Learning in Solving Optimal Control of State-Dependent Switched Systems (Mi Zhou, 2023)

{{<citation>}}

Mi Zhou. (2023)  
**Finite Horizon Reinforcement Learning in Solving Optimal Control of State-Dependent Switched Systems**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.04767v1)  

---


**ABSTRACT**  
In this article, the deep deterministic policy gradient (DDPG) method is used to learn an optimal control policy of a multi-region state-dependent switched system. We observe good performance of this model-free method and explain it in a rigorous mathematical language. The performance of the learning-based methods is compared with the optimal solution given by vanilla differential dynamic programming (DDP) in three customized environments.

{{</citation>}}


### (89/110) Physics-Informed Convolutional Autoencoder for Cyber Anomaly Detection in Power Distribution Grids (Mehdi Jabbari Zideh et al., 2023)

{{<citation>}}

Mehdi Jabbari Zideh, Sarika Khushalani Solanki. (2023)  
**Physics-Informed Convolutional Autoencoder for Cyber Anomaly Detection in Power Distribution Grids**  

---
Primary Category: eess.SY  
Categories: cs-CR, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.04758v1)  

---


**ABSTRACT**  
The growing trend toward the modernization of power distribution systems has facilitated the installation of advanced measurement units and promotion of the cyber communication systems. However, these infrastructures are still prone to stealth cyber attacks. The existing data-driven anomaly detection methods suffer from a lack of knowledge about the system's physics, lack of interpretability, and scalability issues hindering their practical applications in real-world scenarios. To address these concerns, physics-informed neural networks (PINNs) were introduced. This paper proposes a multivariate physics-informed convolutional autoencoder (PIConvAE) to detect stealthy cyber-attacks in power distribution grids. The proposed model integrates the physical principles into the loss function of the neural network by applying Kirchhoff's law. Simulations are performed on the modified IEEE 13-bus and 123-bus systems using OpenDSS software to validate the efficacy of the proposed model for stealth attacks. The numerical results prove the superior performance of the proposed PIConvAE in three aspects: a) it provides more accurate results compared to the data-driven ConvAE model, b) it requires less training time to converge c) the model excels in effectively detecting a wide range of attack magnitudes making it powerful in detecting stealth attacks.

{{</citation>}}


## cs.RO (4)



### (90/110) Kraken: enabling joint trajectory prediction by utilizing Mode Transformer and Greedy Mode Processing (Daniil S. Antonenko et al., 2023)

{{<citation>}}

Daniil S. Antonenko, Stepan Konev, Yuriy Biktairov, Boris Yangel. (2023)  
**Kraken: enabling joint trajectory prediction by utilizing Mode Transformer and Greedy Mode Processing**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.05144v1)  

---


**ABSTRACT**  
Accurate and reliable motion prediction is essential for safe urban autonomy. The most prominent motion prediction approaches are based on modeling the distribution of possible future trajectories of each actor in autonomous system's vicinity. These "independent" marginal predictions might be accurate enough to properly describe casual driving situations where the prediction target is not likely to interact with other actors. They are, however, inadequate for modeling interactive situations where the actors' future trajectories are likely to intersect. To mitigate this issue we propose Kraken -- a real-time trajectory prediction model capable of approximating pairwise interactions between the actors as well as producing accurate marginal predictions. Kraken relies on a simple Greedy Mode Processing technique allowing it to convert a factorized prediction for a pair of agents into a physically-plausible joint prediction. It also utilizes the Mode Transformer module to increase the diversity of predicted trajectories and make the joint prediction more informative. We evaluate Kraken on Waymo Motion Prediction challenge where it held the first place in the Interaction leaderboard and the second place in the Motion leaderboard in October 2021.

{{</citation>}}


### (91/110) Robotic Control of the Deformation of Soft Linear Objects Using Deep Reinforcement Learning (Mélodie Hani Daniel Zakaria et al., 2023)

{{<citation>}}

Mélodie Hani Daniel Zakaria, Miguel Aranda, Laurent Lequièvre, Sébastien Lengagne, Juan Antonio Corrales Ramón, Youcef Mezouar. (2023)  
**Robotic Control of the Deformation of Soft Linear Objects Using Deep Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05056v1)  

---


**ABSTRACT**  
This paper proposes a new control framework for manipulating soft objects. A Deep Reinforcement Learning (DRL) approach is used to make the shape of a deformable object reach a set of desired points by controlling a robotic arm which manipulates it. Our framework is more easily generalizable than existing ones: it can work directly with different initial and desired final shapes without need for relearning. We achieve this by using learning parallelization, i.e., executing multiple agents in parallel on various environment instances. We focus our study on deformable linear objects. These objects are interesting in industrial and agricultural domains, yet their manipulation with robots, especially in 3D workspaces, remains challenging. We simulate the entire environment, i.e., the soft object and the robot, for the training and the testing using PyBullet and OpenAI Gym. We use a combination of state-of-the-art DRL techniques, the main ingredient being a training approach for the learning agent (i.e., the robot) based on Deep Deterministic Policy Gradient (DDPG). Our simulation results support the usefulness and enhanced generality of the proposed approach.

{{</citation>}}


### (92/110) Reinforcement Learning-Based Bionic Reflex Control for Anthropomorphic Robotic Grasping exploiting Domain Randomization (Hirakjyoti Basumatary et al., 2023)

{{<citation>}}

Hirakjyoti Basumatary, Daksh Adhar, Atharva Shrawge, Prathamesh Kanbaskar, Shyamanta M. Hazarika. (2023)  
**Reinforcement Learning-Based Bionic Reflex Control for Anthropomorphic Robotic Grasping exploiting Domain Randomization**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05023v1)  

---


**ABSTRACT**  
Achieving human-level dexterity in robotic grasping remains a challenging endeavor. Robotic hands frequently encounter slippage and deformation during object manipulation, issues rarely encountered by humans due to their sensory receptors, experiential learning, and motor memory. The emulation of the human grasping reflex within robotic hands is referred to as the ``bionic reflex". Past endeavors in the realm of bionic reflex control predominantly relied on model-based and supervised learning approaches, necessitating human intervention during thresholding and labeling tasks. In this study, we introduce an innovative bionic reflex control pipeline, leveraging reinforcement learning (RL); thereby eliminating the need for human intervention during control design. Our proposed bionic reflex controller has been designed and tested on an anthropomorphic hand, manipulating deformable objects in the PyBullet physics simulator, incorporating domain randomization (DR) for enhanced Sim2Real transferability. Our findings underscore the promise of RL as a potent tool for advancing bionic reflex control within anthropomorphic robotic hands. We anticipate that this autonomous, RL-based bionic reflex controller will catalyze the development of dependable and highly efficient robotic and prosthetic hands, revolutionizing human-robot interaction and assistive technologies.

{{</citation>}}


### (93/110) Vision-based Learning for Drones: A Survey (Jiaping Xiao et al., 2023)

{{<citation>}}

Jiaping Xiao, Rangya Zhang, Yuhang Zhang, Mir Feroskhan. (2023)  
**Vision-based Learning for Drones: A Survey**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2312.05019v1)  

---


**ABSTRACT**  
Drones as advanced cyber-physical systems are undergoing a transformative shift with the advent of vision-based learning, a field that is rapidly gaining prominence due to its profound impact on drone autonomy and functionality. Different from existing task-specific surveys, this review offers a comprehensive overview of vision-based learning in drones, emphasizing its pivotal role in enhancing their operational capabilities. We start by elucidating the fundamental principles of vision-based learning, highlighting how it significantly improves drones' visual perception and decision-making processes. We then categorize vision-based control methods into indirect, semi-direct, and end-to-end approaches from the perception-control perspective. We further explore various applications of vision-based drones with learning capabilities, ranging from single-agent systems to more complex multi-agent and heterogeneous system scenarios, and underscore the challenges and innovations characterizing each area. Finally, we explore open questions and potential solutions, paving the way for ongoing research and development in this dynamic and rapidly evolving field. With growing large language models (LLMs) and embodied intelligence, vision-based learning for drones provides a promising but challenging road towards artificial general intelligence (AGI) in 3D physical world.

{{</citation>}}


## cs.CC (1)



### (94/110) Clearing Financial Networks with Derivatives: From Intractability to Algorithms (Stavros D. Ioannidis et al., 2023)

{{<citation>}}

Stavros D. Ioannidis, Bart de Keijzer, Carmine Ventre. (2023)  
**Clearing Financial Networks with Derivatives: From Intractability to Algorithms**  

---
Primary Category: cs.CC  
Categories: cs-CC, cs-CE, cs-GT, cs.CC  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2312.05139v2)  

---


**ABSTRACT**  
Financial networks raise a significant computational challenge in identifying insolvent firms and evaluating their exposure to systemic risk. This task, known as the clearing problem, is computationally tractable when dealing with simple debt contracts. However under the presence of certain derivatives called credit default swaps (CDSes) the clearing problem is $\textsf{FIXP}$-complete. Existing techniques only show $\textsf{PPAD}$-hardness for finding an $\epsilon$-solution for the clearing problem with CDSes within an unspecified small range for $\epsilon$.   We present significant progress in both facets of the clearing problem: (i) intractability of approximate solutions; (ii) algorithms and heuristics for computable solutions. Leveraging $\textsf{Pure-Circuit}$ (FOCS'22), we provide the first explicit inapproximability bound for the clearing problem involving CDSes. Our primal contribution is a reduction from $\textsf{Pure-Circuit}$ which establishes that finding approximate solutions is $\textsf{PPAD}$-hard within a range of roughly 5%.   To alleviate the complexity of the clearing problem, we identify two meaningful restrictions of the class of financial networks motivated by regulations: (i) the presence of a central clearing authority; and (ii) the restriction to covered CDSes. We provide the following results: (i.) The $\textsf{PPAD}$-hardness of approximation persists when central clearing authorities are introduced; (ii.) An optimisation-based method for solving the clearing problem with central clearing authorities; (iii.) A polynomial-time algorithm when the two restrictions hold simultaneously.

{{</citation>}}


## cs.NI (3)



### (95/110) ScalO-RAN: Energy-aware Network Intelligence Scaling in Open RAN (Stefano Maxenti et al., 2023)

{{<citation>}}

Stefano Maxenti, Salvatore D'Oro, Leonardo Bonati, Michele Polese, Antonio Capone, Tommaso Melodia. (2023)  
**ScalO-RAN: Energy-aware Network Intelligence Scaling in Open RAN**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.05096v1)  

---


**ABSTRACT**  
Network virtualization, software-defined infrastructure, and orchestration are pivotal elements in contemporary networks, yielding new vectors for optimization and novel capabilities. In line with these principles, O-RAN presents an avenue to bypass vendor lock-in, circumvent vertical configurations, enable network programmability, and facilitate integrated Artificial Intelligence (AI) support. Moreover, modern container orchestration frameworks (e.g., Kubernetes, Red Hat OpenShift) simplify the way cellular base stations, as well as the newly introduced RAN Intelligent Controllers (RICs), are deployed, managed, and orchestrated. While this enables cost reduction via infrastructure sharing, it also makes it more challenging to meet O-RAN control latency requirements, especially during peak resource utilization. To address this problem, we propose ScalO-RAN, a control framework rooted in optimization and designed as an O-RAN rApp that allocates and scales AI-based O-RAN applications (xApps, rApps, dApps) to: (i) abide by application-specific latency requirements, and (ii) monetize the shared infrastructure while reducing energy consumption. We prototype ScalO-RAN on an OpenShift cluster with base stations, RIC, and a set of AI-based xApps deployed as micro-services. We evaluate ScalO-RAN both numerically and experimentally. Our results show that ScalO-RAN can optimally allocate and distribute O-RAN applications within available computing nodes to accommodate even stringent latency requirements. More importantly, we show that scaling O-RAN applications is primarily a time-constrained problem rather than a resource-constrained one, where scaling policies must account for stringent inference time of AI applications, and not only how many resources they consume.

{{</citation>}}


### (96/110) Physical-Layer Semantic-Aware Network for Zero-Shot Wireless Sensing (Huixiang Zhu et al., 2023)

{{<citation>}}

Huixiang Zhu, Yong Xiao, Yingyu Li, Guangming Shi, Walid Saad. (2023)  
**Physical-Layer Semantic-Aware Network for Zero-Shot Wireless Sensing**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-NI, cs.NI  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.05043v1)  

---


**ABSTRACT**  
Device-free wireless sensing has recently attracted significant interest due to its potential to support a wide range of immersive human-machine interactive applications. However, data heterogeneity in wireless signals and data privacy regulation of distributed sensing have been considered as the major challenges that hinder the wide applications of wireless sensing in large area networking systems. Motivated by the observation that signals recorded by wireless receivers are closely related to a set of physical-layer semantic features, in this paper we propose a novel zero-shot wireless sensing solution that allows models constructed in one or a limited number of locations to be directly transferred to other locations without any labeled data. We develop a novel physical-layer semantic-aware network (pSAN) framework to characterize the correlation between physical-layer semantic features and the sensing data distributions across different receivers. We then propose a pSAN-based zero-shot learning solution in which each receiver can obtain a location-specific gesture recognition model by directly aggregating the already constructed models of other receivers. We theoretically prove that models obtained by our proposed solution can approach the optimal model without requiring any local model training. Experimental results once again verify that the accuracy of models derived by our proposed solution matches that of the models trained by the real labeled data based on supervised learning approach.

{{</citation>}}


### (97/110) A First Look at 5G Core Deployments on Public Cloud: Performance Evaluation of Control and User Planes (Tolga O. Atalay et al., 2023)

{{<citation>}}

Tolga O. Atalay, Dragoslav Stojadinovic, Alireza Famili, Angelos Stavrou, Haining Wang. (2023)  
**A First Look at 5G Core Deployments on Public Cloud: Performance Evaluation of Control and User Planes**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AWS, Amazon  
[Paper Link](http://arxiv.org/abs/2312.04833v1)  

---


**ABSTRACT**  
The Fifth Generation (5G) mobile core network is designed as a set of Virtual Network Functions (VNFs) hosted on Commercial-Off-the-Shelf (COTS) hardware. This creates a growing demand for general-purpose compute resources as 5G deployments continue to expand. Given their elastic infrastructure, cloud services such as Amazon Web Services (AWS) are attractive platforms to address this need. Therefore, it is crucial to understand the control and user plane Quality of Service (QoS) performance associated with deploying the 5G core on top of a public cloud. To account for both software and communication costs, we build a 5G testbed using open-source components spanning multiple locations within AWS. We present an operational breakdown of the performance overhead for various 5G use cases using different core deployment strategies. Our results indicate that moving specific VNFs into edge regions reduces the latency overhead for key 5G operations. Furthermore, we instantiated multiple user plane connections between availability zones and edge regions with different traffic loads. We observed that the deterioration of connection quality varies depending on traffic loads and is use case specific. Ultimately, our findings provide new insights for Mobile Virtual Network Operators (MVNOs) for optimal placements of their 5G core functions.

{{</citation>}}


## cs.CR (6)



### (98/110) A stacked ensemble learning IDS model for Software-defined VANET (Shakil Ibne Ahsan et al., 2023)

{{<citation>}}

Shakil Ibne Ahsan, Phil Legg, S M Iftekharul Alam. (2023)  
**A stacked ensemble learning IDS model for Software-defined VANET**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2312.04956v2)  

---


**ABSTRACT**  
Intrusion Detection Systems (IDS) are widely employed to detect and mitigate external network security events. VANETs (Vehicle ad-hoc Networks) are evolving, especially with the development of Connected Autonomous Vehicles (CAVs). So, it is crucial to assess how traditional IDS approaches can be utilised for emerging technologies. To address this concern, our work presents a stacked ensemble learning approach for IDS, which combines multiple machine learning algorithms to detect threats more effectively than single algorithm methods. Using the CICIDS2017 and the VeReMi benchmark data sets, we compare the performance of our approach with existing machine learning methods and find that it is more accurate at identifying threats. Our method also incorporates hyperparameter optimization and feature selection to improve its performance further. Overall, our results suggest that stacked ensemble learning is a promising technique for enhancing the effectiveness of IDS.

{{</citation>}}


### (99/110) Canaries and Whistles: Resilient Drone Communication Networks with (or without) Deep Reinforcement Learning (Chris Hicks et al., 2023)

{{<citation>}}

Chris Hicks, Vasilios Mavroudis, Myles Foley, Thomas Davies, Kate Highnam, Tim Watson. (2023)  
**Canaries and Whistles: Resilient Drone Communication Networks with (or without) Deep Reinforcement Learning**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs.CR  
Keywords: Drone, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.04940v1)  

---


**ABSTRACT**  
Communication networks able to withstand hostile environments are critically important for disaster relief operations. In this paper, we consider a challenging scenario where drones have been compromised in the supply chain, during their manufacture, and harbour malicious software capable of wide-ranging and infectious disruption. We investigate multi-agent deep reinforcement learning as a tool for learning defensive strategies that maximise communications bandwidth despite continual adversarial interference. Using a public challenge for learning network resilience strategies, we propose a state-of-the-art expert technique and study its superiority over deep reinforcement learning agents. Correspondingly, we identify three specific methods for improving the performance of our learning-based agents: (1) ensuring each observation contains the necessary information, (2) using expert agents to provide a curriculum for learning, and (3) paying close attention to reward. We apply our methods and present a new mixed strategy enabling expert and learning-based agents to work together and improve on all prior results.

{{</citation>}}


### (100/110) Critical Analysis of 5G Networks Traffic Intrusion using PCA, t-SNE and UMAP Visualization and Classifying Attacks (Humera Ghani et al., 2023)

{{<citation>}}

Humera Ghani, Shahram Salekzamankhani, Bal Virdee. (2023)  
**Critical Analysis of 5G Networks Traffic Intrusion using PCA, t-SNE and UMAP Visualization and Classifying Attacks**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.04864v1)  

---


**ABSTRACT**  
Networks, threat models, and malicious actors are advancing quickly. With the increased deployment of the 5G networks, the security issues of the attached 5G physical devices have also increased. Therefore, artificial intelligence based autonomous end-to-end security design is needed that can deal with incoming threats by detecting network traffic anomalies. To address this requirement, in this research, we used a recently published 5G traffic dataset, 5G-NIDD, to detect network traffic anomalies using machine and deep learning approaches. First, we analyzed the dataset using three visualization techniques: t-Distributed Stochastic Neighbor Embedding (t-SNE), Uniform Manifold Approximation and Projection (UMAP), and Principal Component Analysis (PCA). Second, we reduced the data dimensionality using mutual information and PCA techniques. Third, we solve the class imbalance issue by inserting synthetic records of minority classes. Last, we performed classification using six different classifiers and presented the evaluation metrics. We received the best results when K-Nearest Neighbors classifier was used: accuracy (97.2%), detection rate (96.7%), and false positive rate (2.2%).

{{</citation>}}


### (101/110) Using Program Knowledge Graph to Uncover Software Vulnerabilities (M. Xie et al., 2023)

{{<citation>}}

M. Xie, T. Rahat, W. Wang, Y. Tian. (2023)  
**Using Program Knowledge Graph to Uncover Software Vulnerabilities**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.04818v1)  

---


**ABSTRACT**  
In an increasingly interconnected and data-driven world, the importance of robust security measures cannot be overstated. A knowledge graph constructed with information extracted from the system along with the desired security behavior can be utilized to identify complex security vulnerabilities hidden underneath the systems. Unfortunately, existing security knowledge graphs are constructed from coarse-grained information extracted from publicly available vulnerability reports, which are not equipped to check actual security violations in real-world system implementations. In this poster, we present a novel approach of using Program Knowledge Graph that is embedded with fine-grained execution information of the systems (e.g., callgraph, data-flow, etc.) along with information extracted from the public vulnerability and weakness datasets (e.g., CVE and CWE). We further demonstrate that our custom security knowledge graph can be checked against the standard queries generated by LLM, providing a powerful way to identify security vulnerabilities and weaknesses in critical systems.

{{</citation>}}


### (102/110) Exploring the Limits of ChatGPT in Software Security Applications (Fangzhou Wu et al., 2023)

{{<citation>}}

Fangzhou Wu, Qingzhao Zhang, Ati Priya Bajaj, Tiffany Bao, Ning Zhang, Ruoyu "Fish" Wang, Chaowei Xiao. (2023)  
**Exploring the Limits of ChatGPT in Software Security Applications**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI, ChatGPT, GPT, GPT-3.5, GPT-4, Security  
[Paper Link](http://arxiv.org/abs/2312.05275v1)  

---


**ABSTRACT**  
Large language models (LLMs) have undergone rapid evolution and achieved remarkable results in recent times. OpenAI's ChatGPT, backed by GPT-3.5 or GPT-4, has gained instant popularity due to its strong capability across a wide range of tasks, including natural language tasks, coding, mathematics, and engaging conversations. However, the impacts and limits of such LLMs in system security domain are less explored. In this paper, we delve into the limits of LLMs (i.e., ChatGPT) in seven software security applications including vulnerability detection/repair, debugging, debloating, decompilation, patching, root cause analysis, symbolic execution, and fuzzing. Our exploration reveals that ChatGPT not only excels at generating code, which is the conventional application of language models, but also demonstrates strong capability in understanding user-provided commands in natural languages, reasoning about control and data flows within programs, generating complex data structures, and even decompiling assembly code. Notably, GPT-4 showcases significant improvements over GPT-3.5 in most security tasks. Also, certain limitations of ChatGPT in security-related tasks are identified, such as its constrained ability to process long code contexts.

{{</citation>}}


### (103/110) Make Them Spill the Beans! Coercive Knowledge Extraction from (Production) LLMs (Zhuo Zhang et al., 2023)

{{<citation>}}

Zhuo Zhang, Guangyu Shen, Guanhong Tao, Siyuan Cheng, Xiangyu Zhang. (2023)  
**Make Them Spill the Beans! Coercive Knowledge Extraction from (Production) LLMs**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04782v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are now widely used in various applications, making it crucial to align their ethical standards with human values. However, recent jail-breaking methods demonstrate that this alignment can be undermined using carefully constructed prompts. In our study, we reveal a new threat to LLM alignment when a bad actor has access to the model's output logits, a common feature in both open-source LLMs and many commercial LLM APIs (e.g., certain GPT models). It does not rely on crafting specific prompts. Instead, it exploits the fact that even when an LLM rejects a toxic request, a harmful response often hides deep in the output logits. By forcefully selecting lower-ranked output tokens during the auto-regressive generation process at a few critical output positions, we can compel the model to reveal these hidden responses. We term this process model interrogation. This approach differs from and outperforms jail-breaking methods, achieving 92% effectiveness compared to 62%, and is 10 to 20 times faster. The harmful content uncovered through our method is more relevant, complete, and clear. Additionally, it can complement jail-breaking strategies, with which results in further boosting attack performance. Our findings indicate that interrogation can extract toxic knowledge even from models specifically designed for coding tasks.

{{</citation>}}


## cs.IR (2)



### (104/110) Illicit Darkweb Classification via Natural-language Processing: Classifying Illicit Content of Webpages based on Textual Information (Giuseppe Cascavilla et al., 2023)

{{<citation>}}

Giuseppe Cascavilla, Gemma Catolino, Mirella Sangiovanni. (2023)  
**Illicit Darkweb Classification via Natural-language Processing: Classifying Illicit Content of Webpages based on Textual Information**  

---
Primary Category: cs.IR  
Categories: cs-CY, cs-IR, cs.IR  
Keywords: BERT, LSTM, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.04944v1)  

---


**ABSTRACT**  
This work aims at expanding previous works done in the context of illegal activities classification, performing three different steps. First, we created a heterogeneous dataset of 113995 onion sites and dark marketplaces. Then, we compared pre-trained transferable models, i.e., ULMFit (Universal Language Model Fine-tuning), Bert (Bidirectional Encoder Representations from Transformers), and RoBERTa (Robustly optimized BERT approach) with a traditional text classification approach like LSTM (Long short-term memory) neural networks. Finally, we developed two illegal activities classification approaches, one for illicit content on the Dark Web and one for identifying the specific types of drugs. Results show that Bert obtained the best approach, classifying the dark web's general content and the types of Drugs with 96.08% and 91.98% of accuracy.

{{</citation>}}


### (105/110) CAR: Consolidation, Augmentation and Regulation for Recipe Retrieval (Fangzhou Song et al., 2023)

{{<citation>}}

Fangzhou Song, Bin Zhu, Yanbin Hao, Shuo Wang, Xiangnan He. (2023)  
**CAR: Consolidation, Augmentation and Regulation for Recipe Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2312.04763v1)  

---


**ABSTRACT**  
Learning recipe and food image representation in common embedding space is non-trivial but crucial for cross-modal recipe retrieval. In this paper, we propose CAR framework with three novel techniques, i.e., Consolidation, Augmentation and Regulation, for cross-modal recipe retrieval. We introduce adapter layers to consolidate pre-trained CLIP model with much less computation cost than fully cumbersome fine-tuning all the parameters. Furthermore, leveraging on the strong capability of foundation models (i.e., SAM and LLM), we propose to augment recipe and food image by extracting information related to the counterpart. SAM generates image segments corresponding to ingredients in the recipe, while LLM produces a visual imagination description from the recipe, aiming to capture the visual cues of a food image. In addition, we introduce circle loss to regulate cross-modal embedding space, which assigns different penalties for positive and negative pairs. With the extra augmented data from recipe and image, multi-level circle loss is proposed, which applies circle loss not only to original image-recipe pairs, but also to image segments and recipe, visual imagination description and food image as well as any two sections within a recipe. On Recipe1M dataset, our proposed CAR outperforms all the existing methods by a large margin. Extensive ablation studies are conducted to validate the effectiveness of each component of CAR. We will make our code and models publicly available.

{{</citation>}}


## cs.HC (1)



### (106/110) Understanding Teacher Perspectives and Experiences after Deployment of AI Literacy Curriculum in Middle-school Classrooms (Prerna Ravi et al., 2023)

{{<citation>}}

Prerna Ravi, Annalisa Broski, Glenda Stump, Hal Abelson, Eric Klopfer, Cynthia Breazeal. (2023)  
**Understanding Teacher Perspectives and Experiences after Deployment of AI Literacy Curriculum in Middle-school Classrooms**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CY, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04839v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) and its associated applications are ubiquitous in today's world, making it imperative that students and their teachers understand how it works and the ramifications arising from its usage. In this study, we investigate the experiences of seven teachers following their implementation of modules from the MIT RAICA (Responsible AI for Computational Action) curriculum. Through semi-structured interviews, we investigated their instructional strategies as they engaged with the AI curriculum in their classroom, how their teaching and learning beliefs about AI evolved with the curriculum as well as how those beliefs impacted their implementation of the curriculum. Our analysis suggests that the AI modules not only expanded our teachers' knowledge in the field, but also prompted them to recognize its daily applications and their ethical and societal implications, so that they could better engage with the content they deliver to students. Teachers were able to leverage their own interdisciplinary backgrounds to creatively introduce foundational AI topics to students to maximize engagement and playful learning. Our teachers advocated their need for better external support when navigating technological resources, additional time for preparation given the novelty of the curriculum, more flexibility within curriculum timelines, and additional accommodations for students of determination. Our findings provide valuable insights for enhancing future iterations of AI literacy curricula and teacher professional development (PD) resources.

{{</citation>}}


## cs.ET (1)



### (107/110) Thermodynamic Computing System for AI Applications (Denis Melanson et al., 2023)

{{<citation>}}

Denis Melanson, Mohammad Abu Khater, Maxwell Aifer, Kaelan Donatella, Max Hunter Gordon, Thomas Ahle, Gavin Crooks, Antonio J. Martinez, Faris Sbahi, Patrick J. Coles. (2023)  
**Thermodynamic Computing System for AI Applications**  

---
Primary Category: cs.ET  
Categories: cond-mat-stat-mech, cs-AI, cs-ET, cs.ET  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04836v1)  

---


**ABSTRACT**  
Recent breakthroughs in artificial intelligence (AI) algorithms have highlighted the need for novel computing hardware in order to truly unlock the potential for AI. Physics-based hardware, such as thermodynamic computing, has the potential to provide a fast, low-power means to accelerate AI primitives, especially generative AI and probabilistic AI. In this work, we present the first continuous-variable thermodynamic computer, which we call the stochastic processing unit (SPU). Our SPU is composed of RLC circuits, as unit cells, on a printed circuit board, with 8 unit cells that are all-to-all coupled via switched capacitances. It can be used for either sampling or linear algebra primitives, and we demonstrate Gaussian sampling and matrix inversion on our hardware. The latter represents the first thermodynamic linear algebra experiment. We also illustrate the applicability of the SPU to uncertainty quantification for neural network classification. We envision that this hardware, when scaled up in size, will have significant impact on accelerating various probabilistic AI applications.

{{</citation>}}


## cs.MA (1)



### (108/110) Attention-Guided Contrastive Role Representations for Multi-Agent Reinforcement Learning (Zican Hu et al., 2023)

{{<citation>}}

Zican Hu, Zongzhang Zhang, Huaxiong Li, Chunlin Chen, Hongyu Ding, Zhi Wang. (2023)  
**Attention-Guided Contrastive Role Representations for Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs.MA  
Keywords: Attention, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.04819v1)  

---


**ABSTRACT**  
Real-world multi-agent tasks usually involve dynamic team composition with the emergence of roles, which should also be a key to efficient cooperation in multi-agent reinforcement learning (MARL). Drawing inspiration from the correlation between roles and agent's behavior patterns, we propose a novel framework of Attention-guided COntrastive Role representation learning for MARL (ACORM) to promote behavior heterogeneity, knowledge transfer, and skillful coordination across agents. First, we introduce mutual information maximization to formalize role representation learning, derive a contrastive learning objective, and concisely approximate the distribution of negative pairs. Second, we leverage an attention mechanism to prompt the global state to attend to learned role representations in value decomposition, implicitly guiding agent coordination in a skillful role space to yield more expressive credit assignment. Experiments and visualizations on challenging StarCraft II micromanagement tasks demonstrate the state-of-the-art performance of our method and its advantages over existing approaches. Our code is available at https://github.com/NJU-RL/ACORM}{https://github.com/NJU-RL/ACORM.

{{</citation>}}


## cs.GT (1)



### (109/110) AI safety by debate via regret minimization (Xinyi Chen et al., 2023)

{{<citation>}}

Xinyi Chen, Angelica Chen, Dean Foster, Elad Hazan. (2023)  
**AI safety by debate via regret minimization**  

---
Primary Category: cs.GT  
Categories: cs-AI, cs-GT, cs.GT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04792v1)  

---


**ABSTRACT**  
We consider the setting of AI safety by debate as a repeated game. We consider the question of efficient regret minimization in this setting, when the players are either AIs or humans, equipped with access to computationally superior AIs. In such a setting, we characterize when internal and external regret can be minimized efficiently. We conclude with conditions in which a sequence of strategies converges to a correlated equilibrium.

{{</citation>}}


## hep-ex (1)



### (110/110) Induced Generative Adversarial Particle Transformers (Anni Li et al., 2023)

{{<citation>}}

Anni Li, Venkat Krishnamohan, Raghav Kansal, Rounak Sen, Steven Tsan, Zhaoyu Zhang, Javier Duarte. (2023)  
**Induced Generative Adversarial Particle Transformers**  

---
Primary Category: hep-ex  
Categories: cs-LG, hep-ex, hep-ex, physics-data-an  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.04757v1)  

---


**ABSTRACT**  
In high energy physics (HEP), machine learning methods have emerged as an effective way to accurately simulate particle collisions at the Large Hadron Collider (LHC). The message-passing generative adversarial network (MPGAN) was the first model to simulate collisions as point, or ``particle'', clouds, with state-of-the-art results, but suffered from quadratic time complexity. Recently, generative adversarial particle transformers (GAPTs) were introduced to address this drawback; however, results did not surpass MPGAN. We introduce induced GAPT (iGAPT) which, by integrating ``induced particle-attention blocks'' and conditioning on global jet attributes, not only offers linear time complexity but is also able to capture intricate jet substructure, surpassing MPGAN in many metrics. Our experiments demonstrate the potential of iGAPT to simulate complex HEP data accurately and efficiently.

{{</citation>}}
