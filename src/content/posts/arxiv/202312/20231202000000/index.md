---
draft: false
title: "arXiv @ 2023.12.02"
date: 2023-12-02
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.02"
    identifier: arxiv_20231202
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (58)](#cscv-58)
- [cs.CY (4)](#cscy-4)
- [cs.LG (30)](#cslg-30)
- [cs.CL (20)](#cscl-20)
- [cs.SE (5)](#csse-5)
- [cs.ET (1)](#cset-1)
- [stat.AP (1)](#statap-1)
- [quant-ph (3)](#quant-ph-3)
- [astro-ph.IM (1)](#astro-phim-1)
- [cs.IT (3)](#csit-3)
- [eess.IV (5)](#eessiv-5)
- [eess.SY (2)](#eesssy-2)
- [eess.SP (1)](#eesssp-1)
- [stat.ME (1)](#statme-1)
- [cs.IR (3)](#csir-3)
- [stat.ML (1)](#statml-1)
- [cs.AR (1)](#csar-1)
- [cs.SI (3)](#cssi-3)
- [cs.CR (3)](#cscr-3)
- [cs.AI (1)](#csai-1)
- [cs.DC (1)](#csdc-1)
- [cs.SD (1)](#cssd-1)
- [cs.CE (1)](#csce-1)
- [cs.HC (2)](#cshc-2)
- [physics.chem-ph (1)](#physicschem-ph-1)
- [physics.ao-ph (1)](#physicsao-ph-1)
- [cs.MM (2)](#csmm-2)

## cs.CV (58)



### (1/156) Brainformer: Modeling MRI Brain Functions to Machine Vision (Xuan-Bac Nguyen et al., 2023)

{{<citation>}}

Xuan-Bac Nguyen, Xin Li, Samee U. Khan, Khoa Luu. (2023)  
**Brainformer: Modeling MRI Brain Functions to Machine Vision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.00236v1)  

---


**ABSTRACT**  
"Perception is reality". Human perception plays a vital role in forming beliefs and understanding reality. Exploring how the human brain works in the visual system facilitates bridging the gap between human visual perception and computer vision models. However, neuroscientists study the brain via Neuroimaging, i.e., Functional Magnetic Resonance Imaging (fMRI), to discover the brain's functions. These approaches face interpretation challenges where fMRI data can be complex and require expertise. Therefore, neuroscientists make inferences about cognitive processes based on patterns of brain activities, which can lead to potential misinterpretation or limited functional understanding. In this work, we first present a simple yet effective Brainformer approach, a novel Transformer-based framework, to analyze the patterns of fMRI in the human perception system from the machine learning perspective. Secondly, we introduce a novel mechanism incorporating fMRI, which represents the human brain activities, as the supervision for the machine vision model. This work also introduces a novel perspective on transferring knowledge from human perception to neural networks. Through our experiments, we demonstrated that by leveraging fMRI information, the machine vision model can achieve potential results compared to the current State-of-the-art methods in various image recognition tasks.

{{</citation>}}


### (2/156) Raising the Bar of AI-generated Image Detection with CLIP (Davide Cozzolino et al., 2023)

{{<citation>}}

Davide Cozzolino, Giovanni Poggi, Riccardo Corvi, Matthias Nießner, Luisa Verdoliva. (2023)  
**Raising the Bar of AI-generated Image Detection with CLIP**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00195v1)  

---


**ABSTRACT**  
Aim of this work is to explore the potential of pre-trained vision-language models (VLMs) for universal detection of AI-generated images. We develop a lightweight detection strategy based on CLIP features and study its performance in a wide variety of challenging scenarios. We find that, unlike previous belief, it is neither necessary nor convenient to use a large domain-specific dataset for training. On the contrary, by using only a handful of example images from a single generative model, a CLIP-based detector exhibits a surprising generalization ability and high robustness across several different architectures, including recent commercial tools such as Dalle-3, Midjourney v5, and Firefly. We match the SoTA on in-distribution data, and improve largely above it in terms of generalization to out-of-distribution data (+6% in terms of AUC) and robustness to impaired/laundered data (+13%). Our project is available at https://grip-unina.github.io/ClipBased-SyntheticImageDetection/

{{</citation>}}


### (3/156) Fool the Hydra: Adversarial Attacks against Multi-view Object Detection Systems (Bilel Tarchoun et al., 2023)

{{<citation>}}

Bilel Tarchoun, Quazi Mishkatul Alam, Nael Abu-Ghazaleh, Ihsen Alouani. (2023)  
**Fool the Hydra: Adversarial Attacks against Multi-view Object Detection Systems**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs.CV  
Keywords: Adversarial Attack, Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2312.00173v1)  

---


**ABSTRACT**  
Adversarial patches exemplify the tangible manifestation of the threat posed by adversarial attacks on Machine Learning (ML) models in real-world scenarios. Robustness against these attacks is of the utmost importance when designing computer vision applications, especially for safety-critical domains such as CCTV systems. In most practical situations, monitoring open spaces requires multi-view systems to overcome acquisition challenges such as occlusion handling. Multiview object systems are able to combine data from multiple views, and reach reliable detection results even in difficult environments. Despite its importance in real-world vision applications, the vulnerability of multiview systems to adversarial patches is not sufficiently investigated. In this paper, we raise the following question: Does the increased performance and information sharing across views offer as a by-product robustness to adversarial patches? We first conduct a preliminary analysis showing promising robustness against off-the-shelf adversarial patches, even in an extreme setting where we consider patches applied to all views by all persons in Wildtrack benchmark. However, we challenged this observation by proposing two new attacks: (i) In the first attack, targeting a multiview CNN, we maximize the global loss by proposing gradient projection to the different views and aggregating the obtained local gradients. (ii) In the second attack, we focus on a Transformer-based multiview framework. In addition to the focal loss, we also maximize the transformer-specific loss by dissipating its attention blocks. Our results show a large degradation in the detection performance of victim multiview systems with our first patch attack reaching an attack success rate of 73% , while our second proposed attack reduced the performance of its target detector by 62%

{{</citation>}}


### (4/156) GraphDreamer: Compositional 3D Scene Synthesis from Scene Graphs (Gege Gao et al., 2023)

{{<citation>}}

Gege Gao, Weiyang Liu, Anpei Chen, Andreas Geiger, Bernhard Schölkopf. (2023)  
**GraphDreamer: Compositional 3D Scene Synthesis from Scene Graphs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs-LG, cs.CV  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.00093v1)  

---


**ABSTRACT**  
As pretrained text-to-image diffusion models become increasingly powerful, recent efforts have been made to distill knowledge from these text-to-image pretrained models for optimizing a text-guided 3D model. Most of the existing methods generate a holistic 3D model from a plain text input. This can be problematic when the text describes a complex scene with multiple objects, because the vectorized text embeddings are inherently unable to capture a complex description with multiple entities and relationships. Holistic 3D modeling of the entire scene further prevents accurate grounding of text entities and concepts. To address this limitation, we propose GraphDreamer, a novel framework to generate compositional 3D scenes from scene graphs, where objects are represented as nodes and their interactions as edges. By exploiting node and edge information in scene graphs, our method makes better use of the pretrained text-to-image diffusion model and is able to fully disentangle different objects without image-level supervision. To facilitate modeling of object-wise relationships, we use signed distance fields as representation and impose a constraint to avoid inter-penetration of objects. To avoid manual scene graph creation, we design a text prompt for ChatGPT to generate scene graphs based on text inputs. We conduct both qualitative and quantitative experiments to validate the effectiveness of GraphDreamer in generating high-fidelity compositional 3D scenes with disentangled object entities.

{{</citation>}}


### (5/156) Dataset Distillation in Large Data Era (Zeyuan Yin et al., 2023)

{{<citation>}}

Zeyuan Yin, Zhiqiang Shen. (2023)  
**Dataset Distillation in Large Data Era**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.18838v1)  

---


**ABSTRACT**  
Dataset distillation aims to generate a smaller but representative subset from a large dataset, which allows a model to be trained efficiently, meanwhile evaluating on the original testing data distribution to achieve decent performance. Many prior works have aimed to align with diverse aspects of the original datasets, such as matching the training weight trajectories, gradient, feature/BatchNorm distributions, etc. In this work, we show how to distill various large-scale datasets such as full ImageNet-1K/21K under a conventional input resolution of 224$\times$224 to achieve the best accuracy over all previous approaches, including SRe$^2$L, TESLA and MTT. To achieve this, we introduce a simple yet effective ${\bf C}$urriculum ${\bf D}$ata ${\bf A}$ugmentation ($\texttt{CDA}$) during data synthesis that obtains the accuracy on large-scale ImageNet-1K and 21K with 63.2% under IPC (Images Per Class) 50 and 36.1% under IPC 20, respectively. Finally, we show that, by integrating all our enhancements together, the proposed model beats the current state-of-the-art by more than 4% Top-1 accuracy on ImageNet-1K/21K and for the first time, reduces the gap to its full-data training counterpart to less than absolute 15%. Moreover, this work represents the inaugural success in dataset distillation on larger-scale ImageNet-21K under the standard 224$\times$224 resolution. Our code and distilled ImageNet-21K dataset of 20 IPC, 2K recovery budget are available at https://github.com/VILA-Lab/SRe2L/tree/main/CDA.

{{</citation>}}


### (6/156) Just Add $π$! Pose Induced Video Transformers for Understanding Activities of Daily Living (Dominick Reilly et al., 2023)

{{<citation>}}

Dominick Reilly, Srijan Das. (2023)  
**Just Add $π$! Pose Induced Video Transformers for Understanding Activities of Daily Living**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.18840v1)  

---


**ABSTRACT**  
Video transformers have become the de facto standard for human action recognition, yet their exclusive reliance on the RGB modality still limits their adoption in certain domains. One such domain is Activities of Daily Living (ADL), where RGB alone is not sufficient to distinguish between visually similar actions, or actions observed from multiple viewpoints. To facilitate the adoption of video transformers for ADL, we hypothesize that the augmentation of RGB with human pose information, known for its sensitivity to fine-grained motion and multiple viewpoints, is essential. Consequently, we introduce the first Pose Induced Video Transformer: PI-ViT (or $\pi$-ViT), a novel approach that augments the RGB representations learned by video transformers with 2D and 3D pose information. The key elements of $\pi$-ViT are two plug-in modules, 2D Skeleton Induction Module and 3D Skeleton Induction Module, that are responsible for inducing 2D and 3D pose information into the RGB representations. These modules operate by performing pose-aware auxiliary tasks, a design choice that allows $\pi$-ViT to discard the modules during inference. Notably, $\pi$-ViT achieves the state-of-the-art performance on three prominent ADL datasets, encompassing both real-world and large-scale RGB-D datasets, without requiring poses or additional computational overhead at inference.

{{</citation>}}


### (7/156) PoseGPT: Chatting about 3D Human Pose (Yao Feng et al., 2023)

{{<citation>}}

Yao Feng, Jing Lin, Sai Kumar Dwivedi, Yu Sun, Priyanka Patel, Michael J. Black. (2023)  
**PoseGPT: Chatting about 3D Human Pose**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.18836v1)  

---


**ABSTRACT**  
We introduce PoseGPT, a framework employing Large Language Models (LLMs) to understand and reason about 3D human poses from images or textual descriptions. Our work is motivated by the human ability to intuitively understand postures from a single image or a brief description, a process that intertwines image interpretation, world knowledge, and an understanding of body language. Traditional human pose estimation methods, whether image-based or text-based, often lack holistic scene comprehension and nuanced reasoning, leading to a disconnect between visual data and its real-world implications. PoseGPT addresses these limitations by embedding SMPL poses as a distinct signal token within a multi-modal LLM, enabling direct generation of 3D body poses from both textual and visual inputs. This approach not only simplifies pose prediction but also empowers LLMs to apply their world knowledge in reasoning about human poses, fostering two advanced tasks: speculative pose generation and reasoning about pose estimation. These tasks involve reasoning about humans to generate 3D poses from subtle text queries, possibly accompanied by images. We establish benchmarks for these tasks, moving beyond traditional 3D pose generation and estimation methods. Our results show that PoseGPT outperforms existing multimodal LLMs and task-sepcific methods on these newly proposed tasks. Furthermore, PoseGPT's ability to understand and generate 3D human poses based on complex reasoning opens new directions in human pose analysis.

{{</citation>}}


### (8/156) One-step Diffusion with Distribution Matching Distillation (Tianwei Yin et al., 2023)

{{<citation>}}

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T. Freeman, Taesung Park. (2023)  
**One-step Diffusion with Distribution Matching Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.18828v3)  

---


**ABSTRACT**  
Diffusion models generate high-quality images but require dozens of forward passes. We introduce Distribution Matching Distillation (DMD), a procedure to transform a diffusion model into a one-step image generator with minimal impact on image quality. We enforce the one-step image generator match the diffusion model at distribution level, by minimizing an approximate KL divergence whose gradient can be expressed as the difference between 2 score functions, one of the target distribution and the other of the synthetic distribution being produced by our one-step generator. The score functions are parameterized as two diffusion models trained separately on each distribution. Combined with a simple regression loss matching the large-scale structure of the multi-step diffusion outputs, our method outperforms all published few-step diffusion approaches, reaching 2.62 FID on ImageNet 64x64 and 11.49 FID on zero-shot COCO-30k, comparable to Stable Diffusion but orders of magnitude faster. Utilizing FP16 inference, our model generates images at 20 FPS on modern hardware.

{{</citation>}}


### (9/156) CAST: Cross-Attention in Space and Time for Video Action Recognition (Dongho Lee et al., 2023)

{{<citation>}}

Dongho Lee, Jongseo Lee, Jinwoo Choi. (2023)  
**CAST: Cross-Attention in Space and Time for Video Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.18825v1)  

---


**ABSTRACT**  
Recognizing human actions in videos requires spatial and temporal understanding. Most existing action recognition models lack a balanced spatio-temporal understanding of videos. In this work, we propose a novel two-stream architecture, called Cross-Attention in Space and Time (CAST), that achieves a balanced spatio-temporal understanding of videos using only RGB input. Our proposed bottleneck cross-attention mechanism enables the spatial and temporal expert models to exchange information and make synergistic predictions, leading to improved performance. We validate the proposed method with extensive experiments on public benchmarks with different characteristics: EPIC-KITCHENS-100, Something-Something-V2, and Kinetics-400. Our method consistently shows favorable performance across these datasets, while the performance of existing methods fluctuates depending on the dataset characteristics.

{{</citation>}}


### (10/156) ElasticDiffusion: Training-free Arbitrary Size Image Generation (Moayed Haji-Ali et al., 2023)

{{<citation>}}

Moayed Haji-Ali, Guha Balakrishnan, Vicente Ordonez. (2023)  
**ElasticDiffusion: Training-free Arbitrary Size Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18822v1)  

---


**ABSTRACT**  
Diffusion models have revolutionized image generation in recent years, yet they are still limited to a few sizes and aspect ratios. We propose ElasticDiffusion, a novel training-free decoding method that enables pretrained text-to-image diffusion models to generate images with various sizes. ElasticDiffusion attempts to decouple the generation trajectory of a pretrained model into local and global signals. The local signal controls low-level pixel information and can be estimated on local patches, while the global signal is used to maintain overall structural consistency and is estimated with a reference image. We test our method on CelebA-HQ (faces) and LAION-COCO (objects/indoor/outdoor scenes). Our experiments and qualitative results show superior image coherence quality across aspect ratios compared to MultiDiffusion and the standard decoding strategy of Stable Diffusion. Code: https://github.com/MoayedHajiAli/ElasticDiffusion-official.git

{{</citation>}}


### (11/156) LucidDreaming: Controllable Object-Centric 3D Generation (Zhaoning Wang et al., 2023)

{{<citation>}}

Zhaoning Wang, Ming Li, Chen Chen. (2023)  
**LucidDreaming: Controllable Object-Centric 3D Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00588v1)  

---


**ABSTRACT**  
With the recent development of generative models, Text-to-3D generations have also seen significant growth. Nonetheless, achieving precise control over 3D generation continues to be an arduous task, as using text to control often leads to missing objects and imprecise locations. Contemporary strategies for enhancing controllability in 3D generation often entail the introduction of additional parameters, such as customized diffusion models. This often induces hardness in adapting to different diffusion models or creating distinct objects.   In this paper, we present LucidDreaming as an effective pipeline capable of fine-grained control over 3D generation. It requires only minimal input of 3D bounding boxes, which can be deduced from a simple text prompt using a Large Language Model. Specifically, we propose clipped ray sampling to separately render and optimize objects with user specifications. We also introduce object-centric density blob bias, fostering the separation of generated objects. With individual rendering and optimizing of objects, our method excels not only in controlled content generation from scratch but also within the pre-trained NeRF scenes. In such scenarios, existing generative approaches often disrupt the integrity of the original scene, and current editing methods struggle to synthesize new content in empty spaces. We show that our method exhibits remarkable adaptability across a spectrum of mainstream Score Distillation Sampling-based 3D generation frameworks, and achieves superior alignment of 3D content when compared to baseline approaches. We also provide a dataset of prompts with 3D bounding boxes, benchmarking 3D spatial controllability.

{{</citation>}}


### (12/156) MD-Splatting: Learning Metric Deformation from 4D Gaussians in Highly Deformable Scenes (Bardienus P. Duisterhof et al., 2023)

{{<citation>}}

Bardienus P. Duisterhof, Zhao Mandi, Yunchao Yao, Jia-Wei Liu, Mike Zheng Shou, Shuran Song, Jeffrey Ichnowski. (2023)  
**MD-Splatting: Learning Metric Deformation from 4D Gaussians in Highly Deformable Scenes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00583v1)  

---


**ABSTRACT**  
Accurate 3D tracking in highly deformable scenes with occlusions and shadows can facilitate new applications in robotics, augmented reality, and generative AI. However, tracking under these conditions is extremely challenging due to the ambiguity that arises with large deformations, shadows, and occlusions. We introduce MD-Splatting, an approach for simultaneous 3D tracking and novel view synthesis, using video captures of a dynamic scene from various camera poses. MD-Splatting builds on recent advances in Gaussian splatting, a method that learns the properties of a large number of Gaussians for state-of-the-art and fast novel view synthesis. MD-Splatting learns a deformation function to project a set of Gaussians with non-metric, thus canonical, properties into metric space. The deformation function uses a neural-voxel encoding and a multilayer perceptron (MLP) to infer Gaussian position, rotation, and a shadow scalar. We enforce physics-inspired regularization terms based on local rigidity, conservation of momentum, and isometry, which leads to trajectories with smaller trajectory errors. MD-Splatting achieves high-quality 3D tracking on highly deformable scenes with shadows and occlusions. Compared to state-of-the-art, we improve 3D tracking by an average of 23.9 %, while simultaneously achieving high-quality novel view synthesis. With sufficient texture such as in scene 6, MD-Splatting achieves a median tracking error of 3.39 mm on a cloth of 1 x 1 meters in size. Project website: https://md-splatting.github.io/.

{{</citation>}}


### (13/156) X-InstructBLIP: A Framework for aligning X-Modal instruction-aware representations to LLMs and Emergent Cross-modal Reasoning (Artemis Panagopoulou et al., 2023)

{{<citation>}}

Artemis Panagopoulou, Le Xue, Ning Yu, Junnan Li, Dongxu Li, Shafiq Joty, Ran Xu, Silvio Savarese, Caiming Xiong, Juan Carlos Niebles. (2023)  
**X-InstructBLIP: A Framework for aligning X-Modal instruction-aware representations to LLMs and Emergent Cross-modal Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.18799v1)  

---


**ABSTRACT**  
Vision-language pre-training and instruction tuning have demonstrated general-purpose capabilities in 2D visual reasoning tasks by aligning visual encoders with state-of-the-art large language models (LLMs). In this paper, we introduce a simple, yet effective, cross-modality framework built atop frozen LLMs that allows the integration of various modalities without extensive modality-specific customization. To facilitate instruction-modality fine-tuning, we collect high-quality instruction tuning data in an automatic and scalable manner, composed of 24K QA samples for audio and 250K QA samples for 3D. Leveraging instruction-aware representations, our model performs comparably with leading-edge counterparts without the need of extensive modality-specific pre-training or customization. Furthermore, our approach demonstrates cross-modal reasoning abilities across two or more input modalities, despite each modality projection being trained individually. To study the model's cross-modal abilities, we contribute a novel Discriminative Cross-modal Reasoning (DisCRn) evaluation task, comprising 9K audio-video QA samples and 28K image-3D QA samples that require the model to reason discriminatively across disparate input modalities.

{{</citation>}}


### (14/156) Probing and Mitigating Intersectional Social Biases in Vision-Language Models with Counterfactual Examples (Phillip Howard et al., 2023)

{{<citation>}}

Phillip Howard, Avinash Madasu, Tiep Le, Gustavo Lujan Moreno, Anahita Bhiwandiwalla, Vasudev Lal. (2023)  
**Probing and Mitigating Intersectional Social Biases in Vision-Language Models with Counterfactual Examples**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2312.00825v1)  

---


**ABSTRACT**  
While vision-language models (VLMs) have achieved remarkable performance improvements recently, there is growing evidence that these models also posses harmful biases with respect to social attributes such as gender and race. Prior studies have primarily focused on probing such bias attributes individually while ignoring biases associated with intersections between social attributes. This could be due to the difficulty of collecting an exhaustive set of image-text pairs for various combinations of social attributes. To address this challenge, we employ text-to-image diffusion models to produce counterfactual examples for probing intserctional social biases at scale. Our approach utilizes Stable Diffusion with cross attention control to produce sets of counterfactual image-text pairs that are highly similar in their depiction of a subject (e.g., a given occupation) while differing only in their depiction of intersectional social attributes (e.g., race & gender). Through our over-generate-then-filter methodology, we produce SocialCounterfactuals, a high-quality dataset containing over 171k image-text pairs for probing intersectional biases related to gender, race, and physical characteristics. We conduct extensive experiments to demonstrate the usefulness of our generated dataset for probing and mitigating intersectional social biases in state-of-the-art VLMs.

{{</citation>}}


### (15/156) CoDi-2: In-Context, Interleaved, and Interactive Any-to-Any Generation (Zineng Tang et al., 2023)

{{<citation>}}

Zineng Tang, Ziyi Yang, Mahmoud Khademi, Yang Liu, Chenguang Zhu, Mohit Bansal. (2023)  
**CoDi-2: In-Context, Interleaved, and Interactive Any-to-Any Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs-SD, cs.CV, eess-AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.18775v1)  

---


**ABSTRACT**  
We present CoDi-2, a versatile and interactive Multimodal Large Language Model (MLLM) that can follow complex multimodal interleaved instructions, conduct in-context learning (ICL), reason, chat, edit, etc., in an any-to-any input-output modality paradigm. By aligning modalities with language for both encoding and generation, CoDi-2 empowers Large Language Models (LLMs) to not only understand complex modality-interleaved instructions and in-context examples, but also autoregressively generate grounded and coherent multimodal outputs in the continuous feature space. To train CoDi-2, we build a large-scale generation dataset encompassing in-context multimodal instructions across text, vision, and audio. CoDi-2 demonstrates a wide range of zero-shot capabilities for multimodal generation, such as in-context learning, reasoning, and compositionality of any-to-any modality generation through multi-round interactive conversation. CoDi-2 surpasses previous domain-specific models on tasks such as subject-driven image generation, vision transformation, and audio editing. CoDi-2 signifies a substantial breakthrough in developing a comprehensive multimodal foundation model adept at interpreting in-context language-vision-audio interleaved instructions and producing multimodal outputs.

{{</citation>}}


### (16/156) CLIP-QDA: An Explainable Concept Bottleneck Model (Rémi Kazmierczak et al., 2023)

{{<citation>}}

Rémi Kazmierczak, Eloïse Berthier, Goran Frehse, Gianni Franchi. (2023)  
**CLIP-QDA: An Explainable Concept Bottleneck Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00110v1)  

---


**ABSTRACT**  
In this paper, we introduce an explainable algorithm designed from a multi-modal foundation model, that performs fast and explainable image classification. Drawing inspiration from CLIP-based Concept Bottleneck Models (CBMs), our method creates a latent space where each neuron is linked to a specific word. Observing that this latent space can be modeled with simple distributions, we use a Mixture of Gaussians (MoG) formalism to enhance the interpretability of this latent space. Then, we introduce CLIP-QDA, a classifier that only uses statistical values to infer labels from the concepts. In addition, this formalism allows for both local and global explanations. These explanations come from the inner design of our architecture, our work is part of a new family of greybox models, combining performances of opaque foundation models and the interpretability of transparent models. Our empirical findings show that in instances where the MoG assumption holds, CLIP-QDA achieves similar accuracy with state-of-the-art methods CBMs. Our explanations compete with existing XAI methods while being faster to compute.

{{</citation>}}


### (17/156) MLLMs-Augmented Visual-Language Representation Learning (Yanqing Liu et al., 2023)

{{<citation>}}

Yanqing Liu, Kai Wang, Wenqi Shao, Ping Luo, Yu Qiao, Mike Zheng Shou, Kaipeng Zhang, Yang You. (2023)  
**MLLMs-Augmented Visual-Language Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.18765v2)  

---


**ABSTRACT**  
Visual-language pre-training (VLP) has achieved remarkable success in multi-modal tasks, largely attributed to the availability of large-scale image-text datasets. In this work, we demonstrate that multi-modal large language models (MLLMs) can enhance visual-language representation learning by improving data quality. Our approach is simple, utilizing MLLMs to extend multiple captions for each image. To prevent the bias introduced by MLLMs' hallucinations and intrinsic caption styles, we propose "text shearing" to maintain the same length for extended captions as that of the original captions. In image-text retrieval, our method consistently obtains 5.6 ~ 35.0% and 16.8 ~ 46.1% improvement on R@1 under the fine-tuning and zero-shot settings, respectively. Notably, we obtain zero-shot results that are comparable to fine-tuning on target datasets, which encourages more exploration of the versatile use of MLLMs.

{{</citation>}}


### (18/156) Semi-supervised Semantic Segmentation via Boosting Uncertainty on Unlabeled Data (Daoan Zhang et al., 2023)

{{<citation>}}

Daoan Zhang, Yunhao Luo, Jianguo Zhang. (2023)  
**Semi-supervised Semantic Segmentation via Boosting Uncertainty on Unlabeled Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.18758v1)  

---


**ABSTRACT**  
We bring a new perspective to semi-supervised semantic segmentation by providing an analysis on the labeled and unlabeled distributions in training datasets. We first figure out that the distribution gap between labeled and unlabeled datasets cannot be ignored, even though the two datasets are sampled from the same distribution. To address this issue, we theoretically analyze and experimentally prove that appropriately boosting uncertainty on unlabeled data can help minimize the distribution gap, which benefits the generalization of the model. We propose two strategies and design an uncertainty booster algorithm, specially for semi-supervised semantic segmentation. Extensive experiments are carried out based on these theories, and the results confirm the efficacy of the algorithm and strategies. Our plug-and-play uncertainty booster is tiny, efficient, and robust to hyperparameters but can significantly promote performance. Our approach achieves state-of-the-art performance in our experiments compared to the current semi-supervised semantic segmentation methods on the popular benchmarks: Cityscapes and PASCAL VOC 2012 with different train settings.

{{</citation>}}


### (19/156) Merlin:Empowering Multimodal LLMs with Foresight Minds (En Yu et al., 2023)

{{<citation>}}

En Yu, Liang Zhao, Yana Wei, Jinrong Yang, Dongming Wu, Lingyu Kong, Haoran Wei, Tiancai Wang, Zheng Ge, Xiangyu Zhang, Wenbing Tao. (2023)  
**Merlin:Empowering Multimodal LLMs with Foresight Minds**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00589v1)  

---


**ABSTRACT**  
Humans possess the remarkable ability to foresee the future to a certain extent based on present observations, a skill we term as foresight minds. However, this capability remains largely under explored within existing Multimodal Large Language Models (MLLMs), hindering their capacity to learn the fundamental principles of how things operate and the intentions behind the observed subjects. To address this issue, we introduce the integration of future modeling into the existing learning frameworks of MLLMs. By utilizing the subject trajectory, a highly structured representation of a consecutive frame sequence, as a learning objective, we aim to bridge the gap between the past and the future. We propose two innovative methods to empower MLLMs with foresight minds, Foresight Pre-Training (FPT) and Foresight Instruction-Tuning (FIT), which are inspired by the modern learning paradigm of LLMs. Specifically, FPT jointly training various tasks centered on trajectories, enabling MLLMs to learn how to attend and predict entire trajectories from a given initial observation. Then, FIT requires MLLMs to first predict trajectories of related objects and then reason about potential future events based on them. Aided by FPT and FIT, we build a novel and unified MLLM named Merlin that supports multi-images input and analysis about potential actions of multiple objects for the future reasoning. Experimental results show Merlin powerful foresight minds with impressive performance on both future reasoning and visual comprehension tasks.

{{</citation>}}


### (20/156) Improving the Robustness of Quantized Deep Neural Networks to White-Box Attacks using Stochastic Quantization and Information-Theoretic Ensemble Training (Saurabh Farkya et al., 2023)

{{<citation>}}

Saurabh Farkya, Aswin Raghavan, Avi Ziskind. (2023)  
**Improving the Robustness of Quantized Deep Neural Networks to White-Box Attacks using Stochastic Quantization and Information-Theoretic Ensemble Training**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: AI, Quantization  
[Paper Link](http://arxiv.org/abs/2312.00105v1)  

---


**ABSTRACT**  
Most real-world applications that employ deep neural networks (DNNs) quantize them to low precision to reduce the compute needs. We present a method to improve the robustness of quantized DNNs to white-box adversarial attacks. We first tackle the limitation of deterministic quantization to fixed ``bins'' by introducing a differentiable Stochastic Quantizer (SQ). We explore the hypothesis that different quantizations may collectively be more robust than each quantized DNN. We formulate a training objective to encourage different quantized DNNs to learn different representations of the input image. The training objective captures diversity and accuracy via mutual information between ensemble members. Through experimentation, we demonstrate substantial improvement in robustness against $L_\infty$ attacks even if the attacker is allowed to backpropagate through SQ (e.g., > 50\% accuracy to PGD(5/255) on CIFAR10 without adversarial training), compared to vanilla DNNs as well as existing ensembles of quantized DNNs. We extend the method to detect attacks and generate robustness profiles in the adversarial information plane (AIP), towards a unified analysis of different threat models by correlating the MI and accuracy.

{{</citation>}}


### (21/156) RaDialog: A Large Vision-Language Model for Radiology Report Generation and Conversational Assistance (Chantal Pellegrini et al., 2023)

{{<citation>}}

Chantal Pellegrini, Ege Özsoy, Benjamin Busam, Nassir Navab, Matthias Keicher. (2023)  
**RaDialog: A Large Vision-Language Model for Radiology Report Generation and Conversational Assistance**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: AI, Dialog, Language Model  
[Paper Link](http://arxiv.org/abs/2311.18681v1)  

---


**ABSTRACT**  
Conversational AI tools that can generate and discuss clinically correct radiology reports for a given medical image have the potential to transform radiology. Such a human-in-the-loop radiology assistant could facilitate a collaborative diagnostic process, thus saving time and improving the quality of reports. Towards this goal, we introduce RaDialog, the first thoroughly evaluated and publicly available large vision-language model for radiology report generation and interactive dialog. RaDialog effectively integrates visual image features and structured pathology findings with a large language model (LLM) while simultaneously adapting it to a specialized domain using parameter-efficient fine-tuning. To keep the conversational abilities of the underlying LLM, we propose a comprehensive, semi-automatically labeled, image-grounded instruct dataset for chest X-ray radiology tasks. By training with this dataset, our method achieves state-of-the-art clinical correctness in report generation and shows impressive abilities in interactive tasks such as correcting reports and answering questions, serving as a foundational step toward clinical dialog systems. Our code is available on github: https://github.com/ChantalMP/RaDialog.

{{</citation>}}


### (22/156) Cascaded Interaction with Eroded Deep Supervision for Salient Object Detection (Hewen Xiao et al., 2023)

{{<citation>}}

Hewen Xiao, Jie Mei, Guangfu Ma, Weiren Wu. (2023)  
**Cascaded Interaction with Eroded Deep Supervision for Salient Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.18675v1)  

---


**ABSTRACT**  
Deep convolutional neural networks have been widely applied in salient object detection and have achieved remarkable results in this field. However, existing models suffer from information distortion caused by interpolation during up-sampling and down-sampling. In response to this drawback, this article starts from two directions in the network: feature and label. On the one hand, a novel cascaded interaction network with a guidance module named global-local aligned attention (GAA) is designed to reduce the negative impact of interpolation on the feature side. On the other hand, a deep supervision strategy based on edge erosion is proposed to reduce the negative guidance of label interpolation on lateral output. Extensive experiments on five popular datasets demonstrate the superiority of our method.

{{</citation>}}


### (23/156) Learning Part Segmentation from Synthetic Animals (Jiawei Peng et al., 2023)

{{<citation>}}

Jiawei Peng, Ju He, Prakhar Kaushik, Zihao Xiao, Jiteng Mu, Alan Yuille. (2023)  
**Learning Part Segmentation from Synthetic Animals**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.18661v1)  

---


**ABSTRACT**  
Semantic part segmentation provides an intricate and interpretable understanding of an object, thereby benefiting numerous downstream tasks. However, the need for exhaustive annotations impedes its usage across diverse object types. This paper focuses on learning part segmentation from synthetic animals, leveraging the Skinned Multi-Animal Linear (SMAL) models to scale up existing synthetic data generated by computer-aided design (CAD) animal models. Compared to CAD models, SMAL models generate data with a wider range of poses observed in real-world scenarios. As a result, our first contribution is to construct a synthetic animal dataset of tigers and horses with more pose diversity, termed Synthetic Animal Parts (SAP). We then benchmark Syn-to-Real animal part segmentation from SAP to PartImageNet, namely SynRealPart, with existing semantic segmentation domain adaptation methods and further improve them as our second contribution. Concretely, we examine three Syn-to-Real adaptation methods but observe relative performance drop due to the innate difference between the two tasks. To address this, we propose a simple yet effective method called Class-Balanced Fourier Data Mixing (CB-FDM). Fourier Data Mixing aligns the spectral amplitudes of synthetic images with real images, thereby making the mixed images have more similar frequency content to real images. We further use Class-Balanced Pseudo-Label Re-Weighting to alleviate the imbalanced class distribution. We demonstrate the efficacy of CB-FDM on SynRealPart over previous methods with significant performance improvements. Remarkably, our third contribution is to reveal that the learned parts from synthetic tiger and horse are transferable across all quadrupeds in PartImageNet, further underscoring the utility and potential applications of animal part segmentation.

{{</citation>}}


### (24/156) LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning (Sijin Chen et al., 2023)

{{<citation>}}

Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, Tao Chen. (2023)  
**LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.18651v1)  

---


**ABSTRACT**  
Recent advances in Large Multimodal Models (LMM) have made it possible for various applications in human-machine interactions. However, developing LMMs that can comprehend, reason, and plan in complex and diverse 3D environments remains a challenging topic, especially considering the demand for understanding permutation-invariant point cloud 3D representations of the 3D scene. Existing works seek help from multi-view images, and project 2D features to 3D space as 3D scene representations. This, however, leads to huge computational overhead and performance degradation. In this paper, we present LL3DA, a Large Language 3D Assistant that takes point cloud as direct input and respond to both textual-instructions and visual-prompts. This help LMMs better comprehend human interactions and further help to remove the ambiguities in cluttered 3D scenes. Experiments show that LL3DA achieves remarkable results, and surpasses various 3D vision-language models on both 3D Dense Captioning and 3D Question Answering.

{{</citation>}}


### (25/156) Towards Unsupervised Representation Learning: Learning, Evaluating and Transferring Visual Representations (Bonifaz Stuhr, 2023)

{{<citation>}}

Bonifaz Stuhr. (2023)  
**Towards Unsupervised Representation Learning: Learning, Evaluating and Transferring Visual Representations**  

---
Primary Category: cs.CV  
Categories: I-2; I-3; I-4; I-5; I-6, cs-AI, cs-CV, cs-GR, cs-LG, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.00101v1)  

---


**ABSTRACT**  
Unsupervised representation learning aims at finding methods that learn representations from data without annotation-based signals. Abstaining from annotations not only leads to economic benefits but may - and to some extent already does - result in advantages regarding the representation's structure, robustness, and generalizability to different tasks. In the long run, unsupervised methods are expected to surpass their supervised counterparts due to the reduction of human intervention and the inherently more general setup that does not bias the optimization towards an objective originating from specific annotation-based signals. While major advantages of unsupervised representation learning have been recently observed in natural language processing, supervised methods still dominate in vision domains for most tasks. In this dissertation, we contribute to the field of unsupervised (visual) representation learning from three perspectives: (i) Learning representations: We design unsupervised, backpropagation-free Convolutional Self-Organizing Neural Networks (CSNNs) that utilize self-organization- and Hebbian-based learning rules to learn convolutional kernels and masks to achieve deeper backpropagation-free models. (ii) Evaluating representations: We build upon the widely used (non-)linear evaluation protocol to define pretext- and target-objective-independent metrics for measuring and investigating the objective function mismatch between various unsupervised pretext tasks and target tasks. (iii) Transferring representations: We contribute CARLANE, the first 3-way sim-to-real domain adaptation benchmark for 2D lane detection, and a method based on prototypical self-supervised learning. Finally, we contribute a content-consistent unpaired image-to-image translation method that utilizes masks, global and local discriminators, and similarity sampling to mitigate content inconsistencies.

{{</citation>}}


### (26/156) Simple Semantic-Aided Few-Shot Learning (Hai Zhang et al., 2023)

{{<citation>}}

Hai Zhang, Junzhe Xu, Shanlin Jiang, Zhenan He. (2023)  
**Simple Semantic-Aided Few-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2311.18649v1)  

---


**ABSTRACT**  
Learning from a limited amount of data, namely Few-Shot Learning, stands out as a challenging computer vision task. Several works exploit semantics and design complicated semantic fusion mechanisms to compensate for rare representative features within restricted data. However, relying on naive semantics such as class names introduces biases due to their brevity, while acquiring extensive semantics from external knowledge takes a huge time and effort. This limitation severely constrains the potential of semantics in few-shot learning. In this paper, we design an automatic way called Semantic Evolution to generate high-quality semantics. The incorporation of high-quality semantics alleviates the need for complex network structures and learning algorithms used in previous works. Hence, we employ a simple two-layer network termed Semantic Alignment Network to transform semantics and visual features into robust class prototypes with rich discriminative features for few-shot classification. The experimental results show our framework outperforms all previous methods on five benchmarks, demonstrating a simple network with high-quality semantics can beat intricate multi-modal modules on few-shot classification tasks.

{{</citation>}}


### (27/156) Stochastic Vision Transformers with Wasserstein Distance-Aware Attention (Franciskus Xaverius Erick et al., 2023)

{{<citation>}}

Franciskus Xaverius Erick, Mina Rezaei, Johanna Paula Müller, Bernhard Kainz. (2023)  
**Stochastic Vision Transformers with Wasserstein Distance-Aware Attention**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.18645v1)  

---


**ABSTRACT**  
Self-supervised learning is one of the most promising approaches to acquiring knowledge from limited labeled data. Despite the substantial advancements made in recent years, self-supervised models have posed a challenge to practitioners, as they do not readily provide insight into the model's confidence and uncertainty. Tackling this issue is no simple feat, primarily due to the complexity involved in implementing techniques that can make use of the latent representations learned during pre-training without relying on explicit labels. Motivated by this, we introduce a new stochastic vision transformer that integrates uncertainty and distance awareness into self-supervised learning (SSL) pipelines. Instead of the conventional deterministic vector embedding, our novel stochastic vision transformer encodes image patches into elliptical Gaussian distributional embeddings. Notably, the attention matrices of these stochastic representational embeddings are computed using Wasserstein distance-based attention, effectively capitalizing on the distributional nature of these embeddings. Additionally, we propose a regularization term based on Wasserstein distance for both pre-training and fine-tuning processes, thereby incorporating distance awareness into latent representations. We perform extensive experiments across different tasks such as in-distribution generalization, out-of-distribution detection, dataset corruption, semi-supervised settings, and transfer learning to other datasets and tasks. Our proposed method achieves superior accuracy and calibration, surpassing the self-supervised baseline in a wide range of experiments on a variety of datasets.

{{</citation>}}


### (28/156) A Lightweight Clustering Framework for Unsupervised Semantic Segmentation (Yau Shing Jonathan Cheung et al., 2023)

{{<citation>}}

Yau Shing Jonathan Cheung, Xi Chen, Lihe Yang, Hengshuang Zhao. (2023)  
**A Lightweight Clustering Framework for Unsupervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.18628v1)  

---


**ABSTRACT**  
Unsupervised semantic segmentation aims to label each pixel of an image to a corresponding class without the use of annotated data. It is a widely researched area as obtaining labeled datasets are expensive. While previous works in the field demonstrated a gradual improvement in segmentation performance, most of them required neural network training. This made segmentation equally expensive, especially when dealing with large-scale datasets. We thereby propose a lightweight clustering framework for unsupervised semantic segmentation. Attention features of the self-supervised vision transformer exhibit strong foreground-background differentiability. By clustering these features into a small number of clusters, we could separate foreground and background image patches into distinct groupings. In our clustering framework, we first obtain attention features from the self-supervised vision transformer. Then we extract Dataset-level, Category-level and Image-level masks by clustering features within the same dataset, category and image. We further ensure multilevel clustering consistency across the three levels and this allows us to extract patch-level binary pseudo-masks. Finally, the pseudo-mask is upsampled, refined and class assignment is performed according to the CLS token of object regions. Our framework demonstrates great promise in unsupervised semantic segmentation and achieves state-of-the-art results on PASCAL VOC and MS COCO datasets.

{{</citation>}}


### (29/156) Anatomy and Physiology of Artificial Intelligence in PET Imaging (Tyler J. Bradshaw et al., 2023)

{{<citation>}}

Tyler J. Bradshaw, Alan B. McMillan. (2023)  
**Anatomy and Physiology of Artificial Intelligence in PET Imaging**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18614v1)  

---


**ABSTRACT**  
The influence of artificial intelligence (AI) within the field of nuclear medicine has been rapidly growing. Many researchers and clinicians are seeking to apply AI within PET, and clinicians will soon find themselves engaging with AI-based applications all along the chain of molecular imaging, from image reconstruction to enhanced reporting. This expanding presence of AI in PET imaging will result in greater demand for educational resources for those unfamiliar with AI. The objective of this article to is provide an illustrated guide to the core principles of modern AI, with specific focus on aspects that are most likely to be encountered in PET imaging. We describe convolutional neural networks, algorithm training, and explain the components of the commonly used U-Net for segmentation and image synthesis.

{{</citation>}}


### (30/156) Semantic-Aware Frame-Event Fusion based Pattern Recognition via Large Vision-Language Models (Dong Li et al., 2023)

{{<citation>}}

Dong Li, Jiandong Jin, Yuhao Zhang, Yanlin Zhong, Yaoyang Wu, Lan Chen, Xiao Wang, Bin Luo. (2023)  
**Semantic-Aware Frame-Event Fusion based Pattern Recognition via Large Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2311.18592v1)  

---


**ABSTRACT**  
Pattern recognition through the fusion of RGB frames and Event streams has emerged as a novel research area in recent years. Current methods typically employ backbone networks to individually extract the features of RGB frames and event streams, and subsequently fuse these features for pattern recognition. However, we posit that these methods may suffer from key issues like sematic gaps and small-scale backbone networks. In this study, we introduce a novel pattern recognition framework that consolidates the semantic labels, RGB frames, and event streams, leveraging pre-trained large-scale vision-language models. Specifically, given the input RGB frames, event streams, and all the predefined semantic labels, we employ a pre-trained large-scale vision model (CLIP vision encoder) to extract the RGB and event features. To handle the semantic labels, we initially convert them into language descriptions through prompt engineering, and then obtain the semantic features using the pre-trained large-scale language model (CLIP text encoder). Subsequently, we integrate the RGB/Event features and semantic features using multimodal Transformer networks. The resulting frame and event tokens are further amplified using self-attention layers. Concurrently, we propose to enhance the interactions between text tokens and RGB/Event tokens via cross-attention. Finally, we consolidate all three modalities using self-attention and feed-forward layers for recognition. Comprehensive experiments on the HARDVS and PokerEvent datasets fully substantiate the efficacy of our proposed SAFE model. The source code will be made available at https://github.com/Event-AHU/SAFE_LargeVLM.

{{</citation>}}


### (31/156) MaXTron: Mask Transformer with Trajectory Attention for Video Panoptic Segmentation (Ju He et al., 2023)

{{<citation>}}

Ju He, Qihang Yu, Inkyu Shin, Xueqing Deng, Xiaohui Shen, Alan Yuille, Liang-Chieh Chen. (2023)  
**MaXTron: Mask Transformer with Trajectory Attention for Video Panoptic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2311.18537v1)  

---


**ABSTRACT**  
Video panoptic segmentation requires consistently segmenting (for both `thing' and `stuff' classes) and tracking objects in a video over time. In this work, we present MaXTron, a general framework that exploits Mask XFormer with Trajectory Attention to tackle the task. MaXTron enriches an off-the-shelf mask transformer by leveraging trajectory attention. The deployed mask transformer takes as input a short clip consisting of only a few frames and predicts the clip-level segmentation. To enhance the temporal consistency, MaXTron employs within-clip and cross-clip tracking modules, efficiently utilizing trajectory attention. Originally designed for video classification, trajectory attention learns to model the temporal correspondences between neighboring frames and aggregates information along the estimated motion paths. However, it is nontrivial to directly extend trajectory attention to the per-pixel dense prediction tasks due to its quadratic dependency on input size. To alleviate the issue, we propose to adapt the trajectory attention for both the dense pixel features and object queries, aiming to improve the short-term and long-term tracking results, respectively. Particularly, in our within-clip tracking module, we propose axial-trajectory attention that effectively computes the trajectory attention for tracking dense pixels sequentially along the height- and width-axes. The axial decomposition significantly reduces the computational complexity for dense pixel features. In our cross-clip tracking module, since the object queries in mask transformer are learned to encode the object information, we are able to capture the long-term temporal connections by applying trajectory attention to object queries, which learns to track each object across different clips. Without bells and whistles, MaXTron demonstrates state-of-the-art performances on video segmentation benchmarks.

{{</citation>}}


### (32/156) Fast ODE-based Sampling for Diffusion Models in Around 5 Steps (Zhenyu Zhou et al., 2023)

{{<citation>}}

Zhenyu Zhou, Defang Chen, Can Wang, Chun Chen. (2023)  
**Fast ODE-based Sampling for Diffusion Models in Around 5 Steps**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.00094v1)  

---


**ABSTRACT**  
Sampling from diffusion models can be treated as solving the corresponding ordinary differential equations (ODEs), with the aim of obtaining an accurate solution with as few number of function evaluations (NFE) as possible. Recently, various fast samplers utilizing higher-order ODE solvers have emerged and achieved better performance than the initial first-order one. However, these numerical methods inherently result in certain approximation errors, which significantly degrades sample quality with extremely small NFE (e.g., around 5). In contrast, based on the geometric observation that each sampling trajectory almost lies in a two-dimensional subspace embedded in the ambient space, we propose Approximate MEan-Direction Solver (AMED-Solver) that eliminates truncation errors by directly learning the mean direction for fast diffusion sampling. Besides, our method can be easily used as a plugin to further improve existing ODE-based samplers. Extensive experiments on image synthesis with the resolution ranging from 32 to 256 demonstrate the effectiveness of our method. With only 5 NFE, we achieve 7.14 FID on CIFAR-10, 13.75 FID on ImageNet 64$\times$64, and 12.79 FID on LSUN Bedroom. Our code is available at https://github.com/zhyzhouu/amed-solver.

{{</citation>}}


### (33/156) Revisiting Proposal-based Object Detection (Aritra Bhowmik et al., 2023)

{{<citation>}}

Aritra Bhowmik, Martin R. Oswald, Pascal Mettes, Cees G. M. Snoek. (2023)  
**Revisiting Proposal-based Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.18512v1)  

---


**ABSTRACT**  
This paper revisits the pipeline for detecting objects in images with proposals. For any object detector, the obtained box proposals or queries need to be classified and regressed towards ground truth boxes. The common solution for the final predictions is to directly maximize the overlap between each proposal and the ground truth box, followed by a winner-takes-all ranking or non-maximum suppression. In this work, we propose a simple yet effective alternative. For proposal regression, we solve a simpler problem where we regress to the area of intersection between proposal and ground truth. In this way, each proposal only specifies which part contains the object, avoiding a blind inpainting problem where proposals need to be regressed beyond their visual scope. In turn, we replace the winner-takes-all strategy and obtain the final prediction by taking the union over the regressed intersections of a proposal group surrounding an object. Our revisited approach comes with minimal changes to the detection pipeline and can be plugged into any existing method. We show that our approach directly improves canonical object detection and instance segmentation architectures, highlighting the utility of intersection-based regression and grouping.

{{</citation>}}


### (34/156) ZeST-NeRF: Using temporal aggregation for Zero-Shot Temporal NeRFs (Violeta Menéndez González et al., 2023)

{{<citation>}}

Violeta Menéndez González, Andrew Gilbert, Graeme Phillipson, Stephen Jolly, Simon Hadfield. (2023)  
**ZeST-NeRF: Using temporal aggregation for Zero-Shot Temporal NeRFs**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-GR, cs-LG, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.18491v1)  

---


**ABSTRACT**  
In the field of media production, video editing techniques play a pivotal role. Recent approaches have had great success at performing novel view image synthesis of static scenes. But adding temporal information adds an extra layer of complexity. Previous models have focused on implicitly representing static and dynamic scenes using NeRF. These models achieve impressive results but are costly at training and inference time. They overfit an MLP to describe the scene implicitly as a function of position. This paper proposes ZeST-NeRF, a new approach that can produce temporal NeRFs for new scenes without retraining. We can accurately reconstruct novel views using multi-view synthesis techniques and scene flow-field estimation, trained only with unrelated scenes. We demonstrate how existing state-of-the-art approaches from a range of fields cannot adequately solve this new task and demonstrate the efficacy of our solution. The resulting network improves quantitatively by 15% and produces significantly better visual results.

{{</citation>}}


### (35/156) Mixture of Gaussian-distributed Prototypes with Generative Modelling for Interpretable Image Classification (Chong Wang et al., 2023)

{{<citation>}}

Chong Wang, Yuanhong Chen, Fengbei Liu, Davis James McCarthy, Helen Frazer, Gustavo Carneiro. (2023)  
**Mixture of Gaussian-distributed Prototypes with Generative Modelling for Interpretable Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2312.00092v1)  

---


**ABSTRACT**  
Prototypical-part interpretable methods, e.g., ProtoPNet, enhance interpretability by connecting classification predictions to class-specific training prototypes, thereby offering an intuitive insight into their decision-making. Current methods rely on a discriminative classifier trained with point-based learning techniques that provide specific values for prototypes. Such prototypes have relatively low representation power due to their sparsity and potential redundancy, with each prototype containing no variability measure. In this paper, we present a new generative learning of prototype distributions, named Mixture of Gaussian-distributed Prototypes (MGProto), which are represented by Gaussian mixture models (GMM). Such an approach enables the learning of more powerful prototype representations since each learned prototype will own a measure of variability, which naturally reduces the sparsity given the spread of the distribution around each prototype, and we also integrate a prototype diversity objective function into the GMM optimisation to reduce redundancy. Incidentally, the generative nature of MGProto offers a new and effective way for detecting out-of-distribution samples. To improve the compactness of MGProto, we further propose to prune Gaussian-distributed prototypes with a low prior. Experiments on CUB-200-2011, Stanford Cars, Stanford Dogs, and Oxford-IIIT Pets datasets show that MGProto achieves state-of-the-art classification and OoD detection performances with encouraging interpretability results.

{{</citation>}}


### (36/156) Layered Rendering Diffusion Model for Zero-Shot Guided Image Synthesis (Zipeng Qi et al., 2023)

{{<citation>}}

Zipeng Qi, Guoxi Huang, Zebin Huang, Qin Guo, Jinwen Chen, Junyu Han, Jian Wang, Gang Zhang, Lufei Liu, Errui Ding, Jingdong Wang. (2023)  
**Layered Rendering Diffusion Model for Zero-Shot Guided Image Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.18435v1)  

---


**ABSTRACT**  
This paper introduces innovative solutions to enhance spatial controllability in diffusion models reliant on text queries. We present two key innovations: Vision Guidance and the Layered Rendering Diffusion (LRDiff) framework. Vision Guidance, a spatial layout condition, acts as a clue in the perturbed distribution, greatly narrowing down the search space, to focus on the image sampling process adhering to the spatial layout condition. The LRDiff framework constructs an image-rendering process with multiple layers, each of which applies the vision guidance to instructively estimate the denoising direction for a single object. Such a layered rendering strategy effectively prevents issues like unintended conceptual blending or mismatches, while allowing for more coherent and contextually accurate image synthesis. The proposed method provides a more efficient and accurate means of synthesising images that align with specific spatial and contextual requirements. We demonstrate through our experiments that our method provides better results than existing techniques both quantitatively and qualitatively. We apply our method to three practical applications: bounding box-to-image, semantic mask-to-image and image editing.

{{</citation>}}


### (37/156) E2PNet: Event to Point Cloud Registration with Spatio-Temporal Representation Learning (Xiuhong Lin et al., 2023)

{{<citation>}}

Xiuhong Lin, Changjie Qiu, Zhipeng Cai, Siqi Shen, Yu Zang, Weiquan Liu, Xuesheng Bian, Matthias Müller, Cheng Wang. (2023)  
**E2PNet: Event to Point Cloud Registration with Spatio-Temporal Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.18433v1)  

---


**ABSTRACT**  
Event cameras have emerged as a promising vision sensor in recent years due to their unparalleled temporal resolution and dynamic range. While registration of 2D RGB images to 3D point clouds is a long-standing problem in computer vision, no prior work studies 2D-3D registration for event cameras. To this end, we propose E2PNet, the first learning-based method for event-to-point cloud registration. The core of E2PNet is a novel feature representation network called Event-Points-to-Tensor (EP2T), which encodes event data into a 2D grid-shaped feature tensor. This grid-shaped feature enables matured RGB-based frameworks to be easily used for event-to-point cloud registration, without changing hyper-parameters and the training procedure. EP2T treats the event input as spatio-temporal point clouds. Unlike standard 3D learning architectures that treat all dimensions of point clouds equally, the novel sampling and information aggregation modules in EP2T are designed to handle the inhomogeneity of the spatial and temporal dimensions. Experiments on the MVSEC and VECtor datasets demonstrate the superiority of E2PNet over hand-crafted and other learning-based methods. Compared to RGB-based registration, E2PNet is more robust to extreme illumination or fast motion due to the use of event data. Beyond 2D-3D registration, we also show the potential of EP2T for other vision tasks such as flow estimation, event-to-image reconstruction and object recognition. The source code can be found at: https://github.com/Xmu-qcj/E2PNet.

{{</citation>}}


### (38/156) TeG-DG: Textually Guided Domain Generalization for Face Anti-Spoofing (Lianrui Mu et al., 2023)

{{<citation>}}

Lianrui Mu, Jianhong Bai, Xiaoxuan He, Jiangnan Ye, Xiaoyu Liang, Yuchen Yang, Jiedong Zhuang, Haoji Hu. (2023)  
**TeG-DG: Textually Guided Domain Generalization for Face Anti-Spoofing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.18420v1)  

---


**ABSTRACT**  
Enhancing the domain generalization performance of Face Anti-Spoofing (FAS) techniques has emerged as a research focus. Existing methods are dedicated to extracting domain-invariant features from various training domains. Despite the promising performance, the extracted features inevitably contain residual style feature bias (e.g., illumination, capture device), resulting in inferior generalization performance. In this paper, we propose an alternative and effective solution, the Textually Guided Domain Generalization (TeG-DG) framework, which can effectively leverage text information for cross-domain alignment. Our core insight is that text, as a more abstract and universal form of expression, can capture the commonalities and essential characteristics across various attacks, bridging the gap between different image domains. Contrary to existing vision-language models, the proposed framework is elaborately designed to enhance the domain generalization ability of the FAS task. Concretely, we first design a Hierarchical Attention Fusion (HAF) module to enable adaptive aggregation of visual features at different levels; Then, a Textual-Enhanced Visual Discriminator (TEVD) is proposed for not only better alignment between the two modalities but also to regularize the classifier with unbiased text features. TeG-DG significantly outperforms previous approaches, especially in situations with extremely limited source domain data (~14% and ~12% improvements on HTER and AUC respectively), showcasing impressive few-shot performance.

{{</citation>}}


### (39/156) RainAI -- Precipitation Nowcasting from Satellite Data (Rafael Pablos Sarabia et al., 2023)

{{<citation>}}

Rafael Pablos Sarabia, Joachim Nyborg, Morten Birk, Ira Assent. (2023)  
**RainAI -- Precipitation Nowcasting from Satellite Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, physics-ao-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18398v1)  

---


**ABSTRACT**  
This paper presents a solution to the Weather4Cast 2023 competition, where the goal is to forecast high-resolution precipitation with an 8-hour lead time using lower-resolution satellite radiance images. We propose a simple, yet effective method for spatiotemporal feature learning using a 2D U-Net model, that outperforms the official 3D U-Net baseline in both performance and efficiency. We place emphasis on refining the dataset, through importance sampling and dataset preparation, and show that such techniques have a significant impact on performance. We further study an alternative cross-entropy loss function that improves performance over the standard mean squared error loss, while also enabling models to produce probabilistic outputs. Additional techniques are explored regarding the generation of predictions at different lead times, specifically through Conditioning Lead Time. Lastly, to generate high-resolution forecasts, we evaluate standard and learned upsampling methods. The code and trained parameters are available at https://github.com/rafapablos/w4c23-rainai.

{{</citation>}}


### (40/156) TIDE: Test Time Few Shot Object Detection (Weikai Li et al., 2023)

{{<citation>}}

Weikai Li, Hongfeng Wei, Yanlai Wu, Jie Yang, Yudi Ruan, Yuan Li, Ying Tang. (2023)  
**TIDE: Test Time Few Shot Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.18358v1)  

---


**ABSTRACT**  
Few-shot object detection (FSOD) aims to extract semantic knowledge from limited object instances of novel categories within a target domain. Recent advances in FSOD focus on fine-tuning the base model based on a few objects via meta-learning or data augmentation. Despite their success, the majority of them are grounded with parametric readjustment to generalize on novel objects, which face considerable challenges in Industry 5.0, such as (i) a certain amount of fine-tuning time is required, and (ii) the parameters of the constructed model being unavailable due to the privilege protection, making the fine-tuning fail. Such constraints naturally limit its application in scenarios with real-time configuration requirements or within black-box settings. To tackle the challenges mentioned above, we formalize a novel FSOD task, referred to as Test TIme Few Shot DEtection (TIDE), where the model is un-tuned in the configuration procedure. To that end, we introduce an asymmetric architecture for learning a support-instance-guided dynamic category classifier. Further, a cross-attention module and a multi-scale resizer are provided to enhance the model performance. Experimental results on multiple few-shot object detection platforms reveal that the proposed TIDE significantly outperforms existing contemporary methods. The implementation codes are available at https://github.com/deku-0621/TIDE

{{</citation>}}


### (41/156) Multilevel Saliency-Guided Self-Supervised Learning for Image Anomaly Detection (Jianjian Qin et al., 2023)

{{<citation>}}

Jianjian Qin, Chunzhi Gu, Jun Yu, Chao Zhang. (2023)  
**Multilevel Saliency-Guided Self-Supervised Learning for Image Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.18332v1)  

---


**ABSTRACT**  
Anomaly detection (AD) is a fundamental task in computer vision. It aims to identify incorrect image data patterns which deviate from the normal ones. Conventional methods generally address AD by preparing augmented negative samples to enforce self-supervised learning. However, these techniques typically do not consider semantics during augmentation, leading to the generation of unrealistic or invalid negative samples. Consequently, the feature extraction network can be hindered from embedding critical features. In this study, inspired by visual attention learning approaches, we propose CutSwap, which leverages saliency guidance to incorporate semantic cues for augmentation. Specifically, we first employ LayerCAM to extract multilevel image features as saliency maps and then perform clustering to obtain multiple centroids. To fully exploit saliency guidance, on each map, we select a pixel pair from the cluster with the highest centroid saliency to form a patch pair. Such a patch pair includes highly similar context information with dense semantic correlations. The resulting negative sample is created by swapping the locations of the patch pair. Compared to prior augmentation methods, CutSwap generates more subtle yet realistic negative samples to facilitate quality feature learning. Extensive experimental and ablative evaluations demonstrate that our method achieves state-of-the-art AD performance on two mainstream AD benchmark datasets.

{{</citation>}}


### (42/156) MRFP: Learning Generalizable Semantic Segmentation from Sim-2-Real with Multi-Resolution Feature Perturbation (Sumanth Udupa et al., 2023)

{{<citation>}}

Sumanth Udupa, Prajwal Gurunath, Aniruddh Sikdar, Suresh Sundaram. (2023)  
**MRFP: Learning Generalizable Semantic Segmentation from Sim-2-Real with Multi-Resolution Feature Perturbation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.18331v1)  

---


**ABSTRACT**  
Deep neural networks have shown exemplary performance on semantic scene understanding tasks on source domains, but due to the absence of style diversity during training, enhancing performance on unseen target domains using only single source domain data remains a challenging task. Generation of simulated data is a feasible alternative to retrieving large style-diverse real-world datasets as it is a cumbersome and budget-intensive process. However, the large domain-specific inconsistencies between simulated and real-world data pose a significant generalization challenge in semantic segmentation. In this work, to alleviate this problem, we propose a novel MultiResolution Feature Perturbation (MRFP) technique to randomize domain-specific fine-grained features and perturb style of coarse features. Our experimental results on various urban-scene segmentation datasets clearly indicate that, along with the perturbation of style-information, perturbation of fine-feature components is paramount to learn domain invariant robust feature maps for semantic segmentation models. MRFP is a simple and computationally efficient, transferable module with no additional learnable parameters or objective functions, that helps state-of-the-art deep neural networks to learn robust domain invariant features for simulation-to-real semantic segmentation.

{{</citation>}}


### (43/156) Anisotropic Neural Representation Learning for High-Quality Neural Rendering (Y. Wang et al., 2023)

{{<citation>}}

Y. Wang, J. Xu, Y. Zeng, Y. Gong. (2023)  
**Anisotropic Neural Representation Learning for High-Quality Neural Rendering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.18311v1)  

---


**ABSTRACT**  
Neural radiance fields (NeRFs) have achieved impressive view synthesis results by learning an implicit volumetric representation from multi-view images. To project the implicit representation into an image, NeRF employs volume rendering that approximates the continuous integrals of rays as an accumulation of the colors and densities of the sampled points. Although this approximation enables efficient rendering, it ignores the direction information in point intervals, resulting in ambiguous features and limited reconstruction quality. In this paper, we propose an anisotropic neural representation learning method that utilizes learnable view-dependent features to improve scene representation and reconstruction. We model the volumetric function as spherical harmonic (SH)-guided anisotropic features, parameterized by multilayer perceptrons, facilitating ambiguity elimination while preserving the rendering efficiency. To achieve robust scene reconstruction without anisotropy overfitting, we regularize the energy of the anisotropic features during training. Our method is flexiable and can be plugged into NeRF-based frameworks. Extensive experiments show that the proposed representation can boost the rendering quality of various NeRFs and achieve state-of-the-art rendering performance on both synthetic and real-world scenes.

{{</citation>}}


### (44/156) X-Dreamer: Creating High-quality 3D Content by Bridging the Domain Gap Between Text-to-2D and Text-to-3D Generation (Yiwei Ma et al., 2023)

{{<citation>}}

Yiwei Ma, Yijun Fan, Jiayi Ji, Haowei Wang, Xiaoshuai Sun, Guannan Jiang, Annan Shu, Rongrong Ji. (2023)  
**X-Dreamer: Creating High-quality 3D Content by Bridging the Domain Gap Between Text-to-2D and Text-to-3D Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.00085v1)  

---


**ABSTRACT**  
In recent times, automatic text-to-3D content creation has made significant progress, driven by the development of pretrained 2D diffusion models. Existing text-to-3D methods typically optimize the 3D representation to ensure that the rendered image aligns well with the given text, as evaluated by the pretrained 2D diffusion model. Nevertheless, a substantial domain gap exists between 2D images and 3D assets, primarily attributed to variations in camera-related attributes and the exclusive presence of foreground objects. Consequently, employing 2D diffusion models directly for optimizing 3D representations may lead to suboptimal outcomes. To address this issue, we present X-Dreamer, a novel approach for high-quality text-to-3D content creation that effectively bridges the gap between text-to-2D and text-to-3D synthesis. The key components of X-Dreamer are two innovative designs: Camera-Guided Low-Rank Adaptation (CG-LoRA) and Attention-Mask Alignment (AMA) Loss. CG-LoRA dynamically incorporates camera information into the pretrained diffusion models by employing camera-dependent generation for trainable parameters. This integration enhances the alignment between the generated 3D assets and the camera's perspective. AMA loss guides the attention map of the pretrained diffusion model using the binary mask of the 3D object, prioritizing the creation of the foreground object. This module ensures that the model focuses on generating accurate and detailed foreground objects. Extensive evaluations demonstrate the effectiveness of our proposed method compared to existing text-to-3D approaches. Our project webpage: https://xmuxiaoma666.github.io/Projects/X-Dreamer .

{{</citation>}}


### (45/156) Can Protective Perturbation Safeguard Personal Data from Being Exploited by Stable Diffusion? (Zhengyue Zhao et al., 2023)

{{<citation>}}

Zhengyue Zhao, Jinhao Duan, Kaidi Xu, Chenan Wang, Rui Zhangp Zidong Dup Qi Guo, Xing Hu. (2023)  
**Can Protective Perturbation Safeguard Personal Data from Being Exploited by Stable Diffusion?**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00084v1)  

---


**ABSTRACT**  
Stable Diffusion has established itself as a foundation model in generative AI artistic applications, receiving widespread research and application. Some recent fine-tuning methods have made it feasible for individuals to implant personalized concepts onto the basic Stable Diffusion model with minimal computational costs on small datasets. However, these innovations have also given rise to issues like facial privacy forgery and artistic copyright infringement. In recent studies, researchers have explored the addition of imperceptible adversarial perturbations to images to prevent potential unauthorized exploitation and infringements when personal data is used for fine-tuning Stable Diffusion. Although these studies have demonstrated the ability to protect images, it is essential to consider that these methods may not be entirely applicable in real-world scenarios. In this paper, we systematically evaluate the use of perturbations to protect images within a practical threat model. The results suggest that these approaches may not be sufficient to safeguard image privacy and copyright effectively. Furthermore, we introduce a purification method capable of removing protected perturbations while preserving the original image structure to the greatest extent possible. Experiments reveal that Stable Diffusion can effectively learn from purified images over all protective methods.

{{</citation>}}


### (46/156) BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos (Pilhyeon Lee et al., 2023)

{{<citation>}}

Pilhyeon Lee, Hyeran Byun. (2023)  
**BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.00083v1)  

---


**ABSTRACT**  
Temporal sentence grounding aims to localize moments relevant to a language description. Recently, DETR-like approaches have shown notable progress by decoding the center and length of a target moment from learnable queries. However, they suffer from the issue of center misalignment raised by the inherent ambiguity of moment centers, leading to inaccurate predictions. To remedy this problem, we introduce a novel boundary-oriented moment formulation. In our paradigm, the model no longer needs to find the precise center but instead suffices to predict any anchor point within the interval, from which the onset and offset are directly estimated. Based on this idea, we design a Boundary-Aligned Moment Detection Transformer (BAM-DETR), equipped with a dual-pathway decoding process. Specifically, it refines the anchor and boundaries within parallel pathways using global and boundary-focused attention, respectively. This separate design allows the model to focus on desirable regions, enabling precise refinement of moment predictions. Further, we propose a quality-based ranking method, ensuring that proposals with high localization qualities are prioritized over incomplete ones. Extensive experiments verify the advantages of our methods, where our model records new state-of-the-art results on three benchmarks. Code is at https://github.com/Pilhyeon/BAM-DETR.

{{</citation>}}


### (47/156) OmniMotionGPT: Animal Motion Generation with Limited Data (Zhangsihao Yang et al., 2023)

{{<citation>}}

Zhangsihao Yang, Mingyuan Zhou, Mengyi Shan, Bingbing Wen, Ziwei Xuan, Mitch Hill, Junjie Bai, Guo-Jun Qi, Yalin Wang. (2023)  
**OmniMotionGPT: Animal Motion Generation with Limited Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2311.18303v1)  

---


**ABSTRACT**  
Our paper aims to generate diverse and realistic animal motion sequences from textual descriptions, without a large-scale animal text-motion dataset. While the task of text-driven human motion synthesis is already extensively studied and benchmarked, it remains challenging to transfer this success to other skeleton structures with limited data. In this work, we design a model architecture that imitates Generative Pretraining Transformer (GPT), utilizing prior knowledge learned from human data to the animal domain. We jointly train motion autoencoders for both animal and human motions and at the same time optimize through the similarity scores among human motion encoding, animal motion encoding, and text CLIP embedding. Presenting the first solution to this problem, we are able to generate animal motions with high diversity and fidelity, quantitatively and qualitatively outperforming the results of training human motion generation baselines on animal data. Additionally, we introduce AnimalML3D, the first text-animal motion dataset with 1240 animation sequences spanning 36 different animal identities. We hope this dataset would mediate the data scarcity problem in text-driven animal motion generation, providing a new playground for the research community.

{{</citation>}}


### (48/156) TrustMark: Universal Watermarking for Arbitrary Resolution Images (Tu Bui et al., 2023)

{{<citation>}}

Tu Bui, Shruti Agarwal, John Collomosse. (2023)  
**TrustMark: Universal Watermarking for Arbitrary Resolution Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18297v1)  

---


**ABSTRACT**  
Imperceptible digital watermarking is important in copyright protection, misinformation prevention, and responsible generative AI. We propose TrustMark - a GAN-based watermarking method with novel design in architecture and spatio-spectra losses to balance the trade-off between watermarked image quality with the watermark recovery accuracy. Our model is trained with robustness in mind, withstanding various in- and out-place perturbations on the encoded image. Additionally, we introduce TrustMark-RM - a watermark remover method useful for re-watermarking. Our methods achieve state-of-art performance on 3 benchmarks comprising arbitrary resolution images.

{{</citation>}}


### (49/156) Perceptual Group Tokenizer: Building Perception with Iterative Grouping (Zhiwei Deng et al., 2023)

{{<citation>}}

Zhiwei Deng, Ting Chen, Yang Li. (2023)  
**Perceptual Group Tokenizer: Building Perception with Iterative Grouping**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.18296v1)  

---


**ABSTRACT**  
Human visual recognition system shows astonishing capability of compressing visual information into a set of tokens containing rich representations without label supervision. One critical driving principle behind it is perceptual grouping. Despite being widely used in computer vision in the early 2010s, it remains a mystery whether perceptual grouping can be leveraged to derive a neural visual recognition backbone that generates as powerful representations. In this paper, we propose the Perceptual Group Tokenizer, a model that entirely relies on grouping operations to extract visual features and perform self-supervised representation learning, where a series of grouping operations are used to iteratively hypothesize the context for pixels or superpixels to refine feature representations. We show that the proposed model can achieve competitive performance compared to state-of-the-art vision architectures, and inherits desirable properties including adaptive computation without re-training, and interpretability. Specifically, Perceptual Group Tokenizer achieves 80.3% on ImageNet-1K self-supervised learning benchmark with linear probe evaluation, marking a new progress under this paradigm.

{{</citation>}}


### (50/156) SimulFlow: Simultaneously Extracting Feature and Identifying Target for Unsupervised Video Object Segmentation (Lingyi Hong et al., 2023)

{{<citation>}}

Lingyi Hong, Wei Zhang, Shuyong Gao, Hong Lu, WenQiang Zhang. (2023)  
**SimulFlow: Simultaneously Extracting Feature and Identifying Target for Unsupervised Video Object Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.18286v1)  

---


**ABSTRACT**  
Unsupervised video object segmentation (UVOS) aims at detecting the primary objects in a given video sequence without any human interposing. Most existing methods rely on two-stream architectures that separately encode the appearance and motion information before fusing them to identify the target and generate object masks. However, this pipeline is computationally expensive and can lead to suboptimal performance due to the difficulty of fusing the two modalities properly. In this paper, we propose a novel UVOS model called SimulFlow that simultaneously performs feature extraction and target identification, enabling efficient and effective unsupervised video object segmentation. Concretely, we design a novel SimulFlow Attention mechanism to bridege the image and motion by utilizing the flexibility of attention operation, where coarse masks predicted from fused feature at each stage are used to constrain the attention operation within the mask area and exclude the impact of noise. Because of the bidirectional information flow between visual and optical flow features in SimulFlow Attention, no extra hand-designed fusing module is required and we only adopt a light decoder to obtain the final prediction. We evaluate our method on several benchmark datasets and achieve state-of-the-art results. Our proposed approach not only outperforms existing methods but also addresses the computational complexity and fusion difficulties caused by two-stream architectures. Our models achieve 87.4% J & F on DAVIS-16 with the highest speed (63.7 FPS on a 3090) and the lowest parameters (13.7 M). Our SimulFlow also obtains competitive results on video salient object detection datasets.

{{</citation>}}


### (51/156) HKUST at SemEval-2023 Task 1: Visual Word Sense Disambiguation with Context Augmentation and Visual Assistance (Zhuohao Yin et al., 2023)

{{<citation>}}

Zhuohao Yin, Xin Huang. (2023)  
**HKUST at SemEval-2023 Task 1: Visual Word Sense Disambiguation with Context Augmentation and Visual Assistance**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Augmentation, Word Sense Disambiguation  
[Paper Link](http://arxiv.org/abs/2311.18273v1)  

---


**ABSTRACT**  
Visual Word Sense Disambiguation (VWSD) is a multi-modal task that aims to select, among a batch of candidate images, the one that best entails the target word's meaning within a limited context. In this paper, we propose a multi-modal retrieval framework that maximally leverages pretrained Vision-Language models, as well as open knowledge bases and datasets. Our system consists of the following key components: (1) Gloss matching: a pretrained bi-encoder model is used to match contexts with proper senses of the target words; (2) Prompting: matched glosses and other textual information, such as synonyms, are incorporated using a prompting template; (3) Image retrieval: semantically matching images are retrieved from large open datasets using prompts as queries; (4) Modality fusion: contextual information from different modalities are fused and used for prediction. Although our system does not produce the most competitive results at SemEval-2023 Task 1, we are still able to beat nearly half of the teams. More importantly, our experiments reveal acute insights for the field of Word Sense Disambiguation (WSD) and multi-modal learning. Our code is available on GitHub.

{{</citation>}}


### (52/156) Beyond Entropy: Style Transfer Guided Single Image Continual Test-Time Adaptation (Younggeol Cho et al., 2023)

{{<citation>}}

Younggeol Cho, Youngrae Kim, Dongman Lee. (2023)  
**Beyond Entropy: Style Transfer Guided Single Image Continual Test-Time Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2311.18270v1)  

---


**ABSTRACT**  
Continual test-time adaptation (cTTA) methods are designed to facilitate the continual adaptation of models to dynamically changing real-world environments where computational resources are limited. Due to this inherent limitation, existing approaches fail to simultaneously achieve accuracy and efficiency. In detail, when using a single image, the instability caused by batch normalization layers and entropy loss significantly destabilizes many existing methods in real-world cTTA scenarios. To overcome these challenges, we present BESTTA, a novel single image continual test-time adaptation method guided by style transfer, which enables stable and efficient adaptation to the target environment by transferring the style of the input image to the source style. To implement the proposed method, we devise BeIN, a simple yet powerful normalization method, along with the style-guided losses. We demonstrate that BESTTA effectively adapts to the continually changing target environment, leveraging only a single image on both semantic segmentation and image classification tasks. Remarkably, despite training only two parameters in a BeIN layer consuming the least memory, BESTTA outperforms existing state-of-the-art methods in terms of performance.

{{</citation>}}


### (53/156) Diffusion Models Without Attention (Jing Nathan Yan et al., 2023)

{{<citation>}}

Jing Nathan Yan, Jiatao Gu, Alexander M. Rush. (2023)  
**Diffusion Models Without Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention, ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2311.18257v1)  

---


**ABSTRACT**  
In recent advancements in high-fidelity image generation, Denoising Diffusion Probabilistic Models (DDPMs) have emerged as a key player. However, their application at high resolutions presents significant computational challenges. Current methods, such as patchifying, expedite processes in UNet and Transformer architectures but at the expense of representational capacity. Addressing this, we introduce the Diffusion State Space Model (DiffuSSM), an architecture that supplants attention mechanisms with a more scalable state space model backbone. This approach effectively handles higher resolutions without resorting to global compression, thus preserving detailed image representation throughout the diffusion process. Our focus on FLOP-efficient architectures in diffusion training marks a significant step forward. Comprehensive evaluations on both ImageNet and LSUN datasets at two resolutions demonstrate that DiffuSSMs are on par or even outperform existing diffusion models with attention modules in FID and Inception Score metrics while significantly reducing total FLOP usage.

{{</citation>}}


### (54/156) Sketch Input Method Editor: A Comprehensive Dataset and Methodology for Systematic Input Recognition (Guangming Zhu et al., 2023)

{{<citation>}}

Guangming Zhu, Siyuan Wang, Qing Cheng, Kelong Wu, Hao Li, Liang Zhang. (2023)  
**Sketch Input Method Editor: A Comprehensive Dataset and Methodology for Systematic Input Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2311.18254v1)  

---


**ABSTRACT**  
With the recent surge in the use of touchscreen devices, free-hand sketching has emerged as a promising modality for human-computer interaction. While previous research has focused on tasks such as recognition, retrieval, and generation of familiar everyday objects, this study aims to create a Sketch Input Method Editor (SketchIME) specifically designed for a professional C4I system. Within this system, sketches are utilized as low-fidelity prototypes for recommending standardized symbols in the creation of comprehensive situation maps. This paper also presents a systematic dataset comprising 374 specialized sketch types, and proposes a simultaneous recognition and segmentation architecture with multilevel supervision between recognition and segmentation to improve performance and enhance interpretability. By incorporating few-shot domain adaptation and class-incremental learning, the network's ability to adapt to new users and extend to new task-specific classes is significantly enhanced. Results from experiments conducted on both the proposed dataset and the SPG dataset illustrate the superior performance of the proposed architecture. Our dataset and code are publicly available at https://github.com/Anony517/SketchIME.

{{</citation>}}


### (55/156) Label-efficient Training of Small Task-specific Models by Leveraging Vision Foundation Models (Raviteja Vemulapalli et al., 2023)

{{<citation>}}

Raviteja Vemulapalli, Hadi Pouransari, Fartash Faghri, Sachin Mehta, Mehrdad Farajtabar, Mohammad Rastegari, Oncel Tuzel. (2023)  
**Label-efficient Training of Small Task-specific Models by Leveraging Vision Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.18237v1)  

---


**ABSTRACT**  
Large Vision Foundation Models (VFMs) pretrained on massive datasets exhibit impressive performance on various downstream tasks, especially with limited labeled target data. However, due to their high memory and compute requirements, these models cannot be deployed in resource constrained settings. This raises an important question: How can we utilize the knowledge from a large VFM to train a small task-specific model for a new target task with limited labeled training data? In this work, we answer this question by proposing a simple and highly effective task-oriented knowledge transfer approach to leverage pretrained VFMs for effective training of small task-specific models. Our experimental results on four target tasks under limited labeled data settings show that the proposed knowledge transfer approach outperforms task-agnostic VFM distillation, web-scale CLIP pretraining and supervised ImageNet pretraining by 1-10.5%, 2-22% and 2-14%, respectively. We also show that the dataset used for transferring knowledge has a significant effect on the final target task performance, and propose an image retrieval-based approach for curating effective transfer sets.

{{</citation>}}


### (56/156) TCP:Textual-based Class-aware Prompt tuning for Visual-Language Model (Hantao Yao et al., 2023)

{{<citation>}}

Hantao Yao, Rui Zhang, Changsheng Xu. (2023)  
**TCP:Textual-based Class-aware Prompt tuning for Visual-Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2311.18231v1)  

---


**ABSTRACT**  
Prompt tuning represents a valuable technique for adapting pre-trained visual-language models (VLM) to various downstream tasks. Recent advancements in CoOp-based methods propose a set of learnable domain-shared or image-conditional textual tokens to facilitate the generation of task-specific textual classifiers. However, those textual tokens have a limited generalization ability regarding unseen domains, as they cannot dynamically adjust to the distribution of testing classes. To tackle this issue, we present a novel Textual-based Class-aware Prompt tuning(TCP) that explicitly incorporates prior knowledge about classes to enhance their discriminability. The critical concept of TCP involves leveraging Textual Knowledge Embedding (TKE) to map the high generalizability of class-level textual knowledge into class-aware textual tokens. By seamlessly integrating these class-aware prompts into the Text Encoder, a dynamic class-aware classifier is generated to enhance discriminability for unseen domains. During inference, TKE dynamically generates class-aware prompts related to the unseen classes. Comprehensive evaluations demonstrate that TKE serves as a plug-and-play module effortlessly combinable with existing methods. Furthermore, TCP consistently achieves superior performance while demanding less training time.

{{</citation>}}


### (57/156) FS-BAND: A Frequency-Sensitive Banding Detector (Zijian Chen et al., 2023)

{{<citation>}}

Zijian Chen, Wei Sun, Zicheng Zhang, Ru Huang, Fangfang Lu, Xiongkuo Min, Guangtao Zhai, Wenjun Zhang. (2023)  
**FS-BAND: A Frequency-Sensitive Banding Detector**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV, eess-IV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.18216v1)  

---


**ABSTRACT**  
Banding artifact, as known as staircase-like contour, is a common quality annoyance that happens in compression, transmission, etc. scenarios, which largely affects the user's quality of experience (QoE). The banding distortion typically appears as relatively small pixel-wise variations in smooth backgrounds, which is difficult to analyze in the spatial domain but easily reflected in the frequency domain. In this paper, we thereby study the banding artifact from the frequency aspect and propose a no-reference banding detection model to capture and evaluate banding artifacts, called the Frequency-Sensitive BANding Detector (FS-BAND). The proposed detector is able to generate a pixel-wise banding map with a perception correlated quality score. Experimental results show that the proposed FS-BAND method outperforms state-of-the-art image quality assessment (IQA) approaches with higher accuracy in banding classification task.

{{</citation>}}


### (58/156) Compact3D: Compressing Gaussian Splat Radiance Field Models with Vector Quantization (KL Navaneet et al., 2023)

{{<citation>}}

KL Navaneet, Kossar Pourahmadi Meibodi, Soroush Abbasi Koohpayegani, Hamed Pirsiavash. (2023)  
**Compact3D: Compressing Gaussian Splat Radiance Field Models with Vector Quantization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2311.18159v1)  

---


**ABSTRACT**  
3D Gaussian Splatting is a new method for modeling and rendering 3D radiance fields that achieves much faster learning and rendering time compared to SOTA NeRF methods. However, it comes with a drawback in the much larger storage demand compared to NeRF methods since it needs to store the parameters for several 3D Gaussians. We notice that many Gaussians may share similar parameters, so we introduce a simple vector quantization method based on \kmeans algorithm to quantize the Gaussian parameters. Then, we store the small codebook along with the index of the code for each Gaussian. Moreover, we compress the indices further by sorting them and using a method similar to run-length encoding. We do extensive experiments on standard benchmarks as well as a new benchmark which is an order of magnitude larger than the standard benchmarks. We show that our simple yet effective method can reduce the storage cost for the original 3D Gaussian Splatting method by a factor of almost $20\times$ with a very small drop in the quality of rendered images.

{{</citation>}}


## cs.CY (4)



### (59/156) The role of interface design on prompt-mediated creativity in Generative AI (Maddalena Torricelli et al., 2023)

{{<citation>}}

Maddalena Torricelli, Mauro Martino, Andrea Baronchelli, Luca Maria Aiello. (2023)  
**The role of interface design on prompt-mediated creativity in Generative AI**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-HC, cs.CY, physics-soc-ph  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.00233v1)  

---


**ABSTRACT**  
Generative AI models for the creation of images is becoming a staple in the toolkit of digital artists and visual designers. The interaction with these systems is mediated by prompting, a process in which users write a short text to describe the desired image's content and style. The study of prompts offers an unprecedented opportunity to gain insight into the process of human creativity, yet our understanding of how people use them remains limited. We analyze more than 145,000 prompts from the logs of two Generative AI platforms (Stable Diffusion and Pick-a-Pic) to shed light on how people explore new concepts over time, and how their exploration might be influenced by different design choices in human-computer interfaces to Generative AI. We find that users exhibit a tendency towards exploration of new topics over exploitation of concepts visited previously. However, a comparative analysis of the two platforms, which differ both in scope and functionalities, reveals that the introduction of features diverting user focus from prompting and providing instead shortcuts for generating new image variants with simple clicks is associated with a considerable reduction in both exploration of novel concepts and detail in the submitted prompts. These results carry direct implications for the design of human interfaces to Generative AI and raise new questions regarding how the process of prompting should be aided in ways that best support creativity.

{{</citation>}}


### (60/156) Towards Accurate Differential Diagnosis with Large Language Models (Daniel McDuff et al., 2023)

{{<citation>}}

Daniel McDuff, Mike Schaekermann, Tao Tu, Anil Palepu, Amy Wang, Jake Garrison, Karan Singhal, Yash Sharma, Shekoofeh Azizi, Kavita Kulkarni, Le Hou, Yong Cheng, Yun Liu, S Sara Mahdavi, Sushant Prakash, Anupam Pathak, Christopher Semturs, Shwetak Patel, Dale R Webster, Ewa Dominowska, Juraj Gottweis, Joelle Barral, Katherine Chou, Greg S Corrado, Yossi Matias, Jake Sunshine, Alan Karthikesalingam, Vivek Natarajan. (2023)  
**Towards Accurate Differential Diagnosis with Large Language Models**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00164v1)  

---


**ABSTRACT**  
An accurate differential diagnosis (DDx) is a cornerstone of medical care, often reached through an iterative process of interpretation that combines clinical history, physical examination, investigations and procedures. Interactive interfaces powered by Large Language Models (LLMs) present new opportunities to both assist and automate aspects of this process. In this study, we introduce an LLM optimized for diagnostic reasoning, and evaluate its ability to generate a DDx alone or as an aid to clinicians. 20 clinicians evaluated 302 challenging, real-world medical cases sourced from the New England Journal of Medicine (NEJM) case reports. Each case report was read by two clinicians, who were randomized to one of two assistive conditions: either assistance from search engines and standard medical resources, or LLM assistance in addition to these tools. All clinicians provided a baseline, unassisted DDx prior to using the respective assistive tools. Our LLM for DDx exhibited standalone performance that exceeded that of unassisted clinicians (top-10 accuracy 59.1% vs 33.6%, [p = 0.04]). Comparing the two assisted study arms, the DDx quality score was higher for clinicians assisted by our LLM (top-10 accuracy 51.7%) compared to clinicians without its assistance (36.1%) (McNemar's Test: 45.7, p < 0.01) and clinicians with search (44.4%) (4.75, p = 0.03). Further, clinicians assisted by our LLM arrived at more comprehensive differential lists than those without its assistance. Our study suggests that our LLM for DDx has potential to improve clinicians' diagnostic reasoning and accuracy in challenging cases, meriting further real-world evaluation for its ability to empower physicians and widen patients' access to specialist-level expertise.

{{</citation>}}


### (61/156) Situating the social issues of image generation models in the model life cycle: a sociotechnical approach (Amelia Katirai et al., 2023)

{{<citation>}}

Amelia Katirai, Noa Garcia, Kazuki Ide, Yuta Nakashima, Atsuo Kishimoto. (2023)  
**Situating the social issues of image generation models in the model life cycle: a sociotechnical approach**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18345v1)  

---


**ABSTRACT**  
The race to develop image generation models is intensifying, with a rapid increase in the number of text-to-image models available. This is coupled with growing public awareness of these technologies. Though other generative AI models--notably, large language models--have received recent critical attention for the social and other non-technical issues they raise, there has been relatively little comparable examination of image generation models. This paper reports on a novel, comprehensive categorization of the social issues associated with image generation models. At the intersection of machine learning and the social sciences, we report the results of a survey of the literature, identifying seven issue clusters arising from image generation models: data issues, intellectual property, bias, privacy, and the impacts on the informational, cultural, and natural environments. We situate these social issues in the model life cycle, to aid in considering where potential issues arise, and mitigation may be needed. We then compare these issue clusters with what has been reported for large language models. Ultimately, we argue that the risks posed by image generation models are comparable in severity to the risks posed by large language models, and that the social impact of image generation models must be urgently considered.

{{</citation>}}


### (62/156) Generative Artificial Intelligence in Learning Analytics: Contextualising Opportunities and Challenges through the Learning Analytics Cycle (Lixiang Yan et al., 2023)

{{<citation>}}

Lixiang Yan, Roberto Martinez-Maldonado, Dragan Gašević. (2023)  
**Generative Artificial Intelligence in Learning Analytics: Contextualising Opportunities and Challenges through the Learning Analytics Cycle**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-HC, cs.CY  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.00087v1)  

---


**ABSTRACT**  
Generative artificial intelligence (GenAI), exemplified by ChatGPT, Midjourney, and other state-of-the-art large language models and diffusion models, holds significant potential for transforming education and enhancing human productivity. While the prevalence of GenAI in education has motivated numerous research initiatives, integrating these technologies within the learning analytics (LA) cycle and their implications for practical interventions remain underexplored. This paper delves into the prospective opportunities and challenges GenAI poses for advancing LA. We present a concise overview of the current GenAI landscape and contextualise its potential roles within Clow's generic framework of the LA cycle. We posit that GenAI can play pivotal roles in analysing unstructured data, generating synthetic learner data, enriching multimodal learner interactions, advancing interactive and explanatory analytics, and facilitating personalisation and adaptive interventions. As the lines blur between learners and GenAI tools, a renewed understanding of learners is needed. Future research can delve deep into frameworks and methodologies that advocate for human-AI collaboration. The LA community can play a pivotal role in capturing data about human and AI contributions and exploring how they can collaborate most effectively. As LA advances, it is essential to consider the pedagogical implications and broader socioeconomic impact of GenAI for ensuring an inclusive future.

{{</citation>}}


## cs.LG (30)



### (63/156) Uncertainty in Graph Contrastive Learning with Bayesian Neural Networks (Alexander Möllers et al., 2023)

{{<citation>}}

Alexander Möllers, Alexander Immer, Elvin Isufi, Vincent Fortuin. (2023)  
**Uncertainty in Graph Contrastive Learning with Bayesian Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.00232v1)  

---


**ABSTRACT**  
Graph contrastive learning has shown great promise when labeled data is scarce, but large unlabeled datasets are available. However, it often does not take uncertainty estimation into account. We show that a variational Bayesian neural network approach can be used to improve not only the uncertainty estimates but also the downstream performance on semi-supervised node-classification tasks. Moreover, we propose a new measure of uncertainty for contrastive learning, that is based on the disagreement in likelihood due to different positive samples.

{{</citation>}}


### (64/156) Optimal Attack and Defense for Reinforcement Learning (Jeremy McMahan et al., 2023)

{{<citation>}}

Jeremy McMahan, Young Wu, Xiaojin Zhu, Qiaomin Xie. (2023)  
**Optimal Attack and Defense for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-GT, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.00198v1)  

---


**ABSTRACT**  
To ensure the usefulness of Reinforcement Learning (RL) in real systems, it is crucial to ensure they are robust to noise and adversarial attacks. In adversarial RL, an external attacker has the power to manipulate the victim agent's interaction with the environment. We study the full class of online manipulation attacks, which include (i) state attacks, (ii) observation attacks (which are a generalization of perceived-state attacks), (iii) action attacks, and (iv) reward attacks. We show the attacker's problem of designing a stealthy attack that maximizes its own expected reward, which often corresponds to minimizing the victim's value, is captured by a Markov Decision Process (MDP) that we call a meta-MDP since it is not the true environment but a higher level environment induced by the attacked interaction. We show that the attacker can derive optimal attacks by planning in polynomial time or learning with polynomial sample complexity using standard RL techniques. We argue that the optimal defense policy for the victim can be computed as the solution to a stochastic Stackelberg game, which can be further simplified into a partially-observable turn-based stochastic game (POTBSG). Neither the attacker nor the victim would benefit from deviating from their respective optimal policies, thus such solutions are truly robust. Although the defense problem is NP-hard, we show that optimal Markovian defenses can be computed (learned) in polynomial time (sample complexity) in many scenarios.

{{</citation>}}


### (65/156) Robust Concept Erasure via Kernelized Rate-Distortion Maximization (Somnath Basu Roy Chowdhury et al., 2023)

{{<citation>}}

Somnath Basu Roy Chowdhury, Nicholas Monath, Avinava Dubey, Amr Ahmed, Snigdha Chaturvedi. (2023)  
**Robust Concept Erasure via Kernelized Rate-Distortion Maximization**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2312.00194v1)  

---


**ABSTRACT**  
Distributed representations provide a vector space that captures meaningful relationships between data instances. The distributed nature of these representations, however, entangles together multiple attributes or concepts of data instances (e.g., the topic or sentiment of a text, characteristics of the author (age, gender, etc), etc). Recent work has proposed the task of concept erasure, in which rather than making a concept predictable, the goal is to remove an attribute from distributed representations while retaining other information from the original representation space as much as possible. In this paper, we propose a new distance metric learning-based objective, the Kernelized Rate-Distortion Maximizer (KRaM), for performing concept erasure. KRaM fits a transformation of representations to match a specified distance measure (defined by a labeled concept to erase) using a modified rate-distortion function. Specifically, KRaM's objective function aims to make instances with similar concept labels dissimilar in the learned representation space while retaining other information. We find that optimizing KRaM effectively erases various types of concepts: categorical, continuous, and vector-valued variables from data representations across diverse domains. We also provide a theoretical analysis of several properties of KRaM's objective. To assess the quality of the learned representations, we propose an alignment score to evaluate their similarity with the original representation space. Additionally, we conduct experiments to showcase KRaM's efficacy in various settings, from erasing binary gender variables in word embeddings to vector-valued variables in GPT-3 representations.

{{</citation>}}


### (66/156) Benchmarking and Enhancing Disentanglement in Concept-Residual Models (Renos Zabounidis et al., 2023)

{{<citation>}}

Renos Zabounidis, Ini Oguntola, Konghao Zhao, Joseph Campbell, Simon Stepputtis, Katia Sycara. (2023)  
**Benchmarking and Enhancing Disentanglement in Concept-Residual Models**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00192v1)  

---


**ABSTRACT**  
Concept bottleneck models (CBMs) are interpretable models that first predict a set of semantically meaningful features, i.e., concepts, from observations that are subsequently used to condition a downstream task. However, the model's performance strongly depends on the engineered features and can severely suffer from incomplete sets of concepts. Prior works have proposed a side channel -- a residual -- that allows for unconstrained information flow to the downstream task, thus improving model performance but simultaneously introducing information leakage, which is undesirable for interpretability. This work proposes three novel approaches to mitigate information leakage by disentangling concepts and residuals, investigating the critical balance between model performance and interpretability. Through extensive empirical analysis on the CUB, OAI, and CIFAR 100 datasets, we assess the performance of each disentanglement method and provide insights into when they work best. Further, we show how each method impacts the ability to intervene over the concepts and their subsequent impact on task performance.

{{</citation>}}


### (67/156) HeTriNet: Heterogeneous Graph Triplet Attention Network for Drug-Target-Disease Interaction (Farhan Tanvir et al., 2023)

{{<citation>}}

Farhan Tanvir, Khaled Mohammed Saifuddin, Tanvir Hossain, Arunkumar Bagavathi, Esra Akbas. (2023)  
**HeTriNet: Heterogeneous Graph Triplet Attention Network for Drug-Target-Disease Interaction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-bio-BM  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.00189v1)  

---


**ABSTRACT**  
Modeling the interactions between drugs, targets, and diseases is paramount in drug discovery and has significant implications for precision medicine and personalized treatments. Current approaches frequently consider drug-target or drug-disease interactions individually, ignoring the interdependencies among all three entities. Within human metabolic systems, drugs interact with protein targets in cells, influencing target activities and subsequently impacting biological pathways to promote healthy functions and treat diseases. Moving beyond binary relationships and exploring tighter triple relationships is essential to understanding drugs' mechanism of action (MoAs). Moreover, identifying the heterogeneity of drugs, targets, and diseases, along with their distinct characteristics, is critical to model these complex interactions appropriately. To address these challenges, we effectively model the interconnectedness of all entities in a heterogeneous graph and develop a novel Heterogeneous Graph Triplet Attention Network (\texttt{HeTriNet}). \texttt{HeTriNet} introduces a novel triplet attention mechanism within this heterogeneous graph structure. Beyond pairwise attention as the importance of an entity for the other one, we define triplet attention to model the importance of pairs for entities in the drug-target-disease triplet prediction problem. Experimental results on real-world datasets show that \texttt{HeTriNet} outperforms several baselines, demonstrating its remarkable proficiency in uncovering novel drug-target-disease relationships.

{{</citation>}}


### (68/156) Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking (Kaifeng Lyu et al., 2023)

{{<citation>}}

Kaifeng Lyu, Jikai Jin, Zhiyuan Li, Simon S. Du, Jason D. Lee, Wei Hu. (2023)  
**Dichotomy of Early and Late Phase Implicit Biases Can Provably Induce Grokking**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.18817v1)  

---


**ABSTRACT**  
Recent work by Power et al. (2022) highlighted a surprising "grokking" phenomenon in learning arithmetic tasks: a neural net first "memorizes" the training set, resulting in perfect training accuracy but near-random test accuracy, and after training for sufficiently longer, it suddenly transitions to perfect test accuracy. This paper studies the grokking phenomenon in theoretical setups and shows that it can be induced by a dichotomy of early and late phase implicit biases. Specifically, when training homogeneous neural nets with large initialization and small weight decay on both classification and regression tasks, we prove that the training process gets trapped at a solution corresponding to a kernel predictor for a long time, and then a very sharp transition to min-norm/max-margin predictors occurs, leading to a dramatic change in test accuracy.

{{</citation>}}


### (69/156) Multimodal Learning for Crystalline Materials (Viggo Moro et al., 2023)

{{<citation>}}

Viggo Moro, Charlotte Loh, Rumen Dangovski, Ali Ghorashi, Andrew Ma, Zhuo Chen, Peter Y. Lu, Thomas Christensen, Marin Soljačić. (2023)  
**Multimodal Learning for Crystalline Materials**  

---
Primary Category: cs.LG  
Categories: cond-mat-mtrl-sci, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00111v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) has revolutionized the field of materials science by improving the prediction of properties and accelerating the discovery of novel materials. In recent years, publicly available material data repositories containing data for various material properties have grown rapidly. In this work, we introduce Multimodal Learning for Crystalline Materials (MLCM), a new method for training a foundation model for crystalline materials via multimodal alignment, where high-dimensional material properties (i.e. modalities) are connected in a shared latent space to produce highly useful material representations. We show the utility of MLCM on multiple axes: (i) MLCM achieves state-of-the-art performance for material property prediction on the challenging Materials Project database; (ii) MLCM enables a novel, highly accurate method for inverse design, allowing one to screen for stable material with desired properties; and (iii) MLCM allows the extraction of interpretable emergent features that may provide insight to material scientists. Further, we explore several novel methods for aligning an arbitrary number of modalities, improving upon prior art in multimodal learning that focuses on bimodal alignment. Our work brings innovations from the ongoing AI revolution into the domain of materials science and identifies materials as a testbed for the next generation of AI.

{{</citation>}}


### (70/156) MultiResFormer: Transformer with Adaptive Multi-Resolution Modeling for General Time Series Forecasting (Linfeng Du et al., 2023)

{{<citation>}}

Linfeng Du, Ji Xin, Alex Labach, Saba Zuberi, Maksims Volkovs, Rahul G. Krishnan. (2023)  
**MultiResFormer: Transformer with Adaptive Multi-Resolution Modeling for General Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2311.18780v1)  

---


**ABSTRACT**  
Transformer-based models have greatly pushed the boundaries of time series forecasting recently. Existing methods typically encode time series data into $\textit{patches}$ using one or a fixed set of patch lengths. This, however, could result in a lack of ability to capture the variety of intricate temporal dependencies present in real-world multi-periodic time series. In this paper, we propose MultiResFormer, which dynamically models temporal variations by adaptively choosing optimal patch lengths. Concretely, at the beginning of each layer, time series data is encoded into several parallel branches, each using a detected periodicity, before going through the transformer encoder block. We conduct extensive evaluations on long- and short-term forecasting datasets comparing MultiResFormer with state-of-the-art baselines. MultiResFormer outperforms patch-based Transformer baselines on long-term forecasting tasks and also consistently outperforms CNN baselines by a large margin, while using much fewer parameters than these baselines.

{{</citation>}}


### (71/156) Language Model Agents Suffer from Compositional Generalization in Web Automation (Hiroki Furuta et al., 2023)

{{<citation>}}

Hiroki Furuta, Yutaka Matsuo, Aleksandra Faust, Izzeddin Gur. (2023)  
**Language Model Agents Suffer from Compositional Generalization in Web Automation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model, T5  
[Paper Link](http://arxiv.org/abs/2311.18751v1)  

---


**ABSTRACT**  
Language model agents (LMA) recently emerged as a promising paradigm on muti-step decision making tasks, often outperforming humans and other reinforcement learning agents. Despite the promise, their performance on real-world applications that often involve combinations of tasks is still underexplored. In this work, we introduce a new benchmark, called CompWoB -- 50 new compositional web automation tasks reflecting more realistic assumptions. We show that while existing prompted LMAs (gpt-3.5-turbo or gpt-4) achieve 94.0% average success rate on base tasks, their performance degrades to 24.9% success rate on compositional tasks. On the other hand, transferred LMAs (finetuned only on base tasks) show less generalization gap, dropping from 85.4% to 54.8%. By balancing data distribution across tasks, we train a new model, HTML-T5++, that surpasses human-level performance (95.2%) on MiniWoB, and achieves the best zero-shot performance on CompWoB (61.5%). While these highlight the promise of small-scale finetuned and transferred models for compositional generalization, their performance further degrades under different instruction compositions changing combinational order. In contrast to the recent remarkable success of LMA, our benchmark and detailed analysis emphasize the necessity of building LMAs that are robust and generalizable to task compositionality for real-world deployment.

{{</citation>}}


### (72/156) TransCORALNet: A Two-Stream Transformer CORAL Networks for Supply Chain Credit Assessment Cold Start (Jie Shi et al., 2023)

{{<citation>}}

Jie Shi, Arno P. J. M. Siebes, Siamak Mehrkanoon. (2023)  
**TransCORALNet: A Two-Stream Transformer CORAL Networks for Supply Chain Credit Assessment Cold Start**  

---
Primary Category: cs.LG  
Categories: I-2; I-5, cs-AI, cs-LG, cs.LG, q-fin-RM  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.18749v1)  

---


**ABSTRACT**  
This paper proposes an interpretable two-stream transformer CORAL networks (TransCORALNet) for supply chain credit assessment under the segment industry and cold start problem. The model aims to provide accurate credit assessment prediction for new supply chain borrowers with limited historical data. Here, the two-stream domain adaptation architecture with correlation alignment (CORAL) loss is used as a core model and is equipped with transformer, which provides insights about the learned features and allow efficient parallelization during training. Thanks to the domain adaptation capability of the proposed model, the domain shift between the source and target domain is minimized. Therefore, the model exhibits good generalization where the source and target do not follow the same distribution, and a limited amount of target labeled instances exist. Furthermore, we employ Local Interpretable Model-agnostic Explanations (LIME) to provide more insight into the model prediction and identify the key features contributing to supply chain credit assessment decisions. The proposed model addresses four significant supply chain credit assessment challenges: domain shift, cold start, imbalanced-class and interpretability. Experimental results on a real-world data set demonstrate the superiority of TransCORALNet over a number of state-of-the-art baselines in terms of accuracy. The code is available on GitHub https://github.com/JieJieNiu/TransCORALN .

{{</citation>}}


### (73/156) Dimension Mixer: A Generalized Method for Structured Sparsity in Deep Neural Networks (Suman Sapkota et al., 2023)

{{<citation>}}

Suman Sapkota, Binod Bhattarai. (2023)  
**Dimension Mixer: A Generalized Method for Structured Sparsity in Deep Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.18735v1)  

---


**ABSTRACT**  
The recent success of multiple neural architectures like CNNs, Transformers, and MLP-Mixers motivated us to look for similarities and differences between them. We found that these architectures can be interpreted through the lens of a general concept of dimension mixing. Research on coupling flows and the butterfly transform shows that partial and hierarchical signal mixing schemes are sufficient for efficient and expressive function approximation. In this work, we study group-wise sparse, non-linear, multi-layered and learnable mixing schemes of inputs and find that they are complementary to many standard neural architectures. Following our observations and drawing inspiration from the Fast Fourier Transform, we generalize Butterfly Structure to use non-linear mixer function allowing for MLP as mixing function called Butterfly MLP. We were also able to mix along sequence dimension for Transformer-based architectures called Butterfly Attention. Experiments on CIFAR and LRA datasets demonstrate that the proposed Non-Linear Butterfly Mixers are efficient and scale well when the host architectures are used as mixing function. Additionally, we propose Patch-Only MLP-Mixer for processing spatial 2D signals demonstrating a different dimension mixing strategy.

{{</citation>}}


### (74/156) DeepEn2023: Energy Datasets for Edge Artificial Intelligence (Xiaolong Tu et al., 2023)

{{<citation>}}

Xiaolong Tu, Anik Mallik, Haoxin Wang, Jiang Xie. (2023)  
**DeepEn2023: Energy Datasets for Edge Artificial Intelligence**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-PF, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00103v1)  

---


**ABSTRACT**  
Climate change poses one of the most significant challenges to humanity. As a result of these climatic changes, the frequency of weather, climate, and water-related disasters has multiplied fivefold over the past 50 years, resulting in over 2 million deaths and losses exceeding $3.64 trillion USD. Leveraging AI-powered technologies for sustainable development and combating climate change is a promising avenue. Numerous significant publications are dedicated to using AI to improve renewable energy forecasting, enhance waste management, and monitor environmental changes in real time. However, very few research studies focus on making AI itself environmentally sustainable. This oversight regarding the sustainability of AI within the field might be attributed to a mindset gap and the absence of comprehensive energy datasets. In addition, with the ubiquity of edge AI systems and applications, especially on-device learning, there is a pressing need to measure, analyze, and optimize their environmental sustainability, such as energy efficiency. To this end, in this paper, we propose large-scale energy datasets for edge AI, named DeepEn2023, covering a wide range of kernels, state-of-the-art deep neural network models, and popular edge AI applications. We anticipate that DeepEn2023 will improve transparency in sustainability in on-device deep learning across a range of edge AI systems and applications. For more information, including access to the dataset and code, please visit https://amai-gsu.github.io/DeepEn2023.

{{</citation>}}


### (75/156) Predictable Reinforcement Learning Dynamics through Entropy Rate Minimization (Daniel Jarne Ornia et al., 2023)

{{<citation>}}

Daniel Jarne Ornia, Giannis Delimpaltadakis, Jens Kober, Javier Alonso-Mora. (2023)  
**Predictable Reinforcement Learning Dynamics through Entropy Rate Minimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.18703v1)  

---


**ABSTRACT**  
In Reinforcement Learning (RL), agents have no incentive to exhibit predictable behaviors, and are often pushed (through e.g. policy entropy regularization) to randomize their actions in favor of exploration. From a human perspective, this makes RL agents hard to interpret and predict, and from a safety perspective, even harder to formally verify. We propose a novel method to induce predictable behavior in RL agents, referred to as Predictability-Aware RL (PA-RL), which employs the state sequence entropy rate as a predictability measure. We show how the entropy rate can be formulated as an average reward objective, and since its entropy reward function is policy-dependent, we introduce an action-dependent surrogate entropy enabling the use of PG methods. We prove that deterministic policies minimizing the average surrogate reward exist and also minimize the actual entropy rate, and show how, given a learned dynamical model, we are able to approximate the value function associated to the true entropy rate. Finally, we demonstrate the effectiveness of the approach in RL tasks inspired by human-robot use-cases, and show how it produces agents with more predictable behavior while achieving near-optimal rewards.

{{</citation>}}


### (76/156) Handling Cost and Constraints with Off-Policy Deep Reinforcement Learning (Jared Markowitz et al., 2023)

{{<citation>}}

Jared Markowitz, Jesse Silverberg, Gary Collins. (2023)  
**Handling Cost and Constraints with Off-Policy Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.18684v1)  

---


**ABSTRACT**  
By reusing data throughout training, off-policy deep reinforcement learning algorithms offer improved sample efficiency relative to on-policy approaches. For continuous action spaces, the most popular methods for off-policy learning include policy improvement steps where a learned state-action ($Q$) value function is maximized over selected batches of data. These updates are often paired with regularization to combat associated overestimation of $Q$ values. With an eye toward safety, we revisit this strategy in environments with "mixed-sign" reward functions; that is, with reward functions that include independent positive (incentive) and negative (cost) terms. This setting is common in real-world applications, and may be addressed with or without constraints on the cost terms. We find the combination of function approximation and a term that maximizes $Q$ in the policy update to be problematic in such environments, because systematic errors in value estimation impact the contributions from the competing terms asymmetrically. This results in overemphasis of either incentives or costs and may severely limit learning. We explore two remedies to this issue. First, consistent with prior work, we find that periodic resetting of $Q$ and policy networks can be used to reduce value estimation error and improve learning in this setting. Second, we formulate novel off-policy actor-critic methods for both unconstrained and constrained learning that do not explicitly maximize $Q$ in the policy update. We find that this second approach, when applied to continuous action spaces with mixed-sign rewards, consistently and significantly outperforms state-of-the-art methods augmented by resetting. We further find that our approach produces agents that are both competitive with popular methods overall and more reliably competent on frequently-studied control problems that do not have mixed-sign rewards.

{{</citation>}}


### (77/156) FedEmb: A Vertical and Hybrid Federated Learning Algorithm using Network And Feature Embedding Aggregation (Fanfei Meng et al., 2023)

{{<citation>}}

Fanfei Meng, Lele Zhang, Yu Chen, Yuxin Wang. (2023)  
**FedEmb: A Vertical and Hybrid Federated Learning Algorithm using Network And Feature Embedding Aggregation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.00102v2)  

---


**ABSTRACT**  
Federated learning (FL) is an emerging paradigm for decentralized training of machine learning models on distributed clients, without revealing the data to the central server. The learning scheme may be horizontal, vertical or hybrid (both vertical and horizontal). Most existing research work with deep neural network (DNN) modelling is focused on horizontal data distributions, while vertical and hybrid schemes are much less studied. In this paper, we propose a generalized algorithm FedEmb, for modelling vertical and hybrid DNN-based learning. The idea of our algorithm is characterised by higher inference accuracy, stronger privacy-preserving properties, and lower client-server communication bandwidth demands as compared with existing work. The experimental results show that FedEmb is an effective method to tackle both split feature & subject space decentralized problems, shows 0.3% to 4.2% inference accuracy improvement with limited privacy revealing for datasets stored in local clients, and reduces 88.9 % time complexity over vertical baseline method.

{{</citation>}}


### (78/156) Communication-Efficient Heterogeneous Federated Learning with Generalized Heavy-Ball Momentum (Riccardo Zaccone et al., 2023)

{{<citation>}}

Riccardo Zaccone, Carlo Masone, Marco Ciccone. (2023)  
**Communication-Efficient Heterogeneous Federated Learning with Generalized Heavy-Ball Momentum**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.18578v1)  

---


**ABSTRACT**  
Federated Learning (FL) is the state-of-the-art approach for learning from decentralized data in privacy-constrained scenarios. As the current literature reports, the main problems associated with FL refer to system and statistical challenges: the former ones demand for efficient learning from edge devices, including lowering communication bandwidth and frequency, while the latter require algorithms robust to non-iidness. State-of-art approaches either guarantee convergence at increased communication cost or are not sufficiently robust to handle extreme heterogeneous local distributions. In this work we propose a novel generalization of the heavy-ball momentum, and present FedHBM to effectively address statistical heterogeneity in FL without introducing any communication overhead. We conduct extensive experimentation on common FL vision and NLP datasets, showing that our FedHBM algorithm empirically yields better model quality and higher convergence speed w.r.t. the state-of-art, especially in pathological non-iid scenarios. While being designed for cross-silo settings, we show how FedHBM is applicable in moderate-to-high cross-device scenarios, and how good model initializations (e.g. pre-training) can be exploited for prompt acceleration. Extended experimentation on large-scale real-world federated datasets further corroborates the effectiveness of our approach for real-world FL applications.

{{</citation>}}


### (79/156) Class Distribution Shifts in Zero-Shot Learning: Learning Robust Representations (Yuli Slavutsky et al., 2023)

{{<citation>}}

Yuli Slavutsky, Yuval Benjamini. (2023)  
**Class Distribution Shifts in Zero-Shot Learning: Learning Robust Representations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.18575v1)  

---


**ABSTRACT**  
Distribution shifts between training and deployment data often affect the performance of machine learning models. In this paper, we explore a setting where a hidden variable induces a shift in the distribution of classes. These distribution shifts are particularly challenging for zero-shot classifiers, as they rely on representations learned from training classes, but are deployed on new, unseen ones. We introduce an algorithm to learn data representations that are robust to such class distribution shifts in zero-shot verification tasks. We show that our approach, which combines hierarchical data sampling with out-of-distribution generalization techniques, improves generalization to diverse class distributions in both simulations and real-world datasets.

{{</citation>}}


### (80/156) Textual-Knowledge-Guided Numerical Feature Discovery Method for Power Demand Forecasting (Zifan Ning et al., 2023)

{{<citation>}}

Zifan Ning, Min Jin. (2023)  
**Textual-Knowledge-Guided Numerical Feature Discovery Method for Power Demand Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00095v1)  

---


**ABSTRACT**  
Power demand forecasting is a crucial and challenging task for new power system and integrated energy system. However, as public feature databases and the theoretical mechanism of power demand changes are unavailable, the known features of power demand fluctuation are much limited. Recently, multimodal learning approaches have shown great vitality in machine learning and AIGC. In this paper, we interact two modal data and propose a textual-knowledge-guided numerical feature discovery (TKNFD) method for short-term power demand forecasting. TKNFD extensively accumulates qualitative textual knowledge, expands it into a candidate feature-type set, collects numerical data of these features, and eventually builds four-dimensional multivariate source-tracking databases (4DM-STDs). Next, TKNFD presents a two-level quantitative feature identification strategy independent of forecasting models, finds 43-48 features, and systematically analyses feature contribution and dependency correlation. Benchmark experiments in two different regions around the world demonstrate that the forecasting accuracy of TKNFD-discovered features reliably outperforms that of SoTA feature schemes by 16.84% to 36.36% MAPE. In particular, TKNFD reveals many unknown features, especially several dominant features in the unknown energy and astronomical dimensions, which extend the knowledge on the origin of strong randomness and non-linearity in power demand fluctuation. Besides, 4DM-STDs can serve as public baseline databases.

{{</citation>}}


### (81/156) HOT: Higher-Order Dynamic Graph Representation Learning with Efficient Transformers (Maciej Besta et al., 2023)

{{<citation>}}

Maciej Besta, Afonso Claudino Catarino, Lukas Gianinazzi, Nils Blach, Piotr Nyczyk, Hubert Niewiadomski, Torsten Hoefler. (2023)  
**HOT: Higher-Order Dynamic Graph Representation Learning with Efficient Transformers**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Representation Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.18526v1)  

---


**ABSTRACT**  
Many graph representation learning (GRL) problems are dynamic, with millions of edges added or removed per second. A fundamental workload in this setting is dynamic link prediction: using a history of graph updates to predict whether a given pair of vertices will become connected. Recent schemes for link prediction in such dynamic settings employ Transformers, modeling individual graph updates as single tokens. In this work, we propose HOT: a model that enhances this line of works by harnessing higher-order (HO) graph structures; specifically, k-hop neighbors and more general subgraphs containing a given pair of vertices. Harnessing such HO structures by encoding them into the attention matrix of the underlying Transformer results in higher accuracy of link prediction outcomes, but at the expense of increased memory pressure. To alleviate this, we resort to a recent class of schemes that impose hierarchy on the attention matrix, significantly reducing memory footprint. The final design offers a sweetspot between high accuracy and low memory utilization. HOT outperforms other dynamic GRL schemes, for example achieving 9%, 7%, and 15% higher accuracy than - respectively - DyGFormer, TGN, and GraphMixer, for the MOOC dataset. Our design can be seamlessly extended towards other dynamic GRL workloads.

{{</citation>}}


### (82/156) Improving Adversarial Transferability via Model Alignment (Avery Ma et al., 2023)

{{<citation>}}

Avery Ma, Amir-massoud Farahmand, Yangchen Pan, Philip Torr, Jindong Gu. (2023)  
**Improving Adversarial Transferability via Model Alignment**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.18495v1)  

---


**ABSTRACT**  
Neural networks are susceptible to adversarial perturbations that are transferable across different models. In this paper, we introduce a novel model alignment technique aimed at improving a given source model's ability in generating transferable adversarial perturbations. During the alignment process, the parameters of the source model are fine-tuned to minimize an alignment loss. This loss measures the divergence in the predictions between the source model and another, independently trained model, referred to as the witness model. To understand the effect of model alignment, we conduct a geometric anlaysis of the resulting changes in the loss landscape. Extensive experiments on the ImageNet dataset, using a variety of model architectures, demonstrate that perturbations generated from aligned source models exhibit significantly higher transferability than those from the original source model.

{{</citation>}}


### (83/156) How Much Is Hidden in the NAS Benchmarks? Few-Shot Adaptation of a NAS Predictor (Hrushikesh Loya et al., 2023)

{{<citation>}}

Hrushikesh Loya, Łukasz Dudziak, Abhinav Mehrotra, Royson Lee, Javier Fernandez-Marques, Nicholas D. Lane, Hongkai Wen. (2023)  
**How Much Is Hidden in the NAS Benchmarks? Few-Shot Adaptation of a NAS Predictor**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2311.18451v1)  

---


**ABSTRACT**  
Neural architecture search has proven to be a powerful approach to designing and refining neural networks, often boosting their performance and efficiency over manually-designed variations, but comes with computational overhead. While there has been a considerable amount of research focused on lowering the cost of NAS for mainstream tasks, such as image classification, a lot of those improvements stem from the fact that those tasks are well-studied in the broader context. Consequently, applicability of NAS to emerging and under-represented domains is still associated with a relatively high cost and/or uncertainty about the achievable gains. To address this issue, we turn our focus towards the recent growth of publicly available NAS benchmarks in an attempt to extract general NAS knowledge, transferable across different tasks and search spaces. We borrow from the rich field of meta-learning for few-shot adaptation and carefully study applicability of those methods to NAS, with a special focus on the relationship between task-level correlation (domain shift) and predictor transferability; which we deem critical for improving NAS on diverse tasks. In our experiments, we use 6 NAS benchmarks in conjunction, spanning in total 16 NAS settings -- our meta-learning approach not only shows superior (or matching) performance in the cross-validation experiments but also successful extrapolation to a new search space and tasks.

{{</citation>}}


### (84/156) Exploring the Temperature-Dependent Phase Transition in Modern Hopfield Networks (Felix Koulischer et al., 2023)

{{<citation>}}

Felix Koulischer, Cédric Goemaere, Tom van der Meersch, Johannes Deleu, Thomas Demeester. (2023)  
**Exploring the Temperature-Dependent Phase Transition in Modern Hopfield Networks**  

---
Primary Category: cs.LG  
Categories: cond-mat-dis-nn, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.18434v1)  

---


**ABSTRACT**  
The recent discovery of a connection between Transformers and Modern Hopfield Networks (MHNs) has reignited the study of neural networks from a physical energy-based perspective. This paper focuses on the pivotal effect of the inverse temperature hyperparameter $\beta$ on the distribution of energy minima of the MHN. To achieve this, the distribution of energy minima is tracked in a simplified MHN in which equidistant normalised patterns are stored. This network demonstrates a phase transition at a critical temperature $\beta_{\text{c}}$, from a single global attractor towards highly pattern specific minima as $\beta$ is increased. Importantly, the dynamics are not solely governed by the hyperparameter $\beta$ but are instead determined by an effective inverse temperature $\beta_{\text{eff}}$ which also depends on the distribution and size of the stored patterns. Recognizing the role of hyperparameters in the MHN could, in the future, aid researchers in the domain of Transformers to optimise their initial choices, potentially reducing the necessity for time and energy expensive hyperparameter fine-tuning.

{{</citation>}}


### (85/156) Data-efficient Deep Reinforcement Learning for Vehicle Trajectory Control (Bernd Frauenknecht et al., 2023)

{{<citation>}}

Bernd Frauenknecht, Tobias Ehlgen, Sebastian Trimpe. (2023)  
**Data-efficient Deep Reinforcement Learning for Vehicle Trajectory Control**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.18393v1)  

---


**ABSTRACT**  
Advanced vehicle control is a fundamental building block in the development of autonomous driving systems. Reinforcement learning (RL) promises to achieve control performance superior to classical approaches while keeping computational demands low during deployment. However, standard RL approaches like soft-actor critic (SAC) require extensive amounts of training data to be collected and are thus impractical for real-world application. To address this issue, we apply recently developed data-efficient deep RL methods to vehicle trajectory control. Our investigation focuses on three methods, so far unexplored for vehicle control: randomized ensemble double Q-learning (REDQ), probabilistic ensembles with trajectory sampling and model predictive path integral optimizer (PETS-MPPI), and model-based policy optimization (MBPO). We find that in the case of trajectory control, the standard model-based RL formulation used in approaches like PETS-MPPI and MBPO is not suitable. We, therefore, propose a new formulation that splits dynamics prediction and vehicle localization. Our benchmark study on the CARLA simulator reveals that the three identified data-efficient deep RL approaches learn control strategies on a par with or better than SAC, yet reduce the required number of environment interactions by more than one order of magnitude.

{{</citation>}}


### (86/156) Towards Comparable Active Learning (Thorben Werner et al., 2023)

{{<citation>}}

Thorben Werner, Johannes Burchert, Lars Schmidt-Thieme. (2023)  
**Towards Comparable Active Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2311.18356v1)  

---


**ABSTRACT**  
Active Learning has received significant attention in the field of machine learning for its potential in selecting the most informative samples for labeling, thereby reducing data annotation costs. However, we show that the reported lifts in recent literature generalize poorly to other domains leading to an inconclusive landscape in Active Learning research. Furthermore, we highlight overlooked problems for reproducing AL experiments that can lead to unfair comparisons and increased variance in the results. This paper addresses these issues by providing an Active Learning framework for a fair comparison of algorithms across different tasks and domains, as well as a fast and performant oracle algorithm for evaluation. To the best of our knowledge, we propose the first AL benchmark that tests algorithms in 3 major domains: Tabular, Image, and Text. We report empirical results for 6 widely used algorithms on 7 real-world and 2 synthetic datasets and aggregate them into a domain-specific ranking of AL algorithms.

{{</citation>}}


### (87/156) Anomaly Detection via Learning-Based Sequential Controlled Sensing (Geethu Joseph et al., 2023)

{{<citation>}}

Geethu Joseph, Chen Zhong, M. Cenk Gursoy, Senem Velipasalar, Pramod K. Varshney. (2023)  
**Anomaly Detection via Learning-Based Sequential Controlled Sensing**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SP, eess-SY  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.00088v1)  

---


**ABSTRACT**  
In this paper, we address the problem of detecting anomalies among a given set of binary processes via learning-based controlled sensing. Each process is parameterized by a binary random variable indicating whether the process is anomalous. To identify the anomalies, the decision-making agent is allowed to observe a subset of the processes at each time instant. Also, probing each process has an associated cost. Our objective is to design a sequential selection policy that dynamically determines which processes to observe at each time with the goal to minimize the delay in making the decision and the total sensing cost. We cast this problem as a sequential hypothesis testing problem within the framework of Markov decision processes. This formulation utilizes both a Bayesian log-likelihood ratio-based reward and an entropy-based reward. The problem is then solved using two approaches: 1) a deep reinforcement learning-based approach where we design both deep Q-learning and policy gradient actor-critic algorithms; and 2) a deep active inference-based approach. Using numerical experiments, we demonstrate the efficacy of our algorithms and show that our algorithms adapt to any unknown statistical dependence pattern of the processes.

{{</citation>}}


### (88/156) Categorical Traffic Transformer: Interpretable and Diverse Behavior Prediction with Tokenized Latent (Yuxiao Chen et al., 2023)

{{<citation>}}

Yuxiao Chen, Sander Tonkens, Marco Pavone. (2023)  
**Categorical Traffic Transformer: Interpretable and Diverse Behavior Prediction with Tokenized Latent**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs-RO, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.18307v1)  

---


**ABSTRACT**  
Adept traffic models are critical to both planning and closed-loop simulation for autonomous vehicles (AV), and key design objectives include accuracy, diverse multimodal behaviors, interpretability, and downstream compatibility. Recently, with the advent of large language models (LLMs), an additional desirable feature for traffic models is LLM compatibility. We present Categorical Traffic Transformer (CTT), a traffic model that outputs both continuous trajectory predictions and tokenized categorical predictions (lane modes, homotopies, etc.). The most outstanding feature of CTT is its fully interpretable latent space, which enables direct supervision of the latent variable from the ground truth during training and avoids mode collapse completely. As a result, CTT can generate diverse behaviors conditioned on different latent modes with semantic meanings while beating SOTA on prediction accuracy. In addition, CTT's ability to input and output tokens enables integration with LLMs for common-sense reasoning and zero-shot generalization.

{{</citation>}}


### (89/156) Large Language Models for Travel Behavior Prediction (Baichuan Mo et al., 2023)

{{<citation>}}

Baichuan Mo, Hanyong Xu, Dingyi Zhuang, Ruoyun Ma, Xiaotong Guo, Jinhua Zhao. (2023)  
**Large Language Models for Travel Behavior Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.00819v1)  

---


**ABSTRACT**  
Travel behavior prediction is a fundamental task in transportation demand management. The conventional methods for travel behavior prediction rely on numerical data to construct mathematical models and calibrate model parameters to represent human preferences. Recent advancement in large language models (LLMs) has shown great reasoning abilities to solve complex problems. In this study, we propose to use LLMs to predict travel behavior with prompt engineering without data-based parameter learning. Specifically, we carefully design our prompts that include 1) task description, 2) travel characteristics, 3) individual attributes, and 4) guides of thinking with domain knowledge, and ask the LLMs to predict an individual's travel behavior and explain the results. We select the travel mode choice task as a case study. Results show that, though no training samples are provided, LLM-based predictions have competitive accuracy and F1-score as canonical supervised learning methods such as multinomial logit, random forest, and neural networks. LLMs can also output reasons that support their prediction. However, though in most of the cases, the output explanations are reasonable, we still observe cases that violate logic or with hallucinations.

{{</citation>}}


### (90/156) SMaRt: Improving GANs with Score Matching Regularity (Mengfei Xia et al., 2023)

{{<citation>}}

Mengfei Xia, Yujun Shen, Ceyuan Yang, Ran Yi, Wenping Wang, Yong-jin Liu. (2023)  
**SMaRt: Improving GANs with Score Matching Regularity**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.18208v1)  

---


**ABSTRACT**  
Generative adversarial networks (GANs) usually struggle in learning from highly diverse data, whose underlying manifold is complex. In this work, we revisit the mathematical foundations of GANs, and theoretically reveal that the native adversarial loss for GAN training is insufficient to fix the problem of subsets with positive Lebesgue measure of the generated data manifold lying out of the real data manifold. Instead, we find that score matching serves as a valid solution to this issue thanks to its capability of persistently pushing the generated data points towards the real data manifold. We thereby propose to improve the optimization of GANs with score matching regularity (SMaRt). Regarding the empirical evidences, we first design a toy example to show that training GANs by the aid of a ground-truth score function can help reproduce the real data distribution more accurately, and then confirm that our approach can consistently boost the synthesis performance of various state-of-the-art GANs on real-world datasets with pre-trained diffusion models acting as the approximate score function. For instance, when training Aurora on the ImageNet 64x64 dataset, we manage to improve FID from 8.87 to 7.11, on par with the performance of one-step consistency model. The source code will be made public.

{{</citation>}}


### (91/156) SCOPE-RL: A Python Library for Offline Reinforcement Learning and Off-Policy Evaluation (Haruka Kiyohara et al., 2023)

{{<citation>}}

Haruka Kiyohara, Ren Kishimoto, Kosuke Kawakami, Ken Kobayashi, Kazuhide Nakata, Yuta Saito. (2023)  
**SCOPE-RL: A Python Library for Offline Reinforcement Learning and Off-Policy Evaluation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.18206v2)  

---


**ABSTRACT**  
This paper introduces SCOPE-RL, a comprehensive open-source Python software designed for offline reinforcement learning (offline RL), off-policy evaluation (OPE), and selection (OPS). Unlike most existing libraries that focus solely on either policy learning or evaluation, SCOPE-RL seamlessly integrates these two key aspects, facilitating flexible and complete implementations of both offline RL and OPE processes. SCOPE-RL put particular emphasis on its OPE modules, offering a range of OPE estimators and robust evaluation-of-OPE protocols. This approach enables more in-depth and reliable OPE compared to other packages. For instance, SCOPE-RL enhances OPE by estimating the entire reward distribution under a policy rather than its mere point-wise expected value. Additionally, SCOPE-RL provides a more thorough evaluation-of-OPE by presenting the risk-return tradeoff in OPE results, extending beyond mere accuracy evaluations in existing OPE literature. SCOPE-RL is designed with user accessibility in mind. Its user-friendly APIs, comprehensive documentation, and a variety of easy-to-follow examples assist researchers and practitioners in efficiently implementing and experimenting with various offline RL methods and OPE estimators, tailored to their specific problem contexts. The documentation of SCOPE-RL is available at https://scope-rl.readthedocs.io/en/latest/.

{{</citation>}}


### (92/156) An Effective Universal Polynomial Basis for Spectral Graph Neural Networks (Keke Huang et al., 2023)

{{<citation>}}

Keke Huang, Pietro Liò. (2023)  
**An Effective Universal Polynomial Basis for Spectral Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG, eess-SP  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.18177v1)  

---


**ABSTRACT**  
Spectral Graph Neural Networks (GNNs), also referred to as graph filters have gained increasing prevalence for heterophily graphs. Optimal graph filters rely on Laplacian eigendecomposition for Fourier transform. In an attempt to avert the prohibitive computations, numerous polynomial filters by leveraging distinct polynomials have been proposed to approximate the desired graph filters. However, polynomials in the majority of polynomial filters are predefined and remain fixed across all graphs, failing to accommodate the diverse heterophily degrees across different graphs. To tackle this issue, we first investigate the correlation between polynomial bases of desired graph filters and the degrees of graph heterophily via a thorough theoretical analysis. Afterward, we develop an adaptive heterophily basis by incorporating graph heterophily degrees. Subsequently, we integrate this heterophily basis with the homophily basis, creating a universal polynomial basis UniBasis. In consequence, we devise a general polynomial filter UniFilter. Comprehensive experiments on both real-world and synthetic datasets with varying heterophily degrees significantly support the superiority of UniFilter, demonstrating the effectiveness and generality of UniBasis, as well as its promising capability as a new method for graph analysis.

{{</citation>}}


## cs.CL (20)



### (93/156) Relevance-guided Neural Machine Translation (Isidora Chara Tourni et al., 2023)

{{<citation>}}

Isidora Chara Tourni, Derry Wijaya. (2023)  
**Relevance-guided Neural Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation, Transformer  
[Paper Link](http://arxiv.org/abs/2312.00214v1)  

---


**ABSTRACT**  
With the advent of the Transformer architecture, Neural Machine Translation (NMT) results have shown great improvement lately. However, results in low-resource conditions still lag behind in both bilingual and multilingual setups, due to the limited amount of available monolingual and/or parallel data; hence, the need for methods addressing data scarcity in an efficient, and explainable way, is eminent. We propose an explainability-based training approach for NMT, applied in Unsupervised and Supervised model training, for translation of three languages of varying resources, French, Gujarati, Kazakh, to and from English. Our results show our method can be promising, particularly when training in low-resource conditions, outperforming simple training baselines; though the improvement is marginal, it sets the ground for further exploration of the approach and the parameters, and its extension to other languages.

{{</citation>}}


### (94/156) Navigating News Narratives: A Media Bias Analysis Dataset (Shaina Raza, 2023)

{{<citation>}}

Shaina Raza. (2023)  
**Navigating News Narratives: A Media Bias Analysis Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.00168v1)  

---


**ABSTRACT**  
The proliferation of biased news narratives across various media platforms has become a prominent challenge, influencing public opinion on critical topics like politics, health, and climate change. This paper introduces the "Navigating News Narratives: A Media Bias Analysis Dataset", a comprehensive dataset to address the urgent need for tools to detect and analyze media bias. This dataset encompasses a broad spectrum of biases, making it a unique and valuable asset in the field of media studies and artificial intelligence. The dataset is available at https://figshare.com/articles/dataset/news-media-bias_data_json/24422122

{{</citation>}}


### (95/156) What Do Llamas Really Think? Revealing Preference Biases in Language Model Representations (Raphael Tang et al., 2023)

{{<citation>}}

Raphael Tang, Xinyu Zhang, Jimmy Lin, Ferhan Ture. (2023)  
**What Do Llamas Really Think? Revealing Preference Biases in Language Model Representations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2311.18812v1)  

---


**ABSTRACT**  
Do large language models (LLMs) exhibit sociodemographic biases, even when they decline to respond? To bypass their refusal to "speak," we study this research question by probing contextualized embeddings and exploring whether this bias is encoded in its latent representations. We propose a logistic Bradley-Terry probe which predicts word pair preferences of LLMs from the words' hidden vectors. We first validate our probe on three pair preference tasks and thirteen LLMs, where we outperform the word embedding association test (WEAT), a standard approach in testing for implicit association, by a relative 27% in error rate. We also find that word pair preferences are best represented in the middle layers. Next, we transfer probes trained on harmless tasks (e.g., pick the larger number) to controversial ones (compare ethnicities) to examine biases in nationality, politics, religion, and gender. We observe substantial bias for all target classes: for instance, the Mistral model implicitly prefers Europe to Africa, Christianity to Judaism, and left-wing to right-wing politics, despite declining to answer. This suggests that instruction fine-tuning does not necessarily debias contextualized embeddings. Our codebase is at https://github.com/castorini/biasprobe.

{{</citation>}}


### (96/156) Unnatural Error Correction: GPT-4 Can Almost Perfectly Handle Unnatural Scrambled Text (Qi Cao et al., 2023)

{{<citation>}}

Qi Cao, Takeshi Kojima, Yutaka Matsuo, Yusuke Iwasawa. (2023)  
**Unnatural Error Correction: GPT-4 Can Almost Perfectly Handle Unnatural Scrambled Text**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.18805v1)  

---


**ABSTRACT**  
While Large Language Models (LLMs) have achieved remarkable performance in many tasks, much about their inner workings remains unclear. In this study, we present novel experimental insights into the resilience of LLMs, particularly GPT-4, when subjected to extensive character-level permutations. To investigate this, we first propose the Scrambled Bench, a suite designed to measure the capacity of LLMs to handle scrambled input, in terms of both recovering scrambled sentences and answering questions given scrambled context. The experimental results indicate that most powerful LLMs demonstrate the capability akin to typoglycemia, a phenomenon where humans can understand the meaning of words even when the letters within those words are scrambled, as long as the first and last letters remain in place. More surprisingly, we found that only GPT-4 nearly flawlessly processes inputs with unnatural errors, even under the extreme condition, a task that poses significant challenges for other LLMs and often even for humans. Specifically, GPT-4 can almost perfectly reconstruct the original sentences from scrambled ones, decreasing the edit distance by 95%, even when all letters within each word are entirely scrambled. It is counter-intuitive that LLMs can exhibit such resilience despite severe disruption to input tokenization caused by scrambled text.

{{</citation>}}


### (97/156) Mavericks at BLP-2023 Task 1: Ensemble-based Approach Using Language Models for Violence Inciting Text Detection (Saurabh Page et al., 2023)

{{<citation>}}

Saurabh Page, Sudeep Mangalvedhekar, Kshitij Deshpande, Tanmay Chavan, Sheetal Sonawane. (2023)  
**Mavericks at BLP-2023 Task 1: Ensemble-based Approach Using Language Models for Violence Inciting Text Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.18778v1)  

---


**ABSTRACT**  
This paper presents our work for the Violence Inciting Text Detection shared task in the First Workshop on Bangla Language Processing. Social media has accelerated the propagation of hate and violence-inciting speech in society. It is essential to develop efficient mechanisms to detect and curb the propagation of such texts. The problem of detecting violence-inciting texts is further exacerbated in low-resource settings due to sparse research and less data. The data provided in the shared task consists of texts in the Bangla language, where each example is classified into one of the three categories defined based on the types of violence-inciting texts. We try and evaluate several BERT-based models, and then use an ensemble of the models as our final submission. Our submission is ranked 10th in the final leaderboard of the shared task with a macro F1 score of 0.737.

{{</citation>}}


### (98/156) TaskBench: Benchmarking Large Language Models for Task Automation (Yongliang Shen et al., 2023)

{{<citation>}}

Yongliang Shen, Kaitao Song, Xu Tan, Wenqi Zhang, Kan Ren, Siyu Yuan, Weiming Lu, Dongsheng Li, Yueting Zhuang. (2023)  
**TaskBench: Benchmarking Large Language Models for Task Automation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.18760v1)  

---


**ABSTRACT**  
Recently, the incredible progress of large language models (LLMs) has ignited the spark of task automation, which decomposes the complex tasks described by user instructions into sub-tasks, and invokes external tools to execute them, and plays a central role in autonomous agents. However, there lacks a systematic and standardized benchmark to foster the development of LLMs in task automation. To this end, we introduce TaskBench to evaluate the capability of LLMs in task automation. Specifically, task automation can be formulated into three critical stages: task decomposition, tool invocation, and parameter prediction to fulfill user intent. This complexity makes data collection and evaluation more challenging compared to common NLP tasks. To generate high-quality evaluation datasets, we introduce the concept of Tool Graph to represent the decomposed tasks in user intent, and adopt a back-instruct method to simulate user instruction and annotations. Furthermore, we propose TaskEval to evaluate the capability of LLMs from different aspects, including task decomposition, tool invocation, and parameter prediction. Experimental results demonstrate that TaskBench can effectively reflects the capability of LLMs in task automation. Benefiting from the mixture of automated data construction and human verification, TaskBench achieves a high consistency compared to the human evaluation, which can be utilized as a comprehensive and faithful benchmark for LLM-based autonomous agents.

{{</citation>}}


### (99/156) AlignBench: Benchmarking Chinese Alignment of Large Language Models (Xiao Liu et al., 2023)

{{<citation>}}

Xiao Liu, Xuanyu Lei, Shengyuan Wang, Yue Huang, Zhuoer Feng, Bosi Wen, Jiale Cheng, Pei Ke, Yifan Xu, Weng Lam Tam, Xiaohan Zhang, Lichao Sun, Hongning Wang, Jing Zhang, Minlie Huang, Yuxiao Dong, Jie Tang. (2023)  
**AlignBench: Benchmarking Chinese Alignment of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.18743v3)  

---


**ABSTRACT**  
Alignment has become a critical step for instruction-tuned Large Language Models (LLMs) to become helpful assistants. However, effective evaluation of alignment for emerging Chinese LLMs is still significantly lacking, calling for real-scenario grounded, open-ended, challenging and automatic evaluations tailored for alignment. To fill in this gap, we introduce AlignBench, a comprehensive multi-dimensional benchmark for evaluating LLMs' alignment in Chinese. Equipped with a human-in-the-loop data curation pipeline, our benchmark employs a rule-calibrated multi-dimensional LLM-as-Judge with Chain-of-Thought to generate explanations and final ratings as evaluations, ensuring high reliability and interpretability. Furthermore, we report AlignBench evaluated by CritiqueLLM, a dedicated Chinese evaluator LLM that recovers 95% of GPT-4's evaluation ability. We will provide public APIs for evaluating AlignBench with CritiqueLLM to facilitate the evaluation of LLMs' Chinese alignment. All evaluation codes, data, and LLM generations are available at \url{https://github.com/THUDM/AlignBench}.

{{</citation>}}


### (100/156) Mavericks at NADI 2023 Shared Task: Unravelling Regional Nuances through Dialect Identification using Transformer-based Approach (Vedant Deshpande et al., 2023)

{{<citation>}}

Vedant Deshpande, Yash Patwardhan, Kshitij Deshpande, Sudeep Mangalvedhekar, Ravindra Murumkar. (2023)  
**Mavericks at NADI 2023 Shared Task: Unravelling Regional Nuances through Dialect Identification using Transformer-based Approach**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Transformer, Twitter  
[Paper Link](http://arxiv.org/abs/2311.18739v1)  

---


**ABSTRACT**  
In this paper, we present our approach for the "Nuanced Arabic Dialect Identification (NADI) Shared Task 2023". We highlight our methodology for subtask 1 which deals with country-level dialect identification. Recognizing dialects plays an instrumental role in enhancing the performance of various downstream NLP tasks such as speech recognition and translation. The task uses the Twitter dataset (TWT-2023) that encompasses 18 dialects for the multi-class classification problem. Numerous transformer-based models, pre-trained on Arabic language, are employed for identifying country-level dialects. We fine-tune these state-of-the-art models on the provided dataset. The ensembling method is leveraged to yield improved performance of the system. We achieved an F1-score of 76.65 (11th rank on the leaderboard) on the test dataset.

{{</citation>}}


### (101/156) Mavericks at ArAIEval Shared Task: Towards a Safer Digital Space -- Transformer Ensemble Models Tackling Deception and Persuasion (Sudeep Mangalvedhekar et al., 2023)

{{<citation>}}

Sudeep Mangalvedhekar, Kshitij Deshpande, Yash Patwardhan, Vedant Deshpande, Ravindra Murumkar. (2023)  
**Mavericks at ArAIEval Shared Task: Towards a Safer Digital Space -- Transformer Ensemble Models Tackling Deception and Persuasion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2311.18730v1)  

---


**ABSTRACT**  
In this paper, we highlight our approach for the "Arabic AI Tasks Evaluation (ArAiEval) Shared Task 2023". We present our approaches for task 1-A and task 2-A of the shared task which focus on persuasion technique detection and disinformation detection respectively. Detection of persuasion techniques and disinformation has become imperative to avoid distortion of authentic information. The tasks use multigenre snippets of tweets and news articles for the given binary classification problem. We experiment with several transformer-based models that were pre-trained on the Arabic language. We fine-tune these state-of-the-art models on the provided dataset. Ensembling is employed to enhance the performance of the systems. We achieved a micro F1-score of 0.742 on task 1-A (8th rank on the leaderboard) and 0.901 on task 2-A (7th rank on the leaderboard) respectively.

{{</citation>}}


### (102/156) Women Are Beautiful, Men Are Leaders: Gender Stereotypes in Machine Translation and Language Modeling (Matúš Pikuliak et al., 2023)

{{<citation>}}

Matúš Pikuliak, Andrea Hrckova, Stefan Oresko, Marián Šimko. (2023)  
**Women Are Beautiful, Men Are Leaders: Gender Stereotypes in Machine Translation and Language Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.18711v1)  

---


**ABSTRACT**  
We present GEST -- a new dataset for measuring gender-stereotypical reasoning in masked LMs and English-to-X machine translation systems. GEST contains samples that are compatible with 9 Slavic languages and English for 16 gender stereotypes about men and women (e.g., Women are beautiful, Men are leaders). The definition of said stereotypes was informed by gender experts. We used GEST to evaluate 11 masked LMs and 4 machine translation systems. We discovered significant and consistent amounts of stereotypical reasoning in almost all the evaluated models and languages.

{{</citation>}}


### (103/156) CritiqueLLM: Scaling LLM-as-Critic for Effective and Explainable Evaluation of Large Language Model Generation (Pei Ke et al., 2023)

{{<citation>}}

Pei Ke, Bosi Wen, Zhuoer Feng, Xiao Liu, Xuanyu Lei, Jiale Cheng, Shengyuan Wang, Aohan Zeng, Yuxiao Dong, Hongning Wang, Jie Tang, Minlie Huang. (2023)  
**CritiqueLLM: Scaling LLM-as-Critic for Effective and Explainable Evaluation of Large Language Model Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.18702v1)  

---


**ABSTRACT**  
Since the natural language processing (NLP) community started to make large language models (LLMs), such as GPT-4, act as a critic to evaluate the quality of generated texts, most of them only train a critique generation model of a specific scale on specific datasets. We argue that a comprehensive investigation on the key factor of LLM-based evaluation models, such as scaling properties, is lacking, so that it is still inconclusive whether these models have potential to replace GPT-4's evaluation in practical scenarios. In this paper, we propose a new critique generation model called CritiqueLLM, which includes a dialogue-based prompting method for high-quality referenced / reference-free evaluation data. Experimental results show that our model can achieve comparable evaluation performance to GPT-4 especially in system-level correlations, and even outperform GPT-4 in 3 out of 8 tasks in a challenging reference-free setting. We conduct detailed analysis to show promising scaling properties of our model in the quality of generated critiques. We also demonstrate that our generated critiques can act as scalable feedback to directly improve the generation quality of LLMs.

{{</citation>}}


### (104/156) ArcMMLU: A Library and Information Science Benchmark for Large Language Models (Shitou Zhang et al., 2023)

{{<citation>}}

Shitou Zhang, Zuchao Li, Xingshen Liu, Liming Yang, Ping Wang. (2023)  
**ArcMMLU: A Library and Information Science Benchmark for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.18658v1)  

---


**ABSTRACT**  
In light of the rapidly evolving capabilities of large language models (LLMs), it becomes imperative to develop rigorous domain-specific evaluation benchmarks to accurately assess their capabilities. In response to this need, this paper introduces ArcMMLU, a specialized benchmark tailored for the Library & Information Science (LIS) domain in Chinese. This benchmark aims to measure the knowledge and reasoning capability of LLMs within four key sub-domains: Archival Science, Data Science, Library Science, and Information Science. Following the format of MMLU/CMMLU, we collected over 6,000 high-quality questions for the compilation of ArcMMLU. This extensive compilation can reflect the diverse nature of the LIS domain and offer a robust foundation for LLM evaluation. Our comprehensive evaluation reveals that while most mainstream LLMs achieve an average accuracy rate above 50% on ArcMMLU, there remains a notable performance gap, suggesting substantial headroom for refinement in LLM capabilities within the LIS domain. Further analysis explores the effectiveness of few-shot examples on model performance and highlights challenging questions where models consistently underperform, providing valuable insights for targeted improvements. ArcMMLU fills a critical gap in LLM evaluations within the Chinese LIS domain and paves the way for future development of LLMs tailored to this specialized area.

{{</citation>}}


### (105/156) ArthModel: Enhance Arithmetic Skills to Large Language Model (Yingdi Guo, 2023)

{{<citation>}}

Yingdi Guo. (2023)  
**ArthModel: Enhance Arithmetic Skills to Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.18609v1)  

---


**ABSTRACT**  
With the great success of ChatGPT, the research of large language models has become increasingly popular. However, the models have several limitations, such as toxicity and pool performance of arithmetic solving. Meanwhile, LLM may have some potential abilities that have yet to be exploited. In this paper, we choose a different way to enhance the arithmetic ability of LLM. We propose to train LLM to generate a postfix expression related to the arithmetic problem and incorporate it with small pretrained models. Moreover, this small model transfers the token embeddings into real dense numbers and invokes native functions of a deep learning platform to get the correct answer. To generate the final result, we propose prompt injection for adding the result outputs by the small model to LLM. This work provides different ways of thinking, training and using a language model. The codes and models will be released at \url{https://github.com/eteced/arithmetic_finetuning_v1}.

{{</citation>}}


### (106/156) FFT: Towards Harmlessness Evaluation and Analysis for LLMs with Factuality, Fairness, Toxicity (Shiyao Cui et al., 2023)

{{<citation>}}

Shiyao Cui, Zhenyu Zhang, Yilong Chen, Wenyuan Zhang, Tianyun Liu, Siqi Wang, Tingwen Liu. (2023)  
**FFT: Towards Harmlessness Evaluation and Analysis for LLMs with Factuality, Fairness, Toxicity**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18580v1)  

---


**ABSTRACT**  
The widespread of generative artificial intelligence has heightened concerns about the potential harms posed by AI-generated texts, primarily stemming from factoid, unfair, and toxic content. Previous researchers have invested much effort in assessing the harmlessness of generative language models. However, existing benchmarks are struggling in the era of large language models (LLMs), due to the stronger language generation and instruction following capabilities, as well as wider applications. In this paper, we propose FFT, a new benchmark with 2116 elaborated-designed instances, for LLM harmlessness evaluation with factuality, fairness, and toxicity. To investigate the potential harms of LLMs, we evaluate 9 representative LLMs covering various parameter scales, training stages, and creators. Experiments show that the harmlessness of LLMs is still under-satisfactory, and extensive analysis derives some insightful findings that could inspire future research for harmless LLM research.

{{</citation>}}


### (107/156) ESG Accountability Made Easy: DocQA at Your Service (Lokesh Mishra et al., 2023)

{{<citation>}}

Lokesh Mishra, Cesar Berrospi, Kasper Dinkla, Diego Antognini, Francesco Fusco, Benedikt Bothur, Maksym Lysak, Nikolaos Livathinos, Ahmed Nassar, Panagiotis Vagenas, Lucas Morin, Christoph Auer, Michele Dolfi, Peter Staar. (2023)  
**ESG Accountability Made Easy: DocQA at Your Service**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2311.18481v1)  

---


**ABSTRACT**  
We present Deep Search DocQA. This application enables information extraction from documents via a question-answering conversational assistant. The system integrates several technologies from different AI disciplines consisting of document conversion to machine-readable format (via computer vision), finding relevant data (via natural language processing), and formulating an eloquent response (via large language models). Users can explore over 10,000 Environmental, Social, and Governance (ESG) disclosure reports from over 2000 corporations. The Deep Search platform can be accessed at: https://ds4sd.github.io.

{{</citation>}}


### (108/156) IAG: Induction-Augmented Generation Framework for Answering Reasoning Questions (Zhebin Zhang et al., 2023)

{{<citation>}}

Zhebin Zhang, Xinyu Zhang, Yuanhang Ren, Saijiang Shi, Meng Han, Yongkang Wu, Ruofei Lai, Zhao Cao. (2023)  
**IAG: Induction-Augmented Generation Framework for Answering Reasoning Questions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.18397v1)  

---


**ABSTRACT**  
Retrieval-Augmented Generation (RAG), by incorporating external knowledge with parametric memory of language models, has become the state-of-the-art architecture for open-domain QA tasks. However, common knowledge bases are inherently constrained by limited coverage and noisy information, making retrieval-based approaches inadequate to answer implicit reasoning questions. In this paper, we propose an Induction-Augmented Generation (IAG) framework that utilizes inductive knowledge along with the retrieved documents for implicit reasoning. We leverage large language models (LLMs) for deriving such knowledge via a novel prompting method based on inductive reasoning patterns. On top of this, we implement two versions of IAG named IAG-GPT and IAG-Student, respectively. IAG-GPT directly utilizes the knowledge generated by GPT-3 for answer prediction, while IAG-Student gets rid of dependencies on GPT service at inference time by incorporating a student inductor model. The inductor is firstly trained via knowledge distillation and further optimized by back-propagating the generator feedback via differentiable beam scores. Experimental results show that IAG outperforms RAG baselines as well as ChatGPT on two Open-Domain QA tasks. Notably, our best models have won the first place in the official leaderboards of CSQA2.0 (since Nov 1, 2022) and StrategyQA (since Jan 8, 2023).

{{</citation>}}


### (109/156) Hubness Reduction Improves Sentence-BERT Semantic Spaces (Beatrix M. G. Nielsen et al., 2023)

{{<citation>}}

Beatrix M. G. Nielsen, Lars Kai Hansen. (2023)  
**Hubness Reduction Improves Sentence-BERT Semantic Spaces**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SI, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2311.18364v1)  

---


**ABSTRACT**  
Semantic representations of text, i.e. representations of natural language which capture meaning by geometry, are essential for areas such as information retrieval and document grouping. High-dimensional trained dense vectors have received much attention in recent years as such representations. We investigate the structure of semantic spaces that arise from embeddings made with Sentence-BERT and find that the representations suffer from a well-known problem in high dimensions called hubness. Hubness results in asymmetric neighborhood relations, such that some texts (the hubs) are neighbours of many other texts while most texts (so-called anti-hubs), are neighbours of few or no other texts. We quantify the semantic quality of the embeddings using hubness scores and error rate of a neighbourhood based classifier. We find that when hubness is high, we can reduce error rate and hubness using hubness reduction methods. We identify a combination of two methods as resulting in the best reduction. For example, on one of the tested pretrained models, this combined method can reduce hubness by about 75% and error rate by about 9%. Thus, we argue that mitigating hubness in the embedding space provides better semantic representations of text.

{{</citation>}}


### (110/156) Evaluating the Rationale Understanding of Critical Reasoning in Logical Reading Comprehension (Akira Kawabata et al., 2023)

{{<citation>}}

Akira Kawabata, Saku Sugawara. (2023)  
**Evaluating the Rationale Understanding of Critical Reasoning in Logical Reading Comprehension**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.18353v1)  

---


**ABSTRACT**  
To precisely evaluate a language model's capability for logical reading comprehension, we present a dataset for testing the understanding of the rationale behind critical reasoning. For questions taken from an existing multiplechoice logical reading comprehension dataset, we crowdsource rationale texts that explain why we should select or eliminate answer options, resulting in 3,003 multiple-choice subquestions that are associated with 943 main questions. Experiments on our dataset show that recent large language models (e.g., InstructGPT) struggle to answer the subquestions even if they are able to answer the main questions correctly. We find that the models perform particularly poorly in answering subquestions written for the incorrect options of the main questions, implying that the models have a limited capability for explaining why incorrect alternatives should be eliminated. These results suggest that our dataset encourages further investigation into the critical reasoning ability of language models while focusing on the elimination process of relevant alternatives.

{{</citation>}}


### (111/156) LMRL Gym: Benchmarks for Multi-Turn Reinforcement Learning with Language Models (Marwa Abdulhai et al., 2023)

{{<citation>}}

Marwa Abdulhai, Isadora White, Charlie Snell, Charles Sun, Joey Hong, Yuexiang Zhai, Kelvin Xu, Sergey Levine. (2023)  
**LMRL Gym: Benchmarks for Multi-Turn Reinforcement Learning with Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.18232v1)  

---


**ABSTRACT**  
Large language models (LLMs) provide excellent text-generation capabilities, but standard prompting and generation methods generally do not lead to intentional or goal-directed agents and might necessitate considerable prompt tuning. This becomes particularly apparent in multi-turn conversations: even the best current LLMs rarely ask clarifying questions, engage in explicit information gathering, or take actions now that lead to better decisions after multiple turns. Reinforcement learning has the potential to leverage the powerful modeling capabilities of LLMs, as well as their internal representation of textual interactions, to create capable goal-directed language agents. This can enable intentional and temporally extended interactions, such as with humans, through coordinated persuasion and carefully crafted questions, or in goal-directed play through text games to bring about desired final outcomes. However, enabling this requires the community to develop stable and reliable reinforcement learning algorithms that can effectively train LLMs. Developing such algorithms requires tasks that can gauge progress on algorithm design, provide accessible and reproducible evaluations for multi-turn interactions, and cover a range of task properties and challenges in improving reinforcement learning algorithms. Our paper introduces the LMRL-Gym benchmark for evaluating multi-turn RL for LLMs, together with an open-source research framework containing a basic toolkit for getting started on multi-turn RL with offline value-based and policy-based RL methods. Our benchmark consists of 8 different language tasks, which require multiple rounds of language interaction and cover a range of tasks in open-ended dialogue and text games.

{{</citation>}}


### (112/156) Automatic Construction of a Korean Toxic Instruction Dataset for Ethical Tuning of Large Language Models (Sungjoo Byun et al., 2023)

{{<citation>}}

Sungjoo Byun, Dongjun Jang, Hyemi Jo, Hyopil Shin. (2023)  
**Automatic Construction of a Korean Toxic Instruction Dataset for Ethical Tuning of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.18215v1)  

---


**ABSTRACT**  
Caution: this paper may include material that could be offensive or distressing.   The advent of Large Language Models (LLMs) necessitates the development of training approaches that mitigate the generation of unethical language and aptly manage toxic user queries. Given the challenges related to human labor and the scarcity of data, we present KoTox, comprising 39K unethical instruction-output pairs. This collection of automatically generated toxic instructions refines the training of LLMs and establishes a foundational framework for improving LLMs' ethical awareness and response to various toxic inputs, promoting more secure and responsible interactions in Natural Language Processing (NLP) applications.

{{</citation>}}


## cs.SE (5)



### (113/156) EpiTESTER: Testing Autonomous Vehicles with Epigenetic Algorithm and Attention Mechanism (Chengjie Lu et al., 2023)

{{<citation>}}

Chengjie Lu, Shaukat Ali, Tao Yue. (2023)  
**EpiTESTER: Testing Autonomous Vehicles with Epigenetic Algorithm and Attention Mechanism**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-NE, cs-SE, cs.SE  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.00207v1)  

---


**ABSTRACT**  
Testing autonomous vehicles (AVs) under various environmental scenarios that lead the vehicles to unsafe situations is known to be challenging. Given the infinite possible environmental scenarios, it is essential to find critical scenarios efficiently. To this end, we propose a novel testing method, named EpiTESTER, by taking inspiration from epigenetics, which enables species to adapt to sudden environmental changes. In particular, EpiTESTER adopts gene silencing as its epigenetic mechanism, which regulates gene expression to prevent the expression of a certain gene, and the probability of gene expression is dynamically computed as the environment changes. Given different data modalities (e.g., images, lidar point clouds) in the context of AV, EpiTESTER benefits from a multi-model fusion transformer to extract high-level feature representations from environmental factors and then calculates probabilities based on these features with the attention mechanism. To assess the cost-effectiveness of EpiTESTER, we compare it with a classical genetic algorithm (GA) (i.e., without any epigenetic mechanism implemented) and EpiTESTER with equal probability for each gene. We evaluate EpiTESTER with four initial environments from CARLA, an open-source simulator for autonomous driving research, and an end-to-end AV controller, Interfuser. Our results show that EpiTESTER achieved a promising performance in identifying critical scenarios compared to the baselines, showing that applying epigenetic mechanisms is a good option for solving practical problems.

{{</citation>}}


### (114/156) Developer Experiences with a Contextualized AI Coding Assistant: Usability, Expectations, and Outcomes (Gustavo Pinto et al., 2023)

{{<citation>}}

Gustavo Pinto, Cleidson de Souza, Thayssa Rocha, Igor Steinmacher, Alberto de Souza, Edward Monteiro. (2023)  
**Developer Experiences with a Contextualized AI Coding Assistant: Usability, Expectations, and Outcomes**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18452v1)  

---


**ABSTRACT**  
In the rapidly advancing field of artificial intelligence, software development has emerged as a key area of innovation. Despite the plethora of general-purpose AI assistants available, their effectiveness diminishes in complex, domain-specific scenarios. Noting this limitation, both the academic community and industry players are relying on contextualized coding AI assistants. These assistants surpass general-purpose AI tools by integrating proprietary, domain-specific knowledge, offering precise and relevant solutions. Our study focuses on the initial experiences of 62 participants who used a contextualized coding AI assistant -- named StackSpot AI -- in a controlled setting. According to the participants, the assistants' use resulted in significant time savings, easier access to documentation, and the generation of accurate codes for internal APIs. However, challenges associated with the knowledge sources necessary to make the coding assistant access more contextual information as well as variable responses and limitations in handling complex codes were observed. The study's findings, detailing both the benefits and challenges of contextualized AI assistants, underscore their potential to revolutionize software development practices, while also highlighting areas for further refinement.

{{</citation>}}


### (115/156) Lessons from Building CodeBuddy: A Contextualized AI Coding Assistant (gustavo Pinto et al., 2023)

{{<citation>}}

gustavo Pinto, Cleidson de Souza, João Batista Neto, Alberto de Souza, Tarcísio Gotto, Edward Monteiro. (2023)  
**Lessons from Building CodeBuddy: A Contextualized AI Coding Assistant**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.18450v1)  

---


**ABSTRACT**  
With their exceptional natural language processing capabilities, tools based on Large Language Models (LLMs) like ChatGPT and Co-Pilot have swiftly become indispensable resources in the software developer's toolkit. While recent studies suggest the potential productivity gains these tools can unlock, users still encounter drawbacks, such as generic or incorrect answers. Additionally, the pursuit of improved responses often leads to extensive prompt engineering efforts, diverting valuable time from writing code that delivers actual value. To address these challenges, a new breed of tools, built atop LLMs, is emerging. These tools aim to mitigate drawbacks by employing techniques like fine-tuning or enriching user prompts with contextualized information.   In this paper, we delve into the lessons learned by a software development team venturing into the creation of such a contextualized LLM-based application, using retrieval-based techniques, called CodeBuddy. Over a four-month period, the team, despite lacking prior professional experience in LLM-based applications, built the product from scratch. Following the initial product release, we engaged with the development team responsible for the code generative components. Through interviews and analysis of the application's issue tracker, we uncover various intriguing challenges that teams working on LLM-based applications might encounter. For instance, we found three main group of lessons: LLM-based lessons, User-based lessons, and Technical lessons. By understanding these lessons, software development teams could become better prepared to build LLM-based applications.

{{</citation>}}


### (116/156) Autonomous Agents in Software Development: A Vision Paper (Zeeshan Rasheed et al., 2023)

{{<citation>}}

Zeeshan Rasheed, Muhammad Waseem, Kai-Kristian Kemell, Wang Xiaofeng, Anh Nguyen Duc, Kari Systä, Pekka Abrahamsson. (2023)  
**Autonomous Agents in Software Development: A Vision Paper**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: GPT, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.18440v1)  

---


**ABSTRACT**  
Large Language Models (LLM) and Generative Pre-trained Transformers (GPT), are reshaping the field of Software Engineering (SE). They enable innovative methods for executing many software engineering tasks, including automated code generation, debugging, maintenance, etc. However, only a limited number of existing works have thoroughly explored the potential of GPT agents in SE. This vision paper inquires about the role of GPT-based agents in SE. Our vision is to leverage the capabilities of multiple GPT agents to contribute to SE tasks and to propose an initial road map for future work. We argue that multiple GPT agents can perform creative and demanding tasks far beyond coding and debugging. GPT agents can also do project planning, requirements engineering, and software design. These can be done through high-level descriptions given by the human developer. We have shown in our initial experimental analysis for simple software (e.g., Snake Game, Tic-Tac-Toe, Notepad) that multiple GPT agents can produce high-quality code and document it carefully. We argue that it shows a promise of unforeseen efficiency and will dramatically reduce lead-times. To this end, we intend to expand our efforts to understand how we can scale these autonomous capabilities further.

{{</citation>}}


### (117/156) Navigating Privacy and Copyright Challenges Across the Data Lifecycle of Generative AI (Dawen Zhang et al., 2023)

{{<citation>}}

Dawen Zhang, Boming Xia, Yue Liu, Xiwei Xu, Thong Hoang, Zhenchang Xing, Mark Staples, Qinghua Lu, Liming Zhu. (2023)  
**Navigating Privacy and Copyright Challenges Across the Data Lifecycle of Generative AI**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CY, cs-LG, cs-SE, cs.SE  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2311.18252v1)  

---


**ABSTRACT**  
The advent of Generative AI has marked a significant milestone in artificial intelligence, demonstrating remarkable capabilities in generating realistic images, texts, and data patterns. However, these advancements come with heightened concerns over data privacy and copyright infringement, primarily due to the reliance on vast datasets for model training. Traditional approaches like differential privacy, machine unlearning, and data poisoning only offer fragmented solutions to these complex issues. Our paper delves into the multifaceted challenges of privacy and copyright protection within the data lifecycle. We advocate for integrated approaches that combines technical innovation with ethical foresight, holistically addressing these concerns by investigating and devising solutions that are informed by the lifecycle perspective. This work aims to catalyze a broader discussion and inspire concerted efforts towards data privacy and copyright integrity in Generative AI.

{{</citation>}}


## cs.ET (1)



### (118/156) Performance Analysis of Multi-Angle QAOA for p > 1 (Igor Gaidai et al., 2023)

{{<citation>}}

Igor Gaidai, Rebekah Herrman. (2023)  
**Performance Analysis of Multi-Angle QAOA for p > 1**  

---
Primary Category: cs.ET  
Categories: cs-ET, cs.ET, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.00200v1)  

---


**ABSTRACT**  
In this paper we consider the scalability of Multi-Angle QAOA with respect to the number of QAOA layers. We found that MA-QAOA is able to significantly reduce the depth of QAOA circuits, by a factor of up to 4 for the considered data sets. However, MA-QAOA is not optimal for minimization of the total QPU time. Different optimization initialization strategies are considered and compared for both QAOA and MA-QAOA. Among them, a new initialization strategy is suggested for MA-QAOA that is able to consistently and significantly outperform random initialization used in the previous studies.

{{</citation>}}


## stat.AP (1)



### (119/156) Planning Reliability Assurance Tests for Autonomous Vehicles (Simin Zheng et al., 2023)

{{<citation>}}

Simin Zheng, Lu Lu, Yili Hong, Jian Liu. (2023)  
**Planning Reliability Assurance Tests for Autonomous Vehicles**  

---
Primary Category: stat.AP  
Categories: cs-AI, stat-AP, stat.AP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00186v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) technology has become increasingly prevalent and transforms our everyday life. One important application of AI technology is the development of autonomous vehicles (AV). However, the reliability of an AV needs to be carefully demonstrated via an assurance test so that the product can be used with confidence in the field. To plan for an assurance test, one needs to determine how many AVs need to be tested for how many miles and the standard for passing the test. Existing research has made great efforts in developing reliability demonstration tests in the other fields of applications for product development and assessment. However, statistical methods have not been utilized in AV test planning. This paper aims to fill in this gap by developing statistical methods for planning AV reliability assurance tests based on recurrent events data. We explore the relationship between multiple criteria of interest in the context of planning AV reliability assurance tests. Specifically, we develop two test planning strategies based on homogeneous and non-homogeneous Poisson processes while balancing multiple objectives with the Pareto front approach. We also offer recommendations for practical use. The disengagement events data from the California Department of Motor Vehicles AV testing program is used to illustrate the proposed assurance test planning methods.

{{</citation>}}


## quant-ph (3)



### (120/156) Markovian Embeddings of Non-Markovian Quantum Systems: Coupled Stochastic and Quantum Master Equations for Non-Markovian Quantum Systems (H. I. Nurdin, 2023)

{{<citation>}}

H. I. Nurdin. (2023)  
**Markovian Embeddings of Non-Markovian Quantum Systems: Coupled Stochastic and Quantum Master Equations for Non-Markovian Quantum Systems**  

---
Primary Category: quant-ph  
Categories: cs-SY, eess-SY, quant-ph, quant-ph  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.00134v1)  

---


**ABSTRACT**  
Quantum Markov models are employed ubiquitously in quantum physics and in quantum information theory due to their relative simplicity and analytical tractability. In particular, these models are known to give accurate approximations for a wide range of quantum optical and mesoscopic systems. However, in general, the validity of the Markov approximation entails assumptions regarding properties of the system of interest and its environment, which may not be satisfied or accurate in arbitrary physical systems. Therefore, developing useful modelling tools for general non-Markovian quantum systems for which the Markov approximation is inappropriate or deficient is an undertaking of significant importance. This work considers non-Markovian principal quantum systems that can be embedded in a larger Markovian quantum system with one or more compound baths consisting of an auxiliary quantum system and a quantum white noise field, and derives a set of coupled stochastic and quantum master equations for embedded non-Markovian quantum systems. The case of a purely Hamiltonian coupling between the principal and auxiliary systems as a closed system without coupling to white noises is included as a special case. The results are expected to be of interest for (open-loop and feedback) control of continuous-time non-Markovian systems and studying reduced models for numerical simulation of such systems. They may also shed more light on the general structure of continuous-time non-Markovian quantum systems.

{{</citation>}}


### (121/156) A Comparison Between Invariant and Equivariant Classical and Quantum Graph Neural Networks (Roy T. Forestano et al., 2023)

{{<citation>}}

Roy T. Forestano, Marçal Comajoan Cara, Gopal Ramesh Dahale, Zhongtian Dong, Sergei Gleyzer, Daniel Justice, Kyoungchul Kong, Tom Magorsch, Konstantin T. Matchev, Katia Matcheva, Eyup B. Unlu. (2023)  
**A Comparison Between Invariant and Equivariant Classical and Quantum Graph Neural Networks**  

---
Primary Category: quant-ph  
Categories: cs-LG, hep-ph, quant-ph, quant-ph, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.18672v1)  

---


**ABSTRACT**  
Machine learning algorithms are heavily relied on to understand the vast amounts of data from high-energy particle collisions at the CERN Large Hadron Collider (LHC). The data from such collision events can naturally be represented with graph structures. Therefore, deep geometric methods, such as graph neural networks (GNNs), have been leveraged for various data analysis tasks in high-energy physics. One typical task is jet tagging, where jets are viewed as point clouds with distinct features and edge connections between their constituent particles. The increasing size and complexity of the LHC particle datasets, as well as the computational models used for their analysis, greatly motivate the development of alternative fast and efficient computational paradigms such as quantum computation. In addition, to enhance the validity and robustness of deep networks, one can leverage the fundamental symmetries present in the data through the use of invariant inputs and equivariant layers. In this paper, we perform a fair and comprehensive comparison between classical graph neural networks (GNNs) and equivariant graph neural networks (EGNNs) and their quantum counterparts: quantum graph neural networks (QGNNs) and equivariant quantum graph neural networks (EQGNN). The four architectures were benchmarked on a binary classification task to classify the parton-level particle initiating the jet. Based on their AUC scores, the quantum networks were shown to outperform the classical networks. However, seeing the computational advantage of the quantum networks in practice may have to wait for the further development of quantum technology and its associated APIs.

{{</citation>}}


### (122/156) Optimizing ZX-Diagrams with Deep Reinforcement Learning (Maximilian Nägele et al., 2023)

{{<citation>}}

Maximilian Nägele, Florian Marquardt. (2023)  
**Optimizing ZX-Diagrams with Deep Reinforcement Learning**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.18588v1)  

---


**ABSTRACT**  
ZX-diagrams are a powerful graphical language for the description of quantum processes with applications in fundamental quantum mechanics, quantum circuit optimization, tensor network simulation, and many more. The utility of ZX-diagrams relies on a set of local transformation rules that can be applied to them without changing the underlying quantum process they describe. These rules can be exploited to optimize the structure of ZX-diagrams for a range of applications. However, finding an optimal sequence of transformation rules is generally an open problem. In this work, we bring together ZX-diagrams with reinforcement learning, a machine learning technique designed to discover an optimal sequence of actions in a decision-making problem and show that a trained reinforcement learning agent can significantly outperform other optimization techniques like a greedy strategy or simulated annealing. The use of graph neural networks to encode the policy of the agent enables generalization to diagrams much bigger than seen during the training phase.

{{</citation>}}


## astro-ph.IM (1)



### (123/156) Scalable Bayesian uncertainty quantification with data-driven priors for radio interferometric imaging (Tobías I. Liaudat et al., 2023)

{{<citation>}}

Tobías I. Liaudat, Matthijs Mars, Matthew A. Price, Marcelo Pereyra, Marta M. Betcke, Jason D. McEwen. (2023)  
**Scalable Bayesian uncertainty quantification with data-driven priors for radio interferometric imaging**  

---
Primary Category: astro-ph.IM  
Categories: astro-ph-IM, astro-ph.IM, cs-LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.00125v1)  

---


**ABSTRACT**  
Next-generation radio interferometers like the Square Kilometer Array have the potential to unlock scientific discoveries thanks to their unprecedented angular resolution and sensitivity. One key to unlocking their potential resides in handling the deluge and complexity of incoming data. This challenge requires building radio interferometric imaging methods that can cope with the massive data sizes and provide high-quality image reconstructions with uncertainty quantification (UQ). This work proposes a method coined QuantifAI to address UQ in radio-interferometric imaging with data-driven (learned) priors for high-dimensional settings. Our model, rooted in the Bayesian framework, uses a physically motivated model for the likelihood. The model exploits a data-driven convex prior, which can encode complex information learned implicitly from simulations and guarantee the log-concavity of the posterior. We leverage probability concentration phenomena of high-dimensional log-concave posteriors that let us obtain information about the posterior, avoiding MCMC sampling techniques. We rely on convex optimisation methods to compute the MAP estimation, which is known to be faster and better scale with dimension than MCMC sampling strategies. Our method allows us to compute local credible intervals, i.e., Bayesian error bars, and perform hypothesis testing of structure on the reconstructed image. In addition, we propose a novel blazing-fast method to compute pixel-wise uncertainties at different scales. We demonstrate our method by reconstructing radio-interferometric images in a simulated setting and carrying out fast and scalable UQ, which we validate with MCMC sampling. Our method shows an improved image quality and more meaningful uncertainties than the benchmark method based on a sparsity-promoting prior. QuantifAI's source code: https://github.com/astro-informatics/QuantifAI.

{{</citation>}}


## cs.IT (3)



### (124/156) Adversarial Attacks and Defenses for Wireless Signal Classifiers using CDI-aware GANs (Sujata Sinha et al., 2023)

{{<citation>}}

Sujata Sinha, Alkan Soysal. (2023)  
**Adversarial Attacks and Defenses for Wireless Signal Classifiers using CDI-aware GANs**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-NI, cs.IT, eess-SP, math-IT  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2311.18820v1)  

---


**ABSTRACT**  
We introduce a Channel Distribution Information (CDI)-aware Generative Adversarial Network (GAN), designed to address the unique challenges of adversarial attacks in wireless communication systems. The generator in this CDI-aware GAN maps random input noise to the feature space, generating perturbations intended to deceive a target modulation classifier. Its discriminators play a dual role: one enforces that the perturbations follow a Gaussian distribution, making them indistinguishable from Gaussian noise, while the other ensures these perturbations account for realistic channel effects and resemble no-channel perturbations.   Our proposed CDI-aware GAN can be used as an attacker and a defender. In attack scenarios, the CDI-aware GAN demonstrates its prowess by generating robust adversarial perturbations that effectively deceive the target classifier, outperforming known methods. Furthermore, CDI-aware GAN as a defender significantly improves the target classifier's resilience against adversarial attacks.

{{</citation>}}


### (125/156) Learning for Semantic Knowledge Base-Guided Online Feature Transmission in Dynamic Channels (Xiangyu Gao et al., 2023)

{{<citation>}}

Xiangyu Gao, Yaping Sun, Dongyu Wei, Xiaodong Xu, Hao Chen, Hao Yin, Shuguang Cui. (2023)  
**Learning for Semantic Knowledge Base-Guided Online Feature Transmission in Dynamic Channels**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-LG, cs.IT, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18316v1)  

---


**ABSTRACT**  
With the proliferation of edge computing, efficient AI inference on edge devices has become essential for intelligent applications such as autonomous vehicles and VR/AR. In this context, we address the problem of efficient remote object recognition by optimizing feature transmission between mobile devices and edge servers. We propose an online optimization framework to address the challenge of dynamic channel conditions and device mobility in an end-to-end communication system. Our approach builds upon existing methods by leveraging a semantic knowledge base to drive multi-level feature transmission, accounting for temporal factors and dynamic elements throughout the transmission process. To solve the online optimization problem, we design a novel soft actor-critic-based deep reinforcement learning system with a carefully designed reward function for real-time decision-making, overcoming the optimization difficulty of the NP-hard problem and achieving the minimization of semantic loss while respecting latency constraints. Numerical results showcase the superiority of our approach compared to traditional greedy methods under various system setups.

{{</citation>}}


### (126/156) Reasoning with the Theory of Mind for Pragmatic Semantic Communication (Christo Kurisummoottil Thomas et al., 2023)

{{<citation>}}

Christo Kurisummoottil Thomas, Emilio Calvanese Strinati, Walid Saad. (2023)  
**Reasoning with the Theory of Mind for Pragmatic Semantic Communication**  

---
Primary Category: cs.IT  
Categories: cs-AI, cs-IT, cs-LG, cs.IT, math-IT  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2311.18224v1)  

---


**ABSTRACT**  
In this paper, a pragmatic semantic communication framework that enables effective goal-oriented information sharing between two-intelligent agents is proposed. In particular, semantics is defined as the causal state that encapsulates the fundamental causal relationships and dependencies among different features extracted from data. The proposed framework leverages the emerging concept in machine learning (ML) called theory of mind (ToM). It employs a dynamic two-level (wireless and semantic) feedback mechanism to continuously fine-tune neural network components at the transmitter. Thanks to the ToM, the transmitter mimics the actual mental state of the receiver's reasoning neural network operating semantic interpretation. Then, the estimated mental state at the receiver is dynamically updated thanks to the proposed dynamic two-level feedback mechanism. At the lower level, conventional channel quality metrics are used to optimize the channel encoding process based on the wireless communication channel's quality, ensuring an efficient mapping of semantic representations to a finite constellation. Additionally, a semantic feedback level is introduced, providing information on the receiver's perceived semantic effectiveness with minimal overhead. Numerical evaluations demonstrate the framework's ability to achieve efficient communication with a reduced amount of bits while maintaining the same semantics, outperforming conventional systems that do not exploit the ToM-based reasoning.

{{</citation>}}


## eess.IV (5)



### (127/156) Automated interpretation of congenital heart disease from multi-view echocardiograms (Jing Wang et al., 2023)

{{<citation>}}

Jing Wang, Xiaofeng Liu, Fangyun Wang, Lin Zheng, Fengqiao Gao, Hanwen Zhang, Xin Zhang, Wanqing Xie, Binbin Wang. (2023)  
**Automated interpretation of congenital heart disease from multi-view echocardiograms**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, cs-MM, eess-IV, eess.IV, physics-med-ph  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2311.18788v1)  

---


**ABSTRACT**  
Congenital heart disease (CHD) is the most common birth defect and the leading cause of neonate death in China. Clinical diagnosis can be based on the selected 2D key-frames from five views. Limited by the availability of multi-view data, most methods have to rely on the insufficient single view analysis. This study proposes to automatically analyze the multi-view echocardiograms with a practical end-to-end framework. We collect the five-view echocardiograms video records of 1308 subjects (including normal controls, ventricular septal defect (VSD) patients and atrial septal defect (ASD) patients) with both disease labels and standard-view key-frame labels. Depthwise separable convolution-based multi-channel networks are adopted to largely reduce the network parameters. We also approach the imbalanced class problem by augmenting the positive training samples. Our 2D key-frame model can diagnose CHD or negative samples with an accuracy of 95.4\%, and in negative, VSD or ASD classification with an accuracy of 92.3\%. To further alleviate the work of key-frame selection in real-world implementation, we propose an adaptive soft attention scheme to directly explore the raw video data. Four kinds of neural aggregation methods are systematically investigated to fuse the information of an arbitrary number of frames in a video. Moreover, with a view detection module, the system can work without the view records. Our video-based model can diagnose with an accuracy of 93.9\% (binary classification), and 92.1\% (3-class classification) in a collected 2D video testing set, which does not need key-frame selection and view annotation in testing. The detailed ablation study and the interpretability analysis are provided.

{{</citation>}}


### (128/156) DifAugGAN: A Practical Diffusion-style Data Augmentation for GAN-based Single Image Super-resolution (Axi Niu et al., 2023)

{{<citation>}}

Axi Niu, Kang Zhang, Joshua Tian Jin Tee, Trung X. Pham, Jinqiu Sun, Chang D. Yoo, In So Kweon, Yanning Zhang. (2023)  
**DifAugGAN: A Practical Diffusion-style Data Augmentation for GAN-based Single Image Super-resolution**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.18508v1)  

---


**ABSTRACT**  
It is well known the adversarial optimization of GAN-based image super-resolution (SR) methods makes the preceding SR model generate unpleasant and undesirable artifacts, leading to large distortion. We attribute the cause of such distortions to the poor calibration of the discriminator, which hampers its ability to provide meaningful feedback to the generator for learning high-quality images. To address this problem, we propose a simple but non-travel diffusion-style data augmentation scheme for current GAN-based SR methods, known as DifAugGAN. It involves adapting the diffusion process in generative diffusion models for improving the calibration of the discriminator during training motivated by the successes of data augmentation schemes in the field to achieve good calibration. Our DifAugGAN can be a Plug-and-Play strategy for current GAN-based SISR methods to improve the calibration of the discriminator and thus improve SR performance. Extensive experimental evaluations demonstrate the superiority of DifAugGAN over state-of-the-art GAN-based SISR methods across both synthetic and real-world datasets, showcasing notable advancements in both qualitative and quantitative results.

{{</citation>}}


### (129/156) Utilizing Radiomic Feature Analysis For Automated MRI Keypoint Detection: Enhancing Graph Applications (Sahar Almahfouz Nasser et al., 2023)

{{<citation>}}

Sahar Almahfouz Nasser, Shashwat Pathak, Keshav Singhal, Mohit Meena, Nihar Gupte, Ananya Chinmaya, Prateek Garg, Amit Sethi. (2023)  
**Utilizing Radiomic Feature Analysis For Automated MRI Keypoint Detection: Enhancing Graph Applications**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.18281v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) present a promising alternative to CNNs and transformers in certain image processing applications due to their parameter-efficiency in modeling spatial relationships. Currently, a major area of research involves the converting non-graph input data for GNN-based models, notably in scenarios where the data originates from images. One approach involves converting images into nodes by identifying significant keypoints within them. Super-Retina, a semi-supervised technique, has been utilized for detecting keypoints in retinal images. However, its limitations lie in the dependency on a small initial set of ground truth keypoints, which is progressively expanded to detect more keypoints. Having encountered difficulties in detecting consistent initial keypoints in brain images using SIFT and LoFTR, we proposed a new approach: radiomic feature-based keypoint detection. Demonstrating the anatomical significance of the detected keypoints was achieved by showcasing their efficacy in improving registration processes guided by these keypoints. Subsequently, these keypoints were employed as the ground truth for the keypoint detection method (LK-SuperRetina). Furthermore, the study showcases the application of GNNs in image matching, highlighting their superior performance in terms of both the number of good matches and confidence scores. This research sets the stage for expanding GNN applications into various other applications, including but not limited to image classification, segmentation, and registration.

{{</citation>}}


### (130/156) Consensus, dissensus and synergy between clinicians and specialist foundation models in radiology report generation (Ryutaro Tanno et al., 2023)

{{<citation>}}

Ryutaro Tanno, David G. T. Barrett, Andrew Sellergren, Sumedh Ghaisas, Sumanth Dathathri, Abigail See, Johannes Welbl, Karan Singhal, Shekoofeh Azizi, Tao Tu, Mike Schaekermann, Rhys May, Roy Lee, SiWai Man, Zahra Ahmed, Sara Mahdavi, Danielle Belgrave, Vivek Natarajan, Shravya Shetty, Pushmeet Kohli, Po-Sen Huang, Alan Karthikesalingam, Ira Ktena. (2023)  
**Consensus, dissensus and synergy between clinicians and specialist foundation models in radiology report generation**  

---
Primary Category: eess.IV  
Categories: cs-CL, cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18260v1)  

---


**ABSTRACT**  
Radiology reports are an instrumental part of modern medicine, informing key clinical decisions such as diagnosis and treatment. The worldwide shortage of radiologists, however, restricts access to expert care and imposes heavy workloads, contributing to avoidable errors and delays in report delivery. While recent progress in automated report generation with vision-language models offer clear potential in ameliorating the situation, the path to real-world adoption has been stymied by the challenge of evaluating the clinical quality of AI-generated reports. In this study, we build a state-of-the-art report generation system for chest radiographs, Flamingo-CXR, by fine-tuning a well-known vision-language foundation model on radiology data. To evaluate the quality of the AI-generated reports, a group of 16 certified radiologists provide detailed evaluations of AI-generated and human written reports for chest X-rays from an intensive care setting in the United States and an inpatient setting in India. At least one radiologist (out of two per case) preferred the AI report to the ground truth report in over 60$\%$ of cases for both datasets. Amongst the subset of AI-generated reports that contain errors, the most frequently cited reasons were related to the location and finding, whereas for human written reports, most mistakes were related to severity and finding. This disparity suggested potential complementarity between our AI system and human experts, prompting us to develop an assistive scenario in which Flamingo-CXR generates a first-draft report, which is subsequently revised by a clinician. This is the first demonstration of clinician-AI collaboration for report writing, and the resultant reports are assessed to be equivalent or preferred by at least one radiologist to reports written by experts alone in 80$\%$ of in-patient cases and 66$\%$ of intensive care cases.

{{</citation>}}


### (131/156) Automatic Detection of Alzheimer's Disease with Multi-Modal Fusion of Clinical MRI Scans (Long Chen et al., 2023)

{{<citation>}}

Long Chen, Liben Chen, Binfeng Xu, Wenxin Zhang, Narges Razavian. (2023)  
**Automatic Detection of Alzheimer's Disease with Multi-Modal Fusion of Clinical MRI Scans**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI, Clinical  
[Paper Link](http://arxiv.org/abs/2311.18245v1)  

---


**ABSTRACT**  
The aging population of the U.S. drives the prevalence of Alzheimer's disease. Brookmeyer et al. forecasts approximately 15 million Americans will have either clinical AD or mild cognitive impairment by 2060. In response to this urgent call, methods for early detection of Alzheimer's disease have been developed for prevention and pre-treatment. Notably, literature on the application of deep learning in the automatic detection of the disease has been proliferating. This study builds upon previous literature and maintains a focus on leveraging multi-modal information to enhance automatic detection. We aim to predict the stage of the disease - Cognitively Normal (CN), Mildly Cognitive Impairment (MCI), and Alzheimer's Disease (AD), based on two different types of brain MRI scans. We design an AlexNet-based deep learning model that learns the synergy of complementary information from both T1 and FLAIR MRI scans.

{{</citation>}}


## eess.SY (2)



### (132/156) Controlgym: Large-Scale Safety-Critical Control Environments for Benchmarking Reinforcement Learning Algorithms (Xiangyuan Zhang et al., 2023)

{{<citation>}}

Xiangyuan Zhang, Weichao Mao, Saviz Mowlavi, Mouhacine Benosman, Tamer Başar. (2023)  
**Controlgym: Large-Scale Safety-Critical Control Environments for Benchmarking Reinforcement Learning Algorithms**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-CE, cs-LG, cs-SY, eess-SY, eess.SY, math-OC  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.18736v1)  

---


**ABSTRACT**  
We introduce controlgym, a library of thirty-six safety-critical industrial control settings, and ten infinite-dimensional partial differential equation (PDE)-based control problems. Integrated within the OpenAI Gym/Gymnasium (Gym) framework, controlgym allows direct applications of standard reinforcement learning (RL) algorithms like stable-baselines3. Our control environments complement those in Gym with continuous, unbounded action and observation spaces, motivated by real-world control applications. Moreover, the PDE control environments uniquely allow the users to extend the state dimensionality of the system to infinity while preserving the intrinsic dynamics. This feature is crucial for evaluating the scalability of RL algorithms for control. This project serves the learning for dynamics & control (L4DC) community, aiming to explore key questions: the convergence of RL algorithms in learning control policies; the stability and robustness issues of learning-based controllers; and the scalability of RL algorithms to high- and potentially infinite-dimensional systems. We open-source the controlgym project at https://github.com/xiangyuan-zhang/controlgym.

{{</citation>}}


### (133/156) Deep Reinforcement Learning Based Optimal Energy Management of Multi-energy Microgrids with Uncertainties (Yang Cui et al., 2023)

{{<citation>}}

Yang Cui, Yang Xu, Yang Li, Yijian Wang, Xinpeng Zou. (2023)  
**Deep Reinforcement Learning Based Optimal Energy Management of Multi-energy Microgrids with Uncertainties**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.18327v1)  

---


**ABSTRACT**  
Multi-energy microgrid (MEMG) offers an effective approach to deal with energy demand diversification and new energy consumption on the consumer side. In MEMG, it is critical to deploy an energy management system (EMS) for efficient utilization of energy and reliable operation of the system. To help EMS formulate optimal dispatching schemes, a deep reinforcement learning (DRL)-based MEMG energy management scheme with renewable energy source (RES) uncertainty is proposed in this paper. To accurately describe the operating state of the MEMG, the off-design performance model of energy conversion devices is considered in scheduling. The nonlinear optimal dispatching model is expressed as a Markov decision process (MDP) and is then addressed by the twin delayed deep deterministic policy gradient (TD3) algorithm. In addition, to accurately describe the uncertainty of RES, the conditional-least squares generative adversarial networks (C-LSGANs) method based on RES forecast power is proposed to construct the scenarios set of RES power generation. The generated data of RES is used for scheduling to obtain caps and floors for the purchase of electricity and natural gas. Based on this, the superior energy supply sector can formulate solutions in advance to tackle the uncertainty of RES. Finally, the simulation analysis demonstrates the validity and superiority of the method.

{{</citation>}}


## eess.SP (1)



### (134/156) Indoor Millimeter Wave Localization using Multiple Self-Supervised Tiny Neural Networks (Anish Shastri et al., 2023)

{{<citation>}}

Anish Shastri, Andres Garcia-Saavedra, Paolo Casari. (2023)  
**Indoor Millimeter Wave Localization using Multiple Self-Supervised Tiny Neural Networks**  

---
Primary Category: eess.SP  
Categories: cs-LG, cs-NI, eess-SP, eess.SP  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.18732v1)  

---


**ABSTRACT**  
We consider the localization of a mobile millimeter-wave client in a large indoor environment using multilayer perceptron neural networks (NNs). Instead of training and deploying a single deep model, we proceed by choosing among multiple tiny NNs trained in a self-supervised manner. The main challenge then becomes to determine and switch to the best NN among the available ones, as an incorrect NN will fail to localize the client. In order to upkeep the localization accuracy, we propose two switching schemes: one based on a Kalman filter, and one based on the statistical distribution of the training data. We analyze the proposed schemes via simulations, showing that our approach outperforms both geometric localization schemes and the use of a single NN.

{{</citation>}}


## stat.ME (1)



### (135/156) AI in Pharma for Personalized Sequential Decision-Making: Methods, Applications and Opportunities (Yuhan Li et al., 2023)

{{<citation>}}

Yuhan Li, Hongtao Zhang, Keaven Anderson, Songzi Li, Ruoqing Zhu. (2023)  
**AI in Pharma for Personalized Sequential Decision-Making: Methods, Applications and Opportunities**  

---
Primary Category: stat.ME  
Categories: cs-LG, stat-AP, stat-ME, stat-ML, stat.ME  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18725v1)  

---


**ABSTRACT**  
In the pharmaceutical industry, the use of artificial intelligence (AI) has seen consistent growth over the past decade. This rise is attributed to major advancements in statistical machine learning methodologies, computational capabilities and the increased availability of large datasets. AI techniques are applied throughout different stages of drug development, ranging from drug discovery to post-marketing benefit-risk assessment. Kolluri et al. provided a review of several case studies that span these stages, featuring key applications such as protein structure prediction, success probability estimation, subgroup identification, and AI-assisted clinical trial monitoring. From a regulatory standpoint, there was a notable uptick in submissions incorporating AI components in 2021. The most prevalent therapeutic areas leveraging AI were oncology (27%), psychiatry (15%), gastroenterology (12%), and neurology (11%). The paradigm of personalized or precision medicine has gained significant traction in recent research, partly due to advancements in AI techniques \cite{hamburg2010path}. This shift has had a transformative impact on the pharmaceutical industry. Departing from the traditional "one-size-fits-all" model, personalized medicine incorporates various individual factors, such as environmental conditions, lifestyle choices, and health histories, to formulate customized treatment plans. By utilizing sophisticated machine learning algorithms, clinicians and researchers are better equipped to make informed decisions in areas such as disease prevention, diagnosis, and treatment selection, thereby optimizing health outcomes for each individual.

{{</citation>}}


## cs.IR (3)



### (136/156) Routing-Guided Learned Product Quantization for Graph-Based Approximate Nearest Neighbor Search (Qiang Yue et al., 2023)

{{<citation>}}

Qiang Yue, Xiaoliang Xu, Yuxiang Wang, Yikun Tao, Xuliyuan Luo. (2023)  
**Routing-Guided Learned Product Quantization for Graph-Based Approximate Nearest Neighbor Search**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2311.18724v1)  

---


**ABSTRACT**  
Given a vector dataset $\mathcal{X}$, a query vector $\vec{x}_q$, graph-based Approximate Nearest Neighbor Search (ANNS) aims to build a proximity graph (PG) as an index of $\mathcal{X}$ and approximately return vectors with minimum distances to $\vec{x}_q$ by searching over the PG index. It suffers from the large-scale $\mathcal{X}$ because a PG with full vectors is too large to fit into the memory, e.g., a billion-scale $\mathcal{X}$ in 128 dimensions would consume nearly 600 GB memory. To solve this, Product Quantization (PQ) integrated graph-based ANNS is proposed to reduce the memory usage, using smaller compact codes of quantized vectors in memory instead of the large original vectors. Existing PQ methods do not consider the important routing features of PG, resulting in low-quality quantized vectors that affect the ANNS's effectiveness. In this paper, we present an end-to-end Routing-guided learned Product Quantization (RPQ) for graph-based ANNS. It consists of (1) a \textit{differentiable quantizer} used to make the standard discrete PQ differentiable to suit for back-propagation of end-to-end learning, (2) a \textit{sampling-based feature extractor} used to extract neighborhood and routing features of a PG, and (3) a \textit{multi-feature joint training module} with two types of feature-aware losses to continuously optimize the differentiable quantizer. As a result, the inherent features of a PG would be embedded into the learned PQ, generating high-quality quantized vectors. Moreover, we integrate our RPQ with the state-of-the-art DiskANN and existing popular PGs to improve their performance. Comprehensive experiments on real-world large-scale datasets (from 1M to 1B) demonstrate RPQ's superiority, e.g., 1.7$\times$-4.2$\times$ improvement on QPS at the same recall@10 of 95\%.

{{</citation>}}


### (137/156) Search Still Matters: Information Retrieval in the Era of Generative AI (William R. Hersh, 2023)

{{<citation>}}

William R. Hersh. (2023)  
**Search Still Matters: Information Retrieval in the Era of Generative AI**  

---
Primary Category: cs.IR  
Categories: H-3, cs-AI, cs-IR, cs.IR  
Keywords: AI, Generative AI, Information Retrieval  
[Paper Link](http://arxiv.org/abs/2311.18550v1)  

---


**ABSTRACT**  
Objective: Information retrieval (IR, also known as search) systems are ubiquitous in modern times. How does the emergence of generative artificial intelligence (AI), based on large language models (LLMs), fit into the IR process? Process: This perspective explores the use of generative AI in the context of the motivations, considerations, and outcomes of the IR process with a focus on the academic use of such systems. Conclusions: There are many information needs, from simple to complex, that motivate use of IR. Users of such systems, particularly academics, have concerns for authoritativeness, timeliness, and contextualization of search. While LLMs may provide functionality that aids the IR process, the continued need for search systems, and research into their improvement, remains essential.

{{</citation>}}


### (138/156) Enhancing Cross-domain Click-Through Rate Prediction via Explicit Feature Augmentation (Xu Chen et al., 2023)

{{<citation>}}

Xu Chen, Zida Cheng, Jiangchao Yao, Chen Ju, Weilin Huang, Jinsong Lan, Xiaoyi Zeng, Shuai Xiao. (2023)  
**Enhancing Cross-domain Click-Through Rate Prediction via Explicit Feature Augmentation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2312.00078v1)  

---


**ABSTRACT**  
Cross-domain CTR (CDCTR) prediction is an important research topic that studies how to leverage meaningful data from a related domain to help CTR prediction in target domain. Most existing CDCTR works design implicit ways to transfer knowledge across domains such as parameter-sharing that regularizes the model training in target domain. More effectively, recent researchers propose explicit techniques to extract user interest knowledge and transfer this knowledge to target domain. However, the proposed method mainly faces two issues: 1) it usually requires a super domain, i.e. an extremely large source domain, to cover most users or items of target domain, and 2) the extracted user interest knowledge is static no matter what the context is in target domain. These limitations motivate us to develop a more flexible and efficient technique to explicitly transfer knowledge. In this work, we propose a cross-domain augmentation network (CDAnet) being able to perform explicit knowledge transfer between two domains. Specifically, CDAnet contains a designed translation network and an augmentation network which are trained sequentially. The translation network computes latent features from two domains and learns meaningful cross-domain knowledge of each input in target domain by using a designed cross-supervised feature translator. Later the augmentation network employs the explicit cross-domain knowledge as augmented information to boost the target domain CTR prediction. Through extensive experiments on two public benchmarks and one industrial production dataset, we show CDAnet can learn meaningful translated features and largely improve the performance of CTR prediction. CDAnet has been conducted online A/B test in image2product retrieval at Taobao app, bringing an absolute 0.11 point CTR improvement, a relative 0.64% deal growth and a relative 1.26% GMV increase.

{{</citation>}}


## stat.ML (1)



### (139/156) Balancing Summarization and Change Detection in Graph Streams (Shintaro Fukushima et al., 2023)

{{<citation>}}

Shintaro Fukushima, Kenji Yamanishi. (2023)  
**Balancing Summarization and Change Detection in Graph Streams**  

---
Primary Category: stat.ML  
Categories: cs-IT, cs-LG, math-IT, stat-ML, stat.ML  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2311.18694v1)  

---


**ABSTRACT**  
This study addresses the issue of balancing graph summarization and graph change detection. Graph summarization compresses large-scale graphs into a smaller scale. However, the question remains: To what extent should the original graph be compressed? This problem is solved from the perspective of graph change detection, aiming to detect statistically significant changes using a stream of summary graphs. If the compression rate is extremely high, important changes can be ignored, whereas if the compression rate is extremely low, false alarms may increase with more memory. This implies that there is a trade-off between compression rate in graph summarization and accuracy in change detection. We propose a novel quantitative methodology to balance this trade-off to simultaneously realize reliable graph summarization and change detection. We introduce a probabilistic structure of hierarchical latent variable model into a graph, thereby designing a parameterized summary graph on the basis of the minimum description length principle. The parameter specifying the summary graph is then optimized so that the accuracy of change detection is guaranteed to suppress Type I error probability (probability of raising false alarms) to be less than a given confidence level. First, we provide a theoretical framework for connecting graph summarization with change detection. Then, we empirically demonstrate its effectiveness on synthetic and real datasets.

{{</citation>}}


## cs.AR (1)



### (140/156) Splitwise: Efficient generative LLM inference using phase splitting (Pratyush Patel et al., 2023)

{{<citation>}}

Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Aashaka Shah, Saeed Maleki, Ricardo Bianchini. (2023)  
**Splitwise: Efficient generative LLM inference using phase splitting**  

---
Primary Category: cs.AR  
Categories: I-2-0, I-3-1, C-4, cs-AR, cs-DC, cs.AR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18677v1)  

---


**ABSTRACT**  
Recent innovations in generative large language models (LLMs) have made their applications and use-cases ubiquitous. This has led to large-scale deployments of these models, using complex, expensive, and power-hungry AI accelerators, most commonly GPUs. These developments make LLM inference efficiency an important challenge. Based on our extensive characterization, we find that there are two main phases during an LLM inference request: a compute-intensive prompt computation, and a memory-intensive token generation, each with distinct latency, throughput, memory, and power characteristics. Despite state-of-the-art batching and scheduling, the token generation phase underutilizes compute resources. Specifically, unlike compute-intensive prompt computation phases, token generation phases do not require the compute capability of the latest GPUs, and can be run with lower power and cost.   With Splitwise, we propose splitting the two phases of a LLM inference request on to separate machines. This allows us to use hardware that is well-suited for each phase, and provision resources independently per phase. However, splitting an inference request across machines requires state transfer from the machine running prompt computation over to the machine generating tokens. We implement and optimize this state transfer using the fast back-plane interconnects available in today's GPU clusters.   We use the Splitwise technique to design LLM inference clusters using the same or different types of machines for the prompt computation and token generation phases. Our clusters are optimized for three key objectives: throughput, cost, and power. In particular, we show that we can achieve 1.4x higher throughput at 20% lower cost than current designs. Alternatively, we can achieve 2.35x more throughput with the same cost and power budgets.

{{</citation>}}


## cs.SI (3)



### (141/156) DQSSA: A Quantum-Inspired Solution for Maximizing Influence in Online Social Networks (Student Abstract) (Aryaman Rao et al., 2023)

{{<citation>}}

Aryaman Rao, Parth Singh, Dinesh Kumar Vishwakarma, Mukesh Prasad. (2023)  
**DQSSA: A Quantum-Inspired Solution for Maximizing Influence in Online Social Networks (Student Abstract)**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2311.18676v1)  

---


**ABSTRACT**  
Influence Maximization is the task of selecting optimal nodes maximising the influence spread in social networks. This study proposes a Discretized Quantum-based Salp Swarm Algorithm (DQSSA) for optimizing influence diffusion in social networks. By discretizing meta-heuristic algorithms and infusing them with quantum-inspired enhancements, we address issues like premature convergence and low efficacy. The proposed method, guided by quantum principles, offers a promising solution for Influence Maximisation. Experiments on four real-world datasets reveal DQSSA's superior performance as compared to established cutting-edge algorithms.

{{</citation>}}


### (142/156) CrimeGAT: Leveraging Graph Attention Networks for Enhanced Predictive Policing in Criminal Networks (Chen Yang, 2023)

{{<citation>}}

Chen Yang. (2023)  
**CrimeGAT: Leveraging Graph Attention Networks for Enhanced Predictive Policing in Criminal Networks**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Attention, Graph Attention Network  
[Paper Link](http://arxiv.org/abs/2311.18641v1)  

---


**ABSTRACT**  
In this paper, we present CrimeGAT, a novel application of Graph Attention Networks (GATs) for predictive policing in criminal networks. Criminal networks pose unique challenges for predictive analytics due to their complex structure, multi-relational links, and dynamic behavior. Traditional methods often fail to capture these complexities, leading to suboptimal predictions. To address these challenges, we propose the use of GATs, which can effectively leverage both node features and graph structure to make predictions. Our proposed CrimeGAT model integrates attention mechanisms to weigh the importance of a node's neighbors, thereby capturing the local and global structures of criminal networks. We formulate the problem as learning a function that maps node features and graph structure to a prediction of future criminal activity. The experimental results on real-world datasets demonstrate that CrimeGAT out-performs conventional methods in predicting criminal activities, thereby providing a powerful tool for law enforcement agencies to proactively deploy resources. Furthermore, the interpretable nature of the attentionmechanism inGATs offers insights into the key players and relationships in criminal networks. This research opens new avenues for applying deep learning techniques in the Aeld of predictive policing and criminal network analysis.

{{</citation>}}


### (143/156) CrimeGraphNet: Link Prediction in Criminal Networks with Graph Convolutional Networks (Chen Yang, 2023)

{{<citation>}}

Chen Yang. (2023)  
**CrimeGraphNet: Link Prediction in Criminal Networks with Graph Convolutional Networks**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2311.18543v1)  

---


**ABSTRACT**  
In this paper, we introduce CrimeGraphNet, a novel approach for link prediction in criminal networks utilizingGraph Convolutional Networks (GCNs). Criminal networks are intricate and dynamic, with covert links that are challenging to uncover. Accurate prediction of these links can aid in proactive crime prevention and investigation. Existing methods often fail to capture the complex interconnections in such networks. They also struggle in scenarios where only limited labeled data is available for training. To address these challenges, we propose CrimeGraphNet, which leverages the power of GCNs for link prediction in these networks. The GCNmodel effectively captures topological features and node characteristics, making it well-suited for this task. We evaluate CrimeGraphNet on several real-world criminal network datasets. Our results demonstrate that CrimeGraphNet outperforms existing methods in terms of prediction accuracy, robustness, and computational efAciency. Furthermore, our approach enables the extraction of meaningful insights from the predicted links, thereby contributing to a better understanding of the underlying criminal activities. Overall, CrimeGraphNet represents a signiAcant step forward in the use of deep learning for criminal network analysis.

{{</citation>}}


## cs.CR (3)



### (144/156) Scalable and Lightweight Post-Quantum Authentication for Internet of Things (Attila A. Yavuz et al., 2023)

{{<citation>}}

Attila A. Yavuz, Saleh Darzi, Saif E. Nouma. (2023)  
**Scalable and Lightweight Post-Quantum Authentication for Internet of Things**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Falcon  
[Paper Link](http://arxiv.org/abs/2311.18674v2)  

---


**ABSTRACT**  
Internet of Things (IoT) applications are composed of massive quantities of resource-limited devices that collect sensitive data with long-term operational and security requirements. With the threat of emerging quantum computers, Post-Quantum Cryptography (PQC) is a critical requirement for IoTs. In particular, digital signatures offer scalable authentication with non-repudiation and are an essential tool for IoTs. However, as seen in NIST PQC standardization, post-quantum signatures are extremely costly for resource-limited IoTs. Hence, there is a significant need for quantum-safe signatures that respect the processing, memory, and bandwidth limitations of IoTs. In this paper, we created a new lightweight quantum-safe digital signature referred to as INFinity-HORS (INF-HORS), which is (to the best of our knowledge) the first signer-optimal hash-based signature with (polynomially) unbounded signing capability. INF-HORS enables a verifier to non-interactively construct one-time public keys from a master public key via encrypted function evaluations. This strategy avoids the performance bottleneck of hash-based standards (e.g., SPHINCS+) by eliminating hyper-tree structures. It also does not require a trusted party or non-colliding servers to distribute public keys. Our performance analysis confirms that INF-HORS is magnitudes of times more signer computation efficient than selected NIST PQC schemes (e.g., SPHINCS+, Dilithium, Falcon) with a small memory footprint.

{{</citation>}}


### (145/156) Detecting Anomalous Network Communication Patterns Using Graph Convolutional Networks (Yizhak Vaisman et al., 2023)

{{<citation>}}

Yizhak Vaisman, Gilad Katz, Yuval Elovici, Asaf Shabtai. (2023)  
**Detecting Anomalous Network Communication Patterns Using Graph Convolutional Networks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2311.18525v1)  

---


**ABSTRACT**  
To protect an organizations' endpoints from sophisticated cyberattacks, advanced detection methods are required. In this research, we present GCNetOmaly: a graph convolutional network (GCN)-based variational autoencoder (VAE) anomaly detector trained on data that include connection events among internal and external machines. As input, the proposed GCN-based VAE model receives two matrices: (i) the normalized adjacency matrix, which represents the connections among the machines, and (ii) the feature matrix, which includes various features (demographic, statistical, process-related, and Node2vec structural features) that are used to profile the individual nodes/machines. After training the model on data collected for a predefined time window, the model is applied on the same data; the reconstruction score obtained by the model for a given machine then serves as the machine's anomaly score. GCNetOmaly was evaluated on real, large-scale data logged by Carbon Black EDR from a large financial organization's automated teller machines (ATMs) as well as communication with Active Directory (AD) servers in two setups: unsupervised and supervised. The results of our evaluation demonstrate GCNetOmaly's effectiveness in detecting anomalous behavior of machines on unsupervised data.

{{</citation>}}


### (146/156) The Role of Visual Features in Text-Based CAPTCHAs: An fNIRS Study for Usable Security (Emre Mulazimoglu et al., 2023)

{{<citation>}}

Emre Mulazimoglu, Murat P. Cakir, Cengiz Acarturk. (2023)  
**The Role of Visual Features in Text-Based CAPTCHAs: An fNIRS Study for Usable Security**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-HC, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.18436v1)  

---


**ABSTRACT**  
To mitigate dictionary attacks or similar undesirable automated attacks to information systems, developers mostly prefer using CAPTCHA challenges as Human Interactive Proofs (HIPs) to distinguish between human users and scripts. Appropriate use of CAPTCHA requires a setup that balances between robustness and usability during the design of a challenge. The previous research reveals that most usability studies have used accuracy and response time as measurement criteria for quantitative analysis. The present study aims at applying optical neuroimaging techniques for the analysis of CAPTCHA design. The functional Near-Infrared Spectroscopy technique was used to explore the hemodynamic responses in the prefrontal cortex elicited by CAPTCHA stimulus of varying types. )e findings suggest that regions in the left and right dorsolateral and right dorsomedial prefrontal cortex respond to the degrees of line occlusion, rotation, and wave distortions present in a CAPTCHA. The systematic addition of the visual effects introduced nonlinear effects on the behavioral and prefrontal oxygenation measures, indicative of the emergence of Gestalt effects that might have influenced the perception of the overall CAPTCHA figure.

{{</citation>}}


## cs.AI (1)



### (147/156) Solving the Team Orienteering Problem with Transformers (Daniel Fuertes et al., 2023)

{{<citation>}}

Daniel Fuertes, Carlos R. del-Blanco, Fernando Jaureguizar, Narciso García. (2023)  
**Solving the Team Orienteering Problem with Transformers**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.18662v2)  

---


**ABSTRACT**  
Route planning for a fleet of vehicles is an important task in applications such as package delivery, surveillance, or transportation. This problem is usually modeled as a Combinatorial Optimization problem named as Team Orienteering Problem. The most popular Team Orienteering Problem solvers are mainly based on either linear programming, which provides accurate solutions by employing a large computation time that grows with the size of the problem, or heuristic methods, which usually find suboptimal solutions in a shorter amount of time. In this paper, a multi-agent route planning system capable of solving the Team Orienteering Problem in a very fast and accurate manner is presented. The proposed system is based on a centralized Transformer neural network that can learn to encode the scenario (modeled as a graph) and the context of the agents to provide fast and accurate solutions. Several experiments have been performed to demonstrate that the presented system can outperform most of the state-of-the-art works in terms of computation speed. In addition, the code is publicly available at http://gti.ssr.upm.es/data.

{{</citation>}}


## cs.DC (1)



### (148/156) Comparison of Autoscaling Frameworks for Containerised Machine-Learning-Applications in a Local and Cloud Environment (Christian Schroeder et al., 2023)

{{<citation>}}

Christian Schroeder, Rene Boehm, Alexander Lampe. (2023)  
**Comparison of Autoscaling Frameworks for Containerised Machine-Learning-Applications in a Local and Cloud Environment**  

---
Primary Category: cs.DC  
Categories: 94-04, I-2-11, cs-DC, cs.DC  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2311.18659v1)  

---


**ABSTRACT**  
When deploying machine learning (ML) applications, the automated allocation of computing resources-commonly referred to as autoscaling-is crucial for maintaining a consistent inference time under fluctuating workloads. The objective is to maximize the Quality of Service metrics, emphasizing performance and availability, while minimizing resource costs. In this paper, we compare scalable deployment techniques across three levels of scaling: at the application level (TorchServe, RayServe) and the container level (K3s) in a local environment (production server), as well as at the container and machine levels in a cloud environment (Amazon Web Services Elastic Container Service and Elastic Kubernetes Service). The comparison is conducted through the study of mean and standard deviation of inference time in a multi-client scenario, along with upscaling response times. Based on this analysis, we propose a deployment strategy for both local and cloud-based environments.

{{</citation>}}


## cs.SD (1)



### (149/156) Barwise Music Structure Analysis with the Correlation Block-Matching Segmentation Algorithm (Axel Marmoret et al., 2023)

{{<citation>}}

Axel Marmoret, Jérémy E. Cohen, Frédéric Bimbot. (2023)  
**Barwise Music Structure Analysis with the Correlation Block-Matching Segmentation Algorithm**  

---
Primary Category: cs.SD  
Categories: H-5-5, cs-IR, cs-SD, cs.SD, eess-AS  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2311.18604v1)  

---


**ABSTRACT**  
Music Structure Analysis (MSA) is a Music Information Retrieval task consisting of representing a song in a simplified, organized manner by breaking it down into sections typically corresponding to ``chorus'', ``verse'', ``solo'', etc. In this work, we extend an MSA algorithm called the Correlation Block-Matching (CBM) algorithm introduced by (Marmoret et al., 2020, 2022b). The CBM algorithm is a dynamic programming algorithm that segments self-similarity matrices, which are a standard description used in MSA and in numerous other applications. In this work, self-similarity matrices are computed from the feature representation of an audio signal and time is sampled at the bar-scale. This study examines three different standard similarity functions for the computation of self-similarity matrices. Results show that, in optimal conditions, the proposed algorithm achieves a level of performance which is competitive with supervised state-of-the-art methods while only requiring knowledge of bar positions. In addition, the algorithm is made open-source and is highly customizable.

{{</citation>}}


## cs.CE (1)



### (150/156) A Formulation of Structural Design Optimization Problems for Quantum Annealing (Fabian Key et al., 2023)

{{<citation>}}

Fabian Key, Lukas Freinberger. (2023)  
**A Formulation of Structural Design Optimization Problems for Quantum Annealing**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.18565v1)  

---


**ABSTRACT**  
We present a novel formulation of structural design optimization problems specifically tailored to be solved by quantum annealing (QA). Structural design optimization aims to find the best, i.e., material-efficient yet high-performance, configuration of a structure. To this end, computational optimization strategies can be employed, where a recently evolving strategy based on quantum mechanical effects is QA. This approach requires the optimization problem to be present, e.g., as a quadratic unconstrained binary optimization (QUBO) model. Thus, we develop a novel formulation of the optimization problem. The latter typically involves an analysis model for the component. Here, we use energy minimization principles that govern the behavior of structures under applied loads. This allows us to state the optimization problem as one overall minimization problem. Next, we map this to a QUBO problem that can be immediately solved by QA. We validate the proposed approach using a size optimization problem of a compound rod under self-weight loading. To this end, we develop strategies to account for the limitations of currently available hardware and find that the presented formulation is suitable for solving structural design optimization problems through QA and, for small-scale problems, already works on today's hardware.

{{</citation>}}


## cs.HC (2)



### (151/156) Multiple Disciplinary Data Work Practices in Artificial Intelligence Research: a Healthcare Case Study in the UK (Rafael Henkin et al., 2023)

{{<citation>}}

Rafael Henkin, Elizabeth Remfry, Duncan J. Reynolds, Megan Clinch, Michael R. Barnes. (2023)  
**Multiple Disciplinary Data Work Practices in Artificial Intelligence Research: a Healthcare Case Study in the UK**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CY, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18424v1)  

---


**ABSTRACT**  
Developing artificial intelligence (AI) tools for healthcare is a multiple disciplinary effort, bringing data scientists, clinicians, patients and other disciplines together. In this paper, we explore the AI development workflow and how participants navigate the challenges and tensions of sharing and generating knowledge across disciplines. Through an inductive thematic analysis of 13 semi-structured interviews with participants in a large research consortia, our findings suggest that multiple disciplinarity heavily impacts work practices. Participants faced challenges to learn the languages of other disciplines and needed to adapt the tools used for sharing and communicating with their audience, particularly those from a clinical or patient perspective. Large health datasets also posed certain restrictions on work practices. We identified meetings as a key platform for facilitating exchanges between disciplines and allowing for the blending and creation of knowledge. Finally, we discuss design implications for data science and collaborative tools, and recommendations for future research.

{{</citation>}}


### (152/156) Can Large Language Models Be Good Companions? An LLM-Based Eyewear System with Conversational Common Ground (Zhenyu Xu et al., 2023)

{{<citation>}}

Zhenyu Xu, Hailin Xu, Zhouyang Lu, Yingying Zhao, Rui Zhu, Yujiang Wang, Mingzhi Dong, Yuhu Chang, Qin Lv, Robert P. Dick, Fan Yang, Tun Lu, Ning Gu, Li Shang. (2023)  
**Can Large Language Models Be Good Companions? An LLM-Based Eyewear System with Conversational Common Ground**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.18251v1)  

---


**ABSTRACT**  
Developing chatbots as personal companions has long been a goal of artificial intelligence researchers. Recent advances in Large Language Models (LLMs) have delivered a practical solution for endowing chatbots with anthropomorphic language capabilities. However, it takes more than LLMs to enable chatbots that can act as companions. Humans use their understanding of individual personalities to drive conversations. Chatbots also require this capability to enable human-like companionship. They should act based on personalized, real-time, and time-evolving knowledge of their owner. We define such essential knowledge as the \textit{common ground} between chatbots and their owners, and we propose to build a common-ground-aware dialogue system from an LLM-based module, named \textit{OS-1}, to enable chatbot companionship. Hosted by eyewear, OS-1 can sense the visual and audio signals the user receives and extract real-time contextual semantics. Those semantics are categorized and recorded to formulate historical contexts from which the user's profile is distilled and evolves over time, i.e., OS-1 gradually learns about its user. OS-1 combines knowledge from real-time semantics, historical contexts, and user-specific profiles to produce a common-ground-aware prompt input into the LLM module. The LLM's output is converted to audio, spoken to the wearer when appropriate.We conduct laboratory and in-field studies to assess OS-1's ability to build common ground between the chatbot and its user. The technical feasibility and capabilities of the system are also evaluated. OS-1, with its common-ground awareness, can significantly improve user satisfaction and potentially lead to downstream tasks such as personal emotional support and assistance.

{{</citation>}}


## physics.chem-ph (1)



### (153/156) Transfer Learning across Different Chemical Domains: Virtual Screening of Organic Materials with Deep Learning Models Pretrained on Small Molecule and Chemical Reaction Data (Chengwei Zhang et al., 2023)

{{<citation>}}

Chengwei Zhang, Yushuang Zhai, Ziyang Gong, Yuan-Bin She, Yun-Fang Yang, An Su. (2023)  
**Transfer Learning across Different Chemical Domains: Virtual Screening of Organic Materials with Deep Learning Models Pretrained on Small Molecule and Chemical Reaction Data**  

---
Primary Category: physics.chem-ph  
Categories: cs-LG, physics-chem-ph, physics.chem-ph, q-bio-BM  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2311.18377v1)  

---


**ABSTRACT**  
Machine learning prediction of organic materials properties is an efficient virtual screening method ahead of more expensive screening methods. However, this approach has suffered from insufficient labeled data on organic materials to train state-of-the-art machine learning models. In this study, we demonstrate that drug-like small molecule and chemical reaction databases can be used to pretrain the BERT model for the virtual screening of organic materials. Among the BERT models fine-tuned by five virtual screening tasks on organic materials, the USPTO-SMILES pretrained BERT model had R2 > 0.90 for two tasks and R2 > 0.82 for one, which was generally superior to the same models pretrained by the small molecule or organic materials databases, as well as to the other three traditional machine learning models trained directly on the virtual screening task data. The superior performance of the USPTO-SMILES pretrained BERT model is due to the greater variety of organic building blocks in the USPTO database and the broader coverage of the chemical space. The even better performance of the BERT model pretrained externally from a chemical reaction database with additional sources of chemical reactions strengthens our proof of concept that transfer learning across different chemical domains is practical for the virtual screening of organic materials.

{{</citation>}}


## physics.ao-ph (1)



### (154/156) PAUNet: Precipitation Attention-based U-Net for rain prediction from satellite radiance data (P. Jyoteeshkumar Reddy et al., 2023)

{{<citation>}}

P. Jyoteeshkumar Reddy, Harish Baki, Sandeep Chinta, Richard Matear, John Taylor. (2023)  
**PAUNet: Precipitation Attention-based U-Net for rain prediction from satellite radiance data**  

---
Primary Category: physics.ao-ph  
Categories: cs-LG, physics-ao-ph, physics.ao-ph  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.18306v1)  

---


**ABSTRACT**  
This paper introduces Precipitation Attention-based U-Net (PAUNet), a deep learning architecture for predicting precipitation from satellite radiance data, addressing the challenges of the Weather4cast 2023 competition. PAUNet is a variant of U-Net and Res-Net, designed to effectively capture the large-scale contextual information of multi-band satellite images in visible, water vapor, and infrared bands through encoder convolutional layers with center cropping and attention mechanisms. We built upon the Focal Precipitation Loss including an exponential component (e-FPL), which further enhanced the importance across different precipitation categories, particularly medium and heavy rain. Trained on a substantial dataset from various European regions, PAUNet demonstrates notable accuracy with a higher Critical Success Index (CSI) score than the baseline model in predicting rainfall over multiple time slots. PAUNet's architecture and training methodology showcase improvements in precipitation forecasting, crucial for sectors like emergency services and retail and supply chain management.

{{</citation>}}


## cs.MM (2)



### (155/156) mPLUG-PaperOwl: Scientific Diagram Analysis with the Multimodal Large Language Model (Anwen Hu et al., 2023)

{{<citation>}}

Anwen Hu, Yaya Shi, Haiyang Xu, Jiabo Ye, Qinghao Ye, Ming Yan, Chenliang Li, Qi Qian, Ji Zhang, Fei Huang. (2023)  
**mPLUG-PaperOwl: Scientific Diagram Analysis with the Multimodal Large Language Model**  

---
Primary Category: cs.MM  
Categories: cs-CL, cs-MM, cs.MM  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.18248v1)  

---


**ABSTRACT**  
Recently, the strong text creation ability of Large Language Models(LLMs) has given rise to many tools for assisting paper reading or even writing. However, the weak diagram analysis abilities of LLMs or Multimodal LLMs greatly limit their application scenarios, especially for scientific academic paper writing. In this work, towards a more versatile copilot for academic paper writing, we mainly focus on strengthening the multi-modal diagram analysis ability of Multimodal LLMs. By parsing Latex source files of high-quality papers, we carefully build a multi-modal diagram understanding dataset M-Paper. By aligning diagrams in the paper with related paragraphs, we construct professional diagram analysis samples for training and evaluation. M-Paper is the first dataset to support joint comprehension of multiple scientific diagrams, including figures and tables in the format of images or Latex codes. Besides, to better align the copilot with the user's intention, we introduce the `outline' as the control signal, which could be directly given by the user or revised based on auto-generated ones. Comprehensive experiments with a state-of-the-art Mumtimodal LLM demonstrate that training on our dataset shows stronger scientific diagram understanding performance, including diagram captioning, diagram analysis, and outline recommendation. The dataset, code, and model are available at https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/PaperOwl.

{{</citation>}}


### (156/156) DKiS: Decay weight invertible image steganography with private key (Hang Yang et al., 2023)

{{<citation>}}

Hang Yang, Yitian Xu, Xuhua Liu. (2023)  
**DKiS: Decay weight invertible image steganography with private key**  

---
Primary Category: cs.MM  
Categories: cs-CR, cs-CV, cs-LG, cs-MM, cs.MM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.18243v1)  

---


**ABSTRACT**  
Image steganography, the practice of concealing information within another image, traditionally faces security challenges when its methods become publicly known. To counteract this, we introduce a novel private key-based image steganography technique. This approach ensures the security of hidden information, requiring a corresponding private key for access, irrespective of the public knowledge of the steganography method. We present experimental evidence demonstrating our method's effectiveness, showcasing its real-world applicability. Additionally, we identified a critical challenge in the invertible image steganography process: the transfer of non-essential, or `garbage', information from the secret to the host pipeline. To address this, we introduced the decay weight to control the information transfer, filtering out irrelevant data and enhancing the performance of image steganography. Our code is publicly accessible at https://github.com/yanghangAI/DKiS, and a practical demonstration is available at http://yanghang.site/hidekey.

{{</citation>}}
