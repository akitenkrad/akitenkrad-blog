---
draft: false
title: "arXiv @ 2023.09.02"
date: 2023-09-02
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.02"
    identifier: arxiv_20230902
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [eess.AS (3)](#eessas-3)
- [cs.CV (27)](#cscv-27)
- [cs.CY (2)](#cscy-2)
- [cs.CR (6)](#cscr-6)
- [eess.SY (3)](#eesssy-3)
- [cs.SD (5)](#cssd-5)
- [cs.AI (9)](#csai-9)
- [q-fin.ST (1)](#q-finst-1)
- [cs.SE (7)](#csse-7)
- [cs.LG (17)](#cslg-17)
- [cs.CL (15)](#cscl-15)
- [cs.RO (3)](#csro-3)
- [eess.IV (7)](#eessiv-7)
- [cs.IR (2)](#csir-2)
- [q-bio.QM (1)](#q-bioqm-1)
- [math.OC (1)](#mathoc-1)
- [cs.HC (3)](#cshc-3)

## eess.AS (3)



### (1/112) RepCodec: A Speech Representation Codec for Speech Tokenization (Zhichao Huang et al., 2023)

{{<citation>}}

Zhichao Huang, Chutong Meng, Tom Ko. (2023)  
**RepCodec: A Speech Representation Codec for Speech Tokenization**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.00169v1)  

---


**ABSTRACT**  
With recent rapid growth of large language models (LLMs), discrete speech tokenization has played an important role for injecting speech into LLMs. However, this discretization gives rise to a loss of information, consequently impairing overall performance. To improve the performance of these discrete speech tokens, we present RepCodec, a novel speech representation codec for semantic speech tokenization. In contrast to audio codecs which reconstruct the raw audio, RepCodec learns a vector quantization codebook through reconstructing speech representations from speech encoders like HuBERT or data2vec. Together, the speech encoder, the codec encoder and the vector quantization codebook form a pipeline for converting speech waveforms into semantic tokens. The extensive experiments illustrate that RepCodec, by virtue of its enhanced information retention capacity, significantly outperforms the widely used k-means clustering approach in both speech understanding and generation. Furthermore, this superiority extends across various speech encoders and languages, affirming the robustness of RepCodec. We believe our method can facilitate large language modeling research on speech processing.

{{</citation>}}


### (2/112) PhonMatchNet: Phoneme-Guided Zero-Shot Keyword Spotting for User-Defined Keywords (Yong-Hyeok Lee et al., 2023)

{{<citation>}}

Yong-Hyeok Lee, Namhyun Cho. (2023)  
**PhonMatchNet: Phoneme-Guided Zero-Shot Keyword Spotting for User-Defined Keywords**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess-SP, eess.AS  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.16511v1)  

---


**ABSTRACT**  
This study presents a novel zero-shot user-defined keyword spotting model that utilizes the audio-phoneme relationship of the keyword to improve performance. Unlike the previous approach that estimates at utterance level, we use both utterance and phoneme level information. Our proposed method comprises a two-stream speech encoder architecture, self-attention-based pattern extractor, and phoneme-level detection loss for high performance in various pronunciation environments. Based on experimental results, our proposed model outperforms the baseline model and achieves competitive performance compared with full-shot keyword spotting models. Our proposed model significantly improves the EER and AUC across all datasets, including familiar words, proper nouns, and indistinguishable pronunciations, with an average relative improvement of 67% and 80%, respectively. The implementation code of our proposed model is available at https://github.com/ncsoft/PhonMatchNet.

{{</citation>}}


### (3/112) Supervised Contrastive Learning with Nearest Neighbor Search for Speech Emotion Recognition (Xuechen Wang et al., 2023)

{{<citation>}}

Xuechen Wang, Shiwan Zhao, Yong Qin. (2023)  
**Supervised Contrastive Learning with Nearest Neighbor Search for Speech Emotion Recognition**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Contrastive Learning, Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2308.16485v1)  

---


**ABSTRACT**  
Speech Emotion Recognition (SER) is a challenging task due to limited data and blurred boundaries of certain emotions. In this paper, we present a comprehensive approach to improve the SER performance throughout the model lifecycle, including pre-training, fine-tuning, and inference stages. To address the data scarcity issue, we utilize a pre-trained model, wav2vec2.0. During fine-tuning, we propose a novel loss function that combines cross-entropy loss with supervised contrastive learning loss to improve the model's discriminative ability. This approach increases the inter-class distances and decreases the intra-class distances, mitigating the issue of blurred boundaries. Finally, to leverage the improved distances, we propose an interpolation method at the inference stage that combines the model prediction with the output from a k-nearest neighbors model. Our experiments on IEMOCAP demonstrate that our proposed methods outperform current state-of-the-art results.

{{</citation>}}


## cs.CV (27)



### (4/112) Pose-Graph Attentional Graph Neural Network for Lidar Place Recognition (Milad Ramezani et al., 2023)

{{<citation>}}

Milad Ramezani, Liang Wang, Joshua Knights, Zhibin Li, Pauline Pounds, Peyman Moghadam. (2023)  
**Pose-Graph Attentional Graph Neural Network for Lidar Place Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Attention, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2309.00168v1)  

---


**ABSTRACT**  
This paper proposes a lidar place recognition approach, called P-GAT, to increase the receptive field between point clouds captured over time. Instead of comparing pairs of point clouds, we compare the similarity between sets of point clouds to use the maximum spatial and temporal information between neighbour clouds utilising the concept of pose-graph SLAM. Leveraging intra- and inter-attention and graph neural network, P-GAT relates point clouds captured in nearby locations in Euclidean space and their embeddings in feature space. Experimental results on the large-scale publically available datasets demonstrate the effectiveness of our approach in recognising scenes lacking distinct features and when training and testing environments have different distributions (domain adaptation). Further, an exhaustive comparison with the state-of-the-art shows improvements in performance gains. Code will be available upon acceptance.

{{</citation>}}


### (5/112) Self-supervised Semantic Segmentation: Consistency over Transformation (Sanaz Karimijafarbigloo et al., 2023)

{{<citation>}}

Sanaz Karimijafarbigloo, Reza Azad, Amirhossein Kazerouni, Yury Velichko, Ulas Bagci, Dorit Merhof. (2023)  
**Self-supervised Semantic Segmentation: Consistency over Transformation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.00143v1)  

---


**ABSTRACT**  
Accurate medical image segmentation is of utmost importance for enabling automated clinical decision procedures. However, prevailing supervised deep learning approaches for medical image segmentation encounter significant challenges due to their heavy dependence on extensive labeled training data. To tackle this issue, we propose a novel self-supervised algorithm, \textbf{S$^3$-Net}, which integrates a robust framework based on the proposed Inception Large Kernel Attention (I-LKA) modules. This architectural enhancement makes it possible to comprehensively capture contextual information while preserving local intricacies, thereby enabling precise semantic segmentation. Furthermore, considering that lesions in medical images often exhibit deformations, we leverage deformable convolution as an integral component to effectively capture and delineate lesion deformations for superior object boundary definition. Additionally, our self-supervised strategy emphasizes the acquisition of invariance to affine transformations, which is commonly encountered in medical scenarios. This emphasis on robustness with respect to geometric distortions significantly enhances the model's ability to accurately model and handle such distortions. To enforce spatial consistency and promote the grouping of spatially connected image pixels with similar feature representations, we introduce a spatial consistency loss term. This aids the network in effectively capturing the relationships among neighboring pixels and enhancing the overall segmentation quality. The S$^3$-Net approach iteratively learns pixel-level feature representations for image content clustering in an end-to-end manner. Our experimental results on skin lesion and lung organ segmentation tasks show the superior performance of our method compared to the SOTA approaches. https://github.com/mindflow-institue/SSCT

{{</citation>}}


### (6/112) Distraction-free Embeddings for Robust VQA (Atharvan Dogra et al., 2023)

{{<citation>}}

Atharvan Dogra, Deeksha Varshney, Ashwin Kalyan, Ameet Deshpande, Neeraj Kumar. (2023)  
**Distraction-free Embeddings for Robust VQA**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.00133v1)  

---


**ABSTRACT**  
The generation of effective latent representations and their subsequent refinement to incorporate precise information is an essential prerequisite for Vision-Language Understanding (VLU) tasks such as Video Question Answering (VQA). However, most existing methods for VLU focus on sparsely sampling or fine-graining the input information (e.g., sampling a sparse set of frames or text tokens), or adding external knowledge. We present a novel "DRAX: Distraction Removal and Attended Cross-Alignment" method to rid our cross-modal representations of distractors in the latent space. We do not exclusively confine the perception of any input information from various modalities but instead use an attention-guided distraction removal method to increase focus on task-relevant information in latent embeddings. DRAX also ensures semantic alignment of embeddings during cross-modal fusions. We evaluate our approach on a challenging benchmark (SUTD-TrafficQA dataset), testing the framework's abilities for feature and event queries, temporal relation understanding, forecasting, hypothesis, and causal analysis through extensive experiments.

{{</citation>}}


### (7/112) Beyond Self-Attention: Deformable Large Kernel Attention for Medical Image Segmentation (Reza Azad et al., 2023)

{{<citation>}}

Reza Azad, Leon Niggemeier, Michael Huttemann, Amirhossein Kazerouni, Ehsan Khodapanah Aghdam, Yury Velichko, Ulas Bagci, Dorit Merhof. (2023)  
**Beyond Self-Attention: Deformable Large Kernel Attention for Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Self-Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2309.00121v1)  

---


**ABSTRACT**  
Medical image segmentation has seen significant improvements with transformer models, which excel in grasping far-reaching contexts and global contextual information. However, the increasing computational demands of these models, proportional to the squared token count, limit their depth and resolution capabilities. Most current methods process D volumetric image data slice-by-slice (called pseudo 3D), missing crucial inter-slice information and thus reducing the model's overall performance. To address these challenges, we introduce the concept of \textbf{Deformable Large Kernel Attention (D-LKA Attention)}, a streamlined attention mechanism employing large convolution kernels to fully appreciate volumetric context. This mechanism operates within a receptive field akin to self-attention while sidestepping the computational overhead. Additionally, our proposed attention mechanism benefits from deformable convolutions to flexibly warp the sampling grid, enabling the model to adapt appropriately to diverse data patterns. We designed both 2D and 3D adaptations of the D-LKA Attention, with the latter excelling in cross-depth data understanding. Together, these components shape our novel hierarchical Vision Transformer architecture, the \textit{D-LKA Net}. Evaluations of our model against leading methods on popular medical segmentation datasets (Synapse, NIH Pancreas, and Skin lesion) demonstrate its superior performance. Our code implementation is publicly available at the: https://github.com/mindflow-institue/deformableLKA

{{</citation>}}


### (8/112) Laplacian-Former: Overcoming the Limitations of Vision Transformers in Local Texture Detection (Reza Azad et al., 2023)

{{<citation>}}

Reza Azad, Amirhossein Kazerouni, Babak Azad, Ehsan Khodapanah Aghdam, Yury Velichko, Ulas Bagci, Dorit Merhof. (2023)  
**Laplacian-Former: Overcoming the Limitations of Vision Transformers in Local Texture Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.00108v1)  

---


**ABSTRACT**  
Vision Transformer (ViT) models have demonstrated a breakthrough in a wide range of computer vision tasks. However, compared to the Convolutional Neural Network (CNN) models, it has been observed that the ViT models struggle to capture high-frequency components of images, which can limit their ability to detect local textures and edge information. As abnormalities in human tissue, such as tumors and lesions, may greatly vary in structure, texture, and shape, high-frequency information such as texture is crucial for effective semantic segmentation tasks. To address this limitation in ViT models, we propose a new technique, Laplacian-Former, that enhances the self-attention map by adaptively re-calibrating the frequency information in a Laplacian pyramid. More specifically, our proposed method utilizes a dual attention mechanism via efficient attention and frequency attention while the efficient attention mechanism reduces the complexity of self-attention to linear while producing the same output, selectively intensifying the contribution of shape and texture features. Furthermore, we introduce a novel efficient enhancement multi-scale bridge that effectively transfers spatial information from the encoder to the decoder while preserving the fundamental features. We demonstrate the efficacy of Laplacian-former on multi-organ and skin lesion segmentation tasks with +1.87\% and +0.76\% dice scores compared to SOTA approaches, respectively. Our implementation is publically available at https://github.com/mindflow-institue/Laplacian-Former

{{</citation>}}


### (9/112) Open-Vocabulary Semantic Segmentation via Attribute Decomposition-Aggregation (Chaofan Ma et al., 2023)

{{<citation>}}

Chaofan Ma, Yuhuan Yang, Chen Ju, Fei Zhang, Ya Zhang, Yanfeng Wang. (2023)  
**Open-Vocabulary Semantic Segmentation via Attribute Decomposition-Aggregation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.00096v1)  

---


**ABSTRACT**  
Open-vocabulary semantic segmentation is a challenging task that requires segmenting novel object categories at inference time. Recent works explore vision-language pre-training to handle this task, but suffer from unrealistic assumptions in practical scenarios, i.e., low-quality textual category names. For example, this paradigm assumes that new textual categories will be accurately and completely provided, and exist in lexicons during pre-training. However, exceptions often happen when meet with ambiguity for brief or incomplete names, new words that are not present in the pre-trained lexicons, and difficult-to-describe categories for users. To address these issues, this work proposes a novel decomposition-aggregation framework, inspired by human cognition in understanding new concepts. Specifically, in the decomposition stage, we decouple class names into diverse attribute descriptions to enrich semantic contexts. Two attribute construction strategies are designed: using large language models for common categories, and involving manually labelling for human-invented categories. In the aggregation stage, we group diverse attributes into an integrated global description, to form a discriminative classifier that distinguishes the target object from others. One hierarchical aggregation is further designed to achieve multi-level alignment and deep fusion between vision and text. The final result is obtained by computing the embedding similarity between aggregated attributes and images. To evaluate the effectiveness, we annotate three datasets with attribute descriptions, and conduct extensive experiments and ablation studies. The results show the superior performance of attribute decomposition-aggregation.

{{</citation>}}


### (10/112) FACET: Fairness in Computer Vision Evaluation Benchmark (Laura Gustafson et al., 2023)

{{<citation>}}

Laura Gustafson, Chloe Rolland, Nikhila Ravi, Quentin Duval, Aaron Adcock, Cheng-Yang Fu, Melissa Hall, Candace Ross. (2023)  
**FACET: Fairness in Computer Vision Evaluation Benchmark**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2309.00035v1)  

---


**ABSTRACT**  
Computer vision models have known performance disparities across attributes such as gender and skin tone. This means during tasks such as classification and detection, model performance differs for certain classes based on the demographics of the people in the image. These disparities have been shown to exist, but until now there has not been a unified approach to measure these differences for common use-cases of computer vision models. We present a new benchmark named FACET (FAirness in Computer Vision EvaluaTion), a large, publicly available evaluation set of 32k images for some of the most common vision tasks - image classification, object detection and segmentation. For every image in FACET, we hired expert reviewers to manually annotate person-related attributes such as perceived skin tone and hair type, manually draw bounding boxes and label fine-grained person-related classes such as disk jockey or guitarist. In addition, we use FACET to benchmark state-of-the-art vision models and present a deeper understanding of potential performance disparities and challenges across sensitive demographic attributes. With the exhaustive annotations collected, we probe models using single demographics attributes as well as multiple attributes using an intersectional approach (e.g. hair color and perceived skin tone). Our results show that classification, detection, segmentation, and visual grounding models exhibit performance disparities across demographic attributes and intersections of attributes. These harms suggest that not all people represented in datasets receive fair and equitable treatment in these vision tasks. We hope current and future results using our benchmark will contribute to fairer, more robust vision models. FACET is available publicly at https://facet.metademolab.com/

{{</citation>}}


### (11/112) PointLLM: Empowering Large Language Models to Understand Point Clouds (Runsen Xu et al., 2023)

{{<citation>}}

Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, Dahua Lin. (2023)  
**PointLLM: Empowering Large Language Models to Understand Point Clouds**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.16911v1)  

---


**ABSTRACT**  
The unprecedented advancements in Large Language Models (LLMs) have created a profound impact on natural language processing but are yet to fully embrace the realm of 3D understanding. This paper introduces PointLLM, a preliminary effort to fill this gap, thereby enabling LLMs to understand point clouds and offering a new avenue beyond 2D visual data. PointLLM processes colored object point clouds with human instructions and generates contextually appropriate responses, illustrating its grasp of point clouds and common sense. Specifically, it leverages a point cloud encoder with a powerful LLM to effectively fuse geometric, appearance, and linguistic information. We collect a novel dataset comprising 660K simple and 70K complex point-text instruction pairs to enable a two-stage training strategy: initially aligning latent spaces and subsequently instruction-tuning the unified model. To rigorously evaluate our model's perceptual abilities and its generalization capabilities, we establish two benchmarks: Generative 3D Object Classification and 3D Object Captioning, assessed through three different methods, including human evaluation, GPT-4/ChatGPT evaluation, and traditional metrics. Experiment results show that PointLLM demonstrates superior performance over existing 2D baselines. Remarkably, in human-evaluated object captioning tasks, PointLLM outperforms human annotators in over 50% of the samples. Codes, datasets, and benchmarks are available at https://github.com/OpenRobotLab/PointLLM .

{{</citation>}}


### (12/112) TouchStone: Evaluating Vision-Language Models by Language Models (Shuai Bai et al., 2023)

{{<citation>}}

Shuai Bai, Shusheng Yang, Jinze Bai, Peng Wang, Xingxuan Zhang, Junyang Lin, Xinggang Wang, Chang Zhou, Jingren Zhou. (2023)  
**TouchStone: Evaluating Vision-Language Models by Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.16890v1)  

---


**ABSTRACT**  
Large vision-language models (LVLMs) have recently witnessed rapid advancements, exhibiting a remarkable capacity for perceiving, understanding, and processing visual information by connecting visual receptor with large language models (LLMs). However, current assessments mainly focus on recognizing and reasoning abilities, lacking direct evaluation of conversational skills and neglecting visual storytelling abilities. In this paper, we propose an evaluation method that uses strong LLMs as judges to comprehensively evaluate the various abilities of LVLMs. Firstly, we construct a comprehensive visual dialogue dataset TouchStone, consisting of open-world images and questions, covering five major categories of abilities and 27 subtasks. This dataset not only covers fundamental recognition and comprehension but also extends to literary creation. Secondly, by integrating detailed image annotations we effectively transform the multimodal input content into a form understandable by LLMs. This enables us to employ advanced LLMs for directly evaluating the quality of the multimodal dialogue without requiring human intervention. Through validation, we demonstrate that powerful LVLMs, such as GPT-4, can effectively score dialogue quality by leveraging their textual capabilities alone, aligning with human preferences. We hope our work can serve as a touchstone for LVLMs' evaluation and pave the way for building stronger LVLMs. The evaluation code is available at https://github.com/OFA-Sys/TouchStone.

{{</citation>}}


### (13/112) BTSeg: Barlow Twins Regularization for Domain Adaptation in Semantic Segmentation (Johannes Künzel et al., 2023)

{{<citation>}}

Johannes Künzel, Anna Hilsmann, Peter Eisert. (2023)  
**BTSeg: Barlow Twins Regularization for Domain Adaptation in Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.16819v1)  

---


**ABSTRACT**  
Semantic image segmentation is a critical component in many computer vision systems, such as autonomous driving. In such applications, adverse conditions (heavy rain, night time, snow, extreme lighting) on the one hand pose specific challenges, yet are typically underrepresented in the available datasets. Generating more training data is cumbersome and expensive, and the process itself is error-prone due to the inherent aleatoric uncertainty. To address this challenging problem, we propose BTSeg, which exploits image-level correspondences as weak supervision signal to learn a segmentation model that is agnostic to adverse conditions. To this end, our approach uses the Barlow twins loss from the field of unsupervised learning and treats images taken at the same location but under different adverse conditions as "augmentations" of the same unknown underlying base image. This allows the training of a segmentation model that is robust to appearance changes introduced by different adverse conditions. We evaluate our approach on ACDC and the new challenging ACG benchmark to demonstrate its robustness and generalization capabilities. Our approach performs favorably when compared to the current state-of-the-art methods, while also being simpler to implement and train. The code will be released upon acceptance.

{{</citation>}}


### (14/112) Vision-Based Cranberry Crop Ripening Assessment (Faith Johnson et al., 2023)

{{<citation>}}

Faith Johnson, Jack Lowry, Kristin Dana, Peter Oudemans. (2023)  
**Vision-Based Cranberry Crop Ripening Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.00028v1)  

---


**ABSTRACT**  
Agricultural domains are being transformed by recent advances in AI and computer vision that support quantitative visual evaluation. Using drone imaging, we develop a framework for characterizing the ripening process of cranberry crops. Our method consists of drone-based time-series collection over a cranberry growing season, photometric calibration for albedo recovery from pixels, and berry segmentation with semi-supervised deep learning networks using point-click annotations. By extracting time-series berry albedo measurements, we evaluate four different varieties of cranberries and provide a quantification of their ripening rates. Such quantification has practical implications for 1) assessing real-time overheating risks for cranberry bogs; 2) large scale comparisons of progeny in crop breeding; 3) detecting disease by looking for ripening pattern outliers. This work is the first of its kind in quantitative evaluation of ripening using computer vision methods and has impact beyond cranberry crops including wine grapes, olives, blueberries, and maize.

{{</citation>}}


### (15/112) Parsing is All You Need for Accurate Gait Recognition in the Wild (Jinkai Zheng et al., 2023)

{{<citation>}}

Jinkai Zheng, Xinchen Liu, Shuai Wang, Lihao Wang, Chenggang Yan, Wu Liu. (2023)  
**Parsing is All You Need for Accurate Gait Recognition in the Wild**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2308.16739v1)  

---


**ABSTRACT**  
Binary silhouettes and keypoint-based skeletons have dominated human gait recognition studies for decades since they are easy to extract from video frames. Despite their success in gait recognition for in-the-lab environments, they usually fail in real-world scenarios due to their low information entropy for gait representations. To achieve accurate gait recognition in the wild, this paper presents a novel gait representation, named Gait Parsing Sequence (GPS). GPSs are sequences of fine-grained human segmentation, i.e., human parsing, extracted from video frames, so they have much higher information entropy to encode the shapes and dynamics of fine-grained human parts during walking. Moreover, to effectively explore the capability of the GPS representation, we propose a novel human parsing-based gait recognition framework, named ParsingGait. ParsingGait contains a Convolutional Neural Network (CNN)-based backbone and two light-weighted heads. The first head extracts global semantic features from GPSs, while the other one learns mutual information of part-level features through Graph Convolutional Networks to model the detailed dynamics of human walking. Furthermore, due to the lack of suitable datasets, we build the first parsing-based dataset for gait recognition in the wild, named Gait3D-Parsing, by extending the large-scale and challenging Gait3D dataset. Based on Gait3D-Parsing, we comprehensively evaluate our method and existing gait recognition methods. The experimental results show a significant improvement in accuracy brought by the GPS representation and the superiority of ParsingGait. The code and dataset are available at https://gait3d.github.io/gait3d-parsing-hp .

{{</citation>}}


### (16/112) Terrain Diffusion Network: Climatic-Aware Terrain Generation with Geological Sketch Guidance (Zexin Hu et al., 2023)

{{<citation>}}

Zexin Hu, Kun Hu, Clinton Mo, Lei Pan, Zhiyong Wang. (2023)  
**Terrain Diffusion Network: Climatic-Aware Terrain Generation with Geological Sketch Guidance**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-MM, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2308.16725v1)  

---


**ABSTRACT**  
Sketch-based terrain generation seeks to create realistic landscapes for virtual environments in various applications such as computer games, animation and virtual reality. Recently, deep learning based terrain generation has emerged, notably the ones based on generative adversarial networks (GAN). However, these methods often struggle to fulfill the requirements of flexible user control and maintain generative diversity for realistic terrain. Therefore, we propose a novel diffusion-based method, namely terrain diffusion network (TDN), which actively incorporates user guidance for enhanced controllability, taking into account terrain features like rivers, ridges, basins, and peaks. Instead of adhering to a conventional monolithic denoising process, which often compromises the fidelity of terrain details or the alignment with user control, a multi-level denoising scheme is proposed to generate more realistic terrains by taking into account fine-grained details, particularly those related to climatic patterns influenced by erosion and tectonic activities. Specifically, three terrain synthesisers are designed for structural, intermediate, and fine-grained level denoising purposes, which allow each synthesiser concentrate on a distinct terrain aspect. Moreover, to maximise the efficiency of our TDN, we further introduce terrain and sketch latent spaces for the synthesizers with pre-trained terrain autoencoders. Comprehensive experiments on a new dataset constructed from NASA Topology Images clearly demonstrate the effectiveness of our proposed method, achieving the state-of-the-art performance. Our code and dataset will be publicly available.

{{</citation>}}


### (17/112) ViLTA: Enhancing Vision-Language Pre-training through Textual Augmentation (Weihan Wang et al., 2023)

{{<citation>}}

Weihan Wang, Zhen Yang, Bin Xu, Juanzi Li, Yankui Sun. (2023)  
**ViLTA: Enhancing Vision-Language Pre-training through Textual Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Language Model  
[Paper Link](http://arxiv.org/abs/2308.16689v1)  

---


**ABSTRACT**  
Vision-language pre-training (VLP) methods are blossoming recently, and its crucial goal is to jointly learn visual and textual features via a transformer-based architecture, demonstrating promising improvements on a variety of vision-language tasks. Prior arts usually focus on how to align visual and textual features, but strategies for improving the robustness of model and speeding up model convergence are left insufficiently explored.   In this paper, we propose a novel method ViLTA, comprising of two components to further facilitate the model to learn fine-grained representations among image-text pairs. For Masked Language Modeling (MLM), we propose a cross-distillation method to generate soft labels to enhance the robustness of model, which alleviates the problem of treating synonyms of masked words as negative samples in one-hot labels. For Image-Text Matching (ITM), we leverage the current language encoder to synthesize hard negatives based on the context of language input, encouraging the model to learn high-quality representations by increasing the difficulty of the ITM task. By leveraging the above techniques, our ViLTA can achieve better performance on various vision-language tasks. Extensive experiments on benchmark datasets demonstrate that the effectiveness of ViLTA and its promising potential for vision-language pre-training.

{{</citation>}}


### (18/112) Learning with Multi-modal Gradient Attention for Explainable Composed Image Retrieval (Prateksha Udhayanan et al., 2023)

{{<citation>}}

Prateksha Udhayanan, Srikrishna Karanam, Balaji Vasan Srinivasan. (2023)  
**Learning with Multi-modal Gradient Attention for Explainable Composed Image Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.16649v1)  

---


**ABSTRACT**  
We consider the problem of composed image retrieval that takes an input query consisting of an image and a modification text indicating the desired changes to be made on the image and retrieves images that match these changes. Current state-of-the-art techniques that address this problem use global features for the retrieval, resulting in incorrect localization of the regions of interest to be modified because of the global nature of the features, more so in cases of real-world, in-the-wild images. Since modifier texts usually correspond to specific local changes in an image, it is critical that models learn local features to be able to both localize and retrieve better. To this end, our key novelty is a new gradient-attention-based learning objective that explicitly forces the model to focus on the local regions of interest being modified in each retrieval step. We achieve this by first proposing a new visual image attention computation technique, which we call multi-modal gradient attention (MMGrad) that is explicitly conditioned on the modifier text. We next demonstrate how MMGrad can be incorporated into an end-to-end model training strategy with a new learning objective that explicitly forces these MMGrad attention maps to highlight the correct local regions corresponding to the modifier text. By training retrieval models with this new loss function, we show improved grounding by means of better visual attention maps, leading to better explainability of the models as well as competitive quantitative retrieval performance on standard benchmark datasets.

{{</citation>}}


### (19/112) Semi-Supervised SAR ATR Framework with Transductive Auxiliary Segmentation (Chenwei Wang et al., 2023)

{{<citation>}}

Chenwei Wang, Xiaoyu Liu, Yulin Huang, Siyi Luo, Jifang Pei, Jianyu Yang, Deqing Mao. (2023)  
**Semi-Supervised SAR ATR Framework with Transductive Auxiliary Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.16633v1)  

---


**ABSTRACT**  
Convolutional neural networks (CNNs) have achieved high performance in synthetic aperture radar (SAR) automatic target recognition (ATR). However, the performance of CNNs depends heavily on a large amount of training data. The insufficiency of labeled training SAR images limits the recognition performance and even invalidates some ATR methods. Furthermore, under few labeled training data, many existing CNNs are even ineffective. To address these challenges, we propose a Semi-supervised SAR ATR Framework with transductive Auxiliary Segmentation (SFAS). The proposed framework focuses on exploiting the transductive generalization on available unlabeled samples with an auxiliary loss serving as a regularizer. Through auxiliary segmentation of unlabeled SAR samples and information residue loss (IRL) in training, the framework can employ the proposed training loop process and gradually exploit the information compilation of recognition and segmentation to construct a helpful inductive bias and achieve high performance. Experiments conducted on the MSTAR dataset have shown the effectiveness of our proposed SFAS for few-shot learning. The recognition performance of 94.18\% can be achieved under 20 training samples in each class with simultaneous accurate segmentation results. Facing variances of EOCs, the recognition ratios are higher than 88.00\% when 10 training samples each class.

{{</citation>}}


### (20/112) Any-Size-Diffusion: Toward Efficient Text-Driven Synthesis for Any-Size HD Images (Qingping Zheng et al., 2023)

{{<citation>}}

Qingping Zheng, Yuanfan Guo, Jiankang Deng, Jianhua Han, Ying Li, Songcen Xu, Hang Xu. (2023)  
**Any-Size-Diffusion: Toward Efficient Text-Driven Synthesis for Any-Size HD Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.16582v1)  

---


**ABSTRACT**  
Stable diffusion, a generative model used in text-to-image synthesis, frequently encounters resolution-induced composition problems when generating images of varying sizes. This issue primarily stems from the model being trained on pairs of single-scale images and their corresponding text descriptions. Moreover, direct training on images of unlimited sizes is unfeasible, as it would require an immense number of text-image pairs and entail substantial computational expenses. To overcome these challenges, we propose a two-stage pipeline named Any-Size-Diffusion (ASD), designed to efficiently generate well-composed images of any size, while minimizing the need for high-memory GPU resources. Specifically, the initial stage, dubbed Any Ratio Adaptability Diffusion (ARAD), leverages a selected set of images with a restricted range of ratios to optimize the text-conditional diffusion model, thereby improving its ability to adjust composition to accommodate diverse image sizes. To support the creation of images at any desired size, we further introduce a technique called Fast Seamless Tiled Diffusion (FSTD) at the subsequent stage. This method allows for the rapid enlargement of the ASD output to any high-resolution size, avoiding seaming artifacts or memory overloads. Experimental results on the LAION-COCO and MM-CelebA-HQ benchmarks demonstrate that ASD can produce well-structured images of arbitrary sizes, cutting down the inference time by 2x compared to the traditional tiled algorithm.

{{</citation>}}


### (21/112) CL-MAE: Curriculum-Learned Masked Autoencoders (Neelu Madan et al., 2023)

{{<citation>}}

Neelu Madan, Nicolae-Catalin Ristea, Kamal Nasrollahi, Thomas B. Moeslund, Radu Tudor Ionescu. (2023)  
**CL-MAE: Curriculum-Learned Masked Autoencoders**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.16572v1)  

---


**ABSTRACT**  
Masked image modeling has been demonstrated as a powerful pretext task for generating robust representations that can be effectively generalized across multiple downstream tasks. Typically, this approach involves randomly masking patches (tokens) in input images, with the masking strategy remaining unchanged during training. In this paper, we propose a curriculum learning approach that updates the masking strategy to continually increase the complexity of the self-supervised reconstruction task. We conjecture that, by gradually increasing the task complexity, the model can learn more sophisticated and transferable representations. To facilitate this, we introduce a novel learnable masking module that possesses the capability to generate masks of different complexities, and integrate the proposed module into masked autoencoders (MAE). Our module is jointly trained with the MAE, while adjusting its behavior during training, transitioning from a partner to the MAE (optimizing the same reconstruction loss) to an adversary (optimizing the opposite loss), while passing through a neutral state. The transition between these behaviors is smooth, being regulated by a factor that is multiplied with the reconstruction loss of the masking module. The resulting training procedure generates an easy-to-hard curriculum. We train our Curriculum-Learned Masked Autoencoder (CL-MAE) on ImageNet and show that it exhibits superior representation learning capabilities compared to MAE. The empirical results on five downstream tasks confirm our conjecture, demonstrating that curriculum learning can be successfully used to self-supervise masked autoencoders.

{{</citation>}}


### (22/112) Prompt-enhanced Hierarchical Transformer Elevating Cardiopulmonary Resuscitation Instruction via Temporal Action Segmentation (Yang Liu et al., 2023)

{{<citation>}}

Yang Liu, Xiaoyun Zhong, Shiyao Zhai, Zhicheng Du, Zhenyuan Gao, Qiming Huang, Canyang Zhang, Bin Jiang, Vijay Kumar Pandey, Sanyang Han, Runming Wang, Yuxing Han, Peiwu Qin. (2023)  
**Prompt-enhanced Hierarchical Transformer Elevating Cardiopulmonary Resuscitation Instruction via Temporal Action Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.16552v1)  

---


**ABSTRACT**  
The vast majority of people who suffer unexpected cardiac arrest are performed cardiopulmonary resuscitation (CPR) by passersby in a desperate attempt to restore life, but endeavors turn out to be fruitless on account of disqualification. Fortunately, many pieces of research manifest that disciplined training will help to elevate the success rate of resuscitation, which constantly desires a seamless combination of novel techniques to yield further advancement. To this end, we collect a custom CPR video dataset in which trainees make efforts to behave resuscitation on mannequins independently in adherence to approved guidelines, thereby devising an auxiliary toolbox to assist supervision and rectification of intermediate potential issues via modern deep learning methodologies. Our research empirically views this problem as a temporal action segmentation (TAS) task in computer vision, which aims to segment an untrimmed video at a frame-wise level. Here, we propose a Prompt-enhanced hierarchical Transformer (PhiTrans) that integrates three indispensable modules, including a textual prompt-based Video Features Extractor (VFE), a transformer-based Action Segmentation Executor (ASE), and a regression-based Prediction Refinement Calibrator (PRC). The backbone of the model preferentially derives from applications in three approved public datasets (GTEA, 50Salads, and Breakfast) collected for TAS tasks, which accounts for the excavation of the segmentation pipeline on the CPR dataset. In general, we unprecedentedly probe into a feasible pipeline that genuinely elevates the CPR instruction qualification via action segmentation in conjunction with cutting-edge deep learning techniques. Associated experiments advocate our implementation with multiple metrics surpassing 91.0%.

{{</citation>}}


### (23/112) SA6D: Self-Adaptive Few-Shot 6D Pose Estimator for Novel and Occluded Objects (Ning Gao et al., 2023)

{{<citation>}}

Ning Gao, Ngo Anh Vien, Hanna Ziesche, Gerhard Neumann. (2023)  
**SA6D: Self-Adaptive Few-Shot 6D Pose Estimator for Novel and Occluded Objects**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.16528v1)  

---


**ABSTRACT**  
To enable meaningful robotic manipulation of objects in the real-world, 6D pose estimation is one of the critical aspects. Most existing approaches have difficulties to extend predictions to scenarios where novel object instances are continuously introduced, especially with heavy occlusions. In this work, we propose a few-shot pose estimation (FSPE) approach called SA6D, which uses a self-adaptive segmentation module to identify the novel target object and construct a point cloud model of the target object using only a small number of cluttered reference images. Unlike existing methods, SA6D does not require object-centric reference images or any additional object information, making it a more generalizable and scalable solution across categories. We evaluate SA6D on real-world tabletop object datasets and demonstrate that SA6D outperforms existing FSPE methods, particularly in cluttered scenes with occlusions, while requiring fewer reference images.

{{</citation>}}


### (24/112) Unsupervised Recognition of Unknown Objects for Open-World Object Detection (Ruohuan Fang et al., 2023)

{{<citation>}}

Ruohuan Fang, Guansong Pang, Lei Zhou, Xiao Bai, Jin Zheng. (2023)  
**Unsupervised Recognition of Unknown Objects for Open-World Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.16527v1)  

---


**ABSTRACT**  
Open-World Object Detection (OWOD) extends object detection problem to a realistic and dynamic scenario, where a detection model is required to be capable of detecting both known and unknown objects and incrementally learning newly introduced knowledge. Current OWOD models, such as ORE and OW-DETR, focus on pseudo-labeling regions with high objectness scores as unknowns, whose performance relies heavily on the supervision of known objects. While they can detect the unknowns that exhibit similar features to the known objects, they suffer from a severe label bias problem that they tend to detect all regions (including unknown object regions) that are dissimilar to the known objects as part of the background. To eliminate the label bias, this paper proposes a novel approach that learns an unsupervised discriminative model to recognize true unknown objects from raw pseudo labels generated by unsupervised region proposal methods. The resulting model can be further refined by a classification-free self-training method which iteratively extends pseudo unknown objects to the unlabeled regions. Experimental results show that our method 1) significantly outperforms the prior SOTA in detecting unknown objects while maintaining competitive performance of detecting known object classes on the MS COCO dataset, and 2) achieves better generalization ability on the LVIS and Objects365 datasets.

{{</citation>}}


### (25/112) MS23D: A 3D Object Detection Method Using Multi-Scale Semantic Feature Points to Construct 3D Feature Layers (Yongxin Shao et al., 2023)

{{<citation>}}

Yongxin Shao, Aihong Tan, Tianhong Yan, Zhetao Sun. (2023)  
**MS23D: A 3D Object Detection Method Using Multi-Scale Semantic Feature Points to Construct 3D Feature Layers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.16518v1)  

---


**ABSTRACT**  
Lidar point clouds, as a type of data with accurate distance perception, can effectively represent the motion and posture of objects in three-dimensional space. However, the sparsity and disorderliness of point clouds make it challenging to extract features directly from them. Many studies have addressed this issue by transforming point clouds into regular voxel representations. However, these methods often lead to the loss of fine-grained local feature information due to downsampling. Moreover, the sparsity of point clouds poses difficulties in efficiently aggregating features in 3D feature layers using voxel-based two-stage methods. To address these issues, this paper proposes a two-stage 3D detection framework called MS$^{2}$3D. In MS$^{2}$3D, we utilize small-sized voxels to extract fine-grained local features and large-sized voxels to capture long-range local features. Additionally, we propose a method for constructing 3D feature layers using multi-scale semantic feature points, enabling the transformation of sparse 3D feature layers into more compact representations. Furthermore, we compute the offset between feature points in the 3D feature layers and the centroid of objects, aiming to bring them as close as possible to the object's center. It significantly enhances the efficiency of feature aggregation. To validate the effectiveness of our method, we evaluated our method on the KITTI dataset and ONCE dataset together.

{{</citation>}}


### (26/112) Unsupervised discovery of Interpretable Visual Concepts (Caroline Mazini Rodrigues et al., 2023)

{{<citation>}}

Caroline Mazini Rodrigues, Nicolas Boutry, Laurent Najman. (2023)  
**Unsupervised discovery of Interpretable Visual Concepts**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.00018v1)  

---


**ABSTRACT**  
Providing interpretability of deep-learning models to non-experts, while fundamental for a responsible real-world usage, is challenging. Attribution maps from xAI techniques, such as Integrated Gradients, are a typical example of a visualization technique containing a high level of information, but with difficult interpretation. In this paper, we propose two methods, Maximum Activation Groups Extraction (MAGE) and Multiscale Interpretable Visualization (Ms-IV), to explain the model's decision, enhancing global interpretability. MAGE finds, for a given CNN, combinations of features which, globally, form a semantic meaning, that we call concepts. We group these similar feature patterns by clustering in ``concepts'', that we visualize through Ms-IV. This last method is inspired by Occlusion and Sensitivity analysis (incorporating causality), and uses a novel metric, called Class-aware Order Correlation (CaOC), to globally evaluate the most important image regions according to the model's decision space. We compare our approach to xAI methods such as LIME and Integrated Gradients. Experimental results evince the Ms-IV higher localization and faithfulness values. Finally, qualitative evaluation of combined MAGE and Ms-IV demonstrate humans' ability to agree, based on the visualization, on the decision of clusters' concepts; and, to detect, among a given set of networks, the existence of bias.

{{</citation>}}


### (27/112) Latent Painter (Shih-Chieh Su, 2023)

{{<citation>}}

Shih-Chieh Su. (2023)  
**Latent Painter**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.16490v1)  

---


**ABSTRACT**  
Latent diffusers revolutionized the generative AI and inspired creative art. When denoising the latent, the predicted original image at each step collectively animates the formation. However, the animation is limited by the denoising nature of the diffuser, and only renders a sharpening process. This work presents Latent Painter, which uses the latent as the canvas, and the diffuser predictions as the plan, to generate painting animation. Latent Painter also transits one generated image to another, which can happen between images from two different sets of checkpoints.

{{</citation>}}


### (28/112) Sparkles: Unlocking Chats Across Multiple Images for Multimodal Instruction-Following Models (Yupan Huang et al., 2023)

{{<citation>}}

Yupan Huang, Zaiqiao Meng, Fangyu Liu, Yixuan Su, Nigel Collier, Yutong Lu. (2023)  
**Sparkles: Unlocking Chats Across Multiple Images for Multimodal Instruction-Following Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Dialog, Dialogue, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.16463v1)  

---


**ABSTRACT**  
Large language models exhibit enhanced zero-shot performance on various tasks when fine-tuned with instruction-following data. Multimodal instruction-following models extend these capabilities by integrating both text and images. However, existing models such as MiniGPT-4 face challenges in maintaining dialogue coherence in scenarios involving multiple images. A primary reason is the lack of a specialized dataset for this critical application. To bridge these gaps, we present SparklesChat, a multimodal instruction-following model for open-ended dialogues across multiple images. To support the training, we introduce SparklesDialogue, the first machine-generated dialogue dataset tailored for word-level interleaved multi-image and text interactions. Furthermore, we construct SparklesEval, a GPT-assisted benchmark for quantitatively assessing a model's conversational competence across multiple images and dialogue turns. Our experiments validate the effectiveness of SparklesChat in understanding and reasoning across multiple images and dialogue turns. Specifically, SparklesChat outperformed MiniGPT-4 on established vision-and-language benchmarks, including the BISON binary image selection task and the NLVR2 visual reasoning task. Moreover, SparklesChat scored 8.56 out of 10 on SparklesEval, substantially exceeding MiniGPT-4's score of 3.91 and nearing GPT-4's score of 9.26. Qualitative evaluations further demonstrate SparklesChat's generality in handling real-world applications. All resources will be available at https://github.com/HYPJUDY/Sparkles.

{{</citation>}}


### (29/112) Njobvu-AI: An open-source tool for collaborative image labeling and implementation of computer vision models (Jonathan S. Koning et al., 2023)

{{<citation>}}

Jonathan S. Koning, Ashwin Subramanian, Mazen Alotaibi, Cara L. Appel, Christopher M. Sullivan, Thon Chao, Lisa Truong, Robyn L. Tanguay, Pankaj Jaiswal, Taal Levi, Damon B. Lesmeister. (2023)  
**Njobvu-AI: An open-source tool for collaborative image labeling and implementation of computer vision models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.16435v1)  

---


**ABSTRACT**  
Practitioners interested in using computer vision models lack user-friendly and open-source software that combines features to label training data, allow multiple users, train new algorithms, review output, and implement new models. Labeling training data, such as images, is a key step to developing accurate object detection algorithms using computer vision. This step is often not compatible with many cloud-based services for marking or labeling image and video data due to limited internet bandwidth in many regions of the world. Desktop tools are useful for groups working in remote locations, but users often do not have the capability to combine projects developed locally by multiple collaborators. Furthermore, many tools offer features for labeling data or using pre-trained models for classification, but few allow researchers to combine these steps to create and apply custom models. Free, open-source, and user-friendly software that offers a full suite of features (e.g., ability to work locally and online, and train custom models) is desirable to field researchers and conservationists that may have limited coding skills. We developed Njobvu-AI, a free, open-source tool that can be run on both desktop and server hardware using Node.js, allowing users to label data, combine projects for collaboration and review, train custom algorithms, and implement new computer vision models. The name Njobvu-AI (pronounced N-joh-voo AI), incorporating the Chichewa word for elephant, is inspired by a wildlife monitoring program in Malawi that was a primary impetus for the development of this tool and references similarities between the powerful memory of elephants and properties of computer vision models.

{{</citation>}}


### (30/112) Separate and Locate: Rethink the Text in Text-based Visual Question Answering (Chengyang Fang et al., 2023)

{{<citation>}}

Chengyang Fang, Jiangnan Li, Liang Li, Can Ma, Dayong Hu. (2023)  
**Separate and Locate: Rethink the Text in Text-based Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: OCR, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.16383v1)  

---


**ABSTRACT**  
Text-based Visual Question Answering (TextVQA) aims at answering questions about the text in images. Most works in this field focus on designing network structures or pre-training tasks. All these methods list the OCR texts in reading order (from left to right and top to bottom) to form a sequence, which is treated as a natural language ``sentence''. However, they ignore the fact that most OCR words in the TextVQA task do not have a semantical contextual relationship. In addition, these approaches use 1-D position embedding to construct the spatial relation between OCR tokens sequentially, which is not reasonable. The 1-D position embedding can only represent the left-right sequence relationship between words in a sentence, but not the complex spatial position relationship. To tackle these problems, we propose a novel method named Separate and Locate (SaL) that explores text contextual cues and designs spatial position embedding to construct spatial relations between OCR texts. Specifically, we propose a Text Semantic Separate (TSS) module that helps the model recognize whether words have semantic contextual relations. Then, we introduce a Spatial Circle Position (SCP) module that helps the model better construct and reason the spatial position relationships between OCR texts. Our SaL model outperforms the baseline model by 4.44% and 3.96% accuracy on TextVQA and ST-VQA datasets. Compared with the pre-training state-of-the-art method pre-trained on 64 million pre-training samples, our method, without any pre-training tasks, still achieves 2.68% and 2.52% accuracy improvement on TextVQA and ST-VQA. Our code and models will be released at https://github.com/fangbufang/SaL.

{{</citation>}}


## cs.CY (2)



### (31/112) A Task-Interdependency Model of Complex Collaboration Towards Human-Centered Crowd Work (David T. Lee et al., 2023)

{{<citation>}}

David T. Lee, Christos A. Makridis. (2023)  
**A Task-Interdependency Model of Complex Collaboration Towards Human-Centered Crowd Work**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-MA, cs-SI, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.00160v1)  

---


**ABSTRACT**  
Models of crowdsourcing and human computation often assume that individuals independently carry out small, modular tasks. However, while these models have successfully shown how crowds can accomplish significant objectives, they can inadvertently advance a less than human view of crowd workers and fail to capture the unique human capacity for complex collaborative work. We present a model centered on interdependencies -- a phenomenon well understood to be at the core of collaboration -- that allows one to formally reason about diverse challenges to complex collaboration. Our model represents tasks as an interdependent collection of subtasks, formalized as a task graph. We use it to explain challenges to scaling complex collaborative work, underscore the importance of expert workers, reveal critical factors for learning on the job, and explore the relationship between coordination intensity and occupational wages. Using data from O*NET and the Bureau of Labor Statistics, we introduce an index of occupational coordination intensity to validate our theoretical predictions. We present preliminary evidence that occupations with greater coordination intensity are less exposed to displacement by AI, and discuss opportunities for models that emphasize the collaborative capacities of human workers, bridge models of crowd work and traditional work, and promote AI in roles augmenting human collaboration.

{{</citation>}}


### (32/112) Ethical Framework for Harnessing the Power of AI in Healthcare and Beyond (Sidra Nasir et al., 2023)

{{<citation>}}

Sidra Nasir, Rizwan Ahmed Khan, Samita Bai. (2023)  
**Ethical Framework for Harnessing the Power of AI in Healthcare and Beyond**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.00064v1)  

---


**ABSTRACT**  
In the past decade, the deployment of deep learning (Artificial Intelligence (AI)) methods has become pervasive across a spectrum of real-world applications, often in safety-critical contexts. This comprehensive research article rigorously investigates the ethical dimensions intricately linked to the rapid evolution of AI technologies, with a particular focus on the healthcare domain. Delving deeply, it explores a multitude of facets including transparency, adept data management, human oversight, educational imperatives, and international collaboration within the realm of AI advancement. Central to this article is the proposition of a conscientious AI framework, meticulously crafted to accentuate values of transparency, equity, answerability, and a human-centric orientation. The second contribution of the article is the in-depth and thorough discussion of the limitations inherent to AI systems. It astutely identifies potential biases and the intricate challenges of navigating multifaceted contexts. Lastly, the article unequivocally accentuates the pressing need for globally standardized AI ethics principles and frameworks. Simultaneously, it aptly illustrates the adaptability of the ethical framework proposed herein, positioned skillfully to surmount emergent challenges.

{{</citation>}}


## cs.CR (6)



### (33/112) LLM in the Shell: Generative Honeypots (Muris Sladić et al., 2023)

{{<citation>}}

Muris Sladić, Veronica Valeros, Carlos Catania, Sebastian Garcia. (2023)  
**LLM in the Shell: Generative Honeypots**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.00155v1)  

---


**ABSTRACT**  
Honeypots are essential tools in cybersecurity. However, most of them (even the high-interaction ones) lack the required realism to engage and fool human attackers. This limitation makes them easily discernible, hindering their effectiveness. This work introduces a novel method to create dynamic and realistic software honeypots based on Large Language Models. Preliminary results indicate that LLMs can create credible and dynamic honeypots capable of addressing important limitations of previous honeypots, such as deterministic responses, lack of adaptability, etc. We evaluated the realism of each command by conducting an experiment with human attackers who needed to say if the answer from the honeypot was fake or not. Our proposed honeypot, called shelLM, reached an accuracy rate of 0.92.

{{</citation>}}


### (34/112) Study of Zero-Knowledge protocols and Elliptic Curve Cryptography and their implementation in Smart Card environments using Java Card (Carlos Andres Agudelo Serna, 2023)

{{<citation>}}

Carlos Andres Agudelo Serna. (2023)  
**Study of Zero-Knowledge protocols and Elliptic Curve Cryptography and their implementation in Smart Card environments using Java Card**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Smart Car  
[Paper Link](http://arxiv.org/abs/2308.16666v1)  

---


**ABSTRACT**  
This paper studies the problem of Zero-Knowledge Protocol (ZKP) and elliptic curve cryptographic implementation in a computationally limited environment, such as, the smart cards, using Java Card. Besides that, it is explained how the zero-knowledge protocol was selected to implement it on a smart card and how the benchmarking was conducted to select this protocol. The paper also shows a theoretical development to implement the ZKP protocol using elliptic curve cryptography. Keywords: Authentication; Zero-knowledge; Cryptography; Elliptic Curve; Java card; Smart cards

{{</citation>}}


### (35/112) The Power of MEME: Adversarial Malware Creation with Model-Based Reinforcement Learning (Maria Rigaki et al., 2023)

{{<citation>}}

Maria Rigaki, Sebastian Garcia. (2023)  
**The Power of MEME: Adversarial Malware Creation with Model-Based Reinforcement Learning**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.16562v1)  

---


**ABSTRACT**  
Due to the proliferation of malware, defenders are increasingly turning to automation and machine learning as part of the malware detection tool-chain. However, machine learning models are susceptible to adversarial attacks, requiring the testing of model and product robustness. Meanwhile, attackers also seek to automate malware generation and evasion of antivirus systems, and defenders try to gain insight into their methods. This work proposes a new algorithm that combines Malware Evasion and Model Extraction (MEME) attacks. MEME uses model-based reinforcement learning to adversarially modify Windows executable binary samples while simultaneously training a surrogate model with a high agreement with the target model to evade. To evaluate this method, we compare it with two state-of-the-art attacks in adversarial malware creation, using three well-known published models and one antivirus product as targets. Results show that MEME outperforms the state-of-the-art methods in terms of evasion capabilities in almost all cases, producing evasive malware with an evasion rate in the range of 32-73%. It also produces surrogate models with a prediction label agreement with the respective target models between 97-99%. The surrogate could be used to fine-tune and improve the evasion rate in the future.

{{</citation>}}


### (36/112) Privacy-Preserving Medical Image Classification through Deep Learning and Matrix Decomposition (Andreea Bianca Popescu et al., 2023)

{{<citation>}}

Andreea Bianca Popescu, Cosmin Ioan Nita, Ioana Antonia Taca, Anamaria Vizitiu, Lucian Mihai Itu. (2023)  
**Privacy-Preserving Medical Image Classification through Deep Learning and Matrix Decomposition**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs.CR, eess-IV  
Keywords: AI, Image Classification  
[Paper Link](http://arxiv.org/abs/2308.16530v1)  

---


**ABSTRACT**  
Deep learning (DL)-based solutions have been extensively researched in the medical domain in recent years, enhancing the efficacy of diagnosis, planning, and treatment. Since the usage of health-related data is strictly regulated, processing medical records outside the hospital environment for developing and using DL models demands robust data protection measures. At the same time, it can be challenging to guarantee that a DL solution delivers a minimum level of performance when being trained on secured data, without being specifically designed for the given task. Our approach uses singular value decomposition (SVD) and principal component analysis (PCA) to obfuscate the medical images before employing them in the DL analysis. The capability of DL algorithms to extract relevant information from secured data is assessed on a task of angiographic view classification based on obfuscated frames. The security level is probed by simulated artificial intelligence (AI)-based reconstruction attacks, considering two threat actors with different prior knowledge of the targeted data. The degree of privacy is quantitatively measured using similarity indices. Although a trade-off between privacy and accuracy should be considered, the proposed technique allows for training the angiographic view classifier exclusively on secured data with satisfactory performance and with no computational overhead, model adaptation, or hyperparameter tuning. While the obfuscated medical image content is well protected against human perception, the hypothetical reconstruction attack proved that it is also difficult to recover the complete information of the original frames.

{{</citation>}}


### (37/112) Listen to Minority: Encrypted Traffic Classification for Class Imbalance with Contrastive Pre-Training (Xiang Li et al., 2023)

{{<citation>}}

Xiang Li, Juncheng Guo, Qige Song, Jiang Xie, Yafei Sang, Shuyuan Zhao, Yongzheng Zhang. (2023)  
**Listen to Minority: Encrypted Traffic Classification for Class Imbalance with Contrastive Pre-Training**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.16453v1)  

---


**ABSTRACT**  
Mobile Internet has profoundly reshaped modern lifestyles in various aspects. Encrypted Traffic Classification (ETC) naturally plays a crucial role in managing mobile Internet, especially with the explosive growth of mobile apps using encrypted communication. Despite some existing learning-based ETC methods showing promising results, three-fold limitations still remain in real-world network environments, 1) label bias caused by traffic class imbalance, 2) traffic homogeneity caused by component sharing, and 3) training with reliance on sufficient labeled traffic. None of the existing ETC methods can address all these limitations. In this paper, we propose a novel Pre-trAining Semi-Supervised ETC framework, dubbed PASS. Our key insight is to resample the original train dataset and perform contrastive pre-training without using individual app labels directly to avoid label bias issues caused by class imbalance, while obtaining a robust feature representation to differentiate overlapping homogeneous traffic by pulling positive traffic pairs closer and pushing negative pairs away. Meanwhile, PASS designs a semi-supervised optimization strategy based on pseudo-label iteration and dynamic loss weighting algorithms in order to effectively utilize massive unlabeled traffic data and alleviate manual train dataset annotation workload. PASS outperforms state-of-the-art ETC methods and generic sampling approaches on four public datasets with significant class imbalance and traffic homogeneity, remarkably pushing the F1 of Cross-Platform215 with 1.31%, ISCX-17 with 9.12%. Furthermore, we validate the generality of the contrastive pre-training and pseudo-label iteration components of PASS, which can adaptively benefit ETC methods with diverse feature extractors.

{{</citation>}}


### (38/112) Efficient Additions and Montgomery Reductions of Large Integers for SIMD (Pengchang Ren et al., 2023)

{{<citation>}}

Pengchang Ren, Reiji Suda, Vorapong Suppakitpaisarn. (2023)  
**Efficient Additions and Montgomery Reductions of Large Integers for SIMD**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2308.16432v1)  

---


**ABSTRACT**  
This paper presents efficient algorithms, designed to leverage SIMD for performing Montgomery reductions and additions on integers larger than 512 bits. The existing algorithms encounter inefficiencies when parallelized using SIMD due to extensive dependencies in both operations, particularly noticeable in costly operations like ARM's SVE. To mitigate this problem, a novel addition algorithm is introduced that simulates the addition of large integers using a smaller addition, quickly producing the same set of carries. These carries are then utilized to perform parallel additions on large integers. For Montgomery reductions, serial multiplications are replaced with precomputations that can be effectively calculated using SIMD extensions. Experimental evidence demonstrates that these proposed algorithms substantially enhance the performance of state-of-the-art implementations of several post-quantum cryptography algorithms. Notably, they deliver a 30% speed-up from the latest CTIDH implementation, an 11% speed-up from the latest CSIDH implementation in AVX-512 processors, and a 7% speed-up from Microsoft's standard PQCrypto-SIDH for SIKEp503 on A64FX.

{{</citation>}}


## eess.SY (3)



### (39/112) Multi Agent DeepRL based Joint Power and Subchannel Allocation in IAB networks (Lakshya Jagadish et al., 2023)

{{<citation>}}

Lakshya Jagadish, Banashree Sarma, R. Manivasakan. (2023)  
**Multi Agent DeepRL based Joint Power and Subchannel Allocation in IAB networks**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.00144v1)  

---


**ABSTRACT**  
Integrated Access and Backhauling (IAB) is a viable approach for meeting the unprecedented need for higher data rates of future generations, acting as a cost-effective alternative to dense fiber-wired links. The design of such networks with constraints usually results in an optimization problem of non-convex and combinatorial nature. Under those situations, it is challenging to obtain an optimal strategy for the joint Subchannel Allocation and Power Allocation (SAPA) problem. In this paper, we develop a multi-agent Deep Reinforcement Learning (DeepRL) based framework for joint optimization of power and subchannel allocation in an IAB network to maximize the downlink data rate. SAPA using DDQN (Double Deep Q-Learning Network) can handle computationally expensive problems with huge action spaces associated with multiple users and nodes. Unlike the conventional methods such as game theory, fractional programming, and convex optimization, which in practice demand more and more accurate network information, the multi-agent DeepRL approach requires less environment network information. Simulation results show the proposed scheme's promising performance when compared with baseline (Deep Q-Learning Network and Random) schemes.

{{</citation>}}


### (40/112) Security Allocation in Networked Control Systems under Stealthy Attacks (Anh Tung Nguyen et al., 2023)

{{<citation>}}

Anh Tung Nguyen, André M. H. Teixeira, Alexander Medvedev. (2023)  
**Security Allocation in Networked Control Systems under Stealthy Attacks**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.16639v1)  

---


**ABSTRACT**  
This paper considers the problem of security allocation in a networked control system under stealthy attacks in which the system is comprised of interconnected subsystems represented by vertices. A malicious adversary selects a single vertex on which to conduct a stealthy data injection attack to maximally disrupt the local performance while remaining undetected. On the other hand, a defender selects several vertices on which to allocate defense resources against the adversary. First, the objectives of the adversary and the defender with uncertain targets are formulated in probabilistic ways, resulting in an expected worst-case impact of stealthy attacks. Next, we provide a graph-theoretic necessary and sufficient condition under which the cost for the defender and the expected worst-case impact of stealthy attacks are bounded. This condition enables the defender to restrict the admissible actions to a subset of available vertex sets. Then, we cast the problem of security allocation in a Stackelberg game-theoretic framework. Finally, the contribution of this paper is highlighted by utilizing the proposed admissible actions of the defender in the context of large-scale networks. A numerical example of a 50-vertex networked control system is presented to validate the obtained results.

{{</citation>}}


### (41/112) Physics-Based Trajectory Design for Cellular-Connected UAV in Rainy Environments Based on Deep Reinforcement Learning (Hao Qin et al., 2023)

{{<citation>}}

Hao Qin, Zhaozhou Wu, Xingqi Zhang. (2023)  
**Physics-Based Trajectory Design for Cellular-Connected UAV in Rainy Environments Based on Deep Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.00017v1)  

---


**ABSTRACT**  
Cellular-connected unmanned aerial vehicles (UAVs) have gained increasing attention due to their potential to enhance conventional UAV capabilities by leveraging existing cellular infrastructure for reliable communications between UAVs and base stations. They have been used for various applications, including weather forecasting and search and rescue operations. However, under extreme weather conditions such as rainfall, it is challenging for the trajectory design of cellular UAVs, due to weak coverage regions in the sky, limitations of UAV flying time, and signal attenuation caused by raindrops. To this end, this paper proposes a physics-based trajectory design approach for cellular-connected UAVs in rainy environments. A physics-based electromagnetic simulator is utilized to take into account detailed environment information and the impact of rain on radio wave propagation. The trajectory optimization problem is formulated to jointly consider UAV flying time and signal-to-interference ratio, and is solved through a Markov decision process using deep reinforcement learning algorithms based on multi-step learning and double Q-learning. Optimal UAV trajectories are compared in examples with homogeneous atmosphere medium and rain medium. Additionally, a thorough study of varying weather conditions on trajectory design is provided, and the impact of weight coefficients in the problem formulation is discussed. The proposed approach has demonstrated great potential for UAV trajectory design under rainy weather conditions.

{{</citation>}}


## cs.SD (5)



### (42/112) Improving vision-inspired keyword spotting using dynamic module skipping in streaming conformer encoder (Alexandre Bittar et al., 2023)

{{<citation>}}

Alexandre Bittar, Paul Dixon, Mohammad Samragh, Kumari Nishu, Devang Naik. (2023)  
**Improving vision-inspired keyword spotting using dynamic module skipping in streaming conformer encoder**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2309.00140v1)  

---


**ABSTRACT**  
Using a vision-inspired keyword spotting framework, we propose an architecture with input-dependent dynamic depth capable of processing streaming audio. Specifically, we extend a conformer encoder with trainable binary gates that allow us to dynamically skip network modules according to the input audio. Our approach improves detection and localization accuracy on continuous speech using Librispeech top-1000 most frequent words while maintaining a small memory footprint. The inclusion of gates also reduces the average amount of processing without affecting the overall performance. These benefits are shown to be even more pronounced using the Google speech commands dataset placed over background noise where up to 97% of the processing is skipped on non-speech inputs, therefore making our method particularly interesting for an always-on keyword spotter.

{{</citation>}}


### (43/112) QS-TTS: Towards Semi-Supervised Text-to-Speech Synthesis via Vector-Quantized Self-Supervised Speech Representation Learning (Haohan Guo et al., 2023)

{{<citation>}}

Haohan Guo, Fenglong Xie, Jiawen Kang, Yujia Xiao, Xixin Wu, Helen Meng. (2023)  
**QS-TTS: Towards Semi-Supervised Text-to-Speech Synthesis via Vector-Quantized Self-Supervised Speech Representation Learning**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Representation Learning, Self-Supervised, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2309.00126v1)  

---


**ABSTRACT**  
This paper proposes a novel semi-supervised TTS framework, QS-TTS, to improve TTS quality with lower supervised data requirements via Vector-Quantized Self-Supervised Speech Representation Learning (VQ-S3RL) utilizing more unlabeled speech audio. This framework comprises two VQ-S3R learners: first, the principal learner aims to provide a generative Multi-Stage Multi-Codebook (MSMC) VQ-S3R via the MSMC-VQ-GAN combined with the contrastive S3RL, while decoding it back to the high-quality audio; then, the associate learner further abstracts the MSMC representation into a highly-compact VQ representation through a VQ-VAE. These two generative VQ-S3R learners provide profitable speech representations and pre-trained models for TTS, significantly improving synthesis quality with the lower requirement for supervised data. QS-TTS is evaluated comprehensively under various scenarios via subjective and objective tests in experiments. The results powerfully demonstrate the superior performance of QS-TTS, winning the highest MOS over supervised or semi-supervised baseline TTS approaches, especially in low-resource scenarios. Moreover, comparing various speech representations and transfer learning methods in TTS further validates the notable improvement of the proposed VQ-S3RL to TTS, showing the best audio quality and intelligibility metrics. The trend of slower decay in the synthesis quality of QS-TTS with decreasing supervised data further highlights its lower requirements for supervised data, indicating its great potential in low-resource scenarios.

{{</citation>}}


### (44/112) Towards Improving the Expressiveness of Singing Voice Synthesis with BERT Derived Semantic Information (Shaohuan Zhou et al., 2023)

{{<citation>}}

Shaohuan Zhou, Shun Lei, Weiya You, Deyi Tuo, Yuren You, Zhiyong Wu, Shiyin Kang, Helen Meng. (2023)  
**Towards Improving the Expressiveness of Singing Voice Synthesis with BERT Derived Semantic Information**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.16836v1)  

---


**ABSTRACT**  
This paper presents an end-to-end high-quality singing voice synthesis (SVS) system that uses bidirectional encoder representation from Transformers (BERT) derived semantic embeddings to improve the expressiveness of the synthesized singing voice. Based on the main architecture of recently proposed VISinger, we put forward several specific designs for expressive singing voice synthesis. First, different from the previous SVS models, we use text representation of lyrics extracted from pre-trained BERT as additional input to the model. The representation contains information about semantics of the lyrics, which could help SVS system produce more expressive and natural voice. Second, we further introduce an energy predictor to stabilize the synthesized voice and model the wider range of energy variations that also contribute to the expressiveness of singing voice. Last but not the least, to attenuate the off-key issues, the pitch predictor is re-designed to predict the real to note pitch ratio. Both objective and subjective experimental results indicate that the proposed SVS system can produce singing voice with higher-quality outperforming VISinger.

{{</citation>}}


### (45/112) Sequential Pitch Distributions for Raga Detection (Vishwaas Narasinh et al., 2023)

{{<citation>}}

Vishwaas Narasinh, Senthil Raja G. (2023)  
**Sequential Pitch Distributions for Raga Detection**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2308.16421v1)  

---


**ABSTRACT**  
Raga is a fundamental melodic concept in Indian Art Music (IAM). It is characterized by complex patterns. All performances and compositions are based on the raga framework. Raga and tonic detection have been a long-standing research problem in the field of Music Information Retrieval. In this paper, we attempt to detect the raga using a novel feature to extract sequential or temporal information from an audio sample. We call these Sequential Pitch Distributions (SPD), which are distributions taken over pitch values between two given pitch values over time. We also achieve state-of-the-art results on both Hindustani and Carnatic music raga data sets with an accuracy of 99% and 88.13%, respectively. SPD gives a great boost in accuracy over a standard pitch distribution. The main goal of this paper, however, is to present an alternative approach to modeling the temporal aspects of the melody and thereby deducing the raga.

{{</citation>}}


### (46/112) The Biased Journey of MSD_AUDIO.ZIP (Haven Kim et al., 2023)

{{<citation>}}

Haven Kim, Keunwoo Choi, Mateusz Modrzejewski, Cynthia C. S. Liem. (2023)  
**The Biased Journey of MSD_AUDIO.ZIP**  

---
Primary Category: cs.SD  
Categories: cs-CY, cs-SD, cs.SD, eess-AS  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.16389v1)  

---


**ABSTRACT**  
The equitable distribution of academic data is crucial for ensuring equal research opportunities, and ultimately further progress. Yet, due to the complexity of using the API for audio data that corresponds to the Million Song Dataset along with its misreporting (before 2016) and the discontinuation of this API (after 2016), access to this data has become restricted to those within certain affiliations that are connected peer-to-peer. In this paper, we delve into this issue, drawing insights from the experiences of 22 individuals who either attempted to access the data or played a role in its creation. With this, we hope to initiate more critical dialogue and more thoughtful consideration with regard to access privilege in the MIR community.

{{</citation>}}


## cs.AI (9)



### (47/112) Fuzzy Approach for Audio-Video Emotion Recognition in Computer Games for Children (Pavel Kozlov et al., 2023)

{{<citation>}}

Pavel Kozlov, Alisher Akram, Pakizar Shamoi. (2023)  
**Fuzzy Approach for Audio-Video Emotion Recognition in Computer Games for Children**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2309.00138v1)  

---


**ABSTRACT**  
Computer games are widespread nowadays and enjoyed by people of all ages. But when it comes to kids, playing these games can be more than just fun, it is a way for them to develop important skills and build emotional intelligence. Facial expressions and sounds that kids produce during gameplay reflect their feelings, thoughts, and moods. In this paper, we propose a novel framework that integrates a fuzzy approach for the recognition of emotions through the analysis of audio and video data. Our focus lies within the specific context of computer games tailored for children, aiming to enhance their overall user experience. We use the FER dataset to detect facial emotions in video frames recorded from the screen during the game. For the audio emotion recognition of sounds a kid produces during the game, we use CREMA-D, TESS, RAVDESS, and Savee datasets. Next, a fuzzy inference system is used for the fusion of results. Besides this, our system can detect emotion stability and emotion diversity during gameplay, which, together with prevailing emotion report, can serve as valuable information for parents worrying about the effect of certain games on their kids. The proposed approach has shown promising results in the preliminary experiments we conducted, involving 3 different video games, namely fighting, racing, and logic games, and providing emotion-tracking results for kids in each game. Our study can contribute to the advancement of child-oriented game development, which is not only engaging but also accounts for children's cognitive and emotional states.

{{</citation>}}


### (48/112) Exploring the Potential of Large Language Models to Generate Formative Programming Feedback (Natalie Kiesler et al., 2023)

{{<citation>}}

Natalie Kiesler, Dominic Lohr, Hieke Keuning. (2023)  
**Exploring the Potential of Large Language Models to Generate Formative Programming Feedback**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-SE, cs.AI  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.00029v1)  

---


**ABSTRACT**  
Ever since the emergence of large language models (LLMs) and related applications, such as ChatGPT, its performance and error analysis for programming tasks have been subject to research. In this work-in-progress paper, we explore the potential of such LLMs for computing educators and learners, as we analyze the feedback it generates to a given input containing program code. In particular, we aim at (1) exploring how an LLM like ChatGPT responds to students seeking help with their introductory programming tasks, and (2) identifying feedback types in its responses. To achieve these goals, we used students' programming sequences from a dataset gathered within a CS1 course as input for ChatGPT along with questions required to elicit feedback and correct solutions. The results show that ChatGPT performs reasonably well for some of the introductory programming tasks and student errors, which means that students can potentially benefit. However, educators should provide guidance on how to use the provided feedback, as it can contain misleading information for novices.

{{</citation>}}


### (49/112) Agent Teaming Situation Awareness (ATSA): A Situation Awareness Framework for Human-AI Teaming (Qi Gao et al., 2023)

{{<citation>}}

Qi Gao, Wei Xu, Mowei Shen, Zaifeng Gao. (2023)  
**Agent Teaming Situation Awareness (ATSA): A Situation Awareness Framework for Human-AI Teaming**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.16785v1)  

---


**ABSTRACT**  
The rapid advancements in artificial intelligence (AI) have led to a growing trend of human-AI teaming (HAT) in various fields. As machines continue to evolve from mere automation to a state of autonomy, they are increasingly exhibiting unexpected behaviors and human-like cognitive/intelligent capabilities, including situation awareness (SA). This shift has the potential to enhance the performance of mixed human-AI teams over all-human teams, underscoring the need for a better understanding of the dynamic SA interactions between humans and machines. To this end, we provide a review of leading SA theoretical models and a new framework for SA in the HAT context based on the key features and processes of HAT. The Agent Teaming Situation Awareness (ATSA) framework unifies human and AI behavior, and involves bidirectional, and dynamic interaction. The framework is based on the individual and team SA models and elaborates on the cognitive mechanisms for modeling HAT. Similar perceptual cycles are adopted for the individual (including both human and AI) and the whole team, which is tailored to the unique requirements of the HAT context. ATSA emphasizes cohesive and effective HAT through structures and components, including teaming understanding, teaming control, and the world, as well as adhesive transactive part. We further propose several future research directions to expand on the distinctive contributions of ATSA and address the specific and pressing next steps.

{{</citation>}}


### (50/112) StratMed: Relevance Stratification for Low-resource Medication Recommendation (Xiang Li, 2023)

{{<citation>}}

Xiang Li. (2023)  
**StratMed: Relevance Stratification for Low-resource Medication Recommendation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.16781v1)  

---


**ABSTRACT**  
With the growing imbalance between limited medical resources and escalating demands, AI-based clinical tasks have become paramount. Medication recommendation, as a sub-domain, aims to amalgamate longitudinal patient history with medical knowledge, assisting physicians in prescribing safer and more accurate medication combinations. Existing methods overlook the inherent long-tail distribution in medical data, lacking balanced representation between head and tail data, which leads to sub-optimal model performance. To address this challenge, we introduce StratMed, a model that incorporates an innovative relevance stratification mechanism. It harmonizes discrepancies in data long-tail distribution and strikes a balance between the safety and accuracy of medication combinations. Specifically, we first construct a pre-training method using deep learning networks to obtain entity representation. After that, we design a pyramid-like data stratification method to obtain more generalized entity relationships by reinforcing the features of unpopular entities. Based on this relationship, we designed two graph structures to express medication precision and safety at the same level to obtain visit representations. Finally, the patient's historical clinical information is fitted to generate medication combinations for the current health condition. Experiments on the MIMIC-III dataset demonstrate that our method has outperformed current state-of-the-art methods in four evaluation metrics (including safety and accuracy).

{{</citation>}}


### (51/112) Developing a Scalable Benchmark for Assessing Large Language Models in Knowledge Graph Engineering (Lars-Peter Meyer et al., 2023)

{{<citation>}}

Lars-Peter Meyer, Johannes Frey, Kurt Junghanns, Felix Brei, Kirill Bulert, Sabine Gründer-Fahrer, Michael Martin. (2023)  
**Developing a Scalable Benchmark for Assessing Large Language Models in Knowledge Graph Engineering**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-DB, cs.AI  
Keywords: Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2308.16622v1)  

---


**ABSTRACT**  
As the field of Large Language Models (LLMs) evolves at an accelerated pace, the critical need to assess and monitor their performance emerges. We introduce a benchmarking framework focused on knowledge graph engineering (KGE) accompanied by three challenges addressing syntax and error correction, facts extraction and dataset generation. We show that while being a useful tool, LLMs are yet unfit to assist in knowledge graph generation with zero-shot prompting. Consequently, our LLM-KG-Bench framework provides automatic evaluation and storage of LLM responses as well as statistical data and visualization tools to support tracking of prompt engineering and model performance.

{{</citation>}}


### (52/112) High Accuracy Location Information Extraction from Social Network Texts Using Natural Language Processing (Lossan Bonde et al., 2023)

{{<citation>}}

Lossan Bonde, Severin Dembele. (2023)  
**High Accuracy Location Information Extraction from Social Network Texts Using Natural Language Processing**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Information Extraction, NLP, Natural Language Processing, Social Network  
[Paper Link](http://arxiv.org/abs/2308.16615v1)  

---


**ABSTRACT**  
Terrorism has become a worldwide plague with severe consequences for the development of nations. Besides killing innocent people daily and preventing educational activities from taking place, terrorism is also hindering economic growth. Machine Learning (ML) and Natural Language Processing (NLP) can contribute to fighting terrorism by predicting in real-time future terrorist attacks if accurate data is available. This paper is part of a research project that uses text from social networks to extract necessary information to build an adequate dataset for terrorist attack prediction. We collected a set of 3000 social network texts about terrorism in Burkina Faso and used a subset to experiment with existing NLP solutions. The experiment reveals that existing solutions have poor accuracy for location recognition, which our solution resolves. We will extend the solution to extract dates and action information to achieve the project's goal.

{{</citation>}}


### (53/112) The AI Revolution: Opportunities and Challenges for the Finance Sector (Carsten Maple et al., 2023)

{{<citation>}}

Carsten Maple, Lukasz Szpruch, Gregory Epiphaniou, Kalina Staykova, Simran Singh, William Penwarden, Yisi Wen, Zijian Wang, Jagdish Hariharan, Pavle Avramovic. (2023)  
**The AI Revolution: Opportunities and Challenges for the Finance Sector**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.16538v1)  

---


**ABSTRACT**  
This report examines Artificial Intelligence (AI) in the financial sector, outlining its potential to revolutionise the industry and identify its challenges. It underscores the criticality of a well-rounded understanding of AI, its capabilities, and its implications to effectively leverage its potential while mitigating associated risks. The potential of AI potential extends from augmenting existing operations to paving the way for novel applications in the finance sector. The application of AI in the financial sector is transforming the industry. Its use spans areas from customer service enhancements, fraud detection, and risk management to credit assessments and high-frequency trading. However, along with these benefits, AI also presents several challenges. These include issues related to transparency, interpretability, fairness, accountability, and trustworthiness. The use of AI in the financial sector further raises critical questions about data privacy and security. A further issue identified in this report is the systemic risk that AI can introduce to the financial sector. Being prone to errors, AI can exacerbate existing systemic risks, potentially leading to financial crises. Regulation is crucial to harnessing the benefits of AI while mitigating its potential risks. Despite the global recognition of this need, there remains a lack of clear guidelines or legislation for AI use in finance. This report discusses key principles that could guide the formation of effective AI regulation in the financial sector, including the need for a risk-based approach, the inclusion of ethical considerations, and the importance of maintaining a balance between innovation and consumer protection. The report provides recommendations for academia, the finance industry, and regulators.

{{</citation>}}


### (54/112) Expanding Frozen Vision-Language Models without Retraining: Towards Improved Robot Perception (Riley Tavassoli et al., 2023)

{{<citation>}}

Riley Tavassoli, Mani Amani, Reza Akhavian. (2023)  
**Expanding Frozen Vision-Language Models without Retraining: Towards Improved Robot Perception**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-RO, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.16493v1)  

---


**ABSTRACT**  
Vision-language models (VLMs) have shown powerful capabilities in visual question answering and reasoning tasks by combining visual representations with the abstract skill set large language models (LLMs) learn during pretraining. Vision, while the most popular modality to augment LLMs with, is only one representation of a scene. In human-robot interaction scenarios, robot perception requires accurate scene understanding by the robot. In this paper, we define and demonstrate a method of aligning the embedding spaces of different modalities (in this case, inertial measurement unit (IMU) data) to the vision embedding space through a combination of supervised and contrastive training, enabling the VLM to understand and reason about these additional modalities without retraining. We opt to give the model IMU embeddings directly over using a separate human activity recognition model that feeds directly into the prompt to allow for any nonlinear interactions between the query, image, and IMU signal that would be lost by mapping the IMU data to a discrete activity label. Further, we demonstrate our methodology's efficacy through experiments involving human activity recognition using IMU data and visual inputs. Our results show that using multiple modalities as input improves the VLM's scene understanding and enhances its overall performance in various tasks, thus paving the way for more versatile and capable language models in multi-modal contexts.

{{</citation>}}


### (55/112) Contrastive Representation Learning Based on Multiple Node-centered Subgraphs (Dong Li et al., 2023)

{{<citation>}}

Dong Li, Wenjun Wang, Minglai Shao, Chen Zhao. (2023)  
**Contrastive Representation Learning Based on Multiple Node-centered Subgraphs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.16441v1)  

---


**ABSTRACT**  
As the basic element of graph-structured data, node has been recognized as the main object of study in graph representation learning. A single node intuitively has multiple node-centered subgraphs from the whole graph (e.g., one person in a social network has multiple social circles based on his different relationships). We study this intuition under the framework of graph contrastive learning, and propose a multiple node-centered subgraphs contrastive representation learning method to learn node representation on graphs in a self-supervised way. Specifically, we carefully design a series of node-centered regional subgraphs of the central node. Then, the mutual information between different subgraphs of the same node is maximized by contrastive loss. Experiments on various real-world datasets and different downstream tasks demonstrate that our model has achieved state-of-the-art results.

{{</citation>}}


## q-fin.ST (1)



### (56/112) Predicting Financial Market Trends using Time Series Analysis and Natural Language Processing (Ali Asgarov, 2023)

{{<citation>}}

Ali Asgarov. (2023)  
**Predicting Financial Market Trends using Time Series Analysis and Natural Language Processing**  

---
Primary Category: q-fin.ST  
Categories: cs-AI, cs-LG, q-fin-ST, q-fin.ST  
Keywords: Financial, LSTM, Natural Language Processing, Time Series, Twitter  
[Paper Link](http://arxiv.org/abs/2309.00136v1)  

---


**ABSTRACT**  
Forecasting financial market trends through time series analysis and natural language processing poses a complex and demanding undertaking, owing to the numerous variables that can influence stock prices. These variables encompass a spectrum of economic and political occurrences, as well as prevailing public attitudes. Recent research has indicated that the expression of public sentiments on social media platforms such as Twitter may have a noteworthy impact on the determination of stock prices. The objective of this study was to assess the viability of Twitter sentiments as a tool for predicting stock prices of major corporations such as Tesla, Apple. Our study has revealed a robust association between the emotions conveyed in tweets and fluctuations in stock prices. Our findings indicate that positivity, negativity, and subjectivity are the primary determinants of fluctuations in stock prices. The data was analyzed utilizing the Long-Short Term Memory neural network (LSTM) model, which is currently recognized as the leading methodology for predicting stock prices by incorporating Twitter sentiments and historical stock prices data. The models utilized in our study demonstrated a high degree of reliability and yielded precise outcomes for the designated corporations. In summary, this research emphasizes the significance of incorporating public opinions into the prediction of stock prices. The application of Time Series Analysis and Natural Language Processing methodologies can yield significant scientific findings regarding financial market patterns, thereby facilitating informed decision-making among investors. The results of our study indicate that the utilization of Twitter sentiments can serve as a potent instrument for forecasting stock prices, and ought to be factored in when formulating investment strategies.

{{</citation>}}


## cs.SE (7)



### (57/112) ChatGPT and Excel -- trust, but verify (Patrick O'Beirne, 2023)

{{<citation>}}

Patrick O'Beirne. (2023)  
**ChatGPT and Excel -- trust, but verify**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.00120v1)  

---


**ABSTRACT**  
This paper adopts a critical approach to ChatGPT, showing how its huge reach makes it a useful tool for people with simple requirements but a bad, even misleading guide to those with more complex problems which are more rarely present in the training data and even more rarely have straightforward solutions. It works through four exercises in creating lookup formulas using chatbots, showing the need to test the offered solutions. They are a simple lookup, a lookup to the left, a match of two values at the same time, and intentionally confusing the models by using common language with technical meaning in Excel. It concludes with a practical guide for how to add an Excelscript button, with system and user prompts, to the ChatGPT API into the Excel desktop environment, supported by a blog post giving the technical details for those interested.

{{</citation>}}


### (58/112) Experimenting with ChatGPT for Spreadsheet Formula Generation: Evidence of Risk in AI Generated Spreadsheets (Simon Thorne, 2023)

{{<citation>}}

Simon Thorne. (2023)  
**Experimenting with ChatGPT for Spreadsheet Formula Generation: Evidence of Risk in AI Generated Spreadsheets**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.00095v1)  

---


**ABSTRACT**  
Large Language Models (LLM) have become sophisticated enough that complex computer programs can be created through interpretation of plain English sentences and implemented in a variety of modern languages such as Python, Java Script, C++ and Spreadsheets. These tools are powerful and relatively accurate and therefore provide broad access to computer programming regardless of the background or knowledge of the individual using them. This paper presents a series of experiments with ChatGPT to explore the tool's ability to produce valid spreadsheet formulae and related computational outputs in situations where ChatGPT has to deduce, infer and problem solve the answer. The results show that in certain circumstances, ChatGPT can produce correct spreadsheet formulae with correct reasoning, deduction and inference. However, when information is limited, uncertain or the problem is too complex, the accuracy of ChatGPT breaks down as does its ability to reason, infer and deduce. This can also result in false statements and "hallucinations" that all subvert the process of creating spreadsheet formulae.

{{</citation>}}


### (59/112) Toward Automatically Completing GitHub Workflows (Antonio Mastropaolo et al., 2023)

{{<citation>}}

Antonio Mastropaolo, Fiorella Zampetti, Massimiliano Di Penta, Gabriele Bavota. (2023)  
**Toward Automatically Completing GitHub Workflows**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.16774v1)  

---


**ABSTRACT**  
Continuous integration and delivery (CI/CD) are nowadays at the core of software development. Their benefits come at the cost of setting up and maintaining the CI/CD pipeline, which requires knowledge and skills often orthogonal to those entailed in other software-related tasks. While several recommender systems have been proposed to support developers across a variety of tasks, little automated support is available when it comes to setting up and maintaining CI/CD pipelines. We present GH-WCOM (GitHub Workflow COMpletion), a Transformer-based approach supporting developers in writing a specific type of CI/CD pipelines, namely GitHub workflows. To deal with such a task, we designed an abstraction process to help the learning of the transformer while still making GH-WCOM able to recommend very peculiar workflow elements such as tool options and scripting elements. Our empirical study shows that GH-WCOM provides up to 34.23% correct predictions, and the model's confidence is a reliable proxy for the recommendations' correctness likelihood.

{{</citation>}}


### (60/112) Learning to Represent Patches (Xunzhu Tang et al., 2023)

{{<citation>}}

Xunzhu Tang, Haoye Tian, Zhenghan Chen, Weiguo Pian, Saad Ezzini, Abdoul Kader Kabore, Andrew Habib, Jacques Klein, Tegawende F. Bissyande. (2023)  
**Learning to Represent Patches**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2308.16586v1)  

---


**ABSTRACT**  
Patch representation is crucial in automating various software engineering tasks, like determining patch accuracy or summarizing code changes. While recent research has employed deep learning for patch representation, focusing on token sequences or Abstract Syntax Trees (ASTs), they often miss the change's semantic intent and the context of modified lines. To bridge this gap, we introduce a novel method, Patcherizer. It delves into the intentions of context and structure, merging the surrounding code context with two innovative representations. These capture the intention in code changes and the intention in AST structural modifications pre and post-patch. This holistic representation aptly captures a patch's underlying intentions. Patcherizer employs graph convolutional neural networks for structural intention graph representation and transformers for intention sequence representation. We evaluated Patcherizer's embeddings' versatility in three areas: (1) Patch description generation, (2) Patch accuracy prediction, and (3) Patch intention identification. Our experiments demonstrate the representation's efficacy across all tasks, outperforming state-of-the-art methods. For example, in patch description generation, Patcherizer excels, showing an average boost of 19.39% in BLEU, 8.71% in ROUGE-L, and 34.03% in METEOR scores.

{{</citation>}}


### (61/112) An Energy-Aware Approach to Design Self-Adaptive AI-based Applications on the Edge (Alessandro Tundo et al., 2023)

{{<citation>}}

Alessandro Tundo, Marco Mobilio, Shashikant Ilager, Ivona Brandić, Ezio Bartocci, Leonardo Mariani. (2023)  
**An Energy-Aware Approach to Design Self-Adaptive AI-based Applications on the Edge**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs-SY, cs.SE, eess-SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.00022v1)  

---


**ABSTRACT**  
The advent of edge devices dedicated to machine learning tasks enabled the execution of AI-based applications that efficiently process and classify the data acquired by the resource-constrained devices populating the Internet of Things. The proliferation of such applications (e.g., critical monitoring in smart cities) demands new strategies to make these systems also sustainable from an energetic point of view.   In this paper, we present an energy-aware approach for the design and deployment of self-adaptive AI-based applications that can balance application objectives (e.g., accuracy in object detection and frames processing rate) with energy consumption. We address the problem of determining the set of configurations that can be used to self-adapt the system with a meta-heuristic search procedure that only needs a small number of empirical samples. The final set of configurations are selected using weighted gray relational analysis, and mapped to the operation modes of the self-adaptive application.   We validate our approach on an AI-based application for pedestrian detection. Results show that our self-adaptive application can outperform non-adaptive baseline configurations by saving up to 81\% of energy while loosing only between 2% and 6% in accuracy.

{{</citation>}}


### (62/112) Effective Test Generation Using Pre-trained Large Language Models and Mutation Testing (Arghavan Moradi Dakhel et al., 2023)

{{<citation>}}

Arghavan Moradi Dakhel, Amin Nikanjam, Vahid Majdinasab, Foutse Khomh, Michel C. Desmarais. (2023)  
**Effective Test Generation Using Pre-trained Large Language Models and Mutation Testing**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.16557v1)  

---


**ABSTRACT**  
One of the critical phases in software development is software testing. Testing helps with identifying potential bugs and reducing maintenance costs. The goal of automated test generation tools is to ease the development of tests by suggesting efficient bug-revealing tests. Recently, researchers have leveraged Large Language Models (LLMs) of code to generate unit tests. While the code coverage of generated tests was usually assessed, the literature has acknowledged that the coverage is weakly correlated with the efficiency of tests in bug detection. To improve over this limitation, in this paper, we introduce MuTAP for improving the effectiveness of test cases generated by LLMs in terms of revealing bugs by leveraging mutation testing. Our goal is achieved by augmenting prompts with surviving mutants, as those mutants highlight the limitations of test cases in detecting bugs. MuTAP is capable of generating effective test cases in the absence of natural language descriptions of the Program Under Test (PUTs). We employ different LLMs within MuTAP and evaluate their performance on different benchmarks. Our results show that our proposed method is able to detect up to 28% more faulty human-written code snippets. Among these, 17% remained undetected by both the current state-of-the-art fully automated test generation tool (i.e., Pynguin) and zero-shot/few-shot learning approaches on LLMs. Furthermore, MuTAP achieves a Mutation Score (MS) of 93.57% on synthetic buggy code, outperforming all other approaches in our evaluation. Our findings suggest that although LLMs can serve as a useful tool to generate test cases, they require specific post-processing steps to enhance the effectiveness of the generated test cases which may suffer from syntactic or functional errors and may be ineffective in detecting certain types of bugs and testing corner cases PUTs.

{{</citation>}}


### (63/112) MaintainoMATE: A GitHub App for Intelligent Automation of Maintenance Activities (Anas Nadeem et al., 2023)

{{<citation>}}

Anas Nadeem, Muhammad Usman Sarwar, Muhammad Zubair Malik. (2023)  
**MaintainoMATE: A GitHub App for Intelligent Automation of Maintenance Activities**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.16464v1)  

---


**ABSTRACT**  
Software development projects rely on issue tracking systems at the core of tracking maintenance tasks such as bug reports, and enhancement requests. Incoming issue-reports on these issue tracking systems must be managed in an effective manner. First, they must be labelled and then assigned to a particular developer with relevant expertise. This handling of issue-reports is critical and requires thorough scanning of the text entered in an issue-report making it a labor-intensive task. In this paper, we present a unified framework called MaintainoMATE, which is capable of automatically categorizing the issue-reports in their respective category and further assigning the issue-reports to a developer with relevant expertise. We use the Bidirectional Encoder Representations from Transformers (BERT), as an underlying model for MaintainoMATE to learn the contextual information for automatic issue-report labeling and assignment tasks. We deploy the framework used in this work as a GitHub application. We empirically evaluate our approach on GitHub issue-reports to show its capability of assigning labels to the issue-reports. We were able to achieve an F1-score close to 80\%, which is comparable to existing state-of-the-art results. Similarly, our initial evaluations show that we can assign relevant developers to the issue-reports with an F1 score of 54\%, which is a significant improvement over existing approaches. Our initial findings suggest that MaintainoMATE has the potential of improving software quality and reducing maintenance costs by accurately automating activities involved in the maintenance processes. Our future work would be directed towards improving the issue-assignment module.

{{</citation>}}


## cs.LG (17)



### (64/112) Deep Semi-Supervised Anomaly Detection for Finding Fraud in the Futures Market (Timothy DeLise, 2023)

{{<citation>}}

Timothy DeLise. (2023)  
**Deep Semi-Supervised Anomaly Detection for Finding Fraud in the Futures Market**  

---
Primary Category: cs.LG  
Categories: 91G99, cs-LG, cs.LG, q-fin-RM  
Keywords: Anomaly Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2309.00088v1)  

---


**ABSTRACT**  
Modern financial electronic exchanges are an exciting and fast-paced marketplace where billions of dollars change hands every day. They are also rife with manipulation and fraud. Detecting such activity is a major undertaking, which has historically been a job reserved exclusively for humans. Recently, more research and resources have been focused on automating these processes via machine learning and artificial intelligence. Fraud detection is overwhelmingly associated with the greater field of anomaly detection, which is usually performed via unsupervised learning techniques because of the lack of labeled data needed for supervised learning. However, a small quantity of labeled data does often exist. This research article aims to evaluate the efficacy of a deep semi-supervised anomaly detection technique, called Deep SAD, for detecting fraud in high-frequency financial data. We use exclusive proprietary limit order book data from the TMX exchange in Montr\'eal, with a small set of true labeled instances of fraud, to evaluate Deep SAD against its unsupervised predecessor. We show that incorporating a small amount of labeled data into an unsupervised anomaly detection framework can greatly improve its accuracy.

{{</citation>}}


### (65/112) RePo: Resilient Model-Based Reinforcement Learning by Regularizing Posterior Predictability (Chuning Zhu et al., 2023)

{{<citation>}}

Chuning Zhu, Max Simchowitz, Siri Gadipudi, Abhishek Gupta. (2023)  
**RePo: Resilient Model-Based Reinforcement Learning by Regularizing Posterior Predictability**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.00082v1)  

---


**ABSTRACT**  
Visual model-based RL methods typically encode image observations into low-dimensional representations in a manner that does not eliminate redundant information. This leaves them susceptible to spurious variations -- changes in task-irrelevant components such as background distractors or lighting conditions. In this paper, we propose a visual model-based RL method that learns a latent representation resilient to such spurious variations. Our training objective encourages the representation to be maximally predictive of dynamics and reward, while constraining the information flow from the observation to the latent representation. We demonstrate that this objective significantly bolsters the resilience of visual model-based RL methods to visual distractors, allowing them to operate in dynamic environments. We then show that while the learned encoder is resilient to spirious variations, it is not invariant under significant distribution shift. To address this, we propose a simple reward-free alignment procedure that enables test time adaptation of the encoder. This allows for quick adaptation to widely differing environments without having to relearn the dynamics and policy. Our effort is a step towards making model-based RL a practical and useful tool for dynamic, diverse domains. We show its effectiveness in simulation benchmarks with significant spurious variations as well as a real-world egocentric navigation task with noisy TVs in the background. Videos and code at https://zchuning.github.io/repo-website/.

{{</citation>}}


### (66/112) On the Implicit Bias of Adam (Matias D. Cattaneo et al., 2023)

{{<citation>}}

Matias D. Cattaneo, Jason M. Klusowski, Boris Shigida. (2023)  
**On the Implicit Bias of Adam**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-OC, stat-CO, stat-ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.00079v1)  

---


**ABSTRACT**  
In previous literature, backward error analysis was used to find ordinary differential equations (ODEs) approximating the gradient descent trajectory. It was found that finite step sizes implicitly regularize solutions because terms appearing in the ODEs penalize the two-norm of the loss gradients. We prove that the existence of similar implicit regularization in RMSProp and Adam depends on their hyperparameters and the training stage, but with a different "norm" involved: the corresponding ODE terms either penalize the (perturbed) one-norm of the loss gradients or, on the contrary, hinder its decrease (the latter case being typical). We also conduct numerical experiments and discuss how the proven facts can influence generalization.

{{</citation>}}


### (67/112) Transformers as Support Vector Machines (Davoud Ataee Tarzanagh et al., 2023)

{{<citation>}}

Davoud Ataee Tarzanagh, Yingcong Li, Christos Thrampoulidis, Samet Oymak. (2023)  
**Transformers as Support Vector Machines**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG, math-OC  
Keywords: Attention, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.16898v1)  

---


**ABSTRACT**  
Since its inception in "Attention Is All You Need", transformer architecture has led to revolutionary advancements in NLP. The attention layer within the transformer admits a sequence of input tokens $X$ and makes them interact through pairwise similarities computed as softmax$(XQK^\top X^\top)$, where $(K,Q)$ are the trainable key-query parameters. In this work, we establish a formal equivalence between the optimization geometry of self-attention and a hard-margin SVM problem that separates optimal input tokens from non-optimal tokens using linear constraints on the outer-products of token pairs. This formalism allows us to characterize the implicit bias of 1-layer transformers optimized with gradient descent: (1) Optimizing the attention layer with vanishing regularization, parameterized by $(K,Q)$, converges in direction to an SVM solution minimizing the nuclear norm of the combined parameter $W=KQ^\top$. Instead, directly parameterizing by $W$ minimizes a Frobenius norm objective. We characterize this convergence, highlighting that it can occur toward locally-optimal directions rather than global ones. (2) Complementing this, we prove the local/global directional convergence of gradient descent under suitable geometric conditions. Importantly, we show that over-parameterization catalyzes global convergence by ensuring the feasibility of the SVM problem and by guaranteeing a benign optimization landscape devoid of stationary points. (3) While our theory applies primarily to linear prediction heads, we propose a more general SVM equivalence that predicts the implicit bias with nonlinear heads. Our findings are applicable to arbitrary datasets and their validity is verified via experiments. We also introduce several open problems and research directions. We believe these findings inspire the interpretation of transformers as a hierarchy of SVMs that separates and selects optimal tokens.

{{</citation>}}


### (68/112) Irregular Traffic Time Series Forecasting Based on Asynchronous Spatio-Temporal Graph Convolutional Network (Weijia Zhang et al., 2023)

{{<citation>}}

Weijia Zhang, Le Zhang, Jindong Han, Hao Liu, Jingbo Zhou, Yu Mei, Hui Xiong. (2023)  
**Irregular Traffic Time Series Forecasting Based on Asynchronous Spatio-Temporal Graph Convolutional Network**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Convolutional Network, Time Series  
[Paper Link](http://arxiv.org/abs/2308.16818v2)  

---


**ABSTRACT**  
Accurate traffic forecasting at intersections governed by intelligent traffic signals is critical for the advancement of an effective intelligent traffic signal control system. However, due to the irregular traffic time series produced by intelligent intersections, the traffic forecasting task becomes much more intractable and imposes three major new challenges: 1) asynchronous spatial dependency, 2) irregular temporal dependency among traffic data, and 3) variable-length sequence to be predicted, which severely impede the performance of current traffic forecasting methods. To this end, we propose an Asynchronous Spatio-tEmporal graph convolutional nEtwoRk (ASeer) to predict the traffic states of the lanes entering intelligent intersections in a future time window. Specifically, by linking lanes via a traffic diffusion graph, we first propose an Asynchronous Graph Diffusion Network to model the asynchronous spatial dependency between the time-misaligned traffic state measurements of lanes. After that, to capture the temporal dependency within irregular traffic state sequence, a learnable personalized time encoding is devised to embed the continuous time for each lane. Then we propose a Transformable Time-aware Convolution Network that learns meta-filters to derive time-aware convolution filters with transformable filter sizes for efficient temporal convolution on the irregular sequence. Furthermore, a Semi-Autoregressive Prediction Network consisting of a state evolution unit and a semiautoregressive predictor is designed to effectively and efficiently predict variable-length traffic state sequences. Extensive experiments on two real-world datasets demonstrate the effectiveness of ASeer in six metrics.

{{</citation>}}


### (69/112) Rank Collapse Causes Over-Smoothing and Over-Correlation in Graph Neural Networks (Andreas Roth et al., 2023)

{{<citation>}}

Andreas Roth, Thomas Liebig. (2023)  
**Rank Collapse Causes Over-Smoothing and Over-Correlation in Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.16800v1)  

---


**ABSTRACT**  
Our study reveals new theoretical insights into over-smoothing and feature over-correlation in deep graph neural networks. We show the prevalence of invariant subspaces, demonstrating a fixed relative behavior that is unaffected by feature transformations. Our work clarifies recent observations related to convergence to a constant state and a potential over-separation of node states, as the amplification of subspaces only depends on the spectrum of the aggregation function. In linear scenarios, this leads to node representations being dominated by a low-dimensional subspace with an asymptotic convergence rate independent of the feature transformations. This causes a rank collapse of the node representations, resulting in over-smoothing when smooth vectors span this subspace, and over-correlation even when over-smoothing is avoided. Guided by our theory, we propose a sum of Kronecker products as a beneficial property that can provably prevent over-smoothing, over-correlation, and rank collapse. We empirically extend our insights to the non-linear case, demonstrating the inability of existing models to capture linearly independent features.

{{</citation>}}


### (70/112) Efficacy of Neural Prediction-Based NAS for Zero-Shot NAS Paradigm (Minh Le et al., 2023)

{{<citation>}}

Minh Le, Nhan Nguyen, Ngoc Hoang Luong. (2023)  
**Efficacy of Neural Prediction-Based NAS for Zero-Shot NAS Paradigm**  

---
Primary Category: cs.LG  
Categories: I-2-6, cs-AI, cs-LG, cs.LG  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.16775v1)  

---


**ABSTRACT**  
In prediction-based Neural Architecture Search (NAS), performance indicators derived from graph convolutional networks have shown significant success. These indicators, achieved by representing feed-forward structures as component graphs through one-hot encoding, face a limitation: their inability to evaluate architecture performance across varying search spaces. In contrast, handcrafted performance indicators (zero-shot NAS), which use the same architecture with random initialization, can generalize across multiple search spaces. Addressing this limitation, we propose a novel approach for zero-shot NAS using deep learning. Our method employs Fourier sum of sines encoding for convolutional kernels, enabling the construction of a computational feed-forward graph with a structure similar to the architecture under evaluation. These encodings are learnable and offer a comprehensive view of the architecture's topological information. An accompanying multi-layer perceptron (MLP) then ranks these architectures based on their encodings. Experimental results show that our approach surpasses previous methods using graph convolutional networks in terms of correlation on the NAS-Bench-201 dataset and exhibits a higher convergence rate. Moreover, our extracted feature representation trained on each NAS-Benchmark is transferable to other NAS-Benchmarks, showing promising generalizability across multiple search spaces. The code is available at: https://github.com/minh1409/DFT-NPZS-NAS

{{</citation>}}


### (71/112) Robust Representation Learning for Unreliable Partial Label Learning (Yu Shi et al., 2023)

{{<citation>}}

Yu Shi, Dong-Dong Wu, Xin Geng, Min-Ling Zhang. (2023)  
**Robust Representation Learning for Unreliable Partial Label Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.16718v1)  

---


**ABSTRACT**  
Partial Label Learning (PLL) is a type of weakly supervised learning where each training instance is assigned a set of candidate labels, but only one label is the ground-truth. However, this idealistic assumption may not always hold due to potential annotation inaccuracies, meaning the ground-truth may not be present in the candidate label set. This is known as Unreliable Partial Label Learning (UPLL) that introduces an additional complexity due to the inherent unreliability and ambiguity of partial labels, often resulting in a sub-optimal performance with existing methods. To address this challenge, we propose the Unreliability-Robust Representation Learning framework (URRL) that leverages unreliability-robust contrastive learning to help the model fortify against unreliable partial labels effectively. Concurrently, we propose a dual strategy that combines KNN-based candidate label set correction and consistency-regularization-based label disambiguation to refine label quality and enhance the ability of representation learning within the URRL framework. Extensive experiments demonstrate that the proposed method outperforms state-of-the-art PLL methods on various datasets with diverse degrees of unreliability and ambiguity. Furthermore, we provide a theoretical analysis of our approach from the perspective of the expectation maximization (EM) algorithm. Upon acceptance, we pledge to make the code publicly accessible.

{{</citation>}}


### (72/112) Continual Learning From a Stream of APIs (Enneng Yang et al., 2023)

{{<citation>}}

Enneng Yang, Zhenyi Wang, Li Shen, Nan Yin, Tongliang Liu, Guibing Guo, Xingwei Wang, Dacheng Tao. (2023)  
**Continual Learning From a Stream of APIs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.00023v1)  

---


**ABSTRACT**  
Continual learning (CL) aims to learn new tasks without forgetting previous tasks. However, existing CL methods require a large amount of raw data, which is often unavailable due to copyright considerations and privacy risks. Instead, stakeholders usually release pre-trained machine learning models as a service (MLaaS), which users can access via APIs. This paper considers two practical-yet-novel CL settings: data-efficient CL (DECL-APIs) and data-free CL (DFCL-APIs), which achieve CL from a stream of APIs with partial or no raw data. Performing CL under these two new settings faces several challenges: unavailable full raw data, unknown model parameters, heterogeneous models of arbitrary architecture and scale, and catastrophic forgetting of previous APIs. To overcome these issues, we propose a novel data-free cooperative continual distillation learning framework that distills knowledge from a stream of APIs into a CL model by generating pseudo data, just by querying APIs. Specifically, our framework includes two cooperative generators and one CL model, forming their training as an adversarial game. We first use the CL model and the current API as fixed discriminators to train generators via a derivative-free method. Generators adversarially generate hard and diverse synthetic data to maximize the response gap between the CL model and the API. Next, we train the CL model by minimizing the gap between the responses of the CL model and the black-box API on synthetic data, to transfer the API's knowledge to the CL model. Furthermore, we propose a new regularization term based on network similarity to prevent catastrophic forgetting of previous APIs.Our method performs comparably to classic CL with full raw data on the MNIST and SVHN in the DFCL-APIs setting. In the DECL-APIs setting, our method achieves 0.97x, 0.75x and 0.69x performance of classic CL on CIFAR10, CIFAR100, and MiniImageNet.

{{</citation>}}


### (73/112) Curvature-based Pooling within Graph Neural Networks (Cedric Sanders et al., 2023)

{{<citation>}}

Cedric Sanders, Andreas Roth, Thomas Liebig. (2023)  
**Curvature-based Pooling within Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.16516v1)  

---


**ABSTRACT**  
Over-squashing and over-smoothing are two critical issues, that limit the capabilities of graph neural networks (GNNs). While over-smoothing eliminates the differences between nodes making them indistinguishable, over-squashing refers to the inability of GNNs to propagate information over long distances, as exponentially many node states are squashed into fixed-size representations. Both phenomena share similar causes, as both are largely induced by the graph topology. To mitigate these problems in graph classification tasks, we propose CurvPool, a novel pooling method. CurvPool exploits the notion of curvature of a graph to adaptively identify structures responsible for both over-smoothing and over-squashing. By clustering nodes based on the Balanced Forman curvature, CurvPool constructs a graph with a more suitable structure, allowing deeper models and the combination of distant information. We compare it to other state-of-the-art pooling approaches and establish its competitiveness in terms of classification accuracy, computational complexity, and flexibility. CurvPool outperforms several comparable methods across all considered tasks. The most consistent results are achieved by pooling densely connected clusters using the sum aggregation, as this allows additional information about the size of each pool.

{{</citation>}}


### (74/112) Domain-adaptive Message Passing Graph Neural Network (Xiao Shen et al., 2023)

{{<citation>}}

Xiao Shen, Shirui Pan, Kup-Sze Choi, Xi Zhou. (2023)  
**Domain-adaptive Message Passing Graph Neural Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2308.16470v1)  

---


**ABSTRACT**  
Cross-network node classification (CNNC), which aims to classify nodes in a label-deficient target network by transferring the knowledge from a source network with abundant labels, draws increasing attention recently. To address CNNC, we propose a domain-adaptive message passing graph neural network (DM-GNN), which integrates graph neural network (GNN) with conditional adversarial domain adaptation. DM-GNN is capable of learning informative representations for node classification that are also transferrable across networks. Firstly, a GNN encoder is constructed by dual feature extractors to separate ego-embedding learning from neighbor-embedding learning so as to jointly capture commonality and discrimination between connected nodes. Secondly, a label propagation node classifier is proposed to refine each node's label prediction by combining its own prediction and its neighbors' prediction. In addition, a label-aware propagation scheme is devised for the labeled source network to promote intra-class propagation while avoiding inter-class propagation, thus yielding label-discriminative source embeddings. Thirdly, conditional adversarial domain adaptation is performed to take the neighborhood-refined class-label information into account during adversarial domain adaptation, so that the class-conditional distributions across networks can be better matched. Comparisons with eleven state-of-the-art methods demonstrate the effectiveness of the proposed DM-GNN.

{{</citation>}}


### (75/112) BioCoder: A Benchmark for Bioinformatics Code Generation with Contextual Pragmatic Knowledge (Xiangru Tang et al., 2023)

{{<citation>}}

Xiangru Tang, Bill Qian, Rick Gao, Jiakang Chen, Xinyun Chen, Mark Gerstein. (2023)  
**BioCoder: A Benchmark for Bioinformatics Code Generation with Contextual Pragmatic Knowledge**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: ChatGPT, GPT, T5  
[Paper Link](http://arxiv.org/abs/2308.16458v1)  

---


**ABSTRACT**  
Pre-trained language models like ChatGPT have significantly improved code generation. As these models scale up, there is an increasing need for the output to handle more intricate tasks. Moreover, in bioinformatics, generating functional programs poses additional notable challenges due to the amount of domain knowledge, the need for complicated data operations, and intricate functional dependencies between the operations. Here, we present BioCoder, a benchmark developed to evaluate existing pre-trained models in generating bioinformatics code. In relation to function-code generation, BioCoder covers potential package dependencies, class declarations, and global variables. It incorporates 1026 functions and 1243 methods in Python and Java from GitHub and 253 examples from the Rosalind Project. BioCoder incorporates a fuzz-testing framework for evaluation, and we have applied it to evaluate many models including InCoder, CodeGen, CodeGen2, SantaCoder, StarCoder, StarCoder+, InstructCodeT5+, and ChatGPT. Our detailed analysis of these models emphasizes the importance of domain knowledge, pragmatic code generation, and contextual understanding. Our dataset, benchmark, Docker images, and scripts required for testing are all available at https://github.com/gersteinlab/biocoder.

{{</citation>}}


### (76/112) CktGNN: Circuit Graph Neural Network for Electronic Design Automation (Zehao Dong et al., 2023)

{{<citation>}}

Zehao Dong, Weidong Cao, Muhan Zhang, Dacheng Tao, Yixin Chen, Xuan Zhang. (2023)  
**CktGNN: Circuit Graph Neural Network for Electronic Design Automation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2308.16406v1)  

---


**ABSTRACT**  
The electronic design automation of analog circuits has been a longstanding challenge in the integrated circuit field due to the huge design space and complex design trade-offs among circuit specifications. In the past decades, intensive research efforts have mostly been paid to automate the transistor sizing with a given circuit topology. By recognizing the graph nature of circuits, this paper presents a Circuit Graph Neural Network (CktGNN) that simultaneously automates the circuit topology generation and device sizing based on the encoder-dependent optimization subroutines. Particularly, CktGNN encodes circuit graphs using a two-level GNN framework (of nested GNN) where circuits are represented as combinations of subgraphs in a known subgraph basis. In this way, it significantly improves design efficiency by reducing the number of subgraphs to perform message passing. Nonetheless, another critical roadblock to advancing learning-assisted circuit design automation is a lack of public benchmarks to perform canonical assessment and reproducible research. To tackle the challenge, we introduce Open Circuit Benchmark (OCB), an open-sourced dataset that contains $10$K distinct operational amplifiers with carefully-extracted circuit specifications. OCB is also equipped with communicative circuit generation and evaluation capabilities such that it can help to generalize CktGNN to design various analog circuits by producing corresponding datasets. Experiments on OCB show the extraordinary advantages of CktGNN through representation-based optimization frameworks over other recent powerful GNN baselines and human experts' manual designs. Our work paves the way toward a learning-based open-sourced design automation for analog circuits. Our source code is available at \url{https://github.com/zehao-dong/CktGNN}.

{{</citation>}}


### (77/112) BenchTemp: A General Benchmark for Evaluating Temporal Graph Neural Networks (Qiang Huang et al., 2023)

{{<citation>}}

Qiang Huang, Jiawei Jiang, Xi Susie Rao, Ce Zhang, Zhichao Han, Zitao Zhang, Xin Wang, Yongjun He, Quanqing Xu, Yang Zhao, Chuang Hu, Shuo Shang, Bo Du. (2023)  
**BenchTemp: A General Benchmark for Evaluating Temporal Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.16385v1)  

---


**ABSTRACT**  
To handle graphs in which features or connectivities are evolving over time, a series of temporal graph neural networks (TGNNs) have been proposed. Despite the success of these TGNNs, the previous TGNN evaluations reveal several limitations regarding four critical issues: 1) inconsistent datasets, 2) inconsistent evaluation pipelines, 3) lacking workload diversity, and 4) lacking efficient comparison. Overall, there lacks an empirical study that puts TGNN models onto the same ground and compares them comprehensively. To this end, we propose BenchTemp, a general benchmark for evaluating TGNN models on various workloads. BenchTemp provides a set of benchmark datasets so that different TGNN models can be fairly compared. Further, BenchTemp engineers a standard pipeline that unifies the TGNN evaluation. With BenchTemp, we extensively compare the representative TGNN models on different tasks (e.g., link prediction and node classification) and settings (transductive and inductive), w.r.t. both effectiveness and efficiency metrics. We have made BenchTemp publicly available at https://github.com/qianghuangwhu/benchtemp.

{{</citation>}}


### (78/112) Multi-Objective Decision Transformers for Offline Reinforcement Learning (Abdelghani Ghanem et al., 2023)

{{<citation>}}

Abdelghani Ghanem, Philippe Ciblat, Mounir Ghogho. (2023)  
**Multi-Objective Decision Transformers for Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.16379v1)  

---


**ABSTRACT**  
Offline Reinforcement Learning (RL) is structured to derive policies from static trajectory data without requiring real-time environment interactions. Recent studies have shown the feasibility of framing offline RL as a sequence modeling task, where the sole aim is to predict actions based on prior context using the transformer architecture. However, the limitation of this single task learning approach is its potential to undermine the transformer model's attention mechanism, which should ideally allocate varying attention weights across different tokens in the input context for optimal prediction. To address this, we reformulate offline RL as a multi-objective optimization problem, where the prediction is extended to states and returns. We also highlight a potential flaw in the trajectory representation used for sequence modeling, which could generate inaccuracies when modeling the state and return distributions. This is due to the non-smoothness of the action distribution within the trajectory dictated by the behavioral policy. To mitigate this issue, we introduce action space regions to the trajectory representation. Our experiments on D4RL benchmark locomotion tasks reveal that our propositions allow for more effective utilization of the attention mechanism in the transformer model, resulting in performance that either matches or outperforms current state-of-the art methods.

{{</citation>}}


### (79/112) A Survey on Privacy in Graph Neural Networks: Attacks, Preservation, and Applications (Yi Zhang et al., 2023)

{{<citation>}}

Yi Zhang, Yuying Zhao, Zhaoqing Li, Xueqi Cheng, Yu Wang, Olivera Kotevska, Philip S. Yu, Tyler Derr. (2023)  
**A Survey on Privacy in Graph Neural Networks: Attacks, Preservation, and Applications**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.16375v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have gained significant attention owing to their ability to handle graph-structured data and the improvement in practical applications. However, many of these models prioritize high utility performance, such as accuracy, with a lack of privacy consideration, which is a major concern in modern society where privacy attacks are rampant. To address this issue, researchers have started to develop privacy-preserving GNNs. Despite this progress, there is a lack of a comprehensive overview of the attacks and the techniques for preserving privacy in the graph domain. In this survey, we aim to address this gap by summarizing the attacks on graph data according to the targeted information, categorizing the privacy preservation techniques in GNNs, and reviewing the datasets and applications that could be used for analyzing/solving privacy issues in GNNs. We also outline potential directions for future research in order to build better privacy-preserving GNNs.

{{</citation>}}


### (80/112) SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills (Amey Agrawal et al., 2023)

{{<citation>}}

Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Ramachandran Ramjee. (2023)  
**SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2308.16369v1)  

---


**ABSTRACT**  
Large Language Model (LLM) inference consists of two distinct phases - prefill phase which processes the input prompt and decode phase which generates output tokens autoregressively. While the prefill phase effectively saturates GPU compute at small batch sizes, the decode phase results in low compute utilization as it generates one token at a time per request. The varying prefill and decode times also lead to imbalance across micro-batches when using pipeline parallelism, resulting in further inefficiency due to bubbles.   We present SARATHI to address these challenges. SARATHI employs chunked-prefills, which splits a prefill request into equal sized chunks, and decode-maximal batching, which constructs a batch using a single prefill chunk and populates the remaining slots with decodes. During inference, the prefill chunk saturates GPU compute, while the decode requests 'piggyback' and cost up to an order of magnitude less compared to a decode-only batch. Chunked-prefills allows constructing multiple decode-maximal batches from a single prefill request, maximizing coverage of decodes that can piggyback. Furthermore, the uniform compute design of these batches ameliorates the imbalance between micro-batches, significantly reducing pipeline bubbles.   Our techniques yield significant improvements in inference performance across models and hardware. For the LLaMA-13B model on A6000 GPU, SARATHI improves decode throughput by up to 10x, and accelerates end-to-end throughput by up to 1.33x. For LLaMa-33B on A100 GPU, we achieve 1.25x higher end-to-end-throughput and up to 4.25x higher decode throughput. When used with pipeline parallelism on GPT-3, SARATHI reduces bubbles by 6.29x, resulting in an end-to-end throughput improvement of 1.91x.

{{</citation>}}


## cs.CL (15)



### (81/112) YaRN: Efficient Context Window Extension of Large Language Models (Bowen Peng et al., 2023)

{{<citation>}}

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole. (2023)  
**YaRN: Efficient Context Window Extension of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Embedding, LLaMA, Language Model, Position Embedding  
[Paper Link](http://arxiv.org/abs/2309.00071v1)  

---


**ABSTRACT**  
Rotary Position Embeddings (RoPE) have been shown to effectively encode positional information in transformer-based language models. However, these models fail to generalize past the sequence length they were trained on. We present YaRN (Yet another RoPE extensioN method), a compute-efficient method to extend the context window of such models, requiring 10x less tokens and 2.5x less training steps than previous methods. Using YaRN, we show that LLaMA models can effectively utilize and extrapolate to context lengths much longer than their original pre-training would allow, while also surpassing previous the state-of-the-art at context window extension. In addition, we demonstrate that YaRN exhibits the capability to extrapolate beyond the limited context of a fine-tuning dataset. We publish the checkpoints of Llama 2 7B/13B fine-tuned using YaRN with 64k and 128k context windows at https://github.com/jquesnelle/yarn

{{</citation>}}


### (82/112) The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants (Lucas Bandarkar et al., 2023)

{{<citation>}}

Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, Madian Khabsa. (2023)  
**The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP, NLU  
[Paper Link](http://arxiv.org/abs/2308.16884v1)  

---


**ABSTRACT**  
We present Belebele, a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants. Significantly expanding the language coverage of natural language understanding (NLU) benchmarks, this dataset enables the evaluation of text models in high-, medium-, and low-resource languages. Each question is based on a short passage from the Flores-200 dataset and has four multiple-choice answers. The questions were carefully curated to discriminate between models with different levels of general language comprehension. The English dataset on its own proves difficult enough to challenge state-of-the-art language models. Being fully parallel, this dataset enables direct comparison of model performance across all languages. We use this dataset to evaluate the capabilities of multilingual masked language models (MLMs) and large language models (LLMs). We present extensive results and find that despite significant cross-lingual transfer in English-centric LLMs, much smaller MLMs pretrained on balanced multilingual data still understand far more languages. We also observe that larger vocabulary size and conscious vocabulary construction correlate with better performance on low-resource languages. Overall, Belebele opens up new avenues for evaluating and analyzing the multilingual capabilities of NLP systems.

{{</citation>}}


### (83/112) Simple LLM Prompting is State-of-the-Art for Robust and Multilingual Dialogue Evaluation (John Mendonça et al., 2023)

{{<citation>}}

John Mendonça, Patrícia Pereira, João Paulo Carvalho, Alon Lavie, Isabel Trancoso. (2023)  
**Simple LLM Prompting is State-of-the-Art for Robust and Multilingual Dialogue Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2308.16797v1)  

---


**ABSTRACT**  
Despite significant research effort in the development of automatic dialogue evaluation metrics, little thought is given to evaluating dialogues other than in English. At the same time, ensuring metrics are invariant to semantically similar responses is also an overlooked topic. In order to achieve the desired properties of robustness and multilinguality for dialogue evaluation metrics, we propose a novel framework that takes advantage of the strengths of current evaluation models with the newly-established paradigm of prompting Large Language Models (LLMs). Empirical results show our framework achieves state of the art results in terms of mean Spearman correlation scores across several benchmarks and ranks first place on both the Robust and Multilingual tasks of the DSTC11 Track 4 "Automatic Evaluation Metrics for Open-Domain Dialogue Systems", proving the evaluation capabilities of prompted LLMs.

{{</citation>}}


### (84/112) Towards Multilingual Automatic Dialogue Evaluation (John Mendonça et al., 2023)

{{<citation>}}

John Mendonça, Alon Lavie, Isabel Trancoso. (2023)  
**Towards Multilingual Automatic Dialogue Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Machine Translation, Multilingual  
[Paper Link](http://arxiv.org/abs/2308.16795v1)  

---


**ABSTRACT**  
The main limiting factor in the development of robust multilingual dialogue evaluation metrics is the lack of multilingual data and the limited availability of open sourced multilingual dialogue systems. In this work, we propose a workaround for this lack of data by leveraging a strong multilingual pretrained LLM and augmenting existing English dialogue data using Machine Translation. We empirically show that the naive approach of finetuning a pretrained multilingual encoder model with translated data is insufficient to outperform the strong baseline of finetuning a multilingual model with only source data. Instead, the best approach consists in the careful curation of translated data using MT Quality Estimation metrics, excluding low quality translations that hinder its performance.

{{</citation>}}


### (85/112) Ladder-of-Thought: Using Knowledge as Steps to Elevate Stance Detection (Kairui Hu et al., 2023)

{{<citation>}}

Kairui Hu, Ming Yan, Joey Tianyi Zhou, Ivor W. Tsang, Wen Haw Chong, Yong Keong Yap. (2023)  
**Ladder-of-Thought: Using Knowledge as Steps to Elevate Stance Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Stance Detection  
[Paper Link](http://arxiv.org/abs/2308.16763v1)  

---


**ABSTRACT**  
Chain-of-Thought Prompting (CoT) reinforces the reasoning capabilities of Large Language Models (LLMs) through the generation of intermediate rationales. However, these enhancements predominantly benefit large-scale models, leaving small LMs without significant performance improvements when directly applying CoT. Despite the advanced reasoning capabilities of LLMs, CoT relies primarily on their pre-trained internal knowledge. The external knowledge that is previously unknown to the model remains unexploited. This omission becomes pronounced in tasks such as stance detection, where the external background knowledge plays a pivotal role. Additionally, the large-scale architecture of LLMs inevitably present efficiency challenges during deployment. To address these challenges, we introduce the Ladder-of-Thought (LoT) for stance detection. Grounded in a dual-phase Cascaded Optimization framework, LoT directs the model to incorporate high-quality external knowledge, enhancing the intermediate rationales it generates. These bolstered rationales subsequently serve as the foundation for more precise predictions - akin to how a ladder facilitates reaching elevated goals. LoT achieves a balance between efficiency and accuracy, making it an adaptable and efficient framework for stance detection. Our empirical evaluations underscore LoT's effectiveness, marking a 16% improvement over ChatGPT and a 10% enhancement compared to ChatGPT with CoT.

{{</citation>}}


### (86/112) CReHate: Cross-cultural Re-annotation of English Hate Speech Dataset (Nayeon Lee et al., 2023)

{{<citation>}}

Nayeon Lee, Chani Jung, Junho Myung, Jiho Jin, Juho Kim, Alice Oh. (2023)  
**CReHate: Cross-cultural Re-annotation of English Hate Speech Dataset**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.16705v1)  

---


**ABSTRACT**  
English datasets predominantly reflect the perspectives of certain nationalities, which can lead to cultural biases in models and datasets. This is particularly problematic in tasks heavily influenced by subjectivity, such as hate speech detection. To delve into how individuals from different countries perceive hate speech, we introduce CReHate, a cross-cultural re-annotation of the sampled SBIC dataset. This dataset includes annotations from five distinct countries: Australia, Singapore, South Africa, the United Kingdom, and the United States. Our thorough statistical analysis highlights significant differences based on nationality, with only 59.4% of the samples achieving consensus among all countries. We also introduce a culturally sensitive hate speech classifier via transfer learning, adept at capturing perspectives of different nationalities. These findings underscore the need to re-evaluate certain aspects of NLP research, especially with regard to the nuanced nature of hate speech in the English language.

{{</citation>}}


### (87/112) SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models (Xin Zhang et al., 2023)

{{<citation>}}

Xin Zhang, Dong Zhang, Shimin Li, Yaqian Zhou, Xipeng Qiu. (2023)  
**SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.16692v1)  

---


**ABSTRACT**  
Current speech large language models build upon discrete speech representations, which can be categorized into semantic tokens and acoustic tokens. However, existing speech tokens are not specifically designed for speech language modeling. To assess the suitability of speech tokens for building speech language models, we established the first benchmark, SLMTokBench. Our results indicate that neither semantic nor acoustic tokens are ideal for this purpose. Therefore, we propose SpeechTokenizer, a unified speech tokenizer for speech large language models. SpeechTokenizer adopts the Encoder-Decoder architecture with residual vector quantization (RVQ). Unifying semantic and acoustic tokens, SpeechTokenizer disentangles different aspects of speech information hierarchically across different RVQ layers. Furthermore, We construct a Unified Speech Language Model (USLM) leveraging SpeechTokenizer. Experiments show that SpeechTokenizer performs comparably to EnCodec in speech reconstruction and demonstrates strong performance on the SLMTokBench benchmark. Also, USLM outperforms VALL-E in zero-shot Text-to-Speech tasks. Code and models are available at https://github.com/ZhangXInFD/SpeechTokenizer/.

{{</citation>}}


### (88/112) Using Large Language Models to Automate Category and Trend Analysis of Scientific Articles: An Application in Ophthalmology (Hina Raja et al., 2023)

{{<citation>}}

Hina Raja, Asim Munawar, Mohammad Delsoz, Mohammad Elahi, Yeganeh Madadi, Amr Hassan, Hashem Abu Serhan, Onur Inam, Luis Hermandez, Sang Tran, Wuqas Munir, Alaa Abd-Alrazaq, Hao Chen, SiamakYousefi. (2023)  
**Using Large Language Models to Automate Category and Trend Analysis of Scientific Articles: An Application in Ophthalmology**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model, NLP, Natural Language Processing, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.16688v1)  

---


**ABSTRACT**  
Purpose: In this paper, we present an automated method for article classification, leveraging the power of Large Language Models (LLM). The primary focus is on the field of ophthalmology, but the model is extendable to other fields. Methods: We have developed a model based on Natural Language Processing (NLP) techniques, including advanced LLMs, to process and analyze the textual content of scientific papers. Specifically, we have employed zero-shot learning (ZSL) LLM models and compared against Bidirectional and Auto-Regressive Transformers (BART) and its variants, and Bidirectional Encoder Representations from Transformers (BERT), and its variant such as distilBERT, SciBERT, PubmedBERT, BioBERT. Results: The classification results demonstrate the effectiveness of LLMs in categorizing large number of ophthalmology papers without human intervention. Results: To evalute the LLMs, we compiled a dataset (RenD) of 1000 ocular disease-related articles, which were expertly annotated by a panel of six specialists into 15 distinct categories. The model achieved mean accuracy of 0.86 and mean F1 of 0.85 based on the RenD dataset. Conclusion: The proposed framework achieves notable improvements in both accuracy and efficiency. Its application in the domain of ophthalmology showcases its potential for knowledge organization and retrieval in other domains too. We performed trend analysis that enables the researchers and clinicians to easily categorize and retrieve relevant papers, saving time and effort in literature review and information gathering as well as identification of emerging scientific trends within different disciplines. Moreover, the extendibility of the model to other scientific fields broadens its impact in facilitating research and trend analysis across diverse disciplines.

{{</citation>}}


### (89/112) DictaBERT: A State-of-the-Art BERT Suite for Modern Hebrew (Shaltiel Shmidman et al., 2023)

{{<citation>}}

Shaltiel Shmidman, Avi Shmidman, Moshe Koppel. (2023)  
**DictaBERT: A State-of-the-Art BERT Suite for Modern Hebrew**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NLP  
[Paper Link](http://arxiv.org/abs/2308.16687v1)  

---


**ABSTRACT**  
We present DictaBERT, a new state-of-the-art pre-trained BERT model for modern Hebrew, outperforming existing models on most benchmarks. Additionally, we release two fine-tuned versions of the model, designed to perform two specific foundational tasks in the analysis of Hebrew texts: prefix segmentation and morphological tagging. These fine-tuned models allow any developer to perform prefix segmentation and morphological tagging of a Hebrew sentence with a single call to a HuggingFace model, without the need to integrate any additional libraries or code. In this paper we describe the details of the training as well and the results on the different benchmarks. We release the models to the community, along with sample code demonstrating their use. We release these models as part of our goal to help further research and development in Hebrew NLP.

{{</citation>}}


### (90/112) Unsupervised Text Style Transfer with Deep Generative Models (Zhongtao Jiang et al., 2023)

{{<citation>}}

Zhongtao Jiang, Yuanzhe Zhang, Yiming Ju, Kang Liu. (2023)  
**Unsupervised Text Style Transfer with Deep Generative Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2308.16584v1)  

---


**ABSTRACT**  
We present a general framework for unsupervised text style transfer with deep generative models. The framework models each sentence-label pair in the non-parallel corpus as partially observed from a complete quadruplet which additionally contains two latent codes representing the content and style, respectively. These codes are learned by exploiting dependencies inside the observed data. Then a sentence is transferred by manipulating them. Our framework is able to unify previous embedding and prototype methods as two special forms. It also provides a principled perspective to explain previously proposed techniques in the field such as aligned encoder and adversarial training. We further conduct experiments on three benchmarks. Both automatic and human evaluation results show that our methods achieve better or competitive results compared to several strong baselines.

{{</citation>}}


### (91/112) Thesis Distillation: Investigating The Impact of Bias in NLP Models on Hate Speech Detection (Fatma Elsafoury, 2023)

{{<citation>}}

Fatma Elsafoury. (2023)  
**Thesis Distillation: Investigating The Impact of Bias in NLP Models on Hate Speech Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Hate Speech Detection, NLP  
[Paper Link](http://arxiv.org/abs/2308.16549v1)  

---


**ABSTRACT**  
This paper is a summary of the work in my PhD thesis. In which, I investigate the impact of bias in NLP models on the task of hate speech detection from three perspectives: explainability, offensive stereotyping bias, and fairness. I discuss the main takeaways from my thesis and how they can benefit the broader NLP community. Finally, I discuss important future research directions. The findings of my thesis suggest that bias in NLP models impacts the task of hate speech detection from all three perspectives. And that unless we start incorporating social sciences in studying bias in NLP models, we will not effectively overcome the current limitations of measuring and mitigating bias in NLP models.

{{</citation>}}


### (92/112) Transformer Compression via Subspace Projection (Yuxuan Hu et al., 2023)

{{<citation>}}

Yuxuan Hu, Jing Zhang, Chen Zhao, Cuiping Li, Hong Chen. (2023)  
**Transformer Compression via Subspace Projection**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: BERT, GLUE, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2308.16475v1)  

---


**ABSTRACT**  
We propose TCSP, a novel method for compressing a transformer model by focusing on reducing the hidden size of the model. By projecting the whole transform model into a subspace, we enable matrix operations between the weight matrices in the model and features in a reduced-dimensional space, leading to significant reductions in model parameters and computing resources. To establish this subspace, we decompose the feature matrix, derived from different layers of sampled data instances, into a projection matrix. For evaluation, TCSP is applied to compress T5 and BERT models on the GLUE and SQuAD benchmarks. Experimental results demonstrate that TCSP achieves a compression ratio of 44\% with at most 1.6\% degradation in accuracy, surpassing or matching prior compression methods. Furthermore, TCSP exhibits compatibility with other methods targeting filter and attention head size compression.

{{</citation>}}


### (93/112) Enhancing Subtask Performance of Multi-modal Large Language Model (Yongqiang Zhao et al., 2023)

{{<citation>}}

Yongqiang Zhao, Zhenyu Li, Feng Zhang, Xinhai Xu, Donghong Liu. (2023)  
**Enhancing Subtask Performance of Multi-modal Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.16474v1)  

---


**ABSTRACT**  
Multi-modal Large Language Model (MLLM) refers to a model expanded from a Large Language Model (LLM) that possesses the capability to handle and infer multi-modal data. Current MLLMs typically begin by using LLMs to decompose tasks into multiple subtasks, then employing individual pre-trained models to complete specific subtasks, and ultimately utilizing LLMs to integrate the results of each subtasks to obtain the results of the task. In real-world scenarios, when dealing with large projects, it is common practice to break down the project into smaller sub-projects, with different teams providing corresponding solutions or results. The project owner then decides which solution or result to use, ensuring the best possible outcome for each subtask and, consequently, for the entire project. Inspired by this, this study considers selecting multiple pre-trained models to complete the same subtask. By combining the results from multiple pre-trained models, the optimal subtask result is obtained, enhancing the performance of the MLLM. Specifically, this study first selects multiple pre-trained models focused on the same subtask based on distinct evaluation approaches, and then invokes these models in parallel to process input data and generate corresponding subtask results. Finally, the results from multiple pre-trained models for the same subtask are compared using the LLM, and the best result is chosen as the outcome for that subtask. Extensive experiments are conducted in this study using GPT-4 annotated datasets and human-annotated datasets. The results of various evaluation metrics adequately demonstrate the effectiveness of the proposed approach in this paper.

{{</citation>}}


### (94/112) Link Prediction for Wikipedia Articles as a Natural Language Inference Task (Chau-Thang Phan et al., 2023)

{{<citation>}}

Chau-Thang Phan, Quoc-Nam Nguyen, Kiet Van Nguyen. (2023)  
**Link Prediction for Wikipedia Articles as a Natural Language Inference Task**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLI, NLP, Natural Language Inference  
[Paper Link](http://arxiv.org/abs/2308.16469v1)  

---


**ABSTRACT**  
Link prediction task is vital to automatically understanding the structure of large knowledge bases. In this paper, we present our system to solve this task at the Data Science and Advanced Analytics 2023 Competition "Efficient and Effective Link Prediction" (DSAA-2023 Competition) with a corpus containing 948,233 training and 238,265 for public testing. This paper introduces an approach to link prediction in Wikipedia articles by formulating it as a natural language inference (NLI) task. Drawing inspiration from recent advancements in natural language processing and understanding, we cast link prediction as an NLI task, wherein the presence of a link between two articles is treated as a premise, and the task is to determine whether this premise holds based on the information presented in the articles. We implemented our system based on the Sentence Pair Classification for Link Prediction for the Wikipedia Articles task. Our system achieved 0.99996 Macro F1-score and 1.00000 Macro F1-score for the public and private test sets, respectively. Our team UIT-NLP ranked 3rd in performance on the private test set, equal to the scores of the first and second places. Our code is publicly for research purposes.

{{</citation>}}


### (95/112) Knowledge Distillation from Non-streaming to Streaming ASR Encoder using Auxiliary Non-streaming Layer (Kyuhong Shim et al., 2023)

{{<citation>}}

Kyuhong Shim, Jinkyu Lee, Simyung Chang, Kyuwoong Hwang. (2023)  
**Knowledge Distillation from Non-streaming to Streaming ASR Encoder using Auxiliary Non-streaming Layer**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-AS  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2308.16415v1)  

---


**ABSTRACT**  
Streaming automatic speech recognition (ASR) models are restricted from accessing future context, which results in worse performance compared to the non-streaming models. To improve the performance of streaming ASR, knowledge distillation (KD) from the non-streaming to streaming model has been studied, mainly focusing on aligning the output token probabilities. In this paper, we propose a layer-to-layer KD from the teacher encoder to the student encoder. To ensure that features are extracted using the same context, we insert auxiliary non-streaming branches to the student and perform KD from the non-streaming teacher layer to the non-streaming auxiliary layer. We design a special KD loss that leverages the autoregressive predictive coding (APC) mechanism to encourage the streaming model to predict unseen future contexts. Experimental results show that the proposed method can significantly reduce the word error rate compared to previous token probability distillation methods.

{{</citation>}}


## cs.RO (3)



### (96/112) GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields (Yanjie Ze et al., 2023)

{{<citation>}}

Yanjie Ze, Ge Yan, Yueh-Hua Wu, Annabella Macaluso, Yuying Ge, Jianglong Ye, Nicklas Hansen, Li Erran Li, Xiaolong Wang. (2023)  
**GNFactor: Multi-Task Real Robot Learning with Generalizable Neural Feature Fields**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.16891v2)  

---


**ABSTRACT**  
It is a long-standing problem in robotics to develop agents capable of executing diverse manipulation tasks from visual observations in unstructured real-world environments. To achieve this goal, the robot needs to have a comprehensive understanding of the 3D structure and semantics of the scene. In this work, we present $\textbf{GNFactor}$, a visual behavior cloning agent for multi-task robotic manipulation with $\textbf{G}$eneralizable $\textbf{N}$eural feature $\textbf{F}$ields. GNFactor jointly optimizes a generalizable neural field (GNF) as a reconstruction module and a Perceiver Transformer as a decision-making module, leveraging a shared deep 3D voxel representation. To incorporate semantics in 3D, the reconstruction module utilizes a vision-language foundation model ($\textit{e.g.}$, Stable Diffusion) to distill rich semantic information into the deep 3D voxel. We evaluate GNFactor on 3 real robot tasks and perform detailed ablations on 10 RLBench tasks with a limited number of demonstrations. We observe a substantial improvement of GNFactor over current state-of-the-art methods in seen and unseen tasks, demonstrating the strong generalization ability of GNFactor. Our project website is https://yanjieze.com/GNFactor/ .

{{</citation>}}


### (97/112) Developing Social Robots with Empathetic Non-Verbal Cues Using Large Language Models (Yoon Kyung Lee et al., 2023)

{{<citation>}}

Yoon Kyung Lee, Yoonwon Jung, Gyuyi Kang, Sowon Hahn. (2023)  
**Developing Social Robots with Empathetic Non-Verbal Cues Using Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-HC, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.16529v1)  

---


**ABSTRACT**  
We propose augmenting the empathetic capacities of social robots by integrating non-verbal cues. Our primary contribution is the design and labeling of four types of empathetic non-verbal cues, abbreviated as SAFE: Speech, Action (gesture), Facial expression, and Emotion, in a social robot. These cues are generated using a Large Language Model (LLM). We developed an LLM-based conversational system for the robot and assessed its alignment with social cues as defined by human counselors. Preliminary results show distinct patterns in the robot's responses, such as a preference for calm and positive social emotions like 'joy' and 'lively', and frequent nodding gestures. Despite these tendencies, our approach has led to the development of a social robot capable of context-aware and more authentic interactions. Our work lays the groundwork for future studies on human-robot interactions, emphasizing the essential role of both verbal and non-verbal cues in creating social and empathetic robots.

{{</citation>}}


### (98/112) A Policy Adaptation Method for Implicit Multitask Reinforcement Learning Problems (Satoshi Yamamori et al., 2023)

{{<citation>}}

Satoshi Yamamori, Jun Morimoto. (2023)  
**A Policy Adaptation Method for Implicit Multitask Reinforcement Learning Problems**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.16471v1)  

---


**ABSTRACT**  
In dynamic motion generation tasks, including contact and collisions, small changes in policy parameters can lead to extremely different returns. For example, in soccer, the ball can fly in completely different directions with a similar heading motion by slightly changing the hitting position or the force applied to the ball or when the friction of the ball varies. However, it is difficult to imagine that completely different skills are needed for heading a ball in different directions. In this study, we proposed a multitask reinforcement learning algorithm for adapting a policy to implicit changes in goals or environments in a single motion category with different reward functions or physical parameters of the environment. We evaluated the proposed method on the ball heading task using a monopod robot model. The results showed that the proposed method can adapt to implicit changes in the goal positions or the coefficients of restitution of the ball, whereas the standard domain randomization approach cannot cope with different task settings.

{{</citation>}}


## eess.IV (7)



### (99/112) Self-pruning Graph Neural Network for Predicting Inflammatory Disease Activity in Multiple Sclerosis from Brain MR Images (Chinmay Prabhakar et al., 2023)

{{<citation>}}

Chinmay Prabhakar, Hongwei Bran Li, Johannes C. Paetzold, Timo Loehr, Chen Niu, Mark Mühlau, Daniel Rueckert, Benedikt Wiestler, Bjoern Menze. (2023)  
**Self-pruning Graph Neural Network for Predicting Inflammatory Disease Activity in Multiple Sclerosis from Brain MR Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2308.16863v1)  

---


**ABSTRACT**  
Multiple Sclerosis (MS) is a severe neurological disease characterized by inflammatory lesions in the central nervous system. Hence, predicting inflammatory disease activity is crucial for disease assessment and treatment. However, MS lesions can occur throughout the brain and vary in shape, size and total count among patients. The high variance in lesion load and locations makes it challenging for machine learning methods to learn a globally effective representation of whole-brain MRI scans to assess and predict disease. Technically it is non-trivial to incorporate essential biomarkers such as lesion load or spatial proximity. Our work represents the first attempt to utilize graph neural networks (GNN) to aggregate these biomarkers for a novel global representation. We propose a two-stage MS inflammatory disease activity prediction approach. First, a 3D segmentation network detects lesions, and a self-supervised algorithm extracts their image features. Second, the detected lesions are used to build a patient graph. The lesions act as nodes in the graph and are initialized with image features extracted in the first stage. Finally, the lesions are connected based on their spatial proximity and the inflammatory disease activity prediction is formulated as a graph classification task. Furthermore, we propose a self-pruning strategy to auto-select the most critical lesions for prediction. Our proposed method outperforms the existing baseline by a large margin (AUCs of 0.67 vs. 0.61 and 0.66 vs. 0.60 for one-year and two-year inflammatory disease activity, respectively). Finally, our proposed method enjoys inherent explainability by assigning an importance score to each lesion for the overall prediction. Code is available at https://github.com/chinmay5/ms_ida.git

{{</citation>}}


### (100/112) A Sequential Framework for Detection and Classification of Abnormal Teeth in Panoramic X-rays (Tudor Dascalu et al., 2023)

{{<citation>}}

Tudor Dascalu, Shaqayeq Ramezanzade, Azam Bakhshandeh, Lars Bjorndal, Bulat Ibragimov. (2023)  
**A Sequential Framework for Detection and Classification of Abnormal Teeth in Panoramic X-rays**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.00027v1)  

---


**ABSTRACT**  
This paper describes our solution for the Dental Enumeration and Diagnosis on Panoramic X-rays Challenge at MICCAI 2023. Our approach consists of a multi-step framework tailored to the task of detecting and classifying abnormal teeth. The solution includes three sequential stages: dental instance detection, healthy instance filtering, and abnormal instance classification. In the first stage, we employed a Faster-RCNN model for detecting and identifying teeth. In subsequent stages, we designed a model that merged the encoding pathway of a pretrained U-net, optimized for dental lesion detection, with the Vgg16 architecture. The resulting model was first used for filtering out healthy teeth. Then, any identified abnormal teeth were categorized, potentially falling into one or more of the following conditions: embeddded, periapical lesion, caries, deep caries. The model performing dental instance detection achieved an AP score of 0.49. The model responsible for identifying healthy teeth attained an F1 score of 0.71. Meanwhile, the model trained for multi-label dental disease classification achieved an F1 score of 0.76. The code is available at https://github.com/tudordascalu/2d-teeth-detection-challenge.

{{</citation>}}


### (101/112) Towards Optimal Patch Size in Vision Transformers for Tumor Segmentation (Ramtin Mojtahedi et al., 2023)

{{<citation>}}

Ramtin Mojtahedi, Mohammad Hamghalam, Richard K. G. Do, Amber L. Simpson. (2023)  
**Towards Optimal Patch Size in Vision Transformers for Tumor Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.16598v1)  

---


**ABSTRACT**  
Detection of tumors in metastatic colorectal cancer (mCRC) plays an essential role in the early diagnosis and treatment of liver cancer. Deep learning models backboned by fully convolutional neural networks (FCNNs) have become the dominant model for segmenting 3D computerized tomography (CT) scans. However, since their convolution layers suffer from limited kernel size, they are not able to capture long-range dependencies and global context. To tackle this restriction, vision transformers have been introduced to solve FCNN's locality of receptive fields. Although transformers can capture long-range features, their segmentation performance decreases with various tumor sizes due to the model sensitivity to the input patch size. While finding an optimal patch size improves the performance of vision transformer-based models on segmentation tasks, it is a time-consuming and challenging procedure. This paper proposes a technique to select the vision transformer's optimal input multi-resolution image patch size based on the average volume size of metastasis lesions. We further validated our suggested framework using a transfer-learning technique, demonstrating that the highest Dice similarity coefficient (DSC) performance was obtained by pre-training on training data with a larger tumour volume using the suggested ideal patch size and then training with a smaller one. We experimentally evaluate this idea through pre-training our model on a multi-resolution public dataset. Our model showed consistent and improved results when applied to our private multi-resolution mCRC dataset with a smaller average tumor volume. This study lays the groundwork for optimizing semantic segmentation of small objects using vision transformers. The implementation source code is available at:https://github.com/Ramtin-Mojtahedi/OVTPS.

{{</citation>}}


### (102/112) Dual-Decoder Consistency via Pseudo-Labels Guided Data Augmentation for Semi-Supervised Medical Image Segmentation (Yuanbin Chen et al., 2023)

{{<citation>}}

Yuanbin Chen, Tao Wang, Hui Tang, Longxuan Zhao, Ruige Zong, Tao Tan, Xinlin Zhang, Tong Tong. (2023)  
**Dual-Decoder Consistency via Pseudo-Labels Guided Data Augmentation for Semi-Supervised Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Augmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.16573v1)  

---


**ABSTRACT**  
Medical image segmentation methods often rely on fully supervised approaches to achieve excellent performance, which is contingent upon having an extensive set of labeled images for training. However, annotating medical images is both expensive and time-consuming. Semi-supervised learning offers a solution by leveraging numerous unlabeled images alongside a limited set of annotated ones. In this paper, we introduce a semi-supervised medical image segmentation method based on the mean-teacher model, referred to as Dual-Decoder Consistency via Pseudo-Labels Guided Data Augmentation (DCPA). This method combines consistency regularization, pseudo-labels, and data augmentation to enhance the efficacy of semi-supervised segmentation. Firstly, the proposed model comprises both student and teacher models with a shared encoder and two distinct decoders employing different up-sampling strategies. Minimizing the output discrepancy between decoders enforces the generation of consistent representations, serving as regularization during student model training. Secondly, we introduce mixup operations to blend unlabeled data with labeled data, creating mixed data and thereby achieving data augmentation. Lastly, pseudo-labels are generated by the teacher model and utilized as labels for mixed data to compute unsupervised loss. We compare the segmentation results of the DCPA model with six state-of-the-art semi-supervised methods on three publicly available medical datasets. Beyond classical 10\% and 20\% semi-supervised settings, we investigate performance with less supervision (5\% labeled data). Experimental outcomes demonstrate that our approach consistently outperforms existing semi-supervised medical image segmentation methods across the three semi-supervised settings.

{{</citation>}}


### (103/112) MoMA: Momentum Contrastive Learning with Multi-head Attention-based Knowledge Distillation for Histopathology Image Analysis (Trinh Thi Le Vuong et al., 2023)

{{<citation>}}

Trinh Thi Le Vuong, Jin Tae Kwak. (2023)  
**MoMA: Momentum Contrastive Learning with Multi-head Attention-based Knowledge Distillation for Histopathology Image Analysis**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention, Contrastive Learning, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2308.16561v1)  

---


**ABSTRACT**  
There is no doubt that advanced artificial intelligence models and high quality data are the keys to success in developing computational pathology tools. Although the overall volume of pathology data keeps increasing, a lack of quality data is a common issue when it comes to a specific task due to several reasons including privacy and ethical issues with patient data. In this work, we propose to exploit knowledge distillation, i.e., utilize the existing model to learn a new, target model, to overcome such issues in computational pathology. Specifically, we employ a student-teacher framework to learn a target model from a pre-trained, teacher model without direct access to source data and distill relevant knowledge via momentum contrastive learning with multi-head attention mechanism, which provides consistent and context-aware feature representations. This enables the target model to assimilate informative representations of the teacher model while seamlessly adapting to the unique nuances of the target data. The proposed method is rigorously evaluated across different scenarios where the teacher model was trained on the same, relevant, and irrelevant classification tasks with the target model. Experimental results demonstrate the accuracy and robustness of our approach in transferring knowledge to different domains and tasks, outperforming other related methods. Moreover, the results provide a guideline on the learning strategy for different types of tasks and scenarios in computational pathology. Code is available at: \url{https://github.com/trinhvg/MoMA}.

{{</citation>}}


### (104/112) Object Detection for Caries or Pit and Fissure Sealing Requirement in Children's First Permanent Molars (Chenyao Jiang et al., 2023)

{{<citation>}}

Chenyao Jiang, Shiyao Zhai, Hengrui Song, Yuqing Ma, Yachen Fan, Yancheng Fang, Dongmei Yu, Canyang Zhang, Sanyang Han, Runming Wang, Yong Liu, Jianbo Li, Peiwu Qin. (2023)  
**Object Detection for Caries or Pit and Fissure Sealing Requirement in Children's First Permanent Molars**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.16551v1)  

---


**ABSTRACT**  
Dental caries is one of the most common oral diseases that, if left untreated, can lead to a variety of oral problems. It mainly occurs inside the pits and fissures on the occlusal/buccal/palatal surfaces of molars and children are a high-risk group for pit and fissure caries in permanent molars. Pit and fissure sealing is one of the most effective methods that is widely used in prevention of pit and fissure caries. However, current detection of pits and fissures or caries depends primarily on the experienced dentists, which ordinary parents do not have, and children may miss the remedial treatment without timely detection. To address this issue, we present a method to autodetect caries and pit and fissure sealing requirements using oral photos taken by smartphones. We use the YOLOv5 and YOLOX models and adopt a tiling strategy to reduce information loss during image pre-processing. The best result for YOLOXs model with tiling strategy is 72.3 mAP.5, while the best result without tiling strategy is 71.2. YOLOv5s6 model with/without tiling attains 70.9/67.9 mAP.5, respectively. We deploy the pre-trained network to mobile devices as a WeChat applet, allowing in-home detection by parents or children guardian.

{{</citation>}}


### (105/112) Improving Multiple Sclerosis Lesion Segmentation Across Clinical Sites: A Federated Learning Approach with Noise-Resilient Training (Lei Bai et al., 2023)

{{<citation>}}

Lei Bai, Dongang Wang, Michael Barnett, Mariano Cabezas, Weidong Cai, Fernando Calamante, Kain Kyle, Dongnan Liu, Linda Ly, Aria Nguyen, Chun-Chien Shieh, Ryan Sullivan, Hengrui Wang, Geng Zhan, Wanli Ouyang, Chenyu Wang. (2023)  
**Improving Multiple Sclerosis Lesion Segmentation Across Clinical Sites: A Federated Learning Approach with Noise-Resilient Training**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-DC, eess-IV, eess.IV  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2308.16376v1)  

---


**ABSTRACT**  
Accurately measuring the evolution of Multiple Sclerosis (MS) with magnetic resonance imaging (MRI) critically informs understanding of disease progression and helps to direct therapeutic strategy. Deep learning models have shown promise for automatically segmenting MS lesions, but the scarcity of accurately annotated data hinders progress in this area. Obtaining sufficient data from a single clinical site is challenging and does not address the heterogeneous need for model robustness. Conversely, the collection of data from multiple sites introduces data privacy concerns and potential label noise due to varying annotation standards. To address this dilemma, we explore the use of the federated learning framework while considering label noise. Our approach enables collaboration among multiple clinical sites without compromising data privacy under a federated learning paradigm that incorporates a noise-robust training strategy based on label correction. Specifically, we introduce a Decoupled Hard Label Correction (DHLC) strategy that considers the imbalanced distribution and fuzzy boundaries of MS lesions, enabling the correction of false annotations based on prediction confidence. We also introduce a Centrally Enhanced Label Correction (CELC) strategy, which leverages the aggregated central model as a correction teacher for all sites, enhancing the reliability of the correction process. Extensive experiments conducted on two multi-site datasets demonstrate the effectiveness and robustness of our proposed methods, indicating their potential for clinical applications in multi-site collaborations.

{{</citation>}}


## cs.IR (2)



### (106/112) Co-evolving Vector Quantization for ID-based Recommendation (Qijiong Liu et al., 2023)

{{<citation>}}

Qijiong Liu, Jiaren Xiao, Lu Fan, Jieming Zhu, Xiao-Ming Wu. (2023)  
**Co-evolving Vector Quantization for ID-based Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2308.16761v1)  

---


**ABSTRACT**  
Category information plays a crucial role in enhancing the quality and personalization of recommendations. Nevertheless, the availability of item category information is not consistently present, particularly in the context of ID-based recommendations. In this work, we propose an alternative approach to automatically learn and generate entity (i.e., user and item) categorical information at different levels of granularity, specifically for ID-based recommendation. Specifically, we devise a co-evolving vector quantization framework, namely COVE, which enables the simultaneous learning and refinement of code representation and entity embedding in an end-to-end manner, starting from the randomly initialized states. With its high adaptability, COVE can be easily integrated into existing recommendation models. We validate the effectiveness of COVE on various recommendation tasks including list completion, collaborative filtering, and click-through rate prediction, across different recommendation models. We will publish the code and data for other researchers to reproduce our work.

{{</citation>}}


### (107/112) Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations (Xu Huang et al., 2023)

{{<citation>}}

Xu Huang, Jianxun Lian, Yuxuan Lei, Jing Yao, Defu Lian, Xing Xie. (2023)  
**Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.16505v2)  

---


**ABSTRACT**  
Recommender models excel at providing domain-specific item recommendations by leveraging extensive user behavior data. Despite their ability to act as lightweight domain experts, they struggle to perform versatile tasks such as providing explanations and engaging in conversations. On the other hand, large language models (LLMs) represent a significant step towards artificial general intelligence, showcasing remarkable capabilities in instruction comprehension, commonsense reasoning, and human interaction. However, LLMs lack the knowledge of domain-specific item catalogs and behavioral patterns, particularly in areas that diverge from general world knowledge, such as online e-commerce. Finetuning LLMs for each domain is neither economic nor efficient.   In this paper, we bridge the gap between recommender models and LLMs, combining their respective strengths to create a versatile and interactive recommender system. We introduce an efficient framework called InteRecAgent, which employs LLMs as the brain and recommender models as tools. We first outline a minimal set of essential tools required to transform LLMs into InteRecAgent. We then propose an efficient workflow within InteRecAgent for task execution, incorporating key components such as a memory bus, dynamic demonstration-augmented task planning, and reflection. InteRecAgent enables traditional recommender systems, such as those ID-based matrix factorization models, to become interactive systems with a natural language interface through the integration of LLMs. Experimental results on several public datasets show that InteRecAgent achieves satisfying performance as a conversational recommender system, outperforming general-purpose LLMs.

{{</citation>}}


## q-bio.QM (1)



### (108/112) The Smart Data Extractor, a Clinician Friendly Solution to Accelerate and Improve the Data Collection During Clinical Trials (Sophie Quennelle et al., 2023)

{{<citation>}}

Sophie Quennelle, Maxime Douillet, Lisa Friedlander, Olivia Boyer, Anita Burgun, Antoine Neuraz, Nicolas Garcelon. (2023)  
**The Smart Data Extractor, a Clinician Friendly Solution to Accelerate and Improve the Data Collection During Clinical Trials**  

---
Primary Category: q-bio.QM  
Categories: cs-CL, q-bio-QM, q-bio.QM  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2308.16537v1)  

---


**ABSTRACT**  
In medical research, the traditional way to collect data, i.e. browsing patient files, has been proven to induce bias, errors, human labor and costs. We propose a semi-automated system able to extract every type of data, including notes. The Smart Data Extractor pre-populates clinic research forms by following rules. We performed a cross-testing experiment to compare semi-automated to manual data collection. 20 target items had to be collected for 79 patients. The average time to complete one form was 6'81'' for manual data collection and 3'22'' with the Smart Data Extractor. There were also more mistakes during manual data collection (163 for the whole cohort) than with the Smart Data Extractor (46 for the whole cohort). We present an easy to use, understandable and agile solution to fill out clinical research forms. It reduces human effort and provides higher quality data, avoiding data re-entry and fatigue induced errors.

{{</citation>}}


## math.OC (1)



### (109/112) Last Mile Delivery with Drones and Sharing Economy (Mehdi Behroozi et al., 2023)

{{<citation>}}

Mehdi Behroozi, Dinghao Ma. (2023)  
**Last Mile Delivery with Drones and Sharing Economy**  

---
Primary Category: math.OC  
Categories: cs-CG, math-OC, math.OC  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2308.16408v1)  

---


**ABSTRACT**  
We consider a combined system of regular delivery trucks and crowdsourced drones, available via a sharing economy platform, to provide a technology-assisted crowd-based last-mile delivery experience. We develop analytical models and methods for a system in which package delivery is performed by a big truck carrying many packages to a neighborhood or a town in a metropolitan area and then the packages are assigned to crowdsourced drone operators to deliver them to their final destinations. We develop several optimization models for various cases of the problem and use a combination of heuristic algorithms to solve this NP-hard problem. Finally, we present computational results for the models and algorithms, and conduct an exhaustive sensitivity analysis to check the influence of different parameters and assumptions on the final results. We also provide extensive managerial insights on the benefits of our proposed model if implemented in practice.

{{</citation>}}


## cs.HC (3)



### (110/112) Balancing between the Local and Global Structures (LGS) in Graph Embedding (Jacob Miller et al., 2023)

{{<citation>}}

Jacob Miller, Vahan Huroyan, Stephen Kobourov. (2023)  
**Balancing between the Local and Global Structures (LGS) in Graph Embedding**  

---
Primary Category: cs.HC  
Categories: cs-CG, cs-HC, cs-LG, cs.HC  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.16403v1)  

---


**ABSTRACT**  
We present a method for balancing between the Local and Global Structures (LGS) in graph embedding, via a tunable parameter. Some embedding methods aim to capture global structures, while others attempt to preserve local neighborhoods. Few methods attempt to do both, and it is not always possible to capture well both local and global information in two dimensions, which is where most graph drawing live. The choice of using a local or a global embedding for visualization depends not only on the task but also on the structure of the underlying data, which may not be known in advance. For a given graph, LGS aims to find a good balance between the local and global structure to preserve. We evaluate the performance of LGS with synthetic and real-world datasets and our results indicate that it is competitive with the state-of-the-art methods, using established quality metrics such as stress and neighborhood preservation. We introduce a novel quality metric, cluster distance preservation, to assess intermediate structure capture. All source-code, datasets, experiments and analysis are available online.

{{</citation>}}


### (111/112) Science Communications for Explainable Artificial Intelligence (Simon Hudson et al., 2023)

{{<citation>}}

Simon Hudson, Matija Franklin. (2023)  
**Science Communications for Explainable Artificial Intelligence**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.16377v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) has a communication problem. XAI methods have been used to make AI more understandable and helped resolve some of the transparency issues that inhibit AI's broader usability. However, user evaluation studies reveal that the often numerical explanations provided by XAI methods have not always been effective for many types of users of AI systems. This article aims to adapt the major communications models from Science Communications into a framework for practitioners to understand, influence, and integrate the context of audiences both for their communications supporting AI literacy in the public and in designing XAI systems that are more adaptive to different users.

{{</citation>}}


### (112/112) Accused: How students respond to allegations of using ChatGPT on assessments (Tim Gorichanaz, 2023)

{{<citation>}}

Tim Gorichanaz. (2023)  
**Accused: How students respond to allegations of using ChatGPT on assessments**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.16374v1)  

---


**ABSTRACT**  
This study explores student responses to allegations of cheating using ChatGPT, a popular software platform that can be used to generate grammatical and broadly correct text on virtually any topic. Forty-nine posts and the ensuing discussions were collected from Reddit, an online discussion forum, in which students shared their experiences of being accused (the majority falsely) and discussed how to navigate their situations. A thematic analysis was conducted with this material, and five themes were discerned: a legalistic stance, involving argument strategy and evidence gathering; the societal role of higher education as a high-stakes gatekeeper; the vicissitudes of trust in students vs. technology; questions of what constitutes cheating; and the need to rethink assessment. The findings from this study will help instructors and institutions to create more meaningful assessments in the age of AI and develop guidelines for student use of ChatGPT and other AI tools.

{{</citation>}}
