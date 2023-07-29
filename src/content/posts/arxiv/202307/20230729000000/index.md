---
draft: false
title: "arXiv @ 2023.07.29"
date: 2023-07-29
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.29"
    identifier: arxiv_20230729
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (27)](#cscv-27)
- [cs.MM (1)](#csmm-1)
- [cs.IR (2)](#csir-2)
- [cs.CL (15)](#cscl-15)
- [cs.SE (2)](#csse-2)
- [cs.LG (5)](#cslg-5)
- [cs.SI (1)](#cssi-1)
- [eess.IV (4)](#eessiv-4)
- [cs.CR (2)](#cscr-2)
- [eess.AS (1)](#eessas-1)
- [cs.AI (1)](#csai-1)
- [cs.IT (1)](#csit-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.RO (1)](#csro-1)

## cs.CV (27)



### (1/64) To Adapt or Not to Adapt? Real-Time Adaptation for Semantic Segmentation (Marc Botet Colomer et al., 2023)

{{<citation>}}

Marc Botet Colomer, Pier Luigi Dovesi, Theodoros Panagiotakopoulos, Joao Frederico Carvalho, Linus Härenstam-Nielsen, Hossein Azizpour, Hedvig Kjellström, Daniel Cremers, Matteo Poggi. (2023)  
**To Adapt or Not to Adapt? Real-Time Adaptation for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.15063v1)  

---


**ABSTRACT**  
The goal of Online Domain Adaptation for semantic segmentation is to handle unforeseeable domain changes that occur during deployment, like sudden weather events. However, the high computational costs associated with brute-force adaptation make this paradigm unfeasible for real-world applications. In this paper we propose HAMLET, a Hardware-Aware Modular Least Expensive Training framework for real-time domain adaptation. Our approach includes a hardware-aware back-propagation orchestration agent (HAMT) and a dedicated domain-shift detector that enables active control over when and how the model is adapted (LT). Thanks to these advancements, our approach is capable of performing semantic segmentation while simultaneously adapting at more than 29FPS on a single consumer-grade GPU. Our framework's encouraging accuracy and speed trade-off is demonstrated on OnDA and SHIFT benchmarks through experimental results.

{{</citation>}}


### (2/64) Regularized Mask Tuning: Uncovering Hidden Knowledge in Pre-trained Vision-Language Models (Kecheng Zheng et al., 2023)

{{<citation>}}

Kecheng Zheng, Wei Wu, Ruili Feng, Kai Zhu, Jiawei Liu, Deli Zhao, Zheng-Jun Zha, Wei Chen, Yujun Shen. (2023)  
**Regularized Mask Tuning: Uncovering Hidden Knowledge in Pre-trained Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.15049v1)  

---


**ABSTRACT**  
Prompt tuning and adapter tuning have shown great potential in transferring pre-trained vision-language models (VLMs) to various downstream tasks. In this work, we design a new type of tuning method, termed as regularized mask tuning, which masks the network parameters through a learnable selection. Inspired by neural pathways, we argue that the knowledge required by a downstream task already exists in the pre-trained weights but just gets concealed in the upstream pre-training stage. To bring the useful knowledge back into light, we first identify a set of parameters that are important to a given downstream task, then attach a binary mask to each parameter, and finally optimize these masks on the downstream data with the parameters frozen. When updating the mask, we introduce a novel gradient dropout strategy to regularize the parameter selection, in order to prevent the model from forgetting old knowledge and overfitting the downstream data. Experimental results on 11 datasets demonstrate the consistent superiority of our method over previous alternatives. It is noteworthy that we manage to deliver 18.73% performance improvement compared to the zero-shot CLIP via masking an average of only 2.56% parameters. Furthermore, our method is synergistic with most existing parameter-efficient tuning methods and can boost the performance on top of them. Project page can be found here (https://wuw2019.github.io/RMT/).

{{</citation>}}


### (3/64) A Transformer-based Approach for Arabic Offline Handwritten Text Recognition (Saleh Momeni et al., 2023)

{{<citation>}}

Saleh Momeni, Bagher BabaAli. (2023)  
**A Transformer-based Approach for Arabic Offline Handwritten Text Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.15045v1)  

---


**ABSTRACT**  
Handwriting recognition is a challenging and critical problem in the fields of pattern recognition and machine learning, with applications spanning a wide range of domains. In this paper, we focus on the specific issue of recognizing offline Arabic handwritten text. Existing approaches typically utilize a combination of convolutional neural networks for image feature extraction and recurrent neural networks for temporal modeling, with connectionist temporal classification used for text generation. However, these methods suffer from a lack of parallelization due to the sequential nature of recurrent neural networks. Furthermore, these models cannot account for linguistic rules, necessitating the use of an external language model in the post-processing stage to boost accuracy. To overcome these issues, we introduce two alternative architectures, namely the Transformer Transducer and the standard sequence-to-sequence Transformer, and compare their performance in terms of accuracy and speed. Our approach can model language dependencies and relies only on the attention mechanism, thereby making it more parallelizable and less complex. We employ pre-trained Transformers for both image understanding and language modeling. Our evaluation on the Arabic KHATT dataset demonstrates that our proposed method outperforms the current state-of-the-art approaches for recognizing offline Arabic handwritten text.

{{</citation>}}


### (4/64) Self-Supervised Graph Transformer for Deepfake Detection (Aminollah Khormali et al., 2023)

{{<citation>}}

Aminollah Khormali, Jiann-Shiun Yuan. (2023)  
**Self-Supervised Graph Transformer for Deepfake Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2307.15019v1)  

---


**ABSTRACT**  
Deepfake detection methods have shown promising results in recognizing forgeries within a given dataset, where training and testing take place on the in-distribution dataset. However, their performance deteriorates significantly when presented with unseen samples. As a result, a reliable deepfake detection system must remain impartial to forgery types, appearance, and quality for guaranteed generalizable detection performance. Despite various attempts to enhance cross-dataset generalization, the problem remains challenging, particularly when testing against common post-processing perturbations, such as video compression or blur. Hence, this study introduces a deepfake detection framework, leveraging a self-supervised pre-training model that delivers exceptional generalization ability, withstanding common corruptions and enabling feature explainability. The framework comprises three key components: a feature extractor based on vision Transformer architecture that is pre-trained via self-supervised contrastive learning methodology, a graph convolution network coupled with a Transformer discriminator, and a graph Transformer relevancy map that provides a better understanding of manipulated regions and further explains the model's decision. To assess the effectiveness of the proposed framework, several challenging experiments are conducted, including in-data distribution performance, cross-dataset, cross-manipulation generalization, and robustness against common post-production perturbations. The results achieved demonstrate the remarkable effectiveness of the proposed deepfake detection framework, surpassing the current state-of-the-art approaches.

{{</citation>}}


### (5/64) How Good is Google Bard's Visual Understanding? An Empirical Study on Open Challenges (Haotong Qin et al., 2023)

{{<citation>}}

Haotong Qin, Ge-Peng Ji, Salman Khan, Deng-Ping Fan, Fahad Shahbaz Khan, Luc Van Gool. (2023)  
**How Good is Google Bard's Visual Understanding? An Empirical Study on Open Challenges**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, ChatGPT, GPT, Google  
[Paper Link](http://arxiv.org/abs/2307.15016v1)  

---


**ABSTRACT**  
Google's Bard has emerged as a formidable competitor to OpenAI's ChatGPT in the field of conversational AI. Notably, Bard has recently been updated to handle visual inputs alongside text prompts during conversations. Given Bard's impressive track record in handling textual inputs, we explore its capabilities in understanding and interpreting visual data (images) conditioned by text questions. This exploration holds the potential to unveil new insights and challenges for Bard and other forthcoming multi-modal Generative models, especially in addressing complex computer vision problems that demand accurate visual and language understanding. Specifically, in this study, we focus on 15 diverse task scenarios encompassing regular, camouflaged, medical, under-water and remote sensing data to comprehensively evaluate Bard's performance. Our primary finding indicates that Bard still struggles in these vision scenarios, highlighting the significant gap in vision-based understanding that needs to be bridged in future developments. We expect that this empirical study will prove valuable in advancing future models, leading to enhanced capabilities in comprehending and interpreting fine-grained visual data. Our project is released on https://github.com/htqin/GoogleBard-VisUnderstand

{{</citation>}}


### (6/64) Take-A-Photo: 3D-to-2D Generative Pre-training of Point Cloud Models (Ziyi Wang et al., 2023)

{{<citation>}}

Ziyi Wang, Xumin Yu, Yongming Rao, Jie Zhou, Jiwen Lu. (2023)  
**Take-A-Photo: 3D-to-2D Generative Pre-training of Point Cloud Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.14971v1)  

---


**ABSTRACT**  
With the overwhelming trend of mask image modeling led by MAE, generative pre-training has shown a remarkable potential to boost the performance of fundamental models in 2D vision. However, in 3D vision, the over-reliance on Transformer-based backbones and the unordered nature of point clouds have restricted the further development of generative pre-training. In this paper, we propose a novel 3D-to-2D generative pre-training method that is adaptable to any point cloud model. We propose to generate view images from different instructed poses via the cross-attention mechanism as the pre-training scheme. Generating view images has more precise supervision than its point cloud counterpart, thus assisting 3D backbones to have a finer comprehension of the geometrical structure and stereoscopic relations of the point cloud. Experimental results have proved the superiority of our proposed 3D-to-2D generative pre-training over previous pre-training methods. Our method is also effective in boosting the performance of architecture-oriented approaches, achieving state-of-the-art performance when fine-tuning on ScanObjectNN classification and ShapeNetPart segmentation tasks. Code is available at https://github.com/wangzy22/TAP.

{{</citation>}}


### (7/64) Federated Model Aggregation via Self-Supervised Priors for Highly Imbalanced Medical Image Classification (Marawan Elbatel et al., 2023)

{{<citation>}}

Marawan Elbatel, Hualiang Wang, Robert Martí, Huazhu Fu, Xiaomeng Li. (2023)  
**Federated Model Aggregation via Self-Supervised Priors for Highly Imbalanced Medical Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Image Classification, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.14959v1)  

---


**ABSTRACT**  
In the medical field, federated learning commonly deals with highly imbalanced datasets, including skin lesions and gastrointestinal images. Existing federated methods under highly imbalanced datasets primarily focus on optimizing a global model without incorporating the intra-class variations that can arise in medical imaging due to different populations, findings, and scanners. In this paper, we study the inter-client intra-class variations with publicly available self-supervised auxiliary networks. Specifically, we find that employing a shared auxiliary pre-trained model, like MoCo-V2, locally on every client yields consistent divergence measurements. Based on these findings, we derive a dynamic balanced model aggregation via self-supervised priors (MAS) to guide the global model optimization. Fed-MAS can be utilized with different local learning methods for effective model aggregation toward a highly robust and unbiased global model. Our code is available at \url{https://github.com/xmed-lab/Fed-MAS}.

{{</citation>}}


### (8/64) NSA: Naturalistic Support Artifact to Boost Network Confidence (Abhijith Sharma et al., 2023)

{{<citation>}}

Abhijith Sharma, Phil Munz, Apurva Narayan. (2023)  
**NSA: Naturalistic Support Artifact to Boost Network Confidence**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.14917v1)  

---


**ABSTRACT**  
Visual AI systems are vulnerable to natural and synthetic physical corruption in the real-world. Such corruption often arises unexpectedly and alters the model's performance. In recent years, the primary focus has been on adversarial attacks. However, natural corruptions (e.g., snow, fog, dust) are an omnipresent threat to visual AI systems and should be considered equally important. Many existing works propose interesting solutions to train robust models against natural corruption. These works either leverage image augmentations, which come with the additional cost of model training, or place suspicious patches in the scene to design unadversarial examples. In this work, we propose the idea of naturalistic support artifacts (NSA) for robust prediction. The NSAs are shown to be beneficial in scenarios where model parameters are inaccessible and adding artifacts in the scene is feasible. The NSAs are natural looking objects generated through artifact training using DC-GAN to have high visual fidelity in the scene. We test against natural corruptions on the Imagenette dataset and observe the improvement in prediction confidence score by four times. We also demonstrate NSA's capability to increase adversarial accuracy by 8\% on average. Lastly, we qualitatively analyze NSAs using saliency maps to understand how they help improve prediction confidence.

{{</citation>}}


### (9/64) Text-guided Foundation Model Adaptation for Pathological Image Classification (Yunkun Zhang et al., 2023)

{{<citation>}}

Yunkun Zhang, Jin Gao, Mu Zhou, Xiaosong Wang, Yu Qiao, Shaoting Zhang, Dequan Wang. (2023)  
**Text-guided Foundation Model Adaptation for Pathological Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Embedding, Image Classification  
[Paper Link](http://arxiv.org/abs/2307.14901v1)  

---


**ABSTRACT**  
The recent surge of foundation models in computer vision and natural language processing opens up perspectives in utilizing multi-modal clinical data to train large models with strong generalizability. Yet pathological image datasets often lack biomedical text annotation and enrichment. Guiding data-efficient image diagnosis from the use of biomedical text knowledge becomes a substantial interest. In this paper, we propose to Connect Image and Text Embeddings (CITE) to enhance pathological image classification. CITE injects text insights gained from language models pre-trained with a broad range of biomedical texts, leading to adapt foundation models towards pathological image understanding. Through extensive experiments on the PatchGastric stomach tumor pathological image dataset, we demonstrate that CITE achieves leading performance compared with various baselines especially when training data is scarce. CITE offers insights into leveraging in-domain text knowledge to reinforce data-efficient pathological image classification. Code is available at https://github.com/Yunkun-Zhang/CITE.

{{</citation>}}


### (10/64) Mixture of Self-Supervised Learning (Aristo Renaldo Ruslim et al., 2023)

{{<citation>}}

Aristo Renaldo Ruslim, Novanto Yudistira, Budi Darma Setiawan. (2023)  
**Mixture of Self-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.14897v1)  

---


**ABSTRACT**  
Self-supervised learning is popular method because of its ability to learn features in images without using its labels and is able to overcome limited labeled datasets used in supervised learning. Self-supervised learning works by using a pretext task which will be trained on the model before being applied to a specific task. There are some examples of pretext tasks used in self-supervised learning in the field of image recognition, namely rotation prediction, solving jigsaw puzzles, and predicting relative positions on image. Previous studies have only used one type of transformation as a pretext task. This raises the question of how it affects if more than one pretext task is used and to use a gating network to combine all pretext tasks. Therefore, we propose the Gated Self-Supervised Learning method to improve image classification which use more than one transformation as pretext task and uses the Mixture of Expert architecture as a gating network in combining each pretext task so that the model automatically can study and focus more on the most useful augmentations for classification. We test performance of the proposed method in several scenarios, namely CIFAR imbalance dataset classification, adversarial perturbations, Tiny-Imagenet dataset classification, and semi-supervised learning. Moreover, there are Grad-CAM and T-SNE analysis that are used to see the proposed method for identifying important features that influence image classification and representing data for each class and separating different classes properly. Our code is in https://github.com/aristorenaldo/G-SSL

{{</citation>}}


### (11/64) IML-ViT: Image Manipulation Localization by Vision Transformer (Xiaochen Ma et al., 2023)

{{<citation>}}

Xiaochen Ma, Bo Du, Xianggen Liu, Ahmed Y. Al Hammadi, Jizhe Zhou. (2023)  
**IML-ViT: Image Manipulation Localization by Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.14863v1)  

---


**ABSTRACT**  
Advanced image tampering techniques are increasingly challenging the trustworthiness of multimedia, leading to the development of Image Manipulation Localization (IML). But what makes a good IML model? The answer lies in the way to capture artifacts. Exploiting artifacts requires the model to extract non-semantic discrepancies between the manipulated and authentic regions, which needs to compare differences between these two areas explicitly. With the self-attention mechanism, naturally, the Transformer is the best candidate. Besides, artifacts are sensitive to image resolution, amplified under multi-scale features, and massive at the manipulation border. Therefore, we formulate the answer to the former question as building a ViT with high-resolution capacity, multi-scale feature extraction capability, and manipulation edge supervision. We term this simple but effective ViT paradigm as the IML-ViT, which has great potential to become a new benchmark for IML. Extensive experiments on five benchmark datasets verified our model outperforms the state-of-the-art manipulation localization methods. Code and models are available at \url{https://github.com/SunnyHaze/IML-ViT}

{{</citation>}}


### (12/64) Simplified Concrete Dropout -- Improving the Generation of Attribution Masks for Fine-grained Classification (Dimitri Korsch et al., 2023)

{{<citation>}}

Dimitri Korsch, Maha Shadaydeh, Joachim Denzler. (2023)  
**Simplified Concrete Dropout -- Improving the Generation of Attribution Masks for Fine-grained Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.14825v1)  

---


**ABSTRACT**  
Fine-grained classification is a particular case of a classification problem, aiming to classify objects that share the visual appearance and can only be distinguished by subtle differences. Fine-grained classification models are often deployed to determine animal species or individuals in automated animal monitoring systems. Precise visual explanations of the model's decision are crucial to analyze systematic errors. Attention- or gradient-based methods are commonly used to identify regions in the image that contribute the most to the classification decision. These methods deliver either too coarse or too noisy explanations, unsuitable for identifying subtle visual differences reliably. However, perturbation-based methods can precisely identify pixels causally responsible for the classification result. Fill-in of the dropout (FIDO) algorithm is one of those methods. It utilizes the concrete dropout (CD) to sample a set of attribution masks and updates the sampling parameters based on the output of the classification model. A known problem of the algorithm is a high variance in the gradient estimates, which the authors have mitigated until now by mini-batch updates of the sampling parameters. This paper presents a solution to circumvent these computational instabilities by simplifying the CD sampling and reducing reliance on large mini-batch sizes. First, it allows estimating the parameters with smaller mini-batch sizes without losing the quality of the estimates but with a reduced computational effort. Furthermore, our solution produces finer and more coherent attribution masks. Finally, we use the resulting attribution masks to improve the classification performance of a trained model without additional fine-tuning of the model.

{{</citation>}}


### (13/64) Contrastive Knowledge Amalgamation for Unsupervised Image Classification (Shangde Gao et al., 2023)

{{<citation>}}

Shangde Gao, Yichao Fu, Ke Liu, Yuqiang Han. (2023)  
**Contrastive Knowledge Amalgamation for Unsupervised Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2307.14781v1)  

---


**ABSTRACT**  
Knowledge amalgamation (KA) aims to learn a compact student model to handle the joint objective from multiple teacher models that are are specialized for their own tasks respectively. Current methods focus on coarsely aligning teachers and students in the common representation space, making it difficult for the student to learn the proper decision boundaries from a set of heterogeneous teachers. Besides, the KL divergence in previous works only minimizes the probability distribution difference between teachers and the student, ignoring the intrinsic characteristics of teachers. Therefore, we propose a novel Contrastive Knowledge Amalgamation (CKA) framework, which introduces contrastive losses and an alignment loss to achieve intra-class cohesion and inter-class separation.Contrastive losses intra- and inter- models are designed to widen the distance between representations of different classes. The alignment loss is introduced to minimize the sample-level distribution differences of teacher-student models in the common representation space.Furthermore, the student learns heterogeneous unsupervised classification tasks through soft targets efficiently and flexibly in the task-level amalgamation. Extensive experiments on benchmarks demonstrate the generalization capability of CKA in the amalgamation of specific task as well as multiple tasks. Comprehensive ablation studies provide a further insight into our CKA.

{{</citation>}}


### (14/64) pCTFusion: Point Convolution-Transformer Fusion with Semantic Aware Loss for Outdoor LiDAR Point Cloud Segmentation (Abhishek Kuriyal et al., 2023)

{{<citation>}}

Abhishek Kuriyal, Vaibhav Kumar, Bharat Lohani. (2023)  
**pCTFusion: Point Convolution-Transformer Fusion with Semantic Aware Loss for Outdoor LiDAR Point Cloud Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.14777v1)  

---


**ABSTRACT**  
LiDAR-generated point clouds are crucial for perceiving outdoor environments. The segmentation of point clouds is also essential for many applications. Previous research has focused on using self-attention and convolution (local attention) mechanisms individually in semantic segmentation architectures. However, there is limited work on combining the learned representations of these attention mechanisms to improve performance. Additionally, existing research that combines convolution with self-attention relies on global attention, which is not practical for processing large point clouds. To address these challenges, this study proposes a new architecture, pCTFusion, which combines kernel-based convolutions and self-attention mechanisms for better feature learning and capturing local and global dependencies in segmentation. The proposed architecture employs two types of self-attention mechanisms, local and global, based on the hierarchical positions of the encoder blocks. Furthermore, the existing loss functions do not consider the semantic and position-wise importance of the points, resulting in reduced accuracy, particularly at sharp class boundaries. To overcome this, the study models a novel attention-based loss function called Pointwise Geometric Anisotropy (PGA), which assigns weights based on the semantic distribution of points in a neighborhood. The proposed architecture is evaluated on SemanticKITTI outdoor dataset and showed a 5-7% improvement in performance compared to the state-of-the-art architectures. The results are particularly encouraging for minor classes, often misclassified due to class imbalance, lack of space, and neighbor-aware feature encoding. These developed methods can be leveraged for the segmentation of complex datasets and can drive real-world applications of LiDAR point cloud.

{{</citation>}}


### (15/64) Gloss-free Sign Language Translation: Improving from Visual-Language Pretraining (Benjia Zhou et al., 2023)

{{<citation>}}

Benjia Zhou, Zhigang Chen, Albert Clapés, Jun Wan, Yanyan Liang, Sergio Escalera, Zhen Lei, Du Zhang. (2023)  
**Gloss-free Sign Language Translation: Improving from Visual-Language Pretraining**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2307.14768v1)  

---


**ABSTRACT**  
Sign Language Translation (SLT) is a challenging task due to its cross-domain nature, involving the translation of visual-gestural language to text. Many previous methods employ an intermediate representation, i.e., gloss sequences, to facilitate SLT, thus transforming it into a two-stage task of sign language recognition (SLR) followed by sign language translation (SLT). However, the scarcity of gloss-annotated sign language data, combined with the information bottleneck in the mid-level gloss representation, has hindered the further development of the SLT task. To address this challenge, we propose a novel Gloss-Free SLT based on Visual-Language Pretraining (GFSLT-VLP), which improves SLT by inheriting language-oriented prior knowledge from pre-trained models, without any gloss annotation assistance. Our approach involves two stages: (i) integrating Contrastive Language-Image Pre-training (CLIP) with masked self-supervised learning to create pre-tasks that bridge the semantic gap between visual and textual representations and restore masked sentences, and (ii) constructing an end-to-end architecture with an encoder-decoder-like structure that inherits the parameters of the pre-trained Visual Encoder and Text Decoder from the first stage. The seamless combination of these novel designs forms a robust sign language representation and significantly improves gloss-free sign language translation. In particular, we have achieved unprecedented improvements in terms of BLEU-4 score on the PHOENIX14T dataset (>+5) and the CSL-Daily dataset (>+3) compared to state-of-the-art gloss-free SLT methods. Furthermore, our approach also achieves competitive results on the PHOENIX14T dataset when compared with most of the gloss-based methods. Our code is available at https://github.com/zhoubenjia/GFSLT-VLP.

{{</citation>}}


### (16/64) Exploring Annotation-free Image Captioning with Retrieval-augmented Pseudo Sentence Generation (Zhiyuan Li et al., 2023)

{{<citation>}}

Zhiyuan Li, Dongnan Liu, Heng Wang, Chaoyi Zhang, Weidong Cai. (2023)  
**Exploring Annotation-free Image Captioning with Retrieval-augmented Pseudo Sentence Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Image Captioning  
[Paper Link](http://arxiv.org/abs/2307.14750v1)  

---


**ABSTRACT**  
Training an image captioner without annotated image-sentence pairs has gained traction in recent years. Previous approaches can be categorized into two strategies: crawling sentences from mismatching corpora and aligning them with the given images as pseudo annotations, or pre-training the captioner using external image-text pairs. However, the aligning setting seems to reach its performance limit due to the quality problem of pairs, and pre-training requires significant computational resources. To address these challenges, we propose a new strategy ``LPM + retrieval-augmented learning" where the prior knowledge from large pre-trained models (LPMs) is leveraged as supervision, and a retrieval process is integrated to further reinforce its effectiveness. Specifically, we introduce Retrieval-augmented Pseudo Sentence Generation (RaPSG), which adopts an efficient approach to retrieve highly relevant short region descriptions from the mismatching corpora and use them to generate a variety of pseudo sentences with distinct representations as well as high quality via LPMs. In addition, a fluency filter and a CLIP-guided training objective are further introduced to facilitate model optimization. Experimental results demonstrate that our method surpasses the SOTA pre-training model (Flamingo3B) by achieving a CIDEr score of 78.1 (+5.1) while utilizing only 0.3% of its trainable parameters (1.3B VS 33M). Importantly, our approach eliminates the need of computationally expensive pre-training processes on external datasets (e.g., the requirement of 312M image-text pairs for Flamingo3B). We further show that with a simple extension, the generated pseudo sentences can be deployed as weak supervision to boost the 1% semi-supervised image caption benchmark up to 93.4 CIDEr score (+8.9) which showcases the versatility and effectiveness of our approach.

{{</citation>}}


### (17/64) Test Time Adaptation for Blind Image Quality Assessment (Subhadeep Roy et al., 2023)

{{<citation>}}

Subhadeep Roy, Shankhanil Mitra, Soma Biswas, Rajiv Soundararajan. (2023)  
**Test Time Adaptation for Blind Image Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.14735v1)  

---


**ABSTRACT**  
While the design of blind image quality assessment (IQA) algorithms has improved significantly, the distribution shift between the training and testing scenarios often leads to a poor performance of these methods at inference time. This motivates the study of test time adaptation (TTA) techniques to improve their performance at inference time. Existing auxiliary tasks and loss functions used for TTA may not be relevant for quality-aware adaptation of the pre-trained model. In this work, we introduce two novel quality-relevant auxiliary tasks at the batch and sample levels to enable TTA for blind IQA. In particular, we introduce a group contrastive loss at the batch level and a relative rank loss at the sample level to make the model quality aware and adapt to the target data. Our experiments reveal that even using a small batch of images from the test distribution helps achieve significant improvement in performance by updating the batch normalization statistics of the source model.

{{</citation>}}


### (18/64) P2C: Self-Supervised Point Cloud Completion from Single Partial Clouds (Ruikai Cui et al., 2023)

{{<citation>}}

Ruikai Cui, Shi Qiu, Saeed Anwar, Jiawei Liu, Chaoyue Xing, Jing Zhang, Nick Barnes. (2023)  
**P2C: Self-Supervised Point Cloud Completion from Single Partial Clouds**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.14726v1)  

---


**ABSTRACT**  
Point cloud completion aims to recover the complete shape based on a partial observation. Existing methods require either complete point clouds or multiple partial observations of the same object for learning. In contrast to previous approaches, we present Partial2Complete (P2C), the first self-supervised framework that completes point cloud objects using training samples consisting of only a single incomplete point cloud per object. Specifically, our framework groups incomplete point clouds into local patches as input and predicts masked patches by learning prior information from different partial objects. We also propose Region-Aware Chamfer Distance to regularize shape mismatch without limiting completion capability, and devise the Normal Consistency Constraint to incorporate a local planarity assumption, encouraging the recovered shape surface to be continuous and complete. In this way, P2C no longer needs multiple observations or complete point clouds as ground truth. Instead, structural cues are learned from a category-specific dataset to complete partial point clouds of objects. We demonstrate the effectiveness of our approach on both synthetic ShapeNet data and real-world ScanNet data, showing that P2C produces comparable results to methods trained with complete shapes, and outperforms methods learned with multiple partial observations. Code is available at https://github.com/CuiRuikai/Partial2Complete.

{{</citation>}}


### (19/64) vox2vec: A Framework for Self-supervised Contrastive Learning of Voxel-level Representations in Medical Images (Mikhail Goncharov et al., 2023)

{{<citation>}}

Mikhail Goncharov, Vera Soboleva, Anvar Kurmukov, Maxim Pisov, Mikhail Belyaev. (2023)  
**vox2vec: A Framework for Self-supervised Contrastive Learning of Voxel-level Representations in Medical Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.14725v1)  

---


**ABSTRACT**  
This paper introduces vox2vec - a contrastive method for self-supervised learning (SSL) of voxel-level representations. vox2vec representations are modeled by a Feature Pyramid Network (FPN): a voxel representation is a concatenation of the corresponding feature vectors from different pyramid levels. The FPN is pre-trained to produce similar representations for the same voxel in different augmented contexts and distinctive representations for different voxels. This results in unified multi-scale representations that capture both global semantics (e.g., body part) and local semantics (e.g., different small organs or healthy versus tumor tissue). We use vox2vec to pre-train a FPN on more than 6500 publicly available computed tomography images. We evaluate the pre-trained representations by attaching simple heads on top of them and training the resulting models for 22 segmentation tasks. We show that vox2vec outperforms existing medical imaging SSL techniques in three evaluation setups: linear and non-linear probing and end-to-end fine-tuning. Moreover, a non-linear head trained on top of the frozen vox2vec representations achieves competitive performance with the FPN trained from scratch while having 50 times fewer trainable parameters. The code is available at https://github.com/mishgon/vox2vec .

{{</citation>}}


### (20/64) Pre-training Vision Transformers with Very Limited Synthesized Images (Ryo Nakamura1 et al., 2023)

{{<citation>}}

Ryo Nakamura1, Hirokatsu Kataoka, Sora Takashima, Edgar Josafat Martinez Noriega, Rio Yokota, Nakamasa Inoue. (2023)  
**Pre-training Vision Transformers with Very Limited Synthesized Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.14710v1)  

---


**ABSTRACT**  
Formula-driven supervised learning (FDSL) is a pre-training method that relies on synthetic images generated from mathematical formulae such as fractals. Prior work on FDSL has shown that pre-training vision transformers on such synthetic datasets can yield competitive accuracy on a wide range of downstream tasks. These synthetic images are categorized according to the parameters in the mathematical formula that generate them. In the present work, we hypothesize that the process for generating different instances for the same category in FDSL, can be viewed as a form of data augmentation. We validate this hypothesis by replacing the instances with data augmentation, which means we only need a single image per category. Our experiments shows that this one-instance fractal database (OFDB) performs better than the original dataset where instances were explicitly generated. We further scale up OFDB to 21,000 categories and show that it matches, or even surpasses, the model pre-trained on ImageNet-21k in ImageNet-1k fine-tuning. The number of images in OFDB is 21k, whereas ImageNet-21k has 14M. This opens new possibilities for pre-training vision transformers with much smaller datasets.

{{</citation>}}


### (21/64) High Dynamic Range Imaging via Visual Attention Modules (Ali Reza Omrani et al., 2023)

{{<citation>}}

Ali Reza Omrani, Davide Moroni. (2023)  
**High Dynamic Range Imaging via Visual Attention Modules**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.14705v1)  

---


**ABSTRACT**  
Thanks to High Dynamic Range (HDR) imaging methods, the scope of photography has seen profound changes recently. To be more specific, such methods try to reconstruct the lost luminosity of the real world caused by the limitation of regular cameras from the Low Dynamic Range (LDR) images. Additionally, although the State-Of-The-Art methods in this topic perform well, they mainly concentrate on combining different exposures and have less attention to extracting the informative parts of the images. Thus, this paper aims to introduce a new model capable of incorporating information from the most visible areas of each image extracted by a visual attention module (VAM), which is a result of a segmentation strategy. In particular, the model, based on a deep learning architecture, utilizes the extracted areas to produce the final HDR image. The results demonstrate that our method outperformed most of the State-Of-The-Art algorithms.

{{</citation>}}


### (22/64) HTNet for micro-expression recognition (Zhifeng Wang et al., 2023)

{{<citation>}}

Zhifeng Wang, Kaihao Zhang, Wenhan Luo, Ramesh Sankaranarayana. (2023)  
**HTNet for micro-expression recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.14637v1)  

---


**ABSTRACT**  
Facial expression is related to facial muscle contractions and different muscle movements correspond to different emotional states. For micro-expression recognition, the muscle movements are usually subtle, which has a negative impact on the performance of current facial emotion recognition algorithms. Most existing methods use self-attention mechanisms to capture relationships between tokens in a sequence, but they do not take into account the inherent spatial relationships between facial landmarks. This can result in sub-optimal performance on micro-expression recognition tasks.Therefore, learning to recognize facial muscle movements is a key challenge in the area of micro-expression recognition. In this paper, we propose a Hierarchical Transformer Network (HTNet) to identify critical areas of facial muscle movement. HTNet includes two major components: a transformer layer that leverages the local temporal features and an aggregation layer that extracts local and global semantical facial features. Specifically, HTNet divides the face into four different facial areas: left lip area, left eye area, right eye area and right lip area. The transformer layer is used to focus on representing local minor muscle movement with local self-attention in each area. The aggregation layer is used to learn the interactions between eye areas and lip areas. The experiments on four publicly available micro-expression datasets show that the proposed approach outperforms previous methods by a large margin. The codes and models are available at: \url{https://github.com/wangzhifengharrison/HTNet}

{{</citation>}}


### (23/64) NeRF-Det: Learning Geometry-Aware Volumetric Representation for Multi-View 3D Object Detection (Chenfeng Xu et al., 2023)

{{<citation>}}

Chenfeng Xu, Bichen Wu, Ji Hou, Sam Tsai, Ruilong Li, Jialiang Wang, Wei Zhan, Zijian He, Peter Vajda, Kurt Keutzer, Masayoshi Tomizuka. (2023)  
**NeRF-Det: Learning Geometry-Aware Volumetric Representation for Multi-View 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.14620v1)  

---


**ABSTRACT**  
We present NeRF-Det, a novel method for indoor 3D detection with posed RGB images as input. Unlike existing indoor 3D detection methods that struggle to model scene geometry, our method makes novel use of NeRF in an end-to-end manner to explicitly estimate 3D geometry, thereby improving 3D detection performance. Specifically, to avoid the significant extra latency associated with per-scene optimization of NeRF, we introduce sufficient geometry priors to enhance the generalizability of NeRF-MLP. Furthermore, we subtly connect the detection and NeRF branches through a shared MLP, enabling an efficient adaptation of NeRF to detection and yielding geometry-aware volumetric representations for 3D detection. Our method outperforms state-of-the-arts by 3.9 mAP and 3.1 mAP on the ScanNet and ARKITScenes benchmarks, respectively. We provide extensive analysis to shed light on how NeRF-Det works. As a result of our joint-training design, NeRF-Det is able to generalize well to unseen scenes for object detection, view synthesis, and depth estimation tasks without requiring per-scene optimization. Code is available at \url{https://github.com/facebookresearch/NeRF-Det}.

{{</citation>}}


### (24/64) GenCo: An Auxiliary Generator from Contrastive Learning for Enhanced Few-Shot Learning in Remote Sensing (Jing Wu et al., 2023)

{{<citation>}}

Jing Wu, Naira Hovakimyan, Jennifer Hobbs. (2023)  
**GenCo: An Auxiliary Generator from Contrastive Learning for Enhanced Few-Shot Learning in Remote Sensing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, Few-Shot  
[Paper Link](http://arxiv.org/abs/2307.14612v1)  

---


**ABSTRACT**  
Classifying and segmenting patterns from a limited number of examples is a significant challenge in remote sensing and earth observation due to the difficulty in acquiring accurately labeled data in large quantities. Previous studies have shown that meta-learning, which involves episodic training on query and support sets, is a promising approach. However, there has been little attention paid to direct fine-tuning techniques. This paper repurposes contrastive learning as a pre-training method for few-shot learning for classification and semantic segmentation tasks. Specifically, we introduce a generator-based contrastive learning framework (GenCo) that pre-trains backbones and simultaneously explores variants of feature samples. In fine-tuning, the auxiliary generator can be used to enrich limited labeled data samples in feature space. We demonstrate the effectiveness of our method in improving few-shot learning performance on two key remote sensing datasets: Agriculture-Vision and EuroSAT. Empirically, our approach outperforms purely supervised training on the nearly 95,000 images in Agriculture-Vision for both classification and semantic segmentation tasks. Similarly, the proposed few-shot method achieves better results on the land-cover classification task on EuroSAT compared to the results obtained from fully supervised model training on the dataset.

{{</citation>}}


### (25/64) TextManiA: Enriching Visual Feature by Text-driven Manifold Augmentation (Moon Ye-Bin et al., 2023)

{{<citation>}}

Moon Ye-Bin, Jisoo Kim, Hongyeob Kim, Kilho Son, Tae-Hyun Oh. (2023)  
**TextManiA: Enriching Visual Feature by Text-driven Manifold Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.14611v1)  

---


**ABSTRACT**  
Recent label mix-based augmentation methods have shown their effectiveness in generalization despite their simplicity, and their favorable effects are often attributed to semantic-level augmentation. However, we found that they are vulnerable to highly skewed class distribution, because scarce data classes are rarely sampled for inter-class perturbation. We propose TextManiA, a text-driven manifold augmentation method that semantically enriches visual feature spaces, regardless of data distribution. TextManiA augments visual data with intra-class semantic perturbation by exploiting easy-to-understand visually mimetic words, i.e., attributes. To this end, we bridge between the text representation and a target visual feature space, and propose an efficient vector augmentation. To empirically support the validity of our design, we devise two visualization-based analyses and show the plausibility of the bridge between two different modality spaces. Our experiments demonstrate that TextManiA is powerful in scarce samples with class imbalance as well as even distribution. We also show compatibility with the label mix-based approaches in evenly distributed scarce data.

{{</citation>}}


### (26/64) Clustering based Point Cloud Representation Learning for 3D Analysis (Tuo Feng et al., 2023)

{{<citation>}}

Tuo Feng, Wenguan Wang, Xiaohan Wang, Yi Yang, Qinghua Zheng. (2023)  
**Clustering based Point Cloud Representation Learning for 3D Analysis**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Representation Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2307.14605v1)  

---


**ABSTRACT**  
Point cloud analysis (such as 3D segmentation and detection) is a challenging task, because of not only the irregular geometries of many millions of unordered points, but also the great variations caused by depth, viewpoint, occlusion, etc. Current studies put much focus on the adaption of neural networks to the complex geometries of point clouds, but are blind to a fundamental question: how to learn an appropriate point embedding space that is aware of both discriminative semantics and challenging variations? As a response, we propose a clustering based supervised learning scheme for point cloud analysis. Unlike current de-facto, scene-wise training paradigm, our algorithm conducts within-class clustering on the point embedding space for automatically discovering subclass patterns which are latent yet representative across scenes. The mined patterns are, in turn, used to repaint the embedding space, so as to respect the underlying distribution of the entire training dataset and improve the robustness to the variations. Our algorithm is principled and readily pluggable to modern point cloud segmentation networks during training, without extra overhead during testing. With various 3D network architectures (i.e., voxel-based, point-based, Transformer-based, automatically searched), our algorithm shows notable improvements on famous point cloud segmentation datasets (i.e.,2.0-2.6% on single-scan and 2.0-2.2% multi-scan of SemanticKITTI, 1.8-1.9% on S3DIS, in terms of mIoU). Our algorithm also demonstrates utility in 3D detection, showing 2.0-3.4% mAP gains on KITTI.

{{</citation>}}


### (27/64) FakeTracer: Proactively Defending Against Face-swap DeepFakes via Implanting Traces in Training (Pu Sun et al., 2023)

{{<citation>}}

Pu Sun, Honggang Qi, Yuezun Li, Siwei Lyu. (2023)  
**FakeTracer: Proactively Defending Against Face-swap DeepFakes via Implanting Traces in Training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.14593v1)  

---


**ABSTRACT**  
Face-swap DeepFake is an emerging AI-based face forgery technique that can replace the original face in a video with a generated face of the target identity while retaining consistent facial attributes such as expression and orientation. Due to the high privacy of faces, the misuse of this technique can raise severe social concerns, drawing tremendous attention to defend against DeepFakes recently. In this paper, we describe a new proactive defense method called FakeTracer to expose face-swap DeepFakes via implanting traces in training. Compared to general face-synthesis DeepFake, the face-swap DeepFake is more complex as it involves identity change, is subjected to the encoding-decoding process, and is trained unsupervised, increasing the difficulty of implanting traces into the training phase. To effectively defend against face-swap DeepFake, we design two types of traces, sustainable trace (STrace) and erasable trace (ETrace), to be added to training faces. During the training, these manipulated faces affect the learning of the face-swap DeepFake model, enabling it to generate faces that only contain sustainable traces. In light of these two traces, our method can effectively expose DeepFakes by identifying them. Extensive experiments are conducted on the Celeb-DF dataset, compared with recent passive and proactive defense methods, and are studied thoroughly regarding various factors, corroborating the efficacy of our method on defending against face-swap DeepFake.

{{</citation>}}


## cs.MM (1)



### (28/64) Self-Supervised Visual Acoustic Matching (Arjun Somayazulu et al., 2023)

{{<citation>}}

Arjun Somayazulu, Changan Chen, Kristen Grauman. (2023)  
**Self-Supervised Visual Acoustic Matching**  

---
Primary Category: cs.MM  
Categories: cs-CV, cs-MM, cs-SD, cs.MM, eess-AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.15064v1)  

---


**ABSTRACT**  
Acoustic matching aims to re-synthesize an audio clip to sound as if it were recorded in a target acoustic environment. Existing methods assume access to paired training data, where the audio is observed in both source and target environments, but this limits the diversity of training data or requires the use of simulated data or heuristics to create paired samples. We propose a self-supervised approach to visual acoustic matching where training samples include only the target scene image and audio -- without acoustically mismatched source audio for reference. Our approach jointly learns to disentangle room acoustics and re-synthesize audio into the target environment, via a conditional GAN framework and a novel metric that quantifies the level of residual acoustic information in the de-biased audio. Training with either in-the-wild web data or simulated data, we demonstrate it outperforms the state-of-the-art on multiple challenging datasets and a wide variety of real-world audio and environments.

{{</citation>}}


## cs.IR (2)



### (29/64) On (Normalised) Discounted Cumulative Gain as an Offline Evaluation Metric for Top-$n$ Recommendation (Olivier Jeunen et al., 2023)

{{<citation>}}

Olivier Jeunen, Ivan Potapov, Aleksei Ustimenko. (2023)  
**On (Normalised) Discounted Cumulative Gain as an Offline Evaluation Metric for Top-$n$ Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2307.15053v1)  

---


**ABSTRACT**  
Approaches to recommendation are typically evaluated in one of two ways: (1) via a (simulated) online experiment, often seen as the gold standard, or (2) via some offline evaluation procedure, where the goal is to approximate the outcome of an online experiment. Several offline evaluation metrics have been adopted in the literature, inspired by ranking metrics prevalent in the field of Information Retrieval. (Normalised) Discounted Cumulative Gain (nDCG) is one such metric that has seen widespread adoption in empirical studies, and higher (n)DCG values have been used to present new methods as the state-of-the-art in top-$n$ recommendation for many years.   Our work takes a critical look at this approach, and investigates when we can expect such metrics to approximate the gold standard outcome of an online experiment. We formally present the assumptions that are necessary to consider DCG an unbiased estimator of online reward and provide a derivation for this metric from first principles, highlighting where we deviate from its traditional uses in IR. Importantly, we show that normalising the metric renders it inconsistent, in that even when DCG is unbiased, ranking competing methods by their normalised DCG can invert their relative order. Through a correlation analysis between off- and on-line experiments conducted on a large-scale recommendation platform, we show that our unbiased DCG estimates strongly correlate with online reward, even when some of the metric's inherent assumptions are violated. This statement no longer holds for its normalised variant, suggesting that nDCG's practical utility may be limited.

{{</citation>}}


### (30/64) Scaling Session-Based Transformer Recommendations using Optimized Negative Sampling and Loss Functions (Timo Wilm et al., 2023)

{{<citation>}}

Timo Wilm, Philipp Normann, Sophie Baumeister, Paul-Vincent Kobow. (2023)  
**Scaling Session-Based Transformer Recommendations using Optimized Negative Sampling and Loss Functions**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.14906v1)  

---


**ABSTRACT**  
This work introduces TRON, a scalable session-based Transformer Recommender using Optimized Negative-sampling. Motivated by the scalability and performance limitations of prevailing models such as SASRec and GRU4Rec+, TRON integrates top-k negative sampling and listwise loss functions to enhance its recommendation accuracy. Evaluations on relevant large-scale e-commerce datasets show that TRON improves upon the recommendation quality of current methods while maintaining training speeds similar to SASRec. A live A/B test yielded an 18.14% increase in click-through rate over SASRec, highlighting the potential of TRON in practical settings. For further research, we provide access to our source code at https://github.com/otto-de/TRON and an anonymized dataset at https://github.com/otto-de/recsys-dataset.

{{</citation>}}


## cs.CL (15)



### (31/64) Matching Patients to Clinical Trials with Large Language Models (Qiao Jin et al., 2023)

{{<citation>}}

Qiao Jin, Zifeng Wang, Charalampos S. Floudas, Jimeng Sun, Zhiyong Lu. (2023)  
**Matching Patients to Clinical Trials with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Clinical, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.15051v1)  

---


**ABSTRACT**  
Clinical trials are vital in advancing drug development and evidence-based medicine, but their success is often hindered by challenges in patient recruitment. In this work, we investigate the potential of large language models (LLMs) to assist individual patients and referral physicians in identifying suitable clinical trials from an extensive selection. Specifically, we introduce TrialGPT, a novel architecture employing LLMs to predict criterion-level eligibility with detailed explanations, which are then aggregated for ranking and excluding candidate clinical trials based on free-text patient notes. We evaluate TrialGPT on three publicly available cohorts of 184 patients and 18,238 annotated clinical trials. The experimental results demonstrate several key findings: First, TrialGPT achieves high criterion-level prediction accuracy with faithful explanations. Second, the aggregated trial-level TrialGPT scores are highly correlated with expert eligibility annotations. Third, these scores prove effective in ranking clinical trials and exclude ineligible candidates. Our error analysis suggests that current LLMs still make some mistakes due to limited medical knowledge and domain-specific context understanding. Nonetheless, we believe the explanatory capabilities of LLMs are highly valuable. Future research is warranted on how such AI assistants can be integrated into the routine trial matching workflow in real-world settings to improve its efficiency.

{{</citation>}}


### (32/64) Universal and Transferable Adversarial Attacks on Aligned Language Models (Andy Zou et al., 2023)

{{<citation>}}

Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson. (2023)  
**Universal and Transferable Adversarial Attacks on Aligned Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: Adversarial Attack, ChatGPT, Falcon, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2307.15043v1)  

---


**ABSTRACT**  
Because "out-of-the-box" large language models are capable of generating a great deal of objectionable content, recent work has focused on aligning these models in an attempt to prevent undesirable generation. While there has been some success at circumventing these measures -- so-called "jailbreaks" against LLMs -- these attacks have required significant human ingenuity and are brittle in practice. In this paper, we propose a simple and effective attack method that causes aligned language models to generate objectionable behaviors. Specifically, our approach finds a suffix that, when attached to a wide range of queries for an LLM to produce objectionable content, aims to maximize the probability that the model produces an affirmative response (rather than refusing to answer). However, instead of relying on manual engineering, our approach automatically produces these adversarial suffixes by a combination of greedy and gradient-based search techniques, and also improves over past automatic prompt generation methods.   Surprisingly, we find that the adversarial prompts generated by our approach are quite transferable, including to black-box, publicly released LLMs. Specifically, we train an adversarial attack suffix on multiple prompts (i.e., queries asking for many different types of objectionable content), as well as multiple models (in our case, Vicuna-7B and 13B). When doing so, the resulting attack suffix is able to induce objectionable content in the public interfaces to ChatGPT, Bard, and Claude, as well as open source LLMs such as LLaMA-2-Chat, Pythia, Falcon, and others. In total, this work significantly advances the state-of-the-art in adversarial attacks against aligned language models, raising important questions about how such systems can be prevented from producing objectionable information. Code is available at github.com/llm-attacks/llm-attacks.

{{</citation>}}


### (33/64) SuperCLUE: A Comprehensive Chinese Large Language Model Benchmark (Liang Xu et al., 2023)

{{<citation>}}

Liang Xu, Anqi Li, Lei Zhu, Hang Xue, Changtai Zhu, Kangkang Zhao, Haonan He, Xuanwei Zhang, Qiyue Kang, Zhenzhong Lan. (2023)  
**SuperCLUE: A Comprehensive Chinese Large Language Model Benchmark**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.15020v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown the potential to be integrated into human daily lives. Therefore, user preference is the most critical criterion for assessing LLMs' performance in real-world scenarios. However, existing benchmarks mainly focus on measuring models' accuracy using multi-choice questions, which limits the understanding of their capabilities in real applications. We fill this gap by proposing a comprehensive Chinese benchmark SuperCLUE, named after another popular Chinese LLM benchmark CLUE. SuperCLUE encompasses three sub-tasks: actual users' queries and ratings derived from an LLM battle platform (CArena), open-ended questions with single and multiple-turn dialogues (OPEN), and closed-ended questions with the same stems as open-ended single-turn ones (CLOSE). Our study shows that accuracy on closed-ended questions is insufficient to reflect human preferences achieved on open-ended ones. At the same time, they can complement each other to predict actual user preferences. We also demonstrate that GPT-4 is a reliable judge to automatically evaluate human preferences on open-ended questions in a Chinese context. Our benchmark will be released at https://www.CLUEbenchmarks.com

{{</citation>}}


### (34/64) Scaling TransNormer to 175 Billion Parameters (Zhen Qin et al., 2023)

{{<citation>}}

Zhen Qin, Dong Li, Weigao Sun, Weixuan Sun, Xuyang Shen, Xiaodong Han, Yunshen Wei, Baohong Lv, Fei Yuan, Xiao Luo, Yu Qiao, Yiran Zhong. (2023)  
**Scaling TransNormer to 175 Billion Parameters**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Language Model  
[Paper Link](http://arxiv.org/abs/2307.14995v1)  

---


**ABSTRACT**  
We present TransNormerLLM, the first linear attention-based Large Language Model (LLM) that outperforms conventional softmax attention-based models in terms of both accuracy and efficiency. TransNormerLLM evolves from the previous linear attention architecture TransNormer by making advanced modifications that include positional embedding, linear attention acceleration, gating mechanism, tensor normalization, inference acceleration and stabilization. Specifically, we use LRPE together with an exponential decay to avoid attention dilution issues while allowing the model to retain global interactions between tokens. Additionally, we propose Lightning Attention, a cutting-edge technique that accelerates linear attention by more than twice in runtime and reduces memory usage by a remarkable four times. To further enhance the performance of TransNormer, we leverage a gating mechanism to smooth training and a new tensor normalization scheme to accelerate the model, resulting in an impressive acceleration of over 20%. Furthermore, we have developed a robust inference algorithm that ensures numerical stability and consistent inference speed, regardless of the sequence length, showcasing superior efficiency during both training and inference stages. Scalability is at the heart of our model's design, enabling seamless deployment on large-scale clusters and facilitating expansion to even more extensive models, all while maintaining outstanding performance metrics. Rigorous validation of our model design is achieved through a series of comprehensive experiments on our self-collected corpus, boasting a size exceeding 6TB and containing over 2 trillion tokens. To ensure data quality and relevance, we implement a new self-cleaning strategy to filter our collected data. Our pre-trained models will be released to foster community advancements in efficient LLMs.

{{</citation>}}


### (35/64) PanGu-Coder2: Boosting Large Language Models for Code with Ranking Feedback (Bo Shen et al., 2023)

{{<citation>}}

Bo Shen, Jiaxin Zhang, Taihong Chen, Daoguang Zan, Bing Geng, An Fu, Muhan Zeng, Ailun Yu, Jichuan Ji, Jingyang Zhao, Yuenan Guo, Qianxiang Wang. (2023)  
**PanGu-Coder2: Boosting Large Language Models for Code with Ranking Feedback**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-PL, cs-SE, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2307.14936v1)  

---


**ABSTRACT**  
Large Language Models for Code (Code LLM) are flourishing. New and powerful models are released on a weekly basis, demonstrating remarkable performance on the code generation task. Various approaches have been proposed to boost the code generation performance of pre-trained Code LLMs, such as supervised fine-tuning, instruction tuning, reinforcement learning, etc. In this paper, we propose a novel RRTF (Rank Responses to align Test&Teacher Feedback) framework, which can effectively and efficiently boost pre-trained large language models for code generation. Under this framework, we present PanGu-Coder2, which achieves 62.20% pass@1 on the OpenAI HumanEval benchmark. Furthermore, through an extensive evaluation on CoderEval and LeetCode benchmarks, we show that PanGu-Coder2 consistently outperforms all previous Code LLMs.

{{</citation>}}


### (36/64) ARC-NLP at PAN 2023: Transition-Focused Natural Language Inference for Writing Style Detection (Izzet Emre Kucukkaya et al., 2023)

{{<citation>}}

Izzet Emre Kucukkaya, Umitcan Sahin, Cagri Toraman. (2023)  
**ARC-NLP at PAN 2023: Transition-Focused Natural Language Inference for Writing Style Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NLP, Natural Language Inference, Transformer  
[Paper Link](http://arxiv.org/abs/2307.14913v1)  

---


**ABSTRACT**  
The task of multi-author writing style detection aims at finding any positions of writing style change in a given text document. We formulate the task as a natural language inference problem where two consecutive paragraphs are paired. Our approach focuses on transitions between paragraphs while truncating input tokens for the task. As backbone models, we employ different Transformer-based encoders with warmup phase during training. We submit the model version that outperforms baselines and other proposed model versions in our experiments. For the easy and medium setups, we submit transition-focused natural language inference based on DeBERTa with warmup training, and the same model without transition for the hard setup.

{{</citation>}}


### (37/64) ARC-NLP at PAN 2023: Hierarchical Long Text Classification for Trigger Detection (Umitcan Sahin et al., 2023)

{{<citation>}}

Umitcan Sahin, Izzet Emre Kucukkaya, Cagri Toraman. (2023)  
**ARC-NLP at PAN 2023: Hierarchical Long Text Classification for Trigger Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: LSTM, NLP, Text Classification, Transformer  
[Paper Link](http://arxiv.org/abs/2307.14912v1)  

---


**ABSTRACT**  
Fanfiction, a popular form of creative writing set within established fictional universes, has gained a substantial online following. However, ensuring the well-being and safety of participants has become a critical concern in this community. The detection of triggering content, material that may cause emotional distress or trauma to readers, poses a significant challenge. In this paper, we describe our approach for the Trigger Detection shared task at PAN CLEF 2023, where we want to detect multiple triggering content in a given Fanfiction document. For this, we build a hierarchical model that uses recurrence over Transformer-based language models. In our approach, we first split long documents into smaller sized segments and use them to fine-tune a Transformer model. Then, we extract feature embeddings from the fine-tuned Transformer model, which are used as input in the training of multiple LSTM models for trigger detection in a multi-label setting. Our model achieves an F1-macro score of 0.372 and F1-micro score of 0.736 on the validation set, which are higher than the baseline results shared at PAN CLEF 2023.

{{</citation>}}


### (38/64) Exploiting the Potential of Seq2Seq Models as Robust Few-Shot Learners (Jihyeon Lee et al., 2023)

{{<citation>}}

Jihyeon Lee, Dain Kim, Doohae Jung, Boseop Kim, Kyoung-Woon On. (2023)  
**Exploiting the Potential of Seq2Seq Models as Robust Few-Shot Learners**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Few-Shot, Seq2Seq  
[Paper Link](http://arxiv.org/abs/2307.14856v1)  

---


**ABSTRACT**  
In-context learning, which offers substantial advantages over fine-tuning, is predominantly observed in decoder-only models, while encoder-decoder (i.e., seq2seq) models excel in methods that rely on weight updates. Recently, a few studies have demonstrated the feasibility of few-shot learning with seq2seq models; however, this has been limited to tasks that align well with the seq2seq architecture, such as summarization and translation. Inspired by these initial studies, we provide a first-ever extensive experiment comparing the in-context few-shot learning capabilities of decoder-only and encoder-decoder models on a broad range of tasks. Furthermore, we propose two methods to more effectively elicit in-context learning ability in seq2seq models: objective-aligned prompting and a fusion-based approach. Remarkably, our approach outperforms a decoder-only model that is six times larger and exhibits significant performance improvements compared to conventional seq2seq models across a variety of settings. We posit that, with the right configuration and prompt design, seq2seq models can be highly effective few-shot learners for a wide spectrum of applications.

{{</citation>}}


### (39/64) ArcGPT: A Large Language Model Tailored for Real-world Archival Applications (Shitou Zhang et al., 2023)

{{<citation>}}

Shitou Zhang, Jingrui Hou, Siyuan Peng, Zuchao Li, Qibiao Hu, Ping Wang. (2023)  
**ArcGPT: A Large Language Model Tailored for Real-world Archival Applications**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.14852v1)  

---


**ABSTRACT**  
Archives play a crucial role in preserving information and knowledge, and the exponential growth of such data necessitates efficient and automated tools for managing and utilizing archive information resources. Archival applications involve managing massive data that are challenging to process and analyze. Although LLMs have made remarkable progress in diverse domains, there are no publicly available archives tailored LLM. Addressing this gap, we introduce ArcGPT, to our knowledge, the first general-purpose LLM tailored to the archival field. To enhance model performance on real-world archival tasks, ArcGPT has been pre-trained on massive and extensive archival domain data. Alongside ArcGPT, we release AMBLE, a benchmark comprising four real-world archival tasks. Evaluation on AMBLE shows that ArcGPT outperforms existing state-of-the-art models, marking a substantial step forward in effective archival data management. Ultimately, ArcGPT aims to better serve the archival community, aiding archivists in their crucial role of preserving and harnessing our collective information and knowledge.

{{</citation>}}


### (40/64) Turkish Native Language Identification (Ahmet Yavuz Uluslu et al., 2023)

{{<citation>}}

Ahmet Yavuz Uluslu, Gerold Schneider. (2023)  
**Turkish Native Language Identification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Identification, NLI  
[Paper Link](http://arxiv.org/abs/2307.14850v1)  

---


**ABSTRACT**  
In this paper, we present the first application of Native Language Identification (NLI) for the Turkish language. NLI involves predicting the writer's first language by analysing their writing in different languages. While most NLI research has focused on English, our study extends its scope to Turkish. We used the recently constructed Turkish Learner Corpus and employed a combination of three syntactic features (CFG production rules, part-of-speech n-grams and function words) with L2 texts to demonstrate their effectiveness in this task.

{{</citation>}}


### (41/64) Models of reference production: How do they withstand the test of time? (Fahime Same et al., 2023)

{{<citation>}}

Fahime Same, Guanyi Chen, Kees van Deemter. (2023)  
**Models of reference production: How do they withstand the test of time?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2307.14817v1)  

---


**ABSTRACT**  
In recent years, many NLP studies have focused solely on performance improvement. In this work, we focus on the linguistic and scientific aspects of NLP. We use the task of generating referring expressions in context (REG-in-context) as a case study and start our analysis from GREC, a comprehensive set of shared tasks in English that addressed this topic over a decade ago. We ask what the performance of models would be if we assessed them (1) on more realistic datasets, and (2) using more advanced methods. We test the models using different evaluation metrics and feature selection experiments. We conclude that GREC can no longer be regarded as offering a reliable assessment of models' ability to mimic human reference production, because the results are highly impacted by the choice of corpus and evaluation metrics. Our results also suggest that pre-trained language models are less dependent on the choice of corpus than classic Machine Learning models, and therefore make more robust class predictions.

{{</citation>}}


### (42/64) Improving Aspect-Based Sentiment with End-to-End Semantic Role Labeling Model (Pavel Přibáň et al., 2023)

{{<citation>}}

Pavel Přibáň, Ondřej Pražák. (2023)  
**Improving Aspect-Based Sentiment with End-to-End Semantic Role Labeling Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Sentiment Analysis, Transformer  
[Paper Link](http://arxiv.org/abs/2307.14785v1)  

---


**ABSTRACT**  
This paper presents a series of approaches aimed at enhancing the performance of Aspect-Based Sentiment Analysis (ABSA) by utilizing extracted semantic information from a Semantic Role Labeling (SRL) model. We propose a novel end-to-end Semantic Role Labeling model that effectively captures most of the structured semantic information within the Transformer hidden state. We believe that this end-to-end model is well-suited for our newly proposed models that incorporate semantic information. We evaluate the proposed models in two languages, English and Czech, employing ELECTRA-small models. Our combined models improve ABSA performance in both languages. Moreover, we achieved new state-of-the-art results on the Czech ABSA.

{{</citation>}}


### (43/64) Evaluating Generative Models for Graph-to-Text Generation (Shuzhou Yuan et al., 2023)

{{<citation>}}

Shuzhou Yuan, Michael Färber. (2023)  
**Evaluating Generative Models for Graph-to-Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, BLEU, ChatGPT, GPT, T5, Text Generation  
[Paper Link](http://arxiv.org/abs/2307.14712v1)  

---


**ABSTRACT**  
Large language models (LLMs) have been widely employed for graph-to-text generation tasks. However, the process of finetuning LLMs requires significant training resources and annotation work. In this paper, we explore the capability of generative models to generate descriptive text from graph data in a zero-shot setting. Specifically, we evaluate GPT-3 and ChatGPT on two graph-to-text datasets and compare their performance with that of finetuned LLM models such as T5 and BART. Our results demonstrate that generative models are capable of generating fluent and coherent text, achieving BLEU scores of 10.57 and 11.08 for the AGENDA and WebNLG datasets, respectively. However, our error analysis reveals that generative models still struggle with understanding the semantic relations between entities, and they also tend to generate text with hallucinations or irrelevant information. As a part of error analysis, we utilize BERT to detect machine-generated text and achieve high macro-F1 scores. We have made the text generated by generative models publicly available.

{{</citation>}}


### (44/64) Improving Natural Language Inference in Arabic using Transformer Models and Linguistically Informed Pre-Training (Mohammad Majd Saad Al Deen et al., 2023)

{{<citation>}}

Mohammad Majd Saad Al Deen, Maren Pielka, Jörn Hees, Bouthaina Soulef Abdou, Rafet Sifa. (2023)  
**Improving Natural Language Inference in Arabic using Transformer Models and Linguistically Informed Pre-Training**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NER, NLI, NLP, Named Entity Recognition, Natural Language Inference, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2307.14666v1)  

---


**ABSTRACT**  
This paper addresses the classification of Arabic text data in the field of Natural Language Processing (NLP), with a particular focus on Natural Language Inference (NLI) and Contradiction Detection (CD). Arabic is considered a resource-poor language, meaning that there are few data sets available, which leads to limited availability of NLP methods. To overcome this limitation, we create a dedicated data set from publicly available resources. Subsequently, transformer-based machine learning models are being trained and evaluated. We find that a language-specific model (AraBERT) performs competitively with state-of-the-art multilingual approaches, when we apply linguistically informed pre-training methods such as Named Entity Recognition (NER). To our knowledge, this is the first large-scale evaluation for this task in Arabic, as well as the first application of multi-task pre-training in this context.

{{</citation>}}


### (45/64) Metric-Based In-context Learning: A Case Study in Text Simplification (Subha Vadlamannati et al., 2023)

{{<citation>}}

Subha Vadlamannati, Gözde Gül Şahin. (2023)  
**Metric-Based In-context Learning: A Case Study in Text Simplification**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, GPT, GPT-6  
[Paper Link](http://arxiv.org/abs/2307.14632v1)  

---


**ABSTRACT**  
In-context learning (ICL) for large language models has proven to be a powerful approach for many natural language processing tasks. However, determining the best method to select examples for ICL is nontrivial as the results can vary greatly depending on the quality, quantity, and order of examples used. In this paper, we conduct a case study on text simplification (TS) to investigate how to select the best and most robust examples for ICL. We propose Metric-Based in-context Learning (MBL) method that utilizes commonly used TS metrics such as SARI, compression ratio, and BERT-Precision for selection. Through an extensive set of experiments with various-sized GPT models on standard TS benchmarks such as TurkCorpus and ASSET, we show that examples selected by the top SARI scores perform the best on larger models such as GPT-175B, while the compression ratio generally performs better on smaller models such as GPT-13B and GPT-6.7B. Furthermore, we demonstrate that MBL is generally robust to example orderings and out-of-domain test sets, and outperforms strong baselines and state-of-the-art finetuned language models. Finally, we show that the behaviour of large GPT models can be implicitly controlled by the chosen metric. Our research provides a new framework for selecting examples in ICL, and demonstrates its effectiveness in text simplification tasks, breaking new ground for more accurate and efficient NLG systems.

{{</citation>}}


## cs.SE (2)



### (46/64) Multilingual Code Co-Evolution Using Large Language Models (Jiyang Zhang et al., 2023)

{{<citation>}}

Jiyang Zhang, Pengyu Nie, Junyi Jessy Li, Milos Gligoric. (2023)  
**Multilingual Code Co-Evolution Using Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2307.14991v1)  

---


**ABSTRACT**  
Many software projects implement APIs and algorithms in multiple programming languages. Maintaining such projects is tiresome, as developers have to ensure that any change (e.g., a bug fix or a new feature) is being propagated, timely and without errors, to implementations in other programming languages. In the world of ever-changing software, using rule-based translation tools (i.e., transpilers) or machine learning models for translating code from one language to another provides limited value. Translating each time the entire codebase from one language to another is not the way developers work. In this paper, we target a novel task: translating code changes from one programming language to another using large language models (LLMs). We design and implement the first LLM, dubbed Codeditor, to tackle this task. Codeditor explicitly models code changes as edit sequences and learns to correlate changes across programming languages. To evaluate Codeditor, we collect a corpus of 6,613 aligned code changes from 8 pairs of open-source software projects implementing similar functionalities in two programming languages (Java and C#). Results show that Codeditor outperforms the state-of-the-art approaches by a large margin on all commonly used automatic metrics. Our work also reveals that Codeditor is complementary to the existing generation-based models, and their combination ensures even greater performance.

{{</citation>}}


### (47/64) New Interaction Paradigm for Complex EDA Software Leveraging GPT (Boyu Han et al., 2023)

{{<citation>}}

Boyu Han, Xinyu Wang, Yifan Wang, Junyu Yan, Yidong Tian. (2023)  
**New Interaction Paradigm for Complex EDA Software Leveraging GPT**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI, BERT, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.14740v1)  

---


**ABSTRACT**  
In the rapidly growing field of electronic design automation (EDA), professional software such as KiCad, Cadence , and Altium Designer provide increasingly extensive design functionalities. However, the intricate command structure and high learning curve create a barrier, particularly for novice printed circuit board (PCB) designers. This results in difficulties in selecting appropriate functions or plugins for varying design purposes, compounded by the lack of intuitive learning methods beyond traditional documentation, videos, and online forums. To address this challenge, an artificial intelligence (AI) interaction assist plugin for EDA software named SmartonAl is developed here, also KiCad is taken as the first example. SmartonAI is inspired by the HuggingGPT framework and employs large language models, such as GPT and BERT, to facilitate task planning and execution. On receiving a designer request, SmartonAI conducts a task breakdown and efficiently executes relevant subtasks, such as analysis of help documentation paragraphs and execution of different plugins, along with leveraging the built-in schematic and PCB manipulation functions in both SmartonAl itself and software. Our preliminary results demonstrate that SmartonAI can significantly streamline the PCB design process by simplifying complex commands into intuitive language-based interactions. By harnessing the powerful language capabilities of ChatGPT and the rich design functions of KiCad, the plugin effectively bridges the gap between complex EDA software and user-friendly interaction. Meanwhile, the new paradigm behind SmartonAI can also extend to other complex software systems, illustrating the immense potential of AI-assisted user interfaces in advancing digital interactions across various domains.

{{</citation>}}


## cs.LG (5)



### (48/64) Incrementally-Computable Neural Networks: Efficient Inference for Dynamic Inputs (Or Sharir et al., 2023)

{{<citation>}}

Or Sharir, Anima Anandkumar. (2023)  
**Incrementally-Computable Neural Networks: Efficient Inference for Dynamic Inputs**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.14988v1)  

---


**ABSTRACT**  
Deep learning often faces the challenge of efficiently processing dynamic inputs, such as sensor data or user inputs. For example, an AI writing assistant is required to update its suggestions in real time as a document is edited. Re-running the model each time is expensive, even with compression techniques like knowledge distillation, pruning, or quantization. Instead, we take an incremental computing approach, looking to reuse calculations as the inputs change. However, the dense connectivity of conventional architectures poses a major obstacle to incremental computation, as even minor input changes cascade through the network and restrict information reuse. To address this, we use vector quantization to discretize intermediate values in the network, which filters out noisy and unnecessary modifications to hidden neurons, facilitating the reuse of their values. We apply this approach to the transformers architecture, creating an efficient incremental inference algorithm with complexity proportional to the fraction of the modified inputs. Our experiments with adapting the OPT-125M pre-trained language model demonstrate comparable accuracy on document classification while requiring 12.1X (median) fewer operations for processing sequences of atomic edits.

{{</citation>}}


### (49/64) FLARE: Fingerprinting Deep Reinforcement Learning Agents using Universal Adversarial Masks (Buse G. A. Tekgul et al., 2023)

{{<citation>}}

Buse G. A. Tekgul, N. Asokan. (2023)  
**FLARE: Fingerprinting Deep Reinforcement Learning Agents using Universal Adversarial Masks**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.14751v1)  

---


**ABSTRACT**  
We propose FLARE, the first fingerprinting mechanism to verify whether a suspected Deep Reinforcement Learning (DRL) policy is an illegitimate copy of another (victim) policy. We first show that it is possible to find non-transferable, universal adversarial masks, i.e., perturbations, to generate adversarial examples that can successfully transfer from a victim policy to its modified versions but not to independently trained policies. FLARE employs these masks as fingerprints to verify the true ownership of stolen DRL policies by measuring an action agreement value over states perturbed via such masks. Our empirical evaluations show that FLARE is effective (100% action agreement on stolen copies) and does not falsely accuse independent policies (no false positives). FLARE is also robust to model modification attacks and cannot be easily evaded by more informed adversaries without negatively impacting agent performance. We also show that not all universal adversarial masks are suitable candidates for fingerprints due to the inherent characteristics of DRL policies. The spatio-temporal dynamics of DRL problems and sequential decision-making process make characterizing the decision boundary of DRL policies more difficult, as well as searching for universal masks that capture the geometry of it.

{{</citation>}}


### (50/64) TimeGNN: Temporal Dynamic Graph Learning for Time Series Forecasting (Nancy Xu et al., 2023)

{{<citation>}}

Nancy Xu, Chrysoula Kosma, Michalis Vazirgiannis. (2023)  
**TimeGNN: Temporal Dynamic Graph Learning for Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Time Series  
[Paper Link](http://arxiv.org/abs/2307.14680v1)  

---


**ABSTRACT**  
Time series forecasting lies at the core of important real-world applications in many fields of science and engineering. The abundance of large time series datasets that consist of complex patterns and long-term dependencies has led to the development of various neural network architectures. Graph neural network approaches, which jointly learn a graph structure based on the correlation of raw values of multivariate time series while forecasting, have recently seen great success. However, such solutions are often costly to train and difficult to scale. In this paper, we propose TimeGNN, a method that learns dynamic temporal graph representations that can capture the evolution of inter-series patterns along with the correlations of multiple series. TimeGNN achieves inference times 4 to 80 times faster than other state-of-the-art graph-based methods while achieving comparable forecasting performance

{{</citation>}}


### (51/64) Self-Contrastive Graph Diffusion Network (Yixian Ma et al., 2023)

{{<citation>}}

Yixian Ma, Kun Zhan. (2023)  
**Self-Contrastive Graph Diffusion Network**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, Augmentation  
[Paper Link](http://arxiv.org/abs/2307.14613v1)  

---


**ABSTRACT**  
Augmentation techniques and sampling strategies are crucial in contrastive learning, but in most existing works, augmentation techniques require careful design, and their sampling strategies can only capture a small amount of intrinsic supervision information. Additionally, the existing methods require complex designs to obtain two different representations of the data. To overcome these limitations, we propose a novel framework called the Self-Contrastive Graph Diffusion Network (SCGDN). Our framework consists of two main components: the Attentional Module (AttM) and the Diffusion Module (DiFM). AttM aggregates higher-order structure and feature information to get an excellent embedding, while DiFM balances the state of each node in the graph through Laplacian diffusion learning and allows the cooperative evolution of adjacency and feature information in the graph. Unlike existing methodologies, SCGDN is an augmentation-free approach that avoids "sampling bias" and semantic drift, without the need for pre-training. We conduct a high-quality sampling of samples based on structure and feature information. If two nodes are neighbors, they are considered positive samples of each other. If two disconnected nodes are also unrelated on $k$NN graph, they are considered negative samples for each other. The contrastive objective reasonably uses our proposed sampling strategies, and the redundancy reduction term minimizes redundant information in the embedding and can well retain more discriminative information. In this novel framework, the graph self-contrastive learning paradigm gives expression to a powerful force. SCGDN effectively balances between preserving high-order structure information and avoiding overfitting. The results manifest that SCGDN can consistently generate outperformance over both the contrastive methods and the classical methods.

{{</citation>}}


### (52/64) HUTFormer: Hierarchical U-Net Transformer for Long-Term Traffic Forecasting (Zezhi Shao et al., 2023)

{{<citation>}}

Zezhi Shao, Fei Wang, Zhao Zhang, Yuchen Fang, Guangyin Jin, Yongjun Xu. (2023)  
**HUTFormer: Hierarchical U-Net Transformer for Long-Term Traffic Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Transformer  
[Paper Link](http://arxiv.org/abs/2307.14596v1)  

---


**ABSTRACT**  
Traffic forecasting, which aims to predict traffic conditions based on historical observations, has been an enduring research topic and is widely recognized as an essential component of intelligent transportation. Recent proposals on Spatial-Temporal Graph Neural Networks (STGNNs) have made significant progress by combining sequential models with graph convolution networks. However, due to high complexity issues, STGNNs only focus on short-term traffic forecasting, e.g., 1-hour forecasting, while ignoring more practical long-term forecasting. In this paper, we make the first attempt to explore long-term traffic forecasting, e.g., 1-day forecasting. To this end, we first reveal its unique challenges in exploiting multi-scale representations. Then, we propose a novel Hierarchical U-net TransFormer (HUTFormer) to address the issues of long-term traffic forecasting. HUTFormer consists of a hierarchical encoder and decoder to jointly generate and utilize multi-scale representations of traffic data. Specifically, for the encoder, we propose window self-attention and segment merging to extract multi-scale representations from long-term traffic data. For the decoder, we design a cross-scale attention mechanism to effectively incorporate multi-scale representations. In addition, HUTFormer employs an efficient input embedding strategy to address the complexity issues. Extensive experiments on four traffic datasets show that the proposed HUTFormer significantly outperforms state-of-the-art traffic forecasting and long time series forecasting baselines.

{{</citation>}}


## cs.SI (1)



### (53/64) S$^3$: Social-network Simulation System with Large Language Model-Empowered Agents (Chen Gao et al., 2023)

{{<citation>}}

Chen Gao, Xiaochong Lan, Zhihong Lu, Jinzhu Mao, Jinghua Piao, Huandong Wang, Depeng Jin, Yong Li. (2023)  
**S$^3$: Social-network Simulation System with Large Language Model-Empowered Agents**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.14984v1)  

---


**ABSTRACT**  
Social network simulation plays a crucial role in addressing various challenges within social science. It offers extensive applications such as state prediction, phenomena explanation, and policy-making support, among others. In this work, we harness the formidable human-like capabilities exhibited by large language models (LLMs) in sensing, reasoning, and behaving, and utilize these qualities to construct the S$^3$ system (short for $\textbf{S}$ocial network $\textbf{S}$imulation $\textbf{S}$ystem). Adhering to the widely employed agent-based simulation paradigm, we employ prompt engineering and prompt tuning techniques to ensure that the agent's behavior closely emulates that of a genuine human within the social network. Specifically, we simulate three pivotal aspects: emotion, attitude, and interaction behaviors. By endowing the agent in the system with the ability to perceive the informational environment and emulate human actions, we observe the emergence of population-level phenomena, including the propagation of information, attitudes, and emotions. We conduct an evaluation encompassing two levels of simulation, employing real-world social network data. Encouragingly, the results demonstrate promising accuracy. This work represents an initial step in the realm of social network simulation empowered by LLM-based agents. We anticipate that our endeavors will serve as a source of inspiration for the development of simulation systems within, but not limited to, social science.

{{</citation>}}


## eess.IV (4)



### (54/64) Weakly Supervised AI for Efficient Analysis of 3D Pathology Samples (Andrew H. Song et al., 2023)

{{<citation>}}

Andrew H. Song, Mane Williams, Drew F. K. Williamson, Guillaume Jaume, Andrew Zhang, Bowen Chen, Robert Serafin, Jonathan T. C. Liu, Alex Baras, Anil V. Parwani, Faisal Mahmood. (2023)  
**Weakly Supervised AI for Efficient Analysis of 3D Pathology Samples**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV, q-bio-QM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.14907v1)  

---


**ABSTRACT**  
Human tissue and its constituent cells form a microenvironment that is fundamentally three-dimensional (3D). However, the standard-of-care in pathologic diagnosis involves selecting a few two-dimensional (2D) sections for microscopic evaluation, risking sampling bias and misdiagnosis. Diverse methods for capturing 3D tissue morphologies have been developed, but they have yet had little translation to clinical practice; manual and computational evaluations of such large 3D data have so far been impractical and/or unable to provide patient-level clinical insights. Here we present Modality-Agnostic Multiple instance learning for volumetric Block Analysis (MAMBA), a deep-learning-based platform for processing 3D tissue images from diverse imaging modalities and predicting patient outcomes. Archived prostate cancer specimens were imaged with open-top light-sheet microscopy or microcomputed tomography and the resulting 3D datasets were used to train risk-stratification networks based on 5-year biochemical recurrence outcomes via MAMBA. With the 3D block-based approach, MAMBA achieves an area under the receiver operating characteristic curve (AUC) of 0.86 and 0.74, superior to 2D traditional single-slice-based prognostication (AUC of 0.79 and 0.57), suggesting superior prognostication with 3D morphological features. Further analyses reveal that the incorporation of greater tissue volume improves prognostic performance and mitigates risk prediction variability from sampling bias, suggesting the value of capturing larger extents of heterogeneous 3D morphology. With the rapid growth and adoption of 3D spatial biology and pathology techniques by researchers and clinicians, MAMBA provides a general and efficient framework for 3D weakly supervised learning for clinical decision support and can help to reveal novel 3D morphological biomarkers for prognosis and therapeutic response.

{{</citation>}}


### (55/64) Understanding Silent Failures in Medical Image Classification (Till J. Bungert et al., 2023)

{{<citation>}}

Till J. Bungert, Levin Kobelke, Paul F. Jaeger. (2023)  
**Understanding Silent Failures in Medical Image Classification**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2307.14729v1)  

---


**ABSTRACT**  
To ensure the reliable use of classification systems in medical applications, it is crucial to prevent silent failures. This can be achieved by either designing classifiers that are robust enough to avoid failures in the first place, or by detecting remaining failures using confidence scoring functions (CSFs). A predominant source of failures in image classification is distribution shifts between training data and deployment data. To understand the current state of silent failure prevention in medical imaging, we conduct the first comprehensive analysis comparing various CSFs in four biomedical tasks and a diverse range of distribution shifts. Based on the result that none of the benchmarked CSFs can reliably prevent silent failures, we conclude that a deeper understanding of the root causes of failures in the data is required. To facilitate this, we introduce SF-Visuals, an interactive analysis tool that uses latent space clustering to visualize shifts and failures. On the basis of various examples, we demonstrate how this tool can help researchers gain insight into the requirements for safe application of classification systems in the medical domain. The open-source benchmark and tool are at: https://github.com/IML-DKFZ/sf-visuals.

{{</citation>}}


### (56/64) A Weakly Supervised Segmentation Network Embedding Cross-scale Attention Guidance and Noise-sensitive Constraint for Detecting Tertiary Lymphoid Structures of Pancreatic Tumors (Bingxue Wang et al., 2023)

{{<citation>}}

Bingxue Wang, Liwen Zou, Jun Chen, Yingying Cao, Zhenghua Cai, Yudong Qiu, Liang Mao, Zhongqiu Wang, Jingya Chen, Luying Gui, Xiaoping Yang. (2023)  
**A Weakly Supervised Segmentation Network Embedding Cross-scale Attention Guidance and Noise-sensitive Constraint for Detecting Tertiary Lymphoid Structures of Pancreatic Tumors**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention, Embedding  
[Paper Link](http://arxiv.org/abs/2307.14603v1)  

---


**ABSTRACT**  
The presence of tertiary lymphoid structures (TLSs) on pancreatic pathological images is an important prognostic indicator of pancreatic tumors. Therefore, TLSs detection on pancreatic pathological images plays a crucial role in diagnosis and treatment for patients with pancreatic tumors. However, fully supervised detection algorithms based on deep learning usually require a large number of manual annotations, which is time-consuming and labor-intensive. In this paper, we aim to detect the TLSs in a manner of few-shot learning by proposing a weakly supervised segmentation network. We firstly obtain the lymphocyte density maps by combining a pretrained model for nuclei segmentation and a domain adversarial network for lymphocyte nuclei recognition. Then, we establish a cross-scale attention guidance mechanism by jointly learning the coarse-scale features from the original histopathology images and fine-scale features from our designed lymphocyte density attention. A noise-sensitive constraint is introduced by an embedding signed distance function loss in the training procedure to reduce tiny prediction errors. Experimental results on two collected datasets demonstrate that our proposed method significantly outperforms the state-of-the-art segmentation-based algorithms in terms of TLSs detection accuracy. Additionally, we apply our method to study the congruent relationship between the density of TLSs and peripancreatic vascular invasion and obtain some clinically statistical results.

{{</citation>}}


### (57/64) MCPA: Multi-scale Cross Perceptron Attention Network for 2D Medical Image Segmentation (Liang Xu et al., 2023)

{{<citation>}}

Liang Xu, Mingxiao Chen, Yi Cheng, Pengfei Shao, Shuwei Shen, Peng Yao, Ronald X. Xu. (2023)  
**MCPA: Multi-scale Cross Perceptron Attention Network for 2D Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2307.14588v1)  

---


**ABSTRACT**  
The UNet architecture, based on Convolutional Neural Networks (CNN), has demonstrated its remarkable performance in medical image analysis. However, it faces challenges in capturing long-range dependencies due to the limited receptive fields and inherent bias of convolutional operations. Recently, numerous transformer-based techniques have been incorporated into the UNet architecture to overcome this limitation by effectively capturing global feature correlations. However, the integration of the Transformer modules may result in the loss of local contextual information during the global feature fusion process. To overcome these challenges, we propose a 2D medical image segmentation model called Multi-scale Cross Perceptron Attention Network (MCPA). The MCPA consists of three main components: an encoder, a decoder, and a Cross Perceptron. The Cross Perceptron first captures the local correlations using multiple Multi-scale Cross Perceptron modules, facilitating the fusion of features across scales. The resulting multi-scale feature vectors are then spatially unfolded, concatenated, and fed through a Global Perceptron module to model global dependencies. Furthermore, we introduce a Progressive Dual-branch Structure to address the semantic segmentation of the image involving finer tissue structures. This structure gradually shifts the segmentation focus of MCPA network training from large-scale structural features to more sophisticated pixel-level features. We evaluate our proposed MCPA model on several publicly available medical image datasets from different tasks and devices, including the open large-scale dataset of CT (Synapse), MRI (ACDC), fundus camera (DRIVE, CHASE_DB1, HRF), and OCTA (ROSE). The experimental results show that our MCPA model achieves state-of-the-art performance. The code is available at https://github.com/simonustc/MCPA-for-2D-Medical-Image-Segmentation.

{{</citation>}}


## cs.CR (2)



### (58/64) Smart Contract Migration: Security Analysis and Recommendations from Ethereum to Arbitrum (Xueyan Tang et al., 2023)

{{<citation>}}

Xueyan Tang, Lingzhi Shi, Alan Lai, Yuying Du, Jing Deng, Jialu Fu, Jiayi Li. (2023)  
**Smart Contract Migration: Security Analysis and Recommendations from Ethereum to Arbitrum**  

---
Primary Category: cs.CR  
Categories: 68-02 (Primary), J-2; E-3, cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.14773v1)  

---


**ABSTRACT**  
This research aims to explore the security risks posed by compatibility and protocol differences in smart contract migration, using the migration of smart contracts from Ethereum to Arbitrum as a case study. Through literature review, online data collection, expert participation, and analysis of smart contract vulnerability cases, this paper conducts an in-depth research of the differences between Ethereum and Arbitrum in areas such as Messaging, Block Properties, Contract Address Alias, and Gas Fees. The research findings indicate the presence of certain security issues during the migration process from Ethereum to Arbitrum, such as abnormal operation of the sequencer resulting in outdated off-chain data retrieval, time-based logical errors, failed permission checks, DOS attacks, and gas loss due to L1-to-L2 transaction failures. To address these security issues, this paper proposes corresponding solutions and recommendations to ensure the security and meet the requirements of the migration process. Additionally, this research emphasizes the continued attention and support for the security issues of smart contract migration through the case of smart contract migration from Ethereum to Arbitrum. It is worth noting that this research is the first in-depth research of smart contract security migration from Ethereum to Arbitrum.

{{</citation>}}


### (59/64) Backdoor Attacks for In-Context Learning with Language Models (Nikhil Kandpal et al., 2023)

{{<citation>}}

Nikhil Kandpal, Matthew Jagielski, Florian Tramèr, Nicholas Carlini. (2023)  
**Backdoor Attacks for In-Context Learning with Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.14692v1)  

---


**ABSTRACT**  
Because state-of-the-art language models are expensive to train, most practitioners must make use of one of the few publicly available language models or language model APIs. This consolidation of trust increases the potency of backdoor attacks, where an adversary tampers with a machine learning model in order to make it perform some malicious behavior on inputs that contain a predefined backdoor trigger. We show that the in-context learning ability of large language models significantly complicates the question of developing backdoor attacks, as a successful backdoor must work against various prompting strategies and should not affect the model's general purpose capabilities. We design a new attack for eliciting targeted misclassification when language models are prompted to perform a particular target task and demonstrate the feasibility of this attack by backdooring multiple large language models ranging in size from 1.3 billion to 6 billion parameters. Finally we study defenses to mitigate the potential harms of our attack: for example, while in the white-box setting we show that fine-tuning models for as few as 500 steps suffices to remove the backdoor behavior, in the black-box setting we are unable to develop a successful defense that relies on prompt engineering alone.

{{</citation>}}


## eess.AS (1)



### (60/64) The Effect of Spoken Language on Speech Enhancement using Self-Supervised Speech Representation Loss Functions (George Close et al., 2023)

{{<citation>}}

George Close, Thomas Hain, Stefan Goetze. (2023)  
**The Effect of Spoken Language on Speech Enhancement using Self-Supervised Speech Representation Loss Functions**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.14502v1)  

---


**ABSTRACT**  
Recent work in the field of speech enhancement (SE) has involved the use of self-supervised speech representations (SSSRs) as feature transformations in loss functions. However, in prior work, very little attention has been paid to the relationship between the language of the audio used to train the self-supervised representation and that used to train the SE system. Enhancement models trained using a loss function which incorporates a self-supervised representation that shares exactly the language of the noisy data used to train the SE system show better performance than those which do not match exactly. This may lead to enhancement systems which are language specific and as such do not generalise well to unseen languages, unlike models trained using traditional spectrogram or time domain loss functions. In this work, SE models are trained and tested on a number of different languages, with self-supervised representations which themselves are trained using different language combinations and with differing network structures as loss function representations. These models are then tested across unseen languages and their performances are analysed. It is found that the training language of the self-supervised representation appears to have a minor effect on enhancement performance, the amount of training data of a particular language, however, greatly affects performance.

{{</citation>}}


## cs.AI (1)



### (61/64) Fact-Checking of AI-Generated Reports (Razi Mahmood et al., 2023)

{{<citation>}}

Razi Mahmood, Ge Wang, Mannudeep Kalra, Pingkun Yan. (2023)  
**Fact-Checking of AI-Generated Reports**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CR, cs-CV, cs-LG, cs.AI, eess-IV  
Keywords: AI, Fact-Checking  
[Paper Link](http://arxiv.org/abs/2307.14634v1)  

---


**ABSTRACT**  
With advances in generative artificial intelligence (AI), it is now possible to produce realistic-looking automated reports for preliminary reads of radiology images. This can expedite clinical workflows, improve accuracy and reduce overall costs. However, it is also well-known that such models often hallucinate, leading to false findings in the generated reports. In this paper, we propose a new method of fact-checking of AI-generated reports using their associated images. Specifically, the developed examiner differentiates real and fake sentences in reports by learning the association between an image and sentences describing real or potentially fake findings. To train such an examiner, we first created a new dataset of fake reports by perturbing the findings in the original ground truth radiology reports associated with images. Text encodings of real and fake sentences drawn from these reports are then paired with image encodings to learn the mapping to real/fake labels. The utility of such an examiner is demonstrated for verifying automatically generated reports by detecting and removing fake sentences. Future generative AI approaches can use the resulting tool to validate their reports leading to a more responsible use of AI in expediting clinical workflows.

{{</citation>}}


## cs.IT (1)



### (62/64) Multi-Agent Graph Reinforcement Learning based On-Demand Wireless Energy Transfer in Multi-UAV-aided IoT Network (Ze Yu Zhao et al., 2023)

{{<citation>}}

Ze Yu Zhao, Yueling Che, Sheng Luo, Kaishun Wu, Victor C. M. Leung. (2023)  
**Multi-Agent Graph Reinforcement Learning based On-Demand Wireless Energy Transfer in Multi-UAV-aided IoT Network**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.14626v1)  

---


**ABSTRACT**  
This paper proposes a new on-demand wireless energy transfer (WET) scheme of multiple unmanned aerial vehicles (UAVs). Unlike the existing studies that simply pursuing the total or the minimum harvested energy maximization at the Internet of Things (IoT) devices, where the IoT devices' own energy requirements are barely considered, we propose a new metric called the hungry-level of energy (HoE), which reflects the time-varying energy demand of each IoT device based on the energy gap between its required energy and the harvested energy from the UAVs. With the purpose to minimize the overall HoE of the IoT devices whose energy requirements are not satisfied, we optimally determine all the UAVs' trajectories and WET decisions over time, under the practical mobility and energy constraints of the UAVs. Although the proposed problem is of high complexity to solve, by excavating the UAVs' self-attentions for their collaborative WET, we propose the multiagent graph reinforcement learning (MAGRL) based approach. Through the offline training of the MAGRL model, where the global training at the central controller guides the local training at each UAV agent, each UAV then distributively determines its trajectory and WET based on the well-trained local neural networks. Simulation results show that the proposed MAGRL-based approach outperforms various benchmarks for meeting the IoT devices' energy requirements.

{{</citation>}}


## q-bio.QM (1)



### (63/64) Explainable Techniques for Analyzing Flow Cytometry Cell Transformers (Florian Kowarsch et al., 2023)

{{<citation>}}

Florian Kowarsch, Lisa Weijler, FLorian Kleber, Matthias Wödlinger, Michael Reiter, Margarita Maurer-Granofszky, Michael Dworzak. (2023)  
**Explainable Techniques for Analyzing Flow Cytometry Cell Transformers**  

---
Primary Category: q-bio.QM  
Categories: cs-AI, q-bio-QM, q-bio.QM  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.14581v1)  

---


**ABSTRACT**  
Explainability for Deep Learning Models is especially important for clinical applications, where decisions of automated systems have far-reaching consequences.   While various post-hoc explainable methods, such as attention visualization and saliency maps, already exist for common data modalities, including natural language and images, little work has been done to adapt them to the modality of Flow CytoMetry (FCM) data.   In this work, we evaluate the usage of a transformer architecture called ReluFormer that ease attention visualization as well as we propose a gradient- and an attention-based visualization technique tailored for FCM. We qualitatively evaluate the visualization techniques for cell classification and polygon regression on pediatric Acute Lymphoblastic Leukemia (ALL) FCM samples. The results outline the model's decision process and demonstrate how to utilize the proposed techniques to inspect the trained model. The gradient-based visualization not only identifies cells that are most significant for a particular prediction but also indicates the directions in the FCM feature space in which changes have the most impact on the prediction. The attention visualization provides insights on the transformer's decision process when handling FCM data. We show that different attention heads specialize by attending to different biologically meaningful sub-populations in the data, even though the model retrieved solely supervised binary classification signals during training.

{{</citation>}}


## cs.RO (1)



### (64/64) Evaluation of Safety Constraints in Autonomous Navigation with Deep Reinforcement Learning (Brian Angulo et al., 2023)

{{<citation>}}

Brian Angulo, Gregory Gorbov, Aleksandr Panov, Konstantin Yakovlev. (2023)  
**Evaluation of Safety Constraints in Autonomous Navigation with Deep Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.14568v1)  

---


**ABSTRACT**  
While reinforcement learning algorithms have had great success in the field of autonomous navigation, they cannot be straightforwardly applied to the real autonomous systems without considering the safety constraints. The later are crucial to avoid unsafe behaviors of the autonomous vehicle on the road. To highlight the importance of these constraints, in this study, we compare two learnable navigation policies: safe and unsafe. The safe policy takes the constraints into account, while the other does not. We show that the safe policy is able to generate trajectories with more clearance (distance to the obstacles) and makes less collisions while training without sacrificing the overall performance.

{{</citation>}}
