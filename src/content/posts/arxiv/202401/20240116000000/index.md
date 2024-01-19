---
draft: false
title: "arXiv @ 2024.01.16"
date: 2024-01-16
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.16"
    identifier: arxiv_20240116
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (8)](#cscv-8)
- [cs.CY (2)](#cscy-2)
- [cs.CL (9)](#cscl-9)
- [cs.AI (4)](#csai-4)
- [cs.SE (2)](#csse-2)
- [cs.CR (1)](#cscr-1)
- [cs.NE (1)](#csne-1)
- [cs.HC (2)](#cshc-2)
- [math.NA (1)](#mathna-1)
- [stat.ML (2)](#statml-2)
- [cs.LG (5)](#cslg-5)
- [cs.IR (3)](#csir-3)
- [cs.AR (1)](#csar-1)
- [cs.DC (1)](#csdc-1)

## cs.CV (8)



### (1/42) A Strong Inductive Bias: Gzip for binary image classification (Marco Scilipoti et al., 2024)

{{<citation>}}

Marco Scilipoti, Marina Fuster, Rodrigo Ramele. (2024)  
**A Strong Inductive Bias: Gzip for binary image classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias, Computer Vision, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.07392v1)  

---


**ABSTRACT**  
Deep learning networks have become the de-facto standard in Computer Vision for industry and research. However, recent developments in their cousin, Natural Language Processing (NLP), have shown that there are areas where parameter-less models with strong inductive biases can serve as computationally cheaper and simpler alternatives. We propose such a model for binary image classification: a nearest neighbor classifier combined with a general purpose compressor like Gzip. We test and compare it against popular deep learning networks like Resnet, EfficientNet and Mobilenet and show that it achieves better accuracy and utilizes significantly less space, more than two order of magnitude, within a few-shot setting. As a result, we believe that this underlines the untapped potential of models with stronger inductive biases in few-shot scenarios.

{{</citation>}}


### (2/42) Harnessing Machine Learning for Discerning AI-Generated Synthetic Images (Yuyang Wang et al., 2024)

{{<citation>}}

Yuyang Wang, Yizhi Hao, Amando Xu Cong. (2024)  
**Harnessing Machine Learning for Discerning AI-Generated Synthetic Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07358v1)  

---


**ABSTRACT**  
In the realm of digital media, the advent of AI-generated synthetic images has introduced significant challenges in distinguishing between real and fabricated visual content. These images, often indistinguishable from authentic ones, pose a threat to the credibility of digital media, with potential implications for disinformation and fraud. Our research addresses this challenge by employing machine learning techniques to discern between AI-generated and genuine images. Central to our approach is the CIFAKE dataset, a comprehensive collection of images labeled as "Real" and "Fake". We refine and adapt advanced deep learning architectures like ResNet, VGGNet, and DenseNet, utilizing transfer learning to enhance their precision in identifying synthetic images. We also compare these with a baseline model comprising a vanilla Support Vector Machine (SVM) and a custom Convolutional Neural Network (CNN). The experimental results were significant, demonstrating that our optimized deep learning models outperform traditional methods, with DenseNet achieving an accuracy of 97.74%. Our application study contributes by applying and optimizing these advanced models for synthetic image detection, conducting a comparative analysis using various metrics, and demonstrating their superior capability in identifying AI-generated images over traditional machine learning techniques. This research not only advances the field of digital media integrity but also sets a foundation for future explorations into the ethical and technical dimensions of AI-generated content in digital media.

{{</citation>}}


### (3/42) Semi-supervised Semantic Segmentation using Redesigned Self-Training for White Blood Cel (Vinh Quoc Luu et al., 2024)

{{<citation>}}

Vinh Quoc Luu, Duy Khanh Le, Huy Thanh Nguyen, Minh Thanh Nguyen, Thinh Tien Nguyen, Vinh Quang Dinh. (2024)  
**Semi-supervised Semantic Segmentation using Redesigned Self-Training for White Blood Cel**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.07278v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) in healthcare, especially in white blood cell cancer diagnosis, is hindered by two primary challenges: the lack of large-scale labeled datasets for white blood cell (WBC) segmentation and outdated segmentation methods. To address the first challenge, a semi-supervised learning framework should be brought to efficiently annotate the large dataset. In this work, we address this issue by proposing a novel self-training pipeline with the incorporation of FixMatch. We discover that by incorporating FixMatch in the self-training pipeline, the performance improves in the majority of cases. Our performance achieved the best performance with the self-training scheme with consistency on DeepLab-V3 architecture and ResNet-50, reaching 90.69%, 87.37%, and 76.49% on Zheng 1, Zheng 2, and LISC datasets, respectively.

{{</citation>}}


### (4/42) SpineCLUE: Automatic Vertebrae Identification Using Contrastive Learning and Uncertainty Estimation (Sheng Zhang et al., 2024)

{{<citation>}}

Sheng Zhang, Minheng Chen, Junxian Wu, Ziyue Zhang, Tonglong Li, Cheng Xue, Youyong Kong. (2024)  
**SpineCLUE: Automatic Vertebrae Identification Using Contrastive Learning and Uncertainty Estimation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.07271v1)  

---


**ABSTRACT**  
Vertebrae identification in arbitrary fields-of-view plays a crucial role in diagnosing spine disease. Most spine CT contain only local regions, such as the neck, chest, and abdomen. Therefore, identification should not depend on specific vertebrae or a particular number of vertebrae being visible. Existing methods at the spine-level are unable to meet this challenge. In this paper, we propose a three-stage method to address the challenges in 3D CT vertebrae identification at vertebrae-level. By sequentially performing the tasks of vertebrae localization, segmentation, and identification, the anatomical prior information of the vertebrae is effectively utilized throughout the process. Specifically, we introduce a dual-factor density clustering algorithm to acquire localization information for individual vertebra, thereby facilitating subsequent segmentation and identification processes. In addition, to tackle the issue of interclass similarity and intra-class variability, we pre-train our identification network by using a supervised contrastive learning method. To further optimize the identification results, we estimated the uncertainty of the classification network and utilized the message fusion module to combine the uncertainty scores, while aggregating global information about the spine. Our method achieves state-of-the-art results on the VerSe19 and VerSe20 challenge benchmarks. Additionally, our approach demonstrates outstanding generalization performance on an collected dataset containing a wide range of abnormal cases.

{{</citation>}}


### (5/42) 3D Landmark Detection on Human Point Clouds: A Benchmark and A Dual Cascade Point Transformer Framework (Fan Zhang et al., 2024)

{{<citation>}}

Fan Zhang, Shuyi Mao, Qing Li, Xiaojiang Peng. (2024)  
**3D Landmark Detection on Human Point Clouds: A Benchmark and A Dual Cascade Point Transformer Framework**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.07251v1)  

---


**ABSTRACT**  
3D landmark detection plays a pivotal role in various applications such as 3D registration, pose estimation, and virtual try-on. While considerable success has been achieved in 2D human landmark detection or pose estimation, there is a notable scarcity of reported works on landmark detection in unordered 3D point clouds. This paper introduces a novel challenge, namely 3D landmark detection on human point clouds, presenting two primary contributions. Firstly, we establish a comprehensive human point cloud dataset, named HPoint103, designed to support the 3D landmark detection community. This dataset comprises 103 human point clouds created with commercial software and actors, each manually annotated with 11 stable landmarks. Secondly, we propose a Dual Cascade Point Transformer (D-CPT) model for precise point-based landmark detection. D-CPT gradually refines the landmarks through cascade Transformer decoder layers across the entire point cloud stream, simultaneously enhancing landmark coordinates with a RefineNet over local regions. Comparative evaluations with popular point-based methods on HPoint103 and the public dataset DHP19 demonstrate the dramatic outperformance of our D-CPT. Additionally, the integration of our RefineNet into existing methods consistently improves performance.

{{</citation>}}


### (6/42) MIMIC: Mask Image Pre-training with Mix Contrastive Fine-tuning for Facial Expression Recognition (Fan Zhang et al., 2024)

{{<citation>}}

Fan Zhang, Xiaobao Guo, Xiaojiang Peng, Alex Kot. (2024)  
**MIMIC: Mask Image Pre-training with Mix Contrastive Fine-tuning for Facial Expression Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.07245v1)  

---


**ABSTRACT**  
Cutting-edge research in facial expression recognition (FER) currently favors the utilization of convolutional neural networks (CNNs) backbone which is supervisedly pre-trained on face recognition datasets for feature extraction. However, due to the vast scale of face recognition datasets and the high cost associated with collecting facial labels, this pre-training paradigm incurs significant expenses. Towards this end, we propose to pre-train vision Transformers (ViTs) through a self-supervised approach on a mid-scale general image dataset. In addition, when compared with the domain disparity existing between face datasets and FER datasets, the divergence between general datasets and FER datasets is more pronounced. Therefore, we propose a contrastive fine-tuning approach to effectively mitigate this domain disparity. Specifically, we introduce a novel FER training paradigm named Mask Image pre-training with MIx Contrastive fine-tuning (MIMIC). In the initial phase, we pre-train the ViT via masked image reconstruction on general images. Subsequently, in the fine-tuning stage, we introduce a mix-supervised contrastive learning process, which enhances the model with a more extensive range of positive samples by the mixing strategy. Through extensive experiments conducted on three benchmark datasets, we demonstrate that our MIMIC outperforms the previous training paradigm, showing its capability to learn better representations. Remarkably, the results indicate that the vanilla ViT can achieve impressive performance without the need for intricate, auxiliary-designed modules. Moreover, when scaling up the model size, MIMIC exhibits no performance saturation and is superior to the current state-of-the-art methods.

{{</citation>}}


### (7/42) Enhanced Few-Shot Class-Incremental Learning via Ensemble Models (Mingli Zhu et al., 2024)

{{<citation>}}

Mingli Zhu, Zihao Zhu, Sihong Chen, Chen Chen, Baoyuan Wu. (2024)  
**Enhanced Few-Shot Class-Incremental Learning via Ensemble Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2401.07208v1)  

---


**ABSTRACT**  
Few-shot class-incremental learning (FSCIL) aims to continually fit new classes with limited training data, while maintaining the performance of previously learned classes. The main challenges are overfitting the rare new training samples and forgetting old classes. While catastrophic forgetting has been extensively studied, the overfitting problem has attracted less attention in FSCIL. To tackle overfitting challenge, we design a new ensemble model framework cooperated with data augmentation to boost generalization. In this way, the enhanced model works as a library storing abundant features to guarantee fast adaptation to downstream tasks. Specifically, the multi-input multi-output ensemble structure is applied with a spatial-aware data augmentation strategy, aiming at diversifying the feature extractor and alleviating overfitting in incremental sessions. Moreover, self-supervised learning is also integrated to further improve the model generalization. Comprehensive experimental results show that the proposed method can indeed mitigate the overfitting problem in FSCIL, and outperform the state-of-the-art methods.

{{</citation>}}


### (8/42) Left-right Discrepancy for Adversarial Attack on Stereo Networks (Pengfei Wang et al., 2024)

{{<citation>}}

Pengfei Wang, Xiaofei Hui, Beijia Lu, Nimrod Lilith, Jun Liu, Sameer Alam. (2024)  
**Left-right Discrepancy for Adversarial Attack on Stereo Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2401.07188v1)  

---


**ABSTRACT**  
Stereo matching neural networks often involve a Siamese structure to extract intermediate features from left and right images. The similarity between these intermediate left-right features significantly impacts the accuracy of disparity estimation. In this paper, we introduce a novel adversarial attack approach that generates perturbation noise specifically designed to maximize the discrepancy between left and right image features. Extensive experiments demonstrate the superior capability of our method to induce larger prediction errors in stereo neural networks, e.g. outperforming existing state-of-the-art attack methods by 219% MAE on the KITTI dataset and 85% MAE on the Scene Flow dataset. Additionally, we extend our approach to include a proxy network black-box attack method, eliminating the need for access to stereo neural network. This method leverages an arbitrary network from a different vision task as a proxy to generate adversarial noise, effectively causing the stereo network to produce erroneous predictions. Our findings highlight a notable sensitivity of stereo networks to discrepancies in shallow layer features, offering valuable insights that could guide future research in enhancing the robustness of stereo vision systems.

{{</citation>}}


## cs.CY (2)



### (9/42) How do machines learn? Evaluating the AIcon2abs method (Rubens Lacerda Queiroz et al., 2024)

{{<citation>}}

Rubens Lacerda Queiroz, Cabral Lima, Fabio Ferrentini Sampaio, Priscila Machado Vieira Lima. (2024)  
**How do machines learn? Evaluating the AIcon2abs method**  

---
Primary Category: cs.CY  
Categories: K-4-0; K-3-0, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07386v1)  

---


**ABSTRACT**  
This paper evaluates AIcon2abs (Queiroz et al., 2021), a recently proposed method that enables awareness among the general public on machine learning. Such is possible due to the use of WiSARD, an easily understandable machine learning mechanism, thus requiring little effort and no technical background from the target users. WiSARD is adherent to digital computing; training consists of writing to RAM-type memories, and classification consists of reading from these memories. The model enables easy visualization and understanding of training and classification tasks' internal realization through ludic activities. Furthermore, the WiSARD model does not require an Internet connection for training and classification, and it can learn from a few or one example. This feature makes it easier to observe the machine, increasing its accuracy on a particular task with each new example used. WiSARD can also create "mental images" of what it has learned so far, evidencing key features pertaining to a given class. The assessment of the AIcon2abs method's effectiveness was conducted through the evaluation of a remote course with a workload of approximately 6 hours. It was completed by thirty-four Brazilian subjects: 5 children between 8 and 11 years old; 5 adolescents between 12 and 17 years old; and 24 adults between 21 and 72 years old. Data analysis adopted a hybrid approach. AIcon2abs was well-rated by almost 100% of the research subjects, and the data collected revealed quite satisfactory results concerning the intended outcomes. This research has been approved by the CEP/HUCFF/FM/UFRJ Human Research Ethics Committee.

{{</citation>}}


### (10/42) Generative AI in EU Law: Liability, Privacy, Intellectual Property, and Cybersecurity (Claudio Novelli et al., 2024)

{{<citation>}}

Claudio Novelli, Federico Casolari, Philipp Hacker, Giorgio Spedicato, Luciano Floridi. (2024)  
**Generative AI in EU Law: Liability, Privacy, Intellectual Property, and Cybersecurity**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT, Generative AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.07348v1)  

---


**ABSTRACT**  
The advent of Generative AI, particularly through Large Language Models (LLMs) like ChatGPT and its successors, marks a paradigm shift in the AI landscape. Advanced LLMs exhibit multimodality, handling diverse data formats, thereby broadening their application scope. However, the complexity and emergent autonomy of these models introduce challenges in predictability and legal compliance. This paper delves into the legal and regulatory implications of Generative AI and LLMs in the European Union context, analyzing aspects of liability, privacy, intellectual property, and cybersecurity. It critically examines the adequacy of the existing and proposed EU legislation, including the Artificial Intelligence Act (AIA) draft, in addressing the unique challenges posed by Generative AI in general and LLMs in particular. The paper identifies potential gaps and shortcomings in the legislative framework and proposes recommendations to ensure the safe and compliant deployment of generative models, ensuring they align with the EU's evolving digital landscape and legal standards.

{{</citation>}}


## cs.CL (9)



### (11/42) DRLC: Reinforcement Learning with Dense Rewards from LLM Critic (Meng Cao et al., 2024)

{{<citation>}}

Meng Cao, Lei Shu, Lei Yu, Yun Zhu, Nevan Wichers, Yinxiao Liu, Lei Meng. (2024)  
**DRLC: Reinforcement Learning with Dense Rewards from LLM Critic**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.07382v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) can align language models with non-differentiable reward signals, such as human preferences. However, a major challenge arises from the sparsity of these reward signals - typically, there is only one reward for the entire generation. This sparsity of rewards can lead to inefficient and unstable learning. In this paper, we introduce a novel framework leveraging the critique ability of LLMs to produce dense rewards throughout the learning process. Our approach incorporates a critic language model alongside the policy model. This critic is prompted with the task description, question, policy model's output, and environment's reward signal as input, and provides token or span-level dense rewards that reflect the quality of each segment of the output. We assess our approach on three text generation tasks: sentiment control, language model detoxification, and summarization. Experimental results show that incorporating artificial dense rewards in training yields consistent performance gains over the PPO baseline with holistic rewards. Furthermore, in a setting where the same model serves as both policy and critic, we demonstrate that "self-critique" rewards also boost learning efficiency.

{{</citation>}}


### (12/42) Active Learning for NLP with Large Language Models (Xuesong Wang, 2024)

{{<citation>}}

Xuesong Wang. (2024)  
**Active Learning for NLP with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Active Learning, GPT, GPT-3.5, GPT-4, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.07367v1)  

---


**ABSTRACT**  
Human annotation of training samples is expensive, laborious, and sometimes challenging, especially for Natural Language Processing (NLP) tasks. To reduce the labeling cost and enhance the sample efficiency, Active Learning (AL) technique can be used to label as few samples as possible to reach a reasonable or similar results. To reduce even more costs and with the significant advances of Large Language Models (LLMs), LLMs can be a good candidate to annotate samples. This work investigates the accuracy and cost of using LLMs (GPT-3.5 and GPT-4) to label samples on 3 different datasets. A consistency-based strategy is proposed to select samples that are potentially incorrectly labeled so that human annotations can be used for those samples in AL settings, and we call it mixed annotation strategy. Then we test performance of AL under two different settings: (1) using human annotations only; (2) using the proposed mixed annotation strategy. The accuracy of AL models under 3 AL query strategies are reported on 3 text classification datasets, i.e., AG's News, TREC-6, and Rotten Tomatoes. On AG's News and Rotten Tomatoes, the models trained with the mixed annotation strategy achieves similar or better results compared to that with human annotations. The method reveals great potentials of LLMs as annotators in terms of accuracy and cost efficiency in active learning settings.

{{</citation>}}


### (13/42) PersonalityChat: Conversation Distillation for Personalized Dialog Modeling with Facts and Traits (Ehsan Lotfi et al., 2024)

{{<citation>}}

Ehsan Lotfi, Maxime De Bruyn, Jeska Buhmann, Walter Daelemans. (2024)  
**PersonalityChat: Conversation Distillation for Personalized Dialog Modeling with Facts and Traits**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Language Model  
[Paper Link](http://arxiv.org/abs/2401.07363v1)  

---


**ABSTRACT**  
The new wave of Large Language Models (LLM) has offered an efficient tool to curate sizeable conversational datasets. So far studies have mainly focused on task-oriented or generic open-domain dialogs, and have not fully explored the ability of LLMs in following complicated prompts. In this work, we focus on personalization, and employ LLMs to curate a dataset which is difficult and costly to crowd-source: PersonalityChat is a synthetic conversational dataset based upon the popular PersonaChat dataset, but conditioned on both personas and (Big-5) personality traits. Evaluating models fine-tuned on this dataset, we show that the personality trait labels can be used for trait-based personalization of generative dialogue models. We also perform a head-to-head comparison between PersonalityChat and PersonaChat, and show that training on the distilled dataset results in more fluent and coherent dialog agents in the small-model regime.

{{</citation>}}


### (14/42) Promptformer: Prompted Conformer Transducer for ASR (Sergio Duarte-Torres et al., 2024)

{{<citation>}}

Sergio Duarte-Torres, Arunasish Sen, Aman Rana, Lukas Drude, Alejandro Gomez-Alanis, Andreas Schwarz, Leif Rädel, Volker Leutnant. (2024)  
**Promptformer: Prompted Conformer Transducer for ASR**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.07360v1)  

---


**ABSTRACT**  
Context cues carry information which can improve multi-turn interactions in automatic speech recognition (ASR) systems. In this paper, we introduce a novel mechanism inspired by hyper-prompting to fuse textual context with acoustic representations in the attention mechanism. Results on a test set with multi-turn interactions show that our method achieves 5.9% relative word error rate reduction (rWERR) over a strong baseline. We show that our method does not degrade in the absence of context and leads to improvements even if the model is trained without context. We further show that leveraging a pre-trained sentence-piece model for context embedding generation can outperform an external BERT model.

{{</citation>}}


### (15/42) ELLA-V: Stable Neural Codec Language Modeling with Alignment-guided Sequence Reordering (Yakun Song et al., 2024)

{{<citation>}}

Yakun Song, Zhuo Chen, Xiaofei Wang, Ziyang Ma, Xie Chen. (2024)  
**ELLA-V: Stable Neural Codec Language Modeling with Alignment-guided Sequence Reordering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07333v1)  

---


**ABSTRACT**  
The language model (LM) approach based on acoustic and linguistic prompts, such as VALL-E, has achieved remarkable progress in the field of zero-shot audio generation. However, existing methods still have some limitations: 1) repetitions, transpositions, and omissions in the output synthesized speech due to limited alignment constraints between audio and phoneme tokens; 2) challenges of fine-grained control over the synthesized speech with autoregressive (AR) language model; 3) infinite silence generation due to the nature of AR-based decoding, especially under the greedy strategy. To alleviate these issues, we propose ELLA-V, a simple but efficient LM-based zero-shot text-to-speech (TTS) framework, which enables fine-grained control over synthesized audio at the phoneme level. The key to ELLA-V is interleaving sequences of acoustic and phoneme tokens, where phoneme tokens appear ahead of the corresponding acoustic tokens. The experimental findings reveal that our model outperforms VALL-E in terms of accuracy and delivers more stable results using both greedy and sampling-based decoding strategies. The code of ELLA-V will be open-sourced after cleanups. Audio samples are available at https://ereboas.github.io/ELLAV/.

{{</citation>}}


### (16/42) Harnessing Large Language Models Over Transformer Models for Detecting Bengali Depressive Social Media Text: A Comprehensive Study (Ahmadul Karim Chowdhury et al., 2024)

{{<citation>}}

Ahmadul Karim Chowdhury, Md. Saidur Rahman Sujon, Md. Shirajus Salekin Shafi, Tasin Ahmmad, Sifat Ahmed, Khan Md Hasib, Faisal Muhammad Shah. (2024)  
**Harnessing Large Language Models Over Transformer Models for Detecting Bengali Depressive Social Media Text: A Comprehensive Study**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT, GPT-3.5, LSTM, Language Model, Social Media, Transformer  
[Paper Link](http://arxiv.org/abs/2401.07310v1)  

---


**ABSTRACT**  
In an era where the silent struggle of underdiagnosed depression pervades globally, our research delves into the crucial link between mental health and social media. This work focuses on early detection of depression, particularly in extroverted social media users, using LLMs such as GPT 3.5, GPT 4 and our proposed GPT 3.5 fine-tuned model DepGPT, as well as advanced Deep learning models(LSTM, Bi-LSTM, GRU, BiGRU) and Transformer models(BERT, BanglaBERT, SahajBERT, BanglaBERT-Base). The study categorized Reddit and X datasets into "Depressive" and "Non-Depressive" segments, translated into Bengali by native speakers with expertise in mental health, resulting in the creation of the Bengali Social Media Depressive Dataset (BSMDD). Our work provides full architecture details for each model and a methodical way to assess their performance in Bengali depressive text categorization using zero-shot and few-shot learning techniques. Our work demonstrates the superiority of SahajBERT and Bi-LSTM with FastText embeddings in their respective domains also tackles explainability issues with transformer models and emphasizes the effectiveness of LLMs, especially DepGPT, demonstrating flexibility and competence in a range of learning contexts. According to the experiment results, the proposed model, DepGPT, outperformed not only Alpaca Lora 7B in zero-shot and few-shot scenarios but also every other model, achieving a near-perfect accuracy of 0.9796 and an F1-score of 0.9804, high recall, and exceptional precision. Although competitive, GPT-3.5 Turbo and Alpaca Lora 7B show relatively poorer effectiveness in zero-shot and few-shot situations. The work emphasizes the effectiveness and flexibility of LLMs in a variety of linguistic circumstances, providing insightful information about the complex field of depression detection models.

{{</citation>}}


### (17/42) Small Language Model Can Self-correct (Haixia Han et al., 2024)

{{<citation>}}

Haixia Han, Jiaqing Liang, Jie Shi, Qianyu He, Yanghua Xiao. (2024)  
**Small Language Model Can Self-correct**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.07301v1)  

---


**ABSTRACT**  
Generative Language Models (LMs) such as ChatGPT have exhibited remarkable performance across various downstream tasks. Nevertheless, one of their most prominent drawbacks is generating inaccurate or false information with a confident tone. Previous studies have devised sophisticated pipelines and prompts to induce large LMs to exhibit the capability for self-correction. However, large LMs are explicitly prompted to verify and modify its answers separately rather than completing all steps spontaneously like humans. Moreover, these complex prompts are extremely challenging for small LMs to follow. In this paper, we introduce the \underline{I}ntrinsic \underline{S}elf-\underline{C}orrection (ISC) in generative language models, aiming to correct the initial output of LMs in a self-triggered manner, even for those small LMs with 6 billion parameters. Specifically, we devise a pipeline for constructing self-correction data and propose Partial Answer Masking (PAM), aiming to endow the model with the capability for intrinsic self-correction through fine-tuning. We conduct experiments using LMs with parameters sizes ranging from 6 billion to 13 billion in two tasks, including commonsense reasoning and factual knowledge reasoning. Our experiments demonstrate that the outputs generated using ISC outperform those generated without self-correction. We believe that the output quality of even small LMs can be further improved by empowering them with the ability to intrinsic self-correct.

{{</citation>}}


### (18/42) CANDLE: Iterative Conceptualization and Instantiation Distillation from Large Language Models for Commonsense Reasoning (Weiqi Wang et al., 2024)

{{<citation>}}

Weiqi Wang, Tianqing Fang, Chunyang Li, Haochen Shi, Wenxuan Ding, Baixuan Xu, Zhaowei Wang, Jiaxin Bai, Xin Liu, Jiayang Cheng, Chunkit Chan, Yangqiu Song. (2024)  
**CANDLE: Iterative Conceptualization and Instantiation Distillation from Large Language Models for Commonsense Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.07286v1)  

---


**ABSTRACT**  
The sequential process of conceptualization and instantiation is essential to generalizable commonsense reasoning as it allows the application of existing knowledge to unfamiliar scenarios. However, existing works tend to undervalue the step of instantiation and heavily rely on pre-built concept taxonomies and human annotations to collect both types of knowledge, resulting in a lack of instantiated knowledge to complete reasoning, high cost, and limited scalability. To tackle these challenges, we introduce CANDLE, a distillation framework that iteratively performs contextualized conceptualization and instantiation over commonsense knowledge bases by instructing large language models to generate both types of knowledge with critic filtering. By applying CANDLE to ATOMIC, we construct a comprehensive knowledge base comprising six million conceptualizations and instantiated commonsense knowledge triples. Both types of knowledge are firmly rooted in the original ATOMIC dataset, and intrinsic evaluations demonstrate their exceptional quality and diversity. Empirical results indicate that distilling CANDLE on student models provides benefits across four downstream tasks. Our code, data, and models are publicly available at https://github.com/HKUST-KnowComp/CANDLE.

{{</citation>}}


### (19/42) Distilling Event Sequence Knowledge From Large Language Models (Somin Wadhwa et al., 2024)

{{<citation>}}

Somin Wadhwa, Oktie Hassanzadeh, Debarun Bhattacharjya, Ken Barker, Jian Ni. (2024)  
**Distilling Event Sequence Knowledge From Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2401.07237v1)  

---


**ABSTRACT**  
Event sequence models have been found to be highly effective in the analysis and prediction of events. Building such models requires availability of abundant high-quality event sequence data. In certain applications, however, clean structured event sequences are not available, and automated sequence extraction results in data that is too noisy and incomplete. In this work, we explore the use of Large Language Models (LLMs) to generate event sequences that can effectively be used for probabilistic event model construction. This can be viewed as a mechanism of distilling event sequence knowledge from LLMs. Our approach relies on a Knowledge Graph (KG) of event concepts with partial causal relations to guide the generative language model for causal event sequence generation. We show that our approach can generate high-quality event sequences, filling a knowledge gap in the input KG. Furthermore, we explore how the generated sequences can be leveraged to discover useful and more complex structured knowledge from pattern mining and probabilistic event models. We release our sequence generation code and evaluation framework, as well as corpus of event sequence data.

{{</citation>}}


## cs.AI (4)



### (20/42) Reliability and Interpretability in Science and Deep Learning (Luigi Scorzato, 2024)

{{<citation>}}

Luigi Scorzato. (2024)  
**Reliability and Interpretability in Science and Deep Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI, physics-hist-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07359v1)  

---


**ABSTRACT**  
In recent years, the question of the reliability of Machine Learning (ML) methods has acquired significant importance, and the analysis of the associated uncertainties has motivated a growing amount of research. However, most of these studies have applied standard error analysis to ML models, and in particular Deep Neural Network (DNN) models, which represent a rather significant departure from standard scientific modelling. It is therefore necessary to integrate the standard error analysis with a deeper epistemological analysis of the possible differences between DNN models and standard scientific modelling and the possible implications of these differences in the assessment of reliability. This article offers several contributions. First, it emphasises the ubiquitous role of model assumptions (both in ML and traditional Science) against the illusion of theory-free science. Secondly, model assumptions are analysed from the point of view of their (epistemic) complexity, which is shown to be language-independent. It is argued that the high epistemic complexity of DNN models hinders the estimate of their reliability and also their prospect of long-term progress. Some potential ways forward are suggested. Thirdly, this article identifies the close relation between a model's epistemic complexity and its interpretability, as introduced in the context of responsible AI. This clarifies in which sense, and to what extent, the lack of understanding of a model (black-box problem) impacts its interpretability in a way that is independent of individual skills. It also clarifies how interpretability is a precondition for assessing the reliability of any model, which cannot be based on statistical analysis alone. This article focuses on the comparison between traditional scientific models and DNN models. But, Random Forest and Logistic Regression models are also briefly considered.

{{</citation>}}


### (21/42) Small LLMs Are Weak Tool Learners: A Multi-LLM Agent (Weizhou Shen et al., 2024)

{{<citation>}}

Weizhou Shen, Chenliang Li, Hongzhan Chen, Ming Yan, Xiaojun Quan, Hehong Chen, Ji Zhang, Fei Huang. (2024)  
**Small LLMs Are Weak Tool Learners: A Multi-LLM Agent**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07324v1)  

---


**ABSTRACT**  
Large Language Model (LLM) agents significantly extend the capabilities of standalone LLMs, empowering them to interact with external tools (e.g., APIs, functions) and complete complex tasks in a self-directed fashion. The challenge of tool use demands that LLMs not only understand user queries and generate answers but also excel in task planning, memory management, tool invocation, and result summarization. While traditional approaches focus on training a single LLM with all these capabilities, performance limitations become apparent, particularly with smaller models. Moreover, the entire LLM may require retraining when tools are updated. To overcome these challenges, we propose a novel strategy that decomposes the aforementioned capabilities into a planner, caller, and summarizer. Each component is implemented by a single LLM that focuses on a specific capability and collaborates with other components to accomplish the task. This modular framework facilitates individual updates and the potential use of smaller LLMs for building each capability. To effectively train this framework, we introduce a two-stage training paradigm. First, we fine-tune a backbone LLM on the entire dataset without discriminating sub-tasks, providing the model with a comprehensive understanding of the task. Second, the fine-tuned LLM is used to instantiate the planner, caller, and summarizer respectively, which are continually fine-tuned on respective sub-tasks. Evaluation across various tool-use benchmarks illustrates that our proposed multi-LLM framework surpasses the traditional single-LLM approach, highlighting its efficacy and advantages in tool learning.

{{</citation>}}


### (22/42) MapGPT: Map-Guided Prompting for Unified Vision-and-Language Navigation (Jiaqi Chen et al., 2024)

{{<citation>}}

Jiaqi Chen, Bingqian Lin, Ran Xu, Zhenhua Chai, Xiaodan Liang, Kwan-Yee K. Wong. (2024)  
**MapGPT: Map-Guided Prompting for Unified Vision-and-Language Navigation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs-RO, cs.AI  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2401.07314v1)  

---


**ABSTRACT**  
Embodied agents equipped with GPT as their brain have exhibited extraordinary thinking and decision-making abilities across various tasks. However, existing zero-shot agents for vision-and-language navigation (VLN) only prompt the GPT to handle excessive environmental information and select potential locations within localized environments, without constructing an effective ''global-view'' (e.g., a commonly-used map) for the agent to understand the overall environment. In this work, we present a novel map-guided GPT-based path-planning agent, dubbed MapGPT, for the zero-shot VLN task. Specifically, we convert a topological map constructed online into prompts to encourage map-guided global exploration, and require the agent to explicitly output and update multi-step path planning to avoid getting stuck in local exploration. Extensive experiments demonstrate that our MapGPT is effective, achieving impressive performance on both the R2R and REVERIE datasets (38.8% and 28.4% success rate, respectively) and showcasing the newly emerged global thinking and path planning capabilities of the GPT model. Unlike previous VLN agents, which require separate parameters fine-tuning or specific prompt design to accommodate various instruction styles across different datasets, our MapGPT is more unified as it can adapt to different instruction styles seamlessly, which is the first of its kind in this field.

{{</citation>}}


### (23/42) Enabling Collaborative Clinical Diagnosis of Infectious Keratitis by Integrating Expert Knowledge and Interpretable Data-driven Intelligence (Zhengqing Fang et al., 2024)

{{<citation>}}

Zhengqing Fang, Shuowen Zhou, Zhouhang Yuan, Yuxuan Si, Mengze Li, Jinxu Li, Yesheng Xu, Wenjia Xie, Kun Kuang, Yingming Li, Fei Wu, Yu-Feng Yao. (2024)  
**Enabling Collaborative Clinical Diagnosis of Infectious Keratitis by Integrating Expert Knowledge and Interpretable Data-driven Intelligence**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs-HC, cs.AI  
Keywords: AI, Clinical  
[Paper Link](http://arxiv.org/abs/2401.08695v1)  

---


**ABSTRACT**  
Although data-driven artificial intelligence (AI) in medical image diagnosis has shown impressive performance in silico, the lack of interpretability makes it difficult to incorporate the "black box" into clinicians' workflows. To make the diagnostic patterns learned from data understandable by clinicians, we develop an interpretable model, knowledge-guided diagnosis model (KGDM), that provides a visualized reasoning process containing AI-based biomarkers and retrieved cases that with the same diagnostic patterns. It embraces clinicians' prompts into the interpreted reasoning through human-AI interaction, leading to potentially enhanced safety and more accurate predictions. This study investigates the performance, interpretability, and clinical utility of KGDM in the diagnosis of infectious keratitis (IK), which is the leading cause of corneal blindness. The classification performance of KGDM is evaluated on a prospective validation dataset, an external testing dataset, and an publicly available testing dataset. The diagnostic odds ratios (DOR) of the interpreted AI-based biomarkers are effective, ranging from 3.011 to 35.233 and exhibit consistent diagnostic patterns with clinic experience. Moreover, a human-AI collaborative diagnosis test is conducted and the participants with collaboration achieved a performance exceeding that of both humans and AI. By synergistically integrating interpretability and interaction, this study facilitates the convergence of clinicians' expertise and data-driven intelligence. The promotion of inexperienced ophthalmologists with the aid of AI-based biomarkers, as well as increased AI prediction by intervention from experienced ones, demonstrate a promising diagnostic paradigm for infectious keratitis using KGDM, which holds the potential for extension to other diseases where experienced medical practitioners are limited and the safety of AI is concerned.

{{</citation>}}


## cs.SE (2)



### (24/42) Towards Engineering Fair and Equitable Software Systems for Managing Low-Altitude Airspace Authorizations (Usman Gohar et al., 2024)

{{<citation>}}

Usman Gohar, Michael C. Hunter, Agnieszka Marczak-Czajka, Robyn R. Lutz, Myra B. Cohen, Jane Cleland-Huang. (2024)  
**Towards Engineering Fair and Equitable Software Systems for Managing Low-Altitude Airspace Authorizations**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-LG, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07353v1)  

---


**ABSTRACT**  
Small Unmanned Aircraft Systems (sUAS) have gained widespread adoption across a diverse range of applications. This has introduced operational complexities within shared airspaces and an increase in reported incidents, raising safety concerns. In response, the U.S. Federal Aviation Administration (FAA) is developing a UAS Traffic Management (UTM) system to control access to airspace based on an sUAS's predicted ability to safely complete its mission. However, a fully automated system capable of swiftly approving or denying flight requests can be prone to bias and must consider safety, transparency, and fairness to diverse stakeholders. In this paper, we present an initial study that explores stakeholders' perspectives on factors that should be considered in an automated system. Results indicate flight characteristics and environmental conditions were perceived as most important but pilot and drone capabilities should also be considered. Further, several respondents indicated an aversion to any AI-supported automation, highlighting the need for full transparency in automated decision-making. Results provide a societal perspective on the challenges of automating UTM flight authorization decisions and help frame the ongoing design of a solution acceptable to the broader sUAS community.

{{</citation>}}


### (25/42) CodeAgent: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level Coding Challenges (Kechi Zhang et al., 2024)

{{<citation>}}

Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, Zhi Jin. (2024)  
**CodeAgent: Enhancing Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level Coding Challenges**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07339v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown promise in automated code generation but typically excel only in simpler tasks such as generating standalone code units. Real-world software development, however, often involves complex code repositories (named repo) with complex dependencies and extensive documentation. To fill this gap, our research pivots towards evaluating LLMs in a more realistic setting -- real-world repo-level code generation. We introduce CodeAgentBench, a manually curated benchmark for repo-level code generation. This benchmark comprises five high-quality Python projects, encompassing a total of 101 samples. We assess nine leading LLMs on repo-level tasks and observe a decline in their performance. To tackle this, we present CodeAgent, a novel LLM-based agent framework that employs external tools for effective repo-level code generation. CodeAgent integrates five programming tools, enabling interaction with software artifacts for information retrieval, code symbol navigation, and code testing. We implement four agent strategies to optimize these tools' usage. Our experiments on CodeAgentBench show that CodeAgent enhances LLM performance significantly, with improvements ranging from 18.1\% to 250\%. Further tests on the HumanEval benchmark confirm CodeAgent's adaptability and efficacy across various code generation tasks. Notably, CodeAgent outperforms commercial products like Github Copilot, showcasing superior accuracy and efficiency. These results demonstrate CodeAgent's robust capabilities in code generation, highlighting its potential for real-world repo-level coding challenges.

{{</citation>}}


## cs.CR (1)



### (26/42) Privacy-Preserving Intrusion Detection in Software-defined VANET using Federated Learning with BERT (Shakil Ibne Ahsan et al., 2024)

{{<citation>}}

Shakil Ibne Ahsan, Phil Legg, S M Iftekharul Alam. (2024)  
**Privacy-Preserving Intrusion Detection in Software-defined VANET using Federated Learning with BERT**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: BERT, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2401.07343v2)  

---


**ABSTRACT**  
The absence of robust security protocols renders the VANET (Vehicle ad-hoc Networks) network open to cyber threats by compromising passengers and road safety. Intrusion Detection Systems (IDS) are widely employed to detect network security threats. With vehicles' high mobility on the road and diverse environments, VANETs devise ever-changing network topologies, lack privacy and security, and have limited bandwidth efficiency. The absence of privacy precautions, End-to-End Encryption methods, and Local Data Processing systems in VANET also present many privacy and security difficulties. So, assessing whether a novel real-time processing IDS approach can be utilized for this emerging technology is crucial. The present study introduces a novel approach for intrusion detection using Federated Learning (FL) capabilities in conjunction with the BERT model for sequence classification (FL-BERT). The significance of data privacy is duly recognized. According to FL methodology, each client has its own local model and dataset. They train their models locally and then send the model's weights to the server. After aggregation, the server aggregates the weights from all clients to update a global model. After aggregation, the global model's weights are shared with the clients. This practice guarantees the secure storage of sensitive raw data on individual clients' devices, effectively protecting privacy. After conducting the federated learning procedure, we assessed our models' performance using a separate test dataset. The FL-BERT technique has yielded promising results, opening avenues for further investigation in this particular area of research. We reached the result of our approaches by comparing existing research works and found that FL-BERT is more effective for privacy and security concerns. Our results suggest that FL-BERT is a promising technique for enhancing attack detection.

{{</citation>}}


## cs.NE (1)



### (27/42) Attention-based UNet enabled Lightweight Image Semantic Communication System over Internet of Things (Guoxin Ma et al., 2024)

{{<citation>}}

Guoxin Ma, Haonan Tong, Nuocheng Yang, Changchuan Yin. (2024)  
**Attention-based UNet enabled Lightweight Image Semantic Communication System over Internet of Things**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.07329v1)  

---


**ABSTRACT**  
This paper studies the problem of the lightweight image semantic communication system that is deployed on Internet of Things (IoT) devices. In the considered system model, devices must use semantic communication techniques to support user behavior recognition in ultimate video service with high data transmission efficiency. However, it is computationally expensive for IoT devices to deploy semantic codecs due to the complex calculation processes of deep learning (DL) based codec training and inference. To make it affordable for IoT devices to deploy semantic communication systems, we propose an attention-based UNet enabled lightweight image semantic communication (LSSC) system, which achieves low computational complexity and small model size. In particular, we first let the LSSC system train the codec at the edge server to reduce the training computation load on IoT devices. Then, we introduce the convolutional block attention module (CBAM) to extract the image semantic features and decrease the number of downsampling layers thus reducing the floating-point operations (FLOPs). Finally, we experimentally adjust the structure of the codec and find out the optimal number of downsampling layers. Simulation results show that the proposed LSSC system can reduce the semantic codec FLOPs by 14%, and reduce the model size by 55%, with a sacrifice of 3% accuracy, compared to the baseline. Moreover, the proposed scheme can achieve a higher transmission accuracy than the traditional communication scheme in the low channel signal-to-noise (SNR) region.

{{</citation>}}


## cs.HC (2)



### (28/42) Understanding Nonlinear Collaboration between Human and AI Agents: A Co-design Framework for Creative Design (JiayiZhou. Renzhong Li et al., 2024)

{{<citation>}}

JiayiZhou. Renzhong Li, Junxiu Tang, Tan Tang, Haotian Li, Weiwei Cui, Yingcaui Wu. (2024)  
**Understanding Nonlinear Collaboration between Human and AI Agents: A Co-design Framework for Creative Design**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07312v1)  

---


**ABSTRACT**  
Creative design is a nonlinear process where designers generate diverse ideas in the pursuit of an open-ended goal and converge towards consensus through iterative remixing. In contrast, AI-powered design tools often employ a linear sequence of incremental and precise instructions to approximate design objectives. Such operations violate customary creative design practices and thus hinder AI agents' ability to complete creative design tasks. To explore better human-AI co-design tools, we first summarize human designers' practices through a formative study with 12 design experts. Taking graphic design as a representative scenario, we formulate a nonlinear human-AI co-design framework and develop a proof-of-concept prototype, OptiMuse. We evaluate OptiMuse and validate the nonlinear framework through a comparative study. We notice a subconscious change in people's attitudes towards AI agents, shifting from perceiving them as mere executors to regarding them as opinionated colleagues. This shift effectively fostered the exploration and reflection processes of individual designers.

{{</citation>}}


### (29/42) Understanding Emotional Disclosure via Diary-keeping in Quarantine on Social Media (Yue Deng et al., 2024)

{{<citation>}}

Yue Deng, Changyang He, Bo Li. (2024)  
**Understanding Emotional Disclosure via Diary-keeping in Quarantine on Social Media**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-SI, cs.HC  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2401.07230v1)  

---


**ABSTRACT**  
Quarantine is a widely-adopted measure during health crises caused by highly-contagious diseases like COVID-19, yet it poses critical challenges to public mental health. Given this context, emotional disclosure on social media in the form of keeping a diary emerges as a popular way for individuals to express emotions and record their mental health status. However, the exploration of emotional disclosure via diary-keeping on social media during quarantine is underexplored, understanding which could be beneficial to facilitate emotional connections and enlighten health intervention measures. Focusing on this particular form of self-disclosure, this work proposes a quantitative approach to figure out the prevalence and changing patterns of emotional disclosure during quarantine, and the possible factors contributing to the negative emotions. We collected 58, 796 posts with the "Quarantine Diary" keyword on Weibo, a popular social media website in China. Through text classification, we capture diverse emotion categories that characterize public emotion disclosure during quarantine, such as annoyed, anxious, boring, happy, hopeful and appreciative. Based on temporal analysis, we uncover the changing patterns of emotional disclosure from long-term perspectives and period-based perspectives (e.g., the gradual decline of all negative emotions and the upsurge of the annoyed emotion near the end of quarantine). Leveraging topic modeling, we also encapsulate the possible influencing factors of negative emotions, such as freedom restriction and solitude, and uncertainty of infection and supply. We reflect on how our findings could deepen the understanding of mental health on social media and further provide practical and design implications to mitigate mental health issues during quarantine.

{{</citation>}}


## math.NA (1)



### (30/42) Multi-Physics Model Bias Correction with Data-Driven Reduced Order Modelling Techniques: Application to Nuclear Case Studies (Stefano Riva et al., 2024)

{{<citation>}}

Stefano Riva, Carolina Introini, Antonio Cammi. (2024)  
**Multi-Physics Model Bias Correction with Data-Driven Reduced Order Modelling Techniques: Application to Nuclear Case Studies**  

---
Primary Category: math.NA  
Categories: cs-NA, math-NA, math.NA  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.07300v1)  

---


**ABSTRACT**  
Nowadays, interest in combining mathematical knowledge about phenomena and data from the physical system is growing. Past research was devoted to developing so-called high-fidelity models, intending to make them able to catch most of the physical phenomena occurring in the system. Nevertheless, models will always be affected by uncertainties related, for example, to the parameters and inevitably limited by the underlying simplifying hypotheses on, for example, geometry and mathematical equations; thus, in a way, there exists an upper threshold of model performance. Now, research in many engineering sectors also focuses on the so-called data-driven modelling, which aims at extracting information from available data to combine it with the mathematical model. Focusing on the nuclear field, interest in this approach is also related to the Multi-Physics modelling of nuclear reactors. Due to the multiple physics involved and their mutual and complex interactions, developing accurate and stable models both from the physical and numerical point of view remains a challenging task despite the advancements in computational hardware and software, and combining the available mathematical model with data can further improve the performance and the accuracy of the former.   This work investigates this aspect by applying two Data-Driven Reduced Order Modelling (DDROM) techniques, the Generalised Empirical Interpolation Method and the Parametrised-Background Data-Weak formulation, to literature benchmark nuclear case studies. The main goal of this work is to assess the possibility of using data to perform model bias correction, that is, verifying the reliability of DDROM approaches in improving the model performance and accuracy through the information provided by the data. The obtained numerical results are promising, foreseeing further investigation of the DDROM approach to nuclear industrial cases.

{{</citation>}}


## stat.ML (2)



### (31/42) Efficient Frameworks for Generalized Low-Rank Matrix Bandit Problems (Yue Kang et al., 2024)

{{<citation>}}

Yue Kang, Cho-Jui Hsieh, Thomas C. M. Lee. (2024)  
**Efficient Frameworks for Generalized Low-Rank Matrix Bandit Problems**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2401.07298v1)  

---


**ABSTRACT**  
In the stochastic contextual low-rank matrix bandit problem, the expected reward of an action is given by the inner product between the action's feature matrix and some fixed, but initially unknown $d_1$ by $d_2$ matrix $\Theta^*$ with rank $r \ll \{d_1, d_2\}$, and an agent sequentially takes actions based on past experience to maximize the cumulative reward. In this paper, we study the generalized low-rank matrix bandit problem, which has been recently proposed in \cite{lu2021low} under the Generalized Linear Model (GLM) framework. To overcome the computational infeasibility and theoretical restrain of existing algorithms on this problem, we first propose the G-ESTT framework that modifies the idea from \cite{jun2019bilinear} by using Stein's method on the subspace estimation and then leverage the estimated subspaces via a regularization idea. Furthermore, we remarkably improve the efficiency of G-ESTT by using a novel exclusion idea on the estimated subspace instead, and propose the G-ESTS framework. We also show that G-ESTT can achieve the $\tilde{O}(\sqrt{(d_1+d_2)MrT})$ bound of regret while G-ESTS can achineve the $\tilde{O}(\sqrt{(d_1+d_2)^{3/2}Mr^{3/2}T})$ bound of regret under mild assumption up to logarithm terms, where $M$ is some problem dependent value. Under a reasonable assumption that $M = O((d_1+d_2)^2)$ in our problem setting, the regret of G-ESTT is consistent with the current best regret of $\tilde{O}((d_1+d_2)^{3/2} \sqrt{rT}/D_{rr})$~\citep{lu2021low} ($D_{rr}$ will be defined later). For completeness, we conduct experiments to illustrate that our proposed algorithms, especially G-ESTS, are also computationally tractable and consistently outperform other state-of-the-art (generalized) linear matrix bandit methods based on a suite of simulations.

{{</citation>}}


### (32/42) A Survey on Statistical Theory of Deep Learning: Approximation, Training Dynamics, and Generative Models (Namjoon Suh et al., 2024)

{{<citation>}}

Namjoon Suh, Guang Cheng. (2024)  
**A Survey on Statistical Theory of Deep Learning: Approximation, Training Dynamics, and Generative Models**  

---
Primary Category: stat.ML  
Categories: cs-LG, math-ST, stat-ML, stat-TH, stat.ML  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.07187v1)  

---


**ABSTRACT**  
In this article, we review the literature on statistical theories of neural networks from three perspectives. In the first part, results on excess risks for neural networks are reviewed in the nonparametric framework of regression or classification. These results rely on explicit constructions of neural networks, leading to fast convergence rates of excess risks, in that tools from the approximation theory are adopted. Through these constructions, the width and depth of the networks can be expressed in terms of sample size, data dimension, and function smoothness. Nonetheless, their underlying analysis only applies to the global minimizer in the highly non-convex landscape of deep neural networks. This motivates us to review the training dynamics of neural networks in the second part. Specifically, we review papers that attempt to answer ``how the neural network trained via gradient-based methods finds the solution that can generalize well on unseen data.'' In particular, two well-known paradigms are reviewed: the Neural Tangent Kernel (NTK) paradigm, and Mean-Field (MF) paradigm. In the last part, we review the most recent theoretical advancements in generative models including Generative Adversarial Networks (GANs), diffusion models, and in-context learning (ICL) in the Large Language Models (LLMs). The former two models are known to be the main pillars of the modern generative AI era, while ICL is a strong capability of LLMs in learning from a few examples in the context. Finally, we conclude the paper by suggesting several promising directions for deep learning theory.

{{</citation>}}


## cs.LG (5)



### (33/42) BET: Explaining Deep Reinforcement Learning through The Error-Prone Decisions (Xiao Liu et al., 2024)

{{<citation>}}

Xiao Liu, Jie Zhao, Wubing Chen, Mao Tan, Yongxing Su. (2024)  
**BET: Explaining Deep Reinforcement Learning through The Error-Prone Decisions**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.07263v1)  

---


**ABSTRACT**  
Despite the impressive capabilities of Deep Reinforcement Learning (DRL) agents in many challenging scenarios, their black-box decision-making process significantly limits their deployment in safety-sensitive domains. Several previous self-interpretable works focus on revealing the critical states of the agent's decision. However, they cannot pinpoint the error-prone states. To address this issue, we propose a novel self-interpretable structure, named Backbone Extract Tree (BET), to better explain the agent's behavior by identify the error-prone states. At a high level, BET hypothesizes that states in which the agent consistently executes uniform decisions exhibit a reduced propensity for errors. To effectively model this phenomenon, BET expresses these states within neighborhoods, each defined by a curated set of representative states. Therefore, states positioned at a greater distance from these representative benchmarks are more prone to error. We evaluate BET in various popular RL environments and show its superiority over existing self-interpretable models in terms of explanation fidelity. Furthermore, we demonstrate a use case for providing explanations for the agents in StarCraft II, a sophisticated multi-agent cooperative game. To the best of our knowledge, we are the first to explain such a complex scenarios using a fully transparent structure.

{{</citation>}}


### (34/42) Imputation with Inter-Series Information from Prototypes for Irregular Sampled Time Series (Zhihao Yu et al., 2024)

{{<citation>}}

Zhihao Yu, Xu Chu, Liantao Ma, Yasha Wang, Wenwu Zhu. (2024)  
**Imputation with Inter-Series Information from Prototypes for Irregular Sampled Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.07249v1)  

---


**ABSTRACT**  
Irregularly sampled time series are ubiquitous, presenting significant challenges for analysis due to missing values. Despite existing methods address imputation, they predominantly focus on leveraging intra-series information, neglecting the potential benefits that inter-series information could provide, such as reducing uncertainty and memorization effect. To bridge this gap, we propose PRIME, a Prototype Recurrent Imputation ModEl, which integrates both intra-series and inter-series information for imputing missing values in irregularly sampled time series. Our framework comprises a prototype memory module for learning inter-series information, a bidirectional gated recurrent unit utilizing prototype information for imputation, and an attentive prototypical refinement module for adjusting imputations. We conducted extensive experiments on three datasets, and the results underscore PRIME's superiority over the state-of-the-art models by up to 26% relative improvement on mean square error.

{{</citation>}}


### (35/42) The Effects of Data Imbalance Under a Federated Learning Approach for Credit Risk Forecasting (Shuyao Zhang et al., 2024)

{{<citation>}}

Shuyao Zhang, Jordan Tay, Pedro Baiz. (2024)  
**The Effects of Data Imbalance Under a Federated Learning Approach for Credit Risk Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.07234v1)  

---


**ABSTRACT**  
Credit risk forecasting plays a crucial role for commercial banks and other financial institutions in granting loans to customers and minimise the potential loss. However, traditional machine learning methods require the sharing of sensitive client information with an external server to build a global model, potentially posing a risk of security threats and privacy leakage. A newly developed privacy-preserving distributed machine learning technique known as Federated Learning (FL) allows the training of a global model without the necessity of accessing private local data directly. This investigation examined the feasibility of federated learning in credit risk assessment and showed the effects of data imbalance on model performance. Two neural network architectures, Multilayer Perceptron (MLP) and Long Short-Term Memory (LSTM), and one tree ensemble architecture, Extreme Gradient Boosting (XGBoost), were explored across three different datasets under various scenarios involving different numbers of clients and data distribution configurations. We demonstrate that federated models consistently outperform local models on non-dominant clients with smaller datasets. This trend is especially pronounced in highly imbalanced data scenarios, yielding a remarkable average improvement of 17.92% in model performance. However, for dominant clients (clients with more data), federated models may not exhibit superior performance, suggesting the need for special incentives for this type of clients to encourage their participation.

{{</citation>}}


### (36/42) Use of Prior Knowledge to Discover Causal Additive Models with Unobserved Variables and its Application to Time Series Data (Takashi Nicholas Maeda et al., 2024)

{{<citation>}}

Takashi Nicholas Maeda, Shohei Shimizu. (2024)  
**Use of Prior Knowledge to Discover Causal Additive Models with Unobserved Variables and its Application to Time Series Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ME, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.07231v3)  

---


**ABSTRACT**  
This paper proposes two methods for causal additive models with unobserved variables (CAM-UV). CAM-UV assumes that the causal functions take the form of generalized additive models and that latent confounders are present. First, we propose a method that leverages prior knowledge for efficient causal discovery. Then, we propose an extension of this method for inferring causality in time series data. The original CAM-UV algorithm differs from other existing causal function models in that it does not seek the causal order between observed variables, but rather aims to identify the causes for each observed variable. Therefore, the first proposed method in this paper utilizes prior knowledge, such as understanding that certain variables cannot be causes of specific others. Moreover, by incorporating the prior knowledge that causes precedes their effects in time, we extend the first algorithm to the second method for causal discovery in time series data. We validate the first proposed method by using simulated data to demonstrate that the accuracy of causal discovery increases as more prior knowledge is accumulated. Additionally, we test the second proposed method by comparing it with existing time series causal discovery methods, using both simulated data and real-world data.

{{</citation>}}


### (37/42) Reinforcement Learning from LLM Feedback to Counteract Goal Misgeneralization (Houda Nait El Barj et al., 2024)

{{<citation>}}

Houda Nait El Barj, Theophile Sautory. (2024)  
**Reinforcement Learning from LLM Feedback to Counteract Goal Misgeneralization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.07181v1)  

---


**ABSTRACT**  
We introduce a method to address goal misgeneralization in reinforcement learning (RL), leveraging Large Language Model (LLM) feedback during training. Goal misgeneralization, a type of robustness failure in RL occurs when an agent retains its capabilities out-of-distribution yet pursues a proxy rather than the intended one. Our approach utilizes LLMs to analyze an RL agent's policies during training and identify potential failure scenarios. The RL agent is then deployed in these scenarios, and a reward model is learnt through the LLM preferences and feedback. This LLM-informed reward model is used to further train the RL agent on the original dataset. We apply our method to a maze navigation task, and show marked improvements in goal generalization, especially in cases where true and proxy goals are somewhat distinguishable and behavioral biases are pronounced. This study demonstrates how the LLM, despite its lack of task proficiency, can efficiently supervise RL agents, providing scalable oversight and valuable insights for enhancing goal-directed learning in RL through the use of LLMs.

{{</citation>}}


## cs.IR (3)



### (38/42) Lightweight Modality Adaptation to Sequential Recommendation via Correlation Supervision (Hengchang Hu et al., 2024)

{{<citation>}}

Hengchang Hu, Qijiong Liu, Chuang Li, Min-Yen Kan. (2024)  
**Lightweight Modality Adaptation to Sequential Recommendation via Correlation Supervision**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07257v1)  

---


**ABSTRACT**  
In Sequential Recommenders (SR), encoding and utilizing modalities in an end-to-end manner is costly in terms of modality encoder sizes. Two-stage approaches can mitigate such concerns, but they suffer from poor performance due to modality forgetting, where the sequential objective overshadows modality representation. We propose a lightweight knowledge distillation solution that preserves both merits: retaining modality information and maintaining high efficiency. Specifically, we introduce a novel method that enhances the learning of embeddings in SR through the supervision of modality correlations. The supervision signals are distilled from the original modality representations, including both (1) holistic correlations, which quantify their overall associations, and (2) dissected correlation types, which refine their relationship facets (honing in on specific aspects like color or shape consistency). To further address the issue of modality forgetting, we propose an asynchronous learning step, allowing the original information to be retained longer for training the representation learning module. Our approach is compatible with various backbone architectures and outperforms the top baselines by 6.8% on average. We empirically demonstrate that preserving original feature associations from modality encoders significantly boosts task-specific recommendation adaptation. Additionally, we find that larger modality encoders (e.g., Large Language Models) contain richer feature sets which necessitate more fine-grained modeling to reach their full performance potential.

{{</citation>}}


### (39/42) Walert: Putting Conversational Search Knowledge into Action by Building and Evaluating a Large Language Model-Powered Chatbot (Sachin Pathiyan Cherumanal et al., 2024)

{{<citation>}}

Sachin Pathiyan Cherumanal, Lin Tian, Futoon M. Abushaqra, Angel Felipe Magnossao de Paula, Kaixin Ji, Danula Hettiachchi, Johanne R. Trippas, Halil Ali, Falk Scholer, Damiano Spina. (2024)  
**Walert: Putting Conversational Search Knowledge into Action by Building and Evaluating a Large Language Model-Powered Chatbot**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.07216v1)  

---


**ABSTRACT**  
Creating and deploying customized applications is crucial for operational success and enriching user experiences in the rapidly evolving modern business world. A prominent facet of modern user experiences is the integration of chatbots or voice assistants. The rapid evolution of Large Language Models (LLMs) has provided a powerful tool to build conversational applications. We present Walert, a customized LLM-based conversational agent able to answer frequently asked questions about computer science degrees and programs at RMIT University. Our demo aims to showcase how conversational information-seeking researchers can effectively communicate the benefits of using best practices to stakeholders interested in developing and deploying LLM-based chatbots. These practices are well-known in our community but often overlooked by practitioners who may not have access to this knowledge. The methodology and resources used in this demo serve as a bridge to facilitate knowledge transfer from experts, address industry professionals' practical needs, and foster a collaborative environment. The data and code of the demo are available at https://github.com/rmit-ir/walert.

{{</citation>}}


### (40/42) HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval (Zexuan Qiu et al., 2024)

{{<citation>}}

Zexuan Qiu, Jiahong Liu, Yankai Chen, Irwin King. (2024)  
**HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2401.07212v1)  

---


**ABSTRACT**  
Existing unsupervised deep product quantization methods primarily aim for the increased similarity between different views of the identical image, whereas the delicate multi-level semantic similarities preserved between images are overlooked. Moreover, these methods predominantly focus on the Euclidean space for computational convenience, compromising their ability to map the multi-level semantic relationships between images effectively. To mitigate these shortcomings, we propose a novel unsupervised product quantization method dubbed \textbf{Hi}erarchical \textbf{H}yperbolic \textbf{P}roduct \textbf{Q}uantization (HiHPQ), which learns quantized representations by incorporating hierarchical semantic similarity within hyperbolic geometry. Specifically, we propose a hyperbolic product quantizer, where the hyperbolic codebook attention mechanism and the quantized contrastive learning on the hyperbolic product manifold are introduced to expedite quantization. Furthermore, we propose a hierarchical semantics learning module, designed to enhance the distinction between similar and non-matching images for a query by utilizing the extracted hierarchical semantics as an additional training supervision. Experiments on benchmarks show that our proposed method outperforms state-of-the-art baselines.

{{</citation>}}


## cs.AR (1)



### (41/42) Hierarchical Source-to-Post-Route QoR Prediction in High-Level Synthesis with GNNs (Mingzhe Gao et al., 2024)

{{<citation>}}

Mingzhe Gao, Jieru Zhao, Zhe Lin, Minyi Guo. (2024)  
**Hierarchical Source-to-Post-Route QoR Prediction in High-Level Synthesis with GNNs**  

---
Primary Category: cs.AR  
Categories: cs-AI, cs-AR, cs-LG, cs.AR  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.08696v1)  

---


**ABSTRACT**  
High-level synthesis (HLS) notably speeds up the hardware design process by avoiding RTL programming. However, the turnaround time of HLS increases significantly when post-route quality of results (QoR) are considered during optimization. To tackle this issue, we propose a hierarchical post-route QoR prediction approach for FPGA HLS, which features: (1) a modeling flow that directly estimates latency and post-route resource usage from C/C++ programs; (2) a graph construction method that effectively represents the control and data flow graph of source code and effects of HLS pragmas; and (3) a hierarchical GNN training and prediction method capable of capturing the impact of loop hierarchies. Experimental results show that our method presents a prediction error of less than 10% for different types of QoR metrics, which gains tremendous improvement compared with the state-of-the-art GNN methods. By adopting our proposed methodology, the runtime for design space exploration in HLS is shortened to tens of minutes and the achieved ADRS is reduced to 6.91% on average.

{{</citation>}}


## cs.DC (1)



### (42/42) Resource Allocation of Industry 4.0 Micro-Service Applications across Serverless Fog Federation (Razin Farhan Hussain et al., 2024)

{{<citation>}}

Razin Farhan Hussain, Mohsen Amini Salehi. (2024)  
**Resource Allocation of Industry 4.0 Micro-Service Applications across Serverless Fog Federation**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.07194v1)  

---


**ABSTRACT**  
The Industry 4.0 revolution has been made possible via AI-based applications (e.g., for automation and maintenance) deployed on the serverless edge (aka fog) computing platforms at the industrial sites -- where the data is generated. Nevertheless, fulfilling the fault-intolerant and real-time constraints of Industry 4.0 applications on resource-limited fog systems in remote industrial sites (e.g., offshore oil fields) that are uncertain, disaster-prone, and have no cloud access is challenging. It is this challenge that our research aims at addressing. We consider the inelastic nature of the fog systems, software architecture of the industrial applications (micro-service-based versus monolithic), and scarcity of human experts in remote sites. To enable cloud-like elasticity, our approach is to dynamically and seamlessly (i.e., without human intervention) federate nearby fog systems. Then, we develop serverless resource allocation solutions that are cognizant of the applications' software architecture, their latency requirements, and distributed nature of the underlying infrastructure. We propose methods to seamlessly and optimally partition micro-service-based application across the federated fog. Our experimental evaluation express that not only the elasticity is overcome in a serverless manner, but also our developed application partitioning method can serve around 20% more tasks on-time than the existing methods in the literature.

{{</citation>}}
