---
draft: false
title: "arXiv @ 2023.11.21"
date: 2023-11-21
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.21"
    identifier: arxiv_20231121
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (11)](#cscl-11)
- [cs.CV (12)](#cscv-12)
- [cs.SE (1)](#csse-1)
- [cs.IT (1)](#csit-1)
- [cs.CR (3)](#cscr-3)
- [quant-ph (2)](#quant-ph-2)
- [cs.CE (1)](#csce-1)
- [cs.LG (13)](#cslg-13)
- [cs.CY (3)](#cscy-3)
- [cs.DL (1)](#csdl-1)
- [stat.ML (1)](#statml-1)
- [cs.AI (5)](#csai-5)
- [cs.RO (1)](#csro-1)
- [eess.SY (3)](#eesssy-3)
- [cs.SD (1)](#cssd-1)
- [cs.IR (1)](#csir-1)
- [eess.IV (1)](#eessiv-1)
- [eess.SP (1)](#eesssp-1)

## cs.CL (11)



### (1/62) LLM aided semi-supervision for Extractive Dialog Summarization (Nishant Mishra et al., 2023)

{{<citation>}}

Nishant Mishra, Gaurav Sahu, Iacer Calixto, Ameen Abu-Hanna, Issam H. Laradji. (2023)  
**LLM aided semi-supervision for Extractive Dialog Summarization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Summarization  
[Paper Link](http://arxiv.org/abs/2311.11462v1)  

---


**ABSTRACT**  
Generating high-quality summaries for chat dialogs often requires large labeled datasets. We propose a method to efficiently use unlabeled data for extractive summarization of customer-agent dialogs. In our method, we frame summarization as a question-answering problem and use state-of-the-art large language models (LLMs) to generate pseudo-labels for a dialog. We then use these pseudo-labels to fine-tune a chat summarization model, effectively transferring knowledge from the large LLM into a smaller specialized model. We demonstrate our method on the \tweetsumm dataset, and show that using 10\% of the original labelled data set we can achieve 65.9/57.0/61.0 ROUGE-1/-2/-L, whereas the current state-of-the-art trained on the entire training data set obtains 65.16/55.81/64.37 ROUGE-1/-2/-L. In other words, in the worst case (i.e., ROUGE-L) we still effectively retain 94.7% of the performance while using only 10% of the data.

{{</citation>}}


### (2/62) Spot the Bot: Distinguishing Human-Written and Bot-Generated Texts Using Clustering and Information Theory Techniques (Vasilii Gromov et al., 2023)

{{<citation>}}

Vasilii Gromov, Quynh Nhu Dang. (2023)  
**Spot the Bot: Distinguishing Human-Written and Bot-Generated Texts Using Clustering and Information Theory Techniques**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.11441v1)  

---


**ABSTRACT**  
With the development of generative models like GPT-3, it is increasingly more challenging to differentiate generated texts from human-written ones. There is a large number of studies that have demonstrated good results in bot identification. However, the majority of such works depend on supervised learning methods that require labelled data and/or prior knowledge about the bot-model architecture. In this work, we propose a bot identification algorithm that is based on unsupervised learning techniques and does not depend on a large amount of labelled data. By combining findings in semantic analysis by clustering (crisp and fuzzy) and information techniques, we construct a robust model that detects a generated text for different types of bot. We find that the generated texts tend to be more chaotic while literary works are more complex. We also demonstrate that the clustering of human texts results in fuzzier clusters in comparison to the more compact and well-separated clusters of bot-generated texts.

{{</citation>}}


### (3/62) Unveiling Public Perceptions: Machine Learning-Based Sentiment Analysis of COVID-19 Vaccines in India (Milind Gupta et al., 2023)

{{<citation>}}

Milind Gupta, Abhishek Kaushik. (2023)  
**Unveiling Public Perceptions: Machine Learning-Based Sentiment Analysis of COVID-19 Vaccines in India**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2311.11435v1)  

---


**ABSTRACT**  
In March 2020, the World Health Organisation declared COVID-19 a global pandemic as it spread to nearly every country. By mid-2021, India had introduced three vaccines: Covishield, Covaxin, and Sputnik. To ensure successful vaccination in a densely populated country like India, understanding public sentiment was crucial. Social media, particularly Reddit with over 430 million users, played a vital role in disseminating information. This study employs data mining techniques to analyze Reddit data and gauge Indian sentiments towards COVID-19 vaccines. Using Python's Text Blob library, comments are annotated to assess general sentiments. Results show that most Reddit users in India expressed neutrality about vaccination, posing a challenge for the Indian government's efforts to vaccinate a significant portion of the population.

{{</citation>}}


### (4/62) ML-LMCL: Mutual Learning and Large-Margin Contrastive Learning for Improving ASR Robustness in Spoken Language Understanding (Xuxin Cheng et al., 2023)

{{<citation>}}

Xuxin Cheng, Bowen Cao, Qichen Ye, Zhihong Zhu, Hongxiang Li, Yuexian Zou. (2023)  
**ML-LMCL: Mutual Learning and Large-Margin Contrastive Learning for Improving ASR Robustness in Spoken Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning, Spoken Language Understanding  
[Paper Link](http://arxiv.org/abs/2311.11375v1)  

---


**ABSTRACT**  
Spoken language understanding (SLU) is a fundamental task in the task-oriented dialogue systems. However, the inevitable errors from automatic speech recognition (ASR) usually impair the understanding performance and lead to error propagation. Although there are some attempts to address this problem through contrastive learning, they (1) treat clean manual transcripts and ASR transcripts equally without discrimination in fine-tuning; (2) neglect the fact that the semantically similar pairs are still pushed away when applying contrastive learning; (3) suffer from the problem of Kullback-Leibler (KL) vanishing. In this paper, we propose Mutual Learning and Large-Margin Contrastive Learning (ML-LMCL), a novel framework for improving ASR robustness in SLU. Specifically, in fine-tuning, we apply mutual learning and train two SLU models on the manual transcripts and the ASR transcripts, respectively, aiming to iteratively share knowledge between these two models. We also introduce a distance polarization regularizer to avoid pushing away the intra-cluster pairs as much as possible. Moreover, we use a cyclical annealing schedule to mitigate KL vanishing issue. Experiments on three datasets show that ML-LMCL outperforms existing models and achieves new state-of-the-art performance.

{{</citation>}}


### (5/62) Portuguese FAQ for Financial Services (Paulo Finardi et al., 2023)

{{<citation>}}

Paulo Finardi, Wanderley M. Melo, Edgard D. Medeiros Neto, Alex F. Mansano, Pablo B. Costa, Vinicius F. Caridá. (2023)  
**Portuguese FAQ for Financial Services**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Financial, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.11331v1)  

---


**ABSTRACT**  
Scarcity of domain-specific data in the Portuguese financial domain has disfavored the development of Natural Language Processing (NLP) applications. To address this limitation, the present study advocates for the utilization of synthetic data generated through data augmentation techniques. The investigation focuses on the augmentation of a dataset sourced from the Central Bank of Brazil FAQ, employing techniques that vary in semantic similarity. Supervised and unsupervised tasks are conducted to evaluate the impact of augmented data on both low and high semantic similarity scenarios. Additionally, the resultant dataset will be publicly disseminated on the Hugging Face Datasets platform, thereby enhancing accessibility and fostering broader engagement within the NLP research community.

{{</citation>}}


### (6/62) CHAMP: Efficient Annotation and Consolidation of Cluster Hierarchies (Arie Cattan et al., 2023)

{{<citation>}}

Arie Cattan, Tom Hope, Doug Downey, Roy Bar-Haim, Lilach Eden, Yoav Kantor, Ido Dagan. (2023)  
**CHAMP: Efficient Annotation and Consolidation of Cluster Hierarchies**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.11301v1)  

---


**ABSTRACT**  
Various NLP tasks require a complex hierarchical structure over nodes, where each node is a cluster of items. Examples include generating entailment graphs, hierarchical cross-document coreference resolution, annotating event and subevent relations, etc. To enable efficient annotation of such hierarchical structures, we release CHAMP, an open source tool allowing to incrementally construct both clusters and hierarchy simultaneously over any type of texts. This incremental approach significantly reduces annotation time compared to the common pairwise annotation approach and also guarantees maintaining transitivity at the cluster and hierarchy levels. Furthermore, CHAMP includes a consolidation mode, where an adjudicator can easily compare multiple cluster hierarchy annotations and resolve disagreements.

{{</citation>}}


### (7/62) A Cross-Attention Augmented Model for Event-Triggered Context-Aware Story Generation (Chen Tang et al., 2023)

{{<citation>}}

Chen Tang, Tyler Loakman, Chenghua Lin. (2023)  
**A Cross-Attention Augmented Model for Event-Triggered Context-Aware Story Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.11271v1)  

---


**ABSTRACT**  
Despite recent advancements, existing story generation systems continue to encounter difficulties in effectively incorporating contextual and event features, which greatly influence the quality of generated narratives. To tackle these challenges, we introduce a novel neural generation model, EtriCA, that enhances the relevance and coherence of generated stories by employing a cross-attention mechanism to map context features onto event sequences through residual mapping. This feature capturing mechanism enables our model to exploit logical relationships between events more effectively during the story generation process. To further enhance our proposed model, we employ a post-training framework for knowledge enhancement (KeEtriCA) on a large-scale book corpus. This allows EtriCA to adapt to a wider range of data samples. This results in approximately 5\% improvement in automatic metrics and over 10\% improvement in human evaluation. We conduct extensive experiments, including comparisons with state-of-the-art (SOTA) baseline models, to evaluate the performance of our framework on story generation. The experimental results, encompassing both automated metrics and human assessments, demonstrate the superiority of our model over existing state-of-the-art baselines. These results underscore the effectiveness of our model in leveraging context and event features to improve the quality of generated narratives.

{{</citation>}}


### (8/62) Towards Real-World Writing Assistance: A Chinese Character Checking Benchmark with Faked and Misspelled Characters (Yinghui Li et al., 2023)

{{<citation>}}

Yinghui Li, Zishan Xu, Shaoshen Chen, Haojing Huang, Yangning Li, Yong Jiang, Zhongli Li, Qingyu Zhou, Hai-Tao Zheng, Ying Shen. (2023)  
**Towards Real-World Writing Assistance: A Chinese Character Checking Benchmark with Faked and Misspelled Characters**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs-MM, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.11268v1)  

---


**ABSTRACT**  
Writing assistance is an application closely related to human life and is also a fundamental Natural Language Processing (NLP) research field. Its aim is to improve the correctness and quality of input texts, with character checking being crucial in detecting and correcting wrong characters. From the perspective of the real world where handwriting occupies the vast majority, characters that humans get wrong include faked characters (i.e., untrue characters created due to writing errors) and misspelled characters (i.e., true characters used incorrectly due to spelling errors). However, existing datasets and related studies only focus on misspelled characters mainly caused by phonological or visual confusion, thereby ignoring faked characters which are more common and difficult. To break through this dilemma, we present Visual-C$^3$, a human-annotated Visual Chinese Character Checking dataset with faked and misspelled Chinese characters. To the best of our knowledge, Visual-C$^3$ is the first real-world visual and the largest human-crafted dataset for the Chinese character checking scenario. Additionally, we also propose and evaluate novel baseline methods on Visual-C$^3$. Extensive empirical results and analyses show that Visual-C$^3$ is high-quality yet challenging. The Visual-C$^3$ dataset and the baseline methods will be publicly available to facilitate further research in the community.

{{</citation>}}


### (9/62) Rethinking Large Language Models in Mental Health Applications (Shaoxiong Ji et al., 2023)

{{<citation>}}

Shaoxiong Ji, Tianlin Zhang, Kailai Yang, Sophia Ananiadou, Erik Cambria. (2023)  
**Rethinking Large Language Models in Mental Health Applications**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11267v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have become valuable assets in mental health, showing promise in both classification tasks and counseling applications. This paper offers a perspective on using LLMs in mental health applications. It discusses the instability of generative models for prediction and the potential for generating hallucinatory outputs, underscoring the need for ongoing audits and evaluations to maintain their reliability and dependability. The paper also distinguishes between the often interchangeable terms ``explainability'' and ``interpretability'', advocating for developing inherently interpretable methods instead of relying on potentially hallucinated self-explanations generated by LLMs. Despite the advancements in LLMs, human counselors' empathetic understanding, nuanced interpretation, and contextual awareness remain irreplaceable in the sensitive and complex realm of mental health counseling. The use of LLMs should be approached with a judicious and considerate mindset, viewing them as tools that complement human expertise rather than seeking to replace it.

{{</citation>}}


### (10/62) Causal ATE Mitigates Unintended Bias in Controlled Text Generation (Rahul Madhavan et al., 2023)

{{<citation>}}

Rahul Madhavan, Kahini Wadhawan. (2023)  
**Causal ATE Mitigates Unintended Bias in Controlled Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model, Text Generation  
[Paper Link](http://arxiv.org/abs/2311.11229v1)  

---


**ABSTRACT**  
We study attribute control in language models through the method of Causal Average Treatment Effect (Causal ATE). Existing methods for the attribute control task in Language Models (LMs) check for the co-occurrence of words in a sentence with the attribute of interest, and control for them. However, spurious correlation of the words with the attribute in the training dataset, can cause models to hallucinate the presence of the attribute when presented with the spurious correlate during inference. We show that the simple perturbation-based method of Causal ATE removes this unintended effect. Additionally, we offer a theoretical foundation for investigating Causal ATE in the classification task, and prove that it reduces the number of false positives -- thereby mitigating the issue of unintended bias. Specifically, we ground it in the problem of toxicity mitigation, where a significant challenge lies in the inadvertent bias that often emerges towards protected groups post detoxification. We show that this unintended bias can be solved by the use of the Causal ATE metric.

{{</citation>}}


### (11/62) SPLAIN: Augmenting CybersecurityWarnings with Reasons and Data (Vera A. Kazakova et al., 2023)

{{<citation>}}

Vera A. Kazakova, Jena D. Hwang, Bonnie J. Dorr, Yorick Wilks, J. Blake Gage, Alex Memory, Mark A. Clark. (2023)  
**SPLAIN: Augmenting CybersecurityWarnings with Reasons and Data**  

---
Primary Category: cs.CL  
Categories: I-2, cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11215v1)  

---


**ABSTRACT**  
Effective cyber threat recognition and prevention demand comprehensible forecasting systems, as prior approaches commonly offer limited and, ultimately, unconvincing information. We introduce Simplified Plaintext Language (SPLAIN), a natural language generator that converts warning data into user-friendly cyber threat explanations. SPLAIN is designed to generate clear, actionable outputs, incorporating hierarchically organized explanatory details about input data and system functionality. Given the inputs of individual sensor-induced forecasting signals and an overall warning from a fusion module, SPLAIN queries each signal for information on contributing sensors and data signals. This collected data is processed into a coherent English explanation, encompassing forecasting, sensing, and data elements for user review. SPLAIN's template-based approach ensures consistent warning structure and vocabulary. SPLAIN's hierarchical output structure allows each threat and its components to be expanded to reveal underlying explanations on demand. Our conclusions emphasize the need for designers to specify the "how" and "why" behind cyber warnings, advocate for simple structured templates in generating consistent explanations, and recognize that direct causal links in Machine Learning approaches may not always be identifiable, requiring some explanations to focus on general methodologies, such as model and training data.

{{</citation>}}


## cs.CV (12)



### (12/62) Appearance Codes using Joint Embedding Learning of Multiple Modalities (Alex Zhang et al., 2023)

{{<citation>}}

Alex Zhang, Evan Dogariu. (2023)  
**Appearance Codes using Joint Embedding Learning of Multiple Modalities**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.11427v1)  

---


**ABSTRACT**  
The use of appearance codes in recent work on generative modeling has enabled novel view renders with variable appearance and illumination, such as day-time and night-time renders of a scene. A major limitation of this technique is the need to re-train new appearance codes for every scene on inference, so in this work we address this problem proposing a framework that learns a joint embedding space for the appearance and structure of the scene by enforcing a contrastive loss constraint between different modalities. We apply our framework to a simple Variational Auto-Encoder model on the RADIATE dataset \cite{sheeny2021radiate} and qualitatively demonstrate that we can generate new renders of night-time photos using day-time appearance codes without additional optimization iterations. Additionally, we compare our model to a baseline VAE that uses the standard per-image appearance code technique and show that our approach achieves generations of similar quality without learning appearance codes for any unseen images on inference.

{{</citation>}}


### (13/62) DiffSCI: Zero-Shot Snapshot Compressive Imaging via Iterative Spectral Diffusion Model (Zhenghao Pan et al., 2023)

{{<citation>}}

Zhenghao Pan, Haijin Zeng, Jiezhang Cao, Kai Zhang, Yongyong Chen. (2023)  
**DiffSCI: Zero-Shot Snapshot Compressive Imaging via Iterative Spectral Diffusion Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.11417v1)  

---


**ABSTRACT**  
This paper endeavors to advance the precision of snapshot compressive imaging (SCI) reconstruction for multispectral image (MSI). To achieve this, we integrate the advantageous attributes of established SCI techniques and an image generative model, propose a novel structured zero-shot diffusion model, dubbed DiffSCI. DiffSCI leverages the structural insights from the deep prior and optimization-based methodologies, complemented by the generative capabilities offered by the contemporary denoising diffusion model. Specifically, firstly, we employ a pre-trained diffusion model, which has been trained on a substantial corpus of RGB images, as the generative denoiser within the Plug-and-Play framework for the first time. This integration allows for the successful completion of SCI reconstruction, especially in the case that current methods struggle to address effectively. Secondly, we systematically account for spectral band correlations and introduce a robust methodology to mitigate wavelength mismatch, thus enabling seamless adaptation of the RGB diffusion model to MSIs. Thirdly, an accelerated algorithm is implemented to expedite the resolution of the data subproblem. This augmentation not only accelerates the convergence rate but also elevates the quality of the reconstruction process. We present extensive testing to show that DiffSCI exhibits discernible performance enhancements over prevailing self-supervised and zero-shot approaches, surpassing even supervised transformer counterparts across both simulated and real datasets. Our code will be available.

{{</citation>}}


### (14/62) Inspecting Explainability of Transformer Models with Additional Statistical Information (Hoang C. Nguyen et al., 2023)

{{<citation>}}

Hoang C. Nguyen, Haeil Lee, Junmo Kim. (2023)  
**Inspecting Explainability of Transformer Models with Additional Statistical Information**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.11378v1)  

---


**ABSTRACT**  
Transformer becomes more popular in the vision domain in recent years so there is a need for finding an effective way to interpret the Transformer model by visualizing it. In recent work, Chefer et al. can visualize the Transformer on vision and multi-modal tasks effectively by combining attention layers to show the importance of each image patch. However, when applying to other variants of Transformer such as the Swin Transformer, this method can not focus on the predicted object. Our method, by considering the statistics of tokens in layer normalization layers, shows a great ability to interpret the explainability of Swin Transformer and ViT.

{{</citation>}}


### (15/62) SOccDPT: Semi-Supervised 3D Semantic Occupancy from Dense Prediction Transformers trained under memory constraints (Aditya Nalgunda Ganesh, 2023)

{{<citation>}}

Aditya Nalgunda Ganesh. (2023)  
**SOccDPT: Semi-Supervised 3D Semantic Occupancy from Dense Prediction Transformers trained under memory constraints**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semi-Supervised, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.11371v1)  

---


**ABSTRACT**  
We present SOccDPT, a memory-efficient approach for 3D semantic occupancy prediction from monocular image input using dense prediction transformers. To address the limitations of existing methods trained on structured traffic datasets, we train our model on unstructured datasets including the Indian Driving Dataset and Bengaluru Driving Dataset. Our semi-supervised training pipeline allows SOccDPT to learn from datasets with limited labels by reducing the requirement for manual labelling by substituting it with pseudo-ground truth labels to produce our Bengaluru Semantic Occupancy Dataset. This broader training enhances our model's ability to handle unstructured traffic scenarios effectively. To overcome memory limitations during training, we introduce patch-wise training where we select a subset of parameters to train each epoch, reducing memory usage during auto-grad graph construction. In the context of unstructured traffic and memory-constrained training and inference, SOccDPT outperforms existing disparity estimation approaches as shown by the RMSE score of 9.1473, achieves a semantic segmentation IoU score of 46.02% and operates at a competitive frequency of 69.47 Hz. We make our code and semantic occupancy dataset public.

{{</citation>}}


### (16/62) Optimizing rgb-d semantic segmentation through multi-modal interaction and pooling attention (Shuai Zhang et al., 2023)

{{<citation>}}

Shuai Zhang, Minghong Xie. (2023)  
**Optimizing rgb-d semantic segmentation through multi-modal interaction and pooling attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.11312v1)  

---


**ABSTRACT**  
Semantic segmentation of RGB-D images involves understanding the appearance and spatial relationships of objects within a scene, which requires careful consideration of various factors. However, in indoor environments, the simple input of RGB and depth images often results in a relatively limited acquisition of semantic and spatial information, leading to suboptimal segmentation outcomes. To address this, we propose the Multi-modal Interaction and Pooling Attention Network (MIPANet), a novel approach designed to harness the interactive synergy between RGB and depth modalities, optimizing the utilization of complementary information. Specifically, we incorporate a Multi-modal Interaction Fusion Module (MIM) into the deepest layers of the network. This module is engineered to facilitate the fusion of RGB and depth information, allowing for mutual enhancement and correction. Additionally, we introduce a Pooling Attention Module (PAM) at various stages of the encoder. This module serves to amplify the features extracted by the network and integrates the module's output into the decoder in a targeted manner, significantly improving semantic segmentation performance. Our experimental results demonstrate that MIPANet outperforms existing methods on two indoor scene datasets, NYUDv2 and SUN-RGBD, underscoring its effectiveness in enhancing RGB-D semantic segmentation.

{{</citation>}}


### (17/62) Pair-wise Layer Attention with Spatial Masking for Video Prediction (Ping Li et al., 2023)

{{<citation>}}

Ping Li, Chenhan Zhang, Zheng Yang, Xianghua Xu, Mingli Song. (2023)  
**Pair-wise Layer Attention with Spatial Masking for Video Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.11289v1)  

---


**ABSTRACT**  
Video prediction yields future frames by employing the historical frames and has exhibited its great potential in many applications, e.g., meteorological prediction, and autonomous driving. Previous works often decode the ultimate high-level semantic features to future frames without texture details, which deteriorates the prediction quality. Motivated by this, we develop a Pair-wise Layer Attention (PLA) module to enhance the layer-wise semantic dependency of the feature maps derived from the U-shape structure in Translator, by coupling low-level visual cues and high-level features. Hence, the texture details of predicted frames are enriched. Moreover, most existing methods capture the spatiotemporal dynamics by Translator, but fail to sufficiently utilize the spatial features of Encoder. This inspires us to design a Spatial Masking (SM) module to mask partial encoding features during pretraining, which adds the visibility of remaining feature pixels by Decoder. To this end, we present a Pair-wise Layer Attention with Spatial Masking (PLA-SM) framework for video prediction to capture the spatiotemporal dynamics, which reflect the motion trend. Extensive experiments and rigorous ablation studies on five benchmarks demonstrate the advantages of the proposed approach. The code is available at GitHub.

{{</citation>}}


### (18/62) Transcending Forgery Specificity with Latent Space Augmentation for Generalizable Deepfake Detection (Zhiyuan Yan et al., 2023)

{{<citation>}}

Zhiyuan Yan, Yuhao Luo, Siwei Lyu, Qingshan Liu, Baoyuan Wu. (2023)  
**Transcending Forgery Specificity with Latent Space Augmentation for Generalizable Deepfake Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.11278v1)  

---


**ABSTRACT**  
Deepfake detection faces a critical generalization hurdle, with performance deteriorating when there is a mismatch between the distributions of training and testing data. A broadly received explanation is the tendency of these detectors to be overfitted to forgery-specific artifacts, rather than learning features that are widely applicable across various forgeries. To address this issue, we propose a simple yet effective detector called LSDA (\underline{L}atent \underline{S}pace \underline{D}ata \underline{A}ugmentation), which is based on a heuristic idea: representations with a wider variety of forgeries should be able to learn a more generalizable decision boundary, thereby mitigating the overfitting of method-specific features (see Figure. 1). Following this idea, we propose to enlarge the forgery space by constructing and simulating variations within and across forgery features in the latent space. This approach encompasses the acquisition of enriched, domain-specific features and the facilitation of smoother transitions between different forgery types, effectively bridging domain gaps. Our approach culminates in refining a binary classifier that leverages the distilled knowledge from the enhanced features, striving for a generalizable deepfake detector. Comprehensive experiments show that our proposed method is surprisingly effective and transcends state-of-the-art detectors across several widely used benchmarks.

{{</citation>}}


### (19/62) Generalization and Hallucination of Large Vision-Language Models through a Camouflaged Lens (Lv Tang et al., 2023)

{{<citation>}}

Lv Tang, Peng-Tao Jiang, Zhihao Shen, Hao Zhang, Jinwei Chen, Bo Li. (2023)  
**Generalization and Hallucination of Large Vision-Language Models through a Camouflaged Lens**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11273v1)  

---


**ABSTRACT**  
Large Vision-Language Model (LVLM) has seen burgeoning development and increasing attention recently. In this paper, we propose a novel framework, camo-perceptive vision-language framework (CPVLF), to explore whether LVLM can generalize to the challenging camouflaged object detection (COD) scenario in a training-free manner. During the process of generalization, we find that due to hallucination issues within LVLM, it can erroneously perceive objects in camouflaged scenes, producing counterfactual concepts. Moreover, as LVLM is not specifically trained for the precise localization of camouflaged objects, it exhibits a degree of uncertainty in accurately pinpointing these objects. Therefore, we propose chain of visual perception, which enhances LVLM's perception of camouflaged scenes from both linguistic and visual perspectives, reducing the hallucination issue and improving its capability in accurately locating camouflaged objects. We validate the effectiveness of CPVLF on three widely used COD datasets, and the experiments show the potential of LVLM in the COD task.

{{</citation>}}


### (20/62) Adversarial Prompt Tuning for Vision-Language Models (Jiaming Zhang et al., 2023)

{{<citation>}}

Jiaming Zhang, Xingjun Ma, Xin Wang, Lingyu Qiu, Jiaqi Wang, Yu-Gang Jiang, Jitao Sang. (2023)  
**Adversarial Prompt Tuning for Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11261v1)  

---


**ABSTRACT**  
With the rapid advancement of multimodal learning, pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capacities in bridging the gap between visual and language modalities. However, these models remain vulnerable to adversarial attacks, particularly in the image modality, presenting considerable security risks. This paper introduces Adversarial Prompt Tuning (AdvPT), a novel technique to enhance the adversarial robustness of image encoders in VLMs. AdvPT innovatively leverages learnable text prompts and aligns them with adversarial image embeddings, to address the vulnerabilities inherent in VLMs without the need for extensive parameter training or modification of the model architecture. We demonstrate that AdvPT improves resistance against white-box and black-box adversarial attacks and exhibits a synergistic effect when combined with existing image-processing-based defense techniques, further boosting defensive capabilities. Comprehensive experimental analyses provide insights into adversarial prompt tuning, a novel paradigm devoted to improving resistance to adversarial images through textual input modifications, paving the way for future robust multimodal learning research. These findings open up new possibilities for enhancing the security of VLMs. Our code will be available upon publication of the paper.

{{</citation>}}


### (21/62) GaussianDiffusion: 3D Gaussian Splatting for Denoising Diffusion Probabilistic Models with Structured Noise (Xinhai Li et al., 2023)

{{<citation>}}

Xinhai Li, Huaibin Wang, Kuo-Kun Tseng. (2023)  
**GaussianDiffusion: 3D Gaussian Splatting for Denoising Diffusion Probabilistic Models with Structured Noise**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11221v1)  

---


**ABSTRACT**  
Text-to-3D, known for its efficient generation methods and expansive creative potential, has garnered significant attention in the AIGC domain. However, the amalgamation of Nerf and 2D diffusion models frequently yields oversaturated images, posing severe limitations on downstream industrial applications due to the constraints of pixelwise rendering method. Gaussian splatting has recently superseded the traditional pointwise sampling technique prevalent in NeRF-based methodologies, revolutionizing various aspects of 3D reconstruction. This paper introduces a novel text to 3D content generation framework based on Gaussian splatting, enabling fine control over image saturation through individual Gaussian sphere transparencies, thereby producing more realistic images. The challenge of achieving multi-view consistency in 3D generation significantly impedes modeling complexity and accuracy. Taking inspiration from SJC, we explore employing multi-view noise distributions to perturb images generated by 3D Gaussian splatting, aiming to rectify inconsistencies in multi-view geometry. We ingeniously devise an efficient method to generate noise that produces Gaussian noise from diverse viewpoints, all originating from a shared noise source. Furthermore, vanilla 3D Gaussian-based generation tends to trap models in local minima, causing artifacts like floaters, burrs, or proliferative elements. To mitigate these issues, we propose the variational Gaussian splatting technique to enhance the quality and stability of 3D appearance. To our knowledge, our approach represents the first comprehensive utilization of Gaussian splatting across the entire spectrum of 3D content generation processes.

{{</citation>}}


### (22/62) Self-Supervised Versus Supervised Training for Segmentation of Organoid Images (Asmaa Haja et al., 2023)

{{<citation>}}

Asmaa Haja, Eric Brouwer, Lambert Schomaker. (2023)  
**Self-Supervised Versus Supervised Training for Segmentation of Organoid Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.11198v1)  

---


**ABSTRACT**  
The process of annotating relevant data in the field of digital microscopy can be both time-consuming and especially expensive due to the required technical skills and human-expert knowledge. Consequently, large amounts of microscopic image data sets remain unlabeled, preventing their effective exploitation using deep-learning algorithms. In recent years it has been shown that a lot of relevant information can be drawn from unlabeled data. Self-supervised learning (SSL) is a promising solution based on learning intrinsic features under a pretext task that is similar to the main task without requiring labels. The trained result is transferred to the main task - image segmentation in our case. A ResNet50 U-Net was first trained to restore images of liver progenitor organoids from augmented images using the Structural Similarity Index Metric (SSIM), alone, and using SSIM combined with L1 loss. Both the encoder and decoder were trained in tandem. The weights were transferred to another U-Net model designed for segmentation with frozen encoder weights, using Binary Cross Entropy, Dice, and Intersection over Union (IoU) losses. For comparison, we used the same U-Net architecture to train two supervised models, one utilizing the ResNet50 encoder as well as a simple CNN. Results showed that self-supervised learning models using a 25\% pixel drop or image blurring augmentation performed better than the other augmentation techniques using the IoU loss. When trained on only 114 images for the main task, the self-supervised learning approach outperforms the supervised method achieving an F1-score of 0.85, with higher stability, in contrast to an F1=0.78 scored by the supervised method. Furthermore, when trained with larger data sets (1,000 images), self-supervised learning is still able to perform better, achieving an F1-score of 0.92, contrasting to a score of 0.85 for the supervised method.

{{</citation>}}


### (23/62) Attention-Based Real-Time Defenses for Physical Adversarial Attacks in Vision Applications (Giulio Rossolini et al., 2023)

{{<citation>}}

Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo. (2023)  
**Attention-Based Real-Time Defenses for Physical Adversarial Attacks in Vision Applications**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Adversarial Attack, Attention  
[Paper Link](http://arxiv.org/abs/2311.11191v1)  

---


**ABSTRACT**  
Deep neural networks exhibit excellent performance in computer vision tasks, but their vulnerability to real-world adversarial attacks, achieved through physical objects that can corrupt their predictions, raises serious security concerns for their application in safety-critical domains. Existing defense methods focus on single-frame analysis and are characterized by high computational costs that limit their applicability in multi-frame scenarios, where real-time decisions are crucial.   To address this problem, this paper proposes an efficient attention-based defense mechanism that exploits adversarial channel-attention to quickly identify and track malicious objects in shallow network layers and mask their adversarial effects in a multi-frame setting. This work advances the state of the art by enhancing existing over-activation techniques for real-world adversarial attacks to make them usable in real-time applications. It also introduces an efficient multi-frame defense framework, validating its efficacy through extensive experiments aimed at evaluating both defense performance and computational cost.

{{</citation>}}


## cs.SE (1)



### (24/62) Tensor-Aware Energy Accounting (Timur Babakol et al., 2023)

{{<citation>}}

Timur Babakol, Yu David Liu. (2023)  
**Tensor-Aware Energy Accounting**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: AI, BERT  
[Paper Link](http://arxiv.org/abs/2311.11424v1)  

---


**ABSTRACT**  
With the rapid growth of Artificial Intelligence (AI) applications supported by deep learning (DL), the energy efficiency of these applications has an increasingly large impact on sustainability. We introduce Smaragdine, a new energy accounting system for tensor-based DL programs implemented with TensorFlow. At the heart of Smaragdine is a novel white-box methodology of energy accounting: Smaragdine is aware of the internal structure of the DL program, which we call tensor-aware energy accounting. With Smaragdine, the energy consumption of a DL program can be broken down into units aligned with its logical hierarchical decomposition structure. We apply Smaragdine for understanding the energy behavior of BERT, one of the most widely used language models. Layer-by-layer and tensor-by-tensor, Smaragdine is capable of identifying the highest energy/power-consuming components of BERT. Furthermore, we conduct two case studies on how Smaragdine supports downstream toolchain building, one on the comparative energy impact of hyperparameter tuning of BERT, the other on the energy behavior evolution when BERT evolves to its next generation, ALBERT.

{{</citation>}}


## cs.IT (1)



### (25/62) Offline Reinforcement Learning for Wireless Network Optimization with Mixture Datasets (Kun Yang et al., 2023)

{{<citation>}}

Kun Yang, Cong Shen, Jing Yang, Shu-ping Yeh, Jerry Sydir. (2023)  
**Offline Reinforcement Learning for Wireless Network Optimization with Mixture Datasets**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-LG, cs-NI, cs.IT, eess-SP, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.11423v1)  

---


**ABSTRACT**  
The recent development of reinforcement learning (RL) has boosted the adoption of online RL for wireless radio resource management (RRM). However, online RL algorithms require direct interactions with the environment, which may be undesirable given the potential performance loss due to the unavoidable exploration in RL. In this work, we first investigate the use of \emph{offline} RL algorithms in solving the RRM problem. We evaluate several state-of-the-art offline RL algorithms, including behavior constrained Q-learning (BCQ), conservative Q-learning (CQL), and implicit Q-learning (IQL), for a specific RRM problem that aims at maximizing a linear combination {of sum and} 5-percentile rates via user scheduling. We observe that the performance of offline RL for the RRM problem depends critically on the behavior policy used for data collection, and further propose a novel offline RL solution that leverages heterogeneous datasets collected by different behavior policies. We show that with a proper mixture of the datasets, offline RL can produce a near-optimal RL policy even when all involved behavior policies are highly suboptimal.

{{</citation>}}


## cs.CR (3)



### (26/62) A Security Risk Taxonomy for Large Language Models (Erik Derner et al., 2023)

{{<citation>}}

Erik Derner, Kristina Batistič, Jan Zahálka, Robert Babuška. (2023)  
**A Security Risk Taxonomy for Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs-HC, cs-LG, cs.CR  
Keywords: Language Model, Security  
[Paper Link](http://arxiv.org/abs/2311.11415v1)  

---


**ABSTRACT**  
As large language models (LLMs) permeate more and more applications, an assessment of their associated security risks becomes increasingly necessary. The potential for exploitation by malicious actors, ranging from disinformation to data breaches and reputation damage, is substantial. This paper addresses a gap in current research by focusing on the security risks posed by LLMs, which extends beyond the widely covered ethical and societal implications. Our work proposes a taxonomy of security risks along the user-model communication pipeline, explicitly focusing on prompt-based attacks on LLMs. We categorize the attacks by target and attack type within a prompt-based interaction scheme. The taxonomy is reinforced with specific attack examples to showcase the real-world impact of these risks. Through this taxonomy, we aim to inform the development of robust and secure LLM applications, enhancing their safety and trustworthiness.

{{</citation>}}


### (27/62) DNA Encoded Elliptic Curve Cryptography System for IoT Security (Prokash Barmana et al., 2023)

{{<citation>}}

Prokash Barmana, Banani Saha. (2023)  
**DNA Encoded Elliptic Curve Cryptography System for IoT Security**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.11393v1)  

---


**ABSTRACT**  
In the field of Computer Science and Information Technology Internet of Things (IoT) is one of the emerging technologies. In IoT environment several devices are interconnected and transmit data among them. There may be some security vulnerability arise within the IoT environment. Till date, IoT has not been widely accepted due to its security flaws. Hence to keep the IoT environment most robust, we propose a stable security framework of IoT with Elliptic Curve Cryptography (ECC) using DNA Encoding. The ECC is most lightweight cryptography technique among other well known public key cryptography techniques. To increase encryption complexity, DNA encoding mechanism of DNA computing with ECC is preceded.

{{</citation>}}


### (28/62) Systematic Analysis of Security and Vulnerabilities in Miniapps (Yuyang Han et al., 2023)

{{<citation>}}

Yuyang Han, Xu Ji, Zhiqiang Wang, Jianyi Zhang. (2023)  
**Systematic Analysis of Security and Vulnerabilities in Miniapps**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.11382v1)  

---


**ABSTRACT**  
The past few years have witnessed a boom of miniapps, as lightweight applications, miniapps are of great importance in the mobile internet sector. Consequently, the security of miniapps can directly impact compromising the integrity of sensitive data, posing a potential threat to user privacy. However, after a thorough review of the various research efforts in miniapp security, we found that their actions in researching the safety of miniapp web interfaces are limited. This paper proposes a triad threat model focusing on users, servers and attackers to mitigate the security risk of miniapps. By following the principle of least privilege and the direction of permission consistency, we design a novel analysis framework for the security risk assessment of miniapps by this model. Then, we analyzed the correlation between the security risk assessment and the threat model associated with the miniapp. This analysis led to identifying potential scopes and categorisations with security risks. In the case study, we identify nine major categories of vulnerability issues, such as SQL injection, logical vulnerabilities and cross-site scripting. We also assessed a total of 50,628 security risk hazards and provided specific examples.

{{</citation>}}


## quant-ph (2)



### (29/62) Neural Quantum Embedding: Pushing the Limits of Quantum Supervised Learning (Tak Hur et al., 2023)

{{<citation>}}

Tak Hur, Israel F. Araujo, Daniel K. Park. (2023)  
**Neural Quantum Embedding: Pushing the Limits of Quantum Supervised Learning**  

---
Primary Category: quant-ph  
Categories: cs-ET, quant-ph, quant-ph  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.11412v1)  

---


**ABSTRACT**  
Quantum embedding is indispensable for applying quantum machine learning techniques to classical data, and has substantial impacts on performance outcomes. In this study, we present Neural Quantum Embedding (NQE), a method that efficiently optimizes quantum embedding by leveraging classical deep learning techniques. NQE enhances the lower bound of the empirical risk, leading to substantial improvements in classification performance. Moreover, NQE improves robustness against noise. To validate the effectiveness of NQE, we conduct experiments on IBM quantum devices for image data classification, resulting in a remarkable accuracy enhancement from 0.52 to 0.96. Numerical analysis of the local effective dimension highlights that NQE improves the trainability and generalization performance of quantum neural networks. Furthermore, NQE achieves improved generalization in the quantum kernel method, as evidenced by a reduction in the upper bound of the expected risk.

{{</citation>}}


### (30/62) Symbolic Execution for Quantum Error Correction Programs (Wang Fang et al., 2023)

{{<citation>}}

Wang Fang, Mingsheng Ying. (2023)  
**Symbolic Execution for Quantum Error Correction Programs**  

---
Primary Category: quant-ph  
Categories: cs-PL, quant-ph, quant-ph  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2311.11313v1)  

---


**ABSTRACT**  
We define a symbolic execution framework QSE for quantum programs by integrating symbolic variables into quantum states and the outcomes of quantum measurements. The soundness theorem of QSE is proved. We further introduce symbolic stabilizer states, which facilitate the efficient analysis of quantum error correction programs. Within the QSE framework, we can use symbolic expressions to characterize the possible adversarial errors in quantum error correction, providing a significant improvement over existing methods that rely on sampling with simulators. We implement QSE with the support of symbolic stabilizer states in a prototype tool named QuantumSE.jl. With experiments on representative quantum error correction codes, including quantum repetition codes, Kitaev's toric codes, and quantum Tanner codes, we demonstrate the efficiency of QuantumSE.jl for debugging quantum error correction programs with over 1000 qubits. In addition, as a by-product of QSE, QuantumSE.jl's sampling functionality for stabilizer circuits also outperforms the state-of-the-art stabilizer simulator, Google's Stim, in the experiments.

{{</citation>}}


## cs.CE (1)



### (31/62) Attention-based Multi-fidelity Machine Learning Model for Computational Fractional Flow Reserve Assessment (Haizhou Yang et al., 2023)

{{<citation>}}

Haizhou Yang, C. Alberto Figueroa, Krishna Garikipati. (2023)  
**Attention-based Multi-fidelity Machine Learning Model for Computational Fractional Flow Reserve Assessment**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.11397v1)  

---


**ABSTRACT**  
Coronary Artery Disease (CAD) is one of the most common forms of heart disease, which is caused by a buildup of atherosclerotic plaque (known as stenosis) in the coronary arteries, leading to insufficient supplement of blood, oxygen, and nutrients to the heart. Fractional Flow Reserve (FFR), measuring the pressure ratio between the aorta and distal coronary artery, is an invasive physiologic gold standard for assessing the severity of coronary artery stenosis. Despite its benefits, invasive FFR assessment is still underutilized due to its high cost, time-consuming, experimental variability, and increased risk to patients. In this study, an attention-based multi-fidelity machine learning model (AttMulFid) is proposed for computationally efficient and accurate FFR assessment with uncertainty measurement. Within AttMulFid, an autoencoder is utilized to intelligently select geometric features from coronary arteries, with additional attention on the key area. Results show that the geometric features are able to represent the entirety of the geometric information and intelligently allocate attention based on crucial properties of geometry. Furthermore, the AttMulFid is a feasible approach for non-invasive, rapid, and accurate FFR assessment (with 0.002s/simulation).

{{</citation>}}


## cs.LG (13)



### (32/62) Towards interpretable-by-design deep learning algorithms (Plamen Angelov et al., 2023)

{{<citation>}}

Plamen Angelov, Dmitry Kangin, Ziyang Zhang. (2023)  
**Towards interpretable-by-design deep learning algorithms**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.11396v1)  

---


**ABSTRACT**  
The proposed framework named IDEAL (Interpretable-by-design DEep learning ALgorithms) recasts the standard supervised classification problem into a function of similarity to a set of prototypes derived from the training data, while taking advantage of existing latent spaces of large neural networks forming so-called Foundation Models (FM). This addresses the issue of explainability (stage B) while retaining the benefits from the tremendous achievements offered by DL models (e.g., visual transformers, ViT) pre-trained on huge data sets such as IG-3.6B + ImageNet-1K or LVD-142M (stage A). We show that one can turn such DL models into conceptually simpler, explainable-through-prototypes ones.   The key findings can be summarized as follows: (1) the proposed models are interpretable through prototypes, mitigating the issue of confounded interpretations, (2) the proposed IDEAL framework circumvents the issue of catastrophic forgetting allowing efficient class-incremental learning, and (3) the proposed IDEAL approach demonstrates that ViT architectures narrow the gap between finetuned and non-finetuned models allowing for transfer learning in a fraction of time \textbf{without} finetuning of the feature space on a target dataset with iterative supervised methods.

{{</citation>}}


### (33/62) Multi-Task Reinforcement Learning with Mixture of Orthogonal Experts (Ahmed Hendawy et al., 2023)

{{<citation>}}

Ahmed Hendawy, Jan Peters, Carlo D'Eramo. (2023)  
**Multi-Task Reinforcement Learning with Mixture of Orthogonal Experts**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.11385v1)  

---


**ABSTRACT**  
Multi-Task Reinforcement Learning (MTRL) tackles the long-standing problem of endowing agents with skills that generalize across a variety of problems. To this end, sharing representations plays a fundamental role in capturing both unique and common characteristics of the tasks. Tasks may exhibit similarities in terms of skills, objects, or physical properties while leveraging their representations eases the achievement of a universal policy. Nevertheless, the pursuit of learning a shared set of diverse representations is still an open challenge. In this paper, we introduce a novel approach for representation learning in MTRL that encapsulates common structures among the tasks using orthogonal representations to promote diversity. Our method, named Mixture Of Orthogonal Experts (MOORE), leverages a Gram-Schmidt process to shape a shared subspace of representations generated by a mixture of experts. When task-specific information is provided, MOORE generates relevant representations from this shared subspace. We assess the effectiveness of our approach on two MTRL benchmarks, namely MiniGrid and MetaWorld, showing that MOORE surpasses related baselines and establishes a new state-of-the-art result on MetaWorld.

{{</citation>}}


### (34/62) Self-Supervised Pretraining for Heterogeneous Hypergraph Neural Networks (Abdalgader Abubaker et al., 2023)

{{<citation>}}

Abdalgader Abubaker, Takanori Maehara, Madhav Nimishakavi, Vassilis Plachouras. (2023)  
**Self-Supervised Pretraining for Heterogeneous Hypergraph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.11368v1)  

---


**ABSTRACT**  
Recently, pretraining methods for the Graph Neural Networks (GNNs) have been successful at learning effective representations from unlabeled graph data. However, most of these methods rely on pairwise relations in the graph and do not capture the underling higher-order relations between entities. Hypergraphs are versatile and expressive structures that can effectively model higher-order relationships among entities in the data. Despite the efforts to adapt GNNs to hypergraphs (HyperGNN), there are currently no fully self-supervised pretraining methods for HyperGNN on heterogeneous hypergraphs. In this paper, we present SPHH, a novel self-supervised pretraining framework for heterogeneous HyperGNNs. Our method is able to effectively capture higher-order relations among entities in the data in a self-supervised manner. SPHH is consist of two self-supervised pretraining tasks that aim to simultaneously learn both local and global representations of the entities in the hypergraph by using informative representations derived from the hypergraph structure. Overall, our work presents a significant advancement in the field of self-supervised pretraining of HyperGNNs, and has the potential to improve the performance of various graph-based downstream tasks such as node classification and link prediction tasks which are mapped to hypergraph configuration. Our experiments on two real-world benchmarks using four different HyperGNN models show that our proposed SPHH framework consistently outperforms state-of-the-art baselines in various downstream tasks. The results demonstrate that SPHH is able to improve the performance of various HyperGNN models in various downstream tasks, regardless of their architecture or complexity, which highlights the robustness of our framework.

{{</citation>}}


### (35/62) A Generative Model for Accelerated Inverse Modelling Using a Novel Embedding for Continuous Variables (Sébastien Bompas abd Stefan Sandfeld, 2023)

{{<citation>}}

Sébastien Bompas abd Stefan Sandfeld. (2023)  
**A Generative Model for Accelerated Inverse Modelling Using a Novel Embedding for Continuous Variables**  

---
Primary Category: cs.LG  
Categories: cond-mat-mtrl-sci, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.11343v1)  

---


**ABSTRACT**  
In materials science, the challenge of rapid prototyping materials with desired properties often involves extensive experimentation to find suitable microstructures. Additionally, finding microstructures for given properties is typically an ill-posed problem where multiple solutions may exist. Using generative machine learning models can be a viable solution which also reduces the computational cost. This comes with new challenges because, e.g., a continuous property variable as conditioning input to the model is required. We investigate the shortcomings of an existing method and compare this to a novel embedding strategy for generative models that is based on the binary representation of floating point numbers. This eliminates the need for normalization, preserves information, and creates a versatile embedding space for conditioning the generative model. This technique can be applied to condition a network on any number, to provide fine control over generated microstructure images, thereby contributing to accelerated materials design.

{{</citation>}}


### (36/62) Self-Distilled Representation Learning for Time Series (Felix Pieper et al., 2023)

{{<citation>}}

Felix Pieper, Konstantin Ditschuneit, Martin Genzel, Alexandra Lindt, Johannes Otterbach. (2023)  
**Self-Distilled Representation Learning for Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Computer Vision, Natural Language Processing, Representation Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2311.11335v1)  

---


**ABSTRACT**  
Self-supervised learning for time-series data holds potential similar to that recently unleashed in Natural Language Processing and Computer Vision. While most existing works in this area focus on contrastive learning, we propose a conceptually simple yet powerful non-contrastive approach, based on the data2vec self-distillation framework. The core of our method is a student-teacher scheme that predicts the latent representation of an input time series from masked views of the same time series. This strategy avoids strong modality-specific assumptions and biases typically introduced by the design of contrastive sample pairs. We demonstrate the competitiveness of our approach for classification and forecasting as downstream tasks, comparing with state-of-the-art self-supervised learning methods on the UCR and UEA archives as well as the ETT and Electricity datasets.

{{</citation>}}


### (37/62) From Categories to Classifier: Name-Only Continual Learning by Exploring the Web (Ameya Prabhu et al., 2023)

{{<citation>}}

Ameya Prabhu, Hasan Abed Al Kader Hammoud, Ser-Nam Lim, Bernard Ghanem, Philip H. S. Torr, Adel Bibi. (2023)  
**From Categories to Classifier: Name-Only Continual Learning by Exploring the Web**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11293v1)  

---


**ABSTRACT**  
Continual Learning (CL) often relies on the availability of extensive annotated datasets, an assumption that is unrealistically time-consuming and costly in practice. We explore a novel paradigm termed name-only continual learning where time and cost constraints prohibit manual annotation. In this scenario, learners adapt to new category shifts using only category names without the luxury of annotated training data. Our proposed solution leverages the expansive and ever-evolving internet to query and download uncurated webly-supervised data for image classification. We investigate the reliability of our web data and find them comparable, and in some cases superior, to manually annotated datasets. Additionally, we show that by harnessing the web, we can create support sets that surpass state-of-the-art name-only classification that create support sets using generative models or image retrieval from LAION-5B, achieving up to 25% boost in accuracy. When applied across varied continual learning contexts, our method consistently exhibits a small performance gap in comparison to models trained on manually annotated datasets. We present EvoTrends, a class-incremental dataset made from the web to capture real-world trends, created in just minutes. Overall, this paper underscores the potential of using uncurated webly-supervised data to mitigate the challenges associated with manual data labeling in continual learning.

{{</citation>}}


### (38/62) TimeSQL: Improving Multivariate Time Series Forecasting with Multi-Scale Patching and Smooth Quadratic Loss (Site Mo et al., 2023)

{{<citation>}}

Site Mo, Haoxin Wang, Bixiong Li, Songhai Fan, Yuankai Wu, Xianggen Liu. (2023)  
**TimeSQL: Improving Multivariate Time Series Forecasting with Multi-Scale Patching and Smooth Quadratic Loss**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.11285v1)  

---


**ABSTRACT**  
Time series is a special type of sequence data, a sequence of real-valued random variables collected at even intervals of time. The real-world multivariate time series comes with noises and contains complicated local and global temporal dynamics, making it difficult to forecast the future time series given the historical observations. This work proposes a simple and effective framework, coined as TimeSQL, which leverages multi-scale patching and smooth quadratic loss (SQL) to tackle the above challenges. The multi-scale patching transforms the time series into two-dimensional patches with different length scales, facilitating the perception of both locality and long-term correlations in time series. SQL is derived from the rational quadratic kernel and can dynamically adjust the gradients to avoid overfitting to the noises and outliers. Theoretical analysis demonstrates that, under mild conditions, the effect of the noises on the model with SQL is always smaller than that with MSE. Based on the two modules, TimeSQL achieves new state-of-the-art performance on the eight real-world benchmark datasets. Further ablation studies indicate that the key modules in TimeSQL could also enhance the results of other models for multivariate time series forecasting, standing as plug-and-play techniques.

{{</citation>}}


### (39/62) Open Set Dandelion Network for IoT Intrusion Detection (Jiashu Wu et al., 2023)

{{<citation>}}

Jiashu Wu, Hao Dai, Kenneth B. Kent, Jerome Yen, Chengzhong Xu, Yang Wang. (2023)  
**Open Set Dandelion Network for IoT Intrusion Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2311.11249v1)  

---


**ABSTRACT**  
As IoT devices become widely, it is crucial to protect them from malicious intrusions. However, the data scarcity of IoT limits the applicability of traditional intrusion detection methods, which are highly data-dependent. To address this, in this paper we propose the Open-Set Dandelion Network (OSDN) based on unsupervised heterogeneous domain adaptation in an open-set manner. The OSDN model performs intrusion knowledge transfer from the knowledge-rich source network intrusion domain to facilitate more accurate intrusion detection for the data-scarce target IoT intrusion domain. Under the open-set setting, it can also detect newly-emerged target domain intrusions that are not observed in the source domain. To achieve this, the OSDN model forms the source domain into a dandelion-like feature space in which each intrusion category is compactly grouped and different intrusion categories are separated, i.e., simultaneously emphasising inter-category separability and intra-category compactness. The dandelion-based target membership mechanism then forms the target dandelion. Then, the dandelion angular separation mechanism achieves better inter-category separability, and the dandelion embedding alignment mechanism further aligns both dandelions in a finer manner. To promote intra-category compactness, the discriminating sampled dandelion mechanism is used. Assisted by the intrusion classifier trained using both known and generated unknown intrusion knowledge, a semantic dandelion correction mechanism emphasises easily-confused categories and guides better inter-category separability. Holistically, these mechanisms form the OSDN model that effectively performs intrusion knowledge transfer to benefit IoT intrusion detection. Comprehensive experiments on several intrusion datasets verify the effectiveness of the OSDN model, outperforming three state-of-the-art baseline methods by 16.9%.

{{</citation>}}


### (40/62) Unraveling the `Anomaly' in Time Series Anomaly Detection: A Self-supervised Tri-domain Solution (Yuting Sun et al., 2023)

{{<citation>}}

Yuting Sun, Guansong Pang, Guanhua Ye, Tong Chen, Xia Hu, Hongzhi Yin. (2023)  
**Unraveling the `Anomaly' in Time Series Anomaly Detection: A Self-supervised Tri-domain Solution**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2311.11235v1)  

---


**ABSTRACT**  
The ongoing challenges in time series anomaly detection (TSAD), notably the scarcity of anomaly labels and the variability in anomaly lengths and shapes, have led to the need for a more efficient solution. As limited anomaly labels hinder traditional supervised models in TSAD, various SOTA deep learning techniques, such as self-supervised learning, have been introduced to tackle this issue. However, they encounter difficulties handling variations in anomaly lengths and shapes, limiting their adaptability to diverse anomalies. Additionally, many benchmark datasets suffer from the problem of having explicit anomalies that even random functions can detect. This problem is exacerbated by ill-posed evaluation metrics, known as point adjustment (PA), which can result in inflated model performance. In this context, we propose a novel self-supervised learning based Tri-domain Anomaly Detector (TriAD), which addresses these challenges by modeling features across three data domains - temporal, frequency, and residual domains - without relying on anomaly labels. Unlike traditional contrastive learning methods, TriAD employs both inter-domain and intra-domain contrastive loss to learn common attributes among normal data and differentiate them from anomalies. Additionally, our approach can detect anomalies of varying lengths by integrating with a discord discovery algorithm. It is worth noting that this study is the first to reevaluate the deep learning potential in TSAD, utilizing both rigorously designed datasets (i.e., UCR Archive) and evaluation metrics (i.e., PA%K and affiliation). Through experimental results on the UCR dataset, TriAD achieves an impressive three-fold increase in PA%K based F1 scores over SOTA deep learning models, and 50% increase of accuracy as compared to SOTA discord discovery algorithms.

{{</citation>}}


### (41/62) A Universal Framework for Accurate and Efficient Geometric Deep Learning of Molecular Systems (Shuo Zhang et al., 2023)

{{<citation>}}

Shuo Zhang, Yang Liu, Lei Xie. (2023)  
**A Universal Framework for Accurate and Efficient Geometric Deep Learning of Molecular Systems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-BM  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.11228v1)  

---


**ABSTRACT**  
Molecular sciences address a wide range of problems involving molecules of different types and sizes and their complexes. Recently, geometric deep learning, especially Graph Neural Networks, has shown promising performance in molecular science applications. However, most existing works often impose targeted inductive biases to a specific molecular system, and are inefficient when applied to macromolecules or large-scale tasks, thereby limiting their applications to many real-world problems. To address these challenges, we present PAMNet, a universal framework for accurately and efficiently learning the representations of three-dimensional (3D) molecules of varying sizes and types in any molecular system. Inspired by molecular mechanics, PAMNet induces a physics-informed bias to explicitly model local and non-local interactions and their combined effects. As a result, PAMNet can reduce expensive operations, making it time and memory efficient. In extensive benchmark studies, PAMNet outperforms state-of-the-art baselines regarding both accuracy and efficiency in three diverse learning tasks: small molecule properties, RNA 3D structures, and protein-ligand binding affinities. Our results highlight the potential for PAMNet in a broad range of molecular science applications.

{{</citation>}}


### (42/62) TextGuard: Provable Defense against Backdoor Attacks on Text Classification (Hengzhi Pei et al., 2023)

{{<citation>}}

Hengzhi Pei, Jinyuan Jia, Wenbo Guo, Bo Li, Dawn Song. (2023)  
**TextGuard: Provable Defense against Backdoor Attacks on Text Classification**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: AI, Text Classification  
[Paper Link](http://arxiv.org/abs/2311.11225v1)  

---


**ABSTRACT**  
Backdoor attacks have become a major security threat for deploying machine learning models in security-critical applications. Existing research endeavors have proposed many defenses against backdoor attacks. Despite demonstrating certain empirical defense efficacy, none of these techniques could provide a formal and provable security guarantee against arbitrary attacks. As a result, they can be easily broken by strong adaptive attacks, as shown in our evaluation. In this work, we propose TextGuard, the first provable defense against backdoor attacks on text classification. In particular, TextGuard first divides the (backdoored) training data into sub-training sets, achieved by splitting each training sentence into sub-sentences. This partitioning ensures that a majority of the sub-training sets do not contain the backdoor trigger. Subsequently, a base classifier is trained from each sub-training set, and their ensemble provides the final prediction. We theoretically prove that when the length of the backdoor trigger falls within a certain threshold, TextGuard guarantees that its prediction will remain unaffected by the presence of the triggers in training and testing inputs. In our evaluation, we demonstrate the effectiveness of TextGuard on three benchmark text classification tasks, surpassing the certification accuracy of existing certified defenses against backdoor attacks. Furthermore, we propose additional strategies to enhance the empirical performance of TextGuard. Comparisons with state-of-the-art empirical defenses validate the superiority of TextGuard in countering multiple backdoor attacks. Our code and data are available at https://github.com/AI-secure/TextGuard.

{{</citation>}}


### (43/62) Robust Network Slicing: Multi-Agent Policies, Adversarial Attacks, and Defensive Strategies (Feng Wang et al., 2023)

{{<citation>}}

Feng Wang, M. Cenk Gursoy, Senem Velipasalar. (2023)  
**Robust Network Slicing: Multi-Agent Policies, Adversarial Attacks, and Defensive Strategies**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs-MA, cs.LG  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2311.11206v1)  

---


**ABSTRACT**  
In this paper, we present a multi-agent deep reinforcement learning (deep RL) framework for network slicing in a dynamic environment with multiple base stations and multiple users. In particular, we propose a novel deep RL framework with multiple actors and centralized critic (MACC) in which actors are implemented as pointer networks to fit the varying dimension of input. We evaluate the performance of the proposed deep RL algorithm via simulations to demonstrate its effectiveness. Subsequently, we develop a deep RL based jammer with limited prior information and limited power budget. The goal of the jammer is to minimize the transmission rates achieved with network slicing and thus degrade the network slicing agents' performance. We design a jammer with both listening and jamming phases and address jamming location optimization as well as jamming channel optimization via deep RL. We evaluate the jammer at the optimized location, generating interference attacks in the optimized set of channels by switching between the jamming phase and listening phase. We show that the proposed jammer can significantly reduce the victims' performance without direct feedback or prior knowledge on the network slicing policies. Finally, we devise a Nash-equilibrium-supervised policy ensemble mixed strategy profile for network slicing (as a defensive measure) and jamming. We evaluate the performance of the proposed policy ensemble algorithm by applying on the network slicing agents and the jammer agent in simulations to show its effectiveness.

{{</citation>}}


### (44/62) Unmasking and Improving Data Credibility: A Study with Datasets for Training Harmless Language Models (Zhaowei Zhu et al., 2023)

{{<citation>}}

Zhaowei Zhu, Jialu Wang, Hao Cheng, Yang Liu. (2023)  
**Unmasking and Improving Data Credibility: A Study with Datasets for Training Harmless Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11202v1)  

---


**ABSTRACT**  
Language models have shown promise in various tasks but can be affected by undesired data during training, fine-tuning, or alignment. For example, if some unsafe conversations are wrongly annotated as safe ones, the model fine-tuned on these samples may be harmful. Therefore, the correctness of annotations, i.e., the credibility of the dataset, is important. This study focuses on the credibility of real-world datasets, including the popular benchmarks Jigsaw Civil Comments, Anthropic Harmless & Red Team, PKU BeaverTails & SafeRLHF, that can be used for training a harmless language model. Given the cost and difficulty of cleaning these datasets by humans, we introduce a systematic framework for evaluating the credibility of datasets, identifying label errors, and evaluating the influence of noisy labels in the curated language data, specifically focusing on unsafe comments and conversation classification. With the framework, we find and fix an average of 6.16% label errors in 11 datasets constructed from the above benchmarks. The data credibility and downstream learning performance can be remarkably improved by directly fixing label errors, indicating the significance of cleaning existing real-world datasets. Open-source: https://github.com/Docta-ai/docta.

{{</citation>}}


## cs.CY (3)



### (45/62) An Alternative to Regulation: The Case for Public AI (Nicholas Vincent et al., 2023)

{{<citation>}}

Nicholas Vincent, David Bau, Sarah Schwettmann, Joshua Tan. (2023)  
**An Alternative to Regulation: The Case for Public AI**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11350v1)  

---


**ABSTRACT**  
Can governments build AI? In this paper, we describe an ongoing effort to develop ``public AI'' -- publicly accessible AI models funded, provisioned, and governed by governments or other public bodies. Public AI presents both an alternative and a complement to standard regulatory approaches to AI, but it also suggests new technical and policy challenges. We present a roadmap for how the ML research community can help shape this initiative and support its implementation, and how public AI can complement other responsible AI initiatives.

{{</citation>}}


### (46/62) Individual misinformation tagging reinforces echo chambers; Collective tagging does not (Junsol Kim et al., 2023)

{{<citation>}}

Junsol Kim, Zhao Wang, Haohan Shi, Hsin-Keng Ling, James Evans. (2023)  
**Individual misinformation tagging reinforces echo chambers; Collective tagging does not**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-SI, cs.CY  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2311.11282v1)  

---


**ABSTRACT**  
Fears about the destabilizing impact of misinformation online have motivated individuals and platforms to respond. Individuals have become empowered to challenge others' online claims with fact-checks in pursuit of a healthier information ecosystem and to break down echo chambers of self-reinforcing opinion. Using Twitter data, here we show the consequences of individual misinformation tagging: tagged posters had explored novel political information and expanded topical interests immediately prior, but being tagged caused posters to retreat into information bubbles. These unintended consequences were softened by a collective verification system for misinformation moderation. In Twitter's new platform, Community Notes, misinformation tagging was peer-reviewed by other fact-checkers before the exposure. With collective misinformation tagging, posters were less likely to retreat from diverse information consumption. Detailed comparison suggests differences in toxicity, sentiment, readability, and delay in individual versus collective misinformation tagging messages. These findings provide evidence for differential impacts from individual versus collective moderation strategies on the diversity of information consumption and mobility across the information ecosystem.

{{</citation>}}


### (47/62) Assessing AI Impact Assessments: A Classroom Study (Nari Johnson et al., 2023)

{{<citation>}}

Nari Johnson, Hoda Heidari. (2023)  
**Assessing AI Impact Assessments: A Classroom Study**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-HC, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.11193v1)  

---


**ABSTRACT**  
Artificial Intelligence Impact Assessments ("AIIAs"), a family of tools that provide structured processes to imagine the possible impacts of a proposed AI system, have become an increasingly popular proposal to govern AI systems. Recent efforts from government or private-sector organizations have proposed many diverse instantiations of AIIAs, which take a variety of forms ranging from open-ended questionnaires to graded score-cards. However, to date that has been limited evaluation of existing AIIA instruments. We conduct a classroom study (N = 38) at a large research-intensive university (R1) in an elective course focused on the societal and ethical implications of AI. We assign students to different organizational roles (for example, an ML scientist or product manager) and ask participant teams to complete one of three existing AI impact assessments for one of two imagined generative AI systems. In our thematic analysis of participants' responses to pre- and post-activity questionnaires, we find preliminary evidence that impact assessments can influence participants' perceptions of the potential risks of generative AI systems, and the level of responsibility held by AI experts in addressing potential harm. We also discover a consistent set of limitations shared by several existing AIIA instruments, which we group into concerns about their format and content, as well as the feasibility and effectiveness of the activity in foreseeing and mitigating potential harms. Drawing on the findings of this study, we provide recommendations for future work on developing and validating AIIAs.

{{</citation>}}


## cs.DL (1)



### (48/62) Using Causal Threads to Explain Changes in a Dynamic System (Robert B. Allen, 2023)

{{<citation>}}

Robert B. Allen. (2023)  
**Using Causal Threads to Explain Changes in a Dynamic System**  

---
Primary Category: cs.DL  
Categories: cs-AI, cs-DL, cs.DL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11334v1)  

---


**ABSTRACT**  
We explore developing rich semantic models of systems. Specifically, we consider structured causal explanations about state changes in those systems. Essentially, we are developing process-based dynamic knowledge graphs. As an example, we construct a model of the causal threads for geological changes proposed by the Snowball Earth theory. Further, we describe an early prototype of a graphical interface to present the explanations. Unlike statistical approaches to summarization and explanation such as Large Language Models (LLMs), our approach of direct representation can be inspected and verified directly.

{{</citation>}}


## stat.ML (1)



### (49/62) Bounds on Representation-Induced Confounding Bias for Treatment Effect Estimation (Valentyn Melnychuk et al., 2023)

{{<citation>}}

Valentyn Melnychuk, Dennis Frauen, Stefan Feuerriegel. (2023)  
**Bounds on Representation-Induced Confounding Bias for Treatment Effect Estimation**  

---
Primary Category: stat.ML  
Categories: cs-AI, cs-LG, stat-ML, stat.ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.11321v1)  

---


**ABSTRACT**  
State-of-the-art methods for conditional average treatment effect (CATE) estimation make widespread use of representation learning. Here, the idea is to reduce the variance of the low-sample CATE estimation by a (potentially constrained) low-dimensional representation. However, low-dimensional representations can lose information about the observed confounders and thus lead to bias, because of which the validity of representation learning for CATE estimation is typically violated. In this paper, we propose a new, representation-agnostic framework for estimating bounds on the representation-induced confounding bias that comes from dimensionality reduction (or other constraints on the representations) in CATE estimation. First, we establish theoretically under which conditions CATEs are non-identifiable given low-dimensional (constrained) representations. Second, as our remedy, we propose to perform partial identification of CATEs or, equivalently, aim at estimating of lower and upper bounds of the representation-induced confounding bias. We demonstrate the effectiveness of our bounds in a series of experiments. In sum, our framework is of direct relevance in practice where the validity of CATE estimation is of importance.

{{</citation>}}


## cs.AI (5)



### (50/62) TPTU-v2: Boosting Task Planning and Tool Usage of Large Language Model-based Agents in Real-world Systems (Yilun Kong et al., 2023)

{{<citation>}}

Yilun Kong, Jingqing Ruan, Yihong Chen, Bin Zhang, Tianpeng Bao, Shiwei Shi, Guoqing Du, Xiaoru Hu, Hangyu Mao, Ziyue Li, Xingyu Zeng, Rui Zhao. (2023)  
**TPTU-v2: Boosting Task Planning and Tool Usage of Large Language Model-based Agents in Real-world Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11315v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated proficiency in addressing tasks that necessitate a combination of task planning and the usage of external tools that require a blend of task planning and the utilization of external tools, such as APIs. However, real-world complex systems present three prevalent challenges concerning task planning and tool usage: (1) The real system usually has a vast array of APIs, so it is impossible to feed the descriptions of all APIs to the prompt of LLMs as the token length is limited; (2) the real system is designed for handling complex tasks, and the base LLMs can hardly plan a correct sub-task order and API-calling order for such tasks; (3) Similar semantics and functionalities among APIs in real systems create challenges for both LLMs and even humans in distinguishing between them. In response, this paper introduces a comprehensive framework aimed at enhancing the Task Planning and Tool Usage (TPTU) abilities of LLM-based agents operating within real-world systems. Our framework comprises three key components designed to address these challenges: (1) the API Retriever selects the most pertinent APIs for the user task among the extensive array available; (2) LLM Finetuner tunes a base LLM so that the finetuned LLM can be more capable for task planning and API calling; (3) the Demo Selector adaptively retrieves different demonstrations related to hard-to-distinguish APIs, which is further used for in-context learning to boost the final performance. We validate our methods using a real-world commercial system as well as an open-sourced academic dataset, and the outcomes clearly showcase the efficacy of each individual component as well as the integrated framework.

{{</citation>}}


### (51/62) A Comprehensive Review on Sentiment Analysis: Tasks, Approaches and Applications (Sudhanshu Kumar et al., 2023)

{{<citation>}}

Sudhanshu Kumar, Partha Pratim Roy, Debi Prosad Dogra, Byung-Gyu Kim. (2023)  
**A Comprehensive Review on Sentiment Analysis: Tasks, Approaches and Applications**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Natural Language Processing, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2311.11250v1)  

---


**ABSTRACT**  
Sentiment analysis (SA) is an emerging field in text mining. It is the process of computationally identifying and categorizing opinions expressed in a piece of text over different social media platforms. Social media plays an essential role in knowing the customer mindset towards a product, services, and the latest market trends. Most organizations depend on the customer's response and feedback to upgrade their offered products and services. SA or opinion mining seems to be a promising research area for various domains. It plays a vital role in analyzing big data generated daily in structured and unstructured formats over the internet. This survey paper defines sentiment and its recent research and development in different domains, including voice, images, videos, and text. The challenges and opportunities of sentiment analysis are also discussed in the paper.   \keywords{Sentiment Analysis, Machine Learning, Lexicon-based approach, Deep Learning, Natural Language Processing}

{{</citation>}}


### (52/62) Implementation of AI Deep Learning Algorithm For Multi-Modal Sentiment Analysis (Jiazhen Wang, 2023)

{{<citation>}}

Jiazhen Wang. (2023)  
**Implementation of AI Deep Learning Algorithm For Multi-Modal Sentiment Analysis**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2311.11237v1)  

---


**ABSTRACT**  
A multi-modal emotion recognition method was established by combining two-channel convolutional neural network with ring network. This method can extract emotional information effectively and improve learning efficiency. The words were vectorized with GloVe, and the word vector was input into the convolutional neural network. Combining attention mechanism and maximum pool converter BiSRU channel, the local deep emotion and pre-post sequential emotion semantics are obtained. Finally, multiple features are fused and input as the polarity of emotion, so as to achieve the emotion analysis of the target. Experiments show that the emotion analysis method based on feature fusion can effectively improve the recognition accuracy of emotion data set and reduce the learning time. The model has a certain generalization.

{{</citation>}}


### (53/62) Can We Utilize Pre-trained Language Models within Causal Discovery Algorithms? (Chanhui Lee et al., 2023)

{{<citation>}}

Chanhui Lee, Juhyeon Kim, Yongjun Jeong, Juhyun Lyu, Junghee Kim, Sangmin Lee, Sangjun Han, Hyeokjun Choe, Soyeon Park, Woohyung Lim, Sungbin Lim, Sanghack Lee. (2023)  
**Can We Utilize Pre-trained Language Models within Causal Discovery Algorithms?**  

---
Primary Category: cs.AI  
Categories: I-2, cs-AI, cs-LG, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.11212v1)  

---


**ABSTRACT**  
Scaling laws have allowed Pre-trained Language Models (PLMs) into the field of causal reasoning. Causal reasoning of PLM relies solely on text-based descriptions, in contrast to causal discovery which aims to determine the causal relationships between variables utilizing data. Recently, there has been current research regarding a method that mimics causal discovery by aggregating the outcomes of repetitive causal reasoning, achieved through specifically designed prompts. It highlights the usefulness of PLMs in discovering cause and effect, which is often limited by a lack of data, especially when dealing with multiple variables. Conversely, the characteristics of PLMs which are that PLMs do not analyze data and they are highly dependent on prompt design leads to a crucial limitation for directly using PLMs in causal discovery. Accordingly, PLM-based causal reasoning deeply depends on the prompt design and carries out the risk of overconfidence and false predictions in determining causal relationships. In this paper, we empirically demonstrate the aforementioned limitations of PLM-based causal reasoning through experiments on physics-inspired synthetic data. Then, we propose a new framework that integrates prior knowledge obtained from PLM with a causal discovery algorithm. This is accomplished by initializing an adjacency matrix for causal discovery and incorporating regularization using prior knowledge. Our proposed framework not only demonstrates improved performance through the integration of PLM and causal discovery but also suggests how to leverage PLM-extracted prior knowledge with existing causal discovery algorithms.

{{</citation>}}


### (54/62) Leveraging Generative AI for Clinical Evidence Summarization Needs to Achieve Trustworthiness (Gongbo Zhang et al., 2023)

{{<citation>}}

Gongbo Zhang, Qiao Jin, Denis Jered McInerney, Yong Chen, Fei Wang, Curtis L. Cole, Qian Yang, Yanshan Wang, Bradley A. Malin, Mor Peleg, Byron C. Wallace, Zhiyong Lu, Chunhua Weng, Yifan Peng. (2023)  
**Leveraging Generative AI for Clinical Evidence Summarization Needs to Achieve Trustworthiness**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Clinical, Generative AI, Summarization  
[Paper Link](http://arxiv.org/abs/2311.11211v1)  

---


**ABSTRACT**  
Evidence-based medicine aims to improve the quality of healthcare by empowering medical decisions and practices with the best available evidence. The rapid growth of medical evidence, which can be obtained from various sources, poses a challenge in collecting, appraising, and synthesizing the evidential information. Recent advancements in generative AI, exemplified by large language models, hold promise in facilitating the arduous task. However, developing accountable, fair, and inclusive models remains a complicated undertaking. In this perspective, we discuss the trustworthiness of generative AI in the context of automated summarization of medical evidence.

{{</citation>}}


## cs.RO (1)



### (55/62) Tactile Active Inference Reinforcement Learning for Efficient Robotic Manipulation Skill Acquisition (Zihao Liu et al., 2023)

{{<citation>}}

Zihao Liu, Xing Liu, Yizhai Zhang, Zhengxiong Liu, Panfeng Huang. (2023)  
**Tactile Active Inference Reinforcement Learning for Efficient Robotic Manipulation Skill Acquisition**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.11287v1)  

---


**ABSTRACT**  
Robotic manipulation holds the potential to replace humans in the execution of tedious or dangerous tasks. However, control-based approaches are not suitable due to the difficulty of formally describing open-world manipulation in reality, and the inefficiency of existing learning methods. Thus, applying manipulation in a wide range of scenarios presents significant challenges. In this study, we propose a novel method for skill learning in robotic manipulation called Tactile Active Inference Reinforcement Learning (Tactile-AIRL), aimed at achieving efficient training. To enhance the performance of reinforcement learning (RL), we introduce active inference, which integrates model-based techniques and intrinsic curiosity into the RL process. This integration improves the algorithm's training efficiency and adaptability to sparse rewards. Additionally, we utilize a vision-based tactile sensor to provide detailed perception for manipulation tasks. Finally, we employ a model-based approach to imagine and plan appropriate actions through free energy minimization. Simulation results demonstrate that our method achieves significantly high training efficiency in non-prehensile objects pushing tasks. It enables agents to excel in both dense and sparse reward tasks with just a few interaction episodes, surpassing the SAC baseline. Furthermore, we conduct physical experiments on a gripper screwing task using our method, which showcases the algorithm's rapid learning capability and its potential for practical applications.

{{</citation>}}


## eess.SY (3)



### (56/62) Multi-Timescale Control and Communications with Deep Reinforcement Learning -- Part I: Communication-Aware Vehicle Control (Tong Liu et al., 2023)

{{<citation>}}

Tong Liu, Lei Lei, Kan Zheng, Xuemin, Shen. (2023)  
**Multi-Timescale Control and Communications with Deep Reinforcement Learning -- Part I: Communication-Aware Vehicle Control**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.11281v1)  

---


**ABSTRACT**  
An intelligent decision-making system enabled by Vehicle-to-Everything (V2X) communications is essential to achieve safe and efficient autonomous driving (AD), where two types of decisions have to be made at different timescales, i.e., vehicle control and radio resource allocation (RRA) decisions. The interplay between RRA and vehicle control necessitates their collaborative design. In this two-part paper (Part I and Part II), taking platoon control (PC) as an example use case, we propose a joint optimization framework of multi-timescale control and communications (MTCC) based on Deep Reinforcement Learning (DRL). In this paper (Part I), we first decompose the problem into a communication-aware DRL-based PC sub-problem and a control-aware DRL-based RRA sub-problem. Then, we focus on the PC sub-problem assuming an RRA policy is given, and propose the MTCC-PC algorithm to learn an efficient PC policy. To improve the PC performance under random observation delay, the PC state space is augmented with the observation delay and PC action history. Moreover, the reward function with respect to the augmented state is defined to construct an augmented state Markov Decision Process (MDP). It is proved that the optimal policy for the augmented state MDP is optimal for the original PC problem with observation delay. Different from most existing works on communication-aware control, the MTCC-PC algorithm is trained in a delayed environment generated by the fine-grained embedded simulation of C-V2X communications rather than by a simple stochastic delay model. Finally, experiments are performed to compare the performance of MTCC-PC with those of the baseline DRL algorithms.

{{</citation>}}


### (57/62) Multi-Timescale Control and Communications with Deep Reinforcement Learning -- Part II: Control-Aware Radio Resource Allocation (Lei Lei et al., 2023)

{{<citation>}}

Lei Lei, Tong Liu, Kan Zheng, Xuemin, Shen. (2023)  
**Multi-Timescale Control and Communications with Deep Reinforcement Learning -- Part II: Control-Aware Radio Resource Allocation**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.11280v1)  

---


**ABSTRACT**  
In Part I of this two-part paper (Multi-Timescale Control and Communications with Deep Reinforcement Learning -- Part I: Communication-Aware Vehicle Control), we decomposed the multi-timescale control and communications (MTCC) problem in Cellular Vehicle-to-Everything (C-V2X) system into a communication-aware Deep Reinforcement Learning (DRL)-based platoon control (PC) sub-problem and a control-aware DRL-based radio resource allocation (RRA) sub-problem. We focused on the PC sub-problem and proposed the MTCC-PC algorithm to learn an optimal PC policy given an RRA policy. In this paper (Part II), we first focus on the RRA sub-problem in MTCC assuming a PC policy is given, and propose the MTCC-RRA algorithm to learn the RRA policy. Specifically, we incorporate the PC advantage function in the RRA reward function, which quantifies the amount of PC performance degradation caused by observation delay. Moreover, we augment the state space of RRA with PC action history for a more well-informed RRA policy. In addition, we utilize reward shaping and reward backpropagation prioritized experience replay (RBPER) techniques to efficiently tackle the multi-agent and sparse reward problems, respectively. Finally, a sample- and computational-efficient training approach is proposed to jointly learn the PC and RRA policies in an iterative process. In order to verify the effectiveness of the proposed MTCC algorithm, we performed experiments using real driving data for the leading vehicle, where the performance of MTCC is compared with those of the baseline DRL algorithms.

{{</citation>}}


### (58/62) ChatGPT at the Speed of Light: Optical Comb-Based Monolithic Photonic-Electronic Linear-Algebra Accelerators (Tzu-Chien Hsueh et al., 2023)

{{<citation>}}

Tzu-Chien Hsueh, Yeshaiahu Fainman, Bill Lin. (2023)  
**ChatGPT at the Speed of Light: Optical Comb-Based Monolithic Photonic-Electronic Linear-Algebra Accelerators**  

---
Primary Category: eess.SY  
Categories: cs-ET, cs-SY, eess-SY, eess.SY  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.11224v1)  

---


**ABSTRACT**  
This paper proposes to adopt advanced monolithic silicon-photonics integrated-circuits manufacturing capabilities to achieve a system-on-chip photonic-electronic linear-algebra accelerator with the features of optical comb-based broadband incoherent photo-detections and high-dimensional operations of consecutive matrix-matrix multiplications to enable substantial leaps in computation density and energy efficiency, with practical considerations of power/area overhead due to photonic-electronic on-chip conversions, integrations, and calibrations through holistic co-design approaches to support attention-head mechanism based deep-learning neural networks used in Large Language Models and other emergent applications.

{{</citation>}}


## cs.SD (1)



### (59/62) M$^{2}$UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models (Atin Sakkeer Hussain et al., 2023)

{{<citation>}}

Atin Sakkeer Hussain, Shansong Liu, Chenshuo Sun, Ying Shan. (2023)  
**M$^{2}$UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models**  

---
Primary Category: cs.SD  
Categories: cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2311.11255v1)  

---


**ABSTRACT**  
The current landscape of research leveraging large language models (LLMs) is experiencing a surge. Many works harness the powerful reasoning capabilities of these models to comprehend various modalities, such as text, speech, images, videos, etc. They also utilize LLMs to understand human intention and generate desired outputs like images, videos, and music. However, research that combines both understanding and generation using LLMs is still limited and in its nascent stage. To address this gap, we introduce a Multi-modal Music Understanding and Generation (M$^{2}$UGen) framework that integrates LLM's abilities to comprehend and generate music for different modalities. The M$^{2}$UGen framework is purpose-built to unlock creative potential from diverse sources of inspiration, encompassing music, image, and video through the use of pretrained MERT, ViT, and ViViT models, respectively. To enable music generation, we explore the use of AudioLDM 2 and MusicGen. Bridging multi-modal understanding and music generation is accomplished through the integration of the LLaMA 2 model. Furthermore, we make use of the MU-LLaMA model to generate extensive datasets that support text/image/video-to-music generation, facilitating the training of our M$^{2}$UGen framework. We conduct a thorough evaluation of our proposed framework. The experimental results demonstrate that our model achieves or surpasses the performance of the current state-of-the-art models.

{{</citation>}}


## cs.IR (1)



### (60/62) Dependency Relationships-Enhanced Attentive Group Recommendation in HINs (Juntao Zhang et al., 2023)

{{<citation>}}

Juntao Zhang, Sheng Wang, Zhiyu Chen, Xiandi Yang, Zhiyong Peng. (2023)  
**Dependency Relationships-Enhanced Attentive Group Recommendation in HINs**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Attention, Embedding  
[Paper Link](http://arxiv.org/abs/2311.11239v1)  

---


**ABSTRACT**  
Recommending suitable items to a group of users, commonly referred to as the group recommendation task, is becoming increasingly urgent with the development of group activities. The challenges within the group recommendation task involve aggregating the individual preferences of group members as the group's preferences and facing serious sparsity problems due to the lack of user/group-item interactions. To solve these problems, we propose a novel approach called Dependency Relationships-Enhanced Attentive Group Recommendation (DREAGR) for the recommendation task of occasional groups. Specifically, we introduce the dependency relationship between items as side information to enhance the user/group-item interaction and alleviate the interaction sparsity problem. Then, we propose a Path-Aware Attention Embedding (PAAE) method to model users' preferences on different types of paths. Next, we design a gated fusion mechanism to fuse users' preferences into their comprehensive preferences. Finally, we develop an attention aggregator that aggregates users' preferences as the group's preferences for the group recommendation task. We conducted experiments on two datasets to demonstrate the superiority of DREAGR by comparing it with state-of-the-art group recommender models. The experimental results show that DREAGR outperforms other models, especially HR@N and NDCG@N (N=5, 10), where DREAGR has improved in the range of 3.64% to 7.01% and 2.57% to 3.39% on both datasets, respectively.

{{</citation>}}


## eess.IV (1)



### (61/62) Enhancing Radiology Diagnosis through Convolutional Neural Networks for Computer Vision in Healthcare (Keshav Kumar K. et al., 2023)

{{<citation>}}

Keshav Kumar K., Dr N V S L Narasimham. (2023)  
**Enhancing Radiology Diagnosis through Convolutional Neural Networks for Computer Vision in Healthcare**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2311.11234v1)  

---


**ABSTRACT**  
The transformative power of Convolutional Neural Networks (CNNs) in radiology diagnostics is examined in this study, with a focus on interpretability, effectiveness, and ethical issues. With an altered DenseNet architecture, the CNN performs admirably in terms of particularity, sensitivity, as well as accuracy. Its superiority over conventional methods is validated by comparative analyses, which highlight efficiency gains. Nonetheless, interpretability issues highlight the necessity of sophisticated methods in addition to continuous model improvement. Integration issues like interoperability and radiologists' training lead to suggestions for teamwork. Systematic consideration of the ethical implications is carried out, necessitating extensive frameworks. Refinement of architectures, interpretability, alongside ethical considerations need to be prioritized in future work for responsible CNN deployment in radiology diagnostics.

{{</citation>}}


## eess.SP (1)



### (62/62) Link Streams as a Generalization of Graphs and Time Series (Esteban Bautista et al., 2023)

{{<citation>}}

Esteban Bautista, Matthieu Latapy. (2023)  
**Link Streams as a Generalization of Graphs and Time Series**  

---
Primary Category: eess.SP  
Categories: cs-SI, eess-SP, eess.SP  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2311.11187v1)  

---


**ABSTRACT**  
A link stream is a set of possibly weighted triplets (t, u, v) modeling that u and v interacted at time t. Link streams offer an effective model for datasets containing both temporal and relational information, making their proper analysis crucial in many applications. They are commonly regarded as sequences of graphs or collections of time series. Yet, a recent seminal work demonstrated that link streams are more general objects of which graphs are only particular cases. It therefore started the construction of a dedicated formalism for link streams by extending graph theory. In this work, we contribute to the development of this formalism by showing that link streams also generalize time series. In particular, we show that a link stream corresponds to a time-series extended to a relational dimension, which opens the door to also extend the framework of signal processing to link streams. We therefore develop extensions of numerous signal concepts to link streams: from elementary ones like energy, correlation, and differentiation, to more advanced ones like Fourier transform and filters.

{{</citation>}}
