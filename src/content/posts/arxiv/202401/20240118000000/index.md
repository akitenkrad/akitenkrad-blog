---
draft: false
title: "arXiv @ 2024.01.18"
date: 2024-01-18
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.18"
    identifier: arxiv_20240118
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CR (5)](#cscr-5)
- [cs.HC (6)](#cshc-6)
- [cs.CV (33)](#cscv-33)
- [cs.CY (3)](#cscy-3)
- [cs.NI (2)](#csni-2)
- [cs.IR (4)](#csir-4)
- [cs.CL (28)](#cscl-28)
- [cs.SI (3)](#cssi-3)
- [cs.LG (18)](#cslg-18)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.SE (5)](#csse-5)
- [cs.DL (1)](#csdl-1)
- [hep-ex (1)](#hep-ex-1)
- [cs.AI (6)](#csai-6)
- [quant-ph (1)](#quant-ph-1)
- [eess.SY (3)](#eesssy-3)
- [cs.LO (1)](#cslo-1)
- [cs.SD (2)](#cssd-2)
- [physics.comp-ph (1)](#physicscomp-ph-1)
- [eess.IV (1)](#eessiv-1)
- [eess.AS (2)](#eessas-2)
- [stat.ML (1)](#statml-1)
- [cs.DC (1)](#csdc-1)
- [cs.RO (1)](#csro-1)
- [cond-mat.mtrl-sci (1)](#cond-matmtrl-sci-1)
- [cs.IT (1)](#csit-1)

## cs.CR (5)



### (1/132) Whispering Pixels: Exploiting Uninitialized Register Accesses in Modern GPUs (Frederik Dermot Pustelnik et al., 2024)

{{<citation>}}

Frederik Dermot Pustelnik, Xhani Marvin Saß, Jean-Pierre Seifert. (2024)  
**Whispering Pixels: Exploiting Uninitialized Register Accesses in Modern GPUs**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.08881v1)  

---


**ABSTRACT**  
Graphic Processing Units (GPUs) have transcended their traditional use-case of rendering graphics and nowadays also serve as a powerful platform for accelerating ubiquitous, non-graphical rendering tasks. One prominent task is inference of neural networks, which process vast amounts of personal data, such as audio, text or images. Thus, GPUs became integral components for handling vast amounts of potentially confidential data, which has awakened the interest of security researchers. This lead to the discovery of various vulnerabilities in GPUs in recent years. In this paper, we uncover yet another vulnerability class in GPUs: We found that some GPU implementations lack proper register initialization routines before shader execution, leading to unintended register content leakage of previously executed shader kernels. We showcase the existence of the aforementioned vulnerability on products of 3 major vendors - Apple, NVIDIA and Qualcomm. The vulnerability poses unique challenges to an adversary due to opaque scheduling and register remapping algorithms present in the GPU firmware, complicating the reconstruction of leaked data. In order to illustrate the real-world impact of this flaw, we showcase how these challenges can be solved for attacking various workloads on the GPU. First, we showcase how uninitialized registers leak arbitrary pixel data processed by fragment shaders. We further implement information leakage attacks on intermediate data of Convolutional Neural Networks (CNNs) and present the attack's capability to leak and reconstruct the output of Large Language Models (LLMs).

{{</citation>}}


### (2/132) ADVENT: Attack/Anomaly Detection in VANETs (Hamideh Baharlouei et al., 2024)

{{<citation>}}

Hamideh Baharlouei, Adetokunbo Makanju, Nur Zincir-Heywood. (2024)  
**ADVENT: Attack/Anomaly Detection in VANETs**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2401.08564v1)  

---


**ABSTRACT**  
In the domain of Vehicular Ad hoc Networks (VANETs), where the imperative of having a real-world malicious detector capable of detecting attacks in real-time and unveiling their perpetrators is crucial, our study introduces a system with this goal. This system is designed for real-time detection of malicious behavior, addressing the critical need to first identify the onset of attacks and subsequently the responsible actors. Prior work in this area have never addressed both requirements, which we believe are necessary for real world deployment, simultaneously. By seamlessly integrating statistical and machine learning techniques, the proposed system prioritizes simplicity and efficiency. It excels in swiftly detecting attack onsets with a remarkable F1-score of 99.66%, subsequently identifying malicious vehicles with an average F1-score of approximately 97.85%. Incorporating federated learning in both stages enhances privacy and improves the efficiency of malicious node detection, effectively reducing the false negative rate.

{{</citation>}}


### (3/132) Security and Privacy Issues and Solutions in Federated Learning for Digital Healthcare (Hyejun Jeong et al., 2024)

{{<citation>}}

Hyejun Jeong, Tai-Myoung Chung. (2024)  
**Security and Privacy Issues and Solutions in Federated Learning for Digital Healthcare**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.08458v1)  

---


**ABSTRACT**  
The advent of Federated Learning has enabled the creation of a high-performing model as if it had been trained on a considerable amount of data. A multitude of participants and a server cooperatively train a model without the need for data disclosure or collection. The healthcare industry, where security and privacy are paramount, can substantially benefit from this new learning paradigm, as data collection is no longer feasible due to stringent data policies. Nonetheless, unaddressed challenges and insufficient attack mitigation are hampering its adoption. Attack surfaces differ from traditional centralized learning in that the server and clients communicate between each round of training. In this paper, we thus present vulnerabilities, attacks, and defenses based on the widened attack surfaces, as well as suggest promising new research directions toward a more robust FL.

{{</citation>}}


### (4/132) Mitigating Bias in Machine Learning Models for Phishing Webpage Detection (Aditya Kulkarni et al., 2024)

{{<citation>}}

Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das. (2024)  
**Mitigating Bias in Machine Learning Models for Phishing Webpage Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.08363v1)  

---


**ABSTRACT**  
The widespread accessibility of the Internet has led to a surge in online fraudulent activities, underscoring the necessity of shielding users' sensitive information from cybercriminals. Phishing, a well-known cyberattack, revolves around the creation of phishing webpages and the dissemination of corresponding URLs, aiming to deceive users into sharing their sensitive information, often for identity theft or financial gain. Various techniques are available for preemptively categorizing zero-day phishing URLs by distilling unique attributes and constructing predictive models. However, these existing techniques encounter unresolved issues. This proposal delves into persistent challenges within phishing detection solutions, particularly concentrated on the preliminary phase of assembling comprehensive datasets, and proposes a potential solution in the form of a tool engineered to alleviate bias in ML models. Such a tool can generate phishing webpages for any given set of legitimate URLs, infusing randomly selected content and visual-based phishing features. Furthermore, we contend that the tool holds the potential to assess the efficacy of existing phishing detection solutions, especially those trained on confined datasets.

{{</citation>}}


### (5/132) IoTWarden: A Deep Reinforcement Learning Based Real-time Defense System to Mitigate Trigger-action IoT Attacks (Md Morshed Alam et al., 2024)

{{<citation>}}

Md Morshed Alam, Israt Jahan, Weichao Wang. (2024)  
**IoTWarden: A Deep Reinforcement Learning Based Real-time Defense System to Mitigate Trigger-action IoT Attacks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.08141v1)  

---


**ABSTRACT**  
In trigger-action IoT platforms, IoT devices report event conditions to IoT hubs notifying their cyber states and let the hubs invoke actions in other IoT devices based on functional dependencies defined as rules in a rule engine. These functional dependencies create a chain of interactions that help automate network tasks. Adversaries exploit this chain to report fake event conditions to IoT hubs and perform remote injection attacks upon a smart environment to indirectly control targeted IoT devices. Existing defense efforts usually depend on static analysis over IoT apps to develop rule-based anomaly detection mechanisms. We also see ML-based defense mechanisms in the literature that harness physical event fingerprints to determine anomalies in an IoT network. However, these methods often demonstrate long response time and lack of adaptability when facing complicated attacks. In this paper, we propose to build a deep reinforcement learning based real-time defense system for injection attacks. We define the reward functions for defenders and implement a deep Q-network based approach to identify the optimal defense policy. Our experiments show that the proposed mechanism can effectively and accurately identify and defend against injection attacks with reasonable computation overhead.

{{</citation>}}


## cs.HC (6)



### (6/132) Evaluating the Utility of Conformal Prediction Sets for AI-Advised Image Labeling (Dongping Zhang et al., 2024)

{{<citation>}}

Dongping Zhang, Angelos Chatzimparmpas, Negar Kamali, Jessica Hullman. (2024)  
**Evaluating the Utility of Conformal Prediction Sets for AI-Advised Image Labeling**  

---
Primary Category: cs.HC  
Categories: cs-CV, cs-HC, cs-LG, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08876v1)  

---


**ABSTRACT**  
As deep neural networks are more commonly deployed in high-stakes domains, their lack of interpretability makes uncertainty quantification challenging. We investigate the effects of presenting conformal prediction sets$\unicode{x2013}$a method for generating valid confidence sets in distribution-free uncertainty quantification$\unicode{x2013}$to express uncertainty in AI-advised decision-making. Through a large pre-registered experiment, we compare the utility of conformal prediction sets to displays of Top-1 and Top-k predictions for AI-advised image labeling. We find that the utility of prediction sets for accuracy varies with the difficulty of the task: while they result in accuracy on par with or less than Top-1 and Top-k displays for easy images, prediction sets excel at assisting humans in labeling out-of-distribution (OOD) images especially when the set size is small. Our results empirically pinpoint the practical challenges of conformal prediction sets and provide implications on how to incorporate them for real-world decision-making.

{{</citation>}}


### (7/132) Multimodal assessment of best possible self as a self-regulatory activity for the classroom (Batuhan Sayis et al., 2024)

{{<citation>}}

Batuhan Sayis, Marc Beardsley, Marta Portero-Tresserra. (2024)  
**Multimodal assessment of best possible self as a self-regulatory activity for the classroom**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08424v1)  

---


**ABSTRACT**  
Best possible self (BPS) is a positive psychological intervention shown to enhance well-being which involves writing a description of an ideal future scenario. This paper presents a comparison of psychophysiological effects of a BPS activity that has been adapted for classroom settings and a time-matched control activity (NA). Thirty-three undergraduate students participated in the study that assessed state anxiety (State-Trait Anxiety Inventory, STAI), affect (Affective Slider, AS), and cardiac vagal activity (heart-rate variability, HRV) as an indicator of self-regulatory resource usage, at three time periods (PRE, DURING, POST). Results show that BPS led to a significantly greater increase in positive valence (DURING) and overall higher levels of cardiac vagal activity (HRV) compared to NA. These findings suggest that BPS has promising characteristics as a self-regulatory technique aimed at fostering positive affect and positively impacting self-regulatory resources. As BPS does not require expert knowledge nor specialized technology to administer, it may be a suitable activity for educators to use when teaching and having students practice self-regulation. This study presents evidence collected in a replicable multimodal approach of the self-regulatory effects of a brief BPS activity on undergraduate students.

{{</citation>}}


### (8/132) Interrogating AI: Characterizing Emergent Playful Interactions with ChatGPT (Mohammad Ronagh Nikghalb et al., 2024)

{{<citation>}}

Mohammad Ronagh Nikghalb, Jinghui Cheng. (2024)  
**Interrogating AI: Characterizing Emergent Playful Interactions with ChatGPT**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.08405v1)  

---


**ABSTRACT**  
In an era of AI's growing capabilities and influences, recent advancements are reshaping HCI and CSCW's view of AI as mere tools. Playful interactions with AI systems naturally emerged as a way for users to make sense of the ever-changing technology. However, these emergent and playful interactions are underexamined. We target this gap by investigating playful interactions exhibited by users of a recently trending powerful AI technology, ChatGPT. Through a thematic analysis of 372 user-generated posts on the ChatGPT subreddit, we found that a substantial portion of user discourse revolves around playful interactions. The analysis further allowed us to construct a preliminary taxonomy to describe these interactions, categorizing them into six types: reflecting, jesting, imitating, challenging, tricking, and contriving; each included sub-categories. Overall, this study contributes to the field of HCI and CSCW by illuminating the multifaceted nature of playful interactions with AI, underlining their significance in shaping the human-AI relationship.

{{</citation>}}


### (9/132) Understanding User Experience in Large Language Model Interactions (Jiayin Wang et al., 2024)

{{<citation>}}

Jiayin Wang, Weizhi Ma, Peijie Sun, Min Zhang, Jian-Yun Nie. (2024)  
**Understanding User Experience in Large Language Model Interactions**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.08329v1)  

---


**ABSTRACT**  
In the rapidly evolving landscape of large language models (LLMs), most research has primarily viewed them as independent individuals, focusing on assessing their capabilities through standardized benchmarks and enhancing their general intelligence. This perspective, however, tends to overlook the vital role of LLMs as user-centric services in human-AI collaboration. This gap in research becomes increasingly critical as LLMs become more integrated into people's everyday and professional interactions. This study addresses the important need to understand user satisfaction with LLMs by exploring four key aspects: comprehending user intents, scrutinizing user experiences, addressing major user concerns about current LLM services, and charting future research paths to bolster human-AI collaborations. Our study develops a taxonomy of 7 user intents in LLM interactions, grounded in analysis of real-world user interaction logs and human verification. Subsequently, we conduct a user survey to gauge their satisfaction with LLM services, encompassing usage frequency, experiences across intents, and predominant concerns. This survey, compiling 411 anonymous responses, uncovers 11 first-hand insights into the current state of user engagement with LLMs. Based on this empirical analysis, we pinpoint 6 future research directions prioritizing the user perspective in LLM developments. This user-centered approach is essential for crafting LLMs that are not just technologically advanced but also resonate with the intricate realities of human interactions and real-world applications.

{{</citation>}}


### (10/132) TrajVis: a visual clinical decision support system to translate artificial intelligence trajectory models in the precision management of chronic kidney disease (Zuotian Li et al., 2024)

{{<citation>}}

Zuotian Li, Xiang Liu, Ziyang Tang, Pengyue Zhang, Nanxin Jin, Michael Eadon, Qianqian Song, Yingjie Chen, Jing Su. (2024)  
**TrajVis: a visual clinical decision support system to translate artificial intelligence trajectory models in the precision management of chronic kidney disease**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Clinical  
[Paper Link](http://arxiv.org/abs/2401.08067v1)  

---


**ABSTRACT**  
Objective: Our objective is to develop and validate TrajVis, an interactive tool that assists clinicians in using artificial intelligence (AI) models to leverage patients' longitudinal electronic medical records (EMR) for personalized precision management of chronic disease progression. Methods: We first perform requirement analysis with clinicians and data scientists to determine the visual analytics tasks of the TrajVis system as well as its design and functionalities. A graph AI model for chronic kidney disease (CKD) trajectory inference named DEPOT is used for system development and demonstration. TrajVis is implemented as a full-stack web application with synthetic EMR data derived from the Atrium Health Wake Forest Baptist Translational Data Warehouse and the Indiana Network for Patient Care research database. A case study with a nephrologist and a user experience survey of clinicians and data scientists are conducted to evaluate the TrajVis system. Results: The TrajVis clinical information system is composed of four panels: the Patient View for demographic and clinical information, the Trajectory View to visualize the DEPOT-derived CKD trajectories in latent space, the Clinical Indicator View to elucidate longitudinal patterns of clinical features and interpret DEPOT predictions, and the Analysis View to demonstrate personal CKD progression trajectories. System evaluations suggest that TrajVis supports clinicians in summarizing clinical data, identifying individualized risk predictors, and visualizing patient disease progression trajectories, overcoming the barriers of AI implementation in healthcare. Conclusion: TrajVis bridges the gap between the fast-growing AI/ML modeling and the clinical use of such models for personalized and precision management of chronic diseases.

{{</citation>}}


### (11/132) Belief Miner: A Methodology for Discovering Causal Beliefs and Causal Illusions from General Populations (Shahreen Salim et al., 2024)

{{<citation>}}

Shahreen Salim, Md Naimul Hoque, Klaus Mueller. (2024)  
**Belief Miner: A Methodology for Discovering Causal Beliefs and Causal Illusions from General Populations**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2401.08020v1)  

---


**ABSTRACT**  
Causal belief is a cognitive practice that humans apply everyday to reason about cause and effect relations between factors, phenomena, or events. Like optical illusions, humans are prone to drawing causal relations between events that are only coincidental (i.e., causal illusions). Researchers in domains such as cognitive psychology and healthcare often use logistically expensive experiments to understand causal beliefs and illusions. In this paper, we propose Belief Miner, a crowdsourcing method for evaluating people's causal beliefs and illusions. Our method uses the (dis)similarities between the causal relations collected from the crowds and experts to surface the causal beliefs and illusions. Through an iterative design process, we developed a web-based interface for collecting causal relations from a target population. We then conducted a crowdsourced experiment with 101 workers on Amazon Mechanical Turk and Prolific using this interface and analyzed the collected data with Belief Miner. We discovered a variety of causal beliefs and potential illusions, and we report the design implications for future research.

{{</citation>}}


## cs.CV (33)



### (12/132) B-Cos Aligned Transformers Learn Human-Interpretable Features (Manuel Tran et al., 2024)

{{<citation>}}

Manuel Tran, Amal Lahiani, Yashin Dicente Cid, Melanie Boxberg, Peter Lienemann, Christian Matek, Sophia J. Wagner, Fabian J. Theis, Eldad Klaiman, Tingying Peng. (2024)  
**B-Cos Aligned Transformers Learn Human-Interpretable Features**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.08868v2)  

---


**ABSTRACT**  
Vision Transformers (ViTs) and Swin Transformers (Swin) are currently state-of-the-art in computational pathology. However, domain experts are still reluctant to use these models due to their lack of interpretability. This is not surprising, as critical decisions need to be transparent and understandable. The most common approach to understanding transformers is to visualize their attention. However, attention maps of ViTs are often fragmented, leading to unsatisfactory explanations. Here, we introduce a novel architecture called the B-cos Vision Transformer (BvT) that is designed to be more interpretable. It replaces all linear transformations with the B-cos transform to promote weight-input alignment. In a blinded study, medical experts clearly ranked BvTs above ViTs, suggesting that our network is better at capturing biomedically relevant structures. This is also true for the B-cos Swin Transformer (Bwin). Compared to the Swin Transformer, it even improves the F1-score by up to 4.7% on two public datasets.

{{</citation>}}


### (13/132) Cross-Level Multi-Instance Distillation for Self-Supervised Fine-Grained Visual Categorization (Qi Bi et al., 2024)

{{<citation>}}

Qi Bi, Wei Ji, Jingjun Yi, Haolan Zhan, Gui-Song Xia. (2024)  
**Cross-Level Multi-Instance Distillation for Self-Supervised Fine-Grained Visual Categorization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.08860v1)  

---


**ABSTRACT**  
High-quality annotation of fine-grained visual categories demands great expert knowledge, which is taxing and time consuming. Alternatively, learning fine-grained visual representation from enormous unlabeled images (e.g., species, brands) by self-supervised learning becomes a feasible solution. However, recent researches find that existing self-supervised learning methods are less qualified to represent fine-grained categories. The bottleneck lies in that the pre-text representation is built from every patch-wise embedding, while fine-grained categories are only determined by several key patches of an image. In this paper, we propose a Cross-level Multi-instance Distillation (CMD) framework to tackle the challenge. Our key idea is to consider the importance of each image patch in determining the fine-grained pre-text representation by multiple instance learning. To comprehensively learn the relation between informative patches and fine-grained semantics, the multi-instance knowledge distillation is implemented on both the region/image crop pairs from the teacher and student net, and the region-image crops inside the teacher / student net, which we term as intra-level multi-instance distillation and inter-level multi-instance distillation. Extensive experiments on CUB-200-2011, Stanford Cars and FGVC Aircraft show that the proposed method outperforms the contemporary method by upto 10.14% and existing state-of-the-art self-supervised learning approaches by upto 19.78% on both top-1 accuracy and Rank-1 retrieval metric.

{{</citation>}}


### (14/132) Segment Anything Model Can Not Segment Anything: Assessing AI Foundation Model's Generalizability in Permafrost Mapping (Wenwen Li et al., 2024)

{{<citation>}}

Wenwen Li, Chia-Yu Hsu, Sizhe Wang, Yezhou Yang, Hyunho Lee, Anna Liljedahl, Chandi Witharana, Yili Yang, Brendan M. Rogers, Samantha T. Arundel, Matthew B. Jones, Kenton McHenry, Patricia Solis. (2024)  
**Segment Anything Model Can Not Segment Anything: Assessing AI Foundation Model's Generalizability in Permafrost Mapping**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.08787v1)  

---


**ABSTRACT**  
This paper assesses trending AI foundation models, especially emerging computer vision foundation models and their performance in natural landscape feature segmentation. While the term foundation model has quickly garnered interest from the geospatial domain, its definition remains vague. Hence, this paper will first introduce AI foundation models and their defining characteristics. Built upon the tremendous success achieved by Large Language Models (LLMs) as the foundation models for language tasks, this paper discusses the challenges of building foundation models for geospatial artificial intelligence (GeoAI) vision tasks. To evaluate the performance of large AI vision models, especially Meta's Segment Anything Model (SAM), we implemented different instance segmentation pipelines that minimize the changes to SAM to leverage its power as a foundation model. A series of prompt strategies was developed to test SAM's performance regarding its theoretical upper bound of predictive accuracy, zero-shot performance, and domain adaptability through fine-tuning. The analysis used two permafrost feature datasets, ice-wedge polygons and retrogressive thaw slumps because (1) these landform features are more challenging to segment than manmade features due to their complicated formation mechanisms, diverse forms, and vague boundaries; (2) their presence and changes are important indicators for Arctic warming and climate change. The results show that although promising, SAM still has room for improvement to support AI-augmented terrain mapping. The spatial and domain generalizability of this finding is further validated using a more general dataset EuroCrop for agricultural field mapping. Finally, we discuss future research directions that strengthen SAM's applicability in challenging geospatial domains.

{{</citation>}}


### (15/132) MultiPLY: A Multisensory Object-Centric Embodied Large Language Model in 3D World (Yining Hong et al., 2024)

{{<citation>}}

Yining Hong, Zishuo Zheng, Peihao Chen, Yian Wang, Junyan Li, Chuang Gan. (2024)  
**MultiPLY: A Multisensory Object-Centric Embodied Large Language Model in 3D World**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.08577v1)  

---


**ABSTRACT**  
Human beings possess the capability to multiply a melange of multisensory cues while actively exploring and interacting with the 3D world. Current multi-modal large language models, however, passively absorb sensory data as inputs, lacking the capacity to actively interact with the objects in the 3D environment and dynamically collect their multisensory information. To usher in the study of this area, we propose MultiPLY, a multisensory embodied large language model that could incorporate multisensory interactive data, including visual, audio, tactile, and thermal information into large language models, thereby establishing the correlation among words, actions, and percepts. To this end, we first collect Multisensory Universe, a large-scale multisensory interaction dataset comprising 500k data by deploying an LLM-powered embodied agent to engage with the 3D environment. To perform instruction tuning with pre-trained LLM on such generated data, we first encode the 3D scene as abstracted object-centric representations and then introduce action tokens denoting that the embodied agent takes certain actions within the environment, as well as state tokens that represent the multisensory state observations of the agent at each time step. In the inference time, MultiPLY could generate action tokens, instructing the agent to take the action in the environment and obtain the next multisensory state observation. The observation is then appended back to the LLM via state tokens to generate subsequent text or action tokens. We demonstrate that MultiPLY outperforms baselines by a large margin through a diverse set of embodied tasks involving object retrieval, tool use, multisensory captioning, and task decomposition.

{{</citation>}}


### (16/132) Fixed Point Diffusion Models (Xingjian Bai et al., 2024)

{{<citation>}}

Xingjian Bai, Luke Melas-Kyriazi. (2024)  
**Fixed Point Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.08741v1)  

---


**ABSTRACT**  
We introduce the Fixed Point Diffusion Model (FPDM), a novel approach to image generation that integrates the concept of fixed point solving into the framework of diffusion-based generative modeling. Our approach embeds an implicit fixed point solving layer into the denoising network of a diffusion model, transforming the diffusion process into a sequence of closely-related fixed point problems. Combined with a new stochastic training method, this approach significantly reduces model size, reduces memory usage, and accelerates training. Moreover, it enables the development of two new techniques to improve sampling efficiency: reallocating computation across timesteps and reusing fixed point solutions between timesteps. We conduct extensive experiments with state-of-the-art models on ImageNet, FFHQ, CelebA-HQ, and LSUN-Church, demonstrating substantial improvements in performance and efficiency. Compared to the state-of-the-art DiT model, FPDM contains 87% fewer parameters, consumes 60% less memory during training, and improves image generation quality in situations where sampling computation or time is limited. Our code and pretrained models are available at https://lukemelas.github.io/fixed-point-diffusion-models.

{{</citation>}}


### (17/132) SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers (Nanye Ma et al., 2024)

{{<citation>}}

Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, Saining Xie. (2024)  
**SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.08740v1)  

---


**ABSTRACT**  
We present Scalable Interpolant Transformers (SiT), a family of generative models built on the backbone of Diffusion Transformers (DiT). The interpolant framework, which allows for connecting two distributions in a more flexible way than standard diffusion models, makes possible a modular study of various design choices impacting generative models built on dynamical transport: using discrete vs. continuous time learning, deciding the objective for the model to learn, choosing the interpolant connecting the distributions, and deploying a deterministic or stochastic sampler. By carefully introducing the above ingredients, SiT surpasses DiT uniformly across model sizes on the conditional ImageNet 256x256 benchmark using the exact same backbone, number of parameters, and GFLOPs. By exploring various diffusion coefficients, which can be tuned separately from learning, SiT achieves an FID-50K score of 2.06.

{{</citation>}}


### (18/132) Scalable Pre-training of Large Autoregressive Image Models (Alaaeldin El-Nouby et al., 2024)

{{<citation>}}

Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar, Joshua M Susskind, Armand Joulin. (2024)  
**Scalable Pre-training of Large Autoregressive Image Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, ImageNet, Language Model  
[Paper Link](http://arxiv.org/abs/2401.08541v1)  

---


**ABSTRACT**  
This paper introduces AIM, a collection of vision models pre-trained with an autoregressive objective. These models are inspired by their textual counterparts, i.e., Large Language Models (LLMs), and exhibit similar scaling properties. Specifically, we highlight two key findings: (1) the performance of the visual features scale with both the model capacity and the quantity of data, (2) the value of the objective function correlates with the performance of the model on downstream tasks. We illustrate the practical implication of these findings by pre-training a 7 billion parameter AIM on 2 billion images, that achieves 84.0% on ImageNet-1k with a frozen trunk. Interestingly, even at this scale, we observe no sign of saturation in performance, suggesting that AIM potentially represents a new frontier for training large-scale vision models. The pre-training of AIM is similar to the pre-training of LLMs, and does not require any image-specific strategy to stabilize the training at scale.

{{</citation>}}


### (19/132) MICA: Towards Explainable Skin Lesion Diagnosis via Multi-Level Image-Concept Alignment (Yequan Bie et al., 2024)

{{<citation>}}

Yequan Bie, Luyang Luo, Hao Chen. (2024)  
**MICA: Towards Explainable Skin Lesion Diagnosis via Multi-Level Image-Concept Alignment**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08527v1)  

---


**ABSTRACT**  
Black-box deep learning approaches have showcased significant potential in the realm of medical image analysis. However, the stringent trustworthiness requirements intrinsic to the medical field have catalyzed research into the utilization of Explainable Artificial Intelligence (XAI), with a particular focus on concept-based methods. Existing concept-based methods predominantly apply concept annotations from a single perspective (e.g., global level), neglecting the nuanced semantic relationships between sub-regions and concepts embedded within medical images. This leads to underutilization of the valuable medical information and may cause models to fall short in harmoniously balancing interpretability and performance when employing inherently interpretable architectures such as Concept Bottlenecks. To mitigate these shortcomings, we propose a multi-modal explainable disease diagnosis framework that meticulously aligns medical images and clinical-related concepts semantically at multiple strata, encompassing the image level, token level, and concept level. Moreover, our method allows for model intervention and offers both textual and visual explanations in terms of human-interpretable concepts. Experimental results on three skin image datasets demonstrate that our method, while preserving model interpretability, attains high performance and label efficiency for concept detection and disease diagnosis.

{{</citation>}}


### (20/132) Bag of Tricks to Boost Adversarial Transferability (Zeliang Zhang et al., 2024)

{{<citation>}}

Zeliang Zhang, Rongyi Zhu, Wei Yao, Xiaosen Wang, Chenliang Xu. (2024)  
**Bag of Tricks to Boost Adversarial Transferability**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.08734v1)  

---


**ABSTRACT**  
Deep neural networks are widely known to be vulnerable to adversarial examples. However, vanilla adversarial examples generated under the white-box setting often exhibit low transferability across different models. Since adversarial transferability poses more severe threats to practical applications, various approaches have been proposed for better transferability, including gradient-based, input transformation-based, and model-related attacks, \etc. In this work, we find that several tiny changes in the existing adversarial attacks can significantly affect the attack performance, \eg, the number of iterations and step size. Based on careful studies of existing adversarial attacks, we propose a bag of tricks to enhance adversarial transferability, including momentum initialization, scheduled step size, dual example, spectral-based input transformation, and several ensemble strategies. Extensive experiments on the ImageNet dataset validate the high effectiveness of our proposed tricks and show that combining them can further boost adversarial transferability. Our work provides practical insights and techniques to enhance adversarial transferability, and offers guidance to improve the attack performance on the real-world application through simple adjustments.

{{</citation>}}


### (21/132) Video Quality Assessment Based on Swin TransformerV2 and Coarse to Fine Strategy (Zihao Yu et al., 2024)

{{<citation>}}

Zihao Yu, Fengbin Guan, Yiting Lu, Xin Li, Zhibo Chen. (2024)  
**Video Quality Assessment Based on Swin TransformerV2 and Coarse to Fine Strategy**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: QA, Transformer  
[Paper Link](http://arxiv.org/abs/2401.08522v1)  

---


**ABSTRACT**  
The objective of non-reference video quality assessment is to evaluate the quality of distorted video without access to reference high-definition references. In this study, we introduce an enhanced spatial perception module, pre-trained on multiple image quality assessment datasets, and a lightweight temporal fusion module to address the no-reference visual quality assessment (NR-VQA) task. This model implements Swin Transformer V2 as a local-level spatial feature extractor and fuses these multi-stage representations through a series of transformer layers. Furthermore, a temporal transformer is utilized for spatiotemporal feature fusion across the video. To accommodate compressed videos of varying bitrates, we incorporate a coarse-to-fine contrastive strategy to enrich the model's capability to discriminate features from videos of different bitrates. This is an expanded version of the one-page abstract.

{{</citation>}}


### (22/132) ValUES: A Framework for Systematic Validation of Uncertainty Estimation in Semantic Segmentation (Kim-Celine Kahl et al., 2024)

{{<citation>}}

Kim-Celine Kahl, Carsten T. Lüth, Maximilian Zenk, Klaus Maier-Hein, Paul F. Jaeger. (2024)  
**ValUES: A Framework for Systematic Validation of Uncertainty Estimation in Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.08501v1)  

---


**ABSTRACT**  
Uncertainty estimation is an essential and heavily-studied component for the reliable application of semantic segmentation methods. While various studies exist claiming methodological advances on the one hand, and successful application on the other hand, the field is currently hampered by a gap between theory and practice leaving fundamental questions unanswered: Can data-related and model-related uncertainty really be separated in practice? Which components of an uncertainty method are essential for real-world performance? Which uncertainty method works well for which application? In this work, we link this research gap to a lack of systematic and comprehensive evaluation of uncertainty methods. Specifically, we identify three key pitfalls in current literature and present an evaluation framework that bridges the research gap by providing 1) a controlled environment for studying data ambiguities as well as distribution shifts, 2) systematic ablations of relevant method components, and 3) test-beds for the five predominant uncertainty applications: OoD-detection, active learning, failure detection, calibration, and ambiguity modeling. Empirical results on simulated as well as real-world data demonstrate how the proposed framework is able to answer the predominant questions in the field revealing for instance that 1) separation of uncertainty types works on simulated data but does not necessarily translate to real-world data, 2) aggregation of scores is a crucial but currently neglected component of uncertainty methods, 3) While ensembles are performing most robustly across the different downstream tasks and settings, test-time augmentation often constitutes a light-weight alternative. Code is at: https://github.com/IML-DKFZ/values

{{</citation>}}


### (23/132) Improving Limited Supervised Foot Ulcer Segmentation Using Cross-Domain Augmentation (Shang-Jui Kuo et al., 2024)

{{<citation>}}

Shang-Jui Kuo, Po-Han Huang, Chia-Ching Lin, Jeng-Lin Li, Ming-Ching Chang. (2024)  
**Improving Limited Supervised Foot Ulcer Segmentation Using Cross-Domain Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2401.08422v1)  

---


**ABSTRACT**  
Diabetic foot ulcers pose health risks, including higher morbidity, mortality, and amputation rates. Monitoring wound areas is crucial for proper care, but manual segmentation is subjective due to complex wound features and background variation. Expert annotations are costly and time-intensive, thus hampering large dataset creation. Existing segmentation models relying on extensive annotations are impractical in real-world scenarios with limited annotated data. In this paper, we propose a cross-domain augmentation method named TransMix that combines Augmented Global Pre-training AGP and Localized CutMix Fine-tuning LCF to enrich wound segmentation data for model learning. TransMix can effectively improve the foot ulcer segmentation model training by leveraging other dermatology datasets not on ulcer skins or wounds. AGP effectively increases the overall image variability, while LCF increases the diversity of wound regions. Experimental results show that TransMix increases the variability of wound regions and substantially improves the Dice score for models trained with only 40 annotated images under various proportions.

{{</citation>}}


### (24/132) Cross-Domain Few-Shot Segmentation via Iterative Support-Query Correspondence Mining (Jiahao Nie et al., 2024)

{{<citation>}}

Jiahao Nie, Yun Xing, Gongjie Zhang, Pei Yan, Aoran Xiao, Yap-Peng Tan, Alex C. Kot, Shijian Lu. (2024)  
**Cross-Domain Few-Shot Segmentation via Iterative Support-Query Correspondence Mining**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2401.08407v1)  

---


**ABSTRACT**  
Cross-Domain Few-Shot Segmentation (CD-FSS) poses the challenge of segmenting novel categories from a distinct domain using only limited exemplars. In this paper, we undertake a comprehensive study of CD-FSS and uncover two crucial insights: (i) the necessity of a fine-tuning stage to effectively transfer the learned meta-knowledge across domains, and (ii) the overfitting risk during the na\"ive fine-tuning due to the scarcity of novel category examples. With these insights, we propose a novel cross-domain fine-tuning strategy that addresses the challenging CD-FSS tasks. We first design Bi-directional Few-shot Prediction (BFP), which establishes support-query correspondence in a bi-directional manner, crafting augmented supervision to reduce the overfitting risk. Then we further extend BFP into Iterative Few-shot Adaptor (IFA), which is a recursive framework to capture the support-query correspondence iteratively, targeting maximal exploitation of supervisory signals from the sparse novel category samples. Extensive empirical evaluations show that our method significantly outperforms the state-of-the-arts (+7.8\%), which verifies that IFA tackles the cross-domain challenges and mitigates the overfitting simultaneously. Code will be made available.

{{</citation>}}


### (25/132) Hidden Flaws Behind Expert-Level Accuracy of GPT-4 Vision in Medicine (Qiao Jin et al., 2024)

{{<citation>}}

Qiao Jin, Fangyuan Chen, Yiliang Zhou, Ziyang Xu, Justin M. Cheung, Robert Chen, Ronald M. Summers, Justin F. Rousseau, Peiyun Ni, Marc J Landsman, Sally L. Baxter, Subhi J. Al'Aref, Yijia Li, Michael F. Chiang, Yifan Peng, Zhiyong Lu. (2024)  
**Hidden Flaws Behind Expert-Level Accuracy of GPT-4 Vision in Medicine**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-4, Transformer  
[Paper Link](http://arxiv.org/abs/2401.08396v1)  

---


**ABSTRACT**  
Recent studies indicate that Generative Pre-trained Transformer 4 with Vision (GPT-4V) outperforms human physicians in medical challenge tasks. However, these evaluations primarily focused on the accuracy of multi-choice questions alone. Our study extends the current scope by conducting a comprehensive analysis of GPT-4V's rationales of image comprehension, recall of medical knowledge, and step-by-step multimodal reasoning when solving New England Journal of Medicine (NEJM) Image Challenges - an imaging quiz designed to test the knowledge and diagnostic capabilities of medical professionals. Evaluation results confirmed that GPT-4V outperforms human physicians regarding multi-choice accuracy (88.0% vs. 77.0%, p=0.034). GPT-4V also performs well in cases where physicians incorrectly answer, with over 80% accuracy. However, we discovered that GPT-4V frequently presents flawed rationales in cases where it makes the correct final choices (27.3%), most prominent in image comprehension (21.6%). Regardless of GPT-4V's high accuracy in multi-choice questions, our findings emphasize the necessity for further in-depth evaluations of its rationales before integrating such models into clinical workflows.

{{</citation>}}


### (26/132) DoraemonGPT: Toward Understanding Dynamic Scenes with Large Language Models (Zongxin Yang et al., 2024)

{{<citation>}}

Zongxin Yang, Guikun Chen, Xiaodi Li, Wenguan Wang, Yi Yang. (2024)  
**DoraemonGPT: Toward Understanding Dynamic Scenes with Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.08392v1)  

---


**ABSTRACT**  
The field of AI agents is advancing at an unprecedented rate due to the capabilities of large language models (LLMs). However, LLM-driven visual agents mainly focus on solving tasks for the image modality, which limits their ability to understand the dynamic nature of the real world, making it still far from real-life applications, e.g., guiding students in laboratory experiments and identifying their mistakes. Considering the video modality better reflects the ever-changing and perceptually intensive nature of real-world scenarios, we devise DoraemonGPT, a comprehensive and conceptually elegant system driven by LLMs to handle dynamic video tasks. Given a video with a question/task, DoraemonGPT begins by converting the input video with massive content into a symbolic memory that stores \textit{task-related} attributes. This structured representation allows for spatial-temporal querying and reasoning by sub-task tools, resulting in concise and relevant intermediate results. Recognizing that LLMs have limited internal knowledge when it comes to specialized domains (e.g., analyzing the scientific principles underlying experiments), we incorporate plug-and-play tools to assess external knowledge and address tasks across different domains. Moreover, we introduce a novel LLM-driven planner based on Monte Carlo Tree Search to efficiently explore the large planning space for scheduling various tools. The planner iteratively finds feasible solutions by backpropagating the result's reward, and multiple solutions can be summarized into an improved final answer. We extensively evaluate DoraemonGPT in dynamic scenes and provide in-the-wild showcases demonstrating its ability to handle more complex questions than previous studies.

{{</citation>}}


### (27/132) SAMF: Small-Area-Aware Multi-focus Image Fusion for Object Detection (Xilai Li et al., 2024)

{{<citation>}}

Xilai Li, Xiaosong Li, Haishu Tan, Jinyang Li. (2024)  
**SAMF: Small-Area-Aware Multi-focus Image Fusion for Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.08357v1)  

---


**ABSTRACT**  
Existing multi-focus image fusion (MFIF) methods often fail to preserve the uncertain transition region and detect small focus areas within large defocused regions accurately. To address this issue, this study proposes a new small-area-aware MFIF algorithm for enhancing object detection capability. First, we enhance the pixel attributes within the small focus and boundary regions, which are subsequently combined with visual saliency detection to obtain the pre-fusion results used to discriminate the distribution of focused pixels. To accurately ensure pixel focus, we consider the source image as a combination of focused, defocused, and uncertain regions and propose a three-region segmentation strategy. Finally, we design an effective pixel selection rule to generate segmentation decision maps and obtain the final fusion results. Experiments demonstrated that the proposed method can accurately detect small and smooth focus areas while improving object detection performance, outperforming existing methods in both subjective and objective evaluations. The source code is available at https://github.com/ixilai/SAMF.

{{</citation>}}


### (28/132) AesBench: An Expert Benchmark for Multimodal Large Language Models on Image Aesthetics Perception (Yipo Huang et al., 2024)

{{<citation>}}

Yipo Huang, Quan Yuan, Xiangfei Sheng, Zhichao Yang, Haoning Wu, Pengfei Chen, Yuzhe Yang, Leida Li, Weisi Lin. (2024)  
**AesBench: An Expert Benchmark for Multimodal Large Language Models on Image Aesthetics Perception**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.08276v1)  

---


**ABSTRACT**  
With collective endeavors, multimodal large language models (MLLMs) are undergoing a flourishing development. However, their performances on image aesthetics perception remain indeterminate, which is highly desired in real-world applications. An obvious obstacle lies in the absence of a specific benchmark to evaluate the effectiveness of MLLMs on aesthetic perception. This blind groping may impede the further development of more advanced MLLMs with aesthetic perception capacity. To address this dilemma, we propose AesBench, an expert benchmark aiming to comprehensively evaluate the aesthetic perception capacities of MLLMs through elaborate design across dual facets. (1) We construct an Expert-labeled Aesthetics Perception Database (EAPD), which features diversified image contents and high-quality annotations provided by professional aesthetic experts. (2) We propose a set of integrative criteria to measure the aesthetic perception abilities of MLLMs from four perspectives, including Perception (AesP), Empathy (AesE), Assessment (AesA) and Interpretation (AesI). Extensive experimental results underscore that the current MLLMs only possess rudimentary aesthetic perception ability, and there is still a significant gap between MLLMs and humans. We hope this work can inspire the community to engage in deeper explorations on the aesthetic potentials of MLLMs. Source data will be available at https://github.com/yipoh/AesBench.

{{</citation>}}


### (29/132) Human vs. LMMs: Exploring the Discrepancy in Emoji Interpretation and Usage in Digital Communication (Hanjia Lyu et al., 2024)

{{<citation>}}

Hanjia Lyu, Weihong Qi, Zhongyu Wei, Jiebo Luo. (2024)  
**Human vs. LMMs: Exploring the Discrepancy in Emoji Interpretation and Usage in Digital Communication**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.08212v1)  

---


**ABSTRACT**  
Leveraging Large Multimodal Models (LMMs) to simulate human behaviors when processing multimodal information, especially in the context of social media, has garnered immense interest due to its broad potential and far-reaching implications. Emojis, as one of the most unique aspects of digital communication, are pivotal in enriching and often clarifying the emotional and tonal dimensions. Yet, there is a notable gap in understanding how these advanced models, such as GPT-4V, interpret and employ emojis in the nuanced context of online interaction. This study intends to bridge this gap by examining the behavior of GPT-4V in replicating human-like use of emojis. The findings reveal a discernible discrepancy between human and GPT-4V behaviors, likely due to the subjective nature of human interpretation and the limitations of GPT-4V's English-centric training, suggesting cultural biases and inadequate representation of non-English cultures.

{{</citation>}}


### (30/132) Transcending the Limit of Local Window: Advanced Super-Resolution Transformer with Adaptive Token Dictionary (Leheng Zhang et al., 2024)

{{<citation>}}

Leheng Zhang, Yawei Li, Xingyu Zhou, Xiaorui Zhao, Shuhang Gu. (2024)  
**Transcending the Limit of Local Window: Advanced Super-Resolution Transformer with Adaptive Token Dictionary**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.08209v2)  

---


**ABSTRACT**  
Single Image Super-Resolution is a classic computer vision problem that involves estimating high-resolution (HR) images from low-resolution (LR) ones. Although deep neural networks (DNNs), especially Transformers for super-resolution, have seen significant advancements in recent years, challenges still remain, particularly in limited receptive field caused by window-based self-attention. To address these issues, we introduce a group of auxiliary Adaptive Token Dictionary to SR Transformer and establish an ATD-SR method. The introduced token dictionary could learn prior information from training data and adapt the learned prior to specific testing image through an adaptive refinement step. The refinement strategy could not only provide global information to all input tokens but also group image tokens into categories. Based on category partitions, we further propose a category-based self-attention mechanism designed to leverage distant but similar tokens for enhancing input features. The experimental results show that our method achieves the best performance on various single image super-resolution benchmarks.

{{</citation>}}


### (31/132) DPAFNet:Dual Path Attention Fusion Network for Single Image Deraining (Bingcai Wei, 2024)

{{<citation>}}

Bingcai Wei. (2024)  
**DPAFNet:Dual Path Attention Fusion Network for Single Image Deraining**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV, eess-IV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.08185v1)  

---


**ABSTRACT**  
Rainy weather will have a significant impact on the regular operation of the imaging system. Based on this premise, image rain removal has always been a popular branch of low-level visual tasks, especially methods using deep neural networks. However, most neural networks are but-branched, such as only using convolutional neural networks or Transformers, which is unfavourable for the multidimensional fusion of image features. In order to solve this problem, this paper proposes a dual-branch attention fusion network. Firstly, a two-branch network structure is proposed. Secondly, an attention fusion module is proposed to selectively fuse the features extracted by the two branches rather than simply adding them. Finally, complete ablation experiments and sufficient comparison experiments prove the rationality and effectiveness of the proposed method.

{{</citation>}}


### (32/132) Deep Linear Array Pushbroom Image Restoration: A Degradation Pipeline and Jitter-Aware Restoration Network (Zida Chen et al., 2024)

{{<citation>}}

Zida Chen, Ziran Zhang, Haoying Li, Menghao Li, Yueting Chen, Qi Li, Huajun Feng, Zhihai Xu, Shiqi Chen. (2024)  
**Deep Linear Array Pushbroom Image Restoration: A Degradation Pipeline and Jitter-Aware Restoration Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.08171v1)  

---


**ABSTRACT**  
Linear Array Pushbroom (LAP) imaging technology is widely used in the realm of remote sensing. However, images acquired through LAP always suffer from distortion and blur because of camera jitter. Traditional methods for restoring LAP images, such as algorithms estimating the point spread function (PSF), exhibit limited performance. To tackle this issue, we propose a Jitter-Aware Restoration Network (JARNet), to remove the distortion and blur in two stages. In the first stage, we formulate an Optical Flow Correction (OFC) block to refine the optical flow of the degraded LAP images, resulting in pre-corrected images where most of the distortions are alleviated. In the second stage, for further enhancement of the pre-corrected images, we integrate two jitter-aware techniques within the Spatial and Frequency Residual (SFRes) block: 1) introducing Coordinate Attention (CoA) to the SFRes block in order to capture the jitter state in orthogonal direction; 2) manipulating image features in both spatial and frequency domains to leverage local and global priors. Additionally, we develop a data synthesis pipeline, which applies Continue Dynamic Shooting Model (CDSM) to simulate realistic degradation in LAP images. Both the proposed JARNet and LAP image synthesis pipeline establish a foundation for addressing this intricate challenge. Extensive experiments demonstrate that the proposed two-stage method outperforms state-of-the-art image restoration models. Code is available at https://github.com/JHW2000/JARNet.

{{</citation>}}


### (33/132) Mobile Contactless Palmprint Recognition: Use of Multiscale, Multimodel Embeddings (Steven A. Grosz et al., 2024)

{{<citation>}}

Steven A. Grosz, Akash Godbole, Anil K. Jain. (2024)  
**Mobile Contactless Palmprint Recognition: Use of Multiscale, Multimodel Embeddings**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.08111v1)  

---


**ABSTRACT**  
Contactless palmprints are comprised of both global and local discriminative features. Most prior work focuses on extracting global features or local features alone for palmprint matching, whereas this research introduces a novel framework that combines global and local features for enhanced palmprint matching accuracy. Leveraging recent advancements in deep learning, this study integrates a vision transformer (ViT) and a convolutional neural network (CNN) to extract complementary local and global features. Next, a mobile-based, end-to-end palmprint recognition system is developed, referred to as Palm-ID. On top of the ViT and CNN features, Palm-ID incorporates a palmprint enhancement module and efficient dimensionality reduction (for faster matching). Palm-ID balances the trade-off between accuracy and latency, requiring just 18ms to extract a template of size 516 bytes, which can be efficiently searched against a 10,000 palmprint gallery in 0.33ms on an AMD EPYC 7543 32-Core CPU utilizing 128-threads. Cross-database matching protocols and evaluations on large-scale operational datasets demonstrate the robustness of the proposed method, achieving a TAR of 98.06% at FAR=0.01% on a newly collected, time-separated dataset. To show a practical deployment of the end-to-end system, the entire recognition pipeline is embedded within a mobile device for enhanced user privacy and security.

{{</citation>}}


### (34/132) Deep Shape-Texture Statistics for Completely Blind Image Quality Evaluation (Yixuan Li et al., 2024)

{{<citation>}}

Yixuan Li, Peilin Chen, Hanwei Zhu, Keyan Ding, Leida Li, Shiqi Wang. (2024)  
**Deep Shape-Texture Statistics for Completely Blind Image Quality Evaluation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.08107v1)  

---


**ABSTRACT**  
Opinion-Unaware Blind Image Quality Assessment (OU-BIQA) models aim to predict image quality without training on reference images and subjective quality scores. Thereinto, image statistical comparison is a classic paradigm, while the performance is limited by the representation ability of visual descriptors. Deep features as visual descriptors have advanced IQA in recent research, but they are discovered to be highly texture-biased and lack of shape-bias. On this basis, we find out that image shape and texture cues respond differently towards distortions, and the absence of either one results in an incomplete image representation. Therefore, to formulate a well-round statistical description for images, we utilize the shapebiased and texture-biased deep features produced by Deep Neural Networks (DNNs) simultaneously. More specifically, we design a Shape-Texture Adaptive Fusion (STAF) module to merge shape and texture information, based on which we formulate qualityrelevant image statistics. The perceptual quality is quantified by the variant Mahalanobis Distance between the inner and outer Shape-Texture Statistics (DSTS), wherein the inner and outer statistics respectively describe the quality fingerprints of the distorted image and natural images. The proposed DSTS delicately utilizes shape-texture statistical relations between different data scales in the deep domain, and achieves state-of-the-art (SOTA) quality prediction performance on images with artificial and authentic distortions.

{{</citation>}}


### (35/132) Hardware Acceleration for Real-Time Wildfire Detection Onboard Drone Networks (Austin Briley et al., 2024)

{{<citation>}}

Austin Briley, Fatemeh Afghah. (2024)  
**Hardware Acceleration for Real-Time Wildfire Detection Onboard Drone Networks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV, eess-IV  
Keywords: Drone, QA, Quantization  
[Paper Link](http://arxiv.org/abs/2401.08105v1)  

---


**ABSTRACT**  
Early wildfire detection in remote and forest areas is crucial for minimizing devastation and preserving ecosystems. Autonomous drones offer agile access to remote, challenging terrains, equipped with advanced imaging technology that delivers both high-temporal and detailed spatial resolution, making them valuable assets in the early detection and monitoring of wildfires. However, the limited computation and battery resources of Unmanned Aerial Vehicles (UAVs) pose significant challenges in implementing robust and efficient image classification models. Current works in this domain often operate offline, emphasizing the need for solutions that can perform inference in real time, given the constraints of UAVs. To address these challenges, this paper aims to develop a real-time image classification and fire segmentation model. It presents a comprehensive investigation into hardware acceleration using the Jetson Nano P3450 and the implications of TensorRT, NVIDIA's high-performance deep-learning inference library, on fire classification accuracy and speed. The study includes implementations of Quantization Aware Training (QAT), Automatic Mixed Precision (AMP), and post-training mechanisms, comparing them against the latest baselines for fire segmentation and classification. All experiments utilize the FLAME dataset - an image dataset collected by low-altitude drones during a prescribed forest fire. This work contributes to the ongoing efforts to enable real-time, on-board wildfire detection capabilities for UAVs, addressing speed and the computational and energy constraints of these crucial monitoring systems. The results show a 13% increase in classification speed compared to similar models without hardware optimization. Comparatively, loss and accuracy are within 1.225% of the original values.

{{</citation>}}


### (36/132) KTVIC: A Vietnamese Image Captioning Dataset on the Life Domain (Anh-Cuong Pham et al., 2024)

{{<citation>}}

Anh-Cuong Pham, Van-Quang Nguyen, Thi-Hong Vuong, Quang-Thuy Ha. (2024)  
**KTVIC: A Vietnamese Image Captioning Dataset on the Life Domain**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: BLEU, Image Captioning  
[Paper Link](http://arxiv.org/abs/2401.08100v1)  

---


**ABSTRACT**  
Image captioning is a crucial task with applications in a wide range of domains, including healthcare and education. Despite extensive research on English image captioning datasets, the availability of such datasets for Vietnamese remains limited, with only two existing datasets. In this study, we introduce KTVIC, a comprehensive Vietnamese Image Captioning dataset focused on the life domain, covering a wide range of daily activities. This dataset comprises 4,327 images and 21,635 Vietnamese captions, serving as a valuable resource for advancing image captioning in the Vietnamese language. We conduct experiments using various deep neural networks as the baselines on our dataset, evaluating them using the standard image captioning metrics, including BLEU, METEOR, CIDEr, and ROUGE. Our findings underscore the effectiveness of the proposed dataset and its potential contributions to the field of image captioning in the Vietnamese context.

{{</citation>}}


### (37/132) Adversarial Masking Contrastive Learning for vein recognition (Huafeng Qin et al., 2024)

{{<citation>}}

Huafeng Qin, Yiquan Wu, Mounim A. El-Yacoubi, Jun Wang, Guangxiang Yang. (2024)  
**Adversarial Masking Contrastive Learning for vein recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.08079v1)  

---


**ABSTRACT**  
Vein recognition has received increasing attention due to its high security and privacy. Recently, deep neural networks such as Convolutional neural networks (CNN) and Transformers have been introduced for vein recognition and achieved state-of-the-art performance. Despite the recent advances, however, existing solutions for finger-vein feature extraction are still not optimal due to scarce training image samples. To overcome this problem, in this paper, we propose an adversarial masking contrastive learning (AMCL) approach, that generates challenging samples to train a more robust contrastive learning model for the downstream palm-vein recognition task, by alternatively optimizing the encoder in the contrastive learning model and a set of latent variables. First, a huge number of masks are generated to train a robust generative adversarial network (GAN). The trained generator transforms a latent variable from the latent variable space into a mask space. Then, we combine the trained generator with a contrastive learning model to obtain our AMCL, where the generator produces challenging masking images to increase the contrastive loss and the contrastive learning model is trained based on the harder images to learn a more robust feature representation. After training, the trained encoder in the contrastive learning model is combined with a classification layer to build a classifier, which is further fine-tuned on labeled training data for vein recognition. The experimental results on three databases demonstrate that our approach outperforms existing contrastive learning approaches in terms of improving identification accuracy of vein classifiers and achieves state-of-the-art recognition results.

{{</citation>}}


### (38/132) Representation Learning on Event Stream via an Elastic Net-incorporated Tensor Network (Beibei Yang et al., 2024)

{{<citation>}}

Beibei Yang, Weiling Li, Yan Fang. (2024)  
**Representation Learning on Event Stream via an Elastic Net-incorporated Tensor Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.08068v1)  

---


**ABSTRACT**  
Event cameras are neuromorphic sensors that capture asynchronous and sparse event stream when per-pixel brightness changes. The state-of-the-art processing methods for event signals typically aggregate events into a frame or a grid. However, events are dense in time, these works are limited to local information of events due to the stacking. In this paper, we present a novel spatiotemporal representation learning method which can capture the global correlations of all events in the event stream simultaneously by tensor decomposition. In addition, with the events are sparse in space, we propose an Elastic Net-incorporated tensor network (ENTN) model to obtain more spatial and temporal details about event stream. Empirically, the results indicate that our method can represent the spatiotemporal correlation of events with high quality, and can achieve effective results in applications like filtering noise compared with the state-of-the-art methods.

{{</citation>}}


### (39/132) Achieve Fairness without Demographics for Dermatological Disease Diagnosis (Ching-Hao Chiu et al., 2024)

{{<citation>}}

Ching-Hao Chiu, Yu-Jen Chen, Yawen Wu, Yiyu Shi, Tsung-Yi Ho. (2024)  
**Achieve Fairness without Demographics for Dermatological Disease Diagnosis**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08066v1)  

---


**ABSTRACT**  
In medical image diagnosis, fairness has become increasingly crucial. Without bias mitigation, deploying unfair AI would harm the interests of the underprivileged population and potentially tear society apart. Recent research addresses prediction biases in deep learning models concerning demographic groups (e.g., gender, age, and race) by utilizing demographic (sensitive attribute) information during training. However, many sensitive attributes naturally exist in dermatological disease images. If the trained model only targets fairness for a specific attribute, it remains unfair for other attributes. Moreover, training a model that can accommodate multiple sensitive attributes is impractical due to privacy concerns. To overcome this, we propose a method enabling fair predictions for sensitive attributes during the testing phase without using such information during training. Inspired by prior work highlighting the impact of feature entanglement on fairness, we enhance the model features by capturing the features related to the sensitive and target attributes and regularizing the feature entanglement between corresponding classes. This ensures that the model can only classify based on the features related to the target attribute without relying on features associated with sensitive attributes, thereby improving fairness and accuracy. Additionally, we use disease masks from the Segment Anything Model (SAM) to enhance the quality of the learned feature. Experimental results demonstrate that the proposed method can improve fairness in classification compared to state-of-the-art methods in two dermatological disease datasets.

{{</citation>}}


### (40/132) Toward Clinically Trustworthy Deep Learning: Applying Conformal Prediction to Intracranial Hemorrhage Detection (Cooper Gamble et al., 2024)

{{<citation>}}

Cooper Gamble, Shahriar Faghani, Bradley J. Erickson. (2024)  
**Toward Clinically Trustworthy Deep Learning: Applying Conformal Prediction to Intracranial Hemorrhage Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2401.08058v1)  

---


**ABSTRACT**  
As deep learning (DL) continues to demonstrate its ability in radiological tasks, it is critical that we optimize clinical DL solutions to include safety. One of the principal concerns in the clinical adoption of DL tools is trust. This study aims to apply conformal prediction as a step toward trustworthiness for DL in radiology. This is a retrospective study of 491 non-contrast head CTs from the CQ500 dataset, in which three senior radiologists annotated slices containing intracranial hemorrhage (ICH). The dataset was split into definite and challenging subsets, where challenging images were defined to those in which there was disagreement among readers. A DL model was trained on 146 patients (10,815 slices) from the definite data (training dataset) to perform ICH localization and classification for five classes of ICH. To develop an uncertainty-aware DL model, 1,546 cases of the definite data (calibration dataset) was used for Mondrian conformal prediction (MCP). The uncertainty-aware DL model was tested on 8,401 definite and challenging cases to assess its ability to identify challenging cases. After the MCP procedure, the model achieved an F1 score of 0.920 for ICH classification on the test dataset. Additionally, it correctly identified 6,837 of the 6,856 total challenging cases as challenging (99.7% accuracy). It did not incorrectly label any definite cases as challenging. The uncertainty-aware ICH detector performs on par with state-of-the-art models. MCP's performance in detecting challenging cases demonstrates that it is useful in automated ICH detection and promising for trustworthiness in radiological DL.

{{</citation>}}


### (41/132) Robust Tiny Object Detection in Aerial Images amidst Label Noise (Haoran Zhu et al., 2024)

{{<citation>}}

Haoran Zhu, Chang Xu, Wen Yang, Ruixiang Zhang, Yan Zhang, Gui-Song Xia. (2024)  
**Robust Tiny Object Detection in Aerial Images amidst Label Noise**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Object Detection  
[Paper Link](http://arxiv.org/abs/2401.08056v1)  

---


**ABSTRACT**  
Precise detection of tiny objects in remote sensing imagery remains a significant challenge due to their limited visual information and frequent occurrence within scenes. This challenge is further exacerbated by the practical burden and inherent errors associated with manual annotation: annotating tiny objects is laborious and prone to errors (i.e., label noise). Training detectors for such objects using noisy labels often leads to suboptimal performance, with networks tending to overfit on noisy labels. In this study, we address the intricate issue of tiny object detection under noisy label supervision. We systematically investigate the impact of various types of noise on network training, revealing the vulnerability of object detectors to class shifts and inaccurate bounding boxes for tiny objects. To mitigate these challenges, we propose a DeNoising Tiny Object Detector (DN-TOD), which incorporates a Class-aware Label Correction (CLC) scheme to address class shifts and a Trend-guided Learning Strategy (TLS) to handle bounding box noise. CLC mitigates inaccurate class supervision by identifying and filtering out class-shifted positive samples, while TLS reduces noisy box-induced erroneous supervision through sample reweighting and bounding box regeneration. Additionally, Our method can be seamlessly integrated into both one-stage and two-stage object detection pipelines. Comprehensive experiments conducted on synthetic (i.e., noisy AI-TOD-v2.0 and DOTA-v2.0) and real-world (i.e., AI-TOD) noisy datasets demonstrate the robustness of DN-TOD under various types of label noise. Notably, when applied to the strong baseline RFLA, DN-TOD exhibits a noteworthy performance improvement of 4.9 points under 40% mixed noise. Datasets, codes, and models will be made publicly available.

{{</citation>}}


### (42/132) SCoFT: Self-Contrastive Fine-Tuning for Equitable Image Generation (Zhixuan Liu et al., 2024)

{{<citation>}}

Zhixuan Liu, Peter Schaldenbrand, Beverley-Claire Okogwu, Wenxuan Peng, Youngsik Yun, Andrew Hundt, Jihie Kim, Jean Oh. (2024)  
**SCoFT: Self-Contrastive Fine-Tuning for Equitable Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08053v1)  

---


**ABSTRACT**  
Accurate representation in media is known to improve the well-being of the people who consume it. Generative image models trained on large web-crawled datasets such as LAION are known to produce images with harmful stereotypes and misrepresentations of cultures. We improve inclusive representation in generated images by (1) engaging with communities to collect a culturally representative dataset that we call the Cross-Cultural Understanding Benchmark (CCUB) and (2) proposing a novel Self-Contrastive Fine-Tuning (SCoFT) method that leverages the model's known biases to self-improve. SCoFT is designed to prevent overfitting on small datasets, encode only high-level information from the data, and shift the generated distribution away from misrepresentations encoded in a pretrained model. Our user study conducted on 51 participants from 5 different countries based on their self-selected national cultural affiliation shows that fine-tuning on CCUB consistently generates images with higher cultural relevance and fewer stereotypes when compared to the Stable Diffusion baseline, which is further improved with our SCoFT technique.

{{</citation>}}


### (43/132) Forging Vision Foundation Models for Autonomous Driving: Challenges, Methodologies, and Opportunities (Xu Yan et al., 2024)

{{<citation>}}

Xu Yan, Haiming Zhang, Yingjie Cai, Jingming Guo, Weichao Qiu, Bin Gao, Kaiqiang Zhou, Yue Zhao, Huan Jin, Jiantao Gao, Zhen Li, Lihui Jiang, Wei Zhang, Hongbo Zhang, Dengxin Dai, Bingbing Liu. (2024)  
**Forging Vision Foundation Models for Autonomous Driving: Challenges, Methodologies, and Opportunities**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.08045v1)  

---


**ABSTRACT**  
The rise of large foundation models, trained on extensive datasets, is revolutionizing the field of AI. Models such as SAM, DALL-E2, and GPT-4 showcase their adaptability by extracting intricate patterns and performing effectively across diverse tasks, thereby serving as potent building blocks for a wide range of AI applications. Autonomous driving, a vibrant front in AI applications, remains challenged by the lack of dedicated vision foundation models (VFMs). The scarcity of comprehensive training data, the need for multi-sensor integration, and the diverse task-specific architectures pose significant obstacles to the development of VFMs in this field. This paper delves into the critical challenge of forging VFMs tailored specifically for autonomous driving, while also outlining future directions. Through a systematic analysis of over 250 papers, we dissect essential techniques for VFM development, including data preparation, pre-training strategies, and downstream task adaptation. Moreover, we explore key advancements such as NeRF, diffusion models, 3D Gaussian Splatting, and world models, presenting a comprehensive roadmap for future research. To empower researchers, we have built and maintained https://github.com/zhanghm1995/Forge_VFM4AD, an open-access repository constantly updated with the latest advancements in forging VFMs for autonomous driving.

{{</citation>}}


### (44/132) Small Object Detection by DETR via Information Augmentation and Adaptive Feature Fusion (Ji Huang et al., 2024)

{{<citation>}}

Ji Huang, Hui Wang. (2024)  
**Small Object Detection by DETR via Information Augmentation and Adaptive Feature Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2401.08017v1)  

---


**ABSTRACT**  
The main challenge for small object detection algorithms is to ensure accuracy while pursuing real-time performance. The RT-DETR model performs well in real-time object detection, but performs poorly in small object detection accuracy. In order to compensate for the shortcomings of the RT-DETR model in small object detection, two key improvements are proposed in this study. Firstly, The RT-DETR utilises a Transformer that receives input solely from the final layer of Backbone features. This means that the Transformer's input only receives semantic information from the highest level of abstraction in the Deep Network, and ignores detailed information such as edges, texture or color gradients that are critical to the location of small objects at lower levels of abstraction. Including only deep features can introduce additional background noise. This can have a negative impact on the accuracy of small object detection. To address this issue, we propose the fine-grained path augmentation method. This method helps to locate small objects more accurately by providing detailed information to the deep network. So, the input to the transformer contains both semantic and detailed information. Secondly, In RT-DETR, the decoder takes feature maps of different levels as input after concatenating them with equal weight. However, this operation is not effective in dealing with the complex relationship of multi-scale information captured by feature maps of different sizes. Therefore, we propose an adaptive feature fusion algorithm that assigns learnable parameters to each feature map from different levels. This allows the model to adaptively fuse feature maps from different levels and effectively integrate feature information from different scales. This enhances the model's ability to capture object features at different scales, thereby improving the accuracy of detecting small objects.

{{</citation>}}


## cs.CY (3)



### (45/132) Foundation Models in Augmentative and Alternative Communication: Opportunities and Challenges (Ambra Di Paola et al., 2024)

{{<citation>}}

Ambra Di Paola, Serena Muraro, Roberto Marinelli, Christian Pilato. (2024)  
**Foundation Models in Augmentative and Alternative Communication: Opportunities and Challenges**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.08866v1)  

---


**ABSTRACT**  
Augmentative and Alternative Communication (AAC) are essential techniques that help people with communication disabilities. AAC demonstrates its transformative power by replacing spoken language with symbol sequences. However, to unlock its full potential, AAC materials must adhere to specific characteristics, placing the onus on educators to create custom-tailored materials and symbols. This paper introduces AMBRA (Pervasive and Personalized Augmentative and Alternative Communication based on Federated Learning and Generative AI), an open platform that aims to leverage the capabilities of foundation models to tackle many AAC issues, opening new opportunities (but also challenges) for AI-enhanced AAC. We thus present a compelling vision--a roadmap towards a more inclusive society. By leveraging the capabilities of modern technologies, we aspire to not only transform AAC but also guide the way toward a world where communication knows no bounds.

{{</citation>}}


### (46/132) The illusion of artificial inclusion (William Agnew et al., 2024)

{{<citation>}}

William Agnew, A. Stevie Bergman, Jennifer Chien, Mark Díaz, Seliem El-Sayed, Jaylen Pittman, Shakir Mohamed, Kevin R. McKee. (2024)  
**The illusion of artificial inclusion**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08572v1)  

---


**ABSTRACT**  
Human participants play a central role in the development of modern artificial intelligence (AI) technology, in psychological science, and in user research. Recent advances in generative AI have attracted growing interest to the possibility of replacing human participants in these domains with AI surrogates. We survey several such "substitution proposals" to better understand the arguments for and against substituting human participants with modern generative AI. Our scoping review indicates that the recent wave of these proposals is motivated by goals such as reducing the costs of research and development work and increasing the diversity of collected data. However, these proposals ignore and ultimately conflict with foundational values of work with human participants: representation, inclusion, and understanding. This paper critically examines the principles and goals underlying human participation to help chart out paths for future work that truly centers and empowers participants.

{{</citation>}}


### (47/132) Resolving Ethics Trade-offs in Implementing Responsible AI (Conrad Sanderson et al., 2024)

{{<citation>}}

Conrad Sanderson, Emma Schleiger, David Douglas, Petra Kuhnert, Qinghua Lu. (2024)  
**Resolving Ethics Trade-offs in Implementing Responsible AI**  

---
Primary Category: cs.CY  
Categories: K-4-1, cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08103v1)  

---


**ABSTRACT**  
While the operationalisation of high-level AI ethics principles into practical AI/ML systems has made progress, there is still a theory-practice gap in managing tensions between the underlying AI ethics aspects. We cover five approaches for addressing the tensions via trade-offs, ranging from rudimentary to complex. The approaches differ in the types of considered context, scope, methods for measuring contexts, and degree of justification. None of the approaches is likely to be appropriate for all organisations, systems, or applications. To address this, we propose a framework which consists of: (i) proactive identification of tensions, (ii) prioritisation and weighting of ethics aspects, (iii) justification and documentation of trade-off decisions. The proposed framework aims to facilitate the implementation of well-rounded AI/ML systems that are appropriate for potential regulatory requirements.

{{</citation>}}


## cs.NI (2)



### (48/132) Semi-Supervised Learning Approach for Efficient Resource Allocation with Network Slicing in O-RAN (Salar Nouri et al., 2024)

{{<citation>}}

Salar Nouri, Mojdeh Karbalaee Motalleb, Vahid Shah-Mansouri, Seyed Pooya Shariatpanahi. (2024)  
**Semi-Supervised Learning Approach for Efficient Resource Allocation with Network Slicing in O-RAN**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NA, cs-NI, cs.NI, math-NA  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2401.08861v1)  

---


**ABSTRACT**  
The Open Radio Access Network (O-RAN) technology has emerged as a promising solution for network operators, providing them with an open and favorable environment. Ensuring effective coordination of x-applications (xAPPs) is crucial to enhance flexibility and optimize network performance within the O-RAN. In this paper, we introduce an innovative approach to the resource allocation problem, aiming to coordinate multiple independent xAPPs for network slicing and resource allocation in O-RAN. Our proposed method focuses on maximizing the weighted throughput among user equipments (UE), as well as allocating physical resource blocks (PRBs). We prioritize two service types, namely enhanced Mobile Broadband and Ultra Reliable Low Latency Communication. To achieve this, we have designed two xAPPs: a power control xAPP for each UE and a PRB allocation xAPP. The proposed method consists of a two-part training phase, where the first part uses supervised learning with a Variational Autoencoder trained to regress the power transmission as well as the user association and PRB allocation decisions, and the second part uses unsupervised learning with a contrastive loss approach to improve the generalization and robustness of the model. We evaluate the performance of our proposed method by comparing its results to those obtained from an exhaustive search algorithm, deep Q-network algorithm, and by reporting performance metrics for the regression task. We also evaluate the proposed model's performance in different scenarios among the service types. The results show that the proposed method is a more efficient and effective solution for network slicing problems compared to state-of-the-art methods.

{{</citation>}}


### (49/132) Importance-Aware Image Segmentation-based Semantic Communication for Autonomous Driving (Jie Lv et al., 2024)

{{<citation>}}

Jie Lv, Haonan Tong, Qiang Pan, Zhilong Zhang, Xinxin He, Tao Luo, Changchuan Yin. (2024)  
**Importance-Aware Image Segmentation-based Semantic Communication for Autonomous Driving**  

---
Primary Category: cs.NI  
Categories: cs-CV, cs-NI, cs.NI  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.10153v1)  

---


**ABSTRACT**  
This article studies the problem of image segmentation-based semantic communication in autonomous driving. In real traffic scenes, detecting the key objects (e.g., vehicles, pedestrians and obstacles) is more crucial than that of other objects to guarantee driving safety. Therefore, we propose a vehicular image segmentation-oriented semantic communication system, termed VIS-SemCom, where image segmentation features of important objects are transmitted to reduce transmission redundancy. First, to accurately extract image semantics, we develop a semantic codec based on Swin Transformer architecture, which expands the perceptual field thus improving the segmentation accuracy. Next, we propose a multi-scale semantic extraction scheme via assigning the number of Swin Transformer blocks for diverse resolution features, thus highlighting the important objects' accuracy. Furthermore, the importance-aware loss is invoked to emphasize the important objects, and an online hard sample mining (OHEM) strategy is proposed to handle small sample issues in the dataset. Experimental results demonstrate that the proposed VIS-SemCom can achieve a coding gain of nearly 6 dB with a 60% mean intersection over union (mIoU), reduce the transmitted data amount by up to 70% with a 60% mIoU, and improve the segmentation intersection over union (IoU) of important objects by 4%, compared to traditional transmission scheme.

{{</citation>}}


## cs.IR (4)



### (50/132) Exploring Content-Based and Meta-Data Analysis for Detecting Fake News Infodemic: A case study on COVID-19 (Oluwaseun Ajao et al., 2024)

{{<citation>}}

Oluwaseun Ajao, Ashish Garg, Marjory Da Costa-Abreu. (2024)  
**Exploring Content-Based and Meta-Data Analysis for Detecting Fake News Infodemic: A case study on COVID-19**  

---
Primary Category: cs.IR  
Categories: H-3-3, cs-IR, cs.IR  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2401.08841v1)  

---


**ABSTRACT**  
The coronavirus pandemic (COVID-19) is probably the most disruptive global health disaster in recent history. It negatively impacted the whole world and virtually brought the global economy to a standstill. However, as the virus was spreading, infecting people and claiming thousands of lives so was the spread and propagation of fake news, misinformation and disinformation about the event. These included the spread of unconfirmed health advice and remedies on social media. In this paper, false information about the pandemic is identified using a content-based approach and metadata curated from messages posted to online social networks. A content-based approach combined with metadata as well as an initial feature analysis is used and then several supervised learning models are tested for identifying and predicting misleading posts. Our approach shows up to 93% accuracy in the detection of fake news related posts about the COVID-19 pandemic

{{</citation>}}


### (51/132) Content-Aware Tweet Location Inference using Quadtree Spatial Partitioning and Jaccard-Cosine Word Embedding (Oluwaseun Ajao et al., 2024)

{{<citation>}}

Oluwaseun Ajao, Deepayan Bhowmik, Shahrzad Zargari. (2024)  
**Content-Aware Tweet Location Inference using Quadtree Spatial Partitioning and Jaccard-Cosine Word Embedding**  

---
Primary Category: cs.IR  
Categories: H-3-3, cs-IR, cs.IR  
Keywords: Embedding, NLP, Twitter, Word Embedding  
[Paper Link](http://arxiv.org/abs/2401.08506v1)  

---


**ABSTRACT**  
Inferring locations from user texts on social media platforms is a non-trivial and challenging problem relating to public safety. We propose a novel non-uniform grid-based approach for location inference from Twitter messages using Quadtree spatial partitions. The proposed algorithm uses natural language processing (NLP) for semantic understanding and incorporates Cosine similarity and Jaccard similarity measures for feature vector extraction and dimensionality reduction. We chose Twitter as our experimental social media platform due to its popularity and effectiveness for the dissemination of news and stories about recent events happening around the world. Our approach is the first of its kind to make location inference from tweets using Quadtree spatial partitions and NLP, in hybrid word-vector representations. The proposed algorithm achieved significant classification accuracy and outperformed state-of-the-art grid-based content-only location inference methods by up to 24% in correctly predicting tweet locations within a 161km radius and by 300km in median error distance on benchmark datasets.

{{</citation>}}


### (52/132) Generative Multi-Modal Knowledge Retrieval with Large Language Models (Xinwei Long et al., 2024)

{{<citation>}}

Xinwei Long, Jiali Zeng, Fandong Meng, Zhiyuan Ma, Kaiyan Zhang, Bowen Zhou, Jie Zhou. (2024)  
**Generative Multi-Modal Knowledge Retrieval with Large Language Models**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.08206v1)  

---


**ABSTRACT**  
Knowledge retrieval with multi-modal queries plays a crucial role in supporting knowledge-intensive multi-modal applications. However, existing methods face challenges in terms of their effectiveness and training efficiency, especially when it comes to training and integrating multiple retrievers to handle multi-modal queries. In this paper, we propose an innovative end-to-end generative framework for multi-modal knowledge retrieval. Our framework takes advantage of the fact that large language models (LLMs) can effectively serve as virtual knowledge bases, even when trained with limited data. We retrieve knowledge via a two-step process: 1) generating knowledge clues related to the queries, and 2) obtaining the relevant document by searching databases using the knowledge clue. In particular, we first introduce an object-aware prefix-tuning technique to guide multi-grained visual learning. Then, we align multi-grained visual features into the textual feature space of the LLM, employing the LLM to capture cross-modal interactions. Subsequently, we construct instruction data with a unified format for model training. Finally, we propose the knowledge-guided generation strategy to impose prior constraints in the decoding steps, thereby promoting the generation of distinctive knowledge clues. Through experiments conducted on three benchmarks, we demonstrate significant improvements ranging from 3.0% to 14.6% across all evaluation metrics when compared to strong baselines.

{{</citation>}}


### (53/132) A Reproducibility Study of Goldilocks: Just-Right Tuning of BERT for TAR (Xinyu Mao et al., 2024)

{{<citation>}}

Xinyu Mao, Bevan Koopman, Guido Zuccon. (2024)  
**A Reproducibility Study of Goldilocks: Just-Right Tuning of BERT for TAR**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.08104v1)  

---


**ABSTRACT**  
Screening documents is a tedious and time-consuming aspect of high-recall retrieval tasks, such as compiling a systematic literature review, where the goal is to identify all relevant documents for a topic. To help streamline this process, many Technology-Assisted Review (TAR) methods leverage active learning techniques to reduce the number of documents requiring review. BERT-based models have shown high effectiveness in text classification, leading to interest in their potential use in TAR workflows. In this paper, we investigate recent work that examined the impact of further pre-training epochs on the effectiveness and efficiency of a BERT-based active learning pipeline. We first report that we could replicate the original experiments on two specific TAR datasets, confirming some of the findings: importantly, that further pre-training is critical to high effectiveness, but requires attention in terms of selecting the correct training epoch. We then investigate the generalisability of the pipeline on a different TAR task, that of medical systematic reviews. In this context, we show that there is no need for further pre-training if a domain-specific BERT backbone is used within the active learning pipeline. This finding provides practical implications for using the studied active learning pipeline within domain-specific TAR tasks.

{{</citation>}}


## cs.CL (28)



### (54/132) Improving ASR Contextual Biasing with Guided Attention (Jiyang Tang et al., 2024)

{{<citation>}}

Jiyang Tang, Kwangyoun Kim, Suwon Shon, Felix Wu, Prashant Sridhar, Shinji Watanabe. (2024)  
**Improving ASR Contextual Biasing with Guided Attention**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-AS  
Keywords: Attention, Bias  
[Paper Link](http://arxiv.org/abs/2401.08835v1)  

---


**ABSTRACT**  
In this paper, we propose a Guided Attention (GA) auxiliary training loss, which improves the effectiveness and robustness of automatic speech recognition (ASR) contextual biasing without introducing additional parameters. A common challenge in previous literature is that the word error rate (WER) reduction brought by contextual biasing diminishes as the number of bias phrases increases. To address this challenge, we employ a GA loss as an additional training objective besides the Transducer loss. The proposed GA loss aims to teach the cross attention how to align bias phrases with text tokens or audio frames. Compared to studies with similar motivations, the proposed loss operates directly on the cross attention weights and is easier to implement. Through extensive experiments based on Conformer Transducer with Contextual Adapter, we demonstrate that the proposed method not only leads to a lower WER but also retains its effectiveness as the number of bias phrases increases. Specifically, the GA loss decreases the WER of rare vocabularies by up to 19.2% on LibriSpeech compared to the contextual biasing baseline, and up to 49.3% compared to a vanilla Transducer.

{{</citation>}}


### (55/132) HuixiangDou: Overcoming Group Chat Scenarios with LLM-based Technical Assistance (Huanjun Kong et al., 2024)

{{<citation>}}

Huanjun Kong, Songyang Zhang, Kai Chen. (2024)  
**HuixiangDou: Overcoming Group Chat Scenarios with LLM-based Technical Assistance**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.08772v1)  

---


**ABSTRACT**  
In this work, we present HuixiangDou, a technical assistant powered by Large Language Models (LLM). This system is designed to assist algorithm developers by providing insightful responses to questions related to open-source algorithm projects, such as computer vision and deep learning projects from OpenMMLab. We further explore the integration of this assistant into the group chats of instant messaging (IM) tools such as WeChat and Lark. Through several iterative improvements and trials, we have developed a sophisticated technical chat assistant capable of effectively answering users' technical questions without causing message flooding. This paper's contributions include: 1) Designing an algorithm pipeline specifically for group chat scenarios; 2) Verifying the reliable performance of text2vec in task rejection; 3) Identifying three critical requirements for LLMs in technical-assistant-like products, namely scoring ability, In-Context Learning (ICL), and Long Context. We have made the software and source code available at https://github.com/internlm/huixiangdou to aid in future research and application. HuixiangDou is applicable to any group chat within IM tools.

{{</citation>}}


### (56/132) Deductive Closure Training of Language Models for Coherence, Accuracy, and Updatability (Afra Feyza Akyürek et al., 2024)

{{<citation>}}

Afra Feyza Akyürek, Ekin Akyürek, Leshem Choshen, Derry Wijaya, Jacob Andreas. (2024)  
**Deductive Closure Training of Language Models for Coherence, Accuracy, and Updatability**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.08574v1)  

---


**ABSTRACT**  
While language models (LMs) can sometimes generate factually correct text and estimate truth values of individual claims, these generally do not reflect a globally coherent, manipulable model of the world. As a consequence, current LMs also generate incorrect or nonsensical content, and are difficult to edit and bring up to date. We present a method called Deductive Closure Training (DCT) that uses LMs themselves to identify implications of (and contradictions within) the text that they generate, yielding an efficient self-supervised procedure for improving LM factuality. Given a collection of seed documents, DCT prompts LMs to generate additional text implied by these documents, reason globally about the correctness of this generated text, and finally fine-tune on text inferred to be correct. Given seed documents from a trusted source, DCT provides a tool for supervised model updating; if seed documents are sampled from the LM itself, DCT enables fully unsupervised fine-tuning for improved coherence and accuracy. Across the CREAK, MQUaKE, and Reversal Curse datasets, supervised DCT improves LM fact verification and text generation accuracy by 3-26%; on CREAK fully unsupervised DCT improves verification accuracy by 12%. These results show that LMs' reasoning capabilities during inference can be leveraged during training to improve their reliability.

{{</citation>}}


### (57/132) Tuning Language Models by Proxy (Alisa Liu et al., 2024)

{{<citation>}}

Alisa Liu, Xiaochuang Han, Yizhong Wang, Yulia Tsvetkov, Yejin Choi, Noah A. Smith. (2024)  
**Tuning Language Models by Proxy**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2401.08565v1)  

---


**ABSTRACT**  
Despite the general capabilities of large pretrained language models, they consistently benefit from further adaptation to better achieve desired behaviors. However, tuning these models has become increasingly resource-intensive, or impossible when model weights are private. We introduce proxy-tuning, a lightweight decoding-time algorithm that operates on top of black-box LMs to achieve the result of directly tuning the model, but by accessing only its prediction over the output vocabulary. Our method instead tunes a smaller LM, then applies the difference between the predictions of the small tuned and untuned LMs to shift the original predictions of the base model in the direction of tuning, while retaining the benefits of larger scale pretraining. In experiments, when we apply proxy-tuning to Llama2-70B using proxies of only 7B size, we can close 88% of the gap between Llama2-70B and its truly-tuned chat version, when evaluated across knowledge, reasoning, and safety benchmarks. Interestingly, when tested on TruthfulQA, proxy-tuned models are actually more truthful than directly tuned models, possibly because decoding-time guidance better retains the model's factual knowledge. We then demonstrate the generality of proxy-tuning by applying it for domain adaptation on code, and task-specific finetuning on question-answering and math problems. Our work demonstrates the promise of using small tuned LMs to efficiently customize large, potentially proprietary LMs through decoding-time guidance.

{{</citation>}}


### (58/132) The Gaps between Pre-train and Downstream Settings in Bias Evaluation and Debiasing (Masahiro Kaneko et al., 2024)

{{<citation>}}

Masahiro Kaneko, Danushka Bollegala, Timothy Baldwin. (2024)  
**The Gaps between Pre-train and Downstream Settings in Bias Evaluation and Debiasing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2401.08511v1)  

---


**ABSTRACT**  
The output tendencies of Pre-trained Language Models (PLM) vary markedly before and after Fine-Tuning (FT) due to the updates to the model parameters. These divergences in output tendencies result in a gap in the social biases of PLMs. For example, there exits a low correlation between intrinsic bias scores of a PLM and its extrinsic bias scores under FT-based debiasing methods. Additionally, applying FT-based debiasing methods to a PLM leads to a decline in performance in downstream tasks. On the other hand, PLMs trained on large datasets can learn without parameter updates via In-Context Learning (ICL) using prompts. ICL induces smaller changes to PLMs compared to FT-based debiasing methods. Therefore, we hypothesize that the gap observed in pre-trained and FT models does not hold true for debiasing methods that use ICL. In this study, we demonstrate that ICL-based debiasing methods show a higher correlation between intrinsic and extrinsic bias scores compared to FT-based methods. Moreover, the performance degradation due to debiasing is also lower in the ICL case compared to that in the FT case.

{{</citation>}}


### (59/132) EmoLLMs: A Series of Emotional Large Language Models and Annotation Tools for Comprehensive Affective Analysis (Zhiwei Liu et al., 2024)

{{<citation>}}

Zhiwei Liu, Kailai Yang, Tianlin Zhang, Qianqian Xie, Zeping Yu, Sophia Ananiadou. (2024)  
**EmoLLMs: A Series of Emotional Large Language Models and Annotation Tools for Comprehensive Affective Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-4, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.08508v1)  

---


**ABSTRACT**  
Sentiment analysis and emotion detection are important research topics in natural language processing (NLP) and benefit many downstream tasks. With the widespread application of LLMs, researchers have started exploring the application of LLMs based on instruction-tuning in the field of sentiment analysis. However, these models only focus on single aspects of affective classification tasks (e.g. sentimental polarity or categorical emotions), and overlook the regression tasks (e.g. sentiment strength or emotion intensity), which leads to poor performance in downstream tasks. The main reason is the lack of comprehensive affective instruction tuning datasets and evaluation benchmarks, which cover various affective classification and regression tasks. Moreover, although emotional information is useful for downstream tasks, existing downstream datasets lack high-quality and comprehensive affective annotations. In this paper, we propose EmoLLMs, the first series of open-sourced instruction-following LLMs for comprehensive affective analysis based on fine-tuning various LLMs with instruction data, the first multi-task affective analysis instruction dataset (AAID) with 234K data samples based on various classification and regression tasks to support LLM instruction tuning, and a comprehensive affective evaluation benchmark (AEB) with 14 tasks from various sources and domains to test the generalization ability of LLMs. We propose a series of EmoLLMs by fine-tuning LLMs with AAID to solve various affective instruction tasks. We compare our model with a variety of LLMs on AEB, where our models outperform all other open-sourced LLMs, and surpass ChatGPT and GPT-4 in most tasks, which shows that the series of EmoLLMs achieve the ChatGPT-level and GPT-4-level generalization capabilities on affective analysis tasks, and demonstrates our models can be used as affective annotation tools.

{{</citation>}}


### (60/132) The Effect of Group Status on the Variability of Group Representations in LLM-generated Text (Messi H. J. Lee et al., 2024)

{{<citation>}}

Messi H. J. Lee, Jacob M. Montgomery, Calvin K. Lai. (2024)  
**The Effect of Group Status on the Variability of Group Representations in LLM-generated Text**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.08495v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have become pervasive in everyday life, yet their inner workings remain opaque. While scholarly efforts have demonstrated LLMs' propensity to reproduce biases in their training data, they have primarily focused on the association of social groups with stereotypic attributes. In this paper, we extend this line of inquiry to investigate a bias akin to the social-psychological phenomenon where socially dominant groups are perceived to be less homogeneous than socially subordinate groups as it is reproduced by LLMs. We had ChatGPT, a state-of-the-art LLM, generate a diversity of texts about intersectional group identities and compared text homogeneity. We consistently find that LLMs portray African, Asian, and Hispanic Americans as more homogeneous than White Americans. They also portray women as more homogeneous than men, but these differences are small. Finally, we find that the effect of gender differs across racial/ethnic groups such that the effect of gender is consistent within African and Hispanic Americans but not within Asian and White Americans. We speculate possible sources of this bias in LLMs and posit that the bias has the potential to amplify biases in future LLM training and to reinforce stereotypes.

{{</citation>}}


### (61/132) Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models (Tassilo Klein et al., 2024)

{{<citation>}}

Tassilo Klein, Moin Nabi. (2024)  
**Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Perplexity  
[Paper Link](http://arxiv.org/abs/2401.08491v1)  

---


**ABSTRACT**  
The generation of undesirable and factually incorrect content of large language models poses a significant challenge and remains largely an unsolved issue. This paper studies the integration of a contrastive learning objective for fine-tuning LLMs for implicit knowledge editing and controlled text generation. Optimizing the training objective entails aligning text perplexities in a contrastive fashion. To facilitate training the model in a self-supervised fashion, we leverage an off-the-shelf LLM for training data generation. We showcase applicability in the domain of detoxification. Herein, the proposed approach leads to a significant decrease in the generation of toxic content while preserving general utility for downstream tasks such as commonsense reasoning and reading comprehension. The proposed approach is conceptually simple but empirically powerful.

{{</citation>}}


### (62/132) Machine Translation with Large Language Models: Prompt Engineering for Persian, English, and Russian Directions (Nooshin Pourkamali et al., 2024)

{{<citation>}}

Nooshin Pourkamali, Shler Ebrahim Sharifi. (2024)  
**Machine Translation with Large Language Models: Prompt Engineering for Persian, English, and Russian Directions**  

---
Primary Category: cs.CL  
Categories: ACM-class: I-2-2, I-2-2, cs-AI, cs-CL, cs-HC, cs-LG, cs.CL  
Keywords: Language Model, Machine Translation, NLP, PaLM  
[Paper Link](http://arxiv.org/abs/2401.08429v1)  

---


**ABSTRACT**  
Generative large language models (LLMs) have demonstrated exceptional proficiency in various natural language processing (NLP) tasks, including machine translation, question answering, text summarization, and natural language understanding.   To further enhance the performance of LLMs in machine translation, we conducted an investigation into two popular prompting methods and their combination, focusing on cross-language combinations of Persian, English, and Russian. We employed n-shot feeding and tailored prompting frameworks. Our findings indicate that multilingual LLMs like PaLM exhibit human-like machine translation outputs, enabling superior fine-tuning of desired translation nuances in accordance with style guidelines and linguistic considerations. These models also excel in processing and applying prompts. However, the choice of language model, machine translation task, and the specific source and target languages necessitate certain considerations when adopting prompting frameworks and utilizing n-shot in-context learning.   Furthermore, we identified errors and limitations inherent in popular LLMs as machine translation tools and categorized them based on various linguistic metrics. This typology of errors provides valuable insights for utilizing LLMs effectively and offers methods for designing prompts for in-context learning. Our report aims to contribute to the advancement of machine translation with LLMs by improving both the accuracy and reliability of evaluation metrics.

{{</citation>}}


### (63/132) Ask the experts: sourcing high-quality datasets for nutritional counselling through Human-AI collaboration (Simone Balloccu et al., 2024)

{{<citation>}}

Simone Balloccu, Ehud Reiter, Vivek Kumar, Diego Reforgiato Recupero, Daniele Riboni. (2024)  
**Ask the experts: sourcing high-quality datasets for nutritional counselling through Human-AI collaboration**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.08420v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), with their flexible generation abilities, can be powerful data sources in domains with few or no available corpora. However, problems like hallucinations and biases limit such applications. In this case study, we pick nutrition counselling, a domain lacking any public resource, and show that high-quality datasets can be gathered by combining LLMs, crowd-workers and nutrition experts. We first crowd-source and cluster a novel dataset of diet-related issues, then work with experts to prompt ChatGPT into producing related supportive text. Finally, we let the experts evaluate the safety of the generated text. We release HAI-coaching, the first expert-annotated nutrition counselling dataset containing ~2.4K dietary struggles from crowd workers, and ~97K related supportive texts generated by ChatGPT. Extensive analysis shows that ChatGPT while producing highly fluent and human-like text, also manifests harmful behaviours, especially in sensitive topics like mental health, making it unsuitable for unsupervised use.

{{</citation>}}


### (64/132) Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation (Haoran Xu et al., 2024)

{{<citation>}}

Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, Young Jin Kim. (2024)  
**Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.08417v2)  

---


**ABSTRACT**  
Moderate-sized large language models (LLMs) -- those with 7B or 13B parameters -- exhibit promising machine translation (MT) performance. However, even the top-performing 13B LLM-based translation models, like ALMA, does not match the performance of state-of-the-art conventional encoder-decoder translation models or larger-scale LLMs such as GPT-4. In this study, we bridge this performance gap. We first assess the shortcomings of supervised fine-tuning for LLMs in the MT task, emphasizing the quality issues present in the reference data, despite being human-generated. Then, in contrast to SFT which mimics reference translations, we introduce Contrastive Preference Optimization (CPO), a novel approach that trains models to avoid generating adequate but not perfect translations. Applying CPO to ALMA models with only 22K parallel sentences and 12M parameters yields significant improvements. The resulting model, called ALMA-R, can match or exceed the performance of the WMT competition winners and GPT-4 on WMT'21, WMT'22 and WMT'23 test datasets.

{{</citation>}}


### (65/132) RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture (Angels Balaguer et al., 2024)

{{<citation>}}

Angels Balaguer, Vinamra Benara, Renato Luiz de Freitas Cunha, Roberto de M. Estevão Filho, Todd Hendry, Daniel Holstein, Jennifer Marsman, Nick Mecklenburg, Sara Malvar, Leonardo O. Nunes, Rafael Padilha, Morris Sharp, Bruno Silva, Swati Sharma, Vijay Aski, Ranveer Chandra. (2024)  
**RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.08406v2)  

---


**ABSTRACT**  
There are two common ways in which developers are incorporating proprietary and domain-specific data when building applications of Large Language Models (LLMs): Retrieval-Augmented Generation (RAG) and Fine-Tuning. RAG augments the prompt with the external data, while fine-Tuning incorporates the additional knowledge into the model itself. However, the pros and cons of both approaches are not well understood. In this paper, we propose a pipeline for fine-tuning and RAG, and present the tradeoffs of both for multiple popular LLMs, including Llama2-13B, GPT-3.5, and GPT-4. Our pipeline consists of multiple stages, including extracting information from PDFs, generating questions and answers, using them for fine-tuning, and leveraging GPT-4 for evaluating the results. We propose metrics to assess the performance of different stages of the RAG and fine-Tuning pipeline. We conduct an in-depth study on an agricultural dataset. Agriculture as an industry has not seen much penetration of AI, and we study a potentially disruptive application - what if we could provide location-specific insights to a farmer? Our results show the effectiveness of our dataset generation pipeline in capturing geographic-specific knowledge, and the quantitative and qualitative benefits of RAG and fine-tuning. We see an accuracy increase of over 6 p.p. when fine-tuning the model and this is cumulative with RAG, which increases accuracy by 5 p.p. further. In one particular experiment, we also demonstrate that the fine-tuned model leverages information from across geographies to answer specific questions, increasing answer similarity from 47% to 72%. Overall, the results point to how systems built using LLMs can be adapted to respond and incorporate knowledge across a dimension that is critical for a specific industry, paving the way for further applications of LLMs in other industrial domains.

{{</citation>}}


### (66/132) Hallucination Detection and Hallucination Mitigation: An Investigation (Junliang Luo et al., 2024)

{{<citation>}}

Junliang Luo, Tianyu Li, Di Wu, Michael Jenkin, Steve Liu, Gregory Dudek. (2024)  
**Hallucination Detection and Hallucination Mitigation: An Investigation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.08358v1)  

---


**ABSTRACT**  
Large language models (LLMs), including ChatGPT, Bard, and Llama, have achieved remarkable successes over the last two years in a range of different applications. In spite of these successes, there exist concerns that limit the wide application of LLMs. A key problem is the problem of hallucination. Hallucination refers to the fact that in addition to correct responses, LLMs can also generate seemingly correct but factually incorrect responses. This report aims to present a comprehensive review of the current literature on both hallucination detection and hallucination mitigation. We hope that this report can serve as a good reference for both engineers and researchers who are interested in LLMs and applying them to real world tasks.

{{</citation>}}


### (67/132) Salute the Classic: Revisiting Challenges of Machine Translation in the Age of Large Language Models (Jianhui Pang et al., 2024)

{{<citation>}}

Jianhui Pang, Fanghua Ye, Longyue Wang, Dian Yu, Derek F. Wong, Shuming Shi, Zhaopeng Tu. (2024)  
**Salute the Classic: Revisiting Challenges of Machine Translation in the Age of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.08350v2)  

---


**ABSTRACT**  
The evolution of Neural Machine Translation (NMT) has been significantly influenced by six core challenges (Koehn and Knowles, 2017), which have acted as benchmarks for progress in this field. This study revisits these challenges, offering insights into their ongoing relevance in the context of advanced Large Language Models (LLMs): domain mismatch, amount of parallel data, rare word prediction, translation of long sentences, attention model as word alignment, and sub-optimal beam search. Our empirical findings indicate that LLMs effectively lessen the reliance on parallel data for major languages in the pretraining phase. Additionally, the LLM-based translation system significantly enhances the translation of long sentences that contain approximately 80 words and shows the capability to translate documents of up to 512 words. However, despite these significant improvements, the challenges of domain mismatch and prediction of rare words persist. While the challenges of word alignment and beam search, specifically associated with NMT, may not apply to LLMs, we identify three new challenges for LLMs in translation tasks: inference efficiency, translation of low-resource languages in the pretraining phase, and human-aligned evaluation. The datasets and models are released at https://github.com/pangjh3/LLM4MT.

{{</citation>}}


### (68/132) RoTBench: A Multi-Level Benchmark for Evaluating the Robustness of Large Language Models in Tool Learning (Junjie Ye et al., 2024)

{{<citation>}}

Junjie Ye, Yilong Wu, Songyang Gao, Caishuang Huang, Sixian Li, Guanyu Li, Xiaoran Fan, Qi Zhang, Tao Gui, Xuanjing Huang. (2024)  
**RoTBench: A Multi-Level Benchmark for Evaluating the Robustness of Large Language Models in Tool Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.08326v2)  

---


**ABSTRACT**  
Tool learning has generated widespread interest as a vital means of interaction between Large Language Models (LLMs) and the physical world. Current research predominantly emphasizes LLMs' capacity to utilize tools in well-structured environments while overlooking their stability when confronted with the inevitable noise of the real world. To bridge this gap, we introduce RoTBench, a multi-level benchmark for evaluating the robustness of LLMs in tool learning. Specifically, we establish five external environments, each featuring varying levels of noise (i.e., Clean, Slight, Medium, Heavy, and Union), providing an in-depth analysis of the model's resilience across three critical phases: tool selection, parameter identification, and content filling. Experiments involving six widely-used models underscore the urgent necessity for enhancing the robustness of LLMs in tool learning. For instance, the performance of GPT-4 even drops significantly from 80.00 to 58.10 when there is no substantial change in manual accuracy. More surprisingly, the noise correction capability inherent in the GPT family paradoxically impedes its adaptability in the face of mild noise. In light of these findings, we propose RoTTuning, a strategy that enriches the diversity of training environments to bolster the robustness of LLMs in tool learning. The code and data are available at https://github.com/Junjie-Ye/RoTBench.

{{</citation>}}


### (69/132) Application of LLM Agents in Recruitment: A Novel Framework for Resume Screening (Chengguang Gan et al., 2024)

{{<citation>}}

Chengguang Gan, Qinghao Zhang, Tatsunori Mori. (2024)  
**Application of LLM Agents in Recruitment: A Novel Framework for Resume Screening**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.08315v1)  

---


**ABSTRACT**  
The automation of resume screening is a crucial aspect of the recruitment process in organizations. Automated resume screening systems often encompass a range of natural language processing (NLP) tasks. The advent of Large Language Models (LLMs) has notably enhanced the efficacy of these systems, showcasing their robust generalization abilities across diverse language-related tasks. Accompanying these developments are various agents based on LLMs, which facilitate their application in practical scenarios. This paper introduces a novel LLM-based agent framework for resume screening, aimed at enhancing efficiency and time management in recruitment processes. Our framework is distinct in its ability to efficiently summarize and grade each resume from a large dataset. Moreover, it utilizes LLM agents for decision-making, determining which candidates receive job offers, or which ones to bring in for interviews. To evaluate our framework, we constructed a dataset from actual resumes and conducted simulate a resume screening process. Subsequently, the outcomes of the simulation experiment were compared and subjected to detailed analysis. The results demonstrate that our automated resume screening framework is 11 times faster than traditional manual methods. Furthermore, by fine-tuning the LLMs, we observed a significant improvement in the F1 score, reaching 87.73\%, during the resume sentence classification phase. In the resume summarization and grading phase, our fine-tuned model surpassed the baseline performance of the GPT-3.5 model. Analysis of the decision-making efficacy of the LLM agents in the final offer stage further underscores the potential of LLM agents in transforming resume screening processes.

{{</citation>}}


### (70/132) DAPT: A Dual Attention Framework for Parameter-Efficient Continual Learning of Large Language Models (Weixiang Zhao et al., 2024)

{{<citation>}}

Weixiang Zhao, Shilong Wang, Yulin Hu, Yanyan Zhao, Bing Qin, Xuanyu Zhang, Qing Yang, Dongliang Xu, Wanxiang Che. (2024)  
**DAPT: A Dual Attention Framework for Parameter-Efficient Continual Learning of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Language Model  
[Paper Link](http://arxiv.org/abs/2401.08295v1)  

---


**ABSTRACT**  
The continual learning (CL) ability is vital for deploying large language models (LLMs) in the dynamic world. Based on parameter-efficient tuning (PET), existing methods devise the learning module and the selection module to handle the challenges of catastrophic forgetting (CF) and knowledge transfer (KT) in CL. The learning module allocates separate PET blocks for each continually emerged task and the selection module function to choose the correct one for the input at testing time. However, there are limitations in their deigns of both modules and they ignore the potential of aligning the two module to address CF and KT simultaneously. To this end, we propose a novel Dual Attention Framework , to align the PET learning and selection via the Dual Attentive Learning\&Selection module. Extensive Experiments on two CL benchmarks demonstrate the superiority of DAPT to resist CF and facilitate KT at the same time. Moreover, DAPT exhibits the superiority when we scale it to different model sizes (from 770M to 11B) and unseen tasks.

{{</citation>}}


### (71/132) Inferflow: an Efficient and Highly Configurable Inference Engine for Large Language Models (Shuming Shi et al., 2024)

{{<citation>}}

Shuming Shi, Enbo Zhao, Deng Cai, Leyang Cui, Xinting Huang, Huayang Li. (2024)  
**Inferflow: an Efficient and Highly Configurable Inference Engine for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.08294v1)  

---


**ABSTRACT**  
We present Inferflow, an efficient and highly configurable inference engine for large language models (LLMs). With Inferflow, users can serve most of the common transformer models by simply modifying some lines in corresponding configuration files, without writing a single line of source code. Compared with most existing inference engines, Inferflow has some key features. First, by implementing a modular framework of atomic build-blocks and technologies, Inferflow is compositionally generalizable to new models. Second, 3.5-bit quantization is introduced in Inferflow as a tradeoff between 3-bit and 4-bit quantization. Third, hybrid model partitioning for multi-GPU inference is introduced in Inferflow to better balance inference speed and throughput than the existing partition-by-layer and partition-by-tensor strategies.

{{</citation>}}


### (72/132) Large Language Models are Null-Shot Learners (Pittawat Taveekitworachai et al., 2024)

{{<citation>}}

Pittawat Taveekitworachai, Febri Abdullah, Ruck Thawonmas. (2024)  
**Large Language Models are Null-Shot Learners**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.08273v1)  

---


**ABSTRACT**  
This paper presents null-shot prompting. Null-shot prompting exploits hallucination in large language models (LLMs) by instructing LLMs to utilize information from the "Examples" section that never exists within the provided context to perform a task. While reducing hallucination is crucial and non-negligible for daily and critical uses of LLMs, we propose that in the current landscape in which these LLMs still hallucinate, it is possible, in fact, to exploit hallucination to increase performance in performing tasks compared to standard zero-shot prompting. Experiments with six LLMs show improvements in performance across the majority of eight datasets, including reading comprehension, arithmetic reasoning, and closed-book question answering. The observed inconsistency in increased relative performance across LLMs also potentially indicates a different degree of inherent hallucination in each model. These differences show that it is possible to utilize null-shot prompting as a way to detect degrees of hallucination in LLMs using existing benchmarking datasets. We also perform ablation studies, including experimenting with a modified version of null-shot prompting that incorporates ideas from zero-shot chain-of-thought prompting, which shows different trends of results.

{{</citation>}}


### (73/132) A Generative Adversarial Attack for Multilingual Text Classifiers (Tom Roth et al., 2024)

{{<citation>}}

Tom Roth, Inigo Jauregi Unanue, Alsharif Abuadbba, Massimo Piccardi. (2024)  
**A Generative Adversarial Attack for Multilingual Text Classifiers**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Adversarial Attack, Multilingual  
[Paper Link](http://arxiv.org/abs/2401.08255v1)  

---


**ABSTRACT**  
Current adversarial attack algorithms, where an adversary changes a text to fool a victim model, have been repeatedly shown to be effective against text classifiers. These attacks, however, generally assume that the victim model is monolingual and cannot be used to target multilingual victim models, a significant limitation given the increased use of these models. For this reason, in this work we propose an approach to fine-tune a multilingual paraphrase model with an adversarial objective so that it becomes able to generate effective adversarial examples against multilingual classifiers. The training objective incorporates a set of pre-trained models to ensure text quality and language consistency of the generated text. In addition, all the models are suitably connected to the generator by vocabulary-mapping matrices, allowing for full end-to-end differentiability of the overall training pipeline. The experimental validation over two multilingual datasets and five languages has shown the effectiveness of the proposed approach compared to existing baselines, particularly in terms of query efficiency. We also provide a detailed analysis of the generated attacks and discuss limitations and opportunities for future research.

{{</citation>}}


### (74/132) MARIO: MAth Reasoning with code Interpreter Output -- A Reproducible Pipeline (Minpeng Liao et al., 2024)

{{<citation>}}

Minpeng Liao, Wei Luo, Chengxi Li, Jing Wu, Kai Fan. (2024)  
**MARIO: MAth Reasoning with code Interpreter Output -- A Reproducible Pipeline**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.08190v1)  

---


**ABSTRACT**  
Large language models (LLMs) have seen considerable advancements in natural language understanding tasks, yet there remains a gap to bridge before attaining true artificial general intelligence, especially concerning shortcomings in mathematical reasoning capabilities. We postulate that the inherent nature of LLM training, which focuses on predicting probabilities of next token, presents challenges in effectively modeling mathematical reasoning that demands exact calculations, both from data-driven and theoretical standpoints. In this paper, we address this challenge by enriching the data landscape and introducing a novel math dataset, enhanced with a capability to utilize a Python code interpreter. This dataset is derived from GSM8K and MATH and has been further refined through a combination of GPT-4 annotations, human review, and self-training processes, where the errors in the original GSM8K training set have been fixed. Additionally, we propose a tentative, easily replicable protocol for the fine-tuning of math-specific LLMs, which has led to a significant improvement in the performance of a 7B-parameter LLM on the GSM8K and MATH datasets. We are committed to advancing the field of mathematical reasoning in LLMs and, to that end, we have made the model checkpoints and will make the dataset publicly available. We hope this will facilitate further research and development within the community.

{{</citation>}}


### (75/132) A Study on Training and Developing Large Language Models for Behavior Tree Generation (Fu Li et al., 2024)

{{<citation>}}

Fu Li, Xueying Wang, Bin Li, Yunlong Wu, Yanzhen Wang, Xiaodong Yi. (2024)  
**A Study on Training and Developing Large Language Models for Behavior Tree Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-RO, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.08089v1)  

---


**ABSTRACT**  
This paper presents an innovative exploration of the application potential of large language models (LLM) in addressing the challenging task of automatically generating behavior trees (BTs) for complex tasks. The conventional manual BT generation method is inefficient and heavily reliant on domain expertise. On the other hand, existing automatic BT generation technologies encounter bottlenecks related to task complexity, model adaptability, and reliability. In order to overcome these challenges, we propose a novel methodology that leverages the robust representation and reasoning abilities of LLMs. The core contribution of this paper lies in the design of a BT generation framework based on LLM, which encompasses the entire process, from data synthesis and model training to application developing and data verification. Synthetic data is introduced to train the BT generation model (BTGen model), enhancing its understanding and adaptability to various complex tasks, thereby significantly improving its overall performance. In order to ensure the effectiveness and executability of the generated BTs, we emphasize the importance of data verification and introduce a multilevel verification strategy. Additionally, we explore a range of agent design and development schemes with LLM as the central element. We hope that the work in this paper may provide a reference for the researchers who are interested in BT generation based on LLMs.

{{</citation>}}


### (76/132) Enhancing Document-level Translation of Large Language Model via Translation Mixed-instructions (Yachao Li et al., 2024)

{{<citation>}}

Yachao Li, Junhui Li, Jing Jiang, Min Zhang. (2024)  
**Enhancing Document-level Translation of Large Language Model via Translation Mixed-instructions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Language Model  
[Paper Link](http://arxiv.org/abs/2401.08088v1)  

---


**ABSTRACT**  
Existing large language models (LLMs) for machine translation are typically fine-tuned on sentence-level translation instructions and achieve satisfactory performance at the sentence level. However, when applied to document-level translation, these models face a significant challenge, particularly when dealing with documents containing over 512 tokens. This challenge arises from the issue of sentence-level coverage, where subsequent sentences in the document remain untranslated. As a result, the document-level translation capability of LLMs fine-tuned on sentence-level translation instructions is significantly limited. We conjecture that the primary cause of LLMs' weak document-level translation performance is the absence of document-to-document mapping ability. To address the issue, we propose an approach that combines sentence-level and document-level translation instructions of varying lengths to fine-tune LLMs. Our proposed translation mixed-instructions enable LLMs (Llama-2~7B and 13B) to maintain consistent translation performance from the sentence level to documents containing as many as 2048 tokens. Extensive experimental results show that the proposed approach significantly enhances the document-level translation capabilities of LLMs on 10 language pairs, effectively mitigating the sentence-level coverage issue in document-level translation. Experimentation on discourse phenomena has demonstrated that our document-level translation approach significantly improves translation quality, both in terms of BLEU score and discourse coherence.

{{</citation>}}


### (77/132) Top in Chinese Data Processing: English Code Models (Linghan Zheng et al., 2024)

{{<citation>}}

Linghan Zheng, Hui Liu, Xiaojun Lin, Jiayuan Dong, Yue Sheng, Gang Shi, Zhiwei Liu, Hongwei Chen. (2024)  
**Top in Chinese Data Processing: English Code Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.10286v1)  

---


**ABSTRACT**  
While the alignment between tasks and training corpora is a fundamental consensus in the application of language models, our series of experiments and the metrics we designed reveal that code-based Large Language Models (LLMs) significantly outperform models trained on data that is closely matched to the tasks in non-coding Chinese tasks. Moreover, in tasks high sensitivity to Chinese hallucinations, models exhibiting fewer linguistic features of the Chinese language achieve better performance. Our experimental results can be easily replicated in Chinese data processing tasks, such as preparing data for Retrieval-Augmented Generation (RAG), by simply replacing the base model with a code-based model. Additionally, our research offers a distinct perspective for discussion on the philosophical "Chinese Room" thought experiment.

{{</citation>}}


### (78/132) Incremental Extractive Opinion Summarization Using Cover Trees (Somnath Basu Roy Chowdhury et al., 2024)

{{<citation>}}

Somnath Basu Roy Chowdhury, Nicholas Monath, Avinava Dubey, Manzil Zaheer, Andrew McCallum, Amr Ahmed, Snigdha Chaturvedi. (2024)  
**Incremental Extractive Opinion Summarization Using Cover Trees**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2401.08047v1)  

---


**ABSTRACT**  
Extractive opinion summarization involves automatically producing a summary of text about an entity (e.g., a product's reviews) by extracting representative sentences that capture prevalent opinions in the review set. Typically, in online marketplaces user reviews accrue over time, and opinion summaries need to be updated periodically to provide customers with up-to-date information. In this work, we study the task of extractive opinion summarization in an incremental setting, where the underlying review set evolves over time. Many of the state-of-the-art extractive opinion summarization approaches are centrality-based, such as CentroidRank. CentroidRank performs extractive summarization by selecting a subset of review sentences closest to the centroid in the representation space as the summary. However, these methods are not capable of operating efficiently in an incremental setting, where reviews arrive one at a time. In this paper, we present an efficient algorithm for accurately computing the CentroidRank summaries in an incremental setting. Our approach, CoverSumm, relies on indexing review representations in a cover tree and maintaining a reservoir of candidate summary review sentences. CoverSumm's efficacy is supported by a theoretical and empirical analysis of running time. Empirically, on a diverse collection of data (both real and synthetically created to illustrate scaling considerations), we demonstrate that CoverSumm is up to 25x faster than baseline methods, and capable of adapting to nuanced changes in data distribution. We also conduct human evaluations of the generated summaries and find that CoverSumm is capable of producing informative summaries consistent with the underlying review set.

{{</citation>}}


### (79/132) Enhancing Robustness of LLM-Synthetic Text Detectors for Academic Writing: A Comprehensive Analysis (Zhicheng Dou et al., 2024)

{{<citation>}}

Zhicheng Dou, Yuchen Guo, Ching-Chun Chang, Huy H. Nguyen, Isao Echizen. (2024)  
**Enhancing Robustness of LLM-Synthetic Text Detectors for Academic Writing: A Comprehensive Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-4, Transformer  
[Paper Link](http://arxiv.org/abs/2401.08046v1)  

---


**ABSTRACT**  
The emergence of large language models (LLMs), such as Generative Pre-trained Transformer 4 (GPT-4) used by ChatGPT, has profoundly impacted the academic and broader community. While these models offer numerous advantages in terms of revolutionizing work and study methods, they have also garnered significant attention due to their potential negative consequences. One example is generating academic reports or papers with little to no human contribution. Consequently, researchers have focused on developing detectors to address the misuse of LLMs. However, most existing methods prioritize achieving higher accuracy on restricted datasets, neglecting the crucial aspect of generalizability. This limitation hinders their practical application in real-life scenarios where reliability is paramount. In this paper, we present a comprehensive analysis of the impact of prompts on the text generated by LLMs and highlight the potential lack of robustness in one of the current state-of-the-art GPT detectors. To mitigate these issues concerning the misuse of LLMs in academic writing, we propose a reference-based Siamese detector named Synthetic-Siamese which takes a pair of texts, one as the inquiry and the other as the reference. Our method effectively addresses the lack of robustness of previous detectors (OpenAI detector and DetectGPT) and significantly improves the baseline performances in realistic academic writing scenarios by approximately 67% to 95%.

{{</citation>}}


### (80/132) Calpric: Inclusive and Fine-grain Labeling of Privacy Policies with Crowdsourcing and Active Learning (Wenjun Qiu et al., 2024)

{{<citation>}}

Wenjun Qiu, David Lie, Lisa Austin. (2024)  
**Calpric: Inclusive and Fine-grain Labeling of Privacy Policies with Crowdsourcing and Active Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs-HC, cs-LG, cs.CL  
Keywords: Active Learning, Amazon  
[Paper Link](http://arxiv.org/abs/2401.08038v1)  

---


**ABSTRACT**  
A significant challenge to training accurate deep learning models on privacy policies is the cost and difficulty of obtaining a large and comprehensive set of training data. To address these challenges, we present Calpric , which combines automatic text selection and segmentation, active learning and the use of crowdsourced annotators to generate a large, balanced training set for privacy policies at low cost. Automated text selection and segmentation simplifies the labeling task, enabling untrained annotators from crowdsourcing platforms, like Amazon's Mechanical Turk, to be competitive with trained annotators, such as law students, and also reduces inter-annotator agreement, which decreases labeling cost. Having reliable labels for training enables the use of active learning, which uses fewer training samples to efficiently cover the input space, further reducing cost and improving class and data category balance in the data set. The combination of these techniques allows Calpric to produce models that are accurate over a wider range of data categories, and provide more detailed, fine-grain labels than previous work. Our crowdsourcing process enables Calpric to attain reliable labeled data at a cost of roughly $0.92-$1.71 per labeled text segment. Calpric 's training process also generates a labeled data set of 16K privacy policy text segments across 9 Data categories with balanced positive and negative samples.

{{</citation>}}


### (81/132) JustiLM: Few-shot Justification Generation for Explainable Fact-Checking of Real-world Claims (Fengzhu Zeng et al., 2024)

{{<citation>}}

Fengzhu Zeng, Wei Gao. (2024)  
**JustiLM: Few-shot Justification Generation for Explainable Fact-Checking of Real-world Claims**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Fact-Checking  
[Paper Link](http://arxiv.org/abs/2401.08026v1)  

---


**ABSTRACT**  
Justification is an explanation that supports the veracity assigned to a claim in fact-checking. However, the task of justification generation is previously oversimplified as summarization of fact-check article authored by fact-checkers. Therefore, we propose a realistic approach to generate justification based on retrieved evidence. We present a new benchmark dataset called ExClaim for \underline{Ex}plainable fact-checking of real-world \underline{Claim}s, and introduce JustiLM, a novel few-shot \underline{Justi}fication generation based on retrieval-augmented \underline{L}anguage \underline{M}odel by using fact-check articles as auxiliary resource during training only. Experiments show that JustiLM achieves promising performance in justification generation compared to strong baselines, and can also enhance veracity classification with a straightforward extension.

{{</citation>}}


## cs.SI (3)



### (82/132) Topic Diversity and Conspiracy Theories Shape Engagement with COVID-19 Misinformation on X/Twitter (Yuwei Chuai et al., 2024)

{{<citation>}}

Yuwei Chuai, Jichang Zhao, Gabriele Lenzini. (2024)  
**Topic Diversity and Conspiracy Theories Shape Engagement with COVID-19 Misinformation on X/Twitter**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.08832v1)  

---


**ABSTRACT**  
The engagement with online health misinformation, particularly during COVID-19, poses unprecedented threats to societal well-being. The susceptibility to misinformation is heightened within a multi-topic context during health crises. This paper addresses a critical gap in understanding online engagement with multi-topic misinformation related to COVID-19. We conduct a comprehensive analysis of 7273 fact-checked source news claims related to COVID-19 and their corresponding social engagement on X/Twitter through the lens of topic diversity and conspiracy theories. Our analysis yields several key findings: (i) False news, especially when accompanied by conspiracy theories, exhibits higher topic diversity compared to true news. (ii) In terms of engagement from source claims to online posts, false news has a longer lifetime and receives more posts on X/Twitter compared to true news. Additionally, the integration of conspiracy theories is associated with a longer lifetime of COVID-19 misinformation. (iii) News posts characterized by heightened topic diversity receive increased social engagement on X/Twitter in terms of reposts, likes, and replies. However, the effect of topic diversity is moderated by the news veracity. High topic diversity is linked to more engagement with true news posts compared to false news posts. (iiii) The integration of conspiracy theories is linked to more social engagement with misinformation on X/Twitter. False news posts that contain conspiracy theories, on average, receive 40.8% more reposts, 45.2% more likes, and 44.1% more replies compared to false news posts without conspiracy theories. These findings offer insights into understanding the engagement with multi-topic misinformation during health crises and highlight the importance of considering topic diversity and conspiracy theories in developing targeted interventions.

{{</citation>}}


### (83/132) Moral Values Underpinning COVID-19 Online Communication Patterns (Julie Jiang et al., 2024)

{{<citation>}}

Julie Jiang, Luca Luceri, Emilio Ferrara. (2024)  
**Moral Values Underpinning COVID-19 Online Communication Patterns**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.08789v1)  

---


**ABSTRACT**  
The COVID-19 pandemic has triggered profound societal changes, extending beyond its health impacts to the moralization of behaviors. Leveraging insights from moral psychology, this study delves into the moral fabric shaping online discussions surrounding COVID-19 over a span of nearly two years. Our investigation identifies four distinct user groups characterized by differences in morality, political ideology, and communication styles. We underscore the intricate relationship between moral differences and political ideologies, revealing a nuanced picture where moral orientations do not rigidly separate users politically. Furthermore, we uncover patterns of moral homophily within the social network, highlighting the existence of one potential moral echo chamber. Analyzing the moral themes embedded in messages, we observe that messages featuring moral foundations not typically favored by their authors, as well as those incorporating multiple moral foundations, resonate more effectively with out-group members. This research contributes valuable insights into the complex interplay between moral foundations, communication dynamics, and network structures on Twitter.

{{</citation>}}


### (84/132) Interpreting Node Embedding Distances Through $n$-order Proximity Neighbourhoods (Dougal Shakespeare et al., 2024)

{{<citation>}}

Dougal Shakespeare, Camille Roth. (2024)  
**Interpreting Node Embedding Distances Through $n$-order Proximity Neighbourhoods**  

---
Primary Category: cs.SI  
Categories: 68R10, cs-SI, cs.SI  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.08236v1)  

---


**ABSTRACT**  
In the field of node representation learning the task of interpreting latent dimensions has become a prominent, well-studied research topic. The contribution of this work focuses on appraising the interpretability of another rarely-exploited feature of node embeddings increasingly utilised in recommendation and consumption diversity studies: inter-node embedded distances. Introducing a new method to measure how understandable the distances between nodes are, our work assesses how well the proximity weights derived from a network before embedding relate to the node closeness measurements after embedding. Testing several classical node embedding models, our findings reach a conclusion familiar to practitioners albeit rarely cited in literature - the matrix factorisation model SVD is the most interpretable through 1, 2 and even higher-order proximities.

{{</citation>}}


## cs.LG (18)



### (85/132) Stochastic Subnetwork Annealing: A Regularization Technique for Fine Tuning Pruned Subnetworks (Tim Whitaker et al., 2024)

{{<citation>}}

Tim Whitaker, Darrell Whitley. (2024)  
**Stochastic Subnetwork Annealing: A Regularization Technique for Fine Tuning Pruned Subnetworks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2401.08830v1)  

---


**ABSTRACT**  
Pruning methods have recently grown in popularity as an effective way to reduce the size and computational complexity of deep neural networks. Large numbers of parameters can be removed from trained models with little discernible loss in accuracy after a small number of continued training epochs. However, pruning too many parameters at once often causes an initial steep drop in accuracy which can undermine convergence quality. Iterative pruning approaches mitigate this by gradually removing a small number of parameters over multiple epochs. However, this can still lead to subnetworks that overfit local regions of the loss landscape. We introduce a novel and effective approach to tuning subnetworks through a regularization technique we call Stochastic Subnetwork Annealing. Instead of removing parameters in a discrete manner, we instead represent subnetworks with stochastic masks where each parameter has a probabilistic chance of being included or excluded on any given forward pass. We anneal these probabilities over time such that subnetwork structure slowly evolves as mask values become more deterministic, allowing for a smoother and more robust optimization of subnetworks at high levels of sparsity.

{{</citation>}}


### (86/132) AiGen-FoodReview: A Multimodal Dataset of Machine-Generated Restaurant Reviews and Images on Social Media (Alessandro Gambetti et al., 2024)

{{<citation>}}

Alessandro Gambetti, Qiwei Han. (2024)  
**AiGen-FoodReview: A Multimodal Dataset of Machine-Generated Restaurant Reviews and Images on Social Media**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: AI, GPT, GPT-4, Language Model, Social Media  
[Paper Link](http://arxiv.org/abs/2401.08825v1)  

---


**ABSTRACT**  
Online reviews in the form of user-generated content (UGC) significantly impact consumer decision-making. However, the pervasive issue of not only human fake content but also machine-generated content challenges UGC's reliability. Recent advances in Large Language Models (LLMs) may pave the way to fabricate indistinguishable fake generated content at a much lower cost. Leveraging OpenAI's GPT-4-Turbo and DALL-E-2 models, we craft AiGen-FoodReview, a multi-modal dataset of 20,144 restaurant review-image pairs divided into authentic and machine-generated. We explore unimodal and multimodal detection models, achieving 99.80% multimodal accuracy with FLAVA. We use attributes from readability and photographic theories to score reviews and images, respectively, demonstrating their utility as hand-crafted features in scalable and interpretable detection models, with comparable performance. The paper contributes by open-sourcing the dataset and releasing fake review detectors, recommending its use in unimodal and multimodal fake review detection tasks, and evaluating linguistic and visual features in synthetic versus authentic data.

{{</citation>}}


### (87/132) PUPAE: Intuitive and Actionable Explanations for Time Series Anomalies (Audrey Der et al., 2024)

{{<citation>}}

Audrey Der, Chin-Chia Michael Yeh, Yan Zheng, Junpeng Wang, Zhongfang Zhuang, Liang Wang, Wei Zhang, Eamonn J. Keogh. (2024)  
**PUPAE: Intuitive and Actionable Explanations for Time Series Anomalies**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.09489v1)  

---


**ABSTRACT**  
In recent years there has been significant progress in time series anomaly detection. However, after detecting an (perhaps tentative) anomaly, can we explain it? Such explanations would be useful to triage anomalies. For example, in an oil refinery, should we respond to an anomaly by dispatching a hydraulic engineer, or an intern to replace the battery on a sensor? There have been some parallel efforts to explain anomalies, however many proposed techniques produce explanations that are indirect, and often seem more complex than the anomaly they seek to explain. Our review of the literature/checklists/user-manuals used by frontline practitioners in various domains reveals an interesting near-universal commonality. Most practitioners discuss, explain and report anomalies in the following format: The anomaly would be like normal data A, if not for the corruption B. The reader will appreciate that is a type of counterfactual explanation. In this work we introduce a domain agnostic counterfactual explanation technique to produce explanations for time series anomalies. As we will show, our method can produce both visual and text-based explanations that are objectively correct, intuitive and in many circumstances, directly actionable.

{{</citation>}}


### (88/132) Explaining Time Series via Contrastive and Locally Sparse Perturbations (Zichuan Liu et al., 2024)

{{<citation>}}

Zichuan Liu, Yingying Zhang, Tianchun Wang, Zefan Wang, Dongsheng Luo, Mengnan Du, Min Wu, Yi Wang, Chunlin Chen, Lunting Fan, Qingsong Wen. (2024)  
**Explaining Time Series via Contrastive and Locally Sparse Perturbations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.08552v1)  

---


**ABSTRACT**  
Explaining multivariate time series is a compound challenge, as it requires identifying important locations in the time series and matching complex temporal patterns. Although previous saliency-based methods addressed the challenges, their perturbation may not alleviate the distribution shift issue, which is inevitable especially in heterogeneous samples. We present ContraLSP, a locally sparse model that introduces counterfactual samples to build uninformative perturbations but keeps distribution using contrastive learning. Furthermore, we incorporate sample-specific sparse gates to generate more binary-skewed and smooth masks, which easily integrate temporal trends and select the salient features parsimoniously. Empirical studies on both synthetic and real-world datasets show that ContraLSP outperforms state-of-the-art models, demonstrating a substantial improvement in explanation quality for time series data. The code is available for review: https://anonymous.4open.science/r/ContraLSP-1146/

{{</citation>}}


### (89/132) DiConStruct: Causal Concept-based Explanations through Black-Box Distillation (Ricardo Moreira et al., 2024)

{{<citation>}}

Ricardo Moreira, Jacopo Bono, Mário Cardoso, Pedro Saleiro, Mário A. T. Figueiredo, Pedro Bizarro. (2024)  
**DiConStruct: Causal Concept-based Explanations through Black-Box Distillation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-HC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08534v1)  

---


**ABSTRACT**  
Model interpretability plays a central role in human-AI decision-making systems. Ideally, explanations should be expressed using human-interpretable semantic concepts. Moreover, the causal relations between these concepts should be captured by the explainer to allow for reasoning about the explanations. Lastly, explanation methods should be efficient and not compromise the performance of the predictive task. Despite the rapid advances in AI explainability in recent years, as far as we know to date, no method fulfills these three properties. Indeed, mainstream methods for local concept explainability do not produce causal explanations and incur a trade-off between explainability and prediction performance. We present DiConStruct, an explanation method that is both concept-based and causal, with the goal of creating more interpretable local explanations in the form of structural causal models and concept attributions. Our explainer works as a distillation model to any black-box machine learning model by approximating its predictions while producing the respective explanations. Because of this, DiConStruct generates explanations efficiently while not impacting the black-box prediction task. We validate our method on an image dataset and a tabular dataset, showing that DiConStruct approximates the black-box models with higher fidelity than other concept explainability baselines, while providing explanations that include the causal relations between the concepts.

{{</citation>}}


### (90/132) Beyond Weisfeiler-Lehman: A Quantitative Framework for GNN Expressiveness (Bohang Zhang et al., 2024)

{{<citation>}}

Bohang Zhang, Jingchu Gai, Yiheng Du, Qiwei Ye, Di He, Liwei Wang. (2024)  
**Beyond Weisfeiler-Lehman: A Quantitative Framework for GNN Expressiveness**  

---
Primary Category: cs.LG  
Categories: cs-DM, cs-DS, cs-LG, cs.LG, math-CO  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.08514v1)  

---


**ABSTRACT**  
Designing expressive Graph Neural Networks (GNNs) is a fundamental topic in the graph learning community. So far, GNN expressiveness has been primarily assessed via the Weisfeiler-Lehman (WL) hierarchy. However, such an expressivity measure has notable limitations: it is inherently coarse, qualitative, and may not well reflect practical requirements (e.g., the ability to encode substructures). In this paper, we introduce a unified framework for quantitatively studying the expressiveness of GNN architectures, addressing all the above limitations. Specifically, we identify a fundamental expressivity measure termed homomorphism expressivity, which quantifies the ability of GNN models to count graphs under homomorphism. Homomorphism expressivity offers a complete and practical assessment tool: the completeness enables direct expressivity comparisons between GNN models, while the practicality allows for understanding concrete GNN abilities such as subgraph counting. By examining four classes of prominent GNNs as case studies, we derive simple, unified, and elegant descriptions of their homomorphism expressivity for both invariant and equivariant settings. Our results provide novel insights into a series of previous work, unify the landscape of different subareas in the community, and settle several open questions. Empirically, extensive experiments on both synthetic and real-world tasks verify our theory, showing that the practical performance of GNN models aligns well with the proposed metric.

{{</citation>}}


### (91/132) X Hacking: The Threat of Misguided AutoML (Rahul Sharma et al., 2024)

{{<citation>}}

Rahul Sharma, Sergey Redyuk, Sumantrak Mukherjee, Andrea Sipka, Sebastian Vollmer, David Selby. (2024)  
**X Hacking: The Threat of Misguided AutoML**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08513v1)  

---


**ABSTRACT**  
Explainable AI (XAI) and interpretable machine learning methods help to build trust in model predictions and derived insights, yet also present a perverse incentive for analysts to manipulate XAI metrics to support pre-specified conclusions. This paper introduces the concept of X-hacking, a form of p-hacking applied to XAI metrics such as Shap values. We show how an automated machine learning pipeline can be used to search for 'defensible' models that produce a desired explanation while maintaining superior predictive performance to a common baseline. We formulate the trade-off between explanation and accuracy as a multi-objective optimization problem and illustrate the feasibility and severity of X-hacking empirically on familiar real-world datasets. Finally, we suggest possible methods for detection and prevention, and discuss ethical implications for the credibility and reproducibility of XAI research.

{{</citation>}}


### (92/132) Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering (Tal Ridnik et al., 2024)

{{<citation>}}

Tal Ridnik, Dedy Kredo, Itamar Friedman. (2024)  
**Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs-SE, cs.LG  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.08500v1)  

---


**ABSTRACT**  
Code generation problems differ from common natural language problems - they require matching the exact syntax of the target language, identifying happy paths and edge cases, paying attention to numerous small details in the problem spec, and addressing other code-specific issues and requirements. Hence, many of the optimizations and tricks that have been successful in natural language generation may not be effective for code tasks. In this work, we propose a new approach to code generation by LLMs, which we call AlphaCodium - a test-based, multi-stage, code-oriented iterative flow, that improves the performances of LLMs on code problems. We tested AlphaCodium on a challenging code generation dataset called CodeContests, which includes competitive programming problems from platforms such as Codeforces. The proposed flow consistently and significantly improves results. On the validation set, for example, GPT-4 accuracy (pass@5) increased from 19% with a single well-designed direct prompt to 44% with the AlphaCodium flow. Many of the principles and best practices acquired in this work, we believe, are broadly applicable to general code generation tasks. Full implementation is available at: https://github.com/Codium-ai/AlphaCodium

{{</citation>}}


### (93/132) Solving Continual Offline Reinforcement Learning with Decision Transformer (Kaixin Huang et al., 2024)

{{<citation>}}

Kaixin Huang, Li Shen, Chen Zhao, Chun Yuan, Dacheng Tao. (2024)  
**Solving Continual Offline Reinforcement Learning with Decision Transformer**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2401.08478v1)  

---


**ABSTRACT**  
Continuous offline reinforcement learning (CORL) combines continuous and offline reinforcement learning, enabling agents to learn multiple tasks from static datasets without forgetting prior tasks. However, CORL faces challenges in balancing stability and plasticity. Existing methods, employing Actor-Critic structures and experience replay (ER), suffer from distribution shifts, low efficiency, and weak knowledge-sharing. We aim to investigate whether Decision Transformer (DT), another offline RL paradigm, can serve as a more suitable offline continuous learner to address these issues. We first compare AC-based offline algorithms with DT in the CORL framework. DT offers advantages in learning efficiency, distribution shift mitigation, and zero-shot generalization but exacerbates the forgetting problem during supervised parameter updates. We introduce multi-head DT (MH-DT) and low-rank adaptation DT (LoRA-DT) to mitigate DT's forgetting problem. MH-DT stores task-specific knowledge using multiple heads, facilitating knowledge sharing with common components. It employs distillation and selective rehearsal to enhance current task learning when a replay buffer is available. In buffer-unavailable scenarios, LoRA-DT merges less influential weights and fine-tunes DT's decisive MLP layer to adapt to the current task. Extensive experiments on MoJuCo and Meta-World benchmarks demonstrate that our methods outperform SOTA CORL baselines and showcase enhanced learning capabilities and superior memory efficiency.

{{</citation>}}


### (94/132) Bayes Conditional Distribution Estimation for Knowledge Distillation Based on Conditional Mutual Information (Linfeng Ye et al., 2024)

{{<citation>}}

Linfeng Ye, Shayan Mohajer Hamidi, Renhao Tan, En-Hui Yang. (2024)  
**Bayes Conditional Distribution Estimation for Knowledge Distillation Based on Conditional Mutual Information**  

---
Primary Category: cs.LG  
Categories: 68T30, I-2-6, cs-CV, cs-IT, cs-LG, cs.LG, math-IT  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2401.08732v1)  

---


**ABSTRACT**  
It is believed that in knowledge distillation (KD), the role of the teacher is to provide an estimate for the unknown Bayes conditional probability distribution (BCPD) to be used in the student training process. Conventionally, this estimate is obtained by training the teacher using maximum log-likelihood (MLL) method. To improve this estimate for KD, in this paper we introduce the concept of conditional mutual information (CMI) into the estimation of BCPD and propose a novel estimator called the maximum CMI (MCMI) method. Specifically, in MCMI estimation, both the log-likelihood and CMI of the teacher are simultaneously maximized when the teacher is trained. Through Eigen-CAM, it is further shown that maximizing the teacher's CMI value allows the teacher to capture more contextual information in an image cluster. Via conducting a thorough set of experiments, we show that by employing a teacher trained via MCMI estimation rather than one trained via MLL estimation in various state-of-the-art KD frameworks, the student's classification accuracy consistently increases, with the gain of up to 3.32\%. This suggests that the teacher's BCPD estimate provided by MCMI method is more accurate than that provided by MLL method. In addition, we show that such improvements in the student's accuracy are more drastic in zero-shot and few-shot settings. Notably, the student's accuracy increases with the gain of up to 5.72\% when 5\% of the training samples are available to the student (few-shot), and increases from 0\% to as high as 84\% for an omitted class (zero-shot). The code is available at \url{https://github.com/iclr2024mcmi/ICLRMCMI}.

{{</citation>}}


### (95/132) MA2GCN: Multi Adjacency relationship Attention Graph Convolutional Networks for Traffic Prediction using Trajectory data (Zhengke Sun et al., 2024)

{{<citation>}}

Zhengke Sun, Yuliang Ma. (2024)  
**MA2GCN: Multi Adjacency relationship Attention Graph Convolutional Networks for Traffic Prediction using Trajectory data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2401.08727v2)  

---


**ABSTRACT**  
The problem of traffic congestion not only causes a large amount of economic losses, but also seriously endangers the urban environment. Predicting traffic congestion has important practical significance. So far, most studies have been based on historical data from sensors placed on different roads to predict future traffic flow and speed, to analyze the traffic congestion conditions of a certain road segment. However, due to the fixed position of sensors, it is difficult to mine new information. On the other hand, vehicle trajectory data is more flexible and can extract traffic information as needed. Therefore, we proposed a new traffic congestion prediction model - Multi Adjacency relationship Attention Graph Convolutional Networks(MA2GCN). This model transformed vehicle trajectory data into graph structured data in grid form, and proposed a vehicle entry and exit matrix based on the mobility between different grids. At the same time, in order to improve the performance of the model, this paper also built a new adaptive adjacency matrix generation method and adjacency matrix attention module. This model mainly used gated temporal convolution and graph convolution to extract temporal and spatial information, respectively. Compared with multiple baselines, our model achieved the best performance on Shanghai taxi GPS trajectory dataset. The code is available at https://github.com/zachysun/Taxi_Traffic_Benchmark.

{{</citation>}}


### (96/132) Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference (Jinghan Yao et al., 2024)

{{<citation>}}

Jinghan Yao, Quentin Anthony, Aamir Shafi, Hari Subramoni, Dhabaleswar K., Panda. (2024)  
**Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2401.08383v2)  

---


**ABSTRACT**  
In large language models like the Generative Pre-trained Transformer, the Mixture of Experts paradigm has emerged as a powerful technique for enhancing model expressiveness and accuracy. However, deploying GPT MoE models for parallel inference on distributed systems presents significant challenges, primarily due to the extensive Alltoall communication required for expert routing and aggregation. This communication bottleneck exacerbates the already complex computational landscape, hindering the efficient utilization of high-performance computing resources. In this paper, we propose a lightweight optimization technique called ExFlow, to largely accelerate the inference of these MoE models. We take a new perspective on alleviating the communication overhead by exploiting the inter-layer expert affinity. Unlike previous methods, our solution can be directly applied to pre-trained MoE models without any fine-tuning or accuracy degradation. By proposing a context-coherent expert parallelism on distributed systems, our design only uses one Alltoall communication to deliver the same functionality while previous methods all require two Alltoalls. By carefully examining the conditional probability in tokens' routing across multiple layers, we proved that pre-trained GPT MoE models implicitly exhibit a strong inter-layer expert affinity. We then design an efficient integer programming model to capture such features and show that by properly placing the experts on corresponding GPUs, we can reduce up to 67% cross-GPU routing latency. Our solution beats the cutting-edge MoE implementations with experts from 8 to 64, with up to 2.2x improvement in inference throughput. We further provide a detailed study of how the model implicitly acquires this expert affinity at the very early training stage and how this affinity evolves and stabilizes during training.

{{</citation>}}


### (97/132) The Faiss library (Matthijs Douze et al., 2024)

{{<citation>}}

Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, Hervé Jégou. (2024)  
**The Faiss library**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs-SE, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08281v1)  

---


**ABSTRACT**  
Vector databases manage large collections of embedding vectors. As AI applications are growing rapidly, so are the number of embeddings that need to be stored and indexed. The Faiss library is dedicated to vector similarity search, a core functionality of vector databases. Faiss is a toolkit of indexing methods and related primitives used to search, cluster, compress and transform vectors. This paper first describes the tradeoff space of vector search, then the design principles of Faiss in terms of structure, approach to optimization and interfacing. We benchmark key features of the library and discuss a few selected applications to highlight its broad applicability.

{{</citation>}}


### (98/132) Enhancing Wind Speed and Wind Power Forecasting Using Shape-Wise Feature Engineering: A Novel Approach for Improved Accuracy and Robustness (Mulomba Mukendi Christian et al., 2024)

{{<citation>}}

Mulomba Mukendi Christian, Yun Seon Kim, Hyebong Choi, Jaeyoung Lee, SongHee You. (2024)  
**Enhancing Wind Speed and Wind Power Forecasting Using Shape-Wise Feature Engineering: A Novel Approach for Improved Accuracy and Robustness**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.08233v1)  

---


**ABSTRACT**  
Accurate prediction of wind speed and power is vital for enhancing the efficiency of wind energy systems. Numerous solutions have been implemented to date, demonstrating their potential to improve forecasting. Among these, deep learning is perceived as a revolutionary approach in the field. However, despite their effectiveness, the noise present in the collected data remains a significant challenge. This noise has the potential to diminish the performance of these algorithms, leading to inaccurate predictions. In response to this, this study explores a novel feature engineering approach. This approach involves altering the data input shape in both Convolutional Neural Network-Long Short-Term Memory (CNN-LSTM) and Autoregressive models for various forecasting horizons. The results reveal substantial enhancements in model resilience against noise resulting from step increases in data. The approach could achieve an impressive 83% accuracy in predicting unseen data up to the 24th steps. Furthermore, this method consistently provides high accuracy for short, mid, and long-term forecasts, outperforming the performance of individual models. These findings pave the way for further research on noise reduction strategies at different forecasting horizons through shape-wise feature engineering.

{{</citation>}}


### (99/132) LoMA: Lossless Compressed Memory Attention (Yumeng Wang et al., 2024)

{{<citation>}}

Yumeng Wang, Zhenyang Xiao. (2024)  
**LoMA: Lossless Compressed Memory Attention**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Attention, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09486v1)  

---


**ABSTRACT**  
The ability to handle long texts is one of the most important capabilities of Large Language Models (LLMs), but as the text length increases, the consumption of resources also increases dramatically. At present, reducing resource consumption by compressing the KV cache is a common approach. Although there are many existing compression methods, they share a common drawback: the compression is not lossless. That is, information is inevitably lost during the compression process. If the compression rate is high, the probability of losing important information increases dramatically. We propose a new method, Lossless Compressed Memory Attention (LoMA), which allows for lossless compression of information into special memory token KV pairs according to a set compression ratio. Our experiments have achieved remarkable results, demonstrating that LoMA can be efficiently trained and has very effective performance.

{{</citation>}}


### (100/132) Transferring Core Knowledge via Learngenes (Fu Feng et al., 2024)

{{<citation>}}

Fu Feng, Jing Wang, Xin Geng. (2024)  
**Transferring Core Knowledge via Learngenes**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.08139v1)  

---


**ABSTRACT**  
The pre-training paradigm fine-tunes the models trained on large-scale datasets to downstream tasks with enhanced performance. It transfers all knowledge to downstream tasks without discriminating which part is necessary or unnecessary, which may lead to negative transfer. In comparison, knowledge transfer in nature is much more efficient. When passing genetic information to descendants, ancestors encode only the essential knowledge into genes, which act as the medium. Inspired by that, we adopt a recent concept called ``learngene'' and refine its structures by mimicking the structures of natural genes. We propose the Genetic Transfer Learning (GTL) -- a framework to copy the evolutionary process of organisms into neural networks. GTL trains a population of networks, selects superior learngenes by tournaments, performs learngene mutations, and passes the learngenes to next generations. Finally, we successfully extract the learngenes of VGG11 and ResNet12. We show that the learngenes bring the descendant networks instincts and strong learning ability: with 20% parameters, the learngenes bring 12% and 16% improvements of accuracy on CIFAR-FS and miniImageNet. Besides, the learngenes have the scalability and adaptability on the downstream structure of networks and datasets. Overall, we offer a novel insight that transferring core knowledge via learngenes may be sufficient and efficient for neural networks.

{{</citation>}}


### (101/132) Machine Learning-Based Malicious Vehicle Detection for Security Threats and Attacks in Vehicle Ad-hoc Network (VANET) Communications (Thanh Nguyen Canh et al., 2024)

{{<citation>}}

Thanh Nguyen Canh, Xiem HoangVan. (2024)  
**Machine Learning-Based Malicious Vehicle Detection for Security Threats and Attacks in Vehicle Ad-hoc Network (VANET) Communications**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs-NI, cs.LG  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.08135v1)  

---


**ABSTRACT**  
With the rapid growth of Vehicle Ad-hoc Network (VANET) as a promising technology for efficient and reliable communication among vehicles and infrastructure, the security and integrity of VANET communications has become a critical concern. One of the significant threats to VANET is the presence of blackhole attacks, where malicious nodes disrupt the network's functionality and compromise data confidentiality, integrity, and availability. In this paper, we propose a machine learning-based approach for blackhole detection in VANET. To achieve this task, we first create a comprehensive dataset comprising normal and malicious traffic flows. Afterward, we study and define a promising set of features to discriminate the blackhole attacks. Finally, we evaluate various machine learning algorithms, including Gradient Boosting, Random Forest, Support Vector Machines, k-Nearest Neighbors, Gaussian Naive Bayes, and Logistic Regression. Experimental results demonstrate the effectiveness of these algorithms in distinguishing between normal and malicious nodes. Our findings also highlight the potential of machine learning based approach in enhancing the security of VANET by detecting and mitigating blackhole attacks.

{{</citation>}}


### (102/132) Transformer-based approach for Ethereum Price Prediction Using Crosscurrency correlation and Sentiment Analysis (Shubham Singh et al., 2024)

{{<citation>}}

Shubham Singh, Mayur Bhat. (2024)  
**Transformer-based approach for Ethereum Price Prediction Using Crosscurrency correlation and Sentiment Analysis**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-fin-PR  
Keywords: Sentiment Analysis, Transformer  
[Paper Link](http://arxiv.org/abs/2401.08077v1)  

---


**ABSTRACT**  
The research delves into the capabilities of a transformer-based neural network for Ethereum cryptocurrency price forecasting. The experiment runs around the hypothesis that cryptocurrency prices are strongly correlated with other cryptocurrencies and the sentiments around the cryptocurrency. The model employs a transformer architecture for several setups from single-feature scenarios to complex configurations incorporating volume, sentiment, and correlated cryptocurrency prices. Despite a smaller dataset and less complex architecture, the transformer model surpasses ANN and MLP counterparts on some parameters. The conclusion presents a hypothesis on the illusion of causality in cryptocurrency price movements driven by sentiments.

{{</citation>}}


## q-bio.QM (1)



### (103/132) Gene-associated Disease Discovery Powered by Large Language Models (Jiayu Chang et al., 2024)

{{<citation>}}

Jiayu Chang, Shiyu Wang, Chen Ling, Zhaohui Qin, Liang Zhao. (2024)  
**Gene-associated Disease Discovery Powered by Large Language Models**  

---
Primary Category: q-bio.QM  
Categories: cs-IR, q-bio-QM, q-bio.QM  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09490v1)  

---


**ABSTRACT**  
The intricate relationship between genetic variation and human diseases has been a focal point of medical research, evidenced by the identification of risk genes regarding specific diseases. The advent of advanced genome sequencing techniques has significantly improved the efficiency and cost-effectiveness of detecting these genetic markers, playing a crucial role in disease diagnosis and forming the basis for clinical decision-making and early risk assessment. To overcome the limitations of existing databases that record disease-gene associations from existing literature, which often lack real-time updates, we propose a novel framework employing Large Language Models (LLMs) for the discovery of diseases associated with specific genes. This framework aims to automate the labor-intensive process of sifting through medical literature for evidence linking genetic variations to diseases, thereby enhancing the efficiency of disease identification. Our approach involves using LLMs to conduct literature searches, summarize relevant findings, and pinpoint diseases related to specific genes. This paper details the development and application of our LLM-powered framework, demonstrating its potential in streamlining the complex process of literature retrieval and summarization to identify diseases associated with specific genetic variations.

{{</citation>}}


## cs.SE (5)



### (104/132) SpecGen: Automated Generation of Formal Program Specifications via Large Language Models (Lezhi Ma et al., 2024)

{{<citation>}}

Lezhi Ma, Shangqing Liu, Yi Li, Xiaofei Xie, Lei Bu. (2024)  
**SpecGen: Automated Generation of Formal Program Specifications via Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.08807v1)  

---


**ABSTRACT**  
In software development, formal program specifications play a crucial role in various stages. However, manually crafting formal program specifications is rather difficult, making the job time-consuming and labor-intensive. Moreover, it is even more challenging to write specifications that correctly and comprehensively describe the semantics of complex programs. To reduce the burden on software developers, automated specification generation methods have emerged. However, existing methods usually rely on predefined templates or grammar, making them struggle to accurately describe the behavior and functionality of complex real-world programs. To tackle this challenge, we introduce SpecGen, a novel technique for formal program specification generation based on Large Language Models. Our key insight is to overcome the limitations of existing methods by leveraging the code comprehension capability of LLMs. The process of SpecGen consists of two phases. The first phase employs a conversational approach that guides the LLM to generate appropriate specifications for a given program. The second phase, designed for where the LLM fails to generate correct specifications, applies four mutation operators to the model-generated specifications and selects verifiable specifications from the mutated ones through a novel heuristic selection strategy by assigning different weights of variants in an efficient manner. To evaluate the performance of SpecGen, we manually construct a dataset containing 120 test cases. Our experimental results demonstrate that SpecGen succeeds in generating verifiable specifications for 100 out of 120 programs, outperforming the existing purely LLM-based approaches and conventional specification generation tools. Further investigations on the quality of generated specifications indicate that SpecGen can comprehensively articulate the behaviors of the input program.

{{</citation>}}


### (105/132) PlayMyData: a curated dataset of multi-platform video games (Andrea D'Angelo et al., 2024)

{{<citation>}}

Andrea D'Angelo, Claudio Di Sipio, Cristiano Politowski, Riccardo Rubei. (2024)  
**PlayMyData: a curated dataset of multi-platform video games**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08561v2)  

---


**ABSTRACT**  
Being predominant in digital entertainment for decades, video games have been recognized as valuable software artifacts by the software engineering (SE) community just recently. Such an acknowledgment has unveiled several research opportunities, spanning from empirical studies to the application of AI techniques for classification tasks. In this respect, several curated game datasets have been disclosed for research purposes even though the collected data are insufficient to support the application of advanced models or to enable interdisciplinary studies. Moreover, the majority of those are limited to PC games, thus excluding notorious gaming platforms, e.g., PlayStation, Xbox, and Nintendo. In this paper, we propose PlayMyData, a curated dataset composed of 99,864 multi-platform games gathered by IGDB website. By exploiting a dedicated API, we collect relevant metadata for each game, e.g., description, genre, rating, gameplay video URLs, and screenshots. Furthermore, we enrich PlayMyData with the timing needed to complete each game by mining the HLTB website. To the best of our knowledge, this is the most comprehensive dataset in the domain that can be used to support different automated tasks in SE. More importantly, PlayMyData can be used to foster cross-domain investigations built on top of the provided multimedia data.

{{</citation>}}


### (106/132) CodeComplex: A Time-Complexity Dataset for Bilingual Source Codes (Seung-Yeop Baik et al., 2024)

{{<citation>}}

Seung-Yeop Baik, Mingi Jeon, Joonghyuk Hahn, Jungin Kim, Yo-Sub Han, Sang-Ki Ko. (2024)  
**CodeComplex: A Time-Complexity Dataset for Bilingual Source Codes**  

---
Primary Category: cs.SE  
Categories: cs-CC, cs-SE, cs.SE  
Keywords: BERT, ChatGPT, GPT, T5  
[Paper Link](http://arxiv.org/abs/2401.08719v1)  

---


**ABSTRACT**  
Analyzing the worst-case time complexity of a code is a crucial task in computer science and software engineering for ensuring the efficiency, reliability, and robustness of software systems. However, it is well-known that the problem of determining the worst-case time complexity of a given code written in general-purpose programming language is theoretically undecidable by the famous Halting problem proven by Alan Turing. Thus, we move towards more realistic scenarios where the inputs and outputs of a program exist. This allows us to discern the correctness of given codes, challenging to analyze their time complexity exhaustively. In response to this challenge, we introduce CodeComplex, a novel source code dataset where each code is manually annotated with a corresponding worst-case time complexity. CodeComplex comprises 4,900 Java codes and an equivalent number of Python codes, all sourced from programming competitions and annotated with complexity labels by a panel of algorithmic experts. To the best of our knowledge, CodeComplex stands as the most extensive code dataset tailored for predicting complexity. Subsequently, we present the outcomes of our experiments employing various baseline models, leveraging state-of-the-art neural models in code comprehension like CodeBERT, GraphCodeBERT, UniXcoder, PLBART, CodeT5, CodeT5+, and ChatGPT. We analyze how the dataset impacts the model's learning in predicting time complexity.

{{</citation>}}


### (107/132) Game Rewards Vulnerabilities: Software Vulnerability Detection with Zero-Sum Game and Prototype Learning (Xin-Cheng Wen et al., 2024)

{{<citation>}}

Xin-Cheng Wen, Cuiyun Gao, Xinchen Wang, Ruiqi Wang, Tao Zhang, Qing Liao. (2024)  
**Game Rewards Vulnerabilities: Software Vulnerability Detection with Zero-Sum Game and Prototype Learning**  

---
Primary Category: cs.SE  
Categories: cs-CR, cs-SE, cs.SE  
Keywords: Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2401.08131v1)  

---


**ABSTRACT**  
Recent years have witnessed a growing focus on automated software vulnerability detection. Notably, deep learning (DL)-based methods, which employ source code for the implicit acquisition of vulnerability patterns, have demonstrated superior performance compared to other approaches. However, the DL-based approaches are still hard to capture the vulnerability-related information from the whole code snippet, since the vulnerable parts usually account for only a small proportion. As evidenced by our experiments, the approaches tend to excessively emphasize semantic information, potentially leading to limited vulnerability detection performance in practical scenarios. First, they cannot well distinguish between the code snippets before (i.e., vulnerable code) and after (i.e., non-vulnerable code) developers' fixes due to the minimal code changes. Besides, substituting user-defined identifiers with placeholders (e.g., "VAR1" and "FUN1") in obvious performance degradation at up to 14.53% with respect to the F1 score. To mitigate these issues, we propose to leverage the vulnerable and corresponding fixed code snippets, in which the minimal changes can provide hints about semantic-agnostic features for vulnerability detection. In this paper, we propose a software vulneRability dEteCtion framework with zerO-sum game and prototype learNing, named RECON. In RECON, we propose a zero-sum game construction module. Distinguishing the vulnerable code from the corresponding fixed code is regarded as one player (i.e. Calibrator), while the conventional vulnerability detection is another player (i.e. Detector) in the zero-sum game. The goal is to capture the semantic-agnostic features of the first player for enhancing the second player's performance for vulnerability detection. Experiments on the public benchmark dataset show that RECON outperforms the state-of-the-art baseline by 6.29% in F1 score.

{{</citation>}}


### (108/132) A Study of Fairness Concerns in AI-based Mobile App Reviews (Ali Rezaei Nasab et al., 2024)

{{<citation>}}

Ali Rezaei Nasab, Maedeh Dashti, Mojtaba Shahin, Mansooreh Zahedi, Hourieh Khalajzadeh, Chetan Arora, Peng Liang. (2024)  
**A Study of Fairness Concerns in AI-based Mobile App Reviews**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CY, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08097v1)  

---


**ABSTRACT**  
With the growing application of AI-based systems in our lives and society, there is a rising need to ensure that AI-based systems are developed and used in a responsible way. Fairness is one of the socio-technical concerns that must be addressed in AI-based systems for this purpose. Unfair AI-based systems, particularly, unfair AI-based mobile apps, can pose difficulties for a significant proportion of the global populace. This paper aims to deeply analyze fairness concerns in AI-based app reviews. We first manually constructed a ground-truth dataset including a statistical sample of fairness and non-fairness reviews. Leveraging the ground-truth dataset, we then developed and evaluated a set of machine learning and deep learning classifiers that distinguish fairness reviews from non-fairness reviews. Our experiments show that our best-performing classifier can detect fairness reviews with a precision of 94%. We then applied the best-performing classifier on approximately 9.5M reviews collected from 108 AI-based apps and identified around 92K fairness reviews. While the fairness reviews appear in 23 app categories, we found that the 'communication' and 'social' app categories have the highest percentage of fairness reviews. Next, applying the K-means clustering technique to the 92K fairness reviews, followed by manual analysis, led to the identification of six distinct types of fairness concerns (e.g., 'receiving different quality of features and services in different platforms and devices' and 'lack of transparency and fairness in dealing with user-generated content'). Finally, the manual analysis of 2,248 app owners' responses to the fairness reviews identified six root causes (e.g., 'copyright issues', 'external factors', 'development cost') that app owners report to justify fairness concerns.

{{</citation>}}


## cs.DL (1)



### (109/132) Towards a Quality Indicator for Research Data publications and Research Software publications -- A vision from the Helmholtz Association (Wolfgang zu Castell et al., 2024)

{{<citation>}}

Wolfgang zu Castell, Doris Dransch, Guido Juckeland, Marcel Meistring, Bernadette Fritzsch, Ronny Gey, Britta Höpfner, Martin Köhler, Christian Meeßen, Hela Mehrtens, Felix Mühlbauer, Sirko Schindler, Thomas Schnicke, Roland Bertelmann. (2024)  
**Towards a Quality Indicator for Research Data publications and Research Software publications -- A vision from the Helmholtz Association**  

---
Primary Category: cs.DL  
Categories: cs-CY, cs-DL, cs.DL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08804v1)  

---


**ABSTRACT**  
Research data and software are widely accepted as an outcome of scientific work. However, in comparison to text-based publications, there is not yet an established process to assess and evaluate quality of research data and research software publications. This paper presents an attempt to fill this gap. Initiated by the Working Group Open Science of the Helmholtz Association the Task Group Helmholtz Quality Indicators for Data and Software Publications currently develops a quality indicator for research data and research software publications to be used within the Association. This report summarizes the vision of the group of what all contributes to such an indicator. The proposed approach relies on generic well-established concepts for quality criteria, such as the FAIR Principles and the COBIT Maturity Model. It does - on purpose - not limit itself to technical implementation possibilities to avoid using an existing metric for a new purpose. The intention of this paper is to share the current state for further discussion with all stakeholders, particularly with other groups also working on similar metrics but also with entities that use the metrics.

{{</citation>}}


## hep-ex (1)



### (110/132) Robust Anomaly Detection for Particle Physics Using Multi-Background Representation Learning (Abhijith Gandrakota et al., 2024)

{{<citation>}}

Abhijith Gandrakota, Lily Zhang, Aahlad Puli, Kyle Cranmer, Jennifer Ngadiuba, Rajesh Ranganath, Nhan Tran. (2024)  
**Robust Anomaly Detection for Particle Physics Using Multi-Background Representation Learning**  

---
Primary Category: hep-ex  
Categories: cs-LG, hep-ex, hep-ex, hep-ph, physics-data-an  
Keywords: Anomaly Detection, Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.08777v1)  

---


**ABSTRACT**  
Anomaly, or out-of-distribution, detection is a promising tool for aiding discoveries of new particles or processes in particle physics. In this work, we identify and address two overlooked opportunities to improve anomaly detection for high-energy physics. First, rather than train a generative model on the single most dominant background process, we build detection algorithms using representation learning from multiple background types, thus taking advantage of more information to improve estimation of what is relevant for detection. Second, we generalize decorrelation to the multi-background setting, thus directly enforcing a more complete definition of robustness for anomaly detection. We demonstrate the benefit of the proposed robust multi-background anomaly detection algorithms on a high-dimensional dataset of particle decays at the Large Hadron Collider.

{{</citation>}}


## cs.AI (6)



### (111/132) MMToM-QA: Multimodal Theory of Mind Question Answering (Chuanyang Jin et al., 2024)

{{<citation>}}

Chuanyang Jin, Yutong Wu, Jing Cao, Jiannan Xiang, Yen-Ling Kuo, Zhiting Hu, Tomer Ullman, Antonio Torralba, Joshua B. Tenenbaum, Tianmin Shu. (2024)  
**MMToM-QA: Multimodal Theory of Mind Question Answering**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.AI  
Keywords: GPT, GPT-4, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.08743v1)  

---


**ABSTRACT**  
Theory of Mind (ToM), the ability to understand people's minds, is an essential ingredient for developing machines with human-level social intelligence. Recent machine learning models, particularly large language models, seem to show some aspects of ToM understanding. However, existing ToM benchmarks use unimodal datasets - either video or text. Human ToM, on the other hand, is more than video or text understanding. People can flexibly reason about another person's mind based on conceptual representations (e.g., goals, beliefs, plans) extracted from any available data, which can include visual cues, linguistic narratives, or both. To address this, we introduce a multimodal Theory of Mind question answering (MMToM-QA) benchmark. MMToM-QA comprehensively evaluates machine ToM both on multimodal data and on different kinds of unimodal data about a person's activity in a household environment. To engineer multimodal ToM capacity, we propose a novel method, BIP-ALM (Bayesian Inverse Planning Accelerated by Language Models). BIP-ALM extracts unified representations from multimodal data and utilizes language models for scalable Bayesian inverse planning. We conducted a systematic comparison of human performance, BIP-ALM, and state-of-the-art models, including GPT-4. The experiments demonstrate that large language models and large multimodal models still lack robust ToM capacity. BIP-ALM, on the other hand, shows promising results, by leveraging the power of both model-based mental inference and language models.

{{</citation>}}


### (112/132) GATS: Gather-Attend-Scatter (Konrad Zolna et al., 2024)

{{<citation>}}

Konrad Zolna, Serkan Cabi, Yutian Chen, Eric Lau, Claudio Fantacci, Jurgis Pasukonis, Jost Tobias Springenberg, Sergio Gomez Colmenarejo. (2024)  
**GATS: Gather-Attend-Scatter**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08525v1)  

---


**ABSTRACT**  
As the AI community increasingly adopts large-scale models, it is crucial to develop general and flexible tools to integrate them. We introduce Gather-Attend-Scatter (GATS), a novel module that enables seamless combination of pretrained foundation models, both trainable and frozen, into larger multimodal networks. GATS empowers AI systems to process and generate information across multiple modalities at different rates. In contrast to traditional fine-tuning, GATS allows for the original component models to remain frozen, avoiding the risk of them losing important knowledge acquired during the pretraining phase. We demonstrate the utility and versatility of GATS with a few experiments across games, robotics, and multimodal input-output systems.

{{</citation>}}


### (113/132) Supporting Student Decisions on Learning Recommendations: An LLM-Based Chatbot with Knowledge Graph Contextualization for Conversational Explainability and Mentoring (Hasan Abu-Rasheed et al., 2024)

{{<citation>}}

Hasan Abu-Rasheed, Mohamad Hussam Abdulsalam, Christian Weber, Madjid Fathi. (2024)  
**Supporting Student Decisions on Learning Recommendations: An LLM-Based Chatbot with Knowledge Graph Contextualization for Conversational Explainability and Mentoring**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-HC, cs.AI  
Keywords: AI, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.08517v1)  

---


**ABSTRACT**  
Student commitment towards a learning recommendation is not separable from their understanding of the reasons it was recommended to them; and their ability to modify it based on that understanding. Among explainability approaches, chatbots offer the potential to engage the student in a conversation, similar to a discussion with a peer or a mentor. The capabilities of chatbots, however, are still not sufficient to replace a human mentor, despite the advancements of generative AI (GenAI) and large language models (LLM). Therefore, we propose an approach to utilize chatbots as mediators of the conversation and sources of limited and controlled generation of explanations, to harvest the potential of LLMs while reducing their potential risks at the same time. The proposed LLM-based chatbot supports students in understanding learning-paths recommendations. We use a knowledge graph (KG) as a human-curated source of information, to regulate the LLM's output through defining its prompt's context. A group chat approach is developed to connect students with human mentors, either on demand or in cases that exceed the chatbot's pre-defined tasks. We evaluate the chatbot with a user study, to provide a proof-of-concept and highlight the potential requirements and limitations of utilizing chatbots in conversational explainability.

{{</citation>}}


### (114/132) Reinforcement Learning for Conversational Question Answering over Knowledge Graph (Mi Wu, 2024)

{{<citation>}}

Mi Wu. (2024)  
**Reinforcement Learning for Conversational Question Answering over Knowledge Graph**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Knowledge Graph, QA, Question Answering, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.08460v1)  

---


**ABSTRACT**  
Conversational question answering (ConvQA) over law knowledge bases (KBs) involves answering multi-turn natural language questions about law and hope to find answers in the law knowledge base. Despite many methods have been proposed. Existing law knowledge base ConvQA model assume that the input question is clear and can perfectly reflect user's intention. However, in real world, the input questions are noisy and inexplict. This makes the model hard to find the correct answer in the law knowledge bases. In this paper, we try to use reinforcement learning to solve this problem. The reinforcement learning agent can automatically learn how to find the answer based on the input question and the conversation history, even when the input question is inexplicit. We test the proposed method on several real world datasets and the results show the effectivenss of the proposed model.

{{</citation>}}


### (115/132) PRewrite: Prompt Rewriting with Reinforcement Learning (Weize Kong et al., 2024)

{{<citation>}}

Weize Kong, Spurthi Amba Hombaiah, Mingyang Zhang, Qiaozhu Mei, Michael Bendersky. (2024)  
**PRewrite: Prompt Rewriting with Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.08189v1)  

---


**ABSTRACT**  
Prompt engineering is critical for the development of LLM-based applications. However, it is usually done manually in a "trial and error" fashion. This manual procedure can be time consuming, ineffective, and the generated prompts are, in a lot of cases, sub-optimal. Even for the prompts which seemingly work well, there is always a lingering question: can the prompts be made better with further modifications?   To address these questions, in this paper, we investigate prompt engineering automation. We consider a specific use case scenario in which developers/users have drafted initial prompts, but lack the time/expertise to optimize them. We propose PRewrite, an automated tool to rewrite these drafts and to generate highly effective new prompts. PRewrite is based on the Reinforcement Learning (RL) framework which allows for end-to-end optimization and our design allows the RL search to happen in a large action space. The automated tool leverages manually crafted prompts as starting points which makes the rewriting procedure more guided and efficient. The generated prompts are human readable, and self-explanatory, unlike some of those in previous works. We conducted extensive experiments on diverse datasets and found that the prompts generated with this new method not only outperform professionally crafted prompts, but also prompts generated with other previously proposed methods.

{{</citation>}}


### (116/132) Self-Imagine: Effective Unimodal Reasoning with Multimodal Models using Self-Imagination (Syeda Nahida Akter et al., 2024)

{{<citation>}}

Syeda Nahida Akter, Aman Madaan, Sangwu Lee, Yiming Yang, Eric Nyberg. (2024)  
**Self-Imagine: Effective Unimodal Reasoning with Multimodal Models using Self-Imagination**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.08025v1)  

---


**ABSTRACT**  
The potential of Vision-Language Models (\textsc{vlm}s) often remains underutilized in handling complex text-based problems, particularly when these problems could benefit from visual representation. Resonating with humans' ability to solve complex text-based problems by (1) creating a visual diagram from the problem and (2) deducing what steps they need to take to solve it, we propose \textsc{Self-Imagine}. We leverage a single Vision-Language Model (\textsc{vlm}) to generate a structured representation of the question using HTML, then render the HTML as an image, and finally use the same \vlm to answer the question using both the question and the image. Our approach does not require any additional training data or training. We evaluate our approach in three mathematics tasks and nine general-purpose reasoning tasks using state-of-the-art \textsc{vlm}. Our approach boosts the performance of \textsc{vlm} on all math tasks (\gsm: +4.62\%; \asdiv: +4.49\%; \svamp: +9.30\%) and the majority of the general-purpose reasoning tasks by 0.4\% to 13.20\% while achieving comparable performance in other tasks.   Code and data at https://github.com/snat1505027/self-imagine .

{{</citation>}}


## quant-ph (1)



### (117/132) Expanding Hardware-Efficiently Manipulable Hilbert Space via Hamiltonian Embedding (Jiaqi Leng et al., 2024)

{{<citation>}}

Jiaqi Leng, Joseph Li, Yuxiang Peng, Xiaodi Wu. (2024)  
**Expanding Hardware-Efficiently Manipulable Hilbert Space via Hamiltonian Embedding**  

---
Primary Category: quant-ph  
Categories: cs-CE, cs-NA, math-NA, quant-ph, quant-ph  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.08550v1)  

---


**ABSTRACT**  
Many promising quantum applications depend on the efficient quantum simulation of an exponentially large sparse Hamiltonian, a task known as sparse Hamiltonian simulation, which is fundamentally important in quantum computation. Although several theoretically appealing quantum algorithms have been proposed for this task, they typically require a black-box query model of the sparse Hamiltonian, rendering them impractical for near-term implementation on quantum devices.   In this paper, we propose a technique named Hamiltonian embedding. This technique simulates a desired sparse Hamiltonian by embedding it into the evolution of a larger and more structured quantum system, allowing for more efficient simulation through hardware-efficient operations. We conduct a systematic study of this new technique and demonstrate significant savings in computational resources for implementing prominent quantum applications. As a result, we can now experimentally realize quantum walks on complicated graphs (e.g., binary trees, glued-tree graphs), quantum spatial search, and the simulation of real-space Schr\"odinger equations on current trapped-ion and neutral-atom platforms. Given the fundamental role of Hamiltonian evolution in the design of quantum algorithms, our technique markedly expands the horizon of implementable quantum advantages in the NISQ era.

{{</citation>}}


## eess.SY (3)



### (118/132) Dual-Loop Robust Control of Biased Koopman Operator Model by Noisy Data of Nonlinear Systems (Anuj Pal et al., 2024)

{{<citation>}}

Anuj Pal, Tianyi He, Xiang Chen. (2024)  
**Dual-Loop Robust Control of Biased Koopman Operator Model by Noisy Data of Nonlinear Systems**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.08536v1)  

---


**ABSTRACT**  
The Koopman operator approach for data-driven control design of a nonlinear system is on the rise because of its capability to capture the behaviours of global dynamics. However, the measurement noises of inputs and outputs will bias the Koopman model identification and cause model mismatch from the actual nonlinear dynamics. The current work evaluates the bounds of the noise-induced model bias of the Koopman operator model and proposes a data-driven robust dual-loop control framework (Koopman based robust control-KROC) for the biased model. First, the model mismatch is found bounded under radial basis functions (RBF) and the bounded noises, and the bound of model mismatch is assessed. Second, the pitfalls of linear quadratic Gaussian (LQG) control based on the biased Koopman model of Van Der Pol oscillator are shown. Motivated from the pitfalls, the dual-loop control is proposed, which consist of an observer-based state-feedback control based on the nominal Koopman model and an additional robust loop to compensate model mismatch. A linear matrix inequality (LMI) is derived, which can guarantee robust stability and performance under bounded noises for the finite-dimensional Koopman operator model. Finally, the proposed framework is implemented to a nonlinear Van Der Pol oscillator to demonstrate enhanced control performance by the dual-loop robust control.

{{</citation>}}


### (119/132) Learning Stable Koopman Embeddings for Identification and Control (Fletcher Fan et al., 2024)

{{<citation>}}

Fletcher Fan, Bowen Yi, David Rye, Guodong Shi, Ian R. Manchester. (2024)  
**Learning Stable Koopman Embeddings for Identification and Control**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.08153v1)  

---


**ABSTRACT**  
This paper introduces new model parameterizations for learning dynamical systems from data via the Koopman operator, and studies their properties. Whereas most existing works on Koopman learning do not take into account the stability or stabilizability of the model -- two fundamental pieces of prior knowledge about a given system to be identified -- in this paper, we propose new classes of Koopman models that have built-in guarantees of these properties. These models are guaranteed to be stable or stabilizable via a novel {\em direct parameterization approach} that leads to {\em unconstrained} optimization problems with respect to their parameter sets. To explore the representational flexibility of these model sets, we establish novel theoretical connections between the stability of discrete-time Koopman embedding and contraction-based forms of nonlinear stability and stabilizability. The proposed approach is illustrated in applications to stable nonlinear system identification and imitation learning via stabilizable models. Simulation results empirically show that the learning approaches based on the proposed models outperform prior methods lacking stability guarantees.

{{</citation>}}


### (120/132) Bias-Compensated State of Charge and State of Health Joint Estimation for Lithium Iron Phosphate Batteries (Baozhao Yi et al., 2024)

{{<citation>}}

Baozhao Yi, Xinhao Du, Jiawei Zhang, Xiaogang Wu, Qiuhao Hu, Weiran Jiang, Xiaosong Hu, Ziyou Song. (2024)  
**Bias-Compensated State of Charge and State of Health Joint Estimation for Lithium Iron Phosphate Batteries**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.08136v1)  

---


**ABSTRACT**  
Accurate estimation of the state of charge (SOC) and state of health (SOH) is crucial for the safe and reliable operation of batteries. However, the measurement bias of voltage can highly deteriorate the estimation accuracy. One such example is the lithium iron phosphate (LFP) battery, which is highly prone to suffer from this issue owing to its flat open-circuit voltage curve. This work proposes a bias-compensated framework that reliably estimates the SOC and SOH of LFP batteries under the influence of voltage measurement bias. To validate the proposed approach, four LFP batteries are tested at various ambient temperatures and SOH conditions, with two different values of voltage measurement bias added. The results show that the bias-compensated algorithm achieves test errors that are less than 1.5% and 2% for SOC and SOH estimation, respectively. Additionally, the proposed approach outperforms the traditional estimation method that ignores the effects of voltage measurement bias.

{{</citation>}}


## cs.LO (1)



### (121/132) Algebraic Reasoning over Relational Structures (Jan Jurka et al., 2024)

{{<citation>}}

Jan Jurka, Stefan Milius, Henning Urbat. (2024)  
**Algebraic Reasoning over Relational Structures**  

---
Primary Category: cs.LO  
Categories: cs-LO, cs.LO  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.08445v1)  

---


**ABSTRACT**  
Many important computational structures involve an intricate interplay between algebraic features (given by operations on the underlying set) and relational features (taking account of notions such as order or distance). This paper investigates algebras over relational structures axiomatized by an infinitary Horn theory, which subsume, for example, partial algebras, various incarnations of ordered algebras, quantitative algebras introduced by Mardare, Panangaden, and Plotkin, and their recent extension to generalized metric spaces and lifted algebraic signatures by Mio, Sarkis, and Vignudelli. To this end, we develop the notion of clustered equation, which is inspired by Mardare et al.'s basic conditional equations in the theory of quantitative algebras, at the level of generality of arbitrary relational structures, and we prove it to be equivalent to an abstract categorical form of equation earlier introduced by Milius and Urbat. Our main results are a family of Birkhoff-type variety theorems (classifying the expressive power of clustered equations) and an exactness theorem (classifying abstract equations by a congruence property).

{{</citation>}}


## cs.SD (2)



### (122/132) From Coarse to Fine: Efficient Training for Audio Spectrogram Transformers (Jiu Feng et al., 2024)

{{<citation>}}

Jiu Feng, Mehmet Hamza Erol, Joon Son Chung, Arda Senocak. (2024)  
**From Coarse to Fine: Efficient Training for Audio Spectrogram Transformers**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.08415v1)  

---


**ABSTRACT**  
Transformers have become central to recent advances in audio classification. However, training an audio spectrogram transformer, e.g. AST, from scratch can be resource and time-intensive. Furthermore, the complexity of transformers heavily depends on the input audio spectrogram size. In this work, we aim to optimize AST training by linking to the resolution in the time-axis. We introduce multi-phase training of audio spectrogram transformers by connecting the seminal idea of coarse-to-fine with transformer models. To achieve this, we propose a set of methods for temporal compression. By employing one of these methods, the transformer model learns from lower-resolution (coarse) data in the initial phases, and then is fine-tuned with high-resolution data later in a curriculum learning strategy. Experimental results demonstrate that the proposed training mechanism for AST leads to improved (or on-par) performance with faster convergence, i.e. requiring fewer computational resources and less time. This approach is also generalizable to other AST-based methods regardless of their learning paradigms.

{{</citation>}}


### (123/132) Learning Disentangled Speech Representations with Contrastive Learning and Time-Invariant Retrieval (Yimin Deng et al., 2024)

{{<citation>}}

Yimin Deng, Huaizhen Tang, Xulong Zhang, Ning Cheng, Jing Xiao, Jianzong Wang. (2024)  
**Learning Disentangled Speech Representations with Contrastive Learning and Time-Invariant Retrieval**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.08096v2)  

---


**ABSTRACT**  
Voice conversion refers to transferring speaker identity with well-preserved content. Better disentanglement of speech representations leads to better voice conversion. Recent studies have found that phonetic information from input audio has the potential ability to well represent content. Besides, the speaker-style modeling with pre-trained models making the process more complex. To tackle these issues, we introduce a new method named "CTVC" which utilizes disentangled speech representations with contrastive learning and time-invariant retrieval. Specifically, a similarity-based compression module is used to facilitate a more intimate connection between the frame-level hidden features and linguistic information at phoneme-level. Additionally, a time-invariant retrieval is proposed for timbre extraction based on multiple segmentations and mutual information. Experimental results demonstrate that "CTVC" outperforms previous studies and improves the sound quality and similarity of converted results.

{{</citation>}}


## physics.comp-ph (1)



### (124/132) Enhancing Dynamical System Modeling through Interpretable Machine Learning Augmentations: A Case Study in Cathodic Electrophoretic Deposition (Christian Jacobsen et al., 2024)

{{<citation>}}

Christian Jacobsen, Jiayuan Dong, Mehdi Khalloufi, Xun Huan, Karthik Duraisamy, Maryam Akram, Wanjiao Liu. (2024)  
**Enhancing Dynamical System Modeling through Interpretable Machine Learning Augmentations: A Case Study in Cathodic Electrophoretic Deposition**  

---
Primary Category: physics.comp-ph  
Categories: cs-LG, physics-comp-ph, physics.comp-ph  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2401.08414v1)  

---


**ABSTRACT**  
We introduce a comprehensive data-driven framework aimed at enhancing the modeling of physical systems, employing inference techniques and machine learning enhancements. As a demonstrative application, we pursue the modeling of cathodic electrophoretic deposition (EPD), commonly known as e-coating. Our approach illustrates a systematic procedure for enhancing physical models by identifying their limitations through inference on experimental data and introducing adaptable model enhancements to address these shortcomings. We begin by tackling the issue of model parameter identifiability, which reveals aspects of the model that require improvement. To address generalizability , we introduce modifications which also enhance identifiability. However, these modifications do not fully capture essential experimental behaviors. To overcome this limitation, we incorporate interpretable yet flexible augmentations into the baseline model. These augmentations are parameterized by simple fully-connected neural networks (FNNs), and we leverage machine learning tools, particularly Neural Ordinary Differential Equations (Neural ODEs), to learn these augmentations. Our simulations demonstrate that the machine learning-augmented model more accurately captures observed behaviors and improves predictive accuracy. Nevertheless, we contend that while the model updates offer superior performance and capture the relevant physics, we can reduce off-line computational costs by eliminating certain dynamics without compromising accuracy or interpretability in downstream predictions of quantities of interest, particularly film thickness predictions. The entire process outlined here provides a structured approach to leverage data-driven methods. Firstly, it helps us comprehend the root causes of model inaccuracies, and secondly, it offers a principled method for enhancing model performance.

{{</citation>}}


## eess.IV (1)



### (125/132) Faster ISNet for Background Bias Mitigation on Deep Neural Networks (Pedro R. A. S. Bassi et al., 2024)

{{<citation>}}

Pedro R. A. S. Bassi, Sergio Decherchi, Andrea Cavalli. (2024)  
**Faster ISNet for Background Bias Mitigation on Deep Neural Networks**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-CY, cs-LG, eess-IV, eess.IV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.08409v1)  

---


**ABSTRACT**  
Image background features can constitute background bias (spurious correlations) and impact deep classifiers decisions, causing shortcut learning (Clever Hans effect) and reducing the generalization skill on real-world data. The concept of optimizing Layer-wise Relevance Propagation (LRP) heatmaps, to improve classifier behavior, was recently introduced by a neural network architecture named ISNet. It minimizes background relevance in LRP maps, to mitigate the influence of image background features on deep classifiers decisions, hindering shortcut learning and improving generalization. For each training image, the original ISNet produces one heatmap per possible class in the classification task, hence, its training time scales linearly with the number of classes. Here, we introduce reformulated architectures that allow the training time to become independent from this number, rendering the optimization process much faster. We challenged the enhanced models utilizing the MNIST dataset with synthetic background bias, and COVID-19 detection in chest X-rays, an application that is prone to shortcut learning due to background bias. The trained models minimized background attention and hindered shortcut learning, while retaining high accuracy. Considering external (out-of-distribution) test datasets, they consistently proved more accurate than multiple state-of-the-art deep neural network architectures, including a dedicated image semantic segmenter followed by a classifier. The architectures presented here represent a potentially massive improvement in training speed over the original ISNet, thus introducing LRP optimization into a gamut of applications that could not be feasibly handled by the original model.

{{</citation>}}


## eess.AS (2)



### (126/132) An Explainable Proxy Model for Multiabel Audio Segmentation (Théo Mariotte et al., 2024)

{{<citation>}}

Théo Mariotte, Antonio Almudévar, Marie Tahon, Alfonso Ortega. (2024)  
**An Explainable Proxy Model for Multiabel Audio Segmentation**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-LG, cs-SD, eess-AS, eess-SP, eess.AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08268v2)  

---


**ABSTRACT**  
Audio signal segmentation is a key task for automatic audio indexing. It consists of detecting the boundaries of class-homogeneous segments in the signal. In many applications, explainable AI is a vital process for transparency of decision-making with machine learning. In this paper, we propose an explainable multilabel segmentation model that solves speech activity (SAD), music (MD), noise (ND), and overlapped speech detection (OSD) simultaneously. This proxy uses the non-negative matrix factorization (NMF) to map the embedding used for the segmentation to the frequency domain. Experiments conducted on two datasets show similar performances as the pre-trained black box model while showing strong explainability features. Specifically, the frequency bins used for the decision can be easily identified at both the segment level (local explanations) and global level (class prototypes).

{{</citation>}}


### (127/132) ED-TTS: Multi-Scale Emotion Modeling using Cross-Domain Emotion Diarization for Emotional Speech Synthesis (Haobin Tang et al., 2024)

{{<citation>}}

Haobin Tang, Xulong Zhang, Ning Cheng, Jing Xiao, Jianzong Wang. (2024)  
**ED-TTS: Multi-Scale Emotion Modeling using Cross-Domain Emotion Diarization for Emotional Speech Synthesis**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2401.08166v1)  

---


**ABSTRACT**  
Existing emotional speech synthesis methods often utilize an utterance-level style embedding extracted from reference audio, neglecting the inherent multi-scale property of speech prosody. We introduce ED-TTS, a multi-scale emotional speech synthesis model that leverages Speech Emotion Diarization (SED) and Speech Emotion Recognition (SER) to model emotions at different levels. Specifically, our proposed approach integrates the utterance-level emotion embedding extracted by SER with fine-grained frame-level emotion embedding obtained from SED. These embeddings are used to condition the reverse process of the denoising diffusion probabilistic model (DDPM). Additionally, we employ cross-domain SED to accurately predict soft labels, addressing the challenge of a scarcity of fine-grained emotion-annotated datasets for supervising emotional TTS training.

{{</citation>}}


## stat.ML (1)



### (128/132) Statistical Test for Attention Map in Vision Transformer (Tomohiro Shiraishi et al., 2024)

{{<citation>}}

Tomohiro Shiraishi, Daiki Miwa, Teruyuki Katsuoka, Vo Nguyen Le Duy, Kouichi Taji, Ichiro Takeuchi. (2024)  
**Statistical Test for Attention Map in Vision Transformer**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2401.08169v2)  

---


**ABSTRACT**  
The Vision Transformer (ViT) demonstrates exceptional performance in various computer vision tasks. Attention is crucial for ViT to capture complex wide-ranging relationships among image patches, allowing the model to weigh the importance of image patches and aiding our understanding of the decision-making process. However, when utilizing the attention of ViT as evidence in high-stakes decision-making tasks such as medical diagnostics, a challenge arises due to the potential of attention mechanisms erroneously focusing on irrelevant regions. In this study, we propose a statistical test for ViT's attentions, enabling us to use the attentions as reliable quantitative evidence indicators for ViT's decision-making with a rigorously controlled error rate. Using the framework called selective inference, we quantify the statistical significance of attentions in the form of p-values, which enables the theoretically grounded quantification of the false positive detection probability of attentions. We demonstrate the validity and the effectiveness of the proposed method through numerical experiments and applications to brain image diagnoses.

{{</citation>}}


## cs.DC (1)



### (129/132) GMLake: Efficient and Transparent GPU Memory Defragmentation for Large-scale DNN Training with Virtual Memory Stitching (Cong Guo et al., 2024)

{{<citation>}}

Cong Guo, Rui Zhang, Jiale Xu, Jingwen Leng, Zihan Liu, Ziyu Huang, Minyi Guo, Hao Wu, Shouren Zhao, Junping Zhao, Ke Zhang. (2024)  
**GMLake: Efficient and Transparent GPU Memory Defragmentation for Large-scale DNN Training with Virtual Memory Stitching**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.08156v1)  

---


**ABSTRACT**  
Large-scale deep neural networks (DNNs), such as large language models (LLMs), have revolutionized the artificial intelligence (AI) field and become increasingly popular. However, training or fine-tuning such models requires substantial computational power and resources, where the memory capacity of a single acceleration device like a GPU is one of the most important bottlenecks. Owing to the prohibitively large overhead (e.g., $10 \times$) of GPUs' native memory allocator, DNN frameworks like PyTorch and TensorFlow adopt a caching allocator that maintains a memory pool with a splitting mechanism for fast memory (de)allocation. Unfortunately, the caching allocator's efficiency degrades quickly for popular memory reduction techniques such as recomputation, offloading, distributed training, and low-rank adaptation. The primary reason is that those memory reduction techniques introduce frequent and irregular memory (de)allocation requests, leading to severe fragmentation problems for the splitting-based caching allocator. To mitigate this fragmentation problem, we propose a novel memory allocation framework based on low-level GPU virtual memory management called GPU memory lake (GMLake). GMLake employs a novel virtual memory stitching (VMS) mechanism, which can fuse or combine non-contiguous memory blocks with a virtual memory address mapping. GMLake can reduce an average of 9.2 GB (up to 25 GB) GPU memory usage and 15% (up to 33% ) fragmentation among eight LLM models on GPU A100 with 80 GB memory. GMLake is completely transparent to the DNN models and memory reduction techniques and ensures the seamless execution of resource-intensive deep-learning tasks. We have open-sourced GMLake at https://github.com/intelligent-machine-learning/glake/tree/main/GMLake.

{{</citation>}}


## cs.RO (1)



### (130/132) S3M: Semantic Segmentation Sparse Mapping for UAVs with RGB-D Camera (Thanh Nguyen Canh et al., 2024)

{{<citation>}}

Thanh Nguyen Canh, Van-Truong Nguyen, Xiem HoangVan, Armagan Elibol, Nak Young Chong. (2024)  
**S3M: Semantic Segmentation Sparse Mapping for UAVs with RGB-D Camera**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.08134v1)  

---


**ABSTRACT**  
Unmanned Aerial Vehicles (UAVs) hold immense potential for critical applications, such as search and rescue operations, where accurate perception of indoor environments is paramount. However, the concurrent amalgamation of localization, 3D reconstruction, and semantic segmentation presents a notable hurdle, especially in the context of UAVs equipped with constrained power and computational resources. This paper presents a novel approach to address challenges in semantic information extraction and utilization within UAV operations. Our system integrates state-of-the-art visual SLAM to estimate a comprehensive 6-DoF pose and advanced object segmentation methods at the back end. To improve the computational and storage efficiency of the framework, we adopt a streamlined voxel-based 3D map representation - OctoMap to build a working system. Furthermore, the fusion algorithm is incorporated to obtain the semantic information of each frame from the front-end SLAM task, and the corresponding point. By leveraging semantic information, our framework enhances the UAV's ability to perceive and navigate through indoor spaces, addressing challenges in pose estimation accuracy and uncertainty reduction. Through Gazebo simulations, we validate the efficacy of our proposed system and successfully embed our approach into a Jetson Xavier AGX unit for real-world applications.

{{</citation>}}


## cond-mat.mtrl-sci (1)



### (131/132) Structure-based out-of-distribution (OOD) materials property prediction: a benchmark study (Sadman Sadeed Omee et al., 2024)

{{<citation>}}

Sadman Sadeed Omee, Nihang Fu, Rongzhi Dong, Ming Hu, Jianjun Hu. (2024)  
**Structure-based out-of-distribution (OOD) materials property prediction: a benchmark study**  

---
Primary Category: cond-mat.mtrl-sci  
Categories: cond-mat-mtrl-sci, cond-mat.mtrl-sci, cs-LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.08032v1)  

---


**ABSTRACT**  
In real-world material research, machine learning (ML) models are usually expected to predict and discover novel exceptional materials that deviate from the known materials. It is thus a pressing question to provide an objective evaluation of ML model performances in property prediction of out-of-distribution (OOD) materials that are different from the training set distribution. Traditional performance evaluation of materials property prediction models through random splitting of the dataset frequently results in artificially high performance assessments due to the inherent redundancy of typical material datasets. Here we present a comprehensive benchmark study of structure-based graph neural networks (GNNs) for extrapolative OOD materials property prediction. We formulate five different categories of OOD ML problems for three benchmark datasets from the MatBench study. Our extensive experiments show that current state-of-the-art GNN algorithms significantly underperform for the OOD property prediction tasks on average compared to their baselines in the MatBench study, demonstrating a crucial generalization gap in realistic material prediction tasks. We further examine the latent physical spaces of these GNN models and identify the sources of CGCNN, ALIGNN, and DeeperGATGNN's significantly more robust OOD performance than those of the current best models in the MatBench study (coGN and coNGN), and provide insights to improve their performance.

{{</citation>}}


## cs.IT (1)



### (132/132) Spatial Channel State Information Prediction with Generative AI: Towards Holographic Communication and Digital Radio Twin (Lihao Zhang et al., 2024)

{{<citation>}}

Lihao Zhang, Haijian Sun, Yong Zeng, Rose Qingyang Hu. (2024)  
**Spatial Channel State Information Prediction with Generative AI: Towards Holographic Communication and Digital Radio Twin**  

---
Primary Category: cs.IT  
Categories: cs-CV, cs-IT, cs.IT, math-IT  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.08023v1)  

---


**ABSTRACT**  
As 5G technology becomes increasingly established, the anticipation for 6G is growing, which promises to deliver faster and more reliable wireless connections via cutting-edge radio technologies. However, efficient management method of the large-scale antenna arrays deployed by those radio technologies is crucial. Traditional management methods are mainly reactive, usually based on feedback from users to adapt to the dynamic wireless channel. However, a more promising approach lies in the prediction of spatial channel state information (spatial-CSI), which is an all-inclusive channel characterization and consists of all the feasible line-of-sight (LoS) and non-line-of-sight (NLoS) paths between the transmitter (Tx) and receiver (Rx), with the three-dimension (3D) trajectory, attenuation, phase shift, delay, and polarization of each path. Advances in hardware and neural networks make it possible to predict such spatial-CSI using precise environmental information, and further look into the possibility of holographic communication, which implies complete control over every aspect of the radio waves emitted. Based on the integration of holographic communication and digital twin, we proposed a new framework, digital radio twin, which takes advantages from both the digital world and deterministic control over radio waves, supporting a wide range of high-level applications. As a preliminary attempt towards this visionary direction, in this paper, we explore the use of generative artificial intelligence (AI) to pinpoint the valid paths in a given environment, demonstrating promising results, and highlighting the potential of this approach in driving forward the evolution of 6G wireless communication technologies.

{{</citation>}}
