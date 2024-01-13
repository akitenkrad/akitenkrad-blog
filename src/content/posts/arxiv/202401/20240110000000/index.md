---
draft: false
title: "arXiv @ 2024.01.10"
date: 2024-01-10
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.10"
    identifier: arxiv_20240110
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.SE (6)](#csse-6)
- [eess.AS (2)](#eessas-2)
- [cs.LG (20)](#cslg-20)
- [cs.CL (14)](#cscl-14)
- [eess.SP (2)](#eesssp-2)
- [cs.CV (19)](#cscv-19)
- [cs.HC (3)](#cshc-3)
- [cs.RO (4)](#csro-4)
- [cs.CR (2)](#cscr-2)
- [cs.IT (1)](#csit-1)
- [cs.IR (3)](#csir-3)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.SD (1)](#cssd-1)
- [cs.AI (1)](#csai-1)
- [cs.CE (2)](#csce-2)
- [eess.IV (2)](#eessiv-2)
- [cs.AR (1)](#csar-1)
- [cs.CG (1)](#cscg-1)
- [q-fin.CP (1)](#q-fincp-1)
- [cs.DB (1)](#csdb-1)
- [cs.NE (1)](#csne-1)
- [math.OC (1)](#mathoc-1)
- [cs.MA (1)](#csma-1)

## cs.SE (6)



### (1/90) What Is an App Store? The Software Engineering Perspective (Wenhan Zhu et al., 2024)

{{<citation>}}

Wenhan Zhu, Sebastian Proksch, Daniel M. German, Michael W. Godfrey, Li Li, Shane McIntosh. (2024)  
**What Is an App Store? The Software Engineering Perspective**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.04287v1)  

---


**ABSTRACT**  
"App stores" are online software stores where end users may browse, purchase, download, and install software applications. By far, the best known app stores are associated with mobile platforms, such as Google Play for Android and Apple's App Store for iOS. The ubiquity of smartphones has led to mobile app stores becoming a touchstone experience of modern living. However, most of app store research has concentrated on properties of the apps rather than the stores themselves. Today, there is a rich diversity of app stores and these stores have largely been overlooked by researchers: app stores exist on many distinctive platforms, are aimed at different classes of users, and have different end-goals beyond simply selling a standalone app to a smartphone user.   We survey and characterize the broader dimensionality of app stores, and explore how and why they influence software development practices, such as system design and release management. We begin by collecting a set of app store examples from web search queries. By analyzing and curating the results, we derive a set of features common to app stores. We then build a dimensional model of app stores based on these features, and we fit each app store from our web search result set into this model. Next, we performed unsupervised clustering to the app stores to find their natural groupings. Our results suggest that app stores have become an essential stakeholder in modern software development. They control the distribution channel to end users and ensure that the applications are of suitable quality; in turn, this leads to developers adhering to various store guidelines when creating their applications. However, we found the app stores operational model could vary widely between stores, and this variability could in turn affect the generalizability of existing understanding of app stores.

{{</citation>}}


### (2/90) LLM4PLC: Harnessing Large Language Models for Verifiable Programming of PLCs in Industrial Control Systems (Mohamad Fakih et al., 2024)

{{<citation>}}

Mohamad Fakih, Rahul Dharmaji, Yasamin Moghaddas, Gustavo Quiros Araya, Oluwatosin Ogundare, Mohammad Abdullah Al Faruque. (2024)  
**LLM4PLC: Harnessing Large Language Models for Verifiable Programming of PLCs in Industrial Control Systems**  

---
Primary Category: cs.SE  
Categories: D-2-4; I-2-7; I-2-2, cs-AI, cs-CL, cs-PL, cs-SE, cs.SE  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.05443v1)  

---


**ABSTRACT**  
Although Large Language Models (LLMs) have established pre-dominance in automated code generation, they are not devoid of shortcomings. The pertinent issues primarily relate to the absence of execution guarantees for generated code, a lack of explainability, and suboptimal support for essential but niche programming languages. State-of-the-art LLMs such as GPT-4 and LLaMa2 fail to produce valid programs for Industrial Control Systems (ICS) operated by Programmable Logic Controllers (PLCs). We propose LLM4PLC, a user-guided iterative pipeline leveraging user feedback and external verification tools including grammar checkers, compilers and SMV verifiers to guide the LLM's generation. We further enhance the generation potential of LLM by employing Prompt Engineering and model fine-tuning through the creation and usage of LoRAs. We validate this system using a FischerTechnik Manufacturing TestBed (MFTB), illustrating how LLMs can evolve from generating structurally flawed code to producing verifiably correct programs for industrial applications. We run a complete test suite on GPT-3.5, GPT-4, Code Llama-7B, a fine-tuned Code Llama-7B model, Code Llama-34B, and a fine-tuned Code Llama-34B model. The proposed pipeline improved the generation success rate from 47% to 72%, and the Survey-of-Experts code quality from 2.25/10 to 7.75/10. To promote open research, we share the complete experimental setup, the LLM Fine-Tuning Weights, and the video demonstrations of the different programs on our dedicated webpage.

{{</citation>}}


### (3/90) T-FREX: A Transformer-based Feature Extraction Method from Mobile App Reviews (Quim Motger et al., 2024)

{{<citation>}}

Quim Motger, Alessio Miaschi, Felice Dell'Orletta, Xavier Franch, Jordi Marco. (2024)  
**T-FREX: A Transformer-based Feature Extraction Method from Mobile App Reviews**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.03833v1)  

---


**ABSTRACT**  
Mobile app reviews are a large-scale data source for software-related knowledge generation activities, including software maintenance, evolution and feedback analysis. Effective extraction of features (i.e., functionalities or characteristics) from these reviews is key to support analysis on the acceptance of these features, identification of relevant new feature requests and prioritization of feature development, among others. Traditional methods focus on syntactic pattern-based approaches, typically context-agnostic, evaluated on a closed set of apps, difficult to replicate and limited to a reduced set and domain of apps. Meanwhile, the pervasiveness of Large Language Models (LLMs) based on the Transformer architecture in software engineering tasks lays the groundwork for empirical evaluation of the performance of these models to support feature extraction. In this study, we present T-FREX, a Transformer-based, fully automatic approach for mobile app review feature extraction. First, we collect a set of ground truth features from users in a real crowdsourced software recommendation platform and transfer them automatically into a dataset of app reviews. Then, we use this newly created dataset to fine-tune multiple LLMs on a named entity recognition task under different data configurations. We assess the performance of T-FREX with respect to this ground truth, and we complement our analysis by comparing T-FREX with a baseline method from the field. Finally, we assess the quality of new features predicted by T-FREX through an external human evaluation. Results show that T-FREX outperforms on average the traditional syntactic-based method, especially when discovering new features from a domain for which the model has been fine-tuned.

{{</citation>}}


### (4/90) Enhanced Automated Code Vulnerability Repair using Large Language Models (David de-Fitero-Dominguez et al., 2024)

{{<citation>}}

David de-Fitero-Dominguez, Eva Garcia-Lopez, Antonio Garcia-Cabot, Jose-Javier Martinez-Herraiz. (2024)  
**Enhanced Automated Code Vulnerability Repair using Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.03741v1)  

---


**ABSTRACT**  
This research addresses the complex challenge of automated repair of code vulnerabilities, vital for enhancing digital security in an increasingly technology-driven world. The study introduces a novel and efficient format for the representation of code modification, using advanced Large Language Models (LLMs) such as Code Llama and Mistral. These models, fine-tuned on datasets featuring C code vulnerabilities, significantly improve the accuracy and adaptability of automated code repair techniques. A key finding is the enhanced repair accuracy of these models when compared to previous methods such as VulRepair, which underscores their practical utility and efficiency. The research also offers a critical assessment of current evaluation metrics, such as perfect predictions, and their limitations in reflecting the true capabilities of automated repair models in real-world scenarios. Following this, it underscores the importance of using test datasets devoid of train samples, emphasizing the need for dataset integrity to enhance the effectiveness of LLMs in code repair tasks. The significance of this work is its contribution to digital security, setting new standards for automated code vulnerability repair and paving the way for future advancements in the fields of cybersecurity and artificial intelligence. The study does not only highlight the potential of LLMs in enhancing code security but also fosters further exploration and research in these crucial areas.

{{</citation>}}


### (5/90) Assessing AI Detectors in Identifying AI-Generated Code: Implications for Education (Wei Hung Pan et al., 2024)

{{<citation>}}

Wei Hung Pan, Ming Jie Chok, Jonathan Leong Shan Wong, Yung Xin Shin, Yeong Shian Poon, Zhou Yang, Chun Yong Chong, David Lo, Mei Kuan Lim. (2024)  
**Assessing AI Detectors in Identifying AI-Generated Code: Implications for Education**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03676v1)  

---


**ABSTRACT**  
Educators are increasingly concerned about the usage of Large Language Models (LLMs) such as ChatGPT in programming education, particularly regarding the potential exploitation of imperfections in Artificial Intelligence Generated Content (AIGC) Detectors for academic misconduct. In this paper, we present an empirical study where the LLM is examined for its attempts to bypass detection by AIGC Detectors. This is achieved by generating code in response to a given question using different variants. We collected a dataset comprising 5,069 samples, with each sample consisting of a textual description of a coding problem and its corresponding human-written Python solution codes. These samples were obtained from various sources, including 80 from Quescol, 3,264 from Kaggle, and 1,725 from LeetCode. From the dataset, we created 13 sets of code problem variant prompts, which were used to instruct ChatGPT to generate the outputs. Subsequently, we assessed the performance of five AIGC detectors. Our results demonstrate that existing AIGC Detectors perform poorly in distinguishing between human-written code and AI-generated code.

{{</citation>}}


### (6/90) An exploratory study on automatic identification of assumptions in the development of deep learning frameworks (Chen Yang et al., 2024)

{{<citation>}}

Chen Yang, Peng Liang, Zinan Ma. (2024)  
**An exploratory study on automatic identification of assumptions in the development of deep learning frameworks**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: BERT, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.03653v2)  

---


**ABSTRACT**  
Stakeholders constantly make assumptions in the development of deep learning (DL) frameworks. These assumptions are related to various types of software artifacts (e.g., requirements, design decisions, and technical debt) and can turn out to be invalid, leading to system failures. Existing approaches and tools for assumption management usually depend on manual identification of assumptions. However, assumptions are scattered in various sources (e.g., code comments, commits, pull requests, and issues) of DL framework development, and manually identifying assumptions has high costs (e.g., time and resources). To overcome the issues of manually identifying assumptions in DL framework development, we constructed a new and largest dataset (i.e., AssuEval) of assumptions collected from the TensorFlow and Keras repositories on GitHub; explored the performance of seven traditional machine learning models (e.g., Support Vector Machine, Classification and Regression Trees), a popular DL model (i.e., ALBERT), and a large language model (i.e., ChatGPT) of identifying assumptions on the AssuEval dataset. The experiment results show that: ALBERT achieves the best performance (f1-score: 0.9584) of identifying assumptions on the AssuEval dataset, which is much better than the other models (the 2nd best f1-score is 0.6211, achieved by ChatGPT). Though ChatGPT is the most popular large language model, we do not recommend using it to identify assumptions in DL framework development because of its low performance on the task. Fine-tuning ChatGPT specifically for assumption identification could improve the performance. This study provides researchers with the largest dataset of assumptions for further research (e.g., assumption classification, evaluation, and reasoning) and helps practitioners better understand assumptions and how to manage them in their projects.

{{</citation>}}


## eess.AS (2)



### (7/90) FADI-AEC: Fast Score Based Diffusion Model Guided by Far-end Signal for Acoustic Echo Cancellation (Yang Liu et al., 2024)

{{<citation>}}

Yang Liu, Li Wan, Yun Li, Yiteng Huang, Ming Sun, James Luan, Yangyang Shi, Xin Lei. (2024)  
**FADI-AEC: Fast Score Based Diffusion Model Guided by Far-end Signal for Acoustic Echo Cancellation**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2401.04283v1)  

---


**ABSTRACT**  
Despite the potential of diffusion models in speech enhancement, their deployment in Acoustic Echo Cancellation (AEC) has been restricted. In this paper, we propose DI-AEC, pioneering a diffusion-based stochastic regeneration approach dedicated to AEC. Further, we propose FADI-AEC, fast score-based diffusion AEC framework to save computational demands, making it favorable for edge devices. It stands out by running the score model once per frame, achieving a significant surge in processing efficiency. Apart from that, we introduce a novel noise generation technique where far-end signals are utilized, incorporating both far-end and near-end signals to refine the score model's accuracy. We test our proposed method on the ICASSP2023 Microsoft deep echo cancellation challenge evaluation dataset, where our method outperforms some of the end-to-end methods and other diffusion based echo cancellation methods.

{{</citation>}}


### (8/90) LUPET: Incorporating Hierarchical Information Path into Multilingual ASR (Wei Liu et al., 2024)

{{<citation>}}

Wei Liu, Jingyong Hou, Dong Yang, Muyong Cao, Tan Lee. (2024)  
**LUPET: Incorporating Hierarchical Information Path into Multilingual ASR**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2401.03689v1)  

---


**ABSTRACT**  
Many factors have separately shown their effectiveness on improving multilingual ASR. They include language identity (LID) and phoneme information, language-specific processing modules and cross-lingual self-supervised speech representation, etc. However, few studies work on synergistically combining them to contribute a unified solution, which still remains an open question. To this end, a novel view to incorporate hierarchical information path LUPET into multilingual ASR is proposed. The LUPET is a path encoding multiple information in different granularity from shallow to deep encoder layers. Early information in this path is beneficial for deriving later occurred information. Specifically, the input goes from LID prediction to acoustic unit discovery followed by phoneme sharing, and then dynamically routed by mixture-of-expert for final token recognition. Experiments on 10 languages of Common Voice examined the superior performance of LUPET. Importantly, LUPET significantly boosts the recognition on high-resource languages, thus mitigating the compromised phenomenon towards low-resource languages in a multilingual setting.

{{</citation>}}


## cs.LG (20)



### (9/90) Attention versus Contrastive Learning of Tabular Data -- A Data-centric Benchmarking (Shourav B. Rabbani et al., 2024)

{{<citation>}}

Shourav B. Rabbani, Ivan V. Medri, Manar D. Samad. (2024)  
**Attention versus Contrastive Learning of Tabular Data -- A Data-centric Benchmarking**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.04266v1)  

---


**ABSTRACT**  
Despite groundbreaking success in image and text learning, deep learning has not achieved significant improvements against traditional machine learning (ML) when it comes to tabular data. This performance gap underscores the need for data-centric treatment and benchmarking of learning algorithms. Recently, attention and contrastive learning breakthroughs have shifted computer vision and natural language processing paradigms. However, the effectiveness of these advanced deep models on tabular data is sparsely studied using a few data sets with very large sample sizes, reporting mixed findings after benchmarking against a limited number of baselines. We argue that the heterogeneity of tabular data sets and selective baselines in the literature can bias the benchmarking outcomes. This article extensively evaluates state-of-the-art attention and contrastive learning methods on a wide selection of 28 tabular data sets (14 easy and 14 hard-to-classify) against traditional deep and machine learning. Our data-centric benchmarking demonstrates when traditional ML is preferred over deep learning and vice versa because no best learning method exists for all tabular data sets. Combining between-sample and between-feature attentions conquers the invincible traditional ML on tabular data sets by a significant margin but fails on high dimensional data, where contrastive learning takes a robust lead. While a hybrid attention-contrastive learning strategy mostly wins on hard-to-classify data sets, traditional methods are frequently superior on easy-to-classify data sets with presumably simpler decision boundaries. To the best of our knowledge, this is the first benchmarking paper with statistical analyses of attention and contrastive learning performances on a diverse selection of tabular data sets against traditional deep and machine learning baselines to facilitate further advances in this field.

{{</citation>}}


### (10/90) Curiosity & Entropy Driven Unsupervised RL in Multiple Environments (Shaurya Dewan et al., 2024)

{{<citation>}}

Shaurya Dewan, Anisha Jain, Zoe LaLena, Lifan Yu. (2024)  
**Curiosity & Entropy Driven Unsupervised RL in Multiple Environments**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.04198v1)  

---


**ABSTRACT**  
The authors of 'Unsupervised Reinforcement Learning in Multiple environments' propose a method, alpha-MEPOL, to tackle unsupervised RL across multiple environments. They pre-train a task-agnostic exploration policy using interactions from an entire environment class and then fine-tune this policy for various tasks using supervision. We expanded upon this work, with the goal of improving performance. We primarily propose and experiment with five new modifications to the original work: sampling trajectories using an entropy-based probability distribution, dynamic alpha, higher KL Divergence threshold, curiosity-driven exploration, and alpha-percentile sampling on curiosity. Dynamic alpha and higher KL-Divergence threshold both provided a significant improvement over the baseline from the earlier work. PDF-sampling failed to provide any improvement due to it being approximately equivalent to the baseline method when the sample space is small. In high-dimensional environments, the addition of curiosity-driven exploration enhances learning by encouraging the agent to seek diverse experiences and explore the unknown more. However, its benefits are limited in low-dimensional and simpler environments where exploration possibilities are constrained and there is little that is truly unknown to the agent. Overall, some of our experiments did boost performance over the baseline and there are a few directions that seem promising for further research.

{{</citation>}}


### (11/90) Mixtral of Experts (Albert Q. Jiang et al., 2024)

{{<citation>}}

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed. (2024)  
**Mixtral of Experts**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2401.04088v1)  

---


**ABSTRACT**  
We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model. Mixtral has the same architecture as Mistral 7B, with the difference that each layer is composed of 8 feedforward blocks (i.e. experts). For every token, at each layer, a router network selects two experts to process the current state and combine their outputs. Even though each token only sees two experts, the selected experts can be different at each timestep. As a result, each token has access to 47B parameters, but only uses 13B active parameters during inference. Mixtral was trained with a context size of 32k tokens and it outperforms or matches Llama 2 70B and GPT-3.5 across all evaluated benchmarks. In particular, Mixtral vastly outperforms Llama 2 70B on mathematics, code generation, and multilingual benchmarks. We also provide a model fine-tuned to follow instructions, Mixtral 8x7B - Instruct, that surpasses GPT-3.5 Turbo, Claude-2.1, Gemini Pro, and Llama 2 70B - chat model on human benchmarks. Both the base and instruct models are released under the Apache 2.0 license.

{{</citation>}}


### (12/90) MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts (Maciej Pióro et al., 2024)

{{<citation>}}

Maciej Pióro, Kamil Ciebiera, Krystian Król, Jan Ludziejewski, Sebastian Jaszczur. (2024)  
**MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.04081v1)  

---


**ABSTRACT**  
State Space Models (SSMs) have become serious contenders in the field of sequential modeling, challenging the dominance of Transformers. At the same time, Mixture of Experts (MoE) has significantly improved Transformer-based LLMs, including recent state-of-the-art open-source models. We propose that to unlock the potential of SSMs for scaling, they should be combined with MoE. We showcase this on Mamba, a recent SSM-based model that achieves remarkable, Transformer-like performance. Our model, MoE-Mamba, outperforms both Mamba and Transformer-MoE. In particular, MoE-Mamba reaches the same performance as Mamba in 2.2x less training steps while preserving the inference performance gains of Mamba against the Transformer.

{{</citation>}}


### (13/90) A Minimaximalist Approach to Reinforcement Learning from Human Feedback (Gokul Swamy et al., 2024)

{{<citation>}}

Gokul Swamy, Christoph Dann, Rahul Kidambi, Zhiwei Steven Wu, Alekh Agarwal. (2024)  
**A Minimaximalist Approach to Reinforcement Learning from Human Feedback**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.04056v1)  

---


**ABSTRACT**  
We present Self-Play Preference Optimization (SPO), an algorithm for reinforcement learning from human feedback. Our approach is minimalist in that it does not require training a reward model nor unstable adversarial training and is therefore rather simple to implement. Our approach is maximalist in that it provably handles non-Markovian, intransitive, and stochastic preferences while being robust to the compounding errors that plague offline approaches to sequential prediction. To achieve the preceding qualities, we build upon the concept of a Minimax Winner (MW), a notion of preference aggregation from the social choice theory literature that frames learning from preferences as a zero-sum game between two policies. By leveraging the symmetry of this game, we prove that rather than using the traditional technique of dueling two policies to compute the MW, we can simply have a single agent play against itself while maintaining strong convergence guarantees. Practically, this corresponds to sampling multiple trajectories from a policy, asking a rater or preference model to compare them, and then using the proportion of wins as the reward for a particular trajectory. We demonstrate that on a suite of continuous control tasks, we are able to learn significantly more efficiently than reward-model based approaches while maintaining robustness to the intransitive and stochastic preferences that frequently occur in practice when aggregating human judgments.

{{</citation>}}


### (14/90) Empirical Analysis of Efficient Fine-Tuning Methods for Large Pre-Trained Language Models (Nigel Doering et al., 2024)

{{<citation>}}

Nigel Doering, Cyril Gorlla, Trevor Tuttle, Adhvaith Vijay. (2024)  
**Empirical Analysis of Efficient Fine-Tuning Methods for Large Pre-Trained Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: GLUE, Language Model  
[Paper Link](http://arxiv.org/abs/2401.04051v1)  

---


**ABSTRACT**  
Fine-tuning large pre-trained language models for downstream tasks remains a critical challenge in natural language processing. This paper presents an empirical analysis comparing two efficient fine-tuning methods - BitFit and adapter modules - to standard full model fine-tuning. Experiments conducted on GLUE benchmark datasets (MRPC, COLA, STS-B) reveal several key insights. The BitFit approach, which trains only bias terms and task heads, matches full fine-tuning performance across varying amounts of training data and time constraints. It demonstrates remarkable stability even with only 30\% of data, outperforming full fine-tuning at intermediate data levels. Adapter modules exhibit high variability, with inconsistent gains over default models. The findings indicate BitFit offers an attractive balance between performance and parameter efficiency. Our work provides valuable perspectives on model tuning, emphasizing robustness and highlighting BitFit as a promising alternative for resource-constrained or streaming task settings. The analysis offers actionable guidelines for efficient adaptation of large pre-trained models, while illustrating open challenges in stabilizing techniques like adapter modules.

{{</citation>}}


### (15/90) Polynomial Precision Dependence Solutions to Alignment Research Center Matrix Completion Problems (Rico Angell, 2024)

{{<citation>}}

Rico Angell. (2024)  
**Polynomial Precision Dependence Solutions to Alignment Research Center Matrix Completion Problems**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03999v1)  

---


**ABSTRACT**  
We present solutions to the matrix completion problems proposed by the Alignment Research Center that have a polynomial dependence on the precision $\varepsilon$. The motivation for these problems is to enable efficient computation of heuristic estimators to formally evaluate and reason about different quantities of deep neural networks in the interest of AI alignment. Our solutions involve reframing the matrix completion problems as a semidefinite program (SDP) and using recent advances in spectral bundle methods for fast, efficient, and scalable SDP solving.

{{</citation>}}


### (16/90) Behavioural Cloning in VizDoom (Ryan Spick et al., 2024)

{{<citation>}}

Ryan Spick, Timothy Bradley, Ayush Raina, Pierluigi Vito Amadori, Guy Moss. (2024)  
**Behavioural Cloning in VizDoom**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.03993v1)  

---


**ABSTRACT**  
This paper describes methods for training autonomous agents to play the game "Doom 2" through Imitation Learning (IL) using only pixel data as input. We also explore how Reinforcement Learning (RL) compares to IL for humanness by comparing camera movement and trajectory data. Through behavioural cloning, we examine the ability of individual models to learn varying behavioural traits. We attempt to mimic the behaviour of real players with different play styles, and find we can train agents that behave aggressively, passively, or simply more human-like than traditional AIs. We propose these methods of introducing more depth and human-like behaviour to agents in video games. The trained IL agents perform on par with the average players in our dataset, whilst outperforming the worst players. While performance was not as strong as common RL approaches, it provides much stronger human-like behavioural traits to the agent.

{{</citation>}}


### (17/90) Tiny Time Mixers (TTMs): Fast Pretrained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series (Vijay Ekambaram et al., 2024)

{{<citation>}}

Vijay Ekambaram, Arindam Jati, Nam H. Nguyen, Pankaj Dayama, Chandra Reddy, Wesley M. Gifford, Jayant Kalagnanam. (2024)  
**Tiny Time Mixers (TTMs): Fast Pretrained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Few-Shot, Time Series  
[Paper Link](http://arxiv.org/abs/2401.03955v2)  

---


**ABSTRACT**  
Large Pretrained models for zero/few-shot learning excel in language and vision domains but encounter challenges in multivariate time series (TS) due to the diverse nature and scarcity of publicly available pretraining data. Consequently, there has been a recent surge in utilizing pretrained large language models (LLMs) with various adaptations for time series forecasting. These approaches employ cross-domain transfer learning and surprisingly yield impressive results. However, these models are typically very slow and large ($\sim$billion parameters) and do not consider cross-channel correlations. To address this, we present Multi-level Tiny Time Mixers (TTM), a significantly small model based on the lightweight TSMixer architecture. TTM marks the first success in developing tiny general-pretrained models ($\le$1 million parameters), exclusively trained on public TS datasets in a flash of just 4-8 hrs with effective transfer learning capabilities for forecasting. To tackle the complexity of pretraining on multiple datasets with varied temporal resolutions, we introduce several novel enhancements such as adaptive patching, dataset augmentation via downsampling, and resolution prefix tuning. Moreover, we employ a multi-level modeling strategy to effectively model channel correlations and incorporate exogenous signals during fine-tuning, a crucial capability lacking in existing benchmarks. TTM excels in few/zero-shot forecasting, demonstrating significant accuracy gains (12-38%) over existing benchmarks. Further, it achieves a remarkable 14-106X reduction in model parameters, enabling 54-65X faster finetuning/inference as compared to the LLM-TS benchmarks. In fact, TTM's zero-shot often surpasses the few-shot results in many popular benchmarks, highlighting the efficacy of our approach. Code and Pretrained Models will be open-sourced.

{{</citation>}}


### (18/90) Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning (Wenhan Xia et al., 2024)

{{<citation>}}

Wenhan Xia, Chengwei Qin, Elad Hazan. (2024)  
**Chain of LoRA: Efficient Fine-tuning of Language Models via Residual Learning**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.04151v1)  

---


**ABSTRACT**  
Fine-tuning is the primary methodology for tailoring pre-trained large language models to specific tasks. As the model's scale and the diversity of tasks expand, parameter-efficient fine-tuning methods are of paramount importance. One of the most widely used family of methods is low-rank adaptation (LoRA) and its variants. LoRA encodes weight update as the product of two low-rank matrices. Despite its advantages, LoRA falls short of full-parameter fine-tuning in terms of generalization error for certain tasks.   We introduce Chain of LoRA (COLA), an iterative optimization framework inspired by the Frank-Wolfe algorithm, to bridge the gap between LoRA and full parameter fine-tuning, without incurring additional computational costs or memory overheads. COLA employs a residual learning procedure where it merges learned LoRA modules into the pre-trained language model parameters and re-initilize optimization for new born LoRA modules. We provide theoretical convergence guarantees as well as empirical results to validate the effectiveness of our algorithm. Across various models (OPT and llama-2) and seven benchmarking tasks, we demonstrate that COLA can consistently outperform LoRA without additional computational or memory costs.

{{</citation>}}


### (19/90) A Tensor Network Implementation of Multi Agent Reinforcement Learning (Sunny Howard, 2024)

{{<citation>}}

Sunny Howard. (2024)  
**A Tensor Network Implementation of Multi Agent Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.03896v1)  

---


**ABSTRACT**  
Recently it has been shown that tensor networks (TNs) have the ability to represent the expected return of a single-agent finite Markov decision process (FMDP). The TN represents a distribution model, where all possible trajectories are considered. When extending these ideas to a multi-agent setting, distribution models suffer from the curse of dimensionality: the exponential relation between the number of possible trajectories and the number of agents. The key advantage of using TNs in this setting is that there exists a large number of established optimisation and decomposition techniques that are specific to TNs, that one can apply to ensure the most efficient representation is found. In this report, these methods are used to form a TN that represents the expected return of a multi-agent reinforcement learning (MARL) task. This model is then applied to a 2 agent random walker example, where it was shown that the policy is correctly optimised using a DMRG technique. Finally, I demonstrate the use of an exact decomposition technique, reducing the number of elements in the tensors by 97.5%, without experiencing any loss of information.

{{</citation>}}


### (20/90) Inverse Reinforcement Learning with Sub-optimal Experts (Riccardo Poiani et al., 2024)

{{<citation>}}

Riccardo Poiani, Gabriele Curti, Alberto Maria Metelli, Marcello Restelli. (2024)  
**Inverse Reinforcement Learning with Sub-optimal Experts**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.03857v1)  

---


**ABSTRACT**  
Inverse Reinforcement Learning (IRL) techniques deal with the problem of deducing a reward function that explains the behavior of an expert agent who is assumed to act optimally in an underlying unknown task. In several problems of interest, however, it is possible to observe the behavior of multiple experts with different degree of optimality (e.g., racing drivers whose skills ranges from amateurs to professionals). For this reason, in this work, we extend the IRL formulation to problems where, in addition to demonstrations from the optimal agent, we can observe the behavior of multiple sub-optimal experts. Given this problem, we first study the theoretical properties of the class of reward functions that are compatible with a given set of experts, i.e., the feasible reward set. Our results show that the presence of multiple sub-optimal experts can significantly shrink the set of compatible rewards. Furthermore, we study the statistical complexity of estimating the feasible reward set with a generative model. To this end, we analyze a uniform sampling algorithm that results in being minimax optimal whenever the sub-optimal experts' performance level is sufficiently close to the one of the optimal agent.

{{</citation>}}


### (21/90) Inferring Properties of Graph Neural Networks (Dat Nguyen et al., 2024)

{{<citation>}}

Dat Nguyen, Hieu M. Vu, Cong-Thanh Le, Bach Le, David Lo, Corina Pasareanu. (2024)  
**Inferring Properties of Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs-PL, cs-SE, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.03790v1)  

---


**ABSTRACT**  
We propose GNNInfer, the first automatic property inference technique for GNNs. To tackle the challenge of varying input structures in GNNs, GNNInfer first identifies a set of representative influential structures that contribute significantly towards the prediction of a GNN. Using these structures, GNNInfer converts each pair of an influential structure and the GNN to their equivalent FNN and then leverages existing property inference techniques to effectively capture properties of the GNN that are specific to the influential structures. GNNINfer then generalizes the captured properties to any input graphs that contain the influential structures. Finally, GNNInfer improves the correctness of the inferred properties by building a model (either a decision tree or linear regression) that estimates the deviation of GNN output from the inferred properties given full input graphs. The learned model helps GNNInfer extend the inferred properties with constraints to the input and output of the GNN, obtaining stronger properties that hold on full input graphs.   Our experiments show that GNNInfer is effective in inferring likely properties of popular real-world GNNs, and more importantly, these inferred properties help effectively defend against GNNs' backdoor attacks. In particular, out of the 13 ground truth properties, GNNInfer re-discovered 8 correct properties and discovered likely correct properties that approximate the remaining 5 ground truth properties. Using properties inferred by GNNInfer to defend against the state-of-the-art backdoor attack technique on GNNs, namely UGBA, experiments show that GNNInfer's defense success rate is up to 30 times better than existing baselines.

{{</citation>}}


### (22/90) Long-term Safe Reinforcement Learning with Binary Feedback (Akifumi Wachi et al., 2024)

{{<citation>}}

Akifumi Wachi, Wataru Hashimoto, Kazumune Hashimoto. (2024)  
**Long-term Safe Reinforcement Learning with Binary Feedback**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: GLM, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.03786v2)  

---


**ABSTRACT**  
Safety is an indispensable requirement for applying reinforcement learning (RL) to real problems. Although there has been a surge of safe RL algorithms proposed in recent years, most existing work typically 1) relies on receiving numeric safety feedback; 2) does not guarantee safety during the learning process; 3) limits the problem to a priori known, deterministic transition dynamics; and/or 4) assume the existence of a known safe policy for any states. Addressing the issues mentioned above, we thus propose Long-term Binaryfeedback Safe RL (LoBiSaRL), a safe RL algorithm for constrained Markov decision processes (CMDPs) with binary safety feedback and an unknown, stochastic state transition function. LoBiSaRL optimizes a policy to maximize rewards while guaranteeing a long-term safety that an agent executes only safe state-action pairs throughout each episode with high probability. Specifically, LoBiSaRL models the binary safety function via a generalized linear model (GLM) and conservatively takes only a safe action at every time step while inferring its effect on future safety under proper assumptions. Our theoretical results show that LoBiSaRL guarantees the long-term safety constraint, with high probability. Finally, our empirical results demonstrate that our algorithm is safer than existing methods without significantly compromising performance in terms of reward.

{{</citation>}}


### (23/90) Adaptive Experimental Design for Policy Learning (Masahiro Kato et al., 2024)

{{<citation>}}

Masahiro Kato, Kyohei Okumura, Takuya Ishihara, Toru Kitagawa. (2024)  
**Adaptive Experimental Design for Policy Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, econ-EM, stat-ME, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03756v2)  

---


**ABSTRACT**  
Evidence-based targeting has been a topic of growing interest among the practitioners of policy and business. Formulating decision-maker's policy learning as a fixed-budget best arm identification (BAI) problem with contextual information, we study an optimal adaptive experimental design for policy learning with multiple treatment arms. In the sampling stage, the planner assigns treatment arms adaptively over sequentially arriving experimental units upon observing their contextual information (covariates). After the experiment, the planner recommends an individualized assignment rule to the population. Setting the worst-case expected regret as the performance criterion of adaptive sampling and recommended policies, we derive its asymptotic lower bounds, and propose a strategy, Adaptive Sampling-Policy Learning strategy (PLAS), whose leading factor of the regret upper bound aligns with the lower bound as the size of experimental units increases.

{{</citation>}}


### (24/90) From Data to Insights: A Comprehensive Survey on Advanced Applications in Thyroid Cancer Research (Xinyu Zhang et al., 2024)

{{<citation>}}

Xinyu Zhang, Vincent CS Lee, Feng Liu. (2024)  
**From Data to Insights: A Comprehensive Survey on Advanced Applications in Thyroid Cancer Research**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03722v1)  

---


**ABSTRACT**  
Thyroid cancer, the most prevalent endocrine cancer, has gained significant global attention due to its impact on public health. Extensive research efforts have been dedicated to leveraging artificial intelligence (AI) methods for the early detection of this disease, aiming to reduce its morbidity rates. However, a comprehensive understanding of the structured organization of research applications in this particular field remains elusive. To address this knowledge gap, we conducted a systematic review and developed a comprehensive taxonomy of machine learning-based applications in thyroid cancer pathogenesis, diagnosis, and prognosis. Our primary objective was to facilitate the research community's ability to stay abreast of technological advancements and potentially lead the emerging trends in this field. This survey presents a coherent literature review framework for interpreting the advanced techniques used in thyroid cancer research. A total of 758 related studies were identified and scrutinized. To the best of our knowledge, this is the first review that provides an in-depth analysis of the various aspects of AI applications employed in the context of thyroid cancer. Furthermore, we highlight key challenges encountered in this domain and propose future research opportunities for those interested in studying the latest trends or exploring less-investigated aspects of thyroid cancer research. By presenting this comprehensive review and taxonomy, we contribute to the existing knowledge in the field, while providing valuable insights for researchers, clinicians, and stakeholders in advancing the understanding and management of this disease.

{{</citation>}}


### (25/90) Universal Time-Series Representation Learning: A Survey (Patara Trirat et al., 2024)

{{<citation>}}

Patara Trirat, Yooju Shin, Junhyeok Kang, Youngeun Nam, Jihye Na, Minyoung Bae, Joeun Kim, Byunghyun Kim, Jae-Gil Lee. (2024)  
**Universal Time-Series Representation Learning: A Survey**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.03717v1)  

---


**ABSTRACT**  
Time-series data exists in every corner of real-world systems and services, ranging from satellites in the sky to wearable devices on human bodies. Learning representations by extracting and inferring valuable information from these time series is crucial for understanding the complex dynamics of particular phenomena and enabling informed decisions. With the learned representations, we can perform numerous downstream analyses more effectively. Among several approaches, deep learning has demonstrated remarkable performance in extracting hidden patterns and features from time-series data without manual feature engineering. This survey first presents a novel taxonomy based on three fundamental elements in designing state-of-the-art universal representation learning methods for time series. According to the proposed taxonomy, we comprehensively review existing studies and discuss their intuitions and insights into how these methods enhance the quality of learned representations. Finally, as a guideline for future studies, we summarize commonly used experimental setups and datasets and discuss several promising research directions. An up-to-date corresponding resource is available at https://github.com/itouchz/awesome-deep-time-series-representations.

{{</citation>}}


### (26/90) Evaluating Brain-Inspired Modular Training in Automated Circuit Discovery for Mechanistic Interpretability (Jatin Nainani, 2024)

{{<citation>}}

Jatin Nainani. (2024)  
**Evaluating Brain-Inspired Modular Training in Automated Circuit Discovery for Mechanistic Interpretability**  

---
Primary Category: cs.LG  
Categories: I-2-6, cs-LG, cs-NE, cs.LG  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03646v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have experienced a rapid rise in AI, changing a wide range of applications with their advanced capabilities. As these models become increasingly integral to decision-making, the need for thorough interpretability has never been more critical. Mechanistic Interpretability offers a pathway to this understanding by identifying and analyzing specific sub-networks or 'circuits' within these complex systems. A crucial aspect of this approach is Automated Circuit Discovery, which facilitates the study of large models like GPT4 or LLAMA in a feasible manner. In this context, our research evaluates a recent method, Brain-Inspired Modular Training (BIMT), designed to enhance the interpretability of neural networks. We demonstrate how BIMT significantly improves the efficiency and quality of Automated Circuit Discovery, overcoming the limitations of manual methods. Our comparative analysis further reveals that BIMT outperforms existing models in terms of circuit quality, discovery time, and sparsity. Additionally, we provide a comprehensive computational analysis of BIMT, including aspects such as training duration, memory allocation requirements, and inference speed. This study advances the larger objective of creating trustworthy and transparent AI systems in addition to demonstrating how well BIMT works to make neural networks easier to understand.

{{</citation>}}


### (27/90) Unifying Graph Contrastive Learning via Graph Message Augmentation (Ziyan Zhang et al., 2024)

{{<citation>}}

Ziyan Zhang, Bo Jiang, Jin Tang, Bin Luo. (2024)  
**Unifying Graph Contrastive Learning via Graph Message Augmentation**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Augmentation, Contrastive Learning, GNN  
[Paper Link](http://arxiv.org/abs/2401.03638v1)  

---


**ABSTRACT**  
Graph contrastive learning is usually performed by first conducting Graph Data Augmentation (GDA) and then employing a contrastive learning pipeline to train GNNs. As we know that GDA is an important issue for graph contrastive learning. Various GDAs have been developed recently which mainly involve dropping or perturbing edges, nodes, node attributes and edge attributes. However, to our knowledge, it still lacks a universal and effective augmentor that is suitable for different types of graph data. To address this issue, in this paper, we first introduce the graph message representation of graph data. Based on it, we then propose a novel Graph Message Augmentation (GMA), a universal scheme for reformulating many existing GDAs. The proposed unified GMA not only gives a new perspective to understand many existing GDAs but also provides a universal and more effective graph data augmentation for graph self-supervised learning tasks. Moreover, GMA introduces an easy way to implement the mixup augmentor which is natural for images but usually challengeable for graphs. Based on the proposed GMA, we then propose a unified graph contrastive learning, termed Graph Message Contrastive Learning (GMCL), that employs attribution-guided universal GMA for graph contrastive learning. Experiments on many graph learning tasks demonstrate the effectiveness and benefits of the proposed GMA and GMCL approaches.

{{</citation>}}


### (28/90) Learn Once Plan Arbitrarily (LOPA): Attention-Enhanced Deep Reinforcement Learning Method for Global Path Planning (Guoming Huang et al., 2024)

{{<citation>}}

Guoming Huang, Mingxin Hou, Xiaofang Yuan, Shuqiao Huang, Yaonan Wang. (2024)  
**Learn Once Plan Arbitrarily (LOPA): Attention-Enhanced Deep Reinforcement Learning Method for Global Path Planning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Attention, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.04145v1)  

---


**ABSTRACT**  
Deep reinforcement learning (DRL) methods have recently shown promise in path planning tasks. However, when dealing with global planning tasks, these methods face serious challenges such as poor convergence and generalization. To this end, we propose an attention-enhanced DRL method called LOPA (Learn Once Plan Arbitrarily) in this paper. Firstly, we analyze the reasons of these problems from the perspective of DRL's observation, revealing that the traditional design causes DRL to be interfered by irrelevant map information. Secondly, we develop the LOPA which utilizes a novel attention-enhanced mechanism to attain an improved attention capability towards the key information of the observation. Such a mechanism is realized by two steps: (1) an attention model is built to transform the DRL's observation into two dynamic views: local and global, significantly guiding the LOPA to focus on the key information on the given maps; (2) a dual-channel network is constructed to process these two views and integrate them to attain an improved reasoning capability. The LOPA is validated via multi-objective global path planning experiments. The result suggests the LOPA has improved convergence and generalization performance as well as great path planning efficiency.

{{</citation>}}


## cs.CL (14)



### (29/90) MARG: Multi-Agent Review Generation for Scientific Papers (Mike D'Arcy et al., 2024)

{{<citation>}}

Mike D'Arcy, Tom Hope, Larry Birnbaum, Doug Downey. (2024)  
**MARG: Multi-Agent Review Generation for Scientific Papers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.04259v1)  

---


**ABSTRACT**  
We study the ability of LLMs to generate feedback for scientific papers and develop MARG, a feedback generation approach using multiple LLM instances that engage in internal discussion. By distributing paper text across agents, MARG can consume the full text of papers beyond the input length limitations of the base LLM, and by specializing agents and incorporating sub-tasks tailored to different comment types (experiments, clarity, impact) it improves the helpfulness and specificity of feedback. In a user study, baseline methods using GPT-4 were rated as producing generic or very generic comments more than half the time, and only 1.7 comments per paper were rated as good overall in the best baseline. Our system substantially improves the ability of GPT-4 to generate specific and helpful feedback, reducing the rate of generic comments from 60% to 29% and generating 3.7 good comments per paper (a 2.2x improvement).

{{</citation>}}


### (30/90) Distortions in Judged Spatial Relations in Large Language Models: The Dawn of Natural Language Geographic Data? (Nir Fulman et al., 2024)

{{<citation>}}

Nir Fulman, Abdulkadir Memduhoğlu, Alexander Zipf. (2024)  
**Distortions in Judged Spatial Relations in Large Language Models: The Dawn of Natural Language Geographic Data?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.04218v1)  

---


**ABSTRACT**  
We present a benchmark for assessing the capability of Large Language Models (LLMs) to discern intercardinal directions between geographic locations and apply it to three prominent LLMs: GPT-3.5, GPT-4, and Llama-2. This benchmark specifically evaluates whether LLMs exhibit a hierarchical spatial bias similar to humans, where judgments about individual locations' spatial relationships are influenced by the perceived relationships of the larger groups that contain them. To investigate this, we formulated 14 questions focusing on well-known American cities. Seven questions were designed to challenge the LLMs with scenarios potentially influenced by the orientation of larger geographical units, such as states or countries, while the remaining seven targeted locations less susceptible to such hierarchical categorization. Among the tested models, GPT-4 exhibited superior performance with 55.3% accuracy, followed by GPT-3.5 at 47.3%, and Llama-2 at 44.7%. The models showed significantly reduced accuracy on tasks with suspected hierarchical bias. For example, GPT-4's accuracy dropped to 32.9% on these tasks, compared to 85.7% on others. Despite these inaccuracies, the models identified the nearest cardinal direction in most cases, suggesting associative learning, embodying human-like misconceptions. We discuss the potential of text-based data representing geographic relationships directly to improve the spatial reasoning capabilities of LLMs.

{{</citation>}}


### (31/90) FFSplit: Split Feed-Forward Network For Optimizing Accuracy-Efficiency Trade-off in Language Model Inference (Zirui Liu et al., 2024)

{{<citation>}}

Zirui Liu, Qingquan Song, Qiang Charles Xiao, Sathiya Keerthi Selvaraj, Rahul Mazumder, Aman Gupta, Xia Hu. (2024)  
**FFSplit: Split Feed-Forward Network For Optimizing Accuracy-Efficiency Trade-off in Language Model Inference**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2401.04044v1)  

---


**ABSTRACT**  
The large number of parameters in Pretrained Language Models enhance their performance, but also make them resource-intensive, making it challenging to deploy them on commodity hardware like a single GPU. Due to the memory and power limitations of these devices, model compression techniques are often used to decrease both the model's size and its inference latency. This usually results in a trade-off between model accuracy and efficiency. Therefore, optimizing this balance is essential for effectively deploying LLMs on commodity hardware. A significant portion of the efficiency challenge is the Feed-forward network (FFN) component, which accounts for roughly $\frac{2}{3}$ total parameters and inference latency. In this paper, we first observe that only a few neurons of FFN module have large output norm for any input tokens, a.k.a. heavy hitters, while the others are sparsely triggered by different tokens. Based on this observation, we explicitly split the FFN into two parts according to the heavy hitters. We improve the efficiency-accuracy trade-off of existing compression methods by allocating more resource to FFN parts with heavy hitters. In practice, our method can reduce model size by 43.1\% and bring $1.25\sim1.56\times$ wall clock time speedup on different hardware with negligible accuracy drop.

{{</citation>}}


### (32/90) IDoFew: Intermediate Training Using Dual-Clustering in Language Models for Few Labels Text Classification (Abdullah Alsuhaibani et al., 2024)

{{<citation>}}

Abdullah Alsuhaibani, Hamad Zogan, Imran Razzak, Shoaib Jameel, Guandong Xu. (2024)  
**IDoFew: Intermediate Training Using Dual-Clustering in Language Models for Few Labels Text Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model, NLP, Natural Language Processing, Text Classification, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.04025v1)  

---


**ABSTRACT**  
Language models such as Bidirectional Encoder Representations from Transformers (BERT) have been very effective in various Natural Language Processing (NLP) and text mining tasks including text classification. However, some tasks still pose challenges for these models, including text classification with limited labels. This can result in a cold-start problem. Although some approaches have attempted to address this problem through single-stage clustering as an intermediate training step coupled with a pre-trained language model, which generates pseudo-labels to improve classification, these methods are often error-prone due to the limitations of the clustering algorithms. To overcome this, we have developed a novel two-stage intermediate clustering with subsequent fine-tuning that models the pseudo-labels reliably, resulting in reduced prediction errors. The key novelty in our model, IDoFew, is that the two-stage clustering coupled with two different clustering algorithms helps exploit the advantages of the complementary algorithms that reduce the errors in generating reliable pseudo-labels for fine-tuning. Our approach has shown significant improvements compared to strong comparative models.

{{</citation>}}


### (33/90) TextMachina: Seamless Generation of Machine-Generated Text Datasets (Areg Mikael Sarvazyan et al., 2024)

{{<citation>}}

Areg Mikael Sarvazyan, José Ángel González, Marc Franco-Salvador. (2024)  
**TextMachina: Seamless Generation of Machine-Generated Text Datasets**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.03946v1)  

---


**ABSTRACT**  
Recent advancements in Large Language Models (LLMs) have led to high-quality Machine-Generated Text (MGT), giving rise to countless new use cases and applications. However, easy access to LLMs is posing new challenges due to misuse. To address malicious usage, researchers have released datasets to effectively train models on MGT-related tasks. Similar strategies are used to compile these datasets, but no tool currently unifies them. In this scenario, we introduce TextMachina, a modular and extensible Python framework, designed to aid in the creation of high-quality, unbiased datasets to build robust models for MGT-related tasks such as detection, attribution, or boundary detection. It provides a user-friendly pipeline that abstracts away the inherent intricacies of building MGT datasets, such as LLM integrations, prompt templating, and bias mitigation. The quality of the datasets generated by TextMachina has been assessed in previous works, including shared tasks where more than one hundred teams trained robust MGT detectors.

{{</citation>}}


### (34/90) SpeechAgents: Human-Communication Simulation with Multi-Modal Multi-Agent Systems (Dong Zhang et al., 2024)

{{<citation>}}

Dong Zhang, Zhaowei Li, Pengyu Wang, Xin Zhang, Yaqian Zhou, Xipeng Qiu. (2024)  
**SpeechAgents: Human-Communication Simulation with Multi-Modal Multi-Agent Systems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.03945v1)  

---


**ABSTRACT**  
Human communication is a complex and diverse process that not only involves multiple factors such as language, commonsense, and cultural backgrounds but also requires the participation of multimodal information, such as speech. Large Language Model (LLM)-based multi-agent systems have demonstrated promising performance in simulating human society. Can we leverage LLM-based multi-agent systems to simulate human communication? However, current LLM-based multi-agent systems mainly rely on text as the primary medium. In this paper, we propose SpeechAgents, a multi-modal LLM based multi-agent system designed for simulating human communication. SpeechAgents utilizes multi-modal LLM as the control center for individual agent and employes multi-modal signals as the medium for exchanged messages among agents. Additionally, we propose Multi-Agent Tuning to enhance the multi-agent capabilities of LLM without compromising general abilities. To strengthen and evaluate the effectiveness of human communication simulation, we build the Human-Communication Simulation Benchmark. Experimental results demonstrate that SpeechAgents can simulate human communication dialogues with consistent content, authentic rhythm, and rich emotions and demonstrate excellent scalability even with up to 25 agents, which can apply to tasks such as drama creation and audio novels generation. Code and models will be open-sourced at https://github. com/0nutation/SpeechAgents

{{</citation>}}


### (35/90) A Philosophical Introduction to Language Models -- Part I: Continuity With Classic Debates (Raphaël Millière et al., 2024)

{{<citation>}}

Raphaël Millière, Cameron Buckner. (2024)  
**A Philosophical Introduction to Language Models -- Part I: Continuity With Classic Debates**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03910v1)  

---


**ABSTRACT**  
Large language models like GPT-4 have achieved remarkable proficiency in a broad spectrum of language-based tasks, some of which are traditionally associated with hallmarks of human intelligence. This has prompted ongoing disagreements about the extent to which we can meaningfully ascribe any kind of linguistic or cognitive competence to language models. Such questions have deep philosophical roots, echoing longstanding debates about the status of artificial neural networks as cognitive models. This article -- the first part of two companion papers -- serves both as a primer on language models for philosophers, and as an opinionated survey of their significance in relation to classic debates in the philosophy cognitive science, artificial intelligence, and linguistics. We cover topics such as compositionality, language acquisition, semantic competence, grounding, world models, and the transmission of cultural knowledge. We argue that the success of language models challenges several long-held assumptions about artificial neural networks. However, we also highlight the need for further empirical investigation to better understand their internal mechanisms. This sets the stage for the companion paper (Part II), which turns to novel empirical methods for probing the inner workings of language models, and new philosophical questions prompted by their latest developments.

{{</citation>}}


### (36/90) WEBDial, a Multi-domain, Multitask Statistical Dialogue Framework with RDF (Morgan Veyret et al., 2024)

{{<citation>}}

Morgan Veyret, Jean-Baptiste Duchene, Kekeli Afonouvi, Quentin Brabant, Gwenole Lecorve, Lina M. Rojas-Barahona. (2024)  
**WEBDial, a Multi-domain, Multitask Statistical Dialogue Framework with RDF**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2401.03905v1)  

---


**ABSTRACT**  
Typically available dialogue frameworks have adopted a semantic representation based on dialogue-acts and slot-value pairs. Despite its simplicity, this representation has disadvantages such as the lack of expressivity, scalability and explainability. We present WEBDial: a dialogue framework that relies on a graph formalism by using RDF triples instead of slot-value pairs. We describe its overall architecture and the graph-based semantic representation. We show its applicability from simple to complex applications, by varying the complexity of domains and tasks: from single domain and tasks to multiple domains and complex tasks.

{{</citation>}}


### (37/90) Boldly Going Where No Benchmark Has Gone Before: Exposing Bias and Shortcomings in Code Generation Evaluation (Ankit Yadav et al., 2024)

{{<citation>}}

Ankit Yadav, Mayank Singh. (2024)  
**Boldly Going Where No Benchmark Has Gone Before: Exposing Bias and Shortcomings in Code Generation Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.03855v1)  

---


**ABSTRACT**  
Motivated by the increasing popularity of code generation from human descriptions using large language models (LLMs), several benchmarks have been proposed to assess the capabilities of existing and emerging models. This study presents a large-scale human evaluation of HumanEval and MBPP, two widely used benchmarks for Python code generation, focusing on their diversity and difficulty. Our findings reveal a significant bias towards a limited number of programming concepts, with negligible or no representation of most concepts. Additionally, we identify a concerningly high proportion of easy programming questions, potentially leading to an overestimation of model performance on code generation tasks.

{{</citation>}}


### (38/90) We Need to Talk About Classification Evaluation Metrics in NLP (Peter Vickers et al., 2024)

{{<citation>}}

Peter Vickers, Loïc Barrault, Emilio Monti, Nikolaos Aletras. (2024)  
**We Need to Talk About Classification Evaluation Metrics in NLP**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.03831v1)  

---


**ABSTRACT**  
In Natural Language Processing (NLP) classification tasks such as topic categorisation and sentiment analysis, model generalizability is generally measured with standard metrics such as Accuracy, F-Measure, or AUC-ROC. The diversity of metrics, and the arbitrariness of their application suggest that there is no agreement within NLP on a single best metric to use. This lack suggests there has not been sufficient examination of the underlying heuristics which each metric encodes. To address this we compare several standard classification metrics with more 'exotic' metrics and demonstrate that a random-guess normalised Informedness metric is a parsimonious baseline for task performance. To show how important the choice of metric is, we perform extensive experiments on a wide range of NLP tasks including a synthetic scenario, natural language understanding, question answering and machine translation. Across these tasks we use a superset of metrics to rank models and find that Informedness best captures the ideal model characteristics. Finally, we release a Python implementation of Informedness following the SciKitLearn classifier format.

{{</citation>}}


### (39/90) Anatomy of Neural Language Models (Majd Saleh et al., 2024)

{{<citation>}}

Majd Saleh, Stéphane Paquelet. (2024)  
**Anatomy of Neural Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, BERT, GPT, Generative AI, Language Model, NLP, Natural Language Processing, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.03797v1)  

---


**ABSTRACT**  
Generative AI and transfer learning fields have experienced remarkable advancements in recent years especially in the domain of Natural Language Processing (NLP). Transformers were at the heart of these advancements where the cutting-edge transformer-based Language Models (LMs) enabled new state-of-the-art results in a wide spectrum of applications. While the number of research works involving neural LMs is exponentially increasing, their vast majority are high-level and far from self-contained. Consequently, a deep understanding of the literature in this area is a tough task especially at the absence of a unified mathematical framework explaining the main types of neural LMs. We address the aforementioned problem in this tutorial where the objective is to explain neural LMs in a detailed, simplified and unambiguous mathematical framework accompanied with clear graphical illustrations. Concrete examples on widely used models like BERT and GPT2 are explored. Finally, since transformers pretrained on language-modeling-like tasks have been widely adopted in computer vision and time series applications, we briefly explore some examples of such solutions in order to enable readers understand how transformers work in the aforementioned domains and compare this use with the original one in NLP.

{{</citation>}}


### (40/90) Language Models Understand Numbers, at Least Partially (Fangwei Zhu et al., 2024)

{{<citation>}}

Fangwei Zhu, Damai Dai, Zhifang Sui. (2024)  
**Language Models Understand Numbers, at Least Partially**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03735v1)  

---


**ABSTRACT**  
Large language models (LLMs) have exhibited impressive competency in various text-related tasks. However, their opaque internal mechanisms become a hindrance to leveraging them in mathematical problems. In this paper, we study a fundamental question: whether language models understand numbers, which play a basic element in mathematical problems. We assume that to solve mathematical problems, language models should be capable of understanding numbers and compressing these numbers in their hidden states. We construct a synthetic dataset comprising addition problems and utilize linear probes to read out input numbers from the hidden states of models. Experimental results demonstrate evidence supporting the existence of compressed numbers in the LLaMA-2 model family from early layers. However, the compression process seems to be not lossless, presenting difficulty in precisely reconstructing the original numbers. Further experiments show that language models can utilize the encoded numbers to perform arithmetic computations, and the computational ability scales up with the model size. Our preliminary research suggests that language models exhibit a partial understanding of numbers, offering insights into future investigations about the models' capability of solving mathematical problems.

{{</citation>}}


### (41/90) The Butterfly Effect of Altering Prompts: How Small Changes and Jailbreaks Affect Large Language Model Performance (Abel Salinas et al., 2024)

{{<citation>}}

Abel Salinas, Fred Morstatter. (2024)  
**The Butterfly Effect of Altering Prompts: How Small Changes and Jailbreaks Affect Large Language Model Performance**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.03729v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) are regularly being used to label data across many domains and for myriad tasks. By simply asking the LLM for an answer, or ``prompting,'' practitioners are able to use LLMs to quickly get a response for an arbitrary task. This prompting is done through a series of decisions by the practitioner, from simple wording of the prompt, to requesting the output in a certain data format, to jailbreaking in the case of prompts that address more sensitive topics. In this work, we ask: do variations in the way a prompt is constructed change the ultimate decision of the LLM? We answer this using a series of prompt variations across a variety of text classification tasks. We find that even the smallest of perturbations, such as adding a space at the end of a prompt, can cause the LLM to change its answer. Further, we find that requesting responses in XML and commonly used jailbreaks can have cataclysmic effects on the data labeled by LLMs.

{{</citation>}}


### (42/90) Overview of the 2023 ICON Shared Task on Gendered Abuse Detection in Indic Languages (Aatman Vaidya et al., 2024)

{{<citation>}}

Aatman Vaidya, Arnav Arora, Aditya Joshi, Tarunima Prabhakar. (2024)  
**Overview of the 2023 ICON Shared Task on Gendered Abuse Detection in Indic Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.03677v1)  

---


**ABSTRACT**  
This paper reports the findings of the ICON 2023 on Gendered Abuse Detection in Indic Languages. The shared task deals with the detection of gendered abuse in online text. The shared task was conducted as a part of ICON 2023, based on a novel dataset in Hindi, Tamil and the Indian dialect of English. The participants were given three subtasks with the train dataset consisting of approximately 6500 posts sourced from Twitter. For the test set, approximately 1200 posts were provided. The shared task received a total of 9 registrations. The best F-1 scores are 0.616 for subtask 1, 0.572 for subtask 2 and, 0.616 and 0.582 for subtask 3. The paper contains examples of hateful content owing to its topic.

{{</citation>}}


## eess.SP (2)



### (43/90) Estimating an Executive Summary of a Time Series: The Tendency (Caio Alves et al., 2024)

{{<citation>}}

Caio Alves, Juan M. Restrepo, Jorge M. Ramirez. (2024)  
**Estimating an Executive Summary of a Time Series: The Tendency**  

---
Primary Category: eess.SP  
Categories: cs-IT, eess-SP, eess.SP, math-IT  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.04232v1)  

---


**ABSTRACT**  
In this paper we revisit the problem of decomposing a signal into a tendency and a residual. The tendency describes an executive summary of a signal that encapsulates its notable characteristics while disregarding seemingly random, less interesting aspects. Building upon the Intrinsic Time Decomposition (ITD) and information-theoretical analysis, we introduce two alternative procedures for selecting the tendency from the ITD baselines. The first is based on the maximum extrema prominence, namely the maximum difference between extrema within each baseline. Specifically this method selects the tendency as the baseline from which an ITD step would produce the largest decline of the maximum prominence. The second method uses the rotations from the ITD and selects the tendency as the last baseline for which the associated rotation is statistically stationary. We delve into a comparative analysis of the information content and interpretability of the tendencies obtained by our proposed methods and those obtained through conventional low-pass filtering schemes, particularly the Hodrik-Prescott (HP) filter. Our findings underscore a fundamental distinction in the nature and interpretability of these tendencies, highlighting their context-dependent utility with emphasis in multi-scale signals. Through a series of real-world applications, we demonstrate the computational robustness and practical utility of our proposed tendencies, emphasizing their adaptability and relevance in diverse time series contexts.

{{</citation>}}


### (44/90) Representation Learning for Wearable-Based Applications in the Case of Missing Data (Janosch Jungo et al., 2024)

{{<citation>}}

Janosch Jungo, Yutong Xiang, Shkurta Gashi, Christian Holz. (2024)  
**Representation Learning for Wearable-Based Applications in the Case of Missing Data**  

---
Primary Category: eess.SP  
Categories: cs-AI, cs-LG, eess-SP, eess.SP  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.05437v1)  

---


**ABSTRACT**  
Wearable devices continuously collect sensor data and use it to infer an individual's behavior, such as sleep, physical activity, and emotions. Despite the significant interest and advancements in this field, modeling multimodal sensor data in real-world environments is still challenging due to low data quality and limited data annotations. In this work, we investigate representation learning for imputing missing wearable data and compare it with state-of-the-art statistical approaches. We investigate the performance of the transformer model on 10 physiological and behavioral signals with different masking ratios. Our results show that transformers outperform baselines for missing data imputation of signals that change more frequently, but not for monotonic signals. We further investigate the impact of imputation strategies and masking rations on downstream classification tasks. Our study provides insights for the design and development of masking-based self-supervised learning tasks and advocates the adoption of hybrid-based imputation strategies to address the challenge of missing data in wearable devices.

{{</citation>}}


## cs.CV (19)



### (45/90) SOAP: Cross-sensor Domain Adaptation for 3D Object Detection Using Stationary Object Aggregation Pseudo-labelling (Chengjie Huang et al., 2024)

{{<citation>}}

Chengjie Huang, Vahdat Abdelzad, Sean Sedwards, Krzysztof Czarnecki. (2024)  
**SOAP: Cross-sensor Domain Adaptation for 3D Object Detection Using Stationary Object Aggregation Pseudo-labelling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.04230v1)  

---


**ABSTRACT**  
We consider the problem of cross-sensor domain adaptation in the context of LiDAR-based 3D object detection and propose Stationary Object Aggregation Pseudo-labelling (SOAP) to generate high quality pseudo-labels for stationary objects. In contrast to the current state-of-the-art in-domain practice of aggregating just a few input scans, SOAP aggregates entire sequences of point clouds at the input level to reduce the sensor domain gap. Then, by means of what we call quasi-stationary training and spatial consistency post-processing, the SOAP model generates accurate pseudo-labels for stationary objects, closing a minimum of 30.3% domain gap compared to few-frame detectors. Our results also show that state-of-the-art domain adaptation approaches can achieve even greater performance in combination with SOAP, in both the unsupervised and semi-supervised settings.

{{</citation>}}


### (46/90) FunnyNet-W: Multimodal Learning of Funny Moments in Videos in the Wild (Zhi-Song Liu et al., 2024)

{{<citation>}}

Zhi-Song Liu, Robin Courant, Vicky Kalogeiton. (2024)  
**FunnyNet-W: Multimodal Learning of Funny Moments in Videos in the Wild**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-MM, cs-SD, cs.CV, eess-AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.04210v1)  

---


**ABSTRACT**  
Automatically understanding funny moments (i.e., the moments that make people laugh) when watching comedy is challenging, as they relate to various features, such as body language, dialogues and culture. In this paper, we propose FunnyNet-W, a model that relies on cross- and self-attention for visual, audio and text data to predict funny moments in videos. Unlike most methods that rely on ground truth data in the form of subtitles, in this work we exploit modalities that come naturally with videos: (a) video frames as they contain visual information indispensable for scene understanding, (b) audio as it contains higher-level cues associated with funny moments, such as intonation, pitch and pauses and (c) text automatically extracted with a speech-to-text model as it can provide rich information when processed by a Large Language Model. To acquire labels for training, we propose an unsupervised approach that spots and labels funny audio moments. We provide experiments on five datasets: the sitcoms TBBT, MHD, MUStARD, Friends, and the TED talk UR-Funny. Extensive experiments and analysis show that FunnyNet-W successfully exploits visual, auditory and textual cues to identify funny moments, while our findings reveal FunnyNet-W's ability to predict funny moments in the wild. FunnyNet-W sets the new state of the art for funny moment detection with multimodal cues on all datasets with and without using ground truth information.

{{</citation>}}


### (47/90) GPT-4V(ision) is a Human-Aligned Evaluator for Text-to-3D Generation (Tong Wu et al., 2024)

{{<citation>}}

Tong Wu, Guandao Yang, Zhibing Li, Kai Zhang, Ziwei Liu, Leonidas Guibas, Dahua Lin, Gordon Wetzstein. (2024)  
**GPT-4V(ision) is a Human-Aligned Evaluator for Text-to-3D Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.04092v2)  

---


**ABSTRACT**  
Despite recent advances in text-to-3D generative methods, there is a notable absence of reliable evaluation metrics. Existing metrics usually focus on a single criterion each, such as how well the asset aligned with the input text. These metrics lack the flexibility to generalize to different evaluation criteria and might not align well with human preferences. Conducting user preference studies is an alternative that offers both adaptability and human-aligned results. User studies, however, can be very expensive to scale. This paper presents an automatic, versatile, and human-aligned evaluation metric for text-to-3D generative models. To this end, we first develop a prompt generator using GPT-4V to generate evaluating prompts, which serve as input to compare text-to-3D models. We further design a method instructing GPT-4V to compare two 3D assets according to user-defined criteria. Finally, we use these pairwise comparison results to assign these models Elo ratings. Experimental results suggest our metric strongly align with human preference across different evaluation criteria.

{{</citation>}}


### (48/90) Efficient Multiscale Multimodal Bottleneck Transformer for Audio-Video Classification (Wentao Zhu, 2024)

{{<citation>}}

Wentao Zhu. (2024)  
**Efficient Multiscale Multimodal Bottleneck Transformer for Audio-Video Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs-SD, cs.CV, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.04023v1)  

---


**ABSTRACT**  
In recent years, researchers combine both audio and video signals to deal with challenges where actions are not well represented or captured by visual cues. However, how to effectively leverage the two modalities is still under development. In this work, we develop a multiscale multimodal Transformer (MMT) that leverages hierarchical representation learning. Particularly, MMT is composed of a novel multiscale audio Transformer (MAT) and a multiscale video Transformer [43]. To learn a discriminative cross-modality fusion, we further design multimodal supervised contrastive objectives called audio-video contrastive loss (AVC) and intra-modal contrastive loss (IMC) that robustly align the two modalities. MMT surpasses previous state-of-the-art approaches by 7.3% and 2.1% on Kinetics-Sounds and VGGSound in terms of the top-1 accuracy without external training data. Moreover, the proposed MAT significantly outperforms AST [28] by 22.2%, 4.4% and 4.7% on three public benchmark datasets, and is about 3% more efficient based on the number of FLOPs and 9.8% more efficient based on GPU memory usage.

{{</citation>}}


### (49/90) Efficient Selective Audio Masked Multimodal Bottleneck Transformer for Audio-Video Classification (Wentao Zhu, 2024)

{{<citation>}}

Wentao Zhu. (2024)  
**Efficient Selective Audio Masked Multimodal Bottleneck Transformer for Audio-Video Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs-SD, cs.CV, eess-AS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.04154v1)  

---


**ABSTRACT**  
Audio and video are two most common modalities in the mainstream media platforms, e.g., YouTube. To learn from multimodal videos effectively, in this work, we propose a novel audio-video recognition approach termed audio video Transformer, AVT, leveraging the effective spatio-temporal representation by the video Transformer to improve action recognition accuracy. For multimodal fusion, simply concatenating multimodal tokens in a cross-modal Transformer requires large computational and memory resources, instead we reduce the cross-modality complexity through an audio-video bottleneck Transformer. To improve the learning efficiency of multimodal Transformer, we integrate self-supervised objectives, i.e., audio-video contrastive learning, audio-video matching, and masked audio and video learning, into AVT training, which maps diverse audio and video representations into a common multimodal representation space. We further propose a masked audio segment loss to learn semantic audio activities in AVT. Extensive experiments and ablation studies on three public datasets and two in-house datasets consistently demonstrate the effectiveness of the proposed AVT. Specifically, AVT outperforms its previous state-of-the-art counterparts on Kinetics-Sounds by 8%. AVT also surpasses one of the previous state-of-the-art video Transformers [25] by 10% on VGGSound by leveraging the audio signal. Compared to one of the previous state-of-the-art multimodal methods, MBT [32], AVT is 1.3% more efficient in terms of FLOPs and improves the accuracy by 3.8% on Epic-Kitchens-100.

{{</citation>}}


### (50/90) STAIR: Spatial-Temporal Reasoning with Auditable Intermediate Results for Video Question Answering (Yueqian Wang et al., 2024)

{{<citation>}}

Yueqian Wang, Yuxuan Wang, Kai Chen, Dongyan Zhao. (2024)  
**STAIR: Spatial-Temporal Reasoning with Auditable Intermediate Results for Video Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: AI, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.03901v1)  

---


**ABSTRACT**  
Recently we have witnessed the rapid development of video question answering models. However, most models can only handle simple videos in terms of temporal reasoning, and their performance tends to drop when answering temporal-reasoning questions on long and informative videos. To tackle this problem we propose STAIR, a Spatial-Temporal Reasoning model with Auditable Intermediate Results for video question answering. STAIR is a neural module network, which contains a program generator to decompose a given question into a hierarchical combination of several sub-tasks, and a set of lightweight neural modules to complete each of these sub-tasks. Though neural module networks are already widely studied on image-text tasks, applying them to videos is a non-trivial task, as reasoning on videos requires different abilities. In this paper, we define a set of basic video-text sub-tasks for video question answering and design a set of lightweight modules to complete them. Different from most prior works, modules of STAIR return intermediate outputs specific to their intentions instead of always returning attention maps, which makes it easier to interpret and collaborate with pre-trained models. We also introduce intermediate supervision to make these intermediate outputs more accurate. We conduct extensive experiments on several video question answering datasets under various settings to show STAIR's performance, explainability, compatibility with pre-trained models, and applicability when program annotations are not available. Code: https://github.com/yellow-binary-tree/STAIR

{{</citation>}}


### (51/90) Two-stream joint matching method based on contrastive learning for few-shot action recognition (Long Deng et al., 2024)

{{<citation>}}

Long Deng, Ziqiang Li, Bingxin Zhou, Zhongming Chen, Ao Li, Yongxin Ge. (2024)  
**Two-stream joint matching method based on contrastive learning for few-shot action recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.04150v1)  

---


**ABSTRACT**  
Although few-shot action recognition based on metric learning paradigm has achieved significant success, it fails to address the following issues: (1) inadequate action relation modeling and underutilization of multi-modal information; (2) challenges in handling video matching problems with different lengths and speeds, and video matching problems with misalignment of video sub-actions. To address these issues, we propose a Two-Stream Joint Matching method based on contrastive learning (TSJM), which consists of two modules: Multi-modal Contrastive Learning Module (MCL) and Joint Matching Module (JMM). The objective of the MCL is to extensively investigate the inter-modal mutual information relationships, thereby thoroughly extracting modal information to enhance the modeling of action relationships. The JMM aims to simultaneously address the aforementioned video matching problems. The effectiveness of the proposed method is evaluated on two widely used few shot action recognition datasets, namely, SSv2 and Kinetics. Comprehensive ablation experiments are also conducted to substantiate the efficacy of our proposed approach.

{{</citation>}}


### (52/90) Gramformer: Learning Crowd Counting via Graph-Modulated Transformer (Hui Lin et al., 2024)

{{<citation>}}

Hui Lin, Zhiheng Ma, Xiaopeng Hong, Qinnan Shangguan, Deyu Meng. (2024)  
**Gramformer: Learning Crowd Counting via Graph-Modulated Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.03870v1)  

---


**ABSTRACT**  
Transformer has been popular in recent crowd counting work since it breaks the limited receptive field of traditional CNNs. However, since crowd images always contain a large number of similar patches, the self-attention mechanism in Transformer tends to find a homogenized solution where the attention maps of almost all patches are identical. In this paper, we address this problem by proposing Gramformer: a graph-modulated transformer to enhance the network by adjusting the attention and input node features respectively on the basis of two different types of graphs. Firstly, an attention graph is proposed to diverse attention maps to attend to complementary information. The graph is building upon the dissimilarities between patches, modulating the attention in an anti-similarity fashion. Secondly, a feature-based centrality encoding is proposed to discover the centrality positions or importance of nodes. We encode them with a proposed centrality indices scheme to modulate the node features and similarity relationships. Extensive experiments on four challenging crowd counting datasets have validated the competitiveness of the proposed method. Code is available at {https://github.com/LoraLinH/Gramformer}.

{{</citation>}}


### (53/90) TIER: Text-Image Encoder-based Regression for AIGC Image Quality Assessment (Jiquan Yuan et al., 2024)

{{<citation>}}

Jiquan Yuan, Xinyan Cao, Jinming Che, Qinyuan Wang, Sen Liang, Wei Ren, Jinlong Lin, Xixin Cao. (2024)  
**TIER: Text-Image Encoder-based Regression for AIGC Image Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2401.03854v2)  

---


**ABSTRACT**  
Recently, AIGC image quality assessment (AIGCIQA), which aims to assess the quality of AI-generated images (AIGIs) from a human perception perspective, has emerged as a new topic in computer vision. Unlike common image quality assessment tasks where images are derived from original ones distorted by noise, blur, and compression, \textit{etc.}, in AIGCIQA tasks, images are typically generated by generative models using text prompts. Considerable efforts have been made in the past years to advance AIGCIQA. However, most existing AIGCIQA methods regress predicted scores directly from individual generated images, overlooking the information contained in the text prompts of these images. This oversight partially limits the performance of these AIGCIQA methods. To address this issue, we propose a text-image encoder-based regression (TIER) framework. Specifically, we process the generated images and their corresponding text prompts as inputs, utilizing a text encoder and an image encoder to extract features from these text prompts and generated images, respectively. To demonstrate the effectiveness of our proposed TIER method, we conduct extensive experiments on several mainstream AIGCIQA databases, including AGIQA-1K, AGIQA-3K, and AIGCIQA2023. The experimental results indicate that our proposed TIER method generally demonstrates superior performance compared to baseline in most cases.

{{</citation>}}


### (54/90) Aligned with LLM: a new multi-modal training paradigm for encoding fMRI activity in visual cortex (Shuxiao Ma et al., 2024)

{{<citation>}}

Shuxiao Ma, Linyuan Wang, Senbao Hou, Bin Yan. (2024)  
**Aligned with LLM: a new multi-modal training paradigm for encoding fMRI activity in visual cortex**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, q-bio-NC  
Keywords: Computer Vision, GPT, GPT-4, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.03851v1)  

---


**ABSTRACT**  
Recently, there has been a surge in the popularity of pre trained large language models (LLMs) (such as GPT-4), sweeping across the entire Natural Language Processing (NLP) and Computer Vision (CV) communities. These LLMs have demonstrated advanced multi-modal understanding capabilities and showcased strong performance across various benchmarks. The LLM has started to embody traits of artificial general intelligence, which holds vital guidance for enhancing brain-like characteristics within visual encoding models. Hence, This paper proposes a new multi-modal training paradigm, aligning with LLM, for encoding fMRI activity in visual cortex. Based on this paradigm, we trained an encoding model in fMRI data named the LLM-Visual Encoding Model (LLM-VEM). Specifically, we utilize LLM (miniGPT4) to generate descriptive text for all stimulus images, forming a high-quality textual description set. Moreover, we use the pre-trained text encoder (CLIP) to process these detailed descriptions, obtaining the text embedding features. Next, we use the contrast loss function to minimize the distance between the image embedding features and the text embedding features to complete the alignment operation of the stimulus image and text information. With the assistance of the pre-trained LLM, this alignment process facilitates better learning of the visual encoding model, resulting in higher precision. The final experimental results indicate that our training paradigm has significantly aided in enhancing the performance of the visual encoding model.

{{</citation>}}


### (55/90) UFO: Unidentified Foreground Object Detection in 3D Point Cloud (Hyunjun Choi et al., 2024)

{{<citation>}}

Hyunjun Choi, Hawook Jeong, Jin Young Choi. (2024)  
**UFO: Unidentified Foreground Object Detection in 3D Point Cloud**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.03846v1)  

---


**ABSTRACT**  
In this paper, we raise a new issue on Unidentified Foreground Object (UFO) detection in 3D point clouds, which is a crucial technology in autonomous driving in the wild. UFO detection is challenging in that existing 3D object detectors encounter extremely hard challenges in both 3D localization and Out-of-Distribution (OOD) detection. To tackle these challenges, we suggest a new UFO detection framework including three tasks: evaluation protocol, methodology, and benchmark. The evaluation includes a new approach to measure the performance on our goal, i.e. both localization and OOD detection of UFOs. The methodology includes practical techniques to enhance the performance of our goal. The benchmark is composed of the KITTI Misc benchmark and our additional synthetic benchmark for modeling a more diverse range of UFOs. The proposed framework consistently enhances performance by a large margin across all four baseline detectors: SECOND, PointPillars, PV-RCNN, and PartA2, giving insight for future work on UFO detection in the wild.

{{</citation>}}


### (56/90) Fully Attentional Networks with Self-emerging Token Labeling (Bingyin Zhao et al., 2024)

{{<citation>}}

Bingyin Zhao, Zhiding Yu, Shiyi Lan, Yutao Cheng, Anima Anandkumar, Yingjie Lao, Jose M. Alvarez. (2024)  
**Fully Attentional Networks with Self-emerging Token Labeling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.03844v1)  

---


**ABSTRACT**  
Recent studies indicate that Vision Transformers (ViTs) are robust against out-of-distribution scenarios. In particular, the Fully Attentional Network (FAN) - a family of ViT backbones, has achieved state-of-the-art robustness. In this paper, we revisit the FAN models and improve their pre-training with a self-emerging token labeling (STL) framework. Our method contains a two-stage training framework. Specifically, we first train a FAN token labeler (FAN-TL) to generate semantically meaningful patch token labels, followed by a FAN student model training stage that uses both the token labels and the original class label. With the proposed STL framework, our best model based on FAN-L-Hybrid (77.3M parameters) achieves 84.8% Top-1 accuracy and 42.1% mCE on ImageNet-1K and ImageNet-C, and sets a new state-of-the-art for ImageNet-A (46.1%) and ImageNet-R (56.6%) without using extra data, outperforming the original FAN counterpart by significant margins. The proposed framework also demonstrates significantly enhanced performance on downstream tasks such as semantic segmentation, with up to 1.7% improvement in robustness over the counterpart model. Code is available at https://github.com/NVlabs/STL.

{{</citation>}}


### (57/90) WidthFormer: Toward Efficient Transformer-based BEV View Transformation (Chenhongyi Yang et al., 2024)

{{<citation>}}

Chenhongyi Yang, Tianwei Lin, Lichao Huang, Elliot J. Crowley. (2024)  
**WidthFormer: Toward Efficient Transformer-based BEV View Transformation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.03836v3)  

---


**ABSTRACT**  
In this work, we present WidthFormer, a novel transformer-based Bird's-Eye-View (BEV) 3D detection method tailored for real-time autonomous-driving applications. WidthFormer is computationally efficient, robust and does not require any special engineering effort to deploy. In this work, we propose a novel 3D positional encoding mechanism capable of accurately encapsulating 3D geometric information, which enables our model to generate high-quality BEV representations with only a single transformer decoder layer. This mechanism is also beneficial for existing sparse 3D object detectors. Inspired by the recently-proposed works, we further improve our model's efficiency by vertically compressing the image features when serving as attention keys and values. We also introduce two modules to compensate for potential information loss due to feature compression. Experimental evaluation on the widely-used nuScenes 3D object detection benchmark demonstrates that our method outperforms previous approaches across different 3D detection architectures. More importantly, our model is highly efficient. For example, when using $256\times 704$ input images, it achieves 1.5 ms and 2.8 ms latency on NVIDIA 3090 GPU and Horizon Journey-5 edge computing chips, respectively. Furthermore, WidthFormer also exhibits strong robustness to different degrees of camera perturbations. Our study offers valuable insights into the deployment of BEV transformation methods in real-world, complex road environments. Code is available at https://github.com/ChenhongyiYang/WidthFormer .

{{</citation>}}


### (58/90) Monitoring water contaminants in coastal areas through ML algorithms leveraging atmospherically corrected Sentinel-2 data (Francesca Razzano et al., 2024)

{{<citation>}}

Francesca Razzano, Francesco Mauro, Pietro Di Stasio, Gabriele Meoni, Marco Esposito, Gilda Schirinzi, Silvia Liberata Ullo. (2024)  
**Monitoring water contaminants in coastal areas through ML algorithms leveraging atmospherically corrected Sentinel-2 data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.03792v1)  

---


**ABSTRACT**  
Monitoring water contaminants is of paramount importance, ensuring public health and environmental well-being. Turbidity, a key parameter, poses a significant problem, affecting water quality. Its accurate assessment is crucial for safeguarding ecosystems and human consumption, demanding meticulous attention and action. For this, our study pioneers a novel approach to monitor the Turbidity contaminant, integrating CatBoost Machine Learning (ML) with high-resolution data from Sentinel-2 Level-2A. Traditional methods are labor-intensive while CatBoost offers an efficient solution, excelling in predictive accuracy. Leveraging atmospherically corrected Sentinel-2 data through the Google Earth Engine (GEE), our study contributes to scalable and precise Turbidity monitoring. A specific tabular dataset derived from Hong Kong contaminants monitoring stations enriches our study, providing region-specific insights. Results showcase the viability of this integrated approach, laying the foundation for adopting advanced techniques in global water quality management.

{{</citation>}}


### (59/90) Identifying Important Group of Pixels using Interactions (Kosuke Sumiyasu et al., 2024)

{{<citation>}}

Kosuke Sumiyasu, Kazuhiko Kawamoto, Hiroshi Kera. (2024)  
**Identifying Important Group of Pixels using Interactions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.03785v1)  

---


**ABSTRACT**  
To better understand the behavior of image classifiers, it is useful to visualize the contribution of individual pixels to the model prediction. In this study, we propose a method, MoXI~($\textbf{Mo}$del e$\textbf{X}$planation by $\textbf{I}$nteractions), that efficiently and accurately identifies a group of pixels with high prediction confidence. The proposed method employs game-theoretic concepts, Shapley values and interactions, taking into account the effects of individual pixels and the cooperative influence of pixels on model confidence. Theoretical analysis and experiments demonstrate that our method better identifies the pixels that are highly contributing to the model outputs than widely-used visualization methods using Grad-CAM, Attention rollout, and Shapley value. While prior studies have suffered from the exponential computational cost in the computation of Shapley value and interactions, we show that this can be reduced to linear cost for our task.

{{</citation>}}


### (60/90) NeRFmentation: NeRF-based Augmentation for Monocular Depth Estimation (Casimir Feldmann et al., 2024)

{{<citation>}}

Casimir Feldmann, Niall Siegenheim, Nikolas Hars, Lovro Rabuzin, Mert Ertugrul, Luca Wolfart, Marc Pollefeys, Zuria Bauer, Martin R. Oswald. (2024)  
**NeRFmentation: NeRF-based Augmentation for Monocular Depth Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2401.03771v1)  

---


**ABSTRACT**  
The capabilities of monocular depth estimation (MDE) models are limited by the availability of sufficient and diverse datasets. In the case of MDE models for autonomous driving, this issue is exacerbated by the linearity of the captured data trajectories. We propose a NeRF-based data augmentation pipeline to introduce synthetic data with more diverse viewing directions into training datasets and demonstrate the benefits of our approach to model performance and robustness. Our data augmentation pipeline, which we call "NeRFmentation", trains NeRFs on each scene in the dataset, filters out subpar NeRFs based on relevant metrics, and uses them to generate synthetic RGB-D images captured from new viewing directions. In this work, we apply our technique in conjunction with three state-of-the-art MDE architectures on the popular autonomous driving dataset KITTI, augmenting its training set of the Eigen split. We evaluate the resulting performance gain on the original test set, a separate popular driving set, and our own synthetic test set.

{{</citation>}}


### (61/90) Flying Bird Object Detection Algorithm in Surveillance Video (Ziwei Sun et al., 2024)

{{<citation>}}

Ziwei Sun, Zexi Hua, Hengchao Li, Yan Li. (2024)  
**Flying Bird Object Detection Algorithm in Surveillance Video**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Object Detection  
[Paper Link](http://arxiv.org/abs/2401.03749v1)  

---


**ABSTRACT**  
Aiming at the characteristics of the flying bird object in surveillance video, such as the single frame image feature is not obvious, the size is small in most cases, and asymmetric, this paper proposes a Flying Bird Object Detection method for Surveillance Video (FBOD-SV). Firstly, a new feature aggregation module, the Correlation Attention Feature Aggregation (Co-Attention-FA) module, is designed to aggregate the features of the flying bird object according to the bird object's correlation on multiple consecutive frames of images. Secondly, a Flying Bird Object Detection Network (FBOD-Net) with down-sampling and then up-sampling is designed, which uses a large feature layer that fuses fine spatial information and large receptive field information to detect special multi-scale (mostly small-scale) bird objects. Finally, the SimOTA dynamic label allocation method is applied to One-Category object detection, and the SimOTA-OC dynamic label strategy is proposed to solve the difficult problem of label allocation caused by irregular flying bird objects. In this paper, the algorithm's performance is verified by the experimental data set of the surveillance video of the flying bird object of the traction substation. The experimental results show that the surveillance video flying bird object detection method proposed in this paper effectively improves the detection performance of flying bird objects.

{{</citation>}}


### (62/90) FMA-Net: Flow-Guided Dynamic Filtering and Iterative Feature Refinement with Multi-Attention for Joint Video Super-Resolution and Deblurring (Geunhyuk Youk et al., 2024)

{{<citation>}}

Geunhyuk Youk, Jihyong Oh, Munchurl Kim. (2024)  
**FMA-Net: Flow-Guided Dynamic Filtering and Iterative Feature Refinement with Multi-Attention for Joint Video Super-Resolution and Deblurring**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.03707v1)  

---


**ABSTRACT**  
We present a joint learning scheme of video super-resolution and deblurring, called VSRDB, to restore clean high-resolution (HR) videos from blurry low-resolution (LR) ones. This joint restoration problem has drawn much less attention compared to single restoration problems. In this paper, we propose a novel flow-guided dynamic filtering (FGDF) and iterative feature refinement with multi-attention (FRMA), which constitutes our VSRDB framework, denoted as FMA-Net. Specifically, our proposed FGDF enables precise estimation of both spatio-temporally-variant degradation and restoration kernels that are aware of motion trajectories through sophisticated motion representation learning. Compared to conventional dynamic filtering, the FGDF enables the FMA-Net to effectively handle large motions into the VSRDB. Additionally, the stacked FRMA blocks trained with our novel temporal anchor (TA) loss, which temporally anchors and sharpens features, refine features in a course-to-fine manner through iterative updates. Extensive experiments demonstrate the superiority of the proposed FMA-Net over state-of-the-art methods in terms of both quantitative and qualitative quality. Codes and pre-trained models are available at: https://kaist-viclab.github.io/fmanet-site

{{</citation>}}


### (63/90) GloTSFormer: Global Video Text Spotting Transformer (Han Wang et al., 2024)

{{<citation>}}

Han Wang, Yanjie Wang, Yang Li, Can Huang. (2024)  
**GloTSFormer: Global Video Text Spotting Transformer**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.03694v1)  

---


**ABSTRACT**  
Video Text Spotting (VTS) is a fundamental visual task that aims to predict the trajectories and content of texts in a video. Previous works usually conduct local associations and apply IoU-based distance and complex post-processing procedures to boost performance, ignoring the abundant temporal information and the morphological characteristics in VTS. In this paper, we propose a novel Global Video Text Spotting Transformer GloTSFormer to model the tracking problem as global associations and utilize the Gaussian Wasserstein distance to guide the morphological correlation between frames. Our main contributions can be summarized as three folds. 1). We propose a Transformer-based global tracking method GloTSFormer for VTS and associate multiple frames simultaneously. 2). We introduce a Wasserstein distance-based method to conduct positional associations between frames. 3). We conduct extensive experiments on public datasets. On the ICDAR2015 video dataset, GloTSFormer achieves 56.0 MOTA with 4.6 absolute improvement compared with the previous SOTA method and outperforms the previous Transformer-based method by a significant 8.3 MOTA.

{{</citation>}}


## cs.HC (3)



### (64/90) Learning Racing From an AI Coach: Effects of Multimodal Autonomous Driving Explanations on Driving Performance, Cognitive Load, Expertise, and Trust (Robert Kaufman et al., 2024)

{{<citation>}}

Robert Kaufman, Jean Costa, Everlyne Kimani. (2024)  
**Learning Racing From an AI Coach: Effects of Multimodal Autonomous Driving Explanations on Driving Performance, Cognitive Load, Expertise, and Trust**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-RO, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.04206v2)  

---


**ABSTRACT**  
In a pre-post experiment (n = 41), we test the impact of an AI Coach's explanatory communications modeled after the instructions of human driving experts. Participants were divided into four (4) groups to assess two (2) dimensions of the AI coach's explanations: information type ('what' and 'why'-type explanations) and presentation modality (auditory and visual). We directly compare how AI Coaching sessions employing these techniques impact driving performance, cognitive load, confidence, expertise, and trust in an observation learning context. Through interviews, we delineate the learning process of our participants. Results show that an AI driving coach can be useful for teaching performance driving skills to novices. Comparing between groups, we find the type and modality of information influences performance outcomes. We attribute differences to how information directed attention, mitigated uncertainty, and influenced overload experienced by participants. These, in turn, affected how successfully participants were able to learn. Results suggest efficient, modality-appropriate explanations should be opted for when designing effective HMI communications that can instruct without overwhelming. Further, they support the need to align communications with human learning and cognitive processes. Results are synthesized into eight design implications for future autonomous vehicle HMI and AI coach design.

{{</citation>}}


### (65/90) The Role of Text in Visualizations: How Annotations Shape Perceptions of Bias and Influence Predictions (Chase Stokes et al., 2024)

{{<citation>}}

Chase Stokes, Cindy Xiong Bearfield, Marti A. Hearst. (2024)  
**The Role of Text in Visualizations: How Annotations Shape Perceptions of Bias and Influence Predictions**  

---
Primary Category: cs.HC  
Categories: H-5-0, cs-HC, cs.HC  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.04052v1)  

---


**ABSTRACT**  
This paper investigates the role of text in visualizations, specifically the impact of text position, semantic content, and biased wording. Two empirical studies were conducted based on two tasks (predicting data trends and appraising bias) using two visualization types (bar and line charts). While the addition of text had a minimal effect on how people perceive data trends, there was a significant impact on how biased they perceive the authors to be. This finding revealed a relationship between the degree of bias in textual information and the perception of the authors' bias. Exploratory analyses support an interaction between a person's prediction and the degree of bias they perceived. This paper also develops a crowdsourced method for creating chart annotations that range from neutral to highly biased. This research highlights the need for designers to mitigate potential polarization of readers' opinions based on how authors' ideas are expressed.

{{</citation>}}


### (66/90) Bridging the Skills Gap: Evaluating an AI-Assisted Provider Platform to Support Care Providers with Empathetic Delivery of Protocolized Therapy (William R. Kearns et al., 2024)

{{<citation>}}

William R. Kearns, Jessica Bertram, Myra Divina, Lauren Kemp, Yinzhou Wang, Alex Marin, Trevor Cohen, Weichao Yuwen. (2024)  
**Bridging the Skills Gap: Evaluating an AI-Assisted Provider Platform to Support Care Providers with Empathetic Delivery of Protocolized Therapy**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CL, cs-HC, cs-IR, cs-LG, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.03631v1)  

---


**ABSTRACT**  
Despite the high prevalence and burden of mental health conditions, there is a global shortage of mental health providers. Artificial Intelligence (AI) methods have been proposed as a way to address this shortage, by supporting providers with less extensive training as they deliver care. To this end, we developed the AI-Assisted Provider Platform (A2P2), a text-based virtual therapy interface that includes a response suggestion feature, which supports providers in delivering protocolized therapies empathetically. We studied providers with and without expertise in mental health treatment delivering a therapy session using the platform with (intervention) and without (control) AI-assistance features. Upon evaluation, the AI-assisted system significantly decreased response times by 29.34% (p=0.002), tripled empathic response accuracy (p=0.0001), and increased goal recommendation accuracy by 66.67% (p=0.001) across both user groups compared to the control. Both groups rated the system as having excellent usability.

{{</citation>}}


## cs.RO (4)



### (67/90) RePLan: Robotic Replanning with Perception and Language Models (Marta Skreta et al., 2024)

{{<citation>}}

Marta Skreta, Zihan Zhou, Jia Lin Yuan, Kourosh Darvish, Alán Aspuru-Guzik, Animesh Garg. (2024)  
**RePLan: Robotic Replanning with Perception and Language Models**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.04157v1)  

---


**ABSTRACT**  
Advancements in large language models (LLMs) have demonstrated their potential in facilitating high-level reasoning, logical reasoning and robotics planning. Recently, LLMs have also been able to generate reward functions for low-level robot actions, effectively bridging the interface between high-level planning and low-level robot control. However, the challenge remains that even with syntactically correct plans, robots can still fail to achieve their intended goals. This failure can be attributed to imperfect plans proposed by LLMs or to unforeseeable environmental circumstances that hinder the execution of planned subtasks due to erroneous assumptions about the state of objects. One way to prevent these challenges is to rely on human-provided step-by-step instructions, limiting the autonomy of robotic systems. Vision Language Models (VLMs) have shown remarkable success in tasks such as visual question answering and image captioning. Leveraging the capabilities of VLMs, we present a novel framework called Robotic Replanning with Perception and Language Models (RePLan) that enables real-time replanning capabilities for long-horizon tasks. This framework utilizes the physical grounding provided by a VLM's understanding of the world's state to adapt robot actions when the initial plan fails to achieve the desired goal. We test our approach within four environments containing seven long-horizion tasks. We find that RePLan enables a robot to successfully adapt to unforeseen obstacles while accomplishing open-ended, long-horizon goals, where baseline models cannot. Find more information at https://replan-lm.github.io/replan.github.io/

{{</citation>}}


### (68/90) Digital Twin for Autonomous Surface Vessels for Safe Maritime Navigation (Daniel Menges et al., 2024)

{{<citation>}}

Daniel Menges, Andreas Von Brandis, Adil Rasheed. (2024)  
**Digital Twin for Autonomous Surface Vessels for Safe Maritime Navigation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.04032v1)  

---


**ABSTRACT**  
Autonomous surface vessels (ASVs) play an increasingly important role in the safety and sustainability of open sea operations. Since most maritime accidents are related to human failure, intelligent algorithms for autonomous collision avoidance and path following can drastically reduce the risk in the maritime sector. A DT is a virtual representative of a real physical system and can enhance the situational awareness (SITAW) of such an ASV to generate optimal decisions. This work builds on an existing DT framework for ASVs and demonstrates foundations for enabling predictive, prescriptive, and autonomous capabilities. In this context, sophisticated target tracking approaches are crucial for estimating and predicting the position and motion of other dynamic objects. The applied tracking method is enabled by real-time automatic identification system (AIS) data and synthetic light detection and ranging (Lidar) measurements. To guarantee safety during autonomous operations, we applied a predictive safety filter, based on the concept of nonlinear model predictive control (NMPC). The approaches are implemented into a DT built with the Unity game engine. As a result, this work demonstrates the potential of a DT capable of making predictions, playing through various what-if scenarios, and providing optimal control decisions according to its enhanced SITAW.

{{</citation>}}


### (69/90) Task-Oriented Active Learning of Model Preconditions for Inaccurate Dynamics Models (Alex LaGrassa et al., 2024)

{{<citation>}}

Alex LaGrassa, Moonyoung Lee, Oliver Kroemer. (2024)  
**Task-Oriented Active Learning of Model Preconditions for Inaccurate Dynamics Models**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2401.04007v1)  

---


**ABSTRACT**  
When planning with an inaccurate dynamics model, a practical strategy is to restrict planning to regions of state-action space where the model is accurate: also known as a model precondition. Empirical real-world trajectory data is valuable for defining data-driven model preconditions regardless of the model form (analytical, simulator, learned, etc...). However, real-world data is often expensive and dangerous to collect. In order to achieve data efficiency, this paper presents an algorithm for actively selecting trajectories to learn a model precondition for an inaccurate pre-specified dynamics model. Our proposed techniques address challenges arising from the sequential nature of trajectories, and potential benefit of prioritizing task-relevant data. The experimental analysis shows how algorithmic properties affect performance in three planning scenarios: icy gridworld, simulated plant watering, and real-world plant watering. Results demonstrate an improvement of approximately 80% after only four real-world trajectories when using our proposed techniques.

{{</citation>}}


### (70/90) ExTraCT -- Explainable Trajectory Corrections from language inputs using Textual description of features (J-Anne Yow et al., 2024)

{{<citation>}}

J-Anne Yow, Neha Priyadarshini Garg, Manoj Ramanathan, Wei Tech Ang. (2024)  
**ExTraCT -- Explainable Trajectory Corrections from language inputs using Textual description of features**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.03701v1)  

---


**ABSTRACT**  
Natural language provides an intuitive and expressive way of conveying human intent to robots. Prior works employed end-to-end methods for learning trajectory deformations from language corrections. However, such methods do not generalize to new initial trajectories or object configurations. This work presents ExTraCT, a modular framework for trajectory corrections using natural language that combines Large Language Models (LLMs) for natural language understanding and trajectory deformation functions. Given a scene, ExTraCT generates the trajectory modification features (scene-specific and scene-independent) and their corresponding natural language textual descriptions for the objects in the scene online based on a template. We use LLMs for semantic matching of user utterances to the textual descriptions of features. Based on the feature matched, a trajectory modification function is applied to the initial trajectory, allowing generalization to unseen trajectories and object configurations. Through user studies conducted both in simulation and with a physical robot arm, we demonstrate that trajectories deformed using our method were more accurate and were preferred in about 80\% of cases, outperforming the baseline. We also showcase the versatility of our system in a manipulation task and an assistive feeding task.

{{</citation>}}


## cs.CR (2)



### (71/90) Security and Privacy Issues in Cloud Storage (Norah Asiri, 2024)

{{<citation>}}

Norah Asiri. (2024)  
**Security and Privacy Issues in Cloud Storage**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.04076v1)  

---


**ABSTRACT**  
Even with the vast potential that cloud computing has, so far, it has not been adopted by the consumers with the enthusiasm and pace that it be worthy; this is a very reason statement why consumers still hesitated of using cloud computing for their sensitive data and the threats that prevent the consumers from shifting to use cloud computing in general and cloud storage in particular. The cloud computing inherits the traditional potential security and privacy threats besides its own issues due to its unique structures. Some threats related to cloud computing are the insider malicious attacks from the employees that even sometime the provider unconscious about, the lack of transparency of agreement between consumer and provider, data loss, traffic hijacking, shared technology and insecure application interface. Such threats need remedies to make the consumer use its features in secure way. In this review, we spot the light on the most security and privacy issues which can be attributed as gaps that sometimes the consumers or even the enterprises are not aware of. We also define the parties that involve in scenario of cloud computing that also may attack the entire cloud systems. We also show the consequences of these threats.

{{</citation>}}


### (72/90) A Study on the Security Requirements Analysis to build a Zero Trust-based Remote Work Environment (Haena Kim et al., 2024)

{{<citation>}}

Haena Kim, Yejun Kim, Seungjoo Kim. (2024)  
**A Study on the Security Requirements Analysis to build a Zero Trust-based Remote Work Environment**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Amazon, Azure, Google, Microsoft, Security  
[Paper Link](http://arxiv.org/abs/2401.03675v1)  

---


**ABSTRACT**  
Recently, the usage of cloud services has been increasing annually, and with remote work becoming one of the new forms of employment within enterprises, the security of cloud-based remote work environments has become important. The existing work environment relies on a perimeter security model, where accessing one's resources is based on the assumption that everything within the internal network is secure. However, due to the limitations of the perimeter security model, which assumes the safety of everything within the internal network, the adoption of Zero Trust is now being demanded. Accordingly, NIST and DoD have published guidelines related to Zero Trust architecture. However, these guidelines describe security requirements at an abstract level, focusing on logical architecture. In this paper, we conduct a threat modeling for OpenStack cloud to propose more detailed security requirements compared to NIST and DoD guidelines. Subsequently, we perform a security analysis of commercial cloud services such as Microsoft Azure, Amazon Web Service, and Google Cloud to validate these requirements. The security analysis results identify security requirements that each cloud service fails to satisfy, indicating potential exposure to threats. This paper proposes detailed security requirements based on the Zero Trust model and conducts security analyses of various cloud services accordingly. As a result of the security analysis, we proposed potential threats and countermeasures for cloud services with Zero Trust, and this is intended to help build a secure Zero Trust-based remote work environment.

{{</citation>}}


## cs.IT (1)



### (73/90) Physical Layer Security Performance of Dual RIS-aided V2V NOMA Communications (Farshad Rostami Ghadi et al., 2024)

{{<citation>}}

Farshad Rostami Ghadi, Masoud Kaveh, Kai-Kit Wong, Diego Martin. (2024)  
**Physical Layer Security Performance of Dual RIS-aided V2V NOMA Communications**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.04059v1)  

---


**ABSTRACT**  
This paper investigates the performance of physical layer security (PLS) in a vehicle-to-vehicle (V2V) communication system, where a transmitter vehicle exploits a dual reconfigurable intelligent surface (RIS) to send confidential information to legitimate receiver vehicles under the non-orthogonal multiple access (NOMA) scheme in the presence of an eavesdropper vehicle. In particular, it is assumed that an RIS is near the transmitter vehicle and another RIS is close to the receiver vehicles to provide a wider smart radio environment. Besides, we suppose that the channels between two RISs suffer from the Fisher-Snedecor F fading model. Under this scenario, we first provide the marginal distributions of equivalent channels at the legitimate receiver vehicles by exploiting the central limit theorem (CLT). Then, in order to evaluate the PLS performance of the considered secure communication system, we derive analytical expressions of the average secrecy capacity (ASC), secrecy outage probability (SOP), and secrecy energy efficiency (SEE) by using the Gauss-Laguerre quadrature and the Gaussian quadrature techniques. Moreover, to gain more insights into the secrecy performance, the asymptotic expression of the ASC is obtained. The numerical results indicate that incorporating the dual RIS in the secure V2V communication under the NOMA scheme can significantly provide ultra-reliable transmission and guarantee more secure communication for intelligent transportation systems (ITS).

{{</citation>}}


## cs.IR (3)



### (74/90) Unveiling Bias in Fairness Evaluations of Large Language Models: A Critical Literature Review of Music and Movie Recommendation Systems (Chandan Kumar Sah et al., 2024)

{{<citation>}}

Chandan Kumar Sah, Dr. Lian Xiaoli, Muhammad Mirajul Islam. (2024)  
**Unveiling Bias in Fairness Evaluations of Large Language Models: A Critical Literature Review of Music and Movie Recommendation Systems**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-SE, cs.IR  
Keywords: AI, Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2401.04057v1)  

---


**ABSTRACT**  
The rise of generative artificial intelligence, particularly Large Language Models (LLMs), has intensified the imperative to scrutinize fairness alongside accuracy. Recent studies have begun to investigate fairness evaluations for LLMs within domains such as recommendations. Given that personalization is an intrinsic aspect of recommendation systems, its incorporation into fairness assessments is paramount. Yet, the degree to which current fairness evaluation frameworks account for personalization remains unclear. Our comprehensive literature review aims to fill this gap by examining how existing frameworks handle fairness evaluations of LLMs, with a focus on the integration of personalization factors. Despite an exhaustive collection and analysis of relevant works, we discovered that most evaluations overlook personalization, a critical facet of recommendation systems, thereby inadvertently perpetuating unfair practices. Our findings shed light on this oversight and underscore the urgent need for more nuanced fairness evaluations that acknowledge personalization. Such improvements are vital for fostering equitable development within the AI community.

{{</citation>}}


### (75/90) The Impact of Differential Privacy on Recommendation Accuracy and Popularity Bias (Peter Müllner et al., 2024)

{{<citation>}}

Peter Müllner, Elisabeth Lex, Markus Schedl, Dominik Kowald. (2024)  
**The Impact of Differential Privacy on Recommendation Accuracy and Popularity Bias**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.03883v1)  

---


**ABSTRACT**  
Collaborative filtering-based recommender systems leverage vast amounts of behavioral user data, which poses severe privacy risks. Thus, often, random noise is added to the data to ensure Differential Privacy (DP). However, to date, it is not well understood, in which ways this impacts personalized recommendations. In this work, we study how DP impacts recommendation accuracy and popularity bias, when applied to the training data of state-of-the-art recommendation models. Our findings are three-fold: First, we find that nearly all users' recommendations change when DP is applied. Second, recommendation accuracy drops substantially while recommended item popularity experiences a sharp increase, suggesting that popularity bias worsens. Third, we find that DP exacerbates popularity bias more severely for users who prefer unpopular items than for users that prefer popular items.

{{</citation>}}


### (76/90) Reproducibility Analysis and Enhancements for Multi-Aspect Dense Retriever with Aspect Learning (Keping Bi et al., 2024)

{{<citation>}}

Keping Bi, Xiaojie Sun, Jiafeng Guo, Xueqi Cheng. (2024)  
**Reproducibility Analysis and Enhancements for Multi-Aspect Dense Retriever with Aspect Learning**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2401.03648v1)  

---


**ABSTRACT**  
Multi-aspect dense retrieval aims to incorporate aspect information (e.g., brand and category) into dual encoders to facilitate relevance matching. As an early and representative multi-aspect dense retriever, MADRAL learns several extra aspect embeddings and fuses the explicit aspects with an implicit aspect "OTHER" for final representation. MADRAL was evaluated on proprietary data and its code was not released, making it challenging to validate its effectiveness on other datasets. We failed to reproduce its effectiveness on the public MA-Amazon data, motivating us to probe the reasons and re-examine its components. We propose several component alternatives for comparisons, including replacing "OTHER" with "CLS" and representing aspects with the first several content tokens. Through extensive experiments, we confirm that learning "OTHER" from scratch in aspect fusion is harmful. In contrast, our proposed variants can greatly enhance the retrieval performance. Our research not only sheds light on the limitations of MADRAL but also provides valuable insights for future studies on more powerful multi-aspect dense retrieval models. Code will be released at: https://github.com/sunxiaojie99/Reproducibility-for-MADRAL.

{{</citation>}}


## q-bio.QM (1)



### (77/90) Large language models in bioinformatics: applications and perspectives (Jiajia Liu et al., 2024)

{{<citation>}}

Jiajia Liu, Mengyuan Yang, Yankai Yu, Haixia Xu, Kang Li, Xiaobo Zhou. (2024)  
**Large language models in bioinformatics: applications and perspectives**  

---
Primary Category: q-bio.QM  
Categories: cs-CL, q-bio-QM, q-bio.QM  
Keywords: BERT, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2401.04155v1)  

---


**ABSTRACT**  
Large language models (LLMs) are a class of artificial intelligence models based on deep learning, which have great performance in various tasks, especially in natural language processing (NLP). Large language models typically consist of artificial neural networks with numerous parameters, trained on large amounts of unlabeled input using self-supervised or semi-supervised learning. However, their potential for solving bioinformatics problems may even exceed their proficiency in modeling human language. In this review, we will present a summary of the prominent large language models used in natural language processing, such as BERT and GPT, and focus on exploring the applications of large language models at different omics levels in bioinformatics, mainly including applications of large language models in genomics, transcriptomics, proteomics, drug discovery and single cell analysis. Finally, this review summarizes the potential and prospects of large language models in solving bioinformatic problems.

{{</citation>}}


## cs.SD (1)



### (78/90) Cross-Speaker Encoding Network for Multi-Talker Speech Recognition (Jiawen Kang et al., 2024)

{{<citation>}}

Jiawen Kang, Lingwei Meng, Mingyu Cui, Haohan Guo, Xixin Wu, Xunying Liu, Helen Meng. (2024)  
**Cross-Speaker Encoding Network for Multi-Talker Speech Recognition**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.04152v1)  

---


**ABSTRACT**  
End-to-end multi-talker speech recognition has garnered great interest as an effective approach to directly transcribe overlapped speech from multiple speakers. Current methods typically adopt either 1) single-input multiple-output (SIMO) models with a branched encoder, or 2) single-input single-output (SISO) models based on attention-based encoder-decoder architecture with serialized output training (SOT). In this work, we propose a Cross-Speaker Encoding (CSE) network to address the limitations of SIMO models by aggregating cross-speaker representations. Furthermore, the CSE model is integrated with SOT to leverage both the advantages of SIMO and SISO while mitigating their drawbacks. To the best of our knowledge, this work represents an early effort to integrate SIMO and SISO for multi-talker speech recognition. Experiments on the two-speaker LibrispeechMix dataset show that the CES model reduces word error rate (WER) by 8% over the SIMO baseline. The CSE-SOT model reduces WER by 10% overall and by 16% on high-overlap speech compared to the SOT model.

{{</citation>}}


## cs.AI (1)



### (79/90) Advancing Spatial Reasoning in Large Language Models: An In-Depth Evaluation and Enhancement Using the StepGame Benchmark (Fangjun Li et al., 2024)

{{<citation>}}

Fangjun Li, David C. Hogg, Anthony G. Cohn. (2024)  
**Advancing Spatial Reasoning in Large Language Models: An In-Depth Evaluation and Enhancement Using the StepGame Benchmark**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-DB, cs-LO, cs.AI  
Keywords: AI, ChatGPT, GPT, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.03991v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) has made remarkable progress across various domains, with large language models like ChatGPT gaining substantial attention for their human-like text-generation capabilities. Despite these achievements, spatial reasoning remains a significant challenge for these models. Benchmarks like StepGame evaluate AI spatial reasoning, where ChatGPT has shown unsatisfactory performance. However, the presence of template errors in the benchmark has an impact on the evaluation results. Thus there is potential for ChatGPT to perform better if these template errors are addressed, leading to more accurate assessments of its spatial reasoning capabilities. In this study, we refine the StepGame benchmark, providing a more accurate dataset for model evaluation. We analyze GPT's spatial reasoning performance on the rectified benchmark, identifying proficiency in mapping natural language text to spatial relations but limitations in multi-hop reasoning. We provide a flawless solution to the benchmark by combining template-to-relation mapping with logic-based reasoning. This combination demonstrates proficiency in performing qualitative reasoning on StepGame without encountering any errors. We then address the limitations of GPT models in spatial reasoning. We deploy Chain-of-thought and Tree-of-thoughts prompting strategies, offering insights into GPT's ``cognitive process", and achieving remarkable improvements in accuracy. Our investigation not only sheds light on model deficiencies but also proposes enhancements, contributing to the advancement of AI with more robust spatial reasoning capabilities.

{{</citation>}}


## cs.CE (2)



### (80/90) A Wasserstein Graph Distance Based on Distributions of Probabilistic Node Embeddings (Michael Scholkemper et al., 2024)

{{<citation>}}

Michael Scholkemper, Damin Kühn, Gerion Nabbefeld, Simon Musall, Björn Kampa, Michael T. Schaub. (2024)  
**A Wasserstein Graph Distance Based on Distributions of Probabilistic Node Embeddings**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs-SI, cs.CE  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.03913v2)  

---


**ABSTRACT**  
Distance measures between graphs are important primitives for a variety of learning tasks. In this work, we describe an unsupervised, optimal transport based approach to define a distance between graphs. Our idea is to derive representations of graphs as Gaussian mixture models, fitted to distributions of sampled node embeddings over the same space. The Wasserstein distance between these Gaussian mixture distributions then yields an interpretable and easily computable distance measure, which can further be tailored for the comparison at hand by choosing appropriate embeddings. We propose two embeddings for this framework and show that under certain assumptions about the shape of the resulting Gaussian mixture components, further computational improvements of this Wasserstein distance can be achieved. An empirical validation of our findings on synthetic data and real-world Functional Brain Connectivity networks shows promising performance compared to existing embedding methods.

{{</citation>}}


### (81/90) GrainGNN: A dynamic graph neural network for predicting 3D grain microstructure (Yigong Qin et al., 2024)

{{<citation>}}

Yigong Qin, Stephen DeWitt, Balasubramanian Radhakrishnan, George Biros. (2024)  
**GrainGNN: A dynamic graph neural network for predicting 3D grain microstructure**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.03661v1)  

---


**ABSTRACT**  
We propose GrainGNN, a surrogate model for polycrystalline grain formation under rapid solidification conditions in metal additive manufacturing. Such grain formation problems are modeled by a multicomponent partial differential equation PDE with moving interfaces. The inherent randomness of the PDE initial conditions (grain seeds) necessitates ensemble simulations to predict microstructure statistics, e.g., grain size, aspect ratio, and crystallographic orientation. Currently such ensemble simulations are prohibitively expensive and surrogates are necessary.   In GrainGNN, we use a dynamic graph to represent interface motion and topological changes due to grain coarsening. We use a reduced representation of the microstructure using hand-crafted features; we combine pattern finding and altering graph algorithms with two neural networks, a classifier (for topological changes) and a regressor (for interface motion). Both networks have an encoder-decoder architecture; the encoder has a multi-layer transformer long-short-term-memory architecture; the decoder is a single layer perceptron.   We evaluate GrainGNN by comparing it to high-fidelity phase field simulations for in-distribution and out-of-distribution grain configurations for solidification under laser power bed fusion conditions. GrainGNN results in 80\%--90\% pointwise accuracy; and nearly identical distributions of scalar quantities of interest (QoI) between phase field and GrainGNN simulations compared using Kolmogorov-Smirnov test. GrainGNN's inference speedup (PyTorch on single x86 node) over a high-fidelity phase-field simulation (CUDA on a single NVIDIA A100 GPU) is 300$\times$--2000$\times$ for 100-initial grain problem. Further, using GrainGNN, we model the formation of 11,600 grains in 220 seconds on a single CPU core.

{{</citation>}}


## eess.IV (2)



### (82/90) Attention-Guided Erasing: A Novel Augmentation Method for Enhancing Downstream Breast Density Classification (Adarsh Bhandary Panambur et al., 2024)

{{<citation>}}

Adarsh Bhandary Panambur, Hui Yu, Sheethal Bhat, Prathmesh Madhu, Siming Bayer, Andreas Maier. (2024)  
**Attention-Guided Erasing: A Novel Augmentation Method for Enhancing Downstream Breast Density Classification**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention, Augmentation  
[Paper Link](http://arxiv.org/abs/2401.03912v1)  

---


**ABSTRACT**  
The assessment of breast density is crucial in the context of breast cancer screening, especially in populations with a higher percentage of dense breast tissues. This study introduces a novel data augmentation technique termed Attention-Guided Erasing (AGE), devised to enhance the downstream classification of four distinct breast density categories in mammography following the BI-RADS recommendation in the Vietnamese cohort. The proposed method integrates supplementary information during transfer learning, utilizing visual attention maps derived from a vision transformer backbone trained using the self-supervised DINO method. These maps are utilized to erase background regions in the mammogram images, unveiling only the potential areas of dense breast tissues to the network. Through the incorporation of AGE during transfer learning with varying random probabilities, we consistently surpass classification performance compared to scenarios without AGE and the traditional random erasing transformation. We validate our methodology using the publicly available VinDr-Mammo dataset. Specifically, we attain a mean F1-score of 0.5910, outperforming values of 0.5594 and 0.5691 corresponding to scenarios without AGE and with random erasing (RE), respectively. This superiority is further substantiated by t-tests, revealing a p-value of p<0.0001, underscoring the statistical significance of our approach.

{{</citation>}}


### (83/90) Dual-Channel Reliable Breast Ultrasound Image Classification Based on Explainable Attribution and Uncertainty Quantification (Shuge Lei et al., 2024)

{{<citation>}}

Shuge Lei, Haonan Hu, Dasheng Sun, Huabin Zhang, Kehong Yuan, Jian Dai, Jijun Tang, Yan Tong. (2024)  
**Dual-Channel Reliable Breast Ultrasound Image Classification Based on Explainable Attribution and Uncertainty Quantification**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2401.03664v1)  

---


**ABSTRACT**  
This paper focuses on the classification task of breast ultrasound images and researches on the reliability measurement of classification results. We proposed a dual-channel evaluation framework based on the proposed inference reliability and predictive reliability scores. For the inference reliability evaluation, human-aligned and doctor-agreed inference rationales based on the improved feature attribution algorithm SP-RISA are gracefully applied. Uncertainty quantification is used to evaluate the predictive reliability via the Test Time Enhancement. The effectiveness of this reliability evaluation framework has been verified on our breast ultrasound clinical dataset YBUS, and its robustness is verified on the public dataset BUSI. The expected calibration errors on both datasets are significantly lower than traditional evaluation methods, which proves the effectiveness of our proposed reliability measurement.

{{</citation>}}


## cs.AR (1)



### (84/90) FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs (Shulin Zeng et al., 2024)

{{<citation>}}

Shulin Zeng, Jun Liu, Guohao Dai, Xinhao Yang, Tianyu Fu, Hongyi Wang, Wenheng Ma, Hanbo Sun, Shiyao Li, Zixiao Huang, Yadong Dai, Jintao Li, Zehao Wang, Ruoyu Zhang, Kairui Wen, Xuefei Ning, Yu Wang. (2024)  
**FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs**  

---
Primary Category: cs.AR  
Categories: cs-AI, cs-AR, cs.AR  
Keywords: LLaMA, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.03868v2)  

---


**ABSTRACT**  
Transformer-based Large Language Models (LLMs) have made a significant impact on various domains. However, LLMs' efficiency suffers from both heavy computation and memory overheads. Compression techniques like sparsification and quantization are commonly used to mitigate the gap between LLM's computation/memory overheads and hardware capacity. However, existing GPU and transformer-based accelerators cannot efficiently process compressed LLMs, due to the following unresolved challenges: low computational efficiency, underutilized memory bandwidth, and large compilation overheads.   This paper proposes FlightLLM, enabling efficient LLMs inference with a complete mapping flow on FPGAs. In FlightLLM, we highlight an innovative solution that the computation and memory overhead of LLMs can be solved by utilizing FPGA-specific resources (e.g., DSP48 and heterogeneous memory hierarchy). We propose a configurable sparse DSP chain to support different sparsity patterns with high computation efficiency. Second, we propose an always-on-chip decode scheme to boost memory bandwidth with mixed-precision support. Finally, to make FlightLLM available for real-world LLMs, we propose a length adaptive compilation method to reduce the compilation overhead. Implemented on the Xilinx Alveo U280 FPGA, FlightLLM achieves 6.0$\times$ higher energy efficiency and 1.8$\times$ better cost efficiency against commercial GPUs (e.g., NVIDIA V100S) on modern LLMs (e.g., LLaMA2-7B) using vLLM and SmoothQuant under the batch size of one. FlightLLM beats NVIDIA A100 GPU with 1.2$\times$ higher throughput using the latest Versal VHK158 FPGA.

{{</citation>}}


## cs.CG (1)



### (85/90) Range Reporting for Time Series via Rectangle Stabbing (Lotte Blank et al., 2024)

{{<citation>}}

Lotte Blank, Anne Driemel. (2024)  
**Range Reporting for Time Series via Rectangle Stabbing**  

---
Primary Category: cs.CG  
Categories: cs-CG, cs.CG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.03762v1)  

---


**ABSTRACT**  
We study the Fr\'echet queries problem. It is a data structure problem, where we are given a set $S$ of $n$ polygonal curves and a distance threshold $\rho$. The data structure should support queries with a polygonal curve $q$ for the elements of $S$, for which the continuous Fr\'echet distance to $q$ is at most $\rho$. Afshani and Driemel in 2018 studied this problem for two-dimensional polygonal curves and gave upper and lower bounds on the space-query time tradeoff. We study the case that the ambient space of the curves is one-dimensional and show an intimate connection to the well-studied rectangle stabbing problem. Here, we are given a set of hyperrectangles as input and a query with a point $q$ should return all input rectangles that contain this point. Using known data structures for rectangle stabbing or orthogonal range searching this directly leads to a data structure with $\mathcal{O}(n \log ^{t-1} n)$ storage and $\mathcal{O}(\log^{t-1} n+k)$ query time, where $k$ denotes the output size and $t$ can be chosen as the maximum number of vertices of either (a) the stored curves or (b) the query curves. The resulting bounds improve upon the bounds by Afshani and Driemel in both the storage and query time. In addition, we show that known lower bounds for rectangle stabbing and orthogonal range reporting with dimension parameter $d= \lfloor t/2 \rfloor$ can be applied to our problem via reduction. .

{{</citation>}}


## q-fin.CP (1)



### (86/90) Can Large Language Models Beat Wall Street? Unveiling the Potential of AI in Stock Selection (Georgios Fatouros et al., 2024)

{{<citation>}}

Georgios Fatouros, Konstantinos Metaxas, John Soldatos, Dimosthenis Kyriazis. (2024)  
**Can Large Language Models Beat Wall Street? Unveiling the Potential of AI in Stock Selection**  

---
Primary Category: q-fin.CP  
Categories: 68T07, 68T50, 91G10, 91G15, I-2-1; I-2-7; J-4, cs-AI, cs-CE, cs-CL, cs-LG, q-fin-CP, q-fin.CP  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03737v1)  

---


**ABSTRACT**  
In the dynamic and data-driven landscape of financial markets, this paper introduces MarketSenseAI, a novel AI-driven framework leveraging the advanced reasoning capabilities of GPT-4 for scalable stock selection. MarketSenseAI incorporates Chain of Thought and In-Context Learning methodologies to analyze a wide array of data sources, including market price dynamics, financial news, company fundamentals, and macroeconomic reports emulating the decision making process of prominent financial investment teams. The development, implementation, and empirical validation of MarketSenseAI are detailed, with a focus on its ability to provide actionable investment signals (buy, hold, sell) backed by cogent explanations. A notable aspect of this study is the use of GPT-4 not only as a predictive tool but also as an evaluator, revealing the significant impact of the AI-generated explanations on the reliability and acceptance of the suggested investment signals. In an extensive empirical evaluation with S&P 100 stocks, MarketSenseAI outperformed the benchmark index by 13%, achieving returns up to 40%, while maintaining a risk profile comparable to the market. These results demonstrate the efficacy of Large Language Models in complex financial decision-making and mark a significant advancement in the integration of AI into financial analysis and investment strategies. This research contributes to the financial AI field, presenting an innovative approach and underscoring the transformative potential of AI in revolutionizing traditional financial analysis investment methodologies.

{{</citation>}}


## cs.DB (1)



### (87/90) Sibyl: Forecasting Time-Evolving Query Workloads (Hanxian Huang et al., 2024)

{{<citation>}}

Hanxian Huang, Tarique Siddiqui, Rana Alotaibi, Carlo Curino, Jyoti Leeka, Alekh Jindal, Jishen Zhao, Jesus Camacho-Rodriguez, Yuanyuan Tian. (2024)  
**Sibyl: Forecasting Time-Evolving Query Workloads**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs-LG, cs.DB  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.03723v1)  

---


**ABSTRACT**  
Database systems often rely on historical query traces to perform workload-based performance tuning. However, real production workloads are time-evolving, making historical queries ineffective for optimizing future workloads. To address this challenge, we propose SIBYL, an end-to-end machine learning-based framework that accurately forecasts a sequence of future queries, with the entire query statements, in various prediction windows. Drawing insights from real-workloads, we propose template-based featurization techniques and develop a stacked-LSTM with an encoder-decoder architecture for accurate forecasting of query workloads. We also develop techniques to improve forecasting accuracy over large prediction windows and achieve high scalability over large workloads with high variability in arrival rates of queries. Finally, we propose techniques to handle workload drifts. Our evaluation on four real workloads demonstrates that SIBYL can forecast workloads with an $87.3\%$ median F1 score, and can result in $1.7\times$ and $1.3\times$ performance improvement when applied to materialized view selection and index selection applications, respectively.

{{</citation>}}


## cs.NE (1)



### (88/90) Enhancing Adaptive History Reserving by Spiking Convolutional Block Attention Module in Recurrent Neural Networks (Qi Xu et al., 2024)

{{<citation>}}

Qi Xu, Yuyuan Gao, Jiangrong Shen, Yaxin Li, Xuming Ran, Huajin Tang, Gang Pan. (2024)  
**Enhancing Adaptive History Reserving by Spiking Convolutional Block Attention Module in Recurrent Neural Networks**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.03719v1)  

---


**ABSTRACT**  
Spiking neural networks (SNNs) serve as one type of efficient model to process spatio-temporal patterns in time series, such as the Address-Event Representation data collected from Dynamic Vision Sensor (DVS). Although convolutional SNNs have achieved remarkable performance on these AER datasets, benefiting from the predominant spatial feature extraction ability of convolutional structure, they ignore temporal features related to sequential time points. In this paper, we develop a recurrent spiking neural network (RSNN) model embedded with an advanced spiking convolutional block attention module (SCBAM) component to combine both spatial and temporal features of spatio-temporal patterns. It invokes the history information in spatial and temporal channels adaptively through SCBAM, which brings the advantages of efficient memory calling and history redundancy elimination. The performance of our model was evaluated in DVS128-Gesture dataset and other time-series datasets. The experimental results show that the proposed SRNN-SCBAM model makes better use of the history information in spatial and temporal dimensions with less memory space, and achieves higher accuracy compared to other models.

{{</citation>}}


## math.OC (1)



### (89/90) Boosting Column Generation with Graph Neural Networks for Joint Rider Trip Planning and Crew Shift Scheduling (Jiawei Lu et al., 2024)

{{<citation>}}

Jiawei Lu, Tinghan Ye, Wenbo Chen, Pascal Van Hentenryck. (2024)  
**Boosting Column Generation with Graph Neural Networks for Joint Rider Trip Planning and Crew Shift Scheduling**  

---
Primary Category: math.OC  
Categories: cs-LG, math-OC, math.OC  
Keywords: Attention, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.03692v1)  

---


**ABSTRACT**  
Optimizing service schedules is pivotal to the reliable, efficient, and inclusive on-demand mobility. This pressing challenge is further exacerbated by the increasing needs of an aging population, the over-subscription of existing services, and the lack of effective solution methods. This study addresses the intricacies of service scheduling, by jointly optimizing rider trip planning and crew scheduling for a complex dynamic mobility service. The resulting optimization problems are extremely challenging computationally for state-of-the-art methods. To address this fundamental gap, this paper introduces the Joint Rider Trip Planning and Crew Shift Scheduling Problem (JRTPCSSP) and a novel solution method, called AGGNNI-CG (Attention and Gated GNN- Informed Column Generation), that hybridizes column generation and machine learning to obtain near-optimal solutions to the JRTPCSSP with the real-time constraints of the application. The key idea of the machine-learning component is to dramatically reduce the number of paths to explore in the pricing component, accelerating the most time-consuming component of the column generation. The machine learning component is a graph neural network with an attention mechanism and a gated architecture, that is particularly suited to cater for the different input sizes coming from daily operations. AGGNNI-CG has been applied to a challenging, real-world dataset from the Paratransit system of Chatham County in Georgia. It produces dramatic improvements compared to the baseline column generation approach, which typically cannot produce feasible solutions in reasonable time on both medium-sized and large-scale complex instances. AGGNNI-CG also produces significant improvements in service compared to the existing system.

{{</citation>}}


## cs.MA (1)



### (90/90) Why Solving Multi-agent Path Finding with Large Language Model has not Succeeded Yet (Weizhe Chen et al., 2024)

{{<citation>}}

Weizhe Chen, Sven Koenig, Bistra Dilkina. (2024)  
**Why Solving Multi-agent Path Finding with Large Language Model has not Succeeded Yet**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-CL, cs-MA, cs.MA  
Keywords: ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.03630v1)  

---


**ABSTRACT**  
With the explosive influence caused by the success of large language models (LLM) like ChatGPT and GPT-4, there has been an extensive amount of recent work showing that foundation models can be used to solve a large variety of tasks. However, there is very limited work that shares insights on multi-agent planning. Multi-agent planning is different from other domains by combining the difficulty of multi-agent coordination and planning, and making it hard to leverage external tools to facilitate the reasoning needed. In this paper, we focus on the problem of multi-agent path finding (MAPF), which is also known as multi-robot route planning, and study how to solve MAPF with LLMs. We first show the motivating success on an empty room map without obstacles, then the failure to plan on a slightly harder room map. We present our hypothesis of why directly solving MAPF with LLMs has not been successful yet, and we use various experiments to support our hypothesis.

{{</citation>}}
