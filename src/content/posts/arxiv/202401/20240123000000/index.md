---
draft: false
title: "arXiv @ 2024.01.23"
date: 2024-01-23
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.23"
    identifier: arxiv_20240123
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (11)](#cslg-11)
- [cs.HC (1)](#cshc-1)
- [cs.CL (15)](#cscl-15)
- [cs.IT (1)](#csit-1)
- [cs.CV (13)](#cscv-13)
- [quant-ph (3)](#quant-ph-3)
- [cs.CR (1)](#cscr-1)
- [cs.MA (1)](#csma-1)
- [cs.IR (4)](#csir-4)
- [cs.RO (5)](#csro-5)
- [cs.DC (1)](#csdc-1)
- [cs.AR (1)](#csar-1)
- [eess.SP (1)](#eesssp-1)
- [cs.NI (1)](#csni-1)
- [cs.SE (1)](#csse-1)
- [physics.flu-dyn (1)](#physicsflu-dyn-1)

## cs.LG (11)



### (1/61) Reframing Offline Reinforcement Learning as a Regression Problem (Prajwal Koirala et al., 2024)

{{<citation>}}

Prajwal Koirala, Cody Fleming. (2024)  
**Reframing Offline Reinforcement Learning as a Regression Problem**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11630v1)  

---


**ABSTRACT**  
The study proposes the reformulation of offline reinforcement learning as a regression problem that can be solved with decision trees. Aiming to predict actions based on input states, return-to-go (RTG), and timestep information, we observe that with gradient-boosted trees, the agent training and inference are very fast, the former taking less than a minute. Despite the simplification inherent in this reformulated problem, our agent demonstrates performance that is at least on par with established methods. This assertion is validated by testing it across standard datasets associated with D4RL Gym-MuJoCo tasks. We further discuss the agent's ability to generalize by testing it on two extreme cases, how it learns to model the return distributions effectively even with highly skewed expert datasets, and how it exhibits robust performance in scenarios with sparse/delayed rewards.

{{</citation>}}


### (2/61) Freely Long-Thinking Transformer (FraiLT) (Akbay Tabak, 2024)

{{<citation>}}

Akbay Tabak. (2024)  
**Freely Long-Thinking Transformer (FraiLT)**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.11626v1)  

---


**ABSTRACT**  
Freely Long-Thinking Transformer (FraiLT) is an improved transformer model designed to enhance processing capabilities without scaling up size. It utilizes a recursive approach, iterating over a subset of layers multiple times, and introduces iteration encodings to maintain awareness across these cycles. Iteration encoding allows FraiLT to achieve the interpretive depth of larger models in a compact form. When evaluated on a synthetic story dataset, FraiLT outperformed larger models, showcasing its ability to deliver high-quality performance while reducing memory demands. This model represents a step forward towards more efficient and accessible language models.

{{</citation>}}


### (3/61) Graph Edits for Counterfactual Explanations: A Unified GNN Approach (Nikolaos Chaidos et al., 2024)

{{<citation>}}

Nikolaos Chaidos, Angeliki Dimitriou, Maria Lymperaiou, Giorgos Stamou. (2024)  
**Graph Edits for Counterfactual Explanations: A Unified GNN Approach**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.11609v1)  

---


**ABSTRACT**  
Counterfactuals have been established as a popular explainability technique which leverages a set of minimal edits to alter the prediction of a classifier. When considering conceptual counterfactuals, the edits requested should correspond to salient concepts present in the input data. At the same time, conceptual distances are defined by knowledge graphs, ensuring the optimality of conceptual edits. In this work, we extend previous endeavors on conceptual counterfactuals by introducing \textit{graph edits as counterfactual explanations}: should we represent input data as graphs, which is the shortest graph edit path that results in an alternative classification label as provided by a black-box classifier?

{{</citation>}}


### (4/61) Information-Theoretic State Variable Selection for Reinforcement Learning (Charles Westphal et al., 2024)

{{<citation>}}

Charles Westphal, Stephen Hailes, Mirco Musolesi. (2024)  
**Information-Theoretic State Variable Selection for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-IT, cs-LG, cs.LG, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11512v1)  

---


**ABSTRACT**  
Identifying the most suitable variables to represent the state is a fundamental challenge in Reinforcement Learning (RL). These variables must efficiently capture the information necessary for making optimal decisions. In order to address this problem, in this paper, we introduce the Transfer Entropy Redundancy Criterion (TERC), an information-theoretic criterion, which determines if there is \textit{entropy transferred} from state variables to actions during training. We define an algorithm based on TERC that provably excludes variables from the state that have no effect on the final performance of the agent, resulting in more sample efficient learning. Experimental results show that this speed-up is present across three different algorithm classes (represented by tabular Q-learning, Actor-Critic, and Proximal Policy Optimization (PPO)) in a variety of environments. Furthermore, to highlight the differences between the proposed methodology and the current state-of-the-art feature selection approaches, we present a series of controlled experiments on synthetic data, before generalizing to real-world decision-making tasks. We also introduce a representation of the problem that compactly captures the transfer of information from state variables to actions as Bayesian networks.

{{</citation>}}


### (5/61) Sequential Model for Predicting Patient Adherence in Subcutaneous Immunotherapy for Allergic Rhinitis (Li Yin et al., 2024)

{{<citation>}}

Li Yin, Xiong Yu, Fan Wenxin, Wang Kai, Yu Qingqing, Si Liping, van der Smagt Patrick, Tang Jun, Chen Nutan. (2024)  
**Sequential Model for Predicting Patient Adherence in Subcutaneous Immunotherapy for Allergic Rhinitis**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-QM  
Keywords: AI, LSTM  
[Paper Link](http://arxiv.org/abs/2401.11447v1)  

---


**ABSTRACT**  
Objective: Subcutaneous Immunotherapy (SCIT) is the long-lasting causal treatment of allergic rhinitis. How to enhance the adherence of patients to maximize the benefit of allergen immunotherapy (AIT) plays a crucial role in the management of AIT. This study aims to leverage novel machine learning models to precisely predict the risk of non-adherence of patients and related systematic symptom scores, to provide a novel approach in the management of long-term AIT.   Methods: The research develops and analyzes two models, Sequential Latent Actor-Critic (SLAC) and Long Short-Term Memory (LSTM), evaluating them based on scoring and adherence prediction capabilities.   Results: Excluding the biased samples at the first time step, the predictive adherence accuracy of the SLAC models is from $60\,\%$ to $72\%$, and for LSTM models, it is $66\,\%$ to $84\,\%$, varying according to the time steps. The range of Root Mean Square Error (RMSE) for SLAC models is between $0.93$ and $2.22$, while for LSTM models it is between $1.09$ and $1.77$. Notably, these RMSEs are significantly lower than the random prediction error of $4.55$.   Conclusion: We creatively apply sequential models in the long-term management of SCIT with promising accuracy in the prediction of SCIT nonadherence in Allergic Rhinitis (AR) patients. While LSTM outperforms SLAC in adherence prediction, SLAC excels in score prediction for patients undergoing SCIT for AR. The state-action-based SLAC adds flexibility, presenting a novel and effective approach for managing long-term AIT.

{{</citation>}}


### (6/61) Open the Black Box: Step-based Policy Updates for Temporally-Correlated Episodic Reinforcement Learning (Ge Li et al., 2024)

{{<citation>}}

Ge Li, Hongyi Zhou, Dominik Roth, Serge Thilges, Fabian Otto, Rudolf Lioutikov, Gerhard Neumann. (2024)  
**Open the Black Box: Step-based Policy Updates for Temporally-Correlated Episodic Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11437v1)  

---


**ABSTRACT**  
Current advancements in reinforcement learning (RL) have predominantly focused on learning step-based policies that generate actions for each perceived state. While these methods efficiently leverage step information from environmental interaction, they often ignore the temporal correlation between actions, resulting in inefficient exploration and unsmooth trajectories that are challenging to implement on real hardware. Episodic RL (ERL) seeks to overcome these challenges by exploring in parameters space that capture the correlation of actions. However, these approaches typically compromise data efficiency, as they treat trajectories as opaque \emph{black boxes}. In this work, we introduce a novel ERL algorithm, Temporally-Correlated Episodic RL (TCE), which effectively utilizes step information in episodic policy updates, opening the 'black box' in existing ERL methods while retaining the smooth and consistent exploration in parameter space. TCE synergistically combines the advantages of step-based and episodic RL, achieving comparable performance to recent ERL methods while maintaining data efficiency akin to state-of-the-art (SoTA) step-based RL.

{{</citation>}}


### (7/61) Agricultural Recommendation System based on Deep Learning: A Multivariate Weather Forecasting Approach (Md Zubair et al., 2024)

{{<citation>}}

Md Zubair, Md. Shahidul Salim, Mehrab Mustafy Rahman, Mohammad Jahid Ibna Basher, Shahin Imran, Iqbal H. Sarker. (2024)  
**Agricultural Recommendation System based on Deep Learning: A Multivariate Weather Forecasting Approach**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.11410v1)  

---


**ABSTRACT**  
Bangladesh is predominantly an agricultural country, where the agrarian sector plays an essential role in accelerating economic growth and enabling the food security of the people. The performance of this sector has an overwhelming impact on the primary macroeconomic objectives like food security, employment generation, poverty alleviation, human resources development, and other economic and social forces. Although Bangladesh's labor-intensive agriculture has achieved steady increases in food grain production, it often suffered from unfavorable weather conditions such as heavy rainfall, low temperature, and drought. Consequently, these factors hinder the production of food substantially, putting the country's overall food security in danger. In order to have a profitable, sustainable, and farmer-friendly agricultural practice, this paper proposes a context-based crop recommendation system powered by a weather forecast model. With extensive evaluation, the multivariate Stacked Bi-LSTM Network is employed as the weather forecasting model. The proposed weather model can forecast Rainfall, Temperature, Humidity, and Sunshine for any given location in Bangladesh with higher accuracy. These predictions guide our system to assist the farmers in making feasible decisions about planting, irrigation, harvesting, and so on. Additionally, our full-fledged system is capable of alerting the farmers about extreme weather conditions so that preventive measures can be undertaken to protect the crops. Finally, the system is also adept at making knowledge-based crop suggestions for the flood and drought-prone regions of Bangladesh.

{{</citation>}}


### (8/61) Visual Imitation Learning with Calibrated Contrastive Representation (Yunke Wang et al., 2024)

{{<citation>}}

Yunke Wang, Linwei Tao, Bo Du, Yutian Lin, Chang Xu. (2024)  
**Visual Imitation Learning with Calibrated Contrastive Representation**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11396v1)  

---


**ABSTRACT**  
Adversarial Imitation Learning (AIL) allows the agent to reproduce expert behavior with low-dimensional states and actions. However, challenges arise in handling visual states due to their less distinguishable representation compared to low-dimensional proprioceptive features. While existing methods resort to adopt complex network architectures or separate the process of learning representation and decision-making, they overlook valuable intra-agent information within demonstrations. To address this problem, this paper proposes a simple and effective solution by incorporating calibrated contrastive representative learning into visual AIL framework. Specifically, we present an image encoder in visual AIL, utilizing a combination of unsupervised and supervised contrastive learning to extract valuable features from visual states. Based on the fact that the improved agent often produces demonstrations of varying quality, we propose to calibrate the contrastive loss by treating each agent demonstrations as a mixed sample. The incorporation of contrastive learning can be jointly optimized with the AIL framework, without modifying the architecture or incurring significant computational costs. Experimental results on DMControl Suite demonstrate our proposed method is sample efficient and can outperform other compared methods from different aspects.

{{</citation>}}


### (9/61) Causal Generative Explainers using Counterfactual Inference: A Case Study on the Morpho-MNIST Dataset (Will Taylor-Melanson et al., 2024)

{{<citation>}}

Will Taylor-Melanson, Zahra Sadeghi, Stan Matwin. (2024)  
**Causal Generative Explainers using Counterfactual Inference: A Case Study on the Morpho-MNIST Dataset**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11394v1)  

---


**ABSTRACT**  
In this paper, we propose leveraging causal generative learning as an interpretable tool for explaining image classifiers. Specifically, we present a generative counterfactual inference approach to study the influence of visual features (i.e., pixels) as well as causal factors through generative learning. To this end, we first uncover the most influential pixels on a classifier's decision by varying the value of a causal attribute via counterfactual inference and computing both Shapely and contrastive explanations for counterfactual images with these different attribute values. We then establish a Monte-Carlo mechanism using the generator of a causal generative model in order to adapt Shapley explainers to produce feature importances for the human-interpretable attributes of a causal dataset in the case where a classifier has been trained exclusively on the images of the dataset. Finally, we present optimization methods for creating counterfactual explanations of classifiers by means of counterfactual inference, proposing straightforward approaches for both differentiable and arbitrary classifiers. We exploit the Morpho-MNIST causal dataset as a case study for exploring our proposed methods for generating counterfacutl explantions. We employ visual explanation methods from OmnixAI open source toolkit to compare them with our proposed methods. By employing quantitative metrics to measure the interpretability of counterfactual explanations, we find that our proposed methods of counterfactual explanation offer more interpretable explanations compared to those generated from OmnixAI. This finding suggests that our methods are well-suited for generating highly interpretable counterfactual explanations on causal datasets.

{{</citation>}}


### (10/61) MoMA: Model-based Mirror Ascent for Offline Reinforcement Learning (Mao Hong et al., 2024)

{{<citation>}}

Mao Hong, Zhiyue Zhang, Yue Wu, Yanxun Xu. (2024)  
**MoMA: Model-based Mirror Ascent for Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-ST, stat-ME, stat-ML, stat-TH  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11380v1)  

---


**ABSTRACT**  
Model-based offline reinforcement learning methods (RL) have achieved state-of-the-art performance in many decision-making problems thanks to their sample efficiency and generalizability. Despite these advancements, existing model-based offline RL approaches either focus on theoretical studies without developing practical algorithms or rely on a restricted parametric policy space, thus not fully leveraging the advantages of an unrestricted policy space inherent to model-based methods. To address this limitation, we develop MoMA, a model-based mirror ascent algorithm with general function approximations under partial coverage of offline data. MoMA distinguishes itself from existing literature by employing an unrestricted policy class. In each iteration, MoMA conservatively estimates the value function by a minimization procedure within a confidence set of transition models in the policy evaluation step, then updates the policy with general function approximations instead of commonly-used parametric policy classes in the policy improvement step. Under some mild assumptions, we establish theoretical guarantees of MoMA by proving an upper bound on the suboptimality of the returned policy. We also provide a practically implementable, approximate version of the algorithm. The effectiveness of MoMA is demonstrated via numerical studies.

{{</citation>}}


### (11/61) PepHarmony: A Multi-View Contrastive Learning Framework for Integrated Sequence and Structure-Based Peptide Encoding (Ruochi Zhang et al., 2024)

{{<citation>}}

Ruochi Zhang, Haoran Wu, Chang Liu, Huaping Li, Yuqian Wu, Kewei Li, Yifan Wang, Yifan Deng, Jiahui Chen, Fengfeng Zhou, Xin Gao. (2024)  
**PepHarmony: A Multi-View Contrastive Learning Framework for Integrated Sequence and Structure-Based Peptide Encoding**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CE, cs-LG, cs.LG, q-bio-BM  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.11360v1)  

---


**ABSTRACT**  
Recent advances in protein language models have catalyzed significant progress in peptide sequence representation. Despite extensive exploration in this field, pre-trained models tailored for peptide-specific needs remain largely unaddressed due to the difficulty in capturing the complex and sometimes unstable structures of peptides. This study introduces a novel multi-view contrastive learning framework PepHarmony for the sequence-based peptide encoding task. PepHarmony innovatively combines both sequence- and structure-level information into a sequence-level encoding module through contrastive learning. We carefully select datasets from the Protein Data Bank (PDB) and AlphaFold database to encompass a broad spectrum of peptide sequences and structures. The experimental data highlights PepHarmony's exceptional capability in capturing the intricate relationship between peptide sequences and structures compared with the baseline and fine-tuned models. The robustness of our model is confirmed through extensive ablation studies, which emphasize the crucial roles of contrastive loss and strategic data sorting in enhancing predictive performance. The proposed PepHarmony framework serves as a notable contribution to peptide representations, and offers valuable insights for future applications in peptide drug discovery and peptide engineering. We have made all the source code utilized in this study publicly accessible via GitHub at https://github.com/zhangruochi/PepHarmony or http://www.healthinformaticslab.org/supp/.

{{</citation>}}


## cs.HC (1)



### (12/61) Older Adults Imagining Future Technologies in Participatory Design Workshops: Supporting Continuity in the Pursuit of Meaningful Activities (Wei Zhao et al., 2024)

{{<citation>}}

Wei Zhao, Ryan M. Kelly, Melissa J. Rogerson, Jenny Waycott. (2024)  
**Older Adults Imagining Future Technologies in Participatory Design Workshops: Supporting Continuity in the Pursuit of Meaningful Activities**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11628v1)  

---


**ABSTRACT**  
Recent innovations in digital technology offer significant opportunities for older adults to engage in meaningful activities. To investigate older adults' perceptions of using existing and emerging technologies for meaningful activities, we conducted three participatory design workshops and follow-up interviews with adults aged over 65. The workshops encompassed discussions on existing technologies for meaningful activities, demonstrations of emerging technologies such as VR, AR, and AI, and design activities including prototyping and storyboarding. Our findings show that while participants had diverse interpretations of meaningful activities, they sought to use technologies to support continuity in the pursuit of these activities. Specifically, participants highlighted the importance of safe aging at home, which provides a pathway for meaningful activities in later life. We further discuss participants' discerning attitudes when assessing the use of different technologies for meaningful activities and several values and attributes they desire when envisioning future technologies, including simplicity, positivity, proactivity, and integration.

{{</citation>}}


## cs.CL (15)



### (13/61) In-context Learning with Retrieved Demonstrations for Language Models: A Survey (an Luo et al., 2024)

{{<citation>}}

an Luo, Xin Xu, Yue Liu, Panupong Pasupat, Mehran Kazemi. (2024)  
**In-context Learning with Retrieved Demonstrations for Language Models: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.11624v1)  

---


**ABSTRACT**  
Language models, especially pre-trained large language models, have showcased remarkable abilities as few-shot in-context learners (ICL), adept at adapting to new tasks with just a few demonstrations in the input context. However, the model's ability to perform ICL is sensitive to the choice of the few-shot demonstrations. Instead of using a fixed set of demonstrations, one recent development is to retrieve demonstrations tailored to each input query. The implementation of demonstration retrieval is relatively straightforward, leveraging existing databases and retrieval systems. This not only improves the efficiency and scalability of the learning process but also has been shown to reduce biases inherent in manual example selection. In light of the encouraging results and growing research in ICL with retrieved demonstrations, we conduct an extensive review of studies in this area. In this survey, we discuss and compare different design choices for retrieval models, retrieval training procedures, and inference algorithms.

{{</citation>}}


### (14/61) Robust Evaluation Measures for Evaluating Social Biases in Masked Language Models (Yang Liu, 2024)

{{<citation>}}

Yang Liu. (2024)  
**Robust Evaluation Measures for Evaluating Social Biases in Masked Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2401.11601v1)  

---


**ABSTRACT**  
Many evaluation measures are used to evaluate social biases in masked language models (MLMs). However, we find that these previously proposed evaluation measures are lacking robustness in scenarios with limited datasets. This is because these measures are obtained by comparing the pseudo-log-likelihood (PLL) scores of the stereotypical and anti-stereotypical samples using an indicator function. The disadvantage is the limited mining of the PLL score sets without capturing its distributional information. In this paper, we represent a PLL score set as a Gaussian distribution and use Kullback Leibler (KL) divergence and Jensen Shannon (JS) divergence to construct evaluation measures for the distributions of stereotypical and anti-stereotypical PLL scores. Experimental results on the publicly available datasets StereoSet (SS) and CrowS-Pairs (CP) show that our proposed measures are significantly more robust and interpretable than those proposed previously.

{{</citation>}}


### (15/61) CheX-GPT: Harnessing Large Language Models for Enhanced Chest X-ray Report Labeling (Jawook Gu et al., 2024)

{{<citation>}}

Jawook Gu, Han-Cheol Cho, Jiho Kim, Kihyun You, Eun Kyoung Hong, Byungseok Roh. (2024)  
**CheX-GPT: Harnessing Large Language Models for Enhanced Chest X-ray Report Labeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: BERT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.11505v1)  

---


**ABSTRACT**  
Free-text radiology reports present a rich data source for various medical tasks, but effectively labeling these texts remains challenging. Traditional rule-based labeling methods fall short of capturing the nuances of diverse free-text patterns. Moreover, models using expert-annotated data are limited by data scarcity and pre-defined classes, impacting their performance, flexibility and scalability. To address these issues, our study offers three main contributions: 1) We demonstrate the potential of GPT as an adept labeler using carefully designed prompts. 2) Utilizing only the data labeled by GPT, we trained a BERT-based labeler, CheX-GPT, which operates faster and more efficiently than its GPT counterpart. 3) To benchmark labeler performance, we introduced a publicly available expert-annotated test set, MIMIC-500, comprising 500 cases from the MIMIC validation set. Our findings demonstrate that CheX-GPT not only excels in labeling accuracy over existing models, but also showcases superior efficiency, flexibility, and scalability, supported by our introduction of the MIMIC-500 dataset for robust benchmarking. Code and models are available at https://github.com/kakaobrain/CheXGPT.

{{</citation>}}


### (16/61) With Greater Text Comes Greater Necessity: Inference-Time Training Helps Long Text Generation (Y. Wang et al., 2024)

{{<citation>}}

Y. Wang, D. Ma, D. Cai. (2024)  
**With Greater Text Comes Greater Necessity: Inference-Time Training Helps Long Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLEU, Text Generation  
[Paper Link](http://arxiv.org/abs/2401.11504v1)  

---


**ABSTRACT**  
Long text generation, such as novel writing or discourse-level translation with extremely long contexts, presents significant challenges to current language models. Existing methods mainly focus on extending the model's context window through strategies like length extrapolation. However, these approaches demand substantial hardware resources during the training and/or inference phases. Our proposed method, Temp-Lora, introduces an alternative concept. Instead of relying on the KV cache to store all context information, Temp-Lora embeds this information directly into the model's parameters. In the process of long text generation, we use a temporary Lora module, progressively trained with text generated previously. This approach not only efficiently preserves contextual knowledge but also prevents any permanent alteration to the model's parameters given that the module is discarded post-generation. Extensive experiments on the PG19 language modeling benchmark and the GuoFeng discourse-level translation benchmark validate the effectiveness of Temp-Lora. Our results show that: 1) Temp-Lora substantially enhances generation quality for long texts, as indicated by a 13.2% decrease in perplexity on a subset of PG19, and a 29.6% decrease in perplexity along with a 53.2% increase in BLEU score on GuoFeng, 2) Temp-Lora is compatible with and enhances most existing long text generation methods, and 3) Temp-Lora can greatly reduce computational costs by shortening the context window. While ensuring a slight improvement in generation quality (a decrease of 3.8% in PPL), it enables a reduction of 70.5% in the FLOPs required for inference and a 51.5% decrease in latency.

{{</citation>}}


### (17/61) Towards Better Inclusivity: A Diverse Tweet Corpus of English Varieties (Nhi Pham et al., 2024)

{{<citation>}}

Nhi Pham, Lachlan Pham, Adam L. Meyers. (2024)  
**Towards Better Inclusivity: A Diverse Tweet Corpus of English Varieties**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: NLP, Twitter  
[Paper Link](http://arxiv.org/abs/2401.11487v1)  

---


**ABSTRACT**  
The prevalence of social media presents a growing opportunity to collect and analyse examples of English varieties. Whilst usage of these varieties was - and, in many cases, still is - used only in spoken contexts or hard-to-access private messages, social media sites like Twitter provide a platform for users to communicate informally in a scrapeable format. Notably, Indian English (Hinglish), Singaporean English (Singlish), and African-American English (AAE) can be commonly found online. These varieties pose a challenge to existing natural language processing (NLP) tools as they often differ orthographically and syntactically from standard English for which the majority of these tools are built. NLP models trained on standard English texts produced biased outcomes for users of underrepresented varieties. Some research has aimed to overcome the inherent biases caused by unrepresentative data through techniques like data augmentation or adjusting training models.   We aim to address the issue of bias at its root - the data itself. We curate a dataset of tweets from countries with high proportions of underserved English variety speakers, and propose an annotation framework of six categorical classifications along a pseudo-spectrum that measures the degree of standard English and that thereby indirectly aims to surface the manifestations of English varieties in these tweets. Following best annotation practices, our growing corpus features 170,800 tweets taken from 7 countries, labeled by annotators who are from those countries and can communicate in regionally-dominant varieties of English. Our corpus highlights the accuracy discrepancies in pre-trained language identifiers between western English and non-western (i.e., less standard) English varieties. We hope to contribute to the growing literature identifying and reducing the implicit demographic discrepancies in NLP.

{{</citation>}}


### (18/61) Over-Reasoning and Redundant Calculation of Large Language Models (Cheng-Han Chiang et al., 2024)

{{<citation>}}

Cheng-Han Chiang, Hung-yi Lee. (2024)  
**Over-Reasoning and Redundant Calculation of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.11467v1)  

---


**ABSTRACT**  
Large language models (LLMs) can solve problems step-by-step. While this chain-of-thought (CoT) reasoning boosts LLMs' performance, it is unclear if LLMs \textit{know} when to use CoT and whether those CoT are always necessary to answer the question. This paper shows that LLMs tend to generate redundant calculations and reasoning on a manually constructed math QA dataset, GSM8K-Zero. GSM8K-Zero is constructed such that the questions can be answered without any calculations, but LLMs, including Llama-2 models and Claude-2, tend to generate lengthy and unnecessary calculations to answer the questions. We also conduct experiments to explain why LLMs generate redundant calculations and reasonings. GSM8K-Zero is publicly available at https://github.com/d223302/Over-Reasoning-of-LLMs and https://huggingface.co/datasets/dcml0714/GSM8K-Zero.

{{</citation>}}


### (19/61) Linear Alignment: A Closed-form Solution for Aligning Human Preferences without Tuning and Feedback (Songyang Gao et al., 2024)

{{<citation>}}

Songyang Gao, Qiming Ge, Wei Shen, Shihan Dou, Junjie Ye, Xiao Wang, Rui Zheng, Yicheng Zou, Zhi Chen, Hang Yan, Qi Zhang, Dahua Lin. (2024)  
**Linear Alignment: A Closed-form Solution for Aligning Human Preferences without Tuning and Feedback**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11458v1)  

---


**ABSTRACT**  
The success of AI assistants based on Language Models (LLMs) hinges on Reinforcement Learning from Human Feedback (RLHF) to comprehend and align with user intentions. However, traditional alignment algorithms, such as PPO, are hampered by complex annotation and training requirements. This reliance limits the applicability of RLHF and hinders the development of professional assistants tailored to diverse human preferences. In this work, we introduce \textit{Linear Alignment}, a novel algorithm that aligns language models with human preferences in one single inference step, eliminating the reliance on data annotation and model training. Linear alignment incorporates a new parameterization for policy optimization under divergence constraints, which enables the extraction of optimal policy in a closed-form manner and facilitates the direct estimation of the aligned response. Extensive experiments on both general and personalized preference datasets demonstrate that linear alignment significantly enhances the performance and efficiency of LLM alignment across diverse scenarios. Our code and dataset will be published on \url{https://github.com/Wizardcoast/Linear_Alignment.git}.

{{</citation>}}


### (20/61) Majority or Minority: Data Imbalance Learning Method for Named Entity Recognition (Sota Nemoto et al., 2024)

{{<citation>}}

Sota Nemoto, Shunsuke Kitada, Hitoshi Iyatomi. (2024)  
**Majority or Minority: Data Imbalance Learning Method for Named Entity Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, NLP, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2401.11431v1)  

---


**ABSTRACT**  
Data imbalance presents a significant challenge in various machine learning (ML) tasks, particularly named entity recognition (NER) within natural language processing (NLP). NER exhibits a data imbalance with a long-tail distribution, featuring numerous minority classes (i.e., entity classes) and a single majority class (i.e., O-class). The imbalance leads to the misclassifications of the entity classes as the O-class. To tackle the imbalance, we propose a simple and effective learning method, named majority or minority (MoM) learning. MoM learning incorporates the loss computed only for samples whose ground truth is the majority class (i.e., the O-class) into the loss of the conventional ML model. Evaluation experiments on four NER datasets (Japanese and English) showed that MoM learning improves prediction performance of the minority classes, without sacrificing the performance of the majority class and is more effective than widely known and state-of-the-art methods. We also evaluated MoM learning using frameworks as sequential labeling and machine reading comprehension, which are commonly used in NER. Furthermore, MoM learning has achieved consistent performance improvements regardless of language, model, or framework.

{{</citation>}}


### (21/61) SEBERTNets: Sequence Enhanced BERT Networks for Event Entity Extraction Tasks Oriented to the Finance Field (Congqing He et al., 2024)

{{<citation>}}

Congqing He, Xiangyu Zhu, Yuquan Le, Yuzhong Liu, Jianhong Yin. (2024)  
**SEBERTNets: Sequence Enhanced BERT Networks for Event Entity Extraction Tasks Oriented to the Finance Field**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.11408v1)  

---


**ABSTRACT**  
Event extraction lies at the cores of investment analysis and asset management in the financial field, and thus has received much attention. The 2019 China conference on knowledge graph and semantic computing (CCKS) challenge sets up a evaluation competition for event entity extraction task oriented to the finance field. In this task, we mainly focus on how to extract the event entity accurately, and recall all the corresponding event entity effectively. In this paper, we propose a novel model, Sequence Enhanced BERT Networks (SEBERTNets for short), which can inherit the advantages of the BERT,and while capturing sequence semantic information. In addition, motivated by recommendation system, we propose Hybrid Sequence Enhanced BERT Networks (HSEBERTNets for short), which uses a multi-channel recall method to recall all the corresponding event entity. The experimental results show that, the F1 score of SEBERTNets is 0.905 in the first stage, and the F1 score of HSEBERTNets is 0.934 in the first stage, which demonstarate the effectiveness of our methods.

{{</citation>}}


### (22/61) MedLM: Exploring Language Models for Medical Question Answering Systems (Niraj Yagnik et al., 2024)

{{<citation>}}

Niraj Yagnik, Jay Jhaveri, Vivek Sharma, Gabriel Pila, Asma Ben, Jingbo Shang. (2024)  
**MedLM: Exploring Language Models for Medical Question Answering Systems**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.11389v1)  

---


**ABSTRACT**  
In the face of rapidly expanding online medical literature, automated systems for aggregating and summarizing information are becoming increasingly crucial for healthcare professionals and patients. Large Language Models (LLMs), with their advanced generative capabilities, have shown promise in various NLP tasks, and their potential in the healthcare domain, particularly for Closed-Book Generative QnA, is significant. However, the performance of these models in domain-specific tasks such as medical Q&A remains largely unexplored. This study aims to fill this gap by comparing the performance of general and medical-specific distilled LMs for medical Q&A. We aim to evaluate the effectiveness of fine-tuning domain-specific LMs and compare the performance of different families of Language Models. The study will address critical questions about these models' reliability, comparative performance, and effectiveness in the context of medical Q&A. The findings will provide valuable insights into the suitability of different LMs for specific applications in the medical domain.

{{</citation>}}


### (23/61) Using Large Language Model for End-to-End Chinese ASR and NER (Yuang Li et al., 2024)

{{<citation>}}

Yuang Li, Jiawei Yu, Yanqing Zhao, Min Zhang, Mengxin Ren, Xiaofeng Zhao, Xiaosong Qiao, Chang Su, Miaomiao Ma, Hao Yang. (2024)  
**Using Large Language Model for End-to-End Chinese ASR and NER**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GLM, Language Model, NER  
[Paper Link](http://arxiv.org/abs/2401.11382v1)  

---


**ABSTRACT**  
Mapping speech tokens to the same feature space as text tokens has become the paradigm for the integration of speech modality into decoder-only large language models (LLMs). An alternative approach is to use an encoder-decoder architecture that incorporates speech features through cross-attention. This approach, however, has received less attention in the literature. In this work, we connect the Whisper encoder with ChatGLM3 and provide in-depth comparisons of these two approaches using Chinese automatic speech recognition (ASR) and name entity recognition (NER) tasks. We evaluate them not only by conventional metrics like the F1 score but also by a novel fine-grained taxonomy of ASR-NER errors. Our experiments reveal that encoder-decoder architecture outperforms decoder-only architecture with a short context, while decoder-only architecture benefits from a long context as it fully exploits all layers of the LLM. By using LLM, we significantly reduced the entity omission errors and improved the entity ASR accuracy compared to the Conformer baseline. Additionally, we obtained a state-of-the-art (SOTA) F1 score of 0.805 on the AISHELL-NER test set by using chain-of-thought (CoT) NER which first infers long-form ASR transcriptions and then predicts NER labels.

{{</citation>}}


### (24/61) Language Models as Hierarchy Encoders (Yuan He et al., 2024)

{{<citation>}}

Yuan He, Zhangdie Yuan, Jiaoyan Chen, Ian Horrocks. (2024)  
**Language Models as Hierarchy Encoders**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11374v1)  

---


**ABSTRACT**  
Interpreting hierarchical structures latent in language is a key limitation of current language models (LMs). While previous research has implicitly leveraged these hierarchies to enhance LMs, approaches for their explicit encoding are yet to be explored. To address this, we introduce a novel approach to re-train transformer encoder-based LMs as Hierarchy Transformer encoders (HiTs), harnessing the expansive nature of hyperbolic space. Our method situates the output embedding space of pre-trained LMs within a Poincar\'e ball with a curvature that adapts to the embedding dimension, followed by re-training on hyperbolic cluster and centripetal losses. These losses are designed to effectively cluster related entities (input as texts) and organise them hierarchically. We evaluate HiTs against pre-trained and fine-tuned LMs, focusing on their capabilities in simulating transitive inference, predicting subsumptions, and transferring knowledge across hierarchies. The results demonstrate that HiTs consistently outperform both pre-trained and fine-tuned LMs in these tasks, underscoring the effectiveness and transferability of our re-trained hierarchy encoders.

{{</citation>}}


### (25/61) Finding a Needle in the Adversarial Haystack: A Targeted Paraphrasing Approach For Uncovering Edge Cases with Minimal Distribution Distortion (Aly M. Kassem et al., 2024)

{{<citation>}}

Aly M. Kassem, Sherif Saad. (2024)  
**Finding a Needle in the Adversarial Haystack: A Targeted Paraphrasing Approach For Uncovering Edge Cases with Minimal Distribution Distortion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, T5  
[Paper Link](http://arxiv.org/abs/2401.11373v1)  

---


**ABSTRACT**  
Adversarial attacks against NLP Deep Learning models are a significant concern. In particular, adversarial samples exploit the model's sensitivity to small input changes. While these changes appear insignificant on the semantics of the input sample, they result in significant decay in model performance. In this paper, we propose Targeted Paraphrasing via RL (TPRL), an approach to automatically learn a policy to generate challenging samples that most likely improve the model's performance. TPRL leverages FLAN T5, a language model, as a generator and employs a self learned policy using a proximal policy gradient to generate the adversarial examples automatically. TPRL's reward is based on the confusion induced in the classifier, preserving the original text meaning through a Mutual Implication score. We demonstrate and evaluate TPRL's effectiveness in discovering natural adversarial attacks and improving model performance through extensive experiments on four diverse NLP classification tasks via Automatic and Human evaluation. TPRL outperforms strong baselines, exhibits generalizability across classifiers and datasets, and combines the strengths of language modeling and reinforcement learning to generate diverse and influential adversarial examples.

{{</citation>}}


### (26/61) Confidence Preservation Property in Knowledge Distillation Abstractions (Dmitry Vengertsev et al., 2024)

{{<citation>}}

Dmitry Vengertsev, Elena Sherman. (2024)  
**Confidence Preservation Property in Knowledge Distillation Abstractions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2401.11365v1)  

---


**ABSTRACT**  
Social media platforms prevent malicious activities by detecting harmful content of posts and comments. To that end, they employ large-scale deep neural network language models for sentiment analysis and content understanding. Some models, like BERT, are complex, and have numerous parameters, which makes them expensive to operate and maintain. To overcome these deficiencies, industry experts employ a knowledge distillation compression technique, where a distilled model is trained to reproduce the classification behavior of the original model. The distillation processes terminates when the distillation loss function reaches the stopping criteria. This function is mainly designed to ensure that the original and the distilled models exhibit alike classification behaviors. However, besides classification accuracy, there are additional properties of the original model that the distilled model should preserve to be considered as an appropriate abstraction. In this work, we explore whether distilled TinyBERT models preserve confidence values of the original BERT models, and investigate how this confidence preservation property could guide tuning hyperparameters of the distillation process.

{{</citation>}}


### (27/61) ProLex: A Benchmark for Language Proficiency-oriented Lexical Substitution (Xuanming Zhang et al., 2024)

{{<citation>}}

Xuanming Zhang, Zixun Chen, Zhou Yu. (2024)  
**ProLex: A Benchmark for Language Proficiency-oriented Lexical Substitution**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.11356v1)  

---


**ABSTRACT**  
Lexical Substitution discovers appropriate substitutes for a given target word in a context sentence. However, the task fails to consider substitutes that are of equal or higher proficiency than the target, an aspect that could be beneficial for language learners looking to improve their writing. To bridge this gap, we propose a new task, language proficiency-oriented lexical substitution. We also introduce ProLex, a novel benchmark designed to assess systems' ability to generate not only appropriate substitutes but also substitutes that demonstrate better language proficiency. Besides the benchmark, we propose models that can automatically perform the new task. We show that our best model, a Llama2-13B model fine-tuned with task-specific synthetic data, outperforms ChatGPT by an average of 3.2% in F-score and achieves comparable results with GPT-4 on ProLex.

{{</citation>}}


## cs.IT (1)



### (28/61) The Markov-Chain Polytope with Applications (Mordecai J. Golin et al., 2024)

{{<citation>}}

Mordecai J. Golin, Albert John Lalim Patupat. (2024)  
**The Markov-Chain Polytope with Applications**  

---
Primary Category: cs.IT  
Categories: F-2-2; E-4, cs-IT, cs.IT, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11622v1)  

---


**ABSTRACT**  
This paper addresses the problem of finding a minimum-cost $m$-state Markov chain $(S_0,\ldots,S_{m-1})$ in a large set of chains. The chains studied have a reward associated with each state. The cost of a chain is its "gain", i.e., its average reward under its stationary distribution.   Specifically, for each $k=0,\ldots,m-1$ there is a known set ${\mathbb S}_k$ of type-$k$ states. A permissible Markov chain contains exactly one state of each type; the problem is to find a minimum-cost permissible chain.   The original motivation was to find a cheapest binary AIFV-$m$ lossless code on a source alphabet of size $n$. Such a code is an $m$-tuple of trees, in which each tree can be viewed as a Markov Chain state. This formulation was then used to address other problems in lossless compression. The known solution techniques for finding minimum-cost Markov chains were iterative and ran in exponential time.   This paper shows how to map every possible type-$k$ state into a type-$k$ hyperplane and then define a "Markov Chain Polytope" as the lower envelope of all such hyperplanes. Finding a minimum-cost Markov chain can then be shown to be equivalent to finding a "highest" point on this polytope.   The local optimization procedures used in the previous iterative algorithms are shown to be separation oracles for this polytope. Since these were often polynomial time, an application of the Ellipsoid method immediately leads to polynomial time algorithms for these problems.

{{</citation>}}


## cs.CV (13)



### (29/61) A Survey on African Computer Vision Datasets, Topics and Researchers (Abdul-Hakeem Omotayo et al., 2024)

{{<citation>}}

Abdul-Hakeem Omotayo, Ashery Mbilinyi, Lukman Ismaila, Houcemeddine Turki, Mahmoud Abdien, Karim Gamal, Idriss Tondji, Yvan Pimi, Naome A. Etori, Marwa M. Matar, Clifford Broni-Bediako, Abigail Oppong, Mai Gamal, Eman Ehab, Gbetondji Dovonon, Zainab Akinjobi, Daniel Ajisafe, Oluwabukola G. Adegboro, Mennatullah Siam. (2024)  
**A Survey on African Computer Vision Datasets, Topics and Researchers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2401.11617v1)  

---


**ABSTRACT**  
Computer vision encompasses a range of tasks such as object detection, semantic segmentation, and 3D reconstruction. Despite its relevance to African communities, research in this field within Africa represents only 0.06% of top-tier publications over the past decade. This study undertakes a thorough analysis of 63,000 Scopus-indexed computer vision publications from Africa, spanning from 2012 to 2022. The aim is to provide a survey of African computer vision topics, datasets and researchers. A key aspect of our study is the identification and categorization of African Computer Vision datasets using large language models that automatically parse abstracts of these publications. We also provide a compilation of unofficial African Computer Vision datasets distributed through challenges or data hosting platforms, and provide a full taxonomy of dataset categories. Our survey also pinpoints computer vision topics trends specific to different African regions, indicating their unique focus areas. Additionally, we carried out an extensive survey to capture the views of African researchers on the current state of computer vision research in the continent and the structural barriers they believe need urgent attention. In conclusion, this study catalogs and categorizes Computer Vision datasets and topics contributed or initiated by African institutions and identifies barriers to publishing in top-tier Computer Vision venues. This survey underscores the importance of encouraging African researchers and institutions in advancing computer vision research in the continent. It also stresses on the need for research topics to be more aligned with the needs of African communities.

{{</citation>}}


### (30/61) Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers (Katherine Crowson et al., 2024)

{{<citation>}}

Katherine Crowson, Stefan Andreas Baumann, Alex Birch, Tanishq Mathew Abraham, Daniel Z. Kaplan, Enrico Shippole. (2024)  
**Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.11605v1)  

---


**ABSTRACT**  
We present the Hourglass Diffusion Transformer (HDiT), an image generative model that exhibits linear scaling with pixel count, supporting training at high-resolution (e.g. $1024 \times 1024$) directly in pixel-space. Building on the Transformer architecture, which is known to scale to billions of parameters, it bridges the gap between the efficiency of convolutional U-Nets and the scalability of Transformers. HDiT trains successfully without typical high-resolution training techniques such as multiscale architectures, latent autoencoders or self-conditioning. We demonstrate that HDiT performs competitively with existing models on ImageNet $256^2$, and sets a new state-of-the-art for diffusion models on FFHQ-$1024^2$.

{{</citation>}}


### (31/61) Hierarchical Prompts for Rehearsal-free Continual Learning (Yukun Zuo et al., 2024)

{{<citation>}}

Yukun Zuo, Hantao Yao, Lu Yu, Liansheng Zhuang, Changsheng Xu. (2024)  
**Hierarchical Prompts for Rehearsal-free Continual Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.11544v1)  

---


**ABSTRACT**  
Continual learning endeavors to equip the model with the capability to integrate current task knowledge while mitigating the forgetting of past task knowledge. Inspired by prompt tuning, prompt-based methods maintain a frozen backbone and train with slight learnable prompts to minimize the catastrophic forgetting that arises due to updating a large number of backbone parameters. Nonetheless, these learnable prompts tend to concentrate on the discriminatory knowledge of the current task while ignoring past task knowledge, leading to that learnable prompts still suffering from catastrophic forgetting. This paper introduces a novel rehearsal-free paradigm for continual learning termed Hierarchical Prompts (H-Prompts), comprising three categories of prompts -- class prompt, task prompt, and general prompt. To effectively depict the knowledge of past classes, class prompt leverages Bayesian Distribution Alignment to model the distribution of classes in each task. To reduce the forgetting of past task knowledge, task prompt employs Cross-task Knowledge Excavation to amalgamate the knowledge encapsulated in the learned class prompts of past tasks and current task knowledge. Furthermore, general prompt utilizes Generalized Knowledge Exploration to deduce highly generalized knowledge in a self-supervised manner. Evaluations on two benchmarks substantiate the efficacy of the proposed H-Prompts, exemplified by an average accuracy of 87.8% in Split CIFAR-100 and 70.6% in Split ImageNet-R.

{{</citation>}}


### (32/61) Deformable Endoscopic Tissues Reconstruction with Gaussian Splatting (Lingting Zhu et al., 2024)

{{<citation>}}

Lingting Zhu, Zhao Wang, Zhenchao Jin, Guying Lin, Lequan Yu. (2024)  
**Deformable Endoscopic Tissues Reconstruction with Gaussian Splatting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11535v1)  

---


**ABSTRACT**  
Surgical 3D reconstruction is a critical area of research in robotic surgery, with recent works adopting variants of dynamic radiance fields to achieve success in 3D reconstruction of deformable tissues from single-viewpoint videos. However, these methods often suffer from time-consuming optimization or inferior quality, limiting their adoption in downstream tasks. Inspired by 3D Gaussian Splatting, a recent trending 3D representation, we present EndoGS, applying Gaussian Splatting for deformable endoscopic tissue reconstruction. Specifically, our approach incorporates deformation fields to handle dynamic scenes, depth-guided supervision to optimize 3D targets with a single viewpoint, and a spatial-temporal weight mask to mitigate tool occlusion. As a result, EndoGS reconstructs and renders high-quality deformable endoscopic tissues from a single-viewpoint video, estimated depth maps, and labeled tool masks. Experiments on DaVinci robotic surgery videos demonstrate that EndoGS achieves superior rendering quality. Code is available at https://github.com/HKU-MedAI/EndoGS.

{{</citation>}}


### (33/61) Self-Supervised Bird's Eye View Motion Prediction with Cross-Modality Signals (Shaoheng Fang et al., 2024)

{{<citation>}}

Shaoheng Fang, Zuhong Liu, Mingyu Wang, Chenxin Xu, Yiqi Zhong, Siheng Chen. (2024)  
**Self-Supervised Bird's Eye View Motion Prediction with Cross-Modality Signals**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.11499v1)  

---


**ABSTRACT**  
Learning the dense bird's eye view (BEV) motion flow in a self-supervised manner is an emerging research for robotics and autonomous driving. Current self-supervised methods mainly rely on point correspondences between point clouds, which may introduce the problems of fake flow and inconsistency, hindering the model's ability to learn accurate and realistic motion. In this paper, we introduce a novel cross-modality self-supervised training framework that effectively addresses these issues by leveraging multi-modality data to obtain supervision signals. We design three innovative supervision signals to preserve the inherent properties of scene motion, including the masked Chamfer distance loss, the piecewise rigidity loss, and the temporal consistency loss. Through extensive experiments, we demonstrate that our proposed self-supervised framework outperforms all previous self-supervision methods for the motion prediction task.

{{</citation>}}


### (34/61) Inter-Domain Mixup for Semi-Supervised Domain Adaptation (Jichang Li et al., 2024)

{{<citation>}}

Jichang Li, Guanbin Li, Yizhou Yu. (2024)  
**Inter-Domain Mixup for Semi-Supervised Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2401.11453v1)  

---


**ABSTRACT**  
Semi-supervised domain adaptation (SSDA) aims to bridge source and target domain distributions, with a small number of target labels available, achieving better classification performance than unsupervised domain adaptation (UDA). However, existing SSDA work fails to make full use of label information from both source and target domains for feature alignment across domains, resulting in label mismatch in the label space during model testing. This paper presents a novel SSDA approach, Inter-domain Mixup with Neighborhood Expansion (IDMNE), to tackle this issue. Firstly, we introduce a cross-domain feature alignment strategy, Inter-domain Mixup, that incorporates label information into model adaptation. Specifically, we employ sample-level and manifold-level data mixing to generate compatible training samples. These newly established samples, combined with reliable and actual label information, display diversity and compatibility across domains, while such extra supervision thus facilitates cross-domain feature alignment and mitigates label mismatch. Additionally, we utilize Neighborhood Expansion to leverage high-confidence pseudo-labeled samples in the target domain, diversifying the label information of the target domain and thereby further increasing the performance of the adaptation model. Accordingly, the proposed approach outperforms existing state-of-the-art methods, achieving significant accuracy improvements on popular SSDA benchmarks, including DomainNet, Office-Home, and Office-31.

{{</citation>}}


### (35/61) Adaptive Betweenness Clustering for Semi-Supervised Domain Adaptation (Jichang Li et al., 2024)

{{<citation>}}

Jichang Li, Guanbin Li, Yizhou Yu. (2024)  
**Adaptive Betweenness Clustering for Semi-Supervised Domain Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2401.11448v1)  

---


**ABSTRACT**  
Compared to unsupervised domain adaptation, semi-supervised domain adaptation (SSDA) aims to significantly improve the classification performance and generalization capability of the model by leveraging the presence of a small amount of labeled data from the target domain. Several SSDA approaches have been developed to enable semantic-aligned feature confusion between labeled (or pseudo labeled) samples across domains; nevertheless, owing to the scarcity of semantic label information of the target domain, they were arduous to fully realize their potential. In this study, we propose a novel SSDA approach named Graph-based Adaptive Betweenness Clustering (G-ABC) for achieving categorical domain alignment, which enables cross-domain semantic alignment by mandating semantic transfer from labeled data of both the source and target domains to unlabeled target samples. In particular, a heterogeneous graph is initially constructed to reflect the pairwise relationships between labeled samples from both domains and unlabeled ones of the target domain. Then, to degrade the noisy connectivity in the graph, connectivity refinement is conducted by introducing two strategies, namely Confidence Uncertainty based Node Removal and Prediction Dissimilarity based Edge Pruning. Once the graph has been refined, Adaptive Betweenness Clustering is introduced to facilitate semantic transfer by using across-domain betweenness clustering and within-domain betweenness clustering, thereby propagating semantic label information from labeled samples across domains to unlabeled target data. Extensive experiments on three standard benchmark datasets, namely DomainNet, Office-Home, and Office-31, indicated that our method outperforms previous state-of-the-art SSDA approaches, demonstrating the superiority of the proposed G-ABC algorithm.

{{</citation>}}


### (36/61) Geometric Prior Guided Feature Representation Learning for Long-Tailed Classification (Yanbiao Ma et al., 2024)

{{<citation>}}

Yanbiao Ma, Licheng Jiao, Fang Liu, Shuyuan Yang, Xu Liu, Puhua Chen. (2024)  
**Geometric Prior Guided Feature Representation Learning for Long-Tailed Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.11436v1)  

---


**ABSTRACT**  
Real-world data are long-tailed, the lack of tail samples leads to a significant limitation in the generalization ability of the model. Although numerous approaches of class re-balancing perform well for moderate class imbalance problems, additional knowledge needs to be introduced to help the tail class recover the underlying true distribution when the observed distribution from a few tail samples does not represent its true distribution properly, thus allowing the model to learn valuable information outside the observed domain. In this work, we propose to leverage the geometric information of the feature distribution of the well-represented head class to guide the model to learn the underlying distribution of the tail class. Specifically, we first systematically define the geometry of the feature distribution and the similarity measures between the geometries, and discover four phenomena regarding the relationship between the geometries of different feature distributions. Then, based on four phenomena, feature uncertainty representation is proposed to perturb the tail features by utilizing the geometry of the head class feature distribution. It aims to make the perturbed features cover the underlying distribution of the tail class as much as possible, thus improving the model's generalization performance in the test domain. Finally, we design a three-stage training scheme enabling feature uncertainty modeling to be successfully applied. Experiments on CIFAR-10/100-LT, ImageNet-LT, and iNaturalist2018 show that our proposed approach outperforms other similar methods on most metrics. In addition, the experimental phenomena we discovered are able to provide new perspectives and theoretical foundations for subsequent studies.

{{</citation>}}


### (37/61) Exploring Diffusion Time-steps for Unsupervised Representation Learning (Zhongqi Yue et al., 2024)

{{<citation>}}

Zhongqi Yue, Jiankun Wang, Qianru Sun, Lei Ji, Eric I-Chao Chang, Hanwang Zhang. (2024)  
**Exploring Diffusion Time-steps for Unsupervised Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.11430v1)  

---


**ABSTRACT**  
Representation learning is all about discovering the hidden modular attributes that generate the data faithfully. We explore the potential of Denoising Diffusion Probabilistic Model (DM) in unsupervised learning of the modular attributes. We build a theoretical framework that connects the diffusion time-steps and the hidden attributes, which serves as an effective inductive bias for unsupervised learning. Specifically, the forward diffusion process incrementally adds Gaussian noise to samples at each time-step, which essentially collapses different samples into similar ones by losing attributes, e.g., fine-grained attributes such as texture are lost with less noise added (i.e., early time-steps), while coarse-grained ones such as shape are lost by adding more noise (i.e., late time-steps). To disentangle the modular attributes, at each time-step t, we learn a t-specific feature to compensate for the newly lost attribute, and the set of all 1,...,t-specific features, corresponding to the cumulative set of lost attributes, are trained to make up for the reconstruction error of a pre-trained DM at time-step t. On CelebA, FFHQ, and Bedroom datasets, the learned feature significantly improves attribute classification and enables faithful counterfactual generation, e.g., interpolating only one specified attribute between two images, validating the disentanglement quality. Codes are in https://github.com/yue-zhongqi/diti.

{{</citation>}}


### (38/61) Embedded Hyperspectral Band Selection with Adaptive Optimization for Image Semantic Segmentation (Yaniv Zimmer et al., 2024)

{{<citation>}}

Yaniv Zimmer, Oren Glickman. (2024)  
**Embedded Hyperspectral Band Selection with Adaptive Optimization for Image Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: I-4-6; I-4-7; I-4-2; I-4-8, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.11420v1)  

---


**ABSTRACT**  
Hyperspectral band selection plays a pivotal role in remote sensing and image analysis, aiming to identify the most informative spectral bands while minimizing computational overhead. In this paper, we introduce a pioneering approach for hyperspectral band selection that offers an embedded solution, making it well-suited for resource-constrained or real-time applications. Our proposed method, embedded Hyperspectral Band Selection (EHBS), excels in selecting the best bands without the need for prior processing, seamlessly integrating with the downstream task model. This is achieved through the adaptation of the Stochastic Gates (STG) algorithm, originally designed for feature selection, for hyperspectral band selection in the context of image semantic segmentation and the integration of a dynamic optimizer, DoG, which removes the need for the required tuning the learning rate. To assess the performance of our method, we introduce a novel metric for evaluating band selection methods across different target numbers of selected bands quantified by the Area Under the Curve (AUC). We conduct experiments on two distinct semantic-segmentation hyperspectral benchmark datasets, demonstrating its superiority in terms of its resulting accuracy and its ease of use compared to many common and state-of-the-art methods. Furthermore, our contributions extend beyond the realm of hyperspectral band selection. The adaptability of our approach to other tasks, especially those involving grouped features, opens up promising avenues for broader applications within the realm of deep learning, such as feature selection for feature groups. The demonstrated success on the tested datasets and the potential for application to a variety of tasks underscore the value of our method as a substantial addition to the field of computer vision.

{{</citation>}}


### (39/61) S$^3$M-Net: Joint Learning of Semantic Segmentation and Stereo Matching for Autonomous Driving (Zhiyuan Wu et al., 2024)

{{<citation>}}

Zhiyuan Wu, Yi Feng, Chuang-Wei Liu, Fisher Yu, Qijun Chen, Rui Fan. (2024)  
**S$^3$M-Net: Joint Learning of Semantic Segmentation and Stereo Matching for Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.11414v1)  

---


**ABSTRACT**  
Semantic segmentation and stereo matching are two essential components of 3D environmental perception systems for autonomous driving. Nevertheless, conventional approaches often address these two problems independently, employing separate models for each task. This approach poses practical limitations in real-world scenarios, particularly when computational resources are scarce or real-time performance is imperative. Hence, in this article, we introduce S$^3$M-Net, a novel joint learning framework developed to perform semantic segmentation and stereo matching simultaneously. Specifically, S$^3$M-Net shares the features extracted from RGB images between both tasks, resulting in an improved overall scene understanding capability. This feature sharing process is realized using a feature fusion adaption (FFA) module, which effectively transforms the shared features into semantic space and subsequently fuses them with the encoded disparity features. The entire joint learning framework is trained by minimizing a novel semantic consistency-guided (SCG) loss, which places emphasis on the structural consistency in both tasks. Extensive experimental results conducted on the vKITTI2 and KITTI datasets demonstrate the effectiveness of our proposed joint learning framework and its superior performance compared to other state-of-the-art single-task networks. Our project webpage is accessible at mias.group/S3M-Net.

{{</citation>}}


### (40/61) Adversarial Augmentation Training Makes Action Recognition Models More Robust to Realistic Video Distribution Shifts (Kiyoon Kim et al., 2024)

{{<citation>}}

Kiyoon Kim, Shreyank N Gowda, Panagiotis Eustratiadis, Antreas Antoniou, Robert B Fisher. (2024)  
**Adversarial Augmentation Training Makes Action Recognition Models More Robust to Realistic Video Distribution Shifts**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11406v1)  

---


**ABSTRACT**  
Despite recent advances in video action recognition achieving strong performance on existing benchmarks, these models often lack robustness when faced with natural distribution shifts between training and test data. We propose two novel evaluation methods to assess model resilience to such distribution disparity. One method uses two different datasets collected from different sources and uses one for training and validation, and the other for testing. More precisely, we created dataset splits of HMDB-51 or UCF-101 for training, and Kinetics-400 for testing, using the subset of the classes that are overlapping in both train and test datasets. The other proposed method extracts the feature mean of each class from the target evaluation dataset's training data (i.e. class prototype) and estimates test video prediction as a cosine similarity score between each sample to the class prototypes of each target class. This procedure does not alter model weights using the target dataset and it does not require aligning overlapping classes of two different datasets, thus is a very efficient method to test the model robustness to distribution shifts without prior knowledge of the target distribution. We address the robustness problem by adversarial augmentation training - generating augmented views of videos that are "hard" for the classification model by applying gradient ascent on the augmentation parameters - as well as "curriculum" scheduling the strength of the video augmentations. We experimentally demonstrate the superior performance of the proposed adversarial augmentation approach over baselines across three state-of-the-art action recognition models - TSM, Video Swin Transformer, and Uniformer. The presented work provides critical insight into model robustness to distribution shifts and presents effective techniques to enhance video action recognition performance in a real-world deployment.

{{</citation>}}


### (41/61) LLMRA: Multi-modal Large Language Model based Restoration Assistant (Xiaoyu Jin et al., 2024)

{{<citation>}}

Xiaoyu Jin, Yuan Shi, Bin Xia, Wenming Yang. (2024)  
**LLMRA: Multi-modal Large Language Model based Restoration Assistant**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11401v1)  

---


**ABSTRACT**  
Multi-modal Large Language Models (MLLMs) have a significant impact on various tasks, due to their extensive knowledge and powerful perception and generation capabilities. However, it still remains an open research problem on applying MLLMs to low-level vision tasks. In this paper, we present a simple MLLM-based Image Restoration framework to address this gap, namely Multi-modal Large Language Model based Restoration Assistant (LLMRA). We exploit the impressive capabilities of MLLMs to obtain the degradation information for universal image restoration. By employing a pretrained multi-modal large language model and a vision language model, we generate text descriptions and encode them as context embedding with degradation information for the degraded image. Through the proposed Context Enhance Module (CEM) and Degradation Context based Transformer Network (DC-former), we integrate these context embedding into the restoration network, contributing to more accurate and adjustable image restoration. Based on the dialogue with the users, our method leverages image degradation priors from MLLMs, providing low-level attributes descriptions of the input low-quality images and the restored high-quality images simultaneously. Extensive experiments demonstrate the superior performance of our LLMRA in universal image restoration tasks.

{{</citation>}}


## quant-ph (3)



### (42/61) Quantum Architecture Search with Unsupervised Representation Learning (Yize Sun et al., 2024)

{{<citation>}}

Yize Sun, Zixin Wu, Yunpu Ma, Volker Tresp. (2024)  
**Quantum Architecture Search with Unsupervised Representation Learning**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: QA, Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.11576v1)  

---


**ABSTRACT**  
Utilizing unsupervised representation learning for quantum architecture search (QAS) represents a cutting-edge approach poised to realize potential quantum advantage on Noisy Intermediate-Scale Quantum (NISQ) devices. Most QAS algorithms combine their search space and search algorithms together and thus generally require evaluating a large number of quantum circuits during the search process. Predictor-based QAS algorithms can alleviate this problem by directly estimating the performance of circuits according to their structures. However, a high-performance predictor generally requires very time-consuming labeling to obtain a large number of labeled quantum circuits. Recently, a classical neural architecture search algorithm Arch2vec inspires us by showing that architecture search can benefit from decoupling unsupervised representation learning from the search process. Whether unsupervised representation learning can help QAS without any predictor is still an open topic. In this work, we propose a framework QAS with unsupervised representation learning and visualize how unsupervised architecture representation learning encourages quantum circuit architectures with similar connections and operators to cluster together. Specifically, our framework enables the process of QAS to be decoupled from unsupervised architecture representation learning so that the learned representation can be directly applied to different downstream applications. Furthermore, our framework is predictor-free eliminating the need for a large number of labeled quantum circuits. During the search process, we use two algorithms REINFORCE and Bayesian Optimization to directly search on the latent representation, and compare them with the method Random Search. The results show our framework can more efficiently get well-performing candidate circuits within a limited number of searches.

{{</citation>}}


### (43/61) VQC-Based Reinforcement Learning with Data Re-uploading: Performance and Trainability (Rodrigo Coelho et al., 2024)

{{<citation>}}

Rodrigo Coelho, André Sequeira, Luís Paulo Santos. (2024)  
**VQC-Based Reinforcement Learning with Data Re-uploading: Performance and Trainability**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11555v1)  

---


**ABSTRACT**  
Reinforcement Learning (RL) consists of designing agents that make intelligent decisions without human supervision. When used alongside function approximators such as Neural Networks (NNs), RL is capable of solving extremely complex problems. Deep Q-Learning, a RL algorithm that uses Deep NNs, achieved super-human performance in some specific tasks. Nonetheless, it is also possible to use Variational Quantum Circuits (VQCs) as function approximators in RL algorithms. This work empirically studies the performance and trainability of such VQC-based Deep Q-Learning models in classic control benchmark environments. More specifically, we research how data re-uploading affects both these metrics. We show that the magnitude and the variance of the gradients of these models remain substantial throughout training due to the moving targets of Deep Q-Learning. Moreover, we empirically show that increasing the number of qubits does not lead to an exponential vanishing behavior of the magnitude and variance of the gradients for a PQC approximating a 2-design, unlike what was expected due to the Barren Plateau Phenomenon. This hints at the possibility of VQCs being specially adequate for being used as function approximators in such a context.

{{</citation>}}


### (44/61) Quantum Circuit Simulation with Fast Tensor Decision Diagram (Qirui Zhang et al., 2024)

{{<citation>}}

Qirui Zhang, Mehdi Saligane, Hun-Seok Kim, David Blaauw, Georgios Tzimpragos, Dennis Sylvester. (2024)  
**Quantum Circuit Simulation with Fast Tensor Decision Diagram**  

---
Primary Category: quant-ph  
Categories: cs-DS, cs-ET, quant-ph, quant-ph  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.11362v1)  

---


**ABSTRACT**  
Quantum circuit simulation is a challenging computational problem crucial for quantum computing research and development. The predominant approaches in this area center on tensor networks, prized for their better concurrency and less computation than methods using full quantum vectors and matrices. However, even with the advantages, array-based tensors can have significant redundancy. We present a novel open-source framework that harnesses tensor decision diagrams to eliminate overheads and achieve significant speedups over prior approaches. On average, it delivers a speedup of 37$\times$ over Google's TensorNetwork library on redundancy-rich circuits, and 25$\times$ and 144$\times$ over quantum multi-valued decision diagram and prior tensor decision diagram implementation, respectively, on Google random quantum circuits. To achieve this, we introduce a new linear-complexity rank simplification algorithm, Tetris, and edge-centric data structures for recursive tensor decision diagram operations. Additionally, we explore the efficacy of tensor network contraction ordering and optimizations from binary decision diagrams.

{{</citation>}}


## cs.CR (1)



### (45/61) Understanding the Security Risks of Decentralized Exchanges by Uncovering Unfair Trades in the Wild (Jiaqi Chen et al., 2024)

{{<citation>}}

Jiaqi Chen, Yibo Wang, Yuxuan Zhou, Wanning Ding, Yuzhe Tang, XiaoFeng Wang, Kai Li. (2024)  
**Understanding the Security Risks of Decentralized Exchanges by Uncovering Unfair Trades in the Wild**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.11547v1)  

---


**ABSTRACT**  
DEX, or decentralized exchange, is a prominent class of decentralized finance (DeFi) applications on blockchains, attracting a total locked value worth tens of billions of USD today.   This paper presents the first large-scale empirical study that uncovers unfair trades on popular DEX services on Ethereum and Binance Smart Chain (BSC). By joining and analyzing 60 million transactions, we find 671,400 unfair trades on all six measured DEXes, including Uniswap, Balancer, and Curve. Out of these unfair trades, we attribute 55,000 instances, with high confidence, to token thefts that cause a value loss of more than 3.88 million USD. Furthermore, the measurement study uncovers previously unknown causes of extractable value and real-world adaptive strategies to these causes. Finally, we propose countermeasures to redesign secure DEX protocols and to harden deployed services against the discovered security risks.

{{</citation>}}


## cs.MA (1)



### (46/61) Controlling the Misinformation Diffusion in Social Media by the Effect of Different Classes of Agents (Ali Khodabandeh Yalabadi et al., 2024)

{{<citation>}}

Ali Khodabandeh Yalabadi, Mehdi Yazdani-Jahromi, Sina Abdidizaji, Ivan Garibay, Ozlem Ozmen Garibay. (2024)  
**Controlling the Misinformation Diffusion in Social Media by the Effect of Different Classes of Agents**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs-SI, cs.MA  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2401.11524v1)  

---


**ABSTRACT**  
The rapid and widespread dissemination of misinformation through social networks is a growing concern in today's digital age. This study focused on modeling fake news diffusion, discovering the spreading dynamics, and designing control strategies. A common approach for modeling the misinformation dynamics is SIR-based models. Our approach is an extension of a model called 'SBFC' which is a SIR-based model. This model has three states, Susceptible, Believer, and Fact-Checker. The dynamics and transition between states are based on neighbors' beliefs, hoax credibility, spreading rate, probability of verifying the news, and probability of forgetting the current state. Our contribution is to push this model to real social networks by considering different classes of agents with their characteristics. We proposed two main strategies for confronting misinformation diffusion. First, we can educate a minor class, like scholars or influencers, to improve their ability to verify the news or remember their state longer. The second strategy is adding fact-checker bots to the network to spread the facts and influence their neighbors' states. Our result shows that both of these approaches can effectively control the misinformation spread.

{{</citation>}}


## cs.IR (4)



### (47/61) Simple Domain Adaptation for Sparse Retrievers (Mathias Vast et al., 2024)

{{<citation>}}

Mathias Vast, Yuxuan Zong, Basile Van Cooten, Benjamin Piwowarski, Laure Soulier. (2024)  
**Simple Domain Adaptation for Sparse Retrievers**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.11509v1)  

---


**ABSTRACT**  
In Information Retrieval, and more generally in Natural Language Processing, adapting models to specific domains is conducted through fine-tuning. Despite the successes achieved by this method and its versatility, the need for human-curated and labeled data makes it impractical to transfer to new tasks, domains, and/or languages when training data doesn't exist. Using the model without training (zero-shot) is another option that however suffers an effectiveness cost, especially in the case of first-stage retrievers. Numerous research directions have emerged to tackle these issues, most of them in the context of adapting to a task or a language. However, the literature is scarcer for domain (or topic) adaptation. In this paper, we address this issue of cross-topic discrepancy for a sparse first-stage retriever by transposing a method initially designed for language adaptation. By leveraging pre-training on the target data to learn domain-specific knowledge, this technique alleviates the need for annotated data and expands the scope of domain adaptation. Despite their relatively good generalization ability, we show that even sparse retrievers can benefit from our simple domain adaptation method.

{{</citation>}}


### (48/61) Enhancing Recommendation Diversity by Re-ranking with Large Language Models (Diego Carraro et al., 2024)

{{<citation>}}

Diego Carraro, Derek Bridge. (2024)  
**Enhancing Recommendation Diversity by Re-ranking with Large Language Models**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.11506v1)  

---


**ABSTRACT**  
It has long been recognized that it is not enough for a Recommender System (RS) to provide recommendations based only on their relevance to users. Among many other criteria, the set of recommendations may need to be diverse in order to handle uncertainty and offer a meaningful choice. The literature reports many ways of measuring diversity and ways of improving the diversity of a set of recommendations, most notably by re-ranking and selecting from a larger set of candidate recommendations. Driven by promising insights from the literature on how to incorporate versatile Large Language Models (LLMs) into the RS pipeline, in this paper, we show how LLMs can be used for diversity re-ranking.   We begin with an informal study that verifies that LLMs can be used for re-ranking tasks and do have some understanding of the concept of diversity. Then, we design a more rigorous methodology where LLMs are prompted to generate a diverse ranking from a candidate ranking using various prompt templates with different re-ranking instructions in a zero-shot fashion. We conduct comprehensive experiments testing state-of-the-art conversational LLMs from the GPT and Llama families. We compare their re-ranking capabilities with random re-ranking and various traditional re-ranking methods from the literature (MMR, xQuAD and RxQuAD). We find that LLM-based re-ranking outperforms random re-ranking across all the metrics that we use but does not perform as well as the traditional re-ranking methods. We gain insight into prompt design for this task (e.g.\ on the whole, it is better to prompt for diversity rather than a balance of diversity and relevance). Given that no special knowledge engineering is needed, we conclude that LLM-based re-ranking is a promising approach, and we highlight directions for future research. We open-source the code of our experiments for reproducibility.

{{</citation>}}


### (49/61) D2K: Turning Historical Data into Retrievable Knowledge for Recommender Systems (Jiarui Qin et al., 2024)

{{<citation>}}

Jiarui Qin, Weiwen Liu, Ruiming Tang, Weinan Zhang, Yong Yu. (2024)  
**D2K: Turning Historical Data into Retrievable Knowledge for Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.11478v1)  

---


**ABSTRACT**  
A vast amount of user behavior data is constantly accumulating on today's large recommendation platforms, recording users' various interests and tastes. Preserving knowledge from the old data while new data continually arrives is a vital problem for recommender systems. Existing approaches generally seek to save the knowledge implicitly in the model parameters. However, such a parameter-centric approach lacks scalability and flexibility -- the capacity is hard to scale, and the knowledge is inflexible to utilize. Hence, in this work, we propose a framework that turns massive user behavior data to retrievable knowledge (D2K). It is a data-centric approach that is model-agnostic and easy to scale up. Different from only storing unary knowledge such as the user-side or item-side information, D2K propose to store ternary knowledge for recommendation, which is determined by the complete recommendation factors -- user, item, and context. The knowledge retrieved by target samples can be directly used to enhance the performance of any recommendation algorithms. Specifically, we introduce a Transformer-based knowledge encoder to transform the old data into knowledge with the user-item-context cross features. A personalized knowledge adaptation unit is devised to effectively exploit the information from the knowledge base by adapting the retrieved knowledge to the target samples. Extensive experiments on two public datasets show that D2K significantly outperforms existing baselines and is compatible with a major collection of recommendation algorithms.

{{</citation>}}


### (50/61) Towards Reliable and Factual Response Generation: Detecting Unanswerable Questions in Information-Seeking Conversations (Weronika Łajewska et al., 2024)

{{<citation>}}

Weronika Łajewska, Krisztian Balog. (2024)  
**Towards Reliable and Factual Response Generation: Detecting Unanswerable Questions in Information-Seeking Conversations**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.11452v1)  

---


**ABSTRACT**  
Generative AI models face the challenge of hallucinations that can undermine users' trust in such systems. We approach the problem of conversational information seeking as a two-step process, where relevant passages in a corpus are identified first and then summarized into a final system response. This way we can automatically assess if the answer to the user's question is present in the corpus. Specifically, our proposed method employs a sentence-level classifier to detect if the answer is present, then aggregates these predictions on the passage level, and eventually across the top-ranked passages to arrive at a final answerability estimate. For training and evaluation, we develop a dataset based on the TREC CAsT benchmark that includes answerability labels on the sentence, passage, and ranking levels. We demonstrate that our proposed method represents a strong baseline and outperforms a state-of-the-art LLM on the answerability prediction task.

{{</citation>}}


## cs.RO (5)



### (51/61) Integration of Large Language Models in Control of EHD Pumps for Precise Color Synthesis (Yanhong Peng et al., 2024)

{{<citation>}}

Yanhong Peng, Ceng Zhang, Chenlong Hu, Zebing Mao. (2024)  
**Integration of Large Language Models in Control of EHD Pumps for Precise Color Synthesis**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.11500v1)  

---


**ABSTRACT**  
This paper presents an innovative approach to integrating Large Language Models (LLMs) with Arduino-controlled Electrohydrodynamic (EHD) pumps for precise color synthesis in automation systems. We propose a novel framework that employs fine-tuned LLMs to interpret natural language commands and convert them into specific operational instructions for EHD pump control. This approach aims to enhance user interaction with complex hardware systems, making it more intuitive and efficient. The methodology involves four key steps: fine-tuning the language model with a dataset of color specifications and corresponding Arduino code, developing a natural language processing interface, translating user inputs into executable Arduino code, and controlling EHD pumps for accurate color mixing. Conceptual experiment results, based on theoretical assumptions, indicate a high potential for accurate color synthesis, efficient language model interpretation, and reliable EHD pump operation. This research extends the application of LLMs beyond text-based tasks, demonstrating their potential in industrial automation and control systems. While highlighting the limitations and the need for real-world testing, this study opens new avenues for AI applications in physical system control and sets a foundation for future advancements in AI-driven automation technologies.

{{</citation>}}


### (52/61) General Flow as Foundation Affordance for Scalable Robot Learning (Chengbo Yuan et al., 2024)

{{<citation>}}

Chengbo Yuan, Chuan Wen, Tong Zhang, Yang Gao. (2024)  
**General Flow as Foundation Affordance for Scalable Robot Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.11439v1)  

---


**ABSTRACT**  
We address the challenge of acquiring real-world manipulation skills with a scalable framework.Inspired by the success of large-scale auto-regressive prediction in Large Language Models (LLMs), we hold the belief that identifying an appropriate prediction target capable of leveraging large-scale datasets is crucial for achieving efficient and universal learning. Therefore, we propose to utilize flow, which represents the future trajectories of 3D points on objects of interest, as an ideal prediction target in robot learning. To exploit scalable data resources, we turn our attention to cross-embodiment datasets. We develop, for the first time, a language-conditioned prediction model directly from large-scale RGBD human video datasets. Our predicted flow offers actionable geometric and physics guidance, thus facilitating stable zero-shot skill transfer in real-world scenarios.We deploy our method with a policy based on closed-loop flow prediction. Remarkably, without any additional training, our method achieves an impressive 81% success rate in human-to-robot skill transfer, covering 18 tasks in 6 scenes. Our framework features the following benefits: (1) scalability: leveraging cross-embodiment data resources; (2) universality: multiple object categories, including rigid, articulated, and soft bodies; (3) stable skill transfer: providing actionable guidance with a small inference domain-gap. These lead to a new pathway towards scalable general robot learning. Data, code, and model weights will be made publicly available.

{{</citation>}}


### (53/61) Bimanual Deformable Bag Manipulation Using a Structure-of-Interest Based Latent Dynamics Model (Peng Zhou et al., 2024)

{{<citation>}}

Peng Zhou, Pai Zheng, Jiaming Qi, Chenxi Li, Chenguang Yang, David Navarro-Alarcon, Jia Pan. (2024)  
**Bimanual Deformable Bag Manipulation Using a Structure-of-Interest Based Latent Dynamics Model**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2401.11432v1)  

---


**ABSTRACT**  
The manipulation of deformable objects by robotic systems presents a significant challenge due to their complex and infinite-dimensional configuration spaces. This paper introduces a novel approach to Deformable Object Manipulation (DOM) by emphasizing the identification and manipulation of Structures of Interest (SOIs) in deformable fabric bags. We propose a bimanual manipulation framework that leverages a Graph Neural Network (GNN)-based latent dynamics model to succinctly represent and predict the behavior of these SOIs. Our approach involves constructing a graph representation from partial point cloud data of the object and learning the latent dynamics model that effectively captures the essential deformations of the fabric bag within a reduced computational space. By integrating this latent dynamics model with Model Predictive Control (MPC), we empower robotic manipulators to perform precise and stable manipulation tasks focused on the SOIs. We have validated our framework through various empirical experiments demonstrating its efficacy in bimanual manipulation of fabric bags. Our contributions not only address the complexities inherent in DOM but also provide new perspectives and methodologies for enhancing robotic interactions with deformable objects by concentrating on their critical structural elements. Experimental videos can be obtained from https://sites.google.com/view/bagbot.

{{</citation>}}


### (54/61) Multi-Agent Generative Adversarial Interactive Self-Imitation Learning for AUV Formation Control and Obstacle Avoidance (Zheng Fang et al., 2024)

{{<citation>}}

Zheng Fang, Tianhao Chen, Dong Jiang, Zheng Zhang, Guangliang Li. (2024)  
**Multi-Agent Generative Adversarial Interactive Self-Imitation Learning for AUV Formation Control and Obstacle Avoidance**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11378v1)  

---


**ABSTRACT**  
Multiple autonomous underwater vehicles (multi-AUV) can cooperatively accomplish tasks that a single AUV cannot complete. Recently, multi-agent reinforcement learning has been introduced to control of multi-AUV. However, designing efficient reward functions for various tasks of multi-AUV control is difficult or even impractical. Multi-agent generative adversarial imitation learning (MAGAIL) allows multi-AUV to learn from expert demonstration instead of pre-defined reward functions, but suffers from the deficiency of requiring optimal demonstrations and not surpassing provided expert demonstrations. This paper builds upon the MAGAIL algorithm by proposing multi-agent generative adversarial interactive self-imitation learning (MAGAISIL), which can facilitate AUVs to learn policies by gradually replacing the provided sub-optimal demonstrations with self-generated good trajectories selected by a human trainer. Our experimental results in a multi-AUV formation control and obstacle avoidance task on the Gazebo platform with AUV simulator of our lab show that AUVs trained via MAGAISIL can surpass the provided sub-optimal expert demonstrations and reach a performance close to or even better than MAGAIL with optimal demonstrations. Further results indicate that AUVs' policies trained via MAGAISIL can adapt to complex and different tasks as well as MAGAIL learning from optimal demonstrations.

{{</citation>}}


### (55/61) Back-stepping Experience Replay with Application to Model-free Reinforcement Learning for a Soft Snake Robot (Xinda Qi et al., 2024)

{{<citation>}}

Xinda Qi, Dong Chen, Zhaojian Li, Xiaobo Tan. (2024)  
**Back-stepping Experience Replay with Application to Model-free Reinforcement Learning for a Soft Snake Robot**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11372v1)  

---


**ABSTRACT**  
In this paper, we propose a novel technique, Back-stepping Experience Replay (BER), that is compatible with arbitrary off-policy reinforcement learning (RL) algorithms. BER aims to enhance learning efficiency in systems with approximate reversibility, reducing the need for complex reward shaping. The method constructs reversed trajectories using back-stepping transitions to reach random or fixed targets. Interpretable as a bi-directional approach, BER addresses inaccuracies in back-stepping transitions through a distillation of the replay experience during learning. Given the intricate nature of soft robots and their complex interactions with environments, we present an application of BER in a model-free RL approach for the locomotion and navigation of a soft snake robot, which is capable of serpentine motion enabled by anisotropic friction between the body and ground. In addition, a dynamic simulator is developed to assess the effectiveness and efficiency of the BER algorithm, in which the robot demonstrates successful learning (reaching a 100% success rate) and adeptly reaches random targets, achieving an average speed 48% faster than that of the best baseline approach.

{{</citation>}}


## cs.DC (1)



### (56/61) Accelerating Heterogeneous Tensor Parallelism via Flexible Workload Control (Zhigang Wang et al., 2024)

{{<citation>}}

Zhigang Wang, Xu Zhang, Ning Wang, Chuanfei Xu, Jie Nie, Zhiqiang Wei, Yu Gu, Ge Yu. (2024)  
**Accelerating Heterogeneous Tensor Parallelism via Flexible Workload Control**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11469v1)  

---


**ABSTRACT**  
Transformer-based models are becoming deeper and larger recently. For better scalability, an underlying training solution in industry is to split billions of parameters (tensors) into many tasks and then run them across homogeneous accelerators (e.g., GPUs). However, such dedicated compute cluster is prohibitively expensive in academia and moderate companies. An economic replacement is to aggregate existing heterogeneous devices and share resources among multi-tenants. Nevertheless, static hardware configurations and dynamic resource contention definitely cause straggling tasks, which heavily slows down the overall training efficiency. Existing works feature contributions mainly tailored for traditional data parallelism. They cannot work well for the new tensor parallelism due to strict communication and correctness constraints.   In this paper we first present ZERO-resizing, a novel dynamic workload balancing technique without any data migration. We tune workloads in real-time by temporarily resizing matrices involved in core tensor-related computations. We particularly design data imputation and priority selection policies to respectively satisfy consistency constraint required by normal training and reduce the accuracy loss. We also give a lightweight data migration technique without loss of accuracy, to cope with heavy heterogeneity. Our final SEMI-migration solution is built on top of these two techniques and can adaptively distinguish their respective balancing missions, to achieve an overall success in efficiency and accuracy. Extensive experiments on the representative Colossal-AI platform validate the effectiveness of our proposals.

{{</citation>}}


## cs.AR (1)



### (57/61) AttentionLego: An Open-Source Building Block For Spatially-Scalable Large Language Model Accelerator With Processing-In-Memory Technology (Rongqing Cong et al., 2024)

{{<citation>}}

Rongqing Cong, Wenyang He, Mingxuan Li, Bangning Luo, Zebin Yang, Yuchao Yang, Ru Huang, Bonan Yan. (2024)  
**AttentionLego: An Open-Source Building Block For Spatially-Scalable Large Language Model Accelerator With Processing-In-Memory Technology**  

---
Primary Category: cs.AR  
Categories: cs-AI, cs-AR, cs-LG, cs.AR  
Keywords: Attention, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11459v1)  

---


**ABSTRACT**  
Large language models (LLMs) with Transformer architectures have become phenomenal in natural language processing, multimodal generative artificial intelligence, and agent-oriented artificial intelligence. The self-attention module is the most dominating sub-structure inside Transformer-based LLMs. Computation using general-purpose graphics processing units (GPUs) inflicts reckless demand for I/O bandwidth for transferring intermediate calculation results between memories and processing units. To tackle this challenge, this work develops a fully customized vanilla self-attention accelerator, AttentionLego, as the basic building block for constructing spatially expandable LLM processors. AttentionLego provides basic implementation with fully-customized digital logic incorporating Processing-In-Memory (PIM) technology. It is based on PIM-based matrix-vector multiplication and look-up table-based Softmax design. The open-source code is available online: https://bonany.cc/attentionleg.

{{</citation>}}


## eess.SP (1)



### (58/61) Energy Consumption Analysis for Continuous Phase Modulation in Smart-Grid Internet of Things of beyond 5G (Hongjian Gao et al., 2024)

{{<citation>}}

Hongjian Gao, Yang Lu, Shaoshi Yang, Jingsheng Tan, Longlong Nie, Xinyi Qu. (2024)  
**Energy Consumption Analysis for Continuous Phase Modulation in Smart-Grid Internet of Things of beyond 5G**  

---
Primary Category: eess.SP  
Categories: cs-NI, eess-SP, eess.SP  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.11449v1)  

---


**ABSTRACT**  
Wireless sensor network (WSN) underpinning the smart-grid Internet of Things (SG-IoT) has been a popular research topic in recent years due to its great potential for enabling a wide range of important applications. However, the energy consumption (EC) characteristic of sensor nodes is a key factor that affects the operational performance (e.g., lifetime of sensors) and the total cost of ownership of WSNs. In this paper, to find the modulation techniques suitable for WSNs, we investigate the EC characteristic of continuous phase modulation (CPM), which is an attractive modulation scheme candidate for WSNs because of its constant envelope property. We first develop an EC model for the sensor nodes of WSNs by considering the circuits and a typical communication protocol that relies on automatic repeat request (ARQ)-based retransmissions to ensure successful data delivery. Then, we use this model to analyze the EC characteristic of CPM under various configurations of modulation parameters. Furthermore, we compare the EC characteristic of CPM with that of other representative modulation schemes, such as offset quadrature phase-shift keying (OQPSK) and quadrature amplitude modulation (QAM), which are commonly used in communication protocols of WSNs. Our analysis and simulation results provide insights into the EC characteristics of multiple modulation schemes in the context of WSNs; thus, they are beneficial for designing energy-efficient SG-IoT in the beyond-5G (B5G) and the 6G era.

{{</citation>}}


## cs.NI (1)



### (59/61) Interactive AI with Retrieval-Augmented Generation for Next Generation Networking (Ruichen Zhang et al., 2024)

{{<citation>}}

Ruichen Zhang, Hongyang Du, Yinqiu Liu, Dusit Niyato, Jiawen Kang, Sumei Sun, Xuemin Shen, H. Vincent Poor. (2024)  
**Interactive AI with Retrieval-Augmented Generation for Next Generation Networking**  

---
Primary Category: cs.NI  
Categories: cs-IT, cs-NI, cs.NI, math-IT  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2401.11391v1)  

---


**ABSTRACT**  
With the advance of artificial intelligence (AI), the emergence of Google Gemini and OpenAI Q* marks the direction towards artificial general intelligence (AGI). To implement AGI, the concept of interactive AI (IAI) has been introduced, which can interactively understand and respond not only to human user input but also to dynamic system and network conditions. In this article, we explore an integration and enhancement of IAI in networking. We first comprehensively review recent developments and future perspectives of AI and then introduce the technology and components of IAI. We then explore the integration of IAI into the next-generation networks, focusing on how implicit and explicit interactions can enhance network functionality, improve user experience, and promote efficient network management. Subsequently, we propose an IAI-enabled network management and optimization framework, which consists of environment, perception, action, and brain units. We also design the pluggable large language model (LLM) module and retrieval augmented generation (RAG) module to build the knowledge base and contextual memory for decision-making in the brain unit. We demonstrate the effectiveness of the framework through case studies. Finally, we discuss potential research directions for IAI-based networks.

{{</citation>}}


## cs.SE (1)



### (60/61) Revolutionizing API Documentation through Summarization (AmirHossein Naghshzan et al., 2024)

{{<citation>}}

AmirHossein Naghshzan, Sylvie Ratte. (2024)  
**Revolutionizing API Documentation through Summarization**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-SE, cs.SE  
Keywords: BERT, Summarization  
[Paper Link](http://arxiv.org/abs/2401.11361v1)  

---


**ABSTRACT**  
This study tackles the challenges associated with interpreting Application Programming Interface (API) documentation, an integral aspect of software development. Official API documentation, while essential, can be lengthy and challenging to navigate, prompting developers to seek unofficial sources such as Stack Overflow. Leveraging the vast user-generated content on Stack Overflow, including code snippets and discussions, we employ BERTopic and extractive summarization to automatically generate concise and informative API summaries. These summaries encompass key insights like general usage, common developer issues, and potential solutions, sourced from the wealth of knowledge on Stack Overflow. Software developers evaluate these summaries for performance, coherence, and interoperability, providing valuable feedback on the practicality of our approach.

{{</citation>}}


## physics.flu-dyn (1)



### (61/61) Asynchronous Parallel Reinforcement Learning for Optimizing Propulsive Performance in Fin Ray Control (Xin-Yang Liu et al., 2024)

{{<citation>}}

Xin-Yang Liu, Dariush Bodaghi, Qian Xue, Xudong Zheng, Jian-Xun Wang. (2024)  
**Asynchronous Parallel Reinforcement Learning for Optimizing Propulsive Performance in Fin Ray Control**  

---
Primary Category: physics.flu-dyn  
Categories: cs-LG, cs-SY, eess-SY, physics-flu-dyn, physics.flu-dyn  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11349v1)  

---


**ABSTRACT**  
Fish fin rays constitute a sophisticated control system for ray-finned fish, facilitating versatile locomotion within complex fluid environments. Despite extensive research on the kinematics and hydrodynamics of fish locomotion, the intricate control strategies in fin-ray actuation remain largely unexplored. While deep reinforcement learning (DRL) has demonstrated potential in managing complex nonlinear dynamics; its trial-and-error nature limits its application to problems involving computationally demanding environmental interactions. This study introduces a cutting-edge off-policy DRL algorithm, interacting with a fluid-structure interaction (FSI) environment to acquire intricate fin-ray control strategies tailored for various propulsive performance objectives. To enhance training efficiency and enable scalable parallelism, an innovative asynchronous parallel training (APT) strategy is proposed, which fully decouples FSI environment interactions and policy/value network optimization. The results demonstrated the success of the proposed method in discovering optimal complex policies for fin-ray actuation control, resulting in a superior propulsive performance compared to the optimal sinusoidal actuation function identified through a parametric grid search. The merit and effectiveness of the APT approach are also showcased through comprehensive comparison with conventional DRL training strategies in numerical experiments of controlling nonlinear dynamics.

{{</citation>}}
