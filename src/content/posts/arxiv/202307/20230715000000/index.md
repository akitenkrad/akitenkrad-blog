---
draft: false
title: "arXiv @ 2023.07.15"
date: 2023-07-15
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.15"
    identifier: arxiv_20230715
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (11)](#cslg-11)
- [cs.CY (2)](#cscy-2)
- [cs.CL (9)](#cscl-9)
- [cs.CV (9)](#cscv-9)
- [cs.AI (5)](#csai-5)
- [q-fin.ST (1)](#q-finst-1)
- [cs.SE (3)](#csse-3)
- [cs.RO (1)](#csro-1)
- [cs.NE (1)](#csne-1)
- [cs.IR (2)](#csir-2)
- [cs.CE (1)](#csce-1)
- [cs.CR (1)](#cscr-1)

## cs.LG (11)



### (1/46) Seeing is not Believing: Robust Reinforcement Learning against Spurious Correlation (Wenhao Ding et al., 2023)

{{<citation>}}

Wenhao Ding, Laixi Shi, Yuejie Chi, Ding Zhao. (2023)  
**Seeing is not Believing: Robust Reinforcement Learning against Spurious Correlation**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.07907v1)  

---


**ABSTRACT**  
Robustness has been extensively studied in reinforcement learning (RL) to handle various forms of uncertainty such as random perturbations, rare events, and malicious attacks. In this work, we consider one critical type of robustness against spurious correlation, where different portions of the state do not have causality but have correlations induced by unobserved confounders. These spurious correlations are ubiquitous in real-world tasks, for instance, a self-driving car usually observes heavy traffic in the daytime and light traffic at night due to unobservable human activity. A model that learns such useless or even harmful correlation could catastrophically fail when the confounder in the test case deviates from the training one. Although motivated, enabling robustness against spurious correlation poses significant challenges since the uncertainty set, shaped by the unobserved confounder and sequential structure of RL, is difficult to characterize and identify. Existing robust algorithms that assume simple and unstructured uncertainty sets are therefore inadequate to address this challenge. To solve this issue, we propose Robust State-Confounded Markov Decision Processes (RSC-MDPs) and theoretically demonstrate its superiority in breaking spurious correlations compared with other robust RL counterparts. We also design an empirical algorithm to learn the robust optimal policy for RSC-MDPs, which outperforms all baselines in eight realistic self-driving and manipulation tasks.

{{</citation>}}


### (2/46) Does Double Descent Occur in Self-Supervised Learning? (Alisia Lupidi et al., 2023)

{{<citation>}}

Alisia Lupidi, Yonatan Gideoni, Dulhan Jayalath. (2023)  
**Does Double Descent Occur in Self-Supervised Learning?**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.07872v1)  

---


**ABSTRACT**  
Most investigations into double descent have focused on supervised models while the few works studying self-supervised settings find a surprising lack of the phenomenon. These results imply that double descent may not exist in self-supervised models. We show this empirically using a standard and linear autoencoder, two previously unstudied settings. The test loss is found to have either a classical U-shape or to monotonically decrease instead of exhibiting a double-descent curve. We hope that further work on this will help elucidate the theoretical underpinnings of this phenomenon.

{{</citation>}}


### (3/46) Transformers are Universal Predictors (Sourya Basu et al., 2023)

{{<citation>}}

Sourya Basu, Moulik Choraria, Lav R. Varshney. (2023)  
**Transformers are Universal Predictors**  

---
Primary Category: cs.LG
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.07843v1)  

---


**ABSTRACT**  
We find limits to the Transformer architecture for language modeling and show it has a universal prediction property in an information-theoretic sense. We further analyze performance in non-asymptotic data regimes to understand the role of various components of the Transformer architecture, especially in the context of data-efficient training. We validate our theoretical analysis with experiments on both synthetic and real datasets.

{{</citation>}}


### (4/46) RegExplainer: Generating Explanations for Graph Neural Networks in Regression Task (Jiaxing Zhang et al., 2023)

{{<citation>}}

Jiaxing Zhang, Zhuomin Chen, Hao Mei, Dongsheng Luo, Hua Wei. (2023)  
**RegExplainer: Generating Explanations for Graph Neural Networks in Regression Task**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.07840v1)  

---


**ABSTRACT**  
Graph regression is a fundamental task and has received increasing attention in a wide range of graph learning tasks. However, the inference process is often not interpretable. Most existing explanation techniques are limited to understanding GNN behaviors in classification tasks. In this work, we seek an explanation to interpret the graph regression models (XAIG-R). We show that existing methods overlook the distribution shifting and continuously ordered decision boundary, which hinders them away from being applied in the regression tasks. To address these challenges, we propose a novel objective based on the information bottleneck theory and introduce a new mix-up framework, which could support various GNNs in a model-agnostic manner. We further present a contrastive learning strategy to tackle the continuously ordered labels in regression task. To empirically verify the effectiveness of the proposed method, we introduce three benchmark datasets and a real-life dataset for evaluation. Extensive experiments show the effectiveness of the proposed method in interpreting GNN models in regression tasks.

{{</citation>}}


### (5/46) MixupExplainer: Generalizing Explanations for Graph Neural Networks with Data Augmentation (Jiaxing Zhang et al., 2023)

{{<citation>}}

Jiaxing Zhang, Dongsheng Luo, Hua Wei. (2023)  
**MixupExplainer: Generalizing Explanations for Graph Neural Networks with Data Augmentation**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Augmentation, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.07832v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have received increasing attention due to their ability to learn from graph-structured data. However, their predictions are often not interpretable. Post-hoc instance-level explanation methods have been proposed to understand GNN predictions. These methods seek to discover substructures that explain the prediction behavior of a trained GNN. In this paper, we shed light on the existence of the distribution shifting issue in existing methods, which affects explanation quality, particularly in applications on real-life datasets with tight decision boundaries. To address this issue, we introduce a generalized Graph Information Bottleneck (GIB) form that includes a label-independent graph variable, which is equivalent to the vanilla GIB. Driven by the generalized GIB, we propose a graph mixup method, MixupExplainer, with a theoretical guarantee to resolve the distribution shifting issue. We conduct extensive experiments on both synthetic and real-world datasets to validate the effectiveness of our proposed mixup approach over existing approaches. We also provide a detailed analysis of how our proposed approach alleviates the distribution shifting issue.

{{</citation>}}


### (6/46) CatBoost Versus XGBoost and LightGBM: Developing Enhanced Predictive Models for Zero-Inflated Insurance Claim Data (Banghee So, 2023)

{{<citation>}}

Banghee So. (2023)  
**CatBoost Versus XGBoost and LightGBM: Developing Enhanced Predictive Models for Zero-Inflated Insurance Claim Data**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2307.07771v1)  

---


**ABSTRACT**  
In the property and casualty insurance industry, some challenges are presented in constructing claim predictive models due to a highly right-skewed distribution of positive claims with excess zeros. Traditional models, such as Poisson or negative binomial Generalized Linear Models(GLMs), frequently struggle with inflated zeros. In response to this, researchers in actuarial science have employed ``zero-inflated" models that merge a traditional count model and a binary model to address these datasets more effectively. This paper uses boosting algorithms to process insurance claim data, including zero-inflated telematics data, in order to construct claim frequency models. We evaluated and compared three popular gradient boosting libraries - XGBoost, LightGBM, and CatBoost - with the aim of identifying the most suitable library for training insurance claim data and fitting actuarial frequency models. Through a rigorous analysis of two distinct datasets, we demonstrated that CatBoost is superior in developing auto claim frequency models based on predictive performance. We also found that Zero-inflated Poisson boosted tree models, with variations in their assumptions about the relationship between inflation probability and distribution mean, outperformed others depending on data characteristics. Furthermore, by using a specific CatBoost tool, we explored the effects and interactions of different risk features on the frequency model when using telematics data.

{{</citation>}}


### (7/46) randomHAR: Improving Ensemble Deep Learners for Human Activity Recognition with Sensor Selection and Reinforcement Learning (Yiran Huang et al., 2023)

{{<citation>}}

Yiran Huang, Yexu Zhou, Till Riedel, Likun Fang, Michael Beigl. (2023)  
**randomHAR: Improving Ensemble Deep Learners for Human Activity Recognition with Sensor Selection and Reinforcement Learning**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG, eess-SP  
Keywords: LSTM, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.07770v1)  

---


**ABSTRACT**  
Deep learning has proven to be an effective approach in the field of Human activity recognition (HAR), outperforming other architectures that require manual feature engineering. Despite recent advancements, challenges inherent to HAR data, such as noisy data, intra-class variability and inter-class similarity, remain. To address these challenges, we propose an ensemble method, called randomHAR. The general idea behind randomHAR is training a series of deep learning models with the same architecture on randomly selected sensor data from the given dataset. Besides, an agent is trained with the reinforcement learning algorithm to identify the optimal subset of the trained models that are utilized for runtime prediction. In contrast to existing work, this approach optimizes the ensemble process rather than the architecture of the constituent models. To assess the performance of the approach, we compare it against two HAR algorithms, including the current state of the art, on six HAR benchmark datasets. The result of the experiment demonstrates that the proposed approach outperforms the state-of-the-art method, ensembleLSTM.

{{</citation>}}


### (8/46) Learning Expressive Priors for Generalization and Uncertainty Estimation in Neural Networks (Dominik Schnaus et al., 2023)

{{<citation>}}

Dominik Schnaus, Jongseok Lee, Daniel Cremers, Rudolph Triebel. (2023)  
**Learning Expressive Priors for Generalization and Uncertainty Estimation in Neural Networks**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.07753v1)  

---


**ABSTRACT**  
In this work, we propose a novel prior learning method for advancing generalization and uncertainty estimation in deep neural networks. The key idea is to exploit scalable and structured posteriors of neural networks as informative priors with generalization guarantees. Our learned priors provide expressive probabilistic representations at large scale, like Bayesian counterparts of pre-trained models on ImageNet, and further produce non-vacuous generalization bounds. We also extend this idea to a continual learning framework, where the favorable properties of our priors are desirable. Major enablers are our technical contributions: (1) the sums-of-Kronecker-product computations, and (2) the derivations and optimizations of tractable objectives that lead to improved generalization bounds. Empirically, we exhaustively show the effectiveness of this method for uncertainty estimation and generalization.

{{</citation>}}


### (9/46) An Empirical Study of the Effectiveness of Using a Replay Buffer on Mode Discovery in GFlowNets (Nikhil Vemgal et al., 2023)

{{<citation>}}

Nikhil Vemgal, Elaine Lau, Doina Precup. (2023)  
**An Empirical Study of the Effectiveness of Using a Replay Buffer on Mode Discovery in GFlowNets**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.07674v2)  

---


**ABSTRACT**  
Reinforcement Learning (RL) algorithms aim to learn an optimal policy by iteratively sampling actions to learn how to maximize the total expected return, $R(x)$. GFlowNets are a special class of algorithms designed to generate diverse candidates, $x$, from a discrete set, by learning a policy that approximates the proportional sampling of $R(x)$. GFlowNets exhibit improved mode discovery compared to conventional RL algorithms, which is very useful for applications such as drug discovery and combinatorial search. However, since GFlowNets are a relatively recent class of algorithms, many techniques which are useful in RL have not yet been associated with them. In this paper, we study the utilization of a replay buffer for GFlowNets. We explore empirically various replay buffer sampling techniques and assess the impact on the speed of mode discovery and the quality of the modes discovered. Our experimental results in the Hypergrid toy domain and a molecule synthesis environment demonstrate significant improvements in mode discovery when training with a replay buffer, compared to training only with trajectories generated on-policy.

{{</citation>}}


### (10/46) Efficient Adversarial Attacks on Online Multi-agent Reinforcement Learning (Guanlin Liu et al., 2023)

{{<citation>}}

Guanlin Liu, Lifeng Lai. (2023)  
**Efficient Adversarial Attacks on Online Multi-agent Reinforcement Learning**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-CR, cs-LG, cs.LG, math-OC  
Keywords: Adversarial Attack, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.07670v1)  

---


**ABSTRACT**  
Due to the broad range of applications of multi-agent reinforcement learning (MARL), understanding the effects of adversarial attacks against MARL model is essential for the safe applications of this model. Motivated by this, we investigate the impact of adversarial attacks on MARL. In the considered setup, there is an exogenous attacker who is able to modify the rewards before the agents receive them or manipulate the actions before the environment receives them. The attacker aims to guide each agent into a target policy or maximize the cumulative rewards under some specific reward function chosen by the attacker, while minimizing the amount of manipulation on feedback and action. We first show the limitations of the action poisoning only attacks and the reward poisoning only attacks. We then introduce a mixed attack strategy with both the action poisoning and the reward poisoning. We show that the mixed attack strategy can efficiently attack MARL agents even if the attacker has no prior information about the underlying environment and the agents' algorithms.

{{</citation>}}


### (11/46) Efficient Action Robust Reinforcement Learning with Probabilistic Policy Execution Uncertainty (Guanlin Liu et al., 2023)

{{<citation>}}

Guanlin Liu, Zhihan Zhou, Han Liu, Lifeng Lai. (2023)  
**Efficient Action Robust Reinforcement Learning with Probabilistic Policy Execution Uncertainty**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.07666v2)  

---


**ABSTRACT**  
Robust reinforcement learning (RL) aims to find a policy that optimizes the worst-case performance in the face of uncertainties. In this paper, we focus on action robust RL with the probabilistic policy execution uncertainty, in which, instead of always carrying out the action specified by the policy, the agent will take the action specified by the policy with probability $1-\rho$ and an alternative adversarial action with probability $\rho$. We establish the existence of an optimal policy on the action robust MDPs with probabilistic policy execution uncertainty and provide the action robust Bellman optimality equation for its solution. Furthermore, we develop Action Robust Reinforcement Learning with Certificates (ARRLC) algorithm that achieves minimax optimal regret and sample complexity. Furthermore, we conduct numerical experiments to validate our approach's robustness, demonstrating that ARRLC outperforms non-robust RL algorithms and converges faster than the robust TD algorithm in the presence of action perturbations.

{{</citation>}}


## cs.CY (2)



### (12/46) The science of fake news (David M. J. Lazer et al., 2023)

{{<citation>}}

David M. J. Lazer, Matthew A. Baum, Yochai Benkler, Adam J. Berinsky, Kelly M. Greenhill, Filippo Menczer, Miriam J. Metzger, Brendan Nyhan, Gordon Pennycook, David Rothschild, Michael Schudson, Steven A. Sloman, Cass R. Sunstein, Emily A. Thorson, Duncan J. Watts, Jonathan L. Zittrain. (2023)  
**The science of fake news**  

---
Primary Category: cs.CY
Categories: cs-CY, cs.CY  
Keywords: Google, Twitter  
[Paper Link](http://arxiv.org/abs/2307.07903v1)  

---


**ABSTRACT**  
Fake news emerged as an apparent global problem during the 2016 U.S. Presidential election. Addressing it requires a multidisciplinary effort to define the nature and extent of the problem, detect fake news in real time, and mitigate its potentially harmful effects. This will require a better understanding of how the Internet spreads content, how people process news, and how the two interact. We review the state of knowledge in these areas and discuss two broad potential mitigation strategies: better enabling individuals to identify fake news, and intervention within the platforms to reduce the attention given to fake news. The cooperation of Internet platforms (especially Facebook, Google, and Twitter) with researchers will be critical to understanding the scale of the issue and the effectiveness of possible interventions.

{{</citation>}}


### (13/46) Bound by the Bounty: Collaboratively Shaping Evaluation Processes for Queer AI Harms (Organizers of QueerInAI et al., 2023)

{{<citation>}}

Organizers of QueerInAI, Nathan Dennler, Anaelia Ovalle, Ashwin Singh, Luca Soldaini, Arjun Subramonian, Huy Tu, William Agnew, Avijit Ghosh, Kyra Yee, Irene Font Peradejordi, Zeerak Talat, Mayra Russo, Jess de Jesus de Pinho Pinhal. (2023)  
**Bound by the Bounty: Collaboratively Shaping Evaluation Processes for Queer AI Harms**  

---
Primary Category: cs.CY
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2307.10223v1)  

---


**ABSTRACT**  
Bias evaluation benchmarks and dataset and model documentation have emerged as central processes for assessing the biases and harms of artificial intelligence (AI) systems. However, these auditing processes have been criticized for their failure to integrate the knowledge of marginalized communities and consider the power dynamics between auditors and the communities. Consequently, modes of bias evaluation have been proposed that engage impacted communities in identifying and assessing the harms of AI systems (e.g., bias bounties). Even so, asking what marginalized communities want from such auditing processes has been neglected. In this paper, we ask queer communities for their positions on, and desires from, auditing processes. To this end, we organized a participatory workshop to critique and redesign bias bounties from queer perspectives. We found that when given space, the scope of feedback from workshop participants goes far beyond what bias bounties afford, with participants questioning the ownership, incentives, and efficacy of bounties. We conclude by advocating for community ownership of bounties and complementing bounties with participatory processes (e.g., co-creation).

{{</citation>}}


## cs.CL (9)



### (14/46) A Dialogue System for Assessing Activities of Daily Living: Improving Consistency with Grounded Knowledge (Zhecheng Sheng et al., 2023)

{{<citation>}}

Zhecheng Sheng, Raymond Finzel, Michael Lucke, Sheena Dufresne, Maria Gini, Serguei Pakhomov. (2023)  
**A Dialogue System for Assessing Activities of Daily Living: Improving Consistency with Grounded Knowledge**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue, GPT, NLU  
[Paper Link](http://arxiv.org/abs/2307.07544v1)  

---


**ABSTRACT**  
In healthcare, the ability to care for oneself is reflected in the "Activities of Daily Living (ADL)," which serve as a measure of functional ability (functioning). A lack of functioning may lead to poor living conditions requiring personal care and assistance. To accurately identify those in need of support, assistance programs continuously evaluate participants' functioning across various domains. However, the assessment process may encounter consistency issues when multiple assessors with varying levels of expertise are involved. Novice assessors, in particular, may lack the necessary preparation for real-world interactions with participants. To address this issue, we developed a dialogue system that simulates interactions between assessors and individuals of varying functioning in a natural and reproducible way. The dialogue system consists of two major modules, one for natural language understanding (NLU) and one for natural language generation (NLG), respectively. In order to generate responses consistent with the underlying knowledge base, the dialogue system requires both an understanding of the user's query and of biographical details of an individual being simulated. To fulfill this requirement, we experimented with query classification and generated responses based on those biographical details using some recently released InstructGPT-like models.

{{</citation>}}


### (15/46) Zero-shot NLG evaluation through Pairware Comparisons with LLMs (Adian Liusie et al., 2023)

{{<citation>}}

Adian Liusie, Potsawee Manakul, Mark J. F. Gales. (2023)  
**Zero-shot NLG evaluation through Pairware Comparisons with LLMs**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Language Model, Natural Language Generation, T5  
[Paper Link](http://arxiv.org/abs/2307.07889v1)  

---


**ABSTRACT**  
Evaluating Natural Language Generation (NLG) outputs is crucial but laborious and expensive. While various automatic NLG assessment methods have been proposed, they often are quite task-specific and have to be engineered with a particular domain and attribute in mind. In this work, we propose a robust zero-shot approach to NLG evaluation using pairwise comparative judgment with open-source Large Language Models (LLMs). The motivation for this approach is that even as humans, it is easier to determine which of two options are better, than it is to independently objectively score each option. We use this insight and leverage the emergent abilities of LLMs, where we probe FlanT5 to determine which of two candidate responses is better, rather than assigning absolute scores. Our results demonstrate that comparative assessment is a more effective approach than absolute scoring, enabling smaller open-source LLMs to achieve comparable performance to larger public access APIs. We evaluate systems on both summary evaluation and dialogue response generation, and show that opensource LLMs can lead to good correlations with human scores for a range of different attributes.

{{</citation>}}


### (16/46) Is Prompt-Based Finetuning Always Better than Vanilla Finetuning? Insights from Cross-Lingual Language Understanding (Bolei Ma et al., 2023)

{{<citation>}}

Bolei Ma, Ercong Nie, Helmut Schmid, Hinrich Schütze. (2023)  
**Is Prompt-Based Finetuning Always Better than Vanilla Finetuning? Insights from Cross-Lingual Language Understanding**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2307.07880v1)  

---


**ABSTRACT**  
Multilingual pretrained language models (MPLMs) have demonstrated substantial performance improvements in zero-shot cross-lingual transfer across various natural language understanding tasks by finetuning MPLMs on task-specific labelled data of a source language (e.g. English) and evaluating on a wide range of target languages. Recent studies show that prompt-based finetuning surpasses regular finetuning in few-shot scenarios. However, the exploration of prompt-based learning in multilingual tasks remains limited. In this study, we propose the ProFiT pipeline to investigate the cross-lingual capabilities of Prompt-based Finetuning. We conduct comprehensive experiments on diverse cross-lingual language understanding tasks (sentiment classification, paraphrase identification, and natural language inference) and empirically analyze the variation trends of prompt-based finetuning performance in cross-lingual transfer across different few-shot and full-data settings. Our results reveal the effectiveness and versatility of prompt-based finetuning in cross-lingual language understanding. Our findings indicate that prompt-based finetuning outperforms vanilla finetuning in full-data scenarios and exhibits greater advantages in few-shot scenarios, with different performance patterns dependent on task types. Additionally, we analyze underlying factors such as language similarity and pretraining data size that impact the cross-lingual performance of prompt-based finetuning. Overall, our work provides valuable insights into the cross-lingual prowess of prompt-based finetuning.

{{</citation>}}


### (17/46) Large Language Models as Superpositions of Cultural Perspectives (Grgur Kovač et al., 2023)

{{<citation>}}

Grgur Kovač, Masataka Sawayama, Rémy Portelas, Cédric Colas, Peter Ford Dominey, Pierre-Yves Oudeyer. (2023)  
**Large Language Models as Superpositions of Cultural Perspectives**  

---
Primary Category: cs.CL
Categories: 68T07, I-2-7, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.07870v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are often misleadingly recognized as having a personality or a set of values. We argue that an LLM can be seen as a superposition of perspectives with different values and personality traits. LLMs exhibit context-dependent values and personality traits that change based on the induced perspective (as opposed to humans, who tend to have more coherent values and personality traits across contexts). We introduce the concept of perspective controllability, which refers to a model's affordance to adopt various perspectives with differing values and personality traits. In our experiments, we use questionnaires from psychology (PVQ, VSM, IPIP) to study how exhibited values and personality traits change based on different perspectives. Through qualitative experiments, we show that LLMs express different values when those are (implicitly or explicitly) implied in the prompt, and that LLMs express different values even when those are not obviously implied (demonstrating their context-dependent nature). We then conduct quantitative experiments to study the controllability of different models (GPT-4, GPT-3.5, OpenAssistant, StableVicuna, StableLM), the effectiveness of various methods for inducing perspectives, and the smoothness of the models' drivability. We conclude by examining the broader implications of our work and outline a variety of associated scientific questions. The project website is available at https://sites.google.com/view/llm-superpositions .

{{</citation>}}


### (18/46) AspectCSE: Sentence Embeddings for Aspect-based Semantic Textual Similarity using Contrastive Learning and Structured Knowledge (Tim Schopf et al., 2023)

{{<citation>}}

Tim Schopf, Emanuel Gerber, Malte Ostendorff, Florian Matthes. (2023)  
**AspectCSE: Sentence Embeddings for Aspect-based Semantic Textual Similarity using Contrastive Learning and Structured Knowledge**  

---
Primary Category: cs.CL
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Contrastive Learning, Embedding, Sentence Embedding, Textual Similarity  
[Paper Link](http://arxiv.org/abs/2307.07851v1)  

---


**ABSTRACT**  
Generic sentence embeddings provide a coarse-grained approximation of semantic textual similarity but ignore specific aspects that make texts similar. Conversely, aspect-based sentence embeddings provide similarities between texts based on certain predefined aspects. Thus, similarity predictions of texts are more targeted to specific requirements and more easily explainable. In this paper, we present AspectCSE, an approach for aspect-based contrastive learning of sentence embeddings. Results indicate that AspectCSE achieves an average improvement of 3.97% on information retrieval tasks across multiple aspects compared to the previous best results. We also propose using Wikidata knowledge graph properties to train models of multi-aspect sentence embeddings in which multiple specific aspects are simultaneously considered during similarity predictions. We demonstrate that multi-aspect embeddings outperform single-aspect embeddings on aspect-specific information retrieval tasks. Finally, we examine the aspect-based sentence embedding space and demonstrate that embeddings of semantically similar aspect labels are often close, even without explicit similarity training between different aspect labels.

{{</citation>}}


### (19/46) Political Sentiment Analysis of Persian Tweets Using CNN-LSTM Model (Mohammad Dehghani et al., 2023)

{{<citation>}}

Mohammad Dehghani, Zahra Yazdanparast. (2023)  
**Political Sentiment Analysis of Persian Tweets Using CNN-LSTM Model**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-IR, cs.CL  
Keywords: BERT, LSTM, Sentiment Analysis, Twitter  
[Paper Link](http://arxiv.org/abs/2307.07740v1)  

---


**ABSTRACT**  
Sentiment analysis is the process of identifying and categorizing people's emotions or opinions regarding various topics. The analysis of Twitter sentiment has become an increasingly popular topic in recent years. In this paper, we present several machine learning and a deep learning model to analysis sentiment of Persian political tweets. Our analysis was conducted using Bag of Words and ParsBERT for word representation. We applied Gaussian Naive Bayes, Gradient Boosting, Logistic Regression, Decision Trees, Random Forests, as well as a combination of CNN and LSTM to classify the polarities of tweets. The results of this study indicate that deep learning with ParsBERT embedding performs better than machine learning. The CNN-LSTM model had the highest classification accuracy with 89 percent on the first dataset with three classes and 71 percent on the second dataset with seven classes. Due to the complexity of Persian, it was a difficult task to achieve this level of efficiency.

{{</citation>}}


### (20/46) CPET: Effective Parameter-Efficient Tuning for Compressed Large Language Models (Weilin Zhao et al., 2023)

{{<citation>}}

Weilin Zhao, Yuxiang Huang, Xu Han, Zhiyuan Liu, Zhengyan Zhang, Maosong Sun. (2023)  
**CPET: Effective Parameter-Efficient Tuning for Compressed Large Language Models**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.07705v1)  

---


**ABSTRACT**  
Parameter-efficient tuning (PET) has been widely explored in recent years because it tunes much fewer parameters (PET modules) than full-parameter fine-tuning (FT) while still stimulating sufficient knowledge from large language models (LLMs) for downstream tasks. Moreover, when PET is employed to serve multiple tasks, different task-specific PET modules can be built on a frozen LLM, avoiding redundant LLM deployments. Although PET significantly reduces the cost of tuning and deploying LLMs, its inference still suffers from the computational bottleneck of LLMs. To address the above issue, we propose an effective PET framework based on compressed LLMs, named "CPET". In CPET, we evaluate the impact of mainstream LLM compression techniques on PET performance and then introduce knowledge inheritance and recovery strategies to restore the knowledge loss caused by these compression techniques. Our experimental results demonstrate that, owing to the restoring strategies of CPET, collaborating task-specific PET modules with a compressed LLM can achieve comparable performance to collaborating PET modules with the original version of the compressed LLM and outperform directly applying vanilla PET methods to the compressed LLM.

{{</citation>}}


### (21/46) Think-on-Graph: Deep and Responsible Reasoning of Large Language Model with Knowledge Graph (Jiashuo Sun et al., 2023)

{{<citation>}}

Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Heung-Yeung Shum, Jian Guo. (2023)  
**Think-on-Graph: Deep and Responsible Reasoning of Large Language Model with Knowledge Graph**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.07697v1)  

---


**ABSTRACT**  
Large language models (LLMs) have made significant strides in various tasks, yet they often struggle with complex reasoning and exhibit poor performance in scenarios where knowledge traceability, timeliness, and accuracy are crucial. To address these limitations, we present Think-on-Graph (ToG), a novel framework that leverages knowledge graphs to enhance LLMs' ability for deep and responsible reasoning. By employing ToG, we can identify entities relevant to a given question and conduct exploration and reasoning to retrieve related triples from an external knowledge database. This iterative procedure generates multiple reasoning pathways consisting of sequentially connected triplets until sufficient information is gathered to answer the question or the maximum depth is reached. Through experiments on complex multi-hop reasoning question-answering tasks, we demonstrate that ToG outperforms existing methods, effectively addressing the aforementioned limitations of LLMs without incurring additional training costs.

{{</citation>}}


### (22/46) Coupling Large Language Models with Logic Programming for Robust and General Reasoning from Text (Zhun Yang et al., 2023)

{{<citation>}}

Zhun Yang, Adam Ishay, Joohyung Lee. (2023)  
**Coupling Large Language Models with Logic Programming for Robust and General Reasoning from Text**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs-SC, cs.CL  
Keywords: GPT, Language Model, NLP, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.07696v1)  

---


**ABSTRACT**  
While large language models (LLMs), such as GPT-3, appear to be robust and general, their reasoning ability is not at a level to compete with the best models trained for specific natural language reasoning problems. In this study, we observe that a large language model can serve as a highly effective few-shot semantic parser. It can convert natural language sentences into a logical form that serves as input for answer set programs, a logic-based declarative knowledge representation formalism. The combination results in a robust and general system that can handle multiple question-answering tasks without requiring retraining for each new task. It only needs a few examples to guide the LLM's adaptation to a specific task, along with reusable ASP knowledge modules that can be applied to multiple tasks. We demonstrate that this method achieves state-of-the-art performance on several NLP benchmarks, including bAbI, StepGame, CLUTRR, and gSCAN. Additionally, it successfully tackles robot planning tasks that an LLM alone fails to solve.

{{</citation>}}


## cs.CV (9)



### (23/46) Anomaly Detection in Automated Fibre Placement: Learning with Data Limitations (Assef Ghamisi et al., 2023)

{{<citation>}}

Assef Ghamisi, Todd Charter, Li Ji, Maxime Rivard, Gil Lund, Homayoun Najjaran. (2023)  
**Anomaly Detection in Automated Fibre Placement: Learning with Data Limitations**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.07893v1)  

---


**ABSTRACT**  
Current defect detection systems for Automated Fibre Placement (AFP) are mostly based on end-to-end supervised learning methods requiring abundant labelled defective samples, which are not easily generated in sufficient numbers. To address this data scarcity problem, we introduce an autoencoder-based approach compatible with small datasets. Fortunately, the problem from a foundational point of view can be simplified as a binary classification between normal and abnormal samples. The proposed approach uses a depth map of the fibre layup surface, split into small windows aligned to each composite strip (tow). A subset of these windows that do not contain anomalies is passed to an autoencoder to reconstruct the input. Because the autoencoder is trained with normal samples, it produces more accurate reconstructions for these samples than for abnormal ones. Therefore, the value of reconstruction error is used as a quantitative metric for whether there are potential anomalies. These values are combined to produce an anomaly map, which can localize the manufacturing defects in the depth map. The results show that although the autoencoder is trained with a very limited number of scans, the proposed approach can produce sufficient binary classification accuracy and specify the location of the defects.

{{</citation>}}


### (24/46) Handwritten and Printed Text Segmentation: A Signature Case Study (Sina Gholamian et al., 2023)

{{<citation>}}

Sina Gholamian, Ali Vahdat. (2023)  
**Handwritten and Printed Text Segmentation: A Signature Case Study**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: NLP, OCR, Text Segmentation  
[Paper Link](http://arxiv.org/abs/2307.07887v1)  

---


**ABSTRACT**  
While analyzing scanned documents, handwritten text can overlay printed text. This causes difficulties during the optical character recognition (OCR) and digitization process of documents, and subsequently, hurts downstream NLP tasks. Prior research either focuses only on the binary classification of handwritten text, or performs a three-class segmentation of the document, i.e., recognition of handwritten, printed, and background pixels. This results in the assignment of the handwritten and printed overlapping pixels to only one of the classes, and thus, they are not accounted for in the other class. Thus, in this research, we develop novel approaches for addressing the challenges of handwritten and printed text segmentation with the goal of recovering text in different classes in whole, especially improving the segmentation performance on the overlapping parts. As such, to facilitate with this task, we introduce a new dataset, SignaTR6K, collected from real legal documents, as well as a new model architecture for handwritten and printed text segmentation task. Our best configuration outperforms the prior work on two different datasets by 17.9% and 7.3% on IoU scores.

{{</citation>}}


### (25/46) TinyTracker: Ultra-Fast and Ultra-Low-Power Edge Vision In-Sensor for Gaze Estimation (Pietro Bonazzi et al., 2023)

{{<citation>}}

Pietro Bonazzi, Thomas Ruegg, Sizhen Bian, Yawei Li, Michele Magno. (2023)  
**TinyTracker: Ultra-Fast and Ultra-Low-Power Edge Vision In-Sensor for Gaze Estimation**  

---
Primary Category: cs.CV
Categories: cs-CV, cs-RO, cs.CV  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2307.07813v3)  

---


**ABSTRACT**  
Intelligent edge vision tasks encounter the critical challenge of ensuring power and latency efficiency due to the typically heavy computational load they impose on edge platforms.This work leverages one of the first "AI in sensor" vision platforms, IMX500 by Sony, to achieve ultra-fast and ultra-low-power end-to-end edge vision applications. We evaluate the IMX500 and compare it to other edge platforms, such as the Google Coral Dev Micro and Sony Spresense, by exploring gaze estimation as a case study. We propose TinyTracker, a highly efficient, fully quantized model for 2D gaze estimation designed to maximize the performance of the edge vision systems considered in this study. TinyTracker achieves a 41x size reduction (600Kb) compared to iTracker [1] without significant loss in gaze estimation accuracy (maximum of 0.16 cm when fully quantized). TinyTracker's deployment on the Sony IMX500 vision sensor results in end-to-end latency of around 19ms. The camera takes around 17.9ms to read, process and transmit the pixels to the accelerator. The inference time of the network is 0.86ms with an additional 0.24 ms for retrieving the results from the sensor. The overall energy consumption of the end-to-end system is 4.9 mJ, including 0.06 mJ for inference. The end-to-end study shows that IMX500 is 1.7x faster than CoralMicro (19ms vs 34.4ms) and 7x more power efficient (4.9mJ VS 34.2mJ)

{{</citation>}}


### (26/46) Multiscale Memory Comparator Transformer for Few-Shot Video Segmentation (Mennatullah Siam et al., 2023)

{{<citation>}}

Mennatullah Siam, Rezaul Karim, He Zhao, Richard Wildes. (2023)  
**Multiscale Memory Comparator Transformer for Few-Shot Video Segmentation**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Transformer  
[Paper Link](http://arxiv.org/abs/2307.07812v1)  

---


**ABSTRACT**  
Few-shot video segmentation is the task of delineating a specific novel class in a query video using few labelled support images. Typical approaches compare support and query features while limiting comparisons to a single feature layer and thereby ignore potentially valuable information. We present a meta-learned Multiscale Memory Comparator (MMC) for few-shot video segmentation that combines information across scales within a transformer decoder. Typical multiscale transformer decoders for segmentation tasks learn a compressed representation, their queries, through information exchange across scales. Unlike previous work, we instead preserve the detailed feature maps during across scale information exchange via a multiscale memory transformer decoding to reduce confusion between the background and novel class. Integral to the approach, we investigate multiple forms of information exchange across scales in different tasks and provide insights with empirical evidence on which to use in each task. The overall comparisons among query and support features benefit from both rich semantics and precise localization. We demonstrate our approach primarily on few-shot video object segmentation and an adapted version on the fully supervised counterpart. In all cases, our approach outperforms the baseline and yields state-of-the-art performance. Our code is publicly available at https://github.com/MSiam/MMC-MultiscaleMemory.

{{</citation>}}


### (27/46) Joint Adversarial and Collaborative Learning for Self-Supervised Action Recognition (Tianyu Guo et al., 2023)

{{<citation>}}

Tianyu Guo, Mengyuan Liu, Hong Liu, Wenhao Li, Jingwen Guo, Tao Wang, Yidi Li. (2023)  
**Joint Adversarial and Collaborative Learning for Self-Supervised Action Recognition**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.07791v1)  

---


**ABSTRACT**  
Considering the instance-level discriminative ability, contrastive learning methods, including MoCo and SimCLR, have been adapted from the original image representation learning task to solve the self-supervised skeleton-based action recognition task. These methods usually use multiple data streams (i.e., joint, motion, and bone) for ensemble learning, meanwhile, how to construct a discriminative feature space within a single stream and effectively aggregate the information from multiple streams remains an open problem. To this end, we first apply a new contrastive learning method called BYOL to learn from skeleton data and formulate SkeletonBYOL as a simple yet effective baseline for self-supervised skeleton-based action recognition. Inspired by SkeletonBYOL, we further present a joint Adversarial and Collaborative Learning (ACL) framework, which combines Cross-Model Adversarial Learning (CMAL) and Cross-Stream Collaborative Learning (CSCL). Specifically, CMAL learns single-stream representation by cross-model adversarial loss to obtain more discriminative features. To aggregate and interact with multi-stream information, CSCL is designed by generating similarity pseudo label of ensemble learning as supervision and guiding feature generation for individual streams. Exhaustive experiments on three datasets verify the complementary properties between CMAL and CSCL and also verify that our method can perform favorably against state-of-the-art methods using various evaluation protocols. Our code and models are publicly available at \url{https://github.com/Levigty/ACL}.

{{</citation>}}


### (28/46) SoccerKDNet: A Knowledge Distillation Framework for Action Recognition in Soccer Videos (Sarosij Bose et al., 2023)

{{<citation>}}

Sarosij Bose, Saikat Sarkar, Amlan Chakrabarti. (2023)  
**SoccerKDNet: A Knowledge Distillation Framework for Action Recognition in Soccer Videos**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2307.07768v1)  

---


**ABSTRACT**  
Classifying player actions from soccer videos is a challenging problem, which has become increasingly important in sports analytics over the years. Most state-of-the-art methods employ highly complex offline networks, which makes it difficult to deploy such models in resource constrained scenarios. Here, in this paper we propose a novel end-to-end knowledge distillation based transfer learning network pre-trained on the Kinetics400 dataset and then perform extensive analysis on the learned framework by introducing a unique loss parameterization. We also introduce a new dataset named SoccerDB1 containing 448 videos and consisting of 4 diverse classes each of players playing soccer. Furthermore, we introduce an unique loss parameter that help us linearly weigh the extent to which the predictions of each network are utilized. Finally, we also perform a thorough performance study using various changed hyperparameters. We also benchmark the first classification results on the new SoccerDB1 dataset obtaining 67.20% validation accuracy. Apart from outperforming prior arts significantly, our model also generalizes to new datasets easily. The dataset has been made publicly available at: https://bit.ly/soccerdb1

{{</citation>}}


### (29/46) SINC: Self-Supervised In-Context Learning for Vision-Language Tasks (Yi-Syuan Chen et al., 2023)

{{<citation>}}

Yi-Syuan Chen, Yun-Zhu Song, Cheng Yu Yeo, Bei Liu, Jianlong Fu, Hong-Han Shuai. (2023)  
**SINC: Self-Supervised In-Context Learning for Vision-Language Tasks**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Self-Supervised, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.07742v1)  

---


**ABSTRACT**  
Large Pre-trained Transformers exhibit an intriguing capacity for in-context learning. Without gradient updates, these models can rapidly construct new predictors from demonstrations presented in the inputs. Recent works promote this ability in the vision-language domain by incorporating visual information into large language models that can already make in-context predictions. However, these methods could inherit issues in the language domain, such as template sensitivity and hallucination. Also, the scale of these language models raises a significant demand for computations, making learning and operating these models resource-intensive. To this end, we raise a question: ``How can we enable in-context learning for general models without being constrained on large language models?". To answer it, we propose a succinct and general framework, Self-supervised IN-Context learning (SINC), that introduces a meta-model to learn on self-supervised prompts consisting of tailored demonstrations. The learned models can be transferred to downstream tasks for making in-context predictions on-the-fly. Extensive experiments show that SINC outperforms gradient-based methods in various vision-language tasks under few-shot settings. Furthermore, the designs of SINC help us investigate the benefits of in-context learning across different tasks, and the analysis further reveals the essential components for the emergence of in-context learning in the vision-language domain.

{{</citation>}}


### (30/46) PSGformer: Enhancing 3D Point Cloud Instance Segmentation via Precise Semantic Guidance (Lei Pan et al., 2023)

{{<citation>}}

Lei Pan, Wuyang Luan, Yuan Zheng, Qiang Fu, Junhui Li. (2023)  
**PSGformer: Enhancing 3D Point Cloud Instance Segmentation via Precise Semantic Guidance**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.07708v1)  

---


**ABSTRACT**  
Most existing 3D instance segmentation methods are derived from 3D semantic segmentation models. However, these indirect approaches suffer from certain limitations. They fail to fully leverage global and local semantic information for accurate prediction, which hampers the overall performance of the 3D instance segmentation framework. To address these issues, this paper presents PSGformer, a novel 3D instance segmentation network. PSGformer incorporates two key advancements to enhance the performance of 3D instance segmentation. Firstly, we propose a Multi-Level Semantic Aggregation Module, which effectively captures scene features by employing foreground point filtering and multi-radius aggregation. This module enables the acquisition of more detailed semantic information from global and local perspectives. Secondly, PSGformer introduces a Parallel Feature Fusion Transformer Module that independently processes super-point features and aggregated features using transformers. The model achieves a more comprehensive feature representation by the features which connect global and local features. We conducted extensive experiments on the ScanNetv2 dataset. Notably, PSGformer exceeds compared state-of-the-art methods by 2.2% on ScanNetv2 hidden test set in terms of mAP. Our code and models will be publicly released.

{{</citation>}}


### (31/46) Both Spatial and Frequency Cues Contribute to High-Fidelity Image Inpainting (Ze Lu et al., 2023)

{{<citation>}}

Ze Lu, Yalei Lv, Wenqi Wang, Pengfei Xiong. (2023)  
**Both Spatial and Frequency Cues Contribute to High-Fidelity Image Inpainting**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.07678v1)  

---


**ABSTRACT**  
Deep generative approaches have obtained great success in image inpainting recently. However, most generative inpainting networks suffer from either over-smooth results or aliasing artifacts. The former lacks high-frequency details, while the latter lacks semantic structure. To address this issue, we propose an effective Frequency-Spatial Complementary Network (FSCN) by exploiting rich semantic information in both spatial and frequency domains. Specifically, we introduce an extra Frequency Branch and Frequency Loss on the spatial-based network to impose direct supervision on the frequency information, and propose a Frequency-Spatial Cross-Attention Block (FSCAB) to fuse multi-domain features and combine the corresponding characteristics. With our FSCAB, the inpainting network is capable of capturing frequency information and preserving visual consistency simultaneously. Extensive quantitative and qualitative experiments demonstrate that our inpainting network can effectively achieve superior results, outperforming previous state-of-the-art approaches with significantly fewer parameters and less computation cost. The code will be released soon.

{{</citation>}}


## cs.AI (5)



### (32/46) The SocialAI School: Insights from Developmental Psychology Towards Artificial Socio-Cultural Agents (Grgur Kovač et al., 2023)

{{<citation>}}

Grgur Kovač, Rémy Portelas, Peter Ford Dominey, Pierre-Yves Oudeyer. (2023)  
**The SocialAI School: Insights from Developmental Psychology Towards Artificial Socio-Cultural Agents**  

---
Primary Category: cs.AI
Categories: 68T07, I-2-0, cs-AI, cs-LG, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2307.07871v1)  

---


**ABSTRACT**  
Developmental psychologists have long-established the importance of socio-cognitive abilities in human intelligence. These abilities enable us to enter, participate and benefit from human culture. AI research on social interactive agents mostly concerns the emergence of culture in a multi-agent setting (often without a strong grounding in developmental psychology). We argue that AI research should be informed by psychology and study socio-cognitive abilities enabling to enter a culture too. We discuss the theories of Michael Tomasello and Jerome Bruner to introduce some of their concepts to AI and outline key concepts and socio-cognitive abilities. We present The SocialAI school - a tool including a customizable parameterized uite of procedurally generated environments, which simplifies conducting experiments regarding those concepts. We show examples of such experiments with RL agents and Large Language Models. The main motivation of this work is to engage the AI community around the problem of social intelligence informed by developmental psychology, and to provide a tool to simplify first steps in this direction. Refer to the project website for code and additional information: https://sites.google.com/view/socialai-school.

{{</citation>}}


### (33/46) Automated Knowledge Modeling for Cancer Clinical Practice Guidelines (Pralaypati Ta et al., 2023)

{{<citation>}}

Pralaypati Ta, Bhumika Gupta, Arihant Jain, Sneha Sree C, Arunima Sarkar, Keerthi Ram, Mohanasankar Sivaprakasam. (2023)  
**Automated Knowledge Modeling for Cancer Clinical Practice Guidelines**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2307.10231v1)  

---


**ABSTRACT**  
Clinical Practice Guidelines (CPGs) for cancer diseases evolve rapidly due to new evidence generated by active research. Currently, CPGs are primarily published in a document format that is ill-suited for managing this developing knowledge. A knowledge model of the guidelines document suitable for programmatic interaction is required. This work proposes an automated method for extraction of knowledge from National Comprehensive Cancer Network (NCCN) CPGs in Oncology and generating a structured model containing the retrieved knowledge. The proposed method was tested using two versions of NCCN Non-Small Cell Lung Cancer (NSCLC) CPG to demonstrate the effectiveness in faithful extraction and modeling of knowledge. Three enrichment strategies using Cancer staging information, Unified Medical Language System (UMLS) Metathesaurus & National Cancer Institute thesaurus (NCIt) concepts, and Node classification are also presented to enhance the model towards enabling programmatic traversal and querying of cancer care guidelines. The Node classification was performed using a Support Vector Machine (SVM) model, achieving a classification accuracy of 0.81 with 10-fold cross-validation.

{{</citation>}}


### (34/46) Explainable AI with counterfactual paths (Bastian Pfeifer et al., 2023)

{{<citation>}}

Bastian Pfeifer, Mateusz Krzyzinski, Hubert Baniecki, Anna Saranti, Andreas Holzinger, Przemyslaw Biecek. (2023)  
**Explainable AI with counterfactual paths**  

---
Primary Category: cs.AI
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07764v1)  

---


**ABSTRACT**  
Explainable AI (XAI) is an increasingly important area of research in machine learning, which in principle aims to make black-box models transparent and interpretable. In this paper, we propose a novel approach to XAI that uses counterfactual paths generated by conditional permutations. Our method provides counterfactual explanations by identifying alternative paths that could have led to different outcomes. The proposed method is particularly suitable for generating explanations based on counterfactual paths in knowledge graphs. By examining hypothetical changes to the input data in the knowledge graph, we can systematically validate the behaviour of the model and examine the features or combination of features that are most important to the model's predictions. Our approach provides a more intuitive and interpretable explanation for the model's behaviour than traditional feature weighting methods and can help identify and mitigate biases in the model.

{{</citation>}}


### (35/46) RL-ViGen: A Reinforcement Learning Benchmark for Visual Generalization (Zhecheng Yuan et al., 2023)

{{<citation>}}

Zhecheng Yuan, Sizhe Yang, Pu Hua, Can Chang, Kaizhe Hu, Xiaolong Wang, Huazhe Xu. (2023)  
**RL-ViGen: A Reinforcement Learning Benchmark for Visual Generalization**  

---
Primary Category: cs.AI
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.10224v1)  

---


**ABSTRACT**  
Visual Reinforcement Learning (Visual RL), coupled with high-dimensional observations, has consistently confronted the long-standing challenge of generalization. Despite the focus on algorithms aimed at resolving visual generalization problems, we argue that the devil is in the existing benchmarks as they are restricted to isolated tasks and generalization categories, undermining a comprehensive evaluation of agents' visual generalization capabilities. To bridge this gap, we introduce RL-ViGen: a novel Reinforcement Learning Benchmark for Visual Generalization, which contains diverse tasks and a wide spectrum of generalization types, thereby facilitating the derivation of more reliable conclusions. Furthermore, RL-ViGen incorporates the latest generalization visual RL algorithms into a unified framework, under which the experiment results indicate that no single existing algorithm has prevailed universally across tasks. Our aspiration is that RL-ViGen will serve as a catalyst in this area, and lay a foundation for the future creation of universal visual generalization RL agents suitable for real-world scenarios. Access to our code and implemented algorithms is provided at https://gemcollector.github.io/RL-ViGen/.

{{</citation>}}


### (36/46) Leveraging Large Language Models to Generate Answer Set Programs (Adam Ishay et al., 2023)

{{<citation>}}

Adam Ishay, Zhun Yang, Joohyung Lee. (2023)  
**Leveraging Large Language Models to Generate Answer Set Programs**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-CL, cs-SC, cs.AI  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.07699v1)  

---


**ABSTRACT**  
Large language models (LLMs), such as GPT-3 and GPT-4, have demonstrated exceptional performance in various natural language processing tasks and have shown the ability to solve certain reasoning problems. However, their reasoning capabilities are limited and relatively shallow, despite the application of various prompting techniques. In contrast, formal logic is adept at handling complex reasoning, but translating natural language descriptions into formal logic is a challenging task that non-experts struggle with. This paper proposes a neuro-symbolic method that combines the strengths of large language models and answer set programming. Specifically, we employ an LLM to transform natural language descriptions of logic puzzles into answer set programs. We carefully design prompts for an LLM to convert natural language descriptions into answer set programs in a step by step manner. Surprisingly, with just a few in-context learning examples, LLMs can generate reasonably complex answer set programs. The majority of errors made are relatively simple and can be easily corrected by humans, thus enabling LLMs to effectively assist in the creation of answer set programs.

{{</citation>}}


## q-fin.ST (1)



### (37/46) Contrasting the efficiency of stock price prediction models using various types of LSTM models aided with sentiment analysis (Varun Sangwan et al., 2023)

{{<citation>}}

Varun Sangwan, Vishesh Kumar Singh, Bibin Christopher V. (2023)  
**Contrasting the efficiency of stock price prediction models using various types of LSTM models aided with sentiment analysis**  

---
Primary Category: q-fin.ST
Categories: cs-LG, q-fin-ST, q-fin.ST  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.07868v1)  

---


**ABSTRACT**  
Our research aims to find the best model that uses companies projections and sector performances and how the given company fares accordingly to correctly predict equity share prices for both short and long term goals.

{{</citation>}}


## cs.SE (3)



### (38/46) Multilingual Adapter-based Knowledge Aggregation on Code Summarization for Low-Resource Languages (Iman Saberi et al., 2023)

{{<citation>}}

Iman Saberi, Fatemeh Fard, Fuxiang Chen. (2023)  
**Multilingual Adapter-based Knowledge Aggregation on Code Summarization for Low-Resource Languages**  

---
Primary Category: cs.SE
Categories: 68N30, 68T35, D-2-0; I-2-5, cs-SE, cs.SE  
Keywords: Language Model, Low-Resource, Multilingual, Summarization  
[Paper Link](http://arxiv.org/abs/2307.07854v1)  

---


**ABSTRACT**  
Multilingual fine-tuning (of a multilingual Pre-trained Language Model) has shown to improve performance of downstream tasks. However, it was observed that different programming languages may have different structural properties, and thus the learning or fine-tuning of a model may be sub-optimal or even degrade the intended performance by using a multilingual dataset. In this study, we proposed a new modular component architecture, AdvFusion, that leverages the different aspects of programming languages for a target popular low-resource programming language, Ruby. Our result shows that AdvFusion can extract useful features from different programming languages efficiently, and it outperforms the existing state-of-the-art multilingual fine-tuning by 12% on the Code Summarization task.

{{</citation>}}


### (39/46) AIOptimizer -- A reinforcement learning-based software performance optimisation prototype for cost minimisation (Noopur Zambare, 2023)

{{<citation>}}

Noopur Zambare. (2023)  
**AIOptimizer -- A reinforcement learning-based software performance optimisation prototype for cost minimisation**  

---
Primary Category: cs.SE
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07846v1)  

---


**ABSTRACT**  
This research article introduces AIOptimizer, a prototype for a software performance optimisation tool based on cost reduction. AIOptimizer uses a recommendation system driven by reinforcement learning to improve software system efficiency and affordability. The paper highlights AIOptimizer's design factors, such as accuracy, adaptability, scalability, and user-friendliness. To provide effective and user-centric performance optimisation solutions, it emphasises the use of a modular design, data gathering techniques, continuous learning, and resilient integration. The article also investigates AIOptimizer features such as fault identification, cost optimisation recommendations, efficiency prediction, and cooperation. Furthermore, it explores several software development life cycle models and introduces AIOptimizer uses a reinforcement learning-based recommendation engine for cost optimisation. The purpose of this research study is to highlight AIOptimizer as a prototype that uses advanced optimisation techniques and smart recommendation systems to continually enhance software performance and save expenses. The research focuses on various software development life cycle models, such as the Waterfall model, Iterative model, Spiral model, V-Model, Big Bang model and Agile Model. Each model has advantages and disadvantages, and their usefulness is determined by the project's specifications and characteristics. The AIOptimizer tool is a theoretical prototype for such software performance optimizers.

{{</citation>}}


### (40/46) Creating a Dataset Supporting Translation Between OpenMP Fortran and C++ Code (Bin Lei et al., 2023)

{{<citation>}}

Bin Lei, Caiwen Ding, Le Chen, Pei-Hung Lin, Chunhua Liao. (2023)  
**Creating a Dataset Supporting Translation Between OpenMP Fortran and C++ Code**  

---
Primary Category: cs.SE
Categories: cs-AI, cs-LG, cs-SE, cs.SE  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2307.07686v1)  

---


**ABSTRACT**  
In this study, we present a novel dataset for training machine learning models translating between OpenMP Fortran and C++ code. To ensure reliability and applicability, the dataset is initially refined using a meticulous code similarity test. The effectiveness of our dataset is assessed using both quantitative (CodeBLEU) and qualitative (human evaluation) methods. We demonstrate how this dataset can significantly improve the translation capabilities of large-scale language models, with improvements of \times 5.1 for models with no prior coding knowledge and \times 9.9 for models with some coding familiarity. Our work highlights the potential of this dataset to advance the field of code translation for high-performance computing.

{{</citation>}}


## cs.RO (1)



### (41/46) Can Pre-Trained Text-to-Image Models Generate Visual Goals for Reinforcement Learning? (Jialu Gao et al., 2023)

{{<citation>}}

Jialu Gao, Kaizhe Hu, Guowei Xu, Huazhe Xu. (2023)  
**Can Pre-Trained Text-to-Image Models Generate Visual Goals for Reinforcement Learning?**  

---
Primary Category: cs.RO
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.07837v1)  

---


**ABSTRACT**  
Pre-trained text-to-image generative models can produce diverse, semantically rich, and realistic images from natural language descriptions. Compared with language, images usually convey information with more details and less ambiguity. In this study, we propose Learning from the Void (LfVoid), a method that leverages the power of pre-trained text-to-image models and advanced image editing techniques to guide robot learning. Given natural language instructions, LfVoid can edit the original observations to obtain goal images, such as "wiping" a stain off a table. Subsequently, LfVoid trains an ensembled goal discriminator on the generated image to provide reward signals for a reinforcement learning agent, guiding it to achieve the goal. The ability of LfVoid to learn with zero in-domain training on expert demonstrations or true goal observations (the void) is attributed to the utilization of knowledge from web-scale generative models. We evaluate LfVoid across three simulated tasks and validate its feasibility in the corresponding real-world scenarios. In addition, we offer insights into the key considerations for the effective integration of visual generative models into robot learning workflows. We posit that our work represents an initial step towards the broader application of pre-trained visual generative models in the robotics field. Our project page: https://lfvoid-rl.github.io/.

{{</citation>}}


## cs.NE (1)



### (42/46) Generative Meta-Learning Robust Quality-Diversity Portfolio (Kamer Ali Yuksel, 2023)

{{<citation>}}

Kamer Ali Yuksel. (2023)  
**Generative Meta-Learning Robust Quality-Diversity Portfolio**  

---
Primary Category: cs.NE
Categories: cs-NE, cs.NE, q-fin-PM  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.07811v1)  

---


**ABSTRACT**  
This paper proposes a novel meta-learning approach to optimize a robust portfolio ensemble. The method uses a deep generative model to generate diverse and high-quality sub-portfolios combined to form the ensemble portfolio. The generative model consists of a convolutional layer, a stateful LSTM module, and a dense network. During training, the model takes a randomly sampled batch of Gaussian noise and outputs a population of solutions, which are then evaluated using the objective function of the problem. The weights of the model are updated using a gradient-based optimizer. The convolutional layer transforms the noise into a desired distribution in latent space, while the LSTM module adds dependence between generations. The dense network decodes the population of solutions. The proposed method balances maximizing the performance of the sub-portfolios with minimizing their maximum correlation, resulting in a robust ensemble portfolio against systematic shocks. The approach was effective in experiments where stochastic rewards were present. Moreover, the results (Fig. 1) demonstrated that the ensemble portfolio obtained by taking the average of the generated sub-portfolio weights was robust and generalized well. The proposed method can be applied to problems where diversity is desired among co-optimized solutions for a robust ensemble. The source-codes and the dataset are in the supplementary material.

{{</citation>}}


## cs.IR (2)



### (43/46) Prompt Tuning on Graph-augmented Low-resource Text Classification (Zhihao Wen et al., 2023)

{{<citation>}}

Zhihao Wen, Yuan Fang. (2023)  
**Prompt Tuning on Graph-augmented Low-resource Text Classification**  

---
Primary Category: cs.IR
Categories: cs-IR, cs.IR  
Keywords: Text Classification  
[Paper Link](http://arxiv.org/abs/2307.10230v1)  

---


**ABSTRACT**  
Text classification is a fundamental problem in information retrieval with many real-world applications, such as predicting the topics of online articles and the categories of e-commerce product descriptions. However, low-resource text classification, with no or few labeled samples, presents a serious concern for supervised learning. Meanwhile, many text data are inherently grounded on a network structure, such as a hyperlink/citation network for online articles, and a user-item purchase network for e-commerce products. These graph structures capture rich semantic relationships, which can potentially augment low-resource text classification. In this paper, we propose a novel model called Graph-Grounded Pre-training and Prompting (G2P2) to address low-resource text classification in a two-pronged approach. During pre-training, we propose three graph interaction-based contrastive strategies to jointly pre-train a graph-text model; during downstream classification, we explore handcrafted discrete prompts and continuous prompt tuning for the jointly pre-trained model to achieve zero- and few-shot classification, respectively. Besides, for generalizing continuous prompts to unseen classes, we propose conditional prompt tuning on graphs (G2P2$^*$). Extensive experiments on four real-world datasets demonstrate the strength of G2P2 in zero- and few-shot low-resource text classification tasks, and illustrate the advantage of G2P2$^*$ in dealing with unseen classes.

{{</citation>}}


### (44/46) Intuitive Access to Smartphone Settings Using Relevance Model Trained by Contrastive Learning (Joonyoung Kim et al., 2023)

{{<citation>}}

Joonyoung Kim, Kangwook Lee, Haebin Shin, Hurnjoo Lee, Sechun Kang, Byunguk Choi, Dong Shin, Joohyung Lee. (2023)  
**Intuitive Access to Smartphone Settings Using Relevance Model Trained by Contrastive Learning**  

---
Primary Category: cs.IR
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.09177v1)  

---


**ABSTRACT**  
The more new features that are being added to smartphones, the harder it becomes for users to find them. This is because the feature names are usually short, and there are just too many to remember. In such a case, the users may want to ask contextual queries that describe the features they are looking for, but the standard term frequency-based search cannot process them. This paper presents a novel retrieval system for mobile features that accepts intuitive and contextual search queries. We trained a relevance model via contrastive learning from a pre-trained language model to perceive the contextual relevance between query embeddings and indexed mobile features. Also, to make it run efficiently on-device using minimal resources, we applied knowledge distillation to compress the model without degrading much performance. To verify the feasibility of our method, we collected test queries and conducted comparative experiments with the currently deployed search baselines. The results show that our system outperforms the others on contextual sentence queries and even on usual keyword-based queries.

{{</citation>}}


## cs.CE (1)



### (45/46) Evaluation of Deep Reinforcement Learning Algorithms for Portfolio Optimisation (Chung I Lu, 2023)

{{<citation>}}

Chung I Lu. (2023)  
**Evaluation of Deep Reinforcement Learning Algorithms for Portfolio Optimisation**  

---
Primary Category: cs.CE
Categories: cs-CE, cs.CE, q-fin-PM  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.07694v1)  

---


**ABSTRACT**  
We evaluate benchmark deep reinforcement learning (DRL) algorithms on the task of portfolio optimisation under a simulator. The simulator is based on correlated geometric Brownian motion (GBM) with the Bertsimas-Lo (BL) market impact model. Using the Kelly criterion (log utility) as the objective, we can analytically derive the optimal policy without market impact and use it as an upper bound to measure performance when including market impact. We found that the off-policy algorithms DDPG, TD3 and SAC were unable to learn the right Q function due to the noisy rewards and therefore perform poorly. The on-policy algorithms PPO and A2C, with the use of generalised advantage estimation (GAE), were able to deal with the noise and derive a close to optimal policy. The clipping variant of PPO was found to be important in preventing the policy from deviating from the optimal once converged. In a more challenging environment where we have regime changes in the GBM parameters, we found that PPO, combined with a hidden Markov model (HMM) to learn and predict the regime context, is able to learn different policies adapted to each regime. Overall, we find that the sample complexity of these algorithms is too high, requiring more than 2m steps to learn a good policy in the simplest setting, which is equivalent to almost 8,000 years of daily prices.

{{</citation>}}


## cs.CR (1)



### (46/46) Saudi Arabian Perspective of Security, Privacy, and Attitude of Using Facial Recognition Technology (Amani Mohammed Alqarni et al., 2023)

{{<citation>}}

Amani Mohammed Alqarni, Daniel Timko, Muhammad Lutfor Rahman. (2023)  
**Saudi Arabian Perspective of Security, Privacy, and Attitude of Using Facial Recognition Technology**  

---
Primary Category: cs.CR
Categories: cs-CR, cs.CR  
Keywords: AI, Security  
[Paper Link](http://arxiv.org/abs/2307.07671v1)  

---


**ABSTRACT**  
Facial Recognition Technology (FRT) is a pioneering field of mass surveillance that sparks privacy concerns and is considered a growing threat in the modern world. FRT has been widely adopted in the Kingdom of Saudi Arabia to improve public services and surveillance. Accordingly, the following study aims to understand the privacy and security concerns, trust, and acceptance of FRT in Saudi Arabia. Validated Privacy Concerns (IUIPC-8), Security Attitudes (SA-6), and Security Behavior (SeBIS) scales are used along with replicate studies from Pew Research Center trust questions and government trust questions. In addition, we examine potential differences between Saudis and Americans. To gain insights into these concerns, we conducted an online survey involving 53 Saudi Arabia citizens who are residing in the USA. We have collected data in the US instead of Saudi Arabia to avoid the regulatory challenges of the Saudi Data & Artificial Intelligence Authority (SDAIA). Responses from closed-ended questions revealed that Saudis score much lower than Americans when it comes to security attitudes, whereas they score lower when it comes to privacy concerns. We found no significant difference between Saudis' and Americans' acceptance of the use of FRT in different scenarios, but we found that Saudis trust advertisers more than Americans. Additionally, Saudis are more likely than Americans to agree that the government should strictly limit the use of FRT.

{{</citation>}}
