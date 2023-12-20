---
draft: false
title: "arXiv @ 2023.12.20"
date: 2023-12-20
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.20"
    identifier: arxiv_20231220
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.AI (15)](#csai-15)
- [cs.GT (1)](#csgt-1)
- [cs.AR (1)](#csar-1)
- [cs.LG (20)](#cslg-20)
- [cs.DC (1)](#csdc-1)
- [cs.CL (27)](#cscl-27)
- [cs.SE (4)](#csse-4)
- [cs.CV (25)](#cscv-25)
- [cs.RO (5)](#csro-5)
- [eess.SY (3)](#eesssy-3)
- [cs.CR (3)](#cscr-3)
- [cs.HC (6)](#cshc-6)
- [math.OC (1)](#mathoc-1)
- [cs.MA (1)](#csma-1)
- [quant-ph (1)](#quant-ph-1)
- [cs.IR (5)](#csir-5)
- [cs.SI (3)](#cssi-3)
- [cs.DB (1)](#csdb-1)
- [eess.IV (2)](#eessiv-2)
- [math.NA (1)](#mathna-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.SD (5)](#cssd-5)
- [cond-mat.mtrl-sci (1)](#cond-matmtrl-sci-1)
- [cs.NE (1)](#csne-1)

## cs.AI (15)



### (1/134) Poker Hand History File Format Specification (Juho Kim, 2023)

{{<citation>}}

Juho Kim. (2023)  
**Poker Hand History File Format Specification**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11753v1)  

---


**ABSTRACT**  
This paper introduces the Poker Hand History (PHH) file format, designed to standardize the recording of poker hands across different game variants. Despite poker's widespread popularity in the mainstream culture as a mind sport and its prominence in the field of artificial intelligence (AI) research as a benchmark for imperfect information AI agents, it lacks a consistent format that humans can use to document poker hands across different variants that can also easily be parsed by machines. To address this gap in the literature, we propose the PHH format which provides a concise human-readable machine-friendly representation of hand history that comprehensively captures various details of the hand, ranging from initial game parameters and actions to contextual parameters including but not limited to the venue, players, and time control information. In the supplementary, we provide over 10,000 hands covering 11 different variants in the PHH format. Building on our previous work on PokerKit, a premier poker hand simulation tool, we demonstrate the usages of our open-source Python implementation of the PHH parser. The source code of the parser is available on GitHub: https://github.com/uoftcprg/pokerkit

{{</citation>}}


### (2/134) Human-Machine Teaming for UAVs: An Experimentation Platform (Laila El Moujtahid et al., 2023)

{{<citation>}}

Laila El Moujtahid, Sai Krishna Gottipati, Clodéric Mars, Matthew E. Taylor. (2023)  
**Human-Machine Teaming for UAVs: An Experimentation Platform**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-LG, cs-MA, cs.AI, stat-AP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11718v1)  

---


**ABSTRACT**  
Full automation is often not achievable or desirable in critical systems with high-stakes decisions. Instead, human-AI teams can achieve better results. To research, develop, evaluate, and validate algorithms suited for such teaming, lightweight experimentation platforms that enable interactions between humans and multiple AI agents are necessary. However, there are limited examples of such platforms for defense environments. To address this gap, we present the Cogment human-machine teaming experimentation platform, which implements human-machine teaming (HMT) use cases that features heterogeneous multi-agent systems and can involve learning AI agents, static AI agents, and humans. It is built on the Cogment platform and has been used for academic research, including work presented at the ALA workshop at AAMAS this year [1]. With this platform, we hope to facilitate further research on human-machine teaming in critical systems and defense environments.

{{</citation>}}


### (3/134) Agent-based Learning of Materials Datasets from Scientific Literature (Mehrad Ansari et al., 2023)

{{<citation>}}

Mehrad Ansari, Seyed Mohamad Moosavi. (2023)  
**Agent-based Learning of Materials Datasets from Scientific Literature**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11690v1)  

---


**ABSTRACT**  
Advancements in machine learning and artificial intelligence are transforming materials discovery. Yet, the availability of structured experimental data remains a bottleneck. The vast corpus of scientific literature presents a valuable and rich resource of such data. However, manual dataset creation from these resources is challenging due to issues in maintaining quality and consistency, scalability limitations, and the risk of human error and bias. Therefore, in this work, we develop a chemist AI agent, powered by large language models (LLMs), to overcome these challenges by autonomously creating structured datasets from natural language text, ranging from sentences and paragraphs to extensive scientific research articles. Our chemist AI agent, Eunomia, can plan and execute actions by leveraging the existing knowledge from decades of scientific research articles, scientists, the Internet and other tools altogether. We benchmark the performance of our approach in three different information extraction tasks with various levels of complexity, including solid-state impurity doping, metal-organic framework (MOF) chemical formula, and property relations. Our results demonstrate that our zero-shot agent, with the appropriate tools, is capable of attaining performance that is either superior or comparable to the state-of-the-art fine-tuned materials information extraction methods. This approach simplifies compilation of machine learning-ready datasets for various materials discovery applications, and significantly ease the accessibility of advanced natural language processing tools for novice users in natural language. The methodology in this work is developed as an open-source software on https://github.com/AI4ChemS/Eunomia.

{{</citation>}}


### (4/134) Bridging Logic and Learning: A Neural-Symbolic Approach for Enhanced Reasoning in Neural Models (ASPER) (Fadi Al Machot, 2023)

{{<citation>}}

Fadi Al Machot. (2023)  
**Bridging Logic and Learning: A Neural-Symbolic Approach for Enhanced Reasoning in Neural Models (ASPER)**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.11651v1)  

---


**ABSTRACT**  
Neural-symbolic learning, an intersection of neural networks and symbolic reasoning, aims to blend neural networks' learning capabilities with symbolic AI's interpretability and reasoning. This paper introduces an approach designed to improve the performance of neural models in learning reasoning tasks. It achieves this by integrating Answer Set Programming (ASP) solvers and domain-specific expertise, which is an approach that diverges from traditional complex neural-symbolic models. In this paper, a shallow artificial neural network (ANN) is specifically trained to solve Sudoku puzzles with minimal training data. The model has a unique loss function that integrates losses calculated using the ASP solver outputs, effectively enhancing its training efficiency. Most notably, the model shows a significant improvement in solving Sudoku puzzles using only 12 puzzles for training and testing without hyperparameter tuning. This advancement indicates that the model's enhanced reasoning capabilities have practical applications, extending well beyond Sudoku puzzles to potentially include a variety of other domains. The code can be found on GitHub: https://github.com/Fadi2200/ASPEN.

{{</citation>}}


### (5/134) Animal-AI 3: What's New & Why You Should Care (Konstantinos Voudouris et al., 2023)

{{<citation>}}

Konstantinos Voudouris, Ibrahim Alhas, Wout Schellaert, Matthew Crosby, Joel Holmes, John Burden, Niharika Chaubey, Niall Donnelly, Matishalin Patel, Marta Halina, José Hernández-Orallo, Lucy G. Cheke. (2023)  
**Animal-AI 3: What's New & Why You Should Care**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11414v1)  

---


**ABSTRACT**  
The Animal-AI Environment is a unique game-based research platform designed to serve both the artificial intelligence and cognitive science research communities. In this paper, we present Animal-AI 3, the latest version of the environment, outlining several major new features that make the game more engaging for humans and more complex for AI systems. New features include interactive buttons, reward dispensers, and player notifications, as well as an overhaul of the environment's graphics and processing for significant increases in agent training time and quality of the human player experience. We provide detailed guidance on how to build computational and behavioural experiments with Animal-AI 3. We present results from a series of agents, including the state-of-the-art Deep Reinforcement Learning agent (dreamer-v3), on newly designed tests and the Animal-AI Testbed of 900 tasks inspired by research in comparative psychology. Animal-AI 3 is designed to facilitate collaboration between the cognitive sciences and artificial intelligence. This paper serves as a stand-alone document that motivates, describes, and demonstrates Animal-AI 3 for the end user.

{{</citation>}}


### (6/134) Counting Reward Automata: Sample Efficient Reinforcement Learning Through the Exploitation of Reward Function Structure (Tristan Bester et al., 2023)

{{<citation>}}

Tristan Bester, Benjamin Rosman, Steven James, Geraud Nangue Tasse. (2023)  
**Counting Reward Automata: Sample Efficient Reinforcement Learning Through the Exploitation of Reward Function Structure**  

---
Primary Category: cs.AI  
Categories: I-2; F-4, cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11364v1)  

---


**ABSTRACT**  
We present counting reward automata-a finite state machine variant capable of modelling any reward function expressible as a formal language. Unlike previous approaches, which are limited to the expression of tasks as regular languages, our framework allows for tasks described by unrestricted grammars. We prove that an agent equipped with such an abstract machine is able to solve a larger set of tasks than those utilising current approaches. We show that this increase in expressive power does not come at the cost of increased automaton complexity. A selection of learning algorithms are presented which exploit automaton structure to improve sample efficiency. We show that the state machines required in our formulation can be specified from natural language task descriptions using large language models. Empirical results demonstrate that our method outperforms competing approaches in terms of sample efficiency, automaton complexity, and task completion.

{{</citation>}}


### (7/134) Towards Fairness in Online Service with k Servers and its Application on Fair Food Delivery (Daman Deep Singh et al., 2023)

{{<citation>}}

Daman Deep Singh, Amit Kumar, Abhijnan Chakraborty. (2023)  
**Towards Fairness in Online Service with k Servers and its Application on Fair Food Delivery**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11280v1)  

---


**ABSTRACT**  
The k-SERVER problem is one of the most prominent problems in online algorithms with several variants and extensions. However, simplifying assumptions like instantaneous server movements and zero service time has hitherto limited its applicability to real-world problems. In this paper, we introduce a realistic generalization of k-SERVER without such assumptions - the k-FOOD problem, where requests with source-destination locations and an associated pickup time window arrive in an online fashion, and each has to be served by exactly one of the available k servers. The k-FOOD problem offers the versatility to model a variety of real-world use cases such as food delivery, ride sharing, and quick commerce. Moreover, motivated by the need for fairness in online platforms, we introduce the FAIR k-FOOD problem with the max-min objective. We establish that both k-FOOD and FAIR k-FOOD problems are strongly NP-hard and develop an optimal offline algorithm that arises naturally from a time-expanded flow network. Subsequently, we propose an online algorithm DOC4FOOD involving virtual movements of servers to the nearest request location. Experiments on a real-world food-delivery dataset, alongside synthetic datasets, establish the efficacy of the proposed algorithm against state-of-the-art fair food delivery algorithms.

{{</citation>}}


### (8/134) CDRH Seeks Public Comment: Digital Health Technologies for Detecting Prediabetes and Undiagnosed Type 2 Diabetes (Manuel Cossio, 2023)

{{<citation>}}

Manuel Cossio. (2023)  
**CDRH Seeks Public Comment: Digital Health Technologies for Detecting Prediabetes and Undiagnosed Type 2 Diabetes**  

---
Primary Category: cs.AI  
Categories: A-m, cs-AI, cs-CY, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11226v1)  

---


**ABSTRACT**  
This document provides responses to the FDA's request for public comments (Docket No FDA 2023 N 4853) on the role of digital health technologies (DHTs) in detecting prediabetes and undiagnosed type 2 diabetes. It explores current DHT applications in prevention, detection, treatment and reversal of prediabetes, highlighting AI chatbots, online forums, wearables and mobile apps. The methods employed by DHTs to capture health signals like glucose, diet, symptoms and community insights are outlined. Key subpopulations that could benefit most from remote screening tools include rural residents, minority groups, high-risk individuals and those with limited healthcare access. Capturable high-impact risk factors encompass glycemic variability, cardiovascular parameters, respiratory health, blood biomarkers and patient reported symptoms. An array of non-invasive monitoring tools are discussed, although further research into their accuracy for diverse groups is warranted. Extensive health datasets providing immense opportunities for AI and ML based risk modeling are presented. Promising techniques leveraging EHRs, imaging, wearables and surveys to enhance screening through AI and ML algorithms are showcased. Analysis of social media and streaming data further allows disease prediction across populations. Ongoing innovation focused on inclusivity and accessibility is highlighted as pivotal in unlocking DHTs potential for transforming prediabetes and diabetes prevention and care.

{{</citation>}}


### (9/134) Learning Domain-Independent Heuristics for Grounded and Lifted Planning (Dillon Z. Chen et al., 2023)

{{<citation>}}

Dillon Z. Chen, Felipe Trevizan, Sylvie Thiebaux. (2023)  
**Learning Domain-Independent Heuristics for Grounded and Lifted Planning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.11143v1)  

---


**ABSTRACT**  
We present three novel graph representations of planning tasks suitable for learning domain-independent heuristics using Graph Neural Networks (GNNs) to guide search. In particular, to mitigate the issues caused by large grounded GNNs we present the first method for learning domain-independent heuristics with only the lifted representation of a planning task. We also provide a theoretical analysis of the expressiveness of our models, showing that some are more powerful than STRIPS-HGN, the only other existing model for learning domain-independent heuristics. Our experiments show that our heuristics generalise to much larger problems than those in the training set, vastly surpassing STRIPS-HGN heuristics.

{{</citation>}}


### (10/134) Explaining Reinforcement Learning Agents Through Counterfactual Action Outcomes (Yotam Amitai et al., 2023)

{{<citation>}}

Yotam Amitai, Yael Septon, Ofra Amir. (2023)  
**Explaining Reinforcement Learning Agents Through Counterfactual Action Outcomes**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11118v1)  

---


**ABSTRACT**  
Explainable reinforcement learning (XRL) methods aim to help elucidate agent policies and decision-making processes. The majority of XRL approaches focus on local explanations, seeking to shed light on the reasons an agent acts the way it does at a specific world state. While such explanations are both useful and necessary, they typically do not portray the outcomes of the agent's selected choice of action. In this work, we propose ``COViz'', a new local explanation method that visually compares the outcome of an agent's chosen action to a counterfactual one. In contrast to most local explanations that provide state-limited observations of the agent's motivation, our method depicts alternative trajectories the agent could have taken from the given state and their outcomes. We evaluated the usefulness of COViz in supporting people's understanding of agents' preferences and compare it with reward decomposition, a local explanation method that describes an agent's expected utility for different actions by decomposing it into meaningful reward types. Furthermore, we examine the complementary benefits of integrating both methods. Our results show that such integration significantly improved participants' performance.

{{</citation>}}


### (11/134) The Good, The Bad, and Why: Unveiling Emotions in Generative AI (Cheng Li et al., 2023)

{{<citation>}}

Cheng Li, Jindong Wang, Yixuan Zhang, Kaijie Zhu, Xinyi Wang, Wenxin Hou, Jianxun Lian, Fang Luo, Qiang Yang, Xing Xie. (2023)  
**The Good, The Bad, and Why: Unveiling Emotions in Generative AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-HC, cs.AI  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.11111v2)  

---


**ABSTRACT**  
Emotion significantly impacts our daily behaviors and interactions. While recent generative AI models, such as large language models, have shown impressive performance in various tasks, it remains unclear whether they truly comprehend emotions. This paper aims to address this gap by incorporating psychological theories to gain a holistic understanding of emotions in generative AI models. Specifically, we propose three approaches: 1) EmotionPrompt to enhance AI model performance, 2) EmotionAttack to impair AI model performance, and 3) EmotionDecode to explain the effects of emotional stimuli, both benign and malignant. Through extensive experiments involving language and multi-modal models on semantic understanding, logical reasoning, and generation tasks, we demonstrate that both textual and visual EmotionPrompt can boost the performance of AI models while EmotionAttack can hinder it. Additionally, EmotionDecode reveals that AI models can comprehend emotional stimuli akin to the mechanism of dopamine in the human brain. Our work heralds a novel avenue for exploring psychology to enhance our understanding of generative AI models. This paper is an extended version of our previous work EmotionPrompt (arXiv:2307.11760).

{{</citation>}}


### (12/134) Conflict Detection for Temporal Knowledge Graphs:A Fast Constraint Mining Algorithm and New Benchmarks (Jianhao Chen et al., 2023)

{{<citation>}}

Jianhao Chen, Junyang Ren, Wentao Ding, Haoyuan Ouyang, Wei Hu, Yuzhong Qu. (2023)  
**Conflict Detection for Temporal Knowledge Graphs:A Fast Constraint Mining Algorithm and New Benchmarks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-DB, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.11053v1)  

---


**ABSTRACT**  
Temporal facts, which are used to describe events that occur during specific time periods, have become a topic of increased interest in the field of knowledge graph (KG) research. In terms of quality management, the introduction of time restrictions brings new challenges to maintaining the temporal consistency of KGs. Previous studies rely on manually enumerated temporal constraints to detect conflicts, which are labor-intensive and may have granularity issues. To address this problem, we start from the common pattern of temporal facts and propose a pattern-based temporal constraint mining method, PaTeCon. Unlike previous studies, PaTeCon uses graph patterns and statistical information relevant to the given KG to automatically generate temporal constraints, without the need for human experts. In this paper, we illustrate how this method can be optimized to achieve significant speed improvement. We also annotate Wikidata and Freebase to build two new benchmarks for conflict detection. Extensive experiments demonstrate that our pattern-based automatic constraint mining approach is highly effective in generating valuable temporal constraints.

{{</citation>}}


### (13/134) Learning Top-k Subtask Planning Tree based on Discriminative Representation Pre-training for Decision Making (Jingqing Ruan et al., 2023)

{{<citation>}}

Jingqing Ruan, Kaishen Wang, Qingyang Zhang, Dengpeng Xing, Bo Xu. (2023)  
**Learning Top-k Subtask Planning Tree based on Discriminative Representation Pre-training for Decision Making**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11027v1)  

---


**ABSTRACT**  
Many complicated real-world tasks can be broken down into smaller, more manageable parts, and planning with prior knowledge extracted from these simplified pieces is crucial for humans to make accurate decisions. However, replicating this process remains a challenge for AI agents and naturally raises two questions: How to extract discriminative knowledge representation from priors? How to develop a rational plan to decompose complex problems? Most existing representation learning methods employing a single encoder structure are fragile and sensitive to complex and diverse dynamics. To address this issue, we introduce a multiple-encoder and individual-predictor regime to learn task-essential representations from sufficient data for simple subtasks. Multiple encoders can extract adequate task-relevant dynamics without confusion, and the shared predictor can discriminate the task characteristics. We also use the attention mechanism to generate a top-k subtask planning tree, which customizes subtask execution plans in guiding complex decisions on unseen tasks. This process enables forward-looking and globality by flexibly adjusting the depth and width of the planning tree. Empirical results on a challenging platform composed of some basic simple tasks and combinatorially rich synthetic tasks consistently outperform some competitive baselines and demonstrate the benefits of our design.

{{</citation>}}


### (14/134) Dynamic Retrieval Augmented Generation of Ontologies using Artificial Intelligence (DRAGON-AI) (Sabrina Toro et al., 2023)

{{<citation>}}

Sabrina Toro, Anna V Anagnostopoulos, Sue Bello, Kai Blumberg, Rhiannon Cameron, Leigh Carmody, Alexander D Diehl, Damion Dooley, William Duncan, Petra Fey, Pascale Gaudet, Nomi L Harris, Marcin Joachimiak, Leila Kiani, Tiago Lubiana, Monica C Munoz-Torres, Shawn O'Neil, David Osumi-Sutherland, Aleix Puig, Justin P Reese, Leonore Reiser, Sofia Robb, Troy Ruemping, James Seager, Eric Sid, Ray Stefancsik, Magalie Weber, Valerie Wood, Melissa A Haendel, Christopher J Mungall. (2023)  
**Dynamic Retrieval Augmented Generation of Ontologies using Artificial Intelligence (DRAGON-AI)**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.10904v1)  

---


**ABSTRACT**  
Ontologies are fundamental components of informatics infrastructure in domains such as biomedical, environmental, and food sciences, representing consensus knowledge in an accurate and computable form. However, their construction and maintenance demand substantial resources, necessitating substantial collaborative efforts of domain experts, curators, and ontology experts.   We present Dynamic Retrieval Augmented Generation of Ontologies using AI (DRAGON-AI), an ontology generation method employing Large Language Models (LLMs) and Retrieval Augmented Generation (RAG). This method can generate textual and logical ontology components, drawing from existing knowledge in multiple ontologies, as well as unstructured textual sources.   We assessed DRAGON-AI across ten diverse ontologies, making use of extensive manual evaluation of results. We demonstrate high precision for relationship generation, close to but lower than precision from logic-based reasoning. We also demonstrate definition generation comparable with but lower than human-generated definitions. Notably, expert evaluators were better able to discern subtle flaws in AI-generated definitions. We also demonstrated the ability of DRAGON-AI to incorporate natural language instructions in the form of GitHub issues.   These findings suggest DRAGON-AI's potential to substantially aid the manual ontology construction process. However, our results also underscore the importance of having expert curators and ontology editors drive the ontology generation process.

{{</citation>}}


### (15/134) From Google Gemini to OpenAI Q* (Q-Star): A Survey of Reshaping the Generative Artificial Intelligence (AI) Research Landscape (Timothy R. McIntosh et al., 2023)

{{<citation>}}

Timothy R. McIntosh, Teo Susnjak, Tong Liu, Paul Watters, Malka N. Halgamuge. (2023)  
**From Google Gemini to OpenAI Q* (Q-Star): A Survey of Reshaping the Generative Artificial Intelligence (AI) Research Landscape**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2312.10868v1)  

---


**ABSTRACT**  
This comprehensive survey explored the evolving landscape of generative Artificial Intelligence (AI), with a specific focus on the transformative impacts of Mixture of Experts (MoE), multimodal learning, and the speculated advancements towards Artificial General Intelligence (AGI). It critically examined the current state and future trajectory of generative Artificial Intelligence (AI), exploring how innovations like Google's Gemini and the anticipated OpenAI Q* project are reshaping research priorities and applications across various domains, including an impact analysis on the generative AI research taxonomy. It assessed the computational challenges, scalability, and real-world implications of these technologies while highlighting their potential in driving significant progress in fields like healthcare, finance, and education. It also addressed the emerging academic challenges posed by the proliferation of both AI-themed and AI-generated preprints, examining their impact on the peer-review process and scholarly communication. The study highlighted the importance of incorporating ethical and human-centric methods in AI development, ensuring alignment with societal norms and welfare, and outlined a strategy for future AI research that focuses on a balanced and conscientious use of MoE, multimodality, and AGI in generative AI.

{{</citation>}}


## cs.GT (1)



### (16/134) Equilibrium Computation in Multi-Stage Auctions and Contests (Fabian R. Pieroth et al., 2023)

{{<citation>}}

Fabian R. Pieroth, Nils Kohring, Martin Bichler. (2023)  
**Equilibrium Computation in Multi-Stage Auctions and Contests**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs.GT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11751v1)  

---


**ABSTRACT**  
We compute equilibrium strategies in multi-stage games with continuous signal and action spaces as they are widely used in the management sciences and economics. Examples include sequential sales via auctions, multi-stage elimination contests, and Stackelberg competitions. In sequential auctions, analysts are required to derive not just single bids but bid functions for all possible signals or values that a bidder might have in multiple stages. Due to the continuity of the signal and action spaces, these bid functions come from an infinite dimensional space. While such models are fundamental to game theory and its applications, equilibrium strategies are rarely known. The resulting system of non-linear differential equations is considered intractable for all but elementary models. This has been limiting progress in game theory and is a barrier to its adoption in the field. We show that Deep Reinforcement Learning and self-play can learn equilibrium bidding strategies for various multi-stage games without making parametric assumptions on the bid function. We find equilibrium in models that have not yet been explored analytically and new asymmetric equilibrium bid functions for established models of sequential auctions. The verification of equilibrium is challenging in such games due to the continuous signal and action spaces. We introduce a verification algorithm and prove that the error of this verifier decreases when considering Lipschitz continuous strategies with increasing levels of discretization and sample sizes.

{{</citation>}}


## cs.AR (1)



### (17/134) A Heterogeneous Chiplet Architecture for Accelerating End-to-End Transformer Models (Harsh Sharma et al., 2023)

{{<citation>}}

Harsh Sharma, Pratyush Dhingra, Janardhan Rao Doppa, Umit Ogras, Partha Pratim Pande. (2023)  
**A Heterogeneous Chiplet Architecture for Accelerating End-to-End Transformer Models**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-DC, cs.AR  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.11750v1)  

---


**ABSTRACT**  
Transformers have revolutionized deep learning and generative modeling, enabling unprecedented advancements in natural language processing tasks. However, the size of transformer models is increasing continuously, driven by enhanced capabilities across various deep-learning tasks. This trend of ever-increasing model size has given rise to new challenges in terms of memory and computing requirements. Conventional computing platforms, including GPUs, suffer from suboptimal performance due to the memory demands imposed by models with millions/billions of parameters. The emerging chiplet-based platforms provide a new avenue for compute- and data-intensive machine learning (ML) applications enabled by a Network-on-Interposer (NoI). However, designing suitable hardware accelerators for executing Transformer inference workloads is challenging due to a wide variety of complex computing kernels in the Transformer architecture. In this paper, we leverage chiplet-based heterogeneous integration (HI) to design a high-performance and energy-efficient multi-chiplet platform to accelerate transformer workloads. We demonstrate that the proposed NoI architecture caters to the data access patterns inherent in a transformer model. The optimized placement of the chiplets and the associated NoI links and routers enable superior performance compared to the state-of-the-art hardware accelerators. The proposed NoI-based architecture demonstrates scalability across varying transformer models and improves latency and energy efficiency by up to 22.8x and 5.36x respectively.

{{</citation>}}


## cs.LG (20)



### (18/134) Robust Stochastic Graph Generator for Counterfactual Explanations (Mario Alfonso Prado-Romero et al., 2023)

{{<citation>}}

Mario Alfonso Prado-Romero, Bardh Prenkaj, Giovanni Stilo. (2023)  
**Robust Stochastic Graph Generator for Counterfactual Explanations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11747v1)  

---


**ABSTRACT**  
Counterfactual Explanation (CE) techniques have garnered attention as a means to provide insights to the users engaging with AI systems. While extensively researched in domains such as medical imaging and autonomous vehicles, Graph Counterfactual Explanation (GCE) methods have been comparatively under-explored. GCEs generate a new graph similar to the original one, with a different outcome grounded on the underlying predictive model. Among these GCE techniques, those rooted in generative mechanisms have received relatively limited investigation despite demonstrating impressive accomplishments in other domains, such as artistic styles and natural language modelling. The preference for generative explainers stems from their capacity to generate counterfactual instances during inference, leveraging autonomously acquired perturbations of the input graph. Motivated by the rationales above, our study introduces RSGG-CE, a novel Robust Stochastic Graph Generator for Counterfactual Explanations able to produce counterfactual examples from the learned latent space considering a partially ordered generation sequence. Furthermore, we undertake quantitative and qualitative analyses to compare RSGG-CE's performance against SoA generative explainers, highlighting its increased ability to engendering plausible counterfactual candidates.

{{</citation>}}


### (19/134) Stronger Graph Transformer with Regularized Attention Scores (Eugene Ku et al., 2023)

{{<citation>}}

Eugene Ku, Swetha Arunraj. (2023)  
**Stronger Graph Transformer with Regularized Attention Scores**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, GNN, Graph Neural Network, Graph Neural Networks, Transformer  
[Paper Link](http://arxiv.org/abs/2312.11730v1)  

---


**ABSTRACT**  
Graph Neural Networks are notorious for its memory consumption. A recent Transformer based GNN called Graph Transformer are shown to obtain superior performances when long range dependencies exist. However, combining graph data and Transformer architecture led to a combinationally worse memory issue. We propose a novel version of "edge regularization technique" that alleviates the need for Positional Encoding and ultimately alleviate GT's out of memory issue. We observe that it is not clear whether having an edge regularization on top of positional encoding is helpful. However, it seems evident when no positional encoding is applied, edge regularization technique indeed stably improves GT's performance.

{{</citation>}}


### (20/134) Time-Transformer: Integrating Local and Global Features for Better Time Series Generation (Yuansan Liu et al., 2023)

{{<citation>}}

Yuansan Liu, Sudanthi Wijewickrema, Ang Li, Christofer Bester, Stephen O'Leary, James Bailey. (2023)  
**Time-Transformer: Integrating Local and Global Features for Better Time Series Generation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2312.11714v1)  

---


**ABSTRACT**  
Generating time series data is a promising approach to address data deficiency problems. However, it is also challenging due to the complex temporal properties of time series data, including local correlations as well as global dependencies. Most existing generative models have failed to effectively learn both the local and global properties of time series data. To address this open problem, we propose a novel time series generative model named 'Time-Transformer AAE', which consists of an adversarial autoencoder (AAE) and a newly designed architecture named 'Time-Transformer' within the decoder. The Time-Transformer first simultaneously learns local and global features in a layer-wise parallel design, combining the abilities of Temporal Convolutional Networks and Transformer in extracting local features and global dependencies respectively. Second, a bidirectional cross attention is proposed to provide complementary guidance across the two branches and achieve proper fusion between local and global features. Experimental results demonstrate that our model can outperform existing state-of-the-art models in 5 out of 6 datasets, specifically on those with data containing both global and local properties. Furthermore, we highlight our model's advantage on handling this kind of data via an artificial dataset. Finally, we show our model's ability to address a real-world problem: data augmentation to support learning with small datasets and imbalanced datasets.

{{</citation>}}


### (21/134) Prediction and Control in Continual Reinforcement Learning (Nishanth Anand et al., 2023)

{{<citation>}}

Nishanth Anand, Doina Precup. (2023)  
**Prediction and Control in Continual Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11669v1)  

---


**ABSTRACT**  
Temporal difference (TD) learning is often used to update the estimate of the value function which is used by RL agents to extract useful policies. In this paper, we focus on value function estimation in continual reinforcement learning. We propose to decompose the value function into two components which update at different timescales: a permanent value function, which holds general knowledge that persists over time, and a transient value function, which allows quick adaptation to new situations. We establish theoretical results showing that our approach is well suited for continual learning and draw connections to the complementary learning systems (CLS) theory from neuroscience. Empirically, this approach improves performance significantly on both prediction and control problems.

{{</citation>}}


### (22/134) Gibbs Sampling from Human Feedback: A Provable KL- constrained Framework for RLHF (Wei Xiong et al., 2023)

{{<citation>}}

Wei Xiong, Hanze Dong, Chenlu Ye, Han Zhong, Nan Jiang, Tong Zhang. (2023)  
**Gibbs Sampling from Human Feedback: A Provable KL- constrained Framework for RLHF**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11456v1)  

---


**ABSTRACT**  
This paper studies the theoretical framework of the alignment process of generative models with Reinforcement Learning from Human Feedback (RLHF). We consider a standard mathematical formulation, the reverse-KL regularized contextual bandit for RLHF. Despite its widespread practical application, a rigorous theoretical analysis of this formulation remains open. We investigate its theoretical properties both in offline and online settings and propose efficient algorithms with finite-sample theoretical guarantees. Our work bridges the gap between theory and practice by linking our theoretical insights with existing practical alignment algorithms such as Direct Preference Optimization (DPO) and Rejection Sampling Optimization (RSO). Furthermore, these findings and connections also offer both theoretical and practical communities new tools and insights for future algorithmic design of alignment algorithms.

{{</citation>}}


### (23/134) Social Learning: Towards Collaborative Learning with Large Language Models (Amirkeivan Mohtashami et al., 2023)

{{<citation>}}

Amirkeivan Mohtashami, Florian Hartmann, Sian Gooding, Lukas Zilka, Matt Sharifi, Blaise Aguera y Arcas. (2023)  
**Social Learning: Towards Collaborative Learning with Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.11441v1)  

---


**ABSTRACT**  
We introduce the framework of "social learning" in the context of large language models (LLMs), whereby models share knowledge with each other in a privacy-aware manner using natural language. We present and evaluate two approaches for knowledge transfer between LLMs. In the first scenario, we allow the model to generate abstract prompts aiming to teach the task. In our second approach, models transfer knowledge by generating synthetic examples. We evaluate these methods across diverse datasets and quantify memorization as a proxy for privacy loss. These techniques inspired by social learning yield promising results with low memorization of the original data. In particular, we show that performance using these methods is comparable to results with the use of original labels and prompts. Our work demonstrates the viability of social learning for LLMs, establishes baseline approaches and highlights several unexplored areas for future work.

{{</citation>}}


### (24/134) Hypergraph Transformer for Semi-Supervised Classification (Zexi Liu et al., 2023)

{{<citation>}}

Zexi Liu, Bohan Tang, Ziyuan Ye, Xiaowen Dong, Siheng Chen, Yanfeng Wang. (2023)  
**Hypergraph Transformer for Semi-Supervised Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Semi-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2312.11385v1)  

---


**ABSTRACT**  
Hypergraphs play a pivotal role in the modelling of data featuring higher-order relations involving more than two entities. Hypergraph neural networks emerge as a powerful tool for processing hypergraph-structured data, delivering remarkable performance across various tasks, e.g., hypergraph node classification. However, these models struggle to capture global structural information due to their reliance on local message passing. To address this challenge, we propose a novel hypergraph learning framework, HyperGraph Transformer (HyperGT). HyperGT uses a Transformer-based neural network architecture to effectively consider global correlations among all nodes and hyperedges. To incorporate local structural information, HyperGT has two distinct designs: i) a positional encoding based on the hypergraph incidence matrix, offering valuable insights into node-node and hyperedge-hyperedge interactions; and ii) a hypergraph structure regularization in the loss function, capturing connectivities between nodes and hyperedges. Through these designs, HyperGT achieves comprehensive hypergraph representation learning by effectively incorporating global interactions while preserving local connectivity patterns. Extensive experiments conducted on real-world hypergraph node classification tasks showcase that HyperGT consistently outperforms existing methods, establishing new state-of-the-art benchmarks. Ablation studies affirm the effectiveness of the individual designs of our model.

{{</citation>}}


### (25/134) Safeguarded Progress in Reinforcement Learning: Safe Bayesian Exploration for Control Policy Synthesis (Rohan Mitta et al., 2023)

{{<citation>}}

Rohan Mitta, Hosein Hasanbeig, Jun Wang, Daniel Kroening, Yiannis Kantaros, Alessandro Abate. (2023)  
**Safeguarded Progress in Reinforcement Learning: Safe Bayesian Exploration for Control Policy Synthesis**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-LO, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11314v1)  

---


**ABSTRACT**  
This paper addresses the problem of maintaining safety during training in Reinforcement Learning (RL), such that the safety constraint violations are bounded at any point during learning. In a variety of RL applications the safety of the agent is particularly important, e.g. autonomous platforms or robots that work in proximity of humans. As enforcing safety during training might severely limit the agent's exploration, we propose here a new architecture that handles the trade-off between efficient progress and safety during exploration. As the exploration progresses, we update via Bayesian inference Dirichlet-Categorical models of the transition probabilities of the Markov decision process that describes the environment dynamics. This paper proposes a way to approximate moments of belief about the risk associated to the action selection policy. We construct those approximations, and prove the convergence results. We propose a novel method for leveraging the expectation approximations to derive an approximate bound on the confidence that the risk is below a certain level. This approach can be easily interleaved with RL and we present experimental results to showcase the performance of the overall architecture.

{{</citation>}}


### (26/134) Exploring Gradient Explosion in Generative Adversarial Imitation Learning: A Probabilistic Perspective (Wanying Wang et al., 2023)

{{<citation>}}

Wanying Wang, Yichen Zhu, Yirui Zhou, Chaomin Shen, Jian Tang, Zhiyuan Xu, Yaxin Peng, Yangchun Zhang. (2023)  
**Exploring Gradient Explosion in Generative Adversarial Imitation Learning: A Probabilistic Perspective**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11214v1)  

---


**ABSTRACT**  
Generative Adversarial Imitation Learning (GAIL) stands as a cornerstone approach in imitation learning. This paper investigates the gradient explosion in two types of GAIL: GAIL with deterministic policy (DE-GAIL) and GAIL with stochastic policy (ST-GAIL). We begin with the observation that the training can be highly unstable for DE-GAIL at the beginning of the training phase and end up divergence. Conversely, the ST-GAIL training trajectory remains consistent, reliably converging. To shed light on these disparities, we provide an explanation from a theoretical perspective. By establishing a probabilistic lower bound for GAIL, we demonstrate that gradient explosion is an inevitable outcome for DE-GAIL due to occasionally large expert-imitator policy disparity, whereas ST-GAIL does not have the issue with it. To substantiate our assertion, we illustrate how modifications in the reward function can mitigate the gradient explosion challenge. Finally, we propose CREDO, a simple yet effective strategy that clips the reward function during the training phase, allowing the GAIL to enjoy high data efficiency and stable trainability.

{{</citation>}}


### (27/134) AI-Based Energy Transportation Safety: Pipeline Radial Threat Estimation Using Intelligent Sensing System (Chengyuan Zhu et al., 2023)

{{<citation>}}

Chengyuan Zhu, Yiyuan Yang, Kaixiang Yang, Haifeng Zhang, Qinmin Yang, C. L. Philip Chen. (2023)  
**AI-Based Energy Transportation Safety: Pipeline Radial Threat Estimation Using Intelligent Sensing System**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11583v1)  

---


**ABSTRACT**  
The application of artificial intelligence technology has greatly enhanced and fortified the safety of energy pipelines, particularly in safeguarding against external threats. The predominant methods involve the integration of intelligent sensors to detect external vibration, enabling the identification of event types and locations, thereby replacing manual detection methods. However, practical implementation has exposed a limitation in current methods - their constrained ability to accurately discern the spatial dimensions of external signals, which complicates the authentication of threat events. Our research endeavors to overcome the above issues by harnessing deep learning techniques to achieve a more fine-grained recognition and localization process. This refinement is crucial in effectively identifying genuine threats to pipelines, thus enhancing the safety of energy transportation. This paper proposes a radial threat estimation method for energy pipelines based on distributed optical fiber sensing technology. Specifically, we introduce a continuous multi-view and multi-domain feature fusion methodology to extract comprehensive signal features and construct a threat estimation and recognition network. The utilization of collected acoustic signal data is optimized, and the underlying principle is elucidated. Moreover, we incorporate the concept of transfer learning through a pre-trained model, enhancing both recognition accuracy and training efficiency. Empirical evidence gathered from real-world scenarios underscores the efficacy of our method, notably in its substantial reduction of false alarms and remarkable gains in recognition accuracy. More generally, our method exhibits versatility and can be extrapolated to a broader spectrum of recognition tasks and scenarios.

{{</citation>}}


### (28/134) Graph Transformers for Large Graphs (Vijay Prakash Dwivedi et al., 2023)

{{<citation>}}

Vijay Prakash Dwivedi, Yozen Liu, Anh Tuan Luu, Xavier Bresson, Neil Shah, Tong Zhao. (2023)  
**Graph Transformers for Large Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.11109v1)  

---


**ABSTRACT**  
Transformers have recently emerged as powerful neural networks for graph learning, showcasing state-of-the-art performance on several graph property prediction tasks. However, these results have been limited to small-scale graphs, where the computational feasibility of the global attention mechanism is possible. The next goal is to scale up these architectures to handle very large graphs on the scale of millions or even billions of nodes. With large-scale graphs, global attention learning is proven impractical due to its quadratic complexity w.r.t. the number of nodes. On the other hand, neighborhood sampling techniques become essential to manage large graph sizes, yet finding the optimal trade-off between speed and accuracy with sampling techniques remains challenging. This work advances representation learning on single large-scale graphs with a focus on identifying model characteristics and critical design constraints for developing scalable graph transformer (GT) architectures. We argue such GT requires layers that can adeptly learn both local and global graph representations while swiftly sampling the graph topology. As such, a key innovation of this work lies in the creation of a fast neighborhood sampling technique coupled with a local attention mechanism that encompasses a 4-hop reception field, but achieved through just 2-hop operations. This local node embedding is then integrated with a global node embedding, acquired via another self-attention layer with an approximate global codebook, before finally sent through a downstream layer for node predictions. The proposed GT framework, named LargeGT, overcomes previous computational bottlenecks and is validated on three large-scale node classification benchmarks. We report a 3x speedup and 16.8% performance gain on ogbn-products and snap-patents, while we also scale LargeGT on ogbn-papers100M with a 5.9% performance improvement.

{{</citation>}}


### (29/134) Graph Invariant Learning with Subgraph Co-mixup for Out-Of-Distribution Generalization (Tianrui Jia et al., 2023)

{{<citation>}}

Tianrui Jia, Haoyang Li, Cheng Yang, Tao Tao, Chuan Shi. (2023)  
**Graph Invariant Learning with Subgraph Co-mixup for Out-Of-Distribution Generalization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.10988v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have been demonstrated to perform well in graph representation learning, but always lacking in generalization capability when tackling out-of-distribution (OOD) data. Graph invariant learning methods, backed by the invariance principle among defined multiple environments, have shown effectiveness in dealing with this issue. However, existing methods heavily rely on well-predefined or accurately generated environment partitions, which are hard to be obtained in practice, leading to sub-optimal OOD generalization performances. In this paper, we propose a novel graph invariant learning method based on invariant and variant patterns co-mixup strategy, which is capable of jointly generating mixed multiple environments and capturing invariant patterns from the mixed graph data. Specifically, we first adopt a subgraph extractor to identify invariant subgraphs. Subsequently, we design one novel co-mixup strategy, i.e., jointly conducting environment Mixup and invariant Mixup. For the environment Mixup, we mix the variant environment-related subgraphs so as to generate sufficiently diverse multiple environments, which is important to guarantee the quality of the graph invariant learning. For the invariant Mixup, we mix the invariant subgraphs, further encouraging to capture invariant patterns behind graphs while getting rid of spurious correlations for OOD generalization. We demonstrate that the proposed environment Mixup and invariant Mixup can mutually promote each other. Extensive experiments on both synthetic and real-world datasets demonstrate that our method significantly outperforms state-of-the-art under various distribution shifts.

{{</citation>}}


### (30/134) Predicting Financial Literacy via Semi-supervised Learning (David Hason Rudd et al., 2023)

{{<citation>}}

David Hason Rudd, Huan Huo, Guandong Xu. (2023)  
**Predicting Financial Literacy via Semi-supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-CY, cs-LG, cs.LG, econ-EM  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2312.10984v1)  

---


**ABSTRACT**  
Financial literacy (FL) represents a person's ability to turn assets into income, and understanding digital currencies has been added to the modern definition. FL can be predicted by exploiting unlabelled recorded data in financial networks via semi-supervised learning (SSL). Measuring and predicting FL has not been widely studied, resulting in limited understanding of customer financial engagement consequences. Previous studies have shown that low FL increases the risk of social harm. Therefore, it is important to accurately estimate FL to allocate specific intervention programs to less financially literate groups. This will not only increase company profitability, but will also reduce government spending. Some studies considered predicting FL in classification tasks, whereas others developed FL definitions and impacts. The current paper investigated mechanisms to learn customer FL level from their financial data using sampling by synthetic minority over-sampling techniques for regression with Gaussian noise (SMOGN). We propose the SMOGN-COREG model for semi-supervised regression, applying SMOGN to deal with unbalanced datasets and a nonparametric multi-learner co-regression (COREG) algorithm for labeling. We compared the SMOGN-COREG model with six well-known regressors on five datasets to evaluate the proposed models effectiveness on unbalanced and unlabelled financial data. Experimental results confirmed that the proposed method outperformed the comparator models for unbalanced and unlabelled financial data. Therefore, SMOGN-COREG is a step towards using unlabelled data to estimate FL level.

{{</citation>}}


### (31/134) Inducing Point Operator Transformer: A Flexible and Scalable Architecture for Solving PDEs (Seungjun Lee et al., 2023)

{{<citation>}}

Seungjun Lee, Taeil Oh. (2023)  
**Inducing Point Operator Transformer: A Flexible and Scalable Architecture for Solving PDEs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NA, cs.LG, math-NA  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.10975v1)  

---


**ABSTRACT**  
Solving partial differential equations (PDEs) by learning the solution operators has emerged as an attractive alternative to traditional numerical methods. However, implementing such architectures presents two main challenges: flexibility in handling irregular and arbitrary input and output formats and scalability to large discretizations. Most existing architectures are limited by their desired structure or infeasible to scale large inputs and outputs. To address these issues, we introduce an attention-based model called an inducing-point operator transformer (IPOT). Inspired by inducing points methods, IPOT is designed to handle any input function and output query while capturing global interactions in a computationally efficient way. By detaching the inputs/outputs discretizations from the processor with a smaller latent bottleneck, IPOT offers flexibility in processing arbitrary discretizations and scales linearly with the size of inputs/outputs. Our experimental results demonstrate that IPOT achieves strong performances with manageable computational complexity on an extensive range of PDE benchmarks and real-world weather forecasting scenarios, compared to state-of-the-art methods.

{{</citation>}}


### (32/134) Estimation of individual causal effects in network setup for multiple treatments (Abhinav Thorat et al., 2023)

{{<citation>}}

Abhinav Thorat, Ravi Kolla, Niranjan Pedanekar, Naoyuki Onoe. (2023)  
**Estimation of individual causal effects in network setup for multiple treatments**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ME  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2312.11573v1)  

---


**ABSTRACT**  
We study the problem of estimation of Individual Treatment Effects (ITE) in the context of multiple treatments and networked observational data. Leveraging the network information, we aim to utilize hidden confounders that may not be directly accessible in the observed data, thereby enhancing the practical applicability of the strong ignorability assumption. To achieve this, we first employ Graph Convolutional Networks (GCN) to learn a shared representation of the confounders. Then, our approach utilizes separate neural networks to infer potential outcomes for each treatment. We design a loss function as a weighted combination of two components: representation loss and Mean Squared Error (MSE) loss on the factual outcomes. To measure the representation loss, we extend existing metrics such as Wasserstein and Maximum Mean Discrepancy (MMD) from the binary treatment setting to the multiple treatments scenario. To validate the effectiveness of our proposed methodology, we conduct a series of experiments on the benchmark datasets such as BlogCatalog and Flickr. The experimental results consistently demonstrate the superior performance of our models when compared to baseline methods.

{{</citation>}}


### (33/134) Model Stealing Attack against Graph Classification with Authenticity, Uncertainty and Diversity (Zhihao Zhu et al., 2023)

{{<citation>}}

Zhihao Zhu, Chenwang Wu, Rui Fan, Yi Yang, Defu Lian, Enhong Chen. (2023)  
**Model Stealing Attack against Graph Classification with Authenticity, Uncertainty and Diversity**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.10943v1)  

---


**ABSTRACT**  
Recent research demonstrates that GNNs are vulnerable to the model stealing attack, a nefarious endeavor geared towards duplicating the target model via query permissions. However, they mainly focus on node classification tasks, neglecting the potential threats entailed within the domain of graph classification tasks. Furthermore, their practicality is questionable due to unreasonable assumptions, specifically concerning the large data requirements and extensive model knowledge. To this end, we advocate following strict settings with limited real data and hard-label awareness to generate synthetic data, thereby facilitating the stealing of the target model. Specifically, following important data generation principles, we introduce three model stealing attacks to adapt to different actual scenarios: MSA-AU is inspired by active learning and emphasizes the uncertainty to enhance query value of generated samples; MSA-AD introduces diversity based on Mixup augmentation strategy to alleviate the query inefficiency issue caused by over-similar samples generated by MSA-AU; MSA-AUD combines the above two strategies to seamlessly integrate the authenticity, uncertainty, and diversity of the generated samples. Finally, extensive experiments consistently demonstrate the superiority of the proposed methods in terms of concealment, query efficiency, and stealing performance.

{{</citation>}}


### (34/134) Domain adaption and physical constrains transfer learning for shale gas production (Zhaozhong Yang et al., 2023)

{{<citation>}}

Zhaozhong Yang, Liangjie Gou, Chao Min, Duo Yi, Xiaogang Li, Guoquan Wen. (2023)  
**Domain adaption and physical constrains transfer learning for shale gas production**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ME  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.10920v1)  

---


**ABSTRACT**  
Effective prediction of shale gas production is crucial for strategic reservoir development. However, in new shale gas blocks, two main challenges are encountered: (1) the occurrence of negative transfer due to insufficient data, and (2) the limited interpretability of deep learning (DL) models. To tackle these problems, we propose a novel transfer learning methodology that utilizes domain adaptation and physical constraints. This methodology effectively employs historical data from the source domain to reduce negative transfer from the data distribution perspective, while also using physical constraints to build a robust and reliable prediction model that integrates various types of data. The methodology starts by dividing the production data from the source domain into multiple subdomains, thereby enhancing data diversity. It then uses Maximum Mean Discrepancy (MMD) and global average distance measures to decide on the feasibility of transfer. Through domain adaptation, we integrate all transferable knowledge, resulting in a more comprehensive target model. Lastly, by incorporating drilling, completion, and geological data as physical constraints, we develop a hybrid model. This model, a combination of a multi-layer perceptron (MLP) and a Transformer (Transformer-MLP), is designed to maximize interpretability. Experimental validation in China's southwestern region confirms the method's effectiveness.

{{</citation>}}


### (35/134) Semi-Supervised Clustering via Structural Entropy with Different Constraints (Guangjie Zeng et al., 2023)

{{<citation>}}

Guangjie Zeng, Hao Peng, Angsheng Li, Zhiwei Liu, Runze Yang, Chunyang Liu, Lifang He. (2023)  
**Semi-Supervised Clustering via Structural Entropy with Different Constraints**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.10917v1)  

---


**ABSTRACT**  
Semi-supervised clustering techniques have emerged as valuable tools for leveraging prior information in the form of constraints to improve the quality of clustering outcomes. Despite the proliferation of such methods, the ability to seamlessly integrate various types of constraints remains limited. While structural entropy has proven to be a powerful clustering approach with wide-ranging applications, it has lacked a variant capable of accommodating these constraints. In this work, we present Semi-supervised clustering via Structural Entropy (SSE), a novel method that can incorporate different types of constraints from diverse sources to perform both partitioning and hierarchical clustering. Specifically, we formulate a uniform view for the commonly used pairwise and label constraints for both types of clustering. Then, we design objectives that incorporate these constraints into structural entropy and develop tailored algorithms for their optimization. We evaluate SSE on nine clustering datasets and compare it with eleven semi-supervised partitioning and hierarchical clustering methods. Experimental results demonstrate the superiority of SSE on clustering accuracy with different types of constraints. Additionally, the functionality of SSE for biological data analysis is demonstrated by cell clustering experiments conducted on four single-cell RNAseq datasets.

{{</citation>}}


### (36/134) Satellite Captioning: Large Language Models to Augment Labeling (Grant Rosario et al., 2023)

{{<citation>}}

Grant Rosario, David Noever. (2023)  
**Satellite Captioning: Large Language Models to Augment Labeling**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: ChatGPT, GPT, ImageNet, Language Model  
[Paper Link](http://arxiv.org/abs/2312.10905v1)  

---


**ABSTRACT**  
With the growing capabilities of modern object detection networks and datasets to train them, it has gotten more straightforward and, importantly, less laborious to get up and running with a model that is quite adept at detecting any number of various objects. However, while image datasets for object detection have grown and continue to proliferate (the current most extensive public set, ImageNet, contains over 14m images with over 14m instances), the same cannot be said for textual caption datasets. While they have certainly been growing in recent years, caption datasets present a much more difficult challenge due to language differences, grammar, and the time it takes for humans to generate them. Current datasets have certainly provided many instances to work with, but it becomes problematic when a captioner may have a more limited vocabulary, one may not be adequately fluent in the language, or there are simple grammatical mistakes. These difficulties are increased when the images get more specific, such as remote sensing images. This paper aims to address this issue of potential information and communication shortcomings in caption datasets. To provide a more precise analysis, we specify our domain of images to be remote sensing images in the RSICD dataset and experiment with the captions provided here. Our findings indicate that ChatGPT grammar correction is a simple and effective way to increase the performance accuracy of caption models by making data captions more diverse and grammatically correct.

{{</citation>}}


### (37/134) Robust Node Representation Learning via Graph Variational Diffusion Networks (Jun Zhuang et al., 2023)

{{<citation>}}

Jun Zhuang, Mohammad Al Hasan. (2023)  
**Robust Node Representation Learning via Graph Variational Diffusion Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.10903v1)  

---


**ABSTRACT**  
Node representation learning by using Graph Neural Networks (GNNs) has been widely explored. However, in recent years, compelling evidence has revealed that GNN-based node representation learning can be substantially deteriorated by delicately-crafted perturbations in a graph structure. To learn robust node representation in the presence of perturbations, various works have been proposed to safeguard GNNs. Within these existing works, Bayesian label transition has been proven to be more effective, but this method is extensively reliant on a well-built prior distribution. The variational inference could address this limitation by sampling the latent node embedding from a Gaussian prior distribution. Besides, leveraging the Gaussian distribution (noise) in hidden layers is an appealing strategy to strengthen the robustness of GNNs. However, our experiments indicate that such a strategy can cause over-smoothing issues during node aggregation. In this work, we propose the Graph Variational Diffusion Network (GVDN), a new node encoder that effectively manipulates Gaussian noise to safeguard robustness on perturbed graphs while alleviating over-smoothing issues through two mechanisms: Gaussian diffusion and node embedding propagation. Thanks to these two mechanisms, our model can generate robust node embeddings for recovery. Specifically, we design a retraining mechanism using the generated node embedding to recover the performance of node classifications in the presence of perturbations. The experiments verify the effectiveness of our proposed model across six public datasets.

{{</citation>}}


## cs.DC (1)



### (38/134) TPTO: A Transformer-PPO based Task Offloading Solution for Edge Computing Environments (Niloofar Gholipour et al., 2023)

{{<citation>}}

Niloofar Gholipour, Marcos Dias de Assuncao, Pranav Agarwal, julien gascon-samson, Rajkumar Buyya. (2023)  
**TPTO: A Transformer-PPO based Task Offloading Solution for Edge Computing Environments**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.11739v1)  

---


**ABSTRACT**  
Emerging applications in healthcare, autonomous vehicles, and wearable assistance require interactive and low-latency data analysis services. Unfortunately, cloud-centric architectures cannot fulfill the low-latency demands of these applications, as user devices are often distant from cloud data centers. Edge computing aims to reduce the latency by enabling processing tasks to be offloaded to resources located at the network's edge. However, determining which tasks must be offloaded to edge servers to reduce the latency of application requests is not trivial, especially if the tasks present dependencies. This paper proposes a DRL approach called TPTO, which leverages Transformer Networks and PPO to offload dependent tasks of IoT applications in edge computing. We consider users with various preferences, where devices can offload computation to an edge server via wireless channels. Performance evaluation results demonstrate that under fat application graphs, TPTO is more effective than state-of-the-art methods, such as Greedy, HEFT, and MRLCO, by reducing latency by 30.24%, 29.61%, and 12.41%, respectively. In addition, TPTO presents a training time approximately 2.5 times faster than an existing DRL approach.

{{</citation>}}


## cs.CL (27)



### (39/134) Assessing Logical Reasoning Capabilities of Encoder-Only Transformer Models (Paulo Pirozelli et al., 2023)

{{<citation>}}

Paulo Pirozelli, Marcos M. José, Paulo de Tarso P. Filho, Anarosa A. F. Brandão, Fabio G. Cozman. (2023)  
**Assessing Logical Reasoning Capabilities of Encoder-Only Transformer Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Reasoning, Transformer  
[Paper Link](http://arxiv.org/abs/2312.11720v1)  

---


**ABSTRACT**  
Logical reasoning is central to complex human activities, such as thinking, debating, and planning; it is also a central component of many AI systems as well. In this paper, we investigate the extent to which encoder-only transformer language models (LMs) can reason according to logical rules. We ask whether those LMs can deduce theorems in propositional calculus and first-order logic; if their relative success in these problems reflects general logical capabilities; and which layers contribute the most to the task. First, we show for several encoder-only LMs that they can be trained, to a reasonable degree, to determine logical validity on various datasets. Next, by cross-probing fine-tuned models on these datasets, we show that LMs have difficulty in transferring their putative logical reasoning ability, which suggests that they may have learned dataset-specific features, instead of a general capability. Finally, we conduct a layerwise probing experiment, which shows that the hypothesis classification task is mostly solved through higher layers.

{{</citation>}}


### (40/134) Shaping Political Discourse using multi-source News Summarization (Charles Rajan et al., 2023)

{{<citation>}}

Charles Rajan, Nishit Asnani, Shreya Singh. (2023)  
**Shaping Political Discourse using multi-source News Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-IR, cs-LG, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2312.11703v1)  

---


**ABSTRACT**  
Multi-document summarization is the process of automatically generating a concise summary of multiple documents related to the same topic. This summary can help users quickly understand the key information from a large collection of documents. Multi-document summarization systems are more complex than single-document summarization systems due to the need to identify and combine information from multiple sources. In this paper, we have developed a machine learning model that generates a concise summary of a topic from multiple news documents. The model is designed to be unbiased by sampling its input equally from all the different aspects of the topic, even if the majority of the news sources lean one way.

{{</citation>}}


### (41/134) An In-depth Look at Gemini's Language Abilities (Syeda Nahida Akter et al., 2023)

{{<citation>}}

Syeda Nahida Akter, Zichun Yu, Aashiq Muhamed, Tianyue Ou, Alex Bäuerle, Ángel Alexander Cabrera, Krish Dholakia, Chenyan Xiong, Graham Neubig. (2023)  
**An In-depth Look at Gemini's Language Abilities**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, Google  
[Paper Link](http://arxiv.org/abs/2312.11444v1)  

---


**ABSTRACT**  
The recently released Google Gemini class of models are the first to comprehensively report results that rival the OpenAI GPT series across a wide variety of tasks. In this paper, we do an in-depth exploration of Gemini's language abilities, making two contributions. First, we provide a third-party, objective comparison of the abilities of the OpenAI GPT and Google Gemini models with reproducible code and fully transparent results. Second, we take a closer look at the results, identifying areas where one of the two model classes excels. We perform this analysis over 10 datasets testing a variety of language abilities, including reasoning, answering knowledge-based questions, solving math problems, translating between languages, generating code, and acting as instruction-following agents. From this analysis, we find that Gemini Pro achieves accuracy that is close but slightly inferior to the corresponding GPT 3.5 Turbo on all tasks that we benchmarked. We further provide explanations for some of this under-performance, including failures in mathematical reasoning with many digits, sensitivity to multiple-choice answer ordering, aggressive content filtering, and others. We also identify areas where Gemini demonstrates comparably high performance, including generation into non-English languages, and handling longer and more complex reasoning chains. Code and data for reproduction can be found at https://github.com/neulab/gemini-benchmark

{{</citation>}}


### (42/134) Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning (Bingchen Zhao et al., 2023)

{{<citation>}}

Bingchen Zhao, Haoqin Tu, Chen Wei, Jieru Mei, Cihang Xie. (2023)  
**Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: Attention, Language Model  
[Paper Link](http://arxiv.org/abs/2312.11420v1)  

---


**ABSTRACT**  
This paper introduces an efficient strategy to transform Large Language Models (LLMs) into Multi-Modal Large Language Models (MLLMs). By conceptualizing this transformation as a domain adaptation process, i.e., transitioning from text understanding to embracing multiple modalities, we intriguingly note that, within each attention block, tuning LayerNorm suffices to yield strong performance. Moreover, when benchmarked against other tuning approaches like full parameter finetuning or LoRA, its benefits on efficiency are substantial. For example, when compared to LoRA on a 13B model scale, performance can be enhanced by an average of over 20% across five multi-modal tasks, and meanwhile, results in a significant reduction of trainable parameters by 41.9% and a decrease in GPU memory usage by 17.6%. On top of this LayerNorm strategy, we showcase that selectively tuning only with conversational data can improve efficiency further. Beyond these empirical outcomes, we provide a comprehensive analysis to explore the role of LayerNorm in adapting LLMs to the multi-modal domain and improving the expressive power of the model.

{{</citation>}}


### (43/134) News Signals: An NLP Library for Text and Time Series (Chris Hokamp et al., 2023)

{{<citation>}}

Chris Hokamp, Demian Gholipour Ghalandari, Parsa Ghaffari. (2023)  
**News Signals: An NLP Library for Text and Time Series**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Time Series  
[Paper Link](http://arxiv.org/abs/2312.11399v1)  

---


**ABSTRACT**  
We present an open-source Python library for building and using datasets where inputs are clusters of textual data, and outputs are sequences of real values representing one or more time series signals. The news-signals library supports diverse data science and NLP problem settings related to the prediction of time series behaviour using textual data feeds. For example, in the news domain, inputs are document clusters corresponding to daily news articles about a particular entity, and targets are explicitly associated real-valued time series: the volume of news about a particular person or company, or the number of pageviews of specific Wikimedia pages. Despite many industry and research use cases for this class of problem settings, to the best of our knowledge, News Signals is the only open-source library designed specifically to facilitate data science and research settings with natural language inputs and time series targets. In addition to the core codebase for building and interacting with datasets, we also conduct a suite of experiments using several popular Machine Learning libraries, which are used to establish baselines for time series anomaly prediction using textual inputs.

{{</citation>}}


### (44/134) Verb Categorisation for Hindi Word Problem Solving (Harshita Sharma et al., 2023)

{{<citation>}}

Harshita Sharma, Pruthwik Mishra, Dipti Misra Sharma. (2023)  
**Verb Categorisation for Hindi Word Problem Solving**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.11395v1)  

---


**ABSTRACT**  
Word problem Solving is a challenging NLP task that deals with solving mathematical problems described in natural language. Recently, there has been renewed interest in developing word problem solvers for Indian languages. As part of this paper, we have built a Hindi arithmetic word problem solver which makes use of verbs. Additionally, we have created verb categorization data for Hindi. Verbs are very important for solving word problems with addition/subtraction operations as they help us identify the set of operations required to solve the word problems. We propose a rule-based solver that uses verb categorisation to identify operations in a word problem and generate answers for it. To perform verb categorisation, we explore several approaches and present a comparative study.

{{</citation>}}


### (45/134) G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model (Jiahui Gao et al., 2023)

{{<citation>}}

Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, Lingpeng Kong. (2023)  
**G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.11370v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown remarkable proficiency in human-level reasoning and generation capabilities, which encourages extensive research on their application in mathematical problem solving. However, current work has been largely focused on text-based mathematical problems, with limited investigation in problems involving geometric information. Addressing this gap, we aim to enable LLMs to solve geometric problems by understanding image input. We first analyze the limitations of current Multimodal Large Language Models (MLLMs) in this area: they struggle to accurately comprehending basic geometric elements and their relationships. To overcome these challenges, we take advantage of the unique characteristics of geometric problems (such as unique geometric logical form, and geometric scalability) and the capacity of the textual LLMs to build an enriched multimodal geometry dataset based on existing data. The augmented dataset, Geo170K, contains more than 170K geometric image-caption and question-answer pairs. Utilizing our constructed Geo170K dataset, we develop G-LLaVA, which demonstrates exceptional performance in solving geometric problems, significantly outperforming GPT-4-V on the MathVista benchmark with only 7B parameters.

{{</citation>}}


### (46/134) NoMIRACL: Knowing When You Don't Know for Robust Multilingual Retrieval-Augmented Generation (Nandan Thakur et al., 2023)

{{<citation>}}

Nandan Thakur, Luiz Bonifacio, Xinyu Zhang, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Boxing Chen, Mehdi Rezagholizadeh, Jimmy Lin. (2023)  
**NoMIRACL: Knowing When You Don't Know for Robust Multilingual Retrieval-Augmented Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: GPT, GPT-4, Multilingual  
[Paper Link](http://arxiv.org/abs/2312.11361v1)  

---


**ABSTRACT**  
Retrieval-augmented generation (RAG) grounds large language model (LLM) output by leveraging external knowledge sources to reduce factual hallucinations. However, prior works lack a comprehensive evaluation of different language families, making it challenging to evaluate LLM robustness against errors in external retrieved knowledge. To overcome this, we establish NoMIRACL, a human-annotated dataset for evaluating LLM robustness in RAG across 18 typologically diverse languages. NoMIRACL includes both a non-relevant and a relevant subset. Queries in the non-relevant subset contain passages manually judged as non-relevant or noisy, whereas queries in the relevant subset include at least a single judged relevant passage. We measure LLM robustness using two metrics: (i) hallucination rate, measuring model tendency to hallucinate an answer, when the answer is not present in passages in the non-relevant subset, and (ii) error rate, measuring model inaccuracy to recognize relevant passages in the relevant subset. We build a GPT-4 baseline which achieves a 33.2% hallucination rate on the non-relevant and a 14.9% error rate on the relevant subset on average. Our evaluation reveals that GPT-4 hallucinates frequently in high-resource languages, such as French or English. This work highlights an important avenue for future research to improve LLM robustness to learn how to better reject non-relevant information in RAG.

{{</citation>}}


### (47/134) Muted: Multilingual Targeted Offensive Speech Identification and Visualization (Christoph Tillmann et al., 2023)

{{<citation>}}

Christoph Tillmann, Aashka Trivedi, Sara Rosenthal, Santosh Borse, Rong Zhang, Avirup Sil, Bishwaranjan Bhattacharjee. (2023)  
**Muted: Multilingual Targeted Offensive Speech Identification and Visualization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2312.11344v1)  

---


**ABSTRACT**  
Offensive language such as hate, abuse, and profanity (HAP) occurs in various content on the web. While previous work has mostly dealt with sentence level annotations, there have been a few recent attempts to identify offensive spans as well. We build upon this work and introduce Muted, a system to identify multilingual HAP content by displaying offensive arguments and their targets using heat maps to indicate their intensity. Muted can leverage any transformer-based HAP-classification model and its attention mechanism out-of-the-box to identify toxic spans, without further fine-tuning. In addition, we use the spaCy library to identify the specific targets and arguments for the words predicted by the attention heatmaps. We present the model's performance on identifying offensive spans and their targets in existing datasets and present new annotations on German text. Finally, we demonstrate our proposed visualization tool on multilingual inputs.

{{</citation>}}


### (48/134) APE-then-QE: Correcting then Filtering Pseudo Parallel Corpora for MT Training Data Creation (Akshay Batheja et al., 2023)

{{<citation>}}

Akshay Batheja, Sourabh Deoghare, Diptesh Kanojia, Pushpak Bhattacharyya. (2023)  
**APE-then-QE: Correcting then Filtering Pseudo Parallel Corpora for MT Training Data Creation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Machine Translation  
[Paper Link](http://arxiv.org/abs/2312.11312v1)  

---


**ABSTRACT**  
Automatic Post-Editing (APE) is the task of automatically identifying and correcting errors in the Machine Translation (MT) outputs. We propose a repair-filter-use methodology that uses an APE system to correct errors on the target side of the MT training data. We select the sentence pairs from the original and corrected sentence pairs based on the quality scores computed using a Quality Estimation (QE) model. To the best of our knowledge, this is a novel adaptation of APE and QE to extract quality parallel corpus from the pseudo-parallel corpus. By training with this filtered corpus, we observe an improvement in the Machine Translation system's performance by 5.64 and 9.91 BLEU points, for English-Marathi and Marathi-English, over the baseline model. The baseline model is the one that is trained on the whole pseudo-parallel corpus. Our work is not limited by the characteristics of English or Marathi languages; and is language pair-agnostic, given the necessary QE and APE data.

{{</citation>}}


### (49/134) From Generalized Laughter to Personalized Chuckles: Unleashing the Power of Data Fusion in Subjective Humor Detection (Julita Bielaniewicz et al., 2023)

{{<citation>}}

Julita Bielaniewicz, Przemysław Kazienko. (2023)  
**From Generalized Laughter to Personalized Chuckles: Unleashing the Power of Data Fusion in Subjective Humor Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.11296v1)  

---


**ABSTRACT**  
The vast area of subjectivity in Natural Language Processing (NLP) poses a challenge to the solutions typically used in generalized tasks. As exploration in the scope of generalized NLP is much more advanced, it implies the tremendous gap that is still to be addressed amongst all feasible tasks where an opinion, taste, or feelings are inherent, thus creating a need for a solution, where a data fusion could take place. We have chosen the task of funniness, as it heavily relies on the sense of humor, which is fundamentally subjective. Our experiments across five personalized and four generalized datasets involving several personalized deep neural architectures have shown that the task of humor detection greatly benefits from the inclusion of personalized data in the training process. We tested five scenarios of training data fusion that focused on either generalized (majority voting) or personalized approaches to humor detection. The best results were obtained for the setup, in which all available personalized datasets were joined to train the personalized reasoning model. It boosted the prediction performance by up to approximately 35% of the macro F1 score. Such a significant gain was observed for all five personalized test sets. At the same time, the impact of the model's architecture was much less than the personalization itself. It seems that concatenating personalized datasets, even with the cost of normalizing the range of annotations across all datasets, if combined with the personalized models, results in an enormous increase in the performance of humor detection.

{{</citation>}}


### (50/134) LLM-ARK: Knowledge Graph Reasoning Using Large Language Models via Deep Reinforcement Learning (Yuxuan Huang, 2023)

{{<citation>}}

Yuxuan Huang. (2023)  
**LLM-ARK: Knowledge Graph Reasoning Using Large Language Models via Deep Reinforcement Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Knowledge Graph, LLaMA, Language Model, Reasoning, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11282v1)  

---


**ABSTRACT**  
With the evolution of pre-training methods, large language models (LLMs) have exhibited exemplary reasoning capabilities via prompt engineering. However, the absence of Knowledge Graph (KG) environment awareness and the challenge of engineering viable optimization mechanisms for intermediary reasoning processes, constrict the performance of LLMs on KG reasoning tasks compared to smaller models. We introduce LLM-ARK, a LLM grounded KG reasoning agent designed to deliver precise and adaptable predictions on KG paths. LLM-ARK utilizes Full Textual Environment (FTE) prompts to assimilate state information for each step-sized intelligence. Leveraging LLMs to richly encode and represent various types of inputs and integrate the knowledge graph further with path environment data, before making the final decision. Reframing the Knowledge Graph (KG) multi-hop inference problem as a sequential decision-making issue, we optimize our model using the Proximal Policy Optimization (PPO) online policy gradient reinforcement learning algorithm which allows the model to learn from a vast array of reward signals across diverse tasks and environments. We evaluate state-of-the-art LLM(GPT-4) and our method which using open-source models of varying sizes on OpenDialKG dataset. Our experiment shows that LLaMA7B-ARK provides excellent results with a performance rate of 48.75% for the target@1 evaluation metric, far exceeding the current state-of-the-art model by 17.64 percentage points. Meanwhile, GPT-4 accomplished a score of only 14.91%, further highlighting the efficacy and complexity of our methodology. Our code is available on GitHub for further access.

{{</citation>}}


### (51/134) Compositional Generalization for Multi-label Text Classification: A Data-Augmentation Approach (Yuyang Chai et al., 2023)

{{<citation>}}

Yuyang Chai, Zhuang Li, Jiahui Liu, Lei Chen, Fei Li, Donghong Ji, Chong Teng. (2023)  
**Compositional Generalization for Multi-label Text Classification: A Data-Augmentation Approach**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, Text Classification  
[Paper Link](http://arxiv.org/abs/2312.11276v2)  

---


**ABSTRACT**  
Despite significant advancements in multi-label text classification, the ability of existing models to generalize to novel and seldom-encountered complex concepts, which are compositions of elementary ones, remains underexplored. This research addresses this gap. By creating unique data splits across three benchmarks, we assess the compositional generalization ability of existing multi-label text classification models. Our results show that these models often fail to generalize to compositional concepts encountered infrequently during training, leading to inferior performance on tests with these new combinations. To address this, we introduce a data augmentation method that leverages two innovative text generation models designed to enhance the classification models' capacity for compositional generalization. Our experiments show that this data augmentation approach significantly improves the compositional generalization capabilities of classification models on our benchmarks, with both generation models surpassing other text generation baselines.

{{</citation>}}


### (52/134) MAC-SQL: Multi-Agent Collaboration for Text-to-SQL (Bing Wang et al., 2023)

{{<citation>}}

Bing Wang, Changyu Ren, Jian Yang, Xinnian Liang, Jiaqi Bai, Qian-Wen Zhang, Zhao Yan, Zhoujun Li. (2023)  
**MAC-SQL: Multi-Agent Collaboration for Text-to-SQL**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.11242v1)  

---


**ABSTRACT**  
Recent advancements in Text-to-SQL methods employing Large Language Models (LLMs) have demonstrated remarkable performance. Nonetheless, these approaches continue to encounter difficulties when handling extensive databases, intricate user queries, and erroneous SQL results. To tackle these challenges, we present \textbf{MAC-SQL}, a LLM-based multi-agent collaborative Text- to-SQL framework based on LLMs. This framework comprises three agents: the \textit{Selector}, accountable for condensing voluminous databases and preserving relevant table schemas for user questions; the \textit{Decomposer}, which disassembles complex user questions into more straightforward sub-problems and resolves them progressively; and the \textit{Refiner}, tasked with validating and refining defective SQL queries. We perform thorough experiments on two Text-to-SQL datasets, BIRD and Spider, attaining a state-of-the-art execution accuracy of 59.59\% on the BIRD test set. Moreover, we have open-sourced an instruction fine-tuning model, \textbf{SQL-Llama}, based on Code Llama 7B, in addition to an agent instruction dataset derived from training data based on BIRD and Spider. The SQL-Llama model has demonstrated encouraging outcomes on the development sets of both BIRD and Spider. However, when compared to the GPT-4 model, there remains a notable potential for enhancement. Our code and data can be accessed publicly at \href{https://github.com/wbbeyourself/MAC-SQL}{https://github.com/wbbeyourself/MAC-SQL}.

{{</citation>}}


### (53/134) 'Paraphrasing The Original Text' Makes High Accuracy Long-Context QA (Yijiong Yu, 2023)

{{<citation>}}

Yijiong Yu. (2023)  
**'Paraphrasing The Original Text' Makes High Accuracy Long-Context QA**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.11193v2)  

---


**ABSTRACT**  
Although LLMs continue to iterate and improve, most open-source models still have a context window of no more than 4k, limiting their ability to handle long-context problems. Most existing open-source models for long-context chat still lack satisfactory accuracy. To address this issue, I approach it from the perspective of training data and theoretically prove that training the capability to handle long contexts requires "effective" rather than "long" data. Based on this, I propose using the "original text paraphrase" task, and successfully extend the context window of the existing model to 32k by a low-cost and effective method, achieving extremely high accuracy in multi-document-QA and surpassing all existing open-source models of the same scale. The model and training data have been open-sourced on HuggingFace(https://huggingface.co/yuyijiong/Qwen-14b-chat-yarn-32k) and WiseModel(https://wisemodel.cn/models/yuyijiong/Qwen-14b-chat-yarn-32k).

{{</citation>}}


### (54/134) Linear Attention via Orthogonal Memory (Jun Zhang et al., 2023)

{{<citation>}}

Jun Zhang, Shuyang Jiang, Jiangtao Feng, Lin Zheng, Lingpeng Kong. (2023)  
**Linear Attention via Orthogonal Memory**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.11135v1)  

---


**ABSTRACT**  
Efficient attentions have greatly improved the computational efficiency of Transformers. However, most existing linear attention mechanisms suffer from an \emph{efficiency degradation} problem, leading to inefficiencies in causal language modeling and hindering their application in long-range language models. This problem is more pronounced under language modeling with unbounded contexts. In this paper, we propose \textbf{L}inear \textbf{A}ttention \textbf{V}ia \textbf{O}rthogonal memory~(\shortname) to address these limitations, achieving strong performance while maintaining linear complexity. \shortname employs orthogonal decomposition to compress a context into a fixed-size orthogonal memory while effectively minimizing redundancy within the context. Given that orthogonal memory compresses global information, we further dissect the context to amplify fine-grained local information. Additionally, we embed the relative position encoding into \shortname to improve the extrapolation ability. Experimental results show that \shortname greatly improves the efficiency of the causal language model with the best extrapolation performance and outperforms other efficient baselines. Further, we endeavor to employ \shortname for unbounded language modeling and successfully scale the context length to 128K.

{{</citation>}}


### (55/134) Split and Rephrase with Large Language Models (David Ponce et al., 2023)

{{<citation>}}

David Ponce, Thierry Etchegoyhen, Jesús Calleja Pérez, Harritxu Gete. (2023)  
**Split and Rephrase with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.11075v2)  

---


**ABSTRACT**  
The Split and Rephrase task, which consists in splitting complex sentences into a sequence of shorter grammatical sentences, while preserving the original meaning, can facilitate the processing of complex texts for humans and machines alike. In this work, we describe an approach based on large language models, which improves over the state of the art by large margins on all the major metrics for the task, on publicly available datasets. We also describe results from two human evaluations that further establish the significant improvements obtained with large language models and the viability of the approach. We evaluate different strategies, including fine-tuning pretrained language models of varying parameter size, and applying both zero-shot and few-shot in-context learning on instruction-tuned language models. Although the latter were markedly outperformed by fine-tuned models, they still achieved promising results overall. Our results thus demonstrate the strong potential of different variants of large language models for the Split and Rephrase task, using relatively small amounts of training samples and model parameters overall.

{{</citation>}}


### (56/134) Entity or Relation Embeddings? An Analysis of Encoding Strategies for Relation Extraction (Frank Mtumbuka et al., 2023)

{{<citation>}}

Frank Mtumbuka, Steven Schockaert. (2023)  
**Entity or Relation Embeddings? An Analysis of Encoding Strategies for Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2312.11062v1)  

---


**ABSTRACT**  
Relation extraction is essentially a text classification problem, which can be tackled by fine-tuning a pre-trained language model (LM). However, a key challenge arises from the fact that relation extraction cannot straightforwardly be reduced to sequence or token classification. Existing approaches therefore solve the problem in an indirect way: they fine-tune an LM to learn embeddings of the head and tail entities, and then predict the relationship from these entity embeddings. Our hypothesis in this paper is that relation extraction models can be improved by capturing relationships in a more direct way. In particular, we experiment with appending a prompt with a [MASK] token, whose contextualised representation is treated as a relation embedding. While, on its own, this strategy significantly underperforms the aforementioned approach, we find that the resulting relation embeddings are highly complementary to what is captured by embeddings of the head and tail entity. By jointly considering both types of representations, we end up with a simple model that outperforms the state-of-the-art across several relation extraction benchmarks.

{{</citation>}}


### (57/134) TDeLTA: A Light-weight and Robust Table Detection Method based on Learning Text Arrangement (Yang Fan et al., 2023)

{{<citation>}}

Yang Fan, Xiangping Wu, Qingcai Chen, Heng Li, Yan Huang, Zhixiang Cai, Qitian Wu. (2023)  
**TDeLTA: A Light-weight and Robust Table Detection Method based on Learning Text Arrangement**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2312.11043v1)  

---


**ABSTRACT**  
The diversity of tables makes table detection a great challenge, leading to existing models becoming more tedious and complex. Despite achieving high performance, they often overfit to the table style in training set, and suffer from significant performance degradation when encountering out-of-distribution tables in other domains. To tackle this problem, we start from the essence of the table, which is a set of text arranged in rows and columns. Based on this, we propose a novel, light-weighted and robust Table Detection method based on Learning Text Arrangement, namely TDeLTA. TDeLTA takes the text blocks as input, and then models the arrangement of them with a sequential encoder and an attention module. To locate the tables precisely, we design a text-classification task, classifying the text blocks into 4 categories according to their semantic roles in the tables. Experiments are conducted on both the text blocks parsed from PDF and extracted by open-source OCR tools, respectively. Compared to several state-of-the-art methods, TDeLTA achieves competitive results with only 3.1M model parameters on the large-scale public datasets. Moreover, when faced with the cross-domain data under the 0-shot setting, TDeLTA outperforms baselines by a large margin of nearly 7%, which shows the strong robustness and transferability of the proposed model.

{{</citation>}}


### (58/134) Information Type Classification with Contrastive Task-Specialized Sentence Encoders (Philipp Seeberger et al., 2023)

{{<citation>}}

Philipp Seeberger, Tobias Bocklet, Korbinian Riedhammer. (2023)  
**Information Type Classification with Contrastive Task-Specialized Sentence Encoders**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11020v1)  

---


**ABSTRACT**  
User-generated information content has become an important information source in crisis situations. However, classification models suffer from noise and event-related biases which still poses a challenging task and requires sophisticated task-adaptation. To address these challenges, we propose the use of contrastive task-specialized sentence encoders for downstream classification. We apply the task-specialization on the CrisisLex, HumAID, and TrecIS information type classification tasks and show performance gains w.r.t. F1-score. Furthermore, we analyse the cross-corpus and cross-lingual capabilities for two German event relevancy classification datasets.

{{</citation>}}


### (59/134) VinaLLaMA: LLaMA-based Vietnamese Foundation Model (Quan Nguyen et al., 2023)

{{<citation>}}

Quan Nguyen, Huy Pham, Dung Dao. (2023)  
**VinaLLaMA: LLaMA-based Vietnamese Foundation Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2312.11011v1)  

---


**ABSTRACT**  
In this technical report, we present VinaLLaMA, an open-weight, state-of-the-art (SOTA) Large Language Model for the Vietnamese language, built upon LLaMA-2 with an additional 800 billion trained tokens. VinaLLaMA not only demonstrates fluency in Vietnamese but also exhibits a profound understanding of Vietnamese culture, making it a truly indigenous model. VinaLLaMA-7B-chat, trained on 1 million high-quality synthetic samples, achieves SOTA results on key benchmarks, including VLSP, VMLU, and Vicuna Benchmark Vietnamese, marking a significant advancement in the Vietnamese AI landscape and offering a versatile resource for various applications.

{{</citation>}}


### (60/134) Retrieval-Augmented Generation for Large Language Models: A Survey (Yunfan Gao et al., 2023)

{{<citation>}}

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Haofen Wang. (2023)  
**Retrieval-Augmented Generation for Large Language Models: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.10997v1)  

---


**ABSTRACT**  
Large language models (LLMs) demonstrate powerful capabilities, but they still face challenges in practical applications, such as hallucinations, slow knowledge updates, and lack of transparency in answers. Retrieval-Augmented Generation (RAG) refers to the retrieval of relevant information from external knowledge bases before answering questions with LLMs. RAG has been demonstrated to significantly enhance answer accuracy, reduce model hallucination, particularly for knowledge-intensive tasks. By citing sources, users can verify the accuracy of answers and increase trust in model outputs. It also facilitates knowledge updates and the introduction of domain-specific knowledge. RAG effectively combines the parameterized knowledge of LLMs with non-parameterized external knowledge bases, making it one of the most important methods for implementing large language models. This paper outlines the development paradigms of RAG in the era of LLMs, summarizing three paradigms: Naive RAG, Advanced RAG, and Modular RAG. It then provides a summary and organization of the three main components of RAG: retriever, generator, and augmentation methods, along with key technologies in each component. Furthermore, it discusses how to evaluate the effectiveness of RAG models, introducing two evaluation methods for RAG, emphasizing key metrics and abilities for evaluation, and presenting the latest automatic evaluation framework. Finally, potential future research directions are introduced from three aspects: vertical optimization, horizontal scalability, and the technical stack and ecosystem of RAG.

{{</citation>}}


### (61/134) Knowledge Graphs and Pre-trained Language Models enhanced Representation Learning for Conversational Recommender Systems (Zhangchi Qiu et al., 2023)

{{<citation>}}

Zhangchi Qiu, Ye Tao, Shirui Pan, Alan Wee-Chung Liew. (2023)  
**Knowledge Graphs and Pre-trained Language Models enhanced Representation Learning for Conversational Recommender Systems**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: Knowledge Graph, Language Model, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.10967v1)  

---


**ABSTRACT**  
Conversational recommender systems (CRS) utilize natural language interactions and dialogue history to infer user preferences and provide accurate recommendations. Due to the limited conversation context and background knowledge, existing CRSs rely on external sources such as knowledge graphs to enrich the context and model entities based on their inter-relations. However, these methods ignore the rich intrinsic information within entities. To address this, we introduce the Knowledge-Enhanced Entity Representation Learning (KERL) framework, which leverages both the knowledge graph and a pre-trained language model to improve the semantic understanding of entities for CRS. In our KERL framework, entity textual descriptions are encoded via a pre-trained language model, while a knowledge graph helps reinforce the representation of these entities. We also employ positional encoding to effectively capture the temporal information of entities in a conversation. The enhanced entity representation is then used to develop a recommender component that fuses both entity and contextual representations for more informed recommendations, as well as a dialogue component that generates informative entity-related information in the response text. A high-quality knowledge graph with aligned entity descriptions is constructed to facilitate our study, namely the Wiki Movie Knowledge Graph (WikiMKG). The experimental results show that KERL achieves state-of-the-art results in both recommendation and response generation tasks.

{{</citation>}}


### (62/134) Generative linguistic representation for spoken language identification (Peng Shen et al., 2023)

{{<citation>}}

Peng Shen, Xuguang Lu, Hisashi Kawai. (2023)  
**Generative linguistic representation for spoken language identification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2312.10964v1)  

---


**ABSTRACT**  
Effective extraction and application of linguistic features are central to the enhancement of spoken Language IDentification (LID) performance. With the success of recent large models, such as GPT and Whisper, the potential to leverage such pre-trained models for extracting linguistic features for LID tasks has become a promising area of research. In this paper, we explore the utilization of the decoder-based network from the Whisper model to extract linguistic features through its generative mechanism for improving the classification accuracy in LID tasks. We devised two strategies - one based on the language embedding method and the other focusing on direct optimization of LID outputs while simultaneously enhancing the speech recognition tasks. We conducted experiments on the large-scale multilingual datasets MLS, VoxLingua107, and CommonVoice to test our approach. The experimental results demonstrated the effectiveness of the proposed method on both in-domain and out-of-domain datasets for LID tasks.

{{</citation>}}


### (63/134) Aspect-Based Sentiment Analysis with Explicit Sentiment Augmentations (Jihong Ouyang et al., 2023)

{{<citation>}}

Jihong Ouyang, Zhiyao Yang, Silong Liang, Bing Wang, Yimeng Wang, Ximing Li. (2023)  
**Aspect-Based Sentiment Analysis with Explicit Sentiment Augmentations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Augmentation, Sentiment Analysis, T5  
[Paper Link](http://arxiv.org/abs/2312.10961v1)  

---


**ABSTRACT**  
Aspect-based sentiment analysis (ABSA), a fine-grained sentiment classification task, has received much attention recently. Many works investigate sentiment information through opinion words, such as ''good'' and ''bad''. However, implicit sentiment widely exists in the ABSA dataset, which refers to the sentence containing no distinct opinion words but still expresses sentiment to the aspect term. To deal with implicit sentiment, this paper proposes an ABSA method that integrates explicit sentiment augmentations. And we propose an ABSA-specific augmentation method to create such augmentations. Specifically, we post-trains T5 by rule-based data. We employ Syntax Distance Weighting and Unlikelihood Contrastive Regularization in the training procedure to guide the model to generate an explicit sentiment. Meanwhile, we utilize the Constrained Beam Search to ensure the augmentation sentence contains the aspect terms. We test ABSA-ESA on two of the most popular benchmarks of ABSA. The results show that ABSA-ESA outperforms the SOTA baselines on implicit and explicit sentiment accuracy.

{{</citation>}}


### (64/134) Regularized Conditional Alignment for Multi-Domain Text Classification (Juntao Hu et al., 2023)

{{<citation>}}

Juntao Hu, Yuan Wu. (2023)  
**Regularized Conditional Alignment for Multi-Domain Text Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: Text Classification  
[Paper Link](http://arxiv.org/abs/2312.11572v1)  

---


**ABSTRACT**  
The most successful multi-domain text classification (MDTC) approaches employ the shared-private paradigm to facilitate the enhancement of domain-invariant features through domain-specific attributes. Additionally, they employ adversarial training to align marginal feature distributions. Nevertheless, these methodologies encounter two primary challenges: (1) Neglecting class-aware information during adversarial alignment poses a risk of misalignment; (2) The limited availability of labeled data across multiple domains fails to ensure adequate discriminative capacity for the model. To tackle these issues, we propose a method called Regularized Conditional Alignment (RCA) to align the joint distributions of domains and classes, thus matching features within the same category and amplifying the discriminative qualities of acquired features. Moreover, we employ entropy minimization and virtual adversarial training to constrain the uncertainty of predictions pertaining to unlabeled data and enhance the model's robustness. Empirical results on two benchmark datasets demonstrate that our RCA approach outperforms state-of-the-art MDTC techniques.

{{</citation>}}


### (65/134) Generalized Category Discovery with Large Language Models in the Loop (Wenbin An et al., 2023)

{{<citation>}}

Wenbin An, Wenkai Shi, Feng Tian, Haonan Lin, QianYing Wang, Yaqiang Wu, Mingxiang Cai, Luyan Wang, Yan Chen, Haiping Zhu, Ping Chen. (2023)  
**Generalized Category Discovery with Large Language Models in the Loop**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Contrastive Learning, Language Model  
[Paper Link](http://arxiv.org/abs/2312.10897v1)  

---


**ABSTRACT**  
Generalized Category Discovery (GCD) is a crucial task that aims to recognize both known and novel categories from a set of unlabeled data by utilizing a few labeled data with only known categories. Due to the lack of supervision and category information, current methods usually perform poorly on novel categories and struggle to reveal semantic meanings of the discovered clusters, which limits their applications in the real world. To mitigate above issues, we propose Loop, an end-to-end active-learning framework that introduces Large Language Models (LLMs) into the training loop, which can boost model performance and generate category names without relying on any human efforts. Specifically, we first propose Local Inconsistent Sampling (LIS) to select samples that have a higher probability of falling to wrong clusters, based on neighborhood prediction consistency and entropy of cluster assignment probabilities. Then we propose a Scalable Query strategy to allow LLMs to choose true neighbors of the selected samples from multiple candidate samples. Based on the feedback from LLMs, we perform Refined Neighborhood Contrastive Learning (RNCL) to pull samples and their neighbors closer to learn clustering-friendly representations. Finally, we select representative samples from clusters corresponding to novel categories to allow LLMs to generate category names for them. Extensive experiments on three benchmark datasets show that Loop outperforms SOTA models by a large margin and generates accurate category names for the discovered clusters. We will release our code and data after publication.

{{</citation>}}


## cs.SE (4)



### (66/134) How Far Are We? The Triumphs and Trials of Generative AI in Learning Software Engineering (Rudrajit Choudhuri et al., 2023)

{{<citation>}}

Rudrajit Choudhuri, Dylan Liu, Igor Steinmacher, Marco Gerosa, Anita Sarma. (2023)  
**How Far Are We? The Triumphs and Trials of Generative AI in Learning Software Engineering**  

---
Primary Category: cs.SE  
Categories: cs-HC, cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.11719v1)  

---


**ABSTRACT**  
Conversational Generative AI (convo-genAI) is revolutionizing Software Engineering (SE) as engineers and academics embrace this technology in their work. However, there is a gap in understanding the current potential and pitfalls of this technology, specifically in supporting students in SE tasks. In this work, we evaluate through a between-subjects study (N=22) the effectiveness of ChatGPT, a convo-genAI platform, in assisting students in SE tasks. Our study did not find statistical differences in participants' productivity or self-efficacy when using ChatGPT as compared to traditional resources, but we found significantly increased frustration levels. Our study also revealed 5 distinct faults arising from violations of Human-AI interaction guidelines, which led to 7 different (negative) consequences on participants.

{{</citation>}}


### (67/134) PPT4J: Patch Presence Test for Java Binaries (Zhiyuan Pan et al., 2023)

{{<citation>}}

Zhiyuan Pan, Xing Hu, Xin Xia, Xian Zhan, David Lo, Xiaohu Yang. (2023)  
**PPT4J: Patch Presence Test for Java Binaries**  

---
Primary Category: cs.SE  
Categories: cs-CR, cs-SE, cs.SE  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.11013v1)  

---


**ABSTRACT**  
The number of vulnerabilities reported in open source software has increased substantially in recent years. Security patches provide the necessary measures to protect software from attacks and vulnerabilities. In practice, it is difficult to identify whether patches have been integrated into software, especially if we only have binary files. Therefore, the ability to test whether a patch is applied to the target binary, a.k.a. patch presence test, is crucial for practitioners. However, it is challenging to obtain accurate semantic information from patches, which could lead to incorrect results.   In this paper, we propose a new patch presence test framework named PPT4J ($\textbf{P}$atch $\textbf{P}$resence $\textbf{T}$est $\textbf{for}$ $\textbf{J}$ava Binaries). PPT4J is designed for open-source Java libraries. It takes Java binaries (i.e. bytecode files) as input, extracts semantic information from patches, and uses feature-based techniques to identify patch lines in the binaries. To evaluate the effectiveness of our proposed approach PPT4J, we construct a dataset with binaries that include 110 vulnerabilities. The results show that PPT4J achieves an F1 score of 98.5% with reasonable efficiency, improving the baseline by 15.6%. Furthermore, we conduct an in-the-wild evaluation of PPT4J on JetBrains IntelliJ IDEA. The results suggest that a third-party library included in the software is not patched for two CVEs, and we have reported this potential security problem to the vendor.

{{</citation>}}


### (68/134) APIDocBooster: An Extract-Then-Abstract Framework Leveraging Large Language Models for Augmenting API Documentation (Chengran Yang et al., 2023)

{{<citation>}}

Chengran Yang, Jiakun Liu, Bowen Xu, Christoph Treude, Yunbo Lyu, Ming Li, David Lo. (2023)  
**APIDocBooster: An Extract-Then-Abstract Framework Leveraging Large Language Models for Augmenting API Documentation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.10934v1)  

---


**ABSTRACT**  
API documentation is often the most trusted resource for programming. Many approaches have been proposed to augment API documentation by summarizing complementary information from external resources such as Stack Overflow. Existing extractive-based summarization approaches excel in producing faithful summaries that accurately represent the source content without input length restrictions. Nevertheless, they suffer from inherent readability limitations. On the other hand, our empirical study on the abstractive-based summarization method, i.e., GPT-4, reveals that GPT-4 can generate coherent and concise summaries but presents limitations in terms of informativeness and faithfulness.   We introduce APIDocBooster, an extract-then-abstract framework that seamlessly fuses the advantages of both extractive (i.e., enabling faithful summaries without length limitation) and abstractive summarization (i.e., producing coherent and concise summaries). APIDocBooster consists of two stages: (1) \textbf{C}ontext-aware \textbf{S}entence \textbf{S}ection \textbf{C}lassification (CSSC) and (2) \textbf{UP}date \textbf{SUM}marization (UPSUM). CSSC classifies API-relevant information collected from multiple sources into API documentation sections. UPSUM first generates extractive summaries distinct from the original API documentation and then generates abstractive summaries guided by extractive summaries through in-context learning.   To enable automatic evaluation of APIDocBooster, we construct the first dataset for API document augmentation. Our automatic evaluation results reveal that each stage in APIDocBooster outperforms its baselines by a large margin. Our human evaluation also demonstrates the superiority of APIDocBooster over GPT-4 and shows that it improves informativeness, relevance, and faithfulness by 13.89\%, 15.15\%, and 30.56\%, respectively.

{{</citation>}}


### (69/134) Code Ownership in Open-Source AI Software Security (Jiawen Wen et al., 2023)

{{<citation>}}

Jiawen Wen, Dong Yuan, Lei Ma, Huaming Chen. (2023)  
**Code Ownership in Open-Source AI Software Security**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI, Security  
[Paper Link](http://arxiv.org/abs/2312.10861v1)  

---


**ABSTRACT**  
As open-source AI software projects become an integral component in the AI software development, it is critical to develop a novel methods to ensure and measure the security of the open-source projects for developers. Code ownership, pivotal in the evolution of such projects, offers insights into developer engagement and potential vulnerabilities. In this paper, we leverage the code ownership metrics to empirically investigate the correlation with the latent vulnerabilities across five prominent open-source AI software projects. The findings from the large-scale empirical study suggest a positive relationship between high-level ownership (characterised by a limited number of minor contributors) and a decrease in vulnerabilities. Furthermore, we innovatively introduce the time metrics, anchored on the project's duration, individual source code file timelines, and the count of impacted releases. These metrics adeptly categorise distinct phases of open-source AI software projects and their respective vulnerability intensities. With these novel code ownership metrics, we have implemented a Python-based command-line application to aid project curators and quality assurance professionals in evaluating and benchmarking their on-site projects. We anticipate this work will embark a continuous research development for securing and measuring open-source AI project security.

{{</citation>}}


## cs.CV (25)



### (70/134) Squeezed Edge YOLO: Onboard Object Detection on Edge Devices (Edward Humes et al., 2023)

{{<citation>}}

Edward Humes, Mozhgan Navardi, Tinoosh Mohsenin. (2023)  
**Squeezed Edge YOLO: Onboard Object Detection on Edge Devices**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.11716v1)  

---


**ABSTRACT**  
Demand for efficient onboard object detection is increasing due to its key role in autonomous navigation. However, deploying object detection models such as YOLO on resource constrained edge devices is challenging due to the high computational requirements of such models. In this paper, an compressed object detection model named Squeezed Edge YOLO is examined. This model is compressed and optimized to kilobytes of parameters in order to fit onboard such edge devices. To evaluate Squeezed Edge YOLO, two use cases - human and shape detection - are used to show the model accuracy and performance. Moreover, the model is deployed onboard a GAP8 processor with 8 RISC-V cores and an NVIDIA Jetson Nano with 4GB of memory. Experimental results show Squeezed Edge YOLO model size is optimized by a factor of 8x which leads to 76% improvements in energy efficiency and 3.3x faster throughout.

{{</citation>}}


### (71/134) HAAR: Text-Conditioned Generative Model of 3D Strand-based Human Hairstyles (Vanessa Sklyarova et al., 2023)

{{<citation>}}

Vanessa Sklyarova, Egor Zakharov, Otmar Hilliges, Michael J. Black, Justus Thies. (2023)  
**HAAR: Text-Conditioned Generative Model of 3D Strand-based Human Hairstyles**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2312.11666v1)  

---


**ABSTRACT**  
We present HAAR, a new strand-based generative model for 3D human hairstyles. Specifically, based on textual inputs, HAAR produces 3D hairstyles that could be used as production-level assets in modern computer graphics engines. Current AI-based generative models take advantage of powerful 2D priors to reconstruct 3D content in the form of point clouds, meshes, or volumetric functions. However, by using the 2D priors, they are intrinsically limited to only recovering the visual parts. Highly occluded hair structures can not be reconstructed with those methods, and they only model the ''outer shell'', which is not ready to be used in physics-based rendering or simulation pipelines. In contrast, we propose a first text-guided generative method that uses 3D hair strands as an underlying representation. Leveraging 2D visual question-answering (VQA) systems, we automatically annotate synthetic hair models that are generated from a small set of artist-created hairstyles. This allows us to train a latent diffusion model that operates in a common hairstyle UV space. In qualitative and quantitative studies, we demonstrate the capabilities of the proposed model and compare it to existing hairstyle generation approaches.

{{</citation>}}


### (72/134) Orientation-Constrained System for Lamp Detection in Buildings Based on Computer Vision (Francisco Troncoso-Pastoriza et al., 2023)

{{<citation>}}

Francisco Troncoso-Pastoriza, Pablo Eguía-Oller, Rebeca P. Díaz-Redondo, Enrique Granada-Álvarez, Aitor Erkoreka. (2023)  
**Orientation-Constrained System for Lamp Detection in Buildings Based on Computer Vision**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.11380v1)  

---


**ABSTRACT**  
Computer vision is used in this work to detect lighting elements in buildings with the goal of improving the accuracy of previous methods to provide a precise inventory of the location and state of lamps. Using the framework developed in our previous works, we introduce two new modifications to enhance the system: first, a constraint on the orientation of the detected poses in the optimization methods for both the initial and the refined estimates based on the geometric information of the building information modelling (BIM) model; second, an additional reprojection error filtering step to discard the erroneous poses introduced with the orientation restrictions, keeping the identification and localization errors low while greatly increasing the number of detections. These~enhancements are tested in five different case studies with more than 30,000 images, with results showing improvements in the number of detections, the percentage of correct model and state identifications, and the distance between detections and reference positions

{{</citation>}}


### (73/134) CaRe-CNN: Cascading Refinement CNN for Myocardial Infarct Segmentation with Microvascular Obstructions (Franz Thaler et al., 2023)

{{<citation>}}

Franz Thaler, Matthias A. F. Gsell, Gernot Plank, Martin Urschler. (2023)  
**CaRe-CNN: Cascading Refinement CNN for Myocardial Infarct Segmentation with Microvascular Obstructions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11315v2)  

---


**ABSTRACT**  
Late gadolinium enhanced (LGE) magnetic resonance (MR) imaging is widely established to assess the viability of myocardial tissue of patients after acute myocardial infarction (MI). We propose the Cascading Refinement CNN (CaRe-CNN), which is a fully 3D, end-to-end trained, 3-stage CNN cascade that exploits the hierarchical structure of such labeled cardiac data. Throughout the three stages of the cascade, the label definition changes and CaRe-CNN learns to gradually refine its intermediate predictions accordingly. Furthermore, to obtain more consistent qualitative predictions, we propose a series of post-processing steps that take anatomical constraints into account. Our CaRe-CNN was submitted to the FIMH 2023 MYOSAIQ challenge, where it ranked second out of 18 participating teams. CaRe-CNN showed great improvements most notably when segmenting the difficult but clinically most relevant myocardial infarct tissue (MIT) as well as microvascular obstructions (MVO). When computing the average scores over all labels, our method obtained the best score in eight out of ten metrics. Thus, accurate cardiac segmentation after acute MI via our CaRe-CNN allows generating patient-specific models of the heart serving as an important step towards personalized medicine.

{{</citation>}}


### (74/134) The Ultimate Combo: Boosting Adversarial Example Transferability by Composing Data Augmentations (Zebin Yun et al., 2023)

{{<citation>}}

Zebin Yun, Achi-Or Weingarten, Eyal Ronen, Mahmood Sharif. (2023)  
**The Ultimate Combo: Boosting Adversarial Example Transferability by Composing Data Augmentations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, ImageNet  
[Paper Link](http://arxiv.org/abs/2312.11309v1)  

---


**ABSTRACT**  
Transferring adversarial examples (AEs) from surrogate machine-learning (ML) models to target models is commonly used in black-box adversarial robustness evaluation. Attacks leveraging certain data augmentation, such as random resizing, have been found to help AEs generalize from surrogates to targets. Yet, prior work has explored limited augmentations and their composition. To fill the gap, we systematically studied how data augmentation affects transferability. Particularly, we explored 46 augmentation techniques of seven categories originally proposed to help ML models generalize to unseen benign samples, and assessed how they impact transferability, when applied individually or composed. Performing exhaustive search on a small subset of augmentation techniques and genetic search on all techniques, we identified augmentation combinations that can help promote transferability. Extensive experiments with the ImageNet and CIFAR-10 datasets and 18 models showed that simple color-space augmentations (e.g., color to greyscale) outperform the state of the art when combined with standard augmentations, such as translation and scaling. Additionally, we discovered that composing augmentations impacts transferability mostly monotonically (i.e., more methods composed $\rightarrow$ $\ge$ transferability). We also found that the best composition significantly outperformed the state of the art (e.g., 93.7% vs. $\le$ 82.7% average transferability on ImageNet from normally trained surrogates to adversarially trained targets). Lastly, our theoretical analysis, backed up by empirical evidence, intuitively explain why certain augmentations help improve transferability.

{{</citation>}}


### (75/134) Spherical Mask: Coarse-to-Fine 3D Point Cloud Instance Segmentation with Spherical Representation (Sangyun Shin et al., 2023)

{{<citation>}}

Sangyun Shin, Kaichen Zhou, Madhu Vankadari, Andrew Markham, Niki Trigoni. (2023)  
**Spherical Mask: Coarse-to-Fine 3D Point Cloud Instance Segmentation with Spherical Representation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.11269v1)  

---


**ABSTRACT**  
Coarse-to-fine 3D instance segmentation methods show weak performances compared to recent Grouping-based, Kernel-based and Transformer-based methods. We argue that this is due to two limitations: 1) Instance size overestimation by axis-aligned bounding box(AABB) 2) False negative error accumulation from inaccurate box to the refinement phase. In this work, we introduce Spherical Mask, a novel coarse-to-fine approach based on spherical representation, overcoming those two limitations with several benefits. Specifically, our coarse detection estimates each instance with a 3D polygon using a center and radial distance predictions, which avoids excessive size estimation of AABB. To cut the error propagation in the existing coarse-to-fine approaches, we virtually migrate points based on the polygon, allowing all foreground points, including false negatives, to be refined. During inference, the proposal and point migration modules run in parallel and are assembled to form binary masks of instances. We also introduce two margin-based losses for the point migration to enforce corrections for the false positives/negatives and cohesion of foreground points, significantly improving the performance. Experimental results from three datasets, such as ScanNetV2, S3DIS, and STPLS3D, show that our proposed method outperforms existing works, demonstrating the effectiveness of the new instance representation with spherical coordinates.

{{</citation>}}


### (76/134) Leveraging Normalization Layer in Adapters With Progressive Learning and Adaptive Distillation for Cross-Domain Few-Shot Learning (Yongjin Yang et al., 2023)

{{<citation>}}

Yongjin Yang, Taehyeon Kim, Se-Young Yun. (2023)  
**Leveraging Normalization Layer in Adapters With Progressive Learning and Adaptive Distillation for Cross-Domain Few-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.11260v1)  

---


**ABSTRACT**  
Cross-domain few-shot learning presents a formidable challenge, as models must be trained on base classes and then tested on novel classes from various domains with only a few samples at hand. While prior approaches have primarily focused on parameter-efficient methods of using adapters, they often overlook two critical issues: shifts in batch statistics and noisy sample statistics arising from domain discrepancy variations. In this paper, we introduce a novel generic framework that leverages normalization layer in adapters with Progressive Learning and Adaptive Distillation (ProLAD), marking two principal contributions. First, our methodology utilizes two separate adapters: one devoid of a normalization layer, which is more effective for similar domains, and another embedded with a normalization layer, designed to leverage the batch statistics of the target domain, thus proving effective for dissimilar domains. Second, to address the pitfalls of noisy statistics, we deploy two strategies: a progressive training of the two adapters and an adaptive distillation technique derived from features determined by the model solely with the adapter devoid of a normalization layer. Through this adaptive distillation, our approach functions as a modulator, controlling the primary adapter for adaptation, based on each domain. Evaluations on standard cross-domain few-shot learning benchmarks confirm that our technique outperforms existing state-of-the-art methodologies.

{{</citation>}}


### (77/134) Challenges in Multi-centric Generalization: Phase and Step Recognition in Roux-en-Y Gastric Bypass Surgery (Joel L. Lavanchy et al., 2023)

{{<citation>}}

Joel L. Lavanchy, Sanat Ramesh, Diego Dall'Alba, Cristians Gonzalez, Paolo Fiorini, Beat Muller-Stich, Philipp C. Nett, Jacques Marescaux, Didier Mutter, Nicolas Padoy. (2023)  
**Challenges in Multi-centric Generalization: Phase and Step Recognition in Roux-en-Y Gastric Bypass Surgery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11250v1)  

---


**ABSTRACT**  
Most studies on surgical activity recognition utilizing Artificial intelligence (AI) have focused mainly on recognizing one type of activity from small and mono-centric surgical video datasets. It remains speculative whether those models would generalize to other centers. In this work, we introduce a large multi-centric multi-activity dataset consisting of 140 videos (MultiBypass140) of laparoscopic Roux-en-Y gastric bypass (LRYGB) surgeries performed at two medical centers: the University Hospital of Strasbourg (StrasBypass70) and Inselspital, Bern University Hospital (BernBypass70). The dataset has been fully annotated with phases and steps. Furthermore, we assess the generalizability and benchmark different deep learning models in 7 experimental studies: 1) Training and evaluation on BernBypass70; 2) Training and evaluation on StrasBypass70; 3) Training and evaluation on the MultiBypass140; 4) Training on BernBypass70, evaluation on StrasBypass70; 5) Training on StrasBypass70, evaluation on BernBypass70; Training on MultiBypass140, evaluation 6) on BernBypass70 and 7) on StrasBypass70. The model's performance is markedly influenced by the training data. The worst results were obtained in experiments 4) and 5) confirming the limited generalization capabilities of models trained on mono-centric data. The use of multi-centric training data, experiments 6) and 7), improves the generalization capabilities of the models, bringing them beyond the level of independent mono-centric training and validation (experiments 1) and 2)). MultiBypass140 shows considerable variation in surgical technique and workflow of LRYGB procedures between centers. Therefore, generalization experiments demonstrate a remarkable difference in model performance. These results highlight the importance of multi-centric datasets for AI model generalization to account for variance in surgical technique and workflows.

{{</citation>}}


### (78/134) Decoupled Knowledge with Ensemble Learning for Online Distillation (Baitan Shao et al., 2023)

{{<citation>}}

Baitan Shao, Ying Chen. (2023)  
**Decoupled Knowledge with Ensemble Learning for Online Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.11218v1)  

---


**ABSTRACT**  
Offline distillation is a two-stage pipeline that requires expensive resources to train a teacher network and then distill the knowledge to a student for deployment. Online knowledge distillation, on the other hand, is a one-stage strategy that alleviates the requirement with mutual learning and collaborative learning. Recent peer collaborative learning (PCL) integrates online ensemble, collaboration of base networks and temporal mean teacher to construct effective knowledge. However, the model collapses occasionally in PCL due to high homogenization between the student and the teacher. In this paper, the cause of the high homogenization is analyzed and the solution is presented. A decoupled knowledge for online knowledge distillation is generated by an independent teacher, separate from the student. Such design can increase the diversity between the networks and reduce the possibility of model collapse. To obtain early decoupled knowledge, an initialization scheme for the teacher is devised, and a 2D geometry-based analysis experiment is conducted under ideal conditions to showcase the effectiveness of this scheme. Moreover, to improve the teacher's supervisory resilience, a decaying ensemble scheme is devised. It assembles the knowledge of the teacher to which a dynamic weight which is large at the start of the training and gradually decreases with the training process is assigned. The assembled knowledge serves as a strong teacher during the early training and the decreased-weight-assembled knowledge can eliminate the distribution deviation under the potentially overfitted teacher's supervision. A Monte Carlo-based simulation is conducted to evaluate the convergence. Extensive experiments on CIFAR-10, CIFAR-100 and TinyImageNet show the superiority of our method. Ablation studies and further analysis demonstrate the effectiveness.

{{</citation>}}


### (79/134) Cross-Age Contrastive Learning for Age-Invariant Face Recognition (Haoyi Wang et al., 2023)

{{<citation>}}

Haoyi Wang, Victor Sanchez, Chang-Tsun Li. (2023)  
**Cross-Age Contrastive Learning for Age-Invariant Face Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.11195v1)  

---


**ABSTRACT**  
Cross-age facial images are typically challenging and expensive to collect, making noise-free age-oriented datasets relatively small compared to widely-used large-scale facial datasets. Additionally, in real scenarios, images of the same subject at different ages are usually hard or even impossible to obtain. Both of these factors lead to a lack of supervised data, which limits the versatility of supervised methods for age-invariant face recognition, a critical task in applications such as security and biometrics. To address this issue, we propose a novel semi-supervised learning approach named Cross-Age Contrastive Learning (CACon). Thanks to the identity-preserving power of recent face synthesis models, CACon introduces a new contrastive learning method that leverages an additional synthesized sample from the input image. We also propose a new loss function in association with CACon to perform contrastive learning on a triplet of samples. We demonstrate that our method not only achieves state-of-the-art performance in homogeneous-dataset experiments on several age-invariant face recognition benchmarks but also outperforms other methods by a large margin in cross-dataset experiments.

{{</citation>}}


### (80/134) Research on Multilingual Natural Scene Text Detection Algorithm (Tao Wang, 2023)

{{<citation>}}

Tao Wang. (2023)  
**Research on Multilingual Natural Scene Text Detection Algorithm**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Multilingual, Semantic Segmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2312.11153v1)  

---


**ABSTRACT**  
Natural scene text detection is a significant challenge in computer vision, with tremendous potential applications in multilingual, diverse, and complex text scenarios. We propose a multilingual text detection model to address the issues of low accuracy and high difficulty in detecting multilingual text in natural scenes. In response to the challenges posed by multilingual text images with multiple character sets and various font styles, we introduce the SFM Swin Transformer feature extraction network to enhance the model's robustness in detecting characters and fonts across different languages. Dealing with the considerable variation in text scales and complex arrangements in natural scene text images, we present the AS-HRFPN feature fusion network by incorporating an Adaptive Spatial Feature Fusion module and a Spatial Pyramid Pooling module. The feature fusion network improvements enhance the model's ability to detect text sizes and orientations. Addressing diverse backgrounds and font variations in multilingual scene text images is a challenge for existing methods. Limited local receptive fields hinder detection performance. To overcome this, we propose a Global Semantic Segmentation Branch, extracting and preserving global features for more effective text detection, aligning with the need for comprehensive information. In this study, we collected and built a real-world multilingual natural scene text image dataset and conducted comprehensive experiments and analyses. The experimental results demonstrate that the proposed algorithm achieves an F-measure of 85.02\%, which is 4.71\% higher than the baseline model. We also conducted extensive cross-dataset validation on MSRA-TD500, ICDAR2017MLT, and ICDAR2015 datasets to verify the generality of our approach. The code and dataset can be found at https://github.com/wangmelon/CEMLT.

{{</citation>}}


### (81/134) OsmLocator: locating overlapping scatter marks by simulated annealing on clustering-based re-visualization (Yuming Qiu et al., 2023)

{{<citation>}}

Yuming Qiu, Aleksandra Pizurica, Qi Ming, Nicolas Nadisic. (2023)  
**OsmLocator: locating overlapping scatter marks by simulated annealing on clustering-based re-visualization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11146v1)  

---


**ABSTRACT**  
Automated mark localization in scatter images, greatly helpful for discovering knowledge and understanding enormous document images and reasoning in visual question answering AI systems, is a highly challenging problem because of the ubiquity of overlapping marks. Locating overlapping marks faces many difficulties such as no texture, less contextual information, hallow shape and tiny size. Here, we formulate it as a combinatorial optimization problem on clustering-based re-visualization, to locate scatter marks by finding the status of multi-variables when an objective function reaches a minimum. The objective function is constructed on difference between binarized scatter images and corresponding re-visualization based on their clustering. Fundamentally, re-visualization tries to redraw a new scatter graph only taking a rasterized scatter image as an input, and clustering is employed to provide the information for such re-visualization. This method could stably locate severely-overlapping, variable-size and variable-shape marks in scatter images without dependence of any training dataset or reference. Meanwhile, we propose an adaptive variant of simulated annealing which can works on various connected regions. In addition, we especially built a dataset named SML2023 containing hundreds of scatter images with different markers and various levels of overlapping severity, and tested the proposed method and compared it to existing methods. The results show that it can accurately locate most marks in scatter images with different overlapping severity and marker types, with about 0.3 absolute increase on an assignment-cost-based metric in comparison with state-of-the-art methods. This work is of value to data mining on massive web pages and literatures, and shedding new light on image measurement such as bubble counting.

{{</citation>}}


### (82/134) Unleashing the Power of CNN and Transformer for Balanced RGB-Event Video Recognition (Xiao Wang et al., 2023)

{{<citation>}}

Xiao Wang, Yao Rong, Shiao Wang, Yuan Chen, Zhe Wu, Bo Jiang, Yonghong Tian, Jin Tang. (2023)  
**Unleashing the Power of CNN and Transformer for Balanced RGB-Event Video Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.11128v1)  

---


**ABSTRACT**  
Pattern recognition based on RGB-Event data is a newly arising research topic and previous works usually learn their features using CNN or Transformer. As we know, CNN captures the local features well and the cascaded self-attention mechanisms are good at extracting the long-range global relations. It is intuitive to combine them for high-performance RGB-Event based video recognition, however, existing works fail to achieve a good balance between the accuracy and model parameters, as shown in Fig.~\ref{firstimage}. In this work, we propose a novel RGB-Event based recognition framework termed TSCFormer, which is a relatively lightweight CNN-Transformer model. Specifically, we mainly adopt the CNN as the backbone network to first encode both RGB and Event data. Meanwhile, we initialize global tokens as the input and fuse them with RGB and Event features using the BridgeFormer module. It captures the global long-range relations well between both modalities and maintains the simplicity of the whole model architecture at the same time. The enhanced features will be projected and fused into the RGB and Event CNN blocks, respectively, in an interactive manner using F2E and F2V modules. Similar operations are conducted for other CNN blocks to achieve adaptive fusion and local-global feature enhancement under different resolutions. Finally, we concatenate these three features and feed them into the classification head for pattern recognition. Extensive experiments on two large-scale RGB-Event benchmark datasets (PokerEvent and HARDVS) fully validated the effectiveness of our proposed TSCFormer. The source code and pre-trained models will be released at https://github.com/Event-AHU/TSCFormer.

{{</citation>}}


### (83/134) Hyperspectral Image Reconstruction via Combinatorial Embedding of Cross-Channel Spatio-Spectral Clues (Xingxing Yang et al., 2023)

{{<citation>}}

Xingxing Yang, Jie Chen, Zaifeng Yang. (2023)  
**Hyperspectral Image Reconstruction via Combinatorial Embedding of Cross-Channel Spatio-Spectral Clues**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.11119v1)  

---


**ABSTRACT**  
Existing learning-based hyperspectral reconstruction methods show limitations in fully exploiting the information among the hyperspectral bands. As such, we propose to investigate the chromatic inter-dependencies in their respective hyperspectral embedding space. These embedded features can be fully exploited by querying the inter-channel correlations in a combinatorial manner, with the unique and complementary information efficiently fused into the final prediction. We found such independent modeling and combinatorial excavation mechanisms are extremely beneficial to uncover marginal spectral features, especially in the long wavelength bands. In addition, we have proposed a spatio-spectral attention block and a spectrum-fusion attention module, which greatly facilitates the excavation and fusion of information at both semantically long-range levels and fine-grained pixel levels across all dimensions. Extensive quantitative and qualitative experiments show that our method (dubbed CESST) achieves SOTA performance. Code for this project is at: https://github.com/AlexYangxx/CESST.

{{</citation>}}


### (84/134) ConDaFormer: Disassembled Transformer with Local Structure Enhancement for 3D Point Cloud Understanding (Lunhao Duan et al., 2023)

{{<citation>}}

Lunhao Duan, Shanshan Zhao, Nan Xue, Mingming Gong, Gui-Song Xia, Dacheng Tao. (2023)  
**ConDaFormer: Disassembled Transformer with Local Structure Enhancement for 3D Point Cloud Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.11112v1)  

---


**ABSTRACT**  
Transformers have been recently explored for 3D point cloud understanding with impressive progress achieved. A large number of points, over 0.1 million, make the global self-attention infeasible for point cloud data. Thus, most methods propose to apply the transformer in a local region, e.g., spherical or cubic window. However, it still contains a large number of Query-Key pairs, which requires high computational costs. In addition, previous methods usually learn the query, key, and value using a linear projection without modeling the local 3D geometric structure. In this paper, we attempt to reduce the costs and model the local geometry prior by developing a new transformer block, named ConDaFormer. Technically, ConDaFormer disassembles the cubic window into three orthogonal 2D planes, leading to fewer points when modeling the attention in a similar range. The disassembling operation is beneficial to enlarging the range of attention without increasing the computational complexity, but ignores some contexts. To provide a remedy, we develop a local structure enhancement strategy that introduces a depth-wise convolution before and after the attention. This scheme can also capture the local geometric information. Taking advantage of these designs, ConDaFormer captures both long-range contextual information and local priors. The effectiveness is demonstrated by experimental results on several 3D point cloud understanding benchmarks. Code is available at https://github.com/LHDuan/ConDaFormer .

{{</citation>}}


### (85/134) Advancing Image Retrieval with Few-Shot Learning and Relevance Feedback (Boaz Lerner et al., 2023)

{{<citation>}}

Boaz Lerner, Nir Darshan, Rami Ben-Ari. (2023)  
**Advancing Image Retrieval with Few-Shot Learning and Relevance Feedback**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.11078v1)  

---


**ABSTRACT**  
With such a massive growth in the number of images stored, efficient search in a database has become a crucial endeavor managed by image retrieval systems. Image Retrieval with Relevance Feedback (IRRF) involves iterative human interaction during the retrieval process, yielding more meaningful outcomes. This process can be generally cast as a binary classification problem with only {\it few} labeled samples derived from user feedback. The IRRF task frames a unique few-shot learning characteristics including binary classification of imbalanced and asymmetric classes, all in an open-set regime. In this paper, we study this task through the lens of few-shot learning methods. We propose a new scheme based on a hyper-network, that is tailored to the task and facilitates swift adjustment to user feedback. Our approach's efficacy is validated through comprehensive evaluations on multiple benchmarks and two supplementary tasks, supported by theoretical analysis. We demonstrate the advantage of our model over strong baselines on 4 different datasets in IRRF, addressing also retrieval of images with multiple objects. Furthermore, we show that our method can attain SoTA results in few-shot one-class classification and reach comparable results in binary classification task of few-shot open-set recognition.

{{</citation>}}


### (86/134) Multi-Correlation Siamese Transformer Network with Dense Connection for 3D Single Object Tracking (Shihao Feng et al., 2023)

{{<citation>}}

Shihao Feng, Pengpeng Liang, Jin Gao, Erkang Cheng. (2023)  
**Multi-Correlation Siamese Transformer Network with Dense Connection for 3D Single Object Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.11051v1)  

---


**ABSTRACT**  
Point cloud-based 3D object tracking is an important task in autonomous driving. Though great advances regarding Siamese-based 3D tracking have been made recently, it remains challenging to learn the correlation between the template and search branches effectively with the sparse LIDAR point cloud data. Instead of performing correlation of the two branches at just one point in the network, in this paper, we present a multi-correlation Siamese Transformer network that has multiple stages and carries out feature correlation at the end of each stage based on sparse pillars. More specifically, in each stage, self-attention is first applied to each branch separately to capture the non-local context information. Then, cross-attention is used to inject the template information into the search area. This strategy allows the feature learning of the search area to be aware of the template while keeping the individual characteristics of the template intact. To enable the network to easily preserve the information learned at different stages and ease the optimization, for the search area, we densely connect the initial input sparse pillars and the output of each stage to all subsequent stages and the target localization network, which converts pillars to bird's eye view (BEV) feature maps and predicts the state of the target with a small densely connected convolution network. Deep supervision is added to each stage to further boost the performance as well. The proposed algorithm is evaluated on the popular KITTI, nuScenes, and Waymo datasets, and the experimental results show that our method achieves promising performance compared with the state-of-the-art. Ablation study that shows the effectiveness of each component is provided as well. Code is available at https://github.com/liangp/MCSTN-3DSOT.

{{</citation>}}


### (87/134) Long-Tailed 3D Detection via 2D Late Fusion (Yechi Ma et al., 2023)

{{<citation>}}

Yechi Ma, Neehar Peri, Shuoquan Wei, Wei Hua, Deva Ramanan, Yanan Li, Shu Kong. (2023)  
**Long-Tailed 3D Detection via 2D Late Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.10986v1)  

---


**ABSTRACT**  
Autonomous vehicles (AVs) must accurately detect objects from both common and rare classes for safe navigation, motivating the problem of Long-Tailed 3D Object Detection (LT3D). Contemporary LiDAR-based 3D detectors perform poorly on rare classes (e.g., CenterPoint only achieves 5.1 AP on stroller) as it is difficult to recognize objects from sparse LiDAR points alone. RGB images provide visual evidence to help resolve such ambiguities, motivating the study of RGB-LiDAR fusion. In this paper, we delve into a simple late-fusion framework that ensembles independently trained RGB and LiDAR detectors. Unlike recent end-to-end methods which require paired multi-modal training data, our late-fusion approach can easily leverage large-scale uni-modal datasets, significantly improving rare class detection.In particular, we examine three critical components in this late-fusion framework from first principles, including whether to train 2D or 3D RGB detectors, whether to match RGB and LiDAR detections in 3D or the projected 2D image plane, and how to fuse matched detections.Extensive experiments reveal that 2D RGB detectors achieve better recognition accuracy than 3D RGB detectors, matching on the 2D image plane mitigates depth estimation errors, and fusing scores probabilistically with calibration leads to state-of-the-art LT3D performance. Our late-fusion approach achieves 51.4 mAP on the established nuScenes LT3D benchmark, improving over prior work by 5.9 mAP.

{{</citation>}}


### (88/134) MatchDet: A Collaborative Framework for Image Matching and Object Detection (Jinxiang Lai et al., 2023)

{{<citation>}}

Jinxiang Lai, Wenlong Wu, Bin-Bin Gao, Jun Liu, Jiawei Zhan, Congchong Nie, Yi Zeng, Chengjie Wang. (2023)  
**MatchDet: A Collaborative Framework for Image Matching and Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Object Detection  
[Paper Link](http://arxiv.org/abs/2312.10983v1)  

---


**ABSTRACT**  
Image matching and object detection are two fundamental and challenging tasks, while many related applications consider them two individual tasks (i.e. task-individual). In this paper, a collaborative framework called MatchDet (i.e. task-collaborative) is proposed for image matching and object detection to obtain mutual improvements. To achieve the collaborative learning of the two tasks, we propose three novel modules, including a Weighted Spatial Attention Module (WSAM) for Detector, and Weighted Attention Module (WAM) and Box Filter for Matcher. Specifically, the WSAM highlights the foreground regions of target image to benefit the subsequent detector, the WAM enhances the connection between the foreground regions of pair images to ensure high-quality matches, and Box Filter mitigates the impact of false matches. We evaluate the approaches on a new benchmark with two datasets called Warp-COCO and miniScanNet. Experimental results show our approaches are effective and achieve competitive improvements.

{{</citation>}}


### (89/134) Liquid Leak Detection Using Thermal Images (Kalpak Bansod et al., 2023)

{{<citation>}}

Kalpak Bansod, Yanshan Wan, Yugesh Rai. (2023)  
**Liquid Leak Detection Using Thermal Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.10980v1)  

---


**ABSTRACT**  
This paper presents a comprehensive solution to address the critical challenge of liquid leaks in the oil and gas industry, leveraging advanced computer vision and deep learning methodologies. Employing You Only Look Once (YOLO) and Real-Time Detection Transformer (RT DETR) models, our project focuses on enhancing early identification of liquid leaks in key infrastructure components such as pipelines, pumps, and tanks. Through the integration of surveillance thermal cameras and sensors, the combined YOLO and RT DETR models demonstrate remarkable efficacy in the continuous monitoring and analysis of visual data within oil and gas facilities. YOLO's real-time object detection capabilities swiftly recognize leaks and their patterns, while RT DETR excels in discerning specific leak-related features, particularly in thermal images. This approach significantly improves the accuracy and speed of leak detection, ultimately mitigating environmental and financial risks associated with liquid leaks.

{{</citation>}}


### (90/134) A Multimodal Approach for Advanced Pest Detection and Classification (Jinli Duan et al., 2023)

{{<citation>}}

Jinli Duan, Haoyu Ding, Sung Kim. (2023)  
**A Multimodal Approach for Advanced Pest Detection and Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.10948v1)  

---


**ABSTRACT**  
This paper presents a novel multi modal deep learning framework for enhanced agricultural pest detection, combining tiny-BERT's natural language processing with R-CNN and ResNet-18's image processing. Addressing limitations of traditional CNN-based visual methods, this approach integrates textual context for more accurate pest identification. The R-CNN and ResNet-18 integration tackles deep CNN issues like vanishing gradients, while tiny-BERT ensures computational efficiency. Employing ensemble learning with linear regression and random forest models, the framework demonstrates superior discriminate ability, as shown in ROC and AUC analyses. This multi modal approach, blending text and image data, significantly boosts pest detection in agriculture. The study highlights the potential of multi modal deep learning in complex real-world scenarios, suggesting future expansions in diversity of datasets, advanced data augmentation, and cross-modal attention mechanisms to enhance model performance.

{{</citation>}}


### (91/134) SeeBel: Seeing is Believing (Sourajit Saha et al., 2023)

{{<citation>}}

Sourajit Saha, Shubhashis Roy Dipta. (2023)  
**SeeBel: Seeing is Believing**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, Computer Vision, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.10933v1)  

---


**ABSTRACT**  
Semantic Segmentation is a significant research field in Computer Vision. Despite being a widely studied subject area, many visualization tools do not exist that capture segmentation quality and dataset statistics such as a class imbalance in the same view. While the significance of discovering and introspecting the correlation between dataset statistics and AI model performance for dense prediction computer vision tasks such as semantic segmentation is well established in the computer vision literature, to the best of our knowledge, no visualization tools have been proposed to view and analyze the aforementioned tasks. Our project aims to bridge this gap by proposing three visualizations that enable users to compare dataset statistics and AI performance for segmenting all images, a single image in the dataset, explore the AI model's attention on image regions once trained and browse the quality of masks predicted by AI for any selected (by user) number of objects under the same tool. Our project tries to further increase the interpretability of the trained AI model for segmentation by visualizing its image attention weights. For visualization, we use Scatterplot and Heatmap to encode correlation and features, respectively. We further propose to conduct surveys on real users to study the efficacy of our visualization tool in computer vision and AI domain. The full system can be accessed at https://github.com/dipta007/SeeBel

{{</citation>}}


### (92/134) Understanding the Multi-modal Prompts of the Pre-trained Vision-Language Model (Shuailei Ma et al., 2023)

{{<citation>}}

Shuailei Ma, Chen-Wei Xie, Ying Wei, Siyang Sun, Jiaqi Fan, Xiaoyi Bao, Yuxin Guo, Yun Zheng. (2023)  
**Understanding the Multi-modal Prompts of the Pre-trained Vision-Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.11570v1)  

---


**ABSTRACT**  
Prompt learning has emerged as an efficient alternative for fine-tuning foundational models, such as CLIP, for various downstream tasks. However, there is no work that provides a comprehensive explanation for the working mechanism of the multi-modal prompts. In this paper, we conduct a direct analysis of the multi-modal prompts by asking the following questions: $(i)$ How do the learned multi-modal prompts improve the recognition performance? $(ii)$ What do the multi-modal prompts learn? To answer these questions, we begin by isolating the component of the formula where the prompt influences the calculation of self-attention at each layer in two distinct ways, \ie, $(1)$ introducing prompt embeddings makes the $[cls]$ token focus on foreground objects. $(2)$ the prompts learn a bias term during the update of token embeddings, allowing the model to adapt to the target domain. Subsequently, we conduct extensive visualization and statistical experiments on the eleven diverse downstream recognition datasets. From the experiments, we reveal that the learned prompts improve the performance mainly through the second way, which acts as the dataset bias to improve the recognition performance of the pre-trained model on the corresponding dataset. Based on this finding, we propose the bias tuning way and demonstrate that directly incorporating the learnable bias outperforms the learnable prompts in the same parameter settings. In datasets with limited category information, \ie, EuroSAT, bias tuning surpasses prompt tuning by a large margin. With a deeper understanding of the multi-modal prompt, we hope our work can inspire new and solid research in this direction.

{{</citation>}}


### (93/134) MagicScroll: Nontypical Aspect-Ratio Image Generation for Visual Storytelling via Multi-Layered Semantic-Aware Denoising (Bingyuan Wang et al., 2023)

{{<citation>}}

Bingyuan Wang, Hengyu Meng, Zeyu Cai, Lanjiong Li, Yue Ma, Qifeng Chen, Zeyu Wang. (2023)  
**MagicScroll: Nontypical Aspect-Ratio Image Generation for Visual Storytelling via Multi-Layered Semantic-Aware Denoising**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.10899v1)  

---


**ABSTRACT**  
Visual storytelling often uses nontypical aspect-ratio images like scroll paintings, comic strips, and panoramas to create an expressive and compelling narrative. While generative AI has achieved great success and shown the potential to reshape the creative industry, it remains a challenge to generate coherent and engaging content with arbitrary size and controllable style, concept, and layout, all of which are essential for visual storytelling. To overcome the shortcomings of previous methods including repetitive content, style inconsistency, and lack of controllability, we propose MagicScroll, a multi-layered, progressive diffusion-based image generation framework with a novel semantic-aware denoising process. The model enables fine-grained control over the generated image on object, scene, and background levels with text, image, and layout conditions. We also establish the first benchmark for nontypical aspect-ratio image generation for visual storytelling including mediums like paintings, comics, and cinematic panoramas, with customized metrics for systematic evaluation. Through comparative and ablation studies, MagicScroll showcases promising results in aligning with the narrative text, improving visual coherence, and engaging the audience. We plan to release the code and benchmark in the hope of a better collaboration between AI researchers and creative practitioners involving visual storytelling.

{{</citation>}}


### (94/134) Country-Scale Cropland Mapping in Data-Scarce Settings Using Deep Learning: A Case Study of Nigeria (Joaquin Gajardo et al., 2023)

{{<citation>}}

Joaquin Gajardo, Michele Volpi, Daniel Onwude, Thijs Defraeye. (2023)  
**Country-Scale Cropland Mapping in Data-Scarce Settings Using Deep Learning: A Case Study of Nigeria**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Google, LSTM  
[Paper Link](http://arxiv.org/abs/2312.10872v1)  

---


**ABSTRACT**  
Cropland maps are a core and critical component of remote-sensing-based agricultural monitoring, providing dense and up-to-date information about agricultural development. Machine learning is an effective tool for large-scale agricultural mapping, but relies on geo-referenced ground-truth data for model training and testing, which can be scarce or time-consuming to obtain. In this study, we explore the usefulness of combining a global cropland dataset and a hand-labeled dataset to train machine learning models for generating a new cropland map for Nigeria in 2020 at 10 m resolution. We provide the models with pixel-wise time series input data from remote sensing sources such as Sentinel-1 and 2, ERA5 climate data, and DEM data, in addition to binary labels indicating cropland presence. We manually labeled 1827 evenly distributed pixels across Nigeria, splitting them into 50\% training, 25\% validation, and 25\% test sets used to fit the models and test our output map. We evaluate and compare the performance of single- and multi-headed Long Short-Term Memory (LSTM) neural network classifiers, a Random Forest classifier, and three existing 10 m resolution global land cover maps (Google's Dynamic World, ESRI's Land Cover, and ESA's WorldCover) on our proposed test set. Given the regional variations in cropland appearance, we additionally experimented with excluding or sub-setting the global crowd-sourced Geowiki cropland dataset, to empirically assess the trade-off between data quantity and data quality in terms of the similarity to the target data distribution of Nigeria. We find that the existing WorldCover map performs the best with an F1-score of 0.825 and accuracy of 0.870 on the test set, followed by a single-headed LSTM model trained with our hand-labeled training samples and the Geowiki data points in Nigeria, with a F1-score of 0.814 and accuracy of 0.842.

{{</citation>}}


## cs.RO (5)



### (95/134) Indoor and Outdoor 3D Scene Graph Generation via Language-Enabled Spatial Ontologies (Jared Strader et al., 2023)

{{<citation>}}

Jared Strader, Nathan Hughes, William Chen, Alberto Speranzon, Luca Carlone. (2023)  
**Indoor and Outdoor 3D Scene Graph Generation via Language-Enabled Spatial Ontologies**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.11713v1)  

---


**ABSTRACT**  
This paper proposes an approach to build 3D scene graphs in arbitrary (indoor and outdoor) environments. Such extension is challenging; the hierarchy of concepts that describe an outdoor environment is more complex than for indoors, and manually defining such hierarchy is time-consuming and does not scale. Furthermore, the lack of training data prevents the straightforward application of learning-based tools used in indoor settings. To address these challenges, we propose two novel extensions. First, we develop methods to build a spatial ontology defining concepts and relations relevant for indoor and outdoor robot operation. In particular, we use a Large Language Model (LLM) to build such an ontology, thus largely reducing the amount of manual effort required. Second, we leverage the spatial ontology for 3D scene graph construction using Logic Tensor Networks (LTN) to add logical rules, or axioms (e.g., "a beach contains sand"), which provide additional supervisory signals at training time thus reducing the need for labelled data, providing better predictions, and even allowing predicting concepts unseen at training time. We test our approach in a variety of datasets, including indoor, rural, and coastal environments, and show that it leads to a significant increase in the quality of the 3D scene graph generation with sparsely annotated data.

{{</citation>}}


### (96/134) Mastering Stacking of Diverse Shapes with Large-Scale Iterative Reinforcement Learning on Real Robots (Thomas Lampe et al., 2023)

{{<citation>}}

Thomas Lampe, Abbas Abdolmaleki, Sarah Bechtle, Sandy H. Huang, Jost Tobias Springenberg, Michael Bloesch, Oliver Groth, Roland Hafner, Tim Hertweck, Michael Neunert, Markus Wulfmeier, Jingwei Zhang, Francesco Nori, Nicolas Heess, Martin Riedmiller. (2023)  
**Mastering Stacking of Diverse Shapes with Large-Scale Iterative Reinforcement Learning on Real Robots**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11374v1)  

---


**ABSTRACT**  
Reinforcement learning solely from an agent's self-generated data is often believed to be infeasible for learning on real robots, due to the amount of data needed. However, if done right, agents learning from real data can be surprisingly efficient through re-using previously collected sub-optimal data. In this paper we demonstrate how the increased understanding of off-policy learning methods and their embedding in an iterative online/offline scheme (``collect and infer'') can drastically improve data-efficiency by using all the collected experience, which empowers learning from real robot experience only. Moreover, the resulting policy improves significantly over the state of the art on a recently proposed real robot manipulation benchmark. Our approach learns end-to-end, directly from pixels, and does not rely on additional human domain knowledge such as a simulator or demonstrations.

{{</citation>}}


### (97/134) Solving the swing-up and balance task for the Acrobot and Pendubot with SAC (Chi Zhang et al., 2023)

{{<citation>}}

Chi Zhang, Akhil Sathuluri, Markus Zimmermann. (2023)  
**Solving the swing-up and balance task for the Acrobot and Pendubot with SAC**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11311v1)  

---


**ABSTRACT**  
We present a solution of the swing-up and balance task for the pendubot and acrobot for the participation in the AI Olympics competition at IJCAI 2023. Our solution is based on the Soft Actor Crtic (SAC) reinforcement learning (RL) algorithm for training a policy for the swing-up and entering the region of attraction of a linear quadratic regulator(LQR) controller for stabilizing the double pendulum at the top position. Our controller achieves competitive scores in performance and robustness for both, pendubot and acrobot, problem scenarios.

{{</citation>}}


### (98/134) Multi-Agent Reinforcement Learning for Connected and Automated Vehicles Control: Recent Advancements and Future Prospects (Min Hua et al., 2023)

{{<citation>}}

Min Hua, Dong Chen, Xinda Qi, Kun Jiang, Zemin Eitan Liu, Quan Zhou, Hongming Xu. (2023)  
**Multi-Agent Reinforcement Learning for Connected and Automated Vehicles Control: Recent Advancements and Future Prospects**  

---
Primary Category: cs.RO  
Categories: cs-MA, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11084v1)  

---


**ABSTRACT**  
Connected and automated vehicles (CAVs) have emerged as a potential solution to the future challenges of developing safe, efficient, and eco-friendly transportation systems. However, CAV control presents significant challenges, given the complexity of interconnectivity and coordination required among the vehicles. To address this, multi-agent reinforcement learning (MARL), with its notable advancements in addressing complex problems in autonomous driving, robotics, and human-vehicle interaction, has emerged as a promising tool for enhancing the capabilities of CAVs. However, there is a notable absence of current reviews on the state-of-the-art MARL algorithms in the context of CAVs. Therefore, this paper delivers a comprehensive review of the application of MARL techniques within the field of CAV control. The paper begins by introducing MARL, followed by a detailed explanation of its unique advantages in addressing complex mobility and traffic scenarios that involve multiple agents. It then presents a comprehensive survey of MARL applications on the extent of control dimensions for CAVs, covering critical and typical scenarios such as platooning control, lane-changing, and unsignalized intersections. In addition, the paper provides a comprehensive review of the prominent simulation platforms used to create reliable environments for training in MARL. Lastly, the paper examines the current challenges associated with deploying MARL within CAV control and outlines potential solutions that can effectively overcome these issues. Through this review, the study highlights the tremendous potential of MARL to enhance the performance and collaboration of CAV control in terms of safety, travel efficiency, and economy.

{{</citation>}}


### (99/134) Robot Crowd Navigation in Dynamic Environment with Offline Reinforcement Learning (Shuai Zhou et al., 2023)

{{<citation>}}

Shuai Zhou, Hao Fu, Haodong He, Wei Liu. (2023)  
**Robot Crowd Navigation in Dynamic Environment with Offline Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11032v1)  

---


**ABSTRACT**  
Robot crowd navigation has been gaining increasing attention and popularity in various practical applications. In existing research, deep reinforcement learning has been applied to robot crowd navigation by training policies in an online mode. However, this inevitably leads to unsafe exploration, and consequently causes low sampling efficiency during pedestrian-robot interaction. To this end, we propose an offline reinforcement learning based robot crowd navigation algorithm by utilizing pre-collected crowd navigation experience. Specifically, this algorithm integrates a spatial-temporal state into implicit Q-Learning to avoid querying out-of-distribution robot actions of the pre-collected experience, while capturing spatial-temporal features from the offline pedestrian-robot interactions. Experimental results demonstrate that the proposed algorithm outperforms the state-of-the-art methods by means of qualitative and quantitative analysis.

{{</citation>}}


## eess.SY (3)



### (100/134) Opportunities and Challenges of Applying Large Language Models in Building Energy Efficiency and Decarbonization Studies: An Exploratory Overview (Liang Zhang et al., 2023)

{{<citation>}}

Liang Zhang, Zhelun Chen. (2023)  
**Opportunities and Challenges of Applying Large Language Models in Building Energy Efficiency and Decarbonization Studies: An Exploratory Overview**  

---
Primary Category: eess.SY  
Categories: cs-CL, cs-SY, eess-SY, eess.SY  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.11701v1)  

---


**ABSTRACT**  
In recent years, the rapid advancement and impressive capabilities of Large Language Models (LLMs) have been evident across various domains. This paper explores the application, implications, and potential of LLMs in building energy efficiency and decarbonization studies. The wide-ranging capabilities of LLMs are examined in the context of the building energy field, including intelligent control systems, code generation, data infrastructure, knowledge extraction, and education. Despite the promising potential of LLMs, challenges including complex and expensive computation, data privacy, security and copyright, complexity in fine-tuned LLMs, and self-consistency are discussed. The paper concludes with a call for future research focused on the enhancement of LLMs for domain-specific tasks, multi-modal LLMs, and collaborative research between AI and energy experts.

{{</citation>}}


### (101/134) Unsupervised Learning for Fault Detection of HVAC Systems: An OPTICS -based Approach for Terminal Air Handling Units (Farivar Rajabi et al., 2023)

{{<citation>}}

Farivar Rajabi, J. J. McArthur. (2023)  
**Unsupervised Learning for Fault Detection of HVAC Systems: An OPTICS -based Approach for Terminal Air Handling Units**  

---
Primary Category: eess.SY  
Categories: 97M50, cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11405v1)  

---


**ABSTRACT**  
The rise of AI-powered classification techniques has ushered in a new era for data-driven Fault Detection and Diagnosis in smart building systems. While extensive research has championed supervised FDD approaches, the real-world application of unsupervised methods remains limited. Among these, cluster analysis stands out for its potential with Building Management System data. This study introduces an unsupervised learning strategy to detect faults in terminal air handling units and their associated systems. The methodology involves pre-processing historical sensor data using Principal Component Analysis to streamline dimensions. This is then followed by OPTICS clustering, juxtaposed against k-means for comparison. The effectiveness of the proposed strategy was gauged using several labeled datasets depicting various fault scenarios and real-world building BMS data. Results showed that OPTICS consistently surpassed k-means in accuracy across seasons. Notably, OPTICS offers a unique visualization feature for users called reachability distance, allowing a preview of detected clusters before setting thresholds. Moreover, according to the results, while PCA is beneficial for reducing computational costs and enhancing noise reduction, thereby generally improving the clarity of cluster differentiation in reachability distance. It also has its limitations, particularly in complex fault scenarios. In such cases, PCA's dimensionality reduction may result in the loss of critical information, leading to some clusters being less discernible or entirely undetected. These overlooked clusters could be indicative of underlying faults, and their obscurity represents a significant limitation of PCA when identifying potential fault lines in intricate datasets.

{{</citation>}}


### (102/134) Contextual Reinforcement Learning for Offshore Wind Farm Bidding (David Cole et al., 2023)

{{<citation>}}

David Cole, Himanshu Sharma, Wei Wang. (2023)  
**Contextual Reinforcement Learning for Offshore Wind Farm Bidding**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY, math-OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.10884v1)  

---


**ABSTRACT**  
We propose a framework for applying reinforcement learning to contextual two-stage stochastic optimization and apply this framework to the problem of energy market bidding of an off-shore wind farm. Reinforcement learning could potentially be used to learn close to optimal solutions for first stage variables of a two-stage stochastic program under different contexts. Under the proposed framework, these solutions would be learned without having to solve the full two-stage stochastic program. We present initial results of training using the DDPG algorithm and present intended future steps to improve performance.

{{</citation>}}


## cs.CR (3)



### (103/134) Traces of Memorisation in Large Language Models for Code (Ali Al-Kaswan et al., 2023)

{{<citation>}}

Ali Al-Kaswan, Maliheh Izadi, Arie van Deursen. (2023)  
**Traces of Memorisation in Large Language Models for Code**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-SE, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.11658v1)  

---


**ABSTRACT**  
Large language models have gained significant popularity because of their ability to generate human-like text and potential applications in various fields, such as Software Engineering. Large language models for code are commonly trained on large unsanitised corpora of source code scraped from the internet. The content of these datasets is memorised and can be extracted by attackers with data extraction attacks. In this work, we explore memorisation in large language models for code and compare the rate of memorisation with large language models trained on natural language. We adopt an existing benchmark for natural language and construct a benchmark for code by identifying samples that are vulnerable to attack. We run both benchmarks against a variety of models, and perform a data extraction attack. We find that large language models for code are vulnerable to data extraction attacks, like their natural language counterparts. From the training data that was identified to be potentially extractable we were able to extract 47% from a CodeGen-Mono-16B code completion model. We also observe that models memorise more, as their parameter count grows, and that their pre-training data are also vulnerable to attack. We also find that data carriers are memorised at a higher rate than regular code or documentation and that different model architectures memorise different samples. Data leakage has severe outcomes, so we urge the research community to further investigate the extent of this phenomenon using a wider range of models and extraction techniques in order to build safeguards to mitigate this issue.

{{</citation>}}


### (104/134) MAD-MulW: A Multi-Window Anomaly Detection Framework for BGP Security Events (Songtao Peng et al., 2023)

{{<citation>}}

Songtao Peng, Yiping Chen, Xincheng Shu, Wu Shuai, Shenhao Fang, Zhongyuan Ruan, Qi Xuan. (2023)  
**MAD-MulW: A Multi-Window Anomaly Detection Framework for BGP Security Events**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Anomaly Detection, Security  
[Paper Link](http://arxiv.org/abs/2312.11225v1)  

---


**ABSTRACT**  
In recent years, various international security events have occurred frequently and interacted between real society and cyberspace. Traditional traffic monitoring mainly focuses on the local anomalous status of events due to a large amount of data. BGP-based event monitoring makes it possible to perform differential analysis of international events. For many existing traffic anomaly detection methods, we have observed that the window-based noise reduction strategy effectively improves the success rate of time series anomaly detection. Motivated by this observation, we propose an unsupervised anomaly detection model, MAD-MulW, which incorporates a multi-window serial framework. Firstly, we design the W-GAT module to adaptively update the sample weights within the window and retain the updated information of the trailing sample, which not only reduces the outlier samples' noise but also avoids the space consumption of data scale expansion. Then, the W-LAT module based on predictive reconstruction both captures the trend of sample fluctuations over a certain period of time and increases the interclass variation through the reconstruction of the predictive sample. Our model has been experimentally validated on multiple BGP anomalous events with an average F1 score of over 90\%, which demonstrates the significant improvement effect of the stage windows and adaptive strategy on the efficiency and stability of the timing model.

{{</citation>}}


### (105/134) A Comprehensive Survey of Attack Techniques, Implementation, and Mitigation Strategies in Large Language Models (Aysan Esmradi et al., 2023)

{{<citation>}}

Aysan Esmradi, Daniel Wankit Yip, Chun Fai Chan. (2023)  
**A Comprehensive Survey of Attack Techniques, Implementation, and Mitigation Strategies in Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.10982v1)  

---


**ABSTRACT**  
Ensuring the security of large language models (LLMs) is an ongoing challenge despite their widespread popularity. Developers work to enhance LLMs security, but vulnerabilities persist, even in advanced versions like GPT-4. Attackers exploit these weaknesses, highlighting the need for proactive cybersecurity measures in AI model development. This article explores two attack categories: attacks on models themselves and attacks on model applications. The former requires expertise, access to model data, and significant implementation time, while the latter is more accessible to attackers and has seen increased attention. Our study reviews over 100 recent research works, providing an in-depth analysis of each attack type. We identify the latest attack methods and explore various approaches to carry them out. We thoroughly investigate mitigation techniques, assessing their effectiveness and limitations. Furthermore, we summarize future defenses against these attacks. We also examine real-world techniques, including reported and our implemented attacks on LLMs, to consolidate our findings. Our research highlights the urgency of addressing security concerns and aims to enhance the understanding of LLM attacks, contributing to robust defense development in this evolving domain.

{{</citation>}}


## cs.HC (6)



### (106/134) Explore 3D Dance Generation via Reward Model from Automatically-Ranked Demonstrations (Zilin Wang et al., 2023)

{{<citation>}}

Zilin Wang, Haolin Zhuang, Lu Li, Yinmin Zhang, Junjie Zhong, Jun Chen, Yu Yang, Boshi Tang, Zhiyong Wu. (2023)  
**Explore 3D Dance Generation via Reward Model from Automatically-Ranked Demonstrations**  

---
Primary Category: cs.HC  
Categories: I-3-7, cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11442v1)  

---


**ABSTRACT**  
This paper presents an Exploratory 3D Dance generation framework, E3D2, designed to address the exploration capability deficiency in existing music-conditioned 3D dance generation models. Current models often generate monotonous and simplistic dance sequences that misalign with human preferences because they lack exploration capabilities. The E3D2 framework involves a reward model trained from automatically-ranked dance demonstrations, which then guides the reinforcement learning process. This approach encourages the agent to explore and generate high quality and diverse dance movement sequences. The soundness of the reward model is both theoretically and experimentally validated. Empirical experiments demonstrate the effectiveness of E3D2 on the AIST++ dataset. Project Page: https://sites.google.com/view/e3d2.

{{</citation>}}


### (107/134) Improving Student Learning with Hybrid Human-AI Tutoring: A Three-Study Quasi-Experimental Investigation (Danielle R. Thomas et al., 2023)

{{<citation>}}

Danielle R. Thomas, Jionghao Lin, Erin Gatz, Ashish Gurung, Shivang Gupta, Kole Norberg, Stephen E. Fancsali, Vincent Aleven, Lee Branstetter, Emma Brunskill, Kenneth R. Koedinger. (2023)  
**Improving Student Learning with Hybrid Human-AI Tutoring: A Three-Study Quasi-Experimental Investigation**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11274v2)  

---


**ABSTRACT**  
Artificial intelligence (AI) applications to support human tutoring have potential to significantly improve learning outcomes, but engagement issues persist, especially among students from low-income backgrounds. We introduce an AI-assisted tutoring model that combines human and AI tutoring and hypothesize that this synergy will have positive impacts on learning processes. To investigate this hypothesis, we conduct a three-study quasi-experiment across three urban and low-income middle schools: 1) 125 students in a Pennsylvania school; 2) 385 students (50% Latinx) in a California school; and 3) 75 students (100% Black) in a Pennsylvania charter school, all implementing analogous tutoring models. We compare learning analytics of students engaged in human-AI tutoring compared to students using math software only. We find human-AI tutoring has positive effects, particularly in student's proficiency and usage, with evidence suggesting lower achieving students may benefit more compared to higher achieving students. We illustrate the use of quasi-experimental methods adapted to the particulars of different schools and data-availability contexts so as to achieve the rapid data-driven iteration needed to guide an inspired creation into effective innovation. Future work focuses on improving the tutor dashboard and optimizing tutor-student ratios, while maintaining annual costs per students of approximately $700 annually.

{{</citation>}}


### (108/134) Navigating Interfaces with AI for Enhanced User Interaction (Yunpeng Song et al., 2023)

{{<citation>}}

Yunpeng Song, Yiheng Bian, Yongtao Tang, Zhongmin Cai. (2023)  
**Navigating Interfaces with AI for Enhanced User Interaction**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11190v1)  

---


**ABSTRACT**  
This study introduces an innovative framework designed to automate tasks by interacting with UIs through a sequential, human-like problem-solving approach. Our approach initially transforms UI screenshots into natural language explanations through a vision-based UI analysis, circumventing traditional view hierarchy limitations. It then methodically engages with each interface, guiding the LLM to pinpoint and act on relevant UI elements, thus bolstering both precision and functionality. Employing the ERNIE Bot LLM, our approach has been demonstrated to surpass existing methodologies. It delivers superior UI interpretation across various datasets and exhibits remarkable efficiency in automating varied tasks on an Android smartphone, outperforming human capabilities in intricate tasks and significantly enhancing the PBD process.

{{</citation>}}


### (109/134) Emotion Based Prediction in the Context of Optimized Trajectory Planning for Immersive Learning (Akey Sungheetha et al., 2023)

{{<citation>}}

Akey Sungheetha, Rajesh Sharma R, Chinnaiyan R. (2023)  
**Emotion Based Prediction in the Context of Optimized Trajectory Planning for Immersive Learning**  

---
Primary Category: cs.HC  
Categories: cs-CV, cs-HC, cs-MM, cs.HC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2312.11576v1)  

---


**ABSTRACT**  
In the virtual elements of immersive learning, the use of Google Expedition and touch-screen-based emotion are examined. The objective is to investigate possible ways to combine these technologies to enhance virtual learning environments and learners emotional engagement. Pedagogical application, affordances, and cognitive load are the corresponding measures that are involved. Students will gain insight into the reason behind their significantly higher post-assessment Prediction Systems scores compared to preassessment scores through this work that leverages technology. This suggests that it is effective to include emotional elements in immersive learning scenarios. The results of this study may help develop new strategies by leveraging the features of immersive learning technology in educational technologies to improve virtual reality and augmented reality experiences. Furthermore, the effectiveness of immersive learning environments can be raised by utilizing magnetic, optical, or hybrid trackers that considerably improve object tracking.

{{</citation>}}


### (110/134) Application of AI in Nutrition (Ritu Ramakrishnan et al., 2023)

{{<citation>}}

Ritu Ramakrishnan, Tianxiang Xing, Tianfeng Chen, Ming-Hao Lee, Jinzhu Gao. (2023)  
**Application of AI in Nutrition**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-IR, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11569v1)  

---


**ABSTRACT**  
In healthcare, artificial intelligence (AI) has been changing the way doctors and health experts take care of people. This paper will cover how AI is making major changes in the health care system, especially with nutrition. Various machine learning and deep learning algorithms have been developed to extract valuable information from healthcare data which help doctors, nutritionists, and health experts to make better decisions and make our lifestyle healthy. This paper provides an overview of the current state of AI applications in healthcare with a focus on the utilization of AI-driven recommender systems in nutrition. It will discuss the positive outcomes and challenges that arise when AI is used in this field. This paper addresses the challenges to develop AI recommender systems in healthcare, providing a well-rounded perspective on the complexities. Real-world examples and research findings are presented to underscore the tangible and significant impact AI recommender systems have in the field of healthcare, particularly in nutrition. The ongoing efforts of applying AI in nutrition lay the groundwork for a future where personalized recommendations play a pivotal role in guiding individuals toward healthier lifestyles.

{{</citation>}}


### (111/134) The Metacognitive Demands and Opportunities of Generative AI (Lev Tankelevitch et al., 2023)

{{<citation>}}

Lev Tankelevitch, Viktor Kewenig, Auste Simkute, Ava Elizabeth Scott, Advait Sarkar, Abigail Sellen, Sean Rintel. (2023)  
**The Metacognitive Demands and Opportunities of Generative AI**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.10893v1)  

---


**ABSTRACT**  
Generative AI (GenAI) systems offer unprecedented opportunities for transforming professional and personal work, yet present challenges around prompting, evaluating and relying on outputs, and optimizing workflows. We argue that metacognition$\unicode{x2013}$the psychological ability to monitor and control one's thoughts and behavior$\unicode{x2013}$offers a valuable lens to understand and design for these usability challenges. Drawing on research in psychology and cognitive science, and recent GenAI user studies, we illustrate how GenAI systems impose metacognitive demands on users, requiring a high degree of metacognitive monitoring and control. We propose these demands could be addressed by integrating metacognitive support strategies into GenAI systems, and by designing GenAI systems to reduce their metacognitive demand by targeting explainability and customizability. Metacognition offers a coherent framework for understanding the usability challenges posed by GenAI, enabling us to offer research and design directions to advance human-GenAI interaction.

{{</citation>}}


## math.OC (1)



### (112/134) When can you trust feature selection? -- I: A condition-based analysis of LASSO and generalised hardness of approximation (Alexander Bastounis et al., 2023)

{{<citation>}}

Alexander Bastounis, Felipe Cucker, Anders C. Hansen. (2023)  
**When can you trust feature selection? -- I: A condition-based analysis of LASSO and generalised hardness of approximation**  

---
Primary Category: math.OC  
Categories: cs-DS, cs-LG, math-OC, math.OC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11425v1)  

---


**ABSTRACT**  
The arrival of AI techniques in computations, with the potential for hallucinations and non-robustness, has made trustworthiness of algorithms a focal point. However, trustworthiness of the many classical approaches are not well understood. This is the case for feature selection, a classical problem in the sciences, statistics, machine learning etc. Here, the LASSO optimisation problem is standard. Despite its widespread use, it has not been established when the output of algorithms attempting to compute support sets of minimisers of LASSO in order to do feature selection can be trusted. In this paper we establish how no (randomised) algorithm that works on all inputs can determine the correct support sets (with probability $> 1/2$) of minimisers of LASSO when reading approximate input, regardless of precision and computing power. However, we define a LASSO condition number and design an efficient algorithm for computing these support sets provided the input data is well-posed (has finite condition number) in time polynomial in the dimensions and logarithm of the condition number. For ill-posed inputs the algorithm runs forever, hence, it will never produce a wrong answer. Furthermore, the algorithm computes an upper bound for the condition number when this is finite. Finally, for any algorithm defined on an open set containing a point with infinite condition number, there is an input for which the algorithm will either run forever or produce a wrong answer. Our impossibility results stem from generalised hardness of approximation -- within the Solvability Complexity Index (SCI) hierarchy framework -- that generalises the classical phenomenon of hardness of approximation.

{{</citation>}}


## cs.MA (1)



### (113/134) Agent Assessment of Others Through the Lens of Self (Jasmine A. Berry, 2023)

{{<citation>}}

Jasmine A. Berry. (2023)  
**Agent Assessment of Others Through the Lens of Self**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-MA, cs-RO, cs.MA  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11357v1)  

---


**ABSTRACT**  
The maturation of cognition, from introspection to understanding others, has long been a hallmark of human development. This position paper posits that for AI systems to truly emulate or approach human-like interactions, especially within multifaceted environments populated with diverse agents, they must first achieve an in-depth and nuanced understanding of self. Drawing parallels with the human developmental trajectory from self-awareness to mentalizing (also called theory of mind), the paper argues that the quality of an autonomous agent's introspective capabilities of self are crucial in mirroring quality human-like understandings of other agents. While counterarguments emphasize practicality, computational efficiency, and ethical concerns, this position proposes a development approach, blending algorithmic considerations of self-referential processing. Ultimately, the vision set forth is not merely of machines that compute but of entities that introspect, empathize, and understand, harmonizing with the complex compositions of human cognition.

{{</citation>}}


## quant-ph (1)



### (114/134) Challenges for Reinforcement Learning in Quantum Computing (Philipp Altmann et al., 2023)

{{<citation>}}

Philipp Altmann, Adelina Bärligea, Jonas Stein, Michael Kölle, Thomas Gabor, Thomy Phan, Claudia Linnhoff-Popien. (2023)  
**Challenges for Reinforcement Learning in Quantum Computing**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.11337v1)  

---


**ABSTRACT**  
Quantum computing (QC) in the current NISQ-era is still limited. To gain early insights and advantages, hybrid applications are widely considered mitigating those shortcomings. Hybrid quantum machine learning (QML) comprises both the application of QC to improve machine learning (ML), and the application of ML to improve QC architectures. This work considers the latter, focusing on leveraging reinforcement learning (RL) to improve current QC approaches. We therefore introduce various generic challenges arising from quantum architecture search and quantum circuit optimization that RL algorithms need to solve to provide benefits for more complex applications and combinations of those. Building upon these challenges we propose a concrete framework, formalized as a Markov decision process, to enable to learn policies that are capable of controlling a universal set of quantum gates. Furthermore, we provide benchmark results to assess shortcomings and strengths of current state-of-the-art algorithms.

{{</citation>}}


## cs.IR (5)



### (115/134) DRDT: Dynamic Reflection with Divergent Thinking for LLM-based Sequential Recommendation (Yu Wang et al., 2023)

{{<citation>}}

Yu Wang, Zhiwei Liu, Jianguo Zhang, Weiran Yao, Shelby Heinecke, Philip S. Yu. (2023)  
**DRDT: Dynamic Reflection with Divergent Thinking for LLM-based Sequential Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.11336v1)  

---


**ABSTRACT**  
The rise of Large Language Models (LLMs) has sparked interest in their application to sequential recommendation tasks as they can provide supportive item information. However, due to the inherent complexities of sequential recommendation, such as sequential patterns across datasets, noise within sequences, and the temporal evolution of user preferences, existing LLM reasoning strategies, such as in-context learning and chain-of-thought are not fully effective. To address these challenges, we introduce a novel reasoning principle: Dynamic Reflection with Divergent Thinking within a retriever-reranker framework. Our approach starts with a collaborative in-context demonstration retriever, which collects sequences exhibiting collaborative behaviors as in-context examples. Following this, we abstract high-level user preferences across multiple aspects, providing a more nuanced understanding of user interests and circumventing the noise within the raw sequences. The cornerstone of our methodology is dynamic reflection, a process that emulates human learning through probing, critiquing, and reflecting, using user feedback to tailor the analysis more effectively to the target user in a temporal manner. We evaluate our approach on three datasets using six pre-trained LLMs. The superior performance observed across these models demonstrates the efficacy of our reasoning strategy, notably achieved without the need to fine-tune the LLMs. With our principle, we managed to outperform GPT-Turbo-3.5 on three datasets using 7b models e.g., Vicuna-7b and Openchat-7b on NDCG@10. This research not only highlights the potential of LLMs in enhancing sequential recommendation systems but also underscores the importance of developing tailored reasoning strategies to fully harness their capabilities.

{{</citation>}}


### (116/134) CaseGNN: Graph Neural Networks for Legal Case Retrieval with Text-Attributed Graphs (Yanran Tang et al., 2023)

{{<citation>}}

Yanran Tang, Ruihong Qiu, Yilun Liu, Xue Li, Zi Huang. (2023)  
**CaseGNN: Graph Neural Networks for Legal Case Retrieval with Text-Attributed Graphs**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Attention, BERT, GNN, Graph Neural Network, Graph Neural Networks, Legal  
[Paper Link](http://arxiv.org/abs/2312.11229v2)  

---


**ABSTRACT**  
Legal case retrieval is an information retrieval task in the legal domain, which aims to retrieve relevant cases with a given query case. Recent research of legal case retrieval mainly relies on traditional bag-of-words models and language models. Although these methods have achieved significant improvement in retrieval accuracy, there are still two challenges: (1) Legal structural information neglect. Previous neural legal case retrieval models mostly encode the unstructured raw text of case into a case representation, which causes the lack of important legal structural information in a case and leads to poor case representation; (2) Lengthy legal text limitation. When using the powerful BERT-based models, there is a limit of input text lengths, which inevitably requires to shorten the input via truncation or division with a loss of legal context information. In this paper, a graph neural networks-based legal case retrieval model, CaseGNN, is developed to tackle these challenges. To effectively utilise the legal structural information during encoding, a case is firstly converted into a Text-Attributed Case Graph (TACG), followed by a designed Edge Graph Attention Layer and a readout function to obtain the case graph representation. The CaseGNN model is optimised with a carefully designed contrastive loss with easy and hard negative sampling. Since the text attributes in the case graph come from individual sentences, the restriction of using language models is further avoided without losing the legal context. Extensive experiments have been conducted on two benchmarks from COLIEE 2022 and COLIEE 2023, which demonstrate that CaseGNN outperforms other state-of-the-art legal case retrieval methods. The code has been released on https://github.com/yanran-tang/CaseGNN.

{{</citation>}}


### (117/134) UniGen: A Unified Generative Framework for Retrieval and Question Answering with Large Language Models (Xiaoxi Li et al., 2023)

{{<citation>}}

Xiaoxi Li, Yujia Zhou, Zhicheng Dou. (2023)  
**UniGen: A Unified Generative Framework for Retrieval and Question Answering with Large Language Models**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs.IR  
Keywords: Language Model, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.11036v1)  

---


**ABSTRACT**  
Generative information retrieval, encompassing two major tasks of Generative Document Retrieval (GDR) and Grounded Answer Generation (GAR), has gained significant attention in the area of information retrieval and natural language processing. Existing methods for GDR and GAR rely on separate retrieval and reader modules, which hinder simultaneous optimization. To overcome this, we present \textbf{UniGen}, a \textbf{Uni}fied \textbf{Gen}erative framework for retrieval and question answering that integrates both tasks into a single generative model leveraging the capabilities of large language models. UniGen employs a shared encoder and two distinct decoders for generative retrieval and question answering. To facilitate the learning of both tasks, we introduce connectors, generated by large language models, to bridge the gaps between query inputs and generation targets, as well as between document identifiers and answers. Furthermore, we propose an iterative enhancement strategy that leverages generated answers and retrieved documents to iteratively improve both tasks. Through extensive experiments on the MS MARCO and NQ datasets, we demonstrate the effectiveness of UniGen, showcasing its superior performance in both the retrieval and the question answering tasks.

{{</citation>}}


### (118/134) Hypergrah-Enhanced Dual Convolutional Network for Bundle Recommendation (Kangbo Liu et al., 2023)

{{<citation>}}

Kangbo Liu, Yang Li, Yaoxin Wu, Zhaoxuan Wang, Xiaoxu Wang. (2023)  
**Hypergrah-Enhanced Dual Convolutional Network for Bundle Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.11018v1)  

---


**ABSTRACT**  
Bundle recommendations strive to offer users a set of items as a package named bundle, enhancing convenience and contributing to the seller's revenue. While previous approaches have demonstrated notable performance, we argue that they may compromise the ternary relationship among users, items, and bundles. This compromise can result in information loss, ultimately impacting the overall model performance. To address this gap, we develop a unified model for bundle recommendation, termed hypergraph-enhanced dual convolutional neural network (HED). Our approach is characterized by two key aspects. Firstly, we construct a complete hypergraph to capture interaction dynamics among users, items, and bundles. Secondly, we incorporate U-B interaction information to enhance the information representation derived from users and bundle embedding vectors. Extensive experimental results on the Youshu and Netease datasets have demonstrated that HED surpasses state-of-the-art baselines, proving its effectiveness. In addition, various ablation studies and sensitivity analyses revealed the working mechanism and proved our effectiveness. Codes and datasets are available at https://github.com/AAI-Lab/HED

{{</citation>}}


### (119/134) On-Device Recommender Systems: A Tutorial on The New-Generation Recommendation Paradigm (Hongzhi Yin et al., 2023)

{{<citation>}}

Hongzhi Yin, Tong Chen, Liang Qu, Bin Cui. (2023)  
**On-Device Recommender Systems: A Tutorial on The New-Generation Recommendation Paradigm**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2312.10864v1)  

---


**ABSTRACT**  
Given the sheer volume of contemporary e-commerce applications, recommender systems (RSs) have gained significant attention in both academia and industry. However, traditional cloud-based RSs face inevitable challenges, such as resource-intensive computation, reliance on network access, and privacy breaches. In response, a new paradigm called on-device recommender systems (ODRSs) has emerged recently in various industries like Taobao, Google, and Kuaishou. ODRSs unleash the computational capacity of user devices with lightweight recommendation models tailored for resource-constrained environments, enabling real-time inference with users' local data. This tutorial aims to systematically introduce methodologies of ODRSs, including (1) an overview of existing research on ODRSs; (2) a comprehensive taxonomy of ODRSs, where the core technical content to be covered span across three major ODRS research directions, including on-device deployment and inference, on-device training, and privacy/security of ODRSs; (3) limitations and future directions of ODRSs. This tutorial expects to lay the foundation and spark new insights for follow-up research and applications concerning this new recommendation paradigm.

{{</citation>}}


## cs.SI (3)



### (120/134) Topic Shifts as a Proxy for Assessing Politicization in Social Media (Marcelo Sartori Locatelli et al., 2023)

{{<citation>}}

Marcelo Sartori Locatelli, Pedro Calais, Matheus Prado Miranda, João Pedro Junho, Tomas Lacerda Muniz, Wagner Meira Jr., Virgilio Almeida. (2023)  
**Topic Shifts as a Proxy for Assessing Politicization in Social Media**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2312.11326v1)  

---


**ABSTRACT**  
Politicization is a social phenomenon studied by political science characterized by the extent to which ideas and facts are given a political tone. A range of topics, such as climate change, religion and vaccines has been subject to increasing politicization in the media and social media platforms. In this work, we propose a computational method for assessing politicization in online conversations based on topic shifts, i.e., the degree to which people switch topics in online conversations. The intuition is that topic shifts from a non-political topic to politics are a direct measure of politicization -- making something political, and that the more people switch conversations to politics, the more they perceive politics as playing a vital role in their daily lives. A fundamental challenge that must be addressed when one studies politicization in social media is that, a priori, any topic may be politicized. Hence, any keyword-based method or even machine learning approaches that rely on topic labels to classify topics are expensive to run and potentially ineffective. Instead, we learn from a seed of political keywords and use Positive-Unlabeled (PU) Learning to detect political comments in reaction to non-political news articles posted on Twitter, YouTube, and TikTok during the 2022 Brazilian presidential elections. Our findings indicate that all platforms show evidence of politicization as discussion around topics adjacent to politics such as economy, crime and drugs tend to shift to politics. Even the least politicized topics had the rate in which their topics shift to politics increased in the lead up to the elections and after other political events in Brazil -- an evidence of politicization.

{{</citation>}}


### (121/134) Discovering Geo-dependent Stories by Combining Density-based Clustering and Thread-based Aggregation techniques (Héctor Cerezo-Costas et al., 2023)

{{<citation>}}

Héctor Cerezo-Costas, Ana Fernández Vilas, Manuela Martín-Vicente, Rebeca P. Díaz-Redondo. (2023)  
**Discovering Geo-dependent Stories by Combining Density-based Clustering and Thread-based Aggregation techniques**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2312.11076v1)  

---


**ABSTRACT**  
Citizens are actively interacting with their surroundings, especially through social media. Not only do shared posts give important information about what is happening (from the users' perspective), but also the metadata linked to these posts offer relevant data, such as the GPS-location in Location-based Social Networks (LBSNs). In this paper we introduce a global analysis of the geo-tagged posts in social media which supports (i) the detection of unexpected behavior in the city and (ii) the analysis of the posts to infer what is happening. The former is obtained by applying density-based clustering techniques, whereas the latter is consequence of applying natural language processing. We have applied our methodology to a dataset obtained from Instagram activity in New York City for seven months obtaining promising results. The developed algorithms require very low resources, being able to analyze millions of data-points in commodity hardware in less than one hour without applying complex parallelization techniques. Furthermore, the solution can be easily adapted to other geo-tagged data sources without extra effort.

{{</citation>}}


### (122/134) Viral Privacy: Contextual Integrity as a Lens to Understand Content Creators' Privacy Perceptions and Needs After Sudden Attention (Joseph S. Schafer et al., 2023)

{{<citation>}}

Joseph S. Schafer, Annie Denton, Chloe Seelhoff, Jordyn Vo, Kate Starbird. (2023)  
**Viral Privacy: Contextual Integrity as a Lens to Understand Content Creators' Privacy Perceptions and Needs After Sudden Attention**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.10951v1)  

---


**ABSTRACT**  
When designing multi-stakeholder privacy systems, it is important to consider how different groups of social media users have different goals and requirements for privacy. Additionally, we must acknowledge that it is important to keep in mind that even a single creator's needs can change as their online visibility and presence shifts, and that robust multi-stakeholder privacy systems should account for these shifts. Using the framework of contextual integrity, we explain a theoretical basis for how to evaluate the potential changing privacy needs of users as their profiles undergo a sudden rise in online attention, and ongoing projects to understand these potential shifts in perspectives.

{{</citation>}}


## cs.DB (1)



### (123/134) Optimised Storage for Datalog Reasoning (Xinyue Zhang et al., 2023)

{{<citation>}}

Xinyue Zhang, Pan Hu, Yavor Nenov, Ian Horrocks. (2023)  
**Optimised Storage for Datalog Reasoning**  

---
Primary Category: cs.DB  
Categories: cs-AI, cs-DB, cs.DB  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2312.11297v2)  

---


**ABSTRACT**  
Materialisation facilitates Datalog reasoning by precomputing all consequences of the facts and the rules so that queries can be directly answered over the materialised facts. However, storing all materialised facts may be infeasible in practice, especially when the rules are complex and the given set of facts is large. We observe that for certain combinations of rules, there exist data structures that compactly represent the reasoning result and can be efficiently queried when necessary. In this paper, we present a general framework that allows for the integration of such optimised storage schemes with standard materialisation algorithms. Moreover, we devise optimised storage schemes targeting at transitive rules and union rules, two types of (combination of) rules that commonly occur in practice. Our experimental evaluation shows that our approach significantly improves memory consumption, sometimes by orders of magnitude, while remaining competitive in terms of query answering time.

{{</citation>}}


## eess.IV (2)



### (124/134) Self-Supervised Learning for Image Super-Resolution and Deblurring (Jérémy Scanvic et al., 2023)

{{<citation>}}

Jérémy Scanvic, Mike Davies, Patrice Abry, Julián Tachella. (2023)  
**Self-Supervised Learning for Image Super-Resolution and Deblurring**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.11232v1)  

---


**ABSTRACT**  
Self-supervised methods have recently proved to be nearly as effective as supervised methods in various imaging inverse problems, paving the way for learning-based methods in scientific and medical imaging applications where ground truth data is hard or expensive to obtain. This is the case in magnetic resonance imaging and computed tomography. These methods critically rely on invariance to translations and/or rotations of the image distribution to learn from incomplete measurement data alone. However, existing approaches fail to obtain competitive performances in the problems of image super-resolution and deblurring, which play a key role in most imaging systems. In this work, we show that invariance to translations and rotations is insufficient to learn from measurements that only contain low-frequency information. Instead, we propose a new self-supervised approach that leverages the fact that many image distributions are approximately scale-invariant, and that can be applied to any inverse problem where high-frequency information is lost in the measurement process. We demonstrate throughout a series of experiments on real datasets that the proposed method outperforms other self-supervised approaches, and obtains performances on par with fully supervised learning.

{{</citation>}}


### (125/134) PlaNet-S: Automatic Semantic Segmentation of Placenta (Shinnosuke Yamamoto et al., 2023)

{{<citation>}}

Shinnosuke Yamamoto, Isso Saito, Eichi Takaya, Ayaka Harigai, Tomomi Sato, Tomoya Kobayashi, Kei Takase, Takuya Ueda. (2023)  
**PlaNet-S: Automatic Semantic Segmentation of Placenta**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.11580v1)  

---


**ABSTRACT**  
[Purpose] To develop a fully automated semantic placenta segmentation model that integrates the U-Net and SegNeXt architectures through ensemble learning. [Methods] A total of 218 pregnant women with suspected placental anomalies who underwent magnetic resonance imaging (MRI) were enrolled, yielding 1090 annotated images for developing a deep learning model for placental segmentation. The images were standardized and divided into training and test sets. The performance of PlaNet-S, which integrates U-Net and SegNeXt within an ensemble framework, was assessed using Intersection over Union (IoU) and counting connected components (CCC) against the U-Net model. [Results] PlaNet-S had significantly higher IoU (0.73 +/- 0.13) than that of U-Net (0.78 +/- 0.010) (p<0.01). The CCC for PlaNet-S was significantly higher than that for U-Net (p<0.01), matching the ground truth in 86.0\% and 56.7\% of the cases, respectively. [Conclusion]PlaNet-S performed better than the traditional U-Net in placental segmentation tasks. This model addresses the challenges of time-consuming physician-assisted manual segmentation and offers the potential for diverse applications in placental imaging analyses.

{{</citation>}}


## math.NA (1)



### (126/134) Structure-Preserving Transformers for Learning Parametrized Hamiltonian Systems (Benedikt Brantner et al., 2023)

{{<citation>}}

Benedikt Brantner, Guillaume de Romemont, Michael Kraus, Zeyuan Li. (2023)  
**Structure-Preserving Transformers for Learning Parametrized Hamiltonian Systems**  

---
Primary Category: math.NA  
Categories: 68T07, 65D30, 37M15, 65P10, cs-LG, cs-NA, math-NA, math.NA  
Keywords: LSTM, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.11166v1)  

---


**ABSTRACT**  
Two of the many trends in neural network research of the past few years have been (i) the learning of dynamical systems, especially with recurrent neural networks such as long short-term memory networks (LSTMs) and (ii) the introduction of transformer neural networks for natural language processing (NLP) tasks. Both of these trends have created enormous amounts of traction, particularly the second one: transformer networks now dominate the field of NLP. Even though some work has been performed on the intersection of these two trends, this work was largely limited to using the vanilla transformer directly without adjusting its architecture for the setting of a physical system. In this work we use a transformer-inspired neural network to learn a complicated non-linear dynamical system and furthermore (for the first time) imbue it with structure-preserving properties to improve long-term stability. This is shown to be extremely important when applying the neural network to real world applications.

{{</citation>}}


## q-bio.QM (1)



### (127/134) ContraNovo: A Contrastive Learning Approach to Enhance De Novo Peptide Sequencing (Zhi Jin et al., 2023)

{{<citation>}}

Zhi Jin, Sheng Xu, Xiang Zhang, Tianze Ling, Nanqing Dong, Wanli Ouyang, Zhiqiang Gao, Cheng Chang, Siqi Sun. (2023)  
**ContraNovo: A Contrastive Learning Approach to Enhance De Novo Peptide Sequencing**  

---
Primary Category: q-bio.QM  
Categories: cs-AI, cs-LG, q-bio-QM, q-bio.QM  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.11584v1)  

---


**ABSTRACT**  
De novo peptide sequencing from mass spectrometry (MS) data is a critical task in proteomics research. Traditional de novo algorithms have encountered a bottleneck in accuracy due to the inherent complexity of proteomics data. While deep learning-based methods have shown progress, they reduce the problem to a translation task, potentially overlooking critical nuances between spectra and peptides. In our research, we present ContraNovo, a pioneering algorithm that leverages contrastive learning to extract the relationship between spectra and peptides and incorporates the mass information into peptide decoding, aiming to address these intricacies more efficiently. Through rigorous evaluations on two benchmark datasets, ContraNovo consistently outshines contemporary state-of-the-art solutions, underscoring its promising potential in enhancing de novo peptide sequencing. The source code is available at https://github.com/BEAM-Labs/ContraNovo.

{{</citation>}}


## cs.SD (5)



### (128/134) Improved Long-Form Speech Recognition by Jointly Modeling the Primary and Non-primary Speakers (Guru Prakash Arumugam et al., 2023)

{{<citation>}}

Guru Prakash Arumugam, Shuo-yiin Chang, Tara N. Sainath, Rohit Prabhavalkar, Quan Wang, Shaan Bijwadia. (2023)  
**Improved Long-Form Speech Recognition by Jointly Modeling the Primary and Non-primary Speakers**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.11123v1)  

---


**ABSTRACT**  
ASR models often suffer from a long-form deletion problem where the model predicts sequential blanks instead of words when transcribing a lengthy audio (in the order of minutes or hours). From the perspective of a user or downstream system consuming the ASR results, this behavior can be perceived as the model "being stuck", and potentially make the product hard to use. One of the culprits for long-form deletion is training-test data mismatch, which can happen even when the model is trained on diverse and large-scale data collected from multiple application domains. In this work, we introduce a novel technique to simultaneously model different groups of speakers in the audio along with the standard transcript tokens. Speakers are grouped as primary and non-primary, which connects the application domains and significantly alleviates the long-form deletion problem. This improved model neither needs any additional training data nor incurs additional training or inference cost.

{{</citation>}}


### (129/134) 3S-TSE: Efficient Three-Stage Target Speaker Extraction for Real-Time and Low-Resource Applications (Shulin He et al., 2023)

{{<citation>}}

Shulin He, Jinjiang liu, Hao Li, Yang Yang, Fei Chen, Xueliang Zhang. (2023)  
**3S-TSE: Efficient Three-Stage Target Speaker Extraction for Real-Time and Low-Resource Applications**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Low-Resource  
[Paper Link](http://arxiv.org/abs/2312.10979v1)  

---


**ABSTRACT**  
Target speaker extraction (TSE) aims to isolate a specific voice from multiple mixed speakers relying on a registerd sample. Since voiceprint features usually vary greatly, current end-to-end neural networks require large model parameters which are computational intensive and impractical for real-time applications, espetially on resource-constrained platforms. In this paper, we address the TSE task using microphone array and introduce a novel three-stage solution that systematically decouples the process: First, a neural network is trained to estimate the direction of the target speaker. Second, with the direction determined, the Generalized Sidelobe Canceller (GSC) is used to extract the target speech. Third, an Inplace Convolutional Recurrent Neural Network (ICRN) acts as a denoising post-processor, refining the GSC output to yield the final separated speech. Our approach delivers superior performance while drastically reducing computational load, setting a new standard for efficient real-time target speaker extraction.

{{</citation>}}


### (130/134) Speaker Mask Transformer for Multi-talker Overlapped Speech Recognition (Peng Shen et al., 2023)

{{<citation>}}

Peng Shen, Xugang Lu, Hisashi Kawai. (2023)  
**Speaker Mask Transformer for Multi-talker Overlapped Speech Recognition**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition, Transformer  
[Paper Link](http://arxiv.org/abs/2312.10959v1)  

---


**ABSTRACT**  
Multi-talker overlapped speech recognition remains a significant challenge, requiring not only speech recognition but also speaker diarization tasks to be addressed. In this paper, to better address these tasks, we first introduce speaker labels into an autoregressive transformer-based speech recognition model to support multi-speaker overlapped speech recognition. Then, to improve speaker diarization, we propose a novel speaker mask branch to detection the speech segments of individual speakers. With the proposed model, we can perform both speech recognition and speaker diarization tasks simultaneously using a single model. Experimental results on the LibriSpeech-based overlapped dataset demonstrate the effectiveness of the proposed method in both speech recognition and speaker diarization tasks, particularly enhancing the accuracy of speaker diarization in relatively complex multi-talker scenarios.

{{</citation>}}


### (131/134) Leveraged Mel spectrograms using Harmonic and Percussive Components in Speech Emotion Recognition (David Hason Rudd et al., 2023)

{{<citation>}}

David Hason Rudd, Huan Huo, Guandong Xu. (2023)  
**Leveraged Mel spectrograms using Harmonic and Percussive Components in Speech Emotion Recognition**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-HC, cs-LG, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2312.10949v1)  

---


**ABSTRACT**  
Speech Emotion Recognition (SER) affective technology enables the intelligent embedded devices to interact with sensitivity. Similarly, call centre employees recognise customers' emotions from their pitch, energy, and tone of voice so as to modify their speech for a high-quality interaction with customers. This work explores, for the first time, the effects of the harmonic and percussive components of Mel spectrograms in SER. We attempt to leverage the Mel spectrogram by decomposing distinguishable acoustic features for exploitation in our proposed architecture, which includes a novel feature map generator algorithm, a CNN-based network feature extractor and a multi-layer perceptron (MLP) classifier. This study specifically focuses on effective data augmentation techniques for building an enriched hybrid-based feature map. This process results in a function that outputs a 2D image so that it can be used as input data for a pre-trained CNN-VGG16 feature extractor. Furthermore, we also investigate other acoustic features such as MFCCs, chromagram, spectral contrast, and the tonnetz to assess our proposed framework. A test accuracy of 92.79% on the Berlin EMO-DB database is achieved. Our result is higher than previous works using CNN-VGG16.

{{</citation>}}


### (132/134) An Extended Variational Mode Decomposition Algorithm Developed Speech Emotion Recognition Performance (David Hason Rudd et al., 2023)

{{<citation>}}

David Hason Rudd, Huan Huo, Guandong Xu. (2023)  
**An Extended Variational Mode Decomposition Algorithm Developed Speech Emotion Recognition Performance**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-HC, cs-LG, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2312.10937v1)  

---


**ABSTRACT**  
Emotion recognition (ER) from speech signals is a robust approach since it cannot be imitated like facial expression or text based sentiment analysis. Valuable information underlying the emotions are significant for human-computer interactions enabling intelligent machines to interact with sensitivity in the real world. Previous ER studies through speech signal processing have focused exclusively on associations between different signal mode decomposition methods and hidden informative features. However, improper decomposition parameter selections lead to informative signal component losses due to mode duplicating and mixing. In contrast, the current study proposes VGG-optiVMD, an empowered variational mode decomposition algorithm, to distinguish meaningful speech features and automatically select the number of decomposed modes and optimum balancing parameter for the data fidelity constraint by assessing their effects on the VGG16 flattening output layer. Various feature vectors were employed to train the VGG16 network on different databases and assess VGG-optiVMD reproducibility and reliability. One, two, and three-dimensional feature vectors were constructed by concatenating Mel-frequency cepstral coefficients, Chromagram, Mel spectrograms, Tonnetz diagrams, and spectral centroids. Results confirmed a synergistic relationship between the fine-tuning of the signal sample rate and decomposition parameters with classification accuracy, achieving state-of-the-art 96.09% accuracy in predicting seven emotions on the Berlin EMO-DB database.

{{</citation>}}


## cond-mat.mtrl-sci (1)



### (133/134) Position Paper on Materials Design -- A Modern Approach (Willi Grossmann et al., 2023)

{{<citation>}}

Willi Grossmann, Sebastian Eilermann, Tim Rensmeyer, Artur Liebert, Michael Hohmann, Christian Wittke, Oliver Niggemann. (2023)  
**Position Paper on Materials Design -- A Modern Approach**  

---
Primary Category: cond-mat.mtrl-sci  
Categories: cond-mat-mtrl-sci, cond-mat.mtrl-sci, cs-AI, cs-LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.10996v1)  

---


**ABSTRACT**  
Traditional design cycles for new materials and assemblies have two fundamental drawbacks. The underlying physical relationships are often too complex to be precisely calculated and described. Aside from that, many unknown uncertainties, such as exact manufacturing parameters or materials composition, dominate the real assembly behavior. Machine learning (ML) methods overcome these fundamental limitations through data-driven learning. In addition, modern approaches can specifically increase system knowledge. Representation Learning allows the physical, and if necessary, even symbolic interpretation of the learned solution. In this way, the most complex physical relationships can be considered and quickly described. Furthermore, generative ML approaches can synthesize possible morphologies of the materials based on defined conditions to visualize the effects of uncertainties. This modern approach accelerates the design process for new materials and enables the prediction and interpretation of realistic materials behavior.

{{</citation>}}


## cs.NE (1)



### (134/134) Delving Deeper Into Astromorphic Transformers (Md Zesun Ahmed Mia et al., 2023)

{{<citation>}}

Md Zesun Ahmed Mia, Malyaban Bal, Abhronil Sengupta. (2023)  
**Delving Deeper Into Astromorphic Transformers**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.10925v1)  

---


**ABSTRACT**  
Preliminary attempts at incorporating the critical role of astrocytes - cells that constitute more than 50% of human brain cells - in brain-inspired neuromorphic computing remain in infancy. This paper seeks to delve deeper into various key aspects of neuron-synapse-astrocyte interactions to mimic self-attention mechanisms in Transformers. The cross-layer perspective explored in this work involves bio-plausible modeling of Hebbian and pre-synaptic plasticities in neuron-astrocyte networks, incorporating effects of non-linearities and feedback along with algorithmic formulations to map the neuron-astrocyte computations to self-attention mechanism and evaluating the impact of incorporating bio-realistic effects from the machine learning application side. Our analysis on sentiment and image classification tasks on the IMDB and CIFAR10 datasets underscores the importance of constructing Astromorphic Transformers from both accuracy and learning speed improvement perspectives.

{{</citation>}}
